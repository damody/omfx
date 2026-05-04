//! Phase 3 omfx simulator runner.
//!
//! Spawns a worker thread that runs the full omb ECS dispatcher driven by
//! TickBatch input from omb's lockstep wire. Render thread reads from a
//! published `SimWorldSnapshot` Arc<Mutex<...>>.
//!
//! Phase 3.1 = stub. Phase 3.2 = real World init + dispatcher loop. Phase
//! 3.3 will wire `LockstepClient` → channel feeders. Phase 3.4 wires
//! the snapshot into the render side and replaces TickBroadcaster's
//! placeholder state hash with a real ECS hash sourced from this loop.

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::thread;

use crossbeam_channel::{unbounded, Receiver, Sender};
use log::{error, info};

use specs::{Join, World, WorldExt};

// Re-export the Phase 2 PlayerInput proto type from omobab so feeders
// (lockstep_client.rs in Phase 3.3) and the sim_runner share the same
// concrete type. omobab re-exports it from the prost-generated module
// under `lockstep::PlayerInput`.
pub use omobab::lockstep::PlayerInput;

/// Phase 4.2: render-only explosion FX entry, mirrored from
/// `omobab::comp::outcome::ExplosionFx`. Re-exported through
/// `SimWorldSnapshot.explosions` so the render side never needs to
/// touch omobab types directly. `spawn_tick` is the sim tick at which
/// the explosion was emitted; the render thread uses omfx wall clock
/// for the actual ring-aging lifecycle (see lib.rs `active_explosions`).
pub use omobab::comp::ExplosionFx;

const APPLIED_INPUT_ID_RETENTION_TICKS: u32 = 300;

/// Render-thread-readable snapshot of the latest sim tick state.
#[derive(Default, Clone, Debug)]
pub struct SimWorldSnapshot {
    pub tick: u32,
    pub entities: Vec<EntityRenderData>,
    /// Creep checkpoint paths (world coords, raw f32 — render side applies WORLD_SCALE).
    /// Each inner Vec is one named path's ordered list of `(x, y)` checkpoints.
    /// Static after `init_creep_wave`; re-emitted every snapshot so the render
    /// bridge sees them on its first read after GameStart without needing a
    /// dedicated init-only channel.
    pub paths: Vec<Vec<(f32, f32)>>,
    /// Entity ids that were alive in the previous snapshot but are no longer
    /// present in the current ECS world. Used by the render thread to free
    /// per-eid scene caches (labels, batch slots) without needing a wire-side
    /// `entity.death` event. Replaces the legacy omb-side `make_entity_death`
    /// emit; the snapshot diff is computed worker-locally each tick.
    pub removed_entity_ids: Vec<u32>,
    /// TD wave number — 1-based current wave index. 0 before the first
    /// `StartRound` flips `CurrentCreepWave.is_running`. Sourced from the
    /// `CurrentCreepWave` resource (`wave: usize`, cast to u32 at the boundary).
    pub round: u32,
    /// Total number of creep waves loaded for the active scene. Sourced
    /// from `Vec<CreepWave>` resource length. 0 in non-TD modes.
    pub total_rounds: u32,
    /// Current TD player lives (`PlayerLives.0`). 0 in non-TD modes (sentinel
    /// flag — HUD switches mode based on `lives > 0`).
    pub lives: i32,
    /// True while a TD round is running (creeps spawning / on the path);
    /// flips false once the wave is cleared. Mirrors
    /// `CurrentCreepWave.is_running`.
    pub round_is_running: bool,
    /// Phase 4.1: BlockedRegion polygons — non-walkable map regions sourced
    /// from `BlockedRegions(Vec<BlockedRegion>)`. Static map data after
    /// `state::initialization` loads them; cheap to clone each tick (TD_1 is
    /// empty; MVP_1/DEBUG_1 have a handful). Render side draws a red
    /// polygon outline per region; `circle` is currently always `None` since
    /// omb `BlockedRegion` has no radius field, but the field is plumbed
    /// for forward-compat (eg. circular blockers).
    pub blocked_regions: Vec<BlockedRegionSnapshot>,
    /// Phase 4.5: AbilityRegistry — static-ish ability metadata loaded from
    /// the script DLL at game start. `Arc` wrapped so each tick clone is
    /// O(1); rebuilt lazily once the registry is non-empty (script load is
    /// async). Hero panel uses this to resolve `ability_ids[i]` →
    /// display_name / icon / max_level.
    pub abilities: std::sync::Arc<Vec<AbilityDefSnapshot>>,
    /// TD tower templates (right-side build-button menu). Sourced from
    /// `TowerTemplateRegistry` which the script DLL fills via each tower's
    /// `tower_metadata()` at game start. `Arc` wrapped — same lazy-build
    /// pattern as abilities (registry is async-populated). lib.rs reads
    /// once on first non-empty snapshot to seed `td_template_order` +
    /// `td_templates` HashMap; subsequent ticks are O(1) Arc clone.
    pub tower_templates: std::sync::Arc<Vec<TowerTemplateSnapshot>>,
    /// 48 個 tower upgrade defs (4 towers × 3 paths × 4 levels). Sourced
    /// from `omobab::comp::tower_upgrade_registry::TowerUpgradeRegistry`.
    /// Used by the omfx Sell/Upgrade panel to (a) compute refund =
    /// base*0.85 + Σ(upgrades*0.75) and (b) show each upgrade's name in
    /// the upgrade button text. Same lazy-build Arc pattern as
    /// `tower_templates`.
    pub tower_upgrades: std::sync::Arc<Vec<TowerUpgradeDefSnapshot>>,
    /// Phase 4.2: explosion FX events emitted this tick — one entry per
    /// `Outcome::Explosion` processed by `process_outcomes`. Drained from
    /// the sim's `ExplosionFxQueue` resource each tick (`std::mem::take`)
    /// so the queue stays bounded. The render thread spawns a transient
    /// expanding red ring per entry against omfx wall clock; sim never
    /// reads this back, so it's not part of the determinism state.
    pub explosions: Vec<ExplosionFx>,
    /// omfx-only metadata for input-to-render latency pairing; sim ECS does not read it.
    pub applied_input_ids: Vec<u32>,
}

/// Phase 4.1: One polygon region snapshot.
#[derive(Clone, Debug, Default)]
pub struct BlockedRegionSnapshot {
    /// Polygon vertices in world coords (raw f32 — render side applies
    /// WORLD_SCALE + `-x` flip just like the path-segment renderer).
    pub points: Vec<(f32, f32)>,
    /// Optional center + radius for an orange circular blocker. omb
    /// `BlockedRegion` has no radius today, so this is always `None`;
    /// kept for forward-compat with future circular regions.
    pub circle: Option<((f32, f32), f32)>,
}

/// Phase 4.5: AbilityDef projection — only the fields the omfx hero
/// panel needs. Stays lean (no `levels` HashMap or `properties` JSON
/// blob) since the ability bar shows ability_id / max_level / icon.
#[derive(Clone, Debug)]
pub struct AbilityDefSnapshot {
    pub ability_id: String,
    pub display_name: String,
    pub max_level: u8,
    pub icon_path: String,
}

/// TowerUpgradeDef projection — only the fields the Sell/Upgrade panel
/// needs (name for the button label, cost for refund accounting).
#[derive(Clone, Debug)]
pub struct TowerUpgradeDefSnapshot {
    pub tower_kind: String,
    pub path: u8,
    pub level: u8,
    pub name: String,
    pub cost: i32,
}

/// TowerTemplate projection for the right-side TD build menu. Mirrors the
/// fields lib.rs's `TdTemplate` cache needs (label / cost for the button
/// + footprint / range for the placement preview). Sourced from
/// `omobab::comp::tower_registry::TowerTemplateRegistry`.
#[derive(Clone, Debug)]
pub struct TowerTemplateSnapshot {
    pub unit_id: String,
    pub label: String,
    pub cost: i32,
    pub footprint: f32,
    pub range: f32,
    pub splash_radius: f32,
    pub hit_radius: f32,
    pub slow_factor: f32,
    pub slow_duration: f32,
}

/// Per-entity render data extracted from the ECS World at the end of
/// every tick. Already mapped to `f32` at the boundary (Fixed64 →
/// `to_f32_for_render`); render thread does not need to know about
/// the deterministic sim's fixed-point types.
#[derive(Clone, Debug, Default)]
pub struct EntityRenderData {
    pub entity_id: u32,
    pub entity_gen: u32,
    pub kind: EntityKind,
    pub pos_x: f32,
    pub pos_y: f32,
    pub facing_rad: f32,
    pub hp: i32,
    pub max_hp: i32,
    /// `ScriptUnitTag.unit_id` if present (e.g. "tower_dart_monkey",
    /// "creep_balloon_red", "hero_saika_magoichi"). Empty for entities
    /// without a script tag (rare — most spawned units have one).
    pub unit_id: String,
    /// Hero-only metadata. Empty / 0 when entity is not a Hero.
    pub hero_name: String,
    pub hero_title: String,
    pub hero_level: i32,
    pub hero_xp: i32,
    pub hero_xp_next: i32,
    pub hero_skill_points: i32,
    pub hero_primary_attribute: String,
    pub hero_strength: i32,
    pub hero_agility: i32,
    pub hero_intelligence: i32,
    /// Gold (player resource for hero). 0 for non-hero entities.
    pub gold: i32,
    /// Phase 3.3: Aggregated hero stats (final values after BuffStore /
    /// UnitStats aggregation). `None` for non-hero entities — keeps
    /// EntityRenderData small for the 1000-tower / 500-creep stress
    /// path. Boxed so the Hero arm pays a heap alloc but Tower/Creep
    /// rows pay only a single None-pointer.
    pub hero_ext: Option<Box<HeroStatsExt>>,
    /// Phase 4.3: Tower upgrade level pips per path (3 paths × 0-4 levels).
    /// `None` for non-Tower entities. Sourced from `Tower.upgrade_levels`
    /// component field; the existing TD sell/upgrade panel already reads
    /// this off `network_entities`, so the snapshot variant is the
    /// lockstep-side mirror.
    pub upgrade_levels: Option<[u8; 3]>,
}

/// Phase 3.3: Single-buff snapshot for the hero panel.
///
/// Mirrors the legacy `hero.stats` `buffs` array. `remaining_secs`
/// uses `-1.0` as a sentinel for "infinite / toggle" (e.g. base_stats
/// or sniper_mode) — render-side displays it as ∞. Otherwise the
/// render thread decrements `remaining_secs` per frame locally; next
/// authoritative snapshot resets the value, avoiding drift.
#[derive(Clone, Debug, Default)]
pub struct BuffSnapshot {
    pub buff_id: String,
    pub remaining_secs: f32,
    /// Stringified payload JSON. Worst-case fallback to `Debug` repr
    /// when the canonical JSON encoding is unavailable; the panel
    /// listed numeric payload fields with `as_f64()`, so we keep the
    /// JSON round-trip to preserve that.
    pub payload_json: String,
}

/// Phase 3.3: Aggregated hero stats — the omfx-side mirror of the
/// legacy omb `hero.stats` JSON payload. Computed via the same
/// `BuffStore` / `UnitStats` aggregation pipeline omb used (see
/// `omobab::ability_runtime::UnitStats`); read-only against the ECS
/// so lockstep determinism is unaffected.
#[derive(Clone, Debug, Default)]
pub struct HeroStatsExt {
    pub armor: f32,
    pub magic_resist: f32,
    pub attack_damage: f32,
    pub attack_range: f32,
    pub move_speed: f32,
    /// Seconds per attack (asd) — 0 for non-attacking units.
    pub attack_speed_sec: f32,
    pub bullet_speed: f32,
    /// Hero mana / max-mana (Phase 3.3: omb `CProperty` does not yet
    /// have hero mana fields, so these are 0; legacy `hero.stats`
    /// payload also wired 0). Plumbed for forward-compat.
    pub mana: f32,
    pub max_mana: f32,
    pub buffs: Vec<BuffSnapshot>,
    /// Phase 4.4: Inventory item ids per slot. omb `Inventory` has 6
    /// slots (`INVENTORY_SLOTS = 6`); each slot holds an
    /// `Option<ItemInstance>` whose `item_id` is a `String`. `None` for
    /// empty slots. Cooldown intentionally omitted here — the legacy
    /// `hero_state.inventory` HUD already drives a local CD ticker
    /// (`Vec<Option<(String, f32)>>`), and Phase 2.4 ItemUse start_cd
    /// happens host-side; the next snapshot reset is fine.
    pub inventory: [Option<String>; 6],
    /// Phase 4.5: Ability levels per slot (Q/W/E/R = indices 0..3).
    /// Sourced from `Hero.ability_levels: HashMap<String, i32>` keyed
    /// by `ability_ids[i]`. 0 if unlearned / no ability in slot.
    pub ability_levels: [i32; 4],
    /// Phase 4.5: Ability ids per slot. Mirrors `Hero.abilities[i]`
    /// (Q/W/E/R order). `None` if the hero has fewer than 4 abilities
    /// or that slot is unset. Render side looks these up in
    /// `SimWorldSnapshot.abilities` to resolve display name / icon.
    pub ability_ids: [Option<String>; 4],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum EntityKind {
    Hero,
    Tower,
    Creep,
    Projectile,
    #[default]
    Other,
}

/// Channel payload submitted by the lockstep feeder per tick.
#[derive(Clone, Debug)]
pub struct TickBatchPayload {
    pub tick: u32,
    pub inputs: Vec<(u32 /* player_id */, PlayerInput, u32 /* input_id */)>,
}

/// Handle returned to omfx Game so the render thread can read snapshots
/// and the lockstep feeder can push tick inputs.
#[derive(Debug)]
pub struct SimRunnerHandle {
    /// Latest published snapshot. Render thread `lock()`s once per
    /// frame, copies / borrows, and releases.
    pub state: Arc<Mutex<SimWorldSnapshot>>,
    /// Send (tick, inputs) per TickBatch arrival. Phase 3.3 wires this.
    pub tick_input_tx: Sender<TickBatchPayload>,
    /// Send `master_seed` exactly once after `GameStart` arrives. The
    /// worker blocks on this before initializing the World so the
    /// MasterSeed resource is set before the first tick runs.
    pub master_seed_tx: Sender<u64>,
    /// Worker thread join handle. Held but not joined; thread exits on
    /// channel disconnect when `SimRunnerHandle` is dropped.
    _thread: thread::JoinHandle<()>,
}

/// Spawn the simulator worker. Initializes a specs World using
/// `omobab::state::initialization::create_world_for_scene` and runs the
/// shared Phase 3 dispatcher per tick driven by inputs from
/// `tick_input_rx`.
pub fn spawn_sim_runner(
    base_content_dll_path: PathBuf,
    scene_path: PathBuf,
) -> SimRunnerHandle {
    let state = Arc::new(Mutex::new(SimWorldSnapshot::default()));
    let state_for_thread = state.clone();

    let (tick_input_tx, tick_input_rx) = unbounded::<TickBatchPayload>();
    let (master_seed_tx, master_seed_rx) = unbounded::<u64>();

    let handle = thread::Builder::new()
        .name("omfx-sim-runner".into())
        .spawn(move || {
            run_sim_loop(
                state_for_thread,
                tick_input_rx,
                master_seed_rx,
                base_content_dll_path,
                scene_path,
            );
        })
        .expect("spawn omfx-sim-runner thread");

    SimRunnerHandle {
        state,
        tick_input_tx,
        master_seed_tx,
        _thread: handle,
    }
}

fn run_sim_loop(
    state_out: Arc<Mutex<SimWorldSnapshot>>,
    tick_input_rx: Receiver<TickBatchPayload>,
    master_seed_rx: Receiver<u64>,
    dll_path: PathBuf,
    scene_path: PathBuf,
) {
    info!(
        "sim_runner: thread started; waiting for master_seed (dll={:?}, scene={:?})",
        dll_path, scene_path
    );

    // Block on first master_seed (delivered by LockstepClient on
    // GameStart in Phase 3.3). Returning early — without ever ticking —
    // is the expected Phase 3.2 outcome, since LockstepClient does not
    // yet feed this channel.
    let master_seed = match master_seed_rx.recv() {
        Ok(s) => s,
        Err(_) => {
            info!("sim_runner: master_seed channel dropped before GameStart, exiting");
            return;
        }
    };
    info!("sim_runner: got master_seed=0x{:016x}", master_seed);

    // Point omb's script loader at the directory containing the DLL.
    // `load_scripts_dir` reads `OMB_SCRIPTS_DIR` env var; honor caller
    // override but otherwise infer from the DLL path's parent.
    if std::env::var_os("OMB_SCRIPTS_DIR").is_none() {
        if let Some(parent) = dll_path.parent() {
            if let Some(parent_str) = parent.to_str() {
                std::env::set_var("OMB_SCRIPTS_DIR", parent_str);
                info!("sim_runner: set OMB_SCRIPTS_DIR={}", parent_str);
            }
        }
    }

    let mut world = match init_world(&scene_path, master_seed) {
        Ok(w) => w,
        Err(e) => {
            error!("sim_runner: init_world failed: {}", e);
            return;
        }
    };

    let mut dispatcher = match omobab::state::system_dispatcher::build_phase3_dispatcher() {
        Ok(d) => d,
        Err(e) => {
            error!("sim_runner: build_phase3_dispatcher failed: {}", e);
            return;
        }
    };

    // Move ScriptRegistry out of the ECS resource so we can hold an &-borrow
    // across `run_script_dispatch(&mut world, ...)` calls each tick. The omb
    // host does the same — its `State` keeps `script_registry` as a struct
    // field, not in ECS, exactly to avoid the borrow conflict. Replacing the
    // resource with `Default::default()` (empty registry) is fine because
    // nothing else queries the ECS-resident ScriptRegistry.
    let script_registry: omobab::scripting::ScriptRegistry = std::mem::take(
        &mut *world.write_resource::<omobab::scripting::ScriptRegistry>(),
    );

    info!("sim_runner: dispatcher ready, entering tick loop");

    let mut last_starvation_log = std::time::Instant::now();
    // Phase 1b: removed_entity_ids 從 RemovedEntitiesQueue resource drain
    // 取代既有 prev_alive HashSet diff。helper `delete_entity_tracked` 統
    // 一往 queue 推入；`extract_snapshot` 用 `mem::take` 把整批拉到
    // snapshot，render 端對該 list 釋放 per-eid scene caches。

    // Phase 4.5: AbilityRegistry → AbilityDefSnapshot Arc. Built lazily on
    // the first tick where the registry is non-empty (script DLL load is
    // async — the registry is populated by `scripting::registry::load`
    // during world init, but we re-poll each tick until the Arc is set
    // because in some scenes the registry may stay empty until a hero's
    // script registers abilities). After build, every snapshot just clones
    // the Arc (O(1) refcount bump).
    let mut abilities_arc: std::sync::Arc<Vec<AbilityDefSnapshot>> =
        std::sync::Arc::new(Vec::new());
    // Same lazy-build pattern for TD tower templates — registry populated
    // at game start by each tower script's `tower_metadata()`.
    let mut tower_templates_arc: std::sync::Arc<Vec<TowerTemplateSnapshot>> =
        std::sync::Arc::new(Vec::new());
    let mut tower_upgrades_arc: std::sync::Arc<Vec<TowerUpgradeDefSnapshot>> =
        std::sync::Arc::new(Vec::new());
    let mut recent_applied_input_ids: VecDeque<(u32, u32)> = VecDeque::new();
    loop {
        // Use recv_timeout instead of recv() so a wire stall surfaces in the
        // log as "no TickBatch in 1.0s — upstream lockstep client is the
        // suspect" instead of looking like sim_runner is computing slowly.
        let batch = match tick_input_rx.recv_timeout(std::time::Duration::from_secs(1)) {
            Ok(b) => b,
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                let now = std::time::Instant::now();
                if now.duration_since(last_starvation_log).as_secs() >= 2 {
                    let pending = tick_input_rx.len();
                    info!(
                        "sim_runner: no TickBatch in 1.0s (queue_len={}). \
                         Upstream Game→lockstep_client→KCP path is the suspect.",
                        pending,
                    );
                    last_starvation_log = now;
                }
                continue;
            }
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                info!("sim_runner: input channel closed, exiting");
                break;
            }
        };

        for input_id in batch
            .inputs
            .iter()
            .filter_map(|(_, _, input_id)| (*input_id != 0).then_some(*input_id))
        {
            recent_applied_input_ids.push_back((batch.tick, input_id));
        }
        while recent_applied_input_ids
            .front()
            .is_some_and(|(tick, _)| batch.tick.saturating_sub(*tick) > APPLIED_INPUT_ID_RETENTION_TICKS)
        {
            recent_applied_input_ids.pop_front();
        }
        let applied_input_ids = recent_applied_input_ids
            .iter()
            .map(|(_, input_id)| *input_id)
            .collect::<Vec<_>>();
        push_inputs_into_world(&mut world, batch.tick, batch.inputs);

        // Update Tick + Time + DeltaTime so time-gated systems (creep_wave,
        // buff timers, projectile flight) actually advance. Lockstep is 60Hz
        // (TickBroadcaster's tick_period_us = 16_667), so dt = 1/60.
        // Without these the local sim has Tick advancing but Time stuck at 0,
        // which makes `creep_wave` see `totaltime=0` and never spawn — exactly
        // why Start Round fires (is_running flips) but no creeps appear.
        const SIM_DT_S: f32 = 1.0 / 60.0;
        world.write_resource::<omobab::comp::resources::Tick>().0 = batch.tick as u64;
        {
            let mut t = world.write_resource::<omobab::comp::resources::Time>();
            t.0 = (batch.tick as f64) * (SIM_DT_S as f64);
        }
        {
            let mut dt = world.write_resource::<omobab::comp::resources::DeltaTime>();
            dt.0 = omoba_sim::Fixed64::from_raw((SIM_DT_S * 1024.0) as i64);
        }

        dispatcher.dispatch(&world);
        world.maintain();

        // Phase 2.1: drain `PendingTowerSpawnQueue` filled by
        // `player_input_tick::Sys` during the dispatch above. Mirrors the same
        // call in omb's `state::core::tick` so host + replica spawn TD towers
        // deterministically from `PlayerInputEnum::TowerPlace` inputs.
        omobab::comp::GameProcessor::drain_pending_tower_spawns(&mut world);
        world.maintain();

        // Phase 2.2: drain `PendingTowerSellQueue` from TowerSell inputs.
        // Mirrors omb's `state::core::tick`. Refund + entity delete done in
        // sync on host and replica so snapshots stay consistent.
        omobab::comp::GameProcessor::drain_pending_tower_sells(&mut world);
        world.maintain();

        // Phase 2.3: drain `PendingTowerUpgradeQueue` from TowerUpgrade
        // inputs. Mirrors omb's `state::core::tick`. Gold deduction +
        // upgrade_levels increment + BuffStore stat-mod adds need to run on
        // host and replica in sync so the snapshot stays consistent.
        omobab::comp::GameProcessor::drain_pending_tower_upgrades(&mut world);
        world.maintain();

        // Phase 2.4: drain `PendingItemUseQueue` from ItemUse inputs.
        // Mirrors omb's `state::core::tick`. Inventory cooldown + CProperty
        // (HP / msd) mutations need to run on host and replica in sync.
        omobab::comp::GameProcessor::drain_pending_item_uses(&mut world);
        world.maintain();

        // MoveTo (右鍵移動): drain `PendingMoveQueue` — writes MoveTarget on
        // player hero. Mirrors omb's `state::core::tick`.
        omobab::comp::GameProcessor::drain_pending_moves(&mut world);
        world.maintain();

        // Phase 3 dispatcher only schedules tick systems; it does NOT include
        // GameProcessor::process_outcomes. Without this, `creep_wave` produces
        // `Outcome::Creep { cd }` rows that pile up in `Vec<Outcome>` but no
        // entity is ever spawned in the local sim → snapshot.creep stays 0.
        // mqtx is a sink (empty Vec): outcome handlers `try_send` and silently
        // drop messages, which matches the deterministic-sim contract (host
        // owns wire emits; replica is render-only).
        let (sink_tx, _sink_rx) = crossbeam_channel::unbounded::<omobab::transport::OutboundMsg>();
        if let Err(e) = omobab::comp::GameProcessor::process_outcomes(&mut world, &sink_tx) {
            log::warn!("sim_runner: process_outcomes failed: {}", e);
        }
        world.maintain();

        // Run script dispatch so tower / hero / summon `on_tick` hooks fire.
        // Towers are ScriptUnitTag-driven — without this, tower_dart / tower_
        // bomb / tower_ice never decide to attack, so projectile_tick has
        // nothing to advance and damage_tick has nothing to apply.
        // omb's `State::tick` does the same after `run_systems` (see
        // `omb/src/state/core.rs` around the `scripting::run_script_dispatch`
        // call). Replica needs the same call to stay sim-equivalent.
        omobab::scripting::run_script_dispatch(
            &mut world,
            &script_registry,
            batch.tick as u64,
            omoba_sim::Fixed64::from_raw((SIM_DT_S * 1024.0) as i64),
            sink_tx.clone(),
        );
        // Process any outcomes scripts pushed (Projectile / Damage / etc.).
        if let Err(e) = omobab::comp::GameProcessor::process_outcomes(&mut world, &sink_tx) {
            log::warn!("sim_runner: process_outcomes (post-script) failed: {}", e);
        }
        world.maintain();

        // Phase 4.5: rebuild abilities Arc lazily if it's still empty and
        // the registry has populated. After the first non-empty build the
        // Arc never changes (registry is immutable post-load).
        if abilities_arc.is_empty() {
            let reg = world.read_resource::<omobab::ability_runtime::AbilityRegistry>();
            if !reg.is_empty() {
                abilities_arc = std::sync::Arc::new(
                    reg.all()
                        .map(|d| AbilityDefSnapshot {
                            ability_id: d.id.clone(),
                            display_name: d.name.clone(),
                            max_level: d.max_level,
                            icon_path: d.icon.clone().unwrap_or_default(),
                        })
                        .collect(),
                );
                log::info!(
                    "sim_runner: built AbilityRegistry snapshot ({} defs)",
                    abilities_arc.len()
                );
            }
        }

        // TD tower-template registry — same lazy-build pattern. Populated by
        // each tower script's `tower_metadata()` at script load time.
        if tower_templates_arc.is_empty() {
            let reg = world.read_resource::<omobab::comp::tower_registry::TowerTemplateRegistry>();
            if !reg.is_empty() {
                tower_templates_arc = std::sync::Arc::new(
                    reg.iter_ordered()
                        .map(|t| TowerTemplateSnapshot {
                            unit_id: t.unit_id.clone(),
                            label: t.label.clone(),
                            cost: t.cost,
                            footprint: t.footprint,
                            range: t.range,
                            splash_radius: t.splash_radius,
                            hit_radius: t.hit_radius,
                            slow_factor: t.slow_factor,
                            slow_duration: t.slow_duration,
                        })
                        .collect(),
                );
                log::info!(
                    "sim_runner: built TowerTemplateRegistry snapshot ({} templates)",
                    tower_templates_arc.len()
                );
            }
        }

        // TowerUpgradeRegistry — built once at world init (not async like
        // tower templates), so iter_all is non-empty from tick 1. Lazy guard
        // mirrors the other registries for symmetry.
        if tower_upgrades_arc.is_empty() {
            let reg = world.read_resource::<omobab::comp::tower_upgrade_registry::TowerUpgradeRegistry>();
            let mut defs: Vec<TowerUpgradeDefSnapshot> = reg.iter_all()
                .map(|d| TowerUpgradeDefSnapshot {
                    tower_kind: d.tower_kind.clone(),
                    path: d.path,
                    level: d.level,
                    name: d.name.clone(),
                    cost: d.cost,
                })
                .collect();
            if !defs.is_empty() {
                defs.sort_by(|a, b| {
                    a.tower_kind.cmp(&b.tower_kind)
                        .then(a.path.cmp(&b.path))
                        .then(a.level.cmp(&b.level))
                });
                tower_upgrades_arc = std::sync::Arc::new(defs);
                log::info!(
                    "sim_runner: built TowerUpgradeRegistry snapshot ({} defs)",
                    tower_upgrades_arc.len()
                );
            }
        }

        let snapshot = extract_snapshot(
            &mut world,
            batch.tick,
            abilities_arc.clone(),
            tower_templates_arc.clone(),
            tower_upgrades_arc.clone(),
            applied_input_ids,
        );

        // Diagnostic for the "creep HP bars stay full" regression report
        // (Phase 4-5 lockstep cleanup). Every 60 ticks (~1s) sample the
        // first few creeps' HP values. If HP never changes across the
        // run, the mirror's damage path is broken; if HP decreases, the
        // mirror is fine and the regression is render-only. Sampled every
        // 60 ticks to keep log volume low at TD_STRESS scale.
        if batch.tick % 60 == 0 {
            let creep_hps: Vec<(u32, i32, i32)> = snapshot
                .entities
                .iter()
                .filter(|e| matches!(e.kind, EntityKind::Creep))
                .take(5)
                .map(|e| (e.entity_id, e.hp, e.max_hp))
                .collect();
            if !creep_hps.is_empty() {
                log::info!(
                    "[mirror-snapshot] tick={} creep_count={} sample_hp={:?}",
                    batch.tick,
                    snapshot.entities.iter()
                        .filter(|e| matches!(e.kind, EntityKind::Creep))
                        .count(),
                    creep_hps,
                );
            }
        }

        if let Ok(mut s) = state_out.lock() {
            *s = snapshot;
        }
    }
}

fn init_world(scene_path: &Path, master_seed: u64) -> Result<World, failure::Error> {
    let mut world = omobab::state::initialization::create_world_for_scene(scene_path)?;
    // Override the default MasterSeed with the authoritative one from
    // GameStart. Must happen before the first dispatch.
    world.write_resource::<omobab::comp::resources::MasterSeed>().0 = master_seed;
    Ok(world)
}

fn push_inputs_into_world(world: &mut World, tick: u32, inputs: Vec<(u32, PlayerInput, u32)>) {
    // Phase 3.4: write the lockstep TickBatch inputs into the host's
    // `PendingPlayerInputs` resource so omb's `tick::player_input_tick::Sys`
    // can drain them at the start of the dispatcher run.
    //
    // Replaces the resource map wholesale (lockstep contract: at most one
    // input per player per tick — the latest TickBatch is authoritative).
    use omobab::comp::PendingPlayerInputs;

    let mut pending = world.write_resource::<PendingPlayerInputs>();
    pending.tick = tick;
    pending.by_player.clear();
    if !inputs.is_empty() {
        log::trace!("sim_runner: tick {} got {} inputs", tick, inputs.len());
    }
    for (player_id, input, _input_id) in inputs {
        pending.by_player.insert(player_id, input);
    }
}

fn extract_snapshot(
    world: &mut World,
    tick: u32,
    abilities_arc: std::sync::Arc<Vec<AbilityDefSnapshot>>,
    tower_templates_arc: std::sync::Arc<Vec<TowerTemplateSnapshot>>,
    tower_upgrades_arc: std::sync::Arc<Vec<TowerUpgradeDefSnapshot>>,
    applied_input_ids: Vec<u32>,
) -> SimWorldSnapshot {
    // omobab re-exports these via `pub use crate::comp::*;` at the
    // crate root, so go through the flat path instead of the
    // module-by-module one (some submodules like `comp::state` collide
    // with the State struct namespace).
    use omobab::{CProperty, Creep, Facing, Hero, Pos, Projectile, TAttack, Tower};
    use omobab::comp::hero::AttributeType;
    use omobab::comp::gold::Gold;
    use omobab::comp::inventory::Inventory;
    use omobab::scripting::ScriptUnitTag;
    use omobab::ability_runtime::{BuffStore, UnitStats};

    let entities = world.entities();
    let pos_storage = world.read_storage::<Pos>();
    let facing_storage = world.read_storage::<Facing>();
    let cprop_storage = world.read_storage::<CProperty>();
    let hero_storage = world.read_storage::<Hero>();
    let tower_storage = world.read_storage::<Tower>();
    let proj_storage = world.read_storage::<Projectile>();
    let creep_storage = world.read_storage::<Creep>();
    let unit_tag_storage = world.read_storage::<ScriptUnitTag>();
    let gold_storage = world.read_storage::<Gold>();
    // Phase 3.3: TAttack + BuffStore for the hero stats aggregation
    // path. BuffStore is a `World` resource; UnitStats borrows it
    // read-only — no ECS mutation, so determinism is unaffected.
    let tatk_storage = world.read_storage::<TAttack>();
    let buff_store = world.read_resource::<BuffStore>();
    let stats = UnitStats::from_refs(&*buff_store, /*is_building*/ false);
    // Phase 4.4: hero inventory storage — only Hero entities populate this,
    // so the lookup is cheap for non-hero rows (None).
    let inventory_storage = world.read_storage::<Inventory>();

    let mut out = Vec::new();
    for (entity, pos) in (&entities, &pos_storage).join() {
        let kind = if hero_storage.get(entity).is_some() {
            EntityKind::Hero
        } else if tower_storage.get(entity).is_some() {
            EntityKind::Tower
        } else if proj_storage.get(entity).is_some() {
            EntityKind::Projectile
        } else if creep_storage.get(entity).is_some() {
            EntityKind::Creep
        } else {
            EntityKind::Other
        };

        // Convert Angle ticks to f32 radians for render. TAU_TICKS = 4096
        // → divide by TAU_TICKS, multiply by 2π. Done at the boundary so
        // render code never needs to know about the trig-tick encoding.
        let facing = facing_storage
            .get(entity)
            .map(|f| {
                (f.0.ticks() as f32) / (omoba_sim::trig::TAU_TICKS as f32)
                    * 2.0
                    * std::f32::consts::PI
            })
            .unwrap_or(0.0);

        let (hp, max_hp) = cprop_storage
            .get(entity)
            .map(|c| {
                (
                    c.hp.to_f32_for_render() as i32,
                    c.mhp.to_f32_for_render() as i32,
                )
            })
            .unwrap_or((0, 0));

        let (px, py) = pos.xy_f32();

        let unit_id = unit_tag_storage
            .get(entity)
            .map(|t| t.unit_id.clone())
            .unwrap_or_default();
        let gold = gold_storage.get(entity).map(|g| g.0).unwrap_or(0);

        // Hero-only metadata. Read once per Hero entity, keep zero-cost
        // for non-Hero rows.
        let (
            hero_name,
            hero_title,
            hero_level,
            hero_xp,
            hero_xp_next,
            hero_skill_points,
            hero_primary_attribute,
            hero_strength,
            hero_agility,
            hero_intelligence,
        ) = if let Some(h) = hero_storage.get(entity) {
            let attr = match h.primary_attribute {
                AttributeType::Strength => "力量",
                AttributeType::Agility => "敏捷",
                AttributeType::Intelligence => "智力",
            };
            (
                h.name.clone(),
                h.title.clone(),
                h.level,
                h.experience,
                h.experience_to_next,
                h.skill_points,
                attr.to_string(),
                h.strength,
                h.agility,
                h.intelligence,
            )
        } else {
            (
                String::new(),
                String::new(),
                0,
                0,
                0,
                0,
                String::new(),
                0,
                0,
                0,
            )
        };

        // Phase 3.3: aggregate final hero stats (armor / atk / range /
        // move_speed / buffs) the same way omb's
        // `state::resource_management::build_hero_stats_payload` did.
        // Read-only — `UnitStats::final_*` and `BuffStore::iter_for`
        // never mutate the ECS, so lockstep determinism is unaffected.
        // `None` for non-Hero entities so Tower/Creep rows pay only a
        // single null-pointer worth of size.
        let hero_ext = if matches!(kind, EntityKind::Hero) {
            let prop = cprop_storage.get(entity);
            let atk = tatk_storage.get(entity);

            let armor = prop
                .map(|p| stats.final_armor(p.def_physic, entity).to_f32_for_render())
                .unwrap_or(0.0);
            let magic_resist = prop
                .map(|p| stats.final_magic_resist(p.def_magic, entity).to_f32_for_render())
                .unwrap_or(0.0);
            let move_speed = prop
                .map(|p| stats.final_move_speed(p.msd, entity).to_f32_for_render())
                .unwrap_or(0.0);
            let attack_damage = atk
                .map(|a| stats.final_atk(a.atk_physic.v, entity).to_f32_for_render())
                .unwrap_or(0.0);
            let attack_range = atk
                .map(|a| stats.final_attack_range(a.range.v, entity).to_f32_for_render())
                .unwrap_or(0.0);
            // attack_speed_sec = base interval / asd_mult. asd_mult = 1
            // means base; > 1 means faster (lower interval). Mirrors
            // the divide done in `build_hero_stats_payload`.
            let attack_speed_sec = atk
                .map(|a| {
                    let asd_mult =
                        stats.final_attack_speed_mult(entity).to_f32_for_render();
                    let base = a.asd.v.to_f32_for_render();
                    if asd_mult > 0.0 { base / asd_mult } else { base }
                })
                .unwrap_or(0.0);
            let bullet_speed = atk
                .map(|a| a.bullet_speed.to_f32_for_render())
                .unwrap_or(0.0);
            // CProperty has no hero mana fields yet; legacy
            // `build_hero_stats_payload` wired 0 too. Plumbed for
            // forward-compat once mana lands.
            let mana = 0.0_f32;
            let max_mana = 0.0_f32;

            let buffs: Vec<BuffSnapshot> = buff_store
                .iter_for(entity)
                .map(|(id, entry)| {
                    // BuffEntry.remaining is Fixed64 seconds; legacy
                    // wire convention: raw == i32::MAX is the
                    // "infinite / toggle" sentinel (sniper_mode,
                    // base_stats). Map to -1.0 so the panel renders ∞.
                    let remaining_secs = if entry.remaining.raw() == i32::MAX as i64 {
                        -1.0
                    } else {
                        entry.remaining.to_f32_for_render()
                    };
                    BuffSnapshot {
                        buff_id: id.to_string(),
                        remaining_secs,
                        payload_json: serde_json::to_string(&entry.payload)
                            .unwrap_or_default(),
                    }
                })
                .collect();

            // Phase 4.4: inventory slots — `Inventory.slots` is
            // `[Option<ItemInstance>; 6]`; we project to
            // `[Option<String>; 6]` (item_id only). Empty slot → None.
            // Hero may not have an Inventory component (unit tests / pre-
            // pickup); in that case all slots are None.
            let mut inventory: [Option<String>; 6] = Default::default();
            if let Some(inv) = inventory_storage.get(entity) {
                for (i, slot) in inv.slots.iter().enumerate().take(6) {
                    inventory[i] = slot.as_ref().map(|it| it.item_id.clone());
                }
            }

            // Phase 4.5: ability ids + levels per slot (Q/W/E/R = 0..3).
            // `Hero.abilities` is a `Vec<String>` (typically length 4 but
            // we guard against shorter); `ability_levels` is a HashMap
            // keyed by ability id. Missing → 0 / None.
            let mut ability_ids: [Option<String>; 4] = Default::default();
            let mut ability_levels: [i32; 4] = [0; 4];
            if let Some(h) = hero_storage.get(entity) {
                for i in 0..4 {
                    if let Some(id) = h.abilities.get(i) {
                        let lvl = h.ability_levels.get(id).copied().unwrap_or(0);
                        ability_levels[i] = lvl;
                        ability_ids[i] = Some(id.clone());
                    }
                }
            }

            Some(Box::new(HeroStatsExt {
                armor,
                magic_resist,
                attack_damage,
                attack_range,
                move_speed,
                attack_speed_sec,
                bullet_speed,
                mana,
                max_mana,
                buffs,
                inventory,
                ability_levels,
                ability_ids,
            }))
        } else {
            None
        };

        // Phase 4.3: tower upgrade levels — only populated for Tower-kind
        // entities. The 3 paths × 0-4 level array is read directly off the
        // `Tower` component. Other kinds get `None` (zero overhead).
        let upgrade_levels: Option<[u8; 3]> = if matches!(kind, EntityKind::Tower) {
            tower_storage.get(entity).map(|t| t.upgrade_levels)
        } else {
            None
        };

        out.push(EntityRenderData {
            entity_id: entity.id(),
            // specs `Generation::id()` returns i32 (1-based, with sign
            // tracking alive/dead). Cast to u32 for snapshot transport.
            entity_gen: entity.gen().id() as u32,
            kind,
            pos_x: px,
            pos_y: py,
            facing_rad: facing,
            hp,
            max_hp,
            unit_id,
            hero_name,
            hero_title,
            hero_level,
            hero_xp,
            hero_xp_next,
            hero_skill_points,
            hero_primary_attribute,
            hero_strength,
            hero_agility,
            hero_intelligence,
            gold,
            hero_ext,
            upgrade_levels,
        });
    }

    // Creep checkpoint paths — read once per snapshot from the static
    // `BTreeMap<String, Path>` resource populated by `init_creep_wave`. Cheap
    // (BTree iter + small clone); avoids a dedicated init-only channel.
    use omobab::comp::Path;
    use std::collections::BTreeMap;
    let paths: Vec<Vec<(f32, f32)>> = world
        .read_resource::<BTreeMap<String, Path>>()
        .values()
        .map(|p| p.check_points.iter().map(|cp| (cp.pos.x, cp.pos.y)).collect())
        .collect();

    // DIAGNOSTIC: dump entity kind histogram + samples every second so we can
    // pinpoint why sim_runner accumulates ghost entities (411 reported with
    // empty Structures + BlockedRegions). Remove after root cause is fixed.
    if tick % 60 == 0 && !out.is_empty() {
        let mut counts = [0u32; 5];
        for e in &out {
            counts[match e.kind {
                EntityKind::Hero => 0,
                EntityKind::Tower => 1,
                EntityKind::Creep => 2,
                EntityKind::Projectile => 3,
                EntityKind::Other => 4,
            }] += 1;
        }
        log::info!(
            "[sim_runner] tick={} total={} hero={} tower={} creep={} proj={} other={}",
            tick, out.len(), counts[0], counts[1], counts[2], counts[3], counts[4],
        );
        // First 10 non-hero entities — show their pos / unit_id to hint at origin.
        for (i, e) in out.iter().filter(|e| !matches!(e.kind, EntityKind::Hero)).enumerate().take(10) {
            log::info!(
                "  [{}] id={} gen={} kind={:?} unit_id={:?} pos=({:.0},{:.0}) hp={}/{}",
                i, e.entity_id, e.entity_gen, e.kind, e.unit_id,
                e.pos_x, e.pos_y, e.hp, e.max_hp,
            );
        }
    }

    // Phase 1b: drain `RemovedEntitiesQueue` — `delete_entity_tracked` 推入
    // (與 entities().delete(e) 同步配對)。同 ExplosionFxQueue 模式 — sim
    // 不讀此 queue 所以 write 不影響 determinism。取代了原本 prev_alive
    // HashSet 跨 tick state diff 演算法。
    let removed_entity_ids: Vec<u32> = {
        let mut q = world.write_resource::<omobab::comp::RemovedEntitiesQueue>();
        std::mem::take(&mut q.pending)
    };

    // Phase 4.1: BlockedRegion polygons. Static map data — TD_1 is empty,
    // MVP_1/DEBUG_1 have a handful, so cloning each tick is cheap. The
    // omb `BlockedRegion` has `name: String` + `points: Vec<Vec2<f32>>`
    // (no radius); we project to `(f32, f32)` pairs since render-side
    // already speaks raw f32 world coords. `circle` is None today since
    // the source has no radius field; kept Optional for forward-compat.
    let blocked_regions: Vec<BlockedRegionSnapshot> = world
        .read_resource::<omobab::comp::BlockedRegions>()
        .0
        .iter()
        .map(|r| BlockedRegionSnapshot {
            points: r.points.iter().map(|p| (p.x, p.y)).collect(),
            circle: None,
        })
        .collect();

    // Phase 3.2: TD HUD state — Round / Lives / round_is_running.
    // `CurrentCreepWave.wave` is `usize` 1-based once StartRound flips
    // `is_running`; 0 before the first round. `total_rounds` = length of
    // the `Vec<CreepWave>` resource. `PlayerLives` is a tuple-struct
    // wrapping `i32`. All three are read-only reads against ECS
    // resources — no mutation, so determinism is unaffected.
    let round: u32;
    let total_rounds: u32;
    let round_is_running: bool;
    {
        let ccw = world.read_resource::<omobab::comp::CurrentCreepWave>();
        round = ccw.wave as u32;
        round_is_running = ccw.is_running;
    }
    {
        let waves = world.read_resource::<Vec<omobab::comp::CreepWave>>();
        total_rounds = waves.len() as u32;
    }
    let lives = world.read_resource::<omobab::comp::PlayerLives>().0;

    // Phase 4.2: drain `ExplosionFxQueue` — process_outcomes pushes here
    // for every Outcome::Explosion (game_processor + WorldAdapter
    // emit_explosion). `std::mem::take` swaps in default empty Vec so the
    // queue stays O(1) memory; sim never reads the queue back, so the
    // write is invisible to determinism (same reason BlockedRegions is
    // safe to read here).
    let explosions: Vec<ExplosionFx> = {
        let mut q = world.write_resource::<omobab::comp::ExplosionFxQueue>();
        std::mem::take(&mut q.pending)
    };

    SimWorldSnapshot {
        tick,
        entities: out,
        paths,
        removed_entity_ids,
        round,
        total_rounds,
        lives,
        round_is_running,
        blocked_regions,
        abilities: abilities_arc,
        tower_templates: tower_templates_arc,
        tower_upgrades: tower_upgrades_arc,
        explosions,
        applied_input_ids,
    }
}

/// Smoke test that omobab as lib is reachable. Verifies the dep wiring
/// works and the Phase 3.2 helper symbols resolve.
pub fn smoke() -> &'static str {
    let _ = omobab::comp::resources::MasterSeed::default();
    // Reach into the new pub helpers added in Phase 3.2 to confirm
    // they're visible from omfx.
    let _ = omobab::state::system_dispatcher::build_phase3_dispatcher
        as fn() -> Result<specs::Dispatcher<'static, 'static>, failure::Error>;
    "omobab linked"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_links() {
        assert_eq!(smoke(), "omobab linked");
    }

    #[test]
    fn snapshot_default() {
        let s = SimWorldSnapshot::default();
        assert_eq!(s.tick, 0);
        assert!(s.entities.is_empty());
    }

    #[test]
    fn entity_kind_eq() {
        assert_eq!(EntityKind::Hero, EntityKind::Hero);
        assert_ne!(EntityKind::Hero, EntityKind::Tower);
    }
}
