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
    pub inputs: Vec<(u32 /* player_id */, PlayerInput)>,
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
    // Worker-local set of entity ids alive in the previous snapshot. Diff
    // against the current snapshot's ids each tick to populate
    // `SimWorldSnapshot.removed_entity_ids` — the render thread uses this to
    // free per-eid scene caches (labels, batch slots) instead of relying on
    // a wire-side `entity.death` event. Replaces the legacy omb
    // `make_entity_death` emit pair.
    let mut prev_alive: std::collections::HashSet<u32> = std::collections::HashSet::new();
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

        let snapshot = extract_snapshot(&world, batch.tick, &mut prev_alive);
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

fn push_inputs_into_world(world: &mut World, tick: u32, inputs: Vec<(u32, PlayerInput)>) {
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
    for (player_id, input) in inputs {
        pending.by_player.insert(player_id, input);
    }
}

fn extract_snapshot(
    world: &World,
    tick: u32,
    prev_alive: &mut std::collections::HashSet<u32>,
) -> SimWorldSnapshot {
    // omobab re-exports these via `pub use crate::comp::*;` at the
    // crate root, so go through the flat path instead of the
    // module-by-module one (some submodules like `comp::state` collide
    // with the State struct namespace).
    use omobab::{CProperty, Creep, Facing, Hero, Pos, Projectile, Tower};
    use omobab::comp::hero::AttributeType;
    use omobab::comp::gold::Gold;
    use omobab::scripting::ScriptUnitTag;

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

    // Diff vs. previous tick's alive set → the entity ids that just dropped
    // out. Render thread uses this to free per-eid scene caches.
    let current_alive: std::collections::HashSet<u32> =
        out.iter().map(|e| e.entity_id).collect();
    let removed_entity_ids: Vec<u32> =
        prev_alive.difference(&current_alive).copied().collect();
    *prev_alive = current_alive;

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

    SimWorldSnapshot {
        tick,
        entities: out,
        paths,
        removed_entity_ids,
        round,
        total_rounds,
        lives,
        round_is_running,
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
