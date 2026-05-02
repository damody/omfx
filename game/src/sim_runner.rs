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
}

/// Per-entity render data extracted from the ECS World at the end of
/// every tick. Already mapped to `f32` at the boundary (Fixed64 →
/// `to_f32_for_render`); render thread does not need to know about
/// the deterministic sim's fixed-point types.
#[derive(Clone, Debug)]
pub struct EntityRenderData {
    pub entity_id: u32,
    pub entity_gen: u32,
    pub kind: EntityKind,
    pub pos_x: f32,
    pub pos_y: f32,
    pub facing_rad: f32,
    pub hp: i32,
    pub max_hp: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EntityKind {
    Hero,
    Tower,
    Creep,
    Projectile,
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

    info!("sim_runner: dispatcher ready, entering tick loop");

    loop {
        let batch = match tick_input_rx.recv() {
            Ok(b) => b,
            Err(_) => {
                info!("sim_runner: input channel closed, exiting");
                break;
            }
        };

        push_inputs_into_world(&mut world, batch.tick, batch.inputs);

        // Update the Tick resource so omb-side systems see the right
        // tick number. Phase 3.3 may also need to write DeltaTime here.
        world.write_resource::<omobab::comp::resources::Tick>().0 = batch.tick as u64;

        dispatcher.dispatch(&world);
        world.maintain();

        let snapshot = extract_snapshot(&world, batch.tick);
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

fn push_inputs_into_world(_world: &mut World, tick: u32, inputs: Vec<(u32, PlayerInput)>) {
    // Phase 3.2 placeholder: just log. Phase 3.3 will write a
    // `PendingPlayerInputs` resource consumed by player_tick / hero_tick.
    if !inputs.is_empty() {
        log::trace!("sim_runner: tick {} got {} inputs", tick, inputs.len());
    }
}

fn extract_snapshot(world: &World, tick: u32) -> SimWorldSnapshot {
    // omobab re-exports these via `pub use crate::comp::*;` at the
    // crate root, so go through the flat path instead of the
    // module-by-module one (some submodules like `comp::state` collide
    // with the State struct namespace).
    use omobab::{CProperty, Creep, Facing, Hero, Pos, Projectile, Tower};

    let entities = world.entities();
    let pos_storage = world.read_storage::<Pos>();
    let facing_storage = world.read_storage::<Facing>();
    let cprop_storage = world.read_storage::<CProperty>();
    let hero_storage = world.read_storage::<Hero>();
    let tower_storage = world.read_storage::<Tower>();
    let proj_storage = world.read_storage::<Projectile>();
    let creep_storage = world.read_storage::<Creep>();

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
        });
    }

    SimWorldSnapshot {
        tick,
        entities: out,
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
