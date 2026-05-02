//! Phase 3 omfx simulator runner.
//!
//! Spawns a worker thread that runs the full omb ECS dispatcher driven by
//! TickBatch input from omb's lockstep wire. Render thread reads from a
//! published `SimWorldSnapshot` Arc<Mutex<...>>.
//!
//! Phase 3.1 = stub. Phase 3.2 wires the actual ECS / dispatcher / DLL load.

#![allow(dead_code)] // Phase 3.1 stub; full impl in Phase 3.2

use std::sync::{Arc, Mutex};

/// Render-thread-readable snapshot of the latest sim tick state.
/// Phase 3.2 will fill EntityRenderData + populate from ECS.
pub struct SimWorldSnapshot {
    pub tick: u32,
    // pub entities: Vec<EntityRenderData>,  // Phase 3.2
}

impl Default for SimWorldSnapshot {
    fn default() -> Self {
        Self { tick: 0 }
    }
}

/// Handle returned to omfx Game so render can read snapshots and forward inputs.
pub struct SimRunnerHandle {
    pub state: Arc<Mutex<SimWorldSnapshot>>,
    // Phase 3.3:
    // pub tick_input_tx: Sender<(u32, Vec<(u32, PlayerInput)>)>,
    // pub master_seed_tx: Sender<u64>,
    // _thread: thread::JoinHandle<()>,
}

/// Phase 3.1 stub. Phase 3.2 spawns the worker thread + initializes World.
pub fn spawn_sim_runner() -> SimRunnerHandle {
    log::info!("sim_runner: Phase 3.1 stub spawned (no actual sim yet)");
    SimRunnerHandle {
        state: Arc::new(Mutex::new(SimWorldSnapshot::default())),
    }
}

/// Smoke test that omobab as lib is reachable. Verifies the dep wiring works.
/// Returns the omb crate version string.
pub fn smoke() -> &'static str {
    // Reach into omobab to confirm path-dep is active.
    // Don't actually do anything — just take a type reference so compiler links it.
    let _ = omobab::comp::resources::MasterSeed::default();
    "omobab linked"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_links() {
        assert_eq!(smoke(), "omobab linked");
    }
}
