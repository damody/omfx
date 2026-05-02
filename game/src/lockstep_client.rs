//! Phase 2 minimal LockstepClient for omfx.
//!
//! Connects to omb's lockstep wire (KCP tags 0x10–0x16), joins as a Player,
//! and receives TickBatch / StateHash frames.
//!
//! Phase 2 scope = **logging only**. Events are forwarded over a crossbeam
//! channel into the Fyrox main thread, where they are logged at info / debug
//! level (debug for tick batches, sampled every 60 ticks to avoid spam).
//!
//! Phase 3 will replace the legacy GameEvent stream consumer with a real
//! omoba_sim ECS that consumes TickBatch input and drives rendering. For
//! now the legacy NetworkBridge runs unchanged in parallel.
//!
//! Design notes:
//! - Spawns its own background thread + tokio current-thread runtime, mirroring
//!   `NetworkBridge::spawn`. No shared runtime — keeps the two paths isolated
//!   so a hang/crash in one doesn't take down the other.
//! - Uses `omoba_core::KcpClient::join_lockstep` (which sends JoinRequest 0x13
//!   and awaits GameStart 0x14) followed by `subscribe_lockstep` (claims the
//!   already-running lockstep mpsc receiver fed by the kcp reader task).
//! - Outgoing inputs (`input_tx`) are drained non-blocking after each inbound
//!   recv. Phase 2 has no UI sending inputs — Phase 3 will hook keyboard /
//!   mouse to this channel.

use std::thread;

use crossbeam_channel::{unbounded, Receiver, Sender};
use log::{error, info, warn};

use omoba_core::kcp::client::LockstepInbound;
use omoba_core::kcp::game_proto::{PlayerInput, ServerEvent};
use omoba_core::KcpClient;

/// Diagnostics events forwarded from the lockstep background thread to the
/// Fyrox main thread. Phase 2 only logs these; Phase 3+ will route them into
/// the local sim consumer.
#[derive(Debug, Clone)]
pub enum LockstepEvent {
    Connected { master_seed: u64, player_id: u32 },
    /// Phase 3.3: carry the full TickBatch payload (inputs + server events)
    /// rather than just counts, so the sim_runner can drive its ECS
    /// dispatcher with the actual player inputs.
    TickBatch {
        tick: u32,
        inputs: Vec<(u32 /* player_id */, PlayerInput)>,
        server_events: Vec<ServerEvent>,
    },
    StateHash { tick: u32, hash: u64 },
    Disconnected { reason: String },
}

/// Outgoing input message: target tick + the prost PlayerInput payload.
pub type LockstepInputMsg = (u32, PlayerInput);

/// Handle to the background lockstep client. Drop kills the channels and
/// causes the bg thread to exit on its next loop iteration.
#[derive(Debug)]
pub struct LockstepClientHandle {
    pub events_rx: Receiver<LockstepEvent>,
    /// Phase 2 stub — UI does not generate inputs yet. Phase 3 will use this
    /// to forward player input to the bg client which calls `submit_input`.
    pub input_tx: Sender<LockstepInputMsg>,
    /// Keep the join handle alive so dropping the Handle takes the bg thread
    /// down too (channels close → bg thread breaks out of its loop).
    _thread: thread::JoinHandle<()>,
}

/// Spawn the lockstep client background thread.
pub fn spawn_lockstep_client(addr: String, player_name: String) -> LockstepClientHandle {
    let (events_tx, events_rx) = unbounded();
    let (input_tx, input_rx) = unbounded::<LockstepInputMsg>();

    let handle = thread::Builder::new()
        .name("omfx-lockstep-client".into())
        .spawn(move || {
            let rt = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(r) => r,
                Err(e) => {
                    error!("lockstep-client failed to build tokio runtime: {}", e);
                    let _ = events_tx.send(LockstepEvent::Disconnected {
                        reason: format!("runtime build: {}", e),
                    });
                    return;
                }
            };
            rt.block_on(async move {
                run_client(addr, player_name, events_tx, input_rx).await;
            });
        })
        .expect("spawn omfx-lockstep-client thread");

    LockstepClientHandle {
        events_rx,
        input_tx,
        _thread: handle,
    }
}

async fn run_client(
    addr: String,
    player_name: String,
    events_tx: Sender<LockstepEvent>,
    input_rx: Receiver<LockstepInputMsg>,
) {
    info!("lockstep-client connecting to {}", addr);

    // Use a fresh KcpClient — the legacy NetworkBridge has its own. KcpClient
    // sends a SubscribeRequest as part of `connect`; that's fine for Phase 2
    // (server tolerates it; lockstep tags arrive on the same socket anyway).
    let mut client = match KcpClient::connect(&addr, player_name.clone()).await {
        Ok(c) => c,
        Err(e) => {
            error!("lockstep-client connect failed: {}", e);
            let _ = events_tx.send(LockstepEvent::Disconnected {
                reason: format!("connect: {}", e),
            });
            return;
        }
    };

    // Send JoinRequest 0x13 and await GameStart 0x14. Returns master_seed.
    // Phase 2 always joins as Player (observer = false).
    let master_seed = match client.join_lockstep(player_name.clone(), false).await {
        Ok(seed) => seed,
        Err(e) => {
            error!("lockstep-client join_lockstep failed: {}", e);
            let _ = events_tx.send(LockstepEvent::Disconnected {
                reason: format!("join: {}", e),
            });
            return;
        }
    };
    let player_id = client.lockstep_player_id().unwrap_or(0);
    info!(
        "lockstep-client joined: master_seed=0x{:016x} player_id={}",
        master_seed, player_id
    );
    let _ = events_tx.send(LockstepEvent::Connected {
        master_seed,
        player_id,
    });

    // Claim the lockstep inbound stream. After join_lockstep already drained
    // the GameStart frame, this rx yields TickBatch / StateHash / SnapshotResp
    // (and any further GameStart, which Phase 2 doesn't expect).
    let mut rx = match client.subscribe_lockstep() {
        Ok(r) => r,
        Err(e) => {
            error!("lockstep-client subscribe_lockstep failed: {}", e);
            let _ = events_tx.send(LockstepEvent::Disconnected {
                reason: format!("subscribe: {}", e),
            });
            return;
        }
    };

    // Main loop: poll inbound, drain pending outgoing inputs after each recv.
    // crossbeam_channel::try_recv is non-blocking so this stays cheap.
    loop {
        match rx.recv().await {
            Some(LockstepInbound::TickBatch(b)) => {
                // Phase 3.3: extract `Vec<(player_id, PlayerInput)>` from the
                // generated `InputForPlayer` rows. Drop entries whose `input`
                // field is None (proto optional message — should not happen
                // in well-formed server output, but be defensive).
                let inputs: Vec<(u32, PlayerInput)> = b
                    .inputs
                    .into_iter()
                    .filter_map(|ifp| ifp.input.map(|inp| (ifp.player_id, inp)))
                    .collect();
                let server_events = b.server_events;
                let _ = events_tx.send(LockstepEvent::TickBatch {
                    tick: b.tick,
                    inputs,
                    server_events,
                });
            }
            Some(LockstepInbound::StateHash(sh)) => {
                let _ = events_tx.send(LockstepEvent::StateHash {
                    tick: sh.tick,
                    hash: sh.hash,
                });
            }
            Some(LockstepInbound::GameStart(_)) => {
                // Already consumed by join_lockstep; second arrival would be
                // a server bug. Log + ignore.
                warn!("lockstep-client got unexpected GameStart after join — ignoring");
            }
            Some(LockstepInbound::SnapshotResp(resp)) => {
                // Phase 5.3: server now serves real bincode-serialized
                // WorldSnapshot bytes (was empty stub in Phase 2). Client-side
                // fast-forward (deserialize + apply to sim_runner) is a Phase
                // 5+ followup once observer mode is actually exercised. For
                // now, log enough info to confirm the wire path is alive.
                let (bytes_len, schema) = match &resp.state {
                    Some(s) => (s.world_bytes.len(), s.schema_version),
                    None => (0, 0),
                };
                info!(
                    "lockstep-client received SnapshotResp tick={} bytes={} schema={} (Phase 5.3 logs only; apply is Phase 5+)",
                    resp.tick, bytes_len, schema
                );
            }
            None => {
                warn!("lockstep-client stream closed");
                let _ = events_tx.send(LockstepEvent::Disconnected {
                    reason: "stream closed".into(),
                });
                break;
            }
        }

        // Drain any pending input submissions without blocking.
        while let Ok((target_tick, input)) = input_rx.try_recv() {
            if let Err(e) = client.submit_input(target_tick, input).await {
                warn!("lockstep-client submit_input failed: {}", e);
            }
        }
    }
}
