//! 階段 2 的簡化版 `LockstepClient`，供 omfx 使用。
//!
//! 連線到 omb 的 lockstep 通道（KCP tag 0x10–0x16），以 Player 身分加入，
//! 並接收 `TickBatch` / `StateHash` 資料。
//!
//! 階段 2 僅做「紀錄」用途。事件會透過 crossbeam 通道
//! 傳到 Fyrox 主執行緒，以 info/debug 等級輸出；
//! `TickBatch` 採用 debug，且每 60 幀取樣一次以避免訊息刷屏。
//!
//! native frontend 的 gameplay replica 由 `omoba-core::runtime` 驅動；
//! 這個 client 只負責 KCP lockstep wire 邊界與跨 thread 事件轉送。
//!
//! 設計重點：
//! - 會額外啟動自己的背景執行緒與 tokio current-thread runtime，避免網路 I/O
//!   阻塞 Fyrox render thread。
//! - 先呼叫 `omoba_core::KcpClient::join_lockstep`（送出 JoinRequest 0x13，
//!   等待 GameStart 0x14），再呼叫 `subscribe_lockstep` 取得已啟用
//!   的 lockstep mpsc receiver（由 KCP reader task 推資料）。
//! - 每次收到入站後，將 `input_tx` 的輸入非阻塞清空。階段 2
//!   尚未接上 UI 輸入；階段 3 將接鍵盤／滑鼠輸入。
//!

use std::future::Future;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use crossbeam_channel::{unbounded, Receiver, Sender};
use log::{debug, error, info, warn};

use omoba_core::kcp::client::LockstepInbound;
use omoba_core::kcp::game_proto::{PlayerInput, ServerEvent};
use omoba_core::lockstep_timing::LOCKSTEP_TPS;
use omoba_core::KcpClient;
use tokio::sync::watch;

fn wall_clock_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros().min(u128::from(u64::MAX)) as u64)
        .unwrap_or(0)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputOriginKind {
    OsEvent,
    Auto,
    Direct,
}

impl Default for InputOriginKind {
    fn default() -> Self {
        Self::Direct
    }
}

#[derive(Debug, Clone)]
pub struct LockstepInputMsg {
    pub target_tick: u32,
    pub input: PlayerInput,
    pub input_id: u32,
    pub origin_kind: InputOriginKind,
    pub origin_us: u64,
    pub send_lockstep_input_us: u64,
}

#[derive(Debug, Clone)]
pub struct LockstepTickInput {
    pub player_id: u32,
    pub input: PlayerInput,
    pub input_id: u32,
    pub server_receive_tick: u32,
    pub server_drain_tick: u32,
    pub server_queue_us: u64,
    pub client_receive_us: u64,
}

/// 從 lockstep 背景執行緒轉到 Fyrox 主執行緒的診斷事件。
/// 階段 2 僅做紀錄；階段 3 後續會轉交本地模擬消費端。

#[derive(Debug, Clone)]
pub enum LockstepEvent {
    Connected {
        master_seed: u64,
        player_id: u32,
        step_fps: u32,
    },
    /// 階段 3.3：改為攜帶完整 TickBatch 內容（inputs + server events），
    /// 而非只傳筆數；讓 sim_runner 可用實際玩家輸入驅動 ECS dispatcher。
    TickBatch {
        tick: u32,
        inputs: Vec<LockstepTickInput>,
        server_events: Vec<ServerEvent>,
        lua_content_generation: u64,
        lua_content_hash: String,
    },
    InputSubmitted {
        input_id: u32,
        submit_start_us: u64,
        submit_done_us: u64,
    },
    StateHash {
        tick: u32,
        hash: u64,
    },
    /// 自上次上報以來的網路吞吐位元組增量，含入站
    /// （`TickBatch` / `StateHash` / `SnapshotResp` / `GameStart`）與
    /// 出站（`InputSubmit`）兩方向；兩邊都算「lockstep 流量」。
    /// 背景執行緒每次收完一個 frame 會輸出一筆，主執行緒再彙總到
    /// 每秒 HUD 計數。
    NetStats {
        wire_delta: u64,
        logical_delta: u64,
    },
    /// 從最近一次 `PingResponse` 取得 RTT，`pong` 每秒約 1 次更新一次；
    /// HUD 顯示最後一筆結果。
    Latency {
        rtt_us: u64,
    },
    Disconnected {
        reason: String,
    },
}

/// 背景 lockstep client 的操作句柄。
/// Session 結束時必須呼叫 `shutdown`，取消所有非同步等待並 join 背景執行緒。
#[derive(Debug)]
pub struct LockstepClientHandle {
    pub events_rx: Receiver<LockstepEvent>,
    /// 階段 2 暫存區 — UI 還不會產生輸入；階段 3 會透過這裡
    /// 將玩家輸入轉交背景 client，並由背景端呼叫 `submit_input`。
    pub input_tx: Sender<LockstepInputMsg>,
    input_id_counter: AtomicU32,
    latest_tick: Arc<AtomicU32>,
    cancel_tx: watch::Sender<bool>,
    /// 保留 join handle，讓 session shutdown 能等待背景執行緒結束。
    _thread: thread::JoinHandle<()>,
}

impl LockstepClientHandle {
    pub fn next_input_id(&self) -> u32 {
        self.input_id_counter.fetch_add(1, Ordering::Relaxed)
    }

    pub fn latest_tick(&self) -> u32 {
        self.latest_tick.load(Ordering::Relaxed)
    }

    pub fn shutdown(self) {
        let Self {
            input_tx,
            cancel_tx,
            _thread,
            ..
        } = self;
        let _ = cancel_tx.send(true);
        drop(input_tx);
        if _thread.join().is_err() {
            error!("lockstep-client worker panicked during shutdown");
        }
    }
}

async fn cancel_or<F>(cancel_rx: &mut watch::Receiver<bool>, future: F) -> Option<F::Output>
where
    F: Future,
{
    if *cancel_rx.borrow() {
        return None;
    }
    tokio::select! {
        biased;
        changed = cancel_rx.changed() => {
            let _ = changed;
            None
        }
        output = future => Some(output),
    }
}

/// 啟動 lockstep 客戶端背景執行緒。
pub fn spawn_lockstep_client(
    addr: String,
    player_name: String,
    player_id: u32,
) -> LockstepClientHandle {
    let (events_tx, events_rx) = unbounded();
    let (input_tx, input_rx) = unbounded::<LockstepInputMsg>();
    let (cancel_tx, cancel_rx) = watch::channel(false);
    let latest_tick = Arc::new(AtomicU32::new(0));
    let latest_tick_for_thread = latest_tick.clone();

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
                run_client(
                    addr,
                    player_name,
                    player_id,
                    events_tx,
                    input_rx,
                    latest_tick_for_thread,
                    cancel_rx,
                )
                .await;
            });
        })
        .expect("spawn omfx-lockstep-client thread");

    LockstepClientHandle {
        events_rx,
        input_tx,
        input_id_counter: AtomicU32::new(1),
        latest_tick,
        cancel_tx,
        _thread: handle,
    }
}

async fn run_client(
    addr: String,
    player_name: String,
    player_id: u32,
    events_tx: Sender<LockstepEvent>,
    input_rx: Receiver<LockstepInputMsg>,
    latest_tick: Arc<AtomicU32>,
    mut cancel_rx: watch::Receiver<bool>,
) {
    let mut attempt = 0u32;
    loop {
        if *cancel_rx.borrow() {
            return;
        }
        attempt += 1;
        if attempt > 1 {
            // 指數退避：1s, 2s, 4s, 8s 上限
            let wait_secs = std::cmp::min(1u64 << (attempt - 2).min(3), 8);
            info!(
                "lockstep-client: reconnect attempt {} (waiting {}s)",
                attempt, wait_secs
            );
            if cancel_or(
                &mut cancel_rx,
                tokio::time::sleep(std::time::Duration::from_secs(wait_secs)),
            )
            .await
            .is_none()
            {
                return;
            }
            // 清空上一個 session 殘留的輸入
            while input_rx.try_recv().is_ok() {}
        }

        // 若 events 接收端已關閉（LockstepClientHandle 被 drop），停止重試
        macro_rules! send_or_return {
            ($ev:expr) => {
                if events_tx.send($ev).is_err() {
                    return;
                }
            };
        }

        info!(
            "lockstep-client: connecting to {} (attempt {})",
            addr, attempt
        );

        // 用新的 KcpClient，舊的 NetworkBridge 會有自己的。`connect`
        // 會附帶送出 SubscribeRequest；階段 2 在行為上可接受，
        // 伺服器可容忍，且 lockstep tag 仍走同一 socket。
        let mut client = match cancel_or(
            &mut cancel_rx,
            KcpClient::connect(&addr, player_name.clone()),
        )
        .await
        {
            None => return,
            Some(Ok(c)) => c,
            Some(Err(e)) => {
                warn!("lockstep-client connect failed: {}", e);
                send_or_return!(LockstepEvent::Disconnected {
                    reason: format!("connect: {}", e),
                });
                continue;
            }
        };

        // 送出 JoinRequest 0x13 並等待 GameStart 0x14，回傳
        // `master_seed`。階段 2 固定以 Player 身分加入（observer = false）。
        // 10 秒 timeout：若伺服器拒絕 JoinRequest（不回 GameStart），
        // 避免永遠卡住，讓外層重連邏輯可以繼續。
        let join_result = match cancel_or(
            &mut cancel_rx,
            tokio::time::timeout(
                std::time::Duration::from_secs(10),
                client.join_lockstep(player_name.clone(), player_id, false),
            ),
        )
        .await
        {
            Some(result) => result,
            None => return,
        };
        let master_seed = match join_result {
            Err(_timeout) => {
                warn!("lockstep-client join_lockstep timed out after 10s (server may have rejected player_id={} as duplicate)", player_id);
                send_or_return!(LockstepEvent::Disconnected {
                    reason: "join timeout (server did not send GameStart within 10s)".into(),
                });
                continue;
            }
            Ok(Err(e)) => {
                warn!("lockstep-client join_lockstep failed: {}", e);
                send_or_return!(LockstepEvent::Disconnected {
                    reason: format!("join: {}", e),
                });
                continue;
            }
            Ok(Ok(seed)) => seed,
        };
        let player_id = client.lockstep_player_id().unwrap_or(0);
        let step_fps = client.lockstep_step_fps().unwrap_or(LOCKSTEP_TPS);
        info!(
            "lockstep-client joined: master_seed=0x{:016x} player_id={} step_fps={}",
            master_seed, player_id, step_fps
        );
        send_or_return!(LockstepEvent::Connected {
            master_seed,
            player_id,
            step_fps,
        });

        // 接管 lockstep 入站串流。`join_lockstep` 已經耗掉第一筆
        // `GameStart`，之後這個 rx 會回傳 `TickBatch` / `StateHash` /
        // `SnapshotResp`，若又收到額外 `GameStart`（階段 2 不預期）也一併接收。
        let mut rx = match client.subscribe_lockstep() {
            Ok(r) => r,
            Err(e) => {
                warn!("lockstep-client subscribe_lockstep failed: {}", e);
                send_or_return!(LockstepEvent::Disconnected {
                    reason: format!("subscribe: {}", e),
                });
                continue;
            }
        };

        // 成功連線並加入，重設退避計數
        attempt = 0;

        // 主迴圈：輪詢入站並頻繁清空待送輸入。
        // 接收逾時若拉長，會累積到約一個 TickBatch 的輸入延遲，
        // 會讓預期的 +3 tick 前瞻在 localhost 上變晚。
        let mut last_hb_log = std::time::Instant::now();
        let mut last_stall_log = std::time::Instant::now();
        let mut last_tickbatch_time = std::time::Instant::now();
        let mut tick_batches_since_log: u32 = 0;
        let mut last_known_tick: u32 = 0;
        // 每次迴圈的位元組增量，會在尾段輸出到 NetStats。
        // 入站與出站同時計入，讓 HUD 顯示總 lockstep 流量。
        let mut wire_delta: u64;
        let mut logical_delta: u64;
        'inner: loop {
            wire_delta = 0;
            logical_delta = 0;
            let recv_result = match cancel_or(
                &mut cancel_rx,
                tokio::time::timeout(std::time::Duration::from_millis(2), rx.recv()),
            )
            .await
            {
                Some(result) => result,
                None => return,
            };
            match recv_result {
                Err(_elapsed) => {
                    let now = std::time::Instant::now();
                    if now.duration_since(last_stall_log).as_secs() >= 2 {
                        warn!(
                            "[lockstep-client] no KCP frame in 2.0s (last_known_tick={}). \
                             Upstream omb→KCP path is the suspect.",
                            last_known_tick,
                        );
                        last_stall_log = now;
                    }
                    if now.duration_since(last_tickbatch_time).as_secs() >= 30 {
                        warn!(
                            "[lockstep-client] no TickBatch for 30s (last_known_tick={}) — forcing reconnect",
                            last_known_tick,
                        );
                        send_or_return!(LockstepEvent::Disconnected {
                            reason: format!(
                                "stall timeout: no TickBatch for 30s (last_tick={})",
                                last_known_tick
                            ),
                        });
                        break 'inner;
                    }
                }
                Ok(None) => {
                    warn!("lockstep-client stream closed");
                    send_or_return!(LockstepEvent::Disconnected {
                        reason: "stream closed".into(),
                    });
                    break 'inner;
                }
                Ok(Some(LockstepInbound::TickBatch {
                    msg: b,
                    wire_bytes,
                    logical_bytes,
                })) => {
                    let client_receive_us = wall_clock_us();
                    wire_delta += wire_bytes as u64;
                    logical_delta += logical_bytes as u64;
                    tick_batches_since_log += 1;
                    last_known_tick = b.tick;
                    latest_tick.store(b.tick, Ordering::Relaxed);
                    let now = std::time::Instant::now();
                    last_stall_log = now;
                    last_tickbatch_time = now;
                    if now.duration_since(last_hb_log).as_secs() >= 5 {
                        debug!(
                            "[lockstep-client] healthy: {} TickBatch frames in last 5s (latest tick={})",
                            tick_batches_since_log, b.tick,
                        );
                        last_hb_log = now;
                        tick_batches_since_log = 0;
                    }
                    // 階段 3.3：從 `InputForPlayer` 的各列抽出輸入與 edge metadata。
                    let inputs: Vec<LockstepTickInput> = b
                        .inputs
                        .into_iter()
                        .filter_map(|ifp| {
                            ifp.input.map(|inp| LockstepTickInput {
                                player_id: ifp.player_id,
                                input: inp,
                                input_id: ifp.input_id,
                                server_receive_tick: ifp.server_receive_tick,
                                server_drain_tick: ifp.server_drain_tick,
                                server_queue_us: ifp.server_queue_us,
                                client_receive_us,
                            })
                        })
                        .collect();
                    let server_events = b.server_events;
                    if events_tx
                        .send(LockstepEvent::TickBatch {
                            tick: b.tick,
                            inputs,
                            server_events,
                            lua_content_generation: b.lua_content_generation,
                            lua_content_hash: b.lua_content_hash,
                        })
                        .is_err()
                    {
                        return;
                    }
                }
                Ok(Some(LockstepInbound::StateHash {
                    msg: sh,
                    wire_bytes,
                    logical_bytes,
                })) => {
                    wire_delta += wire_bytes as u64;
                    logical_delta += logical_bytes as u64;
                    let _ = events_tx.send(LockstepEvent::StateHash {
                        tick: sh.tick,
                        hash: sh.hash,
                    });
                }
                Ok(Some(LockstepInbound::GameStart {
                    wire_bytes,
                    logical_bytes,
                    ..
                })) => {
                    wire_delta += wire_bytes as u64;
                    logical_delta += logical_bytes as u64;
                    warn!("lockstep-client got unexpected GameStart after join — ignoring");
                }
                Ok(Some(LockstepInbound::Pong {
                    rtt_us,
                    wire_bytes,
                    logical_bytes,
                })) => {
                    wire_delta += wire_bytes as u64;
                    logical_delta += logical_bytes as u64;
                    let _ = events_tx.send(LockstepEvent::Latency { rtt_us });
                }
                Ok(Some(LockstepInbound::SnapshotResp {
                    msg: resp,
                    wire_bytes,
                    logical_bytes,
                })) => {
                    wire_delta += wire_bytes as u64;
                    logical_delta += logical_bytes as u64;
                    let (bytes_len, schema) = match &resp.state {
                        Some(s) => (s.world_bytes.len(), s.schema_version),
                        None => (0, 0),
                    };
                    info!(
                        "lockstep-client received SnapshotResp tick={} bytes={} schema={} (Phase 5.3 logs only; apply is Phase 5+)",
                        resp.tick, bytes_len, schema
                    );
                }
            }

            // 非阻塞清空待送輸入。`InputSubmit` 的位元組也會納入同一個
            // lockstep 流量總數。
            while let Ok(msg) = input_rx.try_recv() {
                let submit_start_us = wall_clock_us();
                let submit_result = match cancel_or(
                    &mut cancel_rx,
                    client.submit_input(msg.target_tick, msg.input, msg.input_id),
                )
                .await
                {
                    Some(result) => result,
                    None => return,
                };
                match submit_result {
                    Ok((logical, wire)) => {
                        let submit_done_us = wall_clock_us();
                        wire_delta += wire as u64;
                        logical_delta += logical as u64;
                        let _ = events_tx.send(LockstepEvent::InputSubmitted {
                            input_id: msg.input_id,
                            submit_start_us,
                            submit_done_us,
                        });
                    }
                    Err(e) => warn!("lockstep-client submit_input failed: {}", e),
                }
            }

            if wire_delta > 0 || logical_delta > 0 {
                let _ = events_tx.send(LockstepEvent::NetStats {
                    wire_delta,
                    logical_delta,
                });
            }
        }
        // inner loop ended → outer loop retries the connection
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    #[test]
    fn cancellation_interrupts_a_pending_handshake() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap();
        let (cancel_tx, mut cancel_rx) = tokio::sync::watch::channel(false);
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(25));
            cancel_tx.send(true).unwrap();
        });

        let started = Instant::now();
        let result = runtime.block_on(cancel_or(&mut cancel_rx, std::future::pending::<()>()));

        assert!(result.is_none());
        assert!(started.elapsed() < Duration::from_secs(1));
    }
}
