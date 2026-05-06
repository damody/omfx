//! 階段 2 的簡化版 `LockstepClient`，供 omfx 使用。
//!
//! 連線到 omb 的 lockstep 通道（KCP tag 0x10–0x16），以 Player 身分加入，
//! 並接收 `TickBatch` / `StateHash` 資料。
//!
//! 階段 2 僅做「紀錄」用途。事件會透過 crossbeam 通道
//! 傳到 Fyrox 主執行緒，以 info/debug 等級輸出；
//! `TickBatch` 採用 debug，且每 60 幀取樣一次以避免訊息刷屏。
//!
//! 階段 3 預計以真正的 omoba_sim ECS 替代舊有 `GameEvent` 流式消費器，
//! 並由 TickBatch 輸入驅動渲染；目前階段先保留舊的 `NetworkBridge` 與其
//! 平行運作。
//!
//! 設計重點：
//! - 會額外啟動自己的背景執行緒與 tokio current-thread runtime，做法上
//!   對齊 `NetworkBridge::spawn`。兩條路徑不共用 runtime，
//!   其中任一發生卡死/崩潰不會拖垮另一條。
//! - 先呼叫 `omoba_core::KcpClient::join_lockstep`（送出 JoinRequest 0x13，
//!   等待 GameStart 0x14），再呼叫 `subscribe_lockstep` 取得已啟用
//!   的 lockstep mpsc receiver（由 KCP reader task 推資料）。
//! - 每次收到入站後，將 `input_tx` 的輸入非阻塞清空。階段 2
//!   尚未接上 UI 輸入；階段 3 將接鍵盤／滑鼠輸入。
//!

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::thread;

use crossbeam_channel::{unbounded, Receiver, Sender};
use log::{error, info, warn};

use omoba_core::kcp::client::LockstepInbound;
use omoba_core::kcp::game_proto::{PlayerInput, ServerEvent};
use omoba_core::KcpClient;

/// 從 lockstep 背景執行緒轉到 Fyrox 主執行緒的診斷事件。
/// 階段 2 僅做紀錄；階段 3 後續會轉交本地模擬消費端。

#[derive(Debug, Clone)]
pub enum LockstepEvent {
    Connected { master_seed: u64, player_id: u32 },
    /// 階段 3.3：改為攜帶完整 TickBatch 內容（inputs + server events），
    /// 而非只傳筆數；讓 sim_runner 可用實際玩家輸入驅動 ECS dispatcher。

    TickBatch {
        tick: u32,
        inputs: Vec<(u32 /* player_id */, PlayerInput, u32 /* input_id */)>,
        server_events: Vec<ServerEvent>,
    },
    StateHash { tick: u32, hash: u64 },
    /// 自上次上報以來的網路吞吐位元組增量，含入站
    /// （`TickBatch` / `StateHash` / `SnapshotResp` / `GameStart`）與
    /// 出站（`InputSubmit`）兩方向；兩邊都算「lockstep 流量」。
    /// 背景執行緒每次收完一個 frame 會輸出一筆，主執行緒再彙總到
    /// 每秒 HUD 計數。
    NetStats { wire_delta: u64, logical_delta: u64 },
    /// 從最近一次 `PingResponse` 取得 RTT，`pong` 每秒約 1 次更新一次；
    /// HUD 顯示最後一筆結果。
    Latency { rtt_us: u64 },
    Disconnected { reason: String },
}

/// 傳給 omfx 的輸入訊息：目標 tick + prost 的 PlayerInput payload +
/// omfx 用來做統計的輸入 id。
pub type LockstepInputMsg = (u32, PlayerInput, u32);

/// 背景 lockstep client 的操作句柄。
/// 丟棄時會關閉通道，下一輪迴圈使背景執行緒自然結束。
#[derive(Debug)]
pub struct LockstepClientHandle {
    pub events_rx: Receiver<LockstepEvent>,
    /// 階段 2 暫存區 — UI 還不會產生輸入；階段 3 會透過這裡
    /// 將玩家輸入轉交背景 client，並由背景端呼叫 `submit_input`。
    pub input_tx: Sender<LockstepInputMsg>,
    input_id_counter: AtomicU32,
    latest_tick: Arc<AtomicU32>,
    /// 保留 join handle，讓 Handle 被釋放時也能關閉背景執行緒；
    /// 通道關閉後背景執行緒會跳出迴圈。
    _thread: thread::JoinHandle<()>,
}

impl LockstepClientHandle {
    pub fn next_input_id(&self) -> u32 {
        self.input_id_counter.fetch_add(1, Ordering::Relaxed)
    }

    pub fn latest_tick(&self) -> u32 {
        self.latest_tick.load(Ordering::Relaxed)
    }
}

/// 啟動 lockstep 客戶端背景執行緒。
pub fn spawn_lockstep_client(addr: String, player_name: String) -> LockstepClientHandle {
    let (events_tx, events_rx) = unbounded();
    let (input_tx, input_rx) = unbounded::<LockstepInputMsg>();
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
                run_client(addr, player_name, events_tx, input_rx, latest_tick_for_thread).await;
            });
        })
        .expect("spawn omfx-lockstep-client thread");

    LockstepClientHandle {
        events_rx,
        input_tx,
        input_id_counter: AtomicU32::new(1),
        latest_tick,
        _thread: handle,
    }
}

async fn run_client(
    addr: String,
    player_name: String,
    events_tx: Sender<LockstepEvent>,
    input_rx: Receiver<LockstepInputMsg>,
    latest_tick: Arc<AtomicU32>,
) {
    info!("lockstep-client connecting to {}", addr);

    // 用新的 KcpClient，舊的 NetworkBridge 會有自己的。`connect`
    // 會附帶送出 SubscribeRequest；階段 2 在行為上可接受，
    // 伺服器可容忍，且 lockstep tag 仍走同一 socket。
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

    // 送出 JoinRequest 0x13 並等待 GameStart 0x14，回傳
    // `master_seed`。階段 2 固定以 Player 身分加入（observer = false）。
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

    // 接管 lockstep 入站串流。`join_lockstep` 已經耗掉第一筆
    // `GameStart`，之後這個 rx 會回傳 `TickBatch` / `StateHash` /
    // `SnapshotResp`，若又收到額外 `GameStart`（階段 2 不預期）也一併接收。
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

    // 主迴圈：輪詢入站並頻繁清空待送輸入。
    // 接收逾時若拉長，會累積到約一個 TickBatch 的輸入延遲，
    // 會讓預期的 +3 tick 前瞻在 localhost 上變晚。
    let mut last_hb_log = std::time::Instant::now();
    let mut last_stall_log = std::time::Instant::now();
    let mut tick_batches_since_log: u32 = 0;
    let mut last_known_tick: u32 = 0;
    // 每次迴圈的位元組增量，會在尾段輸出到 NetStats。
    // 入站與出站同時計入，讓 HUD 顯示總 lockstep 流量。
    let mut wire_delta: u64;
    let mut logical_delta: u64;
    loop {
        wire_delta = 0;
        logical_delta = 0;
        let recv_result = tokio::time::timeout(
            std::time::Duration::from_millis(2),
            rx.recv(),
        ).await;
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
            }
            Ok(None) => {
                warn!("lockstep-client stream closed");
                let _ = events_tx.send(LockstepEvent::Disconnected {
                    reason: "stream closed".into(),
                });
                break;
            }
            Ok(Some(LockstepInbound::TickBatch { msg: b, wire_bytes, logical_bytes })) => {
                wire_delta += wire_bytes as u64;
                logical_delta += logical_bytes as u64;
                tick_batches_since_log += 1;
                last_known_tick = b.tick;
                latest_tick.store(b.tick, Ordering::Relaxed);
                let now = std::time::Instant::now();
                last_stall_log = now;
                if now.duration_since(last_hb_log).as_secs() >= 5 {
                    info!(
                        "[lockstep-client] healthy: {} TickBatch frames in last 5s (latest tick={})",
                        tick_batches_since_log, b.tick,
                    );
                    last_hb_log = now;
                    tick_batches_since_log = 0;
                }
                // 階段 3.3：從 `InputForPlayer` 的各列抽出
                // `Vec<(player_id, PlayerInput)>`。
                let inputs: Vec<(u32, PlayerInput, u32)> = b
                    .inputs
                    .into_iter()
                    .filter_map(|ifp| ifp.input.map(|inp| (ifp.player_id, inp, ifp.input_id)))
                    .collect();
                let server_events = b.server_events;
                let _ = events_tx.send(LockstepEvent::TickBatch {
                    tick: b.tick,
                    inputs,
                    server_events,
                });
            }
            Ok(Some(LockstepInbound::StateHash { msg: sh, wire_bytes, logical_bytes })) => {
                wire_delta += wire_bytes as u64;
                logical_delta += logical_bytes as u64;
                let _ = events_tx.send(LockstepEvent::StateHash {
                    tick: sh.tick,
                    hash: sh.hash,
                });
            }
            Ok(Some(LockstepInbound::GameStart { wire_bytes, logical_bytes, .. })) => {
                wire_delta += wire_bytes as u64;
                logical_delta += logical_bytes as u64;
                warn!("lockstep-client got unexpected GameStart after join — ignoring");
            }
            Ok(Some(LockstepInbound::Pong { rtt_us, wire_bytes, logical_bytes })) => {
                wire_delta += wire_bytes as u64;
                logical_delta += logical_bytes as u64;
                let _ = events_tx.send(LockstepEvent::Latency { rtt_us });
            }
            Ok(Some(LockstepInbound::SnapshotResp { msg: resp, wire_bytes, logical_bytes })) => {
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
        while let Ok((target_tick, input, input_id)) = input_rx.try_recv() {
            match client.submit_input(target_tick, input, input_id).await {
                Ok((logical, wire)) => {
                    wire_delta += wire as u64;
                    logical_delta += logical as u64;
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
}
