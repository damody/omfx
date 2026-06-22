//! 第 3 階段 omfx 模擬器運行程式。
//!
//! 產生一個工作線程，運行由以下驅動的完整 omb ECS 調度程序
//! 來自 omb 鎖步線的 TickBatch 輸入。渲染線程讀取
//! `extract_data_for_render` 發布到 `SimWorldSnapshot` Arc<Mutex<...>> 的資料。
//!
//! 階段 3.1 = 存根。階段 3.2 = 現實世界 init + 調度程式循環。階段
//! 3.3 將連接 `LockstepClient` → 通道饋線。 3.4相線
//! 將快照放入渲染端並替換 TickBroadcaster 的
//! 佔位符狀態雜湊以及源自此迴圈的真實 ECS 雜湊。

#![allow(dead_code)]

use std::collections::{BTreeMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crossbeam_channel::{unbounded, Receiver, Sender, TryRecvError};
use log::{error, info};
use omoba_core::lockstep_timing::LockstepTiming;

use specs::{World, WorldExt};

pub use omoba_core::runtime::PlayerInput;

fn wall_clock_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros().min(u128::from(u64::MAX)) as u64)
        .unwrap_or(0)
}

#[derive(Clone, Debug)]
pub struct TickBatchInput {
    pub player_id: u32,
    pub input: PlayerInput,
    pub input_id: u32,
    pub server_receive_tick: u32,
    pub server_drain_tick: u32,
    pub server_queue_us: u64,
    pub client_receive_us: u64,
    pub game_forward_us: u64,
}

pub use omoba_core::runtime::{
    buff_remaining_secs_for_snapshot, build_ability_def_snapshots, build_tower_template_snapshots,
    build_tower_upgrade_def_snapshots, extract_data_for_render, hero_render_snapshot_for_unit_id,
    retain_recent_render_fx, AbilityDefSnapshot, AppliedInputMeta, AttackCancelFx, AttackPhaseFx,
    BlockedRegionSnapshot, BuffSnapshot, EntityKind, EntityRenderData, ExplosionFx,
    HeroAnimationBindingSnapshot, HeroAnimationSourceSnapshot, HeroRenderSnapshot, HeroStatsExt,
    SimWorldSnapshot, TowerBarrelVariantSnapshot, TowerFireFx, TowerRecoilSnapshot,
    TowerRenderAnimationSnapshot, TowerRenderPointSnapshot, TowerTemplateSnapshot,
    TowerUpgradeDefSnapshot,
};

pub const DEFAULT_EXTRACT_DATA_FOR_RENDER_EVERY_TICKS: u32 = 1;
const WAIT_PRECISION_WINDOW: Duration = Duration::from_millis(2);
const WAIT_STARVATION_TIMEOUT: Duration = Duration::from_secs(1);

#[derive(Clone, Copy, Debug)]
pub struct SimStartMetadata {
    pub master_seed: u64,
    pub step_fps: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct WaitPlan {
    tick_interval: Duration,
    remaining: Duration,
    sleep: Option<Duration>,
    precision_window: Duration,
}

fn plan_tick_wait(
    last_tick_started_at: Instant,
    timing: LockstepTiming,
    precision_window: Duration,
    now: Instant,
) -> WaitPlan {
    let tick_interval = timing.dt_duration();
    let deadline = last_tick_started_at + tick_interval;
    let remaining = deadline.saturating_duration_since(now);
    let sleep = remaining
        .checked_sub(precision_window)
        .filter(|duration| !duration.is_zero());
    WaitPlan {
        tick_interval,
        remaining,
        sleep,
        precision_window,
    }
}
/// 每個時脈週期由鎖步饋送器提交的通道有效負載。
#[derive(Clone, Debug)]
pub struct TickBatchPayload {
    pub tick: u32,
    pub inputs: Vec<TickBatchInput>,
    pub lua_content_generation: u64,
    pub lua_content_hash: String,
}

#[derive(Clone, Debug, Default)]
pub struct SimRunnerDiagnostics {
    pub window_ms: u128,
    pub sim_tps: f32,
    pub latest_tick: u32,
    pub queue_len: usize,
    pub max_queue_len: usize,
    pub waits: u32,
    pub blocking_receives: u32,
    pub backlog_receives: u32,
}

/// 返回 omfx 遊戲的句柄，以便渲染線程可以讀取快照
/// 鎖步饋線可以推送刻度輸入。
#[derive(Debug)]
pub struct SimRunnerHandle {
    /// 最新發布的快照。每個線程渲染一次“lock()”
    /// 框架、複製/借用和發布。
    pub state: Arc<Mutex<SimWorldSnapshot>>,
    /// 每次 TickBatch 到達時發送（刻度、輸入）。第 3.3 階段對此進行接線。
    pub tick_input_tx: Sender<TickBatchPayload>,
    /// Low-frequency sim_runner profile snapshot for render-frame SLO logs.
    pub diagnostics: Arc<Mutex<SimRunnerDiagnostics>>,
    /// 在「GameStart」到達後發送「master_seed」一次。這
    /// 在初始化世界之前，工作人員會阻止此操作，因此
    /// MasterSeed 資源在第一個tick 運行之前設定。
    pub master_seed_tx: Sender<SimStartMetadata>,
    /// 工作線程連接句柄。持有但未加入；線程退出於
    /// 當“SimRunnerHandle”被刪除時，通道會中斷。
    _thread: thread::JoinHandle<()>,
}

/// 生成模擬器工人。使用初始化規格世界
/// `omoba_core::runtime::create_world_for_scene` 並運行
/// 每個蜱蟲的輸入驅動的共享階段 3 調度程序
/// `tick_input_rx`。
pub fn spawn_sim_runner(base_content_dll_path: PathBuf, scene_path: PathBuf) -> SimRunnerHandle {
    spawn_sim_runner_with_render_extract_rate(
        base_content_dll_path,
        scene_path,
        DEFAULT_EXTRACT_DATA_FOR_RENDER_EVERY_TICKS,
    )
}

pub fn spawn_sim_runner_with_render_extract_rate(
    base_content_dll_path: PathBuf,
    scene_path: PathBuf,
    extract_data_for_render_every_ticks: u32,
) -> SimRunnerHandle {
    let state = Arc::new(Mutex::new(SimWorldSnapshot::default()));
    let state_for_thread = state.clone();
    let diagnostics = Arc::new(Mutex::new(SimRunnerDiagnostics::default()));
    let diagnostics_for_thread = diagnostics.clone();

    let (tick_input_tx, tick_input_rx) = unbounded::<TickBatchPayload>();
    let (master_seed_tx, master_seed_rx) = unbounded::<SimStartMetadata>();

    let handle = thread::Builder::new()
        .name("omfx-sim-runner".into())
        .spawn(move || {
            run_sim_loop(
                state_for_thread,
                diagnostics_for_thread,
                tick_input_rx,
                master_seed_rx,
                base_content_dll_path,
                scene_path,
                normalize_extract_data_for_render_every_ticks(extract_data_for_render_every_ticks),
            );
        })
        .expect("spawn omfx-sim-runner thread");

    SimRunnerHandle {
        state,
        tick_input_tx,
        diagnostics,
        master_seed_tx,
        _thread: handle,
    }
}

fn normalize_extract_data_for_render_every_ticks(value: u32) -> u32 {
    value.max(1)
}

fn should_extract_data_for_render(
    tick: u32,
    every_ticks: u32,
    has_current_player_inputs: bool,
) -> bool {
    has_current_player_inputs
        || tick % normalize_extract_data_for_render_every_ticks(every_ticks) == 0
}

fn run_sim_loop(
    state_out: Arc<Mutex<SimWorldSnapshot>>,
    diagnostics_out: Arc<Mutex<SimRunnerDiagnostics>>,
    tick_input_rx: Receiver<TickBatchPayload>,
    master_seed_rx: Receiver<SimStartMetadata>,
    dll_path: PathBuf,
    scene_path: PathBuf,
    extract_data_for_render_every_ticks: u32,
) {
    info!(
        "sim_runner: thread started; waiting for master_seed (dll={:?}, scene={:?}, extract_data_for_render_every_ticks={})",
        dll_path, scene_path, extract_data_for_render_every_ticks
    );

    // 阻止第一個 master_seed（由 LockstepClient 在
    // 遊戲開始於階段 3.3)。提早返回——沒有滴答作響——
    // 是預期的第 3.2 階段結果，因為 LockstepClient 不
    // 還餵這個頻道。
    let start = match master_seed_rx.recv() {
        Ok(s) => s,
        Err(_) => {
            info!("sim_runner: master_seed channel dropped before GameStart, exiting");
            return;
        }
    };
    let timing = match LockstepTiming::new(start.step_fps) {
        Ok(timing) => timing,
        Err(err) => {
            error!("sim_runner: invalid server cadence: {}", err);
            return;
        }
    };
    let applied_input_retention_ticks = timing.ticks_for_seconds(5);
    info!(
        "sim_runner: got master_seed=0x{:016x} step_fps={}",
        start.master_seed,
        timing.step_fps()
    );

    let script_registry = load_script_registry(&dll_path);

    let (mut world, creep_wave_data) =
        match init_world(&scene_path, start.master_seed, script_registry) {
            Ok(w) => w,
            Err(e) => {
                error!("sim_runner: init_world failed: {}", e);
                return;
            }
        };

    let mut dispatcher = match omoba_core::runtime::build_phase3_dispatcher() {
        Ok(d) => d,
        Err(e) => {
            error!("sim_runner: build_phase3_dispatcher failed: {}", e);
            return;
        }
    };

    // 將 ScriptRegistry 從 ECS 資源中移出，以便我們可以保留 & 借用
    // 跨 `run_script_dispatch(&mut world, ...)` 呼叫每個刻度。奧姆
    // 主機做同樣的事情——它的“State”將“script_registry”保留為一個結構體
    // 字段，不在 ECS 中，正是為了避免借用衝突。更換
    // 具有“Default::default()”（空註冊表）的資源很好，因為
    // 沒有其他任何東西會查詢 ECS 駐留的 ScriptRegistry。
    let script_registry: omoba_core::runtime::ScriptRegistry =
        std::mem::take(&mut *world.write_resource::<omoba_core::runtime::ScriptRegistry>());

    info!("sim_runner: dispatcher ready, entering tick loop");

    let mut last_starvation_log = std::time::Instant::now();
    // Full ECS snapshots are initialization-only. Runtime ticks only extract
    // changing data for render and must not call extract_snapshot.
    let mut abilities_arc: std::sync::Arc<Vec<AbilityDefSnapshot>> =
        std::sync::Arc::new(Vec::new());
    let mut tower_templates_arc: std::sync::Arc<Vec<TowerTemplateSnapshot>> =
        std::sync::Arc::new(Vec::new());
    let mut tower_upgrades_arc: std::sync::Arc<Vec<TowerUpgradeDefSnapshot>> =
        std::sync::Arc::new(Vec::new());
    rebuild_metadata_arcs(
        &world,
        &mut abilities_arc,
        &mut tower_templates_arc,
        &mut tower_upgrades_arc,
    );
    if let Ok(mut snapshot) = state_out.lock() {
        *snapshot = build_initial_render_seed(
            &world,
            abilities_arc.clone(),
            tower_templates_arc.clone(),
            tower_upgrades_arc.clone(),
        );
    }
    let mut recent_applied_inputs: VecDeque<(u32, AppliedInputMeta)> = VecDeque::new();
    let mut recent_tower_fire_fx: VecDeque<TowerFireFx> = VecDeque::new();
    let mut recent_attack_phase_fx: VecDeque<AttackPhaseFx> = VecDeque::new();
    let mut recent_attack_cancel_fx: VecDeque<AttackCancelFx> = VecDeque::new();
    let mut profile_started_at = Instant::now();
    let mut profile_processed_ticks: u32 = 0;
    let mut profile_extract_data_for_render: u32 = 0;
    let mut profile_latest_tick: u32 = 0;
    let mut profile_dispatch_ns: u128 = 0;
    let mut profile_drains_ns: u128 = 0;
    let mut profile_script_ns: u128 = 0;
    let mut profile_extract_ns: u128 = 0;
    let mut profile_publish_ns: u128 = 0;
    let mut profile_receive_active_ns: u128 = 0;
    let mut profile_tick_active_ns: u128 = 0;
    let mut profile_idle_wait_ns: u128 = 0;
    let mut profile_wait_count: u32 = 0;
    let mut profile_blocking_receives: u32 = 0;
    let mut profile_backlog_receives: u32 = 0;
    let mut profile_max_queue_len: usize = 0;
    let mut last_tick_started_at = Instant::now();
    loop {
        let queue_len_before_receive = tick_input_rx.len();
        profile_max_queue_len = profile_max_queue_len.max(queue_len_before_receive);
        let (batch, received_from_backlog) = match tick_input_rx.try_recv() {
            Ok(batch) => (batch, queue_len_before_receive > 0),
            Err(TryRecvError::Empty) => {
                // Blocking wait is cadence/idle time, not active receive work.
                let wait_span = tracing::trace_span!(
                    "omfx::sim_runner::wait_tick_batch",
                    perfetto = true,
                    queue_len = queue_len_before_receive,
                )
                .entered();
                let wait_started = Instant::now();
                let batch = match wait_tick_batch(
                    &tick_input_rx,
                    last_tick_started_at,
                    timing,
                    WAIT_PRECISION_WINDOW,
                    WAIT_STARVATION_TIMEOUT,
                ) {
                    Ok(Some(b)) => b,
                    Ok(None) => {
                        profile_idle_wait_ns += wait_started.elapsed().as_nanos();
                        profile_wait_count += 1;
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
                        drop(wait_span);
                        continue;
                    }
                    Err(WaitTickBatchError::Disconnected) => {
                        info!("sim_runner: input channel closed, exiting");
                        drop(wait_span);
                        break;
                    }
                };
                profile_idle_wait_ns += wait_started.elapsed().as_nanos();
                profile_wait_count += 1;
                profile_blocking_receives += 1;
                drop(wait_span);
                (batch, false)
            }
            Err(TryRecvError::Disconnected) => {
                info!("sim_runner: input channel closed, exiting");
                break;
            }
        };
        if received_from_backlog {
            profile_backlog_receives += 1;
        }
        last_tick_started_at = Instant::now();
        let receive_active_started = Instant::now();
        let receive_span = tracing::trace_span!(
            "omfx::sim_runner::receive_tick_batch",
            perfetto = true,
            tick = batch.tick,
            queue_len_before = queue_len_before_receive,
            queue_len_after = tick_input_rx.len(),
            from_backlog = received_from_backlog,
        )
        .entered();
        let input_count = batch.inputs.len();
        let has_current_player_inputs = batch.inputs.iter().any(|input| input.input_id != 0);
        profile_receive_active_ns += receive_active_started.elapsed().as_nanos();
        drop(receive_span);
        let tick_active_started = Instant::now();
        let tick_span = tracing::trace_span!(
            "omfx::sim_runner::tick",
            perfetto = true,
            tick = batch.tick,
            queue_len = tick_input_rx.len(),
            input_count,
            extract_data_for_render = should_extract_data_for_render(
                batch.tick,
                extract_data_for_render_every_ticks,
                has_current_player_inputs,
            ),
        )
        .entered();

        let lua_reload_span =
            tracing::trace_span!("omfx::sim_runner::dev_lua_reload_check", perfetto = true)
                .entered();
        if let Err(err) = ensure_dev_lua_content_for_batch(
            &mut world,
            &script_registry,
            &creep_wave_data,
            &mut abilities_arc,
            &mut tower_templates_arc,
            &mut tower_upgrades_arc,
            &batch,
        ) {
            error!(
                "sim_runner: DEV Lua reload failed before tick {}: {}",
                batch.tick, err
            );
            publish_dev_lua_reload_error(&state_out, &batch, err);
            break;
        }
        drop(lua_reload_span);

        let input_apply_span = tracing::trace_span!(
            "omfx::sim_runner::input_apply_and_time",
            perfetto = true,
            input_count,
        )
        .entered();
        for input in batch.inputs.iter().filter(|input| input.input_id != 0) {
            recent_applied_inputs.push_back((
                batch.tick,
                AppliedInputMeta {
                    input_id: input.input_id,
                    server_receive_tick: input.server_receive_tick,
                    server_drain_tick: input.server_drain_tick,
                    server_queue_us: input.server_queue_us,
                    client_receive_us: input.client_receive_us,
                    game_forward_us: input.game_forward_us,
                    extract_data_for_render_us: 0,
                },
            ));
        }
        while recent_applied_inputs.front().is_some_and(|(tick, _)| {
            batch.tick.saturating_sub(*tick) > applied_input_retention_ticks
        }) {
            recent_applied_inputs.pop_front();
        }
        let applied_input_meta = recent_applied_inputs
            .iter()
            .map(|(_, meta)| meta.clone())
            .collect::<Vec<_>>();
        let applied_input_ids = applied_input_meta
            .iter()
            .map(|meta| meta.input_id)
            .collect::<Vec<_>>();
        push_inputs_into_world(&mut world, batch.tick, batch.inputs);

        // 更新 Tick + Time + DeltaTime，以便時間閘控系統（creep_wave、
        // 增益計時器、彈丸飛行）實際上是提前的。鎖步 cadence 由
        // server GameStart metadata 宣告。
        // 如果沒有這些，本地 sim 會有 Tick 前進，但時間停留在 0，
        // 這使得 `creep_wave` 看到 `totaltime=0` 並且永遠不會產生 — 完全正確
        // 為什麼 Start Round 會觸發（is_running 翻轉）但沒有小兵出現。
        world
            .write_resource::<omoba_core::comp::resources::Tick>()
            .0 = batch.tick as u64;
        let is_paused = world
            .read_resource::<omoba_core::comp::resources::GamePause>()
            .is_paused;
        if is_paused {
            let mut dt = world.write_resource::<omoba_core::comp::resources::DeltaTime>();
            dt.0 = omoba_sim::Fixed64::ZERO;
        } else {
            let speed = world
                .read_resource::<omoba_core::comp::resources::GameSpeed>()
                .multiplier();
            let mut t = world.write_resource::<omoba_core::comp::resources::Time>();
            t.0 += timing.dt_f64() * f64::from(speed);
            drop(t);
            let mut dt = world.write_resource::<omoba_core::comp::resources::DeltaTime>();
            dt.0 = omoba_sim::Fixed64::from_raw(
                timing
                    .fixed_raw_for_tick(batch.tick as u64)
                    .saturating_mul(i64::from(speed)),
            );
        }
        drop(input_apply_span);

        let t_dispatch = Instant::now();
        let dispatch_span =
            tracing::trace_span!("omfx::sim_runner::dispatcher", perfetto = true).entered();
        dispatcher.dispatch(&world);
        world.maintain();
        profile_dispatch_ns += t_dispatch.elapsed().as_nanos();
        drop(dispatch_span);
        {
            let mut events = world.write_resource::<omoba_core::runtime::RuntimeEvents>();
            events.clear();
        }

        let t_drains = Instant::now();
        let drains_span =
            tracing::trace_span!("omfx::sim_runner::pending_queue_drains", perfetto = true)
                .entered();
        omoba_core::runtime::drain_pending_hero_command_clears(&mut world);
        world.maintain();

        // 階段 2.1：drain `PendingTowerSpawnQueue`，與 authoritative runtime
        // 使用相同 tick boundary，讓 TowerPlace input deterministic 地建立 TD tower。
        omoba_core::runtime::drain_pending_tower_spawns(&mut world);
        world.maintain();

        // 階段 2.2：drain TowerSell input queue。退款與 entity removal 必須在
        // authoritative/local replica 同步執行，讓 snapshots 保持一致。
        omoba_core::runtime::drain_pending_tower_sells(&mut world);
        world.maintain();

        // 階段 2.3：drain TowerUpgrade input queue。扣金、upgrade_levels 增量與
        // BuffStore stat-mod 必須在 authoritative/local replica 同步執行。
        omoba_core::runtime::drain_pending_tower_upgrades(&mut world);
        world.maintain();

        omoba_core::runtime::drain_pending_tower_target_priorities(&mut world);
        world.maintain();

        // 階段 2.4：drain ItemUse input queue。庫存冷卻與 CProperty
        // (HP / msd) mutation 需在 authoritative/local replica 同步執行。
        omoba_core::runtime::drain_pending_item_uses(&mut world);
        world.maintain();

        // AbilityUpgrade：消耗 skill point 並在 script dispatch 前排入 SkillLearn，
        // 與 authoritative runtime 使用相同 boundary。
        omoba_core::runtime::drain_pending_ability_upgrades(&mut world);
        world.maintain();

        // AbilityCast：在 script dispatch 前排入 SkillCast。保留在 upgrades 後面，
        // 讓同 tick learn+cast 行為與 host 相符。
        omoba_core::runtime::drain_pending_ability_casts(&mut world);
        world.maintain();

        // MoveTo (右鍵移動): drain `PendingMoveQueue`，在玩家英雄寫入 MoveTarget。
        omoba_core::runtime::drain_pending_moves(&mut world);
        world.maintain();
        profile_drains_ns += t_drains.elapsed().as_nanos();
        drop(drains_span);

        // 階段 3 調度程序僅調度滴答系統；它不包括
        // GameProcessor::process_outcomes。如果沒有這個，`creep_wave`會產生
        // `Outcome::Creep { cd }` 行堆積在 `Vec<Outcome>` 中，但沒有
        // 實體在本機 sim 中產生 → snapshot.creep 保持 0。
        let pre_script_outcomes_span = tracing::trace_span!(
            "omfx::sim_runner::process_outcomes_pre_script",
            perfetto = true,
        )
        .entered();
        let mut event_sink = omoba_core::runtime::RuntimeEventVecSink::default();
        if let Err(e) = omoba_core::runtime::process_outcomes(&mut world, &mut event_sink) {
            log::warn!("sim_runner: process_outcomes failed: {}", e);
        }
        world.maintain();
        drop(pre_script_outcomes_span);

        let t_script = Instant::now();
        let script_span =
            tracing::trace_span!("omfx::sim_runner::script_dispatch", perfetto = true).entered();
        // 運行腳本調度，以便塔/英雄/召喚`on_tick`鉤子火。
        // 塔是 ScriptUnitTag 驅動的 - 沒有這個， tower_dart / tower_
        // 炸彈/ tower_ice從未決定攻擊，所以projectile_tick有
        // 沒有什麼可以提前的，damage_tick 也沒有什麼可以應用的。
        // backend 的 `State::tick` 在 `run_systems` 之後執行相同的操作（請參閱
        // `scripting::run_script_dispatch` 周圍的 backend tick loop
        // 稱呼）。副本需要相同的呼叫來保持 sim 等效。
        omoba_core::runtime::run_script_dispatch(
            &mut world,
            &script_registry,
            batch.tick as u64,
            omoba_sim::Fixed64::from_raw(timing.fixed_raw_for_tick(batch.tick as u64)),
        );
        // 處理推送的任何結果腳本（投射物/損壞/等）。
        let mut event_sink = omoba_core::runtime::RuntimeEventVecSink::default();
        if let Err(e) = omoba_core::runtime::process_outcomes(&mut world, &mut event_sink) {
            log::warn!("sim_runner: process_outcomes (post-script) failed: {}", e);
        }
        world.maintain();
        profile_script_ns += t_script.elapsed().as_nanos();
        drop(script_span);

        // Metadata is static after init except DEV Lua reload, where the reload
        // path rebuilds these Arcs explicitly.
        let metadata_span =
            tracing::trace_span!("omfx::sim_runner::metadata_refresh", perfetto = true).entered();
        if abilities_arc.is_empty() {
            let reg = world.read_resource::<omoba_core::ability_runtime::AbilityRegistry>();
            if !reg.is_empty() {
                abilities_arc = std::sync::Arc::new(build_ability_def_snapshots(&reg));
                log::info!(
                    "sim_runner: built AbilityRegistry snapshot ({} defs)",
                    abilities_arc.len()
                );
            }
        }

        // TD 塔範本註冊表 — 相同的惰性建置模式。人口由
        // 每個塔腳本在腳本載入時的「tower_metadata()」。
        if tower_templates_arc.is_empty() {
            let reg =
                world.read_resource::<omoba_core::comp::tower_registry::TowerTemplateRegistry>();
            if !reg.is_empty() {
                tower_templates_arc = std::sync::Arc::new(build_tower_template_snapshots(&reg));
                log::info!(
                    "sim_runner: built TowerTemplateRegistry snapshot ({} templates)",
                    tower_templates_arc.len()
                );
            }
        }

        // TowerUpgradeRegistry — 在世界初始化時建構一次（不像非同步
        // 塔模板），因此 iter_all 從勾選 1 開始就非空。惰性保護
        // 鏡像其他註冊表以實現對稱。
        if tower_upgrades_arc.is_empty() {
            let reg = world
                .read_resource::<omoba_core::comp::tower_upgrade_registry::TowerUpgradeRegistry>();
            let defs = build_tower_upgrade_def_snapshots(&reg);
            if !defs.is_empty() {
                tower_upgrades_arc = std::sync::Arc::new(defs);
                log::info!(
                    "sim_runner: built TowerUpgradeRegistry snapshot ({} defs)",
                    tower_upgrades_arc.len()
                );
            }
        }
        drop(metadata_span);

        let should_extract_data_for_render = should_extract_data_for_render(
            batch.tick,
            extract_data_for_render_every_ticks,
            has_current_player_inputs,
        );
        if should_extract_data_for_render {
            let t_extract = Instant::now();
            let extract_span = tracing::trace_span!(
                "omfx::sim_runner::extract_data_for_render",
                perfetto = true,
                tick = batch.tick,
            )
            .entered();
            let mut snapshot = extract_data_for_render(
                &mut world,
                batch.tick,
                applied_input_ids,
                applied_input_meta,
            );
            profile_extract_ns += t_extract.elapsed().as_nanos();
            drop(extract_span);
            let fx_span =
                tracing::trace_span!("omfx::sim_runner::render_fx_retention", perfetto = true)
                    .entered();
            let tower_fire_fx = std::mem::take(&mut snapshot.tower_fire_fx);
            snapshot.tower_fire_fx = retain_recent_render_fx(
                &mut recent_tower_fire_fx,
                tower_fire_fx,
                batch.tick,
                |fx| fx.spawn_tick,
            );
            let attack_phase_fx = std::mem::take(&mut snapshot.attack_phase_fx);
            snapshot.attack_phase_fx = retain_recent_render_fx(
                &mut recent_attack_phase_fx,
                attack_phase_fx,
                batch.tick,
                |fx| fx.spawn_tick,
            );
            let attack_cancel_fx = std::mem::take(&mut snapshot.attack_cancel_fx);
            snapshot.attack_cancel_fx = retain_recent_render_fx(
                &mut recent_attack_cancel_fx,
                attack_cancel_fx,
                batch.tick,
                |fx| fx.spawn_tick,
            );
            let extract_data_for_render_us = wall_clock_us();
            for meta in &mut snapshot.applied_input_meta {
                meta.extract_data_for_render_us = extract_data_for_render_us;
            }
            drop(fx_span);

            let t_publish = Instant::now();
            let publish_span = tracing::trace_span!(
                "omfx::sim_runner::render_data_publish",
                perfetto = true,
                snapshot_entities = snapshot.entities.len(),
            )
            .entered();
            if let Ok(mut s) = state_out.lock() {
                snapshot.paths = s.paths.clone();
                snapshot.blocked_regions = s.blocked_regions.clone();
                snapshot.abilities = abilities_arc.clone();
                snapshot.tower_templates = tower_templates_arc.clone();
                snapshot.tower_upgrades = tower_upgrades_arc.clone();
                *s = snapshot;
            }
            profile_publish_ns += t_publish.elapsed().as_nanos();
            drop(publish_span);
            profile_extract_data_for_render += 1;
        }
        profile_processed_ticks += 1;
        profile_latest_tick = batch.tick;
        profile_tick_active_ns += tick_active_started.elapsed().as_nanos();
        let profile_elapsed = profile_started_at.elapsed();
        if profile_elapsed.as_secs_f32() >= 1.0 {
            let ticks = profile_processed_ticks.max(1) as f64;
            if let Ok(mut diagnostics) = diagnostics_out.lock() {
                diagnostics.window_ms = profile_elapsed.as_millis();
                diagnostics.sim_tps =
                    profile_processed_ticks as f32 / profile_elapsed.as_secs_f32();
                diagnostics.latest_tick = profile_latest_tick;
                diagnostics.queue_len = tick_input_rx.len();
                diagnostics.max_queue_len = profile_max_queue_len;
                diagnostics.waits = profile_wait_count;
                diagnostics.blocking_receives = profile_blocking_receives;
                diagnostics.backlog_receives = profile_backlog_receives;
            }
            info!(
                "sim_runner_profile window_ms={} target_tps={} processed_ticks={} extract_data_for_render={} latest_tick={} queue_len={} max_queue_len={} waits={} blocking_receives={} backlog_receives={} avg_ms wait_idle={:.3} receive_active={:.3} tick_active={:.3} dispatch={:.3} drains={:.3} script={:.3} extract={:.3} publish={:.3}",
                profile_elapsed.as_millis(),
                timing.step_fps(),
                profile_processed_ticks,
                profile_extract_data_for_render,
                profile_latest_tick,
                tick_input_rx.len(),
                profile_max_queue_len,
                profile_wait_count,
                profile_blocking_receives,
                profile_backlog_receives,
                profile_idle_wait_ns as f64 / profile_wait_count.max(1) as f64 / 1_000_000.0,
                profile_receive_active_ns as f64 / ticks / 1_000_000.0,
                profile_tick_active_ns as f64 / ticks / 1_000_000.0,
                profile_dispatch_ns as f64 / ticks / 1_000_000.0,
                profile_drains_ns as f64 / ticks / 1_000_000.0,
                profile_script_ns as f64 / ticks / 1_000_000.0,
                profile_extract_ns as f64 / ticks / 1_000_000.0,
                profile_publish_ns as f64 / ticks / 1_000_000.0,
            );
            profile_started_at = Instant::now();
            profile_processed_ticks = 0;
            profile_extract_data_for_render = 0;
            profile_dispatch_ns = 0;
            profile_drains_ns = 0;
            profile_script_ns = 0;
            profile_extract_ns = 0;
            profile_publish_ns = 0;
            profile_receive_active_ns = 0;
            profile_tick_active_ns = 0;
            profile_idle_wait_ns = 0;
            profile_wait_count = 0;
            profile_blocking_receives = 0;
            profile_backlog_receives = 0;
            profile_max_queue_len = 0;
        }
        drop(tick_span);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WaitTickBatchError {
    Disconnected,
}

fn wait_tick_batch(
    rx: &Receiver<TickBatchPayload>,
    last_tick_started_at: Instant,
    timing: LockstepTiming,
    precision_window: Duration,
    starvation_timeout: Duration,
) -> Result<Option<TickBatchPayload>, WaitTickBatchError> {
    let wait_started = Instant::now();
    loop {
        match rx.try_recv() {
            Ok(batch) => return Ok(Some(batch)),
            Err(TryRecvError::Disconnected) => return Err(WaitTickBatchError::Disconnected),
            Err(TryRecvError::Empty) => {}
        }

        if wait_started.elapsed() >= starvation_timeout {
            return Ok(None);
        }

        let now = Instant::now();
        let plan = plan_tick_wait(last_tick_started_at, timing, precision_window, now);
        if let Some(sleep_duration) = plan.sleep {
            let timeout =
                sleep_duration.min(starvation_timeout.saturating_sub(wait_started.elapsed()));
            match rx.recv_timeout(timeout) {
                Ok(batch) => return Ok(Some(batch)),
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                    return Err(WaitTickBatchError::Disconnected);
                }
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
            }
        }

        if plan.remaining.is_zero() {
            let timeout =
                precision_window.min(starvation_timeout.saturating_sub(wait_started.elapsed()));
            match rx.recv_timeout(timeout) {
                Ok(batch) => return Ok(Some(batch)),
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                    return Err(WaitTickBatchError::Disconnected);
                }
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
            }
        }

        let deadline = last_tick_started_at + plan.tick_interval;
        while Instant::now() < deadline && wait_started.elapsed() < starvation_timeout {
            match rx.try_recv() {
                Ok(batch) => return Ok(Some(batch)),
                Err(TryRecvError::Disconnected) => return Err(WaitTickBatchError::Disconnected),
                Err(TryRecvError::Empty) => thread::yield_now(),
            }
        }
    }
}

fn build_initial_render_seed(
    world: &World,
    abilities_arc: std::sync::Arc<Vec<AbilityDefSnapshot>>,
    tower_templates_arc: std::sync::Arc<Vec<TowerTemplateSnapshot>>,
    tower_upgrades_arc: std::sync::Arc<Vec<TowerUpgradeDefSnapshot>>,
) -> SimWorldSnapshot {
    let paths: Vec<Vec<(f32, f32)>> = world
        .read_resource::<BTreeMap<String, omoba_core::runtime::comp::check_point::Path>>()
        .values()
        .map(|p| {
            p.check_points
                .iter()
                .map(|cp| (cp.pos.x, cp.pos.y))
                .collect()
        })
        .collect();

    let blocked_regions: Vec<BlockedRegionSnapshot> = world
        .read_resource::<omoba_core::runtime::BlockedRegions>()
        .0
        .iter()
        .map(|r| BlockedRegionSnapshot {
            points: r.points.iter().map(|p| (p.x, p.y)).collect(),
            circle: None,
        })
        .collect();

    let total_rounds = world
        .read_resource::<Vec<omoba_core::runtime::CreepWave>>()
        .len() as u32;
    let lives = world.read_resource::<omoba_core::runtime::PlayerLives>().0;

    SimWorldSnapshot {
        paths,
        blocked_regions,
        abilities: abilities_arc,
        tower_templates: tower_templates_arc,
        tower_upgrades: tower_upgrades_arc,
        total_rounds,
        lives,
        lua_content_generation: omoba_template_ids::runtime_lua_content_generation()
            .ok()
            .flatten()
            .unwrap_or(0),
        lua_content_hash: omoba_template_ids::runtime_lua_content_hash()
            .ok()
            .flatten()
            .unwrap_or_default(),
        ..Default::default()
    }
}

fn load_script_registry(dll_path: &Path) -> omoba_core::runtime::ScriptRegistry {
    let script_dir = dll_path.parent().unwrap_or_else(|| Path::new("."));
    info!("sim_runner: loading scripts from {:?}", script_dir);
    omoba_core::scripting::loader::load_scripts_dir(script_dir)
}

fn init_world(
    scene_path: &Path,
    master_seed: u64,
    script_registry: omoba_core::runtime::ScriptRegistry,
) -> Result<(World, omoba_core::ue4::import_map::CreepWaveData), failure::Error> {
    use failure::err_msg;

    let story_id = scene_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| err_msg("scene_path does not end in a valid story id"))?;
    let campaign_data =
        omoba_core::ue4::import_campaign::load_generated(story_id).map_err(|e| {
            err_msg(format!(
                "CampaignData::load_generated({}) failed: {}",
                story_id, e
            ))
        })?;
    let creep_wave_data = campaign_data.map.clone();
    let mut world = omoba_core::runtime::create_world_from_loaded_content(
        campaign_data,
        omoba_core::runtime::ItemRegistry::default(),
        script_registry,
    )?;
    // 使用權威的 MasterSeed 覆蓋預設的 MasterSeed
    // 遊戲開始。必須在第一次調度之前發生。
    world
        .write_resource::<omoba_core::comp::resources::MasterSeed>()
        .0 = master_seed;
    Ok((world, creep_wave_data))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DevLuaReloadAction {
    Noop,
    Reload,
}

fn dev_lua_reload_action(
    current_generation: Option<u64>,
    current_hash: Option<&str>,
    target_generation: u64,
    target_hash: &str,
) -> Result<DevLuaReloadAction, String> {
    if target_generation == 0 && target_hash.is_empty() {
        return Ok(DevLuaReloadAction::Noop);
    }
    if target_hash.is_empty() {
        return Err(format!(
            "target DEV Lua generation {} is missing content hash",
            target_generation
        ));
    }
    match (current_generation, current_hash) {
        (Some(generation), Some(hash))
            if generation == target_generation && hash == target_hash =>
        {
            Ok(DevLuaReloadAction::Noop)
        }
        (Some(generation), Some(hash)) if generation == target_generation => Err(format!(
            "runtime Lua content hash mismatch at generation {}: expected {}, got {}",
            target_generation, target_hash, hash
        )),
        (Some(generation), _) if generation > target_generation => Err(format!(
            "local runtime Lua generation {} is ahead of backend target {}",
            generation, target_generation
        )),
        _ => Ok(DevLuaReloadAction::Reload),
    }
}

fn ensure_dev_lua_content_for_batch(
    world: &mut World,
    script_registry: &omoba_core::runtime::ScriptRegistry,
    creep_wave_data: &omoba_core::ue4::import_map::CreepWaveData,
    abilities_arc: &mut std::sync::Arc<Vec<AbilityDefSnapshot>>,
    tower_templates_arc: &mut std::sync::Arc<Vec<TowerTemplateSnapshot>>,
    tower_upgrades_arc: &mut std::sync::Arc<Vec<TowerUpgradeDefSnapshot>>,
    batch: &TickBatchPayload,
) -> Result<(), String> {
    let current_generation = omoba_template_ids::runtime_lua_content_generation()?;
    let current_hash = omoba_template_ids::runtime_lua_content_hash()?;
    let action = dev_lua_reload_action(
        current_generation,
        current_hash.as_deref(),
        batch.lua_content_generation,
        batch.lua_content_hash.as_str(),
    )?;
    if action == DevLuaReloadAction::Noop {
        return Ok(());
    }

    let expected_hash = batch.lua_content_hash.as_str();
    let modules = script_registry.reload_runtime_lua_content_dev(expected_hash)?;
    let (committed_generation, committed_hash) = commit_runtime_lua_content_reload(expected_hash)?;
    if committed_generation != batch.lua_content_generation || committed_hash != expected_hash {
        return Err(format!(
            "runtime Lua reload committed generation/hash mismatch: backend target {} {}, local committed {} {}",
            batch.lua_content_generation, expected_hash, committed_generation, committed_hash
        ));
    }

    omoba_core::runtime::StateInitializer::refresh_dev_lua_gameplay_content(
        world,
        creep_wave_data,
        script_registry,
    );
    rebuild_metadata_arcs(
        world,
        abilities_arc,
        tower_templates_arc,
        tower_upgrades_arc,
    );
    info!(
        "sim_runner: DEV Lua content reloaded generation={} hash={} script_modules={}",
        committed_generation,
        committed_hash,
        modules.len()
    );
    Ok(())
}

fn commit_runtime_lua_content_reload(expected_hash: &str) -> Result<(u64, String), String> {
    #[cfg(feature = "runtime-lua-content")]
    {
        let committed = omoba_template_ids::reload_runtime_lua_content_dev(Some(expected_hash))?
            .ok_or_else(|| "runtime Lua content became inactive during omfx reload".to_string())?;
        return Ok((committed.generation, committed.hash));
    }
    #[cfg(not(feature = "runtime-lua-content"))]
    {
        let _ = expected_hash;
        Err("omfx was built without runtime-lua-content; cannot apply DEV Lua reload".into())
    }
}

fn rebuild_metadata_arcs(
    world: &World,
    abilities_arc: &mut std::sync::Arc<Vec<AbilityDefSnapshot>>,
    tower_templates_arc: &mut std::sync::Arc<Vec<TowerTemplateSnapshot>>,
    tower_upgrades_arc: &mut std::sync::Arc<Vec<TowerUpgradeDefSnapshot>>,
) {
    let ability_reg = world.read_resource::<omoba_core::ability_runtime::AbilityRegistry>();
    *abilities_arc = std::sync::Arc::new(build_ability_def_snapshots(&ability_reg));
    let tower_reg =
        world.read_resource::<omoba_core::comp::tower_registry::TowerTemplateRegistry>();
    *tower_templates_arc = std::sync::Arc::new(build_tower_template_snapshots(&tower_reg));
    let upgrade_reg =
        world.read_resource::<omoba_core::comp::tower_upgrade_registry::TowerUpgradeRegistry>();
    *tower_upgrades_arc = std::sync::Arc::new(build_tower_upgrade_def_snapshots(&upgrade_reg));
}

fn publish_dev_lua_reload_error(
    state_out: &Arc<Mutex<SimWorldSnapshot>>,
    batch: &TickBatchPayload,
    err: String,
) {
    if let Ok(mut snapshot) = state_out.lock() {
        snapshot.tick = batch.tick;
        snapshot.dev_lua_reload_error = Some(err);
    }
}

fn push_inputs_into_world(world: &mut World, tick: u32, inputs: Vec<TickBatchInput>) {
    // 階段 3.4：將鎖步 TickBatch 輸入寫入 shared runtime 的
    // `PendingPlayerInputs` 資源，所以 `tick::player_input_tick::Sys`
    // 可以在調度程序運行開始時耗盡它們。
    //
    // Replace the per-tick input list wholesale. Multiple commands from one
    // player can share a lockstep tick, so keep them in TickBatch order.
    use omoba_core::comp::PendingPlayerInputs;

    let mut pending = world.write_resource::<PendingPlayerInputs>();
    pending.tick = tick;
    pending.inputs.clear();
    if !inputs.is_empty() {
        log::trace!("sim_runner: tick {} got {} inputs", tick, inputs.len());
    }
    for input in inputs {
        pending.inputs.push((input.player_id, input.input));
    }
}

/// 冒煙測試表明 shared runtime 是可以訪問的。驗證 dep 接線
/// 有效且階段 3.2 輔助符號解析。
pub fn smoke() -> &'static str {
    let _ = omoba_core::comp::resources::MasterSeed::default();
    // 進入第 3.2 階段新增的新酒吧助手進行確認
    // 它們可以從 omfx 中看到。
    let _ = omoba_core::runtime::build_phase3_dispatcher
        as fn() -> Result<specs::Dispatcher<'static, 'static>, failure::Error>;
    "omoba-core runtime linked"
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn smoke_links() {
        assert_eq!(smoke(), "omoba-core runtime linked");
    }

    #[test]
    fn wait_plan_keeps_precision_window_for_supported_fps() {
        let now = Instant::now();
        let precision = Duration::from_millis(2);
        for (fps, expected_interval_us) in [(120, 8_333), (90, 11_111), (60, 16_666)] {
            let timing = LockstepTiming::new(fps).unwrap();
            let plan = plan_tick_wait(now, timing, precision, now);
            assert!(
                plan.tick_interval
                    .as_micros()
                    .abs_diff(expected_interval_us)
                    <= 1,
                "fps={fps} interval={:?}",
                plan.tick_interval
            );
            let sleep = plan.sleep.expect("sleep for full frame budget");
            assert!(
                sleep <= plan.tick_interval - precision,
                "fps={fps} sleep={sleep:?} interval={:?}",
                plan.tick_interval
            );
        }
    }

    #[test]
    fn wait_plan_uses_yield_path_inside_precision_window() {
        let now = Instant::now();
        let timing = LockstepTiming::new(120).unwrap();
        let precision = Duration::from_millis(2);
        let consumed = timing.dt_duration() - Duration::from_millis(1);
        let plan = plan_tick_wait(now, timing, precision, now + consumed);

        assert!(plan.remaining <= precision);
        assert_eq!(plan.sleep, None);
    }

    #[test]
    fn wait_tick_batch_returns_ready_payload_without_pacing_sleep() {
        let (tx, rx) = unbounded();
        tx.send(TickBatchPayload {
            tick: 7,
            inputs: Vec::new(),
            lua_content_generation: 0,
            lua_content_hash: String::new(),
        })
        .unwrap();
        let started = Instant::now();
        let batch = wait_tick_batch(
            &rx,
            started,
            LockstepTiming::new(60).unwrap(),
            Duration::from_millis(2),
            Duration::from_millis(50),
        )
        .unwrap()
        .expect("payload");

        assert_eq!(batch.tick, 7);
        assert!(started.elapsed() < Duration::from_millis(10));
    }

    #[test]
    fn extract_data_for_render_default_is_every_tick() {
        assert!(should_extract_data_for_render(
            1,
            DEFAULT_EXTRACT_DATA_FOR_RENDER_EVERY_TICKS,
            false
        ));
        assert!(should_extract_data_for_render(
            2,
            DEFAULT_EXTRACT_DATA_FOR_RENDER_EVERY_TICKS,
            false
        ));
    }

    #[test]
    fn extract_data_for_render_periodic_setting_keeps_input_ticks() {
        assert!(!should_extract_data_for_render(5, 4, false));
        assert!(should_extract_data_for_render(8, 4, false));
        assert!(should_extract_data_for_render(5, 4, true));
    }

    #[test]
    fn dev_lua_reload_action_detects_matching_generation() {
        assert_eq!(
            dev_lua_reload_action(Some(2), Some("abc"), 2, "abc").unwrap(),
            DevLuaReloadAction::Noop
        );
    }

    #[test]
    fn dev_lua_reload_action_requests_reload_for_new_generation() {
        assert_eq!(
            dev_lua_reload_action(Some(1), Some("old"), 2, "new").unwrap(),
            DevLuaReloadAction::Reload
        );
    }

    #[test]
    fn dev_lua_reload_action_rejects_hash_mismatch() {
        let err = dev_lua_reload_action(Some(2), Some("local"), 2, "backend").unwrap_err();
        assert!(err.contains("hash mismatch"));
    }

    #[test]
    fn snapshot_default() {
        let s = SimWorldSnapshot::default();
        assert_eq!(s.tick, 0);
        assert!(s.entities.is_empty());
    }

    #[test]
    fn tower_template_snapshot_carries_render_metadata_in_shared_arc() {
        let templates = Arc::new(vec![TowerTemplateSnapshot {
            unit_id: "tower_dart".to_string(),
            label: "Dart".to_string(),
            cost: 200,
            footprint: 10.0,
            placement_radius: 90.0,
            range: 350.0,
            splash_radius: 0.0,
            hit_radius: 0.0,
            slow_factor: 0.0,
            slow_duration: 0.0,
            render_mode: "base_barrel".to_string(),
            base_image: "assets/towers/tower_dart_base.png".to_string(),
            barrel_image: "assets/towers/tower_dart_barrel.png".to_string(),
            render_visual_size: 180.0,
            barrel_frames: vec!["assets/towers/tower_dart_barrel_frame_01.png".to_string()],
            body_frames: Vec::new(),
            barrel_animation: TowerRenderAnimationSnapshot {
                fps: 12.0,
                loop_animation: true,
                fire_fps: 22.0,
                fire_once: true,
            },
            body_animation: TowerRenderAnimationSnapshot::default(),
            rotation_mode: "targeted".to_string(),
            barrel_layout: "single".to_string(),
            barrel_variants: Vec::new(),
            barrel_offset: TowerRenderPointSnapshot { x: 0.0, y: -6.0 },
            barrel_pivot: TowerRenderPointSnapshot { x: 0.5, y: 0.66 },
            muzzle_offset: TowerRenderPointSnapshot { x: 0.0, y: -30.0 },
            default_angle_deg: 0.0,
            recoil: TowerRecoilSnapshot {
                mode: "directional".to_string(),
                distance: 6.0,
                scale: 0.95,
                duration_ms: 60,
                return_ms: 95,
            },
            attack_windup: 350,
            attack_backswing: 650,
        }]);
        let first = SimWorldSnapshot {
            tower_templates: templates.clone(),
            ..Default::default()
        };
        let second = SimWorldSnapshot {
            tower_templates: templates.clone(),
            ..Default::default()
        };

        assert!(Arc::ptr_eq(&first.tower_templates, &second.tower_templates));
        let dart = &first.tower_templates[0];
        assert_eq!(dart.render_mode, "base_barrel");
        assert_eq!(dart.base_image, "assets/towers/tower_dart_base.png");
        assert_eq!(dart.barrel_image, "assets/towers/tower_dart_barrel.png");
        assert_eq!(dart.rotation_mode, "targeted");
        assert_eq!(dart.recoil.mode, "directional");
    }

    #[test]
    fn saika_hero_render_snapshot_uses_generated_metadata() {
        let render = hero_render_snapshot_for_unit_id("hero_saika_magoichi", true, true)
            .expect("saika render snapshot");
        let generated = omoba_template_ids::active_hero_render_metadata(
            omoba_template_ids::HERO_SAIKA_MAGOICHI,
        )
        .expect("generated metadata");

        assert_eq!(render.render_mode, "model_3d");
        assert_eq!(render.model, generated.model);
        assert_eq!(render.texture, generated.texture);
        assert_eq!(render.scale, generated.scale.to_f32_for_render());
        assert!(render.is_moving);
        assert!(render.sniper_mode);

        for required in [
            "idle", "idle_2", "idle_3", "move", "attack", "critical", "sniper",
        ] {
            assert!(render
                .animation_sources
                .iter()
                .any(|source| source.key == required));
            assert!(render
                .animations
                .iter()
                .any(|binding| binding.action == required));
        }

        let attack_source = render
            .animation_sources
            .iter()
            .find(|source| source.key == "attack")
            .expect("attack source");
        let generated_attack_source = generated
            .animation_sources
            .iter()
            .find(|source| source.key == "attack")
            .expect("generated attack source");
        assert_eq!(attack_source.model, generated_attack_source.model);
        assert_eq!(attack_source.animation, generated_attack_source.animation);
        assert_eq!(
            attack_source.duration_ticks,
            generated_attack_source.duration_ticks.to_f32_for_render()
        );
        assert_eq!(
            attack_source.ticks_per_second,
            generated_attack_source.ticks_per_second.to_f32_for_render()
        );
        assert_eq!(
            attack_source.timeline_offset_ticks,
            generated_attack_source
                .timeline_offset_ticks
                .to_f32_for_render()
        );

        let critical = render
            .animations
            .iter()
            .find(|binding| binding.action == "critical")
            .expect("critical binding");
        let generated_critical = generated
            .animations
            .iter()
            .find(|binding| binding.action == "critical")
            .expect("generated critical binding");
        assert_eq!(critical.source, "critical");
        assert_eq!(
            critical.impact_tick,
            Some(generated_critical.impact_tick.to_f32_for_render())
        );
        assert_eq!(
            critical.repeat_start_tick,
            generated_critical.repeat_start_tick.to_f32_for_render()
        );
        assert!(!critical.loop_animation);

        let sniper = render
            .animations
            .iter()
            .find(|binding| binding.action == "sniper")
            .expect("sniper binding");
        assert!(sniper.loop_animation);
    }

    #[test]
    fn hero_without_render_metadata_has_no_render_snapshot() {
        assert!(hero_render_snapshot_for_unit_id("hero_date_masamune", false, false).is_none());
        assert!(hero_render_snapshot_for_unit_id("tower_dart", false, false).is_none());
    }

    #[test]
    fn snapshot_carries_attack_phase_and_cancel_cues() {
        let snapshot = SimWorldSnapshot {
            attack_phase_fx: vec![AttackPhaseFx {
                entity_id: 7,
                entity_gen: 1,
                spawn_tick: 42,
                attack_seq: 3,
                is_critical: true,
                windup_ms: 120,
                impact_at_ms: 120,
                backswing_ms: 240,
                dir_rad: 0.0,
                target_entity_id: Some(9),
                target_pos_x: None,
                target_pos_y: None,
            }],
            attack_cancel_fx: vec![AttackCancelFx {
                entity_id: 7,
                entity_gen: 1,
                spawn_tick: 43,
                attack_seq: 3,
                phase: omoba_core::comp::AttackCancelPhase::Windup,
                impact_committed: false,
            }],
            ..Default::default()
        };

        assert_eq!(snapshot.attack_phase_fx[0].attack_seq, 3);
        assert!(snapshot.attack_phase_fx[0].is_critical);
        assert_eq!(snapshot.attack_cancel_fx[0].attack_seq, 3);
        assert_eq!(
            snapshot.attack_cancel_fx[0].phase,
            omoba_core::comp::AttackCancelPhase::Windup
        );
        assert!(!snapshot.attack_cancel_fx[0].impact_committed);
    }

    #[test]
    fn entity_kind_eq() {
        assert_eq!(EntityKind::Hero, EntityKind::Hero);
        assert_ne!(EntityKind::Hero, EntityKind::Tower);
    }

    #[test]
    fn permanent_buff_remaining_maps_to_infinity_sentinel() {
        assert_eq!(
            buff_remaining_secs_for_snapshot(omoba_sim::Fixed64::from_raw(i64::MAX)),
            -1.0
        );
        assert_eq!(
            buff_remaining_secs_for_snapshot(omoba_sim::Fixed64::from_raw(i32::MAX as i64 - 1024)),
            -1.0
        );
    }

    #[test]
    fn finite_buff_remaining_still_projects_seconds() {
        assert_eq!(
            buff_remaining_secs_for_snapshot(omoba_sim::Fixed64::from_raw(5 * 1024)),
            5.0
        );
    }
}
