//! 第 3 階段 omfx 模擬器運行程式。
//!
//! 產生一個工作線程，運行由以下驅動的完整 omb ECS 調度程序
//! 來自 omb 鎖步線的 TickBatch 輸入。渲染線程讀取
//! 發布了 `SimWorldSnapshot` Arc<Mutex<...>>。
//!
//! 階段 3.1 = 存根。階段 3.2 = 現實世界 init + 調度程式循環。階段
//! 3.3 將連接 `LockstepClient` → 通道饋線。 3.4相線
//! 將快照放入渲染端並替換 TickBroadcaster 的
//! 佔位符狀態雜湊以及源自此迴圈的真實 ECS 雜湊。

#![allow(dead_code)]

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use crossbeam_channel::{unbounded, Receiver, Sender};
use log::{error, info};
use omoba_core::lockstep_timing::{
    lockstep_dt_fixed_raw_for_tick, ticks_to_seconds_f64, LOCKSTEP_FIVE_SECONDS_TICKS_U32,
    LOCKSTEP_ONE_SECOND_TICKS_U32,
};

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
    buff_remaining_secs_for_snapshot, build_ability_def_snapshots,
    build_tower_template_snapshots, build_tower_upgrade_def_snapshots, extract_snapshot,
    hero_render_snapshot_for_unit_id, retain_recent_render_fx, AbilityDefSnapshot,
    AppliedInputMeta, AttackCancelFx, AttackPhaseFx, BlockedRegionSnapshot, BuffSnapshot,
    EntityKind, EntityRenderData, ExplosionFx, HeroRenderSnapshot, HeroStatsExt,
    HeroAnimationBindingSnapshot, HeroAnimationSourceSnapshot, SimWorldSnapshot,
    TowerBarrelVariantSnapshot, TowerFireFx, TowerRecoilSnapshot,
    TowerRenderAnimationSnapshot, TowerRenderPointSnapshot, TowerTemplateSnapshot,
    TowerUpgradeDefSnapshot,
};

const APPLIED_INPUT_ID_RETENTION_TICKS: u32 = LOCKSTEP_FIVE_SECONDS_TICKS_U32;
/// 每個時脈週期由鎖步饋送器提交的通道有效負載。
#[derive(Clone, Debug)]
pub struct TickBatchPayload {
    pub tick: u32,
    pub inputs: Vec<TickBatchInput>,
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
    /// 在「GameStart」到達後發送「master_seed」一次。這
    /// 在初始化世界之前，工作人員會阻止此操作，因此
    /// MasterSeed 資源在第一個tick 運行之前設定。
    pub master_seed_tx: Sender<u64>,
    /// 工作線程連接句柄。持有但未加入；線程退出於
    /// 當“SimRunnerHandle”被刪除時，通道會中斷。
    _thread: thread::JoinHandle<()>,
}

/// 生成模擬器工人。使用初始化規格世界
/// `omoba_core::runtime::create_world_for_scene` 並運行
/// 每個蜱蟲的輸入驅動的共享階段 3 調度程序
/// `tick_input_rx`。
pub fn spawn_sim_runner(base_content_dll_path: PathBuf, scene_path: PathBuf) -> SimRunnerHandle {
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

    // 阻止第一個 master_seed（由 LockstepClient 在
    // 遊戲開始於階段 3.3)。提早返回——沒有滴答作響——
    // 是預期的第 3.2 階段結果，因為 LockstepClient 不
    // 還餵這個頻道。
    let master_seed = match master_seed_rx.recv() {
        Ok(s) => s,
        Err(_) => {
            info!("sim_runner: master_seed channel dropped before GameStart, exiting");
            return;
        }
    };
    info!("sim_runner: got master_seed=0x{:016x}", master_seed);

    // 將 omb 的腳本載入器指向包含 DLL 的目錄。
    // `load_scripts_dir` 讀取 `OMB_SCRIPTS_DIR` 環境變數；榮譽來電者
    // 覆蓋但以其他方式從 DLL 路徑的父級推斷。
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
    // Phase 1b: removed_entity_ids 從 RemovedEntitiesQueue resource drain
    // 取代既有 prev_alive HashSet diff。helper `delete_entity_tracked` 統
    // 一往 queue 推入；`extract_snapshot` 用 `mem::take` 把整批拉到
    // snapshot，render 端對該 list 釋放 per-eid scene caches。

    // 階段 4.5：AbilityRegistry→AbilityDefSnapshot Arc。懶惰地建構於
    // 註冊表非空的第一個勾號（腳本 DLL 載入為
    // 非同步 — 註冊表由 `scripting::registry::load` 填充
    // 在世界初始化期間，但我們重新輪詢每個刻度，直到設定 Arc
    // 因為在某些場景中，註冊表可能會保持為空，直到英雄出現為止
    // 腳本註冊能力）。建置後，每個快照都只是克隆
    // Arc（O(1) 引用計數凸點）。
    let mut abilities_arc: std::sync::Arc<Vec<AbilityDefSnapshot>> =
        std::sync::Arc::new(Vec::new());
    // TD 塔範本具有相同的延遲建置模式 — 註冊表已填充
    // 在遊戲開始時，每個塔腳本的「tower_metadata()」。
    let mut tower_templates_arc: std::sync::Arc<Vec<TowerTemplateSnapshot>> =
        std::sync::Arc::new(Vec::new());
    let mut tower_upgrades_arc: std::sync::Arc<Vec<TowerUpgradeDefSnapshot>> =
        std::sync::Arc::new(Vec::new());
    let mut recent_applied_inputs: VecDeque<(u32, AppliedInputMeta)> = VecDeque::new();
    let mut recent_tower_fire_fx: VecDeque<TowerFireFx> = VecDeque::new();
    let mut recent_attack_phase_fx: VecDeque<AttackPhaseFx> = VecDeque::new();
    let mut recent_attack_cancel_fx: VecDeque<AttackCancelFx> = VecDeque::new();
    loop {
        // 使用recv_timeout而不是recv()，因此線路停頓會出現在
        // 記錄為「1.0 秒內沒有 TickBatch — 上游鎖步用戶端是
        // 懷疑」而不是看起來像 sim_runner 正在緩慢計算。
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
                    sim_publish_us: 0,
                },
            ));
        }
        while recent_applied_inputs.front().is_some_and(|(tick, _)| {
            batch.tick.saturating_sub(*tick) > APPLIED_INPUT_ID_RETENTION_TICKS
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
        // `omoba_core::lockstep_timing::LOCKSTEP_TPS` 定義。
        // 如果沒有這些，本地 sim 會有 Tick 前進，但時間停留在 0，
        // 這使得 `creep_wave` 看到 `totaltime=0` 並且永遠不會產生 — 完全正確
        // 為什麼 Start Round 會觸發（is_running 翻轉）但沒有小兵出現。
        world.write_resource::<omoba_core::comp::resources::Tick>().0 = batch.tick as u64;
        {
            let mut t = world.write_resource::<omoba_core::comp::resources::Time>();
            t.0 = ticks_to_seconds_f64(batch.tick);
        }
        {
            let mut dt = world.write_resource::<omoba_core::comp::resources::DeltaTime>();
            dt.0 = omoba_sim::Fixed64::from_raw(lockstep_dt_fixed_raw_for_tick(batch.tick as u64));
        }

        dispatcher.dispatch(&world);
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

        // 階段 3 調度程序僅調度滴答系統；它不包括
        // GameProcessor::process_outcomes。如果沒有這個，`creep_wave`會產生
        // `Outcome::Creep { cd }` 行堆積在 `Vec<Outcome>` 中，但沒有
        // 實體在本機 sim 中產生 → snapshot.creep 保持 0。
        // mqtx 是一個接收器（空 Vec）：結果處理程序 `try_send` 並且默默地
        // 丟棄訊息，它與確定性模擬合約（主機
        // 擁有電線發射；副本僅用於渲染）。
        let (sink_tx, _sink_rx) = crossbeam_channel::unbounded::<omoba_core::transport::OutboundMsg>();
        let mut event_sink = omoba_core::runtime::RuntimeEventVecSink::default();
        if let Err(e) = omoba_core::runtime::process_outcomes(&mut world, &mut event_sink) {
            log::warn!("sim_runner: process_outcomes failed: {}", e);
        }
        world.maintain();

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
            omoba_sim::Fixed64::from_raw(lockstep_dt_fixed_raw_for_tick(batch.tick as u64)),
            sink_tx.clone(),
        );
        // 處理推送的任何結果腳本（投射物/損壞/等）。
        let mut event_sink = omoba_core::runtime::RuntimeEventVecSink::default();
        if let Err(e) = omoba_core::runtime::process_outcomes(&mut world, &mut event_sink) {
            log::warn!("sim_runner: process_outcomes (post-script) failed: {}", e);
        }
        world.maintain();

        // 階段 4.5：重建能力 如果仍然為空，則懶惰地弧形並且
        // 註冊表已填入。在第一個非空構建之後
        // Arc 永遠不會改變（註冊表在載入後是不可變的）。
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
            let reg = world.read_resource::<omoba_core::comp::tower_registry::TowerTemplateRegistry>();
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

        let mut snapshot = extract_snapshot(
            &mut world,
            batch.tick,
            abilities_arc.clone(),
            tower_templates_arc.clone(),
            tower_upgrades_arc.clone(),
            applied_input_ids,
            applied_input_meta,
        );
        let tower_fire_fx = std::mem::take(&mut snapshot.tower_fire_fx);
        snapshot.tower_fire_fx =
            retain_recent_render_fx(&mut recent_tower_fire_fx, tower_fire_fx, batch.tick, |fx| {
                fx.spawn_tick
            });
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
        let sim_publish_us = wall_clock_us();
        for meta in &mut snapshot.applied_input_meta {
            meta.sim_publish_us = sim_publish_us;
        }

        // 「蠕變 HP 條保持滿」回歸報告的診斷
        // （第 4-5 階段鎖步清理）。每秒採樣一次
        // 前幾個小兵的 HP 值。如果惠普永遠不會改變
        // 跑，鏡子的傷害路徑被打破；如果 HP 減少，
        // 鏡像很好，回歸僅渲染。採樣每秒一次以將日誌量保持在
        // TD_STRESS 規模的較低水準。
        if batch.tick % LOCKSTEP_ONE_SECOND_TICKS_U32 == 0 {
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
                    snapshot
                        .entities
                        .iter()
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
    let mut world = omoba_core::runtime::create_world_for_scene(scene_path)?;
    // 使用權威的 MasterSeed 覆蓋預設的 MasterSeed
    // 遊戲開始。必須在第一次調度之前發生。
    world
        .write_resource::<omoba_core::comp::resources::MasterSeed>()
        .0 = master_seed;
    Ok(world)
}

fn push_inputs_into_world(world: &mut World, tick: u32, inputs: Vec<TickBatchInput>) {
    // 階段 3.4：將鎖步 TickBatch 輸入寫入 shared runtime 的
    // `PendingPlayerInputs` 資源，所以 `tick::player_input_tick::Sys`
    // 可以在調度程序運行開始時耗盡它們。
    //
    // 替換資源圖批發（鎖步合約：最多一個
    // 每個玩家每個刻度的輸入 — 最新的 TickBatch 是權威的）。
    use omoba_core::comp::PendingPlayerInputs;

    let mut pending = world.write_resource::<PendingPlayerInputs>();
    pending.tick = tick;
    pending.by_player.clear();
    if !inputs.is_empty() {
        log::trace!("sim_runner: tick {} got {} inputs", tick, inputs.len());
    }
    for input in inputs {
        pending.by_player.insert(input.player_id, input.input);
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

        for required in ["idle", "idle_2", "idle_3", "move", "attack", "critical", "sniper"] {
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
            generated_attack_source.timeline_offset_ticks.to_f32_for_render()
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
