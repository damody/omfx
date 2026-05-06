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

# ![允許(dead_code)]

use std::path::{Path, PathBuf};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::thread;

use crossbeam_channel::{unbounded, Receiver, Sender};
use log::{error, info};

use specs::{Join, World, WorldExt};

// 從 omobab so feeders 重新匯出第 2 階段 PlayerInput 原型類型
// （階段3.3中的lockstep_client.rs）和sim_runner共享相同的
// 具體類型。 omobab 從 prost 產生的模組重新匯出它
// 在“lockstep::PlayerInput”下。
pub use omobab::lockstep::PlayerInput;

/// 階段 4.2：僅渲染爆炸 FX 條目，鏡像自
/// `omobab::comp::結果::ExplosionFx`。再出口通過
/// `SimWorldSnapshot.explosions` 因此渲染端永遠不需要
/// 直接觸摸 omobab 類型。 `spawn_tick` 是 sim 刻度
/// 爆炸發生；渲染線程使用 omfx 掛鐘
/// 了解實際的環老化生命週期（請參閱 lib.rs `active_explosions`）。
pub use omobab::comp::ExplosionFx;

const APPLIED_INPUT_ID_RETENTION_TICKS: u32 = 300;

/// 最新 sim 刻度狀態的渲染執行緒可讀快照。
#[derive(Default, Clone, Debug)]
pub struct SimWorldSnapshot {
    pub tick: u32,
    pub entities: Vec<EntityRenderData>,
    /// Creep 檢查點路徑（世界座標，原始 f32 — 渲染側套用 WORLD_SCALE）。
    /// 每個內部 Vec 都是一個命名路徑的「(x, y)」檢查點的有序清單。
    /// `init_creep_wave` 之後靜態；重新發出每個快照，以便渲染
    /// 橋接器在 GameStart 後第一次讀取時即可看到它們，而無需
    /// 專用的僅初始化通道。
    pub paths: Vec<Vec<(f32, f32)>>,
    /// 在上一個快照中還存在但不再存在的實體 ID
    /// 存在於目前的 ECS 世界。由渲染線程用來釋放
    /// 每個 eid 場景快取（標籤、批次槽），無需線路端
    /// `實體.死亡`事件。取代舊版 omb 端 `make_entity_death`
    /// 發射;快照差異是在每個tick 本地計算的。
    pub removed_entity_ids: Vec<u32>,
    /// TD 波數 — 以 1 為基礎的目前波浪指數。第一個之前 0
    /// `StartRound` 翻轉 `CurrentCreepWave.is_running`。來源自
    /// `CurrentCreepWave` 資源（`wave: usize`，在邊界處轉換為 u32）。
    pub round: u32,
    /// 為活動場景載入的蠕動波總數。來源
    /// 來自“Vec<CreepWave>”資源長度。非 TD 模式下為 0。
    pub total_rounds: u32,
    /// 目前 TD 玩家生命值 (`PlayerLives.0`)。非 TD 模式下為 0（哨兵
    /// flag — HUD 依照「生命 > 0」切換模式）。
    pub lives: i32,
    /// 當 TD 回合正在運作時為真（小兵在路徑上產卵）；
    /// 一旦波浪被清除，則翻轉為假。鏡子
    /// `CurrentCreepWave.is_running`。
    pub round_is_running: bool,
    /// 階段 4.1：BlockedRegion 多邊形 — 來源為不可步行的地圖區域
    /// 來自“BlockedRegions(Vec<BlockedRegion>)”。之後的靜態地圖數據
    /// `state::initialization` 載入它們；克隆每個蜱蟲很便宜（TD_1 是
    /// 空的; MVP_1/DEBUG_1 有一些）。渲染端繪製紅色
    /// 每個區域的多邊形輪廓；自此以來，“circle”目前始終為“None”
    /// omb `BlockedRegion` 沒有半徑場，但該場是垂直的
    /// 用於前向相容（例如循環阻塞器）。
    pub blocked_regions: Vec<BlockedRegionSnapshot>,
    /// 階段 4.5：AbilityRegistry — 載入的靜態能力元數據
    /// 遊戲開始時的腳本 DLL。 “Arc” 包裹起來，因此每個蜱蟲克隆都是
    /// O(1)；一旦註冊表非空（腳本載入為
    /// 異步）。英雄面板使用它來解析 `ability_ids[i]` →
    /// 顯示名稱/圖示/最大等級。
    pub abilities: std::sync::Arc<Vec<AbilityDefSnapshot>>,
    /// TD 塔模板（右側建置按鈕選單）。源自
    /// 腳本 DLL 透過每個塔的模板填充“TowerTemplateRegistry”
    /// 遊戲開始時的「tower_metadata()」。 `Arc` 包裹 — 同樣的惰性構建
    /// 模式作為能力（註冊表是異步填充的）。 lib.rs 讀取
    /// 在第一個非空快照上一次播種“td_template_order”+
    /// `td_templates` HashMap；後續的刻度是 O(1) 弧形克隆。
    pub tower_templates: std::sync::Arc<Vec<TowerTemplateSnapshot>>,
    /// 48 個 tower upgrade defs (4 towers × 3 paths × 4 levels). Sourced
    /// 來自 `omobab::comp::tower_upgrade_registry::TowerUpgradeRegistry`。
    /// omfx 銷售/升級面板用於 (a) 計算退款 =
    /// 基礎*0.85 + Σ(升級*0.75) 和 (b) 顯示每個升級的名稱
    /// 升級按鈕文字。與延遲建構弧線模式相同
    /// `塔模板`。
    pub tower_upgrades: std::sync::Arc<Vec<TowerUpgradeDefSnapshot>>,
    /// 階段 4.2：本報價發出爆炸性 FX 事件 — 每個條目一個條目
    /// 由「process_outcomes」處理的「Outcome::Explosion」。排出自
    /// sim 的“ExplosionFxQueue”資源每個刻度（“std::mem::take”）
    /// 所以隊列保持有界。渲染線程產生瞬態
    /// 針對 omfx 掛鐘，每個條目都會擴展紅色環；模擬永遠不會
    /// 讀回這個，所以它不是決定論狀態的一部分。
    pub explosions: Vec<ExplosionFx>,
    /// 用於輸入到渲染延遲配對的僅限 omfx 元資料； sim ECS 不讀取它。
    pub applied_input_ids: Vec<u32>,
}

/// 階段 4.1：一個多邊形區域快照。
#[derive(Clone, Debug, Default)]
pub struct BlockedRegionSnapshot {
    /// 世界座標中的多邊形頂點（原始 f32 — 渲染側適用
    /// WORLD_SCALE + `-x` 翻轉就像路徑段渲染器一樣）。
    pub points: Vec<(f32, f32)>,
    /// 橘色圓形阻擋器的可選中心 + 半徑。奧姆
    /// `BlockedRegion` 今天沒有半徑，所以它始終是 `None`；
    /// 保持與未來循環區域的前向相容。
    pub circle: Option<((f32, f32), f32)>,
}

/// 階段 4.5：AbilityDef 投影 — 僅 omfx 英雄的字段
/// 面板需求。保持精簡（沒有「等級」HashMap 或「屬性」JSON
/// blob），因為能力欄顯示能力 ID / 最大等級 / 圖示。
#[derive(Clone, Debug)]
pub struct AbilityDefSnapshot {
    pub ability_id: String,
    pub display_name: String,
    pub max_level: u8,
    pub icon_path: String,
}

/// TowerUpgradeDef 投影 — 僅銷售/升級面板的字段
/// 需求（按鈕標籤名稱、退款計算成本）。
#[derive(Clone, Debug)]
pub struct TowerUpgradeDefSnapshot {
    pub tower_kind: String,
    pub path: u8,
    pub level: u8,
    pub name: String,
    pub cost: i32,
}

/// 右側 TD 建立選單的 TowerTemplate 投影。鏡像
/// 欄位 lib.rs 的 `TdTemplate` 快取需求（按鈕的標籤/成本
/// + 佈局預覽的足跡/範圍）。源自
/// `omobab::comp::tower_registry::TowerTemplateRegistry`。
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

/// 最後從 ECS World 中提取的每個實體渲染數據
/// 每一個刻度。已經映射到邊界處的 `f32` (Fixed64 →
/// `to_f32_for_render`);渲染線程不需要知道
/// 確定性 sim 的定點類型。
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
    /// `ScriptUnitTag.unit_id` 如果存在（例如“tower_dart_monkey”，
    /// 「creep_balloon_red」、「hero_saika_magoichi」）。實體為空
    /// 沒有腳本標籤（罕見 - 大多數產生的單位都有一個）。
    pub unit_id: String,
    /// 僅限英雄的元資料。當實體不是英雄時為空/0。
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
    /// 金幣（英雄的玩家資源）。 0 表示非英雄實體。
    pub gold: i32,
    /// 階段 3.3：總結英雄統計（BuffStore / 之後的最終值）
    /// UnitStats 聚合）。對於非英雄實體，「None」 - 保留
    /// EntityRenderData 小，適用於 1000 塔/500 蠕變應力
    /// 小路。裝箱，因此英雄手臂支付堆分配，但塔/蠕動
    /// rows 只支付一個 None 指標。
    pub hero_ext: Option<Box<HeroStatsExt>>,
    /// 階段 4.3：每條路徑的塔升級等級點（3 條路徑 × 0-4 級）。
    /// 對於非 Tower 實體為「無」。源自“Tower.upgrade_levels”
    /// 組件欄位；現有的 TD 出售/升級面板已顯示
    /// 這與“network_entities”無關，因此快照變體是
    /// 鎖步側後視鏡。
    pub upgrade_levels: Option<[u8; 3]>,
}

/// 階段3.3：英雄面板的單一buff快照。
///
/// 鏡像舊的“hero.stats”“buffs”數組。 `剩餘秒數`
/// 使用“-1.0”作為“無限/切換”的哨兵（例如base_stats
/// 或 sniper_mode) — 渲染端顯示為 ∞。否則
/// 渲染線程在本地每幀減少“remaining_secs”；下一個
/// 權威快照重置值，避免漂移。
#[derive(Clone, Debug, Default)]
pub struct BuffSnapshot {
    pub buff_id: String,
    pub remaining_secs: f32,
    /// 字串化有效負載 JSON。最壞情況下回退到「調試」repr
    /// 當規範的 JSON 編碼不可用時；面板
    /// 使用“as_f64()”列出了數字有效負載字段，因此我們保留
    /// JSON 往返保留它。
    pub payload_json: String,
}

/// 階段 3.3：總結英雄統計 — omfx 側鏡像
/// 舊版 omb `hero.stats` JSON 負載。透過同樣的計算
/// 使用 `BuffStore` / `UnitStats` 聚合管道 omb（參見
/// `omobab::ability_runtime::UnitStats`);針對 ECS 只讀
/// 所以同步決定論不受影響。
#[derive(Clone, Debug, Default)]
pub struct HeroStatsExt {
    pub armor: f32,
    pub magic_resist: f32,
    pub attack_damage: f32,
    pub attack_range: f32,
    pub move_speed: f32,
    /// 每次攻擊秒數 (asd) — 非攻擊單位為 0。
    pub attack_speed_sec: f32,
    pub bullet_speed: f32,
    /// 英雄法力 / 最大法力（第 3.3 階段： omb `CProperty` 尚未
    /// 有英雄法力場，所以這些是 0；遺留的“hero.stats”
    /// 有效負載也連接到 0)。為前向相容而設計。
    pub mana: f32,
    pub max_mana: f32,
    pub buffs: Vec<BuffSnapshot>,
    /// 階段 4.4：每個插槽的庫存物品 ID。 omb `Inventory` 有 6 個
    /// 插槽（`INVENTORY_SLOTS = 6`）；每個插槽容納一個
    /// “Option<ItemInstance>”，其“item_id”是“String”。 「無」對於
    /// 空插槽。這裡故意省略了冷卻時間——遺產
    /// `hero_state.inventory` HUD 已經驅動本機 CD 行情
    /// (`Vec<Option<(String, f32)>>`)，以及階段 2.4 ItemUse start_cd
    /// 發生在主機端；下一個快照重置就可以了。
    pub inventory: [Option<String>; 6],
    /// 階段 4.5：每個槽位的能力等級（Q/W/E/R = 指數 0..3）。
    /// 源自 `Hero.ability_levels: HashMap<String, i32>` 鍵控
    /// 通過 `ability_ids[i]`。如果未學習/槽中沒有能力，則為 0。
    pub ability_levels: [i32; 4],
    /// 階段 4.5：每個插槽的能力 ID。鏡像“英雄.能力[i]”
    /// （Q/W/E/R 順序）。如果英雄的技能少於 4 個，則為“無”
    /// 或該插槽未設定。渲染端找這些
    /// `SimWorldSnapshot.powered` 解析顯示名稱/圖示。
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

/// 每個時脈週期由鎖步饋送器提交的通道有效負載。
#[derive(Clone, Debug)]
pub struct TickBatchPayload {
    pub tick: u32,
    pub inputs: Vec<(u32 /* player_id */, PlayerInput, u32 /* input_id */)>,
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
/// `omobab::state::initialization::create_world_for_scene` 並運行
/// 每個蜱蟲的輸入驅動的共享階段 3 調度程序
/// `tick_input_rx`。
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

    let mut dispatcher = match omobab::state::system_dispatcher::build_phase3_dispatcher() {
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
    let script_registry: omobab::scripting::ScriptRegistry = std::mem::take(
        &mut *world.write_resource::<omobab::scripting::ScriptRegistry>(),
    );

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
    let mut recent_applied_input_ids: VecDeque<(u32, u32)> = VecDeque::new();
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

        // 更新 Tick + Time + DeltaTime，以便時間閘控系統（creep_wave、
        // 增益計時器、彈丸飛行）實際上是提前的。鎖步為 60Hz
        // （TickBroadcaster 的tick_period_us = 16_667），所以dt = 1/60。
        // 如果沒有這些，本地 sim 會有 Tick 前進，但時間停留在 0，
        // 這使得 `creep_wave` 看到 `totaltime=0` 並且永遠不會產生 — 完全正確
        // 為什麼 Start Round 會觸發（is_running 翻轉）但沒有小兵出現。
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

        // 階段 2.1：耗盡 `PendingTowerSpawnQueue` 填充
        // 上述調度期間的`player_input_tick::Sys`。鏡子都一樣
        // 呼叫 omb 的 `state::core::tick` 以便主機 + 副本產生 TD 塔
        // 確定性地來自“PlayerInputEnum::TowerPlace”輸入。
        omobab::comp::GameProcessor::drain_pending_tower_spawns(&mut world);
        world.maintain();

        // 階段 2.2：從 TowerSell 輸入中排出「PendingTowerSellQueue」。
        // 鏡像 omb 的 `state::core::tick`。退款+實體刪除完成於
        // 在主機和副本上同步，以便快照保持一致。
        omobab::comp::GameProcessor::drain_pending_tower_sells(&mut world);
        world.maintain();

        // 階段 2.3：從 TowerUpgrade 排出 `PendingTowerUpgradeQueue`
        // 輸入。鏡像 omb 的 `state::core::tick`。金扣+
        // Upgrade_levels 增量 + BuffStore stat-mod 添加需要運行
        // 主機和副本同步，因此快照保持一致。
        omobab::comp::GameProcessor::drain_pending_tower_upgrades(&mut world);
        world.maintain();

        // 階段 2.4：從 ItemUse 輸入排出「PendingItemUseQueue」。
        // 鏡像 omb 的 `state::core::tick`。庫存冷卻時間+C屬性
        // (HP / msd) 突變需要在主機和副本上同步運作。
        omobab::comp::GameProcessor::drain_pending_item_uses(&mut world);
        world.maintain();

        // MoveTo (右鍵移動): drain `PendingMoveQueue` — writes MoveTarget on
        // 玩家英雄。鏡像 omb 的 `state::core::tick`。
        omobab::comp::GameProcessor::drain_pending_moves(&mut world);
        world.maintain();

        // 階段 3 調度程序僅調度滴答系統；它不包括
        // GameProcessor::process_outcomes。如果沒有這個，`creep_wave`會產生
        // `Outcome::Creep { cd }` 行堆積在 `Vec<Outcome>` 中，但沒有
        // 實體在本機 sim 中產生 → snapshot.creep 保持 0。
        // mqtx 是一個接收器（空 Vec）：結果處理程序 `try_send` 並且默默地
        // 丟棄訊息，它與確定性模擬合約（主機
        // 擁有電線發射；副本僅用於渲染）。
        let (sink_tx, _sink_rx) = crossbeam_channel::unbounded::<omobab::transport::OutboundMsg>();
        if let Err(e) = omobab::comp::GameProcessor::process_outcomes(&mut world, &sink_tx) {
            log::warn!("sim_runner: process_outcomes failed: {}", e);
        }
        world.maintain();

        // 運行腳本調度，以便塔/英雄/召喚`on_tick`鉤子火。
        // 塔是 ScriptUnitTag 驅動的 - 沒有這個， tower_dart / tower_
        // 炸彈/ tower_ice從未決定攻擊，所以projectile_tick有
        // 沒有什麼可以提前的，damage_tick 也沒有什麼可以應用的。
        // omb 的 `State::tick` 在 `run_systems` 之後執行相同的操作（請參閱
        // `scripting::run_script_dispatch` 周圍的 `omb/src/state/core.rs`
        // 稱呼）。副本需要相同的呼叫來保持 sim 等效。
        omobab::scripting::run_script_dispatch(
            &mut world,
            &script_registry,
            batch.tick as u64,
            omoba_sim::Fixed64::from_raw((SIM_DT_S * 1024.0) as i64),
            sink_tx.clone(),
        );
        // 處理推送的任何結果腳本（投射物/損壞/等）。
        if let Err(e) = omobab::comp::GameProcessor::process_outcomes(&mut world, &sink_tx) {
            log::warn!("sim_runner: process_outcomes (post-script) failed: {}", e);
        }
        world.maintain();

        // 階段 4.5：重建能力 如果仍然為空，則懶惰地弧形並且
        // 註冊表已填入。在第一個非空構建之後
        // Arc 永遠不會改變（註冊表在載入後是不可變的）。
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

        // TD 塔範本註冊表 — 相同的惰性建置模式。人口由
        // 每個塔腳本在腳本載入時的「tower_metadata()」。
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

        // TowerUpgradeRegistry — 在世界初始化時建構一次（不像非同步
        // 塔模板），因此 iter_all 從勾選 1 開始就非空。惰性保護
        // 鏡像其他註冊表以實現對稱。
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

        // 「蠕變 HP 條保持滿」回歸報告的診斷
        // （第 4-5 階段鎖步清理）。每 60 個刻度 (~1s) 採樣一次
        // 前幾個小兵的 HP 值。如果惠普永遠不會改變
        // 跑，鏡子的傷害路徑被打破；如果 HP 減少，
        // 鏡像很好，回歸僅渲染。採樣每個
        // 60 個刻度以將日誌量保持在 TD_STRESS 規模的較低水準。
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
            * s = 快照；
        }
    }
}

fn init_world(scene_path: &Path, master_seed: u64) -> Result<World, failure::Error> {
    let mut world = omobab::state::initialization::create_world_for_scene(scene_path)?;
    // 使用權威的 MasterSeed 覆蓋預設的 MasterSeed
    // 遊戲開始。必須在第一次調度之前發生。
    world.write_resource::<omobab::comp::resources::MasterSeed>().0 = master_seed;
    Ok(world)
}

fn push_inputs_into_world(world: &mut World, tick: u32, inputs: Vec<(u32, PlayerInput, u32)>) {
    // 階段 3.4：將鎖步 TickBatch 輸入寫入主機的
    // `PendingPlayerInputs` 資源，所以 omb 的 `tick::player_input_tick::Sys`
    // 可以在調度程序運行開始時耗盡它們。
    //
    // 替換資源圖批發（鎖步合約：最多一個
    // 每個玩家每個刻度的輸入 — 最新的 TickBatch 是權威的）。
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
    // omobab 通過 `pub use crate::comp::*;` 在
    // 板條箱根部，因此請穿過平坦的路徑而不是
    // 逐個模組（有些子模組如“comp::state”發生衝突
    // 與 State 結構命名空間）。
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
    // 階段 3.3：TAtack + BuffStore 用於英雄統計數據聚合
    // 小路。 BuffStore 是一個「World」資源； UnitStats 借用了它
    // 只讀－沒有 ECS 突變，因此決定論不受影響。
    let tatk_storage = world.read_storage::<TAttack>();
    let buff_store = world.read_resource::<BuffStore>();
    let stats = UnitStats::from_refs(&*buff_store, /*is_building*/ false);
    // 階段 4.4：英雄庫存存儲 — 只有英雄實體才會填入此存儲，
    // 因此對於非英雄行（無），查找很便宜。
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

        // 將角度刻度轉換為 f32 弧度以進行渲染。 TAU_TICKS = 4096
        // → 除以 TAU_TICKS，再乘以 2π。在邊界完成所以
        // 渲染程式碼永遠不需要知道 trig-tick 編碼。
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

        // 僅限英雄的元資料。每個英雄實體讀取一次，保持零成本
        // 對於非英雄行。
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

        // 階段 3.3：最終英雄統計數據總表（護甲/攻擊力/射程/
        // move_speed / buffs) 與 omb 的方式相同
        // `state::resource_management::build_hero_stats_payload` 做到了。
        // 只讀 — `UnitStats::final_*` 和 `BuffStore::iter_for`
        // 永遠不會改變 ECS，因此同步決定論不受影響。
        // 對於非英雄實體“無”，因此塔樓/小兵行只需支付
        // 單一空指針的大小。
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
            // Attack_speed_sec = 基本間隔 / asd_mult。 asd_乘數 = 1
            // 指基礎； > 1 表示更快（間隔更短）。鏡子
            // 在 `build_hero_stats_payload` 中完成分割。
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
            // CProperty還沒有英雄法力場；遺產
            // `build_hero_stats_payload` 也連接為 0。管道用於
            // 法力落地後向前相容。
            let mana = 0.0_f32;
            let max_mana = 0.0_f32;

            let buffs: Vec<BuffSnapshot> = buff_store
                .iter_for(entity)
                .map(|(id, entry)| {
                    // BuffEntry.remaining為Fixed64秒；遺產
                    // 線約定： raw == i32::MAX 是
                    // 「無限/切換」哨兵（sniper_mode，
                    // 基本統計）。映射到 -1.0，以便麵板渲染 ∞。
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

            // 階段 4.4：庫存槽位 — `Inventory.slots` 是
            // `[選項<ItemInstance>; 6]`;我們預計
            // `[選項<字串>; 6]`（僅限 item_id）。空槽 → 無。
            // 英雄可能沒有庫存組件（單元測試/預測試）
            // 撿起）;在這種情況下，所有插槽都是“無”。
            let mut inventory: [Option<String>; 6] = Default::default();
            if let Some(inv) = inventory_storage.get(entity) {
                for (i, slot) in inv.slots.iter().enumerate().take(6) {
                    inventory[i] = slot.as_ref().map(|it| it.item_id.clone());
                }
            }

            // 階段 4.5：能力 ID + 每個槽位的等級（Q/W/E/R = 0..3）。
            // `Hero.bility` 是一個 `Vec<String>` （通常長度為 4，但
            // 我們謹防變短）； `ability_levels` 是一個 HashMap
            // 由能力 id 鍵入。缺失 → 0 / 無。
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

        // 階段 4.3：塔升級等級 — 僅針對塔類型填充
        // 實體。 3路徑×0-4級數組直接從
        // “塔”組件。其他類型得到“None”（零開銷）。
        let upgrade_levels: Option<[u8; 3]> = if matches!(kind, EntityKind::Tower) {
            tower_storage.get(entity).map(|t| t.upgrade_levels)
        } else {
            None
        };

        out.push(EntityRenderData {
            entity_id: entity.id(),
            // 規範 `Generation::id()` 回傳 i32 （從 1 開始，帶符號
            // 跟蹤活著/死亡）。轉換為 u32 以進行快照傳輸。
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

    // 蠕變檢查點路徑 - 每個快照從靜態讀取一次
    // 由「init_creep_wave」填入的「BTreeMap<String, Path>」資源。便宜的
    // (BTree iter + 小克隆);避免專用的僅初始化通道。
    use omobab::comp::Path;
    use std::collections::BTreeMap;
    let paths: Vec<Vec<(f32, f32)>> = world
        .read_resource::<BTreeMap<String, Path>>()
        .values()
        .map(|p| p.check_points.iter().map(|cp| (cp.pos.x, cp.pos.y)).collect())
        .collect();

    // 診斷：轉儲實體類型直方圖+每秒採樣，以便我們可以
    // 找出 sim_runner 累積幽靈實體的原因（411 報告為
    // 空結構+BlockedRegions）。解決根本原因後刪除。
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
        // 前 10 個非英雄實體 — 顯示其 pos / unit_id 以暗示來源。
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

    // 階段 4.1：BlockedRegion 多邊形。靜態地圖資料 — TD_1 為空，
    // MVP_1/DEBUG_1 有一些，因此克隆每個蜱是很便宜的。這
    // omb `BlockedRegion` 有 `name: String` + `points: Vec<Vec2<f32>>`
    // （無半徑）；我們從渲染端開始投影到「(f32, f32)」對
    // 已經講原始的 f32 世界座標。從今天開始，「circle」就沒有了
    // 來源沒有半徑場；為了向前相容，保留可選。
    let blocked_regions: Vec<BlockedRegionSnapshot> = world
        .read_resource::<omobab::comp::BlockedRegions>()
        .0
        .iter()
        .map(|r| BlockedRegionSnapshot {
            points: r.points.iter().map(|p| (p.x, p.y)).collect(),
            circle: None,
        })
        .collect();

    // 階段 3.2：TD HUD 狀態 — Round / Lives / round_is_running。
    // StartRound 翻轉後，`CurrentCreepWave.wave` 是從 1 開始的 `usize`
    // `正在運行`;第一輪前0分。 `total_rounds` = 長度
    // `Vec<CreepWave>` 資源。 `PlayerLives` 是一個元組結構
    // 包裝`i32`。這三個都是 ECS 的唯讀讀取
    // 資源－沒有突變，所以決定論不受影響。
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

    // 階段 4.2：排出 `ExplosionFxQueue` — process_outcomes 推送到這裡
    // 對於每個 Outcome::Explosion (game_processor + WorldAdapter
    // 發射爆炸）。 `std::mem::take` 交換預設的空 Vec，因此
    // 佇列佔用 O(1) 記憶體； sim 永遠不會讀回佇列，所以
    // write 對於決定論來說是不可見的（同樣的原因 BlockedRegions 是
    // 在這裡可以安全閱讀）。
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

/// 冒煙測試表明 omobab 作為 lib 是可以訪問的。驗證 dep 接線
/// 有效且階段 3.2 輔助符號解析。
pub fn smoke() -> &'static str {
    let _ = omobab::comp::resources::MasterSeed::default();
    // 進入第 3.2 階段新增的新酒吧助手進行確認
    // 它們可以從 omfx 中看到。
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
