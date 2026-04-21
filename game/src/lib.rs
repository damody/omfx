//! omfx - 2D Tower Defense Network Renderer (Fyrox 1.0)
//!
//! Pure network renderer: all game state driven by omb backend via gRPC.
//! No local game logic — entities are created/moved/deleted by server events.
#![allow(warnings)]

use fyrox::graph::prelude::*;
use fyrox::{
    core::{
        algebra::{UnitQuaternion, Vector2, Vector3},
        color::Color,
        pool::Handle,
        reflect::prelude::*,
        visitor::prelude::*,
    },
    event::{ElementState, Event, MouseButton, WindowEvent},
    gui::{
        brush::Brush,
        image::{ImageBuilder, ImageMessage},
        message::{MessageDirection, UiMessage},
        text::{Text, TextBuilder, TextMessage},
        widget::{WidgetBuilder, WidgetMessage},
        HorizontalAlignment, UiNode, UserInterface, VerticalAlignment,
    },
    plugin::{error::GameResult, Plugin, PluginContext, PluginRegistrationContext},
    scene::{
        base::BaseBuilder,
        camera::{CameraBuilder, OrthographicProjection, Projection},
        dim2::rectangle::RectangleBuilder,
        light::{point::PointLightBuilder, BaseLightBuilder},
        node::Node,
        transform::TransformBuilder,
        EnvironmentLightingSource, Scene,
    },
};

use std::collections::{HashMap, BinaryHeap};
use std::cmp::{Ordering, Reverse};
use std::time::{SystemTime, UNIX_EPOCH};

use omoba_core::GameEventData;

pub use fyrox;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const GRID_COLS: usize = 12;
const GRID_ROWS: usize = 8;
const CELL_SIZE: f32 = 1.0;
const GRID_ORIGIN_X: f32 = -6.0;
const GRID_ORIGIN_Y: f32 = -4.0;

// Backend → render coordinate scale (backend uses large units like 800)
const WORLD_SCALE: f32 = 0.01; // 800 backend → 8.0 render

// Z layers: smaller Z = closer to camera (near plane) = renders on top.
const Z_BULLET: f32 = 0.000;
const Z_HP_BAR: f32 = 0.0005;
const Z_ENEMY: f32 = 0.001;
const Z_TOWER: f32 = 0.002;
const Z_GRID_CELL: f32 = 0.003;
const Z_PATH: f32 = 0.004;
const Z_BACKGROUND: f32 = 0.005;

// ---------------------------------------------------------------------------
// Network Types
// ---------------------------------------------------------------------------

/// Frontend → Backend command
enum NetCommand {
    PlaceTower { x: f32, y: f32 },
    HeroMove { x: f32, y: f32 },
    ViewportUpdate { cx: f32, cy: f32, hw: f32, hh: f32 },
    CastAbility { slot: String, x: f32, y: f32 },
    UpgradeSkill { slot: String },
    BuyItem { item_id: String },
    SellItem { slot: usize },
    UseItem { slot: usize },
}

/// Timestamped backend event (for sorted buffering)
#[derive(Debug)]
struct TimestampedEvent {
    timestamp_ms: u64,
    msg_type: String,
    action: String,
    data: serde_json::Value,
}

impl PartialEq for TimestampedEvent {
    fn eq(&self, other: &Self) -> bool {
        self.timestamp_ms == other.timestamp_ms
    }
}

impl Eq for TimestampedEvent {}

impl PartialOrd for TimestampedEvent {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TimestampedEvent {
    fn cmp(&self, other: &Self) -> Ordering {
        self.timestamp_ms.cmp(&other.timestamp_ms)
    }
}

/// Event buffer — sorts by timestamp, delays render_delay before consuming
#[derive(Debug)]
struct EventBuffer {
    heap: BinaryHeap<Reverse<TimestampedEvent>>, // min-heap
    render_delay_ms: u64,
    clock_offset_ms: i64,
    synced: bool,
}

impl EventBuffer {
    fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
            render_delay_ms: 100,
            clock_offset_ms: 0,
            synced: false,
        }
    }

    fn push(&mut self, event: TimestampedEvent) {
        self.heap.push(Reverse(event));
    }

    /// Calibrate clock using heartbeat timestamp_ms
    fn sync_clock(&mut self, server_timestamp_ms: u64) {
        let client_now = SystemTime::now()
            .duration_since(UNIX_EPOCH).unwrap().as_millis() as i64;
        let new_offset = server_timestamp_ms as i64 - client_now;
        if self.synced {
            // Exponential moving average, smooth jitter (alpha = 0.1)
            self.clock_offset_ms = self.clock_offset_ms + (new_offset - self.clock_offset_ms) / 10;
        } else {
            self.clock_offset_ms = new_offset;
            self.synced = true;
        }
    }

    /// Estimate current server time
    fn server_now_ms(&self) -> u64 {
        let client_now = SystemTime::now()
            .duration_since(UNIX_EPOCH).unwrap().as_millis() as i64;
        (client_now + self.clock_offset_ms) as u64
    }

    /// Drain all events that are ready to display
    fn drain_ready(&mut self) -> Vec<TimestampedEvent> {
        let deadline = self.server_now_ms().saturating_sub(self.render_delay_ms);
        let mut ready = Vec::new();
        while let Some(Reverse(evt)) = self.heap.peek() {
            if evt.timestamp_ms <= deadline {
                ready.push(self.heap.pop().unwrap().0);
            } else {
                break;
            }
        }
        ready
    }
}

/// Backend entity → Fyrox scene node mapping
#[derive(Debug)]
struct NetworkEntity {
    entity_type: String,
    node: Handle<Node>,
    hp_bar_bg: Option<Handle<Node>>,
    hp_bar_fg: Option<Handle<Node>>,
    position: Vector2<f32>,
    health: Option<(f32, f32)>, // (current, max)
    name: String,
    name_label: Option<Handle<Text>>,
    // Client-side interpolation
    prev_position: Vector2<f32>,
    target_position: Vector2<f32>,
    lerp_elapsed: f32,
    lerp_duration: f32, // derived from move_speed + segment distance
    // Backend-reported move speed (backend units per second); 0 for static entities.
    move_speed: f32,
    // Debug polyline segments (for creep path visualization)
    path_nodes: Vec<Handle<Node>>,
    // Seconds since path_nodes were drawn; used to expire them after PATH_VISIBLE_SECS.
    path_age: f32,
    // 面向角度（radians，0 = +X，CCW 正）
    facing: f32,
    // 箭頭指示面向的子節點
    facing_arrow: Option<Handle<Node>>,
}

/// Seconds that a newly-spawned creep's debug path stays visible.
const PATH_VISIBLE_SECS: f32 = 5.0;

/// Client-side projectile simulation.
///
/// Backend only sends a single C event with `target_id` + `flight_time_ms`;
/// the bullet's position is computed locally each frame as a pursuit lerp
/// from `start_pos` toward the target entity's CURRENT client-side position.
/// This eliminates the per-tick network round-trip lag that made bullets
/// visually trail the creep.
#[derive(Debug)]
struct ClientProjectile {
    node: Handle<Node>,
    target_id: u32,
    start_pos: Vector2<f32>,
    last_target_pos: Vector2<f32>,
    elapsed: f32,
    flight_time: f32,
    // Predicted damage applied client-side when the bullet visually hits;
    // heartbeat HP snapshot reconciles drift every 2s.
    damage: f32,
    applied: bool,
}

/// Heartbeat info (for UI display)
#[derive(Default, Debug)]
struct HeartbeatInfo {
    tick: u64,
    game_time: f64,
    entity_count: u64,
    hero_count: u64,
    creep_count: u64,
}

/// Connection status
#[derive(Default, Clone, PartialEq, Debug)]
enum ConnectionStatus {
    #[default]
    Disconnected,
    Connecting,
    Connected,
    Failed(String),
}

/// Try to spawn the omb backend as a child process.
/// RAII guard that owns the backend `Child` and (on Windows) a Job Object handle.
///
/// Kill semantics:
/// - **Graceful exit** (window close, Ok return, panic unwind): `Drop` runs, calling
///   `child.kill()` + `child.wait()`.
/// - **Hard kill** (task manager, log-off, parent crash): Windows closes all our
///   handles. Because the backend is in a Job Object with
///   `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`, the OS terminates it automatically the
///   moment our process ends and the job handle is implicitly closed.
#[derive(Debug)]
struct BackendGuard {
    child: Option<std::process::Child>,
    #[cfg(windows)]
    job: Option<windows::Win32::Foundation::HANDLE>,
}

impl Drop for BackendGuard {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            log::info!("BackendGuard dropping — killing backend (PID: {})", child.id());
            let _ = child.kill();
            let _ = child.wait();
        }
        #[cfg(windows)]
        {
            if let Some(job) = self.job.take() {
                // Closing the last handle to the job triggers KILL_ON_JOB_CLOSE,
                // which kills any still-alive processes inside.
                use windows::Win32::Foundation::CloseHandle;
                unsafe { let _ = CloseHandle(job); }
            }
        }
    }
}

/// Spawn the backend as a child process, tied to the current process's lifetime.
/// Returns `None` if we can't find the omb directory or the spawn fails.
fn spawn_backend() -> Option<BackendGuard> {
    use std::process::{Command, Stdio};
    use std::path::PathBuf;

    // Find the omb directory relative to cwd
    let candidates = [
        PathBuf::from("omb"),       // cwd = D:\omoba
        PathBuf::from("../omb"),    // cwd = D:\omoba\omfx
        PathBuf::from("../../omb"), // cwd = D:\omoba\omfx\executor
    ];

    let omb_dir = candidates.iter().find(|p| p.join("game.toml").exists());
    let omb_dir = match omb_dir {
        Some(d) => d.clone(),
        None => {
            log::warn!("Cannot find omb directory (no game.toml found), skipping backend spawn");
            return None;
        }
    };

    log::info!("Auto-starting backend from {:?}...", omb_dir);

    // Try pre-built binary first
    let exe_path = omb_dir.join("target/debug/omobab.exe");
    let result = if exe_path.exists() {
        Command::new(&exe_path)
            .current_dir(&omb_dir)
            .stdin(Stdio::null())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .spawn()
    } else {
        log::info!("Pre-built binary not found, falling back to cargo run...");
        Command::new("cargo")
            .args(["run", "--features", "kcp"])
            .current_dir(&omb_dir)
            .stdin(Stdio::null())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .spawn()
    };

    let child = match result {
        Ok(c) => {
            log::info!("Backend process spawned (PID: {})", c.id());
            c
        }
        Err(e) => {
            log::error!("Failed to spawn backend: {}", e);
            return None;
        }
    };

    #[cfg(windows)]
    let job = create_job_and_attach(&child);
    #[cfg(windows)]
    {
        Some(BackendGuard { child: Some(child), job })
    }
    #[cfg(not(windows))]
    {
        Some(BackendGuard { child: Some(child) })
    }
}

/// Create a Windows Job Object with KILL_ON_JOB_CLOSE, attach the given child,
/// and return the job handle. On failure, returns `None` — the `BackendGuard`
/// falls back to `Drop`-time kill only.
#[cfg(windows)]
fn create_job_and_attach(child: &std::process::Child) -> Option<windows::Win32::Foundation::HANDLE> {
    use std::os::windows::io::AsRawHandle;
    use windows::Win32::Foundation::HANDLE;
    use windows::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, SetInformationJobObject,
        JobObjectExtendedLimitInformation, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };

    unsafe {
        let job = match CreateJobObjectW(None, None) {
            Ok(h) if !h.is_invalid() => h,
            _ => {
                log::warn!("CreateJobObjectW failed; falling back to Drop-only cleanup");
                return None;
            }
        };

        let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;

        if SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            &info as *const _ as *const _,
            std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
        )
        .is_err()
        {
            log::warn!("SetInformationJobObject failed; job won't auto-kill children");
        }

        let child_handle = HANDLE(child.as_raw_handle());
        if AssignProcessToJobObject(job, child_handle).is_err() {
            log::warn!("AssignProcessToJobObject failed; backend not tied to our lifetime");
        } else {
            log::info!("Backend process attached to Job Object (auto-kill on exit)");
        }

        Some(job)
    }
}

/// Sync/Async bridge
#[derive(Debug)]
struct NetworkBridge {
    event_rx: crossbeam_channel::Receiver<TimestampedEvent>,
    cmd_tx: crossbeam_channel::Sender<NetCommand>,
    status_rx: crossbeam_channel::Receiver<ConnectionStatus>,
}

impl NetworkBridge {
    fn spawn(server_addr: String, player_name: String) -> Self {
        let (event_tx, event_rx) = crossbeam_channel::unbounded();
        let (cmd_tx, cmd_rx) = crossbeam_channel::unbounded::<NetCommand>();
        let (status_tx, status_rx) = crossbeam_channel::unbounded();

        std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async move {
                let _ = status_tx.send(ConnectionStatus::Connecting);

                // Retry forever, once per second, so the frontend can be started
                // before the backend and will automatically attach when it comes up.
                let mut client = {
                    let mut attempt = 0u32;
                    loop {
                        match omoba_core::KcpClient::connect(&server_addr, player_name.clone()).await {
                            Ok(c) => {
                                let _ = status_tx.send(ConnectionStatus::Connected);
                                break c;
                            }
                            Err(e) => {
                                attempt += 1;
                                log::info!("Connection attempt {} failed: {}, retrying in 1s...", attempt, e);
                                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                            }
                        }
                    }
                };

                let mut grpc_rx = match client.subscribe_events().await {
                    Ok(rx) => rx,
                    Err(e) => {
                        let _ = status_tx.send(ConnectionStatus::Failed(e.to_string()));
                        return;
                    }
                };

                // Event relay: tokio mpsc → crossbeam
                let event_tx_clone = event_tx.clone();
                let relay_events = tokio::spawn(async move {
                    while let Some(evt) = grpc_rx.recv().await {
                        if event_tx_clone.send(TimestampedEvent {
                            timestamp_ms: evt.timestamp_ms,
                            msg_type: evt.msg_type,
                            action: evt.action,
                            data: evt.data,
                        }).is_err() {
                            break;
                        }
                    }
                });

                // Command relay: crossbeam → gRPC
                let relay_commands = tokio::spawn(async move {
                    loop {
                        match cmd_rx.try_recv() {
                            Ok(cmd) => match cmd {
                                NetCommand::PlaceTower { x, y } => {
                                    let _ = client.send_command("tower", "create",
                                        serde_json::json!({"x": x, "y": y})).await;
                                }
                                NetCommand::HeroMove { x, y } => {
                                    let _ = client.send_command("player", "move",
                                        serde_json::json!({"x": x, "y": y})).await;
                                }
                                NetCommand::ViewportUpdate { cx, cy, hw, hh } => {
                                    let _ = client.send_viewport_update(cx, cy, hw, hh).await;
                                }
                                NetCommand::CastAbility { slot, x, y } => {
                                    let _ = client.send_command("player", "cast_ability",
                                        serde_json::json!({"slot": slot, "target_pos": [x, y]})).await;
                                }
                                NetCommand::UpgradeSkill { slot } => {
                                    let _ = client.send_command("player", "upgrade_skill",
                                        serde_json::json!({"slot": slot})).await;
                                }
                                NetCommand::BuyItem { item_id } => {
                                    let _ = client.send_command("player", "buy_item",
                                        serde_json::json!({"item_id": item_id})).await;
                                }
                                NetCommand::SellItem { slot } => {
                                    let _ = client.send_command("player", "sell_item",
                                        serde_json::json!({"slot": slot})).await;
                                }
                                NetCommand::UseItem { slot } => {
                                    let _ = client.send_command("player", "use_item",
                                        serde_json::json!({"slot": slot})).await;
                                }
                            },
                            Err(crossbeam_channel::TryRecvError::Empty) => {
                                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                            }
                            Err(crossbeam_channel::TryRecvError::Disconnected) => break,
                        }
                    }
                });

                let _ = tokio::join!(relay_events, relay_commands);
            });
        });

        NetworkBridge { event_rx, cmd_tx, status_rx }
    }
}

// ---------------------------------------------------------------------------
// Game Plugin
// ---------------------------------------------------------------------------

#[derive(Default, Visit, Reflect, Debug)]
#[reflect(non_cloneable)]
pub struct Game {
    scene: Handle<Scene>,
    camera: Handle<Node>,
    #[visit(skip)] #[reflect(hidden)]
    mouse_world_pos: Vector2<f32>,
    #[visit(skip)] #[reflect(hidden)]
    window_size: Vector2<f32>,

    // --- Network ---
    #[visit(skip)] #[reflect(hidden)]
    network: Option<NetworkBridge>,
    #[visit(skip)] #[reflect(hidden)]
    connection_status: ConnectionStatus,
    #[visit(skip)] #[reflect(hidden)]
    event_buffer: Option<EventBuffer>,
    #[visit(skip)] #[reflect(hidden)]
    network_entities: HashMap<u32, NetworkEntity>,
    #[visit(skip)] #[reflect(hidden)]
    client_projectiles: HashMap<u32, ClientProjectile>,
    #[visit(skip)] #[reflect(hidden)]
    heartbeat: HeartbeatInfo,

    // --- Backend Process ---
    /// Drops → kills backend. Held for the whole Game lifetime so that any exit
    /// path (normal, panic, force-close on Windows via Job Object) brings the
    /// backend down with us.
    #[visit(skip)] #[reflect(hidden)]
    backend_guard: Option<BackendGuard>,

    #[visit(skip)] #[reflect(hidden)]
    pending_label_deletions: Vec<Handle<Text>>,

    // --- UI ---
    #[visit(skip)] #[reflect(hidden)]
    ui_status_text: Handle<Text>,
    #[visit(skip)] #[reflect(hidden)]
    ui_hud_text: Handle<Text>,
    #[visit(skip)] #[reflect(hidden)]
    ui_ability_icons: [Handle<UiNode>; 4],
    #[visit(skip)] #[reflect(hidden)]
    ui_ability_level_text: [Handle<Text>; 4],
    /// 冷卻中央大數字
    #[visit(skip)] #[reflect(hidden)]
    ui_ability_cd_text: [Handle<Text>; 4],
    /// 快捷鍵 cap [W] [E] [R] [T]
    #[visit(skip)] #[reflect(hidden)]
    ui_ability_key_text: [Handle<Text>; 4],
    /// 4 技能圖片資源（HUD icon + tooltip icon 共用）
    #[visit(skip)] #[reflect(hidden)]
    ability_textures: [Option<fyrox::resource::texture::TextureResource>; 4],
    /// 4 icon 的 screen AABB (x, y, w, h) — 供滑鼠 hit-test
    #[visit(skip)] #[reflect(hidden)]
    ability_icon_rects: [(f32, f32, f32, f32); 4],
    /// 技能詳細資訊 map（key = ability id），由 hero.abilities_info 事件填入
    #[visit(skip)] #[reflect(hidden)]
    ability_info_map: HashMap<String, AbilityInfo>,
    /// 原始滑鼠螢幕座標（pixel）
    #[visit(skip)] #[reflect(hidden)]
    mouse_screen_pos: Vector2<f32>,
    /// 目前 hover 的 ability slot index（0-3）
    #[visit(skip)] #[reflect(hidden)]
    hovered_ability: Option<usize>,
    #[visit(skip)] #[reflect(hidden)]
    ui_tooltip_bg: Handle<UiNode>,
    #[visit(skip)] #[reflect(hidden)]
    ui_tooltip_icon: Handle<UiNode>,
    #[visit(skip)] #[reflect(hidden)]
    ui_tooltip_text: Handle<Text>,
    #[visit(skip)] #[reflect(hidden)]
    ui_shop_text: Handle<Text>,
    #[visit(skip)] #[reflect(hidden)]
    ui_end_text: Handle<Text>,

    // --- LoL MVP: Local Hero state cached from hero.* events ---
    #[visit(skip)] #[reflect(hidden)]
    hero_state: LocalHeroState,
    #[visit(skip)] #[reflect(hidden)]
    shop_visible: bool,
    #[visit(skip)] #[reflect(hidden)]
    shift_held: bool,
    #[visit(skip)] #[reflect(hidden)]
    game_ended: bool,
    #[visit(skip)] #[reflect(hidden)]
    viewport_sync_elapsed: f32,
    /// Camera 目前所在 render-world 座標（用於滑鼠座標換算與 label 螢幕換算）
    #[visit(skip)] #[reflect(hidden)]
    camera_world_pos: Vector2<f32>,
    /// 本秒累計的網路事件 payload bytes
    #[visit(skip)] #[reflect(hidden)]
    net_bytes_current: u64,
    /// 上一秒的總 bytes，供顯示用
    #[visit(skip)] #[reflect(hidden)]
    net_bytes_last_sec: u64,
    /// 計時：每滿 1 秒 roll over
    #[visit(skip)] #[reflect(hidden)]
    net_stats_elapsed: f32,
}

/// 技能詳細資訊（後端一次性廣播，用於 tooltip）
#[derive(Debug, Clone, Default)]
struct AbilityInfo {
    id: String,
    name: String,
    description: String,
    key_binding: String,
    max_level: i32,
    cooldown: Vec<f32>,
    mana_cost: Vec<i32>,
    cast_range: Vec<f32>,
    effects: HashMap<String, serde_json::Value>,
}

/// 前端緩存的 hero 狀態（由 hero.stats / hero.inventory 事件驅動）
#[derive(Default, Debug, Clone)]
struct LocalHeroState {
    /// 英雄在後端的 entity id，camera 跟隨用
    entity_id: Option<u32>,
    level: i32,
    xp: i32,
    xp_next: i32,
    skill_points: i32,
    gold: i32,
    hp: f32,
    max_hp: f32,
    abilities: Vec<String>,          // ability ids, index 0=Q, 1=W, 2=E, 3=R
    ability_levels: HashMap<String, i32>,
    /// 技能剩餘冷卻秒數（key = ability id），本地遞減
    ability_cd: HashMap<String, f32>,
    /// 6 個 slot，每個 (item_id, cd)
    inventory: Vec<Option<(String, f32)>>,
}

/// MVP 商店清單（前端固定順序，對應後端 item id）
const SHOP_ITEMS: &[(&str, &str, i32)] = &[
    ("dmg_sword",    "長劍",       500),
    ("dmg_rifle",    "無雙鐵炮",   1600),
    ("hp_vest",      "皮甲",       450),
    ("hp_armor",     "重裝甲",     1400),
    ("mp_orb",       "法力珠",     400),
    ("mp_staff",     "秘法杖",     1200),
    ("ms_boots",     "戰靴",       400),
    ("ms_swift",     "疾風之靴",   1300),
    ("def_plate",    "鎖子甲",     500),
    ("def_bulwark",  "堡壘之盾",   1500),
];

impl Plugin for Game {
    fn register(&self, _context: PluginRegistrationContext) -> GameResult {
        Ok(())
    }

    fn init(&mut self, _scene_path: Option<&str>, mut context: PluginContext) -> GameResult {
        self.window_size = Vector2::new(800.0, 600.0);

        let mut scene = Scene::new();

        // Remove the default built-in skybox (shows as a blue/white gradient behind 2D content)
        scene.set_skybox(None);

        // 2D rendering options
        use fyrox::scene::SceneRenderingOptions;
        scene.rendering_options.set_value_and_mark_modified(SceneRenderingOptions {
            clear_color: Some(Color::from_rgba(30, 80, 30, 255)),
            ambient_lighting_color: Color::WHITE,
            environment_lighting_source: EnvironmentLightingSource::AmbientColor,
            environment_lighting_brightness: 1.0,
            ..Default::default()
        });

        // Orthographic 2D camera
        self.camera = CameraBuilder::new(BaseBuilder::new())
            .with_projection(Projection::Orthographic(OrthographicProjection {
                z_near: -0.1,
                z_far: 16.0,
                vertical_size: 10.0,
            }))
            .build(&mut scene.graph)
            .transmute();

        // Point light covering entire map
        PointLightBuilder::new(
            BaseLightBuilder::new(
                BaseBuilder::new().with_local_transform(
                    TransformBuilder::new()
                        .with_local_position(Vector3::new(0.0, 0.0, 0.0))
                        .build(),
                ),
            )
            .with_scatter_enabled(false),
        )
        .with_radius(40.0)
        .build(&mut scene.graph);

        // Background (dark green)
        RectangleBuilder::new(
            BaseBuilder::new().with_local_transform(
                TransformBuilder::new()
                    .with_local_position(Vector3::new(0.0, 0.0, Z_BACKGROUND))
                    .with_local_scale(Vector3::new(30.0, 22.0, f32::EPSILON))
                    .build(),
            ),
        )
        .with_color(Color::from_rgba(30, 80, 30, 255))
        .build(&mut scene.graph);

        self.scene = context.scenes.add(scene);

        // UI: status text
        context
            .user_interfaces
            .add(UserInterface::new(Default::default()));
        let ui = context.user_interfaces.first_mut();

        // Load CJK font (Microsoft JhengHei) for Chinese text rendering
        if let Ok(font_data) = std::fs::read("C:/Windows/Fonts/msjh.ttc") {
            use fyrox::gui::font::{Font, FontResource, FontStyles};
            use fyrox::asset::untyped::ResourceKind;
            use fyrox::core::uuid::Uuid;
            if let Ok(font) = Font::from_memory(font_data, 1024, FontStyles::default(), vec![]) {
                let font_resource = FontResource::new_ok(
                    Uuid::new_v4(),
                    ResourceKind::Embedded,
                    font,
                );
                ui.default_font = font_resource;
            }
        }

        // 載入孫市四技能圖示（hero1_1..4）
        // 用 Fyrox UI 內建 check.png / add.png 等的同套 pattern：
        //   TextureResource::load_from_memory + CompressionOptions::NoCompression
        // 壓縮 texture 會造成 UI renderer 拿不到可顯示的 GPU 格式 → 空白。
        // 實際位置在 update() 依當前 window_size 置底中央。
        {
            use fyrox::asset::untyped::ResourceKind;
            use fyrox::core::uuid::Uuid;
            use fyrox::resource::texture::{
                CompressionOptions, TextureImportOptions, TextureMinificationFilter,
                TextureResource, TextureResourceExtension,
            };

            let slot_label = ["W", "E", "R", "T"];
            let icon_size = 64.0f32;

            for i in 0..4 {
                let path = format!("data/hero1_{}.png", i + 1);
                let init_x = 500.0 + (i as f32) * 72.0;
                let init_y = 620.0;
                self.ability_icon_rects[i] = (init_x, init_y, icon_size, icon_size);

                let mut debug_state: &'static str = "?";
                // 路徑候選：
                //   1. data/... （CWD=omfx 時，cargo run -p omfx）
                //   2. omfx/data/... （CWD=repo root 時，cargo run --manifest-path omfx\Cargo.toml）
                //   3. ../data/... （備援，CWD 深一層時）
                //   4. {exe_dir}/data/... （打包後 release，exe 與 data 同層）
                let mut candidate_paths: Vec<String> = vec![
                    path.clone(),
                    format!("omfx/{}", path),
                    format!("../{}", path),
                ];
                if let Ok(exe_path) = std::env::current_exe() {
                    if let Some(exe_dir) = exe_path.parent() {
                        candidate_paths.push(
                            exe_dir.join(&path).to_string_lossy().into_owned(),
                        );
                    }
                }
                let read_result = candidate_paths.iter().find_map(|p| {
                    std::fs::read(p).ok().map(|b| (p.clone(), b))
                });
                let texture_opt: Option<TextureResource> = match read_result.as_ref().map(|(_, b)| b) {
                    Some(bytes) => {
                        let opts = TextureImportOptions::default()
                            .with_compression(CompressionOptions::NoCompression)
                            .with_minification_filter(TextureMinificationFilter::LinearMipMapLinear);
                        match TextureResource::load_from_memory(
                            Uuid::new_v4(),
                            ResourceKind::Embedded,
                            bytes,
                            opts,
                        ) {
                            Ok(res) => {
                                debug_state = "OK";
                                Some(res)
                            }
                            Err(_) => {
                                debug_state = "DECODE";
                                None
                            }
                        }
                    }
                    None => {
                        debug_state = "READ";
                        None
                    }
                };

                let _ = debug_state; // reserved for future logging
                let icon_handle: Handle<UiNode> = if let Some(ref resource) = texture_opt {
                    let h: Handle<fyrox::gui::image::Image> = ImageBuilder::new(
                        WidgetBuilder::new()
                            .with_desired_position(Vector2::new(init_x, init_y))
                            .with_width(icon_size)
                            .with_height(icon_size),
                    )
                    .with_texture(resource.clone())
                    .build(&mut ui.build_ctx());
                    h.transmute()
                } else {
                    Handle::NONE
                };
                self.ui_ability_icons[i] = icon_handle;
                self.ability_textures[i] = texture_opt;

                // Icon 下方顯示等級點（● ○ ○ ○ ○）—— 由 update() 每 frame 更新文字
                let lvl = TextBuilder::new(
                    WidgetBuilder::new()
                        .with_desired_position(Vector2::new(init_x, init_y + 64.0))
                        .with_width(64.0)
                        .with_foreground(Brush::Solid(Color::from_rgba(0, 0, 0, 255)).into()),
                )
                .with_text("○ ○ ○ ○ ○".to_string())
                .with_font_size(12.0.into())
                .build(&mut ui.build_ctx());
                self.ui_ability_level_text[i] = lvl;

                // Icon 上方顯示快捷鍵 cap — T 槽（終極）金色加星，其餘白色
                let is_ultimate = i == 3; // T
                let key_color = if is_ultimate {
                    Color::from_rgba(255, 210, 40, 255)
                } else {
                    Color::from_rgba(240, 240, 240, 255)
                };
                let key_str = if is_ultimate {
                    format!("★ {} ★", slot_label[i])
                } else {
                    format!("[{}]", slot_label[i])
                };
                let key = TextBuilder::new(
                    WidgetBuilder::new()
                        .with_desired_position(Vector2::new(init_x + 12.0, init_y - 20.0))
                        .with_width(60.0)
                        .with_foreground(Brush::Solid(key_color).into()),
                )
                .with_text(key_str)
                .with_font_size(16.0.into())
                .build(&mut ui.build_ctx());
                self.ui_ability_key_text[i] = key;

                // Icon 中央的冷卻大數字（CD 結束時清空）
                let cd = TextBuilder::new(
                    WidgetBuilder::new()
                        .with_desired_position(Vector2::new(init_x + 12.0, init_y + 14.0))
                        .with_width(40.0)
                        .with_foreground(Brush::Solid(Color::from_rgba(255, 255, 255, 255)).into()),
                )
                .with_text("".to_string())
                .with_font_size(32.0.into())
                .build(&mut ui.build_ctx());
                self.ui_ability_cd_text[i] = cd;
            }

            // Tooltip：icon + text，初始位置在螢幕外（隱藏）
            // 背景先跳過（text 黑字在淺綠背景已夠清楚），未來有需要再加
            self.ui_tooltip_bg = Handle::NONE;
            self.ui_tooltip_icon = {
                let h: Handle<fyrox::gui::image::Image> = ImageBuilder::new(
                    WidgetBuilder::new()
                        .with_desired_position(Vector2::new(-9999.0, -9999.0))
                        .with_width(80.0)
                        .with_height(80.0),
                )
                .build(&mut ui.build_ctx());
                h.transmute()
            };
            self.ui_tooltip_text = TextBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(-9999.0, -9999.0))
                    .with_width(360.0)
                    .with_foreground(Brush::Solid(Color::from_rgba(0, 0, 0, 255)).into()),
            )
            .with_text("".to_string())
            .with_font_size(14.0.into())
            .build(&mut ui.build_ctx());
        }

        self.ui_status_text = TextBuilder::new(
            WidgetBuilder::new()
                .with_desired_position(Vector2::new(10.0, 10.0))
                .with_foreground(Brush::Solid(Color::from_rgba(0, 0, 0, 255)).into()),
        )
        .with_text("Connecting...".to_string())
        .with_font_size(18.0.into())
        .build(&mut ui.build_ctx());

        // LoL MVP HUD 文字（底部一行）
        self.ui_hud_text = TextBuilder::new(
            WidgetBuilder::new()
                .with_desired_position(Vector2::new(10.0, 540.0))
                .with_width(1900.0)
                .with_foreground(Brush::Solid(Color::from_rgba(0, 0, 0, 255)).into()),
        )
        .with_text("".to_string())
        .with_font_size(18.0.into())
        .build(&mut ui.build_ctx());

        // 商店面板（初始空字串；按 B 切換顯示內容）
        self.ui_shop_text = TextBuilder::new(
            WidgetBuilder::new()
                .with_desired_position(Vector2::new(40.0, 80.0))
                .with_width(500.0)
                .with_foreground(Brush::Solid(Color::from_rgba(0, 0, 0, 255)).into()),
        )
        .with_text("".to_string())
        .with_font_size(20.0.into())
        .build(&mut ui.build_ctx());

        // 結束 overlay（初始隱藏，以空文字表達）
        self.ui_end_text = TextBuilder::new(
            WidgetBuilder::new()
                .with_desired_position(Vector2::new(600.0, 250.0))
                .with_width(800.0)
                .with_foreground(Brush::Solid(Color::from_rgba(0, 0, 0, 255)).into()),
        )
        .with_text("".to_string())
        .with_font_size(72.0.into())
        .build(&mut ui.build_ctx());

        // Inventory 初始 6 格
        self.hero_state.inventory = vec![None; 6];

        // Auto-start backend (tied to our lifetime via BackendGuard + Job Object)
        self.backend_guard = spawn_backend();

        // Network init
        let server_addr = std::env::var("OMB_KCP_ADDR")
            .unwrap_or_else(|_| "127.0.0.1:50061".to_string());
        let player_name = std::env::var("OMB_PLAYER_NAME")
            .unwrap_or_else(|_| "omfx_player".to_string());

        self.network = Some(NetworkBridge::spawn(server_addr, player_name));
        self.connection_status = ConnectionStatus::Connecting;
        self.event_buffer = Some(EventBuffer::new());
        self.network_entities = HashMap::new();
        self.client_projectiles = HashMap::new();

        Ok(())
    }

    fn on_deinit(&mut self, _context: PluginContext) -> GameResult {
        // Drop network bridge (threads will stop when channels disconnect)
        self.network = None;

        // Drop the backend guard — its Drop impl kills the child and closes the Job Object.
        // (If Drop doesn't run, e.g. on hard kill, the OS still terminates the backend
        // thanks to JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE.)
        self.backend_guard = None;

        Ok(())
    }

    fn update(&mut self, context: &mut PluginContext) -> GameResult {
        let scene = &mut context.scenes[self.scene];

        // 1. Check connection status
        if let Some(ref network) = self.network {
            while let Ok(status) = network.status_rx.try_recv() {
                if status == ConnectionStatus::Connected && self.connection_status != ConnectionStatus::Connected {
                    // Send initial viewport on first connect
                    let aspect = self.window_size.x / self.window_size.y;
                    let half_height = 10.0 / WORLD_SCALE; // vertical_size in game coords
                    let half_width = 10.0 * aspect / WORLD_SCALE;
                    let _ = network.cmd_tx.send(NetCommand::ViewportUpdate {
                        cx: 0.0, cy: 0.0, hw: half_width, hh: half_height,
                    });
                }
                self.connection_status = status;
            }
        }

        // 網路流量統計：每秒 roll over
        self.net_stats_elapsed += context.dt;
        if self.net_stats_elapsed >= 1.0 {
            self.net_bytes_last_sec = self.net_bytes_current;
            self.net_bytes_current = 0;
            self.net_stats_elapsed -= 1.0;
        }

        // 2. Receive events from NetworkBridge, push into EventBuffer
        if let (Some(ref network), Some(ref mut buffer)) = (&self.network, &mut self.event_buffer) {
            let mut pending_hp_sync: Option<serde_json::Value> = None;
            for evt in network.event_rx.try_iter() {
                // 估算 event 位元數：msg_type + action + JSON payload + 固定 overhead (~16 for timestamp)
                let payload_bytes = serde_json::to_string(&evt.data).map(|s| s.len()).unwrap_or(0);
                self.net_bytes_current += (evt.msg_type.len() + evt.action.len() + payload_bytes + 16) as u64;
                // Heartbeat: calibrate clock immediately, don't buffer
                if evt.msg_type == "heartbeat" && evt.action == "tick" {
                    buffer.sync_clock(evt.timestamp_ms);
                    if let Some(delay) = evt.data.get("render_delay_ms").and_then(|v| v.as_u64()) {
                        buffer.render_delay_ms = delay;
                    }
                    self.heartbeat = parse_heartbeat(&evt.data);
                    if let Some(snap) = evt.data.get("hp_snapshot").cloned() {
                        pending_hp_sync = Some(snap);
                    }
                } else {
                    buffer.push(evt);
                }
            }

            // 3. Drain ready events from buffer and render
            for evt in buffer.drain_ready() {
                self.apply_event(evt, scene);
            }

            // Heartbeat HP reconciliation: overwrite client-predicted HP with the
            // authoritative backend snapshot (corrects drift accumulated from
            // optimistic damage prediction + missed/overshot hits).
            if let Some(snap) = pending_hp_sync {
                if let Some(arr) = snap.as_array() {
                    for item in arr {
                        let id = item.get("id").and_then(|v| v.as_u64()).map(|v| v as u32);
                        let hp = item.get("hp").and_then(|v| v.as_f64()).map(|v| v as f32);
                        let max_hp = item.get("max_hp").and_then(|v| v.as_f64()).map(|v| v as f32);
                        if let (Some(id), Some(h), Some(m)) = (id, hp, max_hp) {
                            if let Some(entity) = self.network_entities.get_mut(&id) {
                                entity.health = Some((h, m));
                            }
                        }
                    }
                }
            }
        }

        // 4. Interpolate entity positions (client-side lerp)
        let dt = context.dt;
        for entity in self.network_entities.values_mut() {
            // Expire creep debug path after PATH_VISIBLE_SECS
            if !entity.path_nodes.is_empty() {
                entity.path_age += dt;
                if entity.path_age >= PATH_VISIBLE_SECS {
                    for seg in entity.path_nodes.drain(..) {
                        scene.graph.remove_node(seg);
                    }
                }
            }

            entity.lerp_elapsed += dt;
            let t = (entity.lerp_elapsed / entity.lerp_duration).clamp(0.0, 1.0);
            let pos = entity.prev_position.lerp(&entity.target_position, t);
            entity.position = pos;

            // Update node position (keep same Z) — X 取負讓 +X world 投到螢幕右
            let z = scene.graph[entity.node].local_transform().position().z;
            scene.graph[entity.node]
                .local_transform_mut()
                .set_position(Vector3::new(-pos.x, pos.y, z));

            // Update HP bar positions
            if let (Some(bg), Some(fg), Some((h, m))) = (entity.hp_bar_bg, entity.hp_bar_fg, entity.health) {
                let bar_y = pos.y + 0.3;
                let hp_ratio = (h / m).clamp(0.0, 1.0);
                let bar_width = 0.8;

                scene.graph[bg]
                    .local_transform_mut()
                    .set_position(Vector3::new(-pos.x, bar_y, Z_HP_BAR));

                let fg_width = bar_width * hp_ratio;
                let fg_offset = (bar_width - fg_width) * 0.5;
                // 注意 fg_offset 在 X 方向偏移；X 翻轉後 offset 也要反向
                scene.graph[fg]
                    .local_transform_mut()
                    .set_position(Vector3::new(-pos.x - fg_offset, bar_y, Z_HP_BAR - 0.0001))
                    .set_scale(Vector3::new(fg_width, 0.06, f32::EPSILON));
            }

            // 更新面向箭頭位置與角度
            if let Some(arrow) = entity.facing_arrow {
                let render_angle = std::f32::consts::PI - entity.facing;
                let scale = scene.graph[arrow].local_transform().scale();
                let length = scale.x;
                let offset_x = (length * 0.5) * render_angle.cos();
                let offset_y = (length * 0.5) * render_angle.sin();
                let rotation = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), render_angle);
                scene.graph[arrow]
                    .local_transform_mut()
                    .set_position(Vector3::new(-pos.x + offset_x, pos.y + offset_y, 0.0007))
                    .set_rotation(rotation);
            }
        }

        // 4b. Advance client-simulated projectiles (pursuit lerp toward target's
        //     current interpolated position; t forced to 1 at flight_time).
        //     後端改為 100ms batch 發送，client flight_time 與 backend projectile time 已對齊
        //     (game_processor.rs 裡用 initial_dist / bullet_speed 設 safety_time_left 的 1/3)，
        //     所以彈落時 optimistic 扣血與 100ms 內到達的 backend "H" 事件幾乎 sync，不會 bouncing。
        let mut finished: Vec<u32> = Vec::new();
        let mut predicted_damage: Vec<(u32, f32)> = Vec::new();
        for (id, proj) in self.client_projectiles.iter_mut() {
            proj.elapsed += dt;
            let t = (proj.elapsed / proj.flight_time).clamp(0.0, 1.0);
            let target_pos = self
                .network_entities
                .get(&proj.target_id)
                .map(|e| e.position)
                .unwrap_or(proj.last_target_pos);
            proj.last_target_pos = target_pos;
            let pos = proj.start_pos + (target_pos - proj.start_pos) * t;
            scene.graph[proj.node]
                .local_transform_mut()
                .set_position(Vector3::new(-pos.x, pos.y, Z_BULLET));
            if t >= 1.0 {
                if !proj.applied && proj.damage > 0.0 {
                    predicted_damage.push((proj.target_id, proj.damage));
                    proj.applied = true;
                }
                finished.push(*id);
            }
        }
        // Optimistic 扣血：等後續 backend "H" 事件（約 0~100ms 內）reconcile
        for (target_id, dmg) in predicted_damage {
            if let Some(entity) = self.network_entities.get_mut(&target_id) {
                if let Some((h, m)) = entity.health {
                    let new_h = (h - dmg).max(0.0);
                    entity.health = Some((new_h, m));
                }
            }
        }
        for id in finished {
            if let Some(proj) = self.client_projectiles.remove(&id) {
                scene.graph.remove_node(proj.node);
            }
        }

        // 4c. Camera follow hero
        if let Some(hero_id) = self.hero_state.entity_id {
            if let Some(hero_ent) = self.network_entities.get(&hero_id) {
                let pos = hero_ent.position;
                let z = scene.graph[self.camera].local_transform().position().z;
                // 渲染時 X 負號：Fyrox 預設 +X 到螢幕左，我們希望 +X 到右，所以 entity/camera X 都反向
                scene.graph[self.camera]
                    .local_transform_mut()
                    .set_position(Vector3::new(-pos.x, pos.y, z));
                self.camera_world_pos = pos;

                // 週期性同步 viewport 給後端（~2 Hz）
                self.viewport_sync_elapsed += dt;
                if self.viewport_sync_elapsed >= 0.5 {
                    self.viewport_sync_elapsed = 0.0;
                    if let Some(ref network) = self.network {
                        let aspect = self.window_size.x / self.window_size.y.max(1.0);
                        let half_height = 10.0 / WORLD_SCALE;
                        let half_width = 10.0 * aspect / WORLD_SCALE;
                        let _ = network.cmd_tx.send(NetCommand::ViewportUpdate {
                            cx: pos.x / WORLD_SCALE,
                            cy: pos.y / WORLD_SCALE,
                            hw: half_width,
                            hh: half_height,
                        });
                    }
                }
            }
        }

        // 5. Update name labels (UI layer)
        let ui = context.user_interfaces.first_mut();
        let win = self.window_size;

        // Delete labels for removed entities
        for label in self.pending_label_deletions.drain(..) {
            ui.send(label, WidgetMessage::Remove);
        }

        // Create missing labels & update positions
        for entity in self.network_entities.values_mut() {
            if entity.health.is_none() {
                continue; // only show names for entities with HP bars
            }

            // Lazily create label
            if entity.name_label.is_none() {
                let label = TextBuilder::new(
                    WidgetBuilder::new()
                        .with_desired_position(Vector2::new(0.0, 0.0))
                        .with_width(180.0)
                        .with_foreground(Brush::Solid(Color::from_rgba(0, 0, 0, 255)).into()),
                )
                .with_text(entity.name.clone())
                .with_font_size(21.0.into())
                .with_horizontal_text_alignment(HorizontalAlignment::Center)
                .build(&mut ui.build_ctx());
                entity.name_label = Some(label);
            }

            // Update label screen position (above HP bar) + 文字含 HP 數字
            if let Some(label) = entity.name_label {
                let name_world_y = entity.position.y + 0.5;
                let screen_pos = world_to_screen_approx(
                    entity.position.x - self.camera_world_pos.x,
                    name_world_y - self.camera_world_pos.y,
                    win.x, win.y,
                );
                let pos = Vector2::new(screen_pos.x - 90.0, screen_pos.y - 24.0);
                ui.send(label, WidgetMessage::DesiredPosition(pos));

                // 顯示「名字 HP/MaxHP」讓 HP bouncing 肉眼可見
                let text = match entity.health {
                    Some((h, m)) => format!("{} {:.0}/{:.0}", entity.name, h, m),
                    None => entity.name.clone(),
                };
                ui.send(label, TextMessage::Text(text));
            }
        }

        // 6. Update status text
        let status_str = match &self.connection_status {
            ConnectionStatus::Disconnected => "Disconnected".to_string(),
            ConnectionStatus::Connecting => "Connecting...".to_string(),
            ConnectionStatus::Connected => {
                let bps = self.net_bytes_last_sec;
                let net_str = if bps >= 1_000_000 {
                    format!("{:.2} MB/s", bps as f64 / 1_000_000.0)
                } else if bps >= 1_000 {
                    format!("{:.1} KB/s", bps as f64 / 1_000.0)
                } else {
                    format!("{} B/s", bps)
                };
                format!(
                    "Connected | Tick: {} | Time: {:.1} | Entities: {} | Heroes: {} | Creeps: {} | Net: {}",
                    self.heartbeat.tick,
                    self.heartbeat.game_time,
                    self.heartbeat.entity_count,
                    self.heartbeat.hero_count,
                    self.heartbeat.creep_count,
                    net_str,
                )
            }
            ConnectionStatus::Failed(e) => format!("Failed: {}", e),
        };
        ui.send(self.ui_status_text, TextMessage::Text(status_str));

        // LoL MVP HUD: 本地 CD 平滑遞減 + 組 HUD 文字
        {
            for slot in self.hero_state.inventory.iter_mut() {
                if let Some((_, cd)) = slot.as_mut() {
                    if *cd > 0.0 {
                        *cd = (*cd - dt).max(0.0);
                    }
                }
            }
            // ===== 依當前 window_size 置底中央定位 4 個技能 icon =====
            {
                let icon_size = 64.0f32;
                let spacing = 72.0f32;
                let total_w = spacing * 3.0 + icon_size;
                let base_x = (self.window_size.x - total_w) * 0.5;
                let icon_y = self.window_size.y - icon_size - 32.0;
                for i in 0..4 {
                    let x = base_x + (i as f32) * spacing;
                    self.ability_icon_rects[i] = (x, icon_y, icon_size, icon_size);
                    if self.ui_ability_icons[i] != Handle::<UiNode>::NONE {
                        ui.send(
                            self.ui_ability_icons[i],
                            WidgetMessage::DesiredPosition(Vector2::new(x, icon_y)),
                        );
                    }
                    if self.ui_ability_level_text[i] != Handle::<Text>::NONE {
                        ui.send(
                            self.ui_ability_level_text[i],
                            WidgetMessage::DesiredPosition(Vector2::new(x, icon_y + icon_size)),
                        );
                    }
                    if self.ui_ability_key_text[i] != Handle::<Text>::NONE {
                        ui.send(
                            self.ui_ability_key_text[i],
                            WidgetMessage::DesiredPosition(Vector2::new(x + 20.0, icon_y - 18.0)),
                        );
                    }
                    if self.ui_ability_cd_text[i] != Handle::<Text>::NONE {
                        ui.send(
                            self.ui_ability_cd_text[i],
                            WidgetMessage::DesiredPosition(Vector2::new(x + 12.0, icon_y + 14.0)),
                        );
                    }
                }
            }

            // 技能冷卻每 frame 遞減
            for cd in self.hero_state.ability_cd.values_mut() {
                if *cd > 0.0 { *cd = (*cd - dt).max(0.0); }
            }

            let hs = &self.hero_state;
            // 更新技能 icon 下方的等級點 + 中央 CD 數字
            for i in 0..4 {
                let id = hs.abilities.get(i).cloned().unwrap_or_default();
                let lvl = hs.ability_levels.get(&id).copied().unwrap_or(0);
                let max = self.ability_info_map.get(&id).map(|a| a.max_level).unwrap_or(4);
                // 等級點 ● ○
                let dots: String = (0..max.max(1))
                    .map(|n| if n < lvl { "●" } else { "○" })
                    .collect::<Vec<_>>()
                    .join(" ");
                ui.send(self.ui_ability_level_text[i], TextMessage::Text(dots));

                // CD 數字
                let remaining = hs.ability_cd.get(&id).copied().unwrap_or(0.0);
                let cd_str = if remaining >= 1.0 {
                    format!("{:.0}", remaining.ceil())
                } else if remaining > 0.0 {
                    format!("{:.1}", remaining)
                } else {
                    String::new()
                };
                ui.send(self.ui_ability_cd_text[i], TextMessage::Text(cd_str));
            }
            // Inventory 顯示
            let mut inv = String::new();
            for (i, slot) in hs.inventory.iter().enumerate() {
                match slot {
                    Some((id, cd)) => {
                        if *cd > 0.0 {
                            inv.push_str(&format!("[{}]{}({:.0}s) ", i + 1, id, cd));
                        } else {
                            inv.push_str(&format!("[{}]{} ", i + 1, id));
                        }
                    }
                    None => inv.push_str(&format!("[{}]- ", i + 1)),
                }
            }
            let hud = format!(
                "HP {:.0}/{:.0}  LV {}  XP {}/{}  GOLD {}  SP {}  |  {}",
                hs.hp, hs.max_hp, hs.level, hs.xp, hs.xp_next, hs.gold, hs.skill_points, inv,
            );
            ui.send(self.ui_hud_text, TextMessage::Text(hud));

            // 商店顯示 / 隱藏
            let shop = if self.shop_visible {
                let mut s = String::from("=== SHOP (按 B 關閉) ===\n");
                for (i, (id, name, cost)) in SHOP_ITEMS.iter().enumerate() {
                    s.push_str(&format!("{}. {} ({}) — {}g\n", i, name, id, cost));
                }
                s.push_str("按 0-9 購買對應編號裝備（需靠近基地）");
                s
            } else {
                String::new()
            };
            ui.send(self.ui_shop_text, TextMessage::Text(shop));

            let end_str = if self.game_ended { "VICTORY!".to_string() } else { String::new() };
            ui.send(self.ui_end_text, TextMessage::Text(end_str));

            // ===== 技能 tooltip hit-test + 更新 =====
            let mouse = self.mouse_screen_pos;
            let mut new_hover: Option<usize> = None;
            for (i, rect) in self.ability_icon_rects.iter().enumerate() {
                let (rx, ry, rw, rh) = *rect;
                if mouse.x >= rx && mouse.x <= rx + rw && mouse.y >= ry && mouse.y <= ry + rh {
                    new_hover = Some(i);
                    break;
                }
            }
            // 只在 hover 變化時才 rebuild tooltip，並每 frame 重新定位
            if new_hover != self.hovered_ability {
                self.hovered_ability = new_hover;
                match new_hover {
                    Some(idx) => {
                        // 更新 tooltip icon texture
                        if let Some(tex) = self.ability_textures[idx].as_ref() {
                            ui.send(self.ui_tooltip_icon, ImageMessage::Texture(Some(tex.clone())));
                        }
                        // 查 ability info 組 tooltip 字串
                        let hs = &self.hero_state;
                        let ability_id = hs.abilities.get(idx).cloned().unwrap_or_default();
                        let cur_lvl = hs.ability_levels.get(&ability_id).copied().unwrap_or(0);
                        let tooltip_str = if let Some(info) = self.ability_info_map.get(&ability_id) {
                            format_ability_tooltip(info, cur_lvl)
                        } else {
                            format!("(尚未收到技能資訊)\nSlot {}", idx)
                        };
                        ui.send(self.ui_tooltip_text, TextMessage::Text(tooltip_str));
                    }
                    None => {
                        // 隱藏：文字清空 + 移到螢幕外
                        ui.send(self.ui_tooltip_text, TextMessage::Text(String::new()));
                        ui.send(
                            self.ui_tooltip_icon,
                            WidgetMessage::DesiredPosition(Vector2::new(-9999.0, -9999.0)),
                        );
                        ui.send(
                            self.ui_tooltip_text,
                            WidgetMessage::DesiredPosition(Vector2::new(-9999.0, -9999.0)),
                        );
                    }
                }
            }
            // 每 frame 定位（若有 hover）
            if self.hovered_ability.is_some() {
                let mut tx = mouse.x + 16.0;
                let mut ty = mouse.y - 190.0;
                if tx + 460.0 > win.x { tx = (win.x - 460.0).max(0.0); }
                if ty < 0.0 { ty = 0.0; }
                ui.send(
                    self.ui_tooltip_icon,
                    WidgetMessage::DesiredPosition(Vector2::new(tx, ty)),
                );
                ui.send(
                    self.ui_tooltip_text,
                    WidgetMessage::DesiredPosition(Vector2::new(tx + 88.0, ty)),
                );
            }
        }

        Ok(())
    }

    fn on_os_event(&mut self, event: &Event<()>, mut context: PluginContext) -> GameResult {
        match event {
            Event::WindowEvent {
                event: WindowEvent::Resized(size),
                ..
            } => {
                self.window_size = Vector2::new(size.width as f32, size.height as f32);
            }
            Event::WindowEvent {
                event: WindowEvent::CursorMoved { position, .. },
                ..
            } => {
                let local = screen_to_world_approx(
                    position.x as f32,
                    position.y as f32,
                    self.window_size.x,
                    self.window_size.y,
                    20.0, // matches camera vertical_size * 2
                );
                // 加上 camera 位移，得到絕對 render-world 座標
                self.mouse_world_pos = local + self.camera_world_pos;
                // 原始 pixel 座標，供 tooltip hit-test 用
                self.mouse_screen_pos = Vector2::new(position.x as f32, position.y as f32);
            }
            Event::WindowEvent {
                event:
                    WindowEvent::MouseInput {
                        button: MouseButton::Left,
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                let world_pos = self.mouse_world_pos;
                if let Some((col, row)) = world_to_grid(world_pos.x, world_pos.y) {
                    let (cx, cy) = grid_to_world(col, row);
                    if let Some(ref network) = self.network {
                        let _ = network.cmd_tx.send(NetCommand::PlaceTower { x: cx / WORLD_SCALE, y: cy / WORLD_SCALE });
                    }
                }
            }
            // Right click → hero move (world coordinates, no grid snap)
            Event::WindowEvent {
                event:
                    WindowEvent::MouseInput {
                        button: MouseButton::Right,
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                let world_pos = self.mouse_world_pos;
                if let Some(ref network) = self.network {
                    // Convert render coords back to backend coords
                    let _ = network.cmd_tx.send(NetCommand::HeroMove { x: world_pos.x / WORLD_SCALE, y: world_pos.y / WORLD_SCALE });
                }
            }
            // LoL MVP 鍵盤輸入
            Event::WindowEvent {
                event: WindowEvent::KeyboardInput { event: key_event, .. },
                ..
            } => {
                use fyrox::event::ElementState as ES;
                use fyrox::keyboard::{KeyCode, PhysicalKey};
                let pressed = key_event.state == ES::Pressed;
                let key = match key_event.physical_key {
                    PhysicalKey::Code(c) => c,
                    _ => return Ok(()),
                };

                // Shift 狀態追蹤
                match key {
                    KeyCode::ShiftLeft | KeyCode::ShiftRight => {
                        self.shift_held = pressed;
                        return Ok(());
                    }
                    _ => {}
                }
                if !pressed { return Ok(()); }

                let world = self.mouse_world_pos;
                let tx = self.network.as_ref().map(|n| n.cmd_tx.clone());
                let send = |cmd: NetCommand| {
                    if let Some(ref t) = tx { let _ = t.send(cmd); }
                };

                match key {
                    KeyCode::KeyW | KeyCode::KeyE | KeyCode::KeyR | KeyCode::KeyT => {
                        let slot = match key {
                            KeyCode::KeyW => "W",
                            KeyCode::KeyE => "E",
                            KeyCode::KeyR => "R",
                            KeyCode::KeyT => "T",
                            _ => unreachable!(),
                        }.to_string();
                        if self.shift_held {
                            send(NetCommand::UpgradeSkill { slot });
                        } else {
                            // 本地樂觀啟動冷卻（後端真正拒絕時後續事件會校正）
                            let slot_idx = match slot.as_str() {
                                "W" => 0, "E" => 1, "R" => 2, "T" => 3, _ => 0,
                            };
                            let id = self.hero_state.abilities.get(slot_idx).cloned().unwrap_or_default();
                            if !id.is_empty() {
                                let cur_lvl = self.hero_state.ability_levels.get(&id).copied().unwrap_or(0);
                                if cur_lvl > 0 {
                                    if let Some(info) = self.ability_info_map.get(&id) {
                                        let idx = (cur_lvl as usize - 1).min(info.cooldown.len().saturating_sub(1));
                                        if let Some(&cd) = info.cooldown.get(idx) {
                                            if cd > 0.0 {
                                                self.hero_state.ability_cd.insert(id.clone(), cd);
                                            }
                                        }
                                    }
                                }
                            }
                            send(NetCommand::CastAbility {
                                slot,
                                x: world.x / WORLD_SCALE,
                                y: world.y / WORLD_SCALE,
                            });
                        }
                    }
                    KeyCode::KeyB => {
                        self.shop_visible = !self.shop_visible;
                    }
                    // 數字鍵: shop 開啟時購買對應 index 裝備；否則使用對應背包 slot
                    KeyCode::Digit0 | KeyCode::Digit1 | KeyCode::Digit2
                    | KeyCode::Digit3 | KeyCode::Digit4 | KeyCode::Digit5
                    | KeyCode::Digit6 | KeyCode::Digit7 | KeyCode::Digit8
                    | KeyCode::Digit9 => {
                        let idx: usize = match key {
                            KeyCode::Digit0 => 0, KeyCode::Digit1 => 1,
                            KeyCode::Digit2 => 2, KeyCode::Digit3 => 3,
                            KeyCode::Digit4 => 4, KeyCode::Digit5 => 5,
                            KeyCode::Digit6 => 6, KeyCode::Digit7 => 7,
                            KeyCode::Digit8 => 8, KeyCode::Digit9 => 9,
                            _ => unreachable!(),
                        };
                        if self.shop_visible {
                            if let Some((id, _, _)) = SHOP_ITEMS.get(idx) {
                                send(NetCommand::BuyItem { item_id: id.to_string() });
                            }
                        } else if idx >= 1 && idx <= 6 {
                            // 使用 inventory slot (1-6 → 0-5)
                            if self.shift_held {
                                send(NetCommand::SellItem { slot: idx - 1 });
                            } else {
                                send(NetCommand::UseItem { slot: idx - 1 });
                            }
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn on_ui_message(
        &mut self,
        _context: &mut PluginContext,
        _message: &UiMessage,
        _ui_handle: Handle<UserInterface>,
    ) -> GameResult {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Event Processing
// ---------------------------------------------------------------------------

impl Game {
    fn apply_event(&mut self, evt: TimestampedEvent, scene: &mut Scene) {
        match (evt.msg_type.as_str(), evt.action.as_str()) {
            // Projectiles are simulated client-side (pursuit + flight_time).
            ("projectile", "C" | "create") => self.projectile_create(&evt.data, scene),
            ("projectile", "D" | "delete") => self.projectile_delete(&evt.data, scene),
            ("projectile", _) => {} // ignore stray M events (legacy compat)
            ("hero", "stats") => self.hero_stats_update(&evt.data),
            ("hero", "inventory") => self.hero_inventory_update(&evt.data),
            ("hero", "abilities_info") => self.hero_abilities_info_update(&evt.data),
            ("game", "end") => self.game_end(&evt.data),
            (_, "F" | "facing") => self.entity_facing_update(&evt.data),
            (ty, "C" | "create") => self.entity_create(ty, &evt.data, scene),
            (_, "M" | "move") => self.entity_move(&evt.data, scene),
            (_, "H" | "hp") => self.entity_hp_update(&evt.data, scene),
            (_, "D" | "delete") => self.entity_delete(&evt.data, scene),
            _ => {} // "R", "tick" etc. — ignore
        }
    }

    fn entity_facing_update(&mut self, data: &serde_json::Value) {
        let Some(id) = data.get("id").and_then(|v| v.as_u64()).map(|v| v as u32) else { return };
        let Some(f) = data.get("facing").and_then(|v| v.as_f64()) else { return };
        if let Some(entity) = self.network_entities.get_mut(&id) {
            entity.facing = f as f32;
        }
    }

    fn hero_stats_update(&mut self, data: &serde_json::Value) {
        let hs = &mut self.hero_state;
        if let Some(v) = data.get("id").and_then(|v| v.as_u64()) { hs.entity_id = Some(v as u32); }
        if let Some(v) = data.get("level").and_then(|v| v.as_i64()) { hs.level = v as i32; }
        if let Some(v) = data.get("xp").and_then(|v| v.as_i64()) { hs.xp = v as i32; }
        if let Some(v) = data.get("xp_next").and_then(|v| v.as_i64()) { hs.xp_next = v as i32; }
        if let Some(v) = data.get("skill_points").and_then(|v| v.as_i64()) { hs.skill_points = v as i32; }
        if let Some(v) = data.get("gold").and_then(|v| v.as_i64()) { hs.gold = v as i32; }
        if let Some(v) = data.get("hp").and_then(|v| v.as_f64()) { hs.hp = v as f32; }
        if let Some(v) = data.get("max_hp").and_then(|v| v.as_f64()) { hs.max_hp = v as f32; }
        if let Some(arr) = data.get("abilities").and_then(|v| v.as_array()) {
            hs.abilities = arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect();
        }
        if let Some(obj) = data.get("ability_levels").and_then(|v| v.as_object()) {
            hs.ability_levels.clear();
            for (k, v) in obj {
                if let Some(lvl) = v.as_i64() {
                    hs.ability_levels.insert(k.clone(), lvl as i32);
                }
            }
        }
    }

    fn hero_inventory_update(&mut self, data: &serde_json::Value) {
        if let Some(arr) = data.get("slots").and_then(|v| v.as_array()) {
            let mut slots: Vec<Option<(String, f32)>> = vec![None; 6];
            for (i, v) in arr.iter().enumerate().take(6) {
                if v.is_null() { slots[i] = None; continue; }
                let id = v.get("item_id").and_then(|x| x.as_str()).unwrap_or("").to_string();
                let cd = v.get("cd").and_then(|x| x.as_f64()).unwrap_or(0.0) as f32;
                if !id.is_empty() { slots[i] = Some((id, cd)); }
            }
            self.hero_state.inventory = slots;
        }
    }

    fn hero_abilities_info_update(&mut self, data: &serde_json::Value) {
        let arr = match data.get("abilities").and_then(|v| v.as_array()) {
            Some(a) => a,
            None => return,
        };
        self.ability_info_map.clear();
        for v in arr {
            let info = AbilityInfo {
                id: v.get("id").and_then(|x| x.as_str()).unwrap_or("").to_string(),
                name: v.get("name").and_then(|x| x.as_str()).unwrap_or("").to_string(),
                description: v.get("description").and_then(|x| x.as_str()).unwrap_or("").to_string(),
                key_binding: v.get("key_binding").and_then(|x| x.as_str()).unwrap_or("").to_string(),
                max_level: v.get("max_level").and_then(|x| x.as_i64()).unwrap_or(4) as i32,
                cooldown: v.get("cooldown").and_then(|x| x.as_array())
                    .map(|a| a.iter().filter_map(|e| e.as_f64()).map(|f| f as f32).collect())
                    .unwrap_or_default(),
                mana_cost: v.get("mana_cost").and_then(|x| x.as_array())
                    .map(|a| a.iter().filter_map(|e| e.as_i64()).map(|f| f as i32).collect())
                    .unwrap_or_default(),
                cast_range: v.get("cast_range").and_then(|x| x.as_array())
                    .map(|a| a.iter().filter_map(|e| e.as_f64()).map(|f| f as f32).collect())
                    .unwrap_or_default(),
                effects: v.get("effects").and_then(|x| x.as_object())
                    .map(|o| o.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                    .unwrap_or_default(),
            };
            if !info.id.is_empty() {
                self.ability_info_map.insert(info.id.clone(), info);
            }
        }
        log::info!("收到 {} 個技能詳細資訊", self.ability_info_map.len());
    }

    fn game_end(&mut self, data: &serde_json::Value) {
        let winner = data.get("winner").and_then(|v| v.as_str()).unwrap_or("?");
        self.game_ended = true;
        let msg = if winner == "player" { "VICTORY!" } else { "DEFEAT" };
        log::info!("🏆 game.end received: winner={}", winner);
        // 本 frame update 再 push 字串，這裡僅存標記
        // 透過 ui_end_text 欄位於 update 迴圈更新字
        // 為簡化 — 在 update 直接檢查 game_ended
        let _ = msg; // suppress unused
    }

    /// Handle projectile spawn: build a scene node at start_pos and register a
    /// client-side simulation entry. The bullet's motion is computed each frame
    /// in `update()` — backend does NOT stream per-tick projectile M events.
    fn projectile_create(&mut self, data: &serde_json::Value, scene: &mut Scene) {
        let id = data.get("id")
            .or_else(|| data.get("entity_id"))
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);
        let id = match id {
            Some(id) => id,
            None => return,
        };
        if self.client_projectiles.contains_key(&id) {
            return;
        }
        let target_id = data.get("target_id")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);
        let target_id = match target_id {
            Some(t) => t,
            None => return, // can't pursue without a target
        };

        let start = data.get("start_pos")
            .or_else(|| data.get("position"));
        let (sx, sy) = if let Some(pos) = start {
            (
                pos.get("x").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32 * WORLD_SCALE,
                pos.get("y").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32 * WORLD_SCALE,
            )
        } else {
            (0.0, 0.0)
        };
        let start_pos = Vector2::new(sx, sy);

        let flight_time_ms = data.get("flight_time_ms")
            .and_then(|v| v.as_u64())
            .unwrap_or(200);
        let flight_time = (flight_time_ms as f32 / 1000.0).max(0.016); // avoid div-by-zero

        let damage = data.get("damage")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;

        let node: Handle<Node> = RectangleBuilder::new(
            BaseBuilder::new().with_local_transform(
                TransformBuilder::new()
                    .with_local_position(Vector3::new(-sx, sy, Z_BULLET))
                    .with_local_scale(Vector3::new(0.1, 0.1, f32::EPSILON))
                    .build(),
            ),
        )
        .with_color(Color::from_rgba(255, 230, 50, 255))
        .build(&mut scene.graph)
        .transmute();

        self.client_projectiles.insert(id, ClientProjectile {
            node,
            target_id,
            start_pos,
            last_target_pos: start_pos,
            elapsed: 0.0,
            flight_time,
            damage,
            applied: false,
        });
    }

    fn projectile_delete(&mut self, data: &serde_json::Value, scene: &mut Scene) {
        let id = data.get("id")
            .or_else(|| data.get("entity_id"))
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);
        let id = match id {
            Some(id) => id,
            None => return,
        };
        if let Some(proj) = self.client_projectiles.remove(&id) {
            scene.graph.remove_node(proj.node);
        }
    }

    fn entity_create(&mut self, entity_type: &str, data: &serde_json::Value, scene: &mut Scene) {
        let id = data.get("entity_id")
            .or_else(|| data.get("id"))
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);
        let id = match id {
            Some(id) => id,
            None => return,
        };

        // Idempotent: skip if already exists
        if self.network_entities.contains_key(&id) {
            return;
        }

        // Parse position (scale from backend coords to render coords)
        // projectile create events carry `start_pos` instead of `position`.
        let pos_source = data.get("position").or_else(|| data.get("start_pos"));
        let (x, y) = if let Some(pos) = pos_source {
            (
                pos.get("x").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32 * WORLD_SCALE,
                pos.get("y").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32 * WORLD_SCALE,
            )
        } else {
            (
                data.get("x").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32 * WORLD_SCALE,
                data.get("y").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32 * WORLD_SCALE,
            )
        };

        // Parse move_speed (backend units/sec). Used to compute realistic lerp duration on move events.
        let move_speed = data.get("move_speed")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;

        // Parse HP
        let hp = data.get("hp").and_then(|v| v.as_f64()).map(|v| v as f32);
        let max_hp = data.get("max_hp").and_then(|v| v.as_f64()).map(|v| v as f32);
        let health = match (hp, max_hp) {
            (Some(h), Some(m)) => Some((h, m)),
            _ => None,
        };

        // Choose color/size/Z based on entity type
        let (color, size, z) = match entity_type {
            "hero" => (Color::from_rgba(50, 180, 50, 255), 0.4, Z_ENEMY),
            "creep" | "enemy" => (Color::from_rgba(220, 40, 40, 255), 0.3, Z_ENEMY),
            "unit" | "tower" => (Color::from_rgba(50, 100, 220, 255), 0.4, Z_TOWER),
            "bullet" | "projectile" => (Color::from_rgba(255, 230, 50, 255), 0.1, Z_BULLET),
            _ => (Color::from_rgba(200, 200, 200, 255), 0.3, Z_ENEMY),
        };

        let node: Handle<Node> = RectangleBuilder::new(
            BaseBuilder::new().with_local_transform(
                TransformBuilder::new()
                    .with_local_position(Vector3::new(-x, y, z))
                    .with_local_scale(Vector3::new(size, size, f32::EPSILON))
                    .build(),
            ),
        )
        .with_color(color)
        .build(&mut scene.graph)
        .transmute();

        // HP bars (if entity has health)
        let (hp_bar_bg, hp_bar_fg) = if health.is_some() {
            let bar_y = y + size * 0.5 + 0.1;
            let bg = RectangleBuilder::new(
                BaseBuilder::new().with_local_transform(
                    TransformBuilder::new()
                        .with_local_position(Vector3::new(-x, bar_y, Z_HP_BAR))
                        .with_local_scale(Vector3::new(0.8, 0.06, f32::EPSILON))
                        .build(),
                ),
            )
            .with_color(Color::from_rgba(0, 0, 0, 255))
            .build(&mut scene.graph)
            .transmute();

            let fg = RectangleBuilder::new(
                BaseBuilder::new().with_local_transform(
                    TransformBuilder::new()
                        .with_local_position(Vector3::new(-x, bar_y, Z_HP_BAR - 0.0001))
                        .with_local_scale(Vector3::new(0.8, 0.06, f32::EPSILON))
                        .build(),
                ),
            )
            .with_color(Color::from_rgba(0, 220, 0, 255))
            .build(&mut scene.graph)
            .transmute();

            (Some(bg), Some(fg))
        } else {
            (None, None)
        };

        let name = data.get("name").and_then(|v| v.as_str())
            .unwrap_or(entity_type).to_string();

        let pos = Vector2::new(x, y);

        // For creeps: draw debug polyline from current pos through remaining
        // waypoints sent by the backend (`path_points`), AND kick off a lerp
        // toward `path_points[0]` so a creep that is mid-segment when the client
        // joins doesn't freeze waiting for the next server M event.
        let mut initial_target = pos;
        let mut initial_duration = 0.1_f32;
        let path_nodes = if entity_type == "creep" {
            let mut segments = Vec::new();
            if let Some(pts) = data.get("path_points").and_then(|v| v.as_array()) {
                let mut prev = Vector2::new(x, y);
                let mut first_target: Option<Vector2<f32>> = None;
                for pt in pts.iter() {
                    let px = pt.get("x").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32 * WORLD_SCALE;
                    let py = pt.get("y").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32 * WORLD_SCALE;
                    let next = Vector2::new(px, py);
                    if first_target.is_none() {
                        first_target = Some(next);
                    }
                    if let Some(seg) = build_path_segment(scene, prev, next) {
                        segments.push(seg);
                    }
                    prev = next;
                }
                if let Some(t) = first_target {
                    let dx = t.x - x;
                    let dy = t.y - y;
                    let dist_render = (dx * dx + dy * dy).sqrt();
                    let dist_backend = dist_render / WORLD_SCALE;
                    if move_speed > 1.0 && dist_backend > f32::EPSILON {
                        initial_target = t;
                        initial_duration = (dist_backend / move_speed).clamp(0.01, 3600.0);
                    }
                }
            }
            segments
        } else {
            Vec::new()
        };

        // 讀取初始 facing（若 create payload 有帶）
        let initial_facing = data.get("facing").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;

        // 建立面向箭頭（只為有 health 的單位做，tower/creep/hero 都有）
        let facing_arrow = if health.is_some() {
            Some(build_facing_arrow(scene, x, y, size, initial_facing))
        } else {
            None
        };

        self.network_entities.insert(id, NetworkEntity {
            entity_type: entity_type.to_string(),
            node,
            hp_bar_bg,
            hp_bar_fg,
            position: pos,
            health,
            name,
            name_label: None, // created lazily in update()
            prev_position: pos,
            target_position: initial_target,
            lerp_elapsed: 0.0,
            lerp_duration: initial_duration,
            move_speed,
            path_nodes,
            path_age: 0.0,
            facing: initial_facing,
            facing_arrow,
        });
    }

    fn entity_move(&mut self, data: &serde_json::Value, scene: &mut Scene) {
        let id = data.get("entity_id")
            .or_else(|| data.get("id"))
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);
        let id = match id {
            Some(id) => id,
            None => return,
        };

        let entity = match self.network_entities.get_mut(&id) {
            Some(e) => e,
            None => return,
        };

        // Parse new position (scale from backend coords to render coords)
        let (x, y) = if let Some(pos) = data.get("position") {
            (
                pos.get("x").and_then(|v| v.as_f64()).unwrap_or((entity.position.x / WORLD_SCALE) as f64) as f32 * WORLD_SCALE,
                pos.get("y").and_then(|v| v.as_f64()).unwrap_or((entity.position.y / WORLD_SCALE) as f64) as f32 * WORLD_SCALE,
            )
        } else {
            (
                data.get("x").and_then(|v| v.as_f64()).unwrap_or((entity.position.x / WORLD_SCALE) as f64) as f32 * WORLD_SCALE,
                data.get("y").and_then(|v| v.as_f64()).unwrap_or((entity.position.y / WORLD_SCALE) as f64) as f32 * WORLD_SCALE,
            )
        };

        // Start lerp from current interpolated position to new server position.
        // For creeps, anchor prev_position to the PREVIOUS target (i.e. the waypoint
        // the server said we just arrived at), not the mid-flight interpolated pos.
        // Otherwise, if the client lerp is still traversing A→B when the server emits
        // M(→C), we'd lerp from "somewhere on A→B" straight to C — cutting the corner
        // and making the creep appear to skip the B vertex.
        let new_target = Vector2::new(x, y);
        let is_creep = entity.entity_type == "creep";
        entity.prev_position = if is_creep { entity.target_position } else { entity.position };
        entity.position = entity.prev_position;
        entity.target_position = new_target;
        entity.lerp_elapsed = 0.0;
        entity.lerp_duration = {
            let dx = entity.target_position.x - entity.prev_position.x;
            let dy = entity.target_position.y - entity.prev_position.y;
            let dist_render = (dx * dx + dy * dy).sqrt();
            let dist_backend = dist_render / WORLD_SCALE;
            if entity.move_speed > 1.0 && dist_backend > f32::EPSILON {
                // Min clamp 0.01s for dense event streams. Max clamp generous so that
                // slow creeps (e.g. msd=50 walking 800 units = 16s) aren't truncated
                // and forced to idle at waypoints waiting for the next M event.
                (dist_backend / entity.move_speed).clamp(0.01, 3600.0)
            } else {
                0.1
            }
        };

        // Update HP if provided
        let hp = data.get("hp").and_then(|v| v.as_f64()).map(|v| v as f32);
        let max_hp = data.get("max_hp").and_then(|v| v.as_f64()).map(|v| v as f32);
        if let (Some(h), Some(m)) = (hp, max_hp) {
            entity.health = Some((h, m));
        }

        // 更新 facing（若 payload 有帶）
        if let Some(f) = data.get("facing").and_then(|v| v.as_f64()) {
            entity.facing = f as f32;
        }
    }

    /// HP-only update (action "H"). Updates health bar without touching position/lerp —
    /// critical for creeps: a damage event must not reset the ongoing waypoint lerp.
    fn entity_hp_update(&mut self, data: &serde_json::Value, scene: &mut Scene) {
        let id = data.get("entity_id")
            .or_else(|| data.get("id"))
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);
        let id = match id {
            Some(id) => id,
            None => return,
        };

        let entity = match self.network_entities.get_mut(&id) {
            Some(e) => e,
            None => return,
        };

        let hp = data.get("hp").and_then(|v| v.as_f64()).map(|v| v as f32);
        let max_hp = data.get("max_hp").and_then(|v| v.as_f64()).map(|v| v as f32);
        if let (Some(h), Some(m)) = (hp, max_hp) {
            entity.health = Some((h, m));
        }

        // HP bar stays green; width is recomputed in the update loop.
    }

    fn entity_delete(&mut self, data: &serde_json::Value, scene: &mut Scene) {
        let id = data.get("entity_id")
            .or_else(|| data.get("id"))
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);
        let id = match id {
            Some(id) => id,
            None => return,
        };

        if let Some(entity) = self.network_entities.remove(&id) {
            scene.graph.remove_node(entity.node);
            if let Some(bg) = entity.hp_bar_bg {
                scene.graph.remove_node(bg);
            }
            if let Some(fg) = entity.hp_bar_fg {
                scene.graph.remove_node(fg);
            }
            if let Some(arrow) = entity.facing_arrow {
                scene.graph.remove_node(arrow);
            }
            if let Some(label) = entity.name_label {
                self.pending_label_deletions.push(label);
            }
            for seg in entity.path_nodes {
                scene.graph.remove_node(seg);
            }
        }
    }
}

/// Build a thin rotated rectangle representing a line segment from `from` to `to`.
/// Returns `None` if the segment has zero length.
/// 為單位建立一個指向面向方向的箭頭（偏離中心一半 length，讓箭頭伸出單位外）
/// `pos_x/pos_y` 是 backend world 座標（未翻轉），內部會套 `-x` 配合渲染鏡像。
/// 組 tooltip 文字：LoL 風格分區
/// ┌───────────────────────────┐
/// │ 技能名 ★ 終極技            [W] │
/// │ 等級 3 / 5                      │
/// ├─ 描述 ──────────────────── │
/// │ ...                             │
/// ├─ 屬性 ──────────────────── │
/// │ 冷卻 / 魔力 / 射程              │
/// ├─ 效果 ──────────────────── │
/// │ 傷害 / 持續 ...                 │
/// ├─ 下一級 ───────────────── │
/// │ 提升項目                        │
/// └───────────────────────────┘
fn format_ability_tooltip(info: &AbilityInfo, cur_lvl: i32) -> String {
    let max = info.max_level;
    let is_ultimate = info.key_binding == "T";
    let bar = "──────────────────────────\n";

    let show_idx = (cur_lvl.max(1) - 1) as usize;
    let next_idx = (cur_lvl as usize).min((max as usize).saturating_sub(1));
    let show_next = cur_lvl < max;

    fn at_f32(arr: &[f32], idx: usize) -> Option<f32> {
        if arr.is_empty() { None } else { Some(arr[idx.min(arr.len() - 1)]) }
    }
    fn at_i32(arr: &[i32], idx: usize) -> Option<i32> {
        if arr.is_empty() { None } else { Some(arr[idx.min(arr.len() - 1)]) }
    }

    let mut out = String::new();

    // ===== 標題列 =====
    if is_ultimate {
        out.push_str(&format!("★ {}  (終極)   [{}]\n", info.name, info.key_binding));
    } else {
        out.push_str(&format!("{}   [{}]\n", info.name, info.key_binding));
    }
    if cur_lvl == 0 {
        out.push_str(&format!("未學習  (0/{})\n", max));
    } else {
        out.push_str(&format!("等級 {} / {}\n", cur_lvl, max));
    }

    // ===== 描述 =====
    out.push_str(bar);
    out.push_str("【說明】\n");
    out.push_str(&format!("{}\n", info.description));

    // ===== 當前屬性（核心數值）=====
    out.push_str(bar);
    out.push_str("【屬性】\n");
    if let Some(c) = at_f32(&info.cooldown, show_idx) {
        out.push_str(&format!("  冷卻時間：{:.1} 秒\n", c));
    }
    if let Some(c) = at_i32(&info.mana_cost, show_idx) {
        out.push_str(&format!("  魔力消耗：{}\n", c));
    }
    if let Some(c) = at_f32(&info.cast_range, show_idx) {
        if c > 0.0 { out.push_str(&format!("  施放範圍：{:.0}\n", c)); }
    }

    // ===== 效果（傷害 / 其他）=====
    if !info.effects.is_empty() {
        out.push_str(bar);
        out.push_str("【效果】\n");
        // 優先顯示常見欄位（damage / heal / ratio / duration / stun / slow）
        let priority_keys = [
            "damage", "heal", "shield", "duration",
            "stun", "slow", "ad_ratio", "ap_ratio", "ratio",
        ];
        let mut shown: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for pk in priority_keys.iter() {
            if let Some(v) = info.effects.get(*pk) {
                push_effect_line(&mut out, pk, v, show_idx);
                shown.insert(*pk);
            }
        }
        for (k, v) in info.effects.iter() {
            if !shown.contains(k.as_str()) {
                push_effect_line(&mut out, k, v, show_idx);
            }
        }
    }

    // ===== 下一級提升 =====
    if show_next {
        let mut delta_lines: Vec<String> = Vec::new();
        if let (Some(c), Some(n)) = (at_f32(&info.cooldown, show_idx), at_f32(&info.cooldown, next_idx)) {
            if (c - n).abs() > f32::EPSILON {
                delta_lines.push(format!("  冷卻 {:.1}s → {:.1}s", c, n));
            }
        }
        if let (Some(c), Some(n)) = (at_i32(&info.mana_cost, show_idx), at_i32(&info.mana_cost, next_idx)) {
            if c != n {
                delta_lines.push(format!("  魔力 {} → {}", c, n));
            }
        }
        if let (Some(c), Some(n)) = (at_f32(&info.cast_range, show_idx), at_f32(&info.cast_range, next_idx)) {
            if c > 0.0 && (c - n).abs() > f32::EPSILON {
                delta_lines.push(format!("  射程 {:.0} → {:.0}", c, n));
            }
        }
        for (k, v) in info.effects.iter() {
            if let Some(arr) = v.as_array() {
                let cur = arr.get(show_idx).and_then(|e| e.as_f64());
                let nxt = arr.get(next_idx).and_then(|e| e.as_f64());
                if let (Some(c), Some(n)) = (cur, nxt) {
                    if (c - n).abs() > f64::EPSILON {
                        delta_lines.push(format!("  {} {} → {}", k, fmt_num(c), fmt_num(n)));
                    }
                }
            }
        }
        if !delta_lines.is_empty() {
            out.push_str(bar);
            out.push_str("【下一級】\n");
            for l in &delta_lines { out.push_str(l); out.push('\n'); }
        }
    }

    // ===== 升級提示 =====
    out.push_str(bar);
    if cur_lvl < max {
        out.push_str(&format!("Shift + {} 升級（需 1 技能點）\n", info.key_binding));
    } else {
        out.push_str("已達最高等級\n");
    }
    out
}

/// 格式化一個 effect line：array 取 show_idx；scalar 直接印
fn push_effect_line(out: &mut String, key: &str, v: &serde_json::Value, show_idx: usize) {
    let label = effect_label(key);
    if let Some(arr) = v.as_array() {
        if let Some(val) = arr.get(show_idx).and_then(|e| e.as_f64()) {
            out.push_str(&format!("  {}：{}\n", label, fmt_num(val)));
            return;
        }
        if let Some(val) = arr.last().and_then(|e| e.as_f64()) {
            out.push_str(&format!("  {}：{}\n", label, fmt_num(val)));
        }
    } else if let Some(n) = v.as_f64() {
        out.push_str(&format!("  {}：{}\n", label, fmt_num(n)));
    } else if let Some(s) = v.as_str() {
        out.push_str(&format!("  {}：{}\n", label, s));
    } else if let Some(b) = v.as_bool() {
        out.push_str(&format!("  {}：{}\n", label, if b { "是" } else { "否" }));
    }
}

fn fmt_num(n: f64) -> String {
    if (n - n.round()).abs() < 1e-6 { format!("{:.0}", n) } else { format!("{:.2}", n) }
}

fn effect_label(key: &str) -> &str {
    match key {
        "damage" => "傷害",
        "heal" => "治療",
        "shield" => "護盾",
        "duration" => "持續時間",
        "stun" => "暈眩時間",
        "slow" => "減速",
        "ad_ratio" => "攻擊加成",
        "ap_ratio" => "法強加成",
        "ratio" => "係數",
        "radius" => "範圍半徑",
        "speed" => "速度",
        _ => key,
    }
}

fn build_facing_arrow(
    scene: &mut Scene,
    pos_x: f32,
    pos_y: f32,
    entity_size: f32,
    facing: f32,
) -> Handle<Node> {
    let length = (entity_size * 0.7).max(0.12);
    let thickness = (entity_size * 0.15).max(0.04);
    // 渲染時 X 軸鏡像 → 角度用 π - facing 補回
    let render_angle = std::f32::consts::PI - facing;
    let offset_x = (length * 0.5) * render_angle.cos();
    let offset_y = (length * 0.5) * render_angle.sin();
    let rotation = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), render_angle);
    RectangleBuilder::new(
        BaseBuilder::new().with_local_transform(
            TransformBuilder::new()
                .with_local_position(Vector3::new(-pos_x + offset_x, pos_y + offset_y, 0.0007))
                .with_local_rotation(rotation)
                .with_local_scale(Vector3::new(length, thickness, f32::EPSILON))
                .build(),
        ),
    )
    .with_color(Color::from_rgba(255, 200, 0, 255))
    .build(&mut scene.graph)
    .transmute()
}

fn build_path_segment(
    scene: &mut Scene,
    from: Vector2<f32>,
    to: Vector2<f32>,
) -> Option<Handle<Node>> {
    let dx = to.x - from.x;
    let dy = to.y - from.y;
    let length = (dx * dx + dy * dy).sqrt();
    if length < f32::EPSILON {
        return None;
    }
    // X 軸渲染取負（配合 entity 翻轉）；rotation 也要跟著反向
    let center = Vector3::new(-(from.x + to.x) * 0.5, (from.y + to.y) * 0.5, Z_PATH);
    let thickness = 0.05;
    // atan2 裡 dx 要取負，讓旋轉方向與翻轉後一致
    let rotation = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), dy.atan2(-dx));
    let handle = RectangleBuilder::new(
        BaseBuilder::new().with_local_transform(
            TransformBuilder::new()
                .with_local_position(center)
                .with_local_rotation(rotation)
                .with_local_scale(Vector3::new(length, thickness, f32::EPSILON))
                .build(),
        ),
    )
    .with_color(Color::from_rgba(255, 100, 255, 180))
    .build(&mut scene.graph)
    .transmute();
    Some(handle)
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

fn parse_heartbeat(data: &serde_json::Value) -> HeartbeatInfo {
    HeartbeatInfo {
        tick: data.get("tick").and_then(|v| v.as_u64()).unwrap_or(0),
        game_time: data.get("game_time").and_then(|v| v.as_f64()).unwrap_or(0.0),
        entity_count: data.get("entity_count").and_then(|v| v.as_u64()).unwrap_or(0),
        hero_count: data.get("hero_count").and_then(|v| v.as_u64()).unwrap_or(0),
        creep_count: data.get("creep_count").and_then(|v| v.as_u64()).unwrap_or(0),
    }
}

fn grid_to_world(col: usize, row: usize) -> (f32, f32) {
    let x = GRID_ORIGIN_X + col as f32 * CELL_SIZE + CELL_SIZE * 0.5;
    let y = GRID_ORIGIN_Y + row as f32 * CELL_SIZE + CELL_SIZE * 0.5;
    (x, y)
}

fn world_to_grid(wx: f32, wy: f32) -> Option<(usize, usize)> {
    let col = ((wx - GRID_ORIGIN_X) / CELL_SIZE).floor() as i32;
    let row = ((wy - GRID_ORIGIN_Y) / CELL_SIZE).floor() as i32;
    if col >= 0 && col < GRID_COLS as i32 && row >= 0 && row < GRID_ROWS as i32 {
        Some((col as usize, row as usize))
    } else {
        None
    }
}

fn world_to_screen_approx(wx: f32, wy: f32, window_w: f32, window_h: f32) -> Vector2<f32> {
    let world_height = 20.0;
    let aspect = window_w / window_h;
    let world_width = world_height * aspect;
    // +X world → +X screen（camera 的 -1 X scale 已把原本的翻轉抵消）
    let sx = (wx / world_width + 0.5) * window_w;
    // +Y world → 螢幕上方（螢幕 pixel Y 向下，所以要反向）
    let sy = (-wy / world_height + 0.5) * window_h;
    Vector2::new(sx, sy)
}

fn screen_to_world_approx(
    screen_x: f32,
    screen_y: f32,
    window_w: f32,
    window_h: f32,
    world_height: f32,
) -> Vector2<f32> {
    let aspect = window_w / window_h;
    let world_width = world_height * aspect;
    // 螢幕 +X → 世界 +X
    let wx = (screen_x / window_w - 0.5) * world_width;
    // 螢幕 +Y（向下）→ 世界 -Y
    let wy = -(screen_y / window_h - 0.5) * world_height;
    Vector2::new(wx, wy)
}
