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

mod sprite_resources;
mod lockstep_client;
mod sim_runner;
mod render_bridge;

/// Bridge between the two distinct `PlayerInput` Rust types: omoba_core's
/// kcp client uses its own prost-generated copy of `proto/game.proto`,
/// while omobab (omb-as-lib) generates the same proto into a separate
/// crate-local module. They're identical wire format, so we round-trip
/// via prost encode/decode at the boundary instead of hand-mapping every
/// PlayerInput oneof variant.
fn convert_player_input(
    src: &omoba_core::kcp::game_proto::PlayerInput,
) -> Option<sim_runner::PlayerInput> {
    use prost::Message;
    let mut buf = Vec::with_capacity(src.encoded_len());
    if let Err(e) = src.encode(&mut buf) {
        log::error!("[lockstep] convert_player_input encode failed: {}", e);
        return None;
    }
    match sim_runner::PlayerInput::decode(buf.as_slice()) {
        Ok(out) => Some(out),
        Err(e) => {
            log::error!("[lockstep] convert_player_input decode failed: {}", e);
            None
        }
    }
}

/// Phase 4.3: Fixed32 raw scaling. Vec2I stores `real_units * 1024` as sint32
/// (see proto/game.proto: "divide by 1024 to get real units"). Backend logical
/// coords = render coords / WORLD_SCALE; multiply that by 1024 to get the raw.
const FIXED32_ONE: f32 = 1024.0;

/// Convert an omfx render-space world position into the backend Fixed32 raw
/// `Vec2I` used by `PlayerInput::MoveTo` / `CastAbility::target_pos`.
fn world_render_to_vec2i(world: Vector2<f32>) -> omoba_core::kcp::game_proto::Vec2I {
    let backend_x = world.x / WORLD_SCALE;
    let backend_y = world.y / WORLD_SCALE;
    omoba_core::kcp::game_proto::Vec2I {
        x: (backend_x * FIXED32_ONE) as i32,
        y: (backend_y * FIXED32_ONE) as i32,
    }
}

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

// Z layers in 3D camera frustum (camera at z=-100 looking +Z, near=0.1 far=1000).
// SMALLER Z = closer to camera = drawn on top (industry-standard 3D 慣例)。
//
// 為什麼是 +Z 視角不是 -Z：Fyrox 的 `Camera::calculate_matrices` 用
// `Matrix4::look_at_rh(eye, eye+look_vec, up_vec)`，其中 look_vec 來自
// 旋轉矩陣 col 2（identity 給 (0,0,1)），所以 default 看 +Z。
// 旋轉 camera 看 -Z 會被 `look_at_rh` 自己重算 side = forward × up，
// 把原本的 world -X side（跟 omfx `(-bx, by)` x-flip 慣例配對）翻成 world +X，
// 結果整個畫面左右相反。改成 camera 在 -Z 側、看 +Z（default 方向）就避開了。
const Z_BULLET: f32 = 0.5;
const Z_HP_BAR: f32 = 1.0;
const Z_RING: f32 = 1.5;
const Z_ENEMY: f32 = 2.0;
const Z_TOWER: f32 = 2.5;
const Z_REGION: f32 = 3.0;
const Z_GRID_CELL: f32 = 3.5;
const Z_PATH: f32 = 4.0;
const Z_BACKGROUND: f32 = 4.5;

const COLLISION_RING_SEGMENTS: usize = 24;
const COLLISION_RING_THICKNESS: f32 = 0.025;
/// 預設關閉：1000 entity 各 24 段 = 24 K scene node，每幀 transform update
/// 是 stress 場景下最大 CPU 成本之一。改 true 可恢復 debug 可視化。
const COLLISION_RING_ENABLED: bool = false;
/// Per-frame debug 畫每個 entity 的 collision ring（走 SceneDrawingContext，
/// 千個 entity 還是 1 個 draw call，跟 COLLISION_RING_ENABLED 的 24K scene node
/// 路徑不同）。stress 場景開著沒事。
const DEBUG_COLLISION_RINGS: bool = true;
const REGION_LINE_THICKNESS: f32 = 0.04;
const REGION_BLOCKER_SEGMENTS: usize = 12;
const REGION_BLOCKER_THICKNESS: f32 = 0.015;

// ---------------------------------------------------------------------------
// Network Types



/// Seconds that a newly-spawned creep's debug path stays visible.
const PATH_VISIBLE_SECS: f32 = 5.0;

/// Phase 5.1 (pass 3): NetworkEntity is dead — apply_event populated this
/// per-entity render registry from legacy GameEvent stream which was
/// deleted in pass 2. The struct + `Game::network_entities` field stay in
/// source so the orphan render loops in Game::update (interpolation, HP
/// bars, name labels, projectile collision lookups) compile against an
/// always-empty HashMap. Phase 5.x removes the orphan loops + this struct
/// (estimated ~600 lines of update body cleanup).
#[derive(Debug)]
#[allow(dead_code)]
struct NetworkEntity {
    entity_type: String,
    body_slot: u32,
    body_size: f32,
    body_z: f32,
    body_color: [u8; 4],
    hp_bg_slot: Option<u32>,
    hp_fg_slot: Option<u32>,
    facing_slot: Option<u32>,
    position: Vector2<f32>,
    health: Option<(f32, f32)>,
    name: String,
    name_label: Option<Handle<Text>>,
    prev_position: Vector2<f32>,
    target_position: Vector2<f32>,
    lerp_elapsed: f32,
    lerp_duration: f32,
    move_speed: f32,
    path_nodes: Vec<Handle<Node>>,
    path_age: f32,
    facing: f32,
    collision_radius_render: f32,
    collision_ring: Vec<(Handle<Node>, Vector2<f32>)>,
    tower_kind: Option<String>,
    attack_range_backend: f32,
    upgrade_levels: [u8; 3],
    last_label_text: String,
    last_label_pos: Vector2<f32>,
    extrap_velocity: f32,
    extrap_start_pos: Vector2<f32>,
    extrap_direction: Vector2<f32>,
    extrap_elapsed: f32,
    extrap_duration: f32,
}

/// Per-entity UI label tracking for sim_runner-backed sprites.
/// `last_*` fields gate UI message sends to avoid flooding the queue at
/// 60 fps × N entities when nothing visible has changed.
#[derive(Debug)]
struct SimEntityLabel {
    handle: Handle<Text>,
    last_text: String,
    last_pos: Vector2<f32>,
}

/// TD 塔的完整元資料（host + script 合併；前端快取一份，供預覽 / 按鈕 / sell 使用）。
#[derive(Clone, Debug)]
struct TdTemplate {
    label: String,
    cost: i32,
    footprint_backend: f32,
    range_backend: f32,
    splash_radius_backend: f32,
    hit_radius_backend: f32,
    slow_factor: f32,
    slow_duration: f32,
}

/// Bomb 爆炸紅圈特效：由 0 半徑膨脹到 `max_radius`，`duration` 秒後消失。
/// 每 frame 透過 `scene.drawing_context.add_line(...)` 提交 32 段圓環，整批 single draw call。
#[derive(Debug)]
struct ActiveExplosion {
    pos: Vector2<f32>, // render 座標
    max_radius: f32,   // render 單位
    duration: f32,
    elapsed: f32,
}

/// Client-side projectile simulation.
///
/// Backend only sends a single C event with `target_id` + `flight_time_ms`;
/// the bullet's position is computed locally each frame as a pursuit lerp
/// from `start_pos` toward the target entity's CURRENT client-side position.
/// P7 layered prediction entry (per projectile id). Tracks「server 已經宣告
/// 但 server 還沒送 hp_snapshot 反映」這段視窗內，client 想本地視覺上扣多少血。
///
/// Lifecycle:
///   PC arrives        → insert (applied=false)
///   visual t≥1.0 hit  → applied=true（命中時刻才從 display HP 扣下去）
///   D event           → remove（projectile 死了：可能命中、可能 timeout/取消）
///   heartbeat retain  → 不在 server 的 in_flight_projectiles 集合 → remove
///
/// HP bar render 時：display_hp = authoritative_hp(server 權威值) - Σ(applied dmg)。
/// 跟 heartbeat hp_snapshot 不雙重計算因為 server 用 in_flight 顯式告訴 client
/// 哪幾發還沒結算（沒在裡面的就是已經反映在 hp_snapshot 上、應該移除）。
#[derive(Debug)]
struct PendingPredDmg {
    target_id: u32,
    dmg: f32,
    applied: bool,
}

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
    /// 方向性子彈（Tack 放射針）：無 target_id，走直線到 `end_pos`
    directional: bool,
    end_pos: Vector2<f32>,
    /// 命中半徑視覺化圓環（跟著子彈走）；hit_radius > 0 且 directional 時建立
    hit_ring: Vec<(Handle<Node>, Vector2<f32>)>,
    /// Bomb 塔專用：命中後自 spawn 爆炸特效；render 單位
    splash_radius_render: f32,
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


// ---------------------------------------------------------------------------
// Game Plugin
// ---------------------------------------------------------------------------

// ---------- Frame profile (omfx 端 per-frame timing 拆解，類比 omb 的 tick_profile) ----------

#[derive(Default, Debug)]
struct FrameProfile {
    frame_count: u64,
    events_ns: u128,
    interp_ns: u128,
    visual_ns: u128,
    proj_ns: u128,
    cam_ns: u128,
    ui_ns: u128,
    total_ns: u128,
    events_drained: u64,
    creeps_seen: u64,
    projectiles_seen: u64,
    pure_render_ms_total: f64,
    capped_render_ms_total: f64,
    draw_calls_total: u64,
    triangles_total: u64,
    last_fps: usize,
    /// Most recent per-frame snapshot (overwritten each call to `record_render_stats`).
    /// Used by the HUD status text — window averages reset every WINDOW frames so
    /// the instantaneous sample gives smoother live readout.
    last_draw_calls: usize,
    last_triangles: usize,
}

impl FrameProfile {
    const WINDOW: u64 = 60;

    fn finish_frame(&mut self) {
        self.frame_count += 1;
        if self.frame_count % Self::WINDOW == 0 {
            self.emit_log();
            self.reset_window();
        }
    }

    fn emit_log(&self) {
        let w = Self::WINDOW as f64;
        let total_ms = self.total_ns as f64 / w / 1_000_000.0;
        let max_fps = if total_ms > 0.0 { (1000.0 / total_ms) as u32 } else { 0 };
        log::info!(
            "omfx_frame window={} avg(ms) events={:.2} interp={:.2} visual={:.2} proj={:.2} cam={:.2} ui={:.2} total={:.2} (max_fps={}, events_per_frame={:.0}, creeps={:.0}, projectiles={:.0})",
            Self::WINDOW,
            self.events_ns as f64 / w / 1_000_000.0,
            self.interp_ns as f64 / w / 1_000_000.0,
            self.visual_ns as f64 / w / 1_000_000.0,
            self.proj_ns as f64 / w / 1_000_000.0,
            self.cam_ns as f64 / w / 1_000_000.0,
            self.ui_ns as f64 / w / 1_000_000.0,
            total_ms,
            max_fps,
            self.events_drained as f64 / w,
            self.creeps_seen as f64 / w,
            self.projectiles_seen as f64 / w,
        );
        log::info!(
            "omfx_render window={} avg(ms) pure={:.2} capped={:.2} fps={} draw_calls={:.0} triangles={:.0}",
            Self::WINDOW,
            self.pure_render_ms_total / Self::WINDOW as f64,
            self.capped_render_ms_total / Self::WINDOW as f64,
            self.last_fps,
            self.draw_calls_total as f64 / Self::WINDOW as f64,
            self.triangles_total as f64 / Self::WINDOW as f64,
        );
    }

    fn record_render_stats(&mut self, stats: &fyrox::renderer::stats::Statistics) {
        self.pure_render_ms_total += (stats.pure_frame_time as f64) * 1000.0;
        self.capped_render_ms_total += (stats.capped_frame_time as f64) * 1000.0;
        self.draw_calls_total += stats.geometry.draw_calls as u64;
        self.triangles_total += stats.geometry.triangles_rendered as u64;
        self.last_fps = stats.frames_per_second;
        self.last_draw_calls = stats.geometry.draw_calls;
        self.last_triangles = stats.geometry.triangles_rendered;
    }

    fn reset_window(&mut self) {
        self.events_ns = 0;
        self.interp_ns = 0;
        self.visual_ns = 0;
        self.proj_ns = 0;
        self.cam_ns = 0;
        self.ui_ns = 0;
        self.total_ns = 0;
        self.events_drained = 0;
        self.creeps_seen = 0;
        self.projectiles_seen = 0;
        self.pure_render_ms_total = 0.0;
        self.capped_render_ms_total = 0.0;
        self.draw_calls_total = 0;
        self.triangles_total = 0;
        // last_fps is just-overwritten each frame, no reset needed
    }
}

#[derive(Default, Visit, Reflect, Debug)]
#[reflect(non_cloneable)]
pub struct Game {
    scene: Handle<Scene>,
    camera: Handle<Node>,
    #[visit(skip)] #[reflect(hidden)]
    mouse_world_pos: Vector2<f32>,
    #[visit(skip)] #[reflect(hidden)]
    window_size: Vector2<f32>,

    /// Shared sprite GPU resources (single quad + 9 materials).
    /// Lazily initialized on first frame; reused for all entity sprite Meshes.
    #[visit(skip)] #[reflect(hidden)]
    sprite_resources: Option<sprite_resources::SharedSpriteResources>,

    /// 所有 entity body sprite 共用的 batched mesh — 1 個 mesh / 1 draw call 容納
    /// 數千個 entity，取代之前每 entity 1 個 Mesh 的爆量 draw call 浪費。
    /// Capacity 4096 quad（涵蓋 1800 creep + 1000 tower + 餘裕）。
    #[visit(skip)] #[reflect(hidden)]
    body_batch: Option<sprite_resources::BatchedSpriteMesh>,

    /// HP bar 黑底 + 綠條 共用 batched mesh（per-vertex color，bg/fg 兩個 slot
    /// 一個 entity）。Capacity 8192 = 4096 entity × 2 (bg + fg)。
    #[visit(skip)] #[reflect(hidden)]
    hp_batch: Option<sprite_resources::BatchedSpriteMesh>,

    /// Facing arrow 共用 batched mesh（with rotation）。Capacity 4096。
    #[visit(skip)] #[reflect(hidden)]
    facing_batch: Option<sprite_resources::BatchedSpriteMesh>,


    // --- Network ---
    // Phase 5.1: legacy `network: Option<NetworkBridge>` field removed.
    /// Lockstep client (KCP tags 0x10-0x16). Drives sim_runner via
    /// TickBatch / StateHash on a separate background thread.
    #[visit(skip)] #[reflect(hidden)]
    lockstep_handle: Option<lockstep_client::LockstepClientHandle>,
    /// Phase 4.3: most recent `LockstepEvent::TickBatch.tick` observed.
    /// Used to compute `target_tick = current_sim_tick + 3` for input
    /// submissions (50 ms input delay at 60 Hz). Initialized to 0 via
    /// `#[derive(Default)]`; updated each frame in the TickBatch arm of
    /// `Game::update`.
    #[visit(skip)] #[reflect(hidden)]
    current_sim_tick: u32,
    /// Phase 3.2 sim_runner worker (full omb ECS dispatcher running off a
    /// background thread). Dropped on `on_deinit` so the channel
    /// disconnect lets the worker exit. Phase 3.3 will wire input feed
    /// from `lockstep_handle`; until then the worker just blocks on
    /// `master_seed_rx.recv()` and never ticks.
    #[visit(skip)] #[reflect(hidden)]
    sim_runner_handle: Option<sim_runner::SimRunnerHandle>,
    /// Phase 3.4 render bridge: reads `SimWorldSnapshot` per frame and
    /// (Phase 4) spawns / updates / despawns Fyrox sprites for each entity.
    /// Currently a stub that logs entity render data. Always allocated
    /// (cheap default) so the per-frame check in `update` is just a method
    /// call, not an `Option` unwrap.
    #[visit(skip)] #[reflect(hidden)]
    render_bridge: render_bridge::RenderBridge,
    #[visit(skip)] #[reflect(hidden)]
    connection_status: ConnectionStatus,
    // Phase 5.1: `event_buffer: Option<EventBuffer>` field removed.
    // EventBuffer drove the legacy GameEvent reorder/replay pipeline.
    // `network_entities` field kept temporarily — orphan render loops in
    // Game::update still iterate it (always empty since apply_event is
    // gone); Phase 5.x cleans up the loops + this field.
    #[visit(skip)] #[reflect(hidden)]
    network_entities: HashMap<u32, NetworkEntity>,
    /// BlockedRegion 線框 scene node（每個 region 一組 polygon outline segments）。
    #[visit(skip)] #[reflect(hidden)]
    region_line_nodes: Vec<Handle<Node>>,
    /// Region blocker 近似圓 scene node（debug 視覺化；與 region 線框一起在 init 時畫）。
    #[visit(skip)] #[reflect(hidden)]
    region_blocker_nodes: Vec<Handle<Node>>,
    /// TD 模式氣球路線的 scene node（每條 path 一組線段）。
    #[visit(skip)] #[reflect(hidden)]
    td_path_nodes: Vec<Handle<Node>>,
    /// TD 模式右側塔按鈕的 UI Text node（動態 Vec：N 個塔來自 td_template_order 長度）
    #[visit(skip)] #[reflect(hidden)]
    ui_td_tower_buttons: Vec<Handle<Text>>,
    /// 塔按鈕的 hit-test rects（x, y, w, h）—— 每 frame 依 window_size 更新
    #[visit(skip)] #[reflect(hidden)]
    td_tower_button_rects: Vec<(f32, f32, f32, f32)>,
    /// 目前玩家選中的塔 unit_id（例如 "tower_dart"）；None 表示未選
    #[visit(skip)] #[reflect(hidden)]
    selected_tower_kind: Option<String>,
    /// Start Round 按鈕 UI Text node。
    #[visit(skip)] #[reflect(hidden)]
    ui_start_round_button: Handle<Text>,
    /// Start Round 按鈕 hit-test rect（每 frame 依 window_size 更新）。
    #[visit(skip)] #[reflect(hidden)]
    start_round_button_rect: (f32, f32, f32, f32),
    /// TD 當前已完成的波數（1-based 概念；後端推送 `game/round` 時更新）。
    /// 0 表示還沒開始第一波。
    #[visit(skip)] #[reflect(hidden)]
    current_round: u32,
    /// TD 總波數（後端推送 `game/round` 時更新）。
    #[visit(skip)] #[reflect(hidden)]
    total_rounds: u32,
    /// TD 本波是否正在跑（true = 按鈕變灰；false = 按鈕可按）。
    #[visit(skip)] #[reflect(hidden)]
    round_is_running: bool,
    /// 是否為 TD 模式：由首次收到 hero.stats 有 lives>0 時設 true。
    /// 影響相機（固定不跟隨英雄）、zoom（拉遠讓整張路徑可見）。
    #[visit(skip)] #[reflect(hidden)]
    is_td_mode: bool,
    /// 是否已經針對 TD 模式調整過相機 ortho（避免每 tick 重設）。
    #[visit(skip)] #[reflect(hidden)]
    td_camera_configured: bool,
    /// 玩家點選中的已蓋塔 entity id（右側顯示 sell 面板、地圖上畫射程圈）；None = 未選取
    #[visit(skip)] #[reflect(hidden)]
    selected_tower_entity: Option<u32>,
    /// 選中塔右側面板：塔名+等級 文字
    #[visit(skip)] #[reflect(hidden)]
    ui_td_sell_name_text: Handle<Text>,
    /// 選中塔右側面板：Sell 按鈕文字
    #[visit(skip)] #[reflect(hidden)]
    ui_td_sell_button_text: Handle<Text>,
    /// Sell 按鈕 hit-test rect（每 frame 依 window_size 更新；塔未選時放螢幕外）
    #[visit(skip)] #[reflect(hidden)]
    td_sell_button_rect: (f32, f32, f32, f32),
    /// 選中塔右側面板：3 條路線升級按鈕文字
    #[visit(skip)] #[reflect(hidden)]
    ui_td_upgrade_buttons: [Handle<Text>; 3],
    /// 3 條路線升級按鈕 hit-test rect；塔未選時放螢幕外
    #[visit(skip)] #[reflect(hidden)]
    td_upgrade_button_rects: [(f32, f32, f32, f32); 3],
    /// 進行中的爆炸特效（Bomb 塔命中時 spawn）
    #[visit(skip)] #[reflect(hidden)]
    active_explosions: Vec<ActiveExplosion>,
    /// TD 路徑 check_points（render 座標）— 供 placement 預覽計算是否壓到路
    #[visit(skip)] #[reflect(hidden)]
    td_paths_render: Vec<Vec<Vector2<f32>>>,
    /// TD 禁止通行多邊形（render 座標）— 供 placement 預覽計算是否壓到 region
    #[visit(skip)] #[reflect(hidden)]
    td_regions_render: Vec<Vec<Vector2<f32>>>,
    /// 後端送來的 TD 塔 template 快取（unit_id → TdTemplate）
    #[visit(skip)] #[reflect(hidden)]
    td_templates: HashMap<String, TdTemplate>,
    /// Template 的顯示順序（= DLL `units()` 註冊順序），供按鈕排版用
    #[visit(skip)] #[reflect(hidden)]
    td_template_order: Vec<String>,
    #[visit(skip)] #[reflect(hidden)]
    client_projectiles: HashMap<u32, ClientProjectile>,
    /// P7 layered prediction：key = projectile id（server `e.id()`）。
    /// 跟 client_projectiles 同 id；前者是視覺軌跡，這個是傷害預測 ledger。
    /// 視覺命中後 ClientProjectile 移除但 PendingPredDmg 留著，等 heartbeat
    /// 的 in_flight_projectiles 或 D event 真結算才移除。
    #[visit(skip)] #[reflect(hidden)]
    pending_pred_dmg: HashMap<u32, PendingPredDmg>,
    #[visit(skip)] #[reflect(hidden)]
    heartbeat: HeartbeatInfo,

    /// Per-frame profile（每 60 frame 輸出一行 omfx_frame_profile log）。
    #[visit(skip)] #[reflect(hidden)]
    frame_profile: FrameProfile,

    // --- Backend Process ---
    /// Drops → kills backend. Held for the whole Game lifetime so that any exit
    /// path (normal, panic, force-close on Windows via Job Object) brings the
    /// backend down with us.
    #[visit(skip)] #[reflect(hidden)]
    backend_guard: Option<BackendGuard>,

    #[visit(skip)] #[reflect(hidden)]
    pending_label_deletions: Vec<Handle<Text>>,

    /// UI Text labels for entities surfaced by `render_bridge` (sim_runner-backed).
    /// Keyed by `entity_id`. Created on first render of an entity, updated each
    /// frame, removed when the entity drops out of the sim snapshot.
    #[visit(skip)] #[reflect(hidden)]
    sim_entity_labels: HashMap<u32, SimEntityLabel>,

    /// Batched-mesh slot ownership for sim_runner-backed entities, keyed by
    /// `entity_id`. body_batch + hp_batch slots are allocated on first
    /// sighting and freed when the entity drops from the snapshot. This is
    /// the draw-call-saving path: 1000 creeps + 1000 towers ≈ 2 draws total
    /// (one per batch), vs. one node-per-entity which is one draw per quad.
    #[visit(skip)] #[reflect(hidden)]
    sim_entity_slots: HashMap<u32, render_bridge::SimEntitySlots>,

    /// Wall-clock timestamp of first frame; used by the
    /// `OMFX_AUTO_START_AFTER_SEC` / `OMFX_AUTO_EXIT_AFTER_SEC` smoke loop
    /// so a single `cargo run` can reproduce a Start-Round-then-die scenario
    /// without manual clicks. None until first update() tick.
    #[visit(skip)] #[reflect(hidden)]
    auto_clock_start: Option<std::time::Instant>,
    /// Latched once the auto Start-Round input has been emitted, so the
    /// dispatcher only sees one StartRound (subsequent host-side reads
    /// would warn "round already running").
    #[visit(skip)] #[reflect(hidden)]
    auto_start_sent: bool,

    // --- UI ---
    #[visit(skip)] #[reflect(hidden)]
    ui_status_text: Handle<Text>,
    #[visit(skip)] #[reflect(hidden)]
    ui_hud_text: Handle<Text>,
    /// 左下角英雄屬性面板（多行：name/title/Lv/XP/SP/三圍/HP/Gold + 4 技能等級）
    #[visit(skip)] #[reflect(hidden)]
    ui_hero_stats_panel: Handle<Text>,
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
    /// Ctrl 按住：蓋塔後不自動取消選塔模式（方便一次連蓋多個）
    #[visit(skip)] #[reflect(hidden)]
    ctrl_held: bool,
    /// Alt 按住：強制顯示 name label（即使 entity 數超過 NAME_LABEL_HIDE_THRESHOLD）
    #[visit(skip)] #[reflect(hidden)]
    alt_held: bool,
    #[visit(skip)] #[reflect(hidden)]
    game_ended: bool,
    #[visit(skip)] #[reflect(hidden)]
    viewport_sync_elapsed: f32,
    /// 上一次實際送出的 viewport (cx, cy, hw, hh)；值不變就跳過送出避免 omb log 洗版與
    /// 無謂的 KCP decode + mutex/channel work。reconnect 時 reset 為 None 以強制重送。
    #[visit(skip)] #[reflect(hidden)]
    last_sent_viewport: Option<(f32, f32, f32, f32)>,
    /// Camera 目前所在 render-world 座標（用於滑鼠座標換算與 label 螢幕換算）
    #[visit(skip)] #[reflect(hidden)]
    camera_world_pos: Vector2<f32>,
    /// 本秒累計的網路事件 logical (decompressed) payload bytes — UI 看「應用層」量
    #[visit(skip)] #[reflect(hidden)]
    net_bytes_current: u64,
    /// 上一秒的總 logical bytes，供顯示用
    #[visit(skip)] #[reflect(hidden)]
    net_bytes_last_sec: u64,
    /// 本秒累計的真實 wire bytes (壓縮後 + framing) — UI 看真實 bandwidth
    #[visit(skip)] #[reflect(hidden)]
    net_wire_bytes_current: u64,
    /// 上一秒的真實 wire bytes
    #[visit(skip)] #[reflect(hidden)]
    net_wire_bytes_last_sec: u64,
    /// 計時：每滿 1 秒 roll over
    #[visit(skip)] #[reflect(hidden)]
    net_stats_elapsed: f32,
    /// FPS 顯示字串（例 "FPS 250 (4.0ms)"），來自 Fyrox renderer 的 frames_per_second
    /// 統計（plugin update 是 fixed 60 Hz tick，自算 frame count 沒意義）。
    #[visit(skip)] #[reflect(hidden)]
    fps_display: String,
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

/// 前端緩存的單一 buff（由 hero.stats 的 "buffs" 陣列驅動）
#[derive(Default, Debug, Clone)]
struct LocalBuff {
    id: String,
    /// 剩餘秒數；-1.0 代表無限期（toggle）
    remaining: f32,
    /// 原始 payload（例 sniper_mode = {range_bonus:100, damage_bonus:0.15, ...}）
    payload: serde_json::Value,
}

/// 前端緩存的 hero 狀態（由 hero.stats / hero.inventory 事件驅動）
#[derive(Default, Debug, Clone)]
struct LocalHeroState {
    /// 英雄在後端的 entity id，camera 跟隨用
    entity_id: Option<u32>,
    name: String,
    title: String,
    level: i32,
    xp: i32,
    xp_next: i32,
    skill_points: i32,
    gold: i32,
    /// TD 模式的玩家生命；非 TD 模式後端不推送此值，保持初值 0
    lives: i32,
    hp: f32,
    max_hp: f32,
    strength: i32,
    agility: i32,
    intelligence: i32,
    /// "strength" / "agility" / "intelligence"；決定左下角面板 primary * 標記位置
    primary_attribute: String,
    armor: f32,
    magic_resist: f32,
    move_speed: f32,
    attack_damage: f32,
    /// 秒/攻（asd）；0 代表不攻擊
    attack_interval: f32,
    attack_range: f32,
    bullet_speed: f32,
    /// 當前在 BuffStore 裡的 buff 快照（由後端 `hero.stats` 每 0.3 秒 push 一次）。
    /// remaining < 0 代表無限期（例：toggle 型 sniper_mode）。前端每 tick 本地
    /// 遞減 remaining，讓倒數看起來連續；下次 push 會重設成權威值。
    buffs: Vec<LocalBuff>,
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

        // Orthographic 3D camera 放在 z=-100，default look=+Z（不旋轉）。
        // 不旋轉相機，讓 `look_at_rh(eye, eye+look, up)` 算出的 side = world -X，
        // 對接 omfx 各 set_position 寫死的 `(-bx, by, z)` x-flip 慣例：
        // backend +X → world -X → view +X → 螢幕右。詳情見 Z 常數區塊註解。
        self.camera = CameraBuilder::new(
            BaseBuilder::new().with_local_transform(
                TransformBuilder::new()
                    .with_local_position(Vector3::new(0.0, 0.0, -100.0))
                    .build(),
            ),
        )
        .with_projection(Projection::Orthographic(OrthographicProjection {
            z_near: 0.1,
            z_far: 1000.0,
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

        // status bar 緊貼螢幕頂端，留更多 UI 空間給下方資訊
        self.ui_status_text = TextBuilder::new(
            WidgetBuilder::new()
                .with_desired_position(Vector2::new(10.0, 2.0))
                .with_width(1900.0)
                .with_foreground(Brush::Solid(Color::from_rgba(0, 0, 0, 255)).into()),
        )
        .with_text("Connecting...".to_string())
        .with_font_size(18.0.into())
        .build(&mut ui.build_ctx());

        // HUD 文字（左上角，緊貼 status bar 下方）
        self.ui_hud_text = TextBuilder::new(
            WidgetBuilder::new()
                .with_desired_position(Vector2::new(10.0, 24.0))
                .with_width(1900.0)
                .with_foreground(Brush::Solid(Color::from_rgba(0, 0, 0, 255)).into()),
        )
        .with_text("".to_string())
        .with_font_size(18.0.into())
        .build(&mut ui.build_ctx());

        // 左下角英雄屬性面板（多行）；實際位置由 update() 依 window_size 重定位
        self.ui_hero_stats_panel = TextBuilder::new(
            WidgetBuilder::new()
                .with_desired_position(Vector2::new(10.0, 400.0))
                .with_width(480.0)
                .with_foreground(Brush::Solid(Color::from_rgba(0, 0, 0, 255)).into()),
        )
        .with_text("".to_string())
        .with_font_size(18.0.into())
        .build(&mut ui.build_ctx());

        // 選中塔 Sell 面板（右側，塔被選取時才定位到可見位置）
        {
            self.ui_td_sell_name_text = TextBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(-9999.0, -9999.0))
                    .with_width(240.0)
                    .with_foreground(Brush::Solid(Color::from_rgba(0, 0, 0, 255)).into()),
            )
            .with_text(String::new())
            .with_font_size(18.0.into())
            .build(&mut ui.build_ctx());

            self.ui_td_sell_button_text = TextBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(-9999.0, -9999.0))
                    .with_width(240.0)
                    .with_foreground(Brush::Solid(Color::from_rgba(120, 20, 20, 255)).into()),
            )
            .with_text(String::new())
            .with_font_size(20.0.into())
            .build(&mut ui.build_ctx());
            self.td_sell_button_rect = (-9999.0, -9999.0, 240.0, 42.0);

            // 3 條路線升級按鈕（塔被選取時才定位到可見位置）
            for i in 0..3 {
                self.ui_td_upgrade_buttons[i] = TextBuilder::new(
                    WidgetBuilder::new()
                        .with_desired_position(Vector2::new(-9999.0, -9999.0))
                        .with_width(240.0)
                        .with_foreground(Brush::Solid(Color::from_rgba(20, 60, 120, 255)).into()),
                )
                .with_text(String::new())
                .with_font_size(18.0.into())
                .build(&mut ui.build_ctx());
                self.td_upgrade_button_rects[i] = (-9999.0, -9999.0, 240.0, 38.0);
            }
        }

        // Start Round 按鈕（右下角）
        {
            self.ui_start_round_button = TextBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(-9999.0, -9999.0))
                    .with_width(240.0)
                    .with_foreground(Brush::Solid(Color::from_rgba(0, 80, 0, 255)).into()),
            )
            .with_text("▶ Start Round 1".to_string())
            .with_font_size(22.0.into())
            .build(&mut ui.build_ctx());
            self.start_round_button_rect = (-9999.0, -9999.0, 240.0, 48.0);
        }

        // TD 模式右側塔按鈕（text-only，動態 Vec）
        // 收到 game/tower_templates 事件後才建；此時空 Vec
        self.ui_td_tower_buttons = Vec::new();
        self.td_tower_button_rects = Vec::new();

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

        // Phase 5.1: NetworkBridge consumer cut. Phase 4.5 flipped omb's
        // `legacy_broadcast` default OFF so the 0x02 GameEvent stream is
        // silent. Lockstep (KCP tags 0x10–0x16) is now the sole input/state
        // wire — driven by `lockstep_handle` below + sim_runner + render_bridge.
        self.connection_status = ConnectionStatus::Connected;

        // Lockstep wire-up. Runs as a background thread on its own KCP
        // connection. TickBatch / StateHash drive the local sim_runner
        // (Phase 3.x) which renders via render_bridge (Phase 4.2).
        let lockstep_player_name = std::env::var("OMB_LOCKSTEP_PLAYER_NAME")
            .unwrap_or_else(|_| format!("{}_lockstep", player_name));
        self.lockstep_handle = Some(lockstep_client::spawn_lockstep_client(
            server_addr,
            lockstep_player_name,
        ));

        // Phase 3.2: spawn the local sim_runner worker. It blocks on the
        // master_seed channel — Phase 3.3 will wire LockstepClient's
        // GameStart handler to send via `sim_runner_handle.master_seed_tx`.
        // For Phase 3.2 the worker simply parks until shutdown, which
        // verifies the worker thread spawn + symbol resolution.
        {
            use std::path::PathBuf;
            // Default to omb/scripts/ where run.bat copies the freshly-built DLL
            // (debug or release — whichever profile run.bat used). load_scripts_dir
            // takes the parent dir and scans for .dll, so this works for both.
            let dll_path: PathBuf = std::env::var("OMB_DLL_PATH")
                .map(PathBuf::from)
                .unwrap_or_else(|_| {
                    PathBuf::from("D:/omoba/omb/scripts/base_content.dll")
                });
            // omobab::ServerSetting::default reads "game.toml" by relative path
            // (works for omobab.exe with cwd=omb). omfx process cwd is elsewhere,
            // so point the lazy_static at an absolute path.
            if std::env::var("OMB_GAME_TOML").is_err() {
                std::env::set_var("OMB_GAME_TOML", "D:/omoba/omb/game.toml");
            }
            // Sync sim_runner's scene with omb's by parsing STORY from the same
            // game.toml. Otherwise sim_runner loads MVP_1 while omb loads TD_1
            // (per game.toml STORY) and the two ECS worlds diverge — sim_runner
            // ends up with MVP_1's training enemies / blockers (~410 ghost
            // entities) while creep paths / waves don't match omb's.
            let scene_path: PathBuf = std::env::var("OMB_SCENE_PATH")
                .map(PathBuf::from)
                .unwrap_or_else(|_| {
                    let toml_path = std::env::var("OMB_GAME_TOML")
                        .unwrap_or_else(|_| "D:/omoba/omb/game.toml".to_string());
                    let story = std::fs::read_to_string(&toml_path)
                        .ok()
                        .and_then(|s| {
                            s.lines()
                                .map(str::trim)
                                .filter(|l| !l.starts_with('#'))
                                .find_map(|l| {
                                    let mut parts = l.splitn(2, '=');
                                    let key = parts.next()?.trim();
                                    if key != "STORY" { return None; }
                                    let val = parts.next()?.trim()
                                        .trim_start_matches('"')
                                        .trim_end_matches('"')
                                        .to_string();
                                    Some(val)
                                })
                        })
                        .unwrap_or_else(|| {
                            log::warn!("game.toml missing STORY; falling back to MVP_1");
                            "MVP_1".to_string()
                        });
                    log::info!("sim_runner: scene STORY={} (from game.toml)", story);
                    PathBuf::from(format!("D:/omoba/omb/Story/{}", story))
                });
            log::info!(
                "Phase 3.2 sim_runner spawn: dll={:?} scene={:?}",
                dll_path,
                scene_path
            );
            self.sim_runner_handle = Some(sim_runner::spawn_sim_runner(dll_path, scene_path));
        }

        // Phase 5.1: legacy NetworkBridge / EventBuffer / network_entities /
        // client_projectiles state no longer initialized — the consumer is gone.
        // (Field declarations remain to be removed in the same Phase 5.1 cut.)

        Ok(())
    }

    fn on_deinit(&mut self, _context: PluginContext) -> GameResult {
        // Phase 5.1: NetworkBridge no longer owned; nothing to drop here.
        // Drop lockstep client (input_tx drop + events_rx drop → bg thread
        // exits on next iteration when its select sees disconnected channels).
        self.lockstep_handle = None;
        // Drop sim_runner. Channel disconnect signals the worker to
        // exit (whether it's still blocked on master_seed_rx.recv() or
        // looping on tick_input_rx.recv()).
        self.sim_runner_handle = None;

        // Drop the backend guard — its Drop impl kills the child and closes the Job Object.
        // (If Drop doesn't run, e.g. on hard kill, the OS still terminates the backend
        // thanks to JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE.)
        self.backend_guard = None;

        Ok(())
    }

    fn update(&mut self, context: &mut PluginContext) -> GameResult {
        let scene = &mut context.scenes[self.scene];
        // 每 frame 清掉 drawing_context 的 line buffer，避免累積到無限大導致 FPS 為 0。
        // 後續 phase（爆炸 / 路徑 debug 等）會 push 新的 line 進來。
        scene.drawing_context.clear_lines();
        let frame_t0 = std::time::Instant::now();

        // Smoke-loop hooks (read once on first update). Both env vars are
        // independent; either or both may be set. Used by automated test
        // runs so a single `run.bat` can launch → press Start Round →
        // exit, without a human clicking buttons.
        let now = std::time::Instant::now();
        if self.auto_clock_start.is_none() {
            self.auto_clock_start = Some(now);
        }
        let elapsed_s = now
            .duration_since(self.auto_clock_start.unwrap())
            .as_secs_f32();
        if !self.auto_start_sent {
            if let Ok(v) = std::env::var("OMFX_AUTO_START_AFTER_SEC") {
                if let Ok(threshold) = v.parse::<f32>() {
                    if elapsed_s >= threshold {
                        let input = omoba_core::kcp::game_proto::PlayerInput {
                            action: Some(
                                omoba_core::kcp::game_proto::player_input::Action::StartRound(
                                    omoba_core::kcp::game_proto::StartRound {},
                                ),
                            ),
                        };
                        self.send_lockstep_input(input);
                        log::info!(
                            "[auto-smoke] Start Round sent at t={:.2}s (OMFX_AUTO_START_AFTER_SEC={})",
                            elapsed_s, v
                        );
                        self.auto_start_sent = true;
                    }
                }
            }
        }
        if let Ok(v) = std::env::var("OMFX_AUTO_EXIT_AFTER_SEC") {
            if let Ok(threshold) = v.parse::<f32>() {
                if elapsed_s >= threshold {
                    log::info!(
                        "[auto-smoke] exiting at t={:.2}s (OMFX_AUTO_EXIT_AFTER_SEC={})",
                        elapsed_s, v
                    );
                    std::process::exit(0);
                }
            }
        }

        // Lazy init shared sprite resources on first frame.
        if self.sprite_resources.is_none() {
            self.sprite_resources = Some(sprite_resources::SharedSpriteResources::new());
        }

        // Lazy init batched meshes — 4 個獨立 batch（body / hp / facing），N entity = 3 draws。
        if self.body_batch.is_none() {
            let material = self.sprite_resources.as_ref().unwrap().material.clone();
            self.body_batch = Some(sprite_resources::BatchedSpriteMesh::new(
                scene, 4096, material,
            ));
        }
        if self.hp_batch.is_none() {
            let material = self.sprite_resources.as_ref().unwrap().material.clone();
            // 8192 = 4096 entity × 2 (bg + fg)
            self.hp_batch = Some(sprite_resources::BatchedSpriteMesh::new(
                scene, 8192, material,
            ));
        }
        if self.facing_batch.is_none() {
            let material = self.sprite_resources.as_ref().unwrap().material.clone();
            self.facing_batch = Some(sprite_resources::BatchedSpriteMesh::new(
                scene, 4096, material,
            ));
        }

        // Phase 5.1: connection status / initial viewport drain removed
        // (tracked the legacy NetworkBridge handshake which no longer exists).
        // Lockstep `Connected` event below is now the canonical "we're up" signal.

        // Phase 3.3: drain lockstep events and forward to sim_runner.
        // - Connected → push master_seed (unblocks worker's blocking recv)
        // - TickBatch → convert inputs (omoba_core proto type → omobab proto
        //   type) and push as TickBatchPayload so the worker advances its
        //   ECS dispatcher one tick. Both crates generate the same
        //   `proto/game.proto` independently, so we round-trip via prost
        //   encode/decode at the boundary instead of hand-mapping every
        //   PlayerInput variant.
        // - StateHash → log only (Phase 3.4 will compare against the
        //   sim_runner's local hash for desync detection).
        // - Disconnected → log.
        // TickBatch is sampled every 60 ticks (~1s @ 60Hz) to avoid log spam.
        if let (Some(ref lh), Some(ref sim)) = (
            self.lockstep_handle.as_ref(),
            self.sim_runner_handle.as_ref(),
        ) {
            while let Ok(ev) = lh.events_rx.try_recv() {
                match ev {
                    lockstep_client::LockstepEvent::Connected { master_seed, player_id } => {
                        log::info!(
                            "[lockstep] connected master_seed=0x{:016x} player_id={}",
                            master_seed, player_id
                        );
                        if let Err(e) = sim.master_seed_tx.send(master_seed) {
                            log::error!("[lockstep] failed to forward master_seed: {}", e);
                        }
                    }
                    lockstep_client::LockstepEvent::TickBatch { tick, inputs, server_events } => {
                        // Phase 4.3: track latest sim tick for input target_tick math.
                        self.current_sim_tick = tick;
                        if tick % 60 == 0 {
                            log::debug!(
                                "[lockstep] tick={} inputs={} events={}",
                                tick, inputs.len(), server_events.len()
                            );
                        }
                        // Bridge omoba_core's PlayerInput type → omobab's
                        // PlayerInput type via prost re-encode. They are
                        // identical wire format but distinct Rust types.
                        let converted: Vec<(u32, sim_runner::PlayerInput)> = inputs
                            .into_iter()
                            .filter_map(|(pid, inp)| {
                                convert_player_input(&inp).map(|out| (pid, out))
                            })
                            .collect();
                        let payload = sim_runner::TickBatchPayload {
                            tick,
                            inputs: converted,
                        };
                        if let Err(e) = sim.tick_input_tx.send(payload) {
                            log::error!("[lockstep] failed to forward tick batch: {}", e);
                        }
                        // server_events: Phase 3.3 ignored; Phase 5+ will
                        // route them into the sim's event sink.
                    }
                    lockstep_client::LockstepEvent::StateHash { tick, hash } => {
                        log::info!("[lockstep] state_hash@{}=0x{:016x}", tick, hash);
                        // Phase 3.4 will compare this against the
                        // sim_runner's locally computed hash.
                    }
                    lockstep_client::LockstepEvent::Disconnected { reason } => {
                        log::warn!("[lockstep] disconnected: {}", reason);
                    }
                }
            }
        }

        // Phase 3.4: read latest sim snapshot and (stub) update render
        // bridge. Acquired with `try_lock` so a slow render frame doesn't
        // block the sim worker — if the lock is contended we just skip
        // this frame and pick up the next snapshot. Phase 4 will replace
        // the stub `update` body with real Fyrox sprite spawn / update /
        // despawn, retiring the NetworkBridge GameEvent → sprite pipeline
        // below for the entities the sim authoritatively owns.
        if let Some(ref sim) = self.sim_runner_handle {
            if let Ok(snapshot) = sim.state.try_lock() {
                self.render_bridge.update(&*snapshot, scene);

                // Phase 5.x: HUD heartbeat sourced from sim snapshot
                // (NetworkBridge GameEvent stream was cut in Phase 5.1; this
                // restores tick / entity / hero / creep counts on the top-line
                // status text and hp / max_hp on the hero panel).
                self.heartbeat.tick = snapshot.tick as u64;
                // sim_runner runs at the lockstep tick rate (60 Hz pacer).
                self.heartbeat.game_time = (snapshot.tick as f64) / 60.0;
                self.heartbeat.entity_count = snapshot.entities.len() as u64;
                self.heartbeat.hero_count = snapshot.entities.iter()
                    .filter(|e| matches!(e.kind, sim_runner::EntityKind::Hero))
                    .count() as u64;
                self.heartbeat.creep_count = snapshot.entities.iter()
                    .filter(|e| matches!(e.kind, sim_runner::EntityKind::Creep))
                    .count() as u64;

                // First Hero entity drives the hero panel. EntityRenderData now
                // carries hero metadata (name / title / level / xp / gold /
                // strength / agility / intelligence / primary_attribute) so the
                // panel can render the same way the legacy NetworkBridge path did.
                if let Some(hero) = snapshot.entities.iter()
                    .find(|e| matches!(e.kind, sim_runner::EntityKind::Hero))
                {
                    self.hero_state.hp = hero.hp as f32;
                    self.hero_state.max_hp = hero.max_hp as f32;
                    self.hero_state.name = hero.hero_name.clone();
                    self.hero_state.title = hero.hero_title.clone();
                    self.hero_state.level = hero.hero_level;
                    self.hero_state.xp = hero.hero_xp;
                    self.hero_state.xp_next = hero.hero_xp_next;
                    self.hero_state.skill_points = hero.hero_skill_points;
                    self.hero_state.primary_attribute = hero.hero_primary_attribute.clone();
                    self.hero_state.strength = hero.hero_strength;
                    self.hero_state.agility = hero.hero_agility;
                    self.hero_state.intelligence = hero.hero_intelligence;
                    self.hero_state.gold = hero.gold;
                }
            }
        }

        // 網路流量統計：每秒 roll over
        self.net_stats_elapsed += context.dt;
        if self.net_stats_elapsed >= 1.0 {
            self.net_bytes_last_sec = self.net_bytes_current;
            self.net_bytes_current = 0;
            self.net_wire_bytes_last_sec = self.net_wire_bytes_current;
            self.net_wire_bytes_current = 0;
            self.net_stats_elapsed -= 1.0;
        }

        // FPS 顯示：直接用 Fyrox renderer 統計的真實 render fps（plugin update
        // 自己是 fixed 60 Hz tick，自算 frame_count 永遠 60，沒意義）。
        // last_fps 由 frame_profile.record_render_stats 在每 frame 更新。
        let render_fps = self.frame_profile.last_fps;
        if render_fps > 0 {
            let frame_ms = 1000.0 / render_fps as f32;
            self.fps_display = format!("FPS {} ({:.1}ms)", render_fps, frame_ms);
        }

        // Phase 5.1: NetworkBridge event drain + EventBuffer + heartbeat hp/pos
        // reconciliation removed. Lockstep TickBatch (above) is the sole tick
        // source; render_bridge owns sprite spawn/update/despawn from sim state.
        let t_events = std::time::Instant::now();
        let events_drained_local: u64 = 0;
        let events_ns = t_events.elapsed().as_nanos();

        // 4. Interpolate entity positions (client-side lerp)
        let t_interp = std::time::Instant::now();
        let dt = context.dt;
        // P7 layered: 預先 sum 每個 target 的 applied 預測扣血，HP bar 渲染時減去。
        // O(P) where P = pending count，通常 < 50。
        let pending_dmg_by_target: HashMap<u32, f32> = {
            let mut m: HashMap<u32, f32> = HashMap::new();
            for p in self.pending_pred_dmg.values() {
                if p.applied {
                    *m.entry(p.target_id).or_insert(0.0) += p.dmg;
                }
            }
            m
        };
        for (&entity_id, entity) in self.network_entities.iter_mut() {
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
            // P4: velocity extrapolation takes priority over lerp for creeps
            // with an active segment. After `extrap_duration` elapses we lock
            // at `target_position` until the next creep.M arrives; that's how
            // we render idle at a waypoint when the server hasn't decided on
            // the next one yet (e.g. TD path end, blocked by collision).
            let pos = if entity.extrap_velocity > 1.0 && entity.extrap_duration > 0.0 {
                entity.extrap_elapsed += dt;
                if entity.extrap_elapsed >= entity.extrap_duration {
                    entity.target_position
                } else {
                    let travel_backend = entity.extrap_velocity * entity.extrap_elapsed;
                    let travel_render = travel_backend * WORLD_SCALE;
                    entity.extrap_start_pos
                        + Vector2::new(
                            entity.extrap_direction.x * travel_render,
                            entity.extrap_direction.y * travel_render,
                        )
                }
            } else {
                let t = (entity.lerp_elapsed / entity.lerp_duration).clamp(0.0, 1.0);
                entity.prev_position.lerp(&entity.target_position, t)
            };
            entity.position = pos;

            // [DEBUG-STRESS] 抓 NaN / Inf / 怪座標：creep 不該飛出 ±5000 範圍
            if entity.entity_type == "creep" {
                if !pos.x.is_finite() || !pos.y.is_finite() || pos.x.abs() > 5000.0 || pos.y.abs() > 5000.0 {
                    log::warn!(
                        "🤡 weird creep pos id={} pos=({},{}) prev=({},{}) target=({},{}) lerp_t_dur={}/{} extrap_v={} extrap_dur={}",
                        entity_id, pos.x, pos.y,
                        entity.prev_position.x, entity.prev_position.y,
                        entity.target_position.x, entity.target_position.y,
                        entity.lerp_elapsed, entity.lerp_duration,
                        entity.extrap_velocity, entity.extrap_duration,
                    );
                }
            }

            // Body sprite 透過 batched mesh — write_quad 進 cpu_mirror，最後一次性 flush。
            // X 取負讓 +X world 投到螢幕右。
            if let Some(batch) = self.body_batch.as_mut() {
                batch.write_quad(
                    entity.body_slot,
                    &sprite_resources::QuadParams {
                        center: Vector2::new(-pos.x, pos.y),
                        size: Vector2::new(entity.body_size, entity.body_size),
                        color: entity.body_color,
                        rotation: 0.0,
                        z: entity.body_z,
                    },
                );
            }

            // Update HP bar positions — 走 hp_batch
            if let (Some(bg_slot), Some(fg_slot), Some((h, m))) =
                (entity.hp_bg_slot, entity.hp_fg_slot, entity.health)
            {
                let bar_y = pos.y + 0.3;
                // P7 layered display HP：authoritative h 減去已 applied 但 server 還沒
                // 反映的預測扣血，讓 visual 在子彈視覺命中當下就掉血、heartbeat reconcile
                // 後 pending 從 retain 被移除、h 也對應降下，畫面值不會跳。
                let pending_dmg = pending_dmg_by_target.get(&entity_id).copied().unwrap_or(0.0);
                let display_h = (h - pending_dmg).max(0.0);
                let hp_ratio = (display_h / m).clamp(0.0, 1.0);
                let bar_w = 0.8_f32;
                let bar_h = 0.06_f32;

                if let Some(batch) = self.hp_batch.as_mut() {
                    // bg：固定寬度
                    batch.write_quad(
                        bg_slot,
                        &sprite_resources::QuadParams {
                            center: Vector2::new(-pos.x, bar_y),
                            size: Vector2::new(bar_w, bar_h),
                            color: [0, 0, 0, 255],
                            rotation: 0.0,
                            z: Z_HP_BAR,
                        },
                    );
                    // fg：寬度按 hp_ratio，左對齊（中心隨 ratio 內縮）
                    let fg_w = bar_w * hp_ratio;
                    let fg_offset = (bar_w - fg_w) * 0.5;
                    batch.write_quad(
                        fg_slot,
                        &sprite_resources::QuadParams {
                            center: Vector2::new(-pos.x - fg_offset, bar_y),
                            size: Vector2::new(fg_w, bar_h),
                            color: [0, 220, 0, 255],
                            rotation: 0.0,
                            z: Z_HP_BAR - 0.01,
                        },
                    );
                }
            }

            // 更新面向箭頭位置與角度 — 走 facing_batch
            if let Some(slot) = entity.facing_slot {
                let size: f32 = match entity.entity_type.as_str() {
                    "hero" => 0.4,
                    "creep" | "enemy" => 0.3,
                    "unit" | "tower" => 0.4,
                    _ => 0.3,
                };
                let length = (size * 0.7).max(0.12);
                let thickness = (size * 0.15).max(0.04);
                let render_angle = std::f32::consts::PI - entity.facing;
                let offset_x = (length * 0.5) * render_angle.cos();
                let offset_y = (length * 0.5) * render_angle.sin();
                if let Some(batch) = self.facing_batch.as_mut() {
                    batch.write_quad(
                        slot,
                        &sprite_resources::QuadParams {
                            center: Vector2::new(-pos.x + offset_x, pos.y + offset_y),
                            size: Vector2::new(length, thickness),
                            color: [255, 200, 0, 255],
                            rotation: render_angle,
                            z: Z_HP_BAR - 0.02,
                        },
                    );
                }
            }

            // 碰撞半徑圓環：跟隨 entity 中心平移（COLLISION_RING_ENABLED 路徑：
            // 用 RectangleBuilder 24-segment node、每幀 transform update。stress 1000
            // entity = 24K scene node 太重，預設關。
            for (handle, offset) in &entity.collision_ring {
                scene.graph[*handle]
                    .local_transform_mut()
                    .set_position(Vector3::new(-(pos.x + offset.x), pos.y + offset.y, Z_RING));
            }
        }

        // Per-frame debug：每個 entity 的 collision ring 畫成 SceneDrawingContext lines
        // (千個 entity 加起來 1 個 draw call，跟 COLLISION_RING_ENABLED 走 scene node 那條路
        // 完全分開)。要看 hero / creep / tower 互相阻擋的真實 collision 範圍時用。
        if DEBUG_COLLISION_RINGS {
            for entity in self.network_entities.values() {
                if entity.collision_radius_render <= 0.0 {
                    continue;
                }
                if !matches!(entity.entity_type.as_str(), "hero" | "creep" | "unit" | "tower") {
                    continue;
                }
                // 顏色依類型區分，方便辨識
                let color = match entity.entity_type.as_str() {
                    "hero" => Color::from_rgba(80, 220, 80, 220),    // 綠
                    "creep" => Color::from_rgba(255, 60, 60, 220),   // 紅
                    "unit" | "tower" => Color::from_rgba(80, 160, 255, 220), // 藍
                    _ => Color::from_rgba(255, 255, 255, 220),
                };
                add_circle_lines(
                    scene,
                    entity.position,
                    entity.collision_radius_render,
                    24,
                    color,
                    Z_RING,
                );
            }
        }

        // Phase 5.x: write sim_runner-backed entities into body_batch + hp_batch
        // BEFORE flushing. Replaces the per-entity RectangleBuilder spawn from
        // earlier 4.2 render_bridge — each entity used to be a separate scene
        // node = separate draw call (1000 entities → 3000+ draws). Now the
        // entire entity set goes through 2-3 batched meshes = 2-3 draws total.
        self.update_sim_batches();

        // Batched mesh flush：interp loop 寫進各 batch 的 cpu_mirror，這裡一次性
        // upload 整批 vertex buffer 到 GPU。每個 batch = 1 個 mesh = 1 個 draw call。
        if let Some(batch) = self.body_batch.as_mut() {
            batch.flush(scene);
        }
        if let Some(batch) = self.hp_batch.as_mut() {
            batch.flush(scene);
        }
        if let Some(batch) = self.facing_batch.as_mut() {
            batch.flush(scene);
        }

        // Fyrox 1.0.1 doesn't auto-update hierarchical data per-frame (docs at
        // fyrox-impl-1.0.1/src/scene/graph/mod.rs:565). Without this call, our
        // 3D Mesh nodes have stale global_transform = identity → all sprites
        // render at world origin (0,0,0). Force-update once per frame.
        scene.graph.update_hierarchical_data();

        let interp_ns = t_interp.elapsed().as_nanos();

        // TD 塔預覽圓圈：選中塔時每 frame 在滑鼠位置重畫 footprint + 攻擊範圍兩圈。
        // 用 SceneDrawingContext（drawing_context 已在 update() 開頭被 clear_lines()），
        // 不再 per-frame 增刪 24+48=72 個 RectangleBuilder node。
        let t_visual = std::time::Instant::now();
        {
            if let Some(kind) = self.selected_tower_kind.clone() {
              if let Some(tpl) = self.td_templates.get(&kind).cloned() {
                let footprint_backend = tpl.footprint_backend;
                let range_backend = tpl.range_backend;
                let cost = tpl.cost;
                let mwp = self.mouse_world_pos;
                // ===== 本地 placement 驗證（前端即時預覽；後端下最終決定）=====
                const PATH_HALF_WIDTH: f32 = 64.0; // 與後端 PATH_HALF_WIDTH 同步
                let footprint_render = footprint_backend * WORLD_SCALE;
                let clear_render = (footprint_backend + PATH_HALF_WIDTH) * WORLD_SCALE;
                let clear_sq = clear_render * clear_render;
                let mut can_place = self.hero_state.gold >= cost;
                if can_place {
                    // 壓到 path？
                    'outer: for path in &self.td_paths_render {
                        for i in 0..path.len().saturating_sub(1) {
                            if point_segment_dist_sq(mwp, path[i], path[i+1]) < clear_sq {
                                can_place = false;
                                break 'outer;
                            }
                        }
                    }
                }
                if can_place {
                    // 壓到 region？
                    for poly in &self.td_regions_render {
                        if circle_hits_polygon(mwp, footprint_render, poly) {
                            can_place = false;
                            break;
                        }
                    }
                }
                if can_place {
                    // 與其他塔重疊？（只看 TD 塔）
                    for ent in self.network_entities.values() {
                        if ent.entity_type != "tower" { continue }
                        if ent.tower_kind.is_none() { continue }
                        let min_d = ent.collision_radius_render + footprint_render;
                        if (ent.position - mwp).norm_squared() < min_d * min_d {
                            can_place = false;
                            break;
                        }
                    }
                }

                // 可蓋 → 綠；不可蓋 → 紅
                let (foot_color, range_color) = if can_place {
                    (
                        Color::from_rgba(80, 220, 120, 220),
                        Color::from_rgba(255, 255, 255, 160),
                    )
                } else {
                    (
                        Color::from_rgba(230, 50, 50, 240),
                        Color::from_rgba(230, 80, 80, 160),
                    )
                };
                // 內圈：footprint
                add_circle_lines(
                    scene,
                    mwp,
                    footprint_render,
                    24,
                    foot_color,
                    Z_REGION - 0.0002,
                );
                // 外圈：攻擊範圍
                add_circle_lines(
                    scene,
                    mwp,
                    range_backend * WORLD_SCALE,
                    48,
                    range_color,
                    Z_REGION - 0.0001,
                );
              } // end of `if let Some(tpl) = ...`
            }
        }

        // Bomb 爆炸特效：用 Fyrox SceneDrawingContext 提交 32 線段，整批 single draw call。
        // 不再 per-frame remove+create scene graph node（原作法在 1000-tower stress 約 4.6ms / frame）。
        // 座標慣例與 build_line_segment 一致：x 取負（見該函式 center 計算 `-(from.x + to.x) * 0.5`）。
        {
            use fyrox::scene::debug::Line;
            let dt_f = context.dt;
            const SEGS: usize = 32;
            let mut finished_idx: Vec<usize> = Vec::new();
            for (i, ex) in self.active_explosions.iter_mut().enumerate() {
                ex.elapsed += dt_f;
                if ex.elapsed >= ex.duration {
                    finished_idx.push(i);
                    continue;
                }
                let t = (ex.elapsed / ex.duration).clamp(0.0, 1.0);
                let cur_r = ex.max_radius * t;
                // alpha 隨時間衰減（起始不透明 → 結束透明）
                let alpha = (255.0 * (1.0 - t)) as u8;
                let color = Color::from_rgba(230, 70, 40, alpha.max(40));
                if cur_r > 0.02 {
                    let z = Z_REGION - 0.0004;
                    // 起點：θ=0 → (cx + r, cy)；x 翻負與 build_line_segment 對齊
                    let mut prev = Vector3::new(-(ex.pos.x + cur_r), ex.pos.y, z);
                    for k in 1..=SEGS {
                        let theta = (k as f32) * std::f32::consts::TAU / (SEGS as f32);
                        let (s, c) = theta.sin_cos();
                        let next = Vector3::new(
                            -(ex.pos.x + cur_r * c),
                            ex.pos.y + cur_r * s,
                            z,
                        );
                        scene.drawing_context.add_line(Line { begin: prev, end: next, color });
                        prev = next;
                    }
                }
            }
            // 反向刪除以保持 index 有效
            for i in finished_idx.into_iter().rev() {
                self.active_explosions.remove(i);
            }
        }

        // TD 已選中塔的射程圈：每 frame 以塔位置為中心重畫；若塔已消失則自動清選
        // 用 SceneDrawingContext（drawing_context 已在 update() 開頭被 clear_lines()），
        // 不再 per-frame 增刪 48 個 RectangleBuilder node。
        {
            if let Some(tid) = self.selected_tower_entity {
                match self.network_entities.get(&tid) {
                    Some(ent) if ent.entity_type == "tower" && ent.attack_range_backend > 0.0 => {
                        add_circle_lines(
                            scene,
                            ent.position,
                            ent.attack_range_backend * WORLD_SCALE,
                            48,
                            Color::from_rgba(255, 220, 40, 220),
                            Z_REGION - 0.0003,
                        );
                    }
                    _ => {
                        // entity 消失（被賣或打掉）→ 清選
                        self.selected_tower_entity = None;
                    }
                }
            }
        }

        let visual_ns = t_visual.elapsed().as_nanos();

        // 4b. Advance client-simulated projectiles (pursuit lerp toward target's
        //     current interpolated position; t forced to 1 at flight_time).
        //     後端改為 100ms batch 發送，client flight_time 與 backend projectile time 已對齊
        //     (game_processor.rs 裡用 initial_dist / bullet_speed 設 safety_time_left 的 1/3)，
        //     所以彈落時 optimistic 扣血與 100ms 內到達的 backend "H" 事件幾乎 sync，不會 bouncing。
        let t_proj = std::time::Instant::now();
        let mut finished: Vec<u32> = Vec::new();
        // P7 layered：t≥1.0 視覺命中時要 mark 對應 pending_pred_dmg 為 applied=true。
        // 收 id 後在 loop 結束後一起做（避免 self.client_projectiles 與 self.pending_pred_dmg
        // 同時 mut borrow 的 split-borrow 麻煩）。
        let mut predicted_apply_ids: Vec<u32> = Vec::new();
        for (id, proj) in self.client_projectiles.iter_mut() {
            proj.elapsed += dt;
            let t = (proj.elapsed / proj.flight_time).clamp(0.0, 1.0);
            // 方向性子彈走固定直線；追蹤子彈鎖 target 現位
            let target_pos = if proj.directional {
                proj.end_pos
            } else {
                self.network_entities
                    .get(&proj.target_id)
                    .map(|e| e.position)
                    .unwrap_or(proj.last_target_pos)
            };
            proj.last_target_pos = target_pos;
            let pos = proj.start_pos + (target_pos - proj.start_pos) * t;
            scene.graph[proj.node]
                .local_transform_mut()
                .set_position(Vector3::new(-pos.x, pos.y, Z_BULLET));
            // Tack 命中圈跟隨子彈
            for (h, offset) in &proj.hit_ring {
                scene.graph[*h]
                    .local_transform_mut()
                    .set_position(Vector3::new(-(pos.x + offset.x), pos.y + offset.y, Z_BULLET + 0.0001));
            }
            if t >= 1.0 {
                // 方向性子彈的 damage 由後端 H 事件授權，不做 optimistic 扣血
                if !proj.directional && !proj.applied && proj.damage > 0.0 {
                    predicted_apply_ids.push(*id);
                    proj.applied = true;
                }
                // Bomb 塔：命中時在「子彈當前視覺位置」自 spawn 爆炸特效。
                // 子彈視覺 = 追蹤 target 的實時位置，所以爆炸中心永遠落在氣球身上，
                // 不會因為 1-tick 誤差停在舊位置。
                if proj.splash_radius_render > 0.02 && !proj.applied {
                    // `applied` 同時當作「已觸發爆炸」的旗標
                }
                if proj.splash_radius_render > 0.02 {
                    // 當前子彈位置作為爆炸圓心
                    self.active_explosions.push(ActiveExplosion {
                        pos,
                        max_radius: proj.splash_radius_render,
                        duration: 0.35,
                        elapsed: 0.0,
                    });
                }
                finished.push(*id);
            }
        }
        // P7 layered：mark applied，display HP 渲染時才減（在下方 entity update loop
        // 計 pending_dmg_by_target）。不再直接寫 entity.health（authoritative 由 server H
        // / heartbeat hp_snapshot 獨佔）。
        for proj_id in predicted_apply_ids {
            if let Some(p) = self.pending_pred_dmg.get_mut(&proj_id) {
                p.applied = true;
            }
        }
        for id in finished {
            if let Some(proj) = self.client_projectiles.remove(&id) {
                scene.graph.remove_node(proj.node);
                for (h, _) in proj.hit_ring {
                    scene.graph.remove_node(h);
                }
            }
        }

        let proj_ns = t_proj.elapsed().as_nanos();

        // 4c. Camera follow hero（MOBA 模式）或 固定俯視（TD 模式）
        //     TD 模式下：相機固定在地圖中心、拉遠到能看完整條路線。
        let t_cam = std::time::Instant::now();
        if self.is_td_mode {
            if !self.td_camera_configured {
                // 一次性：放大視角、鎖定在原點
                if let Some(cam) = scene.graph[self.camera]
                    .cast_mut::<fyrox::scene::camera::Camera>()
                {
                    cam.set_projection(Projection::Orthographic(OrthographicProjection {
                        z_near: 0.1,
                        z_far: 1000.0,
                        vertical_size: 14.0, // 28 render 高 = 2800 backend，可裝下 ±1200 Y
                    }));
                }
                // Camera at z=-100 looking +Z (default) — preserve that on re-center.
                scene.graph[self.camera]
                    .local_transform_mut()
                    .set_position(Vector3::new(0.0, 0.0, -100.0));
                self.camera_world_pos = Vector2::new(0.0, 0.0);
                self.td_camera_configured = true;
                log::info!("🎥 TD 相機已鎖定：center=(0,0), vertical_size=14");

                // Phase 5.1: viewport push to legacy NetworkBridge removed.
                // Lockstep state is broadcast in full to all clients regardless
                // of viewport, so this hint is no longer needed.
            }
        } else {
            // MOBA 模式：相機不再跟隨英雄移動。保留 camera 在 scene.rgs 載入時的初始位置，
            // camera_world_pos 從 camera 當前 transform 反推（X 渲染負號 → world.x = -cam.x），
            // 確保 name label 螢幕投影仍正確。
            let cam_pos = scene.graph[self.camera].local_transform().position();
            self.camera_world_pos = Vector2::new(-cam_pos.x, cam_pos.y);
            // Phase 5.1: periodic viewport sync to NetworkBridge removed.
        }

        let cam_ns = t_cam.elapsed().as_nanos();

        // 5. Update name labels (UI layer)
        let t_ui = std::time::Instant::now();
        let ui = context.user_interfaces.first_mut();
        let win = self.window_size;

        // Delete labels for removed entities
        for label in self.pending_label_deletions.drain(..) {
            ui.send(label, WidgetMessage::Remove);
        }

        // Stress 場景下隱藏 name label：每個 entity 1 個 UI text widget = 1 個
        // UI draw call。1500+ creep 就是 1500+ 額外 draws，視覺上也是一團糊看不清。
        // entity 數超過 NAME_LABEL_HIDE_THRESHOLD 時暫停建立並把現有的清掉。
        // Alt 按住強制顯示（讓玩家可以在 stress 場景偶爾查 entity 名稱 / HP）。
        const NAME_LABEL_HIDE_THRESHOLD: usize = 200;
        let too_many_entities = self.network_entities.len() > NAME_LABEL_HIDE_THRESHOLD;
        let labels_hidden = too_many_entities && !self.alt_held;

        if labels_hidden {
            // Bulk-remove existing labels（一次清完，避免 frame-by-frame 慢慢清）
            for (_, entity) in self.network_entities.iter_mut() {
                if let Some(label) = entity.name_label.take() {
                    ui.send(label, WidgetMessage::Remove);
                }
            }
        }

        // Create missing labels & update positions
        for (&entity_id, entity) in self.network_entities.iter_mut() {
            if entity.health.is_none() {
                continue; // only show names for entities with HP bars
            }
            if labels_hidden {
                continue; // 太多 entity 且沒按 Alt，不渲染 name label，省 N 個 UI draw call
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
                // 重置 throttle cache：新 widget 的位置是 default (0, 0)，下面的
                // pos_changed 比對必須一定觸發（不然新 label 永遠停在螢幕左上角）。
                entity.last_label_pos = Vector2::new(f32::MIN, f32::MIN);
                entity.last_label_text = String::new();
            }

            // Update label screen position (above HP bar) + 文字含 HP 數字
            // Stress 場景節流：位置差距 < 1 px、文字未變時，整個 entity 跳過
            // 兩條 UI 訊息，避免 1000 entity × 每幀 send 把 Fyrox UI queue 灌爆。
            if let Some(label) = entity.name_label {
                let name_world_y = entity.position.y + 0.5;
                let world_height = if self.is_td_mode { 28.0 } else { 20.0 };
                let screen_pos = world_to_screen_approx(
                    entity.position.x - self.camera_world_pos.x,
                    name_world_y - self.camera_world_pos.y,
                    win.x, win.y, world_height,
                );
                let pos = Vector2::new(screen_pos.x - 90.0, screen_pos.y - 24.0);
                let pos_changed = (pos.x - entity.last_label_pos.x).abs() >= 1.0
                    || (pos.y - entity.last_label_pos.y).abs() >= 1.0;
                if pos_changed {
                    ui.send(label, WidgetMessage::DesiredPosition(pos));
                    entity.last_label_pos = pos;
                }

                // 顯示「名字 HP/MaxHP」讓 HP bouncing 肉眼可見
                // 用 round() 比對避免 0.1 HP 級小波動灌訊息
                // P7 layered：跟 HP bar 一致，扣掉 applied 但 server 還沒反映的預測扣血
                let text = match entity.health {
                    Some((h, m)) => {
                        let pending_dmg = pending_dmg_by_target.get(&entity_id).copied().unwrap_or(0.0);
                        let display_h = (h - pending_dmg).max(0.0);
                        format!("{} {:.0}/{:.0}", entity.name, display_h.round(), m.round())
                    }
                    None => entity.name.clone(),
                };
                if text != entity.last_label_text {
                    ui.send(label, TextMessage::Text(text.clone()));
                    entity.last_label_text = text;
                }
            }
        }

        // sim_runner-backed name labels: Phase 5.x replaces the legacy
        // network_entities-driven loop above. Reads the same snapshot the
        // render_bridge consumes; one Text widget per visible entity, kept
        // in sync via `sim_entity_labels`.
        if let Some(ref sim) = self.sim_runner_handle {
            if let Ok(snapshot) = sim.state.try_lock() {
                let mut alive = std::collections::HashSet::with_capacity(snapshot.entities.len());
                let world_height = if self.is_td_mode { 28.0 } else { 20.0 };
                for entity in &snapshot.entities {
                    if matches!(entity.kind, sim_runner::EntityKind::Other) {
                        continue;
                    }
                    alive.insert(entity.entity_id);

                    // Display name: prefer hero_name (heroes), else unit_id sans
                    // template prefix, else fallback to "#<id>".
                    let display_name = if !entity.hero_name.is_empty() {
                        entity.hero_name.clone()
                    } else if !entity.unit_id.is_empty() {
                        entity.unit_id
                            .strip_prefix("creep_")
                            .or_else(|| entity.unit_id.strip_prefix("tower_"))
                            .or_else(|| entity.unit_id.strip_prefix("hero_"))
                            .or_else(|| entity.unit_id.strip_prefix("unit_"))
                            .unwrap_or(&entity.unit_id)
                            .to_string()
                    } else {
                        format!("#{}", entity.entity_id)
                    };
                    let text = if entity.max_hp > 0 {
                        format!("{} {}/{}", display_name, entity.hp.max(0), entity.max_hp)
                    } else {
                        display_name
                    };

                    // World pos for label = entity center + slight Y offset so
                    // it sits above the sprite + HP bar (~0.6 world units up).
                    let label_world_y = entity.pos_y + 60.0;
                    let screen_pos = world_to_screen_approx(
                        entity.pos_x - self.camera_world_pos.x,
                        label_world_y - self.camera_world_pos.y,
                        win.x,
                        win.y,
                        world_height,
                    );
                    let pos = Vector2::new(screen_pos.x - 90.0, screen_pos.y - 24.0);

                    if let Some(slot) = self.sim_entity_labels.get_mut(&entity.entity_id) {
                        // Update existing — gate to avoid flooding the UI queue.
                        let pos_changed = (pos.x - slot.last_pos.x).abs() >= 1.0
                            || (pos.y - slot.last_pos.y).abs() >= 1.0;
                        if pos_changed {
                            ui.send(slot.handle, WidgetMessage::DesiredPosition(pos));
                            slot.last_pos = pos;
                        }
                        if text != slot.last_text {
                            ui.send(slot.handle, TextMessage::Text(text.clone()));
                            slot.last_text = text;
                        }
                    } else {
                        // First-time spawn for this entity.
                        let handle = TextBuilder::new(
                            WidgetBuilder::new()
                                .with_desired_position(pos)
                                .with_width(180.0)
                                .with_foreground(Brush::Solid(Color::from_rgba(0, 0, 0, 255)).into()),
                        )
                        .with_text(text.clone())
                        .with_font_size(18.0.into())
                        .with_horizontal_text_alignment(HorizontalAlignment::Center)
                        .build(&mut ui.build_ctx());
                        self.sim_entity_labels.insert(entity.entity_id, SimEntityLabel {
                            handle,
                            last_text: text,
                            last_pos: pos,
                        });
                    }
                }

                // Despawn labels for entities no longer in snapshot.
                // Phase 1.6: snapshot now carries explicit `removed_entity_ids`
                // (diff computed worker-locally in sim_runner), replacing the
                // legacy omb `entity.death` GameEvent. We still keep the
                // `alive`-set sweep below as a belt-and-suspenders defense
                // against any cache rows whose eid never appeared in
                // `removed_entity_ids` (e.g. labels created before the very
                // first prev_alive snapshot was populated).
                for &eid in &snapshot.removed_entity_ids {
                    if let Some(slot) = self.sim_entity_labels.remove(&eid) {
                        ui.send(slot.handle, WidgetMessage::Remove);
                    }
                }
                let to_remove: Vec<u32> = self
                    .sim_entity_labels
                    .keys()
                    .filter(|id| !alive.contains(id))
                    .copied()
                    .collect();
                for id in to_remove {
                    if let Some(slot) = self.sim_entity_labels.remove(&id) {
                        ui.send(slot.handle, WidgetMessage::Remove);
                    }
                }
            }
        }

        // 6. Update status text
        let connection_part = match &self.connection_status {
            ConnectionStatus::Disconnected => "Disconnected".to_string(),
            ConnectionStatus::Connecting => "Connecting...".to_string(),
            ConnectionStatus::Connected => {
                let fmt_bps = |bps: u64| -> String {
                    if bps >= 1_000_000 {
                        format!("{:.2} MB/s", bps as f64 / 1_000_000.0)
                    } else if bps >= 1_000 {
                        format!("{:.1} KB/s", bps as f64 / 1_000.0)
                    } else {
                        format!("{} B/s", bps)
                    }
                };
                let wire_str = fmt_bps(self.net_wire_bytes_last_sec);   // 真實 UDP wire (壓縮後)
                let logical_str = fmt_bps(self.net_bytes_last_sec);      // 解壓後 logical
                format!(
                    "Connected | Tick: {} | Time: {:.1} | Entities: {} | Heroes: {} | Creeps: {} | Net: {} wire / {} logical",
                    self.heartbeat.tick,
                    self.heartbeat.game_time,
                    self.heartbeat.entity_count,
                    self.heartbeat.hero_count,
                    self.heartbeat.creep_count,
                    wire_str,
                    logical_str,
                )
            }
            ConnectionStatus::Failed(e) => format!("Failed: {}", e),
        };
        // Render stats from prior frame (record_render_stats() runs at end of update,
        // so values here are 1 frame behind — fine for a live readout).
        let render_stats_part = format!(
            "fps: {} | draws: {} | tris: {}",
            self.frame_profile.last_fps,
            self.frame_profile.last_draw_calls,
            self.frame_profile.last_triangles,
        );
        let status_str = if self.fps_display.is_empty() {
            format!("{} | {}", render_stats_part, connection_part)
        } else {
            format!("{} | {} | {}", self.fps_display, render_stats_part, connection_part)
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

            // ===== TD 模式右側塔按鈕（動態：數量 = td_template_order.len()） =====
            // 數量與順序由後端 tower_templates 事件決定（DLL units() 註冊順序）
            {
                let btn_w = 240.0f32;
                let btn_h = 36.0f32;
                let btn_spacing = 44.0f32;
                let right_margin = 20.0f32;
                let x = self.window_size.x - btn_w - right_margin;
                let base_y = 80.0f32;

                let n = self.td_template_order.len();
                // 不夠就補 TextBuilder
                while self.ui_td_tower_buttons.len() < n {
                    let h = TextBuilder::new(
                        WidgetBuilder::new()
                            .with_desired_position(Vector2::new(-9999.0, -9999.0))
                            .with_width(btn_w)
                            .with_foreground(Brush::Solid(Color::from_rgba(0, 0, 0, 255)).into()),
                    )
                    .with_text(String::new())
                    .with_font_size(18.0.into())
                    .build(&mut ui.build_ctx());
                    self.ui_td_tower_buttons.push(h);
                    self.td_tower_button_rects.push((-9999.0, -9999.0, btn_w, btn_h));
                }
                // 多了就藏起來（不 remove；避免頻繁 create/destroy）
                for i in n..self.ui_td_tower_buttons.len() {
                    ui.send(self.ui_td_tower_buttons[i],
                        WidgetMessage::DesiredPosition(Vector2::new(-9999.0, -9999.0)));
                    self.td_tower_button_rects[i] = (-9999.0, -9999.0, 0.0, 0.0);
                }

                let selected = self.selected_tower_kind.as_deref();
                for i in 0..n {
                    let y = base_y + (i as f32) * btn_spacing;
                    self.td_tower_button_rects[i] = (x, y, btn_w, btn_h);
                    let uid = &self.td_template_order[i];
                    ui.send(
                        self.ui_td_tower_buttons[i],
                        WidgetMessage::DesiredPosition(Vector2::new(x, y)),
                    );
                    let prefix = if selected == Some(uid.as_str()) { "▶ " } else { "  " };
                    let label_cost = match self.td_templates.get(uid) {
                        Some(tpl) => format!("{}  ${}", tpl.label, tpl.cost),
                        None      => uid.clone(),
                    };
                    let text = format!("{}[{}] {}", prefix, i + 1, label_cost);
                    ui.send(self.ui_td_tower_buttons[i], TextMessage::Text(text));
                }
            }

            // ===== TD 模式：選中塔 Sell 面板（右側，4 塔按鈕下方） =====
            {
                let panel_w = 240.0f32;
                let name_h = 28.0f32;
                let btn_h = 42.0f32;
                let right_margin = 20.0f32;
                let x = self.window_size.x - panel_w - right_margin;
                // 定位在 N 塔按鈕下方再留一個 gap（N 動態 = td_template_order.len()）
                let n_btn = self.td_template_order.len().max(1) as f32;
                let y_name = 80.0f32 + n_btn * 44.0 + 20.0;
                let y_btn = y_name + name_h + 4.0;

                // Sell 面板從 td_templates 快取讀 label + cost（單一事實來源）
                // 同時讀 base_cost + upgrade_levels 供升級按鈕顯示
                let info: Option<(String, i32, i32, [u8; 3])> = self.selected_tower_entity.and_then(|tid| {
                    let ent = self.network_entities.get(&tid)?;
                    let kind_key = ent.tower_kind.as_deref()?;
                    let tpl = self.td_templates.get(kind_key)?;
                    let refund = (tpl.cost as f32 * 0.85) as i32;
                    Some((tpl.label.clone(), refund, tpl.cost, ent.upgrade_levels))
                });

                if let Some((label, refund, base_cost, levels)) = info {
                    ui.send(self.ui_td_sell_name_text,
                        WidgetMessage::DesiredPosition(Vector2::new(x, y_name)));
                    ui.send(self.ui_td_sell_name_text,
                        TextMessage::Text(format!("▸ {}", label)));

                    ui.send(self.ui_td_sell_button_text,
                        WidgetMessage::DesiredPosition(Vector2::new(x, y_btn)));
                    ui.send(self.ui_td_sell_button_text,
                        TextMessage::Text(format!("[SELL] ${}", refund)));
                    self.td_sell_button_rect = (x, y_btn, panel_w, btn_h);

                    // 3 條升級按鈕，往下排（每行 btn_h + 4）
                    let up_btn_h = 38.0f32;
                    for path in 0u8..3 {
                        let level = levels[path as usize];
                        let y_up = y_btn + btn_h + 6.0 + (path as f32) * (up_btn_h + 4.0);
                        let filled = level.min(4) as usize;
                        let empty = 4 - filled;
                        let dots: String = "■".repeat(filled) + &"□".repeat(empty);
                        let text = if level >= 4 {
                            format!("{}  [P{}] MAX", dots, path + 1)
                        } else {
                            let next_cost = omoba_core::tower_meta::upgrade_cost(base_cost, level + 1);
                            format!("{}  [P{}] L{}->L{} ${}",
                                dots, path + 1, level, level + 1, next_cost)
                        };
                        ui.send(self.ui_td_upgrade_buttons[path as usize],
                            WidgetMessage::DesiredPosition(Vector2::new(x, y_up)));
                        ui.send(self.ui_td_upgrade_buttons[path as usize],
                            TextMessage::Text(text));
                        self.td_upgrade_button_rects[path as usize] = (x, y_up, panel_w, up_btn_h);
                    }
                } else {
                    // 未選塔：藏在螢幕外
                    ui.send(self.ui_td_sell_name_text,
                        WidgetMessage::DesiredPosition(Vector2::new(-9999.0, -9999.0)));
                    ui.send(self.ui_td_sell_button_text,
                        WidgetMessage::DesiredPosition(Vector2::new(-9999.0, -9999.0)));
                    self.td_sell_button_rect = (-9999.0, -9999.0, 0.0, 0.0);
                    for path in 0..3 {
                        ui.send(self.ui_td_upgrade_buttons[path],
                            WidgetMessage::DesiredPosition(Vector2::new(-9999.0, -9999.0)));
                        self.td_upgrade_button_rects[path] = (-9999.0, -9999.0, 0.0, 0.0);
                    }
                }
            }

            // ===== TD 模式 Start Round 按鈕（右下角） =====
            {
                let btn_w = 260.0f32;
                let btn_h = 48.0f32;
                let right_margin = 20.0f32;
                let bottom_margin = 140.0f32; // 避開技能列
                let x = self.window_size.x - btn_w - right_margin;
                let y = self.window_size.y - btn_h - bottom_margin;
                self.start_round_button_rect = (x, y, btn_w, btn_h);
                if self.ui_start_round_button != Handle::<Text>::NONE {
                    ui.send(
                        self.ui_start_round_button,
                        WidgetMessage::DesiredPosition(Vector2::new(x, y)),
                    );
                    let text = if self.total_rounds > 0 && self.current_round >= self.total_rounds && !self.round_is_running {
                        "✓ ALL ROUNDS CLEAR".to_string()
                    } else if self.round_is_running {
                        format!("⏸ Round {} Running...", self.current_round.max(1))
                    } else {
                        let next = self.current_round + 1;
                        let total = self.total_rounds.max(1);
                        format!("▶ START ROUND {} / {}", next, total)
                    };
                    ui.send(self.ui_start_round_button, TextMessage::Text(text));
                }
            }

            // 技能冷卻每 frame 遞減
            for cd in self.hero_state.ability_cd.values_mut() {
                if *cd > 0.0 { *cd = (*cd - dt).max(0.0); }
            }

            // Buff 倒數：本地每 frame 遞減，讓面板顯示連續變化；
            // 下次 backend push 的 snapshot 會重設成權威值，避免漂移。
            for b in self.hero_state.buffs.iter_mut() {
                if b.remaining > 0.0 { b.remaining = (b.remaining - dt).max(0.0); }
            }
            // remaining = 0 的有限期 buff 從本地清掉（權威值會在下次 push 糾正）
            self.hero_state.buffs.retain(|b| b.remaining != 0.0);

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
            let hud = if hs.lives > 0 {
                format!(
                    "LIVES {}   GOLD {}   |   HP {:.0}/{:.0}  LV {}  XP {}/{}  SP {}",
                    hs.lives, hs.gold, hs.hp, hs.max_hp, hs.level, hs.xp, hs.xp_next, hs.skill_points,
                )
            } else {
                format!(
                    "HP {:.0}/{:.0}  LV {}  XP {}/{}  GOLD {}  SP {}  |  {}",
                    hs.hp, hs.max_hp, hs.level, hs.xp, hs.xp_next, hs.gold, hs.skill_points, inv,
                )
            };
            ui.send(self.ui_hud_text, TextMessage::Text(hud));

            // 左下角英雄屬性面板：每 tick 重組文字並依 window_size 重定位
            {
                let hs = &self.hero_state;
                let header = if hs.name.is_empty() {
                    "(尚未載入英雄)".to_string()
                } else if hs.title.is_empty() {
                    hs.name.clone()
                } else {
                    format!("{} · {}", hs.name, hs.title)
                };
                // 主屬性標記：與主屬性相同的三圍後面加 ★
                let tag = |attr: &str| if hs.primary_attribute == attr { "★" } else { " " };
                let mut ability_lines = String::new();
                for (i, id) in hs.abilities.iter().enumerate().take(4) {
                    let lvl = hs.ability_levels.get(id).copied().unwrap_or(0);
                    let key = ["Q", "W", "E", "R"].get(i).copied().unwrap_or("?");
                    ability_lines.push_str(&format!("\n[{}] {:<22} 等級 {}/4", key, id, lvl));
                }
                let aps = if hs.attack_interval > 0.0 { 1.0 / hs.attack_interval } else { 0.0 };
                // 組 buff 區塊：每行 "[id] 剩餘 X.Xs" 或 "[id] 持續 ∞"
                let mut buff_lines = String::new();
                if hs.buffs.is_empty() {
                    buff_lines.push_str("\n  （無）");
                } else {
                    for b in &hs.buffs {
                        let dur = if b.remaining < 0.0 {
                            "∞".to_string()
                        } else {
                            format!("{:.1}秒", b.remaining)
                        };
                        buff_lines.push_str(&format!("\n  {:<20} 剩餘 {}", b.id, dur));
                        // 列 payload 的數值欄位（range_bonus/damage_bonus/...）
                        if let Some(obj) = b.payload.as_object() {
                            for (k, v) in obj {
                                if let Some(f) = v.as_f64() {
                                    buff_lines.push_str(&format!("\n    {:<22} {:>+6.2}", k, f));
                                }
                            }
                        }
                    }
                }
                // 所有欄位採用 2 字中文標籤 + 右對齊數值，保持垂直對齊
                let panel_text = format!(
                    "{}\n\
                     等級 {:>3}     經驗 {:>4}/{:<4}   技點 {}\n\
                     力量 {:>3}{}   敏捷 {:>3}{}   智力 {:>3}{}\n\
                     血量 {:>4}/{:<4}   金錢 {}\n\
                     護甲 {:>4.1}   魔抗 {:>4.1}   移速 {:>4.0}\n\
                     攻擊 {:>4.0}   攻速 {:>4.2}秒   射程 {:>4.0}\n\
                     彈速 {:>4.0}   每秒 {:>4.2}\n\
                     ── 技能 ──{}\n\
                     ── 效果 ──{}",
                    header,
                    hs.level, hs.xp, hs.xp_next, hs.skill_points,
                    hs.strength, tag("strength"),
                    hs.agility, tag("agility"),
                    hs.intelligence, tag("intelligence"),
                    hs.hp as i32, hs.max_hp as i32, hs.gold,
                    hs.armor, hs.magic_resist, hs.move_speed,
                    hs.attack_damage, hs.attack_interval, hs.attack_range,
                    hs.bullet_speed, aps,
                    ability_lines,
                    buff_lines,
                );
                // 英雄屬性面板移到左上（status bar y=2 + HUD y=24 之下，y=50 起）
                let panel_y = 50.0;
                ui.send(
                    self.ui_hero_stats_panel,
                    WidgetMessage::DesiredPosition(Vector2::new(10.0, panel_y)),
                );
                ui.send(self.ui_hero_stats_panel, TextMessage::Text(panel_text));
            }

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
        let ui_ns = t_ui.elapsed().as_nanos();

        let total_ns = frame_t0.elapsed().as_nanos();
        self.frame_profile.events_ns += events_ns;
        self.frame_profile.interp_ns += interp_ns;
        self.frame_profile.visual_ns += visual_ns;
        self.frame_profile.proj_ns += proj_ns;
        self.frame_profile.cam_ns += cam_ns;
        self.frame_profile.ui_ns += ui_ns;
        self.frame_profile.total_ns += total_ns;
        self.frame_profile.events_drained += events_drained_local;
        self.frame_profile.creeps_seen += self.network_entities.len() as u64;
        self.frame_profile.projectiles_seen += self.client_projectiles.len() as u64;
        // Fyrox renderer stats (real frame time including render submit + GPU + vsync wait)
        if let fyrox::engine::GraphicsContext::Initialized(ref gc) = context.graphics_context {
            self.frame_profile.record_render_stats(&gc.renderer.get_statistics());
        }
        self.frame_profile.finish_frame();

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
                // 3D 相機 → 滑鼠 picking ray vs z=0 平面交點
                // Fyrox `Camera::make_ray(cursor, window_size)` 內部已處理 Y 反轉
                // 與 NDC ↔ world 轉換；無需手算 vertical_size / aspect。
                let cursor = Vector2::new(position.x as f32, position.y as f32);
                let scene = &context.scenes[self.scene];
                if let Some(camera) = scene
                    .graph
                    .try_get(self.camera)
                    .ok()
                    .and_then(|n| n.cast::<fyrox::scene::camera::Camera>())
                {
                    let ray = camera.make_ray(cursor, self.window_size);
                    if ray.dir.z.abs() > 1e-6 {
                        // 相機 z=-100 朝 +Z；z=0 平面交點 t = -origin.z/dir.z > 0
                        let t = -ray.origin.z / ray.dir.z;
                        let render_x = ray.origin.x + t * ray.dir.x;
                        let render_y = ray.origin.y + t * ray.dir.y;
                        // render world +X 為螢幕右；entity.position（logical）的 +X 對應
                        // render -X（見 set_position(-pos.x, ...) 慣例），故 logical = -render
                        self.mouse_world_pos = Vector2::new(-render_x, render_y);
                    }
                }
                // 原始 pixel 座標，供 tooltip hit-test 用
                self.mouse_screen_pos = cursor;
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
                // TD 模式左鍵：依序 Start Round → 4 塔按鈕 → Sell 按鈕 → 放置塔 → 點選已蓋塔
                let screen = self.mouse_screen_pos;
                let mut hit_ui = false;

                // 1. Start Round 按鈕 — Phase 5.x lockstep send
                {
                    let (bx, by, bw, bh) = self.start_round_button_rect;
                    if screen.x >= bx && screen.x <= bx + bw
                        && screen.y >= by && screen.y <= by + bh
                        && !self.round_is_running
                        && !(self.total_rounds > 0 && self.current_round >= self.total_rounds)
                    {
                        let input = omoba_core::kcp::game_proto::PlayerInput {
                            action: Some(
                                omoba_core::kcp::game_proto::player_input::Action::StartRound(
                                    omoba_core::kcp::game_proto::StartRound {},
                                ),
                            ),
                        };
                        self.send_lockstep_input(input);
                        log::info!("Start Round → lockstep PlayerInput::StartRound sent");
                        hit_ui = true;
                    }
                }

                // 2. 4 塔按鈕
                if !hit_ui {
                    // 依動態 template_order 對應按鈕
                    let mut hit_idx: Option<usize> = None;
                    for (i, rect) in self.td_tower_button_rects.iter().enumerate() {
                        let (bx, by, bw, bh) = *rect;
                        if i >= self.td_template_order.len() { break }
                        if screen.x >= bx && screen.x <= bx + bw
                            && screen.y >= by && screen.y <= by + bh
                        {
                            hit_idx = Some(i);
                            break;
                        }
                    }
                    if let Some(i) = hit_idx {
                        let uid = self.td_template_order[i].clone();
                        self.selected_tower_kind = Some(uid.clone());
                        self.selected_tower_entity = None;
                        log::info!("選中塔: {}", uid);
                        hit_ui = true;
                    }
                }

                // 3. Sell 按鈕（只有有已選中塔時生效）
                if !hit_ui && self.selected_tower_entity.is_some() {
                    let (bx, by, bw, bh) = self.td_sell_button_rect;
                    if screen.x >= bx && screen.x <= bx + bw
                        && screen.y >= by && screen.y <= by + bh
                    {
                        if let Some(tid) = self.selected_tower_entity {
                            // Phase 2.2: TowerSell lockstep input. tid is the
                            // tower entity id (specs `Entity::id()` u32);
                            // omb's drain handler resolves Entity, validates
                            // Player faction, refunds 85% base + 75% upgrades,
                            // and deletes the entity (snapshot diff cleans
                            // render). selected_tower_entity is cleared
                            // unconditionally because the entity is going away.
                            let input = omoba_core::kcp::game_proto::PlayerInput {
                                action: Some(
                                    omoba_core::kcp::game_proto::player_input::Action::TowerSell(
                                        omoba_core::kcp::game_proto::TowerSell {
                                            tower_entity_id: tid,
                                        },
                                    ),
                                ),
                            };
                            self.send_lockstep_input(input);
                            log::info!("Tower sell lockstep input submitted: eid={}", tid);
                            self.selected_tower_entity = None;
                        }
                        hit_ui = true;
                    }
                }

                // 3b. 3 條升級按鈕（必須在 tower-deselect 邏輯之前跑）
                if !hit_ui && self.selected_tower_entity.is_some() {
                    for path in 0u8..3 {
                        let (bx, by, bw, bh) = self.td_upgrade_button_rects[path as usize];
                        if bx > -9000.0
                            && screen.x >= bx && screen.x < bx + bw
                            && screen.y >= by && screen.y < by + bh
                        {
                            if let Some(tid) = self.selected_tower_entity {
                                // Phase 5.1: legacy NetCommand::UpgradeTower send removed.
                                // TODO Phase 5.x: route UpgradeTower through lockstep PlayerInput.
                                log::info!("[phase5.1] Upgrade Tower id={} path={} (legacy send removed)", tid, path);
                            }
                            hit_ui = true;
                            break;
                        }
                    }
                }

                // 4. 放置塔（如在選塔模式）。放完後若沒按 Ctrl 則自動取消
                if !hit_ui {
                    if let Some(kind) = self.selected_tower_kind.clone() {
                        let world_pos = self.mouse_world_pos;
                        // Phase 2.1: TowerPlace lockstep input. selected_tower_kind
                        // is the unit_id string (e.g. "tower_dart") — convert to
                        // proto u32 kind_id via omoba_template_ids::tower_by_name.
                        match omoba_template_ids::tower_by_name(&kind) {
                            Some(tid) => {
                                let pos = world_render_to_vec2i(world_pos);
                                let input = omoba_core::kcp::game_proto::PlayerInput {
                                    action: Some(
                                        omoba_core::kcp::game_proto::player_input::Action::TowerPlace(
                                            omoba_core::kcp::game_proto::TowerPlace {
                                                tower_kind_id: tid.0 as u32,
                                                pos: Some(pos),
                                            },
                                        ),
                                    ),
                                };
                                self.send_lockstep_input(input);
                                log::info!(
                                    "Tower place lockstep input submitted: kind='{}' kind_id={} pos=({}, {})",
                                    kind, tid.0, pos.x, pos.y
                                );
                            }
                            None => {
                                log::warn!(
                                    "Tower place: unknown kind name '{}' (no template_ids match) — skipped",
                                    kind
                                );
                            }
                        }
                        if !self.ctrl_held {
                            self.selected_tower_kind = None;
                        }
                        hit_ui = true;
                    }
                }

                // 5. 點選已蓋塔（只有非選塔模式時生效）
                if !hit_ui && self.selected_tower_kind.is_none() {
                    let mwp = self.mouse_world_pos;
                    let mut best: Option<(u32, f32)> = None;
                    for (id, ent) in self.network_entities.iter() {
                        if ent.entity_type != "tower" { continue }
                        if ent.tower_kind.is_none() { continue } // 只選 TD 塔（非 MOBA lane/base）
                        let d = (ent.position - mwp).norm();
                        let pick_radius = (ent.collision_radius_render * 1.6).max(0.6);
                        if d <= pick_radius {
                            if best.map(|(_, bd)| d < bd).unwrap_or(true) {
                                best = Some((*id, d));
                            }
                        }
                    }
                    if let Some((id, _)) = best {
                        self.selected_tower_entity = Some(id);
                        log::info!("點選中塔 id={}", id);
                    } else {
                        // 點空地 → 清掉選取
                        if self.selected_tower_entity.is_some() {
                            self.selected_tower_entity = None;
                        }
                    }
                }
            }
            // Right click：TD 模式優先用來取消選塔；若無任何選取才送 HeroMove
            Event::WindowEvent {
                event:
                    WindowEvent::MouseInput {
                        button: MouseButton::Right,
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                if self.selected_tower_kind.is_some() {
                    self.selected_tower_kind = None;
                    log::info!("RMB 取消放塔預覽");
                } else if self.selected_tower_entity.is_some() {
                    self.selected_tower_entity = None;
                    log::info!("RMB 取消選中塔");
                } else {
                    // Phase 5.1: legacy NetCommand::HeroMove removed; lockstep
                    // PlayerInput::MoveTo (below) is the sole authoritative path.
                    let world_pos = self.mouse_world_pos;
                    let target = world_render_to_vec2i(world_pos);
                    let move_to = omoba_core::kcp::game_proto::MoveTo {
                        target: Some(target),
                    };
                    let input = omoba_core::kcp::game_proto::PlayerInput {
                        action: Some(
                            omoba_core::kcp::game_proto::player_input::Action::MoveTo(move_to),
                        ),
                    };
                    self.send_lockstep_input(input);
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

                // Shift / Ctrl 狀態追蹤
                match key {
                    KeyCode::ShiftLeft | KeyCode::ShiftRight => {
                        self.shift_held = pressed;
                        return Ok(());
                    }
                    KeyCode::ControlLeft | KeyCode::ControlRight => {
                        self.ctrl_held = pressed;
                        return Ok(());
                    }
                    KeyCode::AltLeft | KeyCode::AltRight => {
                        self.alt_held = pressed;
                        return Ok(());
                    }
                    _ => {}
                }
                if !pressed { return Ok(()); }

                let world = self.mouse_world_pos;
                // Phase 5.1: legacy `tx` / `send` closure (NetworkBridge cmd_tx)
                // removed. UpgradeSkill / BuyItem / SellItem / UseItem are now
                // logged-only stubs pending a Phase 5.x lockstep PlayerInput
                // extension; W/E/R/T cast already routes through lockstep below.
                let send_stub = |label: &str, args: &str| {
                    log::info!("[phase5.1] legacy {} send removed (args={})", label, args);
                };

                // Q/W/E/R press → lockstep PlayerInput::CastAbility (ability_index
                // 0/1/2/3). Mouse world pos becomes the optional `target_pos`.
                // Modifier-held cases (Shift = upgrade, not cast) are excluded.
                if !self.shift_held {
                    let ability_index_opt = match key {
                        KeyCode::KeyQ => Some(0u32),
                        KeyCode::KeyW => Some(1u32),
                        KeyCode::KeyE => Some(2u32),
                        KeyCode::KeyR => Some(3u32),
                        _ => None,
                    };
                    if let Some(ability_index) = ability_index_opt {
                        let target = world_render_to_vec2i(world);
                        let cast = omoba_core::kcp::game_proto::CastAbility {
                            ability_index,
                            target_pos: Some(target),
                            target_entity: None,
                        };
                        let input = omoba_core::kcp::game_proto::PlayerInput {
                            action: Some(
                                omoba_core::kcp::game_proto::player_input::Action::CastAbility(
                                    cast,
                                ),
                            ),
                        };
                        self.send_lockstep_input(input);
                    }
                }

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
                            // Phase 5.1: legacy UpgradeSkill removed; lockstep
                            // doesn't yet wire ability upgrade — pending Phase 5.x.
                            send_stub("UpgradeSkill", &slot);
                        } else {
                            // Cast already sent via lockstep above; nothing to do here.
                            // (Optimistic local cooldown bookkeeping was driven by
                            // legacy hero_state cache that is going away with apply_event.)
                            let _ = (slot, world);
                        }
                    }
                    KeyCode::KeyB => {
                        self.shop_visible = !self.shop_visible;
                    }
                    // TD 模式：1-9 鍵盤快捷選塔（依 td_template_order 順序）；Escape 取消選取
                    KeyCode::Digit1 | KeyCode::Digit2 | KeyCode::Digit3 | KeyCode::Digit4
                    | KeyCode::Digit5 | KeyCode::Digit6 | KeyCode::Digit7 | KeyCode::Digit8
                    | KeyCode::Digit9
                        if !self.shop_visible =>
                    {
                        let idx = match key {
                            KeyCode::Digit1 => 0,
                            KeyCode::Digit2 => 1,
                            KeyCode::Digit3 => 2,
                            KeyCode::Digit4 => 3,
                            KeyCode::Digit5 => 4,
                            KeyCode::Digit6 => 5,
                            KeyCode::Digit7 => 6,
                            KeyCode::Digit8 => 7,
                            KeyCode::Digit9 => 8,
                            _ => unreachable!(),
                        };
                        if let Some(uid) = self.td_template_order.get(idx).cloned() {
                            self.selected_tower_kind = Some(uid.clone());
                            log::info!("快捷選中塔: {}", uid);
                        }
                    }
                    KeyCode::Escape => {
                        if self.selected_tower_kind.is_some() {
                            self.selected_tower_kind = None;
                            log::info!("取消選塔");
                        }
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
                        // Phase 5.1: legacy BuyItem / SellItem / UseItem removed
                        // (item shop not yet on lockstep wire).
                        if self.shop_visible {
                            if let Some((id, _, _)) = SHOP_ITEMS.get(idx) {
                                send_stub("BuyItem", id);
                            }
                        } else if idx >= 1 && idx <= 6 {
                            if self.shift_held {
                                send_stub("SellItem", &(idx - 1).to_string());
                            } else {
                                send_stub("UseItem", &(idx - 1).to_string());
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
    /// Phase 4.3: send a `PlayerInput` to the lockstep wire (omb's lockstep
    /// scheduler). No-op if `lockstep_handle` is None (e.g. legacy-only mode).
    /// Target tick = current_sim_tick + 3 (50 ms input delay at 60 Hz). The
    /// underlying `GameClient` (in `omoba_core::kcp::client`) tags the
    /// `InputSubmit` frame with the cached `player_id` from `GameStart`, so
    /// callers don't need to know self-id.
    ///
    /// Runs in **parallel** with the legacy `NetworkBridge::cmd_tx` path —
    /// Phase 4.5 cuts the legacy side. Until then a single click may produce
    /// two server-side messages; omb-side is responsible for de-duping or
    /// (post-Phase-4.5) ignoring the legacy command.
    /// Phase 5.x: each tick, mirror sim_runner snapshot entities into the
    /// shared body_batch + hp_batch CPU mirrors. Allocates a slot per entity
    /// on first sighting; frees slots on dropout. EntityKind::Other is
    /// skipped (internal ECS rows like RegionBlocker should not render).
    fn update_sim_batches(&mut self) {
        let Some(ref sim) = self.sim_runner_handle else { return };
        let Ok(snapshot) = sim.state.try_lock() else { return };

        let mut alive = std::collections::HashSet::with_capacity(snapshot.entities.len());
        for e in &snapshot.entities {
            if matches!(e.kind, sim_runner::EntityKind::Other) {
                continue;
            }
            alive.insert(e.entity_id);

            let pos = render_bridge::world_to_render(e);
            let (color, size, z) = render_bridge::style_for_entity(e);

            // Body slot: alloc on first sighting, then write_quad each tick.
            let slots_entry = self.sim_entity_slots.entry(e.entity_id);
            let slots = slots_entry.or_insert_with(|| {
                let body_slot = self
                    .body_batch
                    .as_mut()
                    .map(|b| b.alloc())
                    .unwrap_or(0);
                render_bridge::SimEntitySlots {
                    body_slot,
                    hp_bg_slot: None,
                    hp_fg_slot: None,
                }
            });

            if let Some(batch) = self.body_batch.as_mut() {
                batch.write_quad(
                    slots.body_slot,
                    &sprite_resources::QuadParams {
                        center: pos,
                        size: Vector2::new(size, size),
                        color,
                        rotation: 0.0,
                        z,
                    },
                );
            }

            // HP bar (bg + fg). Alloc lazily — projectiles + entities w/o hp
            // skip allocating to keep capacity for actual units.
            if e.max_hp > 0 {
                if slots.hp_bg_slot.is_none() {
                    if let Some(batch) = self.hp_batch.as_mut() {
                        slots.hp_bg_slot = Some(batch.alloc());
                        slots.hp_fg_slot = Some(batch.alloc());
                    }
                }
                if let (Some(bg), Some(fg)) = (slots.hp_bg_slot, slots.hp_fg_slot) {
                    let bar_w = (size * 1.6).max(0.4);
                    let bar_h = 0.06_f32;
                    let bar_y = pos.y + size * 0.55;
                    let hp_ratio = (e.hp as f32 / e.max_hp as f32).clamp(0.0, 1.0);
                    let bar_color: [u8; 4] = if hp_ratio < 0.30 {
                        [220, 50, 50, 255]
                    } else if hp_ratio < 0.60 {
                        [220, 200, 60, 255]
                    } else {
                        [40, 220, 60, 255]
                    };
                    if let Some(batch) = self.hp_batch.as_mut() {
                        batch.write_quad(
                            bg,
                            &sprite_resources::QuadParams {
                                center: Vector2::new(pos.x, bar_y),
                                size: Vector2::new(bar_w, bar_h),
                                color: [0, 0, 0, 220],
                                rotation: 0.0,
                                z: Z_HP_BAR + 0.01,
                            },
                        );
                        let fg_w = bar_w * hp_ratio;
                        let fg_offset = (bar_w - fg_w) * 0.5;
                        batch.write_quad(
                            fg,
                            &sprite_resources::QuadParams {
                                center: Vector2::new(pos.x - fg_offset, bar_y),
                                size: Vector2::new(fg_w.max(0.001), bar_h * 0.8),
                                color: bar_color,
                                rotation: 0.0,
                                z: Z_HP_BAR,
                            },
                        );
                    }
                }
            }
        }

        // Free slots for entities that disappeared from the snapshot.
        // Phase 1.6: prefer the explicit `removed_entity_ids` diff produced
        // worker-side in sim_runner over the `alive`-set sweep — it's a
        // tighter signal (only those who died this tick) and replaces the
        // legacy wire-side `entity.death` event. The sweep below stays as
        // defense for early-frame eids that pre-date the first prev_alive set.
        for &eid in &snapshot.removed_entity_ids {
            if let Some(slots) = self.sim_entity_slots.remove(&eid) {
                if let Some(batch) = self.body_batch.as_mut() {
                    batch.free(slots.body_slot);
                }
                if let Some(batch) = self.hp_batch.as_mut() {
                    if let Some(bg) = slots.hp_bg_slot { batch.free(bg); }
                    if let Some(fg) = slots.hp_fg_slot { batch.free(fg); }
                }
            }
        }
        let to_remove: Vec<u32> = self
            .sim_entity_slots
            .keys()
            .filter(|id| !alive.contains(id))
            .copied()
            .collect();
        for id in to_remove {
            if let Some(slots) = self.sim_entity_slots.remove(&id) {
                if let Some(batch) = self.body_batch.as_mut() {
                    batch.free(slots.body_slot);
                }
                if let Some(batch) = self.hp_batch.as_mut() {
                    if let Some(bg) = slots.hp_bg_slot { batch.free(bg); }
                    if let Some(fg) = slots.hp_fg_slot { batch.free(fg); }
                }
            }
        }
    }

    fn send_lockstep_input(&self, input: omoba_core::kcp::game_proto::PlayerInput) {
        let Some(handle) = self.lockstep_handle.as_ref() else { return };
        // +3 was the original Phase 4.3 lookahead for localhost zero-latency.
        // In practice the wire path (input_tx → tokio task → KCP write → server
        // receive → InputBuffer.submit) takes 1-4 server ticks (~16-66ms at
        // 60Hz), so server.current_tick has already passed the target. omb logs
        // "late InputSubmit ... target_tick=N current_tick=N+1..N+4" and drops
        // every input. Bump to +30 (~500ms @ 60Hz) so even loaded servers /
        // brief stalls don't reject inputs. Cost: 500ms input lag in pure
        // singleplayer. Acceptable for now; tune to RTT-based once we have
        // multi-client telemetry.
        let target_tick = self.current_sim_tick.wrapping_add(30);
        if let Err(e) = handle.input_tx.send((target_tick, input)) {
            log::warn!("[lockstep] input_tx send failed: {e}");
        }
    }

    // Phase 5.1 (pass 2): apply_event + 30+ legacy GameEvent handler
    // methods removed (entity_create / entity_move / entity_hp_update /
    // entity_delete / entity_facing_update / entity_speed_update /
    // entity_stall / projectile_create / projectile_delete / hero_stats_update /
    // hero_inventory_update / hero_abilities_info_update / map_paths_update /
    // map_regions_update / map_region_blockers_update / td_round_update /
    // td_lives_update / td_tower_templates_update / td_explosion_spawn /
    // tower_upgrade_apply / game_end). The legacy 0x02 GameEvent stream is
    // gone (Phase 4.5 server side, Phase 5.1 pass 1 client side); render_bridge
    // owns sprite rendering from sim state. Field cleanup deferred to pass 3.
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
    resources: &sprite_resources::SharedSpriteResources,
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
    let handle = resources.build_mesh(scene, resources.surf_facing.clone());
    scene.graph[handle]
        .local_transform_mut()
        .set_position(Vector3::new(-pos_x + offset_x, pos_y + offset_y, Z_HP_BAR - 0.02))
        .set_scale(Vector3::new(length, thickness, 1.0))
        .set_rotation(rotation);
    handle
}

fn build_path_segment(
    scene: &mut Scene,
    from: Vector2<f32>,
    to: Vector2<f32>,
) -> Option<Handle<Node>> {
    build_line_segment(scene, from, to, 0.05, Color::from_rgba(255, 100, 255, 180), Z_PATH)
}

/// 建立一條在 world 座標 (from → to) 的細長線段矩形。
/// 位置/角度皆已套用 X 軸翻轉，與 `build_path_segment` 邏輯一致。
fn build_line_segment(
    scene: &mut Scene,
    from: Vector2<f32>,
    to: Vector2<f32>,
    thickness: f32,
    color: Color,
    z: f32,
) -> Option<Handle<Node>> {
    let dx = to.x - from.x;
    let dy = to.y - from.y;
    let length = (dx * dx + dy * dy).sqrt();
    if length < f32::EPSILON {
        return None;
    }
    let center = Vector3::new(-(from.x + to.x) * 0.5, (from.y + to.y) * 0.5, z);
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
    .with_color(color)
    .build(&mut scene.graph)
    .transmute();
    Some(handle)
}

/// 把多邊形頂點以首尾相連的線段描出邊框。回傳每段的 scene handle。
fn build_polygon_outline(
    scene: &mut Scene,
    points: &[Vector2<f32>],
    thickness: f32,
    color: Color,
    z: f32,
) -> Vec<Handle<Node>> {
    let n = points.len();
    if n < 2 {
        return Vec::new();
    }
    let mut handles = Vec::with_capacity(n);
    for i in 0..n {
        let a = points[i];
        let b = points[(i + 1) % n];
        if let Some(h) = build_line_segment(scene, a, b, thickness, color, z) {
            handles.push(h);
        }
    }
    handles
}

/// 建立圓環：以 `segments` 個等分線段近似。以 `center` 為中心、半徑 `radius`。
/// 回傳 (handle, ring-local offset) 對，供 per-frame 追蹤 entity 位置用。
fn build_circle_outline(
    scene: &mut Scene,
    center: Vector2<f32>,
    radius: f32,
    segments: usize,
    thickness: f32,
    color: Color,
    z: f32,
) -> Vec<(Handle<Node>, Vector2<f32>)> {
    if radius <= 0.0 || segments < 3 {
        return Vec::new();
    }
    let mut pts: Vec<Vector2<f32>> = Vec::with_capacity(segments);
    for i in 0..segments {
        let angle = (i as f32) * std::f32::consts::TAU / (segments as f32);
        pts.push(Vector2::new(radius * angle.cos(), radius * angle.sin()));
    }
    let mut result: Vec<(Handle<Node>, Vector2<f32>)> = Vec::with_capacity(segments);
    for i in 0..segments {
        let a_local = pts[i];
        let b_local = pts[(i + 1) % segments];
        let a_world = Vector2::new(center.x + a_local.x, center.y + a_local.y);
        let b_world = Vector2::new(center.x + b_local.x, center.y + b_local.y);
        let offset = Vector2::new(
            (a_local.x + b_local.x) * 0.5,
            (a_local.y + b_local.y) * 0.5,
        );
        if let Some(h) = build_line_segment(scene, a_world, b_world, thickness, color, z) {
            result.push((h, offset));
        }
    }
    result
}

/// Per-frame circle outline 用 `SceneDrawingContext`（single batched draw call）。
/// 對應 `build_circle_outline` 的 RectangleBuilder 版本——在每 frame rebuild 的呼叫點用這個，
/// 避免 24-48 次 scene-graph 增刪。座標慣例與 `build_line_segment` 一致：x 取負。
/// 注意：drawing_context 每 frame 在 update() 開頭會 `clear_lines()`，所以僅適用 per-frame redraw。
fn add_circle_lines(
    scene: &mut Scene,
    center: Vector2<f32>,
    radius: f32,
    segments: usize,
    color: Color,
    z: f32,
) {
    use fyrox::scene::debug::Line;
    if radius <= 0.0 || segments < 3 {
        return;
    }
    // 起點：θ=0 → (cx + r, cy)；x 翻負與 build_line_segment 對齊
    let mut prev = Vector3::new(-(center.x + radius), center.y, z);
    for k in 1..=segments {
        let theta = (k as f32) * std::f32::consts::TAU / (segments as f32);
        let (s, c) = theta.sin_cos();
        let next = Vector3::new(
            -(center.x + radius * c),
            center.y + radius * s,
            z,
        );
        scene.drawing_context.add_line(Line { begin: prev, end: next, color });
        prev = next;
    }
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

fn world_to_screen_approx(wx: f32, wy: f32, window_w: f32, window_h: f32, world_height: f32) -> Vector2<f32> {
    let aspect = window_w / window_h;
    let world_width = world_height * aspect;
    // +X world → +X screen（camera 的 -1 X scale 已把原本的翻轉抵消）
    let sx = (wx / world_width + 0.5) * window_w;
    // +Y world → 螢幕上方（螢幕 pixel Y 向下，所以要反向）
    let sy = (-wy / world_height + 0.5) * window_h;
    Vector2::new(sx, sy)
}

/// Ray-casting 點在多邊形內判定（凹/凸皆可）。與 omb/src/util/geometry.rs 同演算法。
fn point_in_polygon(p: Vector2<f32>, poly: &[Vector2<f32>]) -> bool {
    if poly.len() < 3 { return false; }
    let mut inside = false;
    let n = poly.len();
    let mut j = n - 1;
    for i in 0..n {
        let pi = poly[i];
        let pj = poly[j];
        let cond = (pi.y > p.y) != (pj.y > p.y)
            && p.x < (pj.x - pi.x) * (p.y - pi.y) / (pj.y - pi.y + f32::EPSILON) + pi.x;
        if cond { inside = !inside; }
        j = i;
    }
    inside
}

/// 點到線段 (a-b) 的最短距離平方。
fn point_segment_dist_sq(p: Vector2<f32>, a: Vector2<f32>, b: Vector2<f32>) -> f32 {
    let ab = b - a;
    let ap = p - a;
    let len_sq = ab.x * ab.x + ab.y * ab.y;
    if len_sq < 1e-8 { return ap.norm_squared(); }
    let t = (ap.x * ab.x + ap.y * ab.y) / len_sq;
    let t = t.clamp(0.0, 1.0);
    let proj = a + ab * t;
    (p - proj).norm_squared()
}

/// 圓 vs 多邊形：圓心在內 → true；或任一邊距圓心 < r → true。
fn circle_hits_polygon(center: Vector2<f32>, r: f32, poly: &[Vector2<f32>]) -> bool {
    if poly.len() < 3 { return false; }
    if point_in_polygon(center, poly) { return true; }
    let r2 = r * r;
    let n = poly.len();
    for i in 0..n {
        let a = poly[i];
        let b = poly[(i + 1) % n];
        if point_segment_dist_sq(center, a, b) < r2 { return true; }
    }
    false
}
