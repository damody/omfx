//! omfx - 2D 塔防網路渲染器 (Fyrox 1.0)
//!
//! 純網路 renderer：所有 game state 都由 omb backend 透過 gRPC 驅動。
//! 不包含本地 game logic；entities 的 create/move/delete 都由 server events 決定。
#![allow(warnings)]

use fyrox::graph::prelude::*;
use fyrox::{
    asset::manager::ResourceManager,
    core::{
        algebra::{UnitQuaternion, Vector2, Vector3},
        color::Color,
        pool::Handle,
        reflect::prelude::*,
        visitor::prelude::*,
    },
    event::{ElementState, Event, MouseButton, MouseScrollDelta, WindowEvent},
    gui::{
        border::BorderBuilder,
        brush::{Brush, GradientPoint},
        canvas::CanvasBuilder,
        formatted_text::WrapMode,
        image::{ImageBuilder, ImageMessage},
        message::{MessageDirection, UiMessage},
        text::{Text, TextBuilder, TextMessage},
        text_box::TextBox,
        widget::{WidgetBuilder, WidgetMessage},
        HorizontalAlignment, Thickness, UiNode, UserInterface, VerticalAlignment,
    },
    material::{Material, MaterialResource},
    plugin::{error::GameResult, Plugin, PluginContext, PluginRegistrationContext},
    resource::{
        model::{Model, ModelResource, ModelResourceExtension},
        texture::{
            CompressionOptions, TextureImportOptions, TextureMinificationFilter, TextureResource,
            TextureResourceExtension,
        },
    },
    scene::{
        animation::{prelude::AnimationPlayerBuilder, Animation},
        base::BaseBuilder,
        camera::{CameraBuilder, OrthographicProjection, Projection},
        dim2::rectangle::{Rectangle, RectangleBuilder},
        light::{point::PointLightBuilder, BaseLightBuilder},
        mesh::Mesh,
        node::Node,
        sound::{DataSource, SoundBuilder, Status},
        transform::TransformBuilder,
        EnvironmentLightingSource, Scene,
    },
};

use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use fyrox_sound::buffer::SoundBufferResourceExtension;

use omoba_core::lockstep_timing::{LockstepTiming, LOCKSTEP_ONE_SECOND_TICKS_U32, LOCKSTEP_TPS};

pub use fyrox;

#[path = "audio_settings.rs"]
pub(crate) mod audio_settings;
#[path = "backend_session.rs"]
pub(crate) mod backend_session;

#[path = "hotkeys.rs"]
pub(crate) mod hotkeys;
#[path = "lockstep_client.rs"]
pub(crate) mod lockstep_client;
#[path = "pregame.rs"]
pub(crate) mod pregame;
#[path = "render_bridge.rs"]
pub(crate) mod render_bridge;
#[path = "sim_runner.rs"]
pub(crate) mod sim_runner;
#[path = "sprite_resources.rs"]
pub(crate) mod sprite_resources;

const ABILITY_ICON_FALLBACK_PATH: &str = "data/ability_icons/ability_default_placeholder.png";
const DEFAULT_DLL_PATH: &str = "scripts/base_content.dll";
const DEFAULT_GAME_TOML_PATH: &str = "game.toml";
const DEFAULT_STORY_DATA_DIR: &str = "scripts/lua_data";

const PENDING_INPUT_MAX_AGE_MS: u64 = 5_000;
const INPUT_LATENCY_CAPACITY: usize = (LOCKSTEP_TPS as usize) * 2;
const RENDER_UPDATE_TPS: u32 = LOCKSTEP_TPS;
const INPUT_LOOKAHEAD_TICKS: u32 = 2;
const INPUT_SAME_FRAME_WAIT_US: u64 = 2_000;
const RENDER_FX_SEEN_RETENTION_TICKS: u32 = LOCKSTEP_ONE_SECOND_TICKS_U32 / 2;

type TowerFireFxKey = (u32, u32, u32);
type AttackPhaseFxKey = (u32, u32, u32, u32);
type AttackCancelFxKey = (u32, u32, u32, u32);

fn perfetto_deep_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("OMFX_PERFETTO_DETAIL")
            .map(|value| value.eq_ignore_ascii_case("deep"))
            .unwrap_or(false)
    })
}

fn input_lookahead_ticks(_timing: LockstepTiming) -> u32 {
    INPUT_LOOKAHEAD_TICKS
}

struct FrontendConfigFile {
    path: PathBuf,
    text: String,
}

fn frontend_config_file() -> Option<&'static FrontendConfigFile> {
    static CONFIG: OnceLock<Option<FrontendConfigFile>> = OnceLock::new();
    CONFIG
        .get_or_init(|| {
            let path = PathBuf::from(
                std::env::var("OMFX_GAME_TOML")
                    .unwrap_or_else(|_| DEFAULT_GAME_TOML_PATH.to_string()),
            );
            std::fs::read_to_string(&path)
                .ok()
                .map(|text| FrontendConfigFile { path, text })
        })
        .as_ref()
}

fn parse_toml_scalar(raw: &str) -> String {
    raw.trim()
        .trim_start_matches('"')
        .trim_end_matches('"')
        .to_string()
}

fn frontend_config_section_value(section: &str, key: &str) -> Option<String> {
    let config = frontend_config_file()?;
    let mut current_section = "";
    for line in config.text.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(name) = line.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
            current_section = name.trim();
            continue;
        }
        if current_section != section {
            continue;
        }
        let mut parts = line.splitn(2, '=');
        let k = parts.next()?.trim();
        if k == key {
            return Some(parse_toml_scalar(parts.next()?));
        }
    }
    None
}

fn frontend_config_value(key: &str) -> Option<String> {
    frontend_config_section_value("client", key)
        .or_else(|| frontend_config_section_value("content", key))
        .or_else(|| frontend_config_section_value("server", key))
}

fn frontend_config_env_or_section_value(env_key: &str, section: &str, key: &str) -> Option<String> {
    std::env::var(env_key)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| frontend_config_section_value(section, key))
}

fn frontend_config_f32(env_key: &str, section: &str, key: &str) -> Option<(String, f32)> {
    let raw = frontend_config_env_or_section_value(env_key, section, key)?;
    raw.parse::<f32>().ok().map(|parsed| (raw, parsed))
}

fn frontend_env_truthy(key: &str) -> bool {
    std::env::var(key)
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn frontend_config_u32_or_default(env_key: &str, section: &str, key: &str, default: u32) -> u32 {
    let Some(raw) = frontend_config_env_or_section_value(env_key, section, key) else {
        return default;
    };
    match raw.parse::<u32>() {
        Ok(0) => {
            log::warn!("{}={} is invalid; using default {}", key, raw, default);
            default
        }
        Ok(value) => value,
        Err(_) => {
            log::warn!("{}={} is invalid; using default {}", key, raw, default);
            default
        }
    }
}

fn resolve_frontend_config_path(value: String) -> PathBuf {
    let path = PathBuf::from(value);
    if path.is_absolute() {
        return path;
    }
    if let Some(config) = frontend_config_file() {
        return config
            .path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(path);
    }
    path
}

fn frontend_config_path(section: &str, key: &str) -> Option<PathBuf> {
    frontend_config_section_value(section, key).map(resolve_frontend_config_path)
}

fn absolute_existing_or_joined_path(path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        return path.canonicalize().unwrap_or(path);
    }
    std::env::current_dir()
        .map(|cwd| cwd.join(&path))
        .unwrap_or_else(|_| path.clone())
        .canonicalize()
        .unwrap_or_else(|_| {
            std::env::current_dir()
                .map(|cwd| cwd.join(&path))
                .unwrap_or(path)
        })
}

fn frontend_server_addr() -> String {
    if let Ok(value) = std::env::var("OMB_KCP_ADDR") {
        if !value.trim().is_empty() {
            return value;
        }
    }
    if let Some(value) = frontend_config_section_value("client", "SERVER_ADDR") {
        return value;
    }
    let ip = frontend_config_section_value("server", "SERVER_IP")
        .unwrap_or_else(|| "127.0.0.1".to_string());
    let port =
        frontend_config_section_value("server", "SERVER_PORT").unwrap_or_else(|| "50061".into());
    format!(
        "{}:{}",
        if ip == "localhost" { "127.0.0.1" } else { &ip },
        port
    )
}

fn set_env_if_missing(name: &str, value: String) {
    let should_set = std::env::var(name)
        .map(|v| v.trim().is_empty())
        .unwrap_or(true);
    if should_set {
        std::env::set_var(name, value);
    }
}

fn bool_config_value(section: &str, key: &str) -> Option<bool> {
    frontend_config_section_value(section, key).and_then(|value| {
        match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        }
    })
}

fn apply_frontend_runtime_env_from_config() {
    if let Some(path) = frontend_config_path("content", "DLL_PATH") {
        set_env_if_missing("OMB_DLL_PATH", path.to_string_lossy().into_owned());
    }
    if let Some(path) = frontend_config_path("content", "SCRIPTS_DIR") {
        set_env_if_missing("OMB_SCRIPTS_DIR", path.to_string_lossy().into_owned());
    }
    if let Some(path) = frontend_config_path("content", "LUA_CONTENT_ROOT") {
        set_env_if_missing("OMB_LUA_CONTENT_ROOT", path.to_string_lossy().into_owned());
    }
    if let Some(path) = frontend_config_path("content", "STORY_DATA_DIR") {
        set_env_if_missing("OMB_STORY_DATA_DIR", path.to_string_lossy().into_owned());
    }
    if let Some(enabled) = bool_config_value("content", "LUA_CONTENT") {
        set_env_if_missing(
            "OMB_LUA_CONTENT",
            if enabled { "1" } else { "0" }.to_string(),
        );
    }
    if let Some(enabled) = bool_config_value("content", "LUA_HOT_RELOAD") {
        set_env_if_missing(
            "OMB_LUA_HOT_RELOAD",
            if enabled { "1" } else { "0" }.to_string(),
        );
    }
}

fn tower_fire_fx_key(cue: &sim_runner::TowerFireFx) -> TowerFireFxKey {
    (cue.entity_id, cue.entity_gen, cue.spawn_tick)
}

fn attack_phase_fx_key(cue: &sim_runner::AttackPhaseFx) -> AttackPhaseFxKey {
    (
        cue.entity_id,
        cue.entity_gen,
        cue.spawn_tick,
        cue.attack_seq,
    )
}

fn attack_cancel_fx_key(cue: &sim_runner::AttackCancelFx) -> AttackCancelFxKey {
    (
        cue.entity_id,
        cue.entity_gen,
        cue.spawn_tick,
        cue.attack_seq,
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputActionKind {
    TowerPlace,
    TowerSell,
    TowerUpgrade,
    TowerAbilityCast,
    ItemUse,
    StartRound,
    TogglePause,
    ToggleGameSpeed,
    DebugSpawnCreep,
    MoveTo,
    AttackMove,
    AttackTarget,
    SetTowerTargetPriority,
    CastAbility,
    UpgradeAbility,
    NoOp,
}

impl InputActionKind {
    fn from_player_input(input: &omoba_core::kcp::game_proto::PlayerInput) -> Self {
        use omoba_core::kcp::game_proto::player_input::Action;
        match input.action.as_ref() {
            Some(Action::TowerPlace(_)) => Self::TowerPlace,
            Some(Action::TowerSell(_)) => Self::TowerSell,
            Some(Action::TowerUpgrade(_)) => Self::TowerUpgrade,
            Some(Action::TowerAbilityCast(_)) => Self::TowerAbilityCast,
            Some(Action::ItemUse(_)) => Self::ItemUse,
            Some(Action::StartRound(_)) => Self::StartRound,
            Some(Action::TogglePause(_)) => Self::TogglePause,
            Some(Action::ToggleGameSpeed(_)) => Self::ToggleGameSpeed,
            Some(Action::DebugSpawnCreep(_)) => Self::DebugSpawnCreep,
            Some(Action::MoveTo(_)) => Self::MoveTo,
            Some(Action::AttackMove(_)) => Self::AttackMove,
            Some(Action::AttackTarget(_)) => Self::AttackTarget,
            Some(Action::SetTowerTargetPriority(_)) => Self::SetTowerTargetPriority,
            Some(Action::CastAbility(_)) => Self::CastAbility,
            Some(Action::UpgradeAbility(_)) => Self::UpgradeAbility,
            Some(Action::NoOp(_)) | None => Self::NoOp,
        }
    }
}

fn tower_owned_by_local(owner_player_id: Option<u32>, local_player_id: u32) -> bool {
    owner_player_id == Some(local_player_id)
}

fn entity_owned_by_local(entity: &NetworkEntity, local_player_id: u32) -> bool {
    tower_owned_by_local(entity.owner_player_id, local_player_id)
}

fn tower_priority_label(priority: &str) -> &'static str {
    match priority {
        "first" => "First",
        "last" => "Last",
        "nearest" => "Nearest",
        "farthest" => "Farthest",
        "highest_health" => "High HP",
        "lowest_health" => "Low HP",
        _ => "First",
    }
}

fn hero_command_status_text(command: &omoba_core::runtime::native::HeroCommandSnapshot) -> String {
    let kind = match command.command_type.as_str() {
        "move_to" => "Move",
        "attack_move" => "Attack Move",
        "attack_target" => "Attack Target",
        _ => "Command",
    };
    let target = command
        .target_entity_id
        .map(|id| format!(" target #{}", id))
        .unwrap_or_default();
    let destination = command
        .destination
        .map(|(x, y)| format!(" dest {:.0},{:.0}", x, y))
        .unwrap_or_default();
    let waypoint = command
        .next_waypoint
        .map(|(x, y)| format!(" next {:.0},{:.0}", x, y))
        .unwrap_or_default();
    format!(
        "{}{}{}{}  queue {}/{}",
        kind, target, destination, waypoint, command.queued_count, command.queue_limit
    )
}

#[derive(Clone, Debug)]
struct PendingInput {
    submit_wall_clock_us: u64,
    submit_instant: Instant,
    base_tick: u32,
    target_tick: u32,
    action_kind: InputActionKind,
    origin_kind: lockstep_client::InputOriginKind,
    origin_us: u64,
    send_lockstep_input_us: u64,
    submit_start_us: Option<u64>,
    submit_done_us: Option<u64>,
    client_receive_tickbatch_us: Option<u64>,
    game_forward_to_sim_us: Option<u64>,
    extract_data_for_render_us: Option<u64>,
    server_receive_tick: Option<u32>,
    server_drain_tick: Option<u32>,
    server_queue_us: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PendingInputDiagnostic {
    input_id: u32,
    action_kind: InputActionKind,
    base_tick: u32,
    target_tick: u32,
    pending_age_ms: u32,
    has_submit_start: bool,
    has_submit_done: bool,
    has_client_receive_tickbatch: bool,
    has_game_forward_to_sim: bool,
    has_extract_data_for_render: bool,
    server_receive_tick: Option<u32>,
    server_drain_tick: Option<u32>,
    server_queue_us: Option<u64>,
}

#[derive(Clone, Debug, Default)]
pub struct LatencyPhaseDurations {
    pub origin_to_send_us: u64,
    pub send_to_submit_start_us: u64,
    pub submit_io_us: u64,
    pub submit_to_client_receive_us: u64,
    pub server_queue_us: u64,
    pub client_receive_to_forward_us: u64,
    pub forward_to_extract_data_for_render_us: u64,
    pub extract_data_for_render_to_pair_us: u64,
}

#[derive(Clone, Debug)]
pub struct LatencySample {
    pub input_id: u32,
    pub action_kind: InputActionKind,
    pub total_ms: u32,
    pub submitted_at: Instant,
    pub origin_kind: lockstep_client::InputOriginKind,
    pub target_tick: u32,
    pub server_receive_tick: Option<u32>,
    pub server_drain_tick: Option<u32>,
    pub phases: LatencyPhaseDurations,
}

#[derive(Debug)]
pub struct InputLatencyMeter {
    samples: VecDeque<LatencySample>,
    last_compute_at: Instant,
    cached_p50_ms: u32,
    cached_p99_ms: u32,
    cached_max_ms: u32,
    cached_latest_ms: u32,
}

impl Default for InputLatencyMeter {
    fn default() -> Self {
        Self {
            samples: VecDeque::new(),
            last_compute_at: Instant::now(),
            cached_p50_ms: 0,
            cached_p99_ms: 0,
            cached_max_ms: 0,
            cached_latest_ms: 0,
        }
    }
}

impl InputLatencyMeter {
    pub fn push(&mut self, sample: LatencySample) {
        self.cached_latest_ms = sample.total_ms;
        self.samples.push_back(sample);
        while self.samples.len() > INPUT_LATENCY_CAPACITY {
            self.samples.pop_front();
        }
        if self.samples.len() == 1 {
            self.cached_p50_ms = self.cached_latest_ms;
            self.cached_p99_ms = self.cached_latest_ms;
            self.cached_max_ms = self.cached_latest_ms;
        }
    }

    pub fn maybe_recompute(&mut self, now: Instant) {
        if now.duration_since(self.last_compute_at) < Duration::from_secs(1) {
            return;
        }
        self.last_compute_at = now;
        if self.samples.is_empty() {
            self.cached_p50_ms = 0;
            self.cached_p99_ms = 0;
            self.cached_max_ms = 0;
            return;
        }
        let mut values: Vec<u32> = self.samples.iter().map(|s| s.total_ms).collect();
        values.sort_unstable();
        let last_idx = values.len() - 1;
        self.cached_p50_ms = values[(values.len() / 2).min(last_idx)];
        self.cached_p99_ms = values[((values.len() as f32 * 0.99) as usize).min(last_idx)];
        self.cached_max_ms = values[last_idx];
        self.cached_latest_ms = self.samples.back().map(|s| s.total_ms).unwrap_or(0);
    }

    fn has_samples(&self) -> bool {
        !self.samples.is_empty()
    }
}

fn pending_input_age_ms(pending: &PendingInput, now_us: u64) -> u32 {
    now_us
        .saturating_sub(pending.submit_wall_clock_us)
        .saturating_div(1_000)
        .min(u64::from(u32::MAX)) as u32
}

fn oldest_pending_input_age_ms(
    pending_inputs: &HashMap<u32, PendingInput>,
    now_us: u64,
) -> Option<u32> {
    pending_inputs
        .values()
        .map(|pending| pending_input_age_ms(pending, now_us))
        .max()
}

fn pending_input_diagnostic(
    input_id: u32,
    pending: &PendingInput,
    now_us: u64,
) -> PendingInputDiagnostic {
    PendingInputDiagnostic {
        input_id,
        action_kind: pending.action_kind,
        base_tick: pending.base_tick,
        target_tick: pending.target_tick,
        pending_age_ms: pending_input_age_ms(pending, now_us),
        has_submit_start: pending.submit_start_us.is_some(),
        has_submit_done: pending.submit_done_us.is_some(),
        has_client_receive_tickbatch: pending.client_receive_tickbatch_us.is_some(),
        has_game_forward_to_sim: pending.game_forward_to_sim_us.is_some(),
        has_extract_data_for_render: pending.extract_data_for_render_us.is_some(),
        server_receive_tick: pending.server_receive_tick,
        server_drain_tick: pending.server_drain_tick,
        server_queue_us: pending.server_queue_us,
    }
}

fn format_input_lag_status(meter: &InputLatencyMeter, oldest_pending_ms: Option<u32>) -> String {
    let paired = if meter.has_samples() {
        Some(format!(
            "p50 {} / p99 {} ms",
            meter.cached_p50_ms, meter.cached_p99_ms
        ))
    } else {
        None
    };
    let pending = oldest_pending_ms
        .filter(|age_ms| !meter.has_samples() || *age_ms > meter.cached_p99_ms)
        .map(|age_ms| format!("pending {} ms", age_ms));

    match (paired, pending) {
        (Some(paired), Some(pending)) => format!("{} | {}", paired, pending),
        (Some(paired), None) => paired,
        (None, Some(pending)) => pending,
        (None, None) => "—".into(),
    }
}

fn wall_clock_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros().min(u128::from(u64::MAX)) as u64)
        .unwrap_or(0)
}

/// 階段 4.3：固定 32 原始縮放。 Vec2I 將 `real_units * 1024` 儲存為 sint32
/// （參見 proto/game.proto：「除以 1024 以獲得真實單位」）。後端邏輯
/// 座標 = 渲染座標 / WORLD_SCALE；將其乘以 1024 即可得到原始資料。
const FIXED32_ONE: f32 = 1024.0;

/// 將 omfx 渲染空間世界位置轉換為後端固定 32 原始位置
/// `PlayerInput::MoveTo` / `CastAbility::target_pos` 使用的 `Vec2I`。
fn world_render_to_vec2i(world: Vector2<f32>) -> omoba_core::kcp::game_proto::Vec2I {
    let backend_x = world.x / WORLD_SCALE;
    let backend_y = world.y / WORLD_SCALE;
    omoba_core::kcp::game_proto::Vec2I {
        x: (backend_x * FIXED32_ONE) as i32,
        y: (backend_y * FIXED32_ONE) as i32,
    }
}

fn ability_key_index(key: fyrox::keyboard::KeyCode) -> Option<u32> {
    use fyrox::keyboard::KeyCode;
    match key {
        KeyCode::KeyW => Some(0),
        KeyCode::KeyE => Some(1),
        KeyCode::KeyR => Some(2),
        KeyCode::KeyT => Some(3),
        _ => None,
    }
}

/// 解析度切換請求（設定頁下拉選單 → update() 套用）。
#[derive(Clone, Copy, Debug)]
enum DisplayModeRequest {
    Fullscreen,
    Windowed(u32, u32),
}

/// 設定頁「螢幕尺寸」下拉選單的 UI 節點與 hit-test 矩形。
#[derive(Default, Debug)]
struct ResolutionDropdownUi {
    built: bool,
    visible_applied: bool,
    bg: Handle<UiNode>,
    /// 每列：(列背景, 文字)
    rows: Vec<(Handle<UiNode>, Handle<Text>)>,
    row_rects: Vec<UiRect>,
    ok_bg: Handle<UiNode>,
    ok_text: Handle<Text>,
    ok_rect: UiRect,
}

/// 下拉選單的解析度選項（None = Fullscreen，放最下面）。
const RESOLUTION_OPTIONS: [Option<(u32, u32)>; 6] = [
    Some((1280, 720)),
    Some((1366, 768)),
    Some((1600, 900)),
    Some((1920, 1080)),
    Some((2560, 1440)),
    None,
];

/// 熱鍵設定面板（F1 開關）的 UI 節點與 hit-test 矩形。
#[derive(Default, Debug)]
struct HotkeyPanelUi {
    built: bool,
    /// 目前面板是否處於顯示位置（避免每 frame 重送隱藏訊息）。
    visible_applied: bool,
    backdrop: Handle<UiNode>,
    bg: Handle<UiNode>,
    header_banner: Handle<UiNode>,
    title: Handle<Text>,
    reset_bg: Handle<UiNode>,
    reset_text: Handle<Text>,
    reset_rect: UiRect,
    close_bg: Handle<UiNode>,
    close_text: Handle<Text>,
    close_rect: UiRect,
    cat_titles: Vec<Handle<Text>>,
    /// 每列：(列卡片背景, 動作 label, 按鍵按鈕背景, 按鍵文字)
    rows: Vec<(Handle<UiNode>, Handle<Text>, Handle<UiNode>, Handle<Text>)>,
    /// 每列按鍵按鈕的 hit-test 矩形（與 rows 對齊）。
    row_rects: Vec<UiRect>,
    /// 每列對應的 action id（與 rows 對齊）。
    row_ids: Vec<&'static str>,
}

// ---------------------------------------------------------------------------
// 常數
// ---------------------------------------------------------------------------

const GRID_COLS: usize = 12;
const GRID_ROWS: usize = 8;
const CELL_SIZE: f32 = 1.0;
const GRID_ORIGIN_X: f32 = -6.0;
const GRID_ORIGIN_Y: f32 = -4.0;

// 後端→渲染座標比例（後端使用800等大單位）
const WORLD_SCALE: f32 = 0.01; // 800 backend → 8.0 render
const TD_PATH_HALF_WIDTH_BACKEND: f32 = 64.0;
const UI_HIDDEN_POS: f32 = -9999.0;
const TD_UI_MAX_UPGRADE_LEVEL: u8 = 4;
const TD_SHOP_LAYOUT_DEBUG_MIN_CARDS: usize = 20;
const TD_UI_REF_W: f32 = 1920.0;
const TD_UI_REF_H: f32 = 1080.0;
const TD_CAMERA_VERTICAL_SIZE: f32 = 16.5;
const TD_CAMERA_WORLD_CENTER_X: f32 = 0.75;
const TD_CAMERA_WORLD_CENTER_Y: f32 = 0.0;

#[derive(Clone, Copy, Debug)]
struct TdCameraViewConfig {
    vertical_size: f32,
    scene_position: Vector2<f32>,
}

fn td_camera_view_config() -> TdCameraViewConfig {
    TdCameraViewConfig {
        vertical_size: TD_CAMERA_VERTICAL_SIZE,
        scene_position: Vector2::new(TD_CAMERA_WORLD_CENTER_X, TD_CAMERA_WORLD_CENTER_Y),
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct UiRect {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
}

impl UiRect {
    fn pos(self) -> Vector2<f32> {
        Vector2::new(self.x, self.y)
    }

    fn tuple(self) -> (f32, f32, f32, f32) {
        (self.x, self.y, self.w, self.h)
    }

    fn right(self) -> f32 {
        self.x + self.w
    }

    fn bottom(self) -> f32 {
        self.y + self.h
    }

    fn contains(self, p: Vector2<f32>) -> bool {
        p.x >= self.x && p.x <= self.right() && p.y >= self.y && p.y <= self.bottom()
    }

    fn intersection(self, other: UiRect) -> Option<UiRect> {
        let x0 = self.x.max(other.x);
        let y0 = self.y.max(other.y);
        let x1 = self.right().min(other.right());
        let y1 = self.bottom().min(other.bottom());
        if x1 > x0 && y1 > y0 {
            Some(UiRect {
                x: x0,
                y: y0,
                w: x1 - x0,
                h: y1 - y0,
            })
        } else {
            None
        }
    }
}

fn pregame_wrap_line(text: &str, max_chars: usize) -> Vec<String> {
    let text = text.trim();
    if text.is_empty() {
        return Vec::new();
    }

    let mut lines = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        if word.chars().count() > max_chars {
            if !current.is_empty() {
                lines.push(current);
                current = String::new();
            }
            let mut chunk = String::new();
            for ch in word.chars() {
                if chunk.chars().count() >= max_chars {
                    lines.push(chunk);
                    chunk = String::new();
                }
                chunk.push(ch);
            }
            if !chunk.is_empty() {
                current = chunk;
            }
            continue;
        }
        let next_len =
            current.chars().count() + if current.is_empty() { 0 } else { 1 } + word.chars().count();
        if !current.is_empty() && next_len > max_chars {
            lines.push(current);
            current = String::new();
        }
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(word);
    }
    if !current.is_empty() {
        lines.push(current);
    }
    if lines.is_empty() {
        lines.push(text.chars().take(max_chars).collect());
    }
    lines
}

fn pregame_button_text(label: &str, description: &str, active: bool, button_w: f32) -> String {
    let max_chars = if button_w >= 420.0 {
        42
    } else if button_w >= 340.0 {
        34
    } else {
        28
    };
    let mut lines = Vec::new();
    lines.push(label.trim().to_string());
    lines.extend(
        pregame_wrap_line(description, max_chars)
            .into_iter()
            .take(2),
    );
    if !active {
        lines.push("Locked".to_string());
    }
    lines.join("\n")
}

fn pregame_ref_rect(window_size: Vector2<f32>, x: f32, y: f32, w: f32, h: f32) -> UiRect {
    let scale = (window_size.x / 2048.0)
        .min(window_size.y / 1152.0)
        .max(0.01);
    let content_w = 2048.0 * scale;
    let content_h = 1152.0 * scale;
    UiRect {
        x: (window_size.x - content_w) * 0.5 + x * scale,
        y: (window_size.y - content_h) * 0.5 + y * scale,
        w: w * scale,
        h: h * scale,
    }
}

fn pregame_button_label(label: &str, description: &str, active: bool) -> String {
    let mut lines = vec![label.trim().to_string()];
    if !description.trim().is_empty() {
        lines.push(description.trim().to_string());
    }
    if !active {
        lines.push("鎖定".to_string());
    }
    lines.join("\n")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PregameVisualRole {
    Button,
    Decoration,
}

impl Default for PregameVisualRole {
    fn default() -> Self {
        Self::Button
    }
}

#[derive(Debug, Default)]
struct PregameButtonUi {
    bg: Handle<UiNode>,
    image: Handle<UiNode>,
    text: Handle<Text>,
    role: PregameVisualRole,
}

#[derive(Debug, Default)]
struct PregameUi {
    background: Handle<UiNode>,
    panel: Handle<UiNode>,
    title: Handle<Text>,
    subtitle: Handle<Text>,
    status: Handle<Text>,
    buttons: Vec<PregameButtonUi>,
}

#[derive(Debug, Default)]
struct InGameReturnUi {
    bg: Handle<UiNode>,
    text: Handle<Text>,
}

#[derive(Debug, Default)]
struct SettingsSlider {
    track: Handle<UiNode>,
    fill: Handle<UiNode>,
    thumb: Handle<UiNode>,
    icon: Handle<UiNode>,
    label: Handle<Text>,
    pct_text: Handle<Text>,
    track_rect: UiRect,
}

#[derive(Debug, Default)]
struct SettingsPlaceholderBtn {
    bg: Handle<UiNode>,
    label: Handle<Text>,
    sublabel: Handle<Text>,
}

// ── 個人統計數據頁（Profile）placeholder 資料 ────────────────────────────
// Phase 1：僅前端版面骨架，數字全為假資料。之後接上真實統計時，
// 只需替換這些常數／改為從別處讀取即可，UI 版面程式碼不用動。
const PROFILE_PLAYER_NAME: &str = "User";
const PROFILE_LEVEL: u32 = 0;
const PROFILE_XP_PCT: f32 = 0.0;
const PROFILE_FOLLOWERS: u32 = 128;
const PROFILE_SHARE_COUNT: (u32, u32) = (3, 7);

/// (label, count)
const PROFILE_MEDALS: [(&str, u32); 8] = [
    ("新手上路", 1),
    ("常勝軍", 5),
    ("完美防守", 3),
    ("速通王", 2),
    ("塔防大師", 1),
    ("收藏家", 12),
    ("連勝紀錄", 7),
    ("探索者", 4),
];

/// (label, usage_count)
/// 名稱＝遊戲真實單位（英雄取自 templates/heroes.lua，塔取自 templates/towers.lua）；
/// usage_count 仍為佔位假資料——遊戲尚未追蹤使用次數（第二階段戰績系統才會有真值）。
/// 遊戲目前只有 2 位英雄，故此處只放 2 位。
const PROFILE_TOP_HEROES: [(&str, u32); 2] = [("雜賀孫市", 42), ("伊達政宗", 35)];
const PROFILE_TOP_MONKEYS: [(&str, u32); 3] =
    [("糖球砲手", 58), ("刺蝟射手", 33), ("馬卡龍砲車", 19)];

/// (label, value)
const PROFILE_STAT_ROWS: [(&str, &str); 8] = [
    ("玩過的遊戲", "272"),
    ("獲勝的遊戲", "93"),
    ("最高回合(全部時間)", "180"),
    ("最高回合(當前版本)", "89"),
    ("最高回合 CHIMPS", "100"),
    ("最高回合放氣", "0"),
    ("使用過的塔數", "1,204"),
    ("擊殺的怪物數", "58,340"),
];

#[derive(Debug, Default)]
struct ProfileMedalUi {
    bg: Handle<UiNode>,
    count_text: Handle<Text>,
}

#[derive(Debug, Default)]
struct ProfilePortraitUi {
    bg: Handle<UiNode>,
    name_text: Handle<Text>,
    count_text: Handle<Text>,
}

#[derive(Debug, Default)]
struct ProfileStatRowUi {
    label_text: Handle<Text>,
    value_text: Handle<Text>,
    checkbox_bg: Handle<UiNode>,
}

#[derive(Debug, Default)]
struct ProfilePanel {
    bg: Handle<UiNode>,
    back_btn: Handle<UiNode>,
    back_btn_text: Handle<Text>,
    back_btn_rect: UiRect,
    // 標頭列
    header_bg: Handle<UiNode>,
    avatar_bg: Handle<UiNode>,
    avatar_text: Handle<Text>,
    name_text: Handle<Text>,
    level_badge_bg: Handle<UiNode>,
    level_text: Handle<Text>,
    xp_track: Handle<UiNode>,
    xp_fill: Handle<UiNode>,
    followers_text: Handle<Text>,
    visibility_toggle_bg: Handle<UiNode>,
    visibility_text: Handle<Text>,
    publish_btn_bg: Handle<UiNode>,
    publish_btn_text: Handle<Text>,
    settings_gear_bg: Handle<UiNode>,
    settings_gear_text: Handle<Text>,
    // 獎牌牆
    medals_title_text: Handle<Text>,
    medals: Vec<ProfileMedalUi>,
    // 左欄：頂級英雄 / 頂級砲塔
    heroes_banner_bg: Handle<UiNode>,
    heroes_banner_text: Handle<Text>,
    heroes: Vec<ProfilePortraitUi>,
    monkeys_banner_bg: Handle<UiNode>,
    monkeys_banner_text: Handle<Text>,
    monkeys: Vec<ProfilePortraitUi>,
    // 右欄：整體統計數據
    stats_header_bg: Handle<UiNode>,
    stats_header_text: Handle<Text>,
    stats_collapse_text: Handle<Text>,
    stats_share_count_text: Handle<Text>,
    stats_rows: Vec<ProfileStatRowUi>,
}

#[derive(Debug, Default)]
struct SettingsPanel {
    bg: Handle<UiNode>,
    title_bg: Handle<UiNode>,
    title_text: Handle<Text>,
    back_btn: Handle<UiNode>,
    back_btn_text: Handle<Text>,
    back_btn_rect: UiRect,
    resolution_badge_bg: Handle<UiNode>,
    resolution_badge_label: Handle<Text>,
    resolution_badge_value: Handle<Text>,
    left_panel_bg: Handle<UiNode>,
    left_panel_jukebox_bg: Handle<UiNode>,
    left_panel_jukebox_inner: Handle<UiNode>,
    left_panel_btn_bg: Handle<UiNode>,
    left_panel_btn_text: Handle<Text>,
    left_panel_toggle_bg: Handle<UiNode>,
    left_panel_enable_label: Handle<Text>,
    slider_panel_bg: Handle<UiNode>,
    sfx: SettingsSlider,
    music: SettingsSlider,
    speed: SettingsSlider,
    placeholder_btns: Vec<SettingsPlaceholderBtn>,
}

fn td_ui_ref_scale(window_size: Vector2<f32>) -> (f32, f32) {
    (
        (window_size.x.max(1.0) / TD_UI_REF_W).max(0.01),
        (window_size.y.max(1.0) / TD_UI_REF_H).max(0.01),
    )
}

fn td_ui_ref_rect(window_size: Vector2<f32>, x: f32, y: f32, w: f32, h: f32) -> UiRect {
    let (sx, sy) = td_ui_ref_scale(window_size);
    UiRect {
        x: x * sx,
        y: y * sy,
        w: w * sx,
        h: h * sy,
    }
}

fn td_start_control_label(
    is_paused: bool,
    round_is_running: bool,
    current_round: u32,
    total_rounds: u32,
    game_speed_multiplier: u32,
) -> &'static str {
    if is_paused {
        "RESUME"
    } else if total_rounds > 0 && current_round >= total_rounds {
        "DONE"
    } else if round_is_running && game_speed_multiplier >= 2 {
        "2X"
    } else if round_is_running {
        "1X"
    } else {
        "READY"
    }
}

fn td_start_control_color(
    is_paused: bool,
    round_is_running: bool,
    game_speed_multiplier: u32,
) -> Color {
    if is_paused {
        Color::from_rgba(45, 100, 145, 255)
    } else if round_is_running && game_speed_multiplier >= 2 {
        Color::from_rgba(205, 90, 20, 255)
    } else if round_is_running {
        Color::from_rgba(35, 120, 55, 255)
    } else {
        Color::from_rgba(0, 80, 0, 255)
    }
}

fn td_round_status_label(_round_is_running: bool, current_round: u32, total_rounds: u32) -> String {
    if total_rounds == 0 {
        return String::new();
    }
    if current_round >= total_rounds {
        return format!("完成 {} / {} 波", total_rounds, total_rounds);
    }
    let display_round = current_round.saturating_add(1);
    format!(
        "第 {} / {} 波",
        display_round.min(total_rounds),
        total_rounds
    )
}

fn td_mode_from_snapshot(lives: i32, total_rounds: u32) -> bool {
    lives > 0 || total_rounds > 0
}

const TOWER_ABILITY_BAR_PAGE_SIZE: usize = 6;
const TOWER_ABILITY_BAR_BOTTOM_MARGIN: f32 = 88.0;
const TOWER_ABILITY_REJECTION_VISIBLE_FOR: Duration = Duration::from_secs(3);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct AbilityBarKey {
    tower_entity_id: u32,
    ability_id: String,
}

#[derive(Clone, Debug, PartialEq)]
struct AbilityBarItem {
    key: AbilityBarKey,
    tower_label: String,
    ability_name: String,
    description: String,
    cooldown_total: f32,
    duration_total: f32,
    cooldown_remaining: f32,
    cooldown_text: String,
    active_remaining: f32,
    activation_serial: u32,
    shortcut: Option<u8>,
    visual_state: AbilityBarVisualState,
}

#[derive(Clone, Debug, PartialEq)]
enum AbilityBarVisualState {
    Ready,
    Cooling {
        fraction: f32,
        text: String,
    },
    Active {
        cooldown_fraction: f32,
        active_remaining: f32,
        text: String,
    },
}

#[derive(Clone, Debug, PartialEq)]
struct AbilityBarRenderedState {
    key: AbilityBarKey,
    visual: AbilityBarVisualState,
    enabled: bool,
}

fn ability_bar_visual_state(
    cooldown_total: f32,
    cooldown_remaining: f32,
    active_remaining: f32,
) -> AbilityBarVisualState {
    let fraction = if cooldown_total > 0.0 {
        (cooldown_remaining / cooldown_total).clamp(0.0, 1.0)
    } else {
        0.0
    };
    if active_remaining > 0.0 {
        let display_active = (active_remaining * 10.0).round() / 10.0;
        AbilityBarVisualState::Active {
            cooldown_fraction: fraction,
            active_remaining,
            text: format!("ACTIVE {display_active:.1}"),
        }
    } else if cooldown_remaining > 0.0 {
        AbilityBarVisualState::Cooling {
            fraction,
            text: format!("{cooldown_remaining:.1}"),
        }
    } else {
        AbilityBarVisualState::Ready
    }
}

fn ability_bar_elapsed_sim(
    elapsed_wall_clock: f32,
    is_paused: bool,
    game_speed_multiplier: u32,
) -> f32 {
    if is_paused {
        0.0
    } else {
        elapsed_wall_clock.max(0.0) * game_speed_multiplier as f32
    }
}

fn tower_ability_bar_y(window_height: f32, slot_height: f32) -> f32 {
    (window_height - slot_height - TOWER_ABILITY_BAR_BOTTOM_MARGIN).max(0.0)
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct AbilityBarInteractionState {
    page: usize,
    debounced: HashSet<AbilityBarKey>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct AbilityBarRejection {
    key: AbilityBarKey,
    reason: String,
    shown_at: Instant,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct AbilityBarRejectionState {
    last_result_serial: u32,
    visible: Option<AbilityBarRejection>,
}

impl AbilityBarRejectionState {
    fn observe_results(
        &mut self,
        results: &[sim_runner::TowerAbilityCastResultSnapshot],
        local_player_id: u32,
        now: Instant,
    ) {
        let Some(result) = results
            .iter()
            .find(|result| result.player_id == local_player_id)
        else {
            return;
        };
        if result.result_serial == self.last_result_serial {
            return;
        }
        self.last_result_serial = result.result_serial;
        if !result.accepted {
            self.visible = Some(AbilityBarRejection {
                key: AbilityBarKey {
                    tower_entity_id: result.tower_entity_id,
                    ability_id: result.ability_id.clone(),
                },
                reason: result.reason.clone(),
                shown_at: now,
            });
        } else {
            self.visible = None;
        }
    }

    fn retain_visible(&mut self, available_keys: &[AbilityBarKey], now: Instant) {
        let should_clear = self.visible.as_ref().is_some_and(|rejection| {
            now.saturating_duration_since(rejection.shown_at) >= TOWER_ABILITY_REJECTION_VISIBLE_FOR
                || !available_keys.iter().any(|key| key == &rejection.key)
        });
        if should_clear {
            self.visible = None;
        }
    }
}

impl AbilityBarInteractionState {
    fn change_page(&mut self, direction: i32, page_count: usize) {
        let last = page_count.max(1) - 1;
        self.page = if direction > 0 {
            (self.page + 1).min(last)
        } else if direction < 0 {
            self.page.saturating_sub(1)
        } else {
            self.page.min(last)
        };
    }

    fn authoritative_boundary(&mut self) {
        self.debounced.clear();
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct AbilityBarPageModel {
    page: usize,
    page_count: usize,
    previous_enabled: bool,
    next_enabled: bool,
    indicator: String,
}

fn ability_bar_page_model(item_count: usize, requested_page: usize) -> AbilityBarPageModel {
    let page_count = item_count.div_ceil(TOWER_ABILITY_BAR_PAGE_SIZE).max(1);
    let page = requested_page.min(page_count - 1);
    AbilityBarPageModel {
        page,
        page_count,
        previous_enabled: page > 0,
        next_enabled: page + 1 < page_count,
        indicator: format!("{} / {}", page + 1, page_count),
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct AbilityBarTooltipModel {
    title: String,
    description: String,
}

fn non_negative_finite_seconds(value: f32) -> f32 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}

fn ability_bar_button_text(name: &str) -> String {
    let name = name.trim();
    let name = if name.is_empty() {
        "未命名技能"
    } else {
        name
    };
    let chars = name.chars().collect::<Vec<_>>();
    if chars.len() <= 6 {
        return name.to_string();
    }

    let split = chars.len().div_ceil(2);
    format!(
        "{}\n{}",
        chars[..split].iter().collect::<String>(),
        chars[split..].iter().collect::<String>()
    )
}

fn ability_bar_tooltip_model(
    item: Option<&AbilityBarItem>,
    rejection: Option<&AbilityBarRejection>,
) -> Option<AbilityBarTooltipModel> {
    item.map(|item| {
        let authored_description = item.description.trim();
        let authored_description = if authored_description.is_empty() {
            "沒有可用的技能描述。"
        } else {
            authored_description
        };
        let duration = if item.duration_total > 0.0 {
            format!("{:.1} 秒", item.duration_total)
        } else {
            "瞬發".to_string()
        };
        let status = match &item.visual_state {
            AbilityBarVisualState::Ready => "準備完成".to_string(),
            AbilityBarVisualState::Cooling { .. } => {
                format!("冷卻中，剩餘 {:.1} 秒", item.cooldown_remaining)
            }
            AbilityBarVisualState::Active { .. } => {
                format!("施放中，剩餘 {:.1} 秒", item.active_remaining)
            }
        };
        let mut lines = vec![
            format!("塔：{}", item.tower_label),
            authored_description.to_string(),
            format!("冷卻：{:.1} 秒", item.cooldown_total),
            format!("持續：{duration}"),
            format!("狀態：{status}"),
        ];
        if let Some(shortcut) = item.shortcut {
            lines.push(format!("快捷鍵：{shortcut}"));
        }
        if let Some(rejection) = rejection.filter(|rejection| rejection.key == item.key) {
            lines.push(format!("上次施放失敗：{}", rejection.reason));
        }
        AbilityBarTooltipModel {
            title: item.ability_name.clone(),
            description: lines.join("\n"),
        }
    })
}

fn reconcile_ability_bar_slots(
    bindings: &mut [Option<AbilityBarKey>],
    visible_keys: &[AbilityBarKey],
) {
    let visible = visible_keys.iter().cloned().collect::<HashSet<_>>();
    for binding in bindings.iter_mut() {
        if binding.as_ref().is_some_and(|key| !visible.contains(key)) {
            *binding = None;
        }
    }
    for key in visible_keys {
        if bindings.iter().flatten().any(|bound| bound == key) {
            continue;
        }
        if let Some(vacancy) = bindings.iter_mut().find(|binding| binding.is_none()) {
            *vacancy = Some(key.clone());
        }
    }
}

fn fallback_tower_display_name(unit_id: &str) -> String {
    unit_id
        .strip_prefix("tower_")
        .unwrap_or(unit_id)
        .split('_')
        .filter(|part| !part.is_empty())
        .map(|part| {
            let mut chars = part.chars();
            chars
                .next()
                .map(|first| first.to_uppercase().collect::<String>() + chars.as_str())
                .unwrap_or_default()
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn ability_bar_items(
    entities: &[sim_runner::EntityRenderData],
    local_player_id: u32,
    page: usize,
    elapsed_since_snapshot: f32,
) -> Vec<AbilityBarItem> {
    ability_bar_items_with_names(
        entities,
        &HashMap::new(),
        local_player_id,
        page,
        elapsed_since_snapshot,
    )
}

fn ability_bar_available_keys(
    entities: &[sim_runner::EntityRenderData],
    local_player_id: u32,
) -> Vec<AbilityBarKey> {
    entities
        .iter()
        .filter(|entity| entity.owner_player_id == Some(local_player_id) && entity.hp > 0)
        .filter_map(|entity| {
            entity
                .tower_active_ability
                .as_ref()
                .map(|ability| AbilityBarKey {
                    tower_entity_id: entity.entity_id,
                    ability_id: ability.ability_id.clone(),
                })
        })
        .collect()
}

fn ability_bar_items_with_names(
    entities: &[sim_runner::EntityRenderData],
    tower_names: &HashMap<String, String>,
    local_player_id: u32,
    page: usize,
    elapsed_since_snapshot: f32,
) -> Vec<AbilityBarItem> {
    let mut source = entities
        .iter()
        .filter(|entity| {
            matches!(entity.kind, sim_runner::EntityKind::Tower)
                && entity.owner_player_id == Some(local_player_id)
                && entity.hp > 0
                && entity.tower_active_ability.is_some()
        })
        .collect::<Vec<_>>();
    source.sort_by_key(|entity| (entity.spawn_order, entity.entity_id));

    let mut duplicate_counts = HashMap::<&str, usize>::new();
    for entity in &source {
        *duplicate_counts.entry(entity.unit_id.as_str()).or_default() += 1;
    }
    let mut duplicate_indices = HashMap::<&str, usize>::new();
    let all_items = source
        .into_iter()
        .map(|entity| {
            let ability = entity.tower_active_ability.as_ref().unwrap();
            let duplicate_key = entity.unit_id.as_str();
            let duplicate_index = duplicate_indices.entry(duplicate_key).or_default();
            *duplicate_index += 1;
            let tower_name = tower_names
                .get(&entity.unit_id)
                .cloned()
                .unwrap_or_else(|| fallback_tower_display_name(&entity.unit_id));
            let tower_label = if duplicate_counts[&duplicate_key] > 1 {
                format!("{} #{}", tower_name, duplicate_index)
            } else {
                tower_name
            };
            let elapsed_since_snapshot = non_negative_finite_seconds(elapsed_since_snapshot);
            let cooldown_total = non_negative_finite_seconds(ability.cooldown_total);
            let duration_total = non_negative_finite_seconds(ability.duration_total);
            let cooldown_remaining =
                non_negative_finite_seconds(ability.cooldown_remaining - elapsed_since_snapshot);
            let active_remaining =
                non_negative_finite_seconds(ability.active_remaining - elapsed_since_snapshot);
            AbilityBarItem {
                key: AbilityBarKey {
                    tower_entity_id: entity.entity_id,
                    ability_id: ability.ability_id.clone(),
                },
                tower_label,
                ability_name: ability.display_name.clone(),
                description: ability.description.clone(),
                cooldown_total,
                duration_total,
                cooldown_remaining,
                cooldown_text: if cooldown_remaining > 0.0 {
                    format!("{cooldown_remaining:.1}")
                } else {
                    "READY".into()
                },
                active_remaining,
                activation_serial: ability.activation_serial,
                shortcut: None,
                visual_state: ability_bar_visual_state(
                    cooldown_total,
                    cooldown_remaining,
                    active_remaining,
                ),
            }
        })
        .collect::<Vec<_>>();

    all_items
        .into_iter()
        .skip(page.saturating_mul(TOWER_ABILITY_BAR_PAGE_SIZE))
        .take(TOWER_ABILITY_BAR_PAGE_SIZE)
        .enumerate()
        .map(|(index, mut item)| {
            item.shortcut = Some(index as u8 + 1);
            item
        })
        .collect()
}

fn text_input_has_focus(ui: &UserInterface) -> bool {
    ui.nodes().pair_iter().any(|(_, node)| {
        node.query_component::<TextBox>()
            .map(|text_box| text_box.has_focus)
            .unwrap_or(false)
    })
}

fn td_pause_control_label(is_paused: bool) -> &'static str {
    if is_paused {
        "PAUSED"
    } else {
        "PAUSE"
    }
}

fn td_pause_control_opacity(is_paused: bool) -> Option<f32> {
    Some(if is_paused { 0.35 } else { 1.0 })
}

fn td_auto_start_checkbox_label(enabled: bool) -> String {
    if enabled {
        "[x] Auto".to_string()
    } else {
        "[ ] Auto".to_string()
    }
}

fn td_should_auto_start_round(
    auto_start_enabled: bool,
    auto_start_sent_for_idle_round: bool,
    is_paused: bool,
    round_is_running: bool,
    current_round: u32,
    total_rounds: u32,
) -> bool {
    auto_start_enabled
        && !auto_start_sent_for_idle_round
        && !is_paused
        && !round_is_running
        && !(total_rounds > 0 && current_round >= total_rounds)
}

fn td_upgrade_effect_text(description: &str) -> String {
    let text = description.trim();
    if text.is_empty() {
        return "效果待補".to_string();
    }
    let text = text
        .replace("，", "\n")
        .replace(", ", "\n")
        .replace(',', "\n")
        .replace('、', "\n");
    let lines = td_wrap_ui_text(&text, 20, 3);
    lines.lines().collect::<Vec<_>>().join("\n")
}

fn td_upgrade_title_text(name: &str) -> String {
    let text = name.trim();
    if text.is_empty() {
        return "?".to_string();
    }
    td_wrap_ui_text(text, 10, 2)
}

fn td_wrap_ui_text(text: &str, max_units: usize, max_lines: usize) -> String {
    let mut lines = Vec::new();
    for raw_line in text.lines() {
        let mut line = String::new();
        let mut units = 0usize;
        for ch in raw_line.trim().chars() {
            let ch_units = if ch.is_ascii() { 1 } else { 2 };
            if units + ch_units > max_units && !line.trim().is_empty() {
                // 斷行前若在 ASCII 單字中間，退回到最近的空格
                let carry = if !ch.is_whitespace() {
                    line.rfind(|c: char| c.is_whitespace())
                        .map(|idx| {
                            let word = line[idx..].trim().to_string();
                            line.truncate(idx);
                            word
                        })
                        .filter(|s| !s.is_empty())
                } else {
                    None
                };
                if !line.trim().is_empty() {
                    lines.push(line.trim().to_string());
                }
                if lines.len() >= max_lines {
                    return lines.join("\n");
                }
                line = carry.unwrap_or_default();
                units = line
                    .chars()
                    .map(|c| if c.is_ascii() { 1usize } else { 2 })
                    .sum();
                if ch.is_whitespace() {
                    continue;
                }
            }
            line.push(ch);
            units += ch_units;
        }
        if !line.trim().is_empty() {
            lines.push(line.trim().to_string());
            if lines.len() >= max_lines {
                return lines.join("\n");
            }
        }
    }
    lines.join("\n")
}

// 3D 相機視錐體中的 Z 層（相機在 z=-100 看 +Z，近=0.1 遠=1000）。
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
const Z_COMMAND_QUEUE: f32 = 1.25;
const Z_RING: f32 = 1.5;
const Z_HERO: f32 = 1.9;
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
/// 但 10k entity 仍會產生 240k line segments；stress 驗收預設關閉。
const DEBUG_COLLISION_RINGS: bool = false;
const STRESS_SAFE_BODY_BATCH_CAPACITY: u32 = 16_384;
const STRESS_SAFE_HP_BATCH_CAPACITY: u32 = STRESS_SAFE_BODY_BATCH_CAPACITY * 2;
const STRESS_SAFE_FACING_BATCH_CAPACITY: u32 = STRESS_SAFE_BODY_BATCH_CAPACITY;
const REGION_LINE_THICKNESS: f32 = 0.04;
const REGION_BLOCKER_SEGMENTS: usize = 12;
const REGION_BLOCKER_THICKNESS: f32 = 0.015;

// ---------------------------------------------------------------------------
// 網路類型

/// 新產生的 Creep 的偵錯路徑保持可見的秒數。
const PATH_VISIBLE_SECS: f32 = 5.0;

/// 階段 5.1（第 3 階段）：NetworkEntity 已死亡 — apply_event 填入了此實體
/// 來自遺留 GameEvent 串流的每個實體渲染註冊表
/// 在第 2 遍中刪除。 struct + `Game::network_entities` 欄位保留在
/// 來源，因此孤立渲染在 Game::update 中循環（插值、HP
/// 欄、名稱標籤、彈道碰撞查找）針對
/// 始終為空的 HashMap。階段 5.x 刪除了孤兒循環 + 此結構
/// （估計約 600 行更新主體清理）。
#[derive(Debug, Default)]
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
    owner_player_id: Option<u32>,
    attack_range_backend: f32,
    upgrade_levels: [u8; 3],
    tower_pops: u32,
    tower_atk: f32,
    tower_asd: f32,
    tower_target_priority: String,
    last_label_text: String,
    last_label_pos: Vector2<f32>,
    extrap_velocity: f32,
    extrap_start_pos: Vector2<f32>,
    extrap_direction: Vector2<f32>,
    extrap_elapsed: f32,
    extrap_duration: f32,
}

/// 針對 sim_runner 支援的 sprite 的每個實體 UI 標籤追蹤。
/// `last_*` 欄位控制 UI 訊息傳送以避免佇列氾濫
/// 當沒有任何可見變化時，60 fps × N 個實體。
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
    placement_radius_backend: f32,
    range_backend: f32,
    splash_radius_backend: f32,
    hit_radius_backend: f32,
    slow_factor: f32,
    slow_duration: f32,
    render_mode: String,
    base_image: String,
    barrel_image: String,
    render_visual_size_backend: f32,
    barrel_frames: Vec<String>,
    body_frames: Vec<String>,
    barrel_animation: sim_runner::TowerRenderAnimationSnapshot,
    body_animation: sim_runner::TowerRenderAnimationSnapshot,
    rotation_mode: String,
    barrel_layout: String,
    barrel_variants: Vec<sim_runner::TowerBarrelVariantSnapshot>,
    barrel_offset: sim_runner::TowerRenderPointSnapshot,
    barrel_pivot: sim_runner::TowerRenderPointSnapshot,
    muzzle_offset: sim_runner::TowerRenderPointSnapshot,
    default_angle_deg: f32,
    recoil: sim_runner::TowerRecoilSnapshot,
    attack_windup: u16,
    attack_backswing: u16,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum TowerPlacementBlockReason {
    InsufficientGold { required: i32, available: i32 },
    TooCloseToRoad,
    BlockedRegion,
    TowerOverlap,
    MissingPlacementMetadata,
}

#[derive(Clone, Copy, Debug)]
struct TowerRoadPlacementProbe {
    path_index: usize,
    segment_index: usize,
    distance_backend: f32,
    required_clearance_backend: f32,
}

impl TowerPlacementBlockReason {
    fn diagnostic_code(&self) -> &'static str {
        match self {
            Self::InsufficientGold { .. } => "insufficient_gold",
            Self::TooCloseToRoad => "too_close_to_road",
            Self::BlockedRegion => "blocked_region",
            Self::TowerOverlap => "tower_overlap",
            Self::MissingPlacementMetadata => "missing_placement_metadata",
        }
    }

    fn localized_text(&self) -> String {
        match self {
            Self::InsufficientGold { required, available } => {
                format!("金幣不足（需要 {required}，目前 {available}）")
            }
            Self::TooCloseToRoad => "離道路太近".into(),
            Self::BlockedRegion => "此區域禁止建塔".into(),
            Self::TowerOverlap => "會與既有塔重疊".into(),
            Self::MissingPlacementMetadata => "塔的放置資料不完整".into(),
        }
    }
}

fn td_template_from_snapshot(t: &sim_runner::TowerTemplateSnapshot) -> TdTemplate {
    TdTemplate {
        label: t.label.clone(),
        cost: t.cost,
        footprint_backend: t.footprint,
        placement_radius_backend: t.placement_radius,
        range_backend: t.range,
        splash_radius_backend: t.splash_radius,
        hit_radius_backend: t.hit_radius,
        slow_factor: t.slow_factor,
        slow_duration: t.slow_duration,
        render_mode: t.render_mode.clone(),
        base_image: t.base_image.clone(),
        barrel_image: t.barrel_image.clone(),
        render_visual_size_backend: t.render_visual_size,
        barrel_frames: t.barrel_frames.clone(),
        body_frames: t.body_frames.clone(),
        barrel_animation: t.barrel_animation.clone(),
        body_animation: t.body_animation.clone(),
        rotation_mode: t.rotation_mode.clone(),
        barrel_layout: t.barrel_layout.clone(),
        barrel_variants: t.barrel_variants.clone(),
        barrel_offset: t.barrel_offset.clone(),
        barrel_pivot: t.barrel_pivot.clone(),
        muzzle_offset: t.muzzle_offset.clone(),
        default_angle_deg: t.default_angle_deg,
        recoil: t.recoil.clone(),
        attack_windup: t.attack_windup,
        attack_backswing: t.attack_backswing,
    }
}

#[derive(Debug)]
struct TowerAnimationState {
    frames: Vec<String>,
    elapsed: f32,
    fps: f32,
    fire_once: bool,
    active: bool,
    last_frame_index: usize,
}

#[derive(Debug)]
struct TowerRecoilState {
    elapsed: f32,
    duration: f32,
    return_duration: f32,
    dir_rad: f32,
}

#[derive(Debug)]
struct TowerCompositeRender {
    base_node: Handle<Node>,
    barrel_node: Option<Handle<Node>>,
    body_node: Option<Handle<Node>>,
    base_material_key: String,
    barrel_material_key: Option<String>,
    body_material_key: Option<String>,
    variant_count: Option<u16>,
    last_aim_direction: f32,
    animation: Option<TowerAnimationState>,
    recoil: Option<TowerRecoilState>,
}

#[derive(Debug)]
struct HeroModelAsset {
    model: ModelResource,
    resolved_model_path: PathBuf,
    texture_path: Option<PathBuf>,
    failed_logged: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HeroAttackPlaybackPhase {
    None,
    Windup,
    Backswing,
}

#[derive(Debug, Clone)]
struct HeroPendingAttackCue {
    cue: sim_runner::AttackPhaseFx,
    action: String,
}

#[derive(Debug)]
struct HeroModelRender {
    root_node: Handle<Node>,
    animation_player: Handle<Node>,
    muzzle_node: Option<Handle<Node>>,
    last_pos: Vector2<f32>,
    render_moving: bool,
    animations_by_source: HashMap<String, Handle<Animation>>,
    action_resources_requested: HashSet<String>,
    active_action: Option<String>,
    active_animation_speed: f32,
    active_attack_seq: Option<u32>,
    last_attack_action: Option<String>,
    attack_repeat_ready: bool,
    pending_attack: Option<HeroPendingAttackCue>,
    idle_cycle_remaining: f32,
    idle_rng_state: u32,
    attack_phase: HeroAttackPlaybackPhase,
    attack_phase_remaining: f32,
    attack_backswing_remaining: f32,
    one_shot_remaining: f32,
    texture_applied: bool,
}

#[derive(Default, Debug)]
struct TdTowerShopCard {
    bg: Handle<UiNode>,
    icon: Handle<UiNode>,
    key_text: Handle<Text>,
    name_text: Handle<Text>,
    price_text: Handle<Text>,
}

/// 英雄知識面板 UI（全螢幕覆蓋面板）。
#[derive(Default, Debug)]
struct GkPanelUi {
    built: bool,
    visible_applied: bool,
    backdrop: Handle<UiNode>,
    bg: Handle<UiNode>,
    title: Handle<Text>,
    kp_text: Handle<Text>,
    close_bg: Handle<UiNode>,
    close_text: Handle<Text>,
    close_rect: UiRect,
    /// 每個知識節點的卡片：(背景, 標題文字, KP 文字, 解鎖按鈕背景, 解鎖按鈕文字)
    node_cards: Vec<(Handle<UiNode>, Handle<Text>, Handle<Text>, Handle<UiNode>, Handle<Text>)>,
    /// 每個解鎖按鈕的 hit-test rect（與 node_cards 對齊）。
    unlock_rects: Vec<UiRect>,
    /// 每個卡片對應的節點 id。
    node_ids: Vec<String>,
    /// 每個卡片的 KP 費用（與 node_ids 對齊）。
    node_kp_costs: Vec<u32>,
    /// 每��卡片的 category（與 node_ids 對齊）。
    node_categories: Vec<String>,
    /// 各 category 的 section header 背景（與 cat_names 對齊）。
    cat_header_bgs: Vec<Handle<UiNode>>,
    /// 各 category 的 section header 文字（與 cat_names 對齊）。
    cat_header_texts: Vec<Handle<Text>>,
    /// 各 category 名稱（依 JSON 出現順序，去重）。
    cat_names: Vec<String>,
    /// 右側商店面板中的進入按鈕（預建置，懶建置）。
    entry_bg: Handle<UiNode>,
    entry_text_h: Handle<Text>,
    entry_rect: UiRect,
}

#[derive(Default, Debug)]
struct TdRightShopPanel {
    bg: Handle<UiNode>,
    title_text: Handle<Text>,
    viewport_bg: Handle<UiNode>,
    scroll_track: Handle<UiNode>,
    scroll_thumb: Handle<UiNode>,
    pause_icon: Handle<UiNode>,
    pause_text: Handle<Text>,
    start_icon: Handle<UiNode>,
    panel_rect: UiRect,
    viewport_rect: UiRect,
    scroll_track_rect: UiRect,
    scroll_thumb_rect: UiRect,
    pause_rect: UiRect,
    start_rect: UiRect,
}

#[derive(Default, Debug)]
struct TdSelectedTowerPanel {
    bg: Handle<UiNode>,
    body_bg: Handle<UiNode>,
    header_strip_bg: Handle<UiNode>,
    header_strip_bottom_mask: Handle<UiNode>,
    tower_card_bg: Handle<UiNode>,
    tower_icon: Handle<UiNode>,
    summary_text: Handle<Text>,
    refund_bg: Handle<UiNode>,
    gold_text: Handle<Text>,
    sell_icon: Handle<UiNode>,
    upgrade_bgs: [Handle<UiNode>; 3],
    upgrade_icons: [Handle<UiNode>; 3],
    upgrade_pip_texts: [Handle<Text>; 3],
    upgrade_status_texts: [Handle<Text>; 3],
    upgrade_price_texts: [Handle<Text>; 3],
    left_anchor_rect: UiRect,
    right_anchor_rect: UiRect,
    panel_rect: UiRect,
    tower_card_rect: UiRect,
    refund_rect: UiRect,
    sell_rect: UiRect,
    upgrade_rects: [UiRect; 3],
    // BTD6-style elements
    header_bg: Handle<UiNode>,
    header_title: Handle<Text>,
    close_btn_bg: Handle<UiNode>,
    close_btn_text: Handle<Text>,
    close_btn_rect: UiRect,
    image_area_bg: Handle<UiNode>,
    path_left_bg: Handle<UiNode>,
    path_left_text: Handle<Text>,
    path_left_rect: UiRect,
    path_right_bg: Handle<UiNode>,
    path_right_text: Handle<Text>,
    path_right_rect: UiRect,
    path_name_label: Handle<Text>,
    level_section_bg: Handle<UiNode>,
    level_title_bar_bg: Handle<UiNode>,
    level_badge_bg: Handle<UiNode>,
    level_num_text: Handle<Text>,
    level_label_text: Handle<Text>,
    flavor_text_node: Handle<Text>,
    upgrade_section_bg: Handle<UiNode>,
    unlock_title_bar_bg: Handle<UiNode>,
    unlock_label_text: Handle<Text>,
    upgrade_green_bg: Handle<UiNode>,
    upgrade_green_price: Handle<Text>,
    upgrade_green_rect: UiRect,
    upgrade_path_btn_bg: Handle<UiNode>,
    upgrade_path_btn_text: Handle<Text>,
    next_effect_text: Handle<Text>,
    sell_section_bg: Handle<UiNode>,
    sell_top_mask: Handle<UiNode>,
    sell_coin_icon: Handle<UiNode>,
    sell_coin_text: Handle<Text>,
    sell_red_bg: Handle<UiNode>,
    sell_red_text: Handle<Text>,
    sell_red_rect: UiRect,
    selected_path: u8,
    btd6_tower_icon: Handle<UiNode>,
    // 新版三行升級佈局
    upgrade_row_bgs: [Handle<UiNode>; 3],
    upgrade_name_texts: [Handle<Text>; 3],
    pops_text: Handle<Text>,
    // i info button + overlay
    info_btn_bg: Handle<UiNode>,
    info_btn_text: Handle<Text>,
    info_btn_rect: UiRect,
    info_overlay_bg: Handle<UiNode>,
    info_stat_texts: [Handle<Text>; 4],
    // 三列升級 tooltip（show_info 時同時顯示，樣式同 hover tooltip）
    info_row_bgs: [Handle<UiNode>; 3],
    info_row_titles: [Handle<Text>; 3],
    info_row_descs: [Handle<Text>; 3],
    info_row_descs2: [Handle<Text>; 3],
    show_info: bool,
}

fn load_texture_from_candidate_paths(candidate_paths: Vec<String>) -> Option<TextureResource> {
    use fyrox::asset::untyped::ResourceKind;
    use fyrox::core::uuid::Uuid;

    let bytes = candidate_paths.iter().find_map(|p| std::fs::read(p).ok())?;
    let opts = TextureImportOptions::default()
        .with_compression(CompressionOptions::NoCompression)
        .with_minification_filter(TextureMinificationFilter::LinearMipMapLinear);
    TextureResource::load_from_memory(Uuid::new_v4(), ResourceKind::Embedded, &bytes, opts).ok()
}

fn load_texture_from_rel_path(rel_path: &str) -> Option<TextureResource> {
    let mut candidate_paths: Vec<String> = vec![
        rel_path.to_string(),
        format!("omfx/{}", rel_path),
        format!("../{}", rel_path),
    ];
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            candidate_paths.push(exe_dir.join(rel_path).to_string_lossy().into_owned());
        }
    }
    load_texture_from_candidate_paths(candidate_paths)
}

fn load_sound_from_path(rel_path: &str) -> Option<fyrox::scene::sound::SoundBufferResource> {
    // CWD when run via run.bat is omoba/omoba/ (project root).
    // Sound assets live in omfx/game/data/ relative to that root.
    let mut candidate_paths: Vec<String> = vec![
        rel_path.to_string(),               // CWD == game/ (dev server mode)
        format!("omfx/game/{}", rel_path),  // CWD == omoba/omoba/ (run.bat)
        format!("game/{}", rel_path),       // CWD == omfx/
        format!("omfx/{}", rel_path),       // legacy omfx/data/ layout
        format!("../{}", rel_path),         // CWD == omfx/executor/
        format!("../../game/{}", rel_path), // CWD == omfx/target/debug/
    ];
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            candidate_paths.push(exe_dir.join(rel_path).to_string_lossy().into_owned());
            // exe in target/debug/ → go up to omfx/ then into game/
            if let Some(omfx_dir) = exe_dir.parent().and_then(|p| p.parent()) {
                candidate_paths.push(
                    omfx_dir
                        .join("game")
                        .join(rel_path)
                        .to_string_lossy()
                        .into_owned(),
                );
            }
        }
    }
    for path in &candidate_paths {
        if let Ok(bytes) = std::fs::read(path) {
            let data_source = DataSource::from_memory(bytes);
            if let Ok(buf) = fyrox::scene::sound::SoundBufferResource::new_generic(data_source) {
                log::info!("Sound loaded from: {}", path);
                return Some(buf);
            }
        }
    }
    log::warn!(
        "Sound not found ({}), tried: {:?}",
        rel_path,
        candidate_paths
    );
    None
}

fn load_td_ui_texture(asset_name: &str) -> Option<TextureResource> {
    let script_rel = format!("scripts/base_content/assets/td_ui/{}", asset_name);
    let frontend_rel = format!("data/td_ui/{}", asset_name);
    let mut candidate_paths: Vec<String> = vec![
        script_rel.clone(),
        format!("../{}", script_rel),
        format!("../../{}", script_rel),
    ];
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            for ancestor in exe_dir.ancestors().take(6) {
                candidate_paths.push(ancestor.join(&script_rel).to_string_lossy().into_owned());
            }
        }
    }

    // 相容舊開發路徑；正式替換位置仍是 scripts/base_content/assets/td_ui。
    candidate_paths.extend([
        frontend_rel.clone(),
        format!("omfx/{}", frontend_rel),
        format!("../{}", frontend_rel),
    ]);
    load_texture_from_candidate_paths(candidate_paths)
}

fn load_pregame_ui_texture(asset_name: &str) -> Option<TextureResource> {
    let asset_name = asset_name.trim().trim_start_matches(['/', '\\']);
    if asset_name.is_empty() {
        return None;
    }
    let asset_path = Path::new(asset_name);
    if asset_path.is_absolute() {
        return load_texture_from_candidate_paths(vec![asset_name.to_string()]);
    }

    let normalized = asset_name.replace('\\', "/");
    let script_rel = if normalized.starts_with("scripts/base_content/") {
        normalized
    } else if normalized.starts_with("assets/pregame_ui/") {
        format!("scripts/base_content/{}", normalized)
    } else {
        format!("scripts/base_content/assets/pregame_ui/{}", normalized)
    };
    let mut candidate_paths: Vec<String> = vec![
        script_rel.clone(),
        format!("../{}", script_rel),
        format!("../../{}", script_rel),
    ];
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            for ancestor in exe_dir.ancestors().take(6) {
                candidate_paths.push(ancestor.join(&script_rel).to_string_lossy().into_owned());
            }
        }
    }
    load_texture_from_candidate_paths(candidate_paths)
}

fn normalize_tower_asset_key(asset_path: &str) -> String {
    asset_path
        .trim()
        .trim_start_matches('/')
        .strip_prefix("scripts/base_content/")
        .unwrap_or_else(|| asset_path.trim().trim_start_matches('/'))
        .replace('\\', "/")
}

fn load_tower_texture(asset_path: &str) -> Option<TextureResource> {
    let asset_key = normalize_tower_asset_key(asset_path);
    let script_rel = if asset_key.is_empty() {
        return None;
    } else if asset_key.starts_with("scripts/base_content/") {
        asset_key
    } else {
        format!("scripts/base_content/{}", asset_key)
    };
    let mut candidate_paths: Vec<String> = vec![
        script_rel.clone(),
        format!("../{}", script_rel),
        format!("../../{}", script_rel),
    ];
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            for ancestor in exe_dir.ancestors().take(6) {
                candidate_paths.push(ancestor.join(&script_rel).to_string_lossy().into_owned());
            }
        }
    }
    load_texture_from_candidate_paths(candidate_paths)
}

fn normalize_scripts_lua_data_asset_path(asset_path: &str) -> Option<String> {
    let path = asset_path.trim().trim_start_matches('/').replace('\\', "/");
    if path.is_empty() || path.contains("..") {
        return None;
    }
    Some(
        path.strip_prefix("scripts/lua_data/")
            .unwrap_or(path.as_str())
            .to_string(),
    )
}

fn scripts_lua_data_candidate_paths(asset_path: &str) -> Vec<PathBuf> {
    let Some(asset_key) = normalize_scripts_lua_data_asset_path(asset_path) else {
        return Vec::new();
    };
    let script_rel = PathBuf::from("scripts/lua_data").join(&asset_key);
    let mut candidates = vec![
        script_rel.clone(),
        PathBuf::from("..").join(&script_rel),
        PathBuf::from("..").join("..").join(&script_rel),
    ];
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            for ancestor in exe_dir.ancestors().take(8) {
                candidates.push(ancestor.join(&script_rel));
            }
        }
    }
    candidates
}

fn resolve_scripts_lua_data_asset_path(asset_path: &str) -> Option<PathBuf> {
    scripts_lua_data_candidate_paths(asset_path)
        .into_iter()
        .find(|path| path.is_file())
}

fn load_scripts_lua_data_texture(asset_path: &str) -> Option<TextureResource> {
    let candidates = scripts_lua_data_candidate_paths(asset_path)
        .into_iter()
        .map(|path| path.to_string_lossy().into_owned())
        .collect();
    load_texture_from_candidate_paths(candidates)
}

fn texture_material_3d(texture: TextureResource) -> MaterialResource {
    let mut material = Material::standard();
    material.bind("diffuseTexture", Some(texture));
    MaterialResource::new_embedded(material)
}

fn texture_material(texture: TextureResource) -> MaterialResource {
    let mut material = Material::standard_2d();
    material.bind("diffuseTexture", Some(texture));
    MaterialResource::new_embedded(material)
}

fn tower_visual_size(tpl: &TdTemplate) -> f32 {
    tpl.render_visual_size_backend * WORLD_SCALE
}

fn tower_placement_radius_render(tpl: &TdTemplate) -> f32 {
    tpl.placement_radius_backend * WORLD_SCALE
}

fn tower_render_offset(point: &sim_runner::TowerRenderPointSnapshot, scale: f32) -> Vector2<f32> {
    Vector2::new(
        point.x * WORLD_SCALE * scale,
        -point.y * WORLD_SCALE * scale,
    )
}

fn rotate_vec2(v: Vector2<f32>, angle: f32) -> Vector2<f32> {
    let (s, c) = angle.sin_cos();
    Vector2::new(v.x * c - v.y * s, v.x * s + v.y * c)
}

fn tower_render_angle_from_facing(facing_rad: f32, default_angle_deg: f32) -> f32 {
    std::f32::consts::FRAC_PI_2 - facing_rad + default_angle_deg.to_radians()
}

fn hero_model_rotation(
    facing_rad: f32,
    render: &sim_runner::HeroRenderSnapshot,
) -> UnitQuaternion<f32> {
    let yaw = (std::f32::consts::PI - facing_rad + render.yaw_offset_deg.to_radians())
        .rem_euclid(std::f32::consts::TAU);
    let pitch = render.pitch_offset_deg.to_radians();
    let roll = render.roll_offset_deg.to_radians();
    UnitQuaternion::from_axis_angle(&Vector3::z_axis(), yaw)
        * UnitQuaternion::from_axis_angle(&Vector3::x_axis(), pitch)
        * UnitQuaternion::from_axis_angle(&Vector3::y_axis(), roll)
}

fn find_descendant_by_name(scene: &Scene, root: Handle<Node>, name: &str) -> Option<Handle<Node>> {
    if name.trim().is_empty() || !scene.graph.is_valid_handle(root) {
        return None;
    }
    let mut stack = vec![root];
    while let Some(handle) = stack.pop() {
        if !scene.graph.is_valid_handle(handle) {
            continue;
        }
        let node = &scene.graph[handle];
        if node.name() == name {
            return Some(handle);
        }
        stack.extend(node.children().iter().copied());
    }
    None
}

fn find_descendant_animation_player(scene: &Scene, root: Handle<Node>) -> Option<Handle<Node>> {
    if !scene.graph.is_valid_handle(root) {
        return None;
    }
    let mut stack = vec![root];
    while let Some(handle) = stack.pop() {
        if !scene.graph.is_valid_handle(handle) {
            continue;
        }
        let node = &scene.graph[handle];
        if node
            .cast::<fyrox::scene::animation::AnimationPlayer>()
            .is_some()
        {
            return Some(handle);
        }
        stack.extend(node.children().iter().copied());
    }
    None
}

fn descendant_animation_players(scene: &Scene, root: Handle<Node>) -> Vec<Handle<Node>> {
    if !scene.graph.is_valid_handle(root) {
        return Vec::new();
    }
    let mut players = Vec::new();
    let mut stack = vec![root];
    while let Some(handle) = stack.pop() {
        if !scene.graph.is_valid_handle(handle) {
            continue;
        }
        let node = &scene.graph[handle];
        if node
            .cast::<fyrox::scene::animation::AnimationPlayer>()
            .is_some()
        {
            players.push(handle);
        }
        stack.extend(node.children().iter().copied());
    }
    players
}

fn disable_animation_player(scene: &mut Scene, player: Handle<Node>) {
    if !scene.graph.is_valid_handle(player) {
        return;
    }
    if let Some(player) = scene.graph[player].cast_mut::<fyrox::scene::animation::AnimationPlayer>()
    {
        for animation in player.animations_mut().get_value_mut_silent().iter_mut() {
            animation.set_enabled(false);
        }
    }
}

fn disable_other_animation_players(
    scene: &mut Scene,
    root: Handle<Node>,
    active_player: Handle<Node>,
) {
    for player in descendant_animation_players(scene, root) {
        if player != active_player {
            disable_animation_player(scene, player);
        }
    }
}

fn hero_animation_playback_speed(base_speed: f32, is_paused: bool) -> f32 {
    if is_paused {
        0.0
    } else {
        base_speed
    }
}

fn tower_animation_dt(dt: f32, is_paused: bool) -> f32 {
    if is_paused {
        0.0
    } else {
        dt.max(0.0)
    }
}

fn select_retargeted_animation_by_duration(
    scene: &Scene,
    player: Handle<Node>,
    handles: &[Handle<Animation>],
    expected_duration_secs: f32,
) -> Option<Handle<Animation>> {
    let player = scene.graph[player].cast::<fyrox::scene::animation::AnimationPlayer>()?;
    let animations = player.animations().get_value_ref();
    handles.iter().copied().min_by(|a, b| {
        let a_delta = animations
            .try_get(*a)
            .map(|animation| (animation.length() - expected_duration_secs).abs())
            .unwrap_or(f32::INFINITY);
        let b_delta = animations
            .try_get(*b)
            .map(|animation| (animation.length() - expected_duration_secs).abs())
            .unwrap_or(f32::INFINITY);
        a_delta
            .partial_cmp(&b_delta)
            .unwrap_or(std::cmp::Ordering::Equal)
    })
}

fn is_hero_idle_action(action: &str) -> bool {
    action == "idle" || action.starts_with("idle_")
}

fn next_idle_rng_state(state: u32) -> u32 {
    let mut x = if state == 0 { 0x9E37_79B9 } else { state };
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    x
}

fn tower_render_dir_from_world_rad(dir_rad: f32) -> Vector2<f32> {
    Vector2::new(-dir_rad.cos(), dir_rad.sin())
}

fn safe_projectile_trail_dir(dir: Vector2<f32>) -> Vector2<f32> {
    let len_sq = dir.x * dir.x + dir.y * dir.y;
    if !len_sq.is_finite() || len_sq <= 1.0e-8 {
        return Vector2::new(1.0, 0.0);
    }
    dir / len_sq.sqrt()
}

fn initial_projectile_trail_dir(
    spawn_pos: Vector2<f32>,
    projectile_pos: Vector2<f32>,
    fallback_dir: Vector2<f32>,
) -> Vector2<f32> {
    let delta = projectile_pos - spawn_pos;
    let len_sq = delta.x * delta.x + delta.y * delta.y;
    if len_sq.is_finite() && len_sq > 1.0e-8 {
        delta / len_sq.sqrt()
    } else {
        safe_projectile_trail_dir(fallback_dir)
    }
}

fn projectile_trail_quad(
    spawn_pos: Vector2<f32>,
    projectile_pos: Vector2<f32>,
    dir: Vector2<f32>,
) -> (Vector2<f32>, f32, f32) {
    let displacement = projectile_pos - spawn_pos;
    let trail_len = (displacement.x * displacement.x + displacement.y * displacement.y).sqrt();
    let max_trail = 0.6_f32;
    let len = trail_len.min(max_trail).max(0.05);
    let dir = safe_projectile_trail_dir(dir);
    let tail = projectile_pos - dir * len;
    let mid = (projectile_pos + tail) * 0.5;
    let rotation = dir.y.atan2(dir.x);
    (mid, len, rotation)
}

/// 甜點子彈種類：「甜點戰爭」每座塔的子彈都有專屬糖果甜點特效。
///
/// 分類優先序：先看「子彈自身 unit_id」（本機 sim 目前子彈實體沒有 ScriptUnitTag，
/// unit_id 為空字串；保留這條讓未來若替子彈打標籤能自動細分 tack/tack_blade 等升級變體），
/// 對不到再退用「發射者（塔／英雄）unit_id」——每座塔對應一種甜點主題，天然涵蓋升級。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DessertBullet {
    CandyBall,  // 糖球砲手 dart：亮彩小糖球
    Toothpick,  // 刺蝟射手 tack：牙籤糖針（白／焦糖細長）
    TackBlade,  // 刺蝟射手升級 tack_blade：旋轉銀色小刀片
    Macaron,    // 馬卡龍砲車 bomb：粉彩馬卡龍 + 引信火花
    BombFrag,   // 馬卡龍砲車碎片 bomb_frag：焦糖小碎餅
    Snowflake,  // 冰晶泰迪 ice：淡藍雪花結晶 + 冰霜點
    Icicle,     // 冰晶泰迪升級 icicle：尖長冰刺
    Churro,     // 吉拿棒迫擊砲 arty：金黃吉拿棒砲彈（體積大、慢飛微旋轉）
    Banana,     // 香蕉回力鏢 boomerang：旋轉香蕉彎月 + 淡殘影
    Shuriken,   // 香蕉回力鏢升級 shuriken：旋轉銀灰星形
    HeroPellet, // 英雄 saika_shot：白熱槍口鉛彈
    Fallback,   // 未知：沿用原橘黃拖尾
}

/// 依子彈自身 unit_id（優先）與發射者 unit_id／kind（退路）判定甜點種類。
/// 比對用 `contains`／`strip_prefix` 容忍前綴後綴差異；全對不到回 `Fallback`。
fn classify_dessert_bullet(
    proj_unit_id: &str,
    owner_unit_id: Option<&str>,
    owner_kind: Option<sim_runner::EntityKind>,
) -> DessertBullet {
    // 1) 子彈自身 unit_id（本機 sim 為空，但未來標籤化後這條會自動生效並細分升級變體）。
    let pid = proj_unit_id.trim();
    let pid = pid.strip_prefix("proj_").unwrap_or(pid); // 容忍 proj_ 前綴
    match pid {
        "dart" | "spike_opult" => return DessertBullet::CandyBall,
        "tack" => return DessertBullet::Toothpick,
        "tack_blade" => return DessertBullet::TackBlade,
        "bomb" => return DessertBullet::Macaron,
        "bomb_frag" => return DessertBullet::BombFrag,
        "ice" => return DessertBullet::Snowflake,
        "icicle" => return DessertBullet::Icicle,
        "boomerang" => return DessertBullet::Banana,
        "shuriken" => return DessertBullet::Shuriken,
        "saika_shot" => return DessertBullet::HeroPellet,
        _ => {}
    }
    // 2) 退用發射者（本機 sim 的實際可靠來源：tower_dart / tower_tack / ... / hero_saika_*）。
    if let Some(owner) = owner_unit_id {
        if owner.contains("saika") {
            return DessertBullet::HeroPellet;
        }
        if owner.contains("dart") {
            return DessertBullet::CandyBall;
        }
        if owner.contains("tack") {
            return DessertBullet::Toothpick;
        }
        if owner.contains("bomb") {
            return DessertBullet::Macaron;
        }
        if owner.contains("ice") {
            return DessertBullet::Snowflake;
        }
        if owner.contains("arty") {
            return DessertBullet::Churro;
        }
        if owner.contains("boomerang") {
            return DessertBullet::Banana;
        }
    }
    // 3) 英雄發射但名字沒對到 → 當英雄鉛彈；其餘 fallback。
    if matches!(owner_kind, Some(sim_runner::EntityKind::Hero)) {
        return DessertBullet::HeroPellet;
    }
    DessertBullet::Fallback
}

/// 一片要寫進 sprite batch 的甜點 quad（座標已在 render 空間，沿用 write_quad → camera）。
#[derive(Debug, Clone, Copy)]
struct DessertQuad {
    center: Vector2<f32>,
    size: Vector2<f32>,
    color: [u8; 4],
    rotation: f32,
}

/// 一顆甜點子彈的畫面：主體必畫，`accent` 選畫（需要額外 slot 的少數甜點）。
#[derive(Debug, Clone, Copy)]
struct DessertRender {
    body: DessertQuad,
    accent: Option<DessertQuad>,
}

/// 依甜點種類產生要畫的 quad。沿用既有拖尾幾何（spawn_pos→pos），
/// 只調 color/size/rotation 或改成「以彈頭為中心隨時間旋轉」。純便宜運算，
/// 無字串配置、無堆配置，適合 stress 場景每 tick 上千顆呼叫。
fn dessert_bullet_render(
    kind: DessertBullet,
    eid: u32,
    tick: u32,
    spawn_pos: Vector2<f32>,
    pos: Vector2<f32>,
    trail_dir: Vector2<f32>,
) -> DessertRender {
    let (_mid, len, trail_rot) = projectile_trail_quad(spawn_pos, pos, trail_dir);
    let dir = safe_projectile_trail_dir(trail_dir);
    // 旋轉相位：tick 取模避免 f32 精度流失；每顆用 eid 錯開，避免同批同步旋轉。
    let phase = (tick % 100_000) as f32;
    let eid_offset = (eid as f32) * 0.6;

    // 拖尾型主體：沿飛行方向、緊貼彈頭；len 依甜點上限裁切。
    let make_trail = |color: [u8; 4], thickness: f32, len_cap: f32| -> DessertQuad {
        let l = len.min(len_cap).max(0.05);
        DessertQuad {
            center: pos - dir * (l * 0.5),
            size: Vector2::new(l, thickness),
            color,
            rotation: trail_rot,
        }
    };
    // 旋轉型主體：以彈頭為中心、隨時間自轉（刀片／回力鏢／星形／吉拿棒）。
    let make_spin = |color: [u8; 4], w: f32, h: f32, speed: f32| -> DessertQuad {
        DessertQuad {
            center: pos,
            size: Vector2::new(w, h),
            color,
            rotation: phase * speed + eid_offset,
        }
    };

    // 2026-07-10 放大加粗：子彈又小又快時各型別糊成一團，Doris 回報「看不出差別」。
    // 全面放大尺寸、拉高色彩飽和與 alpha、誇張形狀輪廓（圓更圓、針更長、旋轉更大更明顯），
    // 讓七種一眼可辨。核心手法：round 型 thickness≈len（讀成圓球）、needle 型細而長、
    // spin 型大塊 + 明顯自轉；再給關鍵型別加亮色 accent 核心增加辨識度。
    match kind {
        DessertBullet::CandyBall => {
            // 圓糖球：紅／粉／橘依 eid 輪替，大而圓、亮，尾端加白色高光核心。
            let color = match eid % 3 {
                0 => [255, 60, 80, 255],   // 紅糖
                1 => [255, 130, 200, 255], // 粉糖
                _ => [255, 165, 45, 255],  // 橘糖
            };
            DessertRender {
                body: make_trail(color, 0.24, 0.26),
                accent: Some(DessertQuad {
                    center: pos,
                    size: Vector2::new(0.1, 0.1),
                    color: [255, 255, 245, 235],
                    rotation: 0.0,
                }),
            }
        }
        DessertBullet::Toothpick => DessertRender {
            // 細長牙籤糖針：奶白、又細又長的明顯長條、沿飛行方向。
            body: make_trail([255, 245, 210, 255], 0.08, 0.78),
            accent: None,
        },
        DessertBullet::TackBlade => DessertRender {
            // 旋轉銀色刀片：大方形銀白、快速自轉。
            body: make_spin([225, 230, 240, 255], 0.26, 0.26, 0.20),
            accent: None,
        },
        DessertBullet::Macaron => DessertRender {
            // 圓馬卡龍：飽和粉、又大又圓；尾端一顆明顯亮黃引信火花。
            body: make_trail([255, 120, 180, 255], 0.34, 0.36),
            accent: Some(DessertQuad {
                center: pos - dir * 0.26,
                size: Vector2::new(0.14, 0.14),
                color: [255, 235, 90, 255],
                rotation: 0.0,
            }),
        },
        DessertBullet::BombFrag => DessertRender {
            // 焦糖小碎餅：比糖球暗、方塊感。
            body: make_trail([205, 130, 55, 250], 0.16, 0.18),
            accent: None,
        },
        DessertBullet::Snowflake => {
            // 冰晶：藍色「旋轉十字雪花」——兩片交叉 quad 組成結晶星、慢轉。刻意跟馬卡龍的
            // 圓粉餅在形狀（十字 vs 圓）＋顏色（藍 vs 粉）＋是否旋轉三方面全部拉開。
            let rot = phase * 0.10 + eid_offset;
            DessertRender {
                body: DessertQuad {
                    center: pos,
                    size: Vector2::new(0.34, 0.1),
                    color: [120, 205, 255, 255],
                    rotation: rot,
                },
                accent: Some(DessertQuad {
                    // 交叉的另一片（+90°）＋更亮的冰白，組成十字冰晶。
                    center: pos,
                    size: Vector2::new(0.34, 0.1),
                    color: [225, 245, 255, 255],
                    rotation: rot + std::f32::consts::FRAC_PI_2,
                }),
            }
        }
        DessertBullet::Icicle => DessertRender {
            // 尖長冰刺：藍白、極細極長、沿方向（跟雪花的圓形成強對比）。
            body: make_trail([190, 235, 255, 255], 0.09, 0.9),
            accent: None,
        },
        DessertBullet::Churro => DessertRender {
            // 金黃吉拿棒砲彈：體積最大、粗長、慢飛明顯旋轉。
            body: make_spin([245, 170, 45, 255], 0.4, 0.2, 0.07),
            accent: None,
        },
        DessertBullet::Banana => DessertRender {
            // 香蕉黃彎月：大、邊飛邊旋轉 + 淡殘影（相位略落後）。
            body: make_spin([255, 220, 40, 255], 0.4, 0.15, 0.14),
            accent: Some(DessertQuad {
                center: pos - dir * 0.12,
                size: Vector2::new(0.34, 0.12),
                color: [255, 220, 40, 110],
                rotation: phase * 0.14 + eid_offset - 0.6,
            }),
        },
        DessertBullet::Shuriken => DessertRender {
            // 旋轉銀灰星形：大方塊近似、極快自轉（比刀片更快、更冷灰）。
            body: make_spin([170, 180, 195, 255], 0.24, 0.24, 0.30),
            accent: None,
        },
        DessertBullet::HeroPellet => DessertRender {
            // 白熱槍口鉛彈：亮白、短快、帶亮尾。
            body: make_trail([255, 253, 235, 255], 0.13, 0.32),
            accent: None,
        },
        DessertBullet::Fallback => DessertRender {
            // 未知：沿用原本的暖色橘黃拖尾（tracer round）。
            body: make_trail([255, 180, 60, 235], 0.1, 0.6),
            accent: None,
        },
    }
}

/// 命中濺射外觀：依甜點種類回傳 (顏色, 半徑 render 單位, 形狀風格)。
/// 半徑校準參考：Bomb 爆炸圈 = splash_radius(200) × WORLD_SCALE(0.01) = 2.0 render 單位（畫面明顯可見）。
/// 先前設 0.3~0.5 比爆炸圈小 4~6 倍、在 ~28 單位高的畫面裡小到看不出形狀，故放大到 1.0~2.2。
/// splash 系（馬卡龍/吉拿棒/冰晶）較大，單體系較小。形狀由 HitStyle 決定，讓各塔一眼可辨。
fn dessert_hit_spec(kind: DessertBullet) -> ([u8; 4], f32, HitStyle) {
    match kind {
        DessertBullet::CandyBall => ([255, 150, 190, 255], 1.1, HitStyle::Sparkle), // 糖球=星芒
        DessertBullet::Toothpick => ([250, 240, 210, 255], 1.0, HitStyle::Sparkle), // 糖針=星芒
        DessertBullet::TackBlade => ([220, 225, 235, 255], 1.3, HitStyle::Slash),   // 刀片=斬擊
        DessertBullet::Macaron => ([255, 110, 180, 255], 2.2, HitStyle::CreamSplat), // 馬卡龍=奶油圈
        DessertBullet::BombFrag => ([205, 130, 55, 255], 1.4, HitStyle::Crumbs),    // 碎片=碎屑
        DessertBullet::Snowflake => ([150, 220, 255, 255], 2.0, HitStyle::Frost),   // 冰晶=結晶爆刺
        DessertBullet::Icicle => ([200, 240, 255, 255], 2.0, HitStyle::Frost),      // 冰刺=結晶爆刺
        DessertBullet::Churro => ([245, 175, 50, 255], 2.2, HitStyle::Crumbs),      // 吉拿棒=金碎屑
        DessertBullet::Banana => ([255, 220, 50, 255], 1.4, HitStyle::CreamSplat),  // 香蕉=黃奶油圈
        DessertBullet::Shuriken => ([185, 195, 210, 255], 1.3, HitStyle::Slash),    // 手裡劍=斬擊
        DessertBullet::HeroPellet => ([255, 250, 235, 255], 1.1, HitStyle::Sparkle), // 英雄=白星芒
        DessertBullet::Fallback => ([255, 190, 90, 255], 1.1, HitStyle::Burst),     // fallback=環
    }
}

fn build_tower_rect_node(
    scene: &mut Scene,
    material: Option<MaterialResource>,
    center: Vector2<f32>,
    size: f32,
    z: f32,
    fallback_color: Color,
) -> Handle<Node> {
    let mut builder = RectangleBuilder::new(
        BaseBuilder::new()
            .with_frustum_culling(false)
            .with_local_transform(
                TransformBuilder::new()
                    .with_local_position(Vector3::new(center.x, center.y, z))
                    .with_local_scale(Vector3::new(size, size, f32::EPSILON))
                    .build(),
            ),
    )
    .with_color(if material.is_some() {
        Color::WHITE
    } else {
        fallback_color
    });
    if let Some(material) = material {
        builder = builder.with_material(material);
    }
    builder.build(&mut scene.graph).transmute()
}

fn set_tower_rect_material(
    scene: &mut Scene,
    node: Handle<Node>,
    material: Option<MaterialResource>,
    fallback_color: Color,
) {
    if node.is_none() {
        return;
    }
    if let Some(rect) = scene.graph[node].cast_mut::<Rectangle>() {
        if let Some(material) = material {
            rect.material_mut().set_value_and_mark_modified(material);
            rect.set_color(Color::WHITE);
        } else {
            rect.material_mut()
                .set_value_and_mark_modified(MaterialResource::new_embedded(
                    Material::standard_2d(),
                ));
            rect.set_color(fallback_color);
        }
    }
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

/// Creep 死亡特效：敵方 creep 被打死時，在其位置爆散出「餅乾屑 + 糖果碎粒」，貼合糖果餅乾主題
/// 與餅乾碎裂音效（`SfxKind::CookieCrunch`）。刻意跟橘色 Bomb 爆炸圈（`ActiveExplosion`）區隔。
/// 視覺：一圈餅乾屑（棕/焦糖色短線）向外飛散、混幾顆糖果亮色碎粒（粉/薄荷/檸檬黃），約 0.45s 淡出。
/// 每隻 creep 用 `seed`（= entity_id）擾動角度/速度/長度，讓每次爆散略有不同、不呆板。
/// 與 `ActiveExplosion` 同樣走 `scene.drawing_context.add_line(...)` 批次線段（single draw call），
/// 不 per-frame 增刪 scene graph node。座標慣例一致：`pos` 存未翻轉的 render 座標，畫線時 x 取負。
#[derive(Debug)]
struct ActiveDeathFx {
    pos: Vector2<f32>, // render 座標（未翻轉，後端 × WORLD_SCALE）
    max_radius: f32,   // render 單位（碎屑飛散的最大距離）
    duration: f32,
    elapsed: f32,
    seed: u32, // 每隻 creep 的爆散擾動種子（= entity_id）
}

/// 子彈命中濺射：子彈打到目標（被移除）時，在命中點炸開該塔主題色的濺射（擴張環 + 放射線），
/// 停留 ~0.25s。快速子彈本身看不清、一閃而過，但命中濺射停在命中點（怪身上，玩家正在看的地方）、
/// 看得清也最有打擊感——這是塔防區分各塔的正解。同走 `drawing_context.add_line` 批次線段，
/// `pos` 存未翻轉 render 座標（畫線時 x 取負）。
/// 命中濺射的形狀風格——靠不同輪廓（不只顏色）讓各塔一眼可辨。
#[derive(Debug, Clone, Copy, PartialEq)]
enum HitStyle {
    Frost,      // 冰晶：尖銳結晶爆刺（很多細長冰刺，無環）
    CreamSplat, // 馬卡龍：柔軟雙層奶油圈（圓潤疊圈 + 短鈍刺）
    Crumbs,     // 吉拿棒/碎片：四散不規則塊狀碎屑
    Sparkle,    // 糖球/英雄：閃亮十字星芒（小巧）
    Slash,      // 手裡劍/刀片：鋒利 X 斬擊
    Burst,      // fallback：環 + 均勻放射線
}

#[derive(Debug)]
struct ActiveHitFx {
    pos: Vector2<f32>,
    color: [u8; 4],
    max_radius: f32,
    duration: f32,
    elapsed: f32,
    style: HitStyle,
    seed: u32,
}

/// 客戶端射彈模擬。
///
/// 後端僅發送帶有 `target_id` + `flight_time_ms` 的單一 C 事件；
/// 子彈的位置在每幀作為追蹤 lerp 進行本地計算
/// 從“start_pos”到目標實體的目前客戶端位置。
/// P7 layered prediction entry（per projectile id）。追蹤「server 已經宣告
/// 但 server 還沒送 hp_snapshot 反映」這段視窗內，client 想本地視覺上扣多少血。
///
/// 生命週期：
/// PC 到達→插入（已套用=假）
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

/// 這消除了產生子彈的每跳動網路往返延遲
/// 視覺上追蹤蠕動。
#[derive(Debug)]
struct ClientProjectile {
    node: Handle<Node>,
    target_id: u32,
    start_pos: Vector2<f32>,
    last_target_pos: Vector2<f32>,
    elapsed: f32,
    flight_time: f32,
    // 當子彈視覺擊中時，預測傷害應用於客戶端；
    // 心跳 HP 快照每 2 秒協調一次漂移。
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

/// 心跳資訊（用於UI顯示）
#[derive(Default, Debug)]
struct HeartbeatInfo {
    tick: u64,
    game_time: f64,
    entity_count: u64,
    hero_count: u64,
    creep_count: u64,
}

/// 連線狀態
#[derive(Default, Clone, PartialEq, Debug)]
enum ConnectionStatus {
    #[default]
    Disconnected,
    Connecting,
    Connected,
    Failed(String),
}

// ---------------------------------------------------------------------------
// 遊戲插件
// ---------------------------------------------------------------------------

// ---------- Frame profile (omfx 端 per-frame timing 拆解，類比 omb 的 tick_profile) ----------

#[derive(Default, Debug)]
struct FrameProfile {
    frame_count: u64,
    events_ns: u128,
    lockstep_ns: u128,
    snapshot_ns: u128,
    render_bridge_ns: u128,
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
    paced_frame_count: u64,
    stale_snapshot_frame_count: u64,
    frame_interval_ms_total: f64,
    max_frame_interval_ms: f64,
    frame_interval_ms_window: Vec<f64>,
    render_target_tps: u32,
    sim_tps_total: f64,
    sim_diag_samples: u64,
    latest_sim_tick: u32,
    sim_queue_len_total: u64,
    sim_max_queue_len: usize,
    sim_waits: u64,
    sim_blocking_receives: u64,
    sim_backlog_receives: u64,
    last_fps: usize,
    /// 最近的每幀快照（覆蓋每次呼叫“record_render_stats”）。
    /// 由 HUD 狀態文字使用 — 視窗平均值會重設每個 WINDOW 幀，以便
    /// 瞬時樣本提供更流暢的即時讀數。
    last_draw_calls: usize,
    last_triangles: usize,
}

impl FrameProfile {
    const WINDOW: u64 = 120;

    fn finish_frame(&mut self) {
        self.frame_count += 1;
        if self.frame_count % Self::WINDOW == 0 {
            self.emit_log();
            self.reset_window();
        }
    }

    fn emit_log(&self) {
        let w = Self::WINDOW as f64;
        let frame_samples = self.frame_interval_ms_window.len().max(1) as f64;
        let total_ms = self.total_ns as f64 / w / 1_000_000.0;
        let max_fps = if total_ms > 0.0 {
            (1000.0 / total_ms) as u32
        } else {
            0
        };
        let (p50_ms, p95_ms, p99_ms, one_pct_low_fps) =
            frame_time_summary(&self.frame_interval_ms_window);
        let avg_fps = if self.frame_interval_ms_total > 0.0 {
            frame_samples * 1000.0 / self.frame_interval_ms_total
        } else {
            0.0
        };
        let pure_render_avg = self.pure_render_ms_total / w;
        let capped_render_avg = self.capped_render_ms_total / w;
        let cap_or_present_wait_avg = (capped_render_avg - pure_render_avg).max(0.0);
        log::debug!(
            "omfx_frame window={} avg(ms) lockstep={:.2} snapshot={:.2} render_bridge={:.2} interp={:.2} visual={:.2} proj={:.2} cam={:.2} ui={:.2} total={:.2} (max_fps={}, events_per_frame={:.0}, creeps={:.0}, projectiles={:.0})",
            Self::WINDOW,
            self.lockstep_ns as f64 / w / 1_000_000.0,
            self.snapshot_ns as f64 / w / 1_000_000.0,
            self.render_bridge_ns as f64 / w / 1_000_000.0,
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
        log::debug!(
            "omfx_frame_slo window={} target_fps={} avg_fps={:.2} one_pct_low_fps={:.2} frame_ms p50={:.2} p95={:.2} p99={:.2} max={:.2} plugin_avg={:.2} pure_avg={:.2} capped_avg={:.2} cap_or_present_wait_avg={:.2} sim_tps={:.2} latest_sim_tick={} sim_queue_avg={:.2} sim_queue_max={} sim_waits={} sim_blocking_receives={} sim_backlog_receives={}",
            Self::WINDOW,
            self.render_target_tps.max(1),
            avg_fps,
            one_pct_low_fps,
            p50_ms,
            p95_ms,
            p99_ms,
            self.max_frame_interval_ms,
            total_ms,
            pure_render_avg,
            capped_render_avg,
            cap_or_present_wait_avg,
            if self.sim_diag_samples > 0 {
                self.sim_tps_total / self.sim_diag_samples as f64
            } else {
                0.0
            },
            self.latest_sim_tick,
            if self.sim_diag_samples > 0 {
                self.sim_queue_len_total as f64 / self.sim_diag_samples as f64
            } else {
                0.0
            },
            self.sim_max_queue_len,
            self.sim_waits,
            self.sim_blocking_receives,
            self.sim_backlog_receives,
        );
        log::debug!(
            "omfx_render window={} target_fps={} target_ms={:.2} avg(ms) pure={:.2} capped={:.2} fps={} paced_frames={} stale_snapshot_frames={} draw_calls={:.0} triangles={:.0}",
            Self::WINDOW,
            self.render_target_tps.max(1),
            1000.0 / self.render_target_tps.max(1) as f32,
            pure_render_avg,
            capped_render_avg,
            self.last_fps,
            self.paced_frame_count,
            self.stale_snapshot_frame_count,
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

    fn record_frame_interval(&mut self, interval: Option<std::time::Duration>) {
        let Some(interval) = interval else {
            return;
        };
        let ms = interval.as_secs_f64() * 1000.0;
        self.frame_interval_ms_total += ms;
        self.max_frame_interval_ms = self.max_frame_interval_ms.max(ms);
        self.frame_interval_ms_window.push(ms);
    }

    fn record_render_pacing(&mut self, snapshot_reused: bool, target_tps: u32) {
        self.paced_frame_count += 1;
        self.render_target_tps = target_tps.max(1);
        if snapshot_reused {
            self.stale_snapshot_frame_count += 1;
        }
    }

    fn record_sim_diagnostics(&mut self, diagnostics: &sim_runner::SimRunnerDiagnostics) {
        self.sim_tps_total += diagnostics.sim_tps as f64;
        self.sim_diag_samples += 1;
        self.latest_sim_tick = diagnostics.latest_tick;
        self.sim_queue_len_total += diagnostics.queue_len as u64;
        self.sim_max_queue_len = self.sim_max_queue_len.max(diagnostics.max_queue_len);
        self.sim_waits += diagnostics.waits as u64;
        self.sim_blocking_receives += diagnostics.blocking_receives as u64;
        self.sim_backlog_receives += diagnostics.backlog_receives as u64;
    }

    fn reset_window(&mut self) {
        self.events_ns = 0;
        self.lockstep_ns = 0;
        self.snapshot_ns = 0;
        self.render_bridge_ns = 0;
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
        self.paced_frame_count = 0;
        self.stale_snapshot_frame_count = 0;
        self.frame_interval_ms_total = 0.0;
        self.max_frame_interval_ms = 0.0;
        self.frame_interval_ms_window.clear();
        self.render_target_tps = LOCKSTEP_TPS;
        self.sim_tps_total = 0.0;
        self.sim_diag_samples = 0;
        self.latest_sim_tick = 0;
        self.sim_queue_len_total = 0;
        self.sim_max_queue_len = 0;
        self.sim_waits = 0;
        self.sim_blocking_receives = 0;
        self.sim_backlog_receives = 0;
        // last_fps 只是覆蓋每一幀，無需重置
    }
}

fn frame_time_summary(samples: &[f64]) -> (f64, f64, f64, f64) {
    if samples.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let percentile = |p: f64| -> f64 {
        let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    };
    let slow_count = ((sorted.len() as f64) * 0.01).ceil().max(1.0) as usize;
    let slow_start = sorted.len().saturating_sub(slow_count);
    let slow_avg_ms = sorted[slow_start..].iter().sum::<f64>() / slow_count as f64;
    let one_pct_low_fps = if slow_avg_ms > 0.0 {
        1000.0 / slow_avg_ms
    } else {
        0.0
    };
    (
        percentile(0.50),
        percentile(0.95),
        percentile(0.99),
        one_pct_low_fps,
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SfxKind {
    ButtonClick,
    TowerPlace,
    CookieCrunch,
}

#[derive(Default, Visit, Reflect, Debug)]
#[reflect(non_cloneable)]
pub struct Game {
    scene: Handle<Scene>,
    camera: Handle<Node>,
    #[visit(skip)]
    #[reflect(hidden)]
    pregame_runtime: pregame::PregameRuntime,
    #[visit(skip)]
    #[reflect(hidden)]
    backend_session: Option<backend_session::BackendSession>,
    #[visit(skip)]
    #[reflect(hidden)]
    ui_pregame: PregameUi,
    #[visit(skip)]
    #[reflect(hidden)]
    ui_in_game_return: InGameReturnUi,
    #[visit(skip)]
    #[reflect(hidden)]
    return_to_title_button_rect: UiRect,
    #[visit(skip)]
    #[reflect(hidden)]
    ui_settings: SettingsPanel,
    #[visit(skip)]
    #[reflect(hidden)]
    ui_profile: ProfilePanel,
    #[visit(skip)]
    #[reflect(hidden)]
    settings_sfx_volume: f32,
    #[visit(skip)]
    #[reflect(hidden)]
    settings_music_volume: f32,
    #[visit(skip)]
    #[reflect(hidden)]
    bgm_handle: Handle<Node>,
    #[visit(skip)]
    #[reflect(hidden)]
    sfx_button_click: Option<fyrox::scene::sound::SoundBufferResource>,
    #[visit(skip)]
    #[reflect(hidden)]
    sfx_tower_place: Option<fyrox::scene::sound::SoundBufferResource>,
    #[visit(skip)]
    #[reflect(hidden)]
    sfx_cookie_crunch: Option<fyrox::scene::sound::SoundBufferResource>,
    #[visit(skip)]
    #[reflect(hidden)]
    sfx_one_shots: Vec<Handle<Node>>,
    #[visit(skip)]
    #[reflect(hidden)]
    sfx_pending: Vec<SfxKind>,
    #[visit(skip)]
    #[reflect(hidden)]
    entity_kind_cache: std::collections::HashMap<u32, sim_runner::EntityKind>,
    #[visit(skip)]
    #[reflect(hidden)]
    hotkeys: hotkeys::HotkeyConfig,
    #[visit(skip)]
    #[reflect(hidden)]
    hotkey_panel_visible: bool,
    #[visit(skip)]
    #[reflect(hidden)]
    hotkey_rebinding: Option<String>,
    #[visit(skip)]
    #[reflect(hidden)]
    hotkey_panel_ui: HotkeyPanelUi,
    /// 設定頁「熱鍵」按鈕的 hit-test 矩形（非設定頁時為 default=不可命中）。
    #[visit(skip)]
    #[reflect(hidden)]
    settings_hotkey_btn_rect: UiRect,
    /// 設定頁「螢幕尺寸」badge 的 hit-test 矩形（點擊開關下拉選單）。
    #[visit(skip)]
    #[reflect(hidden)]
    settings_resolution_rect: UiRect,
    /// 待套用的顯示模式（下一 frame 在 graphics context 可用時套用）。
    #[visit(skip)]
    #[reflect(hidden)]
    pending_display_mode: Option<DisplayModeRequest>,
    #[visit(skip)]
    #[reflect(hidden)]
    resolution_dropdown_open: bool,
    #[visit(skip)]
    #[reflect(hidden)]
    resolution_dropdown_ui: ResolutionDropdownUi,
    /// 目前是否為全螢幕（套用時更新，用於下拉選單高亮）。
    #[visit(skip)]
    #[reflect(hidden)]
    display_fullscreen: bool,
    /// 進入全螢幕前的視窗尺寸（Esc 退出全螢幕時還原用；(0,0)=未記錄）。
    #[visit(skip)]
    #[reflect(hidden)]
    last_windowed_size: (u32, u32),
    #[visit(skip)]
    #[reflect(hidden)]
    settings_dragging_sfx: bool,
    #[visit(skip)]
    #[reflect(hidden)]
    settings_dragging_music: bool,
    #[visit(skip)]
    #[reflect(hidden)]
    settings_dragging_speed: bool,
    #[visit(skip)]
    #[reflect(hidden)]
    settings_speed_value: f32,
    #[visit(skip)]
    #[reflect(hidden)]
    pregame_button_rects: Vec<(UiRect, pregame::PregameAction)>,
    #[visit(skip)]
    #[reflect(hidden)]
    mouse_world_pos: Vector2<f32>,
    #[visit(skip)]
    #[reflect(hidden)]
    window_size: Vector2<f32>,

    /// 共享 sprite GPU 資源（單一四邊形 + 9 個材質）。
    /// 在第一幀上延遲初始化；重用於所有實體 sprite 網格體。
    #[visit(skip)]
    #[reflect(hidden)]
    sprite_resources: Option<sprite_resources::SharedSpriteResources>,

    /// 所有 entity body sprite 共用的 batched mesh — 1 個 mesh / 1 draw call 容納
    /// 數千個 entity，取代之前每 entity 1 個 Mesh 的爆量 draw call 浪費。
    /// Capacity 16k quad，支援 10k stress units + projectile/hero 餘裕。
    #[visit(skip)]
    #[reflect(hidden)]
    body_batch: Option<sprite_resources::BatchedSpriteMesh>,

    /// HP bar 黑底 + 綠條 共用 batched mesh（per-vertex color，bg/fg 兩個 slot
    /// 一個 entity）。Capacity 32k = 16k entity × 2 (bg + fg)。
    #[visit(skip)]
    #[reflect(hidden)]
    hp_batch: Option<sprite_resources::BatchedSpriteMesh>,

    /// Facing arrow 共用 batched mesh（with rotation）。Capacity 16k。
    #[visit(skip)]
    #[reflect(hidden)]
    facing_batch: Option<sprite_resources::BatchedSpriteMesh>,

    // - - 網路 - -
    // 階段 5.1：刪除了舊版「network: Option<NetworkBridge>」欄位。
    /// Lockstep 用戶端（KCP 標籤 0x10-0x16）。透過驅動 sim_runner
    /// TickBatch / StateHash 在單獨的後台執行緒上。
    #[visit(skip)]
    #[reflect(hidden)]
    lockstep_handle: Option<lockstep_client::LockstepClientHandle>,
    /// 階段 4.3：觀察到最近的「LockstepEvent::TickBatch.tick」。
    /// 用於計算 input submit target tick。低延遲 client lookahead 搭配
    /// server late-input retarget 避免偶發晚到直接掉 input。
    /// `#[導出（預設）]`;更新了 TickBatch 手臂中的每一幀
    /// `遊戲::更新`。
    #[visit(skip)]
    #[reflect(hidden)]
    current_sim_tick: u32,
    #[visit(skip)]
    #[reflect(hidden)]
    current_sim_tick_observed_at: Option<Instant>,
    /// Server-authoritative cadence announced by GameStart.
    #[visit(skip)]
    #[reflect(hidden)]
    server_step_fps: u32,
    /// Client-configured lockstep player id, known before connecting.
    #[visit(skip)]
    #[reflect(hidden)]
    local_player_id: u32,
    #[visit(skip)]
    #[reflect(hidden)]
    pending_inputs: HashMap<u32, PendingInput>,
    #[visit(skip)]
    #[reflect(hidden)]
    pending_inputs_evict_at: Option<Instant>,
    #[visit(skip)]
    #[reflect(hidden)]
    pending_inputs_evicted: u64,
    #[visit(skip)]
    #[reflect(hidden)]
    pending_inputs_stale: u64,
    #[visit(skip)]
    #[reflect(hidden)]
    input_latency_meter: InputLatencyMeter,
    /// 階段 3.2 sim_runner 工作執行緒（執行完整的 omb ECS 排程器）
    /// 後台線程）。落在 `on_deinit` 上，所以頻道
    /// 斷開連線讓工作人員退出。階段 3.3 將連接輸入饋電
    /// 來自「lockstep_handle」；直到那時工人就會阻塞
    /// `master_seed_rx.recv()` 並且從不勾選。
    #[visit(skip)]
    #[reflect(hidden)]
    sim_runner_handle: Option<sim_runner::SimRunnerHandle>,
    /// 最近一次 frontend 已套用的 Lua content generation/hash。
    /// 變更時清除 Lua-derived UI/asset caches，讓下一份 snapshot 重新 seed。
    #[visit(skip)]
    #[reflect(hidden)]
    sim_lua_content_generation: u64,
    #[visit(skip)]
    #[reflect(hidden)]
    sim_lua_content_hash: String,
    #[visit(skip)]
    #[reflect(hidden)]
    sim_dev_lua_reload_error: Option<String>,
    #[visit(skip)]
    #[reflect(hidden)]
    sim_speed_last_tick: u32,
    #[visit(skip)]
    #[reflect(hidden)]
    sim_speed_last_at: Option<Instant>,
    #[visit(skip)]
    #[reflect(hidden)]
    sim_speed_tps: f32,
    /// 階段 3.4 渲染橋：每個畫面讀取 `SimWorldSnapshot` 並
    /// （第 4 階段）為每個實體產生/更新/消失 Fyrox sprite。
    /// 目前是記錄實體渲染資料的存根。始終分配
    /// （便宜的預設值）因此“update”中的每個畫面檢查只是一種方法
    /// 調用，而不是“Option”解包。
    #[visit(skip)]
    #[reflect(hidden)]
    render_bridge: render_bridge::RenderBridge,
    #[visit(skip)]
    #[reflect(hidden)]
    session_render_reset_pending: bool,
    #[visit(skip)]
    #[reflect(hidden)]
    connection_status: ConnectionStatus,
    // 階段 5.1：刪除了 `event_buffer: Option<EventBuffer>` 欄位。
    // EventBuffer 驅動了舊版 GameEvent 重新排序/重播管道。
    // `network_entities` 欄位暫時保留 - 孤立渲染循環
    // Game::update 仍然迭代它（總是為空，因為 apply_event 是
    // 消失了）；階段 5.x 清理了循環 + 該欄位。
    #[visit(skip)]
    #[reflect(hidden)]
    network_entities: HashMap<u32, NetworkEntity>,
    #[visit(skip)]
    #[reflect(hidden)]
    latest_entities: Vec<sim_runner::EntityRenderData>,
    /// BlockedRegion 線框 scene node（每個 region 一組 polygon outline segments）。
    #[visit(skip)]
    #[reflect(hidden)]
    region_line_nodes: Vec<Handle<Node>>,
    /// Region blocker 近似圓 scene node（debug 視覺化；與 region 線框一起在 init 時畫）。
    #[visit(skip)]
    #[reflect(hidden)]
    region_blocker_nodes: Vec<Handle<Node>>,
    /// TD 模式氣球路線的 scene node（每條 path 一組線段）。
    #[visit(skip)]
    #[reflect(hidden)]
    td_path_nodes: Vec<Handle<Node>>,
    /// TD 模式右側塔按鈕的 UI Text node（動態 Vec：N 個塔來自 td_template_order 長度）
    #[visit(skip)]
    #[reflect(hidden)]
    ui_td_tower_buttons: Vec<Handle<Text>>,
    /// BTD-style 右側買塔格子（圖示 + 快捷鍵 + 價格）。
    #[visit(skip)]
    #[reflect(hidden)]
    ui_td_tower_cards: Vec<TdTowerShopCard>,
    /// 塔按鈕的 hit-test rects（x, y, w, h）—— 每 frame 依 window_size 更新
    #[visit(skip)]
    #[reflect(hidden)]
    td_tower_button_rects: Vec<(f32, f32, f32, f32)>,
    /// 右側 TD shop viewport 的 scroll offset（1920x1080 reference units）。
    #[visit(skip)]
    #[reflect(hidden)]
    td_shop_scroll_offset: f32,
    #[visit(skip)]
    #[reflect(hidden)]
    td_shop_max_scroll: f32,
    #[visit(skip)]
    #[reflect(hidden)]
    td_shop_scroll_dragging: bool,
    #[visit(skip)]
    #[reflect(hidden)]
    td_shop_scroll_drag_start_y: f32,
    #[visit(skip)]
    #[reflect(hidden)]
    td_shop_scroll_drag_start_offset: f32,
    /// 右側常駐 shop/control panel：買塔 + Start/Pause placeholder。
    #[visit(skip)]
    #[reflect(hidden)]
    ui_td_right_panel: TdRightShopPanel,
    /// 目前玩家選中的塔 unit_id（例如 "tower_dart"）；None 表示未選
    #[visit(skip)]
    #[reflect(hidden)]
    selected_tower_kind: Option<String>,
    /// Start Round 按鈕 UI Text node。
    #[visit(skip)]
    #[reflect(hidden)]
    ui_start_round_button: Handle<Text>,
    /// Auto-start checkbox text above the Start button.
    #[visit(skip)]
    #[reflect(hidden)]
    ui_td_auto_start_checkbox_text: Handle<Text>,
    /// Start Round 按鈕 hit-test rect（每 frame 依 window_size 更新）。
    #[visit(skip)]
    #[reflect(hidden)]
    start_round_button_rect: (f32, f32, f32, f32),
    /// Auto-start checkbox hit-test rect.
    #[visit(skip)]
    #[reflect(hidden)]
    auto_start_checkbox_rect: (f32, f32, f32, f32),
    /// Pause placeholder hit-test rect；目前只攔截點擊，不送 gameplay input。
    #[visit(skip)]
    #[reflect(hidden)]
    pause_button_rect: (f32, f32, f32, f32),
    /// TD 當前已完成的波數（1-based 概念；後端推送 `game/round` 時更新）。
    /// 0 表示還沒開始第一波。
    #[visit(skip)]
    #[reflect(hidden)]
    current_round: u32,
    /// TD 總波數（後端推送 `game/round` 時更新）。
    #[visit(skip)]
    #[reflect(hidden)]
    total_rounds: u32,
    /// TD 本波是否正在跑（true = 按鈕變灰；false = 按鈕可按）。
    #[visit(skip)]
    #[reflect(hidden)]
    round_is_running: bool,
    /// Lockstep-authoritative gameplay pause state.
    #[visit(skip)]
    #[reflect(hidden)]
    is_game_paused: bool,
    /// Lockstep-authoritative gameplay speed multiplier.
    #[visit(skip)]
    #[reflect(hidden)]
    game_speed_multiplier: u32,
    /// Local UI preference: automatically start the next TD wave when idle.
    #[visit(skip)]
    #[reflect(hidden)]
    td_auto_start_enabled: bool,
    /// Debounce auto-start so one idle wave receives one StartRound input.
    #[visit(skip)]
    #[reflect(hidden)]
    td_auto_start_sent_for_idle_round: bool,
    /// 是否為 TD 模式：由首次收到 hero.stats 有 lives>0 時設 true。
    /// 影響相機（固定不跟隨英雄）、zoom（拉遠讓整張路徑可見）。
    #[visit(skip)]
    #[reflect(hidden)]
    is_td_mode: bool,
    /// 是否已經針對 TD 模式調整過相機 ortho（避免每 tick 重設）。
    #[visit(skip)]
    #[reflect(hidden)]
    td_camera_configured: bool,
    /// 玩家點選中的已蓋塔 entity id（右側顯示 sell 面板、地圖上畫射程圈）；None = 未選取
    #[visit(skip)]
    #[reflect(hidden)]
    selected_tower_entity: Option<u32>,
    /// 選中塔右側面板：塔名+等級 文字
    #[visit(skip)]
    #[reflect(hidden)]
    ui_td_sell_name_text: Handle<Text>,
    /// 選中塔右側面板：Sell 按鈕文字
    #[visit(skip)]
    #[reflect(hidden)]
    ui_td_sell_button_text: Handle<Text>,
    /// Sell 按鈕 hit-test rect（每 frame 依 window_size 更新；塔未選時放螢幕外）
    #[visit(skip)]
    #[reflect(hidden)]
    td_sell_button_rect: (f32, f32, f32, f32),
    /// 選中塔 target priority 控制 hit-test rect；塔未選時放螢幕外
    #[visit(skip)]
    #[reflect(hidden)]
    td_target_priority_button_rect: (f32, f32, f32, f32),
    /// 選中塔右側面板：3 條路線升級按鈕文字
    #[visit(skip)]
    #[reflect(hidden)]
    ui_td_upgrade_buttons: [Handle<Text>; 3],
    /// BTD-style 左側 selected tower panel 補充圖示與文字 handles。
    #[visit(skip)]
    #[reflect(hidden)]
    ui_td_selected_panel: TdSelectedTowerPanel,
    /// 3 條路線升級按鈕 hit-test rect；塔未選時放螢幕外
    #[visit(skip)]
    #[reflect(hidden)]
    td_upgrade_button_rects: [(f32, f32, f32, f32); 3],
    /// 進行中的爆炸特效（Bomb 塔命中時 spawn）
    #[visit(skip)]
    #[reflect(hidden)]
    active_explosions: Vec<ActiveExplosion>,
    /// 進行中的 creep 死亡特效（敵方 creep 被打死時 spawn）
    #[visit(skip)]
    #[reflect(hidden)]
    active_death_fx: Vec<ActiveDeathFx>,
    /// Creep 死亡特效用的每幀快取：entity_id → 最後 render 座標（未翻轉，= world_to_render(e)）。
    /// `removed_entity_ids` 只給 id 不給位置，所以在 `update_sim_batches` 的 sprite 迴圈裡每幀快取
    /// 每隻 creep 的位置；該 id 出現在 removed_entity_ids（死亡）時才拿得到座標 spawn 特效。
    /// 刻意跟死亡音效放在同一個 sprite 迴圈/移除迴圈，讓特效與音效由結構保證一起觸發。
    #[visit(skip)]
    #[reflect(hidden)]
    creep_death_cache: std::collections::HashMap<u32, Vector2<f32>>,
    /// 階段 4.2：我們擁有「snapshot.explosions」的最高 sim 刻度
    /// 已排入「active_explosions」。渲染幀可以讀取
    /// 在SIM卡發布之前多次使用相同的快照
    /// 下一個 - 如果沒有這種重複資料刪除，我們就會產生重複的環。
    #[visit(skip)]
    #[reflect(hidden)]
    sim_last_explosion_tick: Option<u32>,
    /// TD 路徑 check_points（render 座標）— 供 placement 預覽計算是否壓到路
    #[visit(skip)]
    #[reflect(hidden)]
    td_paths_render: Vec<Vec<Vector2<f32>>>,
    /// TD 禁止通行多邊形（render 座標）— 供 placement 預覽計算是否壓到 region
    #[visit(skip)]
    #[reflect(hidden)]
    td_regions_render: Vec<Vec<Vector2<f32>>>,
    /// 後端送來的 TD 塔 template 快取（unit_id → TdTemplate）
    #[visit(skip)]
    #[reflect(hidden)]
    td_templates: HashMap<String, TdTemplate>,
    /// Template 的顯示順序（= DLL `units()` 註冊順序），供按鈕排版用
    #[visit(skip)]
    #[reflect(hidden)]
    td_template_order: Vec<String>,
    /// Tower 升級定義快取：（tower_kind、路徑、等級）→（顯示名稱、效果描述、成本）。
    /// 從「snapshot.tower_upgrades」（Arc 延遲建置模式）播種一次。
    /// 由「出售」按鈕用來計算退款（基礎*0.85 + Σ 升級*0.75）
    /// 並透過升級按鈕顯示下一級名稱、效果與成本。
    #[visit(skip)]
    #[reflect(hidden)]
    td_upgrade_defs: HashMap<(String, u8, u8), (String, String, i32)>,
    /// TD UI texture cache：key 是 TD UI asset 檔名；`None` 也 cache，避免缺圖每幀讀檔。
    #[visit(skip)]
    #[reflect(hidden)]
    td_ui_texture_cache: HashMap<String, Option<TextureResource>>,
    /// Pregame UI texture cache：key 是 catalog image 路徑；`None` 也 cache，避免缺圖每幀讀檔。
    #[visit(skip)]
    #[reflect(hidden)]
    pregame_ui_texture_cache: HashMap<String, Option<TextureResource>>,
    /// Tower combat texture/material cache：key 是 scripts/base_content 相對路徑。
    #[visit(skip)]
    #[reflect(hidden)]
    tower_texture_cache: HashMap<String, Option<TextureResource>>,
    #[visit(skip)]
    #[reflect(hidden)]
    tower_material_cache: HashMap<String, Option<MaterialResource>>,
    #[visit(skip)]
    #[reflect(hidden)]
    tower_composites: HashMap<u32, TowerCompositeRender>,
    /// 每 render frame 累積的塔台動畫 dt（在 sim tick guard 之外累積，
    /// tick 變化時一次性取用，確保動畫以正確速度播放而非 ~50% 速度）。
    /// f32 默認為 0.0，不需要額外初始化。
    #[visit(skip)]
    #[reflect(hidden)]
    tower_anim_dt_acc: f32,
    /// 已處理過的 render-only tower fire cue keys。sim snapshot 會短暫保留 FX，
    /// 所以必須以 cue identity 去重，不能只看 snapshot tick。
    #[visit(skip)]
    #[reflect(hidden)]
    sim_seen_tower_fire_fx: HashSet<TowerFireFxKey>,
    /// 已處理過的 attack phase cue keys，避免保留 window 內重複啟動同一段動畫。
    #[visit(skip)]
    #[reflect(hidden)]
    sim_seen_attack_phase_fx: HashSet<AttackPhaseFxKey>,
    /// 已處理過的 attack cancel cue keys，避免 snapshot retention window 重複停止同一段動畫。
    #[visit(skip)]
    #[reflect(hidden)]
    sim_seen_attack_cancel_fx: HashSet<AttackCancelFxKey>,
    #[visit(skip)]
    #[reflect(hidden)]
    hero_model_assets: HashMap<String, HeroModelAsset>,
    #[visit(skip)]
    #[reflect(hidden)]
    hero_action_assets: HashMap<String, HeroModelAsset>,
    #[visit(skip)]
    #[reflect(hidden)]
    hero_asset_failures_logged: HashSet<String>,
    #[visit(skip)]
    #[reflect(hidden)]
    hero_model_nodes: HashMap<u32, HeroModelRender>,
    #[visit(skip)]
    #[reflect(hidden)]
    client_projectiles: HashMap<u32, ClientProjectile>,
    /// P7分層預測：key = 彈丸id（伺服器`e.id()`）。
    /// 跟 client_projectiles 同 id；前者是視覺軌跡，這個是傷害預測 ledger。
    /// 視覺命中後 ClientProjectile 移除但 PendingPredDmg 留著，等 heartbeat
    /// 的 in_flight_projectiles 或 D event 真結算才移除。
    #[visit(skip)]
    #[reflect(hidden)]
    pending_pred_dmg: HashMap<u32, PendingPredDmg>,
    #[visit(skip)]
    #[reflect(hidden)]
    heartbeat: HeartbeatInfo,

    /// Per-frame profile（每 60 frame 輸出一行 omfx_frame_profile log）。
    #[visit(skip)]
    #[reflect(hidden)]
    frame_profile: FrameProfile,

    /// Render pacing follows shared lockstep cadence; executor owns the actual frame cap.
    #[visit(skip)]
    #[reflect(hidden)]
    render_pacing_last_frame_at: Option<Instant>,
    #[visit(skip)]
    #[reflect(hidden)]
    render_pacing_last_snapshot_tick: Option<u32>,
    #[visit(skip)]
    #[reflect(hidden)]
    sim_batches_last_snapshot_tick: Option<u32>,

    #[visit(skip)]
    #[reflect(hidden)]
    pending_label_deletions: Vec<Handle<Text>>,

    /// 由「render_bridge」（sim_runner 支援）呈現的實體的 UI 文字標籤。
    /// 由“entity_id”鍵入。在實體的第一次渲染時創建，每次更新
    /// 框架，當實體從 sim 快照中退出時刪除。
    #[visit(skip)]
    #[reflect(hidden)]
    sim_entity_labels: HashMap<u32, SimEntityLabel>,

    /// sim_runner 支援的實體的批次網格槽所有權，由
    /// `實體_id`。 body_batch + hp_batch 插槽首先分配
    /// 當實體從快照中掉落時看到並釋放。這是
    /// 節省繪製呼叫的路徑：1000 個小兵 + 1000 個塔 ≈ 總共 2 個繪製
    /// （每批一個），與每個實體一個節點，即每個四邊形一次繪製。
    #[visit(skip)]
    #[reflect(hidden)]
    sim_entity_slots: HashMap<u32, render_bridge::SimEntitySlots>,

    /// 子彈拖尾起點：第一次看到 projectile entity 時記下當前 render pos，
    /// 之後每 frame 從此點畫一條暖色拖尾到當前 pos。removed_entity_ids 觸發時
    /// 一起清除。
    #[visit(skip)]
    #[reflect(hidden)]
    projectile_spawn_pos: HashMap<u32, Vector2<f32>>,

    /// 子彈拖尾方向：第一次看到 projectile 時鎖定，避免飛行中因目標移動或
    /// snapshot 追蹤更新而旋轉。
    #[visit(skip)]
    #[reflect(hidden)]
    projectile_trail_dir: HashMap<u32, Vector2<f32>>,

    /// 子彈甜點分類：第一次看到 projectile 時依發射者（塔／英雄）鎖定，之後每 tick
    /// 沿用，省去逐顆逐 tick 字串比對（stress 場景可能上千顆）。removed 時一起清除。
    #[visit(skip)]
    #[reflect(hidden)]
    projectile_dessert: HashMap<u32, DessertBullet>,
    /// 每幀快取子彈的最後 render 座標（未翻轉，供命中濺射取用——子彈被移除時已不在 entities，
    /// 只能靠上一幀快取的位置在命中點炸開濺射）。
    #[visit(skip)]
    #[reflect(hidden)]
    projectile_last_pos: HashMap<u32, Vector2<f32>>,
    /// 進行中的子彈命中濺射特效。
    #[visit(skip)]
    #[reflect(hidden)]
    active_hit_fx: Vec<ActiveHitFx>,

    /// 第一幀的掛鐘時間戳；所使用的
    /// `OMFX_AUTO_START_AFTER_SEC` / `OMFX_AUTO_EXIT_AFTER_SEC` 煙霧循環
    /// 因此，單一「cargo run」可以重現「開始-回合-然後-死亡」的場景
    /// 無需手動點擊。直到第一次 update() 勾選為止。
    #[visit(skip)]
    #[reflect(hidden)]
    auto_clock_start: Option<std::time::Instant>,
    /// 一旦發出自動開始回合輸入就鎖定，因此
    /// 調度程式只看到一個 StartRound（後續主機端讀取
    /// 會警告“回合已在運行”）。
    #[visit(skip)]
    #[reflect(hidden)]
    auto_start_sent: bool,
    #[visit(skip)]
    #[reflect(hidden)]
    auto_noop_next_at_s: Option<f32>,

    // --- UI ---
    #[visit(skip)]
    #[reflect(hidden)]
    ui_status_text: Handle<Text>,
    #[visit(skip)]
    #[reflect(hidden)]
    ui_hud_text: Handle<Text>,
    /// TD 上方 icon HUD：HP / lives / gold。TD 模式用圖示+數字取代文字條。
    #[visit(skip)]
    #[reflect(hidden)]
    ui_td_top_hud_icons: [Handle<UiNode>; 3],
    #[visit(skip)]
    #[reflect(hidden)]
    ui_td_top_hud_texts: [Handle<Text>; 3],
    /// TD 上方目前波數文字。
    #[visit(skip)]
    #[reflect(hidden)]
    ui_td_wave_text: Handle<Text>,
    /// 左下角英雄屬性面板（多行：name/title/Lv/XP/SP/三圍/HP/Gold + 4 技能等級）
    #[visit(skip)]
    #[reflect(hidden)]
    ui_hero_stats_panel: Handle<Text>,
    #[visit(skip)]
    #[reflect(hidden)]
    ui_ability_icons: [Handle<UiNode>; 4],
    #[visit(skip)]
    #[reflect(hidden)]
    ui_ability_level_text: [Handle<Text>; 4],
    /// 冷卻中央大數字
    #[visit(skip)]
    #[reflect(hidden)]
    ui_ability_cd_text: [Handle<Text>; 4],
    /// 快捷鍵 cap [W] [E] [R] [T]
    #[visit(skip)]
    #[reflect(hidden)]
    ui_ability_key_text: [Handle<Text>; 4],
    /// 技能圖示上方三角升級按鈕
    #[visit(skip)]
    #[reflect(hidden)]
    ui_ability_upgrade_buttons: [Handle<Text>; 4],
    /// Six persistent bottom-center tower ability slots; contents are diffed by stable key.
    #[visit(skip)]
    #[reflect(hidden)]
    ui_tower_ability_bar_slots: Vec<Handle<Text>>,
    #[visit(skip)]
    #[reflect(hidden)]
    ui_tower_ability_bar_backgrounds: Vec<Handle<UiNode>>,
    #[visit(skip)]
    #[reflect(hidden)]
    ui_tower_ability_bar_cooldown_overlays: Vec<Handle<UiNode>>,
    #[visit(skip)]
    #[reflect(hidden)]
    ui_tower_ability_bar_active_overlays: Vec<Handle<UiNode>>,
    #[visit(skip)]
    #[reflect(hidden)]
    tower_ability_bar_rects: Vec<UiRect>,
    #[visit(skip)]
    #[reflect(hidden)]
    tower_ability_bar_cached_text: Vec<String>,
    #[visit(skip)]
    #[reflect(hidden)]
    tower_ability_bar_cached_visual: Vec<Option<AbilityBarRenderedState>>,
    #[visit(skip)]
    #[reflect(hidden)]
    tower_ability_bar_slot_bindings: Vec<Option<AbilityBarKey>>,
    #[visit(skip)]
    #[reflect(hidden)]
    tower_ability_bar_interaction: AbilityBarInteractionState,
    #[visit(skip)]
    #[reflect(hidden)]
    tower_ability_bar_snapshot_at: Option<Instant>,
    #[visit(skip)]
    #[reflect(hidden)]
    tower_ability_bar_items: Vec<AbilityBarItem>,
    #[visit(skip)]
    #[reflect(hidden)]
    tower_ability_bar_tower_names: HashMap<String, String>,
    #[visit(skip)]
    #[reflect(hidden)]
    tower_ability_bar_rejection: AbilityBarRejectionState,
    #[visit(skip)]
    #[reflect(hidden)]
    ui_tower_ability_bar_previous: Handle<Text>,
    #[visit(skip)]
    #[reflect(hidden)]
    ui_tower_ability_bar_next: Handle<Text>,
    #[visit(skip)]
    #[reflect(hidden)]
    ui_tower_ability_bar_page_indicator: Handle<Text>,
    #[visit(skip)]
    #[reflect(hidden)]
    tower_ability_bar_previous_rect: UiRect,
    #[visit(skip)]
    #[reflect(hidden)]
    tower_ability_bar_next_rect: UiRect,
    #[visit(skip)]
    #[reflect(hidden)]
    ui_tower_ability_tooltip_bg: Handle<UiNode>,
    #[visit(skip)]
    #[reflect(hidden)]
    ui_tower_ability_tooltip_title: Handle<Text>,
    #[visit(skip)]
    #[reflect(hidden)]
    ui_tower_ability_tooltip_description: Handle<Text>,
    #[visit(skip)]
    #[reflect(hidden)]
    tower_ability_bar_cached_tooltip: Option<AbilityBarTooltipModel>,
    /// 4 技能圖片資源（HUD icon + tooltip icon 共用）
    #[visit(skip)]
    #[reflect(hidden)]
    ability_textures: [Option<fyrox::resource::texture::TextureResource>; 4],
    /// 目前每個技能 slot 已套用的 icon path，用來避免每 frame 重讀圖片。
    #[visit(skip)]
    #[reflect(hidden)]
    ability_icon_paths: [String; 4],
    /// 技能 icon texture cache：key 是 `AbilityDef.icon` 的相對路徑。
    #[visit(skip)]
    #[reflect(hidden)]
    ability_icon_texture_cache: HashMap<String, Option<fyrox::resource::texture::TextureResource>>,
    /// 4 icon 的 screen AABB (x, y, w, h) — 供滑鼠 hit-test
    #[visit(skip)]
    #[reflect(hidden)]
    ability_icon_rects: [(f32, f32, f32, f32); 4],
    /// 4 個三角升級按鈕 hit-test rect；不可點擊時放螢幕外
    #[visit(skip)]
    #[reflect(hidden)]
    ability_upgrade_button_rects: [(f32, f32, f32, f32); 4],
    /// 技能詳細資訊 map（key = ability id），由 hero.abilities_info 事件填入
    #[visit(skip)]
    #[reflect(hidden)]
    ability_info_map: HashMap<String, AbilityInfo>,
    /// 原始滑鼠螢幕座標（pixel）
    #[visit(skip)]
    #[reflect(hidden)]
    mouse_screen_pos: Vector2<f32>,
    /// 目前 hover 的 ability slot index（0-3）
    #[visit(skip)]
    #[reflect(hidden)]
    hovered_ability: Option<usize>,
    /// 目前 hover 的升級按鈕 index（0-2）
    #[visit(skip)]
    #[reflect(hidden)]
    hovered_upgrade: Option<usize>,
    /// 升級說明 tooltip：背景框
    #[visit(skip)]
    #[reflect(hidden)]
    ui_upgrade_tooltip_bg: Handle<UiNode>,
    /// 升級說明 tooltip：升級名稱（綠色）
    #[visit(skip)]
    #[reflect(hidden)]
    ui_upgrade_tooltip_title: Handle<Text>,
    /// 升級說明 tooltip：說明文字第一行（白色）
    #[visit(skip)]
    #[reflect(hidden)]
    ui_upgrade_tooltip_desc: Handle<Text>,
    /// 升級說明 tooltip：說明文字第二行（白色）
    #[visit(skip)]
    #[reflect(hidden)]
    ui_upgrade_tooltip_desc2: Handle<Text>,
    #[visit(skip)]
    #[reflect(hidden)]
    ui_tooltip_bg: Handle<UiNode>,
    #[visit(skip)]
    #[reflect(hidden)]
    ui_tooltip_icon: Handle<UiNode>,
    #[visit(skip)]
    #[reflect(hidden)]
    ui_tooltip_text: Handle<Text>,
    #[visit(skip)]
    #[reflect(hidden)]
    ui_shop_text: Handle<Text>,
    #[visit(skip)]
    #[reflect(hidden)]
    ui_end_text: Handle<Text>,

    // --- LoL MVP：从英雄缓存本地英雄状态。 *事件 ---
    #[visit(skip)]
    #[reflect(hidden)]
    hero_state: LocalHeroState,
    #[visit(skip)]
    #[reflect(hidden)]
    shop_visible: bool,
    #[visit(skip)]
    #[reflect(hidden)]
    shift_held: bool,
    #[visit(skip)]
    #[reflect(hidden)]
    attack_move_armed: bool,
    /// Ctrl 按住：蓋塔後不自動取消選塔模式（方便一次連蓋多個）
    #[visit(skip)]
    #[reflect(hidden)]
    ctrl_held: bool,
    /// Alt 按住：強制顯示 name label（即使 entity 數超過 NAME_LABEL_HIDE_THRESHOLD）
    #[visit(skip)]
    #[reflect(hidden)]
    alt_held: bool,
    #[visit(skip)]
    #[reflect(hidden)]
    game_ended: bool,
    #[visit(skip)]
    #[reflect(hidden)]
    viewport_sync_elapsed: f32,
    /// 上一次實際送出的 viewport (cx, cy, hw, hh)；值不變就跳過送出避免 omb log 洗版與
    /// 無謂的 KCP decode + mutex/channel work。reconnect 時 reset 為 None 以強制重送。
    #[visit(skip)]
    #[reflect(hidden)]
    last_sent_viewport: Option<(f32, f32, f32, f32)>,
    /// Camera 目前所在 render-world 座標（用於滑鼠座標換算與 label 螢幕換算）
    #[visit(skip)]
    #[reflect(hidden)]
    camera_world_pos: Vector2<f32>,
    /// 本秒累計的網路事件 logical (decompressed) payload bytes — UI 看「應用層」量
    #[visit(skip)]
    #[reflect(hidden)]
    net_bytes_current: u64,
    /// 上一秒的總 logical bytes，供顯示用
    #[visit(skip)]
    #[reflect(hidden)]
    net_bytes_last_sec: u64,
    /// 本秒累計的真實 wire bytes (壓縮後 + framing) — UI 看真實 bandwidth
    #[visit(skip)]
    #[reflect(hidden)]
    net_wire_bytes_current: u64,
    /// 上一秒的真實 wire bytes
    #[visit(skip)]
    #[reflect(hidden)]
    net_wire_bytes_last_sec: u64,
    /// 計時：每滿 1 秒 roll over
    #[visit(skip)]
    #[reflect(hidden)]
    net_stats_elapsed: f32,
    /// 最新一次 PingResponse 算出的 RTT (微秒)；None = 尚未收到任何 pong。
    /// 由 lockstep bg thread 透過 LockstepEvent::Latency 推上來，1 Hz 更新。
    #[visit(skip)]
    #[reflect(hidden)]
    latest_rtt_us: Option<u64>,
    /// FPS 顯示字串（例 "FPS 250 (4.0ms)"），來自 Fyrox renderer 的 frames_per_second
    /// 統計（plugin update 是 fixed 60 Hz tick，自算 frame count 沒意義）。
    #[visit(skip)]
    #[reflect(hidden)]
    fps_display: String,

    // ---- 英雄知識（Hero Knowledge）面板 ----
    /// 面板是否可見（點擊 GK 按鈕切換）。
    #[visit(skip)]
    #[reflect(hidden)]
    gk_panel_visible: bool,
    /// GK 面板 UI 狀態。
    #[visit(skip)]
    #[reflect(hidden)]
    gk_panel_ui: GkPanelUi,
    /// 快取：從 player_profile.json 讀取的 KP 和已解鎖節點（僅 UI 顯示用）。
    #[visit(skip)]
    #[reflect(hidden)]
    gk_available_kp: u32,
    /// 已解鎖節點 id 集合（HashSet 供 O(1) 查詢）。
    #[visit(skip)]
    #[reflect(hidden)]
    gk_unlocked_nodes: std::collections::HashSet<String>,
}

/// 技能詳細資訊（後端一次性廣播，用於 tooltip）
#[derive(Debug, Clone, Default)]
struct AbilityInfo {
    id: String,
    name: String,
    icon_path: String,
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
    abilities: Vec<String>, // ability ids, index 0=Q, 1=W, 2=E, 3=R
    ability_levels: HashMap<String, i32>,
    /// 技能剩餘冷卻秒數（key = ability id），本地遞減
    ability_cd: HashMap<String, f32>,
    /// 6 個 slot，每個 (item_id, cd)
    inventory: Vec<Option<(String, f32)>>,
}

/// MVP 商店清單（前端固定順序，對應後端 item id）
const SHOP_ITEMS: &[(&str, &str, i32)] = &[
    ("dmg_sword", "長劍", 500),
    ("dmg_rifle", "無雙鐵炮", 1600),
    ("hp_vest", "皮甲", 450),
    ("hp_armor", "重裝甲", 1400),
    ("mp_orb", "法力珠", 400),
    ("mp_staff", "秘法杖", 1200),
    ("ms_boots", "戰靴", 400),
    ("ms_swift", "疾風之靴", 1300),
    ("def_plate", "鎖子甲", 500),
    ("def_bulwark", "堡壘之盾", 1500),
];

impl Plugin for Game {
    fn register(&self, _context: PluginRegistrationContext) -> GameResult {
        Ok(())
    }

    fn init(&mut self, _scene_path: Option<&str>, mut context: PluginContext) -> GameResult {
        self.window_size = Vector2::new(800.0, 600.0);

        let mut scene = Scene::new();

        // 刪除預設的內建天空盒（在 2D 內容後面顯示為藍色/白色漸層）
        scene.set_skybox(None);

        // 2D 渲染選項
        use fyrox::scene::SceneRenderingOptions;
        scene
            .rendering_options
            .set_value_and_mark_modified(SceneRenderingOptions {
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

        // 覆蓋整個地圖的點光源
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

        // 背景（深綠色）
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

        // UI：狀態文字
        context
            .user_interfaces
            .add(UserInterface::new(Default::default()));
        let ui = context.user_interfaces.first_mut();

        // 載入CJK字體（Microsoft JhengHei）進行中文文字渲染
        if let Ok(font_data) = std::fs::read("C:/Windows/Fonts/msjh.ttc") {
            use fyrox::asset::untyped::ResourceKind;
            use fyrox::core::uuid::Uuid;
            use fyrox::gui::font::{Font, FontResource, FontStyles};
            if let Ok(font) = Font::from_memory(font_data, 1024, FontStyles::default(), vec![]) {
                let font_resource =
                    FontResource::new_ok(Uuid::new_v4(), ResourceKind::Embedded, font);
                ui.default_font = font_resource;
            }
        }

        // Fixed pool: never allocate widgets per snapshot/frame.
        for _ in 0..TOWER_ABILITY_BAR_PAGE_SIZE {
            let background = BorderBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_width(132.0)
                    .with_height(72.0)
                    .with_background(Brush::Solid(Color::from_rgba(30, 70, 42, 245)).into())
                    .with_foreground(Brush::Solid(Color::from_rgba(100, 220, 130, 255)).into()),
            )
            .with_stroke_thickness(Thickness::uniform(2.0).into())
            .with_corner_radius(6.0.into())
            .build(&mut ui.build_ctx())
            .transmute();
            let cooldown_overlay = BorderBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_width(132.0)
                    .with_height(0.0)
                    .with_background(Brush::Solid(Color::from_rgba(5, 8, 12, 185)).into()),
            )
            .build(&mut ui.build_ctx())
            .transmute();
            let active_overlay = BorderBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_width(132.0)
                    .with_height(0.0)
                    .with_background(Brush::Solid(Color::from_rgba(55, 125, 235, 95)).into()),
            )
            .build(&mut ui.build_ctx())
            .transmute();
            let slot = TextBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_width(132.0)
                    .with_height(72.0)
                    .with_foreground(Brush::Solid(Color::from_rgba(245, 245, 245, 255)).into()),
            )
            .with_text(String::new())
            .with_font_size(22.0.into())
            .with_wrap(WrapMode::Word)
            .with_horizontal_text_alignment(HorizontalAlignment::Center)
            .with_vertical_text_alignment(VerticalAlignment::Center)
            .build(&mut ui.build_ctx());
            self.ui_tower_ability_bar_backgrounds.push(background);
            self.ui_tower_ability_bar_cooldown_overlays
                .push(cooldown_overlay);
            self.ui_tower_ability_bar_active_overlays
                .push(active_overlay);
            self.ui_tower_ability_bar_slots.push(slot);
            self.tower_ability_bar_rects.push(UiRect::default());
            self.tower_ability_bar_cached_text.push(String::new());
            self.tower_ability_bar_cached_visual.push(None);
            self.tower_ability_bar_slot_bindings.push(None);
        }
        let page_control = |text: &str, ui: &mut UserInterface| {
            TextBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_width(36.0)
                    .with_height(36.0)
                    .with_background(Brush::Solid(Color::from_rgba(25, 35, 50, 240)).into())
                    .with_foreground(Brush::Solid(Color::from_rgba(245, 245, 245, 255)).into()),
            )
            .with_text(text.to_string())
            .with_font_size(24.0.into())
            .with_horizontal_text_alignment(HorizontalAlignment::Center)
            .with_vertical_text_alignment(VerticalAlignment::Center)
            .build(&mut ui.build_ctx())
        };
        self.ui_tower_ability_bar_previous = page_control("‹", ui);
        self.ui_tower_ability_bar_next = page_control("›", ui);
        self.ui_tower_ability_bar_page_indicator = page_control("1 / 1", ui);
        self.ui_tower_ability_tooltip_bg = BorderBuilder::new(
            WidgetBuilder::new()
                .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                .with_width(360.0)
                .with_height(196.0)
                .with_background(Brush::Solid(Color::from_rgba(20, 25, 36, 245)).into())
                .with_foreground(Brush::Solid(Color::from_rgba(100, 140, 210, 255)).into()),
        )
        .with_stroke_thickness(Thickness::uniform(2.0).into())
        .with_corner_radius(7.0.into())
        .build(&mut ui.build_ctx())
        .transmute();
        self.ui_tower_ability_tooltip_title = TextBuilder::new(
            WidgetBuilder::new()
                .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                .with_width(336.0)
                .with_foreground(Brush::Solid(Color::from_rgba(255, 220, 80, 255)).into()),
        )
        .with_text(String::new())
        .with_font_size(18.0.into())
        .with_wrap(WrapMode::Word)
        .build(&mut ui.build_ctx());
        self.ui_tower_ability_tooltip_description = TextBuilder::new(
            WidgetBuilder::new()
                .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                .with_width(336.0)
                .with_height(150.0)
                .with_foreground(Brush::Solid(Color::from_rgba(235, 235, 240, 255)).into()),
        )
        .with_text(String::new())
        .with_font_size(14.0.into())
        .with_wrap(WrapMode::Word)
        .build(&mut ui.build_ctx());

        // 先用通用 placeholder 建立 4 個 Image node；實際技能 icon 會在收到
        // AbilityRegistry snapshot 後依 AbilityDef.icon 置換。
        // 實際位置在 update() 依當前 window_size 置底中央。
        {
            let slot_label = ["W", "E", "R", "T"];
            let icon_size = 64.0f32;

            for i in 0..4 {
                let path = ABILITY_ICON_FALLBACK_PATH.to_string();
                let init_x = 500.0 + (i as f32) * 72.0;
                let init_y = 620.0;
                self.ability_icon_rects[i] = (init_x, init_y, icon_size, icon_size);
                let texture_opt = load_texture_from_rel_path(&path);
                let mut icon_builder = ImageBuilder::new(
                    WidgetBuilder::new()
                        .with_desired_position(Vector2::new(init_x, init_y))
                        .with_width(icon_size)
                        .with_height(icon_size),
                );
                if let Some(ref resource) = texture_opt {
                    icon_builder = icon_builder.with_texture(resource.clone());
                }
                let h: Handle<fyrox::gui::image::Image> = icon_builder.build(&mut ui.build_ctx());
                let icon_handle: Handle<UiNode> = h.transmute();
                self.ui_ability_icons[i] = icon_handle;
                self.ability_textures[i] = texture_opt;
                self.ability_icon_paths[i] = path;

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

                let upgrade = TextBuilder::new(
                    WidgetBuilder::new()
                        .with_desired_position(Vector2::new(init_x, init_y - 32.0))
                        .with_width(icon_size)
                        .with_height(32.0)
                        .with_foreground(Brush::Solid(Color::from_rgba(80, 255, 80, 255)).into()),
                )
                .with_text("".to_string())
                .with_font_size(32.0.into())
                .with_horizontal_text_alignment(HorizontalAlignment::Center)
                .build(&mut ui.build_ctx());
                self.ui_ability_upgrade_buttons[i] = upgrade;
                self.ability_upgrade_button_rects[i] = (-9999.0, -9999.0, 0.0, 0.0);

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

            // 升級說明 tooltip（BTD6 風格：深色圓角框 + 綠色標題 + 白色說明）
            self.ui_upgrade_tooltip_bg = BorderBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(-9999.0, -9999.0))
                    .with_width(300.0)
                    .with_height(110.0)
                    .with_background(Brush::Solid(Color::from_rgba(55, 38, 20, 230)).into())
                    .with_foreground(Brush::Solid(Color::from_rgba(120, 90, 40, 255)).into()),
            )
            .with_stroke_thickness(Thickness::uniform(2.0).into())
            .with_corner_radius(8.0.into())
            .build(&mut ui.build_ctx())
            .transmute();

            self.ui_upgrade_tooltip_title = TextBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(-9999.0, -9999.0))
                    .with_width(280.0)
                    .with_foreground(Brush::Solid(Color::from_rgba(255, 220, 50, 255)).into()),
            )
            .with_text("".to_string())
            .with_font_size(36.0.into())
            .with_wrap(WrapMode::Letter)
            .with_shadow(true)
            .with_shadow_brush(Brush::Solid(Color::from_rgba(0, 0, 0, 220)))
            .with_shadow_dilation(2.0)
            .with_shadow_offset(Vector2::new(2.0, 2.0))
            .build(&mut ui.build_ctx());

            let desc_builder = || {
                TextBuilder::new(
                    WidgetBuilder::new()
                        .with_desired_position(Vector2::new(-9999.0, -9999.0))
                        .with_width(280.0)
                        .with_foreground(Brush::Solid(Color::from_rgba(255, 255, 255, 255)).into()),
                )
                .with_text("".to_string())
                .with_font_size(28.0.into())
            };
            self.ui_upgrade_tooltip_desc = desc_builder().build(&mut ui.build_ctx());
            self.ui_upgrade_tooltip_desc2 = desc_builder().build(&mut ui.build_ctx());
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

        // TD 上方資源 HUD：用 icon + 數字取代 LIVES/GOLD/HP 文字列。
        for (i, asset) in ["hud_hp.png", "hud_lives.png", "hud_gold.png"]
            .iter()
            .enumerate()
        {
            let mut icon_builder = ImageBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_width(64.0)
                    .with_height(64.0),
            );
            if let Some(tex) = load_td_ui_texture(asset) {
                icon_builder = icon_builder.with_texture(tex);
            }
            self.ui_td_top_hud_icons[i] = icon_builder.build(&mut ui.build_ctx()).transmute();
            self.ui_td_top_hud_texts[i] = TextBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_width(180.0)
                    .with_foreground(Brush::Solid(Color::from_rgba(0, 0, 0, 255)).into()),
            )
            .with_text(String::new())
            .with_font_size(34.0.into())
            .build(&mut ui.build_ctx());
        }
        self.ui_td_wave_text = TextBuilder::new(
            WidgetBuilder::new()
                .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                .with_width(260.0)
                .with_foreground(Brush::Solid(Color::from_rgba(0, 0, 0, 255)).into()),
        )
        .with_text(String::new())
        .with_font_size(34.0.into())
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
                    .with_width(360.0)
                    .with_foreground(Brush::Solid(Color::from_rgba(255, 245, 225, 255)).into()),
            )
            .with_text(String::new())
            .with_font_size(30.0.into())
            .with_horizontal_text_alignment(HorizontalAlignment::Center)
            .build(&mut ui.build_ctx());

            self.ui_td_sell_button_text = TextBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(-9999.0, -9999.0))
                    .with_width(360.0)
                    .with_foreground(Brush::Solid(Color::from_rgba(120, 20, 20, 255)).into()),
            )
            .with_text(String::new())
            .with_font_size(38.0.into())
            .build(&mut ui.build_ctx());
            self.td_sell_button_rect = (-9999.0, -9999.0, 360.0, 42.0);
            self.td_target_priority_button_rect = (-9999.0, -9999.0, 0.0, 0.0);

            // 3 條路線升級按鈕（塔被選取時才定位到可見位置）
            for i in 0..3 {
                self.ui_td_upgrade_buttons[i] = TextBuilder::new(
                    WidgetBuilder::new()
                        .with_desired_position(Vector2::new(-9999.0, -9999.0))
                        .with_width(360.0)
                        .with_foreground(Brush::Solid(Color::from_rgba(255, 255, 255, 255)).into()),
                )
                .with_text(String::new())
                .with_font_size(19.0.into())
                .with_horizontal_text_alignment(HorizontalAlignment::Center)
                .build(&mut ui.build_ctx());
                self.td_upgrade_button_rects[i] = (-9999.0, -9999.0, 360.0, 38.0);
            }
        }

        // BTD-style 左側 selected tower panel：圖片資源缺失時仍由文字 fallback 顯示。
        {
            let mut bg_builder = ImageBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_width(340.0)
                    .with_height(760.0),
            );
            if let Some(tex) = load_td_ui_texture("panel_left.png") {
                bg_builder = bg_builder.with_texture(tex);
            }
            self.ui_td_selected_panel.bg = bg_builder.build(&mut ui.build_ctx()).transmute();
            // Unified body background (full panel height) — brown, square corners
            self.ui_td_selected_panel.body_bg = BorderBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_background(Brush::Solid(Color::from_rgba(145, 100, 55, 255)).into())
                    .with_width(380.0)
                    .with_height(650.0),
            )
            .with_stroke_thickness(Thickness::uniform(0.0).into())
            .build(&mut ui.build_ctx())
            .transmute();
            // Dark header strip — sits above body_bg, square corners
            self.ui_td_selected_panel.header_strip_bg = BorderBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_background(Brush::Solid(Color::from_rgba(75, 40, 12, 255)).into())
                    .with_width(380.0)
                    .with_height(62.0),
            )
            .with_stroke_thickness(Thickness::uniform(0.0).into())
            .build(&mut ui.build_ctx())
            .transmute();
            // Mask covering header_strip_bg's rounded bottom corners → makes strip look flat-bottomed
            self.ui_td_selected_panel.header_strip_bottom_mask = BorderBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_background(Brush::Solid(Color::from_rgba(75, 40, 12, 255)).into())
                    .with_width(380.0)
                    .with_height(20.0),
            )
            .with_stroke_thickness(Thickness::uniform(0.0).into())
            .build(&mut ui.build_ctx())
            .transmute();

            self.ui_td_selected_panel.tower_card_bg = {
                let mut builder = ImageBuilder::new(
                    WidgetBuilder::new()
                        .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                        .with_width(348.0)
                        .with_height(255.0),
                );
                if let Some(tex) = load_td_ui_texture("shop_card.png") {
                    builder = builder.with_texture(tex);
                }
                builder.build(&mut ui.build_ctx()).transmute()
            };

            self.ui_td_selected_panel.tower_icon = {
                let h: Handle<fyrox::gui::image::Image> = ImageBuilder::new(
                    WidgetBuilder::new()
                        .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                        .with_width(128.0)
                        .with_height(128.0),
                )
                .build(&mut ui.build_ctx());
                h.transmute()
            };
            self.ui_td_selected_panel.summary_text = TextBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_width(300.0)
                    .with_foreground(Brush::Solid(Color::from_rgba(55, 32, 12, 255)).into()),
            )
            .with_text(String::new())
            .with_font_size(24.0.into())
            .build(&mut ui.build_ctx());
            self.ui_td_selected_panel.refund_bg = {
                let mut builder = ImageBuilder::new(
                    WidgetBuilder::new()
                        .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                        .with_width(177.0)
                        .with_height(78.0),
                );
                if let Some(tex) = load_td_ui_texture("shop_card_locked.png") {
                    builder = builder.with_texture(tex);
                }
                builder.build(&mut ui.build_ctx()).transmute()
            };
            self.ui_td_selected_panel.gold_text = TextBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_width(177.0)
                    .with_foreground(Brush::Solid(Color::from_rgba(255, 245, 205, 255)).into()),
            )
            .with_text(String::new())
            .with_font_size(30.0.into())
            .build(&mut ui.build_ctx());
            self.ui_td_selected_panel.sell_icon = {
                let mut builder = ImageBuilder::new(
                    WidgetBuilder::new()
                        .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                        .with_width(42.0)
                        .with_height(42.0),
                );
                if let Some(tex) = load_td_ui_texture("sell.png") {
                    builder = builder.with_texture(tex);
                }
                let h: Handle<fyrox::gui::image::Image> = builder.build(&mut ui.build_ctx());
                h.transmute()
            };
            // 載入粗體字型（Microsoft JhengHei Bold）供升級名稱/價格使用
            let bold_font_resource: Option<fyrox::gui::font::FontResource> = {
                use fyrox::asset::untyped::ResourceKind;
                use fyrox::core::uuid::Uuid;
                use fyrox::gui::font::{Font, FontStyles};
                std::fs::read("C:/Windows/Fonts/msjhbd.ttc")
                    .ok()
                    .and_then(|data| {
                        Font::from_memory(data, 1024, FontStyles::default(), vec![])
                            .ok()
                            .map(|font| {
                                fyrox::gui::font::FontResource::new_ok(
                                    Uuid::new_v4(),
                                    ResourceKind::Embedded,
                                    font,
                                )
                            })
                    })
            };
            for i in 0..3 {
                // pip 區背景（稍淺棕色矩形，延伸蓋住綠色按鈕透明缺口，合成完整矩形視覺）
                self.ui_td_selected_panel.upgrade_row_bgs[i] = BorderBuilder::new(
                    WidgetBuilder::new()
                        .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                        .with_background(Brush::Solid(Color::from_rgba(175, 125, 60, 255)).into())
                        .with_width(244.0)
                        .with_height(160.0),
                )
                .with_stroke_thickness(Thickness::uniform(0.0).into())
                .build(&mut ui.build_ctx())
                .transmute();
                // 升級按鈕背景（btn_upgrade.png 512×360，關閉 sync_with_texture_size 避免撐大 layout）
                self.ui_td_selected_panel.upgrade_bgs[i] = {
                    let mut b = ImageBuilder::new(
                        WidgetBuilder::new()
                            .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                            .with_width(332.0)
                            .with_height(53.0),
                    )
                    .with_sync_with_texture_size(false)
                    .with_keep_aspect_ratio(false);
                    if let Some(tex) = load_td_ui_texture("btn_upgrade.png") {
                        b = b.with_texture(tex);
                    }
                    b.build(&mut ui.build_ctx()).transmute()
                };
                // 升級圖示（關閉 sync_with_texture_size 避免被 texture 原始尺寸覆蓋）
                self.ui_td_selected_panel.upgrade_icons[i] = {
                    let mut builder = ImageBuilder::new(
                        WidgetBuilder::new()
                            .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                            .with_width(38.0)
                            .with_height(38.0),
                    )
                    .with_sync_with_texture_size(false)
                    .with_keep_aspect_ratio(false);
                    if let Some(tex) = load_td_ui_texture(&format!("upgrade_p{}.png", i + 1)) {
                        builder = builder.with_texture(tex);
                    }
                    let h: Handle<fyrox::gui::image::Image> = builder.build(&mut ui.build_ctx());
                    h.transmute()
                };
                // 升級名稱文字（按鈕上方，粗體）
                self.ui_td_selected_panel.upgrade_name_texts[i] = {
                    let mut b = TextBuilder::new(
                        WidgetBuilder::new()
                            .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                            .with_width(190.0)
                            .with_foreground(
                                Brush::Solid(Color::from_rgba(255, 255, 255, 255)).into(),
                            ),
                    )
                    .with_text(String::new())
                    .with_font_size(26.0.into())
                    .with_horizontal_text_alignment(HorizontalAlignment::Center)
                    .with_vertical_text_alignment(VerticalAlignment::Center)
                    .with_wrap(WrapMode::Letter)
                    .with_shadow(true)
                    .with_shadow_brush(Brush::Solid(Color::from_rgba(0, 0, 0, 200)))
                    .with_shadow_dilation(2.0)
                    .with_shadow_offset(Vector2::new(2.0, 2.0));
                    if let Some(ref f) = bold_font_resource {
                        b = b.with_font(f.clone());
                    }
                    b.build(&mut ui.build_ctx())
                };
                // 5 格進度點（用字符顯示）
                self.ui_td_selected_panel.upgrade_pip_texts[i] = TextBuilder::new(
                    WidgetBuilder::new()
                        .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                        .with_width(36.0)
                        .with_foreground(Brush::Solid(Color::from_rgba(60, 35, 10, 255)).into()),
                )
                .with_text(String::new())
                .with_font_size(42.0.into())
                .with_horizontal_text_alignment(HorizontalAlignment::Center)
                .with_vertical_text_alignment(VerticalAlignment::Center)
                .build(&mut ui.build_ctx());
                // 未升級 / 級別X 文字
                self.ui_td_selected_panel.upgrade_status_texts[i] = TextBuilder::new(
                    WidgetBuilder::new()
                        .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                        .with_width(130.0)
                        .with_foreground(Brush::Solid(Color::from_rgba(60, 35, 10, 255)).into()),
                )
                .with_text(String::new())
                .with_font_size(40.0.into())
                .with_horizontal_text_alignment(HorizontalAlignment::Center)
                .with_vertical_text_alignment(VerticalAlignment::Center)
                .build(&mut ui.build_ctx());
                // 價格文字（按鈕下方，粗體）
                self.ui_td_selected_panel.upgrade_price_texts[i] = {
                    let mut b = TextBuilder::new(
                        WidgetBuilder::new()
                            .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                            .with_width(190.0)
                            .with_foreground(
                                Brush::Solid(Color::from_rgba(255, 255, 255, 255)).into(),
                            ),
                    )
                    .with_text(String::new())
                    .with_font_size(26.0.into())
                    .with_horizontal_text_alignment(HorizontalAlignment::Center)
                    .with_vertical_text_alignment(VerticalAlignment::Center)
                    .with_shadow(true)
                    .with_shadow_brush(Brush::Solid(Color::from_rgba(0, 0, 0, 200)))
                    .with_shadow_dilation(2.0)
                    .with_shadow_offset(Vector2::new(2.0, 2.0));
                    if let Some(ref f) = bold_font_resource {
                        b = b.with_font(f.clone());
                    }
                    b.build(&mut ui.build_ctx())
                };
            }
        }

        // BTD6-style selected tower panel 新增元素
        {
            // Purple bar: full width, centered in dark strip
            self.ui_td_selected_panel.header_bg = BorderBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_background(Brush::Solid(Color::from_rgba(118, 66, 190, 255)).into())
                    .with_width(380.0)
                    .with_height(46.0),
            )
            .with_stroke_thickness(Thickness::uniform(0.0).into())
            .build(&mut ui.build_ctx())
            .transmute();
            // Title: white with black outline (shadow at offset 0 = all-around glow)
            self.ui_td_selected_panel.header_title = TextBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_width(320.0)
                    .with_foreground(Brush::Solid(Color::from_rgba(255, 255, 255, 255)).into()),
            )
            .with_text(String::new())
            .with_font_size(38.0.into())
            .with_horizontal_text_alignment(HorizontalAlignment::Center)
            .with_vertical_text_alignment(VerticalAlignment::Center)
            .with_shadow(true)
            .with_shadow_brush(Brush::Solid(Color::BLACK))
            .with_shadow_dilation(2.0)
            .with_shadow_offset(Vector2::new(0.0, 0.0))
            .build(&mut ui.build_ctx());
            self.ui_td_selected_panel.pops_text = TextBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_width(150.0)
                    .with_foreground(Brush::Solid(Color::from_rgba(255, 230, 100, 255)).into()),
            )
            .with_text(String::new())
            .with_font_size(22.0.into())
            .with_horizontal_text_alignment(HorizontalAlignment::Left)
            .with_vertical_text_alignment(VerticalAlignment::Center)
            .with_shadow(true)
            .with_shadow_brush(Brush::Solid(Color::BLACK))
            .with_shadow_dilation(1.5)
            .with_shadow_offset(Vector2::new(1.0, 1.0))
            .build(&mut ui.build_ctx());
            self.ui_td_selected_panel.close_btn_bg = BorderBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_background(Brush::Solid(Color::from_rgba(0, 0, 0, 0)).into())
                    .with_width(40.0)
                    .with_height(40.0),
            )
            .with_stroke_thickness(Thickness::uniform(0.0).into())
            .with_corner_radius(0.0_f32.into())
            .build(&mut ui.build_ctx())
            .transmute();
            self.ui_td_selected_panel.close_btn_text = TextBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_width(40.0)
                    .with_foreground(Brush::Solid(Color::from_rgba(255, 255, 255, 255)).into()),
            )
            .with_text("X".to_string())
            .with_font_size(26.0.into())
            .with_horizontal_text_alignment(HorizontalAlignment::Center)
            .with_vertical_text_alignment(VerticalAlignment::Center)
            .build(&mut ui.build_ctx());
            self.ui_td_selected_panel.image_area_bg = BorderBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_background(Brush::Solid(Color::from_rgba(245, 195, 55, 255)).into())
                    .with_width(380.0)
                    .with_height(270.0),
            )
            .with_stroke_thickness(Thickness::uniform(0.0).into())
            .with_corner_radius(12.0_f32.into())
            .build(&mut ui.build_ctx())
            .transmute();
            self.ui_td_selected_panel.path_left_bg = BorderBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_background(Brush::Solid(Color::from_rgba(50, 30, 10, 160)).into())
                    .with_width(36.0)
                    .with_height(36.0),
            )
            .with_stroke_thickness(Thickness::uniform(0.0).into())
            .with_corner_radius(8.0_f32.into())
            .build(&mut ui.build_ctx())
            .transmute();
            self.ui_td_selected_panel.path_left_text = TextBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_width(36.0)
                    .with_foreground(Brush::Solid(Color::from_rgba(255, 255, 255, 255)).into()),
            )
            .with_text("<".to_string())
            .with_font_size(32.0.into())
            .with_horizontal_text_alignment(HorizontalAlignment::Center)
            .with_vertical_text_alignment(VerticalAlignment::Center)
            .build(&mut ui.build_ctx());
            self.ui_td_selected_panel.path_right_bg = BorderBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_background(Brush::Solid(Color::from_rgba(50, 30, 10, 160)).into())
                    .with_width(36.0)
                    .with_height(36.0),
            )
            .with_stroke_thickness(Thickness::uniform(0.0).into())
            .with_corner_radius(8.0_f32.into())
            .build(&mut ui.build_ctx())
            .transmute();
            self.ui_td_selected_panel.path_right_text = TextBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_width(36.0)
                    .with_foreground(Brush::Solid(Color::from_rgba(255, 255, 255, 255)).into()),
            )
            .with_text(">".to_string())
            .with_font_size(32.0.into())
            .with_horizontal_text_alignment(HorizontalAlignment::Center)
            .with_vertical_text_alignment(VerticalAlignment::Center)
            .build(&mut ui.build_ctx());
            self.ui_td_selected_panel.path_name_label = TextBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_width(380.0)
                    .with_foreground(Brush::Solid(Color::from_rgba(255, 255, 255, 255)).into()),
            )
            .with_text(String::new())
            .with_font_size(34.0.into())
            .with_horizontal_text_alignment(HorizontalAlignment::Center)
            .with_vertical_text_alignment(VerticalAlignment::Center)
            .build(&mut ui.build_ctx());
            self.ui_td_selected_panel.level_section_bg = BorderBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_background(Brush::Solid(Color::from_rgba(90, 55, 18, 255)).into())
                    .with_width(364.0)
                    .with_height(128.0),
            )
            .with_stroke_thickness(Thickness::uniform(0.0).into())
            .with_corner_radius(12.0_f32.into())
            .build(&mut ui.build_ctx())
            .transmute();
            self.ui_td_selected_panel.level_title_bar_bg = BorderBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_background(Brush::Solid(Color::from_rgba(50, 27, 8, 230)).into())
                    .with_width(380.0)
                    .with_height(42.0),
            )
            .with_stroke_thickness(Thickness::uniform(0.0).into())
            .build(&mut ui.build_ctx())
            .transmute();
            self.ui_td_selected_panel.level_badge_bg = BorderBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_background(Brush::Solid(Color::from_rgba(70, 130, 200, 255)).into())
                    .with_width(65.0)
                    .with_height(65.0),
            )
            .with_stroke_thickness(Thickness::uniform(0.0).into())
            .with_corner_radius(8.0_f32.into())
            .build(&mut ui.build_ctx())
            .transmute();
            self.ui_td_selected_panel.level_num_text = TextBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_width(36.0)
                    .with_foreground(Brush::Solid(Color::from_rgba(255, 255, 255, 255)).into()),
            )
            .with_text(String::new())
            .with_font_size(22.0.into())
            .with_horizontal_text_alignment(HorizontalAlignment::Center)
            .with_vertical_text_alignment(VerticalAlignment::Center)
            .build(&mut ui.build_ctx());
            self.ui_td_selected_panel.level_label_text = TextBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_width(270.0)
                    .with_foreground(Brush::Solid(Color::from_rgba(255, 255, 255, 255)).into()),
            )
            .with_text(String::new())
            .with_font_size(36.0.into())
            .with_horizontal_text_alignment(HorizontalAlignment::Center)
            .with_vertical_text_alignment(VerticalAlignment::Center)
            .build(&mut ui.build_ctx());
            self.ui_td_selected_panel.flavor_text_node = TextBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_width(350.0)
                    .with_foreground(Brush::Solid(Color::from_rgba(210, 185, 155, 255)).into()),
            )
            .with_text(String::new())
            .with_font_size(28.0.into())
            .with_horizontal_text_alignment(HorizontalAlignment::Center)
            .with_vertical_text_alignment(VerticalAlignment::Center)
            .build(&mut ui.build_ctx());
            self.ui_td_selected_panel.upgrade_section_bg = BorderBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_background(Brush::Solid(Color::from_rgba(90, 55, 18, 255)).into())
                    .with_width(364.0)
                    .with_height(208.0),
            )
            .with_stroke_thickness(Thickness::uniform(0.0).into())
            .with_corner_radius(12.0_f32.into())
            .build(&mut ui.build_ctx())
            .transmute();
            self.ui_td_selected_panel.unlock_title_bar_bg = BorderBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_background(Brush::Solid(Color::from_rgba(50, 27, 8, 230)).into())
                    .with_width(364.0)
                    .with_height(48.0),
            )
            .with_stroke_thickness(Thickness::uniform(0.0).into())
            .build(&mut ui.build_ctx())
            .transmute();
            self.ui_td_selected_panel.unlock_label_text = TextBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_width(350.0)
                    .with_foreground(Brush::Solid(Color::from_rgba(255, 255, 255, 255)).into()),
            )
            .with_text(String::new())
            .with_font_size(36.0.into())
            .with_horizontal_text_alignment(HorizontalAlignment::Center)
            .with_vertical_text_alignment(VerticalAlignment::Center)
            .build(&mut ui.build_ctx());
            self.ui_td_selected_panel.upgrade_green_bg = BorderBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_background(
                        Brush::LinearGradient {
                            from: Vector2::new(0.0, 0.0),
                            to: Vector2::new(0.0, 58.0),
                            stops: vec![
                                GradientPoint {
                                    stop: 0.0,
                                    color: Color::from_rgba(115, 210, 70, 255),
                                },
                                GradientPoint {
                                    stop: 1.0,
                                    color: Color::from_rgba(50, 145, 40, 255),
                                },
                            ],
                        }
                        .into(),
                    )
                    .with_width(164.0)
                    .with_height(58.0),
            )
            .with_stroke_thickness(Thickness::uniform(0.0).into())
            .with_corner_radius(10.0_f32.into())
            .build(&mut ui.build_ctx())
            .transmute();
            self.ui_td_selected_panel.upgrade_green_price = TextBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_width(164.0)
                    .with_foreground(Brush::Solid(Color::from_rgba(255, 255, 255, 255)).into()),
            )
            .with_text(String::new())
            .with_font_size(38.0.into())
            .with_horizontal_text_alignment(HorizontalAlignment::Center)
            .with_vertical_text_alignment(VerticalAlignment::Center)
            .build(&mut ui.build_ctx());
            self.ui_td_selected_panel.upgrade_path_btn_bg = BorderBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_background(Brush::Solid(Color::from_rgba(255, 152, 0, 255)).into())
                    .with_width(84.0)
                    .with_height(58.0),
            )
            .with_stroke_thickness(Thickness::uniform(0.0).into())
            .with_corner_radius(10.0_f32.into())
            .build(&mut ui.build_ctx())
            .transmute();
            self.ui_td_selected_panel.upgrade_path_btn_text = TextBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_width(84.0)
                    .with_foreground(Brush::Solid(Color::from_rgba(255, 255, 255, 255)).into()),
            )
            .with_text(String::new())
            .with_font_size(34.0.into())
            .with_horizontal_text_alignment(HorizontalAlignment::Center)
            .with_vertical_text_alignment(VerticalAlignment::Center)
            .build(&mut ui.build_ctx());
            self.ui_td_selected_panel.next_effect_text = TextBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_width(350.0)
                    .with_foreground(Brush::Solid(Color::from_rgba(240, 220, 185, 255)).into()),
            )
            .with_text(String::new())
            .with_font_size(28.0.into())
            .with_horizontal_text_alignment(HorizontalAlignment::Center)
            .with_vertical_text_alignment(VerticalAlignment::Center)
            .build(&mut ui.build_ctx());
            self.ui_td_selected_panel.sell_section_bg = BorderBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_background(Brush::Solid(Color::from_rgba(40, 22, 8, 255)).into())
                    .with_width(380.0)
                    .with_height(100.0),
            )
            .with_stroke_thickness(Thickness::uniform(0.0).into())
            .with_corner_radius(20.0_f32.into())
            .build(&mut ui.build_ctx())
            .transmute();
            // mask the top two rounded corners of sell_section_bg with body_bg color
            self.ui_td_selected_panel.sell_top_mask = BorderBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_background(Brush::Solid(Color::from_rgba(145, 100, 55, 255)).into())
                    .with_width(380.0)
                    .with_height(22.0),
            )
            .with_stroke_thickness(Thickness::uniform(0.0).into())
            .build(&mut ui.build_ctx())
            .transmute();
            self.ui_td_selected_panel.sell_coin_icon = BorderBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_background(
                        Brush::LinearGradient {
                            from: Vector2::new(0.0, 0.0),
                            to: Vector2::new(0.0, 28.0),
                            stops: vec![
                                GradientPoint {
                                    stop: 0.0,
                                    color: Color::from_rgba(255, 230, 80, 255),
                                },
                                GradientPoint {
                                    stop: 1.0,
                                    color: Color::from_rgba(210, 155, 20, 255),
                                },
                            ],
                        }
                        .into(),
                    )
                    .with_width(28.0)
                    .with_height(28.0),
            )
            .with_stroke_thickness(Thickness::uniform(0.0).into())
            .with_corner_radius(14.0_f32.into())
            .build(&mut ui.build_ctx())
            .transmute();
            self.ui_td_selected_panel.sell_coin_text = TextBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_width(145.0)
                    .with_foreground(Brush::Solid(Color::from_rgba(255, 230, 100, 255)).into()),
            )
            .with_text(String::new())
            .with_font_size(30.0.into())
            .with_horizontal_text_alignment(HorizontalAlignment::Left)
            .with_vertical_text_alignment(VerticalAlignment::Center)
            .build(&mut ui.build_ctx());
            self.ui_td_selected_panel.sell_red_bg = BorderBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_background(
                        Brush::LinearGradient {
                            from: Vector2::new(0.0, 0.0),
                            to: Vector2::new(0.0, 65.0),
                            stops: vec![
                                GradientPoint {
                                    stop: 0.0,
                                    color: Color::from_rgba(235, 130, 40, 255),
                                },
                                GradientPoint {
                                    stop: 1.0,
                                    color: Color::from_rgba(195, 35, 15, 255),
                                },
                            ],
                        }
                        .into(),
                    )
                    .with_width(155.0)
                    .with_height(65.0),
            )
            .with_stroke_thickness(Thickness::uniform(0.0).into())
            .with_corner_radius(10.0_f32.into())
            .build(&mut ui.build_ctx())
            .transmute();
            self.ui_td_selected_panel.sell_red_text = TextBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_width(155.0)
                    .with_foreground(Brush::Solid(Color::from_rgba(255, 255, 255, 255)).into()),
            )
            .with_text("賣出".to_string())
            .with_font_size(36.0.into())
            .with_horizontal_text_alignment(HorizontalAlignment::Center)
            .with_vertical_text_alignment(VerticalAlignment::Center)
            .build(&mut ui.build_ctx());
            // btd6_tower_icon 最後建立，確保 z-order 高於所有背景
            self.ui_td_selected_panel.btd6_tower_icon = {
                let builder = ImageBuilder::new(
                    WidgetBuilder::new()
                        .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                        .with_width(200.0)
                        .with_height(200.0),
                );
                builder.build(&mut ui.build_ctx()).transmute()
            };
            // i 說明按鈕（建在卡片之後確保 z-order 在最上層）
            self.ui_td_selected_panel.info_overlay_bg = BorderBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_background(Brush::Solid(Color::from_rgba(55, 38, 20, 230)).into())
                    .with_foreground(Brush::Solid(Color::from_rgba(120, 90, 40, 255)).into())
                    .with_width(244.0)
                    .with_height(230.0),
            )
            .with_stroke_thickness(Thickness::uniform(2.0).into())
            .with_corner_radius(8.0.into())
            .build(&mut ui.build_ctx())
            .transmute();
            let stat_labels = ["", "", "", ""];
            for i in 0..4usize {
                self.ui_td_selected_panel.info_stat_texts[i] = TextBuilder::new(
                    WidgetBuilder::new()
                        .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                        .with_width(220.0)
                        .with_foreground(Brush::Solid(Color::from_rgba(255, 255, 255, 255)).into()),
                )
                .with_text(stat_labels[i].to_string())
                .with_font_size(28.0.into())
                .with_horizontal_text_alignment(HorizontalAlignment::Left)
                .with_vertical_text_alignment(VerticalAlignment::Center)
                .with_shadow(true)
                .with_shadow_brush(Brush::Solid(Color::BLACK))
                .with_shadow_dilation(1.5)
                .with_shadow_offset(Vector2::new(1.0, 1.0))
                .build(&mut ui.build_ctx());
            }
            self.ui_td_selected_panel.info_btn_bg = BorderBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_background(Brush::Solid(Color::from_rgba(185, 120, 25, 230)).into())
                    .with_width(36.0)
                    .with_height(36.0),
            )
            .with_stroke_thickness(Thickness::uniform(2.0).into())
            .with_corner_radius(18.0_f32.into())
            .build(&mut ui.build_ctx())
            .transmute();
            self.ui_td_selected_panel.info_btn_text = TextBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_width(36.0)
                    .with_foreground(Brush::Solid(Color::from_rgba(255, 240, 180, 255)).into()),
            )
            .with_text("i".to_string())
            .with_font_size(22.0.into())
            .with_horizontal_text_alignment(HorizontalAlignment::Center)
            .with_vertical_text_alignment(VerticalAlignment::Center)
            .with_shadow(true)
            .with_shadow_brush(Brush::Solid(Color::BLACK))
            .with_shadow_dilation(1.0)
            .with_shadow_offset(Vector2::new(1.0, 1.0))
            .build(&mut ui.build_ctx());
            // 三列升級 tooltip（show_info 時同時顯示，樣式同 hover tooltip）
            for i in 0..3usize {
                self.ui_td_selected_panel.info_row_bgs[i] = BorderBuilder::new(
                    WidgetBuilder::new()
                        .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                        .with_width(300.0)
                        .with_height(110.0)
                        .with_background(Brush::Solid(Color::from_rgba(55, 38, 20, 230)).into())
                        .with_foreground(Brush::Solid(Color::from_rgba(120, 90, 40, 255)).into()),
                )
                .with_stroke_thickness(Thickness::uniform(2.0).into())
                .with_corner_radius(8.0.into())
                .build(&mut ui.build_ctx())
                .transmute();
                self.ui_td_selected_panel.info_row_titles[i] = TextBuilder::new(
                    WidgetBuilder::new()
                        .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                        .with_width(280.0)
                        .with_foreground(Brush::Solid(Color::from_rgba(255, 220, 50, 255)).into()),
                )
                .with_text(String::new())
                .with_font_size(36.0.into())
                .with_wrap(WrapMode::Letter)
                .with_shadow(true)
                .with_shadow_brush(Brush::Solid(Color::from_rgba(0, 0, 0, 220)))
                .with_shadow_dilation(2.0)
                .with_shadow_offset(Vector2::new(2.0, 2.0))
                .build(&mut ui.build_ctx());
                let desc_row = |ctx: &mut fyrox::gui::BuildContext<'_>| {
                    TextBuilder::new(
                        WidgetBuilder::new()
                            .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                            .with_width(280.0)
                            .with_foreground(
                                Brush::Solid(Color::from_rgba(255, 255, 255, 255)).into(),
                            ),
                    )
                    .with_text(String::new())
                    .with_font_size(28.0.into())
                    .build(ctx)
                };
                self.ui_td_selected_panel.info_row_descs[i] = desc_row(&mut ui.build_ctx());
                self.ui_td_selected_panel.info_row_descs2[i] = desc_row(&mut ui.build_ctx());
            }
        }

        // BTD-style 右側 shop/control panel：買塔常駐，Start/Pause 固定右側。
        {
            let mut bg_builder = ImageBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_width(320.0)
                    .with_height(920.0),
            );
            if let Some(tex) = load_td_ui_texture("panel_right.png") {
                bg_builder = bg_builder.with_texture(tex);
            }
            self.ui_td_right_panel.bg = bg_builder.build(&mut ui.build_ctx()).transmute();
            self.ui_td_right_panel.title_text = TextBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_width(260.0)
                    .with_foreground(Brush::Solid(Color::from_rgba(255, 245, 225, 255)).into()),
            )
            .with_text("塔商店".to_string())
            .with_font_size(36.0.into())
            .with_horizontal_text_alignment(HorizontalAlignment::Center)
            .build(&mut ui.build_ctx());
            self.ui_td_right_panel.viewport_bg = CanvasBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_width(360.0)
                    .with_height(745.0)
                    .with_clip_to_bounds(true),
            )
            .build(&mut ui.build_ctx())
            .transmute();
            self.ui_td_right_panel.scroll_track = {
                let builder = ImageBuilder::new(
                    WidgetBuilder::new()
                        .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                        .with_width(18.0)
                        .with_height(745.0),
                );
                builder.build(&mut ui.build_ctx()).transmute()
            };
            self.ui_td_right_panel.scroll_thumb = {
                let mut builder = ImageBuilder::new(
                    WidgetBuilder::new()
                        .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                        .with_width(18.0)
                        .with_height(180.0),
                );
                if let Some(tex) = load_td_ui_texture("shop_card_selected.png") {
                    builder = builder.with_texture(tex);
                }
                builder.build(&mut ui.build_ctx()).transmute()
            };
            self.ui_td_right_panel.start_icon = {
                let mut builder = ImageBuilder::new(
                    WidgetBuilder::new()
                        .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                        .with_width(66.0)
                        .with_height(66.0),
                );
                if let Some(tex) = load_td_ui_texture("start_round.png") {
                    builder = builder.with_texture(tex);
                }
                let h: Handle<fyrox::gui::image::Image> = builder.build(&mut ui.build_ctx());
                h.transmute()
            };
            self.ui_td_right_panel.pause_icon = {
                let mut builder = ImageBuilder::new(
                    WidgetBuilder::new()
                        .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                        .with_width(66.0)
                        .with_height(66.0),
                );
                if let Some(tex) = load_td_ui_texture("pause.png") {
                    builder = builder.with_texture(tex);
                }
                let h: Handle<fyrox::gui::image::Image> = builder.build(&mut ui.build_ctx());
                h.transmute()
            };
            self.ui_td_right_panel.pause_text = TextBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_width(96.0)
                    .with_foreground(Brush::Solid(Color::from_rgba(245, 245, 245, 180)).into()),
            )
            .with_text("PAUSE".to_string())
            .with_font_size(22.0.into())
            .with_horizontal_text_alignment(HorizontalAlignment::Center)
            .build(&mut ui.build_ctx());

            // 英雄知識入口按鈕（右側面板頂部）
            self.gk_panel_ui.entry_bg = BorderBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_foreground(Brush::Solid(Color::from_rgba(200, 170, 80, 255)).into())
                    .with_background(
                        Brush::LinearGradient {
                            from: Vector2::new(0.0, 0.0),
                            to: Vector2::new(0.0, 44.0),
                            stops: vec![
                                GradientPoint { stop: 0.0, color: Color::from_rgba(180, 140, 60, 255) },
                                GradientPoint { stop: 1.0, color: Color::from_rgba(120, 90, 30, 255) },
                            ],
                        }
                        .into(),
                    ),
            )
            .with_stroke_thickness(Thickness::uniform(2.0).into())
            .with_corner_radius(10.0_f32.into())
            .build(&mut ui.build_ctx())
            .transmute();

            self.gk_panel_ui.entry_text_h = TextBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_foreground(Brush::Solid(Color::from_rgba(255, 240, 180, 255)).into()),
            )
            .with_text("英雄知識".to_string())
            .with_font_size(22.0.into())
            .with_horizontal_text_alignment(HorizontalAlignment::Center)
            .with_vertical_text_alignment(VerticalAlignment::Center)
            .build(&mut ui.build_ctx());
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
            .with_font_size(30.0.into())
            .build(&mut ui.build_ctx());
            self.ui_td_auto_start_checkbox_text = TextBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(-9999.0, -9999.0))
                    .with_width(142.0)
                    .with_foreground(Brush::Solid(Color::from_rgba(245, 245, 245, 220)).into()),
            )
            .with_text(td_auto_start_checkbox_label(false))
            .with_font_size(24.0.into())
            .with_horizontal_text_alignment(HorizontalAlignment::Center)
            .build(&mut ui.build_ctx());
            self.start_round_button_rect = (-9999.0, -9999.0, 240.0, 48.0);
            self.auto_start_checkbox_rect = (UI_HIDDEN_POS, UI_HIDDEN_POS, 0.0, 0.0);
            self.pause_button_rect = (UI_HIDDEN_POS, UI_HIDDEN_POS, 0.0, 0.0);
        }

        // TD 模式右側塔按鈕（text-only，動態 Vec）
        // 收到 game/tower_templates 事件後才建；此時空 Vec
        self.ui_td_tower_buttons = Vec::new();
        self.ui_td_tower_cards = Vec::new();
        self.td_tower_button_rects = Vec::new();
        self.td_ui_texture_cache = HashMap::new();
        self.tower_texture_cache = HashMap::new();
        self.tower_material_cache = HashMap::new();
        self.tower_composites = HashMap::new();
        self.sim_seen_tower_fire_fx.clear();
        self.sim_seen_attack_phase_fx.clear();
        self.sim_seen_attack_cancel_fx.clear();
        self.hero_model_assets.clear();
        self.hero_action_assets.clear();
        self.hero_asset_failures_logged.clear();
        self.hero_model_nodes.clear();

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

        self.ui_in_game_return.bg = BorderBuilder::new(
            WidgetBuilder::new()
                .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                .with_width(1.0)
                .with_height(1.0)
                .with_background(Brush::Solid(Color::from_rgba(22, 42, 30, 218)).into()),
        )
        .with_stroke_thickness(Thickness::uniform(0.0).into())
        .with_corner_radius(8.0_f32.into())
        .build(&mut ui.build_ctx())
        .transmute();
        self.ui_in_game_return.text = TextBuilder::new(
            WidgetBuilder::new()
                .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                .with_width(1.0)
                .with_height(1.0)
                .with_foreground(Brush::Solid(Color::from_rgba(240, 248, 242, 255)).into()),
        )
        .with_text(String::new())
        .with_font_size(22.0.into())
        .with_horizontal_text_alignment(HorizontalAlignment::Center)
        .with_vertical_text_alignment(VerticalAlignment::Center)
        .build(&mut ui.build_ctx());
        self.return_to_title_button_rect = UiRect {
            x: UI_HIDDEN_POS,
            y: UI_HIDDEN_POS,
            w: 0.0,
            h: 0.0,
        };

        // Inventory 初始 6 格
        self.hero_state.inventory = vec![None; 6];

        self.pregame_runtime =
            pregame::PregameRuntime::new_for_menu(pregame::PregameCatalog::load());
        for diagnostic in &self.pregame_runtime.catalog.diagnostics {
            log::warn!("pregame catalog: {}", diagnostic);
        }
        self.ui_pregame.background = BorderBuilder::new(
            WidgetBuilder::new()
                .with_desired_position(Vector2::new(0.0, 0.0))
                .with_width(800.0)
                .with_height(600.0)
                .with_background(Brush::Solid(Color::from_rgba(118, 202, 132, 255)).into()),
        )
        .with_stroke_thickness(Thickness::uniform(0.0).into())
        .build(&mut ui.build_ctx())
        .transmute();
        self.ui_pregame.panel = BorderBuilder::new(
            WidgetBuilder::new()
                .with_desired_position(Vector2::new(80.0, 80.0))
                .with_width(640.0)
                .with_height(420.0)
                .with_background(Brush::Solid(Color::from_rgba(255, 246, 210, 238)).into()),
        )
        .with_stroke_thickness(Thickness::uniform(0.0).into())
        .with_corner_radius(8.0_f32.into())
        .build(&mut ui.build_ctx())
        .transmute();
        self.ui_pregame.title = TextBuilder::new(
            WidgetBuilder::new()
                .with_desired_position(Vector2::new(120.0, 108.0))
                .with_width(560.0)
                .with_foreground(Brush::Solid(Color::from_rgba(35, 62, 42, 255)).into()),
        )
        .with_text(String::new())
        .with_font_size(48.0.into())
        .with_horizontal_text_alignment(HorizontalAlignment::Center)
        .build(&mut ui.build_ctx());
        self.ui_pregame.subtitle = TextBuilder::new(
            WidgetBuilder::new()
                .with_desired_position(Vector2::new(140.0, 168.0))
                .with_width(520.0)
                .with_foreground(Brush::Solid(Color::from_rgba(70, 80, 58, 255)).into()),
        )
        .with_text(String::new())
        .with_font_size(22.0.into())
        .with_horizontal_text_alignment(HorizontalAlignment::Center)
        .build(&mut ui.build_ctx());
        self.ui_pregame.status = TextBuilder::new(
            WidgetBuilder::new()
                .with_desired_position(Vector2::new(140.0, 470.0))
                .with_width(520.0)
                .with_foreground(Brush::Solid(Color::from_rgba(120, 40, 30, 255)).into()),
        )
        .with_text(String::new())
        .with_font_size(20.0.into())
        .with_horizontal_text_alignment(HorizontalAlignment::Center)
        .build(&mut ui.build_ctx());
        self.ui_pregame.buttons.clear();
        for _ in 0..32 {
            let bg = BorderBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_width(260.0)
                    .with_height(72.0)
                    .with_background(Brush::Solid(Color::from_rgba(245, 178, 54, 255)).into()),
            )
            .with_stroke_thickness(Thickness::uniform(0.0).into())
            .with_corner_radius(8.0_f32.into())
            .build(&mut ui.build_ctx())
            .transmute();
            let image = ImageBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_width(1.0)
                    .with_height(1.0),
            )
            .build(&mut ui.build_ctx())
            .transmute();
            let text = TextBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_width(260.0)
                    .with_height(72.0)
                    .with_foreground(Brush::Solid(Color::from_rgba(55, 32, 12, 255)).into()),
            )
            .with_text(String::new())
            .with_font_size(22.0.into())
            .with_horizontal_text_alignment(HorizontalAlignment::Center)
            .with_vertical_text_alignment(VerticalAlignment::Center)
            .build(&mut ui.build_ctx());
            self.ui_pregame.buttons.push(PregameButtonUi {
                bg,
                image,
                text,
                role: PregameVisualRole::Button,
            });
        }

        // 設定面板初始化
        self.settings_sfx_volume = 1.0;
        self.settings_music_volume = audio_settings::AudioSettings::load_or_create().music_volume;
        self.settings_speed_value = 1.0;
        self.ui_settings.bg = BorderBuilder::new(
            WidgetBuilder::new()
                .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                .with_width(1.0)
                .with_height(1.0)
                .with_background(Brush::Solid(Color::from_rgba(22, 42, 30, 255)).into()),
        )
        .with_stroke_thickness(Thickness::uniform(0.0).into())
        .build(&mut ui.build_ctx())
        .transmute();
        self.ui_settings.title_bg = BorderBuilder::new(
            WidgetBuilder::new()
                .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                .with_width(1.0)
                .with_height(1.0)
                .with_background(Brush::Solid(Color::from_rgba(110, 72, 30, 255)).into()),
        )
        .with_stroke_thickness(Thickness::uniform(0.0).into())
        .with_corner_radius(8.0_f32.into())
        .build(&mut ui.build_ctx())
        .transmute();
        self.ui_settings.title_text = TextBuilder::new(
            WidgetBuilder::new()
                .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                .with_width(1.0)
                .with_height(1.0)
                .with_foreground(Brush::Solid(Color::from_rgba(220, 220, 220, 255)).into()),
        )
        .with_text(String::new())
        .with_font_size(28.0.into())
        .with_horizontal_text_alignment(HorizontalAlignment::Center)
        .with_vertical_text_alignment(VerticalAlignment::Center)
        .build(&mut ui.build_ctx());
        self.ui_settings.back_btn = BorderBuilder::new(
            WidgetBuilder::new()
                .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                .with_width(1.0)
                .with_height(1.0)
                .with_background(Brush::Solid(Color::from_rgba(65, 158, 218, 255)).into()),
        )
        .with_stroke_thickness(Thickness::uniform(0.0).into())
        .with_corner_radius(12.0_f32.into())
        .build(&mut ui.build_ctx())
        .transmute();
        self.ui_settings.back_btn_text = TextBuilder::new(
            WidgetBuilder::new()
                .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                .with_width(1.0)
                .with_height(1.0)
                .with_foreground(Brush::Solid(Color::from_rgba(240, 248, 242, 255)).into()),
        )
        .with_text(String::new())
        .with_font_size(22.0.into())
        .with_horizontal_text_alignment(HorizontalAlignment::Center)
        .with_vertical_text_alignment(VerticalAlignment::Center)
        .build(&mut ui.build_ctx());
        // Resolution badge (top-right)
        self.ui_settings.resolution_badge_bg = BorderBuilder::new(
            WidgetBuilder::new()
                .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                .with_width(1.0)
                .with_height(1.0)
                .with_background(Brush::Solid(Color::from_rgba(50, 42, 28, 255)).into()),
        )
        .with_stroke_thickness(Thickness::uniform(0.0).into())
        .with_corner_radius(6.0_f32.into())
        .build(&mut ui.build_ctx())
        .transmute();
        self.ui_settings.resolution_badge_label = TextBuilder::new(
            WidgetBuilder::new()
                .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                .with_width(1.0)
                .with_height(1.0)
                .with_foreground(Brush::Solid(Color::from_rgba(200, 185, 155, 255)).into()),
        )
        .with_text(String::new())
        .with_font_size(14.0.into())
        .with_horizontal_text_alignment(HorizontalAlignment::Center)
        .build(&mut ui.build_ctx());
        self.ui_settings.resolution_badge_value = TextBuilder::new(
            WidgetBuilder::new()
                .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                .with_width(1.0)
                .with_height(1.0)
                .with_foreground(Brush::Solid(Color::from_rgba(240, 230, 200, 255)).into()),
        )
        .with_text(String::new())
        .with_font_size(16.0.into())
        .with_horizontal_text_alignment(HorizontalAlignment::Center)
        .build(&mut ui.build_ctx());
        // Left decorative panel
        self.ui_settings.left_panel_bg = BorderBuilder::new(
            WidgetBuilder::new()
                .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                .with_width(1.0)
                .with_height(1.0)
                .with_background(Brush::Solid(Color::from_rgba(28, 34, 48, 255)).into()),
        )
        .with_stroke_thickness(Thickness::uniform(0.0).into())
        .with_corner_radius(12.0_f32.into())
        .build(&mut ui.build_ctx())
        .transmute();
        // Jukebox illustration area (top half of left panel)
        self.ui_settings.left_panel_jukebox_bg = BorderBuilder::new(
            WidgetBuilder::new()
                .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                .with_width(1.0)
                .with_height(1.0)
                .with_background(Brush::Solid(Color::from_rgba(20, 26, 40, 255)).into()),
        )
        .with_stroke_thickness(Thickness::uniform(0.0).into())
        .with_corner_radius(10.0_f32.into())
        .build(&mut ui.build_ctx())
        .transmute();
        // Inner decorative block (simulate jukebox body)
        self.ui_settings.left_panel_jukebox_inner = BorderBuilder::new(
            WidgetBuilder::new()
                .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                .with_width(1.0)
                .with_height(1.0)
                .with_background(Brush::Solid(Color::from_rgba(55, 105, 165, 255)).into()),
        )
        .with_stroke_thickness(Thickness::uniform(2.0).into())
        .with_corner_radius(8.0_f32.into())
        .build(&mut ui.build_ctx())
        .transmute();
        self.ui_settings.left_panel_btn_bg = BorderBuilder::new(
            WidgetBuilder::new()
                .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                .with_width(1.0)
                .with_height(1.0)
                .with_background(Brush::Solid(Color::from_rgba(60, 130, 200, 255)).into()),
        )
        .with_stroke_thickness(Thickness::uniform(0.0).into())
        .with_corner_radius(6.0_f32.into())
        .build(&mut ui.build_ctx())
        .transmute();
        self.ui_settings.left_panel_btn_text = TextBuilder::new(
            WidgetBuilder::new()
                .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                .with_width(1.0)
                .with_height(1.0)
                .with_foreground(Brush::Solid(Color::from_rgba(255, 255, 255, 255)).into()),
        )
        .with_text(String::new())
        .with_font_size(18.0.into())
        .with_horizontal_text_alignment(HorizontalAlignment::Center)
        .with_vertical_text_alignment(VerticalAlignment::Center)
        .build(&mut ui.build_ctx());
        self.ui_settings.left_panel_toggle_bg = BorderBuilder::new(
            WidgetBuilder::new()
                .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                .with_width(1.0)
                .with_height(1.0)
                .with_background(Brush::Solid(Color::from_rgba(60, 180, 80, 255)).into()),
        )
        .with_stroke_thickness(Thickness::uniform(0.0).into())
        .with_corner_radius(6.0_f32.into())
        .build(&mut ui.build_ctx())
        .transmute();
        self.ui_settings.left_panel_enable_label = TextBuilder::new(
            WidgetBuilder::new()
                .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                .with_width(1.0)
                .with_height(1.0)
                .with_foreground(Brush::Solid(Color::from_rgba(180, 185, 195, 255)).into()),
        )
        .with_text(String::new())
        .with_font_size(16.0.into())
        .with_vertical_text_alignment(VerticalAlignment::Center)
        .build(&mut ui.build_ctx());
        // Right slider panel background
        self.ui_settings.slider_panel_bg = BorderBuilder::new(
            WidgetBuilder::new()
                .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                .with_width(1.0)
                .with_height(1.0)
                .with_background(Brush::Solid(Color::from_rgba(58, 46, 28, 255)).into()),
        )
        .with_stroke_thickness(Thickness::uniform(0.0).into())
        .with_corner_radius(10.0_f32.into())
        .build(&mut ui.build_ctx())
        .transmute();
        // Helper: build one slider set (icon_rgba, fill_rgba, thumb_rgba)
        let mut make_slider = |ui: &mut UserInterface,
                               icon_c: (u8, u8, u8, u8),
                               fill_c: (u8, u8, u8, u8),
                               thumb_c: (u8, u8, u8, u8)|
         -> SettingsSlider {
            let icon = BorderBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_width(1.0)
                    .with_height(1.0)
                    .with_background(
                        Brush::Solid(Color::from_rgba(icon_c.0, icon_c.1, icon_c.2, icon_c.3))
                            .into(),
                    ),
            )
            .with_stroke_thickness(Thickness::uniform(0.0).into())
            .with_corner_radius(999.0_f32.into())
            .build(&mut ui.build_ctx())
            .transmute();
            let track = BorderBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_width(1.0)
                    .with_height(1.0)
                    .with_background(Brush::Solid(Color::from_rgba(42, 32, 18, 255)).into()),
            )
            .with_stroke_thickness(Thickness::uniform(0.0).into())
            .with_corner_radius(999.0_f32.into())
            .build(&mut ui.build_ctx())
            .transmute();
            let fill = BorderBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_width(1.0)
                    .with_height(1.0)
                    .with_background(
                        Brush::Solid(Color::from_rgba(fill_c.0, fill_c.1, fill_c.2, fill_c.3))
                            .into(),
                    ),
            )
            .with_stroke_thickness(Thickness::uniform(0.0).into())
            .with_corner_radius(999.0_f32.into())
            .build(&mut ui.build_ctx())
            .transmute();
            let thumb = BorderBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_width(1.0)
                    .with_height(1.0)
                    .with_background(
                        Brush::Solid(Color::from_rgba(thumb_c.0, thumb_c.1, thumb_c.2, thumb_c.3))
                            .into(),
                    ),
            )
            .with_stroke_thickness(Thickness::uniform(0.0).into())
            .with_corner_radius(999.0_f32.into())
            .build(&mut ui.build_ctx())
            .transmute();
            let label = TextBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_width(1.0)
                    .with_height(1.0)
                    .with_foreground(Brush::Solid(Color::from_rgba(220, 215, 200, 255)).into()),
            )
            .with_text(String::new())
            .with_font_size(18.0.into())
            .with_horizontal_text_alignment(HorizontalAlignment::Left)
            .with_vertical_text_alignment(VerticalAlignment::Center)
            .build(&mut ui.build_ctx());
            let pct_text = TextBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_width(1.0)
                    .with_height(1.0)
                    .with_foreground(Brush::Solid(Color::from_rgba(220, 215, 200, 255)).into()),
            )
            .with_text(String::new())
            .with_font_size(18.0.into())
            .with_horizontal_text_alignment(HorizontalAlignment::Left)
            .with_vertical_text_alignment(VerticalAlignment::Center)
            .build(&mut ui.build_ctx());
            SettingsSlider {
                track,
                fill,
                thumb,
                icon,
                label,
                pct_text,
                track_rect: UiRect::default(),
            }
        };
        // 音樂：深藍 icon + 藍色 fill；音效：灰藍 icon + 藍色 fill；速度：棕黃 icon + 綠色 fill
        self.ui_settings.music = make_slider(
            ui,
            (55, 115, 185, 255),
            (70, 150, 220, 200),
            (85, 175, 240, 255),
        );
        self.ui_settings.sfx = make_slider(
            ui,
            (55, 115, 185, 255),
            (70, 150, 220, 200),
            (85, 175, 240, 255),
        );
        self.ui_settings.speed = make_slider(
            ui,
            (120, 90, 40, 255),
            (80, 190, 70, 200),
            (85, 175, 240, 255),
        );
        // Placeholder buttons (6) — rounded square icon area + sublabel below
        for _ in 0..6 {
            let bg = BorderBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_width(1.0)
                    .with_height(1.0)
                    .with_background(Brush::Solid(Color::from_rgba(55, 130, 210, 255)).into()),
            )
            .with_stroke_thickness(Thickness::uniform(0.0).into())
            .with_corner_radius(14.0_f32.into())
            .build(&mut ui.build_ctx())
            .transmute();
            let label = TextBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_width(1.0)
                    .with_height(1.0)
                    .with_foreground(Brush::Solid(Color::from_rgba(255, 255, 255, 255)).into()),
            )
            .with_text(String::new())
            .with_font_size(20.0.into())
            .with_horizontal_text_alignment(HorizontalAlignment::Center)
            .with_vertical_text_alignment(VerticalAlignment::Center)
            .build(&mut ui.build_ctx());
            let sublabel = TextBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                    .with_width(1.0)
                    .with_height(1.0)
                    .with_foreground(Brush::Solid(Color::from_rgba(220, 225, 230, 255)).into()),
            )
            .with_text(String::new())
            .with_font_size(16.0.into())
            .with_horizontal_text_alignment(HorizontalAlignment::Center)
            .with_vertical_text_alignment(VerticalAlignment::Top)
            .build(&mut ui.build_ctx());
            self.ui_settings
                .placeholder_btns
                .push(SettingsPlaceholderBtn {
                    bg,
                    label,
                    sublabel,
                });
        }

        // 個人統計數據頁（Profile）初始化 — 版面骨架，資料為 placeholder。
        {
            fn new_border(
                ui: &mut UserInterface,
                bg_rgba: (u8, u8, u8, u8),
                corner: f32,
            ) -> Handle<UiNode> {
                BorderBuilder::new(
                    WidgetBuilder::new()
                        .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                        .with_width(1.0)
                        .with_height(1.0)
                        .with_background(
                            Brush::Solid(Color::from_rgba(
                                bg_rgba.0, bg_rgba.1, bg_rgba.2, bg_rgba.3,
                            ))
                            .into(),
                        ),
                )
                .with_stroke_thickness(Thickness::uniform(0.0).into())
                .with_corner_radius(corner.into())
                .build(&mut ui.build_ctx())
                .transmute()
            }
            fn new_text(
                ui: &mut UserInterface,
                fg_rgba: (u8, u8, u8, u8),
                font_size: f32,
                h_align: HorizontalAlignment,
            ) -> Handle<Text> {
                TextBuilder::new(
                    WidgetBuilder::new()
                        .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                        .with_width(1.0)
                        .with_height(1.0)
                        .with_foreground(
                            Brush::Solid(Color::from_rgba(
                                fg_rgba.0, fg_rgba.1, fg_rgba.2, fg_rgba.3,
                            ))
                            .into(),
                        ),
                )
                .with_text(String::new())
                .with_font_size(font_size.into())
                .with_horizontal_text_alignment(h_align)
                .with_vertical_text_alignment(VerticalAlignment::Center)
                .build(&mut ui.build_ctx())
            }

            const WHITE: (u8, u8, u8, u8) = (255, 255, 255, 255);
            const DARK_TEXT: (u8, u8, u8, u8) = (55, 32, 12, 255);

            self.ui_profile.bg = new_border(ui, (22, 42, 30, 255), 0.0);
            self.ui_profile.back_btn = new_border(ui, (65, 158, 218, 255), 12.0);
            self.ui_profile.back_btn_text =
                new_text(ui, WHITE, 22.0, HorizontalAlignment::Center);

            self.ui_profile.header_bg = new_border(ui, (58, 46, 28, 255), 14.0);
            self.ui_profile.avatar_bg = new_border(ui, (60, 130, 200, 255), 999.0);
            self.ui_profile.avatar_text = new_text(ui, WHITE, 34.0, HorizontalAlignment::Center);
            self.ui_profile.name_text = new_text(ui, WHITE, 26.0, HorizontalAlignment::Left);
            self.ui_profile.level_badge_bg = new_border(ui, (245, 195, 40, 255), 8.0);
            self.ui_profile.level_text =
                new_text(ui, DARK_TEXT, 16.0, HorizontalAlignment::Center);
            self.ui_profile.xp_track = new_border(ui, (42, 32, 18, 255), 999.0);
            self.ui_profile.xp_fill = new_border(ui, (80, 190, 70, 255), 999.0);
            self.ui_profile.followers_text =
                new_text(ui, (220, 225, 230, 255), 18.0, HorizontalAlignment::Left);
            self.ui_profile.visibility_toggle_bg = new_border(ui, (60, 180, 80, 255), 8.0);
            self.ui_profile.visibility_text =
                new_text(ui, WHITE, 18.0, HorizontalAlignment::Center);
            self.ui_profile.publish_btn_bg = new_border(ui, (55, 130, 210, 255), 10.0);
            self.ui_profile.publish_btn_text =
                new_text(ui, WHITE, 18.0, HorizontalAlignment::Center);
            self.ui_profile.settings_gear_bg = new_border(ui, (110, 72, 30, 255), 12.0);
            self.ui_profile.settings_gear_text =
                new_text(ui, WHITE, 26.0, HorizontalAlignment::Center);

            self.ui_profile.medals_title_text =
                new_text(ui, WHITE, 24.0, HorizontalAlignment::Left);
            self.ui_profile.medals.clear();
            for _ in 0..PROFILE_MEDALS.len() {
                let bg = new_border(ui, (195, 150, 80, 255), 10.0);
                let count_text = new_text(ui, WHITE, 15.0, HorizontalAlignment::Center);
                self.ui_profile
                    .medals
                    .push(ProfileMedalUi { bg, count_text });
            }

            self.ui_profile.heroes_banner_bg = new_border(ui, (200, 55, 45, 255), 6.0);
            self.ui_profile.heroes_banner_text =
                new_text(ui, WHITE, 20.0, HorizontalAlignment::Center);
            self.ui_profile.heroes.clear();
            for _ in 0..PROFILE_TOP_HEROES.len() {
                let bg = new_border(ui, (125, 105, 82, 230), 10.0);
                let name_text = new_text(ui, WHITE, 16.0, HorizontalAlignment::Center);
                let count_text =
                    new_text(ui, (220, 225, 230, 255), 14.0, HorizontalAlignment::Center);
                self.ui_profile.heroes.push(ProfilePortraitUi {
                    bg,
                    name_text,
                    count_text,
                });
            }

            self.ui_profile.monkeys_banner_bg = new_border(ui, (200, 55, 45, 255), 6.0);
            self.ui_profile.monkeys_banner_text =
                new_text(ui, WHITE, 20.0, HorizontalAlignment::Center);
            self.ui_profile.monkeys.clear();
            for _ in 0..PROFILE_TOP_MONKEYS.len() {
                let bg = new_border(ui, (125, 105, 82, 230), 10.0);
                let name_text = new_text(ui, WHITE, 16.0, HorizontalAlignment::Center);
                let count_text =
                    new_text(ui, (220, 225, 230, 255), 14.0, HorizontalAlignment::Center);
                self.ui_profile.monkeys.push(ProfilePortraitUi {
                    bg,
                    name_text,
                    count_text,
                });
            }

            self.ui_profile.stats_header_bg = new_border(ui, (110, 72, 30, 255), 10.0);
            self.ui_profile.stats_header_text =
                new_text(ui, WHITE, 20.0, HorizontalAlignment::Left);
            self.ui_profile.stats_collapse_text =
                new_text(ui, WHITE, 20.0, HorizontalAlignment::Center);
            self.ui_profile.stats_share_count_text =
                new_text(ui, (240, 230, 200, 255), 16.0, HorizontalAlignment::Right);
            self.ui_profile.stats_rows.clear();
            for _ in 0..PROFILE_STAT_ROWS.len() {
                let label_text =
                    new_text(ui, (220, 215, 200, 255), 17.0, HorizontalAlignment::Left);
                let value_text = new_text(ui, WHITE, 17.0, HorizontalAlignment::Right);
                let checkbox_bg = new_border(ui, (70, 150, 220, 200), 4.0);
                self.ui_profile.stats_rows.push(ProfileStatRowUi {
                    label_text,
                    value_text,
                    checkbox_bg,
                });
            }
        }

        apply_frontend_runtime_env_from_config();
        self.connection_status = ConnectionStatus::Disconnected;
        if frontend_env_truthy("OMFX_LEGACY_AUTOSTART") {
            if let Some(selection) = self.default_session_selection() {
                log::info!("OMFX_LEGACY_AUTOSTART enabled; starting default pregame session");
                if let Err(err) = self.start_game_session(selection) {
                    log::error!("legacy autostart failed: {}", err);
                    self.pregame_runtime.recover_to_difficulty(err.clone());
                    self.connection_status = ConnectionStatus::Failed(err);
                }
            }
        }

        // 階段 5.1：遺留 NetworkBridge/EventBuffer/network_entities/
        // client_projectiles 狀態不再初始化－消費者消失了。
        // （現場聲明仍需在同一階段 5.1 剪輯中刪除。）

        // 背景音樂
        if let Some(buffer) = load_sound_from_path("data/music/bgm.ogg") {
            let scene = &mut context.scenes[self.scene];
            let bgm_node = SoundBuilder::new(BaseBuilder::new())
                .with_buffer(Some(buffer))
                .with_looping(true)
                .with_status(Status::Playing)
                .with_gain(self.settings_music_volume)
                .build_node();
            self.bgm_handle = scene.graph.add_node(bgm_node);
            log::info!(
                "BGM node added, handle valid: {}",
                !self.bgm_handle.is_none()
            );
        }

        // 音效
        self.sfx_button_click = load_sound_from_path("data/sfx/button_click.wav");
        self.sfx_tower_place = load_sound_from_path("data/sfx/tower_place.wav");
        self.sfx_cookie_crunch = load_sound_from_path("data/sfx/cookie_crunch.wav");

        self.hotkeys = hotkeys::HotkeyConfig::load();
        self.hotkey_panel_visible = false;
        self.hotkey_rebinding = None;

        // 英雄知識：從 omb/player_profile.json 載入 KP 快取
        self.load_gk_profile();

        Ok(())
    }

    fn on_deinit(&mut self, _context: PluginContext) -> GameResult {
        self.shutdown_game_session(false);
        Ok(())
    }

    fn update(&mut self, context: &mut PluginContext) -> GameResult {
        let scene = &mut context.scenes[self.scene];
        let frame_span = tracing::trace_span!(
            "omfx::Plugin::update",
            perfetto = true,
            tick = self.current_sim_tick,
            network_entities = self.network_entities.len(),
            projectiles = self.client_projectiles.len(),
            draw_calls = self.frame_profile.last_draw_calls,
            triangles = self.frame_profile.last_triangles,
            deep = perfetto_deep_enabled(),
        )
        .entered();
        // 每 frame 清掉 drawing_context 的 line buffer，避免累積到無限大導致 FPS 為 0。
        // 後續 phase（爆炸 / 路徑 debug 等）會 push 新的 line 進來。
        scene.drawing_context.clear_lines();

        // 每 frame 同步 BGM 音量（透過 primary audio bus）
        scene
            .graph
            .sound_context
            .state()
            .bus_graph_mut()
            .primary_bus_mut()
            .set_gain(self.settings_music_volume * 2.0);

        // 播放待定音效（由 on_os_event 觸發）
        let pending_sfx: Vec<SfxKind> = std::mem::take(&mut self.sfx_pending);
        for kind in pending_sfx {
            self.play_sfx(scene, kind);
        }
        // 每 60 幀清理一次已停止的一次性音效節點
        if self.current_sim_tick % 60 == 0 {
            self.cleanup_sfx_one_shots(scene);
        }

        // 套用設定頁請求的顯示模式切換（必須在 pregame 早退之前，
        // 因為設定頁本身就在 pregame 模式）
        if self.pending_display_mode.is_some() {
            if let fyrox::engine::GraphicsContext::Initialized(ref gc) = context.graphics_context {
                if let Some(mode) = self.pending_display_mode.take() {
                    match mode {
                        DisplayModeRequest::Fullscreen => {
                            // 記住目前視窗尺寸，Esc 退出全螢幕時還原
                            self.last_windowed_size =
                                (self.window_size.x as u32, self.window_size.y as u32);
                            gc.window
                                .set_fullscreen(Some(fyrox::window::Fullscreen::Borderless(None)));
                            self.display_fullscreen = true;
                            log::info!("顯示模式已套用: Fullscreen");
                        }
                        DisplayModeRequest::Windowed(w, h) => {
                            gc.window.set_fullscreen(None);
                            let _ = gc
                                .window
                                .request_inner_size(fyrox::dpi::PhysicalSize::new(w, h));
                            self.display_fullscreen = false;
                            log::info!("顯示模式已套用: {}x{}", w, h);
                        }
                    }
                }
            }
        }

        if self.session_render_reset_pending {
            self.reset_session_render_state(scene);
            self.session_render_reset_pending = false;
        }
        if !self.pending_label_deletions.is_empty() {
            let ui = context.user_interfaces.first_mut();
            for label in self.pending_label_deletions.drain(..) {
                ui.send(label, WidgetMessage::Remove);
            }
        }

        if self.pregame_runtime.is_pregame() {
            let ui = context.user_interfaces.first_mut();
            self.update_pregame_ui(ui);
            drop(frame_span);
            return Ok(());
        }
        {
            let ui = context.user_interfaces.first_mut();
            self.hide_pregame_ui(ui);
            self.update_in_game_return_button(ui);
        }
        if self.game_ended {
            log::info!("game ended; tearing down active session");
            self.shutdown_game_session(false);
            self.pregame_runtime.state = pregame::PregameState::SessionEnded;
            drop(frame_span);
            return Ok(());
        }
        let frame_t0 = std::time::Instant::now();
        let frame_interval = self
            .render_pacing_last_frame_at
            .map(|prev| frame_t0.duration_since(prev));
        let last_rendered_snapshot_tick = self.render_pacing_last_snapshot_tick;

        // 煙環鉤子（第一次更新時讀取一次）。兩個環境變數都是
        // 獨立的;可以設定其中之一或兩者。供自動化測試使用
        // 執行，以便可以啟動單一「run.bat」→按開始回合→
        // 退出，無需人工點擊按鈕。
        let auto_hooks_span =
            tracing::trace_span!("omfx::frame::auto_hooks", perfetto = true).entered();
        let now = std::time::Instant::now();
        if self.auto_clock_start.is_none() {
            self.auto_clock_start = Some(now);
        }
        let elapsed_s = now
            .duration_since(self.auto_clock_start.unwrap())
            .as_secs_f32();
        if !self.auto_start_sent {
            if let Some((raw, threshold)) = frontend_config_f32(
                "OMFX_AUTO_START_AFTER_SEC",
                "client",
                "AUTO_START_AFTER_SEC",
            ) {
                if elapsed_s >= threshold {
                    let input = omoba_core::kcp::game_proto::PlayerInput {
                        action: Some(
                            omoba_core::kcp::game_proto::player_input::Action::StartRound(
                                omoba_core::kcp::game_proto::StartRound {},
                            ),
                        ),
                    };
                    let origin_us = wall_clock_us();
                    self.send_lockstep_input_from(
                        input,
                        lockstep_client::InputOriginKind::Auto,
                        origin_us,
                    );
                    log::info!(
                        "[auto-smoke] Start Round sent at t={:.2}s (AUTO_START_AFTER_SEC={})",
                        elapsed_s,
                        raw
                    );
                    self.auto_start_sent = true;
                }
            }
        }
        if let Some((_raw, interval_ms)) =
            frontend_config_f32("OMFX_AUTO_NOOP_EVERY_MS", "client", "AUTO_NOOP_EVERY_MS")
        {
            if interval_ms > 0.0 {
                let start_after_s = frontend_config_f32(
                    "OMFX_AUTO_NOOP_START_AFTER_SEC",
                    "client",
                    "AUTO_NOOP_START_AFTER_SEC",
                )
                .map(|(_, v)| v)
                .or_else(|| {
                    frontend_config_f32(
                        "OMFX_AUTO_START_AFTER_SEC",
                        "client",
                        "AUTO_START_AFTER_SEC",
                    )
                    .map(|(_, v)| v)
                })
                .unwrap_or(0.0);
                let interval_s = interval_ms / 1000.0;
                if elapsed_s < start_after_s {
                    self.auto_noop_next_at_s = Some(start_after_s + interval_s);
                } else {
                    let next_at = self
                        .auto_noop_next_at_s
                        .unwrap_or(start_after_s + interval_s);
                    if elapsed_s >= next_at {
                        let input = omoba_core::kcp::game_proto::PlayerInput {
                            action: Some(omoba_core::kcp::game_proto::player_input::Action::NoOp(
                                omoba_core::kcp::game_proto::NoOp {},
                            )),
                        };
                        let origin_us = wall_clock_us();
                        self.send_lockstep_input_from(
                            input,
                            lockstep_client::InputOriginKind::Auto,
                            origin_us,
                        );
                        self.auto_noop_next_at_s = Some(next_at + interval_s);
                    } else {
                        self.auto_noop_next_at_s = Some(next_at);
                    }
                }
            }
        }
        if let Some((raw, threshold)) =
            frontend_config_f32("OMFX_AUTO_EXIT_AFTER_SEC", "client", "AUTO_EXIT_AFTER_SEC")
        {
            if elapsed_s >= threshold {
                log::info!(
                    "[auto-smoke] exiting at t={:.2}s (AUTO_EXIT_AFTER_SEC={})",
                    elapsed_s,
                    raw
                );
                std::process::exit(0);
            }
        }
        drop(auto_hooks_span);

        // 延遲初始化在第一幀上共享 sprite 資源。
        if self.sprite_resources.is_none() {
            self.sprite_resources = Some(sprite_resources::SharedSpriteResources::new());
        }

        // Lazy init batched meshes — 4 個獨立 batch（body / hp / facing），N entity = 3 draws。
        if self.body_batch.is_none() {
            let material = self.sprite_resources.as_ref().unwrap().material.clone();
            self.body_batch = Some(sprite_resources::BatchedSpriteMesh::new(
                scene,
                STRESS_SAFE_BODY_BATCH_CAPACITY,
                material,
            ));
        }
        if self.hp_batch.is_none() {
            let material = self.sprite_resources.as_ref().unwrap().material.clone();
            // 2 slots per entity: background + foreground.
            self.hp_batch = Some(sprite_resources::BatchedSpriteMesh::new(
                scene,
                STRESS_SAFE_HP_BATCH_CAPACITY,
                material,
            ));
        }
        if self.facing_batch.is_none() {
            let material = self.sprite_resources.as_ref().unwrap().material.clone();
            self.facing_batch = Some(sprite_resources::BatchedSpriteMesh::new(
                scene,
                STRESS_SAFE_FACING_BATCH_CAPACITY,
                material,
            ));
        }

        // 階段 5.1：連線狀態/初始視窗排出已移除
        // （追蹤不再存在的舊版 NetworkBridge 握手）。
        // 下面的同步「已連線」事件現在是規範的「我們已經啟動」訊號。

        // 階段 3.3：排出鎖定事件並轉送至 sim_runner。
        // - 連線→推送master_seed（解鎖worker的阻塞recv）
        // - TickBatch → 轉換 transport payload 並推送給 sim_runner，
        // 讓本地 ECS replica 推進一個批次。
        // - StateHash → 僅記錄（階段 3.4 將與
        // sim_runner 用於非同步偵測的本地雜湊）。
        // - 斷開連接 → 記錄。
        // TickBatch 每秒採樣一次，以避免日誌垃圾郵件。
        let mut forwarded_pending_input_ids: Vec<u32> = Vec::new();
        let t_lockstep = std::time::Instant::now();
        let event_drain_span =
            tracing::trace_span!("omfx::frame::lockstep_event_drain", perfetto = true,).entered();
        if let (Some(ref lh), Some(ref sim)) = (
            self.lockstep_handle.as_ref(),
            self.sim_runner_handle.as_ref(),
        ) {
            while let Ok(ev) = lh.events_rx.try_recv() {
                match ev {
                    lockstep_client::LockstepEvent::Connected {
                        master_seed,
                        player_id,
                        step_fps,
                    } => {
                        self.server_step_fps = step_fps;
                        if player_id != self.local_player_id {
                            log::warn!(
                                "[lockstep] server returned player_id={} but local configured player_id={}",
                                player_id,
                                self.local_player_id
                            );
                        }
                        log::info!(
                            "[lockstep] connected master_seed=0x{:016x} player_id={} step_fps={}",
                            master_seed,
                            player_id,
                            step_fps
                        );
                        if let Err(e) = sim.master_seed_tx.send(sim_runner::SimStartMetadata {
                            master_seed,
                            step_fps,
                        }) {
                            log::error!("[lockstep] failed to forward master_seed: {}", e);
                        }
                    }
                    lockstep_client::LockstepEvent::TickBatch {
                        tick,
                        inputs,
                        server_events,
                        lua_content_generation,
                        lua_content_hash,
                    } => {
                        // 階段 4.3：追蹤輸入 target_tick 數學的最新 sim 刻度。
                        self.current_sim_tick = tick;
                        self.current_sim_tick_observed_at = Some(now);
                        if tick % LOCKSTEP_ONE_SECOND_TICKS_U32 == 0 {
                            log::debug!(
                                "[lockstep] tick={} inputs={} events={}",
                                tick,
                                inputs.len(),
                                server_events.len()
                            );
                        }
                        let game_forward_us = wall_clock_us();
                        for input in &inputs {
                            if input.input_id == 0 {
                                continue;
                            }
                            if let Some(pending) = self.pending_inputs.get_mut(&input.input_id) {
                                pending.client_receive_tickbatch_us = Some(input.client_receive_us);
                                pending.game_forward_to_sim_us = Some(game_forward_us);
                                pending.server_receive_tick = Some(input.server_receive_tick);
                                pending.server_drain_tick = Some(input.server_drain_tick);
                                pending.server_queue_us = Some(input.server_queue_us);
                                forwarded_pending_input_ids.push(input.input_id);
                            }
                        }
                        let converted: Vec<sim_runner::TickBatchInput> = inputs
                            .into_iter()
                            .map(|input| sim_runner::TickBatchInput {
                                player_id: input.player_id,
                                input: input.input,
                                input_id: input.input_id,
                                server_receive_tick: input.server_receive_tick,
                                server_drain_tick: input.server_drain_tick,
                                server_queue_us: input.server_queue_us,
                                client_receive_us: input.client_receive_us,
                                game_forward_us,
                            })
                            .collect();
                        let payload = sim_runner::TickBatchPayload {
                            tick,
                            inputs: converted,
                            lua_content_generation,
                            lua_content_hash,
                        };
                        if let Err(e) = sim.tick_input_tx.send(payload) {
                            log::error!("[lockstep] failed to forward tick batch: {}", e);
                        }
                        // server_events：忽略第 3.3 階段； 5+階段將
                        // 將它們路由到 sim 的事件接收器。
                    }
                    lockstep_client::LockstepEvent::StateHash { tick, hash } => {
                        log::info!("[lockstep] state_hash@{}=0x{:016x}", tick, hash);
                        // 第 3.4 階段將將此與
                        // sim_runner 的本地計算雜湊。
                    }
                    lockstep_client::LockstepEvent::NetStats {
                        wire_delta,
                        logical_delta,
                    } => {
                        self.net_wire_bytes_current += wire_delta;
                        self.net_bytes_current += logical_delta;
                    }
                    lockstep_client::LockstepEvent::InputSubmitted {
                        input_id,
                        submit_start_us,
                        submit_done_us,
                    } => {
                        if let Some(pending) = self.pending_inputs.get_mut(&input_id) {
                            pending.submit_start_us = Some(submit_start_us);
                            pending.submit_done_us = Some(submit_done_us);
                        }
                    }
                    lockstep_client::LockstepEvent::Latency { rtt_us } => {
                        self.latest_rtt_us = Some(rtt_us);
                    }
                    lockstep_client::LockstepEvent::Disconnected { reason } => {
                        log::warn!("[lockstep] disconnected: {}", reason);
                    }
                }
            }
        }
        drop(event_drain_span);
        let lockstep_ns = t_lockstep.elapsed().as_nanos();

        // 階段 3.4：讀取最新的 sim 快照並（存根）更新渲染
        // 橋。透過“try_lock”獲取，因此不會出現緩慢的渲染幀
        // 阻止 sim 工作線程——如果鎖被爭用，我們就跳過
        // 此幀並拾取下一個快照。第 4 階段將取代
        // 帶有真實 Fyrox sprite 生成 / 更新 / 的存根「更新」主體
        // despawn，退休 NetworkBridge GameEvent → sprite pipeline
        // 下面是 SIM 權威擁有的實體。
        let mut applied_inputs_to_pair: Option<Vec<sim_runner::AppliedInputMeta>> = None;
        let t_snapshot = std::time::Instant::now();
        let mut render_bridge_ns: u128 = 0;
        let snapshot_span =
            tracing::trace_span!("omfx::frame::snapshot_consumption", perfetto = true,).entered();
        let sim_state_for_frame = if let Some(ref sim) = self.sim_runner_handle {
            self.wait_for_applied_input_render_data(sim, &forwarded_pending_input_ids);
            Some(sim.state.clone())
        } else {
            None
        };
        if let Some(sim_state) = sim_state_for_frame {
            if let Ok(snapshot) = sim_state.try_lock() {
                if self.sim_dev_lua_reload_error != snapshot.dev_lua_reload_error {
                    self.sim_dev_lua_reload_error = snapshot.dev_lua_reload_error.clone();
                    if let Some(err) = &self.sim_dev_lua_reload_error {
                        log::error!("DEV Lua reload error surfaced to frontend: {}", err);
                    }
                }

                let content_changed = self.sim_lua_content_generation
                    != snapshot.lua_content_generation
                    || self.sim_lua_content_hash != snapshot.lua_content_hash;
                if content_changed {
                    self.invalidate_lua_content_caches(
                        scene,
                        snapshot.lua_content_generation,
                        &snapshot.lua_content_hash,
                    );
                    self.sim_lua_content_generation = snapshot.lua_content_generation;
                    self.sim_lua_content_hash = snapshot.lua_content_hash.clone();
                }

                let runtime_tick_changed =
                    self.render_pacing_last_snapshot_tick != Some(snapshot.tick);
                if runtime_tick_changed {
                    let t_render_bridge = std::time::Instant::now();
                    self.render_bridge.update(&*snapshot, scene);
                    render_bridge_ns += t_render_bridge.elapsed().as_nanos();
                    applied_inputs_to_pair = Some(snapshot.applied_input_meta.clone());
                    self.latest_entities = snapshot.entities.clone();
                    self.tower_ability_bar_interaction.authoritative_boundary();
                    let owned_ability_count = snapshot
                        .entities
                        .iter()
                        .filter(|entity| {
                            entity.owner_player_id == Some(self.local_player_id)
                                && entity.hp > 0
                                && entity.tower_active_ability.is_some()
                        })
                        .count();
                    let page_model = ability_bar_page_model(
                        owned_ability_count,
                        self.tower_ability_bar_interaction.page,
                    );
                    self.tower_ability_bar_interaction.page = page_model.page;
                    self.tower_ability_bar_items = ability_bar_items_with_names(
                        &snapshot.entities,
                        &self.tower_ability_bar_tower_names,
                        self.local_player_id,
                        self.tower_ability_bar_interaction.page,
                        0.0,
                    );
                    self.tower_ability_bar_snapshot_at = Some(now);
                    self.tower_ability_bar_rejection.observe_results(
                        &snapshot.tower_ability_cast_results,
                        self.local_player_id,
                        now,
                    );
                    self.tower_ability_bar_rejection.retain_visible(
                        &ability_bar_available_keys(&snapshot.entities, self.local_player_id),
                        now,
                    );

                    // 階段 5.x：HUD 心跳源自 sim 快照
                    // （NetworkBridge GameEvent 串流在第 5.1 階段被刪除；這
                    // 恢復頂線上的蜱/實體/英雄/小兵計數
                    // 英雄面板上的狀態文字和 hp / max_hp）。
                    self.heartbeat.tick = snapshot.tick as u64;
                    self.render_pacing_last_snapshot_tick = Some(snapshot.tick);
                    self.update_sim_speed(snapshot.tick);
                    // sim_runner 以共享 lockstep cadence 運作。
                    self.heartbeat.game_time = self.ticks_to_seconds(snapshot.tick);
                    self.heartbeat.entity_count = snapshot.entities.len() as u64;
                    self.heartbeat.hero_count = snapshot
                        .entities
                        .iter()
                        .filter(|e| matches!(e.kind, sim_runner::EntityKind::Hero))
                        .count() as u64;
                    self.heartbeat.creep_count = snapshot
                        .entities
                        .iter()
                        .filter(|e| matches!(e.kind, sim_runner::EntityKind::Creep))
                        .count() as u64;

                    // 階段 3.2：來自 sim 快照的 TD HUD 狀態。取代了
                    // 遺留的 NetworkBridge `apply_event` 寫入被切入
                    // 階段 5.1，使這些欄位保持預設狀態。開始
                    // 圓形按鈕文字 + LIVES 頂行都顯示這些內容。
                    self.current_round = snapshot.round;
                    self.total_rounds = snapshot.total_rounds;
                    self.round_is_running = snapshot.round_is_running;
                    self.is_game_paused = snapshot.is_paused;
                    self.game_speed_multiplier = snapshot.game_speed_multiplier;
                    if self.round_is_running {
                        self.td_auto_start_sent_for_idle_round = false;
                    }
                    if td_should_auto_start_round(
                        self.td_auto_start_enabled,
                        self.td_auto_start_sent_for_idle_round,
                        self.is_game_paused,
                        self.round_is_running,
                        self.current_round,
                        self.total_rounds,
                    ) {
                        let input = omoba_core::kcp::game_proto::PlayerInput {
                            action: Some(
                                omoba_core::kcp::game_proto::player_input::Action::StartRound(
                                    omoba_core::kcp::game_proto::StartRound {},
                                ),
                            ),
                        };
                        self.send_lockstep_input_from(
                            input,
                            lockstep_client::InputOriginKind::Auto,
                            wall_clock_us(),
                        );
                        self.td_auto_start_sent_for_idle_round = true;
                        log::info!("Auto Start Round → lockstep PlayerInput::StartRound sent");
                    }
                    self.hero_state.lives = snapshot.lives;
                    if td_mode_from_snapshot(snapshot.lives, snapshot.total_rounds) {
                        self.is_td_mode = true;
                    }

                    // 遊戲結束偵測：從 sim 快照推導 defeat / victory 條件。
                    // 舊版 GameEvent game_end 處理器在第 5.1 階段被移除，
                    // 因此改由快照欄位判斷：
                    //   Defeat  — lives <= 0（TD 模式下生命耗盡）
                    //   Victory — round >= total_rounds 且目前波次已結束
                    if !self.game_ended && snapshot.tick > 0 && snapshot.total_rounds > 0 {
                        let is_defeat = snapshot.lives <= 0;
                        let is_victory = snapshot.round >= snapshot.total_rounds
                            && !snapshot.round_is_running;
                        if is_defeat || is_victory {
                            log::info!(
                                "[game_end] defeat={} victory={} lives={} round={}/{} running={}",
                                is_defeat,
                                is_victory,
                                snapshot.lives,
                                snapshot.round,
                                snapshot.total_rounds,
                                snapshot.round_is_running,
                            );
                            self.game_ended = true;
                        }
                    }

                    // 階段 4.2：將模擬爆炸排入本地
                    // `active_explosions` 環形緩衝區。按刻度進行重複資料刪除
                    // 重新讀取相同快照的渲染幀不會
                    // 產生重複的環。環生命週期由下列因素驅動
                    // omfx 掛鐘（`elapsed += dt`）所以爆炸
                    // 動畫以獨立於 sim 的渲染速率運行
                    // 滴答率。
                    if !snapshot.explosions.is_empty()
                        && self.sim_last_explosion_tick != Some(snapshot.tick)
                    {
                        for ex in &snapshot.explosions {
                            // ActiveExplosion 儲存**未翻轉**渲染座標
                            // （後端 × WORLD_SCALE）。渲染路徑位於~第 2136 行
                            // 在餵食 Fyrox 時應用單一「-x」翻轉
                            // SceneDrawingContext（匹配 build_line_segment /
                            // add_circle_lines 約定）。此處預翻
                            // 會雙翻轉→鏡像爆炸位置。
                            let render_pos =
                                Vector2::new(ex.pos_x * WORLD_SCALE, ex.pos_y * WORLD_SCALE);
                            let max_radius = ex.radius * WORLD_SCALE;
                            let duration = (ex.duration_ms as f32 / 1000.0).max(0.05);
                            self.active_explosions.push(ActiveExplosion {
                                pos: render_pos,
                                max_radius,
                                duration,
                                elapsed: 0.0,
                            });
                        }
                        self.sim_last_explosion_tick = Some(snapshot.tick);
                    }

                    // TD 塔建立選單：種子 `td_template_order` + `td_templates`
                    // 來自第一個非空收據上的 snapshot.tower_templates。
                    // 階段 5.1 刪除了使用的遺留「tower_templates」遊戲事件
                    // 透過 apply_event 填充這些；右側建置選單
                    // 卡在 0 個按鈕上。首次建置後靜態（註冊表為
                    // 不可變的後腳本 DLL 載入），因此 !is_empty 防護運行
                    // 作為一擊。
                    if self.td_template_order.is_empty() && !snapshot.tower_templates.is_empty() {
                        for t in snapshot.tower_templates.iter() {
                            self.td_template_order.push(t.unit_id.clone());
                            self.tower_ability_bar_tower_names
                                .insert(t.unit_id.clone(), t.label.clone());
                            self.td_templates
                                .insert(t.unit_id.clone(), td_template_from_snapshot(t));
                        }
                        let layout_placeholder = if self.td_templates.contains_key("tower_dart") {
                            Some("tower_dart".to_string())
                        } else {
                            self.td_template_order.first().cloned()
                        };
                        if let Some(layout_placeholder) = layout_placeholder {
                            while self.td_template_order.len() < TD_SHOP_LAYOUT_DEBUG_MIN_CARDS {
                                self.td_template_order.push(layout_placeholder.clone());
                            }
                        }
                        log::info!(
                            "TD build menu seeded: {} snapshot towers, {} displayed cards",
                            snapshot.tower_templates.len(),
                            self.td_template_order.len()
                        );
                    }

                    // 從 snapshot.tower_upgrades 上種子 `td_upgrade_defs` 緩存
                    // 第一張非空收據。銷售按鈕退款+升級按鈕
                    // 文字均從此處讀取。首次建置後靜態（登錄
                    // 是不可變的後腳本 DLL 載入）。
                    if self.td_upgrade_defs.is_empty() && !snapshot.tower_upgrades.is_empty() {
                        for d in snapshot.tower_upgrades.iter() {
                            self.td_upgrade_defs.insert(
                                (d.tower_kind.clone(), d.path, d.level),
                                (d.name.clone(), d.description.clone(), d.cost),
                            );
                        }
                        log::info!(
                            "TD upgrade defs seeded: {} entries from snapshot",
                            self.td_upgrade_defs.len()
                        );
                    }

                    // 將 Mirror Tower 實體從快照轉換為“network_entities”
                    // 所以塔選擇/出售/升級用戶界面（上面寫著
                    // `network_entities`) 在遺留後繼續工作
                    // GameEvent 路径被切断。僅鏡像塔條目 —
                    // 選擇/出售/升級UI是唯一的消費者
                    // 仍然查詢這張地圖。
                    {
                        use std::collections::HashSet;
                        let mut alive_towers: HashSet<u32> = HashSet::new();
                        for e in snapshot.entities.iter() {
                            if !matches!(e.kind, sim_runner::EntityKind::Tower) {
                                continue;
                            }
                            alive_towers.insert(e.entity_id);
                            let tower_kind = if e.unit_id.is_empty() {
                                None
                            } else {
                                Some(e.unit_id.clone())
                            };
                            let (footprint_backend, template_range_backend) = tower_kind
                                .as_deref()
                                .and_then(|uid| self.td_templates.get(uid))
                                .map(|t| (t.footprint_backend, t.range_backend))
                                .unwrap_or((0.4, 0.0));
                            let range_backend = if e.attack_range > 0.0 {
                                e.attack_range
                            } else {
                                template_range_backend
                            };
                            let pos = Vector2::new(e.pos_x * WORLD_SCALE, e.pos_y * WORLD_SCALE);
                            let entry = self
                                .network_entities
                                .entry(e.entity_id)
                                .or_insert_with(NetworkEntity::default);
                            entry.entity_type = "tower".to_string();
                            entry.position = pos;
                            entry.tower_kind = tower_kind;
                            entry.owner_player_id = e.owner_player_id;
                            entry.upgrade_levels = e.upgrade_levels.unwrap_or([0; 3]);
                            entry.tower_pops = e.tower_pops.unwrap_or(0);
                            entry.tower_atk = e.tower_atk.unwrap_or(0.0);
                            entry.tower_asd = e.tower_asd.unwrap_or(0.0);
                            entry.tower_target_priority = e.tower_target_priority.clone();
                            entry.collision_radius_render = footprint_backend * WORLD_SCALE;
                            entry.attack_range_backend = range_backend;
                        }
                        self.network_entities.retain(|id, ent| {
                            ent.entity_type != "tower" || alive_towers.contains(id)
                        });
                    }

                    // 階段 4.5：AbilityRegistry→ability_info_map。靜止的
                    // 在第一個非空弧之後；僅種子缺失條目
                    // 所以任何後端推送的AbilityInfo (cooldown / mana_cost
                    // 不在登錄中的陣列）不會被破壞。這
                    // display_name/max_level/icon路徑涵蓋了基本
                    // 工具提示路徑；沒有時冷卻查找回落到 0
                    // 存在條目（現有 UI 可以優雅地處理該條目）。
                    if !snapshot.abilities.is_empty() {
                        for def in snapshot.abilities.iter() {
                            let entry = self
                                .ability_info_map
                                .entry(def.ability_id.clone())
                                .or_insert_with(|| AbilityInfo {
                                    id: def.ability_id.clone(),
                                    ..Default::default()
                                });
                            if entry.name.is_empty() {
                                entry.name = def.display_name.clone();
                            }
                            if !def.icon_path.is_empty() && entry.icon_path != def.icon_path {
                                entry.icon_path = def.icon_path.clone();
                            }
                            if entry.max_level == 0 {
                                entry.max_level = def.max_level as i32;
                            }
                        }
                    }

                    // 階段 4.1：渲染座標中的 BlockedRegion 多邊形
                    // 放置驗證（下面檢查“circle_hits_polygon”）。
                    // 世界初始化後靜態，但便宜（〜少數區域最大）；
                    // 覆蓋每個刻度而不是髒標誌追蹤。
                    if !snapshot.blocked_regions.is_empty()
                        && self.td_regions_render.len() != snapshot.blocked_regions.len()
                    {
                        self.td_regions_render = snapshot
                            .blocked_regions
                            .iter()
                            .map(|r| {
                                r.points
                                    .iter()
                                    // Placement checks run in logical frontend world coordinates.
                                    // Scene rendering mirrors X separately, while mouse picking has
                                    // already flipped render X back to logical X.
                                    .map(|(x, y)| Vector2::new(x * WORLD_SCALE, y * WORLD_SCALE))
                                    .collect()
                            })
                            .collect();
                    }
                    // 階段 3.x：渲染座標中的 TD 路徑檢查點
                    // `point_segment_dist_sq` 放置檢查。同樣的一擊
                    // 作為區域的人口格局。
                    if !snapshot.paths.is_empty()
                        && self.td_paths_render.len() != snapshot.paths.len()
                    {
                        self.td_paths_render = snapshot
                            .paths
                            .iter()
                            .map(|p| {
                                p.iter()
                                    // Keep path collision data in the same logical coordinate space
                                    // as mouse_world_pos and outbound backend positions.
                                    .map(|(x, y)| Vector2::new(x * WORLD_SCALE, y * WORLD_SCALE))
                                    .collect()
                            })
                            .collect();
                    }

                    // 本地玩家 owns 的英雄實體驅動英雄面板。現在實體渲染數據
                    // 攜帶英雄元資料（名稱/頭銜/等級/經驗值/金幣/
                    // 力量/敏捷/智力/主要屬性）所以
                    // 面板可以以與舊版 NetworkBridge 路徑相同的方式呈現。
                    let local_hero = snapshot
                        .entities
                        .iter()
                        .find(|e| {
                            matches!(e.kind, sim_runner::EntityKind::Hero)
                                && e.owner_player_id == Some(self.local_player_id)
                        })
                        .or_else(|| {
                            snapshot
                                .entities
                                .iter()
                                .find(|e| matches!(e.kind, sim_runner::EntityKind::Hero))
                        });
                    if let Some(hero) = local_hero {
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
                        self.hero_state.entity_id = Some(hero.entity_id);

                        // 階段 3.3：源自 sim 的派生英雄統計數據
                        // 聚合 (HeroStatsExt) — 取代舊版
                        // omb `hero.stats` 0.3s 廣播階段 5.1
                        // 切。鏡像 omb `build_hero_stats_payload` 1:1。
                        if let Some(ext) = hero.hero_ext.as_deref() {
                            self.hero_state.armor = ext.armor;
                            self.hero_state.magic_resist = ext.magic_resist;
                            self.hero_state.move_speed = ext.move_speed;
                            self.hero_state.attack_damage = ext.attack_damage;
                            self.hero_state.attack_interval = ext.attack_speed_sec;
                            self.hero_state.attack_range = ext.attack_range;
                            self.hero_state.bullet_speed = ext.bullet_speed;

                            // BUFF快照重設為權威值
                            // 每個刻度。渲染端每幀倒數計時
                            // （由現有的 buff 計時器代碼處理
                            // 如下）使顯示的秒數保持平滑
                            // 快照之間。
                            self.hero_state.buffs = ext
                                .buffs
                                .iter()
                                .map(|b| LocalBuff {
                                    id: b.buff_id.clone(),
                                    remaining: b.remaining_secs,
                                    payload: serde_json::from_str(&b.payload_json)
                                        .unwrap_or(serde_json::Value::Null),
                                })
                                .collect();

                            // 階段 4.4：快照中的英雄清單。每個
                            // slot 變成 `Some((item_id, cd))` — cd 開始
                            // 為 0，因為快照不包含每個項目
                            // 今天冷卻（Inventory.ItemInstance 有它
                            // 但我們只投影item_id；本地 CD
                            // 每幀遞減 `(_, cd)` 的程式碼
                            // 當 cd=0 時保持無害）。空槽位圖
                            // 為“None”，匹配舊版 UI 合約。
                            // 調整大小為 6，以防英雄狀態為
                            // 早些時候用較小的 Vec 初始化。
                            if self.hero_state.inventory.len() < 6 {
                                self.hero_state.inventory.resize(6, None);
                            }
                            for (i, slot) in ext.inventory.iter().enumerate().take(6) {
                                let prev_cd = self
                                    .hero_state
                                    .inventory
                                    .get(i)
                                    .and_then(Option::as_ref)
                                    .map(|(prev_id, cd)| {
                                        // 僅在相同項目時保留 CD
                                        // 仍在插槽中（否則
                                        // 插槽已交換 — 重設 CD）。
                                        if Some(prev_id.as_str()) == slot.as_deref() {
                                            *cd
                                        } else {
                                            0.0
                                        }
                                    })
                                    .unwrap_or(0.0);
                                self.hero_state.inventory[i] =
                                    slot.as_ref().map(|id| (id.clone(), prev_cd));
                            }

                            // 階段 4.5：能力 ID + 快照等級。
                            // `Hero.bility` (Vec<String>) 驅動
                            // Q/W/E/R 訂單； `ability_levels[i]` 反映了
                            // omb HashMap 投影。本地 `ability_cd` 是
                            // 每個畫面都打勾；我們只播種到 0
                            // 對於新發現的能力 ID，
                            // 飛行中的 CD 不會在每個快照上重置。
                            let new_abilities: Vec<String> = ext
                                .ability_ids
                                .iter()
                                .filter_map(|opt| opt.clone())
                                .collect();
                            if new_abilities != self.hero_state.abilities {
                                self.hero_state.abilities = new_abilities.clone();
                            }
                            self.hero_state.ability_levels.clear();
                            for (i, id_opt) in ext.ability_ids.iter().enumerate() {
                                if let Some(id) = id_opt {
                                    self.hero_state
                                        .ability_levels
                                        .insert(id.clone(), ext.ability_levels[i]);
                                    self.hero_state.ability_cd.entry(id.clone()).or_insert(0.0);
                                }
                            }
                        } else {
                            // 英雄實體存在但聚合缺失
                            // （不應該發生 - UnitStats 路徑始終
                            // 為英雄手臂奔跑）。歸零以避免
                            // 陳舊的遺留價值。
                            self.hero_state.armor = 0.0;
                            self.hero_state.magic_resist = 0.0;
                            self.hero_state.move_speed = 0.0;
                            self.hero_state.attack_damage = 0.0;
                            self.hero_state.attack_interval = 0.0;
                            self.hero_state.attack_range = 0.0;
                            self.hero_state.bullet_speed = 0.0;
                            self.hero_state.buffs.clear();
                        }
                    }
                }
            }
        }
        drop(snapshot_span);
        let snapshot_ns = t_snapshot.elapsed().as_nanos();
        if let Some(inputs) = applied_inputs_to_pair.as_deref() {
            self.pair_applied_inputs(inputs);
        } else {
            self.evict_stale_pending_inputs();
        }
        self.input_latency_meter.maybe_recompute(now);

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

        // 階段5.1：NetworkBridge事件消耗+EventBuffer+心跳hp/pos
        // 協調已刪除。 Lockstep TickBatch（上圖）是唯一的刻度
        // 來源; render_bridge 擁有來自 sim 狀態的 sprite 生成/更新/消失。
        let t_events = std::time::Instant::now();
        let events_drained_local: u64 = 0;
        let events_ns = t_events.elapsed().as_nanos();

        // 4. 插入實體位置（客戶端 lerp）
        let interp_span = tracing::trace_span!(
            "omfx::frame::entity_interpolation_and_batches",
            perfetto = true,
            network_entities = self.network_entities.len(),
        )
        .entered();
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
            // 塔不會移動——跳過 lerp/extrap 所以快照鏡像的
            // 「position」（在快照消費者區塊中設定）保留下來。
            // 如果沒有這個， lerp_duration=0 → NaN 會破壞位置
            // 然後點擊命中測試失敗。
            if entity.entity_type == "tower" {
                continue;
            }
            // 在 PATH_VISIBLE_SECS 之後使蠕動調試路徑過期
            if !entity.path_nodes.is_empty() {
                entity.path_age += dt;
                if entity.path_age >= PATH_VISIBLE_SECS {
                    for seg in entity.path_nodes.drain(..) {
                        scene.graph.remove_node(seg);
                    }
                }
            }

            entity.lerp_elapsed += dt;
            // P4：對於蠕變，速度外推優先於 lerp
            // 具有活動段。在“extrap_duration”過去後，我們鎖定
            // 在`target_position`直到下一個creep.M到達；就是這樣
            // 當伺服器尚未決定時，我們在某個路徑點渲染空閒
            // 下一個尚未完成（例如 TD 路徑末端，因碰撞而阻塞）。
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
                if !pos.x.is_finite()
                    || !pos.y.is_finite()
                    || pos.x.abs() > 5000.0
                    || pos.y.abs() > 5000.0
                {
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
                let pending_dmg = pending_dmg_by_target
                    .get(&entity_id)
                    .copied()
                    .unwrap_or(0.0);
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
                if !matches!(
                    entity.entity_type.as_str(),
                    "hero" | "creep" | "unit" | "tower"
                ) {
                    continue;
                }
                // 顏色依類型區分，方便辨識
                let color = match entity.entity_type.as_str() {
                    "hero" => Color::from_rgba(80, 220, 80, 220),  // 綠
                    "creep" => Color::from_rgba(255, 60, 60, 220), // 紅
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

        self.draw_hero_command_queue_overlay(scene);

        // 階段 5.x：將 sim_runner 支援的實體寫入 body_batch + hp_batch
        // 沖洗前。替換每個實體的 RectangleBuilder 生成
        // 早期的 4.2 render_bridge — 每個實體曾經是單獨的場景
        // 節點 = 單獨的繪製呼叫（1000 個實體→ 3000+ 繪製）。現在的
        // 整個實體集經歷 2-3 個批次網格 = 總共 2-3 個繪製。
        // 每 render frame 累積動畫時間，確保 sim tick guard 不會讓動畫只跑 ~50% 速度。
        self.tower_anim_dt_acc += tower_animation_dt(context.dt, self.is_game_paused);
        self.update_sim_batches(scene, &context.resource_manager, context.dt);

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

        // Fyrox 1.0.1 不會自動更新每幀的分層資料（文件位於
        // fyrox-impl-1.0.1/src/scene/graph/mod.rs:565)。如果沒有這個電話，我們的
        // 3D 網格節點具有過時的 global_transform = 身分 → 所有 sprite
        // 在世界原點 (0,0,0) 處渲染。每幀強制更新一次。
        scene.graph.update_hierarchical_data();

        let interp_ns = t_interp.elapsed().as_nanos();
        drop(interp_span);

        // TD 塔預覽圓圈：選中塔時每 frame 在滑鼠位置重畫 footprint + 攻擊範圍兩圈。
        // 用 SceneDrawingContext（drawing_context 已在 update() 開頭被 clear_lines()），
        // 不再 per-frame 增刪 24+48=72 個 RectangleBuilder node。
        let t_visual = std::time::Instant::now();
        let visual_span =
            tracing::trace_span!("omfx::frame::visual_debug", perfetto = true).entered();
        {
            if let Some(kind) = self.selected_tower_kind.clone() {
                if let Some(tpl) = self.td_templates.get(&kind).cloned() {
                    let placement_radius_render = tower_placement_radius_render(&tpl);
                    let range_backend = tpl.range_backend;
                    let mwp = self.mouse_world_pos;
                    // ===== 本地 placement 驗證（前端即時預覽；後端下最終決定）=====
                    let can_place = self.can_place_tower_at(&tpl, mwp);

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
                    // 內圈：script-owned placement radius
                    add_circle_lines(
                        scene,
                        mwp,
                        placement_radius_render,
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
                // 橘色爆炸圈 (user-requested)：明亮橘黃，由小到大隨時間擴張、淡出。
                let color = Color::from_rgba(255, 150, 30, alpha.max(60));
                if cur_r > 0.02 {
                    let z = Z_REGION - 0.0004;
                    // 起點：θ=0 → (cx + r, cy)；x 翻負與 build_line_segment 對齊
                    let mut prev = Vector3::new(-(ex.pos.x + cur_r), ex.pos.y, z);
                    for k in 1..=SEGS {
                        let theta = (k as f32) * std::f32::consts::TAU / (SEGS as f32);
                        let (s, c) = theta.sin_cos();
                        let next = Vector3::new(-(ex.pos.x + cur_r * c), ex.pos.y + cur_r * s, z);
                        scene.drawing_context.add_line(Line {
                            begin: prev,
                            end: next,
                            color,
                        });
                        prev = next;
                    }
                }
            }
            // 反向刪除以保持 index 有效
            for i in finished_idx.into_iter().rev() {
                self.active_explosions.remove(i);
            }
        }

        // Creep 死亡特效：餅乾屑 + 糖果碎粒爆散（貼合糖果餅乾主題與餅乾碎裂音效）。
        // 一圈碎屑沿放射方向向外飛散、隨時間 ease-out 減速停住並淡出；每隻用 seed 擾動角度/速度/長度，
        // 讓每次爆散略有不同。同樣用 SceneDrawingContext 批次線段（single draw call），x 取負與爆炸圈一致。
        {
            use fyrox::scene::debug::Line;
            let dt_f = context.dt;
            const CRUMBS: usize = 12;
            let mut finished_idx: Vec<usize> = Vec::new();
            for (i, fx) in self.active_death_fx.iter_mut().enumerate() {
                fx.elapsed += dt_f;
                if fx.elapsed >= fx.duration {
                    finished_idx.push(i);
                    continue;
                }
                let t = (fx.elapsed / fx.duration).clamp(0.0, 1.0);
                // ease-out：碎屑一開始飛快、之後減速停住
                let ease = 1.0 - (1.0 - t) * (1.0 - t);
                let fade = ((230.0 * (1.0 - t)) as u8).max(35);
                let z = Z_REGION - 0.0004;
                let seed = fx.seed;
                // 由 seed 產生穩定的 per-crumb 偽隨機 [0,1)（同一隻怪每幀一致，不會抖動）
                let rnd = |n: u32| -> f32 {
                    let mut h = seed
                        .wrapping_mul(0x9E3779B9)
                        .wrapping_add(n.wrapping_mul(0x85EBCA6B));
                    h ^= h >> 16;
                    h = h.wrapping_mul(0x7FEB352D);
                    h ^= h >> 15;
                    (h & 0xFFFF) as f32 / 65535.0
                };
                for k in 0..CRUMBS {
                    let kk = k as u32;
                    let base = (k as f32) * std::f32::consts::TAU / (CRUMBS as f32);
                    let ang = base + (rnd(kk * 3) - 0.5) * 0.6;
                    let speed = 0.55 + rnd(kk * 3 + 1) * 0.75;
                    let dist = fx.max_radius * ease * speed;
                    let (s, c) = ang.sin_cos();
                    let cx = fx.pos.x + dist * c;
                    let cy = fx.pos.y + dist * s;
                    // 碎屑是一小段沿飛行方向的短棒，長度也隨 seed 變化
                    let seg = 0.045 + rnd(kk * 3 + 2) * 0.07;
                    let bx = cx - seg * c;
                    let by = cy - seg * s;
                    let ex_ = cx + seg * c;
                    let ey = cy + seg * s;
                    // 多數餅乾棕 / 焦糖屑，少數糖果亮色碎粒點綴
                    let color = match k % 5 {
                        0 => Color::from_rgba(255, 120, 185, fade), // 糖果粉
                        1 => Color::from_rgba(110, 220, 190, fade), // 薄荷綠
                        2 => Color::from_rgba(205, 150, 95, fade),  // 焦糖屑
                        3 => Color::from_rgba(255, 210, 80, fade),  // 檸檬黃糖粒
                        _ => Color::from_rgba(150, 95, 55, fade),   // 餅乾棕
                    };
                    scene.drawing_context.add_line(Line {
                        begin: Vector3::new(-bx, by, z),
                        end: Vector3::new(-ex_, ey, z),
                        color,
                    });
                }
            }
            // 反向刪除以保持 index 有效
            for i in finished_idx.into_iter().rev() {
                self.active_death_fx.remove(i);
            }
        }

        // 子彈命中濺射：命中點炸開該塔主題色的擴張環 + 放射線，~0.25s ease-out 淡出。
        // 快速子彈本身看不清，這個停在命中點（怪身上）看得清楚、有打擊感。x 取負同爆炸圈慣例。
        {
            use fyrox::scene::debug::Line;
            let dt_f = context.dt;
            const RING: usize = 16;
            let mut finished_idx: Vec<usize> = Vec::new();
            for (i, fx) in self.active_hit_fx.iter_mut().enumerate() {
                fx.elapsed += dt_f;
                if fx.elapsed >= fx.duration {
                    finished_idx.push(i);
                    continue;
                }
                let t = (fx.elapsed / fx.duration).clamp(0.0, 1.0);
                let ease = 1.0 - (1.0 - t) * (1.0 - t);
                let cur_r = fx.max_radius * ease;
                let a = (((fx.color[3] as f32) * (1.0 - t)) as u8).max(30);
                let color = Color::from_rgba(fx.color[0], fx.color[1], fx.color[2], a);
                let z = Z_REGION - 0.0005;
                if cur_r > 0.02 {
                    let px = fx.pos.x;
                    let py = fx.pos.y;
                    let off = (fx.seed as f32) * 0.3;
                    let seed = fx.seed;
                    // seed 偽隨機（crumbs 抖動用）
                    let rnd = |n: u32| -> f32 {
                        let mut h = seed
                            .wrapping_mul(0x9E3779B9)
                            .wrapping_add(n.wrapping_mul(0x85EBCA6B));
                        h ^= h >> 16;
                        h = h.wrapping_mul(0x7FEB352D);
                        h ^= h >> 15;
                        (h & 0xFFFF) as f32 / 65535.0
                    };
                    let tau = std::f32::consts::TAU;
                    match fx.style {
                        HitStyle::Frost => {
                            // 尖銳結晶爆刺：12 條細長冰刺（無環），長短交錯像雪花碎裂。
                            for k in 0..12u32 {
                                let ang = off + (k as f32) * tau / 12.0;
                                let (s, c) = ang.sin_cos();
                                let outer = cur_r * if k % 2 == 0 { 1.35 } else { 0.75 };
                                let inner = cur_r * 0.08;
                                scene.drawing_context.add_line(Line {
                                    begin: Vector3::new(-(px + inner * c), py + inner * s, z),
                                    end: Vector3::new(-(px + outer * c), py + outer * s, z),
                                    color,
                                });
                            }
                        }
                        HitStyle::CreamSplat => {
                            // 柔軟雙層奶油圈：兩圈（cur_r、0.62cur_r）+ 6 條短鈍刺，圓潤。
                            for rr in [cur_r, cur_r * 0.62] {
                                let mut prev = Vector3::new(-(px + rr), py, z);
                                for k in 1..=18u32 {
                                    let th = (k as f32) * tau / 18.0;
                                    let (s, c) = th.sin_cos();
                                    let next = Vector3::new(-(px + rr * c), py + rr * s, z);
                                    scene.drawing_context
                                        .add_line(Line { begin: prev, end: next, color });
                                    prev = next;
                                }
                            }
                            for k in 0..6u32 {
                                let ang = off + (k as f32) * tau / 6.0;
                                let (s, c) = ang.sin_cos();
                                let (i_r, o_r) = (cur_r * 0.9, cur_r * 1.12);
                                scene.drawing_context.add_line(Line {
                                    begin: Vector3::new(-(px + i_r * c), py + i_r * s, z),
                                    end: Vector3::new(-(px + o_r * c), py + o_r * s, z),
                                    color,
                                });
                            }
                        }
                        HitStyle::Crumbs => {
                            // 四散不規則碎屑：10 塊短棒，角度＋距離＋長度都用 seed 抖動，無環。
                            for k in 0..10u32 {
                                let ang = off + (k as f32) * tau / 10.0 + (rnd(k * 2) - 0.5) * 0.7;
                                let (s, c) = ang.sin_cos();
                                let d = cur_r * (0.5 + rnd(k * 2 + 1) * 0.55);
                                let seg = 0.05 + rnd(k * 2 + 7) * 0.06;
                                let cx = px + d * c;
                                let cy = py + d * s;
                                scene.drawing_context.add_line(Line {
                                    begin: Vector3::new(-(cx - seg * c), cy - seg * s, z),
                                    end: Vector3::new(-(cx + seg * c), cy + seg * s, z),
                                    color,
                                });
                            }
                        }
                        HitStyle::Sparkle => {
                            // 閃亮十字星芒：4 長芒（十字）+ 4 短芒（斜），小巧明亮、由中心放射。
                            for k in 0..8u32 {
                                let ang = off + (k as f32) * std::f32::consts::FRAC_PI_4;
                                let (s, c) = ang.sin_cos();
                                let o_r = if k % 2 == 0 { cur_r * 1.3 } else { cur_r * 0.6 };
                                scene.drawing_context.add_line(Line {
                                    begin: Vector3::new(-px, py, z),
                                    end: Vector3::new(-(px + o_r * c), py + o_r * s, z),
                                    color,
                                });
                            }
                        }
                        HitStyle::Slash => {
                            // 鋒利斬擊：3 條穿過中心的長線（星芒狀 X），銳利。
                            let r = cur_r * 1.3;
                            for k in 0..3u32 {
                                let ang = off + (k as f32) * (std::f32::consts::PI / 3.0);
                                let (s, c) = ang.sin_cos();
                                scene.drawing_context.add_line(Line {
                                    begin: Vector3::new(-(px - r * c), py - r * s, z),
                                    end: Vector3::new(-(px + r * c), py + r * s, z),
                                    color,
                                });
                            }
                        }
                        HitStyle::Burst => {
                            // fallback：環 + 8 均勻放射線。
                            let inner = cur_r * 0.25;
                            for k in 0..8u32 {
                                let ang = off + (k as f32) * tau / 8.0;
                                let (s, c) = ang.sin_cos();
                                scene.drawing_context.add_line(Line {
                                    begin: Vector3::new(-(px + inner * c), py + inner * s, z),
                                    end: Vector3::new(-(px + cur_r * c), py + cur_r * s, z),
                                    color,
                                });
                            }
                            let mut prev = Vector3::new(-(px + cur_r), py, z);
                            for k in 1..=RING {
                                let th = (k as f32) * tau / (RING as f32);
                                let (s, c) = th.sin_cos();
                                let next = Vector3::new(-(px + cur_r * c), py + cur_r * s, z);
                                scene.drawing_context
                                    .add_line(Line { begin: prev, end: next, color });
                                prev = next;
                            }
                        }
                    }
                }
            }
            for i in finished_idx.into_iter().rev() {
                self.active_hit_fx.remove(i);
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
        drop(visual_span);

        // 4b.推進客戶模擬的彈體（追擊目標
        // 目前插值位置； t 在 Flight_time 處強制為 1）。
        //     後端改為 100ms batch 發送，client flight_time 與 backend projectile time 已對齊
        //     (game_processor.rs 裡用 initial_dist / bullet_speed 設 safety_time_left 的 1/3)，
        //     所以彈落時 optimistic 扣血與 100ms 內到達的 backend "H" 事件幾乎 sync，不會 bouncing。
        let t_proj = std::time::Instant::now();
        let mut finished: Vec<u32> = Vec::new();
        // P7 layered：t≥1.0 視覺命中時要 mark 對應 pending_pred_dmg 為 applied=true。
        // 收 id 後在 loop 結束後一起做（避免 self.client_projectiles 與 self.pending_pred_dmg
        // 同時 mut borrow 的 split-borrow 麻煩）。
        let mut predicted_apply_ids: Vec<u32> = Vec::new();
        let projectile_span = tracing::trace_span!(
            "omfx::frame::projectiles_and_vfx",
            perfetto = true,
            projectiles = self.client_projectiles.len(),
            explosions = self.active_explosions.len(),
        )
        .entered();
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
                    .set_position(Vector3::new(
                        -(pos.x + offset.x),
                        pos.y + offset.y,
                        Z_BULLET + 0.0001,
                    ));
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
        drop(projectile_span);

        // 4c. Camera follow hero（MOBA 模式）或 固定俯視（TD 模式）
        //     TD 模式下：相機固定在地圖中心、拉遠到能看完整條路線。
        let t_cam = std::time::Instant::now();
        let camera_span = tracing::trace_span!("omfx::frame::camera", perfetto = true).entered();
        if self.is_td_mode {
            let td_camera = td_camera_view_config();
            // 固定 TD 視角每幀套用一次，讓 hot reload / 同一局調整相機時立即生效。
            if let Some(cam) = scene.graph[self.camera].cast_mut::<fyrox::scene::camera::Camera>() {
                cam.set_projection(Projection::Orthographic(OrthographicProjection {
                    z_near: 0.1,
                    z_far: 1000.0,
                    vertical_size: td_camera.vertical_size,
                }));
            }
            // 相機在 z=-100 處看著 +Z（預設）。scene X 往右移會讓畫面內容往左，
            // 替右側常駐商店留出空間；camera_world_pos 仍依投影慣例反推。
            scene.graph[self.camera]
                .local_transform_mut()
                .set_position(Vector3::new(
                    td_camera.scene_position.x,
                    td_camera.scene_position.y,
                    -100.0,
                ));
            self.camera_world_pos =
                Vector2::new(-td_camera.scene_position.x, td_camera.scene_position.y);
            if !self.td_camera_configured {
                self.td_camera_configured = true;
                log::info!(
                    "🎥 TD 相機已鎖定：scene_pos=({:.1},{:.1}), vertical_size={:.1}",
                    td_camera.scene_position.x,
                    td_camera.scene_position.y,
                    td_camera.vertical_size
                );

                // 階段 5.1：刪除了向舊版 NetworkBridge 的視窗推送。
                // 無論如何，鎖步狀態都會向所有客戶端完整廣播
                // 視口，因此不再需要此提示。
            }
        } else {
            // MOBA 模式：相機不再跟隨英雄移動。保留 camera 在 scene.rgs 載入時的初始位置，
            // camera_world_pos 從 camera 當前 transform 反推（X 渲染負號 → world.x = -cam.x），
            // 確保 name label 螢幕投影仍正確。
            let cam_pos = scene.graph[self.camera].local_transform().position();
            self.camera_world_pos = Vector2::new(-cam_pos.x, cam_pos.y);
            // 階段 5.1：刪除了與 NetworkBridge 的定期視窗同步。
        }

        let cam_ns = t_cam.elapsed().as_nanos();
        drop(camera_span);

        // 5.更新姓名標籤（UI層）
        let t_ui = std::time::Instant::now();
        let ui_span = tracing::trace_span!(
            "omfx::frame::ui",
            perfetto = true,
            labels = self.sim_entity_labels.len(),
            pending_labels = self.pending_label_deletions.len(),
        )
        .entered();
        let ui = context.user_interfaces.first_mut();
        let win = self.window_size;
        self.update_tower_ability_bar_ui(ui, now);

        // 刪除已刪除實體的標籤
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

        // 建立缺失標籤並更新位置
        for (&entity_id, entity) in self.network_entities.iter_mut() {
            if entity.health.is_none() {
                continue; // only show names for entities with HP bars
            }
            if labels_hidden {
                continue; // 太多 entity 且沒按 Alt，不渲染 name label，省 N 個 UI draw call
            }

            // 懶惰地創建標籤
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
                    win.x,
                    win.y,
                    world_height,
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
                        let pending_dmg = pending_dmg_by_target
                            .get(&entity_id)
                            .copied()
                            .unwrap_or(0.0);
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

        // sim_runner 支援的名稱標籤：階段 5.x 取代了舊版本
        // 上面的network_entities驅動的循環。讀取相同的快照
        // render_bridge 消耗；每個可見實體一個文字小工具，保留
        // 透過“sim_entity_labels”同步。
        if let Some(ref sim) = self.sim_runner_handle {
            if let Ok(snapshot) = sim.state.try_lock() {
                // 註：Creep 死亡特效的 spawn 已移到 `update_sim_batches`（音效同一個移除迴圈），
                // 因為這裡的 `try_lock()` 在鎖被 sim 執行緒佔用時會失敗、整段跳過，導致特效有時不觸發。
                const SIM_NAME_LABEL_HIDE_THRESHOLD: usize = 200;
                let labels_hidden =
                    snapshot.entities.len() > SIM_NAME_LABEL_HIDE_THRESHOLD && !self.alt_held;
                if labels_hidden {
                    for (_, slot) in self.sim_entity_labels.drain() {
                        ui.send(slot.handle, WidgetMessage::Remove);
                    }
                } else {
                    let mut alive =
                        std::collections::HashSet::with_capacity(snapshot.entities.len());
                    let world_height = if self.is_td_mode { 28.0 } else { 20.0 };
                    for entity in &snapshot.entities {
                        // Skip Other (internal ECS rows) and Projectile (子彈不需要標名稱).
                        if matches!(
                            entity.kind,
                            sim_runner::EntityKind::Other | sim_runner::EntityKind::Projectile
                        ) {
                            continue;
                        }
                        alive.insert(entity.entity_id);

                        // 顯示名稱：更喜歡hero_name（英雄），否則unit_id sans
                        // 模板前綴，否則回退到“#<id>”。
                        let display_name = if !entity.hero_name.is_empty() {
                            entity.hero_name.clone()
                        } else if !entity.unit_id.is_empty() {
                            entity
                                .unit_id
                                .strip_prefix("creep_")
                                .or_else(|| entity.unit_id.strip_prefix("tower_"))
                                .or_else(|| entity.unit_id.strip_prefix("hero_"))
                                .or_else(|| entity.unit_id.strip_prefix("unit_"))
                                .unwrap_or(&entity.unit_id)
                                .to_string()
                        } else {
                            format!("#{}", entity.entity_id)
                        };
                        // Tower 標籤：只在有升級時顯示「L0/L1/L2」格式，無升級不顯示
                        // 任何文字（也不顯示 HP — 塔不需要 HP 資訊）。Hero/Creep 走
                        // 既有 "name HP/MaxHP" 格式。
                        let is_tower = matches!(entity.kind, sim_runner::EntityKind::Tower);
                        // 跳過為沒有升級的塔繪製標籤小工具。
                        // 標記為不活動，以便循環後保留步驟刪除任何
                        // 從前一幀中徘徊的陳舊小部件
                        // （例如，塔剛剛出售或從未升級）。
                        if is_tower
                            && entity
                                .upgrade_levels
                                .map_or(true, |lv| lv.iter().all(|&n| n == 0))
                        {
                            alive.remove(&entity.entity_id);
                            continue;
                        }
                        let text = if is_tower {
                            match entity.upgrade_levels {
                                Some(lv) if lv.iter().any(|&n| n > 0) => {
                                    format!("{}/{}/{}", lv[0], lv[1], lv[2])
                                }
                                _ => String::new(),
                            }
                        } else if entity.max_hp > 0 {
                            format!("{} {}/{}", display_name, entity.hp.max(0), entity.max_hp)
                        } else {
                            display_name
                        };

                        // BUG FIX (Phase 5.x): 之前用 backend coords 直接餵 world_to_screen_approx。
                        // 該函式註解寫 "camera 的 -1 X scale 已把原本的翻轉抵消"，意思是它
                        // 接受 backend X 直接乘 WORLD_SCALE（不需自己翻 -x），然後函式內 +X
                        // world → +X screen 對應；sprite render 走 fyrox scene graph 經過
                        // camera -1 X 翻轉後才到螢幕，正好抵消 render_bridge 的 -x flip。
                        // 先前我多翻一次 → 名字跟 sprite 左右相反。
                        const WORLD_SCALE: f32 = 0.01;
                        let render_x = entity.pos_x * WORLD_SCALE;
                        let render_y = entity.pos_y * WORLD_SCALE + 0.5;
                        let screen_pos = world_to_screen_approx(
                            render_x - self.camera_world_pos.x,
                            render_y - self.camera_world_pos.y,
                            win.x,
                            win.y,
                            world_height,
                        );
                        let pos = Vector2::new(screen_pos.x - 110.0, screen_pos.y - 24.0);

                        if let Some(slot) = self.sim_entity_labels.get_mut(&entity.entity_id) {
                            // 更新現有的 — gateway 以避免淹沒 UI 隊列。
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
                            // 該實體首次產生。
                            let handle = TextBuilder::new(
                                WidgetBuilder::new()
                                    .with_desired_position(pos)
                                    .with_width(220.0)
                                    .with_foreground(
                                        Brush::Solid(Color::from_rgba(0, 0, 0, 255)).into(),
                                    ),
                            )
                            .with_text(text.clone())
                            .with_font_size(20.0.into())
                            .with_horizontal_text_alignment(HorizontalAlignment::Center)
                            .build(&mut ui.build_ctx());
                            self.sim_entity_labels.insert(
                                entity.entity_id,
                                SimEntityLabel {
                                    handle,
                                    last_text: text,
                                    last_pos: pos,
                                },
                            );
                        }
                    }

                    // 不再出現在快照中的實體的消失標籤。
                    // 階段 1.6：快照現在有明確的 `removed_entity_ids`
                    // （在 sim_runner 中本地計算工人），替換
                    // 遺留 omb `entity.death` 遊戲事件。我們仍然保留著
                    // 「活著」-設置掃掠下方作為腰帶和吊帶防禦
                    // 針對其 eid 從未出現過的任何快取行
                    // `removed_entity_ids`（例如，在先前建立的標籤）
                    // 第一個 prev_alive 快照已填入）。
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
        }

        // 6.更新狀態文本
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
                let wire_str = fmt_bps(self.net_wire_bytes_last_sec); // 真實 UDP wire (壓縮後)
                let logical_str = fmt_bps(self.net_bytes_last_sec); // 解壓後 logical
                let ping_str = match self.latest_rtt_us {
                    Some(us) => format!("{:.1} ms", us as f64 / 1000.0),
                    None => "—".into(),
                };
                let lag_str = format_input_lag_status(
                    &self.input_latency_meter,
                    oldest_pending_input_age_ms(&self.pending_inputs, wall_clock_us()),
                );
                let sim_lag_ticks = self
                    .current_sim_tick
                    .saturating_sub(self.heartbeat.tick as u32);
                format!(
                    "Connected | Ping: {} | Lag: {} | Sim: {:.0}/{:.0}Hz lag={}t | Tick: {} | Time: {:.1} | Entities: {} | Heroes: {} | Creeps: {} | Net: {} wire / {} logical",
                    ping_str,
                    lag_str,
                    self.sim_speed_tps,
                    self.server_timing().step_fps() as f32,
                    sim_lag_ticks,
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
        let connection_part = if let Some(err) = &self.sim_dev_lua_reload_error {
            format!("{} | DEV Lua reload error: {}", connection_part, err)
        } else {
            connection_part
        };
        // 前一幀的渲染統計資訊（record_render_stats() 在更新結束時運行，
        // 所以這裡的值落後 1 幀 — 對於實時讀數來說很好）。
        let render_stats_part = format!(
            "draws: {} | tris: {}",
            self.frame_profile.last_draw_calls, self.frame_profile.last_triangles,
        );
        let mut status_str = if self.fps_display.is_empty() {
            format!("{} | {}", render_stats_part, connection_part)
        } else {
            format!(
                "{} | {} | {}",
                self.fps_display, render_stats_part, connection_part
            )
        };
        if let Some(reason) = self
            .selected_tower_kind
            .as_ref()
            .and_then(|kind| self.td_templates.get(kind))
            .and_then(|template| {
                self.tower_placement_block_reason(template, self.mouse_world_pos)
            })
        {
            status_str.push_str(" | 建塔：");
            status_str.push_str(&reason.localized_text());
        }
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
                    if self.ui_ability_upgrade_buttons[i] != Handle::<Text>::NONE {
                        ui.send(
                            self.ui_ability_upgrade_buttons[i],
                            WidgetMessage::DesiredPosition(Vector2::new(x, icon_y - 32.0)),
                        );
                    }
                }
            }

            // ===== BTD-style 右側 shop/control panel =====
            {
                let (sx, sy) = td_ui_ref_scale(self.window_size);
                let right_panel_x_ref = 1555.0f32;
                let right_panel_w_ref = 365.0f32;
                let panel = td_ui_ref_rect(
                    self.window_size,
                    right_panel_x_ref,
                    0.0,
                    right_panel_w_ref,
                    1080.0,
                );
                self.ui_td_right_panel.panel_rect = panel;
                ui.send(
                    self.ui_td_right_panel.bg,
                    WidgetMessage::DesiredPosition(panel.pos()),
                );
                ui.send(self.ui_td_right_panel.bg, WidgetMessage::Width(panel.w));
                ui.send(self.ui_td_right_panel.bg, WidgetMessage::Height(panel.h));

                // 英雄知識入口按鈕（右側面板頂部，y~68）
                {
                    let gk_r = td_ui_ref_rect(
                        self.window_size,
                        right_panel_x_ref + 25.0,
                        68.0,
                        right_panel_w_ref - 50.0,
                        48.0,
                    );
                    self.gk_panel_ui.entry_rect = gk_r;
                    ui.send(self.gk_panel_ui.entry_bg, WidgetMessage::DesiredPosition(gk_r.pos()));
                    ui.send(self.gk_panel_ui.entry_bg, WidgetMessage::Width(gk_r.w));
                    ui.send(self.gk_panel_ui.entry_bg, WidgetMessage::Height(gk_r.h));
                    ui.send(self.gk_panel_ui.entry_text_h, WidgetMessage::DesiredPosition(gk_r.pos()));
                    ui.send(self.gk_panel_ui.entry_text_h, WidgetMessage::Width(gk_r.w));
                    ui.send(self.gk_panel_ui.entry_text_h, WidgetMessage::Height(gk_r.h));
                }

                let title = td_ui_ref_rect(
                    self.window_size,
                    right_panel_x_ref + 24.0,
                    124.0,
                    right_panel_w_ref - 48.0,
                    40.0,
                );
                ui.send(
                    self.ui_td_right_panel.title_text,
                    WidgetMessage::DesiredPosition(title.pos()),
                );
                ui.send(
                    self.ui_td_right_panel.title_text,
                    WidgetMessage::Width(title.w),
                );
                let shop_title = self
                    .selected_tower_kind
                    .as_ref()
                    .and_then(|k| self.td_templates.get(k))
                    .map(|tpl| tpl.label.clone())
                    .unwrap_or_else(|| "塔商店".to_string());
                ui.send(
                    self.ui_td_right_panel.title_text,
                    TextMessage::Text(shop_title),
                );

                let shop_viewport_y_ref = 170.0f32;
                let shop_viewport_h_ref = 745.0f32;
                let viewport = td_ui_ref_rect(
                    self.window_size,
                    right_panel_x_ref + 17.0,
                    shop_viewport_y_ref,
                    314.0,
                    shop_viewport_h_ref,
                );
                self.ui_td_right_panel.viewport_rect = viewport;
                ui.send(
                    self.ui_td_right_panel.viewport_bg,
                    WidgetMessage::DesiredPosition(viewport.pos()),
                );
                ui.send(
                    self.ui_td_right_panel.viewport_bg,
                    WidgetMessage::Width(viewport.w),
                );
                ui.send(
                    self.ui_td_right_panel.viewport_bg,
                    WidgetMessage::Height(viewport.h),
                );

                let n = self.td_template_order.len();
                while self.ui_td_tower_cards.len() < n {
                    let mut bg_builder = ImageBuilder::new(
                        WidgetBuilder::new()
                            .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                            .with_width(128.0)
                            .with_height(138.0),
                    );
                    if let Some(tex) = load_td_ui_texture("shop_card.png") {
                        bg_builder = bg_builder.with_texture(tex);
                    }
                    let bg: Handle<UiNode> = bg_builder.build(&mut ui.build_ctx()).transmute();
                    let icon: Handle<UiNode> = ImageBuilder::new(
                        WidgetBuilder::new()
                            .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                            .with_width(64.0)
                            .with_height(64.0),
                    )
                    .build(&mut ui.build_ctx())
                    .transmute();
                    let key_text = TextBuilder::new(
                        WidgetBuilder::new()
                            .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                            .with_width(40.0)
                            .with_foreground(
                                Brush::Solid(Color::from_rgba(255, 255, 255, 255)).into(),
                            ),
                    )
                    .with_text(String::new())
                    .with_font_size(18.0.into())
                    .build(&mut ui.build_ctx());
                    let name_text = TextBuilder::new(
                        WidgetBuilder::new()
                            .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                            .with_width(116.0)
                            .with_foreground(Brush::Solid(Color::from_rgba(35, 18, 6, 255)).into()),
                    )
                    .with_text(String::new())
                    .with_font_size(16.0.into())
                    .with_horizontal_text_alignment(HorizontalAlignment::Center)
                    .build(&mut ui.build_ctx());
                    let price_text = TextBuilder::new(
                        WidgetBuilder::new()
                            .with_desired_position(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS))
                            .with_width(112.0)
                            .with_foreground(
                                Brush::Solid(Color::from_rgba(255, 255, 255, 255)).into(),
                            ),
                    )
                    .with_text(String::new())
                    .with_font_size(22.0.into())
                    .with_horizontal_text_alignment(HorizontalAlignment::Center)
                    .build(&mut ui.build_ctx());
                    for node in [bg, icon] {
                        ui.link_nodes(node, self.ui_td_right_panel.viewport_bg, false);
                    }
                    for text in [key_text, name_text, price_text] {
                        ui.link_nodes(text, self.ui_td_right_panel.viewport_bg, false);
                    }
                    self.ui_td_tower_cards.push(TdTowerShopCard {
                        bg,
                        icon,
                        key_text,
                        name_text,
                        price_text,
                    });
                    self.td_tower_button_rects
                        .push((UI_HIDDEN_POS, UI_HIDDEN_POS, 0.0, 0.0));
                }
                for i in n..self.ui_td_tower_cards.len() {
                    let card = &self.ui_td_tower_cards[i];
                    for node in [card.bg, card.icon] {
                        ui.send(
                            node,
                            WidgetMessage::DesiredPosition(Vector2::new(
                                UI_HIDDEN_POS,
                                UI_HIDDEN_POS,
                            )),
                        );
                    }
                    for text in [card.key_text, card.name_text, card.price_text] {
                        ui.send(
                            text,
                            WidgetMessage::DesiredPosition(Vector2::new(
                                UI_HIDDEN_POS,
                                UI_HIDDEN_POS,
                            )),
                        );
                    }
                    if i < self.td_tower_button_rects.len() {
                        self.td_tower_button_rects[i] = (UI_HIDDEN_POS, UI_HIDDEN_POS, 0.0, 0.0);
                    }
                }

                let columns = if self.window_size.x < 1280.0 || self.window_size.y < 720.0 {
                    1
                } else {
                    2
                };
                let ref_card_w = if columns == 2 { 142.0 } else { 286.0 };
                let ref_card_h = if columns == 2 { 118.0 } else { 104.0 };
                let ref_col_gap = if columns == 2 { 151.0 } else { 0.0 };
                let ref_row_gap = if columns == 2 { 124.0 } else { 108.0 };
                let ref_grid_x = if columns == 2 {
                    right_panel_x_ref + 23.0
                } else {
                    right_panel_x_ref + 30.0
                };
                let ref_grid_y = if columns == 2 { 176.0 } else { 172.0 };
                let rows = if n == 0 {
                    0
                } else {
                    (n + columns - 1) / columns
                };
                let content_h = if rows == 0 {
                    0.0
                } else {
                    (rows.saturating_sub(1) as f32) * ref_row_gap + ref_card_h
                };
                self.td_shop_max_scroll = (content_h - shop_viewport_h_ref).max(0.0);
                self.set_td_shop_scroll_offset(self.td_shop_scroll_offset);
                let selected_kind = self.selected_tower_kind.clone();
                for i in 0..n {
                    let uid = self.td_template_order[i].clone();
                    let (_label, cost) = self
                        .td_templates
                        .get(&uid)
                        .map(|tpl| (tpl.label.clone(), tpl.cost))
                        .unwrap_or_else(|| (uid.clone(), 0));
                    let affordable = cost <= 0 || self.hero_state.gold >= cost;
                    let is_selected = selected_kind.as_deref() == Some(uid.as_str());
                    let col = i % columns;
                    let row = i / columns;
                    let card_rect = td_ui_ref_rect(
                        self.window_size,
                        ref_grid_x + col as f32 * ref_col_gap,
                        ref_grid_y + row as f32 * ref_row_gap - self.td_shop_scroll_offset,
                        ref_card_w,
                        ref_card_h,
                    );
                    let visible_rect = card_rect.intersection(viewport);
                    self.td_tower_button_rects[i] = visible_rect.map(|r| r.tuple()).unwrap_or((
                        UI_HIDDEN_POS,
                        UI_HIDDEN_POS,
                        0.0,
                        0.0,
                    ));

                    let bg_asset = if is_selected {
                        "shop_card_selected.png"
                    } else if !affordable {
                        "shop_card_locked.png"
                    } else {
                        "shop_card.png"
                    };
                    let bg_tex = self.td_ui_texture(bg_asset);
                    let icon_tex = self
                        .td_ui_texture(&format!("{}.png", uid))
                        .or_else(|| self.td_ui_texture("tower_fallback.png"));
                    let card = &self.ui_td_tower_cards[i];
                    if visible_rect.is_none() {
                        for node in [card.bg, card.icon] {
                            ui.send(
                                node,
                                WidgetMessage::DesiredPosition(Vector2::new(
                                    UI_HIDDEN_POS,
                                    UI_HIDDEN_POS,
                                )),
                            );
                        }
                        for text in [card.key_text, card.name_text, card.price_text] {
                            ui.send(
                                text,
                                WidgetMessage::DesiredPosition(Vector2::new(
                                    UI_HIDDEN_POS,
                                    UI_HIDDEN_POS,
                                )),
                            );
                        }
                        continue;
                    }
                    let local_card_pos =
                        Vector2::new(card_rect.x - viewport.x, card_rect.y - viewport.y);
                    ui.send(card.bg, WidgetMessage::DesiredPosition(local_card_pos));
                    ui.send(card.bg, WidgetMessage::Width(card_rect.w));
                    ui.send(card.bg, WidgetMessage::Height(card_rect.h));
                    ui.send(card.bg, ImageMessage::Texture(bg_tex));
                    let icon_size = if columns == 2 { 64.0 } else { 58.0 };
                    ui.send(
                        card.icon,
                        WidgetMessage::DesiredPosition(Vector2::new(
                            local_card_pos.x + (card_rect.w - icon_size * sx) * 0.5,
                            local_card_pos.y + 7.0 * sy,
                        )),
                    );
                    ui.send(card.icon, WidgetMessage::Width(icon_size * sx));
                    ui.send(card.icon, WidgetMessage::Height(icon_size * sy));
                    ui.send(card.icon, ImageMessage::Texture(icon_tex));
                    ui.send(
                        card.key_text,
                        WidgetMessage::DesiredPosition(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS)),
                    );
                    ui.send(card.key_text, TextMessage::Text(String::new()));
                    ui.send(
                        card.name_text,
                        WidgetMessage::DesiredPosition(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS)),
                    );
                    ui.send(card.name_text, TextMessage::Text(String::new()));
                    ui.send(
                        card.price_text,
                        WidgetMessage::DesiredPosition(Vector2::new(
                            local_card_pos.x + 8.0 * sx,
                            local_card_pos.y + card_rect.h - 37.0 * sy,
                        )),
                    );
                    ui.send(
                        card.price_text,
                        WidgetMessage::Width(card_rect.w - 16.0 * sx),
                    );
                    ui.send(
                        card.price_text,
                        TextMessage::Text(if affordable {
                            format!("${}", cost)
                        } else {
                            format!("缺 ${}", cost)
                        }),
                    );
                }

                let track = td_ui_ref_rect(
                    self.window_size,
                    right_panel_x_ref + right_panel_w_ref - 28.0,
                    shop_viewport_y_ref,
                    16.0,
                    shop_viewport_h_ref,
                );
                self.ui_td_right_panel.scroll_track_rect = track;
                ui.send(
                    self.ui_td_right_panel.scroll_track,
                    WidgetMessage::DesiredPosition(track.pos()),
                );
                ui.send(
                    self.ui_td_right_panel.scroll_track,
                    WidgetMessage::Width(track.w),
                );
                ui.send(
                    self.ui_td_right_panel.scroll_track,
                    WidgetMessage::Height(track.h),
                );

                let thumb_h_ref = if self.td_shop_max_scroll <= 0.0 || content_h <= 0.0 {
                    shop_viewport_h_ref
                } else {
                    (shop_viewport_h_ref / content_h * shop_viewport_h_ref)
                        .clamp(72.0, shop_viewport_h_ref)
                };
                let thumb_travel_ref = (shop_viewport_h_ref - thumb_h_ref).max(0.0);
                let thumb_y_ref = shop_viewport_y_ref
                    + if self.td_shop_max_scroll > 0.0 {
                        (self.td_shop_scroll_offset / self.td_shop_max_scroll) * thumb_travel_ref
                    } else {
                        0.0
                    };
                let thumb = td_ui_ref_rect(
                    self.window_size,
                    right_panel_x_ref + right_panel_w_ref - 26.0,
                    thumb_y_ref,
                    12.0,
                    thumb_h_ref,
                );
                self.ui_td_right_panel.scroll_thumb_rect = thumb;
                ui.send(
                    self.ui_td_right_panel.scroll_thumb,
                    WidgetMessage::DesiredPosition(thumb.pos()),
                );
                ui.send(
                    self.ui_td_right_panel.scroll_thumb,
                    WidgetMessage::Width(thumb.w),
                );
                ui.send(
                    self.ui_td_right_panel.scroll_thumb,
                    WidgetMessage::Height(thumb.h),
                );

                let pause_rect = td_ui_ref_rect(
                    self.window_size,
                    right_panel_x_ref + 25.0,
                    938.0,
                    142.0,
                    111.0,
                );
                self.ui_td_right_panel.pause_rect = pause_rect;
                self.pause_button_rect = pause_rect.tuple();
                ui.send(
                    self.ui_td_right_panel.pause_icon,
                    WidgetMessage::DesiredPosition(pause_rect.pos()),
                );
                ui.send(
                    self.ui_td_right_panel.pause_icon,
                    WidgetMessage::Width(pause_rect.w),
                );
                ui.send(
                    self.ui_td_right_panel.pause_icon,
                    WidgetMessage::Height(pause_rect.h),
                );
                ui.send(
                    self.ui_td_right_panel.pause_icon,
                    WidgetMessage::Opacity(td_pause_control_opacity(self.is_game_paused)),
                );
                ui.send(
                    self.ui_td_right_panel.pause_text,
                    WidgetMessage::DesiredPosition(Vector2::new(
                        pause_rect.x,
                        pause_rect.y + pause_rect.h + 2.0 * sy,
                    )),
                );
                ui.send(
                    self.ui_td_right_panel.pause_text,
                    WidgetMessage::Width(pause_rect.w),
                );
                ui.send(
                    self.ui_td_right_panel.pause_text,
                    WidgetMessage::Opacity(td_pause_control_opacity(self.is_game_paused)),
                );
                ui.send(
                    self.ui_td_right_panel.pause_text,
                    TextMessage::Text(td_pause_control_label(self.is_game_paused).to_string()),
                );
                let start_rect = td_ui_ref_rect(
                    self.window_size,
                    right_panel_x_ref + 190.0,
                    938.0,
                    142.0,
                    111.0,
                );
                self.ui_td_right_panel.start_rect = start_rect;
                self.start_round_button_rect = start_rect.tuple();
                let auto_rect = td_ui_ref_rect(
                    self.window_size,
                    right_panel_x_ref + 190.0,
                    900.0,
                    142.0,
                    32.0,
                );
                self.auto_start_checkbox_rect = auto_rect.tuple();
                ui.send(
                    self.ui_td_auto_start_checkbox_text,
                    WidgetMessage::DesiredPosition(auto_rect.pos()),
                );
                ui.send(
                    self.ui_td_auto_start_checkbox_text,
                    WidgetMessage::Width(auto_rect.w),
                );
                ui.send(
                    self.ui_td_auto_start_checkbox_text,
                    TextMessage::Text(td_auto_start_checkbox_label(self.td_auto_start_enabled)),
                );
                ui.send(
                    self.ui_td_right_panel.start_icon,
                    WidgetMessage::DesiredPosition(start_rect.pos()),
                );
                ui.send(
                    self.ui_td_right_panel.start_icon,
                    WidgetMessage::Width(start_rect.w),
                );
                ui.send(
                    self.ui_td_right_panel.start_icon,
                    WidgetMessage::Height(start_rect.h),
                );
                if self.ui_start_round_button != Handle::<Text>::NONE {
                    let start_label = td_start_control_label(
                        self.is_game_paused,
                        self.round_is_running,
                        self.current_round,
                        self.total_rounds,
                        self.game_speed_multiplier,
                    );
                    ui.send(
                        self.ui_start_round_button,
                        WidgetMessage::DesiredPosition(Vector2::new(
                            start_rect.x,
                            start_rect.y + start_rect.h + 2.0 * sy,
                        )),
                    );
                    ui.send(
                        self.ui_start_round_button,
                        WidgetMessage::Width(start_rect.w),
                    );
                    ui.send(
                        self.ui_start_round_button,
                        WidgetMessage::Foreground(
                            Brush::Solid(td_start_control_color(
                                self.is_game_paused,
                                self.round_is_running,
                                self.game_speed_multiplier,
                            ))
                            .into(),
                        ),
                    );
                    ui.send(
                        self.ui_start_round_button,
                        TextMessage::Text(start_label.to_string()),
                    );
                }
            }

            // ===== BTD-style selected tower context panel（依塔所在半邊自動換邊） =====
            {
                let (sx, sy) = td_ui_ref_scale(self.window_size);
                self.ui_td_selected_panel.left_anchor_rect =
                    td_ui_ref_rect(self.window_size, 24.0, 45.0, 426.0, 990.0);
                self.ui_td_selected_panel.right_anchor_rect =
                    td_ui_ref_rect(self.window_size, 1053.0, 45.0, 426.0, 990.0);

                let info: Option<(String, i32, [u8; 3], String, f32, String, u32, f32, f32)> =
                    self.selected_tower_entity.and_then(|tid| {
                        let ent = self.network_entities.get(&tid)?;
                        if !entity_owned_by_local(ent, self.local_player_id) {
                            return None;
                        }
                        let kind_key = ent.tower_kind.as_deref()?.to_string();
                        let tpl = self.td_templates.get(&kind_key)?;
                        let mut refund = (tpl.cost as f32 * 0.85) as i32;
                        for path in 0..3u8 {
                            for level in 1..=ent.upgrade_levels[path as usize] {
                                if let Some((_, _, cost)) =
                                    self.td_upgrade_defs.get(&(kind_key.clone(), path, level))
                                {
                                    refund += (*cost as f32 * 0.75) as i32;
                                }
                            }
                        }
                        Some((
                            tpl.label.clone(),
                            refund,
                            ent.upgrade_levels,
                            kind_key,
                            ent.attack_range_backend,
                            ent.tower_target_priority.clone(),
                            ent.tower_pops,
                            ent.tower_atk,
                            ent.tower_asd,
                        ))
                    });

                if let Some((
                    label,
                    refund,
                    levels,
                    kind_key,
                    range,
                    target_priority,
                    pops,
                    tower_atk,
                    tower_asd,
                )) = info
                {
                    let selected_x = self
                        .selected_tower_screen_x()
                        .unwrap_or(self.window_size.x * 0.5);
                    let ws = self.window_size;
                    let anchor_x_ref: f32 = if selected_x < ws.x * 0.5 {
                        1170.0
                    } else {
                        24.0
                    };
                    let rr = |dx: f32, dy: f32, w: f32, h: f32| -> UiRect {
                        td_ui_ref_rect(ws, anchor_x_ref + dx, 45.0 + dy, w, h)
                    };
                    let panel_rect = rr(0.0, 0.0, 380.0, 977.0);
                    self.ui_td_selected_panel.panel_rect = panel_rect;
                    ui.send(
                        self.ui_td_selected_panel.bg,
                        WidgetMessage::DesiredPosition(panel_rect.pos()),
                    );
                    ui.send(
                        self.ui_td_selected_panel.bg,
                        WidgetMessage::Width(panel_rect.w),
                    );
                    ui.send(
                        self.ui_td_selected_panel.bg,
                        WidgetMessage::Height(panel_rect.h),
                    );
                    // Unified body background: full panel, brown background behind everything
                    let body_bg_rect = rr(0.0, 0.0, 380.0, 977.0);
                    ui.send(
                        self.ui_td_selected_panel.body_bg,
                        WidgetMessage::DesiredPosition(body_bg_rect.pos()),
                    );
                    ui.send(
                        self.ui_td_selected_panel.body_bg,
                        WidgetMessage::Width(body_bg_rect.w),
                    );
                    ui.send(
                        self.ui_td_selected_panel.body_bg,
                        WidgetMessage::Height(body_bg_rect.h),
                    );
                    // 深色 header strip 不用了，隱藏
                    ui.send(
                        self.ui_td_selected_panel.header_strip_bg,
                        WidgetMessage::DesiredPosition(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS)),
                    );
                    ui.send(
                        self.ui_td_selected_panel.header_strip_bottom_mask,
                        WidgetMessage::DesiredPosition(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS)),
                    );
                    ui.send(
                        self.ui_td_selected_panel.header_bg,
                        WidgetMessage::DesiredPosition(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS)),
                    );
                    // 塔名置中（棕色 body 延伸上去當背景）
                    let title_rect = rr(0.0, 8.0, 330.0, 46.0);
                    ui.send(
                        self.ui_td_selected_panel.header_title,
                        WidgetMessage::DesiredPosition(title_rect.pos()),
                    );
                    ui.send(
                        self.ui_td_selected_panel.header_title,
                        WidgetMessage::Width(title_rect.w),
                    );
                    ui.send(
                        self.ui_td_selected_panel.header_title,
                        WidgetMessage::Height(title_rect.h),
                    );
                    ui.send(
                        self.ui_td_selected_panel.header_title,
                        TextMessage::Text(label.clone()),
                    );
                    // pops 在塔名左下角（跟著置中名字移動）
                    let pops_rect = rr(110.0, 50.0, 150.0, 22.0);
                    ui.send(
                        self.ui_td_selected_panel.pops_text,
                        WidgetMessage::DesiredPosition(pops_rect.pos()),
                    );
                    ui.send(
                        self.ui_td_selected_panel.pops_text,
                        WidgetMessage::Width(pops_rect.w),
                    );
                    ui.send(
                        self.ui_td_selected_panel.pops_text,
                        WidgetMessage::Height(pops_rect.h),
                    );
                    ui.send(
                        self.ui_td_selected_panel.pops_text,
                        TextMessage::Text(format!("✦ {}", pops)),
                    );
                    // i 按鈕（黃色卡片左上角）
                    let info_btn_rect = rr(22.0, 80.0, 36.0, 36.0);
                    self.ui_td_selected_panel.info_btn_rect = info_btn_rect;
                    ui.send(
                        self.ui_td_selected_panel.info_btn_bg,
                        WidgetMessage::DesiredPosition(info_btn_rect.pos()),
                    );
                    ui.send(
                        self.ui_td_selected_panel.info_btn_bg,
                        WidgetMessage::Width(info_btn_rect.w),
                    );
                    ui.send(
                        self.ui_td_selected_panel.info_btn_bg,
                        WidgetMessage::Height(info_btn_rect.h),
                    );
                    ui.send(
                        self.ui_td_selected_panel.info_btn_text,
                        WidgetMessage::DesiredPosition(info_btn_rect.pos()),
                    );
                    ui.send(
                        self.ui_td_selected_panel.info_btn_text,
                        WidgetMessage::Width(info_btn_rect.w),
                    );
                    ui.send(
                        self.ui_td_selected_panel.info_btn_text,
                        WidgetMessage::Height(info_btn_rect.h),
                    );
                    // info 側面板（面板右側或左側，依 panel_rect 動態計算）
                    let sx = self.window_size.x / TD_UI_REF_W;
                    let sy = self.window_size.y / TD_UI_REF_H;
                    let pr = self.ui_td_selected_panel.panel_rect;
                    let panel_right = if pr.x > self.window_size.x * 0.5 {
                        pr.x - 244.0 * sx - 8.0
                    } else {
                        pr.right() + 8.0
                    };
                    if self.ui_td_selected_panel.show_info {
                        // 對齊黃色卡片頂部 (ref y = 45+78 = 123)
                        let card_top = (45.0 + 78.0) * sy;
                        let box_w = 244.0 * sx;
                        let box_h = 230.0 * sy;
                        ui.send(
                            self.ui_td_selected_panel.info_overlay_bg,
                            WidgetMessage::DesiredPosition(Vector2::new(panel_right, card_top)),
                        );
                        ui.send(
                            self.ui_td_selected_panel.info_overlay_bg,
                            WidgetMessage::Width(box_w),
                        );
                        ui.send(
                            self.ui_td_selected_panel.info_overlay_bg,
                            WidgetMessage::Height(box_h),
                        );
                        let asd_display = if tower_asd > 0.0 {
                            format!("{:.2}s", tower_asd)
                        } else {
                            "-".to_string()
                        };
                        let atk_display = if tower_atk > 0.0 {
                            format!("{:.0}", tower_atk)
                        } else {
                            "-".to_string()
                        };
                        let range_display = format!("{:.0}", range);
                        let stat_lines = [
                            format!("傷害       {}", atk_display),
                            format!("攻速       每 {}", asd_display),
                            format!("射程       {}", range_display),
                            format!("擊破數   {}", pops),
                        ];
                        let pad = 16.0 * sx;
                        for i in 0..4usize {
                            let ty = card_top + (30.0 + i as f32 * 50.0) * sy;
                            ui.send(
                                self.ui_td_selected_panel.info_stat_texts[i],
                                WidgetMessage::DesiredPosition(Vector2::new(panel_right + pad, ty)),
                            );
                            ui.send(
                                self.ui_td_selected_panel.info_stat_texts[i],
                                WidgetMessage::Width(box_w - pad * 2.0),
                            );
                            ui.send(
                                self.ui_td_selected_panel.info_stat_texts[i],
                                WidgetMessage::Height(28.0 * sy),
                            );
                            ui.send(
                                self.ui_td_selected_panel.info_stat_texts[i],
                                TextMessage::Text(stat_lines[i].clone()),
                            );
                        }
                        // 三列升級 tooltip（同 hover tooltip 定位，非鎖定才顯示）
                        let paths_with_levels = levels.iter().filter(|&&l| l > 0).count();
                        for i in 0..3usize {
                            let lvl = levels[i];
                            let locked = paths_with_levels >= 2 && lvl == 0;
                            let hide_row = |s: &mut Self, ui: &mut UserInterface| {
                                ui.send(
                                    s.ui_td_selected_panel.info_row_bgs[i],
                                    WidgetMessage::DesiredPosition(Vector2::new(
                                        UI_HIDDEN_POS,
                                        UI_HIDDEN_POS,
                                    )),
                                );
                                ui.send(
                                    s.ui_td_selected_panel.info_row_titles[i],
                                    WidgetMessage::DesiredPosition(Vector2::new(
                                        UI_HIDDEN_POS,
                                        UI_HIDDEN_POS,
                                    )),
                                );
                                ui.send(
                                    s.ui_td_selected_panel.info_row_descs[i],
                                    WidgetMessage::DesiredPosition(Vector2::new(
                                        UI_HIDDEN_POS,
                                        UI_HIDDEN_POS,
                                    )),
                                );
                                ui.send(
                                    s.ui_td_selected_panel.info_row_descs2[i],
                                    WidgetMessage::DesiredPosition(Vector2::new(
                                        UI_HIDDEN_POS,
                                        UI_HIDDEN_POS,
                                    )),
                                );
                            };
                            if locked {
                                hide_row(self, ui);
                            } else {
                                let row_ref_y = 313.0 + i as f32 * 192.0;
                                let ty = (45.0 + row_ref_y + 15.0) * sy;
                                let box_w = 244.0 * sx;
                                let box_h = 160.0 * sy;
                                ui.send(
                                    self.ui_td_selected_panel.info_row_bgs[i],
                                    WidgetMessage::DesiredPosition(Vector2::new(panel_right, ty)),
                                );
                                ui.send(
                                    self.ui_td_selected_panel.info_row_bgs[i],
                                    WidgetMessage::Width(box_w),
                                );
                                ui.send(
                                    self.ui_td_selected_panel.info_row_bgs[i],
                                    WidgetMessage::Height(box_h),
                                );
                                let pad = 12.0 * sx;
                                let (title, desc) = if lvl >= TD_UI_MAX_UPGRADE_LEVEL {
                                    let last_name = self
                                        .td_upgrade_defs
                                        .get(&(kind_key.clone(), i as u8, TD_UI_MAX_UPGRADE_LEVEL))
                                        .map(|(n, _, _)| n.clone())
                                        .unwrap_or_else(|| "MAX".to_string());
                                    (last_name, "已升至最高級別".to_string())
                                } else {
                                    let next_lvl = lvl + 1;
                                    self.td_upgrade_defs
                                        .get(&(kind_key.clone(), i as u8, next_lvl))
                                        .map(|(n, d, _)| (n.clone(), td_upgrade_effect_text(d)))
                                        .unwrap_or_else(|| ("?".to_string(), "無說明".to_string()))
                                };
                                ui.send(
                                    self.ui_td_selected_panel.info_row_titles[i],
                                    WidgetMessage::DesiredPosition(Vector2::new(
                                        panel_right + pad,
                                        ty - 15.0 * sy,
                                    )),
                                );
                                ui.send(
                                    self.ui_td_selected_panel.info_row_titles[i],
                                    WidgetMessage::Width(box_w - pad * 2.0),
                                );
                                ui.send(
                                    self.ui_td_selected_panel.info_row_titles[i],
                                    TextMessage::Text(title),
                                );
                                let desc_y = ty + 48.0 * sy;
                                ui.send(
                                    self.ui_td_selected_panel.info_row_descs[i],
                                    WidgetMessage::DesiredPosition(Vector2::new(
                                        panel_right + pad,
                                        desc_y,
                                    )),
                                );
                                ui.send(
                                    self.ui_td_selected_panel.info_row_descs[i],
                                    WidgetMessage::Width(box_w - pad * 2.0),
                                );
                                ui.send(
                                    self.ui_td_selected_panel.info_row_descs[i],
                                    TextMessage::Text(desc),
                                );
                                ui.send(
                                    self.ui_td_selected_panel.info_row_descs2[i],
                                    WidgetMessage::DesiredPosition(Vector2::new(-9999.0, -9999.0)),
                                );
                                ui.send(
                                    self.ui_td_selected_panel.info_row_descs2[i],
                                    TextMessage::Text(String::new()),
                                );
                            }
                        }
                        // show_info 時壓制 hover tooltip，避免雙層
                        ui.send(
                            self.ui_upgrade_tooltip_bg,
                            WidgetMessage::DesiredPosition(Vector2::new(-9999.0, -9999.0)),
                        );
                        ui.send(
                            self.ui_upgrade_tooltip_title,
                            WidgetMessage::DesiredPosition(Vector2::new(-9999.0, -9999.0)),
                        );
                        ui.send(
                            self.ui_upgrade_tooltip_desc,
                            WidgetMessage::DesiredPosition(Vector2::new(-9999.0, -9999.0)),
                        );
                        ui.send(
                            self.ui_upgrade_tooltip_desc2,
                            WidgetMessage::DesiredPosition(Vector2::new(-9999.0, -9999.0)),
                        );
                    } else {
                        ui.send(
                            self.ui_td_selected_panel.info_overlay_bg,
                            WidgetMessage::DesiredPosition(Vector2::new(
                                UI_HIDDEN_POS,
                                UI_HIDDEN_POS,
                            )),
                        );
                        for i in 0..4usize {
                            ui.send(
                                self.ui_td_selected_panel.info_stat_texts[i],
                                WidgetMessage::DesiredPosition(Vector2::new(
                                    UI_HIDDEN_POS,
                                    UI_HIDDEN_POS,
                                )),
                            );
                        }
                        for i in 0..3usize {
                            ui.send(
                                self.ui_td_selected_panel.info_row_bgs[i],
                                WidgetMessage::DesiredPosition(Vector2::new(
                                    UI_HIDDEN_POS,
                                    UI_HIDDEN_POS,
                                )),
                            );
                            ui.send(
                                self.ui_td_selected_panel.info_row_titles[i],
                                WidgetMessage::DesiredPosition(Vector2::new(
                                    UI_HIDDEN_POS,
                                    UI_HIDDEN_POS,
                                )),
                            );
                            ui.send(
                                self.ui_td_selected_panel.info_row_descs[i],
                                WidgetMessage::DesiredPosition(Vector2::new(
                                    UI_HIDDEN_POS,
                                    UI_HIDDEN_POS,
                                )),
                            );
                            ui.send(
                                self.ui_td_selected_panel.info_row_descs2[i],
                                WidgetMessage::DesiredPosition(Vector2::new(
                                    UI_HIDDEN_POS,
                                    UI_HIDDEN_POS,
                                )),
                            );
                        }
                    }
                    let close_rect = rr(342.0, 11.0, 30.0, 30.0);
                    self.ui_td_selected_panel.close_btn_rect = close_rect;
                    ui.send(
                        self.ui_td_selected_panel.close_btn_bg,
                        WidgetMessage::DesiredPosition(close_rect.pos()),
                    );
                    ui.send(
                        self.ui_td_selected_panel.close_btn_bg,
                        WidgetMessage::Width(close_rect.w),
                    );
                    ui.send(
                        self.ui_td_selected_panel.close_btn_bg,
                        WidgetMessage::Height(close_rect.h),
                    );
                    ui.send(
                        self.ui_td_selected_panel.close_btn_text,
                        WidgetMessage::DesiredPosition(close_rect.pos()),
                    );
                    ui.send(
                        self.ui_td_selected_panel.close_btn_text,
                        WidgetMessage::Width(close_rect.w),
                    );
                    ui.send(
                        self.ui_td_selected_panel.close_btn_text,
                        WidgetMessage::Height(close_rect.h),
                    );
                    // ── Image area (yellow card, 20px margin all sides) ──
                    // card: x=20..360, y=70..300 (16px from header bottom at y=54, 20px margin)
                    let img_area_rect = rr(20.0, 78.0, 340.0, 230.0);
                    ui.send(
                        self.ui_td_selected_panel.image_area_bg,
                        WidgetMessage::DesiredPosition(img_area_rect.pos()),
                    );
                    ui.send(
                        self.ui_td_selected_panel.image_area_bg,
                        WidgetMessage::Width(img_area_rect.w),
                    );
                    ui.send(
                        self.ui_td_selected_panel.image_area_bg,
                        WidgetMessage::Height(img_area_rect.h),
                    );
                    let tower_tex = self
                        .td_ui_texture(&format!("{}.png", kind_key))
                        .or_else(|| self.td_ui_texture("tower_fallback.png"));
                    // btd6_tower_icon 建立晚於所有背景，z-order 高，不會被遮
                    ui.send(
                        self.ui_td_selected_panel.tower_icon,
                        WidgetMessage::DesiredPosition(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS)),
                    );
                    // tower icon: source is 2816×1536 (16:9), fill card width at correct ratio
                    // card w=340 → h = 340 × (1536/2816) = 185px
                    let icon_rect = rr(20.0, 98.0, 340.0, 185.0);
                    ui.send(
                        self.ui_td_selected_panel.btd6_tower_icon,
                        WidgetMessage::DesiredPosition(icon_rect.pos()),
                    );
                    ui.send(
                        self.ui_td_selected_panel.btd6_tower_icon,
                        WidgetMessage::Width(icon_rect.w),
                    );
                    ui.send(
                        self.ui_td_selected_panel.btd6_tower_icon,
                        WidgetMessage::Height(icon_rect.h),
                    );
                    ui.send(
                        self.ui_td_selected_panel.btd6_tower_icon,
                        ImageMessage::Texture(tower_tex),
                    );
                    // arrows at bottom of yellow card, same row as path name (card bottom y=300)
                    let left_arrow_rect = rr(24.0, 270.0, 36.0, 36.0);
                    self.ui_td_selected_panel.path_left_rect = left_arrow_rect;
                    ui.send(
                        self.ui_td_selected_panel.path_left_bg,
                        WidgetMessage::DesiredPosition(left_arrow_rect.pos()),
                    );
                    ui.send(
                        self.ui_td_selected_panel.path_left_bg,
                        WidgetMessage::Width(left_arrow_rect.w),
                    );
                    ui.send(
                        self.ui_td_selected_panel.path_left_bg,
                        WidgetMessage::Height(left_arrow_rect.h),
                    );
                    ui.send(
                        self.ui_td_selected_panel.path_left_text,
                        WidgetMessage::DesiredPosition(left_arrow_rect.pos()),
                    );
                    ui.send(
                        self.ui_td_selected_panel.path_left_text,
                        WidgetMessage::Width(left_arrow_rect.w),
                    );
                    ui.send(
                        self.ui_td_selected_panel.path_left_text,
                        WidgetMessage::Height(left_arrow_rect.h),
                    );
                    let right_arrow_rect = rr(320.0, 270.0, 36.0, 36.0);
                    self.ui_td_selected_panel.path_right_rect = right_arrow_rect;
                    ui.send(
                        self.ui_td_selected_panel.path_right_bg,
                        WidgetMessage::DesiredPosition(right_arrow_rect.pos()),
                    );
                    ui.send(
                        self.ui_td_selected_panel.path_right_bg,
                        WidgetMessage::Width(right_arrow_rect.w),
                    );
                    ui.send(
                        self.ui_td_selected_panel.path_right_bg,
                        WidgetMessage::Height(right_arrow_rect.h),
                    );
                    ui.send(
                        self.ui_td_selected_panel.path_right_text,
                        WidgetMessage::DesiredPosition(right_arrow_rect.pos()),
                    );
                    ui.send(
                        self.ui_td_selected_panel.path_right_text,
                        WidgetMessage::Width(right_arrow_rect.w),
                    );
                    ui.send(
                        self.ui_td_selected_panel.path_right_text,
                        WidgetMessage::Height(right_arrow_rect.h),
                    );
                    let path = self.ui_td_selected_panel.selected_path as usize;
                    let path_names = ["第一個", "第二個", "第三個"];
                    // path name near bottom of yellow card
                    let name_rect = rr(20.0, 264.0, 340.0, 34.0);
                    ui.send(
                        self.ui_td_selected_panel.path_name_label,
                        WidgetMessage::DesiredPosition(name_rect.pos()),
                    );
                    ui.send(
                        self.ui_td_selected_panel.path_name_label,
                        WidgetMessage::Width(name_rect.w),
                    );
                    ui.send(
                        self.ui_td_selected_panel.path_name_label,
                        WidgetMessage::Height(name_rect.h),
                    );
                    ui.send(
                        self.ui_td_selected_panel.path_name_label,
                        TextMessage::Text(path_names[path].to_string()),
                    );
                    // ── 三行升級路線（同時顯示所有路線）──
                    // BTD6 規則：兩條路徑已升級（level > 0）→ 第三條鎖住
                    let paths_with_levels = levels.iter().filter(|&&l| l > 0).count();
                    for i in 0..3usize {
                        let row_y = 313.0 + i as f32 * 192.0;
                        let lvl = levels[i];
                        let next_lvl = (lvl + 1).min(TD_UI_MAX_UPGRADE_LEVEL);
                        let path_maxed = lvl >= TD_UI_MAX_UPGRADE_LEVEL;
                        let path_locked = paths_with_levels >= 2 && lvl == 0;
                        // 棕色底色（鎖住時調深）
                        let row_rect = rr(8.0, row_y + 15.0, 244.0, 160.0);
                        ui.send(
                            self.ui_td_selected_panel.upgrade_row_bgs[i],
                            WidgetMessage::DesiredPosition(row_rect.pos()),
                        );
                        ui.send(
                            self.ui_td_selected_panel.upgrade_row_bgs[i],
                            WidgetMessage::Width(row_rect.w),
                        );
                        ui.send(
                            self.ui_td_selected_panel.upgrade_row_bgs[i],
                            WidgetMessage::Height(row_rect.h),
                        );
                        ui.send(
                            self.ui_td_selected_panel.upgrade_row_bgs[i],
                            WidgetMessage::Background(if path_locked {
                                Brush::Solid(Color::from_rgba(100, 70, 35, 255)).into()
                            } else {
                                Brush::Solid(Color::from_rgba(175, 125, 60, 255)).into()
                            }),
                        );
                        if path_locked {
                            // 鎖住：鋪滿整列（含按鈕區），置中顯示「路徑關閉」
                            // row_rect 已經是 244 寬，這裡補一個全寬覆蓋層
                            let full_row_rect = rr(8.0, row_y + 15.0, 366.0, 160.0);
                            ui.send(
                                self.ui_td_selected_panel.upgrade_row_bgs[i],
                                WidgetMessage::Width(full_row_rect.w),
                            );
                            ui.send(
                                self.ui_td_selected_panel.upgrade_pip_texts[i],
                                WidgetMessage::DesiredPosition(Vector2::new(
                                    UI_HIDDEN_POS,
                                    UI_HIDDEN_POS,
                                )),
                            );
                            let status_rect = full_row_rect;
                            ui.send(
                                self.ui_td_selected_panel.upgrade_status_texts[i],
                                WidgetMessage::DesiredPosition(status_rect.pos()),
                            );
                            ui.send(
                                self.ui_td_selected_panel.upgrade_status_texts[i],
                                WidgetMessage::Width(status_rect.w),
                            );
                            ui.send(
                                self.ui_td_selected_panel.upgrade_status_texts[i],
                                WidgetMessage::Height(status_rect.h),
                            );
                            ui.send(
                                self.ui_td_selected_panel.upgrade_status_texts[i],
                                TextMessage::Text("路徑關閉".to_string()),
                            );
                            ui.send(
                                self.ui_td_selected_panel.upgrade_status_texts[i],
                                WidgetMessage::Foreground(
                                    Brush::Solid(Color::from_rgba(140, 90, 40, 255)).into(),
                                ),
                            );
                            self.td_upgrade_button_rects[i] =
                                (UI_HIDDEN_POS, UI_HIDDEN_POS, 0.0, 0.0);
                            ui.send(
                                self.ui_td_selected_panel.upgrade_bgs[i],
                                WidgetMessage::DesiredPosition(Vector2::new(
                                    UI_HIDDEN_POS,
                                    UI_HIDDEN_POS,
                                )),
                            );
                            ui.send(
                                self.ui_td_selected_panel.upgrade_icons[i],
                                WidgetMessage::DesiredPosition(Vector2::new(
                                    UI_HIDDEN_POS,
                                    UI_HIDDEN_POS,
                                )),
                            );
                            ui.send(
                                self.ui_td_selected_panel.upgrade_name_texts[i],
                                WidgetMessage::DesiredPosition(Vector2::new(
                                    UI_HIDDEN_POS,
                                    UI_HIDDEN_POS,
                                )),
                            );
                            ui.send(
                                self.ui_td_selected_panel.upgrade_price_texts[i],
                                WidgetMessage::DesiredPosition(Vector2::new(
                                    UI_HIDDEN_POS,
                                    UI_HIDDEN_POS,
                                )),
                            );
                        } else {
                            // 一般顯示
                            ui.send(
                                self.ui_td_selected_panel.upgrade_status_texts[i],
                                WidgetMessage::Foreground(
                                    Brush::Solid(Color::from_rgba(60, 35, 10, 255)).into(),
                                ),
                            );
                            let pip_str: String = (0..TD_UI_MAX_UPGRADE_LEVEL as usize)
                                .map(|j| if (j as u8) < lvl { "■" } else { "□" })
                                .collect::<Vec<_>>()
                                .join("\n");
                            let pip_rect = rr(14.0, row_y + 15.0, 30.0, 145.0);
                            ui.send(
                                self.ui_td_selected_panel.upgrade_pip_texts[i],
                                WidgetMessage::DesiredPosition(pip_rect.pos()),
                            );
                            ui.send(
                                self.ui_td_selected_panel.upgrade_pip_texts[i],
                                WidgetMessage::Width(pip_rect.w),
                            );
                            ui.send(
                                self.ui_td_selected_panel.upgrade_pip_texts[i],
                                WidgetMessage::Height(pip_rect.h),
                            );
                            ui.send(
                                self.ui_td_selected_panel.upgrade_pip_texts[i],
                                TextMessage::Text(pip_str),
                            );
                            let status_str = if lvl == 0 {
                                "未升級".to_string()
                            } else {
                                format!("級別 {}", lvl)
                            };
                            let status_rect = rr(48.0, row_y + 15.0, 130.0, 160.0);
                            ui.send(
                                self.ui_td_selected_panel.upgrade_status_texts[i],
                                WidgetMessage::DesiredPosition(status_rect.pos()),
                            );
                            ui.send(
                                self.ui_td_selected_panel.upgrade_status_texts[i],
                                WidgetMessage::Width(status_rect.w),
                            );
                            ui.send(
                                self.ui_td_selected_panel.upgrade_status_texts[i],
                                WidgetMessage::Height(status_rect.h),
                            );
                            ui.send(
                                self.ui_td_selected_panel.upgrade_status_texts[i],
                                TextMessage::Text(status_str),
                            );
                            let btn_rect = rr(200.0, row_y + 15.0, 175.0, 160.0);
                            // MAX 時也保留 rect，讓 hover tooltip 能偵測到
                            self.td_upgrade_button_rects[i] = btn_rect.tuple();
                            ui.send(
                                self.ui_td_selected_panel.upgrade_bgs[i],
                                WidgetMessage::DesiredPosition(btn_rect.pos()),
                            );
                            ui.send(
                                self.ui_td_selected_panel.upgrade_bgs[i],
                                WidgetMessage::Width(btn_rect.w),
                            );
                            ui.send(
                                self.ui_td_selected_panel.upgrade_bgs[i],
                                WidgetMessage::Height(btn_rect.h),
                            );
                            let icon_tex = self
                                .td_ui_texture(&format!("{}_p{}.png", kind_key, i + 1))
                                .or_else(|| self.td_ui_texture(&format!("upgrade_p{}.png", i + 1)));
                            let icon_rect = rr(225.0, row_y + 43.0, 150.0, 82.0);
                            ui.send(
                                self.ui_td_selected_panel.upgrade_icons[i],
                                WidgetMessage::DesiredPosition(icon_rect.pos()),
                            );
                            ui.send(
                                self.ui_td_selected_panel.upgrade_icons[i],
                                WidgetMessage::Width(icon_rect.w),
                            );
                            ui.send(
                                self.ui_td_selected_panel.upgrade_icons[i],
                                WidgetMessage::Height(icon_rect.h),
                            );
                            ui.send(
                                self.ui_td_selected_panel.upgrade_icons[i],
                                ImageMessage::Texture(icon_tex),
                            );
                            let (upgrade_name, price_str) = if path_maxed {
                                ("MAX".to_string(), "".to_string())
                            } else {
                                let (next_name, next_cost) = self
                                    .td_upgrade_defs
                                    .get(&(kind_key.clone(), i as u8, next_lvl))
                                    .map(|(n, _, c)| (n.as_str(), *c))
                                    .unwrap_or(("?", 0));
                                (td_upgrade_title_text(next_name), format!("${}", next_cost))
                            };
                            let name_rect = rr(253.0, row_y + 0.0, 118.0, 42.0);
                            ui.send(
                                self.ui_td_selected_panel.upgrade_name_texts[i],
                                WidgetMessage::DesiredPosition(name_rect.pos()),
                            );
                            ui.send(
                                self.ui_td_selected_panel.upgrade_name_texts[i],
                                WidgetMessage::Width(name_rect.w),
                            );
                            ui.send(
                                self.ui_td_selected_panel.upgrade_name_texts[i],
                                WidgetMessage::Height(name_rect.h),
                            );
                            ui.send(
                                self.ui_td_selected_panel.upgrade_name_texts[i],
                                TextMessage::Text(upgrade_name),
                            );
                            let price_rect = rr(253.0, row_y + 128.0, 118.0, 42.0);
                            ui.send(
                                self.ui_td_selected_panel.upgrade_price_texts[i],
                                WidgetMessage::DesiredPosition(price_rect.pos()),
                            );
                            ui.send(
                                self.ui_td_selected_panel.upgrade_price_texts[i],
                                WidgetMessage::Width(price_rect.w),
                            );
                            ui.send(
                                self.ui_td_selected_panel.upgrade_price_texts[i],
                                WidgetMessage::Height(price_rect.h),
                            );
                            ui.send(
                                self.ui_td_selected_panel.upgrade_price_texts[i],
                                TextMessage::Text(price_str),
                            );
                        }
                    }
                    // ── 升級按鈕 hover tooltip（BTD6 風格）──
                    {
                        let mouse = self.mouse_screen_pos;
                        let mut new_upgrade_hover: Option<usize> = None;
                        for i in 0..3usize {
                            let (rx, ry, rw, rh) = self.td_upgrade_button_rects[i];
                            if rw > 0.0
                                && mouse.x >= rx
                                && mouse.x <= rx + rw
                                && mouse.y >= ry
                                && mouse.y <= ry + rh
                            {
                                new_upgrade_hover = Some(i);
                                break;
                            }
                        }
                        if new_upgrade_hover != self.hovered_upgrade {
                            self.hovered_upgrade = new_upgrade_hover;
                            match new_upgrade_hover {
                                Some(idx) => {
                                    let path_maxed_tip = levels[idx] >= TD_UI_MAX_UPGRADE_LEVEL;
                                    let (title, desc) = if path_maxed_tip {
                                        // 查最後一級的升級名作為標題
                                        let last_name = self
                                            .td_upgrade_defs
                                            .get(&(
                                                kind_key.clone(),
                                                idx as u8,
                                                TD_UI_MAX_UPGRADE_LEVEL,
                                            ))
                                            .map(|(n, _, _)| n.clone())
                                            .unwrap_or_else(|| "MAX".to_string());
                                        (last_name, "已升至最高級別".to_string())
                                    } else {
                                        let next_lvl = levels[idx] + 1;
                                        self.td_upgrade_defs
                                            .get(&(kind_key.clone(), idx as u8, next_lvl))
                                            .map(|(n, d, _)| (n.clone(), td_upgrade_effect_text(d)))
                                            .unwrap_or_else(|| {
                                                ("?".to_string(), "無說明".to_string())
                                            })
                                    };
                                    ui.send(
                                        self.ui_upgrade_tooltip_title,
                                        TextMessage::Text(title),
                                    );
                                    ui.send(self.ui_upgrade_tooltip_desc, TextMessage::Text(desc));
                                    ui.send(
                                        self.ui_upgrade_tooltip_desc2,
                                        TextMessage::Text(String::new()),
                                    );
                                }
                                None => {
                                    ui.send(
                                        self.ui_upgrade_tooltip_title,
                                        TextMessage::Text(String::new()),
                                    );
                                    ui.send(
                                        self.ui_upgrade_tooltip_desc,
                                        TextMessage::Text(String::new()),
                                    );
                                    ui.send(
                                        self.ui_upgrade_tooltip_desc2,
                                        TextMessage::Text(String::new()),
                                    );
                                    ui.send(
                                        self.ui_upgrade_tooltip_bg,
                                        WidgetMessage::DesiredPosition(Vector2::new(
                                            -9999.0, -9999.0,
                                        )),
                                    );
                                    ui.send(
                                        self.ui_upgrade_tooltip_title,
                                        WidgetMessage::DesiredPosition(Vector2::new(
                                            -9999.0, -9999.0,
                                        )),
                                    );
                                    ui.send(
                                        self.ui_upgrade_tooltip_desc,
                                        WidgetMessage::DesiredPosition(Vector2::new(
                                            -9999.0, -9999.0,
                                        )),
                                    );
                                    ui.send(
                                        self.ui_upgrade_tooltip_desc2,
                                        WidgetMessage::DesiredPosition(Vector2::new(
                                            -9999.0, -9999.0,
                                        )),
                                    );
                                }
                            }
                        }
                        // 每 frame 定位（show_info 時壓制，避免雙層）
                        if let Some(hover_idx) = self
                            .hovered_upgrade
                            .filter(|_| !self.ui_td_selected_panel.show_info)
                        {
                            let sx = self.window_size.x / TD_UI_REF_W;
                            let sy = self.window_size.y / TD_UI_REF_H;
                            // 跟左邊棕色區塊一樣大：參考 244×160
                            let box_w = 244.0 * sx;
                            let box_h = 160.0 * sy;
                            let pr = self.ui_td_selected_panel.panel_rect;
                            let panel_right = if pr.x > self.window_size.x * 0.5 {
                                pr.x - box_w - 8.0
                            } else {
                                pr.right() + 8.0
                            };
                            // 對齊各列的 row_y（305, 497, 689 在參考空間，加上 panel 頂部 45）
                            let row_ref_y = 305.0 + hover_idx as f32 * 192.0;
                            let ty = (45.0 + row_ref_y + 15.0) * sy;
                            ui.send(
                                self.ui_upgrade_tooltip_bg,
                                WidgetMessage::DesiredPosition(Vector2::new(panel_right, ty)),
                            );
                            ui.send(self.ui_upgrade_tooltip_bg, WidgetMessage::Width(box_w));
                            ui.send(self.ui_upgrade_tooltip_bg, WidgetMessage::Height(box_h));
                            let pad = 12.0 * sx;
                            // 標題浮在框框上方（和按鈕一樣，超出 15 ref px）
                            ui.send(
                                self.ui_upgrade_tooltip_title,
                                WidgetMessage::DesiredPosition(Vector2::new(
                                    panel_right + pad,
                                    ty - 15.0 * sy,
                                )),
                            );
                            ui.send(
                                self.ui_upgrade_tooltip_title,
                                WidgetMessage::Width(box_w - pad * 2.0),
                            );
                            // 說明文字（全部放進單一 widget，行距自然一致）
                            let desc_y = ty + 48.0 * sy;
                            ui.send(
                                self.ui_upgrade_tooltip_desc,
                                WidgetMessage::DesiredPosition(Vector2::new(
                                    panel_right + pad,
                                    desc_y,
                                )),
                            );
                            ui.send(
                                self.ui_upgrade_tooltip_desc,
                                WidgetMessage::Width(box_w - pad * 2.0),
                            );
                            ui.send(
                                self.ui_upgrade_tooltip_desc2,
                                WidgetMessage::DesiredPosition(Vector2::new(-9999.0, -9999.0)),
                            );
                        }
                    }
                    // 隱藏舊的單路線元素
                    for node in [
                        self.ui_td_selected_panel.level_section_bg,
                        self.ui_td_selected_panel.level_title_bar_bg,
                        self.ui_td_selected_panel.upgrade_section_bg,
                        self.ui_td_selected_panel.unlock_title_bar_bg,
                        self.ui_td_selected_panel.upgrade_green_bg,
                        self.ui_td_selected_panel.upgrade_path_btn_bg,
                        self.ui_td_selected_panel.sell_top_mask,
                    ] {
                        ui.send(
                            node,
                            WidgetMessage::DesiredPosition(Vector2::new(
                                UI_HIDDEN_POS,
                                UI_HIDDEN_POS,
                            )),
                        );
                    }
                    for text in [
                        self.ui_td_selected_panel.level_num_text,
                        self.ui_td_selected_panel.level_label_text,
                        self.ui_td_selected_panel.flavor_text_node,
                        self.ui_td_selected_panel.unlock_label_text,
                        self.ui_td_selected_panel.upgrade_green_price,
                        self.ui_td_selected_panel.upgrade_path_btn_text,
                        self.ui_td_selected_panel.next_effect_text,
                    ] {
                        ui.send(
                            text,
                            WidgetMessage::DesiredPosition(Vector2::new(
                                UI_HIDDEN_POS,
                                UI_HIDDEN_POS,
                            )),
                        );
                    }
                    // ── Sell section （row3 底部 = 313+192*2+190=887，加 4px 間距）──
                    let sell_section_rect = rr(0.0, 891.0, 380.0, 80.0);
                    ui.send(
                        self.ui_td_selected_panel.sell_section_bg,
                        WidgetMessage::DesiredPosition(sell_section_rect.pos()),
                    );
                    ui.send(
                        self.ui_td_selected_panel.sell_section_bg,
                        WidgetMessage::Width(sell_section_rect.w),
                    );
                    ui.send(
                        self.ui_td_selected_panel.sell_section_bg,
                        WidgetMessage::Height(sell_section_rect.h),
                    );
                    let coin_icon_rect = rr(20.0, 917.0, 28.0, 28.0);
                    ui.send(
                        self.ui_td_selected_panel.sell_coin_icon,
                        WidgetMessage::DesiredPosition(coin_icon_rect.pos()),
                    );
                    ui.send(
                        self.ui_td_selected_panel.sell_coin_icon,
                        WidgetMessage::Width(coin_icon_rect.w),
                    );
                    ui.send(
                        self.ui_td_selected_panel.sell_coin_icon,
                        WidgetMessage::Height(coin_icon_rect.h),
                    );
                    let coin_rect = rr(52.0, 891.0, 110.0, 70.0);
                    ui.send(
                        self.ui_td_selected_panel.sell_coin_text,
                        WidgetMessage::DesiredPosition(coin_rect.pos()),
                    );
                    ui.send(
                        self.ui_td_selected_panel.sell_coin_text,
                        WidgetMessage::Width(coin_rect.w),
                    );
                    ui.send(
                        self.ui_td_selected_panel.sell_coin_text,
                        WidgetMessage::Height(coin_rect.h),
                    );
                    ui.send(
                        self.ui_td_selected_panel.sell_coin_text,
                        TextMessage::Text(format!("${}", refund)),
                    );
                    let sell_red_rect = rr(200.0, 906.0, 155.0, 50.0);
                    self.ui_td_selected_panel.sell_red_rect = sell_red_rect;
                    self.td_sell_button_rect = sell_red_rect.tuple();
                    ui.send(
                        self.ui_td_selected_panel.sell_red_bg,
                        WidgetMessage::DesiredPosition(sell_red_rect.pos()),
                    );
                    ui.send(
                        self.ui_td_selected_panel.sell_red_bg,
                        WidgetMessage::Width(sell_red_rect.w),
                    );
                    ui.send(
                        self.ui_td_selected_panel.sell_red_bg,
                        WidgetMessage::Height(sell_red_rect.h),
                    );
                    let sell_text_rect = rr(200.0, 891.0, 155.0, 70.0);
                    ui.send(
                        self.ui_td_selected_panel.sell_red_text,
                        WidgetMessage::DesiredPosition(sell_text_rect.pos()),
                    );
                    ui.send(
                        self.ui_td_selected_panel.sell_red_text,
                        WidgetMessage::Width(sell_text_rect.w),
                    );
                    ui.send(
                        self.ui_td_selected_panel.sell_red_text,
                        WidgetMessage::Height(sell_text_rect.h),
                    );
                    // Hide legacy elements
                    for node in [
                        self.ui_td_selected_panel.tower_card_bg,
                        self.ui_td_selected_panel.refund_bg,
                        self.ui_td_selected_panel.sell_icon,
                    ] {
                        ui.send(
                            node,
                            WidgetMessage::DesiredPosition(Vector2::new(
                                UI_HIDDEN_POS,
                                UI_HIDDEN_POS,
                            )),
                        );
                    }
                    for text in [
                        self.ui_td_selected_panel.summary_text,
                        self.ui_td_selected_panel.gold_text,
                    ] {
                        ui.send(
                            text,
                            WidgetMessage::DesiredPosition(Vector2::new(
                                UI_HIDDEN_POS,
                                UI_HIDDEN_POS,
                            )),
                        );
                    }
                    for text in [self.ui_td_sell_name_text, self.ui_td_sell_button_text] {
                        ui.send(
                            text,
                            WidgetMessage::DesiredPosition(Vector2::new(
                                UI_HIDDEN_POS,
                                UI_HIDDEN_POS,
                            )),
                        );
                    }
                } else {
                    for node in [
                        self.ui_td_selected_panel.bg,
                        self.ui_td_selected_panel.body_bg,
                        self.ui_td_selected_panel.header_strip_bg,
                        self.ui_td_selected_panel.header_strip_bottom_mask,
                        self.ui_td_selected_panel.header_bg,
                        self.ui_td_selected_panel.close_btn_bg,
                        self.ui_td_selected_panel.image_area_bg,
                        self.ui_td_selected_panel.path_left_bg,
                        self.ui_td_selected_panel.path_right_bg,
                        self.ui_td_selected_panel.tower_icon,
                        self.ui_td_selected_panel.level_section_bg,
                        self.ui_td_selected_panel.level_title_bar_bg,
                        self.ui_td_selected_panel.level_badge_bg,
                        self.ui_td_selected_panel.upgrade_section_bg,
                        self.ui_td_selected_panel.unlock_title_bar_bg,
                        self.ui_td_selected_panel.upgrade_green_bg,
                        self.ui_td_selected_panel.upgrade_path_btn_bg,
                        self.ui_td_selected_panel.sell_section_bg,
                        self.ui_td_selected_panel.sell_top_mask,
                        self.ui_td_selected_panel.sell_coin_icon,
                        self.ui_td_selected_panel.sell_red_bg,
                        self.ui_td_selected_panel.tower_card_bg,
                        self.ui_td_selected_panel.refund_bg,
                        self.ui_td_selected_panel.sell_icon,
                        self.ui_td_selected_panel.btd6_tower_icon,
                    ] {
                        ui.send(
                            node,
                            WidgetMessage::DesiredPosition(Vector2::new(
                                UI_HIDDEN_POS,
                                UI_HIDDEN_POS,
                            )),
                        );
                    }
                    for text in [
                        self.ui_td_selected_panel.header_title,
                        self.ui_td_selected_panel.pops_text,
                        self.ui_td_selected_panel.close_btn_text,
                        self.ui_td_selected_panel.path_left_text,
                        self.ui_td_selected_panel.path_right_text,
                        self.ui_td_selected_panel.path_name_label,
                        self.ui_td_selected_panel.level_num_text,
                        self.ui_td_selected_panel.level_label_text,
                        self.ui_td_selected_panel.flavor_text_node,
                        self.ui_td_selected_panel.unlock_label_text,
                        self.ui_td_selected_panel.upgrade_green_price,
                        self.ui_td_selected_panel.upgrade_path_btn_text,
                        self.ui_td_selected_panel.next_effect_text,
                        self.ui_td_selected_panel.sell_coin_text,
                        self.ui_td_selected_panel.sell_red_text,
                        self.ui_td_selected_panel.summary_text,
                        self.ui_td_selected_panel.gold_text,
                        self.ui_td_sell_name_text,
                        self.ui_td_sell_button_text,
                    ] {
                        ui.send(
                            text,
                            WidgetMessage::DesiredPosition(Vector2::new(
                                UI_HIDDEN_POS,
                                UI_HIDDEN_POS,
                            )),
                        );
                    }
                    for i in 0..3 {
                        for node in [
                            self.ui_td_selected_panel.upgrade_bgs[i],
                            self.ui_td_selected_panel.upgrade_icons[i],
                            self.ui_td_selected_panel.upgrade_row_bgs[i],
                        ] {
                            ui.send(
                                node,
                                WidgetMessage::DesiredPosition(Vector2::new(
                                    UI_HIDDEN_POS,
                                    UI_HIDDEN_POS,
                                )),
                            );
                        }
                        for text in [
                            self.ui_td_selected_panel.upgrade_pip_texts[i],
                            self.ui_td_selected_panel.upgrade_status_texts[i],
                            self.ui_td_selected_panel.upgrade_price_texts[i],
                            self.ui_td_selected_panel.upgrade_name_texts[i],
                        ] {
                            ui.send(
                                text,
                                WidgetMessage::DesiredPosition(Vector2::new(
                                    UI_HIDDEN_POS,
                                    UI_HIDDEN_POS,
                                )),
                            );
                        }
                        ui.send(
                            self.ui_td_upgrade_buttons[i],
                            WidgetMessage::DesiredPosition(Vector2::new(
                                UI_HIDDEN_POS,
                                UI_HIDDEN_POS,
                            )),
                        );
                        self.td_upgrade_button_rects[i] = (UI_HIDDEN_POS, UI_HIDDEN_POS, 0.0, 0.0);
                    }
                    self.td_sell_button_rect = (UI_HIDDEN_POS, UI_HIDDEN_POS, 0.0, 0.0);
                    self.td_target_priority_button_rect = (UI_HIDDEN_POS, UI_HIDDEN_POS, 0.0, 0.0);
                    self.ui_td_selected_panel.close_btn_rect = UiRect::default();
                    self.ui_td_selected_panel.path_left_rect = UiRect::default();
                    self.ui_td_selected_panel.path_right_rect = UiRect::default();
                    self.ui_td_selected_panel.info_btn_rect = UiRect::default();
                    self.ui_td_selected_panel.show_info = false;
                    ui.send(
                        self.ui_td_selected_panel.info_btn_bg,
                        WidgetMessage::DesiredPosition(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS)),
                    );
                    ui.send(
                        self.ui_td_selected_panel.info_btn_text,
                        WidgetMessage::DesiredPosition(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS)),
                    );
                    ui.send(
                        self.ui_td_selected_panel.info_overlay_bg,
                        WidgetMessage::DesiredPosition(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS)),
                    );
                    for i in 0..4usize {
                        ui.send(
                            self.ui_td_selected_panel.info_stat_texts[i],
                            WidgetMessage::DesiredPosition(Vector2::new(
                                UI_HIDDEN_POS,
                                UI_HIDDEN_POS,
                            )),
                        );
                    }
                    // 面板關閉時清除 info row tooltips
                    for i in 0..3usize {
                        ui.send(
                            self.ui_td_selected_panel.info_row_bgs[i],
                            WidgetMessage::DesiredPosition(Vector2::new(
                                UI_HIDDEN_POS,
                                UI_HIDDEN_POS,
                            )),
                        );
                        ui.send(
                            self.ui_td_selected_panel.info_row_titles[i],
                            WidgetMessage::DesiredPosition(Vector2::new(
                                UI_HIDDEN_POS,
                                UI_HIDDEN_POS,
                            )),
                        );
                        ui.send(
                            self.ui_td_selected_panel.info_row_descs[i],
                            WidgetMessage::DesiredPosition(Vector2::new(
                                UI_HIDDEN_POS,
                                UI_HIDDEN_POS,
                            )),
                        );
                        ui.send(
                            self.ui_td_selected_panel.info_row_descs2[i],
                            WidgetMessage::DesiredPosition(Vector2::new(
                                UI_HIDDEN_POS,
                                UI_HIDDEN_POS,
                            )),
                        );
                    }
                    // 面板關閉時清除升級 tooltip
                    if self.hovered_upgrade.is_some() {
                        self.hovered_upgrade = None;
                        ui.send(
                            self.ui_upgrade_tooltip_bg,
                            WidgetMessage::DesiredPosition(Vector2::new(-9999.0, -9999.0)),
                        );
                        ui.send(
                            self.ui_upgrade_tooltip_title,
                            WidgetMessage::DesiredPosition(Vector2::new(-9999.0, -9999.0)),
                        );
                        ui.send(
                            self.ui_upgrade_tooltip_desc,
                            WidgetMessage::DesiredPosition(Vector2::new(-9999.0, -9999.0)),
                        );
                        ui.send(
                            self.ui_upgrade_tooltip_desc2,
                            WidgetMessage::DesiredPosition(Vector2::new(-9999.0, -9999.0)),
                        );
                    }
                }
            }

            // 技能冷卻每 frame 遞減
            for cd in self.hero_state.ability_cd.values_mut() {
                if *cd > 0.0 {
                    *cd = (*cd - dt).max(0.0);
                }
            }

            // Buff 倒數：本地每 frame 遞減，讓面板顯示連續變化；
            // 下次 backend push 的 snapshot 會重設成權威值，避免漂移。
            for b in self.hero_state.buffs.iter_mut() {
                if b.remaining > 0.0 {
                    b.remaining = (b.remaining - dt).max(0.0);
                }
            }
            // remaining = 0 的有限期 buff 從本地清掉（權威值會在下次 push 糾正）
            self.hero_state.buffs.retain(|b| b.remaining != 0.0);

            // 更新技能 icon 下方的等級點 + 中央 CD 數字
            let skill_points = self.hero_state.skill_points;
            for i in 0..4 {
                let id = self
                    .hero_state
                    .abilities
                    .get(i)
                    .cloned()
                    .unwrap_or_default();
                let lvl = self
                    .hero_state
                    .ability_levels
                    .get(&id)
                    .copied()
                    .unwrap_or(0);
                let max = ability_max_level(&self.ability_info_map, &id);
                let icon_path = self
                    .ability_info_map
                    .get(&id)
                    .and_then(|info| {
                        (!info.icon_path.is_empty()).then_some(info.icon_path.as_str())
                    })
                    .unwrap_or(ABILITY_ICON_FALLBACK_PATH)
                    .to_string();
                if self.ability_icon_paths[i] != icon_path {
                    self.ability_icon_paths[i] = icon_path.clone();
                    let icon_tex = self
                        .ability_icon_texture(&icon_path)
                        .or_else(|| self.ability_icon_texture(ABILITY_ICON_FALLBACK_PATH));
                    self.ability_textures[i] = icon_tex.clone();
                    if self.ui_ability_icons[i] != Handle::<UiNode>::NONE {
                        ui.send(self.ui_ability_icons[i], ImageMessage::Texture(icon_tex));
                    }
                }
                // 等級點 ● ○
                let dots: String = (0..max.max(1))
                    .map(|n| if n < lvl { "●" } else { "○" })
                    .collect::<Vec<_>>()
                    .join(" ");
                ui.send(self.ui_ability_level_text[i], TextMessage::Text(dots));

                // CD 數字
                let remaining = self.hero_state.ability_cd.get(&id).copied().unwrap_or(0.0);
                let cd_str = if remaining >= 1.0 {
                    format!("{:.0}", remaining.ceil())
                } else if remaining > 0.0 {
                    format!("{:.1}", remaining)
                } else {
                    String::new()
                };
                ui.send(self.ui_ability_cd_text[i], TextMessage::Text(cd_str));

                let can_upgrade = !id.is_empty() && skill_points > 0 && lvl < max;
                if can_upgrade {
                    let (x, y, w, _) = self.ability_icon_rects[i];
                    self.ability_upgrade_button_rects[i] = (x, y - 32.0, w, 32.0);
                    if self.ui_ability_key_text[i] != Handle::<Text>::NONE {
                        ui.send(
                            self.ui_ability_key_text[i],
                            WidgetMessage::DesiredPosition(Vector2::new(x + 20.0, y - 54.0)),
                        );
                    }
                    ui.send(
                        self.ui_ability_upgrade_buttons[i],
                        TextMessage::Text("▲".to_string()),
                    );
                } else {
                    self.ability_upgrade_button_rects[i] = (-9999.0, -9999.0, 0.0, 0.0);
                    ui.send(
                        self.ui_ability_upgrade_buttons[i],
                        TextMessage::Text(String::new()),
                    );
                }
            }
            // Inventory 顯示
            let hs = &self.hero_state;
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
            let is_td_hud = hs.lives > 0 || self.is_td_mode;
            let hud = if is_td_hud {
                String::new()
            } else {
                format!(
                    "HP {:.0}/{:.0}  LV {}  XP {}/{}  GOLD {}  SP {}  |  {}",
                    hs.hp, hs.max_hp, hs.level, hs.xp, hs.xp_next, hs.gold, hs.skill_points, inv,
                )
            };
            ui.send(self.ui_hud_text, TextMessage::Text(hud));
            if is_td_hud {
                let (sx, sy) = td_ui_ref_scale(self.window_size);
                let hud_items = [
                    ("hp", format!("{:.0}/{:.0}", hs.hp, hs.max_hp), 690.0),
                    ("lives", format!("{}", hs.lives), 900.0),
                    ("gold", format!("${}", hs.gold), 1090.0),
                ];
                for (i, (_, text, x_ref)) in hud_items.iter().enumerate() {
                    let icon_rect = td_ui_ref_rect(self.window_size, *x_ref, 36.0, 58.0, 58.0);
                    ui.send(
                        self.ui_td_top_hud_icons[i],
                        WidgetMessage::DesiredPosition(icon_rect.pos()),
                    );
                    ui.send(
                        self.ui_td_top_hud_icons[i],
                        WidgetMessage::Width(icon_rect.w),
                    );
                    ui.send(
                        self.ui_td_top_hud_icons[i],
                        WidgetMessage::Height(icon_rect.h),
                    );
                    ui.send(
                        self.ui_td_top_hud_texts[i],
                        WidgetMessage::DesiredPosition(Vector2::new(
                            icon_rect.x + 66.0 * sx,
                            icon_rect.y + 12.0 * sy,
                        )),
                    );
                    ui.send(
                        self.ui_td_top_hud_texts[i],
                        WidgetMessage::Width(180.0 * sx),
                    );
                    ui.send(self.ui_td_top_hud_texts[i], TextMessage::Text(text.clone()));
                }
                let wave_rect = td_ui_ref_rect(self.window_size, 1250.0, 48.0, 260.0, 46.0);
                ui.send(
                    self.ui_td_wave_text,
                    WidgetMessage::DesiredPosition(wave_rect.pos()),
                );
                ui.send(self.ui_td_wave_text, WidgetMessage::Width(wave_rect.w));
                ui.send(
                    self.ui_td_wave_text,
                    TextMessage::Text(td_round_status_label(
                        self.round_is_running,
                        self.current_round,
                        self.total_rounds,
                    )),
                );
            } else {
                for i in 0..3 {
                    ui.send(
                        self.ui_td_top_hud_icons[i],
                        WidgetMessage::DesiredPosition(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS)),
                    );
                    ui.send(
                        self.ui_td_top_hud_texts[i],
                        WidgetMessage::DesiredPosition(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS)),
                    );
                }
                ui.send(
                    self.ui_td_wave_text,
                    WidgetMessage::DesiredPosition(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS)),
                );
            }

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
                let tag = |attr: &str| {
                    if hs.primary_attribute == attr {
                        "★"
                    } else {
                        " "
                    }
                };
                let mut ability_lines = String::new();
                for (i, id) in hs.abilities.iter().enumerate().take(4) {
                    let lvl = hs.ability_levels.get(id).copied().unwrap_or(0);
                    let key = ["Q", "W", "E", "R"].get(i).copied().unwrap_or("?");
                    ability_lines.push_str(&format!("\n[{}] {:<22} 等級 {}/4", key, id, lvl));
                }
                let aps = if hs.attack_interval > 0.0 {
                    1.0 / hs.attack_interval
                } else {
                    0.0
                };
                let command_line = hs
                    .entity_id
                    .and_then(|id| {
                        self.latest_entities
                            .iter()
                            .find(|entity| entity.entity_id == id)
                    })
                    .and_then(|entity| entity.hero_command.as_ref())
                    .map(|command| hero_command_status_text(command))
                    .unwrap_or_else(|| "Idle".to_string());
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
                     命令 {}\n\
                     ── 技能 ──{}\n\
                     ── 效果 ──{}",
                    header,
                    hs.level,
                    hs.xp,
                    hs.xp_next,
                    hs.skill_points,
                    hs.strength,
                    tag("strength"),
                    hs.agility,
                    tag("agility"),
                    hs.intelligence,
                    tag("intelligence"),
                    hs.hp as i32,
                    hs.max_hp as i32,
                    hs.gold,
                    hs.armor,
                    hs.magic_resist,
                    hs.move_speed,
                    hs.attack_damage,
                    hs.attack_interval,
                    hs.attack_range,
                    hs.bullet_speed,
                    aps,
                    command_line,
                    ability_lines,
                    buff_lines,
                );
                // 選中塔時隱藏英雄屬性面板，避免文字在面板旁邊穿透顯示。
                let panel_x = if self.selected_tower_entity.is_some() {
                    UI_HIDDEN_POS
                } else {
                    10.0
                };
                let panel_y = 50.0;
                ui.send(
                    self.ui_hero_stats_panel,
                    WidgetMessage::DesiredPosition(Vector2::new(panel_x, panel_y)),
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

            // 熱鍵設定面板（F1 開關）
            self.update_hotkey_panel(ui);
            // 英雄知識面板
            self.update_gk_panel(ui);

            let end_str = if self.game_ended {
                "VICTORY!".to_string()
            } else {
                String::new()
            };
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
                            ui.send(
                                self.ui_tooltip_icon,
                                ImageMessage::Texture(Some(tex.clone())),
                            );
                        }
                        // 查 ability info 組 tooltip 字串
                        let hs = &self.hero_state;
                        let ability_id = hs.abilities.get(idx).cloned().unwrap_or_default();
                        let cur_lvl = hs.ability_levels.get(&ability_id).copied().unwrap_or(0);
                        let tooltip_str = if let Some(info) = self.ability_info_map.get(&ability_id)
                        {
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
                if tx + 460.0 > win.x {
                    tx = (win.x - 460.0).max(0.0);
                }
                if ty < 0.0 {
                    ty = 0.0;
                }
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
        drop(ui_span);

        let total_ns = frame_t0.elapsed().as_nanos();
        let frame_stats_span =
            tracing::trace_span!("omfx::frame::statistics", perfetto = true).entered();
        let snapshot_reused = self.render_pacing_last_snapshot_tick.is_some()
            && self.render_pacing_last_snapshot_tick == last_rendered_snapshot_tick;
        self.render_pacing_last_frame_at = Some(frame_t0);
        let render_target_tps = RENDER_UPDATE_TPS;
        self.frame_profile.record_frame_interval(frame_interval);
        self.frame_profile
            .record_render_pacing(snapshot_reused, render_target_tps);
        if let Some(ref sim) = self.sim_runner_handle {
            if let Ok(diagnostics) = sim.diagnostics.try_lock() {
                self.sim_speed_tps = diagnostics.sim_tps;
                self.frame_profile.record_sim_diagnostics(&diagnostics);
            }
        }
        self.frame_profile.events_ns += events_ns;
        self.frame_profile.lockstep_ns += lockstep_ns;
        self.frame_profile.snapshot_ns += snapshot_ns;
        self.frame_profile.render_bridge_ns += render_bridge_ns;
        self.frame_profile.interp_ns += interp_ns;
        self.frame_profile.visual_ns += visual_ns;
        self.frame_profile.proj_ns += proj_ns;
        self.frame_profile.cam_ns += cam_ns;
        self.frame_profile.ui_ns += ui_ns;
        self.frame_profile.total_ns += total_ns;
        self.frame_profile.events_drained += events_drained_local;
        self.frame_profile.creeps_seen += self.network_entities.len() as u64;
        self.frame_profile.projectiles_seen += self.client_projectiles.len() as u64;
        // Fyrox 渲染器統計資料（即時幀時間，包括渲染提交 + GPU + 垂直同步等待）
        if let fyrox::engine::GraphicsContext::Initialized(ref gc) = context.graphics_context {
            self.frame_profile
                .record_render_stats(&gc.renderer.get_statistics());
        }
        self.frame_profile.finish_frame();
        drop(frame_stats_span);
        drop(frame_span);

        Ok(())
    }

    fn on_os_event(&mut self, event: &Event<()>, mut context: PluginContext) -> GameResult {
        let event_us = wall_clock_us();
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
                if self.td_shop_scroll_dragging {
                    if self.td_shop_max_scroll > 0.0 {
                        let track = self.ui_td_right_panel.scroll_track_rect;
                        let thumb = self.ui_td_right_panel.scroll_thumb_rect;
                        let travel = (track.h - thumb.h).max(1.0);
                        let dy = cursor.y - self.td_shop_scroll_drag_start_y;
                        let offset = self.td_shop_scroll_drag_start_offset
                            + dy / travel * self.td_shop_max_scroll;
                        self.set_td_shop_scroll_offset(offset);
                    } else {
                        self.td_shop_scroll_dragging = false;
                    }
                }
                if self.settings_dragging_sfx {
                    let track = self.ui_settings.sfx.track_rect;
                    let pct = ((cursor.x - track.x) / track.w.max(1.0)).clamp(0.0, 1.0);
                    self.settings_sfx_volume = pct;
                }
                if self.settings_dragging_music {
                    let track = self.ui_settings.music.track_rect;
                    let pct = ((cursor.x - track.x) / track.w.max(1.0)).clamp(0.0, 1.0);
                    self.settings_music_volume = pct;
                }
                if self.settings_dragging_speed {
                    let track = self.ui_settings.speed.track_rect;
                    let pct = ((cursor.x - track.x) / track.w.max(1.0)).clamp(0.0, 1.0);
                    self.settings_speed_value = pct;
                }
            }
            Event::WindowEvent {
                event: WindowEvent::MouseWheel { delta, .. },
                ..
            } => {
                let screen = self.mouse_screen_pos;
                if self
                    .tower_ability_bar_rects
                    .iter()
                    .any(|rect| rect.x > -9000.0 && rect.contains(screen))
                {
                    let direction = match delta {
                        MouseScrollDelta::LineDelta(x, y) => x + y,
                        MouseScrollDelta::PixelDelta(pos) => (pos.x + pos.y) as f32,
                    };
                    let ability_count = self
                        .latest_entities
                        .iter()
                        .filter(|entity| {
                            entity.owner_player_id == Some(self.local_player_id)
                                && entity.hp > 0
                                && entity.tower_active_ability.is_some()
                        })
                        .count();
                    let page_count = ability_count.div_ceil(TOWER_ABILITY_BAR_PAGE_SIZE).max(1);
                    if direction < 0.0 {
                        self.tower_ability_bar_interaction
                            .change_page(1, page_count);
                    } else if direction > 0.0 {
                        self.tower_ability_bar_interaction
                            .change_page(-1, page_count);
                    }
                    return Ok(());
                }
                let viewport = self.ui_td_right_panel.viewport_rect;
                let track = self.ui_td_right_panel.scroll_track_rect;
                if self.td_shop_max_scroll > 0.0
                    && (viewport.contains(screen) || track.contains(screen))
                {
                    let (_, sy) = td_ui_ref_scale(self.window_size);
                    let delta_ref = match delta {
                        MouseScrollDelta::LineDelta(_, y) => -y * 120.0,
                        MouseScrollDelta::PixelDelta(pos) => -(pos.y as f32) / sy.max(0.01),
                    };
                    self.set_td_shop_scroll_offset(self.td_shop_scroll_offset + delta_ref);
                }
            }
            Event::WindowEvent {
                event:
                    WindowEvent::MouseInput {
                        button: MouseButton::Left,
                        state: ElementState::Released,
                        ..
                    },
                ..
            } => {
                self.td_shop_scroll_dragging = false;
                self.settings_dragging_sfx = false;
                self.settings_dragging_music = false;
                self.settings_dragging_speed = false;
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
                // 左鍵 UI：技能升級三角按鈕優先，避免命中後落到地圖/TD 點擊邏輯。
                let screen = self.mouse_screen_pos;
                // 設定面板滑桿優先處理（在 pregame 按鈕前攔截）
                if self.pregame_runtime.state == pregame::PregameState::Settings {
                    let sfx_track = self.ui_settings.sfx.track_rect;
                    let music_track = self.ui_settings.music.track_rect;
                    let speed_track = self.ui_settings.speed.track_rect;
                    if sfx_track.contains(screen) {
                        let pct = ((screen.x - sfx_track.x) / sfx_track.w.max(1.0)).clamp(0.0, 1.0);
                        self.settings_sfx_volume = pct;
                        self.settings_dragging_sfx = true;
                        return Ok(());
                    }
                    if music_track.contains(screen) {
                        let pct =
                            ((screen.x - music_track.x) / music_track.w.max(1.0)).clamp(0.0, 1.0);
                        self.settings_music_volume = pct;
                        self.settings_dragging_music = true;
                        return Ok(());
                    }
                    if speed_track.contains(screen) {
                        let pct =
                            ((screen.x - speed_track.x) / speed_track.w.max(1.0)).clamp(0.0, 1.0);
                        self.settings_speed_value = pct;
                        self.settings_dragging_speed = true;
                        return Ok(());
                    }
                }
                if self.handle_pregame_click(screen) {
                    self.sfx_pending.push(SfxKind::ButtonClick);
                    return Ok(());
                }
                if self.handle_in_game_return_click(screen) {
                    return Ok(());
                }
                let mut hit_ui = false;
                let mut play_button_sfx = false;

                let item_count = self
                    .latest_entities
                    .iter()
                    .filter(|entity| {
                        entity.owner_player_id == Some(self.local_player_id)
                            && entity.hp > 0
                            && entity.tower_active_ability.is_some()
                    })
                    .count();
                let page_model =
                    ability_bar_page_model(item_count, self.tower_ability_bar_interaction.page);
                if self.tower_ability_bar_previous_rect.contains(screen)
                    && page_model.previous_enabled
                {
                    self.tower_ability_bar_interaction
                        .change_page(-1, page_model.page_count);
                    hit_ui = true;
                    play_button_sfx = true;
                } else if self.tower_ability_bar_next_rect.contains(screen)
                    && page_model.next_enabled
                {
                    self.tower_ability_bar_interaction
                        .change_page(1, page_model.page_count);
                    hit_ui = true;
                    play_button_sfx = true;
                }

                if !hit_ui {
                    if let Some(index) = self
                        .tower_ability_bar_rects
                        .iter()
                        .position(|rect| rect.x > -9000.0 && rect.contains(screen))
                    {
                        let item = self.tower_ability_bar_slot_bindings[index]
                            .as_ref()
                            .and_then(|key| {
                                self.tower_ability_bar_items
                                    .iter()
                                    .find(|item| &item.key == key)
                            })
                            .cloned();
                        if let Some(item) = item {
                            play_button_sfx = self.send_tower_ability_cast(item, event_us);
                        }
                        hit_ui = true;
                    }
                }

                for i in 0..4 {
                    if hit_ui {
                        break;
                    }
                    let (bx, by, bw, bh) = self.ability_upgrade_button_rects[i];
                    if bx > -9000.0
                        && screen.x >= bx
                        && screen.x <= bx + bw
                        && screen.y >= by
                        && screen.y <= by + bh
                    {
                        self.send_upgrade_ability_input_from(
                            i as u32,
                            lockstep_client::InputOriginKind::OsEvent,
                            event_us,
                        );
                        hit_ui = true;
                        play_button_sfx = true;
                        break;
                    }
                }

                // 技能 icon 本體點擊施法；三角升級按鈕已在上方優先處理。
                if !hit_ui {
                    for i in 0..4 {
                        let (bx, by, bw, bh) = self.ability_icon_rects[i];
                        if screen.x >= bx
                            && screen.x <= bx + bw
                            && screen.y >= by
                            && screen.y <= by + bh
                        {
                            self.send_cast_ability_input_from(
                                i as u32,
                                Some(self.mouse_world_pos),
                                lockstep_client::InputOriginKind::OsEvent,
                                event_us,
                            );
                            hit_ui = true;
                            play_button_sfx = true;
                            break;
                        }
                    }
                }

                // 右側 shop scrollbar 優先於 Start/買塔/地圖點擊。
                if !hit_ui && self.td_shop_max_scroll > 0.0 {
                    let thumb = self.ui_td_right_panel.scroll_thumb_rect;
                    let track = self.ui_td_right_panel.scroll_track_rect;
                    if thumb.contains(screen) {
                        self.td_shop_scroll_dragging = true;
                        self.td_shop_scroll_drag_start_y = screen.y;
                        self.td_shop_scroll_drag_start_offset = self.td_shop_scroll_offset;
                        hit_ui = true;
                    } else if track.contains(screen) {
                        let direction = if screen.y < thumb.y { -1.0 } else { 1.0 };
                        self.set_td_shop_scroll_offset(
                            self.td_shop_scroll_offset + direction * 630.0,
                        );
                        hit_ui = true;
                    }
                }

                // Auto-start checkbox above Start.
                if !hit_ui {
                    let (bx, by, bw, bh) = self.auto_start_checkbox_rect;
                    if bx > -9000.0
                        && screen.x >= bx
                        && screen.x <= bx + bw
                        && screen.y >= by
                        && screen.y <= by + bh
                    {
                        self.td_auto_start_enabled = !self.td_auto_start_enabled;
                        self.td_auto_start_sent_for_idle_round = false;
                        log::info!(
                            "TD auto-start {}",
                            if self.td_auto_start_enabled {
                                "enabled"
                            } else {
                                "disabled"
                            }
                        );
                        hit_ui = true;
                        play_button_sfx = true;
                    }
                }

                // 熱鍵設定面板：modal 點擊攔截（開啟時吃掉所有點擊）
                if !hit_ui && self.hotkey_panel_visible {
                    self.handle_hotkey_panel_click(screen);
                    hit_ui = true;
                    play_button_sfx = true;
                }

                // 英雄知識面板：modal 點擊攔截（開啟時吃掉所有點擊）
                if !hit_ui && self.gk_panel_visible {
                    self.handle_gk_panel_click(screen);
                    hit_ui = true;
                    play_button_sfx = true;
                }

                // 英雄知識入口按鈕（右側面板頂部）
                if !hit_ui {
                    let gr = self.gk_panel_ui.entry_rect;
                    if gr.w > 0.0 && gr.contains(screen) {
                        self.gk_panel_visible = !self.gk_panel_visible;
                        // 開啟面板時重新讀取 profile，確保顯示最新 KP（包含本局結算）
                        if self.gk_panel_visible {
                            self.load_gk_profile();
                        }
                        hit_ui = true;
                        play_button_sfx = true;
                    }
                }

                // 1. Start Round / speed toggle button — Phase 5.x lockstep send
                if !hit_ui {
                    let (bx, by, bw, bh) = self.start_round_button_rect;
                    if bx > -9000.0
                        && screen.x >= bx
                        && screen.x <= bx + bw
                        && screen.y >= by
                        && screen.y <= by + bh
                    {
                        self.trigger_start_round_button(event_us);
                        hit_ui = true;
                        play_button_sfx = true;
                    }
                }

                // Pause button — authoritative lockstep toggle.
                if !hit_ui {
                    let (bx, by, bw, bh) = self.pause_button_rect;
                    if bx > -9000.0
                        && screen.x >= bx
                        && screen.x <= bx + bw
                        && screen.y >= by
                        && screen.y <= by + bh
                    {
                        if !self.is_game_paused {
                            let input = omoba_core::kcp::game_proto::PlayerInput {
                                action: Some(
                                    omoba_core::kcp::game_proto::player_input::Action::TogglePause(
                                        omoba_core::kcp::game_proto::TogglePause {},
                                    ),
                                ),
                            };
                            self.send_lockstep_input_from(
                                input,
                                lockstep_client::InputOriginKind::OsEvent,
                                event_us,
                            );
                            log::info!("Pause → lockstep PlayerInput::TogglePause sent");
                        }
                        hit_ui = true;
                        play_button_sfx = true;
                    }
                }

                // 2. 4 塔按鈕
                if !hit_ui {
                    // 依動態 template_order 對應按鈕
                    let mut hit_idx: Option<usize> = None;
                    for (i, rect) in self.td_tower_button_rects.iter().enumerate() {
                        let (bx, by, bw, bh) = *rect;
                        if i >= self.td_template_order.len() {
                            break;
                        }
                        if screen.x >= bx
                            && screen.x <= bx + bw
                            && screen.y >= by
                            && screen.y <= by + bh
                        {
                            hit_idx = Some(i);
                            break;
                        }
                    }
                    if let Some(i) = hit_idx {
                        let uid = self.td_template_order[i].clone();
                        self.selected_tower_kind = Some(uid.clone());
                        self.selected_tower_entity = None;
                        self.ui_td_selected_panel.selected_path = 0;
                        log::info!("選中塔: {}", uid);
                        hit_ui = true;
                        play_button_sfx = true;
                    }
                }

                // BTD6 close 按鈕
                if !hit_ui && self.selected_tower_entity.is_some() {
                    let cr = self.ui_td_selected_panel.close_btn_rect;
                    if cr.w > 0.0 && cr.contains(screen) {
                        self.selected_tower_entity = None;
                        self.ui_td_selected_panel.selected_path = 0;
                        hit_ui = true;
                        play_button_sfx = true;
                    }
                }

                // i 說明按鈕 toggle
                if !hit_ui && self.selected_tower_entity.is_some() {
                    let ir = self.ui_td_selected_panel.info_btn_rect;
                    if ir.w > 0.0 && ir.contains(screen) {
                        self.ui_td_selected_panel.show_info = !self.ui_td_selected_panel.show_info;
                        hit_ui = true;
                        play_button_sfx = true;
                    }
                }

                // BTD6 路線切換箭頭 ◀ ▶
                if !hit_ui && self.selected_tower_entity.is_some() {
                    let lr = self.ui_td_selected_panel.path_left_rect;
                    let rr2 = self.ui_td_selected_panel.path_right_rect;
                    log::info!(
                        "[arrow] scr=({:.0},{:.0}) L=[{:.0},{:.0} {}x{}] R=[{:.0},{:.0} {}x{}]",
                        screen.x,
                        screen.y,
                        lr.x,
                        lr.y,
                        lr.w as u32,
                        lr.h as u32,
                        rr2.x,
                        rr2.y,
                        rr2.w as u32,
                        rr2.h as u32
                    );
                    if lr.w > 0.0 && lr.contains(screen) {
                        if self.ui_td_selected_panel.selected_path > 0 {
                            self.ui_td_selected_panel.selected_path -= 1;
                        }
                        hit_ui = true;
                        play_button_sfx = true;
                    } else if rr2.w > 0.0 && rr2.contains(screen) {
                        if self.ui_td_selected_panel.selected_path < 2 {
                            self.ui_td_selected_panel.selected_path += 1;
                        }
                        hit_ui = true;
                        play_button_sfx = true;
                    }
                }

                // 3. Sell 按鈕（只有有已選中塔時生效）
                if !hit_ui && self.selected_tower_entity.is_some() {
                    let (bx, by, bw, bh) = self.td_target_priority_button_rect;
                    if bx > -9000.0
                        && screen.x >= bx
                        && screen.x <= bx + bw
                        && screen.y >= by
                        && screen.y <= by + bh
                    {
                        self.cycle_selected_tower_priority(false, event_us);
                        hit_ui = true;
                        play_button_sfx = true;
                    }
                }

                if !hit_ui && self.selected_tower_entity.is_some() {
                    let (bx, by, bw, bh) = self.td_sell_button_rect;
                    if screen.x >= bx
                        && screen.x <= bx + bw
                        && screen.y >= by
                        && screen.y <= by + bh
                    {
                        // 階段 2.2：TowerSell 鎖步輸入（邏輯在
                        // send_tower_sell_for_selected，與 Backspace 熱鍵共用）。
                        self.send_tower_sell_for_selected(event_us);
                        hit_ui = true;
                        play_button_sfx = true;
                    }
                }

                // 3b. 3 條升級按鈕（必須在 tower-deselect 邏輯之前跑）
                if !hit_ui && self.selected_tower_entity.is_some() {
                    for path in 0u8..3 {
                        let (bx, by, bw, bh) = self.td_upgrade_button_rects[path as usize];
                        if bx > -9000.0
                            && screen.x >= bx
                            && screen.x < bx + bw
                            && screen.y >= by
                            && screen.y < by + bh
                        {
                            // 階段 2.3：TowerUpgrade 鎖步輸入（邏輯在
                            // send_tower_upgrade_for_selected，與 , . / 熱鍵共用）。
                            if self.send_tower_upgrade_for_selected(path, event_us) {
                                play_button_sfx = true;
                            }
                            hit_ui = true;
                            break;
                        }
                    }
                }

                if !hit_ui && self.attack_move_armed {
                    let queued = self.shift_held;
                    self.send_attack_move_input_from(
                        self.mouse_world_pos,
                        queued,
                        lockstep_client::InputOriginKind::OsEvent,
                        event_us,
                    );
                    self.attack_move_armed = false;
                    hit_ui = true;
                }

                // 4. 放置塔（如在選塔模式）。放完後若沒按 Ctrl 則自動取消
                if !hit_ui {
                    if let Some(kind) = self.selected_tower_kind.clone() {
                        let world_pos = self.mouse_world_pos;
                        let template = self.td_templates.get(&kind);
                        let block_reason = template
                            .and_then(|tpl| self.tower_placement_block_reason(tpl, world_pos));
                        if let Some(tpl) = template {
                            let backend_x = world_pos.x / WORLD_SCALE;
                            let backend_y = world_pos.y / WORLD_SCALE;
                            let result = block_reason
                                .as_ref()
                                .map(TowerPlacementBlockReason::diagnostic_code)
                                .unwrap_or("accepted");
                            if let Some(probe) = self.nearest_tower_road_probe(tpl, world_pos) {
                                log::info!(
                                    "TowerPlace probe kind='{}' pos_render=({:.3},{:.3}) pos_backend=({:.1},{:.1}) path_index={} segment_index={} distance_backend={:.2} road_half_width_backend={:.2} footprint_backend={:.2} placement_radius_backend={:.2} required_clearance_backend={:.2} world_scale={:.4} result={}",
                                    kind,
                                    world_pos.x,
                                    world_pos.y,
                                    backend_x,
                                    backend_y,
                                    probe.path_index + 1,
                                    probe.segment_index + 1,
                                    probe.distance_backend,
                                    TD_PATH_HALF_WIDTH_BACKEND,
                                    tpl.footprint_backend,
                                    tpl.placement_radius_backend,
                                    probe.required_clearance_backend,
                                    WORLD_SCALE,
                                    result,
                                );
                            } else {
                                log::info!(
                                    "TowerPlace probe kind='{}' pos_render=({:.3},{:.3}) pos_backend=({:.1},{:.1}) path=none road_half_width_backend={:.2} footprint_backend={:.2} placement_radius_backend={:.2} world_scale={:.4} result={}",
                                    kind,
                                    world_pos.x,
                                    world_pos.y,
                                    backend_x,
                                    backend_y,
                                    TD_PATH_HALF_WIDTH_BACKEND,
                                    tpl.footprint_backend,
                                    tpl.placement_radius_backend,
                                    WORLD_SCALE,
                                    result,
                                );
                            }
                        }
                        let has_template = self.td_templates.contains_key(&kind);
                        if has_template && block_reason.is_none() {
                            // 階段 2.1：TowerPlace 鎖步輸入。選定的塔類型
                            // 是unit_id字串（例如“tower_dart”）－轉換為
                            // 原型 u32 kind_id 通過 omoba_template_ids::tower_by_name。
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
                                    self.send_lockstep_input_from(
                                        input,
                                        lockstep_client::InputOriginKind::OsEvent,
                                        event_us,
                                    );
                                    log::info!(
                                        "Tower place lockstep input submitted: kind='{}' kind_id={} pos=({}, {})",
                                        kind, tid.0, pos.x, pos.y
                                    );
                                    self.sfx_pending.push(SfxKind::TowerPlace);
                                    if !self.ctrl_held {
                                        self.selected_tower_kind = None;
                                    }
                                }
                                None => {
                                    log::warn!(
                                        "Tower place: unknown kind name '{}' (no template_ids match) — skipped",
                                        kind
                                    );
                                }
                            }
                        } else {
                            log::info!(
                                "Tower place skipped by local placement validation: kind='{}' reason='{}'",
                                kind,
                                block_reason
                                    .map(|reason| reason.localized_text())
                                    .unwrap_or_else(|| "找不到塔資料".into())
                            );
                        }
                        hit_ui = true;
                    }
                }

                // 面板吸收：面板可見時，落在 panel_rect 內的點擊不穿透到地圖（防止箭頭旁邊的空白區誤重選塔）
                if !hit_ui && self.selected_tower_entity.is_some() {
                    let panel = self.ui_td_selected_panel.panel_rect;
                    if panel.w > 0.0 && panel.contains(screen) {
                        hit_ui = true;
                    }
                }

                // 5. 點選已蓋塔（只有非選塔模式時生效）
                if !hit_ui && self.selected_tower_kind.is_none() {
                    let mwp = self.mouse_world_pos;
                    let mut best: Option<(u32, f32)> = None;
                    for (id, ent) in self.network_entities.iter() {
                        if ent.entity_type != "tower" {
                            continue;
                        }
                        if ent.tower_kind.is_none() {
                            continue;
                        } // 只選 TD 塔（非 MOBA lane/base）
                        let d = (ent.position - mwp).norm();
                        let pick_radius = (ent.collision_radius_render * 1.6).max(0.6);
                        if d <= pick_radius {
                            if best.map(|(_, bd)| d < bd).unwrap_or(true) {
                                best = Some((*id, d));
                            }
                        }
                    }
                    if let Some((id, _)) = best {
                        let owned_by_local = self
                            .network_entities
                            .get(&id)
                            .map(|ent| entity_owned_by_local(ent, self.local_player_id))
                            .unwrap_or(false);
                        if owned_by_local {
                            self.selected_tower_entity = Some(id);
                            self.ui_td_selected_panel.selected_path = 0;
                            log::info!("點選中塔 id={}", id);
                        } else {
                            self.selected_tower_entity = None;
                            self.ui_td_selected_panel.selected_path = 0;
                            log::info!(
                                "點選非本地玩家塔 id={} owner={:?}; 不顯示升級/出售 UI",
                                id,
                                self.network_entities
                                    .get(&id)
                                    .and_then(|ent| ent.owner_player_id)
                            );
                        }
                    } else {
                        // 點空地 → 清掉選取
                        if self.selected_tower_entity.is_some() {
                            self.selected_tower_entity = None;
                            self.ui_td_selected_panel.selected_path = 0;
                        }
                    }
                }

                if play_button_sfx {
                    self.sfx_pending.push(SfxKind::ButtonClick);
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
                    self.ui_td_selected_panel.selected_path = 0;
                    log::info!("RMB 取消選中塔");
                } else {
                    // 階段 5.1：刪除舊版 NetCommand::HeroMove；步調一致
                    // PlayerInput::MoveTo（如下）是唯一的權威路徑。
                    let world_pos = self.mouse_world_pos;
                    let queued = self.shift_held;
                    if let Some(target_id) = self.enemy_entity_at_world(world_pos) {
                        self.send_attack_target_input_from(
                            target_id,
                            queued,
                            lockstep_client::InputOriginKind::OsEvent,
                            event_us,
                        );
                        self.attack_move_armed = false;
                    } else if self.attack_move_armed {
                        self.send_attack_move_input_from(
                            world_pos,
                            queued,
                            lockstep_client::InputOriginKind::OsEvent,
                            event_us,
                        );
                        self.attack_move_armed = false;
                    } else {
                        let target = world_render_to_vec2i(world_pos);
                        let move_to = omoba_core::kcp::game_proto::MoveTo {
                            target: Some(target),
                            queued,
                        };
                        let input = omoba_core::kcp::game_proto::PlayerInput {
                            action: Some(
                                omoba_core::kcp::game_proto::player_input::Action::MoveTo(move_to),
                            ),
                        };
                        self.send_lockstep_input_from(
                            input,
                            lockstep_client::InputOriginKind::OsEvent,
                            event_us,
                        );
                    }
                }
            }
            // LoL MVP 鍵盤輸入
            Event::WindowEvent {
                event:
                    WindowEvent::KeyboardInput {
                        event: key_event, ..
                    },
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
                if !pressed {
                    return Ok(());
                }
                // F1：開關熱鍵設定面板（pregame 設定頁與遊戲中皆可用）
                if key == KeyCode::F1 {
                    self.hotkey_panel_visible = !self.hotkey_panel_visible;
                    self.hotkey_rebinding = None;
                    return Ok(());
                }
                // 熱鍵設定面板開啟時攔截所有按鍵（重綁模式下下一鍵成為新綁定）
                // 必須在 pregame guard 之前，設定頁開面板時才能重綁
                if self.hotkey_panel_visible {
                    if let Some(rebind_id) = self.hotkey_rebinding.take() {
                        if key == KeyCode::Escape {
                            // Escape 取消重綁
                        } else if hotkeys::is_bindable_key(key) {
                            self.hotkeys.rebind(&rebind_id, key, self.ctrl_held);
                            self.hotkeys.save();
                            log::info!(
                                "熱鍵重綁: {} → {}",
                                rebind_id,
                                self.hotkeys.binding_display(&rebind_id)
                            );
                        } else {
                            // 不可綁定的鍵（F 鍵等）：維持等待狀態
                            self.hotkey_rebinding = Some(rebind_id);
                        }
                    } else if key == KeyCode::Escape {
                        self.hotkey_panel_visible = false;
                    }
                    return Ok(());
                }

                // Esc 退出全螢幕（pregame 選單中；遊戲中的 Esc 保留原本取消功能）
                if key == KeyCode::Escape
                    && !self.ctrl_held
                    && self.display_fullscreen
                    && self.pregame_runtime.is_pregame()
                {
                    let (w, h) = if self.last_windowed_size.0 > 0 {
                        self.last_windowed_size
                    } else {
                        (1920, 1080)
                    };
                    self.pending_display_mode = Some(DisplayModeRequest::Windowed(w, h));
                    log::info!("Esc → 退出全螢幕，還原 {}x{}", w, h);
                    return Ok(());
                }
                if self.pregame_runtime.is_pregame() {
                    return Ok(());
                }
                if key == KeyCode::Escape && self.ctrl_held {
                    log::info!("Ctrl+Escape exit-session action: returning to pregame menu");
                    self.shutdown_game_session(true);
                    return Ok(());
                }

                let world = self.mouse_world_pos;
                // Phase 5.1：legacy `tx` / `send` closure (NetworkBridge cmd_tx)
                // 已移除。BuyItem / SellItem 仍維持只寫 log 的 stub；ability upgrade
                // 與 item use 改走 lockstep input。
                let send_stub = |label: &str, args: &str| {
                    log::info!("[phase5.1] legacy {} send removed (args={})", label, args);
                };

                let tower_ability_slot = match key {
                    KeyCode::Digit1 => Some(0),
                    KeyCode::Digit2 => Some(1),
                    KeyCode::Digit3 => Some(2),
                    KeyCode::Digit4 => Some(3),
                    KeyCode::Digit5 => Some(4),
                    KeyCode::Digit6 => Some(5),
                    _ => None,
                };
                if let Some(slot) = tower_ability_slot {
                    let typing = text_input_has_focus(context.user_interfaces.first());
                    if !typing {
                        if let Some(item) = self.tower_ability_bar_items.get(slot).cloned() {
                            self.send_tower_ability_cast(item, event_us);
                            return Ok(());
                        }
                    }
                }

                // 按 W/E/R/T → lockstep PlayerInput::CastAbility (ability_index
                // 0/1/2/3)。滑鼠 world pos 會成為 optional `target_pos`。
                // Modifier 按住的情境（Shift = 升級，不是施法）會排除。
                // TD 模式下 W/E/R/T 讓位給塔熱鍵；只有非 TD 模式才觸發技能
                if !self.shift_held && self.td_template_order.is_empty() {
                    if let Some(ability_index) = ability_key_index(key) {
                        self.send_cast_ability_input_from(
                            ability_index,
                            Some(world),
                            lockstep_client::InputOriginKind::OsEvent,
                            event_us,
                        );
                        return Ok(());
                    }
                }

                // TD 模式：資料驅動熱鍵（F1 面板可重綁，預設對齊 BTD6）
                if !self.td_template_order.is_empty() && !self.shift_held {
                    if let Some(action_id) = self
                        .hotkeys
                        .resolve(key, self.ctrl_held)
                        .map(|s| s.to_string())
                    {
                        if self.dispatch_td_hotkey(&action_id, event_us) {
                            return Ok(());
                        }
                    }
                }

                match key {
                    // TD 模式下此 arm 讓位；非 TD 模式下 Shift+WERT 升級技能
                    KeyCode::KeyW | KeyCode::KeyE | KeyCode::KeyR | KeyCode::KeyT
                        if self.td_template_order.is_empty() =>
                    {
                        let slot = match key {
                            KeyCode::KeyW => "W",
                            KeyCode::KeyE => "E",
                            KeyCode::KeyR => "R",
                            KeyCode::KeyT => "T",
                            _ => unreachable!(),
                        }
                        .to_string();
                        if self.shift_held {
                            if let Some(ability_index) = ability_key_index(key) {
                                self.send_upgrade_ability_input_from(
                                    ability_index,
                                    lockstep_client::InputOriginKind::OsEvent,
                                    event_us,
                                );
                            }
                        } else {
                            let _ = (slot, world);
                        }
                    }
                    // TD 模式下 B/A 讓位給塔熱鍵（BTD6 對齊）
                    KeyCode::KeyB if self.td_template_order.is_empty() => {
                        self.shop_visible = !self.shop_visible;
                    }
                    KeyCode::KeyA if self.td_template_order.is_empty() => {
                        self.attack_move_armed = true;
                        log::info!("AttackMove armed");
                    }
                    // P 更改目標優先（legacy 綁定，Tab 由熱鍵設定控制）
                    KeyCode::KeyP if self.selected_tower_entity.is_some() => {
                        self.cycle_selected_tower_priority(false, event_us);
                    }
                    // TD 塔操作/選塔/沙箱熱鍵已全部改為資料驅動
                    // （dispatch_td_hotkey，F1 面板可重綁），不再硬編碼。
                    KeyCode::Escape => {
                        let mut consumed = false;
                        if self.attack_move_armed {
                            self.attack_move_armed = false;
                            consumed = true;
                        }
                        if self.selected_tower_kind.is_some() {
                            self.selected_tower_kind = None;
                            log::info!("取消選塔");
                            consumed = true;
                        }
                        // 遊戲中沒有東西可取消時，Esc 也能退出全螢幕
                        if !consumed && self.display_fullscreen {
                            let (w, h) = if self.last_windowed_size.0 > 0 {
                                self.last_windowed_size
                            } else {
                                (1920, 1080)
                            };
                            self.pending_display_mode = Some(DisplayModeRequest::Windowed(w, h));
                            log::info!("Esc → 退出全螢幕，還原 {}x{}", w, h);
                        }
                    }
                    // 數字鍵: shop 開啟時購買對應 index 裝備；否則使用對應背包 slot
                    KeyCode::Digit0
                    | KeyCode::Digit1
                    | KeyCode::Digit2
                    | KeyCode::Digit3
                    | KeyCode::Digit4
                    | KeyCode::Digit5
                    | KeyCode::Digit6
                    | KeyCode::Digit7
                    | KeyCode::Digit8
                    | KeyCode::Digit9 => {
                        let idx: usize = match key {
                            KeyCode::Digit0 => 0,
                            KeyCode::Digit1 => 1,
                            KeyCode::Digit2 => 2,
                            KeyCode::Digit3 => 3,
                            KeyCode::Digit4 => 4,
                            KeyCode::Digit5 => 5,
                            KeyCode::Digit6 => 6,
                            KeyCode::Digit7 => 7,
                            KeyCode::Digit8 => 8,
                            KeyCode::Digit9 => 9,
                            _ => unreachable!(),
                        };
                        // 階段 5.1：刪除舊的 BuyItem / SellItem（項目
                        // 尚未在鎖步線上購物）。階段 2.4：使用項目
                        // 透過鎖步 PlayerInput 連接 — Digit1..=6
                        // （商店關閉）映射到插槽 0..=5；熱欄 UI 尚未
                        // 已實施（第 4.4 階段將在
                        // SimWorldSnapshot.HeroStatsExt + 渲染），所以
                        // 鍵盤綁定是正確的唯一入口點
                        // 現在。 Shift+Digit 保留在遺留存根上，因為
                        // SellItem 尚未進入同步原型。
                        if self.shop_visible {
                            if let Some((id, _, _)) = SHOP_ITEMS.get(idx) {
                                send_stub("BuyItem", id);
                            }
                        } else if idx >= 1 && idx <= 6 {
                            if self.shift_held {
                                send_stub("SellItem", &(idx - 1).to_string());
                            } else {
                                let slot_idx = (idx - 1) as u32;
                                let input = omoba_core::kcp::game_proto::PlayerInput {
                                    action: Some(
                                        omoba_core::kcp::game_proto::player_input::Action::ItemUse(
                                            omoba_core::kcp::game_proto::ItemUse {
                                                item_slot: slot_idx,
                                                target_pos: None,
                                                target_entity: None,
                                            },
                                        ),
                                    ),
                                };
                                self.send_lockstep_input_from(
                                    input,
                                    lockstep_client::InputOriginKind::OsEvent,
                                    event_us,
                                );
                                log::info!("Item use lockstep input submitted: slot={}", slot_idx);
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
// 事件處理
// ---------------------------------------------------------------------------

impl Game {
    fn update_tower_ability_bar_ui(&mut self, ui: &mut UserInterface, now: Instant) {
        let elapsed_wall_clock = self
            .tower_ability_bar_snapshot_at
            .map(|at| now.saturating_duration_since(at).as_secs_f32())
            .unwrap_or(0.0);
        let elapsed_sim = ability_bar_elapsed_sim(
            elapsed_wall_clock,
            self.is_game_paused,
            self.game_speed_multiplier,
        );
        let item_count = self
            .latest_entities
            .iter()
            .filter(|entity| {
                entity.owner_player_id == Some(self.local_player_id)
                    && entity.hp > 0
                    && entity.tower_active_ability.is_some()
            })
            .count();
        let page_model =
            ability_bar_page_model(item_count, self.tower_ability_bar_interaction.page);
        self.tower_ability_bar_interaction.page = page_model.page;
        self.tower_ability_bar_items = ability_bar_items_with_names(
            &self.latest_entities,
            &self.tower_ability_bar_tower_names,
            self.local_player_id,
            page_model.page,
            elapsed_sim,
        );
        self.tower_ability_bar_rejection.retain_visible(
            &ability_bar_available_keys(&self.latest_entities, self.local_player_id),
            now,
        );
        let visible_keys = self
            .tower_ability_bar_items
            .iter()
            .map(|item| item.key.clone())
            .collect::<Vec<_>>();
        reconcile_ability_bar_slots(&mut self.tower_ability_bar_slot_bindings, &visible_keys);

        let slot_w = 142.0;
        let slot_h = 88.0;
        let gap = 8.0;
        let count = self.tower_ability_bar_items.len();
        let total_w = count as f32 * slot_w + count.saturating_sub(1) as f32 * gap;
        let start_x = (self.window_size.x - total_w) * 0.5;
        let y = tower_ability_bar_y(self.window_size.y, slot_h);

        for index in 0..TOWER_ABILITY_BAR_PAGE_SIZE {
            let Some(&handle) = self.ui_tower_ability_bar_slots.get(index) else {
                break;
            };
            let logical_item = self.tower_ability_bar_slot_bindings[index]
                .as_ref()
                .and_then(|key| {
                    self.tower_ability_bar_items
                        .iter()
                        .enumerate()
                        .find(|(_, item)| &item.key == key)
                });
            let logical_index = logical_item.map(|(logical_index, _)| logical_index);
            let item = logical_item.map(|(_, item)| item.clone());
            let (rect, text) = if let Some(item) = item.as_ref() {
                (
                    UiRect {
                        x: start_x + logical_index.unwrap_or(0) as f32 * (slot_w + gap),
                        y,
                        w: slot_w,
                        h: slot_h,
                    },
                    ability_bar_button_text(&item.ability_name),
                )
            } else {
                (
                    UiRect {
                        x: UI_HIDDEN_POS,
                        y: UI_HIDDEN_POS,
                        w: slot_w,
                        h: slot_h,
                    },
                    String::new(),
                )
            };
            if self.tower_ability_bar_rects.get(index).copied() != Some(rect) {
                ui.send(handle, WidgetMessage::DesiredPosition(rect.pos()));
                ui.send(handle, WidgetMessage::Width(rect.w));
                ui.send(handle, WidgetMessage::Height(rect.h));
                self.tower_ability_bar_rects[index] = rect;
                for node in [
                    self.ui_tower_ability_bar_backgrounds[index],
                    self.ui_tower_ability_bar_cooldown_overlays[index],
                    self.ui_tower_ability_bar_active_overlays[index],
                ] {
                    ui.send(node, WidgetMessage::DesiredPosition(rect.pos()));
                    ui.send(node, WidgetMessage::Width(rect.w));
                }
            }
            if self.tower_ability_bar_cached_text.get(index) != Some(&text) {
                ui.send(handle, TextMessage::Text(text.clone()));
                self.tower_ability_bar_cached_text[index] = text;
            }
            let rendered_state = item.as_ref().map(|item| AbilityBarRenderedState {
                key: item.key.clone(),
                visual: item.visual_state.clone(),
                enabled: matches!(item.visual_state, AbilityBarVisualState::Ready)
                    && !self
                        .tower_ability_bar_interaction
                        .debounced
                        .contains(&item.key),
            });
            if self.tower_ability_bar_cached_visual[index] != rendered_state {
                let (color, cooldown_fraction, active) = match rendered_state.as_ref() {
                    Some(AbilityBarRenderedState {
                        visual: AbilityBarVisualState::Ready,
                        enabled: true,
                        ..
                    }) => (Color::from_rgba(30, 90, 48, 245), 0.0, false),
                    Some(AbilityBarRenderedState {
                        visual: AbilityBarVisualState::Cooling { fraction, .. },
                        ..
                    }) => (Color::from_rgba(45, 48, 55, 245), *fraction, false),
                    Some(AbilityBarRenderedState {
                        visual:
                            AbilityBarVisualState::Active {
                                cooldown_fraction, ..
                            },
                        ..
                    }) => (Color::from_rgba(28, 65, 125, 245), *cooldown_fraction, true),
                    Some(_) => (Color::from_rgba(65, 58, 35, 245), 0.0, false),
                    None => (Color::TRANSPARENT, 0.0, false),
                };
                ui.send(
                    self.ui_tower_ability_bar_backgrounds[index],
                    WidgetMessage::Background(Brush::Solid(color).into()),
                );
                ui.send(
                    self.ui_tower_ability_bar_backgrounds[index],
                    WidgetMessage::Height(rect.h),
                );
                let cooldown_h = rect.h * cooldown_fraction.clamp(0.0, 1.0);
                ui.send(
                    self.ui_tower_ability_bar_cooldown_overlays[index],
                    WidgetMessage::DesiredPosition(Vector2::new(
                        rect.x,
                        rect.y + rect.h - cooldown_h,
                    )),
                );
                ui.send(
                    self.ui_tower_ability_bar_cooldown_overlays[index],
                    WidgetMessage::Height(cooldown_h),
                );
                ui.send(
                    self.ui_tower_ability_bar_active_overlays[index],
                    WidgetMessage::Height(if active { rect.h } else { 0.0 }),
                );
                self.tower_ability_bar_cached_visual[index] = rendered_state;
            }
        }

        let hidden = Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS);
        if item_count > 0 {
            let control_y = (y - 40.0).max(0.0);
            let center = self.window_size.x * 0.5;
            self.tower_ability_bar_previous_rect = UiRect {
                x: center - 76.0,
                y: control_y,
                w: 36.0,
                h: 34.0,
            };
            self.tower_ability_bar_next_rect = UiRect {
                x: center + 40.0,
                y: control_y,
                w: 36.0,
                h: 34.0,
            };
            ui.send(
                self.ui_tower_ability_bar_previous,
                WidgetMessage::DesiredPosition(self.tower_ability_bar_previous_rect.pos()),
            );
            ui.send(
                self.ui_tower_ability_bar_next,
                WidgetMessage::DesiredPosition(self.tower_ability_bar_next_rect.pos()),
            );
            ui.send(
                self.ui_tower_ability_bar_page_indicator,
                WidgetMessage::DesiredPosition(Vector2::new(center - 36.0, control_y)),
            );
            ui.send(
                self.ui_tower_ability_bar_page_indicator,
                WidgetMessage::Width(72.0),
            );
            ui.send(
                self.ui_tower_ability_bar_page_indicator,
                TextMessage::Text(page_model.indicator.clone()),
            );
            for (handle, enabled) in [
                (
                    self.ui_tower_ability_bar_previous,
                    page_model.previous_enabled,
                ),
                (self.ui_tower_ability_bar_next, page_model.next_enabled),
            ] {
                ui.send(
                    handle,
                    WidgetMessage::Foreground(
                        Brush::Solid(if enabled {
                            Color::from_rgba(245, 245, 245, 255)
                        } else {
                            Color::from_rgba(100, 100, 105, 255)
                        })
                        .into(),
                    ),
                );
            }
        } else {
            self.tower_ability_bar_previous_rect = UiRect::default();
            self.tower_ability_bar_next_rect = UiRect::default();
            for handle in [
                self.ui_tower_ability_bar_previous,
                self.ui_tower_ability_bar_next,
                self.ui_tower_ability_bar_page_indicator,
            ] {
                ui.send(handle, WidgetMessage::DesiredPosition(hidden));
            }
        }

        let hovered = self
            .tower_ability_bar_rects
            .iter()
            .enumerate()
            .find(|(_, rect)| rect.x > -9000.0 && rect.contains(self.mouse_screen_pos))
            .and_then(|(slot, _)| self.tower_ability_bar_slot_bindings[slot].as_ref())
            .and_then(|key| {
                self.tower_ability_bar_items
                    .iter()
                    .find(|item| &item.key == key)
            });
        let tooltip =
            ability_bar_tooltip_model(hovered, self.tower_ability_bar_rejection.visible.as_ref());
        if self.tower_ability_bar_cached_tooltip != tooltip {
            let pos = if tooltip.is_some() {
                Vector2::new(
                    (self.mouse_screen_pos.x + 16.0).min((self.window_size.x - 370.0).max(0.0)),
                    (y - 206.0).max(0.0),
                )
            } else {
                hidden
            };
            ui.send(
                self.ui_tower_ability_tooltip_bg,
                WidgetMessage::DesiredPosition(pos),
            );
            ui.send(
                self.ui_tower_ability_tooltip_title,
                WidgetMessage::DesiredPosition(pos + Vector2::new(12.0, 8.0)),
            );
            ui.send(
                self.ui_tower_ability_tooltip_description,
                WidgetMessage::DesiredPosition(pos + Vector2::new(12.0, 36.0)),
            );
            ui.send(
                self.ui_tower_ability_tooltip_title,
                TextMessage::Text(
                    tooltip
                        .as_ref()
                        .map(|m| m.title.clone())
                        .unwrap_or_default(),
                ),
            );
            ui.send(
                self.ui_tower_ability_tooltip_description,
                TextMessage::Text(
                    tooltip
                        .as_ref()
                        .map(|m| m.description.clone())
                        .unwrap_or_default(),
                ),
            );
            self.tower_ability_bar_cached_tooltip = tooltip;
        }
    }

    fn send_tower_ability_cast(&mut self, item: AbilityBarItem, event_us: u64) -> bool {
        if !matches!(item.visual_state, AbilityBarVisualState::Ready)
            || !self
                .tower_ability_bar_interaction
                .debounced
                .insert(item.key.clone())
        {
            return false;
        }
        let input = omoba_core::kcp::game_proto::PlayerInput {
            action: Some(
                omoba_core::kcp::game_proto::player_input::Action::TowerAbilityCast(
                    omoba_core::kcp::game_proto::TowerAbilityCastInput {
                        tower_entity_id: item.key.tower_entity_id,
                        ability_id: item.key.ability_id,
                    },
                ),
            ),
        };
        self.send_lockstep_input_from(input, lockstep_client::InputOriginKind::OsEvent, event_us);
        true
    }

    fn server_timing(&self) -> LockstepTiming {
        LockstepTiming::new(if self.server_step_fps == 0 {
            LOCKSTEP_TPS
        } else {
            self.server_step_fps
        })
        .unwrap_or(LockstepTiming::DEFAULT)
    }

    fn ticks_to_seconds(&self, tick: u32) -> f64 {
        self.server_timing().ticks_to_seconds_f64(tick)
    }

    fn update_sim_speed(&mut self, tick: u32) {
        if tick == self.sim_speed_last_tick {
            return;
        }

        let now = Instant::now();
        if let Some(prev_at) = self.sim_speed_last_at {
            let elapsed = now.duration_since(prev_at).as_secs_f32();
            let delta_ticks = tick.saturating_sub(self.sim_speed_last_tick);
            if elapsed > 0.0 && delta_ticks > 0 {
                self.sim_speed_tps = delta_ticks as f32 / elapsed;
                fyrox::engine::executor::set_render_target_tps(RENDER_UPDATE_TPS);
            }
        }
        self.sim_speed_last_tick = tick;
        self.sim_speed_last_at = Some(now);
    }

    fn clear_lua_metadata_caches(&mut self) {
        self.td_templates.clear();
        self.td_template_order.clear();
        self.tower_ability_bar_tower_names.clear();
        self.td_upgrade_defs.clear();
        self.ability_info_map.clear();
        self.ability_icon_texture_cache.clear();
        self.ability_textures = [None, None, None, None];
        self.ability_icon_paths = std::array::from_fn(|_| String::new());
        self.tower_texture_cache.clear();
        self.tower_material_cache.clear();
        self.hero_model_assets.clear();
        self.hero_action_assets.clear();
        self.hero_asset_failures_logged.clear();
    }

    fn invalidate_lua_content_caches(&mut self, scene: &mut Scene, generation: u64, hash: &str) {
        self.clear_lua_metadata_caches();
        let tower_ids: Vec<u32> = self.tower_composites.keys().copied().collect();
        for id in tower_ids {
            self.remove_tower_composite(scene, id);
        }
        let hero_ids: Vec<u32> = self.hero_model_nodes.keys().copied().collect();
        for id in hero_ids {
            self.remove_hero_model(scene, id);
        }
        log::info!(
            "frontend Lua content caches invalidated for generation={} hash={}",
            generation,
            hash
        );
    }

    fn td_ui_texture(&mut self, asset_name: &str) -> Option<TextureResource> {
        if let Some(cached) = self.td_ui_texture_cache.get(asset_name) {
            return cached.clone();
        }
        let texture = load_td_ui_texture(asset_name);
        if texture.is_none() {
            log::warn!(
                "TD UI texture '{}' not found in scripts/base_content/assets/td_ui or fallback paths",
                asset_name
            );
        }
        self.td_ui_texture_cache
            .insert(asset_name.to_string(), texture.clone());
        texture
    }

    fn pregame_ui_texture(&mut self, asset_name: &str) -> Option<TextureResource> {
        if let Some(cached) = self.pregame_ui_texture_cache.get(asset_name) {
            return cached.clone();
        }
        let texture = load_pregame_ui_texture(asset_name);
        if texture.is_none() {
            log::warn!(
                "pregame UI texture '{}' not found in scripts/base_content/assets/pregame_ui or fallback paths",
                asset_name
            );
        }
        self.pregame_ui_texture_cache
            .insert(asset_name.to_string(), texture.clone());
        texture
    }

    fn set_td_shop_scroll_offset(&mut self, offset: f32) {
        self.td_shop_scroll_offset = offset.clamp(0.0, self.td_shop_max_scroll.max(0.0));
    }

    fn tower_placement_block_reason(
        &self,
        tpl: &TdTemplate,
        pos: Vector2<f32>,
    ) -> Option<TowerPlacementBlockReason> {
        if self.hero_state.gold < tpl.cost {
            return Some(TowerPlacementBlockReason::InsufficientGold {
                required: tpl.cost,
                available: self.hero_state.gold,
            });
        }

        let placement_radius_render = tower_placement_radius_render(tpl);
        let clear_render =
            (tpl.placement_radius_backend + TD_PATH_HALF_WIDTH_BACKEND) * WORLD_SCALE;
        let clear_sq = clear_render * clear_render;

        for path in &self.td_paths_render {
            for i in 0..path.len().saturating_sub(1) {
                if point_segment_dist_sq(pos, path[i], path[i + 1]) < clear_sq {
                    return Some(TowerPlacementBlockReason::TooCloseToRoad);
                }
            }
        }

        for poly in &self.td_regions_render {
            if circle_hits_polygon(pos, placement_radius_render, poly) {
                return Some(TowerPlacementBlockReason::BlockedRegion);
            }
        }

        for ent in self.network_entities.values() {
            if ent.entity_type != "tower" {
                continue;
            }
            let Some(existing_radius) = ent
                .tower_kind
                .as_ref()
                .and_then(|kind| self.td_templates.get(kind))
                .map(tower_placement_radius_render)
            else {
                return Some(TowerPlacementBlockReason::MissingPlacementMetadata);
            };
            let min_d = existing_radius + placement_radius_render;
            if (ent.position - pos).norm_squared() < min_d * min_d {
                return Some(TowerPlacementBlockReason::TowerOverlap);
            }
        }

        None
    }

    fn nearest_tower_road_probe(
        &self,
        tpl: &TdTemplate,
        pos: Vector2<f32>,
    ) -> Option<TowerRoadPlacementProbe> {
        let required_clearance_backend =
            tpl.placement_radius_backend + TD_PATH_HALF_WIDTH_BACKEND;
        let mut nearest: Option<TowerRoadPlacementProbe> = None;
        for (path_index, path) in self.td_paths_render.iter().enumerate() {
            for segment_index in 0..path.len().saturating_sub(1) {
                let distance_backend =
                    point_segment_dist_sq(pos, path[segment_index], path[segment_index + 1])
                        .sqrt()
                        / WORLD_SCALE;
                let candidate = TowerRoadPlacementProbe {
                    path_index,
                    segment_index,
                    distance_backend,
                    required_clearance_backend,
                };
                if nearest
                    .map(|current| distance_backend < current.distance_backend)
                    .unwrap_or(true)
                {
                    nearest = Some(candidate);
                }
            }
        }
        nearest
    }

    fn can_place_tower_at(&self, tpl: &TdTemplate, pos: Vector2<f32>) -> bool {
        self.tower_placement_block_reason(tpl, pos).is_none()
    }

    fn selected_tower_screen_x(&self) -> Option<f32> {
        let tid = self.selected_tower_entity?;
        let ent = self.network_entities.get(&tid)?;
        let world_height = if self.is_td_mode { 28.0 } else { 20.0 };
        Some(
            world_to_screen_approx(
                ent.position.x - self.camera_world_pos.x,
                ent.position.y - self.camera_world_pos.y,
                self.window_size.x,
                self.window_size.y,
                world_height,
            )
            .x,
        )
    }

    fn ability_icon_texture(&mut self, rel_path: &str) -> Option<TextureResource> {
        let key = if rel_path.trim().is_empty() {
            ABILITY_ICON_FALLBACK_PATH
        } else {
            rel_path.trim()
        };
        if let Some(cached) = self.ability_icon_texture_cache.get(key) {
            return cached.clone();
        }
        let texture = load_texture_from_rel_path(key);
        self.ability_icon_texture_cache
            .insert(key.to_string(), texture.clone());
        texture
    }

    fn tower_texture_for_key(&mut self, key: &str) -> Option<TextureResource> {
        let key = normalize_tower_asset_key(key);
        if key.is_empty() {
            return None;
        }
        if let Some(cached) = self.tower_texture_cache.get(&key) {
            return cached.clone();
        }
        let texture = load_tower_texture(&key);
        if texture.is_none() {
            log::warn!(
                "tower combat texture '{}' not found; using colored fallback",
                key
            );
        }
        self.tower_texture_cache.insert(key, texture.clone());
        texture
    }

    fn tower_material_for_key(&mut self, key: &str) -> Option<MaterialResource> {
        let key = normalize_tower_asset_key(key);
        if key.is_empty() {
            return None;
        }
        if let Some(cached) = self.tower_material_cache.get(&key) {
            return cached.clone();
        }
        let material = self.tower_texture_for_key(&key).map(texture_material);
        self.tower_material_cache.insert(key, material.clone());
        material
    }

    fn default_session_selection(&self) -> Option<pregame::SessionSelection> {
        let map = self
            .pregame_runtime
            .catalog
            .enabled_maps()
            .into_iter()
            .next()?
            .clone();
        let difficulty = self
            .pregame_runtime
            .catalog
            .enabled_difficulties()
            .into_iter()
            .next()?
            .clone();
        Some(pregame::SessionSelection { map, difficulty })
    }

    fn build_backend_launch_config(
        &self,
        selection: &pregame::SessionSelection,
        session_id: String,
    ) -> backend_session::BackendLaunchConfig {
        let server_addr = frontend_server_addr();
        let story = selection.map.story_id().to_string();
        let difficulty_config = selection.difficulty.config_value().to_string();
        let content_root = std::env::var("OMFX_CONTENT_ROOT")
            .or_else(|_| std::env::var("OMB_CONTENT_ROOT"))
            .or_else(|_| std::env::var("OMB_LUA_CONTENT_ROOT"))
            .ok()
            .map(PathBuf::from)
            .or_else(|| frontend_config_path("content", "LUA_CONTENT_ROOT"))
            .or_else(|| Some(PathBuf::from(DEFAULT_STORY_DATA_DIR)))
            .map(absolute_existing_or_joined_path);
        backend_session::BackendLaunchConfig {
            session_id,
            map_id: selection.map.id.clone(),
            story,
            difficulty_id: selection.difficulty.id.clone(),
            difficulty_config,
            kcp_addr: server_addr,
            content_root,
            executable: backend_session::configured_backend_executable(),
            launcher_enabled: backend_session::launcher_enabled_from_env(),
        }
    }

    fn start_game_session(&mut self, selection: pregame::SessionSelection) -> Result<(), String> {
        if self.lockstep_handle.is_some()
            || self.sim_runner_handle.is_some()
            || self.backend_session.is_some()
        {
            return Err("game session is already active".to_string());
        }

        apply_frontend_runtime_env_from_config();
        let session_id = format!(
            "omfx-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_millis())
                .unwrap_or(0)
        );
        let backend_config = self.build_backend_launch_config(&selection, session_id.clone());
        let server_addr = backend_config.kcp_addr.clone();
        let story = backend_config.story.clone();
        let difficulty_config = backend_config.difficulty_config.clone();
        let backend = backend_session::BackendSession::start(backend_config)?;
        self.backend_session = Some(backend);

        std::env::set_var("OMB_SESSION_ID", &session_id);
        std::env::set_var("OMB_MAP_ID", &selection.map.id);
        std::env::set_var("OMB_STORY", &story);
        std::env::set_var("OMB_DIFFICULTY_ID", &selection.difficulty.id);
        std::env::set_var("OMB_DIFFICULTY", &difficulty_config);
        std::env::set_var("OMB_KCP_ADDR", &server_addr);

        let player_name =
            frontend_config_env_or_section_value("OMB_PLAYER_NAME", "client", "PLAYER_NAME")
                .unwrap_or_else(|| "omfx_player".to_string());
        let lockstep_player_name = std::env::var("OMB_LOCKSTEP_PLAYER_NAME").unwrap_or_else(|_| {
            let suffix = frontend_config_value("LOCKSTEP_PLAYER_SUFFIX")
                .unwrap_or_else(|| "_lockstep".to_string());
            format!("{}{}", player_name, suffix)
        });
        let local_player_id =
            frontend_config_u32_or_default("OMB_PLAYER_ID", "client", "PLAYER_ID", 1);
        self.local_player_id = local_player_id;
        log::info!(
            "frontend session start: session_id={} map_id={} story={} difficulty={} player_name='{}' lockstep_player_name='{}' player_id={}",
            session_id,
            selection.map.id,
            story,
            selection.difficulty.id,
            player_name,
            lockstep_player_name,
            local_player_id
        );
        self.connection_status = ConnectionStatus::Connecting;
        self.lockstep_handle = Some(lockstep_client::spawn_lockstep_client(
            server_addr,
            lockstep_player_name,
            local_player_id,
        ));

        let dll_path: PathBuf = std::env::var("OMB_DLL_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                frontend_config_path("content", "DLL_PATH")
                    .or_else(|| frontend_config_path("client", "DLL_PATH"))
                    .unwrap_or_else(|| PathBuf::from(DEFAULT_DLL_PATH))
            });
        let scene_path: PathBuf = std::env::var("OMB_SCENE_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                let data_root = std::env::var("OMB_STORY_DATA_DIR")
                    .map(PathBuf::from)
                    .unwrap_or_else(|_| {
                        frontend_config_path("content", "STORY_DATA_DIR")
                            .or_else(|| frontend_config_path("client", "STORY_DATA_DIR"))
                            .unwrap_or_else(|| PathBuf::from(DEFAULT_STORY_DATA_DIR))
                    });
                data_root.join(&story)
            });
        let extract_data_for_render_every_ticks = frontend_config_u32_or_default(
            "OMFX_EXTRACT_DATA_FOR_RENDER_EVERY_TICKS",
            "client",
            "EXTRACT_DATA_FOR_RENDER_EVERY_TICKS",
            sim_runner::DEFAULT_EXTRACT_DATA_FOR_RENDER_EVERY_TICKS,
        );
        log::info!(
            "sim_runner spawn: session_id={} map_id={} story={} difficulty={} dll={:?} scene={:?} extract_data_for_render_every_ticks={}",
            session_id,
            selection.map.id,
            story,
            selection.difficulty.id,
            dll_path,
            scene_path,
            extract_data_for_render_every_ticks
        );
        self.sim_runner_handle = Some(sim_runner::spawn_sim_runner_with_render_extract_rate(
            dll_path,
            scene_path,
            extract_data_for_render_every_ticks,
        ));
        self.pregame_runtime.mark_in_game();
        self.connection_status = ConnectionStatus::Connected;
        Ok(())
    }

    fn shutdown_game_session(&mut self, return_to_menu: bool) {
        if let Some(lockstep) = self.lockstep_handle.take() {
            lockstep.shutdown();
        }
        if let Some(sim_runner) = self.sim_runner_handle.take() {
            sim_runner.shutdown();
        }
        if let Some(mut backend) = self.backend_session.take() {
            backend.shutdown();
        }
        self.reset_session_state();
        if return_to_menu {
            self.pregame_runtime.return_to_menu();
        }
    }

    fn reset_session_state(&mut self) {
        self.connection_status = ConnectionStatus::Disconnected;
        self.current_sim_tick = 0;
        self.current_sim_tick_observed_at = None;
        self.server_step_fps = 0;
        self.pending_inputs.clear();
        self.pending_inputs_evict_at = None;
        self.pending_inputs_evicted = 0;
        self.pending_inputs_stale = 0;
        self.input_latency_meter = InputLatencyMeter::default();
        self.net_bytes_current = 0;
        self.net_bytes_last_sec = 0;
        self.net_wire_bytes_current = 0;
        self.net_wire_bytes_last_sec = 0;
        self.latest_rtt_us = None;
        self.sim_lua_content_generation = 0;
        self.sim_lua_content_hash.clear();
        self.sim_dev_lua_reload_error = None;
        self.sim_speed_last_tick = 0;
        self.sim_speed_last_at = None;
        self.sim_speed_tps = 0.0;
        self.render_pacing_last_frame_at = None;
        self.render_pacing_last_snapshot_tick = None;
        self.sim_batches_last_snapshot_tick = None;
        self.network_entities.clear();
        self.latest_entities.clear();
        self.selected_tower_kind = None;
        self.selected_tower_entity = None;
        self.td_shop_scroll_offset = 0.0;
        self.td_shop_max_scroll = 0.0;
        self.td_shop_scroll_dragging = false;
        self.current_round = 0;
        self.total_rounds = 0;
        self.round_is_running = false;
        self.is_game_paused = false;
        self.game_speed_multiplier = 1;
        self.td_auto_start_sent_for_idle_round = false;
        self.is_td_mode = false;
        self.td_camera_configured = false;
        self.active_explosions.clear();
        self.sim_last_explosion_tick = None;
        self.td_paths_render.clear();
        self.td_regions_render.clear();
        self.tower_anim_dt_acc = 0.0;
        self.sim_seen_tower_fire_fx.clear();
        self.sim_seen_attack_phase_fx.clear();
        self.sim_seen_attack_cancel_fx.clear();
        self.pending_pred_dmg.clear();
        self.heartbeat = HeartbeatInfo::default();
        self.projectile_spawn_pos.clear();
        self.projectile_trail_dir.clear();
        self.pending_label_deletions.extend(
            self.sim_entity_labels
                .drain()
                .map(|(_, label)| label.handle),
        );
        self.tower_ability_bar_interaction = AbilityBarInteractionState::default();
        self.tower_ability_bar_snapshot_at = None;
        self.tower_ability_bar_items.clear();
        self.tower_ability_bar_rejection = AbilityBarRejectionState::default();
        self.tower_ability_bar_cached_tooltip = None;
        self.tower_ability_bar_slot_bindings.fill(None);
        self.tower_ability_bar_cached_visual.fill(None);
        self.tower_ability_bar_cached_text.fill(String::new());
        self.clear_lua_metadata_caches();
        self.auto_clock_start = None;
        self.auto_start_sent = false;
        self.auto_noop_next_at_s = None;
        self.game_ended = false;
        self.session_render_reset_pending = true;
    }

    fn reset_session_render_state(&mut self, scene: &mut Scene) {
        self.render_bridge.reset(scene);

        let tower_ids: Vec<u32> = self.tower_composites.keys().copied().collect();
        for entity_id in tower_ids {
            self.remove_tower_composite(scene, entity_id);
        }
        let hero_ids: Vec<u32> = self.hero_model_nodes.keys().copied().collect();
        for entity_id in hero_ids {
            self.remove_hero_model(scene, entity_id);
        }
        for (_, projectile) in self.client_projectiles.drain() {
            if scene.graph.is_valid_handle(projectile.node) {
                scene.graph.remove_node(projectile.node);
            }
            for (ring_node, _) in projectile.hit_ring {
                if scene.graph.is_valid_handle(ring_node) {
                    scene.graph.remove_node(ring_node);
                }
            }
        }

        let slots: Vec<render_bridge::SimEntitySlots> = self
            .sim_entity_slots
            .drain()
            .map(|(_, slots)| slots)
            .collect();
        for slots in slots {
            if let Some(batch) = self.body_batch.as_mut() {
                batch.free(slots.body_slot);
            }
            if let Some(batch) = self.hp_batch.as_mut() {
                if let Some(slot) = slots.hp_bg_slot {
                    batch.free(slot);
                }
                if let Some(slot) = slots.hp_fg_slot {
                    batch.free(slot);
                }
            }
            if let Some(batch) = self.facing_batch.as_mut() {
                if let Some(slot) = slots.turret_slot {
                    batch.free(slot);
                }
            }
        }
        if let Some(batch) = self.body_batch.as_mut() {
            batch.flush(scene);
        }
        if let Some(batch) = self.hp_batch.as_mut() {
            batch.flush(scene);
        }
        if let Some(batch) = self.facing_batch.as_mut() {
            batch.flush(scene);
        }
        self.entity_kind_cache.clear();
    }

    fn current_pregame_buttons(&self) -> Vec<(String, String, bool, pregame::PregameAction)> {
        match self.pregame_runtime.state {
            pregame::PregameState::MainMenu => self
                .pregame_runtime
                .catalog
                .screen("main_menu")
                .map(|screen| {
                    screen
                        .widgets
                        .iter()
                        .map(|widget| {
                            (
                                widget.label.clone(),
                                widget.description.clone(),
                                widget.is_active(),
                                widget.action.clone(),
                            )
                        })
                        .collect()
                })
                .unwrap_or_default(),
            pregame::PregameState::MapSelect => {
                let mut buttons = vec![(
                    "返回".to_string(),
                    String::new(),
                    true,
                    pregame::PregameAction::Back,
                )];
                buttons.extend(self.current_selectable_maps().into_iter().map(|map| {
                    (
                        map.label.clone(),
                        if map.reward.trim().is_empty() {
                            map.description.clone()
                        } else {
                            format!("{} | {}", map.description, map.reward)
                        },
                        map.is_playable(),
                        pregame::PregameAction::SelectMap {
                            map_id: map.id.clone(),
                        },
                    )
                }));
                buttons.extend(self.pregame_runtime.catalog.difficulties.iter().map(
                    |difficulty| {
                        (
                            difficulty.label.clone(),
                            difficulty.description.clone(),
                            difficulty.enabled,
                            pregame::PregameAction::SelectDifficulty {
                                difficulty_id: difficulty.id.clone(),
                            },
                        )
                    },
                ));
                buttons
            }
            pregame::PregameState::DifficultySelect => {
                let mut buttons = vec![(
                    "返回".to_string(),
                    String::new(),
                    true,
                    pregame::PregameAction::Back,
                )];
                buttons.extend(self.pregame_runtime.catalog.difficulties.iter().map(
                    |difficulty| {
                        (
                            difficulty.label.clone(),
                            if difficulty.reward.trim().is_empty() {
                                difficulty.description.clone()
                            } else {
                                format!("{} | {}", difficulty.description, difficulty.reward)
                            },
                            difficulty.enabled,
                            pregame::PregameAction::SelectDifficulty {
                                difficulty_id: difficulty.id.clone(),
                            },
                        )
                    },
                ));
                buttons
            }
            pregame::PregameState::StartingSession => vec![(
                "啟動中...".to_string(),
                "請稍候".to_string(),
                false,
                pregame::PregameAction::NoOp,
            )],
            pregame::PregameState::SessionEnded => vec![(
                "返回選單".to_string(),
                String::new(),
                true,
                pregame::PregameAction::Back,
            )],
            pregame::PregameState::InGame => Vec::new(),
            pregame::PregameState::Settings => Vec::new(),
            pregame::PregameState::Profile => Vec::new(),
        }
    }

    fn current_selectable_maps(&self) -> Vec<pregame::MapEntry> {
        let Some(difficulty) = self.pregame_runtime.selected_difficulty.as_ref() else {
            return self
                .pregame_runtime
                .catalog
                .enabled_maps()
                .into_iter()
                .cloned()
                .collect();
        };
        self.pregame_runtime
            .catalog
            .maps_for_difficulty(&difficulty.id)
            .into_iter()
            .cloned()
            .collect()
    }

    fn handle_pregame_click(&mut self, screen: Vector2<f32>) -> bool {
        if !self.pregame_runtime.is_pregame() {
            return false;
        }
        // 熱鍵面板開啟時為 modal，吃掉所有點擊
        if self.hotkey_panel_visible {
            self.handle_hotkey_panel_click(screen);
            return true;
        }
        // 設定頁「熱鍵」按鈕 → 開啟熱鍵設定面板
        if self.settings_hotkey_btn_rect.w > 0.0 && self.settings_hotkey_btn_rect.contains(screen) {
            self.hotkey_panel_visible = true;
            self.hotkey_rebinding = None;
            return true;
        }
        // 解析度下拉選單開啟中：選項 / 好的 / 外部點擊
        if self.resolution_dropdown_open {
            let mut hit_option: Option<usize> = None;
            for (i, rect) in self.resolution_dropdown_ui.row_rects.iter().enumerate() {
                if rect.contains(screen) {
                    hit_option = Some(i);
                    break;
                }
            }
            if let Some(i) = hit_option {
                let mode = match RESOLUTION_OPTIONS.get(i).copied().flatten() {
                    None => DisplayModeRequest::Fullscreen,
                    Some((w, h)) => DisplayModeRequest::Windowed(w, h),
                };
                self.pending_display_mode = Some(mode);
                log::info!("解析度選項點選: {:?}", mode);
            } else if self.resolution_dropdown_ui.ok_rect.contains(screen) {
                self.resolution_dropdown_open = false;
            } else if !self.settings_resolution_rect.contains(screen) {
                // 點擊選單外側 → 關閉
                self.resolution_dropdown_open = false;
            }
            return true;
        }
        // 設定頁「螢幕尺寸」badge → 開啟下拉選單
        if self.settings_resolution_rect.w > 0.0 && self.settings_resolution_rect.contains(screen) {
            self.resolution_dropdown_open = true;
            return true;
        }
        let action = self
            .pregame_button_rects
            .iter()
            .find_map(|(rect, action)| rect.contains(screen).then(|| action.clone()));
        let Some(action) = action else {
            return true;
        };
        let was_settings = self.pregame_runtime.state == pregame::PregameState::Settings;
        if let Some(selection) = self.pregame_runtime.dispatch(&action) {
            if let Err(err) = self.start_game_session(selection) {
                log::error!("session start failed: {}", err);
                self.shutdown_game_session(false);
                self.pregame_runtime.recover_to_difficulty(err.clone());
                self.connection_status = ConnectionStatus::Failed(err);
            }
        }
        if was_settings && self.pregame_runtime.state != pregame::PregameState::Settings {
            audio_settings::AudioSettings {
                music_volume: self.settings_music_volume,
            }
            .save();
        }
        true
    }

    fn handle_in_game_return_click(&mut self, screen: Vector2<f32>) -> bool {
        if self.pregame_runtime.state != pregame::PregameState::InGame {
            return false;
        }
        if !self.return_to_title_button_rect.contains(screen) {
            return false;
        }
        log::info!("in-game return button clicked: returning to title menu");
        self.shutdown_game_session(true);
        true
    }

    fn place_pregame_node(
        &mut self,
        ui: &mut UserInterface,
        index: &mut usize,
        rect: UiRect,
        text: String,
        active: bool,
        action: pregame::PregameAction,
        role: PregameVisualRole,
        bg_color: Color,
        fg_color: Color,
    ) {
        if *index >= self.ui_pregame.buttons.len() {
            return;
        }
        let node = &mut self.ui_pregame.buttons[*index];
        node.role = role;
        ui.send(node.bg, WidgetMessage::DesiredPosition(rect.pos()));
        ui.send(node.bg, WidgetMessage::Width(rect.w));
        ui.send(node.bg, WidgetMessage::Height(rect.h));
        ui.send(
            node.bg,
            WidgetMessage::Background(Brush::Solid(bg_color).into()),
        );
        ui.send(
            node.image,
            WidgetMessage::DesiredPosition(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS)),
        );
        ui.send(node.image, ImageMessage::Texture(None));
        ui.send(
            node.text,
            WidgetMessage::DesiredPosition(Vector2::new(rect.x + 8.0, rect.y + 4.0)),
        );
        ui.send(node.text, WidgetMessage::Width((rect.w - 16.0).max(1.0)));
        ui.send(node.text, WidgetMessage::Height((rect.h - 8.0).max(1.0)));
        ui.send(
            node.text,
            WidgetMessage::Foreground(Brush::Solid(fg_color).into()),
        );
        ui.send(node.text, TextMessage::Text(text));
        if active {
            self.pregame_button_rects.push((rect, action));
        }
        *index += 1;
    }

    fn place_pregame_map_node(
        &mut self,
        ui: &mut UserInterface,
        index: &mut usize,
        rect: UiRect,
        text: String,
        active: bool,
        action: pregame::PregameAction,
        bg_color: Color,
        fg_color: Color,
        image: Option<&str>,
    ) {
        if *index >= self.ui_pregame.buttons.len() {
            return;
        }
        let texture = image.and_then(|image| self.pregame_ui_texture(image));
        let node = &mut self.ui_pregame.buttons[*index];
        node.role = PregameVisualRole::Button;
        ui.send(node.bg, WidgetMessage::DesiredPosition(rect.pos()));
        ui.send(node.bg, WidgetMessage::Width(rect.w));
        ui.send(node.bg, WidgetMessage::Height(rect.h));
        ui.send(
            node.bg,
            WidgetMessage::Background(Brush::Solid(bg_color).into()),
        );

        let margin = 12.0;
        let image_rect = UiRect {
            x: rect.x + margin,
            y: rect.y + margin,
            w: (rect.w - margin * 2.0).max(1.0),
            h: (rect.h * 0.58).max(1.0),
        };
        ui.send(node.image, WidgetMessage::DesiredPosition(image_rect.pos()));
        ui.send(node.image, WidgetMessage::Width(image_rect.w));
        ui.send(node.image, WidgetMessage::Height(image_rect.h));
        ui.send(node.image, ImageMessage::Texture(texture));

        let text_y = image_rect.y + image_rect.h + 10.0;
        ui.send(
            node.text,
            WidgetMessage::DesiredPosition(Vector2::new(rect.x + 14.0, text_y)),
        );
        ui.send(node.text, WidgetMessage::Width((rect.w - 28.0).max(1.0)));
        ui.send(
            node.text,
            WidgetMessage::Height((rect.y + rect.h - text_y - 10.0).max(1.0)),
        );
        ui.send(
            node.text,
            WidgetMessage::Foreground(Brush::Solid(fg_color).into()),
        );
        ui.send(node.text, TextMessage::Text(text));
        if active {
            self.pregame_button_rects.push((rect, action));
        }
        *index += 1;
    }

    fn hide_unused_pregame_nodes(&mut self, ui: &mut UserInterface, used: usize) {
        for button in self.ui_pregame.buttons.iter_mut().skip(used) {
            button.role = PregameVisualRole::Button;
            ui.send(
                button.bg,
                WidgetMessage::DesiredPosition(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS)),
            );
            ui.send(
                button.image,
                WidgetMessage::DesiredPosition(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS)),
            );
            ui.send(button.image, ImageMessage::Texture(None));
            ui.send(
                button.text,
                WidgetMessage::DesiredPosition(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS)),
            );
            ui.send(button.text, TextMessage::Text(String::new()));
        }
    }

    fn layout_pregame_home(&mut self, ui: &mut UserInterface, node_index: &mut usize) {
        // 頭像入口 block（畫面頂端左側）— 點擊進入個人統計數據頁。
        self.place_pregame_node(
            ui,
            node_index,
            pregame_ref_rect(self.window_size, 30.0, 20.0, 260.0, 92.0),
            format!("{}\nLv.{}", PROFILE_PLAYER_NAME, PROFILE_LEVEL),
            true,
            pregame::PregameAction::Navigate {
                target: "profile".to_string(),
            },
            PregameVisualRole::Button,
            Color::from_rgba(60, 130, 200, 255),
            Color::from_rgba(255, 255, 255, 255),
        );

        let deco = [
            (
                pregame_ref_rect(self.window_size, 250.0, 245.0, 190.0, 170.0),
                "",
                Color::from_rgba(190, 100, 25, 255),
            ),
            (
                pregame_ref_rect(self.window_size, 1510.0, 260.0, 190.0, 170.0),
                "",
                Color::from_rgba(205, 120, 30, 255),
            ),
            (
                pregame_ref_rect(self.window_size, 920.0, 365.0, 210.0, 210.0),
                "神像",
                Color::from_rgba(210, 200, 172, 255),
            ),
            (
                pregame_ref_rect(self.window_size, 160.0, 650.0, 260.0, 80.0),
                "",
                Color::from_rgba(80, 190, 80, 255),
            ),
            (
                pregame_ref_rect(self.window_size, 1470.0, 650.0, 260.0, 80.0),
                "",
                Color::from_rgba(80, 190, 80, 255),
            ),
        ];
        for (rect, text, color) in deco {
            self.place_pregame_node(
                ui,
                node_index,
                rect,
                text.to_string(),
                false,
                pregame::PregameAction::NoOp,
                PregameVisualRole::Decoration,
                color,
                Color::from_rgba(82, 65, 48, 255),
            );
        }

        let utility = [
            ("設定", 28.0, 185.0, Color::from_rgba(35, 195, 235, 255)),
            ("任務", 28.0, 325.0, Color::from_rgba(245, 190, 30, 255)),
            ("商店", 28.0, 465.0, Color::from_rgba(230, 75, 48, 255)),
        ];
        for (label, x, y, color) in utility {
            self.place_pregame_node(
                ui,
                node_index,
                pregame_ref_rect(self.window_size, x, y, 96.0, 96.0),
                label.to_string(),
                false,
                pregame::PregameAction::NoOp,
                PregameVisualRole::Button,
                color,
                Color::from_rgba(255, 255, 255, 255),
            );
        }

        let widgets = self
            .pregame_runtime
            .catalog
            .screen("main_menu")
            .map(|screen| screen.widgets.clone())
            .unwrap_or_default();
        let start = widgets.iter().find(|widget| widget.id == "start");
        let side_widgets: Vec<_> = widgets
            .iter()
            .filter(|widget| widget.id != "start")
            .take(2)
            .collect();
        let nav = [
            (
                side_widgets
                    .get(0)
                    .map(|widget| widget.label.as_str())
                    .unwrap_or("英雄"),
                pregame_ref_rect(self.window_size, 470.0, 930.0, 210.0, 132.0),
                side_widgets
                    .get(0)
                    .map(|widget| widget.action.clone())
                    .unwrap_or(pregame::PregameAction::NoOp),
                side_widgets
                    .get(0)
                    .map(|widget| widget.is_active())
                    .unwrap_or(false),
                Color::from_rgba(238, 145, 38, 255),
            ),
            (
                start.map(|widget| widget.label.as_str()).unwrap_or("開始"),
                pregame_ref_rect(self.window_size, 900.0, 866.0, 248.0, 190.0),
                start.map(|widget| widget.action.clone()).unwrap_or(
                    pregame::PregameAction::Navigate {
                        target: "difficulty_select".to_string(),
                    },
                ),
                start.map(|widget| widget.is_active()).unwrap_or(true),
                Color::from_rgba(50, 225, 35, 255),
            ),
            (
                side_widgets
                    .get(1)
                    .map(|widget| widget.label.as_str())
                    .unwrap_or("知識"),
                pregame_ref_rect(self.window_size, 1368.0, 930.0, 210.0, 132.0),
                side_widgets
                    .get(1)
                    .map(|widget| widget.action.clone())
                    .unwrap_or(pregame::PregameAction::NoOp),
                side_widgets
                    .get(1)
                    .map(|widget| widget.is_active())
                    .unwrap_or(false),
                Color::from_rgba(250, 185, 30, 255),
            ),
        ];
        for (label, rect, action, active, color) in nav {
            self.place_pregame_node(
                ui,
                node_index,
                rect,
                label.to_string(),
                active,
                action,
                PregameVisualRole::Button,
                color,
                Color::from_rgba(255, 255, 255, 255),
            );
        }
    }

    fn layout_pregame_difficulty(&mut self, ui: &mut UserInterface, node_index: &mut usize) {
        self.place_pregame_node(
            ui,
            node_index,
            pregame_ref_rect(self.window_size, 36.0, 28.0, 96.0, 96.0),
            "←\n返回".to_string(),
            true,
            pregame::PregameAction::Back,
            PregameVisualRole::Button,
            Color::from_rgba(35, 195, 235, 255),
            Color::from_rgba(255, 255, 255, 255),
        );

        let rects = [
            (560.0, 430.0, 280.0, 190.0),
            (884.0, 390.0, 280.0, 190.0),
            (1208.0, 430.0, 280.0, 190.0),
        ];
        let difficulties = self.pregame_runtime.catalog.difficulties.clone();
        for (difficulty, (x, y, w, h)) in difficulties.iter().zip(rects) {
            let reward = if difficulty.reward.trim().is_empty() {
                String::new()
            } else {
                format!("獎勵：{}", difficulty.reward)
            };
            self.place_pregame_node(
                ui,
                node_index,
                pregame_ref_rect(self.window_size, x, y, w, h),
                pregame_button_label(&difficulty.label, &reward, difficulty.enabled),
                difficulty.enabled,
                pregame::PregameAction::SelectDifficulty {
                    difficulty_id: difficulty.id.clone(),
                },
                PregameVisualRole::Button,
                Color::from_rgba(195, 150, 80, 255),
                Color::from_rgba(255, 255, 255, 255),
            );
        }

        self.place_pregame_node(
            ui,
            node_index,
            pregame_ref_rect(self.window_size, 42.0, 965.0, 160.0, 120.0),
            "更換英雄".to_string(),
            false,
            pregame::PregameAction::NoOp,
            PregameVisualRole::Button,
            Color::from_rgba(245, 145, 35, 255),
            Color::from_rgba(255, 255, 255, 255),
        );
    }

    fn layout_pregame_maps(&mut self, ui: &mut UserInterface, node_index: &mut usize) {
        self.place_pregame_node(
            ui,
            node_index,
            pregame_ref_rect(self.window_size, 36.0, 28.0, 96.0, 96.0),
            "←\n返回".to_string(),
            true,
            pregame::PregameAction::Back,
            PregameVisualRole::Button,
            Color::from_rgba(35, 195, 235, 255),
            Color::from_rgba(255, 255, 255, 255),
        );

        let rects = [
            (370.0, 120.0, 380.0, 300.0),
            (834.0, 120.0, 380.0, 300.0),
            (1298.0, 120.0, 380.0, 300.0),
            (370.0, 500.0, 380.0, 300.0),
            (834.0, 500.0, 380.0, 300.0),
            (1298.0, 500.0, 380.0, 300.0),
        ];
        let maps = self.current_selectable_maps();
        for (map, (x, y, w, h)) in maps.iter().take(6).zip(rects) {
            let description = if map.reward.trim().is_empty() {
                map.description.clone()
            } else {
                format!("{}\n{}", map.description, map.reward)
            };
            self.place_pregame_map_node(
                ui,
                node_index,
                pregame_ref_rect(self.window_size, x, y, w, h),
                pregame_button_label(&map.label, &description, map.is_playable()),
                map.is_playable(),
                pregame::PregameAction::SelectMap {
                    map_id: map.id.clone(),
                },
                if map.is_playable() {
                    Color::from_rgba(194, 154, 93, 255)
                } else {
                    Color::from_rgba(125, 105, 82, 230)
                },
                Color::from_rgba(255, 255, 255, 255),
                map.image.as_deref(),
            );
        }

        let difficulties = self.pregame_runtime.catalog.difficulties.clone();
        let selected_difficulty_id = self
            .pregame_runtime
            .selected_difficulty
            .as_ref()
            .map(|difficulty| difficulty.id.clone());
        for (i, difficulty) in difficulties.iter().take(4).enumerate() {
            let selected = selected_difficulty_id.as_deref() == Some(difficulty.id.as_str());
            self.place_pregame_node(
                ui,
                node_index,
                pregame_ref_rect(
                    self.window_size,
                    540.0 + i as f32 * 260.0,
                    970.0,
                    170.0,
                    96.0,
                ),
                difficulty.label.clone(),
                difficulty.enabled,
                pregame::PregameAction::SelectDifficulty {
                    difficulty_id: difficulty.id.clone(),
                },
                PregameVisualRole::Button,
                if selected {
                    Color::from_rgba(245, 195, 40, 255)
                } else {
                    Color::from_rgba(35, 170, 205, 255)
                },
                Color::from_rgba(255, 255, 255, 255),
            );
        }
    }

    fn update_pregame_ui(&mut self, ui: &mut UserInterface) {
        self.hide_gameplay_ui_for_pregame(ui);

        let is_settings = self.pregame_runtime.state == pregame::PregameState::Settings;
        if is_settings {
            self.hide_pregame_ui(ui);
            self.hide_profile_elements(ui);
            self.pregame_button_rects.clear();
            self.update_settings_panel(ui);
            return;
        }
        let is_profile = self.pregame_runtime.state == pregame::PregameState::Profile;
        if is_profile {
            self.hide_pregame_ui(ui);
            self.hide_settings_elements(ui);
            self.pregame_button_rects.clear();
            self.update_profile_panel(ui);
            return;
        }
        self.hide_settings_elements(ui);
        self.hide_profile_elements(ui);

        let screen_id = self.pregame_runtime.active_screen_id();
        let title = self
            .pregame_runtime
            .catalog
            .screen(screen_id)
            .map(|screen| screen.title.clone())
            .unwrap_or_else(|| "Omoba 塔防".to_string());
        let subtitle = self
            .pregame_runtime
            .catalog
            .screen(screen_id)
            .map(|screen| screen.subtitle.clone())
            .unwrap_or_default();
        let full = UiRect {
            x: 0.0,
            y: 0.0,
            w: self.window_size.x.max(1.0),
            h: self.window_size.y.max(1.0),
        };
        ui.send(
            self.ui_pregame.background,
            WidgetMessage::DesiredPosition(full.pos()),
        );
        ui.send(self.ui_pregame.background, WidgetMessage::Width(full.w));
        ui.send(self.ui_pregame.background, WidgetMessage::Height(full.h));
        ui.send(
            self.ui_pregame.background,
            WidgetMessage::Background(
                Brush::Solid(match self.pregame_runtime.state {
                    pregame::PregameState::MainMenu => Color::from_rgba(112, 211, 154, 255),
                    _ => Color::from_rgba(12, 91, 92, 238),
                })
                .into(),
            ),
        );

        let panel = match self.pregame_runtime.state {
            pregame::PregameState::MainMenu => full,
            _ => UiRect {
                x: full.w * 0.08,
                y: full.h * 0.08,
                w: full.w * 0.84,
                h: full.h * 0.84,
            },
        };
        ui.send(
            self.ui_pregame.panel,
            WidgetMessage::DesiredPosition(panel.pos()),
        );
        ui.send(self.ui_pregame.panel, WidgetMessage::Width(panel.w));
        ui.send(self.ui_pregame.panel, WidgetMessage::Height(panel.h));
        ui.send(
            self.ui_pregame.panel,
            WidgetMessage::Background(
                Brush::Solid(match self.pregame_runtime.state {
                    pregame::PregameState::MainMenu => Color::from_rgba(125, 221, 116, 0),
                    _ => Color::from_rgba(0, 38, 42, 96),
                })
                .into(),
            ),
        );

        let title_rect = UiRect {
            x: full.x + full.w * 0.18,
            y: full.y + full.h * 0.04,
            w: full.w * 0.64,
            h: 64.0,
        };
        ui.send(
            self.ui_pregame.title,
            WidgetMessage::DesiredPosition(title_rect.pos()),
        );
        ui.send(self.ui_pregame.title, WidgetMessage::Width(title_rect.w));
        ui.send(self.ui_pregame.title, TextMessage::Text(title));
        let subtitle_rect = UiRect {
            x: full.x + full.w * 0.22,
            y: title_rect.bottom() + 2.0,
            w: full.w * 0.56,
            h: 36.0,
        };
        ui.send(
            self.ui_pregame.subtitle,
            WidgetMessage::DesiredPosition(subtitle_rect.pos()),
        );
        ui.send(
            self.ui_pregame.subtitle,
            WidgetMessage::Width(subtitle_rect.w),
        );
        ui.send(self.ui_pregame.subtitle, TextMessage::Text(subtitle));

        let status = self.pregame_runtime.last_error.clone().unwrap_or_default();
        let status_rect = UiRect {
            x: full.x + full.w * 0.18,
            y: full.bottom() - 42.0,
            w: full.w * 0.64,
            h: 30.0,
        };
        ui.send(
            self.ui_pregame.status,
            WidgetMessage::DesiredPosition(status_rect.pos()),
        );
        ui.send(self.ui_pregame.status, WidgetMessage::Width(status_rect.w));
        ui.send(self.ui_pregame.status, TextMessage::Text(status));

        self.pregame_button_rects.clear();
        // 非設定頁時清掉設定頁專屬 rect（設定頁 layout 每 frame 會重新設定）
        self.settings_hotkey_btn_rect = UiRect::default();
        self.settings_resolution_rect = UiRect::default();
        let mut node_index = 0;
        match self.pregame_runtime.state {
            pregame::PregameState::MainMenu => {
                self.layout_pregame_home(ui, &mut node_index);
            }
            pregame::PregameState::DifficultySelect => {
                self.layout_pregame_difficulty(ui, &mut node_index);
            }
            pregame::PregameState::MapSelect => {
                self.layout_pregame_maps(ui, &mut node_index);
            }
            pregame::PregameState::StartingSession | pregame::PregameState::SessionEnded => {
                for (label, description, active, action) in self.current_pregame_buttons() {
                    self.place_pregame_node(
                        ui,
                        &mut node_index,
                        pregame_ref_rect(self.window_size, 744.0, 520.0, 560.0, 120.0),
                        pregame_button_label(&label, &description, active),
                        active,
                        action,
                        PregameVisualRole::Button,
                        Color::from_rgba(245, 178, 54, 255),
                        Color::from_rgba(255, 255, 255, 255),
                    );
                }
            }
            pregame::PregameState::InGame => {}
            pregame::PregameState::Settings => {}
            pregame::PregameState::Profile => {}
        }
        self.hide_unused_pregame_nodes(ui, node_index);
    }

    fn hide_settings_elements(&mut self, ui: &mut UserInterface) {
        let h = Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS);
        for handle in [
            self.ui_settings.bg,
            self.ui_settings.title_bg,
            self.ui_settings.resolution_badge_bg,
            self.ui_settings.left_panel_bg,
            self.ui_settings.left_panel_jukebox_bg,
            self.ui_settings.left_panel_jukebox_inner,
            self.ui_settings.left_panel_btn_bg,
            self.ui_settings.left_panel_toggle_bg,
            self.ui_settings.slider_panel_bg,
            self.ui_settings.back_btn,
            self.ui_settings.sfx.track,
            self.ui_settings.sfx.fill,
            self.ui_settings.sfx.thumb,
            self.ui_settings.sfx.icon,
            self.ui_settings.music.track,
            self.ui_settings.music.fill,
            self.ui_settings.music.thumb,
            self.ui_settings.music.icon,
            self.ui_settings.speed.track,
            self.ui_settings.speed.fill,
            self.ui_settings.speed.thumb,
            self.ui_settings.speed.icon,
        ] {
            ui.send(handle, WidgetMessage::DesiredPosition(h));
        }
        for handle in [
            self.ui_settings.title_text,
            self.ui_settings.back_btn_text,
            self.ui_settings.resolution_badge_label,
            self.ui_settings.resolution_badge_value,
            self.ui_settings.left_panel_btn_text,
            self.ui_settings.left_panel_enable_label,
            self.ui_settings.sfx.label,
            self.ui_settings.sfx.pct_text,
            self.ui_settings.music.label,
            self.ui_settings.music.pct_text,
            self.ui_settings.speed.label,
            self.ui_settings.speed.pct_text,
        ] {
            ui.send(handle, WidgetMessage::DesiredPosition(h));
        }
        for btn in &self.ui_settings.placeholder_btns {
            ui.send(btn.bg, WidgetMessage::DesiredPosition(h));
            ui.send(btn.label, WidgetMessage::DesiredPosition(h));
            ui.send(btn.sublabel, WidgetMessage::DesiredPosition(h));
        }
    }

    fn update_settings_panel(&mut self, ui: &mut UserInterface) {
        let ws = self.window_size;
        let full = UiRect {
            x: 0.0,
            y: 0.0,
            w: ws.x.max(1.0),
            h: ws.y.max(1.0),
        };

        // ── 全螢幕暗底 ──────────────────────────────────────────────
        ui.send(
            self.ui_settings.bg,
            WidgetMessage::DesiredPosition(full.pos()),
        );
        ui.send(self.ui_settings.bg, WidgetMessage::Width(full.w));
        ui.send(self.ui_settings.bg, WidgetMessage::Height(full.h));

        // ── 頂部欄 ──────────────────────────────────────────────────
        let title_h = full.h * 0.075;
        let bar_y = full.h * 0.025;

        // 返回按鈕（左）
        let back = UiRect {
            x: full.w * 0.04,
            y: bar_y,
            w: title_h * 2.2,
            h: title_h,
        };
        ui.send(
            self.ui_settings.back_btn,
            WidgetMessage::DesiredPosition(back.pos()),
        );
        ui.send(self.ui_settings.back_btn, WidgetMessage::Width(back.w));
        ui.send(self.ui_settings.back_btn, WidgetMessage::Height(back.h));
        ui.send(
            self.ui_settings.back_btn_text,
            WidgetMessage::DesiredPosition(back.pos()),
        );
        ui.send(self.ui_settings.back_btn_text, WidgetMessage::Width(back.w));
        ui.send(
            self.ui_settings.back_btn_text,
            WidgetMessage::Height(back.h),
        );
        ui.send(
            self.ui_settings.back_btn_text,
            TextMessage::Text("< 返回".to_string()),
        );
        self.ui_settings.back_btn_rect = back;

        // 標題列（中，寬一點貼近 BTD6）
        let title_bar = UiRect {
            x: full.w * 0.22,
            y: bar_y,
            w: full.w * 0.56,
            h: title_h,
        };
        ui.send(
            self.ui_settings.title_bg,
            WidgetMessage::DesiredPosition(title_bar.pos()),
        );
        ui.send(self.ui_settings.title_bg, WidgetMessage::Width(title_bar.w));
        ui.send(
            self.ui_settings.title_bg,
            WidgetMessage::Height(title_bar.h),
        );
        ui.send(
            self.ui_settings.title_text,
            WidgetMessage::DesiredPosition(title_bar.pos()),
        );
        ui.send(
            self.ui_settings.title_text,
            WidgetMessage::Width(title_bar.w),
        );
        ui.send(
            self.ui_settings.title_text,
            WidgetMessage::Height(title_bar.h),
        );
        ui.send(
            self.ui_settings.title_text,
            TextMessage::Text("設定".to_string()),
        );

        // 解析度 badge（右）——可點擊循環切換解析度
        let badge_w = full.w * 0.14;
        let badge = UiRect {
            x: full.w * 0.96 - badge_w,
            y: bar_y,
            w: badge_w,
            h: title_h,
        };
        self.settings_resolution_rect = badge;
        ui.send(
            self.ui_settings.resolution_badge_bg,
            WidgetMessage::DesiredPosition(badge.pos()),
        );
        ui.send(
            self.ui_settings.resolution_badge_bg,
            WidgetMessage::Width(badge.w),
        );
        ui.send(
            self.ui_settings.resolution_badge_bg,
            WidgetMessage::Height(badge.h),
        );
        let badge_label = UiRect {
            x: badge.x,
            y: badge.y + badge.h * 0.10,
            w: badge.w,
            h: title_h * 0.38,
        };
        ui.send(
            self.ui_settings.resolution_badge_label,
            WidgetMessage::DesiredPosition(badge_label.pos()),
        );
        ui.send(
            self.ui_settings.resolution_badge_label,
            WidgetMessage::Width(badge_label.w),
        );
        ui.send(
            self.ui_settings.resolution_badge_label,
            WidgetMessage::Height(badge_label.h),
        );
        ui.send(
            self.ui_settings.resolution_badge_label,
            TextMessage::Text("螢幕尺寸".to_string()),
        );
        let badge_val = UiRect {
            x: badge.x,
            y: badge.y + title_h * 0.48,
            w: badge.w,
            h: title_h * 0.52,
        };
        ui.send(
            self.ui_settings.resolution_badge_value,
            WidgetMessage::DesiredPosition(badge_val.pos()),
        );
        ui.send(
            self.ui_settings.resolution_badge_value,
            WidgetMessage::Width(badge_val.w),
        );
        ui.send(
            self.ui_settings.resolution_badge_value,
            WidgetMessage::Height(badge_val.h),
        );
        ui.send(
            self.ui_settings.resolution_badge_value,
            TextMessage::Text(format!("{} x {}", ws.x as u32, ws.y as u32)),
        );

        // ── 主內容區（兩欄，佔螢幕高度 62%，不 clamp）────────────────
        let content_y = title_bar.bottom() + full.h * 0.015;
        let content_h = full.h * 0.62;
        let slider_h = content_h / 4.2;
        let row_gap = (content_h - slider_h * 3.0) / 4.0;

        // 左欄裝飾面板
        let left_w = full.w * 0.22;
        let left = UiRect {
            x: full.w * 0.04,
            y: content_y,
            w: left_w,
            h: content_h,
        };
        ui.send(
            self.ui_settings.left_panel_bg,
            WidgetMessage::DesiredPosition(left.pos()),
        );
        ui.send(self.ui_settings.left_panel_bg, WidgetMessage::Width(left.w));
        ui.send(
            self.ui_settings.left_panel_bg,
            WidgetMessage::Height(left.h),
        );

        // Jukebox 插圖區（上半部 50%）
        let juke_pad = left_w * 0.06;
        let juke = UiRect {
            x: left.x + juke_pad,
            y: left.y + juke_pad,
            w: left.w - juke_pad * 2.0,
            h: left.h * 0.48,
        };
        ui.send(
            self.ui_settings.left_panel_jukebox_bg,
            WidgetMessage::DesiredPosition(juke.pos()),
        );
        ui.send(
            self.ui_settings.left_panel_jukebox_bg,
            WidgetMessage::Width(juke.w),
        );
        ui.send(
            self.ui_settings.left_panel_jukebox_bg,
            WidgetMessage::Height(juke.h),
        );
        // 內部裝飾塊（模擬老虎機機身）
        let inner_pad = juke.w * 0.12;
        let inner = UiRect {
            x: juke.x + inner_pad,
            y: juke.y + juke.h * 0.18,
            w: juke.w - inner_pad * 2.0,
            h: juke.h * 0.55,
        };
        ui.send(
            self.ui_settings.left_panel_jukebox_inner,
            WidgetMessage::DesiredPosition(inner.pos()),
        );
        ui.send(
            self.ui_settings.left_panel_jukebox_inner,
            WidgetMessage::Width(inner.w),
        );
        ui.send(
            self.ui_settings.left_panel_jukebox_inner,
            WidgetMessage::Height(inner.h),
        );

        let lpad = left_w * 0.10;
        let btn_h = left_w * 0.28;
        let lbtn = UiRect {
            x: left.x + lpad,
            y: left.y + left.h * 0.60,
            w: left.w - lpad * 2.0,
            h: btn_h,
        };
        ui.send(
            self.ui_settings.left_panel_btn_bg,
            WidgetMessage::DesiredPosition(lbtn.pos()),
        );
        ui.send(
            self.ui_settings.left_panel_btn_bg,
            WidgetMessage::Width(lbtn.w),
        );
        ui.send(
            self.ui_settings.left_panel_btn_bg,
            WidgetMessage::Height(lbtn.h),
        );
        ui.send(
            self.ui_settings.left_panel_btn_text,
            WidgetMessage::DesiredPosition(lbtn.pos()),
        );
        ui.send(
            self.ui_settings.left_panel_btn_text,
            WidgetMessage::Width(lbtn.w),
        );
        ui.send(
            self.ui_settings.left_panel_btn_text,
            WidgetMessage::Height(lbtn.h),
        );
        ui.send(
            self.ui_settings.left_panel_btn_text,
            TextMessage::Text("自動唱機".to_string()),
        );

        let tog_w = left_w * 0.24;
        let tog_h = tog_w * 0.5;
        let tog = UiRect {
            x: left.x + left.w - lpad - tog_w,
            y: left.y + left.h * 0.80,
            w: tog_w,
            h: tog_h,
        };
        ui.send(
            self.ui_settings.left_panel_toggle_bg,
            WidgetMessage::DesiredPosition(tog.pos()),
        );
        ui.send(
            self.ui_settings.left_panel_toggle_bg,
            WidgetMessage::Width(tog.w),
        );
        ui.send(
            self.ui_settings.left_panel_toggle_bg,
            WidgetMessage::Height(tog.h),
        );

        let en_label = UiRect {
            x: left.x + lpad,
            y: tog.y,
            w: tog.x - left.x - lpad * 1.5,
            h: tog_h,
        };
        ui.send(
            self.ui_settings.left_panel_enable_label,
            WidgetMessage::DesiredPosition(en_label.pos()),
        );
        ui.send(
            self.ui_settings.left_panel_enable_label,
            WidgetMessage::Width(en_label.w),
        );
        ui.send(
            self.ui_settings.left_panel_enable_label,
            WidgetMessage::Height(en_label.h),
        );
        ui.send(
            self.ui_settings.left_panel_enable_label,
            TextMessage::Text("啟用".to_string()),
        );

        // 右欄滑桿面板（延伸至 96% 螢幕寬度）
        let right_x = left.right() + full.w * 0.015;
        let right_w = full.w * 0.955 - right_x;
        let right = UiRect {
            x: right_x,
            y: content_y,
            w: right_w,
            h: content_h,
        };
        ui.send(
            self.ui_settings.slider_panel_bg,
            WidgetMessage::DesiredPosition(right.pos()),
        );
        ui.send(
            self.ui_settings.slider_panel_bg,
            WidgetMessage::Width(right.w),
        );
        ui.send(
            self.ui_settings.slider_panel_bg,
            WidgetMessage::Height(right.h),
        );

        let icon_r = full.h * 0.024; // 固定比例，不受 slider_h 影響（1440p ≈ 35px）
        let rpad = right_w * 0.03;
        let icon_cx = right.x + rpad + icon_r;
        let pct_w = (right_w * 0.07).max(55.0);
        let lbl_w = full.h * 0.058;
        let track_x = icon_cx + icon_r + rpad * 0.4 + lbl_w + rpad * 0.3;
        // 在 track 右端留 icon_r 空間（thumb 半徑），避免 100% 時 thumb 蓋住 pct_text
        let track_w = right.right() - rpad - pct_w - track_x - icon_r - rpad * 0.3;

        let layout_slider = |ui: &mut UserInterface,
                             slider: &mut SettingsSlider,
                             row: usize,
                             volume: f32,
                             pct: &str,
                             label_text: &str| {
            let cy = right.y + row_gap + row as f32 * (slider_h + row_gap) + slider_h * 0.5;
            // Icon circle
            let icon_x = icon_cx - icon_r;
            let icon_y = cy - icon_r;
            ui.send(
                slider.icon,
                WidgetMessage::DesiredPosition(Vector2::new(icon_x, icon_y)),
            );
            ui.send(slider.icon, WidgetMessage::Width(icon_r * 2.0));
            ui.send(slider.icon, WidgetMessage::Height(icon_r * 2.0));
            // Label text（圖示右側，軌道左側）
            let lbl_rect = UiRect {
                x: icon_cx + icon_r + rpad * 0.4,
                y: cy - slider_h * 0.5,
                w: lbl_w,
                h: slider_h,
            };
            ui.send(slider.label, WidgetMessage::DesiredPosition(lbl_rect.pos()));
            ui.send(slider.label, WidgetMessage::Width(lbl_rect.w));
            ui.send(slider.label, WidgetMessage::Height(lbl_rect.h));
            ui.send(slider.label, TextMessage::Text(label_text.to_string()));
            // Track
            let track_h = slider_h * 0.28;
            let track = UiRect {
                x: track_x,
                y: cy - track_h * 0.5,
                w: track_w,
                h: track_h,
            };
            slider.track_rect = UiRect {
                x: track_x,
                y: cy - slider_h * 0.5,
                w: track_w,
                h: slider_h,
            };
            ui.send(slider.track, WidgetMessage::DesiredPosition(track.pos()));
            ui.send(slider.track, WidgetMessage::Width(track.w));
            ui.send(slider.track, WidgetMessage::Height(track.h));
            // Fill
            let fill_w = (track.w * volume).max(0.0);
            ui.send(slider.fill, WidgetMessage::DesiredPosition(track.pos()));
            ui.send(slider.fill, WidgetMessage::Width(fill_w.max(1.0)));
            ui.send(slider.fill, WidgetMessage::Height(track.h));
            // Thumb（與 icon 同尺寸，center clamp 在 track 範圍內）
            let thumb_r = icon_r;
            let thumb_cx = (track.x + fill_w).clamp(track.x, track.right());
            let thumb_x = thumb_cx - thumb_r;
            let thumb_y = cy - thumb_r;
            ui.send(
                slider.thumb,
                WidgetMessage::DesiredPosition(Vector2::new(thumb_x, thumb_y)),
            );
            ui.send(slider.thumb, WidgetMessage::Width(thumb_r * 2.0));
            ui.send(slider.thumb, WidgetMessage::Height(thumb_r * 2.0));
            // Percentage text（起點在 thumb 右側，避免 100% 時被蓋住）
            let pct_rect = UiRect {
                x: track.right() + icon_r + rpad * 0.3,
                y: cy - slider_h * 0.5,
                w: pct_w,
                h: slider_h,
            };
            ui.send(
                slider.pct_text,
                WidgetMessage::DesiredPosition(pct_rect.pos()),
            );
            ui.send(slider.pct_text, WidgetMessage::Width(pct_rect.w));
            ui.send(slider.pct_text, WidgetMessage::Height(pct_rect.h));
            ui.send(slider.pct_text, TextMessage::Text(pct.to_string()));
        };

        let sfx_v = self.settings_sfx_volume;
        let music_v = self.settings_music_volume;
        let speed_v = self.settings_speed_value;
        let sfx_pct = format!("{}%", (sfx_v * 100.0) as u32);
        let music_pct = format!("{}%", (music_v * 100.0) as u32);
        let speed_pct = format!("{}%", (speed_v * 100.0) as u32);
        layout_slider(
            ui,
            &mut self.ui_settings.music,
            0,
            music_v,
            &music_pct,
            "音樂",
        );
        layout_slider(ui, &mut self.ui_settings.sfx, 1, sfx_v, &sfx_pct, "音效");
        layout_slider(
            ui,
            &mut self.ui_settings.speed,
            2,
            speed_v,
            &speed_pct,
            "速度",
        );

        // ── 底部 6 個圖示按鈕（不 clamp，跟螢幕等比）────────────────
        let btn_size = full.h * 0.09;
        let sub_h = full.h * 0.035;
        let btn_gap = full.w * 0.025;
        let btn_names = ["備份", "賬戶", "登出", "語言", "熱鍵", "輔助"];
        let btn_icons = ["雲", "人", "出", "文", "鍵", "助"];
        let total_btn_w =
            btn_size * btn_names.len() as f32 + btn_gap * (btn_names.len() - 1) as f32;
        let btn_start_x = (full.w - total_btn_w) * 0.5;
        let btn_row_y = content_y + content_h + full.h * 0.03;
        self.settings_hotkey_btn_rect = UiRect::default();
        for (i, btn) in self.ui_settings.placeholder_btns.iter().enumerate() {
            if i >= btn_names.len() {
                break;
            }
            let bx = btn_start_x + i as f32 * (btn_size + btn_gap);
            let br = UiRect {
                x: bx,
                y: btn_row_y,
                w: btn_size,
                h: btn_size,
            };
            if btn_names[i] == "熱鍵" {
                self.settings_hotkey_btn_rect = br;
            }
            ui.send(btn.bg, WidgetMessage::DesiredPosition(br.pos()));
            ui.send(btn.bg, WidgetMessage::Width(br.w));
            ui.send(btn.bg, WidgetMessage::Height(br.h));
            ui.send(btn.label, WidgetMessage::DesiredPosition(br.pos()));
            ui.send(btn.label, WidgetMessage::Width(br.w));
            ui.send(btn.label, WidgetMessage::Height(br.h));
            ui.send(btn.label, TextMessage::Text(btn_icons[i].to_string()));
            let sub = UiRect {
                x: bx,
                y: br.bottom() + 4.0,
                w: btn_size,
                h: sub_h,
            };
            ui.send(btn.sublabel, WidgetMessage::DesiredPosition(sub.pos()));
            ui.send(btn.sublabel, WidgetMessage::Width(sub.w));
            ui.send(btn.sublabel, WidgetMessage::Height(sub.h));
            ui.send(btn.sublabel, TextMessage::Text(btn_names[i].to_string()));
        }

        // Register back button
        self.pregame_button_rects
            .push((back, pregame::PregameAction::Back));

        // 設定頁也要驅動解析度下拉選單與熱鍵面板
        self.update_resolution_dropdown(ui);
        self.update_hotkey_panel(ui);
    }

    fn hide_profile_elements(&mut self, ui: &mut UserInterface) {
        let h = Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS);
        for handle in [
            self.ui_profile.bg,
            self.ui_profile.back_btn,
            self.ui_profile.header_bg,
            self.ui_profile.avatar_bg,
            self.ui_profile.level_badge_bg,
            self.ui_profile.xp_track,
            self.ui_profile.xp_fill,
            self.ui_profile.visibility_toggle_bg,
            self.ui_profile.publish_btn_bg,
            self.ui_profile.settings_gear_bg,
            self.ui_profile.heroes_banner_bg,
            self.ui_profile.monkeys_banner_bg,
            self.ui_profile.stats_header_bg,
        ] {
            ui.send(handle, WidgetMessage::DesiredPosition(h));
        }
        for handle in [
            self.ui_profile.back_btn_text,
            self.ui_profile.avatar_text,
            self.ui_profile.name_text,
            self.ui_profile.level_text,
            self.ui_profile.followers_text,
            self.ui_profile.visibility_text,
            self.ui_profile.publish_btn_text,
            self.ui_profile.settings_gear_text,
            self.ui_profile.medals_title_text,
            self.ui_profile.heroes_banner_text,
            self.ui_profile.monkeys_banner_text,
            self.ui_profile.stats_header_text,
            self.ui_profile.stats_collapse_text,
            self.ui_profile.stats_share_count_text,
        ] {
            ui.send(handle, WidgetMessage::DesiredPosition(h));
        }
        for medal in &self.ui_profile.medals {
            ui.send(medal.bg, WidgetMessage::DesiredPosition(h));
            ui.send(medal.count_text, WidgetMessage::DesiredPosition(h));
        }
        for portrait in self.ui_profile.heroes.iter().chain(self.ui_profile.monkeys.iter()) {
            ui.send(portrait.bg, WidgetMessage::DesiredPosition(h));
            ui.send(portrait.name_text, WidgetMessage::DesiredPosition(h));
            ui.send(portrait.count_text, WidgetMessage::DesiredPosition(h));
        }
        for row in &self.ui_profile.stats_rows {
            ui.send(row.label_text, WidgetMessage::DesiredPosition(h));
            ui.send(row.value_text, WidgetMessage::DesiredPosition(h));
            ui.send(row.checkbox_bg, WidgetMessage::DesiredPosition(h));
        }
    }

    /// 個人統計數據頁（Profile，Phase 1 版面骨架）。座標全用 2048x1152 參考空間
    /// 透過 `pregame_ref_rect` 換算，隨視窗尺寸等比縮放，與其餘 pregame UI 一致。
    /// 三大區塊：標頭列（頭像/名稱/等級/XP/追蹤者/公開切換/設定齒輪）、
    /// 獎牌牆、下方雙欄（左＝頂級英雄/頂級砲塔、右＝整體統計數據列表）。
    fn update_profile_panel(&mut self, ui: &mut UserInterface) {
        let ws = self.window_size;
        let full = UiRect {
            x: 0.0,
            y: 0.0,
            w: ws.x.max(1.0),
            h: ws.y.max(1.0),
        };

        ui.send(
            self.ui_profile.bg,
            WidgetMessage::DesiredPosition(full.pos()),
        );
        ui.send(self.ui_profile.bg, WidgetMessage::Width(full.w));
        ui.send(self.ui_profile.bg, WidgetMessage::Height(full.h));

        // ── 返回按鈕（左上角）──────────────────────────────────────
        let back = pregame_ref_rect(ws, 20.0, 15.0, 160.0, 80.0);
        ui.send(
            self.ui_profile.back_btn,
            WidgetMessage::DesiredPosition(back.pos()),
        );
        ui.send(self.ui_profile.back_btn, WidgetMessage::Width(back.w));
        ui.send(self.ui_profile.back_btn, WidgetMessage::Height(back.h));
        ui.send(
            self.ui_profile.back_btn_text,
            WidgetMessage::DesiredPosition(back.pos()),
        );
        ui.send(
            self.ui_profile.back_btn_text,
            WidgetMessage::Width(back.w),
        );
        ui.send(
            self.ui_profile.back_btn_text,
            WidgetMessage::Height(back.h),
        );
        ui.send(
            self.ui_profile.back_btn_text,
            TextMessage::Text("< 返回".to_string()),
        );
        self.ui_profile.back_btn_rect = back;
        self.pregame_button_rects
            .push((back, pregame::PregameAction::Back));

        // ── 標頭列 ────────────────────────────────────────────────
        let header = pregame_ref_rect(ws, 20.0, 110.0, 2008.0, 140.0);
        ui.send(
            self.ui_profile.header_bg,
            WidgetMessage::DesiredPosition(header.pos()),
        );
        ui.send(self.ui_profile.header_bg, WidgetMessage::Width(header.w));
        ui.send(self.ui_profile.header_bg, WidgetMessage::Height(header.h));

        let avatar = pregame_ref_rect(ws, 40.0, 125.0, 110.0, 110.0);
        ui.send(
            self.ui_profile.avatar_bg,
            WidgetMessage::DesiredPosition(avatar.pos()),
        );
        ui.send(self.ui_profile.avatar_bg, WidgetMessage::Width(avatar.w));
        ui.send(self.ui_profile.avatar_bg, WidgetMessage::Height(avatar.h));
        ui.send(
            self.ui_profile.avatar_text,
            WidgetMessage::DesiredPosition(avatar.pos()),
        );
        ui.send(self.ui_profile.avatar_text, WidgetMessage::Width(avatar.w));
        ui.send(
            self.ui_profile.avatar_text,
            WidgetMessage::Height(avatar.h),
        );
        ui.send(
            self.ui_profile.avatar_text,
            TextMessage::Text(
                PROFILE_PLAYER_NAME
                    .chars()
                    .next()
                    .map(|c| c.to_string())
                    .unwrap_or_default(),
            ),
        );

        let name = pregame_ref_rect(ws, 170.0, 128.0, 300.0, 38.0);
        ui.send(
            self.ui_profile.name_text,
            WidgetMessage::DesiredPosition(name.pos()),
        );
        ui.send(self.ui_profile.name_text, WidgetMessage::Width(name.w));
        ui.send(self.ui_profile.name_text, WidgetMessage::Height(name.h));
        ui.send(
            self.ui_profile.name_text,
            TextMessage::Text(PROFILE_PLAYER_NAME.to_string()),
        );

        let level_badge = pregame_ref_rect(ws, 170.0, 170.0, 74.0, 34.0);
        ui.send(
            self.ui_profile.level_badge_bg,
            WidgetMessage::DesiredPosition(level_badge.pos()),
        );
        ui.send(
            self.ui_profile.level_badge_bg,
            WidgetMessage::Width(level_badge.w),
        );
        ui.send(
            self.ui_profile.level_badge_bg,
            WidgetMessage::Height(level_badge.h),
        );
        ui.send(
            self.ui_profile.level_text,
            WidgetMessage::DesiredPosition(level_badge.pos()),
        );
        ui.send(
            self.ui_profile.level_text,
            WidgetMessage::Width(level_badge.w),
        );
        ui.send(
            self.ui_profile.level_text,
            WidgetMessage::Height(level_badge.h),
        );
        ui.send(
            self.ui_profile.level_text,
            TextMessage::Text(format!("★ Lv.{}", PROFILE_LEVEL)),
        );

        let xp_track = pregame_ref_rect(ws, 254.0, 178.0, 220.0, 18.0);
        ui.send(
            self.ui_profile.xp_track,
            WidgetMessage::DesiredPosition(xp_track.pos()),
        );
        ui.send(
            self.ui_profile.xp_track,
            WidgetMessage::Width(xp_track.w),
        );
        ui.send(
            self.ui_profile.xp_track,
            WidgetMessage::Height(xp_track.h),
        );
        let xp_fill_w = (xp_track.w * PROFILE_XP_PCT.clamp(0.0, 1.0)).max(1.0);
        ui.send(
            self.ui_profile.xp_fill,
            WidgetMessage::DesiredPosition(xp_track.pos()),
        );
        ui.send(self.ui_profile.xp_fill, WidgetMessage::Width(xp_fill_w));
        ui.send(
            self.ui_profile.xp_fill,
            WidgetMessage::Height(xp_track.h),
        );

        let followers = pregame_ref_rect(ws, 500.0, 150.0, 200.0, 60.0);
        ui.send(
            self.ui_profile.followers_text,
            WidgetMessage::DesiredPosition(followers.pos()),
        );
        ui.send(
            self.ui_profile.followers_text,
            WidgetMessage::Width(followers.w),
        );
        ui.send(
            self.ui_profile.followers_text,
            WidgetMessage::Height(followers.h),
        );
        ui.send(
            self.ui_profile.followers_text,
            TextMessage::Text(format!("追蹤者\n{}", PROFILE_FOLLOWERS)),
        );

        let visibility = pregame_ref_rect(ws, 740.0, 155.0, 130.0, 46.0);
        ui.send(
            self.ui_profile.visibility_toggle_bg,
            WidgetMessage::DesiredPosition(visibility.pos()),
        );
        ui.send(
            self.ui_profile.visibility_toggle_bg,
            WidgetMessage::Width(visibility.w),
        );
        ui.send(
            self.ui_profile.visibility_toggle_bg,
            WidgetMessage::Height(visibility.h),
        );
        ui.send(
            self.ui_profile.visibility_text,
            WidgetMessage::DesiredPosition(visibility.pos()),
        );
        ui.send(
            self.ui_profile.visibility_text,
            WidgetMessage::Width(visibility.w),
        );
        ui.send(
            self.ui_profile.visibility_text,
            WidgetMessage::Height(visibility.h),
        );
        ui.send(
            self.ui_profile.visibility_text,
            TextMessage::Text("公開".to_string()),
        );

        let publish = pregame_ref_rect(ws, 890.0, 150.0, 260.0, 56.0);
        ui.send(
            self.ui_profile.publish_btn_bg,
            WidgetMessage::DesiredPosition(publish.pos()),
        );
        ui.send(
            self.ui_profile.publish_btn_bg,
            WidgetMessage::Width(publish.w),
        );
        ui.send(
            self.ui_profile.publish_btn_bg,
            WidgetMessage::Height(publish.h),
        );
        ui.send(
            self.ui_profile.publish_btn_text,
            WidgetMessage::DesiredPosition(publish.pos()),
        );
        ui.send(
            self.ui_profile.publish_btn_text,
            WidgetMessage::Width(publish.w),
        );
        ui.send(
            self.ui_profile.publish_btn_text,
            WidgetMessage::Height(publish.h),
        );
        ui.send(
            self.ui_profile.publish_btn_text,
            TextMessage::Text("公開統計數據".to_string()),
        );

        let gear = pregame_ref_rect(ws, 1908.0, 140.0, 80.0, 80.0);
        ui.send(
            self.ui_profile.settings_gear_bg,
            WidgetMessage::DesiredPosition(gear.pos()),
        );
        ui.send(
            self.ui_profile.settings_gear_bg,
            WidgetMessage::Width(gear.w),
        );
        ui.send(
            self.ui_profile.settings_gear_bg,
            WidgetMessage::Height(gear.h),
        );
        ui.send(
            self.ui_profile.settings_gear_text,
            WidgetMessage::DesiredPosition(gear.pos()),
        );
        ui.send(
            self.ui_profile.settings_gear_text,
            WidgetMessage::Width(gear.w),
        );
        ui.send(
            self.ui_profile.settings_gear_text,
            WidgetMessage::Height(gear.h),
        );
        ui.send(
            self.ui_profile.settings_gear_text,
            TextMessage::Text("⚙".to_string()),
        );
        // 齒輪按鈕點回「設定」頁（Phase 1：沿用既有設定頁，不重複造輪子）
        self.pregame_button_rects.push((
            gear,
            pregame::PregameAction::Navigate {
                target: "settings".to_string(),
            },
        ));

        // ── 獎牌牆 ────────────────────────────────────────────────
        let medals_title = pregame_ref_rect(ws, 40.0, 270.0, 300.0, 36.0);
        ui.send(
            self.ui_profile.medals_title_text,
            WidgetMessage::DesiredPosition(medals_title.pos()),
        );
        ui.send(
            self.ui_profile.medals_title_text,
            WidgetMessage::Width(medals_title.w),
        );
        ui.send(
            self.ui_profile.medals_title_text,
            WidgetMessage::Height(medals_title.h),
        );
        ui.send(
            self.ui_profile.medals_title_text,
            TextMessage::Text("獎牌".to_string()),
        );

        let medal_size = 96.0;
        let medal_gap = 18.0;
        for (i, ((label, count), medal_ui)) in PROFILE_MEDALS
            .iter()
            .zip(self.ui_profile.medals.iter())
            .enumerate()
        {
            let _ = label;
            let mx = 40.0 + i as f32 * (medal_size + medal_gap);
            let rect = pregame_ref_rect(ws, mx, 320.0, medal_size, medal_size);
            ui.send(medal_ui.bg, WidgetMessage::DesiredPosition(rect.pos()));
            ui.send(medal_ui.bg, WidgetMessage::Width(rect.w));
            ui.send(medal_ui.bg, WidgetMessage::Height(rect.h));
            let badge = UiRect {
                x: rect.right() - rect.w * 0.34,
                y: rect.bottom() - rect.h * 0.34,
                w: rect.w * 0.34,
                h: rect.h * 0.34,
            };
            ui.send(
                medal_ui.count_text,
                WidgetMessage::DesiredPosition(badge.pos()),
            );
            ui.send(medal_ui.count_text, WidgetMessage::Width(badge.w));
            ui.send(medal_ui.count_text, WidgetMessage::Height(badge.h));
            ui.send(
                medal_ui.count_text,
                TextMessage::Text(format!("x{}", count)),
            );
        }

        // ── 下方雙欄 ──────────────────────────────────────────────
        let lower_y = 480.0;
        let left_col_x = 40.0;
        let left_col_w = 960.0;

        let heroes_banner = pregame_ref_rect(ws, left_col_x, lower_y, left_col_w, 44.0);
        ui.send(
            self.ui_profile.heroes_banner_bg,
            WidgetMessage::DesiredPosition(heroes_banner.pos()),
        );
        ui.send(
            self.ui_profile.heroes_banner_bg,
            WidgetMessage::Width(heroes_banner.w),
        );
        ui.send(
            self.ui_profile.heroes_banner_bg,
            WidgetMessage::Height(heroes_banner.h),
        );
        ui.send(
            self.ui_profile.heroes_banner_text,
            WidgetMessage::DesiredPosition(heroes_banner.pos()),
        );
        ui.send(
            self.ui_profile.heroes_banner_text,
            WidgetMessage::Width(heroes_banner.w),
        );
        ui.send(
            self.ui_profile.heroes_banner_text,
            WidgetMessage::Height(heroes_banner.h),
        );
        ui.send(
            self.ui_profile.heroes_banner_text,
            TextMessage::Text("頂級英雄".to_string()),
        );

        let portrait_w = 300.0;
        let portrait_h = 190.0;
        let portrait_gap = 30.0;
        for (i, ((label, count), portrait)) in PROFILE_TOP_HEROES
            .iter()
            .zip(self.ui_profile.heroes.iter())
            .enumerate()
        {
            let px = left_col_x + i as f32 * (portrait_w + portrait_gap);
            let rect = pregame_ref_rect(ws, px, lower_y + 60.0, portrait_w, portrait_h);
            ui.send(portrait.bg, WidgetMessage::DesiredPosition(rect.pos()));
            ui.send(portrait.bg, WidgetMessage::Width(rect.w));
            ui.send(portrait.bg, WidgetMessage::Height(rect.h));
            let name_rect = UiRect {
                x: rect.x,
                y: rect.bottom() - rect.h * 0.30,
                w: rect.w,
                h: rect.h * 0.18,
            };
            ui.send(
                portrait.name_text,
                WidgetMessage::DesiredPosition(name_rect.pos()),
            );
            ui.send(portrait.name_text, WidgetMessage::Width(name_rect.w));
            ui.send(portrait.name_text, WidgetMessage::Height(name_rect.h));
            ui.send(
                portrait.name_text,
                TextMessage::Text(label.to_string()),
            );
            let count_rect = UiRect {
                x: rect.x,
                y: rect.bottom() - rect.h * 0.12,
                w: rect.w,
                h: rect.h * 0.12,
            };
            ui.send(
                portrait.count_text,
                WidgetMessage::DesiredPosition(count_rect.pos()),
            );
            ui.send(portrait.count_text, WidgetMessage::Width(count_rect.w));
            ui.send(portrait.count_text, WidgetMessage::Height(count_rect.h));
            ui.send(
                portrait.count_text,
                TextMessage::Text(format!("使用 {} 次", count)),
            );
        }

        let monkeys_y = lower_y + 60.0 + portrait_h + 40.0;
        let monkeys_banner = pregame_ref_rect(ws, left_col_x, monkeys_y, left_col_w, 44.0);
        ui.send(
            self.ui_profile.monkeys_banner_bg,
            WidgetMessage::DesiredPosition(monkeys_banner.pos()),
        );
        ui.send(
            self.ui_profile.monkeys_banner_bg,
            WidgetMessage::Width(monkeys_banner.w),
        );
        ui.send(
            self.ui_profile.monkeys_banner_bg,
            WidgetMessage::Height(monkeys_banner.h),
        );
        ui.send(
            self.ui_profile.monkeys_banner_text,
            WidgetMessage::DesiredPosition(monkeys_banner.pos()),
        );
        ui.send(
            self.ui_profile.monkeys_banner_text,
            WidgetMessage::Width(monkeys_banner.w),
        );
        ui.send(
            self.ui_profile.monkeys_banner_text,
            WidgetMessage::Height(monkeys_banner.h),
        );
        ui.send(
            self.ui_profile.monkeys_banner_text,
            TextMessage::Text("頂級砲塔".to_string()),
        );

        for (i, ((label, count), portrait)) in PROFILE_TOP_MONKEYS
            .iter()
            .zip(self.ui_profile.monkeys.iter())
            .enumerate()
        {
            let px = left_col_x + i as f32 * (portrait_w + portrait_gap);
            let rect = pregame_ref_rect(ws, px, monkeys_y + 60.0, portrait_w, portrait_h);
            ui.send(portrait.bg, WidgetMessage::DesiredPosition(rect.pos()));
            ui.send(portrait.bg, WidgetMessage::Width(rect.w));
            ui.send(portrait.bg, WidgetMessage::Height(rect.h));
            let name_rect = UiRect {
                x: rect.x,
                y: rect.bottom() - rect.h * 0.30,
                w: rect.w,
                h: rect.h * 0.18,
            };
            ui.send(
                portrait.name_text,
                WidgetMessage::DesiredPosition(name_rect.pos()),
            );
            ui.send(portrait.name_text, WidgetMessage::Width(name_rect.w));
            ui.send(portrait.name_text, WidgetMessage::Height(name_rect.h));
            ui.send(
                portrait.name_text,
                TextMessage::Text(label.to_string()),
            );
            let count_rect = UiRect {
                x: rect.x,
                y: rect.bottom() - rect.h * 0.12,
                w: rect.w,
                h: rect.h * 0.12,
            };
            ui.send(
                portrait.count_text,
                WidgetMessage::DesiredPosition(count_rect.pos()),
            );
            ui.send(portrait.count_text, WidgetMessage::Width(count_rect.w));
            ui.send(portrait.count_text, WidgetMessage::Height(count_rect.h));
            ui.send(
                portrait.count_text,
                TextMessage::Text(format!("使用 {} 次", count)),
            );
        }

        // ── 右欄：整體統計數據 ────────────────────────────────────
        let right_col_x = left_col_x + left_col_w + 30.0;
        let right_col_w = 2008.0 - left_col_w - 30.0;

        let stats_header = pregame_ref_rect(ws, right_col_x, lower_y, right_col_w, 50.0);
        ui.send(
            self.ui_profile.stats_header_bg,
            WidgetMessage::DesiredPosition(stats_header.pos()),
        );
        ui.send(
            self.ui_profile.stats_header_bg,
            WidgetMessage::Width(stats_header.w),
        );
        ui.send(
            self.ui_profile.stats_header_bg,
            WidgetMessage::Height(stats_header.h),
        );
        let header_label_rect = UiRect {
            x: stats_header.x + 16.0,
            y: stats_header.y,
            w: stats_header.w * 0.6,
            h: stats_header.h,
        };
        ui.send(
            self.ui_profile.stats_header_text,
            WidgetMessage::DesiredPosition(header_label_rect.pos()),
        );
        ui.send(
            self.ui_profile.stats_header_text,
            WidgetMessage::Width(header_label_rect.w),
        );
        ui.send(
            self.ui_profile.stats_header_text,
            WidgetMessage::Height(header_label_rect.h),
        );
        ui.send(
            self.ui_profile.stats_header_text,
            TextMessage::Text("整體統計數據".to_string()),
        );
        let collapse_rect = UiRect {
            x: stats_header.right() - stats_header.h,
            y: stats_header.y,
            w: stats_header.h,
            h: stats_header.h,
        };
        ui.send(
            self.ui_profile.stats_collapse_text,
            WidgetMessage::DesiredPosition(collapse_rect.pos()),
        );
        ui.send(
            self.ui_profile.stats_collapse_text,
            WidgetMessage::Width(collapse_rect.w),
        );
        ui.send(
            self.ui_profile.stats_collapse_text,
            WidgetMessage::Height(collapse_rect.h),
        );
        ui.send(
            self.ui_profile.stats_collapse_text,
            TextMessage::Text("▾".to_string()),
        );
        let share_rect = UiRect {
            x: collapse_rect.x - 140.0,
            y: stats_header.y,
            w: 130.0,
            h: stats_header.h,
        };
        ui.send(
            self.ui_profile.stats_share_count_text,
            WidgetMessage::DesiredPosition(share_rect.pos()),
        );
        ui.send(
            self.ui_profile.stats_share_count_text,
            WidgetMessage::Width(share_rect.w),
        );
        ui.send(
            self.ui_profile.stats_share_count_text,
            WidgetMessage::Height(share_rect.h),
        );
        ui.send(
            self.ui_profile.stats_share_count_text,
            TextMessage::Text(format!(
                "分享 {}/{}",
                PROFILE_SHARE_COUNT.0, PROFILE_SHARE_COUNT.1
            )),
        );

        let row_h = 48.0;
        let row_gap = 4.0;
        // rows_y 必須用 ref-space（跟 lower_y 一致）；先前誤用 stats_header.bottom()
        // 那是 screen-space，再丟進 pregame_ref_rect 會被二次縮放、把清單推歪。
        let rows_y = lower_y + 50.0 + 12.0;
        for (i, ((label, value), row_ui)) in PROFILE_STAT_ROWS
            .iter()
            .zip(self.ui_profile.stats_rows.iter())
            .enumerate()
        {
            let ry = rows_y + i as f32 * (row_h + row_gap);
            let row_rect = pregame_ref_rect(ws, right_col_x, ry, right_col_w, row_h);
            // 勾選框移到每列最右邊（對齊原圖 BTD6：標籤左、數值中右、勾選框最右）。
            let cb_size = row_rect.h * 0.5;
            let checkbox = UiRect {
                x: row_rect.right() - cb_size - 8.0,
                y: row_rect.y + row_rect.h * 0.25,
                w: cb_size,
                h: cb_size,
            };
            ui.send(
                row_ui.checkbox_bg,
                WidgetMessage::DesiredPosition(checkbox.pos()),
            );
            ui.send(row_ui.checkbox_bg, WidgetMessage::Width(checkbox.w));
            ui.send(row_ui.checkbox_bg, WidgetMessage::Height(checkbox.h));

            let label_rect = UiRect {
                x: row_rect.x + 14.0,
                y: row_rect.y,
                w: row_rect.w * 0.55,
                h: row_rect.h,
            };
            ui.send(
                row_ui.label_text,
                WidgetMessage::DesiredPosition(label_rect.pos()),
            );
            ui.send(row_ui.label_text, WidgetMessage::Width(label_rect.w));
            ui.send(row_ui.label_text, WidgetMessage::Height(label_rect.h));
            ui.send(row_ui.label_text, TextMessage::Text(label.to_string()));

            // 數值靠右對齊，落在勾選框左側（value_text 已設 HorizontalAlignment::Right）。
            let value_w = row_rect.w * 0.2;
            let value_rect = UiRect {
                x: checkbox.x - value_w - 12.0,
                y: row_rect.y,
                w: value_w,
                h: row_rect.h,
            };
            ui.send(
                row_ui.value_text,
                WidgetMessage::DesiredPosition(value_rect.pos()),
            );
            ui.send(row_ui.value_text, WidgetMessage::Width(value_rect.w));
            ui.send(row_ui.value_text, WidgetMessage::Height(value_rect.h));
            ui.send(row_ui.value_text, TextMessage::Text(value.to_string()));
        }
    }

    fn clear_td_tower_shop_cards(&mut self, ui: &mut UserInterface) {
        for card in self.ui_td_tower_cards.drain(..) {
            for node in [card.bg, card.icon] {
                ui.send(node, WidgetMessage::Remove);
            }
            for text in [card.key_text, card.name_text, card.price_text] {
                ui.send(text, WidgetMessage::Remove);
            }
        }
        self.td_tower_button_rects.clear();
    }

    fn hide_gameplay_ui_for_pregame(&mut self, ui: &mut UserInterface) {
        ui.send(self.ui_status_text, TextMessage::Text(String::new()));
        ui.send(self.ui_hud_text, TextMessage::Text(String::new()));
        ui.send(self.ui_shop_text, TextMessage::Text(String::new()));
        ui.send(self.ui_end_text, TextMessage::Text(String::new()));
        ui.send(self.ui_hero_stats_panel, TextMessage::Text(String::new()));
        self.hide_in_game_return_button(ui);
        ui.send(
            self.ui_td_auto_start_checkbox_text,
            WidgetMessage::DesiredPosition(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS)),
        );
        for handle in self.ui_ability_icons.iter().copied() {
            ui.send(
                handle,
                WidgetMessage::DesiredPosition(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS)),
            );
        }
        for handle in self.ui_ability_level_text.iter().copied() {
            ui.send(
                handle,
                WidgetMessage::DesiredPosition(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS)),
            );
        }
        for handle in self.ui_ability_key_text.iter().copied() {
            ui.send(
                handle,
                WidgetMessage::DesiredPosition(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS)),
            );
        }
        for handle in self.ui_ability_cd_text.iter().copied() {
            ui.send(
                handle,
                WidgetMessage::DesiredPosition(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS)),
            );
        }
        for handle in self.ui_ability_upgrade_buttons.iter().copied() {
            ui.send(
                handle,
                WidgetMessage::DesiredPosition(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS)),
            );
        }
        self.start_round_button_rect = (UI_HIDDEN_POS, UI_HIDDEN_POS, 0.0, 0.0);
        self.auto_start_checkbox_rect = (UI_HIDDEN_POS, UI_HIDDEN_POS, 0.0, 0.0);
        self.pause_button_rect = (UI_HIDDEN_POS, UI_HIDDEN_POS, 0.0, 0.0);
        self.clear_td_tower_shop_cards(ui);
        self.td_sell_button_rect = (UI_HIDDEN_POS, UI_HIDDEN_POS, 0.0, 0.0);
        self.td_target_priority_button_rect = (UI_HIDDEN_POS, UI_HIDDEN_POS, 0.0, 0.0);
        self.td_upgrade_button_rects = [(UI_HIDDEN_POS, UI_HIDDEN_POS, 0.0, 0.0); 3];
    }

    fn update_in_game_return_button(&mut self, ui: &mut UserInterface) {
        let scale = (self.window_size.y.max(1.0) / 720.0).clamp(0.85, 1.25);
        let rect = UiRect {
            x: 16.0 * scale,
            y: 16.0 * scale,
            w: 104.0 * scale,
            h: 42.0 * scale,
        };
        ui.send(
            self.ui_in_game_return.bg,
            WidgetMessage::DesiredPosition(rect.pos()),
        );
        ui.send(self.ui_in_game_return.bg, WidgetMessage::Width(rect.w));
        ui.send(self.ui_in_game_return.bg, WidgetMessage::Height(rect.h));
        ui.send(
            self.ui_in_game_return.text,
            WidgetMessage::DesiredPosition(rect.pos()),
        );
        ui.send(self.ui_in_game_return.text, WidgetMessage::Width(rect.w));
        ui.send(self.ui_in_game_return.text, WidgetMessage::Height(rect.h));
        ui.send(
            self.ui_in_game_return.text,
            TextMessage::Text("返回".to_string()),
        );
        self.return_to_title_button_rect = rect;
    }

    fn hide_in_game_return_button(&mut self, ui: &mut UserInterface) {
        for handle in [
            self.ui_in_game_return.bg,
            self.ui_in_game_return.text.transmute(),
        ] {
            ui.send(
                handle,
                WidgetMessage::DesiredPosition(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS)),
            );
        }
        ui.send(
            self.ui_in_game_return.text,
            TextMessage::Text(String::new()),
        );
        self.return_to_title_button_rect = UiRect {
            x: UI_HIDDEN_POS,
            y: UI_HIDDEN_POS,
            w: 0.0,
            h: 0.0,
        };
    }

    fn hide_pregame_ui(&mut self, ui: &mut UserInterface) {
        for handle in [self.ui_pregame.background, self.ui_pregame.panel] {
            ui.send(
                handle,
                WidgetMessage::DesiredPosition(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS)),
            );
        }
        for handle in [
            self.ui_pregame.title,
            self.ui_pregame.subtitle,
            self.ui_pregame.status,
        ] {
            ui.send(
                handle,
                WidgetMessage::DesiredPosition(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS)),
            );
        }
        for button in &self.ui_pregame.buttons {
            ui.send(
                button.bg,
                WidgetMessage::DesiredPosition(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS)),
            );
            ui.send(
                button.image,
                WidgetMessage::DesiredPosition(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS)),
            );
            ui.send(button.image, ImageMessage::Texture(None));
            ui.send(
                button.text,
                WidgetMessage::DesiredPosition(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS)),
            );
        }
        self.pregame_button_rects.clear();
    }

    fn selected_barrel_variant(
        &self,
        tpl: &TdTemplate,
        upgrade_levels: [u8; 3],
    ) -> Option<sim_runner::TowerBarrelVariantSnapshot> {
        if tpl.barrel_layout != "radial_count_variants" {
            return None;
        }
        tpl.barrel_variants
            .iter()
            .filter(|variant| {
                let path_idx = variant.min_path.saturating_sub(1) as usize;
                let level = upgrade_levels.get(path_idx).copied().unwrap_or(0);
                level >= variant.min_level
            })
            .max_by_key(|variant| (variant.min_level, variant.count))
            .cloned()
    }

    fn play_sfx(&mut self, scene: &mut Scene, kind: SfxKind) {
        let buf = match kind {
            SfxKind::ButtonClick => self.sfx_button_click.clone(),
            SfxKind::TowerPlace => self.sfx_tower_place.clone(),
            SfxKind::CookieCrunch => self.sfx_cookie_crunch.clone(),
        };
        let Some(buf) = buf else {
            return;
        };
        let gain = self.settings_sfx_volume;
        if gain <= 0.001 {
            return;
        }
        let node = SoundBuilder::new(BaseBuilder::new())
            .with_buffer(Some(buf))
            .with_looping(false)
            .with_status(Status::Playing)
            .with_gain(gain)
            .build_node();
        let h = scene.graph.add_node(node);
        self.sfx_one_shots.push(h);
    }

    fn cleanup_sfx_one_shots(&mut self, scene: &mut Scene) {
        use fyrox::scene::sound::Sound;
        let finished: Vec<Handle<Node>> = self
            .sfx_one_shots
            .iter()
            .copied()
            .filter(|&h| {
                scene
                    .graph
                    .try_get(h)
                    .ok()
                    .and_then(|n| n.cast::<Sound>())
                    .map(|s| s.status() == Status::Stopped)
                    .unwrap_or(true)
            })
            .collect();
        for h in &finished {
            scene.graph.remove_node(*h);
        }
        self.sfx_one_shots.retain(|h| !finished.contains(h));
    }

    fn remove_tower_composite(&mut self, scene: &mut Scene, entity_id: u32) {
        if let Some(comp) = self.tower_composites.remove(&entity_id) {
            if let Some(barrel) = comp.barrel_node {
                if barrel != comp.base_node {
                    scene.graph.remove_node(barrel);
                }
            }
            if let Some(body) = comp.body_node {
                if body != comp.base_node {
                    scene.graph.remove_node(body);
                }
            }
            scene.graph.remove_node(comp.base_node);
        }
    }

    fn remove_hero_model(&mut self, scene: &mut Scene, entity_id: u32) {
        if let Some(render) = self.hero_model_nodes.remove(&entity_id) {
            if scene.graph.is_valid_handle(render.root_node) {
                scene.graph.remove_node(render.root_node);
            }
        }
    }

    fn hero_muzzle_render_pos(&self, scene: &Scene, entity_id: u32) -> Option<Vector2<f32>> {
        let render = self.hero_model_nodes.get(&entity_id)?;
        let handle = render.muzzle_node?;
        if !scene.graph.is_valid_handle(handle) {
            return None;
        }
        let pos = scene.graph[handle].global_position();
        Some(Vector2::new(pos.x, pos.y))
    }

    fn projectile_owner_spawn_and_dir(
        &self,
        scene: &Scene,
        entities: &[sim_runner::EntityRenderData],
        owner_id: u32,
    ) -> Option<(Vector2<f32>, Vector2<f32>)> {
        let owner = entities.iter().find(|e| e.entity_id == owner_id)?;
        let fallback_dir = tower_render_dir_from_world_rad(owner.facing_rad);
        if let Some(pos) = self.hero_muzzle_render_pos(scene, owner_id) {
            return Some((pos, fallback_dir));
        }
        if matches!(owner.kind, sim_runner::EntityKind::Tower) {
            if let Some(tpl) = self.td_templates.get(&owner.unit_id) {
                let owner_pos = render_bridge::world_to_render(owner);
                let aim_angle =
                    tower_render_angle_from_facing(owner.facing_rad, tpl.default_angle_deg);
                let muzzle_pos = owner_pos
                    + rotate_vec2(tower_render_offset(&tpl.muzzle_offset, 1.0), aim_angle);
                return Some((muzzle_pos, fallback_dir));
            }
        }
        Some((render_bridge::world_to_render(owner), fallback_dir))
    }

    fn request_hero_model_asset(
        &mut self,
        resource_manager: &ResourceManager,
        model_path: &str,
        texture_path: &str,
        action_source: bool,
    ) -> Option<ModelResource> {
        let cache = if action_source {
            &mut self.hero_action_assets
        } else {
            &mut self.hero_model_assets
        };
        if !cache.contains_key(model_path) {
            let Some(resolved_model_path) = resolve_scripts_lua_data_asset_path(model_path) else {
                if self
                    .hero_asset_failures_logged
                    .insert(format!("model:{model_path}"))
                {
                    log::warn!("hero 3D model asset not found: {}", model_path);
                }
                return None;
            };
            let texture_path = (!texture_path.trim().is_empty())
                .then(|| resolve_scripts_lua_data_asset_path(texture_path))
                .flatten();
            let model = resource_manager.request::<Model>(&resolved_model_path);
            cache.insert(
                model_path.to_string(),
                HeroModelAsset {
                    model,
                    resolved_model_path,
                    texture_path,
                    failed_logged: false,
                },
            );
        }

        let asset = cache.get_mut(model_path)?;
        if asset.model.is_failed_to_load() {
            if !asset.failed_logged {
                log::warn!(
                    "hero 3D model failed to load: {} ({})",
                    model_path,
                    asset.resolved_model_path.display()
                );
                asset.failed_logged = true;
            }
            return None;
        }
        if asset.model.is_ok() {
            Some(asset.model.clone())
        } else {
            None
        }
    }

    fn apply_hero_texture_fallback(
        &mut self,
        scene: &mut Scene,
        root: Handle<Node>,
        texture_path: &str,
    ) {
        let Some(texture) = load_scripts_lua_data_texture(texture_path) else {
            if self
                .hero_asset_failures_logged
                .insert(format!("texture:{texture_path}"))
            {
                log::warn!(
                    "hero 3D texture asset not found or failed to decode: {}",
                    texture_path
                );
            }
            return;
        };
        let material = texture_material_3d(texture);
        let mut stack = vec![root];
        while let Some(handle) = stack.pop() {
            if !scene.graph.is_valid_handle(handle) {
                continue;
            }
            let children = scene.graph[handle].children().to_vec();
            stack.extend(children);
            if let Some(mesh) = scene.graph[handle].cast_mut::<Mesh>() {
                for surface in mesh.surfaces_mut() {
                    surface.set_material(material.clone());
                }
            }
        }
    }

    fn retarget_hero_action_sources(
        &mut self,
        scene: &mut Scene,
        resource_manager: &ResourceManager,
        entity_id: u32,
        render: &sim_runner::HeroRenderSnapshot,
    ) {
        let Some(node) = self.hero_model_nodes.get(&entity_id) else {
            return;
        };
        let missing: Vec<_> = render
            .animation_sources
            .iter()
            .filter(|source| !node.animations_by_source.contains_key(&source.key))
            .cloned()
            .collect();
        let root = node.root_node;
        let player = node.animation_player;
        let _ = node;

        for source in missing {
            let Some(model) =
                self.request_hero_model_asset(resource_manager, &source.model, "", true)
            else {
                continue;
            };
            let handles = model.retarget_animations_to_player(root, player, &mut scene.graph);
            let expected_duration_secs = source.duration_ticks / source.ticks_per_second.max(0.001);
            if let Some(handle) = select_retargeted_animation_by_duration(
                scene,
                player,
                &handles,
                expected_duration_secs,
            ) {
                if let Some(node) = self.hero_model_nodes.get_mut(&entity_id) {
                    node.animations_by_source.insert(source.key.clone(), handle);
                    node.action_resources_requested.insert(source.key.clone());
                }
            } else if self
                .hero_asset_failures_logged
                .insert(format!("animation:{}:{}", source.key, source.model))
            {
                log::warn!(
                    "hero 3D animation source '{}' produced no usable animations from {}",
                    source.key,
                    source.model
                );
            }
        }
    }

    fn play_hero_animation_ticks(
        &mut self,
        scene: &mut Scene,
        entity_id: u32,
        render: &sim_runner::HeroRenderSnapshot,
        action: &str,
        start_tick: f32,
        end_tick: f32,
        loop_animation: bool,
        desired_duration_secs: Option<f32>,
    ) -> bool {
        let Some(node) = self.hero_model_nodes.get_mut(&entity_id) else {
            return false;
        };
        let Some(binding) = render.animations.iter().find(|b| b.action == action) else {
            return false;
        };
        let Some(source) = render
            .animation_sources
            .iter()
            .find(|source| source.key == binding.source)
        else {
            return false;
        };
        let Some(handle) = node.animations_by_source.get(&binding.source).copied() else {
            return false;
        };
        disable_other_animation_players(scene, node.root_node, node.animation_player);
        let Some(player) = scene.graph[node.animation_player]
            .cast_mut::<fyrox::scene::animation::AnimationPlayer>()
        else {
            return false;
        };
        let animations = player.animations_mut().get_value_mut_silent();
        for animation in animations.iter_mut() {
            animation.set_enabled(false);
        }
        let ticks_per_second = source.ticks_per_second.max(0.001);
        let start_sec = (source.timeline_offset_ticks + start_tick) / ticks_per_second;
        let end_sec = (source.timeline_offset_ticks + end_tick) / ticks_per_second;
        let source_duration = (end_sec - start_sec).max(0.001);
        let speed = desired_duration_secs
            .filter(|duration| *duration > 0.001)
            .map(|duration| source_duration / duration)
            .unwrap_or(1.0);
        let animation = animations.get_mut(handle);
        animation.set_time_slice(start_sec..end_sec);
        animation.set_time_position(start_sec);
        animation.set_loop(loop_animation);
        animation.set_speed(speed);
        animation.set_enabled(true);
        node.active_action = Some(action.to_string());
        node.active_animation_speed = speed;
        node.one_shot_remaining = desired_duration_secs.unwrap_or(0.0);
        node.idle_cycle_remaining = if is_hero_idle_action(action) {
            source_duration / speed.max(0.001)
        } else {
            0.0
        };
        true
    }

    fn play_hero_action(
        &mut self,
        scene: &mut Scene,
        entity_id: u32,
        render: &sim_runner::HeroRenderSnapshot,
        action: &str,
    ) {
        if self
            .hero_model_nodes
            .get(&entity_id)
            .and_then(|node| node.active_attack_seq)
            .is_some()
        {
            return;
        }
        if self
            .hero_model_nodes
            .get(&entity_id)
            .map(|node| node.active_action.as_deref() == Some(action))
            .unwrap_or(false)
        {
            return;
        }
        let Some(binding) = render.animations.iter().find(|b| b.action == action) else {
            return;
        };
        if self.play_hero_animation_ticks(
            scene,
            entity_id,
            render,
            action,
            binding.start_tick,
            binding.end_tick,
            binding.loop_animation,
            None,
        ) {
            if let Some(node) = self.hero_model_nodes.get_mut(&entity_id) {
                node.active_attack_seq = None;
                if action == "move" {
                    node.last_attack_action = None;
                    node.attack_repeat_ready = false;
                }
                node.attack_phase = HeroAttackPlaybackPhase::None;
                node.attack_phase_remaining = 0.0;
                node.attack_backswing_remaining = 0.0;
            }
        }
    }

    fn choose_hero_idle_action(
        &mut self,
        entity_id: u32,
        render: &sim_runner::HeroRenderSnapshot,
    ) -> String {
        let candidates: Vec<&str> = render
            .animations
            .iter()
            .filter(|binding| is_hero_idle_action(&binding.action) && binding.loop_animation)
            .map(|binding| binding.action.as_str())
            .collect();
        if candidates.is_empty() {
            return "sniper".to_string();
        }

        let current_idle = self.hero_model_nodes.get(&entity_id).and_then(|node| {
            node.active_action
                .as_deref()
                .filter(|action| is_hero_idle_action(action))
                .map(|action| (action.to_string(), node.idle_cycle_remaining))
        });
        if let Some((current, remaining)) = current_idle.as_ref() {
            if *remaining > 0.0 && candidates.iter().any(|candidate| *candidate == current) {
                return current.clone();
            }
        }

        let Some(node) = self.hero_model_nodes.get_mut(&entity_id) else {
            return candidates[0].to_string();
        };
        node.idle_rng_state = next_idle_rng_state(node.idle_rng_state);
        let mut index = (node.idle_rng_state as usize) % candidates.len();
        if candidates.len() > 1 {
            if let Some((current, _)) = current_idle.as_ref() {
                if candidates[index] == current {
                    index = (index + 1) % candidates.len();
                }
            }
        }
        candidates[index].to_string()
    }

    fn start_hero_attack_action(
        &mut self,
        scene: &mut Scene,
        entity_id: u32,
        render: &sim_runner::HeroRenderSnapshot,
        action: &str,
        cue: &sim_runner::AttackPhaseFx,
        cue_age_secs: f32,
    ) -> bool {
        if self
            .hero_model_nodes
            .get(&entity_id)
            .map(|node| node.active_attack_seq == Some(cue.attack_seq))
            .unwrap_or(false)
        {
            return true;
        }
        let Some(binding) = render.animations.iter().find(|b| b.action == action) else {
            return true;
        };
        let Some(impact_tick) = binding.impact_tick else {
            self.play_hero_action(scene, entity_id, render, action);
            return true;
        };
        let windup_secs = (cue.windup_ms as f32 / 1000.0).max(0.001);
        let backswing_secs = (cue.backswing_ms as f32 / 1000.0).max(0.001);
        let cue_age_secs = cue_age_secs.max(0.0);
        let repeated_attack = self
            .hero_model_nodes
            .get(&entity_id)
            .map(|node| node.attack_repeat_ready && binding.repeat_start_tick > binding.start_tick)
            .unwrap_or(false);
        let visual_start_tick = if repeated_attack {
            binding.repeat_start_tick
        } else {
            binding.start_tick
        };
        let (phase, start_tick, end_tick, duration_secs, backswing_remaining) =
            if cue_age_secs < windup_secs {
                let t = (cue_age_secs / windup_secs).clamp(0.0, 1.0);
                let start_tick = visual_start_tick + (impact_tick - visual_start_tick) * t;
                (
                    HeroAttackPlaybackPhase::Windup,
                    start_tick,
                    impact_tick,
                    (windup_secs - cue_age_secs).max(0.001),
                    backswing_secs,
                )
            } else {
                let backswing_age = cue_age_secs - windup_secs;
                if backswing_age >= backswing_secs {
                    return true;
                }
                let t = (backswing_age / backswing_secs).clamp(0.0, 1.0);
                let start_tick = impact_tick + (binding.end_tick - impact_tick) * t;
                (
                    HeroAttackPlaybackPhase::Backswing,
                    start_tick,
                    binding.end_tick,
                    (backswing_secs - backswing_age).max(0.001),
                    0.0,
                )
            };
        if self.play_hero_animation_ticks(
            scene,
            entity_id,
            render,
            action,
            start_tick,
            end_tick,
            false,
            Some(duration_secs),
        ) {
            if let Some(node) = self.hero_model_nodes.get_mut(&entity_id) {
                node.active_attack_seq = Some(cue.attack_seq);
                node.last_attack_action = Some(action.to_string());
                node.attack_repeat_ready = true;
                node.pending_attack = None;
                node.attack_phase = phase;
                node.attack_phase_remaining = duration_secs;
                node.attack_backswing_remaining = backswing_remaining;
            }
            true
        } else {
            false
        }
    }

    fn tick_hero_attack_action(
        &mut self,
        scene: &mut Scene,
        entity_id: u32,
        render: &sim_runner::HeroRenderSnapshot,
        dt: f32,
    ) {
        let transition = if let Some(node) = self.hero_model_nodes.get_mut(&entity_id) {
            match node.attack_phase {
                HeroAttackPlaybackPhase::Windup => {
                    node.attack_phase_remaining = (node.attack_phase_remaining - dt).max(0.0);
                    if node.attack_phase_remaining <= 0.0 {
                        node.active_action
                            .clone()
                            .map(|action| (action, node.attack_backswing_remaining))
                    } else {
                        None
                    }
                }
                HeroAttackPlaybackPhase::Backswing => {
                    node.attack_phase_remaining = (node.attack_phase_remaining - dt).max(0.0);
                    if node.attack_phase_remaining <= 0.0 {
                        node.active_attack_seq = None;
                        node.attack_phase = HeroAttackPlaybackPhase::None;
                        node.attack_backswing_remaining = 0.0;
                        node.active_action = None;
                    }
                    None
                }
                HeroAttackPlaybackPhase::None => None,
            }
        } else {
            None
        };
        let Some((action, backswing_secs)) = transition else {
            return;
        };
        let Some(binding) = render.animations.iter().find(|b| b.action == action) else {
            return;
        };
        let Some(impact_tick) = binding.impact_tick else {
            return;
        };
        if self.play_hero_animation_ticks(
            scene,
            entity_id,
            render,
            &action,
            impact_tick,
            binding.end_tick,
            false,
            Some(backswing_secs),
        ) {
            if let Some(node) = self.hero_model_nodes.get_mut(&entity_id) {
                node.attack_phase = HeroAttackPlaybackPhase::Backswing;
                node.attack_phase_remaining = backswing_secs;
                node.attack_backswing_remaining = 0.0;
            }
        }
    }

    fn cancel_hero_attack_action(
        &mut self,
        scene: &mut Scene,
        entity_id: u32,
        cue: &sim_runner::AttackCancelFx,
    ) {
        if let Some(node) = self.hero_model_nodes.get_mut(&entity_id) {
            if node
                .pending_attack
                .as_ref()
                .map(|pending| pending.cue.attack_seq == cue.attack_seq)
                .unwrap_or(false)
            {
                node.pending_attack = None;
            }
        }
        let should_stop = self
            .hero_model_nodes
            .get(&entity_id)
            .map(|node| node.active_attack_seq == Some(cue.attack_seq))
            .unwrap_or(false);
        if should_stop {
            self.stop_hero_action(scene, entity_id);
            if let Some(node) = self.hero_model_nodes.get_mut(&entity_id) {
                node.attack_repeat_ready = cue.impact_committed;
            }
        }
    }

    fn stop_hero_action(&mut self, scene: &mut Scene, entity_id: u32) {
        let Some(node) = self.hero_model_nodes.get_mut(&entity_id) else {
            return;
        };
        if let Some(player) = scene.graph[node.animation_player]
            .cast_mut::<fyrox::scene::animation::AnimationPlayer>()
        {
            for animation in player.animations_mut().get_value_mut_silent().iter_mut() {
                animation.set_enabled(false);
            }
        }
        node.active_action = None;
        node.active_animation_speed = 1.0;
        node.active_attack_seq = None;
        node.last_attack_action = None;
        node.attack_repeat_ready = false;
        node.pending_attack = None;
        node.idle_cycle_remaining = 0.0;
        node.attack_phase = HeroAttackPlaybackPhase::None;
        node.attack_phase_remaining = 0.0;
        node.attack_backswing_remaining = 0.0;
        node.one_shot_remaining = 0.0;
    }

    fn update_hero_model(
        &mut self,
        scene: &mut Scene,
        resource_manager: &ResourceManager,
        entity: &sim_runner::EntityRenderData,
        render: &sim_runner::HeroRenderSnapshot,
        pos: Vector2<f32>,
        snapshot_tick: u32,
        dt: f32,
        attack_cue: Option<&sim_runner::AttackPhaseFx>,
        cancel_cue: Option<&sim_runner::AttackCancelFx>,
    ) -> bool {
        if render.render_mode != "model_3d" {
            return false;
        }
        let Some(model) =
            self.request_hero_model_asset(resource_manager, &render.model, &render.texture, false)
        else {
            return false;
        };
        if !self.hero_model_nodes.contains_key(&entity.entity_id) {
            let rotation = hero_model_rotation(entity.facing_rad, render);
            let root = model
                .begin_instantiation(scene)
                .with_position(Vector3::new(pos.x, pos.y, Z_HERO + render.z_offset))
                .with_rotation(rotation)
                .with_scale(Vector3::new(render.scale, render.scale, render.scale))
                .finish();
            let player = find_descendant_animation_player(scene, root).unwrap_or_else(|| {
                let player = AnimationPlayerBuilder::new(
                    BaseBuilder::new().with_name("Hero Animation Player"),
                )
                .build(&mut scene.graph);
                scene.graph.link_nodes(player, root);
                player
            });
            self.apply_hero_texture_fallback(scene, root, &render.texture);
            let muzzle_node = find_descendant_by_name(scene, root, &render.muzzle_bone);
            if !render.muzzle_bone.trim().is_empty() && muzzle_node.is_none() {
                let key = format!("muzzle:{}:{}", render.model, render.muzzle_bone);
                if self.hero_asset_failures_logged.insert(key) {
                    log::warn!(
                        "hero 3D muzzle bone '{}' not found in model {}",
                        render.muzzle_bone,
                        render.model
                    );
                }
            }
            self.hero_model_nodes.insert(
                entity.entity_id,
                HeroModelRender {
                    root_node: root,
                    animation_player: player,
                    muzzle_node,
                    last_pos: pos,
                    render_moving: render.is_moving,
                    animations_by_source: HashMap::new(),
                    action_resources_requested: HashSet::new(),
                    active_action: None,
                    active_animation_speed: 1.0,
                    active_attack_seq: None,
                    last_attack_action: None,
                    attack_repeat_ready: false,
                    pending_attack: None,
                    idle_cycle_remaining: 0.0,
                    idle_rng_state: entity
                        .entity_id
                        .wrapping_mul(747_796_405)
                        .wrapping_add(snapshot_tick),
                    attack_phase: HeroAttackPlaybackPhase::None,
                    attack_phase_remaining: 0.0,
                    attack_backswing_remaining: 0.0,
                    one_shot_remaining: 0.0,
                    texture_applied: true,
                },
            );
        }

        let model_dt = if self.is_game_paused { 0.0 } else { dt };
        if let Some(node) = self.hero_model_nodes.get_mut(&entity.entity_id) {
            if scene.graph.is_valid_handle(node.root_node) {
                let dx = pos.x - node.last_pos.x;
                let dy = pos.y - node.last_pos.y;
                node.render_moving = render.is_moving || dx * dx + dy * dy > 0.000001;
                node.last_pos = pos;
                let rotation = hero_model_rotation(entity.facing_rad, render);
                scene.graph[node.root_node]
                    .local_transform_mut()
                    .set_position(Vector3::new(pos.x, pos.y, Z_HERO + render.z_offset))
                    .set_rotation(rotation)
                    .set_scale(Vector3::new(render.scale, render.scale, render.scale));
                node.one_shot_remaining = (node.one_shot_remaining - model_dt).max(0.0);
                if node
                    .active_action
                    .as_deref()
                    .map(is_hero_idle_action)
                    .unwrap_or(false)
                {
                    node.idle_cycle_remaining = (node.idle_cycle_remaining - model_dt).max(0.0);
                } else {
                    node.idle_cycle_remaining = 0.0;
                }
            }
        }

        self.retarget_hero_action_sources(scene, resource_manager, entity.entity_id, render);
        self.tick_hero_attack_action(scene, entity.entity_id, render, model_dt);

        if let Some(cue) = cancel_cue {
            self.cancel_hero_attack_action(scene, entity.entity_id, cue);
        } else if let Some(cue) = attack_cue {
            let action = if cue.is_critical {
                "critical"
            } else {
                "attack"
            };
            let cue_age_secs =
                self.ticks_to_seconds(snapshot_tick.saturating_sub(cue.spawn_tick)) as f32;
            if !self.start_hero_attack_action(
                scene,
                entity.entity_id,
                render,
                action,
                cue,
                cue_age_secs,
            ) {
                if let Some(node) = self.hero_model_nodes.get_mut(&entity.entity_id) {
                    node.pending_attack = Some(HeroPendingAttackCue {
                        cue: cue.clone(),
                        action: action.to_string(),
                    });
                }
            }
        } else {
            let pending = self
                .hero_model_nodes
                .get(&entity.entity_id)
                .and_then(|node| node.pending_attack.clone());
            if let Some(pending) = pending {
                let cue_age_secs = self
                    .ticks_to_seconds(snapshot_tick.saturating_sub(pending.cue.spawn_tick))
                    as f32;
                if self.start_hero_attack_action(
                    scene,
                    entity.entity_id,
                    render,
                    &pending.action,
                    &pending.cue,
                    cue_age_secs,
                ) {
                    if let Some(node) = self.hero_model_nodes.get_mut(&entity.entity_id) {
                        node.pending_attack = None;
                    }
                }
            }
            let keep_one_shot = self
                .hero_model_nodes
                .get(&entity.entity_id)
                .map(|node| node.active_attack_seq.is_some() || node.one_shot_remaining > 0.0)
                .unwrap_or(false);
            if !keep_one_shot {
                let render_moving = self
                    .hero_model_nodes
                    .get(&entity.entity_id)
                    .map(|node| node.render_moving)
                    .unwrap_or(render.is_moving);
                let action = if render_moving {
                    "move".to_string()
                } else if render.sniper_mode {
                    "sniper".to_string()
                } else {
                    self.choose_hero_idle_action(entity.entity_id, render)
                };
                self.play_hero_action(scene, entity.entity_id, render, &action);
            }
        }
        self.apply_hero_animation_pause_state(scene, entity.entity_id);

        true
    }

    fn apply_hero_animation_pause_state(&mut self, scene: &mut Scene, entity_id: u32) {
        let Some(node) = self.hero_model_nodes.get(&entity_id) else {
            return;
        };
        let speed = hero_animation_playback_speed(node.active_animation_speed, self.is_game_paused);
        if let Some(player) = scene.graph[node.animation_player]
            .cast_mut::<fyrox::scene::animation::AnimationPlayer>()
        {
            for animation in player.animations_mut().get_value_mut_silent().iter_mut() {
                if animation.is_enabled() {
                    animation.set_speed(speed);
                }
            }
        }
    }

    fn start_tower_animation(
        &mut self,
        entity_id: u32,
        frames: Vec<String>,
        animation: &sim_runner::TowerRenderAnimationSnapshot,
        impact_at_ms: u32,
    ) {
        if frames.is_empty() {
            return;
        }
        let impact_secs = impact_at_ms as f32 / 1000.0;
        let fps = if frames.len() > 1 && impact_secs > 0.0 {
            ((frames.len() - 1) as f32 / impact_secs).max(1.0)
        } else {
            animation.fire_fps.max(animation.fps).max(1.0)
        };
        if let Some(comp) = self.tower_composites.get_mut(&entity_id) {
            comp.animation = Some(TowerAnimationState {
                frames,
                elapsed: 0.0,
                fps,
                fire_once: animation.fire_once,
                active: true,
                last_frame_index: usize::MAX,
            });
        }
    }

    fn update_tower_composite(
        &mut self,
        scene: &mut Scene,
        entity: &sim_runner::EntityRenderData,
        tpl: &TdTemplate,
        pos: Vector2<f32>,
        dt: f32,
        attack_cue: Option<&sim_runner::AttackPhaseFx>,
        fire_cue: Option<&sim_runner::TowerFireFx>,
    ) {
        let animation_dt = tower_animation_dt(dt, self.is_game_paused);
        let upgrade_levels = entity.upgrade_levels.unwrap_or([0; 3]);
        let selected_variant = self.selected_barrel_variant(tpl, upgrade_levels);
        let is_animated_area = tpl.render_mode == "animated_area";
        let base_key = normalize_tower_asset_key(if is_animated_area {
            tpl.body_frames
                .first()
                .map(String::as_str)
                .filter(|s| !s.trim().is_empty())
                .unwrap_or(&tpl.base_image)
        } else {
            &tpl.base_image
        });
        let barrel_key = normalize_tower_asset_key(
            selected_variant
                .as_ref()
                .map(|v| v.image.as_str())
                .filter(|s| !s.trim().is_empty())
                .unwrap_or(&tpl.barrel_image),
        );
        let active_frames: Vec<String> = if is_animated_area {
            if tpl.body_frames.is_empty() {
                vec![base_key.clone()]
            } else {
                tpl.body_frames.clone()
            }
        } else if let Some(variant) = selected_variant.as_ref() {
            if variant.frames.is_empty() {
                vec![barrel_key.clone()]
            } else {
                variant.frames.clone()
            }
        } else if tpl.barrel_frames.is_empty() {
            vec![barrel_key.clone()]
        } else {
            tpl.barrel_frames.clone()
        };
        let animation_meta = if is_animated_area {
            &tpl.body_animation
        } else {
            &tpl.barrel_animation
        };
        let base_size = tower_visual_size(tpl);
        let base_material = self.tower_material_for_key(&base_key);
        let barrel_material = (!is_animated_area)
            .then(|| self.tower_material_for_key(&barrel_key))
            .flatten();
        let fallback_color = Color::from_rgba(120, 120, 255, 255);
        if !self.tower_composites.contains_key(&entity.entity_id) {
            let base_node = build_tower_rect_node(
                scene,
                base_material.clone(),
                pos,
                base_size,
                Z_TOWER,
                fallback_color,
            );
            let barrel_node = if is_animated_area {
                None
            } else {
                Some(build_tower_rect_node(
                    scene,
                    barrel_material.clone(),
                    pos,
                    base_size,
                    Z_TOWER - 0.04,
                    Color::from_rgba(35, 35, 35, 255),
                ))
            };
            self.tower_composites.insert(
                entity.entity_id,
                TowerCompositeRender {
                    base_node,
                    barrel_node,
                    body_node: None,
                    base_material_key: base_key.clone(),
                    barrel_material_key: (!is_animated_area).then(|| barrel_key.clone()),
                    body_material_key: is_animated_area.then(|| base_key.clone()),
                    variant_count: selected_variant.as_ref().map(|v| v.count),
                    last_aim_direction: tower_render_angle_from_facing(
                        entity.facing_rad,
                        tpl.default_angle_deg,
                    ),
                    animation: None,
                    recoil: None,
                },
            );
        }

        let mut material_updates: Vec<(Handle<Node>, String, Color)> = Vec::new();
        let mut animation_to_start: Option<(Vec<String>, u32)> = None;
        {
            let Some(comp) = self.tower_composites.get_mut(&entity.entity_id) else {
                return;
            };

            if comp.base_material_key != base_key {
                comp.base_material_key = base_key.clone();
                comp.body_material_key = is_animated_area.then(|| base_key.clone());
                material_updates.push((comp.base_node, base_key.clone(), fallback_color));
            }
            if !is_animated_area {
                let variant_count = selected_variant.as_ref().map(|v| v.count);
                if comp.barrel_material_key.as_deref() != Some(barrel_key.as_str())
                    || comp.variant_count != variant_count
                {
                    comp.barrel_material_key = Some(barrel_key.clone());
                    comp.variant_count = variant_count;
                    comp.animation = None;
                    if let Some(barrel) = comp.barrel_node {
                        material_updates.push((
                            barrel,
                            barrel_key.clone(),
                            Color::from_rgba(35, 35, 35, 255),
                        ));
                    }
                }
            }

            if let Some(cue) = attack_cue.filter(|cue| cue.entity_gen == entity.entity_gen) {
                if tpl.rotation_mode != "fixed" && cue.dir_rad.is_finite() {
                    comp.last_aim_direction =
                        tower_render_angle_from_facing(cue.dir_rad, tpl.default_angle_deg);
                }
                animation_to_start = Some((active_frames.clone(), cue.impact_at_ms));
            }
            if let Some(cue) = fire_cue.filter(|cue| cue.entity_gen == entity.entity_gen) {
                comp.recoil = Some(TowerRecoilState {
                    elapsed: 0.0,
                    duration: (tpl.recoil.duration_ms as f32 / 1000.0).max(0.001),
                    return_duration: (tpl.recoil.return_ms as f32 / 1000.0).max(0.001),
                    dir_rad: cue.dir_rad,
                });
                if comp.animation.as_ref().map(|a| !a.active).unwrap_or(true) {
                    animation_to_start = Some((active_frames.clone(), tpl.recoil.duration_ms));
                }
            }
        }

        if let Some((frames, impact_at_ms)) = animation_to_start {
            self.start_tower_animation(entity.entity_id, frames, animation_meta, impact_at_ms);
        }

        let mut post_material_updates: Vec<(Handle<Node>, String, Color)> = Vec::new();
        {
            let Some(comp) = self.tower_composites.get_mut(&entity.entity_id) else {
                return;
            };
            let default_angle = tpl.default_angle_deg.to_radians();
            let aim_angle = if tpl.rotation_mode == "fixed" {
                default_angle
            } else if entity.facing_rad.is_finite() {
                let angle =
                    tower_render_angle_from_facing(entity.facing_rad, tpl.default_angle_deg);
                comp.last_aim_direction = angle;
                angle
            } else {
                comp.last_aim_direction
            };

            // 待機循環動畫已停用：砲管只在攻擊 cue 到達時才播放，
            // 範圍內無敵人時砲管保持靜止。

            if let Some(anim) = comp.animation.as_mut() {
                if !anim.frames.is_empty() {
                    let frame_duration = (1.0 / anim.fps.max(1.0)).max(0.001);
                    let total_duration = frame_duration * anim.frames.len() as f32;
                    let raw_index = (anim.elapsed / frame_duration).floor() as usize;
                    let frame_index = if anim.fire_once {
                        raw_index.min(anim.frames.len().saturating_sub(1))
                    } else {
                        raw_index % anim.frames.len()
                    };
                    if frame_index != anim.last_frame_index {
                        anim.last_frame_index = frame_index;
                        let key = normalize_tower_asset_key(&anim.frames[frame_index]);
                        let target = if is_animated_area {
                            comp.base_node
                        } else {
                            comp.barrel_node.unwrap_or(comp.base_node)
                        };
                        let color = if is_animated_area {
                            fallback_color
                        } else {
                            Color::from_rgba(35, 35, 35, 255)
                        };
                        post_material_updates.push((target, key.clone(), color));
                        if is_animated_area {
                            comp.base_material_key = key.clone();
                            comp.body_material_key = Some(key);
                        }
                        // 注意：barrel_material_key 不在這裡更新——
                        // 它追蹤靜態砲管/variant 圖片鍵，用於偵測 variant 切換，
                        // 若被動畫幀鍵覆蓋，下一 tick 的 variant 檢查會誤判變動，
                        // 並重置 animation = None，中斷動畫。
                    }
                    anim.elapsed += animation_dt;
                    if anim.fire_once && anim.elapsed >= total_duration {
                        comp.animation = None;
                        // fire_once 播完後把砲管材質還原到靜止幀（barrel_image）。
                        if !is_animated_area {
                            let barrel_node = comp.barrel_node.unwrap_or(comp.base_node);
                            post_material_updates.push((
                                barrel_node,
                                barrel_key.clone(),
                                Color::from_rgba(35, 35, 35, 255),
                            ));
                        }
                    }
                }
            }

            let mut recoil_offset = Vector2::new(0.0, 0.0);
            let mut recoil_scale = 1.0_f32;
            if let Some(recoil) = comp.recoil.as_mut() {
                let total = recoil.duration + recoil.return_duration;
                let t = recoil.elapsed.min(total);
                let attack_phase = t <= recoil.duration;
                let amount = if attack_phase {
                    (t / recoil.duration).clamp(0.0, 1.0)
                } else {
                    (1.0 - ((t - recoil.duration) / recoil.return_duration)).clamp(0.0, 1.0)
                };
                let min_scale = tpl.recoil.scale.clamp(0.2, 1.5);
                recoil_scale = if attack_phase {
                    1.0 + (min_scale - 1.0) * amount
                } else {
                    1.0 + (min_scale - 1.0) * amount
                };
                if tpl.recoil.mode != "scale_pulse" {
                    let dir = tower_render_dir_from_world_rad(recoil.dir_rad);
                    recoil_offset = -dir * (tpl.recoil.distance * WORLD_SCALE * amount);
                }
                recoil.elapsed += animation_dt;
                if recoil.elapsed >= total {
                    comp.recoil = None;
                }
            }

            let scale = recoil_scale.max(0.05);
            scene.graph[comp.base_node]
                .local_transform_mut()
                .set_position(Vector3::new(
                    pos.x + recoil_offset.x,
                    pos.y + recoil_offset.y,
                    Z_TOWER,
                ))
                .set_rotation(UnitQuaternion::identity())
                .set_scale(Vector3::new(
                    base_size * scale,
                    base_size * scale,
                    f32::EPSILON,
                ));

            if let Some(barrel) = comp.barrel_node {
                if !is_animated_area {
                    let offset = tower_render_offset(&tpl.barrel_offset, scale);
                    let pivot_to_center = Vector2::new(
                        (0.5 - tpl.barrel_pivot.x) * base_size * scale,
                        (tpl.barrel_pivot.y - 0.5) * base_size * scale,
                    );
                    let barrel_center =
                        pos + recoil_offset + offset + rotate_vec2(pivot_to_center, aim_angle);
                    scene.graph[barrel]
                        .local_transform_mut()
                        .set_position(Vector3::new(
                            barrel_center.x,
                            barrel_center.y,
                            Z_TOWER - 0.04,
                        ))
                        .set_rotation(UnitQuaternion::from_axis_angle(
                            &Vector3::z_axis(),
                            aim_angle,
                        ))
                        .set_scale(Vector3::new(
                            base_size * scale,
                            base_size * scale,
                            f32::EPSILON,
                        ));
                }
            }
        }

        material_updates.extend(post_material_updates);
        for (node, key, color) in material_updates {
            let material = self.tower_material_for_key(&key);
            set_tower_rect_material(scene, node, material, color);
        }
    }

    /// 階段 4.3：將 `PlayerInput` 傳送到 omoba-core KCP lockstep client。
    /// 如果 `lockstep_handle` 為 `None`，則 no-op。目標刻度 =
    /// `latest_tick + input_lookahead_ticks(server_timing)`。
    /// 底層 `KcpClient` 會用 `GameStart` 快取的 `player_id` 包裝 `InputSubmit`。
    /// 階段 5.x：每個tick，將 sim_runner 快照實體鏡像到
    /// 共享 body_batch + hp_batch CPU 映像。為每個實體分配一個插槽
    /// 第一次見到時；釋放輟學時的插槽。 EntityKind::其他是
    /// 已跳過（不應呈現 RegionBlocker 等內部 ECS 行）。
    fn update_sim_batches(
        &mut self,
        scene: &mut Scene,
        resource_manager: &ResourceManager,
        dt: f32,
    ) {
        let Some(sim_state) = self.sim_runner_handle.as_ref().map(|sim| sim.state.clone()) else {
            return;
        };
        let Ok(snapshot) = sim_state.try_lock() else {
            return;
        };
        if self.sim_batches_last_snapshot_tick == Some(snapshot.tick) {
            return;
        }
        self.sim_batches_last_snapshot_tick = Some(snapshot.tick);
        // 取走自上次 tick 以來累積的動畫 dt（替換舊的 render frame dt，
        // 使動畫以正確的 wall clock 速度播放）。
        let sim_tick_anim_dt = std::mem::take(&mut self.tower_anim_dt_acc);

        self.sim_seen_attack_phase_fx
            .retain(|key| snapshot.tick.saturating_sub(key.2) <= RENDER_FX_SEEN_RETENTION_TICKS);
        self.sim_seen_tower_fire_fx
            .retain(|key| snapshot.tick.saturating_sub(key.2) <= RENDER_FX_SEEN_RETENTION_TICKS);
        self.sim_seen_attack_cancel_fx
            .retain(|key| snapshot.tick.saturating_sub(key.2) <= RENDER_FX_SEEN_RETENTION_TICKS);

        let mut attack_cues: HashMap<u32, &sim_runner::AttackPhaseFx> = HashMap::new();
        for cue in &snapshot.attack_phase_fx {
            if self
                .sim_seen_attack_phase_fx
                .insert(attack_phase_fx_key(cue))
            {
                attack_cues.insert(cue.entity_id, cue);
            }
        }
        let mut fire_cues: HashMap<u32, &sim_runner::TowerFireFx> = HashMap::new();
        for cue in &snapshot.tower_fire_fx {
            if self.sim_seen_tower_fire_fx.insert(tower_fire_fx_key(cue)) {
                fire_cues.insert(cue.entity_id, cue);
            }
        }
        let mut cancel_cues: HashMap<u32, &sim_runner::AttackCancelFx> = HashMap::new();
        for cue in &snapshot.attack_cancel_fx {
            if self
                .sim_seen_attack_cancel_fx
                .insert(attack_cancel_fx_key(cue))
            {
                cancel_cues.insert(cue.entity_id, cue);
            }
        }

        let mut alive = std::collections::HashSet::with_capacity(snapshot.entities.len());
        for e in &snapshot.entities {
            if matches!(e.kind, sim_runner::EntityKind::Other) {
                continue;
            }
            alive.insert(e.entity_id);
            self.entity_kind_cache.insert(e.entity_id, e.kind);

            let pos = render_bridge::world_to_render(e);
            // 每幀快取 creep 的 render 座標，供死亡特效在下方移除迴圈取用（死亡時該 eid 已不在
            // entities 裡，只能靠上一幀的快取拿到最後位置）。與死亡音效共用同一條可靠路徑。
            // 注意座標慣例：`world_to_render` 已對 x 做 -1 翻轉（回傳 -pos_x*scale），但死亡特效
            // 的渲染路徑（同爆炸圈）會再翻一次 x，所以這裡存「未翻轉」座標 `-pos.x` 抵銷，
            // 否則特效會畫到鏡像位置（畫面外）而看不到。爆炸圈 ActiveExplosion 也是存未翻轉座標。
            if matches!(e.kind, sim_runner::EntityKind::Creep) {
                self.creep_death_cache
                    .insert(e.entity_id, Vector2::new(-pos.x, pos.y));
            }
            let (color, size, z) = render_bridge::style_for_entity(e);
            let tower_template = if matches!(e.kind, sim_runner::EntityKind::Tower) {
                self.td_templates.get(&e.unit_id).cloned()
            } else {
                None
            };
            let uses_composite_tower = tower_template.is_some();
            let hp_anchor_size = tower_template
                .as_ref()
                .map(tower_visual_size)
                .unwrap_or(size);
            if let Some(tpl) = tower_template.as_ref() {
                self.update_tower_composite(
                    scene,
                    e,
                    tpl,
                    pos,
                    sim_tick_anim_dt,
                    attack_cues.get(&e.entity_id).copied(),
                    fire_cues.get(&e.entity_id).copied(),
                );
            }
            let hero_model_active = if let Some(render) = e.hero_render.as_deref() {
                self.update_hero_model(
                    scene,
                    resource_manager,
                    e,
                    render,
                    pos,
                    snapshot.tick,
                    dt,
                    attack_cues.get(&e.entity_id).copied(),
                    cancel_cues.get(&e.entity_id).copied(),
                )
            } else {
                self.remove_hero_model(scene, e.entity_id);
                false
            };
            let projectile_initial = if matches!(e.kind, sim_runner::EntityKind::Projectile) {
                e.projectile_owner_entity_id
                    .and_then(|owner_id| {
                        self.projectile_owner_spawn_and_dir(scene, &snapshot.entities, owner_id)
                    })
                    .unwrap_or((pos, Vector2::new(1.0, 0.0)))
            } else {
                (pos, Vector2::new(1.0, 0.0))
            };

            // 主體槽：在第一次看到時分配，然後在每個刻度上 write_quad。
            let slots_entry = self.sim_entity_slots.entry(e.entity_id);
            let slots = slots_entry.or_insert_with(|| {
                let body_slot = self.body_batch.as_mut().map(|b| b.alloc()).unwrap_or(0);
                render_bridge::SimEntitySlots {
                    body_slot,
                    hp_bg_slot: None,
                    hp_fg_slot: None,
                    turret_slot: None,
                    proj_accent_slot: None,
                }
            });

            if matches!(e.kind, sim_runner::EntityKind::Projectile) {
                // 子彈：從發射點到當前位置畫一條甜點拖尾／旋轉甜點（不是中央小方塊）。
                // 第一次看到該 eid 時鎖定 spawn_pos 與方向，避免飛行中旋轉。
                let spawn_pos = *self
                    .projectile_spawn_pos
                    .entry(e.entity_id)
                    .or_insert(projectile_initial.0);
                let trail_dir =
                    *self
                        .projectile_trail_dir
                        .entry(e.entity_id)
                        .or_insert_with(|| {
                            initial_projectile_trail_dir(spawn_pos, pos, projectile_initial.1)
                        });

                // 甜點分類：第一次看到子彈時依「發射者塔／英雄」鎖定，之後每 tick 沿用，
                // 避免每顆子彈每 tick 重做字串比對（本機 sim 的子彈自身 unit_id 為空，
                // 可靠來源是發射者：tower_dart / tower_tack / ... / hero_saika_*）。
                let dessert = *self
                    .projectile_dessert
                    .entry(e.entity_id)
                    .or_insert_with(|| {
                        let owner = e.projectile_owner_entity_id.and_then(|oid| {
                            snapshot.entities.iter().find(|o| o.entity_id == oid)
                        });
                        classify_dessert_bullet(
                            &e.unit_id,
                            owner.map(|o| o.unit_id.as_str()),
                            owner.map(|o| o.kind),
                        )
                    });

                let render = dessert_bullet_render(
                    dessert,
                    e.entity_id,
                    snapshot.tick,
                    spawn_pos,
                    pos,
                    trail_dir,
                );

                // 快取子彈最後位置（未翻轉，供命中時炸開濺射）。pos = world_to_render 已翻 x，
                // 命中濺射走 drawing_context 會再翻一次，故存 -pos.x 抵銷（同死亡特效慣例）。
                self.projectile_last_pos
                    .insert(e.entity_id, Vector2::new(-pos.x, pos.y));

                // 惰性配置 accent slot（僅馬卡龍引信／冰霜點／香蕉殘影需要），且只有在
                // body_batch 尚有容量時才配置，避免 stress 場景撐爆 batch 觸發 debug_assert。
                if render.accent.is_some()
                    && slots.proj_accent_slot.is_none()
                    && self.body_batch.as_ref().map(|b| b.can_alloc()).unwrap_or(false)
                {
                    if let Some(batch) = self.body_batch.as_mut() {
                        slots.proj_accent_slot = Some(batch.alloc());
                    }
                }

                if let Some(batch) = self.body_batch.as_mut() {
                    let body = render.body;
                    batch.write_quad(
                        slots.body_slot,
                        &sprite_resources::QuadParams {
                            center: body.center,
                            size: body.size,
                            color: body.color,
                            rotation: body.rotation,
                            z: z - 0.01,
                        },
                    );
                    if let (Some(acc_slot), Some(acc)) =
                        (slots.proj_accent_slot, render.accent)
                    {
                        batch.write_quad(
                            acc_slot,
                            &sprite_resources::QuadParams {
                                center: acc.center,
                                size: acc.size,
                                color: acc.color,
                                rotation: acc.rotation,
                                z: z - 0.011,
                            },
                        );
                    }
                }
            } else if let Some(batch) = self.body_batch.as_mut() {
                if uses_composite_tower || hero_model_active {
                    batch.write_quad(
                        slots.body_slot,
                        &sprite_resources::QuadParams {
                            center: pos,
                            size: Vector2::new(0.001, 0.001),
                            color: [0, 0, 0, 0],
                            rotation: 0.0,
                            z,
                        },
                    );
                } else {
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
            }

            // HP 條（背景 + 目標）。塔滿血時隱藏，避免畫面被不必要的血條佔滿。
            let tower_full_hp = matches!(e.kind, sim_runner::EntityKind::Tower) && e.hp >= e.max_hp;
            let wants_hp_bar = e.max_hp > 0 && !tower_full_hp;
            if wants_hp_bar {
                if slots.hp_bg_slot.is_none() {
                    if let Some(batch) = self.hp_batch.as_mut() {
                        slots.hp_bg_slot = Some(batch.alloc());
                        slots.hp_fg_slot = Some(batch.alloc());
                    }
                }
                if let (Some(bg), Some(fg)) = (slots.hp_bg_slot, slots.hp_fg_slot) {
                    // Red Alert 2 風格：黑色外框 + 鮮豔填充色。bg 全寬、fg 內縮一圈
                    // 留 2-3 px 黑邊當 outline。HP 從右邊縮（左對齊）— RA2 慣例。
                    let bar_w = (hp_anchor_size * 1.6).max(0.5);
                    let bar_h = 0.18_f32;
                    let bar_y = pos.y + hp_anchor_size * 0.7;
                    let pad = 0.04_f32; // 黑外框視覺寬度
                    let inner_w = (bar_w - pad * 2.0).max(0.001);
                    let inner_h = (bar_h - pad * 2.0).max(0.001);
                    let hp_ratio = (e.hp as f32 / e.max_hp as f32).clamp(0.0, 1.0);
                    // RA2 經典三段配色：鮮綠 / 金黃 / 鮮紅
                    let bar_color: [u8; 4] = if hp_ratio < 0.30 {
                        [240, 40, 30, 255]
                    } else if hp_ratio < 0.60 {
                        [255, 200, 40, 255]
                    } else {
                        [80, 240, 60, 255]
                    };
                    if let Some(batch) = self.hp_batch.as_mut() {
                        // bg = 全寬黑底（外框）
                        batch.write_quad(
                            bg,
                            &sprite_resources::QuadParams {
                                center: Vector2::new(pos.x, bar_y),
                                size: Vector2::new(bar_w, bar_h),
                                color: [0, 0, 0, 255],
                                rotation: 0.0,
                                z: Z_HP_BAR + 0.01,
                            },
                        );
                        // fg = 內縮 pad，依 hp_ratio 縮短，左對齊（HP 從右邊扣）
                        let fg_w = inner_w * hp_ratio;
                        let fg_offset = (inner_w - fg_w) * 0.5;
                        batch.write_quad(
                            fg,
                            &sprite_resources::QuadParams {
                                center: Vector2::new(pos.x - fg_offset, bar_y),
                                size: Vector2::new(fg_w.max(0.001), inner_h),
                                color: bar_color,
                                rotation: 0.0,
                                z: Z_HP_BAR,
                            },
                        );
                    }
                }
            } else if slots.hp_bg_slot.is_some() || slots.hp_fg_slot.is_some() {
                if let Some(batch) = self.hp_batch.as_mut() {
                    if let Some(bg) = slots.hp_bg_slot.take() {
                        batch.free(bg);
                    }
                    if let Some(fg) = slots.hp_fg_slot.take() {
                        batch.free(fg);
                    }
                } else {
                    slots.hp_bg_slot = None;
                    slots.hp_fg_slot = None;
                }
            }

            // 砲塔/砲管指示器：一個小的黑色矩形偏移在
            // 面向方向。僅有意義的面孔的種類（英雄/
            // 塔樓/小兵）獲得一個槽位；彈頭會跳過。遺產
            // NetworkEntity 路徑使用 `entity.faceing_slot` + `faceing_batch`
            // 為了達到相同的效果 - sim_runner 支援的實體重用
            // 相同的“face_batch”，因此兩條路徑共用一個繪製呼叫。
            let wants_turret = matches!(
                e.kind,
                sim_runner::EntityKind::Hero | sim_runner::EntityKind::Creep,
            ) || (matches!(e.kind, sim_runner::EntityKind::Tower)
                && !uses_composite_tower);
            let wants_turret = wants_turret && !hero_model_active;
            if wants_turret {
                if slots.turret_slot.is_none() {
                    if let Some(batch) = self.facing_batch.as_mut() {
                        slots.turret_slot = Some(batch.alloc());
                    }
                }
                if let Some(slot) = slots.turret_slot {
                    // 鏡像傳統的“render_angle”數學（參見
                    // NetworkEntity 面向渲染線 ~1946)：
                    // 身體 sprite 在“world_to_render”中使用“-x”翻轉，
                    // 所以面對rad的世界需要反映出來
                    // 透過 Y 來匹配渲染的方向。
                    let render_angle = std::f32::consts::PI - e.facing_rad;
                    // 砲管 base 在 body 中心，向 facing 方向延伸出去（像坦克砲塔）。
                    // 中心距 body_center = length/2（quad center 算法），所以末端伸到
                    // body_center + length。z = body_z - 0.05 確保畫在 body 上方
                    // (lower z = closer to camera in this scene)；之前 +0.01 反而
                    // 把砲管推到 body 後面只看到伸出 body 那段一點點。
                    let length = (size * 1.2).max(0.20);
                    let thickness = (size * 0.28).max(0.08);
                    let attach_dist = length * 0.5;
                    let offset_x = attach_dist * render_angle.cos();
                    let offset_y = attach_dist * render_angle.sin();
                    let turret_color: [u8; 4] = [25, 25, 25, 255];
                    if let Some(batch) = self.facing_batch.as_mut() {
                        batch.write_quad(
                            slot,
                            &sprite_resources::QuadParams {
                                center: Vector2::new(pos.x + offset_x, pos.y + offset_y),
                                size: Vector2::new(length, thickness),
                                color: turret_color,
                                rotation: render_angle,
                                z: z - 0.05,
                            },
                        );
                    }
                }
            }
        }

        // 用於從快照中消失的實體的空閒插槽。
        // 階段 1.6：偏好產生明確的 `removed_entity_ids` diff
        // sim_runner 中的工作端在「alive」集掃描上 - 這是一個
        // 更嚴格的信號（僅那些在這個時間點死亡的人）並取代
        // 遺留的線路端「entity.death」事件。下面的掃描保持為
        // 對早於第一個 prev_alive 集的早期幀 eids 的防禦。
        for &eid in &snapshot.removed_entity_ids {
            // 每個 Creep 死亡：播一聲餅乾碎裂 + 在其最後位置爆出死亡特效（白環+紅放射線）。
            // 音效與特效共用同一個判斷/迴圈，保證一起觸發、不會有時有無。
            if self.entity_kind_cache.get(&eid)
                .map(|k| matches!(k, sim_runner::EntityKind::Creep))
                .unwrap_or(false)
            {
                self.play_sfx(scene, SfxKind::CookieCrunch);
                if let Some(pos) = self.creep_death_cache.remove(&eid) {
                    self.active_death_fx.push(ActiveDeathFx {
                        pos,
                        max_radius: 0.5, // render 單位：碎屑飛散最大距離
                        duration: 0.45,
                        elapsed: 0.0,
                        seed: eid,
                    });
                }
            }
            self.entity_kind_cache.remove(&eid);
            self.remove_tower_composite(scene, eid);
            self.remove_hero_model(scene, eid);
            if let Some(slots) = self.sim_entity_slots.remove(&eid) {
                if let Some(batch) = self.body_batch.as_mut() {
                    batch.free(slots.body_slot);
                    // 子彈 accent quad 與 body 同屬 body_batch，一起回收。
                    if let Some(acc) = slots.proj_accent_slot {
                        batch.free(acc);
                    }
                }
                if let Some(batch) = self.hp_batch.as_mut() {
                    if let Some(bg) = slots.hp_bg_slot {
                        batch.free(bg);
                    }
                    if let Some(fg) = slots.hp_fg_slot {
                        batch.free(fg);
                    }
                }
                if let Some(batch) = self.facing_batch.as_mut() {
                    if let Some(t) = slots.turret_slot {
                        batch.free(t);
                    }
                }
            }
            // 子彈消失時清拖尾起點 cache（HashMap 不會自己縮）
            // 命中濺射：子彈被移除（命中/超時）→ 在最後位置炸開該塔主題濺射（上限 256 防 stress 爆量）
            if self.active_hit_fx.len() < 256 {
                if let (Some(hit_pos), Some(kind)) = (
                    self.projectile_last_pos.get(&eid).copied(),
                    self.projectile_dessert.get(&eid).copied(),
                ) {
                    let (color, radius, style) = dessert_hit_spec(kind);
                    self.active_hit_fx.push(ActiveHitFx {
                        pos: hit_pos,
                        color,
                        max_radius: radius,
                        duration: 0.35,
                        elapsed: 0.0,
                        style,
                        seed: eid,
                    });
                }
            }
            self.projectile_spawn_pos.remove(&eid);
            self.projectile_trail_dir.remove(&eid);
            self.projectile_dessert.remove(&eid);
            self.projectile_last_pos.remove(&eid);
        }
        let to_remove: Vec<u32> = self
            .sim_entity_slots
            .keys()
            .filter(|id| !alive.contains(id))
            .copied()
            .collect();
        for id in to_remove {
            // 備用路徑（entity 從 alive 集消失但沒進 removed_entity_ids）：同樣播音效 + 死亡特效
            if self.entity_kind_cache.get(&id)
                .map(|k| matches!(k, sim_runner::EntityKind::Creep))
                .unwrap_or(false)
            {
                self.play_sfx(scene, SfxKind::CookieCrunch);
                if let Some(pos) = self.creep_death_cache.remove(&id) {
                    self.active_death_fx.push(ActiveDeathFx {
                        pos,
                        max_radius: 0.5,
                        duration: 0.45,
                        elapsed: 0.0,
                        seed: id,
                    });
                }
            }
            self.entity_kind_cache.remove(&id);
            self.remove_tower_composite(scene, id);
            self.remove_hero_model(scene, id);
            if let Some(slots) = self.sim_entity_slots.remove(&id) {
                if let Some(batch) = self.body_batch.as_mut() {
                    batch.free(slots.body_slot);
                    if let Some(acc) = slots.proj_accent_slot {
                        batch.free(acc);
                    }
                }
                if let Some(batch) = self.hp_batch.as_mut() {
                    if let Some(bg) = slots.hp_bg_slot {
                        batch.free(bg);
                    }
                    if let Some(fg) = slots.hp_fg_slot {
                        batch.free(fg);
                    }
                }
                if let Some(batch) = self.facing_batch.as_mut() {
                    if let Some(t) = slots.turret_slot {
                        batch.free(t);
                    }
                }
            }
            // 命中濺射（備用移除路徑）：同上，在最後位置炸開該塔主題濺射
            if self.active_hit_fx.len() < 256 {
                if let (Some(hit_pos), Some(kind)) = (
                    self.projectile_last_pos.get(&id).copied(),
                    self.projectile_dessert.get(&id).copied(),
                ) {
                    let (color, radius, style) = dessert_hit_spec(kind);
                    self.active_hit_fx.push(ActiveHitFx {
                        pos: hit_pos,
                        color,
                        max_radius: radius,
                        duration: 0.35,
                        elapsed: 0.0,
                        style,
                        seed: id,
                    });
                }
            }
            self.projectile_spawn_pos.remove(&id);
            self.projectile_trail_dir.remove(&id);
            self.projectile_dessert.remove(&id);
            self.projectile_last_pos.remove(&id);
        }
    }

    fn pair_applied_inputs(&mut self, applied_inputs: &[sim_runner::AppliedInputMeta]) {
        if applied_inputs.is_empty() {
            self.evict_stale_pending_inputs();
            return;
        }
        let render_us = wall_clock_us();
        for meta in applied_inputs {
            let input_id = meta.input_id;
            let Some(pending) = self.pending_inputs.remove(&input_id) else {
                continue;
            };
            let submit_start_us = pending
                .submit_start_us
                .unwrap_or(pending.send_lockstep_input_us);
            let submit_done_us = pending.submit_done_us.unwrap_or(submit_start_us);
            let client_receive_us = pending
                .client_receive_tickbatch_us
                .unwrap_or(meta.client_receive_us);
            let game_forward_us = pending
                .game_forward_to_sim_us
                .unwrap_or(meta.game_forward_us);
            let extract_data_for_render_us = pending
                .extract_data_for_render_us
                .unwrap_or(meta.extract_data_for_render_us);
            let total_ms = render_us
                .saturating_sub(pending.submit_wall_clock_us)
                .saturating_div(1_000)
                .min(u64::from(u32::MAX)) as u32;
            let phases = LatencyPhaseDurations {
                origin_to_send_us: pending
                    .send_lockstep_input_us
                    .saturating_sub(pending.origin_us),
                send_to_submit_start_us: submit_start_us
                    .saturating_sub(pending.send_lockstep_input_us),
                submit_io_us: submit_done_us.saturating_sub(submit_start_us),
                submit_to_client_receive_us: client_receive_us.saturating_sub(submit_done_us),
                server_queue_us: pending.server_queue_us.unwrap_or(meta.server_queue_us),
                client_receive_to_forward_us: game_forward_us.saturating_sub(client_receive_us),
                forward_to_extract_data_for_render_us: extract_data_for_render_us
                    .saturating_sub(game_forward_us),
                extract_data_for_render_to_pair_us: render_us
                    .saturating_sub(extract_data_for_render_us),
            };
            let additive_total_us = phases
                .origin_to_send_us
                .saturating_add(phases.send_to_submit_start_us)
                .saturating_add(phases.submit_io_us)
                .saturating_add(phases.submit_to_client_receive_us)
                .saturating_add(phases.client_receive_to_forward_us)
                .saturating_add(phases.forward_to_extract_data_for_render_us)
                .saturating_add(phases.extract_data_for_render_to_pair_us);
            let server_receive_tick = pending
                .server_receive_tick
                .or(Some(meta.server_receive_tick));
            let server_drain_tick = pending.server_drain_tick.or(Some(meta.server_drain_tick));
            let sim_latency_ticks = server_drain_tick
                .map(|drain_tick| drain_tick.wrapping_sub(pending.base_tick))
                .unwrap_or(0);
            let tick_quantized_latency_us =
                u64::from(sim_latency_ticks).saturating_mul(self.server_timing().tick_period_us());
            self.input_latency_meter.push(LatencySample {
                input_id,
                action_kind: pending.action_kind,
                total_ms,
                submitted_at: pending.submit_instant,
                origin_kind: pending.origin_kind,
                target_tick: pending.target_tick,
                server_receive_tick,
                server_drain_tick,
                phases: phases.clone(),
            });
            log::debug!(
                "input_render_latency: id={} kind={:?} base_tick={} target_tick={} submit_us={} render_us={} total_ms={} sim_latency_ticks={} tick_quantized_latency_us={}",
                input_id,
                pending.action_kind,
                pending.base_tick,
                pending.target_tick,
                pending.submit_wall_clock_us,
                render_us,
                total_ms,
                sim_latency_ticks,
                tick_quantized_latency_us,
            );
            log::debug!(
                "input_latency_phase: id={} origin={:?} origin_to_send_us={} send_to_submit_start_us={} submit_io_us={} submit_to_client_receive_us={} server_queue_us={} client_receive_to_forward_us={} forward_to_extract_data_for_render_us={} extract_data_for_render_to_pair_us={} server_receive_tick={:?} server_drain_tick={:?} additive_total_us={} server_queue_nested_us={} base_tick={} sim_latency_ticks={} tick_quantized_latency_us={}",
                input_id,
                pending.origin_kind,
                phases.origin_to_send_us,
                phases.send_to_submit_start_us,
                phases.submit_io_us,
                phases.submit_to_client_receive_us,
                phases.server_queue_us,
                phases.client_receive_to_forward_us,
                phases.forward_to_extract_data_for_render_us,
                phases.extract_data_for_render_to_pair_us,
                server_receive_tick,
                server_drain_tick,
                additive_total_us,
                phases.server_queue_us,
                pending.base_tick,
                sim_latency_ticks,
                tick_quantized_latency_us,
            );
        }
        self.evict_stale_pending_inputs();
    }

    fn wait_for_applied_input_render_data(
        &self,
        sim: &sim_runner::SimRunnerHandle,
        input_ids: &[u32],
    ) {
        if input_ids.is_empty() {
            return;
        }
        let deadline = Instant::now() + Duration::from_micros(INPUT_SAME_FRAME_WAIT_US);
        loop {
            if let Ok(snapshot) = sim.state.try_lock() {
                if snapshot
                    .applied_input_meta
                    .iter()
                    .any(|meta| input_ids.iter().any(|input_id| meta.input_id == *input_id))
                {
                    return;
                }
            }
            if Instant::now() >= deadline {
                return;
            }
            std::thread::yield_now();
        }
    }

    fn evict_stale_pending_inputs(&mut self) {
        let now = Instant::now();
        if let Some(next) = self.pending_inputs_evict_at {
            if now < next {
                return;
            }
        }
        self.pending_inputs_evict_at = Some(now + Duration::from_secs(1));
        let now_us = wall_clock_us();
        let cutoff_us = now_us.saturating_sub(PENDING_INPUT_MAX_AGE_MS * 1_000);
        let stale: Vec<PendingInputDiagnostic> = self
            .pending_inputs
            .iter()
            .filter(|(_, pending)| pending.submit_wall_clock_us < cutoff_us)
            .map(|(input_id, pending)| pending_input_diagnostic(*input_id, pending, now_us))
            .collect();

        for diag in &stale {
            log::warn!(
                "input_pending_stale: id={} kind={:?} base_tick={} target_tick={} pending_age_ms={} submit_start={} submit_done={} client_receive_tickbatch={} game_forward_to_sim={} extract_data_for_render={} server_receive_tick={:?} server_drain_tick={:?} server_queue_us={:?}",
                diag.input_id,
                diag.action_kind,
                diag.base_tick,
                diag.target_tick,
                diag.pending_age_ms,
                diag.has_submit_start,
                diag.has_submit_done,
                diag.has_client_receive_tickbatch,
                diag.has_game_forward_to_sim,
                diag.has_extract_data_for_render,
                diag.server_receive_tick,
                diag.server_drain_tick,
                diag.server_queue_us,
            );
        }

        let stale_count = stale.len() as u64;
        if stale_count > 0 {
            self.pending_inputs_stale = self.pending_inputs_stale.saturating_add(stale_count);
            self.pending_inputs_evicted = self.pending_inputs_evicted.saturating_add(stale_count);
            self.pending_inputs
                .retain(|_, pending| pending.submit_wall_clock_us >= cutoff_us);
        }
    }

    fn send_lockstep_input(&mut self, input: omoba_core::kcp::game_proto::PlayerInput) {
        let now_us = wall_clock_us();
        self.send_lockstep_input_from(input, lockstep_client::InputOriginKind::Direct, now_us);
    }

    fn send_lockstep_input_from(
        &mut self,
        input: omoba_core::kcp::game_proto::PlayerInput,
        origin_kind: lockstep_client::InputOriginKind,
        origin_us: u64,
    ) {
        let Some(handle) = self.lockstep_handle.as_ref() else {
            return;
        };
        let input_id = handle.next_input_id();
        let action_kind = InputActionKind::from_player_input(&input);
        let submit_wall_clock_us = wall_clock_us();
        let submit_instant = Instant::now();
        // 將 target_tick 基於 lockstep bg 線程的最新 TickBatch，而不是
        // 渲染線程的最後一個耗盡的tick；後者添加了一個框架
        // 甚至應用固定前瞻之前的延遲。
        let base_tick = handle.latest_tick();
        let timing = self.server_timing();
        let lookahead_ticks = input_lookahead_ticks(timing);
        let target_tick = base_tick.wrapping_add(lookahead_ticks);
        let lookahead_ms = f64::from(lookahead_ticks) * 1000.0 / f64::from(timing.step_fps());
        log::debug!(
            "input_submit_target: id={} kind={:?} base_tick={} lookahead={} ({:.1}ms @ {}fps) target_tick={}",
            input_id,
            action_kind,
            base_tick,
            lookahead_ticks,
            lookahead_ms,
            timing.step_fps(),
            target_tick,
        );
        if let Err(e) = handle.input_tx.send(lockstep_client::LockstepInputMsg {
            target_tick,
            input,
            input_id,
            origin_kind,
            origin_us,
            send_lockstep_input_us: submit_wall_clock_us,
        }) {
            log::warn!("[lockstep] input_tx send failed: {e}");
            return;
        }
        self.pending_inputs.insert(
            input_id,
            PendingInput {
                submit_wall_clock_us,
                submit_instant,
                base_tick,
                target_tick,
                action_kind,
                origin_kind,
                origin_us,
                send_lockstep_input_us: submit_wall_clock_us,
                submit_start_us: None,
                submit_done_us: None,
                client_receive_tickbatch_us: None,
                game_forward_to_sim_us: None,
                extract_data_for_render_us: None,
                server_receive_tick: None,
                server_drain_tick: None,
                server_queue_us: None,
            },
        );
    }

    fn send_upgrade_ability_input(&mut self, ability_index: u32) {
        let now_us = wall_clock_us();
        self.send_upgrade_ability_input_from(
            ability_index,
            lockstep_client::InputOriginKind::Direct,
            now_us,
        );
    }

    fn send_upgrade_ability_input_from(
        &mut self,
        ability_index: u32,
        origin_kind: lockstep_client::InputOriginKind,
        origin_us: u64,
    ) {
        let input = omoba_core::kcp::game_proto::PlayerInput {
            action: Some(
                omoba_core::kcp::game_proto::player_input::Action::UpgradeAbility(
                    omoba_core::kcp::game_proto::UpgradeAbility { ability_index },
                ),
            ),
        };
        self.send_lockstep_input_from(input, origin_kind, origin_us);
        log::info!(
            "Ability upgrade lockstep input submitted: index={}",
            ability_index
        );
    }

    fn send_cast_ability_input(&mut self, ability_index: u32, target_world: Option<Vector2<f32>>) {
        let now_us = wall_clock_us();
        self.send_cast_ability_input_from(
            ability_index,
            target_world,
            lockstep_client::InputOriginKind::Direct,
            now_us,
        );
    }

    fn send_cast_ability_input_from(
        &mut self,
        ability_index: u32,
        target_world: Option<Vector2<f32>>,
        origin_kind: lockstep_client::InputOriginKind,
        origin_us: u64,
    ) {
        let input = omoba_core::kcp::game_proto::PlayerInput {
            action: Some(
                omoba_core::kcp::game_proto::player_input::Action::CastAbility(
                    omoba_core::kcp::game_proto::CastAbility {
                        ability_index,
                        target_pos: target_world.map(world_render_to_vec2i),
                        target_entity: None,
                    },
                ),
            ),
        };
        self.send_lockstep_input_from(input, origin_kind, origin_us);
        log::info!(
            "Ability cast lockstep input submitted: index={}",
            ability_index
        );
    }

    fn send_attack_move_input_from(
        &mut self,
        target_world: Vector2<f32>,
        queued: bool,
        origin_kind: lockstep_client::InputOriginKind,
        origin_us: u64,
    ) {
        let input = omoba_core::kcp::game_proto::PlayerInput {
            action: Some(
                omoba_core::kcp::game_proto::player_input::Action::AttackMove(
                    omoba_core::kcp::game_proto::AttackMove {
                        target: Some(world_render_to_vec2i(target_world)),
                        queued,
                    },
                ),
            ),
        };
        self.send_lockstep_input_from(input, origin_kind, origin_us);
        log::info!("AttackMove lockstep input submitted queued={}", queued);
    }

    fn send_attack_target_input_from(
        &mut self,
        target_id: u32,
        queued: bool,
        origin_kind: lockstep_client::InputOriginKind,
        origin_us: u64,
    ) {
        let input = omoba_core::kcp::game_proto::PlayerInput {
            action: Some(
                omoba_core::kcp::game_proto::player_input::Action::AttackTarget(
                    omoba_core::kcp::game_proto::AttackTarget { target_id, queued },
                ),
            ),
        };
        self.send_lockstep_input_from(input, origin_kind, origin_us);
        log::info!(
            "AttackTarget lockstep input submitted target_id={} queued={}",
            target_id,
            queued
        );
    }

    fn send_tower_target_priority_input_from(
        &mut self,
        tower_entity_id: u32,
        priority: i32,
        origin_kind: lockstep_client::InputOriginKind,
        origin_us: u64,
    ) {
        let input = omoba_core::kcp::game_proto::PlayerInput {
            action: Some(
                omoba_core::kcp::game_proto::player_input::Action::SetTowerTargetPriority(
                    omoba_core::kcp::game_proto::SetTowerTargetPriority {
                        tower_entity_id,
                        priority,
                    },
                ),
            ),
        };
        self.send_lockstep_input_from(input, origin_kind, origin_us);
        log::info!(
            "SetTowerTargetPriority lockstep input submitted tower={} priority={}",
            tower_entity_id,
            priority
        );
    }

    fn cycle_selected_tower_priority(&mut self, reverse: bool, origin_us: u64) {
        let Some(tid) = self.selected_tower_entity else {
            return;
        };
        let current = self
            .network_entities
            .get(&tid)
            .map(|ent| ent.tower_target_priority.as_str())
            .unwrap_or("first");
        let next = if reverse {
            match current {
                "first" => 5,
                "last" => 0,
                "nearest" => 1,
                "farthest" => 2,
                "highest_health" => 3,
                "lowest_health" => 4,
                _ => 0,
            }
        } else {
            match current {
                "first" => 1,
                "last" => 2,
                "nearest" => 3,
                "farthest" => 4,
                "highest_health" => 5,
                "lowest_health" => 0,
                _ => 0,
            }
        };
        self.send_tower_target_priority_input_from(
            tid,
            next,
            lockstep_client::InputOriginKind::OsEvent,
            origin_us,
        );
    }

    /// 升級選中塔的指定路線（0/1/2）。UI 按鈕與 , . / 熱鍵共用。
    /// 回傳是否實際送出 lockstep input。
    fn send_tower_upgrade_for_selected(&mut self, path: u8, event_us: u64) -> bool {
        let Some(tid) = self.selected_tower_entity else {
            return false;
        };
        let owned_by_local = tower_owned_by_local(
            self.network_entities
                .get(&tid)
                .and_then(|ent| ent.owner_player_id),
            self.local_player_id,
        );
        if !owned_by_local {
            log::warn!(
                "Tower upgrade skipped locally: eid={} owner={:?} local_player_id={}",
                tid,
                self.network_entities
                    .get(&tid)
                    .and_then(|ent| ent.owner_player_id),
                self.local_player_id
            );
            return false;
        }
        let current_level = self
            .network_entities
            .get(&tid)
            .map(|e| e.upgrade_levels[path as usize])
            .unwrap_or(0);
        let target_level = current_level + 1;
        let input = omoba_core::kcp::game_proto::PlayerInput {
            action: Some(
                omoba_core::kcp::game_proto::player_input::Action::TowerUpgrade(
                    omoba_core::kcp::game_proto::TowerUpgradeInput {
                        tower_entity_id: tid,
                        path: path as u32,
                        level: target_level as u32,
                    },
                ),
            ),
        };
        self.send_lockstep_input_from(input, lockstep_client::InputOriginKind::OsEvent, event_us);
        log::info!(
            "Tower upgrade lockstep input submitted: eid={} path={} level={}",
            tid,
            path,
            target_level
        );
        // 樂觀更新：立即更新本機快取讓 UI 即時反應，不用等 server snapshot
        if let Some(ent) = self.network_entities.get_mut(&tid) {
            ent.upgrade_levels[path as usize] = ent.upgrade_levels[path as usize].saturating_add(1);
        }
        true
    }

    /// 賣出選中塔。UI 賣出按鈕與 Backspace 熱鍵共用。
    /// 回傳是否實際送出 lockstep input。
    fn send_tower_sell_for_selected(&mut self, event_us: u64) -> bool {
        let Some(tid) = self.selected_tower_entity else {
            return false;
        };
        let owned_by_local = tower_owned_by_local(
            self.network_entities
                .get(&tid)
                .and_then(|ent| ent.owner_player_id),
            self.local_player_id,
        );
        if !owned_by_local {
            log::warn!(
                "Tower sell skipped locally: eid={} owner={:?} local_player_id={}",
                tid,
                self.network_entities
                    .get(&tid)
                    .and_then(|ent| ent.owner_player_id),
                self.local_player_id
            );
            return false;
        }
        let input = omoba_core::kcp::game_proto::PlayerInput {
            action: Some(
                omoba_core::kcp::game_proto::player_input::Action::TowerSell(
                    omoba_core::kcp::game_proto::TowerSell {
                        tower_entity_id: tid,
                    },
                ),
            ),
        };
        self.send_lockstep_input_from(input, lockstep_client::InputOriginKind::OsEvent, event_us);
        log::info!("Tower sell lockstep input submitted: eid={}", tid);
        self.selected_tower_entity = None;
        self.ui_td_selected_panel.selected_path = 0;
        true
    }

    /// Start Round 按鈕三態邏輯：暫停中→恢復、回合進行中→切換速度、閒置→開始回合。
    /// UI 按鈕與 Space 熱鍵共用。
    fn trigger_start_round_button(&mut self, event_us: u64) {
        if self.is_game_paused {
            let input = omoba_core::kcp::game_proto::PlayerInput {
                action: Some(
                    omoba_core::kcp::game_proto::player_input::Action::TogglePause(
                        omoba_core::kcp::game_proto::TogglePause {},
                    ),
                ),
            };
            self.send_lockstep_input_from(
                input,
                lockstep_client::InputOriginKind::OsEvent,
                event_us,
            );
            log::info!("Start/Resume → lockstep PlayerInput::TogglePause sent");
        } else if self.round_is_running {
            let input = omoba_core::kcp::game_proto::PlayerInput {
                action: Some(
                    omoba_core::kcp::game_proto::player_input::Action::ToggleGameSpeed(
                        omoba_core::kcp::game_proto::ToggleGameSpeed {},
                    ),
                ),
            };
            self.send_lockstep_input_from(
                input,
                lockstep_client::InputOriginKind::OsEvent,
                event_us,
            );
            log::info!("Start Speed Toggle → lockstep PlayerInput::ToggleGameSpeed sent");
        } else if !(self.total_rounds > 0 && self.current_round >= self.total_rounds) {
            let input = omoba_core::kcp::game_proto::PlayerInput {
                action: Some(
                    omoba_core::kcp::game_proto::player_input::Action::StartRound(
                        omoba_core::kcp::game_proto::StartRound {},
                    ),
                ),
            };
            self.send_lockstep_input_from(
                input,
                lockstep_client::InputOriginKind::OsEvent,
                event_us,
            );
            self.td_auto_start_sent_for_idle_round = true;
            log::info!("Start Round → lockstep PlayerInput::StartRound sent");
        }
    }

    /// TD 模式資料驅動熱鍵分派（來源：hotkeys::HotkeyConfig，F1 面板可重綁）。
    /// 回傳 true 表示按鍵已被消耗。
    fn dispatch_td_hotkey(&mut self, id: &str, event_us: u64) -> bool {
        if let Some(rest) = id.strip_prefix("select_tower_") {
            if self.shop_visible {
                return false;
            }
            let Ok(n) = rest.parse::<usize>() else {
                return false;
            };
            let idx = n.saturating_sub(1);
            if let Some(uid) = self.td_template_order.get(idx).cloned() {
                self.selected_tower_kind = Some(uid.clone());
                self.selected_tower_entity = None;
                log::info!("熱鍵選塔 [{}]: {}", idx, uid);
            }
            return true;
        }
        if let Some(rest) = id.strip_prefix("spawn_creep_") {
            let Ok(n) = rest.parse::<u32>() else {
                return false;
            };
            let emitter_index = n.saturating_sub(1);
            let input = omoba_core::kcp::game_proto::PlayerInput {
                action: Some(
                    omoba_core::kcp::game_proto::player_input::Action::DebugSpawnCreep(
                        omoba_core::kcp::game_proto::DebugSpawnCreep {
                            emitter_index,
                            count: 1,
                        },
                    ),
                ),
            };
            self.send_lockstep_input_from(
                input,
                lockstep_client::InputOriginKind::OsEvent,
                event_us,
            );
            log::info!("沙箱生怪熱鍵 → emitter_index={}", emitter_index);
            return true;
        }
        match id {
            "upgrade_path_1" | "upgrade_path_2" | "upgrade_path_3" => {
                if self.selected_tower_entity.is_none() {
                    return false;
                }
                let path: u8 = match id {
                    "upgrade_path_1" => 0,
                    "upgrade_path_2" => 1,
                    _ => 2,
                };
                self.send_tower_upgrade_for_selected(path, event_us);
                true
            }
            "sell_tower" => {
                if self.selected_tower_entity.is_none() {
                    return false;
                }
                self.send_tower_sell_for_selected(event_us);
                true
            }
            "change_target" | "change_target_reverse" => {
                if self.selected_tower_entity.is_none() {
                    return false;
                }
                self.cycle_selected_tower_priority(id == "change_target_reverse", event_us);
                true
            }
            "copy_tower" => {
                let Some(kind) = self
                    .selected_tower_entity
                    .and_then(|tid| self.network_entities.get(&tid))
                    .and_then(|ent| ent.tower_kind.clone())
                else {
                    return false;
                };
                self.selected_tower_kind = Some(kind.clone());
                self.selected_tower_entity = None;
                log::info!("複製塔: {}", kind);
                true
            }
            "start_round" => {
                self.trigger_start_round_button(event_us);
                true
            }
            "toggle_pause" => {
                let input = omoba_core::kcp::game_proto::PlayerInput {
                    action: Some(
                        omoba_core::kcp::game_proto::player_input::Action::TogglePause(
                            omoba_core::kcp::game_proto::TogglePause {},
                        ),
                    ),
                };
                self.send_lockstep_input_from(
                    input,
                    lockstep_client::InputOriginKind::OsEvent,
                    event_us,
                );
                true
            }
            _ => false,
        }
    }

    /// 設定頁解析度下拉選單：lazy 建立、每 frame 佈局與高亮更新。
    fn update_resolution_dropdown(&mut self, ui: &mut UserInterface) {
        let dd = &mut self.resolution_dropdown_ui;
        let hidden = Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS);
        if !self.resolution_dropdown_open {
            if dd.built && dd.visible_applied {
                ui.send(dd.bg, WidgetMessage::DesiredPosition(hidden));
                for (row_bg, text) in &dd.rows {
                    ui.send(*row_bg, WidgetMessage::DesiredPosition(hidden));
                    ui.send(*text, WidgetMessage::DesiredPosition(hidden));
                }
                ui.send(dd.ok_bg, WidgetMessage::DesiredPosition(hidden));
                ui.send(dd.ok_text, WidgetMessage::DesiredPosition(hidden));
                dd.row_rects.clear();
                dd.ok_rect = UiRect::default();
                dd.visible_applied = false;
            }
            return;
        }

        if !dd.built {
            // 棕色圓角容器（BTD6 下拉風格）
            dd.bg = BorderBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(hidden)
                    .with_foreground(Brush::Solid(Color::from_rgba(96, 58, 26, 255)).into())
                    .with_background(Brush::Solid(Color::from_rgba(130, 84, 42, 255)).into()),
            )
            .with_stroke_thickness(Thickness::uniform(2.5).into())
            .with_corner_radius(12.0_f32.into())
            .build(&mut ui.build_ctx())
            .transmute();
            for _ in RESOLUTION_OPTIONS.iter() {
                let row_bg = BorderBuilder::new(
                    WidgetBuilder::new()
                        .with_desired_position(hidden)
                        .with_background(Brush::Solid(Color::from_rgba(70, 44, 20, 255)).into()),
                )
                .with_stroke_thickness(Thickness::uniform(0.0).into())
                .with_corner_radius(8.0_f32.into())
                .build(&mut ui.build_ctx())
                .transmute();
                let text = TextBuilder::new(
                    WidgetBuilder::new()
                        .with_desired_position(hidden)
                        .with_foreground(Brush::Solid(Color::from_rgba(255, 255, 255, 255)).into()),
                )
                .with_font_size(22.0.into())
                .with_horizontal_text_alignment(HorizontalAlignment::Center)
                .with_vertical_text_alignment(VerticalAlignment::Center)
                .build(&mut ui.build_ctx());
                dd.rows.push((row_bg, text));
            }
            dd.ok_bg = BorderBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(hidden)
                    .with_foreground(Brush::Solid(Color::from_rgba(255, 255, 255, 255)).into())
                    .with_background(Brush::Solid(Color::from_rgba(110, 198, 50, 255)).into()),
            )
            .with_stroke_thickness(Thickness::uniform(2.5).into())
            .with_corner_radius(14.0_f32.into())
            .build(&mut ui.build_ctx())
            .transmute();
            dd.ok_text = TextBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(hidden)
                    .with_foreground(Brush::Solid(Color::from_rgba(255, 255, 255, 255)).into()),
            )
            .with_text("好的".to_string())
            .with_font_size(26.0.into())
            .with_horizontal_text_alignment(HorizontalAlignment::Center)
            .with_vertical_text_alignment(VerticalAlignment::Center)
            .build(&mut ui.build_ctx());
            dd.built = true;
        }

        // 佈局：badge 正下方
        let badge = self.settings_resolution_rect;
        let row_h = (badge.h * 0.58).max(30.0);
        let gap = row_h * 0.16;
        let ok_h = row_h * 1.15;
        let x = badge.x;
        let w = badge.w;
        let top = badge.y + badge.h + 6.0;
        let total_h = RESOLUTION_OPTIONS.len() as f32 * (row_h + gap) + ok_h + gap * 2.0 + 16.0;
        let bg_rect = UiRect {
            x: x - 10.0,
            y: top - 8.0,
            w: w + 20.0,
            h: total_h,
        };
        ui.send(dd.bg, WidgetMessage::DesiredPosition(bg_rect.pos()));
        ui.send(dd.bg, WidgetMessage::Width(bg_rect.w));
        ui.send(dd.bg, WidgetMessage::Height(bg_rect.h));

        dd.row_rects.clear();
        let cur_w = self.window_size.x as u32;
        let mut y = top;
        for (i, opt) in RESOLUTION_OPTIONS.iter().enumerate() {
            let rect = UiRect { x, y, w, h: row_h };
            let (row_bg, text) = dd.rows[i];
            let selected = match opt {
                None => self.display_fullscreen,
                Some((ow, _)) => !self.display_fullscreen && cur_w.abs_diff(*ow) < 40,
            };
            let color = if selected {
                Color::from_rgba(110, 198, 50, 255)
            } else {
                Color::from_rgba(70, 44, 20, 255)
            };
            ui.send(row_bg, WidgetMessage::DesiredPosition(rect.pos()));
            ui.send(row_bg, WidgetMessage::Width(rect.w));
            ui.send(row_bg, WidgetMessage::Height(rect.h));
            ui.send(
                row_bg,
                WidgetMessage::Background(Brush::Solid(color).into()),
            );
            ui.send(text, WidgetMessage::DesiredPosition(rect.pos()));
            ui.send(text, WidgetMessage::Width(rect.w));
            ui.send(text, WidgetMessage::Height(rect.h));
            let label = match opt {
                None => "Fullscreen".to_string(),
                Some((ow, oh)) => format!("{} x {}", ow, oh),
            };
            ui.send(text, TextMessage::Text(label));
            dd.row_rects.push(rect);
            y += row_h + gap;
        }
        dd.ok_rect = UiRect {
            x,
            y: y + gap,
            w,
            h: ok_h,
        };
        ui.send(dd.ok_bg, WidgetMessage::DesiredPosition(dd.ok_rect.pos()));
        ui.send(dd.ok_bg, WidgetMessage::Width(dd.ok_rect.w));
        ui.send(dd.ok_bg, WidgetMessage::Height(dd.ok_rect.h));
        ui.send(dd.ok_text, WidgetMessage::DesiredPosition(dd.ok_rect.pos()));
        ui.send(dd.ok_text, WidgetMessage::Width(dd.ok_rect.w));
        ui.send(dd.ok_text, WidgetMessage::Height(dd.ok_rect.h));
        dd.visible_applied = true;
    }

    // ──────────────────────────────────────────────────────────────────────
    // 英雄知識（Hero Knowledge）面板
    // ──────────────────────────────────────────────────────────────────────

    /// 從 `omb/player_profile.json` 讀取 KP 與已解鎖節點到前端快取。
    fn load_gk_profile(&mut self) {
        let path = PathBuf::from("omb").join("player_profile.json");
        match std::fs::read_to_string(&path) {
            Ok(raw) => match serde_json::from_str::<serde_json::Value>(&raw) {
                Ok(v) => {
                    self.gk_available_kp = v
                        .get("total_kp")
                        .and_then(|x| x.as_u64())
                        .unwrap_or(0) as u32;
                    let spent = v
                        .get("spent_kp")
                        .and_then(|x| x.as_u64())
                        .unwrap_or(0) as u32;
                    self.gk_available_kp = self.gk_available_kp.saturating_sub(spent);
                    self.gk_unlocked_nodes.clear();
                    if let Some(arr) = v.get("unlocked_nodes").and_then(|x| x.as_array()) {
                        for item in arr {
                            if let Some(s) = item.as_str() {
                                self.gk_unlocked_nodes.insert(s.to_string());
                            }
                        }
                    }
                }
                Err(e) => {
                    log::warn!("[gk] player_profile.json parse error: {}", e);
                }
            },
            Err(_) => {
                // 首次啟動，profile 不存在是正常情況
            }
        }
    }

    /// 嘗試從前端直接執行解鎖邏輯（寫回 omb/player_profile.json）。
    fn try_unlock_gk_node(&mut self, node_id: &str) {
        // 讀取 knowledge_tree.json 取得費用與前置節點
        let tree_path = PathBuf::from("scripts/lua_data/knowledge_tree.json");
        let tree_json = match std::fs::read_to_string(&tree_path) {
            Ok(s) => s,
            Err(e) => {
                log::warn!("[gk] cannot read knowledge_tree.json: {}", e);
                return;
            }
        };
        let tree: serde_json::Value = match serde_json::from_str(&tree_json) {
            Ok(v) => v,
            Err(e) => {
                log::warn!("[gk] knowledge_tree.json parse error: {}", e);
                return;
            }
        };
        let nodes = match tree.as_array() {
            Some(a) => a,
            None => return,
        };
        // 找到目標節點
        let node = match nodes.iter().find(|n| {
            n.get("id").and_then(|v| v.as_str()) == Some(node_id)
        }) {
            Some(n) => n,
            None => {
                log::warn!("[gk] node '{}' not found in tree", node_id);
                return;
            }
        };
        let kp_cost = node.get("kp_cost").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
        // 驗證前置節點
        if let Some(requires) = node.get("requires").and_then(|v| v.as_array()) {
            for req in requires {
                if let Some(req_id) = req.as_str() {
                    if !self.gk_unlocked_nodes.contains(req_id) {
                        log::info!("[gk] unlock '{}' blocked: requires '{}' first", node_id, req_id);
                        return;
                    }
                }
            }
        }
        // 驗證 KP
        if self.gk_available_kp < kp_cost {
            log::info!("[gk] unlock '{}' blocked: need {} KP, have {}", node_id, kp_cost, self.gk_available_kp);
            return;
        }
        // 讀取現有 profile JSON
        let profile_path = PathBuf::from("omb").join("player_profile.json");
        let mut profile: serde_json::Value = std::fs::read_to_string(&profile_path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_else(|| serde_json::json!({
                "total_kp": 0, "spent_kp": 0, "unlocked_nodes": []
            }));
        // 更新 spent_kp 和 unlocked_nodes
        let spent = profile.get("spent_kp").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
        profile["spent_kp"] = serde_json::json!(spent + kp_cost);
        if let Some(arr) = profile.get_mut("unlocked_nodes").and_then(|v| v.as_array_mut()) {
            arr.push(serde_json::json!(node_id));
        }
        // 寫回
        match serde_json::to_string_pretty(&profile) {
            Ok(json_str) => {
                if let Err(e) = std::fs::write(&profile_path, json_str) {
                    log::warn!("[gk] failed to save profile: {}", e);
                    return;
                }
            }
            Err(e) => {
                log::warn!("[gk] failed to serialize profile: {}", e);
                return;
            }
        }
        // 更新前端快取
        self.gk_unlocked_nodes.insert(node_id.to_string());
        self.gk_available_kp = self.gk_available_kp.saturating_sub(kp_cost);
        log::info!("[gk] unlocked '{}', remaining KP={}", node_id, self.gk_available_kp);
    }

    /// GK 面板點擊處理（面板可見時 modal 攔截所有點擊）。
    fn handle_gk_panel_click(&mut self, screen: Vector2<f32>) {
        let close_rect = self.gk_panel_ui.close_rect;
        if close_rect.w > 0.0 && close_rect.contains(screen) {
            self.gk_panel_visible = false;
            return;
        }
        // 解鎖按鈕 hit-test
        let rects: Vec<UiRect> = self.gk_panel_ui.unlock_rects.clone();
        let ids: Vec<String> = self.gk_panel_ui.node_ids.clone();
        for (i, rect) in rects.iter().enumerate() {
            if rect.w > 0.0 && rect.contains(screen) {
                if let Some(node_id) = ids.get(i) {
                    if !self.gk_unlocked_nodes.contains(node_id) {
                        let id = node_id.clone();
                        self.try_unlock_gk_node(&id);
                    }
                }
                return;
            }
        }
    }

    /// 英雄知識面板：lazy 建立節點、每 frame 更新位置與狀態文字。
    fn update_gk_panel(&mut self, ui: &mut UserInterface) {
        fn gk_cat_display_name(cat: &str) -> &'static str {
            match cat {
                "tower_dart"      => "飛鏢塔",
                "tower_bomb"      => "炸彈塔",
                "tower_ice"       => "冰晶塔",
                "tower_tack"      => "圖釘塔",
                "tower_boomerang" => "迴旋鏢塔",
                "tower_arty"      => "砲兵塔",
                "hero"            => "英雄",
                "global"          => "全域",
                _                 => "其他",
            }
        }
        let panel = &mut self.gk_panel_ui;
        if !self.gk_panel_visible {
            if panel.built && panel.visible_applied {
                let hidden = Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS);
                ui.send(panel.backdrop, WidgetMessage::DesiredPosition(hidden));
                ui.send(panel.bg, WidgetMessage::DesiredPosition(hidden));
                ui.send(panel.title, WidgetMessage::DesiredPosition(hidden));
                ui.send(panel.kp_text, WidgetMessage::DesiredPosition(hidden));
                ui.send(panel.close_bg, WidgetMessage::DesiredPosition(hidden));
                ui.send(panel.close_text, WidgetMessage::DesiredPosition(hidden));
                for (card_bg, name_t, kp_t, btn_bg, btn_t) in &panel.node_cards {
                    for h in [*card_bg, *btn_bg] {
                        ui.send(h, WidgetMessage::DesiredPosition(hidden));
                    }
                    for h in [*name_t, *kp_t, *btn_t] {
                        ui.send(h, WidgetMessage::DesiredPosition(hidden));
                    }
                }
                for h in &panel.cat_header_bgs {
                    ui.send(*h, WidgetMessage::DesiredPosition(hidden));
                }
                for h in &panel.cat_header_texts {
                    ui.send(*h, WidgetMessage::DesiredPosition(hidden));
                }
                panel.unlock_rects.iter_mut().for_each(|r| *r = UiRect::default());
                panel.close_rect = UiRect::default();
                panel.visible_applied = false;
            }
            return;
        }

        // 讀取 knowledge_tree.json（只在面板剛打開且未建置時讀取一次）
        // 節點 id 列表由 built 路徑收集，後續 frame 直接用 node_ids 列表。
        if !panel.built {
            // 載入節點清單
            let tree_path = PathBuf::from("scripts/lua_data/knowledge_tree.json");
            let nodes: Vec<(String, String, u32, Vec<String>)> = // (id, category, kp_cost, requires)
                std::fs::read_to_string(&tree_path)
                    .ok()
                    .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
                    .and_then(|v| v.as_array().cloned())
                    .unwrap_or_default()
                    .iter()
                    .map(|n| {
                        let id = n.get("id").and_then(|v| v.as_str()).unwrap_or("?").to_string();
                        let cat = n.get("category").and_then(|v| v.as_str()).unwrap_or("global").to_string();
                        let cost = n.get("kp_cost").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                        let reqs: Vec<String> = n.get("requires")
                            .and_then(|v| v.as_array())
                            .map(|arr| arr.iter().filter_map(|r| r.as_str().map(|s| s.to_string())).collect())
                            .unwrap_or_default();
                        (id, cat, cost, reqs)
                    })
                    .collect();

            let hidden = Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS);

            // 全螢幕深藍 backdrop
            panel.backdrop = BorderBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(hidden)
                    .with_background(Brush::Solid(Color::from_rgba(16, 28, 58, 245)).into()),
            )
            .with_stroke_thickness(Thickness::uniform(0.0).into())
            .build(&mut ui.build_ctx())
            .transmute();

            // 中央容器（亮藍灰）
            panel.bg = BorderBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(hidden)
                    .with_background(Brush::Solid(Color::from_rgba(150, 172, 212, 255)).into()),
            )
            .with_stroke_thickness(Thickness::uniform(0.0).into())
            .with_corner_radius(18.0_f32.into())
            .build(&mut ui.build_ctx())
            .transmute();

            // 標題（棕色橫幅文字）
            panel.title = TextBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(hidden)
                    .with_foreground(Brush::Solid(Color::from_rgba(255, 230, 150, 255)).into()),
            )
            .with_text("英雄知識".to_string())
            .with_font_size(44.0.into())
            .with_horizontal_text_alignment(HorizontalAlignment::Center)
            .with_vertical_text_alignment(VerticalAlignment::Center)
            .build(&mut ui.build_ctx());

            // KP 顯示
            panel.kp_text = TextBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(hidden)
                    .with_foreground(Brush::Solid(Color::from_rgba(255, 230, 80, 255)).into()),
            )
            .with_text(format!("KP: {}", self.gk_available_kp))
            .with_font_size(28.0.into())
            .with_horizontal_text_alignment(HorizontalAlignment::Right)
            .with_vertical_text_alignment(VerticalAlignment::Center)
            .build(&mut ui.build_ctx());

            // 關閉按鈕
            panel.close_bg = BorderBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(hidden)
                    .with_foreground(Brush::Solid(Color::from_rgba(255, 255, 255, 255)).into())
                    .with_background(
                        Brush::LinearGradient {
                            from: Vector2::new(0.0, 0.0),
                            to: Vector2::new(0.0, 58.0),
                            stops: vec![
                                GradientPoint { stop: 0.0, color: Color::from_rgba(220, 80, 60, 255) },
                                GradientPoint { stop: 1.0, color: Color::from_rgba(160, 40, 30, 255) },
                            ],
                        }
                        .into(),
                    ),
            )
            .with_stroke_thickness(Thickness::uniform(2.0).into())
            .with_corner_radius(12.0_f32.into())
            .build(&mut ui.build_ctx())
            .transmute();

            panel.close_text = TextBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(hidden)
                    .with_foreground(Brush::Solid(Color::from_rgba(255, 255, 255, 255)).into()),
            )
            .with_text("✕".to_string())
            .with_font_size(36.0.into())
            .with_horizontal_text_alignment(HorizontalAlignment::Center)
            .with_vertical_text_alignment(VerticalAlignment::Center)
            .build(&mut ui.build_ctx());

            // 為每個節點建立卡片
            panel.node_ids.clear();
            panel.node_kp_costs.clear();
            panel.node_categories.clear();
            panel.node_cards.clear();
            panel.unlock_rects.clear();
            panel.cat_header_bgs.clear();
            panel.cat_header_texts.clear();
            panel.cat_names.clear();

            for (id, cat, cost, _reqs) in &nodes {
                let card_bg = BorderBuilder::new(
                    WidgetBuilder::new()
                        .with_desired_position(hidden)
                        .with_background(Brush::Solid(Color::from_rgba(100, 120, 170, 230)).into())
                        .with_foreground(Brush::Solid(Color::from_rgba(180, 200, 240, 255)).into()),
                )
                .with_stroke_thickness(Thickness::uniform(2.0).into())
                .with_corner_radius(10.0_f32.into())
                .build(&mut ui.build_ctx())
                .transmute();

                let name_text = TextBuilder::new(
                    WidgetBuilder::new()
                        .with_desired_position(hidden)
                        .with_foreground(Brush::Solid(Color::from_rgba(240, 240, 255, 255)).into()),
                )
                .with_text(id.clone())
                .with_font_size(18.0.into())
                .with_wrap(WrapMode::Word)
                .build(&mut ui.build_ctx());

                let kp_text = TextBuilder::new(
                    WidgetBuilder::new()
                        .with_desired_position(hidden)
                        .with_foreground(Brush::Solid(Color::from_rgba(255, 220, 60, 255)).into()),
                )
                .with_text(String::new())
                .with_font_size(16.0.into())
                .with_horizontal_text_alignment(HorizontalAlignment::Right)
                .build(&mut ui.build_ctx());

                let btn_bg = BorderBuilder::new(
                    WidgetBuilder::new()
                        .with_desired_position(hidden)
                        .with_foreground(Brush::Solid(Color::from_rgba(255, 255, 255, 255)).into())
                        .with_background(
                            Brush::LinearGradient {
                                from: Vector2::new(0.0, 0.0),
                                to: Vector2::new(0.0, 32.0),
                                stops: vec![
                                    GradientPoint { stop: 0.0, color: Color::from_rgba(95, 200, 250, 255) },
                                    GradientPoint { stop: 1.0, color: Color::from_rgba(35, 130, 215, 255) },
                                ],
                            }
                            .into(),
                        ),
                )
                .with_stroke_thickness(Thickness::uniform(2.0).into())
                .with_corner_radius(8.0_f32.into())
                .build(&mut ui.build_ctx())
                .transmute();

                let btn_text = TextBuilder::new(
                    WidgetBuilder::new()
                        .with_desired_position(hidden)
                        .with_foreground(Brush::Solid(Color::from_rgba(255, 255, 255, 255)).into()),
                )
                .with_text("解鎖".to_string())
                .with_font_size(16.0.into())
                .with_horizontal_text_alignment(HorizontalAlignment::Center)
                .with_vertical_text_alignment(VerticalAlignment::Center)
                .build(&mut ui.build_ctx());

                panel.node_cards.push((card_bg, name_text, kp_text, btn_bg, btn_text));
                panel.node_ids.push(id.clone());
                panel.node_kp_costs.push(*cost);
                panel.node_categories.push(cat.clone());
                panel.unlock_rects.push(UiRect::default());
            }

            // Category section headers（依 JSON 出現順序，去重）
            for (_, cat, _, _) in &nodes {
                if !panel.cat_names.contains(cat) {
                    panel.cat_names.push(cat.clone());
                }
            }
            for cat_name in &panel.cat_names.clone() {
                let total_kp: u32 = nodes.iter()
                    .filter(|(_, c, _, _)| c == cat_name)
                    .map(|(_, _, kp, _)| *kp)
                    .sum();
                let display = gk_cat_display_name(cat_name);
                let hdr_bg = BorderBuilder::new(
                    WidgetBuilder::new()
                        .with_desired_position(hidden)
                        .with_background(Brush::Solid(Color::from_rgba(30, 50, 110, 220)).into())
                        .with_foreground(Brush::Solid(Color::from_rgba(80, 110, 180, 255)).into()),
                )
                .with_stroke_thickness(Thickness::uniform(2.0).into())
                .with_corner_radius(8.0_f32.into())
                .build(&mut ui.build_ctx())
                .transmute();
                let hdr_txt = TextBuilder::new(
                    WidgetBuilder::new()
                        .with_desired_position(hidden)
                        .with_foreground(Brush::Solid(Color::from_rgba(255, 220, 80, 255)).into()),
                )
                .with_text(format!("{}  ── 解鎖全部需 {} KP", display, total_kp))
                .with_font_size(22.0.into())
                .with_vertical_text_alignment(VerticalAlignment::Center)
                .build(&mut ui.build_ctx());
                panel.cat_header_bgs.push(hdr_bg);
                panel.cat_header_texts.push(hdr_txt);
            }

            panel.built = true;
        }

        // ──── 每 frame 佈局 ────
        let ws = self.window_size;
        let rr = |x: f32, y: f32, w: f32, h: f32| td_ui_ref_rect(ws, x, y, w, h);
        let send_rect = |ui: &mut UserInterface, h: Handle<UiNode>, r: UiRect| {
            ui.send(h, WidgetMessage::DesiredPosition(r.pos()));
            ui.send(h, WidgetMessage::Width(r.w));
            ui.send(h, WidgetMessage::Height(r.h));
        };
        let send_rect_text = |ui: &mut UserInterface, h: Handle<Text>, r: UiRect| {
            ui.send(h, WidgetMessage::DesiredPosition(r.pos()));
            ui.send(h, WidgetMessage::Width(r.w));
            ui.send(h, WidgetMessage::Height(r.h));
        };

        // 全螢幕 backdrop + 中央容器
        send_rect(ui, panel.backdrop, rr(0.0, 0.0, TD_UI_REF_W, TD_UI_REF_H));
        send_rect(ui, panel.bg, rr(30.0, 148.0, 1860.0, 900.0));

        // 標題橫幅
        send_rect_text(ui, panel.title, rr(710.0, 10.0, 500.0, 80.0));

        // KP 顯示（右上角）
        let kp_str = format!("可用 KP: {}", self.gk_available_kp);
        ui.send(panel.kp_text, TextMessage::Text(kp_str));
        send_rect_text(ui, panel.kp_text, rr(1400.0, 16.0, 440.0, 60.0));

        // 關閉按鈕（左上角）
        let close_r = rr(44.0, 16.0, 70.0, 60.0);
        panel.close_rect = close_r;
        send_rect(ui, panel.close_bg, close_r);
        send_rect_text(ui, panel.close_text, close_r);

        // ── 2 欄 Category 版面 ──
        // 左欄（cat idx 0..4）/ 右欄（cat idx 4+），每欄 2 張卡片並排
        const CAT_COL_LEFT_X: f32 = 50.0;
        const CAT_COL_RIGHT_X: f32 = 980.0;
        const CAT_COL_W: f32 = 900.0;
        const CARDS_PER_ROW: usize = 2;
        const CARD_W: f32 = 420.0;
        const CARD_COL_PITCH: f32 = 450.0;
        const CARD_H: f32 = 88.0;
        const CARD_ROW_PITCH: f32 = 100.0;
        const HEADER_H: f32 = 32.0;
        const HEADER_TO_CARD: f32 = 6.0;
        const CAT_GAP: f32 = 12.0;
        const CAT_START_Y: f32 = 108.0;

        // Snapshot：避免後續 borrow 衝突
        let node_ids_snap: Vec<String> = panel.node_ids.clone();
        let node_kp_snap: Vec<u32> = panel.node_kp_costs.clone();
        let node_cat_snap: Vec<String> = panel.node_categories.clone();
        let cat_names_snap: Vec<String> = panel.cat_names.clone();
        drop(panel);

        // 計算各 category 在各欄的 Y 起始位置並佈局 header
        let mut left_y = CAT_START_Y;
        let mut right_y = CAT_START_Y;
        let mut cat_col_x: Vec<f32> = Vec::new();
        let mut cat_start_y_vec: Vec<f32> = Vec::new();

        for (cat_idx, cat_name) in cat_names_snap.iter().enumerate() {
            let is_right = cat_idx >= 4;
            let col_x = if is_right { CAT_COL_RIGHT_X } else { CAT_COL_LEFT_X };
            let y = if is_right { right_y } else { left_y };
            cat_col_x.push(col_x);
            cat_start_y_vec.push(y);

            let hdr_r = rr(col_x, y, CAT_COL_W, HEADER_H);
            send_rect(ui, self.gk_panel_ui.cat_header_bgs[cat_idx], hdr_r);
            send_rect_text(ui, self.gk_panel_ui.cat_header_texts[cat_idx], hdr_r);

            let n_in_cat = node_cat_snap.iter().filter(|c| c.as_str() == cat_name.as_str()).count();
            let n_rows = (n_in_cat + CARDS_PER_ROW - 1) / CARDS_PER_ROW;
            let cat_h = HEADER_H + HEADER_TO_CARD + n_rows as f32 * CARD_ROW_PITCH + CAT_GAP;
            if is_right { right_y += cat_h; } else { left_y += cat_h; }
        }

        // 佈局每張節點卡片
        let mut cat_card_count: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();

        for (i, node_id) in node_ids_snap.iter().enumerate() {
            let cat = &node_cat_snap[i];
            let cost = node_kp_snap[i];
            let card_in_cat = *cat_card_count.get(cat.as_str()).unwrap_or(&0);
            cat_card_count.insert(cat.clone(), card_in_cat + 1);

            let cat_idx = cat_names_snap.iter().position(|c| c == cat).unwrap_or(0);
            let col_x_base = cat_col_x[cat_idx];
            let cat_y = cat_start_y_vec[cat_idx];

            let col_in_cat = card_in_cat % CARDS_PER_ROW;
            let row_in_cat = card_in_cat / CARDS_PER_ROW;
            let x = col_x_base + col_in_cat as f32 * CARD_COL_PITCH;
            let y = cat_y + HEADER_H + HEADER_TO_CARD + row_in_cat as f32 * CARD_ROW_PITCH;

            let card_r = rr(x, y, CARD_W, CARD_H);
            let is_unlocked = self.gk_unlocked_nodes.contains(node_id.as_str());

            let (card_bg, name_t, kp_t, btn_bg, btn_t) = self.gk_panel_ui.node_cards[i];
            send_rect(ui, card_bg, card_r);
            ui.send(
                card_bg,
                WidgetMessage::Background(
                    if is_unlocked {
                        Brush::Solid(Color::from_rgba(40, 110, 55, 220)).into()
                    } else {
                        Brush::Solid(Color::from_rgba(100, 120, 170, 230)).into()
                    },
                ),
            );

            // 節點名稱（左半部）
            let name_r = rr(x + 8.0, y + 4.0, CARD_W * 0.55, CARD_H - 8.0);
            send_rect_text(ui, name_t, name_r);
            ui.send(name_t, TextMessage::Text(node_id.clone()));

            // KP 費用標籤（右上）
            let kp_info_r = rr(x + CARD_W * 0.57, y + 4.0, CARD_W * 0.4, 34.0);
            send_rect_text(ui, kp_t, kp_info_r);

            // 解鎖按鈕（右下）
            let btn_r = rr(x + CARD_W * 0.57, y + CARD_H - 38.0, CARD_W * 0.4, 32.0);
            if is_unlocked {
                ui.send(btn_bg, WidgetMessage::DesiredPosition(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS)));
                ui.send(btn_t, WidgetMessage::DesiredPosition(Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS)));
                ui.send(kp_t, TextMessage::Text("✓ 已解鎖".to_string()));
                self.gk_panel_ui.unlock_rects[i] = UiRect::default();
            } else {
                send_rect(ui, btn_bg, btn_r);
                send_rect_text(ui, btn_t, btn_r);
                self.gk_panel_ui.unlock_rects[i] = btn_r;
                ui.send(kp_t, TextMessage::Text(format!("需 {} KP", cost)));
                ui.send(btn_t, TextMessage::Text(format!("解鎖 {}KP", cost)));
            }
        }

        self.gk_panel_ui.visible_applied = true;
    }

    /// 熱鍵設定面板點擊處理（in-game 與 pregame 設定頁共用）。
    fn handle_hotkey_panel_click(&mut self, screen: Vector2<f32>) {
        let close_rect = self.hotkey_panel_ui.close_rect;
        let reset_rect = self.hotkey_panel_ui.reset_rect;
        if close_rect.contains(screen) {
            self.hotkey_panel_visible = false;
            self.hotkey_rebinding = None;
        } else if reset_rect.contains(screen) {
            self.hotkeys = hotkeys::HotkeyConfig::defaults();
            self.hotkeys.save();
            self.hotkey_rebinding = None;
            log::info!("熱鍵已恢復預設值");
        } else {
            let mut hit_row: Option<&'static str> = None;
            for (i, rect) in self.hotkey_panel_ui.row_rects.iter().enumerate() {
                if rect.contains(screen) {
                    hit_row = self.hotkey_panel_ui.row_ids.get(i).copied();
                    break;
                }
            }
            if let Some(id) = hit_row {
                self.hotkey_rebinding = Some(id.to_string());
            }
        }
    }

    /// 熱鍵設定面板：lazy 建立節點、每 frame 更新位置與文字。
    fn update_hotkey_panel(&mut self, ui: &mut UserInterface) {
        let panel = &mut self.hotkey_panel_ui;
        if !self.hotkey_panel_visible {
            if panel.built && panel.visible_applied {
                let hidden = Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS);
                ui.send(panel.backdrop, WidgetMessage::DesiredPosition(hidden));
                ui.send(panel.bg, WidgetMessage::DesiredPosition(hidden));
                ui.send(panel.header_banner, WidgetMessage::DesiredPosition(hidden));
                ui.send(panel.title, WidgetMessage::DesiredPosition(hidden));
                ui.send(panel.reset_bg, WidgetMessage::DesiredPosition(hidden));
                ui.send(panel.reset_text, WidgetMessage::DesiredPosition(hidden));
                ui.send(panel.close_bg, WidgetMessage::DesiredPosition(hidden));
                ui.send(panel.close_text, WidgetMessage::DesiredPosition(hidden));
                for h in &panel.cat_titles {
                    ui.send(*h, WidgetMessage::DesiredPosition(hidden));
                }
                for (row_bg, label, key_bg, key_text) in &panel.rows {
                    ui.send(*row_bg, WidgetMessage::DesiredPosition(hidden));
                    ui.send(*label, WidgetMessage::DesiredPosition(hidden));
                    ui.send(*key_bg, WidgetMessage::DesiredPosition(hidden));
                    ui.send(*key_text, WidgetMessage::DesiredPosition(hidden));
                }
                panel.row_rects.clear();
                panel.reset_rect = UiRect::default();
                panel.close_rect = UiRect::default();
                panel.visible_applied = false;
            }
            return;
        }

        let defs = hotkeys::hotkey_defs();
        if !panel.built {
            let hidden = Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS);
            // BTD6 全螢幕深藍 backdrop
            panel.backdrop = BorderBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(hidden)
                    .with_background(Brush::Solid(Color::from_rgba(16, 28, 58, 245)).into()),
            )
            .with_stroke_thickness(Thickness::uniform(0.0).into())
            .build(&mut ui.build_ctx())
            .transmute();
            // 內容大卡片：亮藍灰圓角容器
            panel.bg = BorderBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(hidden)
                    .with_background(Brush::Solid(Color::from_rgba(150, 172, 212, 255)).into()),
            )
            .with_stroke_thickness(Thickness::uniform(0.0).into())
            .with_corner_radius(18.0_f32.into())
            .build(&mut ui.build_ctx())
            .transmute();
            // 棕色木質標題橫幅（BTD6 頂部）
            // 注意：LinearGradient 需要 build 時就有 width/height，否則漸層
            // 以整個畫面空間計算，顏色會隨位置漂移（賣出按鈕同 pattern）。
            panel.header_banner = BorderBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(hidden)
                    .with_width(440.0)
                    .with_height(72.0)
                    .with_foreground(Brush::Solid(Color::from_rgba(96, 58, 26, 255)).into())
                    .with_background(
                        Brush::LinearGradient {
                            from: Vector2::new(0.0, 0.0),
                            to: Vector2::new(0.0, 72.0),
                            stops: vec![
                                GradientPoint {
                                    stop: 0.0,
                                    color: Color::from_rgba(168, 116, 66, 255),
                                },
                                GradientPoint {
                                    stop: 1.0,
                                    color: Color::from_rgba(120, 76, 38, 255),
                                },
                            ],
                        }
                        .into(),
                    ),
            )
            .with_stroke_thickness(Thickness::uniform(3.0).into())
            .with_corner_radius(14.0_f32.into())
            .build(&mut ui.build_ctx())
            .transmute();
            panel.title = TextBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(hidden)
                    .with_foreground(Brush::Solid(Color::from_rgba(255, 255, 255, 255)).into()),
            )
            .with_text("熱鍵".to_string())
            .with_font_size(40.0.into())
            .with_horizontal_text_alignment(HorizontalAlignment::Center)
            .with_vertical_text_alignment(VerticalAlignment::Center)
            .build(&mut ui.build_ctx());
            // 恢復預設值：BTD6 亮藍 + 白描邊
            panel.reset_bg = BorderBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(hidden)
                    .with_width(230.0)
                    .with_height(52.0)
                    .with_foreground(Brush::Solid(Color::from_rgba(255, 255, 255, 255)).into())
                    .with_background(
                        Brush::LinearGradient {
                            from: Vector2::new(0.0, 0.0),
                            to: Vector2::new(0.0, 44.0),
                            stops: vec![
                                GradientPoint {
                                    stop: 0.0,
                                    color: Color::from_rgba(95, 200, 250, 255),
                                },
                                GradientPoint {
                                    stop: 1.0,
                                    color: Color::from_rgba(35, 130, 215, 255),
                                },
                            ],
                        }
                        .into(),
                    ),
            )
            .with_stroke_thickness(Thickness::uniform(2.5).into())
            .with_corner_radius(12.0_f32.into())
            .build(&mut ui.build_ctx())
            .transmute();
            panel.reset_text = TextBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(hidden)
                    .with_foreground(Brush::Solid(Color::from_rgba(255, 255, 255, 255)).into()),
            )
            .with_text("恢復預設值".to_string())
            .with_font_size(22.0.into())
            .with_horizontal_text_alignment(HorizontalAlignment::Center)
            .with_vertical_text_alignment(VerticalAlignment::Center)
            .build(&mut ui.build_ctx());
            // 左上返回鈕（BTD6 藍色圓角，作為關閉）
            panel.close_bg = BorderBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(hidden)
                    .with_width(66.0)
                    .with_height(58.0)
                    .with_foreground(Brush::Solid(Color::from_rgba(255, 255, 255, 255)).into())
                    .with_background(
                        Brush::LinearGradient {
                            from: Vector2::new(0.0, 0.0),
                            to: Vector2::new(0.0, 58.0),
                            stops: vec![
                                GradientPoint {
                                    stop: 0.0,
                                    color: Color::from_rgba(95, 200, 250, 255),
                                },
                                GradientPoint {
                                    stop: 1.0,
                                    color: Color::from_rgba(35, 130, 215, 255),
                                },
                            ],
                        }
                        .into(),
                    ),
            )
            .with_stroke_thickness(Thickness::uniform(2.5).into())
            .with_corner_radius(14.0_f32.into())
            .build(&mut ui.build_ctx())
            .transmute();
            panel.close_text = TextBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(hidden)
                    .with_foreground(Brush::Solid(Color::from_rgba(255, 255, 255, 255)).into()),
            )
            .with_text("<".to_string())
            .with_font_size(36.0.into())
            .with_horizontal_text_alignment(HorizontalAlignment::Center)
            .with_vertical_text_alignment(VerticalAlignment::Center)
            .build(&mut ui.build_ctx());
            // 分類標題：白色大字（放在深藍 backdrop 上，BTD6 風格）
            for cat_label in ["塔選擇", "遊戲玩法", "沙箱"] {
                let h = TextBuilder::new(
                    WidgetBuilder::new()
                        .with_desired_position(hidden)
                        .with_foreground(Brush::Solid(Color::from_rgba(255, 255, 255, 255)).into()),
                )
                .with_text(cat_label.to_string())
                .with_font_size(32.0.into())
                .with_horizontal_text_alignment(HorizontalAlignment::Center)
                .build(&mut ui.build_ctx());
                panel.cat_titles.push(h);
            }
            for def in &defs {
                // 每列灰藍圓角卡片（BTD6 row：比容器暗一階）
                let row_bg = BorderBuilder::new(
                    WidgetBuilder::new()
                        .with_desired_position(hidden)
                        .with_background(Brush::Solid(Color::from_rgba(122, 142, 186, 255)).into()),
                )
                .with_stroke_thickness(Thickness::uniform(0.0).into())
                .with_corner_radius(6.0_f32.into())
                .build(&mut ui.build_ctx())
                .transmute();
                let label = TextBuilder::new(
                    WidgetBuilder::new()
                        .with_desired_position(hidden)
                        .with_foreground(Brush::Solid(Color::from_rgba(255, 255, 255, 255)).into()),
                )
                .with_text(def.label.to_string())
                .with_font_size(32.0.into())
                .with_vertical_text_alignment(VerticalAlignment::Center)
                .build(&mut ui.build_ctx());
                // BTD6 綠色膠囊鈕：純色亮綠 + 白描邊（漸層在絕對座標空間會隨
                // 位置變色，Doris 拍板改純色）
                let key_bg = BorderBuilder::new(
                    WidgetBuilder::new()
                        .with_desired_position(hidden)
                        .with_foreground(Brush::Solid(Color::from_rgba(255, 255, 255, 255)).into())
                        .with_background(Brush::Solid(Color::from_rgba(110, 198, 50, 255)).into()),
                )
                .with_stroke_thickness(Thickness::uniform(2.5).into())
                .with_corner_radius(16.0_f32.into())
                .build(&mut ui.build_ctx())
                .transmute();
                let key_text = TextBuilder::new(
                    WidgetBuilder::new()
                        .with_desired_position(hidden)
                        .with_foreground(Brush::Solid(Color::from_rgba(255, 255, 255, 255)).into()),
                )
                .with_font_size(26.0.into())
                .with_horizontal_text_alignment(HorizontalAlignment::Center)
                .with_vertical_text_alignment(VerticalAlignment::Center)
                .build(&mut ui.build_ctx());
                panel.rows.push((row_bg, label, key_bg, key_text));
                panel.row_ids.push(def.id);
            }
            panel.built = true;
        }

        // 每 frame 佈局（跟隨視窗尺寸縮放）
        let ws = self.window_size;
        let rr = |x: f32, y: f32, w: f32, h: f32| td_ui_ref_rect(ws, x, y, w, h);
        let send_rect = |ui: &mut UserInterface, h: Handle<UiNode>, r: UiRect| {
            ui.send(h, WidgetMessage::DesiredPosition(r.pos()));
            ui.send(h, WidgetMessage::Width(r.w));
            ui.send(h, WidgetMessage::Height(r.h));
        };
        let send_rect_text = |ui: &mut UserInterface, h: Handle<Text>, r: UiRect| {
            ui.send(h, WidgetMessage::DesiredPosition(r.pos()));
            ui.send(h, WidgetMessage::Width(r.w));
            ui.send(h, WidgetMessage::Height(r.h));
        };

        // 全螢幕深藍 backdrop + 中央亮藍灰容器（BTD6 頁面結構）
        send_rect(ui, panel.backdrop, rr(0.0, 0.0, TD_UI_REF_W, TD_UI_REF_H));
        send_rect(ui, panel.bg, rr(30.0, 148.0, 1860.0, 916.0));
        // 棕色標題橫幅置中
        send_rect(ui, panel.header_banner, rr(710.0, 10.0, 500.0, 80.0));
        send_rect_text(ui, panel.title, rr(710.0, 10.0, 500.0, 80.0));
        panel.reset_rect = rr(1660.0, 22.0, 230.0, 52.0);
        send_rect(ui, panel.reset_bg, panel.reset_rect);
        send_rect_text(ui, panel.reset_text, panel.reset_rect);
        // 左上返回鈕 = 關閉
        panel.close_rect = rr(22.0, 14.0, 66.0, 58.0);
        send_rect(ui, panel.close_bg, panel.close_rect);
        send_rect_text(ui, panel.close_text, panel.close_rect);

        // 三欄佈局：塔選擇 / 遊戲玩法 / 沙箱，各占一欄。
        // 列高加大（BTD6 chunky 風格），每欄最多約 9 列。
        const COL_X: [f32; 3] = [70.0, 690.0, 1310.0];
        const CARD_W: f32 = 540.0;
        const ROW_PITCH: f32 = 92.0;
        const CARD_H: f32 = 78.0;
        let cat_rects = [
            rr(COL_X[0], 100.0, CARD_W, 42.0),
            rr(COL_X[1], 100.0, CARD_W, 42.0),
            rr(COL_X[2], 100.0, CARD_W, 42.0),
        ];
        for (i, h) in panel.cat_titles.iter().enumerate() {
            send_rect_text(ui, *h, cat_rects[i]);
        }

        panel.row_rects.clear();
        let hidden_pos = Vector2::new(UI_HIDDEN_POS, UI_HIDDEN_POS);
        let hidden_rect = UiRect {
            x: UI_HIDDEN_POS,
            y: UI_HIDDEN_POS,
            w: 0.0,
            h: 0.0,
        };
        let mut gameplay_row = 0usize;
        let mut sandbox_row = 0usize;
        // tower_seen = 模板位置索引（對應熱鍵綁定）；tower_row = 顯示列（壓縮
        // 掉隱藏列）。td_template_order 會被商店排版 debug padding 塞入重複的
        // placeholder 塔，面板只顯示每個 uid 的第一次出現 + 有模板的位置。
        let mut tower_seen = 0usize;
        let mut tower_row = 0usize;
        for (i, def) in defs.iter().enumerate() {
            let mut tower_idx: Option<usize> = None;
            let visible = match def.category {
                hotkeys::HotkeyCategory::Tower => {
                    let t_idx = tower_seen;
                    tower_seen += 1;
                    let slot_ok = if self.td_template_order.is_empty() {
                        // pregame/設定頁（無地圖塔模板）：顯示已知的固定塔名
                        t_idx < hotkeys::KNOWN_TOWER_COUNT
                    } else {
                        match self.td_template_order.get(t_idx) {
                            Some(uid) => !self.td_template_order[..t_idx].contains(uid),
                            None => false,
                        }
                    };
                    // 單欄最多 9 列，超過的隱藏避免爆版
                    slot_ok && tower_row < 9
                }
                _ => true,
            };
            if !visible {
                let (row_bg, label, key_bg, key_text) = panel.rows[i];
                for h in [row_bg, key_bg] {
                    ui.send(h, WidgetMessage::DesiredPosition(hidden_pos));
                }
                ui.send(label, WidgetMessage::DesiredPosition(hidden_pos));
                ui.send(key_text, WidgetMessage::DesiredPosition(hidden_pos));
                panel.row_rects.push(hidden_rect);
                continue;
            }
            let (col, row) = match def.category {
                hotkeys::HotkeyCategory::Tower => {
                    tower_idx = Some(tower_seen - 1);
                    let r = tower_row;
                    tower_row += 1;
                    (0, r)
                }
                hotkeys::HotkeyCategory::Gameplay => {
                    let r = gameplay_row;
                    gameplay_row += 1;
                    (1, r)
                }
                hotkeys::HotkeyCategory::Sandbox => {
                    let r = sandbox_row;
                    sandbox_row += 1;
                    (2, r)
                }
            };
            let x = COL_X[col];
            let y = 175.0 + row as f32 * ROW_PITCH;
            let (row_bg, label, key_bg, key_text) = panel.rows[i];
            send_rect(ui, row_bg, rr(x, y, CARD_W, CARD_H));
            send_rect_text(ui, label, rr(x + 24.0, y, 280.0, CARD_H));
            // 塔列顯示實際塔名（來自後端 tower templates）；沒有對應塔時
            // 回退顯示「選塔 N」
            if let Some(t_idx) = tower_idx {
                let name = self
                    .td_template_order
                    .get(t_idx)
                    .and_then(|uid| self.td_templates.get(uid))
                    .map(|tpl| tpl.label.clone())
                    .unwrap_or_else(|| def.label.to_string());
                ui.send(label, TextMessage::Text(name));
            }
            let key_rect = rr(x + 310.0, y + 11.0, 210.0, 56.0);
            send_rect(ui, key_bg, key_rect);
            send_rect_text(ui, key_text, key_rect);
            panel.row_rects.push(key_rect);
            let display = if self.hotkey_rebinding.as_deref() == Some(def.id) {
                "按新鍵…".to_string()
            } else {
                self.hotkeys.binding_display(def.id)
            };
            ui.send(key_text, TextMessage::Text(display));
        }
        panel.visible_applied = true;
    }

    fn draw_hero_command_queue_overlay(&self, scene: &mut Scene) {
        let Some(hero_id) = self.hero_state.entity_id else {
            return;
        };
        let Some(hero) = self
            .latest_entities
            .iter()
            .find(|entity| entity.entity_id == hero_id)
        else {
            return;
        };
        let Some(command) = hero.hero_command.as_ref() else {
            return;
        };

        let points: Vec<_> = command
            .queued_targets
            .iter()
            .filter_map(|target| {
                target
                    .target
                    .map(|(x, y)| Vector2::new(x * WORLD_SCALE, y * WORLD_SCALE))
            })
            .collect();
        if points.is_empty() {
            return;
        }

        let line_color = Color::from_rgba(70, 240, 120, 180);
        let hero_pos = Vector2::new(hero.pos_x * WORLD_SCALE, hero.pos_y * WORLD_SCALE);
        add_dashed_world_line(
            scene,
            hero_pos,
            points[0],
            0.28,
            0.14,
            line_color,
            Z_COMMAND_QUEUE + 0.02,
        );
        for pair in points.windows(2) {
            add_dashed_world_line(
                scene,
                pair[0],
                pair[1],
                0.28,
                0.14,
                line_color,
                Z_COMMAND_QUEUE + 0.02,
            );
        }
        for point in points {
            add_command_queue_flag(scene, point, Z_COMMAND_QUEUE);
        }
    }

    fn enemy_entity_at_world(&self, world: Vector2<f32>) -> Option<u32> {
        self.latest_entities
            .iter()
            .filter(|entity| match entity.kind {
                sim_runner::EntityKind::Creep => true,
                sim_runner::EntityKind::Hero => {
                    entity.owner_player_id != Some(self.local_player_id)
                }
                _ => false,
            })
            .filter_map(|entity| {
                let pos = Vector2::new(entity.pos_x * WORLD_SCALE, entity.pos_y * WORLD_SCALE);
                let radius = match entity.kind {
                    sim_runner::EntityKind::Creep => 0.45,
                    sim_runner::EntityKind::Hero => 0.55,
                    _ => 0.35,
                };
                let d = (pos - world).norm();
                (d <= radius).then_some((entity.entity_id, d))
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(id, _)| id)
    }

    // 階段 5.1（第 2 階段）：apply_event + 30+ 舊版 GameEvent 處理程序
    // 刪除方法（entity_create /entity_move/entity_hp_update/
    // 實體_刪除/實體_面向_更新/實體_速度_更新/
    // entity_stall/projectile_create/projectile_delete/hero_stats_update/
    // 英雄庫存更新/英雄能力資訊更新/地圖路徑更新/
    // map_regions_update / map_region_blockers_update / td_round_update /
    // td_lives_update / td_tower_templates_update / td_explosion_spawn /
    // tower_upgrade_apply / game_end）。舊版 0x02 GameEvent 串流是
    // 消失（第 4.5 階段伺服器端，第 5.1 階段透過 1 用戶端）；渲染橋
    // 擁有來自 sim 狀態的 sprite 渲染。現場清理延後到通過 3。
}

fn ability_max_level(ability_info_map: &HashMap<String, AbilityInfo>, ability_id: &str) -> i32 {
    ability_info_map
        .get(ability_id)
        .map(|a| a.max_level)
        .filter(|max| *max > 0)
        .unwrap_or(4)
}

/// 建立一個細長的旋轉矩形，表示從“from”到“to”的線段。
/// 如果段長度為零，則傳回「無」。
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
        if arr.is_empty() {
            None
        } else {
            Some(arr[idx.min(arr.len() - 1)])
        }
    }
    fn at_i32(arr: &[i32], idx: usize) -> Option<i32> {
        if arr.is_empty() {
            None
        } else {
            Some(arr[idx.min(arr.len() - 1)])
        }
    }

    let mut out = String::new();

    // ===== 標題列 =====
    if is_ultimate {
        out.push_str(&format!(
            "★ {}  (終極)   [{}]\n",
            info.name, info.key_binding
        ));
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
        if c > 0.0 {
            out.push_str(&format!("  施放範圍：{:.0}\n", c));
        }
    }

    // ===== 效果（傷害 / 其他）=====
    if !info.effects.is_empty() {
        out.push_str(bar);
        out.push_str("【效果】\n");
        // 優先顯示常見欄位（damage / heal / ratio / duration / stun / slow）
        let priority_keys = [
            "damage", "heal", "shield", "duration", "stun", "slow", "ad_ratio", "ap_ratio", "ratio",
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
        if let (Some(c), Some(n)) = (
            at_f32(&info.cooldown, show_idx),
            at_f32(&info.cooldown, next_idx),
        ) {
            if (c - n).abs() > f32::EPSILON {
                delta_lines.push(format!("  冷卻 {:.1}s → {:.1}s", c, n));
            }
        }
        if let (Some(c), Some(n)) = (
            at_i32(&info.mana_cost, show_idx),
            at_i32(&info.mana_cost, next_idx),
        ) {
            if c != n {
                delta_lines.push(format!("  魔力 {} → {}", c, n));
            }
        }
        if let (Some(c), Some(n)) = (
            at_f32(&info.cast_range, show_idx),
            at_f32(&info.cast_range, next_idx),
        ) {
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
            for l in &delta_lines {
                out.push_str(l);
                out.push('\n');
            }
        }
    }

    // ===== 升級提示 =====
    out.push_str(bar);
    if cur_lvl < max {
        out.push_str(&format!(
            "Shift + {} 升級（需 1 技能點）\n",
            info.key_binding
        ));
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
    if (n - n.round()).abs() < 1e-6 {
        format!("{:.0}", n)
    } else {
        format!("{:.2}", n)
    }
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
        .set_position(Vector3::new(
            -pos_x + offset_x,
            pos_y + offset_y,
            Z_HP_BAR - 0.02,
        ))
        .set_scale(Vector3::new(length, thickness, 1.0))
        .set_rotation(rotation);
    handle
}

fn build_path_segment(
    scene: &mut Scene,
    from: Vector2<f32>,
    to: Vector2<f32>,
) -> Option<Handle<Node>> {
    build_line_segment(
        scene,
        from,
        to,
        0.05,
        Color::from_rgba(255, 100, 255, 180),
        Z_PATH,
    )
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
        let offset = Vector2::new((a_local.x + b_local.x) * 0.5, (a_local.y + b_local.y) * 0.5);
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
fn add_world_line(scene: &mut Scene, from: Vector2<f32>, to: Vector2<f32>, color: Color, z: f32) {
    use fyrox::scene::debug::Line;
    if !from.x.is_finite() || !from.y.is_finite() || !to.x.is_finite() || !to.y.is_finite() {
        return;
    }
    if (to - from).norm_squared() <= f32::EPSILON {
        return;
    }
    scene.drawing_context.add_line(Line {
        begin: Vector3::new(-from.x, from.y, z),
        end: Vector3::new(-to.x, to.y, z),
        color,
    });
}

fn add_dashed_world_line(
    scene: &mut Scene,
    from: Vector2<f32>,
    to: Vector2<f32>,
    dash_len: f32,
    gap_len: f32,
    color: Color,
    z: f32,
) {
    let delta = to - from;
    let len = delta.norm();
    if len <= 0.001 || dash_len <= 0.0 {
        return;
    }

    let dir = delta / len;
    let step = dash_len + gap_len.max(0.0);
    let mut offset = 0.0;
    while offset < len {
        let end_offset = (offset + dash_len).min(len);
        add_world_line(
            scene,
            from + dir * offset,
            from + dir * end_offset,
            color,
            z,
        );
        offset += step.max(dash_len);
    }
}

fn add_command_queue_flag(scene: &mut Scene, base: Vector2<f32>, z: f32) {
    let green = Color::from_rgba(70, 240, 120, 235);
    let pole = Color::from_rgba(35, 140, 75, 235);
    let height = 0.52;
    let width = 0.34;
    let flag_drop = 0.18;
    let top = base + Vector2::new(0.0, height);
    let outer = base + Vector2::new(width, height - flag_drop * 0.5);
    let lower = base + Vector2::new(0.0, height - flag_drop);

    add_world_line(scene, base, top, pole, z);
    add_world_line(scene, top, outer, green, z - 0.001);
    add_world_line(scene, outer, lower, green, z - 0.001);
    add_world_line(scene, lower, top, green, z - 0.001);
    add_circle_lines(scene, base, 0.07, 10, green, z - 0.001);
}

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
        let next = Vector3::new(-(center.x + radius * c), center.y + radius * s, z);
        scene.drawing_context.add_line(Line {
            begin: prev,
            end: next,
            color,
        });
        prev = next;
    }
}

// ---------------------------------------------------------------------------
// 輔助函數
// ---------------------------------------------------------------------------

fn parse_heartbeat(data: &serde_json::Value) -> HeartbeatInfo {
    HeartbeatInfo {
        tick: data.get("tick").and_then(|v| v.as_u64()).unwrap_or(0),
        game_time: data
            .get("game_time")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0),
        entity_count: data
            .get("entity_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0),
        hero_count: data.get("hero_count").and_then(|v| v.as_u64()).unwrap_or(0),
        creep_count: data
            .get("creep_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0),
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

fn world_to_screen_approx(
    wx: f32,
    wy: f32,
    window_w: f32,
    window_h: f32,
    world_height: f32,
) -> Vector2<f32> {
    let aspect = window_w / window_h;
    let world_width = world_height * aspect;
    // +X world → +X screen（camera 的 -1 X scale 已把原本的翻轉抵消）
    let sx = (wx / world_width + 0.5) * window_w;
    // +Y world → 螢幕上方（螢幕 pixel Y 向下，所以要反向）
    let sy = (-wy / world_height + 0.5) * window_h;
    Vector2::new(sx, sy)
}

/// Ray-casting 點在多邊形內判定（凹/凸皆可）。與 `omoba-core` geometry helper 同演算法。
fn point_in_polygon(p: Vector2<f32>, poly: &[Vector2<f32>]) -> bool {
    if poly.len() < 3 {
        return false;
    }
    let mut inside = false;
    let n = poly.len();
    let mut j = n - 1;
    for i in 0..n {
        let pi = poly[i];
        let pj = poly[j];
        let cond = (pi.y > p.y) != (pj.y > p.y)
            && p.x < (pj.x - pi.x) * (p.y - pi.y) / (pj.y - pi.y + f32::EPSILON) + pi.x;
        if cond {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// 點到線段 (a-b) 的最短距離平方。
fn point_segment_dist_sq(p: Vector2<f32>, a: Vector2<f32>, b: Vector2<f32>) -> f32 {
    let ab = b - a;
    let ap = p - a;
    let len_sq = ab.x * ab.x + ab.y * ab.y;
    if len_sq < 1e-8 {
        return ap.norm_squared();
    }
    let t = (ap.x * ab.x + ap.y * ab.y) / len_sq;
    let t = t.clamp(0.0, 1.0);
    let proj = a + ab * t;
    (p - proj).norm_squared()
}

/// 圓 vs 多邊形：圓心在內 → true；或任一邊距圓心 < r → true。
fn circle_hits_polygon(center: Vector2<f32>, r: f32, poly: &[Vector2<f32>]) -> bool {
    if poly.len() < 3 {
        return false;
    }
    if point_in_polygon(center, poly) {
        return true;
    }
    let r2 = r * r;
    let n = poly.len();
    for i in 0..n {
        let a = poly[i];
        let b = poly[(i + 1) % n];
        if point_segment_dist_sq(center, a, b) < r2 {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod input_latency_tests {
    use super::*;

    fn ability_entity(
        entity_id: u32,
        spawn_order: u64,
        unit_id: &str,
        display_name: &str,
        icon: &str,
        cooldown_remaining: f32,
    ) -> sim_runner::EntityRenderData {
        sim_runner::EntityRenderData {
            entity_id,
            spawn_order,
            kind: sim_runner::EntityKind::Tower,
            unit_id: unit_id.into(),
            hp: 100,
            max_hp: 100,
            owner_player_id: Some(7),
            tower_active_ability: Some(sim_runner::TowerActiveAbilitySnapshot {
                ability_id: format!("ability_{entity_id}"),
                display_name: display_name.into(),
                description: "description".into(),
                icon: icon.into(),
                cooldown_total: 10.0,
                duration_total: 5.0,
                cooldown_remaining,
                active_remaining: 0.0,
                activation_serial: 0,
            }),
            ..Default::default()
        }
    }

    #[test]
    fn ability_bar_orders_by_spawn_then_entity_and_limits_shortcuts() {
        let entities = (0..9)
            .rev()
            .map(|i| ability_entity(900 - i, i as u64, "tower_dart", "Volley", "icon.png", 0.0))
            .collect::<Vec<_>>();

        let items = ability_bar_items(&entities, 7, 0, 0.0);

        assert_eq!(items.len(), 6);
        assert_eq!(items[0].key.tower_entity_id, 900);
        assert_eq!(items[1].key.tower_entity_id, 899);
        assert_eq!(items[0].shortcut, Some(1));
        assert_eq!(items[5].shortcut, Some(6));
        let second_page = ability_bar_items(&entities, 7, 1, 0.0);
        assert_eq!(second_page.len(), 3);
        assert_eq!(second_page[0].shortcut, Some(1));
    }

    #[test]
    fn ability_bar_formats_interpolated_countdown_to_one_decimal() {
        let entities = vec![ability_entity(1, 0, "tower_arty", "Fire", "icon.png", 2.04)];

        let items = ability_bar_items(&entities, 7, 0, 0.55);

        assert_eq!(items[0].cooldown_text, "1.5");
    }

    #[test]
    fn ability_bar_elapsed_sim_freezes_while_paused() {
        assert_eq!(ability_bar_elapsed_sim(1.25, true, 2), 0.0);
    }

    #[test]
    fn tower_ability_bar_uses_eighty_eight_pixel_bottom_margin() {
        assert_eq!(tower_ability_bar_y(1080.0, 88.0), 904.0);
        assert_eq!(tower_ability_bar_y(720.0, 88.0), 544.0);
        assert_eq!(tower_ability_bar_y(120.0, 88.0), 0.0);
    }

    #[test]
    fn ability_bar_elapsed_sim_tracks_wall_clock_at_one_x() {
        assert_eq!(ability_bar_elapsed_sim(1.25, false, 1), 1.25);
    }

    #[test]
    fn ability_bar_elapsed_sim_scales_wall_clock_at_two_x() {
        assert_eq!(ability_bar_elapsed_sim(1.25, false, 2), 2.5);
    }

    #[test]
    fn ability_bar_authoritative_snapshot_corrects_interpolated_countdown() {
        let before = vec![ability_entity(1, 0, "tower_arty", "Fire", "icon.png", 8.0)];
        let interpolated = ability_bar_items(&before, 7, 0, ability_bar_elapsed_sim(1.0, false, 2));
        assert_eq!(interpolated[0].cooldown_remaining, 6.0);

        let corrected = vec![ability_entity(1, 0, "tower_arty", "Fire", "icon.png", 5.5)];
        let items = ability_bar_items(&corrected, 7, 0, ability_bar_elapsed_sim(0.0, false, 2));
        assert_eq!(items[0].cooldown_remaining, 5.5);
    }

    #[test]
    fn ability_bar_suffixes_duplicate_tower_names() {
        let entities = vec![
            ability_entity(1, 0, "tower_arty", "Fire", "icon.png", 0.0),
            ability_entity(2, 1, "tower_arty", "Fire", "icon.png", 0.0),
        ];

        let tower_names = HashMap::from([("tower_arty".to_string(), "Artillery".to_string())]);
        let items = ability_bar_items_with_names(&entities, &tower_names, 7, 0, 0.0);

        assert_eq!(items[0].tower_label, "Artillery #1");
        assert_eq!(items[1].tower_label, "Artillery #2");
        assert_eq!(items[0].ability_name, "Fire");
        assert_eq!(items[1].ability_name, "Fire");
    }

    #[test]
    fn ability_bar_visual_state_distinguishes_ready_cooling_and_active() {
        assert_eq!(
            ability_bar_visual_state(10.0, 0.0, 0.0),
            AbilityBarVisualState::Ready
        );
        assert_eq!(
            ability_bar_visual_state(10.0, 2.5, 0.0),
            AbilityBarVisualState::Cooling {
                fraction: 0.25,
                text: "2.5".into(),
            }
        );
        assert_eq!(
            ability_bar_visual_state(10.0, 8.0, 1.25),
            AbilityBarVisualState::Active {
                cooldown_fraction: 0.8,
                active_remaining: 1.25,
                text: "ACTIVE 1.3".into(),
            }
        );
    }

    #[test]
    fn ability_bar_page_model_exposes_controls_and_indicator() {
        assert_eq!(
            ability_bar_page_model(8, 0),
            AbilityBarPageModel {
                page: 0,
                page_count: 2,
                previous_enabled: false,
                next_enabled: true,
                indicator: "1 / 2".into(),
            }
        );
        assert_eq!(ability_bar_page_model(8, 1).indicator, "2 / 2");
    }

    #[test]
    fn ability_bar_paging_preserves_debounce_until_authoritative_boundary() {
        let key = AbilityBarKey {
            tower_entity_id: 42,
            ability_id: "turbo".into(),
        };
        let mut state = AbilityBarInteractionState::default();
        state.debounced.insert(key.clone());

        state.change_page(1, 2);
        assert_eq!(state.page, 1);
        assert!(state.debounced.contains(&key));

        state.authoritative_boundary();
        assert!(state.debounced.is_empty());
    }

    #[test]
    fn ability_bar_slot_reconciliation_preserves_handles_for_stable_keys() {
        let key = |tower_entity_id| AbilityBarKey {
            tower_entity_id,
            ability_id: format!("ability_{tower_entity_id}"),
        };
        let mut bindings = vec![Some(key(1)), Some(key(2)), Some(key(3)), None, None, None];

        reconcile_ability_bar_slots(&mut bindings, &[key(2), key(3), key(4)]);

        assert_eq!(bindings[1], Some(key(2)));
        assert_eq!(bindings[2], Some(key(3)));
        assert_eq!(bindings[0], Some(key(4)));
        assert_eq!(bindings.iter().flatten().count(), 3);
    }

    #[test]
    fn ability_bar_button_contains_only_the_ability_name() {
        assert_eq!(ability_bar_button_text("甜點狂歡"), "甜點狂歡");
    }

    #[test]
    fn ability_bar_button_wraps_a_long_name_into_two_complete_lines() {
        let text = ability_bar_button_text("超級猴子粉絲俱樂部");

        assert_eq!(text.lines().count(), 2);
        assert_eq!(text.replace('\n', ""), "超級猴子粉絲俱樂部");
    }

    #[test]
    fn ability_bar_ready_tooltip_contains_authored_details_and_shortcut() {
        let entities = vec![ability_entity(
            1,
            0,
            "tower_cake_splash",
            "甜點狂歡",
            "party.png",
            0.0,
        )];
        let tower_names =
            HashMap::from([("tower_cake_splash".to_string(), "蛋糕濺射塔".to_string())]);
        let item = ability_bar_items_with_names(&entities, &tower_names, 7, 0, 0.0).remove(0);
        let tooltip = ability_bar_tooltip_model(Some(&item), None).unwrap();

        assert_eq!(tooltip.title, "甜點狂歡");
        assert!(tooltip.description.contains("蛋糕濺射塔"));
        assert!(tooltip.description.contains("description"));
        assert!(tooltip.description.contains("冷卻：10.0 秒"));
        assert!(tooltip.description.contains("持續：5.0 秒"));
        assert!(tooltip.description.contains("狀態：準備完成"));
        assert!(tooltip.description.contains("快捷鍵：1"));
    }

    #[test]
    fn ability_bar_tooltip_reports_active_and_cooling_remaining_time() {
        let mut active = ability_entity(1, 0, "tower_arty", "火力全開", "", 8.0);
        active
            .tower_active_ability
            .as_mut()
            .unwrap()
            .active_remaining = 2.5;
        let active_item = ability_bar_items(&[active], 7, 0, 0.0).remove(0);
        let active_tooltip = ability_bar_tooltip_model(Some(&active_item), None).unwrap();
        assert!(active_tooltip
            .description
            .contains("狀態：施放中，剩餘 2.5 秒"));

        let cooling = ability_entity(2, 0, "tower_bomb", "飛艇刺客", "", 7.5);
        let cooling_item = ability_bar_items(&[cooling], 7, 0, 0.0).remove(0);
        let cooling_tooltip = ability_bar_tooltip_model(Some(&cooling_item), None).unwrap();
        assert!(cooling_tooltip
            .description
            .contains("狀態：冷卻中，剩餘 7.5 秒"));
    }

    #[test]
    fn ability_bar_tooltip_sanitizes_missing_copy_and_invalid_timing() {
        let mut entity = ability_entity(1, 0, "tower_ice", "絕對零度", "", -4.0);
        let ability = entity.tower_active_ability.as_mut().unwrap();
        ability.description.clear();
        ability.cooldown_total = f32::NAN;
        ability.duration_total = f32::INFINITY;
        ability.active_remaining = -2.0;

        let item = ability_bar_items(&[entity], 7, 0, 0.0).remove(0);
        let tooltip = ability_bar_tooltip_model(Some(&item), None).unwrap();

        assert_eq!(item.cooldown_total, 0.0);
        assert_eq!(item.duration_total, 0.0);
        assert_eq!(item.cooldown_remaining, 0.0);
        assert_eq!(item.active_remaining, 0.0);
        assert!(tooltip.description.contains("沒有可用的技能描述。"));
        assert!(tooltip.description.contains("冷卻：0.0 秒"));
        assert!(tooltip.description.contains("持續：瞬發"));
    }

    #[test]
    fn ability_bar_tooltip_includes_only_matching_rejection() {
        let item = ability_bar_items(
            &[ability_entity(1, 0, "tower_tack", "鐵釘風暴", "", 0.0)],
            7,
            0,
            0.0,
        )
        .remove(0);
        let matching = AbilityBarRejection {
            key: item.key.clone(),
            reason: "cooldown".into(),
            shown_at: Instant::now(),
        };
        let other = AbilityBarRejection {
            key: AbilityBarKey {
                tower_entity_id: 99,
                ability_id: "other".into(),
            },
            reason: "wrong rejection".into(),
            shown_at: Instant::now(),
        };

        assert!(ability_bar_tooltip_model(Some(&item), Some(&matching))
            .unwrap()
            .description
            .contains("上次施放失敗：cooldown"));
        assert!(!ability_bar_tooltip_model(Some(&item), Some(&other))
            .unwrap()
            .description
            .contains("wrong rejection"));
        assert_eq!(ability_bar_tooltip_model(None, Some(&matching)), None);
    }

    #[test]
    fn ability_bar_removes_tower_missing_from_authoritative_snapshot() {
        let entities = vec![ability_entity(1, 0, "tower_arty", "Fire", "icon.png", 0.0)];
        assert_eq!(ability_bar_items(&entities, 7, 0, 0.0).len(), 1);

        assert!(ability_bar_items(&[], 7, 0, 0.0).is_empty());
    }

    #[test]
    fn ability_bar_filters_to_local_owner() {
        let mut foreign = ability_entity(1, 0, "tower_arty", "Fire", "icon.png", 0.0);
        foreign.owner_player_id = Some(8);

        assert!(ability_bar_items(&[foreign], 7, 0, 0.0).is_empty());
    }

    fn rejected_cast_result(
        player_id: u32,
        tower_entity_id: u32,
        result_serial: u32,
    ) -> sim_runner::TowerAbilityCastResultSnapshot {
        sim_runner::TowerAbilityCastResultSnapshot {
            player_id,
            tower_entity_id,
            ability_id: format!("ability_{tower_entity_id}"),
            accepted: false,
            reason: "cooldown_active".into(),
            result_serial,
        }
    }

    #[test]
    fn ability_bar_rejection_selects_local_players_result() {
        let now = Instant::now();
        let mut state = AbilityBarRejectionState::default();
        state.observe_results(
            &[
                rejected_cast_result(8, 80, 1),
                rejected_cast_result(7, 70, 2),
            ],
            7,
            now,
        );

        assert_eq!(state.visible.as_ref().unwrap().key.tower_entity_id, 70);
    }

    #[test]
    fn ability_bar_rejection_survives_ordinary_authoritative_snapshots() {
        let now = Instant::now();
        let result = rejected_cast_result(7, 70, 2);
        let key = AbilityBarKey {
            tower_entity_id: 70,
            ability_id: "ability_70".into(),
        };
        let mut state = AbilityBarRejectionState::default();
        state.observe_results(std::slice::from_ref(&result), 7, now);

        for seconds in [0.5, 1.0, 2.9] {
            state.observe_results(
                std::slice::from_ref(&result),
                7,
                now + Duration::from_secs_f32(seconds),
            );
            state.retain_visible(
                std::slice::from_ref(&key),
                now + Duration::from_secs_f32(seconds),
            );
            assert!(state.visible.is_some(), "rejection hidden at {seconds}s");
        }
    }

    #[test]
    fn ability_bar_rejection_expires_after_three_wall_clock_seconds() {
        let now = Instant::now();
        let mut state = AbilityBarRejectionState::default();
        state.observe_results(&[rejected_cast_result(7, 70, 2)], 7, now);
        state.retain_visible(
            &[AbilityBarKey {
                tower_entity_id: 70,
                ability_id: "ability_70".into(),
            }],
            now + Duration::from_secs_f32(3.0),
        );

        assert!(state.visible.is_none());
    }

    #[test]
    fn ability_bar_accepted_result_clears_visible_rejection_before_timeout() {
        let now = Instant::now();
        let mut state = AbilityBarRejectionState::default();
        state.observe_results(&[rejected_cast_result(7, 70, 2)], 7, now);
        assert!(state.visible.is_some());

        let mut accepted = rejected_cast_result(7, 70, 3);
        accepted.accepted = true;
        accepted.reason.clear();
        state.observe_results(&[accepted], 7, now + Duration::from_secs(1));

        assert_eq!(state.last_result_serial, 3);
        assert!(state.visible.is_none());
    }

    #[test]
    fn ability_bar_rejection_clears_when_keyed_tower_disappears() {
        let now = Instant::now();
        let mut state = AbilityBarRejectionState::default();
        state.observe_results(&[rejected_cast_result(7, 70, 2)], 7, now);
        state.retain_visible(&[], now + Duration::from_millis(100));

        assert!(state.visible.is_none());
    }

    fn sample(input_id: u32, total_ms: u32) -> LatencySample {
        LatencySample {
            input_id,
            action_kind: InputActionKind::NoOp,
            total_ms,
            submitted_at: Instant::now(),
            origin_kind: lockstep_client::InputOriginKind::Direct,
            target_tick: input_id,
            server_receive_tick: None,
            server_drain_tick: None,
            phases: LatencyPhaseDurations::default(),
        }
    }

    fn pending_input(submit_wall_clock_us: u64, action_kind: InputActionKind) -> PendingInput {
        PendingInput {
            submit_wall_clock_us,
            submit_instant: Instant::now(),
            base_tick: 22,
            target_tick: 24,
            action_kind,
            origin_kind: lockstep_client::InputOriginKind::Direct,
            origin_us: submit_wall_clock_us.saturating_sub(100),
            send_lockstep_input_us: submit_wall_clock_us,
            submit_start_us: None,
            submit_done_us: None,
            client_receive_tickbatch_us: None,
            game_forward_to_sim_us: None,
            extract_data_for_render_us: None,
            server_receive_tick: None,
            server_drain_tick: None,
            server_queue_us: None,
        }
    }

    #[test]
    fn input_action_kind_maps_debug_spawn_creep() {
        use omoba_core::kcp::game_proto::{player_input::Action, DebugSpawnCreep, PlayerInput};

        let input = PlayerInput {
            action: Some(Action::DebugSpawnCreep(DebugSpawnCreep {
                emitter_index: 2,
                count: 5,
            })),
        };

        assert_eq!(
            InputActionKind::from_player_input(&input),
            InputActionKind::DebugSpawnCreep
        );
    }

    fn sample_tower_template(
        cost: i32,
        range: f32,
        base_image: &str,
    ) -> sim_runner::TowerTemplateSnapshot {
        sim_runner::TowerTemplateSnapshot {
            unit_id: "tower_dart".to_string(),
            label: "Dart".to_string(),
            cost,
            footprint: 10.0,
            placement_radius: 90.0,
            range,
            splash_radius: 0.0,
            hit_radius: 0.0,
            slow_factor: 0.0,
            slow_duration: 0.0,
            render_mode: "base_barrel".to_string(),
            base_image: base_image.to_string(),
            barrel_image: "assets/towers/tower_dart_barrel.png".to_string(),
            render_visual_size: 180.0,
            barrel_frames: vec!["assets/towers/tower_dart_barrel_frame_01.png".to_string()],
            body_frames: Vec::new(),
            barrel_animation: sim_runner::TowerRenderAnimationSnapshot::default(),
            body_animation: sim_runner::TowerRenderAnimationSnapshot::default(),
            rotation_mode: "targeted".to_string(),
            barrel_layout: "single".to_string(),
            barrel_variants: Vec::new(),
            barrel_offset: sim_runner::TowerRenderPointSnapshot { x: 0.0, y: -6.0 },
            barrel_pivot: sim_runner::TowerRenderPointSnapshot { x: 0.5, y: 0.66 },
            muzzle_offset: sim_runner::TowerRenderPointSnapshot { x: 0.0, y: -30.0 },
            default_angle_deg: 0.0,
            recoil: sim_runner::TowerRecoilSnapshot::default(),
            attack_windup: 350,
            attack_backswing: 650,
        }
    }

    #[test]
    fn td_template_from_snapshot_projects_reload_sensitive_fields() {
        let snapshot = sample_tower_template(275, 420.0, "assets/towers/new_base.png");
        let template = td_template_from_snapshot(&snapshot);

        assert_eq!(template.cost, 275);
        assert_eq!(template.range_backend, 420.0);
        assert_eq!(template.base_image, "assets/towers/new_base.png");
    }

    fn backend_to_frontend_world(x: f32, y: f32) -> Vector2<f32> {
        Vector2::new(x * WORLD_SCALE, y * WORLD_SCALE)
    }

    fn twin_gate_render_paths() -> Vec<Vec<Vector2<f32>>> {
        vec![
            vec![
                backend_to_frontend_world(-1400.0, 700.0),
                backend_to_frontend_world(-350.0, 520.0),
                backend_to_frontend_world(850.0, -200.0),
                backend_to_frontend_world(1400.0, 100.0),
            ],
            vec![
                backend_to_frontend_world(-1400.0, 100.0),
                backend_to_frontend_world(-500.0, 100.0),
                backend_to_frontend_world(850.0, -200.0),
                backend_to_frontend_world(1400.0, 100.0),
            ],
            vec![
                backend_to_frontend_world(-1400.0, -650.0),
                backend_to_frontend_world(-500.0, -650.0),
                backend_to_frontend_world(850.0, -200.0),
                backend_to_frontend_world(1400.0, 100.0),
            ],
        ]
    }

    #[test]
    fn twin_gate_reported_green_gap_accepts_tower_placement() {
        let template =
            td_template_from_snapshot(&sample_tower_template(200, 350.0, "base.png"));
        let mut game = Game::default();
        game.hero_state.gold = 1_000;
        game.td_paths_render = twin_gate_render_paths();

        let reason = game
            .tower_placement_block_reason(
                &template,
                backend_to_frontend_world(-420.0, -250.0),
            );

        assert_eq!(reason, None);
    }

    #[test]
    fn twin_gate_frontend_matches_backend_for_reported_blocked_point() {
        let template =
            td_template_from_snapshot(&sample_tower_template(200, 350.0, "base.png"));
        let mut game = Game::default();
        game.hero_state.gold = 1_000;
        game.td_paths_render = twin_gate_render_paths();
        let point = backend_to_frontend_world(270.7, -366.7);

        let probe = game
            .nearest_tower_road_probe(&template, point)
            .expect("nearest Twin Gate road");
        assert_eq!(probe.path_index, 2);
        assert_eq!(probe.segment_index, 1);
        assert!((probe.distance_backend - 25.07).abs() < 0.1);
        assert_eq!(
            game.tower_placement_block_reason(&template, point),
            Some(TowerPlacementBlockReason::TooCloseToRoad)
        );
    }

    #[test]
    fn tower_placement_reports_road_and_gold_rejections() {
        let template =
            td_template_from_snapshot(&sample_tower_template(200, 350.0, "base.png"));
        let mut game = Game::default();
        game.hero_state.gold = 1_000;
        game.td_paths_render = twin_gate_render_paths();
        assert_eq!(
            game.tower_placement_block_reason(
                &template,
                backend_to_frontend_world(-1400.0, 100.0),
            ),
            Some(TowerPlacementBlockReason::TooCloseToRoad)
        );

        game.hero_state.gold = 50;
        assert_eq!(
            game.tower_placement_block_reason(
                &template,
                backend_to_frontend_world(-420.0, -250.0),
            ),
            Some(TowerPlacementBlockReason::InsufficientGold {
                required: 200,
                available: 50,
            })
        );
    }

    #[test]
    fn tower_road_clearance_uses_visual_placement_radius() {
        let template =
            td_template_from_snapshot(&sample_tower_template(200, 350.0, "base.png"));
        let mut game = Game::default();
        game.hero_state.gold = 1_000;
        game.td_paths_render = vec![vec![
            backend_to_frontend_world(-500.0, 0.0),
            backend_to_frontend_world(500.0, 0.0),
        ]];

        let probe = game
            .nearest_tower_road_probe(
                &template,
                backend_to_frontend_world(
                    0.0,
                    TD_PATH_HALF_WIDTH_BACKEND + template.placement_radius_backend + 1.0,
                ),
            )
            .expect("road probe");
        assert!((probe.distance_backend - 155.0).abs() < 0.01);
        assert!((probe.required_clearance_backend - 154.0).abs() < 0.01);

        assert_eq!(
            game.tower_placement_block_reason(
                &template,
                backend_to_frontend_world(
                    0.0,
                    TD_PATH_HALF_WIDTH_BACKEND + template.placement_radius_backend - 1.0,
                ),
            ),
            Some(TowerPlacementBlockReason::TooCloseToRoad)
        );
        assert_eq!(
            game.tower_placement_block_reason(
                &template,
                backend_to_frontend_world(
                    0.0,
                    TD_PATH_HALF_WIDTH_BACKEND + template.placement_radius_backend + 1.0,
                ),
            ),
            None
        );
    }

    #[test]
    fn tower_overlap_still_uses_placement_radius() {
        let snapshot = sample_tower_template(200, 350.0, "base.png");
        let template = td_template_from_snapshot(&snapshot);
        let mut game = Game::default();
        game.hero_state.gold = 1_000;
        game.td_templates
            .insert(snapshot.unit_id.clone(), template.clone());
        game.network_entities.insert(
            1,
            NetworkEntity {
                entity_type: "tower".to_string(),
                tower_kind: Some(snapshot.unit_id),
                position: backend_to_frontend_world(0.0, 0.0),
                ..NetworkEntity::default()
            },
        );

        assert_eq!(
            game.tower_placement_block_reason(
                &template,
                backend_to_frontend_world(template.placement_radius_backend * 1.5, 0.0),
            ),
            Some(TowerPlacementBlockReason::TowerOverlap)
        );
    }

    #[test]
    fn clear_lua_metadata_caches_drops_frontend_asset_keys() {
        let mut game = Game::default();
        let snapshot = sample_tower_template(200, 350.0, "assets/towers/old_base.png");
        game.td_templates.insert(
            snapshot.unit_id.clone(),
            td_template_from_snapshot(&snapshot),
        );
        game.td_template_order.push(snapshot.unit_id.clone());
        game.td_upgrade_defs.insert(
            (snapshot.unit_id.clone(), 1, 1),
            ("Sharp".into(), "More damage".into(), 100),
        );
        game.ability_info_map.insert(
            "ability_test".into(),
            AbilityInfo {
                id: "ability_test".into(),
                icon_path: "assets/abilities/old.png".into(),
                ..Default::default()
            },
        );
        game.ability_icon_texture_cache
            .insert("assets/abilities/old.png".into(), None);
        game.ability_icon_paths = std::array::from_fn(|_| "assets/abilities/old.png".into());
        game.tower_texture_cache
            .insert("assets/towers/old_base.png".into(), None);
        game.tower_material_cache
            .insert("assets/towers/old_base.png".into(), None);

        game.clear_lua_metadata_caches();

        assert!(game.td_templates.is_empty());
        assert!(game.td_template_order.is_empty());
        assert!(game.td_upgrade_defs.is_empty());
        assert!(game.ability_info_map.is_empty());
        assert!(game.ability_icon_texture_cache.is_empty());
        assert!(game.ability_icon_paths.iter().all(String::is_empty));
        assert!(game.tower_texture_cache.is_empty());
        assert!(game.tower_material_cache.is_empty());
    }

    #[test]
    fn tower_action_guard_allows_only_local_owner() {
        assert!(tower_owned_by_local(Some(1), 1));
        assert!(tower_owned_by_local(Some(2), 2));
        assert!(!tower_owned_by_local(Some(2), 1));
        assert!(!tower_owned_by_local(Some(1), 2));
        assert!(!tower_owned_by_local(None, 1));
    }

    #[test]
    fn input_latency_meter_caps_to_recent_capacity_samples() {
        let mut meter = InputLatencyMeter::default();
        for i in 0..(INPUT_LATENCY_CAPACITY as u32 + 80) {
            meter.push(sample(i, i));
        }
        assert_eq!(meter.samples.len(), INPUT_LATENCY_CAPACITY);
        assert_eq!(meter.samples.front().unwrap().input_id, 80);
    }

    #[test]
    fn input_latency_meter_first_sample_populates_hud_cache() {
        let mut meter = InputLatencyMeter::default();
        meter.push(sample(1, 123));
        assert_eq!(meter.cached_p50_ms, 123);
        assert_eq!(meter.cached_p99_ms, 123);
        assert_eq!(meter.cached_max_ms, 123);
        assert_eq!(meter.cached_latest_ms, 123);
    }

    #[test]
    fn input_latency_meter_recomputes_p50_p99_max_latest() {
        let mut meter = InputLatencyMeter::default();
        for i in 0..100 {
            meter.push(sample(i, i));
        }
        let now = meter.last_compute_at + Duration::from_secs(1);
        meter.maybe_recompute(now);
        assert_eq!(meter.cached_p50_ms, 50);
        assert_eq!(meter.cached_p99_ms, 99);
        assert_eq!(meter.cached_max_ms, 99);
        assert_eq!(meter.cached_latest_ms, 99);
    }

    #[test]
    fn frame_time_summary_reports_one_percent_low_from_slowest_frames() {
        let samples = vec![8.0; 118]
            .into_iter()
            .chain([10.0, 12.0])
            .collect::<Vec<_>>();

        let (_p50, _p95, _p99, one_pct_low_fps) = frame_time_summary(&samples);

        // 1% of 120 frames uses the two slowest frames: average 11 ms.
        assert!((one_pct_low_fps - (1000.0 / 11.0)).abs() < 0.001);
    }

    #[test]
    fn input_lookahead_stays_low_latency_at_supported_fps() {
        assert_eq!(input_lookahead_ticks(LockstepTiming::new(120).unwrap()), 2);
        assert_eq!(input_lookahead_ticks(LockstepTiming::new(90).unwrap()), 2);
        assert_eq!(input_lookahead_ticks(LockstepTiming::new(60).unwrap()), 2);
    }

    #[test]
    fn lag_status_exposes_pending_input_when_it_exceeds_p99() {
        let mut meter = InputLatencyMeter::default();
        for i in 0..47 {
            meter.push(sample(i, 46));
        }
        let now = meter.last_compute_at + Duration::from_secs(1);
        meter.maybe_recompute(now);

        let lag = format_input_lag_status(&meter, Some(1_000));

        assert!(lag.contains("p50 46 / p99 46 ms"));
        assert!(lag.contains("pending 1000 ms"));
    }

    #[test]
    fn lag_status_keeps_paired_only_when_pending_is_lower_than_p99() {
        let mut meter = InputLatencyMeter::default();
        meter.push(sample(1, 80));

        let lag = format_input_lag_status(&meter, Some(40));

        assert_eq!(lag, "p50 80 / p99 80 ms");
    }

    #[test]
    fn oldest_pending_input_age_uses_max_pending_age() {
        let mut pending = HashMap::new();
        pending.insert(1, pending_input(10_000, InputActionKind::MoveTo));
        pending.insert(2, pending_input(11_500, InputActionKind::NoOp));

        assert_eq!(oldest_pending_input_age_ms(&pending, 12_000), Some(2));
    }

    #[test]
    fn stale_pending_diagnostic_keeps_last_known_phase() {
        let mut pending = pending_input(1_000, InputActionKind::MoveTo);
        pending.submit_start_us = Some(1_010);
        pending.submit_done_us = Some(1_020);
        pending.client_receive_tickbatch_us = Some(2_000);
        pending.game_forward_to_sim_us = Some(2_100);
        pending.server_receive_tick = Some(22);
        pending.server_drain_tick = Some(24);
        pending.server_queue_us = Some(16_000);

        let diag = pending_input_diagnostic(42, &pending, 2_500_000);

        assert_eq!(diag.input_id, 42);
        assert_eq!(diag.action_kind, InputActionKind::MoveTo);
        assert_eq!(diag.base_tick, 22);
        assert_eq!(diag.target_tick, 24);
        assert_eq!(diag.pending_age_ms, 2_499);
        assert!(diag.has_submit_start);
        assert!(diag.has_submit_done);
        assert!(diag.has_client_receive_tickbatch);
        assert!(diag.has_game_forward_to_sim);
        assert!(!diag.has_extract_data_for_render);
        assert_eq!(diag.server_receive_tick, Some(22));
        assert_eq!(diag.server_drain_tick, Some(24));
        assert_eq!(diag.server_queue_us, Some(16_000));
    }

    #[test]
    fn stale_pending_evicts_without_adding_paired_sample() {
        let mut game = Game::default();
        let now_us = wall_clock_us();
        game.pending_inputs.insert(
            42,
            pending_input(
                now_us.saturating_sub((PENDING_INPUT_MAX_AGE_MS + 500) * 1_000),
                InputActionKind::MoveTo,
            ),
        );

        game.evict_stale_pending_inputs();

        assert!(game.pending_inputs.is_empty());
        assert_eq!(game.pending_inputs_stale, 1);
        assert_eq!(game.pending_inputs_evicted, 1);
        assert!(game.input_latency_meter.samples.is_empty());
    }

    #[test]
    fn latency_sample_preserves_phase_breakdown() {
        let mut s = sample(7, 42);
        s.origin_kind = lockstep_client::InputOriginKind::OsEvent;
        s.server_receive_tick = Some(10);
        s.server_drain_tick = Some(12);
        s.phases.server_queue_us = 15_000;
        s.phases.forward_to_extract_data_for_render_us = 800;

        assert_eq!(s.origin_kind, lockstep_client::InputOriginKind::OsEvent);
        assert_eq!(s.server_receive_tick, Some(10));
        assert_eq!(s.server_drain_tick, Some(12));
        assert_eq!(s.phases.server_queue_us, 15_000);
        assert_eq!(s.phases.forward_to_extract_data_for_render_us, 800);
    }

    #[test]
    fn pending_input_records_auto_origin_and_server_metadata() {
        let pending = PendingInput {
            submit_wall_clock_us: 1_000,
            submit_instant: Instant::now(),
            base_tick: 22,
            target_tick: 24,
            action_kind: InputActionKind::NoOp,
            origin_kind: lockstep_client::InputOriginKind::Auto,
            origin_us: 900,
            send_lockstep_input_us: 1_000,
            submit_start_us: Some(1_010),
            submit_done_us: Some(1_020),
            client_receive_tickbatch_us: Some(2_000),
            game_forward_to_sim_us: Some(2_100),
            extract_data_for_render_us: Some(2_200),
            server_receive_tick: Some(22),
            server_drain_tick: Some(24),
            server_queue_us: Some(16_000),
        };

        assert_eq!(pending.origin_kind, lockstep_client::InputOriginKind::Auto);
        assert_eq!(pending.origin_us, 900);
        assert_eq!(pending.base_tick, 22);
        assert_eq!(pending.server_drain_tick, Some(24));
        assert_eq!(pending.server_queue_us, Some(16_000));
    }

    #[test]
    fn ability_key_index_uses_wert_and_excludes_q() {
        use fyrox::keyboard::KeyCode;

        assert_eq!(ability_key_index(KeyCode::KeyW), Some(0));
        assert_eq!(ability_key_index(KeyCode::KeyE), Some(1));
        assert_eq!(ability_key_index(KeyCode::KeyR), Some(2));
        assert_eq!(ability_key_index(KeyCode::KeyT), Some(3));
        assert_eq!(ability_key_index(KeyCode::KeyQ), None);
    }

    #[test]
    fn game_defaults_to_pregame_without_runtime_resources() {
        let game = Game::default();

        assert!(matches!(
            game.pregame_runtime.state,
            pregame::PregameState::MainMenu
        ));
        assert!(game.lockstep_handle.is_none());
        assert!(game.sim_runner_handle.is_none());
        assert!(game.backend_session.is_none());
    }

    #[test]
    fn pregame_button_model_is_catalog_driven_for_each_screen() {
        let mut game = Game::default();
        game.pregame_runtime =
            pregame::PregameRuntime::new_for_menu(pregame::PregameCatalog::fallback());

        let main = game.current_pregame_buttons();
        assert!(main.iter().any(|(label, _, active, action)| {
            label == "開始"
                && *active
                && matches!(action, pregame::PregameAction::Navigate { target } if target == "difficulty_select")
        }));

        game.pregame_runtime.state = pregame::PregameState::DifficultySelect;
        let difficulties = game.current_pregame_buttons();
        assert!(matches!(difficulties[0].3, pregame::PregameAction::Back));
        assert_eq!(difficulties[0].0, "返回");
        assert!(difficulties.iter().any(|(label, _, active, action)| {
            label == "初級"
                && *active
                && matches!(action, pregame::PregameAction::SelectDifficulty { difficulty_id } if difficulty_id == "novice")
        }));

        game.pregame_runtime.selected_difficulty = Some(
            game.pregame_runtime
                .catalog
                .difficulty("novice")
                .unwrap()
                .clone(),
        );
        game.pregame_runtime.state = pregame::PregameState::MapSelect;
        let maps = game.current_pregame_buttons();
        assert!(matches!(maps[0].3, pregame::PregameAction::Back));
        assert_eq!(maps[0].0, "返回");
        assert!(maps.iter().any(|(label, _, active, action)| {
            label == "綠野路口"
                && *active
                && matches!(action, pregame::PregameAction::SelectMap { map_id } if map_id == "td_green_crossroads")
        }));
        assert!(maps.iter().any(|(label, _, active, action)| {
            label == "高級"
                && *active
                && matches!(action, pregame::PregameAction::SelectDifficulty { difficulty_id } if difficulty_id == "advanced")
        }));

        game.pregame_runtime.selected_difficulty = Some(
            game.pregame_runtime
                .catalog
                .difficulty("intermediate")
                .unwrap()
                .clone(),
        );
        game.pregame_runtime.state = pregame::PregameState::MapSelect;
        let maps = game.current_pregame_buttons();
        let map_labels = maps
            .iter()
            .filter(|(_, _, _, action)| matches!(action, pregame::PregameAction::SelectMap { .. }))
            .map(|(label, _, _, _)| label.as_str())
            .collect::<Vec<_>>();
        assert_eq!(map_labels, vec!["雙門哨站", "潮汐港灣", "礦坑迴廊"]);
        assert!(!maps.iter().any(|(label, _, _, _)| label == "綠野路口"));
    }

    #[test]
    fn default_session_selection_uses_catalog_entries() {
        let mut game = Game::default();
        game.pregame_runtime =
            pregame::PregameRuntime::new_for_menu(pregame::PregameCatalog::fallback());

        let selection = game
            .default_session_selection()
            .expect("fallback catalog has playable selection");

        assert_eq!(selection.map.id, "td_green_crossroads");
        assert_eq!(selection.map.story_id(), "TD_GREEN_CROSSROADS");
        assert_eq!(selection.difficulty.id, "novice");
        assert_eq!(selection.difficulty.config_value(), "novice");
    }

    #[test]
    fn session_backend_config_uses_selected_catalog_metadata() {
        let mut game = Game::default();
        game.pregame_runtime =
            pregame::PregameRuntime::new_for_menu(pregame::PregameCatalog::fallback());
        let selection = game.default_session_selection().unwrap();

        let config = game.build_backend_launch_config(&selection, "session-test".to_string());

        assert_eq!(config.session_id, "session-test");
        assert_eq!(config.map_id, "td_green_crossroads");
        assert_eq!(config.story, "TD_GREEN_CROSSROADS");
        assert_eq!(config.difficulty_id, "novice");
        assert_eq!(config.difficulty_config, "novice");
        assert!(!config.kcp_addr.trim().is_empty());
    }

    #[test]
    fn pregame_click_dispatch_consumes_menu_input_without_starting_session() {
        let mut game = Game::default();
        game.pregame_runtime =
            pregame::PregameRuntime::new_for_menu(pregame::PregameCatalog::fallback());
        game.pregame_button_rects.push((
            UiRect {
                x: 10.0,
                y: 20.0,
                w: 100.0,
                h: 50.0,
            },
            pregame::PregameAction::Navigate {
                target: "difficulty_select".to_string(),
            },
        ));

        assert!(game.handle_pregame_click(Vector2::new(40.0, 40.0)));
        assert!(matches!(
            game.pregame_runtime.state,
            pregame::PregameState::DifficultySelect
        ));
        assert!(game.lockstep_handle.is_none());
        assert!(game.sim_runner_handle.is_none());
        assert!(game.backend_session.is_none());
    }

    #[test]
    fn in_game_return_button_click_returns_to_main_menu() {
        let mut game = Game::default();
        game.pregame_runtime =
            pregame::PregameRuntime::new_for_menu(pregame::PregameCatalog::fallback());
        game.pregame_runtime.mark_in_game();
        game.return_to_title_button_rect = UiRect {
            x: 8.0,
            y: 10.0,
            w: 96.0,
            h: 40.0,
        };
        game.current_sim_tick = 42;
        game.current_round = 7;
        game.round_is_running = true;
        game.is_game_paused = true;
        game.game_speed_multiplier = 2;
        game.td_auto_start_sent_for_idle_round = true;
        game.is_td_mode = true;
        game.td_camera_configured = true;
        game.tower_ability_bar_snapshot_at = Some(Instant::now());
        game.sim_lua_content_generation = 9;
        game.sim_lua_content_hash = "stale".to_string();
        game.sim_dev_lua_reload_error = Some("stale error".to_string());
        game.sim_batches_last_snapshot_tick = Some(42);
        game.sim_last_explosion_tick = Some(42);
        game.sim_seen_tower_fire_fx.insert((1, 2, 42));
        game.sim_seen_attack_phase_fx.insert((1, 2, 3, 42));
        game.sim_seen_attack_cancel_fx.insert((1, 2, 3, 42));
        game.active_explosions.push(ActiveExplosion {
            pos: Vector2::new(1.0, 2.0),
            max_radius: 3.0,
            duration: 1.0,
            elapsed: 0.5,
        });
        game.td_paths_render.push(vec![Vector2::new(1.0, 2.0)]);
        game.td_regions_render.push(vec![Vector2::new(3.0, 4.0)]);
        game.pending_pred_dmg.insert(
            7,
            PendingPredDmg {
                target_id: 8,
                dmg: 9.0,
                applied: true,
            },
        );
        game.heartbeat.tick = 42;
        game.net_bytes_current = 100;
        game.net_bytes_last_sec = 200;
        game.net_wire_bytes_current = 300;
        game.net_wire_bytes_last_sec = 400;
        game.latest_rtt_us = Some(5_000);

        assert!(game.handle_in_game_return_click(Vector2::new(24.0, 24.0)));
        assert!(matches!(
            game.pregame_runtime.state,
            pregame::PregameState::MainMenu
        ));
        assert!(game.lockstep_handle.is_none());
        assert!(game.sim_runner_handle.is_none());
        assert!(game.backend_session.is_none());
        assert_eq!(game.current_sim_tick, 0);
        assert_eq!(game.current_round, 0);
        assert!(!game.round_is_running);
        assert!(!game.is_game_paused);
        assert_eq!(game.game_speed_multiplier, 1);
        assert!(!game.td_auto_start_sent_for_idle_round);
        assert!(!game.is_td_mode);
        assert!(!game.td_camera_configured);
        assert!(game.tower_ability_bar_snapshot_at.is_none());
        assert_eq!(game.sim_lua_content_generation, 0);
        assert!(game.sim_lua_content_hash.is_empty());
        assert!(game.sim_dev_lua_reload_error.is_none());
        assert!(game.sim_batches_last_snapshot_tick.is_none());
        assert!(game.sim_last_explosion_tick.is_none());
        assert!(game.sim_seen_tower_fire_fx.is_empty());
        assert!(game.sim_seen_attack_phase_fx.is_empty());
        assert!(game.sim_seen_attack_cancel_fx.is_empty());
        assert!(game.active_explosions.is_empty());
        assert!(game.td_paths_render.is_empty());
        assert!(game.td_regions_render.is_empty());
        assert!(game.pending_pred_dmg.is_empty());
        assert_eq!(game.heartbeat.tick, 0);
        assert_eq!(game.net_bytes_current, 0);
        assert_eq!(game.net_bytes_last_sec, 0);
        assert_eq!(game.net_wire_bytes_current, 0);
        assert_eq!(game.net_wire_bytes_last_sec, 0);
        assert!(game.latest_rtt_us.is_none());
        assert!(game.session_render_reset_pending);

        assert!(!game.handle_in_game_return_click(Vector2::new(24.0, 24.0)));
    }

    #[test]
    fn session_render_reset_releases_entity_slot_state() {
        let mut game = Game::default();
        game.sim_entity_slots.insert(
            7,
            render_bridge::SimEntitySlots {
                body_slot: 3,
                hp_bg_slot: Some(4),
                hp_fg_slot: Some(5),
                turret_slot: Some(6),
                proj_accent_slot: None,
            },
        );
        game.entity_kind_cache
            .insert(7, sim_runner::EntityKind::Tower);
        let mut scene = Scene::new();

        game.reset_session_render_state(&mut scene);

        assert!(game.sim_entity_slots.is_empty());
        assert!(game.entity_kind_cache.is_empty());
    }

    #[test]
    fn returning_to_pregame_removes_dynamic_td_shop_cards() {
        let mut game = Game::default();
        let mut ui = UserInterface::new(Default::default());
        let bg: Handle<UiNode> = ImageBuilder::new(WidgetBuilder::new())
            .build(&mut ui.build_ctx())
            .transmute();
        let icon: Handle<UiNode> = ImageBuilder::new(WidgetBuilder::new())
            .build(&mut ui.build_ctx())
            .transmute();
        let key_text = TextBuilder::new(WidgetBuilder::new()).build(&mut ui.build_ctx());
        let name_text = TextBuilder::new(WidgetBuilder::new()).build(&mut ui.build_ctx());
        let price_text = TextBuilder::new(WidgetBuilder::new()).build(&mut ui.build_ctx());
        let handles: [Handle<UiNode>; 5] = [
            bg,
            icon,
            key_text.transmute(),
            name_text.transmute(),
            price_text.transmute(),
        ];
        game.ui_td_tower_cards.push(TdTowerShopCard {
            bg,
            icon,
            key_text,
            name_text,
            price_text,
        });
        game.td_tower_button_rects.push((1.0, 2.0, 3.0, 4.0));

        game.clear_td_tower_shop_cards(&mut ui);
        while ui.poll_message().is_some() {}

        assert!(game.ui_td_tower_cards.is_empty());
        assert!(game.td_tower_button_rects.is_empty());
        assert_eq!(
            game.ui_td_tower_cards.len(),
            game.td_tower_button_rects.len()
        );
        for handle in handles {
            assert!(!ui.nodes().is_valid_handle(handle));
        }
    }

    #[test]
    fn projectile_trail_dir_uses_initial_delta() {
        let dir = initial_projectile_trail_dir(
            Vector2::new(0.0, 0.0),
            Vector2::new(3.0, 4.0),
            Vector2::new(-1.0, 0.0),
        );

        assert!((dir.x - 0.6).abs() < 1.0e-6);
        assert!((dir.y - 0.8).abs() < 1.0e-6);
    }

    #[test]
    fn projectile_trail_dir_falls_back_for_zero_delta() {
        let dir = initial_projectile_trail_dir(
            Vector2::new(2.0, 2.0),
            Vector2::new(2.0, 2.0),
            Vector2::new(0.0, -2.0),
        );

        assert!(dir.x.abs() < 1.0e-6);
        assert!((dir.y + 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn projectile_trail_quad_keeps_initial_rotation_after_position_changes() {
        let spawn = Vector2::new(0.0, 0.0);
        let initial_dir =
            initial_projectile_trail_dir(spawn, Vector2::new(1.0, 0.0), Vector2::new(0.0, 1.0));

        let (_, _, rotation_a) = projectile_trail_quad(spawn, Vector2::new(1.0, 0.0), initial_dir);
        let (_, _, rotation_b) = projectile_trail_quad(spawn, Vector2::new(1.0, 1.0), initial_dir);

        assert!(rotation_a.abs() < 1.0e-6);
        assert!(rotation_b.abs() < 1.0e-6);
    }

    #[test]
    fn hero_animation_playback_speed_freezes_only_while_paused() {
        assert_eq!(hero_animation_playback_speed(1.25, true), 0.0);
        assert_eq!(hero_animation_playback_speed(1.25, false), 1.25);
    }

    #[test]
    fn tower_animation_dt_freezes_only_while_paused() {
        assert_eq!(tower_animation_dt(0.25, true), 0.0);
        assert_eq!(tower_animation_dt(0.25, false), 0.25);
    }

    #[test]
    fn td_control_labels_use_start_as_resume_when_paused() {
        assert_eq!(td_start_control_label(true, false, 0, 3, 1), "RESUME");
        assert_eq!(td_start_control_label(false, false, 0, 3, 1), "READY");
        assert_eq!(td_start_control_label(false, true, 0, 3, 1), "1X");
        assert_eq!(td_start_control_label(false, true, 0, 3, 2), "2X");
        assert_eq!(td_start_control_label(false, false, 3, 3, 2), "DONE");
    }

    #[test]
    fn td_round_status_label_shows_current_wave_being_fought() {
        assert_eq!(td_round_status_label(false, 0, 3), "第 1 / 3 波");
        assert_eq!(td_round_status_label(true, 0, 3), "第 1 / 3 波");
        assert_eq!(td_round_status_label(false, 1, 3), "第 2 / 3 波");
        assert_eq!(td_round_status_label(true, 1, 3), "第 2 / 3 波");
        assert_eq!(td_round_status_label(false, 3, 3), "完成 3 / 3 波");
        assert_eq!(td_round_status_label(false, 0, 0), String::new());
    }

    #[test]
    fn td_camera_view_zooms_out_enough_for_right_panel() {
        let config = td_camera_view_config();

        assert!((16.0..=17.0).contains(&config.vertical_size));
        assert!((0.5..=1.0).contains(&config.scene_position.x));
    }

    #[test]
    fn td_mode_is_detected_from_lockstep_snapshot_resources() {
        assert!(td_mode_from_snapshot(200, 40));
        assert!(td_mode_from_snapshot(0, 40));
        assert!(!td_mode_from_snapshot(0, 0));
    }

    #[test]
    fn td_auto_start_round_only_when_idle_and_armed() {
        assert!(td_should_auto_start_round(true, false, false, false, 1, 3));
        assert!(!td_should_auto_start_round(
            false, false, false, false, 1, 3
        ));
        assert!(!td_should_auto_start_round(true, true, false, false, 1, 3));
        assert!(!td_should_auto_start_round(true, false, true, false, 1, 3));
        assert!(!td_should_auto_start_round(true, false, false, true, 1, 3));
        assert!(!td_should_auto_start_round(true, false, false, false, 3, 3));
    }

    #[test]
    fn td_pause_control_opacity_dims_when_paused() {
        assert_eq!(td_pause_control_opacity(true), Some(0.35));
        assert_eq!(td_pause_control_opacity(false), Some(1.0));
    }
}
