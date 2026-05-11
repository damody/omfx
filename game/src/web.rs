//! Browser/WASM implementation of the omfx Fyrox plugin.

use std::cell::RefCell;
use std::rc::Rc;

use fyrox::core::wasm_bindgen::{closure::Closure, JsCast};
use fyrox::{
    core::{
        algebra::{Vector2, Vector3},
        color::Color,
        pool::Handle,
        reflect::prelude::*,
        visitor::prelude::*,
    },
    gui::{
        brush::Brush,
        text::{TextBuilder, TextMessage},
        widget::WidgetBuilder,
        UiNode, UserInterface,
    },
    plugin::{error::GameResult, Plugin, PluginContext, PluginRegistrationContext},
    scene::{
        base::BaseBuilder,
        camera::{CameraBuilder, OrthographicProjection, Projection},
        dim2::rectangle::RectangleBuilder,
        transform::TransformBuilder,
        EnvironmentLightingSource, Scene,
    },
};
use js_sys::{ArrayBuffer, Uint8Array};
use omoba_core::game_proto::{GameStart, JoinRequest, JoinRole, StateHash, SubscribeRequest, TickBatch};
use prost::Message;
use wasm_bindgen_futures::JsFuture;
use web_sys::{BinaryType, CloseEvent, ErrorEvent, Event, MessageEvent, Response, WebSocket};

pub use fyrox;

const DEFAULT_WS_ENDPOINT: &str = "ws://127.0.0.1:50062";
const TAG_GAME_EVENT: u8 = 0x02;
const TAG_SUBSCRIBE_REQUEST: u8 = 0x04;
const TAG_INPUT_SUBMIT: u8 = 0x10;
const TAG_TICK_BATCH: u8 = 0x11;
const TAG_STATE_HASH: u8 = 0x12;
const TAG_JOIN_REQUEST: u8 = 0x13;
const TAG_GAME_START: u8 = 0x14;
const TAG_SNAPSHOT_RESP: u8 = 0x16;
const TAG_PING_RESP: u8 = 0x18;
const COMPRESSION_FLAG: u8 = 0x80;
const REQUIRED_ASSET_PROBE: &str = "data/ability_icons/ability_default_placeholder.png";

#[derive(Default)]
struct WebClientStatus {
    endpoint: String,
    line: String,
    asset_status: String,
    connected: bool,
    player_id: Option<u32>,
    latest_tick: u32,
    tick_batches: u64,
    game_events: u64,
    state_hashes: u64,
    snapshot_responses: u64,
    errors: u64,
}

impl WebClientStatus {
    fn summary(&self) -> String {
        let player = self
            .player_id
            .map(|id| id.to_string())
            .unwrap_or_else(|| "-".to_string());
        format!(
            "omfx Web/WASM\nendpoint: {}\nstatus: {}\nassets: {}\nconnected: {}  player: {}\ntick: {}  batches: {}  events: {}  hashes: {}  snapshots: {}  errors: {}",
            self.endpoint,
            self.line,
            if self.asset_status.is_empty() { "checking" } else { &self.asset_status },
            if self.connected { "yes" } else { "no" },
            player,
            self.latest_tick,
            self.tick_batches,
            self.game_events,
            self.state_hashes,
            self.snapshot_responses,
            self.errors,
        )
    }
}

struct WebSocketClient {
    _socket: WebSocket,
    _on_open: Closure<dyn FnMut(Event)>,
    _on_message: Closure<dyn FnMut(MessageEvent)>,
    _on_error: Closure<dyn FnMut(ErrorEvent)>,
    _on_close: Closure<dyn FnMut(CloseEvent)>,
}

#[derive(Visit, Reflect)]
#[reflect(non_cloneable)]
pub struct Game {
    scene: Handle<Scene>,
    status_text: Handle<UiNode>,
    #[visit(skip)]
    #[reflect(hidden)]
    status: Rc<RefCell<WebClientStatus>>,
    #[visit(skip)]
    #[reflect(hidden)]
    last_status_text: String,
    #[visit(skip)]
    #[reflect(hidden)]
    ws_client: Option<WebSocketClient>,
}

impl Default for Game {
    fn default() -> Self {
        Self {
            scene: Handle::NONE,
            status_text: Handle::NONE,
            status: Rc::new(RefCell::new(WebClientStatus::default())),
            last_status_text: String::new(),
            ws_client: None,
        }
    }
}

impl std::fmt::Debug for Game {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Game")
            .field("scene", &self.scene)
            .field("status_text", &self.status_text)
            .field("last_status_text", &self.last_status_text)
            .finish_non_exhaustive()
    }
}

impl Plugin for Game {
    fn register(&self, _context: PluginRegistrationContext) -> GameResult {
        Ok(())
    }

    fn init(&mut self, _scene_path: Option<&str>, context: PluginContext) -> GameResult {
        let mut scene = Scene::new();
        scene.set_skybox(None);
        scene.rendering_options.set_value_and_mark_modified(
            fyrox::scene::SceneRenderingOptions {
                clear_color: Some(Color::from_rgba(18, 28, 38, 255)),
                ambient_lighting_color: Color::WHITE,
                environment_lighting_source: EnvironmentLightingSource::AmbientColor,
                environment_lighting_brightness: 1.0,
                ..Default::default()
            },
        );

        CameraBuilder::new(
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
        .build(&mut scene.graph);

        RectangleBuilder::new(
            BaseBuilder::new().with_local_transform(
                TransformBuilder::new()
                    .with_local_position(Vector3::new(0.0, 0.0, 0.0))
                    .with_local_scale(Vector3::new(30.0, 18.0, f32::EPSILON))
                    .build(),
            ),
        )
        .with_color(Color::from_rgba(18, 28, 38, 255))
        .build(&mut scene.graph);

        self.scene = context.scenes.add(scene);
        context
            .user_interfaces
            .add(UserInterface::new(Default::default()));
        let ui = context.user_interfaces.first_mut();
        self.status_text = TextBuilder::new(
            WidgetBuilder::new()
                .with_desired_position(Vector2::new(24.0, 24.0))
                .with_width(760.0)
                .with_height(180.0)
                .with_foreground(Brush::Solid(Color::from_rgba(225, 238, 248, 255)).into()),
        )
        .with_font_size(18.0.into())
        .with_text("omfx Web/WASM starting...".to_string())
        .build(&mut ui.build_ctx())
        .transmute();

        let endpoint = configured_ws_endpoint();
        self.status.borrow_mut().endpoint = endpoint.clone();
        match WebSocketClient::connect(endpoint, self.status.clone()) {
            Ok(client) => self.ws_client = Some(client),
            Err(message) => set_status(&self.status, format!("WebSocket unavailable: {message}")),
        }
        probe_required_asset(REQUIRED_ASSET_PROBE, self.status.clone());

        Ok(())
    }

    fn update(&mut self, context: &mut PluginContext) -> GameResult {
        let status_text = self.status.borrow().summary();
        if status_text != self.last_status_text {
            context
                .user_interfaces
                .first_mut()
                .send(self.status_text, TextMessage::Text(status_text.clone()));
            self.last_status_text = status_text;
        }
        Ok(())
    }
}

impl WebSocketClient {
    fn connect(endpoint: String, status: Rc<RefCell<WebClientStatus>>) -> Result<Self, String> {
        let socket = WebSocket::new(&endpoint).map_err(js_error_to_string)?;
        socket.set_binary_type(BinaryType::Arraybuffer);

        set_status(&status, format!("connecting to {endpoint}"));

        let player_name = configured_player_name();
        let on_open_socket = socket.clone();
        let on_open_status = status.clone();
        let on_open = Closure::<dyn FnMut(Event)>::new(move |_event: Event| {
            set_status(&on_open_status, "WebSocket open; sending SubscribeRequest + JoinRequest");
            if let Err(message) = send_initial_join(&on_open_socket, &player_name) {
                let mut state = on_open_status.borrow_mut();
                state.errors += 1;
                state.line = format!("join send failed: {message}");
                console_log(&state.line);
            }
        });
        socket.set_onopen(Some(on_open.as_ref().unchecked_ref()));

        let on_message_status = status.clone();
        let on_message = Closure::<dyn FnMut(MessageEvent)>::new(move |event: MessageEvent| {
            if let Ok(buffer) = event.data().dyn_into::<ArrayBuffer>() {
                let bytes = Uint8Array::new(&buffer).to_vec();
                handle_ws_bytes(&on_message_status, &bytes);
            } else if let Some(text) = event.data().as_string() {
                set_status(&on_message_status, format!("text frame: {text}"));
            } else {
                let mut state = on_message_status.borrow_mut();
                state.errors += 1;
                state.line = "unsupported WebSocket message type".to_string();
            }
        });
        socket.set_onmessage(Some(on_message.as_ref().unchecked_ref()));

        let on_error_status = status.clone();
        let on_error = Closure::<dyn FnMut(ErrorEvent)>::new(move |event: ErrorEvent| {
            let mut state = on_error_status.borrow_mut();
            state.errors += 1;
            state.connected = false;
            state.line = format!("WebSocket error: {}", event.message());
            console_log(&state.line);
        });
        socket.set_onerror(Some(on_error.as_ref().unchecked_ref()));

        let on_close_status = status.clone();
        let on_close = Closure::<dyn FnMut(CloseEvent)>::new(move |event: CloseEvent| {
            let mut state = on_close_status.borrow_mut();
            state.connected = false;
            state.line = format!("WebSocket closed: code={} reason={}", event.code(), event.reason());
            console_log(&state.line);
        });
        socket.set_onclose(Some(on_close.as_ref().unchecked_ref()));

        Ok(Self {
            _socket: socket,
            _on_open: on_open,
            _on_message: on_message,
            _on_error: on_error,
            _on_close: on_close,
        })
    }
}

fn send_initial_join(socket: &WebSocket, player_name: &str) -> Result<(), String> {
    send_proto(
        socket,
        TAG_SUBSCRIBE_REQUEST,
        &SubscribeRequest {
            player_name: player_name.to_string(),
        },
    )?;
    send_proto(
        socket,
        TAG_JOIN_REQUEST,
        &JoinRequest {
            player_name: player_name.to_string(),
            role: JoinRole::RolePlayer as i32,
        },
    )
}

fn handle_ws_bytes(status: &Rc<RefCell<WebClientStatus>>, bytes: &[u8]) {
    match parse_frame(bytes) {
        Ok((tag, payload)) => handle_frame(status, tag, payload),
        Err(message) => {
            let mut state = status.borrow_mut();
            state.errors += 1;
            state.line = message;
        }
    }
}

fn handle_frame(status: &Rc<RefCell<WebClientStatus>>, tag: u8, payload: &[u8]) {
    match tag {
        TAG_GAME_EVENT => {
            let mut state = status.borrow_mut();
            state.connected = true;
            state.game_events += 1;
            state.line = "received GameEvent".to_string();
        }
        TAG_TICK_BATCH => match TickBatch::decode(payload) {
            Ok(batch) => {
                let mut state = status.borrow_mut();
                state.connected = true;
                state.latest_tick = batch.tick;
                state.tick_batches += 1;
                state.line = format!("received TickBatch tick={}", batch.tick);
            }
            Err(e) => record_decode_error(status, "TickBatch", e),
        },
        TAG_STATE_HASH => match StateHash::decode(payload) {
            Ok(hash) => {
                let mut state = status.borrow_mut();
                state.connected = true;
                state.state_hashes += 1;
                state.line = format!("received StateHash tick={} hash=0x{:016x}", hash.tick, hash.hash);
            }
            Err(e) => record_decode_error(status, "StateHash", e),
        },
        TAG_GAME_START => match GameStart::decode(payload) {
            Ok(start) => {
                let mut state = status.borrow_mut();
                state.connected = true;
                state.player_id = Some(start.player_id);
                state.latest_tick = start.start_tick;
                state.line = format!(
                    "joined game player_id={} start_tick={} master_seed=0x{:016x}",
                    start.player_id, start.start_tick, start.master_seed
                );
            }
            Err(e) => record_decode_error(status, "GameStart", e),
        },
        TAG_SNAPSHOT_RESP => {
            let mut state = status.borrow_mut();
            state.connected = true;
            state.snapshot_responses += 1;
            state.line = "received SnapshotResp".to_string();
        }
        TAG_PING_RESP => {
            let mut state = status.borrow_mut();
            state.connected = true;
            state.line = "received PingResp".to_string();
        }
        TAG_INPUT_SUBMIT => {
            let mut state = status.borrow_mut();
            state.connected = true;
            state.line = "received echoed InputSubmit".to_string();
        }
        other => {
            let mut state = status.borrow_mut();
            state.connected = true;
            state.line = format!("received unsupported frame tag=0x{other:02x}");
        }
    }
}

fn record_decode_error(status: &Rc<RefCell<WebClientStatus>>, ty: &str, e: prost::DecodeError) {
    let mut state = status.borrow_mut();
    state.errors += 1;
    state.line = format!("decode {ty} failed: {e}");
}

fn send_proto<M: Message>(socket: &WebSocket, tag: u8, msg: &M) -> Result<(), String> {
    let frame = build_frame(tag, &msg.encode_to_vec());
    socket.send_with_u8_array(&frame).map_err(js_error_to_string)
}

fn build_frame(tag: u8, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(5 + payload.len());
    out.push(tag);
    out.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    out.extend_from_slice(payload);
    out
}

fn parse_frame(bytes: &[u8]) -> Result<(u8, &[u8]), String> {
    if bytes.len() < 5 {
        return Err(format!("short WebSocket frame: {} bytes", bytes.len()));
    }
    let tag = bytes[0];
    if tag & COMPRESSION_FLAG != 0 {
        return Err(format!(
            "compressed WebSocket frame tag=0x{tag:02x} is unsupported by wasm client"
        ));
    }
    let len = u32::from_be_bytes([bytes[1], bytes[2], bytes[3], bytes[4]]) as usize;
    if bytes.len() != 5 + len {
        return Err(format!(
            "bad WebSocket frame length: header={} actual={}",
            len,
            bytes.len().saturating_sub(5)
        ));
    }
    Ok((tag, &bytes[5..]))
}

fn configured_ws_endpoint() -> String {
    query_param("omoba_ws")
        .or_else(|| query_param("ws"))
        .unwrap_or_else(|| DEFAULT_WS_ENDPOINT.to_string())
}

fn probe_required_asset(path: &'static str, status: Rc<RefCell<WebClientStatus>>) {
    let Some(window) = web_sys::window() else {
        status.borrow_mut().asset_status = "no browser window for asset probe".to_string();
        return;
    };
    let promise = window.fetch_with_str(path);
    wasm_bindgen_futures::spawn_local(async move {
        match JsFuture::from(promise).await {
            Ok(value) => match value.dyn_into::<Response>() {
                Ok(response) if response.ok() => {
                    status.borrow_mut().asset_status = format!("ok: {path}");
                }
                Ok(response) => {
                    let mut state = status.borrow_mut();
                    state.errors += 1;
                    state.asset_status = format!(
                        "missing: {path} (HTTP {})",
                        response.status()
                    );
                }
                Err(_) => {
                    let mut state = status.borrow_mut();
                    state.errors += 1;
                    state.asset_status = format!("bad fetch response for {path}");
                }
            },
            Err(e) => {
                let mut state = status.borrow_mut();
                state.errors += 1;
                state.asset_status = format!("fetch failed for {path}: {}", js_error_to_string(e));
            }
        }
    });
}

fn configured_player_name() -> String {
    query_param("player").unwrap_or_else(|| "web-player".to_string())
}

fn query_param(name: &str) -> Option<String> {
    let window = web_sys::window()?;
    let search = window.location().search().ok()?;
    let params = web_sys::UrlSearchParams::new_with_str(&search).ok()?;
    params.get(name).filter(|value| !value.trim().is_empty())
}

fn set_status(status: &Rc<RefCell<WebClientStatus>>, line: impl Into<String>) {
    let line = line.into();
    status.borrow_mut().line = line.clone();
    console_log(&line);
}

fn console_log(message: &str) {
    web_sys::console::log_1(&message.into());
}

fn js_error_to_string(value: fyrox::core::wasm_bindgen::JsValue) -> String {
    value
        .as_string()
        .unwrap_or_else(|| "unknown JavaScript error".to_string())
}
