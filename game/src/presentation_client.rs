use std::{
    collections::BTreeMap,
    fs::OpenOptions,
    io::Write,
    net::SocketAddr,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex,
    },
    thread,
    time::Instant,
};

use crossbeam_channel::{bounded, Receiver, Sender};
use omoba_core::{
    game_proto::{
        player_input, render_lifecycle_event, renderer_input, renderer_ipc_envelope,
        AbilityCastIntent, AttackMoveIntent, ItemUseIntent, MoveToIntent, RendererInput,
        RendererIpcEnvelope, RendererReady, RendererShutdown, TowerActionIntent,
    },
    runtime::{FilteredRenderEntity, FilteredRenderSnapshot},
};
use prost::Message;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpStream,
};

const MAGIC: u32 = 0x4f4d_5254;
use crate::filtered_render_bridge::{
    RenderLifecycleAction, RenderLifecycleBatch as BridgeLifecycleBatch,
};

const VERSION: u32 = 2;
const MAX_FRAME: usize = 8 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompletedInputLatency {
    pub request_id: u64,
    pub input_id: u32,
    pub total_us: u64,
}

#[derive(Debug)]
pub struct RendererPresentationHandle {
    pub snapshots: Receiver<RendererPresentationUpdate>,
    pub lifecycles: Receiver<BridgeLifecycleBatch>,
    inputs: Sender<RendererIpcEnvelope>,
    request_id: Arc<AtomicU64>,
    view_epoch: Arc<AtomicU64>,
    ready: Arc<AtomicBool>,
    pending_inputs: Arc<Mutex<BTreeMap<u64, Instant>>>,
    completed_latencies: Arc<Mutex<Vec<CompletedInputLatency>>>,
}

#[derive(Clone, Debug)]
pub struct RendererPresentationUpdate {
    pub snapshot: FilteredRenderSnapshot,
    pub authoritative_tick: u64,
    pub visible_fog_cells: Vec<(i32, i32)>,
    pub vision_center_raw: Option<(i64, i64)>,
    pub runtime_rtt_us: Option<u64>,
}

impl RendererPresentationHandle {
    pub fn send_player_input(
        &self,
        player_id: u32,
        input: omoba_core::game_proto::PlayerInput,
    ) -> Result<u64, String> {
        let request_id = self.request_id.fetch_add(1, Ordering::Relaxed).max(1);
        let intent = convert_input(input)
            .ok_or_else(|| "renderer-only input is not supported by IPC".to_string())?;
        self.pending_inputs
            .lock()
            .map_err(|_| "renderer latency map poisoned".to_string())?
            .insert(request_id, Instant::now());
        if let Err(error) = self
            .inputs
            .send(envelope(
                request_id,
                renderer_ipc_envelope::Payload::RendererInput(RendererInput {
                    request_id,
                    player_id,
                    disclosure_epoch: self.view_epoch.load(Ordering::Relaxed),
                    intent: Some(intent),
                }),
            ))
            .map_err(|error| error.to_string())
        {
            if let Ok(mut pending) = self.pending_inputs.lock() {
                pending.remove(&request_id);
            }
            return Err(error);
        }
        Ok(request_id)
    }

    pub fn shutdown(&self) {
        let _ = self.inputs.send(envelope(
            u64::MAX,
            renderer_ipc_envelope::Payload::RendererShutdown(RendererShutdown { graceful: true }),
        ));
    }
    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Relaxed)
    }

    pub fn drain_completed_latencies(&self) -> Vec<CompletedInputLatency> {
        self.completed_latencies
            .lock()
            .map(|mut values| std::mem::take(&mut *values))
            .unwrap_or_default()
    }

    pub fn oldest_pending_age_ms(&self) -> Option<u32> {
        let pending = self.pending_inputs.lock().ok()?;
        oldest_pending_age_ms(&pending)
    }
}

pub fn spawn(addr: SocketAddr) -> RendererPresentationHandle {
    let (snapshot_tx, snapshots) = bounded(1);
    let stale_snapshots = snapshots.clone();
    let (lifecycle_tx, lifecycles) = bounded(1024);
    let (inputs, input_rx) = bounded(256);
    let request_id = Arc::new(AtomicU64::new(1));
    let view_epoch = Arc::new(AtomicU64::new(0));
    let worker_view_epoch = Arc::clone(&view_epoch);
    let ready = Arc::new(AtomicBool::new(false));
    let pending_inputs = Arc::new(Mutex::new(BTreeMap::new()));
    let completed_latencies = Arc::new(Mutex::new(Vec::new()));
    let worker_pending_inputs = Arc::clone(&pending_inputs);
    let worker_completed_latencies = Arc::clone(&completed_latencies);
    let worker_ready = Arc::clone(&ready);
    thread::Builder::new()
        .name("omfx-presentation-ipc".into())
        .spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("presentation runtime");
            runtime.block_on(async move {
                loop {
                    match TcpStream::connect(addr).await {
                        Ok(stream) => {
                            let _ = stream.set_nodelay(true);
                            if let Err(error) = run_connection(
                                stream,
                                &snapshot_tx,
                                &stale_snapshots,
                                &lifecycle_tx,
                                &input_rx,
                                &worker_view_epoch,
                                &worker_ready,
                                &worker_pending_inputs,
                                &worker_completed_latencies,
                            )
                            .await
                            {
                                log::warn!("presentation IPC connection ended: {error}");
                            }
                        }
                        Err(error) => log::warn!("presentation IPC connect failed: {error}"),
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(250)).await;
                }
            });
        })
        .expect("presentation client thread");
    RendererPresentationHandle {
        snapshots,
        lifecycles,
        inputs,
        request_id,
        view_epoch,
        ready,
        pending_inputs,
        completed_latencies,
    }
}

async fn run_connection(
    stream: TcpStream,
    snapshots: &Sender<RendererPresentationUpdate>,
    stale_snapshots: &Receiver<RendererPresentationUpdate>,
    lifecycles: &Sender<BridgeLifecycleBatch>,
    inputs: &Receiver<RendererIpcEnvelope>,
    view_epoch: &AtomicU64,
    ready: &AtomicBool,
    pending_inputs: &Mutex<BTreeMap<u64, Instant>>,
    completed_latencies: &Mutex<Vec<CompletedInputLatency>>,
) -> Result<(), String> {
    let (mut reader, mut writer) = stream.into_split();
    write_envelope(
        &mut writer,
        &envelope(
            0,
            renderer_ipc_envelope::Payload::RendererReady(RendererReady {
                latest_snapshot_sequence: 0,
            }),
        ),
    )
    .await?;
    let mut accepted_snapshot_sequence = 0_u64;
    let mut accepted_lifecycle_sequence = 0_u64;
    let mut last_received_heroes = Vec::<String>::new();
    loop {
        tokio::select! {
            inbound = read_envelope(&mut reader) => {
                let inbound = inbound?;
                if matches!(&inbound.payload, Some(renderer_ipc_envelope::Payload::RuntimeReady(_))) { ready.store(true, Ordering::Relaxed); continue; }
                if let Some(renderer_ipc_envelope::Payload::CriticalInputResult(result)) = &inbound.payload {
                    if result.result_code == "APPLIED_TO_PRESENTATION" {
                        if let Some(sample) = record_applied_input_latency(
                            pending_inputs,
                            completed_latencies,
                            result.request_id,
                            result.input_id,
                        ) {
                            log::info!("input_to_presentation: request_id={} input_id={} authoritative_tick={} total_us={} total_ms={:.3}", result.request_id, result.input_id, result.authoritative_tick, sample.total_us, sample.total_us as f64 / 1000.0);
                            append_latency_log(result.request_id, result.input_id, result.authoritative_tick, sample.total_us);
                        }
                    }
                    continue;
                }
                if let Some(renderer_ipc_envelope::Payload::Lifecycle(batch)) = &inbound.payload {
                    if inbound.sequence <= accepted_lifecycle_sequence
                        || batch.view_epoch < view_epoch.load(Ordering::Relaxed)
                    {
                        continue;
                    }
                    accepted_lifecycle_sequence = inbound.sequence;
                    let converted = convert_lifecycle(inbound.sequence, batch.clone())?;
                    if converted.events.iter().any(|event| matches!(event, RenderLifecycleAction::ResetView)) {
                        view_epoch.store(converted.view_epoch, Ordering::Relaxed);
                    }
                    if presentation_trace_enabled() {
                        log::warn!(
                            "presentation_trace stage=renderer_lifecycle_enqueue sequence={} view_epoch={} replica_tick={} events={:?}",
                            converted.sequence,
                            converted.view_epoch,
                            converted.replica_tick,
                            converted.events,
                        );
                    }
                    enqueue_lifecycle(lifecycles, converted)?;
                    continue;
                }
                if let Some(renderer_ipc_envelope::Payload::Snapshot(snapshot)) = inbound.payload {
                    if !accept_presentation_snapshot(
                        &mut accepted_snapshot_sequence,
                        view_epoch,
                        inbound.sequence,
                        snapshot.view_epoch,
                    ) {
                        continue;
                    }
                    ready.store(true, Ordering::Relaxed);
                    view_epoch.store(snapshot.view_epoch, Ordering::Relaxed);
                    let converted = convert_update(snapshot);
                    if presentation_trace_enabled() {
                        let hero_identities = hero_identity_labels(&converted.snapshot);
                        let heroes = hero_labels(&converted.snapshot);
                        if hero_identities != last_received_heroes {
                            log::warn!(
                                "presentation_trace stage=renderer_snapshot_receive sequence={} authoritative_tick={} replica_tick={} heroes={:?}",
                                inbound.sequence,
                                converted.authoritative_tick,
                                converted.snapshot.replica_tick,
                                heroes,
                            );
                            last_received_heroes = hero_identities;
                        }
                    }
                    if let Err(error) = snapshots.try_send(converted) {
                        if error.is_full() {
                            let latest = error.into_inner();
                            let _ = stale_snapshots.try_recv();
                            let _ = snapshots.try_send(latest);
                        }
                    }
                }
            }
            outbound = next_input(inputs) => {
                let Some(outbound) = outbound else { return Ok(()); };
                write_envelope(&mut writer, &outbound).await?;
            }
        }
    }
}

fn enqueue_lifecycle(
    lifecycles: &Sender<BridgeLifecycleBatch>,
    batch: BridgeLifecycleBatch,
) -> Result<(), String> {
    lifecycles.send(batch).map_err(|error| error.to_string())
}

pub(crate) fn presentation_trace_enabled() -> bool {
    std::env::var("OMOBA_PRESENTATION_TRACE")
        .ok()
        .is_some_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn hero_labels(snapshot: &FilteredRenderSnapshot) -> Vec<String> {
    snapshot
        .entities
        .iter()
        .filter_map(|entity| {
            let payload = entity
                .components
                .get(&omoba_core::runtime::DEMO_RENDER_COMPONENT_SCHEMA_ID)?;
            let state = omoba_core::runtime::decode_demo_render_state(payload)?;
            (state.kind == 1).then(|| {
                format!(
                    "{}@{}:player{}:team{}:pos({}, {})",
                    entity.replica_id,
                    entity.disclosure_epoch,
                    state.owner_player_id,
                    state.team_id,
                    state.x_raw,
                    state.y_raw,
                )
            })
        })
        .collect()
}

fn hero_identity_labels(snapshot: &FilteredRenderSnapshot) -> Vec<String> {
    snapshot
        .entities
        .iter()
        .filter_map(|entity| {
            let payload = entity
                .components
                .get(&omoba_core::runtime::DEMO_RENDER_COMPONENT_SCHEMA_ID)?;
            let state = omoba_core::runtime::decode_demo_render_state(payload)?;
            (state.kind == 1).then(|| {
                format!(
                    "{}@{}:player{}:team{}",
                    entity.replica_id,
                    entity.disclosure_epoch,
                    state.owner_player_id,
                    state.team_id,
                )
            })
        })
        .collect()
}

fn append_latency_log(request_id: u64, input_id: u32, authoritative_tick: u64, total_us: u64) {
    let Ok(path) = std::env::var("OMFX_INPUT_LATENCY_LOG") else {
        return;
    };
    let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) else {
        return;
    };
    let _ = writeln!(file, "{{\"request_id\":{request_id},\"input_id\":{input_id},\"authoritative_tick\":{authoritative_tick},\"input_to_presentation_us\":{total_us}}}");
}

fn accept_presentation_snapshot(
    accepted_sequence: &mut u64,
    view_epoch: &AtomicU64,
    sequence: u64,
    snapshot_epoch: u64,
) -> bool {
    if sequence < *accepted_sequence || (sequence == *accepted_sequence && sequence != 0) {
        return false;
    }
    if snapshot_epoch < view_epoch.load(Ordering::Relaxed) {
        return false;
    }
    *accepted_sequence = (*accepted_sequence).max(sequence);
    true
}

async fn next_input(inputs: &Receiver<RendererIpcEnvelope>) -> Option<RendererIpcEnvelope> {
    loop {
        match inputs.try_recv() {
            Ok(value) => return Some(value),
            Err(crossbeam_channel::TryRecvError::Disconnected) => return None,
            Err(crossbeam_channel::TryRecvError::Empty) => {
                tokio::time::sleep(std::time::Duration::from_millis(4)).await;
            }
        }
    }
}

fn convert_update(
    snapshot: omoba_core::game_proto::TeamPresentationSnapshot,
) -> RendererPresentationUpdate {
    let authoritative_tick = snapshot.authoritative_tick;
    let visible_fog_cells = snapshot
        .fog_tiles
        .iter()
        .filter(|tile| tile.visible)
        .map(|tile| (tile.column, tile.row))
        .collect();
    let vision_center_raw = snapshot
        .vision_circles
        .first()
        .map(|circle| (circle.x_raw, circle.y_raw));
    let runtime_rtt_us = (snapshot.runtime_rtt_us > 0).then_some(snapshot.runtime_rtt_us);
    RendererPresentationUpdate {
        snapshot: convert_snapshot(snapshot),
        authoritative_tick,
        visible_fog_cells,
        vision_center_raw,
        runtime_rtt_us,
    }
}

fn record_applied_input_latency(
    pending: &Mutex<BTreeMap<u64, Instant>>,
    completed: &Mutex<Vec<CompletedInputLatency>>,
    request_id: u64,
    input_id: u32,
) -> Option<CompletedInputLatency> {
    let started = pending.lock().ok()?.remove(&request_id)?;
    let sample = CompletedInputLatency {
        request_id,
        input_id,
        total_us: started.elapsed().as_micros().min(u128::from(u64::MAX)) as u64,
    };
    if let Ok(mut values) = completed.lock() {
        values.push(CompletedInputLatency {
            request_id: sample.request_id,
            input_id: sample.input_id,
            total_us: sample.total_us,
        });
    }
    Some(sample)
}

fn oldest_pending_age_ms(pending: &BTreeMap<u64, Instant>) -> Option<u32> {
    pending.values().map(|started| {
        started.elapsed().as_millis().min(u128::from(u32::MAX)) as u32
    }).max()
}

fn convert_snapshot(
    snapshot: omoba_core::game_proto::TeamPresentationSnapshot,
) -> FilteredRenderSnapshot {
    FilteredRenderSnapshot {
        team_id: snapshot.team_id,
        replica_tick: snapshot.replica_tick,
        entities: snapshot
            .entities
            .into_iter()
            .map(|entity| FilteredRenderEntity {
                replica_id: entity.render_id,
                disclosure_epoch: entity.disclosure_epoch,
                entity_kind: entity.entity_kind,
                components: entity
                    .components
                    .into_iter()
                    .map(|component| (component.schema_id, component.safe_payload))
                    .collect::<BTreeMap<_, _>>(),
            })
            .collect(),
        public_events: Vec::new(),
        external_effects: Vec::new(),
        memory_directives: Vec::new(),
    }
}

fn convert_lifecycle(
    sequence: u64,
    batch: omoba_core::game_proto::RenderLifecycleBatch,
) -> Result<BridgeLifecycleBatch, String> {
    let events = batch
        .events
        .into_iter()
        .map(|event| {
            let action = event
                .action
                .ok_or_else(|| "lifecycle action missing".to_string())?;
            Ok(match action {
                render_lifecycle_event::Action::Hide(hide) => RenderLifecycleAction::Hide {
                    replica_id: event.replica_id,
                    disclosure_epoch: event.disclosure_epoch,
                    remember_policy: hide.remember_policy,
                    sanitized_presentation: hide.sanitized_presentation,
                },
                render_lifecycle_event::Action::Forget(_) => RenderLifecycleAction::Forget {
                    replica_id: event.replica_id,
                    disclosure_epoch: event.disclosure_epoch,
                },
                render_lifecycle_event::Action::ResetView(_) => RenderLifecycleAction::ResetView,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(BridgeLifecycleBatch {
        sequence,
        team_id: batch.team_id,
        authoritative_tick: batch.authoritative_tick,
        replica_tick: batch.replica_tick,
        view_epoch: batch.view_epoch,
        events,
    })
}

fn convert_input(input: omoba_core::game_proto::PlayerInput) -> Option<renderer_input::Intent> {
    match input.action? {
        player_input::Action::MoveTo(value) => value.target.map(|target| {
            renderer_input::Intent::MoveTo(MoveToIntent {
                x_raw: i64::from(target.x),
                y_raw: i64::from(target.y),
            })
        }),
        player_input::Action::AttackMove(value) => value.target.map(|target| {
            renderer_input::Intent::AttackMove(AttackMoveIntent {
                x_raw: i64::from(target.x),
                y_raw: i64::from(target.y),
            })
        }),
        player_input::Action::CastAbility(value) => {
            Some(renderer_input::Intent::AbilityCast(AbilityCastIntent {
                ability_index: value.ability_index,
                target_render_id: u64::from(value.target_entity.unwrap_or(0)),
                x_raw: i64::from(value.target_pos.as_ref().map_or(0, |pos| pos.x)),
                y_raw: i64::from(value.target_pos.as_ref().map_or(0, |pos| pos.y)),
            }))
        }
        player_input::Action::ItemUse(value) => {
            Some(renderer_input::Intent::ItemUse(ItemUseIntent {
                item_slot: value.item_slot,
                target_render_id: u64::from(value.target_entity.unwrap_or(0)),
                x_raw: i64::from(value.target_pos.as_ref().map_or(0, |pos| pos.x)),
                y_raw: i64::from(value.target_pos.as_ref().map_or(0, |pos| pos.y)),
            }))
        }
        player_input::Action::TowerPlace(value) => value.pos.map(|pos| {
            renderer_input::Intent::TowerAction(TowerActionIntent {
                action_kind: 1,
                tower_render_id: 0,
                tower_kind_id: value.tower_kind_id,
                path: 0,
                level: 0,
                x_raw: i64::from(pos.x),
                y_raw: i64::from(pos.y),
            })
        }),
        player_input::Action::TowerUpgrade(value) => {
            Some(renderer_input::Intent::TowerAction(TowerActionIntent {
                action_kind: 2,
                tower_render_id: u64::from(value.tower_entity_id),
                tower_kind_id: 0,
                path: value.path,
                level: value.level,
                x_raw: 0,
                y_raw: 0,
            }))
        }
        player_input::Action::TowerSell(value) => {
            Some(renderer_input::Intent::TowerAction(TowerActionIntent {
                action_kind: 3,
                tower_render_id: u64::from(value.tower_entity_id),
                tower_kind_id: 0,
                path: 0,
                level: 0,
                x_raw: 0,
                y_raw: 0,
            }))
        }
        _ => None,
    }
}

fn envelope(sequence: u64, payload: renderer_ipc_envelope::Payload) -> RendererIpcEnvelope {
    RendererIpcEnvelope {
        magic: MAGIC,
        protocol_version: VERSION,
        sequence,
        payload: Some(payload),
    }
}

async fn read_envelope(
    reader: &mut (impl AsyncReadExt + Unpin),
) -> Result<RendererIpcEnvelope, String> {
    let length = reader.read_u32().await.map_err(|error| error.to_string())? as usize;
    if length == 0 || length > MAX_FRAME {
        return Err("invalid presentation frame length".into());
    }
    let mut bytes = vec![0; length];
    reader
        .read_exact(&mut bytes)
        .await
        .map_err(|error| error.to_string())?;
    let envelope =
        RendererIpcEnvelope::decode(bytes.as_slice()).map_err(|error| error.to_string())?;
    if envelope.magic != MAGIC || envelope.protocol_version != VERSION {
        return Err("presentation protocol mismatch".into());
    }
    Ok(envelope)
}

async fn write_envelope(
    writer: &mut (impl AsyncWriteExt + Unpin),
    envelope: &RendererIpcEnvelope,
) -> Result<(), String> {
    let bytes = envelope.encode_to_vec();
    writer
        .write_u32(bytes.len() as u32)
        .await
        .map_err(|error| error.to_string())?;
    writer
        .write_all(&bytes)
        .await
        .map_err(|error| error.to_string())?;
    Ok(())
}

#[cfg(test)]
mod delay_safety_tests {
    use super::*;
    use omoba_core::game_proto::{player_input, MoveTo, TeamPresentationSnapshot, Vec2I};

    #[test]
    fn stale_snapshot_sequence_and_epoch_are_ignored_without_disconnect() {
        let epoch = AtomicU64::new(7);
        let mut sequence = 10;
        assert!(!accept_presentation_snapshot(&mut sequence, &epoch, 9, 7));
        assert!(!accept_presentation_snapshot(&mut sequence, &epoch, 11, 6));
        assert!(accept_presentation_snapshot(&mut sequence, &epoch, 11, 7));
        assert_eq!(sequence, 11)
    }

    #[test]
    fn renderer_update_preserves_remote_ticks() {
        let update = convert_update(TeamPresentationSnapshot {
            team_id: 2,
            authoritative_tick: 720,
            replica_tick: 719,
            runtime_rtt_us: 1_500,
            ..Default::default()
        });
        assert_eq!(update.authoritative_tick, 720);
        assert_eq!(update.snapshot.replica_tick, 719);
        assert_eq!(update.snapshot.team_id, 2);
        assert_eq!(update.runtime_rtt_us, Some(1_500));
    }

    #[test]
    fn missing_runtime_rtt_stays_unset() {
        let update = convert_update(TeamPresentationSnapshot::default());
        assert_eq!(update.runtime_rtt_us, None);
    }

    #[test]
    fn applied_input_latency_moves_pending_to_completed() {
        let pending = Mutex::new(BTreeMap::from([(7_u64, Instant::now())]));
        let completed = Mutex::new(Vec::new());
        let sample = record_applied_input_latency(&pending, &completed, 7, 42).unwrap();
        assert_eq!(sample.request_id, 7);
        assert_eq!(sample.input_id, 42);
        assert!(pending.lock().unwrap().is_empty());
        assert_eq!(completed.lock().unwrap().len(), 1);
    }

    #[test]
    fn oldest_pending_age_uses_the_oldest_request() {
        let mut pending = BTreeMap::new();
        pending.insert(1, Instant::now() - std::time::Duration::from_millis(40));
        pending.insert(2, Instant::now() - std::time::Duration::from_millis(12));
        let age = oldest_pending_age_ms(&pending).unwrap();
        assert!(age >= 40);
    }

    #[test]
    fn move_to_is_encoded_as_renderer_ipc_intent() {
        let intent = convert_input(omoba_core::game_proto::PlayerInput {
            action: Some(player_input::Action::MoveTo(MoveTo {
                target: Some(Vec2I { x: 123, y: -456 }),
                queued: false,
            })),
        });
        let Some(renderer_input::Intent::MoveTo(move_to)) = intent else {
            panic!("MoveTo must be forwarded through renderer IPC");
        };
        assert_eq!((move_to.x_raw, move_to.y_raw), (123, -456));
    }

    fn lifecycle(sequence: u64) -> BridgeLifecycleBatch {
        BridgeLifecycleBatch {
            sequence,
            team_id: 2,
            authoritative_tick: sequence,
            replica_tick: sequence,
            view_epoch: 7,
            events: vec![RenderLifecycleAction::Forget {
                replica_id: sequence,
                disclosure_epoch: 9,
            }],
        }
    }

    #[test]
    fn lifecycle_fifo_keeps_order_when_snapshot_latest_is_replaced() {
        let (lifecycle_tx, lifecycle_rx) = bounded(2);
        let (snapshot_tx, snapshot_rx) = bounded(1);
        enqueue_lifecycle(&lifecycle_tx, lifecycle(1)).unwrap();
        snapshot_tx.send(1_u64).unwrap();
        let _ = snapshot_rx.try_recv().unwrap();
        snapshot_tx.send(2_u64).unwrap();
        enqueue_lifecycle(&lifecycle_tx, lifecycle(2)).unwrap();
        assert_eq!(lifecycle_rx.recv().unwrap().sequence, 1);
        assert_eq!(lifecycle_rx.recv().unwrap().sequence, 2);
        assert_eq!(snapshot_rx.recv().unwrap(), 2);
    }

    #[test]
    fn lifecycle_queue_applies_backpressure_without_discarding() {
        let (tx, rx) = bounded(1);
        enqueue_lifecycle(&tx, lifecycle(1)).unwrap();
        let producer = std::thread::spawn(move || {
            enqueue_lifecycle(&tx, lifecycle(2)).unwrap();
        });
        assert_eq!(rx.recv().unwrap().sequence, 1);
        producer.join().unwrap();
        assert_eq!(rx.recv().unwrap().sequence, 2);
    }

    #[test]
    fn slow_renderer_drains_every_forget_without_leaving_clones() {
        use crate::filtered_render_bridge::FilteredRenderBridge;

        let (tx, rx) = bounded(64);
        let mut bridge = FilteredRenderBridge::default();
        bridge.apply_lifecycle(&BridgeLifecycleBatch {
            sequence: 1,
            team_id: 2,
            authoritative_tick: 1,
            replica_tick: 1,
            view_epoch: 7,
            events: vec![RenderLifecycleAction::ResetView],
        });
        let mut stale_entities = Vec::new();
        for replica_id in 100..120 {
            stale_entities.push(FilteredRenderEntity {
                replica_id,
                disclosure_epoch: 9,
                entity_kind: 1,
                components: BTreeMap::new(),
            });
            enqueue_lifecycle(
                &tx,
                BridgeLifecycleBatch {
                    sequence: replica_id,
                    team_id: 2,
                    authoritative_tick: replica_id,
                    replica_tick: replica_id,
                    view_epoch: 7,
                    events: vec![RenderLifecycleAction::Forget {
                        replica_id,
                        disclosure_epoch: 9,
                    }],
                },
            )
            .unwrap();
        }
        bridge.apply(&FilteredRenderSnapshot {
            team_id: 2,
            replica_tick: 99,
            entities: stale_entities.clone(),
            public_events: Vec::new(),
            external_effects: Vec::new(),
            memory_directives: Vec::new(),
        });
        assert_eq!(bridge.deterministic_count(), 20);

        while let Ok(batch) = rx.try_recv() {
            bridge.apply_lifecycle(&batch);
        }
        assert_eq!(bridge.deterministic_count(), 0);

        bridge.apply(&FilteredRenderSnapshot {
            team_id: 2,
            replica_tick: 99,
            entities: stale_entities,
            public_events: Vec::new(),
            external_effects: Vec::new(),
            memory_directives: Vec::new(),
        });
        assert_eq!(bridge.deterministic_count(), 0);
    }
}
