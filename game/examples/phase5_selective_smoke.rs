use std::collections::BTreeSet;
use std::sync::Arc;

use omoba_core::runtime::*;
use omoba_sim::{Fixed64, Vec2};
use omfx::filtered_render_bridge::{FilteredRenderBridge, FilteredSceneAction};

fn main() -> Result<(), String> {
    let team = 2;
    let tick = 90;
    let canonical = (5u64 << 32) | 8;
    let mut projector = TeamViewProjector::new(team, TeamProjectorConfig::default());
    let start = projector.build_team_game_start(tick, 120);
    let mut replica = SelectiveReplicaRuntime::bootstrap_from_team_game_start(
        &start, BTreeSet::new(), BTreeSet::new(),
    ).map_err(|error| format!("join: {error:?}"))?;
    let view = WaveBReadView {
        tick,
        entities: Arc::from([CommittedEntityView {
            canonical_id: canonical,
            team,
            position: Vec2::new(Fixed64::ZERO, Fixed64::ZERO),
            scope: ReplicationScopeKind::Public,
            owner_team: Some(team),
            stealth_level: 0,
            overrides: Vec::new(),
            remember: RememberDisposition::LastKnown,
            disclosed_baseline: 0u32.to_be_bytes().to_vec(),
        }]),
        vision_sources: Arc::from([]),
    };
    let mut visibility = TeamVisibilityState::new(team, 16);
    let transitions = visibility.resolve(&view, 0);
    let frame = projector.build_frame(
        tick, tick, &visibility.index.current, transitions, &[], &ProjectionDependencyGraph::default(),
    ).map_err(|error| format!("projection: {error:?}"))?;
    let result = replica.apply_encoded_frame(&frame.wire_bytes, &mut NoopDisclosedWorldStepper)
        .map_err(|error| format!("step: {error:?}"))?;
    if !matches!(result, FrameApplyResult::Applied { .. }) { return Err("frame not applied".into()); }
    let render = replica.extract_filtered_render_snapshot();
    let actions = FilteredRenderBridge::default().apply(&render);
    if !actions.iter().any(|action| matches!(action, FilteredSceneAction::UpsertDeterministic(_))) {
        return Err("render handoff lacked deterministic upsert".into());
    }
    println!(
        "phase5-selective-smoke ok team={} joined=true stepped=true render_actions={} acceptance=false",
        team, actions.len(),
    );
    Ok(())
}
