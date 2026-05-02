//! Phase 3.4: bridge from sim World snapshot → Fyrox scene.
//!
//! Reads `SimWorldSnapshot.entities` once per frame and (Phase 3.4 stub)
//! logs entity render data. The architectural goal — sim → render data
//! flow — is wired here even if visual output is currently logs.
//!
//! # Phase 4 plan
//!
//! - For each `EntityRenderData` in the snapshot, look up or spawn a Fyrox
//!   sprite (`scene.graph` `RectangleBuilder` keyed by `entity_id`).
//! - Update `local_position` from `pos_x` / `pos_y` (apply
//!   `WORLD_SCALE`).
//! - Update rotation from `facing_rad` (around Z for top-down).
//! - For entities with `hp / max_hp > 0`, spawn / update HP bar children
//!   (re-using the existing batched HP sprite mesh in `lib.rs`).
//! - Despawn (`scene.graph.remove_node(handle)`) entities present in
//!   `entity_nodes` but missing from the snapshot.
//!
//! Phase 4 will likely retire the existing `NetworkBridge` GameEvent →
//! sprite pipeline once the lockstep render path covers all visual
//! categories (Hero/Tower/Creep/Projectile + HP bars + facing arrows).

use std::collections::{HashMap, HashSet};

use fyrox::core::pool::Handle;
use fyrox::scene::{node::Node, Scene};

use crate::sim_runner::{EntityKind, EntityRenderData, SimWorldSnapshot};

/// Mapping `entity_id → Fyrox node handle`. Phase 3.4 keeps the table empty
/// since spawn / despawn is still a stub; Phase 4 wires real sprites.
#[derive(Default, Debug)]
pub struct RenderBridge {
    entity_nodes: HashMap<u32, Handle<Node>>,
    /// Last applied snapshot tick — used to throttle the per-entity log
    /// in `update`. Snapshots arriving with the same tick are skipped to
    /// avoid spamming the trace log when render fps > sim tps.
    last_applied_tick: Option<u32>,
}

impl RenderBridge {
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply one snapshot to the scene. `_scene` is currently unused
    /// (Phase 3.4 stub) but threaded through so the Phase 4 wiring is a
    /// drop-in replacement of the body, not a signature change.
    pub fn update(&mut self, snapshot: &SimWorldSnapshot, _scene: &mut Scene) {
        // Throttle: render thread runs at display fps (60-240Hz), sim ticks
        // at 60Hz wallclock — most frames receive the same snapshot. Only
        // log/process when the snapshot tick advanced. Phase 4 will need
        // to interpolate (or run sim in lockstep with render) so this
        // throttle becomes unnecessary.
        if self.last_applied_tick == Some(snapshot.tick) {
            return;
        }
        self.last_applied_tick = Some(snapshot.tick);

        let mut alive = HashSet::with_capacity(snapshot.entities.len());
        for entity in &snapshot.entities {
            alive.insert(entity.entity_id);
            self.update_or_spawn(entity, _scene);
        }

        // Despawn entities not in snapshot. Phase 3.4 stub: just clear
        // the entry; Phase 4 also calls `scene.graph.remove_node`.
        let to_remove: Vec<u32> = self
            .entity_nodes
            .keys()
            .filter(|id| !alive.contains(id))
            .copied()
            .collect();
        for id in to_remove {
            if let Some(_node) = self.entity_nodes.remove(&id) {
                // Phase 4: scene.graph.remove_node(_node);
                log::trace!("render_bridge: despawn entity_id={}", id);
            }
        }

        // Periodic sample log so console isn't completely silent while
        // the bridge is in stub mode. Every 60 sim ticks (~1s @ 60Hz).
        if snapshot.tick % 60 == 0 {
            log::debug!(
                "render_bridge: snapshot tick={} entities={} kinds={:?}",
                snapshot.tick,
                snapshot.entities.len(),
                kind_histogram(&snapshot.entities),
            );
        }
    }

    fn update_or_spawn(&mut self, entity: &EntityRenderData, _scene: &mut Scene) {
        // Phase 3.4 stub: log only. Phase 4 will:
        //   - if !entity_nodes.contains_key(&entity.entity_id):
        //         spawn a sprite (RectangleBuilder) for entity.kind, store
        //         the Handle in entity_nodes.
        //   - else:
        //         update the existing node's local_transform position +
        //         rotation, and (if HP changed) the HP bar.
        log::trace!(
            "render_bridge: entity {} kind={:?} pos=({:.2}, {:.2}) facing={:.2} hp={}/{}",
            entity.entity_id,
            entity.kind,
            entity.pos_x,
            entity.pos_y,
            entity.facing_rad,
            entity.hp,
            entity.max_hp,
        );
    }

    /// Number of entities currently tracked. Used by tests and by the
    /// Phase 3.5 integration harness for sanity assertions.
    pub fn tracked_count(&self) -> usize {
        self.entity_nodes.len()
    }
}

fn kind_histogram(entities: &[EntityRenderData]) -> [usize; 5] {
    let mut h = [0usize; 5];
    for e in entities {
        let i = match e.kind {
            EntityKind::Hero => 0,
            EntityKind::Tower => 1,
            EntityKind::Creep => 2,
            EntityKind::Projectile => 3,
            EntityKind::Other => 4,
        };
        h[i] += 1;
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entity(id: u32, kind: EntityKind, x: f32, y: f32) -> EntityRenderData {
        EntityRenderData {
            entity_id: id,
            entity_gen: 1,
            kind,
            pos_x: x,
            pos_y: y,
            facing_rad: 0.0,
            hp: 100,
            max_hp: 100,
        }
    }

    fn make_snapshot(tick: u32, entities: Vec<EntityRenderData>) -> SimWorldSnapshot {
        SimWorldSnapshot { tick, entities }
    }

    #[test]
    fn new_bridge_is_empty() {
        let b = RenderBridge::new();
        assert_eq!(b.tracked_count(), 0);
        assert_eq!(b.last_applied_tick, None);
    }

    #[test]
    fn kind_histogram_counts_correctly() {
        let entities = vec![
            make_entity(1, EntityKind::Hero, 0.0, 0.0),
            make_entity(2, EntityKind::Tower, 0.0, 0.0),
            make_entity(3, EntityKind::Tower, 0.0, 0.0),
            make_entity(4, EntityKind::Creep, 0.0, 0.0),
            make_entity(5, EntityKind::Projectile, 0.0, 0.0),
        ];
        let h = kind_histogram(&entities);
        assert_eq!(h, [1, 2, 1, 1, 0]);
    }
}
