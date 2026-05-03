//! Phase 4.2: bridge from sim World snapshot → Fyrox scene.
//!
//! Reads `SimWorldSnapshot.entities` once per frame and spawns / updates /
//! despawns 2D rectangle sprites for each entity. The architectural goal
//! — sim → render data flow — is now visually observable: sprites driven
//! by `sim_runner` (authoritative ECS) appear on screen alongside the
//! legacy `NetworkBridge` GameEvent → sprite pipeline (Phase 4.5 cuts the
//! legacy path).
//!
//! # Phase 4.2 scope
//!
//! - Per-EntityKind colored quad with simple geometry (cube replacement
//!   for the eventual textured sprite). Color/size tuned to be
//!   distinguishable from the legacy `NetworkBridge` sprites for
//!   debugging — render_bridge sprites are smaller and use a slight Z
//!   offset, so during the parallel period (Phase 4.2 → 4.5) one can
//!   visually tell which pipeline is alive.
//! - Position from `entity.pos_x` / `pos_y` via `WORLD_SCALE` and the
//!   omfx X-flip convention (`-x`).
//! - Despawn nodes whose `entity_id` is missing from the current
//!   snapshot (death / cleanup).
//!
//! NOT included (Phase 4-followup if needed):
//! - Real game textures from the `sprite_resources` pipeline.
//! - HP bars, facing arrows, name labels.
//! - Floating damage text.
//! - Animations / interpolation between ticks (sprites jump per snapshot
//!   tick which at 60Hz looks fine; Phase 5+ may add per-frame lerp).

use std::collections::{HashMap, HashSet};

use fyrox::core::algebra::Vector3;
use fyrox::core::color::Color;
use fyrox::core::pool::Handle;
use fyrox::graph::prelude::*;
use fyrox::scene::base::BaseBuilder;
use fyrox::scene::dim2::rectangle::RectangleBuilder;
use fyrox::scene::transform::TransformBuilder;
use fyrox::scene::{node::Node, Scene};

use crate::sim_runner::{EntityKind, EntityRenderData, SimWorldSnapshot};

// Coordinate transform constants. Kept local (rather than imported from
// `lib.rs`) so render_bridge stays a leaf module without circular access
// to Game-private constants. These mirror the `WORLD_SCALE` / Z-layer
// values used by `NetworkBridge`'s `entity_create`.
//
// Phase 4.5 (NetworkBridge cut) is a natural moment to consolidate these
// to a `pub(crate) const` in lib.rs and `use` from there.
const WORLD_SCALE: f32 = 0.01;

// Slight Z offsets vs. the legacy NetworkBridge constants (Z_ENEMY=2.0,
// Z_TOWER=2.5, Z_BULLET=0.5) so render_bridge sprites render *in front
// of* their legacy counterparts during the parallel period — making it
// trivial to confirm visually that sim_runner is driving the renderer.
// Phase 4.5 deletes the legacy pipeline; at that point we can drop these
// offsets back to the canonical Z values.
const Z_RB_BULLET: f32 = 0.4;
const Z_RB_HERO: f32 = 1.9;
const Z_RB_CREEP: f32 = 1.95;
const Z_RB_TOWER: f32 = 2.4;
const Z_RB_OTHER: f32 = 1.85;

/// Mapping `entity_id → Fyrox node handle`. Each entity in the
/// `SimWorldSnapshot` owns exactly one rectangle node, which is updated
/// in place per snapshot tick and removed when the entity drops out of
/// the snapshot.
#[derive(Default, Debug)]
pub struct RenderBridge {
    entity_nodes: HashMap<u32, Handle<Node>>,
    /// Last applied snapshot tick — used to throttle work in `update`.
    /// Snapshots arriving with the same tick are skipped to avoid
    /// re-mutating the scene graph at render fps when sim tps is lower.
    last_applied_tick: Option<u32>,
    /// Diagnostic counter: total spawn calls since process start. Used
    /// by the periodic info log so console output reflects real
    /// activity, not just snapshot tick counts.
    spawn_count: u64,
    /// Diagnostic counter: total despawn calls since process start.
    despawn_count: u64,
}

impl RenderBridge {
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply one snapshot to the scene. Spawns / updates / despawns
    /// rectangle nodes per `entity.entity_id` to mirror sim state.
    pub fn update(&mut self, snapshot: &SimWorldSnapshot, scene: &mut Scene) {
        // Throttle: render thread runs at display fps (60-240Hz), sim
        // ticks at 60Hz wallclock — most frames see the same snapshot.
        // Skip when the tick hasn't advanced.
        if self.last_applied_tick == Some(snapshot.tick) {
            return;
        }
        self.last_applied_tick = Some(snapshot.tick);

        let mut alive = HashSet::with_capacity(snapshot.entities.len());
        for entity in &snapshot.entities {
            alive.insert(entity.entity_id);
            self.update_or_spawn(entity, scene);
        }

        // Despawn entities not present in this snapshot.
        let to_remove: Vec<u32> = self
            .entity_nodes
            .keys()
            .filter(|id| !alive.contains(id))
            .copied()
            .collect();
        for id in to_remove {
            if let Some(handle) = self.entity_nodes.remove(&id) {
                scene.graph.remove_node(handle);
                self.despawn_count += 1;
                log::trace!("render_bridge: despawn entity_id={}", id);
            }
        }

        // Periodic sample log so console isn't silent. Every 60 sim
        // ticks (~1s @ 60Hz). Keys: tracked count + cumulative spawn /
        // despawn → quick health check that the pipeline is doing work.
        if snapshot.tick % 60 == 0 {
            log::debug!(
                "render_bridge: tick={} entities={} kinds={:?} tracked={} spawned={} despawned={}",
                snapshot.tick,
                snapshot.entities.len(),
                kind_histogram(&snapshot.entities),
                self.entity_nodes.len(),
                self.spawn_count,
                self.despawn_count,
            );
        }
    }

    fn update_or_spawn(&mut self, entity: &EntityRenderData, scene: &mut Scene) {
        // World → render coords. omfx convention: `-x` X-flip (the camera
        // is set up so world +X maps to screen -X; see lib.rs comments
        // around `Z_BULLET`). `pos_y` maps directly to render Y.
        let render_x = -entity.pos_x * WORLD_SCALE;
        let render_y = entity.pos_y * WORLD_SCALE;
        let z = z_for_kind(entity.kind);

        if let Some(&handle) = self.entity_nodes.get(&entity.entity_id) {
            // Existing node — just update transform.
            if let Ok(node) = scene.graph.try_get_mut(handle) {
                node.local_transform_mut()
                    .set_position(Vector3::new(render_x, render_y, z));
                // facing_rad: store on the entity but skip rotating the
                // 2D rectangle for now. The dim2 RectangleBuilder
                // sprites point + Y by default; rotating a colored quad
                // doesn't add visual signal at Phase 4.2 scope. When
                // textures land (Phase 4-followup) facing rotation goes
                // here via `.set_rotation_2d(entity.facing_rad)` or
                // similar API.
            } else {
                // Stale handle (node was somehow removed externally).
                // Drop the entry so next tick spawns a fresh sprite.
                self.entity_nodes.remove(&entity.entity_id);
            }
        } else {
            // First-time spawn for this entity.
            let (color, size) = style_for_entity(entity);
            let handle: Handle<Node> = RectangleBuilder::new(
                BaseBuilder::new().with_local_transform(
                    TransformBuilder::new()
                        .with_local_position(Vector3::new(render_x, render_y, z))
                        .with_local_scale(Vector3::new(size, size, f32::EPSILON))
                        .build(),
                ),
            )
            .with_color(color)
            .build(&mut scene.graph)
            .transmute();

            self.entity_nodes.insert(entity.entity_id, handle);
            self.spawn_count += 1;
            log::trace!(
                "render_bridge: spawn entity_id={} kind={:?} pos=({:.2}, {:.2})",
                entity.entity_id,
                entity.kind,
                entity.pos_x,
                entity.pos_y,
            );
        }
    }

    /// Number of entities currently tracked. Used by tests and by the
    /// Phase 3.5 integration harness for sanity assertions.
    pub fn tracked_count(&self) -> usize {
        self.entity_nodes.len()
    }
}

/// Per-EntityKind visual style. Colors picked to be obviously different
/// from the legacy NetworkBridge palette. render_bridge uses high-saturation
/// neon variants so visual identification is easy. Within a kind, the
/// `unit_id` (e.g. "tower_dart_monkey", "tower_bomb_shooter") drives a
/// hue rotation so different tower types / hero types are distinguishable
/// at a glance.
fn style_for_entity(entity: &EntityRenderData) -> (Color, f32) {
    let (base_rgb, size) = match entity.kind {
        // (r, g, b) baseline + size
        EntityKind::Hero => ((0u8, 255u8, 200u8), 0.30),
        EntityKind::Tower => ((120, 120, 255), 0.32),
        EntityKind::Creep => ((255, 100, 220), 0.22),
        EntityKind::Projectile => ((255, 255, 0), 0.06),
        EntityKind::Other => ((180, 180, 180), 0.20),
    };
    // unit_id-based hue rotation: deterministic, distinguishable.
    // Hash the unit_id into a small u8 then RGB-rotate the channels by it.
    // This keeps the same kind clearly identifiable (size unchanged) but
    // gives e.g. dart / bomb / tack / ice towers different colours.
    let hash = hash_unit_id(&entity.unit_id);
    let (r, g, b) = rotate_rgb(base_rgb, hash);
    (Color::from_rgba(r, g, b, 255), size)
}

/// Tiny FNV-like u8 hash over `unit_id`. Empty `unit_id` → 0 (use
/// the kind's baseline color unchanged).
fn hash_unit_id(unit_id: &str) -> u8 {
    if unit_id.is_empty() {
        return 0;
    }
    let mut h: u32 = 2166136261;
    for b in unit_id.as_bytes() {
        h ^= *b as u32;
        h = h.wrapping_mul(16777619);
    }
    (h & 0xFF) as u8
}

/// Rotate / perturb an (R, G, B) baseline by `hash`. Keeps the same
/// kind family recognisable while making sub-types visually distinct.
/// Bytes are wrapped, not clamped — full color-wheel sweep available.
fn rotate_rgb(base: (u8, u8, u8), hash: u8) -> (u8, u8, u8) {
    let h = hash as i32;
    let r = base.0 as i32 ^ ((h * 7) & 0xFF);
    let g = base.1 as i32 ^ ((h * 13) & 0xFF);
    let b = base.2 as i32 ^ ((h * 23) & 0xFF);
    // Re-saturate to keep it visible against the green background:
    // bias each channel away from mid-grey so neon character is preserved.
    let bump = |v: i32| -> u8 {
        let v = v.clamp(0, 255);
        if v < 80 { (v + 80) as u8 }
        else if v > 200 { v as u8 }
        else { (v + 40).min(255) as u8 }
    };
    (bump(r), bump(g), bump(b))
}

fn z_for_kind(kind: EntityKind) -> f32 {
    match kind {
        EntityKind::Hero => Z_RB_HERO,
        EntityKind::Tower => Z_RB_TOWER,
        EntityKind::Creep => Z_RB_CREEP,
        EntityKind::Projectile => Z_RB_BULLET,
        EntityKind::Other => Z_RB_OTHER,
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
            ..Default::default()
        }
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

    #[test]
    fn style_for_kind_distinct_colors() {
        // Sanity: each kind picks a different color so the parallel
        // render of NetworkBridge + render_bridge stays visually
        // distinguishable. We don't pin the exact RGB (Phase 4-followup
        // may retune the palette), only that no two kinds collide.
        let kinds = [
            EntityKind::Hero,
            EntityKind::Tower,
            EntityKind::Creep,
            EntityKind::Projectile,
            EntityKind::Other,
        ];
        let mut seen = std::collections::HashSet::new();
        for k in kinds {
            let e = make_entity(0, k, 0.0, 0.0);
            let (c, _) = style_for_entity(&e);
            assert!(seen.insert((c.r, c.g, c.b)), "duplicate color for {:?}", k);
        }
    }

    #[test]
    fn unit_id_changes_color_within_same_kind() {
        // Two towers with different unit_id should get different colors so
        // dart / bomb / tack / ice are visually distinguishable on screen.
        let mut a = make_entity(1, EntityKind::Tower, 0.0, 0.0);
        a.unit_id = "tower_dart_monkey".to_string();
        let mut b = make_entity(2, EntityKind::Tower, 0.0, 0.0);
        b.unit_id = "tower_bomb_shooter".to_string();
        let (ca, _) = style_for_entity(&a);
        let (cb, _) = style_for_entity(&b);
        assert_ne!((ca.r, ca.g, ca.b), (cb.r, cb.g, cb.b));
    }

    #[test]
    fn z_for_kind_layers_above_background() {
        // All sprite Z values must be > 0 so they render in front of
        // the dark-green background quad (Z_BACKGROUND=4.5 in lib.rs;
        // smaller Z = closer to camera). Practically Z is in [0.4, 2.4].
        for k in [
            EntityKind::Hero,
            EntityKind::Tower,
            EntityKind::Creep,
            EntityKind::Projectile,
            EntityKind::Other,
        ] {
            let z = z_for_kind(k);
            assert!(z > 0.0 && z < 4.5, "z out of range for {:?}: {}", k, z);
        }
    }
}
