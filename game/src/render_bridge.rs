//! Phase 4.2: bridge from sim World snapshot → Fyrox scene.
//!
//! Phase 5.x: per-entity sprites + HP bars are now driven by Game-side
//! batched meshes (`body_batch` / `hp_batch`) so 1000+ entities collapse
//! into a handful of draw calls instead of one node per entity. This
//! module retains responsibility for **path** rendering only — checkpoint
//! dots + segment lines are static after the first non-empty snapshot,
//! so individual `RectangleBuilder` nodes there are fine.
//!
//! Entities with `EntityKind::Other` are never rendered.

use fyrox::core::algebra::{Vector2, Vector3};
use fyrox::core::color::Color;
use fyrox::core::pool::Handle;
use fyrox::graph::prelude::*;
use fyrox::scene::base::BaseBuilder;
use fyrox::scene::dim2::rectangle::RectangleBuilder;
use fyrox::scene::transform::TransformBuilder;
use fyrox::scene::{node::Node, Scene};

use crate::sim_runner::{EntityKind, EntityRenderData, SimWorldSnapshot};

const WORLD_SCALE: f32 = 0.01;
const Z_RB_PATH: f32 = 4.4;
/// Phase 4.1: Z layer for BlockedRegion outlines. Drawn slightly above
/// the path layer so the red outline is visible if a region overlaps a
/// path (legacy reference: same Z as the old map.regions debug overlay).
const Z_RB_REGION: f32 = 4.5;
/// Phase 4.1: Region outline thickness (render units). Half the path
/// thickness so the red border doesn't dominate over the cream path.
const REGION_LINE_THICKNESS: f32 = PATH_LINE_THICKNESS * 0.5;
/// Phase 4.1: Region outline color — red (matches the legacy
/// "blocked region" overlay convention; the alternative orange is
/// reserved for `circle` blockers when they exist).
const REGION_OUTLINE_COLOR: (u8, u8, u8, u8) = (255, 80, 80, 255);
/// Phase 4.1: Optional circle blocker color — orange. Currently unused
/// (omb `BlockedRegion` has no radius), but plumbed for forward-compat.
const REGION_CIRCLE_COLOR: (u8, u8, u8, u8) = (255, 165, 0, 255);

/// Path zigzag line thickness in render units. Computed `64.0 *
/// WORLD_SCALE * 2.0 = 1.28`. Matches the legacy MVP "thick cream
/// zigzag" reference (image 6) — the prior `0.12` value was the tail
/// end of an earlier per-segment marker design. The thicker line
/// covers the corner waypoints cleanly, so individual checkpoint
/// dots are no longer needed.
const PATH_LINE_THICKNESS: f32 = 64.0 * WORLD_SCALE * 2.0;
/// Pale tan / cream path color (RGBA). Replaces the earlier
/// `(255, 200, 60)` yellow.
const PATH_COLOR: (u8, u8, u8, u8) = (170, 140, 90, 255);

#[derive(Default, Debug)]
pub struct RenderBridge {
    last_applied_tick: Option<u32>,
    path_nodes: Vec<Handle<Node>>,
    paths_drawn: bool,
    /// Phase 4.1: BlockedRegion outline scene nodes (one segment per
    /// polygon edge). Static after first draw — `regions_drawn` gates
    /// re-creation just like `paths_drawn`.
    region_nodes: Vec<Handle<Node>>,
    regions_drawn: bool,
}

impl RenderBridge {
    pub fn new() -> Self {
        Self::default()
    }

    /// Per-tick path-init + diagnostic log. Entity sprites + HP bars are
    /// now drawn by `Game::update_sim_batches` via batched meshes.
    pub fn update(&mut self, snapshot: &SimWorldSnapshot, scene: &mut Scene) {
        if self.last_applied_tick == Some(snapshot.tick) {
            return;
        }
        self.last_applied_tick = Some(snapshot.tick);

        self.ensure_paths_drawn(&snapshot.paths, scene);
        self.ensure_blocked_regions_drawn(&snapshot.blocked_regions, scene);

        if snapshot.tick % 60 == 0 {
            log::debug!(
                "render_bridge: tick={} entities={} kinds={:?} paths_drawn={} regions_drawn={}",
                snapshot.tick,
                snapshot.entities.len(),
                kind_histogram(&snapshot.entities),
                self.paths_drawn,
                self.regions_drawn,
            );
        }
    }

    /// Phase 4.1: draw red polygon outlines for each BlockedRegion plus
    /// optional orange filled circle markers (forward-compat — no source
    /// data has a radius today). One-shot like `ensure_paths_drawn` —
    /// regions are static after `state::initialization`. Called every
    /// tick but only does work the first time `blocked_regions` is
    /// non-empty (matches the lazy paths-init pattern, since the omb
    /// scene loader populates the resource before the first dispatch
    /// but TD_1 has zero regions and won't trigger the draw at all).
    fn ensure_blocked_regions_drawn(
        &mut self,
        regions: &[crate::sim_runner::BlockedRegionSnapshot],
        scene: &mut Scene,
    ) {
        if self.regions_drawn || regions.is_empty() {
            return;
        }
        let mut polygon_segments: usize = 0;
        let mut circles: usize = 0;
        for region in regions {
            if region.points.len() >= 2 {
                // Polygon outline = N edges; close the loop with last→first.
                let n = region.points.len();
                for i in 0..n {
                    let (x1, y1) = region.points[i];
                    let (x2, y2) = region.points[(i + 1) % n];
                    let rx1 = -x1 * WORLD_SCALE;
                    let ry1 = y1 * WORLD_SCALE;
                    let rx2 = -x2 * WORLD_SCALE;
                    let ry2 = y2 * WORLD_SCALE;
                    let dx = rx2 - rx1;
                    let dy = ry2 - ry1;
                    let len = (dx * dx + dy * dy).sqrt().max(f32::EPSILON);
                    let mid_x = (rx1 + rx2) * 0.5;
                    let mid_y = (ry1 + ry2) * 0.5;
                    let angle = dy.atan2(dx);
                    let seg: Handle<Node> = RectangleBuilder::new(
                        BaseBuilder::new().with_local_transform(
                            TransformBuilder::new()
                                .with_local_position(Vector3::new(mid_x, mid_y, Z_RB_REGION))
                                .with_local_rotation(
                                    fyrox::core::algebra::UnitQuaternion::from_axis_angle(
                                        &fyrox::core::algebra::Vector3::z_axis(),
                                        angle,
                                    ),
                                )
                                .with_local_scale(Vector3::new(
                                    len,
                                    REGION_LINE_THICKNESS,
                                    f32::EPSILON,
                                ))
                                .build(),
                        ),
                    )
                    .with_color(Color::from_rgba(
                        REGION_OUTLINE_COLOR.0,
                        REGION_OUTLINE_COLOR.1,
                        REGION_OUTLINE_COLOR.2,
                        REGION_OUTLINE_COLOR.3,
                    ))
                    .build(&mut scene.graph)
                    .transmute();
                    self.region_nodes.push(seg);
                    polygon_segments += 1;
                }
            }
            // Optional circle marker — currently never set since omb
            // BlockedRegion has no radius field. Plumbed so future
            // circular blockers (e.g. tower footprint) can land here
            // without touching render code.
            if let Some(((cx, cy), r)) = region.circle {
                let rx = -cx * WORLD_SCALE;
                let ry = cy * WORLD_SCALE;
                let rr = r * WORLD_SCALE;
                // Approximate circle with a square sprite — cheap and
                // good enough for a debug overlay; renderer can swap to
                // a proper circle texture later.
                let node: Handle<Node> = RectangleBuilder::new(
                    BaseBuilder::new().with_local_transform(
                        TransformBuilder::new()
                            .with_local_position(Vector3::new(rx, ry, Z_RB_REGION))
                            .with_local_scale(Vector3::new(rr * 2.0, rr * 2.0, f32::EPSILON))
                            .build(),
                    ),
                )
                .with_color(Color::from_rgba(
                    REGION_CIRCLE_COLOR.0,
                    REGION_CIRCLE_COLOR.1,
                    REGION_CIRCLE_COLOR.2,
                    REGION_CIRCLE_COLOR.3,
                ))
                .build(&mut scene.graph)
                .transmute();
                self.region_nodes.push(node);
                circles += 1;
            }
        }
        self.regions_drawn = true;
        log::info!(
            "render_bridge: drew {} region segments + {} circle markers for {} region(s)",
            polygon_segments,
            circles,
            regions.len()
        );
    }

    fn ensure_paths_drawn(&mut self, paths: &[Vec<(f32, f32)>], scene: &mut Scene) {
        if self.paths_drawn || paths.is_empty() {
            return;
        }
        for path in paths {
            // Phase 3.1: per-checkpoint marker dots removed — the thicker
            // PATH_LINE_THICKNESS line covers corners cleanly.
            for window in path.windows(2) {
                let (x1, y1) = window[0];
                let (x2, y2) = window[1];
                let rx1 = -x1 * WORLD_SCALE;
                let ry1 = y1 * WORLD_SCALE;
                let rx2 = -x2 * WORLD_SCALE;
                let ry2 = y2 * WORLD_SCALE;
                let dx = rx2 - rx1;
                let dy = ry2 - ry1;
                let len = (dx * dx + dy * dy).sqrt().max(f32::EPSILON);
                let mid_x = (rx1 + rx2) * 0.5;
                let mid_y = (ry1 + ry2) * 0.5;
                let angle = dy.atan2(dx);
                let seg: Handle<Node> = RectangleBuilder::new(
                    BaseBuilder::new().with_local_transform(
                        TransformBuilder::new()
                            .with_local_position(Vector3::new(mid_x, mid_y, Z_RB_PATH))
                            .with_local_rotation(fyrox::core::algebra::UnitQuaternion::from_axis_angle(
                                &fyrox::core::algebra::Vector3::z_axis(),
                                angle,
                            ))
                            .with_local_scale(Vector3::new(len, PATH_LINE_THICKNESS, f32::EPSILON))
                            .build(),
                    ),
                )
                .with_color(Color::from_rgba(PATH_COLOR.0, PATH_COLOR.1, PATH_COLOR.2, PATH_COLOR.3))
                .build(&mut scene.graph)
                .transmute();
                self.path_nodes.push(seg);
            }
        }
        self.paths_drawn = true;
        log::info!("render_bridge: drew {} path nodes for {} path(s)",
            self.path_nodes.len(), paths.len());
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

/// Per-entity batched-mesh slot ownership. lib.rs holds a HashMap<u32,
/// SimEntitySlots> and reuses these slot indices each tick. body_slot is
/// allocated unconditionally; hp_* slots only when max_hp > 0.
#[derive(Debug, Clone, Copy)]
pub struct SimEntitySlots {
    pub body_slot: u32,
    pub hp_bg_slot: Option<u32>,
    pub hp_fg_slot: Option<u32>,
}

/// Style returned to the lib.rs batched-mesh writer. Color comes from a
/// kind-driven baseline + unit_id hash perturbation so dart / bomb / ice
/// towers (etc.) are visually distinct without needing real textures yet.
pub fn style_for_entity(entity: &EntityRenderData) -> ([u8; 4], f32, f32) {
    let (base_rgb, size, z) = match entity.kind {
        EntityKind::Hero => ((0u8, 255u8, 200u8), 0.30, 1.9),
        EntityKind::Tower => ((120, 120, 255), 0.32, 2.4),
        EntityKind::Creep => ((255, 100, 220), 0.22, 1.95),
        EntityKind::Projectile => ((255, 255, 0), 0.06, 0.4),
        EntityKind::Other => ((180, 180, 180), 0.20, 1.85),
    };
    let hash = hash_unit_id(&entity.unit_id);
    let (r, g, b) = rotate_rgb(base_rgb, hash);
    ([r, g, b, 255], size, z)
}

/// World→render coord. Same `-x` flip + WORLD_SCALE the legacy entity_create
/// + body_batch path used; matches the camera's look_at_rh side vector.
pub fn world_to_render(entity: &EntityRenderData) -> Vector2<f32> {
    Vector2::new(-entity.pos_x * WORLD_SCALE, entity.pos_y * WORLD_SCALE)
}

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

fn rotate_rgb(base: (u8, u8, u8), hash: u8) -> (u8, u8, u8) {
    let h = hash as i32;
    let r = base.0 as i32 ^ ((h * 7) & 0xFF);
    let g = base.1 as i32 ^ ((h * 13) & 0xFF);
    let b = base.2 as i32 ^ ((h * 23) & 0xFF);
    let bump = |v: i32| -> u8 {
        let v = v.clamp(0, 255);
        if v < 80 { (v + 80) as u8 }
        else if v > 200 { v as u8 }
        else { (v + 40).min(255) as u8 }
    };
    (bump(r), bump(g), bump(b))
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
        assert_eq!(b.last_applied_tick, None);
        assert!(!b.paths_drawn);
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
    fn unit_id_changes_color_within_same_kind() {
        let mut a = make_entity(1, EntityKind::Tower, 0.0, 0.0);
        a.unit_id = "tower_dart_monkey".to_string();
        let mut b = make_entity(2, EntityKind::Tower, 0.0, 0.0);
        b.unit_id = "tower_bomb_shooter".to_string();
        let (ca, _, _) = style_for_entity(&a);
        let (cb, _, _) = style_for_entity(&b);
        assert_ne!((ca[0], ca[1], ca[2]), (cb[0], cb[1], cb[2]));
    }
}
