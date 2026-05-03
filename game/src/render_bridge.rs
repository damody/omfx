//! Phase 4.2: bridge from sim World snapshot → Fyrox scene.
//!
//! Reads `SimWorldSnapshot.entities` once per frame and spawns / updates /
//! despawns 2D rectangle sprites for each entity, with a child HP bar.
//! Also draws creep checkpoint paths once on first non-empty snapshot.
//!
//! What this owns:
//! - Per-entity sprite (colored quad) + HP bar children
//! - Path render line segments + checkpoint markers (static after first draw)
//!
//! What lib.rs still owns:
//! - UI name labels (need ui + camera + window for screen-space projection;
//!   easier to keep in lib.rs alongside other UI work).
//!
//! Entities with `EntityKind::Other` are explicitly skipped — those are
//! internal ECS rows (e.g. RegionBlocker) that should not appear visually.

use std::collections::{HashMap, HashSet};

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

const Z_RB_BULLET: f32 = 0.4;
const Z_RB_HERO: f32 = 1.9;
const Z_RB_CREEP: f32 = 1.95;
const Z_RB_TOWER: f32 = 2.4;
const Z_RB_PATH: f32 = 4.4;        // just above background (Z_BACKGROUND=4.5)
const Z_RB_HP_BAR: f32 = 1.5;      // in front of all entity sprites

/// Per-entity scene nodes owned by `RenderBridge`.
#[derive(Debug)]
struct EntityNode {
    sprite: Handle<Node>,
    /// Foreground HP bar (green/red rect) — width scaled by hp/max_hp.
    hp_bar_fg: Handle<Node>,
    /// Background HP bar (black backing) — sized to max width.
    hp_bar_bg: Handle<Node>,
    /// Cached size for HP bar foreground scaling (= sprite size × 0.9).
    hp_bar_max_width: f32,
}

/// Mapping `entity_id → EntityNode`. Each rendered entity owns a sprite
/// + child HP bar; updated in place per snapshot tick and removed when
/// the entity drops out.
#[derive(Default, Debug)]
pub struct RenderBridge {
    entities: HashMap<u32, EntityNode>,
    last_applied_tick: Option<u32>,
    spawn_count: u64,
    despawn_count: u64,
    /// Path render line segment + checkpoint marker handles. Drawn once
    /// on the first snapshot whose `paths` is non-empty; static after.
    path_nodes: Vec<Handle<Node>>,
    paths_drawn: bool,
}

impl RenderBridge {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn update(&mut self, snapshot: &SimWorldSnapshot, scene: &mut Scene) {
        if self.last_applied_tick == Some(snapshot.tick) {
            return;
        }
        self.last_applied_tick = Some(snapshot.tick);

        self.ensure_paths_drawn(&snapshot.paths, scene);

        let mut alive = HashSet::with_capacity(snapshot.entities.len());
        for entity in &snapshot.entities {
            // Internal ECS rows (RegionBlocker, etc.) have no Hero/Tower/
            // Creep/Projectile component and surface as `Other`. They are
            // not user-facing units — skip rendering entirely.
            if entity.kind == EntityKind::Other {
                continue;
            }
            alive.insert(entity.entity_id);
            self.update_or_spawn(entity, scene);
        }

        let to_remove: Vec<u32> = self
            .entities
            .keys()
            .filter(|id| !alive.contains(id))
            .copied()
            .collect();
        for id in to_remove {
            if let Some(node) = self.entities.remove(&id) {
                scene.graph.remove_node(node.sprite);
                scene.graph.remove_node(node.hp_bar_fg);
                scene.graph.remove_node(node.hp_bar_bg);
                self.despawn_count += 1;
            }
        }

        if snapshot.tick % 60 == 0 {
            log::debug!(
                "render_bridge: tick={} entities={} kinds={:?} tracked={} spawned={} despawned={} paths_drawn={}",
                snapshot.tick,
                snapshot.entities.len(),
                kind_histogram(&snapshot.entities),
                self.entities.len(),
                self.spawn_count,
                self.despawn_count,
                self.paths_drawn,
            );
        }
    }

    fn update_or_spawn(&mut self, entity: &EntityRenderData, scene: &mut Scene) {
        let render_x = -entity.pos_x * WORLD_SCALE;
        let render_y = entity.pos_y * WORLD_SCALE;
        let z = z_for_kind(entity.kind);

        if let Some(node) = self.entities.get(&entity.entity_id) {
            // Update existing nodes.
            if let Ok(sprite) = scene.graph.try_get_mut(node.sprite) {
                sprite
                    .local_transform_mut()
                    .set_position(Vector3::new(render_x, render_y, z));
            }
            // HP bar above sprite.
            update_hp_bar(scene, node, render_x, render_y, entity);
            return;
        }

        // First-time spawn for this entity.
        let (color, size) = style_for_entity(entity);
        let sprite: Handle<Node> = RectangleBuilder::new(
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

        let bar_max = size * 0.9;
        let bar_height = size * 0.10;
        let bar_offset_y = size * 0.55;

        let hp_bar_bg: Handle<Node> = RectangleBuilder::new(
            BaseBuilder::new().with_local_transform(
                TransformBuilder::new()
                    .with_local_position(Vector3::new(render_x, render_y + bar_offset_y, Z_RB_HP_BAR + 0.01))
                    .with_local_scale(Vector3::new(bar_max, bar_height, f32::EPSILON))
                    .build(),
            ),
        )
        .with_color(Color::from_rgba(0, 0, 0, 200))
        .build(&mut scene.graph)
        .transmute();

        let hp_bar_fg: Handle<Node> = RectangleBuilder::new(
            BaseBuilder::new().with_local_transform(
                TransformBuilder::new()
                    .with_local_position(Vector3::new(render_x, render_y + bar_offset_y, Z_RB_HP_BAR))
                    .with_local_scale(Vector3::new(bar_max, bar_height * 0.8, f32::EPSILON))
                    .build(),
            ),
        )
        .with_color(Color::from_rgba(40, 220, 60, 255))
        .build(&mut scene.graph)
        .transmute();

        let node = EntityNode {
            sprite,
            hp_bar_fg,
            hp_bar_bg,
            hp_bar_max_width: bar_max,
        };
        update_hp_bar(scene, &node, render_x, render_y, entity);
        self.entities.insert(entity.entity_id, node);
        self.spawn_count += 1;
    }

    fn ensure_paths_drawn(&mut self, paths: &[Vec<(f32, f32)>], scene: &mut Scene) {
        if self.paths_drawn || paths.is_empty() {
            return;
        }
        for path in paths {
            // Checkpoint marker dots.
            for &(wx, wy) in path {
                let rx = -wx * WORLD_SCALE;
                let ry = wy * WORLD_SCALE;
                let marker: Handle<Node> = RectangleBuilder::new(
                    BaseBuilder::new().with_local_transform(
                        TransformBuilder::new()
                            .with_local_position(Vector3::new(rx, ry, Z_RB_PATH - 0.01))
                            .with_local_scale(Vector3::new(0.4, 0.4, f32::EPSILON))
                            .build(),
                    ),
                )
                .with_color(Color::from_rgba(255, 220, 0, 230))
                .build(&mut scene.graph)
                .transmute();
                self.path_nodes.push(marker);
            }
            // Connecting segments — thin rectangles aligned + scaled to span
            // each pair of consecutive checkpoints.
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
                            .with_local_scale(Vector3::new(len, 0.12, f32::EPSILON))
                            .build(),
                    ),
                )
                .with_color(Color::from_rgba(255, 200, 60, 200))
                .build(&mut scene.graph)
                .transmute();
                self.path_nodes.push(seg);
            }
        }
        self.paths_drawn = true;
        log::info!("render_bridge: drew {} path nodes for {} path(s)",
            self.path_nodes.len(), paths.len());
    }

    pub fn tracked_count(&self) -> usize {
        self.entities.len()
    }

    /// Iterator over rendered entities for the lib.rs UI label loop.
    /// Returns `(entity_id, world_position)` so the caller can project
    /// onto screen and place a Text widget above each unit.
    pub fn iter_label_anchors<'a>(
        &'a self,
        snapshot: &'a SimWorldSnapshot,
    ) -> impl Iterator<Item = (u32, Vector2<f32>, &'a EntityRenderData)> + 'a {
        snapshot.entities.iter().filter_map(move |e| {
            if e.kind == EntityKind::Other {
                return None;
            }
            if !self.entities.contains_key(&e.entity_id) {
                return None;
            }
            let pos = Vector2::new(-e.pos_x * WORLD_SCALE, e.pos_y * WORLD_SCALE);
            Some((e.entity_id, pos, e))
        })
    }
}

fn update_hp_bar(
    scene: &mut Scene,
    node: &EntityNode,
    render_x: f32,
    render_y: f32,
    entity: &EntityRenderData,
) {
    let hp_frac = if entity.max_hp > 0 {
        (entity.hp as f32 / entity.max_hp as f32).clamp(0.0, 1.0)
    } else {
        1.0
    };
    let bar_width = node.hp_bar_max_width * hp_frac;
    // Red below 30%, yellow below 60%, green otherwise.
    let bar_color = if hp_frac < 0.30 {
        Color::from_rgba(220, 50, 50, 255)
    } else if hp_frac < 0.60 {
        Color::from_rgba(220, 200, 60, 255)
    } else {
        Color::from_rgba(40, 220, 60, 255)
    };
    let bar_offset_y = node.hp_bar_max_width / 0.9 * 0.55;

    if let Ok(bg) = scene.graph.try_get_mut(node.hp_bar_bg) {
        bg.local_transform_mut()
            .set_position(Vector3::new(render_x, render_y + bar_offset_y, Z_RB_HP_BAR + 0.01));
    }
    if let Ok(fg) = scene.graph.try_get_mut(node.hp_bar_fg) {
        fg.local_transform_mut()
            .set_position(Vector3::new(
                render_x - (node.hp_bar_max_width - bar_width) * 0.5,
                render_y + bar_offset_y,
                Z_RB_HP_BAR,
            ))
            .set_scale(Vector3::new(bar_width.max(f32::EPSILON), node.hp_bar_max_width / 0.9 * 0.10 * 0.8, f32::EPSILON));
        if let Some(rect) = fg.cast_mut::<fyrox::scene::dim2::rectangle::Rectangle>() {
            rect.set_color(bar_color);
        }
    }
}

fn style_for_entity(entity: &EntityRenderData) -> (Color, f32) {
    let (base_rgb, size) = match entity.kind {
        EntityKind::Hero => ((0u8, 255u8, 200u8), 0.30),
        EntityKind::Tower => ((120, 120, 255), 0.32),
        EntityKind::Creep => ((255, 100, 220), 0.22),
        EntityKind::Projectile => ((255, 255, 0), 0.06),
        // Other is skipped in update(); style retained only for tests.
        EntityKind::Other => ((180, 180, 180), 0.20),
    };
    let hash = hash_unit_id(&entity.unit_id);
    let (r, g, b) = rotate_rgb(base_rgb, hash);
    (Color::from_rgba(r, g, b, 255), size)
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

fn z_for_kind(kind: EntityKind) -> f32 {
    match kind {
        EntityKind::Hero => Z_RB_HERO,
        EntityKind::Tower => Z_RB_TOWER,
        EntityKind::Creep => Z_RB_CREEP,
        EntityKind::Projectile => Z_RB_BULLET,
        EntityKind::Other => Z_RB_BULLET,
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
        let (ca, _) = style_for_entity(&a);
        let (cb, _) = style_for_entity(&b);
        assert_ne!((ca.r, ca.g, ca.b), (cb.r, cb.g, cb.b));
    }
}
