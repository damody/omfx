//! 階段 4.2：從 sim World snapshot 到 Fyrox scene 的橋接模組。
//!
//! 階段 5.x：每個實體的 sprite 與 HP bar 現在由 Game 端驅動
//! 批次化 mesh（`body_batch` / `hp_batch`）可將 1000+ 實體
//! 合併成少量 draw call，而不是每個實體一個節點。這個模組
//! 僅保留 **path** 渲染責任——checkpoint
//! 點位與路段線在第一個非空快照後即為靜態，
//! 所以保留逐一 `RectangleBuilder` 節點即可。
//!
//! `EntityKind::Other` 的實體永不繪製。

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
/// 階段 4.1：BlockedRegion 外框的 Z 層。繪製時略高於
/// 路徑圖層，避免區域重疊時紅色外框被遮，確保仍可見
/// （沿用舊版參考：同舊 map.regions debug overlay 的 Z）。
const Z_RB_REGION: f32 = 4.5;
/// 階段 4.1：區域外框厚度（render 單位）。取路徑厚度的一半，
/// 以避免紅色邊線比奶油色路徑更醒目。
const REGION_LINE_THICKNESS: f32 = PATH_LINE_THICKNESS * 0.5;
/// 階段 4.1：區域外框顏色用紅色（對齊舊版慣例），
/// "blocked region" overlay 規範；橘色預留給（未來）圓形阻擋物。
///（當 `circle` 阻擋物存在時）。
const REGION_OUTLINE_COLOR: (u8, u8, u8, u8) = (255, 80, 80, 255);
/// 階段 4.1：可選圓形阻擋色彩為橘色。
/// （目前 omb 的 `BlockedRegion` 無半徑值），但為未來向前相容已預留。
const REGION_CIRCLE_COLOR: (u8, u8, u8, u8) = (255, 165, 0, 255);

/// 路徑鋸齒線寬（render 單位）。按 `64.0 *
/// WORLD_SCALE * 2.0 = 1.28` 計算。對齊舊版 MVP 的「粗奶油
/// 鋸齒線」參考值（image 6）；先前 `0.12` 是前一版
/// 每段標記設計的尾段。加粗後可更完整覆蓋轉角標記點，
/// 因此不再需要單獨 checkpoint dots。
/// （上句接續上一行）
const PATH_LINE_THICKNESS: f32 = 64.0 * WORLD_SCALE * 2.0;
/// 淡黃褐（奶油色）路徑顏色（RGBA），替換先前使用的
/// ` (255, 200, 60)` 黃色。
const PATH_COLOR: (u8, u8, u8, u8) = (170, 140, 90, 255);

#[derive(Default, Debug)]
pub struct RenderBridge {
    last_applied_tick: Option<u32>,
    path_nodes: Vec<Handle<Node>>,
    paths_drawn: bool,
    /// 階段 4.1：BlockedRegion 外框 scene 節點（每條邊一段）。
    /// 首次繪製後改為靜態；`regions_drawn` 控制是否重建。
    /// 行為與 `paths_drawn` 相同。
    region_nodes: Vec<Handle<Node>>,
    regions_drawn: bool,
}

impl RenderBridge {
    pub fn new() -> Self {
        Self::default()
    }

    /// 每個 tick 的 path 初始化與診斷 log。實體 sprite 與 HP bar 現在由
    /// `Game::update_sim_batches` 的批次 mesh 渲染。
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

    /// 階段 4.1：為每個 BlockedRegion 繪製紅色多邊形外框，並可同時繪製
    /// 可選的橘色實心圓（向前相容；目前來源資料無該欄位），
    /// 僅在首次需要時繪製一次，行為與 `ensure_paths_drawn` 相同。
    /// 區域在 `state::initialization` 後固定不變。每 tick 會呼叫此函式，
    /// 但僅在 `blocked_regions` 首次非空時進行實際工作。
    /// 與 lazy paths-init 一致，因為 omb 會在第一次 dispatch 前填充資料，
    /// scene loader 會先行載入此資源，但 TD_1 會是空集合，
    /// TD_1 本身沒有任何區域，不會真的觸發此段繪製。
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
                // 多邊形外框有 N 條邊，使用最後一點到第一點作為封閉邊。
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
            // 可選圓形標記目前不會啟用，因為 omb
            // `BlockedRegion` 目前沒有 radius 欄位，這段先保留供未來使用。
            // （例如 tower footprint）等未來圓形阻擋物可直接接到這裡。
            // 不需改到渲染邏輯。
            if let Some(((cx, cy), r)) = region.circle {
                let rx = -cx * WORLD_SCALE;
                let ry = cy * WORLD_SCALE;
                let rr = r * WORLD_SCALE;
                // 以正方形 sprite 近似圓形；
                // 作為 debug overlay 足夠，渲染器日後可換成正規圓形紋理；
                //
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
            // 階段 3.1：移除每個 checkpoint 的 marker 點位，改以較粗線條代替。
            // `PATH_LINE_THICKNESS` 線條可完整覆蓋轉角。
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

/// 每個實體在 batched-mesh 的 slot 所有權。lib.rs 會持有
/// `HashMap<u32, SimEntitySlots>`，並每 tick 重複使用同一索引；`body_slot` 一律
/// 無條件分配；hp_* slot 僅在 `max_hp > 0` 時建立，`turret_slot`
/// 僅適用有朝向意義的類型（Hero / Tower / Creep），子彈不需要。
/// 專門針對 projectile。
#[derive(Debug, Clone, Copy)]
pub struct SimEntitySlots {
    pub body_slot: u32,
    pub hp_bg_slot: Option<u32>,
    pub hp_fg_slot: Option<u32>,
    pub turret_slot: Option<u32>,
}

/// 回傳給 lib.rs batched-mesh 寫入器的樣式。顏色採
    /// 依實體種類基礎色，並以 unit_id hash 做微幅偏移，讓 dart / bomb / ice
/// 等塔類在未導入真材質前仍能視覺區分。
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

/// 世界座標轉渲染座標。沿用 `entity_create` 舊流程的 `-x` 翻轉 + WORLD_SCALE
/// + body_batch 路徑；與 camera 的 look_at_rh 右手邊向向量一致。
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
