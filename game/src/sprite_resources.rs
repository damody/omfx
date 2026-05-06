//! 以 `Material::standard_2d` 共享 3D sprite 渲染所需 GPU 資源。
//!
//! 這裡這麼寫的原因：
//! - `Material::standard_2d` 的 Forward pass 輸出大致為
//!   `ambient_light × vertexColor × diffuseTexture`（白色 fallback texture）。
//!   omfx 場景會設定 `ambient_lighting_color = Color::WHITE`，因此在預設白色
//!   fallback texture 下，最終 fragment color 幾乎等於 `vertexColor`
//!   （僅有材質取樣時的 `S_SRGBToLinear` 影響）。
//! - `SurfaceData::make_quad` 會產生 `StaticVertex`（position / normal /
//!   tex_coord / tangent，沒有 color 欄位），所以 `vertexColor` 讀到的是 0，
//!   sprite 會變黑。這裡改為自訂 quad 的 `SurfaceData`，只保留 shader
//!   真正使用的三個屬性（`vertexPosition`、`vertexTexCoord`、`vertexColor`），
//!   並將顏色烘焙到四個頂點上。
//!
//! 每個實體顏色都會對應獨立的 `SurfaceResource`。所有 `SurfaceResource`
//! 共享同一個 `Material::standard_2d MaterialResource`。只要 `(SurfaceResource, Material)`
//! 組合相同，Fyrox 會自動實例化 Mesh 節點，因此不論實體數量多少，每種顏色仍可維持
//! 一次 draw call。

use fyrox::core::algebra::{Vector2, Vector3};
use fyrox::core::color::Color;
use fyrox::core::pool::Handle;
use fyrox::material::{Material, MaterialResource};
use fyrox::scene::base::BaseBuilder;
use fyrox::core::math::TriangleDefinition;
use fyrox::scene::mesh::buffer::{
    BytesStorage, TriangleBuffer, VertexAttributeDataType, VertexAttributeDescriptor,
    VertexAttributeUsage, VertexBuffer, VertexTrait,
};
use fyrox::scene::mesh::surface::{SurfaceBuilder, SurfaceData, SurfaceResource};
use fyrox::scene::mesh::{MeshBuilder, RenderPath};
use fyrox::scene::node::Node;
use fyrox::scene::Scene;

/// `Standard2DShader` 頂點階段會讀取的每頂點欄位配置：
///
/// ````glsl
/// vec3 vertexPosition 中的佈局（位置 = 0）；
/// vec2 vertexTexCoord 中的佈局（位置 = 1）；
/// vec4 vertexColor 中的佈局（位置 = 2）；
/// ```
#[repr(C)]
pub struct ColoredVertex {
    pub position: Vector3<f32>,
    pub tex_coord: Vector2<f32>,
    pub color: [u8; 4],
}

impl VertexTrait for ColoredVertex {
    fn layout() -> &'static [VertexAttributeDescriptor] {
        vertex_layout()
    }
}

fn vertex_layout() -> &'static [VertexAttributeDescriptor] {
    use std::sync::OnceLock;
    static LAYOUT: OnceLock<Vec<VertexAttributeDescriptor>> = OnceLock::new();
    LAYOUT
        .get_or_init(|| {
            vec![
                VertexAttributeDescriptor {
                    usage: VertexAttributeUsage::Position,
                    data_type: VertexAttributeDataType::F32,
                    size: 3,
                    divisor: 0,
                    shader_location: 0,
                    normalized: false,
                },
                VertexAttributeDescriptor {
                    usage: VertexAttributeUsage::TexCoord0,
                    data_type: VertexAttributeDataType::F32,
                    size: 2,
                    divisor: 0,
                    shader_location: 1,
                    normalized: false,
                },
                VertexAttributeDescriptor {
                    usage: VertexAttributeUsage::Color,
                    data_type: VertexAttributeDataType::U8,
                    size: 4,
                    divisor: 0,
                    shader_location: 2,
                    normalized: true,
                },
            ]
        })
        .as_slice()
}

fn make_colored_quad_surface(color: Color) -> SurfaceResource {
    let c = [color.r, color.g, color.b, color.a];

    // 單位四邊形放在 XY 平面原點，範圍為 [-0.5, 0.5] × [-0.5, 0.5]。
    // 紋理座標原點對齊左上角，與 `SurfaceData::make_quad` 一致。
    let verts: Vec<ColoredVertex> = vec![
        ColoredVertex {
            position: Vector3::new(-0.5, -0.5, 0.0),
            tex_coord: Vector2::new(0.0, 1.0),
            color: c,
        },
        ColoredVertex {
            position: Vector3::new(0.5, -0.5, 0.0),
            tex_coord: Vector2::new(1.0, 1.0),
            color: c,
        },
        ColoredVertex {
            position: Vector3::new(0.5, 0.5, 0.0),
            tex_coord: Vector2::new(1.0, 0.0),
            color: c,
        },
        ColoredVertex {
            position: Vector3::new(-0.5, 0.5, 0.0),
            tex_coord: Vector2::new(0.0, 0.0),
            color: c,
        },
    ];
    let vertex_count = verts.len();
    let bytes = BytesStorage::new(verts);
    let vb = VertexBuffer::new_with_layout(vertex_layout(), vertex_count, bytes)
        .expect("colored quad vertex buffer layout must validate");

    let tris = vec![TriangleDefinition([0, 1, 2]), TriangleDefinition([0, 2, 3])];
    let tb = TriangleBuffer::new(tris);

    SurfaceResource::new_embedded(SurfaceData::new(vb, tb))
}

#[derive(Debug)]
pub struct SharedSpriteResources {
    // 每個顏色一組 quad 的 SurfaceResource（含預先烘焙好的頂點色）。
    pub surf_hero: SurfaceResource,
    pub surf_creep: SurfaceResource,
    pub surf_tower: SurfaceResource,
    pub surf_projectile: SurfaceResource,
    pub surf_default: SurfaceResource,
    pub surf_hp_bg: SurfaceResource,
    pub surf_hp_fg: SurfaceResource,
    pub surf_facing: SurfaceResource,

    // 共用 `Material::standard_2d`（顏色存在頂點資料而非 uniform）。
    pub material: MaterialResource,
}

impl SharedSpriteResources {
    pub fn new() -> Self {
        Self {
            surf_hero: make_colored_quad_surface(Color::from_rgba(50, 180, 50, 255)),
            surf_creep: make_colored_quad_surface(Color::from_rgba(220, 40, 40, 255)),
            surf_tower: make_colored_quad_surface(Color::from_rgba(50, 100, 220, 255)),
            surf_projectile: make_colored_quad_surface(Color::from_rgba(255, 230, 50, 255)),
            surf_default: make_colored_quad_surface(Color::from_rgba(200, 200, 200, 255)),
            surf_hp_bg: make_colored_quad_surface(Color::from_rgba(0, 0, 0, 255)),
            surf_hp_fg: make_colored_quad_surface(Color::from_rgba(0, 220, 0, 255)),
            surf_facing: make_colored_quad_surface(Color::from_rgba(255, 200, 0, 255)),
            material: MaterialResource::new_embedded(Material::standard_2d()),
        }
    }

    pub fn surface_for(&self, entity_type: &str) -> &SurfaceResource {
        match entity_type {
            "hero" => &self.surf_hero,
            "creep" | "enemy" => &self.surf_creep,
            "unit" | "tower" => &self.surf_tower,
            "bullet" | "projectile" => &self.surf_projectile,
            _ => &self.surf_default,
        }
    }

    /// 建立 3D Mesh 節點，使用每種顏色的 quad 與共用材質。
    /// 呼叫端需在後續設定 `local_transform`（位置、縮放、旋轉）。
    pub fn build_mesh(
        &self,
        scene: &mut Scene,
        surface: SurfaceResource,
    ) -> Handle<Node> {
        MeshBuilder::new(BaseBuilder::new().with_frustum_culling(false))
            .with_surfaces(vec![SurfaceBuilder::new(surface)
                .with_material(self.material.clone())
                .build()])
            .with_render_path(RenderPath::Forward)
            .build(&mut scene.graph)
            .to_base()
    }
}

/// 寫一個 quad 到 batched mesh 的參數（world-space center + size + rotation）。
#[derive(Copy, Clone, Debug)]
pub struct QuadParams {
    pub center: Vector2<f32>,
    pub size: Vector2<f32>,
    pub color: [u8; 4],
    pub rotation: f32,
    pub z: f32,
}

/// N 個實體 sprite 共用一個 Mesh，一次 `modify()` 上傳整批 vertex buffer。
/// 在 Fyrox 1.0.1，auto-batch 仍在 bundle 內逐一 instance 呼叫
/// `frame_buffer.draw()`（見 `bundle.rs:710`），不是真正 GPU instanced；
/// 要做到 1 次 draw call 對應 N 個 quad，必須把全部 quad 放在同一個
/// `SurfaceData` 的 vertex buffer。
///
/// 用法：
/// ````文本
/// 讓槽=batch.alloc(); // 實體_創建
/// batch.write_quad(槽, &params); //entity_create + 每幀插值循環
/// batch.flush(scene);                     // per-frame interp loop 結束時一次性 upload
/// 批次.空閒（槽）； // 實體刪除
/// ```
///
/// 退化 slot（`free` 後或尚未分配）使用尺寸為 0 且 alpha=0 的 quad，
/// 不會被看到，也不會增加 fragment 計算。
pub struct BatchedSpriteMesh {
    mesh_handle: Handle<Node>,
    surface: SurfaceResource,
    capacity: u32,
    cpu_mirror: Vec<ColoredVertex>, // capacity * 4
    free_list: Vec<u32>,
    next_slot: u32,
    dirty: bool,
}

impl std::fmt::Debug for BatchedSpriteMesh {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchedSpriteMesh")
            .field("capacity", &self.capacity)
            .field("next_slot", &self.next_slot)
            .field("free_list_len", &self.free_list.len())
            .field("dirty", &self.dirty)
            .finish()
    }
}

impl BatchedSpriteMesh {
    pub fn new(scene: &mut Scene, capacity: u32, material: MaterialResource) -> Self {
        let vertex_count = (capacity * 4) as usize;
        let triangle_count = (capacity * 2) as usize;

        // 先預先填滿退化 quad（全部頂點放在 origin，alpha=0）。
        let cpu_mirror: Vec<ColoredVertex> = vec![
            ColoredVertex {
                position: Vector3::new(0.0, 0.0, 0.0),
                tex_coord: Vector2::new(0.0, 0.0),
                color: [0, 0, 0, 0],
            };
            vertex_count
        ];

        // 三角形索引一次建好後不再變動：每個 slot 兩個三角形。
        // [s*4、s*4+1、s*4+2]、[s*4、s*4+2、s*4+3]
        let mut tris: Vec<TriangleDefinition> = Vec::with_capacity(triangle_count);
        for s in 0..capacity {
            let base = s * 4;
            tris.push(TriangleDefinition([base, base + 1, base + 2]));
            tris.push(TriangleDefinition([base, base + 2, base + 3]));
        }

        let bytes = BytesStorage::new(cpu_mirror.clone());
        let vb = VertexBuffer::new_with_layout(vertex_layout(), vertex_count, bytes)
            .expect("batched sprite vertex buffer layout must validate");
        let tb = TriangleBuffer::new(tris);
        let surface = SurfaceResource::new_embedded(SurfaceData::new(vb, tb));

        let mesh_handle = MeshBuilder::new(BaseBuilder::new().with_frustum_culling(false))
            .with_surfaces(vec![SurfaceBuilder::new(surface.clone())
                .with_material(material)
                .build()])
            .with_render_path(RenderPath::Forward)
            .build(&mut scene.graph)
            .to_base();

        Self {
            mesh_handle,
            surface,
            capacity,
            cpu_mirror,
            free_list: Vec::new(),
            next_slot: 0,
            dirty: false,
        }
    }

    pub fn alloc(&mut self) -> u32 {
        if let Some(slot) = self.free_list.pop() {
            return slot;
        }
        let slot = self.next_slot;
        debug_assert!(
            slot < self.capacity,
            "BatchedSpriteMesh capacity exhausted (cap={})",
            self.capacity
        );
        self.next_slot += 1;
        slot
    }

    pub fn free(&mut self, slot: u32) {
        // 寫入退化 quad（大小為 0、alpha=0）使該 slot 不可見。
        let base = (slot * 4) as usize;
        for i in 0..4 {
            self.cpu_mirror[base + i] = ColoredVertex {
                position: Vector3::new(0.0, 0.0, 0.0),
                tex_coord: Vector2::new(0.0, 0.0),
                color: [0, 0, 0, 0],
            };
        }
        self.free_list.push(slot);
        self.dirty = true;
    }

    pub fn write_quad(&mut self, slot: u32, p: &QuadParams) {
        let base = (slot * 4) as usize;
        let hw = p.size.x * 0.5;
        let hh = p.size.y * 0.5;

        // 非旋轉快速路徑（多數 entity 不會用到旋轉）。
        if p.rotation.abs() < f32::EPSILON {
            self.cpu_mirror[base + 0] = ColoredVertex {
                position: Vector3::new(p.center.x - hw, p.center.y - hh, p.z),
                tex_coord: Vector2::new(0.0, 1.0),
                color: p.color,
            };
            self.cpu_mirror[base + 1] = ColoredVertex {
                position: Vector3::new(p.center.x + hw, p.center.y - hh, p.z),
                tex_coord: Vector2::new(1.0, 1.0),
                color: p.color,
            };
            self.cpu_mirror[base + 2] = ColoredVertex {
                position: Vector3::new(p.center.x + hw, p.center.y + hh, p.z),
                tex_coord: Vector2::new(1.0, 0.0),
                color: p.color,
            };
            self.cpu_mirror[base + 3] = ColoredVertex {
                position: Vector3::new(p.center.x - hw, p.center.y + hh, p.z),
                tex_coord: Vector2::new(0.0, 0.0),
                color: p.color,
            };
        } else {
            let cos_r = p.rotation.cos();
            let sin_r = p.rotation.sin();
            // 將 local corner (lx, ly) 旋轉到世界座標（center + R * corner）。
            let rotate = |lx: f32, ly: f32| -> (f32, f32) {
                (
                    p.center.x + lx * cos_r - ly * sin_r,
                    p.center.y + lx * sin_r + ly * cos_r,
                )
            };
            let (x0, y0) = rotate(-hw, -hh);
            let (x1, y1) = rotate(hw, -hh);
            let (x2, y2) = rotate(hw, hh);
            let (x3, y3) = rotate(-hw, hh);
            self.cpu_mirror[base + 0] = ColoredVertex {
                position: Vector3::new(x0, y0, p.z),
                tex_coord: Vector2::new(0.0, 1.0),
                color: p.color,
            };
            self.cpu_mirror[base + 1] = ColoredVertex {
                position: Vector3::new(x1, y1, p.z),
                tex_coord: Vector2::new(1.0, 1.0),
                color: p.color,
            };
            self.cpu_mirror[base + 2] = ColoredVertex {
                position: Vector3::new(x2, y2, p.z),
                tex_coord: Vector2::new(1.0, 0.0),
                color: p.color,
            };
            self.cpu_mirror[base + 3] = ColoredVertex {
                position: Vector3::new(x3, y3, p.z),
                tex_coord: Vector2::new(0.0, 0.0),
                color: p.color,
            };
        }

        self.dirty = true;
    }

    /// 將 cpu_mirror 上傳到 GPU 的 vertex buffer。若未 dirty 直接跳過。
    /// 一次 `modify()` 對應一次 GPU upload，接著 N 個 quad 共用一個 draw call。
    pub fn flush(&mut self, _scene: &mut Scene) {
        if !self.dirty {
            return;
        }
        let mut data_ref = self.surface.data_ref();
        let mut vb_ref = data_ref.vertex_buffer.modify();
        let dst = vb_ref
            .cast_data_mut::<ColoredVertex>()
            .expect("ColoredVertex layout matches");
        debug_assert_eq!(dst.len(), self.cpu_mirror.len());
        dst.copy_from_slice(&self.cpu_mirror);
        drop(vb_ref); // 主動 drop 可觸發 GPU upload
        self.dirty = false;
    }

    pub fn mesh_handle(&self) -> Handle<Node> {
        self.mesh_handle
    }
}
