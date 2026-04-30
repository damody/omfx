//! Shared GPU resources for 3D sprite rendering using `Material::standard_2d`.
//!
//! Why this layout:
//! - `Material::standard_2d`'s Forward pass output is roughly
//!   `ambient_light × vertexColor × diffuseTexture` (white fallback texture).
//!   The omfx scene sets `ambient_lighting_color = Color::WHITE`, so with the
//!   default white fallback texture the final fragment color is essentially
//!   `vertexColor` (modulo an `S_SRGBToLinear` step on the texture sample).
//! - `SurfaceData::make_quad` emits `StaticVertex` (position / normal /
//!   tex_coord / tangent — NO color attribute), so `vertexColor` reads as 0
//!   and sprites render black. We instead build a custom quad SurfaceData with
//!   only the three attributes the shader actually consumes
//!   (`vertexPosition`, `vertexTexCoord`, `vertexColor`) and bake the color
//!   into all four vertices.
//!
//! Each entity color gets its own SurfaceResource. All SurfaceResources share
//! one Material::standard_2d MaterialResource. Fyrox auto-instances Mesh nodes
//! sharing the same (SurfaceResource, Material) pair, so we still get one
//! draw call per color regardless of entity count.

use fyrox::core::algebra::{Vector2, Vector3};
use fyrox::core::color::Color;
use fyrox::core::pool::Handle;
use fyrox::material::{Material, MaterialResource};
use fyrox::scene::base::BaseBuilder;
use fyrox::core::math::TriangleDefinition;
use fyrox::scene::mesh::buffer::{
    BytesStorage, TriangleBuffer, VertexAttributeDataType, VertexAttributeDescriptor,
    VertexAttributeUsage, VertexBuffer,
};
use fyrox::scene::mesh::surface::{SurfaceBuilder, SurfaceData, SurfaceResource};
use fyrox::scene::mesh::{MeshBuilder, RenderPath};
use fyrox::scene::node::Node;
use fyrox::scene::Scene;

/// Per-vertex layout consumed by `Standard2DShader`'s vertex stage:
///
/// ```glsl
/// layout(location = 0) in vec3 vertexPosition;
/// layout(location = 1) in vec2 vertexTexCoord;
/// layout(location = 2) in vec4 vertexColor;
/// ```
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct ColoredVertex {
    position: Vector3<f32>,
    tex_coord: Vector2<f32>,
    color: [u8; 4],
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

    // Unit quad centered at origin in the XY plane, [-0.5, 0.5] × [-0.5, 0.5].
    // Tex-coord origin top-left to match `SurfaceData::make_quad`.
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
    // Per-color quad SurfaceResources (vertex color baked in).
    pub surf_hero: SurfaceResource,
    pub surf_creep: SurfaceResource,
    pub surf_tower: SurfaceResource,
    pub surf_projectile: SurfaceResource,
    pub surf_default: SurfaceResource,
    pub surf_hp_bg: SurfaceResource,
    pub surf_hp_fg: SurfaceResource,
    pub surf_facing: SurfaceResource,

    // Shared `Material::standard_2d` (color is in the vertices, not a uniform).
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

    /// Build a 3D Mesh node referencing a per-color quad + the shared material.
    /// Caller sets local_transform afterwards (position, scale, rotation).
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
