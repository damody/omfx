//! Shared GPU resources for 3D sprite rendering.
//!
//! All entity sprites (body / HP bar bg / HP bar fg / facing arrow) are
//! 3D Mesh nodes pointing to a single shared 1×1 quad SurfaceResource and
//! a per-color MaterialResource. Fyrox's renderer auto-instances Mesh nodes
//! sharing the same (SurfaceResource, Material) → one draw call per pair.

use fyrox::core::algebra::Matrix4;
use fyrox::core::color::Color;
use fyrox::material::{Material, MaterialProperty, MaterialResource};
use fyrox::scene::mesh::surface::{SurfaceData, SurfaceResource};

#[derive(Debug)]
pub struct SharedSpriteResources {
    pub quad: SurfaceResource,

    pub mat_hero: MaterialResource,
    pub mat_creep: MaterialResource,
    pub mat_tower: MaterialResource,
    pub mat_projectile: MaterialResource,
    pub mat_default: MaterialResource,

    pub mat_hp_bg: MaterialResource,
    pub mat_hp_fg: MaterialResource,

    pub mat_facing: MaterialResource,
}

impl SharedSpriteResources {
    pub fn new() -> Self {
        let quad_data = SurfaceData::make_quad(&Matrix4::identity());
        let quad = SurfaceResource::new_embedded(quad_data);

        Self {
            quad,
            mat_hero: make_color_material(Color::from_rgba(50, 180, 50, 255)),
            mat_creep: make_color_material(Color::from_rgba(220, 40, 40, 255)),
            mat_tower: make_color_material(Color::from_rgba(50, 100, 220, 255)),
            mat_projectile: make_color_material(Color::from_rgba(255, 230, 50, 255)),
            mat_default: make_color_material(Color::from_rgba(200, 200, 200, 255)),
            mat_hp_bg: make_color_material(Color::from_rgba(0, 0, 0, 255)),
            mat_hp_fg: make_color_material(Color::from_rgba(0, 220, 0, 255)),
            mat_facing: make_color_material(Color::from_rgba(255, 200, 0, 255)),
        }
    }

    pub fn material_for(&self, entity_type: &str) -> &MaterialResource {
        match entity_type {
            "hero" => &self.mat_hero,
            "creep" | "enemy" => &self.mat_creep,
            "unit" | "tower" => &self.mat_tower,
            "bullet" | "projectile" => &self.mat_projectile,
            _ => &self.mat_default,
        }
    }

    /// Build a 3D Mesh node referencing the shared quad + given material.
    /// Caller sets local_transform afterwards (position, scale, rotation).
    pub fn build_mesh(
        &self,
        scene: &mut fyrox::scene::Scene,
        material: fyrox::material::MaterialResource,
    ) -> fyrox::core::pool::Handle<fyrox::scene::node::Node> {
        use fyrox::scene::base::BaseBuilder;
        use fyrox::scene::mesh::surface::SurfaceBuilder;
        use fyrox::scene::mesh::MeshBuilder;

        MeshBuilder::new(BaseBuilder::new())
            .with_surfaces(vec![SurfaceBuilder::new(self.quad.clone())
                .with_material(material)
                .build()])
            .build(&mut scene.graph)
            .to_base()
    }
}

fn make_color_material(color: Color) -> MaterialResource {
    // Use standard_2d (unlit) shader so sprites render without a lighting setup.
    // The standard 3D shader writes through GBuffer/deferred — without scene lights
    // the final composited fragment is black even though `diffuseColor` is set.
    // standard_2d's Forward pass output is `ambientLight * vertexColor * diffuseTexture`
    // (white fallback texture), so as long as the scene has ambient light the sprite
    // is visible. We still set `diffuseColor` for forward compatibility / to keep the
    // material self-describing for catalogs.
    let mut mat = Material::standard_2d();
    mat.set_property("diffuseColor", MaterialProperty::Color(color));
    MaterialResource::new_embedded(mat)
}
