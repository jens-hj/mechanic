//! The six atlas aperture centers are isolated from the emissive fracture field.
use bevy::{
    pbr::{ExtendedMaterial, MaterialExtension},
    prelude::*,
    render::render_resource::AsBindGroup,
    shader::ShaderRef,
};
pub(super) struct AperturePlugin;
impl Plugin for AperturePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<ApertureMaterial>::default())
            .add_systems(PostUpdate, sync.after(super::update));
    }
}
type ApertureMaterial = ExtendedMaterial<StandardMaterial, ApertureExtension>;
#[derive(Asset, AsBindGroup, Reflect, Debug, Clone)]
struct ApertureExtension {
    #[uniform(100)]
    multiplier: Vec4,
}
impl MaterialExtension for ApertureExtension {
    fn fragment_shader() -> ShaderRef {
        "shaders/freeze_aperture.wgsl".into()
    }
}
fn sync(
    mut commands: Commands,
    fx: Res<super::ToolFx>,
    visuals: Res<crate::editor::preview::EditorVisuals>,
    materials: Res<Assets<StandardMaterial>>,
    mut extended: ResMut<Assets<ApertureMaterial>>,
    mut handle: Local<Option<Handle<ApertureMaterial>>>,
    query: Query<(Entity, &MeshMaterial3d<StandardMaterial>)>,
) {
    let source = &visuals.authored_materials[crate::AuthoredPart::DimensionLinkEnabled.index()];
    if handle.is_none()
        && let Some(base) = materials.get(source)
    {
        *handle = Some(extended.add(ApertureMaterial {
            base: base.clone(),
            extension: ApertureExtension {
                multiplier: Vec4::ONE,
            },
        }));
    }
    let Some(handle) = handle.as_ref() else {
        return;
    };
    if let Some(mut material) = extended.get_mut(handle) {
        material.extension.multiplier.x = fx.aperture_multiplier;
    }
    for (entity, material) in &query {
        if material.0 == *source {
            commands
                .entity(entity)
                .remove::<MeshMaterial3d<StandardMaterial>>()
                .insert(MeshMaterial3d(handle.clone()));
        }
    }
}
#[cfg(test)]
pub(super) fn aperture_mask(uv: Vec2) -> bool {
    let size = if uv.y < 0.5 {
        Vec2::new(0.5, 0.25)
    } else {
        Vec2::splat(0.25)
    };
    let center = (uv / size).floor() * size + size * 0.5;
    (uv - center).abs().max_element() < 0.034 && uv.y < 0.75 && (uv.y < 0.5 || uv.x < 0.5)
}
