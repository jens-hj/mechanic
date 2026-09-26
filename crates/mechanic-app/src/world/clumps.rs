//! Rendering for world-owned material fragments, using terrain materials.

use super::terrain_render::{
    ATTRIBUTE_TERRAIN_SLOTS, ATTRIBUTE_TERRAIN_WEIGHTS_HIGH, ATTRIBUTE_TERRAIN_WEIGHTS_LOW,
};
use super::{WorldOwned, WorldRuntime};
use bevy::mesh::VertexAttributeValues;
use bevy::prelude::*;
use mechanic_world::{FloatingOrigin, SurfaceId, TerrainMaterial};
use std::collections::BTreeSet;

/// Semi-axis of the ellipsoid holding a box's volume, per unit of its half extent.
const CLOD_SCALE: f32 = 1.240_7;

#[derive(Component)]
pub(super) struct ClumpRender {
    id: u64,
}

/// One unit clod per terrain material, by material code. Every clump of a
/// material shares its mesh and differs only by transform, so the renderer
/// batches them.
#[derive(Default)]
pub(super) struct ClodMeshes(Vec<Handle<Mesh>>);

pub(super) fn sync_clump_rendering(
    mut commands: Commands,
    runtime: Res<WorldRuntime>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut clods: Local<ClodMeshes>,
    mut rendered: Query<(Entity, &ClumpRender, &mut Transform)>,
) {
    let Some(material) = &runtime.terrain_material else {
        return;
    };
    let started = std::time::Instant::now();
    if clods.0.is_empty() {
        clods.0 = vec![Handle::default(); TerrainMaterial::COUNT];
        for material in TerrainMaterial::ALL {
            clods.0[material.code() as usize] = meshes.add(clod_mesh(material));
        }
    }
    let mut present = BTreeSet::new();
    for (entity, marker, mut transform) in &mut rendered {
        let Some(body) = runtime.clumps.bodies.get(&marker.id) else {
            commands.entity(entity).despawn();
            continue;
        };
        present.insert(marker.id);
        *transform = clump_transform(body, runtime.floating_origin);
    }
    for body in runtime
        .clumps
        .bodies
        .values()
        .filter(|body| !present.contains(&body.id))
    {
        commands.spawn((
            Name::new(format!("Material clump {}", body.id)),
            ClumpRender { id: body.id },
            Mesh3d(clods.0[body.material.code() as usize].clone()),
            MeshMaterial3d(material.clone()),
            clump_transform(body, runtime.floating_origin),
            WorldOwned,
        ));
    }
    crate::performance_capture::record(
        "clump_render_sync",
        || serde_json::json!({"duration_ms": started.elapsed().as_secs_f64() * 1000.0, "clumps": runtime.clumps.bodies.len()}),
    );
}

fn clump_transform(body: &mechanic_world::MaterialClump, origin: FloatingOrigin) -> Transform {
    Transform::from_translation((body.position.0 - origin.0).as_vec3())
        .with_rotation(body.rotation.as_quat())
        .with_scale(body.half_extents.as_vec3() * CLOD_SCALE)
}

// A lumpy unit ball carrying one terrain material's blend weights.
fn clod_mesh(material: TerrainMaterial) -> Mesh {
    let mut mesh = Sphere::new(1.0)
        .mesh()
        .ico(1)
        .unwrap_or_else(|_| Mesh::from(Sphere::new(1.0)));
    if let Some(VertexAttributeValues::Float32x3(positions)) =
        mesh.attribute_mut(Mesh::ATTRIBUTE_POSITION)
    {
        for (index, position) in positions.iter_mut().enumerate() {
            // The same dents on every clod; their tumbling tells them apart.
            #[expect(clippy::cast_precision_loss, reason = "a few dozen vertices")]
            let dent = 1.0 + 0.14 * (index as f32 * 2.399).sin();
            *position = position.map(|axis| axis * dent);
        }
    }
    mesh.duplicate_vertices();
    mesh.compute_flat_normals();
    let positions = match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
        Some(VertexAttributeValues::Float32x3(positions)) => positions.clone(),
        _ => Vec::new(),
    };
    // Slot 0 holds the material's plain surface; the shader skips the rest.
    let mut slots = [u32::MAX; 4];
    slots[0] = u32::from(SurfaceId::plain(material).0) | 0xffff_0000;
    mesh.insert_attribute(
        ATTRIBUTE_TERRAIN_WEIGHTS_LOW,
        VertexAttributeValues::Unorm8x4(vec![[u8::MAX, 0, 0, 0]; positions.len()]),
    );
    mesh.insert_attribute(
        ATTRIBUTE_TERRAIN_WEIGHTS_HIGH,
        VertexAttributeValues::Unorm8x4(vec![[0; 4]; positions.len()]),
    );
    mesh.insert_attribute(
        ATTRIBUTE_TERRAIN_SLOTS,
        VertexAttributeValues::Uint32x4(vec![slots; positions.len()]),
    );
    // A clod is a small window on the ground's 1.5 m texture repeat.
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_UV_0,
        positions
            .iter()
            .map(|p| [p[0] / 15.0, p[2] / 15.0])
            .collect::<Vec<_>>(),
    );
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_UV_1,
        positions
            .iter()
            .map(|p| [p[1] / 15.0, 0.0])
            .collect::<Vec<_>>(),
    );
    mesh
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::terrain_render::terrain_vertex_attributes;
    use bevy::mesh::MeshVertexBufferLayouts;

    #[test]
    fn every_clod_carries_the_terrain_vertex_layout() {
        let mut layouts = MeshVertexBufferLayouts::default();
        for material in TerrainMaterial::ALL {
            let layout = clod_mesh(material).get_mesh_vertex_buffer_layout(&mut layouts);
            assert!(
                layout.0.get_layout(&terrain_vertex_attributes()).is_ok(),
                "{material:?} clod cannot be drawn with the terrain material"
            );
        }
    }
}
