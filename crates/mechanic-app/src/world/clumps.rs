//! Rendering for world-owned material fragments, using terrain materials.

use super::{WorldOwned, WorldRuntime};
use bevy::prelude::*;
use mechanic_world::{FloatingOrigin, TerrainMaterial};
use std::collections::BTreeSet;

#[derive(Component)]
pub(super) struct ClumpRender {
    id: u64,
    quanta: u32,
}

pub(super) fn sync_clump_rendering(
    mut commands: Commands,
    runtime: Res<WorldRuntime>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut rendered: Query<(Entity, &mut ClumpRender, &mut Transform, &Mesh3d)>,
) {
    let Some(material) = &runtime.terrain_material else {
        return;
    };
    let mut present = BTreeSet::new();
    for (entity, mut marker, mut transform, mesh) in &mut rendered {
        let Some(body) = runtime.clumps.bodies.get(&marker.id) else {
            meshes.remove(mesh.0.id());
            commands.entity(entity).despawn();
            continue;
        };
        present.insert(marker.id);
        *transform = clump_transform(body, runtime.floating_origin);
        if marker.quanta != body.quanta {
            if let Some(mut mesh) = meshes.get_mut(&mesh.0) {
                *mesh = clump_mesh(body);
            }
            marker.quanta = body.quanta;
        }
    }
    for body in runtime
        .clumps
        .bodies
        .values()
        .filter(|body| !present.contains(&body.id))
    {
        commands.spawn((
            Name::new(format!("Material clump {}", body.id)),
            ClumpRender {
                id: body.id,
                quanta: body.quanta,
            },
            Mesh3d(meshes.add(clump_mesh(body))),
            MeshMaterial3d(material.clone()),
            clump_transform(body, runtime.floating_origin),
            WorldOwned,
        ));
    }
}

fn clump_transform(body: &mechanic_world::MaterialClump, origin: FloatingOrigin) -> Transform {
    Transform::from_translation((body.position.0 - origin.0).as_vec3())
        .with_rotation(body.rotation.as_quat())
}

fn clump_mesh(body: &mechanic_world::MaterialClump) -> Mesh {
    let mut mesh = Mesh::from(Cuboid::from_size((body.half_extents * 2.0).as_vec3()));
    let Some(bevy::mesh::VertexAttributeValues::Float32x3(positions)) =
        mesh.attribute(Mesh::ATTRIBUTE_POSITION)
    else {
        return mesh;
    };
    let positions = positions.clone();
    let mut weights = [0.0; TerrainMaterial::COUNT];
    weights[body.material.code() as usize] = 1.0;
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_COLOR,
        vec![[weights[0], weights[1], weights[2], weights[3]]; positions.len()],
    );
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_UV_0,
        positions
            .iter()
            .map(|p| [p[0] / 1.5, p[2] / 1.5])
            .collect::<Vec<_>>(),
    );
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_UV_1,
        positions
            .iter()
            .map(|p| [p[1] / 1.5, weights[4]])
            .collect::<Vec<_>>(),
    );
    mesh
}
