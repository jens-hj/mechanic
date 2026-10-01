//! Production terrain and water materials in the isolated sky fixture.

use super::*;
use crate::world::terrain_render::{
    ATTRIBUTE_TERRAIN_SLOTS, ATTRIBUTE_TERRAIN_WEIGHTS_HIGH, ATTRIBUTE_TERRAIN_WEIGHTS_LOW,
    advance_terrain_textures, terrain_render_material,
};
use crate::world::{TerrainRenderMaterial, WaterRenderMaterial};
use bevy::mesh::VertexAttributeValues;
use bevy::render::storage::ShaderBuffer;

pub(super) fn ground(app: &mut App) {
    let palette = mechanic_world::TerrainField::new(mechanic_world::WorldSeed(42))
        .palette()
        .clone();
    let server = app.world().resource::<AssetServer>().clone();
    let (material, mut build) =
        app.world_mut()
            .resource_scope(|world, mut images: Mut<Assets<Image>>| {
                terrain_render_material(
                    &server,
                    &mut images,
                    &mut world.resource_mut::<Assets<ShaderBuffer>>(),
                    &palette,
                )
            });
    let mut ready = false;
    for _ in 0..6000 {
        frame(app);
        ready = advance_terrain_textures(
            &mut build,
            &mut app.world_mut().resource_mut::<Assets<Image>>(),
        )
        .unwrap();
        if ready {
            break;
        }
    }
    assert!(ready, "terrain textures must load for the sky capture");
    let mut mesh = Plane3d::default().mesh().size(2000.0, 2000.0).build();
    let count = mesh.count_vertices();
    let Some(VertexAttributeValues::Float32x3(positions)) =
        mesh.attribute(Mesh::ATTRIBUTE_POSITION)
    else {
        panic!("plane positions");
    };
    let uv: Vec<[f32; 2]> = positions.iter().map(|p| [p[0] / 1.5, p[2] / 1.5]).collect();
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uv);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_1, vec![[0.0; 2]; count]);
    mesh.insert_attribute(
        ATTRIBUTE_TERRAIN_WEIGHTS_LOW,
        VertexAttributeValues::Unorm8x4(vec![[255, 0, 0, 0]; count]),
    );
    mesh.insert_attribute(
        ATTRIBUTE_TERRAIN_WEIGHTS_HIGH,
        VertexAttributeValues::Unorm8x4(vec![[0; 4]; count]),
    );
    let soil = u32::from(mechanic_world::SurfaceId::plain(mechanic_world::TerrainMaterial::Soil).0);
    mesh.insert_attribute(
        ATTRIBUTE_TERRAIN_SLOTS,
        VertexAttributeValues::Uint32x4(vec![
            [soil | 0xffff_0000, u32::MAX, u32::MAX, u32::MAX];
            count
        ]),
    );
    let mesh = app.world_mut().resource_mut::<Assets<Mesh>>().add(mesh);
    let material = app
        .world_mut()
        .resource_mut::<Assets<TerrainRenderMaterial>>()
        .add(material);
    app.world_mut()
        .spawn((Mesh3d(mesh), MeshMaterial3d(material)));
}

pub(super) fn water(app: &mut App) {
    let mut mesh = Plane3d::default().mesh().size(12.0, 6.0).build();
    mesh.insert_attribute(
        crate::world::water_render::ATTRIBUTE_WATER,
        vec![[1.5, 0.0, 0.0]; mesh.count_vertices()],
    );
    let mesh = app.world_mut().resource_mut::<Assets<Mesh>>().add(mesh);
    let material = app
        .world_mut()
        .resource_mut::<Assets<WaterRenderMaterial>>()
        .add(WaterRenderMaterial::default());
    app.world_mut().spawn((
        Mesh3d(mesh),
        MeshMaterial3d(material),
        Transform::from_xyz(0.0, 0.05, -7.0),
    ));
}
