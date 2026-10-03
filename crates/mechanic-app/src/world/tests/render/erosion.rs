//! Real-shader pixel checks and captures of a physically eroded terrain pit.

use super::{Pixels, render_frame};
use bevy::math::DVec3;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use mechanic_world::{
    BrickCoord, ErosionConfig, SedimentDiagnostics, TerrainField, TerrainMeshRequest,
    TerrainNodeId, TerrainOctree, TerrainTransitionMask, TerrainWater, WaterWorld, WorldPosition,
    WorldSeed, mesh_chunk,
};

use crate::dev_tools::ErosionMap;
use crate::world::TerrainRenderMaterial;
use crate::world::erosion_overlay::snapshot_map;
use crate::world::terrain_render::{terrain_chunk_mesh, terrain_mesh_is_renderable};

fn frame_pixels(app: &mut App) -> Vec<u8> {
    for _ in 0..8 {
        render_frame(app);
    }
    app.world().resource::<Pixels>().0.clone()
}

/// Proves separate channels, height rejection, and restoration, then captures
/// both modes on real terrain after accelerated water erosion.
pub(super) fn verify(
    app: &mut App,
    material: Handle<TerrainRenderMaterial>,
    target: Handle<Image>,
) {
    let original = frame_pixels(app);
    let mut samples = Vec::new();
    for (removed, deposited, height) in [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 1.0, 0.0),
        (1.0, 0.0, 5.0),
    ] {
        let values: [f32; 4] = [removed, deposited, 0.0, height];
        let image = Image::new(
            Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            bytemuck::cast_slice(&values).to_vec(),
            TextureFormat::Rgba32Float,
            default(),
        );
        let image = app.world_mut().resource_mut::<Assets<Image>>().add(image);
        {
            let mut materials = app
                .world_mut()
                .resource_mut::<Assets<TerrainRenderMaterial>>();
            let mut shown = materials.get_mut(&material).unwrap();
            shown.erosion_map = image;
            shown.erosion_window = Vec4::new(-2.0, -2.0, 4.0, 0.0);
        }
        let pixels = frame_pixels(app);
        let center = (32 * 64 + 32) * 4;
        samples.push(pixels[center..center + 3].to_vec());
    }
    eprintln!("Erosion shader removal/deposition/mixed/wrong-height pixels: {samples:?}");
    assert!(samples[0][0] > samples[0][2] + 20, "removal is orange");
    assert!(samples[1][2] > samples[1][0] + 20, "deposition is blue");
    assert!(
        samples[2][0] > samples[2][1] + 20 && samples[2][2] > samples[2][1] + 20,
        "overlap is purple"
    );
    assert!(
        samples[3][0].abs_diff(samples[3][2]) < 3,
        "unrelated heights stay neutral"
    );
    app.world_mut()
        .resource_mut::<Assets<TerrainRenderMaterial>>()
        .get_mut(&material)
        .unwrap()
        .erosion_window = Vec4::ZERO;
    assert_eq!(
        frame_pixels(app),
        original,
        "disabling diagnostics restores normal shading"
    );
    capture_pit(app, material, target);
}

fn eroded_pit() -> (TerrainField, TerrainOctree, DVec3, SedimentDiagnostics) {
    let field = TerrainField::new(WorldSeed(42));
    let spawn = field.safe_spawn().0;
    let floor = field.topmost_surface(spawn.x, spawn.z).unwrap();
    let centre = DVec3::new(spawn.x, floor - 0.3, spawn.z);
    let mut terrain = TerrainOctree::default();
    let bricks = terrain
        .excavate_sphere(&field, WorldPosition(centre), 0.6)
        .unwrap()
        .changed_brick_coordinates()
        .to_vec();
    let mut water = WaterWorld::new();
    water.set_erosion(ErosionConfig { speed: 1000.0 });
    water.set_sediment_diagnostics(true);
    water.terrain_changed(
        &TerrainWater {
            field: &field,
            edits: &terrain,
        },
        bricks,
    );
    for _ in 0..800 {
        let ground = TerrainWater {
            field: &field,
            edits: &terrain,
        };
        water.deposit(&ground, centre + DVec3::new(0.3, 0.6, 0.0), 0.002);
        water.step(&ground, 0.05);
        let asks = water.sediment_requests();
        if !asks.is_empty() {
            let (outcome, applied) = terrain.exchange_sediment(&field, &asks, Vec::new());
            water.sediment_applied(&applied);
            water.ground_cells_changed(
                &TerrainWater {
                    field: &field,
                    edits: &terrain,
                },
                &outcome.sediment_cells,
            );
        }
    }
    let snapshot = water.sediment_diagnostics().unwrap();
    assert!(
        snapshot
            .columns
            .iter()
            .any(|column| column.accumulated[0] > 0.0)
    );
    assert!(
        snapshot
            .columns
            .iter()
            .any(|column| column.accumulated[1] > 0.0)
    );
    assert!(snapshot.columns.iter().any(|column| column.pending > 0.0));
    (field, terrain, centre, snapshot)
}

fn capture_pit(app: &mut App, material: Handle<TerrainRenderMaterial>, target: Handle<Image>) {
    let (field, terrain, centre, snapshot) = eroded_pit();
    let old = app
        .world_mut()
        .query_filtered::<Entity, With<Mesh3d>>()
        .iter(app.world())
        .collect::<Vec<_>>();
    for entity in old {
        app.world_mut().despawn(entity);
    }
    let brick = WorldPosition(centre).cell().unwrap().brick();
    for z in -1..=1 {
        for x in -1..=1 {
            for y in -1..=1 {
                let chunk = mesh_chunk(
                    &field,
                    &terrain.snapshot(),
                    TerrainMeshRequest {
                        node: TerrainNodeId::leaf(BrickCoord::new(
                            brick.x + x,
                            brick.y + y,
                            brick.z + z,
                        )),
                        generation: 0,
                        transition_mask: TerrainTransitionMask::NONE,
                    },
                );
                let indices = chunk
                    .index_groups
                    .final_indices(TerrainTransitionMask::NONE);
                if !terrain_mesh_is_renderable(&chunk, indices.len()) {
                    continue;
                }
                let mesh = terrain_chunk_mesh(&chunk, indices, field.palette());
                let mesh = app.world_mut().resource_mut::<Assets<Mesh>>().add(mesh);
                app.world_mut().spawn((
                    Mesh3d(mesh),
                    MeshMaterial3d(material.clone()),
                    Transform::from_translation((chunk.origin.0 - centre).as_vec3()),
                ));
            }
        }
    }
    app.world_mut()
        .resource_mut::<Assets<Image>>()
        .get_mut(&target)
        .unwrap()
        .resize(Extent3d {
            width: 512,
            height: 512,
            depth_or_array_layers: 1,
        });
    let mut cameras = app
        .world_mut()
        .query_filtered::<&mut Transform, With<Camera3d>>();
    for mut camera in cameras.iter_mut(app.world_mut()) {
        *camera = Transform::from_xyz(1.4, 2.8, 1.6).looking_at(Vec3::ZERO, Vec3::Y);
    }
    for (mode, label) in [
        (ErosionMap::Accumulated, "accumulated"),
        (ErosionMap::Recent, "recent"),
    ] {
        let (image, window) = snapshot_map(&snapshot, centre, mode);
        let handle = app.world_mut().resource_mut::<Assets<Image>>().add(image);
        {
            let mut materials = app
                .world_mut()
                .resource_mut::<Assets<TerrainRenderMaterial>>();
            let mut shown = materials.get_mut(&material).unwrap();
            shown.erosion_map = handle;
            shown.erosion_window = window;
        }
        let pixels = frame_pixels(app);
        assert_eq!(pixels.len(), 512 * 512 * 4);
        let path = std::env::temp_dir().join(format!("mechanic-erosion-{label}.png"));
        Image::new(
            Extent3d {
                width: 512,
                height: 512,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            pixels,
            TextureFormat::Rgba8UnormSrgb,
            default(),
        )
        .try_into_dynamic()
        .unwrap()
        .save(&path)
        .unwrap();
        eprintln!("Erosion capture: {}", path.display());
    }
}
