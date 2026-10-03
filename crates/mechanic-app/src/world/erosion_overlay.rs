//! Session sediment diagnostics drawn on the existing terrain meshes.

use bevy::asset::RenderAssetUsages;
use bevy::math::DVec3;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use mechanic_world::{MATERIAL_QUANTUM_M3, SedimentDiagnostics, WATER_CELL_METRES};

use super::terrain_render::{TerrainNodeRender, TerrainRenderMaterial};
use super::{AppSpace, WaterRenderMaterial, WorldRuntime};
use crate::camera::MainCamera;
use crate::dev_tools::{DevTools, ErosionMap};

const MAP_EDGE: u32 = 512;

#[derive(Clone, Copy, PartialEq)]
struct MapStamp {
    revision: u64,
    generation: u64,
    mode: ErosionMap,
    first: (i32, i32),
    base: f64,
    origin: DVec3,
}

#[derive(Resource, Default)]
pub(super) struct ErosionOverlay {
    active: bool,
    material: Option<Handle<TerrainRenderMaterial>>,
    image: Option<Handle<Image>>,
    stamp: Option<MapStamp>,
}

pub(super) fn clear(mut overlay: ResMut<ErosionOverlay>, mut dev: ResMut<DevTools>) {
    *overlay = ErosionOverlay::default();
    dev.erosion_map = ErosionMap::Off;
}

pub(super) fn empty_map(images: &mut Assets<Image>) -> Handle<Image> {
    images.add(map_image(1, &[0.0; 4]))
}

fn map_image(edge: u32, texels: &[f32]) -> Image {
    Image::new(
        Extent3d {
            width: edge,
            height: edge,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        bytemuck::cast_slice(texels).to_vec(),
        TextureFormat::Rgba32Float,
        RenderAssetUsages::default(),
    )
}

/// Millimetres of equivalent material depth mapped to a fixed logarithmic scale.
fn intensity(quanta: f64) -> f32 {
    let millimetres = quanta.max(0.0) * MATERIAL_QUANTUM_M3 / WATER_CELL_METRES.powi(2) * 1000.0;
    (millimetres.ln_1p() / 101.0_f64.ln()).min(1.0) as f32
}

fn texels(snapshot: Option<&SedimentDiagnostics>, stamp: MapStamp, edge: u32) -> Vec<f32> {
    let mut result = vec![0.0; edge as usize * edge as usize * 4];
    if let Some(snapshot) = snapshot {
        for column in &snapshot.columns {
            let x = column.column.0 - stamp.first.0;
            let z = column.column.1 - stamp.first.1;
            if x < 0
                || z < 0
                || x >= i32::try_from(edge).expect("small texture")
                || z >= i32::try_from(edge).expect("small texture")
            {
                continue;
            }
            let index = (z as usize * edge as usize + x as usize) * 4;
            let amounts = match stamp.mode {
                ErosionMap::Recent => column.recent,
                ErosionMap::Accumulated | ErosionMap::Off => column.accumulated,
            };
            result[index..index + 4].copy_from_slice(&[
                intensity(amounts[0]),
                intensity(amounts[1]),
                intensity(column.pending),
                (column.height - stamp.base) as f32,
            ]);
        }
    }
    result
}

#[expect(
    clippy::too_many_arguments,
    reason = "one render publication touches the camera, world, and asset stores"
)]
pub(super) fn draw(
    dev: Res<DevTools>,
    space: Res<State<AppSpace>>,
    runtime: Res<WorldRuntime>,
    camera: Query<&GlobalTransform, With<MainCamera>>,
    mut overlay: ResMut<ErosionOverlay>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<TerrainRenderMaterial>>,
    mut terrain: Query<&mut MeshMaterial3d<TerrainRenderMaterial>, With<TerrainNodeRender>>,
    mut water: Query<&mut Visibility, With<MeshMaterial3d<WaterRenderMaterial>>>,
) {
    let active =
        dev.enabled && *space.get() == AppSpace::World && dev.erosion_map != ErosionMap::Off;
    // Once normal rendering has been restored, disabled diagnostics do no
    // per-entity work. Newly streamed entities already use normal materials.
    if !active && !overlay.active {
        return;
    }
    overlay.active = active;
    for mut visibility in &mut water {
        let next = if active {
            Visibility::Hidden
        } else {
            Visibility::Inherited
        };
        if *visibility != next {
            *visibility = next;
        }
    }
    let Some(normal) = runtime.terrain_material.as_ref() else {
        return;
    };
    if !active {
        for mut material in &mut terrain {
            if material.0 != *normal {
                material.0 = normal.clone();
            }
        }
        return;
    }
    let Ok(camera) = camera.single() else { return };
    let Some(mut diagnostic) = materials.get(normal).cloned() else {
        return;
    };
    let origin = runtime.floating_origin.0;
    let position = origin + camera.translation().as_dvec3();
    // Shift in 3.2 m steps, keeping each texel aligned to a water column.
    let first = |value: f64| {
        ((value / WATER_CELL_METRES).floor() as i32).div_euclid(16) * 16
            - i32::try_from(MAP_EDGE / 2).expect("small texture")
    };
    let stamp = MapStamp {
        revision: runtime.water_revision,
        generation: dev.erosion_generation,
        mode: dev.erosion_map,
        first: (first(position.x), first(position.z)),
        base: (position.y / 10.0).floor() * 10.0,
        origin,
    };
    if overlay.stamp != Some(stamp) {
        let data = texels(
            runtime.sediment_diagnostics(dev.erosion_generation),
            stamp,
            MAP_EDGE,
        );
        if let Some(mut image) = overlay
            .image
            .as_ref()
            .and_then(|handle| images.get_mut(handle))
        {
            image.data = Some(bytemuck::cast_slice(&data).to_vec());
        } else {
            overlay.image = Some(images.add(map_image(MAP_EDGE, &data)));
        }
        overlay.stamp = Some(stamp);
    }
    diagnostic.erosion_map = overlay.image.as_ref().expect("map uploaded above").clone();
    diagnostic.erosion_window = Vec4::new(
        (f64::from(stamp.first.0) * WATER_CELL_METRES - origin.x) as f32,
        (f64::from(stamp.first.1) * WATER_CELL_METRES - origin.z) as f32,
        MAP_EDGE as f32 * WATER_CELL_METRES as f32,
        (stamp.base - origin.y) as f32,
    );
    // Copy the normal material's current texture and wetness bindings so live
    // texture loading, wetness recentering and worldgen edits remain visible.
    if let Some(handle) = &overlay.material {
        if materials
            .get(handle)
            .is_some_and(|material| *material != diagnostic)
        {
            *materials.get_mut(handle).expect("material exists") = diagnostic;
        }
    } else {
        overlay.material = Some(materials.add(diagnostic));
    }
    let shown = overlay
        .material
        .as_ref()
        .expect("diagnostic material exists");
    for mut material in &mut terrain {
        if material.0 != *shown {
            material.0 = shown.clone();
        }
    }
}

#[cfg(test)]
pub(super) fn snapshot_map(
    snapshot: &SedimentDiagnostics,
    origin: DVec3,
    mode: ErosionMap,
) -> (Image, Vec4) {
    let first = (
        (origin.x / WATER_CELL_METRES).floor() as i32 - 256,
        (origin.z / WATER_CELL_METRES).floor() as i32 - 256,
    );
    let stamp = MapStamp {
        revision: 0,
        generation: 0,
        mode,
        first,
        base: origin.y,
        origin,
    };
    let data = texels(Some(snapshot), stamp, MAP_EDGE);
    (
        map_image(MAP_EDGE, &data),
        Vec4::new(
            (f64::from(first.0) * WATER_CELL_METRES - origin.x) as f32,
            (f64::from(first.1) * WATER_CELL_METRES - origin.z) as f32,
            MAP_EDGE as f32 * WATER_CELL_METRES as f32,
            0.0,
        ),
    )
}

#[cfg(test)]
#[expect(clippy::float_cmp, reason = "exact copies of the same encoded texels")]
mod tests {
    use super::*;
    use mechanic_world::SedimentDiagnosticColumn;

    #[test]
    fn negative_columns_keep_independent_signals_and_relative_height() {
        let stamp = MapStamp {
            revision: 0,
            generation: 0,
            mode: ErosionMap::Accumulated,
            first: (-2, -3),
            base: 1000.0,
            origin: DVec3::new(100.0, 900.0, 200.0),
        };
        let snapshot = SedimentDiagnostics {
            seconds: 10.0,
            columns: vec![SedimentDiagnosticColumn {
                column: (-1, -3),
                height: 1002.0,
                accumulated: [100.0, 200.0],
                recent: [50.0, 100.0],
                pending: 30.0,
            }],
        };
        let data = texels(Some(&snapshot), stamp, 2);
        assert_eq!(&data[..4], &[0.0; 4]);
        assert_eq!(
            &data[4..8],
            &[intensity(100.0), intensity(200.0), intensity(30.0), 2.0]
        );
        let recent = texels(
            Some(&snapshot),
            MapStamp {
                mode: ErosionMap::Recent,
                ..stamp
            },
            2,
        );
        assert!(recent[4] < data[4]);
        assert_eq!(recent[6], data[6]);
        assert_eq!(texels(None, stamp, 2), vec![0.0; 16]);
    }
    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "one ECS lifecycle fixture verifies initial, streamed, and restored rendering"
    )]
    fn streamed_terrain_uses_diagnostics_water_hides_and_normal_materials_return() {
        let mut app = App::new();
        let mut dev = DevTools::default();
        dev.enabled = true;
        dev.erosion_map = ErosionMap::Accumulated;
        app.init_resource::<WorldRuntime>()
            .init_resource::<Assets<Image>>()
            .init_resource::<Assets<TerrainRenderMaterial>>()
            .init_resource::<ErosionOverlay>()
            .insert_resource(State::new(AppSpace::World))
            .insert_resource(dev)
            .add_systems(Update, draw);
        let normal = app
            .world_mut()
            .resource_mut::<Assets<TerrainRenderMaterial>>()
            .add(TerrainRenderMaterial {
                base_color: default(),
                normal: default(),
                orm: default(),
                tint_mask: default(),
                surfaces: default(),
                wetness: default(),
                wet_window: Vec4::ZERO,
                erosion_map: default(),
                erosion_window: Vec4::ZERO,
                tree_base_color: default(),
                tree_normal: default(),
                tree_orm: default(),
            });
        app.world_mut()
            .resource_mut::<WorldRuntime>()
            .terrain_material = Some(normal.clone());
        app.world_mut()
            .spawn((MainCamera, GlobalTransform::IDENTITY));
        let terrain = app
            .world_mut()
            .spawn((TerrainNodeRender, MeshMaterial3d(normal.clone())))
            .id();
        let clump = app.world_mut().spawn(MeshMaterial3d(normal.clone())).id();
        let water = app
            .world_mut()
            .spawn((
                MeshMaterial3d::<WaterRenderMaterial>::default(),
                Visibility::Inherited,
            ))
            .id();
        app.update();
        assert_ne!(
            app.world()
                .get::<MeshMaterial3d<TerrainRenderMaterial>>(terrain)
                .unwrap()
                .0,
            normal
        );
        assert_eq!(
            app.world()
                .get::<MeshMaterial3d<TerrainRenderMaterial>>(clump)
                .unwrap()
                .0,
            normal
        );
        assert_eq!(
            *app.world().get::<Visibility>(water).unwrap(),
            Visibility::Hidden
        );
        let streamed_water = app
            .world_mut()
            .spawn((
                MeshMaterial3d::<WaterRenderMaterial>::default(),
                Visibility::Inherited,
            ))
            .id();
        let streamed_terrain = app
            .world_mut()
            .spawn((TerrainNodeRender, MeshMaterial3d(normal.clone())))
            .id();
        app.update();
        assert_eq!(
            *app.world().get::<Visibility>(streamed_water).unwrap(),
            Visibility::Hidden
        );
        assert_ne!(
            app.world()
                .get::<MeshMaterial3d<TerrainRenderMaterial>>(streamed_terrain)
                .unwrap()
                .0,
            normal
        );
        app.world_mut().resource_mut::<DevTools>().erosion_map = ErosionMap::Off;
        app.update();
        assert_eq!(
            app.world()
                .get::<MeshMaterial3d<TerrainRenderMaterial>>(terrain)
                .unwrap()
                .0,
            normal
        );
        assert_eq!(
            app.world()
                .get::<MeshMaterial3d<TerrainRenderMaterial>>(streamed_terrain)
                .unwrap()
                .0,
            normal
        );
        assert_eq!(
            *app.world().get::<Visibility>(water).unwrap(),
            Visibility::Inherited
        );
        assert_eq!(
            *app.world().get::<Visibility>(streamed_water).unwrap(),
            Visibility::Inherited
        );
    }
}
