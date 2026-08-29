//! Static garage scenery around the construction platform.

use bevy::{
    asset::RenderAssetUsages, camera::Exposure, mesh::Indices, prelude::*,
    render::render_resource::PrimitiveTopology,
};

use crate::{builder::GROUND_HALF_SIZE, configure_repeating_texture};

pub(crate) const CELL_SIZE: f32 = 5.0;
pub(crate) const CELL_COUNT: u8 = 4;
const CELL_COUNT_FLOAT: f32 = 4.0;
pub(crate) const SIDE_LENGTH: f32 = CELL_SIZE * CELL_COUNT_FLOAT;
pub(crate) const HEIGHT: f32 = CELL_SIZE;
pub(crate) const VOID_COLOR: Color = Color::srgb(0.0196, 0.0314, 0.0431);
pub(crate) const EXPOSURE: Exposure = Exposure::BLENDER;

const HALF_SIDE: f32 = SIDE_LENGTH * 0.5;
const COLUMN_SIZE: f32 = 0.40;
const COLUMN_INSET: f32 = COLUMN_SIZE * 0.5;

#[derive(Clone)]
struct GarageMaterials {
    floor: Handle<StandardMaterial>,
    wall: Handle<StandardMaterial>,
    ceiling: Handle<StandardMaterial>,
    dark: Handle<StandardMaterial>,
    darker: Handle<StandardMaterial>,
    bright: Handle<StandardMaterial>,
    rubber: Handle<StandardMaterial>,
    column: Handle<StandardMaterial>,
    trim: Handle<StandardMaterial>,
    mint_glow: Handle<StandardMaterial>,
    light_glow: Handle<StandardMaterial>,
}

/// Adds the 4 x 4-cell, single-course garage around the existing logical ground plane.
pub(crate) fn spawn(
    commands: &mut Commands,
    asset_server: &AssetServer,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
) {
    debug_assert!((SIDE_LENGTH - GROUND_HALF_SIZE * 2.0).abs() < f32::EPSILON);
    let garage_materials = garage_materials(asset_server, materials);
    spawn_surfaces(commands, meshes, &garage_materials);

    let cube = meshes.add(Cuboid::default());
    spawn_kerbs(commands, &cube, &garage_materials);
    spawn_columns(commands, &cube, &garage_materials);
    spawn_trusses(commands, &cube, &garage_materials);
    spawn_light_bars(commands, &cube, &garage_materials);
    commands.spawn((
        Name::new("Garage ceiling wash"),
        DirectionalLight {
            color: Color::srgb_u8(207, 226, 239),
            illuminance: 350.0,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::from_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2)),
    ));
}

pub(crate) fn fog() -> DistanceFog {
    DistanceFog {
        color: VOID_COLOR,
        falloff: FogFalloff::Exponential { density: 0.034 },
        ..default()
    }
}

fn garage_materials(
    asset_server: &AssetServer,
    materials: &mut Assets<StandardMaterial>,
) -> GarageMaterials {
    GarageMaterials {
        floor: materials.add(surface_material(asset_server, "floor", false)),
        wall: materials.add(surface_material(asset_server, "wall", true)),
        ceiling: materials.add(surface_material(asset_server, "ceiling", false)),
        dark: materials.add(plain_material(Color::srgb_u8(28, 36, 43), 0.42, 0.90)),
        darker: materials.add(plain_material(Color::srgb_u8(18, 24, 32), 0.52, 0.85)),
        bright: materials.add(plain_material(Color::srgb_u8(61, 74, 85), 0.30, 0.95)),
        rubber: materials.add(plain_material(Color::srgb_u8(12, 16, 20), 0.88, 0.10)),
        column: materials.add(plain_material(Color::srgb_u8(51, 63, 74), 0.40, 0.85)),
        trim: materials.add(plain_material(Color::srgb_u8(76, 90, 102), 0.30, 0.92)),
        mint_glow: materials.add(StandardMaterial {
            base_color: Color::srgb_u8(10, 20, 22),
            emissive: LinearRgba::rgb(12.0, 150.0, 104.0),
            perceptual_roughness: 0.5,
            ..default()
        }),
        light_glow: materials.add(StandardMaterial {
            base_color: Color::srgb_u8(10, 14, 18),
            emissive: LinearRgba::rgb(190.0, 240.0, 300.0),
            perceptual_roughness: 0.5,
            ..default()
        }),
    }
}

fn surface_material(asset_server: &AssetServer, surface: &str, emissive: bool) -> StandardMaterial {
    let stem = format!("garage/{surface}/{surface}");
    let texture = |suffix: &str, is_srgb: bool| {
        asset_server
            .load_builder()
            .with_settings(move |settings| configure_repeating_texture(settings, is_srgb))
            .load(format!("{stem}_{suffix}.png"))
    };
    let orm = texture("orm", false);
    StandardMaterial {
        base_color_texture: Some(texture("base_color", true)),
        metallic: if emissive { 0.72 } else { 1.0 },
        perceptual_roughness: 1.0,
        metallic_roughness_texture: Some(orm.clone()),
        occlusion_texture: Some(orm),
        normal_map_texture: Some(texture("normal", false)),
        emissive: if emissive {
            LinearRgba::rgb(1.5, 1.5, 1.5)
        } else {
            LinearRgba::BLACK
        },
        emissive_texture: emissive.then(|| texture("emissive", true)),
        cull_mode: None,
        ..default()
    }
}

fn plain_material(color: Color, roughness: f32, metallic: f32) -> StandardMaterial {
    StandardMaterial {
        base_color: color,
        perceptual_roughness: roughness,
        metallic,
        ..default()
    }
}

fn spawn_surfaces(commands: &mut Commands, meshes: &mut Assets<Mesh>, materials: &GarageMaterials) {
    for (name, mesh, material) in [
        ("Garage floor", floor_mesh(), materials.floor.clone()),
        ("Garage ceiling", ceiling_mesh(), materials.ceiling.clone()),
        (
            "Garage south wall",
            south_wall_mesh(),
            materials.wall.clone(),
        ),
        (
            "Garage north wall",
            north_wall_mesh(),
            materials.wall.clone(),
        ),
        ("Garage west wall", west_wall_mesh(), materials.wall.clone()),
        ("Garage east wall", east_wall_mesh(), materials.wall.clone()),
    ] {
        commands.spawn((
            Name::new(name),
            Mesh3d(meshes.add(mesh)),
            MeshMaterial3d(material),
        ));
    }
}

fn floor_mesh() -> Mesh {
    quad_mesh(
        [
            [-HALF_SIDE, 0.0, -HALF_SIDE],
            [HALF_SIDE, 0.0, -HALF_SIDE],
            [HALF_SIDE, 0.0, HALF_SIDE],
            [-HALF_SIDE, 0.0, HALF_SIDE],
        ],
        Vec3::Y,
        [1.0, 0.0, 0.0, -1.0],
        tiled_uvs(),
    )
}

fn ceiling_mesh() -> Mesh {
    quad_mesh(
        [
            [-HALF_SIDE, HEIGHT, HALF_SIDE],
            [HALF_SIDE, HEIGHT, HALF_SIDE],
            [HALF_SIDE, HEIGHT, -HALF_SIDE],
            [-HALF_SIDE, HEIGHT, -HALF_SIDE],
        ],
        Vec3::NEG_Y,
        [1.0, 0.0, 0.0, 1.0],
        tiled_uvs(),
    )
}

fn south_wall_mesh() -> Mesh {
    wall_mesh(
        [
            [-HALF_SIDE, 0.0, -HALF_SIDE],
            [HALF_SIDE, 0.0, -HALF_SIDE],
            [HALF_SIDE, HEIGHT, -HALF_SIDE],
            [-HALF_SIDE, HEIGHT, -HALF_SIDE],
        ],
        Vec3::Z,
        [1.0, 0.0, 0.0, -1.0],
    )
}

fn north_wall_mesh() -> Mesh {
    wall_mesh(
        [
            [HALF_SIDE, 0.0, HALF_SIDE],
            [-HALF_SIDE, 0.0, HALF_SIDE],
            [-HALF_SIDE, HEIGHT, HALF_SIDE],
            [HALF_SIDE, HEIGHT, HALF_SIDE],
        ],
        Vec3::NEG_Z,
        [-1.0, 0.0, 0.0, -1.0],
    )
}

fn west_wall_mesh() -> Mesh {
    wall_mesh(
        [
            [-HALF_SIDE, 0.0, HALF_SIDE],
            [-HALF_SIDE, 0.0, -HALF_SIDE],
            [-HALF_SIDE, HEIGHT, -HALF_SIDE],
            [-HALF_SIDE, HEIGHT, HALF_SIDE],
        ],
        Vec3::X,
        [0.0, 0.0, -1.0, -1.0],
    )
}

fn east_wall_mesh() -> Mesh {
    wall_mesh(
        [
            [HALF_SIDE, 0.0, -HALF_SIDE],
            [HALF_SIDE, 0.0, HALF_SIDE],
            [HALF_SIDE, HEIGHT, HALF_SIDE],
            [HALF_SIDE, HEIGHT, -HALF_SIDE],
        ],
        Vec3::NEG_X,
        [0.0, 0.0, 1.0, -1.0],
    )
}

fn wall_mesh(positions: [[f32; 3]; 4], normal: Vec3, tangent: [f32; 4]) -> Mesh {
    let repeats = CELL_COUNT_FLOAT;
    quad_mesh(
        positions,
        normal,
        tangent,
        [[0.0, 1.0], [repeats, 1.0], [repeats, 0.0], [0.0, 0.0]],
    )
}

const fn tiled_uvs() -> [[f32; 2]; 4] {
    let repeats = CELL_COUNT_FLOAT;
    [
        [0.0, 0.0],
        [repeats, 0.0],
        [repeats, repeats],
        [0.0, repeats],
    ]
}

fn quad_mesh(
    positions: [[f32; 3]; 4],
    normal: Vec3,
    tangent: [f32; 4],
    uvs: [[f32; 2]; 4],
) -> Mesh {
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions.to_vec())
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, vec![normal.to_array(); 4])
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs.to_vec())
    .with_inserted_attribute(Mesh::ATTRIBUTE_TANGENT, vec![tangent; 4])
    .with_inserted_indices(Indices::U32(vec![0, 1, 2, 0, 2, 3]))
}

fn spawn_kerbs(commands: &mut Commands, cube: &Handle<Mesh>, materials: &GarageMaterials) {
    let kerb_height = 0.20;
    let kerb_depth = 0.12;
    for (name, size, position) in [
        (
            "Garage south kerb",
            Vec3::new(SIDE_LENGTH, kerb_height, kerb_depth),
            Vec3::new(0.0, kerb_height * 0.5, -HALF_SIDE + kerb_depth * 0.5),
        ),
        (
            "Garage north kerb",
            Vec3::new(SIDE_LENGTH, kerb_height, kerb_depth),
            Vec3::new(0.0, kerb_height * 0.5, HALF_SIDE - kerb_depth * 0.5),
        ),
        (
            "Garage west kerb",
            Vec3::new(kerb_depth, kerb_height, SIDE_LENGTH),
            Vec3::new(-HALF_SIDE + kerb_depth * 0.5, kerb_height * 0.5, 0.0),
        ),
        (
            "Garage east kerb",
            Vec3::new(kerb_depth, kerb_height, SIDE_LENGTH),
            Vec3::new(HALF_SIDE - kerb_depth * 0.5, kerb_height * 0.5, 0.0),
        ),
    ] {
        spawn_box(commands, cube, &materials.dark, name, size, position);
        let cap_size = if size.x > size.z {
            Vec3::new(size.x, 0.022, kerb_depth + 0.018)
        } else {
            Vec3::new(kerb_depth + 0.018, 0.022, size.z)
        };
        spawn_box(
            commands,
            cube,
            &materials.rubber,
            format!("{name} cap"),
            cap_size,
            Vec3::new(position.x, kerb_height, position.z),
        );
    }
}

fn spawn_columns(commands: &mut Commands, cube: &Handle<Mesh>, materials: &GarageMaterials) {
    for (index, position) in column_positions().into_iter().enumerate() {
        let at = Vec3::new(position.x, 0.0, position.y);
        let name = |part: &str| format!("Garage column {index} {part}");
        spawn_box(
            commands,
            cube,
            &materials.column,
            name("shaft"),
            Vec3::new(COLUMN_SIZE, HEIGHT, COLUMN_SIZE),
            at + Vec3::Y * (HEIGHT * 0.5),
        );
        spawn_box(
            commands,
            cube,
            &materials.trim,
            name("foot"),
            Vec3::new(0.70, 0.10, 0.70),
            at + Vec3::Y * 0.05,
        );
        spawn_box(
            commands,
            cube,
            &materials.column,
            name("base"),
            Vec3::new(0.56, 0.17, 0.56),
            at + Vec3::Y * 0.185,
        );
        spawn_box(
            commands,
            cube,
            &materials.trim,
            name("base lip"),
            Vec3::new(0.60, 0.025, 0.60),
            at + Vec3::Y * 0.10,
        );
        spawn_box(
            commands,
            cube,
            &materials.column,
            name("capital"),
            Vec3::new(0.62, 0.50, 0.62),
            at + Vec3::Y * (HEIGHT - 0.25),
        );
        spawn_box(
            commands,
            cube,
            &materials.trim,
            name("capital lip"),
            Vec3::new(0.68, 0.045, 0.68),
            at + Vec3::Y * (HEIGHT - 0.478),
        );
        spawn_box(
            commands,
            cube,
            &materials.mint_glow,
            name("LED"),
            Vec3::new(COLUMN_SIZE - 0.22, 0.018, COLUMN_SIZE - 0.22),
            at + Vec3::Y * 1.55,
        );
    }
}

fn column_positions() -> Vec<Vec2> {
    let mut positions = Vec::with_capacity(usize::from(CELL_COUNT) * 4);
    for index in 0..=CELL_COUNT {
        let line = -HALF_SIDE + f32::from(index) * CELL_SIZE;
        let x = line.clamp(-HALF_SIDE + COLUMN_INSET, HALF_SIDE - COLUMN_INSET);
        positions.push(Vec2::new(x, -HALF_SIDE + COLUMN_INSET));
        positions.push(Vec2::new(x, HALF_SIDE - COLUMN_INSET));
    }
    for index in 1..CELL_COUNT {
        let z = -HALF_SIDE + f32::from(index) * CELL_SIZE;
        positions.push(Vec2::new(-HALF_SIDE + COLUMN_INSET, z));
        positions.push(Vec2::new(HALF_SIDE - COLUMN_INSET, z));
    }
    positions
}

fn spawn_trusses(commands: &mut Commands, cube: &Handle<Mesh>, materials: &GarageMaterials) {
    for index in 0..=CELL_COUNT {
        let line = -HALF_SIDE + f32::from(index) * CELL_SIZE;
        let x = line.clamp(-HALF_SIDE + 0.20, HALF_SIDE - 0.20);
        for (part, size, y, material) in [
            (
                "top flange",
                Vec3::new(0.40, 0.045, SIDE_LENGTH),
                HEIGHT - 0.07,
                &materials.bright,
            ),
            (
                "web",
                Vec3::new(0.09, 0.34, SIDE_LENGTH),
                HEIGHT - 0.26,
                &materials.darker,
            ),
            (
                "bottom flange",
                Vec3::new(0.46, 0.055, SIDE_LENGTH),
                HEIGHT - 0.46,
                &materials.dark,
            ),
        ] {
            spawn_box(
                commands,
                cube,
                material,
                format!("Garage truss {index} {part}"),
                size,
                Vec3::new(x, y, 0.0),
            );
        }
    }
}

fn spawn_light_bars(commands: &mut Commands, cube: &Handle<Mesh>, materials: &GarageMaterials) {
    let bar_length = CELL_SIZE - 0.70;
    for x_cell in 0..CELL_COUNT {
        for z_cell in 0..CELL_COUNT {
            let x = -HALF_SIDE + (f32::from(x_cell) + 0.5) * CELL_SIZE;
            let z_start = -HALF_SIDE + f32::from(z_cell) * CELL_SIZE;
            for (rail, offset) in [1.25, 3.75].into_iter().enumerate() {
                let z = z_start + offset;
                let label = format!("Garage light {x_cell}-{z_cell}-{rail}");
                spawn_box(
                    commands,
                    cube,
                    &materials.dark,
                    format!("{label} housing"),
                    Vec3::new(bar_length, 0.10, 0.20),
                    Vec3::new(x, HEIGHT - 0.56, z),
                );
                spawn_box(
                    commands,
                    cube,
                    &materials.light_glow,
                    format!("{label} strip"),
                    Vec3::new(bar_length - 0.14, 0.024, 0.125),
                    Vec3::new(x, HEIGHT - 0.612, z),
                );
            }
        }
    }
}

fn spawn_box(
    commands: &mut Commands,
    cube: &Handle<Mesh>,
    material: &Handle<StandardMaterial>,
    name: impl Into<String>,
    size: Vec3,
    position: Vec3,
) {
    commands.spawn((
        Name::new(name.into()),
        Mesh3d(cube.clone()),
        MeshMaterial3d(material.clone()),
        Transform::from_translation(position).with_scale(size),
    ));
}

#[cfg(test)]
mod tests {
    use bevy::mesh::VertexAttributeValues;

    use super::*;

    #[test]
    fn four_cells_fill_the_twenty_metre_construction_platform() {
        assert_eq!(CELL_COUNT, 4);
        assert!((SIDE_LENGTH - 20.0).abs() < f32::EPSILON);
        assert!((SIDE_LENGTH - GROUND_HALF_SIZE * 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn columns_cover_every_perimeter_cell_line_without_duplicate_corners() {
        let positions = column_positions();
        assert_eq!(positions.len(), 16);
        assert_eq!(
            positions
                .iter()
                .filter(|position| (position.x.abs() - 9.8).abs() < f32::EPSILON)
                .count(),
            10
        );
        assert_eq!(
            positions
                .iter()
                .filter(|position| (position.y.abs() - 9.8).abs() < f32::EPSILON)
                .count(),
            10
        );
    }

    #[test]
    fn floor_uvs_repeat_once_per_five_metre_cell() {
        let mesh = floor_mesh();
        let Some(VertexAttributeValues::Float32x2(uvs)) = mesh.attribute(Mesh::ATTRIBUTE_UV_0)
        else {
            panic!("garage floor has float2 texture coordinates")
        };
        assert_eq!(uvs, &tiled_uvs());
    }

    #[test]
    fn wall_art_runs_from_course_top_to_bottom() {
        let mesh = south_wall_mesh();
        let Some(VertexAttributeValues::Float32x2(uvs)) = mesh.attribute(Mesh::ATTRIBUTE_UV_0)
        else {
            panic!("garage wall has float2 texture coordinates")
        };
        assert!((uvs[0][1] - 1.0).abs() < f32::EPSILON);
        assert!(uvs[3][1].abs() < f32::EPSILON);
    }
}
