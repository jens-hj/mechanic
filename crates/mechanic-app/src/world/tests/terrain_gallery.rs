//! The terrain materials on a real GPU: a gallery of each look to inspect, and
//! a check that procedural ground never repeats.

use bevy::{
    camera::ScalingMode, mesh::Indices, prelude::*, render::render_resource::PrimitiveTopology,
    shader::Shader,
};
use mechanic_world::{SurfaceId, SurfacePalette, TerrainMaterial};

use super::terrain_shader::{Pixels, frame, measure, save_png, scene};
use crate::world::terrain_render::{
    ATTRIBUTE_TERRAIN_SLOTS, ATTRIBUTE_TERRAIN_WEIGHTS_HIGH, ATTRIBUTE_TERRAIN_WEIGHTS_LOW,
};

const SHADER_PATH: &str = "shaders/terrain_material.wgsl";
/// The live shader, or the one `MECHANIC_TERRAIN_CANDIDATE_SHADER` names,
/// read when the test runs so shader edits need no rebuild.
fn candidate() -> String {
    let live = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/assets/shaders/terrain_material.wgsl"
    );
    let path =
        std::env::var_os(crate::env::TERRAIN_CANDIDATE_SHADER).unwrap_or_else(|| live.into());
    std::fs::read_to_string(path).expect("terrain shader source")
}
const GALLERY: UVec2 = UVec2::new(1024, 640);
/// Metres the repetition check sees across its square image.
const SQUARE_METRES: f32 = 12.0;
const SQUARE_PIXELS: u32 = 1024;

/// A heightfield `extent` metres square around the origin, `side` vertices a
/// side, painted one-hot with `slot(x, z)` over `surfaces`.
#[expect(
    clippy::cast_precision_loss,
    reason = "small, bounded procedural fixture grid"
)]
fn ground(
    extent: f32,
    side: u32,
    height: impl Fn(f32, f32) -> f32,
    slot: impl Fn(f32, f32) -> usize,
    surfaces: &[SurfaceId],
) -> Mesh {
    assert!(surfaces.len() <= 8);
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uv = Vec::new();
    let mut uv1 = Vec::new();
    let mut low = Vec::new();
    let mut high = Vec::new();
    let mut indices = Vec::new();
    let step = 0.01;
    for z in 0..side {
        for x in 0..side {
            let px = (x as f32 / (side - 1) as f32 - 0.5) * extent;
            let pz = (z as f32 / (side - 1) as f32 - 0.5) * extent;
            let py = height(px, pz);
            let dx = (height(px + step, pz) - height(px - step, pz)) / (2.0 * step);
            let dz = (height(px, pz + step) - height(px, pz - step)) / (2.0 * step);
            positions.push([px, py, pz]);
            normals.push(Vec3::new(-dx, 1.0, -dz).normalize().to_array());
            uv.push([px / 1.5, pz / 1.5]);
            uv1.push([py / 1.5, 0.0]);
            let slot = slot(px, pz).min(surfaces.len() - 1);
            let mut weights = [[0_u8; 4]; 2];
            weights[slot / 4][slot % 4] = u8::MAX;
            low.push(weights[0]);
            high.push(weights[1]);
            if x + 1 < side && z + 1 < side {
                let a = z * side + x;
                indices.extend_from_slice(&[a, a + side, a + 1, a + 1, a + side, a + side + 1]);
            }
        }
    }
    let mut ids = [u32::from(u16::MAX); 8];
    for (id, surface) in ids.iter_mut().zip(surfaces) {
        *id = u32::from(surface.0);
    }
    let table = [
        ids[0] | ids[1] << 16,
        ids[2] | ids[3] << 16,
        ids[4] | ids[5] << 16,
        ids[6] | ids[7] << 16,
    ];
    let vertex_count = positions.len();
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uv);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_1, uv1);
    mesh.insert_attribute(
        ATTRIBUTE_TERRAIN_WEIGHTS_LOW,
        bevy::mesh::VertexAttributeValues::Unorm8x4(low),
    );
    mesh.insert_attribute(
        ATTRIBUTE_TERRAIN_WEIGHTS_HIGH,
        bevy::mesh::VertexAttributeValues::Unorm8x4(high),
    );
    mesh.insert_attribute(
        ATTRIBUTE_TERRAIN_SLOTS,
        bevy::mesh::VertexAttributeValues::Uint32x4(vec![table; vertex_count]),
    );
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

/// The shader fixture's scene with procedural ground switched on and the
/// cache centred on its camera.
fn procedural_scene(width: u32, height: u32) -> (App, Handle<Shader>, Entity, Entity) {
    let (mut app, shader, terrain, readback) = scene(width, height);
    for (_, material) in app
        .world_mut()
        .resource_mut::<Assets<crate::world::TerrainRenderMaterial>>()
        .iter_mut()
    {
        material.procedural_ground = true;
    }
    app.insert_resource(crate::world::terrain_cache::TerrainCacheFocus(Some(
        bevy::math::DVec3::new(0.0, 4.0, 8.0),
    )));
    (app, shader, terrain, readback)
}

fn flat(_: f32, _: f32) -> f32 {
    0.0
}

fn first(_: f32, _: f32) -> usize {
    0
}

/// A gentle 45° bank rising to a near-vertical cliff along +x.
fn cliff(x: f32, _: f32) -> f32 {
    let bank = (x + 6.0).clamp(0.0, 4.0);
    let wall = ((x - 0.5) * 6.0).clamp(0.0, 5.0);
    bank + wall
}

/// The square, straight-down view of the repetition check and the swatches.
fn top_down(width: f32, height: f32) -> (Transform, Projection) {
    (
        Transform::from_xyz(0.0, 30.0, 0.0).looking_at(Vec3::ZERO, Vec3::NEG_Z),
        Projection::from(OrthographicProjection {
            scaling_mode: ScalingMode::Fixed { width, height },
            ..OrthographicProjection::default_3d()
        }),
    )
}

fn perspective(eye: Vec3, target: Vec3) -> (Transform, Projection) {
    (
        Transform::from_translation(eye).looking_at(target, Vec3::Y),
        Projection::Perspective(PerspectiveProjection::default()),
    )
}

/// Points the scene's camera and swaps the terrain mesh.
fn stage(app: &mut App, terrain: Entity, mesh: Mesh, view: (Transform, Projection)) {
    let mesh = app.world_mut().resource_mut::<Assets<Mesh>>().add(mesh);
    app.world_mut().entity_mut(terrain).insert(Mesh3d(mesh));
    let camera = app
        .world_mut()
        .query_filtered::<Entity, With<Camera3d>>()
        .single(app.world())
        .unwrap();
    app.insert_resource(crate::world::terrain_cache::TerrainCacheFocus(Some(
        view.0.translation.as_dvec3(),
    )));
    app.world_mut().entity_mut(camera).insert(view);
}

/// Draws the staged scene with `source` and returns the read-back pixels.
fn render(app: &mut App, shader: &Handle<Shader>, source: &str) -> Vec<u8> {
    app.world_mut()
        .resource_mut::<Assets<Shader>>()
        .insert(
            shader.id(),
            Shader::from_wgsl(source.to_owned(), SHADER_PATH),
        )
        .unwrap();
    for _ in 0..24 {
        frame(app);
    }
    app.world().resource::<Pixels>().0.clone()
}

fn palette() -> SurfacePalette {
    // The same world the shader fixture loads its palette from.
    mechanic_world::TerrainField::new(mechanic_world::WorldSeed(1))
        .palette()
        .clone()
}

fn plain(material: TerrainMaterial) -> SurfaceId {
    SurfaceId::plain(material)
}

/// Luminance of read-back sRGB pixels, row-major.
fn luminance(pixels: &[u8]) -> Vec<f32> {
    pixels
        .chunks_exact(4)
        .map(|p| 0.2126 * f32::from(p[0]) + 0.7152 * f32::from(p[1]) + 0.0722 * f32::from(p[2]))
        .collect()
}

/// Pearson correlation between the image and itself moved `shift` pixels
/// along x (`vertical` false) or y, over every fourth row of the overlap.
#[expect(
    clippy::cast_precision_loss,
    reason = "pixel counts of a bounded test image"
)]
fn shifted_correlation(image: &[f32], edge: usize, shift: usize, vertical: bool) -> f32 {
    let mut pairs = Vec::new();
    for row in (0..edge).step_by(4) {
        for column in 0..edge {
            let moved = if vertical {
                (row + shift < edge).then(|| (row + shift) * edge + column)
            } else {
                (column + shift < edge).then(|| row * edge + column + shift)
            };
            if let Some(moved) = moved {
                pairs.push((
                    f64::from(image[row * edge + column]),
                    f64::from(image[moved]),
                ));
            }
        }
    }
    let mean =
        |pick: fn(&(f64, f64)) -> f64| pairs.iter().map(pick).sum::<f64>() / pairs.len() as f64;
    let (origin, moved) = (mean(|pair| pair.0), mean(|pair| pair.1));
    let covariance = mean(|pair| pair.0 * pair.1) - origin * moved;
    let origin_spread = (mean(|pair| pair.0 * pair.0) - origin * origin).max(1.0e-9);
    let moved_spread = (mean(|pair| pair.1 * pair.1) - moved * moved).max(1.0e-9);
    #[expect(clippy::cast_possible_truncation, reason = "a correlation in [-1, 1]")]
    let correlation = (covariance / (origin_spread * moved_spread).sqrt()) as f32;
    correlation
}

#[test]
#[ignore = "real GPU; run in release"]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "pixel shifts of a bounded test image"
)]
fn procedural_ground_does_not_repeat() {
    let (mut app, shader, terrain, _) = procedural_scene(SQUARE_PIXELS, SQUARE_PIXELS);
    let candidate = candidate();
    let pixels_per_metre = SQUARE_PIXELS as f32 / SQUARE_METRES;
    let edge = SQUARE_PIXELS as usize;
    let directory = std::env::temp_dir().join("mechanic-terrain-gallery");
    for material in [
        TerrainMaterial::SurfaceCover,
        TerrainMaterial::Soil,
        TerrainMaterial::Rock,
    ] {
        stage(
            &mut app,
            terrain,
            ground(SQUARE_METRES + 2.0, 9, flat, first, &[plain(material)]),
            top_down(SQUARE_METRES, SQUARE_METRES),
        );
        let pixels = render(&mut app, &shader, &candidate);
        save_png(
            &pixels,
            UVec2::splat(SQUARE_PIXELS),
            &directory,
            &format!("repeat-{material:?}"),
        );
        let image = luminance(&pixels);
        // From one metre, past the smallest features, to half the image.
        let mut worst = (f32::MIN, 0.0, false);
        let first = pixels_per_metre as usize;
        for shift in (first..edge / 2).step_by(2) {
            for vertical in [false, true] {
                let correlation = shifted_correlation(&image, edge, shift, vertical);
                if correlation > worst.0 {
                    worst = (correlation, shift as f32 / pixels_per_metre, vertical);
                }
            }
        }
        eprintln!(
            "{material:?}: strongest self-similarity {:.3} at {:.2} m ({})",
            worst.0,
            worst.1,
            if worst.2 { "z" } else { "x" }
        );
        assert!(
            worst.0 < 0.5,
            "{material:?} repeats: correlation {:.3} at a {:.2} m shift",
            worst.0,
            worst.1
        );
    }
}

#[test]
#[ignore = "real GPU art gallery; run in release, set MECHANIC_TERRAIN_REFERENCE_SHADER to compare"]
#[expect(clippy::cast_precision_loss, reason = "image edges")]
fn terrain_material_gallery() {
    let palette = palette();
    let (mut app, shader, terrain, _) = procedural_scene(GALLERY.x, GALLERY.y);
    let directory = std::env::temp_dir().join("mechanic-terrain-gallery");
    let reference = std::env::var_os(crate::env::TERRAIN_REFERENCE_SHADER)
        .map(|path| std::fs::read_to_string(path).expect("reference shader source"));
    let mut shaders = vec![("new", candidate())];
    if let Some(reference) = reference {
        shaders.push(("old", reference));
    }
    let named = |name: &str| {
        palette
            .id(name)
            .unwrap_or_else(|| panic!("palette has no {name}"))
    };
    let looks = [
        "meadow_grass",
        "alpine_turf",
        "karst_moss",
        "loam",
        "mire_mud",
        "granite",
        "limestone",
        "black_rock",
    ];
    let aspect = GALLERY.x as f32 / GALLERY.y as f32;
    for look in looks {
        let surface = [named(look)];
        let shots = [
            (
                "close",
                ground(16.0, 9, flat, first, &surface),
                top_down(6.0 * aspect, 6.0),
            ),
            (
                "field",
                ground(600.0, 97, flat, first, &surface),
                perspective(Vec3::new(0.0, 1.8, 0.0), Vec3::new(0.0, 0.0, -14.0)),
            ),
            (
                "cliff",
                ground(40.0, 321, cliff, first, &surface),
                perspective(Vec3::new(-11.0, 6.0, 7.0), Vec3::new(0.0, 3.5, 0.0)),
            ),
        ];
        for (shot, mesh, view) in shots {
            stage(&mut app, terrain, mesh, view);
            for (version, source) in &shaders {
                let pixels = render(&mut app, &shader, source);
                save_png(
                    &pixels,
                    GALLERY,
                    &directory,
                    &format!("{look}-{shot}-{version}"),
                );
            }
        }
    }
    // Grass, dirt and stone meeting along ragged borders.
    let borders = |x: f32, z: f32| {
        let wobble = (z * 1.7).sin() * 0.6 + (z * 4.3).cos() * 0.25;
        if x + wobble < -1.0 {
            0
        } else if x + wobble < 1.5 {
            1
        } else {
            2
        }
    };
    let surfaces = [
        plain(TerrainMaterial::SurfaceCover),
        plain(TerrainMaterial::Soil),
        plain(TerrainMaterial::Rock),
    ];
    stage(
        &mut app,
        terrain,
        ground(16.0, 129, flat, borders, &surfaces),
        top_down(6.0 * aspect, 6.0),
    );
    for (version, source) in &shaders {
        let pixels = render(&mut app, &shader, source);
        save_png(&pixels, GALLERY, &directory, &format!("borders-{version}"));
    }
    eprintln!("Terrain gallery: {}", directory.display());
}

#[test]
#[ignore = "real GPU paired benchmark; run in release with MECHANIC_TERRAIN_REFERENCE_SHADER"]
fn procedural_terrain_gpu_cost() {
    let reference = std::fs::read_to_string(
        std::env::var_os(crate::env::TERRAIN_REFERENCE_SHADER)
            .expect("MECHANIC_TERRAIN_REFERENCE_SHADER names the shader to compare with"),
    )
    .expect("reference shader source");
    let candidate = candidate();
    // The paired benchmark's full-size view, over one material at a time.
    let (mut app, shader, terrain, readback) = procedural_scene(4096, 2524);
    for material in [
        TerrainMaterial::SurfaceCover,
        TerrainMaterial::Soil,
        TerrainMaterial::Rock,
    ] {
        let mesh = ground(400.0, 65, flat, first, &[plain(material)]);
        let mesh = app.world_mut().resource_mut::<Assets<Mesh>>().add(mesh);
        app.world_mut().entity_mut(terrain).insert(Mesh3d(mesh));
        // A/B/B/A order helps expose warm-up or clock drift.
        let (before, _) = measure(&mut app, &shader, &reference, readback);
        let (after, _) = measure(&mut app, &shader, &candidate, readback);
        let (after_repeat, _) = measure(&mut app, &shader, &candidate, readback);
        let (before_repeat, _) = measure(&mut app, &shader, &reference, readback);
        eprintln!(
            "{material:?}: opaque median reference={before:.3}/{before_repeat:.3} ms candidate={after:.3}/{after_repeat:.3} ms"
        );
    }
}

/// Median whole-frame GPU and wall time over `frames` frames, moving the
/// camera and the cache focus with it by `step` each frame.
fn frame_cost_ms(app: &mut App, start: Vec3, step: Vec3, frames: u32) -> (f64, f64) {
    let camera = app
        .world_mut()
        .query_filtered::<Entity, With<Camera3d>>()
        .single(app.world())
        .unwrap();
    let mut eye = start;
    let mut gpu = Vec::new();
    let mut wall = Vec::new();
    for _ in 0..frames {
        eye += step;
        app.insert_resource(crate::world::terrain_cache::TerrainCacheFocus(Some(
            eye.as_dvec3(),
        )));
        app.world_mut().entity_mut(camera).insert(
            Transform::from_translation(eye).looking_at(eye + Vec3::new(0.0, -1.8, -14.0), Vec3::Y),
        );
        let began = std::time::Instant::now();
        frame(app);
        wall.push(began.elapsed().as_secs_f64() * 1_000.0);
        let snapshot = app
            .world()
            .resource::<crate::render_diagnostics::RenderTimings>()
            .snapshot();
        if let Some(ms) = snapshot.render_gpu_ms {
            gpu.push(ms);
        }
    }
    gpu.sort_by(f64::total_cmp);
    wall.sort_by(f64::total_cmp);
    (gpu[gpu.len() / 2], wall[wall.len() / 2])
}

#[test]
#[ignore = "real GPU timing; run in release"]
fn procedural_cache_update_cost() {
    let (mut app, shader, terrain, _) = procedural_scene(GALLERY.x, GALLERY.y);
    let start = Vec3::new(0.0, 1.8, 0.0);
    stage(
        &mut app,
        terrain,
        ground(
            4_000.0,
            129,
            flat,
            first,
            &[plain(TerrainMaterial::SurfaceCover)],
        ),
        perspective(start, Vec3::new(0.0, 0.0, -14.0)),
    );
    render(&mut app, &shader, &candidate());
    // Standing, walking and driving at 60 frames a second, then a jump of a
    // kilometre every frame, which refills at the budget.
    for (pace, step) in [
        ("still", 0.0),
        ("walking", 0.025),
        ("driving", 0.5),
        ("refilling", 1_000.0),
    ] {
        let (gpu, wall) = frame_cost_ms(&mut app, start, Vec3::new(step, 0.0, 0.0), 60);
        eprintln!("{pace}: frame GPU median {gpu:.3} ms, wall {wall:.3} ms");
    }
}
