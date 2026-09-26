//! Headless pictures of a generated world for authoring biomes: a biome and
//! river map, a shaded relief painted with each surface's look, and one
//! vertical cross-section through every biome showing caves, arches, and
//! overhangs. With `--views`, also a ray-marched perspective view of every
//! biome. With `--carves`, finds places where each carve layer breaks the
//! surface or runs buried, and draws a close view and a cross-section of
//! each, plus how many entrance mouths reach a tunnel. Emits one JSONL
//! summary line.
//!
//! `cargo run -p mechanic-bench --release --bin worldgen-preview -- --seed 42 --out <dir>`
//! with optional `--worldgen <dir>` to preview an authored definition,
//! `--span <metres>` for the map width, `--pixels <n>` for its resolution,
//! and `--centre <x>,<z>`.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "pixel grids of a few hundred and colour bytes"
)]

use std::error::Error;
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use bevy_math::DVec3;
use mechanic_world::{
    SurfaceLook, TerrainField, TextureSet, WorldPosition, WorldSeed, WorldgenSpec,
};

const CROSS_SECTION_WIDTH: f64 = 240.0;
const CROSS_SECTION_PIXELS: usize = 480;
const CROSS_SECTION_ROWS: usize = 320;

fn main() -> Result<(), Box<dyn Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    let value = |name: &str| {
        args.iter()
            .position(|arg| arg == name)
            .and_then(|index| args.get(index + 1))
            .cloned()
    };
    let seed: u64 = value("--seed").map_or(Ok(42), |value| value.parse())?;
    let out = PathBuf::from(value("--out").unwrap_or_else(|| "worldgen-preview".to_owned()));
    let span: f64 = value("--span").map_or(Ok(6_000.0), |value| value.parse())?;
    let pixels: usize = value("--pixels").map_or(Ok(512), |value| value.parse())?;
    let centre = value("--centre").map_or(Ok((0.0, 0.0)), |value| {
        let (x, z) = value.split_once(',').ok_or("--centre takes x,z")?;
        Ok::<_, Box<dyn Error>>((x.parse()?, z.parse()?))
    })?;
    let views = args.iter().any(|arg| arg == "--views");
    let carves = args.iter().any(|arg| arg == "--carves");
    std::fs::create_dir_all(&out)?;

    let spec = match value("--worldgen") {
        Some(directory) => Arc::new(WorldgenSpec::from_dir(Path::new(&directory))?),
        None => WorldgenSpec::embedded(),
    };
    let started = Instant::now();
    let field = TerrainField::from_spec(WorldSeed(seed), Arc::clone(&spec))?;
    let compile_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let names = spec.biome_names().map(str::to_owned).collect::<Vec<_>>();

    let started = Instant::now();
    let map = sample_map(&field, &names, centre, span, pixels);
    let map_ms = started.elapsed().as_secs_f64() * 1_000.0;
    write_png(&out.join("biomes.png"), pixels, pixels, &map.biome_rgb)?;
    write_png(&out.join("relief.png"), pixels, pixels, &map.relief_rgb)?;

    let started = Instant::now();
    let mut sections = Vec::new();
    for (index, name) in names.iter().enumerate() {
        let Some((x, z)) = map.biome_heart[index] else {
            continue;
        };
        if views {
            let target = DVec3::new(x, field.surface_height(x, z) + 20.0, z);
            let view = perspective_view(&field, target, DVec3::new(-190.0, 95.0, -190.0));
            write_png(
                &out.join(format!("view_{name}.png")),
                VIEW_WIDTH,
                VIEW_HEIGHT,
                &view,
            )?;
        }
        let rgb = cross_section(&field, x, z, CROSS_SECTION_WIDTH);
        let file = format!("section_{name}.png");
        write_png(
            &out.join(&file),
            CROSS_SECTION_PIXELS,
            CROSS_SECTION_ROWS,
            &rgb,
        )?;
        sections.push(serde_json::json!({"biome": name, "x": x, "z": z, "file": file}));
    }
    let section_ms = started.elapsed().as_secs_f64() * 1_000.0;

    let carve_report = if carves {
        Some(carve_report(&field, centre, span, &out)?)
    } else {
        None
    };

    let total = (pixels * pixels) as f64;
    let coverage = names
        .iter()
        .enumerate()
        .map(|(index, name)| {
            (
                name.clone(),
                serde_json::json!(map.biome_pixels[index] as f64 / total),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    let start = field.safe_spawn().0;
    println!(
        "{}",
        serde_json::json!({
            "scenario": "worldgen-preview",
            "seed": seed,
            "worldgen_hash": format!("{:016x}", spec.hash()),
            "compile_ms": compile_ms,
            "map_ms": map_ms,
            "map_pixels": pixels * pixels,
            "sections_ms": section_ms,
            "river_segments": field.river_segment_count(),
            "coverage": coverage,
            "spawn": [start.x, start.y, start.z],
            "spawn_biome": field.biome_at(start.x, start.z),
            "sections": sections,
            "carves": carve_report,
            "out": out.display().to_string(),
        })
    );
    Ok(())
}

/// Close views and cross-sections of places each carve layer shows, and
/// how many entrance mouths reach a tunnel.
fn carve_report(
    field: &TerrainField,
    centre: (f64, f64),
    span: f64,
    out: &Path,
) -> Result<serde_json::Value, Box<dyn Error>> {
    let started = Instant::now();
    let found = find_carves(field, centre, span);
    let mut examples = Vec::new();
    for (index, example) in found.iter().enumerate() {
        let stem = format!("carve_{index:02}_{}_{}", example.layer, example.kind);
        let point = DVec3::new(example.x, example.y, example.z);
        // Stand back from the place and look along the way its void
        // continues downward.
        let along = void_direction(field, point);
        let (target, offset) = if example.kind == "open" {
            (
                point + along * 12.0 - DVec3::Y * 2.0,
                -along * 22.0 + DVec3::Y * 8.0,
            )
        } else {
            (point + along * 12.0, -along * 5.0 + DVec3::Y * 0.5)
        };
        let view = perspective_view(field, target, offset);
        write_png(
            &out.join(format!("{stem}_view.png")),
            VIEW_WIDTH,
            VIEW_HEIGHT,
            &view,
        )?;
        let width = if example.layer == "ravines" {
            480.0
        } else {
            120.0
        };
        let section = cross_section(field, example.x, example.z, width);
        write_png(
            &out.join(format!("{stem}_section.png")),
            CROSS_SECTION_PIXELS,
            CROSS_SECTION_ROWS,
            &section,
        )?;
        examples.push(serde_json::json!({
            "layer": example.layer,
            "kind": example.kind,
            "biome": field.biome_at(example.x, example.z),
            "at": [example.x.round(), example.y.round(), example.z.round()],
            "files": stem,
        }));
    }
    let mouths = found
        .iter()
        .filter(|example| example.layer == "entrances" && example.kind == "open")
        .map(|example| DVec3::new(example.x, example.y, example.z))
        .collect::<Vec<_>>();
    let connected = mouths
        .iter()
        .filter(|mouth| reaches_layer(field, **mouth, "tunnels"))
        .count();
    Ok(serde_json::json!({
        "examples": examples,
        "entrance_mouths": mouths.len(),
        "entrances_reaching_tunnels": connected,
        "ms": started.elapsed().as_secs_f64() * 1_000.0,
    }))
}

struct Map {
    biome_rgb: Vec<u8>,
    relief_rgb: Vec<u8>,
    biome_pixels: Vec<usize>,
    /// A column deep inside each biome, for its cross-section.
    biome_heart: Vec<Option<(f64, f64)>>,
}

#[derive(Clone, Copy, Default)]
struct Pixel {
    biome: usize,
    height: Option<f64>,
    colour: [f64; 3],
    river: bool,
}

#[expect(
    clippy::too_many_lines,
    reason = "one pass samples, shades, and finds each biome's heart"
)]
fn sample_map(
    field: &TerrainField,
    names: &[String],
    centre: (f64, f64),
    span: f64,
    pixels: usize,
) -> Map {
    let step = span / pixels as f64;
    let coordinate = |index: usize, middle: f64| middle - span * 0.5 + (index as f64 + 0.5) * step;
    let threads = std::thread::available_parallelism().map_or(4, usize::from);
    let rows_per_thread = pixels.div_ceil(threads);
    let mut samples = vec![Pixel::default(); pixels * pixels];
    std::thread::scope(|scope| {
        for (chunk_index, chunk) in samples.chunks_mut(rows_per_thread * pixels).enumerate() {
            scope.spawn(move || {
                for (offset, sample) in chunk.iter_mut().enumerate() {
                    let index = chunk_index * rows_per_thread * pixels + offset;
                    let x = coordinate(index % pixels, centre.0);
                    let z = coordinate(index / pixels, centre.1);
                    let biome = field.biome_at(x, z);
                    sample.biome = names.iter().position(|name| name == biome).unwrap_or(0);
                    sample.river = field
                        .river_distance(x, z)
                        .is_some_and(|distance| distance < step);
                    sample.height = field.topmost_surface(x, z);
                    if let Some(height) = sample.height {
                        let ground =
                            field.sample_position(WorldPosition(DVec3::new(x, height - 0.03, z)));
                        sample.colour = look_colour(field.palette().look(ground.surface));
                    }
                }
            });
        }
    });
    let height_at = |column: usize, row: usize| {
        samples[row.min(pixels - 1) * pixels + column.min(pixels - 1)]
            .height
            .unwrap_or(field.vertical_range().0)
    };
    let mut biome_rgb = Vec::with_capacity(pixels * pixels * 3);
    let mut relief_rgb = Vec::with_capacity(pixels * pixels * 3);
    let mut biome_pixels = vec![0; names.len()];
    let mut sums = vec![(0.0, 0.0, 0_usize); names.len()];
    for (index, sample) in samples.iter().enumerate() {
        let (column, row) = (index % pixels, index / pixels);
        biome_pixels[sample.biome] += 1;
        let entry = &mut sums[sample.biome];
        entry.0 += coordinate(column, centre.0);
        entry.1 += coordinate(row, centre.1);
        entry.2 += 1;
        let underwater = sample
            .height
            .is_none_or(|height| height < field.sea_level());
        let biome = biome_colour(sample.biome, names.len());
        let biome = if sample.river || underwater {
            mix(biome, [0.1, 0.35, 0.85], 0.7)
        } else {
            biome
        };
        biome_rgb.extend(to_bytes(biome));
        // Light from the north-west.
        let dx =
            (height_at(column + 1, row) - height_at(column.saturating_sub(1), row)) / (2.0 * step);
        let dz =
            (height_at(column, row + 1) - height_at(column, row.saturating_sub(1))) / (2.0 * step);
        let normal = DVec3::new(-dx, 1.0, -dz).normalize();
        let light = normal.dot(DVec3::new(-0.6, 0.7, -0.4).normalize()).max(0.0) * 0.85 + 0.25;
        let mut colour = sample.colour.map(|channel| channel * light);
        if underwater {
            colour = mix(colour, [0.05, 0.2, 0.45], 0.55);
        }
        if sample.river {
            colour = mix(colour, [0.1, 0.35, 0.85], 0.8);
        }
        relief_rgb.extend(to_bytes(colour));
    }
    // A biome's heart: the sample nearest its centroid whose surroundings,
    // a cross-section's half width around it, all belong to the biome.
    let reach = (CROSS_SECTION_WIDTH * 0.5 / step).ceil() as usize;
    let interior = |index: usize, biome: usize, reach: usize| {
        let (column, row) = (index % pixels, index / pixels);
        (column.saturating_sub(reach)..=(column + reach).min(pixels - 1)).all(|x| {
            [
                row.saturating_sub(reach / 2),
                row,
                (row + reach / 2).min(pixels - 1),
            ]
            .iter()
            .all(|&y| samples[y * pixels + x].biome == biome)
        })
    };
    let biome_heart = sums
        .iter()
        .enumerate()
        .map(|(biome, &(sum_x, sum_z, count))| {
            (count > 0).then(|| {
                let (mean_x, mean_z) = (sum_x / count as f64, sum_z / count as f64);
                let nearest = |reach: usize| {
                    samples
                        .iter()
                        .enumerate()
                        .filter(|(index, sample)| {
                            sample.biome == biome && interior(*index, biome, reach)
                        })
                        .map(|(index, _)| {
                            (
                                coordinate(index % pixels, centre.0),
                                coordinate(index / pixels, centre.1),
                            )
                        })
                        .min_by(|first, second| {
                            let distance =
                                |point: &(f64, f64)| (point.0 - mean_x).hypot(point.1 - mean_z);
                            distance(first).total_cmp(&distance(second))
                        })
                };
                nearest(reach)
                    .or_else(|| nearest(reach / 2))
                    .or_else(|| nearest(0))
                    .expect("the biome has samples")
            })
        })
        .collect();
    Map {
        biome_rgb,
        relief_rgb,
        biome_pixels,
        biome_heart,
    }
}

/// A vertical slice along x through `(x, z)`, spanning the column's ground
/// ± half the slice width: solid is painted, open ground is sky or cave.
fn cross_section(field: &TerrainField, x: f64, z: f64, span: f64) -> Vec<u8> {
    let width = CROSS_SECTION_PIXELS;
    let height = CROSS_SECTION_ROWS;
    let step = span / width as f64;
    let top = field.ground_height(x, z) + height as f64 * step * 0.7;
    let mut rgb = vec![0_u8; width * height * 3];
    let threads = std::thread::available_parallelism().map_or(4, usize::from);
    let rows_per_thread = height.div_ceil(threads);
    std::thread::scope(|scope| {
        for (chunk_index, chunk) in rgb.chunks_mut(rows_per_thread * width * 3).enumerate() {
            scope.spawn(move || {
                for (offset, pixel) in chunk.chunks_mut(3).enumerate() {
                    let index = chunk_index * rows_per_thread * width + offset;
                    let (column, row) = (index % width, index / width);
                    let point = DVec3::new(
                        x - span * 0.5 + (column as f64 + 0.5) * step,
                        top - (row as f64 + 0.5) * step,
                        z,
                    );
                    let colour = if field.density(point) > 0.0 {
                        let sample = field.sample_position(WorldPosition(point));
                        look_colour(field.palette().look(sample.surface))
                    } else if point.y < field.sea_level() {
                        [0.08, 0.16, 0.3]
                    } else if field.ground_height(point.x, point.z) - 2.0 > point.y {
                        [0.03, 0.03, 0.04]
                    } else {
                        [0.62, 0.74, 0.86]
                    };
                    pixel.copy_from_slice(&to_bytes(colour));
                }
            });
        }
    });
    rgb
}

const VIEW_WIDTH: usize = 480;
const VIEW_HEIGHT: usize = 300;
const VIEW_DISTANCE: f64 = 700.0;

/// A camera at `offset` from `target` looking at it, ray marched through
/// the density field with sun, sky, and distance haze. A camera above ground
/// is kept clear of it.
fn perspective_view(field: &TerrainField, target: DVec3, offset: DVec3) -> Vec<u8> {
    let eye = target + offset;
    let eye = if offset.y > 5.0 {
        DVec3::new(
            eye.x,
            eye.y.max(field.surface_height(eye.x, eye.z) + 6.0),
            eye.z,
        )
    } else {
        eye
    };
    let forward = (target - eye).normalize();
    let right = forward.cross(DVec3::Y).normalize();
    let up = right.cross(forward);
    let sun = DVec3::new(-0.5, 0.75, -0.3).normalize();
    let tan_half = (60.0_f64.to_radians() * 0.5).tan();
    let mut rgb = vec![0_u8; VIEW_WIDTH * VIEW_HEIGHT * 3];
    let threads = std::thread::available_parallelism().map_or(4, usize::from);
    let rows_per_thread = VIEW_HEIGHT.div_ceil(threads);
    std::thread::scope(|scope| {
        for (chunk_index, chunk) in rgb.chunks_mut(rows_per_thread * VIEW_WIDTH * 3).enumerate() {
            scope.spawn(move || {
                for (offset, pixel) in chunk.chunks_mut(3).enumerate() {
                    let index = chunk_index * rows_per_thread * VIEW_WIDTH + offset;
                    let (column, row) = (index % VIEW_WIDTH, index / VIEW_WIDTH);
                    let u = ((column as f64 + 0.5) / VIEW_WIDTH as f64 * 2.0 - 1.0)
                        * tan_half
                        * VIEW_WIDTH as f64
                        / VIEW_HEIGHT as f64;
                    let v = (1.0 - (row as f64 + 0.5) / VIEW_HEIGHT as f64 * 2.0) * tan_half;
                    let direction = (forward + right * u + up * v).normalize();
                    let sky = mix(
                        [0.72, 0.8, 0.88],
                        [0.36, 0.52, 0.76],
                        direction.y.max(0.0).sqrt(),
                    );
                    let colour = march(field, eye, direction).map_or(sky, |(hit, distance)| {
                        let h = 0.05;
                        let gradient = DVec3::new(
                            field.density(hit + DVec3::X * h) - field.density(hit - DVec3::X * h),
                            field.density(hit + DVec3::Y * h) - field.density(hit - DVec3::Y * h),
                            field.density(hit + DVec3::Z * h) - field.density(hit - DVec3::Z * h),
                        );
                        let normal = (-gradient).normalize_or(DVec3::Y);
                        let sample = field.sample_position(WorldPosition(hit - normal * 0.03));
                        let base = look_colour(field.palette().look(sample.surface));
                        let lit = normal.dot(sun).max(0.0) * 0.8 + 0.25 + normal.y.max(0.0) * 0.1;
                        let shaded = base.map(|channel| channel * lit);
                        let shaded = if hit.y < field.sea_level() {
                            mix(shaded, [0.1, 0.25, 0.45], 0.6)
                        } else {
                            shaded
                        };
                        mix(
                            shaded,
                            [0.72, 0.8, 0.88],
                            (distance / VIEW_DISTANCE).powi(2) * 0.8,
                        )
                    });
                    pixel.copy_from_slice(&to_bytes(colour));
                }
            });
        }
    });
    rgb
}

/// First ground a ray meets, and how far along it lies.
fn march(field: &TerrainField, eye: DVec3, direction: DVec3) -> Option<(DVec3, f64)> {
    let mut distance = 0.5;
    let mut previous = distance;
    while distance < VIEW_DISTANCE {
        let point = eye + direction * distance;
        let density = field.density(point);
        if density > 0.0 {
            let (mut empty, mut solid) = (previous, distance);
            for _ in 0..12 {
                let middle = 0.5 * (empty + solid);
                if field.density(eye + direction * middle) > 0.0 {
                    solid = middle;
                } else {
                    empty = middle;
                }
            }
            return Some((eye + direction * solid, solid));
        }
        previous = distance;
        distance += (-density * 0.45).clamp(0.08 + distance * 0.002, 4.0);
    }
    None
}

/// A place where a carve layer shows.
struct CarveExample {
    layer: String,
    /// `open` where it breaks the surface, `buried` under a roof of rock.
    kind: &'static str,
    x: f64,
    y: f64,
    z: f64,
}

const CARVE_SEARCH_STEP: f64 = 16.0;
const CARVE_EXAMPLES: usize = 3;
const CARVE_EXAMPLE_SPACING: f64 = 400.0;

/// Scans a grid of columns for air a carve layer opened: just below the
/// ground, where it breaks the surface, or deeper under solid rock.
fn find_carves(field: &TerrainField, centre: (f64, f64), span: f64) -> Vec<CarveExample> {
    let steps = (span / CARVE_SEARCH_STEP) as usize;
    let threads = std::thread::available_parallelism().map_or(4, usize::from);
    let rows = steps.div_ceil(threads);
    let mut candidates: Vec<CarveExample> = std::thread::scope(|scope| {
        let handles = (0..threads)
            .map(|thread| {
                scope.spawn(move || {
                    let mut found = Vec::new();
                    for row in thread * rows..((thread + 1) * rows).min(steps) {
                        for column in 0..steps {
                            let x = centre.0 - span * 0.5 + column as f64 * CARVE_SEARCH_STEP;
                            let z = centre.1 - span * 0.5 + row as f64 * CARVE_SEARCH_STEP;
                            let ground = field.ground_height(x, z);
                            for (depth, kind) in [(1.5, "open"), (14.0, "buried"), (40.0, "buried")]
                            {
                                let point = DVec3::new(x, ground - depth, z);
                                if field.density(point) > 0.0 {
                                    continue;
                                }
                                let Some(layer) = field.carve_at(point) else {
                                    continue;
                                };
                                let roofed = kind == "open"
                                    || field.density(DVec3::new(x, ground - 1.0, z)) > 0.0;
                                if roofed {
                                    found.push(CarveExample {
                                        layer: layer.to_owned(),
                                        kind,
                                        x,
                                        y: point.y,
                                        z,
                                    });
                                    break;
                                }
                            }
                        }
                    }
                    found
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .flat_map(|handle| handle.join().expect("search thread"))
            .collect()
    });
    candidates.sort_by(|a, b| {
        let distance = |example: &CarveExample| (example.x - centre.0).hypot(example.z - centre.1);
        distance(a).total_cmp(&distance(b))
    });
    let mut chosen: Vec<CarveExample> = Vec::new();
    for candidate in candidates {
        let same = chosen
            .iter()
            .filter(|kept| kept.layer == candidate.layer && kept.kind == candidate.kind)
            .collect::<Vec<_>>();
        if same.len() < CARVE_EXAMPLES
            && same.iter().all(|kept| {
                (kept.x - candidate.x).hypot(kept.z - candidate.z) > CARVE_EXAMPLE_SPACING
            })
        {
            chosen.push(candidate);
        }
    }
    chosen
}

/// The horizontal direction in which air continues from `point`, a little
/// lower, as a ramp or tunnel would.
fn void_direction(field: &TerrainField, point: DVec3) -> DVec3 {
    let mut best = (f64::INFINITY, DVec3::X);
    for step in 0..32 {
        let angle = f64::from(step) / 32.0 * core::f64::consts::TAU;
        let direction = DVec3::new(angle.cos(), 0.0, angle.sin());
        let density = [6.0, 12.0, 18.0]
            .iter()
            .map(|distance| field.density(point + direction * distance - DVec3::Y * distance * 0.3))
            .sum::<f64>();
        if density < best.0 {
            best = (density, direction);
        }
    }
    best.1
}

/// Whether air connects `start` to air a layer opened, within 120 m, on a
/// 1 m lattice, giving up after 400k cells.
fn reaches_layer(field: &TerrainField, start: DVec3, layer: &str) -> bool {
    const REACH: i32 = 120;
    let mut seen = std::collections::HashSet::new();
    let mut queue = std::collections::VecDeque::from([(0_i32, 0_i32, 0_i32)]);
    seen.insert((0, 0, 0));
    while let Some((i, j, k)) = queue.pop_front() {
        if seen.len() > 400_000 {
            return false;
        }
        let point = start + DVec3::new(f64::from(i), f64::from(j), f64::from(k));
        if field.carve_at(point) == Some(layer) {
            return true;
        }
        for (di, dj, dk) in [
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ] {
            let next = (i + di, j + dj, k + dk);
            if next.0.abs() > REACH || next.2.abs() > REACH || next.1 > 0 || next.1 < -REACH {
                continue;
            }
            let position =
                start + DVec3::new(f64::from(next.0), f64::from(next.1), f64::from(next.2));
            // Stay underground: open sky above would let the search walk
            // over the surface to anything.
            if seen.contains(&next)
                || field.density(position) > 0.0
                || position.y > field.ground_height(position.x, position.z) - 1.0 && next.1 > -3
            {
                continue;
            }
            seen.insert(next);
            queue.push_back(next);
        }
    }
    false
}

/// Rough mean colour of each texture family, before tinting.
fn texture_colour(texture: TextureSet) -> [f64; 3] {
    match texture {
        TextureSet::Grass => [0.28, 0.4, 0.16],
        TextureSet::Dirt => [0.38, 0.28, 0.19],
        TextureSet::Stone => [0.5, 0.49, 0.47],
        TextureSet::Sand => [0.78, 0.7, 0.53],
        TextureSet::Iron => [0.42, 0.3, 0.26],
        TextureSet::Graphite => [0.16, 0.16, 0.17],
        TextureSet::Copper => [0.62, 0.42, 0.27],
    }
}

fn look_colour(look: SurfaceLook) -> [f64; 3] {
    let tint = look.tint.map(f64::from).map(linear_to_srgb);
    let base = texture_colour(look.texture);
    if look.recolor {
        tint
    } else {
        [base[0] * tint[0], base[1] * tint[1], base[2] * tint[2]]
    }
}

fn linear_to_srgb(value: f64) -> f64 {
    if value <= 0.003_130_8 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

fn biome_colour(index: usize, count: usize) -> [f64; 3] {
    let hue = index as f64 / count.max(1) as f64 * 6.0;
    let fraction = hue.fract();
    let (low, high) = (0.3, 0.85);
    let rising = low + (high - low) * fraction;
    let falling = high - (high - low) * fraction;
    match hue as usize % 6 {
        0 => [high, rising, low],
        1 => [falling, high, low],
        2 => [low, high, rising],
        3 => [low, falling, high],
        4 => [rising, low, high],
        _ => [high, low, falling],
    }
}

fn mix(first: [f64; 3], second: [f64; 3], amount: f64) -> [f64; 3] {
    core::array::from_fn(|channel| first[channel] + (second[channel] - first[channel]) * amount)
}

fn to_bytes(colour: [f64; 3]) -> [u8; 3] {
    colour.map(|channel| (channel.clamp(0.0, 1.0) * 255.0).round() as u8)
}

fn write_png(path: &Path, width: usize, height: usize, rgb: &[u8]) -> Result<(), Box<dyn Error>> {
    let mut encoder = png::Encoder::new(
        BufWriter::new(File::create(path)?),
        u32::try_from(width)?,
        u32::try_from(height)?,
    );
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header()?.write_image_data(rgb)?;
    Ok(())
}
