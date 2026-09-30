//! Reproducible timings for stored water flooding open land: a trench cut
//! through a lake's bank lets it run out over lower ground beyond. Emits one
//! JSONL line per simulated second and a summary. `BREACH_PROFILE=1` prints
//! the ground along the breach.

use std::time::Instant;

use bevy_math::DVec3;
use mechanic_bench::stats::percentile;
use mechanic_world::{
    BrickCoord, TerrainField, TerrainOctree, TerrainWater, WaterBody, WaterPhases, WaterWorld,
    WorldPosition, WorldSeed,
};

/// Water steps per second, as the app runs them.
const STEPS_PER_SECOND: u32 = 20;

/// Simulated time, in seconds, unless `BREACH_SECONDS` says otherwise.
const RUN_SECONDS: u32 = 120;

/// A lake point, and a point beyond its bank where the ground lies at least
/// 30 cm under the lake: the breach runs from one to the other.
fn breach(field: &TerrainField) -> Option<(DVec3, DVec3, f64)> {
    let spawn = field.safe_spawn().0;
    for ring in 1..120 {
        let radius = f64::from(ring) * 4.0;
        for step in 0..ring * 8 {
            let angle = f64::from(step) / f64::from(ring * 8) * std::f64::consts::TAU;
            let (x, z) = (
                spawn.x + radius * angle.cos(),
                spawn.z + radius * angle.sin(),
            );
            let Some(surface) = field.water_surface(x, z) else {
                continue;
            };
            let deep = field
                .topmost_surface(x, z)
                .is_some_and(|ground| ground < surface.level - 0.6);
            if !matches!(surface.body, WaterBody::Lake(_)) || !deep {
                continue;
            }
            let level = surface.level;
            for direction in 0..16 {
                let angle = f64::from(direction) * std::f64::consts::TAU / 16.0;
                let mut bank = false;
                for reach in 1..40 {
                    let (bx, bz) = (
                        x + f64::from(reach) * angle.cos(),
                        z + f64::from(reach) * angle.sin(),
                    );
                    if field.water_surface(bx, bz).is_some_and(|other| {
                        other.body == surface.body
                            && field.is_water(DVec3::new(bx, level - 0.1, bz))
                    }) {
                        continue;
                    }
                    let Some(ground) = field.topmost_surface(bx, bz) else {
                        break;
                    };
                    if ground > level + 0.3 {
                        bank = true;
                    } else if bank
                        && ground < level - 0.3
                        && reach > 4
                        && falls_away(field, (bx, bz), angle, level)
                    {
                        return Some((DVec3::new(x, level, z), DVec3::new(bx, ground, bz), level));
                    }
                    if ground > level + 3.0 {
                        break;
                    }
                }
            }
        }
    }
    None
}

/// Whether the ground beyond a point keeps below `level` for 20 m and ends
/// at least 1.5 m under it, with no seed-derived water over it: open land
/// the water runs out over.
fn falls_away(field: &TerrainField, (x, z): (f64, f64), angle: f64, level: f64) -> bool {
    let ground =
        |metres: f64| field.topmost_surface(x + metres * angle.cos(), z + metres * angle.sin());
    let dry = |metres: f64| {
        field
            .water_surface(x + metres * angle.cos(), z + metres * angle.sin())
            .is_none()
    };
    (1..=20).all(|metres| {
        let metres = f64::from(metres);
        dry(metres) && ground(metres).is_some_and(|height| height < level - 0.2)
    }) && ground(20.0).is_some_and(|height| height < level - 1.5)
}

/// Digs a trench from `lake` to `end`, its floor under the lake's level.
/// Returns the edits, the bricks they changed and the trench's length.
fn dig(
    field: &TerrainField,
    lake: DVec3,
    end: DVec3,
    level: f64,
) -> (TerrainOctree, Vec<BrickCoord>, f64) {
    let mut terrain = TerrainOctree::default();
    let mut bricks = Vec::new();
    let length = lake.distance(end);
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a trench tens of metres long"
    )]
    let steps = (length / 0.25).ceil() as u32;
    for step in 0..=steps {
        let point = lake.lerp(end, f64::from(step) / f64::from(steps));
        let centre = WorldPosition(DVec3::new(point.x, level - 0.2, point.z));
        let outcome = terrain
            .excavate_sphere(field, centre, 0.7)
            .expect("the trench is dug");
        bricks.extend_from_slice(outcome.changed_brick_coordinates());
    }
    (terrain, bricks, length)
}

/// Prints the ground along the breach, for choosing a seed.
fn profile(field: &TerrainField, lake: DVec3, end: DVec3, level: f64) {
    let away = (end - lake).normalize();
    for metre in -6..30 {
        let point = end + away * f64::from(metre);
        eprintln!(
            "{metre:>3} m ground {:?} surface {:?} lake {level:.2}",
            field.topmost_surface(point.x, point.z),
            field
                .water_surface(point.x, point.z)
                .map(|surface| (surface.body, surface.level)),
        );
    }
}

/// Every step's timings, and the worst the run saw.
#[derive(Default)]
struct Samples {
    durations: Vec<f64>,
    phases: Vec<WaterPhases>,
    view_ms: Vec<f64>,
    peak_sheets: usize,
    ledger_error: f64,
}

impl Samples {
    fn summary(mut self, seed: u64, length: f64) -> String {
        self.durations.sort_by(f64::total_cmp);
        self.view_ms.sort_by(f64::total_cmp);
        let phases = &self.phases;
        let phase = |pick: fn(&WaterPhases) -> f64| {
            let mut samples = phases.iter().map(pick).collect::<Vec<_>>();
            samples.sort_by(f64::total_cmp);
            percentile(&samples, 95)
        };
        format!(
            "{{\"type\":\"water_breach\",\"seed\":{seed},\"trench_m\":{length:.1},\"steps\":{},\"step_p50_ms\":{:.3},\"step_p95_ms\":{:.3},\"step_max_ms\":{:.3},\"flood_p95_ms\":{:.3},\"exchange_p95_ms\":{:.3},\"sheets_p95_ms\":{:.3},\"joins_p95_ms\":{:.3},\"settle_p95_ms\":{:.3},\"view_p95_ms\":{:.3},\"peak_sheet_cells\":{},\"ledger_error_m3\":{:.3e}}}",
            self.durations.len(),
            percentile(&self.durations, 50),
            percentile(&self.durations, 95),
            self.durations.last().copied().unwrap_or_default(),
            phase(|phases| phases.flood_ms),
            phase(|phases| phases.exchange_ms),
            phase(|phases| phases.sheets_ms),
            phase(|phases| phases.joins_ms),
            phase(|phases| phases.settle_ms),
            percentile(&self.view_ms, 95),
            self.peak_sheets,
            self.ledger_error,
        )
    }
}

fn main() {
    let seed = std::env::args()
        .nth(1)
        .and_then(|value| value.parse().ok())
        .unwrap_or(1);
    let field = TerrainField::new(WorldSeed(seed));
    let Some((lake, end, level)) = breach(&field) else {
        eprintln!("seed {seed}: no lake with lower ground beyond its bank near spawn");
        std::process::exit(1);
    };
    if std::env::var_os("BREACH_PROFILE").is_some() {
        profile(&field, lake, end, level);
    }
    let (terrain, bricks, length) = dig(&field, lake, end, level);
    let ground = TerrainWater {
        field: &field,
        edits: &terrain,
    };
    let mut water = WaterWorld::new();
    water.terrain_changed(&ground, bricks);
    let total = water.ledger().total();
    let mut samples = Samples::default();
    let mut drawn = std::collections::HashMap::new();
    let seconds = std::env::var("BREACH_SECONDS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(RUN_SECONDS);
    for second in 1..=seconds {
        let mut second_ms = Vec::new();
        let mut sheets = 0;
        let mut spent = [0.0; 5];
        for _ in 0..STEPS_PER_SECOND {
            let started = Instant::now();
            let step = water.step(&ground, 1.0 / f64::from(STEPS_PER_SECOND));
            second_ms.push(started.elapsed().as_secs_f64() * 1000.0);
            let phases = step.phases;
            for (total, phase) in spent.iter_mut().zip([
                phases.flood_ms,
                phases.exchange_ms,
                phases.sheets_ms,
                phases.joins_ms,
                phases.settle_ms,
            ]) {
                *total += phase;
            }
            samples.phases.push(phases);
            sheets = step.sheet_cells;
            let started = Instant::now();
            let tiles = water.surface_tiles(&ground, &drawn);
            drawn = tiles
                .tiles
                .iter()
                .map(|tile| (tile.key, tile.fingerprint))
                .collect();
            std::hint::black_box(water.joined_cells());
            samples
                .view_ms
                .push(started.elapsed().as_secs_f64() * 1000.0);
            samples.ledger_error = samples
                .ledger_error
                .max((water.ledger().total() - total).abs());
        }
        samples.peak_sheets = samples.peak_sheets.max(sheets);
        second_ms.sort_by(f64::total_cmp);
        println!(
            "{{\"type\":\"water_second\",\"second\":{second},\"step_p50_ms\":{:.3},\"step_max_ms\":{:.3},\"sheet_cells\":{sheets},\"pools\":{},\"stored_m3\":{:.3},\"running_m3\":{:.3},\"joined_m3\":{:.3},\"spent_ms\":[{:.0},{:.0},{:.0},{:.0},{:.0}]}}",
            percentile(&second_ms, 50),
            second_ms.last().copied().unwrap_or_default(),
            water.pools().count(),
            water.stored_m3(),
            water.running_m3(),
            water.joined_m3(),
            spent[0],
            spent[1],
            spent[2],
            spent[3],
            spent[4],
        );
        samples.durations.extend(second_ms);
    }
    println!("{}", samples.summary(seed, length));
}
