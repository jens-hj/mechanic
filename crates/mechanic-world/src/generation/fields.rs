//! World fields sampled once per world on the drainage grid: the blended
//! ground height and the regional low ground beneath it. Carve layers read
//! them as `Ref("ground")` and `Ref("base")`.

use std::sync::{Arc, OnceLock};

use super::interval::Interval;
use super::rivers::{DRAINAGE_CELL_METRES, drainage_side};
use crate::WORLD_HALF_EXTENT_METERS;

/// Radius of the minimum filter that turns ground into base, in drainage
/// cells: hills narrower than about 400 m sink to the valleys around them.
const BASE_REACH_CELLS: usize = 6;

/// Box-blur passes that smooth the filtered base.
const BASE_BLUR_PASSES: usize = 3;

/// A named field whose grid is filled after the world's biomes compile,
/// since the grid is sampled from them.
pub(crate) struct FieldSlot {
    name: &'static str,
    grid: OnceLock<FieldGrid>,
}

impl core::fmt::Debug for FieldSlot {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "Field({})", self.name)
    }
}

impl FieldSlot {
    fn new(name: &'static str) -> Arc<Self> {
        Arc::new(Self {
            name,
            grid: OnceLock::new(),
        })
    }

    fn grid(&self) -> &FieldGrid {
        self.grid
            .get()
            .expect("world fields are sampled before anything reads them")
    }

    /// Bilinear value at a column.
    pub(crate) fn sample(&self, x: f64, z: f64) -> f64 {
        self.grid().sample(x, z)
    }

    /// Bounds over the columns of a box.
    pub(crate) fn interval(&self, x: Interval, z: Interval) -> Interval {
        self.grid().interval(x, z)
    }
}

/// The fields every world provides.
pub(crate) struct WorldFields {
    pub(crate) ground: Arc<FieldSlot>,
    pub(crate) base: Arc<FieldSlot>,
}

impl WorldFields {
    pub(crate) fn new() -> Self {
        Self {
            ground: FieldSlot::new("ground"),
            base: FieldSlot::new("base"),
        }
    }

    pub(crate) fn named(&self, name: &str) -> Option<&Arc<FieldSlot>> {
        match name {
            "ground" => Some(&self.ground),
            "base" => Some(&self.base),
            _ => None,
        }
    }

    /// Fills both fields from blended heights on the drainage grid.
    pub(crate) fn fill(&self, heights: &[f64]) {
        let side = drainage_side();
        #[expect(
            clippy::cast_possible_truncation,
            reason = "heights are metres well inside f32"
        )]
        let ground = heights
            .iter()
            .map(|&height| height as f32)
            .collect::<Vec<_>>();
        let mut base = min_filter(&ground, side, BASE_REACH_CELLS);
        for _ in 0..BASE_BLUR_PASSES {
            base = box_blur(&base, side);
        }
        let _ = self.ground.grid.set(FieldGrid::new(ground, side));
        let _ = self.base.grid.set(FieldGrid::new(base, side));
    }
}

/// Values at the drainage-grid points, interpolated bilinearly between them.
struct FieldGrid {
    values: Vec<f32>,
    side: usize,
}

impl FieldGrid {
    const fn new(values: Vec<f32>, side: usize) -> Self {
        Self { values, side }
    }

    fn at(&self, column: usize, row: usize) -> f64 {
        f64::from(self.values[column + row * self.side])
    }

    /// Grid coordinate of a world coordinate, clamped to the grid.
    fn cell(&self, value: f64) -> f64 {
        #[expect(clippy::cast_precision_loss, reason = "a few hundred cells")]
        let last = (self.side - 1) as f64;
        ((value + WORLD_HALF_EXTENT_METERS) / DRAINAGE_CELL_METRES).clamp(0.0, last)
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "clamped to the grid"
    )]
    fn sample(&self, x: f64, z: f64) -> f64 {
        let (across, down) = (self.cell(x), self.cell(z));
        let (column, row) = (
            (across.floor() as usize).min(self.side - 2),
            (down.floor() as usize).min(self.side - 2),
        );
        #[expect(clippy::cast_precision_loss, reason = "a few hundred cells")]
        let (along_x, along_z) = (across - column as f64, down - row as f64);
        let near = (self.at(column + 1, row) - self.at(column, row))
            .mul_add(along_x, self.at(column, row));
        let far = (self.at(column + 1, row + 1) - self.at(column, row + 1))
            .mul_add(along_x, self.at(column, row + 1));
        (far - near).mul_add(along_z, near)
    }

    /// Bilinear interpolation never leaves the hull of the grid points
    /// around it, so the extremes of those points bound a box.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "clamped to the grid"
    )]
    fn interval(&self, x: Interval, z: Interval) -> Interval {
        if !(x.lo.is_finite() && x.hi.is_finite() && z.lo.is_finite() && z.hi.is_finite()) {
            return self.range();
        }
        let first = |value: f64| (self.cell(value).floor() as usize).min(self.side - 1);
        let last = |value: f64| (self.cell(value).ceil() as usize).min(self.side - 1);
        let mut bounds = Interval::new(f64::INFINITY, f64::NEG_INFINITY);
        for row in first(z.lo)..=last(z.hi) {
            for column in first(x.lo)..=last(x.hi) {
                let value = self.at(column, row);
                bounds = Interval::new(bounds.lo.min(value), bounds.hi.max(value));
            }
        }
        bounds
    }

    fn range(&self) -> Interval {
        self.values.iter().fold(
            Interval::new(f64::INFINITY, f64::NEG_INFINITY),
            |bounds, &value| {
                Interval::new(
                    bounds.lo.min(f64::from(value)),
                    bounds.hi.max(f64::from(value)),
                )
            },
        )
    }
}

/// Minimum over a square of `reach` cells around each point, separably.
fn min_filter(values: &[f32], side: usize, reach: usize) -> Vec<f32> {
    let pass = |source: &[f32], horizontal: bool| {
        let mut out = vec![0.0; source.len()];
        for row in 0..side {
            for column in 0..side {
                let (along, fixed) = if horizontal {
                    (column, row)
                } else {
                    (row, column)
                };
                let index = |at: usize| {
                    if horizontal {
                        at + fixed * side
                    } else {
                        fixed + at * side
                    }
                };
                out[column + row * side] = (along.saturating_sub(reach)
                    ..=(along + reach).min(side - 1))
                    .map(|at| source[index(at)])
                    .fold(f32::INFINITY, f32::min);
            }
        }
        out
    };
    pass(&pass(values, true), false)
}

/// Mean over each point and its eight neighbours.
fn box_blur(values: &[f32], side: usize) -> Vec<f32> {
    let mut out = vec![0.0; values.len()];
    for row in 0..side {
        for column in 0..side {
            let mut sum = 0.0;
            let mut count = 0.0;
            for z in row.saturating_sub(1)..=(row + 1).min(side - 1) {
                for x in column.saturating_sub(1)..=(column + 1).min(side - 1) {
                    sum += values[x + z * side];
                    count += 1.0;
                }
            }
            out[column + row * side] = sum / count;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{FieldGrid, box_blur, min_filter};
    use crate::generation::interval::Interval;

    #[test]
    fn bilinear_values_stay_inside_their_interval() {
        let side = super::drainage_side();
        #[expect(clippy::cast_precision_loss, reason = "a small test grid")]
        let values = (0..side * side)
            .map(|index| ((index % 7) as f32).mul_add(3.0, (index / side % 5) as f32))
            .collect::<Vec<_>>();
        let grid = FieldGrid::new(values, side);
        for step in 0..200 {
            let x = f64::from(step).mul_add(7.3, -700.0);
            let z = f64::from(step).mul_add(-5.1, 300.0);
            let bounds = grid.interval(
                Interval::new(x - 20.0, x + 20.0),
                Interval::new(z - 9.0, z + 9.0),
            );
            for (dx, dz) in [(-20.0, -9.0), (0.0, 0.0), (20.0, 9.0), (13.0, -4.0)] {
                let value = grid.sample(x + dx, z + dz);
                assert!(
                    bounds.lo <= value && value <= bounds.hi,
                    "{value} outside {bounds:?}"
                );
            }
        }
    }

    #[test]
    fn base_lies_at_or_below_the_ground_it_filters() {
        let side = 40;
        #[expect(clippy::cast_precision_loss, reason = "a small test grid")]
        let ground = (0..side * side)
            .map(|index| (((index % side) as f32) * 0.7).sin() * 30.0 + (index / side) as f32)
            .collect::<Vec<_>>();
        let base = min_filter(&ground, side, 3);
        assert!(
            base.iter()
                .zip(&ground)
                .all(|(base, ground)| base <= ground)
        );
        let blurred = box_blur(&base, side);
        let highest = ground.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(blurred.iter().all(|&value| value <= highest));
    }
}
