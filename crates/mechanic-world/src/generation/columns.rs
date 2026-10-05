//! Every column of a lattice at once.
//!
//! A lattice's columns are independent, so each of their expressions runs
//! once over all the columns that need it rather than once per column. The
//! columns are assembled exactly as [`CompiledWorld::column`] assembles one.

use super::tape::{Lane, ManyScratch};
use super::{COLUMNS, Column, CompiledWorld, Lattice, WORLD_HALF_EXTENT_METERS, Weights};

impl CompiledWorld {
    /// A lattice's columns, z-major then x, as [`Self::cached_column`] gives
    /// them one by one. Columns missing from this thread's cache are
    /// computed together and cached.
    pub(super) fn lattice_columns(&self, lattice: &Lattice) -> Vec<Column> {
        let [nx, _, nz] = lattice.dims;
        let mut columns: Vec<Option<Column>> = Vec::with_capacity(nx * nz);
        let mut missing = Vec::new();
        let mut xs = Vec::new();
        let mut zs = Vec::new();
        COLUMNS.with_borrow(|cache| {
            for k in 0..nz {
                let z = lattice.coordinate(2, k);
                for i in 0..nx {
                    let x = lattice.coordinate(0, i);
                    let key = [x.to_bits(), z.to_bits()];
                    let cached = cache[self.column_slot(key)].and_then(|(id, cached, column)| {
                        (id == self.id && cached == key).then_some(column)
                    });
                    if cached.is_none() {
                        missing.push(columns.len());
                        xs.push(x);
                        zs.push(z);
                    }
                    columns.push(cached);
                }
            }
        });
        let computed = self.columns_many(&xs, &zs);
        COLUMNS.with_borrow_mut(|cache| {
            for (((index, column), x), z) in missing.into_iter().zip(computed).zip(xs).zip(zs) {
                let key = [x.to_bits(), z.to_bits()];
                cache[self.column_slot(key)] = Some((self.id, key, column));
                columns[index] = Some(column);
            }
        });
        columns
            .into_iter()
            .map(|column| column.expect("every column is cached or computed"))
            .collect()
    }

    /// [`Self::column`] at each `(xs[i], zs[i])`, equal to it bit for bit.
    pub(super) fn columns_many(&self, xs: &[f64], zs: &[f64]) -> Vec<Column> {
        let inside: Vec<usize> = (0..xs.len())
            .filter(|&index| {
                xs[index].abs() < WORLD_HALF_EXTENT_METERS
                    && zs[index].abs() < WORLD_HALF_EXTENT_METERS
            })
            .collect();
        let x = gather(xs, &inside);
        let z = gather(zs, &inside);
        let count = inside.len();
        let mut scratch = ManyScratch::default();
        let mut climate: [Vec<f64>; 4] = Default::default();
        for (tape, values) in self.climate.iter().zip(&mut climate) {
            tape.eval_many(
                [Lane::Row(&x), Lane::Splat(0.0), Lane::Row(&z)],
                &[],
                count,
                &mut scratch,
                values,
            );
        }
        let climate_at = |member: usize| climate.each_ref().map(|channel| channel[member]);
        let weights: Vec<Weights> = (0..count)
            .map(|member| self.weights_in(climate_at(member), x[member], z[member]))
            .collect();
        // Each biome's height at the columns it weighs into.
        let biomes = self.biomes.len();
        let mut heights = vec![0.0; count * biomes];
        let mut values = Vec::new();
        for (biome, compiled) in self.biomes.iter().enumerate() {
            let members: Vec<usize> = (0..count)
                .filter(|&member| weights[member].iter().any(|(other, _)| other == biome))
                .collect();
            if members.is_empty() {
                continue;
            }
            values.clear();
            compiled.height.eval_many(
                [
                    Lane::Row(&gather(&x, &members)),
                    Lane::Splat(0.0),
                    Lane::Row(&gather(&z, &members)),
                ],
                &[],
                members.len(),
                &mut scratch,
                &mut values,
            );
            for (&member, &value) in members.iter().zip(&values) {
                heights[member * biomes + biome] = value;
            }
        }
        let bounds = self.carve_bounds_many(&x, &z, &climate, &weights, &mut scratch);
        let layers = self.carves.len();
        let mut columns = vec![self.outside_column(); xs.len()];
        for (member, &index) in inside.iter().enumerate() {
            columns[index] = self.inside_column(
                [x[member], z[member]],
                climate_at(member),
                weights[member],
                |biome| heights[member * biomes + biome],
                |layer| bounds[member * layers + layer],
            );
        }
        columns
    }

    /// Each carve layer's roof, floor, and top at the columns where it
    /// opens, indexed `member * layers + layer`: the same weighted factor
    /// [`Self::inside_column`] sums decides where.
    fn carve_bounds_many(
        &self,
        x: &[f64],
        z: &[f64],
        climate: &[Vec<f64>; 4],
        weights: &[Weights],
        scratch: &mut ManyScratch,
    ) -> Vec<[f64; 3]> {
        let count = x.len();
        let layers = self.carves.len();
        let mut bounds = vec![[0.0; 3]; count * layers];
        let mut values = Vec::new();
        for (layer, compiled) in self.carves.iter().enumerate() {
            let members: Vec<usize> = (0..count)
                .filter(|&member| {
                    let mut factor = 0.0;
                    for (biome, weight) in weights[member].iter() {
                        factor += weight * self.biomes[biome].carves[layer];
                    }
                    factor > 0.0
                })
                .collect();
            if members.is_empty() {
                continue;
            }
            let (member_x, member_z) = (gather(x, &members), gather(z, &members));
            let member_climate = climate.each_ref().map(|channel| gather(channel, &members));
            let inputs = member_climate.each_ref().map(|channel| Lane::Row(channel));
            let point = [Lane::Row(&member_x), Lane::Splat(0.0), Lane::Row(&member_z)];
            let tapes = [
                Some(&compiled.roof),
                Some(&compiled.floor),
                compiled.top.as_ref(),
            ];
            for (bound, tape) in tapes.into_iter().enumerate() {
                values.clear();
                match tape {
                    Some(tape) => {
                        tape.eval_many(point, &inputs, members.len(), scratch, &mut values);
                    }
                    None => values.resize(members.len(), f64::INFINITY),
                }
                for (&member, &value) in members.iter().zip(&values) {
                    bounds[member * layers + layer][bound] = value;
                }
            }
        }
        bounds
    }
}

/// `values` at `members`, in order.
fn gather(values: &[f64], members: &[usize]) -> Vec<f64> {
    members.iter().map(|&member| values[member]).collect()
}
