use std::collections::{HashSet, VecDeque};

use bevy_math::{DVec3, IVec3};

use super::super::forest::MAX_GROWN_STRIDE;
use super::super::model::{Parts, wood_floor};
use super::super::{Part, SpeciesSpec, TreeModel, grow_tree};
use crate::TERRAIN_CELL_METERS;

/// Solid wood corners of a lattice of `stride` cells, found around every
/// segment, as the lattice samples the tree.
fn wood_corners(tree: &TreeModel, stride: i32) -> HashSet<IVec3> {
    let spacing = f64::from(stride) * TERRAIN_CELL_METERS;
    let floor = wood_floor(stride);
    let mut solid = HashSet::new();
    let mut seen = HashSet::new();
    for segment in tree
        .segments
        .iter()
        .filter(|segment| segment.part != Part::Root)
    {
        let radius = segment.ra.max(segment.rb).max(floor);
        let low = ((segment.a.min(segment.b) - radius) / spacing)
            .floor()
            .as_ivec3();
        let high = ((segment.a.max(segment.b) + radius) / spacing)
            .ceil()
            .as_ivec3();
        for z in low.z..=high.z {
            for y in low.y..=high.y {
                for x in low.x..=high.x {
                    let corner = IVec3::new(x, y, z);
                    if !seen.insert(corner) {
                        continue;
                    }
                    let point = corner.as_dvec3() * spacing;
                    if tree
                        .sample_parts(point, Parts::Wood, floor)
                        .is_some_and(|(density, _)| density > 0.0)
                    {
                        solid.insert(corner);
                    }
                }
            }
        }
    }
    solid
}

/// Groups of edge-joined corners that never reach the base: wood a mesher
/// would draw floating.
fn floating_pieces(tree: &TreeModel, solid: &HashSet<IVec3>, stride: i32) -> Vec<usize> {
    let spacing = f64::from(stride) * TERRAIN_CELL_METERS;
    let base = tree.origin.y + 0.5 + spacing;
    let mut left = solid.clone();
    let mut pieces = Vec::new();
    while let Some(&start) = left.iter().next() {
        left.remove(&start);
        let mut queue = VecDeque::from([start]);
        let mut size = 0;
        let mut grounded = false;
        while let Some(corner) = queue.pop_front() {
            size += 1;
            grounded |= f64::from(corner.y) * spacing <= base;
            for step in [IVec3::X, IVec3::Y, IVec3::Z] {
                for next in [corner + step, corner - step] {
                    if left.remove(&next) {
                        queue.push_back(next);
                    }
                }
            }
        }
        if !grounded {
            pieces.push(size);
        }
    }
    pieces
}

#[test]
fn every_branch_holds_together_at_every_grown_stride() {
    for species in SpeciesSpec::embedded() {
        for seed in 0..2_u32 {
            // Off the lattice by a fraction of a cell, as trees stand anywhere.
            let origin = DVec3::new(0.017_f64.mul_add(f64::from(seed), 0.013), 0.031, 0.029);
            let tree = grow_tree(&species, u64::from(seed) * 7_919 + 3, origin);
            let mut stride = 1;
            while stride <= MAX_GROWN_STRIDE {
                let solid = wood_corners(&tree, stride);
                let pieces = floating_pieces(&tree, &solid, stride);
                assert!(
                    pieces.is_empty(),
                    "{} seed {seed} at stride {stride}: {} floating pieces of wood, sizes {:?}",
                    species.name,
                    pieces.len(),
                    &pieces[..pieces.len().min(8)],
                );
                stride *= 2;
            }
        }
    }
}
