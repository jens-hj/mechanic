//! Private held-geometry cache and conservative swept broadphase.
use std::collections::HashMap;

use super::{
    CompiledCreation, GpuTransform, MAX_PAIR_EVALUATIONS, Quat, Vec3, collider_body_radius,
    pose_at, position, rotation_travel, swept_pair_depth,
};
use crate::live_weld::{ConvexGeometry, penetration};

#[derive(Clone)]
struct Collider {
    index: usize,
    body: usize,
    radius: f32,
    local: ConvexGeometry,
}

#[derive(Clone, Copy)]
struct Bounds {
    index: usize,
    min: Vec3,
    max: Vec3,
}

#[derive(Clone)]
struct TerrainNode {
    body: usize,
    center: Vec3,
    radius: f32,
    children: Option<[usize; 2]>,
    collider: usize,
}

/// Geometry is local to the construction generation; world poses are query-local.
#[derive(Clone, Default)]
pub(super) struct ClearanceCache {
    colliders: Vec<Collider>,
    terrain_nodes: Vec<TerrainNode>,
    terrain_roots: Vec<usize>,
    samples: HashMap<(usize, u32), ConvexGeometry>,
    spare: Vec<ConvexGeometry>,
    bounds: Vec<Bounds>,
    pairs: Vec<(usize, usize)>,
    intervals: Vec<(f32, f32, u8)>,
}

impl ClearanceCache {
    pub(super) fn new(creation: &CompiledCreation, held: &[bool]) -> Self {
        let identity = GpuTransform {
            position: [0.0; 4],
            rotation: Quat::IDENTITY.to_array(),
        };
        let mut cache = Self {
            colliders: creation
                .colliders
                .iter()
                .enumerate()
                .filter(|(_, c)| held[c.compound_index as usize])
                .map(|(index, c)| Collider {
                    index,
                    body: c.compound_index as usize,
                    radius: collider_body_radius(c),
                    local: crate::live_weld::geometry(c, identity).expect("identity pose is valid"),
                })
                .collect(),
            ..Self::default()
        };
        let mut bodies = std::collections::BTreeMap::<usize, Vec<usize>>::new();
        for (index, collider) in cache.colliders.iter().enumerate() {
            bodies.entry(collider.body).or_default().push(index);
        }
        for mut indices in bodies.into_values() {
            let root = cache.build_terrain_node(&mut indices);
            cache.terrain_roots.push(root);
        }
        cache
    }

    fn build_terrain_node(&mut self, indices: &mut [usize]) -> usize {
        let (mut min, mut max) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
        for &index in indices.iter() {
            for &v in &self.colliders[index].local.vertices {
                min = min.min(v);
                max = max.max(v);
            }
        }
        let center = (min + max) * 0.5;
        let radius = indices
            .iter()
            .flat_map(|&i| &self.colliders[i].local.vertices)
            .map(|&v| v.distance(center))
            .fold(0.0, f32::max);
        let first = &self.colliders[indices[0]];
        let node = self.terrain_nodes.len();
        self.terrain_nodes.push(TerrainNode {
            body: first.body,
            center,
            radius,
            collider: first.index,
            children: None,
        });
        if indices.len() > 1 {
            let size = max - min;
            let axis = if size.x >= size.y && size.x >= size.z {
                0
            } else if size.y >= size.z {
                1
            } else {
                2
            };
            let coordinate = |i: usize| {
                let vertices = &self.colliders[i].local.vertices;
                vertices
                    .iter()
                    .map(|v| v[axis])
                    .fold(f32::INFINITY, f32::min)
            };
            indices
                .sort_unstable_by(|&a, &b| coordinate(a).total_cmp(&coordinate(b)).then(a.cmp(&b)));
            let middle = indices.len() / 2;
            let (left, right) = indices.split_at_mut(middle);
            let children = [
                self.build_terrain_node(left),
                self.build_terrain_node(right),
            ];
            self.terrain_nodes[node].children = Some(children);
        }
        node
    }

    /// A clear enclosing sphere proves every enclosed collider clear. Descend
    /// uncertain bounds and retain the existing face/triangle and escape tests.
    pub(super) fn terrain_clear(
        &self,
        start: &[GpuTransform],
        end: &[GpuTransform],
        padding: f32,
        terrain: &impl super::TerrainProbe,
        mut precise: impl FnMut(usize) -> bool,
    ) -> bool {
        let mut stack = self.terrain_roots.clone();
        while let Some(index) = stack.pop() {
            let node = &self.terrain_nodes[index];
            let a = start[node.body];
            let b = end[node.body];
            let midpoint = pose_at(a, b, 0.5);
            let center = position(midpoint) + Quat::from_array(midpoint.rotation) * node.center;
            let travel = position(a).distance(position(b))
                + rotation_travel(a, b) * (node.center.length() + node.radius);
            let radius = node.radius + padding + travel * 0.5;
            if terrain
                .penetration(center, radius)
                .is_some_and(|depth| depth <= 1.0e-4)
            {
                continue;
            }
            if let Some(children) = node.children {
                stack.extend(children);
            } else if !precise(node.collider) {
                return false;
            }
        }
        true
    }

    fn clear_samples(&mut self) {
        self.spare
            .extend(self.samples.drain().map(|(_, geometry)| geometry));
    }

    fn sample(
        &mut self,
        index: usize,
        amount: f32,
        start: &[GpuTransform],
        end: &[GpuTransform],
    ) -> bool {
        let key = (index, amount.to_bits());
        if self.samples.contains_key(&key) {
            return true;
        }
        let collider = &self.colliders[index];
        let pose = if key.1 == 0 {
            start[collider.body]
        } else if key.1 == 1.0_f32.to_bits() {
            end[collider.body]
        } else {
            pose_at(start[collider.body], end[collider.body], amount)
        };
        let Ok(frame) =
            mechanic_core::ConstructionFrame::new(position(pose), Quat::from_array(pose.rotation))
        else {
            return false;
        };
        let mut geometry = self.spare.pop().unwrap_or_else(|| ConvexGeometry {
            vertices: Vec::new(),
            normals: Vec::new(),
            edges: Vec::new(),
        });
        geometry.vertices.clear();
        geometry.normals.clear();
        geometry.edges.clear();
        geometry
            .vertices
            .extend(collider.local.vertices.iter().map(|&v| frame.point(v)));
        geometry
            .normals
            .extend(collider.local.normals.iter().map(|&v| frame.vector(v)));
        geometry
            .edges
            .extend(collider.local.edges.iter().map(|&v| frame.vector(v)));
        self.samples.insert(key, geometry);
        true
    }

    fn candidates(
        &mut self,
        creation: &CompiledCreation,
        start: &[GpuTransform],
        end: &[GpuTransform],
        swept: bool,
    ) -> bool {
        self.clear_samples();
        self.bounds.clear();
        self.pairs.clear();
        let amount = if swept { 0.5 } else { 1.0 };
        for index in 0..self.colliders.len() {
            if !self.sample(index, amount, start, end) {
                return false;
            }
            let c = &self.colliders[index];
            let geometry = &self.samples[&(index, amount.to_bits())];
            let (mut min, mut max) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
            for &v in &geometry.vertices {
                min = min.min(v);
                max = max.max(v);
            }
            // Angular arc length bounds every vertex, including offset geometry.
            // The midpoint plus half the total travel covers the entire sweep.
            let margin = if swept {
                ((position(end[c.body]) - position(start[c.body])).abs()
                    + Vec3::splat(rotation_travel(start[c.body], end[c.body]) * c.radius))
                    * 0.5
            } else {
                Vec3::ZERO
            } + Vec3::splat(1.0e-4);
            self.bounds.push(Bounds {
                index,
                min: min - margin,
                max: max + margin,
            });
        }
        self.bounds
            .sort_unstable_by(|a, b| a.min.x.total_cmp(&b.min.x).then(a.index.cmp(&b.index)));
        for (i, a) in self.bounds.iter().enumerate() {
            for b in &self.bounds[i + 1..] {
                if b.min.x > a.max.x {
                    break;
                }
                if a.max.y < b.min.y || b.max.y < a.min.y || a.max.z < b.min.z || b.max.z < a.min.z
                {
                    continue;
                }
                let first = &creation.colliders[self.colliders[a.index].index];
                let second = &creation.colliders[self.colliders[b.index].index];
                if first.compound_index == second.compound_index {
                    continue;
                }
                let pair = [
                    first.compound_index.min(second.compound_index),
                    first.compound_index.max(second.compound_index),
                ];
                if creation.collision_suppression.binary_search(&pair).is_err() {
                    self.pairs
                        .push((a.index.min(b.index), a.index.max(b.index)));
                }
            }
        }
        true
    }

    pub(super) fn endpoint_clear(
        &mut self,
        creation: &CompiledCreation,
        poses: &[GpuTransform],
    ) -> bool {
        let _timing = crate::performance_capture::FreezeStage::new("internal_endpoint");
        self.candidates(creation, poses, poses, false)
            && self.pairs.iter().all(|&(a, b)| {
                penetration(
                    &self.samples[&(a, 1.0_f32.to_bits())],
                    &self.samples[&(b, 1.0_f32.to_bits())],
                ) <= 1.0e-4
            })
    }

    pub(super) fn path_clear(
        &mut self,
        creation: &CompiledCreation,
        start: &[GpuTransform],
        end: &[GpuTransform],
    ) -> bool {
        let _timing = crate::performance_capture::FreezeStage::new("internal_path");
        if !self.candidates(creation, start, end, true) {
            return false;
        }
        for pair_index in 0..self.pairs.len() {
            let (a, b) = self.pairs[pair_index];
            let first = &self.colliders[a];
            let second = &self.colliders[b];
            let (a_body, b_body) = (first.body, second.body);
            let relative_translation = (position(end[b_body]) - position(start[b_body]))
                - (position(end[a_body]) - position(start[a_body]));
            let rotation_bound = rotation_travel(start[a_body], end[a_body]) * first.radius
                + rotation_travel(start[b_body], end[b_body]) * second.radius;
            // A proof for the complete interval needs no initial penetration
            // query: clear pairs and initially tangled pairs both accept it.
            // Broadphase already transformed every collider at this midpoint.
            if (relative_translation == Vec3::ZERO && rotation_bound == 0.0)
                || swept_pair_depth(
                    &self.samples[&(a, 0.5_f32.to_bits())],
                    &self.samples[&(b, 0.5_f32.to_bits())],
                    relative_translation,
                    rotation_bound,
                    0.5,
                    1.0e-4,
                ) <= 1.0e-4
            {
                continue;
            }
            if !self.sample(a, 0.0, start, end) || !self.sample(b, 0.0, start, end) {
                return false;
            }
            let overlap = penetration(&self.samples[&(a, 0)], &self.samples[&(b, 0)]);
            // Preserve the existing escape rule for initially tangled parts.
            if overlap > 1.0e-4 {
                continue;
            }
            let allowed = overlap.max(0.0) + 1.0e-4;
            self.intervals.clear();
            self.intervals.push((0.0, 1.0, 0));
            let mut evaluations = 0;
            while let Some((low, high, depth)) = self.intervals.pop() {
                evaluations += 1;
                if evaluations > MAX_PAIR_EVALUATIONS {
                    return false;
                }
                let midpoint = (low + high) * 0.5;
                if !self.sample(a, midpoint, start, end) || !self.sample(b, midpoint, start, end) {
                    return false;
                }
                let mid_a = &self.samples[&(a, midpoint.to_bits())];
                let mid_b = &self.samples[&(b, midpoint.to_bits())];
                if swept_pair_depth(
                    mid_a,
                    mid_b,
                    relative_translation,
                    rotation_bound,
                    (high - low) * 0.5,
                    allowed,
                ) <= allowed
                {
                    continue;
                }
                if penetration(mid_a, mid_b) > allowed || depth >= 24 {
                    return false;
                }
                self.intervals.push((midpoint, high, depth + 1));
                self.intervals.push((low, midpoint, depth + 1));
            }
        }
        true
    }
}
