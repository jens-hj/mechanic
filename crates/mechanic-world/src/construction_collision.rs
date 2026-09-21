//! CPU construction collision scene built from the authoritative compiled geometry.

#![expect(
    clippy::cast_possible_truncation,
    reason = "validated controller dimensions enter the retained f32 construction frame"
)]

use bevy_math::{DVec3, Mat3, Quat, Vec3};
use mechanic_core::{ColliderShape, CompiledCreation, LocalCollider, MassProperties};

use crate::{KinematicCapsuleConfig, TerrainDensity, WorldPosition};

const INVALID: u32 = u32::MAX;
const COLLISION_SKIN: f32 = 1.0e-4;

/// Published pose and motion of one compiled compound, in the render/GPU-local frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ConstructionBodyState {
    /// Compound centre of mass.
    pub translation: Vec3,
    /// Compound orientation.
    pub rotation: Quat,
    /// Centre-of-mass linear velocity.
    pub linear_velocity: Vec3,
    /// World-space angular velocity.
    pub angular_velocity: Vec3,
}

impl ConstructionBodyState {
    /// Transforms a compound-local point into the construction frame.
    pub fn transform_point(self, point: Vec3) -> Vec3 {
        self.translation + self.rotation * point
    }

    /// Transforms a construction-frame point into compound-local coordinates.
    pub fn inverse_transform_point(self, point: Vec3) -> Vec3 {
        self.rotation.inverse() * (point - self.translation)
    }

    /// Velocity at a construction-frame contact point.
    pub fn velocity_at(self, point: Vec3) -> Vec3 {
        self.linear_velocity + self.angular_velocity.cross(point - self.translation)
    }
}

/// Measurements from the most recent refresh and query.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ConstructionCollisionMetrics {
    /// Dynamic top-level refit time.
    pub dynamic_refit_ms: f64,
    /// Complete construction query time.
    pub query_ms: f64,
    /// Local collider rows entering narrowphase.
    pub candidate_count: u32,
    /// Contacts returned by narrowphase.
    pub contact_count: u32,
}

/// Earliest capsule contact with one compiled collider.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConstructionContact {
    /// Owning compiled compound row.
    pub compound_index: u32,
    /// Owning compiled collider row.
    pub collider_index: u32,
    /// Fraction of the requested displacement at contact.
    pub time_of_impact: f32,
    /// Construction-frame outward normal, pointing toward the capsule.
    pub normal: Vec3,
    /// Approximate construction-frame contact point.
    pub point: Vec3,
    /// Initial-overlap recovery depth.
    pub penetration: f32,
}

/// Composite terrain and construction view consumed by the kinematic controller.
pub struct KinematicCollisionScene<'a, T: TerrainDensity> {
    /// Exact active terrain collision scene.
    pub terrain: &'a T,
    /// Compiled construction collision index, when a creation is published.
    pub construction: Option<&'a mut ConstructionCollisionIndex>,
    /// Floating origin mapping global player coordinates into the construction frame.
    pub floating_origin: DVec3,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Aabb {
    minimum: Vec3,
    maximum: Vec3,
}

impl Aabb {
    fn empty() -> Self {
        Self {
            minimum: Vec3::splat(f32::INFINITY),
            maximum: Vec3::splat(f32::NEG_INFINITY),
        }
    }

    fn include(&mut self, point: Vec3) {
        self.minimum = self.minimum.min(point);
        self.maximum = self.maximum.max(point);
    }

    fn union(self, other: Self) -> Self {
        Self {
            minimum: self.minimum.min(other.minimum),
            maximum: self.maximum.max(other.maximum),
        }
    }

    fn expanded(self, amount: Vec3) -> Self {
        Self {
            minimum: self.minimum - amount,
            maximum: self.maximum + amount,
        }
    }

    fn overlaps(self, other: Self) -> bool {
        self.minimum.cmple(other.maximum).all() && other.minimum.cmple(self.maximum).all()
    }

    fn center(self) -> Vec3 {
        (self.minimum + self.maximum) * 0.5
    }

    fn transformed(self, translation: Vec3, rotation: Quat) -> Self {
        let mut result = Self::empty();
        for x in [self.minimum.x, self.maximum.x] {
            for y in [self.minimum.y, self.maximum.y] {
                for z in [self.minimum.z, self.maximum.z] {
                    result.include(translation + rotation * Vec3::new(x, y, z));
                }
            }
        }
        result
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct BvhNode {
    bounds: Aabb,
    left: u32,
    right: u32,
    item: u32,
}

impl BvhNode {
    fn leaf(bounds: Aabb, item: u32) -> Self {
        Self {
            bounds,
            left: INVALID,
            right: INVALID,
            item,
        }
    }

    fn branch(bounds: Aabb, left: u32, right: u32) -> Self {
        Self {
            bounds,
            left,
            right,
            item: INVALID,
        }
    }

    const fn is_leaf(self) -> bool {
        self.item != INVALID
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct CompoundTree {
    local_root: u32,
}

/// Packed two-level BVH over compiled compound and collider rows.
#[derive(Clone, Debug)]
pub struct ConstructionCollisionIndex {
    colliders: Vec<LocalCollider>,
    compounds: Vec<CompoundTree>,
    mass_properties: Vec<MassProperties>,
    is_static: Vec<bool>,
    poses: Vec<ConstructionBodyState>,
    previous_poses: Vec<ConstructionBodyState>,
    local_nodes: Vec<BvhNode>,
    static_nodes: Vec<BvhNode>,
    dynamic_nodes: Vec<BvhNode>,
    static_root: u32,
    dynamic_root: u32,
    traversal: Vec<u32>,
    candidates: Vec<u32>,
    metrics: ConstructionCollisionMetrics,
}

impl ConstructionCollisionIndex {
    /// Builds static and dynamic top-level trees plus immutable compound-local trees.
    ///
    /// # Panics
    ///
    /// Panics only if a caller constructs a malformed creation exceeding the compiled
    /// `u32` row contract instead of obtaining it from `ConstructionGraph::compile`.
    pub fn new(creation: &CompiledCreation) -> Self {
        let colliders = creation.colliders.clone();
        let mut local_nodes = Vec::with_capacity(colliders.len().saturating_mul(2));
        let mut local_bounds = Vec::with_capacity(colliders.len());
        for collider in &colliders {
            local_bounds.push(collider_local_bounds(collider));
        }
        let mut compounds = Vec::with_capacity(creation.compounds.len());
        for compound in &creation.compounds {
            let mut items = compound.collider_range.clone().collect::<Vec<_>>();
            let root = build_bvh(&mut local_nodes, &mut items, &local_bounds);
            compounds.push(CompoundTree { local_root: root });
        }
        let poses = creation
            .compounds
            .iter()
            .map(|compound| ConstructionBodyState {
                translation: compound.root_translation,
                rotation: compound.root_rotation,
                linear_velocity: Vec3::ZERO,
                angular_velocity: Vec3::ZERO,
            })
            .collect::<Vec<_>>();
        let is_static = creation
            .compounds
            .iter()
            .map(|compound| compound.is_static)
            .collect::<Vec<_>>();
        let mass_properties = creation
            .compounds
            .iter()
            .map(|compound| compound.mass_properties)
            .collect::<Vec<_>>();
        let body_bounds = compounds
            .iter()
            .enumerate()
            .map(|(body, tree)| {
                let local = tree
                    .local_root
                    .checked_sub(0)
                    .and_then(|root| local_nodes.get(root as usize))
                    .map_or_else(Aabb::empty, |node| node.bounds);
                local.transformed(poses[body].translation, poses[body].rotation)
            })
            .collect::<Vec<_>>();
        let mut static_nodes = Vec::with_capacity(creation.compounds.len().saturating_mul(2));
        let mut dynamic_nodes = Vec::with_capacity(creation.compounds.len().saturating_mul(2));
        let mut static_items = (0..creation.compounds.len())
            .filter(|&body| is_static[body])
            .map(|body| u32::try_from(body).expect("body row fits u32"))
            .collect::<Vec<_>>();
        let mut dynamic_items = (0..creation.compounds.len())
            .filter(|&body| !is_static[body])
            .map(|body| u32::try_from(body).expect("body row fits u32"))
            .collect::<Vec<_>>();
        let static_root = build_bvh(&mut static_nodes, &mut static_items, &body_bounds);
        let dynamic_root = build_bvh(&mut dynamic_nodes, &mut dynamic_items, &body_bounds);
        let collider_count = colliders.len();
        let scratch_capacity = collider_count
            .max(local_nodes.len())
            .max(static_nodes.len().saturating_add(dynamic_nodes.len()));
        let previous_poses = poses.clone();
        Self {
            colliders,
            compounds,
            mass_properties,
            is_static,
            poses,
            previous_poses,
            local_nodes,
            static_nodes,
            dynamic_nodes,
            static_root,
            dynamic_root,
            traversal: Vec::with_capacity(scratch_capacity),
            candidates: Vec::with_capacity(collider_count),
            metrics: ConstructionCollisionMetrics::default(),
        }
    }

    /// Replaces published body poses and refits only the dynamic top-level tree.
    ///
    /// Returns false without changing the scene when the body count differs.
    pub fn refit_dynamic(&mut self, poses: &[ConstructionBodyState]) -> bool {
        if poses.len() != self.poses.len() {
            return false;
        }
        let started = std::time::Instant::now();
        self.previous_poses.copy_from_slice(&self.poses);
        self.poses.copy_from_slice(poses);
        for index in (0..self.dynamic_nodes.len()).rev() {
            let node = self.dynamic_nodes[index];
            self.dynamic_nodes[index].bounds = if node.is_leaf() {
                self.body_world_bounds(node.item)
            } else {
                self.dynamic_nodes[node.left as usize]
                    .bounds
                    .union(self.dynamic_nodes[node.right as usize].bounds)
            };
        }
        self.metrics.dynamic_refit_ms = started.elapsed().as_secs_f64() * 1_000.0;
        true
    }

    /// Latest query/refit counters.
    pub const fn metrics(&self) -> ConstructionCollisionMetrics {
        self.metrics
    }

    /// Published body pose.
    pub fn body_pose(&self, compound_index: u32) -> Option<ConstructionBodyState> {
        self.poses.get(compound_index as usize).copied()
    }

    pub(crate) fn walkable_surface_height(
        &self,
        collider_index: u32,
        horizontal_position: Vec3,
        minimum_height: f32,
        maximum_height: f32,
        config: KinematicCapsuleConfig,
        expansion_direction: Option<Vec3>,
    ) -> Option<(f32, Vec3, f32)> {
        let collider = self.colliders.get(collider_index as usize)?;
        let pose = *self.poses.get(collider.compound_index as usize)?;
        let minimum_normal_y = config.maximum_slope.cos() as f32 - 1.0e-6;
        let mut best = None;
        let mut consider = |surface_height: f32, normal: Vec3, contained: bool| {
            let feet_height = surface_height + config.radius as f32 * (normal.y.recip() - 1.0);
            if contained
                && normal.y >= minimum_normal_y
                && feet_height >= minimum_height
                && feet_height <= maximum_height
                && best.is_none_or(|(best_height, _, _)| feet_height > best_height)
            {
                best = Some((feet_height, normal, surface_height));
            }
        };
        match &collider.shape {
            ColliderShape::Cuboid {
                local_rotation,
                half_extents,
            } => {
                let rotation = pose.rotation * *local_rotation;
                let inverse = rotation.inverse();
                let local_expansion = expansion_direction
                    .map(|direction| inverse * direction.with_y(0.0).normalize_or_zero());
                let center = pose.transform_point(collider.local_center);
                for axis in 0..3 {
                    for sign in [-1.0_f32, 1.0] {
                        let mut local_normal = Vec3::ZERO;
                        local_normal[axis] = sign;
                        let normal = rotation * local_normal;
                        if normal.y < minimum_normal_y {
                            continue;
                        }
                        let plane_point = center + normal * half_extents[axis];
                        let height = (normal.dot(plane_point)
                            - normal.x * horizontal_position.x
                            - normal.z * horizontal_position.z)
                            / normal.y;
                        let point = Vec3::new(horizontal_position.x, height, horizontal_position.z);
                        let local_point = inverse * (point - center);
                        let contained = (0..3).filter(|&other| other != axis).all(|other| {
                            let margin =
                                local_expansion.map_or(config.radius as f32, |direction| {
                                    if local_point[other] * direction[other] < -1.0e-6 {
                                        config.radius as f32
                                    } else {
                                        0.0
                                    }
                                });
                            local_point[other].abs()
                                <= half_extents[other] + margin + COLLISION_SKIN
                        });
                        consider(height, normal, contained);
                    }
                }
            }
            ColliderShape::Convex(convex) => {
                for plane in &convex.face_planes {
                    let local_normal = plane.truncate();
                    let normal = pose.rotation * local_normal;
                    if normal.y < minimum_normal_y {
                        continue;
                    }
                    let world_offset = plane.w + normal.dot(pose.translation);
                    let height = (world_offset
                        - normal.x * horizontal_position.x
                        - normal.z * horizontal_position.z)
                        / normal.y;
                    let point = Vec3::new(horizontal_position.x, height, horizontal_position.z);
                    let local_point = pose.inverse_transform_point(point);
                    let contained = convex.face_planes.iter().all(|other| {
                        let margin =
                            expansion_direction.map_or(config.radius as f32, |direction| {
                                let world_normal = pose.rotation * other.truncate();
                                if world_normal.dot(direction.with_y(0.0).normalize_or_zero())
                                    < -1.0e-6
                                {
                                    config.radius as f32
                                } else {
                                    0.0
                                }
                            });
                        other.truncate().dot(local_point) <= other.w + margin + COLLISION_SKIN
                    });
                    consider(height, normal, contained);
                }
            }
        }
        best
    }

    /// Marks the latest published body motion as consumed by one controller tick.
    pub fn finish_motion_snapshot(&mut self) {
        self.previous_poses.copy_from_slice(&self.poses);
    }

    /// True for a fixed compound.
    pub fn body_is_static(&self, compound_index: u32) -> bool {
        self.is_static
            .get(compound_index as usize)
            .copied()
            .unwrap_or(true)
    }

    /// Inverse-mass denominator along a world-space contact direction.
    pub fn body_effective_inverse_mass(
        &self,
        compound_index: u32,
        point: Vec3,
        direction: Vec3,
    ) -> f32 {
        let Some(properties) = self.mass_properties.get(compound_index as usize) else {
            return 0.0;
        };
        if self.body_is_static(compound_index) {
            return 0.0;
        }
        let pose = self.poses[compound_index as usize];
        let arm = point - pose.translation;
        let local_torque = pose.rotation.inverse() * arm.cross(direction);
        let local_angular = properties.inverse_inertia * local_torque;
        let angular = pose.rotation * local_angular;
        properties.inverse_mass + direction.dot(angular.cross(arm))
    }

    /// Sweeps an upright capsule through the construction frame.
    pub fn cast_capsule(
        &mut self,
        feet: Vec3,
        displacement: Vec3,
        config: KinematicCapsuleConfig,
    ) -> Option<ConstructionContact> {
        let started = std::time::Instant::now();
        self.collect_candidates(feet, displacement, config);
        let candidate_count = self.candidates.len();
        let mut best: Option<ConstructionContact> = None;
        for candidate_offset in 0..candidate_count {
            let collider_index = self.candidates[candidate_offset];
            let collider = &self.colliders[collider_index as usize];
            let pose = self.poses[collider.compound_index as usize];
            let previous_pose = self.previous_poses[collider.compound_index as usize];
            let Some(hit) = cast_compiled_collider(
                collider,
                previous_pose,
                pose,
                feet,
                displacement,
                config,
                collider_index,
            ) else {
                continue;
            };
            if best.is_none_or(|current| {
                hit.time_of_impact < current.time_of_impact
                    || (hit.time_of_impact.total_cmp(&current.time_of_impact)
                        == core::cmp::Ordering::Equal
                        && hit.penetration > current.penetration)
            }) {
                best = Some(hit);
            }
        }
        self.metrics.query_ms = started.elapsed().as_secs_f64() * 1_000.0;
        self.metrics.candidate_count = u32::try_from(candidate_count).unwrap_or(u32::MAX);
        self.metrics.contact_count = u32::from(best.is_some());
        best
    }

    /// Current traversal and candidate capacities, used to verify warmed queries do not grow.
    pub fn scratch_capacities(&self) -> (usize, usize) {
        (self.traversal.capacity(), self.candidates.capacity())
    }

    fn body_world_bounds(&self, body: u32) -> Aabb {
        let pose = self.poses[body as usize];
        let previous_pose = self.previous_poses[body as usize];
        let root = self.compounds[body as usize].local_root;
        self.local_nodes
            .get(root as usize)
            .map_or_else(Aabb::empty, |node| {
                node.bounds
                    .transformed(pose.translation, pose.rotation)
                    .union(
                        node.bounds
                            .transformed(previous_pose.translation, previous_pose.rotation),
                    )
            })
    }

    fn collect_candidates(
        &mut self,
        feet: Vec3,
        displacement: Vec3,
        config: KinematicCapsuleConfig,
    ) {
        self.candidates.clear();
        let radius = config.radius as f32;
        let height = config.standing_height as f32;
        let start = Aabb {
            minimum: feet - Vec3::new(radius, 0.0, radius),
            maximum: feet + Vec3::new(radius, height, radius),
        };
        let end = Aabb {
            minimum: start.minimum + displacement,
            maximum: start.maximum + displacement,
        };
        let query = start.union(end).expanded(Vec3::splat(COLLISION_SKIN));
        self.collect_top_level(self.static_root, true, query);
        self.collect_top_level(self.dynamic_root, false, query);
    }

    fn collect_top_level(&mut self, root: u32, static_tree: bool, query: Aabb) {
        if root == INVALID {
            return;
        }
        self.traversal.clear();
        self.traversal.push(root);
        while let Some(index) = self.traversal.pop() {
            let node = if static_tree {
                self.static_nodes[index as usize]
            } else {
                self.dynamic_nodes[index as usize]
            };
            if !node.bounds.overlaps(query) {
                continue;
            }
            if node.is_leaf() {
                self.collect_local(node.item, query);
            } else {
                self.traversal.push(node.left);
                self.traversal.push(node.right);
            }
        }
    }

    fn collect_local(&mut self, body: u32, world_query: Aabb) {
        let pose = self.poses[body as usize];
        let previous_pose = self.previous_poses[body as usize];
        let local_query = world_query
            .transformed(
                -(pose.rotation.inverse() * pose.translation),
                pose.rotation.inverse(),
            )
            .union(world_query.transformed(
                -(previous_pose.rotation.inverse() * previous_pose.translation),
                previous_pose.rotation.inverse(),
            ));
        let root = self.compounds[body as usize].local_root;
        if root == INVALID {
            return;
        }
        let stack_start = self.traversal.len();
        self.traversal.push(root);
        while self.traversal.len() > stack_start {
            let index = self.traversal.pop().expect("local traversal is non-empty");
            let node = self.local_nodes[index as usize];
            if !node.bounds.overlaps(local_query) {
                continue;
            }
            if node.is_leaf() {
                self.candidates.push(node.item);
            } else {
                self.traversal.push(node.left);
                self.traversal.push(node.right);
            }
        }
    }
}

fn collider_local_bounds(collider: &LocalCollider) -> Aabb {
    match &collider.shape {
        ColliderShape::Cuboid {
            local_rotation,
            half_extents,
        } => Aabb {
            minimum: -*half_extents,
            maximum: *half_extents,
        }
        .transformed(collider.local_center, *local_rotation),
        ColliderShape::Convex(convex) => {
            let mut bounds = Aabb::empty();
            for &vertex in &convex.vertices {
                bounds.include(vertex);
            }
            bounds
        }
    }
}

fn build_bvh(nodes: &mut Vec<BvhNode>, items: &mut [u32], bounds: &[Aabb]) -> u32 {
    if items.is_empty() {
        return INVALID;
    }
    let node_index = u32::try_from(nodes.len()).expect("BVH node count fits u32");
    nodes.push(BvhNode::leaf(Aabb::empty(), INVALID));
    if items.len() == 1 {
        nodes[node_index as usize] = BvhNode::leaf(bounds[items[0] as usize], items[0]);
        return node_index;
    }
    let aggregate = items
        .iter()
        .map(|&item| bounds[item as usize])
        .reduce(Aabb::union)
        .unwrap_or_default();
    let extent = aggregate.maximum - aggregate.minimum;
    let axis = if extent.x >= extent.y && extent.x >= extent.z {
        0
    } else if extent.y >= extent.z {
        1
    } else {
        2
    };
    items.sort_unstable_by(|left, right| {
        bounds[*left as usize].center()[axis].total_cmp(&bounds[*right as usize].center()[axis])
    });
    let middle = items.len() / 2;
    let (left_items, right_items) = items.split_at_mut(middle);
    let left = build_bvh(nodes, left_items, bounds);
    let right = build_bvh(nodes, right_items, bounds);
    nodes[node_index as usize] = BvhNode::branch(
        nodes[left as usize]
            .bounds
            .union(nodes[right as usize].bounds),
        left,
        right,
    );
    node_index
}

fn capsule_segment(feet: Vec3, config: KinematicCapsuleConfig) -> (Vec3, Vec3) {
    let radius = config.radius as f32;
    (
        feet + Vec3::Y * radius,
        feet + Vec3::Y * (config.standing_height as f32 - radius),
    )
}

fn cast_compiled_collider(
    collider: &LocalCollider,
    previous_pose: ConstructionBodyState,
    current_pose: ConstructionBodyState,
    feet: Vec3,
    displacement: Vec3,
    config: KinematicCapsuleConfig,
    collider_index: u32,
) -> Option<ConstructionContact> {
    let mut time = 0.0_f32;
    let mut last_normal = Vec3::Y;
    let body_displacement = current_pose.translation - previous_pose.translation;
    let rotation_delta = current_pose.rotation * previous_pose.rotation.inverse();
    let (_, rotation_angle) = rotation_delta.to_axis_angle();
    let rotational_travel =
        rotation_angle.abs() * collider_local_bounds(collider).maximum.abs().max_element();
    for _ in 0..20 {
        let moved_feet = feet + displacement * time;
        let pose = ConstructionBodyState {
            translation: previous_pose
                .translation
                .lerp(current_pose.translation, time),
            rotation: previous_pose.rotation.slerp(current_pose.rotation, time),
            linear_velocity: current_pose.linear_velocity,
            angular_velocity: current_pose.angular_velocity,
        };
        let (distance, normal, penetration) = collider_distance(collider, pose, moved_feet, config);
        last_normal = normal;
        if distance <= COLLISION_SKIN * 2.0 {
            if penetration <= COLLISION_SKIN
                && (displacement - body_displacement).dot(normal) >= -1.0e-7
            {
                return None;
            }
            let (bottom, top) = capsule_segment(moved_feet, config);
            let point =
                closest_on_segment(bottom, top, pose.translation) - normal * config.radius as f32;
            return Some(ConstructionContact {
                compound_index: collider.compound_index,
                collider_index,
                time_of_impact: time,
                normal,
                point,
                penetration,
            });
        }
        let approach = -(displacement - body_displacement).dot(normal) + rotational_travel;
        if approach <= 1.0e-7 {
            return None;
        }
        time += (distance - COLLISION_SKIN) / approach;
        if time > 1.0 {
            return None;
        }
    }
    let _ = last_normal;
    None
}

fn collider_distance(
    collider: &LocalCollider,
    pose: ConstructionBodyState,
    feet: Vec3,
    config: KinematicCapsuleConfig,
) -> (f32, Vec3, f32) {
    let (bottom, top) = capsule_segment(feet, config);
    match &collider.shape {
        ColliderShape::Cuboid {
            local_rotation,
            half_extents,
        } => {
            let rotation = pose.rotation * *local_rotation;
            let center = pose.transform_point(collider.local_center);
            capsule_obb_distance(
                bottom,
                top,
                config.radius as f32,
                center,
                rotation,
                *half_extents,
            )
        }
        ColliderShape::Convex(convex) => {
            let support = |direction: Vec3| {
                convex
                    .vertices
                    .iter()
                    .copied()
                    .max_by(|left, right| left.dot(direction).total_cmp(&right.dot(direction)))
                    .unwrap_or(Vec3::ZERO)
            };
            let local_bottom = pose.inverse_transform_point(bottom);
            let local_top = pose.inverse_transform_point(top);
            let (distance, local_normal, intersects) =
                gjk_capsule_distance(local_bottom, local_top, config.radius as f32, support);
            if intersects {
                let (normal, depth) = convex_penetration(
                    convex.face_planes.iter().copied(),
                    local_bottom,
                    local_top,
                    config.radius as f32,
                );
                (0.0, pose.rotation * normal, depth)
            } else {
                (distance, pose.rotation * local_normal, 0.0)
            }
        }
    }
}

fn capsule_obb_distance(
    bottom: Vec3,
    top: Vec3,
    radius: f32,
    center: Vec3,
    rotation: Quat,
    half_extents: Vec3,
) -> (f32, Vec3, f32) {
    let inverse = rotation.inverse();
    let a = inverse * (bottom - center);
    let b = inverse * (top - center);
    let (segment_point, box_point) = closest_segment_aabb(a, b, -half_extents, half_extents);
    let delta = segment_point - box_point;
    let raw_distance = delta.length();
    if raw_distance > 1.0e-7 {
        let separation = raw_distance - radius;
        return (
            separation.max(0.0),
            rotation * (delta / raw_distance),
            (-separation).max(0.0),
        );
    }
    let point = segment_point.clamp(-half_extents, half_extents);
    let face_depths = half_extents - point.abs();
    let axis = if face_depths.x <= face_depths.y && face_depths.x <= face_depths.z {
        0
    } else if face_depths.y <= face_depths.z {
        1
    } else {
        2
    };
    let mut normal = Vec3::ZERO;
    normal[axis] = if point[axis] >= 0.0 { 1.0 } else { -1.0 };
    (0.0, rotation * normal, radius + face_depths[axis])
}

fn closest_segment_aabb(a: Vec3, b: Vec3, minimum: Vec3, maximum: Vec3) -> (Vec3, Vec3) {
    let direction = b - a;
    let mut breaks = [0.0_f32; 8];
    let mut count = 2;
    breaks[0] = 0.0;
    breaks[1] = 1.0;
    for axis in 0..3 {
        if direction[axis].abs() <= 1.0e-8 {
            continue;
        }
        for bound in [minimum[axis], maximum[axis]] {
            let time = (bound - a[axis]) / direction[axis];
            if time > 0.0 && time < 1.0 {
                breaks[count] = time;
                count += 1;
            }
        }
    }
    breaks[..count].sort_by(f32::total_cmp);
    let mut best_segment = a;
    let mut best_box = a.clamp(minimum, maximum);
    let mut best_distance = (best_segment - best_box).length_squared();
    for interval in breaks[..count].windows(2) {
        let low = interval[0];
        let high = interval[1];
        let middle = (low + high) * 0.5;
        let sample = a + direction * middle;
        let mut coefficient = Vec3::ZERO;
        let mut constant = Vec3::ZERO;
        for axis in 0..3 {
            if sample[axis] < minimum[axis] {
                coefficient[axis] = direction[axis];
                constant[axis] = a[axis] - minimum[axis];
            } else if sample[axis] > maximum[axis] {
                coefficient[axis] = direction[axis];
                constant[axis] = a[axis] - maximum[axis];
            }
        }
        let denominator = coefficient.length_squared();
        let optimum = if denominator > 1.0e-12 {
            (-constant.dot(coefficient) / denominator).clamp(low, high)
        } else {
            low
        };
        for time in [low, optimum, high] {
            let segment = a + direction * time;
            let box_point = segment.clamp(minimum, maximum);
            let distance = (segment - box_point).length_squared();
            if distance < best_distance {
                best_distance = distance;
                best_segment = segment;
                best_box = box_point;
            }
        }
    }
    (best_segment, best_box)
}

fn closest_on_segment(a: Vec3, b: Vec3, point: Vec3) -> Vec3 {
    let direction = b - a;
    let denominator = direction.length_squared();
    if denominator <= 1.0e-12 {
        a
    } else {
        a + direction * ((point - a).dot(direction) / denominator).clamp(0.0, 1.0)
    }
}

#[derive(Clone, Copy)]
struct SimplexPoint(Vec3);

fn gjk_capsule_distance(
    bottom: Vec3,
    top: Vec3,
    radius: f32,
    shape_support: impl Fn(Vec3) -> Vec3,
) -> (f32, Vec3, bool) {
    let capsule_support = |direction: Vec3| {
        let endpoint = if bottom.dot(direction) > top.dot(direction) {
            bottom
        } else {
            top
        };
        endpoint + direction.normalize_or(Vec3::X) * radius
    };
    let support =
        |direction: Vec3| SimplexPoint(capsule_support(direction) - shape_support(-direction));
    let mut simplex = [SimplexPoint(Vec3::ZERO); 4];
    let mut len = 1;
    let mut direction = ((bottom + top) * 0.5 - shape_support(Vec3::X)).normalize_or(Vec3::X);
    simplex[0] = support(-direction);
    let mut closest = simplex[0].0;
    for _ in 0..24 {
        if closest.length_squared() <= 1.0e-12 {
            return (0.0, direction.normalize_or(Vec3::Y), true);
        }
        direction = -closest;
        let candidate = support(direction);
        let improvement = candidate.0.dot(direction) - closest.dot(direction);
        if improvement <= 1.0e-6 * direction.length().max(1.0) {
            let distance = closest.length();
            return (distance, closest.normalize_or(Vec3::Y), false);
        }
        if len == 4 {
            len = 3;
        }
        simplex[len] = candidate;
        len += 1;
        let reduced = reduce_simplex(&simplex[..len]);
        closest = reduced.0;
        len = reduced.1;
        simplex[..len].copy_from_slice(&reduced.2[..len]);
        if reduced.3 {
            return (0.0, direction.normalize_or(Vec3::Y), true);
        }
    }
    let distance = closest.length();
    (
        distance,
        closest.normalize_or(Vec3::Y),
        distance <= COLLISION_SKIN,
    )
}

fn reduce_simplex(points: &[SimplexPoint]) -> (Vec3, usize, [SimplexPoint; 4], bool) {
    let mut best_distance = f32::INFINITY;
    let mut best_point = Vec3::ZERO;
    let mut best = [SimplexPoint(Vec3::ZERO); 4];
    let mut best_len = 0;
    let subset_count = 1_u32 << points.len();
    for mask in 1..subset_count {
        let mut subset = [Vec3::ZERO; 4];
        let mut indices = [0_usize; 4];
        let mut count = 0;
        for (index, point) in points.iter().enumerate() {
            if mask & (1 << index) != 0 {
                subset[count] = point.0;
                indices[count] = index;
                count += 1;
            }
        }
        let Some((candidate, weights)) = closest_affine_origin(&subset[..count]) else {
            continue;
        };
        if weights[..count].iter().any(|weight| *weight < -1.0e-5) {
            continue;
        }
        let distance = candidate.length_squared();
        if distance < best_distance {
            best_distance = distance;
            best_point = candidate;
            best_len = 0;
            for index in 0..count {
                if weights[index] > 1.0e-6 || count == 1 {
                    best[best_len] = points[indices[index]];
                    best_len += 1;
                }
            }
        }
    }
    let contains = best_distance <= 1.0e-12 && points.len() == 4;
    (best_point, best_len.max(1), best, contains)
}

fn closest_affine_origin(points: &[Vec3]) -> Option<(Vec3, [f32; 4])> {
    let mut weights = [0.0_f32; 4];
    match points.len() {
        1 => {
            weights[0] = 1.0;
            Some((points[0], weights))
        }
        2 => {
            let edge = points[1] - points[0];
            let denominator = edge.length_squared();
            let t = if denominator <= 1.0e-12 {
                0.0
            } else {
                -points[0].dot(edge) / denominator
            };
            weights[0] = 1.0 - t;
            weights[1] = t;
            Some((points[0] + edge * t, weights))
        }
        3 => {
            let first = points[1] - points[0];
            let second = points[2] - points[0];
            let aa = first.dot(first);
            let ab = first.dot(second);
            let bb = second.dot(second);
            let rhs_a = -points[0].dot(first);
            let rhs_b = -points[0].dot(second);
            let determinant = aa * bb - ab * ab;
            if determinant.abs() <= 1.0e-12 {
                return None;
            }
            let u = (rhs_a * bb - rhs_b * ab) / determinant;
            let v = (rhs_b * aa - rhs_a * ab) / determinant;
            weights[0] = 1.0 - u - v;
            weights[1] = u;
            weights[2] = v;
            Some((points[0] + first * u + second * v, weights))
        }
        4 => {
            let matrix = Mat3::from_cols(
                points[1] - points[0],
                points[2] - points[0],
                points[3] - points[0],
            );
            if matrix.determinant().abs() <= 1.0e-12 {
                return None;
            }
            let coordinates = matrix.inverse() * -points[0];
            weights[0] = 1.0 - coordinates.x - coordinates.y - coordinates.z;
            weights[1] = coordinates.x;
            weights[2] = coordinates.y;
            weights[3] = coordinates.z;
            Some((Vec3::ZERO, weights))
        }
        _ => None,
    }
}

fn convex_penetration(
    planes: impl Iterator<Item = bevy_math::Vec4>,
    bottom: Vec3,
    top: Vec3,
    radius: f32,
) -> (Vec3, f32) {
    let mut best_normal = Vec3::Y;
    let mut best_separation = f32::NEG_INFINITY;
    for plane in planes {
        let normal = plane.truncate();
        let minimum = bottom.dot(normal).min(top.dot(normal)) - radius;
        let separation = minimum - plane.w;
        if separation > best_separation {
            best_separation = separation;
            best_normal = normal;
        }
    }
    (
        best_normal.normalize_or(Vec3::Y),
        (-best_separation).max(COLLISION_SKIN),
    )
}

/// Converts a global player position to the construction's retained local `f32` frame.
pub(crate) fn construction_feet(position: WorldPosition, floating_origin: DVec3) -> Vec3 {
    (position.0 - floating_origin).as_vec3()
}

#[cfg(test)]
mod tests {
    use bevy_math::{IVec3, Vec3, Vec4};
    use mechanic_core::{
        BuildCommand, BuildOutcome, BuildPose, ColliderShape, CompiledConvex, ConstructionGraph,
        CuboidSpec, CylinderDimensions, CylinderSpec, GridRotation,
    };

    use super::ConstructionCollisionIndex;
    use crate::KinematicCapsuleConfig;

    fn cuboid_creation(dimensions: [u8; 3], translation: IVec3) -> mechanic_core::CompiledCreation {
        let mut graph = ConstructionGraph::new();
        let spec = CuboidSpec::new(
            dimensions,
            BuildPose::new(translation, GridRotation::default()),
        )
        .unwrap();
        let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap() else {
            panic!("cuboid spawn returns its part");
        };
        graph.compile_with_static_parts([part]).unwrap()
    }

    #[test]
    fn swept_capsule_stops_at_a_thin_compiled_wall() {
        let creation = cuboid_creation([1, 8, 8], IVec3::new(4, 4, 0));
        let mut index = ConstructionCollisionIndex::new(&creation);
        let hit = index.cast_capsule(Vec3::ZERO, Vec3::X * 2.0, KinematicCapsuleConfig::default());
        let hit = hit.unwrap_or_else(|| panic!("sweep reaches wall: {:?}", index.metrics()));
        assert!(
            hit.time_of_impact > 0.2 && hit.time_of_impact < 0.4,
            "{hit:?}"
        );
        assert!(hit.normal.dot(Vec3::NEG_X) > 0.99, "{hit:?}");
    }

    #[test]
    fn compiled_floor_and_ceiling_return_opposite_normals() {
        let floor = cuboid_creation([8, 1, 8], IVec3::new(0, 0, 0));
        let mut floor_index = ConstructionCollisionIndex::new(&floor);
        let down = floor_index.cast_capsule(
            Vec3::Y,
            Vec3::NEG_Y * 2.0,
            KinematicCapsuleConfig::default(),
        );
        let down =
            down.unwrap_or_else(|| panic!("capsule lands on floor: {:?}", floor_index.metrics()));
        assert!(down.normal.dot(Vec3::Y) > 0.99, "{down:?}");

        let ceiling = cuboid_creation([8, 1, 8], IVec3::new(0, 8, 0));
        let mut ceiling_index = ConstructionCollisionIndex::new(&ceiling);
        let up = ceiling_index
            .cast_capsule(Vec3::ZERO, Vec3::Y, KinematicCapsuleConfig::default())
            .expect("capsule reaches ceiling");
        assert!(up.normal.dot(Vec3::NEG_Y) > 0.99, "{up:?}");
    }

    #[test]
    fn compiled_cylinder_bore_remains_passable() {
        let mut graph = ConstructionGraph::new();
        let dimensions = CylinderDimensions::new(1.0, 0.6, 1.0).unwrap();
        graph
            .apply(BuildCommand::SpawnCylinder(CylinderSpec::new(
                dimensions,
                BuildPose::default(),
            )))
            .unwrap();
        let creation = graph.compile().unwrap();
        let mut index = ConstructionCollisionIndex::new(&creation);
        let config = KinematicCapsuleConfig {
            radius: 0.1,
            standing_height: 0.2,
            ..KinematicCapsuleConfig::default()
        };
        assert!(
            index
                .cast_capsule(Vec3::Y, Vec3::NEG_Y * 2.0, config)
                .is_none(),
            "source-part bounds must not fill the compiled bore"
        );
    }

    #[test]
    fn compiled_convex_uses_support_mapped_sweep() {
        let mut creation = cuboid_creation([4, 4, 4], IVec3::ZERO);
        creation.colliders[0].shape = ColliderShape::Convex(CompiledConvex {
            vertices: [-0.5, 0.5]
                .into_iter()
                .flat_map(|x| {
                    [-0.5, 0.5]
                        .into_iter()
                        .flat_map(move |y| [-0.5, 0.5].into_iter().map(move |z| Vec3::new(x, y, z)))
                })
                .collect(),
            face_planes: vec![
                Vec4::new(1.0, 0.0, 0.0, 0.5),
                Vec4::new(-1.0, 0.0, 0.0, 0.5),
                Vec4::new(0.0, 1.0, 0.0, 0.5),
                Vec4::new(0.0, -1.0, 0.0, 0.5),
                Vec4::new(0.0, 0.0, 1.0, 0.5),
                Vec4::new(0.0, 0.0, -1.0, 0.5),
            ],
            edge_directions: vec![Vec3::X, Vec3::Y, Vec3::Z],
        });
        let mut index = ConstructionCollisionIndex::new(&creation);
        let config = KinematicCapsuleConfig {
            radius: 0.1,
            standing_height: 0.2,
            ..KinematicCapsuleConfig::default()
        };
        let hit = index
            .cast_capsule(Vec3::new(-2.0, 0.0, 0.0), Vec3::X * 4.0, config)
            .expect("support-mapped sweep reaches convex solid");
        assert!(hit.normal.dot(Vec3::NEG_X) > 0.9, "{hit:?}");
    }

    #[test]
    fn published_body_sweep_pushes_a_stationary_capsule() {
        let creation = cuboid_creation([4, 8, 4], IVec3::new(-8, 4, 0));
        let mut dynamic_creation = creation;
        dynamic_creation.compounds[0].is_static = false;
        dynamic_creation.compounds[0].mass_properties.inverse_mass =
            dynamic_creation.compounds[0].mass_properties.mass.recip();
        let mut index = ConstructionCollisionIndex::new(&dynamic_creation);
        let pose = super::ConstructionBodyState {
            translation: Vec3::new(2.0, 1.0, 0.0),
            rotation: bevy_math::Quat::IDENTITY,
            linear_velocity: Vec3::X * 240.0,
            angular_velocity: Vec3::ZERO,
        };
        assert!(index.refit_dynamic(&[pose]));
        let hit = index
            .cast_capsule(Vec3::ZERO, Vec3::ZERO, KinematicCapsuleConfig::default())
            .expect("body crossing between snapshots reaches stationary capsule");
        assert!(
            hit.time_of_impact > 0.0 && hit.time_of_impact < 1.0,
            "{hit:?}"
        );
        assert!(hit.normal.dot(Vec3::X) > 0.9, "{hit:?}");
    }
}
