//! Density queries, kinematic walking, and terrain-foundation support.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)] // Validated prototype dimensions become grid counts.

use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet, BinaryHeap},
};

use bevy_math::{DVec2, DVec3, Quat, Vec3};
use mechanic_core::PartId;

use crate::{
    BrickCoord, TERRAIN_CELL_METERS, TerrainField, TerrainMaterial, TerrainMeshChunk,
    TerrainNodeId, TerrainOctree, TerrainRayHit, TerrainTransitionMask, WorldPosition,
    construction_collision::{
        ConstructionBodyPose, ConstructionContact, KinematicCollisionScene, construction_feet,
    },
};

/// Read-only density source used by walking, raycasts, and support checks.
pub trait TerrainDensity {
    /// Density at a global point. Positive values are occupied.
    fn density(&self, position: WorldPosition) -> f32;
    /// Material at a global point.
    fn material(&self, position: WorldPosition) -> TerrainMaterial;
}

/// Procedural field with its sparse edits applied.
#[derive(Clone, Copy, Debug)]
pub struct TerrainScene<'a> {
    /// Untouched deterministic terrain.
    pub field: &'a TerrainField,
    /// Promoted edited bricks.
    pub edits: &'a TerrainOctree,
}

impl TerrainDensity for TerrainScene<'_> {
    fn density(&self, position: WorldPosition) -> f32 {
        self.edits.sample_position(self.field, position).density
    }

    fn material(&self, position: WorldPosition) -> TerrainMaterial {
        self.edits.sample_position(self.field, position).material
    }
}

/// Read-only view of the exact active node meshes used by rendering and physics.
#[derive(Clone, Copy, Debug)]
pub struct ActiveTerrainScene<'a> {
    /// Active generation-selected node bundles.
    pub chunks: &'a BTreeMap<TerrainNodeId, TerrainMeshChunk>,
    /// Faces whose adjacent active generation is ready.
    pub ready_faces: &'a BTreeMap<TerrainNodeId, TerrainTransitionMask>,
    /// Incrementally maintained active-node octree index.
    pub spatial_index: &'a TerrainSpatialIndex,
}

impl ActiveTerrainScene<'_> {
    /// Nearest active terrain triangle along a ray.
    pub fn raycast(
        self,
        origin: WorldPosition,
        direction: DVec3,
        maximum_distance: f64,
    ) -> Option<TerrainRayHit> {
        self.spatial_index
            .ray_candidates(origin, direction, maximum_distance)
            .into_iter()
            .filter_map(|id| {
                let chunk = self.chunks.get(&id)?;
                chunk.raycast_sealed(
                    self.ready_faces.get(&id).copied().unwrap_or_default(),
                    origin,
                    direction,
                    maximum_distance,
                )
            })
            .min_by(|first, second| first.distance.total_cmp(&second.distance))
    }

    fn nearest(self, position: WorldPosition) -> Option<(f32, TerrainMaterial)> {
        self.spatial_index.nearest(position, |id| {
            let chunk = self.chunks.get(&id)?;
            let sample = chunk.nearest_sealed(
                self.ready_faces.get(&id).copied().unwrap_or_default(),
                position,
            )?;
            Some(((sample.0, sample.1), sample.2))
        })
    }
}

/// Incrementally maintained sparse octree index of active terrain nodes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TerrainSpatialIndex {
    active: BTreeSet<TerrainNodeId>,
    descendant_counts: BTreeMap<TerrainNodeId, u32>,
}

impl TerrainSpatialIndex {
    /// Inserts an active node and updates its ancestor path.
    pub fn insert(&mut self, id: TerrainNodeId) {
        if !self.active.insert(id) {
            return;
        }
        let mut current = Some(id);
        while let Some(node) = current {
            *self.descendant_counts.entry(node).or_default() += 1;
            current = node.parent();
        }
    }

    /// Removes an active node and prunes empty ancestor paths.
    pub fn remove(&mut self, id: TerrainNodeId) {
        if !self.active.remove(&id) {
            return;
        }
        let mut current = Some(id);
        while let Some(node) = current {
            let remove = if let Some(count) = self.descendant_counts.get_mut(&node) {
                *count = count.saturating_sub(1);
                *count == 0
            } else {
                false
            };
            if remove {
                self.descendant_counts.remove(&node);
            }
            current = node.parent();
        }
    }

    /// True when the exact active node is indexed.
    pub fn contains(&self, id: TerrainNodeId) -> bool {
        self.active.contains(&id)
    }

    fn ray_candidates(
        &self,
        origin: WorldPosition,
        direction: DVec3,
        maximum_distance: f64,
    ) -> Vec<TerrainNodeId> {
        let Some(direction) = direction.try_normalize() else {
            return Vec::new();
        };
        let mut candidates = Vec::new();
        let mut stack = vec![TerrainNodeId::ROOT];
        while let Some(node) = stack.pop() {
            if !self.descendant_counts.contains_key(&node)
                || !ray_intersects_node(origin.0, direction, node, maximum_distance)
            {
                continue;
            }
            if self.active.contains(&node) {
                candidates.push(node);
            } else if let Some(children) = node.children() {
                stack.extend(children);
            }
        }
        candidates
    }

    fn nearest<T>(
        &self,
        position: WorldPosition,
        mut visit: impl FnMut(TerrainNodeId) -> Option<(T, f64)>,
    ) -> Option<T> {
        let mut heap = BinaryHeap::new();
        let mut best_distance = f64::INFINITY;
        let mut nearest = None;
        if self.descendant_counts.contains_key(&TerrainNodeId::ROOT) {
            heap.push(DistanceNode::new(position, TerrainNodeId::ROOT));
        }
        while let Some(entry) = heap.pop() {
            if entry.distance_squared >= best_distance * best_distance {
                break;
            }
            if self.active.contains(&entry.node) {
                if let Some((candidate, distance)) = visit(entry.node)
                    && distance < best_distance
                {
                    best_distance = distance;
                    nearest = Some(candidate);
                }
            } else if let Some(children) = entry.node.children() {
                heap.extend(
                    children
                        .into_iter()
                        .filter(|child| self.descendant_counts.contains_key(child))
                        .map(|child| DistanceNode::new(position, child)),
                );
            }
        }
        nearest
    }
}

#[derive(Clone, Copy, Debug)]
struct DistanceNode {
    distance_squared: f64,
    node: TerrainNodeId,
}

impl DistanceNode {
    fn new(position: WorldPosition, node: TerrainNodeId) -> Self {
        let (minimum, maximum) = node_bounds(node);
        let distance_squared = (0..3)
            .map(|axis| {
                if position.0[axis] < minimum[axis] {
                    minimum[axis] - position.0[axis]
                } else if position.0[axis] > maximum[axis] {
                    position.0[axis] - maximum[axis]
                } else {
                    0.0
                }
            })
            .map(|distance| distance * distance)
            .sum();
        Self {
            distance_squared,
            node,
        }
    }
}

impl PartialEq for DistanceNode {
    fn eq(&self, other: &Self) -> bool {
        self.node == other.node && self.distance_squared == other.distance_squared
    }
}

impl Eq for DistanceNode {}

impl PartialOrd for DistanceNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DistanceNode {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .distance_squared
            .total_cmp(&self.distance_squared)
            .then_with(|| other.node.cmp(&self.node))
    }
}

fn node_bounds(node: TerrainNodeId) -> (DVec3, DVec3) {
    (
        DVec3::from_array(
            node.minimum_cell_i64()
                .map(|cell| cell as f64 * TERRAIN_CELL_METERS),
        ),
        DVec3::from_array(
            node.maximum_cell_exclusive_i64()
                .map(|cell| cell as f64 * TERRAIN_CELL_METERS),
        ),
    )
}

fn ray_intersects_node(
    origin: DVec3,
    direction: DVec3,
    node: TerrainNodeId,
    maximum_distance: f64,
) -> bool {
    let (minimum, maximum) = node_bounds(node);
    let mut near = 0.0_f64;
    let mut far = maximum_distance;
    for axis in 0..3 {
        if direction[axis].abs() <= f64::EPSILON {
            if origin[axis] < minimum[axis] || origin[axis] > maximum[axis] {
                return false;
            }
            continue;
        }
        let inverse = direction[axis].recip();
        let mut first = (minimum[axis] - origin[axis]) * inverse;
        let mut second = (maximum[axis] - origin[axis]) * inverse;
        if first > second {
            core::mem::swap(&mut first, &mut second);
        }
        near = near.max(first);
        far = far.min(second);
        if near > far {
            return false;
        }
    }
    far >= 0.0
}

impl TerrainDensity for ActiveTerrainScene<'_> {
    fn density(&self, position: WorldPosition) -> f32 {
        (*self).nearest(position).map_or(-1.0, |sample| sample.0)
    }

    fn material(&self, position: WorldPosition) -> TerrainMaterial {
        (*self)
            .nearest(position)
            .map_or(TerrainMaterial::Rock, |sample| sample.1)
    }
}

/// Raycasts a density field without depending on a rendered chunk being ready.
pub fn raycast_density(
    terrain: &impl TerrainDensity,
    origin: WorldPosition,
    direction: DVec3,
    maximum_distance: f64,
) -> Option<TerrainRayHit> {
    let direction = direction.try_normalize()?;
    let step = TERRAIN_CELL_METERS * 0.5;
    let mut previous_distance = 0.0;
    let mut previous_density = terrain.density(origin);
    let mut distance = step;
    while distance <= maximum_distance {
        let position = WorldPosition(origin.0 + direction * distance);
        let density = terrain.density(position);
        if (density > 0.0) != (previous_density > 0.0) {
            let mut empty = previous_distance;
            let mut solid = distance;
            if previous_density > 0.0 {
                core::mem::swap(&mut empty, &mut solid);
            }
            for _ in 0..10 {
                let middle = (empty + solid) * 0.5;
                if terrain.density(WorldPosition(origin.0 + direction * middle)) > 0.0 {
                    solid = middle;
                } else {
                    empty = middle;
                }
            }
            let hit_distance = (empty + solid) * 0.5;
            let hit_position = WorldPosition(origin.0 + direction * hit_distance);
            let normal = density_normal(terrain, hit_position);
            let mut material_weights = [0.0; TerrainMaterial::COUNT];
            material_weights[terrain
                .material(WorldPosition(hit_position.0 - DVec3::from(normal) * step))
                .code() as usize] = 1.0;
            return Some(TerrainRayHit {
                position: hit_position,
                normal,
                distance: hit_distance,
                material_weights,
                chunk_generation: 0,
                triangle: 0,
            });
        }
        previous_distance = distance;
        previous_density = density;
        distance += step;
    }
    None
}

/// Prototype kinematic capsule dimensions and movement limits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KinematicCapsuleConfig {
    /// Capsule radius.
    pub radius: f64,
    /// Standing height from feet to head.
    pub standing_height: f64,
    /// Maximum ledge automatically stepped onto.
    pub step_height: f64,
    /// Maximum walkable surface angle in radians.
    pub maximum_slope: f64,
    /// Peak ballistic jump height.
    pub jump_height: f64,
    /// Horizontal walking speed.
    pub walk_speed: f64,
    /// Horizontal sprinting speed.
    pub sprint_speed: f64,
    /// Downward acceleration.
    pub gravity: f64,
    /// Downward acceleration used for the authored jump arc.
    pub airborne_gravity: f64,
    /// Player mass used for equal-and-opposite construction reactions.
    pub mass: f64,
    /// Maximum grounded horizontal acceleration.
    pub ground_acceleration: f64,
    /// Maximum airborne horizontal acceleration.
    pub air_acceleration: f64,
    /// Tangential impulse limit relative to the normal impulse.
    pub traction_coefficient: f64,
    /// Normal velocity retained after impact.
    pub restitution: f64,
}

impl Default for KinematicCapsuleConfig {
    fn default() -> Self {
        Self {
            radius: 0.30,
            standing_height: 1.8,
            step_height: 0.35,
            maximum_slope: 45.0_f64.to_radians(),
            jump_height: 0.75,
            walk_speed: 4.0,
            sprint_speed: 7.0,
            gravity: 9.81,
            airborne_gravity: 24.0,
            mass: 80.0,
            ground_acceleration: 40.0,
            air_acceleration: 10.0,
            traction_coefficient: 0.8,
            restitution: 0.0,
        }
    }
}

/// Input sampled for one fixed 60 Hz controller tick.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct KinematicInput {
    /// Desired horizontal world direction, clamped to unit length.
    pub movement: DVec2,
    /// Whether movement uses sprint speed.
    pub sprint: bool,
    /// True only on the tick a jump is requested.
    pub jump: bool,
}

/// Persistent moving-platform attachment in compound-local coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KinematicSupport {
    /// Supporting compiled compound row.
    pub compound_index: u32,
    /// Supporting compiled collider row.
    pub collider_index: u32,
    /// Contact anchor retained in the body frame.
    pub local_anchor: Vec3,
    /// Pose used to apply the next published support-point delta.
    pub previous_pose: ConstructionBodyPose,
}

/// Equal-and-opposite impulse queued for the next GPU physics tick.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct KinematicContactReaction {
    /// Dynamic compound receiving the reaction.
    pub compound_index: u32,
    /// Contact point in the construction/GPU frame.
    pub world_point: Vec3,
    /// Impulse applied to the construction, opposite the player's response.
    pub impulse: Vec3,
}

/// Maximum distinct contact reactions produced by one controller tick.
pub const MAX_KINEMATIC_REACTIONS: usize = 8;

/// Observable result of one fixed controller tick.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KinematicTickResult {
    /// Fixed-capacity reaction rows; only the prefix ending at `reaction_count` is valid.
    pub reactions: [KinematicContactReaction; MAX_KINEMATIC_REACTIONS],
    /// Number of populated reaction rows.
    pub reaction_count: usize,
    /// World-up yaw inherited from a rotating support.
    pub support_yaw_delta: f32,
    /// Construction contacts resolved during the tick.
    pub resolved_contacts: u32,
    /// Upward construction step taken during this tick.
    pub stepped_height: f32,
}

impl Default for KinematicTickResult {
    fn default() -> Self {
        Self {
            reactions: [KinematicContactReaction::default(); MAX_KINEMATIC_REACTIONS],
            reaction_count: 0,
            support_yaw_delta: 0.0,
            resolved_contacts: 0,
            stepped_height: 0.0,
        }
    }
}

impl KinematicTickResult {
    /// Populated equal-and-opposite reaction rows.
    pub fn reaction_impulses(&self) -> &[KinematicContactReaction] {
        &self.reactions[..self.reaction_count]
    }

    fn push_reaction(&mut self, reaction: KinematicContactReaction) {
        if reaction.impulse.length_squared() <= 1.0e-12 {
            return;
        }
        if let Some(existing) = self.reactions[..self.reaction_count]
            .iter_mut()
            .find(|existing| existing.compound_index == reaction.compound_index)
        {
            existing.world_point = reaction.world_point;
            existing.impulse += reaction.impulse;
        } else if self.reaction_count < MAX_KINEMATIC_REACTIONS {
            self.reactions[self.reaction_count] = reaction;
            self.reaction_count += 1;
        }
    }
}

/// Persistent capsule controller state. Position is the centre of its bottom face.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KinematicCapsule {
    /// Global foot position.
    pub position: WorldPosition,
    /// Continuous global velocity.
    pub velocity: DVec3,
    /// Whether the last tick ended on a walkable surface.
    pub grounded: bool,
    /// Movement limits.
    pub config: KinematicCapsuleConfig,
    /// Moving construction supporting the last tick, if any.
    pub support: Option<KinematicSupport>,
}

impl KinematicCapsule {
    /// Creates a standing controller.
    pub fn new(position: WorldPosition) -> Self {
        Self {
            position,
            velocity: DVec3::ZERO,
            grounded: false,
            config: KinematicCapsuleConfig::default(),
            support: None,
        }
    }

    /// Clears moving-platform state before seating or after collision-scene replacement.
    pub fn clear_support(&mut self) {
        self.support = None;
    }

    /// Advances one fixed tick against terrain and compiled construction.
    #[allow(clippy::too_many_lines)] // Terrain preservation and construction response share one ordered solve.
    pub fn tick<T: TerrainDensity>(
        &mut self,
        scene: &mut KinematicCollisionScene<'_, T>,
        input: KinematicInput,
        delta_seconds: f64,
    ) -> KinematicTickResult {
        let mut result = KinematicTickResult::default();
        let mut support_velocity = Vec3::ZERO;
        if let Some(mut support) = self.support {
            let current_pose = scene
                .construction
                .as_deref()
                .and_then(|index| index.body_pose(support.compound_index));
            if let Some(current_pose) = current_pose {
                let previous_point = support.previous_pose.transform_point(support.local_anchor);
                let current_point = current_pose.transform_point(support.local_anchor);
                self.position.0 += DVec3::from(current_point - previous_point);
                result.support_yaw_delta =
                    world_up_yaw_delta(support.previous_pose.rotation, current_pose.rotation);
                support.previous_pose = current_pose;
                support_velocity = current_pose.velocity_at(current_point);
                self.support = Some(support);
            } else {
                self.support = None;
                self.grounded = false;
            }
        }
        let started_grounded = self.grounded;
        let retained_support = self.support;

        let speed = if input.sprint {
            self.config.sprint_speed
        } else {
            self.config.walk_speed
        };
        let movement = input.movement.clamp_length_max(1.0) * speed;
        let target = movement;
        let horizontal = DVec2::new(self.velocity.x, self.velocity.z);
        let acceleration = if self.grounded {
            self.config.ground_acceleration
        } else {
            self.config.air_acceleration
        };
        let accelerated = move_toward(horizontal, target, acceleration * delta_seconds);
        self.velocity.x = accelerated.x;
        self.velocity.z = accelerated.y;
        let stationary_grounded =
            self.grounded && !input.jump && input.movement.length_squared() <= f64::EPSILON;
        if stationary_grounded {
            let still_supported = self.support.is_some()
                || raycast_density(
                    scene.terrain,
                    self.position,
                    DVec3::NEG_Y,
                    TERRAIN_CELL_METERS,
                )
                .is_some_and(|hit| f64::from(hit.normal.y) >= self.config.maximum_slope.cos());
            if still_supported {
                self.velocity = DVec3::ZERO;
                if self.support.is_none() && scene.construction.is_none() {
                    return result;
                }
            }
            self.grounded = false;
            self.support = None;
        }
        if input.jump && self.grounded {
            self.velocity.x += f64::from(support_velocity.x);
            self.velocity.y = f64::from(support_velocity.y)
                + (2.0 * self.config.airborne_gravity * self.config.jump_height).sqrt();
            self.velocity.z += f64::from(support_velocity.z);
            self.grounded = false;
            self.support = None;
        } else if self.grounded {
            self.velocity.y = 0.0;
        } else {
            self.velocity.y -= self.config.airborne_gravity * delta_seconds;
        }

        let mut displacement = self.velocity * delta_seconds;
        if let Some(index) = scene.construction.as_deref_mut() {
            // A retained support supplies this tick's reference-frame delta and
            // velocity, but must be rediscovered below to survive the tick.
            self.support = None;
            let mut feet = construction_feet(self.position, scene.floating_origin);
            let mut remaining = displacement.as_vec3();
            let mut construction_ground = None;
            for _ in 0..5 {
                let Some(mut contact) = index.cast_capsule(feet, remaining, self.config) else {
                    feet += remaining;
                    break;
                };
                result.resolved_contacts = result.resolved_contacts.saturating_add(1);
                if stationary_grounded && remaining.y < 0.0 {
                    let contact_feet = feet + remaining * contact.time_of_impact;
                    if let Some((_height, normal, surface_height)) = index.walkable_surface_height(
                        contact.collider_index,
                        contact_feet,
                        feet.y - TERRAIN_CELL_METERS as f32 - 1.0e-3,
                        feet.y + 1.0e-3,
                        self.config,
                        None,
                    ) {
                        contact.normal = normal;
                        contact.point.y = surface_height;
                    }
                }
                let walkable = f64::from(contact.normal.y) >= self.config.maximum_slope.cos();
                if started_grounded
                    && !walkable
                    && remaining.x.mul_add(remaining.x, remaining.z * remaining.z) > 0.0
                {
                    let raised = feet + Vec3::Y * self.config.step_height as f32;
                    if index.cast_capsule(raised, remaining, self.config).is_none() {
                        let mut stepped = raised + remaining;
                        let downward = Vec3::NEG_Y * (self.config.step_height as f32 + 1.0e-3);
                        if let Some(mut step_floor) =
                            index.cast_capsule(stepped, downward, self.config)
                            && let Some((height, normal, surface_height)) = index
                                .walkable_surface_height(
                                    step_floor.collider_index,
                                    stepped,
                                    feet.y + 1.0e-4,
                                    raised.y + 1.0e-3,
                                    self.config,
                                    Some(remaining),
                                )
                        {
                            stepped.y = height + 1.0e-4 / normal.y;
                            result.stepped_height =
                                result.stepped_height.max((stepped.y - feet.y).max(0.0));
                            step_floor.normal = normal;
                            step_floor.point.y = surface_height;
                            feet = stepped;
                            construction_ground = Some(step_floor);
                            break;
                        }
                    }
                }
                if contact.penetration > 0.0 {
                    feet += contact.normal * (contact.penetration + 1.0e-4);
                } else {
                    feet += remaining * contact.time_of_impact.max(0.0);
                }
                let body_velocity = index
                    .body_pose(contact.compound_index)
                    .map_or(Vec3::ZERO, |pose| pose.velocity_at(contact.point));
                let player_velocity = self.velocity.as_vec3() + support_velocity;
                let relative_velocity = player_velocity - body_velocity;
                let normal_speed = relative_velocity.dot(contact.normal);
                let inverse_player_mass = (self.config.mass as f32).recip();
                let inverse_body_mass = index.body_effective_inverse_mass(
                    contact.compound_index,
                    contact.point,
                    contact.normal,
                );
                let denominator = inverse_player_mass + inverse_body_mass;
                let impact_impulse = if normal_speed < 0.0 && denominator > 0.0 {
                    -(1.0 + self.config.restitution as f32) * normal_speed / denominator
                } else {
                    0.0
                };
                let weight_impulse = if walkable {
                    self.config.mass as f32
                        * self.config.gravity as f32
                        * delta_seconds as f32
                        * contact.normal.y.max(0.0)
                } else {
                    0.0
                };
                let normal_impulse = impact_impulse.max(weight_impulse);
                let tangent_velocity = relative_velocity - contact.normal * normal_speed;
                let tangent_impulse = if walkable && denominator > 0.0 {
                    let desired = -tangent_velocity / denominator;
                    desired
                        .clamp_length_max(self.config.traction_coefficient as f32 * normal_impulse)
                } else {
                    Vec3::ZERO
                };
                let player_impulse = contact.normal * impact_impulse + tangent_impulse;
                self.velocity += DVec3::from(player_impulse * inverse_player_mass);
                if !index.body_is_static(contact.compound_index) {
                    result.push_reaction(KinematicContactReaction {
                        compound_index: contact.compound_index,
                        world_point: contact.point,
                        impulse: -(contact.normal * normal_impulse + tangent_impulse),
                    });
                }
                let into_surface = self.velocity.as_vec3().dot(contact.normal);
                if into_surface < 0.0 {
                    self.velocity -= DVec3::from(contact.normal * into_surface);
                }
                let unused = 1.0 - contact.time_of_impact.clamp(0.0, 1.0);
                remaining *= unused;
                let into_remaining = remaining.dot(contact.normal);
                if into_remaining < 0.0 {
                    remaining -= contact.normal * into_remaining;
                }
                if walkable && self.velocity.y <= f64::from(body_velocity.y + 0.05) {
                    construction_ground = Some(contact);
                }
                if walkable && stationary_grounded {
                    self.velocity.x = 0.0;
                    self.velocity.z = 0.0;
                    remaining = Vec3::ZERO;
                }
                if remaining.length_squared() <= 1.0e-12 {
                    break;
                }
            }
            displacement = DVec3::ZERO;

            if construction_ground.is_none() && self.velocity.y <= 0.05 {
                construction_ground =
                    index.cast_capsule(feet, Vec3::NEG_Y * TERRAIN_CELL_METERS as f32, self.config);
            }
            if construction_ground.is_none()
                && started_grounded
                && !input.jump
                && let Some(support) = retained_support
                && let Some((height, normal, surface_height)) = index.walkable_surface_height(
                    support.collider_index,
                    feet,
                    feet.y - self.config.step_height as f32 - 1.0e-3,
                    feet.y + self.config.step_height as f32 + 1.0e-3,
                    self.config,
                    None,
                )
            {
                feet.y = height + 1.0e-4 / normal.y;
                construction_ground = Some(ConstructionContact {
                    compound_index: support.compound_index,
                    collider_index: support.collider_index,
                    time_of_impact: 0.0,
                    normal,
                    point: Vec3::new(feet.x, surface_height, feet.z),
                    penetration: 0.0,
                });
            }
            if let Some(mut contact) = construction_ground {
                if let Some((height, normal, surface_height)) = index.walkable_surface_height(
                    contact.collider_index,
                    feet,
                    feet.y - TERRAIN_CELL_METERS as f32 - 1.0e-3,
                    feet.y + 1.0e-3,
                    self.config,
                    None,
                ) {
                    feet.y = height + 1.0e-4 / normal.y;
                    contact.normal = normal;
                    contact.point.y = surface_height;
                    construction_ground = Some(contact);
                } else if f64::from(contact.normal.y) < self.config.maximum_slope.cos() {
                    construction_ground = None;
                }
            }
            if let Some(contact) = construction_ground
                && let Some(pose) = index.body_pose(contact.compound_index)
            {
                self.grounded = true;
                self.velocity.y = 0.0;
                self.support = Some(KinematicSupport {
                    compound_index: contact.compound_index,
                    collider_index: contact.collider_index,
                    local_anchor: pose.inverse_transform_point(contact.point),
                    previous_pose: pose,
                });
                if stationary_grounded {
                    self.velocity.x = 0.0;
                    self.velocity.z = 0.0;
                }
            }
            self.position = WorldPosition(scene.floating_origin + DVec3::from(feet));
        }

        let mut candidate = WorldPosition(self.position.0 + displacement);
        for _ in 0..5 {
            let Some((penetration, normal)) =
                deepest_capsule_penetration(scene.terrain, candidate, self.config)
            else {
                break;
            };
            let walkable = f64::from(normal.y) >= self.config.maximum_slope.cos();
            if !walkable && (displacement.x != 0.0 || displacement.z != 0.0) {
                let stepped = WorldPosition(candidate.0 + DVec3::Y * self.config.step_height);
                if deepest_capsule_penetration(scene.terrain, stepped, self.config).is_none() {
                    candidate = stepped;
                    continue;
                }
            }
            candidate.0 += DVec3::from(normal) * (f64::from(penetration) + 1.0e-4);
            let into_surface = self.velocity.dot(DVec3::from(normal));
            if into_surface < 0.0 {
                self.velocity -= DVec3::from(normal) * into_surface;
            }
        }
        self.position = candidate;

        let construction_grounded = self.support.is_some();
        self.grounded = construction_grounded;
        if self.velocity.y <= 0.05
            && let Some(hit) = raycast_density(
                scene.terrain,
                self.position,
                DVec3::NEG_Y,
                TERRAIN_CELL_METERS,
            )
            && f64::from(hit.normal.y) >= self.config.maximum_slope.cos()
        {
            // Keep the feet just outside the density surface so the next tick does not
            // repeatedly resolve the same contact along a slope's lateral normal.
            self.position = WorldPosition(hit.position.0 + DVec3::Y * 1.0e-4);
            self.velocity.y = 0.0;
            self.grounded = true;
            self.support = None;
        }
        if !self.grounded {
            self.support = None;
        }
        if let Some(index) = scene.construction.as_deref_mut() {
            index.finish_motion_snapshot();
        }
        result
    }
}

fn move_toward(current: DVec2, target: DVec2, maximum_delta: f64) -> DVec2 {
    let delta = target - current;
    if delta.length_squared() <= maximum_delta * maximum_delta {
        target
    } else {
        current + delta.normalize_or_zero() * maximum_delta
    }
}

fn world_up_yaw_delta(previous: Quat, current: Quat) -> f32 {
    let previous_forward = (previous * Vec3::NEG_Z).with_y(0.0).normalize_or_zero();
    let current_forward = (current * Vec3::NEG_Z).with_y(0.0).normalize_or_zero();
    if previous_forward == Vec3::ZERO || current_forward == Vec3::ZERO {
        0.0
    } else {
        previous_forward
            .cross(current_forward)
            .y
            .atan2(previous_forward.dot(current_forward))
    }
}

fn deepest_capsule_penetration(
    terrain: &impl TerrainDensity,
    feet: WorldPosition,
    config: KinematicCapsuleConfig,
) -> Option<(f32, Vec3)> {
    let radius = config.radius;
    let middle_height = config.standing_height * 0.5;
    let upper_height = config.standing_height - radius;
    let radial = [
        DVec3::X,
        DVec3::NEG_X,
        DVec3::Z,
        DVec3::NEG_Z,
        (DVec3::X + DVec3::Z).normalize(),
        (DVec3::X - DVec3::Z).normalize(),
        (-DVec3::X + DVec3::Z).normalize(),
        (-DVec3::X - DVec3::Z).normalize(),
    ];
    let mut deepest = None;
    for offset in [DVec3::ZERO, DVec3::Y * config.standing_height] {
        let point = WorldPosition(feet.0 + offset);
        update_deepest(terrain, point, &mut deepest);
    }
    for height in [radius, middle_height, upper_height] {
        for direction in radial {
            let point = WorldPosition(feet.0 + DVec3::Y * height + direction * radius);
            update_deepest(terrain, point, &mut deepest);
        }
    }
    deepest
}

fn update_deepest(
    terrain: &impl TerrainDensity,
    point: WorldPosition,
    deepest: &mut Option<(f32, Vec3)>,
) {
    let density = terrain.density(point);
    if density > 0.0 && deepest.is_none_or(|(current, _)| density > current) {
        *deepest = Some((density, density_normal(terrain, point)));
    }
}

fn density_normal(terrain: &impl TerrainDensity, position: WorldPosition) -> Vec3 {
    let delta = TERRAIN_CELL_METERS * 0.5;
    let sample = |direction: DVec3| {
        f64::from(terrain.density(WorldPosition(position.0 + direction * delta)))
    };
    let gradient = DVec3::new(
        sample(DVec3::X) - sample(DVec3::NEG_X),
        sample(DVec3::Y) - sample(DVec3::NEG_Y),
        sample(DVec3::Z) - sample(DVec3::NEG_Z),
    );
    (-gradient).as_vec3().normalize_or(Vec3::Y)
}

/// One 5 cm support sample under a terrain-anchored bottom face.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FoundationSample {
    /// Global sample point on the construction's bottom face.
    pub position: WorldPosition,
    /// Whether terrain still reaches this sample.
    pub valid: bool,
}

/// Persistent terrain anchors for one static assembly.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FoundationSupport {
    /// Samples at exact 5 cm spacing.
    pub samples: Vec<FoundationSample>,
}

/// Result of refreshing only the support samples touched by terrain edits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FoundationRefresh {
    /// Samples whose terrain reachability was queried.
    pub sampled: usize,
    /// Previously valid samples that became invalid.
    pub anchors_changed: usize,
    /// No valid terrain anchor remains.
    pub detached: bool,
}

/// Incremental brick-to-foundation lookup used to avoid global anchor sweeps.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FoundationSpatialIndex {
    by_brick: BTreeMap<BrickCoord, BTreeSet<PartId>>,
    by_part: BTreeMap<PartId, BTreeSet<BrickCoord>>,
}

impl FoundationSpatialIndex {
    /// Adds or replaces the indexed footprint for one foundation.
    pub fn insert(&mut self, part: PartId, support: &FoundationSupport) {
        self.remove(part);
        let bricks = support.dependency_bricks();
        for &brick in &bricks {
            self.by_brick.entry(brick).or_default().insert(part);
        }
        if !bricks.is_empty() {
            self.by_part.insert(part, bricks);
        }
    }

    /// Merges a disjoint staged index without recomputing support footprints.
    pub fn append(&mut self, mut other: Self) {
        for (brick, mut parts) in other.by_brick {
            self.by_brick.entry(brick).or_default().append(&mut parts);
        }
        self.by_part.append(&mut other.by_part);
    }

    /// Removes one foundation footprint.
    pub fn remove(&mut self, part: PartId) {
        let Some(bricks) = self.by_part.remove(&part) else {
            return;
        };
        for brick in bricks {
            let remove_entry = self.by_brick.get_mut(&brick).is_some_and(|parts| {
                parts.remove(&part);
                parts.is_empty()
            });
            if remove_entry {
                self.by_brick.remove(&brick);
            }
        }
    }

    /// Foundations whose sample rays overlap at least one changed brick.
    pub fn candidates(&self, changed_bricks: &BTreeSet<BrickCoord>) -> BTreeSet<PartId> {
        changed_bricks
            .iter()
            .filter_map(|brick| self.by_brick.get(brick))
            .flatten()
            .copied()
            .collect()
    }

    /// Number of indexed foundations.
    pub fn len(&self) -> usize {
        self.by_part.len()
    }

    /// True when no foundation is indexed.
    pub fn is_empty(&self) -> bool {
        self.by_part.is_empty()
    }
}

impl FoundationSupport {
    /// Builds support samples for a world-up rectangular bottom face.
    pub fn rectangular(
        terrain: &impl TerrainDensity,
        hit: TerrainRayHit,
        size_x: f64,
        size_z: f64,
    ) -> Self {
        let snapped_x = snap_5_cm(hit.position.0.x);
        let snapped_z = snap_5_cm(hit.position.0.z);
        let count_x = (size_x / TERRAIN_CELL_METERS).round().max(1.0) as i32;
        let count_z = (size_z / TERRAIN_CELL_METERS).round().max(1.0) as i32;
        let minimum_x = snapped_x - (f64::from(count_x - 1) * TERRAIN_CELL_METERS * 0.5);
        let minimum_z = snapped_z - (f64::from(count_z - 1) * TERRAIN_CELL_METERS * 0.5);
        let mut samples = Vec::with_capacity(usize::try_from(count_x * count_z).unwrap_or(0));
        for z in 0..count_z {
            for x in 0..count_x {
                let position = WorldPosition(DVec3::new(
                    minimum_x + f64::from(x) * TERRAIN_CELL_METERS,
                    hit.position.0.y,
                    minimum_z + f64::from(z) * TERRAIN_CELL_METERS,
                ));
                samples.push(FoundationSample {
                    position,
                    valid: terrain_reaches(terrain, position),
                });
            }
        }
        Self { samples }
    }

    /// Refreshes anchors after an edit and returns true when the assembly must detach.
    pub fn refresh(&mut self, terrain: &impl TerrainDensity) -> bool {
        for sample in &mut self.samples {
            if sample.valid {
                sample.valid = terrain_reaches(terrain, sample.position);
            }
        }
        !self.has_valid_anchor()
    }

    /// Refreshes only valid samples whose short support ray overlaps an edited brick.
    pub fn refresh_changed(
        &mut self,
        terrain: &impl TerrainDensity,
        changed_bricks: &BTreeSet<BrickCoord>,
    ) -> FoundationRefresh {
        let mut refresh = FoundationRefresh::default();
        for sample in &mut self.samples {
            if !sample.valid || !sample_overlaps_bricks(sample.position, changed_bricks) {
                continue;
            }
            refresh.sampled += 1;
            sample.valid = terrain_reaches(terrain, sample.position);
            refresh.anchors_changed += usize::from(!sample.valid);
        }
        refresh.detached = !self.has_valid_anchor();
        refresh
    }

    /// Terrain bricks read by this support's short vertical sample rays.
    pub fn dependency_bricks(&self) -> BTreeSet<BrickCoord> {
        self.samples
            .iter()
            .flat_map(|sample| sample_dependency_bricks(sample.position))
            .collect()
    }

    /// True while at least one terrain anchor remains valid.
    pub fn has_valid_anchor(&self) -> bool {
        self.samples.iter().any(|sample| sample.valid)
    }

    /// Number of currently valid support samples.
    pub fn valid_count(&self) -> usize {
        self.samples.iter().filter(|sample| sample.valid).count()
    }

    /// Number of persistent terrain samples in this footprint.
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }
}

fn sample_overlaps_bricks(position: WorldPosition, changed_bricks: &BTreeSet<BrickCoord>) -> bool {
    sample_dependency_bricks(position).any(|brick| changed_bricks.contains(&brick))
}

fn sample_dependency_bricks(position: WorldPosition) -> impl Iterator<Item = BrickCoord> {
    let top = WorldPosition(position.0 + DVec3::Y * TERRAIN_CELL_METERS);
    let bottom = WorldPosition(position.0 - DVec3::Y * TERRAIN_CELL_METERS * 1.1);
    let top = top.cell().ok().map(crate::WorldCell::brick);
    let bottom = bottom.cell().ok().map(crate::WorldCell::brick);
    [top, bottom].into_iter().flatten()
}

fn terrain_reaches(terrain: &impl TerrainDensity, position: WorldPosition) -> bool {
    raycast_density(
        terrain,
        WorldPosition(position.0 + DVec3::Y * TERRAIN_CELL_METERS),
        DVec3::NEG_Y,
        TERRAIN_CELL_METERS * 2.1,
    )
    .is_some()
}

fn snap_5_cm(value: f64) -> f64 {
    (value / TERRAIN_CELL_METERS).round() * TERRAIN_CELL_METERS
}

/// Whether a construction may be edited by the world builder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorldConstructionEditability {
    /// Grounded/static construction accepts placement and deletion.
    GroundedStatic,
    /// Moving creations are deliberately outside this prototype slice.
    MovingBlocked,
}

impl WorldConstructionEditability {
    /// Clear HUD feedback for the temporary limitation.
    pub const fn feedback(self) -> Option<&'static str> {
        match self {
            Self::GroundedStatic => None,
            Self::MovingBlocked => Some("Moving creations cannot be edited in this prototype"),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        cell::Cell,
        collections::{BTreeMap, BTreeSet},
    };

    use bevy_math::{DVec2, DVec3, Vec3};
    use mechanic_core::{
        BuildCommand, BuildOutcome, BuildPose, ConstructionGraph, CuboidSpec, GridRotation,
    };

    use super::{
        ActiveTerrainScene, FoundationSample, FoundationSpatialIndex, FoundationSupport,
        KinematicCapsule, KinematicCollisionScene, KinematicInput, TerrainDensity, TerrainRayHit,
        TerrainSpatialIndex, WorldConstructionEditability, raycast_density,
    };
    use crate::{
        BrickCoord, TerrainMaterial, TerrainMeshRequest, TerrainNodeId, TerrainOctree,
        TerrainTransitionMask, WorldPosition, WorldSeed, mesh_chunk,
    };

    struct Plane;

    impl TerrainDensity for Plane {
        fn density(&self, position: WorldPosition) -> f32 {
            (-position.0.y) as f32
        }

        fn material(&self, _position: WorldPosition) -> TerrainMaterial {
            TerrainMaterial::Soil
        }
    }

    struct Slope;

    impl TerrainDensity for Slope {
        fn density(&self, position: WorldPosition) -> f32 {
            (position.0.x * 0.5 - position.0.y) as f32
        }

        fn material(&self, _position: WorldPosition) -> TerrainMaterial {
            TerrainMaterial::Soil
        }
    }

    fn tick_terrain(
        capsule: &mut KinematicCapsule,
        terrain: &impl TerrainDensity,
        input: KinematicInput,
    ) {
        let mut scene = KinematicCollisionScene {
            terrain,
            construction: None,
            floating_origin: DVec3::ZERO,
        };
        capsule.tick(&mut scene, input, 1.0 / 60.0);
    }

    struct CountingPlane(Cell<usize>);

    struct Empty;

    impl TerrainDensity for Empty {
        fn density(&self, _position: WorldPosition) -> f32 {
            -1.0
        }

        fn material(&self, _position: WorldPosition) -> TerrainMaterial {
            TerrainMaterial::Rock
        }
    }

    impl TerrainDensity for CountingPlane {
        fn density(&self, position: WorldPosition) -> f32 {
            self.0.set(self.0.get() + 1);
            (-position.0.y) as f32
        }

        fn material(&self, _position: WorldPosition) -> TerrainMaterial {
            TerrainMaterial::Soil
        }
    }

    #[test]
    fn changed_bricks_refresh_only_intersecting_foundation_samples() {
        let near = WorldPosition(DVec3::new(0.1, 0.0, 0.1));
        let far = WorldPosition(DVec3::new(4.0, 0.0, 0.1));
        let mut support = FoundationSupport {
            samples: vec![
                FoundationSample {
                    position: near,
                    valid: true,
                },
                FoundationSample {
                    position: far,
                    valid: true,
                },
            ],
        };
        let changed = BTreeSet::from([near.cell().unwrap().brick()]);
        let terrain = CountingPlane(Cell::new(0));

        let refresh = support.refresh_changed(&terrain, &changed);

        assert_eq!(refresh.sampled, 1);
        assert!(terrain.0.get() > 0);
        assert!(support.samples[1].valid);
    }

    #[test]
    fn brick_index_returns_only_overlapping_foundations() {
        let mut graph = ConstructionGraph::default();
        let BuildOutcome::Spawned(near_part) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new([1, 1, 1], BuildPose::default()).unwrap(),
            ))
            .unwrap()
        else {
            panic!("spawn reports a part");
        };
        let BuildOutcome::Spawned(far_part) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [1, 1, 1],
                    BuildPose::from_position_ticks(
                        bevy_math::IVec3::new(1600, 0, 0),
                        GridRotation::default(),
                    ),
                )
                .unwrap(),
            ))
            .unwrap()
        else {
            panic!("spawn reports a part");
        };
        let near_position = WorldPosition(DVec3::new(0.1, 0.0, 0.1));
        let far_position = WorldPosition(DVec3::new(4.0, 0.0, 0.1));
        let near_support = FoundationSupport {
            samples: vec![FoundationSample {
                position: near_position,
                valid: true,
            }],
        };
        let far_support = FoundationSupport {
            samples: vec![FoundationSample {
                position: far_position,
                valid: true,
            }],
        };
        let mut index = FoundationSpatialIndex::default();
        index.insert(near_part, &near_support);
        let mut staged = FoundationSpatialIndex::default();
        staged.insert(far_part, &far_support);
        index.append(staged);

        assert_eq!(
            index.candidates(&BTreeSet::from([near_position.cell().unwrap().brick()])),
            BTreeSet::from([near_part])
        );
        index.remove(near_part);
        assert!(
            index
                .candidates(&BTreeSet::from([near_position.cell().unwrap().brick()]))
                .is_empty()
        );
    }

    #[test]
    fn capsule_lands_and_jumps_to_requested_height() {
        let mut capsule = KinematicCapsule::new(WorldPosition(DVec3::new(0.0, 1.0, 0.0)));
        for _ in 0..120 {
            tick_terrain(&mut capsule, &Plane, KinematicInput::default());
        }
        assert!(capsule.grounded);
        assert!(capsule.position.0.y.abs() < 0.02);
        tick_terrain(
            &mut capsule,
            &Plane,
            KinematicInput {
                movement: DVec2::ZERO,
                sprint: false,
                jump: true,
            },
        );
        let mut peak = capsule.position.0.y;
        let mut airborne_ticks = 1;
        for _ in 0..120 {
            tick_terrain(&mut capsule, &Plane, KinematicInput::default());
            peak = peak.max(capsule.position.0.y);
            airborne_ticks += 1;
            if capsule.grounded {
                break;
            }
        }
        assert!(
            (peak - capsule.config.jump_height).abs() < 0.08,
            "peak was {peak}"
        );
        assert!(
            (28..=32).contains(&airborne_ticks),
            "jump lasted {airborne_ticks} ticks"
        );
    }

    #[test]
    fn grounded_capsule_stays_still_on_a_walkable_slope() {
        let mut capsule = KinematicCapsule::new(WorldPosition(DVec3::new(0.0, 0.1, 0.0)));
        for _ in 0..60 {
            tick_terrain(&mut capsule, &Slope, KinematicInput::default());
        }
        assert!(capsule.grounded);
        let settled = capsule.position;

        for _ in 0..120 {
            tick_terrain(&mut capsule, &Slope, KinematicInput::default());
        }

        assert!(
            capsule.position.0.abs_diff_eq(settled.0, 1.0e-6),
            "stationary capsule drifted from {settled:?} to {:?}",
            capsule.position
        );
    }

    #[test]
    fn grounded_capsule_stays_still_on_a_walkable_construction_slope() {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(part) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new([16, 1, 16], BuildPose::default()).unwrap(),
            ))
            .unwrap()
        else {
            panic!("cuboid spawn returns its part");
        };
        let mut creation = graph.compile_with_static_parts([part]).unwrap();
        let mechanic_core::ColliderShape::Cuboid { local_rotation, .. } =
            &mut creation.colliders[0].shape
        else {
            panic!("unshaped cuboid keeps its fast-path collider");
        };
        *local_rotation = bevy_math::Quat::from_rotation_z(30.0_f32.to_radians());
        let mut index = crate::ConstructionCollisionIndex::new(&creation);
        let mut capsule = KinematicCapsule::new(WorldPosition(DVec3::Y));
        for _ in 0..120 {
            let mut scene = KinematicCollisionScene {
                terrain: &Empty,
                construction: Some(&mut index),
                floating_origin: DVec3::ZERO,
            };
            capsule.tick(&mut scene, KinematicInput::default(), 1.0 / 60.0);
        }
        assert!(capsule.grounded);
        let settled = capsule.position;

        for _ in 0..120 {
            let mut scene = KinematicCollisionScene {
                terrain: &Empty,
                construction: Some(&mut index),
                floating_origin: DVec3::ZERO,
            };
            capsule.tick(&mut scene, KinematicInput::default(), 1.0 / 60.0);
        }

        assert!(
            capsule.position.0.abs_diff_eq(settled.0, 1.0e-5),
            "stationary capsule drifted from {settled:?} to {:?}",
            capsule.position
        );
    }

    #[test]
    fn grounded_capsule_does_not_slide_from_a_flat_construction_edge() {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(part) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new([8, 1, 8], BuildPose::default()).unwrap(),
            ))
            .unwrap()
        else {
            panic!("cuboid spawn returns its part");
        };
        let creation = graph.compile_with_static_parts([part]).unwrap();
        let mut index = crate::ConstructionCollisionIndex::new(&creation);
        let mut capsule = KinematicCapsule::new(WorldPosition(DVec3::Y));
        for _ in 0..120 {
            let mut scene = KinematicCollisionScene {
                terrain: &Empty,
                construction: Some(&mut index),
                floating_origin: DVec3::ZERO,
            };
            capsule.tick(&mut scene, KinematicInput::default(), 1.0 / 60.0);
        }
        assert!(capsule.grounded);
        capsule.position.0.x = 1.05;
        let edge = capsule.position;

        for _ in 0..120 {
            let mut scene = KinematicCollisionScene {
                terrain: &Empty,
                construction: Some(&mut index),
                floating_origin: DVec3::ZERO,
            };
            capsule.tick(&mut scene, KinematicInput::default(), 1.0 / 60.0);
        }

        assert!(capsule.grounded);
        assert!(
            (capsule.position.0.x - edge.0.x).abs() < 1.0e-3
                && (capsule.position.0.z - edge.0.z).abs() < 1.0e-3
                && (capsule.position.0.y - edge.0.y).abs() < 0.01,
            "stationary capsule drifted from {edge:?} to {:?}",
            capsule.position
        );
    }

    #[test]
    fn vertical_construction_contact_does_not_slow_a_fall() {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(part) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [1, 8, 8],
                    BuildPose::new(bevy_math::IVec3::new(4, 4, 0), GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap()
        else {
            panic!("cuboid spawn returns its part");
        };
        let creation = graph.compile_with_static_parts([part]).unwrap();
        let mut index = crate::ConstructionCollisionIndex::new(&creation);
        let mut against_wall = KinematicCapsule::new(WorldPosition(DVec3::new(0.57, 0.1, 0.0)));
        against_wall.velocity = DVec3::new(1.0, -2.0, 0.0);
        let mut free_fall = against_wall;
        let input = KinematicInput {
            movement: DVec2::X,
            sprint: false,
            jump: false,
        };
        let mut wall_scene = KinematicCollisionScene {
            terrain: &Empty,
            construction: Some(&mut index),
            floating_origin: DVec3::ZERO,
        };
        let result = against_wall.tick(&mut wall_scene, input, 1.0 / 60.0);
        let mut empty_scene = KinematicCollisionScene {
            terrain: &Empty,
            construction: None,
            floating_origin: DVec3::ZERO,
        };
        free_fall.tick(&mut empty_scene, input, 1.0 / 60.0);

        assert!(
            result.resolved_contacts > 0,
            "the capsule should reach the wall"
        );
        assert!(
            (against_wall.velocity.y - free_fall.velocity.y).abs() < 1.0e-6,
            "wall contact slowed the fall: {} instead of {}",
            against_wall.velocity.y,
            free_fall.velocity.y,
        );
    }

    #[test]
    fn sprint_uses_configured_faster_speed() {
        let mut walking = KinematicCapsule::new(WorldPosition(DVec3::ZERO));
        walking.grounded = true;
        let mut sprinting = walking;

        for _ in 0..20 {
            tick_terrain(
                &mut walking,
                &Plane,
                KinematicInput {
                    movement: DVec2::X,
                    sprint: false,
                    jump: false,
                },
            );
            tick_terrain(
                &mut sprinting,
                &Plane,
                KinematicInput {
                    movement: DVec2::X,
                    sprint: true,
                    jump: false,
                },
            );
        }

        assert!((walking.velocity.x - walking.config.walk_speed).abs() < f64::EPSILON);
        assert!((sprinting.velocity.x - sprinting.config.sprint_speed).abs() < f64::EPSILON);
        assert!(sprinting.position.0.x > walking.position.0.x);
    }

    #[test]
    fn dynamic_construction_supports_and_receives_opposite_player_impulses() {
        let mut graph = ConstructionGraph::new();
        graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new([8, 1, 8], BuildPose::default()).unwrap(),
            ))
            .unwrap();
        let creation = graph.compile().unwrap();
        let mut index = crate::ConstructionCollisionIndex::new(&creation);
        let mut capsule = KinematicCapsule::new(WorldPosition(DVec3::Y));
        let mut landing_reaction = None;
        for _ in 0..120 {
            let mut scene = KinematicCollisionScene {
                terrain: &Empty,
                construction: Some(&mut index),
                floating_origin: DVec3::ZERO,
            };
            let result = capsule.tick(&mut scene, KinematicInput::default(), 1.0 / 60.0);
            landing_reaction = result
                .reaction_impulses()
                .first()
                .copied()
                .or(landing_reaction);
        }
        assert!(capsule.grounded);
        assert_eq!(
            capsule.support.map(|support| support.compound_index),
            Some(0)
        );
        let reaction = landing_reaction.expect("dynamic floor receives player reaction");
        assert!(reaction.impulse.y < 0.0, "{reaction:?}");

        let mut scene = KinematicCollisionScene {
            terrain: &Empty,
            construction: Some(&mut index),
            floating_origin: DVec3::ZERO,
        };
        let load = capsule.tick(&mut scene, KinematicInput::default(), 1.0 / 60.0);
        assert!(
            load.reaction_impulses()
                .iter()
                .any(|reaction| reaction.impulse.y < 0.0),
            "a stationary player must load a dynamic platform"
        );
    }

    #[test]
    fn moving_support_carries_position_yaw_and_jump_velocity() {
        let mut graph = ConstructionGraph::new();
        graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new([16, 1, 16], BuildPose::default()).unwrap(),
            ))
            .unwrap();
        let creation = graph.compile().unwrap();
        let mut index = crate::ConstructionCollisionIndex::new(&creation);
        let mut capsule = KinematicCapsule::new(WorldPosition(DVec3::new(0.5, 1.0, 0.0)));
        for _ in 0..120 {
            let mut scene = KinematicCollisionScene {
                terrain: &Empty,
                construction: Some(&mut index),
                floating_origin: DVec3::ZERO,
            };
            capsule.tick(&mut scene, KinematicInput::default(), 1.0 / 60.0);
        }
        assert!(capsule.grounded);
        let before = capsule.position;
        let pose = crate::ConstructionBodyPose {
            translation: Vec3::X,
            rotation: bevy_math::Quat::from_rotation_y(core::f32::consts::FRAC_PI_2),
            linear_velocity: Vec3::new(2.0, 0.0, 0.0),
            angular_velocity: Vec3::Y,
        };
        assert!(index.refit_dynamic(&[pose]));
        let mut scene = KinematicCollisionScene {
            terrain: &Empty,
            construction: Some(&mut index),
            floating_origin: DVec3::ZERO,
        };
        let carried = capsule.tick(&mut scene, KinematicInput::default(), 1.0 / 60.0);
        assert_ne!(capsule.position, before);
        assert!((carried.support_yaw_delta - core::f32::consts::FRAC_PI_2).abs() < 1.0e-4);
        let inherited_x = capsule
            .support
            .map(|support| {
                pose.velocity_at(pose.transform_point(support.local_anchor))
                    .x
            })
            .unwrap();

        let mut scene = KinematicCollisionScene {
            terrain: &Empty,
            construction: Some(&mut index),
            floating_origin: DVec3::ZERO,
        };
        capsule.tick(
            &mut scene,
            KinematicInput {
                movement: DVec2::ZERO,
                sprint: false,
                jump: true,
            },
            1.0 / 60.0,
        );
        assert!(!capsule.grounded);
        assert!(
            (capsule.velocity.x - f64::from(inherited_x)).abs() < 0.05,
            "{:?}",
            capsule.velocity
        );
    }

    #[test]
    fn construction_step_climbs_only_to_the_compiled_top_face() {
        let mut graph = ConstructionGraph::new();
        let parts = graph
            .apply_batch([
                BuildCommand::Spawn(CuboidSpec::new([16, 1, 16], BuildPose::default()).unwrap()),
                BuildCommand::Spawn(
                    CuboidSpec::new(
                        [4, 1, 8],
                        BuildPose::new(bevy_math::IVec3::new(4, 1, 0), GridRotation::default()),
                    )
                    .unwrap(),
                ),
            ])
            .unwrap()
            .into_iter()
            .filter_map(|outcome| match outcome {
                BuildOutcome::Spawned(part) => Some(part),
                _ => None,
            })
            .collect::<Vec<_>>();
        let creation = graph.compile_with_static_parts(parts).unwrap();
        let step_top = creation
            .colliders
            .iter()
            .filter_map(|collider| match collider.shape {
                mechanic_core::ColliderShape::Cuboid { half_extents, .. } => Some(
                    creation.compounds[collider.compound_index as usize]
                        .root_translation
                        .y
                        + collider.local_center.y
                        + half_extents.y,
                ),
                mechanic_core::ColliderShape::Convex(_) => None,
            })
            .fold(f32::NEG_INFINITY, f32::max);
        let mut index = crate::ConstructionCollisionIndex::new(&creation);
        let mut capsule = KinematicCapsule::new(WorldPosition(DVec3::new(-0.5, 0.125, 0.0)));
        capsule.grounded = true;
        let mut reported_step = 0.0_f32;
        let mut maximum_height = capsule.position.0.y;
        for _ in 0..25 {
            let before = capsule.position;
            let mut scene = KinematicCollisionScene {
                terrain: &Empty,
                construction: Some(&mut index),
                floating_origin: DVec3::ZERO,
            };
            let result = capsule.tick(
                &mut scene,
                KinematicInput {
                    movement: DVec2::X,
                    sprint: false,
                    jump: false,
                },
                1.0 / 60.0,
            );
            reported_step = reported_step.max(result.stepped_height);
            maximum_height = maximum_height.max(capsule.position.0.y);
            if result.stepped_height > 0.0 {
                assert!(
                    (capsule.position.0.y - before.0.y - f64::from(result.stepped_height)).abs()
                        < 1.0e-5
                );
            }
        }
        for _ in 0..30 {
            let mut scene = KinematicCollisionScene {
                terrain: &Empty,
                construction: Some(&mut index),
                floating_origin: DVec3::ZERO,
            };
            capsule.tick(&mut scene, KinematicInput::default(), 1.0 / 60.0);
        }
        assert!(capsule.position.0.x > 0.5, "{:?}", capsule.position);
        assert!(reported_step > 0.0, "the ledge should use the step path");
        assert!(
            maximum_height <= f64::from(step_top) + 1.0e-3,
            "step overshot its top: {maximum_height} > {step_top}",
        );
        assert!(
            (capsule.position.0.y - f64::from(step_top)).abs() < 0.02,
            "{:?}, step top {step_top}",
            capsule.position,
        );
        assert!(capsule.grounded);
    }

    #[test]
    fn construction_step_can_be_climbed_at_an_angle() {
        let mut graph = ConstructionGraph::new();
        let parts = graph
            .apply_batch([
                BuildCommand::Spawn(CuboidSpec::new([16, 1, 16], BuildPose::default()).unwrap()),
                BuildCommand::Spawn(
                    CuboidSpec::new(
                        [4, 1, 8],
                        BuildPose::new(bevy_math::IVec3::new(4, 1, 0), GridRotation::default()),
                    )
                    .unwrap(),
                ),
            ])
            .unwrap()
            .into_iter()
            .filter_map(|outcome| match outcome {
                BuildOutcome::Spawned(part) => Some(part),
                _ => None,
            })
            .collect::<Vec<_>>();
        let creation = graph.compile_with_static_parts(parts).unwrap();
        let mut index = crate::ConstructionCollisionIndex::new(&creation);
        let direction = DVec2::new(1.0, 1.0).normalize();
        let mut capsule = KinematicCapsule::new(WorldPosition(DVec3::new(0.19, 0.125, -0.5)));
        capsule.grounded = true;
        capsule.velocity = DVec3::new(direction.x, 0.0, direction.y) * capsule.config.walk_speed;
        let mut scene = KinematicCollisionScene {
            terrain: &Empty,
            construction: Some(&mut index),
            floating_origin: DVec3::ZERO,
        };
        let result = capsule.tick(
            &mut scene,
            KinematicInput {
                movement: direction,
                sprint: false,
                jump: false,
            },
            1.0 / 60.0,
        );

        assert!(
            result.stepped_height > 0.0,
            "the angled approach should use the step path: {:?}",
            capsule.position
        );
        assert!(capsule.position.0.y > 0.3, "{:?}", capsule.position);
        assert!(capsule.grounded);
    }

    #[test]
    fn grazing_a_step_edge_does_not_repeat_step_and_fall() {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(part) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [4, 1, 8],
                    BuildPose::new(bevy_math::IVec3::new(4, 0, 0), GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap()
        else {
            panic!("cuboid spawn returns its part");
        };
        let creation = graph.compile_with_static_parts([part]).unwrap();
        let mut index = crate::ConstructionCollisionIndex::new(&creation);
        let mut capsule = KinematicCapsule::new(WorldPosition(DVec3::new(-0.5, 0.0, 1.01)));
        capsule.grounded = true;
        let mut step_count = 0;
        let mut largest_rise = 0.0_f64;
        for _ in 0..120 {
            let previous_height = capsule.position.0.y;
            let mut scene = KinematicCollisionScene {
                terrain: &Plane,
                construction: Some(&mut index),
                floating_origin: DVec3::ZERO,
            };
            let result = capsule.tick(
                &mut scene,
                KinematicInput {
                    movement: DVec2::X,
                    sprint: false,
                    jump: false,
                },
                1.0 / 60.0,
            );
            step_count += usize::from(result.stepped_height > 0.0);
            largest_rise = largest_rise.max(capsule.position.0.y - previous_height);
        }

        assert_eq!(step_count, 0, "a grazing path must stay beside the step");
        assert!(largest_rise < 0.04, "grazing path jumped by {largest_rise}");
    }

    #[test]
    fn foundation_detaches_only_after_final_anchor_is_lost() {
        let hit = raycast_density(
            &Plane,
            WorldPosition(DVec3::new(0.0, 1.0, 0.0)),
            DVec3::NEG_Y,
            2.0,
        )
        .unwrap();
        let mut support = FoundationSupport::rectangular(&Plane, hit, 0.25, 0.25);
        assert_eq!(support.valid_count(), 25);
        assert!(!support.refresh(&Plane));
        assert!(support.has_valid_anchor());
    }

    #[test]
    fn one_block_foundation_samples_twenty_five_terrain_points() {
        let terrain = CountingPlane(Cell::new(0));
        let support = FoundationSupport::rectangular(
            &terrain,
            TerrainRayHit {
                position: WorldPosition(DVec3::ZERO),
                normal: Vec3::Y,
                distance: 0.0,
                material_weights: [0.0; TerrainMaterial::COUNT],
                chunk_generation: 0,
                triangle: 0,
            },
            0.25,
            0.25,
        );

        assert_eq!(support.sample_count(), 25);
        assert_eq!(terrain.0.get(), 500);
    }

    #[test]
    fn active_octree_and_chunk_bvhs_match_direct_chunk_raycast() {
        let field = crate::TerrainField::new(WorldSeed(9));
        let terrain = TerrainOctree::default().snapshot();
        let nodes = [BrickCoord::new(-1, 2, -1), BrickCoord::new(0, 2, -1)];
        let mut chunks = BTreeMap::new();
        let mut ready = BTreeMap::new();
        let mut index = TerrainSpatialIndex::default();
        for coordinate in nodes {
            let id = TerrainNodeId::leaf(coordinate);
            let chunk = mesh_chunk(
                &field,
                &terrain,
                TerrainMeshRequest {
                    node: id,
                    generation: 3,
                    transition_mask: TerrainTransitionMask::NONE,
                },
            );
            chunks.insert(id, chunk);
            ready.insert(id, TerrainTransitionMask::NONE);
            index.insert(id);
        }
        let origin = WorldPosition(DVec3::new(-0.8, 10.0, -0.8));
        let direct = chunks
            .values()
            .filter_map(|chunk| {
                chunk.raycast_sealed(TerrainTransitionMask::NONE, origin, DVec3::NEG_Y, 20.0)
            })
            .min_by(|first, second| first.distance.total_cmp(&second.distance))
            .unwrap();
        let accelerated = ActiveTerrainScene {
            chunks: &chunks,
            ready_faces: &ready,
            spatial_index: &index,
        }
        .raycast(origin, DVec3::NEG_Y, 20.0)
        .unwrap();
        assert!((accelerated.distance - direct.distance).abs() < 1.0e-9);

        index.remove(TerrainNodeId::leaf(nodes[0]));
        assert!(!index.contains(TerrainNodeId::leaf(nodes[0])));
    }

    #[test]
    fn moving_creation_edit_has_clear_feedback() {
        assert!(
            WorldConstructionEditability::GroundedStatic
                .feedback()
                .is_none()
        );
        assert_eq!(
            WorldConstructionEditability::MovingBlocked.feedback(),
            Some("Moving creations cannot be edited in this prototype")
        );
    }
}
