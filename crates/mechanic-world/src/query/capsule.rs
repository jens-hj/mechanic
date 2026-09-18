//! The kinematic capsule the player walks as.

use super::scene::{TerrainDensity, raycast_density};
use crate::construction_collision::{
    ConstructionBodyState, ConstructionContact, KinematicCollisionScene, construction_feet,
};
use crate::{TERRAIN_CELL_METERS, WorldPosition};
use bevy_math::{DVec2, DVec3, Quat, Vec3};

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
    /// Peak ballistic jump height for a tap; holding jump adds up to 50%.
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
            gravity: mechanic_core::STANDARD_GRAVITY_M_S2,
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
    /// Whether jump is still held, sustaining lift until release or the apex.
    pub jump_held: bool,
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
    pub previous_pose: ConstructionBodyState,
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
pub struct KinematicTickOutcome {
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

impl Default for KinematicTickOutcome {
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

impl KinematicTickOutcome {
    /// Populated equal-and-opposite reaction rows.
    pub fn reaction_impulses(&self) -> &[KinematicContactReaction] {
        &self.reactions[..self.reaction_count]
    }

    pub(super) fn push_reaction(&mut self, reaction: KinematicContactReaction) {
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
    pub(super) jump_sustained: bool,
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
            jump_sustained: false,
        }
    }

    /// Clears moving-platform state before seating or after collision-scene replacement.
    pub fn clear_support(&mut self) {
        self.support = None;
    }

    /// Advances one fixed tick against terrain and compiled construction.
    #[expect(
        clippy::too_many_lines,
        reason = "terrain preservation and construction response share one ordered solve"
    )]
    pub fn tick<T: TerrainDensity>(
        &mut self,
        scene: &mut KinematicCollisionScene<'_, T>,
        input: KinematicInput,
        delta_seconds: f64,
    ) -> KinematicTickOutcome {
        let mut result = KinematicTickOutcome::default();
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
        if self.grounded || !input.jump_held || self.velocity.y <= 0.0 {
            self.jump_sustained = false;
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
            self.jump_sustained = input.jump_held;
            self.velocity.x += f64::from(support_velocity.x);
            self.velocity.y = f64::from(support_velocity.y)
                + (2.0 * self.config.airborne_gravity * self.config.jump_height).sqrt();
            self.velocity.z += f64::from(support_velocity.z);
            self.grounded = false;
            self.support = None;
        } else if self.grounded {
            self.velocity.y = 0.0;
        } else {
            // With the same launch speed, two-thirds gravity gives 1.5 times
            // the rise. Release permanently ends the boost for this jump.
            let gravity = if self.jump_sustained {
                self.config.airborne_gravity / 1.5
            } else {
                self.config.airborne_gravity
            };
            self.velocity.y -= gravity * delta_seconds;
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

pub(super) fn move_toward(current: DVec2, target: DVec2, maximum_delta: f64) -> DVec2 {
    let delta = target - current;
    if delta.length_squared() <= maximum_delta * maximum_delta {
        target
    } else {
        current + delta.normalize_or_zero() * maximum_delta
    }
}

pub(super) fn world_up_yaw_delta(previous: Quat, current: Quat) -> f32 {
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

pub(super) fn deepest_capsule_penetration(
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

pub(super) fn update_deepest(
    terrain: &impl TerrainDensity,
    point: WorldPosition,
    deepest: &mut Option<(f32, Vec3)>,
) {
    let density = terrain.density(point);
    if density > 0.0 && deepest.is_none_or(|(current, _)| density > current) {
        *deepest = Some((density, density_normal(terrain, point)));
    }
}

pub(super) fn density_normal(terrain: &impl TerrainDensity, position: WorldPosition) -> Vec3 {
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
