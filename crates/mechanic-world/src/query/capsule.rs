//! The kinematic capsule the player walks as.

mod terrain;

use terrain::support_height;

use super::scene::TerrainDensity;
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
    /// Crouched height from feet to head.
    pub crouch_height: f64,
    /// Rate the capsule grows and shrinks between the two heights.
    pub crouch_transition_speed: f64,
    /// Maximum ledge automatically stepped onto.
    pub step_height: f64,
    /// Maximum walkable surface angle in radians.
    pub maximum_slope: f64,
    /// Steepest lower-capsule contact that permits a recovery jump, in radians.
    pub maximum_jump_slope: f64,
    /// Seconds a jump remains available after losing ground contact.
    pub jump_grace_seconds: f64,
    /// Seconds a jump press waits for an eligible ground contact.
    pub jump_buffer_seconds: f64,
    /// Peak ballistic jump height for a tap; holding jump adds up to 50%.
    pub jump_height: f64,
    /// Horizontal walking speed.
    pub walk_speed: f64,
    /// Horizontal sprinting speed.
    pub sprint_speed: f64,
    /// Horizontal crouching speed.
    pub crouch_speed: f64,
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
            crouch_height: 1.0,
            crouch_transition_speed: 6.0,
            step_height: 0.35,
            maximum_slope: 60.0_f64.to_radians(),
            maximum_jump_slope: 80.0_f64.to_radians(),
            jump_grace_seconds: 0.10,
            jump_buffer_seconds: 0.15,
            jump_height: 0.75,
            walk_speed: 4.0,
            sprint_speed: 7.0,
            crouch_speed: 1.8,
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
#[expect(
    clippy::struct_excessive_bools,
    reason = "one field per held movement key sampled for the tick"
)]
pub struct KinematicInput {
    /// Desired horizontal world direction, clamped to unit length.
    pub movement: DVec2,
    /// Whether movement uses sprint speed.
    pub sprint: bool,
    /// Whether the capsule should shrink to its crouched height.
    pub crouch: bool,
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

/// Height difference below which the capsule counts as fully standing or crouched.
const HEIGHT_EPSILON: f64 = 1.0e-6;

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
    /// Current height from feet to head, between the crouched and standing heights.
    pub height: f64,
    /// Movement limits.
    pub config: KinematicCapsuleConfig,
    /// Moving construction supporting the last tick, if any.
    pub support: Option<KinematicSupport>,
    pub(super) jump_sustained: bool,
    jump_grace_remaining: f64,
    jump_buffer_remaining: f64,
    jump_in_flight: bool,
    jump_support_velocity: Vec3,
}

impl KinematicCapsule {
    /// Creates a standing controller.
    pub fn new(position: WorldPosition) -> Self {
        let config = KinematicCapsuleConfig::default();
        Self {
            position,
            velocity: DVec3::ZERO,
            grounded: false,
            height: config.standing_height,
            config,
            support: None,
            jump_sustained: false,
            jump_grace_remaining: 0.0,
            jump_buffer_remaining: 0.0,
            jump_in_flight: false,
            jump_support_velocity: Vec3::ZERO,
        }
    }

    /// Clears moving-platform state before seating or after collision-scene replacement.
    pub fn clear_support(&mut self) {
        self.support = None;
    }

    /// Clears motion and pending jumps when seating, teleporting, or leaving walking mode.
    pub fn reset_motion(&mut self) {
        self.velocity = DVec3::ZERO;
        self.grounded = false;
        self.support = None;
        self.jump_sustained = false;
        self.jump_in_flight = false;
        self.jump_grace_remaining = 0.0;
        self.jump_buffer_remaining = 0.0;
        self.jump_support_velocity = Vec3::ZERO;
    }

    fn try_jump(&mut self, held: bool, support_velocity: Vec3) -> bool {
        if self.jump_buffer_remaining <= 0.0 || self.jump_grace_remaining <= 0.0 {
            return false;
        }
        self.jump_buffer_remaining = 0.0;
        self.jump_grace_remaining = 0.0;
        self.jump_sustained = held;
        self.jump_in_flight = true;
        self.velocity.x += f64::from(support_velocity.x);
        self.velocity.y = f64::from(support_velocity.y)
            + (2.0 * self.config.airborne_gravity * self.config.jump_height).sqrt();
        self.velocity.z += f64::from(support_velocity.z);
        self.grounded = false;
        self.support = None;
        true
    }

    /// How far the capsule is crouched, `0.0` standing and `1.0` fully crouched.
    pub fn crouch_fraction(&self) -> f64 {
        let range = self.config.standing_height - self.config.crouch_height;
        if range <= 0.0 {
            return 0.0;
        }
        ((self.config.standing_height - self.height) / range).clamp(0.0, 1.0)
    }

    /// Whether the capsule is anywhere below its standing height.
    pub fn crouched(&self) -> bool {
        self.height < self.config.standing_height - HEIGHT_EPSILON
    }

    /// Movement limits whose `standing_height` is the capsule's current height, so
    /// every swept query uses the crouched profile.
    fn collision_config(&self) -> KinematicCapsuleConfig {
        KinematicCapsuleConfig {
            standing_height: self.height,
            ..self.config
        }
    }

    /// Grows or shrinks toward the height `input` asks for. Standing back up waits
    /// until the volume the head would sweep into is clear.
    fn resolve_height<T: TerrainDensity>(
        &mut self,
        scene: &mut KinematicCollisionScene<'_, T>,
        crouch: bool,
        delta_seconds: f64,
    ) {
        let target = if crouch {
            self.config.crouch_height
        } else {
            self.config.standing_height
        };
        if (target - self.height).abs() <= HEIGHT_EPSILON {
            self.height = target;
            return;
        }
        let step = self.config.crouch_transition_speed * delta_seconds;
        let next = if target > self.height {
            (self.height + step).min(target)
        } else {
            (self.height - step).max(target)
        };
        if next > self.height && !self.head_fits_at(scene, next) {
            return;
        }
        self.height = next;
    }

    /// Whether the head sphere clears terrain and construction at `height`.
    fn head_fits_at<T: TerrainDensity>(
        &self,
        scene: &mut KinematicCollisionScene<'_, T>,
        height: f64,
    ) -> bool {
        let radius = self.config.radius;
        let head = KinematicCapsuleConfig {
            standing_height: radius * 2.0,
            ..self.config
        };
        let feet = WorldPosition(self.position.0 + DVec3::Y * (height - radius * 2.0));
        if deepest_capsule_penetration(scene.terrain, feet, head).is_some() {
            return false;
        }
        let floating_origin = scene.floating_origin;
        scene.construction.as_deref_mut().is_none_or(|index| {
            index
                .cast_capsule(construction_feet(feet, floating_origin), Vec3::ZERO, head)
                .is_none()
        })
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
        self.jump_buffer_remaining = (self.jump_buffer_remaining - delta_seconds).max(0.0);
        self.jump_grace_remaining = (self.jump_grace_remaining - delta_seconds).max(0.0);
        if input.jump {
            self.jump_buffer_remaining = self.config.jump_buffer_seconds.max(delta_seconds);
        }
        if self.grounded && !self.jump_in_flight {
            self.jump_grace_remaining = self.config.jump_grace_seconds.max(delta_seconds);
            self.jump_support_velocity = support_velocity;
        }
        let started_grounded = self.grounded;
        let retained_support = self.support;

        self.resolve_height(scene, input.crouch, delta_seconds);
        let collision_config = self.collision_config();

        let speed = if self.crouched() {
            self.config.crouch_speed
        } else if input.sprint {
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
        let jumped = self.try_jump(input.jump_held, self.jump_support_velocity);
        let stationary_grounded = self.grounded && input.movement.length_squared() <= f64::EPSILON;
        if stationary_grounded {
            let still_supported = self.support.is_some()
                || support_height(
                    scene.terrain,
                    self.position,
                    collision_config,
                    TERRAIN_CELL_METERS,
                    TERRAIN_CELL_METERS,
                )
                .is_some();
            if still_supported {
                self.velocity = DVec3::ZERO;
                if self.support.is_none() && scene.construction.is_none() {
                    return result;
                }
            }
            // Rediscover construction contact so standing still still transmits weight.
            self.grounded = false;
            self.support = None;
        }
        if self.grounded {
            self.velocity.y = 0.0;
        } else if !jumped {
            // With the same launch speed, two-thirds gravity gives 1.5 times
            // the rise. Release permanently ends the boost for this jump.
            let gravity = if self.jump_sustained {
                self.config.airborne_gravity / 1.5
            } else {
                self.config.airborne_gravity
            };
            self.velocity.y -= gravity * delta_seconds;
        }

        // Collision projection can create upward velocity while walking uphill.
        // Only an actual ascending jump suppresses support, not that projection.
        let can_land = !self.jump_in_flight || self.velocity.y <= 0.0;
        let descending = self.velocity.y <= 0.0;
        let mut jump_contact = false;
        let mut contact_velocity = Vec3::ZERO;
        let mut displacement = self.velocity * delta_seconds;
        if let Some(index) = scene.construction.as_deref_mut() {
            // A retained support supplies this tick's reference-frame delta and
            // velocity, but must be rediscovered below to survive the tick.
            self.support = None;
            let mut feet = construction_feet(self.position, scene.floating_origin);
            let mut remaining = displacement.as_vec3();
            let mut construction_ground = None;
            for _ in 0..5 {
                let Some(mut contact) = index.cast_capsule(feet, remaining, collision_config)
                else {
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
                        collision_config,
                        None,
                    ) {
                        contact.normal = normal;
                        contact.point.y = surface_height;
                    }
                }
                let walkable = slope_within(contact.normal, self.config.maximum_slope);
                if started_grounded
                    && can_land
                    && !walkable
                    && remaining.x.mul_add(remaining.x, remaining.z * remaining.z) > 0.0
                {
                    let raised = feet + Vec3::Y * self.config.step_height as f32;
                    if index
                        .cast_capsule(raised, remaining, collision_config)
                        .is_none()
                    {
                        let mut stepped = raised + remaining;
                        let downward = Vec3::NEG_Y * (self.config.step_height as f32 + 1.0e-3);
                        if let Some(mut step_floor) =
                            index.cast_capsule(stepped, downward, collision_config)
                            && let Some((height, normal, surface_height)) = index
                                .walkable_surface_height(
                                    step_floor.collider_index,
                                    stepped,
                                    feet.y + 1.0e-4,
                                    raised.y + 1.0e-3,
                                    collision_config,
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
                    if walkable && can_land {
                        feet.y += (contact.penetration + 1.0e-4) / contact.normal.y;
                    } else {
                        feet += contact.normal * (contact.penetration + 1.0e-4);
                    }
                } else {
                    feet += remaining * contact.time_of_impact.max(0.0);
                }
                let body_velocity = index
                    .body_pose(contact.compound_index)
                    .map_or(Vec3::ZERO, |pose| pose.velocity_at(contact.point));
                if can_land
                    && contact.point.y <= feet.y + self.config.radius as f32 + 1.0e-3
                    && slope_within(contact.normal, self.config.maximum_jump_slope)
                {
                    jump_contact = true;
                    contact_velocity = body_velocity;
                }
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
                let driven_horizontal = DVec2::new(self.velocity.x, self.velocity.z);
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
                if walkable && can_land {
                    // Ground locomotion supplies traction: passive impact projection
                    // must not turn commanded uphill speed into downhill drift.
                    self.velocity.x = driven_horizontal.x;
                    self.velocity.z = driven_horizontal.y;
                }
                let unused = 1.0 - contact.time_of_impact.clamp(0.0, 1.0);
                remaining *= unused;
                let into_remaining = remaining.dot(contact.normal);
                if into_remaining < 0.0 {
                    if walkable && can_land {
                        remaining.y -= into_remaining / contact.normal.y;
                    } else {
                        remaining -= contact.normal * into_remaining;
                    }
                }
                if walkable && can_land {
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

            if construction_ground.is_none() && can_land {
                construction_ground = index.cast_capsule(
                    feet,
                    Vec3::NEG_Y * TERRAIN_CELL_METERS as f32,
                    collision_config,
                );
            }
            if construction_ground.is_none()
                && started_grounded
                && can_land
                && let Some(support) = retained_support
                && let Some((height, normal, surface_height)) = index.walkable_surface_height(
                    support.collider_index,
                    feet,
                    feet.y - self.config.step_height as f32 - 1.0e-3,
                    feet.y + self.config.step_height as f32 + 1.0e-3,
                    collision_config,
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
                    collision_config,
                    None,
                ) {
                    feet.y = height + 1.0e-4 / normal.y;
                    contact.normal = normal;
                    contact.point.y = surface_height;
                    construction_ground = Some(contact);
                } else if !slope_within(contact.normal, self.config.maximum_slope) {
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
        let mut terrain_ground = false;
        if can_land {
            let reach = if started_grounded && !stationary_grounded {
                self.config.step_height
            } else {
                TERRAIN_CELL_METERS
            };
            if let Some(height) =
                support_height(scene.terrain, candidate, collision_config, reach, reach)
            {
                if !stationary_grounded || (height - candidate.0.y).abs() > 1.0e-3 {
                    candidate.0.y = height;
                }
                terrain_ground = true;
            }
        }
        for _ in 0..5 {
            let Some((penetration, normal, point)) =
                deepest_capsule_penetration(scene.terrain, candidate, collision_config)
            else {
                break;
            };
            let lower_contact = point.0.y <= candidate.0.y + self.config.radius + 1.0e-3;
            let walkable = lower_contact && slope_within(normal, self.config.maximum_slope);
            if can_land && lower_contact && slope_within(normal, self.config.maximum_jump_slope) {
                jump_contact = true;
                contact_velocity = Vec3::ZERO;
            }
            if walkable && can_land {
                candidate.0.y += (f64::from(penetration) + 1.0e-4) / f64::from(normal.y);
                terrain_ground = true;
                continue;
            }
            if started_grounded && can_land && lower_contact && !walkable {
                let raised = WorldPosition(candidate.0 + DVec3::Y * self.config.step_height);
                if deepest_capsule_penetration(scene.terrain, raised, collision_config).is_none()
                    && let Some(height) = support_height(
                        scene.terrain,
                        raised,
                        collision_config,
                        self.config.step_height,
                        0.0,
                    )
                {
                    candidate.0.y = height;
                    terrain_ground = true;
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
        self.grounded = self.support.is_some();
        if terrain_ground {
            self.velocity.y = 0.0;
            self.grounded = true;
            self.support = None;
            contact_velocity = Vec3::ZERO;
        }
        if self.grounded || (jump_contact && descending) {
            self.jump_in_flight = false;
            self.jump_grace_remaining = self.config.jump_grace_seconds.max(delta_seconds);
            if let Some(support) = self.support {
                contact_velocity = support
                    .previous_pose
                    .velocity_at(support.previous_pose.transform_point(support.local_anchor));
            }
            self.jump_support_velocity = contact_velocity;
            self.try_jump(input.jump_held, contact_velocity);
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
) -> Option<(f32, Vec3, WorldPosition)> {
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
    deepest: &mut Option<(f32, Vec3, WorldPosition)>,
) {
    let density = terrain.density(point);
    if density > 0.0 && deepest.is_none_or(|(current, _, _)| density > current) {
        *deepest = Some((density, density_normal(terrain, point), point));
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

/// Inclusive slope comparison despite f32 contact-normal rounding.
pub(super) fn slope_within(normal: Vec3, maximum: f64) -> bool {
    f64::from(normal.y) + 1.0e-6 >= maximum.cos()
}
