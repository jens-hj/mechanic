//! Loose material: many small things that fall, heap, get pushed and get carried.
//!
//! Spoil is not part of the machine solve. Each clump moves as a sphere of its
//! volume against the voxel field itself, so it agrees with an edit the moment
//! the edit commits and cannot get under a one-sided mesh. The machine pushes
//! and carries it as moving shapes, and feels it one tick later as impulses.

mod ground;
mod machine;

use std::collections::{BTreeMap, BTreeSet, HashMap};

use bevy_math::{DQuat, DVec3, IVec3};
use ground::Ground;
pub use machine::SpoilMachine;
use mechanic_world::{
    BrickCoord, ClumpCollection, GROUND_NORMAL_MIN_Y, MATERIAL_QUANTUM_M3, MaterialClump,
    TerrainField, TerrainOctree, WorldPosition,
};

/// Fastest loose material moves, in m/s. A fragment pinched by a powered tool
/// leaves at this speed at most.
const MAX_SPEED_M_S: f64 = 15.0;
/// Shortest stretch moved between looks at the ground, in metres.
const LEAST_STRIDE_M: f64 = 0.04;
/// Most looks at the ground along one tick's path.
const MAX_STRIDES: u32 = 8;
/// A clump at rest this long falls asleep, in seconds.
const SLEEP_AFTER_S: f64 = 1.0;
/// Soft clods that meet more gently than this stick together, in m/s.
const STICK_SPEED_M_S: f64 = 0.5;
/// The most one clod gathers by sticking, in cells: a clod, not a boulder.
const CLOD_CELLS: u32 = 27;
/// Slower than this over what carries it, a clump is held still, in m/s. A
/// sphere on fine terrain otherwise creeps from one facet to the next for ever.
const HOLD_SPEED_M_S: f64 = 0.12;
/// A held clump stays where it was unless the tick moved it further, in metres.
const HOLD_REACH_M: f64 = 0.01;

/// What one body of the machine felt from the spoil during a tick.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpoilReaction {
    /// Compound row.
    pub body: usize,
    /// Global point the impulse acts at.
    pub point: DVec3,
    /// Impulse in N·s.
    pub impulse: DVec3,
}

/// Outcome of one spoil tick.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SpoilStep {
    /// One summed impulse per machine body the spoil touched.
    pub reactions: Vec<SpoilReaction>,
    /// Clumps that moved this tick.
    pub awake: usize,
    /// Clumps that left the world and were dropped.
    pub lost: usize,
}

/// Steps the world's loose material.
#[derive(Default)]
pub struct SpoilSolver {
    ground: Ground,
    disturbed: BTreeSet<BrickCoord>,
    buckets: HashMap<IVec3, Vec<usize>>,
    touches: Vec<machine::Touch>,
    /// Seconds each clump has lain still on anything at all.
    resting: BTreeMap<u64, f64>,
}

#[expect(
    clippy::struct_excessive_bools,
    reason = "independent facts about what a grain touches"
)]
struct Grain<'a> {
    clump: &'a mut MaterialClump,
    /// Where the tick found it.
    start: DVec3,
    radius: f64,
    mass: f64,
    friction: f64,
    supported: bool,
    /// Touching the ground itself, not only spoil that does.
    grounded: bool,
    /// Lying on the machine, or on spoil that does.
    carried: bool,
    /// Touching the ground at any angle: a bank it leans on counts.
    touching: bool,
    /// Caught between the ground and the machine, neither of them under it.
    wedged: bool,
    /// Out of what the clump lies on, and how that moves.
    rest_normal: DVec3,
    rest_velocity: DVec3,
}

impl Grain<'_> {
    // Whether friction can hold the grain on what it lies on.
    fn holds(&self) -> bool {
        self.rest_normal.y * self.friction
            > (1.0 - self.rest_normal.y * self.rest_normal.y)
                .max(0.0)
                .sqrt()
    }
}

/// Radius of the sphere holding a clump's material, in metres.
pub fn spoil_radius(quanta: u32) -> f64 {
    (f64::from(quanta) * MATERIAL_QUANTUM_M3 * 0.75 / std::f64::consts::PI).cbrt()
}

impl SpoilSolver {
    /// Tells the solver that ground changed in these bricks: what it remembers
    /// of them is stale, and anything asleep there may have lost its footing.
    pub fn ground_changed(&mut self, bricks: impl IntoIterator<Item = BrickCoord>) {
        let bricks = bricks.into_iter().collect::<Vec<_>>();
        self.ground.forget(bricks.iter().copied());
        self.disturbed.extend(bricks);
    }

    /// Forgets all remembered ground, for a different world or terrain.
    pub fn reset(&mut self) {
        self.ground.forget_all();
        self.disturbed.clear();
    }

    /// Advances every clump by `seconds`.
    pub fn step(
        &mut self,
        clumps: &mut ClumpCollection,
        terrain: &TerrainOctree,
        field: &TerrainField,
        machine: &SpoilMachine,
        gravity: DVec3,
        seconds: f64,
    ) -> SpoilStep {
        let mut step = SpoilStep::default();
        if !(seconds.is_finite() && seconds > 0.0 && gravity.is_finite()) {
            return step;
        }
        let before = clumps.bodies.len();
        clumps.bodies.retain(|_, clump| {
            clump.position.0.is_finite()
                && clump.linear_velocity.is_finite()
                && clump.position.is_inside_world()
        });
        step.lost = before - clumps.bodies.len();
        let mut grains = clumps
            .bodies
            .values_mut()
            .map(|clump| {
                let radius = spoil_radius(clump.quanta);
                Grain {
                    start: clump.position.0,
                    radius,
                    mass: clump.mass_kg(),
                    friction: f64::from(clump.material.surface_response().static_friction),
                    supported: false,
                    grounded: false,
                    carried: false,
                    touching: false,
                    wedged: false,
                    rest_normal: DVec3::Y,
                    rest_velocity: DVec3::ZERO,
                    clump,
                }
            })
            .collect::<Vec<_>>();
        for grain in &mut grains {
            if grain.clump.sleeping
                && (grain
                    .clump
                    .position
                    .cell()
                    .is_ok_and(|cell| self.disturbed.contains(&cell.brick()))
                    || machine.stirs(grain.clump.position.0, grain.radius + 0.05))
            {
                grain.clump.sleeping = false;
                grain.clump.settled_seconds = 0.0;
            }
        }
        self.disturbed.clear();

        let mut reactions = BTreeMap::<usize, (DVec3, DVec3)>::new();
        for grain in grains.iter_mut().filter(|grain| !grain.clump.sleeping) {
            self.advance(
                grain,
                terrain,
                field,
                machine,
                gravity,
                seconds,
                &mut reactions,
            );
        }
        let absorbed = self.separate(
            &mut grains,
            terrain,
            field,
            gravity.normalize_or_zero(),
            seconds,
        );
        for grain in grains
            .iter_mut()
            .filter(|grain| !grain.clump.sleeping && grain.clump.quanta > 0)
        {
            step.awake += 1;
            self.turn_and_rest(grain, seconds);
        }
        for id in absorbed {
            clumps.bodies.remove(&id);
        }
        self.resting
            .retain(|id, _| clumps.bodies.get(id).is_some_and(|clump| !clump.sleeping));
        step.reactions = reactions
            .into_iter()
            .filter(|(_, (impulse, _))| impulse.length_squared() > 0.0)
            .map(|(body, (impulse, angular))| SpoilReaction {
                body,
                // The point that gives the summed impulse its summed moment, less
                // any twist about the impulse itself.
                point: machine.body_centre(body)
                    + impulse.cross(angular) / impulse.length_squared(),
                impulse,
            })
            .collect();
        step
    }

    // Turns a clump as it travels, counts how long it has lain still, and lets
    // it fall asleep.
    fn turn_and_rest(&mut self, grain: &mut Grain<'_>, seconds: f64) {
        let clump = &mut *grain.clump;
        // Clods tumble as they travel; the turning is for the eye only. On
        // anything, they roll with how they move over it, and lie still on it.
        let over = clump.linear_velocity - grain.rest_velocity;
        let lying = grain.supported || grain.carried || grain.wedged;
        if lying {
            clump.angular_velocity = if over.length() < 0.05 {
                DVec3::ZERO
            } else {
                grain.rest_normal.cross(over) / grain.radius
            };
        }
        clump.rotation = (DQuat::from_scaled_axis(clump.angular_velocity * seconds)
            * clump.rotation)
            .normalize();
        // Only the ground takes spoil back; a deck or a bucket just holds it.
        // Any ground will do: spoil leaning on a bank runs down it when laid.
        clump.update_settling(grain.supported || grain.touching, seconds);
        if lying && over.length() < 0.05 {
            let rested = self.resting.entry(clump.id).or_default();
            *rested += seconds;
            if *rested >= SLEEP_AFTER_S {
                clump.sleeping = true;
                clump.linear_velocity = DVec3::ZERO;
                clump.angular_velocity = DVec3::ZERO;
            }
        } else {
            self.resting.remove(&clump.id);
        }
    }

    #[expect(clippy::too_many_arguments, reason = "one grain's whole surroundings")]
    fn advance(
        &mut self,
        grain: &mut Grain<'_>,
        terrain: &TerrainOctree,
        field: &TerrainField,
        machine: &SpoilMachine,
        gravity: DVec3,
        seconds: f64,
        reactions: &mut BTreeMap<usize, (DVec3, DVec3)>,
    ) {
        let mut velocity =
            (grain.clump.linear_velocity + gravity * seconds).clamp_length_max(MAX_SPEED_M_S);
        let start = grain.clump.position.0;
        let mut position = start;
        let stride = grain.radius.max(LEAST_STRIDE_M);
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a small positive count"
        )]
        let strides = ((velocity.length() * seconds / stride).ceil() as u32).clamp(1, MAX_STRIDES);
        let mut touches = std::mem::take(&mut self.touches);
        let mut pushed_by_machine = false;
        for _ in 0..strides {
            position += velocity * seconds / f64::from(strides);
            machine.touches(position, grain.radius, &mut touches);
            for touch in &touches {
                pushed_by_machine = true;
                position += touch.normal * touch.depth;
                let change =
                    contact_response(velocity - touch.velocity, touch.normal, touch.friction);
                velocity += change;
                let impulse = -change * grain.mass;
                let point = position - touch.normal * grain.radius;
                let entry = reactions.entry(touch.body).or_default();
                entry.0 += impulse;
                entry.1 += (point - machine.body_centre(touch.body)).cross(impulse);
                if touch.normal.y > GROUND_NORMAL_MIN_Y {
                    grain.carried = true;
                    grain.rest_normal = touch.normal;
                    grain.rest_velocity = touch.velocity;
                }
            }
            for _ in 0..2 {
                let Some((push, normal)) = self.ground_push(terrain, field, position, grain.radius)
                else {
                    break;
                };
                position += push;
                velocity += contact_response(velocity, normal, grain.friction);
                grain.touching = true;
                if normal.y > GROUND_NORMAL_MIN_Y {
                    grain.supported = true;
                    grain.grounded = true;
                    grain.rest_normal = normal;
                    grain.rest_velocity = DVec3::ZERO;
                }
            }
        }
        self.touches = touches;
        // Static friction: on a slope it can hold, a clump that has all but
        // stopped stops, and stays where it was.
        let holds = grain.holds();
        // What it touched may have stopped it where no single contact did: caught
        // in a closing gap it goes nowhere, however gravity pulls. It falls no
        // faster than it fell.
        if grain.touching || pushed_by_machine {
            velocity = no_faster_than_it_fell(
                velocity,
                position - start,
                gravity.normalize_or_zero(),
                seconds,
            );
        }
        // Caught between a bank and a block, friction on both sides holds it.
        grain.wedged = grain.touching && pushed_by_machine && !grain.grounded && !grain.carried;
        if (((grain.grounded || grain.carried) && holds) || grain.wedged)
            && (velocity - grain.rest_velocity).length() < HOLD_SPEED_M_S
        {
            velocity = grain.rest_velocity;
            if grain.grounded && position.distance(start) < HOLD_REACH_M {
                position = start;
            }
        }
        grain.clump.position = WorldPosition(position);
        grain.clump.linear_velocity = velocity;
    }

    // How far and which way the ground moves a sphere out of itself.
    fn ground_push(
        &mut self,
        terrain: &TerrainOctree,
        field: &TerrainField,
        centre: DVec3,
        radius: f64,
    ) -> Option<(DVec3, DVec3)> {
        // Edited ground knows its distance only within half a cell of the
        // surface, so the sphere is felt at its rim as well as its centre.
        let mut deepest: Option<(f64, DVec3)> = None;
        for offset in [
            DVec3::ZERO,
            DVec3::NEG_Y,
            DVec3::X,
            DVec3::NEG_X,
            DVec3::Z,
            DVec3::NEG_Z,
            DVec3::Y,
        ] {
            let (density, slope) = self.ground.sample(terrain, field, centre + offset * radius);
            if density <= 0.0 {
                continue;
            }
            let steepness = slope.length();
            let (normal, depth) = if steepness > 0.2 {
                (
                    -slope / steepness,
                    (density / steepness).min(mechanic_world::TERRAIN_CELL_METERS),
                )
            } else {
                // Deep inside, where the field is flat: climb out.
                (DVec3::Y, mechanic_world::TERRAIN_CELL_METERS)
            };
            if deepest.is_none_or(|(most, _)| depth > most) {
                deepest = Some((depth, normal));
            }
        }
        deepest.map(|(depth, normal)| (normal * depth, normal))
    }

    // Pushes overlapping grains apart and takes out their closing speed. Soft
    // clods of one material that meet gently become one; returns those absorbed.
    fn separate(
        &mut self,
        grains: &mut [Grain<'_>],
        terrain: &TerrainOctree,
        field: &TerrainField,
        down: DVec3,
        seconds: f64,
    ) -> Vec<u64> {
        let mut absorbed = Vec::new();
        for bucket in self.buckets.values_mut() {
            bucket.clear();
        }
        // Buckets as wide as the widest clump: a neighbour is never further
        // than the next bucket, and fine spoil is not sorted into coarse heaps.
        let bucket = grains
            .iter()
            .map(|grain| grain.radius * 2.0)
            .fold(LEAST_STRIDE_M, f64::max);
        let key = |position: DVec3| (position / bucket).floor().as_ivec3();
        for (index, grain) in grains.iter().enumerate() {
            self.buckets
                .entry(key(grain.clump.position.0))
                .or_default()
                .push(index);
        }
        let mut pushed = Vec::new();
        for first in 0..grains.len() {
            if grains[first].clump.sleeping {
                continue;
            }
            let home = key(grains[first].clump.position.0);
            for step in 0..27 {
                let neighbour = home + IVec3::new(step % 3 - 1, step / 3 % 3 - 1, step / 9 - 1);
                let Some(bucket) = self.buckets.get(&neighbour) else {
                    continue;
                };
                for &second in bucket {
                    // An awake pair is met once; a sleeper is met by whoever is awake.
                    if second == first || (second < first && !grains[second].clump.sleeping) {
                        continue;
                    }
                    let (a, b) = pair(grains, first, second);
                    match meet(a, b) {
                        Meeting::Apart => {}
                        Meeting::Stuck => {
                            absorbed.push(b.clump.id);
                            pushed.push(first);
                        }
                        Meeting::Pushed => {
                            pushed.push(first);
                            pushed.push(second);
                        }
                    }
                }
            }
        }
        // Neighbours must not push each other into the ground.
        pushed.sort_unstable();
        pushed.dedup();
        for index in pushed {
            let grain = &mut grains[index];
            if grain.clump.sleeping {
                continue;
            }
            if let Some((push, normal)) =
                self.ground_push(terrain, field, grain.clump.position.0, grain.radius)
            {
                grain.clump.position.0 += push;
                let change = contact_response(grain.clump.linear_velocity, normal, grain.friction);
                grain.clump.linear_velocity += change;
                grain.touching = true;
            }
            // Held up in a heap, it gathers no speed: else it builds up until
            // it plunges through its neighbours and is thrown back, for ever.
            grain.clump.linear_velocity = no_faster_than_it_fell(
                grain.clump.linear_velocity,
                grain.clump.position.0 - grain.start,
                down,
                seconds,
            );
            // Friction holds a heap together as it holds a clod to the ground:
            // what stirs the bottom of a heap does not shiver up through it.
            if grain.supported
                && !grain.carried
                && grain.holds()
                && grain.clump.linear_velocity.length() < HOLD_SPEED_M_S
                && grain.clump.position.0.distance(grain.start) < HOLD_REACH_M
            {
                grain.clump.linear_velocity = DVec3::ZERO;
                grain.clump.position.0 = grain.start;
            }
        }
        absorbed
    }
}

enum Meeting {
    Apart,
    /// The second grain was folded into the first.
    Stuck,
    Pushed,
}

// Resolves two grains that may overlap: gathers them into one, or pushes them
// apart and takes out their closing speed.
fn meet(a: &mut Grain<'_>, b: &mut Grain<'_>) -> Meeting {
    if a.clump.quanta == 0 || b.clump.quanta == 0 {
        return Meeting::Apart;
    }
    let offset = b.clump.position.0 - a.clump.position.0;
    let reach = a.radius + b.radius;
    let distance = offset.length();
    if distance >= reach {
        return Meeting::Apart;
    }
    let normal = if distance > 1e-9 {
        offset / distance
    } else {
        DVec3::Y
    };
    // Spoil on the ground is not pressed into it by spoil on top.
    let a_held = a.grounded && normal.y > GROUND_NORMAL_MIN_Y;
    let b_held = b.clump.sleeping || (b.grounded && normal.y < -GROUND_NORMAL_MIN_Y);
    let share = match (a_held, b_held) {
        (false, true) => 1.0,
        (true, false) => 0.0,
        _ => b.mass / (a.mass + b.mass),
    };
    let closing = (a.clump.linear_velocity - b.clump.linear_velocity).dot(normal);
    if !b.clump.sleeping && stick(a, b, closing) {
        return Meeting::Stuck;
    }
    if b.clump.sleeping && closing > 0.5 {
        b.clump.sleeping = false;
        b.clump.settled_seconds = 0.0;
    }
    a.clump.position.0 -= normal * (reach - distance) * share;
    b.clump.position.0 += normal * (reach - distance) * (1.0 - share);
    if closing > 0.0 {
        a.clump.linear_velocity -= normal * closing * share;
        b.clump.linear_velocity += normal * closing * (1.0 - share);
    }
    // A grain lying on another lies on what that one lies on.
    if normal.y < -GROUND_NORMAL_MIN_Y {
        if !a.grounded && (b.supported || b.clump.sleeping) {
            a.rest_normal = -normal;
        }
        a.supported |= b.supported || b.clump.sleeping;
        if b.carried && !a.carried {
            a.carried = true;
            a.rest_velocity = b.rest_velocity;
        }
    } else if normal.y > GROUND_NORMAL_MIN_Y {
        if !b.grounded && a.supported {
            b.rest_normal = normal;
        }
        b.supported |= a.supported;
        if a.carried && !b.carried {
            b.carried = true;
            b.rest_velocity = a.rest_velocity;
        }
    }
    Meeting::Pushed
}

// Folds `b` into `a` when both are soft clods of one material meeting gently
// and the clod they make is no boulder. `b` is left empty for its owner to drop.
fn stick(a: &mut Grain<'_>, b: &mut Grain<'_>, closing: f64) -> bool {
    let quanta = a.clump.quanta + b.clump.quanta;
    // Spoil in flight stays as it was thrown; it gathers where it lies.
    let lying = a.grounded || a.supported || a.carried || b.grounded || b.supported || b.carried;
    if !lying
        || a.clump.material != b.clump.material
        || closing.abs() > STICK_SPEED_M_S
        || quanta > CLOD_CELLS * mechanic_world::CELL_QUANTA
        || !mechanic_world::BreakageResponse::for_material(a.clump.material).soft
    {
        return false;
    }
    let share = f64::from(b.clump.quanta) / f64::from(quanta);
    a.clump.position.0 = a.clump.position.0.lerp(b.clump.position.0, share);
    a.clump.linear_velocity = a.clump.linear_velocity.lerp(b.clump.linear_velocity, share);
    a.clump.quanta = quanta;
    a.clump.half_extents = DVec3::splat((f64::from(quanta) * MATERIAL_QUANTUM_M3).cbrt() * 0.5);
    a.clump.settled_seconds = a.clump.settled_seconds.min(b.clump.settled_seconds);
    a.supported |= b.supported;
    a.grounded |= b.grounded;
    a.carried |= b.carried;
    a.radius = spoil_radius(quanta);
    a.mass += b.mass;
    b.clump.quanta = 0;
    true
}

fn pair<'s, 'a>(
    grains: &'s mut [Grain<'a>],
    first: usize,
    second: usize,
) -> (&'s mut Grain<'a>, &'s mut Grain<'a>) {
    if first < second {
        let (low, high) = grains.split_at_mut(second);
        (&mut low[first], &mut high[0])
    } else {
        let (low, high) = grains.split_at_mut(first);
        (&mut high[0], &mut low[second])
    }
}

// A velocity whose part along gravity is no more than the tick really moved
// the body that way, and never turned about: what held it took the rest.
fn no_faster_than_it_fell(velocity: DVec3, moved: DVec3, down: DVec3, seconds: f64) -> DVec3 {
    let falling = velocity.dot(down);
    let fell = moved.dot(down) / seconds;
    let kept = if falling > 0.0 {
        fell.clamp(0.0, falling)
    } else {
        fell.clamp(falling, 0.0)
    };
    velocity + down * (kept - falling)
}

// Change of velocity that stops a body closing on a surface and drags it
// along the surface by no more than friction allows. `relative` is the body's
// velocity seen from the surface.
fn contact_response(relative: DVec3, normal: DVec3, friction: f64) -> DVec3 {
    let closing = -relative.dot(normal);
    if closing <= 0.0 {
        return DVec3::ZERO;
    }
    let sliding = relative + normal * closing;
    let speed = sliding.length();
    let drag = if speed > 1e-9 {
        -sliding * (friction * closing / speed).min(1.0)
    } else {
        DVec3::ZERO
    };
    normal * closing + drag
}

#[cfg(test)]
mod tests;
