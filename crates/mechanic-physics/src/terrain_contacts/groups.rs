//! Tick-local ownership and relative-travel budgets for contact queries.
use super::MachineCollisionGeometry;

#[derive(Clone)]
struct Validity {
    travel: f64,
    allowance: [f64; 2],
    sweep_travel: f64,
    angle: f64,
    invalid: bool,
}

impl Default for Validity {
    fn default() -> Self {
        Self {
            travel: 0.0,
            sweep_travel: 0.0,
            angle: 0.0,
            invalid: false,
            allowance: [1.0; 2],
        }
    }
}

fn ratio(distance: f64, allowance: f64) -> f64 {
    if allowance > 0.0 {
        distance / allowance
    } else if distance > 0.0 {
        f64::INFINITY
    } else {
        0.0
    }
}

// Terrain and body-pair travel, rotation and fast flags are kept apart: a
// cylinder meets terrain as a circle, which its turn about its own axis doesn't
// move, but meets other bodies as the prism around it, which it does.
#[derive(Clone, Default)]
struct Assembly {
    terrain: Validity,
    internal: Validity,
    pairs: Validity,
    fast: [bool; 2],
    motion: [[f64; 3]; 2],
}

/// An assembly's measured motion over one trial path, against terrain and
/// against other bodies: contact-budget travel, swept-coverage travel,
/// rotation, and absolute travel.
#[derive(Clone, Copy, Default)]
pub(crate) struct Measured {
    terrain: [f64; 4],
    bodies: [f64; 4],
}

impl Measured {
    pub(crate) fn scale(&mut self, fraction: f64) {
        for value in self.terrain.iter_mut().chain(&mut self.bodies) {
            *value *= fraction;
        }
    }
}

// Cross-assembly pairs refresh when either participant's pair budget expires.
// Independent budgets may include travel before the other participant's last
// refresh: that overestimates relative motion and is conservative. Storage and
// expiry checks stay linear even for scenes full of standalone bodies.
#[derive(Clone)]
pub(crate) struct ContactGroups {
    assemblies: Vec<Assembly>,
    rates: std::sync::Arc<[[f64; 2]]>,
    sweep: Option<f64>,
}

impl ContactGroups {
    pub(crate) fn new(count: usize, margins: &[f64], speculative: f64) -> Self {
        let mut groups = Self {
            assemblies: vec![Assembly::default(); count],
            rates: vec![[0.0; 2]; margins.len()].into(),
            sweep: None,
        };
        groups.reset(count, margins, speculative);
        groups
    }

    // Retain allocations only. All validity state is reset at the tick boundary.
    pub(crate) fn reset(&mut self, count: usize, margins: &[f64], speculative: f64) {
        if self.assemblies.len() != count || self.rates.len() != margins.len() {
            *self = Self::new(count, margins, speculative);
            return;
        }
        self.assemblies.fill(Assembly::default());
        self.sweep = None;
        for (rate, &margin) in std::sync::Arc::make_mut(&mut self.rates)
            .iter_mut()
            .zip(margins)
        {
            *rate = [margin - speculative, margin].map(|allowance| {
                if allowance > 0.0 {
                    allowance.recip().next_up()
                } else {
                    f64::INFINITY
                }
            });
        }
    }

    pub(crate) fn reset_moving(
        &mut self,
        geometry: &MachineCollisionGeometry,
        margins: &[f64],
        speculative: f64,
    ) {
        if self.assemblies.len() != geometry.assembly_count || self.rates.len() != margins.len() {
            *self = Self::new(geometry.assembly_count, margins, speculative);
            return;
        }
        self.assemblies.fill(Assembly::default());
        self.sweep = None;
        let rates = std::sync::Arc::make_mut(&mut self.rates);
        for collider in &geometry.motion_colliders {
            let margin = margins[collider.row];
            rates[collider.row] = [margin - speculative, margin].map(|allowance| {
                if allowance > 0.0 {
                    allowance.recip().next_up()
                } else {
                    f64::INFINITY
                }
            });
        }
    }

    pub(crate) fn includes(
        &self,
        geometry: &MachineCollisionGeometry,
        body: usize,
        other: Option<usize>,
    ) -> bool {
        let a = geometry.assemblies[body];
        let b = other.map(|body| geometry.assemblies[body]);
        let first = &self.assemblies[a];
        if let Some(angle) = self.sweep {
            // The speculative reserve is queried geometry, so it can certify
            // a trial even when contact refresh is due at the next substep.
            let (one, two, fast) = match b {
                None => (&first.terrain, None, first.fast[0]),
                Some(b) if a == b => (&first.internal, None, first.fast[1]),
                Some(b) => (
                    &first.pairs,
                    Some(&self.assemblies[b].pairs),
                    first.fast[1] || self.assemblies[b].fast[1],
                ),
            };
            return fast
                || one.angle > angle
                || two.is_some_and(|v| v.angle > angle)
                || one.sweep_travel + two.map_or(0.0, |v| v.sweep_travel) > 1.0;
        }
        match b {
            None => first.terrain.invalid,
            Some(b) if a == b => first.internal.invalid,
            Some(b) => first.pairs.invalid || self.assemblies[b].pairs.invalid,
        }
    }

    pub(crate) fn invalid_count(&self) -> usize {
        let count = self.assemblies.len();
        let pairs = self.assemblies.iter().filter(|a| a.pairs.invalid).count();
        self.assemblies
            .iter()
            .map(|a| usize::from(a.terrain.invalid) + usize::from(a.internal.invalid))
            .sum::<usize>()
            + pairs * count.saturating_sub(1)
            - pairs * pairs.saturating_sub(1) / 2
    }

    pub(crate) fn len(&self) -> usize {
        let count = self.assemblies.len();
        count * (count + 3) / 2
    }

    pub(crate) fn refreshed(
        &mut self,
        geometry: &MachineCollisionGeometry,
        original: &[f64],
        queried: &[f64],
        speculative: f64,
    ) {
        let mut allowance = vec![[1.0_f64; 2]; self.assemblies.len()];
        for (((body, _), &old), &new) in geometry.collider_reach().zip(original).zip(queried) {
            let scale = &mut allowance[geometry.assemblies[body]];
            if new < old {
                scale[0] = scale[0].min(ratio(
                    (new - speculative).max(0.0),
                    (old - speculative).max(0.0),
                ));
                scale[1] = scale[1].min(ratio(new, old));
            }
        }
        let pairs_refreshed = self.assemblies.iter().any(|a| a.pairs.invalid);
        for (assembly, scale) in self.assemblies.iter_mut().zip(allowance) {
            for (group, pair) in [
                (&mut assembly.terrain, false),
                (&mut assembly.internal, false),
                (&mut assembly.pairs, true),
            ] {
                if group.invalid {
                    *group = Validity {
                        allowance: scale,
                        ..Validity::default()
                    };
                } else if pair && pairs_refreshed {
                    // A narrower pair query also narrows its valid participant's
                    // allowance. Rescale earlier travel instead of erasing it.
                    let next = [
                        group.allowance[0].min(scale[0]),
                        group.allowance[1].min(scale[1]),
                    ];
                    if next[0] < group.allowance[0] {
                        group.travel = ratio(group.travel * group.allowance[0], next[0]);
                    }
                    if next[1] < group.allowance[1] {
                        group.sweep_travel =
                            ratio(group.sweep_travel * group.allowance[1], next[1]);
                    }
                    group.allowance = next;
                }
            }
        }
    }

    pub(crate) fn measure(
        &self,
        geometry: &MachineCollisionGeometry,
        motion: &crate::MachineMotion<'_>,
    ) -> Vec<Measured> {
        let (bounds, spins) = (motion.bounds(), motion.spins());
        let mut measured = vec![Measured::default(); self.assemblies.len()];
        for collider in &geometry.motion_colliders {
            let rate = self.rates[collider.row];
            let bound = bounds[collider.body];
            let distance = bound.point_speed(collider.radius);
            let [surface, turn] = super::terrain_motion(
                collider.round.as_ref(),
                collider.radius,
                bound,
                spins[collider.body],
            );
            let measured = &mut measured[collider.assembly];
            if geometry.round_bodies[collider.body] {
                measured.terrain[2] = measured.terrain[2].max(turn);
            }
            for (bound, distance) in [
                (&mut measured.terrain, surface),
                (&mut measured.bodies, distance),
            ] {
                if distance == 0.0 {
                    continue;
                }
                bound[0] = bound[0].max((distance * rate[0]).next_up());
                bound[1] = bound[1].max((distance * rate[1]).next_up());
                bound[3] = bound[3].max(distance);
            }
        }
        for (body, bound) in bounds.iter().enumerate() {
            let measured = &mut measured[geometry.assemblies[body]];
            measured.bodies[2] = measured.bodies[2].max(bound.angular_speed);
            if !geometry.round_bodies[body] {
                measured.terrain[2] = measured.terrain[2].max(bound.angular_speed);
            }
        }
        measured
    }

    pub(crate) fn advance_measured(
        &mut self,
        geometry: &MachineCollisionGeometry,
        measured: &[Measured],
        angle: f64,
        rewound: bool,
    ) {
        for (assembly, measured) in self.assemblies.iter_mut().zip(measured) {
            let [terrain, bodies] = [measured.terrain, measured.bodies]
                .map(|[travel, sweep, rotation, _]| [travel, sweep, rotation]);
            assembly.motion = [terrain, bodies];
        }
        self.integrate(geometry, angle, rewound);
    }

    pub(crate) fn require_measured(&mut self, measured: &[Measured], threshold: f64, angle: f64) {
        self.sweep = Some(angle);
        for (assembly, motion) in self.assemblies.iter_mut().zip(measured) {
            assembly.fast[0] |= motion.terrain[3] > threshold;
            assembly.fast[1] |= motion.bodies[3] > threshold;
        }
    }

    #[cfg(test)]
    #[allow(clippy::too_many_arguments)] // Integrate solver motion against query allowances.
    pub(crate) fn advance(
        &mut self,
        geometry: &MachineCollisionGeometry,
        travelled: &[f64],
        turned: &[f64],
        angle: f64,
        rewound: bool,
    ) {
        for assembly in &mut self.assemblies {
            assembly.motion = [[0.0; 3]; 2];
        }
        for ((collider, &distance), rate) in geometry
            .colliders
            .iter()
            .zip(travelled)
            .zip(self.rates.iter())
        {
            if distance == 0.0 {
                continue;
            }
            for bound in &mut self.assemblies[geometry.assemblies[collider.body]].motion {
                bound[0] = bound[0].max((distance * rate[0]).next_up());
                bound[1] = bound[1].max((distance * rate[1]).next_up());
            }
        }
        for (body, &rotation) in turned.iter().enumerate() {
            for bound in &mut self.assemblies[geometry.assemblies[body]].motion {
                bound[2] = bound[2].max(rotation);
            }
        }
        self.integrate(geometry, angle, rewound);
    }

    fn integrate(&mut self, geometry: &MachineCollisionGeometry, angle: f64, rewound: bool) {
        for &row in &geometry.moving_assemblies {
            let assembly = &mut self.assemblies[row];
            let [terrain, bodies] = assembly.motion;
            for (group, participants, internal, [distance, sweep_distance, rotation]) in [
                (&mut assembly.terrain, 1.0, false, terrain),
                (&mut assembly.internal, 2.0, true, bodies),
                (&mut assembly.pairs, 1.0, false, bodies),
            ] {
                if internal && !geometry.internal_collisions[row] {
                    continue;
                }
                group.travel += participants * ratio(distance, group.allowance[0]);
                group.sweep_travel += participants * ratio(sweep_distance, group.allowance[1]);
                group.angle += rotation;
                group.invalid |= rewound || group.travel > 1.0 || group.angle > angle;
            }
        }
        // A pair may consume its combined allowance before either body alone.
        // Already expired participants will refresh their pairs regardless.
        let mut largest = [(usize::MAX, 0.0); 2];
        for &row in &geometry.moving_assemblies {
            let assembly = &self.assemblies[row];
            if assembly.pairs.invalid {
                continue;
            }
            if assembly.pairs.travel > largest[0].1 {
                largest[1] = largest[0];
                largest[0] = (row, assembly.pairs.travel);
            } else if assembly.pairs.travel > largest[1].1 {
                largest[1] = (row, assembly.pairs.travel);
            }
        }
        for &row in &geometry.moving_assemblies {
            let assembly = &mut self.assemblies[row];
            let other = largest[usize::from(largest[0].0 == row)].1;
            assembly.pairs.invalid |= assembly.pairs.travel + other > 1.0;
        }
    }

    pub(crate) fn reuse_assembly(&mut self, row: usize) -> usize {
        let before = self.invalid_count();
        self.assemblies[row] = Assembly::default();
        before - self.invalid_count()
    }

    pub(crate) fn needs_refresh(&self, row: usize) -> bool {
        self.assemblies[row].terrain.invalid
            || self.assemblies[row].internal.invalid
            || self.assemblies[row].pairs.invalid
    }

    #[cfg(test)]
    pub(crate) fn require_fast(
        &mut self,
        geometry: &MachineCollisionGeometry,
        travel: &[f64],
        threshold: f64,
        angle: f64,
    ) {
        self.sweep = Some(angle);
        for (collider, &distance) in geometry.colliders.iter().zip(travel) {
            for fast in &mut self.assemblies[geometry.assemblies[collider.body]].fast {
                *fast |= distance > threshold;
            }
        }
    }

    pub(crate) fn needs_sweep(&self, geometry: &MachineCollisionGeometry) -> bool {
        let Some(angle) = self.sweep else {
            return false;
        };
        let mut largest = [0.0_f64; 2];
        for &row in &geometry.moving_assemblies {
            let assembly = &self.assemblies[row];
            let [terrain_fast, bodies_fast] = assembly.fast;
            if terrain_fast
                || assembly.terrain.angle > angle
                || assembly.terrain.sweep_travel > 1.0
                || (geometry.internal_collisions[row]
                    && (bodies_fast
                        || assembly.internal.angle > angle
                        || assembly.internal.sweep_travel > 1.0))
            {
                return true;
            }
            if self.assemblies.len() > 1 && (bodies_fast || assembly.pairs.angle > angle) {
                return true;
            }
            let travel = assembly.pairs.sweep_travel;
            if travel > largest[0] {
                largest[1] = largest[0];
                largest[0] = travel;
            } else {
                largest[1] = largest[1].max(travel);
            }
        }
        self.assemblies.len() > 1 && largest[0] + largest[1] > 1.0
    }

    // Check measured trial coverage without cloning the group state.
    pub(crate) fn covers_trial(
        &self,
        geometry: &MachineCollisionGeometry,
        measured: &[Measured],
        threshold: f64,
        angle: f64,
    ) -> bool {
        let mut largest = [0.0_f64; 2];
        let paired = self.assemblies.len() > 1;
        for &row in &geometry.moving_assemblies {
            let assembly = &self.assemblies[row];
            let internal = geometry.internal_collisions[row];
            let Measured {
                terrain: [_, terrain_extra, terrain_rotation, terrain_distance],
                bodies: [_, extra, rotation, distance],
            } = measured[row];
            if terrain_distance > threshold || ((internal || paired) && distance > threshold) {
                return false;
            }
            for (group, participants, collides, extra, rotation) in [
                (
                    &assembly.terrain,
                    1.0,
                    true,
                    terrain_extra,
                    terrain_rotation,
                ),
                (&assembly.internal, 2.0, internal, extra, rotation),
            ] {
                if !collides {
                    continue;
                }
                if !(group.sweep_travel + participants * ratio(extra, group.allowance[1]) <= 1.0
                    && group.angle + rotation <= angle)
                {
                    return false;
                }
            }
            if paired {
                let pair = &assembly.pairs;
                if pair.angle + rotation > angle {
                    return false;
                }
                let travel = pair.sweep_travel + ratio(extra, pair.allowance[1]);
                if travel > largest[0] {
                    largest[1] = largest[0];
                    largest[0] = travel;
                } else {
                    largest[1] = largest[1].max(travel);
                }
            }
        }
        largest[0] + largest[1] <= 1.0
    }
}

// Geometry ownership is immutable for a tick. Other assemblies are deliberately
// NOT certified: each reuse queries them at their reconstructed current poses.
pub(crate) struct AssemblyClearance<'a> {
    geometry: &'a MachineCollisionGeometry,
    assembly: usize,
    bounds: Vec<[bevy_math::DVec3; 2]>,
    terrain_generation: u64,
    topology_generation: u64,
    origin: bevy_math::DVec3,
}

impl super::TerrainContactScene {
    pub(crate) fn assembly_clearance<'a>(
        &self,
        geometry: &'a MachineCollisionGeometry,
        motion: &crate::MachineMotion<'_>,
        origin: bevy_math::DVec3,
        padding: f64,
        assembly: usize,
    ) -> Option<AssemblyClearance<'a>> {
        if geometry.generation != motion.generation() || !origin.is_finite() {
            return None;
        }
        let bounds = geometry.body_path_bounds(motion, padding).ok()?;
        for (body, b) in bounds.iter().enumerate() {
            if geometry.assemblies[body] == assembly
                && !geometry.body_colliders[body].is_empty()
                && geometry.colliders[geometry.body_colliders[body][0]].moving
                && !self
                    .index
                    .bounds_candidates(mechanic_world::WorldBounds {
                        minimum: mechanic_world::WorldPosition((origin + b[0]).map(f64::next_down)),
                        maximum: mechanic_world::WorldPosition((origin + b[1]).map(f64::next_up)),
                    })
                    .is_empty()
            {
                return None;
            }
        }
        if geometry
            .body_candidates(&bounds)
            .iter()
            .any(|&[a, b]| geometry.assemblies[a] == assembly || geometry.assemblies[b] == assembly)
        {
            return None;
        }
        Some(AssemblyClearance {
            geometry,
            assembly,
            bounds,
            terrain_generation: self.generation,
            topology_generation: geometry.generation,
            origin,
        })
    }
}

impl AssemblyClearance<'_> {
    pub(crate) fn contains(
        &self,
        scene: &super::TerrainContactScene,
        geometry: &MachineCollisionGeometry,
        current: &crate::MachineMotion<'_>,
        origin: bevy_math::DVec3,
        padding: f64,
    ) -> bool {
        if !std::ptr::eq(self.geometry, geometry)
            || self.terrain_generation != scene.generation
            || self.topology_generation != geometry.generation
            || self.topology_generation != current.generation()
            || self.origin != origin
        {
            return false;
        }
        let Ok(mut bounds) = geometry.body_path_bounds(current, padding) else {
            return false;
        };
        for (body, b) in bounds.iter_mut().enumerate() {
            if geometry.assemblies[body] != self.assembly
                || geometry.body_colliders[body].is_empty()
            {
                continue;
            }
            if !b[0].cmpge(self.bounds[body][0]).all() || !b[1].cmple(self.bounds[body][1]).all() {
                return false;
            }
            *b = self.bounds[body];
        }
        !geometry.body_candidates(&bounds).iter().any(|&[a, b]| {
            geometry.assemblies[a] == self.assembly || geometry.assemblies[b] == self.assembly
        })
    }
}
