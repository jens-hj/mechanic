//! Tick-local ownership and relative-travel budgets for contact queries.
use super::MachineCollisionGeometry;

#[derive(Clone, Copy)]
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

impl Validity {
    // A narrower query also narrows a valid participant's allowance. Rescale
    // earlier travel instead of erasing it.
    fn narrow(&mut self, scale: [f64; 2]) {
        let next = [
            self.allowance[0].min(scale[0]),
            self.allowance[1].min(scale[1]),
        ];
        if next[0] < self.allowance[0] {
            self.travel = ratio(self.travel * self.allowance[0], next[0]);
        }
        if next[1] < self.allowance[1] {
            self.sweep_travel = ratio(self.sweep_travel * self.allowance[1], next[1]);
        }
        self.allowance = next;
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

// The two largest values, with the owner of the largest.
fn largest_two(values: impl Iterator<Item = (usize, f64)>) -> [(usize, f64); 2] {
    let mut largest = [(usize::MAX, 0.0); 2];
    for (owner, value) in values {
        if value > largest[0].1 {
            largest[1] = largest[0];
            largest[0] = (owner, value);
        } else if value > largest[1].1 {
            largest[1] = (owner, value);
        }
    }
    largest
}

// Terrain, internal and cross-assembly travel, rotation and fast flags are
// kept apart: a cylinder meets terrain as a circle, which its turn about its
// own axis doesn't move, but meets other bodies as the prism around it, which
// it does; and a tree root carries its whole assembly without changing any
// distance within it. Internal budgets belong to bodies, so a spinning wheel
// refreshes only its own pairs.
#[derive(Clone, Default)]
struct Assembly {
    terrain: Validity,
    pairs: Validity,
    fast: [bool; 3],
    motion: [[f64; 3]; 2],
}

/// Motion measured over one trial path: each assembly against terrain and
/// against other assemblies, and each body within its assembly. Every entry is
/// contact-budget travel, swept-coverage travel, rotation, and absolute travel.
#[derive(Clone, Default)]
pub(crate) struct Measured {
    terrain: Vec<[f64; 4]>,
    pairs: Vec<[f64; 4]>,
    internal: Vec<[f64; 4]>,
}

impl Measured {
    pub(crate) fn scale(&mut self, fraction: f64) {
        for value in self
            .terrain
            .iter_mut()
            .chain(&mut self.pairs)
            .chain(&mut self.internal)
            .flatten()
        {
            *value *= fraction;
        }
    }
}

// Cross-assembly pairs refresh when either participant's pair budget expires,
// and pairs within an assembly when either body's budget does. Independent
// budgets may include travel before the other participant's last refresh: that
// overestimates relative motion and is conservative. Storage and expiry checks
// stay linear even for scenes full of standalone bodies.
#[derive(Clone)]
pub(crate) struct ContactGroups {
    assemblies: Vec<Assembly>,
    bodies: Vec<Validity>,
    body_motion: Vec<[f64; 3]>,
    body_assemblies: std::sync::Arc<[usize]>,
    rates: std::sync::Arc<[[f64; 2]]>,
    sweep: Option<f64>,
}

impl ContactGroups {
    pub(crate) fn new(count: usize, margins: &[f64], speculative: f64) -> Self {
        let mut groups = Self {
            assemblies: vec![Assembly::default(); count],
            bodies: Vec::new(),
            body_motion: Vec::new(),
            body_assemblies: std::sync::Arc::from([]),
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
        self.bodies.fill(Validity::default());
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
        self.bodies.fill(Validity::default());
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

    // Sizes the per-body state for this geometry.
    fn bind(&mut self, geometry: &MachineCollisionGeometry) {
        if *self.body_assemblies != *geometry.assemblies {
            self.bodies = vec![Validity::default(); geometry.bodies];
            self.body_motion = vec![[0.0; 3]; geometry.bodies];
            self.body_assemblies = geometry.assemblies.clone().into();
        }
    }

    fn body(&self, body: usize) -> Validity {
        self.bodies.get(body).copied().unwrap_or_default()
    }

    // Bodies of an assembly with their internal validity.
    fn members(&self, row: usize) -> impl Iterator<Item = (usize, &Validity)> {
        self.bodies
            .iter()
            .enumerate()
            .filter(move |&(body, _)| self.body_assemblies[body] == row)
    }

    fn internal_invalid(&self, row: usize) -> bool {
        self.members(row).any(|(_, validity)| validity.invalid)
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
            let (one, two, fast) = match (b, other) {
                (Some(b), Some(other)) if a == b => {
                    (self.body(body), Some(self.body(other)), first.fast[1])
                }
                (Some(b), _) => (
                    first.pairs,
                    Some(self.assemblies[b].pairs),
                    first.fast[2] || self.assemblies[b].fast[2],
                ),
                (None, _) => (first.terrain, None, first.fast[0]),
            };
            return fast
                || one.angle > angle
                || two.is_some_and(|v| v.angle > angle)
                || one.sweep_travel + two.map_or(0.0, |v| v.sweep_travel) > 1.0;
        }
        match (b, other) {
            (Some(b), Some(other)) if a == b => self.body(body).invalid || self.body(other).invalid,
            (Some(b), _) => first.pairs.invalid || self.assemblies[b].pairs.invalid,
            (None, _) => first.terrain.invalid,
        }
    }

    pub(crate) fn invalid_count(&self) -> usize {
        let count = self.assemblies.len();
        let pairs = self.assemblies.iter().filter(|a| a.pairs.invalid).count();
        self.assemblies
            .iter()
            .enumerate()
            .map(|(row, a)| {
                usize::from(a.terrain.invalid) + usize::from(self.internal_invalid(row))
            })
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
        self.bind(geometry);
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
        let internal_refreshed = (0..self.assemblies.len())
            .map(|row| self.internal_invalid(row))
            .collect::<Vec<_>>();
        for (body, validity) in self.bodies.iter_mut().enumerate() {
            let row = geometry.assemblies[body];
            if validity.invalid {
                *validity = Validity {
                    allowance: allowance[row],
                    ..Validity::default()
                };
            } else if internal_refreshed[row] {
                validity.narrow(allowance[row]);
            }
        }
        let pairs_refreshed = self.assemblies.iter().any(|a| a.pairs.invalid);
        for (assembly, scale) in self.assemblies.iter_mut().zip(allowance) {
            if assembly.terrain.invalid {
                assembly.terrain = Validity {
                    allowance: scale,
                    ..Validity::default()
                };
            }
            if assembly.pairs.invalid {
                assembly.pairs = Validity {
                    allowance: scale,
                    ..Validity::default()
                };
            } else if pairs_refreshed {
                assembly.pairs.narrow(scale);
            }
        }
    }

    pub(crate) fn measure(
        &self,
        geometry: &MachineCollisionGeometry,
        motion: &crate::MachineMotion<'_>,
    ) -> Measured {
        let (bounds, spins) = (motion.bounds(), motion.spins());
        let mut measured = Measured {
            terrain: vec![[0.0; 4]; self.assemblies.len()],
            pairs: vec![[0.0; 4]; self.assemblies.len()],
            internal: vec![[0.0; 4]; geometry.bodies],
        };
        for collider in &geometry.motion_colliders {
            let rate = self.rates[collider.row];
            let bound = bounds[collider.body];
            let distance = bound.point_speed(collider.radius);
            let within = if geometry.tree_frames[collider.assembly] {
                motion.tree_bounds()[collider.body].point_speed(collider.radius)
            } else {
                distance
            };
            let [surface, turn] = super::terrain_motion(
                collider.round.as_ref(),
                collider.radius,
                bound,
                spins[collider.body],
            );
            if geometry.round_bodies[collider.body] {
                let terrain = &mut measured.terrain[collider.assembly];
                terrain[2] = terrain[2].max(turn);
            }
            for (bound, distance) in [
                (&mut measured.terrain[collider.assembly], surface),
                (&mut measured.internal[collider.body], within),
                (&mut measured.pairs[collider.assembly], distance),
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
            let row = geometry.assemblies[body];
            measured.pairs[row][2] = measured.pairs[row][2].max(bound.angular_speed);
            // Contact normals stay fixed in the world, so the root's turn counts.
            measured.internal[body][2] = bound.angular_speed;
            if !geometry.round_bodies[body] {
                measured.terrain[row][2] = measured.terrain[row][2].max(bound.angular_speed);
            }
        }
        measured
    }

    pub(crate) fn advance_measured(
        &mut self,
        geometry: &MachineCollisionGeometry,
        measured: &Measured,
        angle: f64,
        rewound: bool,
    ) {
        self.bind(geometry);
        let budget = |[travel, sweep, rotation, _]: [f64; 4]| [travel, sweep, rotation];
        for ((assembly, &terrain), &pairs) in self
            .assemblies
            .iter_mut()
            .zip(&measured.terrain)
            .zip(&measured.pairs)
        {
            assembly.motion = [budget(terrain), budget(pairs)];
        }
        for (motion, &internal) in self.body_motion.iter_mut().zip(&measured.internal) {
            *motion = budget(internal);
        }
        self.integrate(geometry, angle, rewound);
    }

    pub(crate) fn require_measured(
        &mut self,
        geometry: &MachineCollisionGeometry,
        measured: &Measured,
        threshold: f64,
        angle: f64,
    ) {
        self.sweep = Some(angle);
        for (row, assembly) in self.assemblies.iter_mut().enumerate() {
            assembly.fast[0] |= measured.terrain.get(row).is_some_and(|m| m[3] > threshold);
            assembly.fast[2] |= measured.pairs.get(row).is_some_and(|m| m[3] > threshold);
        }
        for (body, internal) in measured.internal.iter().enumerate() {
            self.assemblies[geometry.assemblies[body]].fast[1] |= internal[3] > threshold;
        }
    }

    #[cfg(test)]
    pub(crate) fn advance(
        &mut self,
        geometry: &MachineCollisionGeometry,
        travelled: &[f64],
        turned: &[f64],
        angle: f64,
        rewound: bool,
    ) {
        self.bind(geometry);
        for assembly in &mut self.assemblies {
            assembly.motion = [[0.0; 3]; 2];
        }
        self.body_motion.fill([0.0; 3]);
        for ((collider, &distance), rate) in geometry
            .colliders
            .iter()
            .zip(travelled)
            .zip(self.rates.iter())
        {
            if distance == 0.0 {
                continue;
            }
            for bound in self.assemblies[geometry.assemblies[collider.body]]
                .motion
                .iter_mut()
                .chain([&mut self.body_motion[collider.body]])
            {
                bound[0] = bound[0].max((distance * rate[0]).next_up());
                bound[1] = bound[1].max((distance * rate[1]).next_up());
            }
        }
        for (body, &rotation) in turned.iter().enumerate() {
            for bound in self.assemblies[geometry.assemblies[body]]
                .motion
                .iter_mut()
                .chain([&mut self.body_motion[body]])
            {
                bound[2] = bound[2].max(rotation);
            }
        }
        self.integrate(geometry, angle, rewound);
    }

    fn integrate(&mut self, geometry: &MachineCollisionGeometry, angle: f64, rewound: bool) {
        let accumulate = |group: &mut Validity, [distance, sweep_distance, rotation]: [f64; 3]| {
            group.travel += ratio(distance, group.allowance[0]);
            group.sweep_travel += ratio(sweep_distance, group.allowance[1]);
            group.angle += rotation;
            group.invalid |= rewound || group.travel > 1.0 || group.angle > angle;
        };
        let mut moving = vec![false; self.assemblies.len()];
        for &row in &geometry.moving_assemblies {
            moving[row] = true;
            let assembly = &mut self.assemblies[row];
            let [terrain, pairs] = assembly.motion;
            accumulate(&mut assembly.terrain, terrain);
            accumulate(&mut assembly.pairs, pairs);
        }
        for (body, (validity, &motion)) in self.bodies.iter_mut().zip(&self.body_motion).enumerate()
        {
            let row = geometry.assemblies[body];
            if moving[row] && geometry.internal_collisions[row] {
                accumulate(validity, motion);
            }
        }
        // A pair may consume its combined allowance before either participant
        // alone. Already expired participants will refresh their pairs regardless.
        let largest = largest_two(
            geometry
                .moving_assemblies
                .iter()
                .map(|&row| &self.assemblies[row])
                .zip(&geometry.moving_assemblies)
                .filter(|(assembly, _)| !assembly.pairs.invalid)
                .map(|(assembly, &row)| (row, assembly.pairs.travel)),
        );
        for &row in &geometry.moving_assemblies {
            let assembly = &mut self.assemblies[row];
            let other = largest[usize::from(largest[0].0 == row)].1;
            assembly.pairs.invalid |= assembly.pairs.travel + other > 1.0;
        }
        let mut within = vec![[(usize::MAX, 0.0); 2]; self.assemblies.len()];
        for (body, validity) in self.bodies.iter().enumerate() {
            if validity.invalid {
                continue;
            }
            let largest = &mut within[geometry.assemblies[body]];
            *largest = largest_two(largest.iter().copied().chain([(body, validity.travel)]));
        }
        for (body, validity) in self.bodies.iter_mut().enumerate() {
            let largest = within[geometry.assemblies[body]];
            let other = largest[usize::from(largest[0].0 == body)].1;
            validity.invalid |= validity.travel + other > 1.0;
        }
    }

    pub(crate) fn reuse_assembly(&mut self, row: usize) -> usize {
        let before = self.invalid_count();
        self.assemblies[row] = Assembly::default();
        for (body, validity) in self.bodies.iter_mut().enumerate() {
            if self.body_assemblies[body] == row {
                *validity = Validity::default();
            }
        }
        before - self.invalid_count()
    }

    pub(crate) fn needs_refresh(&self, row: usize) -> bool {
        self.assemblies[row].terrain.invalid
            || self.internal_invalid(row)
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
            let [terrain_fast, internal_fast, pairs_fast] = assembly.fast;
            let [(_, first), (_, second)] =
                largest_two(self.members(row).map(|(body, v)| (body, v.sweep_travel)));
            if terrain_fast
                || assembly.terrain.angle > angle
                || assembly.terrain.sweep_travel > 1.0
                || (geometry.internal_collisions[row]
                    && (internal_fast
                        || self.members(row).any(|(_, v)| v.angle > angle)
                        || first + second > 1.0))
            {
                return true;
            }
            if self.assemblies.len() > 1 && (pairs_fast || assembly.pairs.angle > angle) {
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
        measured: &Measured,
        threshold: f64,
        angle: f64,
    ) -> bool {
        let mut largest = [0.0_f64; 2];
        let paired = self.assemblies.len() > 1;
        for &row in &geometry.moving_assemblies {
            let assembly = &self.assemblies[row];
            let [_, terrain_extra, terrain_rotation, terrain_distance] = measured.terrain[row];
            let [_, extra, rotation, distance] = measured.pairs[row];
            if terrain_distance > threshold
                || (paired && distance > threshold)
                || !(assembly.terrain.sweep_travel
                    + ratio(terrain_extra, assembly.terrain.allowance[1])
                    <= 1.0
                    && assembly.terrain.angle + terrain_rotation <= angle)
            {
                return false;
            }
            if geometry.internal_collisions[row] {
                let mut within = [(usize::MAX, 0.0); 2];
                for (body, validity) in self.members(row) {
                    let [_, extra, rotation, distance] = measured.internal[body];
                    if distance > threshold || validity.angle + rotation > angle {
                        return false;
                    }
                    let travel = validity.sweep_travel + ratio(extra, validity.allowance[1]);
                    within = largest_two(within.into_iter().chain([(body, travel)]));
                }
                if within[0].1 + within[1].1 > 1.0 {
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
