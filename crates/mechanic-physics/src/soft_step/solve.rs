//! Generalized rows and projected Gauss–Seidel passes for one soft substep.

use std::f64::consts::TAU;

use bevy_math::DVec3;
use mechanic_core::{
    CompiledBearing, CompiledCreation, ContactCylinder, CoordinateDrive, CylinderAnchor, DriveMode,
};

use super::{SoftStepDiagnostics, SoftStepSettings, SoftStepTerrain};
use crate::{
    BodyPose, DynamicsFactor, MachineCollisionGeometry, MachineKinematics, MachineMotion,
    MachineState, PhysicsError, TerrainContact, TerrainContactFeature, TerrainSweepHit,
    TerrainSweepOutcome,
    free_motion::advance_positions,
    joint_forces::{PassiveForce, drive_budget, drive_target},
    joint_machine::bounds,
};

/// Immutable machine inputs shared by every substep of a tick.
pub(super) struct Machine<'a> {
    pub creation: &'a CompiledCreation,
    pub passive: &'a [PassiveForce],
    pub drives: &'a [CoordinateDrive],
    /// Suspension laws of loop-closing bearings, in `dynamics.loops` order.
    pub closure_passive: &'a [PassiveForce],
    /// Generalized velocity rows of held bodies.
    pub held: &'a [bool],
}

/// Generalized inertia added to a held body's rows, so rows touching it see an
/// immovable body.
const HELD_INERTIA: f64 = 1.0e12;

/// One contact point, followed through its bodies' motion for the whole tick.
pub(super) struct Contact {
    pub source: TerrainContact,
    /// Accumulated normal, two tangent and two rolling impulses per substep.
    pub impulses: [f64; 5],
    queried_pose: BodyPose,
    other_queried_pose: Option<BodyPose>,
    local: DVec3,
    // A terrain contact on a cylinder, which stays where it touches as the
    // cylinder turns about its axis instead of following the material around.
    round: Option<(ContactCylinder, CylinderAnchor)>,
    other_local: Option<DVec3>,
    anchor: DVec3,
    other_anchor: DVec3,
    tangent_u: DVec3,
    tangent_v: DVec3,
    rolling: Option<f64>,
    sliding: bool,
    approach: f64,
    loaded: bool,
    /// Approach and slip not yet captured, which happens at the first substep
    /// this contact is solved in.
    fresh: bool,
}

impl Contact {
    pub fn new(
        source: TerrainContact,
        poses: &[BodyPose],
        warm: Option<&[f64; 5]>,
        geometry: &MachineCollisionGeometry,
    ) -> Self {
        let local = |body: usize, world: DVec3| {
            poses[body].rotation.inverse() * (world - poses[body].position)
        };
        let pose = poses[source.body];
        let round = geometry
            .rolling_shape(source.feature.collider)
            .filter(|_| source.other_body.is_none())
            .and_then(|cylinder| {
                let anchor = cylinder
                    .transformed(pose.position, pose.rotation)
                    .ok()?
                    .anchor(source.normal, source.body_point)?;
                Some((*cylinder, anchor))
            });
        let reference = if source.normal.y.abs() > 0.9 {
            DVec3::X
        } else {
            DVec3::Y
        };
        let tangent_u = reference.cross(source.normal).normalize();
        let tangent_v = source.normal.cross(tangent_u);
        let radius = source
            .body_point
            .distance(poses[source.body].position)
            .max(1e-3);
        Self {
            queried_pose: poses[source.body],
            other_queried_pose: source.other_body.map(|body| poses[body]),
            impulses: warm.copied().unwrap_or_default(),
            local: local(source.body, source.body_point),
            round,
            other_local: source
                .other_body
                .map(|body| local(body, source.terrain_point)),
            anchor: source.body_point,
            other_anchor: source.terrain_point,
            tangent_u,
            tangent_v,
            rolling: (source.response[3] > 0.0).then_some(source.response[3] * radius),
            sliding: false,
            approach: 0.0,
            loaded: false,
            fresh: true,
            source,
        }
    }

    // Where a contact on a cylinder acts at these poses.
    fn round_point(&self, poses: &[BodyPose]) -> Option<DVec3> {
        let (cylinder, anchor) = self.round?;
        let pose = poses[self.source.body];
        cylinder
            .transformed(pose.position, pose.rotation)
            .ok()?
            .anchor_point(self.source.normal, anchor)
    }

    // Only a finite contact queried at these exact poses certifies initial
    // support. A retained anchor or a speculative row alone cannot do so.
    fn initial_support(&self, poses: &[BodyPose]) -> Option<TerrainContactFeature> {
        (self.source.feature.corner < super::SUBMERGED_CORNERS
            && self.source.separation
                <= self.source.feature.obstacle.target().activation_distance()
            && self.queried_pose == poses[self.source.body]
            && self.other_queried_pose == self.source.other_body.map(|body| poses[body]))
        .then_some(self.source.feature)
    }

    fn rows(
        &self,
        model: &MachineKinematics,
        factor: &DynamicsFactor,
        output: &mut PointRows,
        jacobian: &mut [f64],
        response: &mut Vec<f64>,
    ) -> Result<(), PhysicsError> {
        let world = |body: usize, local: DVec3| {
            let pose = model.poses[body];
            pose.position + pose.rotation * local
        };
        let anchor = self
            .round_point(&model.poses)
            .unwrap_or_else(|| world(self.source.body, self.local));
        let other = match (self.source.other_body, self.other_local) {
            (Some(body), Some(local)) => world(body, local),
            _ => self.other_anchor,
        };
        let separation = self.source.separation
            + self
                .source
                .normal
                .dot((anchor - self.anchor) - (other - self.other_anchor));
        let mut point = self.source;
        point.body_point = anchor;
        point.terrain_point = other;
        let ranges = model.contact_ranges(&point);
        let count = if self.rolling.is_some() { 5 } else { 3 };
        output.rows.resize_with(count, Row::default);
        for (row, direction) in [
            self.source.normal,
            self.tangent_u,
            self.tangent_v,
            self.tangent_u,
            self.tangent_v,
        ]
        .into_iter()
        .enumerate()
        .take(count)
        {
            model.contact_row(&point, direction, row >= 3, jacobian)?;
            output.rows[row].refresh_local(factor, jacobian, response, &ranges)?;
        }
        output.separation = separation;
        output.moved = 0.0;
        Ok(())
    }
}

/// A contact's rows at one substep's starting pose.
#[derive(Default)]
pub(super) struct PointRows {
    rows: Vec<Row>,
    separation: f64,
    moved: f64,
}

/// Accumulated joint-limit and drive impulses, persisted across a tick's substeps,
/// and loop-closure impulses, which the machine carries across ticks.
pub(super) struct JointImpulses {
    lower: Vec<f64>,
    upper: Vec<f64>,
    drive: Vec<f64>,
    /// Per closure: three position rows, three orientation rows, then the lower
    /// and upper rail stops.
    pub closures: Vec<[f64; 8]>,
}

impl JointImpulses {
    pub fn new(coordinates: usize, closures: Vec<[f64; 8]>) -> Self {
        Self {
            lower: vec![0.0; coordinates],
            upper: vec![0.0; coordinates],
            drive: vec![0.0; coordinates],
            closures,
        }
    }
}

/// A loop-closing bearing's frame: where each body holds the joint and how far
/// the bodies have turned from their authored arrangement.
struct ClosureFrame {
    anchor_a: DVec3,
    anchor_b: DVec3,
    axis_a: DVec3,
    axis_b: DVec3,
    /// Rotation still needed to bring B back to its authored orientation
    /// relative to A, as a world scaled axis.
    rotation_error: DVec3,
}

impl ClosureFrame {
    fn new(creation: &CompiledCreation, poses: &[BodyPose], bearing: &CompiledBearing) -> Self {
        let (a, b) = (bearing.compound_a as usize, bearing.compound_b as usize);
        let (pose_a, pose_b) = (poses[a], poses[b]);
        let authored = creation.compounds[a].root_rotation.as_dquat().inverse()
            * creation.compounds[b].root_rotation.as_dquat();
        let mut delta = (pose_a.rotation * authored * pose_b.rotation.inverse()).normalize();
        if delta.w < 0.0 {
            delta = -delta;
        }
        Self {
            anchor_a: pose_a.position + pose_a.rotation * bearing.local_anchor_a.as_dvec3(),
            anchor_b: pose_b.position + pose_b.rotation * bearing.local_anchor_b.as_dvec3(),
            axis_a: (pose_a.rotation * bearing.local_axis_a.as_dvec3()).normalize(),
            axis_b: (pose_b.rotation * bearing.local_axis_b.as_dvec3()).normalize(),
            rotation_error: delta.to_scaled_axis(),
        }
    }
}

/// Coupled equality rows solved together, with their error and effective mass.
struct Block {
    rows: Vec<Row>,
    errors: Vec<f64>,
    effective: [[f64; 3]; 3],
}

impl Block {
    fn new(rows: Vec<Row>, errors: Vec<f64>) -> Self {
        let mut effective = [[0.0; 3]; 3];
        for (i, row) in rows.iter().enumerate() {
            for (j, other) in rows.iter().enumerate() {
                effective[i][j] = row.coupling(other);
            }
        }
        Self {
            rows,
            errors,
            effective,
        }
    }

    // Box2D's soft joint impulse for the whole block. Separate Gauss–Seidel rows
    // on one anchor converge far too slowly at a single pass.
    fn solve(
        &self,
        velocities: &mut [f64],
        impulses: &mut [f64],
        soft: Soft,
        relax: bool,
        push_out: f64,
    ) {
        let (mass_scale, impulse_scale) = if relax {
            (1.0, 0.0)
        } else {
            (soft.mass_scale, soft.impulse_scale)
        };
        let mut bias = [0.0; 3];
        if !relax {
            for (bias, error) in bias.iter_mut().zip(&self.errors) {
                *bias = soft.bias_rate * error;
            }
            // An open loop closes at a bounded speed instead of snapping shut.
            let length = bias.iter().map(|value| value * value).sum::<f64>().sqrt();
            if length > push_out {
                for value in &mut bias {
                    *value *= push_out / length;
                }
            }
        }
        let mut speed = [0.0; 3];
        for ((speed, row), bias) in speed.iter_mut().zip(&self.rows).zip(bias) {
            *speed = row.speed(velocities) + bias;
        }
        let solved = solve_block(&self.effective, speed, self.rows.len());
        for ((row, impulse), solved) in self.rows.iter().zip(impulses.iter_mut()).zip(solved) {
            let change = -mass_scale * solved - impulse_scale * *impulse;
            row.apply(velocities, change);
            *impulse += change;
        }
    }
}

// Solves a symmetric positive semidefinite system of up to three rows. Directions
// with no effective mass, such as a planar linkage's redundant out-of-plane rows,
// get no impulse instead of an unbounded one.
fn solve_block(effective: &[[f64; 3]; 3], rhs: [f64; 3], size: usize) -> [f64; 3] {
    let scale = (0..size).map(|i| effective[i][i]).fold(0.0_f64, f64::max);
    let mut solution = [0.0; 3];
    if scale <= 0.0 {
        return solution;
    }
    let floor = scale * 1e-9;
    let mut lower = [[0.0; 3]; 3];
    let mut diagonal = [0.0; 3];
    for j in 0..size {
        diagonal[j] = effective[j][j]
            - (0..j)
                .map(|p| lower[j][p] * lower[j][p] * diagonal[p])
                .sum::<f64>();
        for i in j + 1..size {
            let value = effective[i][j]
                - (0..j)
                    .map(|p| lower[i][p] * lower[j][p] * diagonal[p])
                    .sum::<f64>();
            lower[i][j] = if diagonal[j] > floor {
                value / diagonal[j]
            } else {
                0.0
            };
        }
    }
    let mut forward = [0.0; 3];
    for i in 0..size {
        forward[i] = rhs[i] - (0..i).map(|p| lower[i][p] * forward[p]).sum::<f64>();
    }
    for i in (0..size).rev() {
        let scaled = if diagonal[i] > floor {
            forward[i] / diagonal[i]
        } else {
            0.0
        };
        solution[i] = scaled
            - (i + 1..size)
                .map(|p| lower[p][i] * solution[p])
                .sum::<f64>();
    }
    solution
}

/// A sliding closure's travel: its coordinate, stops and the rows that hold them.
struct Rail {
    /// Rate of the travel coordinate, positive as B moves along A's axis.
    jacobian: Vec<f64>,
    position: f64,
    lower: Option<(Row, f64)>,
    upper: Option<(Row, f64)>,
    moved: f64,
}

/// A bearing that closes a mechanism loop, as soft rows between its two bodies
/// at one substep's starting pose. Revolute closures hold the anchor and the
/// axis direction; sliding closures hold the rail line and the orientation, and
/// stop at their travel.
pub(super) struct Closure {
    position: Block,
    orientation: Block,
    rail: Option<Rail>,
}

impl Closure {
    fn new(
        creation: &CompiledCreation,
        model: &MachineKinematics,
        factor: &DynamicsFactor,
        bearing: &CompiledBearing,
        settings: &SoftStepSettings,
    ) -> Result<Self, PhysicsError> {
        let (a, b) = (bearing.compound_a as usize, bearing.compound_b as usize);
        let frame = ClosureFrame::new(creation, &model.poses, bearing);
        // Each row is A's row minus B's, with the size of the two it came from.
        let difference = |[mut row, other]: [Vec<f64>; 2]| {
            let reference = dot(&row, &row) + dot(&other, &other);
            for (value, other) in row.iter_mut().zip(other) {
                *value -= other;
            }
            (row, reference)
        };
        // Both bodies' rows act at the anchors' midpoint. Taken at each body's own
        // anchor, a loaded loop's soft gap turns rigid motion of the whole loop,
        // such as a cart pitching about its wheels, into a relative velocity: the
        // redundant row survives with a gap-long lever and ratchets that motion.
        let midpoint = 0.5 * (frame.anchor_a + frame.anchor_b);
        let relative = |direction: DVec3| -> Result<(Vec<f64>, f64), PhysicsError> {
            Ok(difference([
                model.point_row(a, midpoint, direction)?,
                model.point_row(b, midpoint, direction)?,
            ]))
        };
        let turning = |direction: DVec3| -> Result<(Vec<f64>, f64), PhysicsError> {
            Ok(difference([
                model.angular_row(a, direction)?,
                model.angular_row(b, direction)?,
            ]))
        };
        let block = |directions: &[DVec3],
                     row: &dyn Fn(DVec3) -> Result<(Vec<f64>, f64), PhysicsError>,
                     error: DVec3|
         -> Result<Block, PhysicsError> {
            let mut rows = Vec::with_capacity(directions.len());
            let mut errors = Vec::with_capacity(directions.len());
            for &direction in directions {
                let (jacobian, reference) = row(direction)?;
                // The tree already holds this direction, as a planar linkage holds
                // its out-of-plane motion: the endpoint rows cancel to rounding
                // noise, and solving that noise would fling the machine apart.
                if dot(&jacobian, &jacobian) <= 1e-10 * reference {
                    continue;
                }
                rows.push(Row::new(factor, &jacobian)?);
                errors.push(error.dot(direction));
            }
            Ok(Block::new(rows, errors))
        };
        let separation = frame.anchor_a - frame.anchor_b;
        let (u, v) = perpendicular(frame.axis_a);
        let world = [DVec3::X, DVec3::Y, DVec3::Z];
        if !bearing.kind.is_translational() {
            return Ok(Self {
                position: block(&world, &relative, separation)?,
                orientation: block(&[u, v], &turning, frame.axis_b.cross(frame.axis_a))?,
                rail: None,
            });
        }
        let (mut jacobian, _) = relative(frame.axis_a)?;
        for value in &mut jacobian {
            *value = -*value;
        }
        let position = -separation.dot(frame.axis_a);
        let [lower, upper] = bearing.kind.bounds().map(f64::from);
        let stop = |gap: f64, sign: f64| -> Result<Option<(Row, f64)>, PhysicsError> {
            if gap.is_finite() && gap < settings.speculative {
                let row: Vec<f64> = jacobian.iter().map(|value| sign * value).collect();
                Ok(Some((Row::new(factor, &row)?, gap)))
            } else {
                Ok(None)
            }
        };
        let rail = Rail {
            lower: stop(position - lower, 1.0)?,
            upper: stop(upper - position, -1.0)?,
            jacobian: jacobian.clone(),
            position,
            moved: 0.0,
        };
        Ok(Self {
            position: block(&[u, v], &relative, separation)?,
            orientation: block(&world, &turning, frame.rotation_error)?,
            rail: Some(rail),
        })
    }

    fn len(&self) -> usize {
        self.position.rows.len()
            + self.orientation.rows.len()
            + self.rail.as_ref().map_or(0, |rail| {
                usize::from(rail.lower.is_some()) + usize::from(rail.upper.is_some())
            })
    }

    fn stops(&self) -> [Option<&(Row, f64)>; 2] {
        self.rail.as_ref().map_or([None, None], |rail| {
            [rail.lower.as_ref(), rail.upper.as_ref()]
        })
    }

    fn warm_start(&self, velocities: &mut [f64], impulses: &[f64; 8]) {
        for (row, &impulse) in self.position.rows.iter().zip(&impulses[..3]) {
            row.apply(velocities, impulse);
        }
        for (row, &impulse) in self.orientation.rows.iter().zip(&impulses[3..6]) {
            row.apply(velocities, impulse);
        }
        for (stop, &impulse) in self.stops().into_iter().zip(&impulses[6..]) {
            if let Some((row, _)) = stop {
                row.apply(velocities, impulse);
            }
        }
    }

    #[allow(clippy::too_many_arguments)] // Mirrors `normal`, which the stops share.
    fn solve(
        &self,
        velocities: &mut [f64],
        impulses: &mut [f64; 8],
        soft: Soft,
        relax: bool,
        dt: f64,
        settings: &SoftStepSettings,
    ) {
        let [position, orientation, stops] = impulses.get_disjoint_mut([0..3, 3..6, 6..8]).unwrap();
        self.position
            .solve(velocities, position, soft, relax, settings.push_out);
        self.orientation
            .solve(velocities, orientation, soft, relax, settings.push_out);
        let moved = self.rail.as_ref().map_or(0.0, |rail| rail.moved);
        for ((stop, impulse), sign) in self.stops().into_iter().zip(stops).zip([1.0, -1.0]) {
            if let Some((row, gap)) = stop {
                let gap = gap + if relax { sign * moved } else { 0.0 };
                normal(row, velocities, gap, impulse, soft, relax, dt, settings);
            }
        }
    }
}

fn perpendicular(axis: DVec3) -> (DVec3, DVec3) {
    let reference = if axis.y.abs() > 0.9 {
        DVec3::X
    } else {
        DVec3::Y
    };
    let u = reference.cross(axis).normalize();
    (u, axis.cross(u))
}

/// The widest gap and misalignment across loop-closing bearings at a pose, in
/// metres and radians. A sliding closure's travel along its rail is not a gap.
pub(super) fn closure_errors(creation: &CompiledCreation, poses: &[BodyPose]) -> (f64, f64) {
    creation
        .dynamics
        .loops
        .iter()
        .fold((0.0_f64, 0.0_f64), |(gap, angle), pattern| {
            let bearing = &creation.bearings[pattern.bearing];
            let frame = ClosureFrame::new(creation, poses, bearing);
            let separation = frame.anchor_a - frame.anchor_b;
            let (distance, misalignment) = if bearing.kind.is_translational() {
                (
                    (separation - frame.axis_a * separation.dot(frame.axis_a)).length(),
                    frame.rotation_error.length(),
                )
            } else {
                (
                    separation.length(),
                    frame.axis_a.angle_between(frame.axis_b),
                )
            };
            (gap.max(distance), angle.max(misalignment))
        })
}

#[derive(Default)]
struct Row {
    jacobian: Vec<(usize, f64)>,
    response: Vec<(usize, f64)>,
    mass: f64,
}

impl Row {
    fn new(factor: &DynamicsFactor, jacobian: &[f64]) -> Result<Self, PhysicsError> {
        let mut row = Self::default();
        row.refresh(factor, jacobian, &mut Vec::new())?;
        Ok(row)
    }

    fn refresh(
        &mut self,
        factor: &DynamicsFactor,
        jacobian: &[f64],
        response: &mut Vec<f64>,
    ) -> Result<(), PhysicsError> {
        response.clear();
        response.extend_from_slice(jacobian);
        factor.solve(response)?;
        let inverse = dot(jacobian, response);
        self.mass = if inverse > f64::EPSILON {
            1.0 / inverse
        } else {
            0.0
        };
        self.jacobian.clear();
        self.jacobian.extend(
            jacobian
                .iter()
                .copied()
                .enumerate()
                .filter(|(_, v)| *v != 0.0),
        );
        self.response.clear();
        self.response.extend(
            response
                .iter()
                .copied()
                .enumerate()
                .filter(|(_, v)| *v != 0.0),
        );
        Ok(())
    }

    fn refresh_local(
        &mut self,
        factor: &DynamicsFactor,
        jacobian: &[f64],
        response: &mut Vec<f64>,
        ranges: &[std::ops::Range<usize>],
    ) -> Result<(), PhysicsError> {
        response.resize(jacobian.len(), 0.0);
        for range in ranges {
            response[range.clone()].copy_from_slice(&jacobian[range.clone()]);
        }
        factor.solve_ranges(response, ranges)?;
        let inverse = ranges
            .iter()
            .flat_map(Clone::clone)
            .map(|row| jacobian[row] * response[row])
            .sum::<f64>();
        self.mass = if inverse > f64::EPSILON {
            1.0 / inverse
        } else {
            0.0
        };
        self.jacobian.clear();
        self.jacobian.extend(
            ranges
                .iter()
                .flat_map(Clone::clone)
                .map(|row| (row, jacobian[row]))
                .filter(|(_, v)| *v != 0.0),
        );
        self.response.clear();
        self.response.extend(
            ranges
                .iter()
                .flat_map(Clone::clone)
                .map(|row| (row, response[row]))
                .filter(|(_, v)| *v != 0.0),
        );
        Ok(())
    }

    fn coupling(&self, other: &Self) -> f64 {
        let mut response = other.response.iter().peekable();
        self.jacobian
            .iter()
            .map(|&(row, value)| {
                while response.peek().is_some_and(|&&(index, _)| index < row) {
                    response.next();
                }
                response
                    .peek()
                    .filter(|&&(index, _)| *index == row)
                    .map_or(0.0, |&&(_, v)| value * v)
            })
            .sum()
    }

    fn speed(&self, velocities: &[f64]) -> f64 {
        self.jacobian
            .iter()
            .map(|&(row, value)| value * velocities[row])
            .sum()
    }

    fn apply(&self, velocities: &mut [f64], impulse: f64) {
        for &(row, response) in &self.response {
            velocities[row] += response * impulse;
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Side {
    Lower,
    Upper,
}

struct Limit {
    coordinate: usize,
    side: Side,
    row: Row,
    gap: f64,
    moved: f64,
}

struct Drive {
    coordinate: usize,
    row: Row,
    target: f64,
    capacity: f64,
}

// Box2D's soft constraint coefficients for a stiffness, damping ratio and step.
#[derive(Clone, Copy)]
struct Soft {
    bias_rate: f64,
    mass_scale: f64,
    impulse_scale: f64,
}

impl Soft {
    fn new(hertz: f64, damping_ratio: f64, dt: f64) -> Self {
        if hertz <= 0.0 {
            return Self {
                bias_rate: 0.0,
                mass_scale: 1.0,
                impulse_scale: 0.0,
            };
        }
        let omega = TAU * hertz;
        let a1 = 2.0 * damping_ratio + dt * omega;
        let a2 = dt * omega * a1;
        let a3 = 1.0 / (1.0 + a2);
        Self {
            bias_rate: omega / a1,
            mass_scale: a2 * a3,
            impulse_scale: a3,
        }
    }
}

#[derive(Default)]
pub(super) struct Scratch {
    pub(super) points: Vec<PointRows>,
    jacobian: Vec<f64>,
    response: Vec<f64>,
    factor: Option<DynamicsFactor>,
    diagonal: Vec<f64>,
}

impl Scratch {
    pub(super) fn retained_bytes(&self) -> usize {
        self.points.capacity() * size_of::<PointRows>()
            + self
                .points
                .iter()
                .flat_map(|point| &point.rows)
                .map(|row| {
                    (row.jacobian.capacity() + row.response.capacity()) * size_of::<(usize, f64)>()
                })
                .sum::<usize>()
            + (self.jacobian.capacity() + self.response.capacity() + self.diagonal.capacity())
                * size_of::<f64>()
            + self
                .factor
                .as_ref()
                .map_or(0, DynamicsFactor::retained_bytes)
    }
}

/// Integrates forces, solves and advances positions over one substep.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)] // One ordered substep with explicit inputs and work accounting.
pub(super) fn substep(
    machine: &Machine<'_>,
    state: &mut MachineState,
    contacts: &mut [Contact],
    joints: &mut JointImpulses,
    gravity: DVec3,
    dt: f64,
    settings: &SoftStepSettings,
    terrain: Option<SoftStepTerrain<'_>>,
    coverage: &crate::terrain_contacts::ContactGroups,
    diagnostics: &mut SoftStepDiagnostics,
    scratch: &mut Scratch,
) -> Result<Substep, PhysicsError> {
    let dynamics_started = std::time::Instant::now();
    let creation = machine.creation;
    let model = MachineKinematics::assemble(creation, &state.poses, &state.coordinates)?;
    state.poses.clone_from(&model.poses);
    scratch.diagonal.resize(state.velocities.len(), 0.0);
    scratch.diagonal.fill(0.0);
    let diagonal = &mut scratch.diagonal;
    for (coordinate, &row) in creation.dynamics.coordinate_velocities.iter().enumerate() {
        diagonal[row] = machine.passive[coordinate].implicit_diagonal(dt);
    }
    for (value, _) in diagonal
        .iter_mut()
        .zip(machine.held)
        .filter(|(_, held)| **held)
    {
        *value = HELD_INERTIA;
    }
    model.refactor(
        settings.factorization,
        &state.coordinates,
        diagonal,
        &mut scratch.factor,
    )?;
    let factor = scratch
        .factor
        .as_ref()
        .ok_or(PhysicsError::InvalidDynamics)?;

    diagnostics.dynamics_ms += dynamics_started.elapsed().as_secs_f64() * 1000.0;
    let rows_started = std::time::Instant::now();
    if scratch.points.len() < contacts.len() {
        scratch
            .points
            .resize_with(contacts.len(), PointRows::default);
    }
    scratch.jacobian.resize(state.velocities.len(), 0.0);
    let points = &mut scratch.points[..contacts.len()];
    for (contact, point) in contacts.iter().zip(points.iter_mut()) {
        contact.rows(
            &model,
            factor,
            point,
            &mut scratch.jacobian,
            &mut scratch.response,
        )?;
    }
    let mut closures = creation
        .dynamics
        .loops
        .iter()
        .map(|pattern| {
            Closure::new(
                creation,
                &model,
                factor,
                &creation.bearings[pattern.bearing],
                settings,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    for (closure, impulses) in closures.iter().zip(&mut joints.closures) {
        for (stop, impulse) in closure.stops().into_iter().zip(&mut impulses[6..]) {
            if stop.is_none() {
                *impulse = 0.0;
            }
        }
    }
    // Approach and slip before forces act decide restitution and the friction
    // mode, once per contact: at the tick's first substep or after a re-query.
    for (contact, point) in contacts.iter_mut().zip(points.iter()) {
        if contact.fresh {
            contact.fresh = false;
            contact.approach = point.rows[0].speed(&state.velocities);
            contact.sliding = point.rows[1]
                .speed(&state.velocities)
                .hypot(point.rows[2].speed(&state.velocities))
                > settings.stiction_speed;
        }
    }
    for point in points.iter() {
        diagnostics.maximum_penetration = diagnostics.maximum_penetration.max(-point.separation);
    }

    diagnostics.rows_ms += rows_started.elapsed().as_secs_f64() * 1000.0;
    let dynamics_started = std::time::Instant::now();
    // Gravity, gyroscopic bias and suspension, stiffened by the passive slope.
    let mut force = model.gravity_force(creation, gravity)?;
    let bias = model.inertial_bias(creation, &state.velocities)?;
    for (value, bias) in force.iter_mut().zip(bias) {
        *value -= bias;
    }
    for (coordinate, &row) in creation.dynamics.coordinate_velocities.iter().enumerate() {
        force[row] +=
            machine.passive[coordinate].force(state.coordinates[coordinate], state.velocities[row]);
    }
    // A suspension closing a loop pushes along its rail, explicitly.
    for (closure, passive) in closures.iter().zip(machine.closure_passive) {
        if let Some(rail) = &closure.rail {
            let push = passive.force(rail.position, dot(&rail.jacobian, &state.velocities));
            for (value, rate) in force.iter_mut().zip(&rail.jacobian) {
                *value += push * rate;
            }
        }
    }
    for value in &mut force {
        *value *= dt;
    }
    factor.solve(&mut force)?;
    for (velocity, change) in state.velocities.iter_mut().zip(force) {
        *velocity += change;
    }

    diagnostics.dynamics_ms += dynamics_started.elapsed().as_secs_f64() * 1000.0;
    let rows_started = std::time::Instant::now();
    let (mut limits, drives) = joint_rows(machine, state, joints, factor, dt, settings)?;
    diagnostics.rows = points.iter().map(|point| point.rows.len()).sum::<usize>()
        + limits.len()
        + drives.len()
        + closures.iter().map(Closure::len).sum::<usize>();

    diagnostics.rows_ms += rows_started.elapsed().as_secs_f64() * 1000.0;
    let constraints_started = std::time::Instant::now();
    // Warm start with the impulses the previous substep settled on.
    for (contact, point) in contacts.iter().zip(points.iter()) {
        for (row, &impulse) in point.rows.iter().zip(&contact.impulses) {
            row.apply(&mut state.velocities, impulse);
        }
    }
    for limit in &limits {
        limit
            .row
            .apply(&mut state.velocities, *limit_impulse(joints, limit));
    }
    for drive in &drives {
        let impulse = &mut joints.drive[drive.coordinate];
        *impulse = impulse.clamp(-drive.capacity, drive.capacity);
        drive.row.apply(&mut state.velocities, *impulse);
    }
    for (closure, impulses) in closures.iter().zip(&joints.closures) {
        closure.warm_start(&mut state.velocities, impulses);
    }

    let soft = Soft::new(
        settings.contact_hertz.min(0.25 / dt),
        settings.damping_ratio,
        dt,
    );
    let joint_soft = Soft::new(
        settings.joint_hertz.min(0.25 / dt),
        settings.joint_damping_ratio,
        dt,
    );
    // A contact arriving from beyond one substep's continuous travel needs
    // converged rows: a single pass loads its manifold unevenly and the friction
    // rows then spin the body. Slower arrivals, such as a faceted wheel's rim
    // rolling onto the ground, keep the usual passes.
    let impact = contacts
        .iter()
        .any(|contact| contact.approach * dt < -settings.continuous_travel);
    let (iterations, relax_iterations) = if impact {
        (
            settings.iterations.max(settings.impact_iterations),
            settings.relax_iterations.max(settings.impact_iterations),
        )
    } else {
        (settings.iterations, settings.relax_iterations)
    };
    for _ in 0..iterations {
        pass(
            contacts,
            points,
            &limits,
            &drives,
            &closures,
            joints,
            &mut state.velocities,
            [soft, joint_soft],
            false,
            dt,
            settings,
        );
    }
    diagnostics.constraints_ms += constraints_started.elapsed().as_secs_f64() * 1000.0;
    let continuous_started = std::time::Instant::now();
    let (fraction, motion) = match terrain {
        Some(terrain) => continuous_fraction(
            creation,
            state,
            terrain,
            dt,
            settings,
            coverage,
            contacts,
            diagnostics,
        ),
        None => (1.0, crate::terrain_contacts::Measured::default()),
    };
    diagnostics.continuous_ms += continuous_started.elapsed().as_secs_f64() * 1000.0;
    let constraints_started = std::time::Instant::now();
    let advanced = fraction * dt;
    advance_positions(creation, state, advanced);
    for (coordinate, value) in state.coordinates.iter_mut().enumerate() {
        let [lower, upper] = bounds(creation, machine.drives, coordinate);
        if lower <= upper {
            *value = value.clamp(lower, upper);
        }
    }
    for point in points.iter_mut() {
        point.moved = advanced * point.rows[0].speed(&state.velocities);
    }
    for limit in &mut limits {
        limit.moved = advanced * limit.row.speed(&state.velocities);
    }
    for rail in closures
        .iter_mut()
        .filter_map(|closure| closure.rail.as_mut())
    {
        rail.moved = advanced * dot(&rail.jacobian, &state.velocities);
    }
    for _ in 0..relax_iterations {
        pass(
            contacts,
            points,
            &limits,
            &drives,
            &closures,
            joints,
            &mut state.velocities,
            [soft, joint_soft],
            true,
            dt,
            settings,
        );
    }
    for drive in &drives {
        diagnostics.drive_impulses[drive.coordinate] += joints.drive[drive.coordinate];
    }
    diagnostics.constraints_ms += constraints_started.elapsed().as_secs_f64() * 1000.0;
    Ok(Substep {
        point_count: points.len(),
        motion,
        rewound: fraction < 1.0,
    })
}

/// One substep's contact rows for the restitution pass, each collider's travel
/// bound and the largest body rotation over the advanced part, and whether a
/// continuous hit cut it short.
pub(super) struct Substep {
    pub point_count: usize,
    pub motion: crate::terrain_contacts::Measured,
    pub rewound: bool,
}

// The fraction of the substep positions may advance, stopping short of a
// collision the soft contacts would miss, with each collider's travel bound and
// the largest body rotation over that fraction. Sweep assemblies whose trial
// path exceeds either the activation threshold or proven contact coverage.
#[allow(clippy::too_many_arguments)] // The query needs both motion and current contact coverage.
fn continuous_fraction(
    creation: &CompiledCreation,
    state: &MachineState,
    terrain: SoftStepTerrain<'_>,
    dt: f64,
    settings: &SoftStepSettings,
    coverage: &crate::terrain_contacts::ContactGroups,
    contacts: &[Contact],
    diagnostics: &mut SoftStepDiagnostics,
) -> (f64, crate::terrain_contacts::Measured) {
    let displacement = state
        .velocities
        .iter()
        .map(|velocity| velocity * dt)
        .collect::<Vec<_>>();
    let Ok(motion) =
        MachineMotion::new(creation, terrain.topology_generation, state, &displacement)
    else {
        diagnostics.degrade("continuous path");
        return (1.0, crate::terrain_contacts::Measured::default());
    };
    let mut measured = coverage.measure(terrain.geometry, &motion);
    let mut fraction = 1.0;
    if settings.continuous {
        if coverage.covers_trial(
            terrain.geometry,
            &measured,
            settings.continuous_travel,
            settings.requery_angle,
        ) {
            return (fraction, measured);
        }
        let mut required = coverage.clone();
        required.advance_measured(terrain.geometry, &measured, settings.requery_angle, false);
        required.require_measured(
            terrain.geometry,
            &measured,
            settings.continuous_travel,
            settings.requery_angle,
        );
        if !required.needs_sweep(terrain.geometry) {
            return (fraction, measured);
        }
        // A sweep cuts the substep only for an arrival buried too deep at its
        // end, so where none of the swept groups is, its hits could not change
        // the substep.
        let buried = terrain.scene.recovery_groups(
            terrain.geometry,
            motion.final_poses(),
            terrain.origin,
            Some(&required),
        );
        if buried
            .as_ref()
            .is_ok_and(|buried| !buried_too_deep(terrain, buried, settings))
        {
            return (fraction, measured);
        }
        diagnostics.continuous_sweeps += 1;
        let supported = contacts
            .iter()
            .filter_map(|contact| contact.initial_support(motion.initial_poses()))
            .collect::<Vec<_>>();
        match terrain.scene.sweep_contact_groups(
            terrain.geometry,
            &motion,
            terrain.origin,
            settings.continuous_tolerance,
            settings.continuous_evaluations,
            &required,
            &supported,
        ) {
            Ok(query) => {
                diagnostics.detailed_sweep_preparations += query.detailed_preparations;
                diagnostics.continuous_cached_supports += query.cached_supports;
                diagnostics.continuous_shape_transformations += query.shape_transformations;
                diagnostics.continuous_shape_cache_hits += query.shape_cache_hits;
                diagnostics.continuous_hierarchy_node_pair_tests += query.hierarchy_node_pair_tests;
                diagnostics.continuous_pose_evaluations += query.pose_evaluations;
                diagnostics.continuous_velocity_evaluations += query.velocity_evaluations;
                diagnostics.continuous_separation_evaluations += query.separation_evaluations;
                diagnostics.continuous_collider_pair_candidates += query.collider_pair_candidates;
                diagnostics.continuous_triangle_candidates += query.triangle_candidates;
                if let TerrainSweepOutcome::Impact(hit) | TerrainSweepOutcome::Unconverged(hit) =
                    query.outcome
                    && buried.as_ref().map_or(true, |buried| {
                        missed(
                            buried,
                            hit,
                            collider_radius(terrain, hit.collider),
                            settings,
                        )
                    })
                {
                    // Stop a tolerance short of the arrival, measured along the
                    // fastest collider's path.
                    let backoff = settings.continuous_tolerance
                        / query
                            .maximum_point_displacement
                            .max(settings.continuous_tolerance);
                    let cut = (hit.fraction - backoff).max(0.0);
                    // A collider already at the gap got its rows from the query
                    // before this substep; holding it back would only stall it.
                    if cut > 0.0 {
                        fraction = cut;
                        diagnostics.continuous_hits += 1;
                    }
                }
            }
            Err(_) => diagnostics.degrade("continuous sweep"),
        }
    }
    measured.scale(fraction);
    (fraction, measured)
}

// Whether the contact rows would miss a swept arrival: at the end of the path
// the collider is buried in the target deeper than the rows recover from
// gracefully, which includes having passed into it. A collider merely leaving
// a surface it started near is not a miss. Buried vertices give the depth; a
// clipped manifold deep in overlap only reports the gap at its clipping boundary.
fn missed(
    buried: &crate::terrain_contacts::TerrainContactQuery,
    hit: TerrainSweepHit,
    radius: f64,
    settings: &SoftStepSettings,
) -> bool {
    buried.contacts.iter().any(|contact| {
        contact.feature.touches(hit.collider, hit.target)
            && contact.depth > allowed_depth(radius, settings)
    })
}

// Whether any end-of-path contact is deeper than an arrival at either of its
// colliders may end.
fn buried_too_deep(
    terrain: SoftStepTerrain<'_>,
    buried: &crate::terrain_contacts::TerrainContactQuery,
    settings: &SoftStepSettings,
) -> bool {
    buried.contacts.iter().any(|contact| {
        let other = match contact.feature.obstacle {
            crate::ContactObstacle::Collider(other) => collider_radius(terrain, other),
            crate::ContactObstacle::Terrain { .. } => f64::INFINITY,
        };
        let radius = collider_radius(terrain, contact.feature.collider).min(other);
        contact.depth > allowed_depth(radius, settings)
    })
}

fn collider_radius(terrain: SoftStepTerrain<'_>, row: usize) -> f64 {
    terrain
        .geometry
        .collider_reach()
        .nth(row)
        .map_or(0.0, |(_, radius)| radius)
}

// How deep a swept arrival may end before it cuts the substep.
fn allowed_depth(radius: f64, settings: &SoftStepSettings) -> f64 {
    settings.continuous_depth.min(0.25 * radius)
}

fn joint_rows(
    machine: &Machine<'_>,
    state: &MachineState,
    joints: &mut JointImpulses,
    factor: &DynamicsFactor,
    dt: f64,
    settings: &SoftStepSettings,
) -> Result<(Vec<Limit>, Vec<Drive>), PhysicsError> {
    let creation = machine.creation;
    let size = state.velocities.len();
    let unit = |row: usize, sign: f64| {
        let mut jacobian = vec![0.0; size];
        jacobian[row] = sign;
        Row::new(factor, &jacobian)
    };
    let mut limits = Vec::new();
    let mut drives = Vec::new();
    for (coordinate, &row) in creation.dynamics.coordinate_velocities.iter().enumerate() {
        let [lower, upper] = bounds(creation, machine.drives, coordinate);
        let position = state.coordinates[coordinate];
        let lower_gap = position - lower;
        if lower.is_finite() && lower_gap < settings.speculative {
            limits.push(Limit {
                coordinate,
                side: Side::Lower,
                row: unit(row, 1.0)?,
                gap: lower_gap,
                moved: 0.0,
            });
        } else {
            joints.lower[coordinate] = 0.0;
        }
        let upper_gap = upper - position;
        if upper.is_finite() && upper_gap < settings.speculative {
            limits.push(Limit {
                coordinate,
                side: Side::Upper,
                row: unit(row, -1.0)?,
                gap: upper_gap,
                moved: 0.0,
            });
        } else {
            joints.upper[coordinate] = 0.0;
        }
        let drive = machine.drives[coordinate];
        if drive.mode == DriveMode::Passive {
            joints.drive[coordinate] = 0.0;
            continue;
        }
        let target = drive_target(drive, position);
        let capacity = drive_budget(
            drive,
            f64::from(creation.loop_topology.coordinate_axis_inertia[coordinate]),
            state.velocities[row],
            target,
            dt,
        );
        if capacity > 0.0 {
            drives.push(Drive {
                coordinate,
                row: unit(row, 1.0)?,
                target,
                capacity,
            });
        } else {
            joints.drive[coordinate] = 0.0;
        }
    }
    Ok((limits, drives))
}

fn limit_impulse<'a>(joints: &'a mut JointImpulses, limit: &Limit) -> &'a mut f64 {
    match limit.side {
        Side::Lower => &mut joints.lower[limit.coordinate],
        Side::Upper => &mut joints.upper[limit.coordinate],
    }
}

#[allow(clippy::too_many_arguments)] // One ordered pass over every row family.
fn pass(
    contacts: &mut [Contact],
    points: &[PointRows],
    limits: &[Limit],
    drives: &[Drive],
    closures: &[Closure],
    joints: &mut JointImpulses,
    velocities: &mut [f64],
    [soft, joint_soft]: [Soft; 2],
    relax: bool,
    dt: f64,
    settings: &SoftStepSettings,
) {
    for drive in drives {
        let impulse = &mut joints.drive[drive.coordinate];
        let change = -drive.row.mass * (drive.row.speed(velocities) - drive.target);
        let next = (*impulse + change).clamp(-drive.capacity, drive.capacity);
        drive.row.apply(velocities, next - *impulse);
        *impulse = next;
    }
    for limit in limits {
        let gap = limit.gap + if relax { limit.moved } else { 0.0 };
        let impulse = limit_impulse(joints, limit);
        normal(
            &limit.row, velocities, gap, impulse, soft, relax, dt, settings,
        );
    }
    // Joints before contacts: a loop that pulls apart shows more than a contact
    // that gives a little.
    for (closure, impulses) in closures.iter().zip(&mut joints.closures) {
        closure.solve(velocities, impulses, joint_soft, relax, dt, settings);
    }
    for (contact, point) in contacts.iter_mut().zip(points) {
        let separation = point.separation + if relax { point.moved } else { 0.0 };
        normal(
            &point.rows[0],
            velocities,
            separation,
            &mut contact.impulses[0],
            soft,
            relax,
            dt,
            settings,
        );
        let load = contact.impulses[0];
        contact.loaded |= load > 0.0;
        let coefficient = if contact.sliding {
            contact.source.response[1]
        } else {
            contact.source.response[0]
        };
        let [_, friction @ .., _, _] = &mut contact.impulses;
        disk(
            [&point.rows[1], &point.rows[2]],
            velocities,
            friction,
            coefficient * load,
        );
        if let Some(length) = contact.rolling {
            let [_, _, _, rolling @ ..] = &mut contact.impulses;
            disk(
                [&point.rows[3], &point.rows[4]],
                velocities,
                rolling,
                length * load,
            );
        }
    }
}

// A one-sided row: speculative while separated, soft while overlapping, and
// unbiased during relaxation so pushes don't add energy.
#[allow(clippy::too_many_arguments)] // Shared by contacts and joint limits.
fn normal(
    row: &Row,
    velocities: &mut [f64],
    separation: f64,
    impulse: &mut f64,
    soft: Soft,
    relax: bool,
    dt: f64,
    settings: &SoftStepSettings,
) {
    let (bias, mass_scale, impulse_scale) = if separation > 0.0 {
        (separation / dt, 1.0, 0.0)
    } else if relax {
        (0.0, 1.0, 0.0)
    } else {
        (
            (soft.bias_rate * (separation + settings.slop).min(0.0)).max(-settings.push_out),
            soft.mass_scale,
            soft.impulse_scale,
        )
    };
    let change = -row.mass * mass_scale * (row.speed(velocities) + bias) - impulse_scale * *impulse;
    let next = (*impulse + change).max(0.0);
    row.apply(velocities, next - *impulse);
    *impulse = next;
}

// Two coupled rows whose combined impulse stays within a disk.
fn disk(rows: [&Row; 2], velocities: &mut [f64], impulses: &mut [f64; 2], radius: f64) {
    let mut next = [
        impulses[0] - rows[0].mass * rows[0].speed(velocities),
        impulses[1] - rows[1].mass * rows[1].speed(velocities),
    ];
    let length = next[0].hypot(next[1]);
    if length > radius {
        let scale = if length > 0.0 { radius / length } else { 0.0 };
        next = next.map(|value| value * scale);
    }
    for ((row, impulse), next) in rows.into_iter().zip(impulses.iter_mut()).zip(next) {
        row.apply(velocities, next - *impulse);
        *impulse = next;
    }
}

/// Applies material restitution to loaded contacts that arrived faster than the
/// threshold. Returns the largest approach speed left at a loaded contact.
pub(super) fn restitution(
    contacts: &mut [Contact],
    points: &[PointRows],
    velocities: &mut [f64],
    settings: &SoftStepSettings,
) -> f64 {
    let mut error = 0.0_f64;
    for (contact, point) in contacts.iter_mut().zip(points) {
        let row = &point.rows[0];
        let restitution = contact.source.response[2];
        if contact.loaded && restitution > 0.0 && contact.approach < -settings.restitution_threshold
        {
            let change = -row.mass * (row.speed(velocities) + restitution * contact.approach);
            let next = (contact.impulses[0] + change).max(0.0);
            row.apply(velocities, next - contact.impulses[0]);
            contact.impulses[0] = next;
        }
        if contact.loaded {
            error = error.max(-row.speed(velocities));
        }
    }
    error
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
