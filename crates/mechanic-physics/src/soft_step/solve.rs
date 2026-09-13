//! Generalized rows and projected Gauss–Seidel passes for one soft substep.

use std::f64::consts::TAU;

use bevy_math::DVec3;
use mechanic_core::{CompiledCreation, CoordinateDrive, DriveMode};

use super::{SoftStepDiagnostics, SoftStepSettings};
use crate::{
    BodyPose, DynamicsFactor, MachineDynamics, MachineState, PhysicsError, TerrainContact,
    free_motion::advance_positions,
    joint_forces::{PassiveForce, drive_budget, drive_target},
    joint_machine::bounds,
};

/// Immutable machine inputs shared by every substep of a tick.
pub(super) struct Machine<'a> {
    pub creation: &'a CompiledCreation,
    pub passive: &'a [PassiveForce],
    pub drives: &'a [CoordinateDrive],
}

/// One contact point, followed through its bodies' motion for the whole tick.
pub(super) struct Contact {
    pub source: TerrainContact,
    /// Accumulated normal, two tangent and two rolling impulses per substep.
    pub impulses: [f64; 5],
    local: DVec3,
    other_local: Option<DVec3>,
    anchor: DVec3,
    other_anchor: DVec3,
    tangent_u: DVec3,
    tangent_v: DVec3,
    rolling: Option<f64>,
    sliding: bool,
    approach: f64,
    loaded: bool,
}

impl Contact {
    pub fn new(source: TerrainContact, poses: &[BodyPose], warm: Option<&[f64; 5]>) -> Self {
        let local = |body: usize, world: DVec3| {
            poses[body].rotation.inverse() * (world - poses[body].position)
        };
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
            impulses: warm.copied().unwrap_or_default(),
            local: local(source.body, source.body_point),
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
            source,
        }
    }

    fn rows(
        &self,
        model: &MachineDynamics,
        factor: &DynamicsFactor,
    ) -> Result<PointRows, PhysicsError> {
        let world = |body: usize, local: DVec3| {
            let pose = model.poses[body];
            pose.position + pose.rotation * local
        };
        let anchor = world(self.source.body, self.local);
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
        let mut rows = Vec::with_capacity(5);
        for direction in [self.source.normal, self.tangent_u, self.tangent_v] {
            rows.push(Row::new(factor, point.point_row(model, direction)?)?);
        }
        if self.rolling.is_some() {
            for direction in [self.tangent_u, self.tangent_v] {
                rows.push(Row::new(factor, point.angular_row(model, direction)?)?);
            }
        }
        Ok(PointRows {
            rows,
            separation,
            moved: 0.0,
        })
    }
}

/// A contact's rows at one substep's starting pose.
pub(super) struct PointRows {
    rows: Vec<Row>,
    separation: f64,
    moved: f64,
}

/// Accumulated joint-limit and drive impulses, persisted across a tick's substeps.
pub(super) struct JointImpulses {
    lower: Vec<f64>,
    upper: Vec<f64>,
    drive: Vec<f64>,
}

impl JointImpulses {
    pub fn new(coordinates: usize) -> Self {
        Self {
            lower: vec![0.0; coordinates],
            upper: vec![0.0; coordinates],
            drive: vec![0.0; coordinates],
        }
    }
}

struct Row {
    jacobian: Vec<f64>,
    response: Vec<f64>,
    mass: f64,
}

impl Row {
    fn new(factor: &DynamicsFactor, jacobian: Vec<f64>) -> Result<Self, PhysicsError> {
        let mut response = jacobian.clone();
        factor.solve(&mut response)?;
        let inverse = dot(&jacobian, &response);
        Ok(Self {
            mass: if inverse > f64::EPSILON {
                1.0 / inverse
            } else {
                0.0
            },
            jacobian,
            response,
        })
    }

    fn speed(&self, velocities: &[f64]) -> f64 {
        dot(&self.jacobian, velocities)
    }

    fn apply(&self, velocities: &mut [f64], impulse: f64) {
        for (velocity, response) in velocities.iter_mut().zip(&self.response) {
            *velocity += response * impulse;
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

/// Integrates forces, solves and advances positions over one substep. Returns
/// the contact rows for the restitution pass.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)] // One ordered substep with explicit inputs and work accounting.
pub(super) fn substep(
    machine: &Machine<'_>,
    state: &mut MachineState,
    contacts: &mut [Contact],
    joints: &mut JointImpulses,
    gravity: DVec3,
    dt: f64,
    settings: &SoftStepSettings,
    first: bool,
    diagnostics: &mut SoftStepDiagnostics,
) -> Result<Vec<PointRows>, PhysicsError> {
    let creation = machine.creation;
    let model = MachineDynamics::assemble(creation, &state.poses, &state.coordinates)?;
    state.poses.clone_from(&model.poses);
    let mut diagonal = vec![0.0; state.velocities.len()];
    for (coordinate, &row) in creation.dynamics.coordinate_velocities.iter().enumerate() {
        diagonal[row] = machine.passive[coordinate].implicit_diagonal(dt);
    }
    let factor = settings
        .factorization
        .factor(creation, &model, &state.coordinates, &diagonal)?;

    let mut points = contacts
        .iter()
        .map(|contact| contact.rows(&model, &factor))
        .collect::<Result<Vec<_>, _>>()?;
    if first {
        // Approach and slip before this tick's forces decide restitution and
        // the friction mode.
        for (contact, point) in contacts.iter_mut().zip(&points) {
            contact.approach = point.rows[0].speed(&state.velocities);
            contact.sliding = point.rows[1]
                .speed(&state.velocities)
                .hypot(point.rows[2].speed(&state.velocities))
                > settings.stiction_speed;
        }
    }
    for point in &points {
        diagnostics.maximum_penetration = diagnostics.maximum_penetration.max(-point.separation);
    }

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
    for value in &mut force {
        *value *= dt;
    }
    factor.solve(&mut force)?;
    for (velocity, change) in state.velocities.iter_mut().zip(force) {
        *velocity += change;
    }

    let (mut limits, drives) = joint_rows(machine, state, joints, &factor, dt, settings)?;
    diagnostics.rows =
        points.iter().map(|point| point.rows.len()).sum::<usize>() + limits.len() + drives.len();

    // Warm start with the impulses the previous substep settled on.
    for (contact, point) in contacts.iter().zip(&points) {
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

    let soft = Soft::new(
        settings.contact_hertz.min(0.25 / dt),
        settings.damping_ratio,
        dt,
    );
    for _ in 0..settings.iterations {
        pass(
            contacts,
            &points,
            &limits,
            &drives,
            joints,
            &mut state.velocities,
            soft,
            false,
            dt,
            settings,
        );
    }
    advance_positions(creation, state, dt);
    for (coordinate, value) in state.coordinates.iter_mut().enumerate() {
        let [lower, upper] = bounds(creation, machine.drives, coordinate);
        if lower <= upper {
            *value = value.clamp(lower, upper);
        }
    }
    for point in &mut points {
        point.moved = dt * point.rows[0].speed(&state.velocities);
    }
    for limit in &mut limits {
        limit.moved = dt * limit.row.speed(&state.velocities);
    }
    for _ in 0..settings.relax_iterations {
        pass(
            contacts,
            &points,
            &limits,
            &drives,
            joints,
            &mut state.velocities,
            soft,
            true,
            dt,
            settings,
        );
    }
    for drive in &drives {
        diagnostics.drive_impulses[drive.coordinate] += joints.drive[drive.coordinate];
    }
    Ok(points)
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
        Row::new(factor, jacobian)
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
    joints: &mut JointImpulses,
    velocities: &mut [f64],
    soft: Soft,
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
