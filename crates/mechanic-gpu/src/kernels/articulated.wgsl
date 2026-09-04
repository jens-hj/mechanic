struct TickConfig {
    body_count: u32,
    tick_index: u32,
    snapshot_slot: u32,
    collider_count: u32,
    delta_seconds: f32,
    gravity_y: f32,
    linear_damping: f32,
    angular_damping: f32,
    bearing_count: u32,
    suppression_count: u32,
    pair_capacity: u32,
    flags: u32,
    hash_capacity: u32,
    solver_iterations: u32,
    reserved_a: u32,
    reserved_b: u32,
};

struct Mass {
    inverse_mass: vec4<f32>,
    inverse_inertia_x: vec4<f32>,
    inverse_inertia_y: vec4<f32>,
    inverse_inertia_z: vec4<f32>,
};

struct Bearing {
    local_anchor_a: vec4<f32>,
    local_anchor_b: vec4<f32>,
    local_axis_a: vec4<f32>,
    local_axis_b: vec4<f32>,
    metadata: vec4<u32>,
};

struct MechanismBody {
    metadata: vec4<u32>,
    traversal: vec4<u32>,
    bind_relative_position: vec4<f32>,
    bind_relative_rotation: vec4<f32>,
};

struct Coordinate {
    angle: f32,
    angular_velocity: f32,
};

struct Drive {
    mode: u32,
    max_acceleration: f32,
    max_speed: f32,
    target_speed: f32,
    target_angle: f32,
    min_angle: f32,
    max_angle: f32,
    source_a_max_acceleration: f32,
    source_a_no_load_speed: f32,
    source_b_max_acceleration: f32,
    source_b_no_load_speed: f32,
    padding: f32,
};

struct DriveConstraint {
    bearing: Bearing,
    drive: Drive,
    // Axis inertia, accumulated impulse, current angle, reserved.
    state: vec4<f32>,
    // Child body, parent body, tree direction, coordinate.
    metadata: vec4<u32>,
};

const FIXED_VELOCITY_SCALE: f32 = 1048576.0;
// Diagonal Jacobi rows share off-centre inertia terms and adjacent bodies.
// Under-relaxation keeps their simultaneous impulses dissipative.
const BEARING_PROJECTION_RELAXATION: f32 = 0.5;
const GRAVITY_ALIGNED_BEARING_SLEEP_SPEED: f32 = 0.005;
const INVALID_INDEX: u32 = 0xffffffffu;
const DRIVE_MODE_PASSIVE: u32 = 0u;
const DRIVE_MODE_ANGLE: u32 = 2u;
// The outer position loop converts error to a tapered velocity request. The
// existing torque-limited velocity loop supplies derivative damping, matching
// the cascaded position/velocity control used by physical servos.
const DRIVE_ANGLE_POSITION_GAIN: f32 = 6.0;
const DRIVE_ANGLE_BRAKE_MARGIN: f32 = 0.8;
const DRIVE_ANGLE_DEADBAND: f32 = 0.0005;
const SMALL_MECHANISM_SERIAL_CLEANUP_STEPS: u32 = 5u;
const INVALID_NUMERIC_FLAG: u32 = 2u;

@group(0) @binding(0) var<uniform> config: TickConfig;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> rotations: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> linear_velocities: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> angular_velocities: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> masses: array<Mass>;
@group(0) @binding(6) var<storage, read> bearings: array<Bearing>;
@group(0) @binding(7) var<storage, read> mechanism_bodies: array<MechanismBody>;
@group(0) @binding(8) var<storage, read_write> coordinates: array<Coordinate>;
@group(0) @binding(9) var<storage, read_write> velocity_deltas: array<atomic<i32>>;
@group(0) @binding(10) var<storage, read> preorder: array<u32>;
@group(0) @binding(11) var<storage, read_write> diagnostics: array<atomic<u32>>;
@group(0) @binding(12) var<storage, read> drives: array<Drive>;
@group(0) @binding(13) var<storage, read_write> drive_constraints: array<DriveConstraint>;

fn quat_rotate(rotation: vec4<f32>, vector: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(rotation.xyz, vector);
    return vector + rotation.w * t + cross(rotation.xyz, t);
}

fn finite4(value: vec4<f32>) -> bool {
    return all(value == value) && all(abs(value) < vec4<f32>(3.402823e+38));
}

fn world_inverse_inertia(body: u32, vector: vec3<f32>) -> vec3<f32> {
    let rotation = rotations[body];
    let local = quat_rotate(vec4<f32>(-rotation.xyz, rotation.w), vector);
    let mass = masses[body];
    let local_result = mass.inverse_inertia_x.xyz * local.x
        + mass.inverse_inertia_y.xyz * local.y
        + mass.inverse_inertia_z.xyz * local.z;
    return quat_rotate(rotation, local_result);
}

fn add_delta(body: u32, linear: vec3<f32>, angular: vec3<f32>) {
    let base = body * 6u;
    atomicAdd(&velocity_deltas[base], i32(round(linear.x * FIXED_VELOCITY_SCALE)));
    atomicAdd(&velocity_deltas[base + 1u], i32(round(linear.y * FIXED_VELOCITY_SCALE)));
    atomicAdd(&velocity_deltas[base + 2u], i32(round(linear.z * FIXED_VELOCITY_SCALE)));
    atomicAdd(&velocity_deltas[base + 3u], i32(round(angular.x * FIXED_VELOCITY_SCALE)));
    atomicAdd(&velocity_deltas[base + 4u], i32(round(angular.y * FIXED_VELOCITY_SCALE)));
    atomicAdd(&velocity_deltas[base + 5u], i32(round(angular.z * FIXED_VELOCITY_SCALE)));
}

@compute @workgroup_size(256)
fn prepare_drive_constraints(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let index = invocation.x;
    if index >= config.bearing_count {
        return;
    }
    let coordinate = drive_constraints[index].metadata.w;
    if coordinate != INVALID_INDEX {
        drive_constraints[index].state.y = 0.0;
        drive_constraints[index].state.z = coordinates[coordinate].angle;
    }
}

fn drive_desired_speed(index: u32) -> f32 {
    let constraint = drive_constraints[index];
    let drive = constraint.drive;
    var desired = clamp(drive.target_speed, -drive.max_speed, drive.max_speed);
    if drive.mode == DRIVE_MODE_ANGLE {
        let error = drive.target_angle - constraint.state.z;
        let proportional = DRIVE_ANGLE_POSITION_GAIN * abs(error);
        let brake = DRIVE_ANGLE_BRAKE_MARGIN
            * sqrt(2.0 * drive.max_acceleration * abs(error));
        desired = sign(error) * min(min(proportional, brake), drive.max_speed);
        if abs(error) < DRIVE_ANGLE_DEADBAND {
            desired = 0.0;
        }
    }
    return desired;
}

fn drive_source_fade(measured: f32, requested: f32, no_load_speed: f32) -> f32 {
    if no_load_speed <= 0.0 {
        return 0.0;
    }
    if abs(requested) > 1.0e-6 && measured * requested > 0.0 {
        return clamp(1.0 - abs(measured) / no_load_speed, 0.0, 1.0);
    }
    return 1.0;
}

fn drive_max_impulse(index: u32, measured: f32, desired: f32) -> f32 {
    let constraint = drive_constraints[index];
    let drive = constraint.drive;
    let requested = desired - measured;
    let acceleration =
        drive.source_a_max_acceleration
            * drive_source_fade(measured, requested, drive.source_a_no_load_speed)
        + drive.source_b_max_acceleration
            * drive_source_fade(measured, requested, drive.source_b_no_load_speed);
    return max(acceleration * constraint.state.x * config.delta_seconds, 0.0);
}

fn project_drive_velocity_row(index: u32) {
    let constraint = drive_constraints[index];
    if constraint.metadata.w == INVALID_INDEX || constraint.drive.mode == DRIVE_MODE_PASSIVE {
        return;
    }
    let child = constraint.metadata.x;
    let parent = constraint.metadata.y;
    let bearing = constraint.bearing;
    var axis = normalize(quat_rotate(rotations[parent], bearing.local_axis_a.xyz));
    if constraint.metadata.z != 0u {
        axis = -normalize(quat_rotate(rotations[parent], bearing.local_axis_b.xyz));
    }
    let inverse_parent = world_inverse_inertia(parent, axis);
    let inverse_child = world_inverse_inertia(child, axis);
    let denominator = dot(axis, inverse_parent + inverse_child);
    if denominator <= 1.0e-12 {
        return;
    }
    let measured = dot(angular_velocities[child].xyz - angular_velocities[parent].xyz, axis);
    let desired = drive_desired_speed(index);
    let previous = constraint.state.y;
    let limit = drive_max_impulse(index, measured, desired);
    let accumulated = clamp(previous + (desired - measured) / denominator, -limit, limit);
    let impulse = accumulated - previous;
    drive_constraints[index].state.y = accumulated;
    add_delta(parent, vec3<f32>(0.0), -inverse_parent * impulse);
    add_delta(child, vec3<f32>(0.0), inverse_child * impulse);
}

fn project_drive_velocity_row_immediate(index: u32) {
    let constraint = drive_constraints[index];
    if constraint.metadata.w == INVALID_INDEX || constraint.drive.mode == DRIVE_MODE_PASSIVE {
        return;
    }
    let child = constraint.metadata.x;
    let parent = constraint.metadata.y;
    let bearing = constraint.bearing;
    var axis = normalize(quat_rotate(rotations[parent], bearing.local_axis_a.xyz));
    if constraint.metadata.z != 0u {
        axis = -normalize(quat_rotate(rotations[parent], bearing.local_axis_b.xyz));
    }
    let inverse_parent = world_inverse_inertia(parent, axis);
    let inverse_child = world_inverse_inertia(child, axis);
    let denominator = dot(axis, inverse_parent + inverse_child);
    if denominator <= 1.0e-12 {
        return;
    }
    let measured = dot(angular_velocities[child].xyz - angular_velocities[parent].xyz, axis);
    let desired = drive_desired_speed(index);
    let previous = constraint.state.y;
    let limit = drive_max_impulse(index, measured, desired);
    let accumulated = clamp(previous + (desired - measured) / denominator, -limit, limit);
    let impulse = accumulated - previous;
    drive_constraints[index].state.y = accumulated;
    angular_velocities[parent] = vec4<f32>(
        angular_velocities[parent].xyz - inverse_parent * impulse,
        0.0,
    );
    angular_velocities[child] = vec4<f32>(
        angular_velocities[child].xyz + inverse_child * impulse,
        0.0,
    );
}

fn solve_linear_axis(
    body_a: u32,
    body_b: u32,
    arm_a: vec3<f32>,
    arm_b: vec3<f32>,
    relative: vec3<f32>,
    direction: vec3<f32>,
) {
    let angular_a = cross(arm_a, direction);
    let angular_b = cross(arm_b, direction);
    let denominator = masses[body_a].inverse_mass.x + masses[body_b].inverse_mass.x
        + dot(angular_a, world_inverse_inertia(body_a, angular_a))
        + dot(angular_b, world_inverse_inertia(body_b, angular_b));
    if denominator <= 1.0e-12 {
        return;
    }
    let impulse = direction * (
        BEARING_PROJECTION_RELAXATION * dot(relative, direction) / denominator
    );
    add_delta(
        body_a,
        impulse * masses[body_a].inverse_mass.x,
        world_inverse_inertia(body_a, cross(arm_a, impulse)),
    );
    add_delta(
        body_b,
        -impulse * masses[body_b].inverse_mass.x,
        world_inverse_inertia(body_b, cross(arm_b, -impulse)),
    );
}

fn solve_angular_axis(body_a: u32, body_b: u32, relative: vec3<f32>, axis: vec3<f32>) {
    let inverse_a = world_inverse_inertia(body_a, axis);
    let inverse_b = world_inverse_inertia(body_b, axis);
    let denominator = dot(axis, inverse_a + inverse_b);
    if denominator <= 1.0e-12 {
        return;
    }
    let impulse = BEARING_PROJECTION_RELAXATION * dot(relative, axis) / denominator;
    add_delta(body_a, vec3<f32>(0.0), inverse_a * impulse);
    add_delta(body_b, vec3<f32>(0.0), -inverse_b * impulse);
}

fn solve_linear_axis_immediate(
    body_a: u32,
    body_b: u32,
    arm_a: vec3<f32>,
    arm_b: vec3<f32>,
    relative: vec3<f32>,
    direction: vec3<f32>,
) {
    let angular_a = cross(arm_a, direction);
    let angular_b = cross(arm_b, direction);
    let denominator = masses[body_a].inverse_mass.x + masses[body_b].inverse_mass.x
        + dot(angular_a, world_inverse_inertia(body_a, angular_a))
        + dot(angular_b, world_inverse_inertia(body_b, angular_b));
    if denominator <= 1.0e-12 {
        return;
    }
    let impulse = direction * (dot(relative, direction) / denominator);
    linear_velocities[body_a] = vec4<f32>(
        linear_velocities[body_a].xyz + impulse * masses[body_a].inverse_mass.x,
        0.0,
    );
    angular_velocities[body_a] = vec4<f32>(
        angular_velocities[body_a].xyz
            + world_inverse_inertia(body_a, cross(arm_a, impulse)),
        0.0,
    );
    linear_velocities[body_b] = vec4<f32>(
        linear_velocities[body_b].xyz - impulse * masses[body_b].inverse_mass.x,
        0.0,
    );
    angular_velocities[body_b] = vec4<f32>(
        angular_velocities[body_b].xyz
            + world_inverse_inertia(body_b, cross(arm_b, -impulse)),
        0.0,
    );
}

fn solve_angular_axis_immediate(
    body_a: u32,
    body_b: u32,
    relative: vec3<f32>,
    axis: vec3<f32>,
) {
    let inverse_a = world_inverse_inertia(body_a, axis);
    let inverse_b = world_inverse_inertia(body_b, axis);
    let denominator = dot(axis, inverse_a + inverse_b);
    if denominator <= 1.0e-12 {
        return;
    }
    let impulse = dot(relative, axis) / denominator;
    angular_velocities[body_a] = vec4<f32>(
        angular_velocities[body_a].xyz + inverse_a * impulse,
        0.0,
    );
    angular_velocities[body_b] = vec4<f32>(
        angular_velocities[body_b].xyz - inverse_b * impulse,
        0.0,
    );
}

fn project_bearing_velocity_row(index: u32) {
    let bearing = drive_constraints[index].bearing;
    let body_a = bearing.metadata.x;
    let body_b = bearing.metadata.y;
    let arm_a = quat_rotate(rotations[body_a], bearing.local_anchor_a.xyz);
    let arm_b = quat_rotate(rotations[body_b], bearing.local_anchor_b.xyz);

    var anchor_velocity_a = linear_velocities[body_a].xyz
        + cross(angular_velocities[body_a].xyz, arm_a);
    var anchor_velocity_b = linear_velocities[body_b].xyz
        + cross(angular_velocities[body_b].xyz, arm_b);
    solve_linear_axis_immediate(
        body_a,
        body_b,
        arm_a,
        arm_b,
        anchor_velocity_b - anchor_velocity_a,
        vec3<f32>(1.0, 0.0, 0.0),
    );
    anchor_velocity_a = linear_velocities[body_a].xyz
        + cross(angular_velocities[body_a].xyz, arm_a);
    anchor_velocity_b = linear_velocities[body_b].xyz
        + cross(angular_velocities[body_b].xyz, arm_b);
    solve_linear_axis_immediate(
        body_a,
        body_b,
        arm_a,
        arm_b,
        anchor_velocity_b - anchor_velocity_a,
        vec3<f32>(0.0, 1.0, 0.0),
    );
    anchor_velocity_a = linear_velocities[body_a].xyz
        + cross(angular_velocities[body_a].xyz, arm_a);
    anchor_velocity_b = linear_velocities[body_b].xyz
        + cross(angular_velocities[body_b].xyz, arm_b);
    solve_linear_axis_immediate(
        body_a,
        body_b,
        arm_a,
        arm_b,
        anchor_velocity_b - anchor_velocity_a,
        vec3<f32>(0.0, 0.0, 1.0),
    );

    let axis_a = normalize(quat_rotate(rotations[body_a], bearing.local_axis_a.xyz));
    let axis_b = normalize(quat_rotate(rotations[body_b], bearing.local_axis_b.xyz));
    let hinge_axis = normalize(axis_a + axis_b);
    let helper = select(
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        abs(hinge_axis.x) > 0.8,
    );
    let tangent_a = normalize(cross(hinge_axis, helper));
    let tangent_b = cross(hinge_axis, tangent_a);
    var relative_angular = angular_velocities[body_b].xyz - angular_velocities[body_a].xyz;
    solve_angular_axis_immediate(body_a, body_b, relative_angular, tangent_a);
    relative_angular = angular_velocities[body_b].xyz - angular_velocities[body_a].xyz;
    solve_angular_axis_immediate(body_a, body_b, relative_angular, tangent_b);
}

@compute @workgroup_size(1)
fn project_bearing_velocities_serial() {
    for (var index = 0u; index < config.bearing_count; index += 1u) {
        project_drive_velocity_row_immediate(index);
        project_bearing_velocity_row(index);
    }
    for (var index = config.bearing_count; index > 0u; index -= 1u) {
        project_bearing_velocity_row(index - 1u);
    }
}

@compute @workgroup_size(256)
fn project_bearing_velocities(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let index = invocation.x;
    if index >= config.bearing_count {
        return;
    }
    project_drive_velocity_row(index);
    let bearing = drive_constraints[index].bearing;
    let body_a = bearing.metadata.x;
    let body_b = bearing.metadata.y;
    let arm_a = quat_rotate(rotations[body_a], bearing.local_anchor_a.xyz);
    let arm_b = quat_rotate(rotations[body_b], bearing.local_anchor_b.xyz);
    let anchor_velocity_a = linear_velocities[body_a].xyz
        + cross(angular_velocities[body_a].xyz, arm_a);
    let anchor_velocity_b = linear_velocities[body_b].xyz
        + cross(angular_velocities[body_b].xyz, arm_b);
    let relative_linear = anchor_velocity_b - anchor_velocity_a;
    solve_linear_axis(body_a, body_b, arm_a, arm_b, relative_linear, vec3<f32>(1.0, 0.0, 0.0));
    solve_linear_axis(body_a, body_b, arm_a, arm_b, relative_linear, vec3<f32>(0.0, 1.0, 0.0));
    solve_linear_axis(body_a, body_b, arm_a, arm_b, relative_linear, vec3<f32>(0.0, 0.0, 1.0));

    let axis_a = normalize(quat_rotate(rotations[body_a], bearing.local_axis_a.xyz));
    let axis_b = normalize(quat_rotate(rotations[body_b], bearing.local_axis_b.xyz));
    let hinge_axis = normalize(axis_a + axis_b);
    let helper = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(hinge_axis.x) > 0.8);
    let tangent_a = normalize(cross(hinge_axis, helper));
    let tangent_b = cross(hinge_axis, tangent_a);
    let relative_angular = angular_velocities[body_b].xyz - angular_velocities[body_a].xyz;
    solve_angular_axis(body_a, body_b, relative_angular, tangent_a);
    solve_angular_axis(body_a, body_b, relative_angular, tangent_b);
}

@compute @workgroup_size(256)
fn apply_velocity_deltas(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let body = invocation.x;
    if body >= config.body_count {
        return;
    }
    let base = body * 6u;
    let linear_delta = vec3<f32>(
        f32(atomicExchange(&velocity_deltas[base], 0)),
        f32(atomicExchange(&velocity_deltas[base + 1u], 0)),
        f32(atomicExchange(&velocity_deltas[base + 2u], 0)),
    ) / FIXED_VELOCITY_SCALE;
    let angular_delta = vec3<f32>(
        f32(atomicExchange(&velocity_deltas[base + 3u], 0)),
        f32(atomicExchange(&velocity_deltas[base + 4u], 0)),
        f32(atomicExchange(&velocity_deltas[base + 5u], 0)),
    ) / FIXED_VELOCITY_SCALE;
    linear_velocities[body] = vec4<f32>(linear_velocities[body].xyz + linear_delta, 0.0);
    angular_velocities[body] = vec4<f32>(angular_velocities[body].xyz + angular_delta, 0.0);
}

// Adjacent driven coordinates share bodies, so solve their motor impulses
// serially before the parallel bearing projection. This prevents a wheel drive
// from erasing its steering drive's correction while retaining the fused
// workgroup path for the more numerous bearing rows. The shared accumulators
// still cap every motor to one tick of torque.
@compute @workgroup_size(256)
fn project_small_mechanism_velocities(
    @builtin(local_invocation_index) index: u32,
) {
    for (var iteration = 0u; iteration < max(config.solver_iterations, 1u); iteration += 1u) {
        if index == 0u {
            for (var drive_index = 0u; drive_index < config.bearing_count; drive_index += 1u) {
                project_drive_velocity_row_immediate(drive_index);
            }
        }
        storageBarrier();
        workgroupBarrier();

        if index < config.bearing_count {
            let bearing = drive_constraints[index].bearing;
            let body_a = bearing.metadata.x;
            let body_b = bearing.metadata.y;
            let arm_a = quat_rotate(rotations[body_a], bearing.local_anchor_a.xyz);
            let arm_b = quat_rotate(rotations[body_b], bearing.local_anchor_b.xyz);
            let anchor_velocity_a = linear_velocities[body_a].xyz
                + cross(angular_velocities[body_a].xyz, arm_a);
            let anchor_velocity_b = linear_velocities[body_b].xyz
                + cross(angular_velocities[body_b].xyz, arm_b);
            let relative_linear = anchor_velocity_b - anchor_velocity_a;
            solve_linear_axis(
                body_a,
                body_b,
                arm_a,
                arm_b,
                relative_linear,
                vec3<f32>(1.0, 0.0, 0.0),
            );
            solve_linear_axis(
                body_a,
                body_b,
                arm_a,
                arm_b,
                relative_linear,
                vec3<f32>(0.0, 1.0, 0.0),
            );
            solve_linear_axis(
                body_a,
                body_b,
                arm_a,
                arm_b,
                relative_linear,
                vec3<f32>(0.0, 0.0, 1.0),
            );

            let axis_a = normalize(quat_rotate(rotations[body_a], bearing.local_axis_a.xyz));
            let axis_b = normalize(quat_rotate(rotations[body_b], bearing.local_axis_b.xyz));
            let hinge_axis = normalize(axis_a + axis_b);
            let helper = select(
                vec3<f32>(1.0, 0.0, 0.0),
                vec3<f32>(0.0, 1.0, 0.0),
                abs(hinge_axis.x) > 0.8,
            );
            let tangent_a = normalize(cross(hinge_axis, helper));
            let tangent_b = cross(hinge_axis, tangent_a);
            let relative_angular = angular_velocities[body_b].xyz
                - angular_velocities[body_a].xyz;
            solve_angular_axis(body_a, body_b, relative_angular, tangent_a);
            solve_angular_axis(body_a, body_b, relative_angular, tangent_b);
        }
        storageBarrier();
        workgroupBarrier();

        if index < config.body_count {
            let base = index * 6u;
            let linear_delta = vec3<f32>(
                f32(atomicExchange(&velocity_deltas[base], 0)),
                f32(atomicExchange(&velocity_deltas[base + 1u], 0)),
                f32(atomicExchange(&velocity_deltas[base + 2u], 0)),
            ) / FIXED_VELOCITY_SCALE;
            let angular_delta = vec3<f32>(
                f32(atomicExchange(&velocity_deltas[base + 3u], 0)),
                f32(atomicExchange(&velocity_deltas[base + 4u], 0)),
                f32(atomicExchange(&velocity_deltas[base + 5u], 0)),
            ) / FIXED_VELOCITY_SCALE;
            linear_velocities[index] = vec4<f32>(
                linear_velocities[index].xyz + linear_delta,
                0.0,
            );
            angular_velocities[index] = vec4<f32>(
                angular_velocities[index].xyz + angular_delta,
                0.0,
            );
        }
        storageBarrier();
        workgroupBarrier();
    }
    if index == 0u {
        for (var cleanup = 0u; cleanup < SMALL_MECHANISM_SERIAL_CLEANUP_STEPS; cleanup += 1u) {
            for (var row = 0u; row < config.bearing_count; row += 1u) {
                project_drive_velocity_row_immediate(row);
                project_bearing_velocity_row(row);
            }
            for (var row = config.bearing_count; row > 0u; row -= 1u) {
                project_bearing_velocity_row(row - 1u);
            }
        }
    }
}

fn permitted_axis(body: u32) -> vec3<f32> {
    let mechanism = mechanism_bodies[body];
    let parent = mechanism.metadata.x;
    let bearing = bearings[mechanism.metadata.y];
    var axis = normalize(quat_rotate(rotations[parent], bearing.local_axis_a.xyz));
    if mechanism.metadata.z != 0u {
        axis = -normalize(quat_rotate(rotations[parent], bearing.local_axis_b.xyz));
    }
    return axis;
}

fn permitted_speed(body: u32) -> f32 {
    let mechanism = mechanism_bodies[body];
    let parent = mechanism.metadata.x;
    let axis = permitted_axis(body);
    return dot(angular_velocities[body].xyz - angular_velocities[parent].xyz, axis);
}

fn stabilized_speed(body: u32, speed: f32) -> f32 {
    let gravity_aligned = abs(permitted_axis(body).y) > 0.999;
    return select(
        speed,
        0.0,
        gravity_aligned && abs(speed) < GRAVITY_ALIGNED_BEARING_SLEEP_SPEED,
    );
}

@compute @workgroup_size(256)
fn advance_coordinates(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let body = invocation.x;
    if body >= config.body_count || mechanism_bodies[body].metadata.w != 0u {
        return;
    }
    let bearing = bearings[mechanism_bodies[body].metadata.y];
    let coordinate = bearing.metadata.z;
    if coordinate == INVALID_INDEX {
        return;
    }
    // Body integration has already applied the once-per-tick damping. Integrate
    // the joint velocity produced by the coupled constraint solve unchanged.
    let measured = permitted_speed(body);
    let drive = drives[coordinate];
    var speed = select(measured, stabilized_speed(body, measured), drive.mode == DRIVE_MODE_PASSIVE);
    var angle = coordinates[coordinate].angle + speed * config.delta_seconds;
    if angle < drive.min_angle {
        angle = drive.min_angle;
        speed = 0.0;
    } else if angle > drive.max_angle {
        angle = drive.max_angle;
        speed = 0.0;
    }
    coordinates[coordinate].angular_velocity = speed;
    coordinates[coordinate].angle = angle;
}

@compute @workgroup_size(256)
fn capture_coordinates(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let body = invocation.x;
    if body >= config.body_count || mechanism_bodies[body].metadata.w != 0u {
        return;
    }
    let coordinate = bearings[mechanism_bodies[body].metadata.y].metadata.z;
    if coordinate != INVALID_INDEX {
        // Contact impulses are captured at full strength and receive the normal
        // once-per-tick coordinate damping during the next advance.
        coordinates[coordinate].angular_velocity = stabilized_speed(body, permitted_speed(body));
    }
}

@compute @workgroup_size(1)
fn reconstruct_body_velocities() {
    for (var traversal = 0u; traversal < config.body_count; traversal += 1u) {
        let body = preorder[traversal];
        let mechanism = mechanism_bodies[body];
        if mechanism.metadata.w != 0u {
            continue;
        }
        let parent = mechanism.metadata.x;
        let bearing = bearings[mechanism.metadata.y];
        var axis = normalize(quat_rotate(rotations[parent], bearing.local_axis_a.xyz));
        var parent_anchor = quat_rotate(rotations[parent], bearing.local_anchor_a.xyz);
        var child_anchor = quat_rotate(rotations[body], bearing.local_anchor_b.xyz);
        if mechanism.metadata.z != 0u {
            axis = -normalize(quat_rotate(rotations[parent], bearing.local_axis_b.xyz));
            parent_anchor = quat_rotate(rotations[parent], bearing.local_anchor_b.xyz);
            child_anchor = quat_rotate(rotations[body], bearing.local_anchor_a.xyz);
        }
        let speed = coordinates[bearing.metadata.z].angular_velocity;
        let angular = angular_velocities[parent].xyz + axis * speed;
        let anchor_velocity = linear_velocities[parent].xyz
            + cross(angular_velocities[parent].xyz, parent_anchor);
        linear_velocities[body] = vec4<f32>(anchor_velocity - cross(angular, child_anchor), 0.0);
        angular_velocities[body] = vec4<f32>(angular, 0.0);
    }
}

@compute @workgroup_size(256)
fn validate_articulated_state(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let body = invocation.x;
    if body >= config.body_count {
        return;
    }
    let rotation_length = dot(rotations[body], rotations[body]);
    var valid = finite4(positions[body])
        && finite4(rotations[body])
        && finite4(linear_velocities[body])
        && finite4(angular_velocities[body])
        && rotation_length > 0.5
        && rotation_length < 1.5;
    let mechanism = mechanism_bodies[body];
    if mechanism.metadata.w == 0u {
        let coordinate = bearings[mechanism.metadata.y].metadata.z;
        let state = coordinates[coordinate];
        valid = valid
            && state.angle == state.angle
            && abs(state.angle) < 3.402823e+38
            && state.angular_velocity == state.angular_velocity
            && abs(state.angular_velocity) < 3.402823e+38;
    }
    if !valid {
        atomicOr(&diagnostics[0], INVALID_NUMERIC_FLAG);
    }
}
