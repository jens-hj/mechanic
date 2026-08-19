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

const FIXED_VELOCITY_SCALE: f32 = 1048576.0;
// Diagonal Jacobi rows share off-centre inertia terms and adjacent bodies.
// Under-relaxation keeps their simultaneous impulses dissipative.
const BEARING_PROJECTION_RELAXATION: f32 = 0.5;
const GRAVITY_ALIGNED_BEARING_SLEEP_SPEED: f32 = 0.005;
const INVALID_INDEX: u32 = 0xffffffffu;
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

@compute @workgroup_size(256)
fn project_bearing_velocities(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let index = invocation.x;
    if index >= config.bearing_count {
        return;
    }
    let bearing = bearings[index];
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
    // Constraint projection can feed world-space body motion back into a permitted
    // coordinate after body damping. Damp the authoritative joint speed here so
    // coupled passive bearings cannot retain a numerical limit cycle.
    let speed = stabilized_speed(body, permitted_speed(body) * config.angular_damping);
    coordinates[coordinate].angular_velocity = speed;
    coordinates[coordinate].angle += speed * config.delta_seconds;
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
