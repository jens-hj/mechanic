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

struct ExternalImpulse {
    world_point: vec4<f32>,
    impulse: vec4<f32>,
    metadata: vec4<u32>,
};

const INVALID_NUMERIC_FLAG: u32 = 2u;

@group(0) @binding(0) var<uniform> config: TickConfig;
@group(0) @binding(1) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> rotations: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> linear_velocities: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> angular_velocities: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> inverse_masses: array<f32>;
@group(0) @binding(6) var<storage, read_write> error_flags: atomic<u32>;
@group(0) @binding(7) var<storage, read> masses: array<Mass>;
@group(0) @binding(8) var<uniform> external_impulse: ExternalImpulse;
@group(0) @binding(9) var<storage, read> mechanism_roots: array<u32>;

fn quat_multiply(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz),
        a.w * b.w - dot(a.xyz, b.xyz),
    );
}

fn quat_rotate(rotation: vec4<f32>, vector: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(rotation.xyz, vector);
    return vector + rotation.w * t + cross(rotation.xyz, t);
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

fn finite4(value: vec4<f32>) -> bool {
    return all(value == value) && all(abs(value) < vec4<f32>(3.402823e+38));
}

@compute @workgroup_size(1)
fn apply_external_impulse() {
    let body = external_impulse.metadata.x;
    let inverse_mass = masses[body].inverse_mass.x;
    if inverse_mass <= 0.0 {
        return;
    }
    let impulse = external_impulse.impulse.xyz;
    let arm = external_impulse.world_point.xyz - positions[body].xyz;
    linear_velocities[body] = vec4<f32>(
        linear_velocities[body].xyz + impulse * inverse_mass,
        0.0,
    );
    angular_velocities[body] = vec4<f32>(
        angular_velocities[body].xyz
            + world_inverse_inertia(body, cross(arm, impulse)),
        0.0,
    );
}

@compute @workgroup_size(256)
fn integrate(@builtin(global_invocation_id) invocation: vec3<u32>) {
    if atomicLoad(&error_flags) != 0u {
        return;
    }
    let index = invocation.x;
    if index >= config.body_count {
        return;
    }

    var position = positions[index];
    var rotation = rotations[index];
    var linear = linear_velocities[index];
    var angular = angular_velocities[index];
    if inverse_masses[index] > 0.0 {
        linear.y += config.gravity_y * config.delta_seconds;
        linear = vec4<f32>(linear.xyz * config.linear_damping, linear.w);
        angular = vec4<f32>(angular.xyz * config.angular_damping, angular.w);
        if mechanism_roots[index] != 0u {
            position = vec4<f32>(
                position.xyz + linear.xyz * config.delta_seconds,
                position.w,
            );
            let spin = vec4<f32>(angular.xyz, 0.0);
            rotation += quat_multiply(spin, rotation) * (0.5 * config.delta_seconds);
            let norm_squared = dot(rotation, rotation);
            if norm_squared > 1.0e-20 {
                rotation *= inverseSqrt(norm_squared);
            } else {
                rotation = vec4<f32>(0.0, 0.0, 0.0, 1.0);
                atomicOr(&error_flags, INVALID_NUMERIC_FLAG);
            }
        }
    }

    if !(finite4(position) && finite4(rotation) && finite4(linear) && finite4(angular)) {
        atomicOr(&error_flags, INVALID_NUMERIC_FLAG);
        return;
    }
    positions[index] = position;
    rotations[index] = rotation;
    linear_velocities[index] = linear;
    angular_velocities[index] = angular;
}
