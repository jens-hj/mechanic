struct ExternalImpulse {
    world_point: vec4<f32>,
    impulse: vec4<f32>,
    metadata: vec4<u32>,
};

struct ExternalImpulseBatch {
    metadata: vec4<u32>,
    rows: array<ExternalImpulse, 64>,
};

const POWERED_LINEAR_DAMPING: f32 = 0.99999;
const POWERED_ANGULAR_DAMPING: f32 = 0.9999;

@group(0) @binding(0) var<uniform> config: TickConfig;
@group(0) @binding(1) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> rotations: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> linear_velocities: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> angular_velocities: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> inverse_masses: array<f32>;
@group(0) @binding(6) var<storage, read_write> diagnostics: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read> masses: array<Mass>;
@group(0) @binding(8) var<uniform> external_impulses: ExternalImpulseBatch;
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
    let count = min(external_impulses.metadata.x, 64u);
    for (var row_index = 0u; row_index < count; row_index += 1u) {
        let external_impulse = external_impulses.rows[row_index];
        let body = external_impulse.metadata.x;
        let inverse_mass = masses[body].inverse_mass.x;
        if inverse_mass <= 0.0 {
            continue;
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
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn integrate(@builtin(global_invocation_id) invocation: vec3<u32>) {
    if invocation.x == 0u { atomicOr(&diagnostics[8], 1u); }
    if atomicLoad(&diagnostics[0]) != 0u {
        return;
    }
    let index = invocation.x;
    if index >= config.body_count {
        return;
    }
    atomicAdd(&diagnostics[9], 1u);

    var position = positions[index];
    var rotation = rotations[index];
    var linear = linear_velocities[index];
    var angular = angular_velocities[index];
    if inverse_masses[index] > 0.0 {
        linear.y += config.gravity_y * config.delta_seconds;
        let mechanism_flags = mechanism_roots[index];
        let powered = (mechanism_flags & 2u) != 0u;
        let linear_damping = select(config.linear_damping, POWERED_LINEAR_DAMPING, powered);
        let angular_damping = select(config.angular_damping, POWERED_ANGULAR_DAMPING, powered);
        linear = vec4<f32>(linear.xyz * linear_damping, linear.w);
        angular = vec4<f32>(angular.xyz * angular_damping, angular.w);
        if (mechanism_flags & 1u) != 0u {
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
                atomicOr(&diagnostics[0], INVALID_NUMERIC_FLAG);
            }
        }
    }

    if !(finite4(position) && finite4(rotation) && finite4(linear) && finite4(angular)) {
        atomicOr(&diagnostics[0], INVALID_NUMERIC_FLAG);
        return;
    }
    positions[index] = position;
    rotations[index] = rotation;
    linear_velocities[index] = linear;
    angular_velocities[index] = angular;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn clear_position_corrections(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let body = invocation.x;
    if body >= config.body_count { return; }
    linear_velocities[body] = vec4<f32>(0.0);
    angular_velocities[body] = vec4<f32>(0.0);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn apply_position_correction(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let body = invocation.x;
    if body >= config.body_count || inverse_masses[body] <= 0.0
        || (mechanism_roots[body] & 1u) == 0u || atomicLoad(&diagnostics[0]) != 0u { return; }
    let position = positions[body] + vec4<f32>(linear_velocities[body].xyz * config.delta_seconds, 0.0);
    let spin = angular_velocities[body].xyz * config.delta_seconds;
    let angle = length(spin);
    var delta = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    if angle > 1.0e-8 { delta = vec4<f32>(spin * (sin(angle * 0.5) / angle), cos(angle * 0.5)); }
    let rotation = normalize(quat_multiply(delta, rotations[body]));
    if !(finite4(position) && finite4(rotation)) {
        atomicOr(&diagnostics[0], INVALID_NUMERIC_FLAG);
        return;
    }
    positions[body] = position;
    rotations[body] = rotation;
}
