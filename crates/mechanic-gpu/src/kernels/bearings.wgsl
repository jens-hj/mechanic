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

struct Bearing {
    local_anchor_a: vec4<f32>,
    local_anchor_b: vec4<f32>,
    local_axis_a: vec4<f32>,
    local_axis_b: vec4<f32>,
    metadata: vec4<u32>,
};

struct LinkState {
    position: vec4<f32>,
    rotation: vec4<f32>,
    metadata: vec4<u32>,
};

const CONSTRAINT_NON_CONVERGENCE_FLAG: u32 = 4u;

@group(0) @binding(0) var<uniform> config: TickConfig;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> rotations: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> diagnostics: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> bearings: array<Bearing>;
@group(0) @binding(5) var<storage, read> mechanism_links: array<LinkState>;

fn quat_rotate(rotation: vec4<f32>, vector: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(rotation.xyz, vector);
    return vector + rotation.w * t + cross(rotation.xyz, t);
}

fn record_residual(
    anchor_a: vec3<f32>,
    anchor_b: vec3<f32>,
    axis_a: vec3<f32>,
    axis_b: vec3<f32>,
    is_closure: bool,
) {
    let anchor_micrometers = u32(round(length(anchor_a - anchor_b) * 1000000.0));
    let axis_degrees = atan2(
        length(cross(axis_a, axis_b)),
        clamp(dot(axis_a, axis_b), -1.0, 1.0),
    ) * 57.295779513;
    let axis_microdegrees = u32(round(axis_degrees * 1000000.0));
    atomicMax(&diagnostics[3], anchor_micrometers);
    atomicMax(&diagnostics[4], axis_microdegrees);
    if is_closure && (anchor_micrometers > 10u || axis_microdegrees > 1000u) {
        atomicOr(&diagnostics[0], CONSTRAINT_NON_CONVERGENCE_FLAG);
    }
}

@compute @workgroup_size(256)
fn validate_bearings(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let index = invocation.x;
    if index >= config.bearing_count {
        return;
    }
    let bearing = bearings[index];
    let body_a = bearing.metadata.x;
    let body_b = bearing.metadata.y;
    let anchor_a = positions[body_a].xyz
        + quat_rotate(rotations[body_a], bearing.local_anchor_a.xyz);
    let anchor_b = positions[body_b].xyz
        + quat_rotate(rotations[body_b], bearing.local_anchor_b.xyz);
    let axis_a = normalize(quat_rotate(rotations[body_a], bearing.local_axis_a.xyz));
    let axis_b = normalize(quat_rotate(rotations[body_b], bearing.local_axis_b.xyz));
    record_residual(anchor_a, anchor_b, axis_a, axis_b, bearing.metadata.w != 0u);
}

@compute @workgroup_size(256)
fn validate_mechanism_bearings(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let index = invocation.x;
    if index >= config.bearing_count {
        return;
    }
    let bearing = bearings[index];
    let pose_a = mechanism_links[bearing.metadata.x];
    let pose_b = mechanism_links[bearing.metadata.y];
    let anchor_a = pose_a.position.xyz
        + quat_rotate(pose_a.rotation, bearing.local_anchor_a.xyz);
    let anchor_b = pose_b.position.xyz
        + quat_rotate(pose_b.rotation, bearing.local_anchor_b.xyz);
    let axis_a = normalize(quat_rotate(pose_a.rotation, bearing.local_axis_a.xyz));
    let axis_b = normalize(quat_rotate(pose_b.rotation, bearing.local_axis_b.xyz));
    record_residual(anchor_a, anchor_b, axis_a, axis_b, bearing.metadata.w != 0u);
}
