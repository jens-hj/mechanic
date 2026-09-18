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
    bearing: Bearing,
    rotation_a: vec4<f32>,
    rotation_b: vec4<f32>,
) {
    var residual = anchor_a - anchor_b;
    if bearing.local_axis_a.w == 1.0 {
        let q = dot(anchor_b - anchor_a, axis_a);
        residual += axis_a * clamp(q, bearing.local_anchor_a.w, bearing.local_anchor_b.w);
    }
    let anchor_micrometers = u32(round(length(residual) * 1000000.0));
    var axis_degrees = atan2(
        length(cross(axis_a, axis_b)),
        clamp(dot(axis_a, axis_b), -1.0, 1.0),
    ) * 57.295779513;
    if bearing.local_axis_a.w == 1.0 {
        // Quaternion vector difference remains accurate near zero, unlike acos(dot).
        let aligned_b = select(-rotation_b, rotation_b, dot(rotation_a, rotation_b) >= 0.0);
        axis_degrees = 4.0 * asin(clamp(length(rotation_a - aligned_b) * 0.5, 0.0, 1.0)) * 57.295779513;
    }
    let axis_microdegrees = u32(round(axis_degrees * 1000000.0));
    atomicMax(&diagnostics[3], anchor_micrometers);
    atomicMax(&diagnostics[4], axis_microdegrees);
    if is_closure && (anchor_micrometers > 10u || axis_microdegrees > 1000u) {
        atomicOr(&diagnostics[0], CONSTRAINT_NON_CONVERGENCE_FLAG);
    }
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn validate_bearings(@builtin(global_invocation_id) invocation: vec3<u32>) {
    if invocation.x == 0u { atomicOr(&diagnostics[8], 32u); }
    let index = invocation.x;
    if index >= config.bearing_count {
        return;
    }
    atomicAdd(&diagnostics[11], 1u);
    let bearing = bearings[index];
    if (bearing.metadata.w & BEARING_SUSPENDED_FLAG) != 0u {
        return;
    }
    let body_a = bearing.metadata.x;
    let body_b = bearing.metadata.y;
    let anchor_a = positions[body_a].xyz
        + quat_rotate(rotations[body_a], bearing.local_anchor_a.xyz);
    let anchor_b = positions[body_b].xyz
        + quat_rotate(rotations[body_b], bearing.local_anchor_b.xyz);
    let axis_a = normalize(quat_rotate(rotations[body_a], bearing.local_axis_a.xyz));
    let axis_b = normalize(quat_rotate(rotations[body_b], bearing.local_axis_b.xyz));
    record_residual(anchor_a, anchor_b, axis_a, axis_b, (bearing.metadata.w & BEARING_CLOSURE_FLAG) != 0u, bearing, rotations[body_a], rotations[body_b]);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn validate_mechanism_bearings(@builtin(global_invocation_id) invocation: vec3<u32>) {
    if invocation.x == 0u { atomicOr(&diagnostics[8], 32u); }
    let index = invocation.x;
    if index >= config.bearing_count {
        return;
    }
    atomicAdd(&diagnostics[11], 1u);
    let bearing = bearings[index];
    if (bearing.metadata.w & BEARING_SUSPENDED_FLAG) != 0u {
        return;
    }
    let pose_a = mechanism_links[bearing.metadata.x];
    let pose_b = mechanism_links[bearing.metadata.y];
    let anchor_a = pose_a.position.xyz
        + quat_rotate(pose_a.rotation, bearing.local_anchor_a.xyz);
    let anchor_b = pose_b.position.xyz
        + quat_rotate(pose_b.rotation, bearing.local_anchor_b.xyz);
    let axis_a = normalize(quat_rotate(pose_a.rotation, bearing.local_axis_a.xyz));
    let axis_b = normalize(quat_rotate(pose_b.rotation, bearing.local_axis_b.xyz));
    record_residual(anchor_a, anchor_b, axis_a, axis_b, (bearing.metadata.w & BEARING_CLOSURE_FLAG) != 0u, bearing, pose_a.rotation, pose_b.rotation);
}
