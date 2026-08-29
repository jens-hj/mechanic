@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> rotations: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> snapshot_positions: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> snapshot_rotations: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> diagnostics: array<atomic<u32>>;

@compute @workgroup_size(256)
fn publish_snapshot(@builtin(global_invocation_id) invocation: vec3<u32>) {
    if atomicLoad(&diagnostics[0]) != 0u {
        return;
    }
    let body = invocation.x;
    if body >= arrayLength(&positions) {
        return;
    }
    snapshot_positions[body] = positions[body];
    snapshot_rotations[body] = rotations[body];
}
