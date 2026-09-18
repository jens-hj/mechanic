@group(0) @binding(0) var<uniform> config: TickConfig;
@group(0) @binding(1) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> rotations: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> mechanism_bodies: array<MechanismBody>;
@group(0) @binding(4) var<storage, read> bearings: array<Bearing>;
@group(0) @binding(5) var<storage, read> coordinates: array<Coordinate>;
@group(0) @binding(6) var<storage, read_write> links_a: array<LinkState>;
@group(0) @binding(7) var<storage, read_write> links_b: array<LinkState>;

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

fn axis_rotation(axis: vec3<f32>, angle: f32) -> vec4<f32> {
    let half_angle = angle * 0.5;
    return vec4<f32>(normalize(axis) * sin(half_angle), cos(half_angle));
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn prepare_links(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let body = invocation.x;
    if body >= config.body_count {
        return;
    }
    let mechanism = mechanism_bodies[body];
    if mechanism.metadata.w != 0u {
        links_a[body].position = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        links_a[body].rotation = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        links_a[body].metadata = vec4<u32>(body, 1u, 0u, 0u);
        return;
    }

    let bearing = bearings[mechanism.metadata.y];
    let angle = coordinates[bearing.metadata.z].position;
    var relative_rotation = mechanism.bind_relative_rotation;
    var relative_position = mechanism.bind_relative_position.xyz;
    if bearing.local_axis_a.w == 1.0 {
        let displacement = clamp(angle, bearing.local_anchor_a.w, bearing.local_anchor_b.w);
        if mechanism.metadata.z == 0u {
            relative_position += bearing.local_axis_a.xyz * displacement;
        } else {
            relative_position -= bearing.local_axis_b.xyz * displacement;
        }
    } else if mechanism.metadata.z == 0u {
        relative_rotation = quat_multiply(
            axis_rotation(bearing.local_axis_a.xyz, angle),
            mechanism.bind_relative_rotation,
        );
        relative_position = bearing.local_anchor_a.xyz
            - quat_rotate(relative_rotation, bearing.local_anchor_b.xyz);
    } else {
        relative_rotation = quat_multiply(
            axis_rotation(bearing.local_axis_b.xyz, -angle),
            mechanism.bind_relative_rotation,
        );
        relative_position = bearing.local_anchor_b.xyz
            - quat_rotate(relative_rotation, bearing.local_anchor_a.xyz);
    }
    links_a[body].position = vec4<f32>(relative_position, 0.0);
    links_a[body].rotation = relative_rotation;
    links_a[body].metadata = vec4<u32>(mechanism.metadata.x, 0u, 0u, 0u);
}

fn compose(child: LinkState, parent: LinkState) -> LinkState {
    var result: LinkState;
    result.position = vec4<f32>(
        parent.position.xyz + quat_rotate(parent.rotation, child.position.xyz),
        0.0,
    );
    result.rotation = normalize(quat_multiply(parent.rotation, child.rotation));
    result.metadata = vec4<u32>(parent.metadata.x, 0u, 0u, 0u);
    return result;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn jump_a_to_b(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let body = invocation.x;
    if body >= config.body_count {
        return;
    }
    let child = links_a[body];
    if child.metadata.y != 0u {
        links_b[body] = child;
        return;
    }
    links_b[body] = compose(child, links_a[child.metadata.x]);
    links_b[body].metadata.y = links_a[child.metadata.x].metadata.y;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn jump_b_to_a(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let body = invocation.x;
    if body >= config.body_count {
        return;
    }
    let child = links_b[body];
    if child.metadata.y != 0u {
        links_a[body] = child;
        return;
    }
    links_a[body] = compose(child, links_b[child.metadata.x]);
    links_a[body].metadata.y = links_b[child.metadata.x].metadata.y;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn publish_a(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let body = invocation.x;
    if body >= config.body_count {
        return;
    }
    let link = links_a[body];
    let root = link.metadata.x;
    if body == root {
        return;
    }
    positions[body] = vec4<f32>(
        positions[root].xyz + quat_rotate(rotations[root], link.position.xyz),
        0.0,
    );
    rotations[body] = normalize(quat_multiply(rotations[root], link.rotation));
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn publish_b(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let body = invocation.x;
    if body >= config.body_count {
        return;
    }
    let link = links_b[body];
    let root = link.metadata.x;
    if body == root {
        return;
    }
    positions[body] = vec4<f32>(
        positions[root].xyz + quat_rotate(rotations[root], link.position.xyz),
        0.0,
    );
    rotations[body] = normalize(quat_multiply(rotations[root], link.rotation));
}
