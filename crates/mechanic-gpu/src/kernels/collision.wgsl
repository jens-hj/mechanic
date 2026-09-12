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

struct Collider {
    local_center: vec4<f32>,
    local_rotation: vec4<f32>,
    half_extents: vec4<f32>,
    metadata: vec4<u32>,
    surface_response: vec4<f32>,
    surface_elasticity: vec4<f32>,
    // shape kind, convex-buffer offset, packed element counts, and
    // nonzero when the owning body can never move.
    shape: vec4<u32>,
};

struct Interval {
    minimum: f32,
    maximum: f32,
};

struct Contact {
    metadata: vec4<u32>,
    normal_penetration: vec4<f32>,
    arm_a_impulse: vec4<f32>,
    arm_b: vec4<f32>,
};

struct PersistentManifold {
    pair_tick: vec4<u32>,
    normal_penetration: vec4<f32>,
    point_impulse: vec4<f32>,
    tangent_rolling_impulses: vec4<f32>,
};

struct GroundSurface {
    response: vec4<f32>,
    elasticity: vec4<f32>,
    plane: vec4<f32>,
};

struct TangentBasis {
    u: vec3<f32>,
    v: vec3<f32>,
};

struct ContactImpulse {
    linear: vec3<f32>,
    rolling: vec3<f32>,
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
    suspension: vec4<f32>,
    bump_stop: vec4<f32>,
    metadata: vec4<u32>,
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

struct BearingProjectionFrame {
    arm_a: vec3<f32>,
    arm_b: vec3<f32>,
    tangent_a: vec3<f32>,
    tangent_b: vec3<f32>,
    // For the hinge matrix [A B; B^T D], cache A^-1, A^-1 B,
    // and (D - B^T A^-1 B)^-1 once per tick.
    inverse_linear: mat3x3<f32>,
    linear_angular: mat2x3<f32>,
    inverse_angular: mat2x2<f32>,
    drive_inverse_inertia: f32,
};

var<private> bearing_projection_frames: array<BearingProjectionFrame, 64>;
var<private> bearing_parent_rows: array<u32, 64>;

struct WorldMass {
    position: vec4<f32>,
    inverse_inertia_x_mass: vec4<f32>,
    inverse_inertia_y: vec4<f32>,
    inverse_inertia_z: vec4<f32>,
};

struct SatResult {
    normal: vec3<f32>,
    penetration: f32,
    near_face_axes: u32,
};

const PAIR_OVERFLOW_FLAG: u32 = 1u;
const MANIFOLD_OVERFLOW_FLAG: u32 = 8u;
const MAX_HASH_PROBES: u32 = 96u;
const EMPTY_HASH_KEY: u32 = 0u;
const FIXED_VELOCITY_SCALE: f32 = 1048576.0;
// Per-body counts follow the eight-u32 GpuDiagnostics readback header.
const BODY_CONTACT_COUNT_OFFSET: u32 = 12u;
const PROJECTED_RELAXATION: f32 = 0.125;
const WARM_START_SCALE: f32 = 0.5;
const MAX_ROLLING_RESISTANCE: f32 = 0.04;
const MIN_GAMMA_LOG2: f32 = -28.0;
const MAX_GAMMA_LOG2: f32 = -8.0;
const RESTITUTION_SPEED_THRESHOLD: f32 = 1.0;
const PENETRATION_SLOP: f32 = 0.001;
const MAX_PENETRATION_CORRECTION_SPEED: f32 = 1.0;
const MAX_ANALYTIC_CYLINDER_CORRECTION_SPEED: f32 = 4.0;
const CACHED_NORMAL_ALIGNMENT: f32 = 0.98;
const MAX_CACHED_POINT_MOVEMENT: f32 = 0.02;
const CYLINDER_MANIFOLD_ALIGNMENT: f32 = 0.05;
const MAX_SORTED_SERIAL_CONTACTS: u32 = 64u;
const INVALID_MANIFOLD_SLOT: u32 = 0xffffffffu;
const MAX_MANIFOLD_PROBES: u32 = 256u;
const ANALYTIC_CYLINDER_FLAG: u32 = 0x80000000u;
const CYLINDER_FACE_PAIR_FLAG: u32 = 0x40000000u;
const TERRAIN_CONTACT_FLAG: u32 = 0x20000000u;
const CONTACT_FLAG_MASK: u32 = ANALYTIC_CYLINDER_FLAG | CYLINDER_FACE_PAIR_FLAG | TERRAIN_CONTACT_FLAG;
const DRIVE_MODE_PASSIVE: u32 = 0u;
const DRIVE_MODE_ANGLE: u32 = 2u;
const DRIVE_ANGLE_POSITION_GAIN: f32 = 6.0;
const DRIVE_ANGLE_BRAKE_MARGIN: f32 = 0.8;
const DRIVE_ANGLE_DEADBAND: f32 = 0.0005;
// A full cylinder has sixteen overlapping sector rows. Face landings therefore
// need one sixteenth of the ordinary per-contact Jacobi correction.
const CYLINDER_FACE_RELAXATION_SCALE: f32 = 0.0625;
const COLLIDER_SHAPE_CUBOID: u32 = 0u;
const COLLIDER_SHAPE_CONVEX: u32 = 1u;

fn mixed_collider_response(collider_a: u32, collider_b: u32) -> vec4<f32> {
    let first = colliders[collider_a].surface_response;
    let second = colliders[collider_b].surface_response;
    return vec4<f32>(
        sqrt(first.x * second.x),
        sqrt(first.y * second.y),
        max(first.z, second.z),
        sqrt(first.w * second.w),
    );
}

fn mixed_ground_response(collider: u32) -> vec4<f32> {
    let first = colliders[collider].surface_response;
    let ground = ground_surfaces[collider].response;
    return vec4<f32>(
        sqrt(first.x * ground.x),
        sqrt(first.y * ground.y),
        max(first.z, ground.z),
        sqrt(first.w * ground.w),
    );
}

fn pack_raw_surface_response(response: vec4<f32>) -> f32 {
    let normalized = vec4<f32>(response.xyz, response.w / MAX_ROLLING_RESISTANCE);
    return bitcast<f32>(pack4x8unorm(clamp(normalized, vec4<f32>(0.0), vec4<f32>(1.0))));
}

fn unpack_raw_surface_response(value: f32) -> vec4<f32> {
    let normalized = unpack4x8unorm(bitcast<u32>(value));
    return vec4<f32>(normalized.xyz, normalized.w * MAX_ROLLING_RESISTANCE);
}

fn pack_prepared_surface_response(response: vec4<f32>, gamma: f32) -> f32 {
    let gamma_normalized = clamp(
        (log2(max(gamma, exp2(MIN_GAMMA_LOG2))) - MIN_GAMMA_LOG2)
            / (MAX_GAMMA_LOG2 - MIN_GAMMA_LOG2),
        0.0,
        1.0,
    );
    return bitcast<f32>(pack4x8unorm(vec4<f32>(
        response.x,
        response.y,
        response.w / MAX_ROLLING_RESISTANCE,
        gamma_normalized,
    )));
}

fn unpack_prepared_surface_response(value: f32) -> vec4<f32> {
    let normalized = unpack4x8unorm(bitcast<u32>(value));
    let gamma = exp2(mix(MIN_GAMMA_LOG2, MAX_GAMMA_LOG2, normalized.w));
    return vec4<f32>(normalized.xy, normalized.z * MAX_ROLLING_RESISTANCE, gamma);
}

fn is_analytic_cylinder(contact: Contact) -> bool {
    return (contact.metadata.z & ANALYTIC_CYLINDER_FLAG) != 0u;
}

fn is_cylinder_face_pair(contact: Contact) -> bool {
    return (contact.metadata.z & CYLINDER_FACE_PAIR_FLAG) != 0u;
}

fn tangent_basis(normal: vec3<f32>) -> TangentBasis {
    let reference = select(
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(1.0, 0.0, 0.0),
        abs(normal.y) > 0.9,
    );
    let u = normalize(cross(reference, normal));
    return TangentBasis(u, cross(normal, u));
}

@group(0) @binding(0) var<uniform> config: TickConfig;
@group(0) @binding(1) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> rotations: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> linear_velocities: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> diagnostics: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read> colliders: array<Collider>;
@group(0) @binding(7) var<storage, read_write> hash_keys: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> hash_values: array<u32>;
@group(0) @binding(9) var<storage, read_write> pairs: array<vec2<u32>>;
@group(0) @binding(10) var<storage, read_write> contacts: array<Contact>;
@group(0) @binding(11) var<storage, read> suppressed_pairs: array<vec2<u32>>;
@group(0) @binding(12) var<storage, read_write> indirect_args: array<u32>;
@group(0) @binding(13) var<storage, read_write> velocity_deltas: array<atomic<i32>>;
@group(0) @binding(14) var<storage, read_write> manifold_keys: array<atomic<u32>>;
@group(0) @binding(15) var<storage, read_write> persistent_manifolds: array<PersistentManifold>;
@group(0) @binding(16) var<storage, read_write> active_contacts: array<u32>;
@group(0) @binding(24) var<storage, read> masses: array<Mass>;
@group(0) @binding(25) var<storage, read_write> angular_velocities: array<vec4<f32>>;
@group(0) @binding(26) var<storage, read_write> world_masses: array<WorldMass>;
@group(0) @binding(27) var<storage, read> body_components: array<u32>;
@group(0) @binding(28) var<storage, read> convex_shapes: array<vec4<f32>>;
@group(0) @binding(29) var<storage, read> ground_surfaces: array<GroundSurface>;
@group(0) @binding(30) var<storage, read_write> drive_constraints: array<DriveConstraint>;

struct TerrainRow {
    minimum: vec4<f32>,
    maximum: vec4<f32>,
    first: vec4<f32>,
    second: vec4<f32>,
    third: vec4<f32>,
    response: vec4<f32>,
    metadata: vec4<u32>,
};

@group(0) @binding(31) var<storage, read> terrain_rows: array<TerrainRow>;

struct TerrainPreviousPose {
    position: vec4<f32>,
    rotation: vec4<f32>,
};
@group(0) @binding(32) var<storage, read_write> terrain_previous_poses: array<TerrainPreviousPose>;

@group(0) @binding(33) var<storage, read_write> terrain_sweep_fractions: array<atomic<u32>>;
@group(0) @binding(34) var<storage, read> terrain_free_bodies: array<u32>;
@group(0) @binding(35) var<storage, read_write> terrain_recovery_dispatch: array<u32>;
@group(0) @binding(36) var<storage, read> recovery_contacts: array<u32>;

@compute @workgroup_size(256)
fn capture_terrain_poses(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let body = invocation.x;
    if body >= config.body_count { return; }
    terrain_previous_poses[body] = TerrainPreviousPose(positions[body], rotations[body]);
    atomicStore(&terrain_sweep_fractions[body], bitcast<u32>(1.0));
}

// Reconstruct the isolated root's normalized Euler rotation at fractional time.
// Rescaling the endpoint removes its normalization before interpolation; simple
// nlerp would change rotation timing relative to simultaneous translation.
fn terrain_rotation_at(body: u32, fraction: f32) -> vec4<f32> {
    let previous = terrain_previous_poses[body].rotation;
    let current = select(rotations[body], -rotations[body], dot(previous, rotations[body]) < 0.0);
    let cosine = max(dot(previous, current), 1.0e-12);
    return normalize(mix(previous, current / cosine, fraction));
}

fn terrain_gap_axis(collider: u32, triangle: TerrainRow, raw_axis: vec3<f32>, previous: vec4<f32>) -> vec4<f32> {
    let squared = dot(raw_axis, raw_axis);
    if squared < 1.0e-12 { return previous; }
    let axis = raw_axis * inverseSqrt(squared);
    let interval = project_collider(collider, axis);
    let a = dot(axis, triangle.first.xyz);
    let b = dot(axis, triangle.second.xyz);
    let c = dot(axis, triangle.third.xyz);
    let gap = max(min(a, min(b, c)) - interval.maximum, interval.minimum - max(a, max(b, c)));
    if gap > previous.w { return vec4<f32>(axis, gap); }
    return previous;
}

// Evaluate full finite-triangle SAT at a candidate pose by transforming the
// triangle into the predicted collider frame, reusing all collider families.
fn terrain_gap_at(collider: u32, triangle: TerrainRow, fraction: f32) -> vec4<f32> {
    let body = colliders[collider].metadata.x;
    let rotation = terrain_rotation_at(body, fraction);
    let transform = quat_multiply(rotations[body], vec4<f32>(-rotation.xyz, rotation.w));
    let position = mix(terrain_previous_poses[body].position.xyz, positions[body].xyz, fraction);
    var local = triangle;
    local.first = vec4<f32>(positions[body].xyz + quat_rotate(transform, triangle.first.xyz - position), 0.0);
    local.second = vec4<f32>(positions[body].xyz + quat_rotate(transform, triangle.second.xyz - position), 0.0);
    local.third = vec4<f32>(positions[body].xyz + quat_rotate(transform, triangle.third.xyz - position), 0.0);
    let edges = array<vec3<f32>, 3>(local.second.xyz - local.first.xyz,
        local.third.xyz - local.second.xyz, local.first.xyz - local.third.xyz);
    var gap = terrain_gap_axis(collider, local, cross(edges[0], edges[1]), vec4<f32>(0.0, 0.0, 0.0, -1.0e30));
    for (var face = 0u; face < collider_face_axis_count(collider); face += 1u) {
        gap = terrain_gap_axis(collider, local, collider_face_axis(collider, face), gap);
    }
    for (var edge = 0u; edge < collider_edge_axis_count(collider); edge += 1u) {
        for (var side = 0u; side < 3u; side += 1u) {
            gap = terrain_gap_axis(collider, local, cross(collider_edge_axis(collider, edge), edges[side]), gap);
        }
    }
    return vec4<f32>(quat_rotate(vec4<f32>(-transform.xyz, transform.w), gap.xyz), gap.w);
}

fn terrain_rotational_toi(collider: u32, triangle: TerrainRow, motion: vec3<f32>,
    spin_axis: vec3<f32>, angular_bound: f32, radius: f32) -> f32 {
    // Endpoint intersections already enter the discrete manifold and split
    // recovery. Search here only for crossings that path would otherwise miss.
    if terrain_gap_at(collider, triangle, 1.0).w <= 1.0e-6 { return 1.0; }
    var fraction = 0.0;
    var gap = terrain_gap_at(collider, triangle, fraction);
    // Existing overlaps remain the responsibility of split positional recovery.
    // Clamping those would prevent supported rolling and ordinary contact motion.
    if gap.w <= 1.0e-6 { return 1.0; }
    let total_bound = length(motion) + angular_bound * radius;
    for (var iteration = 0u; iteration < 256u; iteration += 1u) {
        if gap.w <= 2.0e-7 {
            return min(1.0, fraction + 1.0e-5 / max(total_bound, 1.0e-8));
        }
        // This separating axis is held fixed for the bound. Rotation around a
        // parallel axis cannot close its gap (important for spinning cylinders).
        let projection = dot(spin_axis, gap.xyz);
        let speed = abs(dot(motion, gap.xyz))
            + angular_bound * radius * sqrt(max(1.0 - projection * projection, 0.0));
        if speed <= 1.0e-10 { return 1.0; }
        let next = fraction + max(gap.w - 1.0e-7, 0.0) / speed;
        if next >= 1.0 { return 1.0; }
        if next <= fraction { return fraction; }
        fraction = next;
        gap = terrain_gap_at(collider, triangle, fraction);
    }
    // A bounded search may conservatively shorten motion, but must never turn
    // an unresolved interval into permission to cross the surface.
    return fraction;
}

// Triangle BVHs stay chunk-local across floating-origin shifts. Only the
// placement row changes; cached triangle identities remain stable allocations.
fn terrain_row(index: u32) -> TerrainRow {
    var row = terrain_rows[index];
    let shift = terrain_rows[row.metadata.z].first.xyz;
    row.minimum = vec4<f32>(row.minimum.xyz + shift, row.minimum.w);
    row.maximum = vec4<f32>(row.maximum.xyz + shift, row.maximum.w);
    row.first = vec4<f32>(row.first.xyz + shift, row.first.w);
    row.second = vec4<f32>(row.second.xyz + shift, row.second.w);
    row.third = vec4<f32>(row.third.xyz + shift, row.third.w);
    return row;
}

// Stackless top-level traversal returns the next intersecting chunk's range.
fn next_terrain_chunk(minimum: vec3<f32>, maximum: vec3<f32>, start: u32) -> vec3<u32> {
    var row = start;
    let end = terrain_rows[0].metadata.y;
    while row < end {
        let node = terrain_rows[row];
        if any(maximum < node.minimum.xyz) || any(minimum > node.maximum.xyz) {
            row = node.metadata.x;
            continue;
        }
        row += 1u;
        if node.metadata.y == 2u { return vec3<u32>(row, node.metadata.zw); }
    }
    return vec3<u32>(end, 0u, 0u);
}

@compute @workgroup_size(256)
fn sweep_rotating_terrain_colliders(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let collider = invocation.x;
    if collider >= config.collider_count { return; }
    let body = colliders[collider].metadata.x;
    if terrain_free_bodies[body] == 0u { return; }
    let previous = terrain_previous_poses[body];
    let current = select(rotations[body], -rotations[body], dot(previous.rotation, rotations[body]) < 0.0);
    let cosine = max(dot(previous.rotation, current), 1.0e-12);
    let angular_bound = 2.0 * length(current - previous.rotation * cosine) / cosine;
    var radius = length(colliders[collider].local_center.xyz) + length(colliders[collider].half_extents.xyz);
    if collider_is_convex(collider) {
        radius = 0.0;
        for (var vertex = 0u; vertex < convex_vertex_count(collider); vertex += 1u) {
            radius = max(radius, distance(convex_vertex(collider, vertex), positions[body].xyz));
        }
    }
    if angular_bound * radius < 1.0e-4 { return; }
    let delta = quat_multiply(current, vec4<f32>(-previous.rotation.xyz, previous.rotation.w));
    let spin_axis = normalize(delta.xyz);
    let motion = positions[body].xyz - previous.position.xyz;
    let sphere_minimum = min(previous.position.xyz, positions[body].xyz) - vec3<f32>(radius);
    let sphere_maximum = max(previous.position.xyz, positions[body].xyz) + vec3<f32>(radius);
    let x = project_collider(collider, vec3<f32>(1.0, 0.0, 0.0));
    let y = project_collider(collider, vec3<f32>(0.0, 1.0, 0.0));
    let z = project_collider(collider, vec3<f32>(0.0, 0.0, 1.0));
    let low = vec3<f32>(x.minimum, y.minimum, z.minimum);
    let high = vec3<f32>(x.maximum, y.maximum, z.maximum);
    // Both bounds enclose the complete sweep. Intersect them so ordinary small
    // rotations do not visit the full circumscribed sphere's triangle set.
    let angular_margin = vec3<f32>(angular_bound * radius);
    let minimum = max(sphere_minimum, min(low, low - motion) - angular_margin);
    let maximum = min(sphere_maximum, max(high, high - motion) + angular_margin);
    var row = 0u;
    var fraction = 1.0;
    var next_chunk = terrain_rows[0].metadata.x;
    var chunk_end = 0u;
    loop {
        if row >= chunk_end {
            let chunk = next_terrain_chunk(minimum, maximum, next_chunk);
            next_chunk = chunk.x;
            row = chunk.y;
            chunk_end = chunk.z;
            if chunk_end == 0u { break; }
        }
        let triangle = terrain_row(row);
        if any(maximum < triangle.minimum.xyz) || any(minimum > triangle.maximum.xyz) {
            row = triangle.metadata.x;
            continue;
        }
        row += 1u;
        if triangle.metadata.y == 0u { continue; }
        let normal = cross(triangle.second.xyz - triangle.first.xyz, triangle.third.xyz - triangle.first.xyz);
        if dot(normal, normal) < 1.0e-12 { continue; }
        fraction = min(fraction, terrain_rotational_toi(collider, triangle, motion, spin_axis, angular_bound, radius));
    }
    atomicMin(&terrain_sweep_fractions[body], bitcast<u32>(fraction));
}

@compute @workgroup_size(256)
fn apply_terrain_sweep(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let body = invocation.x;
    if body >= config.body_count { return; }
    let fraction = bitcast<f32>(atomicLoad(&terrain_sweep_fractions[body]));
    if fraction >= 1.0 { return; }
    rotations[body] = terrain_rotation_at(body, fraction);
    positions[body] = vec4<f32>(mix(terrain_previous_poses[body].position.xyz, positions[body].xyz, fraction), positions[body].w);
}

// Continuous SAT for translation. All axes use the current collider rotation;
// Angular sweep is not yet covered by this translational TOI.
fn terrain_sweep_axis(collider: u32, triangle: TerrainRow, raw_axis: vec3<f32>, motion: vec3<f32>, window: vec2<f32>) -> vec2<f32> {
    let length_squared = dot(raw_axis, raw_axis);
    if length_squared < 1.0e-12 { return window; }
    let axis = raw_axis * inverseSqrt(length_squared);
    let interval = project_collider(collider, axis);
    let speed = dot(axis, motion);
    let a = dot(axis, triangle.first.xyz);
    let b = dot(axis, triangle.second.xyz);
    let c = dot(axis, triangle.third.xyz);
    let low = min(a, min(b, c)) - (interval.maximum - speed);
    let high = max(a, max(b, c)) - (interval.minimum - speed);
    if abs(speed) < 1.0e-9 {
        if low > 1.0e-6 || high < -1.0e-6 { return vec2<f32>(2.0, -1.0); }
        return window;
    }
    let first = low / speed;
    let last = high / speed;
    return vec2<f32>(max(window.x, min(first, last)), min(window.y, max(first, last)));
}

fn terrain_translation_toi(collider: u32, triangle: TerrainRow, motion: vec3<f32>) -> vec2<f32> {
    let edges = array<vec3<f32>, 3>(triangle.second.xyz - triangle.first.xyz,
        triangle.third.xyz - triangle.second.xyz, triangle.first.xyz - triangle.third.xyz);
    var window = terrain_sweep_axis(collider, triangle, cross(edges[0], edges[1]), motion, vec2<f32>(0.0, 1.0));
    for (var face = 0u; face < collider_face_axis_count(collider); face += 1u) {
        if window.x > window.y { return window; }
        window = terrain_sweep_axis(collider, triangle, collider_face_axis(collider, face), motion, window);
    }
    for (var edge = 0u; edge < collider_edge_axis_count(collider); edge += 1u) {
        for (var triangle_edge = 0u; triangle_edge < 3u; triangle_edge += 1u) {
            if window.x > window.y { return window; }
            window = terrain_sweep_axis(collider, triangle, cross(collider_edge_axis(collider, edge), edges[triangle_edge]), motion, window);
        }
    }
    return window;
}

// Clip the finite triangle to the collider. A plane support test alone invents
// contacts beyond triangle edges, especially for long thin boxes.
struct TerrainPatch {
    points: array<vec3<f32>, 4>,
    count: u32,
};

fn terrain_contact_polygon(collider: u32, triangle: TerrainRow, shift: vec3<f32>) -> TerrainPatch {
    var manifold: TerrainPatch;
    var polygon: array<vec3<f32>, 32>;
    var scratch: array<vec3<f32>, 32>;
    polygon[0] = triangle.first.xyz;
    polygon[1] = triangle.second.xyz;
    polygon[2] = triangle.third.xyz;
    var count = 3u;
    let plane_count = select(6u, convex_face_count(collider), collider_is_convex(collider));
    for (var face = 0u; face < plane_count; face += 1u) {
        var plane: vec4<f32>;
        if collider_is_convex(collider) {
            plane = convex_face_plane(collider, face);
        } else {
            let axis = collider_face_axis(collider, face / 2u) * select(-1.0, 1.0, (face & 1u) == 1u);
            plane = vec4<f32>(axis, project_collider(collider, axis).maximum);
        }
        plane.w += dot(plane.xyz, shift) + 1.0e-6;
        var output = 0u;
        for (var vertex = 0u; vertex < count; vertex += 1u) {
            let first = polygon[vertex];
            let second = polygon[(vertex + 1u) % count];
            let a = dot(plane.xyz, first) - plane.w;
            let b = dot(plane.xyz, second) - plane.w;
            if a <= 0.0 {
                if output >= 32u {
                    atomicOr(&diagnostics[0], PAIR_OVERFLOW_FLAG);
                    return manifold;
                }
                scratch[output] = first;
                output += 1u;
            }
            if (a < 0.0 && b > 0.0) || (a > 0.0 && b < 0.0) {
                if output >= 32u {
                    atomicOr(&diagnostics[0], PAIR_OVERFLOW_FLAG);
                    return manifold;
                }
                scratch[output] = mix(first, second, a / (a - b));
                output += 1u;
            }
        }
        count = output;
        if count == 0u { return manifold; }
        for (var vertex = 0u; vertex < count; vertex += 1u) {
            polygon[vertex] = scratch[vertex];
        }
    }
    let raw_normal = cross(triangle.second.xyz - triangle.first.xyz, triangle.third.xyz - triangle.first.xyz);
    if dot(raw_normal, raw_normal) < 1.0e-12 { return manifold; }
    let basis = tangent_basis(normalize(raw_normal));
    let directions = array<vec3<f32>, 4>(basis.u + basis.v, basis.u - basis.v, -basis.u - basis.v, -basis.u + basis.v);
    for (var corner = 0u; corner < 4u; corner += 1u) {
        var selected = polygon[0];
        var score = dot(selected, directions[corner]);
        for (var vertex = 1u; vertex < count; vertex += 1u) {
            let candidate = dot(polygon[vertex], directions[corner]);
            if candidate > score { selected = polygon[vertex]; score = candidate; }
        }
        var unique = true;
        for (var previous = 0u; previous < manifold.count; previous += 1u) {
            if distance(selected, manifold.points[previous]) < 1.0e-5 { unique = false; }
        }
        if unique { manifold.points[manifold.count] = selected; manifold.count += 1u; }
    }
    return manifold;
}

// Each retained point keeps its source triangle and corner for geometry refresh.
struct TerrainSupport {
    point: vec3<f32>,
    identity: u32,
    shift: vec3<f32>,
};

struct TerrainSupportPlane {
    plane: vec4<f32>,
    response: vec4<f32>,
    supports: array<TerrainSupport, 5>,
    curved: u32,
};

// Measure separation at the retained finite point against the opposing collider
// face. Whole-hull support against each triangle's infinite plane overestimates
// depth on curved terrain, lifting a resting body above the real surface.
fn terrain_opposing_plane(collider: u32, normal: vec3<f32>) -> vec4<f32> {
    let count = select(6u, convex_face_count(collider), collider_is_convex(collider));
    var alignment = 0.0;
    var selected = vec4<f32>(0.0);
    for (var face = 0u; face < count; face += 1u) {
        var plane: vec4<f32>;
        if collider_is_convex(collider) {
            plane = convex_face_plane(collider, face);
        } else {
            let axis = collider_face_axis(collider, face / 2u) * select(-1.0, 1.0, (face & 1u) == 1u);
            plane = vec4<f32>(axis, project_collider(collider, axis).maximum);
        }
        let candidate = dot(plane.xyz, normal);
        if candidate < alignment {
            alignment = candidate;
            selected = plane;
        }
    }
    return selected;
}

fn terrain_point_depth(collider: u32, point: vec3<f32>, normal: vec3<f32>) -> f32 {
    let plane = terrain_opposing_plane(collider, normal);
    return max((dot(plane.xyz, point) - plane.w) / dot(plane.xyz, normal), 0.0);
}

fn emit_terrain_support(collider_index: u32, normal: vec3<f32>,
    response: vec4<f32>, support: TerrainSupport) {
    let depth = terrain_point_depth(collider_index, support.point, normal);
    let output = atomicAdd(&diagnostics[2], 1u);
    if output >= config.pair_capacity {
        atomicOr(&diagnostics[0], PAIR_OVERFLOW_FLAG);
        return;
    }
    let collider = colliders[collider_index];
    contacts[output].metadata = vec4<u32>(collider.metadata.x, INVALID_MANIFOLD_SLOT,
        collider_index | TERRAIN_CONTACT_FLAG, support.identity | 0x80000000u);
    contacts[output].normal_penetration = vec4<f32>(-normal, depth);
    contacts[output].arm_a_impulse = vec4<f32>(support.point - positions[collider.metadata.x].xyz - support.shift, collider.surface_elasticity.x);
    contacts[output].arm_b = vec4<f32>(depth, 0.0, 0.0, pack_raw_surface_response(response));
}

@compute @workgroup_size(256)
fn generate_terrain_contacts(@builtin(global_invocation_id) invocation: vec3<u32>) {
    if invocation.x == 0u { atomicOr(&diagnostics[8], 128u); }
    let collider_index = invocation.x;
    if collider_index >= config.collider_count { return; }
    let collider = colliders[collider_index];
    // A terrain contact carries no second body, so prepare_contacts discards
    // every one whose own body cannot move. Generating them buys nothing and
    // costs a full BVH descent plus a contact slot that can overflow the buffer.
    if collider.shape.w != 0u { return; }
    let x = project_collider(collider_index, vec3<f32>(1.0, 0.0, 0.0));
    let y = project_collider(collider_index, vec3<f32>(0.0, 1.0, 0.0));
    let z = project_collider(collider_index, vec3<f32>(0.0, 0.0, 1.0));
    let motion = positions[collider.metadata.x].xyz - terrain_previous_poses[collider.metadata.x].position.xyz;
    let current_minimum = vec3<f32>(x.minimum, y.minimum, z.minimum);
    let current_maximum = vec3<f32>(x.maximum, y.maximum, z.maximum);
    let minimum = min(current_minimum, current_minimum - motion);
    let maximum = max(current_maximum, current_maximum - motion);
    // Reduce nearby equal-response surface normals while retaining the crown
    // and four outer supports. Each support uses its original triangle normal.
    // Extra groups use the unreduced path, never silent geometry truncation.
    var planes: array<TerrainSupportPlane, 16>;
    var plane_count = 0u;
    var row_index = 0u;
    var next_chunk = terrain_rows[0].metadata.x;
    var chunk_end = 0u;
    loop {
        if row_index >= chunk_end {
            let chunk = next_terrain_chunk(minimum, maximum, next_chunk);
            next_chunk = chunk.x;
            row_index = chunk.y;
            chunk_end = chunk.z;
            if chunk_end == 0u { break; }
        }
        let triangle = terrain_row(row_index);
        if any(maximum < triangle.minimum.xyz) || any(minimum > triangle.maximum.xyz) {
            row_index = triangle.metadata.x;
            continue;
        }
        let identity = row_index;
        row_index += 1u;
        if triangle.metadata.y == 0u { continue; }
        var shift = vec3<f32>(0.0);
        var manifold = terrain_contact_polygon(collider_index, triangle, shift);
        if manifold.count == 0u {
            let window = terrain_translation_toi(collider_index, triangle, motion);
            if window.x > window.y { continue; }
            shift = motion * (window.x - 1.0);
            manifold = terrain_contact_polygon(collider_index, triangle, shift);
        }
        if manifold.count == 0u { continue; }
        let raw_normal = cross(triangle.second.xyz - triangle.first.xyz, triangle.third.xyz - triangle.first.xyz);
        if dot(raw_normal, raw_normal) < 1.0e-12 { continue; }
        let normal = normalize(raw_normal);
        let depth = dot(normal, triangle.first.xyz) - project_collider(collider_index, normal).minimum;
        if depth < 0.0 { continue; }
        let first = collider.surface_response;
        let second = triangle.response;
        let response = vec4<f32>(sqrt(first.x * second.x), sqrt(first.y * second.y), max(first.z, second.z), sqrt(first.w * second.w));
        let plane_distance = dot(normal, triangle.first.xyz);
        var group = plane_count;
        for (var candidate = 0u; candidate < plane_count; candidate += 1u) {
            let reference = planes[candidate].plane;
            let parallel = all(abs(reference.xyz - normal) < vec3<f32>(1.0e-6));
            let separation = abs(dot(normal - reference.xyz, collider_center(collider_index))
                - plane_distance + reference.w);
            let nearby = !parallel && dot(reference.xyz, normal) > 0.995 && separation < 0.025;
            if ((parallel && abs(reference.w - plane_distance) < 1.0e-5) || nearby)
                && all(planes[candidate].response == response) {
                if nearby { planes[candidate].curved = 1u; }
                group = candidate;
                break;
            }
        }
        let new_group = group == plane_count;
        if new_group && group < 16u {
            planes[group].plane = vec4<f32>(normal, plane_distance);
            planes[group].response = response;
            plane_count += 1u;
        }
        let group_normal = select(normal, planes[min(group, 15u)].plane.xyz, group < 16u);
        let basis = tangent_basis(group_normal);
        let directions = array<vec3<f32>, 5>(basis.u + basis.v, basis.u - basis.v, -basis.u - basis.v, -basis.u + basis.v, -terrain_opposing_plane(collider_index, group_normal).xyz);
        for (var point_index = 0u; point_index < manifold.count; point_index += 1u) {
            let support = TerrainSupport(manifold.points[point_index], (identity << 2u) | point_index, shift);
            if group >= 16u {
                emit_terrain_support(collider_index, normal, response, support);
                continue;
            }
            for (var corner = 0u; corner < 5u; corner += 1u) {
                if (new_group && point_index == 0u)
                    || dot(support.point, directions[corner]) > dot(planes[group].supports[corner].point, directions[corner]) {
                    planes[group].supports[corner] = support;
                }
            }
        }
    }
    for (var group = 0u; group < plane_count; group += 1u) {
        let surface = planes[group];
        let count = select(4u, 5u, surface.curved != 0u);
        for (var corner = 0u; corner < count; corner += 1u) {
            var unique = true;
            for (var previous = 0u; previous < corner; previous += 1u) {
                if distance(surface.supports[corner].point, surface.supports[previous].point) < 1.0e-5 {
                    unique = false;
                }
            }
            if unique {
                let triangle = terrain_row(surface.supports[corner].identity >> 2u);
                let normal = normalize(cross(triangle.second.xyz - triangle.first.xyz, triangle.third.xyz - triangle.first.xyz));
                emit_terrain_support(collider_index, normal, surface.response, surface.supports[corner]);
            }
        }
    }
}

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

fn inverse_inertia(body: u32, vector: vec3<f32>) -> vec3<f32> {
    let mass = world_masses[body];
    return mass.inverse_inertia_x_mass.xyz * vector.x
        + mass.inverse_inertia_y.xyz * vector.y
        + mass.inverse_inertia_z.xyz * vector.z;
}

fn impulse_denominator(
    body_a: u32,
    body_b: u32,
    arm_a: vec3<f32>,
    arm_b: vec3<f32>,
    direction: vec3<f32>,
) -> f32 {
    var result = world_masses[body_a].inverse_inertia_x_mass.w;
    let angular_a = cross(inverse_inertia(body_a, cross(arm_a, direction)), arm_a);
    result += dot(angular_a, direction);
    if body_b != INVALID_MANIFOLD_SLOT {
        result += world_masses[body_b].inverse_inertia_x_mass.w;
        let angular_b = cross(inverse_inertia(body_b, cross(arm_b, direction)), arm_b);
        result += dot(angular_b, direction);
    }
    return result;
}

fn world_inverse_inertia_column(body: u32, world_axis: vec3<f32>) -> vec3<f32> {
    let rotation = rotations[body];
    let local_axis = quat_rotate(vec4<f32>(-rotation.xyz, rotation.w), world_axis);
    let mass = masses[body];
    let local_result = mass.inverse_inertia_x.xyz * local_axis.x
        + mass.inverse_inertia_y.xyz * local_axis.y
        + mass.inverse_inertia_z.xyz * local_axis.z;
    return quat_rotate(rotation, local_result);
}

@compute @workgroup_size(256)
fn update_world_masses(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let body = invocation.x;
    if body >= config.body_count {
        return;
    }
    world_masses[body].position = positions[body];
    world_masses[body].inverse_inertia_x_mass = vec4<f32>(
        world_inverse_inertia_column(body, vec3<f32>(1.0, 0.0, 0.0)),
        masses[body].inverse_mass.x,
    );
    world_masses[body].inverse_inertia_y = vec4<f32>(
        world_inverse_inertia_column(body, vec3<f32>(0.0, 1.0, 0.0)),
        0.0,
    );
    world_masses[body].inverse_inertia_z = vec4<f32>(
        world_inverse_inertia_column(body, vec3<f32>(0.0, 0.0, 1.0)),
        0.0,
    );
}

fn collider_center(index: u32) -> vec3<f32> {
    let collider = colliders[index];
    let body = collider.metadata.x;
    return positions[body].xyz + quat_rotate(rotations[body], collider.local_center.xyz);
}

fn collider_rotation(index: u32) -> vec4<f32> {
    let collider = colliders[index];
    return quat_multiply(rotations[collider.metadata.x], collider.local_rotation);
}


fn collider_is_convex(index: u32) -> bool {
    return colliders[index].shape.x == COLLIDER_SHAPE_CONVEX;
}

fn convex_vertex_count(index: u32) -> u32 {
    return colliders[index].shape.z & 0xffu;
}

fn convex_face_count(index: u32) -> u32 {
    return (colliders[index].shape.z >> 8u) & 0xffu;
}

fn convex_edge_count(index: u32) -> u32 {
    return (colliders[index].shape.z >> 16u) & 0xffu;
}

/// One convex vertex in world space. Stored relative to the compound centre of
/// mass, exactly like a box collider's centre.
fn convex_vertex(index: u32, vertex: u32) -> vec3<f32> {
    let collider = colliders[index];
    let body = collider.metadata.x;
    let local = convex_shapes[collider.shape.y + vertex].xyz;
    return positions[body].xyz + quat_rotate(rotations[body], local);
}

/// One convex face plane in world space: `xyz` outward normal, `w` offset.
///
/// Rotating the body turns the normal; translating it shifts the offset by the
/// normal's component along the translation.
fn convex_face_plane(index: u32, face: u32) -> vec4<f32> {
    let collider = colliders[index];
    let body = collider.metadata.x;
    let slot = collider.shape.y + convex_vertex_count(index) + face;
    let plane = convex_shapes[slot];
    let normal = quat_rotate(rotations[body], plane.xyz);
    return vec4<f32>(normal, plane.w + dot(normal, positions[body].xyz));
}

fn convex_edge_direction(index: u32, edge: u32) -> vec3<f32> {
    let collider = colliders[index];
    let slot = collider.shape.y + convex_vertex_count(index) + convex_face_count(index) + edge;
    return quat_rotate(rotations[collider.metadata.x], convex_shapes[slot].xyz);
}

/// Vertices a collider presents to manifold generation.
fn collider_vertex_count(index: u32) -> u32 {
    if collider_is_convex(index) {
        return convex_vertex_count(index);
    }
    return 8u;
}

/// Separating axes contributed by a collider's own faces.
fn collider_face_axis_count(index: u32) -> u32 {
    if collider_is_convex(index) {
        return convex_face_count(index);
    }
    return 3u;
}

fn collider_face_axis(index: u32, axis: u32) -> vec3<f32> {
    if collider_is_convex(index) {
        return convex_face_plane(index, axis).xyz;
    }
    let rotation = collider_rotation(index);
    if axis == 0u {
        return quat_rotate(rotation, vec3<f32>(1.0, 0.0, 0.0));
    }
    if axis == 1u {
        return quat_rotate(rotation, vec3<f32>(0.0, 1.0, 0.0));
    }
    return quat_rotate(rotation, vec3<f32>(0.0, 0.0, 1.0));
}

/// Separating axes contributed by a collider's own edge directions.
fn collider_edge_axis_count(index: u32) -> u32 {
    if collider_is_convex(index) {
        return convex_edge_count(index);
    }
    return 3u;
}

fn collider_edge_axis(index: u32, axis: u32) -> vec3<f32> {
    if collider_is_convex(index) {
        return convex_edge_direction(index, axis);
    }
    return collider_face_axis(index, axis);
}

/// Projection of a collider onto an axis.
fn project_collider(index: u32, axis: vec3<f32>) -> Interval {
    var interval = Interval(3.402823e+38, -3.402823e+38);
    let count = collider_vertex_count(index);
    for (var vertex = 0u; vertex < count; vertex += 1u) {
        let distance = dot(collider_vertex(index, vertex), axis);
        interval.minimum = min(interval.minimum, distance);
        interval.maximum = max(interval.maximum, distance);
    }
    return interval;
}

/// Lowest point a convex piece presents to the ground, averaged across every
/// vertex sharing that height.
///
/// A box resting flat already reports its bottom-face centre, because the
/// horizontal axes drop out of its support function. Taking a single lowest
/// vertex instead would put the whole ground reaction on one corner and make a
/// shaped part rock on it, so the flat case has to be reproduced here.
fn convex_ground_support(index: u32, direction: vec3<f32>) -> vec3<f32> {
    let count = convex_vertex_count(index);
    var furthest = dot(convex_vertex(index, 0u), direction);
    for (var vertex = 1u; vertex < count; vertex += 1u) {
        furthest = max(furthest, dot(convex_vertex(index, vertex), direction));
    }
    var sum = vec3<f32>(0.0);
    var coplanar = 0.0;
    for (var vertex = 0u; vertex < count; vertex += 1u) {
        let point = convex_vertex(index, vertex);
        if dot(point, direction) >= furthest - 1.0e-4 {
            sum += point;
            coplanar += 1.0;
        }
    }
    return sum / coplanar;
}

fn collider_support_point(index: u32, direction: vec3<f32>) -> vec3<f32> {
    if collider_is_convex(index) {
        let count = convex_vertex_count(index);
        var best = convex_vertex(index, 0u);
        var best_distance = dot(best, direction);
        for (var vertex = 1u; vertex < count; vertex += 1u) {
            let point = convex_vertex(index, vertex);
            let distance = dot(point, direction);
            if distance > best_distance {
                best_distance = distance;
                best = point;
            }
        }
        return best;
    }
    let collider = colliders[index];
    let rotation = collider_rotation(index);
    let axes = array<vec3<f32>, 3>(
        quat_rotate(rotation, vec3<f32>(1.0, 0.0, 0.0)),
        quat_rotate(rotation, vec3<f32>(0.0, 1.0, 0.0)),
        quat_rotate(rotation, vec3<f32>(0.0, 0.0, 1.0)),
    );
    var point = collider_center(index);
    for (var axis = 0u; axis < 3u; axis += 1u) {
        let projection = dot(axes[axis], direction);
        if abs(projection) > 1.0e-5 {
            point += axes[axis] * collider.half_extents[axis] * sign(projection);
        }
    }
    return point;
}

fn full_cylinder_contact_count(index: u32, direction: vec3<f32>) -> u32 {
    let rotation = collider_rotation(index);
    let cylinder_axis = quat_rotate(rotation, vec3<f32>(0.0, 1.0, 0.0));
    let axial_projection = dot(cylinder_axis, direction);
    let radial_length = length(direction - cylinder_axis * axial_projection);
    if radial_length < CYLINDER_MANIFOLD_ALIGNMENT {
        return 4u;
    }
    if abs(axial_projection) < CYLINDER_MANIFOLD_ALIGNMENT {
        return 1u;
    }
    return 1u;
}

fn full_cylinder_support_point(index: u32, direction: vec3<f32>, role: u32) -> vec3<f32> {
    let collider = colliders[index];
    let rotation = collider_rotation(index);
    let radial_axis = quat_rotate(rotation, vec3<f32>(1.0, 0.0, 0.0));
    let cylinder_axis = quat_rotate(rotation, vec3<f32>(0.0, 1.0, 0.0));
    var point = collider_center(index) - radial_axis * collider.local_center.w;
    let axial_projection = dot(cylinder_axis, direction);
    let radial_direction = direction - cylinder_axis * axial_projection;
    let radial_length = length(radial_direction);
    if radial_length < CYLINDER_MANIFOLD_ALIGNMENT {
        point += cylinder_axis * sign(axial_projection) * collider.half_extents.y;
        let tangent_a = normalize(
            vec3<f32>(1.0, 0.0, 0.0) - cylinder_axis * cylinder_axis.x,
        );
        let tangent_b = normalize(cross(cylinder_axis, tangent_a));
        if role == 1u {
            point += tangent_a * collider.half_extents.w;
        } else if role == 2u {
            point -= tangent_a * collider.half_extents.w;
        } else if role == 3u {
            point += tangent_b * collider.half_extents.w;
        } else {
            point -= tangent_b * collider.half_extents.w;
        }
    } else {
        point += radial_direction * (collider.half_extents.w / radial_length);
        if abs(axial_projection) >= CYLINDER_MANIFOLD_ALIGNMENT {
            point += cylinder_axis * sign(axial_projection) * collider.half_extents.y;
        }
    }
    return point;
}

fn full_cylinder_face_pair(collider_a: u32, collider_b: u32, normal: vec3<f32>) -> bool {
    if colliders[collider_a].metadata.w == 0u || colliders[collider_b].metadata.w == 0u {
        return false;
    }
    let axis_a = quat_rotate(collider_rotation(collider_a), vec3<f32>(0.0, 1.0, 0.0));
    let axis_b = quat_rotate(collider_rotation(collider_b), vec3<f32>(0.0, 1.0, 0.0));
    return abs(dot(axis_a, normal)) > 1.0 - CYLINDER_MANIFOLD_ALIGNMENT
        && abs(dot(axis_b, normal)) > 1.0 - CYLINDER_MANIFOLD_ALIGNMENT;
}

fn collider_vertex(index: u32, vertex: u32) -> vec3<f32> {
    if collider_is_convex(index) {
        return convex_vertex(index, vertex);
    }
    let collider = colliders[index];
    let rotation = collider_rotation(index);
    let axes = array<vec3<f32>, 3>(
        quat_rotate(rotation, vec3<f32>(1.0, 0.0, 0.0)),
        quat_rotate(rotation, vec3<f32>(0.0, 1.0, 0.0)),
        quat_rotate(rotation, vec3<f32>(0.0, 0.0, 1.0)),
    );
    var point = collider_center(index);
    for (var axis = 0u; axis < 3u; axis += 1u) {
        let direction = select(-1.0, 1.0, (vertex & (1u << axis)) != 0u);
        point += axes[axis] * collider.half_extents[axis] * direction;
    }
    return point;
}

fn collider_contains_point(index: u32, point: vec3<f32>) -> bool {
    if collider_is_convex(index) {
        let count = convex_face_count(index);
        for (var face = 0u; face < count; face += 1u) {
            let plane = convex_face_plane(index, face);
            if dot(plane.xyz, point) > plane.w + 1.0e-5 {
                return false;
            }
        }
        return true;
    }
    let collider = colliders[index];
    let rotation = collider_rotation(index);
    let local = quat_rotate(
        vec4<f32>(-rotation.xyz, rotation.w),
        point - collider_center(index),
    );
    return all(abs(local) <= collider.half_extents.xyz + vec3<f32>(1.0e-5));
}

fn calculate_contact_point(collider_a: u32, collider_b: u32, normal: vec3<f32>) -> vec3<f32> {
    var sum = vec3<f32>(0.0);
    var count = 0.0;
    let count_a = collider_vertex_count(collider_a);
    for (var vertex = 0u; vertex < count_a; vertex += 1u) {
        let point_a = collider_vertex(collider_a, vertex);
        if collider_contains_point(collider_b, point_a) {
            sum += point_a;
            count += 1.0;
        }
    }
    let count_b = collider_vertex_count(collider_b);
    for (var vertex = 0u; vertex < count_b; vertex += 1u) {
        let point_b = collider_vertex(collider_b, vertex);
        if collider_contains_point(collider_a, point_b) {
            sum += point_b;
            count += 1.0;
        }
    }
    if count > 0.0 {
        return sum / count;
    }
    let point_a = collider_support_point(collider_a, normal);
    let point_b = collider_support_point(collider_b, -normal);
    return (point_a + point_b) * 0.5;
}

fn hash_cell(cell: vec3<i32>) -> u32 {
    var value = bitcast<u32>(cell.x) * 0x8da6b343u;
    value ^= bitcast<u32>(cell.y) * 0xd8163841u;
    value ^= bitcast<u32>(cell.z) * 0xcb1ab31fu;
    value ^= value >> 16u;
    return value | 1u;
}

fn pair_is_suppressed(body_a: u32, body_b: u32) -> bool {
    let low = min(body_a, body_b);
    let high = max(body_a, body_b);
    var left = 0u;
    var right = config.suppression_count;
    loop {
        if left >= right {
            break;
        }
        let middle = left + (right - left) / 2u;
        let candidate = suppressed_pairs[middle];
        if candidate.x < low || (candidate.x == low && candidate.y < high) {
            left = middle + 1u;
        } else {
            right = middle;
        }
    }
    return left < config.suppression_count
        && suppressed_pairs[left].x == low
        && suppressed_pairs[left].y == high;
}

@compute @workgroup_size(256)
fn build_hash(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let collider_index = invocation.x;
    if collider_index >= config.collider_count {
        return;
    }
    let cell = vec3<i32>(floor(collider_center(collider_index)));
    let key = hash_cell(cell);
    let mask = config.hash_capacity - 1u;
    let start = key & mask;
    for (var probe = 0u; probe < MAX_HASH_PROBES; probe += 1u) {
        let slot = (start + probe) & mask;
        let result = atomicCompareExchangeWeak(&hash_keys[slot], EMPTY_HASH_KEY, key);
        if result.exchanged {
            hash_values[slot] = collider_index;
            return;
        }
    }
    atomicOr(&diagnostics[0], PAIR_OVERFLOW_FLAG);
}

fn append_pair(collider_a: u32, collider_b: u32) {
    let body_a = colliders[collider_a].metadata.x;
    let body_b = colliders[collider_b].metadata.x;
    if body_a == body_b || pair_is_suppressed(body_a, body_b) {
        return;
    }
    let output = atomicAdd(&diagnostics[1], 1u);
    if output < config.pair_capacity {
        pairs[output] = vec2<u32>(collider_a, collider_b);
    } else {
        atomicOr(&diagnostics[0], PAIR_OVERFLOW_FLAG);
    }
}

@compute @workgroup_size(256)
fn generate_pairs(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let collider_index = invocation.x;
    if collider_index >= config.collider_count {
        return;
    }
    let own_cell = vec3<i32>(floor(collider_center(collider_index)));
    let mask = config.hash_capacity - 1u;
    for (var z = -1; z <= 1; z += 1) {
        for (var y = -1; y <= 1; y += 1) {
            for (var x = -1; x <= 1; x += 1) {
                let key = hash_cell(own_cell + vec3<i32>(x, y, z));
                let start = key & mask;
                for (var probe = 0u; probe < MAX_HASH_PROBES; probe += 1u) {
                    let slot = (start + probe) & mask;
                    let candidate_key = atomicLoad(&hash_keys[slot]);
                    if candidate_key == EMPTY_HASH_KEY {
                        break;
                    }
                    if candidate_key == key {
                        let candidate = hash_values[slot];
                        if candidate > collider_index {
                            append_pair(collider_index, candidate);
                        }
                    }
                }
            }
        }
    }
}

@compute @workgroup_size(1)
fn finalize_pairs() {
    let pair_count = min(atomicLoad(&diagnostics[1]), config.pair_capacity);
    indirect_args[0] = (pair_count + 255u) / 256u;
    indirect_args[1] = 1u;
    indirect_args[2] = 1u;
}

fn projection_radius(axes: array<vec3<f32>, 3>, extents: vec3<f32>, axis: vec3<f32>) -> f32 {
    return abs(dot(axes[0], axis)) * extents.x
        + abs(dot(axes[1], axis)) * extents.y
        + abs(dot(axes[2], axis)) * extents.z;
}

fn test_sat_axis(
    center_delta: vec3<f32>,
    axes_a: array<vec3<f32>, 3>,
    extents_a: vec3<f32>,
    axes_b: array<vec3<f32>, 3>,
    extents_b: vec3<f32>,
    raw_axis: vec3<f32>,
) -> SatResult {
    let axis_length_squared = dot(raw_axis, raw_axis);
    if axis_length_squared < 1.0e-10 {
        return SatResult(vec3<f32>(1.0, 0.0, 0.0), 3.402823e+38, 0u);
    }
    var axis = raw_axis * inverseSqrt(axis_length_squared);
    let signed_distance = dot(center_delta, axis);
    let penetration = projection_radius(axes_a, extents_a, axis)
        + projection_radius(axes_b, extents_b, axis)
        - abs(signed_distance);
    if signed_distance < 0.0 {
        axis = -axis;
    }
    return SatResult(axis, penetration, 0u);
}

fn obb_sat(collider_a: u32, collider_b: u32) -> SatResult {
    let a = colliders[collider_a];
    let b = colliders[collider_b];
    let rotation_a = collider_rotation(collider_a);
    let rotation_b = collider_rotation(collider_b);
    let axes_a = array<vec3<f32>, 3>(
        quat_rotate(rotation_a, vec3<f32>(1.0, 0.0, 0.0)),
        quat_rotate(rotation_a, vec3<f32>(0.0, 1.0, 0.0)),
        quat_rotate(rotation_a, vec3<f32>(0.0, 0.0, 1.0)),
    );
    let axes_b = array<vec3<f32>, 3>(
        quat_rotate(rotation_b, vec3<f32>(1.0, 0.0, 0.0)),
        quat_rotate(rotation_b, vec3<f32>(0.0, 1.0, 0.0)),
        quat_rotate(rotation_b, vec3<f32>(0.0, 0.0, 1.0)),
    );
    let center_delta = collider_center(collider_b) - collider_center(collider_a);
    var minimum = SatResult(vec3<f32>(1.0, 0.0, 0.0), 3.402823e+38, 0u);
    var near_face_axes = 0u;
    for (var i = 0u; i < 3u; i += 1u) {
        let result_a = test_sat_axis(
            center_delta, axes_a, a.half_extents.xyz, axes_b, b.half_extents.xyz, axes_a[i],
        );
        if result_a.penetration < 0.0 {
            return result_a;
        }
        if result_a.penetration <= 1.0e-4 {
            near_face_axes += 1u;
        }
        if result_a.penetration < minimum.penetration {
            minimum = result_a;
        }
        let result_b = test_sat_axis(
            center_delta, axes_a, a.half_extents.xyz, axes_b, b.half_extents.xyz, axes_b[i],
        );
        if result_b.penetration < 0.0 {
            return result_b;
        }
        if result_b.penetration <= 1.0e-4 {
            near_face_axes += 1u;
        }
        if result_b.penetration < minimum.penetration {
            minimum = result_b;
        }
    }
    for (var i = 0u; i < 3u; i += 1u) {
        for (var j = 0u; j < 3u; j += 1u) {
            let result = test_sat_axis(
                center_delta,
                axes_a,
                a.half_extents.xyz,
                axes_b,
                b.half_extents.xyz,
                cross(axes_a[i], axes_b[j]),
            );
            if result.penetration < 0.0 {
                return result;
            }
            if result.penetration < minimum.penetration {
                minimum = result;
            }
        }
    }
    minimum.near_face_axes = near_face_axes;
    return minimum;
}


/// Separating-axis test for any pair involving a convex polytope.
///
/// Axes are the face normals of both shapes plus every cross product of their
/// edge directions. Both lists arrive already deduplicated by the compiler, so a
/// sheared box presents three face axes and three edge axes and costs exactly
/// what a box costs here. The loop keeps `obb_sat`'s early-out on the first
/// separating axis, which is what bounds the cost for the pairs that do not
/// touch.
fn polytope_sat(collider_a: u32, collider_b: u32) -> SatResult {
    let center_delta = collider_center(collider_b) - collider_center(collider_a);
    var minimum = SatResult(vec3<f32>(1.0, 0.0, 0.0), 3.402823e+38, 0u);
    var near_face_axes = 0u;

    let faces_a = collider_face_axis_count(collider_a);
    for (var index = 0u; index < faces_a; index += 1u) {
        let result = test_polytope_axis(
            collider_a, collider_b, center_delta, collider_face_axis(collider_a, index),
        );
        if result.penetration < 0.0 {
            return result;
        }
        if result.penetration <= 1.0e-4 {
            near_face_axes += 1u;
        }
        if result.penetration < minimum.penetration {
            minimum = result;
        }
    }
    let faces_b = collider_face_axis_count(collider_b);
    for (var index = 0u; index < faces_b; index += 1u) {
        let result = test_polytope_axis(
            collider_a, collider_b, center_delta, collider_face_axis(collider_b, index),
        );
        if result.penetration < 0.0 {
            return result;
        }
        if result.penetration <= 1.0e-4 {
            near_face_axes += 1u;
        }
        if result.penetration < minimum.penetration {
            minimum = result;
        }
    }

    let edges_a = collider_edge_axis_count(collider_a);
    let edges_b = collider_edge_axis_count(collider_b);
    for (var first = 0u; first < edges_a; first += 1u) {
        let axis_a = collider_edge_axis(collider_a, first);
        for (var second = 0u; second < edges_b; second += 1u) {
            let result = test_polytope_axis(
                collider_a,
                collider_b,
                center_delta,
                cross(axis_a, collider_edge_axis(collider_b, second)),
            );
            if result.penetration < 0.0 {
                return result;
            }
            if result.penetration < minimum.penetration {
                minimum = result;
            }
        }
    }

    minimum.near_face_axes = near_face_axes;
    return minimum;
}

fn test_polytope_axis(
    collider_a: u32,
    collider_b: u32,
    center_delta: vec3<f32>,
    raw_axis: vec3<f32>,
) -> SatResult {
    let axis_length_squared = dot(raw_axis, raw_axis);
    if axis_length_squared < 1.0e-10 {
        // Parallel edges give no axis; skip it rather than let it win the
        // minimum.
        return SatResult(vec3<f32>(1.0, 0.0, 0.0), 3.402823e+38, 0u);
    }
    var axis = raw_axis * inverseSqrt(axis_length_squared);
    let interval_a = project_collider(collider_a, axis);
    let interval_b = project_collider(collider_b, axis);
    let penetration = min(interval_a.maximum, interval_b.maximum)
        - max(interval_a.minimum, interval_b.minimum);
    if dot(center_delta, axis) < 0.0 {
        axis = -axis;
    }
    return SatResult(axis, penetration, 0u);
}

/// Narrowphase entry: boxes keep the dedicated path untouched.
fn collider_pair_sat(collider_a: u32, collider_b: u32) -> SatResult {
    if colliders[collider_a].shape.x == COLLIDER_SHAPE_CUBOID
        && colliders[collider_b].shape.x == COLLIDER_SHAPE_CUBOID {
        return obb_sat(collider_a, collider_b);
    }
    return polytope_sat(collider_a, collider_b);
}

fn pair_hash(collider_a: u32, collider_b: u32) -> u32 {
    var value = collider_a * 0x9e3779b9u;
    value ^= collider_b * 0x85ebca6bu;
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    value ^= value >> 16u;
    return value;
}

fn acquire_manifold(collider_a: u32, collider_b: u32) -> u32 {
    let low = min(collider_a, collider_b);
    let high = max(collider_a, collider_b);
    let hash = pair_hash(low, high);
    let ready_key = (hash & 0x7ffffffeu) | 2u;
    let mask = config.pair_capacity - 1u;
    let start = hash & mask;
    for (var probe = 0u; probe < MAX_MANIFOLD_PROBES; probe += 1u) {
        let slot = (start + probe * (probe + 1u) / 2u) & mask;
        let state = atomicLoad(&manifold_keys[slot]);
        if state != 0u && state != 1u {
            let pair = persistent_manifolds[slot].pair_tick.xy;
            if state == ready_key && pair.x == low && pair.y == high {
                return slot;
            }
            let age = config.tick_index - persistent_manifolds[slot].pair_tick.z;
            if age > 4u && age < 0x80000000u {
                let reclaimed = atomicCompareExchangeWeak(&manifold_keys[slot], state, 1u);
                if reclaimed.exchanged {
                    persistent_manifolds[slot].pair_tick = vec4<u32>(low, high, 0u, 0u);
                    persistent_manifolds[slot].normal_penetration = vec4<f32>(0.0);
                    persistent_manifolds[slot].point_impulse = vec4<f32>(0.0);
                    persistent_manifolds[slot].tangent_rolling_impulses = vec4<f32>(0.0);
                    atomicStore(&manifold_keys[slot], ready_key);
                    return slot;
                }
            }
        } else if state == 0u {
            let claimed = atomicCompareExchangeWeak(&manifold_keys[slot], 0u, 1u);
            if claimed.exchanged {
                persistent_manifolds[slot].pair_tick = vec4<u32>(low, high, 0u, 0u);
                persistent_manifolds[slot].normal_penetration = vec4<f32>(0.0);
                persistent_manifolds[slot].point_impulse = vec4<f32>(0.0);
                persistent_manifolds[slot].tangent_rolling_impulses = vec4<f32>(0.0);
                atomicStore(&manifold_keys[slot], ready_key);
                return slot;
            }
        }
    }
    atomicOr(&diagnostics[0], MANIFOLD_OVERFLOW_FLAG);
    return INVALID_MANIFOLD_SLOT;
}

fn emit_narrowphase_contact(pair: vec2<u32>) {
    let body_a = colliders[pair.x].metadata.x;
    let body_b = colliders[pair.y].metadata.x;
    let sat = collider_pair_sat(pair.x, pair.y);
    if sat.penetration < -1.0e-5 {
        return;
    }
    let contact_point = calculate_contact_point(pair.x, pair.y, sat.normal);
    let response = mixed_collider_response(pair.x, pair.y);
    let compliance = colliders[pair.x].surface_elasticity.x
        + colliders[pair.y].surface_elasticity.x;
    let flags = select(
        0u,
        CYLINDER_FACE_PAIR_FLAG,
        full_cylinder_face_pair(pair.x, pair.y, sat.normal),
    );
    let output = atomicAdd(&diagnostics[2], 1u);
    if output < config.pair_capacity {
        contacts[output].metadata = vec4<u32>(
            body_a,
            body_b,
            pair.x | flags,
            pair.y,
        );
        contacts[output].normal_penetration = vec4<f32>(sat.normal, max(sat.penetration, 0.0));
        contacts[output].arm_a_impulse = vec4<f32>(
            contact_point - positions[colliders[pair.x].metadata.x].xyz,
            compliance,
        );
        contacts[output].arm_b = vec4<f32>(
            contact_point - positions[colliders[pair.y].metadata.x].xyz,
            pack_raw_surface_response(response),
        );
    } else {
        atomicOr(&diagnostics[0], PAIR_OVERFLOW_FLAG);
    }
}

@compute @workgroup_size(256)
fn narrowphase(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let pair_index = invocation.x;
    let pair_count = min(atomicLoad(&diagnostics[1]), config.pair_capacity);
    if pair_index >= pair_count {
        return;
    }
    let pair = pairs[pair_index];
    let sat = collider_pair_sat(pair.x, pair.y);
    if sat.penetration < -1.0e-5 {
        return;
    }
    let contact_point = calculate_contact_point(pair.x, pair.y, sat.normal);
    let response = mixed_collider_response(pair.x, pair.y);
    let compliance = colliders[pair.x].surface_elasticity.x
        + colliders[pair.y].surface_elasticity.x;
    let flags = select(
        0u,
        CYLINDER_FACE_PAIR_FLAG,
        full_cylinder_face_pair(pair.x, pair.y, sat.normal),
    );
    let output = atomicAdd(&diagnostics[2], 1u);
    if output < config.pair_capacity {
        contacts[output].metadata = vec4<u32>(
            colliders[pair.x].metadata.x,
            colliders[pair.y].metadata.x,
            pair.x | flags,
            pair.y,
        );
        contacts[output].normal_penetration = vec4<f32>(sat.normal, max(sat.penetration, 0.0));
        contacts[output].arm_a_impulse = vec4<f32>(
            contact_point - positions[colliders[pair.x].metadata.x].xyz,
            compliance,
        );
        contacts[output].arm_b = vec4<f32>(
            contact_point - positions[colliders[pair.y].metadata.x].xyz,
            pack_raw_surface_response(response),
        );
    } else {
        atomicOr(&diagnostics[0], PAIR_OVERFLOW_FLAG);
    }
}

@compute @workgroup_size(256)
fn narrowphase_without_mechanism_self_collisions(
    @builtin(global_invocation_id) invocation: vec3<u32>,
) {
    let pair_index = invocation.x;
    let pair_count = min(atomicLoad(&diagnostics[1]), config.pair_capacity);
    if pair_index >= pair_count {
        return;
    }
    let pair = pairs[pair_index];
    let body_a = colliders[pair.x].metadata.x;
    let body_b = colliders[pair.y].metadata.x;
    if body_components[body_a] != body_components[body_b] {
        emit_narrowphase_contact(pair);
    }
}

@compute @workgroup_size(256)
fn generate_ground_contacts(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let collider_index = invocation.x;
    if collider_index >= config.collider_count {
        return;
    }
    let collider = colliders[collider_index];
    let ground_surface = ground_surfaces[collider_index];
    if dot(ground_surface.plane.xyz, ground_surface.plane.xyz) < 1.0e-10 {
        return;
    }
    let normal = normalize(ground_surface.plane.xyz);
    let down = -normal;
    var support_point = collider_support_point(collider_index, down);
    if collider_is_convex(collider_index) {
        support_point = convex_ground_support(collider_index, down);
    }
    if collider.metadata.w != 0u {
        let contact_count = full_cylinder_contact_count(collider_index, down);
        if collider.metadata.w > contact_count {
            return;
        }
        support_point = full_cylinder_support_point(collider_index, down, collider.metadata.w);
    }
    let signed_distance = dot(support_point, normal) - ground_surface.plane.w;
    if signed_distance > 1.0e-5 {
        return;
    }
    let output = atomicAdd(&diagnostics[2], 1u);
    let response = mixed_ground_response(collider_index);
    let compliance = collider.surface_elasticity.x + ground_surface.elasticity.x;
    if output < config.pair_capacity {
        contacts[output].metadata = vec4<u32>(
            collider.metadata.x,
            INVALID_MANIFOLD_SLOT,
            collider_index | select(0u, ANALYTIC_CYLINDER_FLAG, collider.metadata.w != 0u),
            INVALID_MANIFOLD_SLOT,
        );
        contacts[output].normal_penetration = vec4<f32>(down, max(-signed_distance, 0.0));
        contacts[output].arm_a_impulse = vec4<f32>(
            support_point - positions[collider.metadata.x].xyz,
            compliance,
        );
        contacts[output].arm_b = vec4<f32>(
            0.0,
            0.0,
            0.0,
            pack_raw_surface_response(response),
        );
    } else {
        atomicOr(&diagnostics[0], PAIR_OVERFLOW_FLAG);
    }
}

@compute @workgroup_size(1)
fn finalize_contacts() {
    { atomicOr(&diagnostics[8], 8u); }
    let contact_count = min(atomicLoad(&diagnostics[2]), config.pair_capacity);
    indirect_args[3] = (contact_count + 255u) / 256u;
    indirect_args[4] = 1u;
    indirect_args[5] = 1u;
}

// Cached impulses depend on which endpoints can respond. Holding or releasing
// one endpoint changes that response even when the contact geometry is identical.
fn contact_mass_mode(body_a: u32, body_b: u32) -> u32 {
    var mode = 1u | select(0u, 2u, world_masses[body_a].inverse_inertia_x_mass.w > 0.0);
    if body_b != INVALID_MANIFOLD_SLOT {
        mode |= select(0u, 4u, world_masses[body_b].inverse_inertia_x_mass.w > 0.0);
    }
    return mode;
}

@compute @workgroup_size(256)
fn prepare_contacts(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let contact_index = invocation.x;
    let contact_count = min(atomicLoad(&diagnostics[2]), config.pair_capacity);
    if contact_index >= contact_count {
        return;
    }
    let contact = contacts[contact_index];
    let body_a = contact.metadata.x;
    let body_b = contact.metadata.y;
    let arm_a = contact.arm_a_impulse.xyz;
    let arm_b = contact.arm_b.xyz;
    var inverse_b = 0.0;
    var velocity_b = vec3<f32>(0.0);
    if body_b != INVALID_MANIFOLD_SLOT {
        inverse_b = world_masses[body_b].inverse_inertia_x_mass.w;
        velocity_b = contact_velocity(body_b, arm_b);
    }
    if world_masses[body_a].inverse_inertia_x_mass.w + inverse_b <= 0.0 {
        return;
    }
    let relative = velocity_b - contact_velocity(body_a, arm_a);
    let normal_speed = dot(relative, contact.normal_penetration.xyz);
    let raw_response = unpack_raw_surface_response(contact.arm_b.w);
    let bounce_speed = select(
        0.0,
        -normal_speed * raw_response.z,
        normal_speed < -RESTITUTION_SPEED_THRESHOLD,
    );
    let penetration = contact.normal_penetration.w;
    let analytic = is_analytic_cylinder(contact);
    let contact_flags = contact.metadata.z & CONTACT_FLAG_MASK;
    let collider_a = contact.metadata.z & ~CONTACT_FLAG_MASK;
    let collider_b = contact.metadata.w;
    let gamma = contact.arm_a_impulse.w / (config.delta_seconds * config.delta_seconds);
    contacts[contact_index].arm_b.w = pack_prepared_surface_response(raw_response, gamma);
    contacts[contact_index].normal_penetration.w =
        select(penetration_bias(penetration, analytic), 0.0,
            (contact.metadata.z & TERRAIN_CONTACT_FLAG) != 0u) + bounce_speed;
    if penetration <= 1.0e-6 && normal_speed >= 0.0 {
        contacts[contact_index].metadata.z = INVALID_MANIFOLD_SLOT;
        contacts[contact_index].metadata.w = 0u;
        return;
    }
    let manifold_slot = acquire_manifold(collider_a, collider_b);
    if manifold_slot == INVALID_MANIFOLD_SLOT {
        return;
    }
    let cached = persistent_manifolds[manifold_slot];
    let mass_mode = contact_mass_mode(body_a, body_b);
    let cache_matches = cached.pair_tick.z == config.tick_index - 1u
        && cached.pair_tick.w == mass_mode
        && dot(cached.normal_penetration.xyz, contact.normal_penetration.xyz)
            >= CACHED_NORMAL_ALIGNMENT
        && distance(cached.point_impulse.xyz, contact.arm_a_impulse.xyz)
            <= MAX_CACHED_POINT_MOVEMENT;
    let cached_impulse = select(
        0.0,
        cached.point_impulse.w,
        cache_matches,
    );
    let cached_tangent_rolling = select(
        vec4<f32>(0.0),
        cached.tangent_rolling_impulses,
        cache_matches,
    );
    contacts[contact_index].metadata.z = manifold_slot | contact_flags;
    contacts[contact_index].metadata.w = 1u;
    contacts[contact_index].arm_a_impulse.w = cached_impulse;
    persistent_manifolds[manifold_slot].pair_tick.z = config.tick_index;
    persistent_manifolds[manifold_slot].pair_tick.w = mass_mode;
    persistent_manifolds[manifold_slot].normal_penetration = contact.normal_penetration;
    persistent_manifolds[manifold_slot].point_impulse = vec4<f32>(
        contact.arm_a_impulse.xyz,
        cached_impulse,
    );
    persistent_manifolds[manifold_slot].tangent_rolling_impulses = cached_tangent_rolling;
    if penetration > 1.0e-5
        || normal_speed < -1.0e-5
        || cached_impulse > 1.0e-6
        || any(abs(cached_tangent_rolling) > vec4<f32>(1.0e-6))
    {
        let output = atomicAdd(&diagnostics[5], 1u);
        active_contacts[output] = contact_index;
    }
}

@compute @workgroup_size(256)
fn count_body_contacts(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let active_index = invocation.x;
    if active_index >= min(atomicLoad(&diagnostics[5]), config.pair_capacity) {
        return;
    }
    let contact = contacts[active_contacts[active_index]];
    let body_a = contact.metadata.x;
    let body_b = contact.metadata.y;
    if world_masses[body_a].inverse_inertia_x_mass.w > 0.0 {
        atomicAdd(&diagnostics[BODY_CONTACT_COUNT_OFFSET + body_a], 1u);
    }
    if body_b != INVALID_MANIFOLD_SLOT
        && world_masses[body_b].inverse_inertia_x_mass.w > 0.0
    {
        atomicAdd(&diagnostics[BODY_CONTACT_COUNT_OFFSET + body_b], 1u);
    }
}

@compute @workgroup_size(1)
fn finalize_active_contacts() {
    let active_count = min(atomicLoad(&diagnostics[5]), config.pair_capacity);
    if active_count > 0u && (config.flags & 2u) != 0u {
        atomicStore(&diagnostics[6], config.solver_iterations);
        atomicStore(&diagnostics[7], config.solver_iterations);
    }
    indirect_args[6] = (active_count + 255u) / 256u;
    indirect_args[7] = 1u;
    indirect_args[8] = 1u;
    indirect_args[9] = select(0u, (config.body_count + 255u) / 256u, active_count > 0u);
    indirect_args[10] = 1u;
    indirect_args[11] = 1u;
    indirect_args[12] = select(0u, 1u, active_count > 0u);
    indirect_args[13] = 1u;
    indirect_args[14] = 1u;
    active_contacts[active_count] = INVALID_MANIFOLD_SLOT;
}

fn contact_velocity(body: u32, arm: vec3<f32>) -> vec3<f32> {
    return linear_velocities[body].xyz + cross(angular_velocities[body].xyz, arm);
}

fn penetration_bias(penetration: f32, analytic_cylinder_ground: bool) -> f32 {
    let recovery = select(0.2, 1.0, analytic_cylinder_ground);
    let maximum_speed = select(
        MAX_PENETRATION_CORRECTION_SPEED,
        MAX_ANALYTIC_CYLINDER_CORRECTION_SPEED,
        analytic_cylinder_ground,
    );
    return min(
        max(penetration - PENETRATION_SLOP, 0.0) * recovery / config.delta_seconds,
        maximum_speed,
    );
}

fn contact_target_speed(contact: Contact) -> f32 {
    return contact.normal_penetration.w;
}

fn add_velocity_delta(body: u32, linear: vec3<f32>, angular: vec3<f32>) {
    let base = body * 6u;
    atomicAdd(&velocity_deltas[base], i32(round(linear.x * FIXED_VELOCITY_SCALE)));
    atomicAdd(&velocity_deltas[base + 1u], i32(round(linear.y * FIXED_VELOCITY_SCALE)));
    atomicAdd(&velocity_deltas[base + 2u], i32(round(linear.z * FIXED_VELOCITY_SCALE)));
    atomicAdd(&velocity_deltas[base + 3u], i32(round(angular.x * FIXED_VELOCITY_SCALE)));
    atomicAdd(&velocity_deltas[base + 4u], i32(round(angular.y * FIXED_VELOCITY_SCALE)));
    atomicAdd(&velocity_deltas[base + 5u], i32(round(angular.z * FIXED_VELOCITY_SCALE)));
}

@compute @workgroup_size(256)
fn warm_start(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let active_index = invocation.x;
    let active_count = min(atomicLoad(&diagnostics[5]), config.pair_capacity);
    if active_index >= active_count {
        return;
    }
    let contact_index = active_contacts[active_index];
    let contact = contacts[contact_index];
    let body_a = contact.metadata.x;
    let body_b = contact.metadata.y;
    let arm_a = contact.arm_a_impulse.xyz;
    let arm_b = contact.arm_b.xyz;
    var velocity_b = vec3<f32>(0.0);
    if body_b != INVALID_MANIFOLD_SLOT {
        velocity_b = contact_velocity(body_b, arm_b);
    }
    let normal = contact.normal_penetration.xyz;
    let response = unpack_prepared_surface_response(contact.arm_b.w);
    let denominator = impulse_denominator(body_a, body_b, arm_a, arm_b, normal) + response.w;
    if denominator <= 0.0 {
        return;
    }
    let relative = velocity_b - contact_velocity(body_a, arm_a);
    let normal_speed = dot(relative, normal);
    let slot = contact.metadata.z & ~CONTACT_FLAG_MASK;
    let cached = persistent_manifolds[slot].tangent_rolling_impulses;
    if contact_target_speed(contact) <= 1.0e-5
        && normal_speed >= -1.0e-5
        && contact.arm_a_impulse.w <= 1.0e-6
        && all(abs(cached) <= vec4<f32>(1.0e-6))
    {
        contacts[contact_index].arm_a_impulse.w = 0.0;
        return;
    }
    let warm_start_scale = select(WARM_START_SCALE, 1.0, is_analytic_cylinder(contact));
    let warmed_impulse = contact.arm_a_impulse.w * warm_start_scale;
    var warmed_surface = cached * warm_start_scale;
    var accumulated_impulse = max(
        warmed_impulse
            + parallel_contact_relaxation(contact)
                * (-normal_speed + contact_target_speed(contact) - response.w * warmed_impulse)
                / denominator,
        0.0,
    );
    if (contact.metadata.z & TERRAIN_CONTACT_FLAG) != 0u {
        // Cached impact impulses can exceed the next tick's support impulse.
        // Bound their parallel application by the current closing speed and
        // incident row count so warm starting cannot manufacture a rebound.
        let count = max(atomicLoad(&diagnostics[BODY_CONTACT_COUNT_OFFSET + body_a]), 1u);
        let stopping_impulse = max(-normal_speed + contact_target_speed(contact), 0.0)
            / (denominator * f32(count));
        accumulated_impulse = min(accumulated_impulse, stopping_impulse);
        let rolling_limit = response.z * accumulated_impulse * max(length(arm_a), 1.0e-3);
        warmed_surface = vec4<f32>(
            clamp_surface_impulse(warmed_surface.xy, response.x * accumulated_impulse, response.y * accumulated_impulse),
            clamp_surface_impulse(warmed_surface.zw, rolling_limit, rolling_limit));
    }
    contacts[contact_index].arm_a_impulse.w = accumulated_impulse;
    persistent_manifolds[slot].tangent_rolling_impulses = warmed_surface;
    let basis = tangent_basis(normal);
    let impulse = normal * accumulated_impulse
        + basis.u * warmed_surface.x
        + basis.v * warmed_surface.y;
    let rolling = basis.u * warmed_surface.z + basis.v * warmed_surface.w;
    add_velocity_delta(
        body_a,
        -impulse * world_masses[body_a].inverse_inertia_x_mass.w,
        inverse_inertia(body_a, cross(arm_a, -impulse) - rolling),
    );
    if body_b != INVALID_MANIFOLD_SLOT {
        add_velocity_delta(
            body_b,
            impulse * world_masses[body_b].inverse_inertia_x_mass.w,
            inverse_inertia(body_b, cross(arm_b, impulse) + rolling),
        );
    }
}

fn angular_impulse_denominator(body_a: u32, body_b: u32, axis: vec3<f32>) -> f32 {
    var denominator = dot(axis, inverse_inertia(body_a, axis));
    if body_b != INVALID_MANIFOLD_SLOT {
        denominator += dot(axis, inverse_inertia(body_b, axis));
    }
    return denominator;
}

fn clamp_surface_impulse(candidate: vec2<f32>, static_limit: f32, dynamic_limit: f32)
    -> vec2<f32> {
    let magnitude = length(candidate);
    if magnitude <= static_limit || magnitude <= 1.0e-8 {
        return candidate;
    }
    return candidate * (dynamic_limit / magnitude);
}

fn project_surface_impulses(
    contact: Contact,
    normal_impulse: f32,
    relative: vec3<f32>,
    response: vec4<f32>,
    relaxation: f32,
) -> ContactImpulse {
    let body_a = contact.metadata.x;
    let body_b = contact.metadata.y;
    let arm_a = contact.arm_a_impulse.xyz;
    let arm_b = contact.arm_b.xyz;
    let basis = tangent_basis(contact.normal_penetration.xyz);
    let slot = contact.metadata.z & ~CONTACT_FLAG_MASK;
    let previous = persistent_manifolds[slot].tangent_rolling_impulses;

    let tangent_speed = vec2<f32>(dot(relative, basis.u), dot(relative, basis.v));
    var tangent_candidate = previous.xy;
    let tangent_u_denominator = impulse_denominator(body_a, body_b, arm_a, arm_b, basis.u);
    let tangent_v_denominator = impulse_denominator(body_a, body_b, arm_a, arm_b, basis.v);
    if tangent_u_denominator > 0.0 {
        tangent_candidate.x -= relaxation * tangent_speed.x / tangent_u_denominator;
    }
    if tangent_v_denominator > 0.0 {
        tangent_candidate.y -= relaxation * tangent_speed.y / tangent_v_denominator;
    }
    let tangent_impulse = clamp_surface_impulse(
        tangent_candidate,
        response.x * normal_impulse,
        response.y * normal_impulse,
    );

    var angular_b = vec3<f32>(0.0);
    if body_b != INVALID_MANIFOLD_SLOT {
        angular_b = angular_velocities[body_b].xyz;
    }
    let relative_angular = angular_b - angular_velocities[body_a].xyz;
    let rolling_speed = vec2<f32>(dot(relative_angular, basis.u), dot(relative_angular, basis.v));
    var rolling_candidate = previous.zw;
    let rolling_u_denominator = angular_impulse_denominator(body_a, body_b, basis.u);
    let rolling_v_denominator = angular_impulse_denominator(body_a, body_b, basis.v);
    if rolling_u_denominator > 0.0 {
        rolling_candidate.x -= relaxation * rolling_speed.x / rolling_u_denominator;
    }
    if rolling_v_denominator > 0.0 {
        rolling_candidate.y -= relaxation * rolling_speed.y / rolling_v_denominator;
    }
    var effective_radius = length(arm_a);
    if body_b != INVALID_MANIFOLD_SLOT {
        effective_radius = min(effective_radius, length(arm_b));
    }
    let rolling_limit = response.z * normal_impulse * max(effective_radius, 1.0e-3);
    let rolling_impulse = clamp_surface_impulse(
        rolling_candidate,
        rolling_limit,
        rolling_limit,
    );

    persistent_manifolds[slot].tangent_rolling_impulses = vec4<f32>(
        tangent_impulse,
        rolling_impulse,
    );
    return ContactImpulse(
        basis.u * (tangent_impulse.x - previous.x)
            + basis.v * (tangent_impulse.y - previous.y),
        basis.u * (rolling_impulse.x - previous.z)
            + basis.v * (rolling_impulse.y - previous.w),
    );
}

fn project_contact(contact_index: u32, relaxation: f32) -> ContactImpulse {
    let contact = contacts[contact_index];
    let body_a = contact.metadata.x;
    let body_b = contact.metadata.y;
    let arm_a = contact.arm_a_impulse.xyz;
    let arm_b = contact.arm_b.xyz;
    var velocity_b = vec3<f32>(0.0);
    if body_b != INVALID_MANIFOLD_SLOT {
        velocity_b = contact_velocity(body_b, arm_b);
    }
    let relative = velocity_b - contact_velocity(body_a, arm_a);
    let normal = contact.normal_penetration.xyz;
    let normal_speed = dot(relative, normal);
    let response = unpack_prepared_surface_response(contact.arm_b.w);
    let denominator = impulse_denominator(body_a, body_b, arm_a, arm_b, normal) + response.w;
    if denominator <= 0.0 {
        return ContactImpulse(vec3<f32>(0.0), vec3<f32>(0.0));
    }
    let previous_impulse = max(contact.arm_a_impulse.w, 0.0);
    let accumulated_impulse = max(
        previous_impulse
            + relaxation
                * (-normal_speed + contact_target_speed(contact) - response.w * previous_impulse)
                / denominator,
        0.0,
    );
    contacts[contact_index].arm_a_impulse.w = accumulated_impulse;
    let surface = project_surface_impulses(
        contact,
        accumulated_impulse,
        relative,
        response,
        relaxation,
    );
    return ContactImpulse(
        normal * (accumulated_impulse - previous_impulse) + surface.linear,
        surface.rolling,
    );
}

fn parallel_contact_relaxation(contact: Contact) -> f32 {
    if (contact.metadata.z & TERRAIN_CONTACT_FLAG) != 0u {
        return 1.0 / f32(max(atomicLoad(&diagnostics[BODY_CONTACT_COUNT_OFFSET + contact.metadata.x]), 1u));
    }
    return PROJECTED_RELAXATION
        * select(1.0, CYLINDER_FACE_RELAXATION_SCALE, is_cylinder_face_pair(contact));
}

fn distributed_contact_relaxation(contact: Contact) -> f32 {
    if (contact.metadata.z & TERRAIN_CONTACT_FLAG) != 0u {
        return parallel_contact_relaxation(contact);
    }
    let shape_scale = select(1.0, CYLINDER_FACE_RELAXATION_SCALE, is_cylinder_face_pair(contact));
    // Every Jacobi row reads the same body velocity. Split the response among
    // incident contacts so tessellated surfaces cannot multiply one correction
    // by their number of overlapping collider pieces.
    var count = atomicLoad(&diagnostics[BODY_CONTACT_COUNT_OFFSET + contact.metadata.x]);
    if contact.metadata.y != INVALID_MANIFOLD_SLOT {
        count = max(count, atomicLoad(&diagnostics[BODY_CONTACT_COUNT_OFFSET + contact.metadata.y]));
    }
    return min(PROJECTED_RELAXATION, 0.5 / f32(max(count, 1u)))
        * shape_scale;
}

@compute @workgroup_size(256)
fn solve_accumulate(@builtin(global_invocation_id) invocation: vec3<u32>) {
    if invocation.x == 0u { atomicOr(&diagnostics[8], 16u); }
    let active_index = invocation.x;
    let active_count = min(atomicLoad(&diagnostics[5]), config.pair_capacity);
    if active_index >= active_count {
        return;
    }
    let contact_index = active_contacts[active_index];
    let contact = contacts[contact_index];
    let body_a = contact.metadata.x;
    let body_b = contact.metadata.y;
    let arm_a = contact.arm_a_impulse.xyz;
    let arm_b = contact.arm_b.xyz;
    let impulse = project_contact(contact_index, distributed_contact_relaxation(contact));
    add_velocity_delta(
        body_a,
        -impulse.linear * world_masses[body_a].inverse_inertia_x_mass.w,
        inverse_inertia(body_a, cross(arm_a, -impulse.linear) - impulse.rolling),
    );
    if body_b != INVALID_MANIFOLD_SLOT {
        add_velocity_delta(
            body_b,
            impulse.linear * world_masses[body_b].inverse_inertia_x_mass.w,
            inverse_inertia(body_b, cross(arm_b, impulse.linear) + impulse.rolling),
        );
    }
}

fn solve_contact_immediate(contact_index: u32) {
    let contact = contacts[contact_index];
    let body_a = contact.metadata.x;
    let body_b = contact.metadata.y;
    let arm_a = contact.arm_a_impulse.xyz;
    let arm_b = contact.arm_b.xyz;
    let impulse = project_contact(contact_index, 1.0);
    linear_velocities[body_a] = vec4<f32>(
        linear_velocities[body_a].xyz
            - impulse.linear * world_masses[body_a].inverse_inertia_x_mass.w,
        0.0,
    );
    angular_velocities[body_a] = vec4<f32>(
        angular_velocities[body_a].xyz
            + inverse_inertia(body_a, cross(arm_a, -impulse.linear) - impulse.rolling),
        0.0,
    );
    if body_b != INVALID_MANIFOLD_SLOT {
        linear_velocities[body_b] = vec4<f32>(
                linear_velocities[body_b].xyz
                + impulse.linear * world_masses[body_b].inverse_inertia_x_mass.w,
            0.0,
        );
        angular_velocities[body_b] = vec4<f32>(
            angular_velocities[body_b].xyz
                + inverse_inertia(body_b, cross(arm_b, impulse.linear) + impulse.rolling),
            0.0,
        );
    }
}

fn solve_contacts_serial(active_count: u32) {
    if active_count > MAX_SORTED_SERIAL_CONTACTS {
        for (var active_index = 0u; active_index < active_count; active_index += 1u) {
            solve_contact_immediate(active_contacts[active_index]);
        }
        for (var active_index = active_count; active_index > 0u; active_index -= 1u) {
            solve_contact_immediate(active_contacts[active_index - 1u]);
        }
        return;
    }
    var previous_key = vec3<u32>(0u);
    var has_previous = false;
    for (var active_index = 0u; active_index < active_count; active_index += 1u) {
        var selected = INVALID_MANIFOLD_SLOT;
        var selected_key = vec3<u32>(INVALID_MANIFOLD_SLOT);
        for (var candidate = 0u; candidate < active_count; candidate += 1u) {
            let contact_index = active_contacts[candidate];
            let contact = contacts[contact_index];
            let key = vec3<u32>(contact.metadata.x, contact.metadata.y, contact.metadata.z);
            let after_previous = !has_previous
                || key.x > previous_key.x
                || (key.x == previous_key.x && key.y > previous_key.y)
                || (all(key.xy == previous_key.xy) && key.z > previous_key.z);
            let before_selected = selected == INVALID_MANIFOLD_SLOT
                || key.x < selected_key.x
                || (key.x == selected_key.x && key.y < selected_key.y)
                || (all(key.xy == selected_key.xy) && key.z < selected_key.z);
            if after_previous && before_selected {
                selected = contact_index;
                selected_key = key;
            }
        }
        if selected == INVALID_MANIFOLD_SLOT {
            break;
        }
        solve_contact_immediate(selected);
        previous_key = selected_key;
        has_previous = true;
    }
    has_previous = false;
    for (var active_index = active_count; active_index > 0u; active_index -= 1u) {
        var selected = INVALID_MANIFOLD_SLOT;
        var selected_key = vec3<u32>(0u);
        for (var candidate = 0u; candidate < active_count; candidate += 1u) {
            let contact_index = active_contacts[candidate];
            let contact = contacts[contact_index];
            let key = vec3<u32>(contact.metadata.x, contact.metadata.y, contact.metadata.z);
            let before_previous = !has_previous
                || key.x < previous_key.x
                || (key.x == previous_key.x && key.y < previous_key.y)
                || (all(key.xy == previous_key.xy) && key.z < previous_key.z);
            let after_selected = selected == INVALID_MANIFOLD_SLOT
                || key.x > selected_key.x
                || (key.x == selected_key.x && key.y > selected_key.y)
                || (all(key.xy == selected_key.xy) && key.z > selected_key.z);
            if before_previous && after_selected {
                selected = contact_index;
                selected_key = key;
            }
        }
        if selected == INVALID_MANIFOLD_SLOT {
            break;
        }
        solve_contact_immediate(selected);
        previous_key = selected_key;
        has_previous = true;
    }
}

@compute @workgroup_size(1)
fn solve_accumulate_serial() {
    { atomicOr(&diagnostics[8], 16u); }
    solve_contacts_serial(min(atomicLoad(&diagnostics[5]), config.pair_capacity));
}

fn drive_axis(constraint: DriveConstraint) -> vec3<f32> {
    let parent = constraint.metadata.y;
    let bearing = constraint.bearing;
    var axis = normalize(quat_rotate(rotations[parent], bearing.local_axis_a.xyz));
    if constraint.metadata.z != 0u {
        axis = -normalize(quat_rotate(rotations[parent], bearing.local_axis_b.xyz));
    }
    return axis;
}

fn drive_desired_speed(constraint: DriveConstraint) -> f32 {
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

fn project_drive_velocity_row_immediate(index: u32) {
    let constraint = drive_constraints[index];
    if (constraint.bearing.metadata.w & 2u) != 0u {
        return;
    }
    if constraint.metadata.w == INVALID_MANIFOLD_SLOT
        || constraint.drive.mode == DRIVE_MODE_PASSIVE
    {
        return;
    }
    if drive_constraints[index].bearing.local_axis_a.w == 1.0 {
        project_linear_joint(index, true, true);
        return;
    }
    project_bearing_velocity_row_immediate(index);
    let child = constraint.metadata.x;
    let parent = constraint.metadata.y;
    let axis = drive_axis(constraint);
    let inverse_parent = inverse_inertia(parent, axis);
    let inverse_child = inverse_inertia(child, axis);
    let denominator = bearing_projection_frames[index].drive_inverse_inertia;
    if denominator <= 1.0e-12 {
        return;
    }
    let measured = dot(
        angular_velocities[child].xyz - angular_velocities[parent].xyz,
        axis,
    );
    let desired = drive_desired_speed(constraint);
    let drive = constraint.drive;
    let requested = desired - measured;
    let acceleration =
        drive.source_a_max_acceleration
            * drive_source_fade(measured, requested, drive.source_a_no_load_speed)
        + drive.source_b_max_acceleration
            * drive_source_fade(measured, requested, drive.source_b_no_load_speed);
    let limit = max(
        acceleration * constraint.state.x * config.delta_seconds,
        0.0,
    );
    let previous = constraint.state.y;
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

fn project_bearing_velocity_row_immediate(index: u32) {
    let bearing = drive_constraints[index].bearing;
    if (bearing.metadata.w & 2u) != 0u {
        return;
    }
    if drive_constraints[index].bearing.local_axis_a.w == 1.0 {
        project_linear_joint(index, true, false);
        return;
    }
    let body_a = bearing.metadata.x;
    let body_b = bearing.metadata.y;
    let frame = bearing_projection_frames[index];
    let arm_a = frame.arm_a;
    let arm_b = frame.arm_b;

    let anchor_velocity_a = linear_velocities[body_a].xyz
        + cross(angular_velocities[body_a].xyz, arm_a);
    let anchor_velocity_b = linear_velocities[body_b].xyz
        + cross(angular_velocities[body_b].xyz, arm_b);
    let relative_linear = anchor_velocity_b - anchor_velocity_a;
    let relative_angular = angular_velocities[body_b].xyz - angular_velocities[body_a].xyz;
    let angular_error = vec2<f32>(
        dot(relative_angular, frame.tangent_a), dot(relative_angular, frame.tangent_b),
    );
    // Solve the five hinge rows together using their Schur complement. A light
    // off-centre knuckle must not trade anchor slip for forbidden rotation.
    let angular_impulse = frame.inverse_angular
        * (angular_error - transpose(frame.linear_angular) * relative_linear);
    let impulse = frame.inverse_linear * relative_linear
        - frame.linear_angular * angular_impulse;
    let torque = frame.tangent_a * angular_impulse.x + frame.tangent_b * angular_impulse.y;
    linear_velocities[body_a] += vec4<f32>(
        impulse * world_masses[body_a].inverse_inertia_x_mass.w, 0.0,
    );
    linear_velocities[body_b] -= vec4<f32>(
        impulse * world_masses[body_b].inverse_inertia_x_mass.w, 0.0,
    );
    angular_velocities[body_a] += vec4<f32>(
        inverse_inertia(body_a, cross(arm_a, impulse) + torque), 0.0,
    );
    angular_velocities[body_b] -= vec4<f32>(
        inverse_inertia(body_b, cross(arm_b, impulse) + torque), 0.0,
    );
}

fn bearing_linear_response(
    a: u32, b: u32, ra: vec3<f32>, rb: vec3<f32>, axis: vec3<f32>,
) -> vec3<f32> {
    let inverse_mass = world_masses[a].inverse_inertia_x_mass.w
        + world_masses[b].inverse_inertia_x_mass.w;
    return axis * inverse_mass
        + cross(inverse_inertia(a, cross(ra, axis)), ra)
        + cross(inverse_inertia(b, cross(rb, axis)), rb);
}

fn prepare_bearing_block(index: u32) {
    let bearing = drive_constraints[index].bearing;
    if (bearing.metadata.w & 2u) != 0u {
        return;
    }
    let a = bearing.metadata.x;
    let b = bearing.metadata.y;
    let frame = bearing_projection_frames[index];
    let ra = frame.arm_a;
    let rb = frame.arm_b;
    let x = bearing_linear_response(a, b, ra, rb, vec3<f32>(1.0, 0.0, 0.0));
    let y = bearing_linear_response(a, b, ra, rb, vec3<f32>(0.0, 1.0, 0.0));
    let z = bearing_linear_response(a, b, ra, rb, vec3<f32>(0.0, 0.0, 1.0));
    let determinant = dot(x, cross(y, z));
    if determinant <= 1.0e-30 {
        return;
    }
    let inverse_linear = transpose(mat3x3<f32>(cross(y, z), cross(z, x), cross(x, y)))
        * (1.0 / determinant);
    let ia = inverse_inertia(a, frame.tangent_a);
    let ib = inverse_inertia(b, frame.tangent_a);
    let ja = inverse_inertia(a, frame.tangent_b);
    let jb = inverse_inertia(b, frame.tangent_b);
    let coupling = mat2x3<f32>(
        cross(ia, ra) + cross(ib, rb), cross(ja, ra) + cross(jb, rb),
    );
    let linear_angular = inverse_linear * coupling;
    let angular = mat2x2<f32>(
        vec2<f32>(dot(frame.tangent_a, ia + ib), dot(frame.tangent_b, ia + ib)),
        vec2<f32>(dot(frame.tangent_a, ja + jb), dot(frame.tangent_b, ja + jb)),
    ) - transpose(coupling) * linear_angular;
    let angular_determinant = angular[0][0] * angular[1][1] - angular[0][1] * angular[1][0];
    if angular_determinant <= 1.0e-30 {
        return;
    }
    bearing_projection_frames[index].inverse_linear = inverse_linear;
    bearing_projection_frames[index].linear_angular = linear_angular;
    bearing_projection_frames[index].inverse_angular = mat2x2<f32>(
        vec2<f32>(angular[1][1], -angular[0][1]),
        vec2<f32>(-angular[1][0], angular[0][0]),
    ) * (1.0 / angular_determinant);
    // The motor's effective inertia includes the five constrained directions.
    // The impulse budget is still the same accumulated, torque-limited budget.
    let axis = normalize(cross(frame.tangent_a, frame.tangent_b));
    let inverse_axis_a = inverse_inertia(a, axis);
    let inverse_axis_b = inverse_inertia(b, axis);
    let motor_linear = cross(inverse_axis_a, ra) + cross(inverse_axis_b, rb);
    let motor_angular = vec2<f32>(
        dot(frame.tangent_a, inverse_axis_a + inverse_axis_b),
        dot(frame.tangent_b, inverse_axis_a + inverse_axis_b),
    );
    let constrained_angular = bearing_projection_frames[index].inverse_angular
        * (motor_angular - transpose(linear_angular) * motor_linear);
    let constrained_linear = inverse_linear * motor_linear
        - linear_angular * constrained_angular;
    bearing_projection_frames[index].drive_inverse_inertia = max(
        dot(axis, inverse_axis_a + inverse_axis_b)
            - dot(motor_linear, constrained_linear) - dot(motor_angular, constrained_angular),
        0.0,
    );
}

fn prepare_bearing_projection_frames() {
    for (var index = 0u; index < config.bearing_count; index += 1u) {
        bearing_parent_rows[index] = INVALID_MANIFOLD_SLOT;
        let constraint = drive_constraints[index];
        if (constraint.bearing.metadata.w & 2u) != 0u {
            continue;
        }
        if constraint.metadata.w != INVALID_MANIFOLD_SLOT {
            for (var parent_row = 0u; parent_row < config.bearing_count; parent_row += 1u) {
                let parent = drive_constraints[parent_row];
                if (parent.bearing.metadata.w & 2u) == 0u
                    && parent.metadata.w != INVALID_MANIFOLD_SLOT
                    && parent.metadata.x == constraint.metadata.y
                {
                    bearing_parent_rows[index] = parent_row;
                    break;
                }
            }
        }
        let bearing = drive_constraints[index].bearing;
        let body_a = bearing.metadata.x;
        let body_b = bearing.metadata.y;
        let axis_a = normalize(quat_rotate(rotations[body_a], bearing.local_axis_a.xyz));
        let axis_b = normalize(quat_rotate(rotations[body_b], bearing.local_axis_b.xyz));
        let hinge_axis = normalize(axis_a + axis_b);
        let helper = select(
            vec3<f32>(1.0, 0.0, 0.0),
            vec3<f32>(0.0, 1.0, 0.0),
            abs(hinge_axis.x) > 0.8,
        );
        let tangent_a = normalize(cross(hinge_axis, helper));
        bearing_projection_frames[index] = BearingProjectionFrame(
            quat_rotate(rotations[body_a], bearing.local_anchor_a.xyz),
            quat_rotate(rotations[body_b], bearing.local_anchor_b.xyz),
            tangent_a,
            cross(hinge_axis, tangent_a),
            mat3x3<f32>(),
            mat2x3<f32>(),
            mat2x2<f32>(),
            0.0,
        );
        prepare_bearing_block(index);
    }
}

fn project_bearing_velocities_serial_immediate() {
    for (var index = 0u; index < config.bearing_count; index += 1u) {
        project_drive_velocity_row_immediate(index);
        project_bearing_velocity_row_immediate(index);
    }
    for (var index = config.bearing_count; index > 0u; index -= 1u) {
        project_drive_velocity_row_immediate(index - 1u);
        project_bearing_velocity_row_immediate(index - 1u);
    }
}

// Carry a contact correction through the contacted body's tree path before
// solving the next contact. The return pass restores the child's constraints
// after its ancestors have responded. Closures retain the full sweep below.
fn project_contact_bearing_path(body: u32) {
    if body == INVALID_MANIFOLD_SLOT {
        return;
    }
    var row = INVALID_MANIFOLD_SLOT;
    for (var index = 0u; index < config.bearing_count; index += 1u) {
        let constraint = drive_constraints[index];
        if (constraint.bearing.metadata.w & 2u) == 0u
            && constraint.metadata.w != INVALID_MANIFOLD_SLOT && constraint.metadata.x == body
        {
            row = index;
            break;
        }
    }
    var path: array<u32, 64>;
    var count = 0u;
    while row != INVALID_MANIFOLD_SLOT && count < config.bearing_count {
        path[count] = row;
        count += 1u;
        project_drive_velocity_row_immediate(row);
        project_bearing_velocity_row_immediate(row);
        row = bearing_parent_rows[row];
    }
    while count > 0u {
        count -= 1u;
        project_drive_velocity_row_immediate(path[count]);
        project_bearing_velocity_row_immediate(path[count]);
    }
}

fn solve_articulated_contact_immediate(contact_index: u32) {
    solve_contact_immediate(contact_index);
    project_contact_bearing_path(contacts[contact_index].metadata.x);
    project_contact_bearing_path(contacts[contact_index].metadata.y);
}

// The small articulated contact route keeps contact and bearing corrections in
// one ordered dispatch. Each contact includes a bounded leaf/root/leaf bearing
// projection; each sweep ends with a full projection including loop closures.
// The contact sweep budget stays fixed, independent of body mass ratios.
@compute @workgroup_size(1)
fn solve_small_mechanism_contacts() {
    var active_count = 0u;
    while active_count < config.pair_capacity
        && active_contacts[active_count] != INVALID_MANIFOLD_SLOT
    {
        active_count += 1u;
    }
    if active_count == 0u {
        return;
    }
    prepare_bearing_projection_frames();
    var sorted_contacts: array<u32, 64>;
    if active_count <= MAX_SORTED_SERIAL_CONTACTS {
        var previous_key = vec3<u32>(0u);
        var has_previous = false;
        for (var active_index = 0u; active_index < active_count; active_index += 1u) {
            var selected = INVALID_MANIFOLD_SLOT;
            var selected_key = vec3<u32>(INVALID_MANIFOLD_SLOT);
            for (var candidate = 0u; candidate < active_count; candidate += 1u) {
                let contact_index = active_contacts[candidate];
                let contact = contacts[contact_index];
                let key = vec3<u32>(contact.metadata.x, contact.metadata.y, contact.metadata.z);
                let after_previous = !has_previous
                    || key.x > previous_key.x
                    || (key.x == previous_key.x && key.y > previous_key.y)
                    || (all(key.xy == previous_key.xy) && key.z > previous_key.z);
                let before_selected = selected == INVALID_MANIFOLD_SLOT
                    || key.x < selected_key.x
                    || (key.x == selected_key.x && key.y < selected_key.y)
                    || (all(key.xy == selected_key.xy) && key.z < selected_key.z);
                if after_previous && before_selected {
                    selected = contact_index;
                    selected_key = key;
                }
            }
            if selected == INVALID_MANIFOLD_SLOT {
                break;
            }
            sorted_contacts[active_index] = selected;
            previous_key = selected_key;
            has_previous = true;
        }
    }
    project_bearing_velocities_serial_immediate();
    let iterations = config.solver_iterations;
    for (var iteration = 1u; iteration < iterations; iteration += 1u) {
        if active_count <= MAX_SORTED_SERIAL_CONTACTS {
            for (var active_index = 0u; active_index < active_count; active_index += 1u) {
                solve_articulated_contact_immediate(sorted_contacts[active_index]);
            }
            for (var active_index = active_count; active_index > 0u; active_index -= 1u) {
                solve_articulated_contact_immediate(sorted_contacts[active_index - 1u]);
            }
        } else {
            for (var active_index = 0u; active_index < active_count; active_index += 1u) {
                solve_articulated_contact_immediate(active_contacts[active_index]);
            }
            for (var active_index = active_count; active_index > 0u; active_index -= 1u) {
                solve_articulated_contact_immediate(active_contacts[active_index - 1u]);
            }
        }
        project_bearing_velocities_serial_immediate();
    }
}

@compute @workgroup_size(256)
fn solve_apply(@builtin(global_invocation_id) invocation: vec3<u32>) {
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

@compute @workgroup_size(256)
fn persist_contacts(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let active_index = invocation.x;
    let active_count = min(atomicLoad(&diagnostics[5]), config.pair_capacity);
    if active_index >= active_count {
        return;
    }
    let contact_index = active_contacts[active_index];
    let contact = contacts[contact_index];
    let slot = contact.metadata.z & ~CONTACT_FLAG_MASK;
    persistent_manifolds[slot].pair_tick.z = config.tick_index;
    // prepare_contacts already recorded the endpoint mass modes for this tick.
    persistent_manifolds[slot].normal_penetration = contact.normal_penetration;
    persistent_manifolds[slot].point_impulse = vec4<f32>(
        contact.arm_a_impulse.xyz,
        contact.arm_a_impulse.w,
    );
}


// Positive impulse opposes the relative velocity of B at the carriage point.
fn linear_joint_impulse(a: u32, b: u32, ra: vec3<f32>, rb: vec3<f32>, impulse: vec3<f32>, torque: vec3<f32>, immediate: bool) {
    let la = impulse * world_masses[a].inverse_inertia_x_mass.w;
    let lb = -impulse * world_masses[b].inverse_inertia_x_mass.w;
    let wa = inverse_inertia(a, cross(ra, impulse) + torque);
    let wb = inverse_inertia(b, -cross(rb, impulse) - torque);
    linear_velocities[a] += vec4<f32>(la, 0.0);
    linear_velocities[b] += vec4<f32>(lb, 0.0);
    angular_velocities[a] += vec4<f32>(wa, 0.0);
    angular_velocities[b] += vec4<f32>(wb, 0.0);
}

fn project_linear_joint(index: u32, immediate: bool, motor: bool) {
    let constraint = drive_constraints[index];
    if (constraint.bearing.metadata.w & 2u) != 0u {
        return;
    }
    let bearing = constraint.bearing;
    let a = bearing.metadata.x;
    let b = bearing.metadata.y;
    let axis = normalize(quat_rotate(rotations[a], bearing.local_axis_a.xyz));
    let base_arm = quat_rotate(rotations[a], bearing.local_anchor_a.xyz);
    let rb = quat_rotate(rotations[b], bearing.local_anchor_b.xyz);
    let q = dot(world_masses[b].position.xyz + rb - world_masses[a].position.xyz - base_arm, axis);
    let ra = base_arm + axis * q;
    let helper_axis = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(axis.x) > 0.8);
    let u = normalize(cross(axis, helper_axis));
    let v = cross(axis, u);
    let directions = array<vec3<f32>, 3>(u, v, axis);
    let relaxation = select(0.5, 1.0, immediate);
    if !motor {
        // Lock all three relative rotations, including twist about the rail.
        for (var i = 0u; i < 3u; i += 1u) {
            let direction = directions[i];
            let denominator = dot(direction, inverse_inertia(a, direction) + inverse_inertia(b, direction));
            if denominator > 1.0e-12 {
                let error = dot(angular_velocities[b].xyz - angular_velocities[a].xyz, direction);
                linear_joint_impulse(a, b, ra, rb, vec3<f32>(0.0), direction * (relaxation * error / denominator), immediate);
            }
        }
    }
    for (var i = 0u; i < 3u; i += 1u) {
        if motor && i != 2u { continue; }
        let direction = directions[i];
        let ja = cross(ra, direction);
        let jb = cross(rb, direction);
        let denominator = world_masses[a].inverse_inertia_x_mass.w + world_masses[b].inverse_inertia_x_mass.w
            + dot(ja, inverse_inertia(a, ja)) + dot(jb, inverse_inertia(b, jb));
        if denominator <= 1.0e-12 { continue; }
        let relative = linear_velocities[b].xyz + cross(angular_velocities[b].xyz, rb)
            - linear_velocities[a].xyz - cross(angular_velocities[a].xyz, ra);
        var measured = dot(relative, direction);
        var desired = 0.0;
        var impulse = 0.0;
        if motor {
            desired = drive_desired_speed(constraint);
            let drive = constraint.drive;
            let requested = desired - measured;
            let acceleration = drive.source_a_max_acceleration * drive_source_fade(measured, requested, drive.source_a_no_load_speed)
                + drive.source_b_max_acceleration * drive_source_fade(measured, requested, drive.source_b_no_load_speed);
            let limit = max(acceleration * constraint.state.x * config.delta_seconds, 0.0);
            let previous = constraint.state.y;
            let accumulated = clamp(previous + (desired - measured) / denominator, -limit, limit);
            drive_constraints[index].state.y = accumulated;
            impulse = previous - accumulated;
        } else {
            if i == 2u {
                // Passive force uses its own accumulated impulse, independent of motor budgets.
                let spring = bearing.suspension;
                let stop = bearing.bump_stop;
                if spring.x > 0.0 || spring.z > 0.0 || spring.w > 0.0 || stop.x > 0.0 {
                    let dt = config.delta_seconds;
                    let previous = drive_constraints[index].state.w;
                    var accumulated = previous;
                    var solved_linear = false;
                    let damping_candidates = array<f32, 2>(spring.z, spring.w);
                    for (var damping_index = 0u; damping_index < 2u; damping_index += 1u) {
                        let damping = damping_candidates[damping_index];
                        for (var spring_active = 0u; spring_active < 2u; spring_active += 1u) {
                            let stiffness = select(0.0, spring.x, spring_active != 0u);
                            let constant_force = stiffness
                                    * (spring.y - q - dt * measured + dt * denominator * previous)
                                - damping * (measured - denominator * previous);
                            let candidate = dt * constant_force
                                / (1.0 + dt * denominator * (dt * stiffness + damping));
                            let velocity = measured + denominator * (candidate - previous);
                            let spring_compression = spring.y - q - dt * velocity;
                            let raw_crush = stop.w - q - dt * velocity - stop.z;
                            let damping_matches = (damping_index == 0u && velocity < 0.0)
                                || (damping_index == 1u && velocity >= 0.0);
                            let spring_matches = (spring_active != 0u && spring_compression > 0.0)
                                || (spring_active == 0u && spring_compression <= 0.0);
                            if !solved_linear && damping_matches && spring_matches
                                && (stop.x <= 0.0 || raw_crush <= 0.0)
                            {
                                accumulated = candidate;
                                solved_linear = true;
                            }
                        }
                    }
                    // Backward Euler: evaluate rubber at predicted end-of-step
                    // separation, so contact crossed this tick resists immediately.
                    if !solved_linear {
                        for (var iteration = 0u; iteration < 8u; iteration += 1u) {
                            let velocity = measured + denominator * (accumulated - previous);
                            let spring_compression = spring.y - q - dt * velocity;
                            let raw_crush = stop.w - q - dt * velocity - stop.z;
                            let crush = clamp(raw_crush, 0.0, 0.55 * stop.y);
                            let ratio = crush / max(stop.y, 0.000001);
                            let force = spring.x * max(spring_compression, 0.0)
                                + stop.x * crush * (1.0 + (12.8 / 3.0) * ratio * ratio);
                            let stiffness = select(0.0, spring.x, spring_compression > 0.0)
                                + select(0.0, stop.x * (1.0 + 12.8 * ratio * ratio), raw_crush > 0.0 && raw_crush < 0.55 * stop.y);
                            let damping = select(spring.w, spring.z, velocity < 0.0);
                            let residual = accumulated - dt * (force - damping * velocity);
                            let derivative = 1.0 + dt * (damping + dt * stiffness) * denominator;
                            accumulated -= residual / derivative;
                        }
                    }
                    let delta = relaxation * (accumulated - previous);
                    drive_constraints[index].state.w = previous + delta;
                    linear_joint_impulse(a, b, ra, rb, -direction * delta, vec3<f32>(0.0), immediate);
                    measured += denominator * delta;
                }
                // Predictive, non-bouncing unilateral stops. No motor budget applies.
                let minimum = (bearing.local_anchor_a.w - q) / config.delta_seconds;
                let maximum = (bearing.local_anchor_b.w - q) / config.delta_seconds;
                desired = clamp(measured, minimum, maximum);
            }
            impulse = relaxation * (measured - desired) / denominator;
        }
        linear_joint_impulse(a, b, ra, rb, direction * impulse, vec3<f32>(0.0), immediate);
    }
}

// This entry is bound to scratch velocities, never the authoritative velocity
// buffers. It applies no restitution, friction, drive, spring, or damping force.
fn position_pair_impulse(a: u32, b: u32, ra: vec3<f32>, rb: vec3<f32>, impulse: vec3<f32>) {
    linear_velocities[a] -= vec4<f32>(impulse * world_masses[a].inverse_inertia_x_mass.w, 0.0);
    angular_velocities[a] -= vec4<f32>(inverse_inertia(a, cross(ra, impulse)), 0.0);
    if b != INVALID_MANIFOLD_SLOT {
        linear_velocities[b] += vec4<f32>(impulse * world_masses[b].inverse_inertia_x_mass.w, 0.0);
        angular_velocities[b] += vec4<f32>(inverse_inertia(b, cross(rb, impulse)), 0.0);
    }
}

fn position_joint(index: u32) -> bool {
    var changed = false;
    let bearing = drive_constraints[index].bearing;
    if (bearing.metadata.w & 2u) != 0u { return false; }
    let a = bearing.metadata.x;
    let b = bearing.metadata.y;
    let linear_joint = bearing.local_axis_a.w == 1.0;
    let axis = normalize(quat_rotate(rotations[a], bearing.local_axis_a.xyz));
    let basis = tangent_basis(axis);
    var ra = quat_rotate(rotations[a], bearing.local_anchor_a.xyz);
    let rb = quat_rotate(rotations[b], bearing.local_anchor_b.xyz);
    let separation = dot(world_masses[b].position.xyz + rb - world_masses[a].position.xyz - ra, axis);
    if linear_joint { ra += axis * separation; }
    let directions = array<vec3<f32>, 3>(basis.u, basis.v, axis);
    for (var row = 0u; row < select(2u, 3u, linear_joint); row += 1u) {
        let direction = directions[row];
        let denominator = angular_impulse_denominator(a, b, direction);
        if denominator > 1.0e-12 {
            let error = dot(angular_velocities[b].xyz - angular_velocities[a].xyz, direction);
            let torque = direction * (-error / denominator);
            changed = changed || any(torque != vec3<f32>(0.0));
            angular_velocities[a] -= vec4<f32>(inverse_inertia(a, torque), 0.0);
            angular_velocities[b] += vec4<f32>(inverse_inertia(b, torque), 0.0);
        }
    }
    for (var row = 0u; row < 3u; row += 1u) {
        let direction = directions[row];
        let denominator = impulse_denominator(a, b, ra, rb, direction);
        if denominator <= 1.0e-12 { continue; }
        let speed = dot(contact_velocity(b, rb) - contact_velocity(a, ra), direction);
        var target_speed = 0.0;
        if linear_joint && row == 2u {
            target_speed = clamp(speed,
                (bearing.local_anchor_a.w - separation) / config.delta_seconds,
                (bearing.local_anchor_b.w - separation) / config.delta_seconds);
        }
        let impulse = direction * ((target_speed - speed) / denominator);
        changed = changed || any(impulse != vec3<f32>(0.0));
        position_pair_impulse(a, b, ra, rb, impulse);
    }
    return changed;
}

@compute @workgroup_size(1)
fn prepare_terrain_recovery() {
    terrain_recovery_dispatch[0] = 0u;
    terrain_recovery_dispatch[1] = 1u;
    terrain_recovery_dispatch[2] = 1u;
    terrain_recovery_dispatch[3] = 0u;
    terrain_recovery_dispatch[4] = 1u;
    terrain_recovery_dispatch[5] = 1u;
    terrain_recovery_dispatch[8] = 0u;
    terrain_recovery_dispatch[9] = 1u;
    terrain_recovery_dispatch[10] = 1u;
    let count = min(atomicLoad(&diagnostics[5]), config.pair_capacity);
    if count == 0u || atomicLoad(&diagnostics[0]) != 0u { return; }
    let component_offset = 12u + config.pair_capacity + config.bearing_count;
    for (var body = 0u; body < config.body_count; body += 1u) {
        terrain_recovery_dispatch[component_offset + body] = 0u;
    }
    var needed = false;
    for (var row = 0u; row < count; row += 1u) {
        let contact = contacts[active_contacts[row]];
        if (contact.metadata.z & TERRAIN_CONTACT_FLAG) == 0u { continue; }
        let response = unpack_prepared_surface_response(contact.arm_b.w);
        let elastic_depth = response.w * contact.arm_a_impulse.w * config.delta_seconds;
        if contact.arm_b.z >= 0.0 && contact.arm_b.x > PENETRATION_SLOP + elastic_depth + 1.0e-6 {
            needed = true;
            terrain_recovery_dispatch[component_offset + body_components[contact.metadata.x]] = 1u;
        }
    }
    if !needed { return; }
    // Joint-connected components are precompiled. Close over touching components
    // so a correction cannot move through an omitted body/contact constraint.
    loop {
        var expanded = false;
        for (var row = 0u; row < count; row += 1u) {
            let contact = contacts[active_contacts[row]];
            if contact.metadata.y == INVALID_MANIFOLD_SLOT { continue; }
            let a = component_offset + body_components[contact.metadata.x];
            let b = component_offset + body_components[contact.metadata.y];
            if terrain_recovery_dispatch[a] != terrain_recovery_dispatch[b] {
                terrain_recovery_dispatch[a] = 1u;
                terrain_recovery_dispatch[b] = 1u;
                expanded = true;
            }
        }
        if !expanded { break; }
    }
    var contact_count = 0u;
    for (var row = 0u; row < count; row += 1u) {
        let index = active_contacts[row];
        if terrain_recovery_dispatch[component_offset + body_components[contacts[index].metadata.x]] != 0u {
            terrain_recovery_dispatch[12u + contact_count] = index;
            contact_count += 1u;
        }
    }
    var joint_count = 0u;
    for (var joint = 0u; joint < config.bearing_count; joint += 1u) {
        let body = drive_constraints[joint].bearing.metadata.x;
        if terrain_recovery_dispatch[component_offset + body_components[body]] != 0u {
            terrain_recovery_dispatch[12u + config.pair_capacity + joint_count] = joint;
            joint_count += 1u;
        }
    }
    terrain_recovery_dispatch[6] = contact_count;
    terrain_recovery_dispatch[7] = joint_count;
    terrain_recovery_dispatch[0] = (config.body_count + 255u) / 256u;
    terrain_recovery_dispatch[3] = 1u;
    terrain_recovery_dispatch[8] = (config.bearing_count + 255u) / 256u;
}

@compute @workgroup_size(1)
fn solve_terrain_positions() {
    if atomicLoad(&diagnostics[0]) != 0u { return; }
    let count = recovery_contacts[6];
    let joint_count = recovery_contacts[7];
    for (var iteration = 0u; iteration < 32u; iteration += 1u) {
        var changed = false;
        for (var row = 0u; row < count; row += 1u) {
            let index = recovery_contacts[12u + row];
            let contact = contacts[index];
            let a = contact.metadata.x;
            let b = contact.metadata.y;
            let normal = contact.normal_penetration.xyz;
            let terrain = (contact.metadata.z & TERRAIN_CONTACT_FLAG) != 0u;
            if terrain && contact.arm_b.z < 0.0 { continue; }
            let denominator = impulse_denominator(a, b, contact.arm_a_impulse.xyz, contact.arm_b.xyz, normal);
            if denominator <= 1.0e-12 { continue; }
            var relative = -contact_velocity(a, contact.arm_a_impulse.xyz);
            if b != INVALID_MANIFOLD_SLOT { relative += contact_velocity(b, contact.arm_b.xyz); }
            var target_speed = 0.0;
            var previous = 0.0;
            if terrain {
                let response = unpack_prepared_surface_response(contact.arm_b.w);
                let elastic_depth = response.w * contact.arm_a_impulse.w * config.delta_seconds;
                target_speed = max(contact.arm_b.x - PENETRATION_SLOP - elastic_depth, 0.0) / config.delta_seconds;
                previous = contacts[index].arm_b.y;
            }
            let accumulated = max(previous + (target_speed - dot(relative, normal)) / denominator, 0.0);
            if terrain { contacts[index].arm_b.y = accumulated; }
            changed = changed || accumulated != previous;
            position_pair_impulse(a, b, contact.arm_a_impulse.xyz, contact.arm_b.xyz, normal * (accumulated - previous));
        }
        for (var joint = 0u; joint < joint_count; joint += 1u) {
            let applied = position_joint(recovery_contacts[12u + config.pair_capacity + joint]);
            changed = changed || applied;
        }
        for (var joint = joint_count; joint > 0u; joint -= 1u) {
            let applied = position_joint(recovery_contacts[12u + config.pair_capacity + joint - 1u]);
            changed = changed || applied;
        }
        // Exact fixed points are stricter than the existing correction tolerance.
        // Never terminate with an unresolved contact or joint impulse.
        if !changed { break; }
    }
}

// Re-linearize geometry after a split correction. Physical impulses and cached
// manifolds stay untouched; the next position solve sees the corrected pose.
@compute @workgroup_size(256)
fn update_terrain_position_geometry(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let index = invocation.x;
    if index >= min(atomicLoad(&diagnostics[2]), config.pair_capacity)
        || atomicLoad(&diagnostics[0]) != 0u { return; }
    let contact = contacts[index];
    if contact.metadata.w != 1u || (contact.metadata.z & TERRAIN_CONTACT_FLAG) == 0u { return; }
    let slot = contact.metadata.z & ~CONTACT_FLAG_MASK;
    let identity = persistent_manifolds[slot].pair_tick.xy;
    let collider = identity.x;
    let triangle = terrain_row((identity.y & 0x7fffffffu) >> 2u);
    let point_index = identity.y & 3u;
    let normal = -contact.normal_penetration.xyz;
    let manifold = terrain_contact_polygon(collider, triangle, vec3<f32>(0.0));
    contacts[index].arm_b.y = 0.0;
    if point_index >= manifold.count {
        contacts[index].arm_b.x = 0.0;
        contacts[index].arm_b.z = -1.0;
        return;
    }
    contacts[index].arm_b.z = 0.0;
    contacts[index].arm_b.x = terrain_point_depth(collider, manifold.points[point_index], normal);
    contacts[index].arm_a_impulse = vec4<f32>(manifold.points[point_index] - positions[contact.metadata.x].xyz, contact.arm_a_impulse.w);
}
