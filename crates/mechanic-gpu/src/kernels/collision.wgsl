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
    // shape kind, convex-buffer offset, packed element counts, reserved.
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

struct WorldMass {
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
const PROJECTED_RELAXATION: f32 = 0.125;
const WARM_START_SCALE: f32 = 0.5;
const MAX_ROLLING_RESISTANCE: f32 = 0.04;
const MIN_GAMMA_LOG2: f32 = -28.0;
const MAX_GAMMA_LOG2: f32 = -8.0;
const RESTITUTION_SPEED_THRESHOLD: f32 = 1.0;
const PENETRATION_SLOP: f32 = 0.001;
const MAX_PENETRATION_CORRECTION_SPEED: f32 = 1.0;
const CACHED_NORMAL_ALIGNMENT: f32 = 0.98;
const MAX_CACHED_POINT_MOVEMENT: f32 = 0.02;
const CYLINDER_MANIFOLD_ALIGNMENT: f32 = 0.05;
const MAX_SORTED_SERIAL_CONTACTS: u32 = 64u;
const INVALID_MANIFOLD_SLOT: u32 = 0xffffffffu;
const MAX_MANIFOLD_PROBES: u32 = 256u;
const ANALYTIC_CYLINDER_FLAG: u32 = 0x80000000u;
const CYLINDER_FACE_PAIR_FLAG: u32 = 0x40000000u;
const CONTACT_FLAG_MASK: u32 = ANALYTIC_CYLINDER_FLAG | CYLINDER_FACE_PAIR_FLAG;
// A full cylinder has sixteen overlapping sector rows. Face landings therefore
// need one sixteenth of the ordinary per-contact Jacobi correction.
const CYLINDER_FACE_RELAXATION_SCALE: f32 = 0.0625;
const COLLIDER_SHAPE_CUBOID: u32 = 0u;
const COLLIDER_SHAPE_CONVEX: u32 = 1u;

fn mixed_surface_response(collider_a: u32, collider_b: u32) -> vec4<f32> {
    let first = colliders[collider_a].surface_response;
    if collider_b == INVALID_MANIFOLD_SLOT {
        return vec4<f32>(
            sqrt(first.x * ground_surface.response.x),
            sqrt(first.y * ground_surface.response.y),
            max(first.z, ground_surface.response.z),
            sqrt(first.w * ground_surface.response.w),
        );
    }
    let second = colliders[collider_b].surface_response;
    return vec4<f32>(
        sqrt(first.x * second.x),
        sqrt(first.y * second.y),
        max(first.z, second.z),
        sqrt(first.w * second.w),
    );
}

fn combined_contact_compliance(collider_a: u32, collider_b: u32) -> f32 {
    let first = colliders[collider_a].surface_elasticity.x;
    if collider_b == INVALID_MANIFOLD_SLOT {
        return first + ground_surface.elasticity.x;
    }
    return first + colliders[collider_b].surface_elasticity.x;
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
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> rotations: array<vec4<f32>>;
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
@group(0) @binding(29) var<uniform> ground_surface: GroundSurface;

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
fn convex_ground_support(index: u32) -> vec3<f32> {
    let count = convex_vertex_count(index);
    var lowest = convex_vertex(index, 0u).y;
    for (var vertex = 1u; vertex < count; vertex += 1u) {
        lowest = min(lowest, convex_vertex(index, vertex).y);
    }
    var sum = vec3<f32>(0.0);
    var coplanar = 0.0;
    for (var vertex = 0u; vertex < count; vertex += 1u) {
        let point = convex_vertex(index, vertex);
        if point.y <= lowest + 1.0e-4 {
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
    let response = mixed_surface_response(pair.x, pair.y);
    let compliance = combined_contact_compliance(pair.x, pair.y);
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
    let response = mixed_surface_response(pair.x, pair.y);
    let compliance = combined_contact_compliance(pair.x, pair.y);
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
    let down = vec3<f32>(0.0, -1.0, 0.0);
    var support_point = collider_support_point(collider_index, down);
    if collider_is_convex(collider_index) {
        support_point = convex_ground_support(collider_index);
    }
    if collider.metadata.w != 0u {
        let contact_count = full_cylinder_contact_count(collider_index, down);
        if collider.metadata.w > contact_count {
            return;
        }
        support_point = full_cylinder_support_point(collider_index, down, collider.metadata.w);
    }
    let bottom = support_point.y;
    if bottom > 1.0e-5 {
        return;
    }
    let output = atomicAdd(&diagnostics[2], 1u);
    let response = mixed_surface_response(collider_index, INVALID_MANIFOLD_SLOT);
    let compliance = combined_contact_compliance(collider_index, INVALID_MANIFOLD_SLOT);
    if output < config.pair_capacity {
        contacts[output].metadata = vec4<u32>(
            collider.metadata.x,
            INVALID_MANIFOLD_SLOT,
            collider_index | select(0u, ANALYTIC_CYLINDER_FLAG, collider.metadata.w != 0u),
            INVALID_MANIFOLD_SLOT,
        );
        contacts[output].normal_penetration = vec4<f32>(0.0, -1.0, 0.0, max(-bottom, 0.0));
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
    let contact_count = min(atomicLoad(&diagnostics[2]), config.pair_capacity);
    indirect_args[3] = (contact_count + 255u) / 256u;
    indirect_args[4] = 1u;
    indirect_args[5] = 1u;
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
        penetration_bias(penetration, analytic) + bounce_speed;
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
    let cache_matches = cached.pair_tick.z == config.tick_index - 1u
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
    persistent_manifolds[manifold_slot].pair_tick.w = 1u;
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

@compute @workgroup_size(1)
fn finalize_active_contacts() {
    let active_count = min(atomicLoad(&diagnostics[5]), config.pair_capacity);
    indirect_args[6] = (active_count + 255u) / 256u;
    indirect_args[7] = 1u;
    indirect_args[8] = 1u;
    indirect_args[9] = select(0u, (config.body_count + 255u) / 256u, active_count > 0u);
    indirect_args[10] = 1u;
    indirect_args[11] = 1u;
    indirect_args[12] = select(0u, 1u, active_count > 0u);
    indirect_args[13] = 1u;
    indirect_args[14] = 1u;
}

fn contact_velocity(body: u32, arm: vec3<f32>) -> vec3<f32> {
    return linear_velocities[body].xyz + cross(angular_velocities[body].xyz, arm);
}

fn penetration_bias(penetration: f32, analytic_cylinder_ground: bool) -> f32 {
    let recovery = select(0.2, 1.0, analytic_cylinder_ground);
    return min(
        max(penetration - PENETRATION_SLOP, 0.0) * recovery / config.delta_seconds,
        MAX_PENETRATION_CORRECTION_SPEED,
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
    let warmed_surface = cached * warm_start_scale;
    let accumulated_impulse = max(
        warmed_impulse
            + parallel_contact_relaxation(contact)
                * (-normal_speed + contact_target_speed(contact) - response.w * warmed_impulse)
                / denominator,
        0.0,
    );
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
    return PROJECTED_RELAXATION
        * select(1.0, CYLINDER_FACE_RELAXATION_SCALE, is_cylinder_face_pair(contact));
}

@compute @workgroup_size(256)
fn solve_accumulate(@builtin(global_invocation_id) invocation: vec3<u32>) {
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
    let impulse = project_contact(contact_index, parallel_contact_relaxation(contact));
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

@compute @workgroup_size(1)
fn solve_accumulate_serial() {
    let active_count = min(atomicLoad(&diagnostics[5]), config.pair_capacity);
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
    persistent_manifolds[slot].pair_tick.w = contact.metadata.w;
    persistent_manifolds[slot].normal_penetration = contact.normal_penetration;
    persistent_manifolds[slot].point_impulse = vec4<f32>(
        contact.arm_a_impulse.xyz,
        contact.arm_a_impulse.w,
    );
}
