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
    sort_count: u32,
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

const COLLIDER_SHAPE_CONVEX: u32 = 1u;

struct Aabb {
    minimum: vec4<f32>,
    maximum: vec4<f32>,
};

struct SortParams {
    k: u32,
    j: u32,
    reserved_a: u32,
    reserved_b: u32,
};

const PAIR_OVERFLOW_FLAG: u32 = 1u;
const INVALID_NODE: u32 = 0xffffffffu;
const SORT_BLOCK_SIZE: u32 = 256u;

@group(0) @binding(0) var<uniform> config: TickConfig;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> rotations: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> diagnostics: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read> colliders: array<Collider>;
@group(0) @binding(9) var<storage, read_write> pairs: array<vec2<u32>>;
@group(0) @binding(11) var<storage, read> suppressed_pairs: array<vec2<u32>>;
@group(0) @binding(12) var<storage, read_write> indirect_args: array<u32>;
@group(0) @binding(17) var<storage, read_write> collider_aabbs: array<Aabb>;
@group(0) @binding(18) var<storage, read_write> morton_entries: array<vec2<u32>>;
@group(0) @binding(19) var<storage, read_write> node_aabbs: array<Aabb>;
@group(0) @binding(20) var<storage, read_write> node_children: array<vec2<u32>>;
@group(0) @binding(21) var<storage, read_write> node_parents: array<u32>;
@group(0) @binding(22) var<storage, read_write> node_visits: array<atomic<u32>>;
@group(0) @binding(23) var<uniform> sort_params: SortParams;
@group(0) @binding(28) var<storage, read> convex_shapes: array<vec4<f32>>;

var<workgroup> shared_entries: array<vec2<u32>, 256>;

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

fn collider_center(index: u32) -> vec3<f32> {
    let collider = colliders[index];
    return positions[collider.metadata.x].xyz
        + quat_rotate(rotations[collider.metadata.x], collider.local_center.xyz);
}

fn collider_rotation(index: u32) -> vec4<f32> {
    let collider = colliders[index];
    return quat_multiply(rotations[collider.metadata.x], collider.local_rotation);
}

fn expand_morton_bits(input: u32) -> u32 {
    var value = input & 0x000003ffu;
    value = (value | (value << 16u)) & 0x030000ffu;
    value = (value | (value << 8u)) & 0x0300f00fu;
    value = (value | (value << 4u)) & 0x030c30c3u;
    value = (value | (value << 2u)) & 0x09249249u;
    return value;
}

fn morton_code(center: vec3<f32>) -> u32 {
    let quantized = vec3<u32>(clamp(floor(center + vec3<f32>(512.0)), vec3<f32>(0.0), vec3<f32>(1023.0)));
    return expand_morton_bits(quantized.x)
        | (expand_morton_bits(quantized.y) << 1u)
        | (expand_morton_bits(quantized.z) << 2u);
}

fn entry_greater(a: vec2<u32>, b: vec2<u32>) -> bool {
    return a.x > b.x || (a.x == b.x && a.y > b.y);
}

fn entry_less(a: vec2<u32>, b: vec2<u32>) -> bool {
    return a.x < b.x || (a.x == b.x && a.y < b.y);
}

@compute @workgroup_size(256)
fn compute_morton(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let index = invocation.x;
    if index >= config.sort_count {
        return;
    }
    if index >= config.collider_count {
        morton_entries[index] = vec2<u32>(0xffffffffu, index);
        return;
    }
    let collider = colliders[index];
    let center = collider_center(index);
    if collider.shape.x == COLLIDER_SHAPE_CONVEX {
        // A polytope has no half extents; bound it by its own vertices.
        let body = collider.metadata.x;
        let count = collider.shape.z & 0xffu;
        var minimum = vec3<f32>(3.402823e+38);
        var maximum = vec3<f32>(-3.402823e+38);
        for (var vertex = 0u; vertex < count; vertex += 1u) {
            let point = positions[body].xyz
                + quat_rotate(rotations[body], convex_shapes[collider.shape.y + vertex].xyz);
            minimum = min(minimum, point);
            maximum = max(maximum, point);
        }
        collider_aabbs[index].minimum = vec4<f32>(minimum, 0.0);
        collider_aabbs[index].maximum = vec4<f32>(maximum, 0.0);
        morton_entries[index] = vec2<u32>(morton_code(center), index);
        return;
    }
    let rotation = collider_rotation(index);
    let axis_x = quat_rotate(rotation, vec3<f32>(1.0, 0.0, 0.0));
    let axis_y = quat_rotate(rotation, vec3<f32>(0.0, 1.0, 0.0));
    let axis_z = quat_rotate(rotation, vec3<f32>(0.0, 0.0, 1.0));
    let world_extents = abs(axis_x) * collider.half_extents.x
        + abs(axis_y) * collider.half_extents.y
        + abs(axis_z) * collider.half_extents.z;
    collider_aabbs[index].minimum = vec4<f32>(center - world_extents, 0.0);
    collider_aabbs[index].maximum = vec4<f32>(center + world_extents, 0.0);
    morton_entries[index] = vec2<u32>(morton_code(center), index);
}

@compute @workgroup_size(256)
fn sort_local_initial(
    @builtin(global_invocation_id) invocation: vec3<u32>,
    @builtin(local_invocation_id) local_invocation: vec3<u32>,
) {
    let index = invocation.x;
    let local_index = local_invocation.x;
    shared_entries[local_index] = morton_entries[index];
    workgroupBarrier();
    for (var k = 2u; k <= SORT_BLOCK_SIZE; k *= 2u) {
        var j = k / 2u;
        loop {
            let partner = local_index ^ j;
            if partner > local_index {
                let left = shared_entries[local_index];
                let right = shared_entries[partner];
                let ascending = (index & k) == 0u;
                if (ascending && entry_greater(left, right))
                    || (!ascending && entry_less(left, right))
                {
                    shared_entries[local_index] = right;
                    shared_entries[partner] = left;
                }
            }
            workgroupBarrier();
            if j == 1u {
                break;
            }
            j /= 2u;
        }
    }
    morton_entries[index] = shared_entries[local_index];
}

@compute @workgroup_size(256)
fn sort_global_step(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let index = invocation.x;
    if index >= config.sort_count {
        return;
    }
    let partner = index ^ sort_params.j;
    if partner <= index || partner >= config.sort_count {
        return;
    }
    let left = morton_entries[index];
    let right = morton_entries[partner];
    let ascending = (index & sort_params.k) == 0u;
    if (ascending && entry_greater(left, right))
        || (!ascending && entry_less(left, right))
    {
        morton_entries[index] = right;
        morton_entries[partner] = left;
    }
}

@compute @workgroup_size(256)
fn sort_local_merge(
    @builtin(global_invocation_id) invocation: vec3<u32>,
    @builtin(local_invocation_id) local_invocation: vec3<u32>,
) {
    let index = invocation.x;
    let local_index = local_invocation.x;
    shared_entries[local_index] = morton_entries[index];
    workgroupBarrier();
    var j = SORT_BLOCK_SIZE / 2u;
    loop {
        let partner = local_index ^ j;
        if partner > local_index {
            let left = shared_entries[local_index];
            let right = shared_entries[partner];
            let ascending = (index & sort_params.k) == 0u;
            if (ascending && entry_greater(left, right))
                || (!ascending && entry_less(left, right))
            {
                shared_entries[local_index] = right;
                shared_entries[partner] = left;
            }
        }
        workgroupBarrier();
        if j == 1u {
            break;
        }
        j /= 2u;
    }
    morton_entries[index] = shared_entries[local_index];
}

fn common_prefix(left: i32, right: i32) -> i32 {
    let count = i32(config.collider_count);
    if right < 0 || right >= count {
        return -1;
    }
    let left_entry = morton_entries[u32(left)];
    let right_entry = morton_entries[u32(right)];
    let code_difference = left_entry.x ^ right_entry.x;
    if code_difference != 0u {
        return i32(countLeadingZeros(code_difference));
    }
    return 32 + i32(countLeadingZeros(left_entry.y ^ right_entry.y));
}

@compute @workgroup_size(256)
fn build_topology(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let node = invocation.x;
    if config.collider_count <= 1u || node >= config.collider_count - 1u {
        return;
    }
    let index = i32(node);
    let direction = select(-1, 1, common_prefix(index, index + 1) > common_prefix(index, index - 1));
    let minimum_prefix = common_prefix(index, index - direction);
    var maximum_length = 2;
    while common_prefix(index, index + maximum_length * direction) > minimum_prefix {
        maximum_length *= 2;
    }
    var length = 0;
    var step = maximum_length / 2;
    while step >= 1 {
        if common_prefix(index, index + (length + step) * direction) > minimum_prefix {
            length += step;
        }
        step /= 2;
    }
    let other = index + length * direction;
    let node_prefix = common_prefix(index, other);
    var split_offset = 0;
    step = (length + 1) / 2;
    while step >= 1 {
        if common_prefix(index, index + (split_offset + step) * direction) > node_prefix {
            split_offset += step;
        }
        if step == 1 {
            break;
        }
        step = (step + 1) / 2;
    }
    let split = index + split_offset * direction + min(direction, 0);
    let first = min(index, other);
    let last = max(index, other);
    let leaf_base = config.collider_count - 1u;
    let left_index = u32(split);
    let right_index = u32(split + 1);
    let left_child = select(left_index, leaf_base + left_index, split == first);
    let right_child = select(right_index, leaf_base + right_index, split + 1 == last);
    node_children[node] = vec2<u32>(left_child, right_child);
    node_parents[left_child] = node + 1u;
    node_parents[right_child] = node + 1u;
}

@compute @workgroup_size(256)
fn prepare_leaves(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let leaf = invocation.x;
    if leaf >= config.collider_count {
        return;
    }
    let collider_index = morton_entries[leaf].y;
    node_aabbs[config.collider_count - 1u + leaf] = collider_aabbs[collider_index];
}

@compute @workgroup_size(256)
fn build_bounds(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let leaf = invocation.x;
    if config.collider_count <= 1u || leaf >= config.collider_count {
        return;
    }
    var current = config.collider_count - 1u + leaf;
    loop {
        let encoded_parent = node_parents[current];
        if encoded_parent == 0u {
            return;
        }
        let parent = encoded_parent - 1u;
        let arrival = atomicAdd(&node_visits[parent], 1u);
        if arrival == 0u {
            return;
        }
        if arrival > 1u {
            atomicOr(&diagnostics[0], PAIR_OVERFLOW_FLAG);
            return;
        }
        let children = node_children[parent];
        let left = node_aabbs[children.x];
        let right = node_aabbs[children.y];
        node_aabbs[parent].minimum = min(left.minimum, right.minimum);
        node_aabbs[parent].maximum = max(left.maximum, right.maximum);
        storageBarrier();
        current = parent;
    }
}

fn overlaps(a: Aabb, b: Aabb) -> bool {
    return all(a.minimum.xyz <= b.maximum.xyz) && all(b.minimum.xyz <= a.maximum.xyz);
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
fn traverse(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let leaf = invocation.x;
    if leaf >= config.collider_count {
        return;
    }
    let collider_index = morton_entries[leaf].y;
    let query = collider_aabbs[collider_index];
    let leaf_base = config.collider_count - 1u;
    var stack: array<u32, 64>;
    var stack_size = 1u;
    stack[0] = select(leaf_base, 0u, config.collider_count > 1u);
    loop {
        if stack_size == 0u {
            break;
        }
        stack_size -= 1u;
        let node = stack[stack_size];
        if !overlaps(query, node_aabbs[node]) {
            continue;
        }
        if node >= leaf_base {
            let candidate = morton_entries[node - leaf_base].y;
            if candidate > collider_index {
                append_pair(collider_index, candidate);
            }
            continue;
        }
        if stack_size > 61u {
            atomicOr(&diagnostics[0], PAIR_OVERFLOW_FLAG);
            return;
        }
        let children = node_children[node];
        stack[stack_size] = children.x;
        stack[stack_size + 1u] = children.y;
        stack_size += 2u;
    }
}

@compute @workgroup_size(1)
fn finalize_pairs() {
    let pair_count = min(atomicLoad(&diagnostics[1]), config.pair_capacity);
    indirect_args[0] = (pair_count + 255u) / 256u;
    indirect_args[1] = 1u;
    indirect_args[2] = 1u;
}
