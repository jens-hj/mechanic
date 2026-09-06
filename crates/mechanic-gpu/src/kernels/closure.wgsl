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

struct MechanismBody {
    metadata: vec4<u32>,
    traversal: vec4<u32>,
    bind_relative_position: vec4<f32>,
    bind_relative_rotation: vec4<f32>,
};

struct Coordinate {
    position: f32,
    velocity: f32,
};

struct LinkState {
    position: vec4<f32>,
    rotation: vec4<f32>,
    metadata: vec4<u32>,
};

struct CoordinateAccumulator {
    gradient_bits: atomic<u32>,
    diagonal_bits: atomic<u32>,
};

struct PcgRow {
    solution: f32,
    residual: f32,
    preconditioned: f32,
    direction: f32,
    operator_product: f32,
    padding_a: f32,
    padding_b: f32,
    padding_c: f32,
};

struct ClosureVector {
    position: vec3<f32>,
    axis: vec3<f32>,
};

// Linear closures have two transverse rows, an active unilateral stop row,
// and three orientation rows. All residual vectors are in world coordinates.
struct ClosureFrame {
    linear: bool,
    rail_axis: vec3<f32>,
    separation: vec3<f32>,
    rotation_error: vec3<f32>,
    lower: f32,
    upper: f32,
};

const BEARING_CLOSURE_FLAG: u32 = 1u;
const BEARING_SUSPENDED_FLAG: u32 = 2u;
const INVALID_NUMERIC_FLAG: u32 = 2u;
const ANCHOR_TOLERANCE_MICROMETERS: u32 = 10u;
const AXIS_TOLERANCE_MICRODEGREES: u32 = 1000u;
const AXIS_WEIGHT: f32 = 1.0;
const STEP_SCALE: f32 = 0.35;
const MAX_STEP_RADIANS: f32 = 0.2;

@group(0) @binding(0) var<uniform> config: TickConfig;
@group(0) @binding(1) var<storage, read_write> diagnostics: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read> bearings: array<Bearing>;
@group(0) @binding(3) var<storage, read> mechanism_bodies: array<MechanismBody>;
@group(0) @binding(4) var<storage, read_write> coordinates: array<Coordinate>;
@group(0) @binding(5) var<storage, read> mechanism_links: array<LinkState>;
@group(0) @binding(6) var<storage, read_write> accumulators: array<CoordinateAccumulator>;
@group(0) @binding(7) var<storage, read_write> closure_state: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> indirect_args: array<u32>;
@group(0) @binding(9) var<storage, read_write> pcg_rows: array<PcgRow>;

fn quat_rotate(rotation: vec4<f32>, vector: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(rotation.xyz, vector);
    return vector + rotation.w * t + cross(rotation.xyz, t);
}

fn quat_multiply(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz),
        a.w * b.w - dot(a.xyz, b.xyz));
}

fn rotation_log(a: vec4<f32>, b: vec4<f32>) -> vec3<f32> {
    var delta = normalize(quat_multiply(a, vec4<f32>(-b.xyz, b.w)));
    if delta.w < 0.0 {
        delta = -delta;
    }
    let sine = length(delta.xyz);
    if sine < 1.0e-7 {
        return 2.0 * delta.xyz;
    }
    return delta.xyz * (2.0 * atan2(sine, delta.w) / sine);
}

fn closure_frame(bearing: Bearing, a: LinkState, b: LinkState,
    anchor_a: vec3<f32>, anchor_b: vec3<f32>, axis_a: vec3<f32>) -> ClosureFrame {
    return ClosureFrame(bearing.local_axis_a.w == 1.0, axis_a,
        anchor_a - anchor_b, rotation_log(a.rotation, b.rotation),
        -bearing.local_anchor_b.w, -bearing.local_anchor_a.w);
}

fn closure_position(frame: ClosureFrame) -> vec3<f32> {
    if !frame.linear {
        return frame.separation;
    }
    // Separation has the opposite sign to the physical displacement coordinate.
    return frame.separation - frame.rail_axis * clamp(
        dot(frame.separation, frame.rail_axis), frame.lower, frame.upper);
}

fn endpoint_derivative(tree_bearing: Bearing, joint_axis: vec3<f32>,
    joint_anchor: vec3<f32>, endpoint_anchor: vec3<f32>, endpoint_axis: vec3<f32>,
    residual_sign: f32, frame: ClosureFrame) -> ClosureVector {
    var velocity = cross(joint_axis, endpoint_anchor - joint_anchor);
    var angular = joint_axis;
    if tree_bearing.local_axis_a.w == 1.0 {
        velocity = joint_axis;
        angular = vec3<f32>(0.0);
    }
    if !frame.linear {
        return ClosureVector(residual_sign * velocity,
            residual_sign * cross(angular, endpoint_axis));
    }
    let separation_derivative = residual_sign * velocity;
    var axis_derivative = vec3<f32>(0.0);
    if residual_sign > 0.0 {
        axis_derivative = cross(angular, frame.rail_axis);
    }
    let along = dot(frame.separation, frame.rail_axis);
    var position_derivative = separation_derivative
        - axis_derivative * clamp(along, frame.lower, frame.upper);
    if along >= frame.lower && along <= frame.upper {
        position_derivative -= frame.rail_axis * (
            dot(separation_derivative, frame.rail_axis)
            + dot(frame.separation, axis_derivative));
    }
    // Inverse left/right SO(3) Jacobians differentiate log(Ra * inverse(Rb)).
    let angle = length(frame.rotation_error);
    var coefficient = 1.0 / 12.0;
    if angle > 1.0e-3 {
        coefficient = (1.0 - 0.5 * angle / tan(0.5 * angle)) / (angle * angle);
    }
    let first_cross = cross(frame.rotation_error, angular);
    let orientation_derivative = residual_sign * (angular
        - 0.5 * residual_sign * first_cross
        + coefficient * cross(frame.rotation_error, first_cross));
    return ClosureVector(position_derivative, orientation_derivative);
}

fn finite_vector(value: vec3<f32>) -> bool {
    return all(value == value) && all(abs(value) < vec3<f32>(3.402823e+38));
}

fn atomic_add_gradient(coordinate: u32, value: f32) {
    var old_bits = atomicLoad(&accumulators[coordinate].gradient_bits);
    loop {
        let old_value = bitcast<f32>(old_bits);
        let new_bits = bitcast<u32>(old_value + value);
        let result = atomicCompareExchangeWeak(
            &accumulators[coordinate].gradient_bits,
            old_bits,
            new_bits,
        );
        if result.exchanged {
            break;
        }
        old_bits = result.old_value;
    }
}

fn atomic_add_diagonal(coordinate: u32, value: f32) {
    var old_bits = atomicLoad(&accumulators[coordinate].diagonal_bits);
    loop {
        let old_value = bitcast<f32>(old_bits);
        let new_bits = bitcast<u32>(old_value + value);
        let result = atomicCompareExchangeWeak(
            &accumulators[coordinate].diagonal_bits,
            old_bits,
            new_bits,
        );
        if result.exchanged {
            break;
        }
        old_bits = result.old_value;
    }
}

fn accumulate_coordinate(
    coordinate: u32,
    residual_position: vec3<f32>,
    residual_axis: vec3<f32>,
    endpoint_position_derivative: vec3<f32>,
    endpoint_axis_derivative: vec3<f32>,
    residual_sign: f32,
) {
    if coordinate >= config.reserved_b {
        atomicOr(&diagnostics[0], INVALID_NUMERIC_FLAG);
        return;
    }
    let position_derivative = endpoint_position_derivative * residual_sign;
    let axis_derivative = endpoint_axis_derivative * residual_sign;
    let gradient = dot(residual_position, position_derivative)
        + AXIS_WEIGHT * dot(residual_axis, axis_derivative);
    let diagonal = dot(position_derivative, position_derivative)
        + AXIS_WEIGHT * dot(axis_derivative, axis_derivative);
    if gradient == gradient && diagonal == diagonal {
        atomic_add_gradient(coordinate, gradient);
        atomic_add_diagonal(coordinate, diagonal);
    } else {
        atomicOr(&diagnostics[0], INVALID_NUMERIC_FLAG);
    }
}

fn accumulate_branch(
    start_body: u32,
    endpoint_anchor: vec3<f32>,
    endpoint_axis: vec3<f32>,
    residual_position: vec3<f32>,
    residual_axis: vec3<f32>,
    residual_sign: f32,
    frame: ClosureFrame,
) {
    var body = start_body;
    loop {
        let mechanism = mechanism_bodies[body];
        if mechanism.metadata.w != 0u {
            break;
        }
        let parent = mechanism.metadata.x;
        let tree_bearing = bearings[mechanism.metadata.y];
        if (tree_bearing.metadata.w & BEARING_SUSPENDED_FLAG) != 0u {
            break;
        }
        let parent_pose = mechanism_links[parent];
        var joint_axis: vec3<f32>;
        var joint_anchor: vec3<f32>;
        if mechanism.metadata.z == 0u {
            joint_axis = normalize(quat_rotate(parent_pose.rotation, tree_bearing.local_axis_a.xyz));
            joint_anchor = parent_pose.position.xyz
                + quat_rotate(parent_pose.rotation, tree_bearing.local_anchor_a.xyz);
        } else {
            joint_axis = -normalize(quat_rotate(parent_pose.rotation, tree_bearing.local_axis_b.xyz));
            joint_anchor = parent_pose.position.xyz
                + quat_rotate(parent_pose.rotation, tree_bearing.local_anchor_b.xyz);
        }
        let derivative = endpoint_derivative(tree_bearing, joint_axis, joint_anchor,
            endpoint_anchor, endpoint_axis, residual_sign, frame);
        accumulate_coordinate(
            tree_bearing.metadata.z,
            residual_position,
            residual_axis,
            derivative.position,
            derivative.axis,
            1.0,
        );
        body = parent;
    }
}

fn branch_product(
    start_body: u32,
    endpoint_anchor: vec3<f32>,
    endpoint_axis: vec3<f32>,
    residual_sign: f32,
    frame: ClosureFrame,
) -> ClosureVector {
    var product = ClosureVector(vec3<f32>(0.0), vec3<f32>(0.0));
    var body = start_body;
    loop {
        let mechanism = mechanism_bodies[body];
        if mechanism.metadata.w != 0u {
            break;
        }
        let parent = mechanism.metadata.x;
        let tree_bearing = bearings[mechanism.metadata.y];
        if (tree_bearing.metadata.w & BEARING_SUSPENDED_FLAG) != 0u {
            break;
        }
        let parent_pose = mechanism_links[parent];
        var joint_axis: vec3<f32>;
        var joint_anchor: vec3<f32>;
        if mechanism.metadata.z == 0u {
            joint_axis = normalize(quat_rotate(parent_pose.rotation, tree_bearing.local_axis_a.xyz));
            joint_anchor = parent_pose.position.xyz
                + quat_rotate(parent_pose.rotation, tree_bearing.local_anchor_a.xyz);
        } else {
            joint_axis = -normalize(quat_rotate(parent_pose.rotation, tree_bearing.local_axis_b.xyz));
            joint_anchor = parent_pose.position.xyz
                + quat_rotate(parent_pose.rotation, tree_bearing.local_anchor_b.xyz);
        }
        let coordinate = tree_bearing.metadata.z;
        let direction = pcg_rows[coordinate].direction;
        let derivative = endpoint_derivative(tree_bearing, joint_axis, joint_anchor,
            endpoint_anchor, endpoint_axis, residual_sign, frame);
        product.position += derivative.position * direction;
        product.axis += derivative.axis * direction;
        body = parent;
    }
    return product;
}

fn branch_transpose(
    start_body: u32,
    endpoint_anchor: vec3<f32>,
    endpoint_axis: vec3<f32>,
    product: ClosureVector,
    residual_sign: f32,
    frame: ClosureFrame,
) {
    var body = start_body;
    loop {
        let mechanism = mechanism_bodies[body];
        if mechanism.metadata.w != 0u {
            break;
        }
        let parent = mechanism.metadata.x;
        let tree_bearing = bearings[mechanism.metadata.y];
        if (tree_bearing.metadata.w & BEARING_SUSPENDED_FLAG) != 0u {
            break;
        }
        let parent_pose = mechanism_links[parent];
        var joint_axis: vec3<f32>;
        var joint_anchor: vec3<f32>;
        if mechanism.metadata.z == 0u {
            joint_axis = normalize(quat_rotate(parent_pose.rotation, tree_bearing.local_axis_a.xyz));
            joint_anchor = parent_pose.position.xyz
                + quat_rotate(parent_pose.rotation, tree_bearing.local_anchor_a.xyz);
        } else {
            joint_axis = -normalize(quat_rotate(parent_pose.rotation, tree_bearing.local_axis_b.xyz));
            joint_anchor = parent_pose.position.xyz
                + quat_rotate(parent_pose.rotation, tree_bearing.local_anchor_b.xyz);
        }
        let derivative = endpoint_derivative(tree_bearing, joint_axis, joint_anchor,
            endpoint_anchor, endpoint_axis, residual_sign, frame);
        pcg_rows[tree_bearing.metadata.z].operator_product += dot(derivative.position, product.position)
            + AXIS_WEIGHT * dot(derivative.axis, product.axis);
        body = parent;
    }
}

@compute @workgroup_size(256)
fn evaluate_closures(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let index = invocation.x;
    if index >= config.bearing_count {
        return;
    }
    let bearing = bearings[index];
    if (bearing.metadata.w & BEARING_CLOSURE_FLAG) == 0u
        || (bearing.metadata.w & BEARING_SUSPENDED_FLAG) != 0u {
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
    let frame = closure_frame(bearing, pose_a, pose_b, anchor_a, anchor_b, axis_a);
    let residual_position = closure_position(frame);
    let residual_axis = select(axis_a - axis_b, frame.rotation_error, frame.linear);
    if !finite_vector(residual_position) || !finite_vector(residual_axis) {
        atomicOr(&diagnostics[0], INVALID_NUMERIC_FLAG);
        atomicStore(&closure_state[0], 0xffffffffu);
        atomicStore(&closure_state[1], 0xffffffffu);
        return;
    }
    let anchor_micrometers = u32(round(length(residual_position) * 1000000.0));
    let axis_degrees = select(acos(clamp(dot(axis_a, axis_b), -1.0, 1.0)),
        length(frame.rotation_error), frame.linear) * 57.295779513;
    let axis_microdegrees = u32(round(axis_degrees * 1000000.0));
    atomicMax(&closure_state[0], anchor_micrometers);
    atomicMax(&closure_state[1], axis_microdegrees);
    if anchor_micrometers <= ANCHOR_TOLERANCE_MICROMETERS
        && axis_microdegrees <= AXIS_TOLERANCE_MICRODEGREES {
        return;
    }
    accumulate_branch(
        bearing.metadata.x,
        anchor_a,
        axis_a,
        residual_position,
        residual_axis,
        1.0,
        frame,
    );
    accumulate_branch(
        bearing.metadata.y,
        anchor_b,
        axis_b,
        residual_position,
        residual_axis,
        -1.0,
        frame,
    );
}

@compute @workgroup_size(1)
fn finalize_closures() {
    let unconverged = atomicLoad(&closure_state[0]) > ANCHOR_TOLERANCE_MICROMETERS
        || atomicLoad(&closure_state[1]) > AXIS_TOLERANCE_MICRODEGREES;
    indirect_args[0] = select(0u, (config.bearing_count + 255u) / 256u, unconverged);
    indirect_args[1] = 1u;
    indirect_args[2] = 1u;
    indirect_args[3] = select(0u, (config.body_count + 255u) / 256u, unconverged);
    indirect_args[4] = 1u;
    indirect_args[5] = 1u;
}

@compute @workgroup_size(256)
fn solve_closure_pcg(@builtin(global_invocation_id) invocation: vec3<u32>) {
    if invocation.x != 0u {
        return;
    }
    var residual_preconditioned = 0.0;
    for (var coordinate = 0u; coordinate < config.reserved_b; coordinate += 1u) {
        let gradient = bitcast<f32>(atomicLoad(&accumulators[coordinate].gradient_bits));
        let diagonal = bitcast<f32>(atomicLoad(&accumulators[coordinate].diagonal_bits));
        var residual = 0.0;
        var preconditioned = 0.0;
        if diagonal > 1.0e-12 {
            residual = -gradient;
            preconditioned = residual / diagonal;
        }
        pcg_rows[coordinate].solution = 0.0;
        pcg_rows[coordinate].residual = residual;
        pcg_rows[coordinate].preconditioned = preconditioned;
        pcg_rows[coordinate].direction = preconditioned;
        pcg_rows[coordinate].operator_product = 0.0;
        residual_preconditioned += residual * preconditioned;
    }

    for (var iteration = 0u; iteration < 8u; iteration += 1u) {
        for (var coordinate = 0u; coordinate < config.reserved_b; coordinate += 1u) {
            pcg_rows[coordinate].operator_product = 0.0;
        }
        for (var index = 0u; index < config.bearing_count; index += 1u) {
            let bearing = bearings[index];
            if (bearing.metadata.w & BEARING_CLOSURE_FLAG) == 0u
                || (bearing.metadata.w & BEARING_SUSPENDED_FLAG) != 0u {
                continue;
            }
            let pose_a = mechanism_links[bearing.metadata.x];
            let pose_b = mechanism_links[bearing.metadata.y];
            let anchor_a = pose_a.position.xyz
                + quat_rotate(pose_a.rotation, bearing.local_anchor_a.xyz);
            let anchor_b = pose_b.position.xyz
                + quat_rotate(pose_b.rotation, bearing.local_anchor_b.xyz);
            let axis_a = normalize(quat_rotate(pose_a.rotation, bearing.local_axis_a.xyz));
            let axis_b = normalize(quat_rotate(pose_b.rotation, bearing.local_axis_b.xyz));
            let frame = closure_frame(bearing, pose_a, pose_b, anchor_a, anchor_b, axis_a);
            let product_a = branch_product(bearing.metadata.x, anchor_a, axis_a, 1.0, frame);
            let product_b = branch_product(bearing.metadata.y, anchor_b, axis_b, -1.0, frame);
            let product = ClosureVector(
                product_a.position + product_b.position,
                product_a.axis + product_b.axis,
            );
            branch_transpose(bearing.metadata.x, anchor_a, axis_a, product, 1.0, frame);
            branch_transpose(bearing.metadata.y, anchor_b, axis_b, product, -1.0, frame);
        }

        var direction_operator = 0.0;
        for (var coordinate = 0u; coordinate < config.reserved_b; coordinate += 1u) {
            direction_operator += pcg_rows[coordinate].direction
                * pcg_rows[coordinate].operator_product;
        }
        var alpha = 0.0;
        if abs(direction_operator) > 1.0e-20 {
            alpha = residual_preconditioned / direction_operator;
        }
        var next_residual_preconditioned = 0.0;
        for (var coordinate = 0u; coordinate < config.reserved_b; coordinate += 1u) {
            pcg_rows[coordinate].solution += alpha * pcg_rows[coordinate].direction;
            pcg_rows[coordinate].residual -= alpha * pcg_rows[coordinate].operator_product;
            let diagonal = bitcast<f32>(atomicLoad(&accumulators[coordinate].diagonal_bits));
            pcg_rows[coordinate].preconditioned = 0.0;
            if diagonal > 1.0e-12 {
                pcg_rows[coordinate].preconditioned = pcg_rows[coordinate].residual / diagonal;
            }
            next_residual_preconditioned += pcg_rows[coordinate].residual
                * pcg_rows[coordinate].preconditioned;
        }
        var beta = 0.0;
        if abs(residual_preconditioned) > 1.0e-20 {
            beta = next_residual_preconditioned / residual_preconditioned;
        }
        for (var coordinate = 0u; coordinate < config.reserved_b; coordinate += 1u) {
            pcg_rows[coordinate].direction = pcg_rows[coordinate].preconditioned
                + beta * pcg_rows[coordinate].direction;
        }
        residual_preconditioned = next_residual_preconditioned;
    }

    for (var index = 0u; index < config.bearing_count; index += 1u) {
        let bearing = bearings[index];
        if bearing.metadata.w != 0u { continue; }
        let coordinate = bearing.metadata.z;
        // Translation is affine in its coordinate. Apply the full correction;
        // rotational coordinates retain damping for their nonlinear kinematics.
        let scale = select(STEP_SCALE, 1.0, bearing.local_axis_a.w == 1.0);
        let step = clamp(
            scale * pcg_rows[coordinate].solution,
            -MAX_STEP_RADIANS,
            MAX_STEP_RADIANS,
        );
        if step == step {
            coordinates[coordinate].position += step;
        } else {
            atomicOr(&diagnostics[0], INVALID_NUMERIC_FLAG);
        }
    }
    for (var index = 0u; index < config.bearing_count; index += 1u) {
        let bearing = bearings[index];
        if bearing.metadata.w == 0u && bearing.local_axis_a.w == 1.0 {
            let coordinate = bearing.metadata.z;
            let lower = bearing.local_anchor_a.w;
            let upper = bearing.local_anchor_b.w;
            coordinates[coordinate].position = clamp(coordinates[coordinate].position, lower, upper);
            if (coordinates[coordinate].position <= lower && coordinates[coordinate].velocity < 0.0)
                || (coordinates[coordinate].position >= upper && coordinates[coordinate].velocity > 0.0) {
                coordinates[coordinate].velocity = 0.0;
            }
        }
    }
}
