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
