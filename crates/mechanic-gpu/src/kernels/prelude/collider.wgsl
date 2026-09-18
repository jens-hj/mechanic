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
