#import bevy_pbr::{
    forward_io::{VertexOutput, FragmentOutput},
    mesh_functions,
    mesh_view_bindings::globals,
    pbr_fragment::pbr_input_from_vertex_output,
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
    pbr_types::{STANDARD_MATERIAL_FLAGS_ALPHA_MODE_BLEND, STANDARD_MATERIAL_FLAGS_FOG_ENABLED_BIT},
    view_transformations::position_world_to_clip,
}

// Mirrors `WaterRenderMaterial` in world/water_render.rs.
@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> shallow: vec4<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var<uniform> deep: vec4<f32>;

struct WaterVertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    // Depth of water under the vertex, then its current along x and z.
    @location(8) water: vec3<f32>,
}

struct WaterVaryings {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    @location(6) @interpolate(flat) instance_index: u32,
#endif
    @location(8) water: vec3<f32>,
}

@vertex
fn vertex(vertex: WaterVertex) -> WaterVaryings {
    var out: WaterVaryings;
    let world_from_local = mesh_functions::get_world_from_local(vertex.instance_index);
    out.world_position = mesh_functions::mesh_position_local_to_world(
        world_from_local,
        vec4<f32>(vertex.position, 1.0),
    );
    out.position = position_world_to_clip(out.world_position.xyz);
    out.world_normal = mesh_functions::mesh_normal_local_to_world(
        vertex.normal,
        vertex.instance_index,
    );
#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    out.instance_index = vertex.instance_index;
#endif
    out.water = vertex.water;
    return out;
}

// Slope of a few travelling waves at a point: small ripples everywhere,
// carried along by the current.
fn wave_slope(point: vec2<f32>, flow: vec2<f32>, time: f32) -> vec2<f32> {
    let drift = point - flow * time;
    var slope = vec2<f32>(0.0);
    let directions = array<vec2<f32>, 4>(
        vec2<f32>(0.8, 0.6),
        vec2<f32>(-0.6, 0.8),
        vec2<f32>(0.3, -0.95),
        vec2<f32>(-0.9, -0.4),
    );
    let lengths = array<f32, 4>(3.1, 1.7, 0.9, 0.53);
    for (var wave = 0u; wave < 4u; wave += 1u) {
        let k = 6.2831853 / lengths[wave];
        let speed = sqrt(9.81 / k);
        let phase = k * dot(directions[wave], drift) - speed * k * time;
        let amplitude = 0.012 * lengths[wave];
        slope += directions[wave] * (k * amplitude * cos(phase));
    }
    return slope;
}

@fragment
fn fragment(
    varyings: WaterVaryings,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    var in: VertexOutput;
    in.position = varyings.position;
    in.world_position = varyings.world_position;
    in.world_normal = varyings.world_normal;
#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    in.instance_index = varyings.instance_index;
#endif
    var pbr_input = pbr_input_from_vertex_output(in, is_front, false);
    pbr_input.material.flags |= STANDARD_MATERIAL_FLAGS_FOG_ENABLED_BIT
        | STANDARD_MATERIAL_FLAGS_ALPHA_MODE_BLEND;

    let depth = max(varyings.water.x, 0.0);
    let flow = varyings.water.yz;
    let slope = wave_slope(varyings.world_position.xz, flow, globals.time);
    // Ripples fade with distance so far water reads as a calm mirror
    // instead of aliasing.
    let calm = 1.0 / (1.0 + fwidth(varyings.world_position.x) * 4.0);
    var normal = normalize(vec3<f32>(-slope.x * calm, 1.0, -slope.y * calm));
    if !is_front {
        normal = -normal;
    }

    let murk = 1.0 - exp(-depth / 3.0);
    pbr_input.material.base_color = vec4<f32>(
        mix(shallow.rgb, deep.rgb, murk),
        mix(0.25, 0.9, 1.0 - exp(-depth / 1.2)),
    );
    pbr_input.material.perceptual_roughness = 0.06;
    pbr_input.material.metallic = 0.0;
    pbr_input.material.reflectance = vec3<f32>(0.5);
    pbr_input.N = normal;

    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
}
