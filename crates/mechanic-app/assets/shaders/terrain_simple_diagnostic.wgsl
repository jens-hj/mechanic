// Diagnostic only: same terrain geometry and forward lighting, no material maps.
#import bevy_pbr::{
    forward_io::{VertexOutput, FragmentOutput},
    pbr_fragment::pbr_input_from_vertex_output,
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
    pbr_types::STANDARD_MATERIAL_FLAGS_FOG_ENABLED_BIT,
}

@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    var pbr_input = pbr_input_from_vertex_output(in, is_front, false);
    pbr_input.material.flags |= STANDARD_MATERIAL_FLAGS_FOG_ENABLED_BIT;
    pbr_input.material.base_color = vec4<f32>(0.18, 0.25, 0.10, 1.0);
    pbr_input.material.perceptual_roughness = 0.9;
    pbr_input.material.metallic = 0.0;
    pbr_input.diffuse_occlusion = vec3<f32>(1.0);
    pbr_input.specular_occlusion = 1.0;
    pbr_input.N = normalize(pbr_input.world_normal);
    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
}
