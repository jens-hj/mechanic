#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::{alpha_discard,apply_pbr_lighting,main_pass_post_lighting_processing},
    forward_io::{VertexOutput,FragmentOutput},
}
@group(#{MATERIAL_BIND_GROUP}) @binding(100) var<uniform> aperture: vec4<f32>;
fn aperture_mask(uv:vec2<f32>)->bool {
    let size=select(vec2(0.25,0.25),vec2(0.5,0.25),uv.y<0.5);
    let center=floor(uv/size)*size+size*0.5;
    let d=abs(uv-center);
    return max(d.x,d.y)<0.034 && uv.y<0.75 && (uv.y<0.5 || uv.x<0.5);
}
@fragment fn fragment(in:VertexOutput,@builtin(front_facing) is_front:bool)->FragmentOutput {
    var pbr_input=pbr_input_from_standard_material(in,is_front);
    if aperture_mask(in.uv) {pbr_input.material.emissive*=aperture.x;}
    pbr_input.material.base_color=alpha_discard(pbr_input.material,pbr_input.material.base_color);
    var out:FragmentOutput;
    out.color=apply_pbr_lighting(pbr_input);
    out.color=main_pass_post_lighting_processing(pbr_input,out.color);
    return out;
}
