#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::alpha_discard,
}

#ifdef PREPASS_PIPELINE
#import bevy_pbr::{
    prepass_io::{VertexOutput, FragmentOutput},
    pbr_deferred_functions::deferred_output,
}
#else
#import bevy_pbr::{
    forward_io::{VertexOutput, FragmentOutput},
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
}
#endif

@group(#{MATERIAL_BIND_GROUP}) @binding(100) var tint_mask: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(101) var tint_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(102) var<uniform> chroma_material: vec4<f32>;

fn linear_to_oklab(c: vec3<f32>) -> vec3<f32> {
    let l = pow(max(0.4122214708*c.r + 0.5363325363*c.g + 0.0514459929*c.b, 0.0), 1.0/3.0);
    let m = pow(max(0.2119034982*c.r + 0.6806995451*c.g + 0.1073969566*c.b, 0.0), 1.0/3.0);
    let s = pow(max(0.0883024619*c.r + 0.2817188376*c.g + 0.6299787005*c.b, 0.0), 1.0/3.0);
    return vec3<f32>(
        0.2104542553*l + 0.7936177850*m - 0.0040720468*s,
        1.9779984951*l - 2.4285922050*m + 0.4505937099*s,
        0.0259040371*l + 0.7827717662*m - 0.8086757660*s,
    );
}

fn oklab_to_linear(lab: vec3<f32>) -> vec3<f32> {
    let ll = lab.x + 0.3963377774*lab.y + 0.2158037573*lab.z;
    let mm = lab.x - 0.1055613458*lab.y - 0.0638541728*lab.z;
    let ss = lab.x - 0.0894841775*lab.y - 1.2914855480*lab.z;
    let l = ll*ll*ll;
    let m = mm*mm*mm;
    let s = ss*ss*ss;
    return vec3<f32>(
        4.0767416621*l - 3.3077115913*m + 0.2309699292*s,
        -1.2684380046*l + 2.6097574011*m - 0.3413193965*s,
        -0.0041960863*l - 0.7034186147*m + 1.7076147010*s,
    );
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let low = c / 12.92;
    let high = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(high, low, c <= vec3<f32>(0.04045));
}

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    var pbr_input = pbr_input_from_standard_material(in, is_front);
#ifdef VERTEX_COLORS
    let payload = in.color;
    pbr_input.material.base_color /= payload;
    let packed = u32(round(payload.a * 36864.0)) - 1u;
    let style = packed / 4096u;
    let structure = f32(packed % 4096u) / 4095.0 * 3.0;
    let mode = style / 3u;
    let finish = style % 3u;
    let freedom = textureSample(tint_mask, tint_sampler, in.uv).r;
    let baked = pbr_input.material.base_color.rgb;
    if mode != 0u {
        let lab = linear_to_oklab(max(baked, vec3<f32>(0.0)));
        var lightness = lab.x;
        var chroma = length(lab.yz);
        var hue = atan2(lab.z, lab.y);
        if mode == 1u {
            let hue_degrees = (payload.r - 1.1920929e-7) * 360.0 - 180.0;
            hue += radians(hue_degrees) * freedom;
            chroma *= mix(1.0, (payload.g - 1.1920929e-7) * 1.8, freedom);
            lightness *= mix(1.0, (payload.b - 1.1920929e-7) * 2.0, freedom);
        } else {
            let target_srgb = payload.rgb * 256.0 / 255.0 - vec3<f32>(1.0 / 255.0);
            let target_lab = linear_to_oklab(srgb_to_linear(clamp(target_srgb, vec3<f32>(0.0), vec3<f32>(1.0))));
            lightness = target_lab.x + (lab.x - chroma_material.x) * structure;
            chroma = length(target_lab.yz);
            hue = atan2(target_lab.z, target_lab.y);
        }
        var rgb = oklab_to_linear(vec3<f32>(lightness, chroma*cos(hue), chroma*sin(hue)));
        for (var i = 0; i < 6; i += 1) {
            if all(rgb >= vec3<f32>(-0.0005)) && all(rgb <= vec3<f32>(1.0005)) { break; }
            chroma *= 0.85;
            rgb = oklab_to_linear(vec3<f32>(lightness, chroma*cos(hue), chroma*sin(hue)));
        }
        pbr_input.material.base_color = vec4<f32>(mix(baked, clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)), freedom), pbr_input.material.base_color.a);
    }
    if finish == 1u {
        pbr_input.material.metallic = 1.0;
    } else if finish == 2u {
        pbr_input.material.metallic = 0.06;
        pbr_input.material.perceptual_roughness = mix(pbr_input.material.perceptual_roughness, 0.42, 0.85);
    }
#endif
    pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);
#ifdef PREPASS_PIPELINE
    return deferred_output(in, pbr_input);
#else
    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
#endif
}
