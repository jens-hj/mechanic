#import bevy_pbr::{
    forward_io::{VertexOutput, FragmentOutput},
    pbr_fragment::pbr_input_from_vertex_output,
    pbr_functions::{alpha_discard, apply_pbr_lighting, main_pass_post_lighting_processing},
    pbr_types::STANDARD_MATERIAL_FLAGS_FOG_ENABLED_BIT,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var grass_base_color: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var dirt_base_color: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var stone_base_color: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(3) var grass_normal: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(4) var dirt_normal: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(5) var stone_normal: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(6) var grass_orm: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(7) var dirt_orm: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(8) var stone_orm: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(9) var terrain_sampler: sampler;

fn projection_weights(normal: vec3<f32>) -> vec3<f32> {
    // The fourth power is exact for signed components after two squares and
    // avoids the substantially more expensive generic pow implementation.
    let squared = normal * normal;
    let sharpened = squared * squared;
    return sharpened / max(dot(sharpened, vec3<f32>(1.0)), 0.0001);
}

fn dominant_projection(projection: vec3<f32>) -> vec3<f32> {
    if projection.x >= projection.y && projection.x >= projection.z {
        return vec3<f32>(1.0, 0.0, 0.0);
    }
    if projection.y >= projection.z {
        return vec3<f32>(0.0, 1.0, 0.0);
    }
    return vec3<f32>(0.0, 0.0, 1.0);
}

fn footprint_adjusted_projection(
    coordinates: vec3<f32>,
    projection: vec3<f32>,
) -> vec3<f32> {
    // Triplanar blending is important up close. Once a fragment covers a
    // sizeable part of a material repeat, the secondary projections are no
    // longer resolvable and only multiply texture bandwidth. Fade to the
    // dominant projection so distant terrain retains every PBR map and its
    // authored mip detail while using one third as many texture samples.
    let footprint = max(length(dpdx(coordinates)), length(dpdy(coordinates)));
    let dominant_weight = smoothstep(0.04, 0.08, footprint);
    return mix(projection, dominant_projection(projection), dominant_weight);
}

fn sample_triplanar(
    map: texture_2d<f32>,
    map_sampler: sampler,
    coordinates: vec3<f32>,
    projection: vec3<f32>,
) -> vec4<f32> {
    var sampled = vec4<f32>(0.0);
    if projection.x > 0.001 {
        sampled += textureSample(map, map_sampler, coordinates.yz) * projection.x;
    }
    if projection.y > 0.001 {
        sampled += textureSample(map, map_sampler, coordinates.xz) * projection.y;
    }
    if projection.z > 0.001 {
        sampled += textureSample(map, map_sampler, coordinates.xy) * projection.z;
    }
    return sampled;
}

fn unpack_normal(sampled: vec3<f32>) -> vec3<f32> {
    return normalize(sampled * 2.0 - 1.0);
}

fn sample_triplanar_normal(
    map: texture_2d<f32>,
    map_sampler: sampler,
    coordinates: vec3<f32>,
    projection: vec3<f32>,
    geometric_normal: vec3<f32>,
) -> vec3<f32> {
    let direction = select(vec3<f32>(-1.0), vec3<f32>(1.0), geometric_normal >= vec3<f32>(0.0));
    var sampled = vec3<f32>(0.0);
    if projection.x > 0.001 {
        let tangent = unpack_normal(textureSample(map, map_sampler, coordinates.yz).rgb);
        sampled += vec3<f32>(
            tangent.z * direction.x,
            tangent.x,
            tangent.y * direction.x,
        ) * projection.x;
    }
    if projection.y > 0.001 {
        let tangent = unpack_normal(textureSample(map, map_sampler, coordinates.xz).rgb);
        sampled += vec3<f32>(
            tangent.x,
            tangent.z * direction.y,
            -tangent.y * direction.y,
        ) * projection.y;
    }
    if projection.z > 0.001 {
        let tangent = unpack_normal(textureSample(map, map_sampler, coordinates.xy).rgb);
        sampled += vec3<f32>(
            tangent.x * direction.z,
            tangent.y,
            tangent.z * direction.z,
        ) * projection.z;
    }
    return normalize(sampled);
}

@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    var pbr_input = pbr_input_from_vertex_output(in, is_front, false);
    pbr_input.material.flags |= STANDARD_MATERIAL_FLAGS_FOG_ENABLED_BIT;
    let coordinates = vec3<f32>(in.uv.x, in.uv_b.x, in.uv.y);
    let projection = footprint_adjusted_projection(
        coordinates,
        projection_weights(normalize(pbr_input.world_normal)),
    );
    var material_weights = max(in.color.rgb, vec3<f32>(0.0));
    material_weights /= max(dot(material_weights, vec3<f32>(1.0)), 0.0001);

    var base_color = vec4<f32>(0.0);
    var surface = vec3<f32>(0.0);
    var mapped_normal = vec3<f32>(0.0);
    // Mesh vertices carry one-hot weights. Interpolation keeps an unused
    // channel exactly zero, so avoid its three triplanar map lookups without
    // changing material blends along layer boundaries.
    if material_weights.x > 0.0 {
        base_color += sample_triplanar(
            grass_base_color,
            terrain_sampler,
            coordinates,
            projection,
        ) * material_weights.x;
        surface += sample_triplanar(
            grass_orm,
            terrain_sampler,
            coordinates,
            projection,
        ).rgb * material_weights.x;
        mapped_normal += sample_triplanar_normal(
            grass_normal,
            terrain_sampler,
            coordinates,
            projection,
            pbr_input.world_normal,
        ) * material_weights.x;
    }
    if material_weights.y > 0.0 {
        base_color += sample_triplanar(
            dirt_base_color,
            terrain_sampler,
            coordinates,
            projection,
        ) * material_weights.y;
        surface += sample_triplanar(
            dirt_orm,
            terrain_sampler,
            coordinates,
            projection,
        ).rgb * material_weights.y;
        mapped_normal += sample_triplanar_normal(
            dirt_normal,
            terrain_sampler,
            coordinates,
            projection,
            pbr_input.world_normal,
        ) * material_weights.y;
    }
    if material_weights.z > 0.0 {
        base_color += sample_triplanar(
            stone_base_color,
            terrain_sampler,
            coordinates,
            projection,
        ) * material_weights.z;
        surface += sample_triplanar(
            stone_orm,
            terrain_sampler,
            coordinates,
            projection,
        ).rgb * material_weights.z;
        mapped_normal += sample_triplanar_normal(
            stone_normal,
            terrain_sampler,
            coordinates,
            projection,
            pbr_input.world_normal,
        ) * material_weights.z;
    }
    pbr_input.material.base_color = base_color;
    pbr_input.diffuse_occlusion = vec3<f32>(surface.r);
    pbr_input.specular_occlusion = surface.r;
    pbr_input.material.perceptual_roughness = surface.g;
    pbr_input.material.metallic = surface.b;

    pbr_input.N = normalize(mapped_normal);

    pbr_input.material.base_color = alpha_discard(
        pbr_input.material,
        pbr_input.material.base_color,
    );
    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
}
