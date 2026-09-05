#import bevy_pbr::{
    forward_io::{VertexOutput, FragmentOutput},
    pbr_fragment::pbr_input_from_vertex_output,
    pbr_functions::{alpha_discard, apply_pbr_lighting, main_pass_post_lighting_processing},
    pbr_types::STANDARD_MATERIAL_FLAGS_FOG_ENABLED_BIT,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var grass_base_color: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var dirt_base_color: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var stone_base_color: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(3) var sand_base_color: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(4) var iron_base_color: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(5) var graphite_base_color: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(6) var grass_normal: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(7) var dirt_normal: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(8) var stone_normal: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(9) var grass_orm: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(10) var dirt_orm: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(11) var stone_orm: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(12) var sand_orm: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(13) var iron_orm: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(14) var graphite_orm: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(15) var terrain_sampler: sampler;

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

struct TerrainSample {
    color: vec4<f32>,
    surface: vec3<f32>,
    normal: vec3<f32>,
}

// Each projection decision serves all three maps. Keep the per-projection
// normal normalization and per-material normalization used by the full blend.
fn sample_material(
    color_map: texture_2d<f32>,
    surface_map: texture_2d<f32>,
    normal_map: texture_2d<f32>,
    coordinates: vec3<f32>,
    coordinates_dx: vec3<f32>,
    coordinates_dy: vec3<f32>,
    projection: vec3<f32>,
    geometric_normal: vec3<f32>,
) -> TerrainSample {
    let direction = select(vec3<f32>(-1.0), vec3<f32>(1.0), geometric_normal >= vec3<f32>(0.0));
    var sampled: TerrainSample;
    // A single active projection needs one set of texture instructions. Select
    // the matching coordinate derivatives too: derivatives of selected UVs
    // would otherwise change mip selection across projection boundaries.
    let enabled_axes = projection > vec3<f32>(0.001);
    let active_count = select(0u, 1u, enabled_axes.x) + select(0u, 1u, enabled_axes.y) + select(0u, 1u, enabled_axes.z);
    if active_count == 1u {
        let uv = select(select(coordinates.xy, coordinates.xz, enabled_axes.y), coordinates.yz, enabled_axes.x);
        let uv_dx = select(select(coordinates_dx.xy, coordinates_dx.xz, enabled_axes.y), coordinates_dx.yz, enabled_axes.x);
        let uv_dy = select(select(coordinates_dy.xy, coordinates_dy.xz, enabled_axes.y), coordinates_dy.yz, enabled_axes.x);
        let weight = select(select(projection.z, projection.y, enabled_axes.y), projection.x, enabled_axes.x);
        sampled.color = textureSampleGrad(color_map, terrain_sampler, uv, uv_dx, uv_dy) * weight;
        sampled.surface = textureSampleGrad(surface_map, terrain_sampler, uv, uv_dx, uv_dy).rgb * weight;
        let tangent = normalize(textureSampleGrad(normal_map, terrain_sampler, uv, uv_dx, uv_dy).rgb * 2.0 - 1.0);
        let mapped_x = vec3<f32>(tangent.z * direction.x, tangent.x, tangent.y * direction.x);
        let mapped_y = vec3<f32>(tangent.x, tangent.z * direction.y, -tangent.y * direction.y);
        let mapped_z = vec3<f32>(tangent.x * direction.z, tangent.y, tangent.z * direction.z);
        sampled.normal = normalize(select(select(mapped_z, mapped_y, enabled_axes.y), mapped_x, enabled_axes.x) * weight);
        return sampled;
    }
    if projection.x > 0.001 {
        sampled.color += textureSample(color_map, terrain_sampler, coordinates.yz) * projection.x;
        sampled.surface += textureSample(surface_map, terrain_sampler, coordinates.yz).rgb * projection.x;
        let tangent = normalize(textureSample(normal_map, terrain_sampler, coordinates.yz).rgb * 2.0 - 1.0);
        sampled.normal += vec3<f32>(
            tangent.z * direction.x,
            tangent.x,
            tangent.y * direction.x,
        ) * projection.x;
    }
    if projection.y > 0.001 {
        sampled.color += textureSample(color_map, terrain_sampler, coordinates.xz) * projection.y;
        sampled.surface += textureSample(surface_map, terrain_sampler, coordinates.xz).rgb * projection.y;
        let tangent = normalize(textureSample(normal_map, terrain_sampler, coordinates.xz).rgb * 2.0 - 1.0);
        sampled.normal += vec3<f32>(
            tangent.x,
            tangent.z * direction.y,
            -tangent.y * direction.y,
        ) * projection.y;
    }
    if projection.z > 0.001 {
        sampled.color += textureSample(color_map, terrain_sampler, coordinates.xy) * projection.z;
        sampled.surface += textureSample(surface_map, terrain_sampler, coordinates.xy).rgb * projection.z;
        let tangent = normalize(textureSample(normal_map, terrain_sampler, coordinates.xy).rgb * 2.0 - 1.0);
        sampled.normal += vec3<f32>(
            tangent.x * direction.z,
            tangent.y,
            tangent.z * direction.z,
        ) * projection.z;
    }
    sampled.normal = normalize(sampled.normal);
    return sampled;
}

@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    var pbr_input = pbr_input_from_vertex_output(in, is_front, false);
    pbr_input.material.flags |= STANDARD_MATERIAL_FLAGS_FOG_ENABLED_BIT;
    let coordinates = vec3<f32>(in.uv.x, in.uv_b.x, in.uv.y);
    let coordinates_dx = dpdx(coordinates);
    let coordinates_dy = dpdy(coordinates);
    let projection = footprint_adjusted_projection(
        coordinates,
        projection_weights(normalize(pbr_input.world_normal)),
    );
    var material_weights_a = max(in.color, vec4<f32>(0.0));
    var material_weights_b = vec2<f32>(
        max(in.uv_b.y, 0.0),
        max(1.0 - dot(material_weights_a, vec4<f32>(1.0)) - max(in.uv_b.y, 0.0), 0.0),
    );
    let material_weight_sum = dot(material_weights_a, vec4<f32>(1.0))
        + dot(material_weights_b, vec2<f32>(1.0));
    material_weights_a /= max(material_weight_sum, 0.0001);
    material_weights_b /= max(material_weight_sum, 0.0001);

    var base_color = vec4<f32>(0.0);
    var surface = vec3<f32>(0.0);
    var mapped_normal = vec3<f32>(0.0);
    // Mesh vertices carry one-hot weights. Interpolation keeps an unused
    // channel exactly zero, so avoid its three triplanar map lookups without
    // changing material blends along layer boundaries.
    if material_weights_a.x > 0.0 {
        let sampled = sample_material(
            grass_base_color, grass_orm, grass_normal,
            coordinates, coordinates_dx, coordinates_dy, projection, pbr_input.world_normal,
        );
        base_color += sampled.color * material_weights_a.x;
        surface += sampled.surface * material_weights_a.x;
        mapped_normal += sampled.normal * material_weights_a.x;
    }
    if material_weights_a.y > 0.0 {
        let sampled = sample_material(
            dirt_base_color, dirt_orm, dirt_normal,
            coordinates, coordinates_dx, coordinates_dy, projection, pbr_input.world_normal,
        );
        base_color += sampled.color * material_weights_a.y;
        surface += sampled.surface * material_weights_a.y;
        mapped_normal += sampled.normal * material_weights_a.y;
    }
    if material_weights_a.z > 0.0 {
        let sampled = sample_material(
            stone_base_color, stone_orm, stone_normal,
            coordinates, coordinates_dx, coordinates_dy, projection, pbr_input.world_normal,
        );
        base_color += sampled.color * material_weights_a.z;
        surface += sampled.surface * material_weights_a.z;
        mapped_normal += sampled.normal * material_weights_a.z;
    }
    if material_weights_a.w > 0.0 {
        let sampled = sample_material(
            sand_base_color, sand_orm, dirt_normal,
            coordinates, coordinates_dx, coordinates_dy, projection, pbr_input.world_normal,
        );
        base_color += sampled.color * material_weights_a.w;
        surface += sampled.surface * material_weights_a.w;
        mapped_normal += sampled.normal * material_weights_a.w;
    }
    if material_weights_b.x > 0.0 {
        let sampled = sample_material(
            iron_base_color, iron_orm, stone_normal,
            coordinates, coordinates_dx, coordinates_dy, projection, pbr_input.world_normal,
        );
        base_color += sampled.color * material_weights_b.x;
        surface += sampled.surface * material_weights_b.x;
        mapped_normal += sampled.normal * material_weights_b.x;
    }
    if material_weights_b.y > 0.0 {
        let sampled = sample_material(
            graphite_base_color, graphite_orm, stone_normal,
            coordinates, coordinates_dx, coordinates_dy, projection, pbr_input.world_normal,
        );
        base_color += sampled.color * material_weights_b.y;
        surface += sampled.surface * material_weights_b.y;
        mapped_normal += sampled.normal * material_weights_b.y;
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
