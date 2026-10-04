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
// The terrain's focus in x and z, and in w how far from it terrain is drawn.
@group(#{MATERIAL_BIND_GROUP}) @binding(4) var<uniform> horizon: vec4<f32>;

struct WaterVertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    // Depth of water under the vertex, then its current along x and z.
    @location(8) water: vec3<f32>,
#ifdef WATER_SILT
    // Sediment the water carries, in kg per m³.
    @location(9) silt: f32,
#endif
#ifdef WATER_CHURN
    // How white the water churns where it collides, tumbles or takes a
    // fall, from 0 to 1.
    @location(10) churn: f32,
#endif
}

struct WaterVaryings {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    @location(6) @interpolate(flat) instance_index: u32,
#endif
    @location(8) water: vec3<f32>,
    @location(9) silt: f32,
    @location(10) churn: f32,
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
#ifdef WATER_SILT
    out.silt = vertex.silt;
#else
    out.silt = 0.0;
#endif
#ifdef WATER_CHURN
    out.churn = vertex.churn;
#else
    out.churn = 0.0;
#endif
    return out;
}

// Slope of a few travelling waves at a point.
fn wave_slope(point: vec2<f32>, time: f32) -> vec2<f32> {
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
        let phase = k * dot(directions[wave], point) - speed * k * time;
        let amplitude = 0.012 * lengths[wave];
        slope += directions[wave] * (k * amplitude * cos(phase));
    }
    return slope;
}

// Wind ripples carried along by the current, crossfaded between two
// copies as the current's other passengers are.
fn flowing_slope(point: vec2<f32>, flow: vec2<f32>, time: f32) -> vec2<f32> {
    let speed = length(flow);
    let current = select(vec2<f32>(0.0), flow * min(1.0, FASTEST_DRIFT / speed), speed > 0.0);
    let first = fract(time / FLOW_PERIOD);
    let second = fract(time / FLOW_PERIOD + 0.5);
    let weight = 1.0 - abs(1.0 - 2.0 * first);
    return weight * wave_slope(point - current * first * FLOW_PERIOD, time)
        + (1.0 - weight) * wave_slope(point - current * second * FLOW_PERIOD + vec2<f32>(0.37, 0.61), time);
}

#ifdef WATER_NOISE
// Gradient noise and its slope, `water_noise` in world/water_render.rs.
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var noise_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(3) var noise_sampler: sampler;

// Lattice cells along each edge of the noise texture.
const NOISE_CELLS: f32 = 64.0;

// Gradient noise at a point in lattice cells, about -0.7 to 0.7, then its
// slope along x and y.
fn noise(point: vec2<f32>) -> vec3<f32> {
    return textureSampleLevel(noise_texture, noise_sampler, point / NOISE_CELLS, 0.0).xyz;
}
#endif

// How much of a pattern of this many cycles a metre survives at a pixel's
// size: finer patterns fade out rather than alias.
fn resolved(texel: f32, cycles_per_metre: f32) -> f32 {
    return 1.0 - smoothstep(0.2, 0.5, texel * cycles_per_metre);
}

// Linear colour of water thick with silt.
const MUDDY: vec3<f32> = vec3<f32>(0.16, 0.1, 0.045);

// How fast silt hides what lies under it, per kg of it in each m³ of water
// and per metre of water: suspended mud dims light by about 0.06 m² for
// each gram (Kirk), so the ground fades from sight about as deep as a
// Secchi disc does, 2–3 cm into water carrying a gram a litre and 25 cm
// into water carrying a tenth of that.
const SILT_CLOUDING: f32 = 100.0;

// Linear colour of foam, and of water white with air.
const FOAM: vec3<f32> = vec3<f32>(0.85, 0.9, 0.9);
const AERATED: vec3<f32> = vec3<f32>(0.45, 0.6, 0.58);

const GRAVITY: f32 = 9.81;

// Seconds the surface rides the current before it starts afresh.
const FLOW_PERIOD: f32 = 1.5;

// Fastest current the surface drifts with, in m/s.
const FASTEST_DRIFT: f32 = 2.0;

// Fixed headings in half a turn that patterns are drawn out along.
const HEADINGS: f32 = 8.0;

#ifdef WATER_NOISE
// What the current carries along at a point: wrinkles stretched along it,
// their height's slope in metres per metre, fine chop, and foam.
struct Carried {
    slope: vec2<f32>,
    foam: f32,
    wrinkle: f32,
}

// The surface drawn out along one of the fixed headings, at `at`.
fn drawn_out(
    at: vec2<f32>,
    along: vec2<f32>,
    moving: f32,
    white: f32,
    cover: f32,
    texel: f32,
) -> Carried {
    var out: Carried;
    // Wrinkles 60 cm long and 25 cm across, in two octaves.
    let across = vec2<f32>(-along.y, along.x);
    let frame = vec2<f32>(dot(at, along), dot(at, across));
    let scale = vec2<f32>(1.6, 4.0);
    let first = noise(frame * scale);
    let second = noise(frame * scale * 2.1 + vec2<f32>(17.3, 5.1));
    let height = 0.03 * moving;
    let local = height * (first.yz * scale + 0.5 * second.yz * scale * 2.1);
    out.slope = (along * local.x + across * local.y) * resolved(texel, 8.0);
    out.wrinkle = first.x;
    // Foam: the brightest of a noise, as much of it as `cover` says, in
    // flecks drawn out along the current. Flecks on running water gather
    // into lines that drift with it; whitewater foams soft-edged.
    if cover > 0.005 {
        let lines = smoothstep(-0.1, 0.35, noise(frame * vec2<f32>(0.35, 1.3) + vec2<f32>(4.1, 9.3)).x);
        let share = max(cover * lines * 2.0 * (1.0 - white), cover * white);
        let fleck = noise(frame * vec2<f32>(7.0, 13.0) + vec2<f32>(7.1, 2.9)).x + 0.3 * second.x;
        let threshold = mix(0.5, -0.75, min(share, 1.0));
        let near = smoothstep(threshold, threshold + 0.05 + 0.3 * white, fleck);
        out.foam = mix(share, near, resolved(texel, 13.0));
    }
    return out;
}

// The surface the current carries, sampled where `point` was `age` seconds
// ago, drawn out along the two headings either side of the current, `turn`
// in eighths of half a turn, and `cover` the share of the surface foam
// covers.
fn carried(
    point: vec2<f32>,
    drift: vec2<f32>,
    age: f32,
    turn: f32,
    moving: f32,
    white: f32,
    cover: f32,
    texel: f32,
) -> Carried {
    let at = point - drift * age;
    let lower = floor(turn);
    let blend = smoothstep(0.3, 0.7, turn - lower);
    var out = drawn_out(at, heading(lower), moving, white, cover, texel);
    if blend > 0.0 {
        let next = drawn_out(at, heading(lower + 1.0), moving, white, cover, texel);
        out.slope = mix(out.slope, next.slope, blend);
        out.foam = mix(out.foam, next.foam, blend);
        out.wrinkle = mix(out.wrinkle, next.wrinkle, blend);
    }
    // Chop: whitewater roughened all over.
    if white > 0.01 {
        let chop = noise(at * 9.0 + vec2<f32>(3.7, 11.2));
        out.slope += 0.012 * white * 9.0 * chop.yz * resolved(texel, 9.0);
    }
    return out;
}
#endif

// One of the fixed headings the surface's patterns are drawn out along,
// `turn` eighths of half a turn from x. Patterns follow the current by
// fading between the headings either side of it: turning a pattern to the
// current itself turns it about the world's origin, so a current that
// bends even slightly, far from the origin, tears it into arcs and kinks.
fn heading(turn: f32) -> vec2<f32> {
    let angle = turn * 3.14159265 / HEADINGS;
    return vec2<f32>(cos(angle), sin(angle));
}

@fragment
fn fragment(
    varyings: WaterVaryings,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    // Beyond the terrain's horizon there is no ground to hold water.
    if distance(varyings.world_position.xz, horizon.xz) > horizon.w {
        discard;
    }
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

    let point = varyings.world_position.xz;
    let time = globals.time;
    let depth = max(varyings.water.x, 0.0);
    let flow = varyings.water.yz;
    // Water a few millimetres deep never runs faster than half again its
    // own wave speed, however fast its current is reckoned.
    let reckoned = length(flow);
    let speed = min(reckoned, 1.5 * sqrt(GRAVITY * depth));
    let along = select(vec2<f32>(1.0, 0.0), flow / reckoned, reckoned > 1.0e-3);
    // The current's heading, in eighths of half a turn from x.
    let turn = atan2(along.y, along.x) / 3.14159265 * HEADINGS;
    let froude = speed / sqrt(GRAVITY * max(depth, 0.005));
    let deep_enough = smoothstep(0.02, 0.05, depth);
    // Still to running; rapids; white with air, from the rapids or from
    // collisions, tumbles and falls.
    let moving = smoothstep(0.03, 0.25, speed);
    let rapid = smoothstep(0.7, 1.2, froude) * smoothstep(0.25, 0.8, speed) * deep_enough;
    let churn = clamp(varyings.churn, 0.0, 1.0);
    let white = max(
        0.7 * smoothstep(1.0, 1.5, froude) * smoothstep(0.4, 1.2, speed) * deep_enough,
        churn,
    );
    // Metres a pixel spans: ripples fade with it, so far water reads as a
    // calm mirror instead of aliasing.
    let texel = fwidth(varyings.world_position.x);
    let calm = 1.0 / (1.0 + texel * 4.0);

    // Wind ripples, as tall as the water is deep: puddles and ponds lie
    // glassy, lakes ripple. Running water wears its own wrinkles instead.
    let wind = mix(0.1, 1.0, smoothstep(0.02, 0.6, depth)) * (1.0 - 0.6 * moving);
    var slope = wind * calm * flowing_slope(point, flow, time);

    // What the current carries: two copies, each drifting for a period
    // and starting afresh, crossfaded so neither restart shows. The drift
    // never outgrows a period, so a current that changes from step to step
    // moves the surface a little, not by all the distance since launch.
    let drift = select(vec2<f32>(0.0), flow * min(1.0, FASTEST_DRIFT / reckoned), reckoned > 0.0);
    let cover = max(0.05 * moving, 0.8 * white);
    var foam = 0.0;
    var wrinkle = 0.0;
#ifdef WATER_NOISE
    if moving > 0.01 || white > 0.01 {
        let first_age = fract(time / FLOW_PERIOD);
        let second_age = fract(time / FLOW_PERIOD + 0.5);
        let weight = 1.0 - abs(1.0 - 2.0 * first_age);
        let first = carried(point, drift, first_age * FLOW_PERIOD, turn, moving, white, cover, texel);
        let second = carried(
            point + vec2<f32>(0.37, 0.61),
            drift,
            second_age * FLOW_PERIOD,
            turn,
            moving,
            white,
            cover,
            texel,
        );
        slope += weight * first.slope + (1.0 - weight) * second.slope;
        foam = weight * first.foam + (1.0 - weight) * second.foam;
        wrinkle = weight * first.wrinkle + (1.0 - weight) * second.wrinkle;
    }
#endif

    // Standing waves: crests across fast shallow water that hold still over
    // the bed, as long as the wave that travels upstream as fast as the
    // water runs down, broken up by the wrinkles riding through them. Like
    // the wrinkles, they fade between fixed headings, and between lengths
    // a quarter octave apart: a phase that follows the current's own
    // heading or speed slides wherever either changes.
    if rapid > 0.01 {
        let octaves = 4.0 * log2(clamp(6.2831853 * speed * speed / GRAVITY, 0.15, 2.5));
        let shorter = floor(octaves);
        let longer = smoothstep(0.3, 0.7, octaves - shorter);
        let lower = floor(turn);
        let next = smoothstep(0.3, 0.7, turn - lower);
        let height = 0.3 * rapid * (0.8 + 0.2 * sin(1.3 * time + 3.0 * wrinkle));
        let lengths = vec2<f32>(exp2(shorter / 4.0), exp2((shorter + 1.0) / 4.0));
        let shares = vec2<f32>(1.0 - longer, longer)
            * vec2<f32>(resolved(texel, 1.0 / lengths.x), resolved(texel, 1.0 / lengths.y));
        let first = heading(lower);
        let second = heading(lower + 1.0);
        // Whole crests from the origin dropped, so the cosine never sees a
        // phase of thousands of turns.
        let crests = vec4<f32>(dot(point, first) / lengths, dot(point, second) / lengths);
        let waves = cos(6.2831853 * fract(crests) + 2.5 * wrinkle);
        slope += height * (
            first * ((1.0 - next) * dot(shares, waves.xy))
            + second * (next * dot(shares, waves.zw))
        );
    }

    // Ripples ride on the surface's own slope: water down a chute tilts.
    var normal = normalize(
        normalize(varyings.world_normal) + vec3<f32>(-slope.x, 0.0, -slope.y),
    );
    if !is_front {
        normal = -normal;
    }

    // Shallow water shows the ground through it and fades out to nothing at
    // its edges; deep water is murky.
    let murk = 1.0 - exp(-depth / 3.0);
    let edge = smoothstep(0.003, 0.012, depth);
    // Sediment turns water silty brown and hides the ground under it; air
    // turns it pale and hides the ground too.
    let mud = 1.0 - exp(-SILT_CLOUDING * max(varyings.silt, 0.0) * depth);
    let clear = mix(mix(shallow.rgb, deep.rgb, murk), AERATED, 0.35 * white);
    let colour = mix(clear, MUDDY, mud);
    let opacity = max(max(mix(0.35, 0.9, 1.0 - exp(-depth / 1.2)), 0.95 * mud), 0.85 * white);
    pbr_input.material.base_color = vec4<f32>(
        mix(colour, FOAM, foam),
        edge * max(opacity, foam * 0.9),
    );
    // Still water is a mirror; rapids and foam are matte.
    var roughness = mix(0.02, 0.06, moving);
    roughness = mix(roughness, 0.3, max(0.6 * rapid, white));
    pbr_input.material.perceptual_roughness = mix(roughness, 0.6, foam);
    pbr_input.material.metallic = 0.0;
    pbr_input.material.reflectance = vec3<f32>(0.5);
    pbr_input.N = normal;

    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
}
