// A moon on a camera-facing quad: a sphere lit by each star of the system,
// so its phases, terminator, and eclipses follow the light it receives.

#import bevy_pbr::{forward_io::VertexOutput, mesh_view_bindings::view}

// Mirrors `MoonUniform` in sky/moons.rs. Directions are in the quad's frame:
// x right, y up, z towards the camera.
struct Moon {
    surface_x: vec4<f32>,
    surface_y: vec4<f32>,
    surface_z: vec4<f32>,
    light_direction: array<vec4<f32>, 3>,
    light_colour: array<vec4<f32>, 3>,
    reflectance: vec4<f32>,
    planetshine: vec4<f32>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> moon: Moon;

const PI: f32 = 3.141592653589793;

// Exposed brightness of a fully lit moon beyond which a dark-adapted eye
// stops following it. The whole disk scales together, so the sunlit side
// keeps its markings and planetshine stays faint beside it.
const ADAPTATION: f32 = 1.2;

fn hash(cell: vec3<f32>) -> f32 {
    return fract(sin(dot(cell, vec3(127.1, 311.7, 74.7))) * 43758.5453);
}

fn value_noise(point: vec3<f32>) -> f32 {
    let cell = floor(point);
    let f = fract(point);
    let u = f * f * (3.0 - 2.0 * f);
    let x00 = mix(hash(cell), hash(cell + vec3(1.0, 0.0, 0.0)), u.x);
    let x10 = mix(hash(cell + vec3(0.0, 1.0, 0.0)), hash(cell + vec3(1.0, 1.0, 0.0)), u.x);
    let x01 = mix(hash(cell + vec3(0.0, 0.0, 1.0)), hash(cell + vec3(1.0, 0.0, 1.0)), u.x);
    let x11 = mix(hash(cell + vec3(0.0, 1.0, 1.0)), hash(cell + vec3(1.0, 1.0, 1.0)), u.x);
    return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}

// Dark plains and bright highlands over a fine grain, fixed to the surface.
fn markings(surface: vec3<f32>, seed: f32) -> f32 {
    let offset = vec3(seed * 173.0, seed * 311.0, seed * 71.0);
    let broad = value_noise(surface * 2.3 + offset) * 0.65 + value_noise(surface * 5.1 + offset) * 0.35;
    let plains = smoothstep(0.48, 0.66, broad);
    let grain = value_noise(surface * 17.0 + offset) * 0.5 + value_noise(surface * 41.0 + offset) * 0.5;
    return (1.0 - 0.45 * plains) * (0.82 + 0.36 * grain);
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let p = vec2(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0);
    let r = length(p);
    let width = max(fwidth(r), 1e-4);
    let coverage = 1.0 - smoothstep(1.0 - width, 1.0, r);
    if coverage <= 0.0 {
        discard;
    }
    let normal = vec3(p, sqrt(max(1.0 - r * r, 0.0)));
    let surface = vec3(
        dot(moon.surface_x.xyz, normal),
        dot(moon.surface_y.xyz, normal),
        dot(moon.surface_z.xyz, normal),
    );
    var illuminance = vec3(0.0);
    for (var index = 0u; index < 3u; index++) {
        let light = moon.light_direction[index];
        illuminance += moon.light_colour[index].rgb * light.w * max(dot(normal, light.xyz), 0.0);
    }
    // The planet's day side lights the hemisphere that always faces it.
    illuminance += vec3(moon.planetshine.x * max(surface.z, 0.0));
    let luminance = illuminance * moon.reflectance.rgb * markings(surface, moon.reflectance.w) / PI;
    var full = moon.planetshine.x;
    for (var index = 0u; index < 3u; index++) {
        let colour = moon.light_colour[index].rgb;
        full += moon.light_direction[index].w * max(max(colour.r, colour.g), colour.b);
    }
    let reflectance = max(max(moon.reflectance.r, moon.reflectance.g), moon.reflectance.b);
    let brightest = full * reflectance / PI * view.exposure;
    let adapted = min(1.0, ADAPTATION / max(brightest, 1e-6));
    return vec4(luminance * view.exposure * adapted, coverage);
}
