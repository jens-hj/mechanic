use mosaic_macros::displacement;

fn sample(_: f32) -> f32 { 0.0 }

fn main() {
    let _ = displacement!(|p| sample(p.uv.x));
}
