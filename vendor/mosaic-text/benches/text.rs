//! Text shaping and glyph placement cost.
//!
//! Shaping is cached per element across frames; *placement* is not — every
//! painted frame re-places every visible glyph. So `place` is the per-frame
//! cost that scales with what is on screen, and `shape` is the per-edit cost.
//! Fonts come from [`FontContext::embedded_only`] so the numbers do not depend
//! on what is installed on the machine.

use std::hint::black_box;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use mosaic_core::{Color, Vector2};
use mosaic_render::Paint;
use mosaic_text::{FontContext, ShapedText, TextStyle};

/// Roughly 2000 glyphs — a dense paragraph, or one screen of chat backlog.
fn paragraph() -> String {
    let sentence = "The quick brown fox jumps over the lazy dog while the \
                    signed distance field resolves every corner analytically. ";
    sentence.repeat(18)
}

fn style() -> TextStyle {
    TextStyle::inherited()
}

fn shaped(ctx: &mut FontContext, text: &str, wrap: Option<f32>) -> ShapedText {
    ctx.shape(text, &style(), wrap)
}

fn shape(c: &mut Criterion) {
    let mut group = c.benchmark_group("text/shape");
    let text = paragraph();
    group.bench_function("paragraph_wrapped_600", |b| {
        let mut ctx = FontContext::embedded_only();
        // Warm the font system so the first sample does not pay database setup.
        drop(shaped(&mut ctx, &text, Some(600.0)));
        b.iter(|| black_box(shaped(&mut ctx, black_box(&text), Some(600.0))));
    });
    group.bench_function("paragraph_unwrapped", |b| {
        let mut ctx = FontContext::embedded_only();
        drop(shaped(&mut ctx, &text, None));
        b.iter(|| black_box(shaped(&mut ctx, black_box(&text), None)));
    });
    // A short label is the common case by count: a data-dense screen is
    // hundreds of these, not one paragraph.
    group.bench_function("short_label", |b| {
        let mut ctx = FontContext::embedded_only();
        drop(shaped(&mut ctx, "Settings", None));
        b.iter(|| black_box(shaped(&mut ctx, black_box("Settings"), None)));
    });
    group.finish();
}

fn place(c: &mut Criterion) {
    let mut group = c.benchmark_group("text/place");
    let text = paragraph();
    // The steady-state per-frame cost: the shaping is reused, only placement
    // and rasterizer lookups repeat. Every glyph currently costs a bitmap
    // clone, an `Arc`, and a `Paint` allocation here.
    group.bench_function("paragraph_2k_glyphs", |b| {
        let mut ctx = FontContext::embedded_only();
        let run = shaped(&mut ctx, &text, Some(600.0));
        let paint = Paint::solid(Color::WHITE);
        // Warm the swash cache so this measures placement, not rasterization.
        drop(run.place(&mut ctx, paint.clone(), Vector2::ZERO, 1.0));
        b.iter(|| {
            black_box(run.place(
                &mut ctx,
                black_box(paint.clone()),
                Vector2::ZERO,
                black_box(1.0),
            ))
        });
    });
    // HiDPI rasterizes at 2x, doubling the bitmap bytes copied per glyph.
    group.bench_function("paragraph_2k_glyphs_hidpi", |b| {
        let mut ctx = FontContext::embedded_only();
        let run = shaped(&mut ctx, &text, Some(600.0));
        let paint = Paint::solid(Color::WHITE);
        drop(run.place(&mut ctx, paint.clone(), Vector2::ZERO, 2.0));
        b.iter(|| {
            black_box(run.place(
                &mut ctx,
                black_box(paint.clone()),
                Vector2::ZERO,
                black_box(2.0),
            ))
        });
    });
    group.finish();
}

/// Shape-then-place from cold, which is what a changed string costs end to end.
fn edit(c: &mut Criterion) {
    let mut group = c.benchmark_group("text/edit");
    group.bench_function("reshape_and_place_label", |b| {
        b.iter_batched_ref(
            || {
                let mut ctx = FontContext::embedded_only();
                drop(shaped(&mut ctx, "warm", None));
                ctx
            },
            |ctx| {
                let run = shaped(ctx, black_box("Unread messages: 12"), Some(300.0));
                black_box(run.place(ctx, Paint::solid(Color::WHITE), Vector2::ZERO, 1.0))
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

criterion_group!(benches, shape, place, edit);
criterion_main!(benches);
