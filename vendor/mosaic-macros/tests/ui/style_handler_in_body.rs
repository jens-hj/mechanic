use mosaic_layout::Style;
use mosaic_macros::style;
use mosaic_render::PaintSpec;
use mosaic_widgets::{Element, StyleCtx, StyleSet, Visual};

fn blue() -> PaintSpec {
    PaintSpec::default()
}

style! {
    pub #card fill:(blue()) @click:{ go() }
}

fn main() {}
