use mosaic_layout::{Edges, Length, Style};
use mosaic_macros::style;
use mosaic_render::PaintSpec;
use mosaic_widgets::{Element, StyleCtx, StyleSet, Visual};

fn blue() -> PaintSpec {
    PaintSpec::default()
}

style! {
    pub #button.base pad:8px
    #button.primary fill:(blue())
}

fn main() {}
