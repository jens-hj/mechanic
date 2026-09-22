//! A typo'd token name in `view!` is an ordinary unresolved-name error:
//! scheme names are consts, and rustc is the validator.

use mosaic_core::Color;
use mosaic_core::theme::{ColorToken, PaintToken, Theme};
use mosaic_layout::Style;
use mosaic_macros::{scheme, view};
use mosaic_render::PaintSpec;
use mosaic_widgets::{Element, Ui, Visual};

scheme! {
    pub Tokens {
        bg: Paint,
        fg: Color,
    }
}

fn main() {
    let ui = Ui::new();
    let _ambient = ui.enter();
    let root = ui.root();
    root.adopt(&view! {
        el fill:bgg {}
    });
}
