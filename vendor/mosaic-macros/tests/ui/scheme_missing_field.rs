//! A theme must supply every token: the generated struct's fields are
//! mandatory, so an incomplete theme is a missing-field error.

use mosaic_core::Color;
use mosaic_core::theme::{ColorToken, PaintToken, Theme};
use mosaic_macros::scheme;
use mosaic_render::PaintSpec;

scheme! {
    pub Tokens {
        bg: Paint,
        fg: Color,
    }
}

fn main() {
    let _incomplete = Tokens {
        fg: Color::WHITE,
    };
}
