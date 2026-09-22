//! A theme that forgets a token without a default is a compile error: the
//! scheme's companion macro checks the field list and names what's missing.

use mosaic_core::Color;
use mosaic_core::theme::{ColorToken, PaintToken, Theme};
use mosaic_macros::{scheme, theme};
use mosaic_render::PaintSpec;

scheme! {
    pub Tokens {
        bg: Paint,
        fg: Color,
    }
}

fn main() {
    let _incomplete = theme! {
        Tokens {
            fg: Color::WHITE,
        }
    };
}
