//! Omitting a token that has no default is still a compile error: the
//! scheme's companion macro names the dotted path of what's missing.

use mosaic_core::Color;
use mosaic_core::theme::{ColorToken, ScalarToken, Theme};
use mosaic_macros::{scheme, theme};

scheme! {
    pub Tokens {
        text {
            col: Color,
            size: Scalar = 16,
        },
    }
}

fn main() {
    let _incomplete = theme! {
        Tokens {
            text { size: 20 },
        }
    };
}
