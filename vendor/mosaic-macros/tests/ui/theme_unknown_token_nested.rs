//! A misspelled token in a `theme!` is caught by name, with the dotted path
//! the author wrote.

use mosaic_core::theme::{ScalarToken, Theme};
use mosaic_macros::{scheme, theme};

scheme! {
    pub Tokens {
        text {
            size: Scalar = 16,
        },
    }
}

fn main() {
    let _bad = theme! {
        Tokens {
            text { sizes: 20 },
        }
    };
}
