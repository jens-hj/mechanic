//! Two nested paths that flatten to the same `theme!` field name would be
//! indistinguishable in a theme literal, so the scheme rejects them.

use mosaic_macros::scheme;

scheme! {
    pub Tokens {
        text {
            size-title: Scalar,
        },
        text-size {
            title: Scalar,
        },
    }
}

fn main() {}
