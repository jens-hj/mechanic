//! Sibling names must be unique within their group — token or subgroup.

use mosaic_macros::scheme;

scheme! {
    pub Tokens {
        text {
            size: Scalar,
            size: Length,
        }
    }
}

fn main() {}
