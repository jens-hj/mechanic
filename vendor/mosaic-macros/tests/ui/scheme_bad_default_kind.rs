//! A default value must fit its token's kind: a length literal cannot
//! default a `Scalar`.

use mosaic_macros::scheme;

scheme! {
    pub Tokens {
        size: Scalar = 10px,
    }
}

fn main() {}
