//! `with theme` needs a source: a call or a parenthesized expression.

use mosaic_macros::view;
use mosaic_widgets::{Element, Ui};

fn main() {
    let ui = Ui::new();
    let _ambient = ui.enter();
    let root = ui.root();
    root.adopt(&view! {
        col {} with theme
    });
}
