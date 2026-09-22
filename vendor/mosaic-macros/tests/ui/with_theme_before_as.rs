//! The `as` binding comes before `with theme` — after the clause, `as`
//! would read as a Rust cast of the theme expression.

use mosaic_core::theme::Theme;
use mosaic_macros::view;
use mosaic_widgets::{Element, Ui};

struct Fixed;
impl Theme for Fixed {
    fn scheme_tag(&self) -> u32 {
        0
    }
    fn apply(&self) {}
}

fn fixed() -> Fixed {
    Fixed
}

fn main() {
    let ui = Ui::new();
    let _ambient = ui.enter();
    let root = ui.root();
    root.adopt(&view! {
        col {} with theme fixed() as pane
    });
}
