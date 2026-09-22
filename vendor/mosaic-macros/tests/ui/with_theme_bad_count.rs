//! A clause count needs `ancestors` or `children` after the number.

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
        col {} with theme fixed() 2 parents
    });
}
