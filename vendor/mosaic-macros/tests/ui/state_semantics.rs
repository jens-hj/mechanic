use mosaic_layout::Style;
use mosaic_macros::view;
#[allow(unused_imports)]
use mosaic_widgets::{Element, Role, Ui};

fn main() {
    let ui = Ui::new();
    let _ambient = ui.enter();
    ui.root().adopt(&view! {
        row hover { label:"Hovered" } {}
    });
}
