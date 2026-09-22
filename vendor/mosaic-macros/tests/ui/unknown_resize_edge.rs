use mosaic_layout::Style;
use mosaic_macros::view;
#[allow(unused_imports)]
use mosaic_widgets::{Element, ResizeEdges, ResizeOptions, Ui};

fn main() {
    let ui = Ui::new();
    let _ambient = ui.enter();
    let root = ui.root();
    root.adopt(&view! {
        col resizable:middle {}
    });
}
