use mosaic_widgets::Element;
use mosaic_core::Color;
use mosaic_layout::Style;
use mosaic_macros::view;
use mosaic_widgets::{Ui, Visual};

fn main() {
    let ui = Ui::new();
    let _ambient = ui.enter();
    let root = ui.root();
    root.adopt(&view! {
        el fill:(Color::WHITE)
            hover { fill:(Color::BLACK) }
            hover { fill:(Color::WHITE) } {}
    });
}
