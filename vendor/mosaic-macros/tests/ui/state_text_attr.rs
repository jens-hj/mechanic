use mosaic_widgets::Element;
use mosaic_layout::Style;
use mosaic_core::Color;
use mosaic_macros::view;
use mosaic_text::TextStyle;
use mosaic_widgets::{Ui, text};

fn main() {
    let ui = Ui::new();
    let _ambient = ui.enter();
    let root = ui.root();
    root.adopt(&view! {
        el {
            text font-size:14px font-color:(Color::WHITE) hover { font-size:20px } "hi"
        }
    });
}
