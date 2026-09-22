use mosaic_widgets::Element;
use mosaic_layout::Style;
use mosaic_macros::view;
use mosaic_text::TextStyle;
use mosaic_widgets::{Ui, text};

fn main() {
    let ui = Ui::new();
    let _ambient = ui.enter();
    let root = ui.root();
    root.adopt(&view! {
        el {
            text size:14 color:(mosaic_core::Color::WHITE) family:monospace weight:500
                font-line-height:20 "legacy"
        }
    });
}
