//! A scheme token path left half-written names the path itself, rather than
//! taking the next attribute's name for the missing segment and failing
//! somewhere further along the run.

use mosaic_layout::Style;
use mosaic_macros::view;
use mosaic_text::TextStyle;
use mosaic_widgets::Element;
use mosaic_widgets::Ui;

fn main() {
    let ui = Ui::new();
    let _ambient = ui.enter();
    let root = ui.root();
    root.adopt(&view! {
        col {
            text font-size:font. font-weight:700 "hi"
            text font-size:font. "trailing"
        }
    });
}
