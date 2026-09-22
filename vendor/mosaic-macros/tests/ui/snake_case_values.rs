#![allow(unreachable_code, unused_imports)]

use mosaic_layout::Style;
use mosaic_macros::view;
use mosaic_text::TextStyle;
use mosaic_widgets::{Element, ReorderOptions, text};

fn main() {
    let root: Element = todo!();
    root.adopt(&view! {
        row {
            row width:min_content {}
            text font-family:sans_serif "legacy"
            row reorderable:(drag_axis: main) {}
        }
    });
}
