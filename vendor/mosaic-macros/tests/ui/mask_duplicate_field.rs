#![allow(unused_imports)]

use mosaic_layout::Style;
use mosaic_macros::view;
use mosaic_render::{MaskComposite, MaskMode};
use mosaic_widgets::{Element, ImageSource, MaskSpec, ObjectFit, Ui};

fn main() {
    let ui = Ui::new();
    let _ambient = ui.enter();
    let root = ui.root();
    root.adopt(&view! {
        row mask:image("mask.png" mode:alpha mode:luminance) {}
    });
}
