use mosaic_layout::Style;
use mosaic_macros::view;
use mosaic_widgets::{Element, Ui};

fn main() {
    let ui = Ui::new();
    let _ambient = ui.enter();
    let root = ui.root();
    root.adopt(&view! {
        row transition:(width:ease(bezier(x1:120% y1:0% x2:58% y2:100%) 300ms)) {}
    });
}
