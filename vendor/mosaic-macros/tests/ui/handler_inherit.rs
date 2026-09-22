use mosaic_widgets::Element;
use mosaic_layout::Style;
use mosaic_macros::view;
use mosaic_widgets::{Ui, button};

fn main() {
    let ui = Ui::new();
    let _ambient = ui.enter();
    let root = ui.root();
    root.adopt(&view! {
        el {
            button @click:inherit "Save"
        }
    });
}
