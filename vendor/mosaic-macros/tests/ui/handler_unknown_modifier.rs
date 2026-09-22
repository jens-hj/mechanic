use mosaic_layout::Style;
use mosaic_macros::view;
use mosaic_widgets::{Element, Ui, button};

fn main() {
    let ui = Ui::new();
    let _ambient = ui.enter();
    let _ = view! {
        button @click.prevent:{} "Action"
    };
}
