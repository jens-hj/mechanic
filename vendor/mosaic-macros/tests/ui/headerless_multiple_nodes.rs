use mosaic_macros::view;
use mosaic_widgets::{Element, Ui};

fn main() {
    let ui = Ui::new();
    let _ambient = ui.enter();
    let _view: Element = view! {
        col {}
        col {}
    };
}
