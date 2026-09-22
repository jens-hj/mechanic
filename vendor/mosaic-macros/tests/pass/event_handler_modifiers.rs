use mosaic_layout::Style;
use mosaic_macros::view;
use mosaic_widgets::input::{PointerButton, PointerEventKind};
use mosaic_widgets::{Element, Ui, button};

fn main() {
    let ui = Ui::new();
    let _ambient = ui.enter();
    let _view: Element = view! {
        stack @pointer:{ |_, _| {} } {
            button @click.stop:{} @pointer-down.stop:{} "Action"
        }
    };
}
