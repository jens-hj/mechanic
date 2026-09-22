use mosaic_layout::*;
use mosaic_macros::view;
use mosaic_text::TextStyle;
use mosaic_widgets::*;

fn main() {
    let ui = Ui::new();
    let _ambient = ui.enter();
    ui.root().adopt(&view! {
        col {
            text "first" as item
            row { text "second" as item }
        }
    });
}
