use mosaic_layout::*;
use mosaic_macros::view;
use mosaic_text::TextStyle;
use mosaic_widgets::*;

fn main() {
    let ui = Ui::new();
    let _ambient = ui.enter();
    ui.root().adopt(&view! {
        col {
            text (second.layout_rect().size.width.to_string()) as first
            text (first.layout_rect().size.width.to_string()) as second
        }
    });
}
