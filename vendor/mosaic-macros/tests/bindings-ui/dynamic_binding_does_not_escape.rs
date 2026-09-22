use mosaic_core::State;
use mosaic_layout::*;
use mosaic_macros::view;
use mosaic_text::TextStyle;
use mosaic_widgets::*;

fn main() {
    let ui = Ui::new();
    let _ambient = ui.enter();
    let shown = State::new(true);
    ui.root().adopt(&view! {
        col {
            if shown.get() {
                text "conditional" as conditional
            }
            text (conditional.layout_rect().size.width.to_string())
        }
    });
}
