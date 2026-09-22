use mosaic_core::{IntoFontLength, State};
use mosaic_layout::*;
use mosaic_macros::view;
use mosaic_text::TextStyle;
use mosaic_widgets::{Element, Ui};

fn main() {
    let ui = Ui::new();
    let _ambient = ui.enter();
    let parent = ui.root();
    let reader_mode = State::new(true);

    parent.adopt(&view! {
        col selectable searchable gap:8px {
            text "prose"
            text selectable "an opted-in leaf"
            text searchable font-size:13px "findable, not draggable"
            row selectable:false searchable:false {}
            row selectable:($reader_mode) searchable:{ !reader_mode.get() } {}
        }
    });

    let _: Element = parent.clone();
}
