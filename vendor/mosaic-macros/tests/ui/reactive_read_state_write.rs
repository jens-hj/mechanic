use mosaic_core::{ReadState, State};
use mosaic_layout::Style;
use mosaic_macros::view;
use mosaic_widgets::{Element, Ui};

fn main() {
    let ui = Ui::new();
    let _ambient = ui.enter();
    let read = ReadState::from(State::new(1_i32));
    ui.root().adopt(&view! {
        col {
            { $read += 1; }
        }
    });
}
