// An `input` body accepts only the parts the field exposes.
use mosaic_core::{Color, State};
use mosaic_layout::Style;
use mosaic_macros::view;
use mosaic_widgets::{Element, TextInputOptions, Ui, text_input_with_options};

fn main() {
    let ui = Ui::new();
    let _ambient = ui.enter();
    let root = ui.root();
    let email = State::new(String::new());
    root.adopt(&view! {
        el {
            input placeholder:"Email" email {
                placehodler font-color:(Color::WHITE)
            }
        }
    });
}
