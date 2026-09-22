// Hint text takes no pointer input, so an interaction state block on the
// `placeholder` part is refused rather than silently never firing.
#[allow(unused_imports)]
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
                placeholder hover { font-color:(Color::WHITE) }
            }
        }
    });
}
