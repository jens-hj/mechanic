use mosaic_core::{MOCHA, State};
use mosaic_layout::*;
use mosaic_macros::{component, view};
use mosaic_render::*;
use mosaic_text::*;
use mosaic_widgets::*;

#[component]
fn Card(children: Children) -> Element {
    view! {
        col {
            children
        }
    }
}

fn main() {
    let ui = Ui::new();
    let _guard = ui.enter();
    let open = State::new(true);
    let enabled = State::new(true);
    let value = State::new(String::new());
    let _simple = view! {
        el width:fill {
            text "Ordinary child" { tooltip "Leaf help" }
            tooltip "Simple tooltip"
        }
    };
    let _rich = view! {
        el width:fill {
            tooltip summary:"Formatting controls"
                side:top align:center gap:8px
                viewport-pad:8px collision:flip-shift
                pad:10px radius:8px fill:MOCHA.surface0
                trigger:manual open:open {
                col gap:4px {
                    text "Arbitrary content"
                }
            }
        }
    };
    let _attachments = view! {
        col {
            Card {
                text "Component child"
                tooltip "Component help"
            }
            button @click:{} "Save" { tooltip "Button help" }
            input value { placeholder font-color:MOCHA.overlay0 tooltip "Input help" }
            toggle enabled { track fill:MOCHA.surface0 tooltip "Toggle help" }
        }
    };
}
