use mosaic_core::State;
use mosaic_layout::*;
use mosaic_macros::{component, view};
use mosaic_render::{LightSpec, PaintSpec, ShadowSpec, StrokeSpec};
use mosaic_text::TextStyle;
use mosaic_widgets::*;

#[component]
fn Badge() -> Element {
    view! {
        el width:8px height:8px {}
    }
}

fn detached() -> Element {
    view! {
        el width:8px height:8px {}
    }
}

fn main() {
    let ui = Ui::new();
    let _ambient = ui.enter();
    let text_state = State::new(String::new());

    ui.root().adopt(&view! {
        col {
            button @click:{ later_input.focus(); } "Focus later input" as early_button
            text { format!("{}", later_input.layout_rect().size.width) }
            row {
                button @click:{ right_button.focus(); } "Left" as left_button
                button @click:{ left_button.focus(); } "Right" as right_button
                text { format!("{}", deep.layout_rect().size.width) }
                col {
                    text "nested" as deep
                }
            }
            if State::new(true).get() {
                text "dynamic shadow" as deep
            }
            input text_state as later_input
            scroll {
                text "scroll"
            } as scroller
            Badge as badge
            detached() as called
            { let _: &Element = &early_button; }
            { let _: &Scroll = &scroller; }
            { let _: &BadgeHandle = &badge; }
            { let _: Element = called.clone(); }
            { let deep = 1usize; let _ = deep; }
        }
    });
}
