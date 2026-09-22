use mosaic_core::State;
use mosaic_layout::*;
use mosaic_macros::{style, view};
use mosaic_text::TextStyle;
use mosaic_widgets::{Element, StyleCtx, StyleSet, Ui, Visual};

style! {
    #muted_control
        disabled { opacity:0.7 }
}

fn main() {
    let ui = Ui::new();
    let _ambient = ui.enter();
    let parent = ui.root();
    let saving = State::new(false);

    parent.adopt(&view! {
        col disabled:{ $saving } disabled { opacity:0.7 } {
            el disabled {}
            el #muted_control width:10px height:10px {}
            text disabled { format!("Unavailable") }
        }
    });

    let _: Element = parent.clone();
}
