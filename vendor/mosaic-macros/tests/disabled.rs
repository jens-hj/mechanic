use mosaic_core::{Size, State};
use mosaic_layout::Style;
use mosaic_macros::view;
use mosaic_render::PaintCmd;
use mosaic_widgets::*;

mosaic_macros::style! {
    #named_disabled
        disabled { opacity:0.8 }
}

#[test]
fn reactive_disabled_attribute_uses_default_dimming() {
    let ui = Ui::new();
    let _ambient = ui.enter();
    let disabled = State::new(false);
    let node: Element = view! {
        el disabled:{ $disabled } width:20px height:20px {}
    };
    ui.root().adopt(&node);
    ui.frame(Size::new(100.0, 100.0), 1.0).expect("first frame");

    disabled.set(true);
    mosaic_core::reactive::flush();
    let scene = ui
        .frame(Size::new(100.0, 100.0), 1.0)
        .expect("disabled frame");
    assert!(matches!(scene.cmds[0], PaintCmd::PushLayer { opacity, .. } if opacity == 0.5));
}

fn assert_disabled_opacity(ui: &Ui, node: &Element, expected: f32) {
    ui.root().adopt(node);
    let scene = ui.frame(Size::new(100.0, 100.0), 1.0).expect("frame");
    assert!(matches!(scene.cmds[0], PaintCmd::PushLayer { opacity, .. } if opacity == expected));
}

#[test]
fn inline_disabled_opacity_replaces_the_default() {
    let ui = Ui::new();
    let _ambient = ui.enter();
    let node: Element = view! {
        el disabled:true width:20px height:20px disabled { opacity:0.7 } {}
    };
    assert_disabled_opacity(&ui, &node, 0.7);
}

#[test]
fn named_disabled_opacity_replaces_the_default() {
    let ui = Ui::new();
    let _ambient = ui.enter();
    let node: Element = view! {
        el #named_disabled disabled width:20px height:20px {}
    };
    assert_disabled_opacity(&ui, &node, 0.8);
}

#[test]
fn disabled_has_precedence_over_widget_states() {
    let ui = Ui::new();
    let _ambient = ui.enter();
    let on = State::new(true);
    let node = view! {
        toggle disabled on disabled { opacity:0.7 } on { opacity:0.9 }
    };
    assert_disabled_opacity(&ui, node.root(), 0.7);
}
