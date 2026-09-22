//! Compile-fail tests: malformed `view!` inputs must fail during expansion
//! with a readable, correctly-spanned message. These cases error before any
//! widget code is generated, so they need none of the widget stack in scope.

#[test]
fn expansion_errors_are_reported() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/*.rs");
    t.pass("tests/pass/headerless_widget_top_node.rs");
    t.pass("tests/pass/component_callback_and_reactive_props.rs");
    t.pass("tests/pass/component_default_props.rs");
    t.pass("tests/pass/event_handler_modifiers.rs");
    t.pass("tests/pass/grid_layout.rs");
    t.pass("tests/pass/text_participation.rs");
    t.pass("tests/pass/disabled.rs");
    t.pass("tests/pass/find_widget.rs");
    t.pass("tests/pass/order_independent_bindings.rs");
    t.pass("tests/pass/tooltip.rs");
}
