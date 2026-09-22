#[test]
fn stable_binding_errors_are_reported() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/bindings-ui/*.rs");
    t.pass("tests/pass/order_independent_bindings.rs");
}
