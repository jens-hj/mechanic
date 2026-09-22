//! Compile-pass coverage for component and prop documentation syntax.

#[test]
fn component_prop_doc_comments_compile() {
    let t = trybuild::TestCases::new();
    t.pass("tests/pass/component_prop_docs.rs");
}
