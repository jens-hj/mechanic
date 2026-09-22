use mosaic_macros::component;

struct Element;

#[component]
fn Foo(#[prop(nope)] x: i32) -> Element {
    loop {}
}

fn main() {}
