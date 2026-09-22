use mosaic_macros::component;

struct Element;

#[component]
fn Foo(#[prop(optional, default = 3)] value: i32) -> Element {
    let _ = value;
    loop {}
}

fn main() {}
