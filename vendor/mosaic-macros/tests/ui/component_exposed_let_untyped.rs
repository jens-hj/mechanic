// An `#[exposed]` let needs its `State<T>`/`Derived<T>` type written out —
// the annotation is where the macro reads `T` for the `ReadState<T>` accessor.
use mosaic_macros::component;

struct Element;
struct State;
impl State { fn new(_: u32) -> State { State } }

#[component]
fn Foo() -> Element {
    #[exposed]
    let level = State::new(0);
    let _ = level;
    loop {}
}

fn main() {}
