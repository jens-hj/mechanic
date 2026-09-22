//! `mosaic-macros` — the optional `view!{}` proc-macro.
//!
//! The builder + closures API is always the foundation and works on its own;
//! this macro is *sugar* on top for concise, nested UI declarations. Every
//! `view!` block expands to exactly the builder calls a caller could have
//! written by hand — there is no hidden runtime, and nothing here is
//! load-bearing for correctness.
//!
//! # Syntax
//!
//! The canonical form is **header-less**: `view! { … }` builds its single
//! top-level element as a detached root and *returns it*, so a screen and
//! every decomposed piece is a `fn(..) -> Element`, composed by nesting:
//!
//! ```ignore
//! fn setup(ui: &Ui, _app: &AppContext) -> Element {
//!     view! {
//!         col gap:16px pad:24px {
//!             row justify:between align:center {
//!                 text font-size:24px font-color:(ink()) "Title"
//!                 text font-size:14px font-color:(faint()) { format!("{} items", $count) }
//!             }
//!             button @click:{ $count += 1 } "Add"
//!             input query
//!             toolbar(count)                    // a decomposed `fn(..) -> Element`, adopted here
//!             if $open {
//!                 text font-color:(faint()) "expanded"
//!             }
//!             for (id, ()) in $rows.into_iter().map(|id| (id, ())) {
//!                 row pad:8px enter-exit:(fade()) {
//!                     text font-size:15px font-color:(ink()) (label(*id))
//!                 }
//!             }
//!             { let list = scroll(parent); virtual_list(&list, /* … */); }
//!         }
//!     }
//! }
//!
//! fn toolbar(count: State<i64>) -> Element {
//!     view! { row gap:8px { /* … */ } }        // header-less: returns the row
//! }
//! ```
//!
//! `App::run`/the CLI-generated hot export take a `setup(&Ui, &AppContext) -> Element` and mount
//! the returned tree under the root (`ui.root()`). `Ui` methods like
//! `set_clear`, `default_text_color`, and fonts stay ordinary calls in `setup`.
//!
//! Every node reads in one order: **the tag, then its attributes / handlers /
//! state blocks, then its value or `{ children }`, then an optional `as name`
//! binding last.** A container's value is its `{ … }` child block; a widget's is
//! its content (a `text` string, a `button` label, an `input` binding).
//!
//! - **No header (canonical)** — `view! { <node> }` builds one top-level
//!   container, widget, component, or call as a detached root and evaluates to
//!   its [`Element`]. Wrap an `if`/`for` or several siblings in a container.
//! - **Call node** — a lowercase call `toolbar(count)` (a `fn(..) -> Element`),
//!   a path call `widgets::pill(x)`, or a parenthesized `(expr)`, whose value is
//!   a detached element the **structural parent adopts** — the everyday way to
//!   compose functions returning `view!{}` subtrees, with no explicit `parent`.
//!   `as name` binds the adopted handle.
//! - **Containers** `row` / `col` / `stack` / `grid` / `el` open a `{ … }` of children;
//!   attributes before the brace lower to [`Style`]/[`Visual`] builder methods
//!   (`gap`, `pad`, `margin`, `grow`, `shrink`, `basis`, `width`, `height`,
//!   `min-width`, `max-width`, `min-height`, `max-height`, `align`, `justify`,
//!   `place`, `fill`, `radius`, `stroke`, `exponent`, `rotate`, `shadow`), and
//!   the flags `clip` / `focus` / `nohit` / `draggable` / `resizable` /
//!   `reorderable` / `selectable` / `searchable`.
//!   `selectable` lets text here and below be dragged over and copied, and
//!   `searchable` lets it be found — separate capabilities, both off until
//!   granted, both covering the subtree and both taking a bool
//!   (`selectable:false` carves a subtree out of an app that granted them
//!   everywhere through `App::text_selectable`/`text_searchable`).
//!   `@click:{ … }` works on every node without adding button semantics;
//!   `@drag:{ move |event, ctx| … }` observes the 8px-slop drag lifecycle.
//!   `@layout-size:size` binds that element's dimensions as read-only reactive
//!   state for its child block, where `$size.width` and `$size.height` are
//!   ordinary tracked reads.
//!   `draggable:(stay)` retains its release offset, `draggable:(snap)` restores
//!   it, and `draggable:(snap: spring())` animates home. Springs accept
//!   `spring()`, the `smooth`/`snappy`/`bouncy` presets, or perceptual records
//!   such as `spring(response:300ms damping:65%)`; either field may be omitted
//!   and reactive fields use `{ response }ms` / `{ damping }%`. Tweens use
//!   `ease(240ms)` for ease-in-out, `ease(out 220ms)` for a named curve, or
//!   `ease(bezier(x1:42% y1:0% x2:58% y2:100%) 300ms)` for custom control points;
//!   durations may use `ms` or `s`, and reactive values retain their units.
//!   `reorderable` moves
//!   direct children internally; `(controlled)` plus `@reorder` delegates the
//!   final source/destination indices to application state, while the field
//!   form configures `mode`, `axis`, `group`, and `motion`. Size attributes take a literal
//!   (`100px`, `50%`), a keyword (`fill`/`auto`/`min-content`/`max-content` on
//!   `width`/`height`/`basis`; `min-content`/`max-content` on min/max attrs),
//!   or a parenthesized `(expr)` escape hatch. `pad`/`margin` take an
//!   edge length literal (`12px`, `10%`), a parenthesized edge field record
//!   (`pad:(horizontal:16px vertical:6px)`, `margin:(left:8px top:4px)`),
//!   a plain `(expr)` escape hatch, or a reactive `{ … }` block. Bare numbers
//!   without `px`/`%` are rejected.
//!   A `grid` requires `cols` or `rows`; track templates accept numeric auto
//!   track counts, grouped tracks, fixed `repeat`, and responsive `auto-fit`
//!   (`grid cols:(repeat(auto-fit minmax(240px 1fr))) gap:12px { … }`).
//! - **Widgets** `text` / `button` / `input` / `img` call the free functions of the
//!   same name; the widget's value comes **after** its attributes. `text` takes
//!   a string, an expression, or a reactive `{ … }` block
//!   (`text font-size:24px "Title"`). `button` takes a string/expression label —
//!   or, in place of a label, a `{ … }` child block it builds its content into
//!   (`button @click:{ save() } { icon(); text "Save" }`), for buttons that are
//!   more than one line of text. `input` takes its `State<String>` binding
//!   (`input radius:8px query`). A direct image literal such as
//!   `img "assets/photo.png"` is checked and embedded at compile time relative
//!   to `CARGO_MANIFEST_DIR`; parentheses explicitly retain runtime loading
//!   (`img (selected_path)`), as do reactive image sources. Every element accepts the inheritable
//!   `TextStyle` attributes `font-color`, `font-family` (including
//!   `sans-serif`/`serif`/`monospace`), `font-size`, `font-stretch`,
//!   `font-style`, `font-weight`, and `line-height`; on a container they
//!   inherit into the children
//!   (`col font-color:(ink()) font-size:18px { text "Inherited" }`).
//!   Every widget also takes the **full container surface**
//!   above (the layout, visual, and element attributes) — a widget returns an
//!   [`Element`], so `button @click:{ save() } width:fill margin:(top:8px) fill:(accent()) "Save"`
//!   or `input radius:8px query` needs no wrapper container. Layout attributes
//!   *patch* the widget's own root style rather than replacing it (a button
//!   keeps its padding and centering); a `fill`/`stroke`/… on `button` or
//!   `input` participates in the widget's one visual fold. Static authored
//!   values set the resting appearance without erasing hover/press/focus;
//!   authored state blocks override the matching widget state, and an explicit
//!   reactive base value owns the property it computes.
//! - **Attached `tooltip`** — every element body may contain one tooltip mixed
//!   with its ordinary children or part runs; leaf widgets use a tooltip-only
//!   body (`text "Save" { tooltip "Keyboard shortcut: Ctrl+S" }`). It is a
//!   semantic attachment, never a layout child. `tooltip "Help"` supplies the
//!   conventional text appearance and accessible summary; rich content uses
//!   `tooltip summary:"Formatting controls" { col { … } }` and accepts the
//!   normal style, visual, font, handler, state, and enter/exit attributes.
//!   Hover plus keyboard focus opens it by default; `trigger:` also accepts
//!   `hover`, `focus`, `always`, or `manual` (manual requires reactive `open:`).
//!   Placement defaults to top-center with an 8px gap and flip/shift collision
//!   handling. `side:`, `align:`, `gap:`, `viewport-pad:`, and `collision:`
//!   cover the common cases; `position:(anchor:(x:… y:…) content:(x:… y:…)
//!   offset:(x:… y:…))` exposes exact reactive anchor/content-point placement.
//! - **`scroll`** wraps its children in a scrolling viewport (the `scroll`
//!   widget). Its attributes style the *scrolling content* — the column being
//!   declared (`gap`, `pad`, a `fill`, …); the viewport is the box the scroll
//!   sits in, since the widget fills its parent — size an enclosing container
//!   to bound it. `as name` binds the `Scroll` handle (`scroll_to`,
//!   `offset`, …).
//! - **`find`** is the bar the platform's find shortcut opens. Alone among the
//!   widgets it takes no state operand — what it searches is the tree, and
//!   whether it is showing belongs to the `Ui` — so a bare `find` is the whole
//!   node. Its `{ … }` styles the `bar`, `field`, `counter`, `prev`, `next`,
//!   and `close` parts, and it exposes `open` and `empty` (typed, but nothing
//!   matched) as state blocks. `as name` binds the `FindBar` handle. Only text
//!   under a `searchable` element is reachable.
//! - **Element attributes** work on any node: `opacity:`, `translate:`, and
//!   `translate-children:` set (or reactively bind, with `{ … }`) the subtree's
//!   uniform opacity and the two paint-time translations; `translate:(x: 6px,
//!   y:4px)` and `translate:(x:50%)` field shorthand lower to
//!   [`Translate`](https://docs.rs/mosaic-layout) (omitted axes default to
//!   zero); reactive bindings may still return `Vector2`. `transition:(spec)`
//!   attaches a `Transition` so covered attributes glide when their bindings
//!   change (`transition:(translate:ease(out 200ms))` is valid shorthand). A
//!   spec covers the node's descendants too — appearance and spacing, never
//!   their size or position — so an app declares its motion once at the root;
//!   a node with a `transition:` of its own uses that instead, and
//!   `transition:{ Transition::new() }` stops an inherited one. Finally,
//!   `enter:(fx)` / `exit:(fx)` / `enter-exit:(fx)` play a
//!   built-in effect (`fade()`, `fly(x, y)`, `slide()`) when `if`/`for` adds
//!   or removes the node. The lifecycle four take a static `(…)` value.
//! - **State styles** — a bare state name followed by a `{ … }` block in
//!   attribute position (`hover { … }`, `pressed { … }`, `focused { … }`,
//!   `disabled { … }`)
//!   overrides style, visual, `opacity`, and `translate` attributes while the
//!   element is hovered / held / keyboard-focused. The block holds the same
//!   `name:value` attributes (no handlers, no lifecycle, no nested states), so
//!   it reads like the plain attributes it layers over:
//!
//!   ```ignore
//!   button @click:{ remove() }
//!       fill:(red())
//!       hover   { fill:(maroon()) }
//!       pressed { fill:(peach()) }
//!       "Delete"
//!
//!   input stroke:(width:1.0 color:edge())
//!       focused { stroke:+(width:2.0 color:blue() offset:2.0) } query
//!
//!   col pad:16px fill:(surface()) transition:(all:ease(120.0))
//!       hover { translate:(y:-2px) shadow:(offset:(x:0.0 y:6.0) blur:12.0 color:shade()) } {
//!       text font-size:14px "card"
//!   }
//!   ```
//!
//!   Precedence is base < `focused` < `hover` < `pressed` < `disabled`; each active state
//!   folds its attributes over the ones below. An ordinary ordered visual
//!   attribute replaces the inherited channel, while `stroke:+…`,
//!   `shadow:+…`, `light:+…`, or `fill:+…` explicitly appends to it. The
//!   keyword is **`focused`**, not
//!   `focus` (which is the focusable flag). All three work on containers,
//!   widgets, and components. A `transition:` on the same node — or on any
//!   ancestor, since specs inherit — covers the state attributes too, so the
//!   swap glides instead of snapping. Omitting a state
//!   block keeps a widget's built-in behavior for that state.
//! - **Styles** — a `#name` at the head of a node's attribute run applies a
//!   reusable attribute run declared with [`style!`], the framework's analogue
//!   of a CSS class:
//!
//!   ```ignore
//!   mosaic::style! {
//!       pub #card
//!           width:fill  pad:(horizontal:8px vertical:6px)  radius:3px
//!           fill:bg-surface
//!           hover { fill:bg-hover }
//!       pub #elevated shadow:+(offset:(x:0 y:6) blur:12 color:shade)
//!   }
//!
//!   view! {
//!       el #card {}                   // as declared
//!       el #card pad:8px {}           // the inline attribute wins
//!       el #card #elevated {}         // append the explicit + shadow
//!       el #card | #elevated {}       // reset, then apply elevated
//!   }
//!   ```
//!
//!   References come first in the run and fold in *beneath* whatever the node
//!   writes itself, so an inline attribute always overrides the style that
//!   declared it. Listing two splices their runs in order, exactly as if the
//!   attributes had been written inline. Ordinary `fill`, `stroke`, `shadow`,
//!   and `light` declarations replace that ordered channel; repeated ordinary
//!   declarations in one authored run form its replacement collection. A
//!   `name:+value` declaration appends, and `|` resets ordered channels before
//!   the style on its right. Other attributes remain last-wins. A style may
//!   also carry runs for the host's named parts and states; see [`style!`] for
//!   the declaration grammar.
//! - **A value's delimiter picks static vs. reactive:** a bare literal or path
//!   (`font-size:24px`, `align:center`) or a parenthesized `(expr)` is static; a
//!   braced `{expr}` is *reactive* and lowers to the `*_dyn` / `style_dyn` /
//!   `visual_dyn` closure form.
//! - **Reactive state sugar** — `$count` reads as `count.get()` and makes a
//!   bare value reactive (`fill:$color`, `text $label`, `if $open`).
//!   `$switch.on` reads a state exposed by a widget/component handle as
//!   `switch.on().get()`; `$size.width` does the same for a layout-size binding.
//!   Assignments (`$count += 1`, `$open = !$open`) and
//!   common collection mutators (`$items.push(value)`) lower through
//!   `State::update`; explicit `.update(...)` remains the escape hatch for
//!   custom mutations. Parenthesize a state read before accessing a field of
//!   its stored value: `($form).name`.
//! - **`shadow:`/`stroke:`/`transition:`/`pad`/`margin:` field shorthand:**
//!   `(offset:(x:0.0 y:6.0) blur:12.0 color:accent())` or `(width:2.0
//!   color:mint() edges:(right bottom))` build the value without naming its type or calling a
//!   constructor; fields separate on whitespace (a comma is still accepted),
//!   may appear in any order, and be omitted (each defaults to zero).
//!   `transition:(width:ease(in-out 240ms) fill:spring(response:300ms damping:65%))`
//!   likewise skips naming `Transition` — each field is an attribute name
//!   chained as `.field(motion)` onto `Transition::new()` (or
//!   `Transition::all(motion)` if an `all` field is given). On `pad`/`margin`,
//!   `(all:8)`, `(horizontal:16 vertical:6)`, and
//!   `(left:10 top:4)` lower to the corresponding `Edges` constructors.
//!   A plain `(expr)` — e.g. `(Shadow::new(...))`, `(some_edges_var)`, or a
//!   variable — still works. Repeated ordinary `shadow`/`stroke` declarations
//!   form one replacement group; prefix the value with `+` to extend an
//!   inherited group (`stroke:+(width:2.0 color:focus())`).
//! - **Conditionals** `if COND { … } else if OTHER { … } else { … }` lowers
//!   to `Element::switch`: exactly one branch is mounted and changing the
//!   selection recreates branch-local state through the normal enter/exit
//!   lifecycle. Attribute values use the same lazy shape and require a final
//!   `else`, for example `width:if $compact { 80px } else { 160px }`; values
//!   recurse through records and retain the surrounding slot's native type.
//!   An `if` in a container attribute run conditionally folds reversible
//!   attributes and interaction-state blocks; its `else` is optional.
//! - **Loops** `for` chooses by its
//!   source: `for (k, v) in {EACH} { … }` (reactive `{ … }` source) lowers to
//!   `keyed` — reconciled by key across frames; `for PAT in ITER { … }` (any
//!   plain iterable, e.g. `0..5` or an array literal) is a build-time loop that
//!   stamps the body into the parent once, for fixed structure.
//! - **Bindings** `let name = EXPR;` binds a value for the siblings that follow
//!   it and their subtrees — Rust's scoping, so nothing before it sees the
//!   name. It is evaluated **once**, while the node is built, which makes a
//!   `$state` read in its initializer a snapshot rather than a subscription.
//!   `let $name = { EXPR };` is the reactive form: it declares a `Derived`
//!   (requiring `T: PartialEq + Clone`) that recomputes when the state it reads
//!   changes, and `$name` reads it back anywhere the binding is in scope.
//!   Neither form is **structural** — it builds no element and takes no child
//!   slot — so a binding can open a target-less `view!` (ahead of its single
//!   top-level node), and a keyed `for` body of `let …;` plus one node still
//!   reconciles through `keyed_view`, without the per-item wrapper a second
//!   node would force. Statements that *do* build children belong in the bare
//!   `{ … }` escape block below instead.
//! - **Components** — a **PascalCase** node (`Card`, `Counter`, …) invokes a
//!   [`#[component]`](macro@component) function. `name:value` attributes are
//!   matched by name: one declared as a prop reaches the prop; any other style,
//!   visual, or element attribute name (`fill`, `pad`, `width`, `opacity`, …)
//!   **patches the returned root** — style/visual attributes fold over whatever
//!   the component itself declared. A name a prop shadows always goes to the
//!   prop, so **props win** by construction. A `{ expression }` value constructs
//!   a `Derived<T>` for a reactive prop; an already-built `State`/`Derived` can
//!   still be passed via `(expr)`. State blocks
//!   (`hover { … }`) and the lifecycle attributes bind on the returned element,
//!   so they work regardless. Handlers and `as` apply to the returned element
//!   like any other node. `#[prop(optional)]` props use their type default when
//!   omitted; `#[prop(default = EXPR)]` props evaluate an authored default and
//!   may refer to earlier props. A missing required prop is a compile error. A
//!   following `{ … }` becomes the
//!   component's `children` closure **iff its contents parse as view! nodes** —
//!   so a sibling `{ … }` escape block (raw Rust) after a childless component
//!   stays an escape, never mis-read as children.
//! - **`children`** — inside a `#[component]` body, this node invokes the
//!   component's `children` parameter, building the caller's nested nodes at that
//!   point in the tree.
//! - **`node … as name`** binds the built element/handle throughout the whole
//!   `view!` invocation, independent of source order or stable-tree nesting.
//!   Stable names are unique invocation-wide: declaring one twice is a focused
//!   compile error. Handlers and reactive expressions may freely form mutual
//!   references; their resolver reads begin only after all stable handles have
//!   been created. Immediate construction reads are dependency-scheduled while
//!   authored child order is restored before the view returns; a genuine
//!   immediate cycle is a compile error naming its path. Bindings under `if`,
//!   `for`, or a component's `children` closure remain local and source-ordered
//!   because those regions may build zero or multiple instances. Ordinary Rust
//!   locals and closure parameters still shadow a stable name normally.
//!   The `as name` always comes last — after a container's `{ children }` or a
//!   widget's value (`col gap:8 { … } as page`, `text "Hi" as label`).
//! - **A bare `{ … }` block** is emitted verbatim with the enclosing element
//!   available as `parent` — the escape hatch for anything the grammar does
//!   not model (`scroll`, `virtual_list`, custom `paint`/`measure`).
//!
//! The expansion uses the builder methods and free functions unqualified, so
//! bring the same names into scope you would use writing the tree by hand
//! (`Style`, `Visual`, `TextStyle`, `FontFamily`, `Align`, `Justify`, `Edges`,
//! `Dimension`, `SizeBound`, `Length`, `Translate`, `Color`, `Shadow`,
//! `Stroke`, `Transition`, and the widget functions). `use mosaic::prelude::*`
//! brings in all of them.
//!
//! [`Element`]: https://docs.rs/mosaic-widgets
//! [`Style`]: https://docs.rs/mosaic-layout
//! [`Visual`]: https://docs.rs/mosaic-widgets

use proc_macro::TokenStream;
use syn::spanned::Spanned;
use syn::visit::Visit;

use mosaic_syntax::View;

mod codegen;
mod component;
mod nudge;
mod preview;
mod scheme;
mod style;
mod theme;

/// Declarative sugar over the widget builder API. See the crate docs for the
/// grammar.
///
/// Errors are reported per node and per attribute: a broken part becomes an
/// inline `compile_error!` while the rest of the block still expands — so
/// type checking, hover, and completions in the surrounding nodes keep
/// working while one of them is mid-edit.
#[proc_macro]
pub fn view(input: TokenStream) -> TokenStream {
    let source = proc_macro2::TokenStream::from(input.clone()).to_string();
    match syn::parse::<View>(input) {
        Ok(view) => codegen::expand(&view, &source).into(),
        Err(err) => err.to_compile_error().into(),
    }
}

/// Declares a design-token scheme: token *names* and kinds, no values.
///
/// ```ignore
/// mosaic::scheme! {
///     pub MyScheme {
///         bg: Color,        // the theme supplies a `Color`
///         surface: Paint,   // any `PaintSpec`: solid or gradient stack
///         space-md: Length,
///         font-body: Scalar, // an `f32`
///         alignment: Align,
///         glass: Filter,
///         depth: Shadow,
///         entrance: Effect,
///     }
/// }
/// ```
///
/// Expands to a theme struct (`MyScheme`, one mandatory field per token — a
/// theme that misses one fails to compile) implementing `Theme`, plus a
/// typed token `const` per name in the invoking module. Those consts are
/// what make `fill:bg` work in `view!` with no extra syntax: a bareword is
/// a Rust path, so a typo is an unresolved-name error at compile time.
///
/// Token names are **kebab-case**, the way every other name in the DSL is
/// spelled, and `view!` references them the same way (`pad:space-md`,
/// `font-size:font-s.big`). The generated Rust name is the snake_case one
/// (`space_md`), which is what Rust code touching the theme struct's fields
/// sees — and, because the two spellings normalize to the same identifier,
/// what the old snake_case spelling in a `view!` still reaches. A token
/// *declared* snake_case compiles but warns, naming the kebab spelling.
///
/// The generated code refers to prelude names (`Color`, `PaintSpec`,
/// `Length`, the token types, `Theme`), like `view!` does — invoke it where
/// `mosaic::prelude::*` is in scope.
///
/// Fill the struct with [`theme!`] to write the values in the `view!`
/// grammar.
#[proc_macro]
pub fn scheme(input: TokenStream) -> TokenStream {
    match scheme::expand(input.into()) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

/// Declares reusable `view!` attribute runs — the framework's analogue of a
/// CSS class.
///
/// ```ignore
/// mosaic::style! {
///     pub #card
///         width:fill  pad:(horizontal:8px vertical:6px)  radius:3px
///         fill:bg-surface
///         hover { fill:bg-hover }
///
///     pub #button.base    pad:(horizontal:12px vertical:6px) radius:6px
///     pub #button.primary fill:accent font-color:on-accent
///
///     pub #switch fill:bg-main radius:8px {
///         track fill:bg-muted radius:8px
///         thumb fill:fg  hover { fill:accent }
///     } on {
///         thumb fill:accent
///     }
/// }
/// ```
///
/// A declaration is shaped like the node it will be applied to: the attribute
/// run on the header line styles the root element, an optional `{ … }` holds
/// runs for its named parts, and trailing `name { … }` blocks scope styling to
/// an interaction state or any state the host exposes. `view!` names one with
/// the same `#`:
///
/// ```ignore
/// view! {
///     el #card {}                       // as declared
///     el #card pad:8px {}               // the inline attribute wins
///     el #card #elevated {}             // explicit :+ layers append
///     el #card | #elevated {}           // reset, then apply elevated
///     toggle #switch $dark_mode
/// }
/// ```
///
/// References come first in the attribute run, and fold in beneath whatever
/// the node writes itself — so an inline attribute always overrides the style
/// that declared it. Listing two styles splices their runs in order, exactly
/// as if the attributes had been written inline. Ordinary `fill`, `stroke`,
/// `shadow`, and `light` declarations replace that ordered channel; repeated
/// ordinary declarations in one authored run form its replacement collection.
/// A `name:+value` declaration appends, and `|` resets ordered channels before
/// the right-hand style. Other attributes remain last-wins.
///
/// Part and state names are open: a style names them before it knows what it
/// will be applied to, so they resolve against the host at run time. A name
/// the host does not expose warns once and is skipped.
///
/// On a built-in widget, part styling participates in the widget's own visual
/// binding: widget base, authored base, widget-selected state, then authored
/// state. A resting palette value therefore cannot disable interaction state;
/// an explicit reactive base value is the intentional property-level override.
///
/// Each declaration expands to a `const StyleSet` — a `pub` one crosses crate
/// boundaries like any other item. The generated code refers to prelude names,
/// like `view!` does; invoke it where `mosaic::prelude::*` is in scope.
#[proc_macro]
pub fn style(input: TokenStream) -> TokenStream {
    match syn::parse::<mosaic_syntax::StyleSheet>(input) {
        Ok(sheet) => style::expand(&sheet).into(),
        Err(err) => err.to_compile_error().into(),
    }
}

/// Builds a theme — a value for every token of a [`scheme!`] — writing the
/// values in the `view!` grammar rather than in constructor calls.
///
/// ```ignore
/// let p = flavor.palette();
/// mosaic::theme! {
///     MyScheme {
///         bg: p.base,
///         surface: linear(angle:90deg stops:(p.mauve p.blue)),
///         space-md: 10px,
///         font-body: 15,
///     }
/// }
/// ```
///
/// It expands to the struct literal it looks like, so rustc still catches a
/// missing token (`E0063`) or a misspelled one (`E0560`). Each value is
/// parsed with the declared kind's exact `view!` grammar. Records, filters,
/// masks, lights, motion, lifecycle effects, enum keywords, gradients, and
/// units therefore lower identically in an attribute, a scheme default, and a
/// theme override.
#[proc_macro]
pub fn theme(input: TokenStream) -> TokenStream {
    match theme::expand(input.into()) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

/// Contextual value parser called by `scheme!`'s generated companion macro.
/// It is public only because procedural macros expand across crate boundaries.
#[doc(hidden)]
#[proc_macro]
pub fn __mosaic_scheme_value(input: TokenStream) -> TokenStream {
    use syn::parse::{Parse, ParseStream};

    struct Input {
        kind: mosaic_syntax::vocabulary::SchemeKind,
        value: proc_macro2::TokenStream,
    }
    impl Parse for Input {
        fn parse(input: ParseStream) -> syn::Result<Self> {
            let kind: syn::Ident = input.parse()?;
            let kind = mosaic_syntax::vocabulary::SchemeKind::from_name(&kind.to_string())
                .ok_or_else(|| input.error("unknown scheme value kind"))?;
            let content;
            syn::bracketed!(content in input);
            let value = content.parse()?;
            Ok(Self { kind, value })
        }
    }

    let parsed = syn::parse::<Input>(input);
    match parsed.and_then(|parsed| {
        let parser = |input: ParseStream| mosaic_syntax::parse_scheme_value(input, parsed.kind);
        syn::parse::Parser::parse2(parser, parsed.value)
    }) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.to_compile_error().into(),
    }
}

/// Compiles a restricted displacement expression into a renderer-neutral field
/// program. The expression is retained canonically for backend lowering.
#[proc_macro]
pub fn displacement(input: TokenStream) -> TokenStream {
    let closure = match syn::parse::<syn::ExprClosure>(input) {
        Ok(value) => value,
        Err(err) => return err.to_compile_error().into(),
    };
    if let Err(error) = validate_field_closure(&closure, false) {
        return error.to_compile_error().into();
    }
    let (code, captures) = match compile_field(&closure) {
        Ok(value) => value,
        Err(error) => return error.to_compile_error().into(),
    };
    quote::quote!(DisplacementField(FieldProgram::new(
        #code, vec![#(FieldUniform::from(#captures)),*]
    )))
    .into()
}

/// Compiles a restricted scalar surface expression into a profile program.
#[proc_macro]
pub fn surface(input: TokenStream) -> TokenStream {
    let closure = match syn::parse::<syn::ExprClosure>(input) {
        Ok(value) => value,
        Err(err) => return err.to_compile_error().into(),
    };
    if let Err(error) = validate_field_closure(&closure, true) {
        return error.to_compile_error().into();
    }
    let (code, captures) = match compile_field(&closure) {
        Ok(value) => value,
        Err(error) => return error.to_compile_error().into(),
    };
    quote::quote!(SurfaceProfile::Program(FieldProgram::new(
        #code, vec![#(FieldUniform::from(#captures)),*]
    )))
    .into()
}

fn compile_field(closure: &syn::ExprClosure) -> syn::Result<(String, Vec<syn::ExprPath>)> {
    let syn::Pat::Ident(parameter) = &closure.inputs[0] else {
        return Err(syn::Error::new(
            closure.inputs[0].span(),
            "field parameter must be an identifier",
        ));
    };
    let mut compiler = FieldCompiler {
        parameter: parameter.ident.to_string(),
        captures: Vec::new(),
    };
    let expression = compiler.expression(&closure.body)?;
    Ok((expression, compiler.captures))
}

struct FieldCompiler {
    parameter: String,
    captures: Vec<syn::ExprPath>,
}

impl FieldCompiler {
    fn expression(&mut self, expr: &syn::Expr) -> syn::Result<String> {
        use syn::Expr;
        Ok(match expr {
            Expr::Lit(value) => quote::quote!(#value).to_string().replace(' ', ""),
            Expr::Paren(value) => format!("({})", self.expression(&value.expr)?),
            Expr::Group(value) => self.expression(&value.expr)?,
            Expr::Path(value) => {
                if value.path.is_ident(&self.parameter) {
                    self.parameter.clone()
                } else {
                    let key = quote::quote!(#value).to_string();
                    let index = self
                        .captures
                        .iter()
                        .position(|capture| quote::quote!(#capture).to_string() == key)
                        .unwrap_or_else(|| {
                            self.captures.push(value.clone());
                            self.captures.len() - 1
                        });
                    format!("uniforms[{index}].x")
                }
            }
            Expr::Field(value) => {
                let member = &value.member;
                let member = quote::quote!(#member).to_string();
                format!("{}.{member}", self.expression(&value.base)?)
            }
            Expr::Unary(value) => {
                let op = &value.op;
                format!("({}{})", quote::quote!(#op), self.expression(&value.expr)?)
            }
            Expr::Binary(value) => {
                let op = &value.op;
                format!(
                    "({} {} {})",
                    self.expression(&value.left)?,
                    quote::quote!(#op),
                    self.expression(&value.right)?
                )
            }
            Expr::Call(value) => {
                let Expr::Path(function) = &*value.func else {
                    return Err(syn::Error::new(value.func.span(), "unsupported field call"));
                };
                let mut name = quote::quote!(#function).to_string();
                if name == "vec2" {
                    name = "vec2<f32>".into();
                }
                let args = value
                    .args
                    .iter()
                    .map(|arg| self.expression(arg))
                    .collect::<syn::Result<Vec<_>>>()?;
                format!("{name}({})", args.join(", "))
            }
            Expr::MethodCall(value) => {
                let method = value.method.to_string();
                let method = if method == "powf" { "pow" } else { &method };
                let mut args = vec![self.expression(&value.receiver)?];
                args.extend(
                    value
                        .args
                        .iter()
                        .map(|arg| self.expression(arg))
                        .collect::<syn::Result<Vec<_>>>()?,
                );
                format!("{method}({})", args.join(", "))
            }
            Expr::If(value) => {
                let [syn::Stmt::Expr(then_value, None)] = value.then_branch.stmts.as_slice() else {
                    return Err(syn::Error::new(
                        value.then_branch.span(),
                        "field selection branches must be expressions",
                    ));
                };
                let Some((_, else_value)) = &value.else_branch else {
                    return Err(syn::Error::new(
                        value.span(),
                        "field selection requires an else branch",
                    ));
                };
                format!(
                    "select({}, {}, {})",
                    self.expression(else_value)?,
                    self.expression(then_value)?,
                    self.expression(&value.cond)?
                )
            }
            _ => {
                return Err(syn::Error::new(
                    expr.span(),
                    "expression is not supported in field programs",
                ));
            }
        })
    }
}

fn validate_field_closure(closure: &syn::ExprClosure, scalar: bool) -> syn::Result<()> {
    if closure.inputs.len() != 1 {
        return Err(syn::Error::new(
            closure.inputs.span(),
            "field programs take exactly one parameter",
        ));
    }
    let mut validator = FieldValidator { error: None };
    validator.visit_expr(&closure.body);
    if let Some(error) = validator.error {
        return Err(error);
    }
    if scalar && matches!(&*closure.body, syn::Expr::Tuple(_) | syn::Expr::Array(_)) {
        return Err(syn::Error::new(
            closure.body.span(),
            "surface programs must return a scalar",
        ));
    }
    Ok(())
}

struct FieldValidator {
    error: Option<syn::Error>,
}

impl FieldValidator {
    fn reject(&mut self, span: proc_macro2::Span, message: &'static str) {
        if self.error.is_none() {
            self.error = Some(syn::Error::new(span, message));
        }
    }
}

impl<'ast> Visit<'ast> for FieldValidator {
    fn visit_expr_call(&mut self, node: &'ast syn::ExprCall) {
        let allowed = match &*node.func {
            syn::Expr::Path(path) => path.path.get_ident().is_some_and(|name| {
                matches!(
                    name.to_string().as_str(),
                    "vec2"
                        | "atan2"
                        | "min"
                        | "max"
                        | "clamp"
                        | "mix"
                        | "smoothstep"
                        | "dot"
                        | "length"
                        | "normalize"
                )
            }),
            _ => false,
        };
        if !allowed {
            self.reject(
                node.func.span(),
                "arbitrary function calls are not allowed in field programs",
            );
        }
        syn::visit::visit_expr_call(self, node);
    }

    fn visit_expr_method_call(&mut self, node: &'ast syn::ExprMethodCall) {
        if !matches!(
            node.method.to_string().as_str(),
            "sin"
                | "cos"
                | "abs"
                | "sqrt"
                | "powf"
                | "exp"
                | "log"
                | "min"
                | "max"
                | "clamp"
                | "mix"
                | "smoothstep"
                | "dot"
                | "length"
                | "normalize"
        ) {
            self.reject(
                node.method.span(),
                "method is not supported in field programs",
            );
        }
        syn::visit::visit_expr_method_call(self, node);
    }

    fn visit_expr_loop(&mut self, node: &'ast syn::ExprLoop) {
        self.reject(node.span(), "loops are not allowed in field programs");
    }
    fn visit_expr_for_loop(&mut self, node: &'ast syn::ExprForLoop) {
        self.reject(node.span(), "loops are not allowed in field programs");
    }
    fn visit_expr_while(&mut self, node: &'ast syn::ExprWhile) {
        self.reject(node.span(), "loops are not allowed in field programs");
    }
    fn visit_expr_macro(&mut self, node: &'ast syn::ExprMacro) {
        self.reject(node.span(), "macros are not allowed in field programs");
    }
    fn visit_expr_assign(&mut self, node: &'ast syn::ExprAssign) {
        self.reject(node.span(), "assignment is not allowed in field programs");
    }
    fn visit_expr_index(&mut self, node: &'ast syn::ExprIndex) {
        self.reject(
            node.span(),
            "indexing and texture access are not allowed in field programs",
        );
    }
}

/// Turns a function into a Mosaic component usable as a PascalCase node in
/// `view!{}`. The function takes typed props and returns a detached `Element`;
/// this attribute generates the props builder the macro constructs at the call
/// site. Props written as `impl Fn(…) + 'static` accept ordinary closures and
/// are stored in the generated props as shared callbacks. Props marked
/// `#[prop(optional)]` use their type default when unset, while
/// `#[prop(default = EXPR)]` props use an authored expression that may refer to
/// earlier props. A `children: Children` parameter is the slot the `children`
/// node fills. Ordinary `///` comments on parameters document props in Mosaic
/// completion and hover.
#[proc_macro_attribute]
pub fn component(_attr: TokenStream, item: TokenStream) -> TokenStream {
    match component::expand(item.into()) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

/// Marks a function as a live preview: a named `Element` a preview host
/// renders on its own, beside the editor.
///
/// ```ignore
/// #[preview]
/// fn card_in_a_frame() -> Element {
///     view! {
///         stack height:100px width:200px {
///             img "assets/img.png"
///             Card
///         }
///     }
/// }
/// ```
///
/// The function is untouched — it takes no arguments and returns `Element`,
/// so anything that builds a view builds a preview, component or fragment.
/// A body that is nothing but a `view!` is *interpreted* from source as you
/// type, with no rebuild; a body that does more than that (locals, calls,
/// loops) is built by this compiled function instead, refreshed when the
/// crate rebuilds.
///
/// `#[preview(width = 390, height = 844)]` pins the frame the preview renders
/// in — both axes or neither. Unset, the card takes its content's size.
///
/// Previews are registered only under `debug_assertions`; release builds carry
/// none of this.
#[proc_macro_attribute]
pub fn preview(attr: TokenStream, item: TokenStream) -> TokenStream {
    match preview::expand(attr.into(), item.into()) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}
