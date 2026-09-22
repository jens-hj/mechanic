//! `#[component]` — lowers a function into a Mosaic component: a plain builder
//! function plus a typed props builder that `view!{}` constructs when the
//! component is used as a PascalCase node.
//!
//! A component author writes
//!
//! ```ignore
//! #[component]
//! fn Card(
//!     /// The heading displayed at the top.
//!     title: String,
//!     /// Optional supporting text.
//!     #[prop(optional)]
//!     subtitle: String,
//!     children: Children,
//! ) -> Element {
//!     view! {
//!         col gap:4px pad:16px fill:(surface()) {
//!             text font-size:18px (title)
//!             children
//!         }
//!     }
//! }
//! ```
//!
//! Prop `///` comments are shown by Mosaic completion and hover. They work on
//! `#[component]` parameters because this macro consumes them while generating
//! the component API.
//!
//! The body is header-less `view!`: it builds a detached element and returns it,
//! and the call site (a PascalCase node) adopts it into the structural parent —
//! so a component needs no `parent` parameter. This macro emits:
//!
//! - the function rewritten to `fn Card(props: CardProps) -> Element` with the
//!   props destructured back into the parameter names the body expects;
//! - a `CardProps` struct and a `CardProps::builder()` whose setters are named
//!   after the props. The builder tracks per-field set/unset state in its type,
//!   so `build()` only type-checks once every **required** prop is set — a
//!   missing required prop is a compile error. Optional props use either the
//!   type default (`#[prop(optional)]`) or an authored expression
//!   (`#[prop(default = EXPR)]`) when unset.

use proc_macro2::{Span, TokenStream};
use quote::{ToTokens, format_ident, quote};
use syn::spanned::Spanned;
use syn::visit::Visit;
use syn::{
    Error, Expr, FnArg, Ident, ItemFn, Pat, Result, ReturnType, Stmt, TraitBound, Type, parse2,
};

/// How an unset prop is populated by the generated builder.
enum PropDefault {
    /// `#[prop(optional)]`: use the type's ordinary `Default` implementation.
    Type,
    /// `#[prop(default = EXPR)]`: evaluate the authored expression lazily.
    Expr(Expr),
}

pub fn expand(item: TokenStream) -> Result<TokenStream> {
    let func: ItemFn = parse2(item)?;
    Component::parse(func)?.expand()
}

/// One prop parsed off a component function's parameter list.
struct Prop {
    name: Ident,
    /// The concrete type stored in the generated props struct. Callback props
    /// authored as `impl Fn…` are stored as the shared `Rc<dyn Fn…>` form.
    ty: Type,
    /// The `Fn…` bound of an authored callback prop, used by its setter to
    /// accept an ordinary closure before erasing it into [`Self::ty`].
    callback: Option<TraitBound>,
    /// Authored `///` comments, copied onto the generated builder setter so
    /// rust-analyzer hover remains useful inside expanded `view!` code.
    docs: Vec<syn::Attribute>,
    default: Option<PropDefault>,
    /// `#[prop(exposed)]`: the prop (a `State<T>`/`Derived<T>`) is surfaced
    /// on the generated handle as a read-only stylable state.
    exposed: bool,
    /// The `children: Children` slot, whose setter takes a build closure rather
    /// than a plain value.
    children: bool,
}

/// One `#[exposed] let name: State<T> = …;` inner state, surfaced on the
/// generated handle as a read-only stylable state.
struct InnerState {
    name: Ident,
    /// The `T` inside the annotated `State<T>`/`Derived<T>`.
    inner_ty: Type,
}

struct Component {
    func: ItemFn,
    props: Vec<Prop>,
    /// Exposed inner states, in declaration order.
    inner_states: Vec<InnerState>,
    /// `as pub` part names found in the body's `view!`, in source order.
    parts: Vec<(String, Span)>,
}

impl Component {
    fn parse(func: ItemFn) -> Result<Self> {
        if !func.sig.generics.params.is_empty() {
            return Err(Error::new(
                func.sig.generics.span(),
                "a `#[component]` function cannot be generic",
            ));
        }
        match &func.sig.output {
            ReturnType::Type(_, ty) if is_element(ty) => {}
            other => {
                return Err(Error::new(
                    other.span(),
                    "a `#[component]` function must return `Element`",
                ));
            }
        }

        // Every parameter is a prop — a component builds a detached element and
        // is adopted by its structural parent, so it takes no `parent`.
        let mut props = Vec::new();
        for arg in func.sig.inputs.iter() {
            let FnArg::Typed(pat) = arg else {
                return Err(Error::new(arg.span(), "unexpected `self` parameter"));
            };
            let Pat::Ident(pat_ident) = &*pat.pat else {
                return Err(Error::new(
                    pat.pat.span(),
                    "a component prop must be a plain `name: Type` parameter",
                ));
            };
            let name = pat_ident.ident.clone();
            let (default, exposed) = prop_options(&pat.attrs)?;
            let (ty, callback) = callback_storage(&pat.ty)?;
            let docs = pat
                .attrs
                .iter()
                .filter(|attr| attr.path().is_ident("doc"))
                .cloned()
                .collect();
            if exposed {
                if reactive_inner(&pat.ty).is_none() {
                    return Err(Error::new(
                        pat.ty.span(),
                        "an exposed prop must be a `State<T>` or `Derived<T>` \
                         with its type written out",
                    ));
                }
                check_exposed_state_name(&name)?;
            }
            let children = name == "children";
            props.push(Prop {
                name,
                ty,
                callback,
                docs,
                default,
                exposed,
                children,
            });
        }

        let mut func = func;
        let inner_states = collect_inner_states(&mut func)?;
        let parts = collect_view_parts(&func)?;

        // Every exposure becomes a handle accessor, so all names share one
        // namespace.
        let mut names: Vec<(String, Span)> = parts.clone();
        for state in &inner_states {
            names.push((state.name.to_string(), state.name.span()));
        }
        for prop in props.iter().filter(|prop| prop.exposed) {
            names.push((prop.name.to_string(), prop.name.span()));
        }
        for (index, (name, span)) in names.iter().enumerate() {
            if names[..index].iter().any(|(prior, _)| prior == name) {
                return Err(Error::new(
                    *span,
                    format!(
                        "`{name}` is exposed twice — parts, exposed props, and exposed \
                         states share the handle's accessor namespace"
                    ),
                ));
            }
        }

        Ok(Component {
            func,
            props,
            inner_states,
            parts,
        })
    }

    /// Whether the component exposes anything — and so returns a generated
    /// handle instead of a bare `Element`.
    fn exposes(&self) -> bool {
        !self.parts.is_empty()
            || !self.inner_states.is_empty()
            || self.props.iter().any(|prop| prop.exposed)
    }

    fn expand(&self) -> Result<TokenStream> {
        let vis = &self.func.vis;
        let name = &self.func.sig.ident;
        let output = &self.func.sig.output;
        let block = &self.func.block;
        let attrs = &self.func.attrs;

        let props_name = format_ident!("{}Props", name);
        let handle_name = format_ident!("{}Handle", name);
        let builder_name = format_ident!("{}Builder", props_name);
        let present_trait = format_ident!("__{}Present", props_name);
        let slot_trait = format_ident!("__{}Slot", props_name);

        let field_names: Vec<&Ident> = self.props.iter().map(|p| &p.name).collect();
        let field_types: Vec<&Type> = self.props.iter().map(|p| &p.ty).collect();
        let n = self.props.len();

        // The rewritten function: props destructured back into the names the
        // body expects, so authoring code is unchanged. The body runs inside
        // a closure so an early `return` still flows into the patch
        // application on the returned root. A component that exposes parts or
        // states returns a generated handle instead of the bare root; see
        // [`handle_pieces`].
        let (handle_items, rewritten) = if self.exposes() {
            self.handle_pieces(&props_name)?
        } else {
            let handle_alias = quote! {
                #[doc(hidden)]
                #vis type #handle_name = Element;
            };
            let rewritten = quote! {
                #(#attrs)*
                #[allow(non_snake_case)]
                #vis fn #name(__props: #props_name) #output {
                    let #props_name { #(#field_names,)* __patch } = __props;
                    let __root: Element = (move || #block)();
                    __root.inspection_component(stringify!(#name));
                    __patch.apply(&__root);
                    __root
                }
            };
            (handle_alias, rewritten)
        };

        // Support traits, generated per-component so the output is self-contained
        // (no framework path to resolve): `Present` is satisfied only by a set
        // slot `(T,)`, gating required props at compile time; `Slot` extracts a
        // value from a set slot or falls back to a default for an unset one.
        let support = quote! {
            #[doc(hidden)]
            #vis trait #present_trait<T> {
                fn take(self) -> T;
            }
            impl<T> #present_trait<T> for (T,) {
                fn take(self) -> T { self.0 }
            }
            #[doc(hidden)]
            #vis trait #slot_trait<T> {
                fn get<F: ::core::ops::FnOnce() -> T>(self, default: F) -> T;
            }
            impl<T> #slot_trait<T> for () {
                fn get<F: ::core::ops::FnOnce() -> T>(self, default: F) -> T { default() }
            }
            impl<T> #slot_trait<T> for (T,) {
                fn get<F: ::core::ops::FnOnce() -> T>(self, _default: F) -> T { self.0 }
            }
        };

        let props_struct = quote! {
            #vis struct #props_name {
                #(#field_names: #field_types,)*
                __patch: ElementPatch,
            }
        };

        // The initial builder state: one unset `()` slot per field, plus the
        // patch collecting any style/visual/element attributes the call site
        // writes (see [`patch_setters`]).
        let unset_ty = std::iter::repeat_n(quote!(()), n);
        let unset_val = std::iter::repeat_n(quote!(()), n);
        let builder_ctor = quote! {
            #[doc(hidden)]
            #vis struct #builder_name<S> {
                fields: S,
                patch: ElementPatch,
            }
            impl #props_name {
                #vis fn builder() -> #builder_name<( #(#unset_ty,)* )> {
                    #builder_name { fields: ( #(#unset_val,)* ), patch: ElementPatch::new() }
                }
            }
        };

        let setters = self
            .props
            .iter()
            .enumerate()
            .map(|(i, prop)| self.setter(i, prop, &builder_name))
            .collect::<Vec<_>>();

        let patch_setters = self.patch_setters(&builder_name);

        let build = self.build_impl(&props_name, &builder_name, &present_trait, &slot_trait);

        let preview = self.preview_registration(&props_name);

        Ok(quote! {
            #rewritten
            #handle_items
            #props_struct
            #support
            #builder_ctor
            #(#setters)*
            #patch_setters
            #build
            #preview
        })
    }

    /// The constructor a live preview calls, and its registration.
    ///
    /// A preview reads props out of DSL text, so each one has to cross into
    /// the declared Rust type. Which types can make that crossing is decided
    /// *here*, on the written spelling, rather than through a trait bound on
    /// the prop — a bound would turn "this prop cannot come from a preview"
    /// into a compile error in a crate that is otherwise perfectly valid.
    /// A prop this cannot read leaves the component registered but refusing,
    /// with a message naming the prop, so the preview says why instead of
    /// rendering something quietly wrong.
    ///
    /// The props struct is built field by field rather than through the
    /// typestate builder: the builder's `build()` only type-checks once every
    /// required prop is set, which a runtime-driven caller cannot promise.
    fn preview_registration(&self, props_name: &Ident) -> TokenStream {
        let name = &self.func.sig.ident;
        let widgets = widgets_path();
        let constructor = format_ident!("__mosaic_preview_{}", name);

        let field_names: Vec<&Ident> = self.props.iter().map(|prop| &prop.name).collect();
        let body = match self.preview_fields(&widgets) {
            Ok(bindings) => {
                let root = if self.exposes() {
                    quote!(ComponentHandle::root(&#name(__props)).clone())
                } else {
                    quote!(#name(__props))
                };
                // Props bind in authored order, as the generated builder's
                // `build()` does, so a default that refers to an earlier prop
                // (`#[prop(default = size * 0.14)]`) resolves the same way here.
                quote! {
                    #(#bindings)*
                    let __props = #props_name {
                        #(#field_names,)*
                        __patch: ElementPatch::new(),
                    };
                    ::core::result::Result::Ok(#root)
                }
            }
            Err(reason) => quote! {
                let _ = __args;
                ::core::result::Result::Err(::std::string::String::from(#reason))
            },
        };

        quote! {
            #[cfg(debug_assertions)]
            #[doc(hidden)]
            #[allow(non_snake_case)]
            fn #constructor(
                __args: &mut #widgets::preview::Args,
            ) -> ::core::result::Result<Element, ::std::string::String> {
                #body
            }

            #[cfg(debug_assertions)]
            #widgets::preview::inventory::submit! {
                #widgets::preview::Component {
                    name: stringify!(#name),
                    build: #constructor,
                }
            }
        }
    }

    /// One binding per prop, in authored order, or why this component cannot be
    /// built from a preview at all.
    fn preview_fields(
        &self,
        widgets: &TokenStream,
    ) -> std::result::Result<Vec<TokenStream>, String> {
        self.props
            .iter()
            .map(|prop| self.preview_field(prop, widgets))
            .collect()
    }

    fn preview_field(
        &self,
        prop: &Prop,
        widgets: &TokenStream,
    ) -> std::result::Result<TokenStream, String> {
        let name = &prop.name;
        let key = name.to_string();
        let ty = &prop.ty;

        // The children slot is a build closure, not a value: the preview's
        // authored `{ … }` block fills it.
        if prop.children {
            return Ok(quote! {
                let #name: #ty = __args
                    .take_children()
                    .unwrap_or_else(|| ::std::boxed::Box::new(|_: &Element| {}));
            });
        }

        // A callback cannot come from DSL text, but a preview does not need it
        // to: it stands in a no-op so the component still renders and still
        // responds to being pressed. Only callbacks that return nothing are
        // safe to synthesize — anything else would need a value invented here.
        if let Some(bound) = &prop.callback {
            let arity = callback_arity(bound)?;
            let ignored = (0..arity).map(|_| quote!(_));
            return Ok(match &prop.default {
                Some(PropDefault::Expr(expr)) => {
                    quote! { let #name: #ty = ::std::rc::Rc::new(#expr); }
                }
                _ => quote! { let #name: #ty = ::std::rc::Rc::new(|#(#ignored),*| {}); },
            });
        }

        let readable = readable_type(ty).ok_or_else(|| {
            format!(
                "`{}` takes `{}: {}`, a type a preview cannot supply",
                self.func.sig.ident,
                key,
                ty.to_token_stream()
            )
        })?;

        let read = match readable {
            Readable::Value => quote! { __args.prop::<#ty>(#key)? },
            // A state prop is a preview-owned cell seeded with the authored
            // literal, so the component behaves as it would with real state.
            Readable::State(inner) => quote! {
                __args.prop::<#inner>(#key)?.map(State::new)
            },
        };
        let _ = widgets;

        Ok(match &prop.default {
            Some(PropDefault::Type) => quote! {
                let #name: #ty = #read.unwrap_or_else(<#ty as ::core::default::Default>::default);
            },
            Some(PropDefault::Expr(expr)) => quote! {
                let #name: #ty = #read.unwrap_or_else(|| #expr);
            },
            None => quote! {
                let #name: #ty =
                    #read.ok_or_else(|| ::std::format!("prop `{}` is required", #key))?;
            },
        })
    }

    /// The exposing form: the `{Name}Handle` struct (root + one field per
    /// exposed part and state, with accessors and a `ComponentHandle` impl)
    /// and the function rewritten to build it — parts resolved once from the
    /// root's registry, prop states wrapped directly (they are `Copy`, so
    /// they survive the body closure's move), inner states smuggled out by
    /// rewriting the body's tail expression into a tuple.
    fn handle_pieces(&self, props_name: &Ident) -> Result<(TokenStream, TokenStream)> {
        let vis = &self.func.vis;
        let name = &self.func.sig.ident;
        let attrs = &self.func.attrs;
        let handle_name = format_ident!("{}Handle", name);
        let field_names: Vec<&Ident> = self.props.iter().map(|p| &p.name).collect();

        let part_fields: Vec<Ident> = self
            .parts
            .iter()
            .map(|(part, span)| Ident::new(part, *span))
            .collect();
        let part_strs: Vec<&String> = self.parts.iter().map(|(part, _)| part).collect();
        let part_expects: Vec<String> = self
            .parts
            .iter()
            .map(|(part, _)| {
                format!(
                    "exposed part `{part}` is registered by the body's view! \
                     (is the view! the body's tail expression?)"
                )
            })
            .collect();

        let exposed_props: Vec<&Prop> = self.props.iter().filter(|prop| prop.exposed).collect();
        let prop_fields: Vec<&Ident> = exposed_props.iter().map(|prop| &prop.name).collect();
        let prop_inner: Vec<Type> = exposed_props
            .iter()
            .map(|prop| reactive_inner(&prop.ty).expect("validated at parse"))
            .collect();

        let state_fields: Vec<&Ident> = self.inner_states.iter().map(|s| &s.name).collect();
        let state_inner: Vec<&Type> = self.inner_states.iter().map(|s| &s.inner_ty).collect();

        // A `style!` state block names the state it scopes to, so the boolean
        // exposures are registered on the root under those names. Only `bool`
        // ones qualify — a state block is a condition, and the annotated type
        // is written out, so this reads it rather than guessing.
        let flag_fields: Vec<&Ident> = exposed_props
            .iter()
            .zip(&prop_inner)
            .filter(|(_, inner)| is_bool(inner))
            .map(|(prop, _)| &prop.name)
            .chain(
                self.inner_states
                    .iter()
                    .filter(|state| is_bool(&state.inner_ty))
                    .map(|state| &state.name),
            )
            .collect();
        let flag_names: Vec<String> = flag_fields.iter().map(|name| name.to_string()).collect();

        let doc = format!(
            "Handle to a built [`{name}`]: the root element plus accessors for its \
             exposed parts and read-only states, as `view!` styles them."
        );
        let handle_items = quote! {
            #[doc = #doc]
            #[derive(Clone)]
            #[allow(non_snake_case)]
            #vis struct #handle_name {
                root: Element,
                #(#part_fields: Element,)*
                #(#prop_fields: ReadState<#prop_inner>,)*
                #(#state_fields: ReadState<#state_inner>,)*
            }
            impl #handle_name {
                /// The component's root element.
                #vis fn root(&self) -> &Element { &self.root }
                #(#vis fn #part_fields(&self) -> &Element { &self.#part_fields })*
                #(#vis fn #prop_fields(&self) -> ReadState<#prop_inner> { self.#prop_fields })*
                #(#vis fn #state_fields(&self) -> ReadState<#state_inner> { self.#state_fields })*
            }
            impl ComponentHandle for #handle_name {
                fn root(&self) -> &Element { &self.root }
            }
        };

        // With inner states, the tail expression becomes a tuple carrying
        // them out of the body closure alongside the built root. A trailing
        // `view! { … }` parses as a macro statement, not an expression
        // statement — both count as the tail.
        let mut block = (*self.func.block).clone();
        if !self.inner_states.is_empty() {
            let tail: syn::Expr = match block.stmts.last() {
                Some(Stmt::Expr(expr, None)) => expr.clone(),
                Some(Stmt::Macro(stmt)) if stmt.semi_token.is_none() => {
                    syn::Expr::Macro(syn::ExprMacro {
                        attrs: stmt.attrs.clone(),
                        mac: stmt.mac.clone(),
                    })
                }
                _ => {
                    return Err(Error::new(
                        self.func.block.span(),
                        "a component with `#[exposed]` states must end with a tail \
                         expression (the built view), so the states can flow into \
                         the handle",
                    ));
                }
            };
            let idents = &state_fields;
            *block.stmts.last_mut().expect("tail exists") =
                Stmt::Expr(syn::parse_quote!((#tail #(, #idents)*)), None);
        }
        let destructure = if self.inner_states.is_empty() {
            quote!(let __built = (move || #block)();)
        } else {
            quote!(let (__built #(, #state_fields)*) = (move || #block)();)
        };

        let rewritten = quote! {
            #(#attrs)*
            #[allow(non_snake_case)]
            #vis fn #name(__props: #props_name) -> #handle_name {
                let #props_name { #(#field_names,)* __patch } = __props;
                #destructure
                let __root: Element = ComponentHandle::root(&__built).clone();
                __root.inspection_component(stringify!(#name));
                __patch.apply(&__root);
                #(__root.expose_state_flag(#flag_names, move || #flag_fields.get());)*
                #handle_name {
                    #(#part_fields: __root.part(#part_strs).expect(#part_expects),)*
                    #(#prop_fields: ReadState::from(#prop_fields),)*
                    #(#state_fields: ReadState::from(#state_fields),)*
                    root: __root,
                }
            }
        };
        Ok((handle_items, rewritten))
    }

    /// One typestate setter: callable only while its own slot is unset (`()`),
    /// leaving the other slots' states untouched.
    fn setter(&self, i: usize, prop: &Prop, builder_name: &Ident) -> TokenStream {
        let n = self.props.len();
        let others: Vec<Ident> = (0..n)
            .filter(|&j| j != i)
            .map(|j| format_ident!("S{}", j))
            .collect();

        // Input state tuple: slot i is `()`, the rest are the generic `Sj`.
        let in_slots = (0..n).map(|j| {
            if j == i {
                quote!(())
            } else {
                let s = format_ident!("S{}", j);
                quote!(#s)
            }
        });
        // Output state tuple: slot i becomes `(Ty,)`.
        let ty = &prop.ty;
        let out_slots = (0..n).map(|j| {
            if j == i {
                quote!((#ty,))
            } else {
                let s = format_ident!("S{}", j);
                quote!(#s)
            }
        });

        // Rebuild the fields tuple, replacing slot i.
        let binders: Vec<Ident> = (0..n).map(|j| format_ident!("__f{}", j)).collect();
        let rebuilt = (0..n).map(|j| {
            if j == i {
                quote!(__value)
            } else {
                let b = &binders[j];
                quote!(#b)
            }
        });

        let name = &prop.name;
        let docs = &prop.docs;
        let vis = &self.func.vis;
        let (arg, value) = if let Some(callback) = &prop.callback {
            (
                quote!(#name: impl #callback + 'static),
                quote! {{
                    let __callback: #ty = ::std::rc::Rc::new(#name);
                    (__callback,)
                }},
            )
        } else if prop.children {
            // The children slot takes a build closure; the coercion to the
            // `Children` boxed trait object happens through the annotated let.
            (
                quote!(#name: impl ::core::ops::FnOnce(&Element) + 'static),
                quote! {{
                    let __c: #ty = ::std::boxed::Box::new(#name);
                    (__c,)
                }},
            )
        } else {
            (
                quote!(#name: impl ::core::convert::Into<#ty>),
                quote!((::core::convert::Into::into(#name),)),
            )
        };

        quote! {
            impl< #(#others),* > #builder_name<( #(#in_slots,)* )> {
                #(#docs)*
                #[doc(hidden)]
                #[allow(non_snake_case)]
                #vis fn #name(self, #arg) -> #builder_name<( #(#out_slots,)* )> {
                    let ( #(#binders,)* ) = self.fields;
                    let __value = #value;
                    #builder_name { fields: ( #(#rebuilt,)* ), patch: self.patch }
                }
            }
        }
    }

    /// One delegating setter per style/visual/element attribute name — the
    /// same surface `view!` gives every built-in node — accumulated into the
    /// builder's [`ElementPatch`]. A name shadowed by a prop is skipped, so
    /// the prop's own setter is the only one with that name: props win by
    /// construction.
    fn patch_setters(&self, builder_name: &Ident) -> TokenStream {
        let vis = &self.func.vis;
        let widgets = widgets_path();
        let setters = PATCH_ATTRS
            .iter()
            .filter(|(name, _, _)| !self.props.iter().any(|p| p.name == name))
            .map(|(name, ty, chain)| {
                let name = format_ident!("{}", name);
                let ty: TokenStream = ty.parse().expect("patch setter argument type parses");
                let chain: TokenStream = chain.parse().expect("patch setter chain parses");
                quote! {
                    #[doc(hidden)]
                    #vis fn #name(mut self, value: impl ::core::convert::Into<#ty>) -> Self {
                        let value: #ty = ::core::convert::Into::into(value);
                        self.patch = self.patch.#chain;
                        self
                    }
                }
            });
        let semantic_setters = [
            ("role", "IntoSemanticRole", "semantic_role"),
            ("label", "IntoSemanticText", "semantic_label"),
            ("description", "IntoSemanticText", "semantic_description"),
        ]
        .into_iter()
        .filter(|(name, _, _)| !self.props.iter().any(|prop| prop.name == *name))
        .map(|(name, value_trait, chain)| {
            let name = format_ident!("{name}");
            let value_trait = format_ident!("{value_trait}");
            let chain = format_ident!("{chain}");
            quote! {
                #[doc(hidden)]
                #vis fn #name(mut self, value: impl #widgets::#value_trait) -> Self {
                    self.patch = self.patch.#chain(value);
                    self
                }
            }
        });
        let concat_setters = PATCH_CONCAT_ATTRS
            .iter()
            .filter(|(name, _, _)| !self.props.iter().any(|prop| prop.name == *name))
            .map(|(name, ty, chain)| {
                let method = format_ident!("{name}_concat");
                let ty: TokenStream = ty.parse().expect("patch setter argument type parses");
                let chain: TokenStream = chain.parse().expect("patch setter chain parses");
                quote! {
                    #[doc(hidden)]
                    #vis fn #method(mut self, value: impl ::core::convert::Into<#ty>) -> Self {
                        let value: #ty = ::core::convert::Into::into(value);
                        self.patch = self.patch.#chain;
                        self
                    }
                }
            });
        quote! {
            impl<S> #builder_name<S> {
                #(#setters)*
                #(#concat_setters)*
                #(#semantic_setters)*
            }
        }
    }

    /// The `build()` impl: every slot is generic, required props bounded by
    /// `Present` (so an unset one fails to compile) and optional props by `Slot`.
    fn build_impl(
        &self,
        props_name: &Ident,
        builder_name: &Ident,
        present_trait: &Ident,
        slot_trait: &Ident,
    ) -> TokenStream {
        let vis = &self.func.vis;
        let n = self.props.len();
        let slots: Vec<Ident> = (0..n).map(|j| format_ident!("S{}", j)).collect();
        let field_names: Vec<&Ident> = self.props.iter().map(|prop| &prop.name).collect();

        let bounds = self.props.iter().enumerate().map(|(j, prop)| {
            let s = &slots[j];
            let ty = &prop.ty;
            if prop.default.is_some() {
                quote!(#s: #slot_trait<#ty>)
            } else {
                quote!(#s: #present_trait<#ty>)
            }
        });

        let binders: Vec<Ident> = (0..n).map(|j| format_ident!("__f{}", j)).collect();
        // Bind props in authored order. Besides making the generated code easy
        // to inspect, this lets a meaningful default derive from an earlier
        // prop (`#[prop(default = size * 0.14)]`) without inventing a separate
        // dependency language.
        let inits = self.props.iter().enumerate().map(|(j, prop)| {
            let name = &prop.name;
            let ty = &prop.ty;
            let b = &binders[j];
            match &prop.default {
                Some(PropDefault::Type) if prop.children => quote! {
                    let #name: #ty = #slot_trait::get(
                        #b,
                        || ::std::boxed::Box::new(|_: &Element| {}),
                    );
                },
                Some(PropDefault::Type) => quote! {
                    let #name: #ty = #slot_trait::get(
                        #b,
                        || <#ty as ::core::default::Default>::default(),
                    );
                },
                Some(PropDefault::Expr(expr)) if prop.callback.is_some() => quote! {
                    let #name: #ty = #slot_trait::get(
                        #b,
                        || ::std::rc::Rc::new(#expr),
                    );
                },
                Some(PropDefault::Expr(expr)) if prop.children => quote! {
                    let #name: #ty = #slot_trait::get(
                        #b,
                        || ::std::boxed::Box::new(#expr),
                    );
                },
                Some(PropDefault::Expr(expr)) => quote! {
                    let #name: #ty = #slot_trait::get(#b, || #expr);
                },
                None => quote! {
                    let #name: #ty = #present_trait::take(#b);
                },
            }
        });

        quote! {
            impl< #(#bounds),* > #builder_name<( #(#slots,)* )> {
                #[doc(hidden)]
                #vis fn build(self) -> #props_name {
                    let ( #(#binders,)* ) = self.fields;
                    #(#inits)*
                    #props_name { #(#field_names,)* __patch: self.patch }
                }
            }
        }
    }
}

/// A prop type a preview can read out of DSL text.
enum Readable {
    /// Read straight into the declared type.
    Value,
    /// A `State<T>` cell: read the `T` and wrap it. `Derived<T>` is
    /// deliberately not readable — it is computed from something, and a
    /// preview has no way to invent what.
    State(Box<Type>),
}

/// The framework and `std` types the DSL has literal syntax for, matched on
/// the written spelling. An alias for one of them reads as unsupported, which
/// is the safe direction: the preview says so instead of guessing.
const READABLE_TYPES: &[&str] = &[
    "String", "bool", "Color", "Length", "f32", "f64", "i8", "i16", "i32", "i64", "isize", "u8",
    "u16", "u32", "u64", "usize",
];

fn readable_type(ty: &Type) -> Option<Readable> {
    let Type::Path(path) = ty else { return None };
    let segment = path.path.segments.last()?;
    let name = segment.ident.to_string();
    if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
        if name != "State" {
            return None;
        }
        let syn::GenericArgument::Type(inner) = args.args.first()? else {
            return None;
        };
        return readable_type(inner).map(|_| Readable::State(Box::new(inner.clone())));
    }
    READABLE_TYPES
        .contains(&name.as_str())
        .then_some(Readable::Value)
}

/// How many arguments a callback prop's `Fn…` bound takes, provided it returns
/// nothing — a no-op stand-in can be synthesized only then.
fn callback_arity(bound: &TraitBound) -> std::result::Result<usize, String> {
    let segment = bound
        .path
        .segments
        .last()
        .ok_or_else(|| "a callback prop with no `Fn` bound".to_string())?;
    let syn::PathArguments::Parenthesized(args) = &segment.arguments else {
        return Err("a callback prop written without its argument list".to_string());
    };
    match args.output {
        ReturnType::Default => Ok(args.inputs.len()),
        _ => Err(format!(
            "a callback prop returning `{}`, which a preview has no value to supply",
            args.output.to_token_stream()
        )),
    }
}

/// The widget crate as named by the expansion target. Most applications reach
/// it through the `mosaic` umbrella, while the macro UI fixtures depend on the
/// lower-level crate directly; resolving both keeps generated component
/// builders independent of the caller's imports and dependency aliases.
pub(crate) fn widgets_path() -> TokenStream {
    use proc_macro_crate::{FoundCrate, crate_name};

    if let Ok(found) = crate_name("mosaic") {
        return match found {
            FoundCrate::Itself => quote!(crate::widgets),
            FoundCrate::Name(name) => {
                let name = Ident::new(&name, Span::call_site());
                quote!(::#name::widgets)
            }
        };
    }
    match crate_name("mosaic-widgets") {
        Ok(FoundCrate::Itself) => quote!(crate),
        Ok(FoundCrate::Name(name)) => {
            let name = Ident::new(&name, Span::call_site());
            quote!(::#name)
        }
        Err(_) => quote!(::mosaic_widgets),
    }
}

/// The delegating patch setters every props builder grows (minus the names
/// its props shadow): attribute name, the canonical argument type, and the
/// [`ElementPatch`] chain the converted value feeds. One table keeps the
/// component surface identical to the built-in nodes'.
const PATCH_ATTRS: &[(&str, &str, &str)] = &[
    ("gap", "Length", "style(move |__s| __s.gap(value))"),
    ("grow", "f32", "style(move |__s| __s.grow(value))"),
    ("shrink", "f32", "style(move |__s| __s.shrink(value))"),
    ("basis", "Dimension", "style(move |__s| __s.basis(value))"),
    ("width", "Dimension", "style(move |__s| __s.width(value))"),
    ("height", "Dimension", "style(move |__s| __s.height(value))"),
    (
        "min_width",
        "SizeBound",
        "style(move |__s| __s.min_width(value))",
    ),
    (
        "max_width",
        "SizeBound",
        "style(move |__s| __s.max_width(value))",
    ),
    (
        "min_height",
        "SizeBound",
        "style(move |__s| __s.min_height(value))",
    ),
    (
        "max_height",
        "SizeBound",
        "style(move |__s| __s.max_height(value))",
    ),
    ("pad", "Edges", "style(move |__s| __s.padding(value))"),
    ("margin", "Edges", "style(move |__s| __s.margin(value))"),
    ("align", "Align", "style(move |__s| __s.align_items(value))"),
    ("place", "Align", "style(move |__s| __s.align_self(value))"),
    ("justify", "Justify", "style(move |__s| __s.justify(value))"),
    (
        "fill",
        "PaintSpec",
        "visual(move |__v| __v.fill(value.clone()))",
    ),
    ("radius", "Length", "visual(move |__v| __v.radius(value))"),
    (
        "stroke",
        "StrokeSpec",
        "visual(move |__v| __v.reset_strokes().stroke(value.clone()))",
    ),
    ("exponent", "f32", "visual(move |__v| __v.exponent(value))"),
    ("rotate", "f32", "visual(move |__v| __v.rotation(value))"),
    (
        "shadow",
        "ShadowSpec",
        "visual(move |__v| __v.reset_shadows().shadow(value.clone()))",
    ),
    (
        "light",
        "LightSpec",
        "visual(move |__v| __v.reset_lights().light(value.clone()))",
    ),
    ("opacity", "f32", "opacity(value)"),
    ("translate", "Translate", "translate(value)"),
    (
        "translate_children",
        "Translate",
        "translate_children(value)",
    ),
    ("transition", "Transition", "transition(value)"),
    ("enter", "Fx", "enter(value)"),
    ("exit", "Fx", "exit(value)"),
    ("enter_exit", "Fx", "enter_exit(value)"),
];

/// Explicit concatenation setters backing `name:+value` on component roots.
/// Their ordinary counterparts above replace the inherited visual channel.
const PATCH_CONCAT_ATTRS: &[(&str, &str, &str)] = &[
    (
        "fill",
        "PaintSpec",
        "visual(move |__v| __v.paint(value.clone()))",
    ),
    (
        "stroke",
        "StrokeSpec",
        "visual(move |__v| __v.stroke(value.clone()))",
    ),
    (
        "shadow",
        "ShadowSpec",
        "visual(move |__v| __v.shadow(value.clone()))",
    ),
    (
        "light",
        "LightSpec",
        "visual(move |__v| __v.light(value.clone()))",
    ),
];

/// Whether a type is (syntactically) `Element` — a path ending in `Element`.
fn is_element(ty: &Type) -> bool {
    matches!(ty, Type::Path(p) if p.path.segments.last().is_some_and(|s| s.ident == "Element"))
}

/// Whether an exposed state's value type is `bool`, and so can name a
/// `style!` state block.
fn is_bool(ty: &Type) -> bool {
    matches!(ty, Type::Path(path) if path.qself.is_none() && path.path.is_ident("bool"))
}

/// Turns an authored `impl Fn(…) -> … + 'static` prop into the concrete
/// `Rc<dyn Fn(…) -> …>` type stored by the generated props struct. Components
/// are plain functions at the source level, but their generated builders need
/// one named, owned field type; keeping the erasure here makes call sites take
/// ordinary closures without app-local aliases or constructors.
fn callback_storage(ty: &Type) -> Result<(Type, Option<TraitBound>)> {
    let Type::ImplTrait(impl_trait) = ty else {
        return Ok((ty.clone(), None));
    };

    let mut callback = None;
    for bound in &impl_trait.bounds {
        match bound {
            syn::TypeParamBound::Trait(trait_bound)
                if trait_bound
                    .path
                    .segments
                    .last()
                    .is_some_and(|segment| segment.ident == "Fn") =>
            {
                if callback.replace(trait_bound.clone()).is_some() {
                    return Err(Error::new(
                        bound.span(),
                        "a callback prop must contain exactly one `Fn` bound",
                    ));
                }
            }
            syn::TypeParamBound::Lifetime(lifetime) if lifetime.ident == "static" => {}
            _ => {
                return Err(Error::new(
                    bound.span(),
                    "component `impl Trait` props must be callbacks written as \
                     `impl Fn(…) -> … + 'static`",
                ));
            }
        }
    }

    let callback = callback.ok_or_else(|| {
        Error::new(
            ty.span(),
            "component `impl Trait` props must be callbacks written as \
             `impl Fn(…) -> … + 'static`",
        )
    })?;
    let storage = syn::parse2(quote!(::std::rc::Rc<dyn #callback>))?;
    Ok((storage, Some(callback)))
}

/// Reads the `#[prop(optional)]`, `#[prop(default = EXPR)]`, and
/// `#[prop(exposed)]` markers off a
/// parameter, erroring on any other `#[prop(...)]` form so typos surface
/// rather than silently doing nothing.
fn prop_options(attrs: &[syn::Attribute]) -> Result<(Option<PropDefault>, bool)> {
    let mut default = None;
    let mut exposed = false;
    for attr in attrs {
        if !attr.path().is_ident("prop") {
            continue;
        }
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("optional") {
                match default.as_ref() {
                    None => default = Some(PropDefault::Type),
                    Some(PropDefault::Type) => {
                        return Err(meta.error("`optional` may be specified only once"));
                    }
                    Some(PropDefault::Expr(_)) => {
                        return Err(
                            meta.error("`optional` and `default` cannot be combined on one prop")
                        );
                    }
                }
                Ok(())
            } else if meta.path.is_ident("default") {
                match default.as_ref() {
                    None => {}
                    Some(PropDefault::Type) => {
                        return Err(
                            meta.error("`optional` and `default` cannot be combined on one prop")
                        );
                    }
                    Some(PropDefault::Expr(_)) => {
                        return Err(meta.error("`default` may be specified only once"));
                    }
                }
                let expression = meta.value()?.parse()?;
                default = Some(PropDefault::Expr(expression));
                Ok(())
            } else if meta.path.is_ident("exposed") {
                exposed = true;
                Ok(())
            } else {
                Err(meta.error(
                    "unknown `prop` option (expected `optional`, `default = ...`, or `exposed`)",
                ))
            }
        })?;
    }
    Ok((default, exposed))
}

/// The `T` of a literal `State<T>` / `Derived<T>` type, if the type has that
/// shape — what an exposed state's `ReadState<T>` accessor is generic over.
fn reactive_inner(ty: &Type) -> Option<Type> {
    let Type::Path(path) = ty else { return None };
    let segment = path.path.segments.last()?;
    if segment.ident != "State" && segment.ident != "Derived" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(args) = &segment.arguments else {
        return None;
    };
    match args.args.first()? {
        syn::GenericArgument::Type(inner) if args.args.len() == 1 => Some(inner.clone()),
        _ => None,
    }
}

/// Rejects exposed-state names the call-site grammar would misread: reserved
/// view words, and near-misses of the interaction states (a trailing
/// `hovered { … }` block would be reported as a `hover` typo).
fn check_exposed_state_name(name: &Ident) -> Result<()> {
    let word = name.to_string();
    if mosaic_syntax::reserved_exposed_name(&word) {
        return Err(Error::new(
            name.span(),
            format!("`{word}` cannot name an exposed state — it collides with the view grammar"),
        ));
    }
    if let Some(meant) =
        mosaic_syntax::did_you_mean(&word, &["hover", "pressed", "focused", "disabled"])
    {
        return Err(Error::new(
            name.span(),
            format!(
                "exposed state `{word}` reads as a typo of the `{meant}` interaction state \
                 at call sites — pick a more distinct name"
            ),
        ));
    }
    Ok(())
}

/// Collects the `#[exposed] let name: State<T> = …;` statements from the
/// function body's top level, stripping the marker attribute. An `#[exposed]`
/// anywhere deeper is an error — the tail rewrite can only smuggle out
/// top-level bindings.
fn collect_inner_states(func: &mut ItemFn) -> Result<Vec<InnerState>> {
    let mut states = Vec::new();
    for stmt in &mut func.block.stmts {
        let Stmt::Local(local) = stmt else { continue };
        let Some(position) = local
            .attrs
            .iter()
            .position(|attr| attr.path().is_ident("exposed"))
        else {
            continue;
        };
        let attr = local.attrs.remove(position);
        if !matches!(attr.meta, syn::Meta::Path(_)) {
            return Err(Error::new(attr.span(), "`#[exposed]` takes no arguments"));
        }
        let Pat::Type(pat_ty) = &local.pat else {
            return Err(Error::new(
                local.pat.span(),
                "an `#[exposed]` let needs its type written out: \
                 `let name: State<T> = …` or `Derived<T>`",
            ));
        };
        let Pat::Ident(pat_ident) = &*pat_ty.pat else {
            return Err(Error::new(
                pat_ty.pat.span(),
                "an `#[exposed]` let must bind a plain name",
            ));
        };
        let Some(inner_ty) = reactive_inner(&pat_ty.ty) else {
            return Err(Error::new(
                pat_ty.ty.span(),
                "an exposed state must be a `State<T>` or `Derived<T>` \
                 with its type written out",
            ));
        };
        let name = pat_ident.ident.clone();
        check_exposed_state_name(&name)?;
        states.push(InnerState { name, inner_ty });
    }
    // Any `#[exposed]` left after the top-level sweep sits in a nested block.
    let mut nested = NestedExposed(None);
    nested.visit_block(&func.block);
    if let Some(span) = nested.0 {
        return Err(Error::new(
            span,
            "`#[exposed]` only works on a top-level `let` of the component body — \
             the handle is built from the body's tail, where nested bindings are \
             out of scope",
        ));
    }
    Ok(states)
}

/// Finds an `#[exposed]` attribute on a `let` in nested blocks (the top-level
/// ones were already stripped).
struct NestedExposed(Option<Span>);

impl<'ast> Visit<'ast> for NestedExposed {
    fn visit_local(&mut self, local: &'ast syn::Local) {
        if self.0.is_none()
            && let Some(attr) = local
                .attrs
                .iter()
                .find(|attr| attr.path().is_ident("exposed"))
        {
            self.0 = Some(attr.span());
        }
        syn::visit::visit_local(self, local);
    }
}

/// Collects the `as pub` part names from the body's `view!` invocations by
/// parsing their tokens with the same grammar the real expansion uses — the
/// names cannot disagree with what registers at runtime. Parts in more than
/// one `view!` are rejected: the handle resolves parts on the single root the
/// body returns.
fn collect_view_parts(func: &ItemFn) -> Result<Vec<(String, Span)>> {
    struct Views(Vec<(Vec<(String, Span)>, Span)>);
    impl<'ast> Visit<'ast> for Views {
        fn visit_macro(&mut self, mac: &'ast syn::Macro) {
            if mac
                .path
                .segments
                .last()
                .is_some_and(|segment| segment.ident == "view")
                && let Ok(view) = parse2::<mosaic_syntax::View>(mac.tokens.clone())
            {
                let parts = view
                    .exposed_parts()
                    .into_iter()
                    .map(|binding| (binding.name.to_string(), binding.name.span()))
                    .collect::<Vec<_>>();
                if !parts.is_empty() {
                    self.0.push((parts, mac.path.span()));
                }
            }
            syn::visit::visit_macro(self, mac);
        }
    }
    let mut views = Views(Vec::new());
    views.visit_block(&func.block);
    let mut exposing = views.0.into_iter();
    let Some((parts, _)) = exposing.next() else {
        return Ok(Vec::new());
    };
    if let Some((_, span)) = exposing.next() {
        return Err(Error::new(
            span,
            "`as pub` parts in more than one `view!` of one component body — the \
             handle resolves parts on the single root the body returns",
        ));
    }
    Ok(parts)
}
