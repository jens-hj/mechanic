//! `#[preview]` — marks a function as a live preview.
//!
//! ```ignore
//! #[preview]
//! fn card_in_a_frame() -> Element {
//!     view! {
//!         stack height:100px width:200px {
//!             img "assets/img.png"
//!             Card
//!         }
//!     }
//! }
//! ```
//!
//! The function is left exactly as written and registered alongside itself, so
//! a preview host can find it, name it, and build it. `#[preview(width = 390,
//! height = 844)]` pins the frame the card renders in; unset, the card takes
//! its content's size.
//!
//! Registration is gated on `debug_assertions`: a release build carries no
//! previews at all.

use proc_macro2::TokenStream;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::spanned::Spanned;
use syn::{Error, Ident, ItemFn, LitFloat, LitInt, Result, ReturnType, Token, Type, parse2};

use crate::component::widgets_path;

/// The pinned frame `#[preview(width = …, height = …)]` asks for.
#[derive(Default)]
struct Frame {
    width: Option<f32>,
    height: Option<f32>,
}

impl Parse for Frame {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut frame = Frame::default();
        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![=]>()?;
            let value = parse_number(input)?;
            match key.to_string().as_str() {
                "width" => frame.width = Some(value),
                "height" => frame.height = Some(value),
                other => {
                    return Err(Error::new(
                        key.span(),
                        format!("unknown preview option `{other}`; expected `width` or `height`"),
                    ));
                }
            }
            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }
        Ok(frame)
    }
}

fn parse_number(input: ParseStream) -> Result<f32> {
    if input.peek(LitInt) {
        return input.parse::<LitInt>()?.base10_parse::<f32>();
    }
    input.parse::<LitFloat>()?.base10_parse::<f32>()
}

pub fn expand(attr: TokenStream, item: TokenStream) -> Result<TokenStream> {
    let func: ItemFn = parse2(item)?;
    let frame: Frame = parse2(attr)?;

    if !func.sig.inputs.is_empty() {
        return Err(Error::new(
            func.sig.inputs.span(),
            "a `#[preview]` function takes no arguments — it *is* the call site",
        ));
    }
    if !func.sig.generics.params.is_empty() {
        return Err(Error::new(
            func.sig.generics.span(),
            "a `#[preview]` function cannot be generic",
        ));
    }
    match &func.sig.output {
        ReturnType::Type(_, ty) if is_element(ty) => {}
        other => {
            return Err(Error::new(
                other.span(),
                "a `#[preview]` function must return `Element`",
            ));
        }
    }

    let name = &func.sig.ident;
    let widgets = widgets_path();
    let size = match (frame.width, frame.height) {
        // A half-pinned frame is a typo, not a shorthand: the other axis would
        // silently take the content's size and the preview would not be the
        // box it says it is.
        (Some(_), None) | (None, Some(_)) => {
            return Err(Error::new(
                func.sig.ident.span(),
                "a pinned preview frame needs both `width` and `height`",
            ));
        }
        (Some(width), Some(height)) => quote!(::core::option::Option::Some((#width, #height))),
        (None, None) => quote!(::core::option::Option::None),
    };

    Ok(quote! {
        // The registration below is what calls a preview, and it is compiled
        // out of a release build — leaving the function looking dead to a
        // reader who never wrote a dead function. The allow is scoped to
        // exactly that configuration, so an ordinary debug build still reports
        // anything genuinely unreachable.
        #[cfg_attr(not(debug_assertions), allow(dead_code))]
        #func

        #[cfg(debug_assertions)]
        #widgets::preview::inventory::submit! {
            #widgets::preview::Preview {
                name: stringify!(#name),
                file: file!(),
                line: line!(),
                size: #size,
                build: #name,
            }
        }
    })
}

fn is_element(ty: &Type) -> bool {
    let Type::Path(path) = ty else { return false };
    path.path
        .segments
        .last()
        .is_some_and(|segment| segment.ident == "Element")
}
