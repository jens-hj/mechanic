//! Steering a declaration toward the canonical spelling without rejecting it.
//!
//! Kebab-case is how the DSL spells every name — attributes, record fields,
//! and design tokens alike — but a `scheme!` may legitimately declare a token
//! whose name is not kebab (a snake_case one mirroring an existing Rust
//! constant, say), and rejecting it outright would make those schemes
//! unwritable. So the odd spelling compiles and the author is nudged instead.
//!
//! A proc macro has no way to raise a warning on stable, so the nudge is
//! carried by the one lint that fires on demand: the expansion defines a
//! deprecated const named after the token and immediately uses it, which
//! prints `use of deprecated constant …` with our note attached, spanned at
//! the name the author wrote.

use proc_macro2::TokenStream;
use quote::{quote, quote_spanned};
use syn::Ident;

/// The nudge for a token declared in snake_case, or nothing when the name is
/// already canonical. `name` is the *normalized* identifier, so its
/// underscores are the ones the author typed.
pub(crate) fn kebab_case_token(name: &Ident, legacy: bool) -> TokenStream {
    if !legacy {
        return TokenStream::new();
    }
    let written = name.to_string();
    let canonical = written.replace('_', "-");
    let note = format!(
        "token `{written}` is not kebab-case — the canonical spelling is \
         `{canonical}`, and `view!` reaches this token by whichever one you declare"
    );
    // Span the *use* at the name so the warning underlines the declaration
    // rather than the whole macro invocation.
    let used = quote_spanned!(name.span()=> #name);
    quote! {
        const _: () = {
            #[deprecated(note = #note)]
            #[allow(non_upper_case_globals, non_camel_case_types)]
            const #name: () = ();
            let _ = #used;
        };
    }
}
