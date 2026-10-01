//! Desktop clipboard adapter for Mosaic's platform clipboard seam.

use bevy::log::warn;
use mosaic_widgets::{Ui, input::Clipboard};

struct SystemClipboard(arboard::Clipboard);

impl Clipboard for SystemClipboard {
    fn get(&mut self) -> Option<String> {
        self.0.get_text().ok()
    }

    fn set(&mut self, text: String) {
        if let Err(error) = self.0.set_text(text) {
            warn!("cannot copy text to the system clipboard: {error}");
        }
    }
}

/// Keep the clipboard alive with the tree, including on Linux where the
/// clipboard owner must continue serving pasted text to other applications.
pub(crate) fn install(ui: &Ui) {
    match arboard::Clipboard::new() {
        Ok(clipboard) => ui.set_clipboard(SystemClipboard(clipboard)),
        Err(error) => warn!("system clipboard unavailable: {error}"),
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, rc::Rc};

    use bevy::prelude::Entity;
    use mosaic_core::{Size, State};
    use mosaic_text::FontContext;
    use mosaic_widgets::{
        input::{Clipboard, Key, KeyEvent, KeyEventKind, Modifiers},
        text_input,
    };

    use crate::context::MosaicContext;

    struct ExternalClipboard(Rc<RefCell<String>>);

    impl Clipboard for ExternalClipboard {
        fn get(&mut self) -> Option<String> {
            Some(self.0.borrow().clone())
        }

        fn set(&mut self, text: String) {
            *self.0.borrow_mut() = text;
        }
    }

    #[test]
    fn input_shortcuts_exchange_text_with_the_host_clipboard() {
        let context = MosaicContext::with_fonts(Entity::PLACEHOLDER, FontContext::embedded_only());
        let ui = context.ui();
        let clipboard = Rc::new(RefCell::new("external seed".to_owned()));
        ui.set_clipboard(ExternalClipboard(clipboard.clone()));
        let value = State::new("old".to_owned());
        let field = text_input(&ui.root(), value);
        ui.frame(Size::new(400.0, 100.0), 1.0);
        field.focus();
        let shortcut = |key: &str| {
            ui.dispatch_key(KeyEvent {
                kind: KeyEventKind::Down { repeat: false },
                key: Key::Character(key.to_owned()),
                modifiers: Modifiers {
                    meta: cfg!(target_os = "macos"),
                    ctrl: !cfg!(target_os = "macos"),
                    ..Modifiers::default()
                },
            });
        };
        shortcut("a");
        shortcut("v");
        assert_eq!(value.get(), "external seed");
        shortcut("a");
        shortcut("x");
        assert_eq!(value.get(), "");
        assert_eq!(*clipboard.borrow(), "external seed");
        shortcut("v");
        shortcut("a");
        *clipboard.borrow_mut() = String::new();
        shortcut("c");
        assert_eq!(*clipboard.borrow(), "external seed");
    }
}
