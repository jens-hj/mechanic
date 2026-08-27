//! Non-interactive world reticle aligned with the ray used by every tool.

#![allow(clippy::wildcard_imports)]

use bevy_mosaic::ui::*;
use mosaic_macros::{component, view};

#[allow(clippy::wildcard_imports)]
use super::theme::*;

#[component]
pub(crate) fn WorldReticle() -> Element {
    view! {
        stack align:center justify:center nohit {
            canvas width:28px height:28px nohit {
                circle at:(x:14px y:14px) radius:4px stroke:(width:2px color:ink.fg)
                line from:(x:14px y:1px) to:(x:14px y:8px)
                    stroke:(width:2px cap:square color:ink.fg)
                line from:(x:14px y:20px) to:(x:14px y:27px)
                    stroke:(width:2px cap:square color:ink.fg)
                line from:(x:1px y:14px) to:(x:8px y:14px)
                    stroke:(width:2px cap:square color:ink.fg)
                line from:(x:20px y:14px) to:(x:27px y:14px)
                    stroke:(width:2px cap:square color:ink.fg)
            }
        }
    }
}
