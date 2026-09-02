//! Shared Mosaic styles for the product overlay.

#[allow(clippy::wildcard_imports)] // Mosaic's style vocabulary is meant to be globbed.
use bevy_mosaic::ui::*;

#[allow(clippy::wildcard_imports)] // Design tokens are intentionally bare in styles.
use super::theme::*;

mosaic_macros::style! {
    pub(crate) #mechanic.overlay
        font-family:typeface.body font-size:text-size.body font-color:ink.fg
        transition:motion

    pub(crate) #mechanic.panel
        fill:shell radius:radius.panel exponent:1
        stroke:(width:border.hairline color:shell-edge)

    pub(crate) #mechanic.elevated
        radius:radius.elevated exponent:1 shadow:panel-shadow

    pub(crate) #mechanic.badge
        height:26px align:center justify:center radius:radius.badge exponent:1
        fill:port.fill stroke:(width:border.hairline color:accent.key)
        font-size:text-size.value
        font-weight:{ resolved_weight(text_weight.bold) } font-color:ink.fg

    pub(crate) #mechanic.action
        align:center justify:center radius:radius.action exponent:1 fill:control.rest
        stroke:(width:border.hairline color:chip.edge)
        hover { fill:control.hover stroke:(width:border.hairline color:chip.edge-over) }
        pressed { fill:control.pressed stroke:(width:border.hairline color:accent.key) }

    pub(crate) #mechanic.action-danger
        fill:picker.danger font-color:accent.danger
        hover { fill:picker.danger-over stroke:(width:border.hairline color:accent.danger) }
        pressed { fill:wash.delete }

    pub(crate) #mechanic.field
        height:control-size.compact-height radius:radius.field exponent:1 fill:chip.fill
        stroke:(width:border.hairline color:chip.edge)
        font-size:text-size.value font-color:ink.fg
        hover { stroke:(width:border.hairline color:chip.edge-over) }
        focused { stroke:(width:border.strong color:control.focus offset:-1px) }

    pub(crate) #mechanic.chip
        radius:radius.chip exponent:1 fill:chip.fill
        stroke:(width:border.hairline color:chip.edge)

    pub(crate) #mechanic.list-row
        radius:radius.action exponent:1 fill:picker.row
        stroke:(width:border.hairline color:picker.row-edge)
        hover { fill:picker.row-over }
        pressed { fill:control.pressed }

    pub(crate) #mechanic.title
        font-family:typeface.display font-size:text-size.title
        font-weight:{ resolved_weight(text_weight.bold) }
        letter-spacing:text-tracking.title

    pub(crate) #mechanic.section
        font-family:typeface.display font-size:text-size.caption
        font-weight:{ resolved_weight(text_weight.bold) }
        letter-spacing:text-tracking.section font-color:ink.dim

    pub(crate) #mechanic.label
        font-size:text-size.label font-weight:{ resolved_weight(text_weight.bold) }
        letter-spacing:text-tracking.label font-color:ink.dim

    pub(crate) #mechanic.caption
        font-size:text-size.caption font-color:ink.muted

    pub(crate) #mechanic.value
        font-size:text-size.value
        font-weight:{ resolved_weight(text_weight.bold) } font-color:ink.fg

    pub(crate) #mechanic.pause-veil
        fill:picker.veil

    pub(crate) #mechanic.pause-sheet
        fill:picker.sheet radius:radius.elevated exponent:1 shadow:modal-shadow
        stroke:(width:border.hairline color:picker.edge)

    pub(crate) #mechanic.pause-confirm
        fill:picker.danger radius:radius.panel exponent:1
        stroke:(width:border.hairline color:accent.danger)

    pub(crate) #mechanic.pause-slider {
        track height:6px radius:3px exponent:1 fill:dial.track
        fill radius:3px exponent:1 fill:accent.speed
        thumb width:18px height:18px radius:9px exponent:1 fill:ink.fg
            stroke:(width:border.hairline color:accent.speed offset:-0.5px)
            hover { fill:accent.speed }
            focused { stroke:+(width:border.strong color:control.focus offset:2px) }
    }
}
