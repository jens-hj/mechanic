use bevy::{prelude::*, ui::FocusPolicy};

use crate::showcase::CreationPreset;

const BUTTON_BACKGROUND: Color = Color::srgba(0.06, 0.085, 0.12, 0.98);
const BUTTON_HOVER_BACKGROUND: Color = Color::srgba(0.10, 0.16, 0.21, 0.98);
const BUTTON_PRESSED_BACKGROUND: Color = Color::srgba(0.08, 0.28, 0.35, 0.98);

#[derive(Resource, Debug, Default)]
pub(crate) struct CreationMenuState {
    open: bool,
    blocks_pointer: bool,
    requested: Option<CreationPreset>,
}

impl CreationMenuState {
    pub(crate) fn begin_frame(&mut self) {
        self.blocks_pointer = false;
    }

    pub(crate) fn open(&mut self) {
        self.open = true;
        self.blocks_pointer = true;
    }

    pub(crate) fn close(&mut self) {
        self.open = false;
        self.blocks_pointer = true;
    }

    pub(crate) const fn is_open(&self) -> bool {
        self.open
    }

    pub(crate) const fn blocks_pointer(&self) -> bool {
        self.blocks_pointer || self.open
    }

    pub(crate) fn take_request(&mut self) -> Option<CreationPreset> {
        self.requested.take()
    }
}

#[derive(Clone, Copy, Component)]
pub(crate) enum CreationMenuAction {
    Load(CreationPreset),
    Cancel,
}

#[derive(Component)]
pub(crate) struct CreationMenuRoot;

pub(crate) fn spawn(commands: &mut Commands) {
    commands
        .spawn((
            Name::new("Creation picker backdrop"),
            CreationMenuRoot,
            Node {
                position_type: PositionType::Absolute,
                left: px(0),
                top: px(0),
                width: percent(100),
                height: percent(100),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(Color::srgba(0.005, 0.008, 0.012, 0.76)),
            GlobalZIndex(100),
            Visibility::Hidden,
            FocusPolicy::Block,
        ))
        .with_children(|backdrop| {
            backdrop
                .spawn((
                    Name::new("Creation picker"),
                    Node {
                        width: px(600),
                        padding: UiRect::all(px(24)),
                        flex_direction: FlexDirection::Column,
                        row_gap: px(12),
                        border: UiRect::all(px(2)),
                        border_radius: BorderRadius::all(px(10)),
                        ..default()
                    },
                    BackgroundColor(Color::srgba(0.018, 0.026, 0.038, 0.99)),
                    BorderColor::all(Color::srgba(0.36, 0.48, 0.58, 0.95)),
                ))
                .with_children(|dialog| {
                    dialog.spawn((
                        Text::new("OPEN CREATION"),
                        TextFont {
                            font_size: FontSize::Px(25.0),
                            ..default()
                        },
                        TextColor(Color::WHITE),
                    ));
                    dialog.spawn((
                        Text::new("Choose a deterministic scene. This replaces the current construction and can be undone."),
                        TextFont {
                            font_size: FontSize::Px(15.0),
                            ..default()
                        },
                        TextColor(Color::srgb(0.72, 0.78, 0.84)),
                        Node {
                            margin: UiRect::bottom(px(5)),
                            ..default()
                        },
                    ));
                    for preset in CreationPreset::ALL {
                        spawn_preset_button(dialog, preset);
                    }
                    spawn_cancel_button(dialog);
                });
        });
}

fn spawn_preset_button(parent: &mut ChildSpawnerCommands<'_>, preset: CreationPreset) {
    parent
        .spawn((
            Button,
            CreationMenuAction::Load(preset),
            Node {
                width: percent(100),
                min_height: px(70),
                padding: UiRect::axes(px(16), px(11)),
                flex_direction: FlexDirection::Column,
                justify_content: JustifyContent::Center,
                align_items: AlignItems::FlexStart,
                row_gap: px(4),
                border_radius: BorderRadius::all(px(7)),
                ..default()
            },
            BackgroundColor(BUTTON_BACKGROUND),
        ))
        .with_children(|button| {
            button.spawn((
                Text::new(preset.label()),
                TextFont {
                    font_size: FontSize::Px(19.0),
                    ..default()
                },
                TextColor(Color::WHITE),
            ));
            button.spawn((
                Text::new(preset.description()),
                TextFont {
                    font_size: FontSize::Px(14.0),
                    ..default()
                },
                TextColor(Color::srgb(0.68, 0.75, 0.81)),
            ));
        });
}

fn spawn_cancel_button(parent: &mut ChildSpawnerCommands<'_>) {
    parent.spawn((
        Button,
        CreationMenuAction::Cancel,
        Node {
            width: percent(100),
            height: px(42),
            margin: UiRect::top(px(3)),
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
            border_radius: BorderRadius::all(px(7)),
            ..default()
        },
        BackgroundColor(BUTTON_BACKGROUND),
        children![(
            Text::new("Cancel"),
            TextFont {
                font_size: FontSize::Px(16.0),
                ..default()
            },
            TextColor(Color::WHITE),
        )],
    ));
}

#[allow(clippy::type_complexity)]
pub(crate) fn update(
    mut state: ResMut<CreationMenuState>,
    mut root: Single<&mut Visibility, With<CreationMenuRoot>>,
    mut buttons: Query<
        (&Interaction, &CreationMenuAction, &mut BackgroundColor),
        (With<Button>, Changed<Interaction>),
    >,
) {
    if state.open {
        state.blocks_pointer = true;
    }
    for (interaction, action, mut background) in &mut buttons {
        background.0 = match interaction {
            Interaction::Pressed => BUTTON_PRESSED_BACKGROUND,
            Interaction::Hovered => BUTTON_HOVER_BACKGROUND,
            Interaction::None => BUTTON_BACKGROUND,
        };
        if *interaction != Interaction::Pressed || !state.open {
            continue;
        }
        match action {
            CreationMenuAction::Load(preset) => state.requested = Some(*preset),
            CreationMenuAction::Cancel => {}
        }
        state.close();
    }
    **root = if state.open {
        Visibility::Visible
    } else {
        Visibility::Hidden
    };
}

#[cfg(test)]
mod tests {
    use bevy::prelude::*;

    use super::{CreationMenuAction, CreationMenuRoot, CreationMenuState, CreationPreset, update};

    #[test]
    fn pressed_preset_requests_load_and_closes_modal() {
        let mut app = App::new();
        app.init_resource::<CreationMenuState>()
            .add_systems(Update, update);
        app.world_mut().resource_mut::<CreationMenuState>().open();
        app.world_mut()
            .spawn((CreationMenuRoot, Visibility::Hidden));
        app.world_mut().spawn((
            Button,
            CreationMenuAction::Load(CreationPreset::MobileWorkshop1024),
            Interaction::Pressed,
            BackgroundColor::default(),
        ));

        app.update();

        let mut state = app.world_mut().resource_mut::<CreationMenuState>();
        assert!(!state.is_open());
        assert!(state.blocks_pointer());
        assert_eq!(
            state.take_request(),
            Some(CreationPreset::MobileWorkshop1024)
        );
    }
}
