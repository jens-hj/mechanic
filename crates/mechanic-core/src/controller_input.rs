//! Transient, controller-local keyboard sources. Never serialized into creations.

use std::collections::{BTreeMap, BTreeSet};

use crate::{ButtonMode, ConstructionGraph, DriveKey, PartId, PartSpec};

/// A controller key after combining every keyboard and physical source.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ControllerKeys {
    held: BTreeSet<(PartId, DriveKey)>,
    pressed: BTreeSet<(PartId, DriveKey)>,
    released: BTreeSet<(PartId, DriveKey)>,
    buttons: BTreeMap<PartId, (PartId, DriveKey, ButtonMode)>,
}

impl ControllerKeys {
    /// Presses a button's own source; a toggle changes only its own latch.
    pub fn press_button(&mut self, graph: &ConstructionGraph, button: PartId) {
        let Some(source) = button_source(graph, button) else {
            return;
        };
        if source.2 == ButtonMode::Toggle && self.buttons.get(&button) == Some(&source) {
            self.buttons.remove(&button);
        } else {
            self.buttons.insert(button, source);
        }
    }

    /// Releases or cancels a momentary interaction, leaving toggle latches alone.
    pub fn release_button(&mut self, button: PartId) {
        if self
            .buttons
            .get(&button)
            .is_some_and(|source| source.2 == ButtonMode::Momentary)
        {
            self.buttons.remove(&button);
        }
    }

    /// Combines current seated keys and valid button sources, emitting logical edges.
    /// Changed keys, modes, deleted parts and disconnected controllers lose their sources.
    pub fn update(
        &mut self,
        graph: &ConstructionGraph,
        keyboard: impl IntoIterator<Item = (PartId, DriveKey)>,
    ) {
        self.buttons
            .retain(|&button, source| button_source(graph, button) == Some(*source));
        let mut held: BTreeSet<_> = keyboard
            .into_iter()
            .filter(|(controller, _)| graph.is_controller(*controller))
            .collect();
        held.extend(
            self.buttons
                .values()
                .map(|&(controller, key, _)| (controller, key)),
        );
        self.pressed = held.difference(&self.held).copied().collect();
        self.released = self.held.difference(&held).copied().collect();
        self.held = held;
    }

    /// Whether any source holds this key on this controller.
    pub fn held(&self, controller: PartId, key: DriveKey) -> bool {
        self.held.contains(&(controller, key))
    }

    /// Whether a physical button contributes this key, independent of keyboard modifiers.
    pub fn button_held(&self, controller: PartId, key: DriveKey) -> bool {
        self.buttons
            .values()
            .any(|&(owner, button_key, _)| owner == controller && button_key == key)
    }

    /// Whether the combined key changed from released to held this update.
    pub fn pressed(&self, controller: PartId, key: DriveKey) -> bool {
        self.pressed.contains(&(controller, key))
    }

    /// Whether the final held source released this update.
    pub fn released(&self, controller: PartId, key: DriveKey) -> bool {
        self.released.contains(&(controller, key))
    }

    /// Every currently held controller/key pair in stable order.
    pub fn held_keys(&self) -> impl Iterator<Item = (PartId, DriveKey)> + '_ {
        self.held.iter().copied()
    }

    /// Clears latches and keyboard history on simulation reset.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

fn button_source(
    graph: &ConstructionGraph,
    button: PartId,
) -> Option<(PartId, DriveKey, ButtonMode)> {
    if !matches!(graph.part(button), Some(PartSpec::Button(_))) {
        return None;
    }
    let config = graph.input_configuration(button)?;
    Some((config.controller?, config.key?, config.button_mode))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BuildCommand, BuildOutcome, BuildPose, ButtonSpec, ControllerSpec, InputConfiguration,
        InputSize,
    };

    fn spawn(graph: &mut ConstructionGraph, command: BuildCommand) -> PartId {
        let BuildOutcome::Spawned(id) = graph.apply(command).unwrap() else {
            panic!("spawn")
        };
        id
    }

    #[test]
    fn several_sources_emit_one_edge_and_isolate_controllers() {
        let mut graph = ConstructionGraph::new();
        let controller = spawn(
            &mut graph,
            BuildCommand::SpawnController(ControllerSpec::new(BuildPose::default())),
        );
        let other = spawn(
            &mut graph,
            BuildCommand::SpawnController(ControllerSpec::new(BuildPose::default())),
        );
        let key = DriveKey::new('w').unwrap();
        let buttons: Vec<_> = (0..2)
            .map(|_| {
                let button = spawn(
                    &mut graph,
                    BuildCommand::SpawnButton(ButtonSpec::new(
                        InputSize::Panel,
                        BuildPose::default(),
                    )),
                );
                graph
                    .apply(BuildCommand::SetInputConfiguration {
                        input: button,
                        configuration: InputConfiguration {
                            controller: Some(controller),
                            key: Some(key),
                            ..Default::default()
                        },
                    })
                    .unwrap();
                button
            })
            .collect();
        let mut keys = ControllerKeys::default();
        keys.press_button(&graph, buttons[0]);
        keys.update(&graph, []);
        assert!(keys.pressed(controller, key));
        assert!(!keys.held(other, key));
        keys.press_button(&graph, buttons[1]);
        keys.update(&graph, [(controller, key)]);
        assert!(!keys.pressed(controller, key));
        for button in &buttons {
            keys.release_button(*button);
        }
        keys.update(&graph, [(controller, key)]);
        assert!(keys.held(controller, key));
        assert!(!keys.released(controller, key));
        keys.update(&graph, []);
        assert!(keys.released(controller, key));
    }
    #[test]
    fn toggle_latches_are_local_and_clear_on_configuration_changes_and_reset() {
        let mut graph = ConstructionGraph::new();
        let controller = spawn(
            &mut graph,
            BuildCommand::SpawnController(ControllerSpec::new(BuildPose::default())),
        );
        let button = spawn(
            &mut graph,
            BuildCommand::SpawnButton(ButtonSpec::new(InputSize::Panel, BuildPose::default())),
        );
        let key = DriveKey::new('7').unwrap();
        let mut config = InputConfiguration {
            controller: Some(controller),
            key: Some(key),
            button_mode: ButtonMode::Toggle,
            ..Default::default()
        };
        graph
            .apply(BuildCommand::SetInputConfiguration {
                input: button,
                configuration: config.clone(),
            })
            .unwrap();
        let mut keys = ControllerKeys::default();
        keys.press_button(&graph, button);
        keys.release_button(button);
        keys.update(&graph, []);
        assert!(keys.held(controller, key));
        keys.press_button(&graph, button);
        keys.update(&graph, []);
        assert!(keys.released(controller, key));
        keys.press_button(&graph, button);
        keys.update(&graph, []);
        config.key = DriveKey::new('X');
        graph
            .apply(BuildCommand::SetInputConfiguration {
                input: button,
                configuration: config.clone(),
            })
            .unwrap();
        keys.update(&graph, []);
        assert!(!keys.held(controller, key));
        assert!(!keys.held(controller, config.key.unwrap()));
        keys.press_button(&graph, button);
        keys.update(&graph, []);
        keys.reset();
        keys.update(&graph, []);
        assert_eq!(keys.held_keys().count(), 0);
        keys.press_button(&graph, button);
        graph.apply(BuildCommand::Remove(button)).unwrap();
        keys.update(&graph, []);
        assert_eq!(keys.held_keys().count(), 0);
    }
}
