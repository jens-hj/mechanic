//! Saving, loading, and adopting creations into the editor.

use crate::camera::{MainCamera, PlayerCamera, PlayerState};
use crate::control_panel::ControlPanelState;
use crate::controls::GameAction;
use crate::creation_menu::{CreationMenuState, CreationRequest};
use crate::creation_store::CreationStore;
use crate::editor::build_actions::PlacedBearing;
use crate::editor::history::{EditorHistory, EditorSnapshot, cancel_transient_editor_state};
use crate::editor::hover::clear_hover;
use crate::editor::state::{CurrentCreation, EditorGraph, EditorState};
use crate::pause_menu::PauseMenuState;
use crate::{builder, camera, creation_store, showcase, world};
use bevy::prelude::{
    ButtonInput, GlobalTransform, Quat, Res, ResMut, Single, State, Transform, Vec3, With, format,
};
use mechanic_core::{
    BearingSocket, CompiledCreation, ConstructionGraph, CreationDocument, TopologyError,
};
use std::error::Error;
use std::path::Path;

/// Whether the primary modifier plus `S` was pressed this frame.
///
/// A modifier is required because a bare letter binds to a drive state, so
/// plain `S` belongs to a machine rather than to the editor.
pub(crate) fn save_shortcut_requested(actions: &ButtonInput<GameAction>) -> bool {
    actions.just_pressed(GameAction::Save)
}

/// Opens the creations modal with `P`, or with the primary modifier and `S`.
///
/// While it is open the modal owns the keyboard, so neither key reaches here:
/// `p` and `s` type into its name field, and Escape is its own to handle. The
/// control-block panel owns the keyboard the same way, and the two must never
/// both be typing, so neither can open over the other.
#[expect(
    clippy::too_many_arguments,
    reason = "bevy system resources are explicit parameters"
)]
pub(crate) fn handle_creation_menu_shortcut(
    actions: Res<ButtonInput<GameAction>>,
    mut graph: ResMut<EditorGraph>,
    mut state: ResMut<EditorState>,
    space: Res<State<world::AppSpace>>,
    store: Res<CreationStore>,
    current: Res<CurrentCreation>,
    panel: Res<ControlPanelState>,
    pause: Res<PauseMenuState>,
    mut menu: ResMut<CreationMenuState>,
    worlds: Res<world::WorldListState>,
) {
    if worlds.is_open() || menu.is_open() || panel.blocks_keyboard() || pause.blocks_world_input() {
        return;
    }
    let saving = save_shortcut_requested(&actions);
    if !saving && !actions.just_pressed(GameAction::Creations) {
        return;
    }
    if *space.get() == world::AppSpace::World {
        state.feedback =
            Some("Saved creations are managed in the Garage — press F6 first".to_owned());
        return;
    }
    cancel_transient_editor_state(&mut graph.0, &mut state);
    menu.open(
        store.list(),
        current.0.clone().unwrap_or_default(),
        store.directory().to_path_buf(),
    );
    state.feedback = Some(if saving {
        "Type a name, then Enter to save".to_owned()
    } else {
        "Open a creation, or type a name to save this one".to_owned()
    });
}

/// Applies whatever the creations modal decided this frame.
#[expect(
    clippy::too_many_arguments,
    reason = "bevy system resources are explicit parameters"
)]
pub(crate) fn handle_creation_request(
    mut menu: ResMut<CreationMenuState>,
    mut graph: ResMut<EditorGraph>,
    mut state: ResMut<EditorState>,
    mut history: ResMut<EditorHistory>,
    mut current: ResMut<CurrentCreation>,
    store: Res<CreationStore>,
    mut player: ResMut<PlayerState>,
    mut camera: Single<(&mut PlayerCamera, &mut Transform, &mut GlobalTransform), With<MainCamera>>,
    mut world_runtime: ResMut<world::WorldRuntime>,
) {
    let Some(request) = menu.take_request() else {
        return;
    };
    match request {
        CreationRequest::LoadPreset(preset) => {
            let previous = EditorSnapshot::capture(&graph.0, &state);
            match showcase::build_preset(preset).and_then(|candidate| {
                install_editor_graph(&mut graph.0, candidate).map_err(showcase::ShowcaseError::from)
            }) {
                Ok(creation) => {
                    debug_assert_eq!(creation.compounds.len(), preset.body_count());
                    history.commit(previous);
                    current.0 = None;
                    adopt_loaded_creation(
                        &mut state,
                        &mut player,
                        &mut camera,
                        &graph.0,
                        Vec::new(),
                    );
                    state.feedback = Some(format!(
                        "Opened {}: {} welds, {} bearings, {} bodies — F6 enters the live World",
                        preset.label(),
                        graph.0.weld_count(),
                        graph.0.bearing_count(),
                        creation.compounds.len(),
                    ));
                }
                Err(error) => {
                    state.feedback = Some(format!("Could not open creation: {error}"));
                }
            }
        }
        CreationRequest::Load(path) => {
            let previous = EditorSnapshot::capture(&graph.0, &state);
            match load_creation_remapped(&mut graph.0, &path, &mut world_runtime) {
                Ok((name, sockets, creation)) => {
                    history.commit(previous);
                    history.mark_clean();
                    let bodies = creation.as_ref().map(|compiled| compiled.compounds.len());
                    let placed = sockets
                        .into_iter()
                        .map(|socket| PlacedBearing {
                            kind: socket.kind,
                            axis: socket.axis,
                            source: socket.source,
                            anchor: socket.anchor,
                            dimensions: socket.dimensions,
                        })
                        .collect();
                    adopt_loaded_creation(&mut state, &mut player, &mut camera, &graph.0, placed);
                    state.feedback = Some(if let Some(bodies) = bodies {
                        format!(
                            "Opened \"{name}\": {} parts, {} bearings, {bodies} bodies — F6 enters the live World",
                            graph.0.part_count(),
                            graph.0.bearing_count(),
                        )
                    } else {
                        format!(
                            "Opened \"{name}\" — complete matching transmission stacks before entering the World"
                        )
                    });
                    current.0 = Some(name);
                }
                Err(error) => {
                    state.feedback = Some(format!("Could not open creation: {error}"));
                }
            }
        }
        CreationRequest::Save(name) => {
            // Saving reads the graph and writes a file; it changes no
            // construction, so it commits no undo entry.
            let document = capture_creation(&graph.0, &state, &name);
            match store.save(&document) {
                Ok(path) => {
                    history.mark_clean();
                    current.0 = Some(name.clone());
                    state.feedback = Some(format!("Saved \"{name}\" to {}", path.display()));
                }
                Err(error) => {
                    state.feedback = Some(format!("Could not save creation: {error}"));
                }
            }
        }
        CreationRequest::Delete(path) => match creation_store::delete(&path) {
            Ok(()) => {
                state.feedback = Some(format!("Deleted {}", path.display()));
                if menu.is_open() {
                    menu.set_entries(store.list());
                }
            }
            Err(error) => {
                state.feedback = Some(format!("Could not delete creation: {error}"));
                if menu.is_open() {
                    menu.notify(format!("Could not delete: {error}"));
                }
            }
        },
    }
}

/// Captures everything a creation is: the construction, plus the bearing rings
/// the editor is holding that no part hangs from yet.
pub(crate) fn capture_creation(
    graph: &ConstructionGraph,
    state: &EditorState,
    name: &str,
) -> CreationDocument {
    let snapshot = EditorSnapshot::capture(graph, state);
    let sockets = snapshot
        .placed_bearings
        .iter()
        .map(|bearing| BearingSocket {
            kind: bearing.kind,
            axis: bearing.axis,
            source: bearing.source,
            anchor: bearing.anchor,
            dimensions: bearing.dimensions,
        })
        .collect::<Vec<_>>();
    CreationDocument::from_graph(&snapshot.graph, name, &sockets)
}

/// Reads a creation file and installs it, compiling before it commits.
pub(crate) type LoadedCreation = (String, Vec<BearingSocket>, Option<CompiledCreation>);

pub(crate) fn load_creation_remapped(
    current: &mut ConstructionGraph,
    path: &Path,
    runtime: &mut world::WorldRuntime,
) -> Result<LoadedCreation, Box<dyn Error>> {
    let mut document = creation_store::read_document(path)?;
    runtime.remap_imported_dimension_links(&mut document);
    let loaded = world::place_loaded_creation_in_garage(document.into_graph()?)?;
    match loaded.graph.compile() {
        Ok(creation) => {
            *current = loaded.graph;
            Ok((loaded.name, loaded.sockets, Some(creation)))
        }
        Err(TopologyError::TransmissionDepthMismatch { .. }) => {
            *current = loaded.graph;
            Ok((loaded.name, loaded.sockets, None))
        }
        Err(error) => Err(Box::new(error)),
    }
}

/// Clears the transient editing state a freshly opened creation invalidates,
/// then frames the camera on what arrived.
pub(crate) fn adopt_loaded_creation(
    state: &mut EditorState,
    player: &mut PlayerState,
    camera: &mut Single<
        (&mut PlayerCamera, &mut Transform, &mut GlobalTransform),
        With<MainCamera>,
    >,
    graph: &ConstructionGraph,
    placed_bearings: Vec<PlacedBearing>,
) {
    clear_hover(state);
    state.weld.cancel();
    state.suspension.controls.dismiss();
    state.suspension.drag = None;
    state.block_drag = None;
    state.pipe_drag = None;
    state.delete_drag = None;
    state.delete_target = None;
    state.selected_controller = None;
    state.placed_bearings = placed_bearings;
    state.construction_mesh_dirty = true;
    if let Some((minimum, maximum)) = graph_bounds(graph, &state.placed_bearings) {
        let (view, transform, global) = &mut **camera;
        player.place_outside_bounds(view, minimum, maximum);
        **transform = view.apply_pullback(
            player.position + Vec3::Y * camera::EYE_HEIGHT,
            view.look_rotation(),
        );
        **global = GlobalTransform::from(**transform);
    }
}

pub(crate) fn install_editor_graph(
    current: &mut ConstructionGraph,
    candidate: ConstructionGraph,
) -> Result<CompiledCreation, TopologyError> {
    let creation = candidate.compile()?;
    *current = candidate;
    Ok(creation)
}

pub(crate) fn graph_bounds(
    graph: &ConstructionGraph,
    sockets: &[PlacedBearing],
) -> Option<(Vec3, Vec3)> {
    let mut minimum = Vec3::splat(f32::INFINITY);
    let mut maximum = Vec3::splat(f32::NEG_INFINITY);
    for (part, _) in graph.parts() {
        let (part_minimum, part_maximum) = builder::composed_part_world_bounds(graph, part)?;
        minimum = minimum.min(part_minimum);
        maximum = maximum.max(part_maximum);
    }
    for (kind, anchor, axis) in graph
        .bearings()
        .map(|(_, bearing)| (bearing.kind, bearing.shared_anchor, bearing.axis))
        .chain(
            sockets
                .iter()
                .map(|socket| (socket.kind, socket.anchor, socket.axis)),
        )
    {
        if let mechanic_core::BearingKind::Suspension(spec) = kind {
            let rotation = Quat::from_rotation_arc(Vec3::Y, axis);
            for mesh in mechanic_core::suspension_meshes(spec, spec.starting_compression()) {
                for point in mesh.positions {
                    let point = anchor + rotation * Vec3::from_array(point);
                    minimum = minimum.min(point);
                    maximum = maximum.max(point);
                }
            }
        }
        if let mechanic_core::BearingKind::Linear(rail) = kind
            && let Ok(rotation) = rail.rotation(axis)
        {
            for chunk in mechanic_core::linear_bearing_meshes(rail.dimensions) {
                for point in chunk.positions {
                    let point = anchor + rotation * Vec3::from_array(point);
                    minimum = minimum.min(point);
                    maximum = maximum.max(point);
                }
            }
        }
    }
    minimum.is_finite().then_some((minimum, maximum))
}

#[cfg(test)]
mod tests {
    use bevy::prelude::Vec3;
    use mechanic_core::{BearingDimensions, FaceKind, FaceRef};

    use crate::editor::build_actions::PlacedBearing;
    use crate::editor::creation::{capture_creation, install_editor_graph};
    use crate::editor::state::EditorState;
    use crate::{
        creation_store::{CreationStore, read_document},
        showcase,
    };
    use mechanic_core::ConstructionGraph;

    struct TempDir(std::path::PathBuf);

    impl TempDir {
        fn new(label: &str) -> Self {
            let path = std::env::temp_dir()
                .join(format!("mechanic-creations-{}-{label}", std::process::id()));
            let _ = std::fs::remove_dir_all(&path);
            Self(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// A preset construction plus one bearing ring the editor is still holding.
    fn editor_with_a_loose_ring() -> (ConstructionGraph, EditorState) {
        let graph = showcase::build_preset(showcase::CreationPreset::PendulumGarden256)
            .expect("the preset builds");
        let (part, _) = graph.parts().next().expect("the preset has parts");
        let mut state = EditorState::default();
        state.placed_bearings.push(PlacedBearing {
            kind: mechanic_core::BearingKind::Rotational,
            axis: Vec3::ZERO,
            source: FaceRef::part(part, FaceKind::PositiveY),
            anchor: Vec3::new(0.25, 1.5, -0.75),
            dimensions: BearingDimensions::new(0.4, 0.15).expect("the ring is in range"),
        });
        (graph, state)
    }

    #[test]
    fn saving_then_opening_restores_the_construction_and_its_loose_rings() {
        let temporary = TempDir::new("round-trip");
        let store = CreationStore::new(&temporary.0);
        let (graph, state) = editor_with_a_loose_ring();

        let path = store
            .save(&capture_creation(&graph, &state, "Pendulum Rig"))
            .expect("the creation is written");

        let loaded = read_document(&path)
            .expect("the file parses")
            .into_graph()
            .expect("the document rebuilds");
        let mut installed = ConstructionGraph::new();
        let creation =
            install_editor_graph(&mut installed, loaded.graph).expect("the rebuild compiles");

        assert_eq!(loaded.name, "Pendulum Rig");
        assert_eq!(installed.part_count(), graph.part_count());
        assert_eq!(installed.weld_count(), graph.weld_count());
        assert_eq!(installed.bearing_count(), graph.bearing_count());
        assert_eq!(
            creation.compounds.len(),
            graph
                .compile()
                .expect("the original compiles")
                .compounds
                .len()
        );

        let restored = loaded.sockets.first().expect("the loose ring comes back");
        let original = &state.placed_bearings[0];
        assert_eq!(restored.anchor, original.anchor);
        assert_eq!(restored.dimensions, original.dimensions);
        assert_eq!(restored.source.face, original.source.face);
        assert_eq!(
            installed.parts().next().map(|(id, _)| id),
            match restored.source.owner {
                mechanic_core::FaceOwner::Part(part) => Some(part),
                mechanic_core::FaceOwner::Ground => None,
            },
            "the ring still hangs off the first part"
        );
    }

    #[test]
    fn the_listing_summarises_what_a_saved_creation_holds() {
        let temporary = TempDir::new("listing");
        let store = CreationStore::new(&temporary.0);
        let (graph, state) = editor_with_a_loose_ring();
        store
            .save(&capture_creation(&graph, &state, "Pendulum Rig"))
            .expect("the creation is written");

        let listed = store.list();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].name, "Pendulum Rig");
        assert_eq!(listed[0].part_count, graph.part_count());
        assert_eq!(listed[0].joint_count, graph.bearing_count());
    }

    use mechanic_core::{
        AppearanceTarget, BuildCommand, BuildOutcome, ConstructionMaterial, MaterialAppearance,
        PartId,
    };

    /// A 1 m steel block resting on the ground with a 25 cm rubber top layer.
    #[test]
    fn layer_ghost_over_a_block_slab_draws_only_its_outer_skin() {
        let block = |x: i32| {
            mechanic_core::PartSpec::Cuboid(
                mechanic_core::CuboidSpec::new(
                    [4, 4, 4],
                    mechanic_core::BuildPose::from_position_ticks(
                        [x, 200, 0].into(),
                        mechanic_core::GridRotation::default(),
                    ),
                )
                .unwrap(),
            )
            .with_layer(
                mechanic_core::LayerFace::Face(mechanic_core::FaceKind::PositiveY),
                0.25,
                ConstructionMaterial::Rubber,
                MaterialAppearance::BAKED,
            )
            .unwrap()
        };
        let mesh = crate::render::mesh::preview::layer_preview_mesh(&[block(0), block(400)]);
        let Some(bevy::mesh::Indices::U32(indices)) = mesh.indices() else {
            panic!("the ghost has 32-bit indices");
        };
        // Two touching boxes keep five faces each; the shared wall is gone.
        assert_eq!(indices.len(), 10 * 6);
    }

    fn layered_steel_block() -> (ConstructionGraph, PartId) {
        let mut graph = ConstructionGraph::new();
        let block = mechanic_core::PartSpec::Cuboid(
            mechanic_core::CuboidSpec::new(
                [4, 4, 4],
                mechanic_core::BuildPose::from_position_ticks(
                    [0, 200, 0].into(),
                    mechanic_core::GridRotation::default(),
                ),
            )
            .unwrap(),
        )
        .with_layer(
            mechanic_core::LayerFace::Face(mechanic_core::FaceKind::PositiveY),
            0.25,
            ConstructionMaterial::Rubber,
            MaterialAppearance::BAKED,
        )
        .unwrap();
        let BuildOutcome::Spawned(part) = graph
            .apply(BuildCommand::Spawn(block.as_cuboid().unwrap()))
            .unwrap()
        else {
            panic!("spawning a block reports its part");
        };
        (graph, part)
    }

    #[test]
    fn layered_part_renders_into_each_band_material_mesh() {
        let (graph, _) = layered_steel_block();
        for material in [ConstructionMaterial::Steel, ConstructionMaterial::Rubber] {
            let mesh = crate::render::mesh::construction::combined_material_construction_mesh(
                &graph, None, material,
            );
            assert!(mesh.count_vertices() > 0, "{material:?} band is drawn");
        }
        let concrete = ConstructionMaterial::ALL
            .into_iter()
            .find(|material| {
                !matches!(
                    material,
                    ConstructionMaterial::Steel | ConstructionMaterial::Rubber
                )
            })
            .unwrap();
        assert_eq!(
            crate::render::mesh::construction::combined_material_construction_mesh(
                &graph, None, concrete
            )
            .count_vertices(),
            0
        );
    }

    #[test]
    fn chroma_paints_only_the_hovered_band() {
        let (graph, part) = layered_steel_block();
        let mut state = crate::editor::state::EditorState::default();
        let hit = |point| crate::builder::SurfaceHit {
            distance: 1.0,
            point,
            face: mechanic_core::FaceRef::part(part, mechanic_core::FaceKind::PositiveZ),
        };
        state.hovered = Some(hit(Vec3::new(0.0, 0.9, 0.5)));
        assert_eq!(
            crate::editor::build_actions::appearance_target(&graph, &state),
            Some(AppearanceTarget::PartBand { part, band: 0 })
        );
        state.hovered = Some(hit(Vec3::new(0.0, 1.1, 0.5)));
        assert_eq!(
            crate::editor::build_actions::appearance_target(&graph, &state),
            Some(AppearanceTarget::PartBand { part, band: 1 })
        );
    }
}
