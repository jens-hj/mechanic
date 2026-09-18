//! Moving the active assembly between the garage and the world, with clearance checks.

use super::foundations::{PendingFoundationSync, TerrainFoundation};
use super::saving::space_instance;
use super::{
    AppSpace, IVec2, IVec3, Result, SpaceEditorState, String, ToOwned, ToString, Vec, Vec2, Vec3,
    WorldRuntime, format,
};
use crate::builder::bounds::graph_part_bounds;
use crate::builder::{
    GROUND_HALF_SIZE, PlacementSnapIndex, composed_part_world_bounds, part_world_bounds,
};
use crate::editor::build_actions::PlacedBearing;
use crate::editor::history::EditorHistory;
use crate::editor::state::EditorState;
use crate::garage;
use crate::simulation::state::AppSimulation;
use bevy::math::DVec3;
use mechanic_core::{
    BearingSocket, ConstructionGraph, CreationDocument, DimensionLinkId, FaceOwnerDoc, PartId,
    PartSpec,
};
use mechanic_world::{
    FloatingOrigin, FoundationSpatialIndex, FoundationSupport, TerrainDensity, TerrainMaterial,
    TerrainRayHit, TerrainScene, WorldInstanceIndexDoc, WorldPosition, raycast_density,
};
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn collision_free(
    candidate: &ConstructionGraph,
    destination: &ConstructionGraph,
    index: &PlacementSnapIndex,
) -> bool {
    let identity = mechanic_core::ConstructionFrame::IDENTITY;
    if candidate
        .parts()
        .all(|(part, _)| candidate.part_frame(part) == Some(identity))
        && destination
            .parts()
            .all(|(part, _)| destination.part_frame(part) == Some(identity))
    {
        return candidate
            .parts()
            .all(|(_, incoming)| !index.overlaps(*incoming));
    }
    candidate.parts().all(|(part, incoming)| {
        let frame = candidate
            .part_frame(part)
            .expect("candidate part has frame");
        destination.parts().all(|(other, existing)| {
            let other_frame = destination
                .part_frame(other)
                .expect("destination part has frame");
            if frame == identity && other_frame == identity {
                return !crate::builder::parts_overlap(*incoming, *existing);
            }
            // Cuboids retain their exact oriented box. Curved/featured parts use
            // a conservative authored box until transfer shares evaluated overlap.
            mechanic_core::obb_sat(
                transfer_part_box(*incoming, frame),
                transfer_part_box(*existing, other_frame),
            )
            .is_none_or(|contact| contact.penetration <= 1.0e-4)
        })
    })
}

pub(super) fn transfer_part_box(
    spec: PartSpec,
    frame: mechanic_core::ConstructionFrame,
) -> mechanic_core::Obb {
    if let Some(cuboid) = spec.as_cuboid() {
        return mechanic_core::Obb {
            center: frame.point(cuboid.pose.translation()),
            orientation: frame.rotation() * cuboid.pose.rotation.quaternion(),
            half_extents: cuboid.size_meters() * 0.5,
        };
    }
    let (low, high) = part_world_bounds(spec);
    mechanic_core::Obb {
        center: frame.point((low + high) * 0.5),
        orientation: frame.rotation(),
        half_extents: (high - low) * 0.5,
    }
}

pub(super) fn terrain_clear(
    candidate: &ConstructionGraph,
    terrain: &impl TerrainDensity,
    floating_origin: FloatingOrigin,
) -> bool {
    candidate.parts().all(|(part, _)| {
        let (low, high) =
            composed_part_world_bounds(candidate, part).expect("candidate part has frame");
        let inset_low = low + Vec3::splat(0.02);
        let inset_high = high - Vec3::splat(0.02);
        [0.0_f32, 0.5, 1.0].into_iter().all(|x| {
            [0.0_f32, 0.5, 1.0].into_iter().all(|y| {
                [0.0_f32, 0.5, 1.0].into_iter().all(|z| {
                    let local = inset_low + (inset_high - inset_low) * Vec3::new(x, y, z);
                    terrain.density(WorldPosition(floating_origin.0 + local.as_dvec3())) <= 0.0
                })
            })
        })
    })
}

pub(super) fn foundation_clear(
    candidate: &ConstructionGraph,
    terrain: &impl TerrainDensity,
    floating_origin: FloatingOrigin,
) -> bool {
    candidate.parts().all(|(part, _)| {
        !bounds_foundation_support(
            terrain,
            composed_part_world_bounds(candidate, part).expect("candidate part has frame"),
            floating_origin,
        )
        .has_valid_anchor()
    })
}

pub(super) fn framed_part_bounds(
    spec: PartSpec,
    frame: mechanic_core::ConstructionFrame,
) -> (Vec3, Vec3) {
    let (low, high) = part_world_bounds(spec);
    let mut minimum = Vec3::splat(f32::INFINITY);
    let mut maximum = Vec3::splat(f32::NEG_INFINITY);
    for x in [low.x, high.x] {
        for y in [low.y, high.y] {
            for z in [low.z, high.z] {
                let point = frame.point(Vec3::new(x, y, z));
                minimum = minimum.min(point);
                maximum = maximum.max(point);
            }
        }
    }
    (minimum, maximum)
}

pub(super) fn bounds_foundation_support(
    terrain: &impl TerrainDensity,
    (minimum, maximum): (Vec3, Vec3),
    floating_origin: FloatingOrigin,
) -> FoundationSupport {
    FoundationSupport::rectangular(
        terrain,
        TerrainRayHit {
            position: WorldPosition(
                floating_origin.0
                    + Vec3::new(
                        (minimum.x + maximum.x) * 0.5,
                        minimum.y,
                        (minimum.z + maximum.z) * 0.5,
                    )
                    .as_dvec3(),
            ),
            normal: Vec3::Y,
            distance: 0.0,
            material_weights: [0.0; TerrainMaterial::COUNT],
            chunk_generation: 0,
            triangle: 0,
        },
        f64::from(maximum.x - minimum.x),
        f64::from(maximum.z - minimum.z),
    )
}

pub(super) fn player_clear(candidate: &ConstructionGraph, feet: Vec3) -> bool {
    const TRANSFER_CLEARANCE: f32 = 2.0;
    candidate.parts().all(|(part, _)| {
        let (low, high) =
            composed_part_world_bounds(candidate, part).expect("candidate part has frame");
        let closest_x = feet.x.clamp(low.x, high.x);
        let closest_z = feet.z.clamp(low.z, high.z);
        Vec2::new(feet.x - closest_x, feet.z - closest_z).length_squared()
            > TRANSFER_CLEARANCE * TRANSFER_CLEARANCE
    })
}

pub(super) fn deterministic_offsets(maximum_radius_cells: i32) -> Vec<IVec2> {
    let mut offsets = (-maximum_radius_cells..=maximum_radius_cells)
        .flat_map(|z| (-maximum_radius_cells..=maximum_radius_cells).map(move |x| IVec2::new(x, z)))
        .filter(|offset| offset.x * offset.x + offset.y * offset.y <= maximum_radius_cells.pow(2))
        .collect::<Vec<_>>();
    offsets.sort_by_key(|offset| {
        (
            offset.x * offset.x + offset.y * offset.y,
            offset.y,
            offset.x,
        )
    });
    offsets
}

pub(super) fn component_document(
    graph: &ConstructionGraph,
    bearings: &[PlacedBearing],
    name: &str,
) -> CreationDocument {
    let sockets = bearings
        .iter()
        .map(|bearing| BearingSocket {
            kind: bearing.kind,
            axis: bearing.axis,
            source: bearing.source,
            anchor: bearing.anchor,
            dimensions: bearing.dimensions,
        })
        .collect::<Vec<_>>();
    CreationDocument::from_graph(graph, name, &sockets)
}

pub(super) fn detach_authored_ground(document: &mut CreationDocument) {
    document.welds.retain(|weld| {
        !matches!(weld.first.owner, FaceOwnerDoc::Ground)
            && !matches!(weld.second.owner, FaceOwnerDoc::Ground)
    });
}

pub(super) fn returned_component_parts(
    graph: &ConstructionGraph,
    active: DimensionLinkId,
) -> Result<BTreeSet<PartId>, String> {
    let link = graph
        .dimension_link(active)
        .ok_or_else(|| "returned Dimension Link is missing from the World".to_owned())?;
    graph
        .structural_component(link, [])
        .map(|component| component.parts().collect())
        .map_err(|error| error.to_string())
}

pub(super) fn remove_cached_foundations(
    foundations: &mut Vec<TerrainFoundation>,
    index: &mut FoundationSpatialIndex,
    parts: &BTreeSet<PartId>,
) -> bool {
    let removed = foundations
        .iter()
        .any(|foundation| parts.contains(&foundation.part));
    for &part in parts {
        index.remove(part);
    }
    foundations.retain(|foundation| !parts.contains(&foundation.part));
    removed
}

pub(super) fn static_parts_for_physics(
    known_parts: &BTreeMap<PartId, PartSpec>,
    foundations: &[TerrainFoundation],
    pending: Option<&PendingFoundationSync>,
    synced_editor_revision: u64,
    editor_revision: u64,
) -> Option<Vec<PartId>> {
    // Keep the previous physics publication until every new support has been
    // sampled. A partially reconciled cache must never release a ground weld.
    if synced_editor_revision != editor_revision || pending.is_some() {
        return None;
    }
    Some(
        foundations
            .iter()
            .filter(|foundation| {
                foundation.support.has_valid_anchor() && known_parts.contains_key(&foundation.part)
            })
            .map(|foundation| foundation.part)
            .collect(),
    )
}

pub(super) fn merge_document(
    destination: &SpaceEditorState,
    incoming: CreationDocument,
) -> Result<SpaceEditorState, String> {
    let mut document = component_document(
        &destination.graph,
        &destination.placed_bearings,
        "Combined construction",
    );
    document
        .append(incoming)
        .map_err(|error| error.to_string())?;
    let loaded = document.into_graph().map_err(|error| error.to_string())?;
    Ok(SpaceEditorState {
        origin: destination.origin,
        graph: loaded.graph,
        history: EditorHistory::default(),
        placed_bearings: loaded
            .sockets
            .into_iter()
            .map(|socket| PlacedBearing {
                kind: socket.kind,
                axis: socket.axis,
                source: socket.source,
                anchor: socket.anchor,
                dimensions: socket.dimensions,
            })
            .collect(),
    })
}

/// Saved creations use the same editable volume as Dimension Link transfers.
pub(crate) fn place_loaded_creation_in_garage(
    loaded: mechanic_core::LoadedCreation,
) -> Result<mechanic_core::LoadedCreation, String> {
    if loaded.graph.part_count() == 0 {
        return Ok(loaded);
    }
    let bearings = loaded
        .sockets
        .into_iter()
        .map(|socket| PlacedBearing {
            kind: socket.kind,
            axis: socket.axis,
            source: socket.source,
            anchor: socket.anchor,
            dimensions: socket.dimensions,
        })
        .collect::<Vec<_>>();
    let placed = place_in_garage(&loaded.graph, &bearings, &SpaceEditorState::default())?;
    component_document(&placed.graph, &placed.placed_bearings, &loaded.name)
        .into_graph()
        .map_err(|error| error.to_string())
}

pub(super) fn place_in_garage(
    component: &ConstructionGraph,
    bearings: &[PlacedBearing],
    destination: &SpaceEditorState,
) -> Result<SpaceEditorState, String> {
    let mut original = component_document(component, bearings, "Transferred construction");
    detach_authored_ground(&mut original);
    let mut destination_index = PlacementSnapIndex::default();
    destination_index.rebuild(&destination.graph);
    let offsets = deterministic_offsets(40);
    let mut dimension_overage = true;
    for yaw in 0_u8..4 {
        let mut rotated = original.clone();
        rotated.transform_cardinal(yaw, IVec3::ZERO);
        let rotated_loaded = rotated
            .clone()
            .into_graph()
            .map_err(|error| error.to_string())?;
        let Some((minimum, maximum)) = graph_part_bounds(&rotated_loaded.graph) else {
            continue;
        };
        let size = maximum - minimum;
        if size.x > GROUND_HALF_SIZE * 2.0 + 1.0e-4
            || size.z > GROUND_HALF_SIZE * 2.0 + 1.0e-4
            || size.y > garage::BUILD_MAX_Y - garage::BUILD_MIN_Y + 1.0e-4
        {
            continue;
        }
        dimension_overage = false;
        let center = (minimum + maximum) * 0.5;
        let base = IVec3::new(
            (-center.x / 0.125).round() as i32,
            // Rounding down can leave an off-grid creation below the build floor.
            ((garage::BUILD_MIN_Y - minimum.y) / 0.125).ceil() as i32,
            (-center.z / 0.125).round() as i32,
        );
        for offset in &offsets {
            let mut candidate = rotated.clone();
            candidate.transform_cardinal(0, base + IVec3::new(offset.x * 4, 0, offset.y * 4));
            let loaded = candidate
                .clone()
                .into_graph()
                .map_err(|error| error.to_string())?;
            let Some((low, high)) = graph_part_bounds(&loaded.graph) else {
                continue;
            };
            if low.x < -GROUND_HALF_SIZE - 1.0e-4
                || high.x > GROUND_HALF_SIZE + 1.0e-4
                || low.z < -GROUND_HALF_SIZE - 1.0e-4
                || high.z > GROUND_HALF_SIZE + 1.0e-4
                || low.y < garage::BUILD_MIN_Y - 1.0e-4
                || high.y > garage::BUILD_MAX_Y + 1.0e-4
                || !collision_free(&loaded.graph, &destination.graph, &destination_index)
            {
                continue;
            }
            return merge_document(destination, candidate);
        }
    }
    Err(if dimension_overage {
        "Linked assembly exceeds the Garage dimensions in every orientation".to_owned()
    } else {
        "Garage has no collision-free volume for the linked assembly".to_owned()
    })
}

pub(super) fn place_in_world(
    component: &ConstructionGraph,
    bearings: &[PlacedBearing],
    destination: &SpaceEditorState,
    target: Vec3,
    terrain: &impl TerrainDensity,
    floating_origin: FloatingOrigin,
) -> Result<SpaceEditorState, String> {
    let mut original = component_document(component, bearings, "Returned construction");
    detach_authored_ground(&mut original);
    let loaded = original
        .clone()
        .into_graph()
        .map_err(|error| error.to_string())?;
    let (minimum, maximum) =
        graph_part_bounds(&loaded.graph).ok_or_else(|| "linked assembly is empty".to_owned())?;
    let mut destination_index = PlacementSnapIndex::default();
    destination_index.rebuild(&destination.graph);
    let center = (minimum + maximum) * 0.5;
    let horizontal_base = IVec2::new(
        ((target.x - center.x) / 0.125).round() as i32,
        ((target.z - center.z) / 0.125).round() as i32,
    );
    for offset in deterministic_offsets(40) {
        let horizontal = horizontal_base + offset * 4;
        let local_center = Vec3::new(
            center.x + horizontal.x as f32 * 0.125,
            target.y,
            center.z + horizontal.y as f32 * 0.125,
        );
        let ray_origin =
            WorldPosition(floating_origin.0 + (local_center + Vec3::Y * 20.0).as_dvec3());
        let Some(surface) = raycast_density(terrain, ray_origin, DVec3::NEG_Y, 40.0) else {
            continue;
        };
        let surface_y = surface.position.relative_to(floating_origin).y;
        // Keep the authored pose above the foundation probe; live physics owns the landing.
        let vertical = ((surface_y - minimum.y) / 0.125).ceil() as i32 + 1;
        for clearance in 0..=1 {
            let mut candidate = original.clone();
            candidate.transform_cardinal(
                0,
                IVec3::new(horizontal.x, vertical + clearance, horizontal.y),
            );
            let loaded = candidate
                .clone()
                .into_graph()
                .map_err(|error| error.to_string())?;
            if collision_free(&loaded.graph, &destination.graph, &destination_index)
                && terrain_clear(&loaded.graph, terrain, floating_origin)
                && foundation_clear(&loaded.graph, terrain, floating_origin)
                && player_clear(&loaded.graph, target)
            {
                return merge_document(destination, candidate);
            }
        }
    }
    Err("No terrain- and construction-safe return placement was found within 20 m".to_owned())
}

pub(super) enum TransferAttempt {
    PlayerOnly,
    Transferred,
    Refused(String),
}

#[expect(clippy::too_many_lines)]
pub(super) fn transfer_active_assembly(
    space: AppSpace,
    runtime: &mut WorldRuntime,
    graph: &mut ConstructionGraph,
    history: &mut EditorHistory,
    editor: &mut EditorState,
    simulation: &mut AppSimulation,
) -> TransferAttempt {
    let Some(active) = runtime.document.active_dimension_link else {
        return TransferAttempt::PlayerOnly;
    };
    let Some(link_part) = graph.dimension_link(active) else {
        return TransferAttempt::PlayerOnly;
    };
    let component = match graph.structural_component(link_part, runtime.anchored_parts()) {
        Ok(component) => component,
        Err(error) => return TransferAttempt::Refused(error.to_string()),
    };
    if space == AppSpace::World && component.touches_authored_ground() {
        return TransferAttempt::Refused(
            "Linked assembly is terrain-anchored or welded to ground".to_owned(),
        );
    }
    let partition = graph.partition(&component);
    let (component_bearings, remainder_bearings): (Vec<_>, Vec<_>) = editor
        .placed_bearings
        .iter()
        .copied()
        .partition(|bearing| {
            matches!(bearing.source.owner, mechanic_core::FaceOwner::Part(part) if component.contains(part))
        });
    let destination = match space {
        AppSpace::World => {
            let Some(garage) = runtime.garage_editor.as_ref() else {
                return TransferAttempt::Refused("Garage state is unavailable".to_owned());
            };
            match place_in_garage(&partition.component, &component_bearings, garage) {
                Ok(placed) => placed,
                Err(error) => return TransferAttempt::Refused(error),
            }
        }
        AppSpace::Garage => {
            let Some(world) = runtime.world_editor.as_ref() else {
                return TransferAttempt::Refused("World state is unavailable".to_owned());
            };
            let anchor = runtime
                .document
                .return_anchor
                .unwrap_or(runtime.document.player_pose.translation)
                .relative_to(runtime.floating_origin);
            let terrain = TerrainScene {
                field: &runtime.field,
                edits: &runtime.edits,
            };
            match place_in_world(
                &partition.component,
                &component_bearings,
                world,
                anchor,
                &terrain,
                runtime.floating_origin,
            ) {
                Ok(placed) => placed,
                Err(error) => return TransferAttempt::Refused(error),
            }
        }
    };
    let returned_world_parts = match space {
        AppSpace::World => BTreeSet::new(),
        AppSpace::Garage => match returned_component_parts(&destination.graph, active) {
            Ok(parts) => parts,
            Err(error) => return TransferAttempt::Refused(error),
        },
    };
    let remainder = SpaceEditorState {
        origin: runtime.floating_origin,
        graph: partition.remainder,
        history: EditorHistory::default(),
        placed_bearings: remainder_bearings,
    };
    let (world_state, garage_state) = match space {
        AppSpace::World => (&remainder, &destination),
        AppSpace::Garage => (&destination, &remainder),
    };
    let mut world_doc = space_instance(
        &world_state.graph,
        &world_state.placed_bearings,
        "World construction",
    );
    world_doc.root_pose.translation = WorldPosition(world_state.origin.0);
    let garage_doc = space_instance(
        &garage_state.graph,
        &garage_state.placed_bearings,
        "Garage construction",
    );
    let previous_document = runtime.document.clone();
    runtime.document.frozen_creation = None;
    runtime.document.instances = (world_state.graph.part_count() != 0)
        .then(|| WorldInstanceIndexDoc {
            id: 1,
            name: "World construction".to_owned(),
        })
        .into_iter()
        .collect();
    if let Err(error) =
        runtime
            .store
            .save_space_pair(&mut runtime.document, &world_doc, &garage_doc)
    {
        runtime.document = previous_document;
        return TransferAttempt::Refused(format!("Could not persist transfer: {error}"));
    }
    match space {
        AppSpace::World => runtime.garage_editor = Some(destination),
        AppSpace::Garage => {
            let removed = remove_cached_foundations(
                &mut runtime.foundations,
                &mut runtime.foundation_index,
                &returned_world_parts,
            );
            runtime.pending_foundation_sync = None;
            runtime.synced_editor_revision = destination.history.current_revision.wrapping_add(1);
            if removed {
                runtime.foundation_revision = runtime.foundation_revision.wrapping_add(1);
            }
            runtime.world_editor = Some(destination);
        }
    }
    graph.clone_from(&remainder.graph);
    *history = EditorHistory::default();
    editor.placed_bearings = remainder.placed_bearings;
    editor.construction_mesh_dirty = true;
    *simulation = AppSimulation::default();
    runtime.autosave.saved();
    TransferAttempt::Transferred
}
