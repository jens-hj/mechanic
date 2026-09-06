//! Cached procedural rail and carriage visuals sharing construction textures.

use super::{
    AppSimulation, ConstructionRenderMaterial, EditorGraph, EditorState, EditorVisuals,
    PlacedBearing, SelectedTool, Tool, WorldPhysicsRevision, bearing_uses_socket, linear_editor,
    material_index, transform_from_gpu,
};
use bevy::{
    asset::RenderAssetUsages, mesh::Indices, prelude::*, render::render_resource::PrimitiveTopology,
};
use mechanic_core::{
    BearingKind, CompiledCreation, ConstructionGraph, ConstructionMaterial, FaceOwner,
    LINEAR_FINISHES, LinearBearing, LinearBearingDimensions, LinearMeshOwner,
    linear_bearing_meshes,
};

#[derive(Clone, Copy, PartialEq)]
struct RailVisualSpec {
    source: mechanic_core::FaceRef,
    anchor: Vec3,
    axis: Vec3,
    rail: LinearBearing,
    rail_body: Option<u32>,
    carriage_body: Option<u32>,
    preview_valid: Option<bool>,
}

impl RailVisualSpec {
    fn same_rail(self, other: Self) -> bool {
        self.source == other.source
            && self.anchor.distance_squared(other.anchor) < 1.0e-10
            && self.axis.distance_squared(other.axis) < 1.0e-10
            && self.rail.dimensions == other.rail.dimensions
            && self.rail.mount_normal == other.rail.mount_normal
    }
}

#[derive(Component)]
pub(super) struct LinearVisual {
    rail: usize,
    owner: LinearMeshOwner,
}

#[derive(Default)]
#[allow(clippy::type_complexity)] // Cached dimensions contain material-indexed owner meshes.
pub(super) struct LinearRenderCache {
    specs: Vec<RailVisualSpec>,
    revision: Option<WorldPhysicsRevision>,
    live: bool,
    attachment_entity: Option<Entity>,
    attachment_mesh: Option<Handle<Mesh>>,
    entities: Vec<Entity>,
    materials: Vec<Handle<StandardMaterial>>,
    meshes: Vec<(
        LinearBearingDimensions,
        Vec<(LinearMeshOwner, usize, Handle<Mesh>)>,
    )>,
}

fn visual_specs(
    graph: &ConstructionGraph,
    sockets: &[PlacedBearing],
    creation: Option<&CompiledCreation>,
) -> Vec<RailVisualSpec> {
    let mut specs = Vec::new();
    for (id, bearing) in graph.bearings() {
        let BearingKind::Linear(rail) = bearing.kind else {
            continue;
        };
        let compiled = creation.and_then(|creation| {
            creation
                .bearings
                .iter()
                .find(|row| row.source_bearing == id)
        });
        if creation.is_some() && compiled.is_none() {
            // The compiled representative supplies transforms for duplicate joint rows.
            continue;
        }
        let spec = RailVisualSpec {
            source: bearing.source,
            anchor: bearing.shared_anchor,
            axis: bearing.axis,
            rail,
            rail_body: compiled.map(|row| row.compound_a),
            carriage_body: compiled.map(|row| row.compound_b),
            preview_valid: None,
        };
        if !specs
            .iter()
            .any(|existing: &RailVisualSpec| existing.same_rail(spec))
        {
            specs.push(spec);
        }
    }
    for socket in sockets {
        let BearingKind::Linear(rail) = socket.kind else {
            continue;
        };
        let body = creation.and_then(|creation| {
            let FaceOwner::Part(part) = socket.source.owner else {
                return None;
            };
            creation
                .part_to_compound
                .iter()
                .find_map(|&(candidate, body)| (candidate == part).then_some(body))
        });
        let spec = RailVisualSpec {
            source: socket.source,
            anchor: socket.anchor,
            axis: socket.axis,
            rail,
            rail_body: body,
            carriage_body: body,
            preview_valid: None,
        };
        if !specs.iter().any(|existing| existing.same_rail(spec)) {
            specs.push(spec);
        }
    }
    specs
}

fn visual_transform(
    spec: RailVisualSpec,
    owner: LinearMeshOwner,
    simulation: &AppSimulation,
    live: bool,
) -> Transform {
    let rotation = spec.rail.rotation(spec.axis).unwrap_or(Quat::IDENTITY);
    let build = Transform::from_translation(spec.anchor).with_rotation(rotation);
    if !live || spec.preview_valid.is_some() {
        return build;
    }
    let body = match owner {
        LinearMeshOwner::Rail => spec.rail_body,
        LinearMeshOwner::Carriage => spec.carriage_body,
    };
    let Some(body) = body.map(|body| body as usize) else {
        return build;
    };
    let Some(initial) = simulation
        .creation
        .as_ref()
        .and_then(|c| c.compounds.get(body))
    else {
        return build;
    };
    let Some(current) = simulation.transforms.get(body).copied() else {
        return build;
    };
    let current = transform_from_gpu(current);
    let rotation_delta = current.rotation * initial.root_rotation.inverse();
    Transform::from_translation(
        current.translation + rotation_delta * (spec.anchor - initial.root_translation),
    )
    .with_rotation(rotation_delta * rotation)
}

/// Returns rail and carriage world poses using the same ownership as rendering.
pub(super) fn socket_transforms(
    graph: &ConstructionGraph,
    simulation: &AppSimulation,
    socket: PlacedBearing,
) -> (Transform, Transform) {
    let live = simulation.is_running() && simulation.creation.is_some();
    let graph = if live {
        &simulation.published_graph
    } else {
        graph
    };
    let specs = visual_specs(
        graph,
        &[socket],
        live.then_some(simulation.creation.as_ref()).flatten(),
    );
    let Some(spec) = specs.iter().find(|spec| {
        spec.source == socket.source
            && spec.anchor.distance_squared(socket.anchor) < 1.0e-10
            && spec.axis.distance_squared(socket.axis) < 1.0e-10
    }) else {
        return (Transform::IDENTITY, Transform::IDENTITY);
    };
    (
        visual_transform(*spec, LinearMeshOwner::Rail, simulation, live),
        visual_transform(*spec, LinearMeshOwner::Carriage, simulation, live),
    )
}

#[allow(clippy::too_many_arguments)]
fn sync_attachment_highlight(
    commands: &mut Commands,
    graph: &ConstructionGraph,
    state: &EditorState,
    selected: SelectedTool,
    simulation: &AppSimulation,
    visuals: &EditorVisuals,
    meshes: &mut Assets<Mesh>,
    cache: &mut LinearRenderCache,
) {
    let socket = state.linear_attachment.filter(|socket| {
        matches!(selected.active_editor_tool(), Some(Tool::Block | Tool::Cylinder))
            && matches!(socket.kind, BearingKind::Linear(rail) if !graph.bearings().any(|(_, bearing)| {
                bearing_uses_socket(bearing, *socket)
                    && matches!(bearing.kind, BearingKind::Linear(occupied) if occupied.face != rail.face)
            }))
    });
    let Some(socket) = socket else {
        if let Some(entity) = cache.attachment_entity.take() {
            commands.entity(entity).despawn();
        }
        return;
    };
    let BearingKind::Linear(rail) = socket.kind else {
        return;
    };
    let (_, carriage) = socket_transforms(graph, simulation, socket);
    let normal = rail.face.normal();
    let size = rail.face.size(rail.dimensions);
    // A thin translucent surface sits just outside the selected attachment plane.
    let local = Transform::from_translation(rail.face.origin(rail.dimensions) + normal * 0.0005)
        .with_rotation(Quat::from_rotation_arc(Vec3::Y, normal))
        .with_scale(Vec3::new(size.x, 0.0005, size.y));
    let transform = carriage.mul_transform(local);
    let material = if state.preview_error.is_some() {
        visuals.red_preview_material.clone()
    } else {
        visuals.green_preview_material.clone()
    };
    if let Some(entity) = cache.attachment_entity {
        commands
            .entity(entity)
            .insert((transform, MeshMaterial3d(material)));
    } else {
        let mesh = cache
            .attachment_mesh
            .get_or_insert_with(|| meshes.add(Cuboid::default()))
            .clone();
        cache.attachment_entity = Some(
            commands
                .spawn((
                    Name::new("Linear carriage attachment surface"),
                    Mesh3d(mesh),
                    MeshMaterial3d(material),
                    transform,
                    Visibility::Visible,
                ))
                .id(),
        );
    }
}

/// Register after simulation visual synchronization; no startup system is needed.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub(super) fn sync_linear_bearing_visuals(
    mut commands: Commands,
    graph: Res<EditorGraph>,
    state: Res<EditorState>,
    selected: Res<SelectedTool>,
    simulation: Res<AppSimulation>,
    visuals: Res<EditorVisuals>,
    construction_materials: Res<Assets<ConstructionRenderMaterial>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut cache: Local<LinearRenderCache>,
    mut entities: Query<(&LinearVisual, &mut Transform)>,
) {
    let live = simulation.is_running() && simulation.creation.is_some();
    let graph = if live {
        &simulation.published_graph
    } else {
        &graph.0
    };
    sync_attachment_highlight(
        &mut commands,
        graph,
        &state,
        *selected,
        &simulation,
        &visuals,
        &mut meshes,
        &mut cache,
    );
    let mut specs = visual_specs(
        graph,
        &state.placed_bearings,
        live.then_some(simulation.creation.as_ref()).flatten(),
    );
    if selected.active_editor_tool() == Some(Tool::LinearBearing)
        && let Some(socket) = linear_editor::preview_socket(graph, &state)
        && let BearingKind::Linear(rail) = socket.kind
    {
        specs.push(RailVisualSpec {
            source: socket.source,
            anchor: socket.anchor,
            axis: socket.axis,
            rail,
            rail_body: None,
            carriage_body: None,
            preview_valid: Some(state.preview_error.is_none()),
        });
    }
    if cache.specs == specs && cache.revision == simulation.world_revision && cache.live == live {
        if live {
            for (visual, mut transform) in &mut entities {
                if let Some(spec) = specs.get(visual.rail) {
                    *transform = visual_transform(*spec, visual.owner, &simulation, true);
                }
            }
        }
        return;
    }
    if cache.materials.is_empty() {
        let bases = [ConstructionMaterial::Aluminium, ConstructionMaterial::Steel].map(|kind| {
            construction_materials.get(&visuals.construction_materials[material_index(kind)])
        });
        let [Some(aluminium), Some(steel)] = bases else {
            return;
        };
        cache.materials = LINEAR_FINISHES
            .iter()
            .map(|finish| {
                let mut material = if finish.aluminium {
                    &aluminium.base
                } else {
                    &steel.base
                }
                .clone();
                material.base_color =
                    Color::srgb_u8(finish.color[0], finish.color[1], finish.color[2]);
                material.perceptual_roughness = finish.roughness;
                material.metallic = finish.metalness;
                materials.add(material)
            })
            .collect();
    }
    for entity in cache.entities.drain(..) {
        commands.entity(entity).despawn();
    }
    // Drop unused strong mesh handles when dimensions disappear from the scene.
    cache
        .meshes
        .retain(|(dimensions, _)| specs.iter().any(|s| s.rail.dimensions == *dimensions));
    for (index, spec) in specs.iter().enumerate() {
        let dimensions = spec.rail.dimensions;
        if !cache
            .meshes
            .iter()
            .any(|(existing, _)| *existing == dimensions)
        {
            let chunks = linear_bearing_meshes(dimensions)
                .into_iter()
                .map(|chunk| {
                    let mut mesh = Mesh::new(
                        PrimitiveTopology::TriangleList,
                        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
                    )
                    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, chunk.positions)
                    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, chunk.normals)
                    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, chunk.uvs)
                    .with_inserted_indices(Indices::U32(chunk.indices));
                    // Normal maps require tangents; indexed procedural faces carry valid UVs.
                    let _ = mesh.generate_tangents();
                    (chunk.owner, chunk.finish, meshes.add(mesh))
                })
                .collect();
            cache.meshes.push((dimensions, chunks));
        }
        let chunks = cache
            .meshes
            .iter()
            .find(|(existing, _)| *existing == dimensions)
            .expect("dimensions inserted")
            .1
            .clone();
        for (owner, finish, mesh) in chunks {
            let entity = commands
                .spawn((
                    Name::new(format!(
                        "Linear bearing {owner:?} {}",
                        LINEAR_FINISHES[finish].name
                    )),
                    Mesh3d(mesh),
                    MeshMaterial3d(match spec.preview_valid {
                        Some(true) => visuals.green_preview_material.clone(),
                        Some(false) => visuals.red_preview_material.clone(),
                        None => cache.materials[finish].clone(),
                    }),
                    visual_transform(*spec, owner, &simulation, live),
                    Visibility::Visible,
                    LinearVisual { rail: index, owner },
                ))
                .id();
            cache.entities.push(entity);
        }
    }
    cache.specs = specs;
    cache.revision = simulation.world_revision;
    cache.live = live;
}

#[cfg(test)]
mod tests {
    use super::*;
    use mechanic_core::BearingDimensions;
    use mechanic_gpu::GpuTransform;

    #[test]
    fn stopped_simulation_uses_build_pose_even_with_retained_world_snapshot() {
        let mut graph = ConstructionGraph::new();
        let mechanic_core::BuildOutcome::Spawned(part) = graph
            .apply(mechanic_core::BuildCommand::Spawn(
                mechanic_core::CuboidSpec::new([1; 3], mechanic_core::BuildPose::default())
                    .unwrap(),
            ))
            .unwrap()
        else {
            panic!("expected construction part");
        };
        let creation = graph.compile().unwrap();
        let transforms = creation
            .compounds
            .iter()
            .map(|compound| GpuTransform {
                position: (compound.root_translation + Vec3::new(3.0, 4.0, 5.0))
                    .extend(0.0)
                    .to_array(),
                rotation: Quat::IDENTITY.to_array(),
            })
            .collect();
        let simulation = AppSimulation {
            creation: Some(creation),
            published_graph: graph.clone(),
            transforms,
            world_revision: Some((1, 1)),
            ..default()
        };
        let socket = PlacedBearing {
            kind: BearingKind::Linear(LinearBearing {
                dimensions: LinearBearingDimensions::default(),
                mount_normal: Vec3::Y,
                face: mechanic_core::CarriageFace::Top,
            }),
            axis: Vec3::X,
            source: mechanic_core::FaceRef::part(part, mechanic_core::FaceKind::PositiveY),
            anchor: Vec3::Y * 0.125,
            dimensions: BearingDimensions::default(),
        };
        let (rail, carriage) = socket_transforms(&graph, &simulation, socket);
        assert_eq!(rail.translation, socket.anchor);
        assert_eq!(carriage.translation, socket.anchor);
    }
}
