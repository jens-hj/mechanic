//! Cached procedural piston visuals sharing construction textures.
//!
//! The mesh is built once per configuration. The body follows the supporting
//! body, the last stage follows whatever the head carries, and the stages
//! between them are placed from the measured extension, largest first.

use crate::chroma::ConstructionRenderMaterial;
use crate::editor::build_actions::PlacedBearing;
use crate::editor::preview::EditorVisuals;
use crate::editor::state::{EditorGraph, EditorState};
use crate::hotbar::{SelectedTool, Tool};
use crate::piston_editor;
use crate::pose::transform_from_gpu;
use crate::render::materials::material_index;
use crate::simulation::publication::WorldPhysicsRevision;
use crate::simulation::state::AppSimulation;
use bevy::{
    asset::RenderAssetUsages, mesh::Indices, prelude::*, render::render_resource::PrimitiveTopology,
};
use mechanic_core::{
    BearingKind, CompiledCreation, ConstructionGraph, FaceOwner, PISTON_FINISHES, Piston,
    PistonMeshOwner, piston_meshes,
};

#[derive(Clone, Copy, PartialEq)]
struct PistonVisualSpec {
    source: mechanic_core::FaceRef,
    anchor: Vec3,
    axis: Vec3,
    piston: Piston,
    body: Option<u32>,
    head: Option<u32>,
    preview_valid: Option<bool>,
}

impl PistonVisualSpec {
    fn same_piston(self, other: Self) -> bool {
        self.source == other.source
            && self.anchor.distance_squared(other.anchor) < 1.0e-10
            && self.axis.distance_squared(other.axis) < 1.0e-10
            && self.piston == other.piston
    }

    /// The piston-local frame at the collapsed build pose.
    fn build_pose(self) -> Transform {
        Transform::from_translation(self.piston.base_center(self.anchor, self.axis))
            .with_rotation(self.piston.rotation(self.axis).unwrap_or(Quat::IDENTITY))
    }
}

#[derive(Component)]
pub(super) struct PistonVisual {
    piston: usize,
    owner: PistonMeshOwner,
}

#[derive(Default)]
#[expect(
    clippy::type_complexity,
    reason = "cached configurations contain material-indexed owner meshes"
)]
pub(super) struct PistonRenderCache {
    specs: Vec<PistonVisualSpec>,
    revision: Option<WorldPhysicsRevision>,
    live: bool,
    entities: Vec<Entity>,
    materials: Vec<Handle<StandardMaterial>>,
    meshes: Vec<(Piston, Vec<(PistonMeshOwner, usize, Handle<Mesh>)>)>,
}

fn visual_specs(
    graph: &ConstructionGraph,
    sockets: &[PlacedBearing],
    creation: Option<&CompiledCreation>,
) -> Vec<PistonVisualSpec> {
    let mut specs = Vec::new();
    for (id, bearing) in graph.bearings() {
        let BearingKind::Piston(piston) = bearing.kind else {
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
        let spec = PistonVisualSpec {
            source: bearing.source,
            anchor: bearing.shared_anchor,
            axis: bearing.axis,
            piston,
            body: compiled.map(|row| row.compound_a),
            head: compiled.map(|row| row.compound_b),
            preview_valid: None,
        };
        if !specs
            .iter()
            .any(|existing: &PistonVisualSpec| existing.same_piston(spec))
        {
            specs.push(spec);
        }
    }
    for socket in sockets {
        let BearingKind::Piston(piston) = socket.kind else {
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
        let spec = PistonVisualSpec {
            source: socket.source,
            anchor: socket.anchor,
            axis: socket.axis,
            piston,
            body,
            head: body,
            preview_valid: None,
        };
        if !specs.iter().any(|existing| existing.same_piston(spec)) {
            specs.push(spec);
        }
    }
    specs
}

/// The piston-local frame as one simulated body carries it.
fn carried_pose(
    spec: PistonVisualSpec,
    body: Option<u32>,
    simulation: &AppSimulation,
) -> Option<Transform> {
    let body = body? as usize;
    let initial = simulation.creation.as_ref()?.compounds.get(body)?;
    let current = transform_from_gpu(simulation.transforms.get(body).copied()?);
    let build = spec.build_pose();
    let rotation_delta = current.rotation * initial.root_rotation.inverse();
    Some(
        Transform::from_translation(
            current.translation + rotation_delta * (build.translation - initial.root_translation),
        )
        .with_rotation(rotation_delta * build.rotation),
    )
}

/// Body and head poses of the piston-local frame; they differ by the extension.
fn poses(spec: PistonVisualSpec, simulation: &AppSimulation, live: bool) -> (Transform, Transform) {
    let build = spec.build_pose();
    if !live || spec.preview_valid.is_some() {
        return (build, build);
    }
    (
        carried_pose(spec, spec.body, simulation).unwrap_or(build),
        carried_pose(spec, spec.head, simulation).unwrap_or(build),
    )
}

fn visual_transform(
    spec: PistonVisualSpec,
    owner: PistonMeshOwner,
    simulation: &AppSimulation,
    live: bool,
) -> Transform {
    let (body, head) = poses(spec, simulation, live);
    match owner {
        PistonMeshOwner::Body => body,
        PistonMeshOwner::Stage(stage) if stage == spec.piston.dimensions.stages() => head,
        PistonMeshOwner::Stage(stage) => {
            let axis = body.rotation * Vec3::Y;
            let extension = (head.translation - body.translation).dot(axis);
            let offsets = spec.piston.dimensions.stage_offsets(extension);
            let offset = offsets[usize::from(stage.max(1) - 1)];
            Transform::from_translation(body.translation + axis * offset)
                .with_rotation(body.rotation)
        }
    }
}

/// Returns body and head poses using the same ownership as rendering.
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
    specs
        .iter()
        .find(|spec| {
            spec.source == socket.source
                && spec.anchor.distance_squared(socket.anchor) < 1.0e-10
                && spec.axis.distance_squared(socket.axis) < 1.0e-10
        })
        .map_or((Transform::IDENTITY, Transform::IDENTITY), |spec| {
            poses(*spec, simulation, live)
        })
}

fn render_mesh(chunk: mechanic_core::PistonMeshChunk) -> Mesh {
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
    mesh
}

/// One material per guide finish, or `None` until the construction textures have loaded.
fn finish_materials(
    visuals: &EditorVisuals,
    construction_materials: &Assets<ConstructionRenderMaterial>,
) -> Option<Vec<StandardMaterial>> {
    PISTON_FINISHES
        .iter()
        .map(|finish| {
            let index = material_index(finish.material);
            let base = construction_materials.get(&visuals.construction_materials[index])?;
            let mut material = base.base.clone();
            material.base_color = Color::srgb_u8(finish.color[0], finish.color[1], finish.color[2]);
            material.perceptual_roughness = finish.roughness;
            material.metallic = finish.metalness;
            Some(material)
        })
        .collect()
}

/// Register after simulation visual synchronization; no startup system is needed.
#[expect(clippy::too_many_arguments)]
pub(super) fn sync_piston_visuals(
    mut commands: Commands,
    graph: Res<EditorGraph>,
    state: Res<EditorState>,
    selected: Res<SelectedTool>,
    simulation: Res<AppSimulation>,
    visuals: Res<EditorVisuals>,
    construction_materials: Res<Assets<ConstructionRenderMaterial>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut cache: Local<PistonRenderCache>,
    mut entities: Query<(&PistonVisual, &mut Transform)>,
) {
    let live = simulation.is_running() && simulation.creation.is_some();
    let graph = if live {
        &simulation.published_graph
    } else {
        &graph.0
    };
    let mut specs = visual_specs(
        graph,
        &state.placed_bearings,
        live.then_some(simulation.creation.as_ref()).flatten(),
    );
    if selected.active_editor_tool() == Some(Tool::Piston)
        && let Some(socket) = piston_editor::preview_socket(graph, &state)
        && let BearingKind::Piston(piston) = socket.kind
    {
        specs.push(PistonVisualSpec {
            source: socket.source,
            anchor: socket.anchor,
            axis: socket.axis,
            piston,
            body: None,
            head: None,
            preview_valid: Some(state.preview_error.is_none()),
        });
    }
    if cache.specs == specs && cache.revision == simulation.world_revision && cache.live == live {
        if live {
            for (visual, mut transform) in &mut entities {
                if let Some(spec) = specs.get(visual.piston) {
                    *transform = visual_transform(*spec, visual.owner, &simulation, true);
                }
            }
        }
        return;
    }
    if cache.materials.is_empty() {
        let Some(finished) = finish_materials(&visuals, &construction_materials) else {
            return;
        };
        cache.materials = finished
            .into_iter()
            .map(|material| materials.add(material))
            .collect();
    }
    for entity in cache.entities.drain(..) {
        commands.entity(entity).despawn();
    }
    // Drop unused strong mesh handles when configurations disappear from the scene.
    cache
        .meshes
        .retain(|(piston, _)| specs.iter().any(|spec| same_mesh(spec.piston, *piston)));
    for (index, spec) in specs.iter().enumerate() {
        if !cache
            .meshes
            .iter()
            .any(|(existing, _)| same_mesh(*existing, spec.piston))
        {
            let chunks = piston_meshes(spec.piston)
                .into_iter()
                .map(|chunk| (chunk.owner, chunk.finish, meshes.add(render_mesh(chunk))))
                .collect();
            cache.meshes.push((spec.piston, chunks));
        }
        let chunks = cache
            .meshes
            .iter()
            .find(|(existing, _)| same_mesh(*existing, spec.piston))
            .expect("configuration inserted")
            .1
            .clone();
        for (owner, finish, mesh) in chunks {
            let entity = commands
                .spawn((
                    Name::new(format!("Piston {owner:?} {}", PISTON_FINISHES[finish].name)),
                    Mesh3d(mesh),
                    MeshMaterial3d(match spec.preview_valid {
                        Some(true) => visuals.green_preview_material.clone(),
                        Some(false) => visuals.red_preview_material.clone(),
                        None => cache.materials[finish].clone(),
                    }),
                    visual_transform(*spec, owner, &simulation, live),
                    Visibility::Visible,
                    PistonVisual {
                        piston: index,
                        owner,
                    },
                ))
                .id();
            cache.entities.push(entity);
        }
    }
    cache.specs = specs;
    cache.revision = simulation.world_revision;
    cache.live = live;
}

/// Meshes depend on the counts and on whether saddles are present, not on the mount's direction.
fn same_mesh(a: Piston, b: Piston) -> bool {
    a.dimensions == b.dimensions
        && std::mem::discriminant(&a.mount) == std::mem::discriminant(&b.mount)
}

/// Collapsed piston geometry shared by the weld ghost and the normal renderer.
pub(crate) fn weld_preview_mesh(
    graph: &ConstructionGraph,
    parts: &[mechanic_core::PartId],
    sockets: &[PlacedBearing],
) -> Mesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();
    for spec in visual_specs(graph, sockets, None)
        .into_iter()
        .filter(|spec| matches!(spec.source.owner, FaceOwner::Part(part) if parts.contains(&part)))
    {
        if graph.bearings().any(|(_, joint)| {
            joint.source == spec.source
                && !matches!(joint.target.owner, FaceOwner::Part(part) if parts.contains(&part))
        }) {
            continue;
        }
        let frame = spec.build_pose();
        for chunk in piston_meshes(spec.piston) {
            let offset = u32::try_from(positions.len()).expect("joint preview fits u32 indices");
            positions.extend(
                chunk
                    .positions
                    .iter()
                    .map(|&p| frame.transform_point(Vec3::from_array(p)).to_array()),
            );
            normals.extend(
                chunk
                    .normals
                    .iter()
                    .map(|&n| (frame.rotation * Vec3::from_array(n)).to_array()),
            );
            indices.extend(chunk.indices.iter().map(|index| index + offset));
        }
    }
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_indices(Indices::U32(indices))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mechanic_core::{BearingDimensions, PistonDimensions, PistonMount};
    use mechanic_gpu::GpuTransform;

    fn spec(extension: f32) -> (PistonVisualSpec, AppSimulation) {
        let mut graph = ConstructionGraph::new();
        let mut spawn = |ticks| {
            let mechanic_core::BuildOutcome::Spawned(part) = graph
                .apply(mechanic_core::BuildCommand::Spawn(
                    mechanic_core::CuboidSpec::new(
                        [1; 3],
                        mechanic_core::BuildPose::from_position_ticks(
                            ticks,
                            mechanic_core::GridRotation::default(),
                        ),
                    )
                    .unwrap(),
                ))
                .unwrap()
            else {
                panic!("expected construction part");
            };
            part
        };
        let base = spawn(IVec3::new(0, 50, 0));
        let load = spawn(IVec3::new(0, 350, 0));
        let piston = Piston {
            dimensions: PistonDimensions::new(2, 3).unwrap(),
            mount: PistonMount::End,
        };
        graph
            .apply(mechanic_core::BuildCommand::AddBearing(
                mechanic_core::BearingSpec::new(
                    mechanic_core::FaceRef::part(base, mechanic_core::FaceKind::PositiveY),
                    mechanic_core::FaceRef::part(load, mechanic_core::FaceKind::NegativeY),
                    Vec3::Y * 0.25,
                    Vec3::Y,
                )
                .with_kind(BearingKind::Piston(piston)),
            ))
            .unwrap();
        let creation = graph.compile().unwrap();
        let head = creation.bearings[0].compound_b;
        let transforms = creation
            .compounds
            .iter()
            .zip(0_u32..)
            .map(|(compound, row)| GpuTransform {
                position: (compound.root_translation
                    + if row == head {
                        Vec3::Y * extension
                    } else {
                        Vec3::ZERO
                    })
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
        let specs = visual_specs(&graph, &[], simulation.creation.as_ref());
        (specs[0], simulation)
    }

    #[test]
    fn stages_draw_largest_first_between_the_body_and_the_head() {
        let (spec, simulation) = spec(0.75);
        let height = |owner| {
            visual_transform(spec, owner, &simulation, true)
                .translation
                .y
        };
        let base = height(PistonMeshOwner::Body);
        assert!((base - 0.25).abs() < 1.0e-5);
        assert!((height(PistonMeshOwner::Stage(1)) - base - 0.5).abs() < 1.0e-5);
        assert!((height(PistonMeshOwner::Stage(2)) - base - 0.75).abs() < 1.0e-5);
        assert!((height(PistonMeshOwner::Stage(3)) - base - 0.75).abs() < 1.0e-5);
    }

    #[test]
    fn an_unattached_socket_renders_collapsed_on_its_support() {
        let (spec, simulation) = spec(0.0);
        let socket = PlacedBearing {
            kind: BearingKind::Piston(spec.piston),
            axis: Vec3::X,
            source: spec.source,
            anchor: spec.anchor + Vec3::X,
            dimensions: BearingDimensions::default(),
        };
        let (body, head) = socket_transforms(&simulation.published_graph, &simulation, socket);
        assert_eq!(body, head);
    }
}
