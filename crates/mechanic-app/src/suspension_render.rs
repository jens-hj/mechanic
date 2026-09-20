//! Procedural suspension rendering and triangle picking share cached core meshes.
use crate::chroma::ConstructionRenderMaterial;
use crate::editor::build_actions::PlacedBearing;
use crate::editor::preview::EditorVisuals;
use crate::editor::state::{EditorGraph, EditorState};
use crate::pose::transform_from_gpu;
use crate::render::materials::material_index;
use crate::simulation::state::AppSimulation;
use bevy::{
    asset::RenderAssetUsages, mesh::Indices, prelude::*, render::render_resource::PrimitiveTopology,
};
use mechanic_core::{
    CompiledCreation, ConstructionFrame, ConstructionGraph, FaceOwner, JointKind,
    SUSPENSION_FINISHES, SuspensionMeshChunk, SuspensionMeshOwner, SuspensionSpec,
    suspension_meshes,
};
use std::cell::RefCell;

static GEOMETRY_BUILDS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(crate) fn geometry_builds() -> u64 {
    GEOMETRY_BUILDS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Per-frame counters for the work this module repeats while springs move.
///
/// Only read by the performance capture, which takes and clears them once per
/// frame, so a counter left unread never accumulates across a whole session.
static DEFORMATION_REBUILDS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static TANGENT_GENERATIONS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static MATERIAL_WRITES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PICK_TRIANGLES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Deformation rebuilds, tangent generations, material writes and picked
/// triangles since the previous call, clearing each counter.
pub(crate) fn take_visual_work() -> [u64; 4] {
    use std::sync::atomic::Ordering::Relaxed;
    [
        DEFORMATION_REBUILDS.swap(0, Relaxed),
        TANGENT_GENERATIONS.swap(0, Relaxed),
        MATERIAL_WRITES.swap(0, Relaxed),
        PICK_TRIANGLES.swap(0, Relaxed),
    ]
}
fn cached_meshes(spec: SuspensionSpec, compression: f32) -> Vec<SuspensionMeshChunk> {
    GEOMETRY_BUILDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    suspension_meshes(spec, compression)
}

/// A free opposite mount previews its new spacing; attached mounts retain live separation.
pub(crate) fn draft_compression(
    original: SuspensionSpec,
    draft: SuspensionSpec,
    current: f32,
    attached: bool,
) -> f32 {
    if attached {
        (draft.extended_length() - (original.extended_length() - current))
            .clamp(0.0, draft.compression_limit().0)
    } else {
        draft.starting_compression()
    }
}

/// Damping belongs to the solver; changing it cannot invalidate mesh topology.
fn same_geometry(a: SuspensionSpec, b: SuspensionSpec) -> bool {
    let shape = |s: mechanic_core::ShockSpec| (s.length(), s.od(), s.body_end());
    a.spring() == b.spring()
        && a.shock().map(shape) == b.shock().map(shape)
        && a.bump_stop() == b.bump_stop()
        && a.plates() == b.plates()
        && a.appearances() == b.appearances()
}

#[derive(Clone, Copy, PartialEq)]
struct VisualSpec {
    socket: PlacedBearing,
    source_body: Option<u32>,
    joint: Option<usize>,
    preview_valid: Option<bool>,
    preview_host: Option<SuspensionSpec>,
}
impl VisualSpec {
    fn same_geometry(self, other: Self) -> bool {
        same_geometry(self.suspension(), other.suspension())
    }
    fn same_mount(self, other: Self) -> bool {
        self.socket.source == other.socket.source
            && self.socket.anchor.distance_squared(other.socket.anchor) < 1e-10
            && self.socket.axis.distance_squared(other.socket.axis) < 1e-10
    }
    fn suspension(self) -> SuspensionSpec {
        let JointKind::Suspension(spec) = self.socket.kind else {
            unreachable!("suspension visual")
        };
        spec
    }
}
fn visual_specs(
    graph: &ConstructionGraph,
    sockets: &[PlacedBearing],
    creation: Option<&CompiledCreation>,
) -> Vec<VisualSpec> {
    let mut specs = Vec::new();
    for (id, bearing) in graph.bearings() {
        if !matches!(bearing.kind, JointKind::Suspension(_)) {
            continue;
        }
        let joint =
            creation.and_then(|c| c.bearings.iter().position(|row| row.source_bearing == id));
        if creation.is_some() && joint.is_none() {
            continue;
        }
        let spec = VisualSpec {
            socket: PlacedBearing {
                source: bearing.source,
                anchor: bearing.shared_anchor,
                axis: bearing.axis,
                dimensions: bearing.dimensions,
                kind: bearing.kind,
            },
            source_body: joint.and_then(|i| creation.map(|c| c.bearings[i].compound_a)),
            joint,
            preview_valid: None,
            preview_host: None,
        };
        if !specs.iter().any(|s: &VisualSpec| s.same_mount(spec)) {
            specs.push(spec);
        }
    }
    for &socket in sockets {
        if !matches!(socket.kind, JointKind::Suspension(_)) {
            continue;
        }
        let source_body = creation.and_then(|c| {
            let FaceOwner::Part(part) = socket.source.owner else {
                return None;
            };
            c.part_to_compound
                .iter()
                .find_map(|&(p, body)| (p == part).then_some(body))
        });
        let spec = VisualSpec {
            socket,
            source_body,
            joint: None,
            preview_valid: None,
            preview_host: None,
        };
        if !specs.iter().any(|s| s.same_mount(spec)) {
            specs.push(spec);
        }
    }
    specs
}
fn pose(spec: &VisualSpec, simulation: Option<&AppSimulation>) -> (Transform, f32) {
    if let Some(simulation) = simulation.filter(|s| s.is_running() && spec.preview_valid.is_none())
        && let Some(creation) = simulation.creation.as_ref()
    {
        return snapshot_pose(spec, creation, &simulation.transforms);
    }
    build_pose(spec)
}
fn build_pose(spec: &VisualSpec) -> (Transform, f32) {
    (
        Transform::from_translation(spec.socket.anchor)
            .with_rotation(Quat::from_rotation_arc(Vec3::Y, spec.socket.axis)),
        spec.suspension().starting_compression(),
    )
}
fn snapshot_pose(
    spec: &VisualSpec,
    creation: &CompiledCreation,
    transforms: &[mechanic_gpu::GpuTransform],
) -> (Transform, f32) {
    let suspension = spec.suspension();
    let (build, _) = build_pose(spec);
    let rotation = build.rotation;
    let Some(body) = spec.source_body.map(|b| b as usize) else {
        return (build, suspension.starting_compression());
    };
    let Some(initial) = creation.compounds.get(body) else {
        return (build, suspension.starting_compression());
    };
    let Some(current) = transforms.get(body).copied() else {
        return (build, suspension.starting_compression());
    };
    let current = transform_from_gpu(current);
    let delta = current.rotation * initial.root_rotation.inverse();
    let transform = Transform::from_translation(
        current.translation + delta * (spec.socket.anchor - initial.root_translation),
    )
    .with_rotation(delta * rotation);
    let q = spec
        .joint
        .and_then(|index| {
            let row = creation.bearings.get(index)?;
            let a = transform_from_gpu(*transforms.get(row.compound_a as usize)?);
            let b = transform_from_gpu(*transforms.get(row.compound_b as usize)?);
            Some(
                (b.transform_point(row.local_anchor_b) - a.transform_point(row.local_anchor_a))
                    .dot(a.rotation * row.local_axis_a),
            )
        })
        .unwrap_or(0.0);
    (
        transform,
        (suspension.starting_compression() - q).clamp(0.0, suspension.compression_limit().0),
    )
}
// EditorView leaves previews in its local tool frame, but restores committed
// sockets to authored build coordinates. Keep preview identity in build space
// and its moving world transform separate, so motion does not rebuild topology.
fn preview_spec(
    socket: PlacedBearing,
    valid: bool,
    authored: Option<ConstructionFrame>,
) -> VisualSpec {
    VisualSpec {
        socket: authored.map_or(socket, |frame| {
            super::live_edit::transform_bearing(socket, frame)
        }),
        source_body: None,
        joint: None,
        preview_valid: Some(valid),
        preview_host: None,
    }
}
fn insertion_pose(
    preview: &VisualSpec,
    host: SuspensionSpec,
    host_pose: (Transform, f32),
) -> (Transform, f32) {
    let actual_length = host.extended_length() - host_pose.1;
    let replacement = preview.suspension();
    (
        host_pose.0,
        (replacement.extended_length() - actual_length)
            .clamp(0.0, replacement.compression_limit().0),
    )
}
fn render_pose(
    spec: &VisualSpec,
    simulation: Option<&AppSimulation>,
    preview_build_to_world: Option<ConstructionFrame>,
) -> (Transform, f32) {
    let (mut transform, compression) = if let Some(host) = spec.preview_host {
        let host_spec = VisualSpec {
            socket: PlacedBearing {
                kind: JointKind::Suspension(host),
                ..spec.socket
            },
            preview_valid: None,
            preview_host: None,
            ..*spec
        };
        let host_pose = pose(&host_spec, simulation);
        let insertion = insertion_pose(spec, host, host_pose);
        if spec.source_body.is_some()
            && simulation.is_some_and(|s| s.is_running() && s.creation.is_some())
        {
            // The host source frame is already in world space, including live
            // translation and rotation. Do not apply the edit frame twice.
            return insertion;
        }
        insertion
    } else {
        pose(spec, simulation)
    };
    if spec.preview_valid.is_some()
        && let Some(frame) = preview_build_to_world
    {
        transform.translation = frame.point(transform.translation);
        transform.rotation = frame.rotation() * transform.rotation;
    }
    (transform, compression)
}
/// Source frame and physical compression used by rendering and live picking.
pub(super) fn socket_pose(
    graph: &ConstructionGraph,
    simulation: Option<&AppSimulation>,
    socket: PlacedBearing,
) -> (Transform, f32) {
    let live = simulation.filter(|s| s.is_running() && s.creation.is_some());
    let graph = live.map_or(graph, |s| &s.published_graph);
    let specs = visual_specs(graph, &[socket], live.and_then(|s| s.creation.as_ref()));
    let reference = VisualSpec {
        socket,
        source_body: None,
        joint: None,
        preview_valid: None,
        preview_host: None,
    };
    pose(
        &specs
            .into_iter()
            .find(|s| s.same_mount(reference))
            .unwrap_or(reference),
        live,
    )
}

struct RenderChunk {
    entity: Entity,
    mesh: Handle<Mesh>,
    geometry: SuspensionMeshChunk,
}
struct AssemblyVisual {
    chunks: Vec<RenderChunk>,
    compression: f32,
    preview: bool,
}
/// System-local topology, GPU handles, and shared guide finish materials.
#[derive(Default)]
pub(super) struct SuspensionRenderCache {
    specs: Vec<VisualSpec>,
    assemblies: Vec<AssemblyVisual>,
    materials: Vec<Handle<ConstructionRenderMaterial>>,
}
pub(super) fn render_mesh(
    chunk: &SuspensionMeshChunk,
    appearance: Option<mechanic_core::MaterialAppearance>,
) -> Mesh {
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, chunk.positions.clone())
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, chunk.normals.clone())
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, chunk.uvs.clone())
    .with_inserted_indices(Indices::U32(chunk.indices.clone()));
    if let Some(appearance) = appearance {
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_COLOR,
            vec![super::chroma::encode_appearance(appearance); chunk.positions.len()],
        );
    }
    let _ = mesh.generate_tangents();
    mesh
}
/// Maps the construction texture's representative colour to the guide finish,
/// retaining its texture variation instead of multiplying two dark base colours.
/// Captures reuse this same material path as interactive suspension rendering.
pub(super) fn finish_material(
    base: &ConstructionRenderMaterial,
    finish: mechanic_core::HardwareFinish,
) -> ConstructionRenderMaterial {
    let mut material = base.clone();
    let target_color = Color::srgb_u8(finish.color[0], finish.color[1], finish.color[2]);
    material.base.base_color = super::chroma::finish_base_color(finish.material, finish.color);
    material.base.perceptual_roughness = finish.roughness;
    material.base.metallic = finish.metalness;
    material.extension.base_lightness.x = bevy::color::Oklaba::from(target_color).lightness;
    material
}

/// A chunk has one render-material component, including when cached geometry changes role.
fn set_chunk_material(
    commands: &mut Commands,
    entity: Entity,
    preview: Option<Handle<StandardMaterial>>,
    authored: Handle<ConstructionRenderMaterial>,
) {
    let mut entity = commands.entity(entity);
    if let Some(preview) = preview {
        entity
            .remove::<MeshMaterial3d<ConstructionRenderMaterial>>()
            .insert(MeshMaterial3d(preview));
    } else {
        entity
            .remove::<MeshMaterial3d<StandardMaterial>>()
            .insert(MeshMaterial3d(authored));
    }
}

/// Synchronizes cached suspension visuals after the simulation snapshot updates.
#[expect(clippy::too_many_arguments, clippy::too_many_lines)]
pub(super) fn sync_suspension_visuals(
    mut commands: Commands,
    graph: Res<EditorGraph>,
    state: Res<EditorState>,
    simulation: Res<AppSimulation>,
    visuals: Res<EditorVisuals>,
    mut construction_materials: ResMut<Assets<ConstructionRenderMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut cache: Local<SuspensionRenderCache>,
) {
    let live = simulation.is_running() && simulation.creation.is_some();
    let preview_frames = state.edit_context.and_then(|context| {
        graph
            .0
            .part_frame(context.anchor)
            .map(|authored| (authored, context.frame_to_world.compose(authored.inverse())))
    });
    let preview_build_to_world = preview_frames.map(|(_, world)| world);
    let graph = if live {
        &simulation.published_graph
    } else {
        &graph.0
    };
    let mut specs = visual_specs(
        graph,
        &state.placed_bearings,
        if live {
            simulation.creation.as_ref()
        } else {
            None
        },
    );
    if let Some(socket) = state
        .suspension
        .preview
        .filter(|socket| matches!(socket.kind, JointKind::Suspension(_)))
    {
        let mut preview = preview_spec(
            socket,
            state.preview_error.is_none(),
            preview_frames.map(|(authored, _)| authored),
        );
        if let Some(host) = specs.iter().find(|host| host.same_mount(preview)) {
            preview.preview_host = Some(host.suspension());
            preview.source_body = host.source_body;
            preview.joint = host.joint;
        }
        // Insertion previews replace the existing assembly, including resized plates.
        specs.retain(|spec| !spec.same_mount(preview));
        specs.push(preview);
    }
    if let Some(gesture) = state.suspension.controls.gesture.as_ref()
        && let Some(spec) = specs
            .iter_mut()
            .find(|s| crate::suspension_controls::Target(s.socket) == gesture.target)
    {
        spec.socket.kind = JointKind::Suspension(gesture.draft);
        spec.preview_valid = Some(gesture.error.is_none());
        spec.preview_host = Some(gesture.original);
    }
    if cache.materials.is_empty() {
        if SUSPENSION_FINISHES.iter().any(|finish| {
            construction_materials
                .get(&visuals.construction_materials[material_index(finish.material)])
                .is_none()
        }) {
            return;
        }
        for finish in SUSPENSION_FINISHES {
            let Some(base) = construction_materials
                .get(&visuals.construction_materials[material_index(finish.material)])
            else {
                return;
            };
            let material = finish_material(base, finish);
            cache.materials.push(construction_materials.add(material));
        }
    }
    let pose_for = |spec: &VisualSpec| {
        if let Some(gesture) = state
            .suspension
            .controls
            .gesture
            .as_ref()
            .filter(|g| g.target == crate::suspension_controls::Target(spec.socket))
        {
            let (transform, compression) =
                socket_pose(graph, live.then_some(&simulation), gesture.target.0);
            let attached =
                !crate::editor::build_actions::bearing_socket_targets(graph, gesture.target.0)
                    .is_empty();
            return (
                transform,
                draft_compression(gesture.original, gesture.draft, compression, attached),
            );
        }
        render_pose(spec, live.then_some(&simulation), preview_build_to_world)
    };
    let finish_materials = cache.materials.clone();
    let old_specs = std::mem::take(&mut cache.specs);
    let mut old = old_specs
        .into_iter()
        .zip(std::mem::take(&mut cache.assemblies))
        .collect::<Vec<_>>();
    for &spec in &specs {
        if let Some(index) = old
            .iter()
            .position(|(previous, _)| previous.same_geometry(spec))
        {
            cache.assemblies.push(old.swap_remove(index).1);
        } else {
            let (transform, compression) = pose_for(&spec);
            let chunks = cached_meshes(spec.suspension(), compression)
                .into_iter()
                .map(|geometry| {
                    let component = match geometry.owner {
                        SuspensionMeshOwner::Spring => 0,
                        SuspensionMeshOwner::BumpStop => 2,
                        SuspensionMeshOwner::Source | SuspensionMeshOwner::Opposite => {
                            usize::from(spec.suspension().shock().is_some())
                        }
                    };
                    let appearance = Some(spec.suspension().appearances()[component]);
                    let mesh = meshes.add(render_mesh(&geometry, appearance));
                    let entity = commands
                        .spawn((
                            Name::new(format!(
                                "Suspension {:?} {}",
                                geometry.owner, SUSPENSION_FINISHES[geometry.finish].name
                            )),
                            Mesh3d(mesh.clone()),
                            transform,
                            Visibility::Visible,
                        ))
                        .id();
                    set_chunk_material(
                        &mut commands,
                        entity,
                        spec.preview_valid.map(|valid| {
                            if valid {
                                visuals.green_preview_material.clone()
                            } else {
                                visuals.red_preview_material.clone()
                            }
                        }),
                        cache.materials[geometry.finish].clone(),
                    );
                    RenderChunk {
                        entity,
                        mesh,
                        geometry,
                    }
                })
                .collect();
            cache.assemblies.push(AssemblyVisual {
                chunks,
                compression,
                preview: false,
            });
        }
    }
    for (_, assembly) in old {
        for chunk in assembly.chunks {
            commands.entity(chunk.entity).despawn();
        }
    }
    {
        for (spec, assembly) in specs.iter().zip(&mut cache.assemblies) {
            let (transform, compression) = pose_for(spec);
            for chunk in &mut assembly.chunks {
                commands.entity(chunk.entity).insert(transform);
                MATERIAL_WRITES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                set_chunk_material(
                    &mut commands,
                    chunk.entity,
                    spec.preview_valid.map(|valid| {
                        if valid {
                            visuals.green_preview_material.clone()
                        } else {
                            visuals.red_preview_material.clone()
                        }
                    }),
                    finish_materials[chunk.geometry.finish].clone(),
                );
                if assembly.preview != spec.preview_valid.is_some()
                    && let Some(mut mesh) = meshes.get_mut(&chunk.mesh)
                {
                    if spec.preview_valid.is_some() {
                        mesh.remove_attribute(Mesh::ATTRIBUTE_COLOR);
                    } else {
                        let component = match chunk.geometry.owner {
                            SuspensionMeshOwner::Spring => 0,
                            SuspensionMeshOwner::BumpStop => 2,
                            _ => usize::from(spec.suspension().shock().is_some()),
                        };
                        mesh.insert_attribute(
                            Mesh::ATTRIBUTE_COLOR,
                            vec![
                                super::chroma::encode_appearance(
                                    spec.suspension().appearances()[component]
                                );
                                chunk.geometry.positions.len()
                            ],
                        );
                    }
                }
                if (assembly.compression - compression).abs() > 1e-7 {
                    DEFORMATION_REBUILDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    chunk.geometry.update_deformation(compression);
                    if let Some(mut mesh) = meshes.get_mut(&chunk.mesh) {
                        mesh.insert_attribute(
                            Mesh::ATTRIBUTE_POSITION,
                            chunk.geometry.positions.clone(),
                        );
                        mesh.insert_attribute(
                            Mesh::ATTRIBUTE_NORMAL,
                            chunk.geometry.normals.clone(),
                        );
                        if chunk.geometry.owner == SuspensionMeshOwner::Spring
                            || chunk.geometry.finish == 8
                        {
                            TANGENT_GENERATIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            let _ = mesh.generate_tangents();
                        }
                    }
                }
            }
            assembly.compression = compression;
            assembly.preview = spec.preview_valid.is_some();
        }
    }
    cache.specs = specs;
}

thread_local! {
    static PICK_MESHES:RefCell<Vec<(SuspensionSpec,Vec<SuspensionMeshChunk>)>>=const {RefCell::new(Vec::new())};
}
fn triangle_hit(
    origin: Vec3,
    direction: Vec3,
    first_vertex: Vec3,
    second_vertex: Vec3,
    third_vertex: Vec3,
) -> Option<f32> {
    let edge = second_vertex - first_vertex;
    let second = third_vertex - first_vertex;
    let cross = direction.cross(second);
    let determinant = edge.dot(cross);
    if determinant.abs() < 1e-9 {
        return None;
    }
    let offset = origin - first_vertex;
    let u = offset.dot(cross) / determinant;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = offset.cross(edge);
    let v = direction.dot(q) / determinant;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let distance = second.dot(q) / determinant;
    (distance >= 0.0).then_some(distance)
}
/// Nearest actual hardware triangle, with independent component ownership.
pub(super) fn raycast_scene_component(
    graph: &ConstructionGraph,
    simulation: Option<&AppSimulation>,
    sockets: &[PlacedBearing],
    origin: Vec3,
    direction: Vec3,
) -> Option<(usize, f32, SuspensionMeshOwner)> {
    PICK_MESHES.with(|cache| {
        let mut cache = cache.borrow_mut();
        cache.retain(|(spec, _)| {
            sockets.iter().any(
                |s| matches!(s.kind, JointKind::Suspension(other) if same_geometry(*spec, other)),
            )
        });
        let mut hit: Option<(usize, f32, SuspensionMeshOwner)> = None;
        for (index, &socket) in sockets.iter().enumerate() {
            let JointKind::Suspension(spec) = socket.kind else {
                continue;
            };
            let (pose, compression) = socket_pose(graph, simulation, socket);
            let origin = pose.rotation.inverse() * (origin - pose.translation);
            let direction = pose.rotation.inverse() * direction;
            let entry = if let Some(i) = cache
                .iter()
                .position(|(existing, _)| same_geometry(*existing, spec))
            {
                i
            } else {
                cache.push((spec, cached_meshes(spec, compression)));
                cache.len() - 1
            };
            for chunk in &mut cache[entry].1 {
                chunk.update_deformation(compression);
                PICK_TRIANGLES.fetch_add(
                    u64::try_from(chunk.indices.len() / 3).unwrap_or(u64::MAX),
                    std::sync::atomic::Ordering::Relaxed,
                );
                for triangle in chunk.indices.chunks_exact(3) {
                    let [a, b, c] = [triangle[0], triangle[1], triangle[2]]
                        .map(|i| Vec3::from_array(chunk.positions[i as usize]));
                    if let Some(distance) = triangle_hit(origin, direction, a, b, c)
                        && hit.is_none_or(|(_, best, _)| distance < best)
                    {
                        hit = Some((index, distance, chunk.owner));
                    }
                }
            }
        }
        hit
    })
}
/// Nearest suspension socket index and ray distance, using deformed triangles.
pub(super) fn raycast_scene(
    graph: &ConstructionGraph,
    simulation: Option<&AppSimulation>,
    sockets: &[PlacedBearing],
    origin: Vec3,
    direction: Vec3,
) -> Option<(usize, f32)> {
    raycast_scene_component(graph, simulation, sockets, origin, direction)
        .map(|(index, distance, _)| (index, distance))
}

#[cfg(test)]
mod tests;
