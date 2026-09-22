//! Cached authored control meshes; feedback only changes transforms and materials.

use crate::chroma::{ConstructionRenderMaterial, finish_base_color};
use crate::editor::preview::EditorVisuals;
use crate::editor::state::EditorGraph;
use crate::input_parts::InputParts;
use crate::physical_controls::PhysicalControls;
use crate::render::materials::material_index;
use crate::simulation::state::AppSimulation;
use bevy::{
    asset::RenderAssetUsages, mesh::Indices, prelude::*, render::render_resource::PrimitiveTopology,
};
use mechanic_core::{
    ButtonFeedback, DialFeedback, INPUT_FINISHES, InputKind, InputMeshOwner, InputSize, PartId,
    PartSpec, input_meshes,
};
use std::collections::BTreeMap;

#[derive(Component)]
pub(crate) struct InputVisual {
    part: Option<PartId>,
    owner: InputMeshOwner,
    size: InputSize,
    pointer: f32,
    signal: Option<Handle<StandardMaterial>>,
}

#[derive(Default)]
pub(crate) struct InputRenderCache {
    specs: Vec<(PartId, PartSpec)>,
    keys: Vec<(PartId, Option<mechanic_core::DriveKey>)>,
    labels: BTreeMap<(mechanic_core::DriveKey, InputSize), Handle<Mesh>>,
    preview: Option<crate::input_parts::InputPreview>,
    entities: Vec<Entity>,
    meshes: BTreeMap<(InputKind, InputSize), Vec<CachedInputMesh>>,
    materials: Vec<Handle<StandardMaterial>>,
    pointers: BTreeMap<PartId, f32>,
}

struct CachedInputMesh {
    owner: InputMeshOwner,
    finish: usize,
    mesh: Handle<Mesh>,
}

fn kind_size(spec: PartSpec) -> Option<(InputKind, InputSize)> {
    match spec {
        PartSpec::Dial(spec) => Some((InputKind::Dial, spec.size)),
        PartSpec::Button(spec) => Some((InputKind::Button, spec.size)),
        _ => None,
    }
}

#[expect(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "one cached authored-asset publication system"
)]
pub(crate) fn sync_input_visuals(
    mut commands: Commands,
    graph: Res<EditorGraph>,
    inputs: Res<InputParts>,
    simulation: Res<AppSimulation>,
    controls: Res<PhysicalControls>,
    visuals: Res<EditorVisuals>,
    construction_materials: Res<Assets<ConstructionRenderMaterial>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut cache: Local<InputRenderCache>,
    mut entities: Query<(&mut InputVisual, &mut Transform, &mut Visibility)>,
) {
    let live = simulation.is_running();
    let graph = if live {
        simulation.effective_graph()
    } else {
        &graph.0
    };
    let specs: Vec<_> = graph
        .parts()
        .filter(|(_, spec)| kind_size(**spec).is_some())
        .map(|(id, spec)| (id, *spec))
        .collect();
    if specs.is_empty() && inputs.preview.is_none() && cache.entities.is_empty() {
        return;
    }
    if cache.materials.is_empty() {
        let Some(bases) = INPUT_FINISHES
            .iter()
            .map(|finish| {
                construction_materials
                    .get(&visuals.construction_materials[material_index(finish.material)])
            })
            .collect::<Option<Vec<_>>>()
        else {
            return;
        };
        for (finish, base) in INPUT_FINISHES.iter().zip(bases) {
            let mut material = base.base.clone();
            material.base_color = finish_base_color(finish.material, finish.color);
            material.perceptual_roughness = finish.roughness;
            material.metallic = finish.metalness;
            cache.materials.push(materials.add(material));
        }
    }
    let keys = specs
        .iter()
        .map(|(part, _)| {
            (
                *part,
                graph
                    .input_configuration(*part)
                    .and_then(|config| config.key),
            )
        })
        .collect::<Vec<_>>();
    if cache.keys != keys
        || cache.specs != specs
        || cache.preview.map(|p| (p.spec, p.valid)) != inputs.preview.map(|p| (p.spec, p.valid))
    {
        for entity in cache.entities.drain(..) {
            commands.entity(entity).despawn();
        }
        for (part, spec, preview) in specs
            .iter()
            .map(|(id, spec)| (Some(*id), *spec, None))
            .chain(
                inputs
                    .preview
                    .map(|preview| (None, preview.spec, Some(preview))),
            )
        {
            let Some((kind, size)) = kind_size(spec) else {
                continue;
            };
            let chunks = cache.meshes.entry((kind, size)).or_insert_with(|| {
                input_meshes(kind, size)
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
                        let _ = mesh.generate_tangents();
                        CachedInputMesh {
                            owner: chunk.owner,
                            finish: chunk.finish,
                            mesh: meshes.add(mesh),
                        }
                    })
                    .collect()
            });
            let mut chunks: Vec<_> = chunks
                .iter()
                .map(|chunk| (chunk.owner, chunk.finish, chunk.mesh.clone()))
                .collect();
            if kind == InputKind::Button
                && let Some(key) = part
                    .and_then(|part| graph.input_configuration(part))
                    .and_then(|config| config.key)
            {
                let label = cache
                    .labels
                    .entry((key, size))
                    .or_insert_with(|| meshes.add(crate::input_label::mesh(key, size)))
                    .clone();
                chunks.push((InputMeshOwner::Cap, 5, label));
            }
            for (owner, finish, mesh) in chunks {
                let mut material = preview.map_or_else(
                    || cache.materials[finish].clone(),
                    |preview| {
                        if preview.valid {
                            visuals.green_preview_material.clone()
                        } else {
                            visuals.red_preview_material.clone()
                        }
                    },
                );
                let signal = if part.is_some()
                    && (finish == 5 || matches!(owner, InputMeshOwner::Tick(_)))
                {
                    let own = materials
                        .get(&material)
                        .expect("input finish exists")
                        .clone();
                    material = materials.add(own);
                    Some(material.clone())
                } else {
                    None
                };
                let transform = preview.map_or_else(
                    || {
                        let (translation, rotation) = part
                            .and_then(|part| simulation.live_part_pose(graph, part))
                            .unwrap_or((Vec3::ZERO, Quat::IDENTITY));
                        Transform::from_translation(translation).with_rotation(rotation)
                    },
                    |preview| preview.transform,
                );
                let visible = !matches!(
                    owner,
                    InputMeshOwner::Spring(ButtonFeedback::Mixed | ButtonFeedback::On)
                );
                let entity = commands
                    .spawn((
                        Name::new("Physical input"),
                        Mesh3d(mesh),
                        MeshMaterial3d(material),
                        transform,
                        if visible {
                            Visibility::Visible
                        } else {
                            Visibility::Hidden
                        },
                        InputVisual {
                            part,
                            owner,
                            size,
                            pointer: part
                                .and_then(|id| cache.pointers.get(&id).copied())
                                .unwrap_or(0.5),
                            signal,
                        },
                    ))
                    .id();
                cache.entities.push(entity);
            }
        }
        cache
            .pointers
            .retain(|part, _| specs.iter().any(|(id, _)| id == part));
        cache.specs = specs;
        cache.keys = keys;
        cache.preview = inputs.preview;
    }
    for (mut visual, mut transform, mut visibility) in &mut entities {
        let Some(part) = visual.part else {
            if let Some(preview) = inputs.preview {
                *transform = preview.transform;
            }
            continue;
        };
        let Some((translation, rotation)) = simulation.live_part_pose(graph, part) else {
            continue;
        };
        *transform = Transform::from_translation(translation).with_rotation(rotation);
        let config = graph.input_configuration(part);
        let feedback = if config.is_some_and(|config| {
            config
                .controller
                .zip(config.key)
                .is_some_and(|(controller, key)| controls.keys.held(controller, key))
        }) {
            ButtonFeedback::On
        } else {
            ButtonFeedback::Off
        };
        if let Some(mut material) = visual
            .signal
            .as_ref()
            .and_then(|handle| materials.get_mut(handle))
        {
            let (on, color) = match visual.owner {
                InputMeshOwner::Tick(index) => {
                    let dial = DialFeedback::from_values(
                        config
                            .into_iter()
                            .flat_map(|c| &c.analog)
                            .filter_map(|m| graph.numeric_value(m.target).map(|v| (m.range, v))),
                    );
                    let count = match visual.size {
                        InputSize::Panel => 7_u8,
                        InputSize::Utility => 13,
                        InputSize::Industrial => 19,
                    };
                    let on = config.is_some_and(|c| !c.analog.is_empty())
                        && match dial {
                            DialFeedback::Mixed => index % 2 == 0,
                            DialFeedback::Uniform(position) => {
                                position > 0.0
                                    && f32::from(index) / f32::from(count - 1) <= position
                            }
                        };
                    (on, LinearRgba::new(1.0, 0.45, 0.08, 1.0))
                }
                _ => (
                    feedback != ButtonFeedback::Off,
                    LinearRgba::new(0.05, 1.0, 0.55, 1.0),
                ),
            };
            material.emissive = if on { color } else { LinearRgba::BLACK };
            material.base_color = if on {
                Color::from(color)
            } else {
                Color::srgb(0.08, 0.12, 0.14)
            };
        }
        match visual.owner {
            InputMeshOwner::Rotor => {
                let feedback = DialFeedback::from_values(
                    config
                        .into_iter()
                        .flat_map(|config| &config.analog)
                        .filter_map(|mapping| {
                            graph
                                .numeric_value(mapping.target)
                                .map(|value| (mapping.range, value))
                        }),
                );
                if let DialFeedback::Uniform(position) = feedback {
                    visual.pointer = position;
                    cache.pointers.insert(part, position);
                }
                transform.rotation *=
                    Quat::from_rotation_y(-(visual.pointer - 0.5) * std::f32::consts::PI * 1.5);
            }
            InputMeshOwner::Cap => {
                transform.translation -=
                    rotation * Vec3::Y * visual.size.button_travel() * feedback.depression();
            }
            InputMeshOwner::Spring(pose) => {
                *visibility = if pose == feedback {
                    Visibility::Visible
                } else {
                    Visibility::Hidden
                };
            }
            _ => {}
        }
    }
}
