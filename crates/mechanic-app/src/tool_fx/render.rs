//! Three persistent instance buffers; world-space effects in the main HDR pass.
use bevy::core_pipeline::core_3d::TransparentSortingInfo3d;
use bevy::pbr::{
    self, MeshInputUniform, MeshPipelineSystems, MeshUniform, SetMeshViewBindingArrayBindGroup,
    ViewKeyCache,
};
use bevy::{
    core_pipeline::core_3d::Transparent3d,
    ecs::{
        query::QueryItem,
        system::{SystemParamItem, lifetimeless::Read},
    },
    mesh::{MeshVertexBufferLayoutRef, VertexBufferLayout},
    pbr::{MeshPipeline, MeshPipelineKey, RenderMeshInstances, SetMeshViewBindGroup},
    prelude::*,
    render::{
        Render, RenderApp, RenderStartup, RenderSystems,
        batching::gpu_preprocessing::BatchedInstanceBuffers,
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        mesh::RenderMesh,
        render_asset::RenderAssets,
        render_phase::{
            AddRenderCommand, DrawFunctions, PhaseItem, PhaseItemExtraIndex, RenderCommand,
            RenderCommandResult, SetItemPipeline, TrackedRenderPass, ViewSortedRenderPhases,
        },
        render_resource::{
            BlendComponent, BlendFactor, BlendOperation, BlendState, Buffer, BufferDescriptor,
            BufferUsages, PipelineCache, PrimitiveTopology, RenderPipelineDescriptor,
            SpecializedMeshPipeline, SpecializedMeshPipelineError, SpecializedMeshPipelines,
            VertexAttribute, VertexFormat, VertexStepMode,
        },
        renderer::RenderDevice,
        sync_component::SyncComponent,
        sync_world::MainEntity,
        view::ExtractedView,
    },
};
use bytemuck::{Pod, Zeroable};

/// World-space vertices bypass mesh transforms and lighting.
const SHADER_ASSET_PATH: &str = "shaders/tool_fx.wgsl";

#[derive(Component, Clone)]
pub(super) struct InstanceMaterialData {
    pub data: Vec<InstanceData>,
    pub lines: bool,
    pub capacity: usize,
}

impl SyncComponent for InstanceMaterialData {
    type Target = Self;
}

impl ExtractComponent for InstanceMaterialData {
    type QueryData = &'static InstanceMaterialData;
    type QueryFilter = ();
    type Out = Self;

    fn extract_component(item: QueryItem<'_, '_, Self::QueryData>) -> Option<Self> {
        Some(item.clone())
    }
}

#[derive(Component, Clone)]
pub(crate) struct FxCamera;
impl SyncComponent for FxCamera {
    type Target = Self;
}
impl ExtractComponent for FxCamera {
    type QueryData = &'static Self;
    type QueryFilter = ();
    type Out = Self;
    fn extract_component(_: QueryItem<'_, '_, Self::QueryData>) -> Option<Self> {
        Some(Self)
    }
}

pub(super) struct CustomMaterialPlugin;

impl Plugin for CustomMaterialPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            ExtractComponentPlugin::<InstanceMaterialData>::default(),
            ExtractComponentPlugin::<FxCamera>::default(),
        ));
        app.sub_app_mut(RenderApp)
            .add_render_command::<Transparent3d, DrawCustom>()
            .init_resource::<SpecializedMeshPipelines<CustomPipeline>>()
            .add_systems(
                RenderStartup,
                init_custom_pipeline.after(MeshPipelineSystems),
            )
            .add_systems(
                Render,
                (
                    queue_custom.in_set(RenderSystems::QueueMeshes),
                    prepare_instance_buffers.in_set(RenderSystems::PrepareResources),
                ),
            );
    }
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub(super) struct InstanceData {
    pub position_size: [f32; 4],
    pub rotation: [f32; 4],
    pub color: [f32; 4],
    pub endpoint: [f32; 4],
}

#[allow(clippy::too_many_arguments)]
fn queue_custom(
    transparent_3d_draw_functions: Res<DrawFunctions<Transparent3d>>,
    custom_pipeline: Res<CustomPipeline>,
    mut pipelines: ResMut<SpecializedMeshPipelines<CustomPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    meshes: Res<RenderAssets<RenderMesh>>,
    render_mesh_instances: Res<RenderMeshInstances>,
    maybe_batched_instance_buffers: Option<
        Res<BatchedInstanceBuffers<MeshUniform, MeshInputUniform>>,
    >,
    material_meshes: Query<(Entity, &MainEntity, &InstanceMaterialData)>,
    mut transparent_render_phases: ResMut<ViewSortedRenderPhases<Transparent3d>>,
    views: Query<&ExtractedView, With<FxCamera>>,
    view_key_cache: Res<ViewKeyCache>,
) {
    let draw_custom = transparent_3d_draw_functions.read().id::<DrawCustom>();

    for view in &views {
        let Some(transparent_phase) = transparent_render_phases.get_mut(&view.retained_view_entity)
        else {
            continue;
        };

        let Some(&view_key) = view_key_cache.get(&view.retained_view_entity) else {
            continue;
        };

        for (entity, main_entity, data) in &material_meshes {
            if data.data.is_empty() {
                continue;
            }
            let Some(mesh_instance) = render_mesh_instances.render_mesh_queue_data(*main_entity)
            else {
                continue;
            };
            let Some(mesh) = meshes.get(mesh_instance.mesh_asset_id()) else {
                continue;
            };
            let key = view_key
                | MeshPipelineKey::from_primitive_topology_and_strip_index(
                    mesh.primitive_topology(),
                    mesh.index_format(),
                );
            let pipeline = pipelines
                .specialize(
                    &pipeline_cache,
                    &custom_pipeline,
                    (key, data.lines),
                    &mesh.layout,
                )
                .unwrap();
            transparent_phase.add_retained(Transparent3d {
                sorting_info: TransparentSortingInfo3d::Sorted {
                    mesh_center: pbr::get_mesh_instance_world_from_local(
                        *main_entity,
                        mesh_instance.current_uniform_index,
                        &render_mesh_instances,
                        maybe_batched_instance_buffers.as_deref(),
                    )
                    .transform_point3(
                        meshes
                            .get(mesh_instance.mesh_asset_id())
                            .unwrap()
                            .aabb_center,
                    ),
                    depth_bias: if data.lines { -1.0e6 } else { 1.0e6 },
                },
                entity: (entity, *main_entity),
                pipeline,
                draw_function: draw_custom,
                distance: 0.0,
                batch_range: 0..1,
                extra_index: PhaseItemExtraIndex::None,
                indexed: false,
            });
        }
    }
}

#[derive(Component)]
struct InstanceBuffer {
    buffer: Buffer,
    length: u32,
    lines: bool,
}

fn prepare_instance_buffers(
    mut commands: Commands,
    mut query: Query<(Entity, &InstanceMaterialData, Option<&mut InstanceBuffer>)>,
    render_device: Res<RenderDevice>,
    queue: Res<bevy::render::renderer::RenderQueue>,
) {
    for (entity, data, existing) in &mut query {
        let length = u32::try_from(data.data.len()).expect("fixed particle budget");
        if let Some(mut existing) = existing {
            if length > 0 {
                queue.write_buffer(&existing.buffer, 0, bytemuck::cast_slice(&data.data));
            }
            existing.length = length;
        } else {
            let buffer = render_device.create_buffer(&BufferDescriptor {
                label: Some("Tool FX fixed instance buffer"),
                size: (data.capacity * size_of::<InstanceData>()) as u64,
                usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            if length > 0 {
                queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&data.data));
            }
            commands.entity(entity).insert(InstanceBuffer {
                buffer,
                length,
                lines: data.lines,
            });
        }
    }
}

#[derive(Resource)]
struct CustomPipeline {
    shader: Handle<Shader>,
    mesh_pipeline: MeshPipeline,
}

fn init_custom_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mesh_pipeline: Res<MeshPipeline>,
) {
    commands.insert_resource(CustomPipeline {
        shader: asset_server.load(SHADER_ASSET_PATH),
        mesh_pipeline: mesh_pipeline.clone(),
    });
}

impl SpecializedMeshPipeline for CustomPipeline {
    type Key = (MeshPipelineKey, bool);

    fn specialize(
        &self,
        key: Self::Key,
        layout: &MeshVertexBufferLayoutRef,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        let mut descriptor = self.mesh_pipeline.specialize(key.0, layout)?;
        descriptor.layout.truncate(2);
        descriptor.primitive.cull_mode = None;
        descriptor.primitive.topology = if key.1 {
            PrimitiveTopology::LineList
        } else {
            PrimitiveTopology::TriangleList
        };
        if let Some(depth) = &mut descriptor.depth_stencil {
            depth.depth_write_enabled = Some(!key.1);
        }
        let fragment = descriptor.fragment.as_mut().expect("mesh fragment stage");
        for target in fragment.targets.iter_mut().flatten() {
            target.blend = if key.1 {
                Some(BlendState {
                    color: BlendComponent {
                        src_factor: BlendFactor::One,
                        dst_factor: BlendFactor::One,
                        operation: BlendOperation::Add,
                    },
                    alpha: BlendComponent::OVER,
                })
            } else {
                None
            };
        }
        if key.1 {
            descriptor.vertex.shader_defs.push("FX_LINES".into());
        }

        descriptor.vertex.shader = self.shader.clone();
        descriptor.vertex.buffers = vec![VertexBufferLayout {
            array_stride: size_of::<InstanceData>() as u64,
            step_mode: VertexStepMode::Instance,
            attributes: (0..4)
                .map(|i| VertexAttribute {
                    format: VertexFormat::Float32x4,
                    offset: u64::from(i) * 16,
                    shader_location: i,
                })
                .collect(),
        }];
        descriptor.fragment.as_mut().unwrap().shader = self.shader.clone();
        Ok(descriptor)
    }
}

type DrawCustom = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetMeshViewBindingArrayBindGroup<1>,
    DrawMeshInstanced,
);

struct DrawMeshInstanced;

impl<P: PhaseItem> RenderCommand<P> for DrawMeshInstanced {
    type Param = ();
    type ViewQuery = ();
    type ItemQuery = Read<InstanceBuffer>;
    fn render<'w>(
        _item: &P,
        _view: (),
        instance_buffer: Option<&'w InstanceBuffer>,
        (): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let Some(buffer) = instance_buffer else {
            return RenderCommandResult::Skip;
        };
        if buffer.length == 0 {
            return RenderCommandResult::Skip;
        }
        pass.set_vertex_buffer(0, buffer.buffer.slice(..));
        pass.draw(0..if buffer.lines { 2 } else { 3 }, 0..buffer.length);
        RenderCommandResult::Success
    }
}
