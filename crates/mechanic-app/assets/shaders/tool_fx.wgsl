#import bevy_pbr::mesh_view_bindings::view
struct Vertex {
    @builtin(vertex_index) index: u32,
    @location(0) position_size: vec4<f32>,
    @location(1) rotation: vec4<f32>,
    @location(2) color: vec4<f32>,
    @location(3) endpoint: vec4<f32>,
};
struct Output { @builtin(position) position: vec4<f32>, @location(0) color: vec4<f32> };
@vertex fn vertex(v: Vertex) -> Output {
    var p: vec3<f32>;
#ifdef FX_LINES
    p = select(v.position_size.xyz, v.endpoint.xyz, v.index == 1u);
#else
    let triangle = array<vec3<f32>,3>(vec3(-0.5,-0.35,0.0),vec3(0.62,-0.2,0.0),vec3(-0.08,0.6,0.0));
    let local = triangle[v.index] * v.position_size.w;
    let q=v.rotation;
    p=v.position_size.xyz+local+2.0*cross(q.xyz,cross(q.xyz,local)+q.w*local);
#endif
    var out: Output;
    out.position=view.clip_from_world*vec4(p,1.0);
    out.color=v.color;
    return out;
}
@fragment fn fragment(v: Output) -> @location(0) vec4<f32> {return v.color;}
