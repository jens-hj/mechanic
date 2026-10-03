//! Eight-hook coupler hardware, in the same local frame as its half-block envelope.

use crate::hardware_mesh::{MeshSink, ring};
use crate::{GRID_UNIT_METERS, InputMeshChunk, InputMeshOwner};
use bevy_math::{Quat, Vec3};

struct Mesh(InputMeshChunk);

impl MeshSink for Mesh {
    fn vertex_count(&self) -> u32 {
        u32::try_from(self.0.positions.len()).expect("small coupler mesh")
    }
    fn vertex(&mut self, position: Vec3, normal: Vec3, uv: [f32; 2]) {
        self.0.positions.push(position.to_array());
        self.0.normals.push(normal.to_array());
        self.0.uvs.push(uv);
    }
    fn triangles(&mut self, indices: &[u32]) {
        self.0.indices.extend_from_slice(indices);
    }
}

fn chunk(finish: usize) -> Mesh {
    Mesh(InputMeshChunk {
        owner: InputMeshOwner::Housing,
        finish,
        positions: Vec::new(),
        normals: Vec::new(),
        uvs: Vec::new(),
        indices: Vec::new(),
    })
}

fn block(mesh: &mut Mesh, center: Vec3, size: Vec3, rotation: Quat) {
    let half = size * 0.5;
    for (normal, u, v) in [
        (Vec3::X, Vec3::Y, Vec3::Z),
        (-Vec3::X, Vec3::Z, Vec3::Y),
        (Vec3::Y, Vec3::Z, Vec3::X),
        (-Vec3::Y, Vec3::X, Vec3::Z),
        (Vec3::Z, Vec3::X, Vec3::Y),
        (-Vec3::Z, Vec3::Y, Vec3::X),
    ] {
        let base = mesh.vertex_count();
        for (a, b) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
            mesh.vertex(
                center + rotation * ((normal + u * a + v * b) * half),
                rotation * normal,
                [a * 0.1, b * 0.1],
            );
        }
        mesh.triangles(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

pub(crate) fn meshes() -> Vec<InputMeshChunk> {
    let unit = GRID_UNIT_METERS;
    let mut steel = chunk(0);
    let mut dark = chunk(6);
    let mut marks = chunk(4);
    let mut signal = chunk(5);
    block(
        &mut steel,
        Vec3::Y * (-0.21 * unit),
        Vec3::new(unit, 0.08 * unit, unit),
        Quat::IDENTITY,
    );
    ring(
        &mut dark,
        [0.45 * unit, 0.29 * unit],
        0.28 * unit,
        -0.17 * unit,
        64,
        false,
        Vec3::ZERO,
    );
    ring(
        &mut steel,
        [0.43 * unit, 0.30 * unit],
        0.05 * unit,
        0.08 * unit,
        64,
        false,
        Vec3::ZERO,
    );
    for arm in 0_u8..8 {
        let rotation = Quat::from_rotation_y(f32::from(arm) * std::f32::consts::FRAC_PI_4);
        let at = |x, y, z| rotation * (Vec3::new(x, y, z) * unit);
        block(
            &mut steel,
            at(0.385, 0.18, 0.0),
            Vec3::new(0.09, 0.14, 0.08) * unit,
            rotation,
        );
        block(
            &mut steel,
            at(0.385, 0.225, 0.025),
            Vec3::new(0.09, 0.05, 0.13) * unit,
            rotation,
        );
        block(
            &mut marks,
            at(0.385, 0.2499, 0.025),
            Vec3::new(0.045, 0.0001, 0.065) * unit,
            rotation,
        );
    }
    block(
        &mut signal,
        Vec3::new(0.49, -0.17, 0.0) * unit,
        Vec3::new(0.02, 0.05, 0.2) * unit,
        Quat::IDENTITY,
    );
    vec![steel.0, dark.0, marks.0, signal.0]
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mesh_fits_half_block_and_has_valid_triangles() {
        for mesh in meshes() {
            assert_eq!(mesh.positions.len(), mesh.normals.len());
            assert_eq!(mesh.positions.len(), mesh.uvs.len());
            for point in &mesh.positions {
                assert!(point[0].abs() <= GRID_UNIT_METERS * 0.5 + 1e-5);
                assert!(point[1].abs() <= GRID_UNIT_METERS * 0.25 + 1e-5);
                assert!(point[2].abs() <= GRID_UNIT_METERS * 0.5 + 1e-5);
            }
            assert!(
                mesh.indices
                    .iter()
                    .all(|&index| (index as usize) < mesh.positions.len())
            );
        }
    }
}
