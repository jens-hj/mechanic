//! Baked authored input meshes, in local metres about the envelope centre.
//!
//! Regenerate with `node scripts/physical-inputs/export.mjs`. The six models
//! retain independently authored mechanics from the approved concept. Consumers
//! cache by kind/size and animate owners; no geometry is rebuilt during operation.

use crate::ConstructionMaterial::{Aluminium, Plastic, Rubber, Steel};
use crate::{ButtonFeedback, HardwareFinish, InputSize};

/// Existing construction materials used by the input family.
pub const INPUT_FINISHES: [HardwareFinish; 9] = [
    HardwareFinish::new("steel", Steel, [158, 180, 198], 0.4, 1.0),
    HardwareFinish::new("aluminium", Aluminium, [199, 213, 221], 0.35, 1.0),
    HardwareFinish::new("grip", Rubber, [171, 180, 189], 0.8, 0.0),
    HardwareFinish::new("cap", Plastic, [175, 190, 203], 0.5, 0.0),
    HardwareFinish::new("pointer", Plastic, [242, 163, 60], 0.5, 0.0),
    HardwareFinish::new("signal", Plastic, [53, 191, 159], 0.5, 0.0),
    HardwareFinish::new("recess", Steel, [20, 35, 45], 0.5, 0.6),
    HardwareFinish::new("scale", Steel, [120, 132, 139], 0.7, 0.5),
    HardwareFinish::new("litScale", Plastic, [255, 180, 63], 0.5, 0.0),
];

/// Geometry cache key; each family has three independent models.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum InputKind {
    /// Rotary dial with a 270-degree sweep.
    Dial,
    /// Pushbutton with three cached spring poses.
    Button,
}

/// Independently animated or visible section of an input model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputMeshOwner {
    /// Fixed mounting and protective hardware.
    Housing,
    /// Fixed guide rods inside the spring bores.
    Guide,
    /// Knurled grip and pointer; rotate about local Y.
    Rotor,
    /// Cap and signal rim; translate down by the button travel.
    Cap,
    /// Show only for the matching button geometry pose.
    Spring(ButtonFeedback),
    /// Radial scale mark, in normalized sweep order.
    Tick(u8),
}

/// Renderer-neutral triangles using one construction material and motion owner.
#[derive(Clone, Debug)]
pub struct InputMeshChunk {
    /// Rigid motion or visibility group.
    pub owner: InputMeshOwner,
    /// Index into [`INPUT_FINISHES`].
    pub finish: usize,
    /// Local metre positions.
    pub positions: Vec<[f32; 3]>,
    /// Unit outward normals.
    pub normals: Vec<[f32; 3]>,
    /// Texture coordinates at 1.5 metres per repeat.
    pub uvs: Vec<[f32; 2]>,
    /// Counterclockwise triangle indices.
    pub indices: Vec<u32>,
}

/// Decodes an authored model for the application's geometry cache.
///
/// # Panics
/// Panics if the repository's generated asset contract is corrupted.
pub fn input_meshes(kind: InputKind, size: InputSize) -> Vec<InputMeshChunk> {
    let data: &[u8] = match (kind, size) {
        (InputKind::Dial, InputSize::Panel) => {
            include_bytes!("../assets/physical-inputs/dial-5.bin")
        }
        (InputKind::Dial, InputSize::Utility) => {
            include_bytes!("../assets/physical-inputs/dial-10.bin")
        }
        (InputKind::Dial, InputSize::Industrial) => {
            include_bytes!("../assets/physical-inputs/dial-25.bin")
        }
        (InputKind::Button, InputSize::Panel) => {
            include_bytes!("../assets/physical-inputs/button-5.bin")
        }
        (InputKind::Button, InputSize::Utility) => {
            include_bytes!("../assets/physical-inputs/button-10.bin")
        }
        (InputKind::Button, InputSize::Industrial) => {
            include_bytes!("../assets/physical-inputs/button-25.bin")
        }
    };
    let mut words = data
        .chunks_exact(4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("four-byte asset word")));
    let count = words.next().expect("mesh count");
    let chunks = (0..count)
        .map(|_| {
            let owner = match words.next().expect("owner") {
                0 => InputMeshOwner::Housing,
                1 => InputMeshOwner::Rotor,
                2 => InputMeshOwner::Cap,
                3 => InputMeshOwner::Spring(ButtonFeedback::Off),
                4 => InputMeshOwner::Spring(ButtonFeedback::Mixed),
                5 => InputMeshOwner::Spring(ButtonFeedback::On),
                6 => InputMeshOwner::Guide,
                tick @ 10..=28 => {
                    InputMeshOwner::Tick(u8::try_from(tick - 10).expect("bounded scale"))
                }
                _ => panic!("invalid authored mesh owner"),
            };
            let finish = words.next().expect("finish") as usize;
            let vertices = words.next().expect("vertex count");
            let indices = words.next().expect("index count");
            InputMeshChunk {
                owner,
                finish,
                positions: (0..vertices)
                    .map(|_| {
                        std::array::from_fn(|_| f32::from_bits(words.next().expect("position")))
                    })
                    .collect(),
                normals: (0..vertices)
                    .map(|_| std::array::from_fn(|_| f32::from_bits(words.next().expect("normal"))))
                    .collect(),
                uvs: (0..vertices)
                    .map(|_| std::array::from_fn(|_| f32::from_bits(words.next().expect("uv"))))
                    .collect(),
                indices: (0..indices).map(|_| words.next().expect("index")).collect(),
            }
        })
        .collect();
    assert!(words.next().is_none(), "asset has no trailing words");
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BuildPose, ButtonSpec, DialSpec};
    use bevy_math::Vec3;

    #[test]
    fn every_model_fits_its_actual_envelope_with_valid_triangles() {
        for size in InputSize::ALL {
            for kind in [InputKind::Dial, InputKind::Button] {
                let dimensions = match kind {
                    InputKind::Dial => DialSpec::new(size, BuildPose::default()).size_meters(),
                    InputKind::Button => ButtonSpec::new(size, BuildPose::default()).size_meters(),
                };
                for chunk in input_meshes(kind, size) {
                    assert!(chunk.finish < INPUT_FINISHES.len());
                    assert_eq!(chunk.positions.len(), chunk.normals.len());
                    assert_eq!(chunk.positions.len(), chunk.uvs.len());
                    assert_eq!(chunk.indices.len() % 3, 0);
                    for position in &chunk.positions {
                        assert!(
                            (Vec3::from_array(*position).abs() - dimensions * 0.5).max_element()
                                < 1e-6,
                            "{kind:?} {size:?} {position:?}"
                        );
                    }
                    for normal in &chunk.normals {
                        assert!((Vec3::from_array(*normal).length() - 1.0).abs() < 1e-4);
                    }
                    assert!(
                        chunk
                            .indices
                            .iter()
                            .all(|&index| (index as usize) < chunk.positions.len())
                    );
                }
            }
        }
    }

    #[test]
    fn all_sizes_have_independent_rotors_radial_scales_and_three_spring_poses() {
        for (size, tick_count) in InputSize::ALL.into_iter().zip([7, 13, 19]) {
            let dial = input_meshes(InputKind::Dial, size);
            assert!(dial.iter().any(|c| c.owner == InputMeshOwner::Rotor));
            assert_eq!(
                dial.iter()
                    .filter(|c| matches!(c.owner, InputMeshOwner::Tick(_)))
                    .count(),
                tick_count
            );
            let button = input_meshes(InputKind::Button, size);
            for feedback in [
                ButtonFeedback::Off,
                ButtonFeedback::Mixed,
                ButtonFeedback::On,
            ] {
                assert!(
                    button
                        .iter()
                        .any(|c| c.owner == InputMeshOwner::Spring(feedback))
                );
            }
        }
    }
    fn closed_surface(chunk: &InputMeshChunk) -> bool {
        let key = |index: u32| {
            chunk.positions[index as usize].map(|v| if v == 0.0 { 0 } else { v.to_bits() })
        };
        let mut edges = std::collections::BTreeMap::new();
        for triangle in chunk.indices.chunks_exact(3) {
            for [a, b] in [
                [triangle[0], triangle[1]],
                [triangle[1], triangle[2]],
                [triangle[2], triangle[0]],
            ] {
                let (a, b) = (key(a), key(b));
                *edges
                    .entry(if a < b { (a, b) } else { (b, a) })
                    .or_insert(0) += 1;
            }
        }
        edges.values().all(|&count| count == 2)
    }

    #[test]
    fn knurl_is_one_closed_surface_and_scale_marks_point_radially() {
        for size in InputSize::ALL {
            let chunks = input_meshes(InputKind::Dial, size);
            let grips: Vec<_> = chunks
                .iter()
                .filter(|c| c.owner == InputMeshOwner::Rotor && c.finish == 2)
                .collect();
            assert_eq!(grips.len(), 1);
            assert!(closed_surface(grips[0]));
            for tick in chunks
                .iter()
                .filter(|c| matches!(c.owner, InputMeshOwner::Tick(_)))
            {
                let count = f32::from(u16::try_from(tick.positions.len()).unwrap());
                let centre = tick
                    .positions
                    .iter()
                    .copied()
                    .map(Vec3::from_array)
                    .sum::<Vec3>()
                    / count;
                let radial = Vec3::new(centre.x, 0.0, centre.z).normalize();
                let tangent = Vec3::Y.cross(radial);
                let (mut radial_variance, mut tangent_variance, mut cross) = (0.0, 0.0, 0.0);
                for point in &tick.positions {
                    let delta = Vec3::from_array(*point) - centre;
                    let r = delta.dot(radial);
                    let t = delta.dot(tangent);
                    radial_variance += r * r;
                    tangent_variance += t * t;
                    cross += r * t;
                }
                assert!(radial_variance > tangent_variance * 4.0);
                assert!(cross.abs() < radial_variance * 1e-4);
            }
        }
    }

    #[test]
    fn every_cached_spring_pose_clears_the_moving_cap() {
        for size in InputSize::ALL {
            let chunks = input_meshes(InputKind::Button, size);
            let cap_bottom = chunks
                .iter()
                // Guide sleeves sit inside the coil bore, below the load-bearing cap.
                .filter(|c| c.owner == InputMeshOwner::Cap && c.finish != 6)
                .flat_map(|c| c.positions.iter())
                .map(|p| p[1])
                .fold(f32::INFINITY, f32::min);
            let guide_top = chunks
                .iter()
                .filter(|c| c.owner == InputMeshOwner::Guide)
                .flat_map(|c| c.positions.iter())
                .map(|p| p[1])
                .fold(f32::NEG_INFINITY, f32::max);
            assert!(
                guide_top < cap_bottom - size.button_travel(),
                "{size:?} guide rod clearance"
            );
            for feedback in [
                ButtonFeedback::Off,
                ButtonFeedback::Mixed,
                ButtonFeedback::On,
            ] {
                let spring_top = chunks
                    .iter()
                    .filter(|c| c.owner == InputMeshOwner::Spring(feedback))
                    .flat_map(|c| c.positions.iter())
                    .map(|p| p[1])
                    .fold(f32::NEG_INFINITY, f32::max);
                assert!(
                    spring_top < cap_bottom - size.button_travel() * feedback.depression(),
                    "{size:?} {feedback:?}"
                );
            }
        }
    }
}
