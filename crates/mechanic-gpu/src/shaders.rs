//! Compute kernel sources.
//!
//! Every ABI struct has one WGSL definition under `kernels/prelude`, and every
//! constant the CPU and GPU must agree on is generated from its Rust owner in
//! [`crate::abi`]. A kernel lists the preludes it needs; [`Kernel::source`]
//! assembles the module, so no kernel file repeats a layout or a flag value.

use crate::abi;

/// One WGSL compute module: its shared preludes and its own entry points.
pub(crate) struct Kernel {
    /// File stem under `kernels/`, named when validation fails.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "read only by the kernel validation tests")
    )]
    pub(crate) name: &'static str,
    preludes: &'static [&'static str],
    body: &'static str,
}

const TICK_CONFIG: &str = include_str!("kernels/prelude/tick_config.wgsl");
const BEARING: &str = include_str!("kernels/prelude/bearing.wgsl");
const MASS: &str = include_str!("kernels/prelude/mass.wgsl");
const MECHANISM: &str = include_str!("kernels/prelude/mechanism.wgsl");
const LINK_STATE: &str = include_str!("kernels/prelude/link_state.wgsl");
const DRIVE: &str = include_str!("kernels/prelude/drive.wgsl");
const COLLIDER: &str = include_str!("kernels/prelude/collider.wgsl");

/// Free-body integration, external impulses, and state validation.
pub(crate) const PHYSICS: Kernel = Kernel {
    name: "physics",
    preludes: &[TICK_CONFIG, MASS],
    body: include_str!("kernels/physics.wgsl"),
};

/// Snapshot-ring publication.
pub(crate) const SNAPSHOT: Kernel = Kernel {
    name: "snapshot",
    preludes: &[],
    body: include_str!("kernels/snapshot.wgsl"),
};

/// Bearing tolerance validation.
pub(crate) const BEARINGS: Kernel = Kernel {
    name: "bearings",
    preludes: &[TICK_CONFIG, BEARING, LINK_STATE],
    body: include_str!("kernels/bearings.wgsl"),
};

/// Reduced-coordinate forward kinematics.
pub(crate) const MECHANISM_KERNEL: Kernel = Kernel {
    name: "mechanism",
    preludes: &[TICK_CONFIG, BEARING, MECHANISM, LINK_STATE],
    body: include_str!("kernels/mechanism.wgsl"),
};

/// Articulated velocity projection and drives.
pub(crate) const ARTICULATED: Kernel = Kernel {
    name: "articulated",
    preludes: &[TICK_CONFIG, MASS, BEARING, MECHANISM, DRIVE],
    body: include_str!("kernels/articulated.wgsl"),
};

/// Closed-loop position correction.
pub(crate) const CLOSURE: Kernel = Kernel {
    name: "closure",
    preludes: &[TICK_CONFIG, BEARING, MECHANISM, LINK_STATE],
    body: include_str!("kernels/closure.wgsl"),
};

/// Morton codes, bitonic sort, and LBVH construction.
pub(crate) const LBVH: Kernel = Kernel {
    name: "lbvh",
    preludes: &[TICK_CONFIG, COLLIDER],
    body: include_str!("kernels/lbvh.wgsl"),
};

/// Narrowphase, terrain contact, and the projected impulse solver.
pub(crate) const COLLISION: Kernel = Kernel {
    name: "collision",
    preludes: &[TICK_CONFIG, COLLIDER, MASS, BEARING, DRIVE],
    body: include_str!("kernels/collision.wgsl"),
};

/// Every kernel, for validation.
#[cfg(test)]
pub(crate) const ALL: [&Kernel; 8] = [
    &PHYSICS,
    &SNAPSHOT,
    &BEARINGS,
    &MECHANISM_KERNEL,
    &ARTICULATED,
    &CLOSURE,
    &LBVH,
    &COLLISION,
];

impl Kernel {
    /// The complete WGSL module: generated constants, preludes, then the body.
    pub(crate) fn source(&self) -> String {
        let mut source = abi::wgsl_constants();
        for prelude in self.preludes {
            source.push('\n');
            source.push_str(prelude);
        }
        source.push('\n');
        source.push_str(self.body);
        source
    }
}

#[cfg(test)]
mod tests {
    use std::mem::{offset_of, size_of};

    use super::ALL;
    use crate::abi::{
        GpuBearing, GpuCollider, GpuContact, GpuGroundSurface, GpuLinkState, GpuMass,
        GpuMechanismBody, GpuMechanismCoordinate, GpuMechanismDrive, GpuPersistentManifold,
        GpuTickConfig,
    };

    /// Size of a Rust ABI struct and the byte offset of each field, in order.
    type Layout = (usize, Vec<(&'static str, usize)>);

    macro_rules! layout {
        ($rust:ty { $($field:ident),+ }) => {
            (
                size_of::<$rust>(),
                vec![$((stringify!($field), offset_of!($rust, $field))),+],
            )
        };
    }

    fn validated(kernel: &super::Kernel) -> naga::Module {
        let name = kernel.name;
        let module = naga::front::wgsl::parse_str(&kernel.source())
            .unwrap_or_else(|error| panic!("{name} WGSL parses: {error}"));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|error| panic!("{name} WGSL validates: {error:#?}"));
        module
    }

    #[test]
    fn wgsl_kernels_parse_and_validate_without_a_gpu() {
        for kernel in ALL {
            validated(kernel);
        }
    }

    /// Every WGSL struct that mirrors a Rust ABI struct, by its WGSL name.
    #[expect(
        clippy::too_many_lines,
        reason = "one row per ABI field; the table is the whole function"
    )]
    fn rust_layouts() -> Vec<(&'static str, Layout)> {
        vec![
            (
                "TickConfig",
                layout!(GpuTickConfig {
                    body_count,
                    tick_index,
                    snapshot_slot,
                    collider_count,
                    delta_seconds,
                    gravity_y,
                    linear_damping,
                    angular_damping,
                    bearing_count,
                    suppression_count,
                    pair_capacity,
                    flags,
                    hash_capacity,
                    solver_iterations,
                    sort_count,
                    coordinate_count
                }),
            ),
            (
                "Mass",
                layout!(GpuMass {
                    inverse_mass,
                    inverse_inertia_x,
                    inverse_inertia_y,
                    inverse_inertia_z
                }),
            ),
            (
                "Collider",
                layout!(GpuCollider {
                    local_center,
                    local_rotation,
                    half_extents,
                    metadata,
                    surface_response,
                    surface_elasticity,
                    shape
                }),
            ),
            (
                "GroundSurface",
                layout!(GpuGroundSurface {
                    response,
                    elasticity,
                    plane
                }),
            ),
            (
                "Bearing",
                layout!(GpuBearing {
                    local_anchor_a,
                    local_anchor_b,
                    local_axis_a,
                    local_axis_b,
                    suspension,
                    bump_stop,
                    metadata
                }),
            ),
            (
                "Contact",
                layout!(GpuContact {
                    metadata,
                    normal_penetration,
                    arm_a_impulse,
                    arm_b
                }),
            ),
            (
                "PersistentManifold",
                layout!(GpuPersistentManifold {
                    pair_tick,
                    normal_penetration,
                    point_impulse,
                    tangent_rolling_impulses
                }),
            ),
            (
                "MechanismBody",
                layout!(GpuMechanismBody {
                    metadata,
                    traversal,
                    bind_relative_position,
                    bind_relative_rotation
                }),
            ),
            (
                "Coordinate",
                layout!(GpuMechanismCoordinate { position, velocity }),
            ),
            (
                "Drive",
                layout!(GpuMechanismDrive {
                    mode,
                    max_acceleration,
                    max_speed,
                    target_speed,
                    target_angle,
                    min_angle,
                    max_angle,
                    source_a_max_acceleration,
                    source_a_no_load_speed,
                    source_b_max_acceleration,
                    source_b_no_load_speed,
                    padding
                }),
            ),
            (
                "LinkState",
                layout!(GpuLinkState {
                    position,
                    rotation,
                    metadata
                }),
            ),
        ]
    }

    #[test]
    fn wgsl_struct_layouts_match_the_rust_abi() {
        let expected = rust_layouts();
        let mut checked = std::collections::BTreeSet::new();
        for kernel in ALL {
            let module = validated(kernel);
            for (_, ty) in module.types.iter() {
                let (Some(name), naga::TypeInner::Struct { members, span }) = (&ty.name, &ty.inner)
                else {
                    continue;
                };
                let Some((_, (size, fields))) = expected.iter().find(|(wgsl, _)| wgsl == name)
                else {
                    continue;
                };
                let kernel = kernel.name;
                assert_eq!(*span as usize, *size, "{kernel}: size of {name}");
                let actual: Vec<(&str, usize)> = members
                    .iter()
                    .map(|member| (member.name.as_deref().unwrap_or(""), member.offset as usize))
                    .collect();
                assert_eq!(&actual, fields, "{kernel}: fields of {name}");
                checked.insert(name.clone());
            }
        }
        for (name, _) in &expected {
            assert!(checked.contains(*name), "no kernel declares {name}");
        }
    }
}
