use bytemuck::{Pod, Zeroable};

/// Pair generation exceeded its fixed output buffer.
pub const PAIR_OVERFLOW_FLAG: u32 = 1 << 0;

/// A kernel observed NaN, infinity, or an invalid quaternion.
pub const INVALID_NUMERIC_FLAG: u32 = 1 << 1;

/// Bearing closure exceeded the strict anchor or axis tolerance.
pub const CONSTRAINT_NON_CONVERGENCE_FLAG: u32 = 1 << 2;

/// Persistent-manifold table could not retain a generated contact pair.
pub const MANIFOLD_OVERFLOW_FLAG: u32 = 1 << 3;

/// Per-tick uniform shared by physics kernels.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub struct GpuTickConfig {
    /// Number of active compound rows.
    pub body_count: u32,
    /// Monotonic physics tick index.
    pub tick_index: u32,
    /// Snapshot ring destination.
    pub snapshot_slot: u32,
    /// Number of cuboid collider rows.
    pub collider_count: u32,
    /// Fixed delta time in seconds.
    pub delta_seconds: f32,
    /// Gravity in m/s².
    pub gravity_y: f32,
    /// Exponential linear damping factor per tick.
    pub linear_damping: f32,
    /// Exponential angular damping factor per tick.
    pub angular_damping: f32,
    /// Number of passive bearing rows.
    pub bearing_count: u32,
    /// Number of directly connected compound pairs excluded from collision.
    pub suppression_count: u32,
    /// Fixed pair/contact buffer capacity.
    pub pair_capacity: u32,
    /// Bit zero enables the collision pipeline.
    pub flags: u32,
    /// Broadphase hash-table capacity.
    pub hash_capacity: u32,
    /// Projected impulse iteration count.
    pub solver_iterations: u32,
    /// Reserved for aligned ABI growth.
    pub reserved_a: u32,
    /// Reserved for aligned ABI growth.
    pub reserved_b: u32,
}

/// Fixed-size counters and residuals copied to the CPU after each tick.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq, Eq)]
pub struct GpuDiagnostics {
    /// Bitwise OR of terminal kernel failures.
    pub error_flags: u32,
    /// Candidate pairs requested by broadphase.
    pub pair_count: u32,
    /// SAT contacts requested by narrowphase.
    pub contact_count: u32,
    /// Maximum anchor error in micrometres.
    pub max_anchor_micrometers: u32,
    /// Maximum axis error in millionths of a degree.
    pub max_axis_microdegrees: u32,
    /// Contacts requiring a non-zero projected response.
    pub active_contact_count: u32,
    /// Reserved.
    pub reserved_b: u32,
    /// Reserved.
    pub reserved_c: u32,
}

/// Position and unit quaternion stored as two aligned vectors.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub struct GpuTransform {
    /// xyz position; w is unused.
    pub position: [f32; 4],
    /// xyzw unit quaternion.
    pub rotation: [f32; 4],
}

/// Linear and angular velocity stored as two aligned vectors.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub struct GpuVelocity {
    /// xyz linear velocity; w is unused.
    pub linear: [f32; 4],
    /// xyz angular velocity; w is unused.
    pub angular: [f32; 4],
}

/// Inverse mass and inverse inertia rows for one compound.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub struct GpuMass {
    /// inverse mass in x; remaining lanes are reserved.
    pub inverse_mass: [f32; 4],
    /// First inverse-inertia column.
    pub inverse_inertia_x: [f32; 4],
    /// Second inverse-inertia column.
    pub inverse_inertia_y: [f32; 4],
    /// Third inverse-inertia column.
    pub inverse_inertia_z: [f32; 4],
}

/// Direct mass and body-frame rotational inertia for one compound.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub struct GpuSpatialInertia {
    /// mass in x; remaining lanes are reserved.
    pub mass: [f32; 4],
    /// First rotational-inertia column.
    pub inertia_x: [f32; 4],
    /// Second rotational-inertia column.
    pub inertia_y: [f32; 4],
    /// Third rotational-inertia column.
    pub inertia_z: [f32; 4],
}

/// One cuboid collider row.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub struct GpuCollider {
    /// xyz local centre; w is unused.
    pub local_center: [f32; 4],
    /// xyzw local rotation.
    pub local_rotation: [f32; 4],
    /// xyz half extents; w is unused.
    pub half_extents: [f32; 4],
    /// compound index, source-part slot, source generation, flags.
    pub metadata: [u32; 4],
}

/// One exact passive-bearing row.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub struct GpuBearing {
    /// Anchor in source-compound coordinates.
    pub local_anchor_a: [f32; 4],
    /// Anchor in target-compound coordinates.
    pub local_anchor_b: [f32; 4],
    /// Axis in source-compound coordinates.
    pub local_axis_a: [f32; 4],
    /// Axis in target-compound coordinates.
    pub local_axis_b: [f32; 4],
    /// compound a, compound b, coordinate index or `u32::MAX`, flags.
    pub metadata: [u32; 4],
}

/// Broadphase collider pair.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq, Eq)]
pub struct GpuPair {
    /// First collider row.
    pub collider_a: u32,
    /// Second collider row.
    pub collider_b: u32,
}

/// One reduced cuboid contact consumed by the projected impulse solver.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub struct GpuContact {
    /// body a, body b, persistent-manifold slot, point count.
    pub metadata: [u32; 4],
    /// xyz normal from a to b; w penetration depth.
    pub normal_penetration: [f32; 4],
    /// xyz lever arm from body A; w accumulated normal impulse.
    pub arm_a_impulse: [f32; 4],
    /// xyz lever arm from body B; w is unused.
    pub arm_b: [f32; 4],
}

/// One fixed-capacity contact cache row retained across physics ticks.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub struct GpuPersistentManifold {
    /// Ordered collider pair, last touched tick, point count.
    pub pair_tick: [u32; 4],
    /// xyz cached normal; w penetration depth.
    pub normal_penetration: [f32; 4],
    /// xyz representative point relative to body A; w accumulated normal impulse.
    pub point_impulse: [f32; 4],
}

/// Canonical tree-parent information for one derived mechanism body.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub struct GpuMechanismBody {
    /// parent body, tree bearing row, direction, root flag.
    pub metadata: [u32; 4],
    /// component, depth, preorder position, postorder position.
    pub traversal: [u32; 4],
    /// Initial child position relative to its parent.
    pub bind_relative_position: [f32; 4],
    /// Initial child orientation relative to its parent.
    pub bind_relative_rotation: [f32; 4],
}

/// Authoritative permitted state for one tree bearing.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub struct GpuMechanismCoordinate {
    /// Permitted angle in radians.
    pub angle: f32,
    /// Permitted angular velocity in radians per second.
    pub angular_velocity: f32,
}

/// Temporary transform-to-ancestor row used by parallel pointer jumping.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub struct GpuLinkState {
    /// Position in current ancestor coordinates.
    pub position: [f32; 4],
    /// Orientation in current ancestor coordinates.
    pub rotation: [f32; 4],
    /// Current ancestor body in x; remaining lanes reserved.
    pub metadata: [u32; 4],
}

/// One deterministic leaf-to-root articulated contraction item.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq, Eq)]
pub struct GpuContractionNode {
    /// body, parent body, contraction round, component row.
    pub metadata: [u32; 4],
}

const _: () = {
    assert!(size_of::<GpuTickConfig>() == 64);
    assert!(size_of::<GpuDiagnostics>() == 32);
    assert!(size_of::<GpuTransform>() == 32);
    assert!(size_of::<GpuVelocity>() == 32);
    assert!(size_of::<GpuMass>() == 64);
    assert!(size_of::<GpuSpatialInertia>() == 64);
    assert!(size_of::<GpuCollider>() == 64);
    assert!(size_of::<GpuBearing>() == 80);
    assert!(size_of::<GpuPair>() == 8);
    assert!(size_of::<GpuContact>() == 64);
    assert!(size_of::<GpuPersistentManifold>() == 48);
    assert!(size_of::<GpuMechanismBody>() == 64);
    assert!(size_of::<GpuMechanismCoordinate>() == 8);
    assert!(size_of::<GpuLinkState>() == 48);
    assert!(size_of::<GpuContractionNode>() == 16);
};

#[cfg(test)]
mod tests {
    use super::{
        GpuBearing, GpuCollider, GpuContact, GpuContractionNode, GpuDiagnostics, GpuLinkState,
        GpuMass, GpuMechanismBody, GpuMechanismCoordinate, GpuPair, GpuPersistentManifold,
        GpuSpatialInertia, GpuTickConfig, GpuTransform, GpuVelocity,
    };

    #[test]
    fn gpu_rows_are_sixteen_byte_aligned_in_size() {
        for size in [
            size_of::<GpuTickConfig>(),
            size_of::<GpuDiagnostics>(),
            size_of::<GpuTransform>(),
            size_of::<GpuVelocity>(),
            size_of::<GpuMass>(),
            size_of::<GpuSpatialInertia>(),
            size_of::<GpuCollider>(),
            size_of::<GpuBearing>(),
            size_of::<GpuContact>(),
            size_of::<GpuPersistentManifold>(),
            size_of::<GpuMechanismBody>(),
            size_of::<GpuLinkState>(),
            size_of::<GpuContractionNode>(),
        ] {
            assert_eq!(size % 16, 0);
        }
        assert_eq!(size_of::<GpuPair>(), 8);
        assert_eq!(size_of::<GpuMechanismCoordinate>(), 8);
    }
}
