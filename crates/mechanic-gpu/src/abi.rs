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

/// Shape kind for a box collider.
pub const COLLIDER_SHAPE_CUBOID: u32 = 0;

/// Shape kind for a convex polytope collider, whose geometry lives in the
/// packed convex-shape buffer.
pub const COLLIDER_SHAPE_CONVEX: u32 = 1;

/// Packs a convex shape's element counts into one lane.
pub const fn pack_convex_counts(vertices: u32, faces: u32, edges: u32) -> u32 {
    vertices | (faces << 8) | (edges << 16)
}

/// One collider row.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub struct GpuCollider {
    /// xyz local centre; w is the segment-centre radius for full-cylinder ground contact.
    pub local_center: [f32; 4],
    /// xyzw local rotation.
    pub local_rotation: [f32; 4],
    /// xyz half extents; w is the visual outer radius for full-cylinder ground contact.
    pub half_extents: [f32; 4],
    /// compound index, source-part slot, source generation, ground-contact role.
    pub metadata: [u32; 4],
    /// friction, restitution, and two reserved lanes.
    pub contact_properties: [f32; 4],
    /// shape kind, offset into the convex-shape buffer, packed element counts,
    /// and one reserved lane. A box ignores every lane but the kind.
    pub shape: [u32; 4],
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

/// Drive parameters for one tree-bearing coordinate.
///
/// A row in [`DRIVE_MODE_PASSIVE`] with infinite limits is exactly a free
/// joint. The buffer must never be left zeroed: zeroed limits would clamp every
/// joint angle to zero.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub struct GpuMechanismDrive {
    /// What the solver does with this coordinate. See `DRIVE_MODE_*`.
    pub mode: u32,
    /// Largest permitted joint-speed change per second. Infinite when the drive
    /// torque is unlimited.
    pub max_acceleration: f32,
    /// Fastest the joint may turn, in radians per second.
    pub max_speed: f32,
    /// Signed target speed in radians per second, used in speed mode.
    pub target_speed: f32,
    /// Target angle in radians, used in angle mode.
    pub target_angle: f32,
    /// Lower angle limit in radians, or negative infinity.
    pub min_angle: f32,
    /// Upper angle limit in radians, or positive infinity.
    pub max_angle: f32,
    /// Stall acceleration supplied by the first actuator family.
    pub source_a_max_acceleration: f32,
    /// No-load speed of the first actuator family.
    pub source_a_no_load_speed: f32,
    /// Stall acceleration supplied by the second actuator family.
    pub source_b_max_acceleration: f32,
    /// No-load speed of the second actuator family.
    pub source_b_no_load_speed: f32,
    /// Padding to a sixteen-byte multiple.
    pub padding: f32,
}

impl From<mechanic_core::CoordinateDrive> for GpuMechanismDrive {
    fn from(drive: mechanic_core::CoordinateDrive) -> Self {
        Self {
            mode: drive.mode.code(),
            max_acceleration: drive.max_acceleration,
            max_speed: drive.max_speed,
            target_speed: drive.target_speed,
            target_angle: drive.target_angle,
            min_angle: drive.min_angle,
            max_angle: drive.max_angle,
            source_a_max_acceleration: drive.source_a_max_acceleration,
            source_a_no_load_speed: drive.source_a_no_load_speed,
            source_b_max_acceleration: drive.source_b_max_acceleration,
            source_b_no_load_speed: drive.source_b_no_load_speed,
            padding: 0.0,
        }
    }
}

/// Coordinate mode for a joint no control block drives.
pub const DRIVE_MODE_PASSIVE: u32 = 0;

/// Coordinate mode for a joint holding a target speed.
pub const DRIVE_MODE_SPEED: u32 = 1;

/// Coordinate mode for a joint seeking a target angle.
pub const DRIVE_MODE_ANGLE: u32 = 2;

impl GpuMechanismDrive {
    /// Row describing a coordinate no control block drives.
    pub const PASSIVE: Self = Self {
        mode: DRIVE_MODE_PASSIVE,
        max_acceleration: 0.0,
        max_speed: 0.0,
        target_speed: 0.0,
        target_angle: 0.0,
        min_angle: f32::NEG_INFINITY,
        max_angle: f32::INFINITY,
        source_a_max_acceleration: 0.0,
        source_a_no_load_speed: 0.0,
        source_b_max_acceleration: 0.0,
        source_b_no_load_speed: 0.0,
        padding: 0.0,
    };
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
    assert!(size_of::<GpuCollider>() == 96);
    assert!(size_of::<GpuBearing>() == 80);
    assert!(size_of::<GpuPair>() == 8);
    assert!(size_of::<GpuContact>() == 64);
    assert!(size_of::<GpuPersistentManifold>() == 48);
    assert!(size_of::<GpuMechanismBody>() == 64);
    assert!(size_of::<GpuMechanismCoordinate>() == 8);
    assert!(size_of::<GpuMechanismDrive>() == 48);
    assert!(size_of::<GpuLinkState>() == 48);
    assert!(size_of::<GpuContractionNode>() == 16);
};

#[cfg(test)]
mod tests {
    use super::{
        DRIVE_MODE_PASSIVE, GpuBearing, GpuCollider, GpuContact, GpuContractionNode,
        GpuDiagnostics, GpuLinkState, GpuMass, GpuMechanismBody, GpuMechanismCoordinate,
        GpuMechanismDrive, GpuPair, GpuPersistentManifold, GpuSpatialInertia, GpuTickConfig,
        GpuTransform, GpuVelocity,
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
            size_of::<GpuMechanismDrive>(),
        ] {
            assert_eq!(size % 16, 0);
        }
        assert_eq!(size_of::<GpuPair>(), 8);
        assert_eq!(size_of::<GpuMechanismCoordinate>(), 8);
    }

    #[test]
    fn passive_drive_rows_never_clamp_a_joint_angle() {
        let passive = GpuMechanismDrive::PASSIVE;
        assert_eq!(passive.mode, DRIVE_MODE_PASSIVE);
        assert!(passive.max_acceleration.abs() < f32::EPSILON);
        assert!(passive.min_angle.is_infinite() && passive.min_angle < 0.0);
        assert!(passive.max_angle.is_infinite() && passive.max_angle > 0.0);
    }
}
