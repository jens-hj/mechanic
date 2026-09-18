//! Fixed capacities of the GPU runtime. Every buffer is allocated once at these sizes.

/// Maximum number of compound bodies accepted by the milestone runtime.
pub const MAX_BODIES: usize = 131_072;

/// Maximum number of passive bearings accepted by the milestone runtime.
pub const MAX_BEARINGS: usize = 262_144;

/// Maximum uploaded collider rows.
pub const MAX_COLLIDERS: usize = 131_072;

/// Maximum `vec4` slots in the packed convex-shape buffer.
///
/// One shaped piece needs at most eight vertices, twelve face planes, and
/// eighteen edge directions, so this holds a large shaped creation while
/// staying a fixed allocation like every other buffer here.
pub const MAX_CONVEX_SHAPE_SLOTS: usize = 1_048_576;

/// Fixed candidate/contact capacity. Overflow blocks publication.
pub const MAX_CONTACT_PAIRS: usize = 2_097_152;

/// Power-of-two spatial broadphase table capacity.
pub const BROADPHASE_HASH_CAPACITY: usize = 262_144;

/// Number of published snapshots retained entirely on the GPU.
pub const SNAPSHOT_RING_SIZE: usize = 3;
