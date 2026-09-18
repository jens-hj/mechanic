//! What a terrain contact is: obstacles, targets, features, queries, and activation distances.

use bevy_math::DVec3;
use mechanic_world::TerrainNodeId;

/// Geometry opposing a collider at one contact point.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ContactObstacle {
    /// A finite triangle of published terrain.
    Terrain {
        /// Owning terrain node.
        node: TerrainNodeId,
        /// Chunk geometry generation supplied by the world.
        geometry_generation: u64,
        /// Publication that selected this chunk's active groups/materials.
        publication_generation: u64,
        /// Stable triangle row within the immutable BVH.
        triangle: usize,
    },
    /// Another collider row of the same construction, on a different body.
    Collider(usize),
}

/// Geometry a continuous query found approaching, independent of generations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContactTarget {
    /// A terrain triangle.
    Terrain {
        /// Owning terrain chunk.
        node: TerrainNodeId,
        /// Triangle row in the chunk's immutable BVH.
        triangle: usize,
    },
    /// Another collider row of the same construction.
    Collider(usize),
}

impl ContactObstacle {
    /// The approached geometry, without the generations that published it.
    pub fn target(self) -> ContactTarget {
        match self {
            Self::Terrain { node, triangle, .. } => ContactTarget::Terrain { node, triangle },
            Self::Collider(collider) => ContactTarget::Collider(collider),
        }
    }
}

/// Stable source identity; replacing a chunk or topology invalidates its points.
/// A retained corner is refreshed against geometry before any future reuse.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct TerrainContactFeature {
    /// Compiled construction generation.
    pub topology_generation: u64,
    /// Collider row receiving the contact normal.
    pub collider: usize,
    /// Opposing terrain triangle or collider.
    pub obstacle: ContactObstacle,
    /// Retained finite polygon corner, requiring geometry refresh before reuse.
    pub corner: usize,
}

impl TerrainContactFeature {
    /// Whether this point touches the geometry a continuous query reported for
    /// `collider`. A collider pair matches in either order: which of the two
    /// receives the normal depends on the pose, not on the approach.
    pub fn touches(&self, collider: usize, target: ContactTarget) -> bool {
        let own = self.obstacle.target();
        (self.collider == collider && own == target)
            || matches!(
                (own, target),
                (ContactTarget::Collider(other), ContactTarget::Collider(approached))
                    if other == collider && approached == self.collider
            )
    }
}

// Numerical zero for actual finite opposing points, never a speculative margin.
pub(crate) const CONTACT_ACTIVATION_DISTANCE: f64 = 1e-12;

// Numerical zero between two bodies' solids. Their separating-axis gap comes from
// two independently rounded, moving polytopes, and a body rolling over another's
// edge closes its last fraction of a micron far slower than any conservative
// bound, so a terrain-precision window exhausts the event search. The GPU emits
// body contacts from 1e-5 m apart; this stays ten times tighter.
pub(crate) const PAIR_ACTIVATION_DISTANCE: f64 = 1e-6;

impl ContactTarget {
    /// Largest gap at which a contact with this target counts as touching.
    pub(crate) const fn activation_distance(self) -> f64 {
        match self {
            Self::Terrain { .. } => CONTACT_ACTIVATION_DISTANCE,
            Self::Collider(_) => PAIR_ACTIVATION_DISTANCE,
        }
    }
}

/// Actual finite support against terrain or another body, with mixed materials.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerrainContact {
    /// Generations and geometric source of this point.
    pub feature: TerrainContactFeature,
    /// Compound body receiving the contact impulse along `normal`.
    pub body: usize,
    /// Body carrying the opposing surface, which receives the opposite impulse.
    /// None for terrain.
    pub other_body: Option<usize>,
    /// Query-local manifold number within this collider/obstacle, for coupled block solving.
    pub manifold: usize,
    /// Opposing surface point, relative to the query's floating origin.
    pub terrain_point: DVec3,
    /// Receiving convex point, relative to the same origin.
    pub body_point: DVec3,
    /// Opposing surface's outward unit normal, toward the receiving body.
    pub normal: DVec3,
    /// Current penetration along the normal, in metres.
    pub depth: f64,
    /// Signed normal gap between the true opposing points; negative in overlap.
    pub separation: f64,
    /// Static/kinetic friction, restitution, rolling resistance, in that order.
    pub response: [f64; 4],
}

/// Retained manifolds and actual collision query work. No solve is implied.
#[derive(Clone, Debug, Default)]
pub struct TerrainContactQuery {
    /// Contacts sorted by stable source identity.
    pub contacts: Vec<TerrainContact>,
    /// One numerical-zero contact witness per source triangle, before manifold
    /// reduction. Event arrival must not depend on which support corners survive.
    pub activation_features: Vec<TerrainContactFeature>,
    /// Chunk candidates visited across all moving colliders.
    pub chunk_candidates: usize,
    /// Triangle candidates passed to finite narrowphase.
    pub triangle_candidates: usize,
    /// Point count before cross-triangle manifold reduction.
    pub unreduced_points: usize,
    /// Collider pairs on different bodies whose bounds overlap.
    pub collider_pair_candidates: usize,
}
