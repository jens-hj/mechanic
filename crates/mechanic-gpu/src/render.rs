//! Atomic persistent terrain render-arena allocation.

use std::{collections::BTreeMap, ops::Range};

use mechanic_world::TerrainNodeId;
use thiserror::Error;

/// Capacity of the persistent terrain render buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TerrainRenderArenaLimits {
    /// Vertex and vertex-attribute rows.
    pub vertices: usize,
    /// Index rows.
    pub indices: usize,
    /// Chunk bounds and indirect draw commands.
    pub chunks: usize,
}

/// One complete chunk replacement in a terrain render publication.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TerrainRenderChunk {
    /// Stable streamed-node identity.
    pub id: TerrainNodeId,
    /// Exact mesh generation.
    pub generation: u64,
    /// Required persistent vertex rows.
    pub vertex_count: usize,
    /// Required persistent index rows.
    pub index_count: usize,
}

/// Generation-exact terrain render update.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TerrainRenderDelta {
    /// Exact coordinator/publication generation.
    pub generation: u64,
    /// Complete replacement chunks.
    pub replacements: Vec<TerrainRenderChunk>,
    /// Chunk identities no longer visible in this generation.
    pub removals: Vec<TerrainNodeId>,
}

/// Buffer ranges made dirty by one accepted publication.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TerrainRenderDirtyRanges {
    /// Vertex/attribute buffer range.
    pub vertices: Range<usize>,
    /// Index buffer range.
    pub indices: Range<usize>,
}

/// Exact acknowledgement returned only after an atomic visible cutover.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TerrainRenderAcknowledgement {
    /// Accepted coordinator/publication generation.
    pub generation: u64,
    /// Generation now read by indirect draw commands.
    pub visible_generation: u64,
    /// Number of potentially visible indirect commands before frustum culling.
    pub command_count: usize,
    /// Ranges requiring upload for this cutover.
    pub dirty_ranges: Vec<TerrainRenderDirtyRanges>,
}

/// Explicit render-arena failure; callers must retain the preceding complete cut.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum TerrainRenderError {
    /// Indexed multi-draw-indirect is required by the terrain phase.
    #[error("terrain rendering requires indexed multi-draw-indirect support")]
    MissingRequiredFeatures,
    /// The requested generation is not newer than the visible generation.
    #[error("stale terrain render generation {requested}; visible generation is {visible}")]
    StaleGeneration {
        /// Rejected generation.
        requested: u64,
        /// Current visible generation.
        visible: u64,
    },
    /// Replacement allocation could not coexist with the preceding visible cut.
    #[error("terrain render arena exhausted while allocating {kind}")]
    AllocatorExhausted {
        /// Exhausted buffer family.
        kind: &'static str,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TerrainRenderAllocation {
    chunk: TerrainRenderChunk,
    vertices: Range<usize>,
    indices: Range<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RetiredAllocation {
    allocation: TerrainRenderAllocation,
    safe_after_frame: u64,
}

/// CPU owner for persistent render-world terrain ranges and atomic command generations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TerrainRenderArena {
    vertex_allocator: RangeAllocator,
    index_allocator: RangeAllocator,
    chunk_capacity: usize,
    active: BTreeMap<TerrainNodeId, TerrainRenderAllocation>,
    retired: Vec<RetiredAllocation>,
    visible_generation: u64,
    safety_window_frames: u64,
}

impl TerrainRenderArena {
    /// Creates an arena only when the mandatory render feature set is available.
    ///
    /// # Errors
    ///
    /// Returns [`TerrainRenderError::MissingRequiredFeatures`] when indexed
    /// multi-draw-indirect is unavailable.
    pub fn new(
        limits: TerrainRenderArenaLimits,
        indexed_multi_draw_indirect: bool,
        safety_window_frames: u64,
    ) -> Result<Self, TerrainRenderError> {
        if !indexed_multi_draw_indirect {
            return Err(TerrainRenderError::MissingRequiredFeatures);
        }
        Ok(Self {
            vertex_allocator: RangeAllocator::new(limits.vertices),
            index_allocator: RangeAllocator::new(limits.indices),
            chunk_capacity: limits.chunks,
            active: BTreeMap::new(),
            retired: Vec::new(),
            visible_generation: 0,
            safety_window_frames,
        })
    }

    /// Allocates every replacement before atomically flipping the visible command generation.
    ///
    /// On error no allocation, command, or visible generation changes.
    ///
    /// # Errors
    ///
    /// Returns [`TerrainRenderError::StaleGeneration`] for obsolete work or
    /// [`TerrainRenderError::AllocatorExhausted`] when replacement ranges
    /// cannot coexist with the current visible cut.
    pub fn publish(
        &mut self,
        delta: TerrainRenderDelta,
        frame: u64,
    ) -> Result<TerrainRenderAcknowledgement, TerrainRenderError> {
        if delta.generation <= self.visible_generation {
            return Err(TerrainRenderError::StaleGeneration {
                requested: delta.generation,
                visible: self.visible_generation,
            });
        }
        let replacement_ids = delta
            .replacements
            .iter()
            .map(|chunk| chunk.id)
            .collect::<std::collections::BTreeSet<_>>();
        let removal_count = delta
            .removals
            .iter()
            .filter(|id| self.active.contains_key(id) && !replacement_ids.contains(id))
            .count();
        let resulting_chunks = self
            .active
            .len()
            .saturating_sub(removal_count)
            .saturating_add(
                replacement_ids
                    .iter()
                    .filter(|id| !self.active.contains_key(id))
                    .count(),
            );
        if resulting_chunks > self.chunk_capacity {
            return Err(TerrainRenderError::AllocatorExhausted { kind: "chunks" });
        }

        let mut vertices = self.vertex_allocator.clone();
        let mut indices = self.index_allocator.clone();
        let mut replacements = Vec::with_capacity(delta.replacements.len());
        for chunk in delta.replacements {
            let vertex_range = vertices
                .allocate(chunk.vertex_count)
                .ok_or(TerrainRenderError::AllocatorExhausted { kind: "vertices" })?;
            let index_range = indices
                .allocate(chunk.index_count)
                .ok_or(TerrainRenderError::AllocatorExhausted { kind: "indices" })?;
            replacements.push(TerrainRenderAllocation {
                chunk,
                vertices: vertex_range,
                indices: index_range,
            });
        }

        self.vertex_allocator = vertices;
        self.index_allocator = indices;
        let safe_after_frame = frame.saturating_add(self.safety_window_frames);
        for id in delta.removals {
            if replacement_ids.contains(&id) {
                continue;
            }
            if let Some(allocation) = self.active.remove(&id) {
                self.retired.push(RetiredAllocation {
                    allocation,
                    safe_after_frame,
                });
            }
        }
        let dirty_ranges = replacements
            .iter()
            .map(|allocation| TerrainRenderDirtyRanges {
                vertices: allocation.vertices.clone(),
                indices: allocation.indices.clone(),
            })
            .collect();
        for allocation in replacements {
            if let Some(previous) = self.active.insert(allocation.chunk.id, allocation) {
                self.retired.push(RetiredAllocation {
                    allocation: previous,
                    safe_after_frame,
                });
            }
        }
        self.visible_generation = delta.generation;
        Ok(TerrainRenderAcknowledgement {
            generation: delta.generation,
            visible_generation: self.visible_generation,
            command_count: self.active.len(),
            dirty_ranges,
        })
    }

    /// Releases predecessor ranges after the render safety window passes.
    pub fn retire_completed(&mut self, frame: u64) -> usize {
        let mut released = 0;
        self.retired.retain(|retired| {
            if retired.safe_after_frame > frame {
                return true;
            }
            self.vertex_allocator
                .release(retired.allocation.vertices.clone());
            self.index_allocator
                .release(retired.allocation.indices.clone());
            released += 1;
            false
        });
        released
    }

    /// Current atomically visible command generation.
    pub const fn visible_generation(&self) -> u64 {
        self.visible_generation
    }

    /// Current complete visible chunk count.
    pub fn active_chunk_count(&self) -> usize {
        self.active.len()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RangeAllocator {
    free: Vec<Range<usize>>,
}

impl RangeAllocator {
    fn new(capacity: usize) -> Self {
        Self {
            free: (capacity > 0).then_some(0..capacity).into_iter().collect(),
        }
    }

    fn allocate(&mut self, length: usize) -> Option<Range<usize>> {
        if length == 0 {
            return Some(0..0);
        }
        let index = self
            .free
            .iter()
            .position(|range| range.end - range.start >= length)?;
        let start = self.free[index].start;
        self.free[index].start += length;
        if self.free[index].is_empty() {
            self.free.remove(index);
        }
        Some(start..start + length)
    }

    fn release(&mut self, range: Range<usize>) {
        if range.is_empty() {
            return;
        }
        self.free.push(range);
        self.free.sort_by_key(|range| range.start);
        let mut merged: Vec<Range<usize>> = Vec::with_capacity(self.free.len());
        for range in self.free.drain(..) {
            if let Some(previous) = merged.last_mut()
                && previous.end == range.start
            {
                previous.end = range.end;
            } else {
                merged.push(range);
            }
        }
        self.free = merged;
    }
}

#[cfg(test)]
mod tests {
    use mechanic_world::BrickCoord;

    use super::*;

    fn chunk(x: i32, generation: u64, vertices: usize, indices: usize) -> TerrainRenderChunk {
        TerrainRenderChunk {
            id: TerrainNodeId::leaf(BrickCoord::new(x, 0, 0)),
            generation,
            vertex_count: vertices,
            index_count: indices,
        }
    }

    #[test]
    fn cutover_is_atomic_when_replacement_cannot_coexist() {
        let mut arena = TerrainRenderArena::new(
            TerrainRenderArenaLimits {
                vertices: 12,
                indices: 12,
                chunks: 2,
            },
            true,
            2,
        )
        .unwrap();
        arena
            .publish(
                TerrainRenderDelta {
                    generation: 1,
                    replacements: vec![chunk(0, 1, 8, 8)],
                    removals: Vec::new(),
                },
                0,
            )
            .unwrap();
        let error = arena
            .publish(
                TerrainRenderDelta {
                    generation: 2,
                    replacements: vec![chunk(0, 2, 8, 8)],
                    removals: Vec::new(),
                },
                1,
            )
            .unwrap_err();
        assert_eq!(
            error,
            TerrainRenderError::AllocatorExhausted { kind: "vertices" }
        );
        assert_eq!(arena.visible_generation(), 1);
        assert_eq!(arena.active_chunk_count(), 1);
    }

    #[test]
    fn predecessor_ranges_retire_only_after_safety_window() {
        let mut arena = TerrainRenderArena::new(
            TerrainRenderArenaLimits {
                vertices: 16,
                indices: 16,
                chunks: 1,
            },
            true,
            2,
        )
        .unwrap();
        arena
            .publish(
                TerrainRenderDelta {
                    generation: 1,
                    replacements: vec![chunk(0, 1, 8, 8)],
                    removals: Vec::new(),
                },
                0,
            )
            .unwrap();
        arena
            .publish(
                TerrainRenderDelta {
                    generation: 2,
                    replacements: vec![chunk(0, 2, 8, 8)],
                    removals: Vec::new(),
                },
                1,
            )
            .unwrap();
        assert_eq!(arena.retire_completed(2), 0);
        assert_eq!(arena.retire_completed(3), 1);
        arena
            .publish(
                TerrainRenderDelta {
                    generation: 3,
                    replacements: vec![chunk(0, 3, 8, 8)],
                    removals: Vec::new(),
                },
                4,
            )
            .expect("retired predecessor space is reusable");
    }

    #[test]
    fn stale_generation_never_becomes_visible() {
        let mut arena = TerrainRenderArena::new(
            TerrainRenderArenaLimits {
                vertices: 4,
                indices: 4,
                chunks: 1,
            },
            true,
            1,
        )
        .unwrap();
        arena
            .publish(
                TerrainRenderDelta {
                    generation: 4,
                    replacements: vec![chunk(0, 4, 2, 2)],
                    removals: Vec::new(),
                },
                0,
            )
            .unwrap();
        assert!(matches!(
            arena.publish(
                TerrainRenderDelta {
                    generation: 3,
                    replacements: Vec::new(),
                    removals: Vec::new(),
                },
                1,
            ),
            Err(TerrainRenderError::StaleGeneration { .. })
        ));
        assert_eq!(arena.visible_generation(), 4);
    }
}
