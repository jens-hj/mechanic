//! Node faces and the transition masks that stitch levels of detail.

use super::selection::face_axis;

/// One of the six axis-aligned node faces.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum TerrainFace {
    /// Negative X.
    NegativeX = 0,
    /// Positive X.
    PositiveX = 1,
    /// Negative Y.
    NegativeY = 2,
    /// Positive Y.
    PositiveY = 3,
    /// Negative Z.
    NegativeZ = 4,
    /// Positive Z.
    PositiveZ = 5,
}

impl TerrainFace {
    /// Stable face order used by masks and geometry arrays.
    pub const ALL: [Self; 6] = [
        Self::NegativeX,
        Self::PositiveX,
        Self::NegativeY,
        Self::PositiveY,
        Self::NegativeZ,
        Self::PositiveZ,
    ];

    /// Array index.
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Opposite face.
    #[must_use]
    pub const fn opposite(self) -> Self {
        match self {
            Self::NegativeX => Self::PositiveX,
            Self::PositiveX => Self::NegativeX,
            Self::NegativeY => Self::PositiveY,
            Self::PositiveY => Self::NegativeY,
            Self::NegativeZ => Self::PositiveZ,
            Self::PositiveZ => Self::NegativeZ,
        }
    }
}

/// Faces bordering a coarser node plus boundary features that must share its
/// nested scalar samples with neighboring fine chunks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TerrainTransitionMask(pub(super) u32);

pub(super) const BOUNDARY_FEATURES: [u8; 26] = [
    0b00_0001, 0b00_0010, 0b00_0100, 0b00_1000, 0b01_0000, 0b10_0000, 0b00_0101, 0b00_1001,
    0b00_0110, 0b00_1010, 0b01_0001, 0b10_0001, 0b01_0010, 0b10_0010, 0b01_0100, 0b10_0100,
    0b01_1000, 0b10_1000, 0b01_0101, 0b10_0101, 0b01_1001, 0b10_1001, 0b01_0110, 0b10_0110,
    0b01_1010, 0b10_1010,
];

impl TerrainTransitionMask {
    /// Empty mask.
    pub const NONE: Self = Self(0);

    /// Creates a validated six-face mask.
    pub fn from_bits(bits: u8) -> Self {
        let mut mask = Self(u32::from(bits & 0x3f));
        for face in TerrainFace::ALL {
            if mask.contains(face) {
                mask.insert_face_boundary_features(face);
            }
        }
        mask
    }

    /// Raw six bits.
    pub const fn bits(self) -> u8 {
        (self.0 & 0x3f) as u8
    }

    /// True when `face` needs a transition cell.
    pub const fn contains(self, face: TerrainFace) -> bool {
        self.0 & (1 << face as u8) != 0
    }

    pub(super) fn insert(&mut self, face: TerrainFace) {
        self.0 |= 1 << face as u8;
    }

    pub(super) fn insert_boundary_feature(&mut self, faces: u8) {
        if let Some(index) = BOUNDARY_FEATURES
            .iter()
            .position(|&feature| feature == faces)
        {
            self.0 |= 1 << (6 + index);
        }
    }

    pub(super) fn insert_face_boundary_features(&mut self, transition_face: TerrainFace) {
        let transition_bit = 1 << transition_face as u8;
        self.insert_boundary_feature(transition_bit);
        for side in TerrainFace::ALL {
            if face_axis(side) == face_axis(transition_face) {
                continue;
            }
            self.insert_boundary_feature(transition_bit | (1 << side as u8));
        }
        for first in TerrainFace::ALL {
            for second in TerrainFace::ALL {
                if face_axis(first) != face_axis(transition_face)
                    && face_axis(second) != face_axis(transition_face)
                    && face_axis(first) < face_axis(second)
                {
                    self.insert_boundary_feature(
                        transition_bit | (1 << first as u8) | (1 << second as u8),
                    );
                }
            }
        }
    }

    pub(crate) fn synchronizes_boundary_feature(self, faces: u8) -> bool {
        BOUNDARY_FEATURES
            .iter()
            .position(|&feature| feature == faces)
            .is_some_and(|index| self.0 & (1 << (6 + index)) != 0)
    }
}
