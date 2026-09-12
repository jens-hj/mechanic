//! Initial game-tuned surface contact properties; these are not measured soil data.

use crate::{TerrainCollisionChunk, TerrainMaterial};

/// Invalid per-vertex material weights or triangle indices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("terrain triangle has invalid material weights")]
pub struct TerrainMaterialError;

/// Surface response supplied to construction/terrain contacts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerrainSurfaceResponse {
    /// Static Coulomb friction coefficient.
    pub static_friction: f32,
    /// Sliding Coulomb friction coefficient.
    pub dynamic_friction: f32,
    /// Normal coefficient of restitution.
    pub restitution: f32,
    /// Rolling resistance coefficient.
    pub rolling_resistance: f32,
}

impl TerrainSurfaceResponse {
    /// Static friction, dynamic friction, restitution, and rolling resistance.
    pub const fn to_array(self) -> [f32; 4] {
        [
            self.static_friction,
            self.dynamic_friction,
            self.restitution,
            self.rolling_resistance,
        ]
    }
}

impl TerrainCollisionChunk {
    /// Normalized mean of the three vertices' material responses. CPU and GPU
    /// collision use this same law and retain distinct responses during reduction.
    ///
    /// # Errors
    /// Rejects missing, negative, non-finite, or all-zero weights.
    pub fn triangle_surface_response(
        &self,
        indices: [u32; 3],
    ) -> Result<TerrainSurfaceResponse, TerrainMaterialError> {
        let mut weights = [0.0; TerrainMaterial::COUNT];
        for index in indices {
            let source = self
                .material_weights
                .get(index as usize)
                .ok_or(TerrainMaterialError)?;
            for (weight, &source) in weights.iter_mut().zip(source) {
                if !source.is_finite() || source < 0.0 {
                    return Err(TerrainMaterialError);
                }
                *weight += source;
            }
        }
        let total: f32 = weights.iter().sum();
        if total <= 0.0 || !total.is_finite() {
            return Err(TerrainMaterialError);
        }
        let mut response = [0.0; 4];
        for material in TerrainMaterial::ALL {
            let weight = weights[usize::from(material.code())] / total;
            for (target, value) in response
                .iter_mut()
                .zip(material.surface_response().to_array())
            {
                *target += weight * value;
            }
        }
        Ok(TerrainSurfaceResponse {
            static_friction: response[0],
            dynamic_friction: response[1],
            restitution: response[2],
            rolling_resistance: response[3],
        })
    }
}

impl TerrainMaterial {
    /// Initial surface tuning. Permanent deformation is a separate response.
    pub const fn surface_response(self) -> TerrainSurfaceResponse {
        let (static_friction, dynamic_friction, rolling_resistance) = match self {
            Self::SurfaceCover => (0.8, 0.65, 0.03),
            Self::Soil => (0.7, 0.55, 0.025),
            Self::Sand => (0.6, 0.5, 0.04),
            Self::Rock => (0.8, 0.6, 0.01),
            Self::Iron => (0.6, 0.45, 0.01),
            Self::Graphite => (0.3, 0.2, 0.01),
        };
        TerrainSurfaceResponse {
            static_friction,
            dynamic_friction,
            restitution: 0.0,
            rolling_resistance,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triangle_blending_normalizes_weights_and_rejects_invalid_material_data() {
        let mut chunk = TerrainCollisionChunk::default();
        for material in [
            TerrainMaterial::Rock,
            TerrainMaterial::Rock,
            TerrainMaterial::Graphite,
        ] {
            let mut weights = [0.0; TerrainMaterial::COUNT];
            weights[usize::from(material.code())] = 2.0;
            chunk.material_weights.push(weights);
        }
        let response = chunk.triangle_surface_response([0, 1, 2]).unwrap();
        assert!((response.static_friction - (0.8 * 2.0 + 0.3) / 3.0).abs() < 1e-7);
        assert!((response.dynamic_friction - (0.6 * 2.0 + 0.2) / 3.0).abs() < 1e-7);
        assert!(chunk.triangle_surface_response([0, 1, 3]).is_err());
        chunk.material_weights[0][0] = f32::NAN;
        assert!(chunk.triangle_surface_response([0, 1, 2]).is_err());
        chunk.material_weights = vec![[0.0; TerrainMaterial::COUNT]; 3];
        assert!(chunk.triangle_surface_response([0, 1, 2]).is_err());
    }
}
