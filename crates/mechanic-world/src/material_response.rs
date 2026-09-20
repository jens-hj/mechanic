//! Initial game-tuned surface contact properties; these are not measured soil data.

use mechanic_core::{ConstructionMaterial, SurfaceResponse};

use crate::{SoilResponse, TerrainCollisionChunk, TerrainMaterial};

/// Invalid per-vertex material weights or triangle indices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("terrain triangle has invalid material weights")]
pub struct TerrainMaterialError;

impl TerrainCollisionChunk {
    /// Normalized mean of the three vertices' material responses. CPU and GPU
    /// collision use this same law and retain distinct responses during reduction.
    ///
    /// # Errors
    /// Rejects missing, negative, non-finite, or all-zero weights.
    pub fn triangle_surface_response(
        &self,
        indices: [u32; 3],
    ) -> Result<SurfaceResponse, TerrainMaterialError> {
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
        Ok(SurfaceResponse {
            static_friction: response[0],
            dynamic_friction: response[1],
            restitution: response[2],
            rolling_resistance: response[3],
        })
    }
}

impl TerrainCollisionChunk {
    /// Pressure the ground under a triangle carries before it yields, and the
    /// pressure it still resists with once it has, in pascals. The first is its
    /// heaviest material's bearing capacity hardened by the compaction already
    /// there; the second is that capacity undisturbed, because ground that has
    /// failed is broken up, however hard it was packed. Both are infinite for
    /// rock and ore, which never yield.
    ///
    /// # Errors
    /// Rejects missing, negative, non-finite, or all-zero weights.
    pub fn triangle_yield_pa(&self, indices: [u32; 3]) -> Result<[f32; 2], TerrainMaterialError> {
        let mut weights = [0.0_f32; TerrainMaterial::COUNT];
        let mut compaction = 0_u8;
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
            compaction = compaction.max(self.compaction.get(index as usize).copied().unwrap_or(0));
        }
        let material = TerrainMaterial::ALL
            .into_iter()
            .filter(|material| weights[usize::from(material.code())] > 0.0)
            .max_by(|a, b| {
                weights[usize::from(a.code())].total_cmp(&weights[usize::from(b.code())])
            })
            .ok_or(TerrainMaterialError)?;
        let response = SoilResponse::for_material(material);
        Ok([response.capacity_pa(compaction), response.capacity_pa(0)])
    }
}

impl TerrainMaterial {
    /// The construction material this terrain is the same substance as, when it
    /// is one. Such a material answers contact identically as ground and as a
    /// block. Soil, rock, surface cover, and iron ore are terrain of their own:
    /// packed dirt, cut stone, and refined iron are different surfaces.
    pub const fn shared_construction_material(self) -> Option<ConstructionMaterial> {
        match self {
            Self::Sand => Some(ConstructionMaterial::Sand),
            Self::Graphite => Some(ConstructionMaterial::Graphite),
            Self::SurfaceCover | Self::Soil | Self::Rock | Self::Iron => None,
        }
    }

    /// Initial surface tuning. Permanent deformation is a separate response.
    ///
    /// Ground never returns energy: terrain restitution is zero, and a contact
    /// takes the larger restitution of its two surfaces.
    pub const fn surface_response(self) -> SurfaceResponse {
        let (static_friction, dynamic_friction, rolling_resistance) = match self {
            Self::SurfaceCover => (0.8, 0.65, 0.03),
            Self::Soil => (0.7, 0.55, 0.025),
            Self::Rock => (0.8, 0.6, 0.01),
            Self::Iron => (0.6, 0.45, 0.01),
            Self::Sand => Self::shared_friction(ConstructionMaterial::Sand),
            Self::Graphite => Self::shared_friction(ConstructionMaterial::Graphite),
        };
        SurfaceResponse {
            static_friction,
            dynamic_friction,
            restitution: 0.0,
            rolling_resistance,
        }
    }

    /// Static friction, dynamic friction, and rolling resistance of a
    /// construction material this terrain shares its substance with.
    const fn shared_friction(material: ConstructionMaterial) -> (f32, f32, f32) {
        let shared = material.properties();
        (
            shared.static_friction,
            shared.dynamic_friction,
            shared.rolling_resistance,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terrain_sharing_a_construction_material_shares_its_friction() {
        let mut shared = 0;
        for terrain in TerrainMaterial::ALL {
            let Some(construction) = terrain.shared_construction_material() else {
                continue;
            };
            shared += 1;
            let block = construction.properties().surface_response();
            let ground = terrain.surface_response();
            assert_eq!(
                ground.static_friction.to_bits(),
                block.static_friction.to_bits()
            );
            assert_eq!(
                ground.dynamic_friction.to_bits(),
                block.dynamic_friction.to_bits()
            );
            assert_eq!(
                ground.rolling_resistance.to_bits(),
                block.rolling_resistance.to_bits()
            );
            assert_eq!(ground.restitution.to_bits(), 0.0_f32.to_bits());
        }
        assert_eq!(shared, 2);
    }

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
        let rock = TerrainMaterial::Rock.surface_response();
        let graphite = TerrainMaterial::Graphite.surface_response();
        let blend = |rock: f32, graphite: f32| (rock * 2.0 + graphite) / 3.0;
        assert!(
            (response.static_friction - blend(rock.static_friction, graphite.static_friction))
                .abs()
                < 1e-7
        );
        assert!(
            (response.dynamic_friction - blend(rock.dynamic_friction, graphite.dynamic_friction))
                .abs()
                < 1e-7
        );
        assert!(chunk.triangle_surface_response([0, 1, 3]).is_err());
        chunk.material_weights[0][0] = f32::NAN;
        assert!(chunk.triangle_surface_response([0, 1, 2]).is_err());
        chunk.material_weights = vec![[0.0; TerrainMaterial::COUNT]; 3];
        assert!(chunk.triangle_surface_response([0, 1, 2]).is_err());
    }

    #[test]
    fn ground_strength_follows_the_heaviest_material_and_its_compaction() {
        let mut chunk = TerrainCollisionChunk::default();
        for material in [
            TerrainMaterial::Soil,
            TerrainMaterial::Soil,
            TerrainMaterial::Sand,
        ] {
            let mut weights = [0.0; TerrainMaterial::COUNT];
            weights[usize::from(material.code())] = 1.0;
            chunk.material_weights.push(weights);
        }
        let soil = SoilResponse::for_material(TerrainMaterial::Soil);
        let [loose, failed] = chunk.triangle_yield_pa([0, 1, 2]).unwrap();
        assert!((loose - soil.bearing_capacity_pa).abs() < 1.0);
        assert!((failed - soil.bearing_capacity_pa).abs() < 1.0);
        chunk.compaction = vec![0, 255, 0];
        let [packed, failed] = chunk.triangle_yield_pa([0, 1, 2]).unwrap();
        assert!(packed > loose * 10.0);
        assert!((failed - soil.bearing_capacity_pa).abs() < 1.0);
        let mut rock = [0.0; TerrainMaterial::COUNT];
        rock[usize::from(TerrainMaterial::Rock.code())] = 1.0;
        chunk.material_weights = vec![rock; 3];
        assert!(chunk.triangle_yield_pa([0, 1, 2]).unwrap()[1].is_infinite());
        assert!(chunk.triangle_yield_pa([0, 1, 3]).is_err());
    }
}
