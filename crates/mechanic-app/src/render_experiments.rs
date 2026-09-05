//! Launch-only, mutually exclusive rendering experiments. Never saved as settings.

use std::sync::OnceLock;

use bevy::prelude::Msaa;

const ENVIRONMENT_VARIABLE: &str = "MECHANIC_RENDER_EXPERIMENT";
static EXPERIMENT: OnceLock<RenderExperiment> = OnceLock::new();

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum RenderExperiment {
    #[default]
    Baseline,
    NoMsaa,
    SimpleTerrain,
}

impl RenderExperiment {
    fn parse(value: Option<&str>) -> Result<Self, String> {
        match value {
            None | Some("baseline") => Ok(Self::Baseline),
            Some("no-msaa") => Ok(Self::NoMsaa),
            Some("simple-terrain") => Ok(Self::SimpleTerrain),
            Some(value) => Err(format!(
                "Invalid {ENVIRONMENT_VARIABLE}={value:?}; expected baseline, no-msaa, or simple-terrain"
            )),
        }
    }

    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Baseline => "Baseline",
            Self::NoMsaa => "No MSAA",
            Self::SimpleTerrain => "Simple terrain",
        }
    }

    pub(crate) const fn msaa(self) -> Msaa {
        match self {
            Self::NoMsaa => Msaa::Off,
            Self::Baseline | Self::SimpleTerrain => Msaa::Sample4,
        }
    }

    pub(crate) const fn terrain_shader(self) -> &'static str {
        match self {
            Self::SimpleTerrain => "shaders/terrain_simple_diagnostic.wgsl",
            Self::Baseline | Self::NoMsaa => "shaders/terrain_material.wgsl",
        }
    }
}

pub(crate) fn current() -> RenderExperiment {
    *EXPERIMENT.get_or_init(|| {
        let value = std::env::var(ENVIRONMENT_VARIABLE);
        let value = match &value {
            Ok(value) => Some(value.as_str()),
            Err(std::env::VarError::NotPresent) => None,
            Err(error) => panic!("Invalid {ENVIRONMENT_VARIABLE}: {error}"),
        };
        RenderExperiment::parse(value).unwrap_or_else(|error| panic!("{error}"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baseline_preserves_normal_rendering() {
        let mode = RenderExperiment::parse(None).unwrap();
        assert_eq!(mode, RenderExperiment::default());
        assert_eq!(mode.msaa(), Msaa::default());
        assert_eq!(mode.terrain_shader(), "shaders/terrain_material.wgsl");
    }

    #[test]
    fn experiments_change_only_the_selected_variable() {
        let baseline = RenderExperiment::Baseline;
        let no_msaa = RenderExperiment::parse(Some("no-msaa")).unwrap();
        assert_eq!(no_msaa.msaa(), Msaa::Off);
        assert_eq!(no_msaa.terrain_shader(), baseline.terrain_shader());
        let terrain = RenderExperiment::parse(Some("simple-terrain")).unwrap();
        assert_eq!(terrain.msaa(), baseline.msaa());
        assert_eq!(
            terrain.terrain_shader(),
            "shaders/terrain_simple_diagnostic.wgsl"
        );
        assert_eq!(RenderExperiment::parse(Some("baseline")).unwrap(), baseline);
    }

    #[test]
    fn invalid_or_combined_experiments_are_rejected() {
        for value in ["", "off", "no-msaa,simple-terrain", "NO-MSAA"] {
            assert!(RenderExperiment::parse(Some(value)).is_err());
        }
    }
}
