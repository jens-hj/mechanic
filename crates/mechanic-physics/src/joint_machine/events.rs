//! Bounded first-impact trials; only validated prefixes advance physical time.

use super::{
    CompiledCreation, CoordinateDrive, DVec3, JointTickDiagnostics, JointTickSettings,
    MachineDynamics, MachineState, PassiveForce, PhysicsError, SubstepContacts, TerrainSubstep,
    impact, integrate_substep,
};
use crate::{TerrainSweepHit, TerrainSweepOutcome, terrain_contacts::CONTACT_ACTIVATION_DISTANCE};

#[derive(Clone, Debug)]
pub(super) enum TrialOutcome {
    Complete {
        arrived: bool,
        drive_impulses: Vec<(usize, f64)>,
    },
    Refine(TerrainSweepHit),
    Release(VelocityReversal),
    JointStop(super::stops::StopHit),
}

// One initial material-point or joint row, kept only while reintegrating from
// the same start. Its zero velocity is a release-turning event, not a collision.
#[derive(Clone, Debug)]
pub(super) struct VelocityReversal {
    pub fraction: f64,
    pub row: Vec<f64>,
    pub point: Option<(usize, DVec3, DVec3)>,
}

#[derive(Clone, Copy, Debug)]
pub(super) enum PathOutcome {
    Clear,
    Refine(TerrainSweepHit),
    Activate(TerrainSweepHit),
}

// One completely integrated and independently geometry-validated prefix from
// the unchanged interval start. It can be committed if reintegration disproves
// the event bracket's ordering; a forecast is never substituted for this state.
struct ClearPrefix {
    duration: f64,
    state: MachineState,
    drive_impulses: Vec<(usize, f64)>,
}

#[allow(clippy::too_many_arguments)] // Explicit immutable terrain lifetime and transactional candidate.
pub(super) fn advance_interval(
    creation: &CompiledCreation,
    passive: &[PassiveForce],
    drives: &[CoordinateDrive],
    state: &mut MachineState,
    gravity: DVec3,
    duration: f64,
    settings: JointTickSettings,
    terrain: Option<&TerrainSubstep<'_>>,
    diagnostics: &mut JointTickDiagnostics,
) -> Result<(), PhysicsError> {
    advance_interval_cached::<true>(
        creation,
        passive,
        drives,
        state,
        gravity,
        duration,
        settings,
        terrain,
        diagnostics,
    )
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines, clippy::float_cmp)] // Exact remaining-time sentinel; approximate equality could discard time.
pub(super) fn advance_interval_cached<const REUSE: bool>(
    creation: &CompiledCreation,
    passive: &[PassiveForce],
    drives: &[CoordinateDrive],
    state: &mut MachineState,
    gravity: DVec3,
    duration: f64,
    settings: JointTickSettings,
    terrain: Option<&TerrainSubstep<'_>>,
    diagnostics: &mut JointTickDiagnostics,
) -> Result<(), PhysicsError> {
    super::recovery::correct(
        creation,
        drives,
        state,
        settings,
        terrain.filter(|terrain| {
            terrain.integration == super::TerrainIntegration::SlowContactsAtEndpoint
        }),
        diagnostics,
    )?;
    // The borrowed topology, terrain, material generations and origin cannot
    // change during this call. Only accepted state changes invalidate this cache.
    // Numerical factors and velocity-dependent contacts still refresh per trial.
    let mut start_contacts = None;
    let mut start_model = None;
    let mut elapsed = 0.0;
    let mut trial_duration = duration;
    // These times only guide reintegration from the same unchanged state. They
    // do not certify an ODE trajectory or authorize accepting a crossing path.
    let mut impact_bracket: Option<(f64, f64, TerrainSweepHit)> = None;
    let mut clear_prefix: Option<ClearPrefix> = None;
    let mut release_bracket: Option<(f64, f64, VelocityReversal)> = None;
    // Fixed work bound, independent of frame time. Exhaustion is never permission
    // to discard the remainder or publish a partially integrated external tick.
    for _ in 0..terrain.map_or(128, |t| t.maximum_event_trials) {
        diagnostics.event_trials += 1;
        let mut candidate = state.clone();
        if !REUSE {
            start_contacts = None;
            start_model = None;
        }
        if let Some(terrain) = terrain {
            if start_contacts.is_some() {
                diagnostics.terrain_contact_cache_hits += 1;
            } else {
                start_contacts = Some(terrain.contacts(&candidate, diagnostics)?);
            }
        }
        if start_model.is_some() {
            diagnostics.initial_dynamics_cache_hits += 1;
        } else {
            start_model = Some(MachineDynamics::assemble(
                creation,
                &candidate.poses,
                &candidate.coordinates,
            )?);
            diagnostics.dynamics_assemblies += 1;
        }
        let mut outcome = integrate_substep(
            creation,
            passive,
            drives,
            &mut candidate,
            start_model.as_ref().ok_or(PhysicsError::InvalidDynamics)?,
            gravity,
            trial_duration,
            settings,
            start_contacts
                .as_ref()
                .zip(terrain)
                .map(|(query, terrain)| SubstepContacts {
                    query,
                    restitution_threshold: terrain.restitution_threshold,
                    stiction_threshold: terrain.stiction_threshold,
                }),
            terrain,
            diagnostics,
        )?;
        if let TrialOutcome::Refine(hit) = &outcome
            && let Some(prefix) = &clear_prefix
            && trial_duration * hit.fraction <= prefix.duration
        {
            // Force-dependent reintegration need not produce nested geometric
            // paths. An earlier interior grazing prediction cannot bracket an
            // endpoint root against this already clear prefix.
            diagnostics.event_refinements += 1;
            diagnostics.event_prefix_commits += 1;
            let hit = *hit;
            let prefix = clear_prefix.take().ok_or(PhysicsError::InvalidDynamics)?;
            candidate = prefix.state;
            trial_duration = prefix.duration;
            // The prefix is accepted because the event lies at or before its end,
            // so its endpoint is where the contact is. Dropping the arrival here
            // leaves the pair touching, excluded from the next trial's new-impact
            // search as an initial support, and never resolved at all.
            let arrived = activate_clear_endpoint(
                creation,
                drives,
                &mut candidate,
                terrain.ok_or(PhysicsError::InvalidCollision)?,
                hit,
                settings,
                diagnostics,
            )?;
            outcome = TrialOutcome::Complete {
                arrived,
                drive_impulses: prefix.drive_impulses,
            };
            impact_bracket = None;
            release_bracket = None;
        }
        trace_trial(
            diagnostics.event_trials,
            elapsed,
            trial_duration,
            &outcome,
            impact_bracket
                .as_ref()
                .map(|(lower, upper, _)| (*lower, *upper)),
            release_bracket
                .as_ref()
                .map(|(lower, upper, _)| (*lower, *upper)),
            clear_prefix.as_ref().map(|prefix| prefix.duration),
        );
        match outcome {
            TrialOutcome::Complete {
                arrived,
                drive_impulses,
            } => {
                if let Some((_, upper, hit)) = impact_bracket
                    && !arrived
                    && !activate_clear_endpoint(
                        creation,
                        drives,
                        &mut candidate,
                        terrain.ok_or(PhysicsError::InvalidCollision)?,
                        hit,
                        settings,
                        diagnostics,
                    )?
                {
                    // A shortened force integration can finish before the event
                    // predicted by the longer trial. Locate arrival from this
                    // same start instead of committing arbitrarily tiny prefixes.
                    let next_trial = trial_duration + (upper - trial_duration) * 0.5;
                    if next_trial <= trial_duration || next_trial >= upper {
                        break;
                    }
                    diagnostics.event_localizations += 1;
                    impact_bracket = Some((trial_duration, upper, hit));
                    clear_prefix = Some(ClearPrefix {
                        duration: trial_duration,
                        state: candidate,
                        drive_impulses,
                    });
                    trial_duration = next_trial;
                    continue;
                }
                if let Some((_, upper, ref hit)) = release_bracket
                    && !arrived
                {
                    let speed = hit.speed(creation, &candidate, diagnostics)?;
                    if !speed.is_finite() {
                        return Err(PhysicsError::InvalidDynamics);
                    }
                    if speed.abs() > settings.tolerance {
                        let next_trial = trial_duration + (upper - trial_duration) * 0.5;
                        if speed < 0.0 || next_trial <= trial_duration || next_trial >= upper {
                            break;
                        }
                        diagnostics.release_localizations += 1;
                        release_bracket = Some((trial_duration, upper, hit.clone()));
                        trial_duration = next_trial;
                        continue;
                    }
                }
                impact_bracket = None;
                clear_prefix = None;
                release_bracket = None;
                let remaining = duration - elapsed;
                let next = if trial_duration == remaining {
                    duration
                } else {
                    elapsed + trial_duration
                };
                if next <= elapsed || next > duration {
                    break;
                }
                if let Some(terrain) = terrain.filter(|terrain| {
                    terrain.integration == super::TerrainIntegration::SlowContactsAtEndpoint
                }) {
                    super::recovery::correct(
                        creation,
                        drives,
                        &mut candidate,
                        settings,
                        Some(terrain),
                        diagnostics,
                    )?;
                    let model = MachineDynamics::assemble(
                        creation,
                        &candidate.poses,
                        &candidate.coordinates,
                    )?;
                    diagnostics.dynamics_assemblies += 1;
                    let query = terrain.contacts(&candidate, diagnostics)?;
                    let constraints = query.impact_constraints(
                        &model,
                        &candidate.velocities,
                        terrain.restitution_threshold,
                        terrain.stiction_threshold,
                    )?;
                    impact::activate(
                        creation,
                        &model,
                        drives,
                        &mut candidate,
                        &constraints,
                        settings,
                        diagnostics,
                    )?;
                }
                *state = candidate;
                start_contacts = None;
                start_model = None;
                for (coordinate, impulse) in drive_impulses {
                    diagnostics.drive_impulses[coordinate] += impulse;
                }
                diagnostics.accepted_intervals += 1;
                diagnostics.accepted_seconds += trial_duration;
                elapsed = next;
                if elapsed == duration {
                    return Ok(());
                }
                trial_duration = duration - elapsed;
            }
            TrialOutcome::JointStop(hit) => {
                impact_bracket = None;
                clear_prefix = None;
                release_bracket = None;
                diagnostics.joint_stop_refinements += 1;
                diagnostics.last_joint_stop_coordinate = Some(hit.coordinate);
                let shortened = trial_duration * hit.fraction;
                if shortened <= 0.0 || shortened >= trial_duration || elapsed + shortened <= elapsed
                {
                    break;
                }
                trial_duration = shortened;
            }
            TrialOutcome::Release(hit) => {
                impact_bracket = None;
                clear_prefix = None;
                diagnostics.release_refinements += 1;
                let lower = release_bracket.as_ref().map_or(0.0, |(lower, _, _)| *lower);
                let upper = trial_duration;
                if lower >= upper {
                    break;
                }
                let proposal = trial_duration * hit.fraction;
                let shortened = if release_bracket.is_some() {
                    proposal.clamp(lower + (upper - lower) * 0.1, lower + (upper - lower) * 0.9)
                } else {
                    proposal.min(trial_duration * 0.5)
                };
                if shortened <= lower || shortened >= upper || elapsed + shortened <= elapsed {
                    break;
                }
                release_bracket = Some((lower, upper, hit));
                trial_duration = shortened;
            }
            TrialOutcome::Refine(hit) => {
                release_bracket = None;
                diagnostics.event_refinements += 1;
                let lower = impact_bracket.map_or(0.0, |(lower, _, _)| lower);
                let upper = trial_duration;
                if lower >= upper {
                    break;
                }
                let proposal = trial_duration * hit.fraction;
                let shortened = if impact_bracket.is_some() {
                    // Safeguarded interpolation makes bounded progress even when
                    // force-dependent TOI estimates cling to either endpoint.
                    proposal.clamp(lower + (upper - lower) * 0.1, lower + (upper - lower) * 0.9)
                } else {
                    proposal
                };
                if shortened <= lower || shortened >= upper || elapsed + shortened <= elapsed {
                    break;
                }
                impact_bracket = Some((lower, upper, hit));
                // The long trial's TOI is only a proposed time. Recompute inertia,
                // forces, actuator budgets and stops over the shortened duration.
                trial_duration = shortened;
            }
        }
    }
    diagnostics.terrain_impact_holds += 1;
    diagnostics.failure_stage = Some(super::JointFailureStage::EventSearch);
    Err(PhysicsError::NotConverged)
}

// Per-trial trace for diagnosing a search that never localizes an event, gated
// like `MECHANIC_TRACE_CONTINUATION`. Brackets are the state entering the trial.
#[allow(clippy::too_many_arguments)] // One line per trial needs the whole search state.
fn trace_trial(
    trial: usize,
    elapsed: f64,
    duration: f64,
    outcome: &TrialOutcome,
    impact: Option<(f64, f64)>,
    release: Option<(f64, f64)>,
    prefix: Option<f64>,
) {
    if std::env::var_os("MECHANIC_TRACE_EVENTS").is_none() {
        return;
    }
    let kind = match outcome {
        TrialOutcome::Complete { arrived, .. } => format!("complete arrived={arrived}"),
        TrialOutcome::Refine(hit) => format!(
            "refine fraction={:e} collider={} triangle={} separation={:e}",
            hit.fraction, hit.collider, hit.triangle, hit.separation
        ),
        TrialOutcome::Release(hit) => format!("release fraction={:e}", hit.fraction),
        TrialOutcome::JointStop(hit) => {
            format!(
                "stop coordinate={} fraction={:e}",
                hit.coordinate, hit.fraction
            )
        }
    };
    println!(
        "event trial={trial} elapsed={elapsed:e} duration={duration:e} {kind} impact={impact:?} release={release:?} prefix={prefix:?}"
    );
}

#[allow(clippy::too_many_arguments)] // Preserve the operation boundary in failure diagnostics.
pub(super) fn validate_path(
    creation: &CompiledCreation,
    state: &MachineState,
    dt: f64,
    terrain: Option<&TerrainSubstep<'_>>,
    correction: bool,
    endpoint_rates: Option<(&[f64], &[f64])>,
    diagnostics: &mut JointTickDiagnostics,
) -> Result<PathOutcome, PhysicsError> {
    validate_path_candidate(
        creation,
        state,
        dt,
        terrain,
        correction,
        endpoint_rates,
        diagnostics,
    )
    .inspect_err(|_| {
        diagnostics.failure_stage = Some(super::JointFailureStage::TerrainPath);
    })
}

#[allow(clippy::too_many_lines)] // Keep ordered query accounting and acceptance checks together.
fn validate_path_candidate(
    creation: &CompiledCreation,
    state: &MachineState,
    dt: f64,
    terrain: Option<&TerrainSubstep<'_>>,
    correction: bool,
    endpoint_rates: Option<(&[f64], &[f64])>,
    diagnostics: &mut JointTickDiagnostics,
) -> Result<PathOutcome, PhysicsError> {
    let Some(terrain) = terrain else {
        return Ok(PathOutcome::Clear);
    };
    let displacement = state.velocities.iter().map(|v| v * dt).collect::<Vec<_>>();
    let motion =
        crate::MachineMotion::new(creation, terrain.topology_generation, state, &displacement)?;
    diagnostics.terrain_path_poses += motion.preparation_pose_evaluations();
    diagnostics.terrain_path_queries += 1;
    diagnostics.terrain_correction_paths += usize::from(correction);
    diagnostics.terrain_sweep_queries += 1;
    let endpoint_displacements = endpoint_rates
        .filter(|_| {
            terrain.integration == super::TerrainIntegration::SlowContactsAtEndpoint && !correction
        })
        .map(|(incoming, outgoing)| {
            [
                incoming.iter().map(|v| v * dt).collect::<Vec<_>>(),
                outgoing.iter().map(|v| v * dt).collect::<Vec<_>>(),
            ]
        });
    let endpoints = endpoint_displacements
        .as_ref()
        .map(|rates| {
            Ok::<_, PhysicsError>([
                crate::MachineMotion::new(creation, terrain.topology_generation, state, &rates[0])?,
                crate::MachineMotion::new(creation, terrain.topology_generation, state, &rates[1])?,
            ])
        })
        .transpose()?;
    if let Some(endpoints) = &endpoints {
        diagnostics.terrain_policy_poses += endpoints
            .iter()
            .map(crate::MachineMotion::preparation_pose_evaluations)
            .sum::<usize>();
    }
    let slow = endpoints
        .as_ref()
        .map(|endpoints| crate::terrain_contacts::SlowContactMotion {
            initial: endpoints[0].bounds(),
            final_bounds: endpoints[1].bounds(),
            maximum_displacement: terrain.restitution_threshold * dt,
        });
    let sweep = terrain
        .scene
        .sweep_substep_contacts(
            terrain.geometry,
            &motion,
            terrain.origin,
            CONTACT_ACTIVATION_DISTANCE * 0.25,
            terrain.maximum_evaluations,
            slow,
        )
        .inspect_err(|_| {
            diagnostics.terrain_sweep_query_failures += 1;
            diagnostics.terrain_path_rejections += 1;
        })?;
    diagnostics.terrain_slow_contact_deferrals += sweep.slow_contact_deferrals;
    diagnostics.terrain_path_poses += sweep.pose_evaluations;
    diagnostics.terrain_separation_evaluations += sweep.separation_evaluations;
    diagnostics.terrain_initial_contact_evaluations += sweep.initial_contact_evaluations;
    diagnostics.terrain_linear_sweep_evaluations += sweep.linear_interval_evaluations;
    diagnostics.terrain_quadratic_sweep_evaluations += sweep.quadratic_interval_evaluations;
    diagnostics.terrain_velocity_evaluations += sweep.velocity_evaluations;
    diagnostics.terrain_chunk_candidates += sweep.chunk_candidates;
    diagnostics.terrain_triangle_candidates += sweep.triangle_candidates;
    let outcome = match sweep.outcome {
        TerrainSweepOutcome::Clear => PathOutcome::Clear,
        TerrainSweepOutcome::Unconverged(_) => {
            diagnostics.terrain_path_rejections += 1;
            return Err(PhysicsError::NotConverged);
        }
        TerrainSweepOutcome::Impact(hit) => {
            // Bound every collider's travel after the candidate, not just the
            // first visited collider. Round subtraction and product outward.
            let remaining =
                ((1.0 - hit.fraction).next_up() * sweep.maximum_point_displacement).next_up();
            if remaining > CONTACT_ACTIVATION_DISTANCE {
                diagnostics.terrain_path_rejections += 1;
                return Ok(PathOutcome::Refine(hit));
            }
            let mut endpoint = state.clone();
            endpoint.poses = motion.final_poses().to_vec();
            let query = terrain.contacts(&endpoint, diagnostics)?;
            if !contains_hit(&query, hit) {
                diagnostics.terrain_path_rejections += 1;
                return Ok(PathOutcome::Refine(hit));
            }
            PathOutcome::Activate(hit)
        }
    };
    // Excluding initial supports above finds candidates; it is not a proof of
    // clearance. Every accepted interval, including an impact endpoint, needs
    // this independent bound over the full generalized drift.
    let query = terrain
        .scene
        .validate_penetration(
            terrain.geometry,
            &motion,
            terrain.origin,
            terrain.maximum_depth,
            terrain.maximum_evaluations,
        )
        .inspect_err(|_| {
            diagnostics.terrain_envelope_query_failures += 1;
            diagnostics.terrain_path_rejections += 1;
        })?;
    diagnostics.terrain_path_poses += query.pose_evaluations;
    diagnostics.terrain_pose_cache_hits += query.pose_cache_hits;
    diagnostics.terrain_envelopes += query.envelope_evaluations;
    diagnostics.terrain_certified_intervals += query.certified_intervals;
    diagnostics.terrain_chunk_candidates += query.chunk_candidates;
    diagnostics.terrain_triangle_candidates += query.triangle_candidates;
    if query.outcome != crate::TerrainPathOutcome::Bounded {
        diagnostics.terrain_path_rejections += 1;
        return Err(PhysicsError::NotConverged);
    }
    Ok(outcome)
}

// A fully clear and independently depth-certified path can finish inside the
// fixed numerical-zero contact window without producing a sweep hit. Its actual
// endpoint feature, never the bracket or its time width, authorizes activation.
#[allow(clippy::too_many_arguments)]
fn activate_clear_endpoint(
    creation: &CompiledCreation,
    drives: &[CoordinateDrive],
    state: &mut MachineState,
    terrain: &TerrainSubstep<'_>,
    hit: TerrainSweepHit,
    settings: JointTickSettings,
    diagnostics: &mut JointTickDiagnostics,
) -> Result<bool, PhysicsError> {
    let query = terrain.contacts(state, diagnostics)?;
    if !contains_hit(&query, hit) {
        return Ok(false);
    }
    let model = MachineDynamics::assemble(creation, &state.poses, &state.coordinates)?;
    diagnostics.dynamics_assemblies += 1;
    let constraints = query.impact_constraints(
        &model,
        &state.velocities,
        terrain.restitution_threshold,
        terrain.stiction_threshold,
    )?;
    impact::activate(
        creation,
        &model,
        drives,
        state,
        &constraints,
        settings,
        diagnostics,
    )?;
    Ok(true)
}

fn contains_hit(query: &crate::TerrainContactQuery, hit: TerrainSweepHit) -> bool {
    query.contacts.iter().any(|contact| {
        contact.feature.collider == hit.collider
            && contact.feature.node == hit.node
            && contact.feature.triangle == hit.triangle
            && contact.separation <= CONTACT_ACTIVATION_DISTANCE
    })
}

#[allow(clippy::too_many_arguments)] // Fresh endpoint geometry and incoming velocity; no cached impulse.
pub(super) fn activate_endpoint(
    creation: &CompiledCreation,
    drives: &[CoordinateDrive],
    state: &mut MachineState,
    model: &MachineDynamics,
    terrain: &TerrainSubstep<'_>,
    hit: TerrainSweepHit,
    settings: JointTickSettings,
    diagnostics: &mut JointTickDiagnostics,
) -> Result<(), PhysicsError> {
    // A split joint correction could have changed the endpoint. Require the
    // actual event feature again; never apply stale points to the corrected pose.
    let query = terrain.contacts(state, diagnostics)?;
    if !contains_hit(&query, hit) {
        return Err(PhysicsError::NotConverged);
    }
    let constraints = query.impact_constraints(
        model,
        &state.velocities,
        terrain.restitution_threshold,
        terrain.stiction_threshold,
    )?;
    impact::activate(
        creation,
        model,
        drives,
        state,
        &constraints,
        settings,
        diagnostics,
    )?;
    Ok(())
}

// Released points impose no finite support force. Their initial material-point
// rows also identify a candidate normal-velocity reversal; that shorter interval
// must be reintegrated, not interpolated or accepted as an exact event time.
pub(super) struct SustainingSurface {
    pub query: crate::TerrainContactQuery,
    released: Vec<(VelocityReversal, f64)>,
}

impl SustainingSurface {
    pub fn new(
        query: &crate::TerrainContactQuery,
        model: &MachineDynamics,
        velocity: &[f64],
        tolerance: f64,
    ) -> Result<Self, PhysicsError> {
        let mut result = Self {
            query: query.clone(),
            released: Vec::new(),
        };
        result.query.contacts.clear();
        for point in &query.contacts {
            let row = model.point_row(point.body, point.body_point, point.normal)?;
            let speed = row.iter().zip(velocity).map(|(j, v)| j * v).sum::<f64>();
            if !speed.is_finite() {
                return Err(PhysicsError::InvalidDynamics);
            }
            if speed > tolerance {
                let pose = model.poses[point.body];
                result.released.push((
                    VelocityReversal {
                        fraction: 0.0,
                        row,
                        point: Some((
                            point.body,
                            pose.rotation.inverse() * (point.body_point - pose.position),
                            point.normal,
                        )),
                    },
                    speed,
                ));
            } else {
                result.query.contacts.push(*point);
            }
        }
        Ok(result)
    }

    #[allow(clippy::too_many_arguments)] // Immutable material trajectory and explicit event-speed policy.
    pub fn reversal(
        &self,
        creation: &CompiledCreation,
        initial: &MachineState,
        velocity: &[f64],
        dt: f64,
        tolerance: f64,
        event_speed_threshold: f64,
        diagnostics: &mut JointTickDiagnostics,
    ) -> Result<Option<VelocityReversal>, PhysicsError> {
        if self.released.is_empty() {
            return Ok(None);
        }
        let (poses, motions) = super::contact_kinematics::endpoint_motion(
            creation,
            initial,
            velocity,
            dt,
            diagnostics,
        )?;
        let mut earliest: Option<VelocityReversal> = None;
        for (hit, initial) in &self.released {
            let final_speed = hit.motion_speed(creation, &poses, &motions, velocity);
            if final_speed < -tolerance && initial.max(-final_speed) > event_speed_threshold {
                let fraction = initial / (initial - final_speed);
                if earliest.as_ref().is_none_or(|hit| fraction < hit.fraction) {
                    earliest = Some(VelocityReversal {
                        fraction,
                        ..hit.clone()
                    });
                }
            }
        }
        Ok(earliest)
    }
}

impl VelocityReversal {
    fn motion_speed(
        &self,
        creation: &CompiledCreation,
        poses: &[crate::BodyPose],
        motions: &[crate::SpatialMotion],
        velocity: &[f64],
    ) -> f64 {
        if let Some((body, local, normal)) = self.point {
            super::contact_kinematics::point_speed(creation, poses, motions, body, local, normal)
        } else {
            self.row.iter().zip(velocity).map(|(j, v)| j * v).sum()
        }
    }

    fn speed(
        &self,
        creation: &CompiledCreation,
        state: &MachineState,
        diagnostics: &mut JointTickDiagnostics,
    ) -> Result<f64, PhysicsError> {
        if self.point.is_none() {
            return Ok(self
                .row
                .iter()
                .zip(&state.velocities)
                .map(|(j, v)| j * v)
                .sum());
        }
        let motions =
            MachineDynamics::reconstruct_motions(creation, &state.poses, &state.velocities)?;
        diagnostics.contact_velocity_traversals += 1;
        Ok(self.motion_speed(creation, &state.poses, &motions, &state.velocities))
    }
}
