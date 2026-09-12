# Active CPU solver repair

The user explicitly requested fixing the CPU solver so it can be used. This work
is ongoing. The application still uses `GpuPhysics`; no complete CPU backend,
driven-car performance gate, or integrated application gate has passed.

The [frozen repair checkpoint](performance-results/2026-09-11-cpu-solver-repair/README.md)
retains source, rejected controls, independent reference evidence and verification.

## Changes under verification

- Monotone contact smoothing no longer proceeds through a failed stage. A bounded
  central-path search follows folds and checks the original unsmoothed contact
  laws. All work consumes the original 256-direction budget. A monotone stage
  currently gets at most 16 directions, reserving work for branch following.
  An unfinished proposal can improve the unpublished iterate using the original
  residual. The unsuccessful additional Newton-polishing experiment was removed.
- Contact response remains implicit above 128 rows. When the mixed search has
  more than 128 equations, streamed rank-revealing orthogonalization retains at
  most 128 directions without storing the larger square matrix. Dense and
  truncated dense diagnostic controls established that the newly captured
  settling impact is solvable; neither diagnostic storage path remains enabled.
- Continuation supports scalar motor/stop/bilateral bounds alongside contact
  disks. The finite-interval finishing equation is evaluated in slack space to
  avoid subtracting a barrier gap smaller than one impulse ULP. Original impulse
  bounds and final velocity residuals remain authoritative.
- Position recovery can reach a new finite contact along a certified correction
  prefix, refresh constraints and continue. Physical velocity and elapsed time
  are unchanged. Recovery includes numerical-zero contacts with their signed gap.
- Numerical-zero vertex activation retains an actual slightly penetrating vertex
  when rounded face clipping returns empty. The same fixed 1e-12 m window and
  finite-triangle containment check apply; deeper vertices are not a fallback.

## Evidence and limits

The third captured impact now passes its previously failing original-law test.
Additional regression inputs preserve recovery at tick 24, a 240-row settling
impact at tick 30, a 65-row impact at tick 36, motor/contact coupling at tick 57,
and an impact at tick 61. These tick numbers refer to successive builds of the
endpoint-policy experiment, not one accepted 120-tick trajectory.

Direct perturbation tests cover contact and scalar derivatives. Streamed QR tests
cover the implicit transpose, redundant equations, minimum norm, and the fixed
rank bound. The core vertex regression and captured recovery case pass. The
latest broad CPU run before the final search changes had 145 passes and three
required trajectory failures. It is not verification of the later source.

The final full CPU run has **148 passes and four failures**. Default cold settling
still fails at tick 17, endpoint integration at tick 30, and fixed-eight endpoint
integration at tick 1. The latter is earlier than the previous checkpoint's
failure. The captured support-transition case remains failing; its independent
NumPy reference passes the original Rust laws. Final response tests: 56 pass /
1 fail. No candidate is ready for application promotion.

The original third-impact NumPy reference and two independently traced NumPy
candidates disagree by approximately 1.7e-7 m/s despite passing original-law
checks. Tighter naive double-precision refinement deteriorates. This evidence
does not establish a unique converged reference velocity; do not silently update
the reference or claim the earlier 1e-8 comparison passed. Original-law regression
checks remain unchanged. Other retained independent reference comparisons pass.

### Support-transition direction controls (rejected)

`/private/tmp/cpu-fix-support-direction-experiment.py` reconstructs the dense
reference smoothing schedule offline (logs: `cpu-fix-support-direction-*.log`).
With a full 240-row direction and a relative rank cutoff of 1e-12 to 1e-10, it
passes the original laws within 51–77 directions under the 16-direction stage
share. Rank falls from 240 to about 202 as supports transition. Tiny cutoffs
fail: pivoted QR at the runtime's absolute `32·ε·max|A|` cutoff reproduces the
Rust stalls at directions 69 and 101, and row-equilibrated SVD at 5.3e-14 stalls
near 1.8e-5. Armijo-only acceptance changes nothing. An adaptive diagonal search
shift stalls near 5.9e-5.

The finding does not transfer to the runtime's Schur-reduced search:

- A 1e-10 relative cutoff in the shared mixed/central QR fails the monotone line
  search at ε=7e-4 and regresses five captured impacts (response suite 51 / 6).
- Continuing unsettled monotone stages alone regresses later-settling,
  post-drive and third-endpoint (53 / 4).
- At ε=7e-6 the mixed search needs 130 retained rows and is rejected by the
  128-row bound before any cutoff applies.

All were reverted. The evidence indicates this impact needs a full-space
rank-revealing direction over roughly 200–240 weak rows.

### Full-space search (explicit storage decision)

The user explicitly approved exceeding the 128-row bound for this search only.
Contact-only machines with at most 256 rows now search over every row:
`I - W/s` is prepared once per continuation, and each direction solves the
row-equilibrated Newton matrix by column-pivoted QR truncated at 1e-11 of the
leading pivot, then takes the minimum-norm step. Peak storage is reported as
`continuation_full_rows` and in `newton_reduced_storage`. The explicit contact
response, mixed search and central path keep the 128-row bound. Original
contact laws alone accept proposals; no tolerance was relaxed.

One unsettled full-space stage continues to the next parameter. A failed
direction or a second consecutive unsettled stage falls back to the central
path. Continuing without that fallback fails later-settling, post-drive and
third-endpoint (third-endpoint fails every offline full-space cutoff).

Results, native test build:

| Control | Captured impacts | Physics suite |
| --- | --- | --- |
| cutoff 1e-10, never fall back | 12 / 3 | — |
| cutoff 1e-10, consecutive | 14 / 1 | — |
| cutoff 1e-11, never fall back | 12 / 3 | — |
| cutoff 1e-11 or 1e-12, consecutive or 5× progress | 15 / 15 | 149 / 3 |

The support-transition impact passes (residual 1.8e-11, 63 directions). The
retained implementation (1e-11, consecutive fallback) passes the complete CPU
run with **152 passes and three failures**, including focused truncation and
Newton-row tests for the full-space direction. Clippy, formatting and the bench
crate tests pass. The cutoff window is narrow: offline, support fails at 1e-13
and later-settling at 1e-10. The fixed-eight tick-1 capture has 19 rows with
four joint-stop blocks, so it uses the central path rather than the full-space
search; the contact-only offline reference cannot evaluate it. All four passing
controls fail the same trajectory ticks: default event
policy at tick 17 (event search), endpoint policy at tick 37 (previously 30,
finite step), and fixed-eight endpoint at tick 1 (impact). The dense search is
slower; the response suite takes about 26 s instead of 5 s in the test build.

### One-sided scalar rows and warm restart

Active joint stops (`[0, ∞)` scalar rows) now use the full-space search as
frictionless normals. Finite intervals, bilateral rows and systems above 256 rows
keep the central path. The captured tick-1 fixed-eight impact
(`fixed_eight_stop_impact.ron`: 15 contact rows, 4 stops) still failed from the
stalled sweep iterate: accepted line-search fractions fell to 3e-5 and the best
original residual stayed at 1.9e-2. From zero impulses every stage settles, with
residual 5.3e-11. A warm first full-space stage that does not settle therefore
restarts once from zero, within the same direction budget.

The response suite passes 62 / 62, but the budget margin narrowed. Compared with
the first full-space run, post-drive takes 253 of 256 directions (was 138),
captured-endpoint 232 (was 128) and settling 223 (was 98). Always starting from
zero produces the same counts.

The fixed-eight drop now advances from tick 1 to tick 11: a 240-row impact with
48 sliding rolling contacts (`fixed_eight_tick11_impact.ron`), runtime residual
6.9e-7. Offline smoothing stalls near 6.6e-6 at cutoffs 1e-11 and 1e-12 and with
dense `lstsq`. The eliminated-barrier refinement seeded from that endpoint
satisfies its own equation (9e-16) but not the original laws: residual 0.29,
disk error 40. That is a spurious point, not evidence of solvability. Tick 11
remains open.

### Drive intervals and trajectory progress

Finite scalar intervals (drive rows) now use a two-sided smoothed clamp,
`lo + s(a − lo) − s(a − hi)`, in the full-space search. Bilateral and fixed rows
keep the central path. The tick-37 endpoint-policy force solve
(`endpoint_tick37_force.ron`: 7 static contacts, 6 drive rows) passes its
original laws from cold and warm starts (3.3e-11). With cutoff 1e-11 the
endpoint drop advances from tick 37 to tick 94.

Tick 94 fails an impact (`endpoint_tick94_impact.ron`: 240 rows, 48 sliding
rolling contacts), with runtime residual 1.05e-8 against 1e-9. Offline it passes
with dense `lstsq` (1.9e-11) and with cutoff 1e-12 and 16-direction stages
(3.8e-11), where stages 6–10 stay unsettled but keep improving. It fails at 1e-11
(about 2e-9). In Rust at 1e-12, stages 6 and 7 are unsettled (best 2.9e-8 and
2.4e-8), so the consecutive rule falls back to the central path, which fails.

Rejected control: cutoff 1e-12. The response suite's budgets drop (captured
endpoint 72, settling 79), but the endpoint drop regresses to tick 61 (impact
residual 3.9e-6). The fixed-eight drop is unchanged at tick 11. Reverted to 1e-11.

Tick 17 (default event policy): a test-only trial trace (`MECHANIC_TRACE_EVENTS`)
shows repeated cycles of about nine trials that commit roughly 25 µs clear
prefixes. Each full-interval trial predicts an impact at fraction ~1.5e-4 on
colliders 6, 32 or 41, at gaps of 1e-15 to 2.5e-13 m (inside the 1e-12 m
activation distance). Shortened reintegrations end clear and do not activate
the feature, so 128 trials cover about 0.35 ms. A probe of the endpoint gap
through a 1e-6 m proximity query failed or saturated at the margin and was
removed; the endpoint gap is unmeasured. The activation distance is unchanged.

Rejected control: per-step rank selection. Offline, each direction was computed
at cutoffs 1e-10 to 1e-13 and the Armijo-accepted step with the lowest smoothed
merit kept. It regressed later-settling (3.8e-6, fixed 1e-11 passes) and
late-endpoint (5.4e-9), and still failed tick 94, tick 11, third-endpoint and
post-drive. Greedy merit selection is not a substitute for a convergent step.

Rejected control: Levenberg–Marquardt directions (Nielsen damping, no rank
cutoff). Offline it fails every fixture, including full-rank ones: stop impact
1.4e-6, tick 37 1.2e-4, support transition 1.0e-4, tick 94 9.4e-5, with 16- or
32-direction stages. Damped Gauss–Newton steps do not reach the smoothed branch
within the stage shares.

Before manifold merging, the tree (cutoff 1e-11) ran the complete CPU suite at
155 passes and 5 failures: the three drops (ticks 94, 11 and 17) and the tick-94
and tick-11 captured-impact regressions.

### Merged manifold rolling rows

In all four hard 240-row captures, every contact in a manifold has identical
rolling rows and targets: 96 rolling rows have rank 7, and the whole Jacobian
has rank 15 over 16 coordinates. The per-point disks `|r_i| <= L_i n_i` act on
the same axes, so they add exactly to `|R| <= sum_i L_i n_i`. The smoothing search
(full-space and mixed directions) now merges each duplicated pair
(`continuation/manifold.rs`). It treats the manifold as one coupled point, then
splits R in proportion to `L_i n_i` before the unchanged per-point laws validate
the candidate. The central path keeps the original rows.

Offline (cutoff 1e-11, 16-direction stages) the merge solves tick 94 (1.1e-10)
and every fixture that passed before. Third-endpoint, post-drive, tick 11 and the
new tick-115 capture still fail offline.

In Rust the endpoint drop advances from tick 94 to tick 115: an impact
(`endpoint_tick115_impact.ron`) at 1.9e-5, which offline merged search also fails
at 1e-11, 1e-12 and with `lstsq`. Late-endpoint regresses its independent
reference comparison. Stages 6–7 end unsettled at best 1.75e-9, the consecutive
rule falls back to the central path, and the accepted solution passes the
original laws (4.2e-12), but its velocity differs from the NumPy reference by
5.9e-8 (limit 1e-8). The reference is not proven unique (see the third-impact
note above). The comparison stays failing rather than relaxed.

Rejected fallback controls, both with merging:

- Continuing unsettled stages while the best residual falls by √10 per stage
  restores late-endpoint (2e-9) but fails post-drive (3.6e-9) and regresses the
  endpoint drop to tick 61.
- The same rule capped at three unsettled stages also fails post-drive and
  regresses the drop to tick 46.

The consecutive rule is kept because it reaches tick 115.

Verified tree: `cargo test -p mechanic-physics --offline -- --test-threads=1`
gives 157 passes and 8 failures: the drops (endpoint policy tick 115, fixed-eight
tick 11, event policy tick 17), the tick-94, tick-115 and tick-11 captured-impact
regressions, late-endpoint's reference comparison, and the retained tick-17
replay. Workspace Clippy with `-D warnings` and `cargo fmt --check` pass.

### Tick-17 event search

`event_policy_tick17_search_completes_from_captured_state` replays tick 17 from
the captured tick-16 state (`car_tick17_event_state.ron`) in about 1.5 s and
reproduces the full drop's counters exactly: 513 trials, 26 prefix commits, 205
refinements, 150 localizations. The test-only event trace measures the hit
feature (same collider, chunk and triangle) at each clear endpoint that fails
to activate. Across 150 misses on colliders 6, 32 and 41, the true gap is
0.79–1.0 mm, even though the longer trial's sweep reported contact at 1e-16 to
1e-13 m. A trial's sweep follows its own midpoint velocity, so a shortened
reintegration covers much less distance than the longer path predicted, and
localization stalls. A millimetre-scale gap rules out a micrometre contact skin
as the fix.

Rejected control: shortening refinements to `D·sqrt(f)` (arrival from rest under
constant acceleration) instead of `D·f`. Tick 17 still exhausts 513 trials. The
endpoint drop regresses from tick 115 to tick 26. The fixed-eight drop moves from
tick 11 to tick 56 (finite step). Four existing terrain-tick tests fail their
refinement and prefix-commit assertions. Reverted.

Rejected control: Newton localization from the clear endpoint (the event
feature's measured gap over its closing normal speed, else bisection). Every
estimate lies far outside the bracket (0.03 s to about 3000 s, median 0.44 s),
so the tick-17 replay is unchanged at 513 trials. The drops regress: event policy
to tick 5, endpoint policy to tick 67. The joint-machine tests still pass.
Reverted. The estimates show the event feature 0.8 mm away and closing at about
2 mm/s at the clear endpoint, while the longer trial's sweep reported contact
0.19 µs in. That points to a non-physical sweep hit, such as a separating-axis
contact across an internal terrain edge, rather than an arrival the search fails
to localize.

Rejected control: a 1 µm support skin. The collider probe shows the settled car's
lowest point 6e-8 m above the floor; the 0.8 mm figure above is the proximity
query's interior sample, not the true minimum. No support therefore lies within
the 1e-12 m activation window, and the event search chases an arrival five orders
of magnitude below its own motion scale. Retaining supports within a 1 µm skin
(window-parameterised activation query for supports, skin-aware sweep exclusion,
skin bound in impact constraints, impacts still activated at 1e-12 m) passes the
tick-17 replay in one trial and the complete endpoint-policy 120-tick drop (hash
`5fa110893c58cf1f`, maximum depth 0, settling 0), and advances the other drops to
tick 46 and tick 40. It is still rejected: a hard support at a positive gap
contradicts the documented invariant that a positive gap needs continuous
activation and cannot silently become an instantaneous supporting contact, it
holds the car 60 nm off the surface instead of settling, and it fails seven
impact-semantics tests (first-impact restitution timing, separating-support
release, return-and-impact within one tick, car contact reversal, car recovery,
supported ticks). Making those rows speculative instead, with a `gap/dt` closing
allowance, regresses the drops further (event policy tick 3, endpoint tick 70) and
fails ten tests.

Rejected control (third variant): velocity-aware retention with identity
exclusion. A skin contact is retained as a support only when it cannot cross the
surface within its step (`-speed <= gap/dt`), and the new-impact sweep excludes
exactly the pairs the solver already accounts for — retained supports plus
released separating points — instead of testing geometry itself. This removes the
chase: the tick-17 replay passes in one trial, against 513 before. The drops still
regress: default tick 17 (a different state, reached through changed earlier
ticks), endpoint tick 19, fixed-eight tick 14, with six failing semantic tests
(the four first-impact cases, car contact reversal, supported ticks).

All three variants cure the search and none preserves first-impact semantics. The
four `first_impacts` tests, `car_contact_reversal` and `supported_ticks` encode the
1e-12 activation model itself: restitution timing, separating-support release, and
return-and-impact within one tick. A support skin therefore needs a contact-model
decision with its own physical validation, not a search fix. Without one, the
intermediate diagnosis stands: the chase is caused by settled supports sitting
tens of nanometres outside the activation window, and any fix must keep impact
activation unchanged while retaining those supports.

`event_policy_tick17_search_completes_from_captured_state` is retained as a
failing regression input: it reproduces the tick-17 search in 1.5 s.

### Analytic contact semantics

`joint_machine/tests/terrain_ticks/contact_semantics.rs` states the intended
physics for one free box on the flat floor, independent of the saved car. Both the
block material and Rock have zero restitution, so the expectations are exact.

The box is the default steel block, whose restitution is 0.2; Rock's surface
restitution is zero and a contact mixes them by taking the larger value.
Restitution applies only above the policy's 1 m/s threshold. All five cases pass:

- An impact above the threshold rebounds at exactly 0.2 of its incoming speed
  (0.2039 m/s from 1.0194 m/s), and the rest of the tick is ballistic.
- An impact below the threshold stops dead at 1, 2, 4 and 8 substeps, never
  rebounding and never penetrating beyond 1e-6 m.
- A settled box stays within 1e-9 m of the surface for 30 ticks and never
  exhausts its event trials, so the car's 60 nm resting gap is not a general
  integration defect; the drift hypothesis does not hold for a single box.
- A separating box keeps ballistic motion to within 1e-9.
- A box that returns inside one tick stops on the surface.

An earlier draft of these tests asserted zero restitution — it took the terrain
surface value and overlooked the block material — and reported the correct 0.2
rebound as a defect. There is no such defect. Single-box contact semantics,
including both sides of the restitution threshold, are sound, so the car's
remaining failures belong to its multi-contact and suspension specifics: the
60 nm resting gap and the 240-row impacts, not basic impact handling.

Captures: `/private/tmp/cpu-fix-tick37-state.ron`, `cpu-fix-tick17-state.ron`,
`cpu-fix-endpoint-next-{impact,force,state}.ron` (tick 94) and
`cpu-fix-endpoint-r12-{impact,force,state}.ron` (tick 61 under the rejected
cutoff).

## Required continuation

1. The support-transition search failure is resolved by the full-space search.
   Investigate the fixed-eight tick-1 impact rejection (captured at
   `/private/tmp/cpu-fix-fixed8-tick1-impact.ron`), the default policy's tick-17
   event search and the endpoint policy's tick-37 finite step. Preserve each new rejection. Test both cold and warm
   constraint starts; retain bounded failed publication. Failed monotone-stage,
   zero-start, Newton-polishing, and iterative-refinement controls are archived;
   repeating them without a new explanation is not progress.
2. Complete 120 ticks, repeated hashes, independent maximum/settling penetration,
   and fixed 1/2/4/8 comparisons. The default event policy's grazing refinement
   failure remains open; do not hide it by changing its test expectations.
3. Review the new matrix-free solve's storage and operation counters, improve its
   cost, and rerun all CPU/core tests, Clippy, formatting and serial native GPU
   checks. No unavailable adapter counts as a pass.
4. Freeze implementation and rejected-control evidence, update the master status,
   then continue shared world commands/publication and real authored driving.
   World integration, body/body collisions, loops and application CPU selection
   remain outstanding.

Current investigation logs use `/private/tmp/cpu-fix-*`. Preserve them in a dated
performance-results checkpoint before cleanup. The test-only
`MECHANIC_IMPACT_CAPTURE` and `MECHANIC_FORCE_CAPTURE` output exact failed algebra
inputs when explicitly set; they do not supply guesses or alter runtime physics.
