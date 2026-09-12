# Rejected cold-contact event controls

None of these candidates passed the required 120-tick cold drop. All are removed
from runtime source. They did not increase the 128-trial policy, change physical
residuals, enlarge the 1e-12 m activation window, or accept an unvalidated path.
The retained starting point is the [continuity checkpoint](../2026-09-11-contact-continuity/README.md).

| Control | Required drop | Decision |
| --- | --- | --- |
| Retained impact/release locator | Fails tick 6; 650 total trials across retries | Baseline remains failing |
| Endpoint finite-feature gap interpolation | Fails tick 5; 1,205 trials | Reject |
| Certified actual hull vertices before separated-face clipping | Fails tick 6; 641 trials | Reject |
| Require entry into half of the existing activation window | Fails tick 6; 650 trials | Reject |
| Unclamped interior gap interpolation with every fourth proposal bisected | Fails tick 5; 891 trials | Reject |

The fresh event trace shows roughly twenty reintegrations to locate one wheel
arrival, followed by another wheel arrival, within the final 8-substep retry.
Collider/body identities alternate (7/2, 42/7 and 67/9); this does not establish
that duplicate segments on one body cause the remaining event work. No blanket
same-body contact suppression is justified by this trace.

The interpolation controls evaluate exact finite collider/chunk/triangle features
at the reintegrated endpoint. The search remains a proposal, with the existing
collision and penetration proof required before acceptance. The altered proposals
still worsen completion; these logs are not a reason to relax the physical gates.
The last control also includes the exact midpoint cache subsequently retained in
the [articulated-factor checkpoint](../2026-09-11-articulated-factor/README.md).

Raw logs and archived candidate modules accompany this record. The periodic-gap
archive is explicitly **partial source** (events and sweep modules); it is not a
complete reconstructible build identity. Earlier candidates are in
`candidates.tar.gz`. `files.sha256.json` identifies retained evidence. No hardware,
performance, complete-world, or physical gate is claimed by these controls.
