# Contact release and return

Initially separating points no longer impose finite support forces. The private
integrator proposes a shorter interval when a released material-point row changes
from positive to negative normal velocity. It reintegrates from the same start;
the proposed fraction is not accepted as an exact event time. Accepted prefixes
still require whole-path geometry validation. Once separated, the ordinary new
contact query handles the return impact, including return to the same triangle.

The retained pre-fix analytic test launches a touching cube upward at 0.1 m/s.
After 1/60 s under gravity it should remain above the floor at y=0.5003041667 m,
with downward velocity -0.0635 m/s. The prior implementation instead supplied a
support impulse and held its velocity near zero at y=0.5008333333 m. The corrected
trajectory matches the analytic solution at every fixed 1/2/4/8 policy, and its
subsequent inelastic return activates once. A second test starts at 0.05 m/s and
verifies release and return within a single external tick at every policy.

All 95 physics tests pass, including sustained supported-car, first cold-car impact,
friction, spring, stop, impulse and rollback cases. The repeated first cold-car hash
remains `273462a3b30cd1ca`; 16 released points now avoid 80 unnecessary finite contact
rows in that experiment. This is an operation count, not a measured speedup.
Workspace Clippy and formatting pass. Hardware and release-app evidence references
the preceding checkpoint: the app builds, and the same 11 GPU plus one UI failures
remain. No new GPU/app implementation changed in this step.

General nonlinear/rotational trajectory reversals are still open. Initial material
rows provide candidate times; they do not certify a dense continuous force trajectory.
Sustained cold-car drop/settling, full release/re-impact coverage, split recovery,
body/body collision, loops, residency, driving and all performance gates remain open.
Continue through [the current plan](../../contact-impact-next.md).
