# Articulated CCD and coupled surface response

This is a **passing geometry/fixed-pose checkpoint**, not a complete moving tick
or a performance acceptance run. The application remains on the existing GPU
runtime. Apply `implementation-sources.tar.gz` over the frozen source identified
in `identity.json` to reproduce this checkpoint independently of later work.

The CPU trajectory now reconstructs exact tree poses without assembling inertia.
Conservative bounds include floating-root rotation, every ancestor joint, and
suspension extension. Finite triangle sweeps detect an articulated full-turn bar
impact between clear endpoints (fraction 0.07322018415963456, 15 pose evaluations).
Exhausted advancement returns an explicit uncertified result. The saved car's
sampled point speed / derived bound ratio is at most 0.6165717391975718.

The real car's 48 retained finite terrain points produce 12 coupled manifold
blocks and 240 scalar rows, including circular static/kinetic friction and rolling
resistance. The response remains implicit above 128 rows. The final test passes
the unchanged 256-sweep / 1e-8 physical residual bound:

| Quantity | Result |
| --- | --- |
| Projected velocity residual | 6.237585133123822e-9 |
| Block sweeps | 172 |
| Dynamics-factor applications, including preparation | 4,706 |
| Newton proposals / accepted | 172 / 89 |
| Krylov response applications | 1,529 |
| Small preconditioner factorizations attempted | 8,256 |
| Backtracking residual evaluations | 872 |
| Explicit global response matrix slots | 0 |

These work counts are substantial; no speedup follows from the sweep count.
Repeated impulses and generalized velocity changes match exactly. Independent
checks balance total linear/angular momentum, enforce unilateral support and
circular friction/rolling limits, and check dissipation. Analytic finite-cube
impacts verify restitution's velocity threshold and absence of penetration-driven
bounce. Nearly coincident support rows shed load onto the correct outer contact
with both 3 and 129 rows. Joint-only car repeats now hash `e06aa32768ad0b86`;
the changed solve order changes the hash from the earlier archived build.

The initial projected solve stagnated near 8.53e-4 while load redistributed
between nearby wheel support points. Removing rolling and splitting manifolds
did not remove the failure. Diagonal scaling, Anderson proposals, unshifted
Newton, and several fixed-shift candidates failed; their logs and source archives
remain here. Fixed-shift 1e-4 reached tolerance only after 596 sweeps. Adaptive
Newton every fourth sweep required 368 sweeps. The retained schedule attempts a
bounded Newton correction after each unfinished sweep, avoiding three extra
block sweeps between corrections.

Newton differentiates the normal-clamp and circular-disk fixed-point equations.
Its matrix-free GMRES uses at most 32 directions, two-pass orthogonalization,
1e-4 relative linear tolerance, and 8 backtracking trials. Per-point pivoted LU
preconditions the search. A dimensionless shift starts at 1e-3, halves after a
full accepted step, doubles after a shortened step, and increases tenfold after
rejection, within [1e-10, 0.1]. This shift affects search directions only: it adds
no physical compliance or impulse bias. Every accepted proposal reduces the
original physical projected residual. Final impulses are projected and motion
is reconstructed and rechecked before reporting convergence. A friction-mode
change at the iteration bound cannot be reported as a completed solve.

Non-smooth Newton contact solving is established; see
[Macklin et al. (2019)](https://arxiv.org/abs/1907.04587). This implementation uses
the locally derived fixed-point derivative above, not a claim to reproduce that
paper's complete formulation. No external implementation was copied.

Verification: core 251, world 91, physics 54, and saved-car 6 tests pass;
workspace Clippy, formatting, and whitespace checks pass. The earlier full
workspace checkpoint still records 11 GPU failures and one UI failure. Hardware
was not rerun for these subsequent CPU-only changes; this is not a green workspace.

Remaining: integrate refreshed contacts with drives/stops and implicit passive
forces, persistent contact validity, collision-event handling, split correction,
validated publication/retries, body/body collision, loops, and complete actual-car
timing. The wider redesign and all application/scale acceptance gates remain open.
