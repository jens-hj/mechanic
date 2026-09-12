# Exact linear CCD and below-surface separation

Zero accumulated ancestor angular motion now selects exact finite translation SAT,
including root and suspension translation. The already reconstructed endpoint pose
is cached. Diagnostics distinguish preparation poses, query poses and interval
queries. Full rotations retain conservative advancement and explicit exhaustion;
endpoint quaternion equality cannot select the linear route.

Regression cases cover fast crossing with both endpoints clear, near-separated
parallel motion, finite holes, origin rebases and the authored car's affine
suspension path. A separated body below a finite triangle formerly exhausted its
penetration certificate. The envelope now proves separation only when an outward
support bound lies below the entire triangle projection range. Errors still fail.
A displacement envelope that can cross the plane is never classified separated.
The pre-fix failing test and exact source are retained alongside this checkpoint.

CPU package tests, workspace Clippy, formatting and whitespace checks pass; raw logs
and source identities are attached. Hardware and app validation reference the prior
supported-path checkpoint: the same 11 GPU failures and one UI failure remain open.
This is correctness work, with no new performance measurement or accepted scale gate.
New impacts still hold in this frozen source; event integration follows separately.
