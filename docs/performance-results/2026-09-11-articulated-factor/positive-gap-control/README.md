# Positive-gap event proposal controls — rejected

Both candidates retain the 128-trial policy and all acceptance checks. They change
only the search proposal to target 0.25 times the existing 1e-12 m contact window,
using exact finite-feature endpoint SAT separation. No impulse or state can be
accepted from the scalar gap itself.

The first keeps the earlier 10%–90% bracket safeguard: required cold settling
reaches tick 7 then fails (960 trials across substeps/retries in that attempted
tick). The second allows interior secant proposals and bisects every fourth
proposal: it fails at tick 6 (781 trials). Neither completes the required 120 ticks;
both are rejected and restored out of active source. A later failure alone is
not a physical or performance improvement.

The distinction between a positive activation target and zero-gap overshoot was
worth checking, but these controls do not establish it as the root cause. Further
arbitrary interpolation/activation constants are not justified. Preserve finite
geometry and study contact/force trajectory consistency and interval event work
using these fixtures. The broader continuous-force/rotation proof remains open.

Each tarball is a three-file overlay over the parent checkpoint's final
`implementation.tar.gz`. Both use the same required test:

```
cargo test -p mechanic-physics --offline saved_car_cold_drop_remains_bounded_through_settling -- --test-threads=1 --nocapture
```

No accepted physics, simulation duration, or hardware performance gate follows
from these diagnostic runs. Raw logs retain failure phase work and publication
rollback evidence.
