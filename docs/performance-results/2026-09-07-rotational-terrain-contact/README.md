# Rotational terrain contact correctness

Adapter: **Apple M1 Pro / Metal**. Test-profile correctness measurement, not a
release performance or publication-latency benchmark.

```sh
cargo test -p mechanic-gpu rotating_body_cannot --lib -- --nocapture
```

A 0.5 × 0.25 × 0.25 m unjointed cuboid starts at height 0.27 m, rotated 45°
around Z, spinning at 50 rad/s. Its fastest vertices remain below 16 m/s. Both
integration endpoints clear the finite terrain triangle, but an intermediate
orientation crosses it by more than 5 mm. The old translational sweep emitted
zero contacts and allowed the crossing (baseline log:
`/tmp/mechanic-rotational-baseline.log`).

The rotational sweep now emits 2 contacts and stops the pose near
48.298°. Angular Z velocity falls to 28.809 rad/s;
kinetic energy falls to 42.682% of its initial value, including
translation and rotation. Upward COM velocity is a physical transfer from spin
at the off-centre contact, not positional-recovery velocity. Failure flags: zero.
Raw output: [impact.jsonl](impact.jsonl).

Controls verify full motion past a finite triangle whose AABB overlaps the sweep,
rotation parallel to a nearby surface, and zero-area triangles. A paired
unconstrained run verifies that clamped translation and rotation use the same
fraction of the Euler integration step.

Scope: initially separated unjointed bodies crossing between clear endpoints.
Endpoint intersections and initial overlaps retain discrete contact/recovery;
connected mechanisms are excluded from pose clamping. The 256-iteration search
may conservatively shorten motion when it cannot resolve a near-grazing interval.
The remaining fraction of a clamped tick is not re-integrated after the impulse.
Rotational CCD for articulated motion, initial-overlap escape/crossing cases,
additional collider-family rotational fixtures, and performance gates remain open.
