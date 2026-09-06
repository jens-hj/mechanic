// Multitool FX — hybrid vocabulary.
//   shards  : flat hard-edged instanced triangles, quantised 3-tone ramp, no alpha falloff.
//             matter, impact, spatter — anything that is a piece of something.
//   traces  : additive line segments, one frame to a fifth of a second.
//             arcs, beams, spool, shockwave, freeze cast — anything that is energy.
// One shard pool and one trace pool for the whole system: two draw calls, whatever
// is firing. Budget class "mid" — 1400 shards / 900 traces, instanced, emissive-only
// brightness so the existing bloom pass is the only pass needed.
//
// Every tool's particles are that tool's accent, quantised to three tones of it.
// The freeze state is rift violet #BC4FE8 everywhere — outline, cast and Link pulse —
// because the Dimension Link is the device doing the work; the sledge is the remote.

const TAU = Math.PI * 2;

export const RAMPS = {
  sledge:    ['#8A5A1E', '#C98A34', '#F0C77A'],
  matter:    ['#1E8C74', '#2FD8B4', '#A8F5E4'],
  welder:    ['#8E2B2E', '#E2565A', '#FFD9A8'],
  connector: ['#1B6D8E', '#2FA8D8', '#A6E4FA'],
  freeze:    ['#6E2A8C', '#BC4FE8', '#E9BFF7']
};

const HALO = {
  gap: 0.075,      // how far the dashed halo stands off the creation's bounds
  dash: 0.055,
  period: 0.125,
  travel: 0.16,    // m/s along the perimeter — slow enough to read as held, not scanning
  pulse: 0.55      // laps per second of the shared clock
};

const rnd = (a, b) => a + Math.random() * (b - a);
const pick = (arr) => arr[(Math.random() * arr.length) | 0];

export function makeFX(THREE, opts = {}) {
  const SHARD_CAP = opts.shards || 1400;
  const TRACE_CAP = opts.traces || 900;
  const DASH_CAP = 220;

  const group = new THREE.Group();
  group.name = 'fx';

  // ---------------------------------------------------------------- shards
  // A flat scalene triangle, not a quad: a quad catches the light as a plate and
  // reads as a panel from the tool. Three verts, double-sided, unlit.
  const tri = new THREE.BufferGeometry();
  tri.setAttribute('position', new THREE.Float32BufferAttribute(
    [-0.5, -0.35, 0, 0.62, -0.2, 0, -0.08, 0.6, 0], 3
  ));
  tri.computeVertexNormals();

  const shardMat = new THREE.MeshBasicMaterial({ side: THREE.DoubleSide, toneMapped: false });
  const shards = new THREE.InstancedMesh(tri, shardMat, SHARD_CAP);
  shards.name = 'fx_shards';
  shards.frustumCulled = false;
  shards.castShadow = false;
  shards.receiveShadow = false;
  shards.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  group.add(shards);

  const pool = [];
  const dummy = new THREE.Object3D();
  const zero = new THREE.Object3D();
  zero.scale.set(0, 0, 0);
  zero.updateMatrix();
  const col = new THREE.Color();
  const ramps = {};
  for (const k in RAMPS) ramps[k] = RAMPS[k].map((h) => new THREE.Color(h));

  for (let i = 0; i < SHARD_CAP; i++) {
    pool.push({
      live: false, t: 0, life: 1, tone: -1, ramp: 'sledge',
      p: new THREE.Vector3(), v: new THREE.Vector3(),
      ax: new THREE.Vector3(0, 1, 0), spin: 0, ang: 0, size: 0.02, g: 0,
      pullTo: null, pull: 0
    });
    shards.setMatrixAt(i, zero.matrix);
    shards.setColorAt(i, col.set('#000000'));
  }
  shards.instanceMatrix.needsUpdate = true;
  shards.instanceColor.needsUpdate = true;
  let shardCursor = 0;

  function shard(o) {
    let s = null;
    for (let n = 0; n < SHARD_CAP; n++) {
      const c = pool[(shardCursor + n) % SHARD_CAP];
      if (!c.live) { s = c; shardCursor = (shardCursor + n + 1) % SHARD_CAP; break; }
    }
    if (!s) return null;              // pool full: drop it, never grow mid-frame
    s.live = true; s.t = 0; s.tone = -1;
    s.life = o.life; s.ramp = o.ramp; s.size = o.size; s.g = o.g || 0;
    s.p.copy(o.p); s.v.copy(o.v);
    s.ax.set(rnd(-1, 1), rnd(-1, 1), rnd(-1, 1)).normalize();
    s.spin = o.spin != null ? o.spin : rnd(-14, 14);
    s.ang = rnd(0, TAU);
    s.pullTo = o.pullTo || null;
    s.pull = o.pull || 0;
    return s;
  }

  // ---------------------------------------------------------------- traces
  const tPos = new Float32Array(TRACE_CAP * 6);
  const tCol = new Float32Array(TRACE_CAP * 6);
  const traceGeo = new THREE.BufferGeometry();
  traceGeo.setAttribute('position', new THREE.BufferAttribute(tPos, 3).setUsage(THREE.DynamicDrawUsage));
  traceGeo.setAttribute('color', new THREE.BufferAttribute(tCol, 3).setUsage(THREE.DynamicDrawUsage));
  traceGeo.setDrawRange(0, 0);
  const traceMat = new THREE.LineBasicMaterial({
    vertexColors: true, transparent: true, blending: THREE.AdditiveBlending,
    depthWrite: false, toneMapped: false
  });
  const traces = new THREE.LineSegments(traceGeo, traceMat);
  traces.name = 'fx_traces';
  traces.frustumCulled = false;
  group.add(traces);

  const tPool = [];
  for (let i = 0; i < TRACE_CAP; i++) {
    tPool.push({
      live: false, t: 0, life: 1, ramp: 'sledge', tone: 2,
      a: new THREE.Vector3(), b: new THREE.Vector3(),
      v: new THREE.Vector3(), g: 0
    });
  }
  let tCursor = 0;

  function trace(o) {
    let s = null;
    for (let n = 0; n < TRACE_CAP; n++) {
      const c = tPool[(tCursor + n) % TRACE_CAP];
      if (!c.live) { s = c; tCursor = (tCursor + n + 1) % TRACE_CAP; break; }
    }
    if (!s) return null;
    s.live = true; s.t = 0;
    s.life = o.life; s.ramp = o.ramp; s.tone = o.tone != null ? o.tone : 2;
    s.a.copy(o.a); s.b.copy(o.b);
    if (o.v) s.v.copy(o.v); else s.v.set(0, 0, 0);
    s.g = o.g || 0;
    return s;
  }

  // ------------------------------------------------- freeze halo + inner outline
  // The halo is not a dashed material: three.js dashes are baked into line
  // distances and cannot travel. These dashes are rewritten every frame by
  // walking the 12 box edges with a moving phase, so the motion is real and
  // shares its clock with the Link's aperture pulse.
  const haloGroup = new THREE.Group();
  haloGroup.name = 'fx_freeze';
  haloGroup.visible = false;
  group.add(haloGroup);

  const dPos = new Float32Array(DASH_CAP * 6);
  const dashGeo = new THREE.BufferGeometry();
  dashGeo.setAttribute('position', new THREE.BufferAttribute(dPos, 3).setUsage(THREE.DynamicDrawUsage));
  dashGeo.setDrawRange(0, 0);
  const dashMat = new THREE.LineBasicMaterial({
    color: RAMPS.freeze[1], transparent: true, opacity: 0.95,
    blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false
  });
  const dashes = new THREE.LineSegments(dashGeo, dashMat);
  dashes.frustumCulled = false;
  haloGroup.add(dashes);

  const innerMat = new THREE.LineBasicMaterial({
    color: RAMPS.freeze[0], transparent: true, opacity: 0.5,
    blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false
  });
  let inner = null;

  let target = null;         // the frozen creation
  let box = new THREE.Box3(); // its bounds, in its own local space
  let edges = [];             // [{a,b,len}] × 12, in local space
  let radius = 0.5;

  function setTarget(obj) {
    target = obj || null;
    if (inner) { haloGroup.remove(inner); inner.geometry.dispose(); inner = null; }
    edges = [];
    if (!target) return;
    const keep = target.position.clone();
    target.position.set(0, 0, 0);
    target.updateMatrixWorld(true);
    box.setFromObject(target);
    target.position.copy(keep);
    target.updateMatrixWorld(true);
    radius = box.getBoundingSphere(new THREE.Sphere()).radius;

    const g = HALO.gap;
    const mn = box.min.clone().addScalar(-g);
    const mx = box.max.clone().addScalar(g);
    const c = [
      [mn.x, mn.y, mn.z], [mx.x, mn.y, mn.z], [mx.x, mn.y, mx.z], [mn.x, mn.y, mx.z],
      [mn.x, mx.y, mn.z], [mx.x, mx.y, mn.z], [mx.x, mx.y, mx.z], [mn.x, mx.y, mx.z]
    ].map((v) => new THREE.Vector3(v[0], v[1], v[2]));
    const pairs = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
    for (const [i, j] of pairs) {
      edges.push({ a: c[i], b: c[j], len: c[i].distanceTo(c[j]) });
    }
    // Inner solid outline sits on the true bounds — a hairline, so the dashed
    // halo stays the loud element and the creation never gains a border.
    inner = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(
        Math.max(box.max.x - box.min.x, 0.001),
        Math.max(box.max.y - box.min.y, 0.001),
        Math.max(box.max.z - box.min.z, 0.001)
      )),
      innerMat
    );
    box.getCenter(inner.position);
    haloGroup.add(inner);
  }

  const dTmpA = new THREE.Vector3();
  const dTmpB = new THREE.Vector3();

  function writeDashes(offset) {
    let n = 0;
    for (const e of edges) {
      let s = (offset % HALO.period) - HALO.period;
      while (s < e.len && n < DASH_CAP) {
        const s0 = Math.max(s, 0);
        const s1 = Math.min(s + HALO.dash, e.len);
        if (s1 > s0) {
          dTmpA.lerpVectors(e.a, e.b, s0 / e.len);
          dTmpB.lerpVectors(e.a, e.b, s1 / e.len);
          dPos[n * 6 + 0] = dTmpA.x; dPos[n * 6 + 1] = dTmpA.y; dPos[n * 6 + 2] = dTmpA.z;
          dPos[n * 6 + 3] = dTmpB.x; dPos[n * 6 + 4] = dTmpB.y; dPos[n * 6 + 5] = dTmpB.z;
          n++;
        }
        s += HALO.period;
      }
    }
    dashGeo.attributes.position.needsUpdate = true;
    dashGeo.setDrawRange(0, n * 2);
  }

  // --------------------------------------------------------------- link pulse
  let linkMats = [];
  let linkBase = [];
  function setLink(mats) {
    linkMats = Array.isArray(mats) ? mats : (mats ? [mats] : []);
    linkBase = linkMats.map((m) => (m.emissiveIntensity != null ? m.emissiveIntensity : 1));
  }

  // ------------------------------------------------------------------ emitters
  const A = new THREE.Vector3();
  const B = new THREE.Vector3();
  const C = new THREE.Vector3();

  function sledgeImpact(hit, normal) {
    const nrm = C.copy(normal || new THREE.Vector3(0, 1, 0)).normalize();
    for (let i = 0; i < 42; i++) {
      A.set(rnd(-1, 1), rnd(-1, 1), rnd(-1, 1)).normalize().add(nrm).normalize();
      shard({
        p: hit, v: A.clone().multiplyScalar(rnd(1.1, 3.2)), ramp: 'sledge',
        life: rnd(0.32, 0.7), size: rnd(0.011, 0.030), g: 4.2
      });
    }
    // Shockwave: tangential segments on the impact plane, stepped out over 0.22 s.
    const u = new THREE.Vector3(1, 0, 0);
    if (Math.abs(nrm.x) > 0.8) u.set(0, 1, 0);
    const e1 = u.clone().cross(nrm).normalize();
    const e2 = nrm.clone().cross(e1).normalize();
    for (let i = 0; i < 26; i++) {
      const th = (i / 26) * TAU;
      const dir = e1.clone().multiplyScalar(Math.cos(th)).add(e2.clone().multiplyScalar(Math.sin(th)));
      const tan = e1.clone().multiplyScalar(-Math.sin(th)).add(e2.clone().multiplyScalar(Math.cos(th)));
      const r0 = 0.06;
      trace({
        a: hit.clone().add(dir.clone().multiplyScalar(r0)).add(tan.clone().multiplyScalar(-0.03)),
        b: hit.clone().add(dir.clone().multiplyScalar(r0)).add(tan.clone().multiplyScalar(0.03)),
        v: dir.clone().multiplyScalar(2.0), ramp: 'sledge', life: 0.24, tone: 2
      });
    }
  }

  function matterShot(origin, hit) {
    // Shards leave the ring and converge — the pull term is what makes the
    // stream read as a manipulator rather than as a muzzle spray.
    for (let i = 0; i < 22; i++) {
      A.copy(hit).sub(origin);
      const d = A.length();
      A.normalize();
      B.set(rnd(-1, 1), rnd(-1, 1), rnd(-1, 1)).normalize().multiplyScalar(0.55);
      shard({
        p: origin.clone().add(B.clone().multiplyScalar(0.06)),
        v: A.clone().multiplyScalar(rnd(2.4, 3.6)).add(B),
        ramp: 'matter', life: d / 3.0 + rnd(0, 0.1), size: rnd(0.010, 0.020),
        pullTo: hit.clone(), pull: 16
      });
    }
    for (let i = 0; i < 10; i++) {
      A.copy(origin).lerp(hit, i / 10);
      B.copy(origin).lerp(hit, (i + 0.55) / 10);
      trace({ a: A, b: B, ramp: 'matter', life: 0.14, tone: 1 });
    }
  }

  function matterIdle(origin, dt, t) {
    // Payload turning in the ring: short-lived shards respawned on a circle,
    // so nothing has to persist between frames.
    const n = Math.random() < dt * 90 ? 2 : 1;
    for (let i = 0; i < n; i++) {
      const th = t * 2.2 + i * Math.PI;
      A.set(Math.cos(th) * 0.055, Math.sin(th * 0.7) * 0.02, Math.sin(th) * 0.055).add(origin);
      B.set(-Math.sin(th), 0, Math.cos(th)).multiplyScalar(0.16);
      shard({ p: A, v: B, ramp: 'matter', life: 0.30, size: rnd(0.008, 0.014), spin: rnd(-4, 4) });
    }
  }

  function welderArc(tip, dt) {
    for (let i = 0; i < 5; i++) {
      A.copy(tip).add(new THREE.Vector3(rnd(-0.02, 0.02), rnd(-0.02, 0.02), rnd(-0.02, 0.02)));
      B.copy(tip).add(new THREE.Vector3(rnd(-0.05, 0.05), rnd(-0.05, 0.05), rnd(-0.05, 0.05)));
      trace({ a: A, b: B, ramp: 'welder', life: 0.05, tone: 2 });
    }
    if (Math.random() < dt * 160) {
      for (let i = 0; i < 3; i++) {
        A.set(rnd(-1, 1), rnd(-0.2, 1), rnd(-1, 1)).normalize();
        shard({
          p: tip, v: A.clone().multiplyScalar(rnd(0.9, 2.6)), ramp: 'welder',
          life: rnd(0.35, 0.85), size: rnd(0.006, 0.013), g: 9.0
        });
      }
    }
    if (Math.random() < dt * 30) {
      A.copy(tip).add(new THREE.Vector3(rnd(-0.05, 0.05), 0.02, rnd(-0.05, 0.05)));
      B.copy(A).add(new THREE.Vector3(0, 0.06, 0));
      trace({ a: A, b: B, v: new THREE.Vector3(rnd(-0.05, 0.05), 0.42, rnd(-0.05, 0.05)), ramp: 'welder', life: 0.55, tone: 0 });
    }
  }

  function connectorBeam(origin, hit, t) {
    // Rebuilt every frame from one-frame traces: a stepped run, not a straight
    // beam, because the tool is laying a chain rather than firing.
    const N = 11;
    const prev = A.copy(origin);
    let px = prev.x, py = prev.y, pz = prev.z;
    for (let i = 1; i <= N; i++) {
      const f = i / N;
      B.copy(origin).lerp(hit, f);
      const j = 0.035 * Math.sin(f * 9 + t * 7) * (1 - f);
      B.x += j; B.y += 0.045 * Math.sin(f * 6 - t * 5) * (1 - f * 0.6); B.z -= j;
      trace({ a: new THREE.Vector3(px, py, pz), b: B, ramp: 'connector', life: 0.06, tone: i > N - 3 ? 2 : 1 });
      px = B.x; py = B.y; pz = B.z;
    }
    // Spool: a helix winding along the run at 3.8 rad/s, matching the tool's own
    // active state in the rig.
    const ax = C.copy(hit).sub(origin);
    const L = ax.length();
    ax.normalize();
    const u = new THREE.Vector3(0, 1, 0);
    if (Math.abs(ax.y) > 0.85) u.set(1, 0, 0);
    const e1 = u.clone().cross(ax).normalize();
    const e2 = ax.clone().cross(e1).normalize();
    for (let i = 0; i < 4; i++) {
      const f = ((t * 0.55 + i * 0.25) % 1);
      const th = f * 10 + t * 3.8;
      const r = 0.05 * (1 - f * 0.5);
      A.copy(origin).add(ax.clone().multiplyScalar(f * L))
        .add(e1.clone().multiplyScalar(Math.cos(th) * r))
        .add(e2.clone().multiplyScalar(Math.sin(th) * r));
      B.copy(origin).add(ax.clone().multiplyScalar((f + 0.03) * L))
        .add(e1.clone().multiplyScalar(Math.cos(th + 0.5) * r))
        .add(e2.clone().multiplyScalar(Math.sin(th + 0.5) * r));
      trace({ a: A, b: B, ramp: 'connector', life: 0.18, tone: 2 });
    }
  }

  function freezeCast(center) {
    // Violet, not amber: the sledge is the remote, the Link is the device.
    // Everything converges inward — the creation is being taken hold of.
    const r = radius * 1.55;
    for (let i = 0; i < 34; i++) {
      A.set(rnd(-1, 1), rnd(-1, 1), rnd(-1, 1)).normalize();
      B.copy(center).add(A.clone().multiplyScalar(r));
      shard({
        p: B, v: A.clone().multiplyScalar(-rnd(1.2, 2.2)), ramp: 'freeze',
        life: 0.42, size: rnd(0.012, 0.024), pullTo: center.clone(), pull: 5
      });
      if (i % 2 === 0) {
        trace({
          a: B, b: B.clone().add(A.clone().multiplyScalar(-0.12)),
          v: A.clone().multiplyScalar(-2.6), ramp: 'freeze', life: 0.26, tone: 2
        });
      }
    }
  }

  function liftWake(center, dir) {
    // Held out of physics and moved on one axis: the wake is the only motion cue,
    // since nothing about the creation itself is allowed to move.
    for (let i = 0; i < 2; i++) {
      A.copy(center).add(new THREE.Vector3(rnd(-1, 1), rnd(-1, 1), rnd(-1, 1)).multiplyScalar(radius * 0.9));
      B.copy(A).add(new THREE.Vector3(0, -dir * rnd(0.06, 0.14), 0));
      trace({ a: A, b: B, v: new THREE.Vector3(0, -dir * 0.9, 0), ramp: 'freeze', life: 0.30, tone: 1 });
    }
  }

  // ---------------------------------------------------------------- integrate
  let clock = 0;
  let dashOffset = 0;
  let frozen = false;
  let sustain = null;
  let rig = { origin: new THREE.Vector3(), hit: new THREE.Vector3(), normal: new THREE.Vector3(0, 1, 0) };
  let lift = 0;

  function update(dt) {
    dt = Math.min(dt, 0.05);
    clock += dt;

    if (sustain === 'matter') matterIdle(rig.origin, dt, clock);
    else if (sustain === 'welder') welderArc(rig.hit, dt);
    else if (sustain === 'connector') connectorBeam(rig.origin, rig.hit, clock);

    if (frozen && lift !== 0) liftWake(rig.centre || rig.hit, Math.sign(lift));

    // shards
    let nLive = 0;
    for (let i = 0; i < SHARD_CAP; i++) {
      const s = pool[i];
      if (!s.live) continue;
      s.t += dt;
      if (s.t >= s.life) {
        s.live = false;
        shards.setMatrixAt(i, zero.matrix);
        continue;
      }
      if (s.pullTo && s.pull) {
        A.copy(s.pullTo).sub(s.p);
        const d = A.length() || 1;
        s.v.addScaledVector(A.multiplyScalar(1 / d), s.pull * dt * Math.min(1 / d, 4));
      }
      if (s.g) s.v.y -= s.g * dt;
      s.p.addScaledVector(s.v, dt);
      s.ang += s.spin * dt;

      const f = 1 - s.t / s.life;
      const tone = f > 0.62 ? 2 : f > 0.28 ? 1 : 0;
      if (tone !== s.tone) {
        s.tone = tone;
        shards.setColorAt(i, ramps[s.ramp][tone]);
        shards.instanceColor.needsUpdate = true;
      }
      // Size steps down with the tone rather than fading alpha — the shards stay
      // hard-edged all the way out, which is the whole point of the vocabulary.
      const k = s.size * (0.55 + 0.45 * (tone / 2));
      dummy.position.copy(s.p);
      dummy.quaternion.setFromAxisAngle(s.ax, s.ang);
      dummy.scale.set(k, k, k);
      dummy.updateMatrix();
      shards.setMatrixAt(i, dummy.matrix);
      nLive++;
    }
    shards.instanceMatrix.needsUpdate = true;

    // traces
    let n = 0;
    for (let i = 0; i < TRACE_CAP; i++) {
      const s = tPool[i];
      if (!s.live) continue;
      s.t += dt;
      if (s.t >= s.life) { s.live = false; continue; }
      if (s.g) s.v.y -= s.g * dt;
      if (s.v.lengthSq()) { s.a.addScaledVector(s.v, dt); s.b.addScaledVector(s.v, dt); }
      const f = 1 - s.t / s.life;
      const tone = s.tone === 2 ? (f > 0.4 ? 2 : 1) : s.tone === 1 ? (f > 0.4 ? 1 : 0) : 0;
      const c = ramps[s.ramp][tone];
      const o = n * 6;
      tPos[o] = s.a.x; tPos[o + 1] = s.a.y; tPos[o + 2] = s.a.z;
      tPos[o + 3] = s.b.x; tPos[o + 4] = s.b.y; tPos[o + 5] = s.b.z;
      tCol[o] = c.r; tCol[o + 1] = c.g; tCol[o + 2] = c.b;
      tCol[o + 3] = c.r; tCol[o + 4] = c.g; tCol[o + 5] = c.b;
      n++;
    }
    traceGeo.attributes.position.needsUpdate = true;
    traceGeo.attributes.color.needsUpdate = true;
    traceGeo.setDrawRange(0, n * 2);

    // freeze halo + link pulse, one clock
    if (frozen && edges.length) {
      dashOffset += HALO.travel * dt;
      writeDashes(dashOffset);
      // Triangle wave off the same phase the dashes travel on, so the aperture
      // brightens exactly as a dash crosses a corner. Two effects, one system.
      const ph = (clock * HALO.pulse) % 1;
      const pulse = ph < 0.5 ? ph * 2 : 2 - ph * 2;
      dashMat.opacity = 0.55 + 0.45 * pulse;
      innerMat.opacity = 0.28 + 0.30 * pulse;
      for (let i = 0; i < linkMats.length; i++) {
        linkMats[i].emissiveIntensity = linkBase[i] * (0.75 + 1.55 * pulse);
      }
      if (target) haloGroup.position.copy(target.position);
    } else if (linkMats.length) {
      for (let i = 0; i < linkMats.length; i++) linkMats[i].emissiveIntensity = linkBase[i];
    }

    return { shards: nLive, traces: n };
  }

  return {
    group,
    RAMPS,
    setTarget,
    setLink,
    setRig(o) {
      if (o.origin) rig.origin.copy(o.origin);
      if (o.hit) rig.hit.copy(o.hit);
      if (o.normal) rig.normal.copy(o.normal);
      if (o.centre) rig.centre = o.centre.clone();
    },
    setSustain(id) { sustain = id || null; },
    setLift(v) { lift = v || 0; },
    setFrozen(on) {
      frozen = !!on;
      haloGroup.visible = frozen;
      if (frozen) { dashOffset = 0; writeDashes(0); if (target) haloGroup.position.copy(target.position); }
    },
    fire(id) {
      if (id === 'sledge') sledgeImpact(rig.hit, rig.normal);
      else if (id === 'matter') matterShot(rig.origin, rig.hit);
      else if (id === 'freeze') freezeCast(rig.centre || rig.hit);
    },
    update
  };
}
