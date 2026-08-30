// Multitool — 3D pass. Petal shell, four folds.
// Authored in metres, y up, staff centred on the origin. Front of a deployed
// end is +Z. Beats 1 and 2 are shared by every tool; only beat 3 differs.

const PALETTE = {
  steel:   { color: '#7E8B94', rough: 0.42, metal: 0.95 },
  alu:     { color: '#5A6B76', rough: 0.50, metal: 0.90 },
  mid:     { color: '#414F5A', rough: 0.55, metal: 0.85 },
  dark:    { color: '#2A343C', rough: 0.62, metal: 0.70 },
  black:   { color: '#1B232A', rough: 0.78, metal: 0.30 },
  bright:  { color: '#9DAAB2', rough: 0.30, metal: 1.00 },
  grip:    { color: '#14191E', rough: 0.95, metal: 0.05 },
  amber:   { color: '#C98A34', rough: 0.45, metal: 0.80, emissive: '#C98A34', ei: 0.35 },
  mint:    { color: '#2FD8B4', rough: 0.40, metal: 0.30, emissive: '#2FD8B4', ei: 1.10 },
  red:     { color: '#E2565A', rough: 0.40, metal: 0.30, emissive: '#E2565A', ei: 1.20 },
  cyan:    { color: '#2FA8D8', rough: 0.40, metal: 0.30, emissive: '#2FA8D8', ei: 1.00 },
  amberLit:{ color: '#C98A34', rough: 0.40, metal: 0.40, emissive: '#C98A34', ei: 0.75 }
};

const ACCENT = { sledge: 'amber', matter: 'mint', welder: 'red', connector: 'cyan' };

const AF = 0.050;        // hex apothem — 100 mm across flats
const PL = 0.280;        // panel length
const PT = 0.011;        // panel thickness
const PW = 0.052;        // panel width
const PRISM_Y0 = 0.600;  // prism base height on the staff
const HALF_LEN = 0.900;  // staff half length — 1.8 m overall, a real quarterstaff

// Per-tool beat-3 targets, per panel index (0 = +Z front, counter-clockwise).
// Sledge: every panel is head. Three laminate onto each side — the ring
// indexes them around to the two strike quadrants, then they clamp inward.
const SLEDGE = {
  0: { r: -0.010, y: 0, role: 'lamina' },
  1: { r: 0.014, y: -Math.PI / 3, role: 'lamina' },
  5: { r: 0.038, y: Math.PI / 3, spin: Math.PI, role: 'strike' },
  3: { r: -0.010, y: 0, role: 'lamina' },
  2: { r: 0.014, y: Math.PI / 3, role: 'lamina' },
  4: { r: 0.038, y: -Math.PI / 3, spin: Math.PI, role: 'strike' }
};
const SLEDGE_TH = 2.0; // panels forge up to double thickness as they laminate

const FOLDS = {
  sledge: (k) => ({ r: SLEDGE[k].r, s: 0, f: 0, y: SLEDGE[k].y, th: SLEDGE_TH, spin: SLEDGE[k].spin || 0, role: SLEDGE[k].role }),
  matter: (k) => (k % 2 === 0)
    ? { r: 0.004, s: -0.055, f: 0.20, spin: 0, role: 'finger' }
    : { r: 0.000, s: -0.255, f: 0.04, spin: 0, role: 'collar' },
  welder: (k) => (k === 0 || k === 3)
    ? { r: -0.006, s: 0.000, f: 0, spin: 0, role: 'tine' }
    : (k === 1 || k === 5)
      ? { r: 0.006, s: -0.190, f: 1.15, spin: 0, role: 'shield' }
      : { r: 0.002, s: -0.262, f: 0.07, spin: 0, role: 'fin' },
  connector: (k) => ({ r: 0.006, s: -0.100, f: 0.44 + (k % 2 ? 0.06 : 0), spin: 0, role: 'rib' })
};

const EASE = (t) => (t <= 0 ? 0 : t >= 1 ? 1 : t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const CL = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);

export async function mount(stage) {
  const { THREE } = await stage.ready;

  const M = {};
  for (const k in PALETTE) {
    const d = PALETTE[k];
    M[k] = new THREE.MeshStandardMaterial({
      name: k, color: d.color, roughness: d.rough, metalness: d.metal,
      emissive: d.emissive || '#000000', emissiveIntensity: d.emissive ? d.ei : 0
    });
  }

  const box = (x, y, z) => new THREE.BoxGeometry(x, y, z);
  const cyl = (r, h, seg = 32, r2 = r) => new THREE.CylinderGeometry(r, r2, h, seg);
  const mesh = (name, geo, mat, pos, rot) => {
    const m = new THREE.Mesh(geo, mat);
    m.name = name;
    if (pos) m.position.set(pos[0], pos[1], pos[2]);
    if (rot) m.rotation.set(rot[0], rot[1], rot[2]);
    m.castShadow = m.receiveShadow = true;
    return m;
  };

  const root = new THREE.Group(); root.name = 'multitool';
  const pivot = new THREE.Group(); pivot.name = 'pivot';
  pivot.position.y = HALF_LEN + 0.15; // butt on the ground — it is a staff
  root.add(pivot);

  // ---- shaft -------------------------------------------------------------
  const shaft = new THREE.Group(); shaft.name = 'shaft'; pivot.add(shaft);
  shaft.add(mesh('shaft_tube', cyl(0.0235, 2 * (PRISM_Y0 + 0.02), 28), M.mid, [0, 0, 0]));
  shaft.add(mesh('shaft_flute', cyl(0.0255, 2 * (PRISM_Y0 - 0.24), 6), M.alu, [0, 0, 0]));
  for (const s of [1, -1]) {
    shaft.add(mesh('collar_' + s, cyl(0.0305, 0.022, 24), M.alu, [0, s * 0.415, 0]));
    shaft.add(mesh('collar_lip_' + s, cyl(0.0325, 0.006, 24), M.bright, [0, s * 0.404, 0]));
    shaft.add(mesh('prism_root_' + s, cyl(0.0345, 0.030, 6), M.alu, [0, s * (PRISM_Y0 - 0.016), 0]));
  }
  const gripMat = M.grip;
  shaft.add(mesh('grip', cyl(0.0285, 0.360, 28), gripMat, [0, 0, 0]));
  for (let i = 0; i < 11; i++) {
    shaft.add(mesh('grip_rib_' + i, new THREE.TorusGeometry(0.0292, 0.0022, 8, 28), M.black,
      [0, -0.165 + i * 0.033, 0], [Math.PI / 2, 0, 0]));
  }
  shaft.add(mesh('grip_band', cyl(0.0295, 0.012, 28), M.alu, [0, 0.196, 0]));

  // ---- one end ------------------------------------------------------------
  function makeEnd(tag) {
    const group = new THREE.Group(); group.name = 'end_' + tag;

    // bare core, revealed by beat 2
    const core = new THREE.Group(); core.name = 'core_' + tag; group.add(core);
    core.add(mesh('core_spine', cyl(0.026, PL - 0.01, 20), M.dark, [0, PL / 2, 0]));
    for (let i = 0; i < 4; i++) {
      core.add(mesh('core_ring_' + i, new THREE.TorusGeometry(0.0315, 0.004, 8, 22), M.alu,
        [0, 0.045 + i * 0.062, 0], [Math.PI / 2, 0, 0]));
    }
    core.add(mesh('core_base', cyl(0.040, 0.024, 6), M.alu, [0, 0.012, 0]));

    // per-tool cores
    const cores = {};

    const cSledge = new THREE.Group(); cores.sledge = cSledge; group.add(cSledge);
    cSledge.add(mesh('sl_anvil', box(0.078, 0.200, 0.060), M.black, [0, 0.160, 0]));
    cSledge.add(mesh('sl_shoulder', box(0.092, 0.034, 0.062), M.alu, [0, 0.272, 0]));
    cSledge.add(mesh('sl_crown', cyl(0.031, 0.030, 6), M.steel, [0, 0.300, 0]));
    for (let i = 0; i < 3; i++) {
      cSledge.add(mesh('sl_rib_' + i, box(0.086, 0.012, 0.064), M.alu, [0, 0.104 + i * 0.058, 0]));
    }
    cSledge.add(mesh('sl_heel', box(0.092, 0.028, 0.062), M.alu, [0, 0.062, 0]));

    const cMatter = new THREE.Group(); cores.matter = cMatter; group.add(cMatter);
    cMatter.add(mesh('mm_hopper', box(0.058, 0.090, 0.058), M.mid, [0, 0.085, 0]));
    cMatter.add(mesh('mm_window', box(0.030, 0.052, 0.062), M.mint, [0, 0.085, 0]));
    cMatter.add(mesh('mm_neck', cyl(0.026, 0.070, 20), M.alu, [0, 0.165, 0]));
    cMatter.add(mesh('mm_nozzle', cyl(0.013, 0.055, 20, 0.028), M.steel, [0, 0.225, 0]));
    cMatter.add(mesh('mm_lens', cyl(0.012, 0.008, 20), M.mint, [0, 0.253, 0]));
    const ring = mesh('mm_ring', new THREE.TorusGeometry(0.098, 0.0075, 12, 60), M.mint, [0, 0.335, 0]);
    cMatter.add(ring);
    const payload = mesh('mm_payload', box(0.072, 0.072, 0.072), M.mint, [0, 0.335, 0]);
    payload.material = M.mint.clone();
    payload.material.transparent = true;
    payload.material.opacity = 0.45;
    cMatter.add(payload);

    const cWelder = new THREE.Group(); cores.welder = cWelder; group.add(cWelder);
    cWelder.add(mesh('wd_body', box(0.056, 0.120, 0.056), M.mid, [0, 0.095, 0]));
    cWelder.add(mesh('wd_band', box(0.060, 0.012, 0.060), M.red, [0, 0.150, 0]));
    cWelder.add(mesh('wd_feed', cyl(0.020, 0.090, 18), M.alu, [0, 0.205, 0]));
    cWelder.add(mesh('wd_gas_l', cyl(0.008, 0.070, 12), M.dark, [-0.030, 0.200, 0.020]));
    cWelder.add(mesh('wd_gas_r', cyl(0.008, 0.070, 12), M.dark, [0.030, 0.200, 0.020]));
    const arc = mesh('wd_arc', new THREE.SphereGeometry(0.016, 18, 14), M.red, [0, 0.300, 0]);
    cWelder.add(arc);

    const cConn = new THREE.Group(); cores.connector = cConn; group.add(cConn);
    cConn.add(mesh('cn_base', cyl(0.032, 0.055, 6), M.alu, [0, 0.045, 0]));
    cConn.add(mesh('cn_counter', box(0.044, 0.018, 0.010), M.cyan, [0, 0.062, 0.030]));
    cConn.add(mesh('cn_mast', cyl(0.011, 0.230, 16), M.mid, [0, 0.185, 0]));
    cConn.add(mesh('cn_collar', cyl(0.020, 0.014, 16), M.alu, [0, 0.145, 0]));
    cConn.add(mesh('cn_lamp', new THREE.SphereGeometry(0.017, 20, 16), M.cyan, [0, 0.312, 0]));

    // panels
    const panels = [];
    for (let k = 0; k < 6; k++) {
      const carrier = new THREE.Group(); carrier.name = 'panel_carrier_' + k;
      carrier.rotation.y = (k * Math.PI) / 3;
      group.add(carrier);
      const radial = new THREE.Group(); carrier.add(radial);
      const slide = new THREE.Group(); radial.add(slide);
      const hinge = new THREE.Group(); slide.add(hinge);
      const spinner = new THREE.Group(); hinge.add(spinner);

      const accents = {};
      const plate = mesh('panel_' + k, box(PW, PL, PT), M.alu, [0, PL / 2, 0]);
      spinner.add(plate);
      spinner.add(mesh('panel_' + k + '_rail', box(PW - 0.014, PL - 0.03, 0.004), M.dark, [0, PL / 2, -PT / 2]));
      spinner.add(mesh('panel_' + k + '_lip_a', box(PW - 0.003, 0.014, PT - 0.002), M.bright, [0, PL - 0.012, 0]));
      spinner.add(mesh('panel_' + k + '_lip_b', box(PW - 0.003, 0.014, PT - 0.002), M.dark, [0, 0.012, 0]));
      for (const key of ['amber', 'mint', 'red', 'cyan']) {
        const a = mesh('panel_' + k + '_' + key, box(PW - 0.010, 0.014, 0.004), M[key], [0, 0.072, PT / 2]);
        const b = mesh('panel_' + k + '_' + key + '2', box(PW - 0.010, 0.014, 0.004), M[key], [0, PL - 0.072, PT / 2]);
        a.visible = b.visible = false;
        accents[key] = [a, b];
        spinner.add(a); spinner.add(b);
      }
      // machined inner face — only seen once a sledge panel turns inside out
      const innerFace = mesh('panel_' + k + '_face', box(PW - 0.006, PL - 0.05, 0.004), M.steel, [0, PL / 2, -PT / 2 - 0.007]);
      spinner.add(innerFace);

      const tip = new THREE.Group(); tip.name = 'tip_' + k; spinner.add(tip);
      tip.position.y = PL;
      tip.add(mesh('tip_' + k + '_taper', cyl(0.008, 0.080, 12, 0.022), M.alu, [0, 0.040, 0]));
      tip.add(mesh('tip_' + k + '_coil', new THREE.TorusGeometry(0.011, 0.003, 8, 16), M.alu, [0, 0.066, 0], [Math.PI / 2, 0, 0]));
      const tipEm = mesh('tip_' + k + '_emit', new THREE.SphereGeometry(0.010, 16, 12), M.cyan, [0, 0.086, 0]);
      tip.add(tipEm);

      const wing = new THREE.Group(); wing.name = 'wing_' + k; spinner.add(wing);
      wing.add(mesh('wing_' + k + '_plate', box(0.150, 0.150, 0.007), M.alu, [0, PL - 0.070, 0.010]));
      wing.add(mesh('wing_' + k + '_edge', box(0.150, 0.012, 0.014), M.dark, [0, PL + 0.005, 0.010]));
      wing.add(mesh('wing_' + k + '_scorch', box(0.130, 0.010, 0.004), M.red, [0, PL - 0.030, 0.015]));
      wing.add(mesh('wing_' + k + '_strut', box(0.010, 0.090, 0.030), M.dark, [0, PL - 0.115, 0.012]));

      const fins = new THREE.Group(); fins.name = 'fins_' + k; spinner.add(fins);
      for (let i = 0; i < 6; i++) {
        fins.add(mesh('fin_' + k + '_' + i, box(PW + 0.014, 0.008, 0.026), M.alu, [0, 0.050 + i * 0.040, 0.016]));
      }
      fins.add(mesh('fin_' + k + '_root', box(PW + 0.006, 0.230, 0.008), M.red, [0, 0.150, 0.006]));

      const strike = new THREE.Group(); strike.name = 'strike_' + k; spinner.add(strike);
      strike.add(mesh('strike_' + k + '_pad', box(PW + 0.044, PL - 0.014, 0.034), M.steel, [0, PL / 2, -PT / 2 - 0.017]));
      strike.add(mesh('strike_' + k + '_face', box(PW + 0.050, PL - 0.040, 0.010), M.bright, [0, PL / 2, -PT / 2 - 0.037]));
      strike.add(mesh('strike_' + k + '_band_a', box(PW + 0.056, 0.018, 0.050), M.amberLit, [0, PL - 0.048, -PT / 2 - 0.026]));
      strike.add(mesh('strike_' + k + '_band_b', box(PW + 0.056, 0.018, 0.050), M.amberLit, [0, 0.048, -PT / 2 - 0.026]));
      // locking hardware: lugs and a key rib biting into the core, so the face
      // reads as clamped mass rather than a plate hanging in the air
      const clamp = new THREE.Group(); clamp.name = 'clamp_' + k; strike.add(clamp);
      clamp.add(mesh('strike_' + k + '_lug_a', box(PW - 0.006, 0.034, 0.034), M.alu, [0, PL - 0.040, PT / 2 + 0.014]));
      clamp.add(mesh('strike_' + k + '_lug_b', box(PW - 0.006, 0.034, 0.034), M.alu, [0, 0.040, PT / 2 + 0.014]));
      clamp.add(mesh('strike_' + k + '_key', box(0.022, PL - 0.090, 0.030), M.dark, [0, PL / 2, PT / 2 + 0.012]));
      for (const sx of [-1, 1]) {
        clamp.add(mesh('strike_' + k + '_bolt_' + sx, cyl(0.007, 0.052, 10), M.bright,
          [sx * 0.019, PL - 0.040, PT / 2 + 0.006], [Math.PI / 2, 0, 0]));
        clamp.add(mesh('strike_' + k + '_bolt2_' + sx, cyl(0.007, 0.052, 10), M.bright,
          [sx * 0.019, 0.040, PT / 2 + 0.006], [Math.PI / 2, 0, 0]));
      }

      panels.push({ k, carrier, radial, slide, hinge, spinner, tip, wing, fins, strike, clamp, accents, tipEm, innerFace });
    }

    return { group, core, cores, panels };
  }

  const endA = makeEnd('a');
  const endB = makeEnd('b');
  endA.group.position.y = PRISM_Y0;
  endB.group.position.y = -PRISM_Y0;
  endB.group.rotation.x = Math.PI;
  pivot.add(endA.group);
  pivot.add(endB.group);

  // ---- state --------------------------------------------------------------
  const state = {
    active: 'a',
    use: false,
    tool: { a: 'sledge', b: 'connector' },
    p: { a: 0, b: 0 },
    want: { a: 0, b: 0 },
    spin: 0, spinTarget: 0
  };

  function applyTool(end, tool) {
    const accentKey = ACCENT[tool];
    for (const key in end.cores) end.cores[key].visible = false;
    for (const p of end.panels) {
      const t = FOLDS[tool](p.k);
      p.target = t;
      p.tip.visible = tool === 'matter' || t.role === 'finger' || t.role === 'rib';
      p.wing.visible = t.role === 'shield';
      p.fins.visible = t.role === 'fin';
      p.strike.visible = t.role === 'strike';
      p.tipEm.material = M[accentKey];
      p.accentOn = accentKey;
      for (const key in p.accents) {
        const on = key === accentKey;
        p.accents[key][0].visible = false;
        p.accents[key][1].visible = false;
        p.accents[key].on = on;
      }
    }
  }

  const MM_F = { r: 0.004, s: -0.055, f: 0.20 };
  const MM_C = { r: 0.000, s: -0.255, f: 0.04 };
  const lerp = (a, b, u) => a + (b - a) * u;

  function tick(end, key, dt) {
    const want = state.want[key];
    const cur = state.p[key];
    const speed = 1 / 0.75;
    let p = cur + Math.sign(want - cur) * Math.min(Math.abs(want - cur), speed * dt);
    state.p[key] = p;

    const e1 = EASE(CL(p / 0.18));
    const e2 = EASE(CL((p - 0.18) / 0.32));
    const e3 = EASE(CL((p - 0.55) / 0.45));

    end.core.visible = e2 > 0.02;
    const tool = state.tool[key];
    for (const k in end.cores) end.cores[k].visible = (k === tool) && e2 > 0.15;

    const idleT = performance.now() / 1000;
    const useNow = state.use && key === state.active && p > 0.96;

    // matter manipulator: one shot per cycle, then the trios trade places
    if (tool === 'matter') {
      if (end.parity === undefined) { end.parity = true; end.shotT = 9; }
      if (useNow) {
        end.shotT += dt;
        if (end.shotT > 1.15) { end.shotT = 0; end.parity = !end.parity; }
      } else if (end.shotT < 9) {
        end.shotT += dt;
      }
    }
    // welder: heat shake; connector: the ring winds the wire on
    end.windB = (end.windB || 0) + (((useNow && tool === 'connector') ? 1 : 0) - (end.windB || 0)) * Math.min(1, dt * 2.4);
    if (useNow && tool === 'connector') end.wind = (end.wind || 0) + dt * 3.8;

    for (const pn of end.panels) {
      let t = pn.target;
      let extraY = 0;
      let rr = t.r, ss = t.s, ff = t.f;

      if (tool === 'matter') {
        if (pn.parity === undefined) pn.parity = pn.k % 2 === 0;
        const want = pn.parity === end.parity ? 1 : 0;
        pn.blend = pn.blend === undefined ? want : pn.blend + (want - pn.blend) * Math.min(1, dt * 2.0);
        rr = lerp(MM_C.r, MM_F.r, pn.blend);
        ss = lerp(MM_C.s, MM_F.s, pn.blend);
        ff = lerp(MM_C.f, MM_F.f, pn.blend);
        if (end.shotT < 0.26 && pn.blend > 0.5) ff += 0.34 * (1 - end.shotT / 0.26);
      }
      if (tool === 'welder' && useNow) {
        ff += Math.sin(idleT * 61 + pn.k * 1.7) * 0.013;
        rr += Math.sin(idleT * 89 + pn.k * 2.3) * 0.0016;
      }
      if (tool === 'connector') {
        extraY = (end.wind || 0);
        ff = ff * (1 - 0.42 * (end.windB || 0));
      }

      pn.carrier.rotation.y = (pn.k * Math.PI) / 3 + (t.y || 0) * e3 + extraY * e3;
      pn.radial.position.z = AF + 0.0555 - 0.050 + 0.008 * e1 + (rr - 0.008) * e3;
      pn.slide.position.y = -0.100 * e2 + (ss + 0.100) * e3;
      pn.hinge.rotation.x = ff * e3 + (tool === 'connector'
        ? Math.sin(idleT * 1.5 + pn.k * 1.05) * 0.05 * e3 * (1 - (end.windB || 0))
        : 0);
      pn.spinner.scale.z = 1 + ((t.th || 1) - 1) * e3;
      pn.spinner.rotation.y = t.spin * e3;
      // accents only exist once the shell has broken open — a fully stowed
      // prism is anonymous bare metal at both ends
      pn.innerFace.visible = !(t.role === 'lamina' || t.role === 'strike');
      // clamp hardware would spear the inner laminae once the head is stacked
      pn.clamp.visible = t.role === 'strike' && !t.th;
      const lit = e1 > 0.04;
      for (const key in pn.accents) {
        const show = lit && pn.accents[key].on && t.role !== 'strike' && t.role !== 'lamina';
        pn.accents[key][0].visible = show;
        pn.accents[key][1].visible = show;
      }
      pn.tip.scale.setScalar(t.role === 'finger' || t.role === 'rib' || tool === 'matter' ? Math.max(0.001, e3) : 1);
      pn.wing.scale.setScalar(t.role === 'shield' ? Math.max(0.001, e3) : 1);
      pn.strike.scale.setScalar(t.role === 'strike' ? Math.max(0.001, e3) : 1);
    }

    if (tool === 'matter') {
      const pay = end.cores.matter.getObjectByName('mm_payload');
      const rg = end.cores.matter.getObjectByName('mm_ring');
      if (pay) { pay.rotation.y += dt * 0.6; pay.rotation.x += dt * 0.25; pay.scale.setScalar(0.4 + 0.6 * e3); }
      if (rg) rg.rotation.z += dt * 0.4;
    }
    if (tool === 'welder') {
      const a = end.cores.welder.getObjectByName('wd_arc');
      const hot = useNow ? 1 : 0;
      if (a) a.scale.setScalar((0.7 + 0.3 * Math.sin(performance.now() / 40) + hot * 0.9) * Math.max(0.001, e3));
    }
  }

  applyTool(endA, state.tool.a);
  applyTool(endB, state.tool.b);
  stage.setObject(root);

  let last = performance.now();
  function loop(now) {
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    tick(endA, 'a', dt);
    tick(endB, 'b', dt);
    const d = state.spinTarget - state.spin;
    if (Math.abs(d) > 0.001) {
      state.spin += Math.sign(d) * Math.min(Math.abs(d), dt * (Math.PI / 0.85));
      pivot.rotation.z = state.spin;
    }
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);

  return {
    setTool(tool) {
      const key = state.active;
      const end = key === 'a' ? endA : endB;
      state.tool[key] = tool;
      state.p[key] = 0;
      applyTool(end, tool);
      state.want[key] = 1;
      return tool;
    },
    setDeployed(on) { state.want[state.active] = on ? 1 : 0; },
    setUse(on) { state.use = !!on; },
    isDeployed() { return state.want[state.active] > 0.5; },
    otherTool() { return state.tool[state.active === 'a' ? 'b' : 'a']; },
    flip() {
      const from = state.active;
      const to = from === 'a' ? 'b' : 'a';
      state.want[from] = 0;
      state.active = to;
      state.want[to] = 1;
      state.spinTarget += Math.PI;
      return state.tool[to];
    },
    setOtherTool(tool) {
      const key = state.active === 'a' ? 'b' : 'a';
      state.tool[key] = tool;
      applyTool(key === 'a' ? endA : endB, tool);
    }
  };
}
