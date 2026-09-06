//! Numerical port of `fx_pack/fx.js`. Sustained rates are its expected 60 Hz density.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
use bevy::prelude::*;
use std::f32::consts::{PI, TAU};

pub(super) const SHARDS: usize = 1400;
pub(super) const TRACES: usize = 900;
pub(super) const DASHES: usize = 220;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    Sledge,
    Matter,
    Welder,
    Connector,
    Freeze,
}
const RAMPS: [[[u8; 3]; 3]; 5] = [
    [[138, 90, 30], [201, 138, 52], [240, 199, 122]],
    [[30, 140, 116], [47, 216, 180], [168, 245, 228]],
    [[142, 43, 46], [226, 86, 90], [255, 217, 168]],
    [[27, 109, 142], [47, 168, 216], [166, 228, 250]],
    [[110, 42, 140], [188, 79, 232], [233, 191, 247]],
];
pub(super) fn color(kind: Kind, tone: usize) -> [f32; 4] {
    let [r, g, b] = RAMPS[kind as usize][tone];
    Color::srgb_u8(r, g, b).to_linear().to_f32_array()
}
#[derive(Clone, Copy)]
pub(crate) enum Request {
    Sledge { hit: Vec3, normal: Vec3 },
    Matter { hit: Vec3 },
    Freeze { center: Vec3, radius: f32 },
}
#[derive(Clone, Copy)]
pub(crate) struct EmitterFrame {
    pub tool: Kind,
    pub origin: Transform,
    pub target: Vec3,
    pub normal: Vec3,
    pub connector_phase: f32,
    pub connector_plates: [Option<Vec3>; 6],
}
#[derive(Clone, Copy)]
pub(super) struct Shard {
    pub p: Vec3,
    v: Vec3,
    axis: Vec3,
    angle: f32,
    spin: f32,
    pub size: f32,
    g: f32,
    pull: Option<(Vec3, f32)>,
    pub age: f32,
    pub life: f32,
    pub kind: Kind,
}
impl Shard {
    pub fn tone(self) -> usize {
        let f = 1.0 - self.age / self.life;
        if f > 0.62 { 2 } else { usize::from(f > 0.28) }
    }
    pub fn rotation(self) -> Quat {
        Quat::from_axis_angle(self.axis, self.angle)
    }
}
#[derive(Clone, Copy)]
pub(super) struct Trace {
    pub a: Vec3,
    pub b: Vec3,
    v: Vec3,
    age: f32,
    life: f32,
    pub kind: Kind,
    pub tone: usize,
    beam: bool,
}
impl Trace {
    pub fn color(self) -> [f32; 4] {
        color(
            self.kind,
            if 1.0 - self.age / self.life > 0.4 {
                self.tone
            } else {
                self.tone.saturating_sub(1)
            },
        )
    }
}
pub(super) struct Pool<T> {
    pub slots: Box<[Option<T>]>,
    cursor: usize,
}
impl<T: Copy> Pool<T> {
    fn new(cap: usize) -> Self {
        Self {
            slots: vec![None; cap].into_boxed_slice(),
            cursor: 0,
        }
    }
    fn insert(&mut self, value: T) {
        for i in 0..self.slots.len() {
            let j = (self.cursor + i) % self.slots.len();
            if self.slots[j].is_none() {
                self.slots[j] = Some(value);
                self.cursor = (j + 1) % self.slots.len();
                return;
            }
        }
    }
    pub fn clear(&mut self) {
        self.slots.fill(None);
    }
}
pub(super) struct Particles {
    pub shards: Pool<Shard>,
    pub traces: Pool<Trace>,
    pub clock: f32,
    rng: u32,
    credit: [f32; 5],
    previous_tool: Option<Kind>,
}
impl Default for Particles {
    fn default() -> Self {
        Self {
            shards: Pool::new(SHARDS),
            traces: Pool::new(TRACES),
            clock: 0.0,
            rng: 0x6d2b_79f5,
            credit: [0.0; 5],
            previous_tool: None,
        }
    }
}
impl Particles {
    fn rnd(&mut self, a: f32, b: f32) -> f32 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 17;
        self.rng ^= self.rng << 5;
        a + (b - a) * (self.rng as f32 / u32::MAX as f32)
    }
    fn vector(&mut self) -> Vec3 {
        Vec3::new(
            self.rnd(-1.0, 1.0),
            self.rnd(-1.0, 1.0),
            self.rnd(-1.0, 1.0),
        )
    }
    fn direction(&mut self) -> Vec3 {
        self.vector().normalize_or(Vec3::Y)
    }
    #[allow(clippy::too_many_arguments)]
    fn shard(
        &mut self,
        kind: Kind,
        p: Vec3,
        v: Vec3,
        life: Vec2,
        size: Vec2,
        g: f32,
        pull: Option<(Vec3, f32)>,
    ) {
        let s = Shard {
            kind,
            p,
            v,
            life: self.rnd(life.x, life.y),
            size: self.rnd(size.x, size.y),
            g,
            pull,
            axis: self.direction(),
            angle: self.rnd(0.0, TAU),
            spin: if kind == Kind::Matter && pull.is_none() {
                self.rnd(-4.0, 4.0)
            } else {
                self.rnd(-14.0, 14.0)
            },
            age: 0.0,
        };
        self.shards.insert(s);
    }
    fn trace(&mut self, kind: Kind, a: Vec3, b: Vec3, v: Vec3, life: f32, tone: usize) {
        self.traces.insert(Trace {
            kind,
            a,
            b,
            v,
            life,
            tone,
            age: 0.0,
            beam: kind == Kind::Connector,
        });
    }
    pub fn request(&mut self, request: Request, origin: Vec3) {
        match request {
            Request::Sledge { hit, normal } => {
                let normal = normal.normalize_or(Vec3::Y);
                for _ in 0..42 {
                    let v = (self.direction() + normal).normalize_or(normal) * self.rnd(1.1, 3.2);
                    self.shard(
                        Kind::Sledge,
                        hit,
                        v,
                        Vec2::new(0.32, 0.7),
                        Vec2::new(0.011, 0.030),
                        4.2,
                        None,
                    );
                }
                let (u, v) = basis(normal);
                for i in 0..26 {
                    let t = i as f32 / 26.0 * TAU;
                    let d = u * t.cos() + v * t.sin();
                    let tangent = -u * t.sin() + v * t.cos();
                    self.trace(
                        Kind::Sledge,
                        hit + d * 0.06 - tangent * 0.03,
                        hit + d * 0.06 + tangent * 0.03,
                        d * 2.0,
                        0.24,
                        2,
                    );
                }
            }
            Request::Matter { hit } => {
                for _ in 0..22 {
                    let jitter = self.direction() * 0.55;
                    let v = (hit - origin).normalize_or(Vec3::Y) * self.rnd(2.4, 3.6) + jitter;
                    let life = (hit - origin).length() / 3.0;
                    self.shard(
                        Kind::Matter,
                        origin + jitter * 0.06,
                        v,
                        Vec2::new(life.max(0.01), life + 0.1),
                        Vec2::new(0.010, 0.020),
                        0.0,
                        Some((hit, 16.0)),
                    );
                }
                for i in 0..10 {
                    self.trace(
                        Kind::Matter,
                        origin.lerp(hit, i as f32 / 10.0),
                        origin.lerp(hit, (i as f32 + 0.55) / 10.0),
                        Vec3::ZERO,
                        0.14,
                        1,
                    );
                }
            }
            Request::Freeze { center, radius } => {
                for i in 0..34 {
                    let d = self.direction();
                    let p = center + d * radius * 1.55;
                    let v = -d * self.rnd(1.2, 2.2);
                    self.shard(
                        Kind::Freeze,
                        p,
                        v,
                        Vec2::splat(0.42),
                        Vec2::new(0.012, 0.024),
                        0.0,
                        Some((center, 5.0)),
                    );
                    if i % 2 == 0 {
                        self.trace(Kind::Freeze, p, p - d * 0.12, -d * 2.6, 0.26, 2);
                    }
                }
            }
        }
    }
    pub fn advance(&mut self, dt: f32, frame: Option<EmitterFrame>) {
        let dt = dt.clamp(0.0, 0.05);
        self.clock += dt;
        for slot in &mut self.shards.slots {
            if let Some(s) = slot {
                s.age += dt;
                if s.age >= s.life {
                    *slot = None;
                    continue;
                }
                if let Some((target, pull)) = s.pull {
                    let delta = target - s.p;
                    let d = delta.length().max(0.0001);
                    s.v += delta / d * pull * dt * (1.0 / d).min(4.0);
                }
                s.v.y -= s.g * dt;
                s.p += s.v * dt;
                s.angle += s.spin * dt;
            }
        }
        for slot in &mut self.traces.slots {
            if let Some(s) = slot {
                s.age += dt;
                if s.beam || s.age >= s.life {
                    *slot = None;
                } else {
                    s.a += s.v * dt;
                    s.b += s.v * dt;
                }
            }
        }
        let tool = frame.map(|f| f.tool);
        if tool != self.previous_tool {
            self.credit[..4].fill(0.0);
            self.previous_tool = tool;
        }
        if let Some(frame) = frame {
            self.sustain(frame, dt);
        }
    }
    fn count(&mut self, index: usize, rate: f32, dt: f32) -> usize {
        self.credit[index] += rate * dt;
        let n = self.credit[index].floor() as usize;
        self.credit[index] -= n as f32;
        n
    }
    fn sustain(&mut self, f: EmitterFrame, dt: f32) {
        match f.tool {
            Kind::Matter => {
                for i in 0..self.count(0, 120.0, dt) {
                    let th = self.clock * 2.2 + i as f32 * PI;
                    let p = f.origin.transform_point(Vec3::new(
                        th.cos() * 0.055,
                        (th * 0.7).sin() * 0.02,
                        th.sin() * 0.055,
                    ));
                    let v = f.origin.rotation * Vec3::new(-th.sin(), 0.0, th.cos()) * 0.16;
                    self.shard(
                        Kind::Matter,
                        p,
                        v,
                        Vec2::splat(0.30),
                        Vec2::new(0.008, 0.014),
                        0.0,
                        None,
                    );
                }
            }
            Kind::Welder => {
                for _ in 0..self.count(0, 300.0, dt) {
                    let a = f.target + self.vector() * 0.02;
                    let b = f.target + self.vector() * 0.05;
                    self.trace(Kind::Welder, a, b, Vec3::ZERO, 0.05, 2);
                }
                for _ in 0..self.count(1, 180.0, dt) {
                    let (u, v) = basis(f.normal);
                    let dir = (u * self.rnd(-1.0, 1.0)
                        + v * self.rnd(-1.0, 1.0)
                        + f.normal * self.rnd(-0.2, 1.0))
                    .normalize_or(f.normal);
                    let velocity = dir * self.rnd(0.9, 2.6);
                    self.shard(
                        Kind::Welder,
                        f.target,
                        velocity,
                        Vec2::new(0.35, 0.85),
                        Vec2::new(0.006, 0.013),
                        9.0,
                        None,
                    );
                }
                for _ in 0..self.count(2, 30.0, dt) {
                    let a =
                        f.target + Vec3::new(self.rnd(-0.05, 0.05), 0.02, self.rnd(-0.05, 0.05));
                    let v = Vec3::new(self.rnd(-0.05, 0.05), 0.42, self.rnd(-0.05, 0.05));
                    self.trace(Kind::Welder, a, a + Vec3::Y * 0.06, v, 0.55, 0);
                }
            }
            Kind::Connector => self.connector(f),
            _ => {}
        }
    }
    #[allow(clippy::many_single_char_names)]
    fn connector(&mut self, f: EmitterFrame) {
        let origin = f.origin.translation;
        let t = self.clock;
        self.connector_stream(origin, f.target, t);
        for (index, plate) in f.connector_plates.into_iter().enumerate() {
            if let Some(plate) = plate {
                self.connector_stream(plate, f.target, t + index as f32 * 0.37);
            }
        }
        let delta = f.target - origin;
        let axis = delta.normalize_or(Vec3::Y);
        let (u, v) = basis(axis);
        for i in 0..4 {
            let x = (t * 0.55 + i as f32 * 0.25).fract();
            let th = x * 10.0 + f.connector_phase;
            let r = 0.05 * (1.0 - x * 0.5);
            let a = origin + delta * x + (u * th.cos() + v * th.sin()) * r;
            let b = origin + delta * (x + 0.03) + (u * (th + 0.5).cos() + v * (th + 0.5).sin()) * r;
            self.trace(Kind::Connector, a, b, Vec3::ZERO, 0.18, 2);
        }
    }
    #[allow(clippy::many_single_char_names)]
    fn connector_stream(&mut self, origin: Vec3, target: Vec3, t: f32) {
        let mut prev = origin;
        for i in 1..=11 {
            let x = i as f32 / 11.0;
            let j = 0.035 * (x * 9.0 + t * 7.0).sin() * (1.0 - x);
            let p = origin.lerp(target, x)
                + Vec3::new(j, 0.045 * (x * 6.0 - t * 5.0).sin() * (1.0 - x), -j);
            self.trace(
                Kind::Connector,
                prev,
                p,
                Vec3::ZERO,
                0.06,
                if i > 8 { 2 } else { 1 },
            );
            prev = p;
        }
    }
    pub fn wake(&mut self, previous: Vec3, center: Vec3, radius: f32, dt: f32) {
        let dy = center.y - previous.y;
        if dy.abs() < 1.0e-6 {
            self.credit[4] = 0.0;
            return;
        }
        let n = self.count(4, 120.0, dt.clamp(0.0, 0.05));
        for i in 0..n {
            let a =
                previous.lerp(center, (i as f32 + 0.5) / n as f32) + self.vector() * radius * 0.9;
            let b = a - Vec3::Y * dy.signum() * self.rnd(0.06, 0.14);
            self.trace(Kind::Freeze, a, b, -Vec3::Y * dy.signum() * 0.9, 0.30, 1);
        }
    }
    pub fn clear_freeze(&mut self) {
        for slot in &mut self.shards.slots {
            if slot.is_some_and(|s| s.kind == Kind::Freeze) {
                *slot = None;
            }
        }
        for slot in &mut self.traces.slots {
            if slot.is_some_and(|s| s.kind == Kind::Freeze) {
                *slot = None;
            }
        }
        self.credit[4] = 0.0;
    }
    pub fn clear(&mut self) {
        self.shards.clear();
        self.traces.clear();
        self.credit.fill(0.0);
        self.previous_tool = None;
    }
}
fn basis(n: Vec3) -> (Vec3, Vec3) {
    let n = n.normalize_or(Vec3::Y);
    let u = if n.y.abs() > 0.85 { Vec3::X } else { Vec3::Y };
    let a = u.cross(n).normalize();
    (a, n.cross(a).normalize())
}
pub(super) fn pulse(clock: f32) -> f32 {
    let triangle = 1.0 - ((clock * 0.55).fract() * 2.0 - 1.0).abs();
    0.75 + 1.55 * triangle
}
pub(super) fn edges(min: Vec3, max: Vec3) -> [(Vec3, Vec3); 12] {
    let c = [
        Vec3::new(min.x, min.y, min.z),
        Vec3::new(max.x, min.y, min.z),
        Vec3::new(max.x, min.y, max.z),
        Vec3::new(min.x, min.y, max.z),
        Vec3::new(min.x, max.y, min.z),
        Vec3::new(max.x, max.y, min.z),
        Vec3::new(max.x, max.y, max.z),
        Vec3::new(min.x, max.y, max.z),
    ];
    [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    .map(|(a, b)| (c[a], c[b]))
}
pub(super) fn halo(min: Vec3, max: Vec3, clock: f32, mut emit: impl FnMut(Vec3, Vec3, bool)) {
    let outer = edges(min - Vec3::splat(0.075), max + Vec3::splat(0.075));
    let perimeter: f32 = outer.iter().map(|(a, b)| a.distance(*b)).sum();
    // Reserve two clipped segments per edge; never starve the final edges.
    let period = 0.125_f32.max(perimeter / (DASHES - 24) as f32);
    for (a, b) in outer {
        let len = a.distance(b);
        let mut s = (clock * 0.16) % period - period;
        let mut emitted = false;
        while s < len {
            let lo = s.max(0.0);
            let hi = (s + period * 0.44).min(len);
            if hi > lo {
                emit(a.lerp(b, lo / len), a.lerp(b, hi / len), false);
                emitted = true;
            }
            s += period;
        }
        // An edge shorter than the scaled gap still needs a visible hold marker.
        if !emitted {
            let phase = (clock * 0.16 / period).fract();
            let start = phase * 0.56;
            emit(a.lerp(b, start), a.lerp(b, start + 0.44), false);
        }
    }
    for (a, b) in edges(min, max) {
        emit(a, b, true);
    }
}
