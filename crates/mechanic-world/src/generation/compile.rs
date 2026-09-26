//! Lowers authored expressions to tapes: resolves names, applies domain
//! transforms as coordinate arithmetic, seeds noise, and shares repeats.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use super::WorldgenError;
use super::fields::WorldFields;
use super::noise::NoiseGen;
use super::scatter::{MAX_VARS, ScatterGen, mix};
use super::spec::{Dims, Expr, NoiseDoc, ScatterDoc};
use super::tape::{AXIS_Y, Op, Primitive, Reg, Tape};

static SCATTER_IDS: AtomicU64 = AtomicU64::new(1);

/// Names visible to an expression, innermost scope first.
#[derive(Clone, Copy)]
pub(crate) struct Scope<'a> {
    pub(crate) local: &'a BTreeMap<String, Expr>,
    pub(crate) library: &'a BTreeMap<String, Expr>,
    /// World fields, visible to world-level carve layers only.
    pub(crate) fields: Option<&'a WorldFields>,
}

/// Compiles `expr` into a tape whose extra inputs are named by `inputs`.
pub(crate) fn compile(
    expr: &Expr,
    scope: Scope<'_>,
    inputs: &[String],
    seed: u64,
    context: &str,
) -> Result<Tape, WorldgenError> {
    compile_inputs(expr, scope, inputs, &[], seed, context)
}

/// Compiles `expr` with inputs that vary from point to point, named by
/// `varying`, such as a carve's rock depth.
pub(crate) fn compile_varying(
    expr: &Expr,
    scope: Scope<'_>,
    varying: &[String],
    seed: u64,
    context: &str,
) -> Result<Tape, WorldgenError> {
    compile_inputs(expr, scope, &[], varying, seed, context)
}

fn compile_inputs(
    expr: &Expr,
    scope: Scope<'_>,
    inputs: &[String],
    varying: &[String],
    seed: u64,
    context: &str,
) -> Result<Tape, WorldgenError> {
    let mut compiler = Compiler {
        ops: Vec::new(),
        shared: HashMap::new(),
        noises: HashMap::new(),
        scope,
        inputs,
        varying,
        seed,
        stack: Vec::new(),
        context,
    };
    let domain = [
        compiler.push(Op::X),
        compiler.push(Op::Y),
        compiler.push(Op::Z),
    ];
    let result = compiler.expr(expr, domain)?;
    Ok(eliminate_dead(compiler.ops, result))
}

/// Compiles an expression that must not vary with height.
pub(crate) fn compile_planar(
    expr: &Expr,
    scope: Scope<'_>,
    seed: u64,
    context: &str,
) -> Result<Tape, WorldgenError> {
    let tape = compile(expr, scope, &[], seed, context)?;
    if tape.result_axes() & AXIS_Y != 0 {
        return Err(WorldgenError::Invalid {
            context: context.to_owned(),
            message: "must depend on x and z only; use 2D noise and no Y".to_owned(),
        });
    }
    Ok(tape)
}

struct Compiler<'a> {
    ops: Vec<Op>,
    shared: HashMap<String, Reg>,
    /// One generator per distinct noise, so a definition referred to twice
    /// shares its lookups.
    noises: HashMap<String, Arc<NoiseGen>>,
    scope: Scope<'a>,
    inputs: &'a [String],
    varying: &'a [String],
    seed: u64,
    stack: Vec<String>,
    context: &'a str,
}

impl Compiler<'_> {
    fn push(&mut self, op: Op) -> Reg {
        let key = match &op {
            Op::Scatter(..) => None,
            Op::Noise(domain, noise) => Some(format!("N{domain:?}{:p}", Arc::as_ptr(noise))),
            Op::Fissure(domain, noise) => Some(format!("F{domain:?}{:p}", Arc::as_ptr(noise))),
            other => Some(format!("{other:?}")),
        };
        if let Some(key) = &key
            && let Some(&existing) = self.shared.get(key)
        {
            return existing;
        }
        let reg = Reg::try_from(self.ops.len()).expect("tape fits u32");
        self.ops.push(op);
        if let Some(key) = key {
            self.shared.insert(key, reg);
        }
        reg
    }

    fn constant(&mut self, value: f64) -> Reg {
        self.push(Op::Const(value))
    }

    fn error(&self, message: impl Into<String>) -> WorldgenError {
        WorldgenError::Invalid {
            context: if self.stack.is_empty() {
                self.context.to_owned()
            } else {
                format!("{} (in {})", self.context, self.stack.join(" → "))
            },
            message: message.into(),
        }
    }

    fn fold(
        &mut self,
        terms: &[Expr],
        domain: [Reg; 3],
        combine: impl Fn(Reg, Reg) -> Op,
        what: &str,
    ) -> Result<Reg, WorldgenError> {
        let Some((first, rest)) = terms.split_first() else {
            return Err(self.error(format!("{what} needs at least one operand")));
        };
        let mut result = self.expr(first, domain)?;
        for term in rest {
            let next = self.expr(term, domain)?;
            result = self.push(combine(result, next));
        }
        Ok(result)
    }

    fn noise(&mut self, doc: &NoiseDoc, salt: u64, domain: [Reg; 3]) -> Result<Reg, WorldgenError> {
        let noise = self.noise_gen(doc, salt)?;
        Ok(self.push(Op::Noise(domain, noise)))
    }

    fn noise_gen(&mut self, doc: &NoiseDoc, salt: u64) -> Result<Arc<NoiseGen>, WorldgenError> {
        if !(doc.freq.is_finite() && doc.freq > 0.0) {
            return Err(self.error("noise freq must be positive"));
        }
        let key = format!("{doc:?}{salt}");
        if let Some(noise) = self.noises.get(&key) {
            return Ok(Arc::clone(noise));
        }
        let hashed = key.bytes().fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
        });
        #[expect(
            clippy::cast_possible_truncation,
            reason = "noise seeds are hashes folded to the library's i32"
        )]
        let seed = mix(self.seed ^ hashed) as i32;
        let noise = Arc::new(NoiseGen::new(doc, seed));
        self.noises.insert(key, Arc::clone(&noise));
        Ok(noise)
    }

    fn rotate(&mut self, degrees: f64, first: Reg, second: Reg) -> (Reg, Reg) {
        let (sine, cosine) = degrees.to_radians().sin_cos();
        let (sine, cosine) = (self.constant(sine), self.constant(cosine));
        let first_cos = self.push(Op::Mul(first, cosine));
        let second_sin = self.push(Op::Mul(second, sine));
        let first_sin = self.push(Op::Mul(first, sine));
        let second_cos = self.push(Op::Mul(second, cosine));
        (
            self.push(Op::Add(first_cos, second_sin)),
            self.push(Op::Sub(second_cos, first_sin)),
        )
    }

    fn scaled(
        &mut self,
        of: &Expr,
        factors: [f64; 3],
        domain: [Reg; 3],
    ) -> Result<Reg, WorldgenError> {
        if factors
            .iter()
            .any(|factor| !(factor.is_finite() && *factor > 0.0))
        {
            return Err(self.error("scale factors must be positive"));
        }
        let mut scaled = domain;
        for (axis, factor) in factors.into_iter().enumerate() {
            let inverse = self.constant(1.0 / factor);
            scaled[axis] = self.push(Op::Mul(domain[axis], inverse));
        }
        let value = self.expr(of, scaled)?;
        let correction = self.constant(factors.into_iter().fold(f64::INFINITY, f64::min));
        Ok(self.push(Op::Mul(value, correction)))
    }

    fn scatter(&mut self, doc: &ScatterDoc, domain: [Reg; 3]) -> Result<Reg, WorldgenError> {
        if !(doc.cell > 0.0 && doc.reach > 0.0) {
            return Err(self.error("scatter cell and reach must be positive"));
        }
        if doc.vars.len() > MAX_VARS {
            return Err(self.error(format!("scatter allows at most {MAX_VARS} vars")));
        }
        let mut names: Vec<String> = doc.vars.keys().cloned().collect();
        // A shape may read the rock depth where the enclosing tape can.
        let rock = self.varying.iter().position(|name| name == "rock");
        if rock.is_some() {
            names.push("rock".to_owned());
        }
        let seed = mix(self.seed ^ u64::from(doc.seed).wrapping_mul(0x2545_f491_4f6c_dd1d));
        let context = format!("{} scatter", self.context);
        let planar = |expr: &Expr, part: &str| {
            compile_planar(expr, self.scope, seed, &format!("{context} {part}"))
        };
        let mask = doc
            .mask
            .as_ref()
            .map(|mask| planar(mask, "mask"))
            .transpose()?;
        let ground = doc
            .ground
            .as_ref()
            .map(|ground| planar(ground, "ground"))
            .transpose()?;
        let shape = compile(
            &doc.shape,
            self.scope,
            &names,
            seed,
            &format!("{context} shape"),
        )?;
        let scatter = ScatterGen {
            id: SCATTER_IDS.fetch_add(1, Ordering::Relaxed),
            cell: doc.cell,
            reach: doc.reach,
            jitter: doc.jitter,
            chance: doc.chance,
            lift: doc.lift,
            yaw: doc.yaw,
            tilt_radians: doc.tilt.to_radians(),
            vars: doc.vars.values().copied().collect(),
            seed,
            mask,
            ground,
            shape,
        };
        let rock = match rock {
            Some(index) => self.push(Op::Varying(u32::try_from(index).expect("few inputs"))),
            None => self.constant(0.0),
        };
        Ok(self.push(Op::Scatter(
            [domain[0], domain[1], domain[2], rock],
            Arc::new(scatter),
        )))
    }

    #[expect(
        clippy::too_many_lines,
        reason = "one arm per authored node keeps the lowering in one place"
    )]
    #[expect(
        clippy::many_single_char_names,
        reason = "x, y, z and a, b, t name coordinates and operands"
    )]
    fn expr(&mut self, expr: &Expr, domain: [Reg; 3]) -> Result<Reg, WorldgenError> {
        let [x, y, z] = domain;
        Ok(match expr {
            Expr::X => x,
            Expr::Y => y,
            Expr::Z => z,
            Expr::C(value) => self.constant(*value),
            Expr::Ref(name) => {
                if let Some(index) = self.inputs.iter().position(|input| input == name) {
                    return Ok(self.push(Op::Input(u32::try_from(index).expect("few inputs"))));
                }
                if let Some(index) = self.varying.iter().position(|input| input == name) {
                    return Ok(self.push(Op::Varying(u32::try_from(index).expect("few inputs"))));
                }
                if let Some(field) = self.scope.fields.and_then(|fields| fields.named(name)) {
                    return Ok(self.push(Op::Field([x, z], Arc::clone(field))));
                }
                let definition = self
                    .scope
                    .local
                    .get(name)
                    .or_else(|| self.scope.library.get(name))
                    .ok_or_else(|| self.error(format!("unknown name `{name}`")))?;
                if self.stack.contains(name) {
                    return Err(self.error(format!("`{name}` refers to itself")));
                }
                self.stack.push(name.clone());
                let result = self.expr(definition, domain);
                self.stack.pop();
                result?
            }
            Expr::Add(terms) => self.fold(terms, domain, Op::Add, "Add")?,
            Expr::Mul(terms) => self.fold(terms, domain, Op::Mul, "Mul")?,
            Expr::Min(terms) | Expr::Intersect(terms) => {
                self.fold(terms, domain, Op::Min, "Min")?
            }
            Expr::Max(terms) | Expr::Union(terms) => self.fold(terms, domain, Op::Max, "Max")?,
            Expr::SmoothUnion(k, terms) => {
                let k = *k;
                self.fold(terms, domain, |a, b| Op::SmoothMax(a, b, k), "SmoothUnion")?
            }
            Expr::SmoothIntersect(k, terms) => {
                let k = *k;
                self.fold(
                    terms,
                    domain,
                    |a, b| Op::SmoothMin(a, b, k),
                    "SmoothIntersect",
                )?
            }
            Expr::Sub(a, b) => {
                let (a, b) = (self.expr(a, domain)?, self.expr(b, domain)?);
                self.push(Op::Sub(a, b))
            }
            Expr::Div(a, b) => {
                let (a, b) = (self.expr(a, domain)?, self.expr(b, domain)?);
                self.push(Op::Div(a, b))
            }
            Expr::Subtract(a, b) => {
                let a = self.expr(a, domain)?;
                let b = self.expr(b, domain)?;
                let cut = self.push(Op::Neg(b));
                self.push(Op::Min(a, cut))
            }
            Expr::SmoothSubtract(k, a, b) => {
                let a = self.expr(a, domain)?;
                let b = self.expr(b, domain)?;
                let cut = self.push(Op::Neg(b));
                self.push(Op::SmoothMin(a, cut, *k))
            }
            Expr::Neg(a) => {
                let a = self.expr(a, domain)?;
                self.push(Op::Neg(a))
            }
            Expr::Length(terms) => {
                let Some((first, rest)) = terms.split_first() else {
                    return Err(self.error("Length needs at least one operand"));
                };
                let first = self.expr(first, domain)?;
                let mut sum = self.push(Op::Mul(first, first));
                for term in rest {
                    let term = self.expr(term, domain)?;
                    let squared = self.push(Op::Mul(term, term));
                    sum = self.push(Op::Add(sum, squared));
                }
                self.push(Op::Sqrt(sum))
            }
            Expr::Abs(a) => {
                let a = self.expr(a, domain)?;
                self.push(Op::Abs(a))
            }
            Expr::Sin(a) => {
                let a = self.expr(a, domain)?;
                self.push(Op::Sin(a))
            }
            Expr::Cos(a) => {
                let a = self.expr(a, domain)?;
                self.push(Op::Cos(a))
            }
            Expr::Clamp(a, lo, hi) => {
                if lo > hi {
                    return Err(self.error("Clamp needs min <= max"));
                }
                let a = self.expr(a, domain)?;
                self.push(Op::Clamp(a, *lo, *hi))
            }
            Expr::Remap(a, from, to) => {
                if (from.1 - from.0).abs() <= f64::EPSILON {
                    return Err(self.error("Remap needs a non-empty source range"));
                }
                let scale = (to.1 - to.0) / (from.1 - from.0);
                let a = self.expr(a, domain)?;
                let scale_reg = self.constant(scale);
                let offset = self.constant(from.0.mul_add(-scale, to.0));
                let scaled = self.push(Op::Mul(a, scale_reg));
                self.push(Op::Add(scaled, offset))
            }
            Expr::Smoothstep(a, lo, hi) => {
                if lo >= hi {
                    return Err(self.error("Smoothstep needs from < to"));
                }
                let a = self.expr(a, domain)?;
                self.push(Op::Smoothstep(a, *lo, *hi))
            }
            Expr::Pow(a, exponent) => {
                if !(exponent.is_finite() && *exponent > 0.0) {
                    return Err(self.error("Pow needs a positive exponent"));
                }
                let a = self.expr(a, domain)?;
                self.push(Op::Pow(a, *exponent))
            }
            Expr::Spline(a, points) => {
                if points.is_empty() || points.windows(2).any(|pair| pair[1].0 < pair[0].0) {
                    return Err(self.error("Spline needs points sorted by input"));
                }
                let a = self.expr(a, domain)?;
                self.push(Op::Spline(a, points.clone().into()))
            }
            Expr::Lerp(a, b, t) => {
                let (a, b, t) = (
                    self.expr(a, domain)?,
                    self.expr(b, domain)?,
                    self.expr(t, domain)?,
                );
                self.push(Op::Lerp(a, b, t))
            }
            Expr::Terrace {
                of,
                step,
                sharpness,
            } => {
                if !(*step > 0.0 && (0.0..1.0).contains(sharpness)) {
                    return Err(self.error("Terrace needs step > 0 and sharpness in [0, 1)"));
                }
                let a = self.expr(of, domain)?;
                self.push(Op::Terrace(a, *step, *sharpness))
            }
            Expr::Noise(doc) => self.noise(doc, 0, domain)?,
            Expr::Fissure(doc) => {
                if doc.dims != Dims::Two {
                    return Err(self.error("Fissure needs a 2D noise"));
                }
                let noise = self.noise_gen(doc, 4)?;
                self.push(Op::Fissure([x, z], noise))
            }
            Expr::Height(height) => {
                let height = self.expr(height, domain)?;
                self.push(Op::Sub(height, y))
            }
            Expr::Translate(offset, of) => {
                let offsets = [offset.0, offset.1, offset.2];
                let mut moved = domain;
                for (axis, amount) in offsets.into_iter().enumerate() {
                    if amount != 0.0 {
                        let amount = self.constant(amount);
                        moved[axis] = self.push(Op::Sub(domain[axis], amount));
                    }
                }
                self.expr(of, moved)?
            }
            Expr::Scale(factor, of) => self.scaled(of, [*factor; 3], domain)?,
            Expr::Stretch(factors, of) => {
                self.scaled(of, [factors.0, factors.1, factors.2], domain)?
            }
            Expr::RotateX(degrees, of) => {
                let (new_y, new_z) = self.rotate(*degrees, y, z);
                self.expr(of, [x, new_y, new_z])?
            }
            Expr::RotateY(degrees, of) => {
                let (new_z, new_x) = self.rotate(*degrees, z, x);
                self.expr(of, [new_x, y, new_z])?
            }
            Expr::RotateZ(degrees, of) => {
                let (new_x, new_y) = self.rotate(*degrees, x, y);
                self.expr(of, [new_x, new_y, z])?
            }
            Expr::Twist(rate, of) => {
                let rate = self.constant(rate.to_radians());
                let angle = self.push(Op::Mul(y, rate));
                let sine = self.push(Op::Sin(angle));
                let cosine = self.push(Op::Cos(angle));
                let x_cos = self.push(Op::Mul(x, cosine));
                let z_sin = self.push(Op::Mul(z, sine));
                let x_sin = self.push(Op::Mul(x, sine));
                let z_cos = self.push(Op::Mul(z, cosine));
                let new_x = self.push(Op::Sub(x_cos, z_sin));
                let new_z = self.push(Op::Add(x_sin, z_cos));
                self.expr(of, [new_x, y, new_z])?
            }
            Expr::Repeat(period, of) => {
                let periods = [period.0, period.1, period.2];
                let mut repeated = domain;
                for (axis, period) in periods.into_iter().enumerate() {
                    if period > 0.0 {
                        repeated[axis] = self.push(Op::Repeat(domain[axis], period));
                    }
                }
                self.expr(of, repeated)?
            }
            Expr::Warp { by, vertical, of } => {
                let dx = self.noise(by, 1, domain)?;
                let dz = self.noise(by, 3, domain)?;
                let mut warped = [self.push(Op::Add(x, dx)), y, self.push(Op::Add(z, dz))];
                if *vertical != 0.0 {
                    let dy = self.noise(by, 2, domain)?;
                    let factor = self.constant(*vertical);
                    let scaled = self.push(Op::Mul(dy, factor));
                    warped[1] = self.push(Op::Add(y, scaled));
                }
                self.expr(of, warped)?
            }
            Expr::Domain {
                x: new_x,
                y: new_y,
                z: new_z,
                of,
            } => {
                let mut replaced = domain;
                for (axis, replacement) in [new_x, new_y, new_z].into_iter().enumerate() {
                    if let Some(replacement) = replacement {
                        replaced[axis] = self.expr(replacement, domain)?;
                    }
                }
                self.expr(of, replaced)?
            }
            Expr::Sphere(radius) => {
                let radius = self.expr(radius, domain)?;
                let unused = self.constant(0.0);
                self.push(Op::Solid(Primitive::Sphere, domain, [radius, unused]))
            }
            Expr::Box { half, round } => {
                let unused = self.constant(0.0);
                let half = [half.0, half.1, half.2];
                if half.iter().any(|extent| *extent <= 0.0) || *round < 0.0 {
                    return Err(self.error("Box needs positive half extents"));
                }
                self.push(Op::Solid(
                    Primitive::Box {
                        half,
                        round: round.min(half.iter().copied().fold(f64::INFINITY, f64::min)),
                    },
                    domain,
                    [unused, unused],
                ))
            }
            Expr::Torus { major, minor } => {
                let major = self.expr(major, domain)?;
                let minor = self.expr(minor, domain)?;
                self.push(Op::Solid(Primitive::Torus, domain, [major, minor]))
            }
            Expr::Capsule { a, b, radius } => {
                let radius = self.expr(radius, domain)?;
                let unused = self.constant(0.0);
                self.push(Op::Solid(
                    Primitive::Capsule {
                        a: [a.0, a.1, a.2],
                        b: [b.0, b.1, b.2],
                    },
                    domain,
                    [radius, unused],
                ))
            }
            Expr::Cylinder {
                radius,
                half_height,
            } => {
                let radius = self.expr(radius, domain)?;
                let half_height = self.expr(half_height, domain)?;
                self.push(Op::Solid(
                    Primitive::Cylinder,
                    domain,
                    [radius, half_height],
                ))
            }
            Expr::Cone { radius, height } => {
                let radius = self.expr(radius, domain)?;
                let height = self.expr(height, domain)?;
                self.push(Op::Solid(Primitive::Cone, domain, [radius, height]))
            }
            Expr::Plane { normal, offset } => {
                let normal = [normal.0, normal.1, normal.2];
                let length = normal.iter().map(|value| value * value).sum::<f64>().sqrt();
                if length <= f64::EPSILON {
                    return Err(self.error("Plane needs a non-zero normal"));
                }
                let mut result = self.constant(*offset);
                for (axis, component) in normal.into_iter().enumerate() {
                    if component != 0.0 {
                        let factor = self.constant(component / length);
                        let term = self.push(Op::Mul(domain[axis], factor));
                        result = self.push(Op::Sub(result, term));
                    }
                }
                result
            }
            Expr::Gyroid { period, thickness } => {
                if *period <= 0.0 {
                    return Err(self.error("Gyroid needs a positive period"));
                }
                let thickness = self.expr(thickness, domain)?;
                self.push(Op::Gyroid(domain, *period, thickness))
            }
            Expr::Scatter(doc) => self.scatter(doc, domain)?,
        })
    }
}

/// Drops ops the result never reads and renumbers the rest.
fn eliminate_dead(ops: Vec<Op>, result: Reg) -> Tape {
    let mut live = vec![false; ops.len()];
    live[result as usize] = true;
    for index in (0..=result as usize).rev() {
        if live[index] {
            for input in ops[index].inputs() {
                live[input as usize] = true;
            }
        }
    }
    let mut renumbered = vec![Reg::MAX; ops.len()];
    let mut kept = Vec::new();
    for (index, op) in ops.into_iter().enumerate().take(result as usize + 1) {
        if !live[index] {
            continue;
        }
        renumbered[index] = Reg::try_from(kept.len()).expect("tape fits u32");
        kept.push(remap(op, &renumbered));
    }
    Tape::new(kept)
}

fn remap(op: Op, table: &[Reg]) -> Op {
    let r = |reg: Reg| table[reg as usize];
    match op {
        Op::X | Op::Y | Op::Z | Op::Const(_) | Op::Input(_) | Op::Varying(_) => op,
        Op::Add(a, b) => Op::Add(r(a), r(b)),
        Op::Sub(a, b) => Op::Sub(r(a), r(b)),
        Op::Mul(a, b) => Op::Mul(r(a), r(b)),
        Op::Div(a, b) => Op::Div(r(a), r(b)),
        Op::Neg(a) => Op::Neg(r(a)),
        Op::Min(a, b) => Op::Min(r(a), r(b)),
        Op::Max(a, b) => Op::Max(r(a), r(b)),
        Op::Abs(a) => Op::Abs(r(a)),
        Op::Sqrt(a) => Op::Sqrt(r(a)),
        Op::Sin(a) => Op::Sin(r(a)),
        Op::Cos(a) => Op::Cos(r(a)),
        Op::Clamp(a, lo, hi) => Op::Clamp(r(a), lo, hi),
        Op::Smoothstep(a, lo, hi) => Op::Smoothstep(r(a), lo, hi),
        Op::Pow(a, exponent) => Op::Pow(r(a), exponent),
        Op::Spline(a, points) => Op::Spline(r(a), points),
        Op::Lerp(a, b, t) => Op::Lerp(r(a), r(b), r(t)),
        Op::Terrace(a, step, sharpness) => Op::Terrace(r(a), step, sharpness),
        Op::Repeat(a, period) => Op::Repeat(r(a), period),
        Op::SmoothMax(a, b, k) => Op::SmoothMax(r(a), r(b), k),
        Op::SmoothMin(a, b, k) => Op::SmoothMin(r(a), r(b), k),
        Op::Noise(domain, noise) => Op::Noise(domain.map(r), noise),
        Op::Fissure(domain, noise) => Op::Fissure(domain.map(r), noise),
        Op::Field(domain, field) => Op::Field(domain.map(r), field),
        Op::Solid(primitive, domain, params) => Op::Solid(primitive, domain.map(r), params.map(r)),
        Op::Gyroid(domain, period, thickness) => Op::Gyroid(domain.map(r), period, r(thickness)),
        Op::Scatter(domain, scatter) => Op::Scatter(domain.map(r), scatter),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::{Scope, compile, compile_planar};
    use crate::generation::interval::Interval;
    use crate::generation::spec::{Dims, Expr, Fractal, NoiseDoc, NoiseKind};
    use crate::generation::tape::{AXIS_X, AXIS_Y, AXIS_Z};

    fn noise(dims: Dims) -> Expr {
        Expr::Noise(NoiseDoc {
            kind: NoiseKind::Simplex,
            fractal: Fractal::Fbm,
            octaves: 3,
            freq: 0.02,
            lacunarity: 2.0,
            gain: 0.5,
            amp: 10.0,
            offset: 5.0,
            dims,
            seed: 0,
        })
    }

    #[test]
    fn heightfields_depend_on_y_only_through_the_final_subtraction() {
        let empty = BTreeMap::new();
        let scope = Scope {
            local: &empty,
            library: &empty,
            fields: None,
        };
        let tape = compile(
            &Expr::Height(Box::new(noise(Dims::Two))),
            scope,
            &[],
            1,
            "test",
        )
        .unwrap();
        assert_eq!(tape.result_axes(), AXIS_X | AXIS_Y | AXIS_Z);
        let planar_ops = tape.axes.iter().filter(|axes| **axes & AXIS_Y == 0).count();
        assert!(planar_ops >= tape.ops.len() - 2);
        assert!(compile_planar(&noise(Dims::Three), scope, 1, "test").is_err());
    }

    #[test]
    fn unknown_and_recursive_names_are_reported() {
        let mut local = BTreeMap::new();
        local.insert(
            "loop".to_owned(),
            Expr::Add(vec![Expr::Ref("loop".to_owned())]),
        );
        let library = BTreeMap::new();
        let scope = Scope {
            local: &local,
            library: &library,
            fields: None,
        };
        let missing = compile(&Expr::Ref("nowhere".to_owned()), scope, &[], 0, "biome")
            .unwrap_err()
            .to_string();
        assert!(missing.contains("nowhere"), "{missing}");
        let cycle = compile(&Expr::Ref("loop".to_owned()), scope, &[], 0, "biome")
            .unwrap_err()
            .to_string();
        assert!(cycle.contains("itself"), "{cycle}");
    }

    #[test]
    fn grid_evaluation_matches_point_evaluation_bit_for_bit() {
        let empty = BTreeMap::new();
        let scope = Scope {
            local: &empty,
            library: &empty,
            fields: None,
        };
        let expr = Expr::SmoothUnion(
            2.0,
            vec![
                Expr::Height(Box::new(noise(Dims::Two))),
                Expr::Translate(
                    (3.0, 4.0, -2.0),
                    Box::new(Expr::Torus {
                        major: Box::new(Expr::C(4.0)),
                        minor: Box::new(noise(Dims::Three)),
                    }),
                ),
            ],
        );
        let tape = compile(&expr, scope, &[], 9, "test").unwrap();
        let coordinate = |axis: usize, index: usize| {
            (f64::from(u32::try_from(index).unwrap()) + 0.5) * 0.37 + [1.0, -3.0, 2.0][axis]
        };
        let mut grid = Vec::new();
        tape.eval_grid([5, 4, 3], &coordinate, &mut grid);
        let mut index = 0;
        for k in 0..3 {
            for j in 0..4 {
                for i in 0..5 {
                    let point = [coordinate(0, i), coordinate(1, j), coordinate(2, k)];
                    assert_eq!(grid[index].to_bits(), tape.eval(point, &[]).to_bits());
                    index += 1;
                }
            }
        }
        let bounds = tape.interval(
            [
                Interval::new(1.0, 3.0),
                Interval::new(-3.0, -1.5),
                Interval::new(2.0, 3.1),
            ],
            &[],
        );
        assert!(
            grid.iter()
                .all(|value| bounds.lo <= *value && *value <= bounds.hi)
        );
    }
}
