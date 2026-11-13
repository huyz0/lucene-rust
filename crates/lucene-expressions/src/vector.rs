//! Column-at-a-time evaluation of a batch: each operation of the expression
//! runs once over the whole batch, a loop the compiler vectorises, instead of
//! the whole expression once per document.
//!
//! Every operand is computed at every document, so a conditional's arms and
//! the right side of `&&`/`||` are evaluated where the closure evaluator
//! would skip them, and the result picked per document. That gives the same
//! values only where evaluation has no effects: column reads and the default
//! functions. An expression calling a custom function is not compiled here.

use super::{chain_kind, java_max, java_min, Bin, ChainKind, Fast, FunctionBody, Node};
use lucene_util::strict_math;

/// A register: an operand of an operation.
#[derive(Clone, Copy, Debug)]
enum Src {
    /// A `double` register.
    F(u32),
    /// A `long` register.
    L(u32),
    /// An `int` register.
    I(u32),
    /// Variable `i`'s column.
    Col(u32),
    /// A `double` constant.
    Fc(f64),
    /// A `long` constant.
    Lc(i64),
    /// An `int` constant.
    Ic(i32),
}

#[derive(Clone, Copy, Debug)]
enum Conv {
    D2L,
    D2I,
    L2D,
    I2D,
    L2I,
    I2L,
}

#[derive(Clone, Debug)]
enum Op {
    /// `double` arithmetic into an `F`.
    Arith(Bin, u32, Src, Src),
    /// A `double` comparison into an `I`.
    Compare(Bin, u32, Src, Src),
    /// A shift of a `long` by an `int` count into an `L`.
    Shift(Bin, u32, Src, Src),
    /// `long` bitwise into an `L`.
    Bitwise(Bin, u32, Src, Src),
    /// `(a != 0) && (b != 0)` / `||` into an `I`.
    Logic(bool, u32, Src, Src),
    Not(u32, Src),
    BwNot(u32, Src),
    Neg(u32, Src),
    Conv(Conv, u32, Src),
    /// `c != 0 ? a : b` into the destination of `a`'s type.
    Select(u32, Src, Src, Src),
    F1(fn(f64) -> f64, u32, Src),
    F2(fn(f64, f64) -> f64, u32, Src, Src),
    F4(fn(f64, f64, f64, f64) -> f64, u32, [Src; 4]),
    /// A function with a slice form that returns the scalar function's bits.
    Slice(fn(&[f64], &mut [f64]), u32, Src),
    Abs(u32, Src),
    Sqrt(u32, Src),
    Floor(u32, Src),
    Ceil(u32, Src),
    Max(u32, Src, Src),
    Min(u32, Src, Src),
}

/// The expression as operations over registers of a batch's length.
#[derive(Debug)]
pub(crate) struct Program {
    ops: Vec<Op>,
    result: Src,
    regs: [u32; 3],
}

/// Registers in use and free, of one type.
#[derive(Default)]
struct Pool {
    count: u32,
    free: Vec<u32>,
}

impl Pool {
    fn get(&mut self) -> u32 {
        self.free.pop().unwrap_or_else(|| {
            self.count += 1;
            self.count - 1
        })
    }
}

#[derive(Default)]
struct Builder {
    ops: Vec<Op>,
    pools: [Pool; 3],
}

#[derive(Clone, Copy, PartialEq)]
enum Ty {
    F,
    L,
    I,
}

impl Builder {
    fn release(&mut self, s: Src) {
        match s {
            Src::F(r) => self.pools[0].free.push(r),
            Src::L(r) => self.pools[1].free.push(r),
            Src::I(r) => self.pools[2].free.push(r),
            _ => {}
        }
    }

    fn reg(&mut self, ty: Ty) -> (u32, Src) {
        match ty {
            Ty::F => {
                let r = self.pools[0].get();
                (r, Src::F(r))
            }
            Ty::L => {
                let r = self.pools[1].get();
                (r, Src::L(r))
            }
            Ty::I => {
                let r = self.pools[2].get();
                (r, Src::I(r))
            }
        }
    }

    /// An operation of `ty` over `args`: the destination is taken before
    /// the operands are freed, so it is never one of them.
    fn op(&mut self, ty: Ty, args: &[Src], f: impl FnOnce(u32) -> Op) -> Src {
        let (r, dst) = self.reg(ty);
        self.ops.push(f(r));
        for a in args {
            self.release(*a);
        }
        dst
    }

    fn conv(&mut self, c: Conv, ty: Ty, a: Src) -> Src {
        // constants convert here
        match (c, a) {
            (Conv::D2L, Src::Fc(v)) => Src::Lc(v as i64),
            (Conv::D2I, Src::Fc(v)) => Src::Ic(v as i32),
            (Conv::L2D, Src::Lc(v)) => Src::Fc(v as f64),
            (Conv::I2D, Src::Ic(v)) => Src::Fc(f64::from(v)),
            (Conv::L2I, Src::Lc(v)) => Src::Ic(v as i32),
            (Conv::I2L, Src::Ic(v)) => Src::Lc(i64::from(v)),
            _ => self.op(ty, &[a], |r| Op::Conv(c, r, a)),
        }
    }

    fn double(&mut self, node: &Node) -> Option<Src> {
        Some(match node {
            Node::Integer(i) => Src::Fc(*i as f64),
            Node::Decimal(d) => Src::Fc(*d),
            Node::Variable(i) => Src::Col(u32::try_from(*i).ok()?),
            Node::Plus(e) => self.double(e)?,
            Node::Call(f, args) => {
                let a: Vec<Src> = args.iter().map(|a| self.double(a)).collect::<Option<_>>()?;
                match (&f.call, f.fast) {
                    (_, Some(fast)) => {
                        let x = a[0];
                        let y = a.get(1).copied().unwrap_or(x);
                        self.op(Ty::F, &a, |r| match fast {
                            Fast::Abs => Op::Abs(r, x),
                            Fast::Sqrt => Op::Sqrt(r, x),
                            Fast::Floor => Op::Floor(r, x),
                            Fast::Ceil => Op::Ceil(r, x),
                            Fast::Max => Op::Max(r, x, y),
                            Fast::Min => Op::Min(r, x, y),
                            Fast::Ln => Op::Slice(strict_math::log_slice, r, x),
                            Fast::Exp => Op::Slice(strict_math::exp_slice, r, x),
                            Fast::Sin => Op::Slice(strict_math::sin_slice, r, x),
                            Fast::Cos => Op::Slice(strict_math::cos_slice, r, x),
                            // `StrictMath.pow(x, 2.0)` is `x * x` after its NaN
                            // check, and `x * x` keeps a NaN `x` as `x + 2.0` does.
                            Fast::Pow if matches!(y, Src::Fc(c) if c == 2.0) => {
                                Op::Arith(Bin::Mul, r, x, x)
                            }
                            Fast::Pow => Op::F2(strict_math::pow, r, x, y),
                        })
                    }
                    (FunctionBody::F1(g), None) => {
                        let (g, x) = (*g, a[0]);
                        self.op(Ty::F, &a, |r| Op::F1(g, r, x))
                    }
                    (FunctionBody::F2(g), None) => {
                        let (g, x, y) = (*g, a[0], a[1]);
                        self.op(Ty::F, &a, |r| Op::F2(g, r, x, y))
                    }
                    (FunctionBody::F4(g), None) => {
                        let (g, xs) = (*g, [a[0], a[1], a[2], a[3]]);
                        self.op(Ty::F, &a, |r| Op::F4(g, r, xs))
                    }
                    (FunctionBody::Any(_), None) => return None,
                }
            }
            Node::Neg(e) => {
                let a = self.double(e)?;
                if let Src::Fc(v) = a {
                    Src::Fc(-v)
                } else {
                    self.op(Ty::F, &[a], |r| Op::Neg(r, a))
                }
            }
            Node::Arith(op, l, r) => {
                let a = self.double(l)?;
                let b = self.double(r)?;
                self.arith(*op, a, b)
            }
            Node::BwNot(_) | Node::Shift(..) | Node::Bitwise(..) => {
                let v = self.long(node)?;
                self.conv(Conv::L2D, Ty::F, v)
            }
            Node::Not(_) | Node::Compare(..) | Node::And(..) | Node::Or(..) => {
                let v = self.int(node)?;
                self.conv(Conv::I2D, Ty::F, v)
            }
            Node::Conditional(c, a, e) => {
                let c = self.int(c)?;
                let a = self.double(a)?;
                let e = self.double(e)?;
                self.op(Ty::F, &[c, a, e], |r| Op::Select(r, c, a, e))
            }
            Node::Chain(first, steps) => match chain_kind(steps) {
                ChainKind::Arith => self.arith_chain(first, steps)?,
                ChainKind::Shift | ChainKind::Bitwise => {
                    let v = self.long_chain(first, steps)?;
                    self.conv(Conv::L2D, Ty::F, v)
                }
                _ => {
                    let v = self.int_chain(first, steps)?;
                    self.conv(Conv::I2D, Ty::F, v)
                }
            },
        })
    }

    fn long(&mut self, node: &Node) -> Option<Src> {
        Some(match node {
            Node::Integer(i) => Src::Lc(*i),
            Node::Decimal(d) => Src::Lc(*d as i64),
            Node::Variable(_) | Node::Call(..) | Node::Neg(_) | Node::Arith(..) => {
                let v = self.double(node)?;
                self.conv(Conv::D2L, Ty::L, v)
            }
            Node::Plus(e) => self.long(e)?,
            Node::BwNot(e) => {
                let a = self.long(e)?;
                self.op(Ty::L, &[a], |r| Op::BwNot(r, a))
            }
            Node::Shift(op, l, n) => {
                let a = self.long(l)?;
                let n = self.int(n)?;
                let op = *op;
                self.op(Ty::L, &[a, n], |r| Op::Shift(op, r, a, n))
            }
            Node::Bitwise(op, l, r) => {
                let a = self.long(l)?;
                let b = self.long(r)?;
                let op = *op;
                self.op(Ty::L, &[a, b], |r| Op::Bitwise(op, r, a, b))
            }
            Node::Not(_) | Node::Compare(..) | Node::And(..) | Node::Or(..) => {
                let v = self.int(node)?;
                self.conv(Conv::I2L, Ty::L, v)
            }
            Node::Conditional(c, a, e) => {
                let c = self.int(c)?;
                let a = self.long(a)?;
                let e = self.long(e)?;
                self.op(Ty::L, &[c, a, e], |r| Op::Select(r, c, a, e))
            }
            Node::Chain(first, steps) => match chain_kind(steps) {
                ChainKind::Arith => {
                    let v = self.arith_chain(first, steps)?;
                    self.conv(Conv::D2L, Ty::L, v)
                }
                ChainKind::Shift | ChainKind::Bitwise => self.long_chain(first, steps)?,
                _ => {
                    let v = self.int_chain(first, steps)?;
                    self.conv(Conv::I2L, Ty::L, v)
                }
            },
        })
    }

    fn int(&mut self, node: &Node) -> Option<Src> {
        Some(match node {
            Node::Integer(i) => Src::Ic(*i as i32),
            Node::Decimal(d) => Src::Ic(*d as i32),
            Node::Variable(_) | Node::Call(..) | Node::Neg(_) | Node::Arith(..) => {
                let v = self.double(node)?;
                self.conv(Conv::D2I, Ty::I, v)
            }
            Node::Plus(e) => self.int(e)?,
            Node::BwNot(_) | Node::Shift(..) | Node::Bitwise(..) => {
                let v = self.long(node)?;
                self.conv(Conv::L2I, Ty::I, v)
            }
            Node::Not(e) => {
                let a = self.int(e)?;
                self.op(Ty::I, &[a], |r| Op::Not(r, a))
            }
            Node::Compare(op, l, r) => {
                let a = self.double(l)?;
                let b = self.double(r)?;
                let op = *op;
                self.op(Ty::I, &[a, b], |r| Op::Compare(op, r, a, b))
            }
            Node::And(l, r) | Node::Or(l, r) => {
                let and = matches!(node, Node::And(..));
                let a = self.int(l)?;
                let b = self.int(r)?;
                self.op(Ty::I, &[a, b], |r| Op::Logic(and, r, a, b))
            }
            Node::Conditional(c, a, e) => {
                let c = self.int(c)?;
                let a = self.int(a)?;
                let e = self.int(e)?;
                self.op(Ty::I, &[c, a, e], |r| Op::Select(r, c, a, e))
            }
            Node::Chain(first, steps) => match chain_kind(steps) {
                ChainKind::Arith => {
                    let v = self.arith_chain(first, steps)?;
                    self.conv(Conv::D2I, Ty::I, v)
                }
                ChainKind::Shift | ChainKind::Bitwise => {
                    let v = self.long_chain(first, steps)?;
                    self.conv(Conv::L2I, Ty::I, v)
                }
                _ => self.int_chain(first, steps)?,
            },
        })
    }

    fn arith(&mut self, op: Bin, a: Src, b: Src) -> Src {
        self.op(Ty::F, &[a, b], |r| Op::Arith(op, r, a, b))
    }

    fn arith_chain(&mut self, first: &Node, steps: &[(Bin, Node)]) -> Option<Src> {
        let mut v = self.double(first)?;
        for (op, a) in steps {
            let b = self.double(a)?;
            v = self.arith(*op, v, b);
        }
        Some(v)
    }

    fn long_chain(&mut self, first: &Node, steps: &[(Bin, Node)]) -> Option<Src> {
        let mut v = self.long(first)?;
        let shift = chain_kind(steps) == ChainKind::Shift;
        for (op, a) in steps {
            let op = *op;
            v = if shift {
                let n = self.int(a)?;
                self.op(Ty::L, &[v, n], |r| Op::Shift(op, r, v, n))
            } else {
                let b = self.long(a)?;
                self.op(Ty::L, &[v, b], |r| Op::Bitwise(op, r, v, b))
            };
        }
        Some(v)
    }

    fn int_chain(&mut self, first: &Node, steps: &[(Bin, Node)]) -> Option<Src> {
        match chain_kind(steps) {
            ChainKind::Compare => {
                // each comparison's `int` the next one's left `double`
                let mut v = self.double(first)?;
                let last = steps.len() - 1;
                for (k, (op, a)) in steps.iter().enumerate() {
                    let b = self.double(a)?;
                    let op = *op;
                    let r = self.op(Ty::I, &[v, b], |r| Op::Compare(op, r, v, b));
                    if k == last {
                        return Some(r);
                    }
                    v = self.conv(Conv::I2D, Ty::F, r);
                }
                unreachable!("a chain has steps")
            }
            kind => {
                let and = kind == ChainKind::And;
                let mut v = self.int(first)?;
                for (_, a) in steps {
                    let b = self.int(a)?;
                    v = self.op(Ty::I, &[v, b], |r| Op::Logic(and, r, v, b));
                }
                Some(v)
            }
        }
    }
}

/// `root` as a program, or `None` where it calls a custom function.
pub(crate) fn compile(root: &Node) -> Option<Program> {
    let mut b = Builder::default();
    let result = b.double(root)?;
    Some(Program {
        ops: b.ops,
        result,
        regs: [b.pools[0].count, b.pools[1].count, b.pools[2].count],
    })
}

/// The registers of an evaluation, reused from batch to batch: one vector
/// of `n` values per register.
#[derive(Default)]
pub(crate) struct Scratch {
    f: Vec<Vec<f64>>,
    l: Vec<Vec<i64>>,
    i: Vec<Vec<i32>>,
}

/// One operand over a batch: a slice of `n`, or a constant.
enum Arg<'a, T> {
    S(&'a [T]),
    C(T),
}

/// The registers of one run. An operation's destination is taken out of its
/// pool while the operation reads its operands, and put back after, so a
/// destination is never also an operand (the compiler gives every operation
/// a fresh register).
struct Regs<'a> {
    s: &'a mut Scratch,
    cols: &'a [f64],
    n: usize,
}

impl Regs<'_> {
    fn f(&self, s: Src) -> Arg<'_, f64> {
        match s {
            Src::F(r) => Arg::S(self.reg(&self.s.f[r as usize])),
            Src::Col(c) => Arg::S(&self.cols[c as usize * self.n..][..self.n]),
            Src::Fc(v) => Arg::C(v),
            _ => unreachable!("a double operand"),
        }
    }

    fn l(&self, s: Src) -> Arg<'_, i64> {
        match s {
            Src::L(r) => Arg::S(self.reg(&self.s.l[r as usize])),
            Src::Lc(v) => Arg::C(v),
            _ => unreachable!("a long operand"),
        }
    }

    fn i(&self, s: Src) -> Arg<'_, i32> {
        match s {
            Src::I(r) => Arg::S(self.reg(&self.s.i[r as usize])),
            Src::Ic(v) => Arg::C(v),
            _ => unreachable!("an int operand"),
        }
    }

    fn reg<'v, T>(&self, v: &'v [T]) -> &'v [T] {
        debug_assert_eq!(
            v.len(),
            self.n,
            "an operand register is its own destination"
        );
        v
    }
}

/// Runs `$body` with register `$d` of pool `$pool` taken out as `$dest`, then
/// puts it back.
macro_rules! with_dest {
    ($r:ident, $pool:ident, $d:expr, |$dest:ident| $body:expr) => {{
        let slot = $d as usize;
        let mut $dest = std::mem::take(&mut $r.s.$pool[slot]);
        $body;
        $r.s.$pool[slot] = $dest;
    }};
}

#[inline(always)]
fn map1<A: Copy, T: Copy>(out: &mut [T], a: Arg<'_, A>, f: impl Fn(A) -> T) {
    match a {
        Arg::S(a) => {
            for (o, &x) in out.iter_mut().zip(a) {
                *o = f(x);
            }
        }
        Arg::C(x) => {
            let v = f(x);
            for o in out {
                *o = v;
            }
        }
    }
}

#[inline(always)]
fn map2<A: Copy, B: Copy, T: Copy>(
    out: &mut [T],
    a: Arg<'_, A>,
    b: Arg<'_, B>,
    f: impl Fn(A, B) -> T,
) {
    match (a, b) {
        (Arg::S(a), Arg::S(b)) => {
            for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
                *o = f(x, y);
            }
        }
        (Arg::S(a), Arg::C(y)) => {
            for (o, &x) in out.iter_mut().zip(a) {
                *o = f(x, y);
            }
        }
        (Arg::C(x), Arg::S(b)) => {
            for (o, &y) in out.iter_mut().zip(b) {
                *o = f(x, y);
            }
        }
        (Arg::C(x), Arg::C(y)) => {
            for o in out {
                *o = f(x, y);
            }
        }
    }
}

/// `c != 0 ? a : b` per document, branch-free where all three are slices.
#[inline(always)]
fn select<T: Copy>(out: &mut [T], c: Arg<'_, i32>, a: Arg<'_, T>, b: Arg<'_, T>) {
    if let (Arg::S(c), Arg::S(a), Arg::S(b)) = (&c, &a, &b) {
        for (((o, &c), &x), &y) in out.iter_mut().zip(*c).zip(*a).zip(*b) {
            *o = if c != 0 { x } else { y };
        }
        return;
    }
    for (j, o) in out.iter_mut().enumerate() {
        *o = if at(&c, j) != 0 { at(&a, j) } else { at(&b, j) };
    }
}

fn at<T: Copy>(a: &Arg<'_, T>, j: usize) -> T {
    match a {
        Arg::S(s) => s[j],
        Arg::C(c) => *c,
    }
}

/// `count` registers of `n` values each.
fn resize_pool<T: Copy>(pool: &mut Vec<Vec<T>>, count: usize, n: usize, zero: T) {
    pool.resize_with(count, Vec::new);
    for reg in pool.iter_mut() {
        reg.resize(n, zero);
    }
}

impl Program {
    /// The values over a batch of `out.len()`: `cols[i * n + j]` variable `i`
    /// at the `j`th.
    pub(crate) fn run(&self, cols: &[f64], out: &mut [f64], scratch: &mut Scratch) {
        let n = out.len();
        resize_pool(&mut scratch.f, self.regs[0] as usize, n, 0.0);
        resize_pool(&mut scratch.l, self.regs[1] as usize, n, 0);
        resize_pool(&mut scratch.i, self.regs[2] as usize, n, 0);
        let r = Regs {
            s: scratch,
            cols,
            n,
        };
        for op in &self.ops {
            match *op {
                Op::Arith(op, d, a, b) => with_dest!(r, f, d, |d| {
                    let (a, b) = (r.f(a), r.f(b));
                    match op {
                        Bin::Mul => map2(&mut d, a, b, |x, y| x * y),
                        Bin::Div => map2(&mut d, a, b, |x, y| x / y),
                        Bin::Rem => map2(&mut d, a, b, |x, y| x % y),
                        Bin::Add => map2(&mut d, a, b, |x, y| x + y),
                        _ => map2(&mut d, a, b, |x, y| x - y),
                    }
                }),
                Op::Compare(op, d, a, b) => with_dest!(r, i, d, |d| {
                    let (a, b) = (r.f(a), r.f(b));
                    match op {
                        Bin::Lt => map2(&mut d, a, b, |x, y| i32::from(x < y)),
                        Bin::Lte => map2(&mut d, a, b, |x, y| i32::from(x <= y)),
                        Bin::Gt => map2(&mut d, a, b, |x, y| i32::from(x > y)),
                        Bin::Gte => map2(&mut d, a, b, |x, y| i32::from(x >= y)),
                        Bin::Eq => map2(&mut d, a, b, |x, y| i32::from(x == y)),
                        _ => map2(&mut d, a, b, |x, y| i32::from(x != y)),
                    }
                }),
                Op::Shift(op, d, a, b) => with_dest!(r, l, d, |d| {
                    let (a, b) = (r.l(a), r.i(b));
                    match op {
                        Bin::Lsh => map2(&mut d, a, b, |x, n| x.wrapping_shl((n & 63) as u32)),
                        Bin::Rsh => map2(&mut d, a, b, |x, n| x >> (n & 63)),
                        _ => map2(&mut d, a, b, |x, n| ((x as u64) >> (n & 63)) as i64),
                    }
                }),
                Op::Bitwise(op, d, a, b) => with_dest!(r, l, d, |d| {
                    let (a, b) = (r.l(a), r.l(b));
                    match op {
                        Bin::BwAnd => map2(&mut d, a, b, |x, y| x & y),
                        Bin::BwXor => map2(&mut d, a, b, |x, y| x ^ y),
                        _ => map2(&mut d, a, b, |x, y| x | y),
                    }
                }),
                Op::Logic(and, d, a, b) => with_dest!(r, i, d, |d| {
                    let (a, b) = (r.i(a), r.i(b));
                    if and {
                        map2(&mut d, a, b, |x, y| i32::from((x != 0) & (y != 0)));
                    } else {
                        map2(&mut d, a, b, |x, y| i32::from((x != 0) | (y != 0)));
                    }
                }),
                Op::Not(d, a) => {
                    with_dest!(r, i, d, |d| map1(&mut d, r.i(a), |x| i32::from(x == 0)))
                }
                Op::BwNot(d, a) => with_dest!(r, l, d, |d| map1(&mut d, r.l(a), |x| !x)),
                Op::Neg(d, a) => with_dest!(r, f, d, |d| map1(&mut d, r.f(a), |x| -x)),
                Op::Abs(d, a) => with_dest!(r, f, d, |d| map1(&mut d, r.f(a), f64::abs)),
                Op::Sqrt(d, a) => with_dest!(r, f, d, |d| map1(&mut d, r.f(a), f64::sqrt)),
                Op::Floor(d, a) => with_dest!(r, f, d, |d| map1(&mut d, r.f(a), f64::floor)),
                Op::Ceil(d, a) => with_dest!(r, f, d, |d| map1(&mut d, r.f(a), f64::ceil)),
                Op::Conv(c, d, a) => match c {
                    Conv::D2L => with_dest!(r, l, d, |d| map1(&mut d, r.f(a), |x| x as i64)),
                    Conv::D2I => with_dest!(r, i, d, |d| map1(&mut d, r.f(a), |x| x as i32)),
                    Conv::L2D => with_dest!(r, f, d, |d| map1(&mut d, r.l(a), |x| x as f64)),
                    Conv::I2D => with_dest!(r, f, d, |d| map1(&mut d, r.i(a), f64::from)),
                    Conv::L2I => with_dest!(r, i, d, |d| map1(&mut d, r.l(a), |x| x as i32)),
                    Conv::I2L => with_dest!(r, l, d, |d| map1(&mut d, r.i(a), i64::from)),
                },
                Op::Select(d, c, a, b) => match a {
                    Src::F(_) | Src::Col(_) | Src::Fc(_) => {
                        with_dest!(r, f, d, |d| select(&mut d, r.i(c), r.f(a), r.f(b)))
                    }
                    Src::L(_) | Src::Lc(_) => {
                        with_dest!(r, l, d, |d| select(&mut d, r.i(c), r.l(a), r.l(b)))
                    }
                    Src::I(_) | Src::Ic(_) => {
                        with_dest!(r, i, d, |d| select(&mut d, r.i(c), r.i(a), r.i(b)))
                    }
                },
                Op::F1(f, d, a) => with_dest!(r, f, d, |d| map1(&mut d, r.f(a), f)),
                Op::Slice(f, d, a) => with_dest!(r, f, d, |d| match r.f(a) {
                    Arg::S(a) => f(a, &mut d),
                    Arg::C(x) => {
                        let mut v = [0.0];
                        f(&[x], &mut v);
                        d.fill(v[0]);
                    }
                }),
                Op::F2(f, d, a, b) => with_dest!(r, f, d, |d| {
                    let (a, b) = (r.f(a), r.f(b));
                    map2(&mut d, a, b, f);
                }),
                Op::Max(d, a, b) => with_dest!(r, f, d, |d| map2(&mut d, r.f(a), r.f(b), java_max)),
                Op::Min(d, a, b) => with_dest!(r, f, d, |d| map2(&mut d, r.f(a), r.f(b), java_min)),
                Op::F4(f, d, xs) => with_dest!(r, f, d, |d| {
                    let [a, b, c, e] = xs.map(|s| r.f(s));
                    for (j, o) in d.iter_mut().enumerate() {
                        *o = f(at(&a, j), at(&b, j), at(&c, j), at(&e, j));
                    }
                }),
            }
        }
        match r.f(self.result) {
            Arg::S(s) => out.copy_from_slice(s),
            Arg::C(c) => out.fill(c),
        }
    }
}
