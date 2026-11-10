//! A typed register machine for a [specialised](crate::specialize) script.
//!
//! Once the query's facts are folded in, a typical search script is
//! arithmetic on doc values, constants and the score, where every register
//! holds one primitive class (or a string) for every document. [`lower`]
//! turns such a script into flat, unboxed instructions on `u64` registers
//! (ints, longs, floats, doubles and booleans by their bits), with Java's
//! semantics for each operation; a script with anything else stays on the
//! [`Vm`](crate::vm::Vm), which runs it as it is.

use std::collections::HashSet;
use std::rc::Rc;

use crate::api::doc::DocKind;
use crate::api::math;
use crate::types::{prim_of, Prim};
use crate::value::Value;
use crate::vm::{
    BinOp, CastOp, CmpOp, DocAccess, Env, Op, Program, Reg, Tag, Throw, UnOp, LOOP_LIMIT_MESSAGE,
};

/// What a register holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Ty {
    P(Prim),
    Str,
}

/// No index register: the field's first value.
const FIRST: Reg = Reg::MAX;

#[derive(Debug, Clone)]
enum I {
    K(Reg, u64),
    KStr(Reg, Rc<str>),
    Mov(Reg, Reg),
    MovRef(Reg, Reg),
    Int(BinOp, Reg, Reg, Reg),
    Long(BinOp, Reg, Reg, Reg),
    Float(BinOp, Reg, Reg, Reg),
    Double(BinOp, Reg, Reg, Reg),
    Bool(BinOp, Reg, Reg, Reg),
    AddD(Reg, Reg, Reg),
    SubD(Reg, Reg, Reg),
    MulD(Reg, Reg, Reg),
    DivD(Reg, Reg, Reg),
    CmpInt(CmpOp, Reg, Reg, Reg),
    CmpLong(CmpOp, Reg, Reg, Reg),
    CmpFloat(CmpOp, Reg, Reg, Reg),
    CmpDouble(CmpOp, Reg, Reg, Reg),
    CmpBool(CmpOp, Reg, Reg, Reg),
    CmpStr(CmpOp, Reg, Reg, Reg),
    Conv(Reg, Reg, Prim, Prim),
    Neg(Prim, Reg, Reg),
    BwNot(Prim, Reg, Reg),
    Not(Reg, Reg),
    DocLong(Reg, u32, Reg),
    DocDouble(Reg, u32, Reg),
    DocBool(Reg, u32, Reg),
    DocStr(Reg, u32, Reg),
    DocCount(Reg, u32),
    DocFlip(Reg, u32, Reg),
    Score(Reg),
    Math1(Reg, Reg, fn(f64) -> f64),
    Math2(Reg, Reg, Reg, fn(f64, f64) -> f64),
    Round(Reg, Reg),
    Jmp(u32),
    JmpF(Reg, u32),
    Count,
    Ret(Reg, Ty),
    RetNull,
    /// The source offset of what follows ([`Op::At`]); lowering folds it
    /// into [`Typed::locs`].
    At(i32),
}

/// A script lowered to typed instructions.
#[derive(Debug)]
pub struct Typed {
    code: Box<[I]>,
    init: Vec<Init>,
    /// Whether [`Typed::run_batch`] can run it: straight-line code on
    /// numbers, ending in one `double` return.
    batchable: bool,
    regs: usize,
    max_loop_counter: i32,
    /// Each instruction's source offset, as the class's line table maps it.
    locs: Box<[i32]>,
}

/// The registers of a [`Typed`] run, kept across runs.
#[derive(Debug, Default)]
pub struct Regs {
    bits: Vec<u64>,
    refs: Vec<Value>,
    /// The instruction running, or the one that threw.
    pc: usize,
}

/// Lowers `prog`'s `execute` for the fields of [`crate::specialize::Facts`];
/// `None` when any live operation has no typed form.
pub fn lower(prog: &Program, fields: &[(&str, Option<DocKind>)]) -> Option<Typed> {
    let f = prog.funcs.get(prog.entry as usize)?;
    let mut l = Lower {
        fields,
        dead: dead_ops(&f.body, f.regs),
        code: Vec::new(),
        ty: vec![None; f.regs],
        loops: Vec::new(),
    };
    l.block(&f.body)?;
    l.code.push(I::RetNull);
    let (code, init, locs) = fold_moves(l.code);
    let batchable = batchable(&code);
    Some(Typed {
        code: code.into_boxed_slice(),
        init,
        batchable,
        regs: l.ty.len(),
        max_loop_counter: f.max_loop_counter,
        locs: locs.into_boxed_slice(),
    })
}

/// Whether every instruction has a column form and every jump goes
/// forward (branches, no loops), so each document's path runs once.
fn batchable(code: &[I]) -> bool {
    code.iter().enumerate().all(|(at, i)| match i {
        I::Jmp(t) | I::JmpF(_, t) => *t as usize > at,
        I::Ret(_, t) => *t == Ty::P(Prim::Double),
        I::DocLong(_, _, x) | I::DocDouble(_, _, x) => *x == FIRST,
        I::Mov(..)
        | I::Int(..)
        | I::Long(..)
        | I::Float(..)
        | I::Double(..)
        | I::Bool(..)
        | I::AddD(..)
        | I::SubD(..)
        | I::MulD(..)
        | I::DivD(..)
        | I::CmpInt(..)
        | I::CmpLong(..)
        | I::CmpFloat(..)
        | I::CmpDouble(..)
        | I::CmpBool(..)
        | I::Conv(..)
        | I::Neg(..)
        | I::BwNot(..)
        | I::Not(..)
        | I::DocCount(..)
        | I::Score(_)
        | I::Math1(..)
        | I::Math2(..)
        | I::Round(..)
        | I::RetNull => true,
        _ => false,
    })
}

/// What a column-batch run reads, per document of the batch.
pub trait BatchEnv {
    /// Field `field`'s value counts and first values (`.value`, as a
    /// `Longs` field's longs or a `Doubles` field's doubles by their bits;
    /// anything where a document has none).
    fn column(&mut self, field: usize) -> Result<(&[u64], &[u64]), Throw>;
    /// `get_score()` for each document, as the doubles' bits.
    fn scores(&mut self, out: &mut [u64]) -> Result<(), Throw>;
}

/// The registers of a column-batch run: each register a column.
#[derive(Debug, Default)]
pub struct BatchRegs {
    width: usize,
    cols: Vec<u64>,
}

/// What a run returns.
enum Out {
    Bits(u64, Prim),
    Value(Value),
}

/// A register's initial value, set once for every run.
#[derive(Debug, Clone)]
enum Init {
    Bits(Reg, u64),
    Str(Reg, Rc<str>),
}

impl I {
    /// The register the instruction assigns.
    fn dst(&self) -> Option<Reg> {
        Some(match self {
            I::K(d, _) | I::KStr(d, _) | I::Mov(d, _) | I::MovRef(d, _) => *d,
            I::Int(_, d, ..)
            | I::Long(_, d, ..)
            | I::Float(_, d, ..)
            | I::Double(_, d, ..)
            | I::Bool(_, d, ..)
            | I::CmpInt(_, d, ..)
            | I::CmpLong(_, d, ..)
            | I::CmpFloat(_, d, ..)
            | I::CmpDouble(_, d, ..)
            | I::CmpBool(_, d, ..)
            | I::CmpStr(_, d, ..)
            | I::Neg(_, d, _)
            | I::BwNot(_, d, _) => *d,
            I::AddD(d, ..)
            | I::SubD(d, ..)
            | I::MulD(d, ..)
            | I::DivD(d, ..)
            | I::Conv(d, ..)
            | I::Not(d, _)
            | I::DocLong(d, ..)
            | I::DocDouble(d, ..)
            | I::DocBool(d, ..)
            | I::DocStr(d, ..)
            | I::DocCount(d, _)
            | I::DocFlip(d, ..)
            | I::Score(d)
            | I::Math1(d, ..)
            | I::Math2(d, ..)
            | I::Round(d, _) => *d,
            I::Jmp(_) | I::JmpF(..) | I::Count | I::Ret(..) | I::RetNull | I::At(_) => return None,
        })
    }

    /// Calls `f` on each register the instruction reads.
    fn sources(&mut self, f: &mut impl FnMut(&mut Reg)) {
        match self {
            I::K(..) | I::KStr(..) | I::DocCount(..) | I::Score(_) => {}
            I::Jmp(_) | I::Count | I::RetNull | I::At(_) => {}
            I::Mov(_, s) | I::MovRef(_, s) | I::Conv(_, s, ..) | I::Not(_, s) => f(s),
            I::Neg(_, _, s) | I::BwNot(_, _, s) | I::Math1(_, s, _) | I::Round(_, s) => f(s),
            I::Int(_, _, a, b)
            | I::Long(_, _, a, b)
            | I::Float(_, _, a, b)
            | I::Double(_, _, a, b)
            | I::Bool(_, _, a, b)
            | I::CmpInt(_, _, a, b)
            | I::CmpLong(_, _, a, b)
            | I::CmpFloat(_, _, a, b)
            | I::CmpDouble(_, _, a, b)
            | I::CmpBool(_, _, a, b)
            | I::CmpStr(_, _, a, b)
            | I::AddD(_, a, b)
            | I::SubD(_, a, b)
            | I::MulD(_, a, b)
            | I::DivD(_, a, b)
            | I::Math2(_, a, b, _) => {
                f(a);
                f(b);
            }
            I::DocLong(_, _, x)
            | I::DocDouble(_, _, x)
            | I::DocBool(_, _, x)
            | I::DocStr(_, _, x) => {
                if *x != FIRST {
                    f(x)
                }
            }
            I::DocFlip(_, _, x) | I::JmpF(x, _) | I::Ret(x, _) => f(x),
        }
    }
}

/// Moves constants assigned once into the registers' initial values, and
/// drops each move between registers both assigned once, reading the
/// source in its place: Java's definite assignment means no read of the
/// copy can come before the copy, so the two always hold the same value.
fn fold_moves(code: Vec<I>) -> (Vec<I>, Vec<Init>, Vec<i32>) {
    let mut defs: std::collections::HashMap<Reg, u32> = std::collections::HashMap::new();
    for ins in &code {
        if let Some(d) = ins.dst() {
            *defs.entry(d).or_default() += 1;
        }
    }
    let once = |r: Reg| defs.get(&r) == Some(&1);
    let mut alias: std::collections::HashMap<Reg, Reg> = std::collections::HashMap::new();
    let mut init = Vec::new();
    let mut keep = vec![true; code.len()];
    for (at, ins) in code.iter().enumerate() {
        match ins {
            I::K(d, v) if once(*d) => {
                init.push(Init::Bits(*d, *v));
                keep[at] = false;
            }
            I::KStr(d, v) if once(*d) => {
                init.push(Init::Str(*d, v.clone()));
                keep[at] = false;
            }
            I::Mov(d, s) | I::MovRef(d, s) if once(*d) && once(*s) => {
                alias.insert(*d, *s);
                keep[at] = false;
            }
            I::At(_) => keep[at] = false,
            _ => {}
        }
    }
    let resolve = |mut r: Reg| {
        while let Some(&s) = alias.get(&r) {
            r = s;
        }
        r
    };
    // Old positions to new: a jump to a dropped instruction lands on the next kept one.
    let mut at_new = Vec::with_capacity(code.len() + 1);
    let mut n = 0u32;
    for k in &keep {
        at_new.push(n);
        n += u32::from(*k);
    }
    at_new.push(n);
    let mut out = Vec::with_capacity(n as usize);
    let mut locs = Vec::with_capacity(n as usize);
    let mut loc = -1;
    for (mut ins, k) in code.into_iter().zip(keep) {
        if let I::At(o) = ins {
            loc = o;
        }
        if !k {
            continue;
        }
        locs.push(loc);
        ins.sources(&mut |r| *r = resolve(*r));
        match &mut ins {
            I::Jmp(t) | I::JmpF(_, t) => *t = at_new[*t as usize],
            _ => {}
        }
        out.push(ins);
    }
    (out, init, locs)
}

struct Loop {
    breaks: Vec<usize>,
    continues: Vec<usize>,
}

struct Lower<'a> {
    fields: &'a [(&'a str, Option<DocKind>)],
    dead: HashSet<*const Op>,
    code: Vec<I>,
    ty: Vec<Option<Ty>>,
    loops: Vec<Loop>,
}

impl Lower<'_> {
    fn get(&self, r: Reg) -> Option<Ty> {
        self.ty.get(r as usize).copied().flatten()
    }

    fn prim(&self, r: Reg) -> Option<Prim> {
        match self.get(r)? {
            Ty::P(p) => Some(p),
            Ty::Str => None,
        }
    }

    /// Gives `r` its type; every assignment of a register must agree.
    fn set(&mut self, r: Reg, t: Ty) -> Option<()> {
        let slot = self.ty.get_mut(r as usize)?;
        match slot {
            None => *slot = Some(t),
            Some(old) if *old == t => {}
            Some(_) => return None,
        }
        Some(())
    }

    fn temp(&mut self, t: Ty) -> Reg {
        self.ty.push(Some(t));
        (self.ty.len() - 1) as Reg
    }

    fn pc(&self) -> u32 {
        self.code.len() as u32
    }

    fn patch(&mut self, at: usize) {
        let to = self.pc();
        match &mut self.code[at] {
            I::Jmp(t) | I::JmpF(_, t) => *t = to,
            _ => unreachable!("a jump"),
        }
    }

    /// `r` as a `t`: itself, or widened (an `int`-sized value already is
    /// an `int`).
    fn as_type(&mut self, r: Reg, t: Prim) -> Option<Reg> {
        let p = self.prim(r)?;
        if p == t || (t == Prim::Int && matches!(p, Prim::Byte | Prim::Short | Prim::Char)) {
            return Some(r);
        }
        if p == Prim::Bool || t == Prim::Bool || !crate::def::widens(p, t) {
            return None;
        }
        let d = self.temp(Ty::P(t));
        self.code.push(I::Conv(d, r, p, t));
        Some(d)
    }

    fn mov(&mut self, d: Reg, s: Reg) -> Option<()> {
        let t = self.get(s)?;
        self.set(d, t)?;
        self.code.push(match t {
            Ty::Str => I::MovRef(d, s),
            Ty::P(_) => I::Mov(d, s),
        });
        Some(())
    }

    fn block(&mut self, ops: &[Op]) -> Option<()> {
        for op in ops {
            if !self.dead.contains(&(op as *const Op)) {
                self.op(op)?;
            }
        }
        Some(())
    }

    fn op(&mut self, op: &Op) -> Option<()> {
        match op {
            Op::Const(d, Value::Str(s)) => {
                self.set(*d, Ty::Str)?;
                self.code.push(I::KStr(*d, s.clone()));
            }
            Op::Const(d, v) => {
                let p = prim_of(v)?;
                self.set(*d, Ty::P(p))?;
                self.code.push(I::K(*d, bits_of(v)?));
            }
            Op::Move(d, s) => self.mov(*d, *s)?,
            Op::Native {
                dst,
                args,
                tag: Tag::Internal(n),
                ..
            } if &**n == "get_score" && args.is_empty() => {
                self.set(*dst, Ty::P(Prim::Double))?;
                self.code.push(I::Score(*dst));
            }
            Op::Api {
                dst,
                args,
                tag: Tag::Static { class, name },
                ..
            } if matches!(&**class, "java.lang.Math" | "java.lang.StrictMath") => match args[..] {
                [a] if &**name == "round" => {
                    let a = self.as_type(a, Prim::Double)?;
                    self.set(*dst, Ty::P(Prim::Long))?;
                    self.code.push(I::Round(*dst, a));
                }
                [a] => {
                    let f = math::unary(name)?;
                    let a = self.as_type(a, Prim::Double)?;
                    self.set(*dst, Ty::P(Prim::Double))?;
                    self.code.push(I::Math1(*dst, a, f));
                }
                [a, b] => {
                    let f = math::binary(name)?;
                    let a = self.as_type(a, Prim::Double)?;
                    let b = self.as_type(b, Prim::Double)?;
                    self.set(*dst, Ty::P(Prim::Double))?;
                    self.code.push(I::Math2(*dst, a, b, f));
                }
                _ => return None,
            },
            Op::Doc {
                dst,
                field,
                access,
                index,
            } => {
                let kind = self.fields.get(*field as usize)?.1?;
                let (d, f) = (*dst, *field);
                match access {
                    DocAccess::Values => return None,
                    DocAccess::Count => {
                        self.set(d, Ty::P(Prim::Int))?;
                        self.code.push(I::DocCount(d, f));
                    }
                    DocAccess::Flip => {
                        let i = self.as_type(*index, Prim::Int)?;
                        self.set(d, Ty::P(Prim::Int))?;
                        self.code.push(I::DocFlip(d, f, i));
                    }
                    DocAccess::Value | DocAccess::Get => {
                        let i = match access {
                            DocAccess::Get => self.as_type(*index, Prim::Int)?,
                            _ => FIRST,
                        };
                        let (t, ins) = match kind {
                            DocKind::Longs => (Ty::P(Prim::Long), I::DocLong(d, f, i)),
                            DocKind::Doubles => (Ty::P(Prim::Double), I::DocDouble(d, f, i)),
                            DocKind::Booleans => (Ty::P(Prim::Bool), I::DocBool(d, f, i)),
                            DocKind::Strings => (Ty::Str, I::DocStr(d, f, i)),
                        };
                        self.set(d, t)?;
                        self.code.push(ins);
                    }
                }
            }
            Op::Arith { dst, op, ty, a, b } => self.arith(*dst, *op, *ty, *a, *b)?,
            Op::Cmp {
                dst,
                op,
                ty,
                def,
                a,
                b,
            } => self.cmp(*dst, *op, *ty, *def, *a, *b)?,
            Op::Unary {
                dst,
                op,
                ty: Some(t),
                src,
                ret: None,
                ..
            } => {
                let (d, s) = (*dst, *src);
                match op {
                    UnOp::Not => {
                        if self.get(s)? != Ty::P(Prim::Bool) {
                            return None;
                        }
                        self.set(d, Ty::P(Prim::Bool))?;
                        self.code.push(I::Not(d, s));
                    }
                    _ => {
                        let t = promoted(*t);
                        if t == Prim::Bool {
                            return None;
                        }
                        let s = self.as_type(s, t)?;
                        self.set(d, Ty::P(t))?;
                        self.code.push(match op {
                            UnOp::Neg => I::Neg(t, d, s),
                            UnOp::Plus => I::Mov(d, s),
                            _ if matches!(t, Prim::Int | Prim::Long) => I::BwNot(t, d, s),
                            _ => return None,
                        });
                    }
                }
            }
            Op::Cast { dst, src, cast } => self.cast(*dst, *src, cast)?,
            Op::If {
                cond,
                then,
                otherwise,
            } => {
                if self.get(*cond)? != Ty::P(Prim::Bool) {
                    return None;
                }
                let jf = self.code.len();
                self.code.push(I::JmpF(*cond, 0));
                self.block(then)?;
                let j = self.code.len();
                self.code.push(I::Jmp(0));
                self.patch(jf);
                self.block(otherwise)?;
                self.patch(j);
            }
            Op::Loop { head, body, after } => {
                let top = self.pc();
                self.loops.push(Loop {
                    breaks: Vec::new(),
                    continues: Vec::new(),
                });
                // `continue` ends the section it is in, as the VM runs them.
                for section in [head, body, after] {
                    self.block(section)?;
                    let cs = std::mem::take(&mut self.loops.last_mut()?.continues);
                    for c in cs {
                        self.patch(c);
                    }
                }
                self.code.push(I::Jmp(top));
                let l = self.loops.pop()?;
                for b in l.breaks {
                    self.patch(b);
                }
            }
            Op::BreakUnless(r) => {
                if self.get(*r)? != Ty::P(Prim::Bool) {
                    return None;
                }
                let at = self.code.len();
                self.loops.last_mut()?.breaks.push(at);
                self.code.push(I::JmpF(*r, 0));
            }
            Op::Break => {
                let at = self.code.len();
                self.loops.last_mut()?.breaks.push(at);
                self.code.push(I::Jmp(0));
            }
            Op::Continue => {
                let at = self.code.len();
                self.loops.last_mut()?.continues.push(at);
                self.code.push(I::Jmp(0));
            }
            Op::Count => self.code.push(I::Count),
            Op::At(o) => self.code.push(I::At(*o)),
            Op::Return(Some(r)) => {
                let t = self.get(*r)?;
                self.code.push(I::Ret(*r, t));
            }
            Op::Return(None) => self.code.push(I::RetNull),
            _ => return None,
        }
        Some(())
    }

    fn arith(&mut self, d: Reg, op: BinOp, ty: Prim, a: Reg, b: Reg) -> Option<()> {
        use BinOp::*;
        let t = promoted(ty);
        let ins = match t {
            Prim::Bool => {
                if !matches!(op, And | Xor | Or)
                    || self.get(a)? != Ty::P(Prim::Bool)
                    || self.get(b)? != Ty::P(Prim::Bool)
                {
                    return None;
                }
                I::Bool(op, d, a, b)
            }
            Prim::Float | Prim::Double if !matches!(op, Mul | Div | Rem | Add | Sub) => {
                return None
            }
            _ if matches!(op, Find | Match) => return None,
            _ => {
                let (a, b) = (self.as_type(a, t)?, self.as_type(b, t)?);
                match (t, op) {
                    (Prim::Double, Add) => I::AddD(d, a, b),
                    (Prim::Double, Sub) => I::SubD(d, a, b),
                    (Prim::Double, Mul) => I::MulD(d, a, b),
                    (Prim::Double, Div) => I::DivD(d, a, b),
                    (Prim::Double, _) => I::Double(op, d, a, b),
                    (Prim::Float, _) => I::Float(op, d, a, b),
                    (Prim::Long, _) => I::Long(op, d, a, b),
                    _ => I::Int(op, d, a, b),
                }
            }
        };
        self.set(d, Ty::P(t))?;
        self.code.push(ins);
        Some(())
    }

    fn cmp(
        &mut self,
        d: Reg,
        op: CmpOp,
        ty: Option<Prim>,
        def: bool,
        a: Reg,
        b: Reg,
    ) -> Option<()> {
        if matches!(op, CmpOp::EqR | CmpOp::NeR) {
            return None;
        }
        let eq = matches!(op, CmpOp::Eq | CmpOp::Ne);
        let p = match ty {
            Some(p) => p,
            None => match (self.get(a)?, self.get(b)?) {
                (Ty::Str, Ty::Str) if eq => {
                    self.set(d, Ty::P(Prim::Bool))?;
                    self.code.push(I::CmpStr(op, d, a, b));
                    return Some(());
                }
                (Ty::P(Prim::Bool), Ty::P(Prim::Bool)) if def && eq => Prim::Bool,
                (Ty::P(x), Ty::P(y)) if def && x != Prim::Bool && y != Prim::Bool => promote(x, y),
                _ => return None,
            },
        };
        let ins = if p == Prim::Bool {
            if !eq || self.get(a)? != Ty::P(p) || self.get(b)? != Ty::P(p) {
                return None;
            }
            I::CmpBool(op, d, a, b)
        } else {
            let (a, b) = (self.as_type(a, p)?, self.as_type(b, p)?);
            match p {
                Prim::Double => I::CmpDouble(op, d, a, b),
                Prim::Float => I::CmpFloat(op, d, a, b),
                Prim::Long => I::CmpLong(op, d, a, b),
                _ => I::CmpInt(op, d, a, b),
            }
        };
        self.set(d, Ty::P(Prim::Bool))?;
        self.code.push(ins);
        Some(())
    }

    fn cast(&mut self, d: Reg, s: Reg, cast: &CastOp) -> Option<()> {
        let from = self.get(s)?;
        match (cast, from) {
            (CastOp::Prim { to, .. }, Ty::P(p)) => self.convert(d, s, p, *to),
            (
                CastOp::DefTo {
                    to,
                    explicit,
                    boxed,
                },
                Ty::P(p),
            ) => {
                if p == *to {
                    self.mov(d, s)
                } else if !*boxed
                    && p != Prim::Bool
                    && *to != Prim::Bool
                    && (*explicit || crate::def::widens(p, *to))
                {
                    self.convert(d, s, p, *to)
                } else {
                    None
                }
            }
            (CastOp::Check(c), t) => {
                let passes = matches!(
                    &**c,
                    "org.opensearch.painless.lookup.def" | "java.lang.Object"
                ) || match t {
                    Ty::Str => matches!(
                        &**c,
                        "java.lang.String" | "java.lang.CharSequence" | "java.lang.Comparable"
                    ),
                    Ty::P(p) => {
                        &**c == box_class(p) || (&**c == "java.lang.Number" && p != Prim::Bool)
                    }
                };
                if passes {
                    self.mov(d, s)
                } else {
                    None
                }
            }
            (CastOp::DefToString { .. }, Ty::Str) => self.mov(d, s),
            _ => None,
        }
    }

    fn convert(&mut self, d: Reg, s: Reg, from: Prim, to: Prim) -> Option<()> {
        if from == to {
            return self.mov(d, s);
        }
        if from == Prim::Bool || to == Prim::Bool {
            return None;
        }
        self.set(d, Ty::P(to))?;
        self.code.push(I::Conv(d, s, from, to));
        Some(())
    }
}

/// The ops whose results nothing reads and that cannot throw.
fn dead_ops(body: &[Op], regs: usize) -> HashSet<*const Op> {
    let mut reads = vec![0u32; regs];
    let mut pure: Vec<(*const Op, Reg, Option<Reg>)> = Vec::new();
    walk(body, &mut |op| {
        for r in op_reads(op) {
            if let Some(n) = reads.get_mut(r as usize) {
                *n += 1;
            }
        }
        let def = match op {
            Op::Const(d, _) => Some((*d, None)),
            Op::Move(d, s) => Some((*d, Some(*s))),
            Op::Native {
                dst,
                args,
                tag: Tag::Internal(_),
                ..
            } if args.is_empty() => Some((*dst, None)),
            Op::Doc {
                dst,
                access: DocAccess::Values | DocAccess::Count,
                ..
            } => Some((*dst, None)),
            Op::Cast {
                dst,
                src,
                cast: CastOp::Check(c),
            } if matches!(
                &**c,
                "org.opensearch.painless.lookup.def" | "java.lang.Object"
            ) =>
            {
                Some((*dst, Some(*src)))
            }
            _ => None,
        };
        if let Some((d, s)) = def {
            pure.push((op as *const Op, d, s));
        }
    });
    let mut dead = HashSet::new();
    loop {
        let mut changed = false;
        for &(op, d, s) in &pure {
            if reads.get(d as usize) == Some(&0) && dead.insert(op) {
                if let Some(n) = s.and_then(|s| reads.get_mut(s as usize)) {
                    *n -= 1;
                }
                changed = true;
            }
        }
        if !changed {
            return dead;
        }
    }
}

/// Calls `f` on every op, nested ones included.
fn walk(ops: &[Op], f: &mut dyn FnMut(&Op)) {
    for op in ops {
        f(op);
        match op {
            Op::If {
                then, otherwise, ..
            }
            | Op::IfNull {
                then, otherwise, ..
            } => {
                walk(then, f);
                walk(otherwise, f);
            }
            Op::Loop { head, body, after } => {
                walk(head, f);
                walk(body, f);
                walk(after, f);
            }
            Op::Try { body, catches } => {
                walk(body, f);
                for c in catches.iter() {
                    walk(&c.body, f);
                }
            }
            _ => {}
        }
    }
}

/// The registers `op` itself reads (not its nested blocks).
fn op_reads(op: &Op) -> Vec<Reg> {
    match op {
        Op::Const(..)
        | Op::Loop { .. }
        | Op::Try { .. }
        | Op::Break
        | Op::Continue
        | Op::Count
        | Op::At(_)
        | Op::Return(None) => Vec::new(),
        Op::Move(_, s) => vec![*s],
        Op::Api { args, .. }
        | Op::Native { args, .. }
        | Op::Local { args, .. }
        | Op::Concat { args, .. } => args.to_vec(),
        Op::Doc { access, index, .. } => match access {
            DocAccess::Get | DocAccess::Flip => vec![*index],
            _ => Vec::new(),
        },
        Op::Arith { a, b, .. } | Op::DefArith { a, b, .. } | Op::Cmp { a, b, .. } => vec![*a, *b],
        Op::Unary { src, .. } | Op::Cast { src, .. } | Op::InstanceOf { src, .. } => vec![*src],
        Op::If { cond, .. } => vec![*cond],
        Op::IfNull { src, .. } => vec![*src],
        Op::BreakUnless(r) | Op::Throw(r) | Op::Return(Some(r)) => vec![*r],
    }
}

fn promoted(p: Prim) -> Prim {
    match p {
        Prim::Byte | Prim::Short | Prim::Char => Prim::Int,
        p => p,
    }
}

/// `DefMath`'s binary promotion.
fn promote(a: Prim, b: Prim) -> Prim {
    use Prim::*;
    if a == Double || b == Double {
        Double
    } else if a == Float || b == Float {
        Float
    } else if a == Long || b == Long {
        Long
    } else {
        Int
    }
}

fn box_class(p: Prim) -> &'static str {
    match p {
        Prim::Bool => "java.lang.Boolean",
        Prim::Byte => "java.lang.Byte",
        Prim::Short => "java.lang.Short",
        Prim::Char => "java.lang.Character",
        Prim::Int => "java.lang.Integer",
        Prim::Long => "java.lang.Long",
        Prim::Float => "java.lang.Float",
        Prim::Double => "java.lang.Double",
    }
}

fn bits_of(v: &Value) -> Option<u64> {
    Some(match v {
        Value::Bool(b) => u64::from(*b),
        Value::Byte(x) => i32::from(*x) as u32 as u64,
        Value::Short(x) => i32::from(*x) as u32 as u64,
        Value::Char(x) => u64::from(*x),
        Value::Int(x) => *x as u32 as u64,
        Value::Long(x) => *x as u64,
        Value::Float(x) => u64::from(x.to_bits()),
        Value::Double(x) => x.to_bits(),
        _ => return None,
    })
}

fn boxed(b: u64, p: Prim) -> Value {
    match p {
        Prim::Bool => Value::Bool(b != 0),
        Prim::Byte => Value::Byte(b as i32 as i8),
        Prim::Short => Value::Short(b as i32 as i16),
        Prim::Char => Value::Char(b as u16),
        Prim::Int => Value::Int(b as i32),
        Prim::Long => Value::Long(b as i64),
        Prim::Float => Value::Float(f32::from_bits(b as u32)),
        Prim::Double => Value::Double(f64::from_bits(b)),
    }
}

#[inline]
fn f(b: u64) -> f64 {
    f64::from_bits(b)
}

#[inline]
fn ff(b: u64) -> f32 {
    f32::from_bits(b as u32)
}

#[inline]
fn i(b: u64) -> i32 {
    b as i32
}

#[inline]
fn int_bits(x: i32) -> u64 {
    x as u32 as u64
}

/// Java's primitive conversion from `from` to `to`.
#[inline]
fn conv(b: u64, from: Prim, to: Prim) -> u64 {
    use Prim::*;
    enum N {
        I(i32),
        L(i64),
        F(f32),
        D(f64),
    }
    let n = match from {
        Byte | Short | Char | Int => N::I(i(b)),
        Long => N::L(b as i64),
        Float => N::F(ff(b)),
        Double => N::D(f(b)),
        Bool => return b,
    };
    match to {
        Byte | Short | Char | Int => {
            // `l2i` truncates; `f2i` and `d2i` saturate, NaN to 0, as `as` does.
            let x = match n {
                N::I(x) => x,
                N::L(x) => x as i32,
                N::F(x) => x as i32,
                N::D(x) => x as i32,
            };
            int_bits(match to {
                Byte => i32::from(x as i8),
                Short => i32::from(x as i16),
                Char => i32::from(x as u16),
                _ => x,
            })
        }
        Long => {
            (match n {
                N::I(x) => i64::from(x),
                N::L(x) => x,
                N::F(x) => x as i64,
                N::D(x) => x as i64,
            }) as u64
        }
        Float => u64::from(
            match n {
                N::I(x) => x as f32,
                N::L(x) => x as f32,
                N::F(x) => x,
                N::D(x) => x as f32,
            }
            .to_bits(),
        ),
        Double => match n {
            N::I(x) => f64::from(x),
            N::L(x) => x as f64,
            N::F(x) => f64::from(x),
            N::D(x) => x,
        }
        .to_bits(),
        Bool => b,
    }
}

fn div_zero() -> Throw {
    Throw::new("java.lang.ArithmeticException", "/ by zero")
}

#[inline(always)]
fn cmp<T: PartialOrd>(op: CmpOp, x: T, y: T) -> bool {
    match op {
        CmpOp::Eq => x == y,
        CmpOp::Ne => x != y,
        CmpOp::Lt => x < y,
        CmpOp::Lte => x <= y,
        CmpOp::Gt => x > y,
        _ => x >= y,
    }
}

#[inline(always)]
fn int_op(op: BinOp, x: i32, y: i32) -> Result<i32, Throw> {
    Ok(match op {
        BinOp::Mul => x.wrapping_mul(y),
        BinOp::Div if y == 0 => return Err(div_zero()),
        BinOp::Div => x.wrapping_div(y),
        BinOp::Rem if y == 0 => return Err(div_zero()),
        BinOp::Rem => x.wrapping_rem(y),
        BinOp::Add => x.wrapping_add(y),
        BinOp::Sub => x.wrapping_sub(y),
        BinOp::Lsh => x.wrapping_shl(y as u32 & 31),
        BinOp::Rsh => x.wrapping_shr(y as u32 & 31),
        BinOp::Ush => ((x as u32) >> (y as u32 & 31)) as i32,
        BinOp::And => x & y,
        BinOp::Xor => x ^ y,
        BinOp::Or => x | y,
        BinOp::Find | BinOp::Match => unreachable!("lowering refuses regex operators"),
    })
}

#[inline(always)]
fn long_op(op: BinOp, x: i64, y: i64) -> Result<i64, Throw> {
    Ok(match op {
        BinOp::Mul => x.wrapping_mul(y),
        BinOp::Div if y == 0 => return Err(div_zero()),
        BinOp::Div => x.wrapping_div(y),
        BinOp::Rem if y == 0 => return Err(div_zero()),
        BinOp::Rem => x.wrapping_rem(y),
        BinOp::Add => x.wrapping_add(y),
        BinOp::Sub => x.wrapping_sub(y),
        BinOp::Lsh => x.wrapping_shl(y as u32 & 63),
        BinOp::Rsh => x.wrapping_shr(y as u32 & 63),
        BinOp::Ush => ((x as u64) >> (y as u32 & 63)) as i64,
        BinOp::And => x & y,
        BinOp::Xor => x ^ y,
        BinOp::Or => x | y,
        BinOp::Find | BinOp::Match => unreachable!("lowering refuses regex operators"),
    })
}

macro_rules! float_op {
    ($op:expr, $x:expr, $y:expr) => {
        match $op {
            BinOp::Mul => $x * $y,
            BinOp::Div => $x / $y,
            BinOp::Rem => $x % $y,
            BinOp::Add => $x + $y,
            _ => $x - $y,
        }
    };
}

/// Register `x` of the script's register file.
#[inline(always)]
fn reg<T: Clone>(file: &[T], x: Reg) -> T {
    file[x as usize].clone()
}

#[inline(always)]
fn reg_mut<T>(file: &mut [T], x: Reg) -> &mut T {
    &mut file[x as usize]
}

impl Typed {
    /// Registers sized for this script.
    pub fn regs(&self) -> Regs {
        let mut regs = Regs {
            bits: vec![0; self.regs],
            refs: vec![Value::Null; self.regs],
            pc: 0,
        };
        for i in &self.init {
            match i {
                Init::Bits(r, v) => regs.bits[*r as usize] = *v,
                Init::Str(r, v) => regs.refs[*r as usize] = Value::Str(v.clone()),
            }
        }
        regs
    }

    /// The source offset of instruction `pc`.
    fn loc(&self, pc: usize) -> i32 {
        self.locs.get(pc).copied().unwrap_or(-1)
    }

    /// The source offset of the instruction that threw in the last
    /// [`Typed::run`] with `regs`.
    pub fn thrown_at(&self, regs: &Regs) -> i32 {
        self.loc(regs.pc)
    }

    /// Whether [`Typed::run_batch`] runs this script.
    pub fn batchable(&self) -> bool {
        self.batchable
    }

    /// Runs a [batchable](Typed::batchable) script for `out.len()`
    /// documents at once, each instruction over the documents whose path
    /// reaches it: `out[k]` is document `k`'s result. A document that throws
    /// stops alone; the first one (`k`, the exception, its source offset)
    /// is returned, and only the documents before it have results. `Err`
    /// is the runtime's own failure, not a script's.
    pub fn run_batch<E: BatchEnv + ?Sized>(
        &self,
        env: &mut E,
        regs: &mut BatchRegs,
        out: &mut [f64],
    ) -> Result<Option<(usize, Throw, i32)>, Throw> {
        assert!(self.batchable, "a batchable script");
        let n = out.len();
        if regs.width < n || regs.cols.len() != regs.width * self.regs {
            regs.width = n.max(regs.width);
            regs.cols = vec![0; regs.width * self.regs];
            for i in &self.init {
                if let Init::Bits(r, v) = i {
                    let at = *r as usize * regs.width;
                    regs.cols[at..at + regs.width].fill(*v);
                }
            }
        }
        let w = regs.width;
        let cols = &mut regs.cols;
        // The documents on the current path, and those waiting at a jump's
        // target.
        let mut active = vec![true; n];
        let mut waiting: Vec<(usize, Vec<bool>)> = Vec::new();
        // Each document's exception, with where it was thrown: it stops that
        // document alone, as Java's loop would stop at the first.
        let mut failed: Vec<Option<(Throw, usize)>> = vec![None; n];
        let mut tmp = vec![0u64; n];
        // Each instruction computes every lane into `tmp`, then keeps the
        // active lanes: a register another path assigned keeps its value.
        macro_rules! keep {
            ($d:expr, $all:expr) => {{
                let d = *$d as usize * w;
                let dst = &mut cols[d..d + n];
                if $all {
                    dst.copy_from_slice(&tmp);
                } else {
                    for ((o, t), a) in dst.iter_mut().zip(&tmp).zip(&active) {
                        if *a {
                            *o = *t;
                        }
                    }
                }
            }};
        }
        macro_rules! un {
            ($d:expr, $s:expr, $all:expr, |$x:ident| $e:expr) => {{
                let s = *$s as usize * w;
                for (t, &$x) in tmp.iter_mut().zip(&cols[s..s + n]) {
                    *t = $e;
                }
                keep!($d, $all)
            }};
        }
        macro_rules! bin {
            ($d:expr, $a:expr, $b:expr, $all:expr, |$x:ident, $y:ident| $e:expr) => {{
                let (a, b) = (*$a as usize * w, *$b as usize * w);
                for ((t, &$x), &$y) in tmp.iter_mut().zip(&cols[a..a + n]).zip(&cols[b..b + n]) {
                    *t = $e;
                }
                keep!($d, $all)
            }};
        }
        // Lane `k` throws: it leaves the path.
        macro_rules! fail {
            ($k:expr, $t:expr, $pc:expr) => {{
                failed[$k] = Some(($t, $pc));
                active[$k] = false;
            }};
        }
        // A comparison with its operator chosen once for all lanes.
        macro_rules! cmp_lanes {
            ($op:expr, $d:expr, $a:expr, $b:expr, $all:expr, |$x:ident, $y:ident| ($p:expr, $q:expr)) => {
                match $op {
                    CmpOp::Eq => bin!($d, $a, $b, $all, |$x, $y| u64::from($p == $q)),
                    CmpOp::Ne => bin!($d, $a, $b, $all, |$x, $y| u64::from($p != $q)),
                    CmpOp::Lt => bin!($d, $a, $b, $all, |$x, $y| u64::from($p < $q)),
                    CmpOp::Lte => bin!($d, $a, $b, $all, |$x, $y| u64::from($p <= $q)),
                    CmpOp::Gt => bin!($d, $a, $b, $all, |$x, $y| u64::from($p > $q)),
                    _ => bin!($d, $a, $b, $all, |$x, $y| u64::from($p >= $q)),
                }
            };
        }
        let mut count = n;
        for (pc, ins) in self.code.iter().enumerate() {
            if waiting.iter().any(|(at, _)| *at == pc) {
                waiting.retain_mut(|(at, lanes)| {
                    if *at != pc {
                        return true;
                    }
                    for (a, l) in active.iter_mut().zip(lanes.iter()) {
                        *a |= *l;
                    }
                    false
                });
                count = active.iter().filter(|a| **a).count();
            }
            if count == 0 {
                continue;
            }
            let all = count == n;
            match ins {
                I::Mov(d, s) => un!(d, s, all, |x| x),
                // Division throws for an active lane's zero divisor; the
                // others divide by anything.
                I::Int(op @ (BinOp::Div | BinOp::Rem), d, a, b)
                | I::Long(op @ (BinOp::Div | BinOp::Rem), d, a, b) => {
                    let bc = *b as usize * w;
                    let long = matches!(ins, I::Long(..));
                    for k in 0..n {
                        let y = cols[bc + k];
                        if active[k] && (if long { y == 0 } else { i(y) == 0 }) {
                            fail!(k, div_zero(), pc);
                        }
                    }
                    count = active.iter().filter(|a| **a).count();
                    let all = count == n;
                    match (long, op) {
                        (true, BinOp::Div) => bin!(d, a, b, all, |x, y| (x as i64)
                            .wrapping_div(if y == 0 { 1 } else { y as i64 })
                            as u64),
                        (true, _) => bin!(d, a, b, all, |x, y| (x as i64).wrapping_rem(if y == 0 {
                            1
                        } else {
                            y as i64
                        }) as u64),
                        (false, BinOp::Div) => bin!(d, a, b, all, |x, y| int_bits(
                            i(x).wrapping_div(if i(y) == 0 { 1 } else { i(y) })
                        )),
                        (false, _) => bin!(d, a, b, all, |x, y| int_bits(
                            i(x).wrapping_rem(if i(y) == 0 { 1 } else { i(y) })
                        )),
                    }
                }
                I::Int(op, d, a, b) => match op {
                    BinOp::Add => bin!(d, a, b, all, |x, y| int_bits(i(x).wrapping_add(i(y)))),
                    BinOp::Sub => bin!(d, a, b, all, |x, y| int_bits(i(x).wrapping_sub(i(y)))),
                    BinOp::Mul => bin!(d, a, b, all, |x, y| int_bits(i(x).wrapping_mul(i(y)))),
                    _ => bin!(d, a, b, all, |x, y| int_bits(
                        int_op(*op, i(x), i(y)).expect("only division throws")
                    )),
                },
                I::Long(op, d, a, b) => match op {
                    BinOp::Add => bin!(d, a, b, all, |x, y| (x as i64).wrapping_add(y as i64)
                        as u64),
                    BinOp::Sub => bin!(d, a, b, all, |x, y| (x as i64).wrapping_sub(y as i64)
                        as u64),
                    BinOp::Mul => bin!(d, a, b, all, |x, y| (x as i64).wrapping_mul(y as i64)
                        as u64),
                    _ => bin!(d, a, b, all, |x, y| long_op(*op, x as i64, y as i64)
                        .expect("only division throws")
                        as u64),
                },
                I::Float(op, d, a, b) => {
                    bin!(d, a, b, all, |x, y| u64::from(
                        float_op!(op, ff(x), ff(y)).to_bits()
                    ))
                }
                I::Double(op, d, a, b) => {
                    bin!(d, a, b, all, |x, y| float_op!(op, f(x), f(y)).to_bits())
                }
                I::Bool(op, d, a, b) => bin!(d, a, b, all, |x, y| match op {
                    BinOp::And => x & y,
                    BinOp::Xor => x ^ y,
                    _ => x | y,
                }),
                I::AddD(d, a, b) => bin!(d, a, b, all, |x, y| (f(x) + f(y)).to_bits()),
                I::SubD(d, a, b) => bin!(d, a, b, all, |x, y| (f(x) - f(y)).to_bits()),
                I::MulD(d, a, b) => bin!(d, a, b, all, |x, y| (f(x) * f(y)).to_bits()),
                I::DivD(d, a, b) => bin!(d, a, b, all, |x, y| (f(x) / f(y)).to_bits()),
                I::CmpInt(op, d, a, b) => cmp_lanes!(op, d, a, b, all, |x, y| (i(x), i(y))),
                I::CmpLong(op, d, a, b) => {
                    cmp_lanes!(op, d, a, b, all, |x, y| (x as i64, y as i64))
                }
                I::CmpFloat(op, d, a, b) => cmp_lanes!(op, d, a, b, all, |x, y| (ff(x), ff(y))),
                I::CmpDouble(op, d, a, b) => cmp_lanes!(op, d, a, b, all, |x, y| (f(x), f(y))),
                I::CmpBool(op, d, a, b) => cmp_lanes!(op, d, a, b, all, |x, y| (x, y)),
                I::Conv(d, s, from, to) => match (from, to) {
                    (Prim::Long, Prim::Double) => un!(d, s, all, |x| (x as i64 as f64).to_bits()),
                    (Prim::Int, Prim::Double) => un!(d, s, all, |x| f64::from(i(x)).to_bits()),
                    (Prim::Int, Prim::Long) => un!(d, s, all, |x| i64::from(i(x)) as u64),
                    (Prim::Float, Prim::Double) => un!(d, s, all, |x| f64::from(ff(x)).to_bits()),
                    _ => un!(d, s, all, |x| conv(x, *from, *to)),
                },
                I::Neg(t, d, s) => un!(d, s, all, |x| match t {
                    Prim::Long => (x as i64).wrapping_neg() as u64,
                    Prim::Float => u64::from((-ff(x)).to_bits()),
                    Prim::Double => (-f(x)).to_bits(),
                    _ => int_bits(i(x).wrapping_neg()),
                }),
                I::BwNot(t, d, s) => un!(d, s, all, |x| match t {
                    Prim::Long => !x,
                    _ => int_bits(!i(x)),
                }),
                I::Not(d, s) => un!(d, s, all, |x| x ^ 1),
                I::DocLong(d, fld, _) | I::DocDouble(d, fld, _) => {
                    let (counts, firsts) = env.column(*fld as usize)?;
                    for k in 0..n {
                        if active[k] && counts[k] == 0 {
                            fail!(
                                k,
                                Throw::new(
                                    "java.lang.IllegalStateException",
                                    crate::api::doc::NO_VALUE,
                                ),
                                pc
                            );
                        }
                    }
                    tmp.copy_from_slice(&firsts[..n]);
                    count = active.iter().filter(|a| **a).count();
                    let all = count == n;
                    keep!(d, all)
                }
                I::DocCount(d, fld) => {
                    let (counts, _) = env.column(*fld as usize)?;
                    tmp.copy_from_slice(&counts[..n]);
                    keep!(d, all)
                }
                I::Score(d) => {
                    env.scores(&mut tmp)?;
                    keep!(d, all)
                }
                I::Math1(d, s, func) => un!(d, s, all, |x| func(f(x)).to_bits()),
                I::Math2(d, a, b, func) => bin!(d, a, b, all, |x, y| func(f(x), f(y)).to_bits()),
                I::Round(d, s) => un!(d, s, all, |x| math::round(f(x)) as u64),
                I::JmpF(c, t) => {
                    let c = *c as usize * w;
                    let mut lanes = vec![false; n];
                    for k in 0..n {
                        if active[k] && cols[c + k] == 0 {
                            lanes[k] = true;
                            active[k] = false;
                        }
                    }
                    waiting.push((*t as usize, lanes));
                    count = active.iter().filter(|a| **a).count();
                }
                I::Jmp(t) => {
                    waiting.push((*t as usize, std::mem::replace(&mut active, vec![false; n])));
                    count = 0;
                }
                I::Ret(s, _) => {
                    let s = *s as usize * w;
                    for k in 0..n {
                        if active[k] {
                            out[k] = f(cols[s + k]);
                            active[k] = false;
                        }
                    }
                    count = 0;
                }
                I::RetNull => {
                    return Err(crate::compile::missing("a score script returned null"));
                }
                _ => unreachable!("not batchable"),
            }
        }
        Ok(failed
            .into_iter()
            .enumerate()
            .find_map(|(k, f)| f.map(|(t, pc)| (k, t, self.loc(pc)))))
    }

    /// Runs the script once against `env`.
    pub fn run<E: Env + ?Sized>(&self, env: &mut E, regs: &mut Regs) -> Result<Value, Throw> {
        Ok(match self.exec(env, regs)? {
            Out::Bits(b, p) => boxed(b, p),
            Out::Value(v) => v,
        })
    }

    /// [`Typed::run`] for a caller that wants a `double`: the value when the
    /// script returns one, else what it returns.
    #[inline]
    pub fn run_double<E: Env + ?Sized>(
        &self,
        env: &mut E,
        regs: &mut Regs,
    ) -> Result<Result<f64, Value>, Throw> {
        Ok(match self.exec(env, regs)? {
            Out::Bits(b, Prim::Double) => Ok(f(b)),
            Out::Bits(b, p) => Err(boxed(b, p)),
            Out::Value(v) => Err(v),
        })
    }

    fn exec<E: Env + ?Sized>(&self, env: &mut E, regs: &mut Regs) -> Result<Out, Throw> {
        let Regs { bits: r, refs, pc } = regs;
        assert!(
            r.len() >= self.regs && refs.len() >= self.regs,
            "registers of this script"
        );
        let mut counter = self.max_loop_counter;
        let counting = counter > 0;
        let idx = |r: &[u64], x: Reg| if x == FIRST { 0 } else { i(reg(r, x)) };
        *pc = 0;
        loop {
            match &self.code[*pc] {
                I::K(d, v) => *reg_mut(r, *d) = *v,
                I::KStr(d, s) => *reg_mut(refs, *d) = Value::Str(s.clone()),
                I::Mov(d, s) => *reg_mut(r, *d) = reg(r, *s),
                I::MovRef(d, s) => *reg_mut(refs, *d) = reg(refs, *s).clone(),
                I::Int(op, d, a, b) => {
                    *reg_mut(r, *d) = int_bits(int_op(*op, i(reg(r, *a)), i(reg(r, *b)))?)
                }
                I::Long(op, d, a, b) => {
                    *reg_mut(r, *d) = long_op(*op, reg(r, *a) as i64, reg(r, *b) as i64)? as u64
                }
                I::Float(op, d, a, b) => {
                    let (x, y) = (ff(reg(r, *a)), ff(reg(r, *b)));
                    *reg_mut(r, *d) = u64::from(float_op!(op, x, y).to_bits())
                }
                I::Double(op, d, a, b) => {
                    let (x, y) = (f(reg(r, *a)), f(reg(r, *b)));
                    *reg_mut(r, *d) = float_op!(op, x, y).to_bits()
                }
                I::Bool(op, d, a, b) => {
                    let (x, y) = (reg(r, *a), reg(r, *b));
                    *reg_mut(r, *d) = match op {
                        BinOp::And => x & y,
                        BinOp::Xor => x ^ y,
                        _ => x | y,
                    }
                }
                I::AddD(d, a, b) => *reg_mut(r, *d) = (f(reg(r, *a)) + f(reg(r, *b))).to_bits(),
                I::SubD(d, a, b) => *reg_mut(r, *d) = (f(reg(r, *a)) - f(reg(r, *b))).to_bits(),
                I::MulD(d, a, b) => *reg_mut(r, *d) = (f(reg(r, *a)) * f(reg(r, *b))).to_bits(),
                I::DivD(d, a, b) => *reg_mut(r, *d) = (f(reg(r, *a)) / f(reg(r, *b))).to_bits(),
                I::CmpInt(op, d, a, b) => {
                    *reg_mut(r, *d) = u64::from(cmp(*op, i(reg(r, *a)), i(reg(r, *b))))
                }
                I::CmpLong(op, d, a, b) => {
                    *reg_mut(r, *d) = u64::from(cmp(*op, reg(r, *a) as i64, reg(r, *b) as i64))
                }
                I::CmpFloat(op, d, a, b) => {
                    *reg_mut(r, *d) = u64::from(cmp(*op, ff(reg(r, *a)), ff(reg(r, *b))))
                }
                I::CmpDouble(op, d, a, b) => {
                    *reg_mut(r, *d) = u64::from(cmp(*op, f(reg(r, *a)), f(reg(r, *b))))
                }
                I::CmpBool(op, d, a, b) => {
                    *reg_mut(r, *d) = u64::from(cmp(*op, reg(r, *a), reg(r, *b)))
                }
                I::CmpStr(op, d, a, b) => {
                    let eq = reg(refs, *a).java_equals(&reg(refs, *b));
                    *reg_mut(r, *d) = u64::from(eq == (*op == CmpOp::Eq))
                }
                I::Conv(d, s, from, to) => *reg_mut(r, *d) = conv(reg(r, *s), *from, *to),
                I::Neg(t, d, s) => {
                    let x = reg(r, *s);
                    *reg_mut(r, *d) = match t {
                        Prim::Long => (x as i64).wrapping_neg() as u64,
                        Prim::Float => u64::from((-ff(x)).to_bits()),
                        Prim::Double => (-f(x)).to_bits(),
                        _ => int_bits(i(x).wrapping_neg()),
                    }
                }
                I::BwNot(t, d, s) => {
                    let x = reg(r, *s);
                    *reg_mut(r, *d) = match t {
                        Prim::Long => !x,
                        _ => int_bits(!i(x)),
                    }
                }
                I::Not(d, s) => *reg_mut(r, *d) = reg(r, *s) ^ 1,
                I::DocLong(d, fld, x) => {
                    let v = env.doc_long(*fld as usize, idx(r, *x))?;
                    *reg_mut(r, *d) = v as u64
                }
                I::DocDouble(d, fld, x) => {
                    let v = env.doc_double(*fld as usize, idx(r, *x))?;
                    *reg_mut(r, *d) = v.to_bits()
                }
                I::DocBool(d, fld, x) => {
                    let v = env.doc_at(*fld as usize, DocAccess::Get, idx(r, *x))?;
                    *reg_mut(r, *d) = match v {
                        Value::Bool(b) => u64::from(b),
                        v => {
                            return Err(crate::compile::missing(format!(
                                "doc boolean {}",
                                v.class_name()
                            )))
                        }
                    }
                }
                I::DocStr(d, fld, x) => {
                    let v = env.doc_at(*fld as usize, DocAccess::Get, idx(r, *x))?;
                    *reg_mut(refs, *d) = v
                }
                I::DocCount(d, fld) => *reg_mut(r, *d) = int_bits(env.doc_count(*fld as usize)?),
                I::DocFlip(d, fld, x) => {
                    let k = i(reg(r, *x));
                    *reg_mut(r, *d) = int_bits(if k < 0 {
                        k.wrapping_add(env.doc_count(*fld as usize)?)
                    } else {
                        k
                    })
                }
                I::Score(d) => *reg_mut(r, *d) = env.score()?.to_bits(),
                I::Math1(d, s, func) => *reg_mut(r, *d) = func(f(reg(r, *s))).to_bits(),
                I::Math2(d, a, b, func) => {
                    *reg_mut(r, *d) = func(f(reg(r, *a)), f(reg(r, *b))).to_bits()
                }
                I::Round(d, s) => *reg_mut(r, *d) = math::round(f(reg(r, *s))) as u64,
                I::Jmp(t) => {
                    *pc = *t as usize;
                    continue;
                }
                I::JmpF(c, t) => {
                    if reg(r, *c) == 0 {
                        *pc = *t as usize;
                        continue;
                    }
                }
                I::Count => {
                    if counting {
                        counter -= 1;
                        if counter <= 0 {
                            return Err(Throw::new(
                                "org.opensearch.painless.PainlessError",
                                LOOP_LIMIT_MESSAGE,
                            ));
                        }
                    }
                }
                I::Ret(s, Ty::P(p)) => return Ok(Out::Bits(reg(r, *s), *p)),
                I::Ret(s, Ty::Str) => return Ok(Out::Value(reg(refs, *s))),
                I::RetNull => return Ok(Out::Value(Value::Null)),
                I::At(_) => unreachable!("lowering folds marks into the offsets"),
            }
            *pc += 1;
        }
    }
}
