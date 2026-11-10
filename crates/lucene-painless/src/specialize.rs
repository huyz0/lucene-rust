//! Specialises a compiled program to one query: what `params` holds and how
//! each field the script names reads are known before the first document,
//! so `params.w` becomes a constant, `doc['f'].value` a direct doc-values
//! read, and `def` arithmetic on values of known classes typed arithmetic.
//!
//! The rewrites keep Painless's semantics: a read of `params` is folded only
//! when the script never changes or hands `params` on; a `def` operation is
//! typed only when both operands' runtime classes are known, which decides
//! `DefMath`'s promotion exactly as the dynamic path would.

use std::collections::HashMap;

use crate::api::doc::DocKind;
use crate::types::{prim_of, Prim};
use crate::value::Value;
use crate::vm::{CastOp, DocAccess, Func, Op, Program, Reg, Tag};

/// What the query knows before it runs.
pub struct Facts<'a> {
    /// `params`, as every document's execution sees it.
    pub params: Option<&'a Value>,
    /// The fields `doc` serves, in [`crate::vm::Env::doc_at`]'s numbering,
    /// with how each reads (`None`: not through a direct read).
    pub fields: &'a [(&'a str, Option<DocKind>)],
}

/// What a register is known to hold.
#[derive(Debug, Clone)]
enum Fact {
    /// This value, for every execution.
    Const(Value),
    /// `params`.
    Params,
    /// `doc`.
    Doc,
    /// `doc['f']`: field `f`'s values view.
    Field(u32),
    /// A value of this primitive class (a `def` holding one).
    Class(Prim),
    /// A string.
    Str,
}

/// Rewrites every function of `prog` for `facts`.
pub fn specialize(prog: &mut Program, facts: &Facts<'_>) {
    for f in &mut prog.funcs {
        specialize_func(f, facts);
    }
}

fn specialize_func(f: &mut Func, facts: &Facts<'_>) {
    let mut defs = vec![0u32; f.regs];
    let mut reads: Vec<Vec<Use>> = vec![Vec::new(); f.regs];
    scan(&f.body, &mut defs, &mut reads);
    let mut s = Spec {
        facts,
        defs,
        reads,
        known: HashMap::new(),
        reg_types: &f.reg_types,
        params_ok: false,
        next_reg: f.regs as Reg,
    };
    s.params_ok = s.params_foldable(&f.body);
    let mut body = std::mem::take(&mut f.body);
    s.block(&mut body);
    s.sink_conversions(&mut body);
    // Views no read needs any more: their reads became direct.
    s.drop_unread(&mut body);
    f.body = body;
    f.regs = s.next_reg as usize;
    f.reg_types.resize(f.regs, None);
}

/// How an op reads a register.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Use {
    /// As the receiver of a call by this name (`Tag::Call`) with this many
    /// arguments after it.
    Recv(&'static str, usize),
    /// As the receiver of `.name`.
    /// `"value"` and `"length"` by name, any other name as `""`.
    Dot(&'static str),
    /// As the argument of a call (not its receiver).
    Arg,
    /// As the source of a `def` to primitive cast.
    DefTo,
    /// Any other read.
    Other,
}

fn known_name(n: &str) -> &'static str {
    match n {
        "get" => "get",
        "containsKey" => "containsKey",
        "getOrDefault" => "getOrDefault",
        "size" => "size",
        "isEmpty" => "isEmpty",
        "getLength" => "getLength",
        "getValue" => "getValue",
        _ => "",
    }
}

/// Every register's definitions and reads, in the whole function.
fn scan(ops: &[Op], defs: &mut [u32], reads: &mut [Vec<Use>]) {
    let def = |r: Reg, defs: &mut [u32]| defs[r as usize] += 1;
    let read = |r: Reg, u: Use, reads: &mut [Vec<Use>]| reads[r as usize].push(u);
    for op in ops {
        match op {
            Op::Const(d, _) => def(*d, defs),
            Op::Move(d, s) => {
                def(*d, defs);
                read(*s, Use::Other, reads);
            }
            Op::Api { dst, args, tag, .. } => {
                def(*dst, defs);
                for (i, &a) in args.iter().enumerate() {
                    let u = match (tag, i) {
                        (Tag::BraceLoad, 0) => Use::Recv("get", 1),
                        (Tag::FlipIndex, 0) => Use::Recv("size", 0),
                        _ => Use::Other,
                    };
                    read(a, u, reads);
                }
            }
            Op::Local { dst, args, .. } => {
                def(*dst, defs);
                for &a in args.iter() {
                    read(a, Use::Other, reads);
                }
            }
            Op::Native { dst, args, tag, .. } => {
                def(*dst, defs);
                for (i, &a) in args.iter().enumerate() {
                    let u = match (tag, i) {
                        (Tag::Call(n), 0) => Use::Recv(known_name(n), args.len() - 1),
                        (Tag::LoadDot(n), 0) => Use::Dot(match &**n {
                            "value" => "value",
                            "length" => "length",
                            _ => "",
                        }),
                        (Tag::Call(_), _) => Use::Arg,
                        _ => Use::Other,
                    };
                    read(a, u, reads);
                }
            }
            Op::Doc {
                dst, index, access, ..
            } => {
                def(*dst, defs);
                if matches!(access, DocAccess::Get | DocAccess::Flip) {
                    read(*index, Use::Other, reads);
                }
            }
            Op::Arith { dst, a, b, .. }
            | Op::DefArith { dst, a, b, .. }
            | Op::Cmp { dst, a, b, .. } => {
                def(*dst, defs);
                read(*a, Use::Other, reads);
                read(*b, Use::Other, reads);
            }
            Op::Cast {
                dst,
                src,
                cast: CastOp::DefTo { .. },
            } => {
                def(*dst, defs);
                read(*src, Use::DefTo, reads);
            }
            Op::Unary { dst, src, .. }
            | Op::Cast { dst, src, .. }
            | Op::InstanceOf { dst, src, .. } => {
                def(*dst, defs);
                read(*src, Use::Other, reads);
            }
            Op::Concat { dst, args } => {
                def(*dst, defs);
                for &a in args.iter() {
                    read(a, Use::Other, reads);
                }
            }
            Op::If {
                cond,
                then,
                otherwise,
            } => {
                read(*cond, Use::Other, reads);
                scan(then, defs, reads);
                scan(otherwise, defs, reads);
            }
            Op::IfNull {
                src,
                then,
                otherwise,
            } => {
                read(*src, Use::Other, reads);
                scan(then, defs, reads);
                scan(otherwise, defs, reads);
            }
            Op::Loop { head, body, after } => {
                scan(head, defs, reads);
                scan(body, defs, reads);
                scan(after, defs, reads);
            }
            Op::BreakUnless(r) | Op::Throw(r) => read(*r, Use::Other, reads),
            Op::Return(Some(r)) => read(*r, Use::Other, reads),
            Op::Try { body, catches } => {
                scan(body, defs, reads);
                for c in catches.iter() {
                    def(c.slot, defs);
                    scan(&c.body, defs, reads);
                }
            }
            Op::Return(None) | Op::Break | Op::Continue | Op::Count => {}
        }
    }
}

struct Spec<'a, 'f> {
    facts: &'a Facts<'a>,
    defs: Vec<u32>,
    reads: Vec<Vec<Use>>,
    known: HashMap<Reg, Fact>,
    reg_types: &'f [Option<Prim>],
    params_ok: bool,
    /// The next register a rewrite may add.
    next_reg: Reg,
}

impl Spec<'_, '_> {
    fn single(&self, r: Reg) -> bool {
        self.defs[r as usize] == 1
    }

    fn fact(&self, r: Reg) -> Option<&Fact> {
        self.known.get(&r)
    }

    /// What class `r` holds: its fact, else its static type.
    fn class(&self, r: Reg) -> Option<Prim> {
        match self.known.get(&r) {
            Some(Fact::Class(p)) => Some(*p),
            Some(Fact::Const(v)) => prim_of(v),
            _ => self.reg_types.get(r as usize).copied().flatten(),
        }
    }

    fn learn(&mut self, r: Reg, f: Fact) {
        if self.single(r) {
            self.known.insert(r, f);
        }
    }

    /// Whether every register `getParams` reaches (it and the variables it
    /// is moved to) is only read by a non-mutating lookup.
    fn params_foldable(&self, body: &[Op]) -> bool {
        if self.facts.params.is_none() {
            return false;
        }
        let mut regs = Vec::new();
        collect_params(body, &mut regs);
        let mut i = 0;
        while i < regs.len() {
            let r: Reg = regs[i];
            if !self.single(r) {
                return false;
            }
            for u in &self.reads[r as usize] {
                match u {
                    Use::Recv("get" | "containsKey" | "getOrDefault", _) | Use::Dot(_) => {}
                    // A move to a variable: that variable must obey too.
                    Use::Other => match moved_to(body, r) {
                        Some(d) if !regs.contains(&d) => regs.push(d),
                        Some(_) => {}
                        None => return false,
                    },
                    _ => return false,
                }
            }
            i += 1;
        }
        true
    }

    fn block(&mut self, ops: &mut crate::vm::Block) {
        let mut out = Vec::with_capacity(ops.len());
        for mut op in std::mem::take(ops).into_vec() {
            let then = self.op(&mut op);
            out.push(op);
            out.extend(then);
        }
        *ops = out.into_boxed_slice();
    }

    /// Rewrites `op`; returns an op to run right after it.
    fn op(&mut self, op: &mut Op) -> Option<Op> {
        match op {
            Op::Const(d, v) => {
                let f = match v {
                    Value::Str(_) => Fact::Const(v.clone()),
                    v if prim_of(v).is_some() => Fact::Const(v.clone()),
                    _ => return None,
                };
                let d = *d;
                self.learn(d, f);
            }
            Op::Move(d, s) => {
                if let Some(f) = self.fact(*s).cloned() {
                    let d = *d;
                    self.learn(d, f);
                }
            }
            Op::Native { dst, args, tag, .. } => {
                let dst = *dst;
                let new = match tag {
                    Tag::Internal(n) if &**n == "getParams" && args.is_empty() => {
                        if self.params_ok {
                            let v = self.facts.params.cloned().unwrap_or_default();
                            self.learn(dst, Fact::Params);
                            Some(Op::Const(dst, v))
                        } else {
                            None
                        }
                    }
                    Tag::Internal(n) if &**n == "getDoc" && args.is_empty() => {
                        self.learn(dst, Fact::Doc);
                        None
                    }
                    Tag::Call(n) => self.call(dst, n, args),
                    Tag::LoadDot(n) => self.load_dot(dst, n, args[0]),
                    _ => None,
                };
                if let Some(new) = new {
                    *op = new;
                }
            }
            Op::Arith { dst, ty, .. } => {
                let (dst, ty) = (*dst, *ty);
                self.learn(dst, Fact::Class(promoted(ty)));
            }
            Op::DefArith {
                dst,
                op: bop,
                a,
                b,
                flags,
                ret,
            } => {
                let (dst, a, b) = (*dst, *a, *b);
                let (Some(ca), Some(cb)) = (self.class(a), self.class(b)) else {
                    return None;
                };
                // Bit 0 (`ALLOWS_NULL`) cannot matter: both operands are numbers.
                if *flags & !0b111 != 0 || !numeric(ca) || !numeric(cb) {
                    return None;
                }
                use crate::vm::BinOp::*;
                if !matches!(bop, Mul | Div | Rem | Add | Sub) {
                    return None;
                }
                let ty = promote(ca, cb);
                let bop = *bop;
                // `DefMath`'s result conversion: an explicit cast's type, a
                // compound assignment's left side, or the call site's
                // type, which the result must widen to.
                let target = if *flags & EXPLICIT != 0 {
                    ret.as_deref()
                        .and_then(|r| crate::types::prim(r).or_else(|| crate::types::unboxed(r)))
                } else if *flags & COMPOUND != 0 {
                    Some(ca)
                } else {
                    match ret.as_deref().map(crate::types::prim) {
                        None | Some(None) => None,
                        Some(Some(p)) if crate::def::widens(ty, p) => Some(p),
                        Some(Some(_)) => return None,
                    }
                };
                match target {
                    // `cast_result` refuses a number as a boolean.
                    Some(Prim::Bool) => return None,
                    Some(p) if p != ty => {
                        let tmp = self.next_reg;
                        self.next_reg += 1;
                        *op = Op::Arith {
                            dst: tmp,
                            op: bop,
                            ty,
                            a,
                            b,
                        };
                        self.learn(dst, Fact::Class(p));
                        return Some(Op::Cast {
                            dst,
                            src: tmp,
                            cast: CastOp::Prim { to: p, unbox: true },
                        });
                    }
                    _ => {}
                }
                *op = Op::Arith {
                    dst,
                    op: bop,
                    ty,
                    a,
                    b,
                };
                self.learn(dst, Fact::Class(ty));
            }
            Op::Cmp {
                dst, ty, def, a, b, ..
            } => {
                let dst = *dst;
                if *def {
                    if let (Some(ca), Some(cb)) = (self.class(*a), self.class(*b)) {
                        if numeric(ca) && numeric(cb) {
                            *def = false;
                            *ty = Some(promote(ca, cb));
                        }
                    }
                }
                self.learn(dst, Fact::Class(Prim::Bool));
            }
            Op::Cast { dst, src, cast } => {
                let (dst, src) = (*dst, *src);
                match cast {
                    CastOp::DefTo {
                        to,
                        explicit,
                        boxed,
                    } => {
                        let to = *to;
                        if let Some(from) = self.class(src) {
                            if from == to {
                                *op = Op::Move(dst, src);
                            } else if numeric(from)
                                && numeric(to)
                                && (*explicit || crate::def::widens(from, to))
                                && !*boxed
                            {
                                *cast = CastOp::Prim { to, unbox: true };
                            }
                        }
                        self.learn(dst, Fact::Class(to));
                    }
                    CastOp::Prim { to, .. } => {
                        let to = *to;
                        self.learn(dst, Fact::Class(to));
                    }
                    // A `checkcast` to `def` or `Object` leaves the value as it is.
                    CastOp::Check(c) if is_any(c) => {
                        if let Some(f) = self.fact(src).cloned() {
                            self.learn(dst, f);
                        } else if let Some(p) = self.class(src) {
                            self.learn(dst, Fact::Class(p));
                        }
                    }
                    _ => {}
                }
            }
            Op::Api { dst, args, tag, .. }
                if matches!(tag, Tag::BraceLoad | Tag::FlipIndex) && args.len() == 2 =>
            {
                let dst = *dst;
                let new = match (tag, self.fact(args[0]).cloned()) {
                    (Tag::BraceLoad, Some(Fact::Field(i)))
                        if self.class(args[1]) == Some(Prim::Int) =>
                    {
                        self.learn_kind(dst, i);
                        Some(Op::Doc {
                            dst,
                            field: i,
                            access: DocAccess::Get,
                            index: args[1],
                        })
                    }
                    (Tag::BraceLoad, Some(Fact::Params)) => self.param(args[1]).map(|v| {
                        self.learn_value(dst, &v);
                        Op::Const(dst, v)
                    }),
                    (Tag::FlipIndex, Some(Fact::Field(i)))
                        if self.class(args[1]) == Some(Prim::Int) =>
                    {
                        self.learn(dst, Fact::Class(Prim::Int));
                        Some(Op::Doc {
                            dst,
                            field: i,
                            access: DocAccess::Flip,
                            index: args[1],
                        })
                    }
                    _ => None,
                };
                if let Some(new) = new {
                    *op = new;
                }
            }
            Op::Unary {
                dst, ty: Some(t), ..
            } => {
                let (dst, t) = (*dst, *t);
                self.learn(dst, Fact::Class(promoted(t)));
            }
            Op::If {
                then, otherwise, ..
            }
            | Op::IfNull {
                then, otherwise, ..
            } => {
                self.block(then);
                self.block(otherwise);
            }
            Op::Loop { head, body, after } => {
                self.block(head);
                self.block(body);
                self.block(after);
            }
            Op::Try { body, catches } => {
                self.block(body);
                for c in catches.iter_mut() {
                    self.block(&mut c.body);
                }
            }
            _ => {}
        }
        None
    }

    /// A `def` variable assigned in several places and only read by one
    /// conversion to a primitive (`x = c ? 0 : doc['f'].value; (double) x`):
    /// converts each assigned value instead, when every one's class is
    /// known and converts, so the variable holds one type.
    fn sink_conversions(&mut self, body: &mut [Op]) {
        let mut casts = Vec::new();
        collect_def_to(body, &mut casts);
        for (dst, src, to, explicit) in casts {
            if self.single(src) || self.reads[src as usize] != [Use::DefTo] {
                continue;
            }
            let mut ok = true;
            let mut moves = 0;
            for_defs(body, src, &mut |op| match op {
                Op::Move(_, s) => {
                    moves += 1;
                    ok &= self.class(*s).is_some_and(|from| {
                        numeric(from) && numeric(to) && (explicit || crate::def::widens(from, to))
                    });
                }
                _ => ok = false,
            });
            if !ok || moves != self.defs[src as usize] {
                continue;
            }
            for_defs(body, src, &mut |op| {
                if let Op::Move(d, s) = op {
                    *op = Op::Cast {
                        dst: *d,
                        src: *s,
                        cast: CastOp::Prim { to, unbox: true },
                    };
                }
            });
            replace_def_to(body, dst, src);
        }
    }

    /// `recv.name(args)` with a known receiver.
    fn call(&mut self, dst: Reg, name: &str, args: &[Reg]) -> Option<Op> {
        let recv = args[0];
        match (self.fact(recv).cloned(), name, args.len()) {
            (Some(Fact::Params), "get", 2) => {
                let v = self.param(args[1])?;
                self.learn_value(dst, &v);
                Some(Op::Const(dst, v))
            }
            (Some(Fact::Doc), "get", 2) => {
                let Some(Fact::Const(Value::Str(name))) = self.fact(args[1]) else {
                    return None;
                };
                let i = self
                    .facts
                    .fields
                    .iter()
                    .position(|(f, k)| *f == &**name && k.is_some())?;
                self.learn(dst, Fact::Field(i as u32));
                Some(Op::Doc {
                    dst,
                    field: i as u32,
                    access: DocAccess::Values,
                    index: 0,
                })
            }
            (Some(Fact::Field(i)), "getValue", 1) => Some(self.value(dst, i)),
            (Some(Fact::Field(i)), "size" | "getLength", 1) => {
                self.learn(dst, Fact::Class(Prim::Int));
                Some(Op::Doc {
                    dst,
                    field: i,
                    access: DocAccess::Count,
                    index: 0,
                })
            }
            (Some(Fact::Field(i)), "get", 2) => {
                self.learn_kind(dst, i);
                Some(Op::Doc {
                    dst,
                    field: i,
                    access: DocAccess::Get,
                    index: args[1],
                })
            }
            _ => None,
        }
    }

    /// `recv.name` with a known receiver.
    fn load_dot(&mut self, dst: Reg, name: &str, recv: Reg) -> Option<Op> {
        match (self.fact(recv).cloned(), name) {
            (Some(Fact::Params), _) => {
                let v = self.param_named(name)?;
                self.learn_value(dst, &v);
                Some(Op::Const(dst, v))
            }
            (Some(Fact::Field(i)), "value") => Some(self.value(dst, i)),
            (Some(Fact::Field(i)), "length") => {
                self.learn(dst, Fact::Class(Prim::Int));
                Some(Op::Doc {
                    dst,
                    field: i,
                    access: DocAccess::Count,
                    index: 0,
                })
            }
            _ => None,
        }
    }

    fn value(&mut self, dst: Reg, i: u32) -> Op {
        self.learn_kind(dst, i);
        Op::Doc {
            dst,
            field: i,
            access: DocAccess::Value,
            index: 0,
        }
    }

    fn learn_kind(&mut self, dst: Reg, i: u32) {
        let f = match self.facts.fields[i as usize].1 {
            Some(DocKind::Longs) => Fact::Class(Prim::Long),
            Some(DocKind::Doubles) => Fact::Class(Prim::Double),
            Some(DocKind::Booleans) => Fact::Class(Prim::Bool),
            Some(DocKind::Strings) => Fact::Str,
            None => return,
        };
        self.learn(dst, f);
    }

    fn learn_value(&mut self, dst: Reg, v: &Value) {
        if prim_of(v).is_some() || matches!(v, Value::Str(_)) {
            self.learn(dst, Fact::Const(v.clone()));
        }
    }

    /// `params.get(key)` for a constant key.
    fn param(&self, key: Reg) -> Option<Value> {
        let Some(Fact::Const(k)) = self.fact(key) else {
            return None;
        };
        let Some(Value::Map(m)) = self.facts.params else {
            return None;
        };
        let v = crate::api::map::get_value(&m.borrow(), k);
        Some(v)
    }

    fn param_named(&self, name: &str) -> Option<Value> {
        let Some(Value::Map(m)) = self.facts.params else {
            return None;
        };
        let v = crate::api::map::get_value(&m.borrow(), &Value::str(name));
        Some(v)
    }

    /// Replaces the views of fields whose every read became direct with a
    /// constant, so the document's values are not gathered into one.
    fn drop_unread(&self, ops: &mut [Op]) {
        for op in ops.iter_mut() {
            match op {
                Op::Doc {
                    dst,
                    access: DocAccess::Values,
                    ..
                } if self.reads[*dst as usize]
                    .iter()
                    .all(|u| self.direct(*dst, *u)) =>
                {
                    *op = Op::Const(*dst, Value::Null);
                }
                Op::If {
                    then, otherwise, ..
                }
                | Op::IfNull {
                    then, otherwise, ..
                } => {
                    self.drop_unread(then);
                    self.drop_unread(otherwise);
                }
                Op::Loop { head, body, after } => {
                    self.drop_unread(head);
                    self.drop_unread(body);
                    self.drop_unread(after);
                }
                Op::Try { body, catches } => {
                    self.drop_unread(body);
                    for c in catches.iter_mut() {
                        self.drop_unread(&mut c.body);
                    }
                }
                _ => {}
            }
        }
    }

    /// Whether a read of a field view became a direct read.
    fn direct(&self, _r: Reg, u: Use) -> bool {
        matches!(
            u,
            Use::Recv("getValue" | "size" | "getLength", 0)
                | Use::Recv("get", 1)
                | Use::Dot("value" | "length")
        )
    }
}

/// `DefBootstrap`'s operator flags.
const COMPOUND: i32 = 1 << 1;
const EXPLICIT: i32 = 1 << 2;

/// Whether a reference cast to `class` passes every value unchanged.
fn is_any(class: &str) -> bool {
    matches!(
        class,
        "org.opensearch.painless.lookup.def" | "java.lang.Object"
    )
}

/// Every non-boxing `def`-to-primitive cast: `(dst, src, to, explicit)`.
fn collect_def_to(ops: &[Op], out: &mut Vec<(Reg, Reg, Prim, bool)>) {
    for op in ops {
        match op {
            Op::Cast {
                dst,
                src,
                cast:
                    CastOp::DefTo {
                        to,
                        explicit,
                        boxed: false,
                    },
            } => out.push((*dst, *src, *to, *explicit)),
            _ => for_blocks(op, &mut |b| collect_def_to(b, out)),
        }
    }
}

/// Calls `f` on each op that assigns `r`.
fn for_defs(ops: &mut [Op], r: Reg, f: &mut dyn FnMut(&mut Op)) {
    for op in ops.iter_mut() {
        let assigns = match op {
            Op::Const(d, _) | Op::Move(d, _) => *d == r,
            Op::Api { dst, .. }
            | Op::Native { dst, .. }
            | Op::Doc { dst, .. }
            | Op::Local { dst, .. }
            | Op::Arith { dst, .. }
            | Op::DefArith { dst, .. }
            | Op::Cmp { dst, .. }
            | Op::Unary { dst, .. }
            | Op::Cast { dst, .. }
            | Op::InstanceOf { dst, .. }
            | Op::Concat { dst, .. } => *dst == r,
            Op::Try { catches, .. } => catches.iter().any(|c| c.slot == r),
            _ => false,
        };
        if assigns {
            f(op);
        }
        for_blocks_mut(op, &mut |b| for_defs(b, r, f));
    }
}

/// Turns the cast `dst = (T) src` into a move.
fn replace_def_to(ops: &mut [Op], dst: Reg, src: Reg) {
    for op in ops.iter_mut() {
        match op {
            Op::Cast {
                dst: d,
                src: s,
                cast: CastOp::DefTo { .. },
            } if *d == dst && *s == src => *op = Op::Move(dst, src),
            _ => for_blocks_mut(op, &mut |b| replace_def_to(b, dst, src)),
        }
    }
}

/// Calls `f` on each block nested directly in `op`.
fn for_blocks(op: &Op, f: &mut dyn FnMut(&[Op])) {
    match op {
        Op::If {
            then, otherwise, ..
        }
        | Op::IfNull {
            then, otherwise, ..
        } => {
            f(then);
            f(otherwise);
        }
        Op::Loop { head, body, after } => {
            f(head);
            f(body);
            f(after);
        }
        Op::Try { body, catches } => {
            f(body);
            for c in catches.iter() {
                f(&c.body);
            }
        }
        _ => {}
    }
}

fn for_blocks_mut(op: &mut Op, f: &mut dyn FnMut(&mut [Op])) {
    match op {
        Op::If {
            then, otherwise, ..
        }
        | Op::IfNull {
            then, otherwise, ..
        } => {
            f(then);
            f(otherwise);
        }
        Op::Loop { head, body, after } => {
            f(head);
            f(body);
            f(after);
        }
        Op::Try { body, catches } => {
            f(body);
            for c in catches.iter_mut() {
                f(&mut c.body);
            }
        }
        _ => {}
    }
}

/// The registers `getParams` defines.
fn collect_params(ops: &[Op], out: &mut Vec<Reg>) {
    for op in ops {
        match op {
            Op::Native {
                dst,
                tag: Tag::Internal(n),
                args,
                ..
            } if &**n == "getParams" && args.is_empty() => out.push(*dst),
            Op::If {
                then, otherwise, ..
            }
            | Op::IfNull {
                then, otherwise, ..
            } => {
                collect_params(then, out);
                collect_params(otherwise, out);
            }
            Op::Loop { head, body, after } => {
                collect_params(head, out);
                collect_params(body, out);
                collect_params(after, out);
            }
            Op::Try { body, catches } => {
                collect_params(body, out);
                for c in catches.iter() {
                    collect_params(&c.body, out);
                }
            }
            _ => {}
        }
    }
}

/// The register `src` is moved to, when its one `Other` read is a move.
fn moved_to(ops: &[Op], src: Reg) -> Option<Reg> {
    for op in ops {
        let found = match op {
            Op::Move(d, s) if *s == src => Some(*d),
            Op::If {
                then, otherwise, ..
            }
            | Op::IfNull {
                then, otherwise, ..
            } => moved_to(then, src).or_else(|| moved_to(otherwise, src)),
            Op::Loop { head, body, after } => moved_to(head, src)
                .or_else(|| moved_to(body, src))
                .or_else(|| moved_to(after, src)),
            Op::Try { body, catches } => {
                moved_to(body, src).or_else(|| catches.iter().find_map(|c| moved_to(&c.body, src)))
            }
            _ => None,
        };
        if found.is_some() {
            return found;
        }
    }
    None
}

fn numeric(p: Prim) -> bool {
    !matches!(p, Prim::Bool)
}

/// The class a typed operation of type `ty` leaves.
fn promoted(ty: Prim) -> Prim {
    match ty {
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
