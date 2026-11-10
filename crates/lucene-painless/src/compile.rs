//! IR to register program.
//!
//! The IR is a JVM stack program: a consumer node (`LoadMapShortcut`,
//! `StoreBrace`, `InvokeCall`) pops what the nodes before it pushed. The
//! compiler keeps that stack symbolically, one register per pushed value
//! with its JVM word size (so `Dup`'s word counts map onto values), and
//! emits one op per node in IR order. A variable read is pushed as the
//! variable's own register and copied out only if the variable is written
//! while the read is still on the stack, so `x + (x = 5)` reads the old `x`.

use std::collections::HashMap;
use std::rc::Rc;

use crate::api;
use crate::ir::{self, Constant, Member, Node};
use crate::jmap::{JMap, Order};
use crate::lambda::{Lambda, Target};
use crate::types::{self, prim, unboxed, words, Prim, DEF};
use crate::value::{ListKind, MapKind, Value};
use crate::vm::{BinOp, Block, CastOp, Catch, CmpOp, Func, Op, Program, Reg, Tag, Throw, UnOp, Vm};
use crate::{Error, Result};

/// The class Painless compiles a script into: a reference whose delegate
/// is this class names one of the script's functions.
const SCRIPT_CLASS: &str = "org.opensearch.painless.PainlessScript$Script";

fn unsupported(what: impl Into<String>) -> Error {
    Error::Unsupported(what.into())
}

#[derive(Debug, Clone, Copy)]
struct Entry {
    reg: Reg,
    words: usize,
    /// Set while the entry is an unmaterialised read of this variable.
    var: Option<Reg>,
}

struct Shared {
    funcs: HashMap<String, u32>,
    binding_sites: usize,
}

struct Fx<'s> {
    shared: &'s mut Shared,
    regs: u32,
    /// Each register's static primitive type, where it has one.
    reg_types: Vec<Option<Prim>>,
    scopes: Vec<Vec<(String, Reg)>>,
    stack: Vec<Entry>,
    ops: Vec<Op>,
    /// A located node's mark that follows its child's code (a cast, a
    /// store), until the node emits it.
    mark_after: Option<i32>,
    /// Whether Painless's bytecode has an instruction since the last mark.
    code: bool,
    /// The last of a run of marks with no bytecode between them, in force
    /// once the op they label is out: the JVM gives that op the run's
    /// first (an exact match ends its line table search).
    run_last: Option<i32>,
    /// The loop being compiled's mark, for its counter's check.
    loop_mark: Option<i32>,
    /// Whether the function counts loop passes.
    counting: bool,
}

/// Compiles a decoded script.
///
/// # Errors
/// [`Error::Unsupported`] for a construct the runtime does not implement.
pub fn compile(script: &ir::Script) -> Result<Program> {
    let mut shared = Shared {
        funcs: HashMap::new(),
        binding_sites: 0,
    };
    for (i, f) in script.functions.iter().enumerate() {
        shared
            .funcs
            .insert(format!("{}/{}", f.name, f.param_types.len()), i as u32);
    }
    let entry = script
        .functions
        .iter()
        .position(|f| f.name == "execute")
        .ok_or_else(|| Error::Ir("no execute function".into()))? as u32;
    let mut funcs = Vec::with_capacity(script.functions.len());
    for f in &script.functions {
        let mut fx = Fx {
            shared: &mut shared,
            regs: 0,
            reg_types: Vec::new(),
            scopes: vec![Vec::new()],
            stack: Vec::new(),
            ops: Vec::new(),
            mark_after: None,
            code: true,
            run_last: None,
            loop_mark: None,
            counting: f.max_loop_counter > 0,
        };
        for (name, ty) in f.param_names.iter().zip(&f.param_types) {
            let r = fx.reg();
            fx.typed(r, ty);
            fx.scopes[0].push((name.clone(), r));
        }
        fx.stmt(&f.body)?;
        let mut body = fx.ops;
        place_marks(&mut body, &mut -1);
        funcs.push(Func {
            name: f.name.clone(),
            arity: f.param_types.len(),
            regs: fx.regs as usize,
            reg_types: {
                let mut t = fx.reg_types;
                t.resize(fx.regs as usize, None);
                t
            },
            body: body.into_boxed_slice(),
            max_loop_counter: f.max_loop_counter,
        });
    }
    Ok(Program {
        funcs,
        entry,
        binding_sites: shared.binding_sites,
    })
}

/// Whether Painless writes no instruction for `c`: a widening to `def` or
/// `Object` (ASM's `checkCast` skips `Object`), or no change of type.
fn no_instruction(c: &ir::Cast) -> bool {
    let reference = |t: &str| prim(t).is_none();
    c.unbox_from.is_none()
        && c.unbox_to.is_none()
        && c.box_from.is_none()
        && c.box_to.is_none()
        && (c.from == c.to
            || (reference(&c.from)
                && c.from != crate::types::DEF
                && (c.to == crate::types::DEF || c.to == "java.lang.Object")))
}

/// The `Value` of a constant.
pub fn constant(c: &Constant) -> Result<Value> {
    Ok(match c {
        Constant::Null => Value::Null,
        Constant::Bool(b) => Value::Bool(*b),
        Constant::Byte(v) => Value::Byte(*v),
        Constant::Short(v) => Value::Short(*v),
        Constant::Char(v) => Value::Char(*v),
        Constant::Int(v) => Value::Int(*v),
        Constant::Long(v) => Value::Long(*v),
        Constant::Float(v) => Value::Float(*v),
        Constant::Double(v) => Value::Double(*v),
        Constant::Str(s) => Value::str(s),
        Constant::Pattern(p, flags) => crate::api::regex::compile_pattern(p, *flags)
            .map_err(|e| unsupported(format!("regex: {e}")))?,
        Constant::List(items) => Value::list(
            ListKind::ArrayList,
            items.iter().map(constant).collect::<Result<_>>()?,
        ),
        Constant::Map(entries) => {
            let mut m = JMap::with_capacity(Order::Hash, entries.len());
            for (k, v) in entries {
                m.insert(constant(k)?, constant(v)?);
            }
            Value::map(MapKind::HashMap, m)
        }
    })
}

/// The default value of a declared, uninitialised variable.
fn default_of(ty: &str) -> Value {
    match prim(ty) {
        Some(Prim::Bool) => Value::Bool(false),
        Some(Prim::Byte) => Value::Byte(0),
        Some(Prim::Short) => Value::Short(0),
        Some(Prim::Char) => Value::Char(0),
        Some(Prim::Int) => Value::Int(0),
        Some(Prim::Long) => Value::Long(0),
        Some(Prim::Float) => Value::Float(0.0),
        Some(Prim::Double) => Value::Double(0.0),
        None => Value::Null,
    }
}

/// The `CastOp` a `PainlessCast` performs, or `None` for no conversion.
pub fn cast_op(c: &ir::Cast) -> Option<CastOp> {
    let (from, to) = (c.from.as_str(), c.to.as_str());
    if from == "char" && to == "java.lang.String" {
        return Some(CastOp::CharToString);
    }
    if from == "java.lang.String" && to == "char" {
        return Some(CastOp::StringToChar);
    }
    if let (Some(_), Some(bt)) = (&c.unbox_from, &c.box_to) {
        return Some(CastOp::Prim {
            to: prim(bt)?,
            unbox: true,
        });
    }
    if c.unbox_from.is_some() {
        return Some(CastOp::Prim {
            to: prim(to).or_else(|| unboxed(to))?,
            unbox: true,
        });
    }
    if let Some(ut) = &c.unbox_to {
        return Some(CastOp::Prim {
            to: prim(ut).or_else(|| prim(to))?,
            unbox: true,
        });
    }
    if c.box_from.is_some() {
        return match prim(to).or_else(|| unboxed(to)) {
            Some(p) => Some(CastOp::Prim {
                to: p,
                unbox: false,
            }),
            None => Some(CastOp::Check(Rc::from(to))),
        };
    }
    if let Some(bt) = &c.box_to {
        return Some(CastOp::Prim {
            to: prim(bt)?,
            unbox: false,
        });
    }
    if from == DEF {
        if let Some(p) = prim(to) {
            return Some(CastOp::DefTo {
                to: p,
                explicit: c.explicit,
                boxed: false,
            });
        }
        if let Some(p) = unboxed(to) {
            return Some(CastOp::DefTo {
                to: p,
                explicit: c.explicit,
                boxed: true,
            });
        }
        if to == "java.lang.String" {
            return Some(CastOp::DefToString {
                explicit: c.explicit,
            });
        }
        return check(to);
    }
    if let (Some(_), Some(t)) = (prim(from), prim(to)) {
        return Some(CastOp::Prim {
            to: t,
            unbox: false,
        });
    }
    if prim(from).is_some() {
        // A primitive into `def`/`Object`/`Number`: boxing keeps the value.
        return None;
    }
    check(to)
}

fn check(to: &str) -> Option<CastOp> {
    if to == DEF || to == "java.lang.Object" || prim(to).is_some() {
        None
    } else {
        Some(CastOp::Check(Rc::from(to)))
    }
}

fn array_component(ty: &str) -> String {
    let inner = ty.strip_prefix('[').unwrap_or(ty);
    match inner {
        "Z" => "boolean".into(),
        "B" => "byte".into(),
        "S" => "short".into(),
        "C" => "char".into(),
        "I" => "int".into(),
        "J" => "long".into(),
        "F" => "float".into(),
        "D" => "double".into(),
        s if s.starts_with('[') => s.to_string(),
        s => s.trim_start_matches('L').trim_end_matches(';').to_string(),
    }
}

impl Fx<'_> {
    fn reg(&mut self) -> Reg {
        let r = self.regs;
        self.regs += 1;
        r
    }

    fn emit(&mut self, op: Op) {
        self.ops.push(op);
        if let Some(o) = self.run_last.take() {
            self.ops.push(Op::At(o));
        }
    }

    /// A loop pass's count, at the loop's mark when the function counts.
    fn count(&mut self) {
        if let Some(o) = self.loop_mark.take() {
            if self.counting {
                self.mark(o);
            }
        }
        self.emit(Op::Count);
        self.code = true;
    }

    /// A located node's mark at this point of the code.
    fn mark(&mut self, o: i32) {
        if !self.code && matches!(self.ops.last(), Some(Op::At(_))) {
            self.run_last = Some(o);
        } else {
            self.run_last = None;
            self.ops.push(Op::At(o));
        }
        self.code = false;
    }

    /// Records `r`'s static type `ty`, when a primitive.
    fn typed(&mut self, r: Reg, ty: &str) {
        if let Some(p) = prim(ty) {
            let i = r as usize;
            if self.reg_types.len() <= i {
                self.reg_types.resize(i + 1, None);
            }
            self.reg_types[i] = Some(p);
        }
    }

    fn push(&mut self, ty: &str) -> Reg {
        let r = self.reg();
        self.typed(r, ty);
        self.stack.push(Entry {
            reg: r,
            words: words(ty),
            var: None,
        });
        r
    }

    fn push_reg(&mut self, reg: Reg, ty: &str) {
        self.stack.push(Entry {
            reg,
            words: words(ty),
            var: None,
        });
    }

    fn pop(&mut self) -> Result<Reg> {
        self.stack
            .pop()
            .map(|e| e.reg)
            .ok_or_else(|| Error::Ir("stack underflow".into()))
    }

    fn pop_n(&mut self, n: usize) -> Result<Vec<Reg>> {
        if self.stack.len() < n {
            return Err(Error::Ir("stack underflow".into()));
        }
        let at = self.stack.len() - n;
        Ok(self.stack.drain(at..).map(|e| e.reg).collect())
    }

    fn top(&self) -> Result<Reg> {
        self.stack
            .last()
            .map(|e| e.reg)
            .ok_or_else(|| Error::Ir("stack underflow".into()))
    }

    fn lookup(&self, name: &str) -> Result<Reg> {
        for s in self.scopes.iter().rev() {
            if let Some((_, r)) = s.iter().rev().find(|(n, _)| n == name) {
                return Ok(*r);
            }
        }
        Err(Error::Ir(format!("unknown variable {name}")))
    }

    fn declare(&mut self, name: &str) -> Reg {
        let r = self.reg();
        self.scopes
            .last_mut()
            .expect("scope")
            .push((name.to_string(), r));
        r
    }

    /// Copies out every pending read of `var` before it is written.
    fn before_write(&mut self, var: Reg) {
        for i in 0..self.stack.len() {
            if self.stack[i].var == Some(var) {
                let t = self.reg();
                self.ops.push(Op::Move(t, var));
                self.stack[i].reg = t;
                self.stack[i].var = None;
            }
        }
    }

    /// Copies out every pending variable read, before code that runs
    /// conditionally or repeatedly.
    fn settle(&mut self) {
        for i in 0..self.stack.len() {
            if let Some(v) = self.stack[i].var {
                let t = self.reg();
                self.ops.push(Op::Move(t, v));
                self.stack[i].reg = t;
                self.stack[i].var = None;
            }
        }
    }

    fn write_var(&mut self, var: Reg, src: Reg) {
        self.before_write(var);
        if var != src {
            self.emit(Op::Move(var, src));
        }
    }

    /// Compiles `f` into a block of its own.
    /// The pending mark of a cast or store, once its child's code is out.
    fn emit_mark_after(&mut self) {
        if let Some(o) = self.mark_after.take() {
            self.mark(o);
        }
    }

    fn block(&mut self, f: impl FnOnce(&mut Self) -> Result<()>) -> Result<Block> {
        if let Some(o) = self.run_last.take() {
            self.ops.push(Op::At(o));
        }
        let saved = std::mem::take(&mut self.ops);
        let r = f(self);
        let ops = std::mem::replace(&mut self.ops, saved);
        r?;
        Ok(ops.into_boxed_slice())
    }

    fn scoped(&mut self, f: impl FnOnce(&mut Self) -> Result<()>) -> Result<()> {
        self.scopes.push(Vec::new());
        let r = f(self);
        self.scopes.pop();
        r
    }

    fn opt_stmt(&mut self, n: Option<&Node>) -> Result<()> {
        match n {
            Some(n) => self.stmt(n),
            None => Ok(()),
        }
    }

    /// A statement, or an expression whose value is dropped.
    fn stmt_or_expr(&mut self, n: &Node) -> Result<()> {
        if is_statement(n) {
            self.stmt(n)
        } else {
            let depth = self.stack.len();
            self.expr(n)?;
            self.stack.truncate(depth);
            Ok(())
        }
    }

    fn stmt(&mut self, n: &Node) -> Result<()> {
        self.stmt_code(n)?;
        self.code = true;
        Ok(())
    }

    fn stmt_code(&mut self, n: &Node) -> Result<()> {
        match n {
            Node::At(o, ir::At::Loop, inner) => {
                let outer = self.loop_mark.replace(*o);
                let r = self.stmt(inner);
                self.loop_mark = outer;
                r
            }
            Node::Block(stmts) => self.scoped(|fx| {
                for s in stmts {
                    fx.stmt(s)?;
                }
                Ok(())
            }),
            Node::DeclarationBlock(decls) => {
                for d in decls {
                    self.stmt(d)?;
                }
                Ok(())
            }
            Node::Declaration { name, ty, init } => {
                match init {
                    Some(e) => {
                        self.expr(e)?;
                        let v = self.pop()?;
                        let r = self.declare(name);
                        self.typed(r, ty);
                        self.emit(Op::Move(r, v));
                    }
                    None => {
                        let r = self.declare(name);
                        self.typed(r, ty);
                        self.emit(Op::Const(r, default_of(ty)));
                    }
                }
                Ok(())
            }
            Node::StatementExpression(e) => {
                let depth = self.stack.len();
                self.expr(e)?;
                self.stack.truncate(depth);
                Ok(())
            }
            Node::Return(e) => {
                match e {
                    Some(e) => {
                        self.expr(e)?;
                        let r = self.pop()?;
                        self.emit(Op::Return(Some(r)));
                    }
                    None => self.emit(Op::Return(None)),
                }
                Ok(())
            }
            Node::If { cond, then } => {
                self.expr(cond)?;
                let c = self.pop()?;
                self.settle();
                let then = self.block(|fx| fx.stmt(then))?;
                self.emit(Op::If {
                    cond: c,
                    then,
                    otherwise: Box::new([]),
                });
                Ok(())
            }
            Node::IfElse {
                cond,
                then,
                otherwise,
            } => {
                self.expr(cond)?;
                let c = self.pop()?;
                self.settle();
                let then = self.block(|fx| fx.stmt(then))?;
                let otherwise = self.block(|fx| fx.stmt(otherwise))?;
                self.emit(Op::If {
                    cond: c,
                    then,
                    otherwise,
                });
                Ok(())
            }
            Node::While { cond, body, .. } => {
                self.settle();
                let head = self.block(|fx| fx.cond(cond.as_deref()))?;
                let body = self.block(|fx| {
                    fx.count();
                    fx.opt_stmt(body.as_deref())
                })?;
                self.emit(Op::Loop {
                    head,
                    body,
                    after: Box::new([]),
                });
                Ok(())
            }
            Node::DoWhile { cond, body, .. } => {
                self.settle();
                let body = self.block(|fx| {
                    fx.count();
                    fx.opt_stmt(body.as_deref())
                })?;
                let after = self.block(|fx| fx.cond(cond.as_deref()))?;
                self.emit(Op::Loop {
                    head: Box::new([]),
                    body,
                    after,
                });
                Ok(())
            }
            Node::For {
                init,
                cond,
                after,
                body,
                ..
            } => self.scoped(|fx| {
                if let Some(i) = init {
                    fx.stmt_or_expr(i)?;
                }
                fx.settle();
                let head = fx.block(|fx| fx.cond(cond.as_deref()))?;
                let body = fx.block(|fx| {
                    fx.count();
                    fx.opt_stmt(body.as_deref())
                })?;
                let after = fx.block(|fx| match after {
                    Some(a) => fx.stmt_or_expr(a),
                    None => Ok(()),
                })?;
                fx.emit(Op::Loop { head, body, after });
                Ok(())
            }),
            Node::ForEachArray {
                var,
                cast,
                iterable,
                body,
                ..
            } => self.scoped(|fx| {
                fx.expr(iterable)?;
                let arr = fx.pop()?;
                fx.settle();
                let idx = fx.reg();
                fx.emit(Op::Const(idx, Value::Int(0)));
                let len = fx.reg();
                fx.emit(Op::Api {
                    dst: len,
                    f: api::array::length,
                    args: Box::new([arr]),
                    tag: Tag::None,
                });
                let v = fx.declare(var);
                let cast = cast.as_ref().and_then(cast_op);
                let head = fx.block(|fx| {
                    let c = fx.reg();
                    fx.emit(Op::Cmp {
                        dst: c,
                        op: CmpOp::Lt,
                        ty: Some(Prim::Int),
                        def: false,
                        a: idx,
                        b: len,
                    });
                    fx.emit(Op::BreakUnless(c));
                    Ok(())
                })?;
                let one = fx.reg();
                fx.emit(Op::Const(one, Value::Int(1)));
                let body = fx.block(|fx| {
                    fx.count();
                    fx.emit(Op::Api {
                        dst: v,
                        f: api::array::load,
                        args: Box::new([arr, idx]),
                        tag: Tag::None,
                    });
                    if let Some(c) = cast {
                        fx.emit(Op::Cast {
                            dst: v,
                            src: v,
                            cast: c,
                        });
                    }
                    fx.emit(Op::Arith {
                        dst: idx,
                        op: BinOp::Add,
                        ty: Prim::Int,
                        a: idx,
                        b: one,
                    });
                    fx.opt_stmt(body.as_deref())
                })?;
                fx.emit(Op::Loop {
                    head,
                    body,
                    after: Box::new([]),
                });
                Ok(())
            }),
            Node::ForEachIterable {
                var,
                cast,
                iterable,
                body,
                ..
            } => self.scoped(|fx| {
                fx.expr(iterable)?;
                let src = fx.pop()?;
                fx.settle();
                let it = fx.reg();
                fx.emit(Op::Api {
                    dst: it,
                    f: api::iter::iterator,
                    args: Box::new([src]),
                    tag: Tag::None,
                });
                let v = fx.declare(var);
                let cast = cast.as_ref().and_then(cast_op);
                let head = fx.block(|fx| {
                    let c = fx.reg();
                    fx.emit(Op::Api {
                        dst: c,
                        f: api::iter::has_next,
                        args: Box::new([it]),
                        tag: Tag::None,
                    });
                    fx.emit(Op::BreakUnless(c));
                    Ok(())
                })?;
                let body = fx.block(|fx| {
                    fx.count();
                    fx.emit(Op::Api {
                        dst: v,
                        f: api::iter::next,
                        args: Box::new([it]),
                        tag: Tag::None,
                    });
                    if let Some(c) = cast {
                        fx.emit(Op::Cast {
                            dst: v,
                            src: v,
                            cast: c,
                        });
                    }
                    fx.opt_stmt(body.as_deref())
                })?;
                fx.emit(Op::Loop {
                    head,
                    body,
                    after: Box::new([]),
                });
                Ok(())
            }),
            Node::Break => {
                self.emit(Op::Break);
                Ok(())
            }
            Node::Continue => {
                self.emit(Op::Continue);
                Ok(())
            }
            Node::Throw(e) => {
                self.expr(e)?;
                let r = self.pop()?;
                self.emit(Op::Throw(r));
                Ok(())
            }
            Node::Try { body, catches } => {
                self.settle();
                let body = self.block(|fx| fx.stmt(body))?;
                let mut cs = Vec::with_capacity(catches.len());
                for c in catches {
                    let Node::Catch {
                        exception,
                        symbol,
                        body,
                    } = c
                    else {
                        return Err(Error::Ir("try without catch node".into()));
                    };
                    let mut slot = 0;
                    let block = self.block(|fx| {
                        fx.scoped(|fx| {
                            slot = fx.declare(symbol);
                            fx.opt_stmt(body.as_deref())
                        })
                    })?;
                    cs.push(Catch {
                        class: Rc::from(exception.as_str()),
                        slot,
                        body: block,
                    });
                }
                self.emit(Op::Try {
                    body,
                    catches: cs.into_boxed_slice(),
                });
                Ok(())
            }
            other => self.stmt_or_expr_fallback(other),
        }
    }

    fn stmt_or_expr_fallback(&mut self, n: &Node) -> Result<()> {
        if is_statement(n) {
            return Err(Error::Ir(format!("unexpected statement {n:?}")));
        }
        let depth = self.stack.len();
        self.expr(n)?;
        self.stack.truncate(depth);
        Ok(())
    }

    /// A loop condition: break when it is false.
    fn cond(&mut self, cond: Option<&Node>) -> Result<()> {
        if let Some(c) = cond {
            self.expr(c)?;
            let r = self.pop()?;
            self.emit(Op::BreakUnless(r));
        }
        Ok(())
    }

    /// Compiles a branch that leaves one value, moved into `dst`.
    fn branch_value(&mut self, n: &Node, dst: Reg) -> Result<Block> {
        self.block(|fx| {
            fx.expr(n)?;
            let v = fx.pop()?;
            fx.emit(Op::Move(dst, v));
            Ok(())
        })
    }

    fn api(&mut self, ty: &str, f: crate::vm::ApiFn, args: Vec<Reg>) {
        self.api_tagged(ty, f, args, Tag::None);
    }

    fn api_tagged(&mut self, ty: &str, f: crate::vm::ApiFn, args: Vec<Reg>, tag: Tag) {
        let dst = if ty == "void" {
            self.reg()
        } else {
            self.push(ty)
        };
        self.emit(Op::Api {
            dst,
            f,
            args: args.into_boxed_slice(),
            tag,
        });
    }

    fn native(&mut self, ty: &str, f: crate::vm::Native, args: Vec<Reg>) {
        self.native_tagged(ty, f, args, Tag::None);
    }

    fn native_tagged(&mut self, ty: &str, f: crate::vm::Native, args: Vec<Reg>, tag: Tag) {
        let dst = if ty == "void" {
            self.reg()
        } else {
            self.push(ty)
        };
        self.emit(Op::Native {
            dst,
            f,
            args: args.into_boxed_slice(),
            tag,
        });
    }

    fn exprs(&mut self, ns: &[Node]) -> Result<Vec<Reg>> {
        for n in ns {
            self.expr(n)?;
        }
        self.pop_n(ns.len())
    }

    fn captures(&mut self, names: &[String]) -> Result<Vec<Reg>> {
        names.iter().map(|n| self.lookup(n)).collect()
    }

    fn expr(&mut self, n: &Node) -> Result<()> {
        if matches!(n, Node::NewObject { .. }) {
            // `new` and `dup` come before the arguments.
            self.code = true;
        }
        self.expr_code(n)?;
        let silent = match n {
            Node::At(..) | Node::BinaryImpl { .. } => true,
            Node::Cast { cast, .. } => cast.as_ref().is_none_or(no_instruction),
            _ => false,
        };
        if !silent {
            self.code = true;
        }
        Ok(())
    }

    fn expr_code(&mut self, n: &Node) -> Result<()> {
        match n {
            Node::At(offset, ir::At::Before, node) => {
                self.mark(*offset);
                self.expr(node)?;
            }
            Node::At(offset, ir::At::AfterChild, node) => {
                let outer = self.mark_after.replace(*offset);
                let r = self.expr(node);
                self.mark_after = outer;
                r?;
            }
            Node::BinaryImpl { left, right, .. } => {
                self.expr(left)?;
                self.expr(right)?;
            }
            Node::Constant { ty, value } => {
                let v = constant(value)?;
                let r = self.push(ty);
                self.emit(Op::Const(r, v));
            }
            Node::Null { ty } => {
                let r = self.push(ty);
                self.emit(Op::Const(r, Value::Null));
            }
            Node::LoadVariable { ty, name } => {
                let v = self.lookup(name)?;
                self.stack.push(Entry {
                    reg: v,
                    words: words(ty),
                    var: Some(v),
                });
            }
            Node::StoreVariable { name, child, .. } => {
                self.expr(child)?;
                let v = self.pop()?;
                let var = self.lookup(name)?;
                self.write_var(var, v);
            }
            Node::BinaryMath {
                ty,
                op,
                binary_ty,
                flags,
                left,
                right,
                ..
            } => {
                self.expr(left)?;
                self.expr(right)?;
                let b = self.pop()?;
                let a = self.pop()?;
                let op = BinOp::parse(op).ok_or_else(|| Error::Ir(format!("binary op {op}")))?;
                if matches!(op, BinOp::Find | BinOp::Match) {
                    let f = if op == BinOp::Find {
                        api::regex::find_op
                    } else {
                        api::regex::match_op
                    };
                    self.api(ty, f, vec![a, b]);
                    return Ok(());
                }
                let dst = self.push(ty);
                if binary_ty == DEF {
                    let ret = (ty != DEF).then(|| Rc::from(ty.as_str()));
                    self.emit(Op::DefArith {
                        dst,
                        op,
                        a,
                        b,
                        flags: *flags,
                        ret,
                    });
                } else {
                    let p = prim(binary_ty)
                        .ok_or_else(|| unsupported(format!("binary math on {binary_ty}")))?;
                    self.emit(Op::Arith {
                        dst,
                        op,
                        ty: p,
                        a,
                        b,
                    });
                }
            }
            Node::Boolean {
                ty,
                op,
                left,
                right,
            } => {
                self.expr(left)?;
                let a = self.pop()?;
                self.settle();
                let dst = self.reg();
                let (then, otherwise) = match op.as_str() {
                    "AND" => (
                        self.branch_value(right, dst)?,
                        Box::from([Op::Const(dst, Value::Bool(false))]),
                    ),
                    "OR" => (
                        Box::from([Op::Const(dst, Value::Bool(true))]) as Block,
                        self.branch_value(right, dst)?,
                    ),
                    o => return Err(Error::Ir(format!("boolean op {o}"))),
                };
                self.emit(Op::If {
                    cond: a,
                    then,
                    otherwise,
                });
                self.push_reg(dst, ty);
            }
            Node::Comparison {
                ty,
                op,
                cmp_ty,
                left,
                right,
            } => {
                self.expr(left)?;
                self.expr(right)?;
                let b = self.pop()?;
                let a = self.pop()?;
                let op = CmpOp::parse(op).ok_or_else(|| Error::Ir(format!("comparison {op}")))?;
                let dst = self.push(ty);
                self.emit(Op::Cmp {
                    dst,
                    op,
                    ty: prim(cmp_ty),
                    def: cmp_ty == DEF,
                    a,
                    b,
                });
            }
            Node::Conditional {
                ty,
                cond,
                left,
                right,
            } => {
                self.expr(cond)?;
                let c = self.pop()?;
                self.settle();
                let dst = self.reg();
                let then = self.branch_value(left, dst)?;
                let otherwise = self.branch_value(right, dst)?;
                self.emit(Op::If {
                    cond: c,
                    then,
                    otherwise,
                });
                self.push_reg(dst, ty);
            }
            Node::Elvis { ty, left, right } => {
                self.expr(left)?;
                let a = self.pop()?;
                self.settle();
                let dst = self.reg();
                let then = self.branch_value(right, dst)?;
                self.emit(Op::IfNull {
                    src: a,
                    then,
                    otherwise: Box::new([Op::Move(dst, a)]),
                });
                self.push_reg(dst, ty);
            }
            Node::Cast { ty, cast, child } => {
                self.expr(child)?;
                self.emit_mark_after();
                match cast.as_ref().and_then(cast_op) {
                    Some(c) => {
                        let src = self.pop()?;
                        let dst = self.push(ty);
                        self.emit(Op::Cast { dst, src, cast: c });
                    }
                    None => {
                        let e = self
                            .stack
                            .last_mut()
                            .ok_or_else(|| Error::Ir("stack underflow".into()))?;
                        e.words = words(ty);
                    }
                }
            }
            Node::UnaryMath {
                ty,
                op,
                unary_ty,
                explicit,
                child,
                ..
            } => {
                self.expr(child)?;
                let src = self.pop()?;
                let uop = match op.as_str() {
                    "SUB" => UnOp::Neg,
                    "ADD" => UnOp::Plus,
                    "BWNOT" => UnOp::BwNot,
                    "NOT" => UnOp::Not,
                    o => return Err(Error::Ir(format!("unary op {o}"))),
                };
                let uty = unary_ty.as_deref().unwrap_or(DEF);
                let p = if uop == UnOp::Not {
                    Some(Prim::Bool)
                } else {
                    prim(uty)
                };
                let ret = (p.is_none() && ty != DEF).then(|| Rc::from(ty.as_str()));
                let dst = self.push(ty);
                self.emit(Op::Unary {
                    dst,
                    op: uop,
                    ty: p,
                    src,
                    ret,
                    explicit: *explicit,
                });
            }
            Node::Instanceof {
                ty,
                instance_ty,
                child,
            } => {
                self.expr(child)?;
                let src = self.pop()?;
                let class = prim(instance_ty).map_or(instance_ty.as_str(), |p| p.boxed());
                let dst = self.push(ty);
                self.emit(Op::InstanceOf {
                    dst,
                    src,
                    class: Rc::from(class),
                });
            }
            Node::Dup {
                size, depth, child, ..
            } => {
                self.expr(child)?;
                self.dup(*size as usize, *depth as usize)?;
            }
            Node::FlipArrayIndex { child, .. } => self.flip(child, api::array::flip_index)?,
            Node::FlipCollectionIndex { child, .. } => self.flip(child, api::list::flip_index)?,
            Node::FlipDefIndex { child, .. } => {
                self.expr(child)?;
                let idx = self.pop()?;
                let recv = self.top()?;
                self.api_tagged(
                    "int",
                    api::dynamic::flip_index,
                    vec![recv, idx],
                    Tag::FlipIndex,
                );
            }
            Node::InvokeCall {
                ty,
                method,
                boxed,
                args,
            } => {
                let m = method
                    .as_ref()
                    .ok_or_else(|| Error::Ir("call without method".into()))?;
                if boxed.as_deref().is_some_and(|b| prim(b).is_some()) {
                    if let Some(e) = self.stack.last_mut() {
                        e.words = 1;
                    }
                }
                let mut a = self.exprs(args)?;
                if m.is_static && !api::is_augmentation(&m.declaring) {
                    let f = api::static_method(&m.target, &m.name, args.len())
                        .or_else(|| api::static_method(&m.declaring, &m.name, args.len()))
                        .ok_or_else(|| {
                            unsupported(format!(
                                "static method {}.{}/{}",
                                m.target,
                                m.name,
                                args.len()
                            ))
                        })?;
                    let tag = Tag::Static {
                        class: Rc::from(m.target.as_str()),
                        name: Rc::from(m.name.as_str()),
                    };
                    self.api_tagged(ty, f, a, tag);
                } else {
                    let recv = self.pop()?;
                    a.insert(0, recv);
                    let f = api::method_call(m)?;
                    self.native_tagged(ty, f, a, Tag::Call(Rc::from(m.name.as_str())));
                }
            }
            Node::InvokeCallDef { ty, name, args } => {
                let mut a = self.exprs(args)?;
                let recv = self.pop()?;
                a.insert(0, recv);
                let name: Rc<str> = Rc::from(name.as_str());
                let tag = Tag::Call(name.clone());
                self.native_tagged(
                    ty,
                    Rc::new(move |vm: &mut Vm, args: &[Value]| {
                        api::dynamic::call(vm, &args[0], &name, &args[1..])
                    }),
                    a,
                    tag,
                );
            }
            Node::InvokeCallMember { ty, member, args } => self.member(ty, member, args)?,
            Node::ListInit { ty, args, .. } => {
                let a = self.exprs(args)?;
                self.api(ty, api::list::new_array_list_of, a);
            }
            Node::MapInit {
                ty, keys, values, ..
            } => {
                for (k, v) in keys.iter().zip(values) {
                    self.expr(k)?;
                    self.expr(v)?;
                }
                let a = self.pop_n(keys.len() * 2)?;
                self.api(ty, api::map::new_hash_map_of, a);
            }
            Node::NewArray {
                ty,
                initialize,
                args,
            } => {
                let a = self.exprs(args)?;
                let comp: Rc<str> = Rc::from(array_component(ty).as_str());
                let init = *initialize;
                self.native(
                    ty,
                    Rc::new(move |_vm: &mut Vm, args: &[Value]| {
                        if init {
                            api::array::from_items(&comp, args)
                        } else {
                            api::array::new_dims(&comp, args)
                        }
                    }),
                    a,
                );
            }
            Node::NewObject {
                ty,
                ctor,
                read,
                args,
            } => {
                let a = self.exprs(args)?;
                let class = ctor.as_ref().map_or(ty.as_str(), |c| c.class.as_str());
                let dst = if *read { self.push(ty) } else { self.reg() };
                if types::exception_parent(class).is_some() || class == "java.lang.Exception" {
                    let class: Rc<str> = Rc::from(class);
                    let f: crate::vm::Native = Rc::new(move |_vm: &mut Vm, args: &[Value]| {
                        api::objects::exception_of(&class, args)
                    });
                    self.emit(Op::Native {
                        dst,
                        f,
                        args: a.into_boxed_slice(),
                        tag: Tag::None,
                    });
                } else {
                    let f = api::constructor(class, args.len()).ok_or_else(|| {
                        unsupported(format!("constructor {class}/{}", args.len()))
                    })?;
                    self.emit(Op::Api {
                        dst,
                        f,
                        args: a.into_boxed_slice(),
                        tag: Tag::None,
                    });
                }
            }
            Node::LoadBrace { ty } => {
                let a = self.pop_n(2)?;
                self.api(ty, api::array::load, a);
            }
            Node::LoadBraceDef { ty, .. } => {
                let a = self.pop_n(2)?;
                self.api_tagged(ty, api::dynamic::brace_load, a, Tag::BraceLoad);
            }
            Node::LoadDotArrayLength { ty } => {
                let a = self.pop_n(1)?;
                self.api(ty, api::array::length, a);
            }
            Node::LoadDotDef { ty, name } => {
                let a = self.pop_n(1)?;
                let name: Rc<str> = Rc::from(name.as_str());
                let tag = Tag::LoadDot(name.clone());
                self.native_tagged(
                    ty,
                    Rc::new(move |vm: &mut Vm, args: &[Value]| {
                        api::dynamic::load_dot(vm, &args[0], &name)
                    }),
                    a,
                    tag,
                );
            }
            Node::LoadDot { ty, field } => {
                if field.is_static {
                    let v = api::static_field(&field.class, &field.name).ok_or_else(|| {
                        unsupported(format!("static field {}.{}", field.class, field.name))
                    })?;
                    let r = self.push(ty);
                    self.emit(Op::Const(r, v));
                } else {
                    let a = self.pop_n(1)?;
                    let name: Rc<str> = Rc::from(field.name.as_str());
                    self.native(
                        ty,
                        Rc::new(move |vm: &mut Vm, args: &[Value]| {
                            api::dynamic::load_field(vm, &args[0], &name)
                        }),
                        a,
                    );
                }
            }
            Node::LoadDotShortcut { ty, getter }
            | Node::LoadListShortcut { ty, getter }
            | Node::LoadMapShortcut { ty, getter } => {
                let g = getter
                    .as_ref()
                    .ok_or_else(|| Error::Ir("shortcut without getter".into()))?;
                let n = if matches!(n, Node::LoadDotShortcut { .. }) {
                    1
                } else {
                    2
                };
                let a = self.pop_n(n)?;
                let f = api::method_call(g)?;
                self.native_tagged(ty, f, a, Tag::Call(Rc::from(g.name.as_str())));
            }
            Node::StoreBrace { child, .. } => {
                self.expr(child)?;
                self.emit_mark_after();
                let a = self.pop_n(3)?;
                self.api("void", api::array::store, a);
            }
            Node::StoreBraceDef { child, .. } => {
                self.expr(child)?;
                self.emit_mark_after();
                let a = self.pop_n(3)?;
                self.api("void", api::dynamic::brace_store, a);
            }
            Node::StoreDotDef { name, child, .. } => {
                self.expr(child)?;
                self.emit_mark_after();
                let a = self.pop_n(2)?;
                let name: Rc<str> = Rc::from(name.as_str());
                self.native(
                    "void",
                    Rc::new(move |vm: &mut Vm, args: &[Value]| {
                        api::dynamic::store_dot(vm, &args[0], &name, &args[1])
                    }),
                    a,
                );
            }
            Node::StoreDot { field, child, .. } => {
                if field.is_static {
                    return Err(unsupported(format!(
                        "static field store {}.{}",
                        field.class, field.name
                    )));
                }
                self.expr(child)?;
                self.emit_mark_after();
                let a = self.pop_n(2)?;
                let name: Rc<str> = Rc::from(field.name.as_str());
                self.native(
                    "void",
                    Rc::new(move |vm: &mut Vm, args: &[Value]| {
                        api::dynamic::store_dot(vm, &args[0], &name, &args[1])
                    }),
                    a,
                );
            }
            Node::StoreDotShortcut { setter, child, .. }
            | Node::StoreListShortcut { setter, child, .. }
            | Node::StoreMapShortcut { setter, child, .. } => {
                let s = setter
                    .as_ref()
                    .ok_or_else(|| Error::Ir("shortcut without setter".into()))?;
                let n = if matches!(n, Node::StoreDotShortcut { .. }) {
                    2
                } else {
                    3
                };
                self.expr(child)?;
                self.emit_mark_after();
                let a = self.pop_n(n)?;
                let f = api::method_call(s)?;
                self.native("void", f, a);
            }
            Node::StringConcat { ty, args } => {
                let a = self.exprs(args)?;
                let dst = self.push(ty);
                self.emit(Op::Concat {
                    dst,
                    args: a.into_boxed_slice(),
                });
            }
            Node::NullSafeSub { ty, child } => {
                self.settle();
                let e = *self
                    .stack
                    .last()
                    .ok_or_else(|| Error::Ir("stack underflow".into()))?;
                self.stack.pop();
                let dst = self.reg();
                let otherwise = self.block(|fx| {
                    fx.stack.push(e);
                    fx.expr(child)?;
                    let v = fx.pop()?;
                    fx.emit(Op::Move(dst, v));
                    Ok(())
                })?;
                self.emit(Op::IfNull {
                    src: e.reg,
                    then: Box::new([Op::Const(dst, Value::Null)]),
                    otherwise,
                });
                self.push_reg(dst, ty);
            }
            Node::Static { .. } => {}
            Node::LoadFieldMember { name, .. } | Node::StoreFieldMember { name, .. } => {
                return Err(unsupported(format!("script field {name}")));
            }
            Node::TypedInterfaceReference {
                ty,
                reference,
                captures,
            } => {
                let caps = self.captures(captures)?;
                let target = self.reference_target(reference)?;
                let iface: Rc<str> = Rc::from(ty.as_str());
                let ret: Rc<str> = Rc::from(reference.interface_ret.as_str());
                self.native(
                    ty,
                    Rc::new(move |_vm: &mut Vm, args: &[Value]| {
                        Ok(Value::object(Lambda {
                            target: target.clone(),
                            captures: args.to_vec(),
                            iface: Some(iface.clone()),
                            ret: Some(ret.clone()),
                        }))
                    }),
                    caps,
                );
            }
            Node::DefInterfaceReference {
                ty,
                encoding,
                captures,
            } => {
                let caps = self.captures(captures)?;
                let target = self.def_reference_target(encoding)?;
                self.native(
                    ty,
                    Rc::new(move |_vm: &mut Vm, args: &[Value]| {
                        Ok(Value::object(Lambda {
                            target: target.clone(),
                            captures: args.to_vec(),
                            iface: None,
                            ret: None,
                        }))
                    }),
                    caps,
                );
            }
            Node::TypedCaptureReference {
                ty,
                method,
                captures,
            } => {
                let caps = self.captures(captures)?;
                let target = Target::Virtual(Rc::from(method.as_str()));
                let iface: Rc<str> = Rc::from(ty.as_str());
                self.native(
                    ty,
                    Rc::new(move |_vm: &mut Vm, args: &[Value]| {
                        Ok(Value::object(Lambda {
                            target: target.clone(),
                            captures: args.to_vec(),
                            iface: Some(iface.clone()),
                            ret: None,
                        }))
                    }),
                    caps,
                );
            }
            other => return Err(Error::Ir(format!("unexpected expression {other:?}"))),
        }
        Ok(())
    }

    fn local_by_name(&self, name: &str) -> Option<u32> {
        self.shared
            .funcs
            .iter()
            .find(|(k, _)| k.rsplit_once('/').map(|(n, _)| n) == Some(name))
            .map(|(_, &f)| f)
    }

    fn reference_target(&self, r: &ir::Reference) -> Result<Target> {
        if r.delegate_class == SCRIPT_CLASS {
            if let Some(f) = self.local_by_name(&r.delegate_method) {
                return Ok(Target::Local(f));
            }
        }
        Ok(match r.invoke_type {
            6 => Target::Static(
                Rc::from(r.delegate_class.as_str()),
                Rc::from(r.delegate_method.as_str()),
            ),
            8 => {
                if r.delegate_class.starts_with('[') {
                    Target::NewArray(Rc::from(array_component(&r.delegate_class).as_str()))
                } else {
                    Target::Ctor(Rc::from(r.delegate_class.as_str()))
                }
            }
            _ => Target::Virtual(Rc::from(r.delegate_method.as_str())),
        })
    }

    /// Decodes a `def` reference encoding: `S<type>.<call>,<captures>`
    /// (typed) or `D<type>.<call>,<captures>` (on a `def` capture).
    fn def_reference_target(&self, enc: &str) -> Result<Target> {
        let bad = || Error::Ir(format!("reference encoding {enc}"));
        let kind = enc.chars().next().ok_or_else(bad)?;
        let dot = enc.rfind('.').ok_or_else(bad)?;
        let comma = enc.find(',').ok_or_else(bad)?;
        let ty = &enc[1..dot];
        let call = &enc[dot + 1..comma];
        let captures: usize = enc[comma + 1..].parse().map_err(|_| bad())?;
        if kind == 'D' {
            return Ok(Target::Virtual(Rc::from(call)));
        }
        if ty == "this" {
            let f = self
                .local_by_name(call)
                .ok_or_else(|| Error::Ir(format!("unknown function {call}")))?;
            return Ok(Target::Local(f));
        }
        let _ = captures;
        Ok(Target::ByName(
            Rc::from(api::resolve_class(ty).as_str()),
            Rc::from(call),
        ))
    }

    fn member(&mut self, ty: &str, member: &Member, args: &[Node]) -> Result<()> {
        match member {
            Member::Local {
                name,
                internal: true,
                ..
            } => {
                let a = self.exprs(args)?;
                let name: Rc<str> = Rc::from(name.as_str());
                let tag = Tag::Internal(name.clone());
                self.native_tagged(
                    ty,
                    Rc::new(move |vm: &mut Vm, args: &[Value]| vm.env.internal(&name, args)),
                    a,
                    tag,
                );
            }
            Member::Local { name, params, .. } => {
                let a = self.exprs(args)?;
                let func = *self
                    .shared
                    .funcs
                    .get(&format!("{name}/{}", params.len()))
                    .ok_or_else(|| Error::Ir(format!("unknown function {name}")))?;
                let dst = if ty == "void" {
                    self.reg()
                } else {
                    self.push(ty)
                };
                self.emit(Op::Local {
                    dst,
                    func,
                    args: a.into_boxed_slice(),
                });
            }
            Member::Imported(m) => {
                let a = self.exprs(args)?;
                let f = api::static_method(&m.target, &m.name, args.len())
                    .or_else(|| api::static_method(&m.declaring, &m.name, args.len()))
                    .ok_or_else(|| {
                        unsupported(format!(
                            "imported method {}.{}/{}",
                            m.target,
                            m.name,
                            args.len()
                        ))
                    })?;
                self.api(ty, f, a);
            }
            Member::ClassBinding {
                class,
                method,
                ctor_arity,
                offset,
                ..
            } => {
                let a = self.exprs(args)?;
                let ctor_args = (*ctor_arity - *offset).max(0) as usize;
                let binding = api::class_binding(class, method)
                    .ok_or_else(|| unsupported(format!("class binding {class}.{method}")))?;
                let site = self.shared.binding_sites;
                self.shared.binding_sites += 1;
                self.native(
                    ty,
                    Rc::new(move |vm: &mut Vm, args: &[Value]| {
                        let inst = match &vm.bindings[site] {
                            Some(v) => v.clone(),
                            None => {
                                let v = (binding.construct)(vm, &args[..ctor_args])?;
                                vm.bindings[site] = Some(v.clone());
                                v
                            }
                        };
                        (binding.call)(vm, &inst, &args[ctor_args..])
                    }),
                    a,
                );
            }
            Member::InstanceBinding { class, method, .. } => {
                return Err(unsupported(format!("instance binding {class}.{method}")));
            }
        }
        Ok(())
    }

    fn flip(&mut self, child: &Node, f: crate::vm::ApiFn) -> Result<()> {
        self.expr(child)?;
        let idx = self.pop()?;
        let recv = self.top()?;
        self.api("int", f, vec![recv, idx]);
        Ok(())
    }

    /// JVM `dup`/`dup_x1`/`dup_x2`/`dup2*`: copies the top `size` words
    /// below the `depth` words under them.
    fn dup(&mut self, size: usize, depth: usize) -> Result<()> {
        if size == 0 {
            return Ok(());
        }
        let take = |stack: &[Entry], from: usize, w: usize| -> Result<usize> {
            let mut got = 0;
            let mut i = from;
            while got < w {
                if i == 0 {
                    return Err(Error::Ir("dup underflow".into()));
                }
                i -= 1;
                got += stack[i].words;
            }
            if got != w {
                return Err(Error::Ir("dup splits a value".into()));
            }
            Ok(i)
        };
        let len = self.stack.len();
        let top_at = take(&self.stack, len, size)?;
        let below_at = if depth == 0 {
            top_at
        } else {
            take(&self.stack, top_at, depth)?
        };
        let top: Vec<Entry> = self.stack[top_at..].to_vec();
        let at = below_at;
        for (k, e) in top.into_iter().enumerate() {
            self.stack.insert(at + k, e);
        }
        Ok(())
    }
}

/// Makes each op's mark the one the class's line table gives it: the last
/// mark before it in the code's order, whichever path reached it. So each
/// nested block, the code after a compound op, and the caller after a call
/// of one of the script's functions restate the mark in force there.
fn place_marks(ops: &mut Vec<Op>, cur: &mut i32) {
    let mut out = Vec::with_capacity(ops.len() + 1);
    // Whether the last op out is a mark this pass added, which labels no
    // code of its own and so gives way to a mark the script has there.
    let mut added = false;
    if !matches!(ops.first(), Some(Op::At(_))) {
        out.push(Op::At(*cur));
        added = true;
    }
    for mut op in std::mem::take(ops) {
        if let Op::At(o) = op {
            match out.last_mut() {
                Some(Op::At(prev)) if added => *prev = o,
                _ => out.push(op),
            }
            *cur = o;
            added = false;
            continue;
        }
        let mut restate = false;
        match &mut op {
            Op::If {
                then, otherwise, ..
            }
            | Op::IfNull {
                then, otherwise, ..
            } => {
                place_block(then, cur);
                place_block(otherwise, cur);
                restate = true;
            }
            Op::Loop { head, body, after } => {
                place_block(head, cur);
                place_block(body, cur);
                place_block(after, cur);
                restate = true;
            }
            Op::Try { body, catches } => {
                place_block(body, cur);
                for c in catches.iter_mut() {
                    place_block(&mut c.body, cur);
                }
                restate = true;
            }
            Op::Local { .. } => restate = true,
            _ => {}
        }
        out.push(op);
        added = restate;
        if restate {
            out.push(Op::At(*cur));
        }
    }
    *ops = out;
}

fn place_block(b: &mut Block, cur: &mut i32) {
    let mut v = std::mem::take(b).into_vec();
    place_marks(&mut v, cur);
    *b = v.into_boxed_slice();
}

fn is_statement(n: &Node) -> bool {
    if let Node::At(_, _, inner) = n {
        return is_statement(inner);
    }
    matches!(
        n,
        Node::Block(_)
            | Node::DeclarationBlock(_)
            | Node::Declaration { .. }
            | Node::StatementExpression(_)
            | Node::Return(_)
            | Node::If { .. }
            | Node::IfElse { .. }
            | Node::While { .. }
            | Node::DoWhile { .. }
            | Node::For { .. }
            | Node::ForEachArray { .. }
            | Node::ForEachIterable { .. }
            | Node::Break
            | Node::Continue
            | Node::Throw(_)
            | Node::Try { .. }
            | Node::Catch { .. }
    )
}

/// Runtime error for a reached construct the runtime lacks.
pub fn missing(what: impl Into<String>) -> Throw {
    Throw::new(api::UNSUPPORTED, what)
}
