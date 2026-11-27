//! Analysis and lowering: the syntax tree to the IR [`crate::compile`]
//! reads, with every type resolved and every cast explicit, as Painless's
//! own front end leaves it.
//!
//! This covers the core: primitives, `def` and `String`, locals, user
//! functions, operators and control flow. A construct beyond it is refused
//! as [`Refusal::Unsupported`] naming it, never mis-lowered.

mod expr;
mod literal;
#[cfg(test)]
mod tests;
mod ty;

use std::collections::HashMap;

use super::ast;
use crate::ir;
use crate::types::DEF;

/// Why a script did not lower.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    /// Painless would reject the script: a compile error at an offset.
    Compile { offset: u32, message: String },
    /// A construct this front end does not lower yet.
    Unsupported(String),
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Refusal::Compile { offset, message } => write!(f, "{message} at {offset}"),
            Refusal::Unsupported(what) => write!(f, "unsupported: {what}"),
        }
    }
}

pub(super) type Res<T> = std::result::Result<T, Refusal>;

pub(super) fn compile_error<T>(offset: u32, message: impl Into<String>) -> Res<T> {
    Err(Refusal::Compile {
        offset,
        message: message.into(),
    })
}

pub(super) fn unsupported<T>(what: impl Into<String>) -> Res<T> {
    Err(Refusal::Unsupported(what.into()))
}

/// The loop counter limit a function runs under (`max_loop_counter`).
const LOOP_LIMIT: i32 = 1_000_000;

/// A function's signature, as a call resolves it.
#[derive(Clone)]
struct Signature {
    params: Vec<String>,
    ret: String,
}

/// The state while one function lowers.
pub(super) struct Cx<'a> {
    funcs: &'a HashMap<String, Signature>,
    scopes: Vec<HashMap<String, String>>,
    ret: String,
    loops: usize,
}

/// Lowers a parsed script to IR.
pub fn analyze(script: &ast::Script) -> Res<ir::Script> {
    let mut funcs: HashMap<String, Signature> = HashMap::new();
    for f in &script.functions {
        let sig = Signature {
            params: f
                .params
                .iter()
                .map(|(t, _)| ty::resolve(t))
                .collect::<Res<_>>()?,
            ret: ty::resolve(&f.ret)?,
        };
        if funcs.insert(f.name.clone(), sig).is_some() {
            return compile_error(f.pos, format!("function [{}] is already defined", f.name));
        }
    }
    let mut out = Vec::new();
    for f in &script.functions {
        let sig = funcs[&f.name].clone();
        let mut cx = Cx::new(&funcs, sig.ret.clone());
        for ((_, name), ty) in f.params.iter().zip(&sig.params) {
            cx.declare(f.pos, name, ty)?;
        }
        let body = cx.scoped(|cx| cx.body_of(&f.body.stmts, false))?;
        out.push(ir::Function {
            name: f.name.clone(),
            ret: sig.ret,
            param_types: sig.params,
            param_names: f.params.iter().map(|(_, n)| n.clone()).collect(),
            is_static: true,
            synthetic: false,
            max_loop_counter: LOOP_LIMIT,
            body,
        });
    }
    let mut cx = Cx::new(&funcs, DEF.to_string());
    let body = cx.body_of(&script.body, true)?;
    out.insert(
        0,
        ir::Function {
            name: "execute".into(),
            ret: DEF.into(),
            param_types: Vec::new(),
            param_names: Vec::new(),
            is_static: false,
            synthetic: false,
            max_loop_counter: LOOP_LIMIT,
            body,
        },
    );
    Ok(ir::Script {
        functions: out,
        classes: ir::ClassTable::default(),
        sites: Vec::new(),
    })
}

impl<'a> Cx<'a> {
    fn new(funcs: &'a HashMap<String, Signature>, ret: String) -> Self {
        Cx {
            funcs,
            scopes: vec![HashMap::new()],
            ret,
            loops: 0,
        }
    }

    fn declare(&mut self, pos: u32, name: &str, ty: &str) -> Res<()> {
        if self.lookup(name).is_some() {
            return compile_error(pos, format!("variable [{name}] is already defined"));
        }
        self.scopes
            .last_mut()
            .expect("a scope is always open")
            .insert(name.to_string(), ty.to_string());
        Ok(())
    }

    fn lookup(&self, name: &str) -> Option<&String> {
        self.scopes.iter().rev().find_map(|s| s.get(name))
    }

    fn scoped<T>(&mut self, f: impl FnOnce(&mut Self) -> Res<T>) -> Res<T> {
        self.scopes.push(HashMap::new());
        let r = f(self);
        self.scopes.pop();
        r
    }

    /// A function's statements. The last one, when it is an expression
    /// and the function returns a value, is the result; `execute` also
    /// returns `null` when it ends without a `return`.
    fn body_of(&mut self, stmts: &[ast::Stmt], is_execute: bool) -> Res<ir::Node> {
        let mut nodes = Vec::new();
        let valued = self.ret != "void";
        let mut ends_valued = false;
        for (i, s) in stmts.iter().enumerate() {
            match &s.kind {
                ast::StmtKind::Expr(e) if valued && i + 1 == stmts.len() => {
                    let t = self.expr(e)?;
                    if t.ty == "void" && is_execute {
                        nodes.push(ir::Node::StatementExpression(Box::new(t.node)));
                        continue;
                    }
                    let ret = self.ret.clone();
                    let t = self.coerce(e.pos, t, &ret, false)?;
                    nodes.push(ir::Node::Return(Some(Box::new(t.node))));
                    ends_valued = true;
                }
                _ => nodes.push(self.stmt(s)?),
            }
        }
        if !ty::escapes(stmts) && !ends_valued {
            if is_execute {
                nodes.push(ir::Node::Return(Some(Box::new(ir::Node::Null {
                    ty: DEF.into(),
                }))));
            } else if valued {
                return compile_error(0, "not all paths provide a return value");
            } else {
                nodes.push(ir::Node::Return(None));
            }
        }
        Ok(ir::Node::Block(nodes))
    }

    fn block(&mut self, stmts: &[ast::Stmt]) -> Res<ir::Node> {
        self.scoped(|cx| {
            let nodes = stmts.iter().map(|s| cx.stmt(s)).collect::<Res<Vec<_>>>()?;
            Ok(ir::Node::Block(nodes))
        })
    }

    fn cond(&mut self, e: &ast::Expr) -> Res<ir::Node> {
        let t = self.expr(e)?;
        Ok(self.coerce(e.pos, t, "boolean", false)?.node)
    }

    fn body(&mut self, s: &Option<Box<ast::Stmt>>) -> Res<Option<Box<ir::Node>>> {
        let Some(s) = s else { return Ok(None) };
        self.loops += 1;
        let r = self.scoped(|cx| cx.stmt(s));
        self.loops -= 1;
        Ok(Some(Box::new(r?)))
    }

    fn stmt(&mut self, s: &ast::Stmt) -> Res<ir::Node> {
        use ast::StmtKind as K;
        Ok(match &s.kind {
            K::Block(b) => self.block(&b.stmts)?,
            K::Decl(ds) => ir::Node::DeclarationBlock(
                ds.iter().map(|d| self.declaration(d)).collect::<Res<_>>()?,
            ),
            K::Expr(e) => {
                let t = self.statement_expr(e)?;
                ir::Node::StatementExpression(Box::new(t))
            }
            K::Return(e) => match e {
                None if self.ret == "void" => ir::Node::Return(None),
                None => return compile_error(s.pos, "not all paths provide a return value"),
                Some(e) => {
                    if self.ret == "void" {
                        return compile_error(s.pos, "cannot return a value from void");
                    }
                    let t = self.expr(e)?;
                    let ret = self.ret.clone();
                    let t = self.coerce(e.pos, t, &ret, false)?;
                    ir::Node::Return(Some(Box::new(t.node)))
                }
            },
            K::If {
                cond,
                then,
                otherwise,
            } => {
                let c = Box::new(self.cond(cond)?);
                let then = Box::new(self.scoped(|cx| cx.stmt(then))?);
                match otherwise {
                    None => ir::Node::If { cond: c, then },
                    Some(o) => ir::Node::IfElse {
                        cond: c,
                        then,
                        otherwise: Box::new(self.scoped(|cx| cx.stmt(o))?),
                    },
                }
            }
            K::While { cond, body } => {
                let continuous = ty::is_true(cond);
                let c = Box::new(self.cond(cond)?);
                ir::Node::While {
                    continuous,
                    cond: Some(c),
                    body: self.body(body)?,
                }
            }
            K::Do { body, cond } => {
                let continuous = ty::is_true(cond);
                self.loops += 1;
                let b = self.block(&body.stmts);
                self.loops -= 1;
                let c = Box::new(self.cond(cond)?);
                ir::Node::DoWhile {
                    continuous,
                    cond: Some(c),
                    body: Some(Box::new(b?)),
                }
            }
            K::For {
                init,
                cond,
                after,
                body,
            } => self.scoped(|cx| {
                let init = match init {
                    None => None,
                    Some(ast::ForInit::Decl(ds)) => Some(Box::new(ir::Node::DeclarationBlock(
                        ds.iter().map(|d| cx.declaration(d)).collect::<Res<_>>()?,
                    ))),
                    Some(ast::ForInit::Expr(e)) => Some(Box::new(cx.statement_expr(e)?)),
                };
                let continuous = cond.as_ref().is_none_or(ty::is_true);
                let cond = match cond {
                    None => None,
                    Some(c) => Some(Box::new(cx.cond(c)?)),
                };
                let after = match after {
                    None => None,
                    Some(e) => Some(Box::new(cx.statement_expr(e)?)),
                };
                Ok(ir::Node::For {
                    continuous,
                    init,
                    cond,
                    after,
                    body: cx.body(body)?,
                })
            })?,
            K::Break | K::Continue if self.loops == 0 => {
                return compile_error(s.pos, "break or continue outside of a loop")
            }
            K::Break => ir::Node::Break,
            K::Continue => ir::Node::Continue,
            K::Each { .. } => return unsupported("for-each loop"),
            K::Try { .. } => return unsupported("try"),
            K::Throw(_) => return unsupported("throw"),
        })
    }

    fn declaration(&mut self, d: &ast::Declaration) -> Res<ir::Node> {
        let ty = ty::resolve(&d.ty)?;
        if ty == "void" {
            return compile_error(d.pos, "a variable cannot be void");
        }
        let init = match &d.init {
            None => None,
            Some(e) => {
                let t = self.expr(e)?;
                Some(Box::new(self.coerce(e.pos, t, &ty, false)?.node))
            }
        };
        self.declare(d.pos, &d.name, &ty)?;
        Ok(ir::Node::Declaration {
            name: d.name.clone(),
            ty,
            init,
        })
    }
}
