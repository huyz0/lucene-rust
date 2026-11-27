//! Method calls and field reads: against a `def` receiver by name at run
//! time, against a class by what the allowlist lists (by name and arity,
//! as Painless resolves them), and against a class named in the script for
//! its static members.

use super::expr::Typed;
use super::ty::{from_table, is_primitive, STRING};
use super::{compile_error, unsupported, Cx, Res};
use crate::front::allowlist::{Allowlist, MethodSig};
use crate::front::ast::{self, ExprKind};
use crate::ir::{self, Node};
use crate::types::{canonical, DEF};

/// What a call or a field is read from.
enum Prefix {
    /// A class named by the script: its static members.
    Class(String),
    /// A value, lowered, of this type.
    Value(Box<Typed>),
}

fn typed(node: Node, ty: &str) -> Typed {
    Typed {
        node,
        ty: ty.to_string(),
    }
}

/// `left` evaluated, then `right` applied to it.
fn chain(ty: &str, left: Node, right: Node) -> Node {
    Node::BinaryImpl {
        ty: ty.into(),
        left: Box::new(left),
        right: Box::new(right),
    }
}

const MAP: &str = "java.util.Map";

/// The getter a field name stands for: `getX` or `isX`.
fn getter_of<'a>(class: &str, name: &str) -> Option<(String, &'a MethodSig)> {
    let mut c = name.chars();
    let cap: String = c.next()?.to_uppercase().chain(c).collect();
    ["get", "is"].into_iter().find_map(|p| {
        let n = format!("{p}{cap}");
        find_method(class, &n, 0, false).map(|m| (n, m))
    })
}

/// The name a chain of plain member reads spells (`Map.Entry`,
/// `java.lang.Math`), with its first word.
fn dotted(e: &ast::Expr) -> Option<(&str, String)> {
    match &e.kind {
        ExprKind::Var(name) => Some((name, name.clone())),
        ExprKind::Field {
            recv,
            name,
            null_safe: false,
        } => dotted(recv).map(|(root, written)| (root, format!("{written}.{name}"))),
        _ => None,
    }
}

/// The wrapper class of a primitive, or the type itself.
fn lookup_class(ty: &str) -> String {
    if is_primitive(ty) {
        super::ty::boxed(ty).to_string()
    } else {
        ty.to_string()
    }
}

/// The method of `class` Painless resolves: by name and arity, the class's
/// own first, then those it inherits.
pub(super) fn find_method<'a>(
    class: &str,
    name: &str,
    arity: usize,
    is_static: bool,
) -> Option<&'a MethodSig> {
    Allowlist::get()
        .methods(class, name)
        .into_iter()
        .find(|m| m.params.len() == arity && (m.is_static == is_static || m.augment.is_some()))
}

impl Cx<'_> {
    /// The prefix of a member: a class when it names one and no variable
    /// hides it.
    fn prefix(&mut self, e: &ast::Expr) -> Res<Prefix> {
        if let Some((root, written)) = dotted(e) {
            if self.lookup(root).is_none() {
                if let Some(class) = Allowlist::get().resolve(&written) {
                    return Ok(Prefix::Class(class.to_string()));
                }
            }
        }
        self.expr(e).map(|t| Prefix::Value(Box::new(t)))
    }

    /// `recv.name(args)` or `recv?.name(args)`.
    pub(super) fn call(
        &mut self,
        pos: u32,
        recv: &ast::Expr,
        name: &str,
        args: &[ast::Arg],
        null_safe: bool,
    ) -> Res<Typed> {
        let prefix = self.prefix(recv)?;
        let unknown = |class: &str| {
            compile_error(
                pos,
                format!(
                    "Unknown call [{name}] with [{}] arguments on type [{}].",
                    args.len(),
                    canonical(class)
                ),
            )
        };
        let (left, receiver, class, is_static) = match prefix {
            Prefix::Class(c) => (Node::Static { ty: c.clone() }, c.clone(), c, true),
            Prefix::Value(t) if t.ty == DEF => {
                return self.call_def(pos, *t, name, args, null_safe);
            }
            Prefix::Value(t) if t.ty == "void" || t.ty == "null" => {
                return compile_error(pos, format!("cannot call a method on [{}]", t.ty));
            }
            Prefix::Value(t) => {
                let class = lookup_class(&t.ty);
                (t.node, t.ty, class, false)
            }
        };
        if null_safe && (is_static || is_primitive(&receiver)) {
            return unsupported("null-safe call on a class or a primitive");
        }
        let Some(sig) = find_method(&class, name, args.len(), is_static) else {
            return unknown(&class);
        };
        let mut lowered = Vec::new();
        for (a, param) in args.iter().zip(&sig.params) {
            lowered.push(self.arg(a, param)?);
        }
        let ret = from_table(&sig.ret);
        let method = ir::Method {
            target: class.clone(),
            name: name.to_string(),
            declaring: sig.augment.clone().unwrap_or_else(|| class.clone()),
            is_static: sig.augment.is_none() && sig.is_static,
            params: sig.params.iter().map(|p| from_table(p)).collect(),
            ret: ret.clone(),
        };
        let boxed = (!is_static && is_primitive(&receiver)).then(|| receiver.clone());
        let result = if null_safe && is_primitive(&ret) {
            super::ty::boxed(&ret).to_string()
        } else {
            ret.clone()
        };
        let mut call = Node::InvokeCall {
            ty: ret,
            method: Some(method),
            boxed,
            args: lowered,
        };
        if null_safe {
            call = Node::NullSafeSub {
                ty: result.clone(),
                child: Box::new(call),
            };
        }
        Ok(typed(chain(&result, left, call), &result))
    }

    /// A call on a `def` receiver: resolved at run time, its arguments as
    /// written.
    fn call_def(
        &mut self,
        _pos: u32,
        recv: Typed,
        name: &str,
        args: &[ast::Arg],
        null_safe: bool,
    ) -> Res<Typed> {
        let mut lowered = Vec::new();
        for a in args {
            lowered.push(self.def_arg(a)?);
        }
        let mut call = Node::InvokeCallDef {
            ty: DEF.into(),
            name: name.into(),
            args: lowered,
        };
        if null_safe {
            call = Node::NullSafeSub {
                ty: DEF.into(),
                child: Box::new(call),
            };
        }
        Ok(typed(chain(DEF, recv.node, call), DEF))
    }

    /// `recv.name` or `recv?.name`: a static field of a class, a field or
    /// getter of a value, or a `def` member by name.
    pub(super) fn field(
        &mut self,
        pos: u32,
        recv: &ast::Expr,
        name: &str,
        null_safe: bool,
    ) -> Res<Typed> {
        let prefix = self.prefix(recv)?;
        let al = Allowlist::get();
        let (left, class, is_static) = match prefix {
            Prefix::Class(c) => (Node::Static { ty: c.clone() }, c, true),
            Prefix::Value(t) if t.ty == DEF => {
                let mut read = Node::LoadDotDef {
                    ty: DEF.into(),
                    name: name.into(),
                };
                if null_safe {
                    read = Node::NullSafeSub {
                        ty: DEF.into(),
                        child: Box::new(read),
                    };
                }
                return Ok(typed(chain(DEF, t.node, read), DEF));
            }
            Prefix::Value(t) if t.ty == "void" || t.ty == "null" => {
                return compile_error(pos, format!("cannot read a field of [{}]", t.ty));
            }
            Prefix::Value(t) if t.ty.starts_with('[') => {
                if name != "length" || null_safe {
                    return unsupported("field of an array");
                }
                let read = Node::LoadDotArrayLength { ty: "int".into() };
                return Ok(typed(chain("int", t.node, read), "int"));
            }
            Prefix::Value(t) => {
                let class = lookup_class(&t.ty);
                if is_primitive(&t.ty) {
                    return unsupported("field of a primitive");
                }
                (t.node, class, false)
            }
        };
        let (read, ty) =
            if let Some(f) = al.field(&class, name).filter(|f| f.is_static == is_static) {
                let ty = from_table(&f.ty);
                let read = Node::LoadDot {
                    ty: ty.clone(),
                    field: ir::Field {
                        class: class.clone(),
                        name: name.into(),
                        is_static,
                        ty: ty.clone(),
                    },
                };
                (read, ty)
            } else if is_static {
                return compile_error(
                    pos,
                    format!("Unknown field [{name}] for type [{}].", canonical(&class)),
                );
            } else if let Some((getter_name, sig)) = getter_of(&class, name) {
                let ty = from_table(&sig.ret);
                let method = ir::Method {
                    target: class.clone(),
                    name: getter_name,
                    declaring: sig.augment.clone().unwrap_or_else(|| class.clone()),
                    is_static: false,
                    params: Vec::new(),
                    ret: ty.clone(),
                };
                let read = Node::LoadDotShortcut {
                    ty: ty.clone(),
                    getter: Some(method),
                };
                (read, ty)
            } else if al.is_assignable(&class, MAP) {
                let get = find_method(&class, "get", 1, false)
                    .ok_or_else(|| super::Refusal::Unsupported("map without get".into()))?;
                let method = ir::Method {
                    target: class.clone(),
                    name: "get".into(),
                    declaring: get.augment.clone().unwrap_or_else(|| class.clone()),
                    is_static: false,
                    params: get.params.iter().map(|p| from_table(p)).collect(),
                    ret: DEF.into(),
                };
                let key = Node::Constant {
                    ty: STRING.into(),
                    value: ir::Constant::Str(name.into()),
                };
                let read = chain(
                    DEF,
                    key,
                    Node::LoadMapShortcut {
                        ty: DEF.into(),
                        getter: Some(method),
                    },
                );
                (read, DEF.to_string())
            } else if al.is_assignable(&class, "java.util.List") {
                return unsupported("list shortcut");
            } else {
                return compile_error(
                    pos,
                    format!("Unknown field [{name}] for type [{}].", canonical(&class)),
                );
            };
        let result = if null_safe && is_primitive(&ty) {
            super::ty::boxed(&ty).to_string()
        } else {
            ty
        };
        let read = if null_safe {
            Node::NullSafeSub {
                ty: result.clone(),
                child: Box::new(read),
            }
        } else {
            read
        };
        Ok(typed(chain(&result, left, read), &result))
    }
}
