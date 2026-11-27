//! Expressions: typing, conversion and the operators.

use super::literal::number;
use super::ty::{boxed, is_numeric, promote, promote_one, widens, STRING};
use super::{compile_error, unsupported, Cx, Res};
use crate::front::ast::{self, AssignOp, BinOp, ExprKind, UnOp};
use crate::ir::{self, Constant, Node};
use crate::types::{words, DEF};

/// A lowered expression and its Painless type (`"null"` for the literal).
pub struct Typed {
    pub node: Node,
    pub ty: String,
}

/// `BinaryMath`'s flag for a compound assignment (`def.rs`'s `COMPOUND`).
const COMPOUND: i32 = 1 << 1;

fn typed(node: Node, ty: &str) -> Typed {
    Typed {
        node,
        ty: ty.to_string(),
    }
}

fn no_cast() -> ir::Cast {
    ir::Cast {
        from: String::new(),
        to: String::new(),
        explicit: false,
        unbox_from: None,
        unbox_to: None,
        box_from: None,
        box_to: None,
    }
}

impl Cx<'_> {
    /// Converts `t` to `to`, implicitly or (`explicit`) as a cast does.
    pub(super) fn coerce(&self, pos: u32, t: Typed, to: &str, explicit: bool) -> Res<Typed> {
        let from = t.ty.as_str();
        if from == to {
            return Ok(t);
        }
        let refuse = || {
            compile_error(
                pos,
                format!(
                    "cannot cast from [{}] to [{to}]",
                    crate::types::canonical(from)
                ),
            )
        };
        if from == "void" || to == "void" {
            return refuse();
        }
        if from == "null" {
            return if super::ty::is_numeric(to) || to == "boolean" {
                refuse()
            } else {
                Ok(typed(Node::Null { ty: to.into() }, to))
            };
        }
        let cast = |from: &str, node: Node, extra: ir::Cast| {
            typed(
                Node::Cast {
                    ty: to.into(),
                    cast: Some(ir::Cast {
                        from: from.into(),
                        to: to.into(),
                        explicit,
                        ..extra
                    }),
                    child: Box::new(node),
                },
                to,
            )
        };
        let from_prim = is_numeric(from) || from == "boolean";
        let to_prim = is_numeric(to) || to == "boolean";
        if to == "java.lang.Object" && !from_prim {
            return Ok(typed(t.node, to));
        }
        if to == DEF {
            return Ok(if from_prim {
                cast(
                    boxed(from),
                    t.node,
                    ir::Cast {
                        box_from: Some(from.into()),
                        ..no_cast()
                    },
                )
            } else {
                typed(t.node, DEF)
            });
        }
        if from == DEF {
            return if to_prim || to == STRING {
                Ok(cast(DEF, t.node, no_cast()))
            } else {
                refuse()
            };
        }
        if let (
            Node::Constant {
                value: Constant::Int(v),
                ..
            },
            true,
        ) = (
            &t.node,
            matches!(to, "byte" | "short" | "char") && from == "int",
        ) {
            let folded = match to {
                "byte" => i8::try_from(*v).ok().map(Constant::Byte),
                "short" => i16::try_from(*v).ok().map(Constant::Short),
                _ => u16::try_from(*v).ok().map(Constant::Char),
            };
            if let Some(value) = folded {
                return Ok(typed(
                    Node::Constant {
                        ty: to.into(),
                        value,
                    },
                    to,
                ));
            }
        }
        if from_prim && to_prim {
            let ok = if from == "boolean" || to == "boolean" {
                false
            } else {
                explicit || widens(from, to)
            };
            return if ok {
                Ok(cast(from, t.node, no_cast()))
            } else {
                refuse()
            };
        }
        if from == STRING && to == "char" {
            return match &t.node {
                Node::Constant {
                    value: Constant::Str(s),
                    ..
                } if s.encode_utf16().count() == 1 => Ok(typed(
                    Node::Constant {
                        ty: "char".into(),
                        value: Constant::Char(s.encode_utf16().next().unwrap_or(0)),
                    },
                    "char",
                )),
                _ if explicit => Ok(cast(STRING, t.node, no_cast())),
                _ => refuse(),
            };
        }
        if from == "char" && to == STRING && explicit {
            return Ok(cast("char", t.node, no_cast()));
        }
        refuse()
    }

    pub(super) fn expr(&mut self, e: &ast::Expr) -> Res<Typed> {
        Ok(match &e.kind {
            ExprKind::Num(kind, text) => {
                let (value, ty) = number(e.pos, *kind, text, false)?;
                typed(
                    Node::Constant {
                        ty: ty.into(),
                        value,
                    },
                    ty,
                )
            }
            ExprKind::Bool(b) => typed(
                Node::Constant {
                    ty: "boolean".into(),
                    value: Constant::Bool(*b),
                },
                "boolean",
            ),
            ExprKind::Null => typed(Node::Null { ty: "null".into() }, "null"),
            ExprKind::Str(s) => typed(
                Node::Constant {
                    ty: STRING.into(),
                    value: Constant::Str(s.clone()),
                },
                STRING,
            ),
            ExprKind::Paren(inner) => self.expr(inner)?,
            ExprKind::Var(name) => match self.lookup(name) {
                Some(ty) => {
                    let ty = ty.clone();
                    typed(
                        Node::LoadVariable {
                            ty: ty.clone(),
                            name: name.clone(),
                        },
                        &ty,
                    )
                }
                None => return unsupported(format!("variable [{name}]")),
            },
            ExprKind::Unary(UnOp::Minus, inner) => match &inner.kind {
                ExprKind::Num(kind, text) => {
                    let (value, ty) = number(inner.pos, *kind, text, true)?;
                    typed(
                        Node::Constant {
                            ty: ty.into(),
                            value,
                        },
                        ty,
                    )
                }
                _ => {
                    let t = self.expr(inner)?;
                    self.unary(e.pos, UnOp::Minus, t)?
                }
            },
            ExprKind::Unary(op, inner) => {
                let t = self.expr(inner)?;
                self.unary(e.pos, *op, t)?
            }
            ExprKind::Binary(op, l, r) => {
                let l = self.expr(l)?;
                let r = self.expr(r)?;
                self.binary(e.pos, *op, l, r, 0)?
            }
            ExprKind::Cond(c, a, b) => self.conditional(c, a, b)?,
            ExprKind::Cast(ty, inner) => {
                let to = super::ty::resolve(ty)?;
                let t = self.expr(inner)?;
                self.coerce(e.pos, t, &to, true)?
            }
            ExprKind::Assign(op, target, value) => self.assign(e.pos, *op, target, value, true)?,
            ExprKind::IncDec {
                prefix: true,
                inc,
                target,
            } => self.incdec(e.pos, *inc, target, true)?,
            ExprKind::IncDec { .. } => return unsupported("postfix increment as a value"),
            ExprKind::CallLocal(name, args) => self.call_local(e.pos, name, args)?,
            other => return unsupported(kind_name(other)),
        })
    }

    /// An expression used as a statement: an assignment, an increment or a
    /// call.
    pub(super) fn statement_expr(&mut self, e: &ast::Expr) -> Res<Node> {
        match &e.kind {
            ExprKind::Assign(op, target, value) => {
                Ok(self.assign(e.pos, *op, target, value, false)?.node)
            }
            ExprKind::IncDec { inc, target, .. } => {
                Ok(self.incdec(e.pos, *inc, target, false)?.node)
            }
            ExprKind::CallLocal(name, args) => Ok(self.call_local(e.pos, name, args)?.node),
            ExprKind::Call { .. } => unsupported("method call"),
            _ => compile_error(e.pos, "not a statement"),
        }
    }

    fn unary(&mut self, pos: u32, op: UnOp, t: Typed) -> Res<Typed> {
        if op == UnOp::Not {
            let t = self.coerce(pos, t, "boolean", false)?;
            return Ok(typed(
                Node::UnaryMath {
                    ty: "boolean".into(),
                    op: "NOT".into(),
                    unary_ty: Some("boolean".into()),
                    cat: false,
                    explicit: false,
                    child: Box::new(t.node),
                },
                "boolean",
            ));
        }
        let Some(p) = promote_one(&t.ty) else {
            return compile_error(pos, format!("cannot apply unary operator to [{}]", t.ty));
        };
        if op == UnOp::BitNot && !matches!(p.as_str(), "int" | "long" | DEF) {
            return compile_error(pos, format!("cannot apply [~] to [{}]", t.ty));
        }
        let t = self.coerce(pos, t, &p, false)?;
        let name = match op {
            UnOp::Minus => "SUB",
            UnOp::Plus => "ADD",
            _ => "BWNOT",
        };
        Ok(typed(
            Node::UnaryMath {
                ty: p.clone(),
                op: name.into(),
                unary_ty: Some(p.clone()),
                cat: false,
                explicit: false,
                child: Box::new(t.node),
            },
            &p,
        ))
    }

    fn conditional(&mut self, c: &ast::Expr, a: &ast::Expr, b: &ast::Expr) -> Res<Typed> {
        let cond = self.cond(c)?;
        let (l, r) = (self.expr(a)?, self.expr(b)?);
        let ty = if l.ty == r.ty {
            l.ty.clone()
        } else if l.ty == DEF || r.ty == DEF {
            DEF.into()
        } else if l.ty == "null" {
            r.ty.clone()
        } else if r.ty == "null" {
            l.ty.clone()
        } else if let Some(n) = narrow_to(&l, &r).or_else(|| narrow_to(&r, &l)) {
            n
        } else if let Some(p) = promote(&l.ty, &r.ty) {
            p
        } else {
            DEF.into()
        };
        if ty == "null" {
            return unsupported("conditional of two nulls");
        }
        let l = self.coerce(a.pos, l, &ty, false)?;
        let r = self.coerce(b.pos, r, &ty, false)?;
        Ok(typed(
            Node::Conditional {
                ty: ty.clone(),
                cond: Box::new(cond),
                left: Box::new(l.node),
                right: Box::new(r.node),
            },
            &ty,
        ))
    }

    pub(super) fn binary(
        &mut self,
        pos: u32,
        op: BinOp,
        l: Typed,
        r: Typed,
        flags: i32,
    ) -> Res<Typed> {
        use BinOp as B;
        match op {
            B::And | B::Or => {
                let l = self.coerce(pos, l, "boolean", false)?;
                let r = self.coerce(pos, r, "boolean", false)?;
                Ok(typed(
                    Node::Boolean {
                        ty: "boolean".into(),
                        op: if op == B::And { "AND" } else { "OR" }.into(),
                        left: Box::new(l.node),
                        right: Box::new(r.node),
                    },
                    "boolean",
                ))
            }
            B::Lt | B::Lte | B::Gt | B::Gte => {
                let Some(p) = promote(&l.ty, &r.ty) else {
                    return compile_error(
                        pos,
                        format!("cannot compare [{}] with [{}]", l.ty, r.ty),
                    );
                };
                self.compare(pos, op, &p, l, r)
            }
            B::Eq | B::Ne | B::EqRef | B::NeRef => self.equality(pos, op, l, r),
            B::Add if l.ty == STRING || r.ty == STRING => {
                let mut args = Vec::new();
                for t in [l, r] {
                    match t.node {
                        Node::StringConcat { args: inner, .. } => args.extend(inner),
                        node => args.push(node),
                    }
                }
                Ok(typed(
                    Node::StringConcat {
                        ty: STRING.into(),
                        args,
                    },
                    STRING,
                ))
            }
            B::Mul | B::Div | B::Rem | B::Add | B::Sub => {
                let Some(p) = promote(&l.ty, &r.ty) else {
                    return compile_error(
                        pos,
                        format!("cannot apply operator to [{}] and [{}]", l.ty, r.ty),
                    );
                };
                let name = match op {
                    B::Mul => "MUL",
                    B::Div => "DIV",
                    B::Rem => "REM",
                    B::Add => "ADD",
                    _ => "SUB",
                };
                self.math(pos, name, &p, l, r, flags)
            }
            B::BitAnd | B::Xor | B::BitOr => {
                let name = match op {
                    B::BitAnd => "BWAND",
                    B::Xor => "XOR",
                    _ => "BWOR",
                };
                if l.ty == "boolean" && r.ty == "boolean" {
                    return self.math(pos, name, "boolean", l, r, flags);
                }
                let boolish = |t: &str| t == "boolean" || t == DEF;
                if (l.ty == DEF || r.ty == DEF) && boolish(&l.ty) && boolish(&r.ty) {
                    return self.math(pos, name, DEF, l, r, flags);
                }
                match promote(&l.ty, &r.ty) {
                    Some(p) if matches!(p.as_str(), "int" | "long" | DEF) => {
                        self.math(pos, name, &p, l, r, flags)
                    }
                    _ => compile_error(
                        pos,
                        format!("cannot apply bitwise operator to [{}] and [{}]", l.ty, r.ty),
                    ),
                }
            }
            B::Lsh | B::Rsh | B::Ush => unsupported("shift operators"),
            B::Find | B::Match => unsupported("regex operators"),
            B::Elvis => unsupported("elvis operator"),
        }
    }

    fn math(&mut self, pos: u32, op: &str, p: &str, l: Typed, r: Typed, flags: i32) -> Res<Typed> {
        let l = self.coerce(pos, l, p, false)?;
        let r = self.coerce(pos, r, p, false)?;
        Ok(typed(
            Node::BinaryMath {
                ty: p.into(),
                op: op.into(),
                binary_ty: p.into(),
                shift_ty: None,
                flags,
                regex_limit: 0,
                left: Box::new(l.node),
                right: Box::new(r.node),
            },
            p,
        ))
    }

    fn compare(&mut self, pos: u32, op: BinOp, p: &str, l: Typed, r: Typed) -> Res<Typed> {
        let l = self.coerce(pos, l, p, false)?;
        let r = self.coerce(pos, r, p, false)?;
        let name = match op {
            BinOp::Lt => "LT",
            BinOp::Lte => "LTE",
            BinOp::Gt => "GT",
            BinOp::Gte => "GTE",
            BinOp::Eq => "EQ",
            BinOp::Ne => "NE",
            BinOp::EqRef => "EQR",
            _ => "NER",
        };
        Ok(typed(
            Node::Comparison {
                ty: "boolean".into(),
                op: name.into(),
                cmp_ty: p.into(),
                left: Box::new(l.node),
                right: Box::new(r.node),
            },
            "boolean",
        ))
    }

    fn equality(&mut self, pos: u32, op: BinOp, l: Typed, r: Typed) -> Res<Typed> {
        let reference = matches!(op, BinOp::EqRef | BinOp::NeRef);
        let objects = |t: &Typed| t.ty == STRING || t.ty == "null";
        if objects(&l) && objects(&r) {
            return self.compare(pos, op, "java.lang.Object", l, r);
        }
        if reference {
            return unsupported("reference equality outside strings");
        }
        if l.ty == DEF || r.ty == DEF {
            return self.compare(pos, op, DEF, l, r);
        }
        if l.ty == "boolean" && r.ty == "boolean" {
            return self.compare(pos, op, "boolean", l, r);
        }
        match promote(&l.ty, &r.ty) {
            Some(p) => self.compare(pos, op, &p, l, r),
            None => compile_error(pos, format!("cannot compare [{}] with [{}]", l.ty, r.ty)),
        }
    }

    fn assign(
        &mut self,
        pos: u32,
        op: AssignOp,
        target: &ast::Expr,
        value: &ast::Expr,
        want_value: bool,
    ) -> Res<Typed> {
        let (name, ty) = self.target(target)?;
        let v = self.expr(value)?;
        let child = match op {
            AssignOp::Set => self.coerce(value.pos, v, &ty, false)?,
            AssignOp::With(b) => {
                let load = typed(
                    Node::LoadVariable {
                        ty: ty.clone(),
                        name: name.clone(),
                    },
                    &ty,
                );
                let r = self.binary(pos, b, load, v, COMPOUND)?;
                self.coerce(pos, r, &ty, true)?
            }
        };
        Ok(self.store(&name, &ty, child, want_value))
    }

    fn incdec(&mut self, pos: u32, inc: bool, target: &ast::Expr, want_value: bool) -> Res<Typed> {
        let (name, ty) = self.target(target)?;
        let load = typed(
            Node::LoadVariable {
                ty: ty.clone(),
                name: name.clone(),
            },
            &ty,
        );
        let one = typed(
            Node::Constant {
                ty: "int".into(),
                value: Constant::Int(1),
            },
            "int",
        );
        let op = if inc { BinOp::Add } else { BinOp::Sub };
        let r = self.binary(pos, op, load, one, COMPOUND)?;
        let child = self.coerce(pos, r, &ty, true)?;
        Ok(self.store(&name, &ty, child, want_value))
    }

    fn target(&self, target: &ast::Expr) -> Res<(String, String)> {
        match &unparen(target).kind {
            ExprKind::Var(name) => match self.lookup(name) {
                Some(ty) => Ok((name.clone(), ty.clone())),
                None => unsupported(format!("variable [{name}]")),
            },
            ExprKind::Paren(_) => {
                compile_error(target.pos, "cannot assign to a parenthesised expression")
            }
            other => unsupported(format!("assignment to {}", kind_name(other))),
        }
    }

    /// The store of `child` to a local; with `want_value`, its value stays
    /// on the stack as the expression's.
    fn store(&self, name: &str, ty: &str, child: Typed, want_value: bool) -> Typed {
        let (result, child) = if want_value {
            (
                ty,
                Node::Dup {
                    ty: ty.into(),
                    size: words(ty) as i32,
                    depth: 0,
                    child: Box::new(child.node),
                },
            )
        } else {
            ("void", child.node)
        };
        typed(
            Node::StoreVariable {
                ty: result.into(),
                store_ty: ty.into(),
                name: name.into(),
                child: Box::new(child),
            },
            result,
        )
    }

    fn call_local(&mut self, pos: u32, name: &str, args: &[ast::Arg]) -> Res<Typed> {
        let Some(sig) = self.funcs.get(name).cloned() else {
            return unsupported(format!("function [{name}]"));
        };
        if sig.params.len() != args.len() {
            return compile_error(
                pos,
                format!("function [{name}] takes {} arguments", sig.params.len()),
            );
        }
        let mut lowered = Vec::new();
        for (arg, ty) in args.iter().zip(&sig.params) {
            let ast::Arg::Expr(e) = arg else {
                return unsupported("lambda or method reference argument");
            };
            let t = self.expr(e)?;
            lowered.push(self.coerce(e.pos, t, ty, false)?.node);
        }
        Ok(typed(
            Node::InvokeCallMember {
                ty: sig.ret.clone(),
                member: ir::Member::Local {
                    name: name.into(),
                    internal: false,
                    is_static: true,
                    params: sig.params,
                    ret: sig.ret.clone(),
                },
                args: lowered,
            },
            &sig.ret,
        ))
    }
}

/// `byte`, `short` or `char` when `a` is one and `b` an `int` constant that
/// fits it: the conditional then keeps the narrow type.
fn narrow_to(a: &Typed, b: &Typed) -> Option<String> {
    let Node::Constant {
        value: Constant::Int(v),
        ..
    } = &b.node
    else {
        return None;
    };
    let fits = match a.ty.as_str() {
        "byte" => i8::try_from(*v).is_ok(),
        "short" => i16::try_from(*v).is_ok(),
        "char" => u16::try_from(*v).is_ok(),
        _ => false,
    };
    (fits && b.ty == "int").then(|| a.ty.clone())
}

fn unparen(e: &ast::Expr) -> &ast::Expr {
    match &e.kind {
        ExprKind::Paren(inner) => unparen(inner),
        _ => e,
    }
}

fn kind_name(k: &ExprKind) -> &'static str {
    match k {
        ExprKind::Regex(..) => "regex literal",
        ExprKind::List(_) => "list initialiser",
        ExprKind::Map(_) => "map initialiser",
        ExprKind::CallLocal(..) => "function call",
        ExprKind::NewObject(..) => "new object",
        ExprKind::NewArray(..) | ExprKind::NewArrayInit(..) => "new array",
        ExprKind::Call { .. } => "method call",
        ExprKind::Field { .. } => "field access",
        ExprKind::Index(..) => "index",
        ExprKind::InstanceOf(..) => "instanceof",
        ExprKind::Assign(..) => "assignment",
        _ => "expression",
    }
}
