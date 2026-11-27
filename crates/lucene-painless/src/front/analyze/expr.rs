//! Expressions: typing, conversion and the operators.

use super::literal::number;
use super::place::PlaceUpdate;
use super::ty::{boxed, is_numeric, is_primitive, promote, promote_one, unboxed, widens, STRING};
use super::{compile_error, unsupported, Cx, Res};
use crate::front::allowlist::Allowlist;
use crate::front::ast::{self, AssignOp, BinOp, ExprKind, UnOp};
use crate::ir::{self, Constant, Node};
use crate::types::{words, DEF};

/// A lowered expression and its Painless type (`"null"` for the literal).
pub struct Typed {
    pub node: Node,
    pub ty: String,
}

const PATTERN: &str = "java.util.regex.Pattern";

/// The compiler's default `regex_limit_factor`, which `FIND` and `MATCH` carry.
pub(super) const REGEX_LIMIT_FACTOR: i32 = 6;

/// `BinaryMath`'s flag for a compound assignment (`def.rs`'s `COMPOUND`).
pub(super) const COMPOUND: i32 = 1 << 1;

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
            return if to_prim
                || to == STRING
                || to.starts_with('[')
                || Allowlist::get().class(to).is_some()
            {
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
        if from_prim
            || to_prim
            || (Allowlist::get().class(from).is_some() && Allowlist::get().class(to).is_some())
        {
            return self.coerce_class(pos, t, to, explicit);
        }
        if from.starts_with('[') && to.starts_with('[') {
            return unsupported(format!("conversion of [{from}] to [{to}]"));
        }
        refuse()
    }

    /// A conversion with a class on one side: boxing, unboxing, or the
    /// class hierarchy. What Painless would do and this does not know is
    /// refused as unsupported, never guessed.
    fn coerce_class(&self, pos: u32, t: Typed, to: &str, explicit: bool) -> Res<Typed> {
        let from = t.ty.clone();
        let al = Allowlist::get();
        let cast = |extra: ir::Cast, node: Node| {
            typed(
                Node::Cast {
                    ty: to.into(),
                    cast: Some(ir::Cast {
                        from: from.clone(),
                        to: to.into(),
                        explicit,
                        ..extra
                    }),
                    child: Box::new(node),
                },
                to,
            )
        };
        let refuse = || {
            compile_error(
                pos,
                format!(
                    "cannot cast from [{}] to [{}]",
                    crate::types::canonical(&from),
                    crate::types::canonical(to)
                ),
            )
        };
        if (from == STRING && is_primitive(to)) || (is_primitive(&from) && to == STRING) {
            return refuse();
        }
        if is_primitive(&from) {
            if let Some(p) = unboxed(to) {
                let ok = from == p
                    || widens(&from, p)
                    || (explicit && is_numeric(&from) && is_numeric(p));
                return if ok {
                    Ok(cast(
                        ir::Cast {
                            box_to: Some(p.into()),
                            ..no_cast()
                        },
                        t.node,
                    ))
                } else {
                    refuse()
                };
            }
            if al.is_assignable(boxed(&from), to) {
                return Ok(cast(
                    ir::Cast {
                        box_from: Some(from.clone()),
                        ..no_cast()
                    },
                    t.node,
                ));
            }
            return if al.class(to).is_some() && !explicit {
                refuse()
            } else {
                unsupported(format!("cast of [{from}] to [{to}]"))
            };
        }
        if let (Some(p), true) = (unboxed(&from), is_primitive(to)) {
            let ok = widens(p, to) || (explicit && is_numeric(p) && is_numeric(to));
            return if ok {
                Ok(cast(
                    ir::Cast {
                        unbox_from: Some(p.into()),
                        ..no_cast()
                    },
                    t.node,
                ))
            } else {
                refuse()
            };
        }
        if is_primitive(to) {
            return unsupported(format!("cast of [{from}] to [{to}]"));
        }
        if let (Some(p), Some(q)) = (unboxed(&from), unboxed(to)) {
            if is_numeric(p) && is_numeric(q) && p != q && (explicit || widens(p, q)) {
                return Ok(cast(
                    ir::Cast {
                        unbox_from: Some(p.into()),
                        box_to: Some(q.into()),
                        ..no_cast()
                    },
                    t.node,
                ));
            }
        }
        if al.is_assignable(&from, to) {
            return Ok(typed(t.node, to));
        }
        if explicit && al.is_assignable(to, &from) {
            return Ok(cast(no_cast(), t.node));
        }
        if explicit {
            return unsupported(format!("cast of [{from}] to [{to}]"));
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
            ExprKind::Regex(pattern, flags) => {
                let mut bits = 0;
                for f in flags.chars() {
                    bits |= match f {
                        'c' => 128,
                        'i' => 2,
                        'l' => 16,
                        'm' => 8,
                        's' => 32,
                        'U' => 256,
                        'u' => 64,
                        'x' => 4,
                        _ => {
                            return compile_error(e.pos, format!("unknown regex flag [{f}]"));
                        }
                    };
                }
                typed(
                    Node::Constant {
                        ty: PATTERN.into(),
                        value: Constant::Pattern(pattern.clone(), bits),
                    },
                    PATTERN,
                )
            }
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
                None if name == "params" => typed(
                    Node::InvokeCallMember {
                        ty: "java.util.Map".into(),
                        member: ir::Member::Local {
                            name: "getParams".into(),
                            internal: true,
                            is_static: false,
                            params: Vec::new(),
                            ret: "java.util.Map".into(),
                        },
                        args: Vec::new(),
                    },
                    "java.util.Map",
                ),
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
            } => self.incdec(e.pos, *inc, target, true, true)?,
            ExprKind::IncDec { inc, target, .. } => {
                self.incdec(e.pos, *inc, target, true, false)?
            }
            ExprKind::CallLocal(name, args) => self.call_local(e.pos, name, args)?,
            ExprKind::InstanceOf(inner, ty) => {
                let instance = super::ty::resolve(ty)?;
                let t = self.expr(inner)?;
                if t.ty == "void" || t.ty == "null" {
                    return compile_error(inner.pos, "instanceof needs a value");
                }
                let t = if is_primitive(&t.ty) {
                    self.coerce(inner.pos, t, "java.lang.Object", false)?
                } else {
                    t
                };
                typed(
                    Node::Instanceof {
                        ty: "boolean".into(),
                        instance_ty: instance,
                        child: Box::new(t.node),
                    },
                    "boolean",
                )
            }
            ExprKind::NewObject(ty, args) => self.new_object(e.pos, ty, args, true)?,
            ExprKind::NewArray(ty, dims) => self.new_array(e.pos, ty, dims)?,
            ExprKind::NewArrayInit(ty, values) => self.new_array_init(e.pos, ty, values)?,
            ExprKind::Index(recv, index) => self.index_read(recv, index)?,
            ExprKind::List(values) => self.list_init(values)?,
            ExprKind::Map(pairs) => self.map_init(pairs)?,
            ExprKind::Call {
                recv,
                name,
                args,
                null_safe,
            } => self.call(e.pos, recv, name, args, *null_safe)?,
            ExprKind::Field {
                recv,
                name,
                null_safe,
            } => self.field(e.pos, recv, name, *null_safe)?,
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
                Ok(self.incdec(e.pos, *inc, target, false, true)?.node)
            }
            ExprKind::CallLocal(name, args) => Ok(self.call_local(e.pos, name, args)?.node),
            ExprKind::Call {
                recv,
                name,
                args,
                null_safe,
            } => Ok(self.call(e.pos, recv, name, args, *null_safe)?.node),
            ExprKind::NewObject(ty, args) => Ok(self.new_object(e.pos, ty, args, false)?.node),
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
        let plain = |t: &str| is_primitive(t) || matches!(t, DEF | STRING | "null");
        let ty = if l.ty == r.ty {
            l.ty.clone()
        } else if l.ty == DEF || r.ty == DEF {
            DEF.into()
        } else if !(plain(&l.ty) && plain(&r.ty)) {
            // Two different classes (or a class and a primitive): their
            // common type is `Object`.
            "java.lang.Object".into()
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
            B::Lsh | B::Rsh | B::Ush => self.shift(pos, op, l, r, flags),
            B::Find | B::Match => {
                let l = self.coerce(pos, l, STRING, false)?;
                let r = self.coerce(pos, r, PATTERN, false)?;
                let name = if op == B::Find { "FIND" } else { "MATCH" };
                Ok(typed(
                    Node::BinaryMath {
                        ty: "boolean".into(),
                        op: name.into(),
                        binary_ty: "boolean".into(),
                        shift_ty: None,
                        flags,
                        regex_limit: REGEX_LIMIT_FACTOR,
                        left: Box::new(l.node),
                        right: Box::new(r.node),
                    },
                    "boolean",
                ))
            }
            B::Elvis => {
                let ty = if l.ty == r.ty || r.ty == "null" {
                    l.ty.clone()
                } else if l.ty == DEF || r.ty == DEF {
                    DEF.into()
                } else {
                    "java.lang.Object".into()
                };
                if is_primitive(&ty) || ty == "null" {
                    return compile_error(pos, "elvis cannot apply to a primitive");
                }
                let l = self.coerce(pos, l, &ty, false)?;
                let r = self.coerce(pos, r, &ty, false)?;
                Ok(typed(
                    Node::Elvis {
                        ty: ty.clone(),
                        left: Box::new(l.node),
                        right: Box::new(r.node),
                    },
                    &ty,
                ))
            }
        }
    }

    /// `<<`, `>>`, `>>>`: the left operand is promoted to `int` or `long`
    /// and the distance to one of them on its own; `def` on either side
    /// makes both `def`.
    fn shift(&mut self, pos: u32, op: BinOp, l: Typed, r: Typed, flags: i32) -> Res<Typed> {
        let name = match op {
            BinOp::Lsh => "LSH",
            BinOp::Rsh => "RSH",
            _ => "USH",
        };
        let (pl, pr) = (promote_one(&l.ty), promote_one(&r.ty));
        let int_or_long = |t: &Option<String>| {
            t.as_deref()
                .filter(|t| matches!(*t, "int" | "long" | DEF))
                .map(str::to_string)
        };
        let (Some(pl), Some(pr)) = (int_or_long(&pl), int_or_long(&pr)) else {
            return compile_error(pos, format!("cannot shift [{}] by [{}]", l.ty, r.ty));
        };
        let (bt, st) = if pl == DEF || pr == DEF {
            (DEF.to_string(), DEF.to_string())
        } else {
            (pl, pr)
        };
        let l = self.coerce(pos, l, &bt, false)?;
        let r = self.coerce(pos, r, &st, false)?;
        Ok(typed(
            Node::BinaryMath {
                ty: bt.clone(),
                op: name.into(),
                binary_ty: bt.clone(),
                shift_ty: Some(st),
                flags,
                regex_limit: 0,
                left: Box::new(l.node),
                right: Box::new(r.node),
            },
            &bt,
        ))
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
        let objects = |t: &Typed| t.ty == STRING || t.ty == "null";
        if objects(&l) && objects(&r) {
            return self.compare(pos, op, "java.lang.Object", l, r);
        }
        if l.ty == DEF || r.ty == DEF {
            return self.compare(pos, op, DEF, l, r);
        }
        if l.ty == "boolean" && r.ty == "boolean" {
            return self.compare(pos, op, "boolean", l, r);
        }
        if !(is_numeric(&l.ty) && is_numeric(&r.ty))
            && (!is_primitive(&l.ty) || !is_primitive(&r.ty))
        {
            return self.compare(pos, op, "java.lang.Object", l, r);
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
        if let Some(place) = self.place_of(target)? {
            let v = self.expr(value)?;
            let update = match op {
                AssignOp::Set => PlaceUpdate::Set(v, value.pos),
                AssignOp::With(b) => PlaceUpdate::With(b, v),
            };
            return self.assign_place(pos, place, update, want_value);
        }
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

    /// `++x`, `x++`, `--x`, `x--`; a postfix one used for its value leaves
    /// the variable's old value, kept by a `Dup` before the arithmetic.
    fn incdec(
        &mut self,
        pos: u32,
        inc: bool,
        target: &ast::Expr,
        want_value: bool,
        prefix: bool,
    ) -> Res<Typed> {
        if let Some(place) = self.place_of(target)? {
            return self.assign_place(pos, place, PlaceUpdate::Step { inc, prefix }, want_value);
        }
        let (name, ty) = self.target(target)?;
        let mut load = typed(
            Node::LoadVariable {
                ty: ty.clone(),
                name: name.clone(),
            },
            &ty,
        );
        let post = want_value && !prefix;
        if post {
            load.node = Node::Dup {
                ty: ty.clone(),
                size: words(&ty) as i32,
                depth: 0,
                child: Box::new(load.node),
            };
        }
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
        if post {
            return Ok(typed(
                Node::StoreVariable {
                    ty: ty.clone(),
                    store_ty: ty.clone(),
                    name,
                    child: Box::new(child.node),
                },
                &ty,
            ));
        }
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
            lowered.push(self.arg(arg, ty)?);
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

pub(super) fn unparen(e: &ast::Expr) -> &ast::Expr {
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
