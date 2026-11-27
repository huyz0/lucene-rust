//! Type names and the numeric rules: promotion, widening, and what a
//! statement list does at its end.

use super::{unsupported, Res};
use crate::front::ast;
use crate::types::DEF;

pub const STRING: &str = "java.lang.String";

/// The IR name of a written type.
pub fn resolve(t: &ast::Type) -> Res<String> {
    if t.dims > 0 {
        return unsupported("array types");
    }
    Ok(match t.name.as_str() {
        "def" => DEF.to_string(),
        "boolean" | "byte" | "short" | "char" | "int" | "long" | "float" | "double" | "void" => {
            t.name.clone()
        }
        "String" | "java.lang.String" => STRING.to_string(),
        other => return unsupported(format!("type {other}")),
    })
}

pub fn is_numeric(ty: &str) -> bool {
    matches!(
        ty,
        "byte" | "short" | "char" | "int" | "long" | "float" | "double"
    )
}

fn rank(ty: &str) -> u8 {
    match ty {
        "byte" => 1,
        "short" => 2,
        "char" => 2,
        "int" => 3,
        "long" => 4,
        "float" => 5,
        _ => 6,
    }
}

/// The type both operands of an arithmetic or relational operator are
/// converted to: `def` when either is, else the wider of the two, at least
/// `int`. `None` when an operand is not numeric.
pub fn promote(a: &str, b: &str) -> Option<String> {
    let ok = |t: &str, other: &str| is_numeric(t) || t == DEF || (t == "null" && other == DEF);
    if !ok(a, b) || !ok(b, a) {
        return None;
    }
    if a == DEF || b == DEF {
        return Some(DEF.into());
    }
    Some(
        match rank(a).max(rank(b)) {
            6 => "double",
            5 => "float",
            4 => "long",
            _ => "int",
        }
        .into(),
    )
}

/// One operand's promotion (unary operators and shifts): `def` stays,
/// numeric types widen to at least `int`.
pub fn promote_one(a: &str) -> Option<String> {
    promote(a, "int")
}

/// Whether a primitive widens to another implicitly.
pub fn widens(from: &str, to: &str) -> bool {
    if from == to {
        return true;
    }
    if !is_numeric(from) || !is_numeric(to) {
        return false;
    }
    match from {
        "byte" => matches!(to, "short" | "int" | "long" | "float" | "double"),
        "short" => matches!(to, "int" | "long" | "float" | "double"),
        "char" => matches!(to, "int" | "long" | "float" | "double"),
        "int" => matches!(to, "long" | "float" | "double"),
        "long" => matches!(to, "float" | "double"),
        "float" => to == "double",
        _ => false,
    }
}

/// The wrapper class of a primitive.
pub fn boxed(prim: &str) -> &'static str {
    match prim {
        "boolean" => "java.lang.Boolean",
        "byte" => "java.lang.Byte",
        "short" => "java.lang.Short",
        "char" => "java.lang.Character",
        "int" => "java.lang.Integer",
        "long" => "java.lang.Long",
        "float" => "java.lang.Float",
        _ => "java.lang.Double",
    }
}

/// `true` as a condition: a literal.
pub fn is_true(e: &ast::Expr) -> bool {
    match &e.kind {
        ast::ExprKind::Bool(b) => *b,
        ast::ExprKind::Paren(inner) => is_true(inner),
        _ => false,
    }
}

/// Whether a statement list always leaves its function: it ends in a
/// `return` or `throw`, in an `if`/`else` whose branches both do, or in a
/// loop that never ends and is never broken out of.
pub fn escapes(stmts: &[ast::Stmt]) -> bool {
    stmts.last().is_some_and(stmt_escapes)
}

fn stmt_escapes(s: &ast::Stmt) -> bool {
    use ast::StmtKind as K;
    match &s.kind {
        K::Return(_) | K::Throw(_) => true,
        K::Block(b) => escapes(&b.stmts),
        K::If {
            then,
            otherwise: Some(o),
            ..
        } => stmt_escapes(then) && stmt_escapes(o),
        K::While { cond, body } => is_true(cond) && !body.as_deref().is_some_and(breaks),
        K::Do { body, cond } => {
            (is_true(cond) && !body.stmts.iter().any(breaks)) || escapes(&body.stmts)
        }
        K::For { cond, body, .. } => {
            cond.as_ref().is_none_or(is_true) && !body.as_deref().is_some_and(breaks)
        }
        _ => false,
    }
}

/// Whether a `break` of this loop is in a statement (not one of a loop
/// nested in it).
fn breaks(s: &ast::Stmt) -> bool {
    use ast::StmtKind as K;
    match &s.kind {
        K::Break => true,
        K::Block(b) => b.stmts.iter().any(breaks),
        K::If {
            then, otherwise, ..
        } => breaks(then) || otherwise.as_deref().is_some_and(breaks),
        K::Try { body, catches } => {
            body.stmts.iter().any(breaks) || catches.iter().any(|c| c.body.stmts.iter().any(breaks))
        }
        _ => false,
    }
}
