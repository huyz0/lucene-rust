//! Object construction and the list and map initialisers.

use super::expr::Typed;
use super::ty::{self, from_table};
use super::{compile_error, unsupported, Cx, Res};
use crate::front::allowlist::Allowlist;
use crate::front::ast;
use crate::ir::{self, Node};
use crate::types::{canonical, DEF};

const ARRAY_LIST: &str = "java.util.ArrayList";
const HASH_MAP: &str = "java.util.HashMap";

fn typed(node: Node, ty: &str) -> Typed {
    Typed {
        node,
        ty: ty.to_string(),
    }
}

impl Cx<'_> {
    /// `new T(args)`: the constructor Painless resolves by arity. With
    /// `read` false (a statement) the object is not left on the stack.
    pub(super) fn new_object(
        &mut self,
        pos: u32,
        written: &ast::Type,
        args: &[ast::Arg],
        read: bool,
    ) -> Res<Typed> {
        let class = ty::resolve(written)?;
        let ctors = Allowlist::get().ctors(&class);
        let Some(ctor) = ctors.iter().find(|c| c.params.len() == args.len()) else {
            return compile_error(
                pos,
                format!(
                    "Unknown new call on type [{}] with [{}] arguments.",
                    canonical(&class),
                    args.len()
                ),
            );
        };
        let mut lowered = Vec::new();
        for (arg, param) in args.iter().zip(&ctor.params) {
            let ast::Arg::Expr(e) = arg else {
                return unsupported("lambda or method reference argument");
            };
            let t = self.expr(e)?;
            lowered.push(self.coerce(e.pos, t, &from_table(param), false)?.node);
        }
        let ty = if read { class.as_str() } else { "void" };
        Ok(typed(
            Node::NewObject {
                ty: ty.into(),
                ctor: Some(ir::Constructor {
                    class: class.clone(),
                    params: ctor.params.iter().map(|p| from_table(p)).collect(),
                }),
                read,
                args: lowered,
            },
            ty,
        ))
    }

    /// `[a, b]`: an `ArrayList` of the values as `def`.
    pub(super) fn list_init(&mut self, values: &[ast::Expr]) -> Res<Typed> {
        let mut args = Vec::new();
        for e in values {
            args.push(self.as_def(e)?);
        }
        Ok(typed(
            Node::ListInit {
                ty: ARRAY_LIST.into(),
                ctor: Some(ir::Constructor {
                    class: ARRAY_LIST.into(),
                    params: Vec::new(),
                }),
                method: Some(ir::Method {
                    target: ARRAY_LIST.into(),
                    name: "add".into(),
                    declaring: ARRAY_LIST.into(),
                    is_static: false,
                    params: vec![DEF.into()],
                    ret: "boolean".into(),
                }),
                args,
            },
            ARRAY_LIST,
        ))
    }

    /// `[k : v]` and `[:]`: a `HashMap` of keys and values as `def`.
    pub(super) fn map_init(&mut self, pairs: &[(ast::Expr, ast::Expr)]) -> Res<Typed> {
        let (mut keys, mut values) = (Vec::new(), Vec::new());
        for (k, v) in pairs {
            keys.push(self.as_def(k)?);
            values.push(self.as_def(v)?);
        }
        Ok(typed(
            Node::MapInit {
                ty: HASH_MAP.into(),
                ctor: Some(ir::Constructor {
                    class: HASH_MAP.into(),
                    params: Vec::new(),
                }),
                method: Some(ir::Method {
                    target: HASH_MAP.into(),
                    name: "put".into(),
                    declaring: HASH_MAP.into(),
                    is_static: false,
                    params: vec![DEF.into(), DEF.into()],
                    ret: DEF.into(),
                }),
                keys,
                values,
            },
            HASH_MAP,
        ))
    }

    fn as_def(&mut self, e: &ast::Expr) -> Res<Node> {
        let t = self.expr(e)?;
        Ok(self.coerce(e.pos, t, DEF, false)?.node)
    }
}
