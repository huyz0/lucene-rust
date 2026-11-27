//! Things a script reads and writes that are not locals: an element of an
//! array, list or map, and a field or a getter and setter pair. A place
//! pushes the words that name it, then a load or a store consumes them, so
//! a compound assignment duplicates them between the load and the store
//! and a value kept for the expression is duplicated below them.

use super::expr::Typed;
use super::ty::{array_of, boxed, component, from_table, is_primitive, STRING};
use super::{compile_error, unsupported, Cx, Res};
use crate::front::allowlist::Allowlist;
use crate::front::ast::{self, ExprKind};
use crate::ir::{self, Node};
use crate::types::{canonical, words, DEF};

/// A non-local place: what names it, how it is read, how it is written.
pub(super) struct Place {
    /// Pushes the words that name the place (a receiver, a receiver and an
    /// index).
    pub pre: Node,
    pub words: i32,
    /// The type a value read from it has, and one written to it is
    /// converted to.
    pub ty: String,
    /// Consumes the words and pushes the value.
    pub load: Node,
    /// The store of a child's value: consumes the words and the value.
    pub store: Box<dyn Fn(Node) -> Node>,
}

fn typed(node: Node, ty: &str) -> Typed {
    Typed {
        node,
        ty: ty.to_string(),
    }
}

pub(super) fn chain(ty: &str, left: Node, right: Node) -> Node {
    Node::BinaryImpl {
        ty: ty.into(),
        left: Box::new(left),
        right: Box::new(right),
    }
}

fn dup(ty: &str, size: i32, depth: i32, child: Node) -> Node {
    Node::Dup {
        ty: ty.into(),
        size,
        depth,
        child: Box::new(child),
    }
}

fn method(class: &str, name: &str, params: &[&str], ret: &str) -> ir::Method {
    ir::Method {
        target: class.into(),
        name: name.into(),
        declaring: class.into(),
        is_static: false,
        params: params.iter().map(|p| p.to_string()).collect(),
        ret: ret.into(),
    }
}

impl Cx<'_> {
    /// The place `recv[index]` names, by the receiver's type.
    pub(super) fn index_place(&mut self, recv: &ast::Expr, index: &ast::Expr) -> Res<Place> {
        let r = self.expr(recv)?;
        let al = Allowlist::get();
        if let Some(elem) = component(&r.ty) {
            let i = self.expr(index)?;
            let i = self.coerce(index.pos, i, "int", false)?;
            let pre = chain(
                "int",
                r.node,
                Node::FlipArrayIndex {
                    ty: "int".into(),
                    child: Box::new(i.node),
                },
            );
            let t = elem.clone();
            return Ok(Place {
                pre,
                words: 2,
                ty: elem.clone(),
                load: Node::LoadBrace { ty: elem.clone() },
                store: Box::new(move |child| Node::StoreBrace {
                    ty: "void".into(),
                    store_ty: t.clone(),
                    child: Box::new(child),
                }),
            });
        }
        if r.ty == DEF {
            let i = self.expr(index)?;
            if i.ty == "void" {
                return compile_error(index.pos, "cannot index by void");
            }
            let index_ty = i.ty.clone();
            let pre = chain(
                DEF,
                r.node,
                Node::FlipDefIndex {
                    ty: DEF.into(),
                    child: Box::new(i.node),
                },
            );
            let it = index_ty.clone();
            return Ok(Place {
                pre,
                words: 2,
                ty: DEF.into(),
                load: Node::LoadBraceDef {
                    ty: DEF.into(),
                    index_ty,
                },
                store: Box::new(move |child| Node::StoreBraceDef {
                    ty: "void".into(),
                    store_ty: DEF.into(),
                    index_ty: it.clone(),
                    child: Box::new(child),
                }),
            });
        }
        if al.is_assignable(&r.ty, "java.util.List") {
            let i = self.expr(index)?;
            let i = self.coerce(index.pos, i, "int", false)?;
            let pre = chain(
                "int",
                r.node,
                Node::FlipCollectionIndex {
                    ty: "int".into(),
                    child: Box::new(i.node),
                },
            );
            let class = r.ty.clone();
            return Ok(Place {
                pre,
                words: 2,
                ty: DEF.into(),
                load: Node::LoadListShortcut {
                    ty: DEF.into(),
                    getter: Some(method(&class, "get", &["int"], DEF)),
                },
                store: Box::new(move |child| Node::StoreListShortcut {
                    ty: "void".into(),
                    store_ty: DEF.into(),
                    setter: Some(method(&class, "set", &["int", DEF], DEF)),
                    child: Box::new(child),
                }),
            });
        }
        if al.is_assignable(&r.ty, "java.util.Map") {
            let k = self.expr(index)?;
            let k = self.coerce(index.pos, k, DEF, false)?;
            let pre = chain(DEF, r.node, k.node);
            let class = r.ty.clone();
            return Ok(Place {
                pre,
                words: 2,
                ty: DEF.into(),
                load: Node::LoadMapShortcut {
                    ty: DEF.into(),
                    getter: Some(method(&class, "get", &[DEF], DEF)),
                },
                store: Box::new(move |child| Node::StoreMapShortcut {
                    ty: "void".into(),
                    store_ty: DEF.into(),
                    setter: Some(method(&class, "put", &[DEF, DEF], DEF)),
                    child: Box::new(child),
                }),
            });
        }
        compile_error(
            recv.pos,
            format!("Cannot apply [] to type [{}].", canonical(&r.ty)),
        )
    }

    /// The place `recv.name` names: a `def` member, an instance field, or
    /// the getter and setter a name stands for, or a map's key.
    pub(super) fn field_place(
        &mut self,
        pos: u32,
        recv: &ast::Expr,
        name: &str,
        null_safe: bool,
    ) -> Res<Place> {
        if null_safe {
            return compile_error(pos, "invalid assignment to a null-safe access");
        }
        let r = self.expr(recv)?;
        let al = Allowlist::get();
        let n = name.to_string();
        if r.ty == DEF {
            let (n1, n2) = (n.clone(), n);
            return Ok(Place {
                pre: r.node,
                words: 1,
                ty: DEF.into(),
                load: Node::LoadDotDef {
                    ty: DEF.into(),
                    name: n1,
                },
                store: Box::new(move |child| Node::StoreDotDef {
                    ty: "void".into(),
                    store_ty: DEF.into(),
                    name: n2.clone(),
                    child: Box::new(child),
                }),
            });
        }
        if is_primitive(&r.ty) || r.ty == "void" || r.ty == "null" || r.ty.starts_with('[') {
            return unsupported("assignment to a member of a primitive or an array");
        }
        let class = r.ty.clone();
        if let Some(f) = al.field(&class, name).filter(|f| !f.is_static) {
            let ty = from_table(&f.ty);
            let field = ir::Field {
                class: class.clone(),
                name: n,
                is_static: false,
                ty: ty.clone(),
            };
            let (lf, sf, st) = (field.clone(), field, ty.clone());
            return Ok(Place {
                pre: r.node,
                words: 1,
                ty: ty.clone(),
                load: Node::LoadDot { ty, field: lf },
                store: Box::new(move |child| Node::StoreDot {
                    ty: "void".into(),
                    store_ty: st.clone(),
                    field: sf.clone(),
                    child: Box::new(child),
                }),
            });
        }
        let cap = {
            let mut c = name.chars();
            c.next()
                .map(|f| f.to_uppercase().chain(c).collect::<String>())
                .unwrap_or_default()
        };
        let find = |n: &str, arity: usize| {
            al.methods(&class, n)
                .into_iter()
                .find(|m| m.params.len() == arity && !m.is_static)
        };
        let setter = find(&format!("set{cap}"), 1);
        let getter = find(&format!("get{cap}"), 0).or_else(|| find(&format!("is{cap}"), 0));
        if let (Some(g), Some(s)) = (getter, setter) {
            let ty = from_table(&g.ret);
            let sty = from_table(&s.params[0]);
            let gm = method(&class, &format!("get{cap}"), &[], &ty);
            let sm = method(&class, &format!("set{cap}"), &[&sty], "void");
            let st = sty.clone();
            return Ok(Place {
                pre: r.node,
                words: 1,
                ty: sty,
                load: Node::LoadDotShortcut {
                    ty,
                    getter: Some(gm),
                },
                store: Box::new(move |child| Node::StoreDotShortcut {
                    ty: "void".into(),
                    store_ty: st.clone(),
                    setter: Some(sm.clone()),
                    child: Box::new(child),
                }),
            });
        }
        if al.is_assignable(&class, "java.util.Map") {
            let key = Node::Constant {
                ty: STRING.into(),
                value: ir::Constant::Str(name.into()),
            };
            return Ok(Place {
                pre: chain(DEF, r.node, key),
                words: 2,
                ty: DEF.into(),
                load: Node::LoadMapShortcut {
                    ty: DEF.into(),
                    getter: Some(method(&class, "get", &[DEF], DEF)),
                },
                store: Box::new(move |child| Node::StoreMapShortcut {
                    ty: "void".into(),
                    store_ty: DEF.into(),
                    setter: Some(method(&class, "put", &[DEF, DEF], DEF)),
                    child: Box::new(child),
                }),
            });
        }
        unsupported("assignment to this member")
    }

    /// The place an assignment target names, when it is not a local.
    pub(super) fn place_of(&mut self, target: &ast::Expr) -> Res<Option<Place>> {
        match &super::expr::unparen(target).kind {
            ExprKind::Index(recv, index) => self.index_place(recv, index).map(Some),
            ExprKind::Field {
                recv,
                name,
                null_safe,
            } => self
                .field_place(target.pos, recv, name, *null_safe)
                .map(Some),
            _ => Ok(None),
        }
    }

    /// `place = value`, `place op= value`, `++place` and the like. A
    /// postfix one used for its value yields the old value.
    pub(super) fn assign_place(
        &mut self,
        pos: u32,
        place: Place,
        update: PlaceUpdate,
        want_value: bool,
    ) -> Res<Typed> {
        let ty = place.ty.clone();
        let w = words(&ty) as i32;
        let (mut child, post) = match update {
            PlaceUpdate::Set(value, vpos) => {
                let v = self.coerce(vpos, value, &ty, false)?;
                (v.node, false)
            }
            PlaceUpdate::With(op, value) => {
                let load = self.loaded(&place);
                (self.compound(pos, op, load, value, &ty)?, false)
            }
            PlaceUpdate::Step { inc, prefix } => {
                let mut load = self.loaded(&place);
                let post = want_value && !prefix;
                if post {
                    load.node = dup(&ty, w, place.words, load.node);
                }
                let one = typed(
                    Node::Constant {
                        ty: "int".into(),
                        value: ir::Constant::Int(1),
                    },
                    "int",
                );
                let op = if inc {
                    ast::BinOp::Add
                } else {
                    ast::BinOp::Sub
                };
                (self.compound(pos, op, load, one, &ty)?, post)
            }
        };
        if want_value && !post {
            child = dup(&ty, w, place.words, child);
        }
        let result = if want_value { ty.as_str() } else { "void" };
        let store = (place.store)(child);
        Ok(typed(chain(result, place.pre, store), result))
    }

    /// The place's current value, its words duplicated first.
    fn loaded(&self, place: &Place) -> Typed {
        let kept = dup(
            "void",
            place.words,
            0,
            Node::Static {
                ty: "java.lang.Object".into(),
            },
        );
        typed(chain(&place.ty, kept, place.load.clone()), &place.ty)
    }

    fn compound(
        &mut self,
        pos: u32,
        op: ast::BinOp,
        load: Typed,
        value: Typed,
        ty: &str,
    ) -> Res<Node> {
        let r = self.binary(pos, op, load, value, super::expr::COMPOUND)?;
        Ok(self.coerce(pos, r, ty, true)?.node)
    }

    /// `recv[index]` read.
    pub(super) fn index_read(&mut self, recv: &ast::Expr, index: &ast::Expr) -> Res<Typed> {
        let p = self.index_place(recv, index)?;
        let ty = p.ty.clone();
        Ok(typed(chain(&ty, p.pre, p.load), &ty))
    }

    /// `new T[a][b]` and `new T[] {a, b}`.
    pub(super) fn new_array(&mut self, pos: u32, ty: &ast::Type, dims: &[ast::Expr]) -> Res<Typed> {
        let base = super::ty::resolve(&ast::Type {
            pos: ty.pos,
            name: ty.name.clone(),
            dims: 0,
        })?;
        if base == "void" || dims.is_empty() {
            return compile_error(pos, "invalid array creation");
        }
        let name = array_of(&base, dims.len());
        let mut args = Vec::new();
        for d in dims {
            let t = self.expr(d)?;
            args.push(self.coerce(d.pos, t, "int", false)?.node);
        }
        Ok(typed(
            Node::NewArray {
                ty: name.clone(),
                initialize: false,
                args,
            },
            &name,
        ))
    }

    pub(super) fn new_array_init(
        &mut self,
        pos: u32,
        ty: &ast::Type,
        values: &[ast::Expr],
    ) -> Res<Typed> {
        let name = super::ty::resolve(ty)?;
        let Some(elem) = component(&name) else {
            return compile_error(pos, "invalid array creation");
        };
        let mut args = Vec::new();
        for v in values {
            let t = self.expr(v)?;
            args.push(self.coerce(v.pos, t, &elem, false)?.node);
        }
        let _ = boxed;
        Ok(typed(
            Node::NewArray {
                ty: name.clone(),
                initialize: true,
                args,
            },
            &name,
        ))
    }
}

/// What is done to a place.
pub(super) enum PlaceUpdate {
    /// `= value`, and where it is.
    Set(Typed, u32),
    /// `op= value`.
    With(ast::BinOp, Typed),
    /// `++` or `--`.
    Step { inc: bool, prefix: bool },
}
