//! Lambdas and method references: a lambda becomes a synthetic function
//! whose leading parameters are the variables it captures, and a reference
//! node that binds them.

use std::collections::HashMap;

use super::call::find_method;
use super::ty::{self, from_table};
use super::{compile_error, unsupported, Cx, Res, LOOP_LIMIT};
use crate::compile::SCRIPT_CLASS;
use crate::front::allowlist::Allowlist;
use crate::front::ast::{self, Arg, ExprKind, StmtKind};
use crate::ir::{self, Node, Reference};
use crate::types::DEF;

/// The `invoke_type` of a static method and of a constructor, as the
/// reference nodes write them; any other is a call on a receiver.
const INVOKE_STATIC: i32 = 6;
const INVOKE_VIRTUAL: i32 = 5;
const INVOKE_NEW: i32 = 8;

/// The variable names a lambda's body mentions, in first-use order.
#[derive(Default)]
struct Mentions(Vec<String>);

impl Mentions {
    fn name(&mut self, n: &str) {
        if !self.0.iter().any(|m| m == n) {
            self.0.push(n.to_string());
        }
    }

    fn args(&mut self, args: &[Arg]) {
        for a in args {
            match a {
                Arg::Expr(e) => self.expr(e),
                Arg::Lambda(l) => match &l.body {
                    ast::LambdaBody::Block(b) => self.stmts(&b.stmts),
                    ast::LambdaBody::Expr(e) => self.expr(e),
                },
                Arg::FuncRef(r) => {
                    if let Some(t) = &r.ty {
                        self.name(&t.name);
                    }
                }
            }
        }
    }

    fn opt(&mut self, e: &Option<ast::Expr>) {
        if let Some(e) = e {
            self.expr(e);
        }
    }

    fn decls(&mut self, ds: &[ast::Declaration]) {
        for d in ds {
            self.opt(&d.init);
        }
    }

    fn stmts(&mut self, ss: &[ast::Stmt]) {
        for s in ss {
            self.stmt(s);
        }
    }

    fn body(&mut self, s: &Option<Box<ast::Stmt>>) {
        if let Some(s) = s {
            self.stmt(s);
        }
    }

    fn stmt(&mut self, s: &ast::Stmt) {
        match &s.kind {
            StmtKind::If {
                cond,
                then,
                otherwise,
            } => {
                self.expr(cond);
                self.stmt(then);
                if let Some(o) = otherwise {
                    self.stmt(o);
                }
            }
            StmtKind::While { cond, body } => {
                self.expr(cond);
                self.body(body);
            }
            StmtKind::Do { body, cond } => {
                self.stmts(&body.stmts);
                self.expr(cond);
            }
            StmtKind::For {
                init,
                cond,
                after,
                body,
            } => {
                match init {
                    Some(ast::ForInit::Decl(ds)) => self.decls(ds),
                    Some(ast::ForInit::Expr(e)) => self.expr(e),
                    None => {}
                }
                self.opt(cond);
                self.opt(after);
                self.body(body);
            }
            StmtKind::Each { iterable, body, .. } => {
                self.expr(iterable);
                self.stmt(body);
            }
            StmtKind::Try { body, catches } => {
                self.stmts(&body.stmts);
                for c in catches {
                    self.stmts(&c.body.stmts);
                }
            }
            StmtKind::Decl(ds) => self.decls(ds),
            StmtKind::Return(e) => self.opt(e),
            StmtKind::Throw(e) | StmtKind::Expr(e) => self.expr(e),
            StmtKind::Block(b) => self.stmts(&b.stmts),
            StmtKind::Continue | StmtKind::Break => {}
        }
    }

    fn expr(&mut self, e: &ast::Expr) {
        match &e.kind {
            ExprKind::Var(n) => self.name(n),
            ExprKind::Num(..)
            | ExprKind::Bool(_)
            | ExprKind::Null
            | ExprKind::Str(_)
            | ExprKind::Regex(..) => {}
            ExprKind::List(es) | ExprKind::NewArray(_, es) | ExprKind::NewArrayInit(_, es) => {
                es.iter().for_each(|e| self.expr(e))
            }
            ExprKind::Map(kv) => kv.iter().for_each(|(k, v)| {
                self.expr(k);
                self.expr(v);
            }),
            ExprKind::CallLocal(_, args) | ExprKind::NewObject(_, args) => self.args(args),
            ExprKind::Call { recv, args, .. } => {
                self.expr(recv);
                self.args(args);
            }
            ExprKind::Field { recv, .. } => self.expr(recv),
            ExprKind::Index(a, b) | ExprKind::Binary(_, a, b) | ExprKind::Assign(_, a, b) => {
                self.expr(a);
                self.expr(b);
            }
            ExprKind::Unary(_, a)
            | ExprKind::Cast(_, a)
            | ExprKind::InstanceOf(a, _)
            | ExprKind::Paren(a) => self.expr(a),
            ExprKind::IncDec { target, .. } => self.expr(target),
            ExprKind::Cond(a, b, c) => {
                self.expr(a);
                self.expr(b);
                self.expr(c);
            }
        }
    }
}

impl Cx<'_> {
    /// An argument for a parameter of type `param` (an allowlist type name):
    /// an expression converted to it, or a lambda or method reference
    /// implementing it.
    pub(super) fn arg(&mut self, a: &Arg, param: &str) -> Res<Node> {
        let param = from_table(param);
        match a {
            Arg::Expr(e) => {
                let t = self.expr(e)?;
                Ok(self.coerce(e.pos, t, &param, false)?.node)
            }
            Arg::Lambda(l) => self.lambda(l, &param),
            Arg::FuncRef(r) => self.func_ref(r, &param),
        }
    }

    /// An argument of a call resolved at run time: a function value is
    /// bound to no interface.
    pub(super) fn def_arg(&mut self, a: &Arg) -> Res<Node> {
        match a {
            Arg::Expr(e) => {
                let t = self.expr(e)?;
                if t.ty == "void" {
                    return compile_error(e.pos, "cannot pass a void argument");
                }
                let t = if t.ty == "null" {
                    self.coerce(e.pos, t, "java.lang.Object", false)?
                } else {
                    t
                };
                Ok(t.node)
            }
            Arg::Lambda(l) => self.lambda(l, DEF),
            Arg::FuncRef(r) => self.func_ref(r, DEF),
        }
    }

    /// The interface method a function value of type `target` implements.
    fn interface_of(&self, pos: u32, target: &str) -> Res<(String, Vec<String>, String)> {
        match Allowlist::get().functional_method(target) {
            Some((name, sig)) => Ok((
                name.to_string(),
                sig.params.iter().map(|p| from_table(p)).collect(),
                from_table(&sig.ret),
            )),
            None => compile_error(
                pos,
                format!(
                    "Cannot convert a function value to [{}]: not a functional interface",
                    crate::types::canonical(target)
                ),
            ),
        }
    }

    fn visible(&self) -> HashMap<String, String> {
        let mut all = HashMap::new();
        for s in &self.scopes {
            for (k, v) in s {
                all.insert(k.clone(), v.clone());
            }
        }
        all
    }

    fn lambda(&mut self, l: &ast::Lambda, target: &str) -> Res<Node> {
        let (iface, iface_params, iface_ret) = if target == DEF {
            (String::new(), Vec::new(), DEF.to_string())
        } else {
            self.interface_of(l.pos, target)?
        };
        if target != DEF && iface_params.len() != l.params.len() {
            return compile_error(
                l.pos,
                format!(
                    "Incorrect number of parameters for [{iface}] in [{}]",
                    crate::types::canonical(target)
                ),
            );
        }
        let visible = self.visible();
        let mut params = Vec::new();
        for (i, (written, name)) in l.params.iter().enumerate() {
            if visible.contains_key(name) {
                return compile_error(l.pos, format!("variable [{name}] is already defined"));
            }
            let ty = match written {
                Some(t) => ty::resolve(t)?,
                None if target == DEF => DEF.to_string(),
                None => iface_params[i].clone(),
            };
            params.push((name.clone(), ty));
        }
        let mut mentions = Mentions::default();
        match &l.body {
            ast::LambdaBody::Block(b) => mentions.stmts(&b.stmts),
            ast::LambdaBody::Expr(e) => mentions.expr(e),
        }
        let captures: Vec<(String, String)> = mentions
            .0
            .iter()
            .filter(|n| !params.iter().any(|(p, _)| p == *n))
            .filter_map(|n| visible.get(n).map(|t| (n.clone(), t.clone())))
            .collect();
        let id = self.lambdas.next();
        let name = format!("lambda${id}");
        let mut inner = Cx::new(self.funcs, self.lambdas, iface_ret.clone());
        for (n, t) in captures.iter().chain(&params) {
            inner.declare(l.pos, n, t)?;
        }
        let stmts = match &l.body {
            ast::LambdaBody::Block(b) => b.stmts.clone(),
            ast::LambdaBody::Expr(e) => vec![ast::Stmt {
                pos: e.pos,
                kind: StmtKind::Expr((**e).clone()),
            }],
        };
        let body = inner.scoped(|cx| cx.body_of(&stmts, false))?;
        let all: Vec<(String, String)> = captures.iter().chain(&params).cloned().collect();
        self.lambdas.push(ir::Function {
            name: name.clone(),
            ret: iface_ret.clone(),
            param_types: all.iter().map(|(_, t)| t.clone()).collect(),
            param_names: all.iter().map(|(n, _)| n.clone()).collect(),
            is_static: true,
            synthetic: true,
            max_loop_counter: LOOP_LIMIT,
            body,
        });
        let capture_names: Vec<String> = captures.iter().map(|(n, _)| n.clone()).collect();
        if target == DEF {
            return Ok(Node::DefInterfaceReference {
                ty: DEF.into(),
                encoding: format!("Sthis.{name},{}", capture_names.len()),
                captures: capture_names,
            });
        }
        Ok(Node::TypedInterfaceReference {
            ty: target.into(),
            reference: Reference {
                interface_method: iface,
                delegate_class: SCRIPT_CLASS.into(),
                delegate_interface: false,
                delegate_augmented: false,
                invoke_type: INVOKE_STATIC,
                delegate_method: name,
                delegate_params: all.into_iter().map(|(_, t)| t).collect(),
                delegate_ret: iface_ret.clone(),
                interface_params: iface_params,
                interface_ret: iface_ret,
                factory_arity: capture_names.len() as i32,
            },
            captures: capture_names,
        })
    }

    fn func_ref(&mut self, r: &ast::FuncRef, target: &str) -> Res<Node> {
        let (iface, iface_params, iface_ret) = if target == DEF {
            (String::new(), Vec::new(), DEF.to_string())
        } else {
            self.interface_of(r.pos, target)?
        };
        let typed_ref =
            |class: &str, invoke: i32, method: &str, params: Vec<String>, ret: String| {
                Node::TypedInterfaceReference {
                    ty: target.into(),
                    reference: Reference {
                        interface_method: iface.clone(),
                        delegate_class: class.into(),
                        delegate_interface: false,
                        delegate_augmented: false,
                        invoke_type: invoke,
                        delegate_method: method.into(),
                        delegate_params: params,
                        delegate_ret: ret,
                        interface_params: iface_params.clone(),
                        interface_ret: iface_ret.clone(),
                        factory_arity: 0,
                    },
                    captures: Vec::new(),
                }
            };
        let def_ref = |encoding: String, captures: Vec<String>| Node::DefInterfaceReference {
            ty: DEF.into(),
            encoding,
            captures,
        };
        let Some(written) = &r.ty else {
            // `this::name`
            let name = r.name.clone().unwrap_or_default();
            let Some(sig) = self.funcs.get(&name).cloned() else {
                return compile_error(r.pos, format!("function [{name}] is not defined"));
            };
            if target == DEF {
                return Ok(def_ref(format!("Sthis.{name},0"), Vec::new()));
            }
            if sig.params.len() != iface_params.len() {
                return compile_error(
                    r.pos,
                    format!("function [{name}] takes a different number of arguments"),
                );
            }
            return Ok(typed_ref(
                SCRIPT_CLASS,
                INVOKE_STATIC,
                &name,
                sig.params,
                sig.ret,
            ));
        };
        // `x::name` on a variable.
        if written.dims == 0 {
            if let Some(var_ty) = self.lookup(&written.name).cloned() {
                let Some(name) = r.name.clone() else {
                    return unsupported("constructor reference on a variable");
                };
                if target == DEF || var_ty == DEF {
                    return Ok(def_ref(
                        format!("D{}.{name},1", written.name),
                        vec![written.name.clone()],
                    ));
                }
                return Ok(Node::TypedCaptureReference {
                    ty: target.into(),
                    method: name,
                    captures: vec![written.name.clone()],
                });
            }
        }
        let class = ty::resolve(written)?;
        match &r.name {
            None => {
                if target == DEF {
                    return Ok(def_ref(format!("S{class}.new,0"), Vec::new()));
                }
                let ret = class.clone();
                Ok(typed_ref(
                    &class,
                    INVOKE_NEW,
                    "new",
                    iface_params.clone(),
                    ret,
                ))
            }
            Some(name) => {
                if target == DEF {
                    return Ok(def_ref(format!("S{class}.{name},0"), Vec::new()));
                }
                let n = iface_params.len();
                if let Some(sig) = find_method(&class, name, n, true) {
                    return Ok(typed_ref(
                        &class,
                        INVOKE_STATIC,
                        name,
                        sig.params.iter().map(|p| from_table(p)).collect(),
                        from_table(&sig.ret),
                    ));
                }
                if n > 0 {
                    if let Some(sig) = find_method(&class, name, n - 1, false) {
                        let mut params = vec![class.clone()];
                        params.extend(sig.params.iter().map(|p| from_table(p)));
                        return Ok(typed_ref(
                            &class,
                            INVOKE_VIRTUAL,
                            name,
                            params,
                            from_table(&sig.ret),
                        ));
                    }
                }
                compile_error(
                    r.pos,
                    format!(
                        "function reference [{}::{name}] matching [{}] is not defined",
                        crate::types::canonical(&class),
                        crate::types::canonical(target)
                    ),
                )
            }
        }
    }
}
