//! The site table of a lowered script: every member the script resolves by
//! name on a receiver's runtime class, with Painless's own answer for each
//! class of value the native runtime makes, as the Java front end writes it
//! (`PainlessIr.siteTable`). `check` compares the native runtime's answers
//! with these; here they come from the allowlist.

use crate::check::native_classes;
use crate::front::allowlist::Allowlist;
use crate::ir::{ClassTable, Method, Node, Script, Site, SiteKind};

const OBJECT: &str = "java.lang.Object";

/// Fills `script.classes` and `script.sites`.
pub(super) fn fill(script: &mut Script) {
    let mut found: Vec<(SiteKind, String, usize, String)> = Vec::new();
    for f in &script.functions {
        walk(&f.body, &mut found);
    }
    let classes: Vec<&str> = native_classes().collect();
    let al = Allowlist::get();
    script.classes = ClassTable::of(classes.iter().copied());
    script.sites = found
        .into_iter()
        .map(|(kind, name, arity, receiver)| {
            let admits: Vec<bool> = classes
                .iter()
                .map(|c| receiver == OBJECT || is_a(al, c, &receiver))
                .collect();
            let resolves = classes
                .iter()
                .zip(&admits)
                .map(|(c, &admitted)| admitted && resolves(al, c, kind, &name, arity))
                .collect();
            Site {
                kind,
                name,
                arity,
                receiver,
                admits,
                resolves,
            }
        })
        .collect();
}

/// The allowlisted classes a native class's members are looked up in: its
/// own, or the interface its Java class implements.
fn roots(native: &str) -> Vec<&str> {
    let root = match native {
        "java.util.ImmutableCollections$ListN" | "java.util.Arrays$ArrayList" => "java.util.List",
        "java.util.ImmutableCollections$MapN" => "java.util.Map",
        "java.util.ImmutableCollections$SetN" => "java.util.Set",
        "java.util.HashMap$Node" => "java.util.Map$Entry",
        n if n.starts_with('[') => return vec![OBJECT],
        // The doc values are lists in Java.
        n if n.starts_with("org.opensearch.index.fielddata.ScriptDocValues$") => {
            return vec![n, "java.util.List", OBJECT];
        }
        n => n,
    };
    vec![root, OBJECT]
}

/// Whether a value of the native class `native` is a `target`.
fn is_a(al: &Allowlist, native: &str, target: &str) -> bool {
    native == target
        || target == OBJECT
        || roots(native)
            .into_iter()
            .any(|r| al.is_assignable(r, target))
}

fn resolves(al: &Allowlist, native: &str, kind: SiteKind, name: &str, arity: usize) -> bool {
    let roots = roots(native);
    match kind {
        SiteKind::InstanceOf => is_a(al, native, name),
        SiteKind::Method => roots.iter().any(|r| {
            al.methods(r, name).iter().any(|m| {
                m.params.len() - m.inject.len() == arity && (m.augment.is_some() || !m.is_static)
            })
        }),
        SiteKind::Getter => roots.iter().any(|r| {
            al.field(r, name).is_some_and(|f| !f.is_static)
                || accessor(name, &["get", "is"]).any(|g| {
                    al.methods(r, &g)
                        .iter()
                        .any(|m| m.params.is_empty() && (m.augment.is_some() || !m.is_static))
                })
        }),
        SiteKind::Setter => roots.iter().any(|r| {
            al.field(r, name).is_some_and(|f| !f.is_static)
                || accessor(name, &["set"]).any(|g| {
                    al.methods(r, &g)
                        .iter()
                        .any(|m| m.params.len() == 1 && (m.augment.is_some() || !m.is_static))
                })
        }),
    }
}

/// The method names a property `name` is read or written by: `get` + `Name`.
fn accessor<'a>(name: &'a str, prefixes: &'a [&'a str]) -> impl Iterator<Item = String> + 'a {
    let mut chars = name.chars();
    let cap: String = chars
        .next()
        .map(|c| c.to_uppercase().chain(chars).collect())
        .unwrap_or_default();
    prefixes.iter().map(move |p| format!("{p}{cap}"))
}

type Found = Vec<(SiteKind, String, usize, String)>;

fn add(found: &mut Found, kind: SiteKind, name: &str, arity: usize, receiver: &str) {
    if !found
        .iter()
        .any(|(k, n, a, r)| *k == kind && n == name && *a == arity && r == receiver)
    {
        found.push((kind, name.to_string(), arity, receiver.to_string()));
    }
}

/// A method dispatched on its receiver: an instance or augmentation method.
/// Its receiver class is the one the allowlist declares it on.
fn method_site(found: &mut Found, m: &Option<Method>) {
    if let Some(m) = m {
        if !m.is_static || m.declaring.ends_with("Augmentation") {
            // The constants the compiler injects are not the script's arguments.
            let injected = Allowlist::get()
                .methods(&m.target, &m.name)
                .iter()
                .find(|s| s.params.len() == m.params.len())
                .map_or(0, |s| s.inject.len());
            let arity = m.params.len() - injected;
            let owner = declared_on(&m.target, &m.name, m.params.len());
            add(found, SiteKind::Method, &m.name, arity, &owner);
        }
    }
}

/// The class of `class`'s hierarchy the allowlist declares `name/arity` on.
fn declared_on(class: &str, name: &str, arity: usize) -> String {
    let al = Allowlist::get();
    let Some(c) = al.class(class) else {
        return class.to_string();
    };
    std::iter::once(class)
        .chain(c.parents.iter().map(String::as_str))
        .find(|owner| {
            al.class(owner)
                .and_then(|c| c.methods.get(name))
                .is_some_and(|ms| ms.iter().any(|m| m.params.len() == arity))
        })
        .unwrap_or(class)
        .to_string()
}

fn walk(n: &Node, found: &mut Found) {
    use Node::*;
    let opt = |c: &Option<Box<Node>>, found: &mut Found| {
        if let Some(c) = c {
            walk(c, found);
        }
    };
    match n {
        At(_, _, c) | StatementExpression(c) | Throw(c) => walk(c, found),
        Block(v) | DeclarationBlock(v) | StringConcat { args: v, .. } => {
            v.iter().for_each(|c| walk(c, found))
        }
        Declaration { init, .. } => opt(init, found),
        Return(c) => opt(c, found),
        If { cond, then } => {
            walk(cond, found);
            walk(then, found);
        }
        IfElse {
            cond,
            then,
            otherwise,
        } => {
            walk(cond, found);
            walk(then, found);
            walk(otherwise, found);
        }
        While { cond, body, .. } | DoWhile { cond, body, .. } => {
            opt(cond, found);
            opt(body, found);
        }
        For {
            init,
            cond,
            after,
            body,
            ..
        } => {
            opt(init, found);
            opt(cond, found);
            opt(after, found);
            opt(body, found);
        }
        ForEachArray { iterable, body, .. } | ForEachIterable { iterable, body, .. } => {
            walk(iterable, found);
            opt(body, found);
        }
        Try { body, catches } => {
            walk(body, found);
            catches.iter().for_each(|c| walk(c, found));
        }
        Catch { body, .. } => opt(body, found),
        BinaryImpl { left, right, .. }
        | BinaryMath { left, right, .. }
        | Boolean { left, right, .. }
        | Comparison { left, right, .. }
        | Elvis { left, right, .. } => {
            walk(left, found);
            walk(right, found);
        }
        Conditional {
            cond, left, right, ..
        } => {
            walk(cond, found);
            walk(left, found);
            walk(right, found);
        }
        Cast { cast, child, .. } => {
            if let Some(c) = cast {
                if !is_primitive(&c.to)
                    && !is_primitive(&c.from)
                    && !c.to.starts_with('[')
                    && c.unbox_from.is_none()
                    && c.unbox_to.is_none()
                    && c.box_from.is_none()
                    && c.box_to.is_none()
                    && c.to != OBJECT
                    && !is_def(&c.to)
                {
                    add(found, SiteKind::InstanceOf, &c.to, 0, OBJECT);
                }
            }
            walk(child, found);
        }
        Instanceof {
            instance_ty, child, ..
        } => {
            if !is_primitive(instance_ty) && instance_ty != OBJECT && !is_def(instance_ty) {
                add(found, SiteKind::InstanceOf, instance_ty, 0, OBJECT);
            }
            walk(child, found);
        }
        StoreVariable { child, .. }
        | UnaryMath { child, .. }
        | Dup { child, .. }
        | FlipArrayIndex { child, .. }
        | FlipCollectionIndex { child, .. }
        | FlipDefIndex { child, .. }
        | StoreBrace { child, .. }
        | StoreBraceDef { child, .. }
        | StoreFieldMember { child, .. }
        | NullSafeSub { child, .. } => walk(child, found),
        InvokeCall { method, args, .. } => {
            method_site(found, method);
            args.iter().for_each(|c| walk(c, found));
        }
        InvokeCallDef { name, args, .. } => {
            add(found, SiteKind::Method, name, args.len(), OBJECT);
            args.iter().for_each(|c| walk(c, found));
        }
        InvokeCallMember { args, .. }
        | ListInit { args, .. }
        | NewArray { args, .. }
        | NewObject { args, .. } => args.iter().for_each(|c| walk(c, found)),
        MapInit { keys, values, .. } => {
            keys.iter().for_each(|c| walk(c, found));
            values.iter().for_each(|c| walk(c, found));
        }
        LoadDotDef { name, .. } => add(found, SiteKind::Getter, name, 0, OBJECT),
        LoadDot { field, .. } => {
            if !field.is_static {
                add(found, SiteKind::Getter, &field.name, 0, &field.class);
            }
        }
        LoadDotShortcut { getter, .. }
        | LoadListShortcut { getter, .. }
        | LoadMapShortcut { getter, .. } => method_site(found, getter),
        StoreDotDef { name, child, .. } => {
            add(found, SiteKind::Setter, name, 1, OBJECT);
            walk(child, found);
        }
        StoreDot { field, child, .. } => {
            if !field.is_static {
                add(found, SiteKind::Setter, &field.name, 1, &field.class);
            }
            walk(child, found);
        }
        StoreDotShortcut { setter, child, .. }
        | StoreListShortcut { setter, child, .. }
        | StoreMapShortcut { setter, child, .. } => {
            method_site(found, setter);
            walk(child, found);
        }
        Break
        | Continue
        | Constant { .. }
        | Null { .. }
        | LoadVariable { .. }
        | LoadBrace { .. }
        | LoadBraceDef { .. }
        | LoadDotArrayLength { .. }
        | Static { .. }
        | LoadFieldMember { .. }
        | TypedInterfaceReference { .. }
        | DefInterfaceReference { .. }
        | TypedCaptureReference { .. } => {}
    }
}

fn is_primitive(t: &str) -> bool {
    matches!(
        t,
        "boolean" | "byte" | "short" | "char" | "int" | "long" | "float" | "double"
    )
}

fn is_def(t: &str) -> bool {
    t == "def" || t == "org.opensearch.painless.lookup.def"
}
