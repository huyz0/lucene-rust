//! Whether the native runtime runs a script exactly as Painless would,
//! decided before it runs anything.
//!
//! A script either runs natively from its first document to its last, or
//! not at all: nothing of it executes natively and then again in Java. The
//! decision is static, made on the compiled and specialised program:
//!
//! - **Members by name.** Painless resolves a call, a `def` field read or
//!   write and a cast on the receiver's runtime class. The plugin writes
//!   every such site with Painless's own answer for each class of value the
//!   native runtime makes ([`crate::ir::Script::sites`]); the native runtime's own
//!   answer must be the same for every class the site's static type admits.
//!   Where both say no, both raise the same `IllegalArgumentException`.
//! - **`doc`.** The document lookup is read only by a constant field name the
//!   query mapped (`doc['f']`, `doc.get('f')`, `doc.containsKey('f')`), and
//!   every field's type has a native reading.
//! - **`params`.** Every read of `params` is folded to its value (a script
//!   that changes or hands on `params` is refused), and no folded value is
//!   one the native runtime cannot serve (`_source`).
//! - **The script class's own methods** are ones the context serves.
//!
//! Whatever a script that passes then throws, natively, is the exception
//! Painless throws: it is the script's error, never a reason to run it again.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::api::doc::{DocKind, DocLookup, DocValues};
use crate::api::dynamic;
use crate::ir::{ClassTable, Script, Site, SiteKind};
use crate::jmap::{JMap, Order};
use crate::value::{ListKind, MapKind, SetKind, Value};
use crate::vm::{Block, CastOp, Env, Op, Program, Reg, Tag, Throw, Vm};

/// What the context serves the script.
pub struct Served<'a> {
    /// The fields `doc` reads, in [`crate::vm::Env::doc_at`]'s numbering,
    /// with how each reads; `None` for a mapped field without a native
    /// reading.
    pub fields: &'a [(&'a str, Option<DocKind>)],
    /// Fields named in the script that the index does not map: a read
    /// raises Painless's `IllegalArgumentException`.
    pub unmapped: &'a [&'a str],
    /// The script class's own methods the context implements.
    pub internals: &'a [&'a str],
}

/// Why the native runtime does not run a script: it runs in Java.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refused(pub String);

impl std::fmt::Display for Refused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Checks `prog` (compiled from `script` and specialised to the query)
/// against what the context serves.
pub fn check(script: &Script, prog: &Program, served: &Served<'_>) -> Result<(), Refused> {
    check_fields(served)?;
    check_sites(script)?;
    check_flows(prog, served)
}

fn check_fields(served: &Served<'_>) -> Result<(), Refused> {
    match served.fields.iter().find(|(_, k)| k.is_none()) {
        Some((name, _)) if !served.unmapped.contains(name) => {
            Err(Refused(format!("doc['{name}'] has no native reading")))
        }
        _ => Ok(()),
    }
}

/// The classes of the values the native runtime makes that a script can
/// reach a member of, each with a value of it.
/// A class's name and a sample value of it.
type Sample = (&'static str, fn() -> Value);

const NATIVE: &[Sample] = &[
    ("java.lang.Boolean", || Value::Bool(true)),
    ("java.lang.Byte", || Value::Byte(1)),
    ("java.lang.Short", || Value::Short(1)),
    ("java.lang.Character", || Value::Char(u16::from(b'a'))),
    ("java.lang.Integer", || Value::Int(1)),
    ("java.lang.Long", || Value::Long(1)),
    ("java.lang.Float", || Value::Float(1.0)),
    ("java.lang.Double", || Value::Double(1.0)),
    ("java.lang.String", || Value::str("a")),
    ("java.lang.StringBuilder", || {
        Value::object(crate::api::string::JStringBuilder {
            buf: RefCell::new("a".into()),
        })
    }),
    ("java.util.ArrayList", || list(ListKind::ArrayList)),
    ("java.util.LinkedList", || list(ListKind::LinkedList)),
    ("java.util.ImmutableCollections$ListN", || {
        list(ListKind::Immutable)
    }),
    ("java.util.Arrays$ArrayList", || list(ListKind::FixedSize)),
    ("java.util.HashMap", || map(MapKind::HashMap)),
    ("java.util.LinkedHashMap", || map(MapKind::LinkedHashMap)),
    ("java.util.TreeMap", || map(MapKind::TreeMap)),
    ("java.util.ImmutableCollections$MapN", || {
        map(MapKind::Immutable)
    }),
    ("java.util.HashSet", || set(SetKind::HashSet)),
    ("java.util.LinkedHashSet", || set(SetKind::LinkedHashSet)),
    ("java.util.TreeSet", || set(SetKind::TreeSet)),
    ("java.util.ImmutableCollections$SetN", || {
        set(SetKind::Immutable)
    }),
    ("java.util.HashMap$Node", || {
        Value::object(crate::api::map::MapEntry {
            key: Value::str("k"),
            value: RefCell::new(Value::Int(1)),
            map: None,
        })
    }),
    ("java.util.Iterator", || {
        crate::api::iter::JIter::value(vec![Value::Int(1)])
    }),
    ("java.util.Comparator", || {
        Value::object(crate::api::functional::Composed::NaturalOrder)
    }),
    ("java.util.function.Function", || {
        Value::object(crate::api::functional::Composed::Identity)
    }),
    ("java.util.function.Predicate", || {
        Value::object(crate::api::functional::Composed::Negate(Value::object(
            crate::api::functional::Composed::Identity,
        )))
    }),
    ("java.util.regex.Pattern", pattern),
    ("java.util.regex.Matcher", || {
        let mut env = NoEnv;
        let mut vm = probe_vm(&mut env);
        dynamic::call(&mut vm, &pattern(), "matcher", &[Value::str("a")]).expect("Pattern.matcher")
    }),
    ("java.lang.Exception", || {
        Throw::new("java.lang.Exception", "e").0
    }),
    ("[I", || Value::array("int", vec![Value::Int(1)])),
    ("[Ljava.lang.Object;", || {
        Value::array("java.lang.Object", vec![Value::Int(1)])
    }),
    (
        "org.opensearch.index.fielddata.ScriptDocValues$Longs",
        || DocValues::value(DocKind::Longs, vec![Value::Long(1)]),
    ),
    (
        "org.opensearch.index.fielddata.ScriptDocValues$Doubles",
        || DocValues::value(DocKind::Doubles, vec![Value::Double(1.0)]),
    ),
    (
        "org.opensearch.index.fielddata.ScriptDocValues$Booleans",
        || DocValues::value(DocKind::Booleans, vec![Value::Bool(true)]),
    ),
    (
        "org.opensearch.index.fielddata.ScriptDocValues$Strings",
        || DocValues::value(DocKind::Strings, vec![Value::str("a")]),
    ),
];

fn list(kind: ListKind) -> Value {
    Value::list(kind, vec![Value::Int(1)])
}

fn map(kind: MapKind) -> Value {
    let mut m = JMap::new(match kind {
        MapKind::TreeMap => Order::Sorted,
        MapKind::LinkedHashMap => Order::Insertion,
        _ => Order::Hash,
    });
    m.insert(Value::str("k"), Value::Int(1));
    Value::map(kind, m)
}

fn set(kind: SetKind) -> Value {
    let mut m = JMap::new(match kind {
        SetKind::TreeSet => Order::Sorted,
        SetKind::LinkedHashSet => Order::Insertion,
        _ => Order::Hash,
    });
    m.insert(Value::Int(1), Value::Null);
    Value::set(kind, m)
}

fn pattern() -> Value {
    crate::compile::constant(&crate::ir::Constant::Pattern("a".into(), 0)).expect("a pattern")
}

/// The classes [`check`] needs Painless's answers for.
pub fn native_classes() -> impl Iterator<Item = &'static str> {
    NATIVE.iter().map(|(c, _)| *c)
}

struct NoEnv;

impl Env for NoEnv {
    fn internal(&mut self, name: &str, _args: &[Value]) -> Result<Value, Throw> {
        Err(crate::compile::missing(format!("probe {name}")))
    }
}

fn probe_vm(env: &mut NoEnv) -> Vm<'_> {
    Vm::new(Rc::new(Program::default()), env)
}

thread_local! {
    /// The native runtime's answers, by class, site kind, name and arity.
    static ANSWERS: RefCell<HashMap<(usize, SiteKind, String, usize), bool>> =
        RefCell::new(HashMap::new());
}

/// Whether the native runtime resolves `site` on class `NATIVE[class]`.
fn resolves(class: usize, site: &Site) -> bool {
    let key = (class, site.kind, site.name.clone(), site.arity);
    if let Some(a) = ANSWERS.with(|m| m.borrow().get(&key).copied()) {
        return a;
    }
    let mut env = NoEnv;
    let mut vm = probe_vm(&mut env);
    let recv = (NATIVE[class].1)();
    let found = |r: Option<Result<Value, Throw>>| match r {
        None => false,
        // A member the runtime names but lacks.
        Some(Err(t)) => t.class() != crate::api::UNSUPPORTED,
        Some(Ok(_)) => true,
    };
    let a = match site.kind {
        SiteKind::Method => found(dynamic::try_call(
            &mut vm,
            &recv,
            &site.name,
            &vec![Value::Null; site.arity],
        )),
        SiteKind::Getter => found(dynamic::try_getter(&mut vm, &recv, &site.name)),
        SiteKind::Setter => found(dynamic::try_setter(
            &mut vm,
            &recv,
            &site.name,
            &Value::Null,
        )),
        SiteKind::InstanceOf => crate::types::instance_of(&recv, &site.name),
    };
    ANSWERS.with(|m| m.borrow_mut().insert(key, a));
    a
}

fn check_sites(script: &Script) -> Result<(), Refused> {
    if script.classes != ClassTable::of(native_classes()) {
        return Err(Refused(
            "the site table is for other classes than the native runtime's".into(),
        ));
    }
    for site in &script.sites {
        for (class, (name, _)) in NATIVE.iter().enumerate() {
            if !site.admits[class] {
                continue;
            }
            let java = site.resolves[class];
            if resolves(class, site) != java {
                return Err(Refused(format!(
                    "{} on {}: Painless {}, the native runtime {}",
                    describe(site),
                    name,
                    if java { "has it" } else { "does not" },
                    if java { "does not" } else { "has it" },
                )));
            }
        }
    }
    Ok(())
}

fn describe(site: &Site) -> String {
    match site.kind {
        SiteKind::Method => format!("method {}/{}", site.name, site.arity),
        SiteKind::Getter => format!("field read .{}", site.name),
        SiteKind::Setter => format!("field write .{}", site.name),
        SiteKind::InstanceOf => format!("cast to {}", site.name),
    }
}

/// A value whose every use the check follows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Watched {
    /// `doc`.
    Doc,
    /// `params` itself, unfolded.
    Params,
    /// A value no use may reach (`params._source`).
    Foreign,
    /// A collection inside `params`, or a value read from one: Painless's
    /// script shares it with every segment's script (`ScoreScript` copies
    /// `params` shallowly), so nothing may change it.
    Shared,
    /// What a view call made (`keySet`, `values`, `entrySet`, `subList`
    /// ...), or an iterator over it: Java's is a live view of its own
    /// class, the runtime's a copy, so it may only be read.
    View(View),
    /// An entry of an `entrySet` view: Java's writes through to its map.
    Entry,
}

/// Which view a [`Watched::View`] is: the reference casts Java's passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum View {
    /// A `Set` (`keySet`, a sorted set's sub-set).
    Set,
    /// `entrySet`: a `Set` of live entries.
    Entries,
    /// A `Collection` only (`values`).
    Collection,
    /// A `List` (`subList`).
    List,
    /// A `Map` (a sorted map's sub-map).
    Map,
}

/// The calls whose result Java keeps as a live view of the receiver.
fn view_of(name: &str) -> Option<View> {
    Some(match name {
        "keySet" | "navigableKeySet" | "descendingKeySet" | "headSet" | "tailSet" | "subSet"
        | "descendingSet" => View::Set,
        "entrySet" => View::Entries,
        "values" => View::Collection,
        "subList" => View::List,
        "headMap" | "tailMap" | "subMap" | "descendingMap" => View::Map,
        _ => return None,
    })
}

/// Whether Java's view passes a reference cast to `class`.
fn view_casts(view: View, class: &str) -> bool {
    match class {
        "java.lang.Object" | "org.opensearch.painless.lookup.def" => true,
        "java.lang.Iterable" | "java.util.Collection" => view != View::Map,
        "java.util.Set" => matches!(view, View::Set | View::Entries),
        "java.util.List" => view == View::List,
        "java.util.Map" => view == View::Map,
        _ => false,
    }
}

/// The calls that read a view (or its iterator) without changing it or its
/// backing collection and without exposing the view's own class or
/// identity.
const VIEW_READS: &[&str] = &[
    "get",
    "getOrDefault",
    "containsKey",
    "containsValue",
    "contains",
    "containsAll",
    "size",
    "isEmpty",
    "indexOf",
    "lastIndexOf",
    "toString",
    "iterator",
    "hasNext",
    "next",
    "first",
    "last",
    "firstKey",
    "lastKey",
];

/// The calls that keep an argument inside the receiver: a view handed to
/// one escapes the check's flows.
const STORES: &[&str] = &[
    "add",
    "addFirst",
    "addLast",
    "offer",
    "push",
    "put",
    "putIfAbsent",
    "set",
    "replace",
    "merge",
    "compute",
    "computeIfAbsent",
    "computeIfPresent",
    "setValue",
];

/// The calls that read a collection without changing it.
const READS: &[&str] = &[
    "get",
    "getOrDefault",
    "containsKey",
    "containsValue",
    "contains",
    "containsAll",
    "size",
    "isEmpty",
    "indexOf",
    "lastIndexOf",
    "equals",
    "hashCode",
    "toString",
    "iterator",
    "hasNext",
    "next",
    "getLength",
];

fn watched_const(v: &Value) -> Option<Watched> {
    match v {
        Value::Object(o) if o.as_any().is::<DocLookup>() => Some(Watched::Doc),
        Value::Object(o) if !native_classes().any(|c| c == o.class_name()) => {
            if o.as_any().is::<crate::lambda::Lambda>() {
                None
            } else {
                Some(Watched::Foreign)
            }
        }
        Value::Map(m)
            if m.borrow()
                .map
                .iter()
                .any(|(_, v)| watched_const(v).is_some()) =>
        {
            Some(Watched::Params)
        }
        // Script literals compile to no collection constant: one is a
        // folded `params` value.
        Value::Map(_) | Value::List(_) | Value::Set(_) | Value::Array(_) => Some(Watched::Shared),
        _ => None,
    }
}

/// Every register's flows, over the whole program: which watched value it
/// may hold, and the constant it holds when it has a single definition.
struct Flows {
    /// Per function, per register.
    watched: Vec<Vec<Option<Watched>>>,
    consts: Vec<Vec<Option<Value>>>,
    /// Per function: what its return may be.
    returns: Vec<Option<Watched>>,
}

fn merge(a: &mut Option<Watched>, b: Option<Watched>) -> bool {
    match (*a, b) {
        (_, None) => false,
        (None, Some(_)) => {
            *a = b;
            true
        }
        (Some(x), Some(y)) if x == y => false,
        // Two kinds through one register: nothing reads it.
        (Some(Watched::Foreign), _) => false,
        (Some(_), Some(_)) => {
            *a = Some(Watched::Foreign);
            true
        }
    }
}

fn check_flows(prog: &Program, served: &Served<'_>) -> Result<(), Refused> {
    let mut fl = Flows {
        watched: prog.funcs.iter().map(|f| vec![None; f.regs]).collect(),
        consts: prog.funcs.iter().map(|f| vec![None; f.regs]).collect(),
        returns: vec![None; prog.funcs.len()],
    };
    for (i, f) in prog.funcs.iter().enumerate() {
        let mut defs = vec![0u32; f.regs];
        count_defs(&f.body, &mut defs);
        single_consts(&f.body, &defs, &mut fl.consts[i]);
    }
    // To a fixed point: moves, calls of the script's functions and returns.
    loop {
        let mut changed = false;
        for (i, f) in prog.funcs.iter().enumerate() {
            changed |= propagate(prog, i, &f.body, served, &mut fl)?;
        }
        if !changed {
            break;
        }
    }
    for (i, f) in prog.funcs.iter().enumerate() {
        uses(i, &f.body, served, &fl)?;
    }
    Ok(())
}

fn count_defs(ops: &[Op], defs: &mut [u32]) {
    for_each_op(ops, &mut |op| {
        if let Some(d) = dst(op) {
            defs[d as usize] += 1;
        }
        if let Op::Try { catches, .. } = op {
            for c in catches.iter() {
                defs[c.slot as usize] += 1;
            }
        }
    });
}

fn single_consts(ops: &[Op], defs: &[u32], out: &mut [Option<Value>]) {
    for_each_op(ops, &mut |op| {
        if let Op::Const(d, v) = op {
            if defs[*d as usize] == 1 {
                out[*d as usize] = Some(v.clone());
            }
        }
    });
}

/// Every op, nested blocks included, in order.
fn for_each_op(ops: &[Op], f: &mut dyn FnMut(&Op)) {
    for op in ops {
        f(op);
        for b in blocks(op) {
            for_each_op(b, f);
        }
    }
}

fn blocks(op: &Op) -> Vec<&Block> {
    match op {
        Op::If {
            then, otherwise, ..
        }
        | Op::IfNull {
            then, otherwise, ..
        } => vec![then, otherwise],
        Op::Loop { head, body, after } => vec![head, body, after],
        Op::Try { body, catches } => {
            let mut v = vec![body];
            v.extend(catches.iter().map(|c| &c.body));
            v
        }
        _ => Vec::new(),
    }
}

fn dst(op: &Op) -> Option<Reg> {
    Some(match op {
        Op::Const(d, _) | Op::Move(d, _) => *d,
        Op::Api { dst, .. }
        | Op::Native { dst, .. }
        | Op::Doc { dst, .. }
        | Op::Local { dst, .. }
        | Op::Arith { dst, .. }
        | Op::DefArith { dst, .. }
        | Op::Cmp { dst, .. }
        | Op::Unary { dst, .. }
        | Op::Cast { dst, .. }
        | Op::InstanceOf { dst, .. }
        | Op::Concat { dst, .. } => *dst,
        _ => return None,
    })
}

/// A cast that hands its value on unchanged: to a reference type.
fn passes_on(cast: &CastOp) -> bool {
    !matches!(cast, CastOp::DefTo { .. } | CastOp::Prim { .. })
}

fn propagate(
    prog: &Program,
    func: usize,
    ops: &[Op],
    served: &Served<'_>,
    fl: &mut Flows,
) -> Result<bool, Refused> {
    let mut changed = false;
    let mut err = None;
    for_each_op(ops, &mut |op| {
        let w = &fl.watched[func];
        let (d, from) = match op {
            Op::Const(d, v) => (*d, watched_const(v)),
            Op::Move(d, s) => (*d, w[*s as usize]),
            Op::Cast { dst, src, cast } if passes_on(cast) => (*dst, w[*src as usize]),
            Op::Native {
                dst,
                tag: Tag::Internal(n),
                args,
                ..
            } if args.is_empty() => match &**n {
                "getDoc" => (*dst, Some(Watched::Doc)),
                "getParams" => (*dst, Some(Watched::Params)),
                n if served.internals.contains(&n) => return,
                n => {
                    err.get_or_insert(Refused(format!("the script class's method {n}")));
                    return;
                }
            },
            Op::Local { dst, func: f, args } => {
                let f = *f as usize;
                for (p, &a) in args.iter().enumerate() {
                    let from = fl.watched[func][a as usize];
                    if p < fl.watched[f].len() {
                        changed |= merge(&mut fl.watched[f][p], from);
                    }
                }
                (*dst, fl.returns[f])
            }
            Op::Return(Some(r)) => {
                let from = w[*r as usize];
                changed |= merge(&mut fl.returns[func], from);
                return;
            }
            // A read of a shared collection: what it reads is shared too.
            Op::Native {
                dst,
                tag: Tag::Call(_) | Tag::LoadDot(_),
                args,
                ..
            }
            | Op::Api {
                dst,
                tag: Tag::BraceLoad,
                args,
                ..
            } if args
                .first()
                .is_some_and(|&a| w[a as usize] == Some(Watched::Shared)) =>
            {
                (*dst, Some(Watched::Shared))
            }
            // A for-each loop over a shared collection or a view.
            Op::Api {
                dst,
                tag: Tag::ForEach(n),
                args,
                ..
            } => match (*n, args.first().and_then(|&a| w[a as usize])) {
                (_, Some(Watched::Shared)) => (*dst, Some(Watched::Shared)),
                ("iterator", Some(Watched::View(v))) => (*dst, Some(Watched::View(v))),
                ("next", Some(Watched::View(View::Entries))) => (*dst, Some(Watched::Entry)),
                _ => return,
            },
            // A view, an iterator over one, and an entry set's entries.
            Op::Native {
                dst,
                tag: Tag::Call(n),
                args,
                ..
            } => match (&**n, args.first().and_then(|&a| w[a as usize])) {
                ("iterator", Some(Watched::View(v))) => (*dst, Some(Watched::View(v))),
                ("next", Some(Watched::View(View::Entries))) => (*dst, Some(Watched::Entry)),
                (n, _) => match view_of(n) {
                    Some(v) => (*dst, Some(Watched::View(v))),
                    None => return,
                },
            },
            _ => return,
        };
        changed |= merge(&mut fl.watched[func][d as usize], from);
    });
    let _ = prog;
    match err {
        Some(e) => Err(e),
        None => Ok(changed),
    }
}

/// Every read of a watched register must be one the native runtime serves
/// as Painless does.
fn uses(func: usize, ops: &[Op], served: &Served<'_>, fl: &Flows) -> Result<(), Refused> {
    let w = &fl.watched[func];
    let consts = &fl.consts[func];
    let mut err: Option<Refused> = None;
    let mut refuse = |what: String| {
        err.get_or_insert(Refused(what));
    };
    let known_field = |r: Reg| match &consts[r as usize] {
        Some(Value::Str(s)) => {
            served.fields.iter().any(|(f, _)| **f == **s) || served.unmapped.contains(&&**s)
        }
        _ => false,
    };
    for_each_op(ops, &mut |op| {
        let watched_args: Vec<(usize, Watched)> = reads(op)
            .into_iter()
            .enumerate()
            .filter_map(|(i, r)| w[r as usize].map(|k| (i, k)))
            .collect();
        for (i, kind) in watched_args {
            let ok = match (kind, op) {
                // Hand-offs the flows follow.
                (_, Op::Move(..) | Op::Local { .. } | Op::Return(Some(_))) => true,
                (
                    Watched::View(v),
                    Op::Cast {
                        cast: CastOp::Check(class),
                        ..
                    },
                ) => view_casts(v, class),
                (
                    Watched::Entry,
                    Op::Cast {
                        cast: CastOp::Check(class),
                        ..
                    },
                ) => matches!(
                    &**class,
                    "java.util.Map$Entry"
                        | "java.lang.Object"
                        | "org.opensearch.painless.lookup.def"
                ),
                (Watched::View(_) | Watched::Entry, Op::Cast { .. } | Op::InstanceOf { .. }) => {
                    false
                }
                (_, Op::Cast { cast, .. }) if passes_on(cast) => true,
                // Identity tests read no member.
                (_, Op::InstanceOf { .. } | Op::IfNull { .. }) => true,
                (_, Op::Cmp { def: false, .. }) => true,
                (
                    Watched::Doc,
                    Op::Native {
                        tag: Tag::Call(n),
                        args,
                        ..
                    },
                ) => {
                    i == 0
                        && matches!((&**n, args.len()), ("get" | "containsKey", 2))
                        && known_field(args[1])
                }
                (
                    Watched::Doc,
                    Op::Api {
                        tag: Tag::BraceLoad,
                        args,
                        ..
                    },
                ) => i == 0 && args.len() == 2 && known_field(args[1]),
                (
                    Watched::Doc,
                    Op::Native {
                        tag: Tag::LoadDot(n),
                        ..
                    },
                ) => {
                    i == 0
                        && (served.fields.iter().any(|(f, _)| **f == **n)
                            || served.unmapped.contains(&&**n))
                }
                // A shared value may be read and computed with, never changed or
                // handed to a call that may change it.
                (
                    Watched::Shared,
                    Op::Native {
                        tag: Tag::Call(n), ..
                    },
                ) => READS.contains(&&**n),
                (
                    Watched::Shared,
                    Op::Native {
                        tag: Tag::LoadDot(_),
                        ..
                    }
                    | Op::Api {
                        tag: Tag::BraceLoad | Tag::FlipIndex,
                        ..
                    },
                ) => i == 0,
                (
                    Watched::Shared,
                    Op::Api {
                        tag: Tag::Static { class, .. },
                        ..
                    },
                ) => &**class == "java.lang.Math",
                (
                    Watched::View(_),
                    Op::Native {
                        tag: Tag::Call(n), ..
                    },
                ) => {
                    if i == 0 {
                        VIEW_READS.contains(&&**n)
                    } else {
                        !STORES.contains(&&**n)
                    }
                }
                (
                    Watched::Entry,
                    Op::Native {
                        tag: Tag::Call(n), ..
                    },
                ) => {
                    if i == 0 {
                        matches!(&**n, "getKey" | "getValue" | "toString")
                    } else {
                        !STORES.contains(&&**n)
                    }
                }
                (
                    Watched::Shared | Watched::View(_),
                    Op::Api {
                        tag: Tag::ForEach(_),
                        ..
                    },
                ) => true,
                (
                    Watched::View(_) | Watched::Entry,
                    Op::Api {
                        tag: Tag::Static { class, .. },
                        ..
                    },
                ) => &**class != "java.util.Collections",
                (
                    Watched::View(_) | Watched::Entry,
                    Op::Arith { .. }
                    | Op::Cmp { .. }
                    | Op::Concat { .. }
                    | Op::If { .. }
                    | Op::BreakUnless(_),
                ) => true,
                (
                    Watched::Shared,
                    Op::Arith { .. }
                    | Op::DefArith { .. }
                    | Op::Cmp { .. }
                    | Op::Unary { .. }
                    | Op::Cast { .. }
                    | Op::Concat { .. }
                    | Op::If { .. }
                    | Op::BreakUnless(_)
                    | Op::Doc { .. },
                ) => true,
                _ => false,
            };
            if !ok {
                refuse(match kind {
                    Watched::Doc => "doc read other than by a mapped constant field name".into(),
                    Watched::Params => "params changed, handed on or read by a computed key".into(),
                    Watched::Foreign => "a value the native runtime cannot serve (_source)".into(),
                    Watched::Shared => "a params collection changed or handed on".into(),
                    Watched::View(_) | Watched::Entry => {
                        "a collection view (keySet, values, entrySet, subList) changed, cast or handed on".into()
                    }
                });
            }
        }
    });
    match err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// The registers `op` reads (not those of its nested blocks).
fn reads(op: &Op) -> Vec<Reg> {
    match op {
        Op::Move(_, s) => vec![*s],
        Op::Api { args, .. } | Op::Native { args, .. } | Op::Local { args, .. } => args.to_vec(),
        Op::Concat { args, .. } => args.to_vec(),
        Op::Doc { index, access, .. } => {
            if matches!(
                access,
                crate::vm::DocAccess::Get | crate::vm::DocAccess::Flip
            ) {
                vec![*index]
            } else {
                Vec::new()
            }
        }
        Op::Arith { a, b, .. } | Op::DefArith { a, b, .. } | Op::Cmp { a, b, .. } => vec![*a, *b],
        Op::Unary { src, .. } | Op::Cast { src, .. } | Op::InstanceOf { src, .. } => vec![*src],
        Op::If { cond, .. } => vec![*cond],
        Op::IfNull { src, .. } => vec![*src],
        Op::BreakUnless(r) | Op::Throw(r) | Op::Return(Some(r)) => vec![*r],
        _ => Vec::new(),
    }
}
