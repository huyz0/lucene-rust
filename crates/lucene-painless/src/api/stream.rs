//! `java.util.stream`: `Stream`, `IntStream`, `LongStream`, `DoubleStream`
//! and `Collectors`, as a lazy per-element pipeline over a snapshot of the
//! source. A stream runs once; a second use throws `IllegalStateException`.

use std::any::Any;
use std::cell::RefCell;

use crate::api::collection::{new_list, new_set};
use crate::api::regex::CharSeq;
use crate::lambda::{check_as, invoke_as};
use crate::object::JObject;
use crate::types::{as_f64, as_i64};
use crate::value::{SetKind, Value};
use crate::vm::{ApiFn, Throw, Vm};

const FN_PKG: &str = "java.util.function.";
const COLLECTORS: &str = "java.util.stream.Collectors";

/// What a stream's elements are.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Obj,
    Int,
    Long,
    Double,
}

impl Kind {
    /// The prefix of a primitive stream's functional interfaces.
    fn prefix(self) -> &'static str {
        match self {
            Kind::Obj => "",
            Kind::Int => "Int",
            Kind::Long => "Long",
            Kind::Double => "Double",
        }
    }

    fn class(self) -> &'static str {
        match self {
            Kind::Obj => "java.util.stream.Stream",
            Kind::Int => "java.util.stream.IntStream",
            Kind::Long => "java.util.stream.LongStream",
            Kind::Double => "java.util.stream.DoubleStream",
        }
    }

    fn component(self) -> &'static str {
        match self {
            Kind::Obj => "java.lang.Object",
            Kind::Int => "int",
            Kind::Long => "long",
            Kind::Double => "double",
        }
    }

    /// An element as this kind holds it.
    fn coerce(self, v: Value) -> Value {
        match self {
            Kind::Obj => v,
            Kind::Int => as_i64(&v).map_or(v, |x| Value::Int(x as i32)),
            Kind::Long => as_i64(&v).map_or(v, Value::Long),
            Kind::Double => as_f64(&v).map_or(v, Value::Double),
        }
    }
}

/// One stage of a pipeline.
#[derive(Debug, Clone)]
enum Step {
    /// `map`, `mapToInt`, ...: the function; the stage's kind is what it yields.
    Map(Value),
    /// `boxed`, `asLongStream`, ...: the same elements as another kind.
    Cast,
    Filter(Value),
    Peek(Value),
    Limit(usize),
    Skip(usize),
    Distinct,
}

#[derive(Debug)]
struct Pipe {
    kind: Kind,
    source: Vec<Value>,
    steps: Vec<(Step, Kind)>,
}

/// A stream: its pipeline until a stage or terminal consumes it.
#[derive(Debug)]
struct Stream {
    pipe: RefCell<Option<Pipe>>,
    kind: Kind,
}

/// A stream of `source`'s elements.
pub fn of(source: Vec<Value>) -> Value {
    make(Pipe {
        kind: Kind::Obj,
        source,
        steps: Vec::new(),
    })
}

/// The `IntStream` of `source`.
pub fn of_ints(source: Vec<Value>) -> Value {
    make(Pipe {
        kind: Kind::Int,
        source,
        steps: Vec::new(),
    })
}

fn make(pipe: Pipe) -> Value {
    let kind = pipe.steps.last().map_or(pipe.kind, |s| s.1);
    Value::object(Stream {
        kind,
        pipe: RefCell::new(Some(pipe)),
    })
}

/// A `Collector` of `Collectors`.
#[derive(Debug)]
enum Collector {
    List,
    Set,
    Joining(String, String, String),
}

impl JObject for Collector {
    fn class_name(&self) -> &str {
        "java.util.stream.Collector"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// The static methods of `Collectors`.
pub fn static_lookup(class: &str, name: &str, arity: usize) -> Option<ApiFn> {
    if class != COLLECTORS {
        return None;
    }
    Some(match (name, arity) {
        ("toList", 0) => |_vm, _a| Ok(Value::object(Collector::List)),
        ("toSet", 0) => |_vm, _a| Ok(Value::object(Collector::Set)),
        ("joining", 0) => |_vm, _a| {
            Ok(Value::object(Collector::Joining(
                String::new(),
                String::new(),
                String::new(),
            )))
        },
        ("joining", 1) => |_vm, a| {
            Ok(Value::object(Collector::Joining(
                a[0].java_to_string_checked()?,
                String::new(),
                String::new(),
            )))
        },
        ("joining", 3) => |_vm, a| {
            Ok(Value::object(Collector::Joining(
                a[0].java_to_string_checked()?,
                a[1].java_to_string_checked()?,
                a[2].java_to_string_checked()?,
            )))
        },
        _ => return None,
    })
}

fn consumed() -> Throw {
    Throw::new(
        "java.lang.IllegalStateException",
        "stream has already been operated upon or closed",
    )
}

fn truthy(v: &Value) -> bool {
    matches!(v, Value::Bool(true))
}

/// Pushes each element through the pipeline, stopping when `sink` says so.
fn drive(
    vm: &mut Vm,
    pipe: Pipe,
    sink: &mut dyn FnMut(&mut Vm, Value) -> Result<bool, Throw>,
) -> Result<(), Throw> {
    let Pipe {
        kind,
        source,
        steps,
    } = pipe;
    // Per-step counters (limit, skip) and the elements `distinct` has seen.
    let mut seen: Vec<usize> = vec![0; steps.len()];
    let mut distinct: Vec<Vec<Value>> = vec![Vec::new(); steps.len()];
    'elements: for item in source {
        // A spent `limit` ends the source before it is asked for more.
        if steps
            .iter()
            .enumerate()
            .any(|(i, (s, _))| matches!(s, Step::Limit(n) if seen[i] >= *n))
        {
            return Ok(());
        }
        let mut cur = item;
        let mut cur_kind = kind;
        for (i, (step, out)) in steps.iter().enumerate() {
            match step {
                Step::Map(f) => {
                    let (iface, method) = map_iface(cur_kind, *out);
                    cur = out.coerce(invoke_as(vm, f, &iface, &method, &[cur])?);
                }
                Step::Cast => cur = out.coerce(cur),
                Step::Filter(f) => {
                    let p = format!("{FN_PKG}{}Predicate", cur_kind.prefix());
                    if !truthy(&invoke_as(vm, f, &p, "test", std::slice::from_ref(&cur))?) {
                        continue 'elements;
                    }
                }
                Step::Peek(f) => {
                    let c = format!("{FN_PKG}{}Consumer", cur_kind.prefix());
                    invoke_as(vm, f, &c, "accept", std::slice::from_ref(&cur))?;
                }
                Step::Limit(_) => seen[i] += 1,
                Step::Skip(n) => {
                    if seen[i] < *n {
                        seen[i] += 1;
                        continue 'elements;
                    }
                }
                Step::Distinct => {
                    if distinct[i].iter().any(|d| d.java_equals(&cur)) {
                        continue 'elements;
                    }
                    distinct[i].push(cur.clone());
                }
            }
            cur_kind = *out;
        }
        if !sink(vm, cur)? {
            return Ok(());
        }
    }
    Ok(())
}

/// The interface and method a mapping from `from` to `to` calls.
fn map_iface(from: Kind, to: Kind) -> (String, String) {
    let (f, t) = (from.prefix(), to.prefix());
    match (from, to) {
        (Kind::Obj, Kind::Obj) => (format!("{FN_PKG}Function"), "apply".into()),
        (_, Kind::Obj) => (format!("{FN_PKG}{f}Function"), "apply".into()),
        (Kind::Obj, _) => (format!("{FN_PKG}To{t}Function"), format!("applyAs{t}")),
        _ if from == to => (format!("{FN_PKG}{f}UnaryOperator"), format!("applyAs{f}")),
        _ => (format!("{FN_PKG}{f}To{t}Function"), format!("applyAs{t}")),
    }
}

impl Stream {
    fn take(&self) -> Result<Pipe, Throw> {
        self.pipe.borrow_mut().take().ok_or_else(consumed)
    }

    /// A new stage after this one.
    fn stage(&self, vm: &Vm, step: Step, out: Kind) -> Result<Value, Throw> {
        // A function of the wrong arity is refused when the stage binds it.
        let from = self.kind;
        match &step {
            Step::Map(f) => {
                let (iface, method) = map_iface(from, out);
                check_as(vm, f, &iface, &method, 1)?;
            }
            Step::Filter(f) => {
                let p = format!("{FN_PKG}{}Predicate", from.prefix());
                check_as(vm, f, &p, "test", 1)?;
            }
            Step::Peek(f) => {
                let c = format!("{FN_PKG}{}Consumer", from.prefix());
                check_as(vm, f, &c, "accept", 1)?;
            }
            _ => {}
        }
        let mut pipe = self.take()?;
        pipe.steps.push((step, out));
        Ok(make(pipe))
    }

    fn collect_all(&self, vm: &mut Vm) -> Result<Vec<Value>, Throw> {
        let pipe = self.take()?;
        let mut out = Vec::new();
        drive(vm, pipe, &mut |_, v| {
            out.push(v);
            Ok(true)
        })?;
        Ok(out)
    }

    /// Whether any element satisfies `p` (`want` true), or any fails it.
    fn any(&self, vm: &mut Vm, p: &Value, want: bool) -> Result<bool, Throw> {
        let pipe = self.take()?;
        let iface = format!("{FN_PKG}{}Predicate", self.kind.prefix());
        let mut hit = false;
        drive(vm, pipe, &mut |vm, v| {
            let ok = truthy(&invoke_as(vm, p, &iface, "test", &[v])?);
            if ok == want {
                hit = true;
            }
            Ok(!hit)
        })?;
        Ok(hit)
    }

    fn call_stream(&self, vm: &mut Vm, name: &str, args: &[Value]) -> Result<Value, Throw> {
        let kind = self.kind;
        let count = |v: &Value| as_i64(v).map_or(0, |n| n.max(0) as usize);
        Ok(match (name, args) {
            ("map", [f]) => self.stage(vm, Step::Map(f.clone()), kind)?,
            ("filter", [f]) => self.stage(vm, Step::Filter(f.clone()), kind)?,
            ("peek", [f]) => self.stage(vm, Step::Peek(f.clone()), kind)?,
            ("limit", [n]) => {
                if as_i64(n).is_some_and(|n| n < 0) {
                    return Err(Throw::illegal_arg(as_i64(n).unwrap_or(0).to_string()));
                }
                self.stage(vm, Step::Limit(count(n)), kind)?
            }
            ("skip", [n]) => {
                if as_i64(n).is_some_and(|n| n < 0) {
                    return Err(Throw::illegal_arg(as_i64(n).unwrap_or(0).to_string()));
                }
                self.stage(vm, Step::Skip(count(n)), kind)?
            }
            ("distinct", []) => self.stage(vm, Step::Distinct, kind)?,
            ("mapToInt", [f]) => self.stage(vm, Step::Map(f.clone()), Kind::Int)?,
            ("mapToLong", [f]) => self.stage(vm, Step::Map(f.clone()), Kind::Long)?,
            ("mapToDouble", [f]) => self.stage(vm, Step::Map(f.clone()), Kind::Double)?,
            ("mapToObj", [f]) if kind != Kind::Obj => {
                self.stage(vm, Step::Map(f.clone()), Kind::Obj)?
            }
            ("boxed", []) if kind != Kind::Obj => self.stage(vm, Step::Cast, Kind::Obj)?,
            ("asLongStream", []) if kind == Kind::Int => self.stage(vm, Step::Cast, Kind::Long)?,
            ("asDoubleStream", []) if kind == Kind::Int || kind == Kind::Long => {
                self.stage(vm, Step::Cast, Kind::Double)?
            }
            ("count", []) => {
                let mut n = 0i64;
                let pipe = self.take()?;
                drive(vm, pipe, &mut |_, _| {
                    n += 1;
                    Ok(true)
                })?;
                Value::Long(n)
            }
            ("forEach" | "forEachOrdered", [f]) => {
                let pipe = self.take()?;
                let iface = format!("{FN_PKG}{}Consumer", kind.prefix());
                drive(vm, pipe, &mut |vm, v| {
                    invoke_as(vm, f, &iface, "accept", &[v])?;
                    Ok(true)
                })?;
                Value::Null
            }
            ("anyMatch", [p]) => Value::Bool(self.any(vm, p, true)?),
            ("noneMatch", [p]) => Value::Bool(!self.any(vm, p, true)?),
            ("allMatch", [p]) => Value::Bool(!self.any(vm, p, false)?),
            ("findFirst" | "findAny", []) if kind == Kind::Obj => {
                let pipe = self.take()?;
                let mut first = None;
                drive(vm, pipe, &mut |_, v| {
                    first = Some(v);
                    Ok(false)
                })?;
                if matches!(first, Some(Value::Null)) {
                    return Err(Throw::no_message("java.lang.NullPointerException"));
                }
                super::optional::wrap(first)
            }
            ("sum", []) if kind != Kind::Obj => {
                let items = self.collect_all(vm)?;
                match kind {
                    Kind::Int => Value::Int(
                        items
                            .iter()
                            .fold(0i32, |a, v| a.wrapping_add(as_i64(v).unwrap_or(0) as i32)),
                    ),
                    Kind::Long => Value::Long(
                        items
                            .iter()
                            .fold(0i64, |a, v| a.wrapping_add(as_i64(v).unwrap_or(0))),
                    ),
                    _ => Value::Double(items.iter().map(|v| as_f64(v).unwrap_or(0.0)).sum()),
                }
            }
            ("toArray", []) => {
                let items = self.collect_all(vm)?;
                Value::array(kind.component(), items)
            }
            ("toArray", [generator]) if kind == Kind::Obj => {
                let items = self.collect_all(vm)?;
                let made = invoke_as(
                    vm,
                    generator,
                    &format!("{FN_PKG}IntFunction"),
                    "apply",
                    &[Value::Int(items.len() as i32)],
                )?;
                match made {
                    Value::Array(a) => {
                        a.borrow_mut().items = items;
                        Value::Array(a)
                    }
                    _ => return Err(crate::compile::missing("Stream.toArray")),
                }
            }
            ("collect", [c]) if kind == Kind::Obj => {
                let items = self.collect_all(vm)?;
                let Value::Object(o) = c else {
                    return Err(Throw::no_message("java.lang.NullPointerException"));
                };
                match o.as_any().downcast_ref::<Collector>() {
                    Some(Collector::List) => new_list(items),
                    Some(Collector::Set) => new_set(SetKind::HashSet, items)?,
                    Some(Collector::Joining(sep, pre, suf)) => {
                        let parts: Result<Vec<String>, Throw> = items
                            .iter()
                            .map(|v| match v {
                                Value::Str(_) | Value::Null => v.java_to_string_checked(),
                                _ => Err(crate::compile::missing("Collectors.joining")),
                            })
                            .collect();
                        Value::string(format!("{pre}{}{suf}", parts?.join(sep)))
                    }
                    None => return Err(crate::compile::missing("Stream.collect")),
                }
            }
            _ => return Err(crate::compile::missing(format!("{}.{name}", kind.class()))),
        })
    }
}

impl JObject for Stream {
    fn class_name(&self) -> &str {
        self.kind.class()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn call(
        &self,
        vm: &mut Vm,
        _this: &Value,
        name: &str,
        args: &[Value],
    ) -> Option<Result<Value, Throw>> {
        Some(self.call_stream(vm, name, args))
    }
}
