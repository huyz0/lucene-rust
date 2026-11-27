//! An ingest `script` processor: Painless source run over one document's
//! `ctx`, the document a map, and the map it leaves.
//!
//! The source goes through the front end (context `10`), the checker (the
//! native runtime must equal Painless on it) and the VM, over plain values:
//! `null`, booleans, numbers, strings, lists and maps with string keys.

use std::rc::Rc;

use crate::check::{check, Served};
use crate::compile::{compile, constant};
use crate::front::source::{self, Refused};
use crate::ir::Constant;
use crate::jmap::{JMap, Order};
use crate::specialize::{specialize, Facts};
use crate::value::{ListKind, MapKind, Value};
use crate::vm::{Env, Throw, Vm};

/// Why an ingest script did not run to its end.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IngestError {
    /// Painless rejects the source: the message and where.
    Compile { offset: usize, message: String },
    /// The front end or the runtime does not serve a construct yet.
    Unsupported(String),
    /// The runtime does not run the script exactly as Painless does.
    Refused(String),
    /// The script threw: the exception's class and message.
    Thrown {
        class: String,
        message: Option<String>,
    },
    /// The script left a value in `ctx` a document cannot hold.
    Unstorable(String),
}

impl std::fmt::Display for IngestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IngestError::Compile { offset, message } => write!(f, "{message} at offset {offset}"),
            IngestError::Unsupported(what) | IngestError::Refused(what) => f.write_str(what),
            IngestError::Thrown { class, message } => match message {
                Some(m) => write!(f, "{class}: {m}"),
                None => f.write_str(class),
            },
            IngestError::Unstorable(what) => write!(f, "cannot store a value of {what}"),
        }
    }
}

impl From<Refused> for IngestError {
    fn from(r: Refused) -> Self {
        match r {
            Refused::Compile { offset, message } => IngestError::Compile { offset, message },
            Refused::Unsupported(what) => IngestError::Unsupported(what),
        }
    }
}

/// The context's own methods: `getParams` and `getCtx`.
struct IngestEnv {
    params: Value,
    ctx: Value,
}

impl Env for IngestEnv {
    fn internal(&mut self, name: &str, _args: &[Value]) -> Result<Value, Throw> {
        match name {
            "getParams" => Ok(self.params.clone()),
            "getCtx" => Ok(self.ctx.clone()),
            _ => Err(Throw::unsupported(format!(
                "[{name}] is not available in this context"
            ))),
        }
    }
}

/// The document as a `LinkedHashMap`, the order it was read in, with lists
/// as `ArrayList`s.
pub fn to_value(c: &Constant) -> Value {
    match c {
        Constant::List(items) => {
            Value::list(ListKind::ArrayList, items.iter().map(to_value).collect())
        }
        Constant::Map(entries) => {
            let mut m = JMap::with_capacity(Order::Insertion, entries.len());
            for (k, v) in entries {
                m.insert(to_value(k), to_value(v));
            }
            Value::map(MapKind::LinkedHashMap, m)
        }
        // Scalars carry no identity: the constant's own value.
        other => constant(other).unwrap_or(Value::Null),
    }
}

/// The constant a value stands for in a document, or the class that cannot.
pub fn from_value(v: &Value) -> Result<Constant, IngestError> {
    Ok(match v {
        Value::Null => Constant::Null,
        Value::Bool(b) => Constant::Bool(*b),
        Value::Byte(n) => Constant::Int(i32::from(*n)),
        Value::Short(n) => Constant::Int(i32::from(*n)),
        Value::Char(n) => Constant::Str(String::from_utf16_lossy(&[*n])),
        Value::Int(n) => Constant::Int(*n),
        Value::Long(n) => Constant::Long(*n),
        Value::Float(n) => Constant::Double(f64::from(*n)),
        Value::Double(n) => Constant::Double(*n),
        Value::Str(s) => Constant::Str(s.to_string()),
        Value::List(l) => Constant::List(
            l.borrow()
                .items
                .iter()
                .map(from_value)
                .collect::<Result<_, _>>()?,
        ),
        Value::Array(a) => Constant::List(
            a.borrow()
                .items
                .iter()
                .map(from_value)
                .collect::<Result<_, _>>()?,
        ),
        Value::Map(m) => {
            let mut out = Vec::new();
            for (k, v) in m.borrow().map.iter() {
                let Value::Str(key) = k else {
                    return Err(IngestError::Unstorable(format!(
                        "a map key of {}",
                        k.class_name()
                    )));
                };
                out.push((Constant::Str(key.to_string()), from_value(v)?));
            }
            Constant::Map(out)
        }
        other => return Err(IngestError::Unstorable(other.class_name())),
    })
}

/// Runs `source` with `params` and the document `ctx` (a map), returning the
/// document the script leaves.
///
/// # Errors
/// [`IngestError`] for source Painless rejects, one this runtime does not
/// serve or cannot run as Painless does, an exception the script throws, or a
/// value the document cannot hold.
pub fn run(source: &str, params: &Constant, ctx: &Constant) -> Result<Constant, IngestError> {
    let script = source::compile(source, 10)?;
    let mut program = compile(&script).map_err(|e| IngestError::Unsupported(e.to_string()))?;
    let params = constant(params).map_err(|e| IngestError::Unsupported(e.to_string()))?;
    specialize(
        &mut program,
        &Facts {
            params: Some(&params),
            fields: &[],
        },
    );
    check(
        &script,
        &program,
        &Served {
            fields: &[],
            unmapped: &[],
            internals: &["getParams", "getCtx"],
        },
    )
    .map_err(|r| IngestError::Refused(r.0))?;
    let doc = to_value(ctx);
    let mut env = IngestEnv {
        params,
        ctx: doc.clone(),
    };
    Vm::new(Rc::new(program), &mut env)
        .execute(&[])
        .map_err(|t| {
            if t.class() == crate::api::UNSUPPORTED {
                IngestError::Unsupported(t.message().unwrap_or_default())
            } else {
                IngestError::Thrown {
                    class: t.class(),
                    message: t.message(),
                }
            }
        })?;
    from_value(&doc)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &str) -> Constant {
        Constant::Str(v.into())
    }

    fn doc(entries: Vec<(&str, Constant)>) -> Constant {
        Constant::Map(entries.into_iter().map(|(k, v)| (s(k), v)).collect())
    }

    fn get<'a>(d: &'a Constant, key: &str) -> Option<&'a Constant> {
        match d {
            Constant::Map(m) => m.iter().find(|(k, _)| *k == s(key)).map(|(_, v)| v),
            _ => None,
        }
    }

    #[test]
    fn a_script_reads_and_sets_fields_of_the_document() {
        let ctx = doc(vec![("a", Constant::Int(2)), ("b", s("x"))]);
        let params = doc(vec![("k", Constant::Int(10))]);
        let out = run(
            "ctx.c = ctx.a * params.k; ctx.b = ctx.b + 'y'; ctx.remove('a');",
            &params,
            &ctx,
        )
        .expect("runs");
        assert_eq!(get(&out, "c"), Some(&Constant::Int(20)));
        assert_eq!(get(&out, "b"), Some(&s("xy")));
        assert_eq!(get(&out, "a"), None);
    }

    #[test]
    fn nested_maps_and_lists_are_edited_in_place() {
        let ctx = doc(vec![
            ("o", doc(vec![("n", Constant::Int(1))])),
            ("l", Constant::List(vec![Constant::Int(1)])),
        ]);
        let out = run(
            "ctx.o.n += 1; ctx.l.add(2); ctx.o.m = 'new';",
            &doc(vec![]),
            &ctx,
        )
        .expect("runs");
        assert_eq!(
            get(&out, "l"),
            Some(&Constant::List(vec![Constant::Int(1), Constant::Int(2)]))
        );
        let o = get(&out, "o").expect("o");
        assert_eq!(get(o, "n"), Some(&Constant::Int(2)));
        assert_eq!(get(o, "m"), Some(&s("new")));
    }

    #[test]
    fn what_painless_rejects_or_throws_is_reported() {
        let ctx = doc(vec![]);
        assert!(matches!(
            run("int x = 'a';", &doc(vec![]), &ctx),
            Err(IngestError::Compile { .. })
        ));
        assert!(matches!(
            run("throw new IllegalArgumentException('no');", &doc(vec![]), &ctx),
            Err(IngestError::Thrown { class, message })
                if class.ends_with("IllegalArgumentException") && message.as_deref() == Some("no")
        ));
    }

    #[test]
    fn a_value_a_document_cannot_hold_is_refused() {
        let out = run("ctx.x = new StringBuilder();", &doc(vec![]), &doc(vec![]));
        assert!(out.is_err(), "{out:?}");
    }
}
