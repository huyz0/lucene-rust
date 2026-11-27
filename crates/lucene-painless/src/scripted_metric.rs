//! The scripts of a `scripted_metric` aggregation that need no document:
//! `init_script` (`params` and an empty `state`), `combine_script` (`params`
//! and the shard's `state`, returning the shard's value) and `reduce_script`
//! (`params` and `states`, returning the aggregation's value). The
//! `map_script` reads documents and runs where they are.
//!
//! Each source goes through the front end, the checker and the VM, over
//! plain values: `null`, booleans, numbers, strings, lists and maps with
//! string keys.

use std::rc::Rc;

use crate::check::{check, Served};
use crate::compile::{compile, constant};
use crate::front::source;
use crate::ingest::{from_value, to_value, IngestError};
use crate::ir::Constant;
use crate::jmap::{JMap, Order};
use crate::specialize::{specialize, Facts};
use crate::value::{MapKind, Value};
use crate::vm::{Env, Throw, Vm};

/// A script's context: the engine's context number and what it reads.
#[derive(Clone, Copy)]
enum Phase {
    Init,
    Combine,
    Reduce,
}

struct MetricEnv {
    params: Value,
    state: Value,
    states: Value,
}

impl Env for MetricEnv {
    fn internal(&mut self, name: &str, _args: &[Value]) -> Result<Value, Throw> {
        match name {
            "getParams" => Ok(self.params.clone()),
            "getState" => Ok(self.state.clone()),
            "getStates" => Ok(self.states.clone()),
            _ => Err(Throw::unsupported(format!(
                "[{name}] is not available in this context"
            ))),
        }
    }
}

fn execute(
    source: &str,
    phase: Phase,
    params: &Constant,
    state: Value,
    states: Value,
) -> Result<Value, IngestError> {
    let (context, internals): (u8, &[&str]) = match phase {
        Phase::Init => (12, &["getParams", "getState"]),
        Phase::Combine => (13, &["getParams", "getState"]),
        Phase::Reduce => (14, &["getParams", "getStates"]),
    };
    let script = source::compile(source, context)?;
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
            internals,
        },
    )
    .map_err(|r| IngestError::Refused(r.0))?;
    let mut env = MetricEnv {
        params,
        state,
        states,
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
        })
}

fn empty_state() -> Value {
    Value::map(MapKind::HashMap, JMap::with_capacity(Order::Hash, 0))
}

/// Runs `init_script` and returns the `state` it leaves.
///
/// # Errors
/// [`IngestError`] for source Painless rejects, one this runtime does not
/// serve or cannot run as Painless does, an exception the script throws, or
/// a state a document cannot hold.
pub fn init(source: &str, params: &Constant) -> Result<Constant, IngestError> {
    let state = empty_state();
    execute(source, Phase::Init, params, state.clone(), Value::Null)?;
    from_value(&state)
}

/// Runs `combine_script` over a shard's `state` and returns its value.
///
/// # Errors
/// As [`init`].
pub fn combine(source: &str, params: &Constant, state: &Constant) -> Result<Constant, IngestError> {
    let value = execute(source, Phase::Combine, params, to_value(state), Value::Null)?;
    from_value(&value)
}

/// Runs `reduce_script` over every shard's combined value and returns the
/// aggregation's value.
///
/// # Errors
/// As [`init`].
pub fn reduce(
    source: &str,
    params: &Constant,
    states: &[Constant],
) -> Result<Constant, IngestError> {
    let list = Constant::List(states.to_vec());
    let value = execute(source, Phase::Reduce, params, Value::Null, to_value(&list))?;
    from_value(&value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn none() -> Constant {
        Constant::Map(Vec::new())
    }

    fn s(v: &str) -> Constant {
        Constant::Str(v.into())
    }

    #[test]
    fn init_builds_the_state_combine_returns_the_shards_value() {
        let params = Constant::Map(vec![(s("start"), Constant::Int(5))]);
        let state = init(
            "state.sum = params.start; state.items = new ArrayList();",
            &params,
        )
        .expect("init");
        let Constant::Map(m) = &state else {
            panic!("{state:?}")
        };
        assert!(m.contains(&(s("sum"), Constant::Int(5))));
        let value = combine("return state.sum * 2", &params, &state).expect("combine");
        assert_eq!(value, Constant::Int(10));
    }

    #[test]
    fn reduce_folds_the_shards_values() {
        let states = [Constant::Int(3), Constant::Int(4), Constant::Int(5)];
        let value = reduce(
            "double total = 0; for (s in states) { total += s } return total",
            &none(),
            &states,
        )
        .expect("reduce");
        assert_eq!(value, Constant::Double(12.0));
    }

    #[test]
    fn what_painless_rejects_or_throws_is_reported() {
        assert!(matches!(
            init("int x = 'a';", &none()),
            Err(IngestError::Compile { .. })
        ));
        assert!(matches!(
            reduce("throw new IllegalStateException('no')", &none(), &[]),
            Err(IngestError::Thrown { class, .. }) if class.ends_with("IllegalStateException")
        ));
        // `states` is not in an init script
        assert!(matches!(
            init("return states.size()", &none()),
            Err(IngestError::Compile { .. })
        ));
    }
}
