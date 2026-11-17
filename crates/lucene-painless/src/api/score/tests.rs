#![cfg(test)]
use std::rc::Rc;

use super::*;
use crate::vm::{Env, Program};

#[test]
fn mix32_is_bitmixers() {
    // BitMixer.mix32(1), BitMixer.mix32(-1).
    assert_eq!(mix32(1), 1_364_076_727);
    assert_eq!(mix32(0), 0);
}

/// An environment answering what the random functions ask for.
struct ScoreEnv {
    index: Value,
    shard: i32,
    doc_base: i32,
    field_value: Value,
}

impl Env for ScoreEnv {
    fn internal(&mut self, name: &str, _args: &[Value]) -> Result<Value, Throw> {
        match name {
            "_getIndex" => Ok(self.index.clone()),
            "_getShardId" => Ok(Value::Int(self.shard)),
            "_getDocBaseId" => Ok(Value::Int(self.doc_base)),
            _ => Err(Throw::unsupported(name)),
        }
    }

    fn doc_values(&mut self, _field: &str) -> Result<Value, Throw> {
        Ok(Value::list(
            crate::value::ListKind::ArrayList,
            vec![self.field_value.clone()],
        ))
    }
}

fn env() -> ScoreEnv {
    ScoreEnv {
        index: Value::str("idx"),
        shard: 3,
        doc_base: 42,
        field_value: Value::Long(7),
    }
}

fn with_vm<R>(env: &mut ScoreEnv, f: impl FnOnce(&mut Vm) -> R) -> R {
    let mut vm = Vm::new(Rc::new(Program::default()), env);
    f(&mut vm)
}

fn f(v: Result<Value, Throw>) -> f64 {
    match v.expect("a value") {
        Value::Double(d) => d,
        other => panic!("not a double: {other:?}"),
    }
}

fn d(x: f64) -> Value {
    Value::Double(x)
}

fn decay(name: &str, origin: f64, scale: f64, offset: f64, rate: f64) -> (ClassBinding, Value) {
    let b = class_binding(&format!("{UTILS}${name}")).expect("a binding");
    let state = with_vm(&mut env(), |vm| {
        (b.construct)(vm, &[d(origin), d(scale), d(offset), d(rate)])
    })
    .expect("a state");
    (b, state)
}

fn at(b: &ClassBinding, state: &Value, x: f64) -> f64 {
    f(with_vm(&mut env(), |vm| (b.call)(vm, state, &[d(x)])))
}

#[test]
fn saturation_and_sigmoid() {
    let sat = static_lookup("saturation", 2).unwrap();
    let sig = static_lookup("sigmoid", 3).unwrap();
    let run = |f: ApiFn, a: &[Value]| with_vm(&mut env(), |vm| f(vm, a));
    assert_eq!(f(run(sat, &[d(2.0), d(2.0)])), 0.5);
    assert_eq!(f(run(sig, &[d(1.0), d(1.0), d(2.0)])), 0.5);
    assert!((f(run(sig, &[d(3.0), d(1.0), d(2.0)])) - 0.9).abs() < 1e-12);
    assert!(static_lookup("other", 2).is_none());
}

#[test]
fn numeric_decays_halve_at_the_scale() {
    let (b, s) = decay("DecayNumericLinear", 0.0, 10.0, 0.0, 0.5);
    assert!((at(&b, &s, 0.0) - 1.0).abs() < 1e-12);
    assert!((at(&b, &s, 5.0) - 0.75).abs() < 1e-12);
    assert!((at(&b, &s, 10.0) - 0.5).abs() < 1e-12);
    assert_eq!(at(&b, &s, 1000.0), 0.0);
    let (b, s) = decay("DecayNumericExp", 0.0, 10.0, 0.0, 0.5);
    assert!((at(&b, &s, 10.0) - 0.5).abs() < 1e-12);
    assert!((at(&b, &s, -10.0) - 0.5).abs() < 1e-12);
    let (b, s) = decay("DecayNumericGauss", 0.0, 10.0, 0.0, 0.5);
    assert!((at(&b, &s, 10.0) - 0.5).abs() < 1e-12);
    // Inside the offset nothing decays.
    let (b, s) = decay("DecayNumericExp", 100.0, 10.0, 5.0, 0.5);
    assert_eq!(at(&b, &s, 103.0), 1.0);
    assert!(class_binding("other.Class").is_none());
    assert!(class_binding(&format!("{UTILS}$Nope")).is_none());
}

#[test]
fn random_scores_are_fractions_fixed_by_seed_and_document() {
    let doc = class_binding(&format!("{UTILS}$RandomScoreDoc")).unwrap();
    let s = with_vm(&mut env(), |vm| (doc.construct)(vm, &[Value::Int(1)])).unwrap();
    let a = f(with_vm(&mut env(), |vm| (doc.call)(vm, &s, &[])));
    let b = f(with_vm(&mut env(), |vm| (doc.call)(vm, &s, &[])));
    assert_eq!(a, b);
    assert!((0.0..1.0).contains(&a));
    let mut other = env();
    other.doc_base = 43;
    let c = f(with_vm(&mut other, |vm| (doc.call)(vm, &s, &[])));
    assert_ne!(a, c);
    let field = class_binding(&format!("{UTILS}$RandomScoreField")).unwrap();
    let fs = with_vm(&mut env(), |vm| {
        (field.construct)(vm, &[Value::Int(1), Value::str("price")])
    })
    .unwrap();
    let x = f(with_vm(&mut env(), |vm| (field.call)(vm, &fs, &[])));
    assert!((0.0..1.0).contains(&x));
    let mut missing_index = env();
    missing_index.index = Value::Null;
    let err = with_vm(&mut missing_index, |vm| {
        (doc.construct)(vm, &[Value::Int(1)])
    });
    assert!(err.is_err());
    assert_eq!(
        random("42", 5).java_to_string(),
        random("42", 5).java_to_string()
    );
}
