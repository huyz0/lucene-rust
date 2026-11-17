#![cfg(test)]
//! Driven by the registry of native classes: every entry must be coherent,
//! since the static check trusts it to say what the runtime serves.
use super::*;

#[test]
fn every_registered_class_names_and_serves_itself() {
    let mut env = NoEnv;
    for (name, sample) in NATIVE {
        let v = sample();
        assert_eq!(v.class_name(), *name, "the sample of {name}");
        let mut vm = probe_vm(&mut env);
        for (method, arity) in [("hashCode", 0), ("toString", 0), ("equals", 1)] {
            let args = vec![Value::Null; arity];
            let served = match dynamic::try_call(&mut vm, &v, method, &args) {
                None => false,
                Some(Err(t)) => t.class() != crate::api::UNSUPPORTED,
                Some(Ok(_)) => true,
            };
            assert!(served, "{name}.{method}/{arity} is served natively");
        }
    }
}

#[test]
fn the_registry_has_no_duplicates() {
    let mut seen = std::collections::HashSet::new();
    for c in native_classes() {
        assert!(seen.insert(c), "{c} is registered twice");
    }
}
