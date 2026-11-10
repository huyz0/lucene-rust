//! The document a search script reads: `doc` (OpenSearch's
//! `LeafDocLookup`, a map from field name to its values) and each field's
//! `ScriptDocValues` view.
//!
//! The values themselves come from the host through [`crate::vm::Env`]:
//! `doc['f']` asks [`Env::doc_values`](crate::vm::Env::doc_values) for the
//! current document's values of `f`.

use std::rc::Rc;

use crate::api::{self, int_arg};
use crate::object::JObject;
use crate::value::{ListKind, Value};
use crate::vm::{Throw, Vm};

/// `LeafDocLookup`'s class.
pub const DOC_LOOKUP_CLASS: &str = "org.opensearch.search.lookup.LeafDocLookup";

/// `ScriptDocValues.getValue` and `get` on a document without a value.
pub const NO_VALUE: &str = "A document doesn't have a value for a field! Use doc[<field>].size()==0 to check if a document is missing a field!";

/// `doc`: forwards every lookup to the host.
#[derive(Debug, Default)]
pub struct DocLookup;

impl JObject for DocLookup {
    fn class_name(&self) -> &str {
        DOC_LOOKUP_CLASS
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn instance_of(&self, class: &str) -> bool {
        matches!(
            class,
            DOC_LOOKUP_CLASS | "java.util.Map" | "java.lang.Object"
        )
    }

    fn call(
        &self,
        vm: &mut Vm,
        _this: &Value,
        name: &str,
        args: &[Value],
    ) -> Option<Result<Value, Throw>> {
        Some(match (name, args) {
            ("get", [key]) => field_name(key).and_then(|f| vm.env.doc_values(&f)),
            ("containsKey", [key]) => match key {
                Value::Str(f) => vm.env.doc_contains(f).map(Value::Bool),
                // `LeafDocLookup.containsKey` casts its key to a String.
                other => Err(Throw::class_cast(format!(
                    "class {} cannot be cast to class java.lang.String",
                    other.class_name()
                ))),
            },
            _ => Err(crate::compile::missing(format!(
                "doc.{name}/{}",
                args.len()
            ))),
        })
    }

    fn field(&self, vm: &mut Vm, name: &str) -> Option<Result<Value, Throw>> {
        Some(vm.env.doc_values(name))
    }
}

fn field_name(key: &Value) -> Result<Rc<str>, Throw> {
    match key {
        Value::Str(s) => Ok(s.clone()),
        Value::Null => Err(api::npe()),
        other => Err(Throw::class_cast(format!(
            "class {} cannot be cast to class java.lang.String",
            other.class_name()
        ))),
    }
}

/// The kinds of `ScriptDocValues` a field reads as.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocKind {
    /// `ScriptDocValues.Longs`: the integral types.
    Longs,
    /// `ScriptDocValues.Doubles`: double, float, half_float, scaled_float.
    Doubles,
    /// `ScriptDocValues.Booleans`.
    Booleans,
    /// `ScriptDocValues.Strings`: keyword.
    Strings,
}

impl DocKind {
    pub fn class_name(self) -> &'static str {
        match self {
            DocKind::Longs => "org.opensearch.index.fielddata.ScriptDocValues$Longs",
            DocKind::Doubles => "org.opensearch.index.fielddata.ScriptDocValues$Doubles",
            DocKind::Booleans => "org.opensearch.index.fielddata.ScriptDocValues$Booleans",
            DocKind::Strings => "org.opensearch.index.fielddata.ScriptDocValues$Strings",
        }
    }
}

/// One field's values in the current document, ascending (a keyword's
/// distinct and sorted, as its sorted-set doc values hold them).
#[derive(Debug)]
pub struct DocValues {
    pub kind: DocKind,
    pub items: Vec<Value>,
}

impl DocValues {
    pub fn value(kind: DocKind, items: Vec<Value>) -> Value {
        Value::object(DocValues { kind, items })
    }

    fn get(&self, index: &Value) -> Result<Value, Throw> {
        let i = int_arg(index)?;
        if self.items.is_empty() {
            return Err(Throw::new("java.lang.IllegalStateException", NO_VALUE));
        }
        match usize::try_from(i).ok().and_then(|i| self.items.get(i)) {
            Some(v) => Ok(v.clone()),
            // Java reads its reused array past the document's values: a
            // stale value of an earlier document, or an index error.
            None => Err(crate::compile::missing(format!(
                "doc values index {i} of {}",
                self.items.len()
            ))),
        }
    }

    fn as_list(&self) -> Value {
        Value::list(ListKind::Immutable, self.items.clone())
    }
}

impl JObject for DocValues {
    fn class_name(&self) -> &str {
        self.kind.class_name()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn instance_of(&self, class: &str) -> bool {
        class == self.kind.class_name()
            || matches!(
                class,
                "org.opensearch.index.fielddata.ScriptDocValues"
                    | "java.util.AbstractList"
                    | "java.util.AbstractCollection"
                    | "java.util.List"
                    | "java.util.Collection"
                    | "java.lang.Iterable"
                    | "java.lang.Object"
            )
    }

    fn java_equals(&self, other: &dyn JObject) -> bool {
        // `AbstractList.equals`: the same elements in the same order.
        match other.as_any().downcast_ref::<DocValues>() {
            Some(o) => {
                self.items.len() == o.items.len()
                    && self
                        .items
                        .iter()
                        .zip(&o.items)
                        .all(|(a, b)| a.java_equals(b))
            }
            None => false,
        }
    }

    fn java_hash(&self) -> i32 {
        self.as_list().java_hash()
    }

    fn java_to_string(&self) -> String {
        self.as_list().java_to_string()
    }

    fn call(
        &self,
        vm: &mut Vm,
        _this: &Value,
        name: &str,
        args: &[Value],
    ) -> Option<Result<Value, Throw>> {
        Some(match (name, args) {
            ("getValue", []) => self.get(&Value::Int(0)),
            ("get", [i]) => self.get(i),
            ("size", []) | ("getLength", []) => Ok(Value::Int(self.items.len() as i32)),
            ("isEmpty", []) => Ok(Value::Bool(self.items.is_empty())),
            ("add" | "remove" | "set" | "sort" | "replaceAll", _) => {
                Err(Throw::unsupported("doc values are unmodifiable"))
            }
            ("clear" | "addAll" | "removeAll" | "retainAll" | "removeIf", _) => {
                Err(Throw::no_message("java.lang.UnsupportedOperationException"))
            }
            _ => api::dynamic::call(vm, &self.as_list(), name, args),
        })
    }

    fn field(&self, _vm: &mut Vm, name: &str) -> Option<Result<Value, Throw>> {
        // `doc['f'].value`, `.empty` and `.length`: the getters, directly.
        Some(match name {
            "value" => self.get(&Value::Int(0)),
            "empty" => Ok(Value::Bool(self.items.is_empty())),
            "length" => Ok(Value::Int(self.items.len() as i32)),
            _ => return None,
        })
    }

    fn iterate(&self, _vm: &mut Vm) -> Option<Result<Vec<Value>, Throw>> {
        Some(Ok(self.items.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::{Env, Program};

    struct OneField;

    impl Env for OneField {
        fn internal(&mut self, _name: &str, _args: &[Value]) -> Result<Value, Throw> {
            unreachable!()
        }
        fn doc_values(&mut self, field: &str) -> Result<Value, Throw> {
            match field {
                "n" => Ok(DocValues::value(
                    DocKind::Longs,
                    vec![Value::Long(3), Value::Long(7)],
                )),
                "e" => Ok(DocValues::value(DocKind::Doubles, vec![])),
                _ => Err(Throw::illegal_arg(format!(
                    "No field found for [{field}] in mapping"
                ))),
            }
        }
        fn doc_contains(&mut self, field: &str) -> Result<bool, Throw> {
            Ok(matches!(field, "n" | "e"))
        }
    }

    fn vm_call(recv: &Value, name: &str, args: &[Value]) -> Result<Value, Throw> {
        let prog = Rc::new(Program {
            funcs: vec![],
            entry: 0,
            binding_sites: 0,
        });
        let mut env = OneField;
        let mut vm = Vm::new(prog, &mut env);
        api::dynamic::call(&mut vm, recv, name, args)
    }

    #[test]
    fn doc_lookup_forwards_to_the_host() {
        let doc = Value::object(DocLookup);
        let n = vm_call(&doc, "get", &[Value::str("n")]).unwrap();
        assert_eq!(vm_call(&n, "getValue", &[]).unwrap().java_to_string(), "3");
        assert_eq!(
            vm_call(&n, "get", &[Value::Int(1)])
                .unwrap()
                .java_to_string(),
            "7"
        );
        assert_eq!(vm_call(&n, "size", &[]).unwrap().java_to_string(), "2");
        assert_eq!(
            vm_call(&n, "contains", &[Value::Long(7)])
                .unwrap()
                .java_to_string(),
            "true"
        );
        assert_eq!(n.java_to_string(), "[3, 7]");
        assert_eq!(
            vm_call(&doc, "containsKey", &[Value::str("x")])
                .unwrap()
                .java_to_string(),
            "false"
        );
        let missing = vm_call(&doc, "get", &[Value::str("x")]).unwrap_err();
        assert_eq!(missing.class(), "java.lang.IllegalArgumentException");
        let e = vm_call(&doc, "get", &[Value::str("e")]).unwrap();
        let err = vm_call(&e, "getValue", &[]).unwrap_err();
        assert_eq!(err.class(), "java.lang.IllegalStateException");
        assert_eq!(err.message().as_deref(), Some(NO_VALUE));
        let add = vm_call(&n, "add", &[Value::Long(1)]).unwrap_err();
        assert_eq!(add.class(), "java.lang.UnsupportedOperationException");
        assert_eq!(
            vm_call(&n, "get", &[Value::Int(5)]).unwrap_err().class(),
            api::UNSUPPORTED
        );
    }
}
