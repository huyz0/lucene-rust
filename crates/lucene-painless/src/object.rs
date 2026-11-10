//! Host objects: values of classes that are neither primitives, strings
//! nor the collections [`Value`] models directly (a `StringBuilder`, an
//! iterator, a lambda, an exception, a doc-values view).

use std::any::Any;
use std::fmt::Debug;
use std::rc::Rc;

use crate::value::Value;
use crate::vm::{Throw, Vm};

/// An object a script holds. `call` answers the methods the object's class
/// has; `None` means the class has no such method, which the caller turns
/// into Painless's "dynamic method not found".
pub trait JObject: Debug + Any {
    /// `getClass().getName()`.
    fn class_name(&self) -> &str;

    fn as_any(&self) -> &dyn Any;

    /// Whether the object is an instance of the class with binary name
    /// `class` (its own class, a superclass or an interface).
    fn instance_of(&self, class: &str) -> bool {
        class == self.class_name() || class == "java.lang.Object"
    }

    fn java_equals(&self, other: &dyn JObject) -> bool {
        std::ptr::addr_eq(
            self as *const Self as *const (),
            other as *const dyn JObject as *const (),
        )
    }

    fn java_hash(&self) -> i32 {
        identity_hash(self as *const Self as *const ())
    }

    fn java_to_string(&self) -> String {
        format!("{}@{:x}", self.class_name(), self.java_hash())
    }

    /// Calls method `name` with `args` on the object; `this` is the value
    /// holding it.
    fn call(
        &self,
        _vm: &mut Vm,
        _this: &Value,
        _name: &str,
        _args: &[Value],
    ) -> Option<Result<Value, Throw>> {
        None
    }

    /// Reads a public field or a getter shortcut (`doc.value`).
    fn field(&self, _vm: &mut Vm, _name: &str) -> Option<Result<Value, Throw>> {
        None
    }

    /// Iterates the object, for an `Iterable`.
    fn iterate(&self, _vm: &mut Vm) -> Option<Result<Vec<Value>, Throw>> {
        None
    }
}

/// `System.identityHashCode` stand-in: stable for the object's lifetime.
pub fn identity_hash(p: *const ()) -> i32 {
    let a = p as usize as u64;
    ((a >> 4) ^ (a >> 36)) as i32 & 0x7fff_ffff
}

/// Downcasts a value to a host object of type `T`.
pub fn downcast<T: 'static>(v: &Value) -> Option<&T> {
    match v {
        Value::Object(o) => o.as_any().downcast_ref::<T>(),
        _ => None,
    }
}

/// The `Rc` of a host object of type `T`.
pub fn downcast_rc<T: 'static>(v: &Value) -> Option<Rc<dyn JObject>> {
    match v {
        Value::Object(o) if o.as_any().is::<T>() => Some(o.clone()),
        _ => None,
    }
}
