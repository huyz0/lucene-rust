//! Lambdas and method references: a function value a script passes to a
//! `List.sort`, a `Map.computeIfAbsent`, or calls through its interface.

use std::rc::Rc;

use crate::api;
use crate::object::JObject;
use crate::value::Value;
use crate::vm::{Throw, Vm};

/// What a function value calls.
#[derive(Debug, Clone)]
pub enum Target {
    /// One of the script's functions (a lambda body or a user function).
    Local(u32),
    /// A static method `class.name`.
    Static(Rc<str>, Rc<str>),
    /// An instance method: the first argument is the receiver.
    Virtual(Rc<str>),
    /// A constructor.
    Ctor(Rc<str>),
    /// `int[]::new`: the component type.
    NewArray(Rc<str>),
    /// `Type::name` resolved when called: static if the class has a static
    /// method of that arity, otherwise an instance method of the receiver.
    ByName(Rc<str>, Rc<str>),
}

#[derive(Debug, Clone)]
pub struct Lambda {
    pub target: Target,
    pub captures: Vec<Value>,
    /// The functional interface, when the compiler knew it.
    pub iface: Option<Rc<str>>,
    /// The interface method's return type.
    pub ret: Option<Rc<str>>,
}

impl JObject for Lambda {
    fn class_name(&self) -> &str {
        self.iface
            .as_deref()
            .unwrap_or("org.opensearch.painless.Lambda")
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn instance_of(&self, class: &str) -> bool {
        match &self.iface {
            Some(i) => &**i == class || class == "java.lang.Object",
            None => true,
        }
    }

    fn call(
        &self,
        vm: &mut Vm,
        this: &Value,
        name: &str,
        args: &[Value],
    ) -> Option<Result<Value, Throw>> {
        Some(match name {
            "equals" if args.len() == 1 => Ok(Value::Bool(crate::def::identical(this, &args[0]))),
            "hashCode" if args.is_empty() => Ok(Value::Int(self.java_hash())),
            "toString" if args.is_empty() => Ok(Value::string(self.java_to_string())),
            // Default methods of the functional interfaces.
            "reversed" if args.is_empty() => Ok(api::functional::reversed(this.clone())),
            "thenComparing" if args.len() == 1 => Ok(api::functional::then_comparing(
                this.clone(),
                args[0].clone(),
            )),
            "andThen" if args.len() == 1 => {
                Ok(api::functional::and_then(this.clone(), args[0].clone()))
            }
            "compose" if args.len() == 1 => {
                Ok(api::functional::and_then(args[0].clone(), this.clone()))
            }
            "negate" if args.is_empty() => Ok(api::functional::negate(this.clone())),
            "and" if args.len() == 1 => Ok(api::functional::and(this.clone(), args[0].clone())),
            "or" if args.len() == 1 => Ok(api::functional::or(this.clone(), args[0].clone())),
            // Any other name is the interface's single abstract method.
            _ => invoke_lambda(vm, self, args),
        })
    }
}

/// Calls a function value with `args`.
pub fn invoke(vm: &mut Vm, f: &Value, args: &[Value]) -> Result<Value, Throw> {
    match f {
        Value::Object(o) => {
            if let Some(l) = o.as_any().downcast_ref::<Lambda>() {
                return invoke_lambda(vm, l, args);
            }
            if let Some(c) = o.as_any().downcast_ref::<api::functional::Composed>() {
                return c.invoke(vm, args);
            }
            Err(Throw::class_cast(format!(
                "{} is not a function",
                o.class_name()
            )))
        }
        Value::Null => Err(Throw::no_message("java.lang.NullPointerException")),
        other => Err(Throw::class_cast(format!(
            "{} is not a function",
            other.class_name()
        ))),
    }
}

fn invoke_lambda(vm: &mut Vm, l: &Lambda, args: &[Value]) -> Result<Value, Throw> {
    let mut all: Vec<Value> = Vec::with_capacity(l.captures.len() + args.len());
    all.extend_from_slice(&l.captures);
    all.extend_from_slice(args);
    let r = match &l.target {
        Target::Local(f) => vm.call_local(*f, &all)?,
        Target::Static(class, name) => match api::static_method(class, name, all.len()) {
            Some(f) => f(vm, &all)?,
            None => {
                return Err(crate::compile::missing(format!(
                    "static method {class}.{name}/{}",
                    all.len()
                )))
            }
        },
        Target::Virtual(name) => {
            let (recv, rest) = all
                .split_first()
                .ok_or_else(|| Throw::illegal_arg("reference without a receiver"))?;
            api::dynamic::call(vm, recv, name, rest)?
        }
        Target::Ctor(class) => match api::constructor(class, all.len()) {
            Some(f) => f(vm, &all)?,
            None => {
                return Err(crate::compile::missing(format!(
                    "constructor {class}/{}",
                    all.len()
                )))
            }
        },
        Target::NewArray(comp) => api::array::new_dims(comp, &all)?,
        Target::ByName(class, name) => {
            if let Some(f) = api::static_method(class, name, all.len()) {
                f(vm, &all)?
            } else if name.as_ref() == "new" {
                match api::constructor(class, all.len()) {
                    Some(f) => f(vm, &all)?,
                    None => {
                        return Err(crate::compile::missing(format!(
                            "constructor {class}/{}",
                            all.len()
                        )))
                    }
                }
            } else {
                let (recv, rest) = all
                    .split_first()
                    .ok_or_else(|| Throw::illegal_arg("reference without a receiver"))?;
                api::dynamic::call(vm, recv, name, rest)?
            }
        }
    };
    Ok(match l.ret.as_deref() {
        Some(ret) => adapt_return(r, ret)?,
        None => r,
    })
}

/// The conversion `LambdaMetafactory` applies to a delegate's result for
/// the interface's return type.
fn adapt_return(v: Value, ret: &str) -> Result<Value, Throw> {
    use crate::types::{convert, prim};
    match prim(ret) {
        Some(p) if !v.is_null() => Ok(convert(&v, p).unwrap_or(v)),
        Some(_) => Err(Throw::no_message("java.lang.NullPointerException")),
        None => Ok(v),
    }
}
