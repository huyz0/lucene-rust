//! The register program a script compiles to, and its interpreter.
//!
//! Every value an IR node pushes on the JVM stack gets a register, so the
//! program evaluates in the IR's order (Java's left-to-right) without an
//! operand stack. Control flow is structured: a block runs to its end or
//! leaves with a [`Flow`].

use std::rc::Rc;

use crate::object::JObject;
use crate::types::{self, Prim};
use crate::value::Value;

pub type Reg = u32;
pub type Block = Box<[Op]>;

/// A runtime library function: `args[0]` is the receiver for an instance
/// method.
pub type ApiFn = fn(&mut Vm, &[Value]) -> Result<Value, Throw>;

/// A compiled call whose behaviour depends on more than a plain function.
pub type Native = Rc<dyn Fn(&mut Vm, &[Value]) -> Result<Value, Throw>>;

/// A thrown Java exception (or error): the exception object.
#[derive(Debug, Clone)]
pub struct Throw(pub Value);

impl Throw {
    pub fn new(class: &str, message: impl Into<String>) -> Throw {
        Throw(Value::object(JException {
            class: class.into(),
            message: Some(message.into()),
            cause: None,
        }))
    }

    pub fn no_message(class: &str) -> Throw {
        Throw(Value::object(JException {
            class: class.into(),
            message: None,
            cause: None,
        }))
    }

    pub fn npe(message: impl Into<String>) -> Throw {
        Throw::new("java.lang.NullPointerException", message)
    }

    pub fn class_cast(message: impl Into<String>) -> Throw {
        Throw::new("java.lang.ClassCastException", message)
    }

    pub fn illegal_arg(message: impl Into<String>) -> Throw {
        Throw::new("java.lang.IllegalArgumentException", message)
    }

    pub fn unsupported(message: impl Into<String>) -> Throw {
        Throw::new("java.lang.UnsupportedOperationException", message)
    }

    pub fn index(class: &str, index: i64, len: usize) -> Throw {
        Throw::new(
            class,
            format!("Index {index} out of bounds for length {len}"),
        )
    }

    /// The thrown object's class.
    pub fn class(&self) -> String {
        self.0.class_name()
    }

    /// `getMessage()`, for a native exception.
    pub fn message(&self) -> Option<String> {
        crate::object::downcast::<JException>(&self.0).and_then(|e| e.message.clone())
    }
}

/// A Java exception created by the runtime or by a script's `new`.
#[derive(Debug, Clone)]
pub struct JException {
    pub class: String,
    pub message: Option<String>,
    pub cause: Option<Value>,
}

impl JObject for JException {
    fn class_name(&self) -> &str {
        &self.class
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn instance_of(&self, class: &str) -> bool {
        types::exception_is(&self.class, class)
    }

    fn java_to_string(&self) -> String {
        match &self.message {
            Some(m) => format!("{}: {m}", self.class),
            None => self.class.clone(),
        }
    }

    fn call(
        &self,
        _vm: &mut Vm,
        _this: &Value,
        name: &str,
        args: &[Value],
    ) -> Option<Result<Value, Throw>> {
        Some(Ok(match (name, args.len()) {
            ("getMessage" | "getLocalizedMessage", 0) => {
                self.message.as_deref().map_or(Value::Null, Value::str)
            }
            ("getCause", 0) => self.cause.clone().unwrap_or_default(),
            ("toString", 0) => Value::string(self.java_to_string()),
            _ => return None,
        }))
    }
}

/// How a block ended.
#[derive(Debug)]
pub enum Flow {
    Normal,
    Break,
    Continue,
    Return(Value),
}

/// Arithmetic on typed operands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Mul,
    Div,
    Rem,
    Add,
    Sub,
    Lsh,
    Rsh,
    Ush,
    And,
    Xor,
    Or,
    Find,
    Match,
}

impl BinOp {
    pub fn parse(s: &str) -> Option<BinOp> {
        Some(match s {
            "MUL" => BinOp::Mul,
            "DIV" => BinOp::Div,
            "REM" => BinOp::Rem,
            "ADD" => BinOp::Add,
            "SUB" => BinOp::Sub,
            "LSH" => BinOp::Lsh,
            "RSH" => BinOp::Rsh,
            "USH" => BinOp::Ush,
            "BWAND" => BinOp::And,
            "XOR" => BinOp::Xor,
            "BWOR" => BinOp::Or,
            "FIND" => BinOp::Find,
            "MATCH" => BinOp::Match,
            _ => return None,
        })
    }

    pub fn symbol(self) -> &'static str {
        match self {
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Rem => "%",
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Lsh => "<<",
            BinOp::Rsh => ">>",
            BinOp::Ush => ">>>",
            BinOp::And => "&",
            BinOp::Xor => "^",
            BinOp::Or => "|",
            BinOp::Find => "=~",
            BinOp::Match => "==~",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    Ne,
    EqR,
    NeR,
    Lt,
    Lte,
    Gt,
    Gte,
}

impl CmpOp {
    pub fn parse(s: &str) -> Option<CmpOp> {
        Some(match s {
            "EQ" => CmpOp::Eq,
            "NE" => CmpOp::Ne,
            "EQR" => CmpOp::EqR,
            "NER" => CmpOp::NeR,
            "LT" => CmpOp::Lt,
            "LTE" => CmpOp::Lte,
            "GT" => CmpOp::Gt,
            "GTE" => CmpOp::Gte,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnOp {
    Neg,
    Plus,
    BwNot,
    Not,
}

/// A conversion a `PainlessCast` performs on a value.
#[derive(Debug, Clone)]
pub enum CastOp {
    /// `char` to a one-character `String`.
    CharToString,
    /// A one-character `String` to `char`.
    StringToChar,
    /// Primitive conversion; `unbox` makes a null throw.
    Prim {
        to: Prim,
        unbox: bool,
    },
    /// `def` to a primitive (`boxed`: to its box class, null passes).
    DefTo {
        to: Prim,
        explicit: bool,
        boxed: bool,
    },
    DefToString {
        explicit: bool,
    },
    /// A reference cast (`checkcast`).
    Check(Rc<str>),
}

/// A `catch` clause.
#[derive(Debug)]
pub struct Catch {
    pub class: Rc<str>,
    pub slot: Reg,
    pub body: Block,
}

pub enum Op {
    Const(Reg, Value),
    Move(Reg, Reg),
    /// A library call with a fixed function.
    Api {
        dst: Reg,
        f: ApiFn,
        args: Box<[Reg]>,
    },
    /// A library call with captured state.
    Native {
        dst: Reg,
        f: Native,
        args: Box<[Reg]>,
    },
    /// A call of one of the script's functions.
    Local {
        dst: Reg,
        func: u32,
        args: Box<[Reg]>,
    },
    /// Arithmetic on primitives of the given promoted type.
    Arith {
        dst: Reg,
        op: BinOp,
        ty: Prim,
        a: Reg,
        b: Reg,
    },
    /// `def` arithmetic; `cast` is the call site's conversion of the result.
    DefArith {
        dst: Reg,
        op: BinOp,
        a: Reg,
        b: Reg,
        flags: i32,
        ret: Option<Rc<str>>,
    },
    Cmp {
        dst: Reg,
        op: CmpOp,
        ty: Option<Prim>,
        def: bool,
        a: Reg,
        b: Reg,
    },
    Unary {
        dst: Reg,
        op: UnOp,
        ty: Option<Prim>,
        src: Reg,
        ret: Option<Rc<str>>,
        explicit: bool,
    },
    Cast {
        dst: Reg,
        src: Reg,
        cast: CastOp,
    },
    InstanceOf {
        dst: Reg,
        src: Reg,
        class: Rc<str>,
    },
    Concat {
        dst: Reg,
        args: Box<[Reg]>,
    },
    If {
        cond: Reg,
        then: Block,
        otherwise: Block,
    },
    IfNull {
        src: Reg,
        then: Block,
        otherwise: Block,
    },
    Loop {
        head: Block,
        body: Block,
        after: Block,
    },
    BreakUnless(Reg),
    Break,
    Continue,
    Return(Option<Reg>),
    Throw(Reg),
    Try {
        body: Block,
        catches: Box<[Catch]>,
    },
    /// One loop iteration against the function's loop counter.
    Count,
}

impl std::fmt::Debug for Op {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Op::Const(d, v) => write!(f, "r{d} = {v:?}"),
            Op::Move(d, s) => write!(f, "r{d} = r{s}"),
            Op::Api { dst, args, .. } => write!(f, "r{dst} = api{args:?}"),
            Op::Native { dst, args, .. } => write!(f, "r{dst} = native{args:?}"),
            Op::Local { dst, func, args } => write!(f, "r{dst} = fn{func}{args:?}"),
            Op::Arith { dst, op, ty, a, b } => write!(f, "r{dst} = r{a} {op:?}.{ty:?} r{b}"),
            Op::DefArith { dst, op, a, b, .. } => write!(f, "r{dst} = r{a} {op:?}.def r{b}"),
            Op::Cmp { dst, op, a, b, .. } => write!(f, "r{dst} = r{a} {op:?} r{b}"),
            Op::Unary { dst, op, src, .. } => write!(f, "r{dst} = {op:?} r{src}"),
            Op::Cast { dst, src, cast } => write!(f, "r{dst} = ({cast:?}) r{src}"),
            Op::InstanceOf { dst, src, class } => write!(f, "r{dst} = r{src} instanceof {class}"),
            Op::Concat { dst, args } => write!(f, "r{dst} = concat{args:?}"),
            Op::If {
                cond,
                then,
                otherwise,
            } => write!(f, "if r{cond} {then:?} else {otherwise:?}"),
            Op::IfNull {
                src,
                then,
                otherwise,
            } => write!(f, "ifnull r{src} {then:?} else {otherwise:?}"),
            Op::Loop { head, body, after } => write!(f, "loop {head:?} {body:?} {after:?}"),
            Op::BreakUnless(r) => write!(f, "break unless r{r}"),
            Op::Break => write!(f, "break"),
            Op::Continue => write!(f, "continue"),
            Op::Return(r) => write!(f, "return {r:?}"),
            Op::Throw(r) => write!(f, "throw r{r}"),
            Op::Try { body, catches } => write!(f, "try {body:?} {catches:?}"),
            Op::Count => write!(f, "count"),
        }
    }
}

/// A compiled function.
#[derive(Debug)]
pub struct Func {
    pub name: String,
    pub arity: usize,
    pub regs: usize,
    pub body: Block,
    pub max_loop_counter: i32,
}

/// A compiled script.
#[derive(Debug, Default)]
pub struct Program {
    pub funcs: Vec<Func>,
    /// The function `execute` compiles to.
    pub entry: u32,
    /// Class bindings: the call sites that keep a bound instance.
    pub binding_sites: usize,
}

/// What a script's context provides: its parameters, the document, the
/// score, and the context's internal methods.
pub trait Env {
    /// Calls one of the script class's own methods (`getParams`, `getDoc`,
    /// `get_score`, `getCtx`).
    fn internal(&mut self, name: &str, args: &[Value]) -> Result<Value, Throw>;
}

/// An [`Env`] with only `params`.
#[derive(Debug, Default)]
pub struct ParamsEnv {
    pub params: Value,
}

impl Env for ParamsEnv {
    fn internal(&mut self, name: &str, _args: &[Value]) -> Result<Value, Throw> {
        match name {
            "getParams" => Ok(self.params.clone()),
            _ => Err(Throw::unsupported(format!(
                "[{name}] is not available in this context"
            ))),
        }
    }
}

/// Deepest call nesting before a `StackOverflowError`.
const MAX_CALL_DEPTH: u32 = 200;

pub const LOOP_LIMIT_MESSAGE: &str =
    "The maximum number of statements that can be executed in a loop has been reached.";

/// An execution: the program, the context, and per-execution state.
pub struct Vm<'a> {
    pub prog: Rc<Program>,
    pub env: &'a mut dyn Env,
    depth: u32,
    /// Class binding instances by call site.
    pub bindings: Vec<Option<Value>>,
}

struct Frame {
    regs: Vec<Value>,
    counter: i32,
    counting: bool,
}

impl<'a> Vm<'a> {
    pub fn new(prog: Rc<Program>, env: &'a mut dyn Env) -> Self {
        let sites = prog.binding_sites;
        Vm {
            prog,
            env,
            depth: 0,
            bindings: vec![None; sites],
        }
    }

    /// Runs the script's `execute` with `args`.
    pub fn execute(&mut self, args: &[Value]) -> Result<Value, Throw> {
        let entry = self.prog.entry;
        self.call_local(entry, args)
    }

    /// Calls function `func` of the script.
    pub fn call_local(&mut self, func: u32, args: &[Value]) -> Result<Value, Throw> {
        if self.depth >= MAX_CALL_DEPTH {
            return Err(Throw::no_message("java.lang.StackOverflowError"));
        }
        let prog = self.prog.clone();
        let f = &prog.funcs[func as usize];
        let mut regs = vec![Value::Null; f.regs.max(args.len())];
        for (r, a) in regs.iter_mut().zip(args) {
            *r = a.clone();
        }
        let mut frame = Frame {
            regs,
            counter: f.max_loop_counter,
            counting: f.max_loop_counter > 0,
        };
        self.depth += 1;
        let r = self.run(&mut frame, &f.body);
        self.depth -= 1;
        match r? {
            Flow::Return(v) => Ok(v),
            _ => Ok(Value::Null),
        }
    }

    fn args(frame: &Frame, regs: &[Reg]) -> Vec<Value> {
        regs.iter()
            .map(|&r| frame.regs[r as usize].clone())
            .collect()
    }

    fn run(&mut self, frame: &mut Frame, ops: &[Op]) -> Result<Flow, Throw> {
        for op in ops {
            match op {
                Op::Const(d, v) => frame.regs[*d as usize] = v.clone(),
                Op::Move(d, s) => frame.regs[*d as usize] = frame.regs[*s as usize].clone(),
                Op::Api { dst, f, args } => {
                    let a = Self::args(frame, args);
                    frame.regs[*dst as usize] = f(self, &a)?;
                }
                Op::Native { dst, f, args } => {
                    let a = Self::args(frame, args);
                    frame.regs[*dst as usize] = f(self, &a)?;
                }
                Op::Local { dst, func, args } => {
                    let a = Self::args(frame, args);
                    frame.regs[*dst as usize] = self.call_local(*func, &a)?;
                }
                Op::Arith { dst, op, ty, a, b } => {
                    let v = crate::def::arith(
                        *op,
                        *ty,
                        &frame.regs[*a as usize],
                        &frame.regs[*b as usize],
                    )?;
                    frame.regs[*dst as usize] = v;
                }
                Op::DefArith {
                    dst,
                    op,
                    a,
                    b,
                    flags,
                    ret,
                } => {
                    let v = crate::def::def_arith(
                        self,
                        *op,
                        &frame.regs[*a as usize],
                        &frame.regs[*b as usize],
                        *flags,
                        ret.as_deref(),
                    )?;
                    frame.regs[*dst as usize] = v;
                }
                Op::Cmp {
                    dst,
                    op,
                    ty,
                    def,
                    a,
                    b,
                } => {
                    let v = crate::def::compare(
                        *op,
                        *ty,
                        *def,
                        &frame.regs[*a as usize],
                        &frame.regs[*b as usize],
                    )?;
                    frame.regs[*dst as usize] = Value::Bool(v);
                }
                Op::Unary {
                    dst,
                    op,
                    ty,
                    src,
                    ret,
                    explicit,
                } => {
                    let v = crate::def::unary(
                        *op,
                        *ty,
                        &frame.regs[*src as usize],
                        ret.as_deref(),
                        *explicit,
                    )?;
                    frame.regs[*dst as usize] = v;
                }
                Op::Cast { dst, src, cast } => {
                    let v = crate::def::cast(&frame.regs[*src as usize], cast)?;
                    frame.regs[*dst as usize] = v;
                }
                Op::InstanceOf { dst, src, class } => {
                    let v = types::instance_of(&frame.regs[*src as usize], class);
                    frame.regs[*dst as usize] = Value::Bool(v);
                }
                Op::Concat { dst, args } => {
                    let mut s = String::new();
                    for &r in args.iter() {
                        frame.regs[r as usize].write_java(&mut s, 0);
                    }
                    frame.regs[*dst as usize] = Value::string(s);
                }
                Op::If {
                    cond,
                    then,
                    otherwise,
                } => {
                    let c = matches!(frame.regs[*cond as usize], Value::Bool(true));
                    let flow = self.run(frame, if c { then } else { otherwise })?;
                    if !matches!(flow, Flow::Normal) {
                        return Ok(flow);
                    }
                }
                Op::IfNull {
                    src,
                    then,
                    otherwise,
                } => {
                    let c = frame.regs[*src as usize].is_null();
                    let flow = self.run(frame, if c { then } else { otherwise })?;
                    if !matches!(flow, Flow::Normal) {
                        return Ok(flow);
                    }
                }
                Op::Loop { head, body, after } => loop {
                    match self.run(frame, head)? {
                        Flow::Normal | Flow::Continue => {}
                        Flow::Break => break,
                        r @ Flow::Return(_) => return Ok(r),
                    }
                    match self.run(frame, body)? {
                        Flow::Normal | Flow::Continue => {}
                        Flow::Break => break,
                        r @ Flow::Return(_) => return Ok(r),
                    }
                    match self.run(frame, after)? {
                        Flow::Normal | Flow::Continue => {}
                        Flow::Break => break,
                        r @ Flow::Return(_) => return Ok(r),
                    }
                },
                Op::BreakUnless(r) => {
                    if !matches!(frame.regs[*r as usize], Value::Bool(true)) {
                        return Ok(Flow::Break);
                    }
                }
                Op::Break => return Ok(Flow::Break),
                Op::Continue => return Ok(Flow::Continue),
                Op::Return(r) => {
                    return Ok(Flow::Return(
                        r.map_or(Value::Null, |r| frame.regs[r as usize].clone()),
                    ))
                }
                Op::Throw(r) => {
                    let v = frame.regs[*r as usize].clone();
                    if v.is_null() {
                        return Err(Throw::no_message("java.lang.NullPointerException"));
                    }
                    return Err(Throw(v));
                }
                Op::Try { body, catches } => match self.run(frame, body) {
                    Ok(Flow::Normal) => {}
                    Ok(flow) => return Ok(flow),
                    Err(t) => {
                        let class = t.class();
                        let Some(c) = catches.iter().find(|c| {
                            types::exception_is(&class, &c.class)
                                || types::instance_of(&t.0, &c.class)
                        }) else {
                            return Err(t);
                        };
                        frame.regs[c.slot as usize] = t.0;
                        let flow = self.run(frame, &c.body)?;
                        if !matches!(flow, Flow::Normal) {
                            return Ok(flow);
                        }
                    }
                },
                Op::Count => {
                    if frame.counting {
                        frame.counter -= 1;
                        if frame.counter <= 0 {
                            return Err(Throw::new(
                                "org.opensearch.painless.PainlessError",
                                LOOP_LIMIT_MESSAGE,
                            ));
                        }
                    }
                }
            }
        }
        Ok(Flow::Normal)
    }
}
