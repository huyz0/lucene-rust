//! Search scripts in JavaScript (`lang: "js"`), run on V8.
//!
//! A script is an expression, or a function body that `return`s, over the
//! same names a Painless script reads: `doc['f'].value`, `doc['f'].size()`,
//! `doc['f'].get(i)`, `doc['f'].empty`, `params`, and `_score`. It runs in
//! strict mode, in a context of its own whose built-ins and global object are
//! frozen: no script changes what another document or request sees. V8
//! itself has no files, network, timers or modules, and code generation from
//! strings (`eval`, `Function`) is off, as is WebAssembly.
//!
//! Each thread keeps one isolate, with every script it has compiled; a
//! script compiles once per thread. A search's calls share one deadline
//! ([`Limits::timeout`]), past which V8 stops the script; an isolate past its
//! heap limit ([`Limits::max_heap`]) stops the script and is dropped.
//!
//! Field values cross into V8 a column at a time: the first read of a field
//! in a call asks the host ([`Fields::load`]) for its values in every document
//! of the call, into typed arrays the thread reuses.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::ffi::c_void;
use std::hash::{Hash, Hasher};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Once, OnceLock};
use std::time::{Duration, Instant};

const PRELUDE: &str = include_str!("prelude.js");

/// How a field reads in `doc`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kind {
    /// A number: a long (exact up to 2^53), double, float or half float.
    Number,
    /// A boolean.
    Boolean,
    /// A keyword: its terms, read by ordinal ([`Fields::term`]).
    Keyword,
    /// A mapped field without doc values a script can read: any read fails.
    Unreadable,
}

impl Kind {
    fn code(self) -> i32 {
        match self {
            Kind::Number => 0,
            Kind::Boolean => 1,
            Kind::Keyword => 2,
            Kind::Unreadable => 3,
        }
    }
}

/// A script's limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// How long one search may run the script, over all its calls.
    pub timeout: Duration,
    /// The heap each thread's isolate may grow to (the first script a thread
    /// runs sets it).
    pub max_heap: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            timeout: Duration::from_secs(10),
            max_heap: 128 << 20,
        }
    }
}

/// A script's `params`.
#[derive(Debug, Clone, PartialEq)]
pub enum Param {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    List(Vec<Param>),
    Map(Vec<(String, Param)>),
}

impl Param {
    fn json(&self, out: &mut String) {
        match self {
            Param::Null => out.push_str("null"),
            Param::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Param::Number(n) if n.is_finite() => out.push_str(&format!("{n:?}")),
            // JSON has no NaN or infinities: the prelude-free form V8 parses.
            Param::Number(_) => out.push_str("null"),
            Param::String(s) => json_string(s, out),
            Param::List(items) => {
                out.push('[');
                for (i, p) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    p.json(out);
                }
                out.push(']');
            }
            Param::Map(entries) => {
                out.push('{');
                for (i, (k, v)) in entries.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    json_string(k, out);
                    out.push(':');
                    v.json(out);
                }
                out.push('}');
            }
        }
    }
}

fn json_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 || c == '\u{2028}' || c == '\u{2029}' => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// A request's `params` for a script, as it reads them: deep-frozen.
#[derive(Debug, Clone)]
pub struct Params {
    json: Arc<str>,
    key: u64,
}

impl Params {
    /// `params` (a map); NaN and the infinities read as `null`.
    pub fn new(params: &Param) -> Self {
        let mut json = String::new();
        params.json(&mut json);
        let key = hash(&json);
        Params {
            json: json.into(),
            key,
        }
    }
}

fn hash<T: Hash + ?Sized>(t: &T) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    t.hash(&mut h);
    h.finish()
}

/// A script and the fields it may read, for one search.
#[derive(Debug)]
pub struct Script {
    source: Arc<str>,
    fields: Vec<(String, Kind)>,
    limits: Limits,
    key: u64,
    /// The search's deadline, from when it first ran the script: as an
    /// instant and in the watchdog's nanoseconds.
    deadline: OnceLock<(Instant, u64)>,
}

impl Script {
    /// `source` reading `fields` (by the names `doc` holds) under `limits`.
    pub fn new(source: &str, fields: Vec<(String, Kind)>, limits: Limits) -> Self {
        let key = hash(&(source, &fields));
        Script {
            source: source.into(),
            fields,
            limits,
            key,
            deadline: OnceLock::new(),
        }
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    pub fn fields(&self) -> &[(String, Kind)] {
        &self.fields
    }

    /// Whether the script compiles: `Err` with V8's syntax error if not.
    pub fn check(&self) -> Result<(), Failure> {
        with_engine(self.limits, |e| e.compiled(self).map(|_| ()))
    }

    fn deadline(&self) -> (Instant, u64) {
        *self.deadline.get_or_init(|| {
            let d = Instant::now() + self.limits.timeout;
            (d, nanos(d))
        })
    }
}

/// What reads a call's field values: the documents of the call, in order.
pub trait Fields {
    /// Field `field`'s values in each document of the call appended to
    /// `values`, with `starts` holding where each document's begin and,
    /// last, where the last one's end (so one more entry than documents).
    /// A boolean is `1` or `0`, a keyword its ordinal.
    fn load(
        &mut self,
        field: usize,
        values: &mut Vec<f64>,
        starts: &mut Vec<i32>,
    ) -> Result<(), String>;

    /// Keyword field `field`'s term `ord`.
    fn term(&mut self, field: usize, ord: i64) -> Result<String, String>;
}

/// What a call returns for its document.
#[derive(Debug, Clone, PartialEq)]
pub enum Out {
    /// `null` or `undefined`.
    Null,
    Number(f64),
    Bool(bool),
    String(String),
    List(Vec<Out>),
    /// Anything else, by its type.
    Other(String),
}

/// Why a script did not run.
#[derive(Debug, Clone, PartialEq)]
pub enum Failure {
    /// It does not compile.
    Compile {
        message: String,
        line: Option<usize>,
        column: Option<usize>,
    },
    /// It threw on the call's document `at`.
    Thrown {
        at: usize,
        name: String,
        message: String,
        /// Where in the script, when it threw from its own code.
        line: Option<usize>,
        column: Option<usize>,
    },
    /// The search ran it past its deadline.
    TimedOut(Duration),
    /// Its isolate reached its heap limit.
    OutOfMemory(usize),
    /// The host failed reading a field.
    Host(String),
}

impl std::fmt::Display for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let at =
            |f: &mut std::fmt::Formatter<'_>, line: &Option<usize>, column: &Option<usize>| match (
                line, column,
            ) {
                (Some(l), Some(c)) => write!(f, " (line {l}, column {})", c + 1),
                (Some(l), None) => write!(f, " (line {l})"),
                _ => Ok(()),
            };
        match self {
            Failure::Compile {
                message,
                line,
                column,
            } => {
                write!(f, "compile error: {message}")?;
                at(f, line, column)
            }
            Failure::Thrown {
                name,
                message,
                line,
                column,
                ..
            } => {
                write!(f, "{name}: {message}")?;
                at(f, line, column)
            }
            Failure::TimedOut(t) => write!(
                f,
                "the js script ran past its time limit of [{}ms]",
                t.as_millis()
            ),
            Failure::OutOfMemory(n) => write!(
                f,
                "the js script ran past its heap limit of [{}mb]",
                n >> 20
            ),
            Failure::Host(m) => f.write_str(m),
        }
    }
}

/// Runs `script` on the call's `count` documents into `out`: numbers, or
/// with `filter` booleans as `1` and `0`. When the script throws on a
/// document, `out` holds the values of the documents before it. `scores` are the documents'
/// `_score`s; `segment` names the segment, whose keyword ordinals the
/// thread's term cache is keyed by.
#[allow(clippy::too_many_arguments)]
pub fn run_batch(
    script: &Script,
    params: &Params,
    segment: u64,
    count: usize,
    scores: Option<&[f32]>,
    filter: bool,
    fields: &mut dyn Fields,
    out: &mut [f64],
) -> Result<(), Failure> {
    assert!(out.len() >= count && scores.is_none_or(|s| s.len() >= count));
    if count == 0 {
        return Ok(());
    }
    with_engine(script.limits, |e| {
        e.call(script, params, fields, count, |tc, c, args| {
            let out_arr = c.out.ensure(tc, count);
            let scores_arr: v8::Local<v8::Value> = match scores {
                Some(s) => {
                    let a = c.scores.ensure(tc, count);
                    for (k, v) in s[..count].iter().enumerate() {
                        c.scores.set(k, f64::from(*v));
                    }
                    a
                }
                None => v8::null(tc).into(),
            };
            let batch = v8::Local::new(tc, &c.batch);
            let undefined = v8::undefined(tc).into();
            let n = v8::Number::new(tc, count as f64).into();
            let mode = v8::Number::new(tc, if filter { 1.0 } else { 0.0 }).into();
            let seg = v8::Number::new(tc, segment as f64).into();
            let r = batch.call(
                tc,
                undefined,
                &[n, out_arr, scores_arr, args.params, mode, seg],
            );
            // On a failure, the documents before the one that failed.
            let upto = match r {
                Some(_) => count,
                None => (c.state.get(0).max(0) as usize).min(count),
            };
            for (k, o) in out[..upto].iter_mut().enumerate() {
                *o = c.out.get(k);
            }
            r.map(|_| Out::Null)
        })
        .map(|_| ())
    })
}

/// Runs `script` on the call's one document, with `_score` `score`.
pub fn run_one(
    script: &Script,
    params: &Params,
    segment: u64,
    score: Option<f64>,
    fields: &mut dyn Fields,
) -> Result<Out, Failure> {
    with_engine(script.limits, |e| {
        e.call(script, params, fields, 1, |tc, c, args| {
            let one = v8::Local::new(tc, &c.one);
            let undefined = v8::undefined(tc).into();
            let s = match score {
                Some(s) => v8::Number::new(tc, s).into(),
                None => undefined,
            };
            let seg = v8::Number::new(tc, segment as f64).into();
            let r = one.call(tc, undefined, &[s, args.params, seg])?;
            Some(out(tc, r, 0))
        })
    })
}

/// Deepest array a result may nest.
const MAX_DEPTH: usize = 16;

fn out(scope: &mut v8::HandleScope<'_>, v: v8::Local<'_, v8::Value>, depth: usize) -> Out {
    if v.is_null_or_undefined() {
        Out::Null
    } else if v.is_number() {
        Out::Number(v.number_value(scope).unwrap_or(f64::NAN))
    } else if v.is_boolean() {
        Out::Bool(v.boolean_value(scope))
    } else if v.is_string() {
        Out::String(v.to_rust_string_lossy(scope))
    } else if v.is_array() && depth < MAX_DEPTH {
        let a: v8::Local<v8::Array> = v.try_into().expect("an array");
        let mut items = Vec::with_capacity(a.length() as usize);
        for i in 0..a.length() {
            let item = a
                .get_index(scope, i)
                .unwrap_or_else(|| v8::undefined(scope).into());
            items.push(out(scope, item, depth + 1));
        }
        Out::List(items)
    } else {
        Out::Other(v.type_of(scope).to_rust_string_lossy(scope))
    }
}

// ---------------------------------------------------------------------------
// The thread's isolate
// ---------------------------------------------------------------------------

/// V8's flags: a stack limit well inside a JVM thread's (OpenSearch runs
/// them at 1 MB, and the isolate's limit counts from where it is first
/// entered), and no WebAssembly.
const FLAGS: &str = "--stack-size=256 --no-expose-wasm";

fn initialize() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        v8::V8::set_flags_from_string(FLAGS);
        let platform = v8::new_default_platform(0, false).make_shared();
        v8::V8::initialize_platform(platform);
        v8::V8::initialize();
    });
}

/// Most scripts a thread keeps compiled; past this many it starts over.
const MAX_SCRIPTS: usize = 64;

/// Most params a compiled script keeps converted.
const MAX_PARAMS: usize = 16;

/// Keys that are already hashes, used as they are.
#[derive(Default, Clone, Copy)]
struct KeyHasher(u64);

impl Hasher for KeyHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.0 = self.0.rotate_left(8) ^ u64::from(*b);
        }
    }
    fn write_u64(&mut self, v: u64) {
        self.0 = v;
    }
}

type ByKey<V> = HashMap<u64, V, std::hash::BuildHasherDefault<KeyHasher>>;

thread_local! {
    static ENGINE: RefCell<Option<Engine>> = const { RefCell::new(None) };
}

fn with_engine<R>(
    limits: Limits,
    f: impl FnOnce(&mut Engine) -> Result<R, Failure>,
) -> Result<R, Failure> {
    ENGINE.with(|cell| {
        let mut slot = cell.borrow_mut();
        let engine = slot.get_or_insert_with(|| Engine::new(limits));
        let r = f(engine);
        if engine.heap.hit.load(Ordering::Relaxed) {
            // Its heap is past the limit: the next call starts a new isolate.
            *slot = None;
        }
        r
    })
}

/// What the heap-limit callback reads.
struct Heap {
    handle: v8::IsolateHandle,
    hit: AtomicBool,
    max: usize,
}

extern "C" fn near_heap_limit(data: *mut c_void, current: usize, _initial: usize) -> usize {
    // SAFETY: `data` is the engine's boxed `Heap`, which outlives its
    // isolate (`Engine` drops the isolate first).
    let heap = unsafe { &*(data as *const Heap) };
    heap.hit.store(true, Ordering::Relaxed);
    heap.handle.terminate_execution();
    // Room to unwind the script; the isolate is dropped after.
    current.saturating_add(32 << 20)
}

struct Engine {
    // Dropped before the isolate their handles belong to.
    scripts: ByKey<Compiled>,
    isolate: v8::OwnedIsolate,
    heap: Box<Heap>,
    watch: Arc<Watch>,
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.watch.retire();
    }
}

/// A script compiled in its own context.
struct Compiled {
    source: Arc<str>,
    fields: Vec<(String, Kind)>,
    context: v8::Global<v8::Context>,
    batch: v8::Global<v8::Function>,
    one: v8::Global<v8::Function>,
    state: Floats<i32>,
    slots: Vec<Slot>,
    out: Floats<f64>,
    scores: Floats<f64>,
    params: ByKey<(Arc<str>, v8::Global<v8::Value>)>,
}

/// A field's column in the call, as the script reads it.
struct Slot {
    values: Floats<f64>,
    starts: Floats<i32>,
    pair: Option<v8::Global<v8::Array>>,
}

/// A typed array the thread keeps, and its memory.
struct Floats<T> {
    array: Option<v8::Global<v8::Value>>,
    store: Option<v8::SharedRef<v8::BackingStore>>,
    len: usize,
    _t: std::marker::PhantomData<T>,
}

trait Element: Copy {
    fn array<'s>(
        scope: &mut v8::HandleScope<'s>,
        buf: v8::Local<'s, v8::ArrayBuffer>,
        len: usize,
    ) -> v8::Local<'s, v8::Value>;
}

impl Element for f64 {
    fn array<'s>(
        scope: &mut v8::HandleScope<'s>,
        buf: v8::Local<'s, v8::ArrayBuffer>,
        len: usize,
    ) -> v8::Local<'s, v8::Value> {
        v8::Float64Array::new(scope, buf, 0, len)
            .expect("a Float64Array")
            .into()
    }
}

impl Element for i32 {
    fn array<'s>(
        scope: &mut v8::HandleScope<'s>,
        buf: v8::Local<'s, v8::ArrayBuffer>,
        len: usize,
    ) -> v8::Local<'s, v8::Value> {
        v8::Int32Array::new(scope, buf, 0, len)
            .expect("an Int32Array")
            .into()
    }
}

impl<T: Element> Floats<T> {
    fn new() -> Self {
        Floats {
            array: None,
            store: None,
            len: 0,
            _t: std::marker::PhantomData,
        }
    }

    /// The array, grown to hold `n`; whether it is a new one.
    fn grow<'s>(
        &mut self,
        scope: &mut v8::HandleScope<'s>,
        n: usize,
    ) -> (v8::Local<'s, v8::Value>, bool) {
        if let (Some(a), true) = (&self.array, n <= self.len) {
            return (v8::Local::new(scope, a), false);
        }
        let len = n.max(self.len.saturating_mul(2)).max(16);
        let buf = v8::ArrayBuffer::new(scope, len * std::mem::size_of::<T>());
        self.store = Some(buf.get_backing_store());
        let a = T::array(scope, buf, len);
        self.array = Some(v8::Global::new(scope, a));
        self.len = len;
        (a, true)
    }

    fn ensure<'s>(
        &mut self,
        scope: &mut v8::HandleScope<'s>,
        n: usize,
    ) -> v8::Local<'s, v8::Value> {
        self.grow(scope, n).0
    }

    fn ptr(&self) -> *mut T {
        let store = self.store.as_ref().expect("an array");
        store
            .data()
            .map_or(NonNull::dangling(), |p| p.cast::<T>())
            .as_ptr()
    }

    fn set(&self, k: usize, v: T) {
        assert!(k < self.len);
        // SAFETY: `k` is inside the array's backing store, which `store`
        // keeps alive and V8 aligns for any element type; no script runs
        // while the host writes it.
        unsafe { self.ptr().add(k).write(v) }
    }

    fn get(&self, k: usize) -> T {
        assert!(k < self.len);
        // SAFETY: as `set`.
        unsafe { self.ptr().add(k).read() }
    }

    fn copy_from(&self, src: &[T]) {
        assert!(src.len() <= self.len);
        // SAFETY: as `set`, for `src.len()` elements.
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), self.ptr(), src.len()) }
    }
}

/// What a call hands its body.
struct Args<'s> {
    params: v8::Local<'s, v8::Value>,
}

/// The call in progress on this thread, which the `load` and `term`
/// callbacks read.
struct Call {
    fields: *mut (dyn Fields + 'static),
    slots: *mut Vec<Slot>,
    docs: usize,
    values: Vec<f64>,
    starts: Vec<i32>,
    error: Option<String>,
}

thread_local! {
    static CALL: Cell<*mut Call> = const { Cell::new(std::ptr::null_mut()) };
}

/// Sets this thread's call for as long as it lives.
struct Entered(*mut Call);

impl Entered {
    fn new(call: &mut Call) -> Self {
        let prev = CALL.with(|c| c.replace(call));
        Entered(prev)
    }
}

impl Drop for Entered {
    fn drop(&mut self) {
        CALL.with(|c| c.set(self.0));
    }
}

fn throw(scope: &mut v8::HandleScope<'_>, message: &str) {
    let m = v8::String::new(scope, message).expect("a message");
    let e = v8::Exception::error(scope, m);
    scope.throw_exception(e);
}

/// `load(f)`: field `f`'s column for the call, as `[values, starts]`.
fn load_callback(
    scope: &mut v8::HandleScope<'_>,
    args: v8::FunctionCallbackArguments<'_>,
    mut rv: v8::ReturnValue<'_>,
) {
    let call = CALL.with(Cell::get);
    if call.is_null() {
        return throw(scope, "no call is running");
    }
    // SAFETY: `call` is the `Call` an `Entered` guard set for the duration
    // of the V8 call this callback runs inside, and nothing else holds it.
    let call = unsafe { &mut *call };
    let f = args.get(0).int32_value(scope).unwrap_or(-1);
    // SAFETY: as `call`: the slots of the compiled script running.
    let slots = unsafe { &mut *call.slots };
    let Some(slot) = usize::try_from(f).ok().and_then(|f| slots.get_mut(f)) else {
        return throw(scope, "no such field");
    };
    call.values.clear();
    call.starts.clear();
    // SAFETY: as `call`: the host's fields, borrowed for the call.
    let fields = unsafe { &mut *call.fields };
    if let Err(e) = fields.load(f as usize, &mut call.values, &mut call.starts) {
        call.error = Some(e.clone());
        return throw(scope, &e);
    }
    if call.starts.len() != call.docs + 1 {
        let e = format!(
            "field {f}: {} starts for {} documents",
            call.starts.len(),
            call.docs
        );
        call.error = Some(e.clone());
        return throw(scope, &e);
    }
    let (values, new_values) = slot.values.grow(scope, call.values.len());
    let (starts, new_starts) = slot.starts.grow(scope, call.starts.len());
    slot.values.copy_from(&call.values);
    slot.starts.copy_from(&call.starts);
    if new_values || new_starts || slot.pair.is_none() {
        let pair = v8::Array::new_with_elements(scope, &[values, starts]);
        slot.pair = Some(v8::Global::new(scope, pair));
    }
    let pair = v8::Local::new(scope, slot.pair.as_ref().expect("set above"));
    rv.set(pair.into());
}

/// `term(f, ord)`: keyword field `f`'s term `ord`.
fn term_callback(
    scope: &mut v8::HandleScope<'_>,
    args: v8::FunctionCallbackArguments<'_>,
    mut rv: v8::ReturnValue<'_>,
) {
    let call = CALL.with(Cell::get);
    if call.is_null() {
        return throw(scope, "no call is running");
    }
    // SAFETY: as in `load_callback`.
    let call = unsafe { &mut *call };
    let f = args.get(0).int32_value(scope).unwrap_or(-1);
    let ord = args.get(1).number_value(scope).unwrap_or(-1.0) as i64;
    let Ok(f) = usize::try_from(f) else {
        return throw(scope, "no such field");
    };
    // SAFETY: as in `load_callback`.
    let fields = unsafe { &mut *call.fields };
    match fields.term(f, ord) {
        Ok(t) => match v8::String::new(scope, &t) {
            Some(s) => rv.set(s.into()),
            None => throw(scope, "a term too long for a js string"),
        },
        Err(e) => {
            call.error = Some(e.clone());
            throw(scope, &e)
        }
    }
}

/// The name of the script's own code in V8's messages.
const RESOURCE: &str = "script";

impl Engine {
    fn new(limits: Limits) -> Self {
        initialize();
        let params = v8::CreateParams::default().heap_limits(0, limits.max_heap);
        let mut isolate = v8::Isolate::new(params);
        isolate.set_microtasks_policy(v8::MicrotasksPolicy::Explicit);
        // A message's stack, for where in the script an error the runner
        // raises comes from.
        isolate.set_capture_stack_trace_for_uncaught_exceptions(true, 16);
        let heap = Box::new(Heap {
            handle: isolate.thread_safe_handle(),
            hit: AtomicBool::new(false),
            max: limits.max_heap,
        });
        isolate.add_near_heap_limit_callback(near_heap_limit, &*heap as *const Heap as *mut c_void);
        let watch = Watch::new(isolate.thread_safe_handle());
        Engine {
            scripts: ByKey::default(),
            isolate,
            heap,
            watch,
        }
    }

    /// `script` compiled on this thread.
    fn compiled(&mut self, script: &Script) -> Result<&mut Compiled, Failure> {
        let hit = self
            .scripts
            .get(&script.key)
            .is_some_and(|c| c.source == script.source && c.fields == script.fields);
        if !hit {
            let c = compile(&mut self.isolate, script)?;
            if self.scripts.len() >= MAX_SCRIPTS {
                self.scripts.clear();
            }
            self.scripts.insert(script.key, c);
        }
        Ok(self.scripts.get_mut(&script.key).expect("inserted"))
    }

    /// A call of `script` on `docs` documents: `body` runs it.
    fn call(
        &mut self,
        script: &Script,
        params: &Params,
        fields: &mut dyn Fields,
        docs: usize,
        body: impl FnOnce(
            &mut v8::TryCatch<'_, v8::HandleScope<'_>>,
            &mut Compiled,
            Args<'_>,
        ) -> Option<Out>,
    ) -> Result<Out, Failure> {
        let (at, deadline) = script.deadline();
        if Instant::now() >= at {
            return Err(Failure::TimedOut(script.limits.timeout));
        }
        let max_heap = self.heap.max;
        self.compiled(script)?;
        let Engine {
            scripts,
            isolate,
            heap,
            watch,
        } = self;
        let c = scripts.get_mut(&script.key).expect("compiled above");
        let scope = &mut v8::HandleScope::new(isolate);
        let context = v8::Local::new(scope, &c.context);
        let scope = &mut v8::ContextScope::new(scope, context);
        let tc = &mut v8::TryCatch::new(scope);
        // A stop the watchdog asked for after the last call ended.
        if watch.fired.swap(false, Ordering::AcqRel) {
            tc.cancel_terminate_execution();
        }
        let params_v = match c.params.get(&params.key) {
            Some((json, v)) if *json == params.json => v8::Local::new(tc, v),
            _ => {
                let v = parse_params(tc, &params.json)?;
                if c.params.len() >= MAX_PARAMS {
                    c.params.clear();
                }
                c.params
                    .insert(params.key, (params.json.clone(), v8::Global::new(tc, v)));
                v
            }
        };
        // SAFETY: the `'static` is a lie the `Entered` guard keeps: the
        // pointer is cleared before `fields`'s borrow ends, and only this
        // call's callbacks read it.
        let fields: *mut (dyn Fields + 'static) = unsafe { std::mem::transmute(fields) };
        let mut call = Call {
            fields,
            slots: &mut c.slots,
            docs,
            values: Vec::new(),
            starts: Vec::new(),
            error: None,
        };
        let r = {
            let _entered = Entered::new(&mut call);
            let _watch = watch.call(deadline);
            body(tc, c, Args { params: params_v })
        };
        if tc.is_execution_terminating() || tc.has_terminated() {
            tc.cancel_terminate_execution();
        }
        match r {
            Some(out) => Ok(out),
            None => {
                if let Some(e) = call.error {
                    return Err(Failure::Host(e));
                }
                if tc.has_terminated() || tc.exception().is_none() {
                    // The heap callback or the watchdog stopped it.
                    return Err(if heap.hit.load(Ordering::Relaxed) {
                        Failure::OutOfMemory(max_heap)
                    } else {
                        Failure::TimedOut(script.limits.timeout)
                    });
                }
                let at = c.state.get(0).max(0) as usize;
                Err(thrown(tc, at))
            }
        }
    }
}

fn parse_params<'s>(
    tc: &mut v8::TryCatch<'s, v8::HandleScope<'_>>,
    json: &str,
) -> Result<v8::Local<'s, v8::Value>, Failure> {
    let s = v8::String::new(tc, json).ok_or_else(|| Failure::Host("params too long".into()))?;
    let v = v8::json::parse(tc, s).ok_or_else(|| Failure::Host("params: not JSON".into()))?;
    freeze(tc, v, 0);
    Ok(v)
}

fn freeze(scope: &mut v8::HandleScope<'_>, v: v8::Local<'_, v8::Value>, depth: usize) {
    if depth > 64 {
        return;
    }
    let Ok(o) = v8::Local::<v8::Object>::try_from(v) else {
        return;
    };
    if let Some(names) = o.get_own_property_names(scope, Default::default()) {
        for i in 0..names.length() {
            let Some(k) = names.get_index(scope, i) else {
                continue;
            };
            if let Some(child) = o.get(scope, k) {
                freeze(scope, child, depth + 1);
            }
        }
    }
    o.set_integrity_level(scope, v8::IntegrityLevel::Frozen);
}

fn thrown(tc: &mut v8::TryCatch<'_, v8::HandleScope<'_>>, at: usize) -> Failure {
    let exception = tc.exception();
    let (mut name, mut message) = ("Error".to_string(), String::new());
    if let Some(e) = exception {
        match v8::Local::<v8::Object>::try_from(e) {
            Ok(o) if e.is_native_error() => {
                let n = v8::String::new(tc, "name").expect("a key");
                if let Some(v) = o.get(tc, n.into()) {
                    name = v.to_rust_string_lossy(tc);
                }
                let m = v8::String::new(tc, "message").expect("a key");
                if let Some(v) = o.get(tc, m.into()) {
                    message = v.to_rust_string_lossy(tc);
                }
            }
            _ => {
                name = "Thrown".to_string();
                message = e.to_rust_string_lossy(tc);
            }
        }
    }
    let (line, column) = position(tc);
    Failure::Thrown {
        at,
        name,
        message,
        line,
        column,
    }
}

/// Where the caught exception was thrown in the script's own code: where
/// it was raised, or else the innermost of its stack's frames in the script
/// (an error the runner raises for what a script read or returned). Lines
/// count from 1, columns from 0.
fn position(tc: &mut v8::TryCatch<'_, v8::HandleScope<'_>>) -> (Option<usize>, Option<usize>) {
    let Some(m) = tc.message() else {
        return (None, None);
    };
    let own = |tc: &mut v8::TryCatch<'_, v8::HandleScope<'_>>,
               name: Option<v8::Local<'_, v8::Value>>| {
        name.is_some_and(|n| n.to_rust_string_lossy(tc) == RESOURCE)
    };
    let name = m.get_script_resource_name(tc);
    if own(tc, name) {
        return (m.get_line_number(tc), Some(m.get_start_column()));
    }
    let Some(stack) = m.get_stack_trace(tc) else {
        return (None, None);
    };
    for i in 0..stack.get_frame_count() {
        let Some(frame) = stack.get_frame(tc, i) else {
            continue;
        };
        let name = frame.get_script_name(tc).map(Into::into);
        if own(tc, name) {
            return (
                Some(frame.get_line_number()),
                Some(frame.get_column().saturating_sub(1)),
            );
        }
    }
    (None, None)
}

/// The script's function: an expression's value, or a body that returns.
fn compile(isolate: &mut v8::OwnedIsolate, script: &Script) -> Result<Compiled, Failure> {
    let scope = &mut v8::HandleScope::new(isolate);
    let context = v8::Context::new(scope, Default::default());
    context.set_allow_generation_from_strings(false);
    let scope = &mut v8::ContextScope::new(scope, context);
    let tc = &mut v8::TryCatch::new(scope);

    let prelude = v8::String::new(tc, PRELUDE).expect("the prelude");
    let factory = v8::Script::compile(tc, prelude, None)
        .and_then(|s| s.run(tc))
        .and_then(|v| v8::Local::<v8::Function>::try_from(v).ok())
        .expect("the prelude runs");

    let user = user_function(tc, &script.source)?;

    let names: Vec<v8::Local<v8::Value>> = script
        .fields
        .iter()
        .map(|(n, _)| v8::String::new(tc, n).expect("a field name").into())
        .collect();
    let names = v8::Array::new_with_elements(tc, &names);
    let kinds: Vec<v8::Local<v8::Value>> = script
        .fields
        .iter()
        .map(|(_, k)| v8::Integer::new(tc, k.code()).into())
        .collect();
    let kinds = v8::Array::new_with_elements(tc, &kinds);
    let mut state = Floats::<i32>::new();
    let state_arr = state.ensure(tc, 1);
    let load = v8::Function::new(tc, load_callback).expect("load");
    let term = v8::Function::new(tc, term_callback).expect("term");
    let undefined = v8::undefined(tc).into();
    let api = factory
        .call(
            tc,
            undefined,
            &[
                load.into(),
                term.into(),
                user.into(),
                names.into(),
                kinds.into(),
                state_arr,
            ],
        )
        .and_then(|v| v8::Local::<v8::Object>::try_from(v).ok())
        .expect("the prelude builds a runner");
    let get = |tc: &mut v8::TryCatch<'_, v8::HandleScope<'_>>, name: &str| {
        let k = v8::String::new(tc, name).expect("a key");
        let f = api
            .get(tc, k.into())
            .and_then(|v| v8::Local::<v8::Function>::try_from(v).ok())
            .expect("a runner function");
        v8::Global::new(tc, f)
    };
    let batch = get(tc, "batch");
    let one = get(tc, "one");
    Ok(Compiled {
        source: script.source.clone(),
        fields: script.fields.clone(),
        context: v8::Global::new(tc, context),
        batch,
        one,
        state,
        slots: (0..script.fields.len())
            .map(|_| Slot {
                values: Floats::new(),
                starts: Floats::new(),
                pair: None,
            })
            .collect(),
        out: Floats::new(),
        scores: Floats::new(),
        params: ByKey::default(),
    })
}

fn user_function<'s>(
    tc: &mut v8::TryCatch<'s, v8::HandleScope<'_>>,
    source: &str,
) -> Result<v8::Local<'s, v8::Function>, Failure> {
    // The script's line 1 is the body's line 2, after the strict directive.
    let expression = format!("\"use strict\"; return (\n{source}\n);");
    if let Some(f) = compile_body(tc, &expression) {
        return Ok(f);
    }
    tc.reset();
    let body = format!("\"use strict\";\n{source}");
    if let Some(f) = compile_body(tc, &body) {
        return Ok(f);
    }
    let message = tc
        .message()
        .map(|m| {
            let m = m.get(tc).to_rust_string_lossy(tc);
            m.strip_prefix("Uncaught ").map(str::to_string).unwrap_or(m)
        })
        .unwrap_or_else(|| "the script does not compile".to_string());
    let (line, column) = position(tc);
    Err(Failure::Compile {
        message,
        line,
        column,
    })
}

fn compile_body<'s>(
    tc: &mut v8::TryCatch<'s, v8::HandleScope<'_>>,
    body: &str,
) -> Option<v8::Local<'s, v8::Function>> {
    let src = v8::String::new(tc, body)?;
    let name = v8::String::new(tc, RESOURCE)?;
    let origin = v8::ScriptOrigin::new(
        tc,
        name.into(),
        -1,
        0,
        false,
        0,
        None,
        false,
        false,
        false,
        None,
    );
    let mut source = v8::script_compiler::Source::new(src, Some(&origin));
    let args = [
        v8::String::new(tc, "doc")?,
        v8::String::new(tc, "params")?,
        v8::String::new(tc, "_score")?,
    ];
    v8::script_compiler::compile_function(
        tc,
        &mut source,
        &args,
        &[],
        v8::script_compiler::CompileOptions::NoCompileOptions,
        v8::script_compiler::NoCacheReason::NoReason,
    )
}

// ---------------------------------------------------------------------------
// Deadlines
// ---------------------------------------------------------------------------

/// How often the watchdog looks at the calls running.
const TICK: Duration = Duration::from_millis(5);

/// An isolate's call in progress, as the watchdog sees it: its deadline in
/// nanoseconds after [`EPOCH`], or `0` when none runs.
struct Watch {
    handle: v8::IsolateHandle,
    deadline: std::sync::atomic::AtomicU64,
    /// Whether the watchdog stopped a call, which the next call clears.
    fired: AtomicBool,
}

/// Stops each isolate's call at its deadline.
struct Watchdog {
    watches: Mutex<Vec<Arc<Watch>>>,
    /// Whether the watchdog sleeps until a call starts.
    idle: AtomicBool,
    thread: OnceLock<std::thread::Thread>,
}

static EPOCH: OnceLock<Instant> = OnceLock::new();

fn nanos(t: Instant) -> u64 {
    let e = *EPOCH.get_or_init(Instant::now);
    u64::try_from(t.saturating_duration_since(e).as_nanos())
        .unwrap_or(u64::MAX)
        .max(1)
}

fn watchdog() -> &'static Watchdog {
    static WATCHDOG: OnceLock<Watchdog> = OnceLock::new();
    let w = WATCHDOG.get_or_init(|| Watchdog {
        watches: Mutex::new(Vec::new()),
        idle: AtomicBool::new(false),
        thread: OnceLock::new(),
    });
    w.thread.get_or_init(|| {
        std::thread::Builder::new()
            .name("js-script-watchdog".into())
            .spawn(|| watchdog().run())
            .expect("the js watchdog thread")
            .thread()
            .clone()
    });
    w
}

/// A call the watchdog watches until dropped.
struct Watched<'w>(&'w Watch);

impl Drop for Watched<'_> {
    fn drop(&mut self) {
        self.0.deadline.store(0, Ordering::Release);
    }
}

impl Watch {
    fn new(handle: v8::IsolateHandle) -> Arc<Self> {
        let w = Arc::new(Watch {
            handle,
            deadline: std::sync::atomic::AtomicU64::new(0),
            fired: AtomicBool::new(false),
        });
        let dog = watchdog();
        dog.watches
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push(Arc::clone(&w));
        w
    }

    /// Watches a call until `deadline` (the watchdog's nanoseconds).
    fn call(&self, deadline: u64) -> Watched<'_> {
        self.deadline.store(deadline, Ordering::Release);
        let dog = watchdog();
        if dog.idle.swap(false, Ordering::AcqRel) {
            if let Some(t) = dog.thread.get() {
                t.unpark();
            }
        }
        Watched(self)
    }

    /// No longer watched: its isolate is gone.
    fn retire(self: &Arc<Self>) {
        let dog = watchdog();
        dog.watches
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .retain(|w| !Arc::ptr_eq(w, self));
    }
}

impl Watchdog {
    fn run(&self) {
        loop {
            let now = nanos(Instant::now());
            let mut running = false;
            {
                let watches = self.watches.lock().unwrap_or_else(|p| p.into_inner());
                for w in watches.iter() {
                    let d = w.deadline.load(Ordering::Acquire);
                    if d == 0 {
                        continue;
                    }
                    running = true;
                    if d <= now {
                        w.fired.store(true, Ordering::Release);
                        w.handle.terminate_execution();
                    }
                }
            }
            if running {
                std::thread::sleep(TICK);
            } else {
                self.idle.store(true, Ordering::Release);
                // A call that started before `idle` was set is seen next time.
                std::thread::park_timeout(Duration::from_secs(1));
            }
        }
    }
}

#[cfg(test)]
mod tests;
