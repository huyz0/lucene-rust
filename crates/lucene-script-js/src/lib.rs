//! Search scripts in JavaScript (`lang: "js"`), run on V8.
//!
//! A script is an expression, or a function body that `return`s, over the
//! same names a Painless script reads: `doc['f'].value`, `doc['f'].size()`,
//! `doc['f'].get(i)`, `doc['f'].empty`, `params`, and `_score`. It runs in
//! strict mode, in a context of its own where everything a script can reach
//! is frozen: no script changes what another document or request sees. V8
//! itself has no files, network, timers or modules, and code generation from
//! strings (`eval`, `Function`) is off, as is WebAssembly; typed arrays,
//! whose memory V8's heap limit does not count, are not there either. (The
//! locale-taking string and number methods still reach ICU, whose objects V8
//! caches per locale.) As everything is frozen, assigning a property a
//! built-in prototype holds (`e.name = ...` on an error) throws in strict
//! mode: a script defines it instead (`Object.defineProperty`).
//!
//! Each thread keeps one isolate, with every script it has compiled; a
//! script compiles once per thread. A search's calls share one deadline
//! ([`Limits::timeout`]), past which V8 stops the script; an isolate past its
//! heap limit ([`Limits::max_heap`]) stops the script and is dropped.
//!
//! Field values cross into V8 a column at a time, into typed arrays the
//! thread reuses: the host ([`Fields::load`]) loads a field's values in
//! every document of the call before the script runs, once a call of the
//! script has read it. A read of a field the call has not loaded stops the
//! call, which runs that document again once the host has loaded it.

#![forbid(unsafe_code)]

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::ffi::c_void;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
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
/// document, `out` holds the values of the documents before it.
/// `scores` are the documents' `_score`s; `segment` names the segment,
/// whose keyword ordinals the thread's term cache is keyed by.
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
        e.call(script, params, segment, fields, count, |tc, c, args| {
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
            let from = v8::Number::new(tc, args.from as f64).into();
            let n = v8::Number::new(tc, count as f64).into();
            let mode = v8::Number::new(tc, if filter { 1.0 } else { 0.0 }).into();
            let r = batch.call(
                tc,
                undefined,
                &[from, n, out_arr, scores_arr, args.params, mode],
            );
            // Else the documents before the one that stopped the call.
            let faulted = NEEDS.with(|n| !n.borrow().is_empty());
            let upto = match r {
                Some(_) if !faulted => count,
                _ => (c.state.get(0).max(0) as usize).min(count),
            };
            for (k, o) in out.iter_mut().enumerate().take(upto).skip(args.from) {
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
        e.call(script, params, segment, fields, 1, |tc, c, args| {
            let one = v8::Local::new(tc, &c.one);
            let undefined = v8::undefined(tc).into();
            let s = match score {
                Some(s) => v8::Number::new(tc, s).into(),
                None => undefined,
            };
            let r = one.call(tc, undefined, &[s, args.params])?;
            Some(result(tc, r))
        })
    })
}

/// Deepest array a result may nest.
const MAX_DEPTH: usize = 16;

/// Most values a result holds, over all its arrays.
const MAX_VALUES: usize = 1 << 20;

/// What the script returned, as the host reads it.
fn result(scope: &mut v8::HandleScope<'_>, v: v8::Local<'_, v8::Value>) -> Out {
    let mut left = MAX_VALUES;
    out(scope, v, 0, &mut left)
}

fn out(
    scope: &mut v8::HandleScope<'_>,
    v: v8::Local<'_, v8::Value>,
    depth: usize,
    left: &mut usize,
) -> Out {
    if *left == 0 {
        return Out::Other(format!("a result of more than {MAX_VALUES} values"));
    }
    *left -= 1;
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
        let len = a.length() as usize;
        if len > *left {
            *left = 0;
            return Out::Other(format!("a result of more than {MAX_VALUES} values"));
        }
        let mut items = Vec::new();
        for i in 0..a.length() {
            // None once the call is stopped.
            let Some(item) = a.get_index(scope, i) else {
                break;
            };
            match out(scope, item, depth + 1, left) {
                Out::Other(t) if *left == 0 => return Out::Other(t),
                o => items.push(o),
            }
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
        // Without thread-isolated allocations (memory protection keys), which
        // need every thread that runs V8 to descend from the one that
        // initialised it: the JVM's search threads do not.
        let platform = v8::new_unprotected_default_platform(0, false).make_shared();
        v8::V8::initialize_platform(platform);
        v8::V8::initialize();
    });
}

/// Most scripts a thread keeps compiled; past this many it starts over.
const MAX_SCRIPTS: usize = 64;

/// Most params a compiled script keeps converted.
const MAX_PARAMS: usize = 16;

/// Most terms a keyword field keeps converted for its segment.
const MAX_TERMS: usize = 4096;

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
    fn write_i64(&mut self, v: i64) {
        self.0 = v as u64;
    }
}

type ByKey<V> = HashMap<u64, V, std::hash::BuildHasherDefault<KeyHasher>>;

thread_local! {
    static ENGINE: RefCell<Option<Engine>> = const { RefCell::new(None) };
    /// The thread's isolate, for the heap-limit callback to stop.
    static HEAP: RefCell<Option<v8::IsolateHandle>> = const { RefCell::new(None) };
    /// Whether the thread's isolate reached its heap limit.
    static HEAP_HIT: Cell<bool> = const { Cell::new(false) };
    /// The thread's isolate's heap limit.
    static HEAP_MAX: Cell<usize> = const { Cell::new(usize::MAX) };
    /// The thread's isolate's watch, for the heap check to reach.
    static WATCH: RefCell<Option<Arc<Watch>>> = const { RefCell::new(None) };
    /// The fields the running call asked for (`need`).
    static NEEDS: RefCell<Vec<i32>> = const { RefCell::new(Vec::new()) };
}

fn with_engine<R>(
    limits: Limits,
    f: impl FnOnce(&mut Engine) -> Result<R, Failure>,
) -> Result<R, Failure> {
    ENGINE.with(|cell| {
        let mut slot = cell.borrow_mut();
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let engine = slot.get_or_insert_with(|| Engine::new(limits));
            f(engine)
        }));
        let r = match r {
            Ok(r) => r,
            Err(panic) => {
                // The isolate's state is unknown: the next call starts anew.
                *slot = None;
                let why = panic
                    .downcast_ref::<&str>()
                    .map(|s| s.to_string())
                    .or_else(|| panic.downcast_ref::<String>().cloned())
                    .unwrap_or_default();
                return Err(Failure::Host(format!("the js runtime failed: {why}")));
            }
        };
        if HEAP_HIT.with(Cell::get) {
            // Its heap is past the limit: the next call starts a new isolate.
            *slot = None;
        }
        r
    })
}

/// V8 is near the heap limit: the script stops, and the isolate is dropped
/// after. The room it gets the first time covers the largest object V8 makes
/// at once (a string or array of about 1 GB), which would otherwise end the
/// process: a thread's isolate holds at most its limit and that.
extern "C" fn near_heap_limit(_data: *mut c_void, current: usize, _initial: usize) -> usize {
    let again = HEAP_HIT.try_with(|h| h.replace(true)).unwrap_or(true);
    let _ = HEAP.try_with(|h| {
        if let Ok(h) = h.try_borrow() {
            if let Some(h) = h.as_ref() {
                h.terminate_execution();
            }
        }
    });
    // Once: past that, only room for the script to unwind.
    if again {
        current.saturating_add(16 << 20)
    } else {
        current.saturating_add(current.max(1 << 30))
    }
}

/// The watchdog's look at a running script's memory: V8 lets one large
/// object (a long string, say) past its heap limit without asking, so a
/// script that keeps making them stops here.
extern "C" fn heap_check(isolate: &mut v8::Isolate, _data: *mut c_void) {
    let mut s = v8::HeapStatistics::default();
    isolate.get_heap_statistics(&mut s);
    let max = HEAP_MAX.try_with(Cell::get).unwrap_or(usize::MAX);
    // On the heap only: scripts make no typed arrays, and the host's columns
    // are outside it.
    if s.used_heap_size() > max {
        let _ = HEAP_HIT.try_with(|h| h.set(true));
        // A stop asked for inside an interrupt waits for the next one: the
        // watchdog asks.
        let _ = WATCH.try_with(|w| {
            if let Ok(w) = w.try_borrow() {
                if let Some(w) = w.as_ref() {
                    w.over.store(true, Ordering::Release);
                }
            }
        });
    }
}

struct Engine {
    // Dropped before the isolate their handles belong to.
    scripts: ByKey<Compiled>,
    isolate: v8::OwnedIsolate,
    max_heap: usize,
    watch: Arc<Watch>,
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.watch.retire();
        let _ = HEAP.try_with(|h| h.replace(None));
        let _ = WATCH.try_with(|w| w.replace(None));
    }
}

/// A script compiled in its own context.
struct Compiled {
    source: Arc<str>,
    fields: Vec<(String, Kind)>,
    context: v8::Global<v8::Context>,
    batch: v8::Global<v8::Function>,
    one: v8::Global<v8::Function>,
    harden: v8::Global<v8::Function>,
    /// The script's columns as it reads them (`cols` in the prelude).
    cols: v8::Global<v8::Array>,
    state: Floats<i32>,
    slots: Vec<Slot>,
    out: Floats<f64>,
    scores: Floats<f64>,
    params: ByKey<(Arc<str>, v8::Global<v8::Value>)>,
    /// What the host loads a column into.
    values: Vec<f64>,
    starts: Vec<i32>,
}

/// A field's column in the call, as the script reads it.
struct Slot {
    /// Whether a call has read the field: each call loads it up front.
    used: bool,
    /// Whether this call loaded it.
    loaded: bool,
    values: Floats<f64>,
    starts: Floats<i32>,
    /// `[values, starts]`, or with a keyword's terms.
    pair: Option<v8::Global<v8::Array>>,
    /// A keyword's terms by ordinal, for segment `segment`.
    terms: HashMap<i64, v8::Global<v8::String>>,
    segment: u64,
}

/// A typed array the thread keeps, and its memory.
struct Floats<T> {
    array: Option<v8::Global<v8::Value>>,
    store: Option<v8::SharedRef<v8::BackingStore>>,
    len: usize,
    _t: std::marker::PhantomData<T>,
}

trait Element: Copy {
    const SIZE: usize;
    fn array<'s>(
        scope: &mut v8::HandleScope<'s>,
        buf: v8::Local<'s, v8::ArrayBuffer>,
        len: usize,
    ) -> v8::Local<'s, v8::Value>;
    fn write(self, to: &[Cell<u8>]);
    fn read(from: &[Cell<u8>]) -> Self;
}

impl Element for f64 {
    const SIZE: usize = 8;
    fn array<'s>(
        scope: &mut v8::HandleScope<'s>,
        buf: v8::Local<'s, v8::ArrayBuffer>,
        len: usize,
    ) -> v8::Local<'s, v8::Value> {
        v8::Float64Array::new(scope, buf, 0, len)
            .expect("a Float64Array")
            .into()
    }
    fn write(self, to: &[Cell<u8>]) {
        for (c, b) in to.iter().zip(self.to_ne_bytes()) {
            c.set(b);
        }
    }
    fn read(from: &[Cell<u8>]) -> Self {
        let mut b = [0u8; 8];
        for (b, c) in b.iter_mut().zip(from) {
            *b = c.get();
        }
        f64::from_ne_bytes(b)
    }
}

impl Element for i32 {
    const SIZE: usize = 4;
    fn array<'s>(
        scope: &mut v8::HandleScope<'s>,
        buf: v8::Local<'s, v8::ArrayBuffer>,
        len: usize,
    ) -> v8::Local<'s, v8::Value> {
        v8::Int32Array::new(scope, buf, 0, len)
            .expect("an Int32Array")
            .into()
    }
    fn write(self, to: &[Cell<u8>]) {
        for (c, b) in to.iter().zip(self.to_ne_bytes()) {
            c.set(b);
        }
    }
    fn read(from: &[Cell<u8>]) -> Self {
        let mut b = [0u8; 4];
        for (b, c) in b.iter_mut().zip(from) {
            *b = c.get();
        }
        i32::from_ne_bytes(b)
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
        let buf = v8::ArrayBuffer::new(scope, len * T::SIZE);
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

    fn bytes(&self) -> &[Cell<u8>] {
        self.store.as_ref().map_or(&[], |s| &s[..])
    }

    fn set(&self, k: usize, v: T) {
        v.write(&self.bytes()[k * T::SIZE..(k + 1) * T::SIZE]);
    }

    fn get(&self, k: usize) -> T {
        T::read(&self.bytes()[k * T::SIZE..(k + 1) * T::SIZE])
    }

    fn copy_from(&self, src: &[T]) {
        let to = &self.bytes()[..src.len() * T::SIZE];
        for (v, to) in src.iter().zip(to.chunks_exact(T::SIZE)) {
            v.write(to);
        }
    }
}

/// What a call hands its body.
struct Args<'s> {
    params: v8::Local<'s, v8::Value>,
    /// The first document to run: past those an earlier try finished.
    from: usize,
}

/// `need(f)`: the script reads field `f`, which this call has not loaded:
/// the call stops, and runs again once the host loads it.
fn need_callback(
    scope: &mut v8::HandleScope<'_>,
    args: v8::FunctionCallbackArguments<'_>,
    _rv: v8::ReturnValue<'_>,
) {
    let f = args.get(0).int32_value(scope).unwrap_or(-1);
    let _ = NEEDS.try_with(|n| {
        if let Ok(mut n) = n.try_borrow_mut() {
            n.push(f);
        }
    });
    scope.terminate_execution();
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
        HEAP_HIT.with(|h| h.set(false));
        HEAP_MAX.with(|h| h.set(limits.max_heap));
        HEAP.with(|h| h.replace(Some(isolate.thread_safe_handle())));
        isolate.add_near_heap_limit_callback(near_heap_limit, std::ptr::null_mut());
        let watch = Watch::new(isolate.thread_safe_handle());
        WATCH.with(|w| w.replace(Some(Arc::clone(&watch))));
        Engine {
            scripts: ByKey::default(),
            isolate,
            max_heap: limits.max_heap,
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
            let c = compile(&mut self.isolate, script).map_err(|f| {
                if HEAP_HIT.with(Cell::get) {
                    Failure::OutOfMemory(self.max_heap)
                } else {
                    f
                }
            })?;
            if self.scripts.len() >= MAX_SCRIPTS {
                self.scripts.clear();
            }
            self.scripts.insert(script.key, c);
        }
        Ok(self.scripts.get_mut(&script.key).expect("inserted"))
    }

    /// A call of `script` on `docs` documents of `segment`: `body` runs it
    /// from `Args::from`, again after each field the script asks for.
    #[allow(clippy::too_many_arguments)]
    fn call(
        &mut self,
        script: &Script,
        params: &Params,
        segment: u64,
        fields: &mut dyn Fields,
        docs: usize,
        mut body: impl FnMut(
            &mut v8::TryCatch<'_, v8::HandleScope<'_>>,
            &mut Compiled,
            Args<'_>,
        ) -> Option<Out>,
    ) -> Result<Out, Failure> {
        let (at, deadline) = script.deadline();
        let timed_out = Failure::TimedOut(script.limits.timeout);
        if Instant::now() >= at {
            return Err(timed_out);
        }
        let max_heap = self.max_heap;
        self.compiled(script)?;
        let Engine {
            scripts,
            isolate,
            watch,
            ..
        } = self;
        let c = scripts.get_mut(&script.key).expect("compiled above");
        let scope = &mut v8::HandleScope::new(isolate);
        let context = v8::Local::new(scope, &c.context);
        let scope = &mut v8::ContextScope::new(scope, context);
        let tc = &mut v8::TryCatch::new(scope);
        let params_v = c.params(tc, params)?;
        for slot in &mut c.slots {
            slot.loaded = false;
        }
        let mut from = 0;
        loop {
            for f in 0..c.slots.len() {
                if c.slots[f].used && !c.slots[f].loaded {
                    c.load(tc, f, segment, fields, docs)?;
                }
            }
            NEEDS.with(|n| n.borrow_mut().clear());
            let r = {
                let watched = watch.call(deadline);
                let r = body(
                    tc,
                    c,
                    Args {
                        params: params_v,
                        from,
                    },
                );
                (r, watched.end())
            };
            let (r, stopped) = r;
            let needs = NEEDS.with(|n| std::mem::take(&mut *n.borrow_mut()));
            if tc.is_execution_terminating() || tc.has_terminated() || stopped || !needs.is_empty()
            {
                tc.cancel_terminate_execution();
            }
            if HEAP_HIT.with(Cell::get) {
                return Err(Failure::OutOfMemory(max_heap));
            }
            if stopped {
                return Err(timed_out);
            }
            if !needs.is_empty() {
                for f in needs {
                    let slot = usize::try_from(f).ok().filter(|&f| {
                        f < c.slots.len() && c.fields[f].1 != Kind::Unreadable && !c.slots[f].loaded
                    });
                    let Some(f) = slot else {
                        return Err(Failure::Host(format!(
                            "the script asked for field {f}, which it cannot read"
                        )));
                    };
                    c.slots[f].used = true;
                }
                from = (c.state.get(0).max(0) as usize).min(docs);
                tc.reset();
                if Instant::now() >= at {
                    return Err(timed_out);
                }
                continue;
            }
            return match r {
                Some(out) => Ok(out),
                None if tc.exception().is_some() && !tc.has_terminated() => {
                    let at = c.state.get(0).max(0) as usize;
                    Err(thrown(tc, at))
                }
                // Stopped with nothing thrown and no reason the host knows.
                None => Err(timed_out),
            };
        }
    }
}

impl Compiled {
    /// `params` as the script reads them: parsed and frozen once.
    fn params<'s>(
        &mut self,
        tc: &mut v8::TryCatch<'s, v8::HandleScope<'_>>,
        params: &Params,
    ) -> Result<v8::Local<'s, v8::Value>, Failure> {
        if let Some((json, v)) = self.params.get(&params.key) {
            if *json == params.json {
                return Ok(v8::Local::new(tc, v));
            }
        }
        let s = v8::String::new(tc, &params.json)
            .ok_or_else(|| Failure::Host("params too long".into()))?;
        let v = v8::json::parse(tc, s).ok_or_else(|| Failure::Host("params: not JSON".into()))?;
        let harden = v8::Local::new(tc, &self.harden);
        let undefined = v8::undefined(tc).into();
        let v = harden
            .call(tc, undefined, &[v])
            .ok_or_else(|| Failure::Host("params: too deep to freeze".into()))?;
        if self.params.len() >= MAX_PARAMS {
            self.params.clear();
        }
        self.params
            .insert(params.key, (params.json.clone(), v8::Global::new(tc, v)));
        Ok(v)
    }

    /// Field `f`'s column for the call's `docs` documents, from the host.
    fn load(
        &mut self,
        scope: &mut v8::HandleScope<'_>,
        f: usize,
        segment: u64,
        fields: &mut dyn Fields,
        docs: usize,
    ) -> Result<(), Failure> {
        self.values.clear();
        self.starts.clear();
        fields
            .load(f, &mut self.values, &mut self.starts)
            .map_err(Failure::Host)?;
        if self.starts.len() != docs + 1 {
            return Err(Failure::Host(format!(
                "field {f}: {} starts for {docs} documents",
                self.starts.len()
            )));
        }
        let slot = &mut self.slots[f];
        let (values, new_values) = slot.values.grow(scope, self.values.len());
        let (starts, new_starts) = slot.starts.grow(scope, self.starts.len());
        slot.values.copy_from(&self.values);
        slot.starts.copy_from(&self.starts);
        let keyword = self.fields[f].1 == Kind::Keyword;
        if keyword || new_values || new_starts || slot.pair.is_none() {
            let mut items = vec![values, starts];
            if keyword {
                if slot.segment != segment || slot.terms.len() > MAX_TERMS {
                    slot.terms.clear();
                    slot.segment = segment;
                }
                let mut terms = Vec::with_capacity(self.values.len());
                for &v in &self.values {
                    let ord = v as i64;
                    let t = match slot.terms.get(&ord) {
                        Some(t) => v8::Local::new(scope, t),
                        None => {
                            let t = fields.term(f, ord).map_err(Failure::Host)?;
                            let t = v8::String::new(scope, &t).ok_or_else(|| {
                                Failure::Host("a term too long for a js string".into())
                            })?;
                            slot.terms.insert(ord, v8::Global::new(scope, t));
                            t
                        }
                    };
                    terms.push(t.into());
                }
                items.push(v8::Array::new_with_elements(scope, &terms).into());
            }
            let pair = v8::Array::new_with_elements(scope, &items);
            slot.pair = Some(v8::Global::new(scope, pair));
            let cols = v8::Local::new(scope, &self.cols);
            cols.set_index(scope, f as u32, pair.into());
        }
        slot.loaded = true;
        Ok(())
    }
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
    let setup = |what: &str| {
        Failure::Host(format!(
            "the js runtime could not set up the script: {what}"
        ))
    };

    let prelude = v8::String::new(tc, PRELUDE).ok_or_else(|| setup("the prelude"))?;
    let factory = v8::Script::compile(tc, prelude, None)
        .and_then(|s| s.run(tc))
        .and_then(|v| v8::Local::<v8::Function>::try_from(v).ok())
        .ok_or_else(|| setup("the prelude"))?;

    let user = user_function(tc, &script.source)?;

    let mut names = Vec::with_capacity(script.fields.len());
    for (n, _) in &script.fields {
        names.push(
            v8::String::new(tc, n)
                .ok_or_else(|| setup("a field name"))?
                .into(),
        );
    }
    let names = v8::Array::new_with_elements(tc, &names);
    let kinds: Vec<v8::Local<v8::Value>> = script
        .fields
        .iter()
        .map(|(_, k)| v8::Integer::new(tc, k.code()).into())
        .collect();
    let kinds = v8::Array::new_with_elements(tc, &kinds);
    let cols = v8::Array::new(tc, 0);
    let mut state = Floats::<i32>::new();
    let state_arr = state.ensure(tc, 1);
    let need = v8::Function::new(tc, need_callback).ok_or_else(|| setup("need"))?;
    let undefined = v8::undefined(tc).into();
    let api = factory
        .call(
            tc,
            undefined,
            &[
                need.into(),
                user.into(),
                names.into(),
                kinds.into(),
                cols.into(),
                state_arr,
            ],
        )
        .and_then(|v| v8::Local::<v8::Object>::try_from(v).ok())
        .ok_or_else(|| setup("the runner"))?;
    let mut get = |name: &str| -> Result<v8::Global<v8::Function>, Failure> {
        let k = v8::String::new(tc, name).ok_or_else(|| setup(name))?;
        let f = api
            .get(tc, k.into())
            .and_then(|v| v8::Local::<v8::Function>::try_from(v).ok())
            .ok_or_else(|| setup(name))?;
        Ok(v8::Global::new(tc, f))
    };
    let batch = get("batch")?;
    let one = get("one")?;
    let harden = get("params")?;
    Ok(Compiled {
        source: script.source.clone(),
        fields: script.fields.clone(),
        context: v8::Global::new(tc, context),
        batch,
        one,
        harden,
        cols: v8::Global::new(tc, cols),
        state,
        slots: (0..script.fields.len())
            .map(|_| Slot {
                used: false,
                loaded: false,
                values: Floats::new(),
                starts: Floats::new(),
                pair: None,
                terms: HashMap::new(),
                segment: u64::MAX,
            })
            .collect(),
        out: Floats::new(),
        scores: Floats::new(),
        params: ByKey::default(),
        values: Vec::new(),
        starts: Vec::new(),
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

/// A [`Watch`] with no call running.
const IDLE: u64 = 0;
/// The watchdog is stopping the call.
const FIRING: u64 = 1;
/// The watchdog stopped the call.
const FIRED: u64 = 2;

/// An isolate's call in progress, as the watchdog sees it: [`IDLE`],
/// [`FIRING`], [`FIRED`], or the call's deadline in nanoseconds after
/// [`EPOCH`] (above `FIRED`).
struct Watch {
    handle: v8::IsolateHandle,
    state: AtomicU64,
    /// Whether the call is past its heap limit, which stops it as its
    /// deadline does.
    over: AtomicBool,
}

/// Stops each isolate's call at its deadline.
struct Watchdog {
    watches: Mutex<Vec<Arc<Watch>>>,
    /// Whether the watchdog may sleep until a call starts.
    idle: AtomicBool,
    thread: OnceLock<std::thread::Thread>,
}

static EPOCH: OnceLock<Instant> = OnceLock::new();

fn nanos(t: Instant) -> u64 {
    let e = *EPOCH.get_or_init(Instant::now);
    u64::try_from(t.saturating_duration_since(e).as_nanos())
        .unwrap_or(u64::MAX)
        .max(FIRED + 1)
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

/// A call the watchdog watches until it ends.
struct Watched<'w> {
    watch: &'w Watch,
    ended: bool,
}

impl Watched<'_> {
    /// The call is over: whether the watchdog stopped it. Once this returns,
    /// the watchdog stops nothing more of it: a stop it asked for is pending
    /// on the isolate only when this says so.
    fn end(mut self) -> bool {
        self.ended = true;
        self.watch.end()
    }
}

impl Drop for Watched<'_> {
    fn drop(&mut self) {
        if !self.ended {
            self.watch.end();
        }
    }
}

impl Watch {
    fn new(handle: v8::IsolateHandle) -> Arc<Self> {
        let w = Arc::new(Watch {
            handle,
            state: AtomicU64::new(IDLE),
            over: AtomicBool::new(false),
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
        self.state.store(deadline, Ordering::Release);
        let dog = watchdog();
        if dog.idle.swap(false, Ordering::AcqRel) {
            if let Some(t) = dog.thread.get() {
                t.unpark();
            }
        }
        Watched {
            watch: self,
            ended: false,
        }
    }

    fn end(&self) -> bool {
        loop {
            let s = self.state.load(Ordering::Acquire);
            if s == FIRING {
                // The watchdog is between deciding and stopping.
                std::hint::spin_loop();
                continue;
            }
            if self
                .state
                .compare_exchange(s, IDLE, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return s == FIRED;
            }
        }
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
            // Set before looking, so a call that starts after the look wakes
            // the watchdog.
            self.idle.store(true, Ordering::Release);
            let now = nanos(Instant::now());
            let mut running = false;
            {
                let watches = self.watches.lock().unwrap_or_else(|p| p.into_inner());
                for w in watches.iter() {
                    let d = w.state.load(Ordering::Acquire);
                    if d <= FIRED {
                        continue;
                    }
                    running = true;
                    w.handle.request_interrupt(heap_check, std::ptr::null_mut());
                    let over = w.over.swap(false, Ordering::AcqRel);
                    if (d <= now || over)
                        && w.state
                            .compare_exchange(d, FIRING, Ordering::AcqRel, Ordering::Acquire)
                            .is_ok()
                    {
                        w.handle.terminate_execution();
                        w.state.store(FIRED, Ordering::Release);
                    }
                }
            }
            if running {
                self.idle.store(false, Ordering::Release);
                std::thread::sleep(TICK);
            } else {
                std::thread::park_timeout(Duration::from_secs(1));
            }
        }
    }
}

#[cfg(test)]
mod tests;
