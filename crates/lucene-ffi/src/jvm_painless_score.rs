//! OpenSearch's `script_score` (`ScriptScoreQuery`) running a Painless
//! script natively, as node `43` of the query tree ([`crate::jvm_queries`]).
//!
//! The plugin runs Painless's own front end and sends the script's IR
//! ([`lucene_painless::ir`]); each segment compiles it and runs it per
//! document with the score context's variables: `params` (with `doc`,
//! `_doc`, `_source` and `_fields`, as `ScoreScript` adds them), `doc`, and
//! `_score`, the sub-query's score.
//!
//! A document's score is `(float) execute()` checked as `ScriptScorer`
//! checks it, and Lucene's [`FunctionScoreQuery`] multiplies it by the
//! boost as `ScriptScorer` does, in `float`.
//!
//! The plugin asks [`check`] first, before anything runs: a script the native
//! runtime would not run exactly as Painless does runs in Java from the
//! start ([`lucene_painless::check`]). A script that passes runs natively to
//! the end; an exception it throws is the script's error, reported as
//! [`script_error`] for the plugin to raise as Painless's `ScriptException`,
//! and never a reason to run it again in Java.
//!
//! The node, after its kind byte:
//!
//! | field | layout |
//! |---|---|
//! | sub-query | a node |
//! | context | `u8`: `0` score, `1` score reading `_score`, `2` filter |
//! | IR | `i32` length, then the `PIR2` blob |
//! | params | `i32` length, then a map constant ([`lucene_painless::ir::decode_constant`]) |
//! | index | string: `_getIndex()` |
//! | shard | `i32`: `_getShardId()` |
//! | script | `i32`: the plugin's number for the script within the request |
//! | fields | `i32` count, then per field: string name, `u8` [kind](FieldKind) |
//!
//! The fields are the ones the script names by a constant (`doc['f']`); the
//! check refuses a script that names one any other way.
//!
//! OpenSearch's `script` query (`ScriptQuery`, a `FilterScript`) is node
//! `45`: the same fields with no sub-query and flag `2`. Its `params` are the
//! script's alone, and a document matches when `execute()` returns `true`,
//! as a [`FunctionMatchQuery`] at `ScriptQuery`'s match cost.

use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use lucene_codecs::doc_values::NumericReader;
use lucene_painless::api::doc::{DocKind, DocLookup, DocValues};
use lucene_painless::check::{Refused, Served};
use lucene_painless::compile::missing;
use lucene_painless::ir;
use lucene_painless::jmap::{JMap, Order};
use lucene_painless::object::JObject;
use lucene_painless::specialize::{specialize, Facts};
use lucene_painless::typed::{lower, BatchEnv, BatchRegs, Regs, Typed};
use lucene_painless::value::{MapKind, Value};
use lucene_painless::vm::{DocAccess, Env, Program, Throw, Vm};
use lucene_search::extended_query::ExtendedQuery;
use lucene_search::function::{FunctionMatchQuery, FunctionScoreQuery};
use lucene_search::query::Clause;
use lucene_search::reader::doc_values::{
    direct_numeric, direct_sorted_numeric_singleton, get_sorted_numeric, get_sorted_set,
};
use lucene_search::reader::{SortedNumericDocValues, SortedSetDocValues};
use lucene_search::values_source::{
    BoxDoubleValues, DoubleValues, DoubleValuesSource, ValuesContext,
};
use lucene_search::{Error, Result};
use lucene_util::half_float::sortable_short_to_half_float;
use lucene_util::numeric_utils::{sortable_int_to_float, sortable_long_to_double};

use crate::error::FfiStatus;
use crate::jvm_nodes::invalid;
use crate::jvm_reader::Cursor;

pub(crate) const NODE_PAINLESS_SCORE: u8 = 43;
pub(crate) const NODE_PAINLESS_FILTER: u8 = 45;

/// `ScriptQuery`'s `TwoPhaseIterator.matchCost()`.
const FILTER_MATCH_COST: f32 = 1000.0;

/// How a field's doc values read as `ScriptDocValues`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FieldKind {
    /// `0`: an integral type's sorted numerics, as `Longs`.
    Long,
    /// `1`: a double's sortable longs, as `Doubles`.
    Double,
    /// `2`: a float's sortable ints, as `Doubles`.
    Float,
    /// `3`: a half_float's sortable shorts, as `Doubles`.
    HalfFloat,
    /// `4`: a boolean's 0 and 1, as `Booleans`.
    Boolean,
    /// `5`: a keyword's sorted set, as `Strings`.
    Keyword,
    /// `254`: a mapped field the native runtime cannot read.
    Unsupported,
    /// `255`: no field of that name is mapped.
    Unmapped,
}

impl FieldKind {
    /// How a script reads the field directly, when it can.
    fn doc_kind(self) -> Option<DocKind> {
        match self {
            FieldKind::Long => Some(DocKind::Longs),
            FieldKind::Double | FieldKind::Float | FieldKind::HalfFloat => Some(DocKind::Doubles),
            FieldKind::Boolean => Some(DocKind::Booleans),
            FieldKind::Keyword => Some(DocKind::Strings),
            FieldKind::Unsupported | FieldKind::Unmapped => None,
        }
    }
}

#[derive(Debug, Clone)]
struct FieldSpec {
    /// The name the script reads the field by.
    name: String,
    /// The name its doc values are under: an alias's target.
    read: String,
    kind: FieldKind,
}

/// The script, for each segment to compile and run.
#[derive(Debug)]
pub(crate) struct PainlessScoreSource(Arc<Spec>);

#[derive(Debug)]
struct Spec {
    script: ir::Script,
    /// For [`DoubleValuesSource::equality_key`]: the blob the script came from.
    ir_blob: Vec<u8>,
    params: ir::Constant,
    params_blob: Vec<u8>,
    needs_score: bool,
    /// A `FilterScript`: `execute()` returns whether the document matches.
    filter: bool,
    index: String,
    shard: i32,
    /// The plugin's number for the script, in the errors it raises.
    script_id: i32,
    fields: Vec<FieldSpec>,
}

pub(crate) fn decode(
    c: &mut Cursor<'_>,
    depth: usize,
    nodes: &mut usize,
) -> std::result::Result<Clause, FfiStatus> {
    let sub = crate::jvm_reader::decode_node(c, depth + 1, nodes)?;
    let spec = decode_spec(c)?;
    if spec.filter {
        return Err(invalid(
            "query tree: painless score node with a filter script",
        ));
    }
    // The plugin checked it before sending it: a refusal here is a plugin
    // that did not.
    spec.check()
        .map_err(|r| invalid(format!("query tree: painless script not checked: {r}")))?;
    let source = PainlessScoreSource(Arc::new(spec));
    Ok(Clause::Extended(Box::new(ExtendedQuery::FunctionScore(
        FunctionScoreQuery::new(sub, Arc::new(source)),
    ))))
}

/// The `script` query node: its script, matching where it returns `true`.
pub(crate) fn decode_filter(c: &mut Cursor<'_>) -> std::result::Result<Clause, FfiStatus> {
    let spec = decode_spec(c)?;
    if !spec.filter {
        return Err(invalid(
            "query tree: painless filter node with a score script",
        ));
    }
    spec.check()
        .map_err(|r| invalid(format!("query tree: painless script not checked: {r}")))?;
    let source = PainlessScoreSource(Arc::new(spec));
    Ok(Clause::Extended(Box::new(ExtendedQuery::FunctionMatch(
        FunctionMatchQuery::with_match_cost(
            Arc::new(source),
            Arc::new(|v: f64| v != 0.0),
            FILTER_MATCH_COST,
        ),
    ))))
}

/// The node's script, after its sub-query.
fn decode_spec(c: &mut Cursor<'_>) -> std::result::Result<Spec, FfiStatus> {
    let (needs_score, filter) = match c.u8()? {
        0 => (false, false),
        1 => (true, false),
        2 => (false, true),
        other => {
            return Err(invalid(format!(
                "query tree: painless script context {other} (expected 0..=2)"
            )))
        }
    };
    let ir_blob = c.bytes()?.to_vec(); // alloc-ok: the script, bounded by the blob
    let script =
        ir::decode(&ir_blob).map_err(|e| invalid(format!("query tree: painless script: {e}")))?;
    let params_blob = c.bytes()?.to_vec(); // alloc-ok: the params, bounded by the blob
    let params = ir::decode_constant(&params_blob)
        .map_err(|e| invalid(format!("query tree: painless params: {e}")))?;
    if !matches!(params, ir::Constant::Map(_)) {
        return Err(invalid("query tree: painless params are not a map"));
    }
    let index = string(c)?;
    let shard = c.i32()?;
    let script_id = c.i32()?;
    let n = c.len()?;
    let mut fields = Vec::with_capacity(n.min(c.remaining())); // alloc-ok: bounded by the blob
    for _ in 0..n {
        let name = string(c)?;
        let read = string(c)?;
        let kind = match c.u8()? {
            0 => FieldKind::Long,
            1 => FieldKind::Double,
            2 => FieldKind::Float,
            3 => FieldKind::HalfFloat,
            4 => FieldKind::Boolean,
            5 => FieldKind::Keyword,
            254 => FieldKind::Unsupported,
            255 => FieldKind::Unmapped,
            other => {
                return Err(invalid(format!(
                    "query tree: painless field kind {other} (expected 0..=5, 254 or 255)"
                )))
            }
        };
        fields.push(FieldSpec { name, read, kind });
    }
    Ok(Spec {
        script,
        ir_blob,
        params,
        params_blob,
        needs_score,
        filter,
        index,
        shard,
        script_id,
        fields,
    })
}

/// Whether the native runtime runs the script of `blob` (a node's fields
/// after its sub-query) exactly as Painless would: `None` when it does,
/// else why not.
pub(crate) fn check(blob: &[u8]) -> std::result::Result<Option<String>, FfiStatus> {
    let mut c = Cursor::new(blob);
    let spec = decode_spec(&mut c)?;
    if c.remaining() != 0 {
        return Err(invalid("painless check: trailing bytes"));
    }
    Ok(spec.check().err().map(|r| r.0))
}

/// Where a script threw: the segment, the document and the score it ran
/// with.
#[derive(Clone, Copy)]
struct At {
    leaf: usize,
    doc: i32,
    score: f32,
}

/// The error a script's exception fails the search with: the plugin raises
/// Painless's own `ScriptException` for the document `at` names, or else
/// one around the exception `class` (with `message`, `None` for none), at
/// source offset `offset`.
fn script_error(script_id: i32, at: At, offset: i32, t: &Throw) -> Error {
    // A gap of the runtime the check let through is the plugin's failure,
    // not the script's: never shown as the script's exception.
    if t.class() == lucene_painless::api::UNSUPPORTED {
        return runtime_failure(t);
    }
    let message = match t.message() {
        Some(m) => format!("1{m}"),
        None => "0".to_string(),
    };
    Error::IllegalArgument(format!(
        "{SCRIPT_ERROR}{script_id}\u{1}{}\u{1}{}\u{1}{}\u{1}{offset}\u{1}{}\u{1}{message}",
        at.leaf,
        at.doc,
        at.score.to_bits(),
        t.class()
    ))
}

/// What a [`script_error`] message starts with.
const SCRIPT_ERROR: &str = "\u{1}painless-script-error\u{1}";

/// What the message of `ScriptScorer`'s own `IllegalArgumentException` (a
/// negative or NaN score) starts with; the message follows.
const SCORE_ERROR: &str = "\u{1}painless-score-error\u{1}";

/// What the message of an `OpenSearchException` a query holding a script
/// raises (`function_score`'s missing value or invalid score) starts with.
const OPENSEARCH_ERROR: &str = "\u{1}opensearch-error\u{1}";

/// An `IllegalArgumentException` with `message` that fails a search holding
/// a script without running it again in Java.
pub(crate) fn score_error(message: impl std::fmt::Display) -> Error {
    Error::IllegalArgument(format!("{SCORE_ERROR}{message}"))
}

/// An `OpenSearchException` with `message` that fails a search holding a
/// script without running it again in Java.
pub(crate) fn opensearch_error(message: impl std::fmt::Display) -> Error {
    Error::IllegalArgument(format!("{OPENSEARCH_ERROR}{message}"))
}

fn string(c: &mut Cursor<'_>) -> std::result::Result<String, FfiStatus> {
    Ok(std::str::from_utf8(c.bytes()?)
        .map_err(|_| FfiStatus::InvalidUtf8)?
        .to_string())
}

/// A script's exception as the search's error.
/// A failure of the runtime itself while running a checked script (an
/// unreadable segment): not the script's error, and not one Java would
/// answer differently.
fn runtime_failure(t: &Throw) -> Error {
    Error::IllegalArgument(format!(
        "painless runtime: {}: {}",
        t.class(),
        t.message().unwrap_or_default()
    ))
}

/// What `params._source` and `params._fields` hold: lookups the native
/// runtime does not have, so any use of one fails the search.
#[derive(Debug)]
struct Unavailable(&'static str);

impl JObject for Unavailable {
    fn class_name(&self) -> &str {
        self.0
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn call(
        &self,
        _vm: &mut Vm,
        _this: &Value,
        name: &str,
        _args: &[Value],
    ) -> Option<std::result::Result<Value, Throw>> {
        Some(Err(missing(format!("{}.{name}", self.0))))
    }
    fn field(&self, _vm: &mut Vm, name: &str) -> Option<std::result::Result<Value, Throw>> {
        Some(Err(missing(format!("{}.{name}", self.0))))
    }
    fn iterate(&self, _vm: &mut Vm) -> Option<std::result::Result<Vec<Value>, Throw>> {
        Some(Err(missing(self.0)))
    }
}

impl Spec {
    /// The script class's own methods the score context serves.
    fn internals(&self) -> &'static [&'static str] {
        const WITHOUT_SCORE: &[&str] = &[
            "getParams",
            "getDoc",
            "_getDocId",
            "_getDocBaseId",
            "_getIndex",
            "_getShardId",
        ];
        const WITH_SCORE: &[&str] = &[
            "getParams",
            "getDoc",
            "get_score",
            "_getDocId",
            "_getDocBaseId",
            "_getIndex",
            "_getShardId",
        ];
        const FILTER: &[&str] = &["getParams", "getDoc"];
        if self.filter {
            FILTER
        } else if self.needs_score {
            WITH_SCORE
        } else {
            WITHOUT_SCORE
        }
    }

    /// [`lucene_painless::check`] on the program each segment runs.
    fn check(&self) -> std::result::Result<(), Refused> {
        let mut prog =
            lucene_painless::compile::compile(&self.script).map_err(|e| Refused(e.to_string()))?;
        let doc_lookup = Value::object(DocLookup);
        let params = self
            .params(&doc_lookup)
            .map_err(|t| Refused(format!("params: {}", t.message().unwrap_or_default())))?;
        let fields: Vec<(&str, Option<DocKind>)> = self
            .fields
            .iter()
            .map(|f| (f.name.as_str(), f.kind.doc_kind()))
            .collect();
        let unmapped: Vec<&str> = self
            .fields
            .iter()
            .filter(|f| f.kind == FieldKind::Unmapped)
            .map(|f| f.name.as_str())
            .collect();
        specialize(
            &mut prog,
            &Facts {
                params: Some(&params),
                fields: &fields,
            },
        );
        lucene_painless::check::check(
            &self.script,
            &prog,
            &Served {
                fields: &fields,
                unmapped: &unmapped,
                internals: self.internals(),
            },
        )
    }

    /// `ScoreScript`'s `params`: a copy of the script's, then
    /// `LeafSearchLookup.asMap()`; `FilterScript`'s: the script's.
    fn params(&self, doc: &Value) -> std::result::Result<Value, Throw> {
        let ir::Constant::Map(entries) = &self.params else {
            unreachable!("decode checks the params are a map")
        };
        let mut m = JMap::for_copy(Order::Hash, entries.len());
        for (k, v) in entries {
            let k = lucene_painless::compile::constant(k).map_err(|e| missing(e.to_string()))?;
            let v = lucene_painless::compile::constant(v).map_err(|e| missing(e.to_string()))?;
            m.insert(k, v);
        }
        if self.filter {
            return Ok(Value::map(MapKind::HashMap, m));
        }
        m.insert(Value::str("doc"), doc.clone());
        m.insert(Value::str("_doc"), doc.clone());
        m.insert(
            Value::str("_source"),
            Value::object(Unavailable("org.opensearch.search.lookup.SourceLookup")),
        );
        m.insert(
            Value::str("_fields"),
            Value::object(Unavailable("org.opensearch.search.lookup.LeafFieldsLookup")),
        );
        Ok(Value::map(MapKind::HashMap, m))
    }
}

/// A field's open doc values in a segment.
enum Opened<'c> {
    Numeric(Box<dyn SortedNumericDocValues + 'c>),
    Set(Box<dyn SortedSetDocValues + 'c>),
}

/// A keyword slot `ScriptDocValues.Strings` grew but no document wrote:
/// an empty `BytesRefBuilder`.
const EMPTY_ORD: i64 = -1;

/// `ArrayUtil.oversize(min, bytes_per_element)` on a 64-bit JVM.
fn oversize(min: usize, bytes_per_element: usize) -> usize {
    if min == 0 {
        return 0;
    }
    let size = min + (min >> 3).max(3);
    match bytes_per_element {
        4 => (size + 1) & !1,
        2 => (size + 3) & !3,
        1 => (size + 7) & !7,
        _ => size,
    }
}

/// `ScriptDocValues`'s `resize(n)`: `values` grown to hold `n`, the way
/// each view's array grows (`Strings` holds object references, compressed
/// on a heap under 32 GB).
fn resize(items: &mut Vec<i64>, n: usize, kind: FieldKind) {
    if n > items.len() {
        let (bytes, fill) = match kind {
            FieldKind::Boolean => (1, 0),
            FieldKind::Keyword => (4, EMPTY_ORD),
            _ => (8, 0),
        };
        items.resize(oversize(n, bytes), fill);
    }
}

/// `ArrayIndexOutOfBoundsException` for `index` into an array of `len`.
fn out_of_bounds(index: i32, len: usize) -> Throw {
    Throw::new(
        "java.lang.ArrayIndexOutOfBoundsException",
        format!("Index {index} out of bounds for length {len}"),
    )
}

/// Most keyword values a segment keeps as strings by ordinal.
const MAX_CACHED_TERMS: usize = 4096;

/// A field's values in the document last read.
struct Current {
    doc: i32,
    /// `ScriptDocValues`'s values array, which it reuses from document to
    /// document and grows as `ArrayUtil.grow` does: the document's values
    /// first, then whatever earlier documents left (a keyword's never
    /// written slots, [`EMPTY_ORD`], read as ""). Numbers by their bits: a
    /// long, a double, or 0 and 1; a keyword's ordinals.
    items: Vec<i64>,
    /// How many of `items` are the document's.
    count: usize,
    /// The `ScriptDocValues` view made of them, once asked for.
    view: Option<Value>,
}

/// The score context of one segment.
struct ScoreEnv<'c> {
    source: Arc<Spec>,
    reader: &'c lucene_search::directory_reader::SegmentReader,
    /// The segment's number in the reader.
    leaf: usize,
    /// Per field of `source.fields`: its doc values once read.
    opened: Vec<Option<Opened<'c>>>,
    /// Per field: its values in the document last read.
    current: Vec<Current>,
    /// Per keyword field: strings by ordinal.
    terms: Vec<HashMap<i64, Value>>,
    /// The documents of the batch [`BatchEnv`] reads, and their scores.
    batch: Vec<i32>,
    batch_scores: Vec<f32>,
    /// Per field: the batch's value counts and first values, once read.
    columns: Vec<Column>,
    /// Which batch `columns` hold.
    epoch: u64,
    /// Per field: its one-value-per-document column, read without an
    /// iterator, when the codec stores it so (`None` until first tried).
    direct: Vec<Option<Option<NumericReader<'c>>>>,
    doc: i32,
    score: f64,
    params: Value,
    doc_lookup: Value,
    index: Value,
}

/// A field's value counts and first values (decoded, by their bits) in
/// each document of a batch.
#[derive(Default)]
struct Column {
    epoch: u64,
    counts: Vec<u64>,
    firsts: Vec<u64>,
}

/// A numeric field's stored longs as its `ScriptDocValues` values' bits.
fn decode_column(kind: FieldKind, vs: &mut [u64]) {
    match kind {
        FieldKind::Long | FieldKind::Boolean => {}
        FieldKind::Double => {
            for v in vs {
                *v = sortable_long_to_double(*v as i64).to_bits();
            }
        }
        FieldKind::Float => {
            for v in vs {
                *v = f64::from(sortable_int_to_float(*v as i32)).to_bits();
            }
        }
        FieldKind::HalfFloat => {
            for v in vs {
                *v = f64::from(sortable_short_to_half_float(*v as i16)).to_bits();
            }
        }
        FieldKind::Keyword | FieldKind::Unsupported | FieldKind::Unmapped => {}
    }
}

fn no_value() -> Throw {
    Throw::new(
        "java.lang.IllegalStateException",
        lucene_painless::api::doc::NO_VALUE,
    )
}

impl ScoreEnv<'_> {
    fn field(&self, name: &str) -> Option<usize> {
        self.source.fields.iter().position(|f| f.name == name)
    }

    /// Field `i`'s values in the current document.
    fn load(&mut self, i: usize) -> std::result::Result<&Current, Throw> {
        if self.current[i].doc != self.doc {
            self.read(i)
                .map_err(|e| missing(format!("doc[{}]: {e}", self.source.fields[i].name)))?;
        }
        Ok(&self.current[i])
    }

    fn read(&mut self, i: usize) -> Result<()> {
        let spec = &self.source.fields[i];
        let kind = spec.kind;
        if self.opened[i].is_none() {
            self.opened[i] = Some(match kind {
                FieldKind::Keyword => Opened::Set(get_sorted_set(self.reader, &spec.read)?),
                _ => Opened::Numeric(get_sorted_numeric(self.reader, &spec.read)?),
            });
        }
        let doc = self.doc;
        let cur = &mut self.current[i];
        cur.count = 0;
        cur.view = None;
        match self.opened[i].as_mut() {
            Some(Opened::Numeric(dv)) => {
                if dv.advance_exact(doc)? {
                    let n = dv.doc_value_count() as usize;
                    resize(&mut cur.items, n, kind);
                    if let Some(vs) = dv.take_values() {
                        cur.items[..n].copy_from_slice(&vs[..n]);
                    } else {
                        for slot in &mut cur.items[..n] {
                            *slot = dv.next_value()?;
                        }
                    }
                    cur.count = n;
                    let decode: fn(i64) -> i64 = match kind {
                        FieldKind::Long => |v| v,
                        FieldKind::Double => |v| sortable_long_to_double(v).to_bits() as i64,
                        FieldKind::Float => {
                            |v| f64::from(sortable_int_to_float(v as i32)).to_bits() as i64
                        }
                        FieldKind::HalfFloat => {
                            |v| f64::from(sortable_short_to_half_float(v as i16)).to_bits() as i64
                        }
                        // `ScriptDocValues.Booleans`: `nextValue() == 1`.
                        _ => |v| i64::from(v == 1),
                    };
                    if kind != FieldKind::Long {
                        for v in &mut cur.items[..n] {
                            *v = decode(*v);
                        }
                    }
                }
            }
            Some(Opened::Set(dv)) => {
                if dv.advance_exact(doc)? {
                    let n = dv.doc_value_count() as usize;
                    resize(&mut cur.items, n, kind);
                    for slot in &mut cur.items[..n] {
                        *slot = dv.next_ord()?;
                    }
                    cur.count = n;
                }
            }
            None => unreachable!("opened above"),
        }
        cur.doc = doc;
        Ok(())
    }

    /// Field `i`'s checked kind.
    fn kind(&self, i: usize) -> std::result::Result<FieldKind, Throw> {
        let f = &self.source.fields[i];
        match f.kind {
            FieldKind::Unmapped => Err(Throw::illegal_arg(format!(
                "No field found for [{}] in mapping",
                f.name
            ))),
            FieldKind::Unsupported => Err(missing(format!("doc[{}]'s type", f.name))),
            k => Ok(k),
        }
    }

    /// Value `index` of field `i` as `ScriptDocValues.get` reads it.
    fn item(&mut self, i: usize, index: i32) -> std::result::Result<i64, Throw> {
        let cur = self.load(i)?;
        if cur.count == 0 {
            return Err(no_value());
        }
        // Past the document's values Java reads its reused array.
        match usize::try_from(index).ok().and_then(|x| cur.items.get(x)) {
            Some(v) => Ok(*v),
            None => Err(out_of_bounds(index, cur.items.len())),
        }
    }

    fn term(&mut self, i: usize, ord: i64) -> std::result::Result<Value, Throw> {
        if ord == EMPTY_ORD {
            return Ok(Value::str(""));
        }
        if let Some(v) = self.terms[i].get(&ord) {
            return Ok(v.clone());
        }
        let Some(Opened::Set(dv)) = self.opened[i].as_mut() else {
            unreachable!("a keyword's values are open once read")
        };
        let bytes = dv
            .lookup_ord(ord)
            .map_err(|e| missing(format!("doc[{}]: {e}", self.source.fields[i].name)))?;
        let v = Value::string(String::from_utf8_lossy(&bytes).into_owned());
        if self.terms[i].len() >= MAX_CACHED_TERMS {
            self.terms[i].clear();
        }
        self.terms[i].insert(ord, v.clone());
        Ok(v)
    }

    /// Value `index` of field `i` as a script value.
    fn boxed_item(&mut self, i: usize, index: i32) -> std::result::Result<Value, Throw> {
        let kind = self.kind(i)?;
        let v = self.item(i, index)?;
        Ok(match kind {
            FieldKind::Long => Value::Long(v),
            FieldKind::Boolean => Value::Bool(v != 0),
            FieldKind::Keyword => return self.term(i, v),
            _ => Value::Double(f64::from_bits(v as u64)),
        })
    }

    /// `doc['f']` for field `i`.
    fn view(&mut self, i: usize) -> std::result::Result<Value, Throw> {
        let kind = self.kind(i)?;
        if let Some(v) = &self.load(i)?.view {
            return Ok(v.clone());
        }
        let n = self.current[i].count;
        // With no values Java throws before reading its array: no stale slots.
        let len = if n == 0 {
            0
        } else {
            self.current[i].items.len()
        };
        let mut items = Vec::with_capacity(n); // alloc-ok: the document's values, already read
        let mut stale = Vec::new();
        for x in 0..len {
            let v = self.boxed_item(i, x as i32)?;
            if x < n {
                items.push(v);
            } else {
                stale.push(v);
            }
        }
        let doc_kind = match kind {
            FieldKind::Long => DocKind::Longs,
            FieldKind::Boolean => DocKind::Booleans,
            FieldKind::Keyword => DocKind::Strings,
            _ => DocKind::Doubles,
        };
        let v = DocValues::with_stale(doc_kind, items, stale);
        self.current[i].view = Some(v.clone());
        Ok(v)
    }

    /// Field `i`'s column for the current batch.
    fn column(&mut self, i: usize) -> std::result::Result<&Column, Throw> {
        if self.columns[i].epoch != self.epoch {
            self.read_column(i)
                .map_err(|e| missing(format!("doc[{}]: {e}", self.source.fields[i].name)))?;
        }
        Ok(&self.columns[i])
    }

    fn read_column(&mut self, i: usize) -> Result<()> {
        let spec = &self.source.fields[i];
        let kind = spec.kind;
        if kind == FieldKind::Keyword {
            // Only counted in a batch (`size()`): its ordinals, document by document.
            let mut col = std::mem::take(&mut self.columns[i]);
            col.counts.clear();
            col.firsts.clear();
            let doc = self.doc;
            for b in 0..self.batch.len() {
                self.doc = self.batch[b];
                let read = self.read(i);
                if let Err(e) = read {
                    self.doc = doc;
                    self.columns[i] = col;
                    return Err(e);
                }
                col.counts.push(self.current[i].count as u64);
                col.firsts.push(0);
            }
            self.doc = doc;
            self.current[i].doc = -1;
            col.epoch = self.epoch;
            self.columns[i] = col;
            return Ok(());
        }
        let reader = self.reader;
        let direct = self.direct[i].get_or_insert_with(|| {
            direct_sorted_numeric_singleton(reader, &spec.read)
                .or_else(|| direct_numeric(reader, &spec.read))
        });
        let col = &mut self.columns[i];
        if let Some(dv) = direct {
            col.counts.clear();
            col.firsts.clear();
            for &doc in &self.batch {
                match dv.value(doc)? {
                    Some(v) => {
                        col.counts.push(1);
                        col.firsts.push(v as u64);
                    }
                    None => {
                        col.counts.push(0);
                        col.firsts.push(0);
                    }
                }
            }
            decode_column(kind, &mut col.firsts);
            col.epoch = self.epoch;
            return Ok(());
        }
        if self.opened[i].is_none() {
            self.opened[i] = Some(Opened::Numeric(get_sorted_numeric(
                self.reader,
                &spec.read,
            )?));
        }
        let Some(Opened::Numeric(dv)) = self.opened[i].as_mut() else {
            unreachable!("a batch reads numeric fields")
        };
        let col = &mut self.columns[i];
        col.counts.clear();
        col.firsts.clear();
        for &doc in &self.batch {
            let (n, first) = if dv.advance_exact(doc)? {
                let n = dv.doc_value_count();
                let first = match dv.take_values() {
                    Some(vs) => vs.first().copied().unwrap_or(0),
                    None if n > 0 => dv.next_value()?,
                    None => 0,
                };
                (n, first)
            } else {
                (0, 0)
            };
            col.counts.push(n as u64);
            col.firsts.push(first as u64);
        }
        decode_column(kind, &mut col.firsts);
        col.epoch = self.epoch;
        // The per-document reads must move the doc values again.
        self.current[i].doc = -1;
        Ok(())
    }

    fn count(&mut self, i: usize) -> std::result::Result<i32, Throw> {
        self.kind(i)?;
        Ok(self.load(i)?.count as i32)
    }
}

impl BatchEnv for ScoreEnv<'_> {
    fn column(&mut self, field: usize) -> std::result::Result<(&[u64], &[u64]), Throw> {
        let col = ScoreEnv::column(self, field)?;
        Ok((&col.counts, &col.firsts))
    }

    fn scores(&mut self, out: &mut [u64]) -> std::result::Result<(), Throw> {
        if !self.source.needs_score {
            return Err(missing("score script method get_score"));
        }
        for (o, s) in out.iter_mut().zip(&self.batch_scores) {
            *o = f64::from(*s).to_bits();
        }
        Ok(())
    }
}

impl Env for ScoreEnv<'_> {
    fn internal(&mut self, name: &str, _args: &[Value]) -> std::result::Result<Value, Throw> {
        match name {
            "getParams" => Ok(self.params.clone()),
            "getDoc" => Ok(self.doc_lookup.clone()),
            "get_score" if self.source.needs_score => Ok(Value::Double(self.score)),
            "_getDocId" => Ok(Value::Int(self.doc)),
            "_getDocBaseId" => Ok(Value::Int(self.reader.doc_base + self.doc)),
            "_getIndex" => Ok(self.index.clone()),
            "_getShardId" => Ok(Value::Int(self.source.shard)),
            _ => Err(missing(format!("score script method {name}"))),
        }
    }

    fn doc_values(&mut self, field: &str) -> std::result::Result<Value, Throw> {
        match self.field(field) {
            Some(i) => self.view(i),
            None => Err(missing(format!("doc[{field}] not named by a constant"))),
        }
    }

    fn doc_at(
        &mut self,
        field: usize,
        access: DocAccess,
        index: i32,
    ) -> std::result::Result<Value, Throw> {
        match access {
            DocAccess::Values => self.view(field),
            DocAccess::Value => self.boxed_item(field, 0),
            DocAccess::Get => self.boxed_item(field, index),
            DocAccess::Count => Ok(Value::Int(self.count(field)?)),
            DocAccess::Flip => Ok(Value::Int(if index < 0 {
                index.wrapping_add(self.count(field)?)
            } else {
                index
            })),
        }
    }

    fn doc_long(&mut self, field: usize, index: i32) -> std::result::Result<i64, Throw> {
        self.item(field, index)
    }

    fn doc_double(&mut self, field: usize, index: i32) -> std::result::Result<f64, Throw> {
        Ok(f64::from_bits(self.item(field, index)? as u64))
    }

    fn doc_count(&mut self, field: usize) -> std::result::Result<i32, Throw> {
        Ok(self.load(field)?.count as i32)
    }

    fn score(&mut self) -> std::result::Result<f64, Throw> {
        if self.source.needs_score {
            Ok(self.score)
        } else {
            Err(missing("score script method get_score"))
        }
    }

    fn doc_contains(&mut self, field: &str) -> std::result::Result<bool, Throw> {
        match self.field(field) {
            Some(i) => Ok(self.source.fields[i].kind != FieldKind::Unmapped),
            None => Err(missing(format!(
                "doc.containsKey({field}) not named by a constant"
            ))),
        }
    }
}

/// One segment's scores.
struct ScriptScores<'c> {
    /// The script on the typed machine, when it lowers to it.
    typed: Option<(Typed, Regs)>,
    prog: Rc<Program>,
    env: ScoreEnv<'c>,
    bindings: Vec<Option<Value>>,
    /// The VM's register files, kept across documents.
    pool: Vec<Vec<Value>>,
    /// The column-batch machine's registers.
    columns: BatchRegs,
    args: Vec<Value>,
    scores: Option<BoxDoubleValues<'c>>,
    value: f32,
}

impl ScriptScores<'_> {
    /// The score of the current document (`env.doc`, `env.score`), checked
    /// as `ScriptScorer` checks it; a filter's `1.0` or `0.0`.
    fn run(&mut self) -> Result<f32> {
        let doc = self.env.doc;
        let v = self.run_value()? as f32;
        if self.env.source.filter {
            return Ok(v);
        }
        checked(v, doc)
    }

    /// What the script returns for the current document, as a `double`.
    fn run_value(&mut self) -> Result<f64> {
        let id = self.env.source.script_id;
        let at = At {
            leaf: self.env.leaf,
            doc: self.env.doc,
            score: self.env.score as f32,
        };
        let r = match &mut self.typed {
            Some((t, regs)) => match t.run_double(&mut self.env, regs) {
                Ok(Ok(v)) => return Ok(v),
                Ok(Err(v)) => Ok(v),
                Err(e) => Err(script_error(id, at, t.thrown_at(regs), &e)),
            },
            None => {
                let mut vm = Vm::new(self.prog.clone(), &mut self.env);
                vm.bindings = std::mem::take(&mut self.bindings);
                vm.pool = std::mem::take(&mut self.pool);
                let r = vm.execute(&self.args);
                let loc = vm.loc;
                self.bindings = std::mem::take(&mut vm.bindings);
                self.pool = std::mem::take(&mut vm.pool);
                r.map_err(|e| script_error(id, at, loc, &e))
            }
        };
        let r = r?;
        let v = match r {
            Value::Double(v) => v,
            Value::Bool(b) => f64::from(u8::from(b)),
            r => lucene_painless::types::as_f64(&r).ok_or_else(|| {
                Error::IllegalArgument(format!(
                    "painless script returned {} for a score",
                    r.class_name()
                ))
            })?,
        };
        Ok(v)
    }
}

/// `ScriptScorer.score()`'s check.
fn checked(v: f32, doc: i32) -> Result<f32> {
    if v < 0.0 || v.is_nan() {
        return Err(Error::IllegalArgument(format!(
            "{SCORE_ERROR}script_score script returned an invalid score [{}] for doc [{doc}]. Must be a non-negative score!",
            lucene_painless::number::float_to_string(v)
        )));
    }
    Ok(v)
}

impl ScriptScores<'_> {
    /// [`DoubleValues::fill_batch`] on the column-batch machine.
    fn fill_columns(
        &mut self,
        docs: &[i32],
        scores: &[f32],
        out: &mut [f64],
        has: &mut [bool],
    ) -> Result<()> {
        let Some((t, _)) = &self.typed else {
            unreachable!("a typed script")
        };
        let env = &mut self.env;
        env.epoch += 1;
        env.batch.clear();
        env.batch.extend_from_slice(docs);
        env.batch_scores.clear();
        env.batch_scores.extend_from_slice(&scores[..docs.len()]);
        let out = &mut out[..docs.len()];
        let failed = t
            .run_batch(env, &mut self.columns, out)
            .map_err(|e| runtime_failure(&e))?;
        // In document order, as Java scores them: a score check fails
        // before a later document's exception.
        let upto = failed.as_ref().map_or(out.len(), |(k, ..)| *k);
        if !env.source.filter {
            for (k, o) in out[..upto].iter_mut().enumerate() {
                *o = f64::from(checked(*o as f32, docs[k])?);
            }
        }
        if let Some((k, e, offset)) = failed {
            let at = At {
                leaf: env.leaf,
                doc: docs[k],
                score: env.batch_scores[k],
            };
            return Err(script_error(env.source.script_id, at, offset, &e));
        }
        has[..docs.len()].fill(true);
        Ok(())
    }
}

impl DoubleValues for ScriptScores<'_> {
    fn advance_exact(&mut self, doc: i32) -> Result<bool> {
        self.env.doc = doc;
        if let Some(s) = self.scores.as_mut() {
            self.env.score = if s.advance_exact(doc)? {
                s.double_value()?
            } else {
                0.0
            };
        }
        self.value = self.run()?;
        Ok(true)
    }

    fn double_value(&mut self) -> Result<f64> {
        Ok(f64::from(self.value))
    }

    /// The script reads the score only through `get_score`, which the
    /// batch's scores serve.
    fn batch_capable(&self) -> bool {
        true
    }

    fn fill_batch(
        &mut self,
        docs: &[i32],
        scores: &[f32],
        out: &mut [f64],
        has: &mut [bool],
    ) -> Result<()> {
        if let Some((t, _)) = &self.typed {
            if t.batchable() && !docs.is_empty() {
                return self.fill_columns(docs, scores, out, has);
            }
        }
        for (i, &doc) in docs.iter().enumerate() {
            self.env.doc = doc;
            if self.scores.is_some() {
                self.env.score = f64::from(scores[i]);
            }
            out[i] = f64::from(self.run()?);
            has[i] = true;
        }
        Ok(())
    }
}

impl DoubleValuesSource for PainlessScoreSource {
    fn get_values<'c>(
        &self,
        ctx: &ValuesContext<'c>,
        leaf: usize,
        scores: Option<BoxDoubleValues<'c>>,
    ) -> Result<BoxDoubleValues<'c>> {
        Ok(Box::new(scripts(&self.0, ctx, leaf, scores)?))
    }

    fn needs_scores(&self) -> bool {
        self.0.needs_score
    }

    fn is_cacheable(&self, _ctx: &ValuesContext<'_>, _leaf: usize) -> bool {
        false
    }

    fn describe(&self) -> String {
        if self.0.filter {
            "painless_filter"
        } else {
            "painless_score"
        }
        .to_string()
    }

    fn equality_key(&self) -> String {
        let s = &self.0;
        let hex = |b: &[u8]| b.iter().map(|x| format!("{x:02x}")).collect::<String>();
        let fields: Vec<String> = s
            .fields
            .iter()
            .map(|f| format!("{}:{:?}", f.name, f.kind))
            .collect();
        format!(
            "painless_score(ir={}, params={}, needs_score={}, filter={}, index={}, shard={}, fields=[{}])",
            hex(&s.ir_blob),
            hex(&s.params_blob),
            s.needs_score,
            s.filter,
            s.index,
            s.shard,
            fields.join(",")
        )
    }
}

/// The script compiled, specialised and ready to run on segment `leaf`.
fn scripts<'c>(
    spec: &Arc<Spec>,
    ctx: &ValuesContext<'c>,
    leaf: usize,
    scores: Option<BoxDoubleValues<'c>>,
) -> Result<ScriptScores<'c>> {
    let reader = ctx.leaf_reader(leaf)?;
    let mut prog = lucene_painless::compile::compile(&spec.script)
        .map_err(|e| Error::IllegalArgument(format!("painless script: {e}")))?;
    let arity = prog.funcs[prog.entry as usize].arity as usize;
    let source = Arc::clone(spec);
    let doc_lookup = Value::object(DocLookup);
    let params = source
        .params(&doc_lookup)
        .map_err(|t| runtime_failure(&t))?;
    let fields: Vec<(&str, Option<DocKind>)> = source
        .fields
        .iter()
        .map(|f| (f.name.as_str(), f.kind.doc_kind()))
        .collect();
    specialize(
        &mut prog,
        &Facts {
            params: Some(&params),
            fields: &fields,
        },
    );
    let typed = lower(&prog, &fields).map(|t| {
        let regs = t.regs();
        (t, regs)
    });
    drop(fields);
    let n = source.fields.len();
    let env = ScoreEnv {
        index: Value::str(&source.index),
        reader,
        leaf,
        opened: (0..n).map(|_| None).collect(),
        current: (0..n)
            .map(|_| Current {
                doc: -1,
                items: Vec::new(),
                count: 0,
                view: None,
            })
            .collect(),
        terms: vec![HashMap::new(); n], // alloc-ok: one per field the node names
        batch: Vec::new(),
        batch_scores: Vec::new(),
        columns: (0..n).map(|_| Column::default()).collect(),
        epoch: 0,
        direct: (0..n).map(|_| None).collect(),
        source,
        doc: -1,
        score: 0.0,
        params,
        doc_lookup,
    };
    Ok(ScriptScores {
        typed,
        pool: Vec::new(),
        columns: BatchRegs::default(),
        bindings: vec![None; prog.binding_sites], // alloc-ok: the compiled program's own count
        prog: Rc::new(prog),
        env,
        // `execute(ExplanationHolder)`: no explanation while scoring.
        args: vec![Value::Null; arity], // alloc-ok: execute's own arity
        scores,
        value: 0.0,
    })
}

/// A `function_score` `script_score` function's script
/// (`ScriptScoreFunction`), checked.
#[derive(Debug)]
pub(crate) struct ScriptFunction(Arc<Spec>);

impl ScriptFunction {
    /// The function's fields ([`decode_spec`]'s layout).
    pub(crate) fn decode(c: &mut Cursor<'_>) -> std::result::Result<Self, FfiStatus> {
        let spec = decode_spec(c)?;
        if spec.filter {
            return Err(invalid(
                "query tree: painless score function with a filter script",
            ));
        }
        spec.check()
            .map_err(|r| invalid(format!("query tree: painless script not checked: {r}")))?;
        Ok(ScriptFunction(Arc::new(spec)))
    }

    /// Whether the script reads `_score`.
    pub(crate) fn needs_score(&self) -> bool {
        self.0.needs_score
    }

    /// The doc-values fields the script reads.
    pub(crate) fn fields(&self) -> impl Iterator<Item = &str> {
        self.0.fields.iter().map(|f| f.read.as_str())
    }

    /// The script on segment `leaf`.
    pub(crate) fn leaf<'c>(&self, ctx: &ValuesContext<'c>, leaf: usize) -> Result<LeafScript<'c>> {
        Ok(LeafScript(scripts(&self.0, ctx, leaf, None)?))
    }
}

impl PartialEq for ScriptFunction {
    fn eq(&self, other: &Self) -> bool {
        self.0.ir_blob == other.0.ir_blob && self.0.params_blob == other.0.params_blob
    }
}

/// A [`ScriptFunction`] on one segment.
pub(crate) struct LeafScript<'c>(ScriptScores<'c>);

impl LeafScript<'_> {
    /// `LeafScoreFunction.score(docId, subQueryScore)`: the script's
    /// `double`, refused when negative (a `NaN` passes, as in Java).
    pub(crate) fn score(&mut self, doc: i32, sub: f32) -> Result<f64> {
        let s = &mut self.0;
        s.env.doc = doc;
        s.env.score = f64::from(sub);
        let v = s.run_value()?;
        if v < 0.0 {
            return Err(score_error(format!(
                "script score function must not produce negative scores, but got: [{}]",
                lucene_painless::number::double_to_string(v)
            )));
        }
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jvm_nodes::tests::{open, Blob};
    use crate::jvm_reader::ffi_close_jvm_reader;
    use crate::jvm_reader::tests::run_limit;
    use lucene_util::test_support::TempDir;

    /// The fixture's cases, read from `fixtures/data/painless-score/expected.bin`
    /// (`opensearch-plugin/tools/painless-corpus/ScoreCorpus.java`).
    struct Case {
        source: String,
        /// The node, when the plugin would send one.
        node: Option<Blob>,
        /// The node's fields after its sub-query: what the plugin checks.
        spec: Vec<u8>,
        /// The sub-query, and the boost over the node (1 for none).
        sub: Blob,
        boost: f32,
        /// A `script` query's filter script, without a sub-query.
        filter: bool,
        /// OpenSearch's hits by doc, or the exception it threw.
        outcome: Outcome,
    }

    #[derive(Debug)]
    enum Outcome {
        Hits(Vec<(i32, f32)>),
        /// The class `ScriptException` wraps, its message, the offset it
        /// points at.
        Threw(String, Option<String>, i32),
        /// `ScriptScorer`'s own error: a negative or NaN score.
        Refused(String),
    }

    struct Reader<'a>(&'a [u8]);

    impl Reader<'_> {
        fn u8(&mut self) -> u8 {
            let v = self.0[0];
            self.0 = &self.0[1..];
            v
        }
        fn i32(&mut self) -> i32 {
            let v = i32::from_le_bytes(self.0[..4].try_into().unwrap());
            self.0 = &self.0[4..];
            v
        }
        fn bytes(&mut self) -> Vec<u8> {
            let n = self.i32() as usize;
            let v = self.0[..n].to_vec();
            self.0 = &self.0[n..];
            v
        }
        fn str(&mut self) -> String {
            String::from_utf8(self.bytes()).unwrap()
        }
    }

    fn fixture(name: &str) -> String {
        format!(
            "{}/../../fixtures/data/painless-score/{name}",
            env!("CARGO_MANIFEST_DIR")
        )
    }

    fn cases() -> Vec<Case> {
        let data = std::fs::read(fixture("expected.bin")).unwrap();
        assert_eq!(&data[..4], b"PSC2");
        let mut r = Reader(&data[4..]);
        let n = r.i32();
        (0..n)
            .map(|_| {
                let source = r.str();
                let kind = r.u8();
                let filter = kind == 2;
                let sub = match kind {
                    0 => Blob::default().term("body", "alpha"),
                    1 => Blob::default().u8(5),
                    _ => Blob::default(),
                };
                let boost = f32::from_bits(r.i32() as u32);
                assert_eq!(r.u8(), 1, "{source}: every fixture script compiles to IR");
                let ir = r.bytes();
                let params = r.bytes();
                let needs = r.u8();
                let mut spec = Blob::default()
                    .u8(needs)
                    .bytes(&ir)
                    .bytes(&params)
                    .str("idx")
                    .i32(3)
                    .i32(0);
                let fields = r.i32();
                spec = spec.i32(fields);
                for _ in 0..fields {
                    let name = r.str();
                    spec = spec.str(&name).str(&name).u8(r.u8());
                }
                let mut b = Blob::tree();
                if boost != 1.0 {
                    b = b.u8(3).f32(boost);
                }
                let node = Some(if filter {
                    b.u8(NODE_PAINLESS_FILTER).raw(&spec)
                } else {
                    b.u8(NODE_PAINLESS_SCORE).raw(&sub).raw(&spec)
                });
                let case_sub = sub.clone();
                let outcome = match r.u8() {
                    0 => {
                        let n = r.i32();
                        Outcome::Hits(
                            (0..n)
                                .map(|_| (r.i32(), f32::from_bits(r.i32() as u32)))
                                .collect(),
                        )
                    }
                    kind => {
                        let class = r.str();
                        let message = (r.u8() == 1).then(|| r.str());
                        let offset = r.i32();
                        if kind == 1 {
                            Outcome::Threw(class, message, offset)
                        } else {
                            assert_eq!(class, "java.lang.IllegalArgumentException");
                            Outcome::Refused(message.unwrap())
                        }
                    }
                };
                Case {
                    source,
                    node,
                    spec: spec.0,
                    sub: case_sub,
                    boost,
                    filter,
                    outcome,
                }
            })
            .collect()
    }

    /// Scripts the check leaves to Java, before any of them runs.
    const IN_JAVA: &[&str] = &[
        // A date field's `JodaCompatibleZonedDateTime`.
        "doc['d'].value.toInstant().toEpochMilli() / 1e12",
        // `_source`.
        "params._source.size()",
        // Streams.
        "doc['vals'].stream().mapToLong(x -> x).sum()",
        // A field Painless does not see named: `params.doc[...]`.
        "params.doc['likes'].value + params._doc['flag'].value.compareTo(true) + 3",
        // A collection of `params` changed, which every segment's script shares.
        "params.weights.add(1); return params.weights.size()",
        // `params` changed through a second variable.
        "def p = params; def q = p; p['w'] = 5; return q.w",
        // A live view (`keySet`) changed.
        "Map m = ['a': 1, 'b': 2]; m.keySet().remove('a'); return m.size()",
    ];

    /// Every script the check passes runs as OpenSearch's does: the same
    /// hits and scores, or the same exception at the same offset, as a
    /// `script_score` query or, for a filter script, a `script` query. The ones
    /// it refuses never reach the native runtime: a node carrying one is
    /// refused before any document is read.
    #[test]
    fn scripts_score_as_opensearch_does() {
        let tmp = TempDir::new("painless-score");
        for e in std::fs::read_dir(fixture("index")).unwrap() {
            let e = e.unwrap();
            std::fs::copy(e.path(), tmp.path().join(e.file_name())).unwrap();
        }
        let h = open(&tmp);
        let (mut same, mut java) = (0, 0);
        let mut wrong = Vec::new();
        let cases = cases();
        for case in &cases {
            let node = case.node.as_ref().unwrap();
            let refused = check(&case.spec).unwrap();
            let got = run_limit(h, &node.0, 1000, i64::MAX);
            if let Some(why) = refused {
                if !IN_JAVA.contains(&case.source.as_str()) {
                    wrong.push(format!("{}: refused: {why}", case.source));
                    continue;
                }
                let e = crate::error::last_error();
                assert!(
                    got.is_err() && e.contains("not checked"),
                    "{}: {e}",
                    case.source
                );
                java += 1;
                continue;
            }
            assert!(
                !IN_JAVA.contains(&case.source.as_str()),
                "{}: passed the check",
                case.source
            );
            let e = crate::error::last_error();
            let bad = match (&case.outcome, got) {
                (Outcome::Hits(want), Ok((mut hits, _, _))) => {
                    hits.sort_by_key(|&(d, _)| d);
                    let bits = |v: &[(i32, f32)]| -> Vec<(i32, u32)> {
                        v.iter().map(|&(d, s)| (d, s.to_bits())).collect()
                    };
                    (bits(&hits) != bits(want)).then(|| format!("hits {hits:?}, want {want:?}"))
                }
                (Outcome::Hits(_), Err(_)) => Some(format!("failed: {e:?}")),
                (Outcome::Threw(class, message, offset), Err(_)) => {
                    let message = match message {
                        Some(m) => format!("1{m}"),
                        None => "0".to_string(),
                    };
                    // Then the segment, document and score it threw at.
                    let script = format!("{SCRIPT_ERROR}0\u{1}0\u{1}");
                    let want = format!("\u{1}{offset}\u{1}{class}\u{1}{message}");
                    (!e.contains(&script) || !e.ends_with(&want))
                        .then(|| format!("{e:?}, want {script:?}..{want:?}"))
                }
                (Outcome::Refused(message), Err(_)) => {
                    let want = format!("{SCORE_ERROR}{message}");
                    (!e.ends_with(&want)).then(|| format!("{e:?}, want {want:?}"))
                }
                (_, Ok(r)) => Some(format!("OpenSearch threw, native scored {r:?}")),
            };
            if let Some(bad) = bad {
                wrong.push(format!("{}: {bad}", case.source));
                continue;
            }
            same += 1;
        }
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
        assert_eq!(java, IN_JAVA.len());
        assert_eq!(same + java, cases.len());
        assert_eq!(ffi_close_jvm_reader(h), 0);
    }

    /// **`function_score` script functions.** Each fixture script the check
    /// passes, as the one function of a `function_score` that replaces the
    /// score: the same hits and scores as the script's `script_score` (both
    /// narrow the script's `double` to the hit's `float`), the same
    /// exception at the same offset. A negative score fails as
    /// `ScriptScoreFunction` does; `NaN` passes the function and fails the
    /// query.
    #[test]
    fn script_functions_score_as_script_score_does() {
        use crate::jvm_nodes::NODE_FUNCTION_SCORE;
        let tmp = TempDir::new("painless-score-function");
        for e in std::fs::read_dir(fixture("index")).unwrap() {
            let e = e.unwrap();
            std::fs::copy(e.path(), tmp.path().join(e.file_name())).unwrap();
        }
        let h = open(&tmp);
        let function = |sub: &Blob, spec: &[u8]| {
            Blob::tree()
                .u8(NODE_FUNCTION_SCORE)
                .u8(1) // replace
                .u8(0) // first
                .f32(f32::MAX)
                .raw(sub)
                .i32(1)
                .u8(0)
                .u8(5)
                .raw(&Blob(spec.to_vec()))
        };
        let mut wrong = Vec::new();
        let mut ran = 0;
        for case in cases() {
            if case.filter || case.boost != 1.0 || check(&case.spec).unwrap().is_some() {
                continue;
            }
            let got = run_limit(h, &function(&case.sub, &case.spec).0, 1000, i64::MAX);
            let e = crate::error::last_error();
            let bad = match (&case.outcome, got) {
                (Outcome::Hits(want), Ok((mut hits, _, _))) => {
                    hits.sort_by_key(|&(d, _)| d);
                    let bits = |v: &[(i32, f32)]| -> Vec<(i32, u32)> {
                        v.iter().map(|&(d, s)| (d, s.to_bits())).collect()
                    };
                    (bits(&hits) != bits(want)).then(|| format!("hits {hits:?}, want {want:?}"))
                }
                (Outcome::Threw(class, _, offset), Err(_)) => {
                    let want = format!("\u{1}{offset}\u{1}{class}\u{1}");
                    (!e.contains(SCRIPT_ERROR) || !e.contains(&want)).then(|| format!("{e:?}"))
                }
                (Outcome::Refused(message), Err(_)) => {
                    // ScriptScoreQuery's check, not the function's: the
                    // function passes NaN, and the query refuses it.
                    let ok = if message.contains("[NaN]") {
                        e.contains(OPENSEARCH_ERROR) && e.contains("invalid score: NaN for doc")
                    } else {
                        e.contains(SCORE_ERROR)
                            && e.contains("script score function must not produce negative scores")
                    };
                    (!ok).then(|| format!("{e:?}"))
                }
                (o, r) => Some(format!("{o:?} vs {r:?}: {e}")),
            };
            if let Some(bad) = bad {
                wrong.push(format!("{}: {bad}", case.source));
            }
            ran += 1;
        }
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
        assert!(ran > 30, "{ran} scripts run");
        assert_eq!(ffi_close_jvm_reader(h), 0);
    }

    /// `cargo test --release -p lucene-ffi --lib painless_check_cost --
    /// --ignored --nocapture`: the static check's cost per fixture script
    /// (decode, compile, specialise, check), once per distinct script and
    /// params, as the plugin caches the answer.
    #[test]
    #[ignore]
    fn painless_check_cost() {
        let mut all = Vec::new();
        for case in &cases() {
            let mut t: Vec<u128> = (0..200)
                .map(|_| {
                    let start = std::time::Instant::now();
                    std::hint::black_box(check(&case.spec).unwrap());
                    start.elapsed().as_nanos()
                })
                .collect();
            t.sort_unstable();
            all.push(t[t.len() / 2]);
            println!("{:>8} ns  {}", t[t.len() / 2], case.source);
        }
        all.sort_unstable();
        println!(
            "median {} ns, max {} ns",
            all[all.len() / 2],
            all[all.len() - 1]
        );
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    use crate::jvm_nodes::tests::{open, Blob};
    use crate::jvm_reader::tests::run_limit;
    use lucene_util::test_support::TempDir;

    fn unhex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    /// `PAINLESS_BENCH_INDEX=<dir> PAINLESS_BENCH_BLOBS=<tsv> cargo test
    /// --release -p lucene-ffi --lib painless_score_bench -- --ignored
    /// --nocapture`: each script's ns per document over a match-all query.
    #[test]
    #[ignore]
    fn painless_score_bench() {
        let from = std::env::var("PAINLESS_BENCH_INDEX").unwrap();
        let tmp = TempDir::new("painless-bench");
        for e in std::fs::read_dir(from).unwrap() {
            let e = e.unwrap();
            std::fs::copy(e.path(), tmp.path().join(e.file_name())).unwrap();
        }
        let h = open(&tmp);
        let time = |blob: &Blob| {
            let mut best = f64::MAX;
            let mut total = 0;
            for _ in 0..5 {
                let t = std::time::Instant::now();
                total = run_limit(h, &blob.0, 10, i64::MAX)
                    .unwrap_or_else(|_| panic!("{}", crate::error::last_error()))
                    .1;
                best = best.min(t.elapsed().as_nanos() as f64);
            }
            best / total as f64
        };
        let base = time(&Blob::tree().u8(5));
        println!("match_all\t{base:.2} ns/doc");
        let blobs =
            std::fs::read_to_string(std::env::var("PAINLESS_BENCH_BLOBS").unwrap()).unwrap();
        for line in blobs.lines() {
            let cols: Vec<&str> = line.split('\t').collect();
            let mut b = Blob::tree()
                .u8(super::NODE_PAINLESS_SCORE)
                .u8(5)
                .u8(1)
                .bytes(&unhex(cols[1]))
                .bytes(&unhex(cols[2]))
                .str("idx")
                .i32(0)
                .i32(0)
                .i32(cols.len() as i32 - 3);
            for f in &cols[3..] {
                let (name, kind) = f.split_once(':').unwrap();
                b = b.str(name).u8(kind.parse().unwrap());
            }
            let t = time(&b);
            println!("{}\t{:.2} ns/doc (script {:.2})", cols[0], t, t - base);
        }
    }

    struct OneAndHalf;

    impl DoubleValues for OneAndHalf {
        fn advance_exact(&mut self, _doc: i32) -> Result<bool> {
            Ok(true)
        }
        fn double_value(&mut self) -> Result<f64> {
            Ok(1.5)
        }
    }

    /// The script alone per document, as the Java harness times
    /// `setDocument` + `execute`: `PAINLESS_BENCH_INDEX` (one segment) and
    /// `PAINLESS_BENCH_BLOBS` as for [`painless_score_bench`].
    #[test]
    #[ignore]
    fn painless_script_loop_bench() {
        let dir = lucene_store::directory::FsDirectory::open(
            std::env::var("PAINLESS_BENCH_INDEX").unwrap(),
        );
        let reader = lucene_search::directory_reader::DirectoryReader::open(&dir).unwrap();
        let seg = &reader.segment_readers()[0];
        let ctx = ValuesContext::for_reader(seg);
        let blobs =
            std::fs::read_to_string(std::env::var("PAINLESS_BENCH_BLOBS").unwrap()).unwrap();
        for line in blobs.lines() {
            let cols: Vec<&str> = line.split('\t').collect();
            let ir_blob = unhex(cols[1]);
            let params_blob = unhex(cols[2]);
            let fields = cols[3..]
                .iter()
                .map(|f| {
                    let (name, kind) = f.split_once(':').unwrap();
                    let kind = match kind {
                        "0" => FieldKind::Long,
                        "1" => FieldKind::Double,
                        "5" => FieldKind::Keyword,
                        other => panic!("kind {other}"),
                    };
                    FieldSpec {
                        name: name.to_string(),
                        read: name.to_string(),
                        kind,
                    }
                })
                .collect();
            let source = PainlessScoreSource(Arc::new(Spec {
                script: ir::decode(&ir_blob).unwrap(),
                params: ir::decode_constant(&params_blob).unwrap(),
                ir_blob,
                params_blob,
                needs_score: true,
                filter: false,
                index: "idx".into(),
                shard: 0,
                script_id: 0,
                fields,
            }));
            let n = seg.max_doc;
            let mut best = f64::MAX;
            let mut sum = 0.0;
            for _ in 0..8 {
                let mut v = source
                    .get_values(&ctx, 0, Some(Box::new(OneAndHalf)))
                    .unwrap();
                let t = std::time::Instant::now();
                sum = 0.0;
                if std::env::var("PAINLESS_BENCH_PER_DOC").is_ok() {
                    for d in 0..n {
                        v.advance_exact(d).unwrap();
                        sum += v.double_value().unwrap();
                    }
                } else {
                    // The function score scorer's batches.
                    let (mut docs, scores) = (Vec::new(), vec![1.5f32; 64]);
                    let (mut out, mut has) = (vec![0.0; 64], vec![false; 64]);
                    for start in (0..n).step_by(64) {
                        docs.clear();
                        docs.extend(start..(start + 64).min(n));
                        v.fill_batch(&docs, &scores, &mut out, &mut has).unwrap();
                        sum += out[..docs.len()].iter().sum::<f64>();
                    }
                }
                best = best.min(t.elapsed().as_nanos() as f64 / f64::from(n));
            }
            println!("{}\t{best:.2} ns/doc\tsum={sum}", cols[0]);
        }
    }
}
