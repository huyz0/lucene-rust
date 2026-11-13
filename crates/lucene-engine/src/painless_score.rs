//! OpenSearch's `script_score` (`ScriptScoreQuery`) running a Painless
//! script natively, as node `43` of the query tree ([`crate::queries`]).
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
//! | context | `u8`: `0` score, `1` score reading `_score`, `2` filter, `3` number sort, `4` number sort reading `_score`, `5` aggregation, `6` `terms_set` minimum, `7` interval filter, `8` similarity, `9` similarity weight |
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
use lucene_search::explain::Explanation;
use lucene_search::extended_query::ExtendedQuery;
use lucene_search::function::{FunctionMatchQuery, FunctionScoreQuery};
use lucene_search::query::Clause;
use lucene_search::reader::doc_values::{
    direct_numeric, direct_sorted_numeric_singleton, get_sorted_numeric, get_sorted_set,
};
use lucene_search::reader::{SortedNumericDocValues, SortedSetDocValues};
use lucene_search::similarities::{CollectionStatistics, SimScorer, Similarity, TermStatistics};
use lucene_search::top_field::{
    FieldComparator, FieldComparatorSource, LeafCtx, LeafFieldComparator, SortValue,
};
use lucene_search::values_source::{
    BoxDoubleValues, BoxLongValues, DoubleValues, DoubleValuesSource, LongValues, LongValuesSource,
    ValuesContext,
};
use lucene_search::{Error, Result};
use lucene_util::half_float::sortable_short_to_half_float;
use lucene_util::numeric_utils::{sortable_int_to_float, sortable_long_to_double};

use crate::nodes::invalid;
use crate::query_blob::Cursor;
use crate::status::Status;

pub const NODE_PAINLESS_SCORE: u8 = 43;
pub const NODE_PAINLESS_FILTER: u8 = 45;

/// `ScriptQuery`'s `TwoPhaseIterator.matchCost()`.
pub const FILTER_MATCH_COST: f32 = 1000.0;

/// How a field's doc values read as `ScriptDocValues`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldKind {
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
    pub fn doc_kind(self) -> Option<DocKind> {
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
pub struct FieldSpec {
    /// The name the script reads the field by.
    pub name: String,
    /// The name its doc values are under: an alias's target.
    pub read: String,
    pub kind: FieldKind,
}

/// The script, for each segment to compile and run.
#[derive(Debug)]
pub struct PainlessScoreSource(pub Arc<Spec>);

#[derive(Debug)]
pub struct Spec {
    pub script: ir::Script,
    /// For [`DoubleValuesSource::equality_key`]: the blob the script came from.
    pub ir_blob: Vec<u8>,
    pub params: ir::Constant,
    pub params_blob: Vec<u8>,
    pub needs_score: bool,
    /// A `FilterScript`: `execute()` returns whether the document matches.
    pub filter: bool,
    /// A `NumberSortScript`: `execute()` returns the document's sort value.
    pub sort: bool,
    /// An `AggregationScript`: `execute()` returns the document's values.
    pub agg: bool,
    /// A `TermsSetQueryScript`: `execute()` returns the document's
    /// minimum number of matching terms.
    pub terms_set: bool,
    /// An `IntervalFilterScript`: `execute(interval)` returns whether the
    /// interval passes.
    pub interval: bool,
    /// A `SimilarityScript`: `execute(weight, query, field, term, doc)`
    /// returns a term's score.
    pub similarity: bool,
    /// A `SimilarityWeightScript`: `execute(query, field, term)` returns a
    /// term's weight.
    pub similarity_weight: bool,
    pub index: String,
    pub shard: i32,
    /// The plugin's number for the script, in the errors it raises.
    pub script_id: i32,
    /// Where `script_id` is in the blob: no part of the script's equality.
    pub script_id_at: usize,
    pub fields: Vec<FieldSpec>,
}

pub fn decode(
    c: &mut Cursor<'_>,
    depth: usize,
    nodes: &mut usize,
) -> std::result::Result<Clause, Status> {
    let sub = crate::query_blob::decode_node(c, depth + 1, nodes)?;
    let spec = decode_spec(c)?;
    if !spec.is_score() {
        return Err(invalid(
            "query tree: painless score node with another context's script",
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
pub fn decode_filter(c: &mut Cursor<'_>) -> std::result::Result<Clause, Status> {
    let spec = decode_spec(c)?;
    if !spec.filter {
        return Err(invalid(
            "query tree: painless filter node with another context's script",
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
pub fn decode_spec(c: &mut Cursor<'_>) -> std::result::Result<Spec, Status> {
    let (mut agg, mut sort, mut terms_set, mut interval) = (false, false, false, false);
    let (mut similarity, mut similarity_weight) = (false, false);
    let (needs_score, filter) = match c.u8()? {
        0 => (false, false),
        1 => (true, false),
        2 => (false, true),
        3 => {
            sort = true;
            (false, false)
        }
        4 => {
            sort = true;
            (true, false)
        }
        5 => {
            agg = true;
            (false, false)
        }
        6 => {
            terms_set = true;
            (false, false)
        }
        7 => {
            interval = true;
            (false, false)
        }
        8 => {
            similarity = true;
            (false, false)
        }
        9 => {
            similarity_weight = true;
            (false, false)
        }
        other => {
            return Err(invalid(format!(
                "query tree: painless script context {other} (expected 0..=9)"
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
    let script_id_at = c.pos();
    let script_id = c.i32()?;
    let n = c.length()?;
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
        sort,
        agg,
        terms_set,
        interval,
        similarity,
        similarity_weight,
        index,
        shard,
        script_id,
        script_id_at,
        fields,
    })
}

/// Whether the native runtime runs the script of `blob` (a node's fields
/// after its sub-query) exactly as Painless would: `None` when it does,
/// else why not.
pub fn check(blob: &[u8]) -> std::result::Result<Option<String>, Status> {
    let mut c = Cursor::new(blob);
    let spec = decode_spec(&mut c)?;
    if c.remaining() != 0 {
        return Err(invalid("painless check: trailing bytes"));
    }
    Ok(spec.check().err().map(|r| r.0))
}

/// A segment, in a script's error: by its number in the reader, or (where
/// only the segment is known, as to a sort key's comparator) by its doc
/// base, written `b<base>`.
#[derive(Clone, Copy)]
pub enum Leaf {
    Ord(usize),
    Base(i32),
}

impl std::fmt::Display for Leaf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Leaf::Ord(n) => write!(f, "{n}"),
            Leaf::Base(b) => write!(f, "b{b}"),
        }
    }
}

/// Where a script threw: the segment, the document and the score it ran
/// with.
#[derive(Clone, Copy)]
pub struct At {
    pub leaf: Leaf,
    pub doc: i32,
    pub score: f32,
}

/// Where an interval filter script threw: the interval, written
/// `start:end:gaps` where a document goes.
pub fn interval_error(script_id: i32, interval: (i32, i32, i32), offset: i32, t: &Throw) -> Error {
    let (start, end, gaps) = interval;
    slot_error(script_id, &format!("{start}:{end}:{gaps}"), offset, t)
}

/// A [`script_error`] of a script that runs on no document: `slot`, where
/// the document goes, says what it ran on.
pub fn slot_error(script_id: i32, slot: &str, offset: i32, t: &Throw) -> Error {
    if t.class() == lucene_painless::api::UNSUPPORTED {
        return runtime_failure(t);
    }
    let message = match t.message() {
        Some(m) => format!("1{m}"),
        None => "0".to_string(),
    };
    Error::IllegalArgument(format!(
        "{SCRIPT_ERROR}{script_id}\u{1}0\u{1}{slot}\u{1}0\u{1}{offset}\u{1}{}\u{1}{message}",
        t.class()
    ))
}

/// The error a script's exception fails the search with: the plugin raises
/// Painless's own `ScriptException` for the document `at` names, or else
/// one around the exception `class` (with `message`, `None` for none), at
/// source offset `offset`.
pub fn script_error(script_id: i32, at: At, offset: i32, t: &Throw) -> Error {
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
pub const SCRIPT_ERROR: &str = "\u{1}painless-script-error\u{1}";

/// What the message of `ScriptScorer`'s own `IllegalArgumentException` (a
/// negative or NaN score) starts with; the message follows.
pub const SCORE_ERROR: &str = "\u{1}painless-score-error\u{1}";

/// What the message of an `OpenSearchException` a query holding a script
/// raises (`function_score`'s missing value or invalid score) starts with.
pub const OPENSEARCH_ERROR: &str = "\u{1}opensearch-error\u{1}";

/// What an aggregation's `AggregationExecutionException` starts with: a
/// script value an aggregation cannot read.
pub const AGGREGATION_ERROR: &str = "\u{1}aggregation-error\u{1}";

/// An `IllegalArgumentException` with `message` that fails a search holding
/// a script without running it again in Java.
pub fn score_error(message: impl std::fmt::Display) -> Error {
    Error::IllegalArgument(format!("{SCORE_ERROR}{message}"))
}

/// An `OpenSearchException` with `message` that fails a search holding a
/// script without running it again in Java.
pub fn opensearch_error(message: impl std::fmt::Display) -> Error {
    Error::IllegalArgument(format!("{OPENSEARCH_ERROR}{message}"))
}

pub fn string(c: &mut Cursor<'_>) -> std::result::Result<String, Status> {
    Ok(std::str::from_utf8(c.bytes()?)
        .map_err(|_| Status::InvalidUtf8)?
        .to_string())
}

/// A script's exception as the search's error.
/// A failure of the runtime itself while running a checked script (an
/// unreadable segment): not the script's error, and not one Java would
/// answer differently.
pub fn runtime_failure(t: &Throw) -> Error {
    Error::IllegalArgument(format!(
        "painless runtime: {}: {}",
        t.class(),
        t.message().unwrap_or_default()
    ))
}

/// What `params._source` and `params._fields` hold: lookups the native
/// runtime does not have, so any use of one fails the search.
#[derive(Debug)]
pub struct Unavailable(pub &'static str);

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
    /// `name` over what the script runs: its IR, params, context, index,
    /// shard and fields, not its run number.
    pub fn key(&self, name: &str) -> String {
        let hex = |b: &[u8]| b.iter().map(|x| format!("{x:02x}")).collect::<String>();
        let fields: Vec<String> = self
            .fields
            .iter()
            .map(|f| format!("{}:{:?}", f.name, f.kind))
            .collect();
        format!(
            "{name}(ir={}, params={}, needs_score={}, filter={}, index={}, shard={}, fields=[{}])",
            hex(&self.ir_blob),
            hex(&self.params_blob),
            self.needs_score,
            self.filter,
            self.index,
            self.shard,
            fields.join(",")
        )
    }

    /// Whether the script is a `ScoreScript`.
    pub fn is_score(&self) -> bool {
        !self.filter
            && !self.sort
            && !self.agg
            && !self.terms_set
            && !self.interval
            && !self.similarity
            && !self.similarity_weight
    }

    /// Where a spec blob holds its script's run number.
    #[cfg(any(test, feature = "test-support"))]
    pub fn at_script_id(blob: &[u8]) -> usize {
        decode_spec(&mut Cursor::new(blob)).unwrap().script_id_at
    }

    /// The script class's own methods the score context serves.
    pub fn internals(&self) -> &'static [&'static str] {
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
        const SORT_WITH_SCORE: &[&str] = &["getParams", "getDoc", "get_score"];
        if self.interval || self.similarity || self.similarity_weight {
            &[]
        } else if self.filter || self.agg || self.terms_set || (self.sort && !self.needs_score) {
            FILTER
        } else if self.sort {
            SORT_WITH_SCORE
        } else if self.needs_score {
            WITH_SCORE
        } else {
            WITHOUT_SCORE
        }
    }

    /// [`lucene_painless::check`] on the program each segment runs.
    pub fn check(&self) -> std::result::Result<(), Refused> {
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
    pub fn params(&self, doc: &Value) -> std::result::Result<Value, Throw> {
        let ir::Constant::Map(entries) = &self.params else {
            unreachable!("decode checks the params are a map")
        };
        let mut m = JMap::for_copy(Order::Hash, entries.len());
        for (k, v) in entries {
            let k = lucene_painless::compile::constant(k).map_err(|e| missing(e.to_string()))?;
            let v = lucene_painless::compile::constant(v).map_err(|e| missing(e.to_string()))?;
            m.insert(k, v);
        }
        if self.filter || self.interval || self.similarity || self.similarity_weight {
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
pub enum Opened<'c> {
    Numeric(Box<dyn SortedNumericDocValues + 'c>),
    Set(Box<dyn SortedSetDocValues + 'c>),
}

/// A keyword slot `ScriptDocValues.Strings` grew but no document wrote:
/// an empty `BytesRefBuilder`.
pub const EMPTY_ORD: i64 = -1;

/// `ArrayUtil.oversize(min, bytes_per_element)` on a 64-bit JVM.
pub fn oversize(min: usize, bytes_per_element: usize) -> usize {
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
pub fn resize(items: &mut Vec<i64>, n: usize, kind: FieldKind) {
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
pub fn out_of_bounds(index: i32, len: usize) -> Throw {
    Throw::new(
        "java.lang.ArrayIndexOutOfBoundsException",
        format!("Index {index} out of bounds for length {len}"),
    )
}

/// Most keyword values a segment keeps as strings by ordinal.
pub const MAX_CACHED_TERMS: usize = 4096;

/// A field's values in the document last read.
pub struct Current {
    pub doc: i32,
    /// `ScriptDocValues`'s values array, which it reuses from document to
    /// document and grows as `ArrayUtil.grow` does: the document's values
    /// first, then whatever earlier documents left (a keyword's never
    /// written slots, [`EMPTY_ORD`], read as ""). Numbers by their bits: a
    /// long, a double, or 0 and 1; a keyword's ordinals.
    pub items: Vec<i64>,
    /// How many of `items` are the document's.
    pub count: usize,
    /// The `ScriptDocValues` view made of them, once asked for.
    pub view: Option<Value>,
}

/// The score context of one segment.
pub struct ScoreEnv<'c> {
    pub source: Arc<Spec>,
    pub reader: &'c lucene_search::directory_reader::SegmentReader,
    /// The segment, for errors.
    pub leaf: Leaf,
    /// Per field of `source.fields`: its doc values once read.
    pub opened: Vec<Option<Opened<'c>>>,
    /// Per field: its values in the document last read.
    pub current: Vec<Current>,
    /// Per keyword field: strings by ordinal.
    pub terms: Vec<HashMap<i64, Value>>,
    /// The documents of the batch [`BatchEnv`] reads, and their scores.
    pub batch: Vec<i32>,
    pub batch_scores: Vec<f32>,
    /// Per field: the batch's value counts and first values, once read.
    pub columns: Vec<Column>,
    /// Which batch `columns` hold.
    pub epoch: u64,
    /// Per field: its one-value-per-document column, read without an
    /// iterator, when the codec stores it so (`None` until first tried).
    pub direct: Vec<Option<Option<NumericReader<'c>>>>,
    pub doc: i32,
    pub score: f64,
    pub params: Value,
    pub doc_lookup: Value,
    pub index: Value,
}

/// A field's value counts and first values (decoded, by their bits) in
/// each document of a batch.
#[derive(Default)]
pub struct Column {
    pub epoch: u64,
    pub counts: Vec<u64>,
    pub firsts: Vec<u64>,
}

/// A numeric field's stored longs as its `ScriptDocValues` values' bits.
pub fn decode_column(kind: FieldKind, vs: &mut [u64]) {
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

pub fn no_value() -> Throw {
    Throw::new(
        "java.lang.IllegalStateException",
        lucene_painless::api::doc::NO_VALUE,
    )
}

impl ScoreEnv<'_> {
    pub fn field(&self, name: &str) -> Option<usize> {
        self.source.fields.iter().position(|f| f.name == name)
    }

    /// Field `i`'s values in the current document.
    pub fn load(&mut self, i: usize) -> std::result::Result<&Current, Throw> {
        if self.current[i].doc != self.doc {
            self.read(i)
                .map_err(|e| missing(format!("doc[{}]: {e}", self.source.fields[i].name)))?;
        }
        Ok(&self.current[i])
    }

    pub fn read(&mut self, i: usize) -> Result<()> {
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
    pub fn kind(&self, i: usize) -> std::result::Result<FieldKind, Throw> {
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
    pub fn item(&mut self, i: usize, index: i32) -> std::result::Result<i64, Throw> {
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

    pub fn term(&mut self, i: usize, ord: i64) -> std::result::Result<Value, Throw> {
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
    pub fn boxed_item(&mut self, i: usize, index: i32) -> std::result::Result<Value, Throw> {
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
    pub fn view(&mut self, i: usize) -> std::result::Result<Value, Throw> {
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
    pub fn column(&mut self, i: usize) -> std::result::Result<&Column, Throw> {
        if self.columns[i].epoch != self.epoch {
            self.read_column(i)
                .map_err(|e| missing(format!("doc[{}]: {e}", self.source.fields[i].name)))?;
        }
        Ok(&self.columns[i])
    }

    pub fn read_column(&mut self, i: usize) -> Result<()> {
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

    pub fn count(&mut self, i: usize) -> std::result::Result<i32, Throw> {
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
pub struct ScriptScores<'c> {
    /// The script on the typed machine, when it lowers to it.
    pub typed: Option<(Typed, Regs)>,
    pub prog: Rc<Program>,
    pub env: ScoreEnv<'c>,
    pub bindings: Vec<Option<Value>>,
    /// The VM's register files, kept across documents.
    pub pool: Vec<Vec<Value>>,
    /// The column-batch machine's registers.
    pub columns: BatchRegs,
    pub args: Vec<Value>,
    pub scores: Option<BoxDoubleValues<'c>>,
    pub value: f32,
}

impl ScriptScores<'_> {
    /// The score of the current document (`env.doc`, `env.score`), checked
    /// as `ScriptScorer` checks it; a filter's `1.0` or `0.0`.
    pub fn run(&mut self) -> Result<f32> {
        let doc = self.env.doc;
        let v = self.run_value()? as f32;
        if self.env.source.filter {
            return Ok(v);
        }
        checked(v, doc)
    }

    /// What the script returns for the current document.
    pub fn run_object(&mut self) -> Result<Value> {
        let id = self.env.source.script_id;
        let at = At {
            leaf: self.env.leaf,
            doc: self.env.doc,
            score: self.env.score as f32,
        };
        match &mut self.typed {
            Some((t, regs)) => t
                .run(&mut self.env, regs)
                .map_err(|e| script_error(id, at, t.thrown_at(regs), &e)),
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
        }
    }

    /// What the script returns for the current document, as a `double`.
    pub fn run_value(&mut self) -> Result<f64> {
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
pub fn checked(v: f32, doc: i32) -> Result<f32> {
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
    pub fn fill_columns(
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
        Ok(Box::new(scripts(
            &self.0,
            ctx.leaf_reader(leaf)?,
            Leaf::Ord(leaf),
            scores,
        )?))
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
        self.0.key("painless_score")
    }
}

/// The script compiled, specialised and ready to run on segment `leaf`.
pub fn scripts<'c>(
    spec: &Arc<Spec>,
    reader: &'c lucene_search::directory_reader::SegmentReader,
    leaf: Leaf,
    scores: Option<BoxDoubleValues<'c>>,
) -> Result<ScriptScores<'c>> {
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
pub struct ScriptFunction(pub Arc<Spec>);

impl ScriptFunction {
    /// The function's fields ([`decode_spec`]'s layout).
    pub fn decode(c: &mut Cursor<'_>) -> std::result::Result<Self, Status> {
        let spec = decode_spec(c)?;
        if !spec.is_score() {
            return Err(invalid(
                "query tree: painless score function with another context's script",
            ));
        }
        spec.check()
            .map_err(|r| invalid(format!("query tree: painless script not checked: {r}")))?;
        Ok(ScriptFunction(Arc::new(spec)))
    }

    /// Where the script's run number is in the node blob, which a node's
    /// equality leaves out (Java's `ScriptScoreFunction` compares the
    /// script alone).
    pub fn script_id_at(&self) -> usize {
        self.0.script_id_at
    }

    /// Whether the script reads `_score`.
    pub fn needs_score(&self) -> bool {
        self.0.needs_score
    }

    /// The doc-values fields the script reads.
    pub fn fields(&self) -> impl Iterator<Item = &str> {
        self.0.fields.iter().map(|f| f.read.as_str())
    }

    /// The script on segment `leaf`.
    pub fn leaf<'c>(&self, ctx: &ValuesContext<'c>, leaf: usize) -> Result<LeafScript<'c>> {
        Ok(LeafScript(scripts(
            &self.0,
            ctx.leaf_reader(leaf)?,
            Leaf::Ord(leaf),
            None,
        )?))
    }
}

impl PartialEq for ScriptFunction {
    fn eq(&self, other: &Self) -> bool {
        self.0.ir_blob == other.0.ir_blob && self.0.params_blob == other.0.params_blob
    }
}

/// A [`ScriptFunction`] on one segment.
pub struct LeafScript<'c>(pub ScriptScores<'c>);

impl LeafScript<'_> {
    /// `LeafScoreFunction.score(docId, subQueryScore)`: the script's
    /// `double`, refused when negative (a `NaN` passes, as in Java).
    pub fn score(&mut self, doc: i32, sub: f32) -> Result<f64> {
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

/// A `_script` sort key of type `number`: `ScriptSortBuilder`'s
/// `DoubleValuesComparatorSource` over a `NumberSortScript`, the script's
/// `double` for every document (never missing), compared as
/// `Double.compare` compares them. A hit's value is the double's sortable
/// long ([`lucene_util::numeric_utils::double_to_sortable_long`]).
#[derive(Debug)]
pub struct PainlessSort(pub Arc<Spec>);

impl PainlessSort {
    /// The key's script ([`decode_spec`]'s layout), checked.
    pub fn decode(c: &mut Cursor<'_>) -> std::result::Result<Self, Status> {
        let spec = decode_spec(c)?;
        if !spec.sort {
            return Err(invalid(
                "sort blob: painless sort key with another context's script",
            ));
        }
        spec.check()
            .map_err(|r| invalid(format!("sort blob: painless script not checked: {r}")))?;
        Ok(PainlessSort(Arc::new(spec)))
    }
}

impl FieldComparatorSource for PainlessSort {
    fn new_comparator(&self, _: &str, _: usize, _: bool) -> Box<dyn FieldComparator> {
        Box::new(SortComparator(Arc::clone(&self.0)))
    }

    fn value_string(&self, value: i64) -> String {
        lucene_painless::number::double_to_string(sortable_long_to_double(value))
    }
}

pub struct SortComparator(pub Arc<Spec>);

impl FieldComparator for SortComparator {
    fn leaf<'a>(&self, ctx: LeafCtx<'a>) -> Result<Box<dyn LeafFieldComparator + 'a>> {
        let leaf = scripts(&self.0, ctx.reader, Leaf::Base(ctx.doc_base), None)?;
        Ok(Box::new(SortLeaf(leaf)))
    }

    fn compare_values(&self, a: &SortValue, b: &SortValue) -> std::cmp::Ordering {
        match (a, b) {
            (SortValue::Long(a), SortValue::Long(b)) => a.cmp(b),
            _ => unreachable!("a script sort key's values are longs"),
        }
    }

    fn needs_scores(&self) -> bool {
        self.0.needs_score
    }
}

/// A [`PainlessSort`] key's script on one segment.
pub struct SortLeaf<'c>(pub ScriptScores<'c>);

impl LeafFieldComparator for SortLeaf<'_> {
    fn value(&mut self, doc: i32, score: f32) -> Result<SortValue> {
        let s = &mut self.0;
        s.env.doc = doc;
        s.env.score = f64::from(score);
        let v = s.run_value()?;
        Ok(SortValue::Long(
            lucene_util::numeric_utils::double_to_sortable_long(v),
        ))
    }
}

/// A numeric aggregation's script (`ValuesSource.Numeric.Script` over an
/// `AggregationScript`): per document what `ScriptDoubleValues` reads from
/// `execute()` -- nothing for `null` or an empty array or collection, a
/// number, a boolean as `1` or `0`, each element of an array or collection
/// -- sorted as `Double.compare` orders them, as sortable longs.
#[derive(Debug)]
pub struct PainlessAggValues(pub Arc<Spec>);

impl PainlessAggValues {
    /// The node's script ([`decode_spec`]'s layout), checked.
    pub fn decode(c: &mut Cursor<'_>) -> std::result::Result<Self, Status> {
        let spec = decode_spec(c)?;
        if !spec.agg {
            return Err(invalid(
                "aggregation tree: painless values with another context's script",
            ));
        }
        spec.check().map_err(|r| {
            invalid(format!(
                "aggregation tree: painless script not checked: {r}"
            ))
        })?;
        Ok(PainlessAggValues(Arc::new(spec)))
    }
}

impl lucene_aggs::bucket_aggs::ScriptValuesSource for PainlessAggValues {
    fn open<'a>(
        &self,
        reader: &'a lucene_search::directory_reader::SegmentReader,
    ) -> Result<Box<dyn lucene_aggs::bucket_aggs::ScriptValues + 'a>> {
        let leaf = scripts(&self.0, reader, Leaf::Base(reader.doc_base), None)?;
        Ok(Box::new(AggLeaf(leaf)))
    }
}

/// A [`PainlessAggValues`] on one segment.
pub struct AggLeaf<'c>(pub ScriptScores<'c>);

/// `ScriptDoubleValues.toDoubleValue`.
pub fn agg_double(v: &Value) -> Result<f64> {
    Ok(match v {
        Value::Bool(b) => f64::from(u8::from(*b)),
        Value::Byte(n) => f64::from(*n),
        Value::Short(n) => f64::from(*n),
        Value::Int(n) => f64::from(*n),
        Value::Long(n) => *n as f64,
        Value::Float(n) => f64::from(*n),
        Value::Double(n) => *n,
        other => {
            return Err(Error::IllegalArgument(format!(
            "{AGGREGATION_ERROR}Unsupported script value [{}], expected a number, date, or boolean",
            other.java_to_string()
        )))
        }
    })
}

impl lucene_aggs::bucket_aggs::ScriptValues for AggLeaf<'_> {
    fn values(&mut self, doc: i32, out: &mut Vec<i64>) -> Result<()> {
        let s = &mut self.0;
        s.env.doc = doc;
        let v = s.run_object()?;
        let from = out.len();
        let items: Vec<Value> = match &v {
            Value::Null => return Ok(()),
            Value::List(l) => l.borrow().items.clone(),
            Value::Set(l) => l.borrow().map.iter().map(|(k, _)| k.clone()).collect(),
            Value::Array(a) => a.borrow().items.clone(),
            Value::Object(o) => match o.as_any().downcast_ref::<DocValues>() {
                Some(d) => d.items.clone(),
                None => vec![v.clone()],
            },
            _ => vec![v.clone()],
        };
        for item in &items {
            out.push(lucene_util::numeric_utils::double_to_sortable_long(
                agg_double(item)?,
            ));
        }
        out[from..].sort_unstable();
        Ok(())
    }
}

/// `terms_set`'s `minimum_should_match_script`
/// (`TermsSetQueryBuilder.ScriptLongValueSource`), checked: each document's
/// minimum is `execute().longValue()`, none when the script returns `null`.
#[derive(Debug)]
pub struct PainlessMinimum(pub Arc<Spec>);

impl PainlessMinimum {
    /// The covering node's script ([`decode_spec`]'s layout), checked.
    pub fn decode(c: &mut Cursor<'_>) -> std::result::Result<Self, Status> {
        let spec = decode_spec(c)?;
        if !spec.terms_set {
            return Err(invalid(
                "query tree: a covering minimum with another context's script",
            ));
        }
        spec.check()
            .map_err(|r| invalid(format!("query tree: painless script not checked: {r}")))?;
        Ok(PainlessMinimum(Arc::new(spec)))
    }
}

impl LongValuesSource for PainlessMinimum {
    fn get_values<'c>(
        &self,
        ctx: &ValuesContext<'c>,
        leaf: usize,
        _scores: Option<BoxDoubleValues<'c>>,
    ) -> Result<BoxLongValues<'c>> {
        Ok(Box::new(MinimumLeaf {
            script: scripts(&self.0, ctx.leaf_reader(leaf)?, Leaf::Ord(leaf), None)?,
            current: 0,
        }))
    }

    fn needs_scores(&self) -> bool {
        false
    }

    /// `ScriptLongValueSource.isCacheable`: never.
    fn is_cacheable(&self, _ctx: &ValuesContext<'_>, _leaf: usize) -> bool {
        false
    }

    /// The script's identity: `ScriptLongValueSource.equals` compares
    /// the script, and `CoveringQuery`'s equality compares this.
    fn describe(&self) -> String {
        self.0.key("painless_minimum")
    }
}

/// A [`PainlessMinimum`] on one segment.
pub struct MinimumLeaf<'c> {
    pub script: ScriptScores<'c>,
    pub current: i64,
}

/// `Number.longValue()` of what a `TermsSetQueryScript` returns.
pub fn long_value(v: &Value) -> Result<i64> {
    Ok(match v {
        Value::Byte(n) => i64::from(*n),
        Value::Short(n) => i64::from(*n),
        Value::Int(n) => i64::from(*n),
        Value::Long(n) => *n,
        // `(long)`: saturating, NaN to 0, as Rust's cast is.
        Value::Float(n) => *n as i64,
        Value::Double(n) => *n as i64,
        other => {
            return Err(Error::IllegalArgument(format!(
                "painless runtime: a terms_set minimum {}",
                other.java_to_string()
            )))
        }
    })
}

impl LongValues for MinimumLeaf<'_> {
    fn advance_exact(&mut self, doc: i32) -> Result<bool> {
        self.script.env.doc = doc;
        let v = self.script.run_object()?;
        if matches!(v, Value::Null) {
            return Ok(false);
        }
        self.current = long_value(&v)?;
        Ok(true)
    }

    fn long_value(&mut self) -> Result<i64> {
        Ok(self.current)
    }
}

/// An `intervals` query's `script` filter (OpenSearch's
/// `ScriptFilterSource`), checked.
#[derive(Debug)]
pub struct PainlessInterval(pub Arc<Spec>);

impl PainlessInterval {
    /// The filter's script ([`decode_spec`]'s layout), checked.
    pub fn decode(c: &mut Cursor<'_>) -> std::result::Result<Self, Status> {
        let spec = decode_spec(c)?;
        if !spec.interval {
            return Err(invalid(
                "query tree: an interval filter with another context's script",
            ));
        }
        if !spec.fields.is_empty() {
            return Err(invalid(
                "query tree: an interval filter script reads no fields",
            ));
        }
        spec.check()
            .map_err(|r| invalid(format!("query tree: painless script not checked: {r}")))?;
        Ok(PainlessInterval(Arc::new(spec)))
    }
}

impl lucene_search::intervals::IntervalScript for PainlessInterval {
    fn accepter(&self) -> Result<Box<dyn lucene_search::intervals::IntervalAccept>> {
        let spec = &self.0;
        let mut prog = lucene_painless::compile::compile(&spec.script)
            .map_err(|e| Error::IllegalArgument(format!("painless script: {e}")))?;
        let params = spec.params(&Value::Null).map_err(|t| runtime_failure(&t))?;
        specialize(
            &mut prog,
            &Facts {
                params: Some(&params),
                fields: &[],
            },
        );
        Ok(Box::new(IntervalAccepter {
            bindings: vec![None; prog.binding_sites], // alloc-ok: the compiled program's own count
            prog: Rc::new(prog),
            env: lucene_painless::vm::ParamsEnv { params },
            pool: Vec::new(),
            script_id: spec.script_id,
        }))
    }
}

/// A [`PainlessInterval`] ready to run.
pub struct IntervalAccepter {
    pub prog: Rc<Program>,
    pub env: lucene_painless::vm::ParamsEnv,
    pub bindings: Vec<Option<Value>>,
    pub pool: Vec<Vec<Value>>,
    pub script_id: i32,
}

/// `IntervalFilterScript.Interval`: the interval the filter is looking at.
#[derive(Debug)]
pub struct Interval {
    pub start: i32,
    pub end: i32,
    pub gaps: i32,
}

impl JObject for Interval {
    fn class_name(&self) -> &str {
        "org.opensearch.index.query.IntervalFilterScript$Interval"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn call(
        &self,
        _vm: &mut Vm,
        _this: &Value,
        name: &str,
        args: &[Value],
    ) -> Option<std::result::Result<Value, Throw>> {
        if !args.is_empty() {
            return None;
        }
        Some(Ok(Value::Int(match name {
            "getStart" => self.start,
            "getEnd" => self.end,
            "getGaps" => self.gaps,
            _ => return None,
        })))
    }
    /// The getter shortcuts: `interval.start` for `getStart()`.
    fn field(&self, _vm: &mut Vm, name: &str) -> Option<std::result::Result<Value, Throw>> {
        Some(Ok(Value::Int(match name {
            "start" => self.start,
            "end" => self.end,
            "gaps" => self.gaps,
            _ => return None,
        })))
    }
}

impl lucene_search::intervals::IntervalAccept for IntervalAccepter {
    fn accept(&mut self, start: i32, end: i32, gaps: i32) -> Result<bool> {
        let interval = Value::object(Interval { start, end, gaps });
        let mut vm = Vm::new(self.prog.clone(), &mut self.env);
        vm.bindings = std::mem::take(&mut self.bindings);
        vm.pool = std::mem::take(&mut self.pool);
        let r = vm.execute(&[interval]);
        let loc = vm.loc;
        self.bindings = std::mem::take(&mut vm.bindings);
        self.pool = std::mem::take(&mut vm.pool);
        match r {
            Ok(Value::Bool(b)) => Ok(b),
            Ok(other) => Err(Error::IllegalArgument(format!(
                "painless runtime: an interval filter returned {}",
                other.java_to_string()
            ))),
            Err(t) => Err(interval_error(self.script_id, (start, end, gaps), loc, &t)),
        }
    }
}

/// OpenSearch's `ScriptedSimilarity`: a field's similarity whose term
/// scores are a Painless `SimilarityScript`, each term's weight an optional
/// `SimilarityWeightScript`.
///
/// `SimScorer.score` returns a `float` and has no way to fail, so a
/// script's exception is kept, the first one, for the search to fail with
/// once it ends ([`Similarity::take_failure`]); the scores after it are `0` and
/// run no script. The plugin raises it as Painless's `ScriptException`,
/// running the script once more in Java on the values the error names.
///
/// The similarity, after its kind byte in the similarity table
/// ([`crate::similarity`]):
///
/// | field | layout |
/// |---|---|
/// | discount overlaps | `u8` |
/// | negative scores | `u8`: [`NegativeScores`] |
/// | description | string: `ScriptedSimilarity.toString()` |
/// | weight script | `u8` `0` none, or `1` and a spec of context `9` |
/// | script | a spec of context `8` |
#[derive(Debug, Clone)]
pub struct ScriptedSimilarity(pub Arc<Scripted>);

#[derive(Debug)]
pub struct Scripted {
    pub script: Spec,
    pub weight: Option<Spec>,
    pub discount_overlaps: bool,
    pub negative: NegativeScores,
    pub description: String,
    /// Whether a script failed: read before every score.
    pub failed: std::sync::atomic::AtomicBool,
    /// The first error, with when it was met ([`next_order`]).
    pub failure: std::sync::Mutex<Option<(u64, Error)>>,
    /// Each term's weight by its statistics: computed once per query, as
    /// Java's `createWeight` does, though each segment makes its scorers.
    pub weights: std::sync::Mutex<HashMap<[u64; 6], f64>>,
    /// This similarity's compiled score script in each thread's [`PROGRAMS`].
    pub id: u64,
}

impl Scripted {
    pub fn fail(&self, e: Error) {
        let mut slot = self.failure.lock().unwrap_or_else(|p| p.into_inner());
        if slot.is_none() {
            *slot = Some((next_order(), e));
        }
        self.failed
            .store(true, std::sync::atomic::Ordering::Release);
    }

    pub fn has_failed(&self) -> bool {
        self.failed.load(std::sync::atomic::Ordering::Acquire)
    }
}

impl ScriptedSimilarity {
    /// The similarity after its kind byte, its scripts checked.
    pub fn decode(c: &mut Cursor<'_>) -> std::result::Result<Self, Status> {
        let discount_overlaps = c.u8()? != 0;
        let negative = match c.u8()? {
            0 => NegativeScores::Kept,
            1 => NegativeScores::Fail,
            2 => NegativeScores::Zero,
            other => {
                return Err(invalid(format!(
                    "similarity table: negative scores mode {other} (expected 0 to 2)"
                )))
            }
        };
        let description = string(c)?;
        let weight = match c.u8()? {
            0 => None,
            1 => Some(similarity_spec(c, true)?),
            other => {
                return Err(invalid(format!(
                    "similarity table: weight script flag {other} (expected 0 or 1)"
                )))
            }
        };
        let script = similarity_spec(c, false)?;
        Ok(ScriptedSimilarity(Arc::new(Scripted {
            script,
            weight,
            discount_overlaps,
            negative,
            description,
            failed: std::sync::atomic::AtomicBool::new(false),
            failure: std::sync::Mutex::new(None),
            weights: std::sync::Mutex::new(HashMap::new()),
            id: next_scorer_id(),
        })))
    }
}

/// What a scripted similarity's negative score does. OpenSearch's
/// `SimilarityService` wraps every `scripted` similarity in
/// `NonNegativeScoresSimilarity`, which fails the search on one unless the
/// `opensearch.enforce.positive.scores` system property is `false`, and then
/// scores it `0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NegativeScores {
    /// Unwrapped: the score as the script returned it.
    Kept,
    /// `NonNegativeScoresSimilarity` enforcing: an `IllegalArgumentException`.
    Fail,
    /// `NonNegativeScoresSimilarity` not enforcing: `0`.
    Zero,
}

/// A similarity script's spec: of the weight context or the score one, no
/// fields, checked.
pub fn similarity_spec(c: &mut Cursor<'_>, weight: bool) -> std::result::Result<Spec, Status> {
    let spec = decode_spec(c)?;
    if (weight && !spec.similarity_weight) || (!weight && !spec.similarity) {
        return Err(invalid(
            "similarity table: a scripted similarity with another context's script",
        ));
    }
    if !spec.fields.is_empty() {
        return Err(invalid(
            "similarity table: a similarity script reads no fields",
        ));
    }
    spec.check().map_err(|r| {
        invalid(format!(
            "similarity table: painless script not checked: {r}"
        ))
    })?;
    Ok(spec)
}

/// `spec` compiled and specialised to its (empty) params.
pub fn similarity_program(spec: &Spec) -> Result<(Program, lucene_painless::vm::ParamsEnv)> {
    let mut prog = lucene_painless::compile::compile(&spec.script)
        .map_err(|e| Error::IllegalArgument(format!("painless script: {e}")))?;
    let params = spec.params(&Value::Null).map_err(|t| runtime_failure(&t))?;
    specialize(
        &mut prog,
        &Facts {
            params: Some(&params),
            fields: &[],
        },
    );
    Ok((prog, lucene_painless::vm::ParamsEnv { params }))
}

impl lucene_index::similarity::NormSimilarity for ScriptedSimilarity {
    /// `ScriptedSimilarity.computeNorm`: `SmallFloat.intToByte4` of the
    /// length, less the overlaps when they are discounted (a `DOCS` field
    /// too, unlike `Similarity`'s default).
    fn compute_norm(
        &self,
        _field: &str,
        state: &lucene_index::similarity::FieldInvertState,
    ) -> i64 {
        let n = if self.0.discount_overlaps {
            state.length.wrapping_sub(state.num_overlap)
        } else {
            state.length
        };
        i64::from(lucene_util::small_float::int_to_byte4(n as u32) as i8)
    }

    fn discount_overlaps(&self) -> bool {
        self.0.discount_overlaps
    }
}

/// A term's statistics, as `ScriptedSimilarity.scorer` hands its scripts
/// them.
#[derive(Debug, Clone, Copy)]
pub struct SimStats {
    pub boost: f32,
    pub doc_count: i64,
    pub sum_doc_freq: i64,
    pub sum_total_term_freq: i64,
    pub doc_freq: i64,
    pub total_term_freq: i64,
}

impl SimStats {
    /// `query`, `field` and `term`, as the scripts' arguments.
    pub fn objects(self) -> [Value; 3] {
        [
            Value::object(SimQuery(self.boost)),
            Value::object(SimField {
                doc_count: self.doc_count,
                sum_doc_freq: self.sum_doc_freq,
                sum_total_term_freq: self.sum_total_term_freq,
            }),
            Value::object(SimTerm {
                doc_freq: self.doc_freq,
                total_term_freq: self.total_term_freq,
            }),
        ]
    }

    /// The statistics as bits: a [`Scripted::weights`] key.
    pub fn key(self) -> [u64; 6] {
        [
            u64::from(self.boost.to_bits()),
            self.doc_count as u64,
            self.sum_doc_freq as u64,
            self.sum_total_term_freq as u64,
            self.doc_freq as u64,
            self.total_term_freq as u64,
        ]
    }

    /// The statistics where a [`slot_error`] names what the script ran on.
    pub fn slot(self) -> String {
        format!(
            "{}:{}:{}:{}:{}:{}",
            self.boost.to_bits(),
            self.doc_count,
            self.sum_doc_freq,
            self.sum_total_term_freq,
            self.doc_freq,
            self.total_term_freq
        )
    }
}

/// A getter of a similarity script's argument: `name()` with no arguments.
pub fn getter(args: &[Value], value: Option<Value>) -> Option<std::result::Result<Value, Throw>> {
    if args.is_empty() {
        value.map(Ok)
    } else {
        None
    }
}

/// `ScriptedSimilarity.Query`.
#[derive(Debug)]
pub struct SimQuery(pub f32);

impl JObject for SimQuery {
    fn class_name(&self) -> &str {
        "org.opensearch.index.similarity.ScriptedSimilarity$Query"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn call(
        &self,
        _: &mut Vm,
        _: &Value,
        name: &str,
        args: &[Value],
    ) -> Option<std::result::Result<Value, Throw>> {
        getter(args, (name == "getBoost").then_some(Value::Float(self.0)))
    }
    fn field(&self, _: &mut Vm, name: &str) -> Option<std::result::Result<Value, Throw>> {
        (name == "boost").then_some(Ok(Value::Float(self.0)))
    }
}

/// `ScriptedSimilarity.Field`.
#[derive(Debug)]
pub struct SimField {
    pub doc_count: i64,
    pub sum_doc_freq: i64,
    pub sum_total_term_freq: i64,
}

impl SimField {
    pub fn get(&self, property: &str) -> Option<Value> {
        Some(Value::Long(match property {
            "docCount" => self.doc_count,
            "sumDocFreq" => self.sum_doc_freq,
            "sumTotalTermFreq" => self.sum_total_term_freq,
            _ => return None,
        }))
    }
}

impl JObject for SimField {
    fn class_name(&self) -> &str {
        "org.opensearch.index.similarity.ScriptedSimilarity$Field"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn call(
        &self,
        _: &mut Vm,
        _: &Value,
        name: &str,
        args: &[Value],
    ) -> Option<std::result::Result<Value, Throw>> {
        getter(
            args,
            name.strip_prefix("get")
                .and_then(|p| self.get(&decapitalize(p))),
        )
    }
    fn field(&self, _: &mut Vm, name: &str) -> Option<std::result::Result<Value, Throw>> {
        self.get(name).map(Ok)
    }
}

/// `ScriptedSimilarity.Term`.
#[derive(Debug)]
pub struct SimTerm {
    pub doc_freq: i64,
    pub total_term_freq: i64,
}

impl SimTerm {
    pub fn get(&self, property: &str) -> Option<Value> {
        Some(Value::Long(match property {
            "docFreq" => self.doc_freq,
            "totalTermFreq" => self.total_term_freq,
            _ => return None,
        }))
    }
}

impl JObject for SimTerm {
    fn class_name(&self) -> &str {
        "org.opensearch.index.similarity.ScriptedSimilarity$Term"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn call(
        &self,
        _: &mut Vm,
        _: &Value,
        name: &str,
        args: &[Value],
    ) -> Option<std::result::Result<Value, Throw>> {
        getter(
            args,
            name.strip_prefix("get")
                .and_then(|p| self.get(&decapitalize(p))),
        )
    }
    fn field(&self, _: &mut Vm, name: &str) -> Option<std::result::Result<Value, Throw>> {
        self.get(name).map(Ok)
    }
}

/// A getter's property: `DocCount` as `docCount`.
pub fn decapitalize(p: &str) -> String {
    let mut c = p.chars();
    match c.next() {
        Some(f) => f.to_lowercase().chain(c).collect(),
        None => String::new(),
    }
}

/// `ScriptedSimilarity.Doc`: the document a term's scorer is on, set before
/// each run.
#[derive(Debug, Default)]
pub struct SimDoc {
    pub freq: std::cell::Cell<f32>,
    pub norm: std::cell::Cell<i64>,
}

impl SimDoc {
    /// `getLength()`: `SmallFloat.byte4ToInt((byte) norm)`.
    pub fn length(&self) -> i32 {
        lucene_util::small_float::byte4_to_int(self.norm.get() as u8) as i32
    }

    pub fn get(&self, property: &str) -> Option<Value> {
        match property {
            "length" => Some(Value::Int(self.length())),
            "freq" => Some(Value::Float(self.freq.get())),
            _ => None,
        }
    }
}

impl JObject for SimDoc {
    fn class_name(&self) -> &str {
        "org.opensearch.index.similarity.ScriptedSimilarity$Doc"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn call(
        &self,
        _: &mut Vm,
        _: &Value,
        name: &str,
        args: &[Value],
    ) -> Option<std::result::Result<Value, Throw>> {
        getter(
            args,
            name.strip_prefix("get")
                .and_then(|p| self.get(&decapitalize(p))),
        )
    }
    fn field(&self, _: &mut Vm, name: &str) -> Option<std::result::Result<Value, Throw>> {
        self.get(name).map(Ok)
    }
}

impl Similarity for ScriptedSimilarity {
    /// `ScriptedSimilarity.scorer`: per term, its weight from the weight
    /// script (`1` without one), then a scorer running the script; the sum
    /// of the terms' scores, in `double`, for more than one.
    fn scorer(
        &self,
        _field: &str,
        boost: f32,
        collection: &CollectionStatistics,
        terms: &[TermStatistics],
    ) -> Arc<dyn SimScorer> {
        let doc_count = if collection.doc_count == -1 {
            collection.max_doc
        } else {
            collection.doc_count
        };
        let mut scorers: Vec<ScriptedScorer> = terms
            .iter()
            .map(|t| {
                let stats = SimStats {
                    boost,
                    doc_count,
                    sum_doc_freq: collection.sum_doc_freq,
                    sum_total_term_freq: collection.sum_total_term_freq,
                    doc_freq: t.doc_freq,
                    total_term_freq: t.total_term_freq,
                };
                ScriptedScorer {
                    weight: self.weight(stats),
                    stats,
                    shared: self.0.clone(),
                    id: next_scorer_id(),
                }
            })
            .collect();
        let scorer: Arc<dyn SimScorer> = if scorers.len() == 1 {
            Arc::new(scorers.pop().expect("one scorer"))
        } else {
            Arc::new(SummedScorer(scorers))
        };
        match self.0.negative {
            NegativeScores::Kept => scorer,
            _ => Arc::new(NonNegativeScorer {
                inner: scorer,
                shared: self.0.clone(),
            }),
        }
    }

    fn simple_name(&self) -> &str {
        "ScriptedSimilarity"
    }

    fn shared(&self) -> Option<Arc<dyn Similarity>> {
        Some(Arc::new(self.clone()))
    }

    fn take_failure(&self) -> Option<Error> {
        if !self.0.has_failed() {
            return None;
        }
        self.0
            .failure
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .take()
            .map(|(_, e)| e)
    }

    fn failure_order(&self) -> Option<u64> {
        if !self.0.has_failed() {
            return None;
        }
        self.0
            .failure
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
            .map(|&(o, _)| o)
    }
}

impl ScriptedSimilarity {
    /// `computeWeight`: the weight script's value, or `1`.
    pub fn weight(&self, stats: SimStats) -> f64 {
        let Some(spec) = &self.0.weight else {
            return 1.0;
        };
        if self.0.has_failed() {
            return 0.0;
        }
        let key = stats.key();
        let memo = || self.0.weights.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(&w) = memo().get(&key) {
            return w;
        }
        let run = || -> Result<f64> {
            let (prog, mut env) = similarity_program(spec)?;
            let prog = Rc::new(prog);
            let mut vm = Vm::new(prog.clone(), &mut env);
            vm.bindings = vec![None; prog.binding_sites]; // alloc-ok: the compiled program's own count
            match vm.execute(&stats.objects()) {
                Ok(v) => script_double(&v),
                Err(t) => Err(slot_error(spec.script_id, &stats.slot(), vm.loc, &t)),
            }
        };
        match run() {
            Ok(w) => {
                memo().insert(key, w);
                w
            }
            Err(e) => {
                self.0.fail(e);
                0.0
            }
        }
    }
}

/// A similarity script's `double` result.
pub fn script_double(v: &Value) -> Result<f64> {
    match v {
        Value::Double(d) => Ok(*d),
        other => Err(Error::IllegalArgument(format!(
            "painless runtime: a similarity script returned {}",
            other.java_to_string()
        ))),
    }
}

pub fn next_scorer_id() -> u64 {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// When a scripted similarity failed, in one order across every one of them.
pub fn next_order() -> u64 {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// One term's scorer: `ScriptedSimilarity`'s anonymous `SimScorer`.
#[derive(Debug)]
pub struct ScriptedScorer {
    pub shared: Arc<Scripted>,
    pub weight: f64,
    pub stats: SimStats,
    /// This scorer's state in each thread's [`SCORERS`].
    pub id: u64,
}

/// A [`ScriptedScorer`]'s compiled script and arguments on one thread:
/// the runtime's values are not `Send`, so each thread that scores makes
/// its own.
pub struct ScorerState {
    pub prog: Rc<Program>,
    pub env: lucene_painless::vm::ParamsEnv,
    pub bindings: Vec<Option<Value>>,
    pub pool: Vec<Vec<Value>>,
    pub doc: Rc<SimDoc>,
    pub args: [Value; 5],
}

/// At most this many scorers' states per thread; past it the thread starts
/// over.
pub const MAX_SCORER_STATES: usize = 256;

thread_local! {
    static SCORERS: std::cell::RefCell<HashMap<u64, ScorerState>> =
        std::cell::RefCell::new(HashMap::new());
    /// Each scripted similarity's compiled score script, by its id: compiled
    /// once per thread, not per scorer.
    static PROGRAMS: std::cell::RefCell<HashMap<u64, Rc<Program>>> =
        std::cell::RefCell::new(HashMap::new());
}

/// Makes room for one more entry in a per-thread `map` at most
/// [`MAX_SCORER_STATES`] long: one entry goes, not all of them.
pub fn make_room<V>(map: &mut HashMap<u64, V>) {
    if map.len() >= MAX_SCORER_STATES {
        if let Some(&k) = map.keys().next() {
            map.remove(&k);
        }
    }
}

impl ScriptedScorer {
    pub fn state(&self) -> Result<ScorerState> {
        let id = self.shared.id;
        let cached = PROGRAMS.with(|p| p.borrow().get(&id).cloned());
        let (prog, env) = match cached {
            // Its own params: a script may change them.
            Some(prog) => {
                let params = self
                    .shared
                    .script
                    .params(&Value::Null)
                    .map_err(|t| runtime_failure(&t))?;
                (prog, lucene_painless::vm::ParamsEnv { params })
            }
            None => {
                let (prog, env) = similarity_program(&self.shared.script)?;
                let prog = Rc::new(prog);
                PROGRAMS.with(|p| {
                    let mut p = p.borrow_mut();
                    make_room(&mut p);
                    p.insert(id, prog.clone());
                });
                (prog, env)
            }
        };
        let doc = Rc::new(SimDoc::default());
        let [query, field, term] = self.stats.objects();
        Ok(ScorerState {
            bindings: vec![None; prog.binding_sites], // alloc-ok: the compiled program's own count
            prog,
            env,
            pool: Vec::new(),
            args: [
                Value::Double(self.weight),
                query,
                field,
                term,
                Value::Object(doc.clone()),
            ],
            doc,
        })
    }

    pub fn run(&self, freq: f32, norm: i64) -> Result<f32> {
        SCORERS.with(|states| {
            let mut states = states.borrow_mut();
            if !states.contains_key(&self.id) {
                make_room(&mut states);
                states.insert(self.id, self.state()?);
            }
            let st = states.get_mut(&self.id).expect("inserted above");
            st.doc.freq.set(freq);
            st.doc.norm.set(norm);
            let mut vm = Vm::new(st.prog.clone(), &mut st.env);
            vm.bindings = std::mem::take(&mut st.bindings);
            vm.pool = std::mem::take(&mut st.pool);
            let r = vm.execute(&st.args);
            let loc = vm.loc;
            st.bindings = std::mem::take(&mut vm.bindings);
            st.pool = std::mem::take(&mut vm.pool);
            match r {
                Ok(v) => Ok(script_double(&v)? as f32),
                Err(t) => Err(slot_error(
                    self.shared.script.script_id,
                    &format!(
                        "{}:{}:{}:{}",
                        self.weight.to_bits(),
                        self.stats.slot(),
                        freq.to_bits(),
                        norm
                    ),
                    loc,
                    &t,
                )),
            }
        })
    }
}

impl SimScorer for ScriptedScorer {
    fn score(&self, freq: f32, norm: i64) -> f32 {
        if self.shared.has_failed() {
            return 0.0;
        }
        self.run(freq, norm).unwrap_or_else(|e| {
            self.shared.fail(e);
            0.0
        })
    }

    /// `ScriptedSimilarity`'s explanation: the score, then each value the
    /// script read.
    fn explain(&self, freq: Explanation, norm: i64) -> Explanation {
        let score = self.score(freq.value, norm);
        let s = self.stats;
        let length = lucene_util::small_float::byte4_to_int(norm as u8) as i64;
        Explanation::match_(
            score,
            format!("score from {} computed from:", self.shared.description),
        )
        .with_details(vec![
            Explanation::match_(self.weight as f32, "weight"),
            Explanation::match_(s.boost, "query.boost"),
            Explanation::match_long(s.doc_count, "field.docCount"),
            Explanation::match_long(s.sum_doc_freq, "field.sumDocFreq"),
            Explanation::match_long(s.sum_total_term_freq, "field.sumTotalTermFreq"),
            Explanation::match_long(s.doc_freq, "term.docFreq"),
            Explanation::match_long(s.total_term_freq, "term.totalTermFreq"),
            Explanation::match_value_of(&freq, "doc.freq").with_details(freq.details.clone()),
            Explanation::match_long(length, "doc.length"),
        ])
    }
}

/// More than one term's scorers: their scores summed in `double`.
#[derive(Debug)]
pub struct SummedScorer(pub Vec<ScriptedScorer>);

impl SimScorer for SummedScorer {
    fn score(&self, freq: f32, norm: i64) -> f32 {
        // From `0.0` as Java's `double` starts: `Sum` starts from `-0.0`.
        self.0
            .iter()
            .fold(0.0f64, |sum, s| sum + f64::from(s.score(freq, norm))) as f32
    }

    fn explain(&self, freq: Explanation, norm: i64) -> Explanation {
        let details = self
            .0
            .iter()
            .map(|s| s.explain(freq.clone(), norm))
            .collect();
        Explanation::match_(self.score(freq.value, norm), "Sum of:").with_details(details)
    }
}

/// `NonNegativeScoresSimilarity`'s scorer around a scripted similarity's.
pub struct NonNegativeScorer {
    pub inner: Arc<dyn SimScorer>,
    pub shared: Arc<Scripted>,
}

impl std::fmt::Debug for NonNegativeScorer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NonNegativeScorer")
            .field("negative", &self.shared.negative)
            .finish_non_exhaustive()
    }
}

impl SimScorer for NonNegativeScorer {
    /// A negative score fails the search, its explanation the message, or
    /// is `0` when not enforced; `NaN` and `-0.0` pass.
    fn score(&self, freq: f32, norm: i64) -> f32 {
        let score = self.inner.score(freq, norm);
        if score >= 0.0 || score.is_nan() {
            return score;
        }
        if self.shared.negative == NegativeScores::Fail {
            let why = self
                .inner
                .explain(Explanation::match_(freq, "term frequency"), norm);
            self.shared.fail(score_error(format!(
                "Similarities must not produce negative scores, but got:\n{why}"
            )));
        }
        0.0
    }

    /// A negative explanation as the `max of` it and `0`.
    fn explain(&self, freq: Explanation, norm: i64) -> Explanation {
        let e = self.inner.explain(freq, norm);
        if e.matched && e.value < 0.0 {
            Explanation::match_(0.0, "max of:")
                .with_details(vec![e, Explanation::match_(0.0, "Minimum allowed score")])
        } else {
            e
        }
    }
}

impl Drop for Scripted {
    fn drop(&mut self) {
        // The dropping thread's program; another thread's goes when that
        // thread passes MAX_SCORER_STATES.
        let _ = PROGRAMS.try_with(|p| p.borrow_mut().remove(&self.id));
    }
}

impl Drop for ScriptedScorer {
    fn drop(&mut self) {
        // The dropping thread's state; another thread's goes when that
        // thread passes MAX_SCORER_STATES.
        let _ = SCORERS.try_with(|s| s.borrow_mut().remove(&self.id));
    }
}
