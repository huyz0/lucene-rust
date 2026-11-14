//! The query-tree nodes that take the last Lucene queries OpenSearch builds
//! off the Java fallback: multi-term queries over any
//! automaton, the doc-values-only queries, `CoveringQuery` and the two
//! distance-feature queries.
//!
//! | kind | layout after the kind byte | Lucene query (OpenSearch DSL) |
//! |---|---|---|
//! | `27` multi-term | `field`, `rewrite: u8` (`0` `CONSTANT_SCORE_BLENDED_REWRITE`, `1` `CONSTANT_SCORE_REWRITE`, `2` `DOC_VALUES_REWRITE`), the `CompiledAutomaton` its terms enumeration runs ([`decode_compiled_automaton`]) | any `AutomatonQuery` (`wildcard`, `regexp` with flags or `case_insensitive`, `prefix` on `index_prefixes`), `FuzzyQuery`, under a constant-score or doc-values rewrite |
//! | `28` doc-values term set | `field`, `count: i32`, the terms | `TermInSetQuery` under `DOC_VALUES_REWRITE` (`terms` on a doc-values-only `keyword`) |
//! | `29` doc-values numeric range | `field`, `lower: i64`, `upper: i64` (inclusive) | `SortedNumericDocValuesRangeQuery` (`range`/`term` on a doc-values-only number or date) |
//! | `30` doc-values numeric set | `field`, `count: i32`, the `i64`s | `SortedNumericDocValuesSetQuery` (`terms` on a doc-values-only number) |
//! | `31` doc-values ordinal range | `field`, `flags: u8` (as `NODE_TERM_RANGE`'s), the bounds it says are present | `SortedSetDocValuesRangeQuery` (`range`/`term` on a doc-values-only `keyword` or `ip`) |
//! | `32` doc-values multi-range | `field`, `count: i32`, each `lower`, `upper` (inclusive) | `SortedSetDocValuesMultiRangeQuery` (`terms` on a doc-values-only `ip`) |
//! | `33` covering | `count: i32`, the sub-queries (nodes), the minimum: `0` + `i64` constant, or `1` + `field` (the field's first sorted-numeric value), or `2` + a checked Painless `terms_set` script (`jvm_painless_score::PainlessMinimum`) | `CoveringQuery` (`terms_set`) |
//! | `34` long distance feature | `field`, `origin: i64`, `pivot: i64` | `LongDistanceFeatureQuery` (`distance_feature` on a `date`) |
//! | `35` lat/lon distance feature | `field`, `lat: f64`, `lon: f64`, `pivot: f64` (metres) | `LatLonPointDistanceFeatureQuery` (`distance_feature` on a `geo_point`) |
//! | `36` synonym | `field`, `count: i32`, each `term: bytes`, `boost: f32` bits as `i32` | `SynonymQuery` (a `match` whose analyzer stacks tokens: `hunspell`, `synonym_graph`, `keyword_repeat`) |
//! | `37` feature | `field`, `feature: bytes`, `function: u8` (`0` linear; `1` log, then `scaling_factor: f32`; `2` saturation, then `pivot: f32`; `3` sigmoid, then `pivot: f32`, `exponent: f32`), each `f32` as `i32` bits | `FeatureField`'s `FeatureQuery` (`rank_feature`, `neural_sparse` on `rank_features`), its weight a boost node around it |
//! | `39` KNN | `field`, the target ([`decode_knn_target`]), `kind: u8` (`0` top `k`: `k: i32`, `ef_search: i32`; `1` radial: `min_score: f32` bits), `has_filter: u8` and the filter (a node) when `1` | the k-NN plugin's per-segment `knn` (`faiss` engine: `k` hits per segment; `min_score`/`max_distance`) |
//! | `40` doc and score | `segments: i32`, each segment's `doc_base: i32` (ascending), then `count: i32`, each `doc: i32` (global), `score: f32` bits | `DocAndScoreQuery`: a whole-reader KNN search already run (`lucene-ffi`'s `jvm_knn`) |
//! | `41` to-child block join | the parent filter (a node), the parent query (a node) | `ToChildBlockJoinQuery` with a `QueryBitSetProducer` (a `knn` filter on a nested field, as the k-NN plugin joins it down) |
//! | `38` sparse ANN | `field`, `count: i32`, each `token: i32`, `weight: f32` bits; `top_n: i32`, `k: i32`, `heap_factor: f32` bits, `has_filter: u8` and the filter (a node) when `1`, then the fallback (a node) | neural-search's `SparseVectorQuery` (`neural_sparse` with `method_parameters` on a `sparse_vector`) |
//!
//! The doc-values, distance-feature, feature and sparse nodes become
//! [`DocumentClause`]s keyed by their own wire bytes, as the geo nodes do.
//! The sparse node's filter runs as `filter AND exists(field)`, as
//! neural-search builds it.
//! Every node is built through the constructor that checks what Java's
//! checks; a count is bounded by the node limit or by the blob before it
//! sizes anything.

use std::sync::Arc;

use lucene_sandbox::search::{ByteRange, CoveringQuery, SortedSetDocValuesMultiRangeQuery};
use lucene_search::document::geo::LatLonPointDistanceFeatureQuery;
use lucene_search::document::{
    DocValuesTermInSetQuery, DocumentQuery, FeatureFunction, FeatureQuery,
    LongDistanceFeatureQuery, SortedNumericDocValuesRangeQuery, SortedNumericDocValuesSetQuery,
    SortedSetDocValuesRangeQuery,
};
use lucene_search::extended_query::{
    AutomatonQuery, DocumentClause, ExtendedQuery, IndexOrDocValuesQuery, MultiTermQuery,
    MultiTermSource, NumericDocValuesRangeQuery, RewriteMethod, SubstringKind, SubstringQuery,
    SynonymQuery, VisitEvent, VisitRecord,
};
use lucene_search::query::{BooleanQuery, Clause, FieldExistsQuery, PointsRangeQuery};
use lucene_search::reader::doc_values::get_sorted_numeric;
use lucene_search::reader::SortedNumericDocValues;
use lucene_search::values_source::{
    long_constant, BoxDoubleValues, BoxLongValues, LongValues, LongValuesSource, ValuesContext,
};
use lucene_sparse::SparseAnnQuery;

use crate::clauses::check_clause_count;
use crate::nodes::{decode_compiled_automaton, invalid};
use crate::query_blob::{hex, Cursor};
use crate::status::Status;

pub const NODE_MULTI_TERM: u8 = 27;
pub const NODE_DV_TERM_SET: u8 = 28;
pub const NODE_DV_NUMERIC_RANGE: u8 = 29;
pub const NODE_DV_NUMERIC_SET: u8 = 30;
pub const NODE_DV_ORD_RANGE: u8 = 31;
pub const NODE_DV_MULTI_RANGE: u8 = 32;
pub const NODE_COVERING: u8 = 33;
pub const NODE_LONG_DISTANCE_FEATURE: u8 = 34;
pub const NODE_GEO_DISTANCE_FEATURE: u8 = 35;
/// `SynonymQuery`: `field`, `n: i32`, then `n` of `term: bytes`, `boost: f32`.
pub const NODE_SYNONYM: u8 = 36;
pub const NODE_FEATURE: u8 = 37;
pub const NODE_SPARSE_ANN: u8 = 38;
pub const NODE_KNN: u8 = 39;
pub const NODE_DOC_AND_SCORE: u8 = 40;
pub const NODE_TO_CHILD: u8 = 41;
/// `script_score` with `knn_score` ([`crate::knn_score`]).
pub const NODE_KNN_SCORE: u8 = crate::knn_score::NODE_KNN_SCORE;
/// `script_score` running a Painless script ([`crate::painless_score`]).
pub const NODE_PAINLESS_SCORE: u8 = crate::painless_score::NODE_PAINLESS_SCORE;
/// `max_sim` over a `multi_vector` field ([`crate::max_sim_query`]).
pub const NODE_MAX_SIM: u8 = crate::max_sim_query::NODE_MAX_SIM;
/// `script` query running a Painless script ([`crate::painless_score`]).
pub const NODE_PAINLESS_FILTER: u8 = crate::painless_score::NODE_PAINLESS_FILTER;
/// A phrase over a `pattern_text` field ([`crate::pattern_phrase`]).
pub const NODE_PATTERN_PHRASE: u8 = crate::pattern_phrase::NODE_PATTERN_PHRASE;
/// The first and last kinds this module decodes.
pub const NODE_FIRST: u8 = NODE_MULTI_TERM;
/// OpenSearch's `has_child`/`has_parent` (`HasChildQueryBuilder.LateParsingQuery`):
/// `join_field`, `score_mode: u8` (as the to-parent node's), `min: i32`,
/// `max: i32`, the to query (a node), the from query (a node). Searched once
/// `join::resolve` has run its from side.
pub const NODE_JOIN: u8 = 46;
/// OpenSearch's range on a long, date or double field with doc values
/// (`IndexOrDocValuesQuery` of `LongPoint.newRangeQuery` and
/// `SortedNumericDocValuesField.newSlowRangeQuery` over the same field and
/// bounds): `field`, `min: i64`, `max: i64`, the inclusive bounds as sortable
/// longs. Kept a choice so a cheap lead (a tenant's doc-id run) checks its
/// documents' doc values instead of collecting the range's points.
pub const NODE_POINT_RANGE_OR_DV: u8 = 48;
/// A query on the plugin's `compact_wildcard` field ([`SubstringQuery`]):
/// `field`, `kind: u8` ([`SUBSTRING_WILDCARD`], [`SUBSTRING_REGEXP`],
/// [`SUBSTRING_CONTAINS`] or [`SUBSTRING_EXACT`]), `case_insensitive: u8`,
/// `pattern: bytes` (UTF-8). Constant-scored 1.
pub const NODE_SUBSTRING: u8 = 49;
/// A `PhraseQuery` with its terms' own positions (a removed stopword leaves a
/// gap): `field`, `slop: i32`, `count: i32`, then per term `position: i32`
/// and the term. Only the highlighter's request holds one
/// (`lucene-ffi`'s `jvm_highlight`): the search falls such a phrase back.
pub const NODE_PHRASE_POSITIONS: u8 = 50;
/// `MultiPhraseQuery`, for the highlighter: `field`, `slop: i32`, `count:
/// i32`, then per position `position: i32`, `terms: i32` and the terms
/// (`bytes`).
pub const NODE_MULTI_PHRASE: u8 = 51;
/// OpenSearch's `MultiPhrasePrefixQuery` (`match_phrase_prefix`), for the
/// highlighters: laid out as [`NODE_MULTI_PHRASE`] with `max_expansions:
/// i32` after the slop, the last position's terms the prefixes.
pub const NODE_MULTI_PHRASE_PREFIX: u8 = 52;
/// `FuzzyQuery`, for the `plain` highlighter (which weights each term it
/// expands to by its similarity, and keeps every one): `field`, `term:
/// bytes`, `max_edits: u8`, `prefix_length: i32`, `transpositions: u8`.
pub const NODE_FUZZY: u8 = 53;
/// OpenSearch's `FunctionScoreQuery` as the `unified` highlighter reads it
/// (its query visited as `MUST`, unknown to the phrase helper): the
/// wrapped query, a node.
pub const NODE_VISIT_ONLY: u8 = 54;
/// A query the `unified` highlighter reads only through its `visit`: what
/// that visit did, a record -- `count: i32`, then per event a kind
/// (`VISIT_*`) and: terms, a guard (`present: u8`, field) then `count: i32`
/// `(field, term)` pairs; a leaf, a guard; a sub-visitor, `occur: u8` (0
/// `MUST`, 1 `FILTER`, 2 `SHOULD`, 3 `MUST_NOT`) and a record; a query
/// handed over whole, a node.
pub const NODE_VISITED: u8 = 55;
pub const NODE_LAST: u8 = NODE_VISITED;

pub const VISIT_TERMS: u8 = 0;
pub const VISIT_LEAF: u8 = 1;
pub const VISIT_SUB: u8 = 2;
pub const VISIT_NODE: u8 = 3;

/// [`NODE_SUBSTRING`]'s kinds.
pub const SUBSTRING_WILDCARD: u8 = 0;
pub const SUBSTRING_REGEXP: u8 = 1;
pub const SUBSTRING_CONTAINS: u8 = 2;
pub const SUBSTRING_EXACT: u8 = 3;

/// The KNN node's kinds.
pub const KNN_TOP_K: u8 = 0;
pub const KNN_RADIAL: u8 = 1;
/// A KNN target's encodings.
pub const KNN_TARGET_FLOAT: u8 = 0;
pub const KNN_TARGET_BYTE: u8 = 1;

/// A KNN query vector: `encoding: u8` ([`KNN_TARGET_FLOAT`] or
/// [`KNN_TARGET_BYTE`]), `dimension: i32`, then the components (`f32` bits
/// as `i32` each, or one signed byte each).
pub fn decode_knn_target(
    c: &mut Cursor<'_>,
) -> Result<lucene_search::knn_search::KnnTarget, Status> {
    use lucene_search::knn_search::KnnTarget;
    let encoding = c.u8()?;
    let dim = usize::try_from(c.i32()?)
        .ok()
        .filter(|&d| d > 0 && d <= MAX_KNN_DIMENSION)
        .ok_or_else(|| invalid("query tree: KNN target dimension out of range"))?;
    Ok(match encoding {
        KNN_TARGET_FLOAT => {
            let mut v = Vec::with_capacity(dim); // alloc-ok: dim is bounded by MAX_KNN_DIMENSION
            for _ in 0..dim {
                let x = f32(c)?;
                if !x.is_finite() {
                    return Err(invalid("query tree: a KNN target component is not finite"));
                }
                v.push(x);
            }
            KnnTarget::Float(v)
        }
        KNN_TARGET_BYTE => {
            let mut v = Vec::with_capacity(dim); // alloc-ok: dim is bounded by MAX_KNN_DIMENSION
            for _ in 0..dim {
                v.push(c.u8()?);
            }
            KnnTarget::Byte(v)
        }
        other => {
            return Err(invalid(format!(
                "query tree: KNN target encoding {other} (expected 0..=1)"
            )))
        }
    })
}

/// The largest vector dimension a query may carry: the k-NN plugin's own
/// limit (`KNNVectorFieldMapper.MAX_DIMENSION`, 16,000).
pub const MAX_KNN_DIMENSION: usize = 16_000;

/// The feature node's functions.
pub const FEATURE_LINEAR: u8 = 0;
pub const FEATURE_LOG: u8 = 1;
pub const FEATURE_SATURATION: u8 = 2;
pub const FEATURE_SIGMOID: u8 = 3;

/// The multi-term node's rewrite methods.
pub const REWRITE_CONSTANT_SCORE_BLENDED: u8 = 0;
pub const REWRITE_CONSTANT_SCORE: u8 = 1;
pub const REWRITE_DOC_VALUES: u8 = 2;

/// The covering node's minimum sources.
pub const MINIMUM_CONSTANT: u8 = 0;
pub const MINIMUM_FIELD: u8 = 1;
pub const MINIMUM_PAINLESS: u8 = 2;

/// `NODE_TERM_RANGE`'s flags, which the ordinal range shares.
pub const RANGE_HAS_LOWER: u8 = 1;
pub const RANGE_HAS_UPPER: u8 = 2;
pub const RANGE_INCLUDE_LOWER: u8 = 4;
pub const RANGE_INCLUDE_UPPER: u8 = 8;

pub fn utf8(b: &[u8]) -> Result<&str, Status> {
    std::str::from_utf8(b).map_err(|_| Status::InvalidUtf8)
}

pub fn f32(c: &mut Cursor<'_>) -> Result<f32, Status> {
    Ok(f32::from_bits(c.i32()? as u32))
}

/// A node of kinds 27-41, its kind byte read; `start` is the node's offset.
pub fn decode(
    c: &mut Cursor<'_>,
    kind: u8,
    start: usize,
    depth: usize,
    nodes: &mut usize,
) -> Result<Clause, Status> {
    let field =
        |c: &mut Cursor<'_>| -> Result<String, Status> { Ok(utf8(c.bytes()?)?.to_string()) };
    let built = |what: &str, e: lucene_search::Error| invalid(format!("query tree: {what}: {e}"));
    let document: Arc<dyn DocumentQuery> = match kind {
        NODE_MULTI_TERM => {
            let field = field(c)?;
            let rewrite = match c.u8()? {
                REWRITE_CONSTANT_SCORE_BLENDED => RewriteMethod::ConstantScoreBlended,
                REWRITE_CONSTANT_SCORE => RewriteMethod::ConstantScore,
                REWRITE_DOC_VALUES => RewriteMethod::DocValues,
                other => {
                    return Err(invalid(format!(
                        "query tree: unknown multi-term rewrite {other} (expected 0..=2)"
                    )))
                }
            };
            // `CompiledAutomaton`'s byte automaton: what `getTermsEnum`
            // intersects the terms with, whatever the query class built it.
            let automaton = decode_compiled_automaton(c)?;
            let source = MultiTermSource::Automaton(AutomatonQuery::new(field, automaton, true));
            return Ok(Clause::Extended(Box::new(ExtendedQuery::MultiTerm(
                MultiTermQuery::new(source, rewrite),
            ))));
        }
        NODE_DV_TERM_SET => {
            let field = field(c)?;
            let count = c.length()?;
            *nodes = nodes.saturating_add(count);
            check_clause_count(*nodes)?;
            let mut terms = Vec::new();
            for _ in 0..count {
                terms.push(c.bytes()?.to_vec());
            }
            Arc::new(DocValuesTermInSetQuery::new(field, terms))
        }
        NODE_DV_NUMERIC_RANGE => {
            let field = field(c)?;
            let (lower, upper) = (c.i64()?, c.i64()?);
            Arc::new(SortedNumericDocValuesRangeQuery::new(field, lower, upper))
        }
        NODE_DV_NUMERIC_SET => {
            // One leaf to Lucene's clause visitor, however many numbers; each
            // is eight bytes of the blob.
            let field = field(c)?;
            let count = c.length()?;
            let mut numbers = Vec::new();
            for _ in 0..count {
                numbers.push(c.i64()?);
            }
            Arc::new(SortedNumericDocValuesSetQuery::new(field, numbers))
        }
        NODE_DV_ORD_RANGE => {
            let field = field(c)?;
            let flags = c.u8()?;
            if flags
                & !(RANGE_HAS_LOWER | RANGE_HAS_UPPER | RANGE_INCLUDE_LOWER | RANGE_INCLUDE_UPPER)
                != 0
            {
                return Err(invalid(format!(
                    "query tree: doc-values range flags {flags:#x}"
                )));
            }
            let lower = if flags & RANGE_HAS_LOWER != 0 {
                Some(c.bytes()?.to_vec())
            } else {
                None
            };
            let upper = if flags & RANGE_HAS_UPPER != 0 {
                Some(c.bytes()?.to_vec())
            } else {
                None
            };
            Arc::new(SortedSetDocValuesRangeQuery::new(
                field,
                lower,
                upper,
                flags & RANGE_INCLUDE_LOWER != 0,
                flags & RANGE_INCLUDE_UPPER != 0,
            ))
        }
        NODE_DV_MULTI_RANGE => {
            let field = field(c)?;
            let count = c.length()?;
            if count == 0 {
                return Err(invalid(
                    "query tree: a doc-values multi-range with no ranges",
                ));
            }
            let mut ranges = Vec::new();
            for _ in 0..count {
                let lower = c.bytes()?.to_vec();
                let upper = c.bytes()?.to_vec();
                ranges.push(ByteRange::new(lower, upper));
            }
            Arc::new(SortedSetDocValuesMultiRangeQuery::new(field, ranges))
        }
        NODE_COVERING => {
            let count = c.length()?;
            check_clause_count(nodes.saturating_add(count))?;
            let mut queries = Vec::new();
            for _ in 0..count {
                queries.push(crate::query_blob::decode_node(c, depth + 1, nodes)?);
            }
            let minimum: Arc<dyn LongValuesSource> = match c.u8()? {
                MINIMUM_CONSTANT => long_constant(c.i64()?),
                MINIMUM_FIELD => Arc::new(FirstValueSource(field(c)?)),
                MINIMUM_PAINLESS => Arc::new(crate::painless_score::PainlessMinimum::decode(c)?),
                other => {
                    return Err(invalid(format!(
                        "query tree: unknown covering minimum {other} (expected 0..=2)"
                    )))
                }
            };
            let q = CoveringQuery::new(queries, minimum).map_err(|e| built("covering query", e))?;
            return Ok(Clause::Extended(Box::new(ExtendedQuery::Covering(q))));
        }
        NODE_LONG_DISTANCE_FEATURE => {
            let field = field(c)?;
            let (origin, pivot) = (c.i64()?, c.i64()?);
            Arc::new(
                LongDistanceFeatureQuery::new(field, origin, pivot)
                    .map_err(|e| built("distance feature", e))?,
            )
        }
        NODE_SYNONYM => {
            let field = field(c)?;
            let count = c.length()?;
            *nodes = nodes.saturating_add(count);
            check_clause_count(*nodes)?;
            let mut terms = Vec::with_capacity(count); // alloc-ok: count is within the clause limit
            for _ in 0..count {
                let term = c.bytes()?.to_vec();
                terms.push((term, f32::from_bits(c.i32()? as u32)));
            }
            let q = SynonymQuery::new(field, terms).map_err(|e| built("synonym query", e))?;
            return Ok(Clause::Extended(Box::new(ExtendedQuery::Synonym(q))));
        }
        NODE_FEATURE => {
            let field = field(c)?;
            let feature = utf8(c.bytes()?)?.to_string();
            let function = match c.u8()? {
                FEATURE_LINEAR => FeatureFunction::Linear,
                FEATURE_LOG => FeatureFunction::Log {
                    scaling_factor: f32(c)?,
                },
                FEATURE_SATURATION => FeatureFunction::Saturation {
                    pivot: Some(f32(c)?),
                },
                FEATURE_SIGMOID => FeatureFunction::Sigmoid {
                    pivot: f32(c)?,
                    a: f32(c)?,
                },
                other => {
                    return Err(invalid(format!(
                        "query tree: unknown feature function {other} (expected 0..=3)"
                    )))
                }
            };
            Arc::new(FeatureQuery::new(field, feature, function))
        }
        NODE_SPARSE_ANN => {
            let field = field(c)?;
            let count = c.length()?;
            *nodes = nodes.saturating_add(count);
            check_clause_count(*nodes)?;
            let mut tokens = Vec::with_capacity(count); // alloc-ok: count is within the clause limit
            for _ in 0..count {
                let token = u32::try_from(c.i32()?)
                    .map_err(|_| invalid("query tree: a negative sparse token"))?;
                tokens.push((token, f32(c)?));
            }
            let positive = |what: &str, v: i32| {
                usize::try_from(v).ok().filter(|&n| n > 0).ok_or_else(|| {
                    invalid(format!("query tree: sparse {what} {v} is not positive"))
                })
            };
            let top_n = positive("top_n", c.i32()?)?;
            let k = positive("k", c.i32()?)?;
            let heap_factor = f32(c)?;
            let filter = match c.u8()? {
                0 => None,
                1 => Some(crate::query_blob::decode_node(c, depth + 1, nodes)?),
                other => {
                    return Err(invalid(format!(
                        "query tree: sparse filter flag {other} (expected 0..=1)"
                    )))
                }
            };
            let fallback = crate::query_blob::decode_node(c, depth + 1, nodes)?;
            let query = SparseAnnQuery {
                field: field.clone(),
                tokens,
                top_n,
                k,
                heap_factor,
            };
            let mut clause =
                DocumentClause::new(hex(c.since(start)), Arc::new(query)).with_fallback(fallback);
            if let Some(filter) = filter {
                let mut both = BooleanQuery::new();
                both.filter.push(filter);
                both.filter
                    .push(Clause::Exists(FieldExistsQuery::new(field)));
                clause = clause.with_filter(both.into());
            }
            return Ok(clause.into());
        }
        NODE_KNN => {
            use lucene_search::knn_search::{KnnLeafKind, KnnLeafQuery};
            let field = field(c)?;
            let target = decode_knn_target(c)?;
            let kind = match c.u8()? {
                KNN_TOP_K => {
                    let k = usize::try_from(c.i32()?)
                        .map_err(|_| invalid("query tree: a negative KNN k"))?;
                    let ef_search = usize::try_from(c.i32()?)
                        .map_err(|_| invalid("query tree: a negative KNN ef_search"))?;
                    KnnLeafKind::TopK { k, ef_search }
                }
                KNN_RADIAL => KnnLeafKind::Radial { min_score: f32(c)? },
                other => {
                    return Err(invalid(format!(
                        "query tree: KNN kind {other} (expected 0..=1)"
                    )))
                }
            };
            let filter = match c.u8()? {
                0 => None,
                1 => Some(crate::query_blob::decode_node(c, depth + 1, nodes)?),
                other => {
                    return Err(invalid(format!(
                        "query tree: KNN filter flag {other} (expected 0..=1)"
                    )))
                }
            };
            let query =
                KnnLeafQuery::new(field, target, kind).map_err(|e| built("KNN query", e))?;
            let mut clause = DocumentClause::new(hex(c.since(start)), Arc::new(query));
            if let Some(filter) = filter {
                clause = clause.with_filter(filter);
            }
            return Ok(clause.into());
        }
        NODE_KNN_SCORE => return crate::knn_score::decode(c, depth, nodes),
        NODE_PAINLESS_SCORE => return crate::painless_score::decode(c, depth, nodes),
        NODE_MAX_SIM => return crate::max_sim_query::decode(c, depth, nodes),
        NODE_PAINLESS_FILTER => return crate::painless_score::decode_filter(c),
        NODE_PATTERN_PHRASE => return crate::pattern_phrase::decode(c),
        NODE_POINT_RANGE_OR_DV => {
            let field = field(c)?;
            let (min, max) = (c.i64()?, c.i64()?);
            return Ok(IndexOrDocValuesQuery::new(
                PointsRangeQuery::new(field.clone(), min, max),
                NumericDocValuesRangeQuery::new(field, min, max),
            )
            .into());
        }
        NODE_PHRASE_POSITIONS => {
            let field = field(c)?;
            let slop = u32::try_from(c.i32()?)
                .map_err(|_| invalid("query tree: a phrase with a negative slop"))?;
            let count = c.length()?;
            *nodes = nodes.saturating_add(count);
            crate::clauses::check_clause_count(*nodes)?;
            if count == 0 {
                return Err(invalid("query tree: a phrase with no terms"));
            }
            let mut terms = Vec::new();
            for _ in 0..count {
                let position = c.i32()?;
                terms.push((c.bytes()?.to_vec(), position));
            }
            let phrase = lucene_search::query::PhraseQuery::with_positions(field, terms)
                .map_err(|e| invalid(format!("query tree: {e}")))?
                .with_slop(slop);
            return Ok(Clause::Phrase(phrase));
        }
        NODE_MULTI_PHRASE | NODE_MULTI_PHRASE_PREFIX => {
            let field = field(c)?;
            let slop = u32::try_from(c.i32()?)
                .map_err(|_| invalid("query tree: a multi-phrase with a negative slop"))?;
            let max_expansions = if kind == NODE_MULTI_PHRASE_PREFIX {
                c.i32()?
            } else {
                0
            };
            let count = c.length()?;
            if count == 0 {
                return Err(invalid("query tree: a multi-phrase with no terms"));
            }
            let mut arrays = crate::alloc::try_with_capacity(count.min(1024))?;
            for _ in 0..count {
                let position = c.i32()?;
                let n = c.length()?;
                if n == 0 {
                    return Err(invalid("query tree: a multi-phrase position with no terms"));
                }
                *nodes = nodes.saturating_add(n);
                crate::clauses::check_clause_count(*nodes)?;
                let mut terms = crate::alloc::try_with_capacity(n.min(1024))?;
                for _ in 0..n {
                    terms.push(c.bytes()?.to_vec());
                }
                arrays.push((terms, position));
            }
            if kind == NODE_MULTI_PHRASE_PREFIX {
                let (terms, positions) = arrays.into_iter().unzip();
                let q = lucene_search::extended_query::MultiPhrasePrefixQuery::new(
                    field, terms, positions, slop,
                )
                .map_err(|e| invalid(format!("query tree: {e}")))?
                .with_max_expansions(max_expansions);
                return Ok(q.into());
            }
            let phrase = lucene_search::query::MultiPhraseQuery::with_positions(field, arrays)
                .map_err(|e| invalid(format!("query tree: {e}")))?
                .with_slop(slop);
            return Ok(Clause::MultiPhrase(phrase));
        }
        NODE_VISITED => {
            let record = visit_record(c, depth + 1, nodes)?;
            return Ok(Clause::Extended(Box::new(ExtendedQuery::Visited(record))));
        }
        NODE_VISIT_ONLY => {
            let inner = crate::query_blob::decode_node(c, depth + 1, nodes)?;
            return Ok(Clause::Extended(Box::new(ExtendedQuery::VisitOnly(
                Box::new(inner),
            ))));
        }
        NODE_FUZZY => {
            let field = field(c)?;
            let term = c.bytes()?.to_vec();
            let max_edits = c.u8()?;
            if max_edits > 2 {
                return Err(invalid(format!("query tree: fuzzy max edits {max_edits}")));
            }
            let prefix_length = usize::try_from(c.i32()?)
                .map_err(|_| invalid("query tree: a fuzzy with a negative prefix length"))?;
            let transpositions = c.u8()? != 0;
            return Ok(Clause::Fuzzy(
                lucene_search::query::FuzzyQuery::new(field, term)
                    .with_max_edits(max_edits)
                    .with_prefix_length(prefix_length)
                    .with_transpositions(transpositions),
            ));
        }
        NODE_SUBSTRING => {
            let field = field(c)?;
            let kind = match c.u8()? {
                SUBSTRING_WILDCARD => SubstringKind::Wildcard,
                SUBSTRING_REGEXP => SubstringKind::Regexp,
                SUBSTRING_CONTAINS => SubstringKind::Contains,
                SUBSTRING_EXACT => SubstringKind::Exact,
                other => {
                    return Err(invalid(format!(
                        "query tree: substring kind {other} (expected 0..=3)"
                    )))
                }
            };
            let case_insensitive = match c.u8()? {
                0 => false,
                1 => true,
                other => {
                    return Err(invalid(format!(
                        "query tree: substring case flag {other} (expected 0..=1)"
                    )))
                }
            };
            let pattern = std::str::from_utf8(c.bytes()?)
                .map_err(|_| Status::InvalidUtf8)?
                .to_string();
            let query = SubstringQuery::new(field, kind, pattern, case_insensitive)
                .map_err(|e| invalid(format!("query tree: {e}")))?;
            return Ok(Clause::Extended(Box::new(ExtendedQuery::Substring(query))));
        }
        NODE_JOIN => {
            let join_field = std::str::from_utf8(c.bytes()?)
                .map_err(|_| Status::InvalidUtf8)?
                .to_string();
            let score_mode = crate::nodes::decode_score_mode(c)?;
            let min = c.i32()?;
            let max = c.i32()?;
            if min < 0 || max < min {
                return Err(invalid(format!(
                    "query tree: join min_children {min} and max_children {max}"
                )));
            }
            let to_query = crate::query_blob::decode_node(c, depth + 1, nodes)?;
            let from_query = crate::query_blob::decode_node(c, depth + 1, nodes)?;
            return Ok(Clause::Extended(Box::new(ExtendedQuery::LateJoin(
                lucene_search::join::LateJoinQuery {
                    join_field,
                    from_query: Box::new(from_query),
                    to_query: Box::new(to_query),
                    score_mode,
                    min,
                    max,
                },
            ))));
        }
        NODE_TO_CHILD => {
            let parents = crate::nodes::decode_parents(c, depth + 1, nodes)?;
            let parent = crate::query_blob::decode_node(c, depth + 1, nodes)?;
            return Ok(Clause::Extended(Box::new(ExtendedQuery::ToChildBlockJoin(
                lucene_search::join::ToChildBlockJoinQuery::new(parent, parents),
            ))));
        }
        NODE_DOC_AND_SCORE => {
            let segments = c.length()?;
            let mut bases = Vec::with_capacity(segments.min(c.remaining() / 4)); // alloc-ok: bounded by the blob
            for _ in 0..segments {
                let base = c.i32()?;
                if base < 0 || bases.last().is_some_and(|&b| b > base) {
                    return Err(invalid("query tree: doc bases must ascend from 0"));
                }
                bases.push(base);
            }
            let count = c.length()?;
            *nodes = nodes.saturating_add(count / 64);
            check_clause_count(*nodes)?;
            let mut hits = Vec::with_capacity(count.min(c.remaining() / 8)); // alloc-ok: bounded by the blob
            for _ in 0..count {
                let doc = c.i32()?;
                if doc < 0 {
                    return Err(invalid("query tree: a negative doc id"));
                }
                hits.push((doc, f32(c)?));
            }
            hits.sort_unstable_by_key(|&(d, _)| d);
            if hits.windows(2).any(|w| w[0].0 == w[1].0) {
                return Err(invalid("query tree: a doc id listed twice"));
            }
            return Ok(Clause::Extended(Box::new(ExtendedQuery::DocAndScore(
                lucene_search::extended_query::DocAndScoreQuery::new(hits, &bases),
            ))));
        }
        NODE_GEO_DISTANCE_FEATURE => {
            let field = field(c)?;
            let (lat, lon, pivot) = (c.f64()?, c.f64()?, c.f64()?);
            Arc::new(
                LatLonPointDistanceFeatureQuery::new(field, lat, lon, pivot)
                    .map_err(|e| built("distance feature", e))?,
            )
        }
        other => {
            return Err(invalid(format!(
                "query tree: unknown node kind {other} (expected {NODE_FIRST}..={NODE_LAST})"
            )))
        }
    };
    let key = hex(c.since(start));
    Ok(DocumentClause::new(key, document).into())
}

/// OpenSearch's `TermsSetQueryBuilder.FieldValuesSource` over plain
/// sorted-numeric field data: a document's first (smallest) value, none
/// for a document without one (which `CoveringQuery` never matches).
pub struct FirstValueSource(pub String);

pub struct FirstValues<'c> {
    pub values: Box<dyn SortedNumericDocValues + 'c>,
    pub current: i64,
}

impl LongValues for FirstValues<'_> {
    fn advance_exact(&mut self, doc: i32) -> lucene_search::Result<bool> {
        if !self.values.advance_exact(doc)? {
            return Ok(false);
        }
        self.current = self.values.next_value()?;
        Ok(true)
    }

    fn long_value(&mut self) -> lucene_search::Result<i64> {
        Ok(self.current)
    }
}

impl LongValuesSource for FirstValueSource {
    fn get_values<'c>(
        &self,
        ctx: &ValuesContext<'c>,
        leaf: usize,
        _scores: Option<BoxDoubleValues<'c>>,
    ) -> lucene_search::Result<BoxLongValues<'c>> {
        Ok(Box::new(FirstValues {
            values: get_sorted_numeric(ctx.leaf_reader(leaf)?, &self.0)?,
            current: -1,
        }))
    }

    fn needs_scores(&self) -> bool {
        false
    }

    /// `FieldValuesSource.isCacheable`: always.
    fn is_cacheable(&self, _ctx: &ValuesContext<'_>, _leaf: usize) -> bool {
        true
    }

    fn describe(&self) -> String {
        format!("long({})", self.0)
    }
}

/// A [`NODE_VISITED`] record.
pub fn visit_record(
    c: &mut Cursor<'_>,
    depth: usize,
    nodes: &mut usize,
) -> Result<VisitRecord, Status> {
    use crate::clauses::MAX_CLAUSE_DEPTH;
    use lucene_search::query_visitor::{Occur, Term};
    if depth >= MAX_CLAUSE_DEPTH {
        return Err(invalid(format!(
            "query tree: nesting depth exceeds the maximum of {MAX_CLAUSE_DEPTH}"
        )));
    }
    let utf8 = |b: &[u8]| {
        std::str::from_utf8(b)
            .map(str::to_owned)
            .map_err(|_| Status::InvalidUtf8)
    };
    let guard = |c: &mut Cursor<'_>| -> Result<Option<String>, Status> {
        match c.u8()? {
            0 => Ok(None),
            1 => Ok(Some(utf8(c.bytes()?)?)),
            other => Err(invalid(format!(
                "query tree: visit guard flag {other} (expected 0..=1)"
            ))),
        }
    };
    let count = c.length()?;
    *nodes = nodes.saturating_add(count);
    check_clause_count(*nodes)?;
    let mut events = Vec::with_capacity(count); // alloc-ok: count is within the clause limit
    for _ in 0..count {
        events.push(match c.u8()? {
            VISIT_TERMS => {
                let guard = guard(c)?;
                let n = c.length()?;
                *nodes = nodes.saturating_add(n);
                check_clause_count(*nodes)?;
                let mut terms = Vec::with_capacity(n); // alloc-ok: n is within the clause limit
                for _ in 0..n {
                    let field = utf8(c.bytes()?)?;
                    terms.push(Term::new(field, c.bytes()?.to_vec()));
                }
                VisitEvent::Terms { guard, terms }
            }
            VISIT_LEAF => VisitEvent::Leaf { guard: guard(c)? },
            VISIT_SUB => {
                let occur = match c.u8()? {
                    0 => Occur::Must,
                    1 => Occur::Filter,
                    2 => Occur::Should,
                    3 => Occur::MustNot,
                    other => {
                        return Err(invalid(format!(
                            "query tree: unknown Occur tag {other} (expected 0..=3)"
                        )))
                    }
                };
                VisitEvent::Sub(occur, visit_record(c, depth + 1, nodes)?)
            }
            VISIT_NODE => VisitEvent::Node(crate::query_blob::decode_node(c, depth + 1, nodes)?),
            other => {
                return Err(invalid(format!(
                    "query tree: unknown visit event {other} (expected 0..=3)"
                )))
            }
        });
    }
    Ok(VisitRecord { events })
}
