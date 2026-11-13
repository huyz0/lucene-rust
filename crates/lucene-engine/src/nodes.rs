//! Query-tree nodes in `lucene-ffi`'s `jvm_reader`'s
//! `QUERY_TREE` blob: the OpenSearch shapes `lucene-join` and
//! `lucene-queries` back natively.
//!
//! | kind | layout after the kind byte | Lucene query (OpenSearch DSL) |
//! |---|---|---|
//! | `20` to-parent block join | `score_mode: u8` (`ScoreMode`'s ordinal: `0` None, `1` Avg, `2` Max, `3` Total, `4` Min), the parent filter (a node), the child query (a node) | `ToParentBlockJoinQuery` with a `QueryBitSetProducer` (`nested`) |
//! | `21` span | a span tree ([`decode_span`]) | the `spans` package (`span_*`, `field_masking_span`) |
//! | `22` interval | `field`, `scoring: u8` (`0` saturation `pivot: f32`, `1` sigmoid `pivot: f32`, `exp: f32`), a source tree ([`decode_source`]) | `IntervalQuery` (`intervals`) |
//! | `23` combined field | `term`, `count: i32`, then `count` times `field`, `weight: f32` | `CombinedFieldQuery` (`combined_fields`) |
//! | `24` OpenSearch function score | [`crate::function_score`] | OpenSearch's `FunctionScoreQuery` (`function_score`) |
//!
//! Every node is decoded into the query `lucene-search` ports, through the
//! constructors that check what Java's check; a structure Java's would refuse
//! is [`Status::InvalidArgument`], never a panic. Nesting below these
//! nodes counts against the same depth and node limits as the rest of the
//! tree.
//!
//! Every count read off the blob is bounded before it sizes anything: list
//! lengths count against the node limit as they are read (and each element
//! must be in the blob), a repeating source's copies -- each an iterator with
//! its own cached positions -- against the clause limit, a multi-term
//! source's `max_expansions` likewise, an automaton's states at 65,536 (its
//! transitions are in the blob). The remaining integers (slops, positions,
//! gaps, widths, `pre`/`post`) are thresholds the ported iterators compare,
//! never sizes. No `check-port-invariants.py` rule covers this: telling a
//! wire-decoded count from any other `usize` needs data flow, not a pattern.
//!
//! A parent filter's bit sets are cached across requests ([`SharedParents`]),
//! as OpenSearch's `BitsetFilterCache` caches them: per segment core and
//! filter, deleted documents included, at most 64 MiB, and dropped when the
//! last reader holding their segment closes.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use lucene_search::extended_query::{CombinedFieldQuery, ExtendedQuery};
use lucene_search::intervals::{
    IntervalFilterKind, IntervalQuery, IntervalScoreFunction, IntervalsSource, MultiTermPattern,
};
use lucene_search::join::{BitSetProducer, QueryBitSetProducer, ScoreMode, ToParentBlockJoinQuery};
use lucene_search::multi_segment::OpenSegment;
use lucene_search::query::{BooleanQuery, Clause};
use lucene_search::spans::SpanNode;
use lucene_util::automaton::Automaton;
use lucene_util::fixed_bit_set::FixedBitSet;

use crate::clauses::{check_clause_count, MAX_CLAUSE_DEPTH};
use crate::query_blob::Cursor;
use crate::status::{set_last_error, Status};

pub const NODE_TO_PARENT: u8 = 20;
pub const NODE_SPAN: u8 = 21;
pub const NODE_INTERVAL: u8 = 22;
pub const NODE_COMBINED_FIELD: u8 = 23;
pub const NODE_FUNCTION_SCORE: u8 = 24;
/// The highest node kind the tree knows.
pub const NODE_LAST: u8 = NODE_FUNCTION_SCORE;

/// [`Status::InvalidArgument`] with `message` as the last error.
pub fn invalid(message: impl Into<String>) -> Status {
    set_last_error(message.into());
    Status::InvalidArgument
}

/// A query a constructor refused, as the decode's error.
pub fn refused(what: &str, e: lucene_search::Error) -> Status {
    invalid(format!("query tree: {what}: {e}"))
}

pub fn utf8(b: &[u8]) -> Result<&str, Status> {
    std::str::from_utf8(b).map_err(|_| Status::InvalidUtf8)
}

/// One more nested level and one more node, against the tree's limits.
pub fn descend(depth: usize, nodes: &mut usize) -> Result<(), Status> {
    if depth >= MAX_CLAUSE_DEPTH {
        return Err(invalid(format!(
            "query tree: nesting depth exceeds the maximum of {MAX_CLAUSE_DEPTH}"
        )));
    }
    *nodes = nodes.saturating_add(1);
    check_clause_count(*nodes)
}

/// A count of children, at least `min`, each also counted as a node.
pub fn count(
    c: &mut Cursor<'_>,
    min: usize,
    nodes: &mut usize,
    what: &str,
) -> Result<usize, Status> {
    let n = c.length()?;
    if n < min {
        return Err(invalid(format!(
            "query tree: {what} with {n} sources (want at least {min})"
        )));
    }
    check_clause_count(nodes.saturating_add(n))?;
    Ok(n)
}

/// A parent filter (a node) as the bit-set producer of a block join, its
/// sets shared across requests ([`SharedParents`]).
pub fn decode_parents(
    c: &mut Cursor<'_>,
    depth: usize,
    nodes: &mut usize,
) -> Result<Arc<dyn BitSetProducer>, Status> {
    let start = c.pos();
    let parents = crate::query_blob::decode_node(c, depth, nodes)?;
    let key = crate::query_blob::hex(c.since(start));
    Ok(Arc::new(SharedParents {
        key,
        query: BooleanQuery {
            must: vec![parents],
            ..Default::default()
        },
    }))
}

/// A block or query-time join's `ScoreMode`, by ordinal: `0` None, `1` Avg,
/// `2` Max, `3` Total, `4` Min.
pub fn decode_score_mode(c: &mut Cursor<'_>) -> Result<ScoreMode, Status> {
    Ok(match c.u8()? {
        0 => ScoreMode::None,
        1 => ScoreMode::Avg,
        2 => ScoreMode::Max,
        3 => ScoreMode::Total,
        4 => ScoreMode::Min,
        other => {
            return Err(invalid(format!(
                "query tree: unknown block-join score mode {other} (expected 0..=4)"
            )))
        }
    })
}

/// A node of kind 20-24, its kind byte read; `start` is the node's offset.
pub fn decode(
    c: &mut Cursor<'_>,
    kind: u8,
    start: usize,
    depth: usize,
    nodes: &mut usize,
) -> Result<Clause, Status> {
    let extended = match kind {
        NODE_TO_PARENT => {
            let score_mode = decode_score_mode(c)?;
            let parents = decode_parents(c, depth + 1, nodes)?;
            let child = crate::query_blob::decode_node(c, depth + 1, nodes)?;
            ExtendedQuery::ToParentBlockJoin(ToParentBlockJoinQuery::new(
                child, parents, score_mode,
            ))
        }
        NODE_SPAN => ExtendedQuery::Span(decode_span(c, depth + 1, nodes)?),
        NODE_INTERVAL => {
            let field = utf8(c.bytes()?)?.to_string();
            let scoring = c.u8()?;
            let pivot = f32::from_bits(c.i32()? as u32);
            let function = match scoring {
                0 => IntervalScoreFunction::saturation(pivot),
                1 => IntervalScoreFunction::sigmoid(pivot, f32::from_bits(c.i32()? as u32)),
                other => {
                    return Err(invalid(format!(
                        "query tree: unknown interval scoring {other} (expected 0 or 1)"
                    )))
                }
            }
            .map_err(|e| refused("interval query", e))?;
            let source = decode_source(c, depth + 1, nodes)?;
            ExtendedQuery::Interval(IntervalQuery {
                field,
                source,
                score_function: function,
            })
        }
        NODE_COMBINED_FIELD => {
            let term = c.bytes()?.to_vec();
            let n = count(c, 1, nodes, "combined field query")?;
            *nodes = nodes.saturating_add(n);
            let mut fields = Vec::new();
            for _ in 0..n {
                let field = utf8(c.bytes()?)?.to_string();
                fields.push((field, f32::from_bits(c.i32()? as u32)));
            }
            ExtendedQuery::CombinedField(
                CombinedFieldQuery::new(term, fields)
                    .map_err(|e| refused("combined field query", e))?,
            )
        }
        _ => return crate::function_score::decode(c, start, depth, nodes),
    };
    Ok(Clause::Extended(Box::new(extended)))
}

// ---------------------------------------------------------------------------
// Spans
// ---------------------------------------------------------------------------

pub const SPAN_TERM: u8 = 0;
pub const SPAN_NEAR: u8 = 1;
pub const SPAN_OR: u8 = 2;
pub const SPAN_FIRST: u8 = 3;
pub const SPAN_POSITION_RANGE: u8 = 4;
pub const SPAN_NOT: u8 = 5;
pub const SPAN_CONTAINING: u8 = 6;
pub const SPAN_WITHIN: u8 = 7;
pub const SPAN_FIELD_MASKING: u8 = 8;

/// One span query, a tag and its payload:
///
/// | tag | layout | Lucene |
/// |---|---|---|
/// | `0` | `field`, `term` | `SpanTermQuery` |
/// | `1` | `slop: i32`, `in_order: u8`, `count: i32`, clauses | `SpanNearQuery` |
/// | `2` | `count: i32`, clauses | `SpanOrQuery` |
/// | `3` | `end: i32`, clause | `SpanFirstQuery` |
/// | `4` | `start: i32`, `end: i32`, clause | `SpanPositionRangeQuery` |
/// | `5` | `pre: i32`, `post: i32`, include, exclude | `SpanNotQuery` |
/// | `6` | big, little | `SpanContainingQuery` |
/// | `7` | big, little | `SpanWithinQuery` |
/// | `8` | `field`, clause | `FieldMaskingSpanQuery` |
///
/// Clauses of different fields are refused, as Java's constructors refuse
/// them.
pub fn decode_span(
    c: &mut Cursor<'_>,
    depth: usize,
    nodes: &mut usize,
) -> Result<SpanNode, Status> {
    descend(depth, nodes)?;
    let built = |e| refused("span query", e);
    let list = |c: &mut Cursor<'_>, nodes: &mut usize| -> Result<Vec<SpanNode>, Status> {
        let n = count(c, 1, nodes, "span query")?;
        let mut out = Vec::new();
        for _ in 0..n {
            out.push(decode_span(c, depth + 1, nodes)?);
        }
        Ok(out)
    };
    let tag = c.u8()?;
    Ok(match tag {
        SPAN_TERM => {
            let field = utf8(c.bytes()?)?.to_string();
            SpanNode::term(field, c.bytes()?.to_vec())
        }
        SPAN_NEAR => {
            let slop = c.i32()?;
            let in_order = c.u8()? != 0;
            SpanNode::near(list(c, nodes)?, slop, in_order).map_err(built)?
        }
        SPAN_OR => SpanNode::or(list(c, nodes)?).map_err(built)?,
        SPAN_FIRST => {
            let end = c.i32()?;
            SpanNode::first(decode_span(c, depth + 1, nodes)?, end)
        }
        SPAN_POSITION_RANGE => {
            let (start, end) = (c.i32()?, c.i32()?);
            SpanNode::position_range(decode_span(c, depth + 1, nodes)?, start, end)
        }
        SPAN_NOT => {
            let (pre, post) = (c.i32()?, c.i32()?);
            let include = decode_span(c, depth + 1, nodes)?;
            let exclude = decode_span(c, depth + 1, nodes)?;
            SpanNode::not(include, exclude, pre, post).map_err(built)?
        }
        SPAN_CONTAINING | SPAN_WITHIN => {
            let big = decode_span(c, depth + 1, nodes)?;
            let little = decode_span(c, depth + 1, nodes)?;
            if tag == SPAN_CONTAINING {
                SpanNode::containing(big, little).map_err(built)?
            } else {
                SpanNode::within(big, little).map_err(built)?
            }
        }
        SPAN_FIELD_MASKING => {
            let field = utf8(c.bytes()?)?.to_string();
            SpanNode::field_masking(decode_span(c, depth + 1, nodes)?, field)
        }
        other => {
            return Err(invalid(format!(
                "query tree: unknown span tag {other} (expected 0..={SPAN_FIELD_MASKING})"
            )))
        }
    })
}

// ---------------------------------------------------------------------------
// Intervals
// ---------------------------------------------------------------------------

pub const SOURCE_TERM: u8 = 0;
pub const SOURCE_BLOCK: u8 = 1;
pub const SOURCE_DISJUNCTION: u8 = 2;
pub const SOURCE_ORDERED: u8 = 3;
pub const SOURCE_UNORDERED: u8 = 4;
pub const SOURCE_REPEATING: u8 = 5;
pub const SOURCE_FILTERED: u8 = 6;
pub const SOURCE_EXTENDED: u8 = 7;
pub const SOURCE_OFFSET: u8 = 8;
pub const SOURCE_FIXED_FIELD: u8 = 9;
pub const SOURCE_NO_MATCH: u8 = 10;
pub const SOURCE_CONTAINING: u8 = 11;
pub const SOURCE_CONTAINED_BY: u8 = 12;
pub const SOURCE_NOT_CONTAINING: u8 = 13;
pub const SOURCE_NOT_CONTAINED_BY: u8 = 14;
pub const SOURCE_OVERLAPPING: u8 = 15;
pub const SOURCE_NON_OVERLAPPING: u8 = 16;
pub const SOURCE_MIN_SHOULD_MATCH: u8 = 17;
pub const SOURCE_MULTI_TERM: u8 = 18;

/// `CompiledAutomaton.AUTOMATON_TYPE`'s cases a multi-term source carries.
pub const AUTOMATON_NONE: u8 = 0;
pub const AUTOMATON_ALL: u8 = 1;
pub const AUTOMATON_SINGLE: u8 = 2;
pub const AUTOMATON_NORMAL: u8 = 3;

/// The most states a transferred automaton may have (Lucene's default
/// determinization work limit is 10 000 states' worth of effort).
pub const MAX_AUTOMATON_STATES: usize = 1 << 16;

/// One `IntervalsSource`, as the Java object it was read from -- each tag one
/// class, its fields in order (`count: i32` before a list, at least one):
///
/// | tag | class | layout |
/// |---|---|---|
/// | `0` | `TermIntervalsSource` | `term` |
/// | `1` | `BlockIntervalsSource` | sources |
/// | `2` | `DisjunctionIntervalsSource` | `pull_up: u8`, sources |
/// | `3` / `4` | `Ordered`/`UnorderedIntervalsSource` | sources |
/// | `5` | `RepeatingIntervalsSource` | `count: i32` (at least 1), `name: u8` (`0` none, `1` `ORDERED`, `2` `UNORDERED`), source |
/// | `6` | `FilteredIntervalsSource` | `kind: u8`, then `0` `MaxGaps` / `1` `MaxWidth`: `n: i32`; `2` OpenSearch's `ScriptFilterSource`: `name` (bytes), a checked Painless interval filter script ([`crate::painless_score::PainlessInterval`]); then the source |
/// | `7` | `ExtendedIntervalsSource` | `before: i32`, `after: i32`, source |
/// | `8` | `OffsetIntervalsSource` | `before: u8`, source |
/// | `9` | `FixedFieldIntervalsSource` | `field`, source |
/// | `10` | `NoMatchIntervalsSource` | `reason` |
/// | `11`-`16` | containing, contained by, not containing, not contained by, overlapping, non-overlapping | the two sources in the Java constructor's order |
/// | `17` | `MinimumShouldMatchIntervalsSource` | `min: i32` (1..=count), sources |
/// | `18` | `MultiTermIntervalsSource` | `max_expansions: i32` (at most 1024), `pattern` (UTF-8), the `CompiledAutomaton`: `type: u8` (`0` none, `1` all, `2` single + `term`, `3` normal + its binary automaton: `states: i32`, then per state `accept: u8`, `transitions: i32`, each `dest: i32`, `min: i32`, `max: i32`) |
pub fn decode_source(
    c: &mut Cursor<'_>,
    depth: usize,
    nodes: &mut usize,
) -> Result<IntervalsSource, Status> {
    descend(depth, nodes)?;
    let sub = |c: &mut Cursor<'_>, nodes: &mut usize| -> Result<Box<IntervalsSource>, Status> {
        Ok(Box::new(decode_source(c, depth + 1, nodes)?))
    };
    let list = |c: &mut Cursor<'_>, nodes: &mut usize| -> Result<Vec<IntervalsSource>, Status> {
        let n = count(c, 1, nodes, "interval source")?;
        let mut out = Vec::new();
        for _ in 0..n {
            out.push(decode_source(c, depth + 1, nodes)?);
        }
        Ok(out)
    };
    let tag = c.u8()?;
    Ok(match tag {
        SOURCE_TERM => IntervalsSource::Term(c.bytes()?.to_vec()),
        SOURCE_BLOCK => IntervalsSource::Block(list(c, nodes)?),
        SOURCE_DISJUNCTION => {
            let pull_up = c.u8()? != 0;
            IntervalsSource::Disjunction {
                sources: list(c, nodes)?,
                pull_up,
            }
        }
        SOURCE_ORDERED => IntervalsSource::Ordered(list(c, nodes)?),
        SOURCE_UNORDERED => IntervalsSource::Unordered(list(c, nodes)?),
        SOURCE_REPEATING => {
            let count = c.i32()?;
            // Each copy is a sub-iterator with its own cached positions
            // (`DuplicateIntervals`), its matches and its `toString` part: a
            // count only the blob bounds would size an allocation from the
            // wire. Java builds one only from that many equal clauses of
            // `Intervals.ordered`/`unordered`, so the clause limit bounds it,
            // and each copy counts as a node.
            if count < 1
                || usize::try_from(count).map_or(true, |n| n > crate::clauses::MAX_CLAUSE_COUNT)
            {
                return Err(invalid(format!(
                    "query tree: a repeating interval source of {count} copies (want 1..={})",
                    crate::clauses::MAX_CLAUSE_COUNT
                )));
            }
            *nodes = nodes.saturating_add(usize::try_from(count).unwrap_or(usize::MAX));
            check_clause_count(*nodes)?;
            let name = match c.u8()? {
                0 => None,
                1 => Some("ORDERED"),
                2 => Some("UNORDERED"),
                other => {
                    return Err(invalid(format!(
                        "query tree: unknown repeating source name {other} (expected 0..=2)"
                    )))
                }
            };
            IntervalsSource::Repeating {
                source: sub(c, nodes)?,
                count,
                name,
            }
        }
        SOURCE_FILTERED => {
            let filter = match c.u8()? {
                0 => IntervalFilterKind::MaxGaps(c.i32()?),
                1 => IntervalFilterKind::MaxWidth(c.i32()?),
                2 => IntervalFilterKind::Script {
                    name: std::str::from_utf8(c.bytes()?)
                        .map_err(|_| Status::InvalidUtf8)?
                        .to_string(),
                    script: Arc::new(crate::painless_score::PainlessInterval::decode(c)?),
                },
                other => {
                    return Err(invalid(format!(
                        "query tree: unknown interval filter {other} (expected 0..=2)"
                    )))
                }
            };
            IntervalsSource::Filtered {
                source: sub(c, nodes)?,
                filter,
            }
        }
        SOURCE_EXTENDED => {
            let (before, after) = (c.i32()?, c.i32()?);
            IntervalsSource::Extended {
                source: sub(c, nodes)?,
                before,
                after,
            }
        }
        SOURCE_OFFSET => {
            let before = c.u8()? != 0;
            IntervalsSource::Offset {
                source: sub(c, nodes)?,
                before,
            }
        }
        SOURCE_FIXED_FIELD => {
            let field = utf8(c.bytes()?)?.to_string();
            IntervalsSource::FixedField {
                field,
                source: sub(c, nodes)?,
            }
        }
        SOURCE_NO_MATCH => IntervalsSource::NoMatch(utf8(c.bytes()?)?.to_string()),
        SOURCE_CONTAINING..=SOURCE_NON_OVERLAPPING => {
            let (a, b) = (sub(c, nodes)?, sub(c, nodes)?);
            match tag {
                SOURCE_CONTAINING => IntervalsSource::Containing { big: a, small: b },
                SOURCE_CONTAINED_BY => IntervalsSource::ContainedBy { small: a, big: b },
                SOURCE_NOT_CONTAINING => IntervalsSource::NotContaining {
                    minuend: a,
                    subtrahend: b,
                },
                SOURCE_NOT_CONTAINED_BY => IntervalsSource::NotContainedBy {
                    minuend: a,
                    subtrahend: b,
                },
                SOURCE_OVERLAPPING => IntervalsSource::Overlapping {
                    source: a,
                    reference: b,
                },
                _ => IntervalsSource::NonOverlapping {
                    minuend: a,
                    subtrahend: b,
                },
            }
        }
        SOURCE_MIN_SHOULD_MATCH => {
            let min = c.i32()?;
            let sources = list(c, nodes)?;
            if min < 1 || usize::try_from(min).map_or(true, |m| m > sources.len()) {
                return Err(invalid(format!(
                    "query tree: minimum should match {min} of {} interval sources",
                    sources.len()
                )));
            }
            IntervalsSource::MinimumShouldMatch {
                sources,
                min_should_match: min,
            }
        }
        SOURCE_MULTI_TERM => {
            let max_expansions = c.length()?;
            if max_expansions > crate::clauses::MAX_CLAUSE_COUNT {
                return Err(invalid(format!(
                    "query tree: maxExpansions [{max_expansions}] cannot be greater than {}",
                    crate::clauses::MAX_CLAUSE_COUNT
                )));
            }
            let name = utf8(c.bytes()?)?.to_string();
            let automaton = decode_compiled_automaton(c)?;
            IntervalsSource::MultiTerm {
                pattern: MultiTermPattern::Automaton {
                    automaton: Arc::new(automaton),
                    binary: true,
                },
                max_expansions,
                name,
            }
        }
        other => {
            return Err(invalid(format!(
                "query tree: unknown interval source tag {other} (expected 0..={SOURCE_MULTI_TERM})"
            )))
        }
    })
}

/// A `CompiledAutomaton` as the byte automaton its terms enumeration runs:
/// every case as a deterministic automaton over bytes.
pub fn decode_compiled_automaton(c: &mut Cursor<'_>) -> Result<Automaton, Status> {
    let mut a = Automaton::new();
    match c.u8()? {
        AUTOMATON_NONE => {
            a.create_state();
        }
        AUTOMATON_ALL => {
            let s = a.create_state();
            a.set_accept(s, true);
            a.add_transition(s, s, 0, 255);
        }
        AUTOMATON_SINGLE => {
            let term = c.bytes()?;
            let mut s = a.create_state();
            for &b in term {
                let next = a.create_state();
                a.add_transition(s, next, i32::from(b), i32::from(b));
                s = next;
            }
            a.set_accept(s, true);
        }
        AUTOMATON_NORMAL => {
            let states = c.length()?;
            if states == 0 || states > MAX_AUTOMATON_STATES {
                return Err(invalid(format!(
                    "query tree: an automaton of {states} states"
                )));
            }
            let last = i32::try_from(states).map_err(|_| Status::InvalidArgument)?;
            for _ in 0..states {
                a.create_state();
            }
            for s in 0..last {
                let accept = c.u8()? != 0;
                a.set_accept(s, accept);
                let transitions = c.length()?;
                for _ in 0..transitions {
                    let (dest, min, max) = (c.i32()?, c.i32()?, c.i32()?);
                    if !(0..last).contains(&dest)
                        || !(0..=255).contains(&min)
                        || !(min..=255).contains(&max)
                    {
                        return Err(invalid(format!(
                            "query tree: automaton transition {s} -> {dest} over {min}..={max}"
                        )));
                    }
                    a.add_transition(s, dest, min, max);
                }
            }
        }
        other => {
            return Err(invalid(format!(
                "query tree: unknown automaton type {other} (expected 0..=3)"
            )))
        }
    }
    a.finish_state();
    if !a.is_deterministic() {
        return Err(invalid(
            "query tree: a multi-term automaton must be deterministic",
        ));
    }
    Ok(a)
}

// ---------------------------------------------------------------------------
// The parent filter's cache
// ---------------------------------------------------------------------------

/// A segment core: its id, the 16 random bytes `SegmentInfo` carries, which
/// identify its immutable files (borrowed for a lookup, never cloned).
pub type SegmentKey = [u8; 16];

/// Every cached parent bit set, across requests and readers: a bit set
/// ignores deletions and a segment's core never changes, so an entry stays
/// right for as long as the segment exists.
///
/// Bounded twice: by [`MAX_CACHED_PARENT_BYTES`] of bit sets (cleared whole
/// when a new set would pass it), and by the open readers -- closing a JVM
/// reader drops the sets of every segment no open reader still holds
/// ([`retain_parent_sets`]).
#[derive(Default)]
pub struct ParentsCache {
    /// Segment -> (filter key -> its set there, `None` without parents).
    pub sets: HashMap<SegmentKey, HashMap<String, Option<Arc<FixedBitSet>>>>,
    pub bytes: usize,
    pub entries: usize,
}

impl ParentsCache {
    pub fn get(&self, segment: &SegmentKey, filter: &str) -> Option<Option<Arc<FixedBitSet>>> {
        self.sets.get(segment)?.get(filter).cloned()
    }

    pub fn insert(&mut self, segment: SegmentKey, filter: String, bits: Option<Arc<FixedBitSet>>) {
        let size = bits.as_ref().map_or(0, |b| b.len() / 8);
        if self.bytes.saturating_add(size) > MAX_CACHED_PARENT_BYTES {
            self.clear();
        }
        if let Some(old) = self.sets.entry(segment).or_default().insert(filter, bits) {
            self.bytes = self.bytes.saturating_sub(old.map_or(0, |b| b.len() / 8));
        } else {
            self.entries = self.entries.saturating_add(1);
        }
        self.bytes = self.bytes.saturating_add(size);
    }

    pub fn clear(&mut self) {
        self.sets.clear();
        self.bytes = 0;
        self.entries = 0;
    }

    pub fn retain(&mut self, live: &std::collections::HashSet<SegmentKey>) {
        self.sets.retain(|k, _| live.contains(k));
        self.entries = self.sets.values().map(HashMap::len).sum();
        self.bytes = self
            .sets
            .values()
            .flat_map(HashMap::values)
            .map(|b| b.as_ref().map_or(0, |b| b.len() / 8))
            .sum();
    }
}

pub fn parents_cache() -> &'static Mutex<ParentsCache> {
    static CACHE: OnceLock<Mutex<ParentsCache>> = OnceLock::new();
    CACHE.get_or_init(Mutex::default)
}

pub fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// At most this many bytes of parent bit sets are cached (64 MiB).
pub const MAX_CACHED_PARENT_BYTES: usize = 64 << 20;

/// Drops the cached sets of every segment outside `live` (the segments the
/// open JVM readers hold): called when a reader closes.
pub fn retain_parent_sets(live: &std::collections::HashSet<SegmentKey>) {
    lock(parents_cache()).retain(live);
}

/// `(entries, bytes)` of the parent bit-set cache, for the plugin's stats.
pub fn parent_cache_stats() -> (usize, usize) {
    let c = lock(parents_cache());
    (c.entries, c.bytes)
}

/// A nested query's parent filter: OpenSearch's
/// `BitsetFilterCache.getBitSetProducer(query)`, a `QueryBitSetProducer`
/// whose sets outlive the request.
#[derive(Debug)]
pub struct SharedParents {
    /// The filter node's bytes, hex: equal filters share their sets.
    pub key: String,
    pub query: BooleanQuery,
}

impl BitSetProducer for SharedParents {
    fn bit_set(&self, leaf: &OpenSegment<'_>) -> lucene_search::Result<Option<Arc<FixedBitSet>>> {
        let segment = leaf.reader.map(|r| r.segment_id());
        if let Some(segment) = &segment {
            if let Some(hit) = lock(parents_cache()).get(segment, &self.key) {
                return Ok(hit);
            }
        }
        let bits = QueryBitSetProducer::new(self.query.clone()).bit_set(leaf)?;
        if let Some(segment) = segment {
            lock(parents_cache()).insert(segment, self.key.clone(), bits.clone());
        }
        Ok(bits)
    }

    fn key(&self) -> String {
        format!("parents({})", self.key)
    }
}
