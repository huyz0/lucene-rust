//! The query-tree nodes M10 T10.7 adds to [`crate::jvm_reader`]'s
//! `QUERY_TREE` blob: the OpenSearch shapes `lucene-join` and
//! `lucene-queries` back natively.
//!
//! | kind | layout after the kind byte | Lucene query (OpenSearch DSL) |
//! |---|---|---|
//! | `20` to-parent block join | `score_mode: u8` (`ScoreMode`'s ordinal: `0` None, `1` Avg, `2` Max, `3` Total, `4` Min), the parent filter (a node), the child query (a node) | `ToParentBlockJoinQuery` with a `QueryBitSetProducer` (`nested`) |
//! | `21` span | a span tree ([`decode_span`]) | the `spans` package (`span_*`, `field_masking_span`) |
//! | `22` interval | `field`, `scoring: u8` (`0` saturation `pivot: f32`, `1` sigmoid `pivot: f32`, `exp: f32`), a source tree ([`decode_source`]) | `IntervalQuery` (`intervals`) |
//! | `23` combined field | `term`, `count: i32`, then `count` times `field`, `weight: f32` | `CombinedFieldQuery` (`combined_fields`) |
//! | `24` OpenSearch function score | [`crate::jvm_function_score`] | OpenSearch's `FunctionScoreQuery` (`function_score`) |
//!
//! Every node is decoded into the query `lucene-search` ports, through the
//! constructors that check what Java's check; a structure Java's would refuse
//! is [`FfiStatus::InvalidArgument`], never a panic. Nesting below these
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

use crate::error::{set_last_error, FfiStatus};
use crate::jvm_reader::Cursor;
use crate::query::{check_clause_count, MAX_CLAUSE_DEPTH};

pub(crate) const NODE_TO_PARENT: u8 = 20;
pub(crate) const NODE_SPAN: u8 = 21;
pub(crate) const NODE_INTERVAL: u8 = 22;
pub(crate) const NODE_COMBINED_FIELD: u8 = 23;
pub(crate) const NODE_FUNCTION_SCORE: u8 = 24;
/// The highest node kind the tree knows.
pub(crate) const NODE_LAST: u8 = NODE_FUNCTION_SCORE;

/// [`FfiStatus::InvalidArgument`] with `message` as the last error.
pub(crate) fn invalid(message: impl Into<String>) -> FfiStatus {
    set_last_error(message.into());
    FfiStatus::InvalidArgument
}

/// A query a constructor refused, as the decode's error.
fn refused(what: &str, e: lucene_search::Error) -> FfiStatus {
    invalid(format!("query tree: {what}: {e}"))
}

fn utf8(b: &[u8]) -> Result<&str, FfiStatus> {
    std::str::from_utf8(b).map_err(|_| FfiStatus::InvalidUtf8)
}

/// One more nested level and one more node, against the tree's limits.
fn descend(depth: usize, nodes: &mut usize) -> Result<(), FfiStatus> {
    if depth >= MAX_CLAUSE_DEPTH {
        return Err(invalid(format!(
            "query tree: nesting depth exceeds the maximum of {MAX_CLAUSE_DEPTH}"
        )));
    }
    *nodes = nodes.saturating_add(1);
    check_clause_count(*nodes)
}

/// A count of children, at least `min`, each also counted as a node.
fn count(
    c: &mut Cursor<'_>,
    min: usize,
    nodes: &mut usize,
    what: &str,
) -> Result<usize, FfiStatus> {
    let n = c.len()?;
    if n < min {
        return Err(invalid(format!(
            "query tree: {what} with {n} sources (want at least {min})"
        )));
    }
    check_clause_count(nodes.saturating_add(n))?;
    Ok(n)
}

/// A node of kind 20-24, its kind byte read; `start` is the node's offset.
pub(crate) fn decode(
    c: &mut Cursor<'_>,
    kind: u8,
    start: usize,
    depth: usize,
    nodes: &mut usize,
) -> Result<Clause, FfiStatus> {
    let extended = match kind {
        NODE_TO_PARENT => {
            let score_mode = match c.u8()? {
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
            };
            let filter_start = c.pos();
            let parents = crate::jvm_reader::decode_node(c, depth + 1, nodes)?;
            let key = crate::jvm_reader::hex(c.since(filter_start));
            let child = crate::jvm_reader::decode_node(c, depth + 1, nodes)?;
            let parents: Arc<dyn BitSetProducer> = Arc::new(SharedParents {
                key,
                query: BooleanQuery {
                    must: vec![parents],
                    ..Default::default()
                },
            });
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
        _ => return crate::jvm_function_score::decode(c, start, depth, nodes),
    };
    Ok(Clause::Extended(Box::new(extended)))
}

// ---------------------------------------------------------------------------
// Spans
// ---------------------------------------------------------------------------

const SPAN_TERM: u8 = 0;
const SPAN_NEAR: u8 = 1;
const SPAN_OR: u8 = 2;
const SPAN_FIRST: u8 = 3;
const SPAN_POSITION_RANGE: u8 = 4;
const SPAN_NOT: u8 = 5;
const SPAN_CONTAINING: u8 = 6;
const SPAN_WITHIN: u8 = 7;
const SPAN_FIELD_MASKING: u8 = 8;

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
pub(crate) fn decode_span(
    c: &mut Cursor<'_>,
    depth: usize,
    nodes: &mut usize,
) -> Result<SpanNode, FfiStatus> {
    descend(depth, nodes)?;
    let built = |e| refused("span query", e);
    let list = |c: &mut Cursor<'_>, nodes: &mut usize| -> Result<Vec<SpanNode>, FfiStatus> {
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

const SOURCE_TERM: u8 = 0;
const SOURCE_BLOCK: u8 = 1;
const SOURCE_DISJUNCTION: u8 = 2;
const SOURCE_ORDERED: u8 = 3;
const SOURCE_UNORDERED: u8 = 4;
const SOURCE_REPEATING: u8 = 5;
const SOURCE_FILTERED: u8 = 6;
const SOURCE_EXTENDED: u8 = 7;
const SOURCE_OFFSET: u8 = 8;
const SOURCE_FIXED_FIELD: u8 = 9;
const SOURCE_NO_MATCH: u8 = 10;
const SOURCE_CONTAINING: u8 = 11;
const SOURCE_CONTAINED_BY: u8 = 12;
const SOURCE_NOT_CONTAINING: u8 = 13;
const SOURCE_NOT_CONTAINED_BY: u8 = 14;
const SOURCE_OVERLAPPING: u8 = 15;
const SOURCE_NON_OVERLAPPING: u8 = 16;
const SOURCE_MIN_SHOULD_MATCH: u8 = 17;
const SOURCE_MULTI_TERM: u8 = 18;

/// `CompiledAutomaton.AUTOMATON_TYPE`'s cases a multi-term source carries.
const AUTOMATON_NONE: u8 = 0;
const AUTOMATON_ALL: u8 = 1;
const AUTOMATON_SINGLE: u8 = 2;
const AUTOMATON_NORMAL: u8 = 3;

/// The most states a transferred automaton may have (Lucene's default
/// determinization work limit is 10 000 states' worth of effort).
const MAX_AUTOMATON_STATES: usize = 1 << 16;

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
/// | `6` | `FilteredIntervalsSource` | `kind: u8` (`0` `MaxGaps`, `1` `MaxWidth`), `n: i32`, source |
/// | `7` | `ExtendedIntervalsSource` | `before: i32`, `after: i32`, source |
/// | `8` | `OffsetIntervalsSource` | `before: u8`, source |
/// | `9` | `FixedFieldIntervalsSource` | `field`, source |
/// | `10` | `NoMatchIntervalsSource` | `reason` |
/// | `11`-`16` | containing, contained by, not containing, not contained by, overlapping, non-overlapping | the two sources in the Java constructor's order |
/// | `17` | `MinimumShouldMatchIntervalsSource` | `min: i32` (1..=count), sources |
/// | `18` | `MultiTermIntervalsSource` | `max_expansions: i32` (at most 1024), `pattern` (UTF-8), the `CompiledAutomaton`: `type: u8` (`0` none, `1` all, `2` single + `term`, `3` normal + its binary automaton: `states: i32`, then per state `accept: u8`, `transitions: i32`, each `dest: i32`, `min: i32`, `max: i32`) |
pub(crate) fn decode_source(
    c: &mut Cursor<'_>,
    depth: usize,
    nodes: &mut usize,
) -> Result<IntervalsSource, FfiStatus> {
    descend(depth, nodes)?;
    let sub = |c: &mut Cursor<'_>, nodes: &mut usize| -> Result<Box<IntervalsSource>, FfiStatus> {
        Ok(Box::new(decode_source(c, depth + 1, nodes)?))
    };
    let list = |c: &mut Cursor<'_>, nodes: &mut usize| -> Result<Vec<IntervalsSource>, FfiStatus> {
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
                || usize::try_from(count).map_or(true, |n| n > crate::query::MAX_CLAUSE_COUNT)
            {
                return Err(invalid(format!(
                    "query tree: a repeating interval source of {count} copies (want 1..={})",
                    crate::query::MAX_CLAUSE_COUNT
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
            let kind = c.u8()?;
            let n = c.i32()?;
            let filter = match kind {
                0 => IntervalFilterKind::MaxGaps(n),
                1 => IntervalFilterKind::MaxWidth(n),
                other => {
                    return Err(invalid(format!(
                        "query tree: unknown interval filter {other} (expected 0 or 1)"
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
            let max_expansions = c.len()?;
            if max_expansions > crate::query::MAX_CLAUSE_COUNT {
                return Err(invalid(format!(
                    "query tree: maxExpansions [{max_expansions}] cannot be greater than {}",
                    crate::query::MAX_CLAUSE_COUNT
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
fn decode_compiled_automaton(c: &mut Cursor<'_>) -> Result<Automaton, FfiStatus> {
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
            let states = c.len()?;
            if states == 0 || states > MAX_AUTOMATON_STATES {
                return Err(invalid(format!(
                    "query tree: an automaton of {states} states"
                )));
            }
            let last = i32::try_from(states).map_err(|_| FfiStatus::InvalidArgument)?;
            for _ in 0..states {
                a.create_state();
            }
            for s in 0..last {
                let accept = c.u8()? != 0;
                a.set_accept(s, accept);
                let transitions = c.len()?;
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
type SegmentKey = [u8; 16];

/// Every cached parent bit set, across requests and readers: a bit set
/// ignores deletions and a segment's core never changes, so an entry stays
/// right for as long as the segment exists.
///
/// Bounded twice: by [`MAX_CACHED_PARENT_BYTES`] of bit sets (cleared whole
/// when a new set would pass it), and by the open readers -- closing a JVM
/// reader drops the sets of every segment no open reader still holds
/// ([`retain_parent_sets`]).
#[derive(Default)]
struct ParentsCache {
    /// Segment -> (filter key -> its set there, `None` without parents).
    sets: HashMap<SegmentKey, HashMap<String, Option<Arc<FixedBitSet>>>>,
    bytes: usize,
    entries: usize,
}

impl ParentsCache {
    fn get(&self, segment: &SegmentKey, filter: &str) -> Option<Option<Arc<FixedBitSet>>> {
        self.sets.get(segment)?.get(filter).cloned()
    }

    fn insert(&mut self, segment: SegmentKey, filter: String, bits: Option<Arc<FixedBitSet>>) {
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

    fn clear(&mut self) {
        self.sets.clear();
        self.bytes = 0;
        self.entries = 0;
    }

    fn retain(&mut self, live: &std::collections::HashSet<SegmentKey>) {
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

fn parents_cache() -> &'static Mutex<ParentsCache> {
    static CACHE: OnceLock<Mutex<ParentsCache>> = OnceLock::new();
    CACHE.get_or_init(Mutex::default)
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// At most this many bytes of parent bit sets are cached (64 MiB).
const MAX_CACHED_PARENT_BYTES: usize = 64 << 20;

/// Drops the cached sets of every segment outside `live` (the segments the
/// open JVM readers hold): called when a reader closes.
pub(crate) fn retain_parent_sets(live: &std::collections::HashSet<SegmentKey>) {
    lock(parents_cache()).retain(live);
}

/// `(entries, bytes)` of the parent bit-set cache, for the plugin's stats.
pub(crate) fn parent_cache_stats() -> (usize, usize) {
    let c = lock(parents_cache());
    (c.entries, c.bytes)
}

/// A nested query's parent filter: OpenSearch's
/// `BitsetFilterCache.getBitSetProducer(query)`, a `QueryBitSetProducer`
/// whose sets outlive the request.
#[derive(Debug)]
pub(crate) struct SharedParents {
    /// The filter node's bytes, hex: equal filters share their sets.
    key: String,
    query: BooleanQuery,
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

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::jvm_reader::tests::run_limit;
    use crate::jvm_reader::{ffi_close_jvm_reader, ffi_open_jvm_reader, QUERY_TREE};
    use lucene_index::document::{
        Document, NumericDocValuesField, SortedNumericDocValuesField, SortedSetDocValuesField,
        Store, StringField, TextField,
    };
    use lucene_index::index_writer::IndexWriter;
    use lucene_index::segment_info::LuceneVersion;
    use lucene_store::FsDirectory;
    use lucene_util::test_support::TempDir;

    /// A little-endian blob, as the Java encoders write one.
    #[derive(Default, Clone)]
    pub(crate) struct Blob(pub(crate) Vec<u8>);

    impl Blob {
        pub(crate) fn tree() -> Self {
            Blob(vec![QUERY_TREE])
        }
        pub(crate) fn u8(mut self, v: u8) -> Self {
            self.0.push(v);
            self
        }
        pub(crate) fn i32(mut self, v: i32) -> Self {
            self.0.extend_from_slice(&v.to_le_bytes());
            self
        }
        pub(crate) fn f32(self, v: f32) -> Self {
            self.i32(v.to_bits() as i32)
        }
        pub(crate) fn f64(mut self, v: f64) -> Self {
            self.0.extend_from_slice(&v.to_bits().to_le_bytes());
            self
        }
        pub(crate) fn bytes(self, b: &[u8]) -> Self {
            let mut s = self.i32(b.len() as i32);
            s.0.extend_from_slice(b);
            s
        }
        pub(crate) fn str(self, s: &str) -> Self {
            self.bytes(s.as_bytes())
        }
        pub(crate) fn raw(mut self, b: &Blob) -> Self {
            self.0.extend_from_slice(&b.0);
            self
        }
        /// A term node.
        pub(crate) fn term(self, field: &str, t: &str) -> Self {
            self.u8(0).str(field).str(t)
        }
        /// An exists node.
        pub(crate) fn exists(self, field: &str) -> Self {
            self.u8(13).str(field)
        }
    }

    const VERSION: LuceneVersion = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };

    pub(crate) const BODIES: [&str; 6] = [
        "alpha beta gamma",
        "beta alpha delta alpha",
        "gamma gamma alpha",
        "delta beta",
        "alpha delta beta gamma alpha",
        "epsilon",
    ];

    /// Twelve blocks in two segments, OpenSearch's nested layout: block `i`
    /// has `i % 3` nested `comments` before its root, which carries `body`,
    /// `title`, `_primary_term`, and the numeric and keyword doc values the
    /// function scores read (`pos` = `i + 1`, `sd` a double, `sf` a float,
    /// `si` two integers on even blocks, `kw`).
    pub(crate) fn index(tag: &str) -> TempDir {
        let tmp = TempDir::new(tag);
        let dir = FsDirectory::open(tmp.path());
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
        for i in 0..12usize {
            let mut docs = Vec::new();
            for j in 0..i % 3 {
                let mut c = Document::new();
                c.add(StringField::new("_nested_path", "comments", Store::No));
                c.add(TextField::new(
                    "c_body",
                    if (i + j) % 2 == 0 {
                        "alpha"
                    } else {
                        "beta alpha"
                    },
                    Store::No,
                ));
                docs.push(c);
            }
            let mut d = Document::new();
            d.add(StringField::new("id", i.to_string(), Store::Yes));
            d.add(TextField::new("body", BODIES[i % 6], Store::No));
            d.add(TextField::new("title", BODIES[(i + 2) % 6], Store::No));
            d.add(NumericDocValuesField::new("_primary_term", 1));
            d.add(SortedNumericDocValuesField::new("pos", i as i64 + 1));
            let sd = (i as f64 - 5.5) * 3.25;
            let bits = sd.to_bits() as i64;
            d.add(SortedNumericDocValuesField::new(
                "sd",
                bits ^ ((bits >> 63) & i64::MAX),
            ));
            let sf = (i as f32) * 0.75 - 2.0;
            let fbits = sf.to_bits() as i32;
            d.add(SortedNumericDocValuesField::new(
                "sf",
                i64::from(fbits ^ ((fbits >> 31) & i32::MAX)),
            ));
            if i % 2 == 0 {
                d.add(SortedNumericDocValuesField::new("si", i as i64));
                d.add(SortedNumericDocValuesField::new("si", 3 * i as i64 + 1));
            }
            d.add(SortedSetDocValuesField::new(
                "kw",
                format!("k{}", i % 4).into_bytes(),
            ));
            docs.push(d);
            w.add_fields_documents(&docs).unwrap();
            if i == 5 {
                w.commit().unwrap();
            }
        }
        w.commit().unwrap();
        tmp
    }

    /// The index opened as the plugin opens a searcher's reader.
    pub(crate) fn open(tmp: &TempDir) -> u64 {
        let dir = FsDirectory::open(tmp.path());
        let reader = lucene_search::directory_reader::DirectoryReader::open(&dir).unwrap();
        let name = std::fs::read_dir(tmp.path())
            .unwrap()
            .filter_map(|e| e.ok()?.file_name().into_string().ok())
            .filter(|n| n.starts_with("segments_"))
            .max()
            .unwrap();
        let generation = i64::from_str_radix(&name["segments_".len()..], 36).unwrap();
        let bytes = std::fs::read(tmp.path().join(&name)).unwrap();
        let max_docs: Vec<i32> = reader.segment_readers().iter().map(|s| s.max_doc).collect();
        let path = tmp.path().to_str().unwrap();
        let mut handle = 0u64;
        // SAFETY: live buffers of the stated lengths.
        let rc = unsafe {
            ffi_open_jvm_reader(
                path.as_ptr().cast(),
                path.len(),
                bytes.as_ptr(),
                bytes.len(),
                generation,
                0,
                max_docs.as_ptr(),
                max_docs.len(),
                std::ptr::null(),
                std::ptr::null(),
                &mut handle,
            )
        };
        assert_eq!(rc, 0, "{}", crate::error::last_error());
        handle
    }

    /// `(hits, total)` of `blob`, top 20, counted exactly.
    pub(crate) fn search(h: u64, blob: &Blob) -> (Vec<(i32, f32)>, i64) {
        let (hits, total, _) = run_limit(h, &blob.0, 20, i64::MAX)
            .unwrap_or_else(|rc| panic!("status {rc}: {}", crate::error::last_error()));
        (hits, total)
    }

    /// The status a blob fails with, and the last error.
    pub(crate) fn fails(h: u64, blob: &Blob) -> (i32, String) {
        match run_limit(h, &blob.0, 5, i64::MAX) {
            Ok(r) => panic!("expected an error, got {r:?}"),
            Err(rc) => (rc, crate::error::last_error()),
        }
    }

    fn decode(blob: &Blob) -> std::result::Result<Clause, FfiStatus> {
        let mut c = Cursor::new(&blob.0[1..]);
        let mut nodes = 0;
        crate::jvm_reader::decode_node(&mut c, 0, &mut nodes)
    }

    /// The roots: documents with `_primary_term`.
    fn roots(h: u64) -> Vec<i32> {
        let mut d: Vec<i32> = search(h, &Blob::tree().exists("_primary_term"))
            .0
            .into_iter()
            .map(|(d, _)| d)
            .collect();
        d.sort_unstable();
        d
    }

    fn nested(mode: u8, child: Blob) -> Blob {
        Blob::tree()
            .u8(NODE_TO_PARENT)
            .u8(mode)
            .exists("_primary_term")
            .u8(1)
            .i32(0)
            .i32(2)
            .u8(0)
            .raw(&child)
            .u8(1)
            .term("_nested_path", "comments")
    }

    #[test]
    fn a_nested_query_finds_the_roots_of_matching_children_in_every_score_mode() {
        let tmp = index("jvm-nodes-nested");
        let h = open(&tmp);
        let roots = roots(h);
        assert_eq!(roots.len(), 12);
        // Blocks 1, 2, 4, 5, 7, 8, 10, 11 have children, all with "alpha".
        let mut scores = Vec::new();
        for mode in 0..=4u8 {
            let (hits, total) = search(h, &nested(mode, Blob::default().term("c_body", "alpha")));
            assert_eq!(total, 8, "mode {mode}");
            assert!(
                hits.iter().all(|(d, _)| roots.contains(d)),
                "mode {mode}: {hits:?}"
            );
            scores.push(hits.iter().map(|&(_, s)| s).fold(0.0f32, f32::max));
        }
        // None scores 0; Total of two children is above Max.
        assert_eq!(scores[0], 0.0);
        assert!(scores[3] > scores[2], "{scores:?}");
        // "beta" is on half the children only.
        let (_, beta) = search(h, &nested(2, Blob::default().term("c_body", "beta")));
        assert!(beta > 0 && beta < 8, "{beta}");
        // Equal filters share their cached sets (and the cache survives a second run).
        let again = search(h, &nested(1, Blob::default().term("c_body", "alpha")));
        assert_eq!(again.1, 8);
        // A child query that matches a root is Lucene's IllegalStateException.
        let unfiltered = Blob::tree()
            .u8(NODE_TO_PARENT)
            .u8(1)
            .exists("_primary_term")
            .u8(5);
        let (rc, msg) = fails(h, &unfiltered);
        assert_eq!(rc, FfiStatus::Search.code(), "{msg}");
        ffi_close_jvm_reader(h);
    }

    #[test]
    fn the_parent_cache_is_bounded_by_bytes_and_by_the_open_segments() {
        let p = SharedParents {
            key: "aa".into(),
            query: BooleanQuery::default(),
        };
        assert_eq!(p.key(), "parents(aa)");
        // A cache of its own: the process-wide one is shared by parallel tests.
        let mut c = ParentsCache::default();
        let seg = |n: &str| [n.as_bytes()[0]; 16];
        let set = |bits: usize| Some(Arc::new(FixedBitSet::new(bits)));
        c.insert(seg("a"), "f".into(), set(8 * 1024));
        c.insert(seg("a"), "g".into(), None);
        c.insert(seg("b"), "f".into(), set(8 * 2048));
        assert_eq!((c.entries, c.bytes), (3, 3072));
        assert!(c.get(&seg("a"), "g").is_some_and(|b| b.is_none()));
        assert!(c.get(&seg("a"), "h").is_none());
        // Replacing an entry accounts for the old set.
        c.insert(seg("b"), "f".into(), set(8 * 1024));
        assert_eq!((c.entries, c.bytes), (3, 2048));
        // Only the open segments' sets survive a reader closing.
        c.retain(&[seg("b")].into_iter().collect());
        assert_eq!((c.entries, c.bytes), (1, 1024));
        // A set that would pass the byte cap clears the cache first.
        c.insert(
            seg("c"),
            "f".into(),
            set(8 * (MAX_CACHED_PARENT_BYTES - 512)),
        );
        assert_eq!(c.entries, 1);
        assert!(c.get(&seg("b"), "f").is_none());
        c.clear();
        assert_eq!((c.entries, c.bytes), (0, 0));
    }

    /// The index opened with block 1 (its child, document 1, and root,
    /// document 2, in the first segment) deleted.
    fn open_without_block_1(tmp: &TempDir) -> u64 {
        let dir = FsDirectory::open(tmp.path());
        let reader = lucene_search::directory_reader::DirectoryReader::open(&dir).unwrap();
        let name = std::fs::read_dir(tmp.path())
            .unwrap()
            .filter_map(|e| e.ok()?.file_name().into_string().ok())
            .filter(|n| n.starts_with("segments_"))
            .max()
            .unwrap();
        let generation = i64::from_str_radix(&name["segments_".len()..], 36).unwrap();
        let bytes = std::fs::read(tmp.path().join(&name)).unwrap();
        let max_docs: Vec<i32> = reader.segment_readers().iter().map(|s| s.max_doc).collect();
        let words: Vec<u64> = vec![
            ((1u64 << max_docs[0]) - 1) & !0b110,
            (1u64 << max_docs[1]) - 1,
        ];
        let counts = [1usize, 1];
        let path = tmp.path().to_str().unwrap();
        let mut handle = 0u64;
        // SAFETY: live buffers of the stated lengths.
        let rc = unsafe {
            ffi_open_jvm_reader(
                path.as_ptr().cast(),
                path.len(),
                bytes.as_ptr(),
                bytes.len(),
                generation,
                0,
                max_docs.as_ptr(),
                max_docs.len(),
                words.as_ptr(),
                counts.as_ptr(),
                &mut handle,
            )
        };
        assert_eq!(rc, 0, "{}", crate::error::last_error());
        handle
    }

    #[test]
    fn parent_sets_are_shared_across_readers_and_dropped_with_their_segments() {
        let tmp = index("jvm-nodes-cache");
        let dir = FsDirectory::open(tmp.path());
        let reader = lucene_search::directory_reader::DirectoryReader::open(&dir).unwrap();
        let segments: Vec<SegmentKey> = reader
            .segment_readers()
            .iter()
            .map(|r| r.segment_id())
            .collect();
        let filter = crate::jvm_reader::hex(&Blob::default().exists("_primary_term").0);
        let cached = |s: &SegmentKey| lock(parents_cache()).get(s, &filter);
        assert!(segments.iter().all(|s| cached(s).is_none()));
        let h = open(&tmp);
        let blob = nested(2, Blob::default().term("c_body", "alpha"));
        let (first, total) = search(h, &blob);
        assert_eq!(total, 8);
        // Both segments' sets are cached, with the parents the filter matches.
        for s in &segments {
            let bits = cached(s).expect("cached").expect("parents");
            assert_eq!(bits.cardinality(), 6);
        }
        // A reader over the same segments with block 1 deleted reuses the sets
        // (deletions are not in them) and no longer finds that block.
        let h2 = open_without_block_1(&tmp);
        let set = cached(&segments[0]).unwrap().unwrap();
        let (hits, total) = search(h2, &blob);
        assert_eq!(total, 7);
        assert!(hits.iter().all(|&(d, _)| d != 2), "{hits:?}");
        assert!(first.iter().any(|&(d, _)| d == 2), "{first:?}");
        assert!(Arc::ptr_eq(&set, &cached(&segments[0]).unwrap().unwrap()));
        // Closing one reader keeps the sets the other still reads; closing the
        // last drops them.
        assert_eq!(ffi_close_jvm_reader(h), 0);
        assert!(segments.iter().all(|s| cached(s).is_some()));
        assert_eq!(ffi_close_jvm_reader(h2), 0);
        assert!(segments.iter().all(|s| cached(s).is_none()));
        // The process-wide counts track the cache (other tests may hold sets).
        let (entries, bytes) = parent_cache_stats();
        assert!(entries > 0 || bytes == 0);
    }

    fn span(tag: u8) -> Blob {
        Blob::default().u8(tag)
    }

    fn st(t: &str) -> Blob {
        span(SPAN_TERM).str("body").str(t)
    }

    fn spans(blob: Blob) -> Blob {
        Blob::tree().u8(NODE_SPAN).raw(&blob)
    }

    fn total(h: u64, blob: Blob) -> i64 {
        search(h, &blob).1
    }

    #[test]
    fn span_queries_decode_into_every_span_class() {
        let tmp = index("jvm-nodes-spans");
        let h = open(&tmp);
        // "alpha" is in bodies 0, 1, 2, 4 of every six.
        assert_eq!(total(h, spans(st("alpha"))), 8);
        let near = |slop: i32, ordered: u8| {
            span(SPAN_NEAR)
                .i32(slop)
                .u8(ordered)
                .i32(2)
                .raw(&st("alpha"))
                .raw(&st("beta"))
        };
        // "alpha beta" adjacent in order: bodies 0 only ("alpha beta gamma").
        assert_eq!(total(h, spans(near(0, 1))), 2);
        // Unordered within 1: bodies 0, 1 ("beta alpha"), 4 ("delta beta" ... "alpha delta beta").
        assert!(total(h, spans(near(1, 0))) >= 4);
        let or = span(SPAN_OR).i32(2).raw(&st("epsilon")).raw(&st("delta"));
        assert_eq!(total(h, spans(or)), 8);
        assert_eq!(
            total(h, spans(span(SPAN_FIRST).i32(1).raw(&st("alpha")))),
            4
        );
        assert_eq!(
            total(
                h,
                spans(span(SPAN_POSITION_RANGE).i32(1).i32(2).raw(&st("alpha")))
            ),
            2
        );
        let not = span(SPAN_NOT)
            .i32(1)
            .i32(1)
            .raw(&st("alpha"))
            .raw(&st("beta"));
        assert!(total(h, spans(not)) > 0);
        let big = span(SPAN_NEAR)
            .i32(5)
            .u8(0)
            .i32(2)
            .raw(&st("alpha"))
            .raw(&st("gamma"));
        let containing = span(SPAN_CONTAINING).raw(&big).raw(&st("beta"));
        assert!(total(h, spans(containing)) > 0);
        let within = span(SPAN_WITHIN).raw(&big).raw(&st("beta"));
        assert!(total(h, spans(within)) > 0);
        let masked = span(SPAN_NEAR).i32(3).u8(0).i32(2).raw(&st("alpha")).raw(
            &span(SPAN_FIELD_MASKING)
                .str("body")
                .raw(&span(SPAN_TERM).str("title").str("gamma")),
        );
        assert!(total(h, spans(masked)) > 0);
        // Malformed: an unknown tag, an empty near, clauses of two fields.
        let (rc, msg) = fails(h, &spans(span(99)));
        assert_eq!(rc, FfiStatus::InvalidArgument.code());
        assert!(msg.contains("unknown span tag 99"), "{msg}");
        let (_, msg) = fails(h, &spans(span(SPAN_OR).i32(0)));
        assert!(msg.contains("at least 1"), "{msg}");
        let two_fields = span(SPAN_OR)
            .i32(2)
            .raw(&st("a"))
            .raw(&span(SPAN_TERM).str("title").str("b"));
        let (_, msg) = fails(h, &spans(two_fields));
        assert!(msg.contains("span query"), "{msg}");
        let (_, msg) = fails(
            h,
            &spans(
                span(SPAN_NOT)
                    .i32(0)
                    .i32(0)
                    .raw(&st("a"))
                    .raw(&span(SPAN_TERM).str("title").str("b")),
            ),
        );
        assert!(msg.contains("span query"), "{msg}");
        let (_, msg) = fails(
            h,
            &spans(
                span(SPAN_CONTAINING)
                    .raw(&st("a"))
                    .raw(&span(SPAN_TERM).str("title").str("b")),
            ),
        );
        assert!(msg.contains("span query"), "{msg}");
        let (_, msg) = fails(
            h,
            &spans(
                span(SPAN_WITHIN)
                    .raw(&st("a"))
                    .raw(&span(SPAN_TERM).str("title").str("b")),
            ),
        );
        assert!(msg.contains("span query"), "{msg}");
        let (_, msg) = fails(
            h,
            &spans(
                span(SPAN_NEAR)
                    .i32(0)
                    .u8(1)
                    .i32(2)
                    .raw(&st("a"))
                    .raw(&span(SPAN_TERM).str("title").str("b")),
            ),
        );
        assert!(msg.contains("span query"), "{msg}");
        // Too deep: 40 nested span_first.
        let mut deep = st("alpha");
        for _ in 0..40 {
            deep = span(SPAN_FIRST).i32(9).raw(&deep);
        }
        let (_, msg) = fails(h, &spans(deep));
        assert!(msg.contains("nesting depth"), "{msg}");
        // Not UTF-8.
        let bad = spans(span(SPAN_TERM).bytes(&[0xff]).str("x"));
        assert_eq!(fails(h, &bad).0, FfiStatus::InvalidUtf8.code());
        ffi_close_jvm_reader(h);
    }

    fn src(tag: u8) -> Blob {
        Blob::default().u8(tag)
    }

    fn it(t: &str) -> Blob {
        src(SOURCE_TERM).str(t)
    }

    fn intervals(source: Blob) -> Blob {
        Blob::tree()
            .u8(NODE_INTERVAL)
            .str("body")
            .u8(0)
            .f32(1.0)
            .raw(&source)
    }

    fn list(tag: u8, items: &[Blob]) -> Blob {
        let mut b = src(tag).i32(items.len() as i32);
        for i in items {
            b = b.raw(i);
        }
        b
    }

    #[test]
    fn interval_sources_decode_into_every_source_class() {
        let tmp = index("jvm-nodes-intervals");
        let h = open(&tmp);
        assert_eq!(total(h, intervals(it("alpha"))), 8);
        // Sigmoid scoring.
        let sig = Blob::tree()
            .u8(NODE_INTERVAL)
            .str("body")
            .u8(1)
            .f32(2.0)
            .f32(0.5)
            .raw(&it("alpha"));
        assert_eq!(search(h, &sig).1, 8);
        let ab = [it("alpha"), it("beta")];
        assert_eq!(total(h, intervals(list(SOURCE_BLOCK, &ab))), 2);
        assert!(
            total(
                h,
                intervals(src(SOURCE_DISJUNCTION).u8(1).i32(2).raw(&ab[0]).raw(&ab[1]))
            ) >= 8
        );
        assert!(total(h, intervals(list(SOURCE_ORDERED, &ab))) > 0);
        assert!(total(h, intervals(list(SOURCE_UNORDERED, &ab))) > 0);
        // Two "alpha"s: bodies 1 and 4.
        let rep = src(SOURCE_REPEATING).i32(2).u8(1).raw(&it("alpha"));
        assert_eq!(total(h, intervals(rep)), 4);
        let rep2 = src(SOURCE_REPEATING).i32(2).u8(2).raw(&it("alpha"));
        assert_eq!(total(h, intervals(rep2.clone())), 4);
        let rep0 = src(SOURCE_REPEATING).i32(2).u8(0).raw(&it("alpha"));
        assert_eq!(total(h, intervals(rep0)), 4);
        let gaps = src(SOURCE_FILTERED)
            .u8(0)
            .i32(0)
            .raw(&list(SOURCE_UNORDERED, &ab));
        assert!(total(h, intervals(gaps)) > 0);
        let width = src(SOURCE_FILTERED)
            .u8(1)
            .i32(2)
            .raw(&list(SOURCE_UNORDERED, &ab));
        assert!(total(h, intervals(width)) > 0);
        assert_eq!(
            total(
                h,
                intervals(src(SOURCE_EXTENDED).i32(1).i32(1).raw(&it("alpha")))
            ),
            8
        );
        assert!(total(h, intervals(src(SOURCE_OFFSET).u8(1).raw(&it("beta")))) > 0);
        assert!(total(h, intervals(src(SOURCE_OFFSET).u8(0).raw(&it("beta")))) > 0);
        assert!(
            total(
                h,
                intervals(src(SOURCE_FIXED_FIELD).str("title").raw(&it("gamma")))
            ) > 0
        );
        assert_eq!(total(h, intervals(src(SOURCE_NO_MATCH).str("nothing"))), 0);
        let big = list(SOURCE_UNORDERED, &ab);
        for tag in SOURCE_CONTAINING..=SOURCE_NON_OVERLAPPING {
            let (a, b) = if tag == SOURCE_CONTAINED_BY {
                (it("gamma"), big.clone())
            } else {
                (big.clone(), it("gamma"))
            };
            search(h, &intervals(src(tag).raw(&a).raw(&b)));
        }
        let msm = src(SOURCE_MIN_SHOULD_MATCH)
            .i32(2)
            .i32(3)
            .raw(&it("alpha"))
            .raw(&it("beta"))
            .raw(&it("epsilon"));
        assert!(total(h, intervals(msm)) > 0);
        // Multi-term sources: every automaton type.
        let mt = |b: Blob| src(SOURCE_MULTI_TERM).i32(16).str("p").raw(&b);
        assert_eq!(
            total(h, intervals(mt(Blob::default().u8(AUTOMATON_NONE)))),
            0
        );
        assert_eq!(
            total(
                h,
                intervals(mt(Blob::default().u8(AUTOMATON_SINGLE).str("delta")))
            ),
            6
        );
        // `al*` over bytes: a -> l -> anything.
        let normal = Blob::default()
            .u8(AUTOMATON_NORMAL)
            .i32(3)
            .u8(0)
            .i32(1)
            .i32(1)
            .i32(i32::from(b'a'))
            .i32(i32::from(b'a'))
            .u8(0)
            .i32(1)
            .i32(2)
            .i32(i32::from(b'l'))
            .i32(i32::from(b'l'))
            .u8(1)
            .i32(1)
            .i32(2)
            .i32(0)
            .i32(255);
        assert_eq!(total(h, intervals(mt(normal))), 8);
        // Every term: more than 2 expansions is Lucene's error.
        let all = src(SOURCE_MULTI_TERM).i32(2).str("p").u8(AUTOMATON_ALL);
        let (rc, msg) = fails(h, &intervals(all));
        assert_eq!(rc, FfiStatus::Search.code(), "{msg}");
        ffi_close_jvm_reader(h);
    }

    #[test]
    fn malformed_interval_nodes_are_refused() {
        let tmp = index("jvm-nodes-intervals-bad");
        let h = open(&tmp);
        let bad = |b: Blob, want: &str| {
            let (rc, msg) = fails(h, &b);
            assert_eq!(rc, FfiStatus::InvalidArgument.code(), "{msg}");
            assert!(msg.contains(want), "{msg} lacks {want}");
        };
        bad(intervals(src(99)), "unknown interval source tag 99");
        bad(
            intervals(src(SOURCE_REPEATING).i32(0).u8(0).raw(&it("a"))),
            "0 copies",
        );
        // Each copy is an iterator and its cached positions: bounded as the
        // clauses Java's `Intervals.ordered` could have deduplicated.
        bad(
            intervals(src(SOURCE_REPEATING).i32(5000).u8(0).raw(&it("a"))),
            "5000 copies",
        );
        bad(
            intervals(src(SOURCE_REPEATING).i32(i32::MAX).u8(0).raw(&it("a"))),
            "copies",
        );
        bad(
            intervals(src(SOURCE_REPEATING).i32(2).u8(7).raw(&it("a"))),
            "repeating source name 7",
        );
        bad(
            intervals(src(SOURCE_FILTERED).u8(5).i32(1).raw(&it("a"))),
            "interval filter 5",
        );
        bad(intervals(list(SOURCE_ORDERED, &[])), "at least 1");
        bad(
            intervals(src(SOURCE_MIN_SHOULD_MATCH).i32(3).i32(1).raw(&it("a"))),
            "minimum should match 3",
        );
        bad(
            intervals(src(SOURCE_MIN_SHOULD_MATCH).i32(0).i32(1).raw(&it("a"))),
            "minimum should match 0",
        );
        let mt = |b: Blob| src(SOURCE_MULTI_TERM).i32(16).str("p").raw(&b);
        bad(
            intervals(src(SOURCE_MULTI_TERM).i32(5000).str("p").u8(0)),
            "maxExpansions [5000]",
        );
        bad(
            intervals(mt(Blob::default().u8(9))),
            "unknown automaton type 9",
        );
        bad(
            intervals(mt(Blob::default().u8(AUTOMATON_NORMAL).i32(0))),
            "automaton of 0 states",
        );
        bad(
            intervals(mt(Blob::default()
                .u8(AUTOMATON_NORMAL)
                .i32(1)
                .u8(1)
                .i32(1)
                .i32(5)
                .i32(0)
                .i32(1))),
            "transition 0 -> 5",
        );
        bad(
            intervals(mt(Blob::default()
                .u8(AUTOMATON_NORMAL)
                .i32(1)
                .u8(1)
                .i32(1)
                .i32(0)
                .i32(3)
                .i32(1))),
            "over 3..=1",
        );
        // Two transitions from one state over the same byte: not deterministic.
        let nfa = Blob::default()
            .u8(AUTOMATON_NORMAL)
            .i32(2)
            .u8(0)
            .i32(2)
            .i32(1)
            .i32(97)
            .i32(97)
            .i32(0)
            .i32(97)
            .i32(97)
            .u8(1)
            .i32(0);
        bad(intervals(mt(nfa)), "deterministic");
        let scoring = |kind: u8, pivot: f32| {
            Blob::tree()
                .u8(NODE_INTERVAL)
                .str("body")
                .u8(kind)
                .f32(pivot)
                .f32(1.0)
                .raw(&it("a"))
        };
        bad(scoring(7, 1.0), "unknown interval scoring 7");
        bad(scoring(0, -1.0), "pivot must be > 0");
        let mut deep = it("alpha");
        for _ in 0..40 {
            deep = src(SOURCE_EXTENDED).i32(0).i32(0).raw(&deep);
        }
        bad(intervals(deep), "nesting depth");
        assert_eq!(
            fails(
                h,
                &intervals(src(SOURCE_FIXED_FIELD).bytes(&[0xfe]).raw(&it("a")))
            )
            .0,
            FfiStatus::InvalidUtf8.code()
        );
        assert_eq!(
            fails(h, &intervals(src(SOURCE_NO_MATCH).bytes(&[0xfe]))).0,
            FfiStatus::InvalidUtf8.code()
        );
        assert_eq!(
            fails(
                h,
                &Blob::tree()
                    .u8(NODE_INTERVAL)
                    .bytes(&[0xfe])
                    .u8(0)
                    .f32(1.0)
                    .raw(&it("a"))
            )
            .0,
            FfiStatus::InvalidUtf8.code()
        );
        ffi_close_jvm_reader(h);
    }

    fn combined(fields: &[(&str, f32)], term: &str) -> Blob {
        let mut b = Blob::tree()
            .u8(NODE_COMBINED_FIELD)
            .str(term)
            .i32(fields.len() as i32);
        for (f, w) in fields {
            b = b.str(f).f32(*w);
        }
        b
    }

    #[test]
    fn combined_fields_and_the_other_node_errors() {
        let tmp = index("jvm-nodes-combined");
        let h = open(&tmp);
        let (hits, total) = search(h, &combined(&[("body", 1.0), ("title", 2.0)], "gamma"));
        assert!(
            total == 6 && hits.iter().all(|&(_, s)| s > 0.0),
            "{total} {hits:?}"
        );
        let (rc, msg) = fails(h, &combined(&[("body", 0.5)], "gamma"));
        assert_eq!(rc, FfiStatus::InvalidArgument.code());
        assert!(msg.contains("weight must be greater"), "{msg}");
        assert!(fails(h, &combined(&[], "gamma")).1.contains("at least 1"));
        assert_eq!(
            fails(
                h,
                &Blob::tree()
                    .u8(NODE_COMBINED_FIELD)
                    .str("t")
                    .i32(1)
                    .bytes(&[0xff])
                    .f32(1.0)
            )
            .0,
            FfiStatus::InvalidUtf8.code()
        );
        // An unknown score mode, and kinds past the last.
        let (_, msg) = fails(h, &Blob::tree().u8(NODE_TO_PARENT).u8(9));
        assert!(msg.contains("score mode 9"), "{msg}");
        let (_, msg) = fails(h, &Blob::tree().u8(NODE_LAST + 1));
        assert!(msg.contains("unknown node kind"), "{msg}");
        // The decode on its own: the nodes are extended clauses.
        assert!(matches!(decode(&spans(st("a"))), Ok(Clause::Extended(_))));
        ffi_close_jvm_reader(h);
    }

    /// The fuzz target's M10 seeds are well-formed blobs.
    #[test]
    fn the_fuzz_seeds_decode() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/fuzz/seeds/jvm_search");
        let mut seen = 0;
        for e in std::fs::read_dir(dir).unwrap() {
            let e = e.unwrap();
            if e.file_name().to_string_lossy().starts_with("m10_") {
                let data = std::fs::read(e.path()).unwrap();
                crate::jvm_reader::decode_request(&data[2..]).unwrap_or_else(|s| {
                    panic!("{:?}: {s:?} {}", e.path(), crate::error::last_error())
                });
                seen += 1;
            }
        }
        assert_eq!(seen, 5);
    }
}
