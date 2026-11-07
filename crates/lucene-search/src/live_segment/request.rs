//! From a query to what it reads from the writer's buffers, and from what
//! the buffers gave back to segments the query runs on.

use std::collections::{BTreeMap, BTreeSet};

use lucene_codecs::postings_writer::TermPostings;
use lucene_index::concurrent_writer::{
    FieldTotals, LiveBuffer, LiveBufferField, LiveFieldRequest, LivePattern, LiveRequest,
    LiveSnapshot,
};

use super::{LiveField, LiveSegment};
use crate::query::Clause;
use crate::Result;

/// The terms `clause` reads, by field, with whether it reads their
/// positions; `None` when it has a clause the buffers cannot answer yet
/// (multi-term, points, doc values, spans and other extended queries).
pub fn live_request(clause: &Clause) -> Option<LiveRequest> {
    let mut fields = Fields::new();
    collect(clause, &mut fields, false)?;
    Some(request_of(fields))
}

/// The terms `clauses` read, as [`live_request`], for searching segments
/// that hold the documents' doc values and points but none of their terms
/// ([`crate::directory_reader::SegmentReader::with_postings`]): a points
/// or doc values clause (ranges, geo) reads nothing from the buffers then,
/// and a prefix, wildcard or regexp clause reads the buffered terms it
/// matches.
pub fn whole_request(clauses: &[&Clause]) -> Option<LiveRequest> {
    let mut fields = Fields::new();
    for c in clauses {
        collect(c, &mut fields, true)?;
    }
    Some(request_of(fields))
}

fn request_of(fields: Fields<'_>) -> LiveRequest {
    LiveRequest {
        fields: fields
            .into_iter()
            .map(|(field, f)| LiveFieldRequest {
                field: field.to_string(),
                terms: f.terms.into_iter().map(<[u8]>::to_vec).collect(),
                positions: f.positions,
                patterns: f.patterns,
            })
            .collect(),
    }
}

/// What a request reads of one field.
#[derive(Default)]
struct Field<'a> {
    terms: BTreeSet<&'a [u8]>,
    positions: bool,
    patterns: Vec<LivePattern>,
}

type Fields<'a> = BTreeMap<&'a str, Field<'a>>;

fn add<'a>(fields: &mut Fields<'a>, field: &'a str, term: &'a [u8], positions: bool) {
    let e = fields.entry(field).or_default();
    e.terms.insert(term);
    e.positions |= positions;
}

fn add_pattern<'a>(fields: &mut Fields<'a>, field: &'a str, pattern: LivePattern) {
    fields.entry(field).or_default().patterns.push(pattern);
}

/// With `lean`, for segments holding doc values and points but no terms:
/// points and doc values clauses read nothing and multi-term clauses read
/// patterns.
fn collect<'a>(clause: &'a Clause, fields: &mut Fields<'a>, lean: bool) -> Option<()> {
    match clause {
        Clause::Term(t) => add(fields, &t.field, &t.term, false),
        Clause::TermInSet(q) => {
            for t in &q.terms {
                add(fields, &q.field, t, false);
            }
        }
        Clause::Phrase(p) => {
            for t in &p.terms {
                add(fields, &p.field, t, true);
            }
        }
        Clause::MultiPhrase(p) => {
            for t in p.term_arrays.iter().flatten() {
                add(fields, &p.field, t, true);
            }
        }
        Clause::Boolean(b) => {
            for c in b
                .must
                .iter()
                .chain(&b.filter)
                .chain(&b.should)
                .chain(&b.must_not)
            {
                collect(c, fields, lean)?;
            }
        }
        Clause::DisjunctionMax(d) => {
            for c in &d.disjuncts {
                collect(c, fields, lean)?;
            }
        }
        Clause::ConstantScore(c) => collect(&c.inner, fields, lean)?,
        Clause::Boost(b) => collect(&b.inner, fields, lean)?,
        Clause::MatchAllDocs(_) | Clause::MatchNoDocs(_) => {}
        Clause::PointsRange(_) if lean => {}
        Clause::Extended(e) if lean => match &**e {
            crate::extended_query::ExtendedQuery::PointRange(_)
            | crate::extended_query::ExtendedQuery::PointInSet(_)
            | crate::extended_query::ExtendedQuery::NumericDocValuesRange(_) => {}
            crate::extended_query::ExtendedQuery::IndexSortRange(q) => {
                collect(&q.fallback, fields, lean)?;
            }
            crate::extended_query::ExtendedQuery::IndexOrDocValues(q) => {
                collect(&q.index_query, fields, lean)?;
                collect(&q.dv_query, fields, lean)?;
            }
            crate::extended_query::ExtendedQuery::Document(d) if d.query.reads_no_terms() => {
                for c in d.filter.iter().chain(&d.fallback) {
                    collect(c, fields, lean)?;
                }
            }
            _ => return None,
        },
        Clause::Prefix(q) if lean => add_pattern(
            fields,
            &q.field,
            LivePattern::Wildcard(lucene_codecs::wildcard::WildcardPattern::prefix(&q.prefix)),
        ),
        Clause::Wildcard(q) if lean => add_pattern(
            fields,
            &q.field,
            LivePattern::Wildcard(lucene_codecs::wildcard::WildcardPattern::new(&q.pattern)),
        ),
        Clause::Regexp(q) if lean => {
            // Only one the dictionaries expand with a compiled automaton:
            // the buffers' terms are picked by the same one.
            let p = lucene_codecs::regexp::RegexpPattern::new(q.pattern.as_bytes()).ok()?;
            p.dfa_matches(b"")?;
            add_pattern(fields, &q.field, LivePattern::Regexp(p));
        }
        _ => return None,
    }
    Some(())
}

impl From<LiveBufferField> for LiveField {
    fn from(f: LiveBufferField) -> Self {
        LiveField {
            info: f.info,
            terms: f.terms,
            norms: f.norms,
            num_terms: f.num_terms,
            sum_total_term_freq: f.sum_total_term_freq,
            sum_doc_freq: f.sum_doc_freq,
            doc_count: f.doc_count,
        }
    }
}

/// One [`LiveSegment`] per buffer of `snapshot`, in its order.
pub fn live_segments(snapshot: LiveSnapshot) -> Result<Vec<LiveSegment>> {
    snapshot
        .buffers
        .into_iter()
        .map(|b| {
            let fields: Vec<LiveField> = b.fields.into_iter().map(LiveField::from).collect();
            LiveSegment::build(&fields, b.max_doc, &b.deleted)
        })
        .collect()
}

/// Documents `start..` of `buffer` as a segment of their own, numbered
/// from `start`, with their norms and deletes. Each field's statistics
/// are those of the documents: `at_start(field_number)` gives the
/// buffer's at `start`. From 0 it is [`live_segments`]' segment of the
/// buffer.
///
/// # Errors
/// What [`LiveSegment::build`] reports.
pub fn tail_segment(
    buffer: LiveBuffer,
    start: i32,
    at_start: &dyn Fn(i32) -> FieldTotals,
) -> Result<LiveSegment> {
    if start <= 0 {
        let fields: Vec<LiveField> = buffer.fields.into_iter().map(LiveField::from).collect();
        return LiveSegment::build(&fields, buffer.max_doc, &buffer.deleted);
    }
    let skip = usize::try_from(start).unwrap_or(usize::MAX);
    let fields: Vec<LiveField> = buffer
        .fields
        .iter()
        .map(|f| {
            let at = at_start(f.info.number);
            let freqs = !matches!(
                f.info.index_options,
                lucene_codecs::field_infos::IndexOptions::Docs
                    | lucene_codecs::field_infos::IndexOptions::None
            );
            let terms: Vec<TermPostings> = f
                .terms
                .iter()
                .filter_map(|t| range_of(t, start, buffer.max_doc))
                .collect();
            LiveField {
                info: f.info.clone(),
                num_terms: terms.len() as i64,
                terms,
                norms: f
                    .norms
                    .as_ref()
                    .map(|n| n.get(skip..).unwrap_or_default().to_vec()),
                sum_total_term_freq: f.sum_total_term_freq.saturating_sub(if freqs {
                    at.sum_total_term_freq
                } else {
                    at.sum_doc_freq
                }),
                sum_doc_freq: f.sum_doc_freq.saturating_sub(at.sum_doc_freq),
                doc_count: f.doc_count.saturating_sub(at.doc_count),
            }
        })
        .collect();
    let deleted: Vec<i32> = buffer
        .deleted
        .iter()
        .filter(|&&d| d >= start)
        .map(|&d| d.saturating_sub(start))
        .collect();
    LiveSegment::build(&fields, buffer.max_doc.saturating_sub(start), &deleted)
}

/// Documents `start..end` of `buffer`'s requested terms as a segment of
/// their own, numbered from `start`, for
/// [`crate::directory_reader::SegmentReader::with_postings`] over the
/// segment those documents were written to. Each field's statistics are
/// those of the documents: `totals(field_number)` gives the buffer's at
/// `start` and at `end`. No norms and no deletes: the segment has its own.
///
/// # Errors
/// What [`LiveSegment::build`] reports.
pub fn range_segment(
    buffer: &LiveBuffer,
    start: i32,
    end: i32,
    totals: &dyn Fn(i32) -> (FieldTotals, FieldTotals),
) -> Result<LiveSegment> {
    let fields: Vec<LiveField> = buffer
        .fields
        .iter()
        .map(|f| {
            let (from, to) = totals(f.info.number);
            let terms: Vec<TermPostings> = f
                .terms
                .iter()
                .filter_map(|t| range_of(t, start, end))
                .collect();
            LiveField {
                info: f.info.clone(),
                // The terms held, not the chunk's: a reader that takes a
                // dictionary as complete when its size is the field's
                // ordinal count must not take this one so.
                num_terms: terms.len() as i64,
                terms,
                norms: None,
                sum_total_term_freq: to
                    .sum_total_term_freq
                    .saturating_sub(from.sum_total_term_freq),
                sum_doc_freq: to.sum_doc_freq.saturating_sub(from.sum_doc_freq),
                doc_count: to.doc_count.saturating_sub(from.doc_count),
            }
        })
        .collect();
    LiveSegment::build(&fields, end.saturating_sub(start), &[])
}

/// `t`'s documents in `start..end`, renumbered from `start`; `None` when it
/// has none there.
fn range_of(t: &TermPostings, start: i32, end: i32) -> Option<TermPostings> {
    let lo = t.docs.partition_point(|&(d, _)| d < start);
    let hi = t.docs.partition_point(|&(d, _)| d < end);
    if lo == hi {
        return None;
    }
    // Payloads are one length per occurrence, the occurrences of every
    // document in turn.
    let occurrences = |docs: &[Vec<i32>]| docs.iter().map(Vec::len).sum::<usize>();
    let (occ_lo, occ_hi) = if t.payload_lengths.is_empty() {
        (0, 0)
    } else {
        let before = occurrences(t.positions.get(..lo).unwrap_or_default());
        (
            before,
            before + occurrences(t.positions.get(lo..hi).unwrap_or_default()),
        )
    };
    let byte = |occ: usize| -> usize {
        t.payload_lengths
            .get(..occ)
            .unwrap_or_default()
            .iter()
            .map(|&l| l as usize)
            .sum()
    };
    Some(TermPostings {
        term: t.term.clone(),
        docs: t.docs[lo..hi]
            .iter()
            .map(|&(d, f)| (d.saturating_sub(start), f))
            .collect(),
        positions: t.positions.get(lo..hi).unwrap_or_default().to_vec(),
        offsets: t.offsets.get(lo..hi).unwrap_or_default().to_vec(),
        payload_bytes: if t.payload_lengths.is_empty() {
            Vec::new()
        } else {
            t.payload_bytes
                .get(byte(occ_lo)..byte(occ_hi))
                .unwrap_or_default()
                .to_vec()
        },
        payload_lengths: t
            .payload_lengths
            .get(occ_lo..occ_hi)
            .unwrap_or_default()
            .to_vec(),
    })
}
