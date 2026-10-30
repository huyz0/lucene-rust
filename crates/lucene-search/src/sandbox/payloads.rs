//! `org.apache.lucene.payloads` (lucene-sandbox): `PayloadSpanUtil`, which
//! turns a query into span queries and returns the payload of every term
//! occurrence inside each of their spans, and the `PayloadSpanCollector` it
//! collects them with.
//!
//! `getPayloadsForQuery` walks the query: a boolean query's clauses that are
//! not `MUST_NOT`; a disjunction-max query's disjuncts; a `TermQuery` as a
//! `SpanTermQuery`; a `PhraseQuery` as a `SpanNearQuery` of its terms
//! (its slop, in order only at slop `0`; its positions ignored); a
//! `MultiPhraseQuery` as a `SpanNearQuery` of one `SpanOrQuery` per distinct
//! position, in position order, with the slop widened by the number of
//! unused positions below the largest; a span query as itself. Any other
//! query contributes nothing. Each span query is rewritten against the
//! searcher (a `SpanMultiTermQueryWrapper` expands), then every segment's
//! spans are read with payloads -- every document, deleted or not, as
//! `SpanWeight.getSpans` does not look at live docs -- and each span's
//! payloads collected in the order its spans hand them over, a position
//! without a payload contributing nothing.
//!
//! Differences: Java's `IndexReaderContext` is the [`IndexSearcher`] over
//! the same segments; a boolean query's clauses are walked `MUST`, `FILTER`
//! then `SHOULD` (this port's [`BooleanQuery`] keeps them by occurrence,
//! not interleaved as they were added); a core [`SpanQuery`] is read as the
//! [`SpanNode`] it denotes.

use crate::exec::{LeafContext, NO_MORE_DOCS};
use crate::extended_query::ExtendedQuery;
use crate::index_searcher::IndexSearcher;
use crate::multi_segment::OpenSegment;
use crate::query::{BooleanQuery, Clause, MultiPhraseQuery, PhraseQuery, SpanQuery};
use crate::spans::{self, SpanCollector, SpanNode, Spans, TermSpans, NO_MORE_POSITIONS};
use crate::Result;

/// `PayloadSpanCollector`: the payloads of the terms of the spans it is
/// handed, each copied out of its postings.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct PayloadSpanCollector {
    payloads: Vec<Vec<u8>>,
}

impl PayloadSpanCollector {
    /// `new PayloadSpanCollector()`.
    pub fn new() -> Self {
        Self::default()
    }

    /// `getPayloads()`.
    pub fn payloads(&self) -> &[Vec<u8>] {
        &self.payloads
    }

    /// `reset()`: forgets the payloads collected.
    pub fn reset(&mut self) {
        self.payloads.clear();
    }
}

impl SpanCollector for PayloadSpanCollector {
    /// `collectLeaf`: the occurrence's payload, when it has one.
    fn collect_leaf(&mut self, leaf: &mut TermSpans<'_>, _position: i32) -> Result<()> {
        if let Some(payload) = leaf.payload()? {
            self.payloads.push(payload.to_vec());
        }
        Ok(())
    }

    fn reset(&mut self) {
        PayloadSpanCollector::reset(self);
    }
}

/// `PayloadSpanUtil`: the payloads a query's spans cover, over a searcher's
/// segments.
#[derive(Clone, Copy)]
pub struct PayloadSpanUtil<'x, 's, 'a> {
    searcher: &'x IndexSearcher<'s, 'a>,
}

impl std::fmt::Debug for PayloadSpanUtil<'_, '_, '_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PayloadSpanUtil")
            .field("segments", &self.searcher.segments().len())
            .finish()
    }
}

impl<'x, 's, 'a> PayloadSpanUtil<'x, 's, 'a> {
    /// `new PayloadSpanUtil(context)`.
    pub fn new(searcher: &'x IndexSearcher<'s, 'a>) -> Self {
        PayloadSpanUtil { searcher }
    }

    /// `getPayloadsForQuery(query)`.
    ///
    /// # Errors
    /// What reading the postings reports; a span term of a field indexed
    /// without positions; a near query of fewer than two clauses (a phrase
    /// of one term); clauses of different fields.
    pub fn get_payloads_for_query(&self, query: &Clause) -> Result<Vec<Vec<u8>>> {
        let mut payloads = Vec::new();
        self.query_to_span_query(query, &mut payloads)?;
        Ok(payloads)
    }

    fn query_to_span_query(&self, query: &Clause, payloads: &mut Vec<Vec<u8>>) -> Result<()> {
        match query {
            Clause::Boolean(b) => {
                let BooleanQuery {
                    must,
                    filter,
                    should,
                    ..
                } = b.as_ref();
                for c in must.iter().chain(filter).chain(should) {
                    self.query_to_span_query(c, payloads)?;
                }
            }
            Clause::Phrase(p) => self.get_payloads(payloads, &phrase_spans(p)?)?,
            Clause::Term(t) => {
                self.get_payloads(payloads, &SpanNode::term(t.field.clone(), t.term.clone()))?
            }
            Clause::Span(s) => self.get_payloads(payloads, &span_node(s)?)?,
            Clause::Extended(e) => {
                if let ExtendedQuery::Span(s) = e.as_ref() {
                    self.get_payloads(payloads, s)?;
                }
            }
            Clause::DisjunctionMax(d) => {
                for q in &d.disjuncts {
                    self.query_to_span_query(q, payloads)?;
                }
            }
            Clause::MultiPhrase(mpq) => {
                if let Some(sp) = multi_phrase_spans(mpq)? {
                    self.get_payloads(payloads, &sp)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// `getPayloads(payloads, query)`.
    fn get_payloads(&self, payloads: &mut Vec<Vec<u8>>, query: &SpanNode) -> Result<()> {
        let query = query.rewrite(self.searcher)?;
        let mut collector = PayloadSpanCollector::new();
        for seg in self.searcher.segments() {
            let ctx = leaf_context(seg);
            let Some(mut spans) = spans::spans_with(&ctx, &query, true)? else {
                continue;
            };
            while spans.next_doc()? != NO_MORE_DOCS {
                // `Spans.nextDoc()` stops on a matching document only.
                if !spans.matches()? {
                    continue;
                }
                while spans.next_start_position()? != NO_MORE_POSITIONS {
                    collector.reset();
                    spans.collect(&mut collector)?;
                    payloads.append(&mut collector.payloads);
                }
            }
        }
        Ok(())
    }
}

/// The segment's readers without its live docs: spans read every document.
fn leaf_context<'a>(seg: &OpenSegment<'a>) -> LeafContext<'a> {
    LeafContext {
        fields: seg.fields,
        doc_in: seg.doc_in,
        pos_in: seg.pos_in,
        pay_in: seg.pay_in,
        live_docs: None,
        points: seg.points,
        norms: None,
        global: None,
        max_doc: seg.max_doc,
        cache: None,
        reader: seg.reader,
        similarity: seg.similarity,
    }
}

/// A slop as Java's `int`: past `i32::MAX` it saturates (a `u32` slop of
/// that size is no phrase Java can build).
fn slop(slop: u32) -> i32 {
    i32::try_from(slop).unwrap_or(i32::MAX)
}

/// A `PhraseQuery` as `new SpanNearQuery(terms, slop, slop == 0)`.
fn phrase_spans(p: &PhraseQuery) -> Result<SpanNode> {
    let clauses = p
        .terms
        .iter()
        .map(|t| SpanNode::term(p.field.clone(), t.clone()))
        .collect();
    SpanNode::near(clauses, slop(p.slop), p.slop == 0)
}

/// A `MultiPhraseQuery` as a `SpanNearQuery` of a `SpanOrQuery` per
/// distinct position; `None` for one without positions.
fn multi_phrase_spans(mpq: &MultiPhraseQuery) -> Result<Option<SpanNode>> {
    let positions: Vec<i32> = if mpq.positions.is_empty() {
        (0..mpq.term_arrays.len())
            .map(|i| i32::try_from(i).unwrap_or(i32::MAX))
            .collect()
    } else if mpq.positions.len() == mpq.term_arrays.len() {
        mpq.positions.clone()
    } else {
        // Java's builder keeps the two arrays in step; this struct does not.
        return Err(crate::Error::IllegalArgument(format!(
            "{} positions for {} term arrays",
            mpq.positions.len(),
            mpq.term_arrays.len()
        )));
    };
    let Some(&max_position) = positions.iter().max() else {
        return Ok(None);
    };
    let slots = usize::try_from(max_position)
        .ok()
        .and_then(|m| m.checked_add(1))
        .ok_or_else(|| {
            crate::Error::IllegalArgument(format!("negative phrase position: {max_position}"))
        })?;
    let mut disjunct_lists: Vec<Option<Vec<SpanNode>>> = vec![None; slots];
    for (terms, &position) in mpq.term_arrays.iter().zip(&positions) {
        let at = usize::try_from(position).map_err(|_| {
            crate::Error::IllegalArgument(format!("negative phrase position: {position}"))
        })?;
        let disjuncts = disjunct_lists[at].get_or_insert_with(Vec::new);
        disjuncts.extend(
            terms
                .iter()
                .map(|t| SpanNode::term(mpq.field.clone(), t.clone())),
        );
    }
    let mut position_gaps: i32 = 0;
    let mut clauses = Vec::new();
    for disjuncts in disjunct_lists {
        match disjuncts {
            Some(d) => clauses.push(SpanNode::or(d)?),
            None => position_gaps = position_gaps.saturating_add(1),
        }
    }
    let slop = slop(mpq.slop);
    Ok(Some(SpanNode::near(
        clauses,
        slop.wrapping_add(position_gaps),
        mpq.slop == 0,
    )?))
}

/// The core [`SpanQuery`] as the [`SpanNode`] it denotes.
fn span_node(q: &SpanQuery) -> Result<SpanNode> {
    Ok(match q {
        SpanQuery::SpanTerm { field, term } => SpanNode::term(field.clone(), term.clone()),
        SpanQuery::SpanNear {
            clauses,
            slop: s,
            in_order,
        } => SpanNode::near(
            clauses.iter().map(span_node).collect::<Result<_>>()?,
            slop(*s),
            *in_order,
        )?,
        SpanQuery::SpanOr { clauses } => {
            SpanNode::or(clauses.iter().map(span_node).collect::<Result<_>>()?)?
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::directory_reader::DirectoryReader;
    use crate::index_searcher::SegmentNorms;
    use crate::query::{MatchAllDocsQuery, TermQuery};
    use lucene_store::FsDirectory;

    /// `f` over `GenSandboxPayloads`' index.
    fn with_util(f: impl FnOnce(&PayloadSpanUtil<'_, '_, '_>)) {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/data/sandbox_payloads/index");
        let reader = DirectoryReader::open(&FsDirectory::open(dir)).unwrap();
        let opened = reader.open_segments().unwrap();
        let segments = opened.as_open_segments();
        let norms: Vec<SegmentNorms<'_, '_>> = vec![None; segments.len()];
        let searcher = IndexSearcher::new(&segments, &norms).unwrap();
        f(&PayloadSpanUtil::new(&searcher));
    }

    fn mpq(term_arrays: &[&[&str]], positions: Vec<i32>, slop: u32) -> Clause {
        Clause::MultiPhrase(MultiPhraseQuery {
            field: "body".into(),
            term_arrays: term_arrays
                .iter()
                .map(|ts| ts.iter().map(|t| t.as_bytes().to_vec()).collect())
                .collect(),
            slop,
            positions,
        })
    }

    #[test]
    fn a_core_span_query_reads_as_its_span_node() {
        with_util(|u| {
            let core = Clause::Span(SpanQuery::span_near(
                [
                    SpanQuery::span_term("body", "w1"),
                    SpanQuery::span_or([
                        SpanQuery::span_term("body", "w2"),
                        SpanQuery::span_term("body", "w3"),
                    ]),
                ],
                2,
                true,
            ));
            let node = SpanNode::near(
                vec![
                    SpanNode::term("body", "w1"),
                    SpanNode::or(vec![
                        SpanNode::term("body", "w2"),
                        SpanNode::term("body", "w3"),
                    ])
                    .unwrap(),
                ],
                2,
                true,
            )
            .unwrap();
            let want = u
                .get_payloads_for_query(&Clause::Extended(Box::new(ExtendedQuery::Span(node))))
                .unwrap();
            assert!(!want.is_empty());
            assert_eq!(u.get_payloads_for_query(&core).unwrap(), want);
            // Clauses of different fields are refused, as Java's constructor does.
            let mixed = Clause::Span(SpanQuery::span_or([
                SpanQuery::span_term("body", "w1"),
                SpanQuery::span_term("plain", "w1"),
            ]));
            assert!(u.get_payloads_for_query(&mixed).is_err());
        });
    }

    #[test]
    fn multi_phrase_positions_default_to_their_order() {
        with_util(|u| {
            let implicit = mpq(&[&["w1"], &["w2", "w3"]], Vec::new(), 1);
            let explicit = mpq(&[&["w1"], &["w2", "w3"]], vec![0, 1], 1);
            let want = u.get_payloads_for_query(&explicit).unwrap();
            assert!(!want.is_empty());
            assert_eq!(u.get_payloads_for_query(&implicit).unwrap(), want);
            // No positions: nothing to read.
            assert!(u
                .get_payloads_for_query(&mpq(&[], Vec::new(), 0))
                .unwrap()
                .is_empty());
            // A negative position is no phrase Java could build either.
            for bad in [vec![-1, 0], vec![0, -1]] {
                let e = u
                    .get_payloads_for_query(&mpq(&[&["w1"], &["w2"]], bad, 0))
                    .unwrap_err();
                assert!(format!("{e}").contains("negative phrase position"), "{e}");
            }
            // Positions out of step with the term arrays.
            let e = u
                .get_payloads_for_query(&mpq(&[&["w1"], &["w2"]], vec![0], 0))
                .unwrap_err();
            assert!(
                format!("{e}").contains("1 positions for 2 term arrays"),
                "{e}"
            );
        });
    }

    #[test]
    fn other_queries_contribute_nothing() {
        with_util(|u| {
            let all = Clause::from(MatchAllDocsQuery::new(40));
            assert!(u.get_payloads_for_query(&all).unwrap().is_empty());
            let ext = Clause::Extended(Box::new(ExtendedQuery::Span(SpanNode::term(
                "nosuchfield",
                "w1",
            ))));
            assert!(u.get_payloads_for_query(&ext).unwrap().is_empty());
            let term = Clause::Term(TermQuery::new("body", "w1"));
            assert_eq!(u.get_payloads_for_query(&term).unwrap().len(), 15);
            assert!(format!("{u:?}").contains("segments: 2"));
        });
    }

    #[test]
    fn the_collector_resets() {
        let mut c = PayloadSpanCollector::new();
        c.payloads.push(vec![1, 2]);
        assert_eq!(c.payloads(), &[vec![1u8, 2]]);
        SpanCollector::reset(&mut c);
        assert!(c.payloads().is_empty());
        assert_eq!(slop(u32::MAX), i32::MAX);
    }
}
