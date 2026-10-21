//! `PhraseHelper`: the position-sensitive part of a query (its phrases and
//! span queries, as `WeightedSpanTermExtractor` turns them into
//! `SpanQuery`s) and the terms that need no positions, and the offsets of
//! the terms of every span those queries match in a document.

use std::collections::{BTreeMap, BTreeSet, HashSet};

use super::super::extractor::{collect_span_query_fields, must_rewrite};
use super::super::Result;
use super::passage::{OfPostings, OffsetsEnum};
use super::strategies::FieldMatcher;
use crate::exec::LeafContext;
use crate::extended_query::ExtendedQuery;
use crate::index_searcher::IndexSearcher;
use crate::multi_segment::OpenSegment;
use crate::query::{Clause, MultiPhraseQuery, PhraseQuery};
use crate::query_visitor::{self, QueryVisitor, Term};
use crate::spans::{spans_with, SpanCollector, SpanNode, Spans, TermSpans, NO_MORE_POSITIONS};

/// `PhraseHelper`. [`Default`] is `PhraseHelper.NONE`.
#[derive(Clone, Default)]
pub struct PhraseHelper {
    /// `fieldName`.
    field: String,
    /// `fieldMatcher`; `None` is `NONE`'s, which accepts nothing.
    field_matcher: Option<FieldMatcher>,
    /// `positionInsensitiveTerms` (the `TermQuery`-like terms).
    insensitive: BTreeSet<Vec<u8>>,
    /// `spanQueries`, a set: each distinct query once.
    span_queries: Vec<SpanNode>,
    /// `willRewrite`.
    will_rewrite: bool,
}

impl PhraseHelper {
    /// `new PhraseHelper(query, field, fieldMatcher, ..., ignoreQueriesNeedingRewrite)`
    /// with the default hooks: the query read as `WeightedSpanTermExtractor`
    /// reads it, multi-term queries expanded (`setExpandMultiTermQuery(true)`).
    pub fn new(
        query: &Clause,
        field: &str,
        field_matcher: FieldMatcher,
        ignore_queries_needing_rewrite: bool,
    ) -> Self {
        let mut helper = PhraseHelper {
            field: field.to_string(),
            field_matcher: Some(field_matcher),
            ..PhraseHelper::default()
        };
        helper.extract(query, ignore_queries_needing_rewrite);
        helper
    }

    fn accepts(&self, field: &str) -> bool {
        self.field_matcher.as_ref().is_some_and(|m| m(field))
    }

    /// `WeightedSpanTermExtractor.extract`, every query it does not know
    /// "unsupported" (`isQueryUnsupported` is always true here).
    fn extract(&mut self, query: &Clause, ignore: bool) {
        match query {
            Clause::Boost(b) => self.extract(&b.inner, ignore),
            Clause::Boolean(b) => {
                for c in b.must.iter().chain(&b.filter).chain(&b.should) {
                    self.extract(c, ignore);
                }
            }
            Clause::Phrase(p) => self.extract_phrase(p, ignore),
            Clause::Term(_) => self.extract_weighted_terms(query),
            Clause::Span(s) => self.extract_weighted_span_terms(&SpanNode::from(s), ignore),
            Clause::ConstantScore(c) => self.extract(&c.inner, ignore),
            Clause::DisjunctionMax(d) => {
                for c in &d.disjuncts {
                    self.extract(c, ignore);
                }
            }
            Clause::MultiPhrase(m) => self.extract_multi_phrase(m, ignore),
            Clause::Extended(e) => match e.as_ref() {
                ExtendedQuery::Synonym(_) | ExtendedQuery::CommonTerms(_) => {
                    self.extract_weighted_terms(query);
                }
                ExtendedQuery::IndexOrDocValues(q) => self.extract(&q.index_query, ignore),
                ExtendedQuery::FunctionScore(q) => self.extract(&q.in_query, ignore),
                ExtendedQuery::Span(node) => self.extract_weighted_span_terms(node, ignore),
                _ => {}
            },
            // `MatchAllDocsQuery`, and what is unsupported
            _ => {}
        }
    }

    /// A `PhraseQuery` as `WeightedSpanTermExtractor` turns it into a span
    /// query: one term alone, else a `SpanNearQuery` whose slop takes in
    /// the position gaps, in order when the slop is 0.
    fn extract_phrase(&mut self, p: &PhraseQuery, ignore: bool) {
        let clauses: Vec<SpanNode> = p
            .terms
            .iter()
            .map(|t| SpanNode::term(p.field.clone(), t.clone()))
            .collect();
        if clauses.len() == 1 {
            return self.extract_weighted_span_terms(&clauses[0], ignore);
        }
        if clauses.is_empty() {
            // a phrase of no terms matches nothing
            return;
        }
        let positions = p.positions();
        let n = i32::try_from(positions.len()).unwrap_or(i32::MAX);
        let position_gaps = (positions[positions.len() - 1] - positions[0] - n + 1).max(0);
        let slop = i32::try_from(p.slop).unwrap_or(i32::MAX);
        let node = SpanNode::Near {
            clauses,
            slop: slop.saturating_add(position_gaps),
            in_order: slop == 0,
        };
        self.extract_weighted_span_terms(&node, ignore);
    }

    /// A `MultiPhraseQuery` as a `SpanNearQuery` of a `SpanOrQuery` per
    /// position.
    fn extract_multi_phrase(&mut self, m: &MultiPhraseQuery, ignore: bool) {
        let positions = m.positions();
        let Some(&max_position) = positions.iter().max() else {
            return;
        };
        let len = usize::try_from(max_position).unwrap_or(0) + 1;
        let mut disjunct_lists: Vec<Option<Vec<SpanNode>>> = vec![None; len];
        for (array, &at) in m.term_arrays.iter().zip(&positions) {
            let list =
                disjunct_lists[usize::try_from(at).unwrap_or(0)].get_or_insert_with(Vec::new);
            for t in array {
                list.push(SpanNode::term(m.field.clone(), t.clone()));
            }
        }
        let mut position_gaps = 0i32;
        let mut clauses = Vec::new();
        for list in disjunct_lists {
            match list {
                Some(l) => clauses.push(SpanNode::Or { clauses: l }),
                None => position_gaps += 1,
            }
        }
        if clauses.len() == 1 {
            return self.extract_weighted_span_terms(&clauses[0], ignore);
        }
        let slop = i32::try_from(m.slop).unwrap_or(i32::MAX);
        let node = SpanNode::Near {
            clauses,
            slop: slop.saturating_add(position_gaps),
            in_order: slop == 0,
        };
        self.extract_weighted_span_terms(&node, ignore);
    }

    /// `extractWeightedTerms`: the terms of a position-insensitive query on
    /// a matching field.
    fn extract_weighted_terms(&mut self, query: &Clause) {
        struct V<'h> {
            helper: &'h PhraseHelper,
            out: Vec<Vec<u8>>,
        }
        impl QueryVisitor for V<'_> {
            fn accept_field(&self, field: &str) -> bool {
                self.helper.accepts(field)
            }
            fn consume_terms(&mut self, _query: &Clause, terms: &[Term]) {
                self.out.extend(terms.iter().map(|t| t.bytes.clone()));
            }
        }
        let mut v = V {
            helper: self,
            out: Vec::new(),
        };
        query_visitor::visit(query, &mut v);
        let out = v.out;
        self.insensitive.extend(out);
    }

    /// `extractWeightedSpanTerms`: a span query all of whose fields match,
    /// unless it needs rewriting and such queries are ignored.
    fn extract_weighted_span_terms(&mut self, span_query: &SpanNode, ignore: bool) {
        let mut fields = HashSet::new();
        collect_span_query_fields(span_query, &mut fields);
        if fields.iter().any(|f| !self.accepts(f)) {
            return;
        }
        let must = must_rewrite(span_query, true);
        if ignore && must {
            return;
        }
        self.will_rewrite |= must;
        if !self.span_queries.contains(span_query) {
            self.span_queries.push(span_query.clone());
        }
    }

    /// `getSpanQueries()`.
    pub fn span_queries(&self) -> &[SpanNode] {
        &self.span_queries
    }

    /// `hasPositionSensitivity()`.
    pub fn has_position_sensitivity(&self) -> bool {
        !self.span_queries.is_empty()
    }

    /// `willRewrite()`.
    pub fn will_rewrite(&self) -> bool {
        self.will_rewrite
    }

    /// `getAllPositionInsensitiveTerms()`, sorted.
    pub fn all_position_insensitive_terms(&self) -> Vec<Vec<u8>> {
        self.insensitive.iter().cloned().collect()
    }

    /// `createOffsetsEnumsForSpans(leafReader, docId, results)` over one
    /// segment: every field read as the highlighted one
    /// (`SingleFieldWithOffsetsFilterLeafReader`), each span query
    /// rewritten against the segment and positioned on `doc` (leaf-local),
    /// and the offsets of the terms of each of its spans collected.
    ///
    /// # Errors
    /// What reading the segment reports; a span term of a field indexed
    /// without positions.
    pub(crate) fn offsets_enums_for_spans(
        &self,
        segment: &OpenSegment<'_>,
        doc: i32,
        results: &mut Vec<Box<dyn OffsetsEnum>>,
    ) -> Result<()> {
        let norms = [None];
        let searcher = IndexSearcher::new(std::slice::from_ref(segment), &norms)?;
        let ctx = leaf_context(segment);
        let (mut accepted, mut rejected) = (HashSet::new(), HashSet::new());
        let nodes: Vec<SpanNode> = self
            .span_queries
            .iter()
            .map(|q| self.single_field(q, &mut accepted, &mut rejected))
            .collect();
        rejected.retain(|t| !accepted.contains(t));
        let mut collector = OffsetSpanCollector {
            helper: self,
            rejected_only: rejected,
            by_term: BTreeMap::new(),
        };
        for mut node in nodes {
            if node.needs_rewrite() {
                node = node.rewrite(&searcher)?;
            }
            let Some(mut spans) = spans_with(&ctx, &node, false)? else {
                continue;
            };
            // preposition, and do nothing where the document has no span
            if spans.advance(doc)? != doc || !spans.matches()? {
                continue;
            }
            while spans.next_start_position()? != NO_MORE_POSITIONS {
                spans.collect(&mut collector)?;
            }
        }
        for (term, offsets) in collector.by_term {
            let freq = i32::try_from(offsets.len()).unwrap_or(i32::MAX);
            results.push(Box::new(OfPostings::new(term, freq, offsets)));
        }
        Ok(())
    }

    /// The query as `SingleFieldWithOffsetsFilterLeafReader` reads it: every
    /// term's field is the highlighted one. The terms whose own field the
    /// matcher accepts go to `accepted`, the others to `rejected` (the
    /// collector skips the latter, as `OffsetSpanCollector` tests
    /// `term.field()`).
    fn single_field(
        &self,
        node: &SpanNode,
        accepted: &mut HashSet<Vec<u8>>,
        rejected: &mut HashSet<Vec<u8>>,
    ) -> SpanNode {
        let mut out = node.clone();
        self.retarget_node(&mut out, accepted, rejected);
        out
    }

    fn retarget_node(
        &self,
        node: &mut SpanNode,
        accepted: &mut HashSet<Vec<u8>>,
        rejected: &mut HashSet<Vec<u8>>,
    ) {
        match node {
            SpanNode::Term { field, term } => {
                if self.accepts(field) {
                    accepted.insert(term.clone());
                } else {
                    rejected.insert(term.clone());
                }
                field.clone_from(&self.field);
            }
            SpanNode::MultiTerm(m) => m.source.field_mut().clone_from(&self.field),
            SpanNode::Near { clauses, .. } | SpanNode::Or { clauses } => {
                for c in clauses {
                    self.retarget_node(c, accepted, rejected);
                }
            }
            SpanNode::First { inner, .. } | SpanNode::PositionRange { inner, .. } => {
                self.retarget_node(inner, accepted, rejected);
            }
            SpanNode::FieldMasking { inner, field } => {
                self.retarget_node(inner, accepted, rejected);
                field.clone_from(&self.field);
            }
            SpanNode::Not {
                include, exclude, ..
            } => {
                self.retarget_node(include, accepted, rejected);
                self.retarget_node(exclude, accepted, rejected);
            }
            SpanNode::Containing { big, little } | SpanNode::Within { big, little } => {
                self.retarget_node(big, accepted, rejected);
                self.retarget_node(little, accepted, rejected);
            }
            SpanNode::PayloadCheck(p) => self.retarget_node(&mut p.inner, accepted, rejected),
            SpanNode::PayloadScore(p) => self.retarget_node(&mut p.inner, accepted, rejected),
        }
    }
}

/// A segment as the spans read it: no live docs (`SpanWeight` positions
/// on the document asked for), no scoring.
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
        similarity: None,
    }
}

/// `OffsetSpanCollector` with `SpanCollectedOffsetsEnum`: per term, its
/// distinct offset pairs, sorted.
struct OffsetSpanCollector<'h> {
    helper: &'h PhraseHelper,
    /// Terms the query holds only on fields the matcher rejects.
    rejected_only: HashSet<Vec<u8>>,
    by_term: BTreeMap<Vec<u8>, Vec<(i32, i32)>>,
}

impl SpanCollector for OffsetSpanCollector<'_> {
    fn collect_leaf(&mut self, leaf: &mut TermSpans<'_>, _position: i32) -> crate::Result<()> {
        let term = &leaf.term.1;
        if self.rejected_only.contains(term) {
            return Ok(());
        }
        // a position-insensitive term is handled outside of the helper
        if !self.by_term.contains_key(term) && self.helper.insensitive.contains(term) {
            return Ok(());
        }
        let term = term.clone();
        let Some(occ) = leaf.occurrence()? else {
            return Ok(());
        };
        let pair = (occ.start_offset, occ.end_offset);
        let offsets = self.by_term.entry(term).or_default();
        if let Err(at) = offsets.binary_search(&pair) {
            offsets.insert(at, pair);
        }
        Ok(())
    }

    fn reset(&mut self) {}
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::extended_query::{
        IndexOrDocValuesQuery, MultiTermQuery, MultiTermSource, RewriteMethod,
    };
    use crate::function::FunctionScoreQuery;
    use crate::query::{
        BoostQuery, ConstantScoreQuery, DisjunctionMaxQuery, MatchAllDocsQuery, PrefixQuery,
        TermQuery,
    };

    fn helper(q: impl Into<Clause>) -> PhraseHelper {
        let only_f: FieldMatcher = Arc::new(|f| f == "f");
        PhraseHelper::new(&q.into(), "f", only_f, false)
    }

    fn term(t: &str) -> Clause {
        TermQuery::new("f", t).into()
    }

    #[test]
    fn wrappers_are_read_through_and_unknown_queries_skipped() {
        let ab = || PhraseQuery::new("f", ["a", "b"]);
        let h = helper(IndexOrDocValuesQuery::new(ab(), MatchAllDocsQuery::new(1)));
        assert_eq!(h.span_queries().len(), 1);
        let h = helper(FunctionScoreQuery::new(
            ab(),
            crate::values_source::constant(1.0),
        ));
        assert_eq!(h.span_queries().len(), 1);
        let h = helper(DisjunctionMaxQuery::new(
            [
                Clause::from(BoostQuery::new(term("x"), 2.0)),
                Clause::from(ConstantScoreQuery::new(term("y"), 1.0)),
            ],
            0.0,
        ));
        assert!(!h.has_position_sensitivity());
        assert_eq!(
            h.all_position_insensitive_terms(),
            [b"x".to_vec(), b"y".to_vec()]
        );
        // a multi-term query outside a span query, and a match-all: nothing
        let prefix = MultiTermQuery::new(
            MultiTermSource::Prefix(PrefixQuery::new("f", "q")),
            RewriteMethod::default(),
        );
        assert!(!helper(prefix).has_position_sensitivity());
        assert!(!helper(MatchAllDocsQuery::new(1)).has_position_sensitivity());
        // a span query on another field than the matcher accepts is left out
        let other = SpanNode::near(
            vec![SpanNode::term("g", "a"), SpanNode::term("g", "b")],
            0,
            true,
        )
        .unwrap();
        assert!(!helper(other).has_position_sensitivity());
    }

    #[test]
    fn degenerate_phrases_and_multi_phrases() {
        assert!(!helper(PhraseQuery::new("f", Vec::<Vec<u8>>::new())).has_position_sensitivity());
        assert_eq!(
            helper(PhraseQuery::new("f", ["a"])).span_queries(),
            [SpanNode::term("f", "a")]
        );
        assert!(
            !helper(MultiPhraseQuery::new("f", Vec::<Vec<Vec<u8>>>::new()))
                .has_position_sensitivity()
        );
        // one position: its disjunction alone
        let one = helper(MultiPhraseQuery::new(
            "f",
            [vec![b"a".to_vec(), b"b".to_vec()]],
        ));
        assert!(matches!(one.span_queries(), [SpanNode::Or { clauses }] if clauses.len() == 2));
        // a gap in the positions widens the slop
        let gapped =
            MultiPhraseQuery::with_positions("f", [(vec!["a"], 0), (vec!["b"], 2)]).unwrap();
        assert!(matches!(
            helper(gapped).span_queries(),
            [SpanNode::Near {
                slop: 1,
                in_order: true,
                ..
            }]
        ));
    }

    #[test]
    fn every_span_shape_is_read_from_the_highlighted_field() {
        let h = helper(term("x"));
        let masked = SpanNode::field_masking(SpanNode::term("g", "r"), "f");
        let node = SpanNode::not(
            SpanNode::first(
                SpanNode::containing(
                    SpanNode::near(vec![SpanNode::term("f", "a"), masked], 0, true).unwrap(),
                    SpanNode::position_range(SpanNode::term("f", "b"), 0, 3),
                )
                .unwrap(),
                5,
            ),
            SpanNode::multi_term(MultiTermQuery::new(
                MultiTermSource::Prefix(PrefixQuery::new("f", "q")),
                RewriteMethod::default(),
            )),
            0,
            0,
        )
        .unwrap();
        let (mut accepted, mut rejected) = (HashSet::new(), HashSet::new());
        let out = h.single_field(&node, &mut accepted, &mut rejected);
        let mut fields = HashSet::new();
        collect_span_query_fields(&out, &mut fields);
        assert_eq!(fields, HashSet::from(["f".to_string()]));
        assert!(!format!("{out:?}").contains("\"g\""));
        assert_eq!(accepted, HashSet::from([b"a".to_vec(), b"b".to_vec()]));
        assert_eq!(rejected, HashSet::from([b"r".to_vec()]));
    }
}
