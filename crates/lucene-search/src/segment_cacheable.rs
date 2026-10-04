//! `SegmentCacheable.isCacheable(ctx)` for the weights of this port's
//! clauses: whether a query's matches on a segment may be cached against it.
//!
//! What depends only on segment-immutable structures (postings, points,
//! norms) is cacheable; what reads doc values is cacheable only while the
//! field has no doc-values updates (`DocValues.isCacheable`: `dvGen == -1`);
//! a boolean or dis-max of more than 16 clauses never is
//! (`BOOLEAN_REWRITE_TERM_COUNT_THRESHOLD`, so a large boolean is no
//! workaround for an uncacheable large term set). The scorer tree's query
//! cache consults it per segment before looking a clause up
//! (`CachingWrapperWeight.scorerSupplier`'s short-circuit).
//!
//! Values sources answer the same question through
//! [`crate::values_source::DoubleValuesSource::is_cacheable`].

use lucene_codecs::field_infos::DocValuesType;

use crate::directory_reader::SegmentReader;
use crate::query::Clause;

/// `AbstractMultiTermQueryConstantScoreWrapper.BOOLEAN_REWRITE_TERM_COUNT_THRESHOLD`.
pub const BOOLEAN_REWRITE_TERM_COUNT_THRESHOLD: usize = 16;

/// `DocValues.isCacheable(ctx, fields...)`: every named field that exists
/// has no doc-values updates.
pub fn doc_values_cacheable(reader: &SegmentReader, fields: &[&str]) -> bool {
    fields.iter().all(|f| {
        reader
            .field_infos()
            .field_by_name(f)
            .is_none_or(|i| i.doc_values_gen == -1)
    })
}

/// `Weight.isCacheable(ctx)` for `clause` over the segment `reader` reads.
/// Without a reader nothing doc-values-backed can be checked, so such a
/// clause is not cacheable.
pub fn is_cacheable(clause: &Clause, reader: Option<&SegmentReader>) -> bool {
    match clause {
        Clause::Boolean(b) => {
            let n = b.must.len() + b.filter.len() + b.should.len() + b.must_not.len();
            n <= BOOLEAN_REWRITE_TERM_COUNT_THRESHOLD
                && b.must
                    .iter()
                    .chain(&b.filter)
                    .chain(&b.should)
                    .chain(&b.must_not)
                    .all(|c| is_cacheable(c, reader))
        }
        Clause::DisjunctionMax(d) => {
            d.disjuncts.len() <= BOOLEAN_REWRITE_TERM_COUNT_THRESHOLD
                && d.disjuncts.iter().all(|c| is_cacheable(c, reader))
        }
        Clause::ConstantScore(c) => is_cacheable(&c.inner, reader),
        // The block joins: their weights delegate to the wrapped query's
        // (`FilterWeight`); `ParentChildrenBlockJoinQuery` is never cacheable.
        Clause::Extended(e) => match e.as_ref() {
            crate::extended_query::ExtendedQuery::ParentChildrenBlockJoin(_) => false,
            // `GlobalOrdinals*Query.W.isCacheable`: never.
            crate::extended_query::ExtendedQuery::GlobalOrdinals(_)
            | crate::extended_query::ExtendedQuery::GlobalOrdinalsWithScore(_) => false,
            // `FunctionWeight`/`FunctionRangeWeight.isCacheable`: never.
            crate::extended_query::ExtendedQuery::Function(_)
            | crate::extended_query::ExtendedQuery::FunctionRange(_) => false,
            // `FunctionMatchQuery`: its source's; `FunctionScoreWeight`: the
            // wrapped query's and its source's.
            crate::extended_query::ExtendedQuery::FunctionMatch(q) => {
                crate::function::source_cacheable(q.source.as_ref(), reader)
            }
            crate::extended_query::ExtendedQuery::FunctionScore(q) => {
                is_cacheable(&q.in_query, reader)
                    && crate::function::source_cacheable(q.source.as_ref(), reader)
            }
            crate::extended_query::ExtendedQuery::ToParentBlockJoin(_)
            | crate::extended_query::ExtendedQuery::ToChildBlockJoin(_)
            | crate::extended_query::ExtendedQuery::ParentsChildrenBlockJoin(_) => {
                e.children().into_iter().all(|c| is_cacheable(c, reader))
            }
            _ => true,
        },
        Clause::Boost(b) => is_cacheable(&b.inner, reader),
        // `FieldExistsQuery`: through doc values only when the field has
        // them; norms and vectors are segment-immutable.
        Clause::Exists(q) => match reader {
            Some(r) => match r.field_infos().field_by_name(&q.field) {
                Some(i) if i.doc_values_type != DocValuesType::None => {
                    doc_values_cacheable(r, &[&q.field])
                }
                _ => true,
            },
            None => false,
        },
        // Terms, phrases, multi-term queries, points, match-all/none and
        // spans read only postings and points.
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::{BooleanQuery, DisjunctionMaxQuery, FieldExistsQuery, TermQuery};

    #[test]
    fn large_compounds_are_never_cacheable() {
        let mut b = BooleanQuery::new();
        let terms: Vec<Clause> = (0..17)
            .map(|i| Clause::Term(TermQuery::new("f", format!("t{i}").into_bytes())))
            .collect();
        b.should = terms.clone();
        assert!(!is_cacheable(&Clause::Boolean(Box::new(b.clone())), None));
        b.should.truncate(16);
        assert!(is_cacheable(&Clause::Boolean(Box::new(b)), None));
        let d = DisjunctionMaxQuery::new(terms, 0.0);
        assert!(!is_cacheable(&Clause::DisjunctionMax(Box::new(d)), None));
        assert!(!is_cacheable(
            &Clause::Exists(FieldExistsQuery::new("f")),
            None
        ));
    }
}
