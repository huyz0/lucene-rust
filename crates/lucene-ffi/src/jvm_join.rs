//! OpenSearch's parent-join queries (`has_child`, `has_parent`): the
//! [`lucene_engine::queries::NODE_JOIN`] nodes of a decoded request made into
//! the global-ordinals joins they rewrite to, against the reader the request
//! searches.
//!
//! `HasChildQueryBuilder.LateParsingQuery.rewrite` searches its from side
//! (`JoinUtil.createJoinQuery` with the join field's global ordinals) before
//! the main query is searched; [`resolve`] does the same here, with the
//! request's similarity and the reader's cached global ordinals
//! ([`lucene_search::directory_reader::DirectoryReader::global_ords`], which
//! OpenSearch's fielddata cache stands for), so every search entry point that
//! takes a query tree finds only queries it can score.

use std::collections::HashMap;

use lucene_search::field_norms::FieldNorms;
use lucene_search::index_searcher::IndexSearcher;
use lucene_search::join::{has_late_joins, resolve_late_joins};
use lucene_search::multi_segment::OpenSegment;
use lucene_search::query::{BooleanQuery, Clause};
use lucene_search::similarities::{PerFieldSimilarity, Similarity};

use crate::error::{set_last_error, FfiStatus};
use crate::registry::JvmReaderHandle;
use lucene_engine::clauses::map_search_error;
use lucene_engine::query_blob::JvmQuery;

/// `query` with its joins resolved against `h`'s reader; the query itself
/// when it has none.
///
/// # Errors
/// What opening the segments or searching a from side reports.
pub(crate) fn resolve(
    h: &JvmReaderHandle,
    query: JvmQuery,
    sim: Option<&PerFieldSimilarity>,
) -> Result<JvmQuery, FfiStatus> {
    let JvmQuery::Boolean(b) = &query else {
        return Ok(query);
    };
    if !has_late_joins(b) {
        return Ok(query);
    }
    Ok(JvmQuery::Boolean(resolve_boolean(h, b, sim)?))
}

/// [`resolve`] of a boolean known to hold a join.
pub(crate) fn resolve_boolean(
    h: &JvmReaderHandle,
    b: &BooleanQuery,
    sim: Option<&PerFieldSimilarity>,
) -> Result<BooleanQuery, FfiStatus> {
    let mut opened = h.reader.open_segments().map_err(|e| {
        set_last_error(format!("opening segment postings: {e}"));
        FfiStatus::Decode
    })?;
    let from_sides = from_sides(b);
    if from_sides
        .iter()
        .any(lucene_engine::query_blob::boolean_uses_points)
    {
        opened.open_points().map_err(|e| {
            set_last_error(format!("opening segment points: {e}"));
            FfiStatus::Decode
        })?;
    }
    let segments: Vec<OpenSegment<'_>> = opened
        .as_open_segments()
        .into_iter()
        .zip(&h.live_docs)
        .map(|(mut s, live)| {
            s.live_docs = live.as_ref();
            s.similarity = sim.map(|p| p as &dyn Similarity);
            s
        })
        .collect();
    let mut fields: Vec<String> = Vec::new();
    for side in &from_sides {
        for f in lucene_engine::clauses::clause_field_names(side) {
            if !fields.iter().any(|g| g == f) {
                fields.push(f.to_string());
            }
        }
    }
    let owned = h
        .reader
        .field_norms_by_field(&fields)
        .map_err(|e| map_search_error(e.into()))?;
    let norms: Vec<Option<&HashMap<String, FieldNorms<'_>>>> =
        owned.iter().map(|m| (!m.is_empty()).then_some(m)).collect();
    let mut searcher = IndexSearcher::new(&segments, &norms).map_err(map_search_error)?;
    if let Some(p) = sim {
        searcher.set_similarity(p);
    }
    let ords = |field: &str| -> lucene_search::Result<Option<std::sync::Arc<_>>> {
        Ok(Some(h.reader.global_ords(field)?.ordinal_map()))
    };
    let resolved = resolve_late_joins(b, &searcher, &ords).map_err(map_search_error)?;
    Ok(resolved.unwrap_or_else(|| b.clone()))
}

/// Every join's from side in `b`, as booleans, for the norms and points
/// they read.
fn from_sides(b: &BooleanQuery) -> Vec<BooleanQuery> {
    use lucene_search::extended_query::ExtendedQuery;
    fn clause(c: &Clause, out: &mut Vec<BooleanQuery>) {
        match c {
            Clause::Extended(e) => {
                if let ExtendedQuery::LateJoin(q) = e.as_ref() {
                    out.push(BooleanQuery {
                        must: vec![(*q.from_query).clone()],
                        ..Default::default()
                    });
                    clause(&q.from_query, out);
                    clause(&q.to_query, out);
                }
                for child in e.children() {
                    clause(child, out);
                }
            }
            Clause::Boolean(b) => boolean(b, out),
            Clause::DisjunctionMax(d) => d.disjuncts.iter().for_each(|c| clause(c, out)),
            Clause::ConstantScore(cs) => clause(&cs.inner, out),
            Clause::Boost(b) => clause(&b.inner, out),
            _ => {}
        }
    }
    fn boolean(b: &BooleanQuery, out: &mut Vec<BooleanQuery>) {
        b.must
            .iter()
            .chain(&b.filter)
            .chain(&b.should)
            .chain(&b.must_not)
            .for_each(|c| clause(c, out));
    }
    let mut out = Vec::new();
    boolean(b, &mut out);
    out
}
