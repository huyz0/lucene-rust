//! The native API a statement is answered with: term lookups, scored top-k
//! searches, nearest-neighbour searches and document reads, all through the
//! same searcher the engine opens for every search. The router calls these
//! directly; the DataFusion table provider calls them from its scans, so the
//! two paths cannot disagree about what a predicate means.

use datafusion::arrow::array::RecordBatch;
use lucene_search::knn_search::{knn_top_hits, KnnTarget};
use lucene_search::query::{BooleanQuery, Clause, TermQuery};

use crate::columns::Columns;
use crate::error::{Error, Result};
use crate::snapshot::{Searcher, Snapshot};
use crate::spec::ColumnKind;

/// Candidates a nearest-neighbour search collects per segment before keeping
/// the best `k`.
pub const KNN_CANDIDATES: usize = 50;

/// A matched document: its global doc id and, for a scored search, its score.
pub type Hit = (i32, Option<f32>);

pub(crate) fn term_clause(field: &str, term: &str) -> Clause {
    Clause::Term(TermQuery::new(field, term.as_bytes().to_vec()))
}

/// A query matching documents containing any of `terms` in `field`.
pub(crate) fn any_of(field: &str, terms: &[String]) -> BooleanQuery {
    BooleanQuery {
        should: terms.iter().map(|t| term_clause(field, t)).collect(),
        minimum_should_match: 1,
        ..BooleanQuery::default()
    }
}

/// `filters` (field, term) as one clause every document must match, scoring
/// nothing.
pub(crate) fn all_of(filters: &[(String, String)]) -> Clause {
    Clause::Boolean(Box::new(BooleanQuery {
        filter: filters.iter().map(|(f, t)| term_clause(f, t)).collect(),
        ..BooleanQuery::default()
    }))
}

impl Snapshot {
    fn column_of(&self, name: &str, wanted: &str, ok: fn(&ColumnKind) -> bool) -> Result<usize> {
        let spec = self.spec();
        let i = spec
            .columns()
            .iter()
            .position(|c| c.name == name && ok(&c.kind))
            .ok_or_else(|| {
                Error::unsupported(format!(
                    "`{name}` is not a {wanted} column of `{}`",
                    spec.name()
                ))
            })?;
        Ok(i)
    }

    /// The live document whose `key` column equals `value`, if any.
    ///
    /// # Errors
    /// `key` is not a term-indexed column, or the index is unreadable.
    pub fn lookup(&self, key: &str, value: &str) -> Result<Option<i32>> {
        self.column_of(key, "key or keyword", ColumnKind::is_term_indexed)?;
        self.with_searcher(|s| lookup_in(s, key, value))
    }

    /// The `k` best live documents for `text` (analysed as the table's
    /// analyzer says) in the text column `field`, best first.
    ///
    /// # Errors
    /// `field` is not a text column, or the index is unreadable.
    pub fn match_top_k(&self, field: &str, text: &str, k: usize) -> Result<Vec<Hit>> {
        self.column_of(field, "text", |k| *k == ColumnKind::Text)?;
        let terms = self.spec().text_analyzer().terms(text);
        self.with_searcher(|s| match_in(s, field, &terms, k))
    }

    /// The `k` live documents nearest `vector` in the vector column `field`,
    /// best first.
    ///
    /// # Errors
    /// `field` is not a vector column, the vector has the wrong dimension, or
    /// the index is unreadable.
    pub fn knn_top_k(&self, field: &str, vector: &[f32], k: usize) -> Result<Vec<Hit>> {
        let c = self.column_of(field, "vector", |k| matches!(k, ColumnKind::Vector { .. }))?;
        check_dimension(self, c, vector)?;
        self.with_searcher(|s| knn_in(s, field, vector, k, None))
    }

    /// The columns `projection` names (indices into the table: declared
    /// columns, then `_score`) for `docs`, as one batch.
    ///
    /// # Errors
    /// A column that cannot be read, or an unreadable index.
    pub fn read(&self, projection: &[usize], docs: &[Hit]) -> Result<RecordBatch> {
        let arrays = Columns::new(self).for_docs(projection, docs)?;
        let schema = self.spec().arrow_schema();
        let schema = std::sync::Arc::new(schema.project(projection).map_err(Error::from_arrow)?);
        RecordBatch::try_new(schema, arrays).map_err(Error::from_arrow)
    }
}

pub(crate) fn check_dimension(snapshot: &Snapshot, column: usize, vector: &[f32]) -> Result<()> {
    if let ColumnKind::Vector { dim } = snapshot.spec().columns()[column].kind {
        if vector.len() != dim {
            return Err(Error::unsupported(format!(
                "the query vector has {} dimensions, `{}` has {dim}",
                vector.len(),
                snapshot.spec().columns()[column].name
            )));
        }
    }
    Ok(())
}

pub(crate) fn lookup_in(s: &Searcher<'_, '_>, field: &str, value: &str) -> Result<Option<i32>> {
    let query = BooleanQuery {
        must: vec![term_clause(field, value)],
        ..BooleanQuery::default()
    };
    let top = s.searcher.search(&query, 1)?;
    Ok(top.score_docs.first().map(|h| h.doc))
}

pub(crate) fn match_in(
    s: &Searcher<'_, '_>,
    field: &str,
    terms: &[String],
    k: usize,
) -> Result<Vec<Hit>> {
    if terms.is_empty() || k == 0 {
        return Ok(Vec::new());
    }
    let top = s.searcher.search(&any_of(field, terms), k)?;
    Ok(top
        .score_docs
        .iter()
        .map(|h| (h.doc, Some(h.score)))
        .collect())
}

pub(crate) fn knn_in(
    s: &Searcher<'_, '_>,
    field: &str,
    vector: &[f32],
    k: usize,
    filter: Option<&Clause>,
) -> Result<Vec<Hit>> {
    if k == 0 {
        return Ok(Vec::new());
    }
    let hits = knn_top_hits(
        s.segments,
        field,
        &KnnTarget::Float(vector.to_vec()),
        k,
        KNN_CANDIDATES.max(k),
        filter,
    )?;
    Ok(hits.iter().map(|h| (h.doc_id, Some(h.score))).collect())
}
