//! The native API a statement is answered with: term lookups, scored top-k
//! searches, nearest-neighbour searches and document reads, all through the
//! same searcher the engine opens for every search. The router calls these
//! directly; the DataFusion table provider calls them from its scans, so the
//! two paths cannot disagree about what a predicate means.

use arrow::array::RecordBatch;
#[cfg(feature = "sql")]
use lucene_search::document::long_point;
#[cfg(feature = "sql")]
use lucene_search::extended_query::{
    ExtendedQuery, IndexOrDocValuesQuery, NumericDocValuesRangeQuery,
};
use lucene_search::knn_search::{knn_top_hits, KnnTarget};
use lucene_search::query::{BooleanQuery, Clause, TermQuery};
#[cfg(feature = "sql")]
use lucene_search::query::{FieldExistsQuery, MatchAllDocsQuery, PrefixQuery};
#[cfg(feature = "sql")]
use lucene_search::top_field::{SortField, SortType};

use crate::columns::Columns;
use crate::error::{Error, Result};
#[cfg(feature = "sql")]
use crate::pred::Pred;
use crate::snapshot::{Searcher, Snapshot};
use crate::spec::ColumnKind;

/// Candidates a nearest-neighbour search collects per segment before keeping
/// the best `k`.
pub const KNN_CANDIDATES: usize = 50;

/// One key of an `ORDER BY`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SortKey {
    /// The column.
    pub column: usize,
    /// `DESC`.
    pub desc: bool,
    /// Whether a document without a value sorts before the others.
    pub nulls_first: bool,
}

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

/// `filters` as one clause every document must match, scoring nothing.
pub(crate) fn all_of_clauses(filters: Vec<Clause>) -> Clause {
    Clause::Boolean(Box::new(BooleanQuery {
        filter: filters,
        ..BooleanQuery::default()
    }))
}

#[cfg(feature = "sql")]
/// The non-scoring predicates as filter clauses. Ranges on one column are
/// intersected into one clause, so a `BETWEEN` is a single range the point
/// tree can answer (and count) without visiting documents.
pub(crate) fn filter_clauses(snapshot: &Snapshot, preds: &[Pred]) -> Vec<Clause> {
    let name = |c: usize| snapshot.spec().columns()[c].name.as_str();
    let mut ranges: Vec<(usize, i64, i64)> = Vec::new();
    let mut clauses = Vec::new();
    for p in preds {
        match p {
            Pred::Term { column, value } => clauses.push(term_clause(name(*column), value)),
            Pred::Any { column, values } => {
                clauses.push(Clause::Boolean(Box::new(any_of(name(*column), values))));
            }
            Pred::Range {
                column,
                lower,
                upper,
            } => match ranges.iter_mut().find(|r| r.0 == *column) {
                Some(r) => {
                    r.1 = r.1.max(*lower);
                    r.2 = r.2.min(*upper);
                }
                None => ranges.push((*column, *lower, *upper)),
            },
            Pred::Present { column } => clauses.push(exists_clause(name(*column))),
            Pred::Missing { column } => clauses.push(missing_clause(snapshot, name(*column))),
            Pred::Prefix { column, prefix } => clauses.push(Clause::Prefix(PrefixQuery::new(
                name(*column),
                prefix.as_bytes().to_vec(),
            ))),
            Pred::Not(inner) => {
                let mut negated = filter_clauses(snapshot, std::slice::from_ref(inner));
                let mut exists = Vec::new();
                if let Some(c) = column_of_pred(inner) {
                    exists.push(exists_clause(name(c)));
                }
                if let Some(clause) = negated.pop() {
                    clauses.push(Clause::Boolean(Box::new(BooleanQuery {
                        filter: exists,
                        must_not: vec![clause],
                        ..BooleanQuery::default()
                    })));
                }
            }
            Pred::Or(branches) => {
                let should = branches
                    .iter()
                    .map(|b| conjunction(filter_clauses(snapshot, b)))
                    .collect();
                clauses.push(Clause::Boolean(Box::new(BooleanQuery {
                    should,
                    minimum_should_match: 1,
                    ..BooleanQuery::default()
                })));
            }
            Pred::Match { .. } | Pred::Knn { .. } => {}
        }
    }
    clauses.extend(
        ranges
            .into_iter()
            .map(|(c, lo, hi)| range_clause(snapshot, c, lo, hi)),
    );
    clauses
}

#[cfg(feature = "sql")]
fn exists_clause(field: &str) -> Clause {
    Clause::Exists(FieldExistsQuery::new(field))
}

#[cfg(feature = "sql")]
/// The documents without a value for `field`.
fn missing_clause(snapshot: &Snapshot, field: &str) -> Clause {
    Clause::Boolean(Box::new(BooleanQuery {
        must: vec![Clause::MatchAllDocs(MatchAllDocsQuery::new(
            snapshot.max_doc(),
        ))],
        must_not: vec![exists_clause(field)],
        ..BooleanQuery::default()
    }))
}

#[cfg(feature = "sql")]
/// The one clause every one of `clauses` must meet.
fn conjunction(mut clauses: Vec<Clause>) -> Clause {
    if clauses.len() == 1 {
        clauses.remove(0)
    } else {
        all_of_clauses(clauses)
    }
}

#[cfg(feature = "sql")]
/// The column a single-column predicate is about.
fn column_of_pred(p: &Pred) -> Option<usize> {
    match p {
        Pred::Term { column, .. }
        | Pred::Any { column, .. }
        | Pred::Range { column, .. }
        | Pred::Prefix { column, .. } => Some(*column),
        _ => None,
    }
}

#[cfg(feature = "sql")]
/// Documents whose long `field` has a value in `lower..=upper`, from the
/// point tree or the doc values, whichever is cheaper for the segment, when
/// the field has both.
pub(crate) fn range_clause(snapshot: &Snapshot, column: usize, lower: i64, upper: i64) -> Clause {
    let field = &snapshot.spec().columns()[column].name;
    let values = NumericDocValuesRangeQuery::new(field.as_str(), lower, upper);
    if snapshot.long_has_points(column) {
        if let Ok(points) = long_point::new_range_query(field, lower, upper) {
            return Clause::from(IndexOrDocValuesQuery::new(
                Clause::from(points),
                Clause::from(values),
            ));
        }
    }
    Clause::Extended(Box::new(ExtendedQuery::NumericDocValuesRange(values)))
}

#[cfg(feature = "sql")]
impl Snapshot {
    /// The query matching every live document that meets all the
    /// non-scoring predicates in `preds`.
    pub(crate) fn filter_query(&self, preds: &[Pred]) -> BooleanQuery {
        let filter = filter_clauses(self, preds);
        BooleanQuery {
            must: if filter.is_empty() {
                vec![Clause::MatchAllDocs(
                    lucene_search::query::MatchAllDocsQuery::new(self.max_doc()),
                )]
            } else {
                Vec::new()
            },
            filter,
            ..BooleanQuery::default()
        }
    }

    /// The documents meeting the non-scoring predicates in `preds`, ordered
    /// by `order` (the document order without keys), from `offset` on, at
    /// most `limit` of them (all without one).
    ///
    /// Returns `None` when a sort column has no doc values to order by.
    ///
    /// # Errors
    /// A sort column is not a long, key or keyword column, or the index is
    /// unreadable.
    pub fn rows(
        &self,
        preds: &[Pred],
        order: &[SortKey],
        limit: Option<usize>,
        offset: usize,
    ) -> Result<Option<Vec<Hit>>> {
        if !order.iter().all(|k| self.has_doc_values(k.column)) {
            return Ok(None);
        }
        let query = self.filter_query(preds);
        let want = limit.map_or(usize::MAX, |l| l.saturating_add(offset));
        let docs: Vec<i32> = if want == 0 {
            Vec::new()
        } else if order.is_empty() && limit.is_none() {
            self.with_searcher(|s| {
                let mut all = CollectAll(Vec::new());
                s.searcher.search_collector(&query, &mut all)?;
                Ok(all.0)
            })?
        } else if order.is_empty() {
            self.with_searcher(|s| {
                let top = s.searcher.search(&query, want)?;
                Ok(top.score_docs.iter().map(|h| h.doc).collect())
            })?
        } else {
            let sorts = order
                .iter()
                .map(|k| self.sort_field(k))
                .collect::<Result<Vec<_>>>()?;
            self.with_searcher(|s| {
                let top = s
                    .searcher
                    .search_sorted(s.leaves, &query, want, &sorts, None)?;
                Ok(top.hits.iter().map(|h| h.doc).collect())
            })?
        };
        Ok(Some(
            docs.into_iter()
                .skip(offset)
                .take(limit.unwrap_or(usize::MAX))
                .map(|d| (d, None))
                .collect(),
        ))
    }

    fn sort_field(&self, key: &SortKey) -> Result<SortField> {
        let column = &self.spec().columns()[key.column];
        // A missing value sorts as the greatest or the least value; which
        // one is the key's `NULLS FIRST`/`LAST` read with its direction.
        let greatest = key.nulls_first == key.desc;
        let mut field = match column.kind {
            ColumnKind::Long => {
                let mut f = SortField::numeric(&column.name, SortType::Long, key.desc);
                f.missing = if greatest { i64::MAX } else { i64::MIN };
                f
            }
            ColumnKind::Key | ColumnKind::Keyword => {
                let mut f = SortField::string(&column.name, key.desc);
                f.missing = i64::from(greatest);
                f
            }
            _ => {
                return Err(Error::unsupported(format!(
                    "`{}` cannot be ordered by",
                    column.name
                )))
            }
        };
        field.reverse = key.desc;
        Ok(field)
    }

    /// How many live documents meet all the non-scoring predicates in
    /// `preds`; a range on a long column with points and a term on its own
    /// are counted from the index without visiting documents.
    ///
    /// # Errors
    /// The index is unreadable.
    pub fn count(&self, preds: &[Pred]) -> Result<u64> {
        let query = self.filter_query(preds);
        self.with_searcher(|s| Ok(s.searcher.count(&query)?))
    }
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
        self.match_with(field, text, k, Vec::new())
    }

    /// [`Self::match_top_k`] among the documents meeting every non-scoring
    /// predicate in `preds`.
    ///
    /// # Errors
    /// As [`Self::match_top_k`].
    #[cfg(feature = "sql")]
    pub fn match_top_k_where(
        &self,
        field: &str,
        text: &str,
        k: usize,
        preds: &[Pred],
    ) -> Result<Vec<Hit>> {
        self.match_with(field, text, k, filter_clauses(self, preds))
    }

    fn match_with(
        &self,
        field: &str,
        text: &str,
        k: usize,
        filters: Vec<Clause>,
    ) -> Result<Vec<Hit>> {
        self.column_of(field, "text", |k| *k == ColumnKind::Text)?;
        let terms = self.spec().text_analyzer(field).terms(text);
        self.with_searcher(|s| match_in(s, field, &terms, k, filters))
    }

    /// The `k` live documents nearest `vector` in the vector column `field`,
    /// best first.
    ///
    /// # Errors
    /// `field` is not a vector column, the vector has the wrong dimension, or
    /// the index is unreadable.
    pub fn knn_top_k(&self, field: &str, vector: &[f32], k: usize) -> Result<Vec<Hit>> {
        self.knn_with(field, vector, k, Vec::new())
    }

    /// [`Self::knn_top_k`] among the documents meeting every non-scoring
    /// predicate in `preds`.
    ///
    /// # Errors
    /// As [`Self::knn_top_k`].
    #[cfg(feature = "sql")]
    pub fn knn_top_k_where(
        &self,
        field: &str,
        vector: &[f32],
        k: usize,
        preds: &[Pred],
    ) -> Result<Vec<Hit>> {
        self.knn_with(field, vector, k, filter_clauses(self, preds))
    }

    fn knn_with(
        &self,
        field: &str,
        vector: &[f32],
        k: usize,
        filters: Vec<Clause>,
    ) -> Result<Vec<Hit>> {
        let c = self.column_of(field, "vector", |k| matches!(k, ColumnKind::Vector { .. }))?;
        check_dimension(self, c, vector)?;
        let filter = (!filters.is_empty()).then(|| all_of_clauses(filters));
        self.with_searcher(|s| knn_in(s, field, vector, k, filter.as_ref()))
    }

    /// The columns `projection` names (indices into the table: declared
    /// columns, then `_score`) for `docs`, as one batch.
    ///
    /// # Errors
    /// A column that cannot be read, or an unreadable index.
    pub fn read(&self, projection: &[usize], docs: &[Hit]) -> Result<RecordBatch> {
        let arrays = Columns::new(self).for_docs(projection, docs)?;
        let schema = self.spec().arrow_schema();
        let schema = std::sync::Arc::new(schema.project(projection).map_err(Error::from)?);
        RecordBatch::try_new(schema, arrays).map_err(Error::from)
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
    filters: Vec<Clause>,
) -> Result<Vec<Hit>> {
    if terms.is_empty() || k == 0 {
        return Ok(Vec::new());
    }
    let mut query = any_of(field, terms);
    query.filter = filters;
    let top = s.searcher.search(&query, k)?;
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

struct CollectAll(Vec<i32>);

impl lucene_search::collector::ScoringCollector for CollectAll {
    fn collect(&mut self, doc_id: i32, _score: f32) {
        self.0.push(doc_id);
    }
}

/// Every live document matching all of `filters` (field, term), by global
/// doc id in ascending order.
pub(crate) fn matching_docs(
    s: &Searcher<'_, '_>,
    filters: &[(String, String)],
) -> Result<Vec<i32>> {
    let q = BooleanQuery {
        filter: filters.iter().map(|(f, t)| term_clause(f, t)).collect(),
        ..BooleanQuery::default()
    };
    let mut c = CollectAll(Vec::new());
    s.searcher.search_collector(&q, &mut c)?;
    Ok(c.0)
}
