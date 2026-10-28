//! `org.apache.lucene.monitor.TermFilteredPresearcher`,
//! `MultipassTermFilteredPresearcher` and `Presearcher`: the term-filter
//! index that narrows a document batch to the queries that could match it.
//!
//! A registered query is indexed as the set of terms its [`QueryTree`]
//! names, per field. A batch is matched by building one `TermInSetQuery` per
//! field over the batch's terms, ORed with the any-token marker, which every
//! query with no narrowing term carries. Java runs these queries against a
//! Lucene index of the registered queries; this port returns the two halves
//! as values, and T13.5c's query index runs them. The rules -- which terms
//! are indexed, which are filtered out, the metadata-field filter and the
//! multipass expansion -- are Java's.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::monitor::analyzer::QueryAnalyzer;
use crate::monitor::handlers::CustomQueryHandler;
use crate::monitor::tree::{QueryTree, TermWeightor, ANYTOKEN, ANYTOKEN_FIELD};
use crate::query::{BooleanQuery, Clause, MatchAllDocsQuery, TermInSetQuery, TermQuery};

/// The terms of one query, by field, each once and in first-seen order: the
/// `BytesRefHash` a `Document` field is built from.
pub type FieldTerms = BTreeMap<String, Vec<Vec<u8>>>;

/// `TermFilteredPresearcher.collectTerms`: the tree's terms per field, each
/// field's terms distinct.
fn collect_terms(tree: &QueryTree) -> FieldTerms {
    let mut terms = Vec::new();
    tree.collect_terms(&mut terms);
    let mut out: FieldTerms = BTreeMap::new();
    for t in terms {
        out.entry(t.field).or_default().push(t.bytes);
    }
    // `BytesRefHashIterator` walks each field's hash in sorted term order,
    // each term once
    for v in out.values_mut() {
        v.sort_unstable();
        v.dedup();
    }
    out
}

/// The per-field, per-term view of a batch: `(field, term)` pairs the
/// presearcher is asked about.
pub trait TermAcceptor {
    fn accept(&self, field: &str, term: &[u8]) -> bool;
}

impl<F: Fn(&str, &[u8]) -> bool> TermAcceptor for F {
    fn accept(&self, field: &str, term: &[u8]) -> bool {
        self(field, term)
    }
}

/// `TermFilteredPresearcher`.
#[derive(Clone)]
pub struct TermFilteredPresearcher {
    extractor: QueryAnalyzer,
    /// `queryHandlers`: each wraps a field's batch terms (`wrapTermStream`).
    handlers: Vec<Arc<dyn CustomQueryHandler>>,
    weightor: TermWeightor,
    filter_fields: Vec<String>,
    /// `MultipassTermFilteredPresearcher`'s passes; `1` is the plain one.
    passes: usize,
    /// `minWeight`: below it a pass stops narrowing the tree.
    min_weight: f64,
}

impl std::fmt::Debug for TermFilteredPresearcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TermFilteredPresearcher")
            .field("filter_fields", &self.filter_fields)
            .field("passes", &self.passes)
            .finish()
    }
}

impl Default for TermFilteredPresearcher {
    /// `new TermFilteredPresearcher()`: `TermWeightor.DEFAULT`, no filter
    /// fields.
    fn default() -> Self {
        Self::new(TermWeightor::default(), Vec::new())
    }
}

impl TermFilteredPresearcher {
    /// `new TermFilteredPresearcher(weightor, customQueryHandlers, filterFields)`.
    /// The custom handlers are not ported (see the ledger), so there are none.
    pub fn new(weightor: TermWeightor, filter_fields: Vec<String>) -> Self {
        Self::with_handlers(weightor, Vec::new(), filter_fields)
    }

    /// `new TermFilteredPresearcher(weightor, customQueryHandlers,
    /// filterFields)`.
    pub fn with_handlers(
        weightor: TermWeightor,
        handlers: Vec<Arc<dyn CustomQueryHandler>>,
        filter_fields: Vec<String>,
    ) -> Self {
        TermFilteredPresearcher {
            extractor: QueryAnalyzer::new(handlers.clone()),
            handlers,
            weightor,
            filter_fields,
            passes: 1,
            min_weight: 0.0,
        }
    }

    /// `new MultipassTermFilteredPresearcher(passes, minWeight, weightor,
    /// customQueryHandlers, filterFields)` without handlers.
    pub fn multipass(
        passes: usize,
        min_weight: f64,
        weightor: TermWeightor,
        filter_fields: Vec<String>,
    ) -> Self {
        Self::multipass_with_handlers(passes, min_weight, weightor, Vec::new(), filter_fields)
    }

    /// `new MultipassTermFilteredPresearcher(passes, minWeight, weightor,
    /// customQueryHandlers, filterFields)`.
    pub fn multipass_with_handlers(
        passes: usize,
        min_weight: f64,
        weightor: TermWeightor,
        handlers: Vec<Arc<dyn CustomQueryHandler>>,
        filter_fields: Vec<String>,
    ) -> Self {
        TermFilteredPresearcher {
            passes,
            min_weight,
            ..Self::with_handlers(weightor, handlers, filter_fields)
        }
    }

    /// The fields whose batch values filter the queries (`filterFields`).
    pub fn filter_fields(&self) -> &[String] {
        &self.filter_fields
    }

    /// `indexQuery(query, metadata)`: the query's terms, per field, as the
    /// document a query index stores. Each filter field the metadata names
    /// carries its value as a term too. A multipass presearcher indexes each
    /// pass's terms under the field's suffixed name `field_<pass>`, and the
    /// plain name for the term-stream pass.
    pub fn index_query(
        &self,
        query: &Clause,
        metadata: &BTreeMap<String, String>,
    ) -> Result<IndexedQuery, String> {
        let mut tree = self.extractor.build_tree(query, &self.weightor)?;
        let mut fields = Vec::new();
        for pass in 0..self.passes {
            for (field, terms) in collect_terms(&tree) {
                fields.push((field_for_pass(&field, pass, self.passes), terms.clone()));
                if self.passes > 1 {
                    fields.push((field.clone(), terms));
                }
            }
            if self.passes > 1 {
                let _ = tree.advance_phase(self.min_weight);
            }
        }
        let mut metadata_fields = Vec::new();
        for field in &self.filter_fields {
            if let Some(value) = metadata.get(field) {
                metadata_fields.push((field.clone(), value.clone()));
            }
        }
        Ok(IndexedQuery {
            fields,
            metadata: metadata_fields,
        })
    }

    /// `buildQuery(reader, termAcceptor)`: the batch's query over the
    /// registered queries. `batch` holds each field's distinct terms, already
    /// filtered by `accept` (a field in the filter set is never a term here).
    /// `filters` is the filter-field clause, when the batch has one value per
    /// filter field (`buildFilterFields`).
    pub fn build_query(
        &self,
        batch: &BTreeMap<String, Vec<Vec<u8>>>,
        accept: &dyn TermAcceptor,
        filters: Option<Clause>,
    ) -> Clause {
        let mut query = self.batch_query(batch, accept);
        if !self.filter_fields.is_empty() {
            let mut bq = BooleanQuery::new();
            bq.must.push(query);
            if let Some(f) = filters {
                bq.filter.push(f);
            }
            query = Clause::Boolean(Box::new(bq));
        }
        query
    }

    /// `TermFilteredPresearcher.buildQuery`'s term query: one `TermInSetQuery`
    /// per field, each pass a `MUST` of its own, and the any-token marker.
    fn batch_query(
        &self,
        batch: &BTreeMap<String, Vec<Vec<u8>>>,
        accept: &dyn TermAcceptor,
    ) -> Clause {
        // `buildQuery`: each field's terms through the handlers' wrappers, then
        // the filter fields and the acceptor; only a field that kept a term
        // reaches the builder
        let mut kept_by_field: Vec<(&String, Vec<Vec<u8>>)> = Vec::new();
        for (field, terms) in batch {
            if self.filter_fields.contains(field) {
                continue;
            }
            let mut stream = terms.clone();
            for handler in &self.handlers {
                stream = handler.wrap_term_stream(field, stream);
            }
            let mut kept: Vec<Vec<u8>> = stream
                .into_iter()
                .filter(|t| accept.accept(field, t))
                .collect();
            // a `TermInSetQuery` holds its terms sorted and distinct
            kept.sort_unstable();
            kept.dedup();
            if !kept.is_empty() {
                kept_by_field.push((field, kept));
            }
        }
        let mut query = BooleanQuery::new();
        if self.passes > 1 {
            let mut parent = BooleanQuery::new();
            for pass in 0..self.passes {
                let mut child = BooleanQuery::new();
                for (field, kept) in &kept_by_field {
                    child.should.push(Clause::TermInSet(TermInSetQuery {
                        field: field_for_pass(field, pass, self.passes),
                        terms: kept.clone(),
                    }));
                }
                parent.must.push(Clause::Boolean(Box::new(child)));
            }
            // `MultipassDocumentQueryBuilder.build`'s `parent`: the passes, one
            // group, which the presearcher ORs with the any-token marker
            query.should.push(Clause::Boolean(Box::new(parent)));
        } else {
            let mut shoulds = BooleanQuery::new();
            for (field, kept) in &kept_by_field {
                shoulds.should.push(Clause::TermInSet(TermInSetQuery {
                    field: (*field).clone(),
                    terms: kept.clone(),
                }));
            }
            query.should.push(Clause::Boolean(Box::new(shoulds)));
        }
        query.should.push(Clause::Term(TermQuery::new(
            ANYTOKEN_FIELD,
            ANYTOKEN.to_vec(),
        )));
        Clause::Boolean(Box::new(query))
    }

    /// `buildFilterClause`: a filter field's clause over the batch's values
    /// for it. Every document of the batch must carry the same value set, as
    /// Java checks (`docFreq == maxDoc`); a field with no values has none.
    pub fn build_filter_clause(
        &self,
        field: &str,
        values: &[(Vec<u8>, usize)],
        docs_in_batch: usize,
    ) -> Result<Option<Clause>, String> {
        let mut bq = BooleanQuery::new();
        for (term, doc_freq) in values {
            if *doc_freq != docs_in_batch {
                return Err(format!(
                    "Some documents in this batch do not have a term value of {field}:{}",
                    String::from_utf8_lossy(term)
                ));
            }
            bq.should
                .push(Clause::Term(TermQuery::new(field, term.clone())));
        }
        if bq.should.is_empty() {
            return Ok(None);
        }
        Ok(Some(Clause::Boolean(Box::new(bq))))
    }

    /// `buildFilterFields`: the filter fields' clauses, each a `MUST`; `None`
    /// when no field has a value.
    pub fn build_filter_fields(&self, per_field: &[(String, Option<Clause>)]) -> Option<Clause> {
        let mut bq = BooleanQuery::new();
        for (_, clause) in per_field {
            if let Some(c) = clause {
                bq.must.push(c.clone());
            }
        }
        if bq.must.is_empty() {
            return None;
        }
        Some(Clause::Boolean(Box::new(bq)))
    }
}

/// The name a pass's terms are indexed under: `field_<pass>` for a
/// multipass presearcher, the field itself for the plain one.
pub fn field_for_pass(field: &str, pass: usize, passes: usize) -> String {
    if passes > 1 {
        format!("{field}_{pass}")
    } else {
        field.to_string()
    }
}

/// `TermFilteredPresearcher.indexQuery`'s result: the query's indexed fields,
/// and the metadata values of its filter fields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexedQuery {
    pub fields: Vec<(String, Vec<Vec<u8>>)>,
    pub metadata: Vec<(String, String)>,
}

/// `Presearcher.NO_FILTERING`: every query is a candidate, and nothing is
/// indexed.
pub fn no_filtering() -> Clause {
    Clause::MatchAllDocs(MatchAllDocsQuery::new(0))
}

/// Shared handle, as a monitor holds its presearcher.
pub type SharedPresearcher = Arc<TermFilteredPresearcher>;

/// `Query.toString()` for the clauses a presearcher's batch query is built
/// from, as Lucene prints them: a `TermInSetQuery` as `field:(terms)`, sorted;
/// a term as `field:text`; and a `BooleanQuery` as its clauses, each with its
/// occurrence prefix, a nested boolean child in parentheses, and the boolean
/// itself unparenthesised (`BooleanQuery.toString` parenthesises only a
/// minimum-should-match boolean).
pub fn batch_string(clause: &Clause) -> String {
    match clause {
        Clause::TermInSet(t) => {
            let mut terms: Vec<String> = t
                .terms
                .iter()
                .map(|b| String::from_utf8_lossy(b).into_owned())
                .collect();
            terms.sort();
            format!("{}:({})", t.field, terms.join(" "))
        }
        Clause::Term(t) => format!("{}:{}", t.field, String::from_utf8_lossy(&t.term)),
        Clause::Boolean(b) => {
            let mut parts: Vec<String> = Vec::new();
            let mut add = |occur: &str, c: &Clause| {
                let text = batch_string(c);
                let text = if matches!(c, Clause::Boolean(_)) {
                    format!("({text})")
                } else {
                    text
                };
                parts.push(format!("{occur}{text}"));
            };
            for c in &b.must {
                add("+", c);
            }
            for c in &b.filter {
                add("#", c);
            }
            for c in &b.must_not {
                add("-", c);
            }
            for c in &b.should {
                add("", c);
            }
            parts.join(" ")
        }
        other => format!("{other:?}"),
    }
}

#[cfg(test)]
mod printer_tests {
    use super::*;
    use crate::query::{BooleanQuery, TermQuery};

    #[test]
    fn a_nested_boolean_group_is_parenthesised_and_the_marker_is_a_sibling() {
        let group = {
            let mut b = BooleanQuery::new();
            b.should.push(Clause::TermInSet(TermInSetQuery {
                field: "body".into(),
                terms: vec![b"fox".to_vec(), b"quick".to_vec()],
            }));
            Clause::Boolean(Box::new(b))
        };
        let mut top = BooleanQuery::new();
        top.should.push(group);
        top.should.push(Clause::Term(TermQuery::new(
            "__anytokenfield",
            b"__ANYTOKEN__".to_vec(),
        )));
        let got = batch_string(&Clause::Boolean(Box::new(top)));
        assert_eq!(got, "(body:(fox quick)) __anytokenfield:__ANYTOKEN__");
    }
}
