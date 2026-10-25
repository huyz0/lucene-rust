//! Test support: small faceted indexes (an index and its taxonomy, in
//! memory) and a searcher over them, for the facet counters' unit tests.
#![allow(clippy::arithmetic_side_effects)]

use lucene_index::buffered_updates::Term;
use lucene_index::document::{Document, NumericDocValuesField, Store, StringField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::merge_policy::upgrade::LATEST;
use lucene_store::ByteBuffersDirectory;

use super::taxonomy::directory::{DirectoryTaxonomyReader, DirectoryTaxonomyWriter};
use super::taxonomy::{AssociationFacetField, TaxonomyWriter};
use super::{FacetEntry, FacetField, FacetsCollector, FacetsCollectorManager, FacetsConfig};
use crate::directory_reader::DirectoryReader;
use crate::index_searcher::IndexSearcher;
use crate::query::{BooleanQuery, Clause, MatchAllDocsQuery, TermQuery};

/// An index and its taxonomy.
pub(crate) struct FacetIndex {
    pub(crate) index: ByteBuffersDirectory,
    pub(crate) taxo: ByteBuffersDirectory,
}

/// One document: its facet fields and the rest.
pub(crate) type Doc = (Vec<FacetEntry>, Document);

pub(crate) fn ff(dim: &str, path: &[&str]) -> FacetEntry {
    FacetField::new(dim, path).unwrap().into()
}

pub(crate) fn int_assoc(v: i32, dim: &str, path: &[&str]) -> FacetEntry {
    AssociationFacetField::int(v, dim, path).unwrap().into()
}

pub(crate) fn float_assoc(v: f32, dim: &str, path: &[&str]) -> FacetEntry {
    AssociationFacetField::float(v, dim, path).unwrap().into()
}

/// A document with an `id` and the given facets.
pub(crate) fn doc(id: &str, facets: Vec<FacetEntry>) -> Doc {
    let mut d = Document::new();
    d.add(StringField::new("id", id, Store::No));
    (facets, d)
}

/// Indexes `docs` through `config`, flushing a segment after each document
/// whose index is in `flush_after`, then deletes the documents with an `id`
/// in `delete`.
pub(crate) fn build(
    config: &FacetsConfig,
    docs: Vec<Doc>,
    flush_after: &[usize],
    delete: &[&str],
) -> FacetIndex {
    let index = ByteBuffersDirectory::new();
    let taxo = ByteBuffersDirectory::new();
    {
        let mut w = IndexWriter::open(&index, Vec::new(), "Lucene104", LATEST).unwrap();
        let tw = DirectoryTaxonomyWriter::open(&taxo).unwrap();
        for (i, (facets, d)) in docs.into_iter().enumerate() {
            let built = config
                .build(Some(&tw as &dyn TaxonomyWriter), &facets, d)
                .unwrap();
            w.add_fields_document(&built).unwrap();
            if flush_after.contains(&i) {
                w.flush().unwrap();
            }
        }
        if !delete.is_empty() {
            let terms: Vec<Term> = delete.iter().map(|id| Term::new("id", *id)).collect();
            w.delete_documents_by_term(&terms).unwrap();
        }
        w.commit().unwrap();
        tw.close().unwrap();
    }
    FacetIndex { index, taxo }
}

impl FacetIndex {
    /// Runs `f` with a searcher over the index, the reader and the taxonomy.
    pub(crate) fn with<R>(
        &self,
        f: impl FnOnce(&IndexSearcher<'_, '_>, &DirectoryReader, &DirectoryTaxonomyReader<'_>) -> R,
    ) -> R {
        let reader = DirectoryReader::open(&self.index).unwrap();
        let opened = reader.open_segments().unwrap();
        let segments = opened.as_open_segments();
        let owned = reader.field_norms_by_field(&["id".to_string()]).unwrap();
        let norms: Vec<_> = owned.iter().map(Some).collect();
        let searcher = IndexSearcher::new(&segments, &norms).unwrap();
        let taxo = DirectoryTaxonomyReader::open(&self.taxo).unwrap();
        f(&searcher, &reader, &taxo)
    }
}

/// Documents without facet fields, indexed as [`build`] does.
pub(crate) fn plain_index(docs: Vec<Document>, flush_after: &[usize]) -> FacetIndex {
    let docs = docs.into_iter().map(|d| (Vec::new(), d)).collect();
    build(&FacetsConfig::new(), docs, flush_after, &[])
}

/// A document with an `id` of `id`.
pub(crate) fn id_doc(id: &str) -> Document {
    doc(id, Vec::new()).1
}

/// `MatchAllDocsQuery`.
pub(crate) fn match_all(reader: &DirectoryReader) -> BooleanQuery {
    BooleanQuery {
        must: vec![Clause::MatchAllDocs(MatchAllDocsQuery::new(
            reader.max_doc(),
        ))],
        ..Default::default()
    }
}

/// `TermQuery(field:value)`.
pub(crate) fn term(field: &str, value: &str) -> BooleanQuery {
    BooleanQuery {
        must: vec![Clause::Term(TermQuery::new(field, value))],
        ..Default::default()
    }
}

/// `searcher.search(q, new FacetsCollectorManager(keepScores))`.
pub(crate) fn collect<'a>(
    searcher: &IndexSearcher<'_, 'a>,
    q: &BooleanQuery,
    keep_scores: bool,
) -> FacetsCollector<'a> {
    FacetsCollectorManager::new(keep_scores)
        .collect(searcher, q)
        .unwrap()
}

/// `TestTaxonomyFacetCounts.testBasic`'s configuration.
pub(crate) fn basic_config() -> FacetsConfig {
    let mut c = FacetsConfig::new();
    c.set_hierarchical("Publish Date", true);
    c
}

/// `TestTaxonomyFacetCounts.testBasic`'s documents (in three segments, as
/// it commits between them), each with an `id` of `d<i>` and a `num`.
pub(crate) fn basic_docs() -> Vec<Doc> {
    let rows: [(&str, [&str; 3]); 5] = [
        ("Bob", ["2010", "10", "15"]),
        ("Lisa", ["2010", "10", "20"]),
        ("Lisa", ["2012", "1", "1"]),
        ("Susan", ["2012", "1", "7"]),
        ("Frank", ["1999", "5", "5"]),
    ];
    rows.iter()
        .enumerate()
        .map(|(i, (author, date))| {
            let (f, mut d) = doc(
                &format!("d{i}"),
                vec![ff("Author", &[author]), ff("Publish Date", date)],
            );
            d.add(NumericDocValuesField::new("num", i as i64 + 1));
            (f, d)
        })
        .collect()
}

/// [`basic_docs`], indexed.
pub(crate) fn basic_index() -> FacetIndex {
    build(&basic_config(), basic_docs(), &[1, 3], &[])
}

/// `TestTaxonomyFacetAssociations`' configuration.
pub(crate) fn assoc_config() -> FacetsConfig {
    let mut c = FacetsConfig::new();
    c.set_index_field_name("int", "$facets.int");
    c.set_multi_valued("int", true);
    c.set_index_field_name("float", "$facets.float");
    c.set_multi_valued("float", true);
    c
}

/// 30 documents: every 4th has `int/a` 2, every 3rd `int/b` 3, every 5th
/// `float/a` 0.5 and `float/b` 0.2; the even ones have `id:even`.
pub(crate) fn assoc_index() -> FacetIndex {
    let docs = (0..30)
        .map(|i| {
            let mut f = Vec::new();
            if i % 4 == 0 {
                f.push(int_assoc(2, "int", &["a"]));
            }
            if i % 3 == 0 {
                f.push(int_assoc(3, "int", &["b"]));
            }
            if i % 5 == 0 {
                f.push(float_assoc(0.5, "float", &["a"]));
                f.push(float_assoc(0.2, "float", &["b"]));
            }
            doc(if i % 2 == 0 { "even" } else { "odd" }, f)
        })
        .collect();
    build(&assoc_config(), docs, &[9, 19], &[])
}
