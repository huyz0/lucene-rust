//! Lucene 10.6 (`branch_10_6`, #16484): `TestSpanMultiTermQueryWrapper.
//! testFrequentTermsRewrite`, its index and expected terms, over an index
//! written here in three segments (Java's `RandomIndexWriter` segments it at
//! random; the rewrite sums each term's statistics over them).

// Test fixtures' own arithmetic -- see `docs/arithmetic-gate.md`'s "Test code".
#![allow(clippy::arithmetic_side_effects)]

use std::collections::BTreeSet;
use std::sync::Arc;

use lucene_analysis::core_analysis::WhitespaceAnalyzer;
use lucene_analysis::Analyzer;
use lucene_index::document::{Document, Store, TextField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::extended_query::{MultiTermQuery, MultiTermSource, RewriteMethod};
use lucene_search::index_searcher::IndexSearcher;
use lucene_search::query::{PrefixQuery, WildcardQuery};
use lucene_search::spans::{FrequentTermsSpanBooleanQueryRewrite, SpanNode};
use lucene_store::ByteBuffersDirectory;

/// `term8` and `term54` occur in every document; `termN` (odd) accumulates,
/// so a lower odd `N` is in more documents; `termN` (even) is in one only.
fn index() -> ByteBuffersDirectory {
    let dir = ByteBuffersDirectory::new();
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", version).unwrap();
    w.set_analyzer(Some(Arc::new(Analyzer::new(WhitespaceAnalyzer::default()))));
    let mut terms = String::from(" term1");
    for i in 0..300 {
        let mut value = String::from("term8 term54");
        if i % 2 == 0 {
            value.push_str(&format!(" term{i}"));
        } else {
            terms.push_str(&format!(" term{i}"));
            value.push_str(&terms);
        }
        let mut doc = Document::new();
        doc.add(TextField::new("field", value, Store::No));
        w.add_fields_document(&doc).unwrap();
        if i % 100 == 99 {
            w.commit().unwrap();
        }
    }
    w.close().unwrap();
    dir
}

fn terms(node: &SpanNode) -> Vec<String> {
    let SpanNode::Or { clauses } = node else {
        panic!("a span disjunction: {node}")
    };
    clauses
        .iter()
        .map(|c| match c {
            SpanNode::Term { field, term } => {
                assert_eq!(field, "field");
                String::from_utf8(term.clone()).unwrap()
            }
            other => panic!("a span term: {other}"),
        })
        .collect()
}

#[test]
fn frequent_terms_rewrite() {
    let dir = index();
    let reader = DirectoryReader::open(&dir).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    assert_eq!(segments.len(), 3);
    let norms = vec![None; segments.len()];
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();

    let size = 8;
    let wildcard = MultiTermQuery::new(
        MultiTermSource::Wildcard(WildcardQuery::new("field", "term*")),
        RewriteMethod::default(),
    );
    let rewrite = FrequentTermsSpanBooleanQueryRewrite::with_order(
        size,
        FrequentTermsSpanBooleanQueryRewrite::DF_THEN_TTF_ORDER,
    );
    let got = terms(&rewrite.rewrite(&searcher, &wildcard).unwrap());
    assert_eq!(got.len(), size);
    let actual: BTreeSet<&str> = got.iter().map(String::as_str).collect();
    let expected: BTreeSet<&str> = [
        "term8", "term54", "term1", "term3", "term5", "term7", "term9", "term11",
    ]
    .into_iter()
    .collect();
    assert_eq!(actual, expected);
    // Popped least frequent first, ties in the heap's order: the order
    // Java's class (`branch_10_6`, compiled against the 10.5.2 jars) prints
    // over the same three segments.
    let java = [
        "term11", "term9", "term7", "term5", "term3", "term1", "term8", "term54",
    ];
    assert_eq!(got, java);

    // `DF_ORDER`: `term8` and `term54` tie on every count here.
    let df = FrequentTermsSpanBooleanQueryRewrite::new(size);
    assert_eq!(terms(&df.rewrite(&searcher, &wildcard).unwrap()), java);

    // Eleven terms under `term29`, six of them in one document: three of
    // those ties are kept, in the heap's order.
    let few = MultiTermQuery::new(
        MultiTermSource::Prefix(PrefixQuery::new("field", "term29")),
        RewriteMethod::default(),
    );
    assert_eq!(
        terms(&df.rewrite(&searcher, &few).unwrap()),
        ["term292", "term296", "term294", "term297", "term295", "term293", "term291", "term29"]
    );
    // No matching term: an empty disjunction; a size of 0 keeps none.
    let none = MultiTermQuery::new(
        MultiTermSource::Prefix(PrefixQuery::new("field", "nosuch")),
        RewriteMethod::default(),
    );
    assert!(terms(&df.rewrite(&searcher, &none).unwrap()).is_empty());
    let zero = FrequentTermsSpanBooleanQueryRewrite::new(0);
    assert!(terms(&zero.rewrite(&searcher, &wildcard).unwrap()).is_empty());
}
