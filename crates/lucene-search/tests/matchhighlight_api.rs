//! `MatchRegionRetriever`'s and `MatchHighlighter`'s arguments and limits,
//! over the `matchhighlight` fixture index.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::Arc;

use lucene_analysis::{Analyzer, StandardAnalyzer};
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::field_norms::FieldNorms;
use lucene_search::highlight::{
    compute_offset_retrieval_strategies, Error, FieldValueHighlighters, FieldValueProvider,
    MatchHighlighter, MatchRegionRetriever, OffsetRange,
};
use lucene_search::index_searcher::IndexSearcher;
use lucene_search::query::Clause;
use lucene_search::queryparser::classic::QueryParser;

fn reader() -> DirectoryReader {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/matchhighlight");
    DirectoryReader::open(&lucene_store::FsDirectory::open(dir.join("index"))).unwrap()
}

fn analyzer() -> Arc<Analyzer> {
    Arc::new(Analyzer::new(StandardAnalyzer::new()))
}

fn query(reader: &DirectoryReader, text: &str) -> Clause {
    QueryParser::new("offs", analyzer())
        .parse(text)
        .unwrap()
        .to_clause(reader.max_doc())
        .rewrite()
}

fn norms(reader: &DirectoryReader) -> Vec<HashMap<String, FieldNorms<'_>>> {
    let mut owned: Vec<HashMap<String, FieldNorms<'_>>> = Vec::new();
    for field in ["id", "offs", "pos", "docs", "title"] {
        for (i, n) in reader.field_norms(field).into_iter().enumerate() {
            if owned.len() <= i {
                owned.push(HashMap::new());
            }
            if let Some(n) = n {
                owned[i].insert(field.to_string(), n);
            }
        }
    }
    owned
}

type Hits = Vec<(i32, BTreeMap<String, Vec<OffsetRange>>)>;

/// The offsets `retriever` finds in `docs`, at most `max_hits` per field.
fn retrieve(
    retriever: &MatchRegionRetriever,
    searcher: &IndexSearcher<'_, '_>,
    reader: &DirectoryReader,
    docs: &[i32],
    max_hits: usize,
) -> Result<Hits, Error> {
    let mut out = Vec::new();
    retriever.highlight_documents(
        searcher,
        reader,
        docs,
        &mut |doc, _local, _values: &dyn FieldValueProvider, hits| {
            out.push((doc, hits.clone()));
            Ok(())
        },
        &|_| max_hits,
    )?;
    Ok(out)
}

#[test]
fn hits_per_field_are_limited_and_documents_must_be_sorted() {
    let r = reader();
    let opened = r.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let owned = norms(&r);
    let norms: Vec<Option<&HashMap<String, FieldNorms<'_>>>> = owned.iter().map(Some).collect();
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    let strategies = compute_offset_retrieval_strategies(&r, analyzer());
    let q = query(&r, "offs:fox pos:fox docs:fox");
    let retriever = MatchRegionRetriever::new(&q, strategies.as_ref(), |_| false, &|_| true);
    let docs: Vec<i32> = (0..r.max_doc()).collect();
    let all = retrieve(&retriever, &searcher, &r, &docs, usize::MAX).unwrap();
    let one = retrieve(&retriever, &searcher, &r, &docs, 1).unwrap();
    assert_eq!(all.len(), one.len());
    let mut limited = 0;
    for ((_, a), (_, b)) in all.iter().zip(&one) {
        for (field, ranges) in b {
            let full = &a[field];
            assert!(ranges.len() <= full.len(), "{field}");
            if field == "offs" {
                // one match: its offsets
                assert_eq!(ranges.as_slice(), &full[..1]);
            }
            if ranges.len() < full.len() {
                limited += 1;
            }
        }
    }
    assert!(limited > 0);
    // no hits at all
    let none = retrieve(&retriever, &searcher, &r, &docs, 0).unwrap();
    assert!(none.iter().all(|(_, h)| h.is_empty()));
    let e = retrieve(&retriever, &searcher, &r, &[3, 1], usize::MAX).unwrap_err();
    assert!(e.to_string().contains("must be sorted"), "{e}");
}

#[test]
fn a_field_the_index_does_not_know_has_no_strategy() {
    let r = reader();
    let opened = r.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let owned = norms(&r);
    let norms: Vec<Option<&HashMap<String, FieldNorms<'_>>>> = owned.iter().map(Some).collect();
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    let strategies = compute_offset_retrieval_strategies(&r, analyzer());
    // every field gets the strategy of a field the index lacks
    let unknown = |_: &str| strategies("nosuchfield");
    let q = query(&r, "offs:fox");
    let retriever = MatchRegionRetriever::new(&q, &unknown, |_| false, &|_| true);
    let e = retrieve(&retriever, &searcher, &r, &[0], usize::MAX).unwrap_err();
    assert!(
        e.to_string()
            .contains("FieldInfo is null for field: nosuchfield"),
        "{e}"
    );
}

#[test]
fn always_fetched_fields_come_back_without_matches() {
    let r = reader();
    let opened = r.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let owned = norms(&r);
    let norms: Vec<Option<&HashMap<String, FieldNorms<'_>>>> = owned.iter().map(Some).collect();
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    let render = |mh: &MatchHighlighter| {
        let hits = mh
            .highlight(&searcher, &r, &[0], &[query(&r, "offs:fox")])
            .unwrap();
        hits[0]
            .fields
            .iter()
            .map(|(f, _)| f.clone())
            .collect::<Vec<_>>()
    };
    let mut mh = MatchHighlighter::new(&r, analyzer());
    mh.append_field_highlighter(FieldValueHighlighters::verbatim_value(&["title"]));
    mh.append_field_highlighter(FieldValueHighlighters::verbatim_value(&["offs"]));
    // `title` is always fetched by its verbatim highlighter
    let with_title = render(&mh);
    assert!(with_title.contains(&"title".to_string()), "{with_title:?}");
    assert!(!with_title.contains(&"id".to_string()), "{with_title:?}");
    let mut mh = MatchHighlighter::new(&r, analyzer());
    mh.always_fetch_fields(&["id"]);
    mh.append_field_highlighter(FieldValueHighlighters::verbatim_value(&["id", "offs"]));
    let with_id = render(&mh);
    assert!(with_id.contains(&"id".to_string()), "{with_id:?}");
}
