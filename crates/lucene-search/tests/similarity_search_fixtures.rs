//! Searching and indexing under a similarity other than the default,
//! differentially against Lucene 10.5.0: `fixtures/src/GenSimilaritySearch.java`.
//!
//! `IndexSearcher.setSimilarity`: a three-segment index with deletions and
//! sparse norms, searched under twelve similarities with term, boolean
//! (`SHOULD`/`MUST`/`MUST_NOT`, boosted, constant-scored, dis-max) and phrase
//! (exact and sloppy) queries; every top-20 hit and its score bits must be
//! Java's. The scorer tree prunes with the similarity's own bounds, so a
//! bound below a real score would drop a hit here.
//!
//! `IndexWriterConfig.setSimilarity`: documents written by this port's
//! `IndexWriter` under a per-field similarity -- overlaps not discounted, a
//! norm wider than a byte, a `DOCS` field whose norm counts distinct terms --
//! must carry the norms Java stored for the same documents.

mod simgrammar;

use std::collections::HashMap;
use std::sync::Arc;

use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_index::similarity::NormSimilarity;
use lucene_search::collector::TotalHitsRelation;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::field_norms::FieldNorms;
use lucene_search::index_searcher::IndexSearcher;
use lucene_search::multi_segment::search_boolean_query_multi_segment_with_similarity;
use lucene_search::query::{BooleanQuery, Clause, ConstantScoreQuery, SpanQuery};
use lucene_search::reader::exitable::QueryTimeout;
use lucene_search::similarities::*;
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;
use simgrammar::{query, sim};

fn data(path: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/data")
        .join(path)
}

#[test]
fn searches_under_every_similarity_match_lucene_bit_for_bit() {
    let dir = data("similarity_search_index");
    let text = std::fs::read_to_string(dir.join("searches.tsv"))
        .expect("run scripts/gen-fixtures.sh --only GenSimilaritySearch");
    let reader = DirectoryReader::open(&FsDirectory::open(&dir)).unwrap();
    assert_eq!(reader.segment_readers().len(), 3);
    assert!(
        reader
            .segment_readers()
            .iter()
            .any(|s| s.live_docs().is_some()),
        "the fixture deletes documents"
    );
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let owned = reader.field_norms_by_field(&["body".to_string(), "title".to_string()]);
    let norms: Vec<Option<&HashMap<String, FieldNorms<'_>>>> = owned.iter().map(Some).collect();

    let (mut cases, mut hits, mut failures) = (0, 0, Vec::new());
    let mut sims = std::collections::BTreeSet::new();
    for line in text.lines() {
        let [name, q, want] = line.split('\t').collect::<Vec<_>>()[..] else {
            panic!("bad line {line}");
        };
        let want: Vec<(i32, u32)> = want
            .split(',')
            .filter(|s| !s.is_empty())
            .map(|h| {
                let (d, b) = h.split_once(':').unwrap();
                (d.parse().unwrap(), u32::from_str_radix(b, 16).unwrap())
            })
            .collect();
        let similarity = sim(name);
        let got = search_boolean_query_multi_segment_with_similarity(
            &segments,
            &query(q),
            &norms,
            20,
            similarity.as_ref(),
        )
        .unwrap();
        let got: Vec<(i32, u32)> = got.iter().map(|h| (h.doc_id, h.score.to_bits())).collect();
        cases += 1;
        hits += want.len();
        sims.insert(name);
        if got != want {
            failures.push(format!("{name} [{q}]\n  rust {got:x?}\n  java {want:x?}"));
        }
    }
    assert!(
        cases >= 12 * 21 && sims.len() == 12 && hits > cases * 15,
        "{cases} searches, {} similarities, {hits} hits",
        sims.len()
    );
    assert!(
        failures.is_empty(),
        "{} of {cases} searches differ:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// Span, fuzzy and multi-phrase clauses score through the similarity (no
/// longer refused under a non-default one), and a `constant_score` around
/// one scores its constant whatever the similarity.
#[test]
fn every_scoring_clause_scores_through_the_similarity() {
    let dir = data("similarity_search_index");
    let reader = DirectoryReader::open(&FsDirectory::open(&dir)).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let owned = reader.field_norms_by_field(&["body".to_string()]);
    let norms: Vec<Option<&HashMap<String, FieldNorms<'_>>>> = owned.iter().map(Some).collect();
    let span = || {
        Clause::Span(SpanQuery::span_near(
            [
                SpanQuery::span_term("body", "t2"),
                SpanQuery::span_term("body", "t0"),
            ],
            3,
            false,
        ))
    };
    let run = |q: &BooleanQuery, s: &dyn Similarity| {
        search_boolean_query_multi_segment_with_similarity(&segments, q, &norms, 5, s).unwrap()
    };
    let scored = BooleanQuery::new().with_should([span()]);
    let bm25 = run(&scored, &Bm25Similarity::default());
    let classic = run(&scored, &ClassicSimilarity::default());
    assert!(!bm25.is_empty() && !classic.is_empty());
    assert_ne!(bm25[0].score, classic[0].score);
    assert!(bm25.iter().all(|h| h.score != 1.0), "{bm25:?}");
    let constant = BooleanQuery::new().with_must([Clause::ConstantScore(Box::new(
        ConstantScoreQuery::new(span(), 1.0),
    ))]);
    let hits = run(&constant, &ClassicSimilarity::default());
    assert!(!hits.is_empty() && hits.iter().all(|h| h.score == 1.0));
}

fn field_info(number: i32, name: &str, index_options: IndexOptions, vectors: bool) -> FieldInfo {
    FieldInfo {
        name: name.to_string(),
        number,
        store_term_vectors: vectors,
        omit_norms: false,
        store_payloads: false,
        soft_deletes_field: false,
        parent_field: false,
        index_options,
        doc_values_type: DocValuesType::None,
        doc_values_skip_index_type: DocValuesSkipIndexType::None,
        doc_values_gen: -1,
        attributes: Vec::new(),
        point_dimension_count: 0,
        point_index_dimension_count: 0,
        point_num_bytes: 0,
        vector_dimension: 0,
        vector_encoding: VectorEncoding::Float32,
        vector_similarity_function: VectorSimilarityFunction::Euclidean,
    }
}

/// `GenSimilaritySearch.WideNormSimilarity`: norm `length + 100 *
/// uniqueTermCount`, wider than a byte.
#[derive(Debug)]
struct WideNorm;

impl NormSimilarity for WideNorm {
    fn compute_norm(&self, _field: &str, state: &FieldInvertState) -> i64 {
        i64::from(state.length) + 100 * i64::from(state.unique_term_count)
    }
}

impl Similarity for WideNorm {
    fn scorer(
        &self,
        _field: &str,
        boost: f32,
        _collection: &CollectionStatistics,
        _terms: &[TermStatistics],
    ) -> Arc<dyn SimScorer> {
        BooleanSimilarity.scorer("", boost, _collection, _terms)
    }
}

/// Writes `norms_docs.tsv` through this port's `IndexWriter` under the
/// generator's per-field similarity and returns every document's norm per
/// field, as `norms.tsv` lists them.
fn rust_norms(term_vectors: bool) -> Vec<String> {
    let docs = std::fs::read_to_string(data("similarity_norms/norms_docs.tsv")).unwrap();
    let tmp = TempDir::new(&format!("similarity-norms-{term_vectors}"));
    let dir = FsDirectory::open(&tmp);
    let positions = IndexOptions::DocsAndFreqsAndPositions;
    let fields = vec![
        field_info(0, "body", positions, term_vectors),
        field_info(1, "title", positions, false),
        field_info(2, "tags", IndexOptions::Docs, false),
    ];
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version).unwrap();
    for f in ["body", "title", "tags"] {
        writer.add_postings_field(f).unwrap();
    }
    if term_vectors {
        // The general inversion path (`InMemoryInvertedIndex`), not the
        // `IndexingChain`-shaped one.
        writer.set_term_vector_field(Some("body")).unwrap();
    }
    let per_field: Arc<dyn Similarity> = Arc::new(
        PerFieldSimilarity::new(Arc::new(Bm25Similarity::default()))
            .with_field("body", Arc::new(ClassicSimilarity::new(false)))
            .with_field("title", Arc::new(WideNorm)),
    );
    // The object a searcher scores with configures the writer's norms.
    let norm_similarity: Arc<dyn NormSimilarity> = per_field;
    writer.set_similarity(Some(norm_similarity));
    let mut n = 0;
    for line in docs.lines() {
        let values: Vec<&str> = line.split('\t').collect();
        let fields = values
            .iter()
            .enumerate()
            .filter(|(_, v)| **v != "-")
            .map(|(i, v)| StoredField {
                field_number: i as i32,
                value: FieldValue::String(v.to_string()),
            })
            .collect();
        writer.add_document(Document { fields }).unwrap();
        n += 1;
    }
    writer.commit().unwrap();
    drop(writer);

    let reader = DirectoryReader::open(&dir).unwrap();
    let [seg] = reader.segment_readers() else {
        panic!("one segment");
    };
    let data = seg.norms_data().unwrap();
    let mut out = Vec::new();
    for (number, field) in ["body", "title", "tags"].iter().enumerate() {
        let entry = seg.norms_entry(number as i32).unwrap();
        for doc in 0..n {
            let norm = lucene_codecs::norms::norm_value(data, entry, doc).unwrap();
            out.push(format!(
                "{field}\t{doc}\t{}",
                norm.map_or("-".to_string(), |v| v.to_string())
            ));
        }
    }
    out
}

#[test]
fn norms_written_under_a_similarity_match_lucene() {
    let want: Vec<String> = std::fs::read_to_string(data("similarity_norms/norms.tsv"))
        .unwrap()
        .lines()
        .map(str::to_string)
        .collect();
    assert_eq!(want.len(), 180);
    // The corpus exercises what makes these norms differ from the default:
    // a norm wider than a byte, an empty field's `0`, an absent one.
    assert!(want
        .iter()
        .any(|l| l.starts_with("title\t") && l.ends_with("\t0")));
    assert!(want
        .iter()
        .any(|l| l.starts_with("title\t") && l.ends_with("\t-")));
    assert!(want
        .iter()
        .filter_map(|l| l.rsplit('\t').next()?.parse::<i64>().ok())
        .any(|v| v > 255));
    for term_vectors in [false, true] {
        let got = rust_norms(term_vectors);
        let diff: Vec<String> = got
            .iter()
            .zip(&want)
            .filter(|(g, w)| g != w)
            .map(|(g, w)| format!("rust {g:?} java {w:?}"))
            .collect();
        assert!(
            diff.is_empty() && got.len() == want.len(),
            "term vectors {term_vectors}: {} norms differ:\n{}",
            diff.len(),
            diff.join("\n")
        );
    }
}

/// `GenSimilaritySearch.CountingTimeout`: `after:N` exits on every check past
/// the `N`-th, `once:N` on the `N`-th only.
#[derive(Debug)]
struct CountingTimeout {
    once: bool,
    limit: usize,
    calls: std::sync::atomic::AtomicUsize,
}

impl CountingTimeout {
    fn new(spec: &str) -> Self {
        let (kind, n) = spec.split_once(':').unwrap();
        Self {
            once: kind == "once",
            limit: n.parse().unwrap(),
            calls: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    fn calls(&self) -> usize {
        self.calls.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl QueryTimeout for CountingTimeout {
    fn should_exit(&self) -> bool {
        let calls = self
            .calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1;
        if self.once {
            calls == self.limit
        } else {
            calls > self.limit
        }
    }
}

/// `IndexSearcher.setTimeout`: every leaf's bulk scorer asks the timeout
/// before each window of documents (100, growing by half), a leaf it stops
/// keeps what it collected and the next leaf is searched, and `timedOut()`
/// reports it. Against Lucene, over a 1200 + 300 document index: the number
/// of checks, the flag, the total and its relation, and every hit's score
/// bits, under BM25 and Classic, for timeouts that fire on the first check,
/// part-way, never, and on one check only.
#[test]
fn timeouts_stop_the_search_where_lucene_stops_it() {
    let dir = data("similarity_timeout_index");
    let text = std::fs::read_to_string(dir.join("timeouts.tsv"))
        .expect("run scripts/gen-fixtures.sh --only GenSimilaritySearch");
    let reader = DirectoryReader::open(&FsDirectory::open(&dir)).unwrap();
    assert_eq!(reader.segment_readers().len(), 2);
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let owned = reader.field_norms_by_field(&["body".to_string(), "title".to_string()]);
    let norms: Vec<Option<&HashMap<String, FieldNorms<'_>>>> = owned.iter().map(Some).collect();

    let (mut cases, mut partial, mut failures) = (0, 0, Vec::new());
    for line in text.lines() {
        let [name, spec, q, timed_out, checks, total, relation, want] =
            line.split('\t').collect::<Vec<_>>()[..]
        else {
            panic!("bad line {line}");
        };
        let similarity = sim(name);
        let mut searcher = IndexSearcher::new(&segments, &norms).unwrap();
        if name != "bm25" {
            searcher.set_similarity(similarity.as_ref());
        }
        let timeout = Arc::new(CountingTimeout::new(spec));
        assert!(searcher.timeout().is_none() && !searcher.timed_out());
        searcher.set_timeout(Some(timeout.clone()));
        assert!(searcher.timeout().is_some());
        let td = searcher.search(&query(q), 20).unwrap();
        let got_hits: Vec<String> = td
            .score_docs
            .iter()
            .map(|h| format!("{}:{:x}", h.doc, h.score.to_bits()))
            .collect();
        let got_relation = match td.total_hits.relation {
            TotalHitsRelation::EqualTo => "EQUAL_TO",
            TotalHitsRelation::GreaterThanOrEqualTo => "GREATER_THAN_OR_EQUAL_TO",
        };
        let got = format!(
            "{}\t{}\t{}\t{}\t{}",
            searcher.timed_out(),
            timeout.calls(),
            td.total_hits.value,
            got_relation,
            got_hits.join(",")
        );
        let want = format!("{timed_out}\t{checks}\t{total}\t{relation}\t{want}");
        cases += 1;
        partial += usize::from(timed_out == "true");
        if got != want {
            failures.push(format!("{name} {spec} [{q}]\n  rust {got}\n  java {want}"));
        }
    }
    assert!(
        cases == 2 * 5 * 9 && partial > cases / 2 && partial < cases,
        "{cases} searches, {partial} partial"
    );
    // `partialResult` is never reset: a later search without a timeout
    // leaves `timedOut()` set.
    let mut searcher = IndexSearcher::new(&segments, &norms).unwrap();
    searcher.set_timeout(Some(Arc::new(CountingTimeout::new("after:0"))));
    assert!(searcher
        .search(&query("T body t0"), 5)
        .unwrap()
        .score_docs
        .is_empty());
    assert!(searcher.timed_out());
    searcher.set_timeout(None);
    assert!(!searcher
        .search(&query("T body t0"), 5)
        .unwrap()
        .score_docs
        .is_empty());
    assert!(searcher.timed_out());
    assert!(
        failures.is_empty(),
        "{} of {cases} searches differ:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
