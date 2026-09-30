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
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::field_norms::FieldNorms;
use lucene_search::multi_segment::search_boolean_query_multi_segment_with_similarity;
use lucene_search::query::{
    BooleanQuery, BoostQuery, Clause, ConstantScoreQuery, DisjunctionMaxQuery, PhraseQuery,
    TermQuery,
};
use lucene_search::similarities::*;
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;

fn data(path: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/data")
        .join(path)
}

/// `GenSimilaritySearch.sims()`, by the same names.
fn sim(name: &str) -> Arc<dyn Similarity> {
    let lmjm = || -> Arc<dyn Similarity> {
        Arc::new(LmJelinekMercerSimilarity::new(CollectionModel::Default, true, 0.7).unwrap())
    };
    match name {
        "bm25" => Arc::new(Bm25Similarity::default()),
        "bm25_k2_b03" => Arc::new(Bm25Similarity::new(2.0, 0.3, true).unwrap()),
        "classic" => Arc::new(ClassicSimilarity::default()),
        "boolean" => Arc::new(BooleanSimilarity),
        "dfr_In_B_H2" => Arc::new(
            DfrSimilarity::new(BasicModel::In, AfterEffect::B, Normalization::H2_DEFAULT).unwrap(),
        ),
        "ib_LL_DF_H2" => Arc::new(
            IbSimilarity::new(Distribution::LL, Lambda::DF, Normalization::H2_DEFAULT).unwrap(),
        ),
        "dfi_saturated" => Arc::new(DfiSimilarity::new(Independence::Saturated)),
        "lmdirichlet" => Arc::new(LmDirichletSimilarity::default()),
        "lmjm_0.7" => lmjm(),
        "ax_f2exp" => {
            Arc::new(AxiomaticSimilarity::new(AxiomaticVariant::F2Exp, true, 0.5, 1, 0.2).unwrap())
        }
        "multi" => Arc::new(
            MultiSimilarity::new(vec![
                Arc::new(Bm25Similarity::default()),
                Arc::new(ClassicSimilarity::default()),
            ])
            .unwrap(),
        ),
        "perfield" => Arc::new(
            PerFieldSimilarity::new(lmjm())
                .with_field("title", Arc::new(ClassicSimilarity::default())),
        ),
        other => panic!("unknown similarity {other}"),
    }
}

/// `GenSimilaritySearch.parse`: the queries' prefix syntax.
fn parse(tok: &mut std::slice::Iter<'_, &str>) -> Clause {
    let mut next = || *tok.next().expect("truncated query");
    match next() {
        "T" => {
            let field = next();
            Clause::Term(TermQuery::new(field, next()))
        }
        "P" => {
            let field = next();
            let slop: u32 = next().parse().unwrap();
            let n: usize = next().parse().unwrap();
            let terms: Vec<&str> = (0..n).map(|_| next()).collect();
            Clause::Phrase(PhraseQuery::new(field, terms).with_slop(slop))
        }
        "B" => {
            let must: usize = next().parse().unwrap();
            let should: usize = next().parse().unwrap();
            let must_not: usize = next().parse().unwrap();
            let must: Vec<Clause> = (0..must).map(|_| parse(tok)).collect();
            let should: Vec<Clause> = (0..should).map(|_| parse(tok)).collect();
            let must_not: Vec<Clause> = (0..must_not).map(|_| parse(tok)).collect();
            Clause::Boolean(Box::new(
                BooleanQuery::new()
                    .with_must(must)
                    .with_should(should)
                    .with_must_not(must_not),
            ))
        }
        "X" => {
            let boost: f32 = next().parse().unwrap();
            Clause::Boost(Box::new(BoostQuery::new(parse(tok), boost)))
        }
        "C" => {
            let score: f32 = next().parse().unwrap();
            Clause::ConstantScore(Box::new(ConstantScoreQuery::new(parse(tok), score)))
        }
        "D" => {
            let tie: f32 = next().parse().unwrap();
            let n: usize = next().parse().unwrap();
            let disjuncts: Vec<Clause> = (0..n).map(|_| parse(tok)).collect();
            Clause::DisjunctionMax(Box::new(DisjunctionMaxQuery::new(disjuncts, tie)))
        }
        other => panic!("query op {other}"),
    }
}

fn query(text: &str) -> BooleanQuery {
    let tokens: Vec<&str> = text.split(' ').collect();
    let mut it = tokens.iter();
    let clause = parse(&mut it);
    assert!(it.next().is_none(), "trailing tokens in {text}");
    match clause {
        Clause::Boolean(b) => *b,
        // `BooleanQuery.rewrite`: a lone `MUST` clause is that clause.
        other => BooleanQuery::new().with_must([other]),
    }
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

/// A scoring clause that scores BM25 only is refused under another
/// similarity rather than scored with the wrong formula; a default BM25
/// takes the fast path whatever the clause.
#[test]
fn bm25_only_clauses_are_refused_under_another_similarity() {
    let dir = data("similarity_search_index");
    let reader = DirectoryReader::open(&FsDirectory::open(&dir)).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let norms = vec![None; segments.len()];
    let fuzzy = BooleanQuery::new()
        .with_should([Clause::Fuzzy(lucene_search::FuzzyQuery::new("body", "t10"))]);
    let classic = ClassicSimilarity::default();
    let err =
        search_boolean_query_multi_segment_with_similarity(&segments, &fuzzy, &norms, 5, &classic)
            .unwrap_err();
    assert!(
        matches!(err, lucene_search::Error::SimilarityUnsupported(_)),
        "{err}"
    );
    // Under a constant score or a filter it does not score: accepted.
    let wrapped = BooleanQuery::new()
        .with_must([Clause::ConstantScore(Box::new(ConstantScoreQuery::new(
            Clause::Fuzzy(lucene_search::FuzzyQuery::new("body", "t10")),
            1.0,
        )))])
        .with_filter([Clause::Fuzzy(lucene_search::FuzzyQuery::new("body", "t1"))]);
    search_boolean_query_multi_segment_with_similarity(&segments, &wrapped, &norms, 5, &classic)
        .unwrap();
    assert!(!search_boolean_query_multi_segment_with_similarity(
        &segments,
        &fuzzy,
        &norms,
        5,
        &Bm25Similarity::default(),
    )
    .unwrap()
    .is_empty());
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
