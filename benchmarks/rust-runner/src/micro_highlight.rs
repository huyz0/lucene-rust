//! M13 T13.2's pairs (`benchmarks/micro/java/HighlightMicro.java` is the Java
//! twin): the unified highlighter and the fast vector highlighter over the
//! fixture indexes, every templated query of `uhighlight-queries.txt` /
//! `vhighlight-queries.txt` against every document, default configuration.
//! `uh_postings`: field `offs` (offsets in postings); `uh_analysis`: field
//! `docs` (re-analysed); `uh_tv`: field `tv` (term vectors); `fvh`: the fast
//! vector highlighter on field `tv` (30 characters, three fragments).

use std::collections::HashMap;
use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

use lucene_analysis::{Analyzer, StandardAnalyzer};
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::field_norms::FieldNorms;
use lucene_search::highlight::unified::UnifiedHighlighter;
use lucene_search::highlight::vectorhighlight::FastVectorHighlighter;
use lucene_search::index_searcher::IndexSearcher;
use lucene_search::query::Clause;
use lucene_search::queryparser::classic::QueryParser;

use super::measure;

fn templated(file: &str, field: &str, a: &Arc<Analyzer>, max_doc: i32) -> Vec<Clause> {
    let parser = QueryParser::new("offs", Arc::clone(a));
    std::fs::read_to_string(file)
        .unwrap()
        .lines()
        .filter(|q| q.contains("F:"))
        .map(|q| {
            parser
                .parse(&q.replace("F:", &format!("{field}:")))
                .unwrap()
                .to_clause(max_doc)
        })
        .collect()
}

pub fn bench_highlight(w: Duration, m: Duration) {
    let a = Arc::new(Analyzer::new(StandardAnalyzer::new()));
    let reader = DirectoryReader::open(&lucene_store::FsDirectory::open(
        "fixtures/data/uhighlight/index",
    ))
    .unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let mut owned: Vec<HashMap<String, FieldNorms<'_>>> = Vec::new();
    for field in ["id", "offs", "pos", "docs", "tv", "both", "tvpos"] {
        for (i, n) in reader.field_norms(field).unwrap().into_iter().enumerate() {
            if owned.len() <= i {
                owned.push(HashMap::new());
            }
            if let Some(n) = n {
                owned[i].insert(field.to_string(), n);
            }
        }
    }
    let norms: Vec<Option<&HashMap<String, FieldNorms<'_>>>> = owned.iter().map(Some).collect();
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    let docs: Vec<i32> = (0..reader.max_doc()).collect();
    let uh = UnifiedHighlighter::new(Some(&reader), Some(Arc::clone(&a)));
    // `withWeightMatches(false)`: the classic path, phrases by `PhraseHelper`
    let classic = UnifiedHighlighter::new(Some(&reader), Some(Arc::clone(&a)))
        .with_flags(true, true, true, false);
    for (name, field, uh) in [
        ("uh_postings", "offs", &uh),
        ("uh_analysis", "docs", &uh),
        ("uh_tv", "tv", &uh),
        ("uh_phrases", "offs", &classic),
    ] {
        let queries = templated(
            "fixtures/corpus/uhighlight-queries.txt",
            field,
            &a,
            reader.max_doc(),
        );
        measure(name, w, m, || {
            for q in &queries {
                black_box(uh.highlight(&searcher, field, q, &docs, 2).unwrap());
            }
            (queries.len() * docs.len()) as u64
        });
    }

    let vreader = DirectoryReader::open(&lucene_store::FsDirectory::open(
        "fixtures/data/vhighlight/index",
    ))
    .unwrap();
    let queries = templated(
        "fixtures/corpus/vhighlight-queries.txt",
        "tv",
        &a,
        vreader.max_doc(),
    );
    let mut fvh = FastVectorHighlighter::default();
    measure("fvh", w, m, || {
        for q in &queries {
            let fq = fvh.get_field_query(q, Some(&vreader)).unwrap();
            for doc in 0..vreader.max_doc() {
                black_box(
                    fvh.get_best_fragments(&fq, &vreader, doc, "tv", 30, 3)
                        .unwrap(),
                );
            }
        }
        (queries.len() as u64) * u64::try_from(vreader.max_doc()).unwrap()
    });
}
