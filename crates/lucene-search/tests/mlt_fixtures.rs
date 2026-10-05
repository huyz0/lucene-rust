//! M10 T10.6, differentially against Lucene 10.5.0:
//! `fixtures/src/GenMoreLikeThis.java`'s `mlt.tsv`.
//!
//! `mlt.tsv`: `MoreLikeThis` under twelve settings over a four-segment index
//! with deletions -- for six documents (from term vectors, or stored values
//! re-analyzed) the interesting terms, the query's clauses and its hits;
//! `like(field, texts)` and `like(Map)`; `MoreLikeThisQuery` with hits and
//! explanations. The file is rebuilt here line for line and compared with
//! Lucene's (`common.tsv` is `common_terms_fixtures.rs`').

// Test fixtures' own arithmetic -- see `docs/arithmetic-gate.md`'s "Test code".
#![allow(clippy::arithmetic_side_effects)]

use std::collections::HashSet;
use std::sync::Arc;

use lucene_analysis::Analyzer;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::index_searcher::{IndexSearcher, SegmentNorms};
use lucene_search::mlt::{MoreLikeThis, MoreLikeThisQuery};
use lucene_search::query::{BooleanQuery, Clause};
use lucene_search::{Error, Result};
use lucene_store::FsDirectory;

fn data() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/mlt")
}

fn hex32(f: f32) -> String {
    format!("{:x}", f.to_bits())
}

fn err(e: &Error) -> String {
    match e {
        Error::Unsupported(_) => "!UnsupportedOperationException".into(),
        Error::IllegalArgument(_) => "!IllegalArgumentException".into(),
        Error::IllegalState(_) => "!IllegalStateException".into(),
        other => format!("!{other:?}"),
    }
}

fn clean(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('\t', "\\t")
        .replace('\n', "\\n")
}

fn g(r: Result<String>) -> String {
    match r {
        Ok(s) => clean(&s),
        Err(e) => err(&e),
    }
}

fn hits(s: &IndexSearcher<'_, '_>, q: &BooleanQuery) -> Result<String> {
    let td = s.search(q, 1000)?;
    let mut b = format!("{} ", td.total_hits.value);
    for sd in &td.score_docs {
        b.push_str(&format!("{}:{},", sd.doc, hex32(sd.score)));
    }
    Ok(b)
}

fn clauses(bq: &BooleanQuery) -> String {
    let mut b = format!("msm={}", bq.minimum_should_match);
    let tagged = bq
        .must
        .iter()
        .map(|c| ('M', c))
        .chain(bq.should.iter().map(|c| ('S', c)));
    for (occur, c) in tagged {
        let (q, boost) = match c {
            Clause::Boost(bq2) => (bq2.inner.as_ref(), format!("^{}", hex32(bq2.boost))),
            other => (other, String::new()),
        };
        let Clause::Term(t) = q else { panic!("{q:?}") };
        b.push_str(&format!(
            " {occur}{}:{}{boost}",
            t.field,
            String::from_utf8_lossy(&t.term)
        ));
    }
    b
}

const TEXTS: [&str; 3] = [
    "river stone light house river garden stone winter",
    "glacier tundra quartz QUARTZ zephyr anchor anchor",
    "the the the bridge",
];

const DOCS: [i32; 6] = [0, 3, 11, 17, 44, 102];

fn configure<'r>(
    reader: &'r DirectoryReader,
    s: &[&str],
    analyzer: &'r Analyzer,
) -> Result<MoreLikeThis<'r>> {
    let mut m = MoreLikeThis::new(reader);
    m.field_names = Some(s[1].split('|').map(str::to_string).collect());
    m.min_term_freq = s[2].parse().unwrap();
    m.min_doc_freq = s[3].parse().unwrap();
    if let Some(p) = s[4].strip_prefix("pct") {
        m.set_max_doc_freq_pct(p.parse().unwrap())?;
    } else if s[4] != "-1" {
        m.max_doc_freq = s[4].parse().unwrap();
    }
    m.max_query_terms = s[5].parse().unwrap();
    m.boost = s[6] == "true";
    m.boost_factor = s[7].parse().unwrap();
    m.min_word_len = s[8].parse().unwrap();
    m.max_word_len = s[9].parse().unwrap();
    if s[10] != "-" {
        m.stop_words = Some(s[10].split('|').map(str::to_string).collect());
    }
    if s[0] != "noanalyzer" {
        m.analyzer = Some(analyzer);
    }
    Ok(m)
}

fn mlt_lines(
    reader: &DirectoryReader,
    s: &IndexSearcher<'_, '_>,
    settings: &[&str],
    analyzer: &Analyzer,
    out: &mut Vec<String>,
) {
    for doc in DOCS {
        let head = format!("{}\tdoc {doc}", settings[0]);
        out.push(format!(
            "{head}\tterms\t{}",
            g(configure(reader, settings, analyzer)
                .and_then(|mut m| m.retrieve_interesting_terms(doc))
                .map(|t| t.join(",")))
        ));
        out.push(format!(
            "{head}\tquery\t{}",
            g(configure(reader, settings, analyzer)
                .and_then(|mut m| m.like_doc(doc))
                .map(|q| clauses(&q)))
        ));
        out.push(format!(
            "{head}\thits\t{}",
            g(configure(reader, settings, analyzer)
                .and_then(|mut m| m.like_doc(doc))
                .and_then(|q| hits(s, &q)))
        ));
    }
    for (i, text) in TEXTS.iter().enumerate() {
        let next = TEXTS[(i + 1) % TEXTS.len()];
        let head = format!("{}\ttext {i}", settings[0]);
        out.push(format!(
            "{head}\tterms\t{}",
            g(configure(reader, settings, analyzer)
                .and_then(|m| m.retrieve_interesting_terms_of_text(text, "body"))
                .map(|t| t.join(",")))
        ));
        out.push(format!(
            "{head}\tquery\t{}",
            g(configure(reader, settings, analyzer)
                .and_then(|m| m.like_texts("body", &[text, next]))
                .map(|q| clauses(&q)))
        ));
        out.push(format!(
            "{head}\tmap\t{}",
            g(configure(reader, settings, analyzer)
                .and_then(|mut m| {
                    m.like_fields(&[
                        ("body", vec![text.to_string()]),
                        ("tv", vec![next.to_string(), "42".to_string()]),
                    ])
                })
                .map(|q| clauses(&q)))
        ));
    }
}

const SETTINGS: [&str; 12] = [
    "defaults\tbody\t2\t5\t-1\t25\tfalse\t1\t0\t0\t-",
    "tf1\tbody\t1\t1\t-1\t25\tfalse\t1\t0\t0\t-",
    "tv\ttv\t1\t1\t-1\t25\tfalse\t1\t0\t0\t-",
    "both\tbody|tv\t1\t2\t-1\t8\tfalse\t1\t0\t0\t-",
    "boosted\tbody|tv|title\t1\t1\t-1\t6\ttrue\t2.5\t0\t0\t-",
    "few\tbody\t1\t1\t-1\t3\ttrue\t1\t0\t0\t-",
    "words\ttv\t1\t1\t-1\t25\tfalse\t1\t5\t6\t-",
    "stops\tbody\t1\t1\t-1\t25\tfalse\t1\t0\t0\tthe|river|stone",
    "maxdf\tbody\t1\t1\t40\t25\tfalse\t1\t0\t0\t-",
    "maxdfpct\ttv\t1\t1\tpct30\t25\tfalse\t1\t0\t0\t-",
    "nofields\ttitle|missing\t1\t1\t-1\t25\tfalse\t1\t0\t0\t-",
    "noanalyzer\tbody\t1\t1\t-1\t25\tfalse\t1\t0\t0\t-",
];

const MLTQ: [&str; 4] = [
    "0,body,body,0.3,1,5,-1,-",
    "1,body|tv,tv,0.5,1,4,2,-",
    "0,tv,body,0,2,10,-1,river",
    "2,body,body,1,1,5,-1,-",
];

fn as_boolean(c: Clause) -> BooleanQuery {
    BooleanQuery {
        must: vec![c],
        ..Default::default()
    }
}

fn mltq_lines(
    s: &IndexSearcher<'_, '_>,
    spec: &str,
    analyzer: &Arc<Analyzer>,
    out: &mut Vec<String>,
) {
    let q: Vec<&str> = spec.split(',').collect();
    let head = format!("mltq\t{spec}");
    let mut mq = MoreLikeThisQuery::new(
        TEXTS[q[0].parse::<usize>().unwrap()],
        q[1].split('|').map(str::to_string).collect(),
        Arc::clone(analyzer),
        q[2],
    );
    mq.percent_terms_to_match = q[3].parse().unwrap();
    mq.min_term_frequency = q[4].parse().unwrap();
    mq.max_query_terms = q[5].parse().unwrap();
    mq.min_doc_freq = q[6].parse().unwrap();
    if q[7] != "-" {
        mq.stop_words = Some(q[7].split('|').map(str::to_string).collect::<HashSet<_>>());
    }
    out.push(format!(
        "{head}\trewrite\t{}",
        // `searcher.rewrite(mq)` runs to a fixpoint: the boolean's own
        // rewrite follows (one clause is that clause, which the generator's
        // cast to `BooleanQuery` then refuses).
        g(mq.rewrite(s).map(|c| match c.rewrite() {
            Clause::Boolean(b) => clauses(&b),
            _ => "!ClassCastException".to_string(),
        }))
    ));
    let query = as_boolean(Clause::from(mq));
    out.push(format!("{head}\thits\t{}", g(hits(s, &query))));
    for doc in [0, 5, 50] {
        out.push(format!(
            "{head}\texplain {doc}\t{}",
            g(s.explain(&query, doc).map(|e| e.to_string()))
        ));
    }
}

fn compare(want: &str, got: &[String]) {
    let want: Vec<&str> = want.lines().collect();
    let mut bad = 0;
    for (i, (w, g)) in want.iter().zip(got).enumerate() {
        if *w != g {
            bad += 1;
            if bad <= 20 {
                eprintln!("line {}:\n  java {w}\n  rust {g}", i + 1);
            }
        }
    }
    assert_eq!(want.len(), got.len(), "line counts");
    assert_eq!(bad, 0, "{bad} of {} lines differ", want.len());
}

#[test]
fn more_like_this_matches_lucene() {
    let dir = data();
    let reader = DirectoryReader::open(&FsDirectory::open(dir.join("index"))).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let owned =
        reader.field_norms_by_field(&["body".to_string(), "tv".to_string(), "title".to_string()]);
    let norms: Vec<SegmentNorms<'_, '_>> = owned.iter().map(Some).collect();
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    let analyzer = Arc::new(Analyzer::standard(None));

    let mut got = Vec::new();
    for spec in SETTINGS {
        let settings: Vec<&str> = spec.split('\t').collect();
        mlt_lines(&reader, &searcher, &settings, &analyzer, &mut got);
    }
    for spec in MLTQ {
        mltq_lines(&searcher, spec, &analyzer, &mut got);
    }
    compare(&std::fs::read_to_string(dir.join("mlt.tsv")).unwrap(), &got);
}
