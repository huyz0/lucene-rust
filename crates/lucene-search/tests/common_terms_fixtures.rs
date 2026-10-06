//! M10 T10.6, differentially against Lucene 10.5.0:
//! `fixtures/src/GenMoreLikeThis.java`'s `common.tsv`.
//!
//! `CommonTermsQuery` over a four-segment index with deletions: fourteen
//! term sets and settings (a count or a fraction of `maxDoc` as the
//! frequent-term threshold, every legal occur on either side, `FILTER`
//! included, minimum-should-match counts and fractions, boosts, a term absent
//! everywhere, a field of its own, no terms and one term) -- `toString`,
//! every hit's score bits and four documents' explanations; then hits and
//! explanations for eight boosted booleans of `MUST`/`FILTER`/`SHOULD`/
//! `MUST_NOT` term, phrase and nested-boolean clauses (a boost the term
//! weights take into their explanation; the filters' and exclusions' weights
//! created without scores, two of the eight under `ClassicSimilarity`). The file is rebuilt here line for line and compared with
//! Lucene's.

// Test fixtures' own arithmetic -- see `docs/arithmetic-gate.md`'s "Test code".
#![allow(clippy::arithmetic_side_effects)]

use lucene_search::common_terms::CommonTermsQuery;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::index_searcher::{IndexSearcher, SegmentNorms};
use lucene_search::query::{BooleanQuery, BoostQuery, Clause, PhraseQuery, TermQuery};
use lucene_search::query_visitor::Occur;
use lucene_search::similarities::ClassicSimilarity;
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

fn as_boolean(c: Clause) -> BooleanQuery {
    BooleanQuery {
        must: vec![c],
        ..Default::default()
    }
}

const COMMON: [&str; 14] = [
    "body:river|body:glacier|body:quartz,0.3,SHOULD,SHOULD,0,0,1,1",
    "body:river|body:stone|body:glacier|body:tundra,0.25,SHOULD,MUST,0,0,1,1",
    "body:river|body:stone|body:light,0.1,SHOULD,SHOULD,0,0,1,1",
    "body:river|body:stone|body:light,0.1,MUST,SHOULD,0,0,1,1",
    "body:river|body:stone|body:light,0.1,SHOULD,SHOULD,0,2,1,1",
    "body:river|body:glacier|body:zephyr|body:quartz|body:anchor,20,SHOULD,SHOULD,0.5,0,1.5,0.5",
    "body:river|body:glacier|body:zephyr|body:quartz|body:anchor,20,MUST,SHOULD,2,0,1,2",
    "body:glacier|body:nosuch|tv:river,0.05,SHOULD,SHOULD,0,0,1,1",
    "body:river,0.1,SHOULD,SHOULD,0,0,1,1",
    ",0.1,SHOULD,SHOULD,0,0,1,1",
    "title:river|title:stone|title:quartz,0.1,SHOULD,SHOULD,0.4,0.6,1,1",
    "body:river|body:stone,0.1,MUST_NOT,SHOULD,0,0,1,1",
    "body:river|body:glacier|body:quartz,0.3,SHOULD,FILTER,0,0,2,1",
    "body:river|body:stone|body:glacier|body:tundra,0.25,MUST,FILTER,0,0,1,0.5",
];

/// `GenMoreLikeThis.BOOSTED`: `+`/`#`/`-` prefixed clauses -- a term
/// `f:t`, a phrase `"f:a,b"`, a nested boolean of `SHOULD` terms
/// `(f:a|f:b)` -- the boost, and the searcher's similarity (`""` for the
/// default).
const BOOSTED: [(&str, &str, &str); 8] = [
    ("+body:river #body:stone -body:glacier", "2.5", ""),
    ("#body:light body:river body:stone -body:quartz", "0.5", ""),
    ("#body:river #body:stone", "3", ""),
    ("+body:stone -body:river -body:zephyr", "1.5", ""),
    ("+body:river #\"body:the,the\" -body:quartz", "2", ""),
    (
        "#(body:light|body:winter) body:river -\"body:light,house\"",
        "1.5",
        "",
    ),
    (
        "+body:river #\"body:the,the\" #(body:light|body:winter) -body:quartz",
        "1",
        "classic",
    ),
    (
        "#body:river -\"body:river,window\" -(body:glacier|body:compass)",
        "1",
        "classic",
    ),
];

/// `GenMoreLikeThis.clause`.
fn clause(t: &str) -> Clause {
    if let Some(p) = t.strip_prefix('"') {
        let (f, words) = p.trim_end_matches('"').split_once(':').unwrap();
        return Clause::Phrase(PhraseQuery::new(f, words.split(',')));
    }
    if let Some(b) = t.strip_prefix('(') {
        return Clause::from(BooleanQuery {
            should: b.trim_end_matches(')').split('|').map(clause).collect(),
            ..Default::default()
        });
    }
    let (f, text) = t.split_once(':').unwrap();
    Clause::Term(TermQuery::new(f, text))
}

fn boosted_lines(
    plain: &IndexSearcher<'_, '_>,
    classic: &IndexSearcher<'_, '_>,
    (clauses, boost, sim): (&str, &str, &str),
    out: &mut Vec<String>,
) {
    let s = if sim.is_empty() { plain } else { classic };
    let mut b = BooleanQuery::new();
    for c in clauses.split(' ') {
        let (list, t) = match c.as_bytes()[0] {
            b'+' => (&mut b.must, &c[1..]),
            b'#' => (&mut b.filter, &c[1..]),
            b'-' => (&mut b.must_not, &c[1..]),
            _ => (&mut b.should, c),
        };
        list.push(clause(t));
    }
    let q = Clause::Boost(Box::new(BoostQuery::new(b, boost.parse().unwrap())));
    let sim = if sim.is_empty() {
        String::new()
    } else {
        format!("@{sim}")
    };
    let head = format!("boost\t{clauses}^{boost}{sim}");
    let query = as_boolean(q);
    out.push(format!("{head}\thits\t{}", g(hits(s, &query))));
    for doc in [0, 7, 33, 60] {
        out.push(format!(
            "{head}\texplain {doc}\t{}",
            g(s.explain(&query, doc).map(|e| e.to_string()))
        ));
    }
}

fn occur(s: &str) -> Occur {
    match s {
        "MUST" => Occur::Must,
        "SHOULD" => Occur::Should,
        "MUST_NOT" => Occur::MustNot,
        _ => Occur::Filter,
    }
}

fn common_lines(s: &IndexSearcher<'_, '_>, spec: &str, out: &mut Vec<String>) {
    let c: Vec<&str> = spec.split(',').collect();
    let head = spec.to_string();
    let mut q = match CommonTermsQuery::new(occur(c[2]), occur(c[3]), c[1].parse().unwrap()) {
        Ok(q) => q,
        Err(e) => {
            out.push(format!("{head}\tnew\t{}", err(&e)));
            return;
        }
    };
    if !c[0].is_empty() {
        for t in c[0].split('|') {
            let (f, text) = t.split_once(':').unwrap();
            q.add(f, text);
        }
    }
    q.low_freq_min_nr_should_match = c[4].parse().unwrap();
    q.high_freq_min_nr_should_match = c[5].parse().unwrap();
    q.low_freq_boost = c[6].parse().unwrap();
    q.high_freq_boost = c[7].parse().unwrap();
    out.push(format!("{head}\ttostring\t{}", clean(&q.to_string())));
    let query = as_boolean(Clause::from(q));
    out.push(format!("{head}\thits\t{}", g(hits(s, &query))));
    for doc in [0, 7, 33, 60] {
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
fn common_terms_match_lucene() {
    let dir = data();
    let reader = DirectoryReader::open(&FsDirectory::open(dir.join("index"))).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let owned =
        reader.field_norms_by_field(&["body".to_string(), "tv".to_string(), "title".to_string()]);
    let norms: Vec<SegmentNorms<'_, '_>> = owned.iter().map(Some).collect();
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    let mut got = Vec::new();
    for spec in COMMON {
        common_lines(&searcher, spec, &mut got);
    }
    let classic_sim = ClassicSimilarity::default();
    let mut classic = IndexSearcher::new(&segments, &norms).unwrap();
    classic.set_similarity(&classic_sim);
    for spec in BOOSTED {
        boosted_lines(&searcher, &classic, spec, &mut got);
    }
    compare(
        &std::fs::read_to_string(dir.join("common.tsv")).unwrap(),
        &got,
    );
}
