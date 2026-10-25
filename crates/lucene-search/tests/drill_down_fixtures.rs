//! M13 T13.4c against `fixtures/src/GenDrillDown.java`: drill-down and
//! drill-sideways.
//!
//! - `DrillSideways.search(ddq, 10)` for every base query and drill-down,
//!   sequentially (the `DrillSidewaysQuery`), with an executor, and with
//!   `scoreSubDocsAtOnce`, and the `search(ddq, TotalHitCountCollectorManager)`
//!   form: the hits, the drill-down and sideways documents and the facets
//!   render here exactly as Lucene's did (`sideways.tsv`).
//! - `search(after, ...)`, a filter and topN past `maxDoc` (`paging.tsv`).
//! - `DrillDownQuery`, `FacetQuery` and `MultiFacetQuery`: `toString`, the
//!   rewrite and a plain search (`queries.tsv`); the refusals (`errors.tsv`).
// Test-support code opts out of the arithmetic gate at the file boundary:
// see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use lucene_search::collector::{ScoreDoc, ScoreMode, TotalHitsRelation};
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::facets::taxonomy::directory::DirectoryTaxonomyReader;
use lucene_search::facets::{
    DrillDownQuery, DrillSideways, DrillSidewaysResult, FacetQuery, FacetResult, FacetValue,
    Facets, FacetsCollector, FacetsCollectorManager, FacetsConfig, MultiFacetQuery,
};
use lucene_search::field_norms::FieldNorms;
use lucene_search::grouping::GroupingCollectorManager;
use lucene_search::index_searcher::{IndexSearcher, SegmentNorms};
use lucene_search::leaf_collector::SegmentCollector;
use lucene_search::multi_segment::OpenSegment;
use lucene_search::query::{BooleanQuery, Clause, MatchAllDocsQuery, PhraseQuery, TermQuery};
use lucene_search::reader::{DocIdSetIterator, NO_MORE_DOCS};
use lucene_search::top_docs::TopDocs;
use lucene_search::{Error, Result};
use lucene_store::FsDirectory;

fn data() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/drill_down")
}

fn read(p: &Path) -> String {
    fs::read_to_string(p).unwrap_or_else(|e| {
        panic!(
            "{}: {e} (run scripts/gen-fixtures.sh --only GenDrillDown)",
            p.display()
        )
    })
}

/// Line-by-line comparison that names the first differing lines.
fn assert_text_eq(what: &str, got: &str, want: &str) {
    let (g, w): (Vec<&str>, Vec<&str>) = (got.lines().collect(), want.lines().collect());
    let mut diffs = Vec::new();
    for i in 0..g.len().max(w.len()) {
        let (a, b) = (g.get(i).copied(), w.get(i).copied());
        if a != b {
            diffs.push(format!(
                "line {}:\n  rust: {}\n  java: {}",
                i + 1,
                a.unwrap_or("<none>"),
                b.unwrap_or("<none>")
            ));
        }
    }
    assert!(
        diffs.is_empty(),
        "{what}: {} of {} lines differ\n{}",
        diffs.len(),
        w.len(),
        diffs.into_iter().take(12).collect::<Vec<_>>().join("\n")
    );
}

/// `GenDrillDown.config()`.
fn config() -> FacetsConfig {
    let mut c = FacetsConfig::new();
    c.set_hierarchical("Publish Date", true);
    c.set_multi_valued("Tag", true);
    c.set_index_field_name("Color", "$color");
    c.set_hierarchical("Color", true);
    c
}

/// `GenDrillDown.bases()`.
fn bases() -> Vec<(&'static str, Option<Clause>)> {
    let t = |f: &str, v: &str| Clause::Term(TermQuery::new(f, v));
    vec![
        ("null", None),
        ("all", Some(Clause::MatchAllDocs(MatchAllDocsQuery::new(0)))),
        ("even", Some(t("q", "even"))),
        ("rare", Some(t("q", "rare"))),
        ("veryrare", Some(t("q", "veryrare"))),
        ("alpha", Some(t("body", "alpha"))),
        ("gamma", Some(t("body", "gamma"))),
        (
            "phrase",
            Some(Clause::Phrase(PhraseQuery::new("body", ["alpha", "beta"]))),
        ),
        (
            "bool",
            Some(Clause::Boolean(Box::new(BooleanQuery {
                should: vec![t("body", "beta"), t("body", "gamma")],
                ..Default::default()
            }))),
        ),
    ]
}

type Steps = &'static [&'static [&'static str]];

/// `GenDrillDown.drillDowns()`.
fn drill_downs() -> Vec<(&'static str, Steps)> {
    vec![
        ("none", &[]),
        ("author", &[&["Author", "a3"]]),
        ("author_or", &[&["Author", "a3"], &["Author", "a5"]]),
        ("author_tag", &[&["Author", "a3"], &["Tag", "t1"]]),
        (
            "three",
            &[&["Author", "a3"], &["Tag", "t1"], &["Publish Date", "2001"]],
        ),
        ("deep", &[&["Publish Date", "2001", "3"]]),
        ("color", &[&["Color", "c1"]]),
        ("rare_author", &[&["Rare", "r5"], &["Author", "a3"]]),
        (
            "rare_author_tag",
            &[&["Rare", "r5"], &["Author", "a3"], &["Tag", "t6"]],
        ),
        ("absent_one", &[&["Author", "zz"], &["Tag", "t1"]]),
        ("absent_two", &[&["Author", "zz"], &["Tag", "tzz"]]),
        ("dim_root", &[&["Tag"]]),
        ("custom", &[&["Author", "a3"], &["Even", "*even"]]),
        ("phrase_dim", &[&["Tag", "t2"], &["Body", "*beta alpha"]]),
        (
            "two_phrases",
            &[
                &["Body", "*alpha beta"],
                &["Body2", "*beta alpha"],
                &["Author", "a1"],
            ],
        ),
        (
            "mixed_or",
            &[&["Tag", "t0"], &["Tag", "*t6"], &["Color", "c0", "s1"]],
        ),
    ]
}

fn drill_down(name: &str) -> Steps {
    drill_downs()
        .into_iter()
        .find(|(n, _)| *n == name)
        .map(|(_, s)| s)
        .unwrap()
}

/// `GenDrillDown.custom(spec)`.
fn custom(spec: &str) -> Clause {
    let s = &spec[1..];
    match s {
        "even" => Clause::Term(TermQuery::new("q", "even")),
        "t6" => Clause::Term(DrillDownQuery::term("$facets", "Tag", &["t6"]).unwrap()),
        _ => Clause::Phrase(PhraseQuery::new("body", s.split(' '))),
    }
}

/// `GenDrillDown.ddq(config, base, steps)`.
fn ddq<'c>(config: &'c FacetsConfig, base: Option<Clause>, steps: Steps) -> DrillDownQuery<'c> {
    let mut q = match base {
        Some(b) => DrillDownQuery::with_base(config, b),
        None => DrillDownQuery::new(config),
    };
    for step in steps {
        if step.len() == 2 && step[1].starts_with('*') {
            q.add_query(step[0], custom(step[1]));
        } else {
            q.add(step[0], &step[1..]).unwrap();
        }
    }
    q
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/// `GenDrillDown.esc`.
fn esc(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        if (c as u32) < 0x20 || c == '\\' {
            write!(out, "\\u{:04x}", c as u32).unwrap();
        } else {
            out.push(c);
        }
    }
    out
}

fn num(v: FacetValue) -> String {
    match v {
        FacetValue::Int(i) => format!("i{i}"),
        FacetValue::Float(f) => format!("f{:x}", f.to_bits()),
    }
}

/// `GenDrillDown.res`.
fn res(r: Option<FacetResult>) -> String {
    let Some(r) = r else {
        return "null".into();
    };
    let children: Vec<String> = r
        .label_values
        .iter()
        .map(|l| format!("{}={}={}", l.label, num(l.value), l.count))
        .collect();
    format!(
        "{}|{}|{}|{}|{}",
        r.dim,
        r.path.join("/"),
        num(r.value),
        r.child_count,
        children.join(",")
    )
}

/// `GenDrillDown.topDocs`.
fn top_docs(t: &TopDocs) -> String {
    let relation = match t.total_hits.relation {
        TotalHitsRelation::EqualTo => "EQUAL_TO",
        TotalHitsRelation::GreaterThanOrEqualTo => "GREATER_THAN_OR_EQUAL_TO",
    };
    let docs: Vec<String> = t
        .score_docs
        .iter()
        .map(|sd| format!("{}:{:x}", sd.doc, sd.score.to_bits()))
        .collect();
    format!("{} {relation} {}", t.total_hits.value, docs.join(","))
}

/// `GenDrillDown.digest`: hit count and FNV-1a of the global document ids.
fn digest(fc: &FacetsCollector<'_>) -> String {
    let mut mds: Vec<_> = fc.matching_docs().iter().collect();
    mds.sort_by_key(|m| m.context.doc_base);
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let (mut hits, mut n) = (0i64, 0i64);
    for m in mds {
        hits += i64::from(m.total_hits);
        let mut it = m.bits.iterator();
        let mut d = it.next_doc().unwrap();
        while d != NO_MORE_DOCS {
            h ^= (i64::from(m.context.doc_base + d)) as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
            n += 1;
            d = it.next_doc().unwrap();
        }
    }
    assert_eq!(n, hits, "bits != totalHits");
    format!("{hits}:{h:x}")
}

/// `GenDrillDown.sideways`.
fn sideways<R>(r: &DrillSidewaysResult<'_, '_, R>) -> String {
    let (Some(dims), Some(fcs)) = (&r.drill_sideways_dims, &r.drill_sideways_facets_collector)
    else {
        return "-".into();
    };
    dims.iter()
        .zip(fcs)
        .map(|(d, fc)| format!("{d}={}", digest(fc)))
        .collect::<Vec<_>>()
        .join(";")
}

fn err_or<T>(r: Result<T>, f: impl FnOnce(T) -> String) -> String {
    match r {
        Ok(v) => f(v),
        Err(Error::IllegalArgument(m)) => format!("err:{}", esc(&m)),
        Err(other) => panic!("{other:?}"),
    }
}

/// `GenDrillDown.facets`.
fn facets(f: &dyn Facets, steps: Steps) -> String {
    let mut dims: Vec<&str> = Vec::new();
    for s in steps {
        if !dims.contains(&s[0]) {
            dims.push(s[0]);
        }
    }
    for d in ["Author", "Tag"] {
        if !dims.contains(&d) {
            dims.push(d);
        }
    }
    let top: Vec<String> = dims
        .iter()
        .map(|d| err_or(f.get_top_children(3, d, &[]), res))
        .collect();
    let all = err_or(f.get_all_dims(3), |l| {
        l.into_iter()
            .map(|r| res(Some(r)))
            .collect::<Vec<_>>()
            .join(";")
    });
    format!("{all}\t{}", top.join(";"))
}

/// `GenDrillDown.result`, its `hits` field given.
fn result<R>(r: &DrillSidewaysResult<'_, '_, R>, hits: &str, steps: Steps) -> String {
    format!(
        "hits\t{hits}\tdd\t{}\tsw\t{}\tfacets\t{}",
        digest(&r.drill_down_facets_collector),
        sideways(r),
        facets(&r.facets, steps)
    )
}

// ---------------------------------------------------------------------------
// TotalHitCountCollectorManager
// ---------------------------------------------------------------------------

struct CountManager;
struct Count(i64);

impl<'a> SegmentCollector<'a> for Count {
    fn score_mode(&self) -> ScoreMode {
        ScoreMode::CompleteNoScores
    }
    fn set_next_reader(&mut self, _ord: usize, _leaf: &OpenSegment<'a>) -> Result<()> {
        Ok(())
    }
    fn collect(&mut self, _doc: i32, _score: f32) -> Result<()> {
        self.0 += 1;
        Ok(())
    }
}

impl<'a> GroupingCollectorManager<'a> for CountManager {
    type Collector = Count;
    type Output = i64;
    fn new_collector(&self) -> Result<Count> {
        Ok(Count(0))
    }
    fn reduce(&self, collectors: Vec<Count>) -> Result<i64> {
        Ok(collectors.iter().map(|c| c.0).sum())
    }
}

// ---------------------------------------------------------------------------
// The test
// ---------------------------------------------------------------------------

#[test]
fn drill_down_and_sideways_match_lucene() {
    let reader = DirectoryReader::open(&FsDirectory::open(data().join("index"))).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let owned = reader
        .field_norms_by_field(&["body", "q", "$facets", "$color"].map(String::from))
        .unwrap();
    let norms: Vec<SegmentNorms<'_, '_>> = owned
        .iter()
        .map(|m: &HashMap<String, FieldNorms<'_>>| Some(m))
        .collect();
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    assert_eq!(segments.len(), 3);
    let taxo_dir = FsDirectory::open(data().join("taxo"));
    let taxo = DirectoryTaxonomyReader::open(&taxo_dir).unwrap();
    let config = config();
    let seq = DrillSideways::new(&searcher, &config, &taxo);
    let conc = DrillSideways::new(&searcher, &config, &taxo).with_executor();
    let at_once = DrillSideways::new(&searcher, &config, &taxo).with_score_sub_docs_at_once(true);
    let modes = [("seq", &seq), ("conc", &conc), ("atonce", &at_once)];

    let mut sideways_out = String::new();
    let mut queries = String::new();
    for (bn, base) in bases() {
        for (dn, steps) in drill_downs() {
            let pre = format!("{bn}\t{dn}\t");
            for (mn, ds) in modes {
                let r = ds.search(&ddq(&config, base.clone(), steps), 10).unwrap();
                writeln!(
                    sideways_out,
                    "{pre}{mn}\t{}",
                    result(&r, &top_docs(&r.hits), steps)
                )
                .unwrap();
            }
            let c = seq
                .search_with(&ddq(&config, base.clone(), steps), &CountManager)
                .unwrap();
            let rendered = result(&c, "-", steps);
            writeln!(
                sideways_out,
                "{pre}count\tcount\t{}\t{}",
                c.hits,
                &rendered["hits\t-\t".len()..]
            )
            .unwrap();

            let q = ddq(&config, base.clone(), steps);
            writeln!(
                queries,
                "{pre}{}\t{}\t{}",
                esc(&q.to_string()),
                esc(&rewritten_to_string(&q)),
                top_docs(&searcher.search(&q.rewrite(), 20).unwrap())
            )
            .unwrap();
        }
    }

    // Paging and filters.
    let mut paging = String::new();
    let base_of = |name: &str| bases().into_iter().find(|(n, _)| *n == name).unwrap().1;
    for bn in ["null", "alpha", "bool"] {
        let base = base_of(bn);
        for dn in ["author", "author_tag", "none"] {
            let steps = drill_down(dn);
            let first = seq.search(&ddq(&config, base.clone(), steps), 4).unwrap();
            let last = first.hits.score_docs.last().unwrap();
            let after = ScoreDoc {
                doc_id: last.doc,
                score: last.score,
            };
            let next = seq
                .search_after(Some(after), &ddq(&config, base.clone(), steps), 6)
                .unwrap();
            writeln!(
                paging,
                "{bn}\t{dn}\tafter\t{}:{:x}\t{}",
                last.doc,
                last.score.to_bits(),
                result(&next, &top_docs(&next.hits), steps)
            )
            .unwrap();
            let filter = || Some(Clause::Term(TermQuery::new("q", "even")));
            let f = seq
                .search_filtered(&ddq(&config, base.clone(), steps), filter(), None, 7)
                .unwrap();
            writeln!(
                paging,
                "{bn}\t{dn}\tfilter\t-\t{}",
                result(&f, &top_docs(&f.hits), steps)
            )
            .unwrap();
            let fa = seq
                .search_filtered(&ddq(&config, base.clone(), steps), filter(), Some(after), 7)
                .unwrap();
            writeln!(
                paging,
                "{bn}\t{dn}\tfilter_after\t-\t{}",
                result(&fa, &top_docs(&fa.hits), steps)
            )
            .unwrap();
            let big = seq
                .search(&ddq(&config, base.clone(), steps), 100_000)
                .unwrap();
            writeln!(
                paging,
                "{bn}\t{dn}\tbig\t-\t{}",
                result(&big, &top_docs(&big.hits), steps)
            )
            .unwrap();
        }
    }

    // FacetQuery and MultiFacetQuery.
    enum Fq {
        One(FacetQuery),
        Multi(MultiFacetQuery),
    }
    let fqs = vec![
        (
            "facet_author",
            Fq::One(FacetQuery::new(&config, "Author", &["a3"]).unwrap()),
        ),
        (
            "facet_default",
            Fq::One(FacetQuery::with_default_config("Tag", &["t1"]).unwrap()),
        ),
        (
            "facet_color",
            Fq::One(FacetQuery::new(&config, "Color", &["c1", "s0"]).unwrap()),
        ),
        (
            "facet_dim",
            Fq::One(FacetQuery::new(&config, "Publish Date", &[]).unwrap()),
        ),
        (
            "facet_absent",
            Fq::One(FacetQuery::new(&config, "Author", &["zz"]).unwrap()),
        ),
        (
            "multi_tag",
            Fq::Multi(MultiFacetQuery::new(&config, "Tag", &[&["t6"], &["t1"]]).unwrap()),
        ),
        (
            "multi_default",
            Fq::Multi(
                MultiFacetQuery::with_default_config("Author", &[&["a5"], &["a3"], &["a5"]])
                    .unwrap(),
            ),
        ),
        (
            "multi_color",
            Fq::Multi(MultiFacetQuery::new(&config, "Color", &[&["c2", "s1"], &["c0"]]).unwrap()),
        ),
        (
            "multi_none",
            Fq::Multi(MultiFacetQuery::new(&config, "Tag", &[]).unwrap()),
        ),
    ];
    for (name, q) in fqs {
        let (s, sf, clause) = match q {
            Fq::One(q) => (
                q.to_string(),
                q.to_string_with_field("$facets"),
                Clause::from(q),
            ),
            Fq::Multi(q) => (
                q.to_string(),
                q.to_string_with_field("$facets"),
                Clause::from(q),
            ),
        };
        let bq = BooleanQuery {
            must: vec![clause],
            ..Default::default()
        };
        writeln!(
            queries,
            "facetquery\t{name}\t{}\t{}\t{}",
            esc(&s),
            esc(&sf),
            top_docs(&searcher.search(&bq, 20).unwrap())
        )
        .unwrap();
    }
    let term = DrillDownQuery::term("$facets", "Publish Date", &["2001", "3"]).unwrap();
    writeln!(
        queries,
        "term\t-\t{}",
        esc(&format!(
            "{}:{}",
            term.field,
            String::from_utf8_lossy(&term.term)
        ))
    )
    .unwrap();
    let mut nested = DrillDownQuery::with_base(&config, Clause::Term(TermQuery::new("q", "even")));
    nested.add_query(
        "Tag",
        FacetQuery::new(&config, "Tag", &["t1"]).unwrap().into(),
    );
    nested.add_query(
        "Tag",
        MultiFacetQuery::new(&config, "Tag", &[&["t6"]])
            .unwrap()
            .into(),
    );
    let nr = seq.search(&nested, 10).unwrap();
    writeln!(
        queries,
        "nested\t-\t{}\t{}",
        esc(&nested.to_string()),
        result(&nr, &top_docs(&nr.hits), &[&["Tag"]])
    )
    .unwrap();

    // Refusals.
    let mut errors = String::new();
    let mut error = |name: &str, r: Result<()>| {
        let shown = match r {
            Ok(()) => "ok".to_string(),
            Err(Error::IllegalArgument(m)) => esc(&m),
            Err(other) => panic!("{name}: {other:?}"),
        };
        writeln!(errors, "{name}\t{shown}").unwrap();
    };
    let author = || ddq(&config, None, drill_down("author"));
    error("search_topn_0", seq.search(&author(), 0).map(drop));
    error("search_topn_negative", seq.search(&author(), -3).map(drop));
    error(
        "add_empty_component",
        DrillDownQuery::new(&config).add("Author", &[""]),
    );
    error(
        "facet_query_empty",
        FacetQuery::new(&config, "Author", &[""]).map(drop),
    );
    error(
        "multi_facet_query_empty",
        MultiFacetQuery::new(&config, "Tag", &[&[""]]).map(drop),
    );
    let fcm = FacetsCollectorManager::new(false);
    error(
        "managers_size",
        seq.search_managers(
            &ddq(&config, None, drill_down("author_tag")),
            &fcm,
            Some(&[fcm][..]),
        )
        .map(drop),
    );
    error(
        "managers_null",
        seq.search_managers::<_, FacetsCollectorManager>(&author(), &fcm, None)
            .map(drop),
    );
    error(
        "managers_null_no_dims",
        seq.search_managers::<_, FacetsCollectorManager>(
            &ddq(&config, None, drill_down("none")),
            &fcm,
            None,
        )
        .map(drop),
    );
    error(
        "managers_ok",
        seq.search_managers(&author(), &fcm, Some(&[fcm][..]))
            .map(drop),
    );

    let dir = data();
    assert_text_eq(
        "sideways.tsv",
        &sideways_out,
        &read(&dir.join("sideways.tsv")),
    );
    assert_text_eq("paging.tsv", &paging, &read(&dir.join("paging.tsv")));
    assert_text_eq(
        "queries.tsv",
        &past_threshold(&queries),
        &past_threshold(&read(&dir.join("queries.tsv"))),
    );
    assert_text_eq("errors.tsv", &errors, &read(&dir.join("errors.tsv")));
}

/// A plain `searcher.search(q, 20)` stops counting past `IndexSearcher`'s
/// 1,000-hit threshold wherever its bulk scorer's window ends, so a
/// `GREATER_THAN_OR_EQUAL_TO` total says only "at least 1,000": that much
/// is checked, then the number is left out of the comparison (the
/// drill-sideways hits count every hit, and are compared exactly).
fn past_threshold(text: &str) -> String {
    const GTE: &str = " GREATER_THAN_OR_EQUAL_TO ";
    let mut out = String::new();
    for line in text.lines() {
        let fields: Vec<String> = line
            .split('\t')
            .map(|f| match f.split_once(GTE) {
                Some((total, docs)) => {
                    let n: u64 = total.parse().unwrap();
                    assert!(n >= 1000, "{line}");
                    format!(">=1000{GTE}{docs}")
                }
                None => f.to_string(),
            })
            .collect();
        writeln!(out, "{}", fields.join("\t")).unwrap();
    }
    out
}

/// `rewrite(searcher).toString()`: `*:*`, or the boolean query (the
/// `DrillDownQuery`'s own `toString`).
fn rewritten_to_string(q: &DrillDownQuery<'_>) -> String {
    match q.rewrite_clause() {
        Clause::MatchAllDocs(_) => "*:*".into(),
        Clause::Boolean(_) => q.to_string(),
        other => panic!("not a rewrite: {other:?}"),
    }
}
