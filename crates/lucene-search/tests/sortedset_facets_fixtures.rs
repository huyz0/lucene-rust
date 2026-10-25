//! M13 T13.4d2 against `fixtures/src/GenSortedSetFacets.java`: sorted-set
//! facet counting.
//!
//! - `DefaultSortedSetDocValuesReaderState` over a four-segment index with
//!   deletions and a one-segment index: each flat dim's ordinal range, each
//!   hierarchical dim's tree and the refusals (`state.tsv`).
//! - Every call the generator makes on `SortedSetDocValuesFacetCounts` and
//!   `ConcurrentSortedSetDocValuesFacetCounts`, counting all documents and a
//!   match-all, a half, a sparse and an empty query's hits (`facets.tsv`).
//! - `DrillSideways` over the `$facets` state, sequentially, with an executor
//!   and with `scoreSubDocsAtOnce` (`sideways.tsv`).
//! - The `Sort` overloads of `FacetsCollectorManager.search`/`searchAfter`
//!   and `DrillSideways.search` (`sorted.tsv`).
//! - `RandomSamplingFacetsCollector`'s sample for each query, size and seed,
//!   the sorted-set counts over it and their `amortizeFacetCounts`
//!   (`sampling.tsv`).
//! - The refusals (`errors.tsv`).
// Test-support code opts out of the arithmetic gate at the file boundary:
// see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use lucene_search::collector::{TotalHits, TotalHitsRelation};
use lucene_search::directory_reader::{DirectoryReader, SegmentReader};
use lucene_search::facets::{
    ConcurrentSortedSetDocValuesFacetCounts, DefaultSortedSetDocValuesReaderState, DrillDownQuery,
    DrillSideways, DrillSidewaysResult, FacetResult, FacetValue, Facets, FacetsCollector,
    FacetsCollectorManager, FacetsConfig, RandomSamplingFacetsCollectorManager,
    SortedSetDocValuesFacetCounts,
};
use lucene_search::field_norms::FieldNorms;
use lucene_search::index_searcher::{IndexSearcher, SegmentNorms};
use lucene_search::query::{BooleanQuery, Clause, MatchAllDocsQuery, TermQuery};
use lucene_search::reader::{DocIdSetIterator, NO_MORE_DOCS};
use lucene_search::top_docs::TopDocs;
use lucene_search::top_field::{FieldDoc, SortField, SortType, TopFieldDocs};
use lucene_search::{Error, Result};
use lucene_store::FsDirectory;

fn data() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/sortedset_facets")
}

fn read(p: &Path) -> String {
    fs::read_to_string(p).unwrap_or_else(|e| {
        panic!(
            "{}: {e} (run scripts/gen-fixtures.sh --only GenSortedSetFacets)",
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

// ---------------------------------------------------------------------------
// Rendering, as GenSortedSetFacets renders
// ---------------------------------------------------------------------------

/// `GenSortedSetFacets.esc`.
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

/// `GenSortedSetFacets.num`.
fn num(v: FacetValue) -> String {
    match v {
        FacetValue::Int(i) => format!("i{i}"),
        FacetValue::Float(f) => format!("f{:x}", f.to_bits()),
    }
}

/// `GenSortedSetFacets.res`.
fn res(r: Option<&FacetResult>) -> String {
    let Some(r) = r else {
        return "null".into();
    };
    let kids: Vec<String> = r
        .label_values
        .iter()
        .map(|l| format!("{}={}={}", esc(&l.label), num(l.value), l.count))
        .collect();
    format!(
        "{}|{}|{}|{}|{}",
        esc(&r.dim),
        esc(&r.path.join("/")),
        num(r.value),
        r.child_count,
        kids.join(",")
    )
}

/// `GenSortedSetFacets.list`.
fn list(l: Vec<FacetResult>) -> String {
    l.iter().map(|r| res(Some(r))).collect::<Vec<_>>().join(";")
}

/// `GenSortedSetFacets.attempt`.
fn attempt(r: Result<String>) -> String {
    match r {
        Ok(s) => s,
        Err(Error::IllegalArgument(m)) => format!("IAE:{}", esc(&m)),
        Err(Error::IllegalState(m)) => format!("ISE:{}", esc(&m)),
        Err(Error::Unsupported(m)) => format!("UOE:{}", esc(&m)),
        Err(other) => panic!("{other:?}"),
    }
}

fn r(f: Result<Option<FacetResult>>) -> String {
    attempt(f.map(|r| res(r.as_ref())))
}

fn l(f: Result<Vec<FacetResult>>) -> String {
    attempt(f.map(list))
}

/// `GenSortedSetFacets.digest`: hits, set bits and FNV-1a of the global
/// document ids.
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
            h ^= (m.context.doc_base + d) as i64 as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
            n += 1;
            d = it.next_doc().unwrap();
        }
    }
    format!("{hits}:{n}:{h:x}")
}

// ---------------------------------------------------------------------------
// The inputs, as GenSortedSetFacets builds them
// ---------------------------------------------------------------------------

/// `GenSortedSetFacets.config`.
fn config() -> FacetsConfig {
    let mut c = FacetsConfig::new();
    c.set_hierarchical("Publish Date", true);
    c.set_multi_valued("Tag", true);
    c.set_multi_valued("Genre", true);
    c.set_require_dim_count("Genre", true);
    c.set_index_field_name("Color", "$color");
    c
}

fn term(field: &str, v: &str) -> BooleanQuery {
    BooleanQuery {
        must: vec![Clause::Term(TermQuery::new(field, v))],
        ..Default::default()
    }
}

/// `GenSortedSetFacets.queries`.
fn queries(max_doc: i32) -> Vec<(&'static str, BooleanQuery)> {
    vec![
        (
            "all",
            BooleanQuery {
                must: vec![Clause::MatchAllDocs(MatchAllDocsQuery::new(max_doc))],
                ..Default::default()
            },
        ),
        ("even", term("q", "even")),
        ("rare", term("q", "rare")),
        ("none", term("q", "none")),
    ]
}

fn open(index: &str) -> DirectoryReader {
    DirectoryReader::open(&FsDirectory::open(data().join(index))).unwrap()
}

fn with_searcher<T>(reader: &DirectoryReader, f: impl FnOnce(&IndexSearcher<'_, '_>) -> T) -> T {
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let owned = reader
        .field_norms_by_field(&["q", "body", "$facets"].map(String::from))
        .unwrap();
    let norms: Vec<SegmentNorms<'_, '_>> = owned
        .iter()
        .map(|m: &HashMap<String, FieldNorms<'_>>| Some(m))
        .collect();
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    f(&searcher)
}

// ---------------------------------------------------------------------------
// The state
// ---------------------------------------------------------------------------

/// `GenSortedSetFacets.tree`.
fn tree(sb: &mut String, t: &lucene_search::facets::DimTree, ord: i64) {
    let kids: Vec<i64> = t.children(ord).collect();
    if kids.is_empty() {
        return;
    }
    let joined: Vec<String> = kids.iter().map(i64::to_string).collect();
    write!(sb, " {ord}:{}", joined.join(",")).unwrap();
    for k in kids {
        tree(sb, t, k);
    }
}

/// `GenSortedSetFacets.state`.
fn state(sb: &mut String, index: &str, s: &DefaultSortedSetDocValuesReaderState<'_>) {
    let mut dims: Vec<&str> = s.dims().collect();
    dims.sort_by(|a, b| a.encode_utf16().cmp(b.encode_utf16()));
    let f = s.field();
    writeln!(sb, "{index}\t{f}\tsize\t{}", s.size()).unwrap();
    for dim in dims {
        if s.facets_config().dim_config(dim).hierarchical {
            let t = s.dim_tree(dim).unwrap().unwrap();
            let mut tb = t.dim_start_ord.to_string();
            tree(&mut tb, t, t.dim_start_ord);
            writeln!(sb, "{index}\t{f}\ttree\t{}\t{tb}", esc(dim)).unwrap();
            let range = attempt(s.ord_range(dim).map(|r| format!("{r:?}")));
            writeln!(sb, "{index}\t{f}\tordRange\t{}\t{range}", esc(dim)).unwrap();
        } else {
            let r = s.ord_range(dim).unwrap().unwrap();
            writeln!(
                sb,
                "{index}\t{f}\trange\t{}\t{}-{}",
                esc(dim),
                r.start,
                r.end
            )
            .unwrap();
            let tree = attempt(s.dim_tree(dim).map(|t| format!("{t:?}")));
            writeln!(sb, "{index}\t{f}\tdimTree\t{}\t{tree}", esc(dim)).unwrap();
        }
    }
    let absent = s.ord_range("Nope").unwrap();
    assert!(absent.is_none());
    writeln!(sb, "{index}\t{f}\tabsentRange\tnull").unwrap();
}

// ---------------------------------------------------------------------------
// Counting
// ---------------------------------------------------------------------------

/// `GenSortedSetFacets.calls`.
fn calls(sb: &mut String, pre: &str, f: &dyn Facets, color_field: bool) {
    let mut line = |call: &str, result: String| writeln!(sb, "{pre}\t{call}\t{result}").unwrap();
    if color_field {
        line("dims5", l(f.get_all_dims(5)));
        line("topdims2_3", l(f.get_top_dims(2, 3)));
        line("top3:Color", r(f.get_top_children(3, "Color", &[])));
        line("all:Color", r(f.get_all_children("Color", &[])));
        line(
            "spec:Color/c1",
            attempt(f.get_specific_value("Color", &["c1"]).map(num)),
        );
        line("top3:Author", r(f.get_top_children(3, "Author", &[])));
        return;
    }
    line("dims5", l(f.get_all_dims(5)));
    line("dims1", l(f.get_all_dims(1)));
    line("topdims2_3", l(f.get_top_dims(2, 3)));
    line("topdims10_2", l(f.get_top_dims(10, 2)));
    line("topdims1_1", l(f.get_top_dims(1, 1)));
    line("topdims4_5", l(f.get_top_dims(4, 5)));
    for dim in ["Author", "Tag", "Genre", "Publish Date", "Nope", "Color"] {
        line(&format!("top3:{dim}"), r(f.get_top_children(3, dim, &[])));
        line(&format!("all:{dim}"), r(f.get_all_children(dim, &[])));
    }
    line("top5:Big", r(f.get_top_children(5, "Big", &[])));
    line("top1:Big", r(f.get_top_children(1, "Big", &[])));
    line("top1000:Tag", r(f.get_top_children(1000, "Tag", &[])));
    let paths: [&[&str]; 5] = [
        &["Publish Date", "2001"],
        &["Publish Date", "2001", "3"],
        &["Publish Date", "2003", "12"],
        &["Publish Date", "1999"],
        &["Publish Date", "2001", "3", "3"],
    ];
    for p in paths {
        let name = esc(&p.join("/"));
        line(
            &format!("top3:{name}"),
            r(f.get_top_children(3, p[0], &p[1..])),
        );
        line(&format!("all:{name}"), r(f.get_all_children(p[0], &p[1..])));
    }
    let specs: [&[&str]; 12] = [
        &["Author", "a3"],
        &["Tag", "t1"],
        &["Tag", "zz"],
        &["Genre", "g1"],
        &["Genre", "g9"],
        &["Genre"],
        &["Publish Date"],
        &["Publish Date", "2001"],
        &["Publish Date", "2001", "3"],
        &["Publish Date", "2001", "3", "15"],
        &["Big", "b7"],
        &["Nope", "x"],
    ];
    for p in specs {
        line(
            &format!("spec:{}", esc(&p.join("/"))),
            attempt(f.get_specific_value(p[0], &p[1..]).map(num)),
        );
    }
    line("err:top0", r(f.get_top_children(0, "Author", &[])));
    line("err:dims0", l(f.get_all_dims(0)));
    line("err:topdims0", l(f.get_top_dims(0, 1)));
    line("err:topdims_children0", l(f.get_top_dims(1, 0)));
    line("err:flatpath", r(f.get_top_children(3, "Author", &["x"])));
    line("err:flatpath_all", r(f.get_all_children("Author", &["x"])));
    line(
        "err:spec2",
        attempt(f.get_specific_value("Author", &["a3", "x"]).map(num)),
    );
}

/// `GenSortedSetFacets.counts`.
fn counts(
    sb: &mut String,
    index: &str,
    reader: &DirectoryReader,
    state: &DefaultSortedSetDocValuesReaderState<'_>,
) {
    let color = state.field() == "$color";
    let field = state.field();
    calls(
        sb,
        &format!("{index}\tssdv\t{field}\tcountall"),
        &SortedSetDocValuesFacetCounts::new(state).unwrap(),
        color,
    );
    calls(
        sb,
        &format!("{index}\tconc\t{field}\tcountall"),
        &ConcurrentSortedSetDocValuesFacetCounts::new(state).unwrap(),
        color,
    );
    with_searcher(reader, |searcher| {
        let manager = FacetsCollectorManager::new(false);
        for (name, q) in queries(reader.max_doc()) {
            let fc = manager.collect(searcher, &q).unwrap();
            calls(
                sb,
                &format!("{index}\tssdv\t{field}\t{name}"),
                &SortedSetDocValuesFacetCounts::with_hits(state, &fc).unwrap(),
                color,
            );
            calls(
                sb,
                &format!("{index}\tconc\t{field}\t{name}"),
                &ConcurrentSortedSetDocValuesFacetCounts::with_hits(state, &fc).unwrap(),
                color,
            );
        }
    });
}

#[test]
fn state_and_counts_match_lucene() {
    let config = config();
    let (mut state_tsv, mut facets_tsv) = (String::new(), String::new());
    for index in ["multi", "single"] {
        let reader = open(index);
        for field in ["$facets", "$color"] {
            let s =
                DefaultSortedSetDocValuesReaderState::with_field(&reader, field, &config).unwrap();
            state(&mut state_tsv, index, &s);
            counts(&mut facets_tsv, index, &reader, &s);
        }
    }
    assert_text_eq("state.tsv", &state_tsv, &read(&data().join("state.tsv")));
    assert_text_eq("facets.tsv", &facets_tsv, &read(&data().join("facets.tsv")));
}

/// The `errors.tsv` lines this file reproduces, by key.
fn java_errors() -> HashMap<String, String> {
    read(&data().join("errors.tsv"))
        .lines()
        .map(|l| {
            let (k, v) = l.split_once('\t').unwrap();
            (k.to_string(), v.to_string())
        })
        .collect()
}

#[test]
fn state_and_count_refusals_match_lucene() {
    let want = java_errors();
    let config = config();
    let multi = open("multi");
    let single = open("single");
    let refused = |reader: &DirectoryReader, field: &str| {
        attempt(
            DefaultSortedSetDocValuesReaderState::with_field(reader, field, &config)
                .map(|s| format!("{s:?}")),
        )
    };
    assert_eq!(refused(&multi, "$nope"), want["state:absent"]);
    assert_eq!(refused(&multi, "sorted"), want["state:sorted"]);
    assert_eq!(refused(&single, "sorted"), want["state:sorted_single"]);
    assert_eq!(refused(&multi, "id"), want["state:id"]);

    let ms = DefaultSortedSetDocValuesReaderState::new(&multi, &config).unwrap();
    with_searcher(&single, |searcher| {
        let other = FacetsCollectorManager::new(false)
            .collect(searcher, &queries(single.max_doc())[0].1)
            .unwrap();
        let ssdv = SortedSetDocValuesFacetCounts::with_hits(&ms, &other).map(|_| String::new());
        assert_eq!(attempt(ssdv), want["count:mismatch"]);
        let conc =
            ConcurrentSortedSetDocValuesFacetCounts::with_hits(&ms, &other).map(|_| String::new());
        assert_eq!(attempt(conc), want["conc:mismatch"]);
    });
}

// ---------------------------------------------------------------------------
// Drill sideways, the sorted searches and sampling
// ---------------------------------------------------------------------------

type Steps = &'static [&'static [&'static str]];

/// `GenSortedSetFacets.bases`.
fn bases(max_doc: i32) -> Vec<(&'static str, Option<Clause>)> {
    vec![
        ("null", None),
        (
            "all",
            Some(Clause::MatchAllDocs(MatchAllDocsQuery::new(max_doc))),
        ),
        ("even", Some(Clause::Term(TermQuery::new("q", "even")))),
        ("rare", Some(Clause::Term(TermQuery::new("q", "rare")))),
        ("alpha", Some(Clause::Term(TermQuery::new("body", "alpha")))),
    ]
}

/// `GenSortedSetFacets.drillDowns`.
fn drill_downs() -> Vec<(&'static str, Steps)> {
    vec![
        ("none", &[]),
        ("author", &[&["Author", "a3"]]),
        ("author_or", &[&["Author", "a3"], &["Author", "a5"]]),
        ("author_tag", &[&["Author", "a3"], &["Tag", "t1"]]),
        ("date", &[&["Publish Date", "2001"]]),
        ("deep", &[&["Publish Date", "2001", "3"]]),
        ("genre_tag", &[&["Genre", "g1"], &["Tag", "t6"]]),
        ("absent", &[&["Author", "zz"]]),
    ]
}

/// `GenSortedSetFacets.ddq`.
fn ddq<'c>(config: &'c FacetsConfig, base: Option<Clause>, steps: Steps) -> DrillDownQuery<'c> {
    let mut q = match base {
        Some(b) => DrillDownQuery::with_base(config, b),
        None => DrillDownQuery::new(config),
    };
    for step in steps {
        q.add(step[0], &step[1..]).unwrap();
    }
    q
}

fn relation(t: &TotalHits) -> String {
    let relation = match t.relation {
        TotalHitsRelation::EqualTo => "EQUAL_TO",
        TotalHitsRelation::GreaterThanOrEqualTo => "GREATER_THAN_OR_EQUAL_TO",
    };
    format!("{} {relation}", t.value)
}

/// `GenSortedSetFacets.topDocs` of a `TopDocs`.
fn top_docs(t: &TopDocs) -> String {
    let docs: Vec<String> = t
        .score_docs
        .iter()
        .map(|sd| format!("{}:{:x}", sd.doc, sd.score.to_bits()))
        .collect();
    format!("{} {}", relation(&t.total_hits), docs.join(","))
}

/// `GenSortedSetFacets.topDocs` of a `TopFieldDocs`, its scores beside it.
fn top_field_docs(t: &TopFieldDocs, scores: &[f32]) -> String {
    let docs: Vec<String> = t
        .hits
        .iter()
        .zip(scores)
        .map(|(fd, s)| {
            let fields: Vec<String> = fd.values.iter().map(i64::to_string).collect();
            format!("{}:{}:{:x}", fd.doc, fields.join("/"), s.to_bits())
        })
        .collect();
    format!("{} {}", relation(&t.total), docs.join(","))
}

/// `GenSortedSetFacets.sideways`.
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

/// `GenSortedSetFacets.facets`.
fn ds_facets(f: &dyn Facets, steps: Steps) -> String {
    let mut dims: Vec<&str> = Vec::new();
    for s in steps {
        if !dims.contains(&s[0]) {
            dims.push(s[0]);
        }
    }
    for d in ["Author", "Tag", "Genre"] {
        if !dims.contains(&d) {
            dims.push(d);
        }
    }
    let top: Vec<String> = dims
        .iter()
        .map(|d| r(f.get_top_children(3, d, &[])))
        .collect();
    format!("{}\t{}", l(f.get_all_dims(3)), top.join(";"))
}

/// `GenSortedSetFacets.dsResult`, the hits rendered.
fn ds_result<R>(r: &DrillSidewaysResult<'_, '_, R>, hits: &str, steps: Steps) -> String {
    format!(
        "{hits}\t{}\t{}\t{}",
        digest(&r.drill_down_facets_collector),
        sideways(r),
        ds_facets(&r.facets, steps)
    )
}

/// `GenSortedSetFacets.sort`.
fn sort(name: &str) -> Vec<SortField> {
    match name {
        "num" => vec![SortField::numeric("num", SortType::Long, false)],
        "num_rev" => vec![
            SortField::numeric("num", SortType::Long, true),
            SortField::numeric("num2", SortType::Long, false),
        ],
        _ => unreachable!("{name}"),
    }
}

#[test]
fn drill_sideways_over_a_state_matches_lucene() {
    let config = config();
    let reader = open("multi");
    let state = DefaultSortedSetDocValuesReaderState::new(&reader, &config).unwrap();
    let mut out = String::new();
    with_searcher(&reader, |searcher| {
        let seq = DrillSideways::with_state(searcher, &config, &state);
        let conc = DrillSideways::with_state(searcher, &config, &state).with_executor();
        let at_once =
            DrillSideways::with_state(searcher, &config, &state).with_score_sub_docs_at_once(true);
        let modes = [("seq", &seq), ("conc", &conc), ("atonce", &at_once)];
        for (bn, base) in bases(reader.max_doc()) {
            for (dn, steps) in drill_downs() {
                for (mn, ds) in modes {
                    let res = ds.search(&ddq(&config, base.clone(), steps), 10).unwrap();
                    let row = ds_result(&res, &top_docs(&res.hits), steps);
                    writeln!(out, "{bn}\t{dn}\t{mn}\t{row}").unwrap();
                }
            }
        }
    });
    assert_text_eq("sideways.tsv", &out, &read(&data().join("sideways.tsv")));
}

fn fcm_sorted(
    out: &mut String,
    searcher: &IndexSearcher<'_, '_>,
    readers: &[SegmentReader],
    max_doc: i32,
) {
    for qn in ["all", "even", "rare"] {
        let q = if qn == "all" {
            term("body", "alpha")
        } else {
            queries(max_doc)
                .into_iter()
                .find(|(n, _)| *n == qn)
                .unwrap()
                .1
        };
        for sn in ["num", "num_rev"] {
            for scores in [false, true] {
                let pre = format!("fcm\t{qn}\t{sn}\t{scores}");
                let fcm = FacetsCollectorManager::new(false);
                let r = FacetsCollectorManager::search_sorted(
                    searcher,
                    readers,
                    &q,
                    7,
                    &sort(sn),
                    scores,
                    &fcm,
                )
                .unwrap();
                let td = top_field_docs(&r.top_docs, &r.scores);
                writeln!(out, "{pre}\tfirst\t{td}\t{}", digest(&r.facets_collector)).unwrap();
                if r.top_docs.hits.len() > 3 {
                    let after = &r.top_docs.hits[3];
                    let next = FacetsCollectorManager::search_after_sorted(
                        searcher,
                        readers,
                        after,
                        &q,
                        5,
                        &sort(sn),
                        scores,
                        &fcm,
                    )
                    .unwrap();
                    let td = top_field_docs(&next.top_docs, &next.scores);
                    writeln!(
                        out,
                        "{pre}\tafter\t{td}\t{}",
                        digest(&next.facets_collector)
                    )
                    .unwrap();
                }
            }
        }
        let fcm = FacetsCollectorManager::new(false);
        let zero = FacetsCollectorManager::search_sorted(
            searcher,
            readers,
            &q,
            0,
            &sort("num"),
            false,
            &fcm,
        )
        .unwrap();
        let td = top_field_docs(&zero.top_docs, &zero.scores);
        writeln!(
            out,
            "fcm\t{qn}\tnum\tfalse\tzero\t{td}\t{}",
            digest(&zero.facets_collector)
        )
        .unwrap();
    }
}

#[test]
fn sorted_searches_match_lucene() {
    let config = config();
    let reader = open("multi");
    let readers = reader.segment_readers();
    let state = DefaultSortedSetDocValuesReaderState::new(&reader, &config).unwrap();
    let want_errors = java_errors();
    let mut out = String::new();
    with_searcher(&reader, |searcher| {
        fcm_sorted(&mut out, searcher, readers, reader.max_doc());
        let seq = DrillSideways::with_state(searcher, &config, &state);
        let all = drill_downs();
        let base_of = |bn: &str| {
            bases(reader.max_doc())
                .into_iter()
                .find(|(n, _)| *n == bn)
                .unwrap()
                .1
        };
        for bn in ["null", "alpha"] {
            for dn in ["none", "author", "author_tag"] {
                let steps = all.iter().find(|(n, _)| *n == dn).unwrap().1;
                for sn in ["num", "num_rev"] {
                    for scores in [false, true] {
                        let pre = format!("ds\t{bn}/{dn}\t{sn}\t{scores}");
                        let q = ddq(&config, base_of(bn), steps);
                        let first = seq
                            .search_sorted(readers, &q, None, None, 6, &sort(sn), scores)
                            .unwrap();
                        let hits = top_field_docs(&first.hits.0, &first.hits.1);
                        writeln!(out, "{pre}\tfirst\t{}", ds_result(&first, &hits, steps)).unwrap();
                        let after = first.hits.0.hits[2].clone();
                        let next = seq
                            .search_sorted(
                                readers,
                                &q,
                                Some(Clause::Term(TermQuery::new("q", "even"))),
                                Some(&after),
                                4,
                                &sort(sn),
                                scores,
                            )
                            .unwrap();
                        let hits = top_field_docs(&next.hits.0, &next.hits.1);
                        writeln!(
                            out,
                            "{pre}\tafter_filtered\t{}",
                            ds_result(&next, &hits, steps)
                        )
                        .unwrap();
                    }
                }
            }
        }

        let fcm = FacetsCollectorManager::new(false);
        let all_q = &queries(reader.max_doc())[0].1;
        let nullsort =
            FacetsCollectorManager::search_sorted(searcher, readers, all_q, 5, &[], false, &fcm)
                .map(|_| String::new());
        assert_eq!(attempt(nullsort), want_errors["fcm:nullsort"]);
        let past = FieldDoc {
            doc: reader.max_doc(),
            values: vec![0],
            terms: Vec::new(),
        };
        let past = FacetsCollectorManager::search_after_sorted(
            searcher,
            readers,
            &past,
            all_q,
            5,
            &sort("num"),
            false,
            &fcm,
        )
        .map(|_| String::new());
        assert_eq!(attempt(past), want_errors["fcm:after_past_limit"]);
        let ds_nullsort = seq
            .search_sorted(readers, &ddq(&config, None, &[]), None, None, 5, &[], false)
            .map(|_| String::new());
        assert_eq!(attempt(ds_nullsort), want_errors["fcm:nullsort"]);
    });
    assert_text_eq("sorted.tsv", &out, &read(&data().join("sorted.tsv")));
}

#[test]
fn random_sampling_matches_lucene() {
    let config = config();
    let reader = open("multi");
    let state = DefaultSortedSetDocValuesReaderState::new(&reader, &config).unwrap();
    let mut out = String::new();
    with_searcher(&reader, |searcher| {
        for qn in ["all", "even", "rare"] {
            let q = queries(reader.max_doc())
                .into_iter()
                .find(|(n, _)| *n == qn)
                .unwrap()
                .1;
            for size in [0, 1, 10, 50, 200, 5000] {
                for seed in [0i64, 42, -7] {
                    let mut c = RandomSamplingFacetsCollectorManager::new(size, seed)
                        .collect(searcher, &q)
                        .unwrap();
                    let original = digest(c.original_matching_docs());
                    let sampled = c.matching_docs().unwrap();
                    let sampled_digest = digest(sampled);
                    let paths: [&[&str]; 5] = [
                        &["Author"],
                        &["Tag"],
                        &["Genre"],
                        &["Publish Date"],
                        &["Publish Date", "2001"],
                    ];
                    let results: Vec<Option<FacetResult>> = {
                        let f = SortedSetDocValuesFacetCounts::with_hits(&state, sampled).unwrap();
                        paths
                            .iter()
                            .map(|p| f.get_top_children(5, p[0], &p[1..]).unwrap())
                            .collect()
                    };
                    write!(
                        out,
                        "{qn}\t{size}\t{seed}\t{original}\t{sampled_digest}\t{:x}",
                        c.sampling_rate().to_bits()
                    )
                    .unwrap();
                    for res_ in results {
                        let amortized = c
                            .amortize_facet_counts(res_.clone(), &config, &reader)
                            .unwrap();
                        write!(out, "\t{}\t{}", res(res_.as_ref()), res(amortized.as_ref()))
                            .unwrap();
                    }
                    out.push('\n');
                }
            }
        }
    });
    assert_text_eq("sampling.tsv", &out, &read(&data().join("sampling.tsv")));
}
