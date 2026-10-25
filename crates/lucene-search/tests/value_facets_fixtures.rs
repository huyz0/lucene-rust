//! M13 T13.4d1 against `fixtures/src/GenValueFacets.java`: value, range,
//! range-on-range and facet-set counting.
//!
//! - Every call `GenValueFacets` makes on `LongValueFacetCounts`,
//!   `StringValueFacetCounts`, `LongRangeFacetCounts`,
//!   `DoubleRangeFacetCounts`, the range-on-range counters and
//!   `MatchingFacetSetsCounts` -- over a four-segment index with deletions
//!   and a one-segment index, for a match-all, a half, a sparse and an empty
//!   query, with and without a fast-match query -- renders here exactly as
//!   Lucene's did (`facets.tsv`).
//! - `LongRange.getQuery`/`DoubleRange.getQuery` (`queries.tsv`),
//!   `DynamicRangeUtil` (`dynamic.tsv`) and the refusals (`errors.tsv`).
// Test-support code opts out of the arithmetic gate at the file boundary:
// see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use lucene_index::document::RangeQueryType;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::facets::facetset::{
    decode_ints, decode_longs, DimRange, DoubleFacetSet, ExactFacetSetMatcher, FacetSet,
    FacetSetDecoder, FacetSetMatcher, FacetSetsField, FloatFacetSet, IntFacetSet, LongFacetSet,
    MatchingFacetSetsCounts, RangeFacetSetMatcher,
};
use lucene_search::facets::range::dynamic_range_util::{
    compute_dynamic_numeric_ranges, compute_dynamic_ranges, DynamicRangeInfo,
};
use lucene_search::facets::range::{
    DoubleRange, DoubleRangeFacetCounts, LongRange, LongRangeFacetCounts,
};
use lucene_search::facets::rangeonrange::{
    self, DoubleRangeOnRangeFacetCounts, LongRangeOnRangeFacetCounts,
};
use lucene_search::facets::{
    multi_double_values_source as mdvs, multi_long_values_source as mlvs, FacetResult, FacetValue,
    Facets, FacetsCollector, FacetsCollectorManager, LongValueFacetCounts,
    StringDocValuesReaderState, StringValueFacetCounts,
};
use lucene_search::field_norms::FieldNorms;
use lucene_search::index_searcher::{IndexSearcher, SegmentNorms};
use lucene_search::query::{BooleanQuery, Clause, MatchAllDocsQuery, TermQuery};
use lucene_search::reader::{DocIdSetIterator, IndexReader, NO_MORE_DOCS};
use lucene_search::values_source::{self, ValuesContext};
use lucene_search::{Error, Result};
use lucene_store::FsDirectory;

fn data() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/value_facets")
}

fn read(p: &Path) -> String {
    fs::read_to_string(p).unwrap_or_else(|e| {
        panic!(
            "{}: {e} (run scripts/gen-fixtures.sh --only GenValueFacets)",
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
// Rendering, as GenValueFacets renders
// ---------------------------------------------------------------------------

/// `GenValueFacets.esc`.
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

/// `String.compareTo`: UTF-16 order.
fn java_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    a.encode_utf16().cmp(b.encode_utf16())
}

/// `GenValueFacets.num`.
fn num(v: FacetValue) -> String {
    match v {
        FacetValue::Int(i) => format!("i{i}"),
        FacetValue::Float(f) => format!("f{:x}", f.to_bits()),
    }
}

/// `GenValueFacets.res`.
fn res(r: Option<FacetResult>, sort: bool) -> String {
    let Some(r) = r else {
        return "null".into();
    };
    let mut kids: Vec<String> = r
        .label_values
        .iter()
        .map(|l| format!("{}={}={}", esc(&l.label), num(l.value), l.count))
        .collect();
    if sort {
        kids.sort_by(|a, b| java_cmp(a, b));
    }
    format!(
        "{}|{}|{}|{}|{}",
        esc(&r.dim),
        r.path.join("/"),
        num(r.value),
        r.child_count,
        kids.join(",")
    )
}

/// `GenValueFacets.list`.
fn list(l: Vec<FacetResult>) -> String {
    l.into_iter()
        .map(|r| res(Some(r), false))
        .collect::<Vec<_>>()
        .join(";")
}

/// `GenValueFacets.attempt`.
fn attempt(r: Result<String>) -> String {
    match r {
        Ok(s) => s,
        Err(Error::IllegalArgument(m)) => format!("IAE:{}", esc(&m)),
        Err(Error::IllegalState(m)) => format!("ISE:{}", esc(&m)),
        Err(Error::Unsupported(_)) => "UOE".into(),
        Err(other) => panic!("{other:?}"),
    }
}

fn r(f: Result<Option<FacetResult>>) -> Result<String> {
    f.map(|r| res(r, false))
}

/// `GenValueFacets.sortedLines`.
fn sorted_lines(s: &str) -> String {
    let lines: Vec<&str> = s.split('\n').collect();
    let mut body: Vec<&str> = lines[1..].to_vec();
    body.sort_by(|a, b| java_cmp(a, b));
    esc(&format!("{}\n{}", lines[0], body.join("\n")))
}

struct Out<'s> {
    sb: &'s mut String,
    index: &'s str,
    query: &'s str,
}

impl Out<'_> {
    fn line(&mut self, call: &str, result: String) {
        writeln!(self.sb, "{}\t{}\t{call}\t{result}", self.index, self.query).unwrap();
    }

    /// `GenValueFacets.standard`.
    fn standard(&mut self, name: &str, f: &dyn Facets, dim: &str) {
        self.line(
            &format!("{name} top1"),
            attempt(r(f.get_top_children(1, dim, &[]))),
        );
        self.line(
            &format!("{name} top3"),
            attempt(r(f.get_top_children(3, dim, &[]))),
        );
        self.line(
            &format!("{name} top100"),
            attempt(r(f.get_top_children(100, dim, &[]))),
        );
        self.line(
            &format!("{name} dims2"),
            attempt(f.get_all_dims(2).map(list)),
        );
        self.line(
            &format!("{name} top0"),
            attempt(r(f.get_top_children(0, dim, &[]))),
        );
        self.line(
            &format!("{name} baddim"),
            attempt(r(f.get_top_children(3, "nope", &[]))),
        );
        self.line(
            &format!("{name} badpath"),
            attempt(r(f.get_top_children(3, dim, &["x"]))),
        );
        self.line(
            &format!("{name} allbaddim"),
            attempt(r(f.get_all_children("nope", &[]))),
        );
        self.line(
            &format!("{name} allbadpath"),
            attempt(r(f.get_all_children(dim, &["x"]))),
        );
        self.line(
            &format!("{name} spec"),
            attempt(f.get_specific_value(dim, &["1"]).map(num)),
        );
    }

    /// `GenValueFacets.longValues`.
    fn long_values(&mut self, name: &str, f: Result<LongValueFacetCounts>) {
        let f = f.unwrap_or_else(|e| panic!("{name}: {e:?}"));
        let dim = &name[name.find(':').unwrap() + 1..name.find('@').unwrap()];
        self.standard(name, &f, dim);
        self.line(
            &format!("{name} all"),
            attempt(f.get_all_children(dim, &[]).map(|r| res(r, true))),
        );
        self.line(
            &format!("{name} sorted"),
            res(Some(f.get_all_children_sort_by_value()), false),
        );
        self.line(&format!("{name} str"), sorted_lines(&f.to_string()));
    }

    /// `GenValueFacets.stringValues`.
    fn string_values(&mut self, name: &str, f: &StringValueFacetCounts<'_>, dim: &str) {
        for n in [1, 3, 100, 5000] {
            self.line(
                &format!("{name} top{n}"),
                attempt(r(f.get_top_children(n, dim, &[]))),
            );
        }
        self.line(
            &format!("{name} dims2"),
            attempt(f.get_all_dims(2).map(list)),
        );
        self.line(
            &format!("{name} top0"),
            attempt(r(f.get_top_children(0, dim, &[]))),
        );
        self.line(
            &format!("{name} baddim"),
            attempt(r(f.get_top_children(3, "nope", &[]))),
        );
        self.line(
            &format!("{name} badpath"),
            attempt(r(f.get_top_children(3, dim, &["x"]))),
        );
        self.line(
            &format!("{name} allbaddim"),
            attempt(r(f.get_all_children("nope", &[]))),
        );
        self.line(
            &format!("{name} all"),
            attempt(f.get_all_children(dim, &[]).map(|r| res(r, true))),
        );
        for label in ["s3", "s22", "zzz", "m1", "h0007", "x0001", "y3", "été"] {
            self.line(
                &format!("{name} spec:{}", esc(label)),
                attempt(f.get_specific_value(dim, &[label]).map(num)),
            );
        }
        self.line(
            &format!("{name} specbaddim"),
            attempt(f.get_specific_value("nope", &["s1"]).map(num)),
        );
        self.line(
            &format!("{name} specnopath"),
            attempt(f.get_specific_value(dim, &[]).map(num)),
        );
    }

    /// `GenValueFacets.ranges`.
    fn ranges<F: Facets + std::fmt::Display>(&mut self, name: &str, f: Result<F>, dim: &str) {
        let f = f.unwrap_or_else(|e| panic!("{name}: {e:?}"));
        self.standard(name, &f, dim);
        self.line(
            &format!("{name} all"),
            attempt(r(f.get_all_children(dim, &[]))),
        );
        self.line(&format!("{name} str"), esc(&f.to_string()));
    }

    /// `GenValueFacets.ranges2`.
    fn ranges2(&mut self, name: &str, f: Result<impl Facets>, dim: &str) {
        let f = f.unwrap_or_else(|e| panic!("{name}: {e:?}"));
        self.standard(name, &f, dim);
        self.line(
            &format!("{name} all"),
            attempt(r(f.get_all_children(dim, &[]))),
        );
    }

    /// `GenValueFacets.facetSets`.
    fn facet_sets(&mut self, name: &str, f: &dyn Facets, dim: &str) {
        for n in [1, 2, 100] {
            self.line(
                &format!("{name} top{n}"),
                attempt(r(f.get_top_children(n, dim, &[]))),
            );
        }
        self.line(
            &format!("{name} topother"),
            attempt(r(f.get_top_children(2, "other", &[]))),
        );
        self.line(
            &format!("{name} dims2"),
            attempt(f.get_all_dims(2).map(list)),
        );
        self.line(
            &format!("{name} top0"),
            attempt(r(f.get_top_children(0, dim, &[]))),
        );
        self.line(
            &format!("{name} all"),
            attempt(r(f.get_all_children(dim, &[]))),
        );
        self.line(
            &format!("{name} allbaddim"),
            attempt(r(f.get_all_children("nope", &[]))),
        );
        self.line(
            &format!("{name} allbadpath"),
            attempt(r(f.get_all_children(dim, &["x"]))),
        );
        self.line(
            &format!("{name} spec"),
            attempt(f.get_specific_value(dim, &["x"]).map(num)),
        );
    }
}

/// `GenValueFacets.digest`.
fn digest(fc: &FacetsCollector<'_>) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut hits = 0i64;
    for m in fc.matching_docs() {
        hits += i64::from(m.total_hits);
        let mut it = m.bits.iterator();
        let mut d = it.next_doc().unwrap();
        while d != NO_MORE_DOCS {
            h ^= (m.context.doc_base + d) as i64 as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
            d = it.next_doc().unwrap();
        }
    }
    format!("{hits}:{h:x}")
}

/// `GenValueFacets.ranges(List<DynamicRangeInfo>)`.
fn dyn_ranges(l: &[DynamicRangeInfo]) -> String {
    l.iter()
        .map(|r| {
            format!(
                "{},{},{},{},{:x}",
                r.count,
                r.weight,
                r.min,
                r.max,
                r.centroid.to_bits()
            )
        })
        .collect::<Vec<_>>()
        .join(";")
}

/// `GenValueFacets.attemptDynamic`.
fn attempt_dynamic(r: Result<Vec<DynamicRangeInfo>>) -> String {
    match r {
        Ok(l) => dyn_ranges(&l),
        Err(Error::IllegalArgument(m)) => format!("IAE:{}", esc(&m)),
        Err(other) => panic!("{other:?}"),
    }
}

// ---------------------------------------------------------------------------
// The inputs, as GenValueFacets builds them
// ---------------------------------------------------------------------------

fn term(v: &str) -> BooleanQuery {
    BooleanQuery {
        must: vec![Clause::Term(TermQuery::new("q", v))],
        ..Default::default()
    }
}

fn queries(max_doc: i32) -> Vec<(&'static str, BooleanQuery)> {
    vec![
        (
            "all",
            BooleanQuery {
                must: vec![Clause::MatchAllDocs(MatchAllDocsQuery::new(max_doc))],
                ..Default::default()
            },
        ),
        ("even", term("even")),
        ("rare", term("rare")),
        ("none", term("zzz")),
    ]
}

/// `GenValueFacets.FAST`.
fn fast() -> Option<BooleanQuery> {
    Some(term("even"))
}

fn long_ranges(name: &str) -> Vec<LongRange> {
    let l = |label, min, mi, max, ma| LongRange::new(label, min, mi, max, ma).unwrap();
    match name {
        "excl" => vec![
            l("lt0", i64::MIN, true, 0, false),
            l("0-10", 0, true, 10, true),
            l("10-20", 10, false, 20, true),
            l("big", 1000, true, i64::MAX, true),
        ],
        "gaps" => vec![l("b", 5, true, 6, true), l("a", -3, true, -1, true)],
        "overl" => vec![
            l("a", 0, true, 5, true),
            l("b", 3, true, 10, true),
            l("c", -5, true, 100, false),
            l("every", i64::MIN, true, i64::MAX, true),
            l("pt", 7, true, 7, true),
            l("hi", 1024, true, 1100, true),
            l("max", i64::MAX, true, i64::MAX, true),
        ],
        "empty" => vec![],
        _ => unreachable!(),
    }
}

fn double_ranges(name: &str) -> Vec<DoubleRange> {
    let d = |label, min, mi, max, ma| DoubleRange::new(label, min, mi, max, ma).unwrap();
    match name {
        "excl" => vec![
            d("neg", f64::NEG_INFINITY, true, 0.0, false),
            d("0-1", 0.0, true, 1.0, true),
            d("open", 1.0, false, f64::INFINITY, true),
        ],
        "overl" => vec![
            d("a", -1.5, true, 2.0, false),
            d("b", 0.5, true, 3.0, true),
            d("zero", -0.0, true, 0.0, true),
            d("inf", 2.0, true, f64::INFINITY, true),
            d("a", -1.5, true, 2.0, false),
        ],
        _ => unreachable!(),
    }
}

fn matchers(field: &str) -> Vec<Box<dyn FacetSetMatcher>> {
    let dl = |a, ai, b, bi| DimRange::from_longs(a, ai, b, bi).unwrap();
    let exact = |label: &str, fs: &dyn FacetSet| -> Box<dyn FacetSetMatcher> {
        Box::new(ExactFacetSetMatcher::new(label, fs).unwrap())
    };
    let range = |label: &str, r: &[DimRange]| -> Box<dyn FacetSetMatcher> {
        Box::new(RangeFacetSetMatcher::new(label, r).unwrap())
    };
    match field {
        "fs_long" => vec![
            exact("e12", &LongFacetSet::new(vec![1, 2]).unwrap()),
            range("r", &[dl(0, true, 2, true), dl(0, true, 10, false)]),
            exact("e37", &LongFacetSet::new(vec![3, 7]).unwrap()),
            range("none", &[dl(100, true, 200, true), dl(0, false, 1, true)]),
            exact("e00", &LongFacetSet::new(vec![0, 0]).unwrap()),
        ],
        "fs_int" => vec![
            exact("i120", &IntFacetSet::new(vec![1, 2, 0]).unwrap()),
            range(
                "neg",
                &[
                    dl(0, true, 2, true),
                    dl(i64::MIN, true, i64::MAX, true),
                    dl(-1, true, 0, false),
                ],
            ),
            exact("i000", &IntFacetSet::new(vec![0, 0, 0]).unwrap()),
        ],
        "fs_dbl" => vec![
            exact("half", &DoubleFacetSet::new(vec![0.5]).unwrap()),
            range(
                "neg",
                &[DimRange::from_doubles(-1.0, true, 0.0, false).unwrap()],
            ),
            range(
                "pos",
                &[DimRange::from_doubles(0.0, true, 10.0, true).unwrap()],
            ),
            exact("negzero", &DoubleFacetSet::new(vec![-0.0]).unwrap()),
        ],
        "fs_flt" => vec![
            exact("f", &FloatFacetSet::new(vec![0.5, -1.5]).unwrap()),
            range(
                "rf",
                &[
                    DimRange::from_floats(0.0, false, 1.0, true).unwrap(),
                    DimRange::from_floats(f32::NEG_INFINITY, true, 0.0, false).unwrap(),
                ],
            ),
        ],
        _ => unreachable!(),
    }
}

fn decoder(field: &str) -> &'static FacetSetDecoder {
    if field == "fs_int" || field == "fs_flt" {
        &decode_ints
    } else {
        &decode_longs
    }
}

fn ror_long(label: &str, min: i64, mi: bool, max: i64, ma: bool) -> rangeonrange::LongRange {
    rangeonrange::LongRange::new(label, min, mi, max, ma).unwrap()
}

fn ror_double(label: &str, min: f64, mi: bool, max: f64, ma: bool) -> rangeonrange::DoubleRange {
    rangeonrange::DoubleRange::new(label, min, mi, max, ma).unwrap()
}

const QUERY_TYPES: [(RangeQueryType, &str); 4] = [
    (RangeQueryType::Intersects, "INTERSECTS"),
    (RangeQueryType::Within, "WITHIN"),
    (RangeQueryType::Contains, "CONTAINS"),
    (RangeQueryType::Crosses, "CROSSES"),
];

// ---------------------------------------------------------------------------
// The run
// ---------------------------------------------------------------------------

struct Outputs {
    facets: String,
    queries: String,
    dynamic: String,
}

fn with_searcher<T>(reader: &DirectoryReader, f: impl FnOnce(&IndexSearcher<'_, '_>) -> T) -> T {
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let owned = reader.field_norms_by_field(&["q".to_string()]).unwrap();
    let norms: Vec<SegmentNorms<'_, '_>> = owned
        .iter()
        .map(|m: &HashMap<String, FieldNorms<'_>>| Some(m))
        .collect();
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    f(&searcher)
}

fn run_index(index: &str, reader: &DirectoryReader, out: &mut Outputs) {
    with_searcher(reader, |searcher| {
        let ctx = ValuesContext::new(searcher);
        let long1 = values_source::long_from_long_field("long1");
        let mut o = Out {
            sb: &mut out.facets,
            index,
            query: "-",
        };
        // Count-all forms.
        o.long_values(
            "lv:long1@field",
            LongValueFacetCounts::count_all("long1", reader),
        );
        o.long_values(
            "lv:longm@field",
            LongValueFacetCounts::count_all("longm", reader),
        );
        o.long_values(
            "lv:long1@src",
            LongValueFacetCounts::count_all_with_values_source(
                "long1",
                long1.as_ref(),
                &ctx,
                reader,
            ),
        );
        o.long_values(
            "lv:longm@multi",
            LongValueFacetCounts::count_all_with_multi_values_source(
                "longm",
                mlvs::from_long_field("longm").as_ref(),
                &ctx,
                reader,
            ),
        );
        o.long_values(
            "lv:long1@single",
            LongValueFacetCounts::count_all_with_multi_values_source(
                "long1",
                mlvs::from_single_valued(Arc::clone(&long1)).as_ref(),
                &ctx,
                reader,
            ),
        );
        o.long_values(
            "lv:nofield@field",
            LongValueFacetCounts::count_all("nofield", reader),
        );
        for field in ["str", "strm", "strhi", "strhim", "nofield"] {
            let state = StringDocValuesReaderState::new(reader, field).unwrap();
            let f = StringValueFacetCounts::new(&state).unwrap();
            o.string_values(&format!("sv:{field}"), &f, field);
        }

        let manager = FacetsCollectorManager::new(false);
        for (query, q) in queries(reader.max_doc()) {
            let fc = manager.collect(searcher, &q).unwrap();
            let fc_all = manager
                .collect(searcher, &queries(reader.max_doc())[0].1)
                .unwrap();
            let mut o = Out {
                sb: &mut out.facets,
                index,
                query,
            };
            o.line("hits", digest(&fc));

            // LongValueFacetCounts.
            let lv_src = |name: &str, src: Arc<dyn mlvs::MultiLongValuesSource>, field: &str| {
                (
                    name.to_string(),
                    LongValueFacetCounts::with_multi_values_source(field, src.as_ref(), &ctx, &fc),
                )
            };
            o.long_values("lv:long1@field", LongValueFacetCounts::new("long1", &fc));
            o.long_values("lv:longm@field", LongValueFacetCounts::new("longm", &fc));
            o.long_values(
                "lv:nofield@field",
                LongValueFacetCounts::new("nofield", &fc),
            );
            o.long_values(
                "lv:long1@src",
                LongValueFacetCounts::with_values_source("long1", long1.as_ref(), &ctx, &fc),
            );
            for (name, f) in [
                lv_src("lv:longm@multi", mlvs::from_long_field("longm"), "longm"),
                lv_src(
                    "lv:long1@single",
                    mlvs::from_single_valued(Arc::clone(&long1)),
                    "long1",
                ),
                lv_src(
                    "lv:dblm@tolong",
                    mdvs::to_multi_long_values_source(mdvs::from_double_field("dblm")),
                    "dblm",
                ),
                lv_src(
                    "lv:flt@tolong",
                    mdvs::to_multi_long_values_source(mdvs::from_float_field("flt")),
                    "flt",
                ),
                lv_src(
                    "lv:dblm@sortable",
                    mdvs::to_sortable_multi_long_values_source(mdvs::from_double_field("dblm")),
                    "dblm",
                ),
                lv_src("lv:longm@int", mlvs::from_int_field("longm"), "longm"),
            ] {
                o.long_values(&name, f);
            }

            // StringValueFacetCounts.
            for field in ["str", "strm", "strhi", "strhim", "nofield"] {
                let state = StringDocValuesReaderState::new(reader, field).unwrap();
                let f = StringValueFacetCounts::with_hits(&state, &fc).unwrap();
                o.string_values(&format!("sv:{field}"), &f, field);
            }

            // LongRangeFacetCounts.
            for set in ["excl", "gaps", "overl", "empty"] {
                let lr = long_ranges(set);
                let p = format!("lr:{set}:");
                o.ranges(
                    &format!("{p}long1@field"),
                    LongRangeFacetCounts::new("long1", &fc, &lr),
                    "long1",
                );
                o.ranges(
                    &format!("{p}longm@field"),
                    LongRangeFacetCounts::new("longm", &fc, &lr),
                    "longm",
                );
                o.ranges(
                    &format!("{p}long1@src"),
                    LongRangeFacetCounts::with_values_source(
                        "long1",
                        long1.as_ref(),
                        &ctx,
                        &fc,
                        None,
                        &lr,
                    ),
                    "long1",
                );
                o.ranges(
                    &format!("{p}longm@multi"),
                    LongRangeFacetCounts::with_multi_values_source(
                        "longm",
                        mlvs::from_long_field("longm").as_ref(),
                        &ctx,
                        &fc,
                        None,
                        &lr,
                    ),
                    "longm",
                );
                o.ranges(
                    &format!("{p}long1@single"),
                    LongRangeFacetCounts::with_multi_values_source(
                        "long1",
                        mlvs::from_single_valued(Arc::clone(&long1)).as_ref(),
                        &ctx,
                        &fc,
                        None,
                        &lr,
                    ),
                    "long1",
                );
                o.ranges(
                    &format!("{p}nofield@field"),
                    LongRangeFacetCounts::new("nofield", &fc, &lr),
                    "nofield",
                );
                if query == "all" {
                    let mut f = Out {
                        sb: &mut *o.sb,
                        index,
                        query: "fast",
                    };
                    f.ranges(
                        &format!("{p}long1@field"),
                        LongRangeFacetCounts::with_fast_match(
                            "long1",
                            Some(searcher),
                            &fc_all,
                            fast(),
                            &lr,
                        ),
                        "long1",
                    );
                    f.ranges(
                        &format!("{p}longm@field"),
                        LongRangeFacetCounts::with_fast_match(
                            "longm",
                            Some(searcher),
                            &fc_all,
                            fast(),
                            &lr,
                        ),
                        "longm",
                    );
                    f.ranges(
                        &format!("{p}long1@src"),
                        LongRangeFacetCounts::with_values_source(
                            "long1",
                            long1.as_ref(),
                            &ctx,
                            &fc_all,
                            fast(),
                            &lr,
                        ),
                        "long1",
                    );
                    f.ranges(
                        &format!("{p}longm@multi"),
                        LongRangeFacetCounts::with_multi_values_source(
                            "longm",
                            mlvs::from_long_field("longm").as_ref(),
                            &ctx,
                            &fc_all,
                            fast(),
                            &lr,
                        ),
                        "longm",
                    );
                    let mut n = Out {
                        sb: &mut *o.sb,
                        index,
                        query: "nomatch",
                    };
                    n.ranges(
                        &format!("{p}long1@src"),
                        LongRangeFacetCounts::with_values_source(
                            "long1",
                            long1.as_ref(),
                            &ctx,
                            &fc_all,
                            Some(term("zzz")),
                            &lr,
                        ),
                        "long1",
                    );
                }
            }

            // DoubleRangeFacetCounts.
            for set in ["excl", "overl"] {
                let dr = double_ranges(set);
                let p = format!("dr:{set}:");
                let src = |s: Arc<dyn values_source::DoubleValuesSource>,
                           f: &FacetsCollector<'_>,
                           field: &str,
                           fast: Option<BooleanQuery>| {
                    DoubleRangeFacetCounts::with_values_source(
                        field,
                        s.as_ref(),
                        &ctx,
                        f,
                        fast,
                        &dr,
                    )
                };
                let multi = |s: Arc<dyn mdvs::MultiDoubleValuesSource>,
                             f: &FacetsCollector<'_>,
                             field: &str,
                             fast: Option<BooleanQuery>| {
                    DoubleRangeFacetCounts::with_multi_values_source(
                        field,
                        s.as_ref(),
                        &ctx,
                        f,
                        fast,
                        &dr,
                    )
                };
                o.ranges(
                    &format!("{p}dbl@field"),
                    DoubleRangeFacetCounts::new("dbl", &fc, &dr),
                    "dbl",
                );
                o.ranges(
                    &format!("{p}dblm@field"),
                    DoubleRangeFacetCounts::new("dblm", &fc, &dr),
                    "dblm",
                );
                o.ranges(
                    &format!("{p}dbl@src"),
                    src(values_source::from_double_field("dbl"), &fc, "dbl", None),
                    "dbl",
                );
                o.ranges(
                    &format!("{p}flt@src"),
                    src(values_source::from_float_field("flt"), &fc, "flt", None),
                    "flt",
                );
                o.ranges(
                    &format!("{p}dblm@multi"),
                    multi(mdvs::from_double_field("dblm"), &fc, "dblm", None),
                    "dblm",
                );
                o.ranges(
                    &format!("{p}flt@multi"),
                    multi(mdvs::from_float_field("flt"), &fc, "flt", None),
                    "flt",
                );
                o.ranges(
                    &format!("{p}longm@todouble"),
                    multi(
                        mlvs::to_multi_double_values_source(mlvs::from_long_field("longm")),
                        &fc,
                        "longm",
                        None,
                    ),
                    "longm",
                );
                o.ranges(
                    &format!("{p}longm@dmulti"),
                    multi(mdvs::from_long_field("longm"), &fc, "longm", None),
                    "longm",
                );
                o.ranges(
                    &format!("{p}dbl@single"),
                    multi(
                        mdvs::from_single_valued(values_source::from_double_field("dbl")),
                        &fc,
                        "dbl",
                        None,
                    ),
                    "dbl",
                );
                if query == "all" {
                    let mut f = Out {
                        sb: &mut *o.sb,
                        index,
                        query: "fast",
                    };
                    f.ranges(
                        &format!("{p}dbl@field"),
                        DoubleRangeFacetCounts::with_fast_match(
                            "dbl",
                            Some(searcher),
                            &fc_all,
                            fast(),
                            &dr,
                        ),
                        "dbl",
                    );
                    f.ranges(
                        &format!("{p}dblm@field"),
                        DoubleRangeFacetCounts::with_fast_match(
                            "dblm",
                            Some(searcher),
                            &fc_all,
                            fast(),
                            &dr,
                        ),
                        "dblm",
                    );
                    f.ranges(
                        &format!("{p}dbl@src"),
                        src(
                            values_source::from_double_field("dbl"),
                            &fc_all,
                            "dbl",
                            fast(),
                        ),
                        "dbl",
                    );
                    f.ranges(
                        &format!("{p}dblm@multi"),
                        multi(mdvs::from_double_field("dblm"), &fc_all, "dblm", fast()),
                        "dblm",
                    );
                }
            }

            // Range-on-range.
            for (ty, tn) in QUERY_TYPES {
                let p = format!("ror:{tn}:");
                let l1 = vec![
                    ror_long("a", 0, true, 5, true),
                    ror_long("b", -3, false, 2, true),
                    ror_long("c", 10, true, 100, false),
                    ror_long("every", i64::MIN, true, i64::MAX, true),
                    ror_long("pt", 1, true, 1, true),
                ];
                let l2 = vec![
                    rangeonrange::LongRange::with_dims("q", vec![0, 0], vec![5, 5]).unwrap(),
                    rangeonrange::LongRange::with_dims("r", vec![3, 1], vec![4, 2]).unwrap(),
                    rangeonrange::LongRange::with_dims("s", vec![-10, -10], vec![20, 20]).unwrap(),
                ];
                let d1 = vec![
                    ror_double("a", -1.0, true, 1.0, true),
                    ror_double("b", 0.0, false, 2.5, false),
                    ror_double("c", f64::NEG_INFINITY, true, f64::INFINITY, true),
                    ror_double("d", 3.0, true, 3.0, true),
                ];
                o.ranges2(
                    &format!("{p}rorlong"),
                    LongRangeOnRangeFacetCounts::new("rorlong", &fc, ty, &l1),
                    "rorlong",
                );
                o.ranges2(
                    &format!("{p}rorlong2"),
                    LongRangeOnRangeFacetCounts::new("rorlong2", &fc, ty, &l2),
                    "rorlong2",
                );
                o.ranges2(
                    &format!("{p}rordbl"),
                    DoubleRangeOnRangeFacetCounts::new("rordbl", &fc, ty, &d1),
                    "rordbl",
                );
                o.ranges2(
                    &format!("{p}nofield"),
                    LongRangeOnRangeFacetCounts::new("nofield", &fc, ty, &l1),
                    "nofield",
                );
                if query == "all" {
                    let mut f = Out {
                        sb: &mut *o.sb,
                        index,
                        query: "fast",
                    };
                    f.ranges2(
                        &format!("{p}rorlong"),
                        LongRangeOnRangeFacetCounts::with_fast_match(
                            "rorlong",
                            Some(searcher),
                            &fc_all,
                            ty,
                            fast(),
                            &l1,
                        ),
                        "rorlong",
                    );
                    f.ranges2(
                        &format!("{p}rordbl"),
                        DoubleRangeOnRangeFacetCounts::with_fast_match(
                            "rordbl",
                            Some(searcher),
                            &fc_all,
                            ty,
                            fast(),
                            &d1,
                        ),
                        "rordbl",
                    );
                }
            }

            // Facet sets.
            for field in ["fs_long", "fs_int", "fs_dbl", "fs_flt"] {
                let ms = matchers(field);
                let refs: Vec<&dyn FacetSetMatcher> = ms.iter().map(|m| m.as_ref()).collect();
                let f = MatchingFacetSetsCounts::new(field, &fc, decoder(field), &refs).unwrap();
                o.facet_sets(&format!("fs:{field}"), &f, field);
                if query == "all" {
                    let fast_f = MatchingFacetSetsCounts::with_fast_match(
                        field,
                        Some(searcher),
                        &fc_all,
                        decoder(field),
                        fast(),
                        &refs,
                    )
                    .unwrap();
                    let mut f = Out {
                        sb: &mut *o.sb,
                        index,
                        query: "fast",
                    };
                    f.facet_sets(&format!("fs:{field}"), &fast_f, field);
                }
            }

            // DynamicRangeUtil.
            let w = values_source::long_from_long_field("w");
            for top_n in [0, 1, 3, 5, 100] {
                let r = compute_dynamic_ranges("w", w.as_ref(), long1.as_ref(), &fc, top_n, &ctx);
                writeln!(
                    out.dynamic,
                    "{index}\t{query}\tw:long1:{top_n}\t{}",
                    attempt_dynamic(r)
                )
                .unwrap();
            }
            for wf in ["wx", "wy", "nofield"] {
                let weight = values_source::long_from_long_field(wf);
                let value =
                    values_source::long_from_long_field(if wf == "wy" { "long1" } else { "w" });
                let r = compute_dynamic_ranges(wf, weight.as_ref(), value.as_ref(), &fc, 4, &ctx);
                writeln!(
                    out.dynamic,
                    "{index}\t{query}\t{wf}:4\t{}",
                    attempt_dynamic(r)
                )
                .unwrap();
            }
        }

        // Range.getQuery.
        let lq = LongRange::new("lq", 0, true, 10, true).unwrap();
        let dq = DoubleRange::new("dq", -1.0, false, 2.0, true).unwrap();
        let qs = vec![
            ("long:src", lq.get_query(None, Arc::clone(&long1))),
            ("long:fast", lq.get_query(fast(), Arc::clone(&long1))),
            (
                "long:nomatch",
                lq.get_query(Some(term("zzz")), Arc::clone(&long1)),
            ),
            (
                "long:multi",
                lq.get_query_multi(None, mlvs::from_long_field("longm")),
            ),
            (
                "long:multifast",
                lq.get_query_multi(fast(), mlvs::from_long_field("longm")),
            ),
            (
                "long:single",
                lq.get_query_multi(None, mlvs::from_single_valued(Arc::clone(&long1))),
            ),
            (
                "double:src",
                dq.get_query(None, values_source::from_double_field("dbl")),
            ),
            (
                "double:fast",
                dq.get_query(fast(), values_source::from_double_field("dbl")),
            ),
            (
                "double:multi",
                dq.get_query_multi(None, mdvs::from_double_field("dblm")),
            ),
            (
                "double:multifast",
                dq.get_query_multi(fast(), mdvs::from_double_field("dblm")),
            ),
            (
                "double:single",
                dq.get_query_multi(
                    None,
                    mdvs::from_single_valued(values_source::from_float_field("flt")),
                ),
            ),
        ];
        for (name, q) in qs {
            let shown = q.to_string();
            let bq = BooleanQuery {
                must: vec![q.into()],
                ..Default::default()
            };
            let fc = FacetsCollectorManager::new(false)
                .collect(searcher, &bq)
                .unwrap();
            writeln!(
                out.queries,
                "{index}\t{name}\t{}\t{}",
                esc(&shown),
                digest(&fc)
            )
            .unwrap();
        }
    });
}

fn open(index: &str) -> DirectoryReader {
    DirectoryReader::open(&FsDirectory::open(data().join(index))).unwrap()
}

#[test]
fn value_range_and_facet_set_counts_match_lucene() {
    let mut out = Outputs {
        facets: String::new(),
        queries: String::new(),
        dynamic: String::new(),
    };
    for index in ["multi", "single"] {
        let reader = open(index);
        run_index(index, &reader, &mut out);
    }

    // Hand-built DynamicRangeUtil inputs.
    let arrays: [(&str, Vec<i64>, Vec<i64>); 4] = [
        ("ties", vec![5, 3, 5, 1, 3, 5, 9], vec![2, 1, 1, 4, 1, 3, 2]),
        ("neg", vec![-4, 8, 0, 2], vec![3, -1, 5, 2]),
        ("wrap", vec![i64::MAX, i64::MAX, 1], vec![1, 1, 1]),
        ("one", vec![42], vec![7]),
    ];
    for (name, values, weights) in &arrays {
        for len in [0, values.len() as i64 - 1, values.len() as i64] {
            if len < 0 {
                continue;
            }
            let len = len as usize;
            for top_n in [0, 1, 2, 3, 10] {
                let (mut v, mut w) = (values.clone(), weights.clone());
                let t = w[..len].iter().fold(0i64, |a, &b| a.wrapping_add(b));
                let r = compute_dynamic_numeric_ranges(&mut v, &mut w, len, t, top_n);
                writeln!(
                    out.dynamic,
                    "arrays\t{name}\t{len}:{top_n}\t{}",
                    dyn_ranges(&r)
                )
                .unwrap();
            }
        }
    }

    assert_text_eq("facets.tsv", &out.facets, &read(&data().join("facets.tsv")));
    assert_text_eq(
        "queries.tsv",
        &out.queries,
        &read(&data().join("queries.tsv")),
    );
    assert_text_eq(
        "dynamic.tsv",
        &out.dynamic,
        &read(&data().join("dynamic.tsv")),
    );
}

/// `GenValueFacets.error`.
fn error(sb: &mut String, name: &str, r: Result<String>) {
    let r = attempt(r);
    let r = if r.starts_with("IAE:") || r.starts_with("ISE:") || r == "UOE" {
        r
    } else {
        format!("ok:{r}")
    };
    writeln!(sb, "{name}\t{r}").unwrap();
}

fn shown<T: std::fmt::Display>(r: Result<T>) -> Result<String> {
    r.map(|v| v.to_string())
}

#[test]
fn refusals_match_lucene() {
    let multi = open("multi");
    let single = open("single");
    let mut sb = String::new();
    error(
        &mut sb,
        "longrange:minexcl",
        shown(LongRange::new("x", i64::MAX, false, 5, true)),
    );
    error(
        &mut sb,
        "longrange:maxexcl",
        shown(LongRange::new("x", 0, true, i64::MIN, false)),
    );
    error(
        &mut sb,
        "longrange:empty",
        shown(LongRange::new("x", 5, true, 5, false)),
    );
    error(
        &mut sb,
        "longrange:ok",
        shown(LongRange::new("x", 5, false, 7, false)),
    );
    error(
        &mut sb,
        "doublerange:minnan",
        shown(DoubleRange::new("x", f64::NAN, true, 5.0, true)),
    );
    error(
        &mut sb,
        "doublerange:maxnan",
        shown(DoubleRange::new("x", 0.0, true, f64::NAN, true)),
    );
    error(
        &mut sb,
        "doublerange:empty",
        shown(DoubleRange::new("x", 1.0, false, 1.0, true)),
    );
    let rl = rangeonrange::LongRange::new;
    let rlw = rangeonrange::LongRange::with_dims;
    error(
        &mut sb,
        "ror:long:minexcl",
        shown(rl("x", i64::MAX, false, 5, true)),
    );
    error(
        &mut sb,
        "ror:long:maxexcl",
        shown(rl("x", 0, true, i64::MIN, false)),
    );
    error(
        &mut sb,
        "ror:long:inverted",
        shown(rl("x", 5, true, 4, true)),
    );
    error(
        &mut sb,
        "ror:long:emptydims",
        shown(rlw("x", vec![], vec![])),
    );
    error(
        &mut sb,
        "ror:long:mismatch",
        shown(rlw("x", vec![1], vec![2, 3])),
    );
    error(
        &mut sb,
        "ror:long:invertdim",
        shown(rlw("x", vec![1, 5], vec![2, 3])),
    );
    let rd = rangeonrange::DoubleRange::new;
    let rdw = rangeonrange::DoubleRange::with_dims;
    error(
        &mut sb,
        "ror:double:minnan",
        shown(rd("x", f64::NAN, true, 1.0, true)),
    );
    error(
        &mut sb,
        "ror:double:maxnan",
        shown(rd("x", 0.0, true, f64::NAN, true)),
    );
    error(
        &mut sb,
        "ror:double:inverted",
        shown(rd("x", 1.0, false, 1.0, true)),
    );
    error(
        &mut sb,
        "ror:double:emptydims",
        shown(rdw("x", vec![], vec![])),
    );
    error(
        &mut sb,
        "ror:double:mismatch",
        shown(rdw("x", vec![1.0], vec![])),
    );
    error(
        &mut sb,
        "ror:double:invertdim",
        shown(rdw("x", vec![3.0], vec![2.0])),
    );
    with_searcher(&multi, |searcher| {
        let all = FacetsCollectorManager::new(false)
            .collect(searcher, &queries(multi.max_doc())[0].1)
            .unwrap();
        let ror = |min: f64, max: f64| {
            DoubleRangeOnRangeFacetCounts::new(
                "rordbl",
                &all,
                RangeQueryType::Intersects,
                &[rdw("x", vec![min], vec![max]).unwrap()],
            )
            .map(|_| "counted".to_string())
        };
        error(&mut sb, "ror:double:nanarray", ror(f64::NAN, 1.0));
        error(&mut sb, "ror:double:nanmax", ror(1.0, f64::NAN));
        error(
            &mut sb,
            "fs:long:empty",
            shown(LongFacetSet::new(vec![]).map(|_| "")),
        );
        error(
            &mut sb,
            "fs:int:empty",
            shown(IntFacetSet::new(vec![]).map(|_| "")),
        );
        error(
            &mut sb,
            "fs:float:empty",
            shown(FloatFacetSet::new(vec![]).map(|_| "")),
        );
        error(
            &mut sb,
            "fs:double:empty",
            shown(DoubleFacetSet::new(vec![]).map(|_| "")),
        );
        error(
            &mut sb,
            "fs:field:empty",
            shown(FacetSetsField::create("f", &[]).map(|_| "")),
        );
        let l12 = LongFacetSet::new(vec![1, 2]).unwrap();
        let l1 = LongFacetSet::new(vec![1]).unwrap();
        let d1 = DoubleFacetSet::new(vec![1.0]).unwrap();
        error(
            &mut sb,
            "fs:field:dims",
            shown(FacetSetsField::create("f", &[&l12, &l1]).map(|_| "")),
        );
        error(
            &mut sb,
            "fs:field:type",
            shown(FacetSetsField::create("f", &[&l1, &d1]).map(|_| "")),
        );
        error(
            &mut sb,
            "fs:dim:minexcl",
            shown(DimRange::from_longs(i64::MAX, false, 1, true)),
        );
        error(
            &mut sb,
            "fs:dim:maxexcl",
            shown(DimRange::from_longs(1, true, i64::MIN, false)),
        );
        error(
            &mut sb,
            "fs:dim:inverted",
            shown(DimRange::from_longs(5, true, 5, false)),
        );
        error(
            &mut sb,
            "fs:dim:dminnan",
            shown(DimRange::from_doubles(f64::NAN, true, 1.0, true)),
        );
        error(
            &mut sb,
            "fs:dim:dmaxnan",
            shown(DimRange::from_doubles(1.0, true, f64::NAN, true)),
        );
        error(
            &mut sb,
            "fs:dim:dinverted",
            shown(DimRange::from_doubles(1.0, false, 1.0, true)),
        );
        error(
            &mut sb,
            "fs:dim:fminnan",
            shown(DimRange::from_floats(f32::NAN, true, 1.0, true)),
        );
        error(
            &mut sb,
            "fs:dim:fmaxnan",
            shown(DimRange::from_floats(1.0, true, f32::NAN, true)),
        );
        error(
            &mut sb,
            "fs:dim:finverted",
            shown(DimRange::from_floats(1.0, true, 1.0, false)),
        );
        error(
            &mut sb,
            "fs:dim:dvalues",
            shown(DimRange::from_doubles(-0.0, true, 0.0, false)),
        );
        error(
            &mut sb,
            "fs:dim:fvalues",
            shown(DimRange::from_floats(-1.5, false, 2.5, true)),
        );
        error(
            &mut sb,
            "fs:matcher:empty",
            shown(RangeFacetSetMatcher::new("x", &[]).map(|_| "")),
        );
        error(
            &mut sb,
            "fs:counts:nomatchers",
            shown(MatchingFacetSetsCounts::new("fs_long", &all, &decode_longs, &[]).map(|_| "")),
        );
        let a = ExactFacetSetMatcher::new("a", &l12).unwrap();
        let b = ExactFacetSetMatcher::new("b", &l1).unwrap();
        error(
            &mut sb,
            "fs:counts:dims",
            shown(
                MatchingFacetSetsCounts::new("fs_long", &all, &decode_longs, &[&a, &b]).map(|_| ""),
            ),
        );
        let state = StringDocValuesReaderState::new(&single, "str").unwrap();
        error(
            &mut sb,
            "sv:state",
            shown(StringValueFacetCounts::with_hits(&state, &all).map(|_| "")),
        );
        let other = open("multi");
        let state = StringDocValuesReaderState::new(&other, "str").unwrap();
        error(
            &mut sb,
            "sv:state:multi",
            shown(StringValueFacetCounts::with_hits(&state, &all).map(|_| "")),
        );
        error(
            &mut sb,
            "lv:wrongtype",
            shown(LongValueFacetCounts::new("str", &all).map(|_| "")),
        );
        error(
            &mut sb,
            "sv:wrongtype",
            StringDocValuesReaderState::new(&multi, "long1")
                .and_then(|s| StringValueFacetCounts::with_hits(&s, &all).map(|_| String::new())),
        );
    });
    assert_text_eq("errors.tsv", &sb, &read(&data().join("errors.tsv")));
    let _: &dyn IndexReader = &multi;
}
