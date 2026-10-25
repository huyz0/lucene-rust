//! M14.f against `fixtures/src/GenSandboxFacet.java`: lucene-sandbox's
//! facet package (`org.apache.lucene.sandbox.facet`).
//!
//! - `builders.tsv`: every builder kind through one `FacetOrchestrator`
//!   (builders of one key sharing a collection) beside a hit counter, per
//!   base query.
//! - `sideways.tsv`: `DrillSidewaysFacetOrchestrator` per drill-down.
//! - `recorders.tsv`: the cutters, recorders, iterators, comparables and
//!   labels used directly.
//! - `post.tsv`: `PostCollectionFaceting` over `FacetsCollector` results.
//! - `histograms.tsv`: `HistogramCollectorManager` per query, field, width
//!   and bucket limit (the BKD bulk path, the skip-index path, the naive
//!   ones and their errors).
//! - `errors.tsv`: the refusals.
// Test-support code opts out of the arithmetic gate at the file boundary:
// see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]
// The facet cutters and recorders are shared within one search through
// `Arc`, without `Send`/`Sync` bounds (the taxonomy cutter borrows its
// reader), as the module does.
#![allow(clippy::arc_with_non_send_sync)]

use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use lucene_search::collector::ScoreMode;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::extended_query::{ExtendedQuery, PointRangeQuery};
use lucene_search::facets::multi_double_values_source;
use lucene_search::facets::multi_long_values_source;
use lucene_search::facets::range::DoubleRange;
use lucene_search::facets::range::LongRange;
use lucene_search::facets::taxonomy::directory::DirectoryTaxonomyReader;
use lucene_search::facets::taxonomy::facet_label::FacetLabel;
use lucene_search::facets::taxonomy::taxonomy_reader::TaxonomyReader;
use lucene_search::facets::{
    DrillDownQuery, DrillSideways, FacetResult, FacetValue, FacetsCollector,
    FacetsCollectorManager, FacetsConfig,
};
use lucene_search::field_norms::FieldNorms;
use lucene_search::grouping::{search_manager, GroupingCollectorManager};
use lucene_search::index_searcher::{IndexSearcher, SegmentNorms};
use lucene_search::leaf_collector::SegmentCollector;
use lucene_search::multi_segment::OpenSegment;
use lucene_search::query::{
    BooleanQuery, BoostQuery, Clause, ConstantScoreQuery, MatchAllDocsQuery, PointsRangeQuery,
    TermQuery,
};
use lucene_search::sandbox::facet::builders::{
    DrillSidewaysFacetOrchestrator, FacetBuilder, FacetOrchestrator, RangeFacetBuilderFactory,
};
use lucene_search::sandbox::facet::comparable::{
    self, ByAggregatedValue, ByCountAndLongValue, ByLongValue, ByOrdinal,
};
use lucene_search::sandbox::facet::cutters::{LongValueFacetCutter, TaxonomyFacetsCutter};
use lucene_search::sandbox::facet::histogram::{histogram_search, HistogramCollectorManager};
use lucene_search::sandbox::facet::iterators::{
    CandidateSetOrdinalIterator, LengthOrdinalIterator, TaxonomyChildrenOrdinalIterator,
    TopnOrdinalIterator,
};
use lucene_search::sandbox::facet::labels::{
    LabelToOrd, OrdToLabel, RangeOrdToLabel, TaxonomyOrdLabelBiMap,
};
use lucene_search::sandbox::facet::post_collection::PostCollectionFaceting;
use lucene_search::sandbox::facet::recorders::{
    CountFacetRecorder, LongAggregationsFacetRecorder, MultiFacetsRecorder, Reducer,
};
use lucene_search::sandbox::facet::{
    from_array, EmptyOrdinalIterator, FacetFieldCollectorManager, FacetRecorder, OrdinalIterator,
};
use lucene_search::values_source::long_from_long_field;
use lucene_search::{values_source, Error, Result};
use lucene_store::FsDirectory;

fn data() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/sandbox_facet")
}

fn read(p: &Path) -> String {
    fs::read_to_string(p).unwrap_or_else(|e| {
        panic!(
            "{}: {e} (run scripts/gen-fixtures.sh --only GenSandboxFacet)",
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

/// `GenSandboxFacet.config()`.
fn config() -> FacetsConfig {
    let mut c = FacetsConfig::new();
    c.set_multi_valued("Author", false);
    c.set_hierarchical("Publish Date", true);
    c.set_multi_valued("Tag", true);
    c.set_multi_valued("Rating", true);
    c.set_require_dim_count("Rating", true);
    c.set_index_field_name("Color", "$color");
    c.set_hierarchical("Color", true);
    c
}

fn term(f: &str, v: &str) -> Clause {
    Clause::Term(TermQuery::new(f, v))
}

fn query(c: Clause) -> BooleanQuery {
    BooleanQuery {
        must: vec![c],
        ..Default::default()
    }
}

/// `GenSandboxFacet.bases()`.
fn bases() -> Vec<(&'static str, Clause)> {
    vec![
        ("all", Clause::MatchAllDocs(MatchAllDocsQuery::new(0))),
        ("even", term("q", "even")),
        ("rare", term("q", "rare")),
        ("none", term("q", "absent")),
    ]
}

fn long_range(label: &str, min: i64, min_inc: bool, max: i64, max_inc: bool) -> LongRange {
    LongRange::new(label, min, min_inc, max, max_inc).unwrap()
}

fn disjoint_ranges() -> Vec<LongRange> {
    vec![
        long_range("neg", -20, true, -1, true),
        long_range("low", 0, true, 10, false),
        long_range("mid", 10, true, 30, true),
        long_range("top", 40, false, i64::MAX, true),
    ]
}

fn overlapping_ranges() -> Vec<LongRange> {
    vec![
        long_range("all", i64::MIN, true, i64::MAX, true),
        long_range("a", 0, true, 20, true),
        long_range("b", 10, true, 40, true),
        long_range("c", 20, true, 20, true),
        long_range("d", -5, true, 5, false),
        long_range("e", 500, true, 900, true),
        long_range("f", 15, true, 15, true),
    ]
}

fn double_ranges() -> Vec<DoubleRange> {
    vec![
        DoubleRange::new("neg", f64::NEG_INFINITY, true, 0.0, false).unwrap(),
        DoubleRange::new("zero_two", 0.0, true, 2.0, true).unwrap(),
        DoubleRange::new("one_five", 1.0, false, 5.0, true).unwrap(),
        DoubleRange::new("big", 5.0, false, f64::INFINITY, true).unwrap(),
    ]
}

/// `GenSandboxFacet.builders(config, taxo)`.
fn builders<'t>(
    config: &'t FacetsConfig,
    taxo: &'t dyn TaxonomyReader,
) -> Vec<(&'static str, FacetBuilder<'t>)> {
    let tb = |dim: &str, path: &[&str]| FacetBuilder::taxonomy(config, taxo, dim, path).unwrap();
    let single = || multi_long_values_source::from_single_valued(long_from_long_field("single"));
    let common = Arc::new(LongValueFacetCutter::new("single"));
    vec![
        ("author", tb("Author", &[])),
        ("author_top3", tb("Author", &[]).with_top_n(3)),
        ("date", tb("Publish Date", &[])),
        ("date_2001", tb("Publish Date", &["2001"]).with_top_n(5)),
        ("tag", tb("Tag", &[])),
        ("rating", tb("Rating", &[])),
        ("color", tb("Color", &[])),
        ("color_c1", tb("Color", &["c1"])),
        ("price", FacetBuilder::long_value("price")),
        (
            "price_count_top4",
            FacetBuilder::long_value("price")
                .with_sort_by_count()
                .with_top_n(4),
        ),
        ("single", FacetBuilder::long_value("single").with_top_n(6)),
        (
            "skp_count",
            FacetBuilder::long_value("skp")
                .with_sort_by_count()
                .with_top_n(5),
        ),
        (
            "range_disjoint",
            RangeFacetBuilderFactory::for_long_ranges("price", &disjoint_ranges()),
        ),
        (
            "range_overlap",
            RangeFacetBuilderFactory::for_long_ranges("price", &overlapping_ranges()),
        ),
        (
            "range_single_disjoint",
            RangeFacetBuilderFactory::for_long_ranges_source(
                "single_dim",
                single(),
                &disjoint_ranges(),
            ),
        ),
        (
            "range_single_overlap",
            RangeFacetBuilderFactory::for_long_ranges_source(
                "single_dim",
                single(),
                &overlapping_ranges(),
            ),
        ),
        (
            "range_multi_on_single",
            RangeFacetBuilderFactory::for_long_ranges("single", &overlapping_ranges())
                .with_top_n(3),
        ),
        (
            "range_none",
            RangeFacetBuilderFactory::for_long_ranges("price", &[]),
        ),
        (
            "double_multi",
            RangeFacetBuilderFactory::for_double_ranges("weight", &double_ranges()),
        ),
        (
            "double_single",
            RangeFacetBuilderFactory::for_double_ranges_source(
                "dweight",
                multi_double_values_source::from_single_valued(values_source::from_double_field(
                    "dweight",
                )),
                &double_ranges(),
            ),
        ),
        (
            "common",
            FacetBuilder::common("custom", common.clone(), common).with_top_n(4),
        ),
    ]
}

/// `GenSandboxFacet.esc`.
fn esc(s: &str) -> String {
    let mut out = String::new();
    for c in s.encode_utf16() {
        if c < 0x20 || c == u16::from(b'\\') {
            write!(out, "\\u{c:04x}").unwrap();
        } else {
            out.push_str(&String::from_utf16_lossy(&[c]));
        }
    }
    out
}

fn value(v: FacetValue) -> String {
    match v {
        FacetValue::Int(i) => i.to_string(),
        FacetValue::Float(f) => f.to_string(),
    }
}

/// `GenSandboxFacet.render(FacetResult)`.
fn render(r: &FacetResult) -> String {
    let mut sb = format!(
        "{}|{}|{}|{}|",
        esc(&r.dim),
        esc(&r.path.join("/")),
        value(r.value),
        r.child_count
    );
    for (i, lv) in r.label_values.iter().enumerate() {
        if i > 0 {
            sb.push(',');
        }
        write!(sb, "{}={}/{}", esc(&lv.label), value(lv.value), lv.count).unwrap();
    }
    sb
}

/// `GenSandboxFacet.path(FacetLabel)`.
fn path(l: Option<&FacetLabel>) -> String {
    l.map_or_else(|| "null".to_string(), |l| esc(&l.components().join("/")))
}

fn err(e: &Error) -> String {
    match e {
        Error::IllegalArgument(m) => format!("IllegalArgumentException: {m}"),
        Error::IllegalState(m) => format!("IllegalStateException: {m}"),
        Error::Unsupported(m) => format!("UnsupportedOperationException: {m}"),
        other => format!("{other:?}"),
    }
}

/// `TotalHitCountCollectorManager`.
struct CountManager;
struct Count(i32);

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
    type Output = i32;
    fn new_collector(&self) -> Result<Count> {
        Ok(Count(0))
    }
    fn reduce(&self, collectors: Vec<Count>) -> Result<i32> {
        Ok(collectors.iter().map(|c| c.0).sum())
    }
}

fn ords_of(it: &mut dyn OrdinalIterator) -> Vec<i32> {
    it.to_array().unwrap()
}

/// `GenSandboxFacet.recorders`.
fn recorders(
    q: &str,
    query: &BooleanQuery,
    searcher: &IndexSearcher<'_, '_>,
    config: &FacetsConfig,
    taxo: &dyn TaxonomyReader,
    out: &mut String,
) {
    let cutter = Arc::new(TaxonomyFacetsCutter::new("$facets", config, taxo));
    let count = Arc::new(CountFacetRecorder::new());
    let agg = Arc::new(
        LongAggregationsFacetRecorder::new(
            vec![
                long_from_long_field("single"),
                long_from_long_field("single"),
                long_from_long_field("skp"),
            ],
            vec![Reducer::Sum, Reducer::Max, Reducer::Sum],
        )
        .unwrap(),
    );
    let multi = Arc::new(MultiFacetsRecorder::new(vec![count.clone(), agg.clone()]));
    search_manager(
        searcher,
        query,
        &FacetFieldCollectorManager::new(cutter, multi),
    )
    .unwrap();
    let labels = TaxonomyOrdLabelBiMap::new(taxo);
    writeln!(out, "{q}\ttaxo_empty\t{}", count.is_empty().unwrap()).unwrap();
    let mut all = ords_of(&mut *count.recorded_ords().unwrap());
    all.sort_unstable();
    let mut sb = String::new();
    for &ord in &all {
        write!(
            sb,
            "{}={}/{}/{}/{},",
            path(labels.get_label(ord).unwrap().as_ref()),
            count.get_count(ord).unwrap(),
            agg.get_recorded_value(ord, 0).unwrap(),
            agg.get_recorded_value(ord, 1).unwrap(),
            agg.get_recorded_value(ord, 2).unwrap()
        )
        .unwrap();
    }
    writeln!(out, "{q}\ttaxo_all\t{sb}").unwrap();
    let arrays = taxo.parallel_taxonomy_arrays().unwrap();
    let label_list = |ords: &[i32]| -> String {
        labels
            .get_labels(ords)
            .unwrap()
            .iter()
            .map(|l| format!("{},", path(l.as_ref())))
            .collect()
    };
    for dim in ["Author", "Tag", "Publish Date", "Rating"] {
        let dim_ord = labels.get_ord(&FacetLabel::new([dim]).unwrap()).unwrap();
        for agid in 0..3 {
            let children = TaxonomyChildrenOrdinalIterator::new(
                count.recorded_ords().unwrap(),
                arrays.clone(),
                dim_ord,
            );
            let top =
                TopnOrdinalIterator::new(children, ByAggregatedValue::new(&count, &agg, agid), 4)
                    .unwrap()
                    .to_array()
                    .unwrap();
            writeln!(out, "{q}\ttop_agg{agid}:{dim}\t{}", label_list(&top)).unwrap();
        }
        let mut children = TaxonomyChildrenOrdinalIterator::new(
            count.recorded_ords().unwrap(),
            arrays.clone(),
            dim_ord,
        )
        .to_array()
        .unwrap();
        comparable::sort(&mut children, &ByOrdinal).unwrap();
        writeln!(out, "{q}\tby_ordinal:{dim}\t{}", label_list(&children)).unwrap();
    }
    let candidates = [
        FacetLabel::new(["Author", "a1"]).unwrap(),
        FacetLabel::new(["Author", "zz"]).unwrap(),
        FacetLabel::new(["Tag", "t3"]).unwrap(),
        FacetLabel::new(["Author", "a1"]).unwrap(),
        FacetLabel::new(["Nope"]).unwrap(),
    ];
    let cand = CandidateSetOrdinalIterator::new(count.as_ref(), &candidates, &labels)
        .unwrap()
        .to_array()
        .unwrap();
    let mut cs = String::new();
    for &ord in &cand {
        write!(
            cs,
            "{}={},",
            path(labels.get_label(ord).unwrap().as_ref()),
            count.get_count(ord).unwrap()
        )
        .unwrap();
    }
    writeln!(out, "{q}\tcandidates\t{cs}").unwrap();

    // No rollup.
    let no_rollup = Arc::new(TaxonomyFacetsCutter::with_rollup(
        "$facets", config, taxo, true,
    ));
    let plain = Arc::new(CountFacetRecorder::new());
    search_manager(
        searcher,
        query,
        &FacetFieldCollectorManager::new(no_rollup, plain.clone()),
    )
    .unwrap();
    let mut nr = String::new();
    let empty = plain.is_empty().unwrap();
    for dim in ["Author", "Publish Date", "Rating", "Tag"] {
        let dim_ord = labels.get_ord(&FacetLabel::new([dim]).unwrap()).unwrap();
        let c = if empty {
            -1
        } else {
            plain.get_count(dim_ord).unwrap()
        };
        write!(nr, "{dim}={c},").unwrap();
    }
    let y2001 = labels
        .get_ord(&FacetLabel::new(["Publish Date", "2001"]).unwrap())
        .unwrap();
    let c = if empty {
        -1
    } else {
        plain.get_count(y2001).unwrap()
    };
    write!(nr, "2001={c}").unwrap();
    writeln!(out, "{q}\tno_rollup\t{nr}").unwrap();

    // Long values.
    let lv = Arc::new(LongValueFacetCutter::new("price"));
    let lv_count = Arc::new(CountFacetRecorder::new());
    let lv_agg = Arc::new(
        LongAggregationsFacetRecorder::new(vec![long_from_long_field("pt")], vec![Reducer::Max])
            .unwrap(),
    );
    let lv_multi = Arc::new(MultiFacetsRecorder::new(vec![
        lv_count.clone(),
        lv_agg.clone(),
    ]));
    search_manager(
        searcher,
        query,
        &FacetFieldCollectorManager::new(lv.clone(), lv_multi),
    )
    .unwrap();
    let mut lv_ords = ords_of(&mut *lv_count.recorded_ords().unwrap());
    lv_ords.sort_unstable();
    let mut lvs = String::new();
    for &ord in &lv_ords {
        write!(
            lvs,
            "{ord}:{}={}/{},",
            lv.get_value(ord).unwrap(),
            lv_count.get_count(ord).unwrap(),
            lv_agg.get_recorded_value(ord, 0).unwrap()
        )
        .unwrap();
    }
    writeln!(out, "{q}\tlong_ords\t{lvs}").unwrap();
    let mut by_count = ords_of(&mut *lv_count.recorded_ords().unwrap());
    comparable::sort(&mut by_count, &ByCountAndLongValue::new(&lv_count, &lv)).unwrap();
    let bc: String = lv
        .get_labels(&by_count)
        .unwrap()
        .iter()
        .map(|l| format!("{},", path(l.as_ref())))
        .collect();
    writeln!(out, "{q}\tlong_by_count_value\t{bc}").unwrap();
    let mut by_value = ords_of(&mut *lv_count.recorded_ords().unwrap());
    comparable::sort(&mut by_value, &ByLongValue::new(&lv)).unwrap();
    let bv: String = by_value
        .iter()
        .map(|&o| format!("{},", lv.get_value(o).unwrap()))
        .collect();
    writeln!(out, "{q}\tlong_by_value\t{bv}").unwrap();

    let rl = RangeOrdToLabel::for_long_ranges(&disjoint_ranges());
    let rls: String = rl
        .get_labels(&[0, 3, -1, 4])
        .unwrap()
        .iter()
        .map(|l| format!("{},", path(l.as_ref())))
        .collect();
    writeln!(out, "{q}\trange_labels\t{rls}").unwrap();
}

/// `GenSandboxFacet.counts(rec, labels)`.
fn counts(rec: &CountFacetRecorder, labels: &TaxonomyOrdLabelBiMap<'_>) -> String {
    if rec.is_empty().unwrap() {
        return "empty".into();
    }
    let mut ords = ords_of(&mut *rec.recorded_ords().unwrap());
    ords.sort_unstable();
    ords.iter()
        .map(|&o| {
            format!(
                "{}={},",
                path(labels.get_label(o).unwrap().as_ref()),
                rec.get_count(o).unwrap()
            )
        })
        .collect()
}

/// `GenSandboxFacet.post`.
fn post(
    q: &str,
    base: &Clause,
    searcher: &IndexSearcher<'_, '_>,
    config: &FacetsConfig,
    taxo: &dyn TaxonomyReader,
    out: &mut String,
) {
    let labels = TaxonomyOrdLabelBiMap::new(taxo);
    let cutter = || Arc::new(TaxonomyFacetsCutter::new("$facets", config, taxo));
    let fc: FacetsCollector<'_> = search_manager(
        searcher,
        &query(base.clone()),
        &FacetsCollectorManager::new(false),
    )
    .unwrap();
    let down = Arc::new(CountFacetRecorder::new());
    let manager = FacetFieldCollectorManager::new(cutter(), down);
    let p = PostCollectionFaceting::<_, FacetFieldCollectorManager<'_, CountFacetRecorder>>::new(
        searcher,
        Some(&manager),
        &[],
        &fc,
        &[],
    );
    let r = p.collect().unwrap();
    writeln!(
        out,
        "{q}\tdown_only\t{}",
        counts(r.drill_down_result.as_ref().unwrap(), &labels)
    )
    .unwrap();

    let mut ddq = DrillDownQuery::with_base(config, base.clone());
    ddq.add("Author", &["a3"]).unwrap();
    ddq.add("Tag", &["t1"]).unwrap();
    let ds = DrillSideways::new(searcher, config, taxo);
    let dsr = ds
        .search_managers(
            &ddq,
            &FacetsCollectorManager::new(false),
            Some(&[
                FacetsCollectorManager::new(false),
                FacetsCollectorManager::new(false),
            ]),
        )
        .unwrap();
    let sideways_docs = dsr.drill_sideways_results.unwrap();
    let managers: Vec<(&str, FacetFieldCollectorManager<'_, CountFacetRecorder>)> =
        ["Author", "Tag", "Unused"]
            .into_iter()
            .map(|d| {
                (
                    d,
                    FacetFieldCollectorManager::new(cutter(), Arc::new(CountFacetRecorder::new())),
                )
            })
            .collect();
    let manager_refs: Vec<(&str, &FacetFieldCollectorManager<'_, CountFacetRecorder>)> =
        managers.iter().map(|(d, m)| (*d, m)).collect();
    let docs = [("Author", &sideways_docs[0]), ("Tag", &sideways_docs[1])];
    let dd = FacetFieldCollectorManager::new(cutter(), Arc::new(CountFacetRecorder::new()));
    let ps = PostCollectionFaceting::new(
        searcher,
        Some(&dd),
        &manager_refs,
        &dsr.drill_down_result,
        &docs,
    );
    let rs = ps.collect().unwrap();
    writeln!(
        out,
        "{q}\tsideways_down\t{}",
        counts(rs.drill_down_result.as_ref().unwrap(), &labels)
    )
    .unwrap();
    let mut dims: Vec<&String> = rs.drill_sideways_results.keys().collect();
    dims.sort();
    for dim in &dims {
        writeln!(
            out,
            "{q}\tsideways:{dim}\t{}",
            counts(&rs.drill_sideways_results[*dim], &labels)
        )
        .unwrap();
    }
    let dim_list: Vec<&str> = dims.iter().map(|d| d.as_str()).collect();
    writeln!(out, "{q}\tsideways_dims\t[{}]", dim_list.join(", ")).unwrap();

    let only = FacetFieldCollectorManager::new(cutter(), Arc::new(CountFacetRecorder::new()));
    let pn = PostCollectionFaceting::<FacetFieldCollectorManager<'_, CountFacetRecorder>, _>::new(
        searcher,
        None,
        &[("Author", &only)],
        &dsr.drill_down_result,
        &docs,
    );
    let rn = pn.collect().unwrap();
    writeln!(
        out,
        "{q}\tsideways_only\tnull {}",
        counts(&rn.drill_sideways_results["Author"], &labels)
    )
    .unwrap();
}

/// `GenSandboxFacet.histograms`.
fn histograms(searcher: &IndexSearcher<'_, '_>, out: &mut String, errors: &mut String) {
    let pr = |f: &str, a: i64, b: i64| Clause::PointsRange(PointsRangeQuery::new(f, a, b));
    // `IntPoint.newRangeQuery`: a 4-byte packed range.
    let ipr = |f: &str, a: i32, b: i32| {
        let enc = |v: i32| {
            let mut out = vec![0u8; 4];
            lucene_util::numeric_utils::int_to_sortable_bytes(v, &mut out, 0);
            out
        };
        Clause::Extended(Box::new(ExtendedQuery::PointRange(
            PointRangeQuery::new(f, 1, enc(a), enc(b)).unwrap(),
        )))
    };
    let queries = [
        ("all", Clause::MatchAllDocs(MatchAllDocsQuery::new(0))),
        ("even", term("q", "even")),
        ("rare", term("q", "rare")),
        ("pt_range", pr("pt", 100, 700)),
        (
            "pt_range_wrapped",
            Clause::ConstantScore(Box::new(ConstantScoreQuery::new(
                Clause::Boost(Box::new(BoostQuery::new(pr("pt", -50, 333), 2.0))),
                1.0,
            ))),
        ),
        ("pt_range_empty", pr("pt", 5000, 6000)),
        ("ipt_range", ipr("ipt", -100, 150)),
        (
            "pt_range_should",
            Clause::Boolean(Box::new(BooleanQuery {
                should: vec![pr("pt", 20, 900)],
                ..Default::default()
            })),
        ),
    ];
    let specs: [(&str, i64, i32); 18] = [
        ("pt", 2, 1024),
        ("pt", 7, 1024),
        ("pt", 100, 1024),
        ("pt", 500, 1024),
        ("pt", 1000, 1024),
        ("pt", 100_000, 1024),
        ("pt", 3, 100),
        ("ipt", 50, 1024),
        ("ipt", 300, 1024),
        ("ipt", 1000, 1024),
        ("skp", 10, 1024),
        ("skp", 1000, 1024),
        ("skp", 2, 1024),
        ("price", 5, 1024),
        ("price", 2, 20),
        ("single", 64, 1024),
        ("missing", 10, 1024),
        ("kw", 10, 1024),
    ];
    for (qn, q) in &queries {
        let q = query(q.clone());
        for &(field, width, max) in &specs {
            write!(out, "{qn}\t{field}\t{width}\t{max}\t").unwrap();
            let m = HistogramCollectorManager::with_max_buckets(field, width, max).unwrap();
            match histogram_search(searcher, &q, &m) {
                Ok(buckets) => {
                    for (k, v) in buckets {
                        write!(out, "{k}={v},").unwrap();
                    }
                }
                Err(e) => write!(out, "error {}", err(&e)).unwrap(),
            }
            out.push('\n');
        }
    }
    for (field, width, max) in [("pt", 1i64, 10i32), ("pt", 10, 0)] {
        match HistogramCollectorManager::with_max_buckets(field, width, max) {
            Ok(_) => writeln!(errors, "histogram {width} {max}\tnone").unwrap(),
            Err(e) => writeln!(errors, "histogram {width} {max}\t{}", err(&e)).unwrap(),
        }
    }
}

/// `GenSandboxFacet.errors`.
fn misc_errors(config: &FacetsConfig, taxo: &dyn TaxonomyReader, out: &mut String) {
    match FacetBuilder::taxonomy(config, taxo, "Unconfigured", &[] as &[&str]) {
        Ok(_) => writeln!(out, "unconfigured_dim\tnone").unwrap(),
        Err(e) => writeln!(out, "unconfigured_dim\t{}", err(&e)).unwrap(),
    }
    match TopnOrdinalIterator::new(EmptyOrdinalIterator, ByOrdinal, 0) {
        Ok(_) => writeln!(out, "topn_zero\tnone").unwrap(),
        Err(e) => writeln!(out, "topn_zero\t{}", err(&e)).unwrap(),
    }
    let mut li = LengthOrdinalIterator::new(from_array(vec![3, -1, 5, -1]));
    match li.length() {
        Ok(_) => writeln!(out, "length_early\tnone").unwrap(),
        Err(e) => writeln!(out, "length_early\t{}", err(&e)).unwrap(),
    }
    let arr = li.to_array().unwrap();
    let joined: Vec<String> = arr.iter().map(i32::to_string).collect();
    writeln!(
        out,
        "from_array\t{} {}",
        joined.join(","),
        li.length().unwrap()
    )
    .unwrap();
    let agg = LongAggregationsFacetRecorder::new(
        vec![long_from_long_field("single")],
        vec![Reducer::Sum],
    )
    .unwrap();
    for idx in [-1, 1] {
        match agg.get_recorded_value(0, idx) {
            Ok(_) => writeln!(out, "agg_value_id {idx}\tnone").unwrap(),
            Err(e) => writeln!(out, "agg_value_id {idx}\t{}", err(&e)).unwrap(),
        }
    }
    let multi = MultiFacetsRecorder::new(Vec::new());
    if let Err(e) = multi.recorded_ords() {
        writeln!(out, "multi_recorded_ords\t{}", err(&e)).unwrap();
    }
    if let Err(e) = multi.is_empty() {
        writeln!(out, "multi_is_empty\t{}", err(&e)).unwrap();
    }
    if let Err(e) = multi.contains(1) {
        writeln!(out, "multi_contains\t{}", err(&e)).unwrap();
    }
    for (a, b) in [(i64::MAX, 1i64), (i64::from(i32::MAX), 1)] {
        write!(out, "sum {a}\t").unwrap();
        let java_err = |e: Error| match e {
            Error::IllegalArgument(m) => format!("ArithmeticException: {m}"),
            other => panic!("{other:?}"),
        };
        match Reducer::Sum.reduce_i64(a, b) {
            Ok(l) => {
                write!(out, "{l} ").unwrap();
                match Reducer::Sum.reduce_i32(a as i32, b as i32) {
                    Ok(i) => writeln!(out, "{i}").unwrap(),
                    Err(e) => writeln!(out, "{}", java_err(e)).unwrap(),
                }
            }
            Err(e) => writeln!(out, "{}", java_err(e)).unwrap(),
        }
    }
    writeln!(
        out,
        "max\t{} {:?} {:?} {:?}",
        Reducer::Max.reduce_i64(-3, 7).unwrap(),
        Reducer::Max.reduce_f64(2.5, -1.0),
        Reducer::Max.reduce_f32(f32::NAN, 1.0),
        Reducer::Sum.reduce_f64(0.1, 0.2)
    )
    .unwrap();
}

#[test]
fn sandbox_facets_match_lucene() {
    let reader = DirectoryReader::open(&FsDirectory::open(data().join("index"))).unwrap();
    let mut opened = reader.open_segments().unwrap();
    opened.open_points().unwrap();
    let segments = opened.as_open_segments();
    let owned = reader
        .field_norms_by_field(&["q", "$facets", "$color"].map(String::from))
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

    let mut builders_out = String::new();
    let mut recorders_out = String::new();
    let mut post_out = String::new();
    for (bn, base) in bases() {
        let bs = builders(&config, &taxo);
        let mut orchestrator = FacetOrchestrator::new();
        for (_, b) in &bs {
            orchestrator = orchestrator.add_builder(b);
        }
        let hits = orchestrator
            .collect_with(&query(base.clone()), &searcher, Some(&CountManager))
            .unwrap()
            .unwrap();
        writeln!(builders_out, "{bn}\t#hits\t{hits}").unwrap();
        for (name, b) in &bs {
            writeln!(
                builders_out,
                "{bn}\t{name}\t{}",
                render(&b.get_result().unwrap())
            )
            .unwrap();
        }
        recorders(
            bn,
            &query(base.clone()),
            &searcher,
            &config,
            &taxo,
            &mut recorders_out,
        );
        post(bn, &base, &searcher, &config, &taxo, &mut post_out);
    }
    assert_text_eq(
        "builders.tsv",
        &builders_out,
        &read(&data().join("builders.tsv")),
    );
    assert_text_eq(
        "recorders.tsv",
        &recorders_out,
        &read(&data().join("recorders.tsv")),
    );
    assert_text_eq("post.tsv", &post_out, &read(&data().join("post.tsv")));

    // Drill sideways.
    let drills: [(&str, &[(&str, &str)]); 5] = [
        ("none", &[]),
        ("author", &[("Author", "a3")]),
        ("author_tag", &[("Author", "a3"), ("Tag", "t1")]),
        ("tag_author", &[("Tag", "t2"), ("Author", "a5")]),
        ("date", &[("Publish Date", "2001")]),
    ];
    let mut sideways_out = String::new();
    let ds = DrillSideways::new(&searcher, &config, &taxo);
    for (dn, steps) in drills {
        for (bn, base) in bases().into_iter().take(2) {
            let mut ddq = DrillDownQuery::with_base(&config, base);
            for (dim, v) in steps {
                ddq.add(dim, &[*v]).unwrap();
            }
            let mut named: Vec<(String, FacetBuilder<'_>)> = Vec::new();
            for (dim, _) in steps {
                named.push((
                    format!("sideways:{dim}"),
                    FacetBuilder::taxonomy(&config, &taxo, dim, &[] as &[&str]).unwrap(),
                ));
            }
            let n_sideways = named.len();
            named.push((
                "down:date".into(),
                FacetBuilder::taxonomy(&config, &taxo, "Publish Date", &[] as &[&str])
                    .unwrap()
                    .with_top_n(2),
            ));
            named.push((
                "down:price".into(),
                FacetBuilder::long_value("price").with_top_n(3),
            ));
            named.push((
                "down:range".into(),
                RangeFacetBuilderFactory::for_long_ranges("price", &disjoint_ranges()),
            ));
            let mut o = DrillSidewaysFacetOrchestrator::new();
            for ((_, b), (dim, _)) in named.iter().zip(steps.iter()) {
                o = o.add_drill_sideways_builder(dim, b);
            }
            for (_, b) in &named[n_sideways..] {
                o = o.add_drill_down_builder(b);
            }
            let key = format!("{dn}/{bn}");
            let hits = o
                .collect_with(&ddq, &ds, Some(&CountManager))
                .unwrap()
                .unwrap();
            writeln!(sideways_out, "{key}\t#hits\t{hits}").unwrap();
            for (name, b) in &named {
                writeln!(
                    sideways_out,
                    "{key}\t{name}\t{}",
                    render(&b.get_result().unwrap())
                )
                .unwrap();
            }
        }
    }
    assert_text_eq(
        "sideways.tsv",
        &sideways_out,
        &read(&data().join("sideways.tsv")),
    );

    let mut errors_out = String::new();
    for (case, extra) in [
        ("sideways_extra_dim", true),
        ("sideways_two_managers", false),
    ] {
        let mut ddq = DrillDownQuery::new(&config);
        ddq.add("Author", &["a3"]).unwrap();
        let author = FacetBuilder::taxonomy(&config, &taxo, "Author", &[] as &[&str]).unwrap();
        let tag = FacetBuilder::taxonomy(&config, &taxo, "Tag", &[] as &[&str]).unwrap();
        let price = FacetBuilder::long_value("price");
        let mut o =
            DrillSidewaysFacetOrchestrator::new().add_drill_sideways_builder("Author", &author);
        o = if extra {
            o.add_drill_sideways_builder("Tag", &tag)
        } else {
            o.add_drill_sideways_builder("Author", &price)
        };
        match o.collect(&ddq, &ds) {
            Ok(()) => writeln!(errors_out, "{case}\tnone").unwrap(),
            Err(e) => writeln!(errors_out, "{case}\t{}", err(&e)).unwrap(),
        }
    }
    let mut hist_out = String::new();
    histograms(&searcher, &mut hist_out, &mut errors_out);
    misc_errors(&config, &taxo, &mut errors_out);
    assert_text_eq(
        "histograms.tsv",
        &hist_out,
        &read(&data().join("histograms.tsv")),
    );
    assert_text_eq("errors.tsv", &errors_out, &read(&data().join("errors.tsv")));
}
