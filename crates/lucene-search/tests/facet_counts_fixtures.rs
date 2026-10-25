//! M13 T13.4b against `fixtures/src/GenFacetCounts.java`: taxonomy facet
//! counting.
//!
//! - Every `Facets` call `GenFacetCounts.probe` makes -- over
//!   `FastTaxonomyFacetCounts` (a collector's hits and every live document),
//!   `TaxonomyFacetIntAssociations`, `TaxonomyFacetFloatAssociations` (the
//!   indexed values, a `DoubleValuesSource`, the kept scores) and
//!   `MultiFacets`, each for a match-all, a half, a sparse and an empty
//!   query -- renders here exactly as Lucene's did (`calls.tsv`).
//! - `FacetsCollectorManager.search`/`searchAfter` (`search.tsv`),
//!   `TaxonomyFacetLabels` (`labels.tsv`), `FacetsConfig.build` of
//!   association fields (`build.tsv`) and the refusals (`errors.tsv`).
// Test-support code opts out of the arithmetic gate at the file boundary:
// see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use lucene_index::document::{
    DocValuesType, Document, Number, NumericDocValuesField, Store, StringField,
};
use lucene_search::collector::{ScoreDoc, TotalHitsRelation};
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::facets::taxonomy::directory::{
    DirectoryTaxonomyReader, DirectoryTaxonomyWriter,
};
use lucene_search::facets::taxonomy::{
    AssociationAggregationFunction, AssociationFacetField, FacetLabel, FastTaxonomyFacetCounts,
    TaxonomyFacetFloatAssociations, TaxonomyFacetIntAssociations, TaxonomyFacetLabels,
    TaxonomyReader, TaxonomyWriter,
};
use lucene_search::facets::{
    FacetEntry, FacetField, FacetResult, FacetValue, Facets, FacetsCollector,
    FacetsCollectorManager, FacetsConfig, MultiFacets,
};
use lucene_search::field_norms::FieldNorms;
use lucene_search::index_searcher::{IndexSearcher, SegmentNorms};
use lucene_search::query::{BooleanQuery, Clause, MatchAllDocsQuery, TermQuery};
use lucene_search::reader::IndexReader;
use lucene_search::top_docs::TopDocs;
use lucene_search::values_source::{self, ValuesContext};
use lucene_search::{Error, Result};
use lucene_store::{ByteBuffersDirectory, FsDirectory};

fn data() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/facet_counts")
}

fn read(p: &Path) -> String {
    fs::read_to_string(p).unwrap_or_else(|e| {
        panic!(
            "{}: {e} (run scripts/gen-fixtures.sh --only GenFacetCounts)",
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

/// `GenFacetCounts.config()`.
fn config() -> FacetsConfig {
    let mut c = FacetsConfig::new();
    c.set_hierarchical("Publish Date", true);
    c.set_multi_valued("Tag", true);
    c.set_multi_valued("Cat", true);
    c.set_require_dim_count("Cat", true);
    c.set_hierarchical("Path", true);
    c.set_multi_valued("Path", true);
    c.set_require_dim_count("Path", true);
    c.set_hierarchical("Loc", true);
    c.set_multi_valued("Loc", true);
    c.set_index_field_name("Color", "$color");
    c.set_hierarchical("Color", true);
    c.set_index_field_name("IntA", "$int_assoc");
    c.set_multi_valued("IntA", true);
    c.set_index_field_name("IntB", "$int_assoc");
    c.set_index_field_name("FloatA", "$float_assoc");
    c.set_multi_valued("FloatA", true);
    c.set_index_field_name("FloatB", "$float_assoc");
    c
}

fn ff(dim: &str, path: &[&str]) -> FacetEntry {
    FacetField::new(dim, path).unwrap().into()
}

/// `GenFacetCounts.doc(i)`, split into its facet fields and the rest.
fn doc(i: i32) -> (Vec<FacetEntry>, Document) {
    let mut d = Document::new();
    d.add(StringField::new("q", "all", Store::No));
    if i % 2 == 0 {
        d.add(StringField::new("q", "even", Store::No));
    }
    if i % 37 == 5 {
        d.add(StringField::new("q", "rare", Store::No));
    }
    if i % 11 == 3 {
        d.add(StringField::new("del", "y", Store::No));
    }
    if i % 10 != 0 {
        d.add(NumericDocValuesField::new("num", i64::from(i % 17)));
    }
    let f = vec![
        ff("Author", &[&format!("a{}", i * 7 % 23)]),
        ff(
            "Publish Date",
            &[
                &(2000 + i % 5).to_string(),
                &(1 + i % 12).to_string(),
                &(1 + i % 28).to_string(),
            ],
        ),
        ff("Tag", &[&format!("t{}", i % 9)]),
        ff("Tag", &[&format!("t{}", i % 7 + 2)]),
    ];
    let mut f = f;
    if i % 3 != 0 {
        f.push(ff("Cat", &[&format!("c{}", i % 4)]));
        f.push(ff("Cat", &[&format!("c{}", i % 6)]));
    }
    f.push(ff(
        "Path",
        &[&format!("p{}", i % 3), &format!("q{}", i % 5)],
    ));
    f.push(ff("Path", &[&format!("p{}", i % 2)]));
    if i % 4 != 1 {
        f.push(ff("Loc", &[&format!("l{}", i % 4), &format!("m{}", i % 3)]));
    }
    f.push(ff("Many", &[&format!("m{}", i % 1100)]));
    f.push(ff(
        "Color",
        &[&format!("c{}", i % 5), &format!("s{}", i % 3)],
    ));
    let int = |v: i32, dim: &str, label: String| -> FacetEntry {
        AssociationFacetField::int(v, dim, &[label]).unwrap().into()
    };
    let float = |v: f32, dim: &str, label: String| -> FacetEntry {
        AssociationFacetField::float(v, dim, &[label])
            .unwrap()
            .into()
    };
    f.push(int(i % 10 - 3, "IntA", format!("x{}", i % 6)));
    f.push(int((i * 3) % 10 - 4, "IntA", format!("x{}", (i + 1) % 6)));
    f.push(int(i % 7 + 1, "IntB", format!("z{}", i % 4)));
    f.push(float(
        (i % 13) as f32 * 0.37f32 - 1.1f32,
        "FloatA",
        format!("y{}", i % 5),
    ));
    f.push(float(
        i as f32 * 0.013f32,
        "FloatA",
        format!("y{}", (i + 2) % 5),
    ));
    f.push(float(
        i as f32 * 0.013f32 + 0.5f32,
        "FloatB",
        format!("w{}", i % 3),
    ));
    (f, d)
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// `GenFacetCounts.dumpBuilt`.
fn dump_built(sb: &mut String, i: i32, built: &Document) {
    for f in built.fields() {
        let ft = f.field_type();
        let (kind, value) = match ft.doc_values_type() {
            DocValuesType::SortedNumeric | DocValuesType::Numeric => (
                if ft.doc_values_type() == DocValuesType::Numeric {
                    "ndv"
                } else {
                    "sndv"
                },
                match f.numeric_value() {
                    Some(Number::Long(v)) => v.to_string(),
                    other => panic!("{other:?}"),
                },
            ),
            DocValuesType::Binary => ("bdv", hex(&f.binary_value().unwrap())),
            DocValuesType::None => ("string", f.string_value().unwrap().into_owned()),
            other => panic!("{other:?}"),
        };
        writeln!(sb, "{i}\t{}\t{kind}\t{value}", f.name()).unwrap();
    }
}

#[test]
fn facets_config_build_of_associations_matches_lucene() {
    let config = config();
    let d = ByteBuffersDirectory::new();
    let w = DirectoryTaxonomyWriter::open(&d).unwrap();
    let mut sb = String::new();
    for i in 0..3 {
        let (facets, doc) = doc(i);
        let built = config.build(Some(&w), &facets, doc).unwrap();
        dump_built(&mut sb, i, &built);
    }
    assert_text_eq("build.tsv", &sb, &read(&data().join("build.tsv")));
}

// ---------------------------------------------------------------------------
// Rendering, as GenFacetCounts renders
// ---------------------------------------------------------------------------

/// `GenFacetCounts.num`.
fn num(v: FacetValue) -> String {
    match v {
        FacetValue::Int(i) => format!("i{i}"),
        FacetValue::Float(f) => format!("f{:x}", f.to_bits()),
    }
}

/// `GenFacetCounts.res`.
fn res(r: Option<FacetResult>, sorted: bool) -> String {
    let Some(r) = r else {
        return "null".into();
    };
    let mut lv = r.label_values.clone();
    if sorted {
        lv.sort_by(|a, b| a.label.cmp(&b.label));
    }
    let children: Vec<String> = lv
        .iter()
        .map(|l| format!("{}={}={}", l.label, num(l.value), l.count))
        .collect();
    format!(
        "{}\t{}\t{}\t{}\t{}\t{}",
        if sorted { "ress" } else { "res" },
        r.dim,
        r.path.join("|"),
        num(r.value),
        r.child_count,
        children.join(",")
    )
}

/// `GenFacetCounts.list`.
fn list(l: Vec<FacetResult>) -> String {
    let mut sb = format!("list\t{}", l.len());
    for r in l {
        write!(sb, "\t||\t{}", res(Some(r), false)).unwrap();
    }
    sb
}

/// `GenFacetCounts.line`: a call's outcome, an `IllegalArgumentException`
/// as `err`.
fn line(sb: &mut String, call: &str, outcome: Result<String>) {
    let shown = match outcome {
        Ok(s) => s,
        Err(Error::IllegalArgument(m)) => format!("err\t{m}"),
        Err(other) => panic!("{call}: {other:?}"),
    };
    writeln!(sb, "{call}\t{shown}").unwrap();
}

const DEFAULT_DIMS: &[&str] = &[
    "Author",
    "Publish Date",
    "Tag",
    "Cat",
    "Path",
    "Loc",
    "Many",
    "Color",
    "IntA",
    "Nope",
];
const COLOR_DIMS: &[&str] = &["Color", "Author", "Nope"];
const INT_DIMS: &[&str] = &["IntA", "IntB", "Author", "FloatA", "Nope"];
const FLOAT_DIMS: &[&str] = &["FloatA", "FloatB", "IntB", "Nope"];

/// `GenFacetCounts.paths(dim)`.
fn paths(dim: &str) -> Vec<Vec<&'static str>> {
    match dim {
        "Author" => vec![vec!["a3"], vec!["a99"]],
        "Publish Date" => vec![
            vec!["2001"],
            vec!["2001", "3"],
            vec!["2001", "3", "4"],
            vec!["1900"],
        ],
        "Tag" => vec![vec!["t3"]],
        "Cat" => vec![vec!["c1"]],
        "Path" => vec![vec!["p1"], vec!["p0"]],
        "Loc" => vec![vec!["l2"], vec!["l1"]],
        "Many" => vec![vec!["m5"]],
        "Color" => vec![vec!["c1"], vec!["c9"]],
        "IntA" => vec![vec!["x1"]],
        "IntB" => vec![vec!["z2"]],
        "FloatA" => vec![vec!["y1"]],
        "FloatB" => vec![vec!["w0"]],
        _ => vec![vec!["x"]],
    }
}

/// `GenFacetCounts.probe`.
fn probe(sb: &mut String, name: &str, qn: &str, f: &dyn Facets, sparse: bool) {
    let dims = if name.starts_with("color") || name == "count_all_color" {
        COLOR_DIMS
    } else if name.starts_with("int") {
        INT_DIMS
    } else if name.starts_with("float") {
        FLOAT_DIMS
    } else {
        DEFAULT_DIMS
    };
    let pre = format!("{name}\t{qn}\t");
    for &dim in dims {
        for n in [0, 1, 3, 50] {
            line(
                sb,
                &format!("{pre}top\t{n}\t0\t{dim}"),
                f.get_top_children(n, dim, &[]).map(|r| res(r, false)),
            );
        }
        if dim != "Many" || qn == "rare" {
            line(
                sb,
                &format!("{pre}all\t0\t0\t{dim}"),
                f.get_all_children(dim, &[]).map(|r| res(r, sparse)),
            );
        }
        line(
            sb,
            &format!("{pre}spec\t0\t0\t{dim}"),
            f.get_specific_value(dim, &[])
                .map(|v| format!("num\t{}", num(v))),
        );
        for p in paths(dim) {
            let dp = format!("{dim}|{}", p.join("|"));
            line(
                sb,
                &format!("{pre}top\t5\t0\t{dp}"),
                f.get_top_children(5, dim, &p).map(|r| res(r, false)),
            );
            line(
                sb,
                &format!("{pre}all\t0\t0\t{dp}"),
                f.get_all_children(dim, &p).map(|r| res(r, sparse)),
            );
            line(
                sb,
                &format!("{pre}spec\t0\t0\t{dp}"),
                f.get_specific_value(dim, &p)
                    .map(|v| format!("num\t{}", num(v))),
            );
        }
    }
    for n in [0, 1, 4] {
        line(
            sb,
            &format!("{pre}dims\t{n}\t0\t-"),
            f.get_all_dims(n).map(list),
        );
    }
    for (a, b) in [(0, 1), (1, 0), (1, 1), (2, 3), (3, 2), (100, 100)] {
        line(
            sb,
            &format!("{pre}topdims\t{a}\t{b}\t-"),
            f.get_top_dims(a, b).map(list),
        );
    }
}

// ---------------------------------------------------------------------------
// The searcher
// ---------------------------------------------------------------------------

struct Fixture {
    reader: DirectoryReader,
}

impl Fixture {
    fn open() -> Self {
        Fixture {
            reader: DirectoryReader::open(&FsDirectory::open(data().join("index"))).unwrap(),
        }
    }
}

fn queries(max_doc: i32) -> Vec<(&'static str, BooleanQuery)> {
    let b = |c: Clause| BooleanQuery {
        must: vec![c],
        ..Default::default()
    };
    vec![
        (
            "all",
            b(Clause::MatchAllDocs(MatchAllDocsQuery::new(max_doc))),
        ),
        ("even", b(Clause::Term(TermQuery::new("q", "even")))),
        ("rare", b(Clause::Term(TermQuery::new("q", "rare")))),
        ("none", b(Clause::Term(TermQuery::new("q", "none")))),
    ]
}

/// `GenFacetCounts.sparse(fc)`.
fn sparse(fc: &FacetsCollector<'_>) -> bool {
    let (mut max_doc, mut hits) = (0i32, 0i32);
    for m in fc.matching_docs() {
        hits += m.total_hits;
        max_doc += m.context.reader.max_doc();
    }
    hits < max_doc / 10
}

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
    format!("{}\t{relation}\t{}", t.total_hits.value, docs.join(","))
}

fn dims(f: &dyn Facets) -> String {
    f.get_all_dims(3)
        .unwrap()
        .iter()
        .map(|r| format!("{}={};", r.dim, num(r.value)))
        .collect()
}

#[test]
fn facets_calls_and_searches_match_lucene() {
    let fixture = Fixture::open();
    let reader = &fixture.reader;
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let owned = reader.field_norms_by_field(&["q".to_string()]).unwrap();
    let norms: Vec<SegmentNorms<'_, '_>> = owned
        .iter()
        .map(|m: &HashMap<String, FieldNorms<'_>>| Some(m))
        .collect();
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    let ctx = ValuesContext::new(&searcher);
    let taxo_dir = FsDirectory::open(data().join("taxo"));
    let taxo = DirectoryTaxonomyReader::open(&taxo_dir).unwrap();
    let config = config();
    let num_source = values_source::from_long_field("num");
    let scores_source = values_source::scores();

    let mut calls = String::new();
    let mut search = String::new();
    for (qn, q) in queries(reader.max_doc()) {
        let fc = FacetsCollectorManager::new(false)
            .collect(&searcher, &q)
            .unwrap();
        let fcs = FacetsCollectorManager::new(true)
            .collect(&searcher, &q)
            .unwrap();
        let is_sparse = taxo.get_size().unwrap() >= 1024 && sparse(&fc);
        let counts = FastTaxonomyFacetCounts::new(&taxo, &config, &fc).unwrap();
        let color = FastTaxonomyFacetCounts::with_field("$color", &taxo, &config, &fc).unwrap();
        let sum = AssociationAggregationFunction::Sum;
        let max = AssociationAggregationFunction::Max;
        let int_sum =
            TaxonomyFacetIntAssociations::with_field("$int_assoc", &taxo, &config, &fc, sum)
                .unwrap();
        let int_max =
            TaxonomyFacetIntAssociations::with_field("$int_assoc", &taxo, &config, &fc, max)
                .unwrap();
        let float_sum = (!is_sparse).then(|| {
            TaxonomyFacetFloatAssociations::with_field("$float_assoc", &taxo, &config, &fc, sum)
                .unwrap()
        });
        let vs_num = (!is_sparse).then(|| {
            TaxonomyFacetFloatAssociations::with_values_source(
                &taxo,
                &config,
                &fc,
                sum,
                num_source.as_ref(),
                &ctx,
            )
            .unwrap()
        });
        let float_max =
            TaxonomyFacetFloatAssociations::with_field("$float_assoc", &taxo, &config, &fc, max)
                .unwrap();
        let vs_score_max = TaxonomyFacetFloatAssociations::with_values_source(
            &taxo,
            &config,
            &fcs,
            max,
            scores_source.as_ref(),
            &ctx,
        )
        .unwrap();
        let multi = MultiFacets::new(
            vec![("Color".to_string(), &color as &dyn Facets)],
            Some(&counts),
        );
        let multi_nodefault =
            MultiFacets::new(vec![("Color".to_string(), &color as &dyn Facets)], None);
        let (count_all, count_all_color) = if qn == "all" {
            (
                Some(
                    FastTaxonomyFacetCounts::count_all("$facets", reader, &taxo, &config).unwrap(),
                ),
                Some(FastTaxonomyFacetCounts::count_all("$color", reader, &taxo, &config).unwrap()),
            )
        } else {
            (None, None)
        };

        let mut all: Vec<(&str, &dyn Facets)> = vec![
            ("counts", &counts),
            ("color", &color),
            ("int_sum", &int_sum),
            ("int_max", &int_max),
        ];
        if let (Some(fs), Some(vn)) = (&float_sum, &vs_num) {
            all.push(("float_sum", fs));
            all.push(("vs_num", vn));
        }
        all.push(("float_max", &float_max));
        all.push(("vs_score_max", &vs_score_max));
        all.push(("multi", &multi));
        all.push(("multi_nodefault", &multi_nodefault));
        if let (Some(a), Some(c)) = (&count_all, &count_all_color) {
            all.push(("count_all", a));
            all.push(("count_all_color", c));
        }
        for (name, f) in all {
            probe(
                &mut calls,
                name,
                qn,
                f,
                is_sparse && !name.starts_with("count_all"),
            );
        }

        // `FacetsCollectorManager.search` / `searchAfter`.
        let ns: &[i32] = if qn == "rare" || qn == "none" {
            &[0, 1, 7, 100_000]
        } else {
            &[0, 1, 7]
        };
        let fcm = FacetsCollectorManager::new(false);
        for &n in ns {
            let r = FacetsCollectorManager::search(&searcher, &q, n, &fcm).unwrap();
            let f = FastTaxonomyFacetCounts::new(&taxo, &config, &r.facets_collector).unwrap();
            writeln!(
                search,
                "{qn}\tsearch\t{n}\t{}\t{}",
                top_docs(&r.top_docs),
                dims(&f)
            )
            .unwrap();
        }
        let first = FacetsCollectorManager::search(&searcher, &q, 3, &fcm)
            .unwrap()
            .top_docs
            .score_docs;
        if first.len() == 3 {
            let after = ScoreDoc {
                doc_id: first[2].doc,
                score: first[2].score,
            };
            let r = FacetsCollectorManager::search_after(&searcher, after, &q, 5, &fcm).unwrap();
            writeln!(
                search,
                "{qn}\tafter\t{}\t{:x}\t{}",
                after.doc_id,
                after.score.to_bits(),
                top_docs(&r.top_docs)
            )
            .unwrap();
        }
    }
    let past = ScoreDoc {
        doc_id: reader.max_doc(),
        score: 1.0,
    };
    let match_all = &queries(reader.max_doc())[0].1;
    match FacetsCollectorManager::search_after(
        &searcher,
        past,
        match_all,
        5,
        &FacetsCollectorManager::new(false),
    ) {
        Ok(_) => search.push_str("error\tnone\n"),
        Err(Error::IllegalArgument(m)) => writeln!(search, "error\t{m}").unwrap(),
        Err(other) => panic!("{other:?}"),
    }

    assert_text_eq("calls.tsv", &calls, &read(&data().join("calls.tsv")));
    assert_text_eq("search.tsv", &search, &read(&data().join("search.tsv")));
}

fn joined(l: &FacetLabel) -> String {
    l.components().join("/")
}

#[test]
fn taxonomy_facet_labels_match_lucene() {
    let fixture = Fixture::open();
    let reader = &fixture.reader;
    let taxo_dir = FsDirectory::open(data().join("taxo"));
    let taxo = DirectoryTaxonomyReader::open(&taxo_dir).unwrap();
    let tfl = TaxonomyFacetLabels::new(&taxo, "$facets");
    let tfl_color = TaxonomyFacetLabels::new(&taxo, "$color");
    let mut sb = String::new();
    for ctx in IndexReader::leaves(reader) {
        let mut all = tfl.get_facet_label_reader(ctx).unwrap();
        let mut by_dim = tfl.get_facet_label_reader(ctx).unwrap();
        let mut by_dim2 = tfl.get_facet_label_reader(ctx).unwrap();
        let mut color = tfl_color.get_facet_label_reader(ctx).unwrap();
        for doc in (0..ctx.reader.max_doc()).step_by(97) {
            write!(sb, "{}\t{doc}\tall", ctx.ord).unwrap();
            while let Some(l) = all.next_facet_label(doc).unwrap() {
                write!(sb, "\t{}", joined(&l)).unwrap();
            }
            sb.push('\n');
            write!(sb, "{}\t{doc}\tPublish Date", ctx.ord).unwrap();
            while let Some(l) = by_dim.next_facet_label_in(doc, "Publish Date").unwrap() {
                write!(sb, "\t{}", joined(&l)).unwrap();
            }
            sb.push('\n');
            write!(sb, "{}\t{doc}\tTag+Path", ctx.ord).unwrap();
            for dim in ["Tag", "Path"] {
                let l = by_dim2.next_facet_label_in(doc, dim).unwrap();
                let shown = l.as_ref().map_or("null".to_string(), joined);
                write!(sb, "\t{shown}").unwrap();
            }
            sb.push('\n');
            write!(sb, "{}\t{doc}\tcolor", ctx.ord).unwrap();
            while let Some(l) = color.next_facet_label(doc).unwrap() {
                write!(sb, "\t{}", joined(&l)).unwrap();
            }
            sb.push('\n');
        }
    }
    assert_text_eq("labels.tsv", &sb, &read(&data().join("labels.tsv")));
}

/// `GenFacetCounts.error`: `name, class, message`.
fn outcome<T>(sb: &mut String, name: &str, r: Result<T>) {
    match r {
        Ok(_) => writeln!(sb, "{name}\tnone\t").unwrap(),
        Err(e) => {
            let class = match &e {
                Error::IllegalArgument(_) => "IllegalArgumentException",
                Error::IllegalState(_) => "IllegalStateException",
                other => panic!("{name}: {other:?}"),
            };
            writeln!(sb, "{name}\t{class}\t{e}").unwrap();
        }
    }
}

#[test]
fn refusals_match_lucene() {
    let fixture = Fixture::open();
    let reader = &fixture.reader;
    let taxo_dir = FsDirectory::open(data().join("taxo"));
    let taxo = DirectoryTaxonomyReader::open(&taxo_dir).unwrap();
    let tfl = TaxonomyFacetLabels::new(&taxo, "$facets");
    let leaf0 = IndexReader::leaves(reader)[0];
    let mut sb = String::new();
    outcome(&mut sb, "labels_out_of_order", {
        let mut r = tfl.get_facet_label_reader(leaf0).unwrap();
        r.next_facet_label(5).and_then(|_| r.next_facet_label(4))
    });
    outcome(&mut sb, "labels_out_of_order_dim", {
        let mut r = tfl.get_facet_label_reader(leaf0).unwrap();
        r.next_facet_label_in(5, "Tag")
            .and_then(|_| r.next_facet_label_in(4, "Tag"))
    });
    outcome(
        &mut sb,
        "labels_missing_dim",
        tfl.get_facet_label_reader(leaf0)
            .and_then(|mut r| r.next_facet_label_in(0, "Nope")),
    );

    // `GenFacetCounts.buildErrors`.
    let d = ByteBuffersDirectory::new();
    let w = DirectoryTaxonomyWriter::open(&d).unwrap();
    let wd: Option<&dyn TaxonomyWriter> = Some(&w);
    let mut c = FacetsConfig::new();
    c.set_hierarchical("H", true);
    c.set_require_dim_count("R", true);
    c.set_multi_valued("R", true);
    let int = |v: i32, dim: &str, label: &str| -> FacetEntry {
        AssociationFacetField::int(v, dim, &[label]).unwrap().into()
    };
    let float = |v: f32, dim: &str, label: &str| -> FacetEntry {
        AssociationFacetField::float(v, dim, &[label])
            .unwrap()
            .into()
    };
    let build = |c: &FacetsConfig, w: Option<&dyn TaxonomyWriter>, f: Vec<FacetEntry>| {
        c.build(w, &f, Document::new())
    };
    outcome(
        &mut sb,
        "assoc_hierarchical",
        build(&c, wd, vec![int(1, "H", "a")]),
    );
    outcome(
        &mut sb,
        "assoc_require_dim_count",
        build(&c, wd, vec![int(1, "R", "a")]),
    );
    outcome(
        &mut sb,
        "assoc_not_multivalued",
        build(&c, wd, vec![int(1, "A", "a"), int(2, "A", "b")]),
    );
    outcome(
        &mut sb,
        "assoc_first_int",
        build(&c, wd, vec![int(1, "A", "a")]),
    );
    outcome(
        &mut sb,
        "assoc_mix_float",
        build(&c, wd, vec![float(1.0, "B", "a")]),
    );
    outcome(
        &mut sb,
        "assoc_mix_float_again",
        build(&c, wd, vec![float(1.0, "B", "a")]),
    );
    let c2 = FacetsConfig::new();
    outcome(
        &mut sb,
        "assoc_first_float",
        build(&c2, wd, vec![float(1.0, "B", "a")]),
    );
    outcome(
        &mut sb,
        "assoc_mix_int",
        build(&c2, wd, vec![int(1, "A", "a")]),
    );
    outcome(
        &mut sb,
        "assoc_no_taxo_writer",
        build(&FacetsConfig::new(), None, vec![int(1, "A", "a")]),
    );
    outcome(
        &mut sb,
        "assoc_empty_path",
        AssociationFacetField::int::<&str>(1, "A", &[]),
    );
    outcome(
        &mut sb,
        "assoc_empty_label",
        AssociationFacetField::int(1, "A", &[""]),
    );
    outcome(
        &mut sb,
        "assoc_empty_dim",
        AssociationFacetField::float(1.0, "", &["a"]),
    );
    writeln!(
        sb,
        "int_to_string\t{}",
        AssociationFacetField::int(-7, "A", &["a", "b"]).unwrap()
    )
    .unwrap();
    writeln!(
        sb,
        "float_to_string\t{}",
        AssociationFacetField::float(1.5, "A", &["a"]).unwrap()
    )
    .unwrap();
    assert_text_eq("errors.tsv", &sb, &read(&data().join("errors.tsv")));
}
