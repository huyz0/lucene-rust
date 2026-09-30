#![allow(clippy::arithmetic_side_effects)]
//! **Values sources, `NumericFieldStats` and the rescorers against real
//! Lucene.**
//!
//! `fixtures/src/GenValuesRescore.java` writes two segments (the first with
//! deletions) carrying sparse numeric/float/double/sorted/sorted-set/
//! sorted-numeric doc values, points, a skip-indexed field, float and byte
//! vectors and a `LateInteractionField`, and records:
//!
//! * every document's value under sixteen double and five long values
//!   sources (fields, constants, query scores, vector similarities, late
//!   interaction, conversions) -- compared bit for bit, except the vector
//!   similarities, compared within `1e-6` relative: Lucene's float dot
//!   products sum in a different lane order (the tolerance
//!   `vector_query_fixtures.rs` uses; the documents with a value are
//!   compared exactly);
//! * `NumericFieldStats.getStats` from points (8- and 4-byte), from a skip
//!   index, and for fields with neither;
//! * over three first passes, `QueryRescorer.rescore` (three queries, two
//!   weights, two `topN`s), a `DoubleValuesSourceRescorer` over three
//!   sources, both `LateInteractionRescorer`s, `SortRescorer` under ten
//!   sorts (numeric, keyword missing first/last, `SortedSetSelector`
//!   `MIN`/`MAX`/`MIDDLE_MIN`/`MIDDLE_MAX`, sorted-numeric `MAX` and
//!   `DOUBLE`, score, doc) and `RescoreTopNQuery` over three sources -- every
//!   hit by document and score bits (and sort values).

mod m7support;

use std::collections::HashMap;
use std::sync::Arc;

use lucene_codecs::field_infos::VectorSimilarityFunction;
use lucene_codecs::vectors::FlatVectorsReader;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::field_norms::FieldNorms;
use lucene_search::index_searcher::IndexSearcher;
use lucene_search::query::{BooleanQuery, Clause, TermQuery};
use lucene_search::rescorer::{
    late_interaction_rescorer, late_interaction_rescorer_with_fallback, DoubleValuesSourceRescorer,
    QueryRescorer, RescoreTopNQuery, Rescorer, SortRescorer,
};
use lucene_search::top_docs::{ShardScoreDoc, TopDocs};
use lucene_search::top_field::{Selector, SortField, SortType};
use lucene_search::values_source::{
    self as vs, DoubleValuesSource, LateInteractionFloatValuesSource, LongValuesSource,
    ValuesContext,
};
use lucene_search::vector_query::VectorsInput;
use lucene_search::{TotalHits, TotalHitsRelation};
use lucene_store::FsDirectory;
use m7support::{fixture, Grammar, Manifest};

const QV: [f32; 4] = [0.5, -1.25, 2.0, 0.75];
const QEU: [f32; 3] = [1.0, 0.5, -0.5];
const QB: [i8; 4] = [3, -7, 12, 1];

fn qmv() -> Vec<Vec<f32>> {
    vec![vec![0.5, 1.0, -0.25], vec![1.5, -0.5, 0.25]]
}

const GRAMMAR: Grammar = Grammar {
    text: "body",
    range: "r",
};

fn pairs(s: &str) -> Vec<(i32, i64)> {
    if s.is_empty() {
        return Vec::new();
    }
    s.split(',')
        .map(|p| {
            let (d, v) = p.split_once(':').unwrap();
            (d.parse().unwrap(), v.parse().unwrap())
        })
        .collect()
}

fn hits(s: &str) -> Vec<Vec<String>> {
    if s.is_empty() {
        return Vec::new();
    }
    s.split(',')
        .map(|h| h.split(':').map(str::to_string).collect())
        .collect()
}

fn top_docs(s: &str, total: u64) -> TopDocs {
    TopDocs {
        total_hits: TotalHits {
            value: total,
            relation: TotalHitsRelation::EqualTo,
        },
        score_docs: hits(s)
            .into_iter()
            .map(|h| {
                ShardScoreDoc::new(
                    h[0].parse().unwrap(),
                    f32::from_bits(h[1].parse::<i32>().unwrap() as u32),
                )
            })
            .collect(),
    }
}

fn got_hits(td: &TopDocs) -> Vec<(i32, u32)> {
    td.score_docs
        .iter()
        .map(|h| (h.doc, h.score.to_bits()))
        .collect()
}

fn want_hits(s: &str) -> Vec<(i32, u32)> {
    hits(s)
        .into_iter()
        .map(|h| (h[0].parse().unwrap(), h[1].parse::<i32>().unwrap() as u32))
        .collect()
}

/// Every document's value under `source`, as `(global doc, value bits)`.
fn double_values(
    ctx: &ValuesContext<'_>,
    source: &dyn DoubleValuesSource,
    max_docs: &[i32],
) -> Vec<(i32, i64)> {
    let mut out = Vec::new();
    for (leaf, &max_doc) in max_docs.iter().enumerate() {
        let base = ctx.searcher().unwrap().segments()[leaf].doc_base;
        let mut v = source.get_values(ctx, leaf, None).unwrap();
        for doc in 0..max_doc {
            if v.advance_exact(doc).unwrap() {
                out.push((base + doc, v.double_value().unwrap().to_bits() as i64));
            }
        }
    }
    out
}

fn long_values(
    ctx: &ValuesContext<'_>,
    source: &dyn LongValuesSource,
    max_docs: &[i32],
) -> Vec<(i32, i64)> {
    let mut out = Vec::new();
    for (leaf, &max_doc) in max_docs.iter().enumerate() {
        let base = ctx.searcher().unwrap().segments()[leaf].doc_base;
        let mut v = source.get_values(ctx, leaf, None).unwrap();
        for doc in 0..max_doc {
            if v.advance_exact(doc).unwrap() {
                out.push((base + doc, v.long_value().unwrap()));
            }
        }
    }
    out
}

fn assert_close(name: &str, got: &[(i32, i64)], want: &[(i32, i64)], tolerant: bool) {
    let gd: Vec<i32> = got.iter().map(|p| p.0).collect();
    let wd: Vec<i32> = want.iter().map(|p| p.0).collect();
    assert_eq!(gd, wd, "{name}: documents with a value");
    for (g, w) in got.iter().zip(want) {
        if tolerant {
            let (g, w) = (f64::from_bits(g.1 as u64), f64::from_bits(w.1 as u64));
            assert!(
                (g - w).abs() <= 1e-6 * w.abs().max(1.0),
                "{name}: {g} vs {w}"
            );
        } else {
            assert_eq!(g.1, w.1, "{name}: doc {}", g.0);
        }
    }
}

/// `NumericUtils.doubleToSortableLong` over raw bits.
fn sortable(bits: i64) -> i64 {
    bits ^ ((bits >> 63) & 0x7fff_ffff_ffff_ffff)
}

fn sort_of(spec: &str) -> (Vec<SortField>, Vec<String>) {
    let mut sort = Vec::new();
    let mut kinds = Vec::new();
    for k in spec.split(',') {
        let p: Vec<&str> = k.split(':').collect();
        let reverse = p[2] == "true";
        let sf = match p[1] {
            "long" => SortField::numeric(p[0], SortType::Long, reverse),
            "string_first" => SortField::string(p[0], reverse),
            "string_last" => {
                let mut s = SortField::string(p[0], reverse);
                s.missing = 1;
                s
            }
            "min" | "max" | "middle_min" | "middle_max" => {
                let mut s = SortField::string(p[0], reverse);
                s.selector = match p[1] {
                    "min" => Selector::Min,
                    "max" => Selector::Max,
                    "middle_min" => Selector::MiddleMin,
                    _ => Selector::MiddleMax,
                };
                s
            }
            "max_long" => {
                let mut s = SortField::numeric(p[0], SortType::Long, reverse);
                s.selector = Selector::Max;
                s
            }
            "double" => SortField::numeric(p[0], SortType::Double, reverse),
            "score" => {
                let mut s = SortField::score();
                s.reverse = reverse;
                s
            }
            "doc" => {
                let mut s = SortField::doc();
                s.reverse = reverse;
                s
            }
            other => panic!("{other}"),
        };
        sort.push(sf);
        kinds.push(p[1].to_string());
    }
    (sort, kinds)
}

#[test]
fn values_sources_and_rescorers_match_real_lucene() {
    let dir = fixture("values_rescore_index");
    let m = Manifest::load(&format!("{dir}/manifest.properties"));
    let reader = DirectoryReader::open(&FsDirectory::open(&dir)).expect("open reader");
    let mut opened = reader.open_segments().expect("open postings");
    opened.open_points().expect("open points");
    let segments = opened.as_open_segments();
    let owned: Vec<HashMap<String, FieldNorms<'_>>> = reader
        .field_norms("body")
        .into_iter()
        .map(|n| n.into_iter().map(|n| ("body".to_string(), n)).collect())
        .collect();
    let norms: Vec<Option<&HashMap<String, FieldNorms<'_>>>> = owned.iter().map(Some).collect();
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    let max_docs: Vec<i32> = reader.segment_readers().iter().map(|r| r.max_doc).collect();

    // Each segment's vectors.
    let bytes: Vec<(Vec<u8>, Vec<u8>, String)> = (0..segments.len())
        .map(|s| {
            let read = |ext: &str| {
                std::fs::read(format!("{dir}/{}", m.get(&format!("s{s}.{ext}_file")))).unwrap()
            };
            (
                read("vemf"),
                read("vec"),
                m.get(&format!("s{s}.vector_suffix")).to_string(),
            )
        })
        .collect();
    let inputs: Vec<VectorsInput<'_>> = reader
        .segment_readers()
        .iter()
        .zip(&bytes)
        .map(|(r, (vemf, vec, suffix))| VectorsInput {
            flat: FlatVectorsReader::open(vemf, vec, &r.segment_id(), suffix).expect("vectors"),
            hnsw: None,
            field_infos: r.field_infos(),
            live_docs: r.live_docs(),
            filter: None,
            max_doc: r.max_doc,
        })
        .collect();
    let vectors: Vec<Option<&VectorsInput<'_>>> = inputs.iter().map(Some).collect();
    let ctx = ValuesContext::new(&searcher).with_vectors(&vectors);

    // Values sources.
    let q = |k: usize| GRAMMAR.query(m.get(&format!("query_source.{k}")));
    let doubles: Vec<(&str, Arc<dyn DoubleValuesSource>, bool)> = vec![
        ("long", vs::from_long_field("n"), false),
        ("int", vs::from_int_field("i"), false),
        ("float", vs::from_float_field("f"), false),
        ("double", vs::from_double_field("d"), false),
        ("const", vs::constant(2.5), false),
        ("query0", vs::from_query(q(0)), false),
        ("query1", vs::from_query(q(1)), false),
        ("query2", vs::from_query(q(2)), false),
        (
            "fvec",
            vs::float_vector_similarity("vec", QV.to_vec()),
            true,
        ),
        (
            "feu",
            vs::float_vector_similarity("veu", QEU.to_vec()),
            true,
        ),
        (
            "bvec",
            vs::byte_vector_similarity("bvec", QB.iter().map(|&b| b as u8).collect()),
            false,
        ),
        (
            "full",
            vs::full_precision_float_vector_similarity("vec", QV.to_vec(), None),
            true,
        ),
        (
            "full_mip",
            vs::full_precision_float_vector_similarity(
                "veu",
                QEU.to_vec(),
                Some(VectorSimilarityFunction::MaximumInnerProduct),
            ),
            true,
        ),
        (
            "late_cos",
            Arc::new(
                LateInteractionFloatValuesSource::with_function(
                    "li",
                    qmv(),
                    VectorSimilarityFunction::Cosine,
                )
                .unwrap(),
            ),
            true,
        ),
        (
            "late_eu",
            Arc::new(
                LateInteractionFloatValuesSource::with_function(
                    "li",
                    qmv(),
                    VectorSimilarityFunction::Euclidean,
                )
                .unwrap(),
            ),
            true,
        ),
        (
            "from_long",
            vs::to_double_values_source(vs::long_from_long_field("n")),
            false,
        ),
    ];
    for (name, source, tolerant) in &doubles {
        let got = double_values(&ctx, source.as_ref(), &max_docs);
        let want = pairs(m.get(&format!("vs.{name}")));
        assert!(!want.is_empty(), "{name}");
        assert_close(name, &got, &want, *tolerant);
    }
    let longs: Vec<(&str, Arc<dyn LongValuesSource>)> = vec![
        ("long", vs::long_from_long_field("n")),
        ("int", vs::long_from_int_field("i")),
        ("const", vs::long_constant(-7)),
        (
            "cast",
            vs::to_long_values_source(vs::from_double_field("d")),
        ),
        (
            "sortable",
            vs::to_sortable_long_values_source(vs::from_float_field("f")),
        ),
    ];
    for (name, source) in &longs {
        let got = long_values(&ctx, source.as_ref(), &max_docs);
        assert_eq!(got, pairs(m.get(&format!("ls.{name}"))), "{name}");
    }
    // Cacheability: plain doc values with no updates are; scores are not.
    assert!(doubles[0].1.is_cacheable(&ctx, 0));
    assert!(!doubles[5].1.is_cacheable(&ctx, 0));
    assert!(doubles[8].1.is_cacheable(&ctx, 1));

    // Sorting by a values source (`getSortField`).
    let readers = reader.segment_readers();
    let late = || -> Arc<dyn DoubleValuesSource> {
        Arc::new(
            LateInteractionFloatValuesSource::with_function(
                "li",
                qmv(),
                VectorSimilarityFunction::Cosine,
            )
            .unwrap(),
        )
    };
    let vsorts: usize = m.get("vsort_count").parse().unwrap();
    for i in 0..vsorts {
        let k = |f: &str| m.get(&format!("vsort.{i}.{f}")).to_string();
        let sort = match k("sort").as_str() {
            "float" => vec![vs::double_sort_field(vs::from_float_field("f"), false, 0.0)],
            "double_rev" => vec![vs::double_sort_field(
                vs::from_double_field("d"),
                true,
                -1.5,
            )],
            "long" => vec![vs::long_sort_field(vs::long_from_long_field("n"), false, 0)],
            "long_rev" => vec![vs::long_sort_field(vs::long_from_long_field("n"), true, 42)],
            "scores" => vec![vs::double_sort_field(vs::scores(), true, 0.0)],
            "late" => vec![vs::double_sort_field(late(), true, 0.0)],
            // Sources that need the searcher or the vectors: the sort is
            // rewritten against them (`Sort.rewrite(searcher)`).
            "query" => vec![vs::double_sort_field(vs::from_query(q(1)), false, 0.0)],
            "query_rev" => vec![
                vs::double_sort_field(vs::from_query(q(0)), true, 0.0),
                vs::double_sort_field(vs::scores(), true, 0.0),
            ],
            "full" => vec![vs::double_sort_field(
                vs::full_precision_float_vector_similarity(
                    "veu",
                    QEU.to_vec(),
                    Some(VectorSimilarityFunction::MaximumInnerProduct),
                ),
                true,
                0.0,
            )],
            "query_long" => vec![vs::long_sort_field(
                vs::to_long_values_source(vs::from_query(q(2))),
                true,
                0,
            )],
            "float_scores" => vec![
                vs::double_sort_field(vs::from_float_field("f"), true, 0.0),
                vs::double_sort_field(vs::scores(), false, 0.0),
            ],
            other => panic!("vsort {other}"),
        };
        let sort = lucene_search::top_field::rewrite_sort(&sort, &ctx).unwrap();
        let got = lucene_search::top_field::search_sorted(
            &segments,
            readers,
            &GRAMMAR.query(&k("query")),
            &norms,
            &sort,
            15,
            u64::MAX,
            None,
        )
        .unwrap();
        let want = hits(&k("hits"));
        assert_eq!(got.hits.len(), want.len(), "vsort.{i}");
        assert_eq!(got.total.value, k("total").parse::<u64>().unwrap());
        for (g, w) in got.hits.iter().zip(&want) {
            let what = format!("vsort.{i} {} doc {}", k("sort"), w[0]);
            assert_eq!(g.doc.to_string(), w[0], "{what}");
            for (j, raw) in w[1..].iter().enumerate() {
                let v = match raw.as_bytes()[0] {
                    b'd' => sortable(raw[1..].parse().unwrap()),
                    _ => raw[1..].parse().unwrap(),
                };
                assert_eq!(g.values[j], v, "{what}");
            }
        }
    }

    // NumericFieldStats.
    for f in ["r", "ip", "rs", "n", "missing"] {
        let got = vs::numeric_field_stats(&searcher, f).unwrap();
        let want = m.get(&format!("stats.{f}"));
        let got = got.map_or("null".to_string(), |s| {
            format!("{}:{}:{}", s.min, s.max, s.doc_count)
        });
        assert_eq!(got, want, "stats {f}");
    }

    // First passes: our own search must give Lucene's hits, then every
    // rescorer is run over Lucene's.
    let mut firsts = Vec::new();
    for f in 0.. {
        let Some(text) = m.opt(&format!("fp.{f}.query")) else {
            break;
        };
        let want = m.get(&format!("fp.{f}.hits"));
        let total: u64 = m.get(&format!("fp.{f}.total")).parse().unwrap();
        let ours = searcher.search(&GRAMMAR.query(text), 40).unwrap();
        assert_eq!(got_hits(&ours), want_hits(want), "first pass {text}");
        firsts.push(top_docs(want, total));
    }
    assert_eq!(firsts.len(), 3);

    let count = |k: &str| -> usize { m.get(&format!("{k}_count")).parse().unwrap() };
    for i in 0..count("qr") {
        let k = |f: &str| m.get(&format!("qr.{i}.{f}")).to_string();
        let fp = &firsts[k("fp").parse::<usize>().unwrap()];
        let got = QueryRescorer::rescore_with_weight(
            &ctx,
            fp,
            GRAMMAR.query(&k("query")),
            k("weight").parse().unwrap(),
            k("top_n").parse().unwrap(),
        )
        .unwrap();
        assert_eq!(got_hits(&got), want_hits(&k("hits")), "qr.{i}");
        assert_eq!(got.total_hits, fp.total_hits);
    }

    let combine: Arc<vs_combine::Combine> = Arc::new(|first: f32, present: bool, value: f64| {
        if present {
            (f64::from(first) + 0.5 * value) as f32
        } else {
            first * 0.5
        }
    });
    for i in 0..count("dvr") {
        let k = |f: &str| m.get(&format!("dvr.{i}.{f}")).to_string();
        let fp = &firsts[k("fp").parse::<usize>().unwrap()];
        let (source, tolerant): (Arc<dyn DoubleValuesSource>, bool) = match k("source").as_str() {
            "float" => (vs::from_float_field("f"), false),
            "fvec" => (
                vs::full_precision_float_vector_similarity("vec", QV.to_vec(), None),
                true,
            ),
            _ => (vs::from_query(q(0)), false),
        };
        let r = DoubleValuesSourceRescorer::new(source, Arc::clone(&combine));
        let got = r.rescore(&ctx, fp, k("top_n").parse().unwrap()).unwrap();
        assert_hits(&format!("dvr.{i}"), &got, &k("hits"), tolerant);
    }

    for i in 0..count("late") {
        let k = |f: &str| m.get(&format!("late.{i}.{f}")).to_string();
        let fp = &firsts[k("fp").parse::<usize>().unwrap()];
        let r = if k("kind") == "plain" {
            late_interaction_rescorer("li", qmv(), VectorSimilarityFunction::Cosine).unwrap()
        } else {
            late_interaction_rescorer_with_fallback(
                "li",
                qmv(),
                VectorSimilarityFunction::DotProduct,
            )
            .unwrap()
        };
        let got = r.rescore(&ctx, fp, 15).unwrap();
        assert_hits(&format!("late.{i}"), &got, &k("hits"), true);
    }

    for i in 0..count("sort") {
        let k = |f: &str| m.get(&format!("sort.{i}.{f}")).to_string();
        let fp = &firsts[k("fp").parse::<usize>().unwrap()];
        let (sort, kinds) = sort_of(&k("sort"));
        let r = SortRescorer::new(sort);
        let got = r
            .rescore_field_docs(&ctx, fp, k("top_n").parse().unwrap())
            .unwrap();
        let want = hits(&k("hits"));
        assert_eq!(got.hits.len(), want.len(), "sort.{i} {}", k("sort"));
        for (g, w) in got.hits.iter().zip(&want) {
            let what = format!("sort.{i} {} doc {}", k("sort"), w[0]);
            assert_eq!(g.fields.doc.to_string(), w[0], "{what}");
            assert_eq!(
                g.score.to_bits().to_string(),
                (w[1].parse::<i32>().unwrap() as u32).to_string(),
                "{what}: score"
            );
            for (j, kind) in kinds.iter().enumerate() {
                let raw = &w[2 + j];
                match kind.as_str() {
                    "string_first" | "string_last" | "min" | "max" | "middle_min"
                    | "middle_max" => {
                        let t = g.fields.terms[j].as_ref();
                        if raw == "-" {
                            assert!(t.is_none(), "{what}: missing term");
                        } else {
                            let hex: String =
                                t.unwrap().iter().map(|b| format!("{b:02x}")).collect();
                            assert_eq!(format!("x{hex}"), *raw, "{what}: term");
                        }
                    }
                    "double" => assert_eq!(
                        g.fields.values[j],
                        sortable(raw.parse().unwrap()),
                        "{what}: double"
                    ),
                    "score" => assert_eq!(
                        g.fields.values[j],
                        i64::from(raw.parse::<i32>().unwrap() as u32),
                        "{what}: score value"
                    ),
                    _ => assert_eq!(g.fields.values[j].to_string(), *raw, "{what}: value"),
                }
            }
        }
        let plain = r.rescore(&ctx, fp, k("top_n").parse().unwrap()).unwrap();
        assert_eq!(plain.score_docs.len(), want.len());
    }

    for i in 0..count("rtn") {
        let k = |f: &str| m.get(&format!("rtn.{i}.{f}")).to_string();
        let fpi = k("fp").parse::<usize>().unwrap();
        let inner = GRAMMAR.query(m.get(&format!("fp.{fpi}.query")));
        let n: usize = k("n").parse().unwrap();
        let (query, tolerant) = match k("source").as_str() {
            "float" => (
                RescoreTopNQuery::new(inner, vs::from_float_field("f"), n).unwrap(),
                false,
            ),
            "full" => (
                RescoreTopNQuery::full_precision(inner, QV.to_vec(), "vec", n).unwrap(),
                true,
            ),
            _ => (
                RescoreTopNQuery::late_interaction(
                    inner,
                    n,
                    "li",
                    qmv(),
                    VectorSimilarityFunction::Cosine,
                )
                .unwrap(),
                true,
            ),
        };
        let got = query.search(&ctx, 20).unwrap();
        assert_hits(&format!("rtn.{i}"), &got, &k("hits"), tolerant);
        assert_eq!(
            got.total_hits.value,
            k("total").parse::<u64>().unwrap(),
            "rtn.{i} total"
        );
    }

    // `RescoreTopNQuery` as a clause of a boolean, searched through the
    // searcher (which rewrites it to a `DocAndScoreQuery` first).
    assert!(count("rtnb") > 0);
    for i in 0..count("rtnb") {
        let k = |f: &str| m.get(&format!("rtnb.{i}.{f}")).to_string();
        let fpi = k("fp").parse::<usize>().unwrap();
        let inner = GRAMMAR.query(m.get(&format!("fp.{fpi}.query")));
        let n: usize = k("n").parse().unwrap();
        let rtn = RescoreTopNQuery::new(inner, vs::from_float_field("f"), n).unwrap();
        let mut q = BooleanQuery::new();
        q.must.push(Clause::from(rtn));
        q.should
            .push(Clause::Term(TermQuery::new("body", b"w1".to_vec())));
        let got = searcher.search(&q, 20).unwrap();
        assert_hits(&format!("rtnb.{i}"), &got, &k("hits"), false);
        assert_eq!(
            got.total_hits.value,
            k("total").parse::<u64>().unwrap(),
            "rtnb.{i} total"
        );
    }
}

mod vs_combine {
    pub type Combine = dyn Fn(f32, bool, f64) -> f32 + Send + Sync;
}

/// Hits by document, in order; scores exactly, or within `1e-6` relative
/// where they come from float vector arithmetic.
fn assert_hits(what: &str, got: &TopDocs, want: &str, tolerant: bool) {
    let want = want_hits(want);
    let g: Vec<i32> = got.score_docs.iter().map(|h| h.doc).collect();
    let w: Vec<i32> = want.iter().map(|h| h.0).collect();
    if !tolerant {
        assert_eq!(got_hits(got), want, "{what}");
        return;
    }
    // A near-tie may order differently under a one-ulp difference: compare
    // the documents as sets of equal-score runs, and each score closely.
    let mut gs = g.clone();
    let mut ws = w.clone();
    gs.sort_unstable();
    ws.sort_unstable();
    assert_eq!(gs, ws, "{what}: documents");
    let by_doc: HashMap<i32, f32> = want.iter().map(|&(d, b)| (d, f32::from_bits(b))).collect();
    for h in &got.score_docs {
        let w = by_doc[&h.doc];
        assert!(
            (h.score - w).abs() <= 1e-6 * w.abs().max(1.0),
            "{what}: doc {} {} vs {w}",
            h.doc,
            h.score
        );
    }
}
