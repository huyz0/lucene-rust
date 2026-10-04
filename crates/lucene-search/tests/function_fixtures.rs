//! M10 T10.5, differentially against Lucene 10.5.0:
//! `fixtures/src/GenFunction.java`.
//!
//! `values.tsv`: every value source of `lucene-queries`' `function` package
//! (and the bridges to `DoubleValuesSource`), read for every document of a
//! four-segment index with deletions, missing and multi-valued fields --
//! every getter's value bits, string, object, vector and filled mutable
//! value, or the exception Lucene threw. `searches.tsv`: `FunctionQuery`,
//! `FunctionRangeQuery`, `FunctionScoreQuery` (`boostByValue`,
//! `boostByQuery`, the `IndexReaderFunctions`), `FunctionMatchQuery`, alone,
//! boosted and inside booleans, and sorts by a value source: every hit's
//! score bits, and four documents' explanations. `groups.tsv`:
//! `GroupingSearch` by a value source (`lucene-grouping`'s
//! `ValueSourceGroupSelector`). Every file is rebuilt here line for line and
//! compared with Lucene's.

// Test fixtures' own arithmetic -- see `docs/arithmetic-gate.md`'s "Test code".
#![allow(clippy::arithmetic_side_effects)]

use std::collections::HashMap;
use std::sync::Arc;

use lucene_codecs::field_infos::VectorSimilarityFunction;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::function::index_reader_functions as irf;
use lucene_search::function::valuesource::*;
use lucene_search::function::{
    as_double_values_source, as_long_values_source, from_double_values_source, java_byte_array,
    java_float_array, queries_stats, sort_field, FunctionContext, FunctionMatchQuery,
    FunctionQuery, FunctionRangeQuery, FunctionScoreQuery, FunctionValues, MutableValue, ObjectVal,
    TopLevel, ValueLeaf, ValueSource,
};
use lucene_search::grouping::{GroupingSearch, ValueSourceGroupSelector};
use lucene_search::index_searcher::{IndexSearcher, SegmentNorms};
use lucene_search::query::{BooleanQuery, BoostQuery, Clause, MatchAllDocsQuery, TermQuery};
use lucene_search::similarities::ClassicSimilarity;
use lucene_search::top_field::SortField;
use lucene_search::values_source::{self as dvs_mod, DoubleValuesSource};
use lucene_search::Error;
use lucene_store::FsDirectory;

fn data() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/function")
}

// ---------------------------------------------------------------------------
// The spec grammar
// ---------------------------------------------------------------------------

fn split(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let (mut depth, mut start) = (0i32, 0usize);
    for (i, c) in s.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => depth -= 1,
            ',' if depth == 0 => {
                out.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    if start < s.len() || !out.is_empty() {
        out.push(&s[start..]);
    }
    out
}

fn name(spec: &str) -> &str {
    spec.split_once('(').map_or(spec, |(n, _)| n)
}

fn args(spec: &str) -> Vec<&str> {
    match spec.find('(') {
        None => Vec::new(),
        Some(p) => split(&spec[p + 1..spec.len() - 1]),
    }
}

fn term(word: &str) -> Clause {
    Clause::Term(TermQuery::new("body", word.as_bytes().to_vec()))
}

fn query(spec: &str) -> Clause {
    let p: Vec<&str> = spec.split(':').collect();
    match p[0] {
        "all" => Clause::MatchAllDocs(MatchAllDocsQuery::new(0)),
        "t" => term(p[1]),
        "or" => Clause::Boolean(Box::new(BooleanQuery {
            should: vec![term(p[1]), term(p[2])],
            ..Default::default()
        })),
        "and" => Clause::Boolean(Box::new(BooleanQuery {
            must: vec![term(p[1]), term(p[2])],
            ..Default::default()
        })),
        _ => panic!("query {spec}"),
    }
}

fn f32_of(s: &str) -> f32 {
    s.parse().unwrap()
}

fn vs_list(a: &[&str]) -> Vec<Arc<dyn ValueSource>> {
    a.iter().map(|s| vs(s)).collect()
}

fn numeric_selector(s: &str) -> NumericSelector {
    match s {
        "MIN" => NumericSelector::Min,
        "MAX" => NumericSelector::Max,
        _ => panic!("{s}"),
    }
}

fn similarity(s: &str) -> VectorSimilarityFunction {
    match s {
        "EUCLIDEAN" => VectorSimilarityFunction::Euclidean,
        "DOT_PRODUCT" => VectorSimilarityFunction::DotProduct,
        "COSINE" => VectorSimilarityFunction::Cosine,
        "MAXIMUM_INNER_PRODUCT" => VectorSimilarityFunction::MaximumInnerProduct,
        _ => panic!("{s}"),
    }
}

fn vs(spec: &str) -> Arc<dyn ValueSource> {
    let a = args(spec);
    match name(spec) {
        "int" => Arc::new(IntFieldSource::new(a[0])),
        "long" => Arc::new(LongFieldSource::new(a[0])),
        "float" => Arc::new(FloatFieldSource::new(a[0])),
        "double" => Arc::new(DoubleFieldSource::new(a[0])),
        "mint" => Arc::new(MultiValuedIntFieldSource::new(a[0], numeric_selector(a[1]))),
        "mlong" => Arc::new(MultiValuedLongFieldSource::new(
            a[0],
            numeric_selector(a[1]),
        )),
        "mfloat" => Arc::new(MultiValuedFloatFieldSource::new(
            a[0],
            numeric_selector(a[1]),
        )),
        "mdouble" => Arc::new(MultiValuedDoubleFieldSource::new(
            a[0],
            numeric_selector(a[1]),
        )),
        "bytes" => Arc::new(BytesRefFieldSource::new(a[0])),
        "sset" => Arc::new(SortedSetFieldSource::with_selector(
            a[0],
            match a[1] {
                "MIN" => SetSelector::Min,
                "MAX" => SetSelector::Max,
                "MIDDLE_MIN" => SetSelector::MiddleMin,
                "MIDDLE_MAX" => SetSelector::MiddleMax,
                s => panic!("{s}"),
            },
        )),
        "enum" => {
            let names = ["zero", "one", "two", "three"];
            let i2s: HashMap<i32, String> = names
                .iter()
                .enumerate()
                .map(|(i, n)| (i as i32, n.to_string()))
                .collect();
            let s2i: HashMap<String, i32> = names
                .iter()
                .enumerate()
                .map(|(i, n)| (n.to_string(), i as i32))
                .collect();
            Arc::new(EnumFieldSource::new(a[0], i2s, s2i))
        }
        "joindf" => Arc::new(JoinDocFreqValueSource::new(a[0], a[1])),
        "const" => Arc::new(ConstValueSource::new(f32_of(a[0]))),
        "dconst" => Arc::new(DoubleConstValueSource::new(a[0].parse().unwrap())),
        "literal" => Arc::new(LiteralValueSource::new(a[0])),
        "sum" => Arc::new(SumFloatFunction::new(vs_list(&a))),
        "product" => Arc::new(ProductFloatFunction::new(vs_list(&a))),
        "max" => Arc::new(MaxFloatFunction::new(vs_list(&a))),
        "min" => Arc::new(MinFloatFunction::new(vs_list(&a))),
        "div" => Arc::new(DivFloatFunction::new(vs(a[0]), vs(a[1]))),
        "pow" => Arc::new(PowFloatFunction::new(vs(a[0]), vs(a[1]))),
        "linear" => Arc::new(LinearFloatFunction::new(
            vs(a[0]),
            f32_of(a[1]),
            f32_of(a[2]),
        )),
        "recip" => Arc::new(ReciprocalFloatFunction::new(
            vs(a[0]),
            f32_of(a[1]),
            f32_of(a[2]),
            f32_of(a[3]),
        )),
        "map" => Arc::new(RangeMapFloatFunction::new(
            vs(a[0]),
            f32_of(a[1]),
            f32_of(a[2]),
            vs(a[3]),
            (a[4] != "null").then(|| vs(a[4])),
        )),
        "scale" => Arc::new(ScaleFloatFunction::new(
            vs(a[0]),
            f32_of(a[1]),
            f32_of(a[2]),
        )),
        "def" => Arc::new(DefFunction::new(vs_list(&a))),
        "if" => Arc::new(IfFunction::new(vs(a[0]), vs(a[1]), vs(a[2]))),
        "exists" => Arc::new(SimpleBoolFunction::new(
            "exists",
            vs(a[0]),
            Arc::new(|doc, vals: &mut dyn FunctionValues| vals.exists(doc)),
        )),
        "and" | "or" => {
            let and = name(spec) == "and";
            Arc::new(MultiBoolFunction::new(
                if and { "and" } else { "or" },
                vs_list(&a),
                Arc::new(move |doc, vals: &mut [Box<dyn FunctionValues + '_>]| {
                    for v in vals.iter_mut() {
                        if v.bool_val(doc)? != and {
                            return Ok(!and);
                        }
                    }
                    Ok(and)
                }),
            ))
        }
        "gt" => Arc::new(ComparisonBoolFunction::new(
            vs(a[0]),
            vs(a[1]),
            "gt",
            Arc::new(
                |doc, l: &mut dyn FunctionValues, r: &mut dyn FunctionValues| {
                    Ok(l.double_val(doc)? > r.double_val(doc)?)
                },
            ),
        )),
        "sqrt" => Arc::new(SimpleFloatFunction::new(
            "sqrt",
            vs(a[0]),
            Arc::new(|doc, vals: &mut dyn FunctionValues| {
                Ok(f64::from(vals.float_val(doc)?).sqrt() as f32)
            }),
        )),
        "docfreq" => Arc::new(DocFreqValueSource::new(a[0], a[1], a[0], a[1].as_bytes())),
        "idf" => Arc::new(IDFValueSource::new(a[0], a[1], a[0], a[1].as_bytes())),
        "termfreq" => Arc::new(TermFreqValueSource::new(a[0], a[1], a[0], a[1].as_bytes())),
        "tf" => Arc::new(TFValueSource::new(a[0], a[1], a[0], a[1].as_bytes())),
        "ttf" => Arc::new(TotalTermFreqValueSource::new(
            a[0],
            a[1],
            a[0],
            a[1].as_bytes(),
        )),
        "sttf" => Arc::new(SumTotalTermFreqValueSource::new(a[0])),
        "numdocs" => Arc::new(NumDocsValueSource::new()),
        "maxdoc" => Arc::new(MaxDocValueSource::new()),
        "norm" => Arc::new(NormValueSource::new(a[0])),
        "query" => Arc::new(QueryValueSource::new(query(a[0]), f32_of(a[1]))),
        "fvec" => Arc::new(FloatKnnVectorFieldSource::new(a[0])),
        "bvec" => Arc::new(ByteKnnVectorFieldSource::new(a[0])),
        "cfvec" => {
            Arc::new(ConstKnnFloatValueSource::new(a.iter().map(|s| f32_of(s)).collect()).unwrap())
        }
        "cbvec" => Arc::new(ConstKnnByteVectorValueSource::new(
            a.iter().map(|s| s.parse::<i8>().unwrap() as u8).collect(),
        )),
        "fsim" => Arc::new(FloatVectorSimilarityFunction::new(
            similarity(a[0]),
            vs(a[1]),
            vs(a[2]),
        )),
        "bsim" => Arc::new(ByteVectorSimilarityFunction::new(
            similarity(a[0]),
            vs(a[1]),
            vs(a[2]),
        )),
        "vector" => Arc::new(VectorValueSource::new(vs_list(&a))),
        "fromdvs" => from_double_values_source(dvs(a[0])),
        n => panic!("value source {n}"),
    }
}

fn dvs(spec: &str) -> Arc<dyn DoubleValuesSource> {
    let a = args(spec);
    match name(spec) {
        "vs" => as_double_values_source(vs(a[0])),
        "lvs" => dvs_mod::to_double_values_source(as_long_values_source(vs(a[0]))),
        "dint" => dvs_mod::from_int_field(a[0]),
        "dlong" => dvs_mod::from_long_field(a[0]),
        "dfloat" => dvs_mod::from_float_field(a[0]),
        "ddouble" => dvs_mod::from_double_field(a[0]),
        "dconst" => dvs_mod::constant(a[0].parse().unwrap()),
        "scores" => dvs_mod::scores(),
        "dquery" => dvs_mod::from_clause(query(a[0])),
        "irdocfreq" => irf::doc_freq(a[0], a[1].as_bytes()),
        "irmaxdoc" => irf::max_doc(),
        "irnumdocs" => irf::num_docs(),
        "irnumdeleted" => irf::num_deleted_docs(),
        "irsttf" => dvs_mod::to_double_values_source(irf::sum_total_term_freq(a[0])),
        "irtermfreq" => irf::term_freq(a[0], a[1].as_bytes()),
        "irttf" => irf::total_term_freq(a[0], a[1].as_bytes()),
        "irsumdocfreq" => irf::sum_doc_freq(a[0]),
        "irdoccount" => irf::doc_count(a[0]),
        n => panic!("values source {n}"),
    }
}

fn predicate(spec: &str) -> Arc<dyn Fn(f64) -> bool + Send + Sync> {
    let p: Vec<&str> = spec.split(':').collect();
    match p[0] {
        "gt" => {
            let x: f64 = p[1].parse().unwrap();
            Arc::new(move |v| v > x)
        }
        "le" => {
            let x: f64 = p[1].parse().unwrap();
            Arc::new(move |v| v <= x)
        }
        "nan" => Arc::new(f64::is_nan),
        _ => panic!("{spec}"),
    }
}

// ---------------------------------------------------------------------------
// Printing, as the generator prints
// ---------------------------------------------------------------------------

fn hex32(f: f32) -> String {
    format!("{:x}", if f.is_nan() { 0x7fc0_0000 } else { f.to_bits() })
}

fn hex64(d: f64) -> String {
    format!(
        "{:x}",
        if d.is_nan() {
            0x7ff8_0000_0000_0000
        } else {
            d.to_bits()
        }
    )
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

fn g(r: lucene_search::Result<String>) -> String {
    match r {
        Ok(s) => clean(&s),
        Err(e) => err(&e),
    }
}

fn object(o: &ObjectVal) -> String {
    match o {
        ObjectVal::Null => "null".into(),
        ObjectVal::Float(f) => format!("F{}", hex32(*f)),
        ObjectVal::Double(d) => format!("D{}", hex64(*d)),
        ObjectVal::Int(i) => format!("I{i}"),
        ObjectVal::Long(l) => format!("L{l}"),
        ObjectVal::Str(s) => format!("S{s}"),
        ObjectVal::Bool(b) => format!("B{b}"),
    }
}

fn fill(m: &MutableValue) -> String {
    let kind = match m {
        MutableValue::Bool { .. } => "Bool",
        MutableValue::Int { .. } => "Int",
        MutableValue::Long { .. } => "Long",
        MutableValue::Float { .. } => "Float",
        MutableValue::Double { .. } => "Double",
        MutableValue::Str { .. } => "Str",
        MutableValue::Date { .. } => "Date",
    };
    format!("{kind}:{}:{m}", m.exists())
}

fn java_list<T: ToString>(v: &[T]) -> String {
    let parts: Vec<String> = v.iter().map(ToString::to_string).collect();
    format!("[{}]", parts.join(", "))
}

// ---------------------------------------------------------------------------
// values.tsv
// ---------------------------------------------------------------------------

fn dump(
    searcher: &IndexSearcher<'_, '_>,
    sim: &str,
    spec: &str,
    out: &mut Vec<String>,
) -> lucene_search::Result<()> {
    let vs = vs(spec);
    let head = format!("{sim}\t{spec}");
    out.push(format!("{head}\tdesc\t{}", clean(&vs.description())));
    let stats = queries_stats(searcher, vs.as_ref())?;
    let top = TopLevel::of_searcher(searcher, Some(&stats));
    let fcx = match FunctionContext::create(vs.as_ref(), &top) {
        Ok(f) => f,
        Err(e) => {
            out.push(format!("{head}\tweight\t{}", err(&e)));
            return Ok(());
        }
    };
    let multi = name(spec) == "vector";
    for seg in 0..searcher.segments().len() {
        let leaf = ValueLeaf::of_searcher(searcher, seg, Some(&stats))?;
        let max_doc = leaf.max_doc()?;
        let pair = vs
            .get_values(&fcx, &leaf)
            .and_then(|a| Ok((a, vs.get_values(&fcx, &leaf)?)));
        let (mut fv, mut filler) = match pair {
            Ok(p) => p,
            Err(e) => {
                out.push(format!("{head}\t{seg}\t{}", err(&e)));
                continue;
            }
        };
        let mut mval = filler.new_value();
        for d in 0..max_doc {
            let mut cols = vec![g(fv.exists(d).map(|b| b.to_string()))];
            if multi {
                let n = args(spec).len();
                cols.push(g({
                    let mut v = vec![0f32; n];
                    fv.float_vals(d, &mut v)
                        .map(|()| v.iter().map(|x| hex32(*x)).collect::<Vec<_>>().join(" "))
                }));
                cols.push(g({
                    let mut v = vec![0f64; n];
                    fv.double_vals(d, &mut v)
                        .map(|()| v.iter().map(|x| hex64(*x)).collect::<Vec<_>>().join(" "))
                }));
                cols.push(g({
                    let mut v = vec![0i32; n];
                    fv.int_vals(d, &mut v).map(|()| java_list(&v))
                }));
                cols.push(g({
                    let mut v = vec![0i64; n];
                    fv.long_vals(d, &mut v).map(|()| java_list(&v))
                }));
                cols.push(g({
                    let mut v = vec![0i8; n];
                    fv.byte_vals(d, &mut v).map(|()| java_list(&v))
                }));
                cols.push(g({
                    let mut v = vec![0i16; n];
                    fv.short_vals(d, &mut v).map(|()| java_list(&v))
                }));
                cols.push(g({
                    let mut v = vec![None; n];
                    fv.str_vals(d, &mut v).map(|()| {
                        let s: Vec<String> = v
                            .into_iter()
                            .map(|x| x.unwrap_or_else(|| "null".into()))
                            .collect();
                        java_list(&s)
                    })
                }));
                cols.push(g(fv.to_string_doc(d)));
            } else {
                cols.push(g(fv.byte_val(d).map(|v| v.to_string())));
                cols.push(g(fv.short_val(d).map(|v| v.to_string())));
                cols.push(g(fv.float_val(d).map(hex32)));
                cols.push(g(fv.double_val(d).map(hex64)));
                cols.push(g(fv.int_val(d).map(|v| v.to_string())));
                cols.push(g(fv.long_val(d).map(|v| v.to_string())));
                cols.push(g(fv.bool_val(d).map(|v| v.to_string())));
                cols.push(g(fv.str_val(d).map(|v| v.unwrap_or_else(|| "null".into()))));
                cols.push(g({
                    let mut b = Vec::new();
                    fv.bytes_val(d, &mut b)
                        .map(|has| format!("{has}:{}", String::from_utf8_lossy(&b)))
                }));
                cols.push(g(fv.object_val(d).map(|o| object(&o))));
                cols.push(g(fv.ord_val(d).map(|v| v.to_string())));
                cols.push(g(fv.float_vector_val(d).map(|v| {
                    v.map_or_else(|| "null".into(), |v| java_float_array(&v))
                })));
                cols.push(g(fv.byte_vector_val(d).map(|v| {
                    v.map_or_else(|| "null".into(), |v| java_byte_array(&v))
                })));
                cols.push(g(fv.to_string_doc(d)));
                cols.push(g(filler.fill_value(d, &mut mval).map(|()| fill(&mval))));
            }
            out.push(format!("{head}\t{seg}\t{d}\t{}", cols.join("\t")));
        }
        out.push(format!(
            "{head}\t{seg}\tnumOrd\t{}",
            g(fv.num_ord().map(|v| v.to_string()))
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// searches.tsv
// ---------------------------------------------------------------------------

fn as_boolean(c: Clause) -> BooleanQuery {
    match c {
        Clause::Boolean(b) => *b,
        other => BooleanQuery {
            must: vec![other],
            ..Default::default()
        },
    }
}

fn search_lines(
    searcher: &IndexSearcher<'_, '_>,
    kind: &str,
    spec: &str,
    q: &BooleanQuery,
    out: &mut Vec<String>,
) {
    let head = format!("{kind}\t{spec}");
    match searcher.search(q, 1000) {
        Ok(td) => {
            let mut b = format!("{} ", td.total_hits.value);
            for sd in &td.score_docs {
                b.push_str(&format!("{}:{},", sd.doc, hex32(sd.score)));
            }
            out.push(format!("{head}\thits\t{b}"));
        }
        Err(e) => out.push(format!("{head}\thits\t{}", err(&e))),
    }
    // A filter's term explains itself unscored in Lucene, which explain
    // does not render: the generator records only the hits.
    if kind == "fsqfilter" {
        return;
    }
    for doc in [0, 7, 29, 61] {
        let e = g(searcher.explain(q, doc).map(|e| e.to_string()));
        out.push(format!("{head}\texplain {doc}\t{e}"));
    }
}

fn fsq(spec: &str) -> FunctionScoreQuery {
    let parts: Vec<&str> = spec.split('|').collect();
    let inner = query(parts[0]);
    match parts[1] {
        "plain" => FunctionScoreQuery::new(inner, dvs(parts[2])),
        "boostval" => FunctionScoreQuery::boost_by_value(inner, dvs(parts[2])),
        _ => {
            let (q, b) = parts[2].rsplit_once(':').unwrap();
            FunctionScoreQuery::boost_by_query(inner, query(q), f32_of(b))
        }
    }
}

fn opt(s: &str) -> Option<&str> {
    (s != "null").then_some(s)
}

#[allow(clippy::too_many_arguments)]
fn searches(
    bm25: &IndexSearcher<'_, '_>,
    classic: &IndexSearcher<'_, '_>,
    readers: &[lucene_search::directory_reader::SegmentReader],
    kind: &str,
    spec: &str,
    out: &mut Vec<String>,
) {
    let fq = |s: &str| Clause::from(FunctionQuery::new(vs(s)));
    match kind {
        "fq" => search_lines(bm25, kind, spec, &as_boolean(fq(spec)), out),
        "fqclassic" => search_lines(classic, kind, spec, &as_boolean(fq(spec)), out),
        "fqboost" => search_lines(
            bm25,
            kind,
            spec,
            &as_boolean(Clause::Boost(Box::new(BoostQuery::new(fq(spec), 2.5)))),
            out,
        ),
        "fqbool" => {
            let q = BooleanQuery {
                must: vec![term("red")],
                should: vec![fq(spec)],
                ..Default::default()
            };
            search_lines(bm25, kind, spec, &q, out)
        }
        "frange" | "frangefilter" => {
            let p: Vec<&str> = spec.split('|').collect();
            let fr = Clause::from(FunctionRangeQuery::new(
                vs(p[0]),
                opt(p[1]),
                opt(p[2]),
                p[3] == "true",
                p[4] == "true",
            ));
            let q = if kind == "frange" {
                as_boolean(fr)
            } else {
                BooleanQuery {
                    must: vec![term("blue")],
                    filter: vec![fr],
                    ..Default::default()
                }
            };
            search_lines(bm25, kind, spec, &q, out)
        }
        "fsq" | "fsqboost" | "fsqbool" | "fsqfilter" => {
            let f = Clause::from(fsq(spec));
            let q = match kind {
                "fsq" => as_boolean(f),
                "fsqboost" => as_boolean(Clause::Boost(Box::new(BoostQuery::new(f, 3.0)))),
                "fsqbool" => BooleanQuery {
                    should: vec![f, term("slow")],
                    ..Default::default()
                },
                _ => BooleanQuery {
                    must: vec![term("big")],
                    filter: vec![f],
                    ..Default::default()
                },
            };
            search_lines(bm25, kind, spec, &q, out)
        }
        "fmq" | "fmqboost" => {
            let (s, p) = spec.split_once('|').unwrap();
            let f = Clause::from(FunctionMatchQuery::new(dvs(s), predicate(p)));
            let q = if kind == "fmq" {
                as_boolean(f)
            } else {
                as_boolean(Clause::Boost(Box::new(BoostQuery::new(f, 0.5))))
            };
            search_lines(bm25, kind, spec, &q, out)
        }
        "sort" => {
            let (s, rev) = spec.split_once('|').unwrap();
            let head = format!("{kind}\t{spec}");
            let sort = vec![sort_field(vs(s), rev == "true"), SortField::doc()];
            let all = as_boolean(Clause::MatchAllDocs(MatchAllDocsQuery::new(0)));
            let r = lucene_search::top_field::rewrite_sort(
                &sort,
                &lucene_search::values_source::ValuesContext::new(bm25),
            )
            .and_then(|sort| bm25.search_sorted(readers, &all, 12, &sort, None));
            match r {
                Ok(td) => {
                    let b: String = td.hits.iter().map(|h| format!("{},", h.doc)).collect();
                    out.push(format!("{head}\thits\t{b}"));
                }
                Err(e) => out.push(format!("{head}\thits\t{}", err(&e))),
            }
        }
        k => panic!("search kind {k}"),
    }
}

/// Compares Rust's lines with Lucene's, reporting the first differences.
fn compare(what: &str, want: &[&str], got: &[String]) {
    let mut bad = 0;
    for (i, w) in want.iter().enumerate() {
        let g = got.get(i).map_or("<missing>", String::as_str);
        if *w != g {
            bad += 1;
            if bad <= 12 {
                eprintln!("{what} line {i}:\n  lucene: {w}\n  rust:   {g}");
            }
        }
    }
    assert_eq!(want.len(), got.len(), "{what}: line counts");
    assert_eq!(bad, 0, "{what}: {bad} of {} lines differ", want.len());
}

#[test]
fn function_queries_match_lucene() {
    let dir = data();
    let reader = DirectoryReader::open(&FsDirectory::open(dir.join("index"))).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let owned = reader.field_norms_by_field(&["body".to_string(), "id".to_string()]);
    let norms: Vec<SegmentNorms<'_, '_>> = owned.iter().map(Some).collect();
    let bm25 = IndexSearcher::new(&segments, &norms).unwrap();
    let classic_sim = ClassicSimilarity::default();
    let mut classic = IndexSearcher::new(&segments, &norms).unwrap();
    classic.set_similarity(&classic_sim);

    // values.tsv
    let text = std::fs::read_to_string(dir.join("values.tsv")).unwrap();
    let want: Vec<&str> = text.lines().collect();
    let mut specs: Vec<(&str, &str)> = Vec::new();
    for line in &want {
        let mut p = line.split('\t');
        let (sim, spec) = (p.next().unwrap(), p.next().unwrap());
        if specs.last() != Some(&(sim, spec)) {
            specs.push((sim, spec));
        }
    }
    let mut got = Vec::new();
    for (sim, spec) in &specs {
        let s = if *sim == "classic" { &classic } else { &bm25 };
        dump(s, sim, spec, &mut got).unwrap();
    }
    compare("values.tsv", &want, &got);

    // searches.tsv
    let text = std::fs::read_to_string(dir.join("searches.tsv")).unwrap();
    let want: Vec<&str> = text.lines().collect();
    let mut runs: Vec<(&str, &str)> = Vec::new();
    for line in &want {
        let mut p = line.split('\t');
        let (kind, spec) = (p.next().unwrap(), p.next().unwrap());
        if runs.last() != Some(&(kind, spec)) {
            runs.push((kind, spec));
        }
    }
    let mut got = Vec::new();
    for (kind, spec) in &runs {
        searches(
            &bm25,
            &classic,
            reader.segment_readers(),
            kind,
            spec,
            &mut got,
        );
    }
    compare("searches.tsv", &want, &got);
    eprintln!(
        "function fixtures: {} value-source specs, {} searches",
        specs.len(),
        runs.len()
    );
}

fn group_value(v: Option<&MutableValue>) -> String {
    match v {
        None => "missing".into(),
        Some(m) => fill(m),
    }
}

fn group_line(searcher: &IndexSearcher<'_, '_>, spec: &str, qs: &str) -> String {
    let head = format!("{spec}\t{qs}");
    let source = vs(spec);
    let run = || -> lucene_search::Result<String> {
        let stats = queries_stats(searcher, source.as_ref())?;
        let top = TopLevel::of_searcher(searcher, Some(&stats));
        let context = Arc::new(FunctionContext::create(source.as_ref(), &top)?);
        let gs = GroupingSearch::new(|| {
            ValueSourceGroupSelector::new(Arc::clone(&source), Arc::clone(&context))
                .with_top_level(TopLevel::of_searcher(searcher, Some(&stats)))
        })
        .set_group_docs_limit(2)
        .set_all_groups(true);
        let r = gs.search(searcher, &as_boolean(query(qs)), 0, 20)?;
        let tg = &r.top_groups;
        let mut b = format!(
            "{} {} {} {}",
            tg.total_hit_count,
            tg.total_grouped_hit_count,
            tg.total_group_count
                .map_or("null".to_string(), |c| c.to_string()),
            r.matching_groups.len()
        );
        for g in &tg.groups {
            b.push_str(&format!(" | {} =", group_value(g.group_value.as_ref())));
            for d in &g.score_docs {
                b.push_str(&format!(" {}:{}", d.doc, hex32(d.score)));
            }
        }
        Ok(b)
    };
    match run() {
        Ok(b) => format!("{head}\t{}", clean(&b)),
        Err(e) => format!("{head}\t{}", err(&e)),
    }
}

#[test]
fn value_source_grouping_matches_lucene() {
    let dir = data();
    let reader = DirectoryReader::open(&FsDirectory::open(dir.join("index"))).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let owned = reader.field_norms_by_field(&["body".to_string()]);
    let norms: Vec<SegmentNorms<'_, '_>> = owned.iter().map(Some).collect();
    let bm25 = IndexSearcher::new(&segments, &norms).unwrap();
    let text = std::fs::read_to_string(dir.join("groups.tsv")).unwrap();
    let want: Vec<&str> = text.lines().collect();
    let got: Vec<String> = want
        .iter()
        .map(|line| {
            let mut p = line.split('\t');
            group_line(&bm25, p.next().unwrap(), p.next().unwrap())
        })
        .collect();
    compare("groups.tsv", &want, &got);
}
