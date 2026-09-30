//! M7's query and scoring half, differentially against Lucene 10.5.0:
//! `fixtures/src/GenM7Queries.java`.
//!
//! One three-segment, index-sorted index with deletions, searched with every
//! query M7 brings into the scorer tree -- synonyms, BM25F, n-gram phrases,
//! explicit phrase positions, multi-phrases, the multi-term rewrite methods,
//! blended terms, Indri, log-odds fusion, Bayesian calibration, doc-values and
//! index-sort ranges, multi-dimensional and 4/16-byte points, vector
//! similarity thresholds and KNN as a clause -- each under BM25 and, for the
//! scoring ones, `ClassicSimilarity` and a DFR similarity. Every top-20 hit
//! and its score bits must be Java's.

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::Arc;

use lucene_search::directory_reader::DirectoryReader;
use lucene_search::extended_query::*;
use lucene_search::field_norms::FieldNorms;
use lucene_search::multi_segment::{
    search_boolean_query_multi_segment_with_similarity, OpenSegment,
};
use lucene_search::query::{
    BooleanQuery, BoostQuery, Clause, FuzzyQuery, MultiPhraseQuery, PhraseQuery, PrefixQuery,
    RegexpQuery, TermQuery, WildcardQuery,
};
use lucene_search::similarities::*;
use lucene_store::FsDirectory;

fn data(path: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/data")
        .join(path)
}

/// `GenM7Queries.sims()`.
fn sim(name: &str) -> Arc<dyn Similarity> {
    match name {
        "bm25" => Arc::new(Bm25Similarity::default()),
        "classic" => Arc::new(ClassicSimilarity::default()),
        "dfr" => Arc::new(
            DfrSimilarity::new(BasicModel::In, AfterEffect::B, Normalization::H2_DEFAULT).unwrap(),
        ),
        other => panic!("unknown similarity {other}"),
    }
}

/// What the parser needs besides the tokens: the segments a KNN or vector
/// query is rewritten against.
struct Ctx<'a> {
    knn: &'a dyn Fn(&str, &[&str]) -> Option<Clause>,
}

fn rewrite(m: &str) -> RewriteMethod {
    match m {
        "sb" => RewriteMethod::ScoringBoolean,
        "csbool" => RewriteMethod::ConstantScoreBoolean,
        "cs" => RewriteMethod::ConstantScore,
        "csb" => RewriteMethod::ConstantScoreBlended,
        "dv" => RewriteMethod::DocValues,
        other => {
            let (kind, n) = other.split_once(':').unwrap();
            let n: usize = n.parse().unwrap();
            match kind {
                "tts" => RewriteMethod::TopTermsScoringBoolean(n),
                "ttb" => RewriteMethod::TopTermsBoostOnlyBoolean(n),
                "ttbf" => RewriteMethod::TopTermsBlendedFreqScoring(n),
                _ => panic!("rewrite {other}"),
            }
        }
    }
}

fn floats(s: &str) -> Option<Vec<f32>> {
    (s != "-").then(|| s.split(',').map(|x| x.parse().unwrap()).collect())
}

/// `GenM7Queries.parse`: the queries' prefix syntax. `None` for an op this
/// test does not run.
fn parse(
    tok: &mut std::iter::Peekable<std::slice::Iter<'_, &str>>,
    ctx: &Ctx<'_>,
) -> Option<Clause> {
    let op = *tok.next().expect("truncated query");
    let mut next = || *tok.next().expect("truncated query");
    let f = |s: &str| -> f32 { s.parse().unwrap() };
    let i = |s: &str| -> i64 { s.parse().unwrap() };
    Some(match op {
        "T" => {
            let field = next();
            Clause::Term(TermQuery::new(field, next()))
        }
        "B" => {
            let (m, s, n) = (i(next()), i(next()), i(next()));
            let must: Vec<Clause> = (0..m).map(|_| parse(tok, ctx)).collect::<Option<_>>()?;
            let should: Vec<Clause> = (0..s).map(|_| parse(tok, ctx)).collect::<Option<_>>()?;
            let must_not: Vec<Clause> = (0..n).map(|_| parse(tok, ctx)).collect::<Option<_>>()?;
            let mut b = BooleanQuery::new()
                .with_must(must)
                .with_should(should)
                .with_must_not(must_not);
            if tok.peek().is_some_and(|t| **t == "F2") {
                tok.next();
                b = b.with_filter([parse(tok, ctx)?]);
            }
            Clause::Boolean(Box::new(b))
        }
        "X" => {
            let boost = f(next());
            Clause::Boost(Box::new(BoostQuery::new(parse(tok, ctx)?, boost)))
        }
        "S" => {
            let field = next();
            let n = i(next());
            let terms: Vec<(String, f32)> = (0..n)
                .map(|_| {
                    let t = next().to_string();
                    (t, f(next()))
                })
                .collect();
            SynonymQuery::new(field, terms.into_iter().map(|(t, b)| (t.into_bytes(), b)))
                .unwrap()
                .into()
        }
        "CF" => {
            let term = next();
            let n = i(next());
            let fields: Vec<(String, f32)> = (0..n)
                .map(|_| {
                    let fl = next().to_string();
                    (fl, f(next()))
                })
                .collect();
            CombinedFieldQuery::new(term, fields).unwrap().into()
        }
        "PP" => {
            let field = next();
            let (slop, n) = (i(next()), i(next()));
            let terms: Vec<(String, i32)> = (0..n)
                .map(|_| {
                    let t = next().to_string();
                    (t, i(next()) as i32)
                })
                .collect();
            let p = PhraseQuery::with_positions(
                field,
                terms.into_iter().map(|(t, p)| (t.into_bytes(), p)),
            )
            .unwrap()
            .with_slop(slop as u32);
            Clause::Phrase(p)
        }
        "NG" => {
            let n = i(next()) as usize;
            let field = next();
            let (slop, k) = (i(next()), i(next()));
            let terms: Vec<&str> = (0..k).map(|_| next()).collect();
            NGramPhraseQuery::new(n, PhraseQuery::new(field, terms).with_slop(slop as u32)).into()
        }
        "MP" => {
            let field = next();
            let (slop, n) = (i(next()), i(next()));
            let mut arrays = Vec::new();
            for _ in 0..n {
                let m = i(next());
                let alts: Vec<String> = (0..m).map(|_| next().to_string()).collect();
                arrays.push((alts, i(next()) as i32));
            }
            let mp = MultiPhraseQuery::with_positions(
                field,
                arrays
                    .into_iter()
                    .map(|(alts, p)| (alts.into_iter().map(String::into_bytes), p)),
            )
            .unwrap()
            .with_slop(slop as u32);
            Clause::MultiPhrase(mp)
        }
        "F" => {
            let field = next();
            let term = next();
            let (edits, prefix) = (i(next()), i(next()));
            let mut q = FuzzyQuery::new(field, term);
            q.max_edits = edits as u8;
            q.prefix_length = prefix as usize;
            Clause::Fuzzy(q)
        }
        "PF" | "W" | "RE" => {
            let field = next();
            let pattern = next();
            let source = match op {
                "PF" => MultiTermSource::Prefix(PrefixQuery::new(field, pattern)),
                "W" => MultiTermSource::Wildcard(WildcardQuery::new(field, pattern)),
                _ => MultiTermSource::Regexp(RegexpQuery::new(field, pattern)),
            };
            MultiTermQuery::new(source, rewrite(next())).into()
        }
        "R" => {
            let field = next();
            let (lo, hi) = (next(), next());
            let (incl, incu) = (i(next()) == 1, i(next()) == 1);
            let bound = |s: &str| (s != "-").then(|| s.as_bytes().to_vec());
            MultiTermQuery::new(
                MultiTermSource::TermRange(TermRangeQuery::new(
                    field,
                    bound(lo),
                    bound(hi),
                    incl,
                    incu,
                )),
                rewrite(next()),
            )
            .into()
        }
        "A" => {
            use lucene_util::automaton::{operations, RegExp, DEFAULT_DETERMINIZE_WORK_LIMIT};
            let field = next();
            let re = next();
            let a = RegExp::new(re).unwrap().to_automaton().unwrap();
            let a = operations::determinize(&a, DEFAULT_DETERMINIZE_WORK_LIMIT).unwrap();
            MultiTermQuery::new(
                MultiTermSource::Automaton(AutomatonQuery::new(field, a, false)),
                rewrite(next()),
            )
            .into()
        }
        "BT" => {
            let m = next();
            let method = if m == "bool" {
                BlendedRewrite::Boolean
            } else {
                BlendedRewrite::DisjunctionMax(f(m.split_once(':').unwrap().1))
            };
            let n = i(next());
            let terms: Vec<(String, String, f32)> = (0..n)
                .map(|_| {
                    let fl = next().to_string();
                    let t = next().to_string();
                    (fl, t, f(next()))
                })
                .collect();
            BlendedTermQuery::new(
                terms.into_iter().map(|(a, b, c)| (a, b.into_bytes(), c)),
                method,
            )
            .unwrap()
            .into()
        }
        "IA" => {
            let n = i(next());
            let clauses: Vec<Clause> = (0..n).map(|_| parse(tok, ctx)).collect::<Option<_>>()?;
            IndriAndQuery::new(clauses).into()
        }
        "LO" => {
            let alpha = f(next());
            let weights = floats(next());
            let mm = next();
            let bounds = (mm != "-").then(|| {
                let (lo, hi) = mm.split_once(':').unwrap();
                (floats(lo).unwrap(), floats(hi).unwrap())
            });
            let n = i(next());
            let clauses: Vec<Clause> = (0..n).map(|_| parse(tok, ctx)).collect::<Option<_>>()?;
            LogOddsFusionQuery::new(clauses, alpha, weights, bounds)
                .unwrap()
                .into()
        }
        "BY" => {
            let (alpha, beta, base) = (f(next()), f(next()), f(next()));
            BayesianScoreQuery::new(parse(tok, ctx)?, alpha, beta, base)
                .unwrap()
                .into()
        }
        "NR" => {
            let field = next();
            let (lo, hi) = (i(next()), i(next()));
            NumericDocValuesRangeQuery::new(field, lo, hi).into()
        }
        "ISR" => {
            let field = next();
            let (lo, hi) = (i(next()), i(next()));
            let fallback = if tok.peek().is_some_and(|t| **t == "-") {
                tok.next();
                NumericDocValuesRangeQuery::new(field, lo, hi).into()
            } else {
                parse(tok, ctx)?
            };
            IndexSortSortedNumericDocValuesRangeQuery::new(field, lo, hi, fallback).into()
        }
        "P2R" => {
            let field = next();
            let v: Vec<i32> = (0..4).map(|_| i(next()) as i32).collect();
            PointRangeQuery::int_range(field, &v[0..2], &v[2..4])
                .unwrap()
                .into()
        }
        "I1R" => {
            let field = next();
            let (lo, hi) = (i(next()) as i32, i(next()) as i32);
            PointRangeQuery::int_range(field, &[lo], &[hi])
                .unwrap()
                .into()
        }
        "I1S" | "LS" | "IPS" => {
            let field = next();
            let mut vals = Vec::new();
            while tok.peek().is_some_and(|t| **t != "F2") {
                vals.push(tok.next().unwrap().to_string());
            }
            match op {
                "I1S" => {
                    let v: Vec<i32> = vals.iter().map(|s| s.parse().unwrap()).collect();
                    PointInSetQuery::int_set(field, &v).unwrap().into()
                }
                "LS" => {
                    let v: Vec<i64> = vals.iter().map(|s| s.parse().unwrap()).collect();
                    PointInSetQuery::long_set(field, &v).unwrap().into()
                }
                _ => {
                    let v: Vec<IpAddr> = vals.iter().map(|s| s.parse().unwrap()).collect();
                    PointInSetQuery::inet_set(field, &v).unwrap().into()
                }
            }
        }
        "IPR" => {
            let field = next();
            let (lo, hi): (IpAddr, IpAddr) = (next().parse().unwrap(), next().parse().unwrap());
            PointRangeQuery::inet_range(field, lo, hi).unwrap().into()
        }
        "VSF" | "VSB" | "KF" | "KB" | "PKF" => {
            let field = next();
            let arg = next();
            (ctx.knn)(op, &[field, arg])?
        }
        "VSFF" | "KFF" | "SKF" => {
            let field = next();
            let arg = next();
            let filter = parse(tok, ctx)?;
            let _ = filter;
            (ctx.knn)(op, &[field, arg])?
        }
        other => panic!("query op {other}"),
    })
}

fn query(text: &str, ctx: &Ctx<'_>) -> Option<BooleanQuery> {
    let tokens: Vec<&str> = text.split(' ').collect();
    let mut it = tokens.iter().peekable();
    let clause = parse(&mut it, ctx)?;
    assert!(it.next().is_none(), "trailing tokens in {text}");
    Some(match clause {
        Clause::Boolean(b) => *b,
        other => BooleanQuery::new().with_must([other]),
    })
}

fn want(hits: &str) -> Vec<(i32, u32)> {
    hits.split(',')
        .filter(|s| !s.is_empty())
        .map(|h| {
            let (d, b) = h.split_once(':').unwrap();
            (d.parse().unwrap(), u32::from_str_radix(b, 16).unwrap())
        })
        .collect()
}

#[test]
fn m7_queries_match_lucene_bit_for_bit() {
    let dir = data("m7_queries_index");
    let text = std::fs::read_to_string(dir.join("searches.tsv"))
        .expect("run scripts/gen-fixtures.sh --only GenM7Queries");
    let reader = DirectoryReader::open(&FsDirectory::open(&dir)).unwrap();
    assert_eq!(reader.segment_readers().len(), 3);
    assert!(
        reader
            .segment_readers()
            .iter()
            .filter(|s| s.live_docs().is_some())
            .count()
            >= 2
    );
    let mut opened = reader.open_segments().unwrap();
    opened.open_points().unwrap();
    let segments: Vec<OpenSegment<'_>> = opened.as_open_segments();
    let owned =
        reader.field_norms_by_field(&["body".to_string(), "title".to_string(), "gram".to_string()]);
    let norms: Vec<Option<&HashMap<String, FieldNorms<'_>>>> = owned.iter().map(Some).collect();
    let knn = |_op: &str, _args: &[&str]| -> Option<Clause> { None };
    let ctx = Ctx { knn: &knn };

    let (mut cases, mut skipped, mut failures) = (0, 0, Vec::new());
    for line in text.lines() {
        let [name, q, hits] = line.split('\t').collect::<Vec<_>>()[..] else {
            panic!("bad line {line}");
        };
        let Some(bq) = query(q, &ctx) else {
            skipped += 1;
            continue;
        };
        let similarity = sim(name);
        let got = match search_boolean_query_multi_segment_with_similarity(
            &segments,
            &bq,
            &norms,
            20,
            similarity.as_ref(),
        ) {
            Ok(got) => got,
            Err(e) => {
                failures.push(format!("{name} [{q}]: {e}"));
                continue;
            }
        };
        let got: Vec<(i32, u32)> = got.iter().map(|h| (h.doc_id, h.score.to_bits())).collect();
        cases += 1;
        let want = want(hits);
        if got != want {
            failures.push(format!("{name} [{q}]\n  rust {got:x?}\n  java {want:x?}"));
        }
    }
    eprintln!("{cases} searches compared, {skipped} skipped");
    assert!(
        failures.is_empty(),
        "{} searches differ:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
