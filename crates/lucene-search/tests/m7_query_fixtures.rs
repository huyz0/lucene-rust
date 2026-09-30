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

use lucene_codecs::hnsw_vectors::HnswVectorsReader;
use lucene_codecs::vectors::FlatVectorsReader;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::extended_query::*;
use lucene_search::field_norms::FieldNorms;
use lucene_search::multi_segment::{
    search_boolean_query_multi_segment_with_deadline,
    search_boolean_query_multi_segment_with_similarity, OpenSegment,
};
use lucene_search::query::{
    BooleanQuery, BoostQuery, Clause, FuzzyQuery, MultiPhraseQuery, PhraseQuery, PrefixQuery,
    RegexpQuery, TermQuery, WildcardQuery,
};
use lucene_search::similarities::*;
use lucene_search::vector_query::{
    byte_vector_similarity_clause, filter_bitsets, float_vector_similarity_clause,
    knn_hits_to_clause, knn_seed_docs, search_knn_byte_vector_query_multi_segment,
    search_knn_float_vector_query_multi_segment,
    search_patience_knn_float_vector_query_multi_segment,
    search_seeded_knn_float_vector_query_multi_segment, ByteVectorSimilarityQuery,
    FloatVectorSimilarityQuery, KnnByteVectorQuery, KnnFloatVectorQuery, KnnSegment,
    PatienceKnnVectorQuery, VectorsInput,
};
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
    /// `(op, field, argument, the filter or seed clause)` to the rewritten
    /// clause.
    knn: &'a dyn Fn(&str, &str, &str, Option<Clause>) -> Option<Clause>,
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
            (ctx.knn)(op, field, arg, None)?
        }
        "VSFF" | "KFF" | "SKF" => {
            let field = next();
            let arg = next();
            let filter = parse(tok, ctx)?;
            (ctx.knn)(op, field, arg, Some(filter))?
        }
        "KFS" => {
            let field = next();
            let k = next();
            let threshold = next();
            let filter = parse(tok, ctx)?;
            (ctx.knn)(op, field, &format!("{k}:{threshold}"), Some(filter))?
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

/// `GenM7Queries.QVEC`/`QBVEC`.
const QVEC: [f32; 4] = [0.5, -0.25, 0.75, 0.1];
const QBVEC: [u8; 4] = [12, (-7i8) as u8, 30, 1];

/// Each segment's `Lucene99HnswVectorsFormat` files, read once.
struct VectorFiles {
    per_segment: Vec<[Vec<u8>; 4]>,
}

impl VectorFiles {
    fn read(dir: &std::path::Path, reader: &DirectoryReader) -> Self {
        let per_segment = reader
            .segment_readers()
            .iter()
            .map(|seg| {
                let file = |ext: &str| {
                    std::fs::read(dir.join(format!(
                        "{}_Lucene99HnswVectorsFormat_0.{ext}",
                        seg.segment_name
                    )))
                    .unwrap()
                };
                [file("vemf"), file("vec"), file("vem"), file("vex")]
            })
            .collect();
        Self { per_segment }
    }

    fn segments<'a>(
        &'a self,
        reader: &'a DirectoryReader,
        filters: Option<&'a [lucene_util::fixed_bit_set::FixedBitSet]>,
    ) -> Vec<KnnSegment<'a>> {
        const SUFFIX: &str = "Lucene99HnswVectorsFormat_0";
        reader
            .segment_readers()
            .iter()
            .zip(&self.per_segment)
            .enumerate()
            .map(|(i, (seg, [vemf, vec, vem, vex]))| {
                let id = seg.segment_id();
                KnnSegment {
                    vectors: VectorsInput {
                        flat: FlatVectorsReader::open(vemf, vec, &id, SUFFIX).unwrap(),
                        hnsw: Some(HnswVectorsReader::open(vem, vex, &id, SUFFIX).unwrap()),
                        field_infos: seg.field_infos(),
                        live_docs: seg.live_docs(),
                        filter: filters.map(|f| &f[i]),
                        max_doc: seg.max_doc,
                    },
                    doc_base: seg.doc_base,
                }
            })
            .collect()
    }
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
    let cases = check_fixture("m7_queries_index", &QVEC, 20, true);
    assert!(cases >= 200, "{cases} searches");
}

/// The vector queries over graphs large enough for filtered
/// (`FilteredHnswGraphSearcher`), patient and seeded walks to happen.
#[test]
fn m7_knn_queries_match_lucene_bit_for_bit() {
    let cases = check_fixture("m7_knn_index", &QVEC8, 50, false);
    assert!(cases >= 16, "{cases} searches");
}

/// `GenM7Queries.QVEC8`.
const QVEC8: [f32; 8] = [0.1, -0.3, 0.25, 0.6, -0.05, 0.4, -0.2, 0.15];

/// Runs every search `name`'s `searches.tsv` records and fails on the first
/// difference from Lucene's hits; returns how many ran.
fn check_fixture(name: &str, qvec: &[f32], top: usize, estimator: bool) -> usize {
    let dir = data(name);
    let text = std::fs::read_to_string(dir.join("searches.tsv"))
        .expect("run scripts/gen-fixtures.sh --only GenM7Queries");
    let reader = DirectoryReader::open(&FsDirectory::open(&dir)).unwrap();
    assert!(reader.segment_readers().len() >= 2);
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
    let vectors = VectorFiles::read(&dir, &reader);
    let knn = |op: &str, field: &str, arg: &str, clause: Option<Clause>| -> Option<Clause> {
        let filters = match (op, &clause) {
            ("KFF" | "VSFF" | "KFS", Some(f)) => Some(filter_bitsets(&segments, f).unwrap()),
            _ => None,
        };
        let knn_segments = vectors.segments(&reader, filters.as_deref());
        let float = || qvec.to_vec();
        Some(match op {
            "KF" | "KFF" => {
                let q = KnnFloatVectorQuery::new(field, float(), arg.parse().unwrap()).unwrap();
                let hits = search_knn_float_vector_query_multi_segment(&knn_segments, &q).unwrap();
                knn_hits_to_clause(&knn_segments, &hits)
            }
            "KFS" => {
                let (k, threshold) = arg.split_once(':').unwrap();
                let q = KnnFloatVectorQuery::new(field, float(), k.parse().unwrap())
                    .unwrap()
                    .with_filtered_search_threshold(threshold.parse().unwrap());
                let hits = search_knn_float_vector_query_multi_segment(&knn_segments, &q).unwrap();
                knn_hits_to_clause(&knn_segments, &hits)
            }
            "KB" => {
                let q =
                    KnnByteVectorQuery::new(field, QBVEC.to_vec(), arg.parse().unwrap()).unwrap();
                let hits = search_knn_byte_vector_query_multi_segment(&knn_segments, &q).unwrap();
                knn_hits_to_clause(&knn_segments, &hits)
            }
            "PKF" => {
                let q = KnnFloatVectorQuery::new(field, float(), arg.parse().unwrap()).unwrap();
                let q = PatienceKnnVectorQuery::from_float_query(q);
                let hits = search_patience_knn_float_vector_query_multi_segment(&knn_segments, &q)
                    .unwrap();
                knn_hits_to_clause(&knn_segments, &hits)
            }
            "SKF" => {
                let k: usize = arg.parse().unwrap();
                let q = KnnFloatVectorQuery::new(field, float(), k).unwrap();
                let seed = clause.expect("a seed query");
                let seeds =
                    knn_seed_docs(&segments, &norms, &knn_segments, field, &seed, None, k).unwrap();
                let hits =
                    search_seeded_knn_float_vector_query_multi_segment(&knn_segments, &q, &seeds)
                        .unwrap();
                knn_hits_to_clause(&knn_segments, &hits)
            }
            "VSF" | "VSFF" => {
                let q = FloatVectorSimilarityQuery::new(field, float(), arg.parse().unwrap(), 0.5)
                    .unwrap();
                float_vector_similarity_clause(&knn_segments, &q).unwrap()
            }
            "VSB" => {
                let q = ByteVectorSimilarityQuery::new(
                    field,
                    QBVEC.to_vec(),
                    arg.parse().unwrap(),
                    0.5,
                )
                .unwrap();
                byte_vector_similarity_clause(&knn_segments, &q).unwrap()
            }
            other => panic!("vector op {other}"),
        })
    };
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
            top,
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
        if name == "bm25" {
            // `TimeLimitingBulkScorer`: a deadline that never comes changes
            // nothing; one already past stops before the first window.
            let far = std::time::Instant::now() + std::time::Duration::from_secs(3600);
            let (timed, cut) =
                search_boolean_query_multi_segment_with_deadline(&segments, &bq, &norms, top, far)
                    .unwrap();
            let timed: Vec<(i32, u32)> = timed
                .iter()
                .map(|h| (h.doc_id, h.score.to_bits()))
                .collect();
            if timed != want || cut {
                failures.push(format!(
                    "deadline [{q}]\n  rust {timed:x?}\n  java {want:x?}"
                ));
            }
            let past = std::time::Instant::now();
            let (none, cut) =
                search_boolean_query_multi_segment_with_deadline(&segments, &bq, &norms, top, past)
                    .unwrap();
            assert!(none.is_empty() && cut, "an expired deadline scores nothing");
        }
    }
    // `BayesianScoreEstimator.estimate` over `body`.
    let est = if estimator {
        std::fs::read_to_string(dir.join("estimator.tsv")).unwrap()
    } else {
        String::new()
    };
    let mut estimates = 0;
    for line in est.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        let (n, tokens, seed) = (
            f[0].parse().unwrap(),
            f[1].parse().unwrap(),
            f[2].parse().unwrap(),
        );
        let p =
            lucene_search::bayesian_estimator::estimate(&segments, &norms, "body", n, tokens, seed)
                .unwrap();
        let bits = |s: &str| u32::from_str_radix(s, 16).unwrap();
        let got = (p.alpha.to_bits(), p.beta.to_bits(), p.base_rate.to_bits());
        let want = (bits(f[3]), bits(f[4]), bits(f[5]));
        if got != want {
            failures.push(format!(
                "estimate {line}\n  rust {got:x?}\n  java {want:x?}"
            ));
        }
        estimates += 1;
    }
    assert_eq!(estimates, if estimator { 3 } else { 0 });
    eprintln!("{cases} searches compared, {skipped} skipped");
    assert_eq!(skipped, 0, "every recorded search runs");
    assert!(
        failures.is_empty(),
        "{} searches differ:\n{}",
        failures.len(),
        failures.join("\n")
    );
    cases
}
