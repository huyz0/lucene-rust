//! The `painless_score` tests that run through the exported entry points.

#[cfg(test)]
pub(crate) mod tests {
    use crate::engine_tests::nodes::tests::{open, Blob};
    use crate::jvm_reader::ffi_close_jvm_reader;
    use crate::jvm_reader::tests::run_limit;

    use lucene_engine::painless_score::*;
    use lucene_engine::query_blob::Cursor;
    use lucene_engine::status::Status;

    use lucene_painless::value::Value;
    use lucene_search::explain::Explanation;

    use lucene_search::similarities::{SimScorer, Similarity};
    use lucene_search::values_source::{LongValuesSource, ValuesContext};
    use lucene_search::Error;

    use lucene_util::test_support::TempDir;

    use std::sync::Arc;

    /// The fixture's cases, read from `fixtures/data/painless-score/expected.bin`
    /// (`opensearch-plugin/tools/painless-corpus/ScoreCorpus.java`).
    struct Case {
        source: String,
        /// The node, when the plugin would send one.
        node: Option<Blob>,
        /// The node's fields after its sub-query: what the plugin checks.
        spec: Vec<u8>,
        /// The sub-query, and the boost over the node (1 for none).
        sub: Blob,
        boost: f32,
        /// A `script` query's filter script, without a sub-query.
        filter: bool,
        /// A number sort script: the sort blob, over every document.
        sort: Option<Vec<u8>>,
        /// OpenSearch's hits by doc, or the exception it threw.
        outcome: Outcome,
    }

    #[derive(Debug)]
    enum Outcome {
        Hits(Vec<(i32, f32)>),
        /// The class `ScriptException` wraps, its message, the offset it
        /// points at.
        Threw(String, Option<String>, i32),
        /// `ScriptScorer`'s own error, a negative or NaN score, or
        /// `NonNegativeScoresSimilarity`'s: its message.
        Refused(String),
        /// A sort script's hits, in the sort's order, with their sortable
        /// longs.
        Sorted(Vec<(i32, i64)>),
    }

    struct Reader<'a>(&'a [u8]);

    impl Reader<'_> {
        fn u8(&mut self) -> u8 {
            let v = self.0[0];
            self.0 = &self.0[1..];
            v
        }
        fn i32(&mut self) -> i32 {
            let v = i32::from_le_bytes(self.0[..4].try_into().unwrap());
            self.0 = &self.0[4..];
            v
        }
        fn bytes(&mut self) -> Vec<u8> {
            let n = self.i32() as usize;
            let v = self.0[..n].to_vec();
            self.0 = &self.0[n..];
            v
        }
        fn str(&mut self) -> String {
            String::from_utf8(self.bytes()).unwrap()
        }
    }

    fn fixture(name: &str) -> String {
        format!(
            "{}/../../fixtures/data/painless-score/{name}",
            env!("CARGO_MANIFEST_DIR")
        )
    }

    fn cases() -> Vec<Case> {
        let data = std::fs::read(fixture("expected.bin")).unwrap();
        assert_eq!(&data[..4], b"PSC2");
        let mut r = Reader(&data[4..]);
        let n = r.i32();
        (0..n)
            .map(|_| {
                let source = r.str();
                let kind = r.u8();
                let filter = kind == 2;
                let sorting = kind == 3;
                let sub = match kind {
                    0 => Blob::default().term("body", "alpha"),
                    1 => Blob::default().u8(5),
                    _ => Blob::default(),
                };
                let boost = f32::from_bits(r.i32() as u32);
                assert_eq!(r.u8(), 1, "{source}: every fixture script compiles to IR");
                let ir = r.bytes();
                let params = r.bytes();
                let needs = r.u8();
                let mut spec = Blob::default()
                    .u8(needs)
                    .bytes(&ir)
                    .bytes(&params)
                    .str("idx")
                    .i32(3)
                    .i32(0);
                let fields = r.i32();
                spec = spec.i32(fields);
                for _ in 0..fields {
                    let name = r.str();
                    spec = spec.str(&name).str(&name).u8(r.u8());
                }
                let mut b = Blob::tree();
                if boost != 1.0 {
                    b = b.u8(3).f32(boost);
                }
                let sort = sorting.then(|| {
                    let reverse = u8::from(boost < 0.0);
                    let mut b = vec![1, crate::jvm_reader::SORT_SCRIPT, reverse];
                    b.extend_from_slice(&spec.0);
                    // No search_after, no options, no slices.
                    b.extend_from_slice(&[0, 0, 0, 0, 0, 0]);
                    b
                });
                let node = Some(if sorting {
                    Blob::tree().u8(5)
                } else if filter {
                    b.u8(NODE_PAINLESS_FILTER).raw(&spec)
                } else {
                    b.u8(NODE_PAINLESS_SCORE).raw(&sub).raw(&spec)
                });
                let case_sub = sub.clone();
                let outcome = match r.u8() {
                    0 => {
                        let n = r.i32();
                        Outcome::Hits(
                            (0..n)
                                .map(|_| (r.i32(), f32::from_bits(r.i32() as u32)))
                                .collect(),
                        )
                    }
                    3 => {
                        let n = r.i32();
                        let mut v: Vec<(i32, i64)> = (0..n)
                            .map(|_| {
                                let doc = r.i32();
                                let lo = r.i32() as u32 as u64;
                                let hi = r.i32() as u32 as u64;
                                let d = f64::from_bits(lo | hi << 32);
                                (doc, lucene_util::numeric_utils::double_to_sortable_long(d))
                            })
                            .collect();
                        // TopFieldCollector's order: the key, then the doc.
                        v.sort_by(|a, b| {
                            let k = if boost < 0.0 {
                                b.1.cmp(&a.1)
                            } else {
                                a.1.cmp(&b.1)
                            };
                            k.then(a.0.cmp(&b.0))
                        });
                        Outcome::Sorted(v)
                    }
                    kind => {
                        let class = r.str();
                        let message = (r.u8() == 1).then(|| r.str());
                        let offset = r.i32();
                        if kind == 1 {
                            Outcome::Threw(class, message, offset)
                        } else {
                            assert_eq!(class, "java.lang.IllegalArgumentException");
                            Outcome::Refused(message.unwrap())
                        }
                    }
                };
                Case {
                    source,
                    node,
                    spec: spec.0,
                    sub: case_sub,
                    boost,
                    filter,
                    sort,
                    outcome,
                }
            })
            .collect()
    }

    /// A similarity script's spec of `context`: no params, no fields.
    fn similarity_spec_blob(context: u8, ir: &[u8], run: i32) -> Blob {
        Blob::default()
            .u8(context)
            .bytes(ir)
            .bytes(&[12, 0, 0, 0, 0])
            .str("")
            .i32(0)
            .i32(run)
            .i32(0)
    }

    /// The query of `ScoreCorpus.simQuery(kind)`, behind a similarity table
    /// giving `body` the scripted similarity `sim`.
    fn similarity_query(sim: &Blob, kind: u8) -> Blob {
        let table = Blob::default()
            .u8(lucene_engine::similarity::QUERY_SIMILARITY)
            .i32(1)
            .str("body")
            .u8(9)
            .raw(sim);
        let tree = Blob::tree();
        let tree = match kind {
            0 => tree.term("body", "alpha"),
            1 => tree
                .u8(1)
                .i32(0)
                .i32(3)
                .u8(2)
                .term("body", "alpha")
                .u8(2)
                .term("body", "beta")
                .u8(2)
                .term("body", "gamma"),
            2 => tree.u8(3).f32(2.5).term("body", "delta"),
            _ => tree
                .u8(7)
                .str("body")
                .i32(0)
                .i32(2)
                .i32(0)
                .str("beta")
                .i32(1)
                .str("alpha"),
        };
        table.raw(&tree)
    }

    /// **Scripted similarity.** Each fixture similarity (a score script,
    /// with or without a weight script) scores a term, a disjunction, a
    /// boosted term and a phrase (several terms' statistics, summed) as
    /// OpenSearch's `ScriptedSimilarity` does, bit for bit, or fails with
    /// the same exception at the same offset, naming the arguments it ran
    /// on.
    #[test]
    fn scripted_similarities_score_as_opensearch_does() {
        let tmp = TempDir::new("painless-similarity");
        for e in std::fs::read_dir(fixture("index")).unwrap() {
            let e = e.unwrap();
            std::fs::copy(e.path(), tmp.path().join(e.file_name())).unwrap();
        }
        let h = open(&tmp);
        let data = std::fs::read(fixture("similarity.bin")).unwrap();
        assert_eq!(&data[..4], b"PSS1");
        let mut r = Reader(&data[4..]);
        let n = r.i32();
        let (mut same, mut wrong) = (0, Vec::new());
        for _ in 0..n {
            let source = r.str();
            let ir = r.bytes();
            let weight = (r.u8() == 1).then(|| {
                let _source = r.str();
                r.bytes()
            });
            let description = r.str();
            // Wrapped in an enforcing NonNegativeScoresSimilarity, as OpenSearch builds it.
            let mut sim = Blob::default().u8(1).u8(1).str(&description);
            sim = match &weight {
                Some(w) => sim.u8(1).raw(&similarity_spec_blob(9, w, 1)),
                None => sim.u8(0),
            };
            let sim = sim.raw(&similarity_spec_blob(8, &ir, 0));
            for kind in 0..4u8 {
                let outcome = match r.u8() {
                    0 => {
                        let n = r.i32();
                        Outcome::Hits(
                            (0..n)
                                .map(|_| (r.i32(), f32::from_bits(r.i32() as u32)))
                                .collect(),
                        )
                    }
                    k => {
                        let class = r.str();
                        let message = (r.u8() == 1).then(|| r.str());
                        let offset = r.i32();
                        if k == 2 {
                            assert_eq!(class, "java.lang.IllegalArgumentException", "{source}");
                            Outcome::Refused(message.expect("a message"))
                        } else {
                            Outcome::Threw(class, message, offset)
                        }
                    }
                };
                let got = run_limit(h, &similarity_query(&sim, kind).0, 1000, i64::MAX);
                let e = lucene_engine::status::last_error();
                let bad = match (&outcome, got) {
                    (Outcome::Hits(want), Ok((mut hits, _, _))) => {
                        hits.sort_by_key(|&(d, _)| d);
                        let bits = |v: &[(i32, f32)]| -> Vec<(i32, u32)> {
                            v.iter().map(|&(d, s)| (d, s.to_bits())).collect()
                        };
                        (bits(&hits) != bits(want)).then(|| format!("hits {hits:?}, want {want:?}"))
                    }
                    (Outcome::Threw(class, message, offset), Err(_)) => {
                        let message = match message {
                            Some(m) => format!("1{m}"),
                            None => "0".to_string(),
                        };
                        let run = if weight.is_some() && source == "return weight;" {
                            1
                        } else {
                            0
                        };
                        let script = format!("{SCRIPT_ERROR}{run}\u{1}0\u{1}");
                        let want = format!("\u{1}{offset}\u{1}{class}\u{1}{message}");
                        (!e.contains(&script) || !e.ends_with(&want))
                            .then(|| format!("{e:?}, want {script:?}..{want:?}"))
                    }
                    (Outcome::Refused(message), Err(_)) => {
                        // A disjunction visits its clauses in another order than
                        // Lucene's bulk scorer, so it can name another negative doc.
                        let want = if kind == 1 {
                            let first = message.split_inclusive('\n').next().unwrap_or("");
                            format!("{SCORE_ERROR}{first}-")
                        } else {
                            format!("{SCORE_ERROR}{message}")
                        };
                        let got = &e[e.find(SCORE_ERROR).unwrap_or(0)..];
                        let same = if kind == 1 {
                            got.starts_with(&want)
                        } else {
                            got == want
                        };
                        (!same).then(|| format!("{e:?}, want {want:?}"))
                    }
                    (o, r) => Some(format!("{o:?} vs {r:?}: {e}")),
                };
                match bad {
                    Some(bad) => wrong.push(format!("{source} (query {kind}): {bad}")),
                    None => same += 1,
                }
            }
        }
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
        assert_eq!(same, n * 4);
        assert_eq!(ffi_close_jvm_reader(h), 0);
    }

    /// A scorer of a constant score, under [`NonNegativeScorer`].
    #[derive(Debug)]
    struct Constant(f32);

    impl SimScorer for Constant {
        fn score(&self, _: f32, _: i64) -> f32 {
            self.0
        }
    }

    /// The fixture's first scripted similarity, its negative scores `mode`.
    fn first_fixture_similarity(mode: u8) -> std::result::Result<ScriptedSimilarity, Status> {
        let data = std::fs::read(fixture("similarity.bin")).unwrap();
        let mut r = Reader(&data[8..]);
        let _source = r.str();
        let ir = r.bytes();
        assert_eq!(r.u8(), 0, "the first case has no weight script");
        let blob = Blob::default()
            .u8(1)
            .u8(mode)
            .str("d")
            .u8(0)
            .raw(&similarity_spec_blob(8, &ir, 0));
        ScriptedSimilarity::decode(&mut Cursor::new(&blob.0))
    }

    /// **Two failed similarities.** Of two fields' scripted similarities
    /// that both failed, the search reports the one that failed first,
    /// whichever field it is.
    #[test]
    fn the_first_similarity_failure_is_reported() {
        use lucene_search::similarities::{PerFieldSimilarity, Similarity};
        for first_is_default in [true, false] {
            let (a, b) = (
                first_fixture_similarity(0).unwrap(),
                first_fixture_similarity(0).unwrap(),
            );
            let (first, second) = if first_is_default { (&a, &b) } else { (&b, &a) };
            first.0.fail(Error::IllegalArgument("first".into()));
            second.0.fail(Error::IllegalArgument("second".into()));
            let per =
                PerFieldSimilarity::new(Arc::new(a.clone())).with_field("x", Arc::new(b.clone()));
            assert!(per.failure_order().is_some());
            assert!(per.take_failure().unwrap().to_string().contains("first"));
        }
    }

    /// **`ScriptedSimilarity.computeNorm`.** `intToByte4` of the length,
    /// less the overlaps when they are discounted, a `DOCS` field too; the
    /// byte sign-extended as Java's `(byte)` then `long`.
    #[test]
    fn scripted_norms_as_opensearch_computes_them() {
        use lucene_index::similarity::{FieldInvertState, NormSimilarity};
        let state = |docs_only, length, num_overlap| FieldInvertState {
            docs_only,
            length,
            num_overlap,
            ..FieldInvertState::default()
        };
        let byte4 = |n: u32| i64::from(lucene_util::small_float::int_to_byte4(n) as i8);
        let discounting = first_fixture_similarity(0).unwrap();
        assert!(discounting.discount_overlaps());
        assert_eq!(discounting.compute_norm("f", &state(false, 7, 2)), byte4(5));
        assert_eq!(discounting.compute_norm("f", &state(true, 7, 2)), byte4(5));
        // A length whose byte is negative as a Java byte.
        assert!(byte4(1_000_000) < 0);
        assert_eq!(
            discounting.compute_norm("f", &state(false, 1_000_000, 0)),
            byte4(1_000_000)
        );
        let mut sim = first_fixture_similarity(0).unwrap();
        Arc::get_mut(&mut sim.0).unwrap().discount_overlaps = false;
        assert!(!sim.discount_overlaps());
        assert_eq!(sim.compute_norm("f", &state(false, 7, 2)), byte4(7));
    }

    /// **`NonNegativeScoresSimilarity`.** A negative score is `0` when not
    /// enforced and fails the search when enforced, its explanation the
    /// message; `NaN` and `-0.0` pass; the explanation of a negative score is
    /// the `max of` it and `0`.
    #[test]
    fn negative_scores_as_non_negative_scores_similarity() {
        let wrap = |mode, score| {
            let sim = first_fixture_similarity(mode).unwrap();
            NonNegativeScorer {
                inner: Arc::new(Constant(score)),
                shared: sim.0.clone(),
            }
        };
        let zero = wrap(2, -1.5);
        assert_eq!(zero.score(1.0, 1).to_bits(), 0.0f32.to_bits());
        assert!(ScriptedSimilarity(zero.shared.clone())
            .take_failure()
            .is_none());
        assert_eq!(wrap(2, -0.0).score(1.0, 1).to_bits(), (-0.0f32).to_bits());
        assert!(wrap(1, f32::NAN).score(1.0, 1).is_nan());
        let e = zero.explain(Explanation::match_(1.0, "freq"), 1);
        assert_eq!(
            e.to_string(),
            "0.0 = max of:\n  -1.5 = score(freq=1.0), with freq of:\n    1.0 = freq\n  0.0 = Minimum allowed score\n"
        );
        assert_eq!(
            wrap(2, 2.0)
                .explain(Explanation::match_(1.0, "freq"), 1)
                .value,
            2.0
        );

        let fail = wrap(1, -1.5);
        assert_eq!(fail.score(3.0, 1).to_bits(), 0.0f32.to_bits());
        let msg = ScriptedSimilarity(fail.shared.clone())
            .take_failure()
            .unwrap()
            .to_string();
        assert!(
            msg.ends_with(&format!(
                "{SCORE_ERROR}Similarities must not produce negative scores, but got:\n-1.5 = score(freq=3.0), with freq of:\n  3.0 = term frequency\n"
            )),
            "{msg}"
        );

        for mode in [0u8, 1, 2] {
            assert!(first_fixture_similarity(mode).is_ok());
        }
        assert!(first_fixture_similarity(3).is_err());
    }

    /// Scripts the check leaves to Java, before any of them runs.
    const IN_JAVA: &[&str] = &[
        // A date field's `JodaCompatibleZonedDateTime`.
        "doc['d'].value.toInstant().toEpochMilli() / 1e12",
        // `_source`.
        "params._source.size()",
        // Streams.
        "doc['vals'].stream().mapToLong(x -> x).sum()",
        // A field Painless does not see named: `params.doc[...]`.
        "params.doc['likes'].value + params._doc['flag'].value.compareTo(true) + 3",
        // A collection of `params` changed, which every segment's script shares.
        "params.weights.add(1); return params.weights.size()",
        // `params` changed through a second variable.
        "def p = params; def q = p; p['w'] = 5; return q.w",
        // A live view (`keySet`) changed.
        "Map m = ['a': 1, 'b': 2]; m.keySet().remove('a'); return m.size()",
    ];

    /// Every script the check passes runs as OpenSearch's does: the same
    /// hits and scores, or the same exception at the same offset, as a
    /// `script_score` query or, for a filter script, a `script` query. The ones
    /// it refuses never reach the native runtime: a node carrying one is
    /// refused before any document is read.
    /// **A count alone.** A `size: 0` search over a filter script counts
    /// what the search with hits matches: the count prepares the script over
    /// every segment as the search does, or fails as it does.
    #[test]
    fn filter_scripts_count_as_their_searches_match() {
        let tmp = TempDir::new("painless-count");
        for e in std::fs::read_dir(fixture("index")).unwrap() {
            let e = e.unwrap();
            std::fs::copy(e.path(), tmp.path().join(e.file_name())).unwrap();
        }
        let h = open(&tmp);
        let mut counted = 0;
        for case in cases().iter().filter(|c| c.filter) {
            let node = case.node.as_ref().unwrap();
            let searched = run_limit(h, &node.0, 1000, i64::MAX);
            let count = run_limit(h, &node.0, 0, i64::MAX);
            match (searched, count) {
                (Ok((_, want, _)), Ok((hits, total, _))) => {
                    assert!(hits.is_empty(), "{}", case.source);
                    assert_eq!(total, want, "{}", case.source);
                    counted += 1;
                }
                (Err(_), Err(_)) => {}
                (a, b) => panic!(
                    "{}: {a:?} vs {b:?}: {}",
                    case.source,
                    lucene_engine::status::last_error()
                ),
            }
        }
        assert!(counted > 0);
        assert_eq!(ffi_close_jvm_reader(h), 0);
    }

    #[test]
    fn scripts_score_as_opensearch_does() {
        let tmp = TempDir::new("painless-score");
        for e in std::fs::read_dir(fixture("index")).unwrap() {
            let e = e.unwrap();
            std::fs::copy(e.path(), tmp.path().join(e.file_name())).unwrap();
        }
        let h = open(&tmp);
        let (mut same, mut java) = (0, 0);
        let mut wrong = Vec::new();
        let cases = cases();
        for case in &cases {
            let node = case.node.as_ref().unwrap();
            let refused = check(&case.spec).unwrap();
            if let Some(sort) = &case.sort {
                assert!(refused.is_none(), "{}: refused: {refused:?}", case.source);
                let got = crate::jvm_reader::tests::run_sorted(h, &node.0, sort, 1000, i64::MAX);
                let e = lucene_engine::status::last_error();
                let bad = match (&case.outcome, got) {
                    (Outcome::Sorted(want), Ok((hits, _, _))) => {
                        let hits: Vec<(i32, i64)> = hits.iter().map(|(d, v)| (*d, v[0])).collect();
                        (&hits != want).then(|| format!("sorted {hits:?}, want {want:?}"))
                    }
                    (Outcome::Threw(class, _, offset), Err(_)) => {
                        let want = format!("\u{1}{offset}\u{1}{class}\u{1}");
                        (!e.contains(SCRIPT_ERROR) || !e.contains(&want) || !e.contains("\u{1}b"))
                            .then(|| format!("{e:?}, want ..{want:?}"))
                    }
                    (o, r) => Some(format!("{o:?} vs {r:?}: {e}")),
                };
                if let Some(bad) = bad {
                    wrong.push(format!("{}: {bad}", case.source));
                    continue;
                }
                same += 1;
                continue;
            }
            let got = run_limit(h, &node.0, 1000, i64::MAX);
            if let Some(why) = refused {
                if !IN_JAVA.contains(&case.source.as_str()) {
                    wrong.push(format!("{}: refused: {why}", case.source));
                    continue;
                }
                let e = lucene_engine::status::last_error();
                assert!(
                    got.is_err() && e.contains("not checked"),
                    "{}: {e}",
                    case.source
                );
                java += 1;
                continue;
            }
            assert!(
                !IN_JAVA.contains(&case.source.as_str()),
                "{}: passed the check",
                case.source
            );
            let e = lucene_engine::status::last_error();
            let bad = match (&case.outcome, got) {
                (Outcome::Hits(want), Ok((mut hits, _, _))) => {
                    hits.sort_by_key(|&(d, _)| d);
                    let bits = |v: &[(i32, f32)]| -> Vec<(i32, u32)> {
                        v.iter().map(|&(d, s)| (d, s.to_bits())).collect()
                    };
                    (bits(&hits) != bits(want)).then(|| format!("hits {hits:?}, want {want:?}"))
                }
                (Outcome::Hits(_) | Outcome::Sorted(_), Err(_)) => Some(format!("failed: {e:?}")),
                (Outcome::Threw(class, message, offset), Err(_)) => {
                    let message = match message {
                        Some(m) => format!("1{m}"),
                        None => "0".to_string(),
                    };
                    // Then the segment, document and score it threw at.
                    let script = format!("{SCRIPT_ERROR}0\u{1}0\u{1}");
                    let want = format!("\u{1}{offset}\u{1}{class}\u{1}{message}");
                    (!e.contains(&script) || !e.ends_with(&want))
                        .then(|| format!("{e:?}, want {script:?}..{want:?}"))
                }
                (Outcome::Refused(message), Err(_)) => {
                    let want = format!("{SCORE_ERROR}{message}");
                    (!e.ends_with(&want)).then(|| format!("{e:?}, want {want:?}"))
                }
                (_, Ok(r)) => Some(format!("OpenSearch threw, native scored {r:?}")),
            };
            if let Some(bad) = bad {
                wrong.push(format!("{}: {bad}", case.source));
                continue;
            }
            same += 1;
        }
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
        assert_eq!(java, IN_JAVA.len());
        assert_eq!(same + java, cases.len());
        assert_eq!(ffi_close_jvm_reader(h), 0);
    }

    /// **`function_score` script functions.** Each fixture script the check
    /// passes, as the one function of a `function_score` that replaces the
    /// score: the same hits and scores as the script's `script_score` (both
    /// narrow the script's `double` to the hit's `float`), the same
    /// exception at the same offset. A negative score fails as
    /// `ScriptScoreFunction` does; `NaN` passes the function and fails the
    /// query.
    #[test]
    fn script_functions_score_as_script_score_does() {
        use lucene_engine::nodes::NODE_FUNCTION_SCORE;
        let tmp = TempDir::new("painless-score-function");
        for e in std::fs::read_dir(fixture("index")).unwrap() {
            let e = e.unwrap();
            std::fs::copy(e.path(), tmp.path().join(e.file_name())).unwrap();
        }
        let h = open(&tmp);
        let function = |sub: &Blob, spec: &[u8]| {
            Blob::tree()
                .u8(NODE_FUNCTION_SCORE)
                .u8(1) // replace
                .u8(0) // first
                .f32(f32::MAX)
                .raw(sub)
                .i32(1)
                .u8(0)
                .u8(5)
                .raw(&Blob(spec.to_vec()))
        };
        let mut wrong = Vec::new();
        let mut ran = 0;
        for case in cases() {
            if case.filter
                || case.sort.is_some()
                || case.boost != 1.0
                || check(&case.spec).unwrap().is_some()
            {
                continue;
            }
            let got = run_limit(h, &function(&case.sub, &case.spec).0, 1000, i64::MAX);
            let e = lucene_engine::status::last_error();
            let bad = match (&case.outcome, got) {
                (Outcome::Hits(want), Ok((mut hits, _, _))) => {
                    hits.sort_by_key(|&(d, _)| d);
                    let bits = |v: &[(i32, f32)]| -> Vec<(i32, u32)> {
                        v.iter().map(|&(d, s)| (d, s.to_bits())).collect()
                    };
                    (bits(&hits) != bits(want)).then(|| format!("hits {hits:?}, want {want:?}"))
                }
                (Outcome::Threw(class, _, offset), Err(_)) => {
                    let want = format!("\u{1}{offset}\u{1}{class}\u{1}");
                    (!e.contains(SCRIPT_ERROR) || !e.contains(&want)).then(|| format!("{e:?}"))
                }
                (Outcome::Refused(message), Err(_)) => {
                    // ScriptScoreQuery's check, not the function's: the
                    // function passes NaN, and the query refuses it.
                    let ok = if message.contains("[NaN]") {
                        e.contains(OPENSEARCH_ERROR) && e.contains("invalid score: NaN for doc")
                    } else {
                        e.contains(SCORE_ERROR)
                            && e.contains("script score function must not produce negative scores")
                    };
                    (!ok).then(|| format!("{e:?}"))
                }
                (o, r) => Some(format!("{o:?} vs {r:?}: {e}")),
            };
            if let Some(bad) = bad {
                wrong.push(format!("{}: {bad}", case.source));
            }
            ran += 1;
        }
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
        assert!(ran > 30, "{ran} scripts run");
        assert_eq!(ffi_close_jvm_reader(h), 0);
    }

    /// **Aggregation scripts.** Each fixture sort script, run as a metric
    /// aggregation's values: each document's one value is the double the
    /// sort read for it, and where the sort threw, the aggregation throws the
    /// same exception at the same offset. Each filter script's `boolean` is
    /// 1 on the documents the `script` query matched and 0 on the rest, as
    /// `ScriptDoubleValues` reads it.
    #[test]
    fn aggregation_scripts_read_what_opensearch_reads() {
        use lucene_aggs::bucket_aggs::ScriptValuesSource;
        use lucene_util::numeric_utils::double_to_sortable_long;
        let tmp = TempDir::new("painless-score-agg");
        for e in std::fs::read_dir(fixture("index")).unwrap() {
            let e = e.unwrap();
            std::fs::copy(e.path(), tmp.path().join(e.file_name())).unwrap();
        }
        let h = open(&tmp);
        let handle = crate::jvm_reader::lookup(h, "reader").unwrap();
        let mut wrong = Vec::new();
        let mut ran = 0;
        for case in cases() {
            if !case.filter && case.sort.is_none() {
                continue;
            }
            let mut spec = case.spec.clone();
            spec[0] = 5;
            let Ok(values) = PainlessAggValues::decode(&mut Cursor::new(&spec)) else {
                let e = lucene_engine::status::last_error();
                // A filter script's `params` has no `doc` or `_source`; an
                // aggregation script's has them, as Java serves them.
                let java = [
                    "params.doc == null && doc['likes'].value < 100",
                    "params._source.size() > 0",
                ];
                if !IN_JAVA.contains(&case.source.as_str()) && !java.contains(&case.source.as_str())
                {
                    wrong.push(format!("{}: refused: {e}", case.source));
                }
                continue;
            };
            let mut got: Vec<(i32, Vec<i64>)> = Vec::new();
            let mut threw = None;
            'segments: for seg in handle.reader.segment_readers() {
                let mut leaf = values.open(seg).unwrap();
                for doc in 0..seg.max_doc {
                    let mut out = Vec::new();
                    if let Err(e) = leaf.values(doc, &mut out) {
                        threw = Some(e.to_string());
                        break 'segments;
                    }
                    got.push((seg.doc_base + doc, out));
                }
            }
            got.sort_by_key(|&(d, _)| d);
            let bad = match (&case.outcome, threw) {
                (Outcome::Sorted(want), None) => {
                    let mut want: Vec<(i32, Vec<i64>)> =
                        want.iter().map(|&(d, v)| (d, vec![v])).collect();
                    want.sort_by_key(|&(d, _)| d);
                    (got != want).then(|| format!("values {got:?}, want {want:?}"))
                }
                (Outcome::Hits(hits), None) => {
                    let matched: Vec<i32> = hits.iter().map(|&(d, _)| d).collect();
                    let want: Vec<(i32, Vec<i64>)> = got
                        .iter()
                        .map(|&(d, _)| {
                            let v = if matched.contains(&d) { 1.0 } else { 0.0 };
                            (d, vec![double_to_sortable_long(v)])
                        })
                        .collect();
                    (got != want).then(|| format!("values {got:?}, want {want:?}"))
                }
                (Outcome::Threw(class, _, offset), Some(e)) => {
                    let want = format!("\u{1}{offset}\u{1}{class}\u{1}");
                    (!e.contains(SCRIPT_ERROR) || !e.contains(&want) || !e.contains("\u{1}b"))
                        .then(|| format!("{e:?}, want ..{want:?}"))
                }
                (o, r) => Some(format!("{o:?} vs {r:?}")),
            };
            if let Some(bad) = bad {
                wrong.push(format!("{}: {bad}", case.source));
            }
            ran += 1;
        }
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
        assert!(ran >= 20, "{ran} scripts run");
        drop(handle);
        assert_eq!(ffi_close_jvm_reader(h), 0);
    }

    #[test]
    fn terms_set_minimums_of_other_scripts_differ() {
        // `CoveringQuery` equality compares `describe()`: two scripts there
        // must not merge in the boolean rewrite.
        let specs: Vec<Vec<u8>> = cases()
            .into_iter()
            .filter(|c| c.sort.is_some())
            .map(|c| {
                let mut spec = c.spec.clone();
                spec[0] = 6;
                spec
            })
            .take(2)
            .collect();
        let describe = |spec: &[u8]| {
            PainlessMinimum::decode(&mut Cursor::new(spec))
                .unwrap()
                .describe()
        };
        assert_ne!(describe(&specs[0]), describe(&specs[1]));
        assert_eq!(describe(&specs[0]), describe(&specs[0]));
    }

    /// **`terms_set` minimums.** Each fixture sort script, run as a
    /// `terms_set` query's `minimum_should_match_script`: each document's
    /// minimum is the sort's double as `Number.longValue()` reads it, and
    /// where the sort threw, the minimum throws the same exception at the
    /// same offset.
    #[test]
    fn terms_set_minimums_read_what_opensearch_reads() {
        let tmp = TempDir::new("painless-score-terms-set");
        for e in std::fs::read_dir(fixture("index")).unwrap() {
            let e = e.unwrap();
            std::fs::copy(e.path(), tmp.path().join(e.file_name())).unwrap();
        }
        let h = open(&tmp);
        let handle = crate::jvm_reader::lookup(h, "reader").unwrap();
        let mut wrong = Vec::new();
        let mut ran = 0;
        for case in cases() {
            if case.sort.is_none() {
                continue;
            }
            let mut spec = case.spec.clone();
            spec[0] = 6;
            let minimum = PainlessMinimum::decode(&mut Cursor::new(&spec)).unwrap_or_else(|_| {
                panic!("{}: {}", case.source, lucene_engine::status::last_error())
            });
            let mut got: Vec<(i32, i64)> = Vec::new();
            let mut threw = None;
            'segments: for seg in handle.reader.segment_readers() {
                let ctx = ValuesContext::for_reader(seg);
                let mut values = minimum.get_values(&ctx, 0, None).unwrap();
                for doc in 0..seg.max_doc {
                    match values.advance_exact(doc) {
                        Ok(true) => got.push((seg.doc_base + doc, values.long_value().unwrap())),
                        Ok(false) => {}
                        Err(e) => {
                            threw = Some(e.to_string());
                            break 'segments;
                        }
                    }
                }
            }
            got.sort_by_key(|&(d, _)| d);
            let bad = match (&case.outcome, threw) {
                (Outcome::Sorted(want), None) => {
                    let mut want: Vec<(i32, i64)> = want
                        .iter()
                        .map(|&(d, v)| {
                            let d_value = lucene_util::numeric_utils::sortable_long_to_double(v);
                            (d, d_value as i64)
                        })
                        .collect();
                    want.sort_by_key(|&(d, _)| d);
                    (got != want).then(|| format!("minimums {got:?}, want {want:?}"))
                }
                (Outcome::Threw(class, _, offset), Some(e)) => {
                    let want = format!("\u{1}{offset}\u{1}{class}\u{1}");
                    (!e.contains(SCRIPT_ERROR) || !e.contains(&want))
                        .then(|| format!("{e:?}, want ..{want:?}"))
                }
                (o, r) => Some(format!("{o:?} vs {r:?}")),
            };
            if let Some(bad) = bad {
                wrong.push(format!("{}: {bad}", case.source));
            }
            ran += 1;
        }
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
        assert!(ran >= 10, "{ran} scripts run");
        drop(handle);
        assert_eq!(ffi_close_jvm_reader(h), 0);
    }

    /// `Number.longValue()`: Java's narrowing, saturating and `NaN` to 0.
    #[test]
    fn terms_set_minimums_narrow_as_java_does() {
        assert_eq!(long_value(&Value::Int(-3)).unwrap(), -3);
        assert_eq!(long_value(&Value::Double(2.9)).unwrap(), 2);
        assert_eq!(long_value(&Value::Double(-2.9)).unwrap(), -2);
        assert_eq!(long_value(&Value::Double(f64::NAN)).unwrap(), 0);
        assert_eq!(long_value(&Value::Double(1e300)).unwrap(), i64::MAX);
        assert_eq!(long_value(&Value::Float(-1e30)).unwrap(), i64::MIN);
        assert!(long_value(&Value::Bool(true)).is_err());
    }

    /// `ScriptDoubleValues.toDoubleValue`: numbers and booleans, and the
    /// error OpenSearch raises for anything else.
    #[test]
    fn aggregation_values_are_numbers_or_booleans() {
        assert_eq!(agg_double(&Value::Bool(true)).unwrap(), 1.0);
        assert_eq!(agg_double(&Value::Bool(false)).unwrap(), 0.0);
        assert_eq!(agg_double(&Value::Int(-3)).unwrap(), -3.0);
        assert_eq!(
            agg_double(&Value::Long(1 << 40)).unwrap(),
            (1u64 << 40) as f64
        );
        assert_eq!(agg_double(&Value::Float(0.5)).unwrap(), 0.5);
        let e = agg_double(&Value::Null).unwrap_err().to_string();
        assert!(e.contains(AGGREGATION_ERROR), "{e}");
        assert!(
            e.ends_with("Unsupported script value [null], expected a number, date, or boolean"),
            "{e}"
        );
    }

    /// Two `function_score` nodes that differ only in their script's run
    /// number are equal, as Java's `ScriptScoreFunction`s comparing the
    /// script are (so a boolean query merges them as duplicates).
    #[test]
    fn script_function_equality_ignores_the_run_number() {
        use lucene_engine::nodes::NODE_FUNCTION_SCORE;
        let case = cases()
            .into_iter()
            .find(|c| check(&c.spec).unwrap().is_none())
            .unwrap();
        let decoded = |run: i32| {
            let mut spec = case.spec.clone();
            let at = Spec::at_script_id(&spec);
            spec[at..at + 4].copy_from_slice(&run.to_le_bytes());
            let node = Blob::default()
                .u8(NODE_FUNCTION_SCORE)
                .u8(1)
                .u8(0)
                .f32(f32::MAX)
                .raw(&case.sub)
                .i32(1)
                .u8(0)
                .u8(5)
                .raw(&Blob(spec));
            let mut nodes = 0;
            lucene_engine::query_blob::decode_node(&mut Cursor::new(&node.0), 0, &mut nodes)
                .unwrap()
        };
        assert_eq!(decoded(0), decoded(7));
    }

    /// `cargo test --release -p lucene-ffi --lib painless_check_cost --
    /// --ignored --nocapture`: the static check's cost per fixture script
    /// (decode, compile, specialise, check), once per distinct script and
    /// params, as the plugin caches the answer.
    #[test]
    #[ignore]
    fn painless_check_cost() {
        let mut all = Vec::new();
        for case in &cases() {
            let mut t: Vec<u128> = (0..200)
                .map(|_| {
                    let start = std::time::Instant::now();
                    std::hint::black_box(check(&case.spec).unwrap());
                    start.elapsed().as_nanos()
                })
                .collect();
            t.sort_unstable();
            all.push(t[t.len() / 2]);
            println!("{:>8} ns  {}", t[t.len() / 2], case.source);
        }
        all.sort_unstable();
        println!(
            "median {} ns, max {} ns",
            all[all.len() / 2],
            all[all.len() - 1]
        );
    }
}

#[cfg(test)]
pub(crate) mod bench {
    use crate::engine_tests::nodes::tests::{open, Blob};
    use crate::jvm_reader::tests::run_limit;

    use lucene_engine::painless_score::*;

    use lucene_painless::ir;

    use lucene_search::values_source::{DoubleValues, DoubleValuesSource, ValuesContext};
    use lucene_search::Result;

    use lucene_util::test_support::TempDir;

    use std::sync::Arc;

    fn unhex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    /// `PAINLESS_BENCH_INDEX=<dir> PAINLESS_BENCH_BLOBS=<tsv> cargo test
    /// --release -p lucene-ffi --lib painless_score_bench -- --ignored
    /// --nocapture`: each script's ns per document over a match-all query.
    #[test]
    #[ignore]
    fn painless_score_bench() {
        let from = std::env::var("PAINLESS_BENCH_INDEX").unwrap();
        let tmp = TempDir::new("painless-bench");
        for e in std::fs::read_dir(from).unwrap() {
            let e = e.unwrap();
            std::fs::copy(e.path(), tmp.path().join(e.file_name())).unwrap();
        }
        let h = open(&tmp);
        let time = |blob: &Blob| {
            let mut best = f64::MAX;
            let mut total = 0;
            for _ in 0..5 {
                let t = std::time::Instant::now();
                total = run_limit(h, &blob.0, 10, i64::MAX)
                    .unwrap_or_else(|_| panic!("{}", lucene_engine::status::last_error()))
                    .1;
                best = best.min(t.elapsed().as_nanos() as f64);
            }
            best / total as f64
        };
        let base = time(&Blob::tree().u8(5));
        println!("match_all\t{base:.2} ns/doc");
        let blobs =
            std::fs::read_to_string(std::env::var("PAINLESS_BENCH_BLOBS").unwrap()).unwrap();
        for line in blobs.lines() {
            let cols: Vec<&str> = line.split('\t').collect();
            let mut b = Blob::tree()
                .u8(NODE_PAINLESS_SCORE)
                .u8(5)
                .u8(1)
                .bytes(&unhex(cols[1]))
                .bytes(&unhex(cols[2]))
                .str("idx")
                .i32(0)
                .i32(0)
                .i32(cols.len() as i32 - 3);
            for f in &cols[3..] {
                let (name, kind) = f.split_once(':').unwrap();
                b = b.str(name).u8(kind.parse().unwrap());
            }
            let t = time(&b);
            println!("{}\t{:.2} ns/doc (script {:.2})", cols[0], t, t - base);
        }
    }

    struct OneAndHalf;

    impl DoubleValues for OneAndHalf {
        fn advance_exact(&mut self, _doc: i32) -> Result<bool> {
            Ok(true)
        }
        fn double_value(&mut self) -> Result<f64> {
            Ok(1.5)
        }
    }

    /// The script alone per document, as the Java harness times
    /// `setDocument` + `execute`: `PAINLESS_BENCH_INDEX` (one segment) and
    /// `PAINLESS_BENCH_BLOBS` as for [`painless_score_bench`].
    #[test]
    #[ignore]
    fn painless_script_loop_bench() {
        let dir = lucene_store::directory::FsDirectory::open(
            std::env::var("PAINLESS_BENCH_INDEX").unwrap(),
        );
        let reader = lucene_search::directory_reader::DirectoryReader::open(&dir).unwrap();
        let seg = &reader.segment_readers()[0];
        let ctx = ValuesContext::for_reader(seg);
        let blobs =
            std::fs::read_to_string(std::env::var("PAINLESS_BENCH_BLOBS").unwrap()).unwrap();
        for line in blobs.lines() {
            let cols: Vec<&str> = line.split('\t').collect();
            let ir_blob = unhex(cols[1]);
            let params_blob = unhex(cols[2]);
            let fields = cols[3..]
                .iter()
                .map(|f| {
                    let (name, kind) = f.split_once(':').unwrap();
                    let kind = match kind {
                        "0" => FieldKind::Long,
                        "1" => FieldKind::Double,
                        "5" => FieldKind::Keyword,
                        other => panic!("kind {other}"),
                    };
                    FieldSpec {
                        name: name.to_string(),
                        read: name.to_string(),
                        kind,
                    }
                })
                .collect();
            let source = PainlessScoreSource(Arc::new(Spec {
                script: ir::decode(&ir_blob).unwrap(),
                params: ir::decode_constant(&params_blob).unwrap(),
                ir_blob,
                params_blob,
                needs_score: true,
                filter: false,
                sort: false,
                agg: false,
                terms_set: false,
                interval: false,
                similarity: false,
                similarity_weight: false,
                index: "idx".into(),
                shard: 0,
                script_id: 0,
                script_id_at: 0,
                fields,
            }));
            let n = seg.max_doc;
            let mut best = f64::MAX;
            let mut sum = 0.0;
            for _ in 0..8 {
                let mut v = source
                    .get_values(&ctx, 0, Some(Box::new(OneAndHalf)))
                    .unwrap();
                let t = std::time::Instant::now();
                sum = 0.0;
                if std::env::var("PAINLESS_BENCH_PER_DOC").is_ok() {
                    for d in 0..n {
                        v.advance_exact(d).unwrap();
                        sum += v.double_value().unwrap();
                    }
                } else {
                    // The function score scorer's batches.
                    let (mut docs, scores) = (Vec::new(), vec![1.5f32; 64]);
                    let (mut out, mut has) = (vec![0.0; 64], vec![false; 64]);
                    for start in (0..n).step_by(64) {
                        docs.clear();
                        docs.extend(start..(start + 64).min(n));
                        v.fill_batch(&docs, &scores, &mut out, &mut has).unwrap();
                        sum += out[..docs.len()].iter().sum::<f64>();
                    }
                }
                best = best.min(t.elapsed().as_nanos() as f64 / f64::from(n));
            }
            println!("{}\t{best:.2} ns/doc\tsum={sum}", cols[0]);
        }
    }
}
