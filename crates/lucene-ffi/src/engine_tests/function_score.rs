//! The `function_score` tests that run through the exported entry points.

#[cfg(test)]
pub(crate) mod tests {
    use crate::engine_tests::nodes::tests::{fails, index, open, search, Blob, BODIES};
    use crate::jvm_reader::ffi_close_jvm_reader;

    use lucene_engine::function_score::*;

    use lucene_engine::nodes::NODE_FUNCTION_SCORE;

    use lucene_engine::query_blob::{decode_node, Cursor};
    use lucene_engine::status::Status;

    use lucene_search::extended_query::ExtendedQuery;

    use lucene_search::query::Clause;

    use lucene_search::values_source::ValuesContext;

    use std::collections::HashMap;

    /// Block `i`'s root document: after every earlier block and its own
    /// `i % 3` nested documents.
    fn root(i: usize) -> i32 {
        ((0..i).map(|k| k % 3 + 1).sum::<usize>() + i % 3) as i32
    }

    fn block_of(doc: i32) -> usize {
        (0..12).find(|&i| root(i) == doc).expect("a root")
    }

    /// The function score of `functions` (each an optional filter node and a
    /// function) over `body:alpha`.
    fn fs(combine: u8, mode: u8, max_boost: f32, functions: &[(Option<Blob>, Blob)]) -> Blob {
        let mut b = Blob::tree()
            .u8(NODE_FUNCTION_SCORE)
            .u8(combine)
            .u8(mode)
            .f32(max_boost)
            .term("body", "alpha")
            .i32(functions.len() as i32);
        for (filter, f) in functions {
            b = match filter {
                Some(node) => b.u8(1).raw(node),
                None => b.u8(0),
            }
            .raw(f);
        }
        b
    }

    fn weight(w: f32) -> Blob {
        Blob::default().u8(FUNCTION_WEIGHT).f32(w).u8(FUNCTION_ONE)
    }

    fn fvf(field: &str, numeric: u8, factor: f32, modifier: u8, missing: Option<f64>) -> Blob {
        Blob::default()
            .u8(FUNCTION_FIELD_VALUE)
            .str(field)
            .str(field)
            .u8(numeric)
            .f32(factor)
            .u8(modifier)
            .u8(u8::from(missing.is_some()))
            .f64(missing.unwrap_or(0.0))
    }

    fn decay(
        kind: u8,
        field: &str,
        numeric: u8,
        origin: f64,
        scale: f64,
        offset: f64,
        mode: u8,
    ) -> Blob {
        Blob::default()
            .u8(FUNCTION_DECAY)
            .u8(kind)
            .str(field)
            .u8(numeric)
            .f64(origin)
            .f64(scale)
            .f64(offset)
            .u8(mode)
    }

    /// `body:alpha`'s scores by document.
    fn alpha(h: u64) -> HashMap<i32, f32> {
        search(h, &Blob::tree().term("body", "alpha"))
            .0
            .into_iter()
            .collect()
    }

    /// Every hit of `blob` scores `want(doc, sub_score)`, and they are
    /// `body:alpha`'s documents.
    fn scores_as(h: u64, blob: &Blob, want: impl Fn(i32, f32) -> f32) {
        let subs = alpha(h);
        let (hits, total) = search(h, blob);
        assert_eq!(total, subs.len() as i64);
        for (doc, score) in hits {
            let w = want(doc, subs[&doc]);
            assert_eq!(score.to_bits(), w.to_bits(), "doc {doc}: {score} vs {w}");
        }
    }

    #[test]
    fn every_boost_mode_combines_as_opensearch_does() {
        let tmp = index("jvm-fs-combine");
        let h = open(&tmp);
        for (combine, f) in [
            (0u8, (|q: f64, f: f64| q * f) as fn(f64, f64) -> f64),
            (1, |_, f| f),
            (2, |q, f| q + f),
            (3, |q, f| (f + q) / 2.0),
            (4, |q: f64, f: f64| q.min(f)),
            (5, |q: f64, f: f64| q.max(f)),
        ] {
            // weight 3 capped by max_boost 2.
            scores_as(h, &fs(combine, 0, 2.0, &[(None, weight(3.0))]), |_, s| {
                f(f64::from(s), 2.0) as f32
            });
        }
        ffi_close_jvm_reader(h);
    }

    #[test]
    fn every_score_mode_folds_the_matching_functions() {
        let tmp = index("jvm-fs-modes");
        let h = open(&tmp);
        // Weight 2 on documents with "beta", weight 5 everywhere.
        let beta = Some(Blob::default().term("body", "beta"));
        let funcs = [(beta.clone(), weight(2.0)), (None, weight(5.0))];
        let has_beta = |doc: i32| BODIES[block_of(doc) % 6].contains("beta");
        for (mode, f) in [
            (
                0u8,
                (|b: bool| if b { 2.0 } else { 5.0 }) as fn(bool) -> f64,
            ),
            // (2 + 5) / (2 + 5), or 5 / 5.
            (1, |_| 1.0),
            (2, |_| 5.0),
            (3, |b| if b { 7.0 } else { 5.0 }),
            (4, |b| if b { 2.0 } else { 5.0 }),
            (5, |b| if b { 10.0 } else { 5.0 }),
        ] {
            scores_as(h, &fs(0, mode, f32::MAX, &funcs), |doc, s| {
                (f64::from(s) * f(has_beta(doc))) as f32
            });
        }
        // No function matches: the factor is 1 in every mode.
        let none = [(Some(Blob::default().term("body", "zzz")), weight(9.0))];
        for mode in 0..=5u8 {
            scores_as(h, &fs(0, mode, f32::MAX, &none), |_, s| s);
        }
        ffi_close_jvm_reader(h);
    }

    #[test]
    fn field_value_factor_reads_each_numeric_type_through_its_modifier() {
        let tmp = index("jvm-fs-fvf");
        let h = open(&tmp);
        let pos = |doc: i32| block_of(doc) as f64 + 1.0;
        let modifiers: [fn(f64) -> f64; 10] = [
            |n| n,
            f64::log10,
            |n| (n + 1.0).log10(),
            |n| (n + 2.0).log10(),
            f64::ln,
            f64::ln_1p,
            |n| (n + 1.0).ln_1p(),
            |n| n * n,
            f64::sqrt,
            |n| 1.0 / n,
        ];
        for (m, f) in modifiers.iter().enumerate() {
            scores_as(
                h,
                &fs(1, 0, f32::MAX, &[(None, fvf("pos", 0, 1.5, m as u8, None))]),
                |doc, _| f(pos(doc) * 1.5) as f32,
            );
        }
        // A double and a float, squared (`sd` is negative on early blocks).
        let sd = |doc: i32| (block_of(doc) as f64 - 5.5) * 3.25;
        scores_as(
            h,
            &fs(1, 0, f32::MAX, &[(None, fvf("sd", 1, 1.0, 7, None))]),
            |doc, _| (sd(doc) * sd(doc)) as f32,
        );
        let sf = |doc: i32| f64::from(block_of(doc) as f32 * 0.75 - 2.0);
        scores_as(
            h,
            &fs(1, 0, f32::MAX, &[(None, fvf("sf", 2, 1.0, 7, None))]),
            |doc, _| (sf(doc) * sf(doc)) as f32,
        );
        // `si` is on even blocks only: its first value, or the missing value.
        scores_as(
            h,
            &fs(1, 0, f32::MAX, &[(None, fvf("si", 0, 1.0, 0, Some(0.25)))]),
            |doc, _| {
                let i = block_of(doc);
                if i.is_multiple_of(2) {
                    i as f32
                } else {
                    0.25
                }
            },
        );
        // Without one, a missing value is OpenSearch's error; so is a negative score.
        let (rc, msg) = fails(
            h,
            &fs(1, 0, f32::MAX, &[(None, fvf("si", 0, 1.0, 0, None))]),
        );
        assert_eq!(rc, Status::Search.code());
        assert!(msg.contains("Missing value for field [si]"), "{msg}");
        let (_, msg) = fails(
            h,
            &fs(1, 0, f32::MAX, &[(None, fvf("sd", 1, 1.0, 0, None))]),
        );
        assert!(msg.contains("must not produce negative scores"), "{msg}");
        let (_, msg) = fails(h, &fs(0, 0, f32::MAX, &[(None, weight(-1.0))]));
        assert!(msg.contains("invalid score"), "{msg}");
        ffi_close_jvm_reader(h);
    }

    #[test]
    fn random_scores_hash_the_document_or_its_field() {
        let tmp = index("jvm-fs-random");
        let h = open(&tmp);
        let random_fn = |kind: u8, field: Option<&str>| {
            let b = Blob::default().u8(FUNCTION_RANDOM).i32(12345).u8(kind);
            match field {
                Some(f) => b.str(f),
                None => b,
            }
        };
        scores_as(
            h,
            &fs(1, 0, f32::MAX, &[(None, random_fn(0, None))]),
            |doc, _| random(mix(doc, 12345)) as f32,
        );
        scores_as(
            h,
            &fs(1, 0, f32::MAX, &[(None, random_fn(1, Some("kw")))]),
            |doc, _| {
                let term = format!("k{}", block_of(doc) % 4);
                random(lucene_util::string_helper::murmurhash3_x86_32(
                    term.as_bytes(),
                    12345,
                )) as f32
            },
        );
        scores_as(
            h,
            &fs(1, 0, f32::MAX, &[(None, random_fn(2, Some("si")))]),
            |doc, _| {
                let i = block_of(doc);
                // `si` holds `i` and `3i + 1`; OpenSearch hashes the bytewise
                // smaller decimal string ("13" before "4" on block 4).
                let (a, b) = (i.to_string(), (3 * i + 1).to_string());
                let first = if a.as_bytes() <= b.as_bytes() { a } else { b };
                random(if i.is_multiple_of(2) {
                    lucene_util::string_helper::murmurhash3_x86_32(first.as_bytes(), 12345)
                } else {
                    12345
                }) as f32
            },
        );
        // A keyword with no value hashes the seed.
        scores_as(
            h,
            &fs(1, 0, f32::MAX, &[(None, random_fn(1, Some("nokw")))]),
            |_, _| random(12345) as f32,
        );
        // `BitMixer.mix` (HPPC 0.10), for a few keys and seeds.
        assert_eq!(mix(0, 0), 0);
        assert_eq!(mix(1, 0), 1_364_076_727);
        assert_eq!(mix(5, 42), mix(5 ^ 42, 0));
        ffi_close_jvm_reader(h);
    }

    #[test]
    fn decay_functions_score_the_selected_distance() {
        let tmp = index("jvm-fs-decay");
        let h = open(&tmp);
        let evals: [fn(f64, f64) -> f64; 3] = [
            |v, s| (0.5 * (v * v) / s).exp(),
            |v, s| (s * v).exp(),
            |v, s| 0.0f64.max((s - v) / s),
        ];
        let (origin, offset) = (4.0, 0.5);
        for (kind, eval) in evals.iter().enumerate() {
            let scale = [-12.5, -0.1, 9.0][kind];
            for mode in 0..=4u8 {
                let blob = fs(
                    1,
                    0,
                    f32::MAX,
                    &[(
                        None,
                        decay(kind as u8, "si", 0, origin, scale, offset, mode),
                    )],
                );
                scores_as(h, &blob, |doc, _| {
                    let i = block_of(doc) as f64;
                    let d = if block_of(doc).is_multiple_of(2) {
                        let mut v = [
                            0.0f64.max((i - origin).abs() - offset),
                            0.0f64.max((3.0 * i + 1.0 - origin).abs() - offset),
                        ];
                        v.sort_by(f64::total_cmp);
                        match mode {
                            0 => v[0] + v[1],
                            1 => (v[0] + v[1]) / 2.0,
                            2 => (v[0] + v[1]) / 2.0,
                            3 => v[0],
                            _ => v[1],
                        }
                    } else {
                        0.0
                    };
                    eval(d, scale) as f32
                });
            }
        }
        // One value: the median is it.
        assert_eq!(MultiValue::Median.pick(&[1.0, 2.0, 7.0]), 2.0);
        assert_eq!(MultiValue::Avg.pick(&[1.0, 2.0]), 1.5);
        ffi_close_jvm_reader(h);
    }

    #[test]
    fn java_min_and_max_keep_nan_and_signed_zeros() {
        assert!(java_min(f64::NAN, 1.0).is_nan());
        assert!(java_max(1.0, f64::NAN).is_nan());
        assert!(java_min(0.0, -0.0).is_sign_negative());
        assert!(java_min(-0.0, 0.0).is_sign_negative());
        assert!(java_max(-0.0, 0.0).is_sign_positive());
        assert!(java_max(0.0, -0.0).is_sign_positive());
        assert_eq!(java_min(1.0, 2.0), 1.0);
        assert_eq!(java_max(1.0, 2.0), 2.0);
        assert_eq!(Numeric::Long.decode(-3), -3.0);
    }

    #[test]
    fn the_source_describes_itself_and_its_filters() {
        let tmp = index("jvm-fs-source");
        let h = open(&tmp);
        let blob = fs(
            0,
            0,
            f32::MAX,
            &[(
                Some(Blob::default().term("body", "beta")),
                fvf("pos", 0, 1.0, 0, None),
            )],
        );
        let mut c = Cursor::new(&blob.0[1..]);
        let mut nodes = 0;
        let clause = decode_node(&mut c, 0, &mut nodes).unwrap();
        let Clause::Extended(e) = &clause else {
            panic!("{clause:?}")
        };
        assert_eq!(queries(e).len(), 1);
        assert!(
            queries(&ExtendedQuery::Span(lucene_search::spans::SpanNode::term(
                "f", "t"
            )))
            .is_empty()
        );
        let ExtendedQuery::FunctionScore(q) = e.as_ref() else {
            panic!()
        };
        assert!(q
            .source
            .describe()
            .starts_with("opensearch_function_score("));
        assert!(q.source.needs_scores());
        let dir = lucene_store::FsDirectory::open(tmp.path());
        let reader = lucene_search::directory_reader::DirectoryReader::open(&dir).unwrap();
        let ctx = ValuesContext::for_reader(&reader.segment_readers()[0]);
        assert!(q.source.is_cacheable(&ctx, 0));
        // Also as a filter, where the source is never asked.
        let filtered = Blob::tree()
            .u8(1)
            .i32(0)
            .i32(2)
            .u8(0)
            .term("body", "gamma")
            .u8(1)
            .raw(&Blob(blob.0[1..].to_vec()));
        assert!(search(h, &filtered).1 > 0);
        ffi_close_jvm_reader(h);
    }

    #[test]
    fn malformed_function_scores_are_refused() {
        let tmp = index("jvm-fs-bad");
        let h = open(&tmp);
        let bad = |b: Blob, want: &str| {
            let (rc, msg) = fails(h, &b);
            assert_eq!(rc, Status::InvalidArgument.code(), "{msg}");
            assert!(msg.contains(want), "{msg} lacks {want}");
        };
        let head = |combine: u8, mode: u8| {
            Blob::tree()
                .u8(NODE_FUNCTION_SCORE)
                .u8(combine)
                .u8(mode)
                .f32(1.0)
                .term("body", "alpha")
        };
        bad(head(9, 0).i32(1).u8(0).u8(0), "boost mode 9");
        bad(head(0, 9).i32(1).u8(0).u8(0), "function score mode 9");
        bad(head(0, 0).i32(0), "without functions");
        bad(
            fs(0, 0, 1.0, &[(None, Blob::default().u8(77))]),
            "unknown score function 77",
        );
        bad(
            fs(
                0,
                0,
                1.0,
                &[(
                    None,
                    Blob::default()
                        .u8(FUNCTION_WEIGHT)
                        .f32(1.0)
                        .raw(&weight(2.0)),
                )],
            ),
            "unknown score function 1 at depth 1",
        );
        bad(
            fs(0, 0, 1.0, &[(None, fvf("pos", 7, 1.0, 0, None))]),
            "numeric field type 7",
        );
        bad(
            fs(0, 0, 1.0, &[(None, fvf("pos", 0, 1.0, 42, None))]),
            "modifier 42",
        );
        bad(
            fs(
                0,
                0,
                1.0,
                &[(None, Blob::default().u8(FUNCTION_RANDOM).i32(1).u8(9))],
            ),
            "random score field kind 9",
        );
        bad(
            fs(0, 0, 1.0, &[(None, decay(5, "pos", 0, 0.0, 1.0, 0.0, 0))]),
            "decay function 5",
        );
        bad(
            fs(0, 0, 1.0, &[(None, decay(0, "pos", 0, 0.0, 1.0, 0.0, 8))]),
            "multi-value mode 8",
        );
        let utf = Blob::default().u8(FUNCTION_FIELD_VALUE).bytes(&[0xff]);
        assert_eq!(
            fails(h, &fs(0, 0, 1.0, &[(None, utf)])).0,
            Status::InvalidUtf8.code()
        );
        ffi_close_jvm_reader(h);
    }
}
