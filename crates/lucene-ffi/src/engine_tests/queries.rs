//! The `queries` tests that run through the exported entry points.

#[cfg(test)]
pub(crate) mod tests {
    use crate::engine_tests::nodes::tests::{fails, open, search, Blob, BODIES};
    use crate::jvm_reader::ffi_close_jvm_reader;

    use lucene_engine::queries::*;

    use lucene_index::document::{
        Document, FeatureField, LatLonDocValuesField, LatLonPoint, LongField,
        SortedNumericDocValuesField, SortedSetDocValuesField, Store, StringField, TextField,
    };
    use lucene_index::index_writer::IndexWriter;
    use lucene_index::segment_info::LuceneVersion;

    use lucene_store::FsDirectory;
    use lucene_util::test_support::TempDir;

    /// Twelve documents in two segments (global doc `i` is document `i`):
    /// `body` (`BODIES[i % 6]`), and doc values only -- `kw` (`k{i % 4}`),
    /// `n` (`i`), `msm` (`i % 3`) -- plus `date` (points and doc values,
    /// `i * 1000`) and `loc` (latitude `i`, longitude 0).
    fn index(tag: &str) -> TempDir {
        let tmp = TempDir::new(tag);
        let dir = FsDirectory::open(tmp.path());
        let version = LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        };
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", version).unwrap();
        for i in 0..12i64 {
            let mut d = Document::new();
            d.add(StringField::new("id", i.to_string(), Store::Yes));
            d.add(TextField::new("body", BODIES[(i % 6) as usize], Store::No));
            d.add(SortedSetDocValuesField::new(
                "kw",
                format!("k{}", i % 4).into_bytes(),
            ));
            d.add(SortedNumericDocValuesField::new("n", i));
            d.add(SortedNumericDocValuesField::new("msm", i % 3));
            d.add(LongField::new("date", i * 1000, Store::No));
            d.add(LatLonPoint::new("loc", i as f64, 0.0).unwrap());
            d.add(LatLonDocValuesField::new("loc", i as f64, 0.0).unwrap());
            w.add_fields_document(&d).unwrap();
            if i == 5 {
                w.commit().unwrap();
            }
        }
        w.commit().unwrap();
        tmp
    }

    fn docs(h: u64, blob: &Blob) -> Vec<i32> {
        let mut d: Vec<i32> = search(h, blob).0.into_iter().map(|(d, _)| d).collect();
        d.sort_unstable();
        d
    }

    fn matching(f: impl Fn(i32) -> bool) -> Vec<i32> {
        (0..12).filter(|&i| f(i)).collect()
    }

    /// `k0` or `k2`: `k`, then `0` or `2`, as a normal byte automaton.
    fn k0_or_k2() -> Blob {
        Blob::default()
            .u8(3)
            .i32(3)
            // State 0: not accepting, `k` to 1.
            .u8(0)
            .i32(1)
            .i32(1)
            .i32(i32::from(b'k'))
            .i32(i32::from(b'k'))
            // State 1: `0` or `2` to 2.
            .u8(0)
            .i32(2)
            .i32(2)
            .i32(i32::from(b'0'))
            .i32(i32::from(b'0'))
            .i32(2)
            .i32(i32::from(b'2'))
            .i32(i32::from(b'2'))
            // State 2: accepting.
            .u8(1)
            .i32(0)
    }

    fn multi_term(field: &str, rewrite: u8, automaton: &Blob) -> Blob {
        Blob::tree()
            .u8(NODE_MULTI_TERM)
            .str(field)
            .u8(rewrite)
            .raw(automaton)
    }

    #[test]
    fn a_multi_term_node_matches_its_automaton_under_every_rewrite() {
        let tmp = index("jvm-queries-multi-term");
        let h = open(&tmp);
        let alpha = matching(|i| BODIES[(i % 6) as usize].contains("alpha"));
        for rewrite in [REWRITE_CONSTANT_SCORE_BLENDED, REWRITE_CONSTANT_SCORE] {
            let single = multi_term("body", rewrite, &Blob::default().u8(2).str("alpha"));
            let (hits, total) = search(h, &single);
            assert_eq!(total, alpha.len() as i64, "rewrite {rewrite}");
            assert!(hits.iter().all(|&(_, s)| s == 1.0), "constant scores");
            assert_eq!(docs(h, &single), alpha);
            // Every term, and none.
            assert_eq!(
                docs(h, &multi_term("body", rewrite, &Blob::default().u8(1))).len(),
                12
            );
            assert!(docs(h, &multi_term("body", rewrite, &Blob::default().u8(0))).is_empty());
        }
        // A doc-values-only keyword, through its ordinals.
        let even = matching(|i| i % 2 == 0);
        assert_eq!(
            docs(h, &multi_term("kw", REWRITE_DOC_VALUES, &k0_or_k2())),
            even
        );
        // Under a boost, scored as it.
        let boosted = Blob::tree()
            .u8(3)
            .f32(2.5)
            .u8(NODE_MULTI_TERM)
            .str("kw")
            .u8(REWRITE_DOC_VALUES)
            .raw(&k0_or_k2());
        assert!(search(h, &boosted).0.iter().all(|&(_, s)| s == 2.5));
        let (rc, e) = fails(h, &multi_term("kw", 9, &k0_or_k2()));
        assert_ne!(rc, 0);
        assert!(e.contains("multi-term rewrite 9"), "{e}");
        assert_eq!(ffi_close_jvm_reader(h), 0);
    }

    fn synonym(terms: &[(&str, f32)]) -> Blob {
        let mut b = Blob::tree()
            .u8(NODE_SYNONYM)
            .str("body")
            .i32(terms.len() as i32);
        for (t, boost) in terms {
            b = b.str(t).f32(*boost);
        }
        b
    }

    #[test]
    fn a_synonym_node_scores_its_terms_as_one() {
        let tmp = index("jvm-queries-synonym");
        let h = open(&tmp);
        let either = matching(|i| {
            let b = BODIES[(i % 6) as usize];
            b.contains("delta") || b.contains("epsilon")
        });
        let q = synonym(&[("epsilon", 1.0), ("delta", 0.5)]);
        assert_eq!(docs(h, &q), either);
        let (hits, _) = search(h, &q);
        // One pseudo-term: a document's score depends on its summed weighted
        // frequency and its length only, so the two `delta beta` documents tie.
        let score = |d: i32| hits.iter().find(|&&(x, _)| x == d).unwrap().1;
        assert_eq!(score(3), score(9));
        assert!(hits.iter().all(|&(_, s)| s > 0.0));
        // No term matches nothing; a boost outside (0, 1] is Java's builder error.
        assert!(docs(h, &synonym(&[])).is_empty());
        let (rc, e) = fails(h, &synonym(&[("delta", 2.0)]));
        assert_ne!(rc, 0);
        assert!(e.contains("synonym query"), "{e}");
        assert_eq!(ffi_close_jvm_reader(h), 0);
    }

    #[test]
    fn a_range_with_doc_values_matches_its_points() {
        let tmp = index("jvm-queries-range-or-dv");
        let h = open(&tmp);
        let range = |kind: u8, lo: i64, hi: i64| {
            let b = Blob::tree().u8(kind).str("date");
            Blob([b.0, lo.to_le_bytes().to_vec(), hi.to_le_bytes().to_vec()].concat())
        };
        let want = matching(|i| (3..=7).contains(&i));
        assert_eq!(docs(h, &range(NODE_POINT_RANGE_OR_DV, 3000, 7000)), want);
        assert_eq!(
            docs(h, &range(NODE_POINT_RANGE_OR_DV, 3000, 7000)),
            docs(
                h,
                &range(lucene_engine::query_blob::NODE_POINT_RANGE, 3000, 7000)
            )
        );
        assert!(docs(h, &range(NODE_POINT_RANGE_OR_DV, 7000, 3000)).is_empty());
        let mut short = range(NODE_POINT_RANGE_OR_DV, 3000, 7000);
        short.0.truncate(short.0.len() - 1);
        assert_ne!(fails(h, &short).0, 0);
        assert_eq!(ffi_close_jvm_reader(h), 0);
    }

    #[test]
    fn the_doc_values_nodes_match_what_the_doc_values_hold() {
        let tmp = index("jvm-queries-doc-values");
        let h = open(&tmp);
        let set = Blob::tree()
            .u8(NODE_DV_TERM_SET)
            .str("kw")
            .i32(3)
            .str("k3")
            .str("k1")
            .str("absent");
        assert_eq!(docs(h, &set), matching(|i| i % 2 == 1));
        let range = |lo: i64, hi: i64| {
            let b = Blob::tree().u8(NODE_DV_NUMERIC_RANGE).str("n");
            Blob([b.0, lo.to_le_bytes().to_vec(), hi.to_le_bytes().to_vec()].concat())
        };
        assert_eq!(docs(h, &range(3, 7)), matching(|i| (3..=7).contains(&i)));
        assert!(docs(h, &range(7, 3)).is_empty());
        let numbers = [1i64, 5, 100, i64::MIN];
        let mut numeric_set = Blob::tree()
            .u8(NODE_DV_NUMERIC_SET)
            .str("n")
            .i32(numbers.len() as i32);
        for n in numbers {
            numeric_set.0.extend_from_slice(&n.to_le_bytes());
        }
        assert_eq!(docs(h, &numeric_set), vec![1, 5]);
        // [k1, k2): k1; (-inf, k1]: k0 and k1.
        let ords = Blob::tree()
            .u8(NODE_DV_ORD_RANGE)
            .str("kw")
            .u8(RANGE_HAS_LOWER | RANGE_HAS_UPPER | RANGE_INCLUDE_LOWER)
            .str("k1")
            .str("k2");
        assert_eq!(docs(h, &ords), matching(|i| i % 4 == 1));
        let open_lower = Blob::tree()
            .u8(NODE_DV_ORD_RANGE)
            .str("kw")
            .u8(RANGE_HAS_UPPER | RANGE_INCLUDE_UPPER)
            .str("k1");
        assert_eq!(docs(h, &open_lower), matching(|i| i % 4 <= 1));
        let (_, e) = fails(h, &Blob::tree().u8(NODE_DV_ORD_RANGE).str("kw").u8(0x10));
        assert!(e.contains("doc-values range flags"), "{e}");
        let multi = Blob::tree()
            .u8(NODE_DV_MULTI_RANGE)
            .str("kw")
            .i32(2)
            .str("k3")
            .str("k3")
            .str("k0")
            .str("k0");
        assert_eq!(docs(h, &multi), matching(|i| i % 4 == 0 || i % 4 == 3));
        let (_, e) = fails(h, &Blob::tree().u8(NODE_DV_MULTI_RANGE).str("kw").i32(0));
        assert!(e.contains("no ranges"), "{e}");
        assert_eq!(ffi_close_jvm_reader(h), 0);
    }

    #[test]
    fn a_covering_node_needs_each_documents_minimum_of_its_queries() {
        let tmp = index("jvm-queries-covering");
        let h = open(&tmp);
        let words = ["alpha", "beta", "gamma"];
        let count = |i: i32| {
            words
                .iter()
                .filter(|w| BODIES[(i % 6) as usize].split(' ').any(|t| t == **w))
                .count() as i64
        };
        let covering = |minimum: Blob| {
            let mut b = Blob::tree().u8(NODE_COVERING).i32(3);
            for w in words {
                b = b.term("body", w);
            }
            b.raw(&minimum)
        };
        // `msm` is `i % 3`; below 1 counts as 1.
        let by_field = covering(Blob::default().u8(MINIMUM_FIELD).str("msm"));
        let (hits, total) = search(h, &by_field);
        let want = matching(|i| count(i) >= i64::from(i % 3).max(1));
        assert_eq!(total, want.len() as i64);
        assert_eq!(docs(h, &by_field), want);
        assert!(
            hits.iter().all(|&(_, s)| s > 0.0),
            "scored by the sum of the matches"
        );
        // A missing field: no document has a minimum, so none matches.
        assert!(docs(
            h,
            &covering(Blob::default().u8(MINIMUM_FIELD).str("absent"))
        )
        .is_empty());
        let mut constant = Blob::default().u8(MINIMUM_CONSTANT);
        constant.0.extend_from_slice(&2i64.to_le_bytes());
        assert_eq!(docs(h, &covering(constant)), matching(|i| count(i) >= 2));
        // A covering query scores a document by the sum of its matching
        // queries, as the same SHOULD disjunction does.
        let mut boolean = Blob::tree().u8(1).i32(1).i32(3);
        for w in words {
            boolean = boolean.u8(2).term("body", w);
        }
        let (or, _) = search(h, &boolean);
        for (doc, score) in hits {
            let want = or
                .iter()
                .find(|&&(d, _)| d == doc)
                .expect("in the disjunction")
                .1;
            assert_eq!(score, want, "doc {doc}");
        }
        let (_, e) = fails(h, &covering(Blob::default().u8(7)));
        assert!(e.contains("covering minimum 7"), "{e}");
        assert_eq!(ffi_close_jvm_reader(h), 0);
    }

    #[test]
    fn the_distance_feature_nodes_score_by_closeness_to_the_origin() {
        let tmp = index("jvm-queries-distance");
        let h = open(&tmp);
        let mut long = Blob::tree().u8(NODE_LONG_DISTANCE_FEATURE).str("date");
        long.0.extend_from_slice(&5000i64.to_le_bytes());
        long.0.extend_from_slice(&1000i64.to_le_bytes());
        let (hits, total) = search(h, &long);
        assert_eq!(total, 12);
        assert_eq!(hits[0], (5, 1.0), "{hits:?}");
        assert_eq!(hits[1].1, 0.5);
        let geo = Blob::tree()
            .u8(NODE_GEO_DISTANCE_FEATURE)
            .str("loc")
            .f64(5.0)
            .f64(0.0)
            .f64(1000.0);
        let (hits, total) = search(h, &geo);
        assert_eq!(total, 12);
        assert_eq!(hits[0].0, 5);
        let mut bad = Blob::tree().u8(NODE_LONG_DISTANCE_FEATURE).str("date");
        bad.0.extend_from_slice(&0i64.to_le_bytes());
        bad.0.extend_from_slice(&0i64.to_le_bytes());
        let (_, e) = fails(h, &bad);
        assert!(e.contains("pivotDistance must be > 0"), "{e}");
        let (_, e) = fails(
            h,
            &Blob::tree()
                .u8(NODE_GEO_DISTANCE_FEATURE)
                .str("loc")
                .f64(91.0)
                .f64(0.0)
                .f64(1.0),
        );
        assert!(e.contains("distance feature"), "{e}");
        assert_eq!(ffi_close_jvm_reader(h), 0);
    }

    /// The seismic fixture (`fixtures/data/seismic/merged`, written by
    /// OpenSearch with neural-search) copied into a temporary directory.
    fn seismic_index(tag: &str) -> TempDir {
        let tmp = TempDir::new(tag);
        let from = format!(
            "{}/../../fixtures/data/seismic/merged",
            env!("CARGO_MANIFEST_DIR")
        );
        for e in std::fs::read_dir(from).unwrap() {
            let e = e.unwrap();
            std::fs::copy(e.path(), tmp.path().join(e.file_name())).unwrap();
        }
        tmp
    }

    /// The fixture's cases: `(tokens, top_n, k, heap_factor, filter tag,
    /// expected scores on the merged index)`.
    #[allow(clippy::type_complexity)]
    fn seismic_cases() -> Vec<(Vec<(i32, f32)>, i32, i32, f32, Option<String>, Vec<f32>)> {
        let text = std::fs::read_to_string(format!(
            "{}/../../fixtures/data/seismic/expected.tsv",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap();
        let mut cases = Vec::new();
        let mut current = None;
        for line in text.lines() {
            let cols: Vec<&str> = line.split('\t').collect();
            match cols[0] {
                "query" => {
                    current = Some((
                        Vec::new(),
                        cols[1].parse().unwrap(),
                        cols[2].parse().unwrap(),
                        cols[3].parse().unwrap(),
                        (cols[4] != "-").then(|| cols[4].to_string()),
                        Vec::new(),
                    ))
                }
                "tokens" => {
                    current.as_mut().unwrap().0 = cols[1]
                        .split(',')
                        .map(|p| {
                            let (t, w) = p.split_once(':').unwrap();
                            (t.parse().unwrap(), w.parse().unwrap())
                        })
                        .collect()
                }
                "hits" if cols[1] == "merged" => {
                    let mut case = current.take().unwrap();
                    if cols.len() > 2 && !cols[2].is_empty() {
                        case.5 = cols[2]
                            .split(',')
                            .map(|p| p.split_once(':').unwrap().1.parse().unwrap())
                            .collect();
                    }
                    cases.push(case);
                }
                _ => {}
            }
        }
        cases
    }

    fn sparse(
        tokens: &[(i32, f32)],
        top_n: i32,
        k: i32,
        heap_factor: f32,
        filter: Option<&Blob>,
        fallback: &Blob,
    ) -> Blob {
        let mut b = Blob::tree()
            .u8(NODE_SPARSE_ANN)
            .str("emb")
            .i32(tokens.len() as i32);
        for &(t, w) in tokens {
            b = b.i32(t).f32(w);
        }
        b = b.i32(top_n).i32(k).f32(heap_factor);
        b = match filter {
            None => b.u8(0),
            Some(f) => b.u8(1).raw(f),
        };
        b.raw(fallback)
    }

    fn feature(field: &str, feature: &str, function: &Blob) -> Blob {
        Blob::default()
            .u8(NODE_FEATURE)
            .str(field)
            .str(feature)
            .raw(function)
    }

    #[test]
    fn the_sparse_node_scores_as_neural_search_does() {
        let tmp = seismic_index("jvm-queries-sparse");
        let h = open(&tmp);
        let none = Blob::default().u8(6);
        let mut checked = 0;
        for (tokens, top_n, k, heap_factor, tag, want) in seismic_cases() {
            if want.len() > 20 {
                continue;
            }
            let filter = tag.map(|t| Blob::default().term("tag", &t));
            let q = sparse(&tokens, top_n, k, heap_factor, filter.as_ref(), &none);
            let got: Vec<f32> = search(h, &q).0.into_iter().map(|(_, s)| s).collect();
            assert_eq!(got, want, "{tokens:?} {top_n} {k} {heap_factor}");
            checked += 1;
        }
        assert!(checked >= 30, "{checked}");
        for (bad, what) in [
            (
                sparse(&[(-1, 1.0)], 1, 1, 1.0, None, &none),
                "negative sparse token",
            ),
            (sparse(&[(1, 1.0)], 0, 1, 1.0, None, &none), "top_n 0"),
            (sparse(&[(1, 1.0)], 1, -1, 1.0, None, &none), "k -1"),
        ] {
            let (_, e) = fails(h, &bad);
            assert!(e.contains(what), "{e}");
        }
        let mut flag = sparse(&[(1, 1.0)], 1, 1, 1.0, None, &none);
        let at = flag.0.len() - 2;
        flag.0[at] = 2;
        let (_, e) = fails(h, &flag);
        assert!(e.contains("sparse filter flag 2"), "{e}");
        assert_eq!(ffi_close_jvm_reader(h), 0);
    }

    #[test]
    fn the_feature_node_scores_the_features_value() {
        let tmp = TempDir::new("jvm-queries-feature");
        let dir = FsDirectory::open(tmp.path());
        let version = LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        };
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", version).unwrap();
        for v in [1.0f32, 4.0, 2.0] {
            let mut d = Document::new();
            d.add(FeatureField::new("f", "x", v).unwrap());
            w.add_fields_document(&d).unwrap();
        }
        w.commit().unwrap();
        drop(w);
        let h = open(&tmp);
        let value = |v: f32| FeatureField::decode_feature_value((v.to_bits() >> 15) as f32);
        let linear = Blob::tree()
            .u8(3)
            .f32(2.0)
            .raw(&feature("f", "x", &Blob::default().u8(0)));
        assert_eq!(
            search(h, &linear),
            (
                vec![
                    (1, 2.0 * value(4.0)),
                    (2, 2.0 * value(2.0)),
                    (0, 2.0 * value(1.0))
                ],
                3
            )
        );
        // Each function: the same documents, ordered alike.
        for function in [
            Blob::default().u8(1).f32(1.0),
            Blob::default().u8(2).f32(1.0),
            Blob::default().u8(3).f32(1.0).f32(0.5),
        ] {
            let (hits, n) = search(h, &Blob::tree().raw(&feature("f", "x", &function)));
            assert_eq!(n, 3);
            assert_eq!(hits.iter().map(|&(d, _)| d).collect::<Vec<_>>(), [1, 2, 0]);
            assert!(hits.iter().all(|&(_, s)| s > 0.0));
        }
        let none = feature("f", "nope", &Blob::default().u8(0));
        assert_eq!(search(h, &Blob::tree().raw(&none)).1, 0);
        let (_, e) = fails(
            h,
            &Blob::tree().raw(&feature("f", "x", &Blob::default().u8(4))),
        );
        assert!(e.contains("unknown feature function 4"), "{e}");
        assert_eq!(ffi_close_jvm_reader(h), 0);
    }
}
