#![cfg(test)]
use super::*;
use std::collections::HashSet;

#[test]
fn codec_pickers_answer_javas_formats() {
    let codec = get_default_codec();
    assert_eq!(codec.name(), "Lucene104");
    assert_eq!(get_postings_format("f"), "Lucene104");
    assert_eq!(get_doc_values_format("f"), "Lucene90");
    assert_eq!(
        get_default_postings_format(),
        Lucene104PostingsFormat::default()
    );
    let sized = get_default_postings_format_sized(10, 20).unwrap();
    assert_eq!(
        (sized.min_term_block_size, sized.max_term_block_size),
        (10, 20)
    );
    assert!(get_default_postings_format_sized(30, 40).is_err());
    assert_eq!(
        get_default_doc_values_format(),
        Lucene90DocValuesFormat::default()
    );
    assert_eq!(get_default_knn_vectors_format().graph(), Some((16, 100)));

    let asserting = InstalledAssertingCodec::new();
    assert_eq!(get_postings_format_of(&asserting, "f"), "Asserting");
    assert_eq!(get_doc_values_format_of(&asserting, "f"), "Asserting");
    let ords = get_postings_format_with_ords(&mut JavaRandom::new(1));
    let codec = always_postings_format(ords);
    assert_eq!(codec.name(), "Asserting");
    assert_eq!(get_postings_format_of(codec.as_ref(), "x"), ords.name());
    assert_eq!(get_doc_values_format_of(codec.as_ref(), "x"), "Asserting");
    let codec = always_doc_values_format(Lucene90DocValuesFormat::default());
    assert_eq!(get_postings_format_of(codec.as_ref(), "x"), "Asserting");
    assert_eq!(get_doc_values_format_of(codec.as_ref(), "x"), "Lucene90");
    let codec = always_knn_vectors_format(get_default_knn_vectors_format());
    assert_eq!(
        codec.knn_vectors_format_for_field("x"),
        get_default_knn_vectors_format()
    );
    assert!(codec.stored_fields_writer_factory().is_some());
}

#[test]
fn postings_format_with_ords_is_one_next_int_draw() {
    for seed in 0..64 {
        let want = JavaRandom::new(seed * 7919).next_int_bounded(2);
        let got = get_postings_format_with_ords(&mut JavaRandom::new(seed * 7919));
        let name = if want == 0 {
            "LuceneFixedGap"
        } else {
            "BlockTreeOrds"
        };
        assert_eq!(got.name(), name);
    }
}

#[test]
fn next_long_small_and_large_ranges() {
    let mut r = JavaRandom::new(1);
    let mut twin = JavaRandom::new(1);
    assert_eq!(
        next_long(&mut r, 5, 9),
        5 + i64::from(twin.next_int_bounded(5))
    );
    for _ in 0..500 {
        let v = next_long(&mut r, i64::MIN, i64::MAX);
        let _ = v;
        let w = next_long(&mut r, -10, 1 << 40);
        assert!((-10..=1 << 40).contains(&w));
    }
    // The large branch is `start + floor(range * nextDouble)`.
    let mut r = JavaRandom::new(9);
    let d = JavaRandom::new(9).next_double();
    let want = ((1u128 << 40) as f64 * d) as i64;
    assert_eq!(next_long(&mut r, 0, (1 << 40) - 1), want);
}

#[test]
fn next_int_in_covers_each_branch_of_bounded_next_int() {
    // A power-of-two range masks the one draw.
    let (mut r, mut twin) = (JavaRandom::new(3), JavaRandom::new(3));
    assert_eq!(next_int_in(&mut r, 10, 18), 10 + (twin.next_int() & 7));
    // An odd range rejects the over-represented top quarter here.
    let mut r = JavaRandom::new(4);
    for _ in 0..200 {
        let v = next_int_in(&mut r, -5, 0x6000_0000 - 5);
        assert!((-5..0x6000_0000 - 5).contains(&v));
    }
    // A range wider than `int` redraws until the value is inside.
    for _ in 0..50 {
        let v = next_int_in(&mut r, -2, i32::MAX);
        assert!((-2..i32::MAX).contains(&v));
    }
}

#[test]
#[should_panic(expected = "bound must be greater than origin")]
fn next_int_in_refuses_an_empty_range() {
    next_int_in(&mut JavaRandom::new(1), 3, 3);
}

#[test]
#[should_panic(expected = "start=2,end=1")]
fn next_long_refuses_an_empty_range() {
    next_long(&mut JavaRandom::new(1), 2, 1);
}

#[test]
fn simple_strings_stay_in_range() {
    let mut r = JavaRandom::new(2);
    for _ in 0..200 {
        let s = random_simple_string(&mut r);
        assert!(s.len() <= 10 && s.bytes().all(|b| b.is_ascii_lowercase()));
        let s = random_simple_string_between(&mut r, 3, 5);
        assert!((3..=5).contains(&s.len()));
        let u = random_simple_string_range(&mut r, 0xd800, 0xd801, 4);
        assert!(u.len() <= 4 && u.iter().all(|&c| (0xd800..=0xd801).contains(&c)));
    }
    assert_eq!(random_simple_string_max(&mut r, 0), "");
}

#[test]
fn unicode_strings_are_well_formed() {
    let mut r = JavaRandom::new(3);
    for _ in 0..500 {
        assert!(random_unicode_string(&mut r).encode_utf16().count() <= 20);
        let _ = random_realistic_unicode_string(&mut r);
        let s = random_realistic_unicode_string_max(&mut r, 3);
        assert!(s.chars().count() <= 3);
        for len in [0, 1, 2, 3, 4, 9] {
            assert_eq!(
                random_fixed_byte_length_unicode_string(&mut r, len).len(),
                len as usize
            );
        }
    }
    let mut buf = [0u16; 8];
    random_fixed_length_unicode_string(&mut r, &mut buf, 2, 4);
    assert_eq!(&buf[..2], &[0, 0]);
}

#[test]
fn regexpish_and_htmlish_strings_draw_every_piece() {
    let mut r = JavaRandom::new(4);
    let mut seen = String::new();
    for _ in 0..300 {
        seen.push_str(&random_regexpish_string(&mut r));
        seen.push_str(&random_htmlish_string(&mut r, 20));
    }
    for piece in ["{0,5}", "<p>", "<!--#", "&#x", "</script>", "\'", "-->"] {
        assert!(seen.contains(piece), "{piece}");
    }
    assert_eq!(random_htmlish_string(&mut r, 0), "");
}

#[test]
fn recasing_keeps_the_length() {
    let mut r = JavaRandom::new(5);
    for _ in 0..20 {
        let s = randomly_recase_code_points(&mut r, "ScRiPt\u{10400}");
        assert_eq!(s.encode_utf16().count(), 8);
        assert_eq!(s.to_lowercase(), "script\u{10428}");
    }
}

#[test]
fn binary_terms_and_big_integers() {
    let mut r = JavaRandom::new(6);
    assert!(random_binary_term(&mut r).len() < 15);
    assert_eq!(random_binary_term_len(&mut r, 7).len(), 7);
    let b = next_big_integer(&mut r, 3);
    assert!((1..=3).contains(&b.len()));
}

#[test]
fn analysis_strings_honour_their_bounds() {
    let mut r = JavaRandom::new(7);
    for _ in 0..300 {
        for simple in [true, false] {
            let s = random_analysis_string(&mut r, 30, simple);
            assert!(s.encode_utf16().count() <= 60);
        }
    }
    assert_eq!(random_sub_string(&mut r, 0, false), "");
    for _ in 0..300 {
        let s = random_sub_string(&mut r, 5, false);
        assert!(s.encode_utf16().count() <= 5, "{s:?}");
    }
}

#[test]
fn codepoint_order_is_not_code_unit_order() {
    let supp: Vec<u16> = "\u{10000}".encode_utf16().collect();
    let bmp: Vec<u16> = "\u{ffff}".encode_utf16().collect();
    assert_eq!(string_codepoint_compare(&supp, &bmp), Ordering::Greater);
    assert!(supp < bmp);
    assert_eq!(
        string_codepoint_compare(&[0x61], &[0x61, 0x62]),
        Ordering::Less
    );
    assert_eq!(
        string_codepoint_compare(&[0xd800], &[0xd800]),
        Ordering::Equal
    );
}

#[test]
fn ulp_equality() {
    assert!(float_ulp_equals(1.0, 1.0, 0));
    assert!(float_ulp_equals(
        1.0,
        f32::from_bits(1.0f32.to_bits() + 1),
        1
    ));
    assert!(!float_ulp_equals(
        1.0,
        f32::from_bits(1.0f32.to_bits() + 2),
        1
    ));
    assert!(float_ulp_equals(0.0, -0.0, 0));
    assert!(!float_ulp_equals(1.0, -1.0, 100));
    assert!(!float_ulp_equals(f32::NAN, f32::NAN, 100));
    assert!(double_ulp_equals(1.0, 1.0, 0));
    assert!(double_ulp_equals(f64::from_bits(1), -f64::from_bits(1), 2));
    assert!(!double_ulp_equals(f64::from_bits(1), -f64::from_bits(2), 2));
    assert!(!double_ulp_equals(1.0, -1.0, 100));
    assert!(!double_ulp_equals(
        1.0,
        f64::from_bits(1.0f64.to_bits() + 2),
        1
    ));
    assert!(!double_ulp_equals(f64::NAN, f64::NAN, 100));
}

#[test]
fn tables_pair_up() {
    assert!(BLOCK_STARTS
        .iter()
        .zip(BLOCK_ENDS.iter())
        .all(|(s, e)| s < e));
    assert_eq!(HTML_CHAR_ENTITIES[0], "AElig");
}

#[test]
fn iterators_and_read_only_collections() {
    check_iterator_sized([Some(1), None].into_iter(), 2, true);
    check_iterator([Some(1), Some(2)].into_iter());
    check_read_only([1, 2, 3].iter());
    let short = std::panic::catch_unwind(|| check_iterator_sized([Some(1)].into_iter(), 2, false));
    assert!(short.is_err());
    let long = std::panic::catch_unwind(|| check_iterator_sized([Some(1)].into_iter(), 0, false));
    assert!(long.is_err());
    let null =
        std::panic::catch_unwind(|| check_iterator_sized([None::<i32>].into_iter(), 1, false));
    assert!(null.is_err());
    let null = std::panic::catch_unwind(|| check_iterator([None::<i32>].into_iter()));
    assert!(null.is_err());

    /// An iterator that yields again after its end, and lies about its size.
    struct Broken(u32);
    impl Iterator for Broken {
        type Item = Option<u32>;
        fn next(&mut self) -> Option<Option<u32>> {
            self.0 += 1;
            (self.0 != 2).then_some(Some(self.0))
        }
    }
    impl ExactSizeIterator for Broken {
        fn len(&self) -> usize {
            7
        }
    }
    assert!(std::panic::catch_unwind(|| check_iterator(Broken(1))).is_err());
    assert!(std::panic::catch_unwind(|| check_iterator_sized(Broken(1), 0, false)).is_err());
    struct Finite(Broken);
    impl Iterator for Finite {
        type Item = Option<u32>;
        fn next(&mut self) -> Option<Option<u32>> {
            (self.0 .0 < 3).then(|| self.0.next()).flatten()
        }
        fn size_hint(&self) -> (usize, Option<usize>) {
            (7, Some(7))
        }
    }
    impl ExactSizeIterator for Finite {}
    assert!(std::panic::catch_unwind(|| check_read_only(Finite(Broken(5)))).is_err());
}

#[test]
fn directory_helpers() {
    use lucene_store::{ByteBuffersDirectory, DataOutput, Directory};
    let dir = ByteBuffersDirectory::new();
    assert!(!any_files_except_write_lock(&dir).unwrap());
    for name in [
        "write.lock",
        "segments_1",
        "_0.cfs",
        "_0_Lucene104_0.doc",
        "extra0",
        "notes",
    ] {
        let mut out = dir.create_output(name).unwrap();
        out.write_bytes(name.as_bytes());
        out.close().unwrap();
        assert_eq!(
            any_files_except_write_lock(&dir).unwrap(),
            name != "write.lock"
        );
    }
    let ram = ram_copy_of(&dir).unwrap();
    let mut copied = ram.list_all().unwrap();
    copied.sort();
    assert_eq!(copied, ["_0.cfs", "_0_Lucene104_0.doc", "segments_1"]);
    assert_eq!(&*ram.open("_0.cfs").unwrap(), b"_0.cfs");
}

#[test]
fn strings_of_bytes() {
    assert_eq!(bytes_ref_to_string(None), "(null)");
    assert_eq!(bytes_ref_to_string(Some(b"ab")), "ab [61 62]");
    assert_eq!(bytes_ref_to_string(Some(&[0xff, 0x0a])), "[ff a]");
    let mut r = JavaRandom::new(1);
    let mut twin = JavaRandom::new(1);
    assert_eq!(string_to_char_sequence("héllo", &mut r), "héllo");
    twin.next_int_bounded(5);
    assert_eq!(r.next_int(), twin.next_int(), "one nextInt(5) drawn");
}

#[test]
fn docs_draws_its_postings_features_like_java() {
    use lucene_codecs::blocktree::SeekStatus;
    use lucene_search::reader::{PostingsEnum, PostingsFlags, TermsEnum};

    /// Refuses every postings request, naming the flags it was asked for.
    struct Flags;
    impl TermsEnum for Flags {
        fn next(&mut self) -> lucene_search::Result<Option<&[u8]>> {
            Ok(None)
        }
        fn term(&self) -> Option<&[u8]> {
            None
        }
        fn try_seek_ceil(&mut self, _: &[u8]) -> lucene_search::Result<SeekStatus> {
            Ok(SeekStatus::End)
        }
        fn doc_freq(&mut self) -> lucene_search::Result<i32> {
            Ok(0)
        }
        fn total_term_freq(&mut self) -> lucene_search::Result<i64> {
            Ok(0)
        }
        fn postings(
            &mut self,
            flags: PostingsFlags,
        ) -> lucene_search::Result<Box<dyn PostingsEnum>> {
            Err(lucene_search::Error::Unsupported(format!("{flags:?}")))
        }
    }
    let mut seen = HashSet::new();
    let mut r = JavaRandom::new(17);
    let mut twin = JavaRandom::new(17);
    for i in 0..200 {
        let expected = if twin.next_boolean() {
            if twin.next_boolean() {
                ["Positions", "Offsets", "Payloads", "All"][twin.next_int_bounded(4) as usize]
            } else {
                "Freqs"
            }
        } else {
            "None"
        };
        let mut replay = r.clone();
        let Err(lucene_search::Error::Unsupported(got)) =
            docs(&mut r, &mut Flags, PostingsFlags::None)
        else {
            panic!("the stub refuses");
        };
        assert_eq!(got, expected, "draw {i}");
        seen.insert(got);
        let Err(lucene_search::Error::Unsupported(kept)) =
            docs(&mut replay, &mut Flags, PostingsFlags::All)
        else {
            panic!("the stub refuses");
        };
        // the random position request replaces the caller's flags; FREQS
        // never lowers ALL
        let replaced = !matches!(expected, "None" | "Freqs");
        assert_eq!(kept, if replaced { expected } else { "All" }, "draw {i}");
    }
    assert_eq!(seen.len(), 6);
}

#[test]
fn consistent_top_docs() {
    use lucene_search::collector::{TotalHits, TotalHitsRelation as Rel};
    use lucene_search::top_docs::{ShardScoreDoc, TopDocs};
    use lucene_search::top_field::{FieldDoc, TopFieldDocs};
    let hits = |v, relation| TotalHits { value: v, relation };
    let td = |total, scores: &[(i32, f32)]| TopDocs {
        total_hits: total,
        score_docs: scores
            .iter()
            .map(|&(doc, score)| ShardScoreDoc {
                doc,
                score,
                shard_index: -1,
            })
            .collect(),
    };
    let a = td(hits(5, Rel::EqualTo), &[(1, 2.0), (3, f32::NAN)]);
    assert_consistent(&a, &a.clone());
    assert_consistent(
        &a,
        &td(
            hits(4, Rel::GreaterThanOrEqualTo),
            &[(1, 2.0), (3, f32::NAN)],
        ),
    );
    let lower = td(
        hits(3, Rel::GreaterThanOrEqualTo),
        &[(1, 2.0), (3, f32::NAN)],
    );
    assert_consistent(&lower, &a);
    assert_consistent(&lower, &lower.clone());
    let zero = td(hits(0, Rel::EqualTo), &[]);
    for bad in [
        td(hits(6, Rel::EqualTo), &[(1, 2.0), (3, f32::NAN)]),
        td(
            hits(6, Rel::GreaterThanOrEqualTo),
            &[(1, 2.0), (3, f32::NAN)],
        ),
        td(hits(5, Rel::EqualTo), &[(1, 2.0)]),
        td(hits(5, Rel::EqualTo), &[(2, 2.0), (3, f32::NAN)]),
        td(hits(5, Rel::EqualTo), &[(1, 2.5), (3, f32::NAN)]),
        zero.clone(),
    ] {
        let a = a.clone();
        assert!(std::panic::catch_unwind(move || assert_consistent(&a, &bad)).is_err());
    }
    let small = td(hits(2, Rel::EqualTo), &[(1, 2.0), (3, f32::NAN)]);
    assert!(std::panic::catch_unwind(move || assert_consistent(&lower, &small)).is_err());

    let tf = |values: Vec<i64>| TopFieldDocs {
        hits: vec![FieldDoc {
            doc: 4,
            values,
            terms: vec![],
        }],
        total: hits(1, Rel::EqualTo),
        max_score: f32::NAN,
    };
    assert_consistent_field_docs(&tf(vec![7]), &tf(vec![7]));
    assert!(
        std::panic::catch_unwind(|| assert_consistent_field_docs(&tf(vec![7]), &tf(vec![8])))
            .is_err()
    );
    let mut none = tf(vec![7]);
    none.hits.clear();
    assert!(
        std::panic::catch_unwind(|| assert_consistent_field_docs(&tf(vec![7]), &none)).is_err()
    );
    let mut other = tf(vec![7]);
    other.hits[0].doc = 5;
    assert!(
        std::panic::catch_unwind(|| assert_consistent_field_docs(&tf(vec![7]), &other)).is_err()
    );
}
