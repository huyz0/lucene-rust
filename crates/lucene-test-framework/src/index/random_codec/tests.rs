#![cfg(test)]
use super::*;

#[test]
fn hash_keeps_java_units_collisions_and_overflow() {
    assert_eq!(java_string_hash(&[]), 0);
    assert_eq!(
        java_string_hash(&[0x41, 0x61]),
        java_string_hash(&[0x42, 0x42])
    );
    assert_eq!(java_string_hash(&[0xd83d, 0xde00]), 1_772_899);
    assert_eq!(java_string_hash(&[0xd800]), 0xd800);
    assert_eq!(java_string_hash(&[0xdc00]), 0xdc00);
    assert_eq!(
        java_string_hash(&"field2".encode_utf16().collect::<Vec<_>>()),
        -1_274_708_360
    );
}

#[test]
fn boxed_cache_keeps_java_hash_collisions_distinct() {
    let first_key = [0x41, 0x61];
    let second_key = [0x42, 0x42];
    assert_eq!(java_string_hash(&first_key), java_string_hash(&second_key));

    let formats = vec![Rc::new(1), Rc::new(2)];
    let mut cache: FieldCache<i32> = HashMap::new();
    let first = select(&formats, &mut cache, 0, Some(&first_key)).unwrap();
    let second = select(&formats, &mut cache, 0, Some(&second_key)).unwrap();
    assert!(Rc::ptr_eq(&first, &second));
    assert_eq!(cache.len(), 2);
    assert!(Rc::ptr_eq(&cache[first_key.as_slice()], &first));
    assert!(Rc::ptr_eq(&cache[second_key.as_slice()], &second));
}

#[test]
fn signed_minimum_selection_and_insert_before_assert_are_java() {
    let formats = vec![Rc::new(1), Rc::new(2), Rc::new(3)];
    let mut cache: FieldCache<i32> = HashMap::new();
    assert_eq!(
        select(&formats, &mut cache, i32::MIN, Some(&[])),
        Err(Error::NegativeFormatIndex)
    );
    assert_eq!(
        select(&formats[..2], &mut cache, i32::MIN, Some(&[])),
        Ok(Rc::clone(&formats[0]))
    );
    assert_eq!(select(&[], &mut cache, 0, None), Err(Error::NullField));
    assert_eq!(
        select(&[], &mut cache, 0, Some(&[1])),
        Err(Error::EmptyFormats)
    );
    cache.clear();
    for i in 0..9999u16 {
        assert!(select(&formats, &mut cache, 0, Some(&[i])).is_ok());
    }
    assert_eq!(
        select(&formats, &mut cache, 0, Some(&[9999])),
        Err(Error::TooManyFields)
    );
    let retry = select(&formats, &mut cache, 0, Some(&[9999])).unwrap();
    assert!(Rc::ptr_eq(&retry, &formats[0]));
    assert_eq!(cache.len(), 10000);
}

#[test]
fn shared_avoid_mutation_does_not_rebuild_candidates_or_merge_equal_instances() {
    let avoid = Rc::new(RefCell::new(BTreeSet::new()));
    let mut random = JavaRandom::new(42);
    let mut codec =
        RandomCodec::with_avoid(&mut random, false, 1, Some(Rc::clone(&avoid))).unwrap();
    assert!(Rc::ptr_eq(codec.avoid_codecs(), &avoid));
    let before = codec
        .postings_formats()
        .iter()
        .map(Rc::clone)
        .collect::<Vec<_>>();
    avoid
        .borrow_mut()
        .extend(codec.format_names().iter().cloned());
    assert!(before
        .iter()
        .zip(codec.postings_formats())
        .all(|(a, b)| Rc::ptr_eq(a, b)));
    assert_eq!(codec.format_names().len(), 10);
    let equal: Vec<_> = codec
        .doc_values_formats()
        .iter()
        .filter(|f| matches!(***f, DocValuesFormat::Lucene90 { .. }))
        .collect();
    assert_eq!(equal[0], equal[1]);
    assert!(!Rc::ptr_eq(equal[0], equal[1]));
    assert_eq!(codec.postings_format_for_field(None), Err(Error::NullField));
    assert_eq!(
        codec.doc_values_format_for_field(None),
        Err(Error::NullField)
    );
    let points = codec.points_format();
    let another = codec.points_format();
    assert_eq!(points, another);
    assert!(!std::ptr::eq(points.as_ref(), another.as_ref()));
    assert_eq!(
        points.field_split_seed(&[0xd800]),
        points.bkd_split_random_seed ^ 0xd800
    );
    assert_eq!(codec.postings_mapping_count(), 0);
    assert_eq!(codec.doc_values_mapping_count(), 0);
    assert_eq!(codec.per_field_seed(), -1_170_105_035);
}

#[test]
fn errors_have_explicit_assertion_enabled_classes_and_messages() {
    for (error, class, message) in [
        (
            Error::NullAvoidCodecs,
            "NullPointerException",
            "null avoidance set",
        ),
        (Error::NullField, "NullPointerException", "null field name"),
        (Error::EmptyFormats, "ArithmeticException", "/ by zero"),
        (
            Error::NegativeFormatIndex,
            "IndexOutOfBoundsException",
            "negative format index after Math.abs(Integer.MIN_VALUE)",
        ),
        (Error::TooManyFields, "AssertionError", "test went insane"),
    ] {
        assert_eq!(error.java_class(), class);
        assert_eq!(error.to_string(), message);
    }
}

fn outcome(codec: &mut RandomCodec, dv: bool, field: Option<&[u16]>) -> String {
    let result = if dv {
        codec
            .doc_values_format_for_field(field)
            .map(|format| format.name())
    } else {
        codec
            .postings_format_for_field(field)
            .map(|format| format.name())
    };
    match result {
        Ok(name) => format!("ok:{name}"),
        Err(error) => error.java_class().to_string(),
    }
}

#[test]
fn actual_java_injected_minimum_cache_limit_and_null_constructor_rows_match() {
    for (nightly, multiplier, fixture) in [
        (
            false,
            1,
            include_str!("../../../../../fixtures/testframework/data/random_codec_false_1.tsv"),
        ),
        (
            false,
            3,
            include_str!("../../../../../fixtures/testframework/data/random_codec_false_3.tsv"),
        ),
        (
            true,
            1,
            include_str!("../../../../../fixtures/testframework/data/random_codec_true_1.tsv"),
        ),
        (
            true,
            3,
            include_str!("../../../../../fixtures/testframework/data/random_codec_true_3.tsv"),
        ),
    ] {
        let mut actual = String::new();
        for dv in [false, true] {
            let mut codec = RandomCodec::new(&mut JavaRandom::new(42), nightly, multiplier);
            if !dv {
                codec.formats.truncate(3);
            }
            codec.per_field_seed = i32::MIN;
            actual.push_str(&format!(
                "min-abs\t{dv}\t{}\n",
                outcome(&mut codec, dv, Some(&[]))
            ));
            actual.push_str(&format!(
                "min-abs-retry\t{dv}\t{}\n",
                outcome(&mut codec, dv, Some(&[]))
            ));
            if dv {
                codec.dv_formats.truncate(2);
            } else {
                codec.formats.truncate(2);
            }
            actual.push_str(&format!(
                "min-abs-zero-remainder\t{dv}\t{}\n",
                outcome(&mut codec, dv, Some(&[]))
            ));

            let mut codec = RandomCodec::new(&mut JavaRandom::new(42), nightly, multiplier);
            for i in 0..9999 {
                let key: Vec<_> = format!("cache{i}").encode_utf16().collect();
                assert!(outcome(&mut codec, dv, Some(&key)).starts_with("ok:"));
            }
            let key: Vec<_> = "cache9999".encode_utf16().collect();
            actual.push_str(&format!(
                "cache-limit\t{dv}\t{}\n",
                outcome(&mut codec, dv, Some(&key))
            ));
            actual.push_str(&format!(
                "cache-retry\t{dv}\t{}\n",
                outcome(&mut codec, dv, Some(&key))
            ));
            let next: Vec<_> = "cache10000".encode_utf16().collect();
            actual.push_str(&format!(
                "cache-next\t{dv}\t{}\n",
                outcome(&mut codec, dv, Some(&next))
            ));
            let size = if dv {
                codec.doc_values_mapping_count()
            } else {
                codec.postings_mapping_count()
            };
            actual.push_str(&format!("cache-size\t{dv}\t{size}\n"));
        }
        let avoid = Rc::new(RefCell::new(BTreeSet::new()));
        let codec = RandomCodec::with_avoid(
            &mut JavaRandom::new(42),
            nightly,
            multiplier,
            Some(Rc::clone(&avoid)),
        )
        .unwrap();
        let before: Vec<_> = codec.postings_formats().iter().map(Rc::clone).collect();
        avoid
            .borrow_mut()
            .extend(codec.format_names().iter().cloned());
        actual.push_str(&format!(
            "avoid-mutation\t{}\t{}\t{}\n",
            Rc::ptr_eq(codec.avoid_codecs(), &avoid),
            before
                .iter()
                .zip(codec.postings_formats())
                .all(|(a, b)| Rc::ptr_eq(a, b)),
            avoid.borrow().len()
        ));
        let mut random = JavaRandom::new(42);
        let error = RandomCodec::with_avoid(&mut random, nightly, multiplier, None).unwrap_err();
        actual.push_str(&format!(
            "null-avoid\t{}\t{}\n",
            error.java_class(),
            random.next_int()
        ));
        let expected = fixture
            .lines()
            .skip_while(|row| !row.starts_with("min-abs\t"))
            .map(|row| format!("{row}\n"))
            .collect::<String>();
        assert_eq!(
            actual, expected,
            "nightly={nightly},multiplier={multiplier}"
        );
    }
}
