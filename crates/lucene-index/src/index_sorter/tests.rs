#![cfg(test)]
use super::*;
use crate::segment_info::{NumericSortKey, StringMissingValue};
use lucene_codecs::doc_values::DenseField;
use lucene_codecs::field_infos::{DocValuesType, FieldInfo, FieldInfos};

#[test]
fn selector_picks_java_s_positions() {
    let one = [7];
    let even = [1, 2, 3, 4];
    let odd = [1, 2, 3];
    for sel in [
        SortedSetSelector::Min,
        SortedSetSelector::Max,
        SortedSetSelector::MiddleMin,
        SortedSetSelector::MiddleMax,
    ] {
        assert_eq!(select_sorted_set(&one, sel), Some(&7));
        assert_eq!(select_sorted_set::<i32>(&[], sel), None);
    }
    assert_eq!(select_sorted_set(&even, SortedSetSelector::Min), Some(&1));
    assert_eq!(select_sorted_set(&even, SortedSetSelector::Max), Some(&4));
    // (4 - 1) >>> 1 == 1 and 4 >>> 1 == 2: the two middle values.
    assert_eq!(
        select_sorted_set(&even, SortedSetSelector::MiddleMin),
        Some(&2)
    );
    assert_eq!(
        select_sorted_set(&even, SortedSetSelector::MiddleMax),
        Some(&3)
    );
    assert_eq!(
        select_sorted_set(&odd, SortedSetSelector::MiddleMin),
        Some(&2)
    );
    assert_eq!(
        select_sorted_set(&odd, SortedSetSelector::MiddleMax),
        Some(&2)
    );
}

#[test]
fn ranks_are_global_unsigned_byte_ordinals() {
    let a = vec![
        Some(b"b".to_vec()),
        None,
        Some(vec![0xff]),
        Some(b"b".to_vec()),
    ];
    let b = vec![Some(b"a".to_vec()), Some(b"ba".to_vec()), None];
    let ranked = rank_terms(&[a, b]);
    // Distinct values in unsigned order: "a" < "b" < "ba" < [0xff] --
    // 0xff is the *largest* byte, not -1 as a signed compare would say.
    assert_eq!(ranked[0], vec![Some(1), None, Some(3), Some(1)]);
    assert_eq!(ranked[1], vec![Some(0), Some(2), None]);
    assert!(rank_terms(&[]).is_empty());
    assert_eq!(rank_terms(&[vec![None]]), vec![vec![None]]);
}

fn sort(kind: IndexSortKind) -> IndexSortField {
    IndexSortField {
        field: "f".to_string(),
        reverse: false,
        kind,
    }
}

/// One segment of four documents with three byte-keyed columns, sparse
/// in each: field 0 SORTED, field 1 SORTED_SET (one document with
/// four values), field 2 BINARY.
fn segment() -> (DocValuesMeta, Vec<u8>) {
    let id = [7u8; 16];
    let sorted = [(0, b"m".to_vec()), (2, b"c".to_vec()), (3, b"m".to_vec())];
    let set = [
        (
            0,
            vec![b"d".to_vec(), b"a".to_vec(), b"c".to_vec(), b"b".to_vec()],
        ),
        (3, vec![b"z".to_vec()]),
    ];
    let binary = [(1, vec![0xffu8]), (2, b"".to_vec())];
    let (dvm, dvd, _dvs) = doc_values::write_dense_fields(
        &[
            DenseField::SparseSorted(0, &sorted),
            DenseField::SparseSortedSet(1, &set),
            DenseField::SparseBinary(2, &binary),
        ],
        4,
        &id,
        "",
    )
    .expect("write doc values");
    let types = [
        DocValuesType::Sorted,
        DocValuesType::SortedSet,
        DocValuesType::Binary,
    ];
    let fis = FieldInfos {
        fields: types
            .iter()
            .enumerate()
            .map(|(n, t)| {
                let mut fi = FieldInfo::new(format!("f{n}"), n as i32);
                fi.doc_values_type = *t;
                fi
            })
            .collect(),
    };
    let (_, meta) = doc_values::parse_meta(&dvm, &id, "", &fis).expect("parse meta");
    (meta, dvd)
}

#[test]
fn reads_each_byte_keyed_column_per_document() {
    let (meta, dvd) = segment();
    let s = |b: &[u8]| Some(b.to_vec());
    let string = sort(IndexSortKind::String(StringMissingValue::None));
    assert_eq!(
        read_segment_terms(&dvd, &meta, &string, 0, 4).unwrap(),
        Some(vec![s(b"m"), None, s(b"c"), s(b"m")])
    );
    for (selector, doc0) in [
        (SortedSetSelector::Min, b"a"),
        (SortedSetSelector::Max, b"d"),
        (SortedSetSelector::MiddleMin, b"b"),
        (SortedSetSelector::MiddleMax, b"c"),
    ] {
        let set = sort(IndexSortKind::SortedSet {
            selector,
            missing: StringMissingValue::Last,
        });
        assert_eq!(
            read_segment_terms(&dvd, &meta, &set, 1, 4).unwrap(),
            Some(vec![s(doc0), None, None, s(b"z")]),
            "{selector:?}"
        );
    }
    let binary = sort(IndexSortKind::Binary(StringMissingValue::First));
    assert_eq!(
        read_segment_terms(&dvd, &meta, &binary, 2, 4).unwrap(),
        Some(vec![None, s(&[0xff]), s(b""), None])
    );
}

#[test]
fn numeric_kinds_and_absent_columns_are_not_byte_keyed() {
    let (meta, dvd) = segment();
    let numeric = sort(IndexSortKind::Numeric(NumericSortKey::Long(None)));
    assert!(read_segment_terms(&dvd, &meta, &numeric, 0, 4)
        .unwrap()
        .is_none());
    // Field 9 has no column at all, and field 2 has no SORTED one.
    for (kind, field) in [
        (IndexSortKind::String(StringMissingValue::None), 9),
        (IndexSortKind::String(StringMissingValue::None), 2),
        (
            IndexSortKind::SortedSet {
                selector: SortedSetSelector::Min,
                missing: StringMissingValue::None,
            },
            9,
        ),
        (IndexSortKind::Binary(StringMissingValue::None), 9),
    ] {
        assert!(read_segment_terms(&dvd, &meta, &sort(kind), field, 4)
            .unwrap()
            .is_none());
    }
}

#[test]
fn an_ordinal_past_the_dictionary_is_corruption() {
    let err = term_at(&[b"a".to_vec()], Some(1)).unwrap_err();
    assert!(
        err.to_string().contains("outside a dictionary of 1 terms"),
        "{err}"
    );
    assert!(term_at(&[], Some(-1)).is_err());
    assert_eq!(term_at(&[b"a".to_vec()], None).unwrap(), None);
}
