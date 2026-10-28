//! Unit tests.

use lucene_index::document::{NumericDocValuesField, SortedNumericDocValuesField};

use super::*;
use crate::facets::test_index::{collect, id_doc, plain_index, term};

#[test]
fn counts_single_and_multi_valued_fields() {
    let mut docs = Vec::new();
    for (id, s, m) in [
        ("a", 3, vec![1, 1, 2]),
        ("b", 3, vec![]),
        ("a", -7, vec![2]),
    ] {
        let mut d = id_doc(id);
        d.add(NumericDocValuesField::new("s", s));
        for v in m {
            d.add(SortedNumericDocValuesField::new("m", v));
        }
        docs.push(d);
    }
    plain_index(docs, &[0]).with(|searcher, reader, _| {
        let f = LongValueFacetCounts::count_all("s", reader).unwrap();
        assert!(f.get_specific_value("s", &["3"]).is_err());
        let all = f.get_all_children("s", &[]).unwrap().unwrap();
        assert_eq!(all.value, FacetValue::Int(3));
        assert_eq!(format!("{f:?}"), f.to_string());
        let sorted = f.get_all_children_sort_by_value();
        assert_eq!(sorted.label_values[0].label, "-7");

        let a = collect(searcher, &term("id", "a"), false);
        let m = LongValueFacetCounts::new("m", &a).unwrap();
        // The first document's 1 counts once.
        let top = m.get_top_children(1, "m", &[]).unwrap().unwrap();
        assert_eq!((top.value, top.child_count), (FacetValue::Int(2), 2));
    });
}

/// Lucene 10.5's `countAll(reader, field)` skips a document `liveDocs`
/// clears, but `countAll(reader, LongValuesSource)` and `countAll(reader,
/// MultiLongValuesSource)` loop over `0..maxDoc` and count it (the
/// `value_facets` fixture's `multi` index, which has deletions, records
/// both): pinned here on a small index with one deleted document.
#[test]
fn count_all_skips_deleted_documents_only_in_the_field_form() {
    use crate::facets::multi_long_values_source as mlvs;
    use crate::facets::test_index::build;
    use crate::facets::FacetsConfig;
    use crate::values_source::{long_from_long_field, ValuesContext};
    let mut docs = Vec::new();
    for (id, s, m) in [
        ("keep", 3, vec![1, 2]),
        ("gone", 4, vec![2, 9]),
        ("keep", 3, vec![9]),
    ] {
        let mut d = id_doc(id);
        d.add(NumericDocValuesField::new("s", s));
        for v in m {
            d.add(SortedNumericDocValuesField::new("m", v));
        }
        docs.push((Vec::new(), d));
    }
    build(&FacetsConfig::new(), docs, &[], &["gone"]).with(|searcher, reader, _| {
        assert_eq!(reader.num_docs(), 2);
        let ctx = ValuesContext::new(searcher);
        let labels =
            |f: &LongValueFacetCounts, dim: &str| -> (FacetValue, Vec<(String, FacetValue)>) {
                let all = f.get_all_children(dim, &[]).unwrap().unwrap();
                let lv = all
                    .label_values
                    .iter()
                    .map(|lv| (lv.label.clone(), lv.value))
                    .collect();
                (all.value, lv)
            };
        let s_forms = [
            LongValueFacetCounts::count_all("s", reader).unwrap(),
            LongValueFacetCounts::count_all_with_values_source(
                "s",
                long_from_long_field("s").as_ref(),
                &ctx,
                reader,
            )
            .unwrap(),
            LongValueFacetCounts::count_all_with_multi_values_source(
                "s",
                mlvs::from_single_valued(long_from_long_field("s")).as_ref(),
                &ctx,
                reader,
            )
            .unwrap(),
        ];
        assert_eq!(
            labels(&s_forms[0], "s"),
            (
                FacetValue::Int(2),
                vec![("3".to_string(), FacetValue::Int(2))]
            ),
            "the field form skips the deleted document's 4"
        );
        for f in &s_forms[1..] {
            assert_eq!(
                labels(f, "s"),
                (
                    FacetValue::Int(3),
                    vec![
                        ("3".to_string(), FacetValue::Int(2)),
                        ("4".to_string(), FacetValue::Int(1)),
                    ]
                ),
                "a values source counts the deleted document's 4: {f}"
            );
        }
        let m_forms = [
            LongValueFacetCounts::count_all("m", reader).unwrap(),
            LongValueFacetCounts::count_all_with_multi_values_source(
                "m",
                mlvs::from_long_field("m").as_ref(),
                &ctx,
                reader,
            )
            .unwrap(),
        ];
        assert_eq!(
            labels(&m_forms[0], "m"),
            (
                FacetValue::Int(2),
                vec![
                    ("1".to_string(), FacetValue::Int(1)),
                    ("2".to_string(), FacetValue::Int(1)),
                    ("9".to_string(), FacetValue::Int(1)),
                ]
            ),
            "the field form skips the deleted document's 2 and 9"
        );
        assert_eq!(
            labels(&m_forms[1], "m"),
            (
                FacetValue::Int(3),
                vec![
                    ("1".to_string(), FacetValue::Int(1)),
                    ("2".to_string(), FacetValue::Int(2)),
                    ("9".to_string(), FacetValue::Int(2)),
                ]
            ),
            "a multi-valued source counts the deleted document too"
        );
    });
}
