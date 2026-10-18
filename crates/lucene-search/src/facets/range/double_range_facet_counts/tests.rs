//! Unit tests.

use lucene_index::document::NumericDocValuesField;

use super::*;
use crate::facets::test_index::{collect, id_doc, match_all, plain_index, term};

#[test]
fn counts_a_double_field_and_prints_as_java() {
    let mut docs = Vec::new();
    for (id, v) in [("a", 0.5), ("b", 2.0), ("a", -1.0)] {
        let mut d = id_doc(id);
        d.add(NumericDocValuesField::new("d", f64::to_bits(v) as i64));
        docs.push(d);
    }
    let ranges = [
        DoubleRange::new("neg", f64::NEG_INFINITY, true, 0.0, false).unwrap(),
        DoubleRange::new("pos", 0.0, true, 10.0, true).unwrap(),
    ];
    plain_index(docs, &[0]).with(|searcher, reader, _| {
        let all = collect(searcher, &match_all(reader), false);
        let f = DoubleRangeFacetCounts::new("d", &all, &ranges).unwrap();
        let r = f.get_all_children("d", &[]).unwrap().unwrap();
        assert_eq!(r.value, FacetValue::Int(3));
        assert_eq!(
            format!("{f:?}"),
            "RangeFacetCounts totCount=3:\n  neg -> count=1\n  pos -> count=2\n"
        );
        assert_eq!(format!("{f}"), format!("{f:?}"));
        let f = DoubleRangeFacetCounts::with_fast_match(
            "d",
            Some(searcher),
            &all,
            Some(term("id", "b")),
            &ranges,
        )
        .unwrap();
        // The single-valued path adds each segment's hits, not the
        // query's, as Java's does.
        let r = f.get_all_children("d", &[]).unwrap().unwrap();
        assert_eq!(r.value, FacetValue::Int(2));
        assert_eq!(r.label_values[1].value, FacetValue::Int(1));
    });
}
