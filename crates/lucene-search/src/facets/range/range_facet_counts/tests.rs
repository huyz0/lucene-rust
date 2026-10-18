//! Unit tests.

use super::*;

fn counts(labels: &[&str], counts: Option<Vec<i32>>) -> RangeFacetCounts {
    let mut c = RangeFacetCounts::new(
        "f",
        labels.iter().map(|l| l.to_string()).collect(),
        Vec::new(),
        None,
    );
    c.counts = counts;
    c.tot_count = 9;
    c
}

fn kids(r: FacetResult) -> Vec<(String, i32)> {
    r.label_values
        .into_iter()
        .map(|l| (l.label, l.value.int_value()))
        .collect()
}

#[test]
fn top_children_order_by_count_then_label() {
    let c = counts(&["b", "a", "z", "c"], Some(vec![2, 2, 0, 5]));
    let top = c.get_top_children(3, "f", &[]).unwrap().unwrap();
    assert_eq!(top.child_count, 3);
    assert_eq!(
        kids(top),
        vec![("c".into(), 5), ("a".into(), 2), ("b".into(), 2)]
    );
    assert!(c.get_top_children(0, "f", &[]).is_err());
    assert!(c.get_top_children(1, "g", &[]).is_err());
    assert!(c.get_top_children(1, "f", &["x"]).is_err());
    assert!(matches!(
        c.get_specific_value("f", &["a"]),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn nothing_counted_is_all_zeros() {
    let c = counts(&["a", "b"], None);
    let top = c.get_top_children(5, "f", &[]).unwrap().unwrap();
    assert_eq!((top.value, top.child_count), (FacetValue::Int(9), 0));
    assert_eq!(
        kids(c.get_all_children("f", &[]).unwrap().unwrap()),
        vec![("a".into(), 0), ("b".into(), 0)]
    );
    assert_eq!(
        c.to_string(),
        "RangeFacetCounts totCount=9:\n  a -> count=0\n  b -> count=0\n"
    );
    assert_eq!(c.get_all_dims(1).unwrap().len(), 1);
}
