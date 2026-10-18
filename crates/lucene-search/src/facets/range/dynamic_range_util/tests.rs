//! Unit tests.

use super::*;

fn spans(r: &[DynamicRangeInfo]) -> Vec<(i32, i64, i64, i64)> {
    r.iter()
        .map(|i| (i.count, i.weight, i.min, i.max))
        .collect()
}

#[test]
fn ranges_cut_where_the_weight_reaches_its_share() {
    let mut v = vec![4, 1, 3, 2];
    let mut w = vec![1, 1, 1, 1];
    let r = compute_dynamic_numeric_ranges(&mut v, &mut w, 4, 4, 2);
    assert_eq!(spans(&r), vec![(2, 2, 1, 2), (2, 2, 3, 4)]);
    assert_eq!(r[0].centroid, 1.5);
    // Sorted in place.
    assert_eq!(v, vec![1, 2, 3, 4]);
}

#[test]
fn the_remainder_is_the_last_range_and_nothing_is_no_ranges() {
    let mut v = vec![1, 2, 3];
    let mut w = vec![5, 1, 1];
    let r = compute_dynamic_numeric_ranges(&mut v, &mut w, 3, 7, 2);
    assert_eq!(spans(&r), vec![(1, 5, 1, 1), (2, 2, 2, 3)]);
    assert!(compute_dynamic_numeric_ranges(&mut v, &mut w, 0, 0, 2).is_empty());
    assert!(compute_dynamic_numeric_ranges(&mut v, &mut w, 3, 7, 0).is_empty());
    assert!(compute_dynamic_numeric_ranges(&mut v, &mut w, 3, 7, -1).is_empty());
}
