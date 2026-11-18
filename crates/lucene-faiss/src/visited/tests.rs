#![cfg(test)]
use super::Visited;

#[test]
fn set_seen_and_clear() {
    let mut v = Visited::new(4);
    assert!(!v.seen(2));
    v.set(2);
    assert!(v.seen(2));
    v.clear();
    assert!(!v.seen(2));
}

#[test]
fn epoch_wraps_without_stale_marks() {
    let mut v = Visited::new(3);
    v.set(1);
    v.epoch = u32::MAX;
    v.set(0);
    v.begin(3);
    assert_eq!(v.epoch, 1);
    assert!(!v.seen(0) && !v.seen(1));
    v.begin(8);
    assert!(!v.seen(7));
}
