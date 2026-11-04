use super::*;

fn label(s: &str) -> FacetLabel {
    FacetLabel::new(["dim", s]).unwrap()
}

#[test]
fn put_reports_overflow_and_make_room_drops_eldest() {
    let mut c = NameIntCacheLru::new(3);
    assert_eq!(c.max_size(), 3);
    assert!(!c.put(&label("a"), 1));
    assert!(!c.put(&label("b"), 2));
    assert!(!c.put(&label("c"), 3));
    assert!(!c.make_room_lru(), "not over the limit");
    // An access makes "a" the most recent.
    assert_eq!(c.get(&label("a")), Some(1));
    assert!(c.put(&label("d"), 4));
    assert_eq!(c.size(), 4);
    // Down to 2*3/3 = 2: "b" and "c" go.
    assert!(c.make_room_lru());
    assert_eq!(c.size(), 2);
    assert_eq!(c.get(&label("b")), None);
    assert_eq!(c.get(&label("c")), None);
    assert_eq!(c.get(&label("a")), Some(1));
    assert_eq!(c.get(&label("d")), Some(4));
    assert_eq!(c.stats(), "#miss=2 #hit=3");
    assert_eq!((c.n_misses, c.n_hits), (2, 3));
    c.clear();
    assert_eq!(c.size(), 0);
}

#[test]
fn put_prefix_keys_by_subpath() {
    let mut c = NameIntCacheLru::new(10);
    let full = FacetLabel::new(["a", "b", "c"]).unwrap();
    c.put_prefix(&full, 2, 7);
    assert_eq!(c.get(&FacetLabel::new(["a", "b"]).unwrap()), Some(7));
    assert_eq!(c.get(&full), None);
}

#[test]
fn hashed_cache_keys_by_long_hash() {
    let mut c = NameHashIntCacheLru::new(2);
    assert!(!c.put(&label("a"), 1));
    assert_eq!(c.get(&label("a")), Some(1));
    assert_eq!(ByLongHash::key(&label("a")), label("a").long_hash_code());
    assert_eq!(ByLabel::key(&label("a")), label("a"));
    assert!(!c.put(&label("b"), 2));
    assert!(c.put(&label("c"), 3));
    assert!(c.make_room_lru());
    // keep = 2*2/3 = 1.
    assert_eq!(c.size(), 1);
}

/// A limit of zero: `keep` is 0, so everything goes.
#[test]
fn zero_limit_empties() {
    let mut c = NameIntCacheLru::new(0);
    assert!(c.put(&label("a"), 1));
    assert!(c.make_room_lru());
    assert_eq!(c.size(), 0);
    // A negative limit: every put overflows.
    let mut neg = NameIntCacheLru::new(-3);
    assert!(neg.put(&label("a"), 1));
    // keep = -2, n = 3 > 1 entry: drops what there is.
    assert!(neg.make_room_lru());
    assert_eq!(neg.size(), 0);
}

/// `i32::MAX` is Java's plain `HashMap`: a `get` finds the entry (by label
/// or by hash) without reordering, so it is never full.
#[test]
fn unbounded_cache_is_a_plain_map() {
    let mut c = NameIntCacheLru::new(i32::MAX);
    let mut h = NameHashIntCacheLru::new(i32::MAX);
    for (i, s) in ["a", "b", "c"].iter().enumerate() {
        assert!(!c.put(&label(s), i as i32));
        assert!(!h.put(&label(s), i as i32));
    }
    assert_eq!(c.get(&label("a")), Some(0));
    assert_eq!(h.get(&label("a")), Some(0));
    assert_eq!(c.get(&label("x")), None);
    assert_eq!(h.get(&label("x")), None);
    assert_eq!((c.n_hits, c.n_misses, h.n_hits, h.n_misses), (1, 1, 1, 1));
    assert!(!c.make_room_lru());
    assert_eq!(c.size(), 3);
}
