use super::*;

fn label(s: &str) -> FacetLabel {
    FacetLabel::new(["dim", s]).unwrap()
}

/// `TestLruTaxonomyWriterCache.testDefaultLRUTypeIsCollisionSafe` and the
/// fill/evict behaviour of both key types.
#[test]
fn fills_and_evicts_both_types() {
    assert_eq!(LruType::default(), LruType::LruString);
    for ty in [LruType::LruString, LruType::LruHashed] {
        let mut c = LruTaxonomyWriterCache::with_type(3, ty);
        assert_eq!(c.get(&label("a")), -1);
        assert!(!c.put(&label("a"), 1));
        assert!(!c.put(&label("b"), 2));
        assert!(!c.is_full());
        assert!(!c.put(&label("c"), 3));
        assert!(c.is_full());
        assert_eq!(c.size(), 3);
        // Over the limit: the cache trims itself to two thirds.
        assert!(c.put(&label("d"), 4));
        assert_eq!(c.size(), 2);
        assert_eq!(c.get(&label("d")), 4);
        assert_eq!(c.get(&label("a")), -1);
        c.clear();
        assert_eq!(c.size(), 0);
        c.close();
        assert_eq!(c.size(), 0);
        assert_eq!(c.get(&label("d")), -1);
        assert!(!c.put(&label("e"), 5));
        assert!(!c.is_full());
        c.clear();
    }
}

/// Two labels with the same `longHashCode` are distinct under `LRU_STRING`
/// and collide under `LRU_HASHED`.
#[test]
fn hashed_collides_string_does_not() {
    // "Aa" and "BB" have the same Java `String.hashCode`.
    let a = FacetLabel::new(["Aa"]).unwrap();
    let b = FacetLabel::new(["BB"]).unwrap();
    assert_eq!(a.long_hash_code(), b.long_hash_code());
    let mut s = LruTaxonomyWriterCache::new(100);
    s.put(&a, 1);
    s.put(&b, 2);
    assert_eq!((s.get(&a), s.get(&b)), (1, 2));
    let mut h = LruTaxonomyWriterCache::with_type(100, LruType::LruHashed);
    h.put(&a, 1);
    h.put(&b, 2);
    assert_eq!((h.get(&a), h.get(&b)), (2, 2));
}
