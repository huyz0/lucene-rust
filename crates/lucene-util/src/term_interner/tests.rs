#![cfg(test)]
use super::*;

#[test]
fn same_bytes_intern_to_same_id() {
    let mut pool = TermInterner::new();
    let a = pool.intern(b"apple");
    let b = pool.intern(b"apple");
    assert_eq!(a, b);
    assert_eq!(pool.len(), 1);
}

#[test]
fn different_bytes_get_different_ids() {
    let mut pool = TermInterner::new();
    let a = pool.intern(b"apple");
    let b = pool.intern(b"banana");
    assert_ne!(a, b);
    assert_eq!(pool.len(), 2);
}

#[test]
fn round_trip_lookup_by_id() {
    let mut pool = TermInterner::new();
    let a = pool.intern(b"lucene");
    let b = pool.intern(b"rust");
    assert_eq!(pool.get(a), Some(&b"lucene"[..]));
    assert_eq!(pool.get(b), Some(&b"rust"[..]));
}

#[test]
fn empty_term_is_valid_and_distinct() {
    let mut pool = TermInterner::new();
    let empty1 = pool.intern(b"");
    let empty2 = pool.intern(b"");
    let nonempty = pool.intern(b"x");
    assert_eq!(empty1, empty2);
    assert_ne!(empty1, nonempty);
    assert_eq!(pool.get(empty1), Some(&b""[..]));
    assert_eq!(pool.len(), 2);
}

#[test]
fn lookup_of_unknown_id_returns_none() {
    let pool = TermInterner::new();
    assert_eq!(pool.get(TermId(0)), None);

    let mut pool2 = TermInterner::new();
    pool2.intern(b"only-term");
    assert_eq!(pool2.get(TermId(5)), None);
}

#[test]
fn index_returns_sequential_assignment_order() {
    let mut pool = TermInterner::new();
    let a = pool.intern(b"first");
    let b = pool.intern(b"second");
    let c = pool.intern(b"third");
    assert_eq!(a.index(), 0);
    assert_eq!(b.index(), 1);
    assert_eq!(c.index(), 2);
}

#[test]
fn intern_str_matches_intern_bytes() {
    let mut pool = TermInterner::new();
    let a = pool.intern_str("hello");
    let b = pool.intern(b"hello");
    assert_eq!(a, b);
}

#[test]
fn is_empty_reflects_state() {
    let mut pool = TermInterner::new();
    assert!(pool.is_empty());
    pool.intern(b"x");
    assert!(!pool.is_empty());
}

#[test]
fn stress_dedup_reduces_stored_entries() {
    // Simulate a realistic postings-like stream: a small vocabulary of
    // terms recurring across many "postings", the way the same term
    // bytes show up in document after document. Interning must collapse
    // this down to exactly the distinct-vocabulary count, not one entry
    // per call.
    let vocab: Vec<String> = (0..50).map(|i| format!("term-{i}")).collect();
    let mut pool = TermInterner::with_capacity(vocab.len());

    let total_calls = 20_000;
    let mut ids = Vec::with_capacity(total_calls);
    for i in 0..total_calls {
        let term = &vocab[i % vocab.len()];
        ids.push(pool.intern_str(term));
    }

    // Dedup actually happened: far fewer stored terms than calls.
    assert_eq!(pool.len(), vocab.len());
    assert!(pool.len() < total_calls);

    // Every occurrence of the same vocabulary word maps to the same ID,
    // and round-trips back to the right bytes.
    for i in 0..total_calls {
        let expected_word = &vocab[i % vocab.len()];
        assert_eq!(
            pool.get(ids[i]).unwrap(),
            expected_word.as_bytes(),
            "call {i}"
        );
        assert_eq!(ids[i], ids[i % vocab.len()], "call {i}");
    }
}

/// Terms sharing a hash chain: every one is still found, and only by
/// its own bytes (the chain walk compares them).
#[test]
fn colliding_hashes_chain() {
    let mut pool = TermInterner::new();
    let a = pool.intern(b"a");
    let b = pool.intern(b"b");
    // Force a collision: hang `b` under `a`'s hash too.
    let ha = pool.hasher.hash_one(&b"a"[..]);
    pool.next[b.0 as usize] = a.0;
    pool.heads.insert(ha, b.0);
    assert_eq!(pool.intern(b"a"), a);
    assert_eq!(pool.intern(b"b"), b);
    let c = pool.intern(b"c");
    assert_eq!((pool.len(), c.index()), (3, 2));
    let mut h = HashIsKey::default();
    h.write(&[1, 2]);
    assert_eq!(h.finish(), 0x102);
}

#[test]
fn many_distinct_terms_each_get_unique_ids() {
    let mut pool = TermInterner::new();
    let n = 5_000;
    let mut ids = Vec::with_capacity(n);
    for i in 0..n {
        ids.push(pool.intern_str(&format!("unique-{i}")));
    }
    assert_eq!(pool.len(), n);

    // All IDs distinct.
    let mut sorted = ids.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), n);

    // Round trip every one.
    for (i, id) in ids.iter().enumerate() {
        assert_eq!(pool.get(*id).unwrap(), format!("unique-{i}").as_bytes());
    }
}
