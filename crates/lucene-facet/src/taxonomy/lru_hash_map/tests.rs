use super::*;

/// `TestLRUHashMap`: the least recently *used* entry goes first.
#[test]
fn evicts_least_recently_used() {
    let mut m: LruHashMap<&str, i32> = LruHashMap::new(3);
    assert!(m.is_empty());
    assert_eq!(m.max_size(), 3);
    m.put("one", 1);
    m.put("two", 2);
    m.put("three", 3);
    assert_eq!(m.len(), 3);
    m.put("four", 4);
    assert_eq!(m.len(), 3);
    assert!(!m.contains_key(&"one"));
    assert_eq!(m.get(&"two"), Some(&2));
    m.put("five", 5);
    assert!(m.contains_key(&"two"));
    assert!(!m.contains_key(&"three"));
    assert_eq!(m.keys(), [&"four", &"two", &"five"]);
    // Re-putting a key refreshes it and returns the old value.
    assert_eq!(m.put("four", 44), Some(4));
    assert_eq!(m.keys(), [&"two", &"five", &"four"]);
    m.set_max_size(1);
    m.put("six", 6);
    assert_eq!(m.len(), 3, "one eviction per put");
    // Replacing an existing key's value never evicts, over the limit or not.
    assert_eq!(m.put("six", 66), Some(6));
    assert_eq!(m.len(), 3, "a replace does not evict");
    m.clear();
    assert!(m.is_empty());
    assert_eq!(m.get(&"six"), None);
}

#[test]
fn linked_lru_reuses_slots() {
    let mut l: LinkedLru<i32, i32> = LinkedLru::default();
    assert_eq!(l.remove_eldest(), None);
    for i in 0..4 {
        l.insert(i, i * 10);
    }
    // Touching the tail moves nothing.
    assert_eq!(l.get(&3), Some(&30));
    // Touching the head.
    assert_eq!(l.get(&0), Some(&0));
    assert_eq!(l.keys(), [&1, &2, &3, &0]);
    assert_eq!(l.remove_eldest(), Some((1, 10)));
    assert_eq!(l.remove_eldest(), Some((2, 20)));
    l.insert(7, 70);
    assert_eq!(l.slots.len(), 4, "a freed slot is reused");
    assert_eq!(l.keys(), [&3, &0, &7]);
    assert_eq!(l.len(), 3);
    assert!(!l.is_empty());
    // A middle entry.
    assert_eq!(l.get(&0), Some(&0));
    assert_eq!(l.keys(), [&3, &7, &0]);
    while l.remove_eldest().is_some() {}
    assert!(l.is_empty());
    assert!(l.keys().is_empty());
    assert!(l.clone().is_empty());
}
