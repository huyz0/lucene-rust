#![cfg(test)]
// Test fixtures' own arithmetic -- see `docs/arithmetic-gate.md`'s "Test code".
#![allow(clippy::arithmetic_side_effects)]
use super::*;

fn key(s: &str) -> Arc<str> {
    Arc::from(s)
}

#[test]
fn pages_round_trip_through_slots_and_clock_spares_referenced_and_pinned_ones() {
    let tmp = lucene_util::test_support::TempDir::new("page-disk-clock");
    // Pages of 4 KiB: slots of 4 KiB and 8 KiB (a page plus overhead),
    // under a budget of 16 KiB.
    let d = DiskTier::open(&tmp.path().join("t"), 4 * 4096, 4096).unwrap();
    assert_eq!(d.class_len, [4096, 8192]);
    let (a, b) = (key("a"), key("b"));
    d.put(&a, 0, &[1; 100]);
    d.pin("a");
    d.put(&a, 1, &[2; 4096]);
    d.put(&b, 0, &[3; 4097]);
    assert_eq!(d.used(), 100 + 4096 + 4097);
    assert_eq!(d.footprint(), 4096 * 2 + 8192);
    assert_eq!(&d.get(&a, 1).unwrap()[..], &[2; 4096][..]);
    assert_eq!(&d.get(&b, 0).unwrap()[..], &[3; 4097][..]);
    // Full: CLOCK skips the pinned pages of `a` and evicts `b`'s page,
    // whose 8 KiB then hold two small pages.
    d.put(&b, 1, &[4; 10]);
    d.put(&b, 2, &[5; 10]);
    assert!(d.contains(&a, 0) && d.contains(&a, 1));
    assert!(!d.contains(&b, 0) && d.contains(&b, 1) && d.contains(&b, 2));
    assert_eq!(d.footprint(), 4 * 4096);
    // A pinned page still gets a slot, past the budget, evicting what
    // it can first; an unpinned one does not when only pinned remain.
    d.put(&a, 2, &[6; 10]);
    d.put(&a, 3, &[7; 10]);
    d.put(&a, 4, &[8; 10]);
    assert!(d.contains(&a, 2) && d.contains(&a, 3) && d.contains(&a, 4));
    assert!(!d.contains(&b, 1) && !d.contains(&b, 2));
    d.put(&b, 3, &[9; 10]);
    assert!(!d.contains(&b, 3), "every slot pinned");
    // A second put of a cached page is a no-op; an oversized one too.
    d.put(&a, 0, &[9; 100]);
    assert_eq!(&d.get(&a, 0).unwrap()[..], &[1; 100][..]);
    d.put(&b, 4, &vec![0; 9000]);
    assert!(!d.contains(&b, 4));
    assert!(d.get(&b, 99).is_none() && d.get(&key("z"), 0).is_none());
    d.forget_prefix("a");
    assert_eq!((d.used(), d.footprint()), (0, 0));
    d.put(&b, 5, &[7; 10]);
    assert_eq!(&d.get(&b, 5).unwrap()[..], &[7; 10][..]);
    assert!(d.metadata_bytes() > 0);
}

#[test]
fn small_pages_take_small_slots_and_freed_space_is_released() {
    let tmp = lucene_util::test_support::TempDir::new("page-disk-classes");
    let root = tmp.path().join("t");
    // 256 KiB pages: classes 4 KiB to 256 KiB, then 260 KiB (a page
    // plus a decoder's overhead).
    let d = DiskTier::open(&root, 1 << 30, 256 << 10).unwrap();
    assert_eq!(d.class_len.len(), 8);
    assert_eq!(*d.class_len.last().unwrap(), (256 + 4) << 10);
    // More classes than a slot id has bits for are refused.
    assert!(DiskTier::open(&tmp.path().join("big"), 1 << 30, 128 << 20).is_err());
    let k = key("f");
    d.put(&k, 0, &[1; 100]);
    d.put(&k, 1, &[2; 20_000]);
    d.put(&k, 2, &vec![3; 256 << 10]);
    d.put(&k, 3, &vec![5; (256 << 10) + 1]);
    assert_eq!(d.footprint(), 4096 + 32_768 + (256 << 10) + (260 << 10));
    assert_eq!(&d.get(&k, 1).unwrap()[..], &[2; 20_000][..]);
    assert_eq!(&d.get(&k, 2).unwrap()[..], &vec![3; 256 << 10][..]);
    d.forget_prefix("f");
    assert_eq!(d.footprint(), 0);
    // On Linux the freed slots' blocks go back to the filesystem.
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::MetadataExt;
        let blocks: u64 = std::fs::read_dir(&root)
            .unwrap()
            .map(|e| e.unwrap().metadata().unwrap().blocks())
            .sum();
        assert!(blocks * 512 < 64 << 10, "{blocks} blocks still allocated");
    }
    // A freed slot is reused by its class.
    d.put(&k, 0, &[4; 100]);
    assert_eq!(&d.get(&k, 0).unwrap()[..], &[4; 100][..]);
    assert_eq!(std::fs::read_dir(&root).unwrap().count(), 4);
}

#[test]
fn a_page_that_cannot_fit_evicts_nothing() {
    let tmp = lucene_util::test_support::TempDir::new("page-disk-nofit");
    // 4 KiB pages under a 16 KiB budget: 8 KiB pinned, two small pages.
    let d = DiskTier::open(&tmp.path().join("t"), 4 * 4096, 4096).unwrap();
    let (a, b) = (key("a"), key("b"));
    d.pin("a");
    d.put(&a, 0, &[1; 4097]);
    d.put(&b, 0, &[2; 10]);
    d.put(&b, 1, &[3; 10]);
    assert_eq!(d.footprint(), 4 * 4096);
    // A second 8 KiB pinned page fits only past the budget; then an
    // unpinned 8 KiB page could not fit even with both small pages
    // gone, so they stay.
    d.put(&a, 1, &[4; 4097]);
    assert!(d.contains(&a, 1));
    d.put(&key("c"), 0, &[5; 4097]);
    assert!(!d.contains(&key("c"), 0));
    d.forget_prefix("a");
    assert_eq!(lock_ignoring_poison(&d.state).held, 0);
    d.put(&key("c"), 0, &[5; 4097]);
    assert!(d.contains(&key("c"), 0));
}

#[test]
fn many_pages_cost_a_few_bytes_each_and_few_files() {
    let tmp = lucene_util::test_support::TempDir::new("page-disk-many");
    let d = DiskTier::open(&tmp.path().join("t"), 20_000 * 8192, 4096).unwrap();
    let k = key("f");
    for p in 0..10_000u64 {
        d.put(&k, p, &[p as u8; 16]);
    }
    // 16-byte slots, 4-byte ring and page entries, plus vector growth
    // slack.
    assert!(d.metadata_bytes() < 10_000 * 48, "{}", d.metadata_bytes());
    assert_eq!(std::fs::read_dir(tmp.path().join("t")).unwrap().count(), 3);
    assert_eq!(&d.get(&k, 9_999).unwrap()[..], &[(9_999u64) as u8; 16][..]);
}

#[test]
fn a_failed_write_frees_its_slot_and_a_forgotten_file_is_not_published() {
    let tmp = lucene_util::test_support::TempDir::new("page-disk-fail");
    let root = tmp.path().join("t");
    let d = DiskTier::open(&root, 8 * 4096, 4096).unwrap();
    let k = key("f");
    d.put(&k, 0, &[1; 10]);
    // Slab files gone and their directory replaced by a file: new slabs
    // cannot be created.
    std::fs::remove_dir_all(&root).unwrap();
    std::fs::write(&root, b"x").unwrap();
    lock_ignoring_poison(&d.slabs)
        .iter_mut()
        .for_each(Vec::clear);
    d.put(&k, 1, &[2; 10]);
    assert!(!d.contains(&k, 1));
    assert!(d.get(&k, 0).is_none(), "unreadable slab is a miss");
    assert!(!d.contains(&k, 0), "and dropped, so it is cached again");
    assert_eq!(d.used(), 0);
    assert!(DiskTier::open(&root.join("sub"), 1, 4096).is_err());
    assert!(DiskTier::open(&tmp.path().join("u"), 1, usize::MAX).is_err());
}
