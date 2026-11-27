//! Many shards: what an idle tier costs, and what a refresh costs across them.
#![allow(clippy::arithmetic_side_effects)] // Test arithmetic on small constants.

use std::alloc::{GlobalAlloc, Layout, System};
use std::convert::Infallible;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use lucene_codecs::stored_fields::{FieldValue, StoredField};
use lucene_index::buffered_updates::Term;
use lucene_index::hot_tier::{node_bytes, set_node_limit, HeldOp, HotTier, HotTierConfig};
use lucene_index::index_writer::{ExplicitDocument, ExplicitFields};

struct Counting;
static LIVE: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every call is forwarded unchanged to the system allocator.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        LIVE.fetch_add(l.size(), Ordering::Relaxed);
        // SAFETY: `l` is the caller's layout.
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        LIVE.fetch_sub(l.size(), Ordering::Relaxed);
        // SAFETY: `p` and `l` are the pair the caller got from `alloc`.
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        LIVE.fetch_add(n, Ordering::Relaxed);
        LIVE.fetch_sub(l.size(), Ordering::Relaxed);
        // SAFETY: `p` and `l` are the pair the caller got from `alloc`.
        unsafe { System.realloc(p, l, n) }
    }
}

#[global_allocator]
static A: Counting = Counting;

/// The node pool is process-wide, so the tests of this file take turns.
static TURN: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn op(id: u32, version: u8) -> HeldOp {
    let doc = ExplicitDocument {
        stored: vec![StoredField {
            field_number: 0,
            value: FieldValue::Binary(vec![version; 4096]),
        }],
        fields: ExplicitFields::default(),
    };
    HeldOp::new(
        None,
        Term::new("_id", format!("d{id}").into_bytes()),
        vec![doc],
        None,
        Vec::new(),
    )
}

fn offer(tier: &HotTier, id: u32, version: u8) {
    let mut sink = |_: HeldOp| -> Result<(), Infallible> { Ok(()) };
    let _ = tier.offer(op(id, version), &mut sink);
}

#[test]
fn what_a_tier_costs_when_idle_and_at_a_refresh() {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let n = 10_000;
    let before = LIVE.load(Ordering::Relaxed);
    let tiers: Vec<HotTier> = (0..n)
        .map(|_| HotTier::new(HotTierConfig::with_budget(8 << 20)))
        .collect();
    let per = (LIVE.load(Ordering::Relaxed) - before) / n + std::mem::size_of::<HotTier>();
    // What a shard that has seen an update costs, before its sketch.
    assert!(per < 8 << 10, "{per} bytes");
    assert_eq!(node_bytes(), 0);
    let t = Instant::now();
    for tier in &tiers {
        let _ = tier.drain(&mut |_| Ok::<(), Infallible>(()));
    }
    let drain_ns = t.elapsed().as_nanos() as usize / n;
    println!("tier: {per} bytes; refresh of an empty tier: {drain_ns} ns");
    assert!(drain_ns < 1_000, "{drain_ns} ns");
}

#[test]
fn many_busy_shards_share_the_node_budget_fairly() {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let limit = 4 << 20;
    set_node_limit(limit);
    // 2,000 shards, each allowed 8 MiB, each with 20 hot rows of 4 KiB.
    let tiers: Vec<HotTier> = (0..2_000)
        .map(|_| HotTier::new(HotTierConfig::with_budget(8 << 20)))
        .collect();
    for round in 0..10u8 {
        for tier in &tiers {
            for id in 0..20u32 {
                offer(tier, id, round);
            }
        }
        assert!(node_bytes() <= limit, "{} > {limit}", node_bytes());
    }
    let held: Vec<usize> = tiers.iter().map(|t| t.bytes()).collect();
    let busy = held.iter().filter(|b| **b > 0).count();
    let total: usize = held.iter().sum();
    assert_eq!(total, node_bytes());
    // Each tier that holds anything holds at most an equal share of the
    // limit (plus the share's own rounding), and the node is mostly used.
    let share = limit / busy.max(1);
    assert!(held.iter().all(|b| *b <= share * 2), "share {share}");
    assert!(total > limit / 2, "{total} of {limit}");
    println!(
        "{busy} of {} shards hold {total} of {limit} bytes",
        tiers.len()
    );
    // A closed shard gives its bytes back, and a drained one too.
    drop(tiers);
    assert_eq!(node_bytes(), 0);
    set_node_limit(0);
}

#[test]
fn a_lone_busy_shard_may_use_the_whole_node_budget() {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    set_node_limit(2 << 20);
    let tier = HotTier::new(HotTierConfig::with_budget(8 << 20));
    for round in 0..10u8 {
        for id in 0..400u32 {
            offer(&tier, id, round);
        }
    }
    assert!(tier.bytes() > (2 << 20) * 3 / 4, "{}", tier.bytes());
    assert!(tier.bytes() <= 2 << 20);
    drop(tier);
    assert_eq!(node_bytes(), 0);
    set_node_limit(0);
}
