//! The hot tier's byte estimate against what the allocator really holds: the
//! budget is only as good as `op_bytes`, so a counting allocator checks that
//! the estimate is never below the real memory and not far above it.
#![allow(clippy::arithmetic_side_effects)] // Test arithmetic on small constants.

use std::alloc::{GlobalAlloc, Layout, System};
use std::convert::Infallible;
use std::sync::atomic::{AtomicUsize, Ordering};

use lucene_codecs::stored_fields::{FieldValue, StoredField};
use lucene_index::buffered_updates::{DocValuesUpdate, Term};
use lucene_index::hot_tier::{HeldOp, HotTier, HotTierConfig, Offer};
use lucene_index::index_writer::{ExplicitDocument, ExplicitFields, InvertedField, InvertedTerm};

struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every call is forwarded unchanged to the system allocator, which
// upholds the contract; only a counter is added.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        LIVE.fetch_add(l.size(), Ordering::Relaxed);
        // SAFETY: `l` is the caller's layout, passed through as given.
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

fn op(i: u32, source: usize, terms: usize, nested: usize) -> HeldOp {
    let doc = |n: usize| {
        let body: Vec<InvertedTerm> = (0..terms)
            .map(|t| InvertedTerm {
                term: format!("word{t}x{n}").into_bytes(),
                freq: 2,
                positions: vec![1, 5],
                offsets: Vec::new(),
                payloads: Vec::new(),
            })
            .collect();
        ExplicitDocument {
            stored: vec![
                StoredField {
                    field_number: 1,
                    value: FieldValue::Binary(vec![7; source]),
                },
                StoredField {
                    field_number: 2,
                    value: FieldValue::String(format!("document-{i}")),
                },
            ],
            fields: ExplicitFields {
                inverted: vec![
                    InvertedField {
                        field_number: 3,
                        terms: vec![InvertedTerm {
                            term: format!("d{i}").into_bytes(),
                            freq: 1,
                            positions: Vec::new(),
                            offsets: Vec::new(),
                            payloads: Vec::new(),
                        }]
                        .into(),
                        norm: None,
                    },
                    InvertedField {
                        field_number: 4,
                        terms: body.into(),
                        norm: Some(terms as i64),
                    },
                ],
                doc_values: (5..8)
                    .map(|f| StoredField {
                        field_number: f,
                        value: FieldValue::Long(i64::from(i)),
                    })
                    .collect(),
                ..ExplicitFields::default()
            },
        }
    };
    let id = Term::new("_id", format!("d{i}").into_bytes());
    HeldOp::new(
        None,
        id.clone(),
        (0..=nested).map(doc).collect(),
        None,
        vec![DocValuesUpdate::Numeric {
            term: id,
            field: "__soft_deletes".to_string(),
            value: Some(1),
        }],
    )
}

/// Held rows of a few shapes: the estimate covers what the allocator holds.
#[test]
fn the_estimate_covers_the_real_memory_without_far_exceeding_it() {
    let mut ratios = Vec::new();
    for (source, terms, nested, rows) in
        [(600, 70, 0, 2000), (20_000, 400, 0, 300), (300, 20, 3, 800)]
    {
        let tier = HotTier::new(HotTierConfig::with_budget(1 << 30));
        let base_estimate = tier.bytes();
        let mut sink = |_: HeldOp| -> Result<(), Infallible> { Ok(()) };
        let before = LIVE.load(Ordering::Relaxed);
        for i in 0..rows {
            // The first two updates go through and the third is held.
            for _ in 0..3 {
                if let Ok(Offer::Through { .. }) =
                    tier.offer(op(i, source, terms, nested), &mut sink)
                {}
            }
        }
        let real = LIVE.load(Ordering::Relaxed).saturating_sub(before);
        let estimate = tier.bytes() - base_estimate;
        assert_eq!(tier.stats().held_rows, rows as usize);
        let ratio = estimate as f64 / real as f64;
        println!("source {source} terms {terms} nested {nested}: estimate {estimate} real {real} ratio {ratio:.2}");
        ratios.push(ratio);
        let _ = tier.drain(&mut sink);
    }
    for r in ratios {
        assert!((1.0..=1.6).contains(&r), "estimate/real = {r:.2}");
    }
}
