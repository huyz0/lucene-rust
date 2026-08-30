//! Settles c6 finding on `FieldNorms`'s sparse norms lookup: an eagerly decoded
//! `Vec<i32>` of every doc-with-a-norm plus a binary search per lookup, against
//! a per-scorer `IndexedDISI` cursor walked forward across the scan.
//!
//! Four things are measured, because the change moves cost between them:
//!
//! 1. **`open`** -- what building a `FieldNorms` costs. The old shape decoded
//!    the whole `IndexedDISI` region into a `Vec<i32>` *in the constructor*, and
//!    a `FieldNorms` is built per query per leaf, so that was per-query
//!    O(documents with the field) time and 4 bytes per such document of
//!    allocation. The new shape slices the region and stops.
//! 2. **A whole query's scan** -- construction *plus* every matching document
//!    in ascending order, which is what one leaf of one query actually costs.
//!    Both arms build their own structure inside the timed loop, because both
//!    shapes really do build one per query per leaf.
//! 3. **One isolated lookup, both structures already built** -- the honest
//!    like-for-like: a binary search over the decoded `Vec` against a one-shot
//!    cursor walking the block headers from the start. This is the arm the old
//!    shape *wins*, and it is measured that way deliberately: an earlier cut of
//!    this bench built the `Vec` inside `b.iter()` while the cursor arm reused a
//!    `FieldNorms` built outside, which charged the eager arm a full region
//!    decode per iteration and so measured construction, not lookup.
//! 4. **A step of an already-positioned cursor** -- what a scan actually pays
//!    per document once it is walking forward, as opposed to the one-shot.
//!
//! Read together they say: the change moves cost out of construction (paid per
//! query per leaf) and into the isolated random lookup (which a scan amortises
//! away, and which only `explain` does in isolation). A net win, not a win on
//! every axis -- which is why all four are measured.
//!
//! Run with `cargo bench -p lucene-search --bench field_norms_sparse`.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use lucene_codecs::norms::NormsEntry;
use lucene_search::field_norms::FieldNorms;

/// `IndexedDISI`'s "no rank table" marker (Java's `denseRankPower == -1`).
const NO_RANK: u8 = 0xFF;

/// A sparse-norms field where one document in `stride` has a norm.
fn sparse_field(num_present: usize, stride: i32) -> (Vec<u8>, NormsEntry, Vec<i32>) {
    let present: Vec<i32> = (0..num_present as i32).map(|i| i * stride).collect();
    let (disi, disi_jumps) = lucene_codecs::indexed_disi::write(&present);
    let mut data = disi.clone();
    let norms_offset = data.len() as i64;
    data.extend((0..present.len()).map(|i| (i % 251 + 1) as u8));
    let entry = NormsEntry {
        field_number: 0,
        docs_with_field_offset: 0,
        docs_with_field_length: disi.len() as i64,
        jump_table_entry_count: disi_jumps,
        dense_rank_power: NO_RANK,
        num_docs_with_field: present.len() as i32,
        bytes_per_norm: 1,
        norms_offset,
    };
    (data, entry, present)
}

/// The pre-c6 shape, reproduced here so the two are measured against each other
/// rather than against a remembered number: decode every doc id once, then
/// binary-search that `Vec` per lookup.
fn eager_doc_ids(data: &[u8], entry: &NormsEntry) -> Vec<i32> {
    let start = entry.docs_with_field_offset as usize;
    let len = entry.docs_with_field_length as usize;
    lucene_codecs::indexed_disi::decode_doc_ids(&data[start..start + len], entry.dense_rank_power)
        .expect("legal region")
}

fn bench(c: &mut Criterion) {
    for &num_present in &[1_000usize, 100_000] {
        let (data, entry, present) = sparse_field(num_present, 3);

        let mut group = c.benchmark_group("field_norms_sparse/construct");
        group.bench_function(BenchmarkId::new("eager_vec", num_present), |b| {
            b.iter(|| black_box(eager_doc_ids(black_box(&data), black_box(&entry))).len())
        });
        group.bench_function(BenchmarkId::new("cursor", num_present), |b| {
            b.iter(|| {
                black_box(FieldNorms::from_field_stats(
                    black_box(&data),
                    black_box(entry),
                    250_000,
                    num_present as i32,
                ))
                .avg_field_length
            })
        });
        group.finish();

        let norms = FieldNorms::from_field_stats(&data, entry, 250_000, num_present as i32);

        // Construction included in BOTH arms: a `FieldNorms` is built per query
        // per leaf either way, so charging it to only one of them would be the
        // same mistake the `lookup_only` group's comment describes, inverted.
        let mut group = c.benchmark_group("field_norms_sparse/per_query_scan");
        group.bench_function(BenchmarkId::new("eager_vec", num_present), |b| {
            b.iter(|| {
                let ids = eager_doc_ids(&data, &entry);
                let mut acc = 0.0f32;
                for &doc in &present {
                    let ordinal = lucene_codecs::indexed_disi::rank_of(&ids, doc).unwrap();
                    acc +=
                        lucene_codecs::norms::read_value_at_ordinal(&data, &entry, ordinal as i64)
                            .unwrap() as f32;
                }
                black_box(acc)
            })
        });
        group.bench_function(BenchmarkId::new("cursor", num_present), |b| {
            b.iter(|| {
                let built = FieldNorms::from_field_stats(&data, entry, 250_000, num_present as i32);
                let mut cursor = built.cursor();
                let mut acc = 0.0f32;
                for &doc in &present {
                    acc += cursor.field_length(doc).unwrap();
                }
                black_box(acc)
            })
        });
        group.finish();

        // Both structures built OUTSIDE the timed loop, so this measures the
        // lookup and nothing else. Building the `Vec` inside `b.iter()` -- which
        // an earlier cut of this bench did -- charges the eager arm a full
        // region decode per iteration and makes the cursor look ~1000x faster at
        // a job it is in fact slower at.
        let mut group = c.benchmark_group("field_norms_sparse/lookup_only");
        let target = present[present.len() / 2];
        let ids = eager_doc_ids(&data, &entry);
        group.bench_function(BenchmarkId::new("eager_vec", num_present), |b| {
            b.iter(|| {
                black_box(lucene_codecs::indexed_disi::rank_of(
                    &ids,
                    black_box(target),
                ))
            })
        });
        group.bench_function(BenchmarkId::new("cursor_one_shot", num_present), |b| {
            b.iter(|| black_box(norms.field_length(black_box(target)).unwrap()))
        });
        group.finish();

        // The same lookup on an already-positioned cursor: one forward step,
        // which is what the scan above pays per document.
        let mut group = c.benchmark_group("field_norms_sparse/warm_cursor_step");
        group.bench_function(BenchmarkId::new("cursor", num_present), |b| {
            let mut cursor = norms.cursor();
            let mut i = 0usize;
            b.iter(|| {
                // Restart when the field is exhausted; amortised over
                // `present.len()` steps, so it does not distort the per-step
                // number.
                if i == present.len() {
                    cursor = norms.cursor();
                    i = 0;
                }
                let doc = present[i];
                i += 1;
                black_box(cursor.field_length(doc).unwrap())
            })
        });
        group.finish();
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
