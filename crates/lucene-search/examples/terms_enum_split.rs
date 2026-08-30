//! What `TermsEnum::next()` costs with and without its `TermStats`
//! (ledger item 14's sibling, item 21 / c1's F-14).
//!
//! Java's `SegmentTermsEnum.next()` decodes only the term bytes; `docFreq()`
//! and `postings()` trigger `decodeMetaData` later, and a caller that walks
//! terms to filter on their bytes never pays it. This port's `try_next`
//! returned `(&[u8], TermStats)` and so always decoded -- a stats vint plus a
//! full `TermMetadata` (the `.doc`/`.pos`/`.pay` file pointers and the
//! singleton pulsing) per term, thrown away by every term the caller rejects.
//!
//! Both arms run over the same field of the same index, alternating, in one
//! process, and every figure is a **min of N repetitions** -- criterion
//! reported 83/91/129 µs for identical code on this host
//! (`docs/sweep/m2/c24-arith-codecs.md`).
//!
//! ```text
//! cargo build -p lucene-search --release --example terms_enum_split
//! ./target/release/examples/terms_enum_split <index-dir> [reps]
//! ```
// A measurement harness's own arithmetic. See `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::hint::black_box;
use std::time::Instant;

use lucene_search::directory_reader::DirectoryReader;
use lucene_store::MmapDirectory;

fn main() {
    let mut args = std::env::args().skip(1);
    let index = args
        .next()
        .expect("usage: terms_enum_split <index-dir> [reps]");
    let reps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(40);

    let dir = MmapDirectory::open(index.clone());
    let reader = DirectoryReader::open(&dir).expect("open index");
    let opened = reader.open_segments().expect("open segments");
    let segments = opened.as_open_segments();
    let segment = segments.first().expect("index has a segment");

    // The field with the most terms: the smaller ones are dominated by the
    // per-call overhead this is not measuring.
    let (field_name, terms) = segment
        .fields
        .iter_fields()
        .max_by_key(|(_, t)| t.num_terms)
        .expect("index has an indexed field");
    println!(
        "index: {index}  field: {field_name}  terms: {}",
        terms.num_terms
    );

    let mut fused = u128::MAX;
    let mut bytes_only = u128::MAX;
    let mut counted = 0u64;
    for _ in 0..reps {
        let t = Instant::now();
        let mut it = terms.iter();
        let mut n = 0u64;
        while let Some((term, stats)) = it.try_next().expect("next") {
            n += (term.len() as u64) ^ (stats.doc_freq as u64);
        }
        fused = fused.min(t.elapsed().as_nanos());
        counted = black_box(n);

        let t = Instant::now();
        let mut it = terms.iter();
        let mut n = 0u64;
        while let Some(term) = it.try_next_term().expect("next_term") {
            n += term.len() as u64;
        }
        bytes_only = bytes_only.min(t.elapsed().as_nanos());
        black_box(n);
    }
    black_box(counted);

    let per = |ns: u128| ns as f64 / terms.num_terms.max(1) as f64;
    println!(
        "try_next      (term + stats)  {:>10.3} ms   {:>7.2} ns/term",
        fused as f64 / 1e6,
        per(fused)
    );
    println!(
        "try_next_term (term only)     {:>10.3} ms   {:>7.2} ns/term",
        bytes_only as f64 / 1e6,
        per(bytes_only)
    );
    println!(
        "speedup for a bytes-only scan: {:.2}x",
        fused as f64 / bytes_only.max(1) as f64
    );
}
