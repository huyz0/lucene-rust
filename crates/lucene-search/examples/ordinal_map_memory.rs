//! What `OrdinalMap::build` costs in memory, and how much of that cost is the
//! **materialized input** rather than the map itself.
//!
//! `OrdinalMap.build` in Java takes `TermsEnum[]` and never holds a
//! dictionary; this port takes every segment's complete term list, because the
//! only way to read a SORTED_SET doc-values dictionary here is
//! `terms_dict::decode_all_terms`, which returns all of it. c12 recorded that
//! divergence without a number; this is the number, and the argument for (or
//! against) building a streaming cursor lives on it. See
//! `docs/sweep/m2/c29-search-carryovers.md`.
//!
//! ```text
//! cargo build -p lucene-search --release --example ordinal_map_memory
//! ./target/release/examples/ordinal_map_memory <segments> <terms-per-segment> <overlap>
//! ```
//!
//! `overlap` is the fraction of each segment's dictionary shared with the
//! others (`1.0` == identical dictionaries). RSS is read from
//! `/proc/self/statm`, so this is Linux-only and deliberately coarse -- the
//! accounted byte totals beside each RSS figure are the exact allocation
//! sizes, and the two agreeing is what makes the RSS reading trustworthy.
#![allow(clippy::arithmetic_side_effects)] // A measurement harness's own sizes.

use lucene_search::ordinal_map::OrdinalMap;

fn rss_kb() -> u64 {
    let s = std::fs::read_to_string("/proc/self/statm").unwrap();
    let pages: u64 = s.split_whitespace().nth(1).unwrap().parse().unwrap();
    pages * 4
}

fn main() {
    let segments: usize = std::env::args().nth(1).unwrap().parse().unwrap();
    let terms: usize = std::env::args().nth(2).unwrap().parse().unwrap();
    let overlap: f64 = std::env::args().nth(3).unwrap().parse().unwrap();

    let base = rss_kb();
    // Each segment's dictionary: `terms` ascending byte strings. `overlap`
    // controls how much of the vocabulary segments share (1.0 == identical
    // dictionaries, the best case for a global map).
    let distinct = (terms as f64 / overlap) as usize;
    let mut input: Vec<Vec<Vec<u8>>> = Vec::with_capacity(segments);
    for s in 0..segments {
        let mut seg: Vec<Vec<u8>> = Vec::with_capacity(terms);
        let offset = (s * (distinct - terms)) / segments.max(1);
        for t in 0..terms {
            seg.push(format!("term-{:012}", offset + t).into_bytes());
        }
        input.push(seg);
    }
    let after_input = rss_kb();
    let input_bytes: usize = input
        .iter()
        .map(|s| {
            s.capacity() * std::mem::size_of::<Vec<u8>>()
                + s.iter().map(|t| t.capacity()).sum::<usize>()
        })
        .sum();

    let t0 = std::time::Instant::now();
    let map = OrdinalMap::build(&input);
    let elapsed = t0.elapsed();
    let after_build = rss_kb();
    let out_bytes: usize = (0..segments)
        .map(|s| map.segment_ords(s).map_or(0, |o| o.len()) * 8)
        .sum::<usize>()
        + map.value_count() as usize * (4 + 8);

    println!(
        "segments={segments} terms/segment={terms} distinct={} global={}",
        distinct,
        map.value_count()
    );
    println!(
        "  input  (materialized dictionaries): {:>8.1} MB   [accounted {:.1} MB]",
        (after_input - base) as f64 / 1024.0,
        input_bytes as f64 / 1048576.0
    );
    println!(
        "  build  (map on top of the input):   {:>8.1} MB   [accounted {:.1} MB]",
        (after_build - after_input) as f64 / 1024.0,
        out_bytes as f64 / 1048576.0
    );
    println!(
        "  build time:                         {:>8.1} ms",
        elapsed.as_secs_f64() * 1000.0
    );
    println!(
        "  peak RSS delta:                     {:>8.1} MB",
        (after_build - base) as f64 / 1024.0
    );
    std::hint::black_box(&map);
    std::hint::black_box(&input);
}
