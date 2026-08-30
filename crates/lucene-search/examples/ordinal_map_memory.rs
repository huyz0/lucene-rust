//! What `OrdinalMap::build` costs in memory, and how much of that cost is the
//! **materialized input** rather than the map itself.
//!
//! `OrdinalMap.build` in Java takes `TermsEnum[]` and never holds a
//! dictionary. c29 measured what taking every segment's complete term list
//! instead cost, and c38 closed it: [`OrdinalMap::build_streaming`] takes
//! cursors. This example runs either arm, one per process, so the peak RSS it
//! reports is the arm's own and not whatever the other arm left behind in the
//! allocator.
//!
//! ```text
//! cargo build -p lucene-search --release --example ordinal_map_memory
//! ./target/release/examples/ordinal_map_memory <segments> <terms> <overlap> [arm]
//! ```
//!
//! `overlap` is the fraction of each segment's dictionary shared with the
//! others (`1.0` == identical dictionaries). `arm` is `materialized` (the
//! default) or `streaming`. RSS is read from `/proc/self/status`, so this is
//! Linux-only and deliberately coarse -- the accounted byte totals beside each
//! figure are the exact allocation sizes, and the two agreeing is what makes
//! the RSS reading trustworthy.
#![allow(clippy::arithmetic_side_effects)] // A measurement harness's own sizes.

use lucene_search::ordinal_map::{OrdinalMap, TermCursor};

/// A [`TermCursor`] that *generates* its segment's dictionary as it is asked
/// for it, into one reused buffer -- standing in for
/// `lucene_codecs::terms_dict::TermsCursor` over a real `.dvd`, which this
/// example has no file to open. What the measurement is about is what
/// `build_streaming` itself holds, so what matters is that the cursor holds
/// nothing and allocates nothing per term.
struct GeneratedCursor {
    offset: usize,
    terms: usize,
    next: usize,
    term: Vec<u8>,
}

impl TermCursor for GeneratedCursor {
    fn next_term(&mut self) -> lucene_store::Result<Option<&[u8]>> {
        if self.next >= self.terms {
            return Ok(None);
        }
        write_term(&mut self.term, self.offset + self.next);
        self.next += 1;
        Ok(Some(&self.term))
    }
}

/// `term-%012d`, written in place so the streaming arm's per-term cost is a
/// dozen stores rather than a `String` allocation -- otherwise the harness,
/// not the merge, is what the timing measures.
fn write_term(out: &mut Vec<u8>, n: usize) {
    out.clear();
    out.extend_from_slice(b"term-");
    let mut digits = [b'0'; 12];
    let mut v = n;
    for slot in digits.iter_mut().rev() {
        *slot = b'0' + (v % 10) as u8;
        v /= 10;
    }
    out.extend_from_slice(&digits);
}

fn status_kb(key: &str) -> u64 {
    let s = std::fs::read_to_string("/proc/self/status").unwrap();
    s.lines()
        .find(|l| l.starts_with(key))
        .and_then(|l| l.split_whitespace().nth(1).and_then(|v| v.parse().ok()))
        .unwrap_or(0)
}

fn rss_kb() -> u64 {
    status_kb("VmRSS:")
}

fn main() {
    let segments: usize = std::env::args().nth(1).unwrap().parse().unwrap();
    let terms: usize = std::env::args().nth(2).unwrap().parse().unwrap();
    let overlap: f64 = std::env::args().nth(3).unwrap().parse().unwrap();
    let arm = std::env::args()
        .nth(4)
        .unwrap_or_else(|| "materialized".into());

    let base = rss_kb();
    let whole = std::time::Instant::now();
    let distinct = (terms as f64 / overlap) as usize;
    let offset = |s: usize| (s * (distinct - terms)) / segments.max(1);

    let (elapsed, map, input_kb, input_bytes) = match arm.as_str() {
        // Every segment's dictionary, held whole, then merged -- what this
        // port did before c38 and what a caller that needs the lists anyway
        // still does.
        "materialized" => {
            let mut input: Vec<Vec<Vec<u8>>> = Vec::with_capacity(segments);
            for s in 0..segments {
                let mut seg: Vec<Vec<u8>> = Vec::with_capacity(terms);
                let mut scratch = Vec::new();
                for t in 0..terms {
                    write_term(&mut scratch, offset(s) + t);
                    seg.push(scratch.clone());
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
            (elapsed, map, after_input - base, input_bytes)
        }
        // Java's own shape: enumerators in, no dictionary held.
        "streaming" => {
            let mut generated: Vec<GeneratedCursor> = (0..segments)
                .map(|s| GeneratedCursor {
                    offset: offset(s),
                    terms,
                    next: 0,
                    term: Vec::new(),
                })
                .collect();
            let mut cursors: Vec<&mut dyn TermCursor> = generated
                .iter_mut()
                .map(|c| c as &mut dyn TermCursor)
                .collect();
            let t0 = std::time::Instant::now();
            let map = OrdinalMap::build_streaming(&mut cursors).expect("build_streaming");
            let elapsed = t0.elapsed();
            (elapsed, map, 0, 0)
        }
        other => panic!("unknown arm {other:?}"),
    };

    let out_bytes: usize = (0..segments)
        .map(|s| map.segment_ords(s).map_or(0, |o| o.len()) * 8)
        .sum::<usize>()
        + map.value_count() as usize * (4 + 8);

    println!(
        "arm={arm} segments={segments} terms/segment={terms} distinct={distinct} global={}",
        map.value_count()
    );
    println!(
        "  input  (materialized dictionaries): {:>8.1} MB   [accounted {:.1} MB]",
        input_kb as f64 / 1024.0,
        input_bytes as f64 / 1048576.0
    );
    println!(
        "  the map itself:                     {:>8.1} MB",
        out_bytes as f64 / 1048576.0
    );
    println!(
        "  build time:                         {:>8.1} ms",
        elapsed.as_secs_f64() * 1000.0
    );
    // The materialized arm's `build time` excludes producing the term lists,
    // which the streaming arm produces inside its own timer -- so this is the
    // only figure the two arms can be compared on.
    println!(
        "  total (terms produced + build):     {:>8.1} ms",
        whole.elapsed().as_secs_f64() * 1000.0
    );
    println!(
        "  peak RSS (VmHWM) over the baseline: {:>8.1} MB",
        (status_kb("VmHWM:") - base) as f64 / 1024.0
    );
    std::hint::black_box(&map);
}
