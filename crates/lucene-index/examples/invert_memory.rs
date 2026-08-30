//! What one flush's inverted form costs in memory, against the **indexed
//! field's** text it was built from -- the instrument for `LEDGER.md` item 15,
//! the block-pool redesign.
//!
//! Not directly comparable to c3's "8.3 MB of text becomes 78.5 MB, 9.4x":
//! that ratio's denominator is the whole document arena, stored `id` field
//! included, where this one is the body text alone (4.90 MB on the default
//! shape). The MB column is what carries across; the ratio is not.
//!
//! Java's `IndexingChain` inverts into `BytesRefHash` +
//! `ByteBlockPool`/`IntBlockPool`/`ByteSlicePool` and pays *zero* heap objects
//! per occurrence; this port allocates per token, per term and per posting, so
//! the ratio between the two columns below is the whole finding.
//!
//! ```text
//! cargo build -p lucene-index --release --example invert_memory
//! ./target/release/examples/invert_memory [docs] [tokens-per-doc] [vocab]
//! ```
//!
//! The default shape is `benchmarks/rust-runner`'s `index-bench` corpus:
//! 20 000 documents x 40 tokens drawn from a 20 000-word vocabulary. RSS is
//! read from `/proc/self/status`, so this is Linux-only; `ram_bytes_used` is
//! an exact count of the structure and is the number to compare across runs,
//! with RSS beside it as the corroborating (allocator-dependent) view.
#![allow(clippy::arithmetic_side_effects)] // A measurement harness's own sizes.

use lucene_analysis::Analyzer;
use lucene_index::indexing_chain::invert_documents;

fn status_kb(key: &str) -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with(key))
                .and_then(|l| l.split_whitespace().nth(1).and_then(|v| v.parse().ok()))
        })
        .unwrap_or(0)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let n_docs: usize = a.get(1).and_then(|v| v.parse().ok()).unwrap_or(20_000);
    let tokens: usize = a.get(2).and_then(|v| v.parse().ok()).unwrap_or(40);
    let vocab_size: usize = a.get(3).and_then(|v| v.parse().ok()).unwrap_or(20_000);

    // `index-bench`'s corpus generator, reproduced exactly so the two
    // measurements describe the same shape.
    let vocab: Vec<String> = (0..vocab_size).map(|i| format!("t{i}")).collect();
    let mut state: u32 = 0x9E37_79B9;
    let mut texts: Vec<String> = Vec::with_capacity(n_docs);
    for _ in 0..n_docs {
        let mut body = String::with_capacity(tokens * 7);
        for _ in 0..tokens {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            if !body.is_empty() {
                body.push(' ');
            }
            body.push_str(&vocab[state as usize % vocab.len()]);
        }
        texts.push(body);
    }
    let text_bytes: usize = texts.iter().map(|t| t.len()).sum();
    let docs: Vec<(i32, &str, &str)> = texts
        .iter()
        .enumerate()
        .map(|(i, t)| (i as i32, "body", t.as_str()))
        .collect();

    let base_rss = status_kb("VmRSS:");
    let t0 = std::time::Instant::now();
    let index = invert_documents(&docs, &Analyzer::standard(None));
    let elapsed = t0.elapsed();
    let after_rss = status_kb("VmRSS:");
    let peak_rss = status_kb("VmHWM:");
    let inverted = index.ram_bytes_used();
    let terms = index.terms.len();
    let postings: usize = index.terms.values().map(|l| l.entries.len()).sum();
    let occurrences: usize = index
        .terms
        .values()
        .flat_map(|l| l.entries.iter())
        .map(|e| e.occurrences.len())
        .sum();

    println!("docs={n_docs} tokens/doc={tokens} vocab={vocab_size}");
    println!(
        "  text                       {:>9.2} MB",
        text_bytes as f64 / 1048576.0
    );
    println!(
        "  InMemoryInvertedIndex      {:>9.2} MB   ({:.2}x text)",
        inverted as f64 / 1048576.0,
        inverted as f64 / text_bytes as f64
    );
    println!(
        "  RSS delta over the invert  {:>9.2} MB   peak {:.2} MB",
        (after_rss.saturating_sub(base_rss)) as f64 / 1024.0,
        peak_rss as f64 / 1024.0
    );
    // Where the bytes are, so "port the block pools" has a target rather than
    // a slogan. These four columns are the whole of `ram_bytes_used` except
    // for the fixed `size_of::<InMemoryInvertedIndex>()`, so they are printed
    // with the remainder rather than left to look complete on their own.
    // `payload runs` is always zero here: this example calls
    // `invert_documents`, the no-payloads entry point, because item 15 is
    // about what every field pays and not about what a `store_payloads` field
    // adds (`index-bench`'s `payloads` arm measures that).
    let mut key_bytes = 0usize;
    let mut entry_slot_bytes = 0usize;
    let mut occurrence_bytes = 0usize;
    let mut payload_bytes = 0usize;
    for ((field, term), list) in &index.terms {
        key_bytes += std::mem::size_of::<(
            lucene_index::indexing_chain::TermKey,
            lucene_index::indexing_chain::TermPostingList,
        )>() + field.capacity()
            + term.capacity();
        entry_slot_bytes += list.entries.capacity()
            * std::mem::size_of::<lucene_index::indexing_chain::PostingEntry>();
        payload_bytes += list.payload_bytes.capacity() + list.payload_lengths.capacity() * 4;
        for entry in &list.entries {
            occurrence_bytes += entry.occurrences.capacity()
                * std::mem::size_of::<lucene_index::indexing_chain::Occurrence>();
        }
    }
    let mb = |b: usize| b as f64 / 1048576.0;
    let accounted = key_bytes + entry_slot_bytes + occurrence_bytes + payload_bytes;
    println!(
        "    of which: keys {:.2} MB, posting-entry slots {:.2} MB, \
occurrence vectors {:.2} MB, payload runs {:.2} MB, unaccounted {:.2} MB",
        mb(key_bytes),
        mb(entry_slot_bytes),
        mb(occurrence_bytes),
        mb(payload_bytes),
        mb(inverted.saturating_sub(accounted))
    );
    println!("  keys={terms} posting-entries={postings} occurrences={occurrences}");
    println!("  invert took {:.3} s", elapsed.as_secs_f64());
}
