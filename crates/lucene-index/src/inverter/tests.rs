#![cfg(test)]
// Test code opts out of the arithmetic gate at the module boundary; see
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use super::*;
use crate::indexing_chain::invert_documents;

/// `FieldInvertState` as `computeNorm` reads it: tokens across every
/// value, distinct terms, the `DOCS` flag; `None` without a value.
#[test]
fn invert_state_counts_length_and_distinct_terms() {
    let analyzer = Analyzer::standard(None);
    let mut inv = FieldInverter::new(IndexOptions::Docs);
    inv.add_document(0, &["a a b", "b c"], &analyzer);
    inv.add_document(1, &[], &analyzer);
    inv.add_document(2, &["..."], &analyzer);
    let s = inv.invert_state(0).unwrap();
    assert!(s.docs_only);
    assert_eq!((s.length, s.num_overlap, s.unique_term_count), (5, 0, 3));
    assert_eq!(inv.invert_state(1), None);
    let empty = inv.invert_state(2).unwrap();
    assert_eq!((empty.length, empty.unique_term_count), (0, 0));
    assert_eq!(inv.invert_state(3), None);
}

/// The rest of `FieldInvertState`: the position and offset after every
/// value's gap, the largest frequency (for `DOCS`, 1 only while a term is
/// new to the segment) and `end()`'s attributes.
#[test]
fn invert_state_positions_offsets_and_max_frequency() {
    let analyzer = Analyzer::standard(None)
        .with_position_increment_gap(10)
        .with_offset_gap(3);
    let mut inv = FieldInverter::new(IndexOptions::DocsAndFreqsAndPositions);
    inv.add_document(0, &["a b a", "c "], &analyzer);
    let s = inv.invert_state(0).unwrap();
    assert_eq!((s.position, s.offset, s.max_term_frequency), (23, 13, 2));
    let end = s.attribute_source.unwrap();
    assert_eq!((end.end_offset(), end.position_increment()), (2, 0));
    let mut docs = FieldInverter::new(IndexOptions::Docs);
    docs.add_document(0, &["x x"], &analyzer);
    docs.add_document(1, &["x"], &analyzer);
    assert_eq!(docs.invert_state(0).unwrap().max_term_frequency, 1);
    assert_eq!(docs.invert_state(1).unwrap().max_term_frequency, 0);
}

/// The flush trigger's inputs: the term count and the heap estimate, which
/// grows with what is buffered; and positions clamp at `MAX_POSITION`
/// rather than overflow, as `IndexingChain`'s do.
#[test]
fn term_count_ram_estimate_and_position_clamp() {
    let analyzer = Analyzer::standard(None);
    let mut inv = FieldInverter::new(IndexOptions::DocsAndFreqsAndPositions);
    let empty = inv.ram_bytes_used();
    assert_eq!(inv.num_terms(), 0);
    inv.add_document(0, &["alpha beta gamma alpha"], &analyzer);
    assert_eq!(inv.num_terms(), 3);
    let one_doc = inv.ram_bytes_used();
    assert!(one_doc >= empty);
    for d in 1..200 {
        inv.add_document(d, &[&format!("term{d} alpha")], &analyzer);
    }
    assert_eq!(inv.num_terms(), 202);
    assert!(inv.ram_bytes_used() > one_doc);

    assert_eq!(advance_position(3, 4), 7);
    assert_eq!(advance_position(MAX_POSITION, 1), MAX_POSITION);
    assert_eq!(advance_position(i32::MAX, 1), MAX_POSITION);
}

/// Short terms that differ only after their first byte or two -- the
/// shape of a real vocabulary, and of the benchmark's `t0..t19999` --
/// must spread over the table. Probing from the hash's low bits put them
/// on a handful of slots (displacements in the thousands); this pins the
/// longest probe run to something a well-mixed table actually produces.
#[test]
fn short_similar_terms_spread_over_the_table() {
    let mut h = TermHash::new();
    for i in 0..20_000 {
        let (_, added) = h.add(format!("t{i}").as_bytes());
        assert!(added);
    }
    assert_eq!(h.add(b"t123"), (123, false));
    let size = h.table.len();
    let mask = size - 1;
    let longest = h
        .table
        .iter()
        .enumerate()
        .filter(|(_, &e)| e != 0)
        .map(|(slot, &e)| {
            let home = home_slot(h.hashes[(e as u32 - 1) as usize], size);
            slot.wrapping_sub(home) & mask
        })
        .max()
        .unwrap();
    assert!(
        longest < 64,
        "longest probe run {longest} in a {size}-slot table"
    );
}

/// The general path's answer for the same input, reshaped the way
/// `IndexWriter::build_postings_output` reshapes it.
fn reference(
    docs: &[(i32, &str, &str)],
    analyzer: &Analyzer,
    field: &str,
    index_options: IndexOptions,
) -> Vec<TermPostings> {
    let inverted = invert_documents(docs, analyzer);
    let has_positions = matches!(
        index_options,
        IndexOptions::DocsAndFreqsAndPositions | IndexOptions::DocsAndFreqsAndPositionsAndOffsets
    );
    let has_offsets = index_options == IndexOptions::DocsAndFreqsAndPositionsAndOffsets;
    let mut out = Vec::new();
    for ((f, term), list) in &inverted.terms {
        if f != field {
            continue;
        }
        out.push(TermPostings {
            term: term.as_bytes().to_vec(),
            docs: list
                .entries
                .iter()
                .map(|e| (e.doc_id, e.term_freq()))
                .collect(),
            positions: if has_positions {
                list.entries.iter().map(|e| e.positions()).collect()
            } else {
                Vec::new()
            },
            offsets: if has_offsets {
                list.entries.iter().map(|e| e.offsets()).collect()
            } else {
                Vec::new()
            },
            payload_bytes: Vec::new(),
            payload_lengths: Vec::new(),
        });
    }
    out
}

fn corpus() -> Vec<(i32, Vec<&'static str>)> {
    vec![
        (0, vec!["The quick brown fox", "jumps over the lazy dog"]),
        (1, vec![]),
        (2, vec!["fox fox FOX; café, naïve résumé 3.14 1,000 U.S.A."]),
        (3, vec!["", "  ,, "]),
        (4, vec!["the the the"]),
        (5, vec!["don't stop O'Brien e-mail x_y"]),
    ]
}

#[test]
fn matches_the_general_inverter_for_every_index_option_and_analyzer() {
    let stop = lucene_analysis::english_stop_words();
    let analyzers = [
        Analyzer::standard(None),
        Analyzer::standard(Some(&stop)),
        Analyzer::standard(Some(&stop))
            .with_position_increment_gap(7)
            .with_offset_gap(3),
        Analyzer::standard(None)
            .with_ascii_folding()
            .with_stemming(),
    ];
    for analyzer in &analyzers {
        for options in [
            IndexOptions::Docs,
            IndexOptions::DocsAndFreqs,
            IndexOptions::DocsAndFreqsAndPositions,
            IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
        ] {
            let mut inv = FieldInverter::new(options);
            let mut triples = Vec::new();
            for (doc, values) in corpus() {
                inv.add_document(doc, &values, analyzer);
                for v in values {
                    triples.push((doc, "body", v));
                }
            }
            let want = reference(&triples, analyzer, "body", options);
            let lengths = inv.lengths().to_vec();
            let doc_count = inv.doc_count();
            let got = inv.into_term_postings();
            assert_eq!(got.len(), want.len(), "{options:?}");
            for (g, w) in got.iter().zip(&want) {
                assert_eq!(g.term, w.term);
                assert_eq!(g.docs, w.docs, "{:?}", String::from_utf8_lossy(&g.term));
                assert_eq!(g.positions, w.positions);
                assert_eq!(g.offsets, w.offsets);
            }
            // Doc count: documents that produced at least one token.
            let mut with_tokens: Vec<i32> = want
                .iter()
                .flat_map(|t| t.docs.iter().map(|d| d.0))
                .collect();
            with_tokens.sort_unstable();
            with_tokens.dedup();
            assert_eq!(doc_count as usize, with_tokens.len());
            // Lengths: the sum of every term's freq per document.
            for (doc, values) in corpus() {
                let expected: u32 = want
                    .iter()
                    .flat_map(|t| t.docs.iter())
                    .filter(|d| d.0 == doc)
                    .map(|d| d.1 as u32)
                    .sum();
                let got_len = lengths[doc as usize];
                if values.is_empty() {
                    assert_eq!(got_len, None);
                } else {
                    assert_eq!(got_len, Some(expected), "doc {doc}");
                }
            }
        }
    }
}

/// A corpus big enough for `invert_parallel` to split: repeated and
/// unique terms, empty and missing values, multi-valued documents.
fn big_corpus() -> Vec<Vec<String>> {
    let mut state = 0x9E37_79B9u32;
    (0..3001)
        .map(|doc| {
            let mut values = Vec::new();
            if doc % 97 == 0 {
                return values;
            }
            for _ in 0..1 + doc % 3 {
                let mut text = String::new();
                for _ in 0..doc % 23 {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    text.push_str(&format!("w{} ", state % 700));
                }
                if doc % 11 == 0 {
                    text.push_str(&format!("only{doc}"));
                }
                values.push(text);
            }
            values
        })
        .collect()
}

fn assert_same_inversion(got: InvertedField, want: InvertedField, docs: usize) {
    assert_eq!(got.doc_count(), want.doc_count());
    assert_eq!(got.lengths(), want.lengths());
    for doc in 0..docs {
        assert_eq!(got.invert_state(doc), want.invert_state(doc), "doc {doc}");
    }
    let (got, want) = (got.into_term_postings(), want.into_term_postings());
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(&want) {
        assert_eq!(
            (&g.term, &g.docs, &g.positions, &g.offsets),
            (&w.term, &w.docs, &w.positions, &w.offsets)
        );
    }
}

/// `invert_parallel` (on a four-thread pool, so it really splits) and
/// `merge_runs` over hand-made runs both build exactly the serial
/// inversion, `DOCS`' new-term flag included.
#[test]
fn parallel_and_appended_runs_match_the_serial_inversion() {
    let corpus = big_corpus();
    let analyzer = Analyzer::standard(None);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    for options in [
        IndexOptions::Docs,
        IndexOptions::DocsAndFreqs,
        IndexOptions::DocsAndFreqsAndPositions,
        IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
    ] {
        let serial = || {
            let mut inv = FieldInverter::new(options);
            for (doc, values) in corpus.iter().enumerate() {
                let values: Vec<&str> = values.iter().map(String::as_str).collect();
                inv.add_document(doc as i32, &values, &analyzer);
            }
            inv
        };
        let parallel = pool.install(|| {
            FieldInverter::invert_parallel(
                options,
                corpus.len(),
                |doc, out: &mut Vec<&str>| out.extend(corpus[doc].iter().map(String::as_str)),
                &analyzer,
            )
        });
        assert_same_inversion(parallel, serial().finish(), corpus.len());

        // Uneven runs, one of them empty, merged by hand.
        let mut runs = Vec::new();
        for run in [0..1, 1..1, 1..700, 700..2900, 2900..3001] {
            let mut part = FieldInverter::new(options);
            for (local, doc) in run.enumerate() {
                let values: Vec<&str> = corpus[doc].iter().map(String::as_str).collect();
                part.add_document(local as i32, &values, &analyzer);
            }
            runs.push(part);
        }
        let merged = pool.install(|| merge_runs(options, runs));
        assert_same_inversion(merged, serial().finish(), corpus.len());
    }
    // Too few documents to split: the serial path, same answer.
    let small = FieldInverter::invert_parallel(
        IndexOptions::DocsAndFreqs,
        10,
        |doc, out: &mut Vec<&str>| out.extend(corpus[doc].iter().map(String::as_str)),
        &analyzer,
    );
    assert_eq!(small.lengths().len(), 10);
}

#[test]
fn the_term_hash_survives_growth_and_keeps_ids_stable() {
    let mut h = TermHash::new();
    let mut ids = Vec::new();
    for i in 0..5000u32 {
        let t = format!("t{i}");
        let (id, new) = h.add(t.as_bytes());
        assert!(new);
        assert_eq!(id, i);
        ids.push(t);
    }
    for (i, t) in ids.iter().enumerate() {
        assert_eq!(h.add(t.as_bytes()), (i as u32, false));
        assert_eq!(h.term(i as u32), t.as_bytes());
    }
    let sorted = h.sorted_ids();
    let mut want = ids.clone();
    want.sort();
    let got: Vec<&[u8]> = sorted.iter().map(|&id| h.term(id)).collect();
    let want: Vec<&[u8]> = want.iter().map(|s| s.as_bytes()).collect();
    assert_eq!(got, want);
}

/// The prefix-key sort against a plain byte sort where the eight-byte
/// key alone cannot decide: the empty term, zero bytes (which the key's
/// padding also produces), high bytes, and terms sharing a prefix of
/// eight bytes or more.
#[test]
fn sorted_ids_order_by_bytes_past_the_prefix_key() {
    let terms: Vec<&[u8]> = vec![
        b"abcdefghij",
        b"abcdefgh",
        b"",
        b"abcdefgh\0",
        b"a",
        b"a\0",
        b"\0",
        b"\xff\xff",
        b"abcdefghi",
        b"abcdefgha",
        b"abcdefgh\xff",
        b"abcdefg",
        b"b",
    ];
    let mut h = TermHash::new();
    for t in &terms {
        h.add(t);
    }
    let got: Vec<&[u8]> = h.sorted_ids().iter().map(|&id| h.term(id)).collect();
    let mut want = terms.clone();
    want.sort();
    assert_eq!(got, want);
}
