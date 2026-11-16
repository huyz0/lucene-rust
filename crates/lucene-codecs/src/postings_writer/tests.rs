#![cfg(test)]
// A test's `i + 1` is not a length read off disk; see
// `docs/arithmetic-gate.md`'s "Test code" section.
#![allow(clippy::arithmetic_side_effects)]

use super::*;
use crate::blocktree::{self, FieldTerms};
use crate::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, FieldInfos, VectorEncoding,
    VectorSimilarityFunction,
};
use crate::postings::DocInput;

const SEG_ID: [u8; ID_LENGTH] = [9u8; ID_LENGTH];
const SUFFIX: &str = "";

/// `CompetitiveImpactAccumulator`'s core property: the result is the
/// Pareto frontier of the input pairs -- every input is dominated by some
/// entry (same-or-higher freq at same-or-lower norm), and no entry is
/// dominated by another.
fn assert_is_frontier(input: &[(i32, i64)], got: &[Impact]) {
    for &(freq, norm) in input {
        assert!(
            got.iter()
                .any(|i| i.freq >= freq && (i.norm as u64) <= (norm as u64)),
            "({freq}, {norm}) is not dominated by any of {got:?}"
        );
    }
    for (a, b) in got.iter().zip(got.iter().skip(1)) {
        assert!(
            b.freq > a.freq && (b.norm as u64) > (a.norm as u64),
            "impacts must strictly increase in both freq and unsigned norm: {a:?} then {b:?}"
        );
    }
}

#[test]
fn competitive_impacts_keep_only_the_frontier_for_byte_norms() {
    // (freq, norm): the middle pair is dominated by the first (higher
    // freq, lower norm), so it must not survive.
    let input = [(5i32, 3i64), (3, 7), (9, 40), (2, 2)];
    let mut acc = CompetitiveImpactAccumulator::default();
    for &(freq, norm) in &input {
        acc.add(freq, norm);
    }
    let got = acc.get_competitive_freq_norm_pairs();
    assert_eq!(
        got,
        vec![
            Impact { freq: 2, norm: 2 },
            Impact { freq: 5, norm: 3 },
            Impact { freq: 9, norm: 40 },
        ]
    );
    assert_is_frontier(&input, &got);
}

/// The `otherFreqNormPairs` branch: norms outside `-128..=127`, which is
/// what a `.nvd` storing more than one byte per norm produces and which
/// a merge of such a segment hands this writer. Java keeps them in a
/// `TreeSet` rather than the 256-entry table; this keeps them in a sorted
/// `Vec`, and the answer has to be the same frontier either way.
#[test]
fn competitive_impacts_handle_norms_wider_than_a_byte() {
    let input = [
        (4i32, 1_000i64),
        (2, 500),
        (7, 5_000),
        (3, 900), // dominated by (4, 1000)? no -- lower norm, lower freq
        (1, 100_000),
    ];
    let mut acc = CompetitiveImpactAccumulator::default();
    for &(freq, norm) in &input {
        acc.add(freq, norm);
    }
    let got = acc.get_competitive_freq_norm_pairs();
    assert_is_frontier(&input, &got);
    // (1, 100000) is dominated by (2, 500) and must be gone.
    assert!(!got.iter().any(|i| i.freq == 1));
    // And every entry came from the overflow list, not the byte table.
    assert!(got.iter().all(|i| i.norm > 127));
}

/// Byte-valued and wide norms in the same block: Java merges the table's
/// derived impacts into the tree, and the frontier must span both.
#[test]
fn competitive_impacts_merge_byte_and_wide_norms() {
    let input = [(10i32, 5i64), (3, 2), (12, 9_000), (2, 40_000)];
    let mut acc = CompetitiveImpactAccumulator::default();
    for &(freq, norm) in &input {
        acc.add(freq, norm);
    }
    let got = acc.get_competitive_freq_norm_pairs();
    assert_is_frontier(&input, &got);
    assert!(got.iter().any(|i| i.norm <= 127));
    assert!(got.iter().any(|i| i.norm > 127));
    // (2, 40000) is dominated by (3, 2).
    assert!(!got.iter().any(|i| i.norm == 40_000));
}

/// Java's norms are **sign-extended bytes**, so byte 200 arrives as
/// `-56`, and its ordering against byte 100 is `Long.compareUnsigned` --
/// under which `-56` is the larger (longer field, lower score). Getting
/// the comparison signed instead would order the frontier backwards and
/// `CheckIndex` would reject the segment.
#[test]
fn competitive_impacts_order_sign_extended_norms_unsigned() {
    let mut acc = CompetitiveImpactAccumulator::default();
    acc.add(3, i64::from(100i8)); // byte 100
    acc.add(9, i64::from(200u8 as i8)); // byte 200 -> -56
    let got = acc.get_competitive_freq_norm_pairs();
    assert_eq!(
        got,
        vec![
            Impact { freq: 3, norm: 100 },
            Impact {
                freq: 9,
                norm: i64::from(200u8 as i8)
            },
        ],
        "byte 200 must sort *after* byte 100"
    );
}

/// The unsigned comparisons *inside* `add_to_set`, which the test above
/// cannot reach: two byte-valued norms both land in the `max_freqs`
/// table, `other` stays empty, and `get_competitive_freq_norm_pairs`
/// returns at its fast path. Ordering there comes from walking `0..256`,
/// which is unsigned by construction.
///
/// Reaching `add_to_set` needs one norm wide enough for the overflow set
/// and one byte norm that sign-extends negative, so the merge compares
/// them against each other. That is the shape the overflow set was ported
/// for -- a `.nvd` storing more than one byte per norm -- and it is where
/// signed-vs-unsigned actually changes the answer:
///
/// - unsigned, `1000 >= 0xFFFF_FFFF_FFFF_FFC8` is false, the backward walk
///   stops, and both entries survive;
/// - signed, `1000 >= -56` is true, so `(3, 1000)` -- which dominates
///   nothing -- is dropped, leaving a frontier that no longer bounds the
///   freq-3 documents. An under-bound block is a *skipped hit* under
///   MAXSCORE, not merely a slower search.
#[test]
fn add_to_set_compares_norms_unsigned_across_the_overflow_set() {
    let mut acc = CompetitiveImpactAccumulator::default();
    acc.add(9, i64::from(200u8 as i8)); // byte 200 -> -56, byte table
    acc.add(3, 1_000); // too wide for the table -> `other`

    let got = acc.get_competitive_freq_norm_pairs();
    assert_eq!(
        got,
        vec![
            Impact {
                freq: 3,
                norm: 1_000
            },
            Impact {
                freq: 9,
                norm: i64::from(200u8 as i8)
            },
        ],
        "a negative (sign-extended) norm is the *largest* unsigned norm, so \
             it dominates nothing and the freq-3 entry must survive"
    );
}

/// With no norms every document is norm 1, so the frontier is the single
/// `(maxFreq, 1)` pair this writer emitted before item 18 -- which is what
/// keeps the normless path byte-identical.
#[test]
fn competitive_impacts_without_norms_are_a_single_max_freq_pair() {
    let docs = [(0i32, 3i32), (1, 7), (2, 5)];
    assert_eq!(
        competitive_impacts(&docs, None),
        vec![Impact { freq: 7, norm: 1 }]
    );
}

/// A doc id past the end of the norms column reads as norm 1 -- Java's
/// `norms.advanceExact(docID) == false` branch.
#[test]
fn a_document_past_the_norms_column_reads_as_norm_one() {
    let norms = [5i64, 6];
    assert_eq!(norm_of(Some(&norms), 0), 5);
    assert_eq!(norm_of(Some(&norms), 1), 6);
    assert_eq!(norm_of(Some(&norms), 2), 1);
    assert_eq!(norm_of(Some(&norms), -1), 1);
    assert_eq!(norm_of(None, 0), 1);
}

/// [`write_impacts`] round-trips through the *read* side's
/// [`crate::postings::decode_impacts`], including the explicit
/// zig-zag `normDelta` branch (norm jumping by more than one) and the
/// folded single-byte branch (norm increasing by exactly one).
#[test]
fn write_impacts_round_trips_through_decode_impacts() {
    for impacts in [
        vec![Impact { freq: 1, norm: 1 }],
        // Consecutive norms: every entry takes the folded one-byte form.
        vec![
            Impact { freq: 1, norm: 1 },
            Impact { freq: 2, norm: 2 },
            Impact { freq: 3, norm: 3 },
        ],
        // Gaps, and a sign-extended norm past 127.
        vec![
            Impact { freq: 2, norm: 4 },
            Impact { freq: 40, norm: 90 },
            Impact {
                freq: 900,
                norm: i64::from(250u8 as i8),
            },
        ],
    ] {
        let mut bytes = Vec::new();
        write_impacts(&impacts, &mut bytes);
        assert_eq!(
            crate::postings::decode_impacts(&bytes).expect("decode"),
            impacts
        );
    }
}

/// Builds a [`TermPostings`] flat payload run from one occurrence's
/// payload per element, in doc order and then occurrence order -- which
/// is the whole layout: doc boundaries live in `docs`' frequencies, not
/// in the run.
fn payload_run(occurrences: &[&[u8]]) -> (Vec<u8>, Vec<u32>) {
    let mut bytes = Vec::new();
    let mut lengths = Vec::new();
    for occurrence in occurrences {
        lengths.push(occurrence.len() as u32);
        bytes.extend_from_slice(occurrence);
    }
    (bytes, lengths)
}

fn field_info(number: i32, name: &str, index_options: IndexOptions) -> FieldInfo {
    FieldInfo {
        name: name.to_string(),
        number,
        store_term_vectors: false,
        omit_norms: false,
        store_payloads: false,
        soft_deletes_field: false,
        parent_field: false,
        index_options,
        doc_values_type: DocValuesType::None,
        doc_values_skip_index_type: DocValuesSkipIndexType::None,
        doc_values_gen: -1,
        attributes: Vec::new(),
        point_dimension_count: 0,
        point_index_dimension_count: 0,
        point_num_bytes: 0,
        vector_dimension: 0,
        vector_encoding: VectorEncoding::Float32,
        vector_similarity_function: VectorSimilarityFunction::Euclidean,
    }
}

/// Same as [`field_info`] but with `store_payloads` set — needed for
/// every payload round-trip test below, since [`FieldTerms::positions`]/
/// [`crate::blocktree::FieldTerms`] reads `has_payloads` off the opened
/// `FieldInfo`, not off the writer's [`FieldPostingsInput::has_payloads`]
/// (which only controls what bytes this writer emits).
fn field_info_with_payloads(number: i32, name: &str, index_options: IndexOptions) -> FieldInfo {
    FieldInfo {
        store_payloads: true,
        ..field_info(number, name, index_options)
    }
}

fn open_written<'a>(
    output: &'a Output,
    field_infos: &FieldInfos,
    max_doc: i32,
) -> (blocktree::BlockTreeFields, DocInput<'a>) {
    let fields = blocktree::open(
        &output.tim,
        &output.tip,
        &output.tmd,
        field_infos,
        &SEG_ID,
        SUFFIX,
        max_doc,
    )
    .expect("write_single_field's own bytes must open cleanly");
    let doc_in = DocInput::open(&output.doc, &SEG_ID, SUFFIX).expect("open .doc");
    (fields, doc_in)
}

/// The regexp term walk against a brute-force filter over a 70 000-term
/// dictionary with multi-byte UTF-8 and ill-formed bytes: entry filtering,
/// the end-of-block jump, `ALL` semantics, the lazy automaton for patterns
/// too large to determinize, and that each yielded term's state opens its
/// own postings. This crate's terms writer lays a field out as one leaf
/// block, so floor skips, sub-block descent and trie-node tracking are
/// **not** exercised here -- only by `tests/regexp_intersect_fixtures.rs`,
/// whose dictionary Lucene wrote.
#[test]
fn lockstep_regexp_walk_agrees_with_a_brute_force_scan() {
    fn base36(mut n: u32) -> String {
        const DIGITS: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";
        let mut out = Vec::new();
        loop {
            out.push(DIGITS[(n % 36) as usize]);
            n /= 36;
            if n == 0 {
                break;
            }
        }
        out.reverse();
        String::from_utf8(out).unwrap()
    }
    let mut raw: Vec<Vec<u8>> = Vec::new();
    for i in 0..60_000u32 {
        raw.push(format!("t{}", base36(i)).into_bytes());
    }
    for i in 0..3_000u32 {
        raw.push(format!("{}x{}", base36(i % 97), i % 13).into_bytes());
        raw.push(format!("é{}€", i % 500).into_bytes());
        raw.push(format!("aaaaaaaaaaaaaaaa{i}").into_bytes());
    }
    for i in 0..=255u8 {
        raw.push(vec![b't', 0xFF, i]);
        raw.push(vec![i]);
    }
    raw.sort_unstable();
    raw.dedup();
    const MAX_DOC: i32 = 1_000;
    let terms: Vec<TermPostings> = raw
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let first = (i % 997) as i32;
            let mut docs = vec![(first, 1 + (i % 3) as i32)];
            if i % 5 == 0 {
                docs.push((first + 1, 1));
            }
            TermPostings {
                term: t.clone(),
                docs,
                ..Default::default()
            }
        })
        .collect();
    let input = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqs,
        doc_count: MAX_DOC,
        has_payloads: false,
        terms: &terms,
    };
    let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
    let fis = FieldInfos {
        fields: vec![field_info(0, "body", IndexOptions::DocsAndFreqs)],
    };
    let (fields, doc_in) = open_written(&output, &fis, MAX_DOC);
    let field = fields.field("body").unwrap();

    for src in [
        "t1[0-9]",
        ".*z",
        "t[0-9a-f]+",
        "(t1|t2)[a-z]",
        "t.*9",
        "[a-z][0-9]{2}",
        ".*1.*2.*",
        "t",
        "",
        ".*",
        "t.",
        "tz.*",
        "[^t].*",
        "é[0-9]+€",
        "é.*",
        "a{16}[0-9]+",
        "a+1.*",
        ".x1[0-2]?",
        "t<100-2000>",
        "t[a-z]+&.*q.*",
        "#",
        "(t0|t1|tz|t10)",
        "t1a.",
        "t[0-9]{2}[a-z]",
        ".*[qz][0-9]",
        ".",
        ".{2,3}",
        "[0-9a-z]x1",
        "t9zz.*",
        "zzzz",
        "t.*(0|9)",
        "@",
        // Too large to determinize whole (`MAX_DFA_STATES`): the walk
        // runs the lazily determinized automaton instead.
        ".*a.{13}",
        ".*[a-c].{15}",
        ".*[a1].{14}.*",
    ] {
        let pattern = crate::regexp::RegexpPattern::new(src.as_bytes()).unwrap();
        if src.contains(".{1") {
            assert!(
                pattern.to_dfa().is_none(),
                "{src} should need the lazy automaton"
            );
            let expected: Vec<&Vec<u8>> = raw.iter().filter(|t| pattern.matches(t)).collect();
            let got: Vec<Vec<u8>> = field
                .regexp_intersect_states(&pattern)
                .map(|r| r.unwrap().0)
                .collect();
            assert!(!expected.is_empty(), "{src} matches nothing here");
            assert_eq!(got.iter().collect::<Vec<_>>(), expected, "pattern {src}");
            continue;
        }
        // Lucene's `AUTOMATON_TYPE.ALL`: a pattern accepting every
        // code-point string enumerates every term, ill-formed ones too.
        let dfa = pattern.to_dfa().unwrap();
        let total =
            dfa.start() != crate::automaton::DEAD && dfa.utf8_total_states()[dfa.start() as usize];
        let expected: Vec<&TermPostings> = terms
            .iter()
            .filter(|t| total || pattern.matches(&t.term))
            .collect();
        let got: Vec<(Vec<u8>, blocktree::SeekedTerm)> = field
            .regexp_intersect_states(&pattern)
            .map(|r| r.unwrap())
            .collect();
        assert_eq!(got.len(), expected.len(), "pattern {src}");
        for ((t, seeked), want) in got.iter().zip(&expected) {
            assert_eq!(t, &want.term, "pattern {src}");
            assert_eq!(
                seeked.stats.doc_freq as usize,
                want.docs.len(),
                "{src} {t:?}"
            );
            let mut c = field
                .lazy_postings_for(seeked, &doc_in, crate::postings::PostingsFlags::Freqs)
                .unwrap();
            assert_eq!(c.next_doc().unwrap(), want.docs[0].0, "{src} {t:?}");
            assert_eq!(c.freq(), Some(want.docs[0].1), "{src} {t:?}");
        }
    }
}

/// Mixed singleton/multi-doc terms, round-tripped through the existing
/// unmodified `blocktree::open` + `postings::DocInput` read side (no
/// query layer here — see
/// `crates/lucene-search/tests/postings_writer_round_trip.rs` for the
/// required end-to-end `search_term_query` proof, which lives in
/// `lucene-search` rather than here since this crate must not depend
/// upward on `lucene-search`, see the `architecture` skill).
#[test]
fn mixed_singleton_and_multi_doc_terms_round_trip() {
    let terms = vec![
        TermPostings {
            term: b"fox".to_vec(),
            docs: vec![(1, 2), (4, 1), (7, 3)],
            ..Default::default()
        },
        TermPostings {
            term: b"quick".to_vec(),
            docs: vec![(4, 1)], // singleton
            ..Default::default()
        },
        TermPostings {
            term: b"the".to_vec(),
            docs: vec![(0, 1), (1, 1), (4, 2), (7, 1)],
            ..Default::default()
        },
    ];
    let input = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqs,
        doc_count: 8,
        has_payloads: false,
        terms: &terms,
    };
    let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();

    let fis = FieldInfos {
        fields: vec![field_info(0, "body", IndexOptions::DocsAndFreqs)],
    };
    let (fields, doc_in) = open_written(&output, &fis, 8);

    let field = fields.field("body").unwrap();
    assert_eq!(field.num_terms, 3);
    assert_eq!(field.min_term, b"fox");
    assert_eq!(field.max_term, b"the");

    let postings = field.postings(b"fox", Some(&doc_in)).unwrap().unwrap();
    assert_eq!(postings.docs, vec![1, 4, 7]);
    assert_eq!(postings.freqs, vec![2, 1, 3]);

    let postings = field.postings(b"quick", Some(&doc_in)).unwrap().unwrap();
    assert_eq!(postings.docs, vec![4]);
    assert_eq!(postings.freqs, vec![1]);

    let postings = field.postings(b"the", Some(&doc_in)).unwrap().unwrap();
    assert_eq!(postings.docs, vec![0, 1, 4, 7]);
    assert_eq!(postings.freqs, vec![1, 1, 2, 1]);

    assert!(field.seek_exact(b"missing").is_none());
}

/// A pulsed single-document term opens through `lazy_postings_for` as a
/// one-document cursor: its document and `totalTermFreq` come from the
/// term metadata, and it walks and advances like any other cursor. On a
/// field indexed without frequencies its frequency is 1, as every other
/// block shape reports there -- never 0, which would score it 0.
#[test]
fn singleton_term_opens_as_a_one_document_cursor() {
    for options in [IndexOptions::Docs, IndexOptions::DocsAndFreqs] {
        singleton_cursor_case(options);
    }
}

fn singleton_cursor_case(options: IndexOptions) {
    let with_freqs = options != IndexOptions::Docs;
    let terms = vec![
        TermPostings {
            term: b"many".to_vec(),
            docs: vec![(0, 1), (3, 2), (5, 1)],
            ..Default::default()
        },
        TermPostings {
            term: b"once".to_vec(),
            docs: vec![(4, 3)],
            ..Default::default()
        },
    ];
    let input = FieldPostingsInput {
        field_number: 0,
        index_options: options,
        doc_count: 4,
        has_payloads: false,
        terms: &terms,
    };
    let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
    let fis = FieldInfos {
        fields: vec![field_info(0, "body", options)],
    };
    let (fields, doc_in) = open_written(&output, &fis, 6);
    let field = fields.field("body").unwrap();
    let once = field.seek_term_state(b"once").unwrap().unwrap();
    assert_eq!(once.stats.doc_freq, 1);
    let freqs = crate::postings::PostingsFlags::Freqs;
    let mut c = field.lazy_postings_for(&once, &doc_in, freqs).unwrap();
    assert_eq!(c.next_doc().unwrap(), 4);
    assert_eq!(
        c.freq(),
        Some(if with_freqs { 3 } else { 1 }),
        "{options:?}"
    );
    assert_eq!(c.next_doc().unwrap(), crate::postings::NO_MORE_DOCS);
    let mut c = field.lazy_postings_for(&once, &doc_in, freqs).unwrap();
    assert_eq!(c.advance(2).unwrap(), 4);
    let mut c = field.lazy_postings_for(&once, &doc_in, freqs).unwrap();
    assert_eq!(c.advance(5).unwrap(), crate::postings::NO_MORE_DOCS);
    let docs_only = crate::postings::PostingsFlags::DocsOnly;
    let mut c = field.lazy_postings_for(&once, &doc_in, docs_only).unwrap();
    assert_eq!(c.next_doc().unwrap(), 4);
    assert_eq!(c.freq(), Some(1));
    // Only a pulsed term has a singleton document to open.
    let many = field.seek_term_state(b"many").unwrap().unwrap();
    let meta = field.term_metadata(b"many").unwrap().unwrap();
    assert!(doc_in
        .singleton_cursor(meta, many.stats.total_term_freq, options, false, freqs)
        .is_err());
}

/// `LazyDocsCursor::advance` into a bit-set-encoded block searches the
/// kept bits (`bits_advance`) rather than expanding the block.
/// Whatever it skips, the document it lands on -- and, since `freqs[i]`
/// must still pair with the block's `i`-th document, that document's
/// frequency -- must be exactly what a plain scan finds, for targets in
/// every word of the bit set, before the block, and past the end.
///
/// Docs-only cursors too, whose bit-set advance keeps no rank at all, with
/// a `next_doc` every third move so the rank is recovered from the
/// document (`materialize_bits`); and a term in every document, whose
/// blocks are all the consecutive (`bitsPerValue == 0`) encoding, which
/// the cursor keeps as a full bit set the way Lucene does.
#[test]
fn lazy_advance_into_bit_set_blocks_matches_a_linear_scan() {
    // Densities between 1/2 and 1/5 make `Lucene104PostingsWriter` pick
    // the bit-set encoding; the irregular one mixes it with packed blocks.
    // Past `LEVEL1_NUM_DOCS` (8192) for the dense terms, so the walks
    // cross level-1 spans as well as blocks.
    const NUM_DOCS: i32 = 20_000;
    let mut terms = Vec::new();
    let mut x = 0x2545_F491_4F6C_DD1Du64;
    for (name, keep) in [
        // Sorted, as the writer requires.
        (&b"all"[..], 1u64),
        (&b"fifth"[..], 5),
        (&b"half"[..], 2),
        (&b"mixed"[..], 0),
        (&b"third"[..], 3),
    ] {
        let mut docs = Vec::new();
        for d in 0..NUM_DOCS {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let hit = if keep == 0 {
                // Runs of dense and sparse stretches.
                x.is_multiple_of(if (d / 700) % 2 == 0 { 2 } else { 40 })
            } else {
                (d as u64).is_multiple_of(keep)
            };
            if hit {
                docs.push((d, 1 + (x % 5) as i32));
            }
        }
        terms.push(TermPostings {
            term: name.to_vec(),
            docs,
            ..Default::default()
        });
    }
    use crate::postings::PostingsFlags;
    // A field without freqs too: its headers carry no impacts region, so
    // the docs-only header walk takes its no-seek branch.
    for options in [IndexOptions::Docs, IndexOptions::DocsAndFreqs] {
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: options,
            doc_count: NUM_DOCS,
            has_payloads: false,
            terms: &terms,
        };
        let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
        let fis = FieldInfos {
            fields: vec![field_info(0, "body", options)],
        };
        let (fields, doc_in) = open_written(&output, &fis, NUM_DOCS);
        let field = fields.field("body").unwrap();
        for t in &terms {
            // The first entry at or after `pos` in doc order.
            let at = |pos: usize| t.docs.get(pos).copied();
            for flags in [PostingsFlags::Freqs, PostingsFlags::DocsOnly] {
                for gap in [1i32, 7, 63, 64, 65, 200, 1000, 9000] {
                    let mut cursor = field
                        .lazy_postings_with_flags(&t.term, &doc_in, flags)
                        .unwrap()
                        .unwrap();
                    let mut target = 0i32;
                    for step in 0u32.. {
                        let (got, want) = if step % 3 == 2 {
                            let prev = cursor.doc_id();
                            let want = at(t.docs.partition_point(|&(d, _)| d <= prev));
                            (cursor.next_doc().unwrap(), want)
                        } else {
                            let want = at(t.docs.partition_point(|&(d, _)| d < target));
                            (cursor.advance(target).unwrap(), want)
                        };
                        let ctx = format!(
                            "term {:?} {options:?} {flags:?} gap {gap} step {step}",
                            t.term
                        );
                        let Some((d, f)) = want else {
                            assert_eq!(got, crate::postings::NO_MORE_DOCS, "{ctx}");
                            break;
                        };
                        assert_eq!(got, d, "{ctx}");
                        // Without freqs on either side, every document
                        // counts once -- including a docs-only cursor on a
                        // bit-set block, whose position in the block is
                        // not tracked at all.
                        let want_freq = if flags == PostingsFlags::Freqs
                            && options == IndexOptions::DocsAndFreqs
                        {
                            f
                        } else {
                            1
                        };
                        assert_eq!(cursor.freq(), Some(want_freq), "{ctx}");
                        target = got + gap;
                    }
                }
            }
        }
    }
}

/// Byte-level correctness on `docFreq`/`totalTermFreq`/`seek_exact`
/// alone (no query layer), for `IndexOptions::Docs` (no freqs at all —
/// `totalTermFreq == docFreq` aliasing) to make sure that branch, not
/// just `DocsAndFreqs`, round-trips.
#[test]
fn docs_only_index_options_round_trips() {
    let terms = vec![
        TermPostings {
            term: b"a".to_vec(),
            docs: vec![(0, 1), (2, 1)],
            ..Default::default()
        },
        TermPostings {
            term: b"b".to_vec(),
            docs: vec![(1, 1)],
            ..Default::default()
        },
    ];
    let input = FieldPostingsInput {
        field_number: 3,
        index_options: IndexOptions::Docs,
        doc_count: 3,
        has_payloads: false,
        terms: &terms,
    };
    let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
    let fis = FieldInfos {
        fields: vec![field_info(3, "f", IndexOptions::Docs)],
    };
    let (fields, doc_in) = open_written(&output, &fis, 3);
    let field: &FieldTerms = fields.field("f").unwrap();
    assert_eq!(
        field.seek_exact(b"a"),
        Some(blocktree::TermStats {
            doc_freq: 2,
            total_term_freq: 2
        })
    );
    let postings = field.postings(b"a", Some(&doc_in)).unwrap().unwrap();
    assert_eq!(postings.docs, vec![0, 2]);
    assert_eq!(postings.freqs, vec![1, 1]); // freqs default to 1 when the field has no freqs

    let postings = field.postings(b"b", Some(&doc_in)).unwrap().unwrap();
    assert_eq!(postings.docs, vec![1]);
}

/// All terms singleton (`docFreq == 1`): no `.doc` bytes are needed at
/// all — `postings()` must still resolve every term purely from the
/// term-dictionary metadata (`postings::singleton_postings`).
#[test]
fn all_singleton_terms_need_no_doc_file() {
    let terms = vec![
        TermPostings {
            term: b"alpha".to_vec(),
            docs: vec![(2, 1)],
            ..Default::default()
        },
        TermPostings {
            term: b"beta".to_vec(),
            docs: vec![(5, 4)],
            ..Default::default()
        },
    ];
    let input = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqs,
        doc_count: 2,
        has_payloads: false,
        terms: &terms,
    };
    let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
    let fis = FieldInfos {
        fields: vec![field_info(0, "f", IndexOptions::DocsAndFreqs)],
    };
    let fields = blocktree::open(
        &output.tim,
        &output.tip,
        &output.tmd,
        &fis,
        &SEG_ID,
        SUFFIX,
        6,
    )
    .unwrap();
    let field = fields.field("f").unwrap();
    // No `.doc` file opened at all -- `doc_in: None` is fine since every
    // term here is a singleton.
    let postings = field.postings(b"beta", None).unwrap().unwrap();
    assert_eq!(postings.docs, vec![5]);
    assert_eq!(postings.freqs, vec![4]);
}

#[test]
fn rejects_empty_terms() {
    let input: FieldPostingsInput<'_> = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqs,
        doc_count: 0,
        has_payloads: false,
        terms: &[],
    };
    assert!(matches!(
        write_single_field(&input, &SEG_ID, SUFFIX),
        Err(Error::EmptyTerms)
    ));
}

#[test]
fn rejects_unsorted_terms() {
    let terms = vec![
        TermPostings {
            term: b"b".to_vec(),
            docs: vec![(0, 1)],
            ..Default::default()
        },
        TermPostings {
            term: b"a".to_vec(),
            docs: vec![(0, 1)],
            ..Default::default()
        },
    ];
    let input = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqs,
        doc_count: 1,
        has_payloads: false,
        terms: &terms,
    };
    assert!(matches!(
        write_single_field(&input, &SEG_ID, SUFFIX),
        Err(Error::TermsNotSorted(1))
    ));
}

#[test]
fn rejects_duplicate_terms() {
    let terms = vec![
        TermPostings {
            term: b"a".to_vec(),
            docs: vec![(0, 1)],
            ..Default::default()
        },
        TermPostings {
            term: b"a".to_vec(),
            docs: vec![(1, 1)],
            ..Default::default()
        },
    ];
    let input = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqs,
        doc_count: 2,
        has_payloads: false,
        terms: &terms,
    };
    assert!(matches!(
        write_single_field(&input, &SEG_ID, SUFFIX),
        Err(Error::TermsNotSorted(1))
    ));
}

#[test]
fn rejects_empty_postings_for_a_term() {
    let terms = vec![TermPostings {
        term: b"a".to_vec(),
        docs: vec![],
        ..Default::default()
    }];
    let input = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqs,
        doc_count: 0,
        has_payloads: false,
        terms: &terms,
    };
    assert!(matches!(
        write_single_field(&input, &SEG_ID, SUFFIX),
        Err(Error::EmptyPostings(0))
    ));
}

#[test]
fn rejects_non_ascending_doc_ids() {
    let terms = vec![TermPostings {
        term: b"a".to_vec(),
        docs: vec![(2, 1), (1, 1)],
        ..Default::default()
    }];
    let input = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqs,
        doc_count: 3,
        has_payloads: false,
        terms: &terms,
    };
    assert!(matches!(
        write_single_field(&input, &SEG_ID, SUFFIX),
        Err(Error::DocIdsNotSorted { index: 0 })
    ));
}

#[test]
fn rejects_duplicate_doc_ids() {
    let terms = vec![TermPostings {
        term: b"a".to_vec(),
        docs: vec![(1, 1), (1, 1)],
        ..Default::default()
    }];
    let input = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqs,
        doc_count: 2,
        has_payloads: false,
        terms: &terms,
    };
    assert!(matches!(
        write_single_field(&input, &SEG_ID, SUFFIX),
        Err(Error::DocIdsNotSorted { index: 0 })
    ));
}

#[test]
fn rejects_non_positive_freq() {
    let terms = vec![TermPostings {
        term: b"a".to_vec(),
        docs: vec![(0, 0), (1, 1)],
        ..Default::default()
    }];
    let input = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqs,
        doc_count: 2,
        has_payloads: false,
        terms: &terms,
    };
    assert!(matches!(
        write_single_field(&input, &SEG_ID, SUFFIX),
        Err(Error::NonPositiveFreq { index: 0 })
    ));
}

/// Round-trips a `docFreq` at each level-1-relevant boundary through the
/// existing, unmodified `blocktree::open`/`DocInput::read_postings` --
/// asserting the full doc/freq lists, not just "didn't error". Covers:
/// exactly `LEVEL1_NUM_DOCS` (one level-1 span, no remainder), one more
/// than that (one span + a one-doc tail), and two full level-1 spans
/// back to back, proving `write_level1_span`'s `level1_last_doc_id`/
/// `prev_doc_id` threading across more than one span.
#[test]
fn docfreq_at_level1_boundaries_round_trips() {
    for doc_freq in [LEVEL1_NUM_DOCS, LEVEL1_NUM_DOCS + 1, 2 * LEVEL1_NUM_DOCS] {
        let term = varied_docs_term(b"a", doc_freq);
        let max_doc = term.docs.last().unwrap().0 + 1;
        let terms = vec![term.clone()];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count: term.docs.len() as i32,
            has_payloads: false,
            terms: &terms,
        };
        let output = write_single_field(&input, &SEG_ID, SUFFIX)
            .unwrap_or_else(|e| panic!("doc_freq={doc_freq}: {e}"));
        let fis = FieldInfos {
            fields: vec![field_info(0, "f", IndexOptions::DocsAndFreqs)],
        };
        let (fields, doc_in) = open_written(&output, &fis, max_doc);
        let field = fields.field("f").unwrap();
        assert_eq!(
            field.seek_exact(b"a").unwrap().doc_freq,
            doc_freq,
            "doc_freq={doc_freq}"
        );
        let postings = field.postings(b"a", Some(&doc_in)).unwrap().unwrap();
        let expected_docs: Vec<i32> = term.docs.iter().map(|&(d, _)| d).collect();
        let expected_freqs: Vec<i32> = term.docs.iter().map(|&(_, f)| f).collect();
        assert_eq!(postings.docs, expected_docs, "doc_freq={doc_freq}");
        assert_eq!(postings.freqs, expected_freqs, "doc_freq={doc_freq}");
    }
}

/// Same boundaries as [`docfreq_at_level1_boundaries_round_trips`] but
/// through [`crate::postings::DocInput::lazy_cursor`]'s `advance`, which
/// is what actually exercises `LazyDocsCursor::skip_level1_to` --
/// jumping straight past whole level-1 spans without decoding their
/// level-0 blocks. Advancing to the very last doc after a full span (or
/// two) proves the skip landed in the right place.
#[test]
fn docfreq_at_level1_boundaries_advance_via_lazy_cursor() {
    for doc_freq in [LEVEL1_NUM_DOCS, LEVEL1_NUM_DOCS + 1, 2 * LEVEL1_NUM_DOCS] {
        let term = varied_docs_term(b"a", doc_freq);
        let max_doc = term.docs.last().unwrap().0 + 1;
        let terms = vec![term.clone()];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count: term.docs.len() as i32,
            has_payloads: false,
            terms: &terms,
        };
        let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
        let fis = FieldInfos {
            fields: vec![field_info(0, "f", IndexOptions::DocsAndFreqs)],
        };
        let (fields, doc_in) = open_written(&output, &fis, max_doc);
        let field = fields.field("f").unwrap();
        let mut cursor = field
            .lazy_postings(b"a", &doc_in)
            .unwrap()
            .expect("term must exist");
        let last_doc = term.docs.last().unwrap().0;
        assert_eq!(
            cursor.advance(last_doc).unwrap(),
            last_doc,
            "doc_freq={doc_freq}"
        );
    }
}

/// `PostingsEnum` flags. With
/// [`crate::postings::PostingsFlags::DocsOnly`] every frequency block on
/// the wire is stepped over (`PForUtil.skip`) instead of unpacked, and
/// every reported frequency is `1` -- so the *doc ids* must be
/// bit-for-bit what the freqs-decoding path produces, across all three
/// block shapes at once (full level-0 blocks, a group-varint tail, and a
/// level-1 span).
#[test]
fn docs_only_flags_decode_the_same_doc_ids_and_report_freq_one() {
    use crate::postings::PostingsFlags;
    for doc_freq in [
        300,                  // one full block + an 44-doc tail
        2 * BLOCK_SIZE,       // full blocks, no tail
        LEVEL1_NUM_DOCS + 77, // a whole level-1 span plus a tail
    ] {
        let term = irregular_docs_term(b"a", doc_freq);
        let max_doc = term.docs.last().unwrap().0 + 1;
        let terms = vec![term.clone()];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count: term.docs.len() as i32,
            has_payloads: false,
            terms: &terms,
        };
        let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
        let fis = FieldInfos {
            fields: vec![field_info(0, "f", IndexOptions::DocsAndFreqs)],
        };
        let (fields, doc_in) = open_written(&output, &fis, max_doc);
        let field = fields.field("f").unwrap();

        let with_freqs = field.postings(b"a", Some(&doc_in)).unwrap().unwrap();
        let docs_only = field
            .postings_with_flags(b"a", Some(&doc_in), PostingsFlags::DocsOnly)
            .unwrap()
            .unwrap();
        assert_eq!(docs_only.docs, with_freqs.docs, "doc_freq={doc_freq}");
        assert!(
            with_freqs.freqs.iter().any(|&f| f != 1),
            "doc_freq={doc_freq}: the fixture must have real freqs, or \
                 this test proves nothing"
        );
        assert!(
            docs_only.freqs.iter().all(|&f| f == 1),
            "doc_freq={doc_freq}: DocsOnly must report freq 1 everywhere"
        );

        // ... and the same through the lazy cursor, walked to exhaustion.
        let mut lazy = field
            .lazy_postings_with_flags(b"a", &doc_in, PostingsFlags::DocsOnly)
            .unwrap()
            .unwrap();
        let mut walked = Vec::with_capacity(doc_freq as usize);
        loop {
            let d = lazy.next_doc().unwrap();
            if d == crate::postings::NO_MORE_DOCS {
                break;
            }
            assert_eq!(lazy.freq(), Some(1), "doc_freq={doc_freq}");
            walked.push(d);
        }
        assert_eq!(walked, with_freqs.docs, "doc_freq={doc_freq} lazy walk");

        // A skip-heavy walk exercises the same skip on the advance path.
        let mut lazy = field
            .lazy_postings_with_flags(b"a", &doc_in, PostingsFlags::DocsOnly)
            .unwrap()
            .unwrap();
        let last = *with_freqs.docs.last().unwrap();
        assert_eq!(lazy.advance(last).unwrap(), last, "doc_freq={doc_freq}");
    }
}

/// A field that indexes no frequencies at all is unaffected by the flag:
/// there is no frequency block on the wire to skip, and both paths report
/// `1`.
#[test]
fn docs_only_flags_are_a_no_op_for_a_field_without_frequencies() {
    use crate::postings::PostingsFlags;
    let doc_freq = 300;
    let mut term = irregular_docs_term(b"a", doc_freq);
    for d in &mut term.docs {
        d.1 = 1;
    }
    let max_doc = term.docs.last().unwrap().0 + 1;
    let terms = vec![term.clone()];
    let input = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::Docs,
        doc_count: term.docs.len() as i32,
        has_payloads: false,
        terms: &terms,
    };
    let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
    let fis = FieldInfos {
        fields: vec![field_info(0, "f", IndexOptions::Docs)],
    };
    let (fields, doc_in) = open_written(&output, &fis, max_doc);
    let field = fields.field("f").unwrap();
    let with_freqs = field.postings(b"a", Some(&doc_in)).unwrap().unwrap();
    let docs_only = field
        .postings_with_flags(b"a", Some(&doc_in), PostingsFlags::DocsOnly)
        .unwrap()
        .unwrap();
    assert_eq!(docs_only.docs, with_freqs.docs);
    assert_eq!(docs_only.freqs, with_freqs.freqs);
}

/// Same as [`docfreq_at_level1_boundaries_round_trips`] but with
/// [`irregular_docs_term`]'s non-constant doc-ID gaps and widely varying
/// freqs instead of [`varied_docs_term`]'s constant delta-of-2 -- a
/// delta/length-accounting bug in `write_level1_span` could plausibly
/// only surface once the span's actual byte length varies unpredictably
/// with real data, not a uniform pattern.
#[test]
fn docfreq_at_level1_boundary_with_irregular_gaps_and_varying_freqs() {
    let doc_freq = LEVEL1_NUM_DOCS + 100;
    let term = irregular_docs_term(b"a", doc_freq);
    let max_doc = term.docs.last().unwrap().0 + 1;
    let terms = vec![term.clone()];
    let input = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqs,
        doc_count: term.docs.len() as i32,
        has_payloads: false,
        terms: &terms,
    };
    let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
    let fis = FieldInfos {
        fields: vec![field_info(0, "f", IndexOptions::DocsAndFreqs)],
    };
    let (fields, doc_in) = open_written(&output, &fis, max_doc);
    let field = fields.field("f").unwrap();
    assert_eq!(field.seek_exact(b"a").unwrap().doc_freq, doc_freq);
    let postings = field.postings(b"a", Some(&doc_in)).unwrap().unwrap();
    let expected_docs: Vec<i32> = term.docs.iter().map(|&(d, _)| d).collect();
    let expected_freqs: Vec<i32> = term.docs.iter().map(|&(_, f)| f).collect();
    assert_eq!(postings.docs, expected_docs);
    assert_eq!(postings.freqs, expected_freqs);

    // Also confirm the lazy cursor can advance straight to the last doc
    // (exercising skip_level1_to against this same irregular span).
    let mut cursor = field
        .lazy_postings(b"a", &doc_in)
        .unwrap()
        .expect("term must exist");
    let last_doc = term.docs.last().unwrap().0;
    assert_eq!(cursor.advance(last_doc).unwrap(), last_doc);
}

/// The write-side analogue of `postings`'s own
/// `lazy_cursor_advance_skips_whole_corrupted_level1_span_without_decoding_it`
/// test: writes a real level-1 span via [`write_level1_span`], then
/// corrupts its first level-0 block's header bytes in place. An
/// `advance()` to a doc in the trailing tail (past the whole span) must
/// still succeed -- proving `skip_level1_to` jumped straight to
/// `doc_end_fp` without ever reading the corrupted block 0 header. A
/// control `advance()` to a target inside the span forces that same
/// header to be decoded and must surface the corruption, confirming the
/// first assertion wasn't passing by luck (e.g. because the corruption
/// was inert).
#[test]
fn writer_level1_span_advance_past_it_skips_corrupted_first_block_header() {
    let doc_freq = LEVEL1_NUM_DOCS + 8;
    // `irregular_docs_term` (not `varied_docs_term`'s constant delta-2
    // docs): a constant delta of 2 makes `docRange == BLOCK_SIZE * 2`
    // land exactly on the writer's bit-set-vs-packed boundary (see
    // `write_full_block`'s doc comment), so the first level-0 block
    // would be written in the dense bit-set shape rather than the
    // generic packed shape this test's header corruption assumes.
    // `irregular_docs_term`'s widely varying deltas keep the packed
    // shape (IndexOptions::Docs below -> freq ignored anyway).
    let term = irregular_docs_term(b"a", doc_freq);
    let max_doc = term.docs.last().unwrap().0 + 1;
    let terms = vec![term.clone()];
    let input = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::Docs,
        doc_count: term.docs.len() as i32,
        has_payloads: false,
        terms: &terms,
    };
    let mut output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();

    // Locate the level-1 span's first byte in `.doc`: the term's only
    // level-1 entry starts right after the `.doc` index header (this is
    // the field's one and only term), and -- since `IndexOptions::Docs`
    // has no freq, `index_has_freq` is false, so the entry is just
    // `vint(doc_delta)` then `vlong(span_len)` with no freq-gated
    // header fields (see `write_level1_span`'s doc comment) -- the span
    // bytes start immediately after those two fields.
    use lucene_store::data_input::{DataInput, SliceInput};
    let mut r = SliceInput::new(&output.doc);
    codec_util::check_index_header(&mut r, DOC_CODEC, 0, DOC_VERSION_CURRENT, &SEG_ID, SUFFIX)
        .unwrap();
    r.read_vint().unwrap(); // doc_delta
    r.read_vlong().unwrap(); // span_len
    let span_start = r.position();

    // Corrupt the first level-0 block's header (`level0NumBytes`
    // vlong + `docDelta`/`blockLength` fields) and well into its body
    // with bytes whose continuation bits never terminate -- 40 bytes
    // (not just the ~5-byte header) because `write_full_block` can pick
    // any of three doc-delta shapes, and a shorter corrupted run was
    // observed to occasionally decode "successfully" (silently wrong,
    // not an error) for the wider dense-bit-set body; 40 bytes reliably
    // errors regardless of which shape this block took.
    for b in output.doc[span_start..span_start + 40].iter_mut() {
        *b = 0xFF;
    }

    let fis = FieldInfos {
        fields: vec![field_info(0, "f", IndexOptions::Docs)],
    };
    let (fields, doc_in) = open_written(&output, &fis, max_doc);
    let field = fields.field("f").unwrap();

    // Past the whole (corrupted) span, in the tail: must succeed.
    let last_doc = term.docs.last().unwrap().0;
    let mut cursor = field
        .lazy_postings(b"a", &doc_in)
        .unwrap()
        .expect("term must exist");
    assert_eq!(cursor.advance(last_doc).unwrap(), last_doc);

    // Control: a target inside the span forces decoding the corrupted
    // block 0 header, which must surface an error.
    let mut cursor2 = field
        .lazy_postings(b"a", &doc_in)
        .unwrap()
        .expect("term must exist");
    assert!(cursor2.advance(100).is_err());
}

/// `docFreq == LEVEL1_NUM_DOCS - 1` (8191): the largest term size this
/// writer accepts, one doc short of the rejection boundary tested above.
/// Round-tripped through the unmodified reader, not just checked for an
/// `Ok` result.
#[test]
fn docfreq_one_less_than_level1_num_docs_is_accepted() {
    let doc_freq = LEVEL1_NUM_DOCS - 1;
    let term = varied_docs_term(b"a", doc_freq);
    let max_doc = term.docs.last().unwrap().0 + 1;
    let doc_count = term.docs.len() as i32;
    let terms = vec![term.clone()];
    let input = FieldPostingsInput {
        has_payloads: false,
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqs,
        doc_count,
        terms: &terms,
    };
    let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
    let fis = FieldInfos {
        fields: vec![field_info(0, "f", IndexOptions::DocsAndFreqs)],
    };
    let (fields, doc_in) = open_written(&output, &fis, max_doc);
    let field = fields.field("f").unwrap();
    assert_eq!(field.seek_exact(b"a").unwrap().doc_freq, doc_freq);
    let postings = field.postings(b"a", Some(&doc_in)).unwrap().unwrap();
    let expected_docs: Vec<i32> = term.docs.iter().map(|&(d, _)| d).collect();
    let expected_freqs: Vec<i32> = term.docs.iter().map(|&(_, f)| f).collect();
    assert_eq!(postings.docs, expected_docs);
    assert_eq!(postings.freqs, expected_freqs);
}

#[test]
fn rejects_unsupported_index_options() {
    let terms = vec![TermPostings {
        term: b"a".to_vec(),
        docs: vec![(0, 1)],
        ..Default::default()
    }];
    let input = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::None,
        doc_count: 1,
        has_payloads: false,
        terms: &terms,
    };
    assert!(matches!(
        write_single_field(&input, &SEG_ID, SUFFIX),
        Err(Error::UnsupportedIndexOptions(IndexOptions::None))
    ));
}

/// `IndexOptions::DocsAndCustomFreqs` is wire-identical to `DocsAndFreqs`
/// (see the module doc): this round-trips a multi-doc, multi-freq term
/// through the real writer + unmodified reader under that option, proving
/// it's accepted end-to-end rather than just not-rejected.
#[test]
fn docs_and_custom_freqs_round_trips_like_docs_and_freqs() {
    let term = TermPostings {
        term: b"a".to_vec(),
        docs: vec![(0, 3), (1, 1), (5, 7)],
        ..Default::default()
    };
    let max_doc = term.docs.last().unwrap().0 + 1;
    let doc_count = term.docs.len() as i32;
    let terms = vec![term.clone()];
    let input = FieldPostingsInput {
        has_payloads: false,
        field_number: 0,
        index_options: IndexOptions::DocsAndCustomFreqs,
        doc_count,
        terms: &terms,
    };
    let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
    let fis = FieldInfos {
        fields: vec![field_info(0, "f", IndexOptions::DocsAndCustomFreqs)],
    };
    let (fields, doc_in) = open_written(&output, &fis, max_doc);
    let field = fields.field("f").unwrap();
    assert_eq!(field.seek_exact(b"a").unwrap().doc_freq, 3);
    let postings = field.postings(b"a", Some(&doc_in)).unwrap().unwrap();
    let expected_docs: Vec<i32> = term.docs.iter().map(|&(d, _)| d).collect();
    let expected_freqs: Vec<i32> = term.docs.iter().map(|&(_, f)| f).collect();
    assert_eq!(postings.docs, expected_docs);
    assert_eq!(postings.freqs, expected_freqs);
}

/// Many terms, each with several docs, all under `BLOCK_SIZE` -- checks
/// the running `doc_start_fp` delta-threading across more than a
/// handful of terms (the earlier tests only ever have 2-3 terms).
#[test]
fn many_terms_many_docs_each() {
    let mut terms = Vec::new();
    for i in 0..20 {
        let term = format!("term{i:02}").into_bytes();
        let docs: Vec<(i32, i32)> = (0..5).map(|d| (i * 5 + d, (d + 1))).collect();
        terms.push(TermPostings {
            term,
            docs,
            ..Default::default()
        });
    }
    let input = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqs,
        doc_count: 100,
        has_payloads: false,
        terms: &terms,
    };
    let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
    let fis = FieldInfos {
        fields: vec![field_info(0, "f", IndexOptions::DocsAndFreqs)],
    };
    let (fields, doc_in) = open_written(&output, &fis, 100);
    let field = fields.field("f").unwrap();
    assert_eq!(field.num_terms, 20);
    for i in 0..20 {
        let term = format!("term{i:02}").into_bytes();
        let postings = field.postings(&term, Some(&doc_in)).unwrap().unwrap();
        let expected_docs: Vec<i32> = (0..5).map(|d| i * 5 + d).collect();
        let expected_freqs: Vec<i32> = (0..5).map(|d| d + 1).collect();
        assert_eq!(postings.docs, expected_docs, "term{i:02}");
        assert_eq!(postings.freqs, expected_freqs, "term{i:02}");
    }
}

/// 26 terms, one per lowercase letter (`"a0".."z0"`), so the field spans
/// 26 distinct leading bytes.
///
/// **A field whose terms span many leading bytes reads back
/// term-for-term** through the unmodified `blocktree::open`/
/// `postings::DocInput`, whatever blocks `crate::blocktree_writer` cuts
/// it into. (An earlier writer split such a field into one block per
/// leading byte under a root with no output, which real Lucene cannot
/// read.) Every term is looked up independently,
/// not just the first and last. See
/// `crates/lucene-search/tests/postings_writer_round_trip.rs`'s
/// `term_query_finds_correct_docs_across_multiple_tim_blocks` for the same
/// property through a real `search_term_query`.
#[test]
fn a_field_spanning_every_lowercase_leading_byte_reads_back_term_for_term() {
    let mut terms = Vec::new();
    for (i, c) in (b'a'..=b'z').enumerate() {
        let term = vec![c, b'0'];
        let docs: Vec<(i32, i32)> = (0..3).map(|d| ((i as i32) * 3 + d, d + 1)).collect();
        terms.push(TermPostings {
            term,
            docs,
            ..Default::default()
        });
    }
    let input = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqs,
        doc_count: 78,
        has_payloads: false,
        terms: &terms,
    };
    let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
    let fis = FieldInfos {
        fields: vec![field_info(0, "f", IndexOptions::DocsAndFreqs)],
    };
    let (fields, doc_in) = open_written(&output, &fis, 78);
    let field = fields.field("f").unwrap();
    assert_eq!(field.num_terms, 26);
    assert_eq!(field.min_term, b"a0");
    assert_eq!(field.max_term, b"z0");
    for (i, c) in (b'a'..=b'z').enumerate() {
        let term = vec![c, b'0'];
        let postings = field.postings(&term, Some(&doc_in)).unwrap().unwrap();
        let expected_docs: Vec<i32> = (0..3).map(|d| (i as i32) * 3 + d).collect();
        let expected_freqs: Vec<i32> = (0..3).map(|d| d + 1).collect();
        assert_eq!(postings.docs, expected_docs, "term index {i}");
        assert_eq!(postings.freqs, expected_freqs, "term index {i}");
    }
    // A term that doesn't exist must still miss cleanly across a
    // multi-child trie (not just the single-block case).
    assert!(field.seek_exact(b"zz").is_none());
}

fn leading_byte_group_terms(n: u8) -> Vec<TermPostings> {
    (0..n)
        .map(|i| TermPostings {
            term: vec![i, b'x'],
            docs: vec![(i as i32, 1)],
            ..Default::default()
        })
        .collect()
}

/// Terms spanning many distinct leading bytes all land in the one block
/// this writer emits, with no per-byte grouping and so no upper bound on
/// how many distinct leading bytes a field may have. The previous
/// per-leading-byte split capped this at 33 *and* produced a `.tip` root
/// real Lucene could not follow -- see this module's `.tim`/`.tip`
/// comment. 40 is past that old cap on purpose.
#[test]
fn many_distinct_leading_bytes_all_round_trip_through_one_block() {
    for n in [2u8, 33, 34, 40] {
        let terms = leading_byte_group_terms(n);
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count: n as i32,
            has_payloads: false,
            terms: &terms,
        };
        let output = write_single_field(&input, &SEG_ID, SUFFIX)
            .unwrap_or_else(|e| panic!("{n} leading bytes must succeed: {e}"));
        let fis = FieldInfos {
            fields: vec![field_info(0, "f", IndexOptions::DocsAndFreqs)],
        };
        let fields = blocktree::open(
            &output.tim,
            &output.tip,
            &output.tmd,
            &fis,
            &SEG_ID,
            SUFFIX,
            n as i32,
        )
        .unwrap_or_else(|e| panic!("{n}-leading-byte output must open: {e}"));
        let f = fields.field("f").unwrap();
        assert_eq!(f.num_terms, n as i64, "n={n}");
        for i in 0..n {
            let stats = f.seek_exact(&[i, b'x']).unwrap_or_else(|| {
                panic!("term {i} missing with n={n}");
            });
            assert_eq!(stats.doc_freq, 1, "term {i} with n={n}");
        }
    }
}

/// A field whose first term is the empty byte string, alongside terms with
/// several distinct leading bytes.
///
/// The empty term has no leading byte at all, which is what made the
/// removed multi-block writer fall back to a single block for such a
/// field. Every field now takes that path, so what this pins is the
/// remaining half: the empty term is a legal term, sorts first, and reads
/// back as itself rather than as the block's prefix.
#[test]
fn an_empty_term_alongside_distinct_leading_bytes_reads_back_as_itself() {
    let terms = vec![
        TermPostings {
            term: b"".to_vec(),
            docs: vec![(0, 1)],
            ..Default::default()
        },
        TermPostings {
            term: b"m".to_vec(),
            docs: vec![(1, 1)],
            ..Default::default()
        },
        TermPostings {
            term: b"z".to_vec(),
            docs: vec![(2, 1)],
            ..Default::default()
        },
    ];
    let input = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqs,
        doc_count: 3,
        has_payloads: false,
        terms: &terms,
    };
    let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
    let fis = FieldInfos {
        fields: vec![field_info(0, "f", IndexOptions::DocsAndFreqs)],
    };
    let (fields, doc_in) = open_written(&output, &fis, 3);
    let field = fields.field("f").unwrap();
    assert_eq!(field.num_terms, 3);
    assert_eq!(
        field.postings(b"", Some(&doc_in)).unwrap().unwrap().docs,
        vec![0]
    );
    assert_eq!(
        field.postings(b"m", Some(&doc_in)).unwrap().unwrap().docs,
        vec![1]
    );
    assert_eq!(
        field.postings(b"z", Some(&doc_in)).unwrap().unwrap().docs,
        vec![2]
    );
}

/// Positions write-side byte-level round trip through the existing
/// unmodified `postings::read_positions` (no query layer here -- see
/// `crates/lucene-search/tests/postings_writer_round_trip.rs`'s
/// `phrase_query_finds_correct_docs_over_freshly_written_positions` for
/// the required phrase-query capstone proof). Covers a singleton term
/// (`"beta"`, `docFreq == 1`, still needs `.pos` bytes since positions
/// are independent of the `.doc` singleton-pulsing optimization), a
/// multi-doc term, and per-doc freq > 1 (multiple occurrences in one
/// doc), to exercise the position-accumulator reset at each doc's first
/// occurrence.
#[test]
fn positions_round_trip_via_read_positions() {
    let terms = vec![
        TermPostings {
            payload_bytes: Vec::new(),
            payload_lengths: Vec::new(),
            term: b"alpha".to_vec(),
            docs: vec![(0, 2), (3, 1)],
            positions: vec![vec![1, 4], vec![2]],
            offsets: Vec::new(),
        },
        TermPostings {
            payload_bytes: Vec::new(),
            payload_lengths: Vec::new(),
            term: b"beta".to_vec(),
            docs: vec![(1, 3)], // singleton doc, but freq == 3 occurrences
            positions: vec![vec![0, 5, 6]],
            offsets: Vec::new(),
        },
    ];
    let input = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqsAndPositions,
        doc_count: 3,
        has_payloads: false,
        terms: &terms,
    };
    let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();

    let fis = FieldInfos {
        fields: vec![field_info(
            0,
            "body",
            IndexOptions::DocsAndFreqsAndPositions,
        )],
    };
    let (fields, doc_in) = open_written(&output, &fis, 4);
    let pos_in = crate::postings::PosInput::open(&output.pos, &SEG_ID, SUFFIX).expect("open .pos");

    let field = fields.field("body").unwrap();

    let positions = field
        .positions(b"alpha", Some(&doc_in), &pos_in, None)
        .unwrap()
        .unwrap();
    assert_eq!(positions.len(), 2);
    assert_eq!(
        positions[0].iter().map(|p| p.position).collect::<Vec<_>>(),
        vec![1, 4]
    );
    assert_eq!(
        positions[1].iter().map(|p| p.position).collect::<Vec<_>>(),
        vec![2]
    );

    let positions = field
        .positions(b"beta", Some(&doc_in), &pos_in, None)
        .unwrap()
        .unwrap();
    assert_eq!(positions.len(), 1);
    assert_eq!(
        positions[0].iter().map(|p| p.position).collect::<Vec<_>>(),
        vec![0, 5, 6]
    );
}

#[test]
fn rejects_missing_positions_when_index_options_needs_them() {
    let terms = vec![TermPostings {
        payload_bytes: Vec::new(),
        payload_lengths: Vec::new(),
        term: b"a".to_vec(),
        docs: vec![(0, 1)],
        positions: vec![], // no positions supplied, but index_options needs them
        offsets: Vec::new(),
    }];
    let input = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqsAndPositions,
        doc_count: 1,
        has_payloads: false,
        terms: &terms,
    };
    assert!(matches!(
        write_single_field(&input, &SEG_ID, SUFFIX),
        Err(Error::MissingPositions {
            index: 0,
            doc_index: 0
        })
    ));
}

#[test]
fn rejects_positions_freq_mismatch() {
    let terms = vec![TermPostings {
        payload_bytes: Vec::new(),
        payload_lengths: Vec::new(),
        term: b"a".to_vec(),
        docs: vec![(0, 2)],
        positions: vec![vec![1]], // only 1 position but freq == 2
        offsets: Vec::new(),
    }];
    let input = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqsAndPositions,
        doc_count: 1,
        has_payloads: false,
        terms: &terms,
    };
    assert!(matches!(
        write_single_field(&input, &SEG_ID, SUFFIX),
        Err(Error::PositionsFreqMismatch {
            index: 0,
            doc_index: 0,
            positions: 1,
            freq: 2,
        })
    ));
}

#[test]
fn rejects_decreasing_positions_within_a_doc_but_not_repeated_ones() {
    let write = |positions: Vec<i32>| {
        let terms = vec![TermPostings {
            payload_bytes: Vec::new(),
            payload_lengths: Vec::new(),
            term: b"a".to_vec(),
            docs: vec![(0, 2)],
            positions: vec![positions],
            offsets: Vec::new(),
        }];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqsAndPositions,
            doc_count: 1,
            has_payloads: false,
            terms: &terms,
        };
        write_single_field(&input, &SEG_ID, SUFFIX)
    };
    assert!(matches!(
        write(vec![4, 3]),
        Err(Error::PositionsNotAscending {
            index: 0,
            doc_index: 0,
        })
    ));
    // A position increment of 0 repeating the term: Java writes it, as a
    // delta of 0, and it reads back as two occurrences at one position.
    let output = write(vec![3, 3]).expect("a repeated position is legal");
    let fis = FieldInfos {
        fields: vec![field_info(
            0,
            "body",
            IndexOptions::DocsAndFreqsAndPositions,
        )],
    };
    let (fields, doc_in) = open_written(&output, &fis, 1);
    let pos_in = crate::postings::PosInput::open(&output.pos, &SEG_ID, SUFFIX).expect("open .pos");
    let positions = fields
        .field("body")
        .unwrap()
        .positions(b"a", Some(&doc_in), &pos_in, None)
        .unwrap()
        .unwrap();
    assert_eq!(
        positions[0].iter().map(|p| p.position).collect::<Vec<_>>(),
        vec![3, 3]
    );
}

/// Two fields ("title": term-freq only, "body": positions) written in
/// ONE [`write_fields`] call, sharing the same physical `.doc`/`.pos`/
/// `.tim`/`.tip`/`.tmd` buffers — `numFields == 2` in `.tmd`, and each
/// field must be independently seekable/queryable through the existing
/// unmodified `blocktree::open` read side with no cross-contamination
/// (see `crates/lucene-search/tests/postings_writer_round_trip.rs`'s
/// `multi_field_segment_term_queries_are_isolated_per_field` for the
/// required real `search_term_query` end-to-end proof of the same
/// property).
#[test]
fn write_fields_two_fields_share_one_tmd_and_stay_isolated() {
    let title_terms = vec![
        TermPostings {
            term: b"rust".to_vec(),
            docs: vec![(0, 1)],
            ..Default::default()
        },
        TermPostings {
            term: b"tokyo".to_vec(),
            docs: vec![(1, 1)],
            ..Default::default()
        },
    ];
    let body_terms = vec![
        TermPostings {
            payload_bytes: Vec::new(),
            payload_lengths: Vec::new(),
            term: b"fox".to_vec(),
            docs: vec![(0, 1), (2, 1)],
            positions: vec![vec![3], vec![0]],
            offsets: Vec::new(),
        },
        TermPostings {
            payload_bytes: Vec::new(),
            payload_lengths: Vec::new(),
            term: b"rust".to_vec(), // same bytes as "title"'s term, different field
            docs: vec![(1, 2)],
            positions: vec![vec![0, 5]],
            offsets: Vec::new(),
        },
    ];
    let inputs = vec![
        FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count: 2,
            has_payloads: false,
            terms: &title_terms,
        },
        FieldPostingsInput {
            field_number: 1,
            index_options: IndexOptions::DocsAndFreqsAndPositions,
            doc_count: 3,
            has_payloads: false,
            terms: &body_terms,
        },
    ];
    let output = write_fields(&inputs, &SEG_ID, SUFFIX).unwrap();

    let fis = FieldInfos {
        fields: vec![
            field_info(0, "title", IndexOptions::DocsAndFreqs),
            field_info(1, "body", IndexOptions::DocsAndFreqsAndPositions),
        ],
    };
    let fields = blocktree::open(
        &output.tim,
        &output.tip,
        &output.tmd,
        &fis,
        &SEG_ID,
        SUFFIX,
        3,
    )
    .expect("write_fields' own bytes must open cleanly");
    assert!(fields.field("title").is_some());
    assert!(fields.field("body").is_some());

    let doc_in = DocInput::open(&output.doc, &SEG_ID, SUFFIX).expect("open .doc");
    let pos_in = crate::postings::PosInput::open(&output.pos, &SEG_ID, SUFFIX).expect("open .pos");

    let title = fields.field("title").unwrap();
    assert_eq!(title.num_terms, 2);
    let p = title.postings(b"rust", Some(&doc_in)).unwrap().unwrap();
    assert_eq!(p.docs, vec![0]);
    assert!(title.seek_exact(b"fox").is_none()); // no cross-contamination from "body"

    let body = fields.field("body").unwrap();
    assert_eq!(body.num_terms, 2);
    let p = body.postings(b"fox", Some(&doc_in)).unwrap().unwrap();
    assert_eq!(p.docs, vec![0, 2]);
    // "rust" exists in both fields with different postings -- prove
    // "body"'s copy is independent of "title"'s.
    let p = body.postings(b"rust", Some(&doc_in)).unwrap().unwrap();
    assert_eq!(p.docs, vec![1]);
    assert_eq!(p.freqs, vec![2]);
    let positions = body
        .positions(b"rust", Some(&doc_in), &pos_in, None)
        .unwrap()
        .unwrap();
    assert_eq!(
        positions[0]
            .iter()
            .map(|pp| pp.position)
            .collect::<Vec<_>>(),
        vec![0, 5]
    );
}

#[test]
fn write_fields_rejects_an_empty_inputs_slice() {
    assert!(matches!(
        write_fields::<TermPostings>(&[], &SEG_ID, SUFFIX),
        Err(Error::EmptyTerms)
    ));
}

#[test]
fn shared_postings_writer_accepts_an_empty_field_group() {
    for (has_prox, has_pay) in [(false, false), (true, false), (true, true)] {
        let files = write_postings::<TermPostings, Error, _>(
            &[],
            &[],
            Some(has_prox),
            Some(has_pay),
            &SEG_ID,
            SUFFIX,
            |_| panic!("an empty group must not call its terms dictionary"),
        )
        .unwrap();
        crate::postings::DocInput::open(&files.doc, &SEG_ID, SUFFIX).unwrap();
        assert_eq!(!files.pos.is_empty(), has_prox);
        assert_eq!(!files.pay.is_empty(), has_pay);
        if has_prox {
            crate::postings::PosInput::open(&files.pos, &SEG_ID, SUFFIX).unwrap();
        }
        if has_pay {
            crate::postings::PayInput::open(&files.pay, &SEG_ID, SUFFIX).unwrap();
        }
        for bytes in [&files.doc, &files.psm, &files.pos, &files.pay]
            .into_iter()
            .filter(|b| !b.is_empty())
        {
            let footer = bytes.len().checked_sub(codec_util::FOOTER_LENGTH).unwrap();
            codec_util::check_whole_file_footer(bytes, footer).unwrap();
        }
    }
}

#[test]
fn write_fields_three_fields_each_stay_isolated() {
    let a_terms = vec![TermPostings {
        term: b"alpha".to_vec(),
        docs: vec![(0, 1)],
        ..Default::default()
    }];
    let b_terms = vec![TermPostings {
        term: b"beta".to_vec(),
        docs: vec![(1, 1)],
        ..Default::default()
    }];
    let c_terms = vec![TermPostings {
        term: b"gamma".to_vec(),
        docs: vec![(2, 1)],
        ..Default::default()
    }];
    let inputs = vec![
        FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count: 1,
            has_payloads: false,
            terms: &a_terms,
        },
        FieldPostingsInput {
            field_number: 1,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count: 1,
            has_payloads: false,
            terms: &b_terms,
        },
        FieldPostingsInput {
            field_number: 2,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count: 1,
            has_payloads: false,
            terms: &c_terms,
        },
    ];
    let output = write_fields(&inputs, &SEG_ID, SUFFIX).unwrap();

    let fis = FieldInfos {
        fields: vec![
            field_info(0, "a", IndexOptions::DocsAndFreqs),
            field_info(1, "b", IndexOptions::DocsAndFreqs),
            field_info(2, "c", IndexOptions::DocsAndFreqs),
        ],
    };
    let fields = blocktree::open(
        &output.tim,
        &output.tip,
        &output.tmd,
        &fis,
        &SEG_ID,
        SUFFIX,
        3,
    )
    .expect("write_fields' own bytes must open cleanly for 3 fields");

    let doc_in = DocInput::open(&output.doc, &SEG_ID, SUFFIX).expect("open .doc");
    let a = fields.field("a").unwrap();
    assert_eq!(
        a.postings(b"alpha", Some(&doc_in)).unwrap().unwrap().docs,
        vec![0]
    );
    assert!(a.seek_exact(b"beta").is_none());
    assert!(a.seek_exact(b"gamma").is_none());

    let b = fields.field("b").unwrap();
    assert_eq!(
        b.postings(b"beta", Some(&doc_in)).unwrap().unwrap().docs,
        vec![1]
    );
    assert!(b.seek_exact(b"alpha").is_none());
    assert!(b.seek_exact(b"gamma").is_none());

    let c = fields.field("c").unwrap();
    assert_eq!(
        c.postings(b"gamma", Some(&doc_in)).unwrap().unwrap().docs,
        vec![2]
    );
    assert!(c.seek_exact(b"alpha").is_none());
    assert!(c.seek_exact(b"beta").is_none());
}

/// `total_term_freq >= BLOCK_SIZE` alone (via a single doc with a huge
/// freq, so `docFreq == 1`) is accepted. This is the "one doc, many
/// positions" full-position-block case (see
/// [`positions_full_block_from_one_doc_round_trips`] for the round-trip
/// proof); this test only checks it does not error.
#[test]
fn total_term_freq_at_or_above_block_size_from_one_doc_is_now_accepted() {
    let positions: Vec<Vec<i32>> = vec![(0..BLOCK_SIZE).collect()];
    let terms = vec![TermPostings {
        payload_bytes: Vec::new(),
        payload_lengths: Vec::new(),
        term: b"a".to_vec(),
        docs: vec![(0, BLOCK_SIZE)],
        positions,
        offsets: Vec::new(),
    }];
    let input = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqsAndPositions,
        doc_count: 1,
        has_payloads: false,
        terms: &terms,
    };
    write_single_field(&input, &SEG_ID, SUFFIX)
        .expect("docFreq == 1 stays well under BLOCK_SIZE; only docFreq is now bounded");
}

/// `docFreq >= BLOCK_SIZE` while indexing positions is accepted: a `.doc`
/// full block of a positions-indexing field carries pos/pay skip
/// sub-fields, which this writer emits ([`PosSkipWriter`]). The level-0
/// header it produces is read back by the
/// unmodified `crate::postings::read_full_block_header`, which would
/// mis-frame the block if the sub-fields were missing or misplaced.
#[test]
fn doc_freq_at_or_above_block_size_while_indexing_positions_round_trips() {
    let doc_freq = BLOCK_SIZE + 3;
    let terms = vec![TermPostings {
        payload_bytes: Vec::new(),
        payload_lengths: Vec::new(),
        term: b"a".to_vec(),
        docs: (0..doc_freq).map(|i| (i * 2, 1 + (i % 3))).collect(),
        positions: (0..doc_freq)
            .map(|i| (0..1 + (i % 3)).map(|k| k * 2 + (i % 5)).collect())
            .collect(),
        offsets: Vec::new(),
    }];
    let input = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqsAndPositions,
        doc_count: doc_freq,
        has_payloads: false,
        terms: &terms,
    };
    let out = write_single_field(&input, &SEG_ID, SUFFIX).expect("write");
    let fis = FieldInfos {
        fields: vec![field_info(0, "f0", IndexOptions::DocsAndFreqsAndPositions)],
    };
    let (fields, doc_in) = open_written(&out, &fis, doc_freq * 2);
    let f = fields.field("f0").expect("field");
    let postings = f
        .postings(b"a", Some(&doc_in))
        .expect("postings")
        .expect("term present");
    assert_eq!(
        postings.docs,
        (0..doc_freq).map(|i| i * 2).collect::<Vec<i32>>()
    );
    assert_eq!(
        postings.freqs,
        (0..doc_freq).map(|i| 1 + (i % 3)).collect::<Vec<i32>>()
    );
}

/// Builds a term with `doc_freq` docs, doc IDs `0, 2, 4, .. 2*(doc_freq-1)`
/// (varied deltas, not all-1, so `write_full_block` never takes a trivial
/// all-equal-delta shortcut) and per-doc freq `1 + (doc_index % 5)`
/// (varied, some `!= 1`, so the tail-block's freq-exception path and the
/// full block's `pfor_encode` both see non-trivial input).
fn varied_docs_term(term: &[u8], doc_freq: i32) -> TermPostings {
    TermPostings {
        term: term.to_vec(),
        docs: (0..doc_freq).map(|i| (i * 2, 1 + (i % 5))).collect(),
        ..Default::default()
    }
}

/// Unlike [`varied_docs_term`] (a constant doc-delta of 2), this
/// produces genuinely irregular gaps -- deltas cycling through
/// 1/1/1/50/1/1/1/300/... -- and widely varying freqs (1 up to 1000),
/// exercising `write_full_block`'s per-block `bits_required(max_delta)`
/// computation against a real spread of values rather than one that
/// happens to be uniform.
fn irregular_docs_term(term: &[u8], doc_freq: i32) -> TermPostings {
    let deltas = [1i32, 1, 1, 50, 1, 1, 1, 300];
    let mut doc_id = 0i32;
    let mut docs = Vec::with_capacity(doc_freq as usize);
    for i in 0..doc_freq {
        if i > 0 {
            doc_id += deltas[(i as usize) % deltas.len()];
        }
        let freq = 1 + (i * 37) % 1000;
        docs.push((doc_id, freq));
    }
    TermPostings {
        term: term.to_vec(),
        docs,
        ..Default::default()
    }
}

/// Calls [`write_full_block`] directly with `index_has_freq: false` (so
/// `rest` -- the block body -- starts immediately with the `bitsPerValue`
/// token, no impacts-length prefix in front of it) and returns that
/// token, decoded straight off the wire bytes: `level0NumBytes` (plain
/// vlong, always `0` here), then `vint15`/`vlong15` (the doc-delta and
/// blockLength header fields), then the token byte itself. This lets
/// tests assert *which shape the writer picked* (the byte value), not
/// just that the reader can still decode whatever shape came out.
fn full_block_bits_per_value_token(block: &[(i32, i32)], prev_doc_id: i32) -> i8 {
    use lucene_store::data_input::{DataInput, SliceInput};
    let mut out = Vec::new();
    write_full_block(
        &mut out,
        block,
        prev_doc_id,
        &mut BlockWriteCtx {
            index_has_freq: false,
            norms: None,
            maxima: &mut PostingsMaxima::default(),
            skip: &mut PosSkipWriter::new(None, &[], false),
            scratch: &mut BlockScratch::default(),
        },
    );
    let mut r = SliceInput::new(&out);
    let _level0_num_bytes = r.read_vlong().unwrap();
    // vint15: i16, non-negative fast path or a following vint for the
    // high bits -- our test blocks' doc deltas are always small enough
    // for the fast path, but handle both for robustness.
    let s = r.read_i16().unwrap();
    if s < 0 {
        r.read_vint().unwrap();
    }
    // vlong15: same shape, long-widening.
    let s = r.read_i16().unwrap();
    if s < 0 {
        r.read_vlong().unwrap();
    }
    r.read_byte().unwrap() as i8
}

/// All 256 doc deltas are exactly 1 (a term present in 256 consecutive
/// docs with no gaps) -- real Lucene's `docRange == BLOCK_SIZE` case.
/// Asserts the writer picks the `bitsPerValue == 0` "all-256-consecutive"
/// marker (not just that the block round-trips), then round-trips the
/// whole term through the unmodified reader and checks the exact doc ID
/// sequence.
#[test]
fn full_block_all_consecutive_picks_zero_token() {
    let block: Vec<(i32, i32)> = (0..BLOCK_SIZE).map(|i| (i, 1)).collect();
    assert_eq!(full_block_bits_per_value_token(&block, -1), 0);

    let term = TermPostings {
        term: b"a".to_vec(),
        docs: block.clone(),
        ..Default::default()
    };
    let max_doc = term.docs.last().unwrap().0 + 1;
    let doc_count = term.docs.len() as i32;
    let terms = vec![term.clone()];
    let input = FieldPostingsInput {
        has_payloads: false,
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqs,
        doc_count,
        terms: &terms,
    };
    let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
    let fis = FieldInfos {
        fields: vec![field_info(0, "f", IndexOptions::DocsAndFreqs)],
    };
    let (fields, doc_in) = open_written(&output, &fis, max_doc);
    let field = fields.field("f").unwrap();
    let postings = field.postings(b"a", Some(&doc_in)).unwrap().unwrap();
    let expected_docs: Vec<i32> = term.docs.iter().map(|&(d, _)| d).collect();
    assert_eq!(postings.docs, expected_docs);
}

/// A block dense enough that the bit-set shape beats the next
/// `bitsPerValue` step: 256 docs packed into the smallest possible
/// doc-ID span (deltas of 1 except the very last delta of 2, so
/// `docRange == 257` -- one more than `BLOCK_SIZE`, avoiding the
/// `docRange == BLOCK_SIZE` all-consecutive shortcut while staying as
/// dense as possible). `numBitSetLongs = bits2words(257) = 5`, so the
/// bit set costs `5 * 64 = 320` bits, while the next `bitsPerValue` step
/// above `bitsRequired(2) = 2` is `3`, costing `3 * 256 = 768` bits --
/// the bit set wins. Asserts the writer picks `bitsPerValue < 0` with
/// the expected `numLongs`, then round-trips through the unmodified
/// reader.
#[test]
fn full_block_dense_picks_bitset_token() {
    let mut block: Vec<(i32, i32)> = (0..BLOCK_SIZE).map(|i| (i, 1)).collect();
    let last = block.len() - 1;
    block[last].0 += 1; // doc IDs 0..254, then 256 (skipping 255): docRange == 257.
    let token = full_block_bits_per_value_token(&block, -1);
    assert_eq!(token, -5);

    let term = TermPostings {
        term: b"a".to_vec(),
        docs: block.clone(),
        ..Default::default()
    };
    let max_doc = term.docs.last().unwrap().0 + 1;
    let doc_count = term.docs.len() as i32;
    let terms = vec![term.clone()];
    let input = FieldPostingsInput {
        has_payloads: false,
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqs,
        doc_count,
        terms: &terms,
    };
    let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
    let fis = FieldInfos {
        fields: vec![field_info(0, "f", IndexOptions::DocsAndFreqs)],
    };
    let (fields, doc_in) = open_written(&output, &fis, max_doc);
    let field = fields.field("f").unwrap();
    let postings = field.postings(b"a", Some(&doc_in)).unwrap().unwrap();
    let expected_docs: Vec<i32> = term.docs.iter().map(|&(d, _)| d).collect();
    assert_eq!(postings.docs, expected_docs);
}

/// The band that separates Lucene **10.5.0** -- the version this port
/// pins -- from Lucene `main`'s later, looser rule.
///
/// 10.5.0's `Lucene104PostingsWriter.flushDocBlock` picks packed FOR when
/// `numBitsNextBitsPerValue <= docRange`, i.e. against the bit set's
/// *unrounded* size. `main` compares against
/// `numBitSetLongs * Long.SIZE == ceil(docRange/64)*64` instead, which is
/// `>= docRange`, so every block landing in
/// `docRange < numBitsNext <= ceil(docRange/64)*64` gets packed FOR on
/// `main` and the bit set on 10.5.0. Both shapes are legal and this
/// port's reader takes either, so the round-trip assertions in the tests
/// around this one pass whichever way it goes; only asserting the chosen
/// token catches a version mix-up here.
///
/// Construction: 208 deltas of 3 and 48 of 2, so `docRange == 720` and
/// `bitsRequired(3) == 2`, giving `numBitsNextBitsPerValue == 3 * 256 ==
/// 768`. Then `768 > 720` (10.5.0: a 12-long bit set) while
/// `768 <= bits2words(720) * 64 == 12 * 64 == 768` (`main`: packed FOR at
/// `bitsPerValue == 2`). The band is real but narrow, which is why an
/// arbitrary block does not land in it -- the sibling
/// `full_block_dense_picks_bitset_token` (`docRange == 257`) pins the
/// side of the comparison where both versions agree.
#[test]
fn full_block_encoding_choice_matches_lucene_in_the_disputed_band() {
    let deltas: Vec<i32> = std::iter::repeat_n(3, 208)
        .chain(std::iter::repeat_n(2, 48))
        .collect();
    assert_eq!(deltas.len(), BLOCK_SIZE as usize);
    let doc_range: i32 = deltas.iter().sum();
    assert_eq!(doc_range, 720, "test construction");

    let mut doc_id = -1;
    let block: Vec<(i32, i32)> = deltas
        .iter()
        .map(|d| {
            doc_id += d;
            (doc_id, 1)
        })
        .collect();

    // 10.5.0 compares 768 against the unrounded docRange of 720, so the
    // packed branch loses and a 12-long bit set is written. (Lucene
    // `main` compares against 12 * 64 == 768 and would write packed FOR
    // at bitsPerValue == 2 here; asserting `2` means the port drifted
    // onto `main`.)
    assert_eq!(full_block_bits_per_value_token(&block, -1), -12);

    let term = TermPostings {
        term: b"a".to_vec(),
        docs: block.clone(),
        ..Default::default()
    };
    let max_doc = term.docs.last().unwrap().0 + 1;
    let doc_count = term.docs.len() as i32;
    let terms = vec![term.clone()];
    let input = FieldPostingsInput {
        has_payloads: false,
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqs,
        doc_count,
        terms: &terms,
    };
    let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
    let fis = FieldInfos {
        fields: vec![field_info(0, "f", IndexOptions::DocsAndFreqs)],
    };
    let (fields, doc_in) = open_written(&output, &fis, max_doc);
    let field = fields.field("f").unwrap();
    let postings = field.postings(b"a", Some(&doc_in)).unwrap().unwrap();
    let expected_docs: Vec<i32> = term.docs.iter().map(|&(d, _)| d).collect();
    assert_eq!(postings.docs, expected_docs);
}

/// A block whose deltas alternate 1/100 -- sparse enough (`docRange`
/// around 12,900) that the dense bit-set shape (`~203` words, `~12,992`
/// bits) is no cheaper than the next `bitsPerValue` step above
/// `bitsRequired(100) == 7` (`8 * 256 == 2048` bits) -- confirms the
/// plain positive-`bitsPerValue` `ForUtil` path (pre-existing behavior)
/// is still chosen when neither special shape wins.
#[test]
fn full_block_irregular_picks_plain_packed_token() {
    let block: Vec<(i32, i32)> = (0..BLOCK_SIZE)
        .scan(0i32, |doc_id, i| {
            if i > 0 {
                *doc_id += if i % 2 == 0 { 1 } else { 100 };
            }
            Some((*doc_id, 1))
        })
        .collect();
    let token = full_block_bits_per_value_token(&block, -1);
    assert_eq!(token, 7); // bitsRequired(100) == 7.

    let term = TermPostings {
        term: b"a".to_vec(),
        docs: block.clone(),
        ..Default::default()
    };
    let max_doc = term.docs.last().unwrap().0 + 1;
    let doc_count = term.docs.len() as i32;
    let terms = vec![term.clone()];
    let input = FieldPostingsInput {
        has_payloads: false,
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqs,
        doc_count,
        terms: &terms,
    };
    let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
    let fis = FieldInfos {
        fields: vec![field_info(0, "f", IndexOptions::DocsAndFreqs)],
    };
    let (fields, doc_in) = open_written(&output, &fis, max_doc);
    let field = fields.field("f").unwrap();
    let postings = field.postings(b"a", Some(&doc_in)).unwrap().unwrap();
    let expected_docs: Vec<i32> = term.docs.iter().map(|&(d, _)| d).collect();
    assert_eq!(postings.docs, expected_docs);
}

/// `docFreq == BLOCK_SIZE` (256): exactly one full block, no tail block
/// at all -- the boundary the module doc's "no per-term upper bound"
/// claim rests on. Round-tripped through the existing, unmodified
/// `blocktree::open`/`DocInput::read_postings` (not just "didn't
/// panic" -- every doc/freq is asserted).
#[test]
fn docfreq_exactly_one_full_block_no_tail() {
    let term = varied_docs_term(b"a", BLOCK_SIZE);
    let max_doc = term.docs.last().unwrap().0 + 1;
    let doc_count = term.docs.len() as i32;
    let terms = vec![term.clone()];
    let input = FieldPostingsInput {
        has_payloads: false,
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqs,
        doc_count,
        terms: &terms,
    };
    let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
    let fis = FieldInfos {
        fields: vec![field_info(0, "f", IndexOptions::DocsAndFreqs)],
    };
    let (fields, doc_in) = open_written(&output, &fis, max_doc);
    let field = fields.field("f").unwrap();
    let postings = field.postings(b"a", Some(&doc_in)).unwrap().unwrap();
    let expected_docs: Vec<i32> = term.docs.iter().map(|&(d, _)| d).collect();
    let expected_freqs: Vec<i32> = term.docs.iter().map(|&(_, f)| f).collect();
    assert_eq!(postings.docs, expected_docs);
    assert_eq!(postings.freqs, expected_freqs);
}

/// `docFreq == BLOCK_SIZE + 1` (257): one full block plus a one-doc
/// tail block, proving `prev_doc_id` threads correctly from the full
/// block into the tail block's delta base.
#[test]
fn docfreq_one_full_block_plus_one_doc_tail() {
    let term = varied_docs_term(b"a", BLOCK_SIZE + 1);
    let max_doc = term.docs.last().unwrap().0 + 1;
    let doc_count = term.docs.len() as i32;
    let terms = vec![term.clone()];
    let input = FieldPostingsInput {
        has_payloads: false,
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqs,
        doc_count,
        terms: &terms,
    };
    let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
    let fis = FieldInfos {
        fields: vec![field_info(0, "f", IndexOptions::DocsAndFreqs)],
    };
    let (fields, doc_in) = open_written(&output, &fis, max_doc);
    let field = fields.field("f").unwrap();
    let postings = field.postings(b"a", Some(&doc_in)).unwrap().unwrap();
    let expected_docs: Vec<i32> = term.docs.iter().map(|&(d, _)| d).collect();
    let expected_freqs: Vec<i32> = term.docs.iter().map(|&(_, f)| f).collect();
    assert_eq!(postings.docs, expected_docs);
    assert_eq!(postings.freqs, expected_freqs);
}

/// `docFreq == 600`: two full blocks plus an 88-doc tail, exercising
/// full-block-to-full-block `prev_doc_id` chaining (not just
/// full-block-to-tail).
#[test]
fn docfreq_spans_multiple_full_blocks_plus_tail() {
    let doc_freq = 600;
    assert_eq!(doc_freq / BLOCK_SIZE, 2);
    assert_eq!(doc_freq % BLOCK_SIZE, 88);
    let term = varied_docs_term(b"a", doc_freq);
    let max_doc = term.docs.last().unwrap().0 + 1;
    let doc_count = term.docs.len() as i32;
    let terms = vec![term.clone()];
    let input = FieldPostingsInput {
        has_payloads: false,
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqs,
        doc_count,
        terms: &terms,
    };
    let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
    let fis = FieldInfos {
        fields: vec![field_info(0, "f", IndexOptions::DocsAndFreqs)],
    };
    let (fields, doc_in) = open_written(&output, &fis, max_doc);
    let field = fields.field("f").unwrap();
    assert_eq!(field.seek_exact(b"a").unwrap().doc_freq, doc_freq);
    let postings = field.postings(b"a", Some(&doc_in)).unwrap().unwrap();
    let expected_docs: Vec<i32> = term.docs.iter().map(|&(d, _)| d).collect();
    let expected_freqs: Vec<i32> = term.docs.iter().map(|&(_, f)| f).collect();
    assert_eq!(postings.docs, expected_docs);
    assert_eq!(postings.freqs, expected_freqs);
}

/// `docFreq == 2 * BLOCK_SIZE` with irregular, non-constant doc-ID gaps
/// and widely varying freqs (see [`irregular_docs_term`]) -- every
/// other full-block test in this module uses a constant doc-delta,
/// which can't distinguish "the per-block bit width was computed from
/// the real max delta in that block" from "it happened to be right
/// because every delta was identical."
#[test]
fn docfreq_spans_full_blocks_with_irregular_gaps_and_varying_freqs() {
    let doc_freq = 2 * BLOCK_SIZE;
    let term = irregular_docs_term(b"a", doc_freq);
    let max_doc = term.docs.last().unwrap().0 + 1;
    let doc_count = term.docs.len() as i32;
    let terms = vec![term.clone()];
    let input = FieldPostingsInput {
        has_payloads: false,
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqs,
        doc_count,
        terms: &terms,
    };
    let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
    let fis = FieldInfos {
        fields: vec![field_info(0, "f", IndexOptions::DocsAndFreqs)],
    };
    let (fields, doc_in) = open_written(&output, &fis, max_doc);
    let field = fields.field("f").unwrap();
    let postings = field.postings(b"a", Some(&doc_in)).unwrap().unwrap();
    let expected_docs: Vec<i32> = term.docs.iter().map(|&(d, _)| d).collect();
    let expected_freqs: Vec<i32> = term.docs.iter().map(|&(_, f)| f).collect();
    assert_eq!(postings.docs, expected_docs);
    assert_eq!(postings.freqs, expected_freqs);
}

/// A field with `IndexOptions::Docs` (no freqs at all) at `docFreq ==
/// BLOCK_SIZE` still round-trips through a full block -- proves the
/// `index_has_freq == false` branch (no impacts field, no `pfor_encode`
/// freq body) is wired correctly too, not just the freq-carrying case.
#[test]
fn docfreq_exactly_one_full_block_no_freqs() {
    let doc_freq = BLOCK_SIZE;
    let docs: Vec<(i32, i32)> = (0..doc_freq).map(|i| (i * 3, 1)).collect();
    let max_doc = docs.last().unwrap().0 + 1;
    let doc_count = docs.len() as i32;
    let terms = vec![TermPostings {
        term: b"a".to_vec(),
        docs: docs.clone(),
        ..Default::default()
    }];
    let input = FieldPostingsInput {
        has_payloads: false,
        field_number: 0,
        index_options: IndexOptions::Docs,
        doc_count,
        terms: &terms,
    };
    let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
    let fis = FieldInfos {
        fields: vec![field_info(0, "f", IndexOptions::Docs)],
    };
    let (fields, doc_in) = open_written(&output, &fis, max_doc);
    let field = fields.field("f").unwrap();
    let postings = field.postings(b"a", Some(&doc_in)).unwrap().unwrap();
    let expected_docs: Vec<i32> = docs.iter().map(|&(d, _)| d).collect();
    assert_eq!(postings.docs, expected_docs);
}

/// Two fields written in one [`write_fields`] call, only one of which
/// has a full-block term (`docFreq == BLOCK_SIZE`) -- proves full-block
/// emission for one field doesn't corrupt or bleed into a neighboring
/// field's own (small, tail-only) postings, mirroring this module's
/// established multi-field-isolation pattern
/// (`write_fields_two_fields_share_one_tmd_and_stay_isolated`).
#[test]
fn full_block_field_and_small_field_stay_isolated() {
    let full_term = varied_docs_term(b"big", BLOCK_SIZE);
    let small_terms = vec![TermPostings {
        term: b"small".to_vec(),
        docs: vec![(0, 1), (2, 3)],
        ..Default::default()
    }];
    let full_max_doc = full_term.docs.last().unwrap().0 + 1;
    let full_doc_count = full_term.docs.len() as i32;
    let full_terms = vec![full_term.clone()];
    let inputs = vec![
        FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count: full_doc_count,
            has_payloads: false,
            terms: &full_terms,
        },
        FieldPostingsInput {
            field_number: 1,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count: 2,
            has_payloads: false,
            terms: &small_terms,
        },
    ];
    let max_doc = full_max_doc.max(3);
    let output = write_fields(&inputs, &SEG_ID, SUFFIX).unwrap();
    let fis = FieldInfos {
        fields: vec![
            field_info(0, "big_field", IndexOptions::DocsAndFreqs),
            field_info(1, "small_field", IndexOptions::DocsAndFreqs),
        ],
    };
    let fields = blocktree::open(
        &output.tim,
        &output.tip,
        &output.tmd,
        &fis,
        &SEG_ID,
        SUFFIX,
        max_doc,
    )
    .unwrap();
    let doc_in = DocInput::open(&output.doc, &SEG_ID, SUFFIX).expect("open .doc");

    let big = fields.field("big_field").unwrap();
    let big_postings = big.postings(b"big", Some(&doc_in)).unwrap().unwrap();
    let expected_docs: Vec<i32> = full_term.docs.iter().map(|&(d, _)| d).collect();
    let expected_freqs: Vec<i32> = full_term.docs.iter().map(|&(_, f)| f).collect();
    assert_eq!(big_postings.docs, expected_docs);
    assert_eq!(big_postings.freqs, expected_freqs);

    let small = fields.field("small_field").unwrap();
    let small_postings = small.postings(b"small", Some(&doc_in)).unwrap().unwrap();
    assert_eq!(small_postings.docs, vec![0, 2]);
    assert_eq!(small_postings.freqs, vec![1, 3]);
}

/// Several terms in one field, some below and some spanning full
/// blocks, each independently seekable -- proves full-block emission
/// doesn't disturb the term-dictionary metadata threading
/// (`doc_start_fp` deltas) for neighboring terms in the same block-tree
/// leaf block.
#[test]
fn mixed_small_and_full_block_terms_in_one_field() {
    let small = TermPostings {
        term: b"small".to_vec(),
        docs: vec![(0, 2), (5, 1)],
        ..Default::default()
    };
    let big = varied_docs_term(b"zzz", BLOCK_SIZE + 10);
    let max_doc = big.docs.last().unwrap().0 + 1;
    let doc_count = small.docs.len() as i32 + big.docs.len() as i32;
    let terms = vec![small.clone(), big.clone()];
    let input = FieldPostingsInput {
        has_payloads: false,
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqs,
        doc_count,
        terms: &terms,
    };
    let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
    let fis = FieldInfos {
        fields: vec![field_info(0, "f", IndexOptions::DocsAndFreqs)],
    };
    let (fields, doc_in) = open_written(&output, &fis, max_doc);
    let field = fields.field("f").unwrap();

    let postings = field.postings(b"small", Some(&doc_in)).unwrap().unwrap();
    assert_eq!(postings.docs, vec![0, 5]);
    assert_eq!(postings.freqs, vec![2, 1]);

    let postings = field.postings(b"zzz", Some(&doc_in)).unwrap().unwrap();
    let expected_docs: Vec<i32> = big.docs.iter().map(|&(d, _)| d).collect();
    let expected_freqs: Vec<i32> = big.docs.iter().map(|&(_, f)| f).collect();
    assert_eq!(postings.docs, expected_docs);
    assert_eq!(postings.freqs, expected_freqs);
}

/// Builds a term whose `total_term_freq` is exactly `total`, spread
/// across a handful of docs (`docFreq` well under `BLOCK_SIZE`, the
/// ceiling positions once had) with genuinely
/// irregular per-occurrence position deltas -- cycling through
/// 1/1/4/1/1/30/1/1/2/... rather than a uniform delta, so a bug in
/// [`write_full_position_block`]'s flat cross-doc buffering (e.g. an
/// off-by-one at a doc boundary, or the accumulator failing to reset at
/// each doc's first occurrence) would produce a wrong position sequence
/// rather than silently passing on uniform test data. Occurrences are
/// spread across `num_docs` docs as evenly as possible (the last doc
/// absorbing any remainder), so a 256-or-257-long chunk genuinely spans
/// several doc boundaries.
fn irregular_positions_term(term: &[u8], total: i32, num_docs: i32) -> TermPostings {
    let delta_cycle = [1i32, 1, 4, 1, 1, 30, 1, 1, 2, 7];
    let base_freq = total / num_docs;
    let mut freqs = vec![base_freq; num_docs as usize];
    freqs[(num_docs - 1) as usize] += total - base_freq * num_docs;

    let mut docs = Vec::with_capacity(num_docs as usize);
    let mut positions = Vec::with_capacity(num_docs as usize);
    let mut cycle_idx = 0usize;
    for (doc_idx, &freq) in freqs.iter().enumerate() {
        let doc_id = (doc_idx as i32) * 3; // arbitrary but strictly ascending
        docs.push((doc_id, freq));
        let mut doc_positions = Vec::with_capacity(freq as usize);
        let mut pos = 0i32;
        for _ in 0..freq {
            pos += delta_cycle[cycle_idx % delta_cycle.len()];
            cycle_idx += 1;
            doc_positions.push(pos);
        }
        positions.push(doc_positions);
    }
    TermPostings {
        payload_bytes: Vec::new(),
        payload_lengths: Vec::new(),
        term: term.to_vec(),
        docs,
        positions,
        offsets: Vec::new(),
    }
}

/// `total_term_freq == BLOCK_SIZE` (256): exactly one full `.pos`
/// `PForUtil` block, no vint tail at all -- the boundary
/// [`write_position_tail`]'s "no per-term upper bound" claim rests on.
/// Round-tripped through the existing, unmodified
/// `crate::postings::read_positions` (via `FieldTerms::positions`),
/// asserting the exact irregular position sequence per doc, not just
/// counts.
#[test]
fn total_term_freq_exactly_one_full_position_block_round_trips() {
    let term = irregular_positions_term(b"a", BLOCK_SIZE, 5);
    let max_doc = term.docs.last().unwrap().0 + 1;
    let doc_count = term.docs.len() as i32;
    let expected_positions = term.positions.clone();
    let terms = vec![term.clone()];
    let input = FieldPostingsInput {
        has_payloads: false,
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqsAndPositions,
        doc_count,
        terms: &terms,
    };
    let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
    let fis = FieldInfos {
        fields: vec![field_info(0, "f", IndexOptions::DocsAndFreqsAndPositions)],
    };
    let (fields, doc_in) = open_written(&output, &fis, max_doc);
    let pos_in = crate::postings::PosInput::open(&output.pos, &SEG_ID, SUFFIX).expect("open .pos");
    let field = fields.field("f").unwrap();

    let positions = field
        .positions(b"a", Some(&doc_in), &pos_in, None)
        .unwrap()
        .unwrap();
    assert_eq!(positions.len(), expected_positions.len());
    for (doc_idx, (got, expected)) in positions.iter().zip(&expected_positions).enumerate() {
        let got_positions: Vec<i32> = got.iter().map(|p| p.position).collect();
        assert_eq!(&got_positions, expected, "doc index {doc_idx}");
    }
}

/// `total_term_freq == BLOCK_SIZE + 1` (257): one full `.pos` block plus
/// a single-occurrence vint tail, proving the flat cross-doc delta
/// buffer's `start` offset threads correctly from the full block into
/// the tail. Same irregular-delta construction and per-doc assertion
/// style as [`total_term_freq_exactly_one_full_position_block_round_trips`].
/// `lastPosBlockOffset` must be the real byte offset of the vint position
/// tail, not a constant `0`.
///
/// This port's own `read_positions` re-derives the full-block/tail split
/// from `total_term_freq` and never reads the field, so every round-trip
/// test in this module passes either way. Real Lucene does not:
/// `Lucene104PostingsReader.reset` computes `lastPosBlockFP = posStartFP +
/// lastPosBlockOffset` and `refillPositions` switches to
/// `refillLastPositionBlock` the instant `posIn.getFilePointer()` reaches
/// it -- with `0` that is true at the term's very first position block, so
/// Lucene decodes a `PForUtil` block as a vint tail.
///
/// Asserted two ways: the metadata bytes actually written into `.tim`
/// decode back (through the unmodified `postings::decode_term_metadata`)
/// to the expected offset, and reading `.pos` from `posStartFP + that
/// offset` really does land on the vint tail -- the last
/// `totalTermFreq % BLOCK_SIZE` position deltas, in order.
#[test]
fn last_pos_block_offset_locates_the_vint_position_tail() {
    use lucene_store::data_input::{DataInput, SliceInput};

    // One doc, 300 occurrences: one full 256-position `PForUtil` block
    // plus a 44-occurrence vint tail. `docFreq == 1` keeps this inside the
    // writer's positions scope (`docFreq < BLOCK_SIZE`) while
    // `totalTermFreq == 300 > BLOCK_SIZE` puts `lastPosBlockOffset` on the
    // wire.
    let total = 300usize;
    let positions: Vec<i32> = (0..total as i32).map(|i| i * 3).collect();
    let term = TermPostings {
        term: b"a".to_vec(),
        docs: vec![(0, total as i32)],
        positions: vec![positions.clone()],
        ..Default::default()
    };
    let terms = vec![term];
    let input = FieldPostingsInput {
        has_payloads: false,
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqsAndPositions,
        doc_count: 1,
        terms: &terms,
    };
    let out = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();

    // Walk the `.tim` leaf block down to its per-term metadata region --
    // one term, so one block (`crate::blocktree_writer`), read back in
    // wire order.
    let mut r = SliceInput::new(&out.tim);
    codec_util::check_index_header(
        &mut r,
        TERMS_CODEC_NAME,
        BLOCKTREE_VERSION_CURRENT,
        BLOCKTREE_VERSION_CURRENT,
        &SEG_ID,
        SUFFIX,
    )
    .unwrap();
    let code = r.read_vint().unwrap();
    assert_eq!(code >> 1, 1, "one term in the block");
    let code_l = r.read_vlong().unwrap();
    r.skip((code_l >> 3) as usize).unwrap(); // suffix bytes
    let lengths_code = r.read_vint().unwrap();
    r.skip((lengths_code >> 1) as usize).unwrap(); // suffix lengths
    let stats_len = r.read_vint().unwrap() as usize;
    r.skip(stats_len).unwrap(); // per-term stats
    let meta_len = r.read_vint().unwrap() as usize;
    let meta_start = r.position();
    let meta_bytes = r.slice(meta_start, meta_start + meta_len).unwrap();

    let mut mr = SliceInput::new(meta_bytes);
    let meta = crate::postings::decode_term_metadata(
        &mut mr,
        1, // docFreq
        true,
        crate::postings::TermMetadata::EMPTY,
        IndexOptions::DocsAndFreqsAndPositions,
        false,
        total as i64,
    )
    .unwrap();
    assert_ne!(
        meta.last_pos_block_offset, 0,
        "lastPosBlockOffset must point past the full position block"
    );

    // It must land exactly on the vint tail: the remaining 44 position
    // deltas, plain vints, and nothing left over before the footer.
    let tail_fp = meta.pos_start_fp as i64 + meta.last_pos_block_offset;
    let mut pr = SliceInput::new(&out.pos);
    pr.seek(tail_fp as usize).unwrap();
    for i in (BLOCK_SIZE as usize)..total {
        assert_eq!(
            pr.read_vint().unwrap(),
            positions[i] - positions[i - 1],
            "occurrence {i} of the vint tail"
        );
    }
    assert_eq!(
        pr.position(),
        out.pos.len() - codec_util::FOOTER_LENGTH,
        "the vint tail must end exactly where the .pos footer begins"
    );
}

#[test]
fn total_term_freq_one_full_position_block_plus_tail_round_trips() {
    let term = irregular_positions_term(b"a", BLOCK_SIZE + 1, 7);
    let max_doc = term.docs.last().unwrap().0 + 1;
    let doc_count = term.docs.len() as i32;
    let expected_positions = term.positions.clone();
    let terms = vec![term.clone()];
    let input = FieldPostingsInput {
        has_payloads: false,
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqsAndPositions,
        doc_count,
        terms: &terms,
    };
    let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
    let fis = FieldInfos {
        fields: vec![field_info(0, "f", IndexOptions::DocsAndFreqsAndPositions)],
    };
    let (fields, doc_in) = open_written(&output, &fis, max_doc);
    let pos_in = crate::postings::PosInput::open(&output.pos, &SEG_ID, SUFFIX).expect("open .pos");
    let field = fields.field("f").unwrap();

    let positions = field
        .positions(b"a", Some(&doc_in), &pos_in, None)
        .unwrap()
        .unwrap();
    assert_eq!(positions.len(), expected_positions.len());
    for (doc_idx, (got, expected)) in positions.iter().zip(&expected_positions).enumerate() {
        let got_positions: Vec<i32> = got.iter().map(|p| p.position).collect();
        assert_eq!(&got_positions, expected, "doc index {doc_idx}");
    }
}

/// A `docs: vec![(0, 1), (2, 1)...]`-shaped round trip specifically for the "one doc, huge
/// freq" full-position-block case accepted by
/// [`total_term_freq_at_or_above_block_size_from_one_doc_is_now_accepted`]
/// above -- proves that acceptance actually round-trips correctly, not
/// just "doesn't error."
#[test]
fn positions_full_block_from_one_doc_round_trips() {
    let delta_cycle = [1i32, 1, 4, 1, 1, 30, 1, 1, 2, 7];
    let mut doc_positions = Vec::with_capacity(BLOCK_SIZE as usize);
    let mut pos = 0i32;
    for i in 0..BLOCK_SIZE {
        pos += delta_cycle[(i as usize) % delta_cycle.len()];
        doc_positions.push(pos);
    }
    let terms = vec![TermPostings {
        payload_bytes: Vec::new(),
        payload_lengths: Vec::new(),
        term: b"a".to_vec(),
        docs: vec![(0, BLOCK_SIZE)],
        positions: vec![doc_positions.clone()],
        offsets: Vec::new(),
    }];
    let input = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqsAndPositions,
        doc_count: 1,
        has_payloads: false,
        terms: &terms,
    };
    let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
    let fis = FieldInfos {
        fields: vec![field_info(0, "f", IndexOptions::DocsAndFreqsAndPositions)],
    };
    let (fields, doc_in) = open_written(&output, &fis, 1);
    let pos_in = crate::postings::PosInput::open(&output.pos, &SEG_ID, SUFFIX).expect("open .pos");
    let field = fields.field("f").unwrap();

    let positions = field
        .positions(b"a", Some(&doc_in), &pos_in, None)
        .unwrap()
        .unwrap();
    assert_eq!(positions.len(), 1);
    let got_positions: Vec<i32> = positions[0].iter().map(|p| p.position).collect();
    assert_eq!(got_positions, doc_positions);
}

/// Derives deterministic, non-uniform `(startOffset, endOffset)` pairs
/// from a term's already-built `positions` shape: `startOffset =
/// position * 10` (strictly increasing since positions are strictly
/// increasing within a doc, satisfying real Lucene's `startOffset >=
/// lastStartOffset` assertion) and a length cycling through 1/3/2/5/1 so
/// [`Error`]-free offset lengths aren't all identical (which would hide a
/// bug where the writer always took the "length unchanged" tail-encoding
/// branch).
fn offsets_from_positions(positions: &[Vec<i32>]) -> Vec<Vec<(i32, i32)>> {
    let length_cycle = [1i32, 3, 2, 5, 1];
    let mut cycle_idx = 0usize;
    positions
        .iter()
        .map(|doc_positions| {
            doc_positions
                .iter()
                .map(|&p| {
                    let start = p * 10;
                    let len = length_cycle[cycle_idx % length_cycle.len()];
                    cycle_idx += 1;
                    (start, start + len)
                })
                .collect()
        })
        .collect()
}

/// Single position per doc, with offsets: every doc's lone occurrence
/// still needs a correct `startOffsetDelta`/`length` pair even though
/// there's only ever one occurrence to reset against per doc (no
/// intra-doc delta to get wrong, but the accumulator must still reset to
/// `0` at each new doc rather than leaking the previous doc's last
/// `startOffset`). Round-tripped through the existing, unmodified
/// `crate::postings::read_positions` (via `FieldTerms::positions`),
/// asserting exact `(startOffset, endOffset)` pairs, not just positions.
#[test]
fn single_position_per_doc_with_offsets_round_trips() {
    let terms = vec![TermPostings {
        payload_bytes: Vec::new(),
        payload_lengths: Vec::new(),
        term: b"a".to_vec(),
        docs: vec![(0, 1), (2, 1), (5, 1)],
        positions: vec![vec![0], vec![3], vec![7]],
        offsets: vec![vec![(0, 4)], vec![(30, 33)], vec![(70, 77)]],
    }];
    let input = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
        doc_count: 3,
        has_payloads: false,
        terms: &terms,
    };
    let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
    let fis = FieldInfos {
        fields: vec![field_info(
            0,
            "f",
            IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
        )],
    };
    let (fields, doc_in) = open_written(&output, &fis, 6);
    let pos_in = crate::postings::PosInput::open(&output.pos, &SEG_ID, SUFFIX).expect("open .pos");
    let pay_in = crate::postings::PayInput::open(&output.pay, &SEG_ID, SUFFIX).expect("open .pay");
    let field = fields.field("f").unwrap();

    let positions = field
        .positions(b"a", Some(&doc_in), &pos_in, Some(&pay_in))
        .unwrap()
        .unwrap();
    assert_eq!(positions.len(), 3);
    let expected = [(0, 0, 4), (3, 30, 33), (7, 70, 77)];
    for (doc_idx, (got, &(pos, start, end))) in positions.iter().zip(&expected).enumerate() {
        assert_eq!(got.len(), 1, "doc index {doc_idx}");
        assert_eq!(got[0].position, pos, "doc index {doc_idx}");
        assert_eq!(got[0].start_offset, start, "doc index {doc_idx}");
        assert_eq!(got[0].end_offset, end, "doc index {doc_idx}");
    }
}

/// Multiple positions per doc, with offsets: confirms the
/// `startOffsetDelta` is computed relative to the *previous occurrence in
/// the same doc*, not the absolute `startOffset`, and that it resets to
/// `0` at each new doc's first occurrence (not carried over from the
/// previous doc's last `startOffset`).
#[test]
fn multi_position_per_doc_with_offsets_round_trips() {
    let terms = vec![TermPostings {
        payload_bytes: Vec::new(),
        payload_lengths: Vec::new(),
        term: b"a".to_vec(),
        docs: vec![(0, 3), (4, 2)],
        positions: vec![vec![1, 4, 9], vec![0, 2]],
        offsets: vec![vec![(5, 9), (20, 24), (45, 50)], vec![(0, 3), (10, 15)]],
    }];
    let input = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
        doc_count: 2,
        has_payloads: false,
        terms: &terms,
    };
    let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
    let fis = FieldInfos {
        fields: vec![field_info(
            0,
            "f",
            IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
        )],
    };
    let (fields, doc_in) = open_written(&output, &fis, 5);
    let pos_in = crate::postings::PosInput::open(&output.pos, &SEG_ID, SUFFIX).expect("open .pos");
    let pay_in = crate::postings::PayInput::open(&output.pay, &SEG_ID, SUFFIX).expect("open .pay");
    let field = fields.field("f").unwrap();

    let positions = field
        .positions(b"a", Some(&doc_in), &pos_in, Some(&pay_in))
        .unwrap()
        .unwrap();
    assert_eq!(positions.len(), 2);
    let got0: Vec<(i32, i32, i32)> = positions[0]
        .iter()
        .map(|p| (p.position, p.start_offset, p.end_offset))
        .collect();
    assert_eq!(got0, vec![(1, 5, 9), (4, 20, 24), (9, 45, 50)]);
    let got1: Vec<(i32, i32, i32)> = positions[1]
        .iter()
        .map(|p| (p.position, p.start_offset, p.end_offset))
        .collect();
    assert_eq!(got1, vec![(0, 0, 3), (2, 10, 15)]);
}

/// `total_term_freq` large enough to force at least one full
/// `PForUtil`-encoded `.pos`/`.pay` block ([`write_full_position_block`]/
/// [`write_full_offset_block`]), not just the vint-tail path -- proves
/// the full-block offset encoding round-trips exact `(startOffset,
/// endOffset)` pairs, including a length that changes from one
/// occurrence to the next inside a full block (exercising
/// `read_positions`'s `PForUtil`-decoded `offset_lengths` array, not the
/// tail's "reuse unless changed" path). Occurrences span several docs
/// (`docFreq` well under `BLOCK_SIZE`, the ceiling positions once had)
/// via [`irregular_positions_term`], with offsets derived by
/// [`offsets_from_positions`].
#[test]
fn total_term_freq_full_block_with_offsets_round_trips() {
    let mut term = irregular_positions_term(b"a", BLOCK_SIZE + 1, 5);
    term.offsets = offsets_from_positions(&term.positions);
    let max_doc = term.docs.last().unwrap().0 + 1;
    let doc_count = term.docs.len() as i32;
    let expected_positions = term.positions.clone();
    let expected_offsets = term.offsets.clone();
    let terms = vec![term.clone()];
    let input = FieldPostingsInput {
        has_payloads: false,
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
        doc_count,
        terms: &terms,
    };
    let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
    let fis = FieldInfos {
        fields: vec![field_info(
            0,
            "f",
            IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
        )],
    };
    let (fields, doc_in) = open_written(&output, &fis, max_doc);
    let pos_in = crate::postings::PosInput::open(&output.pos, &SEG_ID, SUFFIX).expect("open .pos");
    let pay_in = crate::postings::PayInput::open(&output.pay, &SEG_ID, SUFFIX).expect("open .pay");
    let field = fields.field("f").unwrap();

    let positions = field
        .positions(b"a", Some(&doc_in), &pos_in, Some(&pay_in))
        .unwrap()
        .unwrap();
    assert_eq!(positions.len(), expected_positions.len());
    for (doc_idx, (got, (expected_pos, expected_off))) in positions
        .iter()
        .zip(expected_positions.iter().zip(&expected_offsets))
        .enumerate()
    {
        let got_positions: Vec<i32> = got.iter().map(|p| p.position).collect();
        assert_eq!(&got_positions, expected_pos, "doc index {doc_idx}");
        let got_offsets: Vec<(i32, i32)> =
            got.iter().map(|p| (p.start_offset, p.end_offset)).collect();
        assert_eq!(&got_offsets, expected_off, "doc index {doc_idx}");
    }
}

#[test]
fn rejects_missing_offsets_when_index_options_needs_them() {
    let terms = vec![TermPostings {
        payload_bytes: Vec::new(),
        payload_lengths: Vec::new(),
        term: b"a".to_vec(),
        docs: vec![(0, 1)],
        positions: vec![vec![0]],
        offsets: vec![], // no offsets supplied, but index_options needs them
    }];
    let input = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
        doc_count: 1,
        has_payloads: false,
        terms: &terms,
    };
    assert!(matches!(
        write_single_field(&input, &SEG_ID, SUFFIX),
        Err(Error::MissingOffsets {
            index: 0,
            doc_index: 0
        })
    ));
}

#[test]
fn rejects_offsets_freq_mismatch() {
    let terms = vec![TermPostings {
        payload_bytes: Vec::new(),
        payload_lengths: Vec::new(),
        term: b"a".to_vec(),
        docs: vec![(0, 2)],
        positions: vec![vec![0, 1]],
        offsets: vec![vec![(0, 1)]], // only 1 offset pair but freq == 2
    }];
    let input = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
        doc_count: 1,
        has_payloads: false,
        terms: &terms,
    };
    assert!(matches!(
        write_single_field(&input, &SEG_ID, SUFFIX),
        Err(Error::OffsetsFreqMismatch {
            index: 0,
            doc_index: 0,
            offsets: 1,
            freq: 2,
        })
    ));
}

#[test]
fn rejects_invalid_offsets() {
    let terms = vec![TermPostings {
        payload_bytes: Vec::new(),
        payload_lengths: Vec::new(),
        term: b"a".to_vec(),
        docs: vec![(0, 2)],
        positions: vec![vec![0, 1]],
        offsets: vec![vec![(5, 8), (3, 6)]], // startOffset decreases
    }];
    let input = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
        doc_count: 1,
        has_payloads: false,
        terms: &terms,
    };
    assert!(matches!(
        write_single_field(&input, &SEG_ID, SUFFIX),
        Err(Error::InvalidOffsets {
            index: 0,
            doc_index: 0,
            occurrence: 1,
        })
    ));
}

/// Single position per doc, with payloads and no offsets: every doc's
/// lone occurrence gets its own distinct payload, so the vint-tail path's
/// "reuse unless length changes" convention gets exercised across doc
/// boundaries too (a length change is forced on essentially every
/// occurrence here). Round-tripped through the existing, unmodified
/// `crate::postings::read_positions` (via `FieldTerms::positions`).
#[test]
fn single_position_per_doc_with_payloads_round_trips() {
    let terms = vec![TermPostings {
        term: b"a".to_vec(),
        docs: vec![(0, 1), (2, 1), (5, 1)],
        positions: vec![vec![0], vec![3], vec![7]],
        offsets: Vec::new(),
        payload_bytes: payload_run(&[b"x", b"yy", b"zzz"]).0,
        payload_lengths: payload_run(&[b"x", b"yy", b"zzz"]).1,
    }];
    let input = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqsAndPositions,
        doc_count: 3,
        has_payloads: true,
        terms: &terms,
    };
    let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
    let fis = FieldInfos {
        fields: vec![field_info_with_payloads(
            0,
            "f",
            IndexOptions::DocsAndFreqsAndPositions,
        )],
    };
    let (fields, doc_in) = open_written(&output, &fis, 6);
    let pos_in = crate::postings::PosInput::open(&output.pos, &SEG_ID, SUFFIX).expect("open .pos");
    let pay_in = crate::postings::PayInput::open(&output.pay, &SEG_ID, SUFFIX).expect("open .pay");
    let field = fields.field("f").unwrap();

    let positions = field
        .positions(b"a", Some(&doc_in), &pos_in, Some(&pay_in))
        .unwrap()
        .unwrap();
    assert_eq!(positions.len(), 3);
    let expected: [(i32, &[u8]); 3] = [(0, b"x"), (3, b"yy"), (7, b"zzz")];
    for (doc_idx, (got, &(pos, payload))) in positions.iter().zip(&expected).enumerate() {
        assert_eq!(got.len(), 1, "doc index {doc_idx}");
        assert_eq!(got[0].position, pos, "doc index {doc_idx}");
        assert_eq!(got[0].payload, payload, "doc index {doc_idx}");
    }
}

/// Multiple positions per doc, with payloads: one doc whose occurrences
/// repeat the *same* payload bytes back-to-back (proving the vint tail's
/// payload-length-unchanged suppression correctly reuses the previous
/// length rather than re-writing it, per `Lucene104PostingsWriter.java:
/// 604-617`) and another doc whose occurrences have varying-length
/// payloads (forcing a length rewrite each time). Round-tripped through
/// the existing, unmodified `crate::postings::read_positions`.
#[test]
fn multi_position_per_doc_with_payloads_round_trips() {
    let terms = vec![TermPostings {
        term: b"a".to_vec(),
        docs: vec![(0, 3), (4, 2)],
        positions: vec![vec![1, 4, 9], vec![0, 2]],
        offsets: Vec::new(),
        // doc 0: same 2-byte payload repeated for all 3 occurrences
        // (length-suppression path). doc 1: varying lengths (1 byte,
        // then 3 bytes).
        payload_bytes: payload_run(&[b"ab", b"ab", b"ab", b"c", b"def"]).0,
        payload_lengths: payload_run(&[b"ab", b"ab", b"ab", b"c", b"def"]).1,
    }];
    let input = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqsAndPositions,
        doc_count: 2,
        has_payloads: true,
        terms: &terms,
    };
    let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
    let fis = FieldInfos {
        fields: vec![field_info_with_payloads(
            0,
            "f",
            IndexOptions::DocsAndFreqsAndPositions,
        )],
    };
    let (fields, doc_in) = open_written(&output, &fis, 5);
    let pos_in = crate::postings::PosInput::open(&output.pos, &SEG_ID, SUFFIX).expect("open .pos");
    let pay_in = crate::postings::PayInput::open(&output.pay, &SEG_ID, SUFFIX).expect("open .pay");
    let field = fields.field("f").unwrap();

    let positions = field
        .positions(b"a", Some(&doc_in), &pos_in, Some(&pay_in))
        .unwrap()
        .unwrap();
    assert_eq!(positions.len(), 2);
    let got0: Vec<Vec<u8>> = positions[0].iter().map(|p| p.payload.clone()).collect();
    assert_eq!(got0, vec![b"ab".to_vec(), b"ab".to_vec(), b"ab".to_vec()]);
    let got1: Vec<Vec<u8>> = positions[1].iter().map(|p| p.payload.clone()).collect();
    assert_eq!(got1, vec![b"c".to_vec(), b"def".to_vec()]);
}

/// Payloads combined with offsets on the same field: proves the correct
/// per-position wire interleaving (payload length/bytes *before* offset
/// fields, in both the full-block `.pay` layout and the vint-tail `.pos`
/// layout — see [`write_position_tail`]'s doc comment) by asserting both
/// payload bytes and `(startOffset, endOffset)` decode correctly for the
/// same occurrences.
#[test]
fn payloads_combined_with_offsets_round_trip() {
    let terms = vec![TermPostings {
        term: b"a".to_vec(),
        docs: vec![(0, 2), (3, 1)],
        positions: vec![vec![0, 2], vec![1]],
        offsets: vec![vec![(0, 3), (20, 22)], vec![(10, 15)]],
        payload_bytes: payload_run(&[b"p1", b"", b"p3"]).0,
        payload_lengths: payload_run(&[b"p1", b"", b"p3"]).1,
    }];
    let input = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
        doc_count: 2,
        has_payloads: true,
        terms: &terms,
    };
    let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
    let fis = FieldInfos {
        fields: vec![field_info_with_payloads(
            0,
            "f",
            IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
        )],
    };
    let (fields, doc_in) = open_written(&output, &fis, 4);
    let pos_in = crate::postings::PosInput::open(&output.pos, &SEG_ID, SUFFIX).expect("open .pos");
    let pay_in = crate::postings::PayInput::open(&output.pay, &SEG_ID, SUFFIX).expect("open .pay");
    let field = fields.field("f").unwrap();

    let positions = field
        .positions(b"a", Some(&doc_in), &pos_in, Some(&pay_in))
        .unwrap()
        .unwrap();
    assert_eq!(positions.len(), 2);
    assert_eq!(positions[0].len(), 2);
    assert_eq!(positions[0][0].position, 0);
    assert_eq!(positions[0][0].payload, b"p1");
    assert_eq!(
        (positions[0][0].start_offset, positions[0][0].end_offset),
        (0, 3)
    );
    assert_eq!(positions[0][1].position, 2);
    assert!(positions[0][1].payload.is_empty());
    assert_eq!(
        (positions[0][1].start_offset, positions[0][1].end_offset),
        (20, 22)
    );
    assert_eq!(positions[1].len(), 1);
    assert_eq!(positions[1][0].position, 1);
    assert_eq!(positions[1][0].payload, b"p3");
    assert_eq!(
        (positions[1][0].start_offset, positions[1][0].end_offset),
        (10, 15)
    );
}

/// `total_term_freq` large enough to force at least one full `PForUtil`-
/// encoded `.pos`/`.pay` block ([`write_full_position_block`]/
/// [`write_full_payload_length_block`]), not just the vint-tail path --
/// proves the full-block payload-length/bytes encoding round-trips
/// exact payload bytes, including varying lengths inside a full block
/// (exercising `read_positions`'s `PForUtil`-decoded `payload_lengths`
/// array and the `.pay` byte-run it gates, not the tail's "reuse unless
/// changed" path). Occurrences span several docs (`docFreq` well under
/// `BLOCK_SIZE`, the ceiling positions once had) via
/// [`irregular_positions_term`], with payload lengths cycling through
/// 1/0/3/2 bytes (including an empty payload) so a bug that assumed every
/// payload in a block has the same length would produce wrong bytes.
#[test]
fn total_term_freq_full_block_with_payloads_round_trips() {
    let mut term = irregular_positions_term(b"a", BLOCK_SIZE + 1, 5);
    let length_cycle = [1usize, 0, 3, 2];
    let mut next_byte = 0u8;
    // The per-doc nesting is the *expectation*; the term itself carries
    // the flat run, so this also pins that the two describe the same
    // occurrences in the same order.
    let expected_payloads: Vec<Vec<Vec<u8>>> = term
        .positions
        .iter()
        .map(|doc_positions| {
            doc_positions
                .iter()
                .enumerate()
                .map(|(occ_idx, _)| {
                    let len = length_cycle[occ_idx % length_cycle.len()];
                    (0..len)
                        .map(|_| {
                            let b = next_byte;
                            next_byte = next_byte.wrapping_add(1);
                            b
                        })
                        .collect::<Vec<u8>>()
                })
                .collect()
        })
        .collect();
    for doc_payloads in &expected_payloads {
        for payload in doc_payloads {
            term.payload_lengths.push(payload.len() as u32);
            term.payload_bytes.extend_from_slice(payload);
        }
    }
    let max_doc = term.docs.last().unwrap().0 + 1;
    let doc_count = term.docs.len() as i32;
    let expected_positions = term.positions.clone();
    let terms = vec![term.clone()];
    let input = FieldPostingsInput {
        has_payloads: true,
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqsAndPositions,
        doc_count,
        terms: &terms,
    };
    let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
    let fis = FieldInfos {
        fields: vec![field_info_with_payloads(
            0,
            "f",
            IndexOptions::DocsAndFreqsAndPositions,
        )],
    };
    let (fields, doc_in) = open_written(&output, &fis, max_doc);
    let pos_in = crate::postings::PosInput::open(&output.pos, &SEG_ID, SUFFIX).expect("open .pos");
    let pay_in = crate::postings::PayInput::open(&output.pay, &SEG_ID, SUFFIX).expect("open .pay");
    let field = fields.field("f").unwrap();

    let positions = field
        .positions(b"a", Some(&doc_in), &pos_in, Some(&pay_in))
        .unwrap()
        .unwrap();
    assert_eq!(positions.len(), expected_positions.len());
    for (doc_idx, (got, (expected_pos, expected_pay))) in positions
        .iter()
        .zip(expected_positions.iter().zip(&expected_payloads))
        .enumerate()
    {
        let got_positions: Vec<i32> = got.iter().map(|p| p.position).collect();
        assert_eq!(&got_positions, expected_pos, "doc index {doc_idx}");
        let got_payloads: Vec<Vec<u8>> = got.iter().map(|p| p.payload.clone()).collect();
        assert_eq!(&got_payloads, expected_pay, "doc index {doc_idx}");
    }
}

#[test]
fn rejects_missing_payloads_when_has_payloads_is_set() {
    let terms = vec![TermPostings {
        term: b"a".to_vec(),
        docs: vec![(0, 1)],
        positions: vec![vec![0]],
        offsets: Vec::new(),
        // no payload run supplied, but has_payloads is set
        payload_bytes: Vec::new(),
        payload_lengths: Vec::new(),
    }];
    let input = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqsAndPositions,
        doc_count: 1,
        has_payloads: true,
        terms: &terms,
    };
    assert!(matches!(
        write_single_field(&input, &SEG_ID, SUFFIX),
        Err(Error::PayloadsFreqMismatch {
            index: 0,
            payload_lengths: 0,
            total_term_freq: 1,
        })
    ));
}

/// `permute_payload_run` has the direction hazard every reorder has, plus
/// one of its own: the run carries no per-document index, so it has to be
/// re-sliced from the **pre-permutation** occurrence counts. Pinned with a
/// 3-cycle over three documents whose occurrence counts *and* payload
/// lengths all differ, so neither a reversed permutation nor a
/// fixed-width slicing bug can pass.
#[test]
fn permute_payload_run_moves_each_documents_payloads_with_it() {
    // doc order 0,1,2: doc 0 has 1 occurrence (1 byte), doc 1 has 2
    // (1 and 3 bytes), doc 2 has 1 (2 bytes).
    let bytes = vec![0xA0, 0xB0, 0xB1, 0xB2, 0xB3, 0xC0, 0xC1];
    let lengths = vec![1u32, 1, 3, 2];
    let counts = vec![1u32, 2, 1];
    let (got_bytes, got_lengths) = permute_payload_run(&bytes, &lengths, &counts, &[2, 0, 1]);
    assert_eq!(got_lengths, vec![2, 1, 1, 3]);
    assert_eq!(got_bytes, vec![0xC0, 0xC1, 0xA0, 0xB0, 0xB1, 0xB2, 0xB3]);
}

/// A truncated run must not panic: `PayloadsFreqMismatch` is where a
/// caller that built the two out of step is meant to be told, and a slice
/// panic here would take the process down first. The out-of-range
/// permutation entry is the same rule from the other side.
#[test]
fn permute_payload_run_saturates_on_a_run_shorter_than_its_counts() {
    // Two documents claiming 1 and 3 occurrences over a run that only has
    // one length, and a permutation whose second entry names a document
    // that does not exist. Both truncations drop their run rather than
    // panicking or reading somebody else's bytes.
    let (bytes, lengths) = permute_payload_run(&[1, 2], &[1u32], &[1u32, 3], &[1, 5]);
    assert!(lengths.is_empty(), "got {lengths:?}");
    assert!(bytes.is_empty(), "got {bytes:?}");
    // The in-range half of the same run still moves correctly.
    let (bytes, lengths) = permute_payload_run(&[1, 2], &[1u32], &[1u32, 3], &[0, 1]);
    assert_eq!(lengths, vec![1]);
    assert_eq!(bytes, vec![1]);
}

/// The flat run's second invariant: the bytes must be exactly the
/// concatenation its lengths describe. The nested shape could not express
/// this state at all (a `Vec<u8>` is its own length), so it is new
/// validation for a new representation rather than a re-spelling.
#[test]
fn rejects_payload_bytes_that_do_not_match_their_lengths() {
    let terms = vec![TermPostings {
        term: b"a".to_vec(),
        docs: vec![(0, 2)],
        positions: vec![vec![0, 1]],
        offsets: Vec::new(),
        // lengths sum to 5, but only 3 bytes are supplied
        payload_bytes: b"abc".to_vec(),
        payload_lengths: vec![2, 3],
    }];
    let input = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqsAndPositions,
        doc_count: 1,
        has_payloads: true,
        terms: &terms,
    };
    assert!(matches!(
        write_single_field(&input, &SEG_ID, SUFFIX),
        Err(Error::PayloadBytesMismatch {
            index: 0,
            payload_bytes: 3,
            expected: 5,
        })
    ));
}

/// A payload length is a `u32` in the run and an `int` on the wire, so a
/// length past `i32::MAX` has to be rejected rather than written as a
/// negative vint. Unreachable through `IndexWriter` (no allocation that
/// large exists here) and reachable by any direct `postings_writer`
/// caller, which is the whole reason the check is at the boundary.
#[test]
fn rejects_a_payload_length_that_does_not_fit_an_i32() {
    let terms = vec![TermPostings {
        term: b"a".to_vec(),
        docs: vec![(0, 1)],
        positions: vec![vec![0]],
        offsets: Vec::new(),
        payload_bytes: Vec::new(),
        payload_lengths: vec![i32::MAX as u32 + 1],
    }];
    let input = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqsAndPositions,
        doc_count: 1,
        has_payloads: true,
        terms: &terms,
    };
    assert!(matches!(
        write_single_field(&input, &SEG_ID, SUFFIX),
        Err(Error::PayloadLengthTooLarge {
            index: 0,
            occurrence: 0,
            length: 2_147_483_648,
        })
    ));
}

#[test]
fn rejects_payloads_freq_mismatch() {
    let terms = vec![TermPostings {
        term: b"a".to_vec(),
        docs: vec![(0, 2)],
        positions: vec![vec![0, 1]],
        offsets: Vec::new(),
        // only 1 payload length but freq == 2
        payload_bytes: payload_run(&[b"x"]).0,
        payload_lengths: payload_run(&[b"x"]).1,
    }];
    let input = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqsAndPositions,
        doc_count: 1,
        has_payloads: true,
        terms: &terms,
    };
    assert!(matches!(
        write_single_field(&input, &SEG_ID, SUFFIX),
        Err(Error::PayloadsFreqMismatch {
            index: 0,
            payload_lengths: 1,
            total_term_freq: 2,
        })
    ));
}

/// [`StreamingFieldsWriter`] writes [`write_fields_with_options`]'s
/// bytes whatever the batch split, and a field that gets no terms adds
/// nothing.
#[test]
fn streaming_writer_matches_whole_field_writer() {
    let docs_terms: Vec<TermPostings> = (0..40u8)
        .map(|i| match i % 4 {
            0 => varied_docs_term(&[b'a', i], 1),
            1 => irregular_docs_term(&[b'a', i], 300),
            2 => varied_docs_term(&[b'a', i], 9000),
            _ => irregular_docs_term(&[b'a', i], 17),
        })
        .collect();
    let rich_terms: Vec<TermPostings> = (0..30u8)
        .map(|i| {
            let mut t = irregular_positions_term(
                &[b'b', i],
                40 + 50 * i32::from(i),
                1 + i32::from(i % 7) * 3,
            );
            t.offsets = t
                .positions
                .iter()
                .map(|ps| ps.iter().map(|&p| (p * 5, p * 5 + 3)).collect())
                .collect();
            let total: usize = t.positions.iter().map(Vec::len).sum();
            t.payload_lengths = (0..total).map(|k| (k % 4) as u32).collect();
            t.payload_bytes = (0..t.payload_lengths.iter().sum::<u32>())
                .map(|b| b as u8)
                .collect();
            t
        })
        .collect();
    let pos_terms: Vec<TermPostings> = (0..25u8)
        .map(|i| irregular_positions_term(&[b'c', i], 1 + 20 * i32::from(i), 1 + i32::from(i % 5)))
        .collect();
    let max_doc = 30_000usize;
    let norm_values: Vec<i64> = (0..max_doc as i64).map(|d| 1 + d % 13).collect();
    let norms = [FieldNorms {
        field_number: 2,
        values: &norm_values,
    }];
    let fields = [
        (1, IndexOptions::DocsAndFreqs, false, &docs_terms),
        (
            2,
            IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
            true,
            &rich_terms,
        ),
        (3, IndexOptions::DocsAndFreqsAndPositions, false, &pos_terms),
    ];
    let inputs: Vec<FieldPostingsInput<'_>> = fields
        .iter()
        .map(
            |&(field_number, index_options, has_payloads, terms)| FieldPostingsInput {
                field_number,
                index_options,
                doc_count: 7,
                has_payloads,
                terms,
            },
        )
        .collect();
    let options = WriteOptions {
        has_prox: Some(true),
        has_payloads_or_offsets: Some(true),
        ..WriteOptions::default()
    };
    let id = [7u8; ID_LENGTH];
    let expected =
        write_fields_with_options(&inputs, &norms, &options, &id, "Lucene104_0").unwrap();
    for batch in [1usize, 3, 7, usize::MAX] {
        let mut w = StreamingFieldsWriter::new(&options, &id, "Lucene104_0").unwrap();
        // A field whose terms were all deleted: no record at all.
        w.start_field(0, IndexOptions::Docs, false).unwrap();
        w.finish_field(0).unwrap();
        for input in &inputs {
            w.start_field(input.field_number, input.index_options, input.has_payloads)
                .unwrap();
            for chunk in input.terms.chunks(batch.min(input.terms.len())) {
                w.add_terms(chunk, &norms).unwrap();
            }
            w.finish_field(input.doc_count).unwrap();
        }
        let got = w.finish().unwrap();
        for (name, a, b) in [
            ("doc", &got.doc, &expected.doc),
            ("psm", &got.psm, &expected.psm),
            ("pos", &got.pos, &expected.pos),
            ("pay", &got.pay, &expected.pay),
            ("tim", &got.tim, &expected.tim),
            ("tip", &got.tip, &expected.tip),
            ("tmd", &got.tmd, &expected.tmd),
        ] {
            assert!(a == b, "batch {batch}: .{name} differs");
        }
    }
    // Streamed to files, the same bytes, the dictionary's pointers
    // rebased over what already went out.
    use lucene_store::Directory;
    assert!(expected.doc.len() + expected.pos.len() > 8 * SPILL_BYTES);
    for batch in [1usize, 7, usize::MAX] {
        let root = lucene_util::test_support::TempDir::new("postings-spill");
        let dir = lucene_store::FsDirectory::open(root.path());
        let mut w = StreamingFieldsWriter::new(&options, &id, "Lucene104_0").unwrap();
        w.stream_to(
            dir.create_output("_0.doc").unwrap(),
            Some(dir.create_output("_0.pos").unwrap()),
            Some(dir.create_output("_0.pay").unwrap()),
        );
        for input in &inputs {
            w.start_field(input.field_number, input.index_options, input.has_payloads)
                .unwrap();
            for chunk in input.terms.chunks(batch.min(input.terms.len())) {
                w.add_terms(chunk, &norms).unwrap();
            }
            w.finish_field(input.doc_count).unwrap();
        }
        let got = w.finish().unwrap();
        assert!(got.doc.is_empty() && got.pos.is_empty() && got.pay.is_empty());
        for (name, a, b) in [
            ("psm", &got.psm, &expected.psm),
            ("tim", &got.tim, &expected.tim),
            ("tip", &got.tip, &expected.tip),
            ("tmd", &got.tmd, &expected.tmd),
        ] {
            assert!(a == b, "streamed, batch {batch}: .{name} differs");
        }
        for (ext, b) in [
            ("doc", &expected.doc),
            ("pos", &expected.pos),
            ("pay", &expected.pay),
        ] {
            let file = dir.open(&format!("_0.{ext}")).unwrap();
            assert!(
                file.as_ref() == b.as_slice(),
                "streamed, batch {batch}: .{ext} differs"
            );
        }
    }
}

#[test]
fn streaming_writer_rejects_out_of_order_terms_and_calls() {
    let options = WriteOptions {
        has_prox: Some(false),
        has_payloads_or_offsets: Some(false),
        ..WriteOptions::default()
    };
    let id = [1u8; ID_LENGTH];
    let mut w = StreamingFieldsWriter::new(&options, &id, "").unwrap();
    let b = [varied_docs_term(b"b", 2)];
    let a = [varied_docs_term(b"a", 2)];
    assert!(matches!(
        w.add_terms(&b, &[]),
        Err(Error::StreamingOrder(_))
    ));
    w.start_field(0, IndexOptions::DocsAndFreqs, false).unwrap();
    assert!(matches!(
        w.start_field(1, IndexOptions::DocsAndFreqs, false),
        Err(Error::StreamingOrder(_))
    ));
    w.add_terms(&b, &[]).unwrap();
    assert!(matches!(
        w.add_terms(&a, &[]),
        Err(Error::TermsNotSorted(1))
    ));
    assert!(matches!(
        w.add_terms(&b, &[]),
        Err(Error::TermsNotSorted(1))
    ));
}

/// `payloadByteUpto` as `addPosition` keeps it, one occurrence at a
/// time with the reset when a 256-occurrence block fills, against the
/// on-demand sum the skip writer samples.
#[test]
fn payload_byte_upto_matches_addpositions_running_counter() {
    let lengths: Vec<u32> = (0..1_000u32).map(|i| (i * 37) % 11).collect();
    let mut upto = 0u64;
    for occ in 0..=lengths.len() {
        assert_eq!(
            payload_byte_upto(&lengths, occ as u64),
            upto as i32,
            "occ {occ}"
        );
        if let Some(&l) = lengths.get(occ) {
            upto += u64::from(l);
            if (occ + 1) % BLOCK_SIZE as usize == 0 {
                upto = 0;
            }
        }
    }
    assert_eq!(payload_byte_upto(&[], 300), 0);
}
