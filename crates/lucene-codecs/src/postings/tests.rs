#![cfg(test)]
// The arithmetic gate is about values read off disk; a test's `i + 1` is
// not one (`docs/arithmetic-gate.md`, "Test code").
#![allow(clippy::arithmetic_side_effects)]

use super::*;
use lucene_store::data_output::DataOutput;

/// Run the existing nested oracle and the allocation-reduced decoder on
/// every valid and corrupt case below.
fn read_positions_and_flat(
    pos: &PosInput<'_>,
    pay: Option<&PayInput<'_>>,
    meta: TermMetadata,
    freqs: &[i32],
    total_term_freq: i64,
    index_options: IndexOptions,
    has_payloads: bool,
) -> Result<Vec<Vec<Position>>> {
    let nested = super::read_positions(
        pos,
        pay,
        meta,
        freqs,
        total_term_freq,
        index_options,
        has_payloads,
    );
    let flat = read_occurrences_flat(
        pos,
        pay,
        meta,
        freqs,
        total_term_freq,
        index_options,
        has_payloads,
    );
    match (&nested, &flat) {
        (Ok(expected), Ok(flat)) => {
            for (index, expected) in expected.iter().enumerate() {
                let actual = flat.document(index).expect("decoded document boundaries");
                assert_eq!(
                    actual.positions,
                    expected.iter().map(|p| p.position).collect::<Vec<_>>()
                );
                if index_options.subsumes_offsets() {
                    assert_eq!(
                        actual.start_offsets,
                        expected.iter().map(|p| p.start_offset).collect::<Vec<_>>()
                    );
                    assert_eq!(
                        actual.end_offsets,
                        expected.iter().map(|p| p.end_offset).collect::<Vec<_>>()
                    );
                } else {
                    assert!(actual.start_offsets.is_empty() && actual.end_offsets.is_empty());
                }
                if has_payloads {
                    assert_eq!(
                        actual.payload_lengths,
                        expected
                            .iter()
                            .map(|p| p.payload.len() as u32)
                            .collect::<Vec<_>>()
                    );
                    assert_eq!(
                        actual.payload_bytes,
                        expected
                            .iter()
                            .flat_map(|p| p.payload.iter().copied())
                            .collect::<Vec<_>>()
                    );
                } else {
                    assert!(actual.payload_lengths.is_empty() && actual.payload_bytes.is_empty());
                }
            }
            assert!(flat.document(expected.len()).is_none());
            assert!(flat.document(usize::MAX).is_none());
        }
        (Err(nested), Err(flat)) => assert_eq!(nested.to_string(), flat.to_string()),
        _ => panic!("nested/flat decode disagree: {nested:?} versus {flat:?}"),
    }
    nested
}

#[test]
fn flat_occurrences_empty_documents_and_wrapping_accumulators() {
    let id = [61; ID_LENGTH];
    let (mut pos, footer) = pos_header_and_footer(&id);
    let pos_start_fp = pos.len() as u64;
    // No payloads: raw position deltas can reach i32::MAX. Offset start
    // deltas use unsigned-shifted codes, so two maximum deltas wrap.
    for delta in [i32::MAX, 1, 1] {
        pos.write_vint(delta);
        pos.write_vint(-1); // unsigned >> 1 = i32::MAX; length changes
        pos.write_vint(2);
    }
    pos.extend_from_slice(&footer);
    let pos_in = PosInput::open(&pos, &id, "").unwrap();
    let meta = TermMetadata {
        pos_start_fp,
        ..TermMetadata::EMPTY
    };
    let nested = read_positions_and_flat(
        &pos_in,
        None,
        meta,
        &[0, 2, -1, 1, 0],
        3,
        IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
        false,
    )
    .unwrap();
    assert_eq!(
        nested.iter().map(Vec::len).collect::<Vec<_>>(),
        [0, 2, 0, 1, 0]
    );
    assert_eq!(nested[1][1].position, i32::MIN);
    assert_eq!(nested[1][0].end_offset, i32::MIN + 1);
    assert_eq!(nested[1][1].start_offset, -2);
    assert_eq!(nested[3][0].start_offset, i32::MAX);
    let empty = read_occurrences_flat(
        &pos_in,
        None,
        meta,
        &[],
        0,
        IndexOptions::DocsAndFreqsAndPositions,
        false,
    )
    .unwrap();
    assert!(empty.document(0).is_none());
    let zero_docs = read_occurrences_flat(
        &pos_in,
        None,
        meta,
        &[0, -1],
        0,
        IndexOptions::DocsAndFreqsAndPositions,
        true,
    )
    .unwrap();
    for index in 0..2 {
        assert_eq!(zero_docs.document(index).unwrap().positions, []);
        assert!(zero_docs.document(index).unwrap().payload_bytes.is_empty());
    }
}

#[test]
fn flat_occurrences_truncation_and_invalid_counts_match_nested_errors() {
    let options = IndexOptions::DocsAndFreqsAndPositions;
    for bytes in [&[][..], &[0x80][..]] {
        let pos = PosInput {
            buf: bytes,
            format: PostingsFormat::Lucene104,
        };
        assert!(
            read_positions_and_flat(&pos, None, TermMetadata::EMPTY, &[1], 1, options, false,)
                .is_err()
        );
    }
    let pos = PosInput {
        buf: &[],
        format: PostingsFormat::Lucene104,
    };
    assert!(
        read_positions_and_flat(&pos, None, TermMetadata::EMPTY, &[], -1, options, false,).is_err()
    );
    let past_end = TermMetadata {
        pos_start_fp: 1,
        ..TermMetadata::EMPTY
    };
    assert!(read_positions_and_flat(&pos, None, past_end, &[], 0, options, false,).is_err());
    // A tail's explicit payload length larger than the remaining file and
    // a negative length are rejected by the common wire decoder before
    // either representation can return partially decoded documents.
    for bytes in [&[3, 4, 7][..], &[3, 255, 255, 255, 255, 15][..]] {
        let pos = PosInput {
            buf: bytes,
            format: PostingsFormat::Lucene104,
        };
        assert!(
            read_positions_and_flat(&pos, None, TermMetadata::EMPTY, &[1], 1, options, true,)
                .is_err()
        );
    }
}

#[test]
fn flat_occurrences_document_accessor_checks_private_boundaries() {
    let mut flat = FlatOccurrences {
        positions: vec![1],
        start_offsets: vec![2],
        end_offsets: vec![3],
        payload_lengths: vec![1],
        payload_bytes: vec![7],
        doc_starts: vec![0, 1],
        payload_starts: vec![0, 1],
        has_offsets: true,
        has_payloads: true,
    };
    assert_eq!(flat.document(0).unwrap().payload_bytes, [7]);
    flat.doc_starts[1] = 2;
    assert!(flat.document(0).is_none());
    flat.doc_starts = vec![1, 0];
    assert!(flat.document(0).is_none());
    flat.doc_starts = vec![0, 1];
    flat.start_offsets.clear();
    assert!(flat.document(0).is_none());
    flat.start_offsets.push(2);
    flat.end_offsets.clear();
    assert!(flat.document(0).is_none());
    flat.end_offsets.push(3);
    flat.payload_lengths.clear();
    assert!(flat.document(0).is_none());
    flat.payload_lengths.push(1);
    flat.payload_starts[1] = 2;
    assert!(flat.document(0).is_none());
    flat.payload_starts = vec![1, 0];
    assert!(flat.document(0).is_none());
    flat.payload_starts.clear();
    assert!(flat.document(0).is_none());
}

/// `postings_skip_index` (`GenPostingsSkip.java`) opened: its `pskip`
/// field, the `.doc`/`.pos`/`.pay` inputs, and its dense term -- in all
/// 8 500 documents, one to three positions each, so a positions block
/// holds many documents and a `.doc` block spans several positions
/// blocks.
fn with_skip_fixture(
    f: impl FnOnce(&crate::blocktree::FieldTerms, &DocInput<'_>, &PosInput<'_>, &PayInput<'_>, &[u8]),
) {
    let dir = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/postings_skip_index/"
    );
    let text = std::fs::read_to_string(format!("{dir}manifest.properties")).unwrap();
    let get = |k: &str| {
        text.lines()
            .filter_map(|l| l.split_once('='))
            .find(|(key, _)| *key == k)
            .map(|(_, v)| v.to_string())
            .unwrap()
    };
    let raw = |k: &str| std::fs::read(format!("{dir}{}.raw", get(k))).unwrap();
    let hex = get("id_hex");
    let mut id = [0u8; 16];
    for (i, b) in id.iter_mut().enumerate() {
        *b = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap();
    }
    let suffix = get("segment_suffix");
    let (fnm, tim, tip, tmd) = (
        raw("fnm_file_name"),
        raw("tim_file_name"),
        raw("tip_file_name"),
        raw("tmd_file_name"),
    );
    let (doc, pos, pay) = (
        raw("doc_file_name"),
        raw("pos_file_name"),
        raw("pay_file_name"),
    );
    let infos = crate::field_infos::parse(&fnm, &id, "").unwrap();
    let fields = crate::blocktree::open(
        &tim,
        &tip,
        &tmd,
        &infos,
        &id,
        &suffix,
        get("max_doc").parse().unwrap(),
    )
    .unwrap();
    let field = fields.field("pskip").unwrap();
    let doc_in = DocInput::open(&doc, &id, &suffix).unwrap();
    let pos_in = PosInput::open(&pos, &id, &suffix).unwrap();
    let pay_in = PayInput::open(&pay, &id, &suffix).unwrap();
    f(field, &doc_in, &pos_in, &pay_in, get("term").as_bytes());
}

/// A term whose documents' frequencies claim more positions than its
/// `totalTermFreq` accounts for (a corrupt `.tim`): once the full
/// positions blocks run out, the "last block" holds none. Reading
/// positions must then fail, not spin on an empty refill forever --
/// a hang `catch_unwind` at the FFI boundary could never rescue.
#[test]
fn positions_past_a_short_total_term_freq_are_an_error_not_a_hang() {
    with_skip_fixture(|field, doc_in, pos_in, _, term| {
        let (stats, meta) = field.term_state(term).unwrap().unwrap();
        assert!(stats.total_term_freq > 2 * BLOCK_SIZE as i64);
        for bulk in [false, true] {
            let docs = doc_in
                .lazy_cursor_with_flags(
                    meta,
                    stats.doc_freq,
                    IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
                    true,
                    PostingsFlags::Freqs,
                )
                .unwrap();
            // A multiple of the block size: the tail the real total leaves
            // (at `lastPosBlockOffset`) now holds no positions at all.
            let short = stats.total_term_freq / BLOCK_SIZE as i64 * BLOCK_SIZE as i64;
            let mut c = PositionsCursor::new(docs, pos_in, meta, short, true, true).unwrap();
            let mut out = Vec::new();
            let err = loop {
                if c.next_doc().unwrap() == NO_MORE_DOCS {
                    panic!("every position read past a short totalTermFreq");
                }
                let r = if bulk {
                    out.clear();
                    c.positions_into(&mut out)
                } else {
                    (0..c.freq()).try_for_each(|_| c.next_position().map(drop))
                };
                if let Err(e) = r {
                    break e;
                }
            };
            assert!(err.to_string().contains("totalTermFreq"), "{err}");
        }
    });
}

/// The inline step ([`PositionsCursor::try_next_position`] and
/// `try_start_doc`) against a cursor reading every document whole
/// ([`PositionsCursor::positions_into`]): documents skipped, `k < freq`
/// positions read and the rest left behind, under several patterns, each
/// position compared. The walk must take `try_start_doc`'s step over
/// skipped documents inside a `.doc` block, with positions left over from
/// the last document read, and onto a document whose first position is
/// exactly at the end of the decoded positions block (the next read
/// refills) -- each counted, so a fixture change that stopped reaching
/// one fails here.
#[test]
fn inline_position_steps_match_whole_document_reads() {
    with_skip_fixture(|field, doc_in, pos_in, _, term| {
        let open = || field.lazy_positions(term, doc_in, pos_in).unwrap().unwrap();
        let mut whole = open();
        let mut expected = Vec::new();
        while whole.next_doc().unwrap() != NO_MORE_DOCS {
            let mut out = Vec::new();
            whole.positions_into(&mut out).unwrap();
            expected.push(out);
        }
        assert!(expected.iter().map(Vec::len).sum::<usize>() > 2 * BLOCK_SIZE as usize);
        let (mut skipped_between, mut left_over, mut at_block_end) = (0, 0, 0);
        let mut seed = 0x5eed_u64;
        for (skip_one_in, read) in [(1u64, 0usize), (2, 1), (3, 2), (5, usize::MAX), (7, 1)] {
            let mut c = open();
            let mut i = 0;
            while c.next_doc().unwrap() != NO_MORE_DOCS {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                if skip_one_in > 1 && (seed >> 33).is_multiple_of(skip_one_in) {
                    i += 1;
                    continue;
                }
                let want = &expected[i];
                let k = if read == 0 {
                    want.len()
                } else {
                    read.min(want.len())
                };
                // What `try_start_doc` is about to step over.
                let between = c.pos_doc_upto < c.docs.block_pos && c.docs.block_gen == c.block_gen;
                let doc_left = c.doc_left;
                if c.try_start_doc() {
                    skipped_between += usize::from(between);
                    left_over += usize::from(doc_left > 0);
                    at_block_end += usize::from(c.buf_upto == c.block.len);
                }
                let got: Vec<i32> = (0..k).map(|_| c.next_position().unwrap()).collect();
                assert_eq!(got, want[..k], "doc #{i}, pattern {skip_one_in}/{read}");
                i += 1;
            }
            assert_eq!(i, expected.len());
        }
        assert!(skipped_between > 0, "no inline step over skipped documents");
        assert!(left_over > 0, "no inline step over a partly read document");
        assert!(
            at_block_end > 0,
            "no inline step landing at the block's end"
        );
    });
}

/// A corrupt per-document frequency in the current `.doc` block: zero
/// for the document read (`try_start_doc` declines it, `start_doc`
/// reports it), negative for one stepped over.
#[test]
fn a_non_positive_frequency_fails_the_next_position() {
    with_skip_fixture(|field, doc_in, pos_in, _, term| {
        let open = || field.lazy_positions(term, doc_in, pos_in).unwrap().unwrap();
        let mut c = open();
        c.next_doc().unwrap();
        let at = c.docs.block_pos;
        c.docs.block_freqs[at] = 0;
        let err = c.next_position().unwrap_err().to_string();
        assert!(err.contains("is not positive"), "{err}");

        let mut c = open();
        c.next_doc().unwrap();
        c.next_position().unwrap();
        c.next_doc().unwrap();
        c.next_doc().unwrap();
        let at = c.docs.block_pos;
        assert!(at >= 2 && c.pos_doc_upto < at);
        c.docs.block_freqs[at - 1] = -1;
        let err = c.next_position().unwrap_err().to_string();
        assert!(err.contains("negative per-doc frequency"), "{err}");
    });
}

/// A payload is the last position's of the current document: after the
/// cursor moves on, and before the new document's first position is
/// read, there is none -- not the previous document's last.
#[test]
fn payload_is_none_before_the_documents_first_position() {
    with_skip_fixture(|field, doc_in, pos_in, pay_in, term| {
        let mut c = field.lazy_positions(term, doc_in, pos_in).unwrap().unwrap();
        c.read_payloads(pay_in).unwrap();
        let mut seen = 0;
        while c.next_doc().unwrap() != NO_MORE_DOCS && seen < 50 {
            assert_eq!(c.payload().unwrap(), None, "doc {}", c.doc_id());
            for _ in 0..c.freq() {
                c.next_position().unwrap();
                seen += usize::from(c.payload().unwrap().is_some());
            }
        }
        assert!(seen > 0, "no payload read at all");
    });
}

/// Test-only encoder mirroring `Lucene104PostingsWriter.writeImpacts`
/// (`Lucene104PostingsWriter.java:540-556`) exactly, so
/// `decode_impacts`'s own tests can round-trip through the real writer
/// logic rather than hand-picked bytes only.
fn write_impacts(out: &mut Vec<u8>, impacts: &[Impact]) {
    let mut prev_freq = 0i32;
    let mut prev_norm = 0i64;
    for impact in impacts {
        let freq_delta = impact.freq - prev_freq - 1;
        let norm_delta = impact.norm - prev_norm - 1;
        if norm_delta == 0 {
            out.write_vint(freq_delta << 1);
        } else {
            out.write_vint((freq_delta << 1) | 1);
            out.write_zlong(norm_delta);
        }
        prev_freq = impact.freq;
        prev_norm = impact.norm;
    }
}

#[test]
fn decode_impacts_empty_bytes_is_empty_list() {
    assert_eq!(decode_impacts(&[]).unwrap(), Vec::new());
}

#[test]
fn decode_impacts_single_entry_implicit_norm_delta() {
    // freq=5, norm=1 (the common "norm only ever increases by 1" case,
    // encoded with the low bit clear and no zlong at all).
    let impacts = vec![Impact { freq: 5, norm: 1 }];
    let mut bytes = Vec::new();
    write_impacts(&mut bytes, &impacts);
    assert_eq!(bytes.len(), 1); // single vint, no zlong byte
    assert_eq!(decode_impacts(&bytes).unwrap(), impacts);
}

#[test]
fn decode_impacts_multiple_entries_with_explicit_norm_deltas() {
    // Strictly increasing freq and (unsigned) norm, some entries with a
    // norm jump bigger than 1 (forces the zlong branch), including a
    // negative-looking (but unsigned-larger) norm value to exercise
    // zigzag encoding of a negative delta relative to the accumulated
    // norm (Lucene norms are `long`, not necessarily positive bytes).
    let impacts = vec![
        Impact { freq: 1, norm: 1 },
        Impact { freq: 3, norm: 2 },   // normDelta 0 -> implicit
        Impact { freq: 10, norm: 50 }, // normDelta 47 -> explicit zlong
        Impact {
            freq: 20,
            norm: i64::MAX,
        }, // huge normDelta -> explicit zlong, multi-byte
    ];
    let mut bytes = Vec::new();
    write_impacts(&mut bytes, &impacts);
    assert_eq!(decode_impacts(&bytes).unwrap(), impacts);
}

/// The pre-visitor decoder, every varint through `SliceInput`: the
/// reference [`for_each_impact`]'s one-byte fast paths must agree with.
fn decode_impacts_reference(bytes: &[u8]) -> Result<Impacts> {
    let mut r = SliceInput::new(bytes);
    let (mut freq, mut norm, mut out) = (0i32, 0i64, Vec::new());
    while r.position() < bytes.len() {
        let freq_delta = r.read_vint()?;
        freq = freq.wrapping_add(1i32.wrapping_add(((freq_delta as u32) >> 1) as i32));
        norm = if freq_delta & 1 != 0 {
            norm.wrapping_add(1i64.wrapping_add(r.read_zlong()?))
        } else {
            norm.wrapping_add(1)
        };
        out.push(Impact { freq, norm });
    }
    Ok(out)
}

#[test]
fn for_each_impact_agrees_with_the_reference_decoder() {
    // Deltas of one byte and several, negative and positive norm deltas,
    // and every truncation of each run.
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for _ in 0..400 {
        let mut bytes = Vec::new();
        for _ in 0..(next() % 12) {
            let freq_delta = (next() % 3) as u32;
            let freq_delta = [freq_delta, 40 + freq_delta * 50, 70_000][(next() % 3) as usize];
            let explicit = next() % 2 == 1;
            bytes.write_vint(((freq_delta << 1) | u32::from(explicit)) as i32);
            if explicit {
                let deltas = [0i64, 3, -2, 63, -64, 64, -65, 1 << 40];
                bytes.write_zlong(deltas[(next() % deltas.len() as u64) as usize]);
            }
        }
        for cut in 0..=bytes.len() {
            let run = &bytes[..cut];
            let mut got = Vec::new();
            let visited = for_each_impact(run, |freq, norm| got.push(Impact { freq, norm }));
            match decode_impacts_reference(run) {
                Ok(want) => {
                    assert!(visited.is_ok(), "{run:?}");
                    assert_eq!(got, want, "{run:?}");
                }
                Err(_) => assert!(visited.is_err(), "{run:?}"),
            }
        }
    }
}

#[test]
fn decode_impacts_rejects_truncated_final_vint() {
    // A freqDelta vint whose continuation bit is set but with no
    // following byte in the slice -- decode_impacts must surface this as
    // an EOF error, not silently stop or panic.
    let bytes = [0x80u8]; // continuation bit set, no next byte
    assert!(decode_impacts(&bytes).is_err());
}

#[test]
fn decode_impacts_rejects_truncated_zlong_after_explicit_flag() {
    // freqDelta with the low bit set (an explicit normDelta follows) but
    // the zlong bytes are missing entirely.
    let mut bytes = Vec::new();
    bytes.write_vint(1); // freqDelta << 1 | 1, i.e. freqDelta=0, explicit flag
                         // no zlong bytes follow
    assert!(decode_impacts(&bytes).is_err());
}

#[test]
fn an_impacts_freq_delta_at_the_top_of_i32_wraps_rather_than_panicking() {
    // `readImpacts` accumulates `freq += 1 + (freqDelta >>> 1)` in a Java
    // `int`, which wraps. `freqDelta` is a `readVInt`, and a five-byte
    // varint with the sign bit set decodes to a negative `i32`, so
    // `(freqDelta as u32) >> 1` reaches `i32::MAX` -- and the `1 +` in
    // front of it overflowed *before* the accumulator's `wrapping_add`
    // ever saw it. That is a debug-build panic, which through the FFI is
    // a dead JVM, for an impacts run a corrupt `.doc` can carry.
    //
    // A single-bit flip cannot produce this five-byte pattern, which is
    // why the byte-flip sweeps do not reach it.
    let mut bytes = Vec::new();
    bytes.write_vint(-2); // freqDelta == -2: (0xFFFF_FFFE >> 1) == i32::MAX,
                          // and the low bit is clear so no zlong follows
    let impacts = decode_impacts(&bytes).expect("a wrapped freq, not a panic");
    assert_eq!(impacts.len(), 1);
    // 0 + 1 + i32::MAX, wrapped: exactly what Java's `int` produces.
    assert_eq!(impacts[0].freq, i32::MIN);
    assert_eq!(impacts[0].norm, 1); // low bit clear -> implicit +1
}

fn header_and_footer(codec: &str, id: &[u8; ID_LENGTH]) -> (Vec<u8>, Vec<u8>) {
    let mut before = Vec::new();
    codec_util::write_index_header(&mut before, codec, VERSION_CURRENT, id, "");
    let mut after = Vec::new();
    codec_util::write_footer(&mut after);
    (before, after)
}

#[test]
fn open_rejects_bad_header() {
    let id = [1u8; ID_LENGTH];
    let mut doc = Vec::new();
    codec_util::write_index_header(&mut doc, "WrongCodec", VERSION_CURRENT, &id, "");
    codec_util::write_footer(&mut doc);
    assert!(DocInput::open(&doc, &id, "").is_err());
}

#[test]
fn read_postings_two_docs_with_freqs() {
    let id = [2u8; ID_LENGTH];
    let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
    let doc_start_fp = doc.len() as u64;
    // docFreq=2: deltas [3, 2] (docIDs 2 and 4), freqs [2, 1].
    // group-varint packing: (delta<<1)|(freq==1?1:0)
    doc.write_group_vints(&[3 << 1, (2 << 1) | 1]);
    doc.write_vint(2); // explicit freq for the first doc (freq != 1)
    doc.extend_from_slice(&footer);

    let input = DocInput::open(&doc, &id, "").unwrap();
    let meta = TermMetadata {
        doc_start_fp,
        singleton_doc_id: -1,
        ..TermMetadata::EMPTY
    };
    let postings = input
        .read_postings(meta, 2, IndexOptions::DocsAndFreqs, false)
        .unwrap();
    assert_eq!(postings.docs, vec![2, 4]);
    assert_eq!(postings.freqs, vec![2, 1]);
}

#[test]
fn read_postings_docs_only_no_freqs() {
    let id = [3u8; ID_LENGTH];
    let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
    let doc_start_fp = doc.len() as u64;
    // docFreq=3, plain deltas (no freq bit-packing): docIDs 0,1,5 -> deltas 1,1,4
    doc.write_group_vints(&[1, 1, 4]);
    doc.extend_from_slice(&footer);

    let input = DocInput::open(&doc, &id, "").unwrap();
    let meta = TermMetadata {
        doc_start_fp,
        singleton_doc_id: -1,
        ..TermMetadata::EMPTY
    };
    let postings = input
        .read_postings(meta, 3, IndexOptions::Docs, false)
        .unwrap();
    assert_eq!(postings.docs, vec![0, 1, 5]);
    assert_eq!(postings.freqs, vec![1, 1, 1]);
}

#[test]
fn read_postings_all_freq_one_docs_only_bit_path() {
    // Every doc has freq==1 (bit set), so no trailing freq vints at all --
    // exercises the branch where the second (freq-exception) loop in
    // `read_postings` never fires.
    let id = [6u8; ID_LENGTH];
    let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
    let doc_start_fp = doc.len() as u64;
    // docIDs 0, 3, 4 (deltas 1, 3, 1), freq==1 for all -> bit always set.
    doc.write_group_vints(&[(1 << 1) | 1, (3 << 1) | 1, (1 << 1) | 1]);
    doc.extend_from_slice(&footer);

    let input = DocInput::open(&doc, &id, "").unwrap();
    let meta = TermMetadata {
        doc_start_fp,
        singleton_doc_id: -1,
        ..TermMetadata::EMPTY
    };
    let postings = input
        .read_postings(meta, 3, IndexOptions::DocsAndFreqs, false)
        .unwrap();
    assert_eq!(postings.docs, vec![0, 3, 4]);
    assert_eq!(postings.freqs, vec![1, 1, 1]);
}

/// `IndexOptions::DocsAndCustomFreqs` is wire-identical to `DocsAndFreqs`
/// (see the module doc): the exact same `.doc` bytes as
/// `read_postings_all_freq_one_docs_only_bit_path` above decode
/// identically when read under this option, for both the eager
/// [`DocInput::read_postings`] and the lazy [`DocInput::lazy_cursor`]
/// path.
#[test]
fn read_postings_docs_and_custom_freqs_matches_docs_and_freqs() {
    let id = [6u8; ID_LENGTH];
    let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
    let doc_start_fp = doc.len() as u64;
    doc.write_group_vints(&[(1 << 1) | 1, (3 << 1) | 1, (1 << 1) | 1]);
    doc.extend_from_slice(&footer);

    let input = DocInput::open(&doc, &id, "").unwrap();
    let meta = TermMetadata {
        doc_start_fp,
        singleton_doc_id: -1,
        ..TermMetadata::EMPTY
    };
    let postings = input
        .read_postings(meta, 3, IndexOptions::DocsAndCustomFreqs, false)
        .unwrap();
    assert_eq!(postings.docs, vec![0, 3, 4]);
    assert_eq!(postings.freqs, vec![1, 1, 1]);

    let mut cursor = input
        .lazy_cursor(meta, 3, IndexOptions::DocsAndCustomFreqs, false)
        .unwrap();
    let mut docs = Vec::new();
    loop {
        let d = cursor.next_doc().unwrap();
        if d == NO_MORE_DOCS {
            break;
        }
        docs.push(d);
    }
    assert_eq!(docs, vec![0, 3, 4]);
}

/// `IndexOptions::None` isn't a valid postings-carrying option (real
/// Lucene never writes `.doc` bytes for an unindexed field) -- both
/// decode entry points reject it, distinct from the now-accepted
/// `DocsAndCustomFreqs` case above.
#[test]
fn read_postings_rejects_index_options_none() {
    let id = [6u8; ID_LENGTH];
    let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
    let doc_start_fp = doc.len() as u64;
    doc.extend_from_slice(&footer);
    let input = DocInput::open(&doc, &id, "").unwrap();
    let meta = TermMetadata {
        doc_start_fp,
        singleton_doc_id: -1,
        ..TermMetadata::EMPTY
    };
    assert!(matches!(
        input.read_postings(meta, 3, IndexOptions::None, false),
        Err(Error::Unsupported(_))
    ));
    assert!(matches!(
        input.lazy_cursor(meta, 3, IndexOptions::None, false),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn read_postings_block_size_minus_one_docs() {
    // docFreq == BLOCK_SIZE - 1 (255): the largest docFreq this slice's
    // group-varint (non-PFOR) path supports -- one below the boundary
    // where `read_postings` rejects with Unsupported.
    let id = [7u8; ID_LENGTH];
    let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
    let doc_start_fp = doc.len() as u64;
    let n = (BLOCK_SIZE - 1) as usize;
    // Consecutive doc IDs 0..n, delta=1 each, freq==2 for every doc (bit
    // clear) so every doc also needs a trailing freq vint.
    let deltas: Vec<u32> = (0..n).map(|_| 1u32 << 1).collect();
    doc.write_group_vints(&deltas);
    for _ in 0..n {
        doc.write_vint(2);
    }
    doc.extend_from_slice(&footer);

    let input = DocInput::open(&doc, &id, "").unwrap();
    let meta = TermMetadata {
        doc_start_fp,
        singleton_doc_id: -1,
        ..TermMetadata::EMPTY
    };
    let postings = input
        .read_postings(meta, n as i32, IndexOptions::DocsAndFreqs, false)
        .unwrap();
    assert_eq!(postings.docs, (0..n as i32).collect::<Vec<_>>());
    assert!(postings.freqs.iter().all(|&f| f == 2));
    assert_eq!(postings.freqs.len(), n);
}

#[test]
fn read_postings_rejects_singleton_doc_freq() {
    let id = [4u8; ID_LENGTH];
    let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
    doc.extend_from_slice(&footer);
    let input = DocInput::open(&doc, &id, "").unwrap();
    let meta = TermMetadata {
        doc_start_fp: 0,
        singleton_doc_id: 7,
        ..TermMetadata::EMPTY
    };
    let err = input
        .read_postings(meta, 1, IndexOptions::Docs, false)
        .unwrap_err();
    assert!(matches!(err, Error::Unsupported(_)));
}

/// Writes a level-1 skip entry for `IndexOptions::Docs` (no freqs): just
/// the vInt doc-delta and the vLong `doc_end_fp` delta, which (measured
/// from right after the vLong) must equal `span.len()` so the entry points
/// exactly past its own span. See [`read_level1_entry`].
fn write_level1_entry_docs(doc: &mut Vec<u8>, doc_delta: i32, span: &[u8]) {
    doc.write_vint(doc_delta);
    doc.write_vlong(span.len() as i64);
}

/// Writes a level-1 skip entry for a field with freqs but no positions
/// (`IndexOptions::DocsAndFreqs`), with a real (possibly non-empty)
/// impacts byte run — the freq-gated counterpart to
/// [`write_level1_entry_docs`], needed to exercise `PostingsCursor::
/// level1_impacts`/`Postings::level1_impacts` with real level-1 impacts
/// data. See [`read_level1_entry`] for the exact field layout this
/// mirrors.
fn write_level1_entry_with_impacts(
    doc: &mut Vec<u8>,
    doc_delta: i32,
    span: &[u8],
    impacts: &[Impact],
) {
    doc.write_vint(doc_delta);
    doc.write_vlong(span.len() as i64);
    let mut impact_bytes = Vec::new();
    write_impacts(&mut impact_bytes, impacts);
    // `skip1EndFP` is measured from right after the short that carries it
    // (i.e. from the start of `numImpactBytes`) through the end of this
    // entry's freq-gated metadata: here that's just `numImpactBytes` (2
    // bytes) plus the impact bytes themselves (no pos/pay sub-fields,
    // since this helper is scoped to DocsAndFreqs).
    doc.write_i16((2 + impact_bytes.len()) as i16);
    doc.write_i16(impact_bytes.len() as i16);
    doc.write_bytes(&impact_bytes);
}

#[test]
fn read_postings_level1_span_plus_tail() {
    // docFreq == LEVEL1_NUM_DOCS + 8 (8200): one level-1 entry, then a
    // span of 32 all-consecutive full blocks (docs 0..8191), then an
    // 8-doc group-varint tail (docs 8192..8199) with no more level-1
    // entries.
    let id = [5u8; ID_LENGTH];
    let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
    let doc_start_fp = doc.len() as u64;

    let mut span = Vec::new();
    for _ in 0..LEVEL1_FACTOR {
        write_full_block(&mut span, false, 0);
    }
    // doc_delta 8192 -> the span's last doc is -1 + 8192 = 8191.
    write_level1_entry_docs(&mut doc, LEVEL1_NUM_DOCS, &span);
    doc.extend_from_slice(&span);
    // Tail: 8 consecutive docs (deltas all 1) from prevDocID 8191.
    doc.write_group_vints(&[1; 8]);
    doc.extend_from_slice(&footer);

    let input = DocInput::open(&doc, &id, "").unwrap();
    let meta = TermMetadata {
        doc_start_fp,
        singleton_doc_id: -1,
        ..TermMetadata::EMPTY
    };
    let postings = input
        .read_postings(meta, LEVEL1_NUM_DOCS + 8, IndexOptions::Docs, false)
        .unwrap();
    assert_eq!(postings.docs, (0..LEVEL1_NUM_DOCS + 8).collect::<Vec<_>>());
    assert!(postings.freqs.iter().all(|&f| f == 1));
}

#[test]
fn postings_cursor_level0_impacts_varies_within_block_and_across_blocks() {
    // docFreq == 2 * BLOCK_SIZE (512): two full blocks, each with its own
    // distinct (freq, norm) impacts list -- mirrors the "impacts varying
    // within a level-0 block" real-Lucene case
    // (`big_field_impacts_match_real_lucene_impacts_enum`), but hand-built
    // so `PostingsCursor::level0_impacts`/`Postings::level0_impacts` are
    // exercised directly against known-good expected values rather than
    // only cross-checked against the lazy path or real Lucene bytes.
    let id = [20u8; ID_LENGTH];
    let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
    let doc_start_fp = doc.len() as u64;

    let block0_impacts = vec![Impact { freq: 1, norm: 1 }, Impact { freq: 5, norm: 10 }];
    let block1_impacts = vec![Impact { freq: 2, norm: 3 }];
    write_full_block_with_impacts(&mut doc, true, 3, &block0_impacts);
    write_full_block_with_impacts(&mut doc, true, 7, &block1_impacts);
    doc.extend_from_slice(&footer);

    let input = DocInput::open(&doc, &id, "").unwrap();
    let meta = TermMetadata {
        doc_start_fp,
        singleton_doc_id: -1,
        ..TermMetadata::EMPTY
    };
    let postings = input
        .read_postings(meta, 2 * BLOCK_SIZE, IndexOptions::DocsAndFreqs, false)
        .unwrap();
    assert_eq!(
        postings.level0_impacts,
        vec![
            (BLOCK_SIZE - 1, block0_impacts.clone()),
            (2 * BLOCK_SIZE - 1, block1_impacts.clone()),
        ]
    );

    let mut cursor = PostingsCursor::new(&postings);
    assert_eq!(cursor.advance(0), 0);
    assert_eq!(cursor.level0_impacts(), block0_impacts.as_slice());
    assert!(cursor.level1_impacts().is_empty());

    // Exact last doc of block 0 -- the boundary value a `<` vs `<=` bug
    // in `find_impacts`'s binary search would only surface at.
    assert_eq!(cursor.advance(BLOCK_SIZE - 1), BLOCK_SIZE - 1);
    assert_eq!(cursor.level0_impacts(), block0_impacts.as_slice());

    assert_eq!(cursor.advance(BLOCK_SIZE), BLOCK_SIZE);
    assert_eq!(cursor.level0_impacts(), block1_impacts.as_slice());
    assert!(cursor.level1_impacts().is_empty());
}

#[test]
fn postings_cursor_level1_impacts_reachable_across_span() {
    // docFreq == LEVEL1_NUM_DOCS + 8 (8200): one level-1 entry carrying a
    // real (non-empty) merged span impacts list, a span of 32 full blocks
    // (each with its own distinct level-0 impacts), then an 8-doc tail --
    // mirrors "l1_field_impacts_match_real_lucene_impacts_enum" but hand-
    // built so both `level0_impacts()` (varying block-to-block within the
    // span) and `level1_impacts()` (constant across the whole span, empty
    // once past it in the tail) are checked against known-good values.
    let id = [21u8; ID_LENGTH];
    let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
    let doc_start_fp = doc.len() as u64;

    let first_block_impacts = vec![Impact { freq: 1, norm: 1 }];
    let last_block_impacts = vec![Impact { freq: 4, norm: 9 }];
    let span_impacts = vec![Impact { freq: 4, norm: 9 }, Impact { freq: 8, norm: 20 }];

    let mut span = Vec::new();
    write_full_block_with_impacts(&mut span, true, 1, &first_block_impacts);
    for _ in 1..LEVEL1_FACTOR - 1 {
        write_full_block_with_impacts(&mut span, true, 1, &[]);
    }
    write_full_block_with_impacts(&mut span, true, 1, &last_block_impacts);

    write_level1_entry_with_impacts(&mut doc, LEVEL1_NUM_DOCS, &span, &span_impacts);
    doc.extend_from_slice(&span);
    // Tail: 8 consecutive docs (deltas all 1) from prevDocID 8191, no
    // freq exceptions (freq==1 bit path).
    doc.write_group_vints(&[(1 << 1) | 1; 8]);
    doc.extend_from_slice(&footer);

    let input = DocInput::open(&doc, &id, "").unwrap();
    let meta = TermMetadata {
        doc_start_fp,
        singleton_doc_id: -1,
        ..TermMetadata::EMPTY
    };
    let postings = input
        .read_postings(meta, LEVEL1_NUM_DOCS + 8, IndexOptions::DocsAndFreqs, false)
        .unwrap();
    assert_eq!(
        postings.level1_impacts,
        vec![(LEVEL1_NUM_DOCS - 1, span_impacts.clone())]
    );

    let mut cursor = PostingsCursor::new(&postings);

    // First doc of the span: level-0 impacts are the first block's own,
    // level-1 impacts are the whole span's merged list.
    assert_eq!(cursor.advance(0), 0);
    assert_eq!(cursor.level0_impacts(), first_block_impacts.as_slice());
    assert_eq!(cursor.level1_impacts(), span_impacts.as_slice());

    // Last full block of the span: level-0 impacts change, level-1
    // impacts stay the same (whole-span merge, not per-block).
    assert_eq!(
        cursor.advance(LEVEL1_NUM_DOCS - BLOCK_SIZE),
        LEVEL1_NUM_DOCS - BLOCK_SIZE
    );
    assert_eq!(cursor.level0_impacts(), last_block_impacts.as_slice());
    assert_eq!(cursor.level1_impacts(), span_impacts.as_slice());

    // Exact last doc of the entire span -- the boundary value a `<` vs
    // `<=` bug in `find_impacts` would only surface at, distinct from
    // the "first doc of the last block" check just above.
    assert_eq!(cursor.advance(LEVEL1_NUM_DOCS - 1), LEVEL1_NUM_DOCS - 1);
    assert_eq!(cursor.level0_impacts(), last_block_impacts.as_slice());
    assert_eq!(cursor.level1_impacts(), span_impacts.as_slice());

    // Into the trailing tail (past the level-1 span entirely): both are
    // empty, same as `LazyDocsCursor`'s contract for the tail block.
    assert_eq!(cursor.advance(LEVEL1_NUM_DOCS), LEVEL1_NUM_DOCS);
    assert!(cursor.level0_impacts().is_empty());
    assert!(cursor.level1_impacts().is_empty());
}

#[test]
fn read_level1_entry_decodes_all_fields() {
    // A DocsAndFreqsAndPositions(+payloads) level-1 entry, hand-built, to
    // confirm every field is read in the right order and doc_delta/
    // doc_end_fp come out exactly right (and the internal skip1EndFP
    // consistency check holds -- it's a debug_assert that fires in tests).
    let mut bytes = Vec::new();
    bytes.write_vint(8191); // doc_delta
    bytes.write_vlong(100); // doc_end_fp delta
    let pos_after_vlong = bytes.len();
    // freq metadata: skip1EndFP short, then numImpactBytes short, impacts,
    // then pos (vlong+byte) and pay (vlong+vint) sub-fields.
    // metadata after short1 = short2 (2) + 3 impact bytes + pos(vlong 1 +
    // byte 1) + pay(vlong 1 + vint 1) = 2 + 3 + 4 = 9.
    bytes.write_i16(9); // skip1EndFP offset
    bytes.write_i16(3); // numImpactBytes
                        // 3 valid impact bytes: freqDelta=0, normDelta=0 (implicit +1 each)
                        // three times over, decoding to (freq=1,norm=1), (2,2), (3,3).
    bytes.write_bytes(&[0x00, 0x00, 0x00]);
    bytes.write_vlong(50); // posEndFP delta
    bytes.write_byte(7); // posBufferUpto
    bytes.write_vlong(60); // payEndFP delta
    bytes.write_vint(9); // payloadByteUpto (parsed, then recomputed -- see
                         // `read_pos_skip`)
    let metadata_end = bytes.len();

    let mut r = SliceInput::new(&bytes);
    let entry = read_level1_entry(&mut r, true, true, true).unwrap();
    assert_eq!(entry.doc_delta, 8191);
    assert_eq!(entry.doc_end_fp, 100 + pos_after_vlong);
    // r is left at the start of the span's first level-0 block header.
    assert_eq!(r.position(), metadata_end);
    assert_eq!(
        decode_impacts(entry.impact_bytes).unwrap(),
        vec![
            Impact { freq: 1, norm: 1 },
            Impact { freq: 2, norm: 2 },
            Impact { freq: 3, norm: 3 },
        ]
    );
    // The `.pos`/`.pay` skip pointers are retained now, not discarded.
    let skip = entry.pos_skip.expect("a positions-indexing field has them");
    assert_eq!(skip.pos_end_fp_delta, 50);
    assert_eq!(skip.pos_buffer_upto, 7);
    assert_eq!(skip.pay_end_fp_delta, 60);
}

#[test]
fn read_level1_entry_docs_only_reads_just_delta_fields() {
    // IndexOptions::Docs: no freq gate, so only the vInt doc-delta and
    // vLong doc_end_fp delta are present -- no impacts/pos/pay.
    let mut bytes = Vec::new();
    bytes.write_vint(500);
    bytes.write_vlong(40);
    let pos_after_vlong = bytes.len();
    bytes.write_bytes(&[0x11, 0x22]); // "span" bytes that must NOT be read
    let mut r = SliceInput::new(&bytes);
    let entry = read_level1_entry(&mut r, false, false, false).unwrap();
    assert_eq!(entry.doc_delta, 500);
    assert_eq!(entry.doc_end_fp, 40 + pos_after_vlong);
    assert_eq!(r.position(), pos_after_vlong);
}

/// Test-only encoder for a full 256-doc block's level-0 header +
/// doc-delta/freq payload (`Lucene104PostingsWriter.flushDocBlock`'s
/// `else` branch), specialized to the `bitsPerValue == 0`
/// ("all consecutive") doc-delta encoding and (optionally) the
/// `PForUtil` all-equal fast path for freqs, since those need no
/// `ForUtil`/`PForUtil` packed body to hand-construct — the
/// lane-interleaved bit-packed paths are exercised by the
/// `for_util` module's own tests and by the `GenBlockTree.java`
/// differential fixture (real `IndexWriter` bytes).
///
/// `docDelta` and `blockLength` are real, consistent header fields here
/// (not filler) — [`LazyDocsCursor`]'s skip-ahead relies on them being
/// accurate, unlike the pre-lazy-cursor version of this helper, which
/// only needed `read_full_block_header`'s wire-order-only decode to work.
/// `docDelta` is always `BLOCK_SIZE` (256), matching the "all 256 deltas
/// are 1" body this helper always writes.
fn write_full_block(out: &mut Vec<u8>, index_has_freq: bool, freq_value: i32) {
    write_full_block_with_impacts(out, index_has_freq, freq_value, &[]);
}

/// Same as [`write_full_block`], but with a real (possibly non-empty)
/// level-0 impacts byte run instead of always writing an empty one —
/// needed to exercise `PostingsCursor::level0_impacts`/`Postings::
/// level0_impacts` with real per-block impacts data rather than a
/// perpetually-empty section.
fn write_full_block_with_impacts(
    out: &mut Vec<u8>,
    index_has_freq: bool,
    freq_value: i32,
    impacts: &[Impact],
) {
    // The metadata region (impacts, and the pos/pay skip fields this
    // helper's fields never have) is measured separately from the block
    // body, because `level0NumBytes` spans the former plus the two header
    // fields while `blockLength` spans the former plus the latter.
    let mut meta_region = Vec::new();
    if index_has_freq {
        let mut impact_bytes = Vec::new();
        write_impacts(&mut impact_bytes, impacts);
        // Impacts byte-length is a plain vint on the wire here
        // (`read_full_block_header`'s doc comment) — `write_vint` matches
        // that (not `write_vlong`, even though the two happen to agree
        // byte-for-byte for any value small enough to fit both).
        meta_region.write_vint(impact_bytes.len() as i32);
        meta_region.write_bytes(&impact_bytes);
    }
    let mut block_body = Vec::new();
    block_body.write_byte(0); // bitsPerValue == 0: all 256 deltas are 1.
    if index_has_freq {
        block_body.write_byte(0); // PForUtil token: bitsPerValue=0, numExceptions=0
        block_body.write_vint(freq_value);
    }

    // `blockLength` is measured from right after this field (i.e. from
    // right here) through the end of the whole block -- see
    // `read_full_block_header`'s doc comment.
    //
    // `level0NumBytes` (`numSkipBytes`) is measured from right after
    // *itself* through the start of the block body, i.e. the two
    // fixed-width header fields plus the metadata region --
    // `Lucene104PostingsWriter.flushDocBlock`'s `numSkipBytes =
    // level0Output.size()` sampled after the impacts/pos/pay region, then
    // `+= scratchOutput.size()`. `body.len()` is neither, and
    // `read_full_block_header` checks the two agree.
    let header_fields_len = 4; // vint15 docDelta + vlong15 blockLength
    out.write_vlong((meta_region.len() + header_fields_len) as i64);
    out.write_i16(BLOCK_SIZE as i16); // docDelta via writeVInt15
    out.write_i16((meta_region.len() + block_body.len()) as i16); // blockLength via writeVLong15
    out.write_bytes(&meta_region);
    out.write_bytes(&block_body);
}

#[test]
fn read_postings_exactly_one_full_block_no_tail() {
    // docFreq == BLOCK_SIZE (256): one full block, no tail bytes at all.
    let id = [8u8; ID_LENGTH];
    let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
    let doc_start_fp = doc.len() as u64;
    write_full_block(&mut doc, true, 3);
    doc.extend_from_slice(&footer);

    let input = DocInput::open(&doc, &id, "").unwrap();
    let meta = TermMetadata {
        doc_start_fp,
        singleton_doc_id: -1,
        ..TermMetadata::EMPTY
    };
    let postings = input
        .read_postings(meta, BLOCK_SIZE, IndexOptions::DocsAndFreqs, false)
        .unwrap();
    assert_eq!(postings.docs, (0..BLOCK_SIZE).collect::<Vec<_>>());
    assert!(postings.freqs.iter().all(|&f| f == 3));
    assert_eq!(postings.docs.len(), BLOCK_SIZE as usize);
}

#[test]
fn read_postings_one_full_block_plus_one_doc_tail() {
    // docFreq == BLOCK_SIZE + 1 (257): one full block (docs 0..256) then
    // a 1-doc group-varint tail block, prevDocID chained from the full
    // block's last doc (255).
    let id = [9u8; ID_LENGTH];
    let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
    let doc_start_fp = doc.len() as u64;
    write_full_block(&mut doc, true, 3);
    // Tail: 1 doc, delta=5 from prevDocID=255 -> docID 260, freq=7 (bit
    // clear, explicit freq vint follows).
    doc.write_group_vints(&[5 << 1]);
    doc.write_vint(7);
    doc.extend_from_slice(&footer);

    let input = DocInput::open(&doc, &id, "").unwrap();
    let meta = TermMetadata {
        doc_start_fp,
        singleton_doc_id: -1,
        ..TermMetadata::EMPTY
    };
    let postings = input
        .read_postings(meta, BLOCK_SIZE + 1, IndexOptions::DocsAndFreqs, false)
        .unwrap();
    assert_eq!(postings.docs.len(), BLOCK_SIZE as usize + 1);
    assert_eq!(
        &postings.docs[..BLOCK_SIZE as usize],
        &(0..BLOCK_SIZE).collect::<Vec<_>>()[..]
    );
    assert_eq!(postings.docs[BLOCK_SIZE as usize], 260);
    assert_eq!(postings.freqs[BLOCK_SIZE as usize], 7);
    assert!(postings.freqs[..BLOCK_SIZE as usize]
        .iter()
        .all(|&f| f == 3));
}

#[test]
fn read_postings_multi_block_docs_only_no_freqs() {
    // IndexOptions::Docs (no freqs): full block omits impacts and the
    // PForUtil freq block entirely.
    let id = [10u8; ID_LENGTH];
    let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
    let doc_start_fp = doc.len() as u64;
    write_full_block(&mut doc, false, 0);
    doc.extend_from_slice(&footer);

    let input = DocInput::open(&doc, &id, "").unwrap();
    let meta = TermMetadata {
        doc_start_fp,
        singleton_doc_id: -1,
        ..TermMetadata::EMPTY
    };
    let postings = input
        .read_postings(meta, BLOCK_SIZE, IndexOptions::Docs, false)
        .unwrap();
    assert_eq!(postings.docs, (0..BLOCK_SIZE).collect::<Vec<_>>());
    assert!(postings.freqs.iter().all(|&f| f == 1));
}

#[test]
fn read_full_block_bitset_encoding_decodes_dense_docs() {
    // bitsPerValue < 0: dense unary bit-set doc-delta encoding. 8 words
    // of 0x5555...5 (every even bit set, 32 per word) give exactly
    // BLOCK_SIZE (256) set bits at positions 0,2,4,...,510 -- docIDs
    // 0,2,...,510 since prevDocID=-1 puts docBitSetBase at 0. This is
    // the same branch real Lucene picks for the `big`/"everywhere"
    // fixture term (see `blocktree_fixtures.rs`).
    let id = [11u8; ID_LENGTH];
    let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
    let doc_start_fp = doc.len() as u64;
    let mut body = Vec::new();
    let num_longs = 8u8;
    body.write_byte((-(num_longs as i8)) as u8);
    for _ in 0..num_longs {
        body.write_bytes(&0x5555_5555_5555_5555u64.to_le_bytes());
    }
    doc.write_vlong(4); // level0NumBytes: the two header fields, no metadata region
    doc.write_i16(1);
    doc.write_i16(body.len() as i16);
    doc.write_bytes(&body);
    doc.extend_from_slice(&footer);

    let input = DocInput::open(&doc, &id, "").unwrap();
    let meta = TermMetadata {
        doc_start_fp,
        singleton_doc_id: -1,
        ..TermMetadata::EMPTY
    };
    let postings = input
        .read_postings(meta, BLOCK_SIZE, IndexOptions::Docs, false)
        .unwrap();
    let expected: Vec<i32> = (0..BLOCK_SIZE).map(|i| i * 2).collect();
    assert_eq!(postings.docs, expected);
    assert!(postings.freqs.iter().all(|&f| f == 1));
}

#[test]
fn read_full_block_bitset_encoding_rejects_a_set_bit_count_other_than_block_size() {
    // A corrupted bit set with fewer than BLOCK_SIZE set bits must be a
    // decode error, not a silently short postings list -- and one with
    // more, which a kept bit set would otherwise answer from (Lucene
    // asserts exactly BLOCK_SIZE).
    // 4 longs = 256 bits, none set; 5 longs, all 320 set.
    for (num_longs, fill) in [(4usize, 0u8), (5, 0xFF)] {
        let id = [12u8; ID_LENGTH];
        let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
        let doc_start_fp = doc.len() as u64;
        let mut body = Vec::new();
        body.write_byte((-(num_longs as i8)) as u8);
        body.extend_from_slice(&vec![fill; num_longs * 8]);
        doc.write_vlong(4); // level0NumBytes: the two header fields, no metadata region
        doc.write_i16(1);
        doc.write_i16(body.len() as i16);
        doc.write_bytes(&body);
        doc.extend_from_slice(&footer);

        let input = DocInput::open(&doc, &id, "").unwrap();
        let meta = TermMetadata {
            doc_start_fp,
            singleton_doc_id: -1,
            ..TermMetadata::EMPTY
        };
        let err = input
            .read_postings(meta, BLOCK_SIZE, IndexOptions::Docs, false)
            .unwrap_err();
        assert!(
            matches!(err, Error::Store(lucene_store::Error::Corrupted(_))),
            "{num_longs} words of {fill:#x}: {err:?}"
        );
    }
}

#[test]
fn read_full_block_packed_encoding_decodes_bit_packed_deltas() {
    // bitsPerValue > 0: the `for_util::for_decode` packed-delta branch,
    // the encoding real Lucene picks when the doc IDs are neither fully
    // consecutive nor dense enough for the bit-set path (see the
    // `bitsPerValue < 0` test above). Deltas alternate 1/3 (needs 2 bits
    // per value), encoded via `for_util`'s own test-only encoder so this
    // exercises the exact same wire format `for_decode` expects.
    let id = [13u8; ID_LENGTH];
    let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
    let doc_start_fp = doc.len() as u64;

    let mut deltas = [0u32; for_util::BLOCK_SIZE];
    for (i, d) in deltas.iter_mut().enumerate() {
        *d = if i % 2 == 0 { 1 } else { 3 };
    }
    let mut source = deltas;
    let bits_per_value = 2u32;
    let mut packed = Vec::new();
    // `for_encode` collapses lanes in place (as Java's `ForUtil.encode`
    // does), so hand it a scratch copy -- `deltas` is the expectation below.
    for_util::for_encode(&mut source, bits_per_value, &mut packed);

    let mut body = Vec::new();
    body.write_byte(bits_per_value as u8);
    body.write_bytes(&packed);
    doc.write_vlong(4); // level0NumBytes: the two header fields, no metadata region
    doc.write_i16(1);
    doc.write_i16(body.len() as i16);
    doc.write_bytes(&body);
    doc.extend_from_slice(&footer);

    let input = DocInput::open(&doc, &id, "").unwrap();
    let meta = TermMetadata {
        doc_start_fp,
        singleton_doc_id: -1,
        ..TermMetadata::EMPTY
    };
    let postings = input
        .read_postings(meta, BLOCK_SIZE, IndexOptions::Docs, false)
        .unwrap();
    let mut expected = Vec::with_capacity(BLOCK_SIZE as usize);
    let mut sum = -1i32;
    for &d in &deltas {
        sum += d as i32;
        expected.push(sum);
    }
    assert_eq!(postings.docs, expected);
}

#[test]
fn singleton_postings_reconstructs_from_metadata() {
    let meta = TermMetadata {
        doc_start_fp: 123,
        singleton_doc_id: 9,
        ..TermMetadata::EMPTY
    };
    let postings = singleton_postings(meta, 4).unwrap();
    assert_eq!(postings.docs, vec![9]);
    assert_eq!(postings.freqs, vec![4]);
}

#[test]
fn singleton_postings_rejects_non_singleton_metadata() {
    let meta = TermMetadata {
        doc_start_fp: 0,
        singleton_doc_id: -1,
        ..TermMetadata::EMPTY
    };
    assert!(singleton_postings(meta, 1).is_err());
}

fn postings(docs: &[i32], freqs: &[i32]) -> Postings {
    Postings {
        docs: docs.to_vec(),
        freqs: freqs.to_vec(),
        level0_impacts: Vec::new(),
        level1_impacts: Vec::new(),
    }
}

#[test]
fn cursor_starts_before_first_doc() {
    let p = postings(&[2, 5, 9], &[1, 1, 1]);
    let cursor = PostingsCursor::new(&p);
    assert_eq!(cursor.doc_id(), -1);
    assert_eq!(cursor.freq(), None);
}

#[test]
fn cursor_next_doc_walks_in_order() {
    let p = postings(&[2, 5, 9], &[3, 4, 5]);
    let mut cursor = PostingsCursor::new(&p);
    assert_eq!(cursor.next_doc(), 2);
    assert_eq!(cursor.freq(), Some(3));
    assert_eq!(cursor.next_doc(), 5);
    assert_eq!(cursor.freq(), Some(4));
    assert_eq!(cursor.next_doc(), 9);
    assert_eq!(cursor.freq(), Some(5));
    assert_eq!(cursor.next_doc(), NO_MORE_DOCS);
    assert_eq!(cursor.freq(), None);
    // Calling next_doc() again once exhausted stays exhausted (idempotent).
    assert_eq!(cursor.next_doc(), NO_MORE_DOCS);
}

#[test]
fn cursor_advance_before_first_doc_lands_on_first() {
    let p = postings(&[2, 5, 9], &[1, 1, 1]);
    let mut cursor = PostingsCursor::new(&p);
    // target 0 is before the first doc (2): should land on 2.
    assert_eq!(cursor.advance(0), 2);
    assert_eq!(cursor.freq(), Some(1));
}

#[test]
fn cursor_advance_exact_match() {
    let p = postings(&[2, 5, 9], &[1, 2, 3]);
    let mut cursor = PostingsCursor::new(&p);
    assert_eq!(cursor.advance(5), 5);
    assert_eq!(cursor.freq(), Some(2));
}

#[test]
fn cursor_advance_between_docs_lands_on_next_higher() {
    let p = postings(&[2, 5, 9], &[1, 1, 1]);
    let mut cursor = PostingsCursor::new(&p);
    // target 6 is between 5 and 9: should land on 9.
    assert_eq!(cursor.advance(6), 9);
}

#[test]
fn cursor_advance_past_last_doc_exhausts() {
    let p = postings(&[2, 5, 9], &[1, 1, 1]);
    let mut cursor = PostingsCursor::new(&p);
    assert_eq!(cursor.advance(100), NO_MORE_DOCS);
    assert_eq!(cursor.freq(), None);
    // Once exhausted, further advances stay exhausted.
    assert_eq!(cursor.advance(200), NO_MORE_DOCS);
}

#[test]
fn cursor_advance_on_empty_postings() {
    let p = postings(&[], &[]);
    let mut cursor = PostingsCursor::new(&p);
    assert_eq!(cursor.doc_id(), -1);
    assert_eq!(cursor.advance(0), NO_MORE_DOCS);
    assert_eq!(cursor.freq(), None);
}

#[test]
fn cursor_advance_to_doc_before_current_is_a_documented_no_op() {
    // advance() to a target <= the current doc ID does not rewind: it
    // is a documented no-op (binary search never looks backward from
    // the cursor's current index) rather than an error.
    let p = postings(&[2, 5, 9, 20], &[1, 1, 1, 1]);
    let mut cursor = PostingsCursor::new(&p);
    assert_eq!(cursor.advance(9), 9);
    assert_eq!(cursor.advance(5), 9, "no-op: target is behind current doc");
    assert_eq!(cursor.advance(9), 9, "no-op: target equals current doc");
    // Cursor can still move forward normally afterward.
    assert_eq!(cursor.advance(20), 20);
}

#[test]
fn cursor_advance_then_next_doc_continues_from_landed_position() {
    let p = postings(&[2, 5, 9, 20], &[1, 2, 3, 4]);
    let mut cursor = PostingsCursor::new(&p);
    assert_eq!(cursor.advance(6), 9);
    assert_eq!(cursor.next_doc(), 20);
    assert_eq!(cursor.freq(), Some(4));
    assert_eq!(cursor.next_doc(), NO_MORE_DOCS);
}

#[test]
fn cursor_advance_to_no_more_docs_target_exhausts() {
    let p = postings(&[2, 5], &[1, 1]);
    let mut cursor = PostingsCursor::new(&p);
    assert_eq!(cursor.advance(NO_MORE_DOCS), NO_MORE_DOCS);
}

#[test]
fn lazy_cursor_rejects_singleton_doc_freq() {
    let id = [20u8; ID_LENGTH];
    let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
    doc.extend_from_slice(&footer);
    let input = DocInput::open(&doc, &id, "").unwrap();
    let meta = TermMetadata {
        doc_start_fp: 0,
        singleton_doc_id: 7,
        ..TermMetadata::EMPTY
    };
    let err = input
        .lazy_cursor(meta, 1, IndexOptions::Docs, false)
        .unwrap_err();
    assert!(matches!(err, Error::Unsupported(_)));
}

#[test]
fn lazy_cursor_level1_sequential_next_doc_matches_read_postings() {
    // docFreq == LEVEL1_NUM_DOCS + 8 (8200): one level-1 span (32 blocks)
    // + an 8-doc tail. A full `next_doc()` walk through the lazy cursor
    // (which reads the level-1 entry, decodes each of the span's 32
    // blocks on demand, then the tail) must match the eager
    // `read_postings` result byte-for-byte across the span boundary.
    let id = [21u8; ID_LENGTH];
    let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
    let doc_start_fp = doc.len() as u64;

    let mut span = Vec::new();
    for _ in 0..LEVEL1_FACTOR {
        write_full_block(&mut span, false, 0);
    }
    write_level1_entry_docs(&mut doc, LEVEL1_NUM_DOCS, &span);
    doc.extend_from_slice(&span);
    doc.write_group_vints(&[1; 8]);
    doc.extend_from_slice(&footer);

    let input = DocInput::open(&doc, &id, "").unwrap();
    let meta = TermMetadata {
        doc_start_fp,
        singleton_doc_id: -1,
        ..TermMetadata::EMPTY
    };
    let doc_freq = LEVEL1_NUM_DOCS + 8;
    let eager = input
        .read_postings(meta, doc_freq, IndexOptions::Docs, false)
        .unwrap();

    let mut cursor = input
        .lazy_cursor(meta, doc_freq, IndexOptions::Docs, false)
        .unwrap();
    let mut lazy_docs = Vec::new();
    loop {
        let d = cursor.next_doc().unwrap();
        if d == NO_MORE_DOCS {
            break;
        }
        lazy_docs.push(d);
    }
    assert_eq!(lazy_docs, eager.docs);
}

#[test]
fn lazy_cursor_advance_skips_whole_corrupted_level1_span_without_decoding_it() {
    // The level-1 analogue of
    // `lazy_cursor_advance_skips_corrupted_earlier_block_without_decoding_it`:
    // an entire 32-block level-1 span is corrupt (its first block has a
    // valid level-0 frame but a zero-bit bit-set body, the rest is
    // garbage). `advance(target)` to a doc in the *tail* past the whole
    // span must succeed -- proving `skip_level1_to` jumps straight to the
    // level-1 entry's `doc_end_fp` without ever reading a single level-0
    // block header in the span. A control `advance()` into the span still
    // surfaces the corruption, confirming the skip wasn't luck.
    let id = [22u8; ID_LENGTH];
    let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
    let doc_start_fp = doc.len() as u64;

    // The span: block 0 has a valid frame (docDelta=256 -> last doc 255)
    // but a corrupt zero-bit bit-set body; blocks 1..31 are pure garbage
    // neither tested path ever reads (the skip jumps over them via
    // doc_end_fp; the control errors on block 0 before reaching them).
    let mut span = Vec::new();
    let mut body0 = Vec::new();
    body0.write_byte((-4i8) as u8); // 4 longs = 256 bits, none set -> corrupt
    body0.extend_from_slice(&[0u8; 32]);
    span.write_vlong(4); // level0NumBytes: the two header fields, no metadata region
    span.write_i16(BLOCK_SIZE as i16); // docDelta = 256
    span.write_i16(body0.len() as i16); // blockLength
    span.write_bytes(&body0);
    span.extend_from_slice(&[0xFFu8; 64]); // garbage stand-in for blocks 1..31

    // Level-1 entry: doc_delta 8192 -> span last doc 8191; doc_end_fp
    // lands exactly at the tail (span.len() bytes past the vLong).
    write_level1_entry_docs(&mut doc, LEVEL1_NUM_DOCS, &span);
    doc.extend_from_slice(&span);
    // Valid 8-doc tail (docs 8192..8199), chained from prevDocID 8191.
    doc.write_group_vints(&[1; 8]);
    doc.extend_from_slice(&footer);

    let input = DocInput::open(&doc, &id, "").unwrap();
    let meta = TermMetadata {
        doc_start_fp,
        singleton_doc_id: -1,
        ..TermMetadata::EMPTY
    };
    let doc_freq = LEVEL1_NUM_DOCS + 8;

    let mut cursor = input
        .lazy_cursor(meta, doc_freq, IndexOptions::Docs, false)
        .unwrap();
    // Target 8195 is in the tail, past the whole corrupt span.
    assert_eq!(cursor.advance(8195).unwrap(), 8195);
    assert_eq!(cursor.freq(), Some(1));

    // Control: a target inside the span forces decoding block 0's corrupt
    // body, which must surface the corruption error.
    let mut cursor2 = input
        .lazy_cursor(meta, doc_freq, IndexOptions::Docs, false)
        .unwrap();
    let err = cursor2.advance(100).unwrap_err();
    assert!(matches!(
        err,
        Error::Store(lucene_store::Error::Corrupted(_))
    ));
}

#[test]
fn lazy_cursor_sequential_next_doc_matches_read_postings() {
    // Two full blocks (docs 0..256, 256..512) plus a 3-doc tail: proves
    // the lazy per-block decode-on-demand path produces byte-identical
    // results to the eager whole-term `read_postings` across both the
    // full-block/full-block and full-block/tail-block boundaries.
    let id = [22u8; ID_LENGTH];
    let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
    let doc_start_fp = doc.len() as u64;
    write_full_block(&mut doc, true, 3);
    write_full_block(&mut doc, true, 4);
    doc.write_group_vints(&[5 << 1, 1 << 1, 2 << 1]);
    doc.extend_from_slice(&footer);

    let input = DocInput::open(&doc, &id, "").unwrap();
    let meta = TermMetadata {
        doc_start_fp,
        singleton_doc_id: -1,
        ..TermMetadata::EMPTY
    };
    let doc_freq = 2 * BLOCK_SIZE + 3;
    let eager = input
        .read_postings(meta, doc_freq, IndexOptions::DocsAndFreqs, false)
        .unwrap();

    let mut cursor = input
        .lazy_cursor(meta, doc_freq, IndexOptions::DocsAndFreqs, false)
        .unwrap();
    let mut lazy_docs = Vec::new();
    let mut lazy_freqs = Vec::new();
    loop {
        let d = cursor.next_doc().unwrap();
        if d == NO_MORE_DOCS {
            break;
        }
        lazy_docs.push(d);
        lazy_freqs.push(cursor.freq().unwrap());
    }
    assert_eq!(lazy_docs, eager.docs);
    assert_eq!(lazy_freqs, eager.freqs);
}

#[test]
fn a_cursor_without_impacts_reads_the_same_postings_and_no_impacts() {
    // Three full blocks with impacts on the wire and a tail: the
    // no-impacts cursor must step over the headers it skips and still
    // land on the same documents with the same frequencies.
    let id = [23u8; ID_LENGTH];
    let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
    let doc_start_fp = doc.len() as u64;
    write_full_block_with_impacts(&mut doc, true, 3, &[Impact { freq: 3, norm: 1 }]);
    write_full_block_with_impacts(&mut doc, true, 4, &[Impact { freq: 4, norm: 1 }]);
    write_full_block_with_impacts(&mut doc, true, 5, &[Impact { freq: 5, norm: 1 }]);
    doc.write_group_vints(&[5 << 1, 1 << 1, 2 << 1]);
    doc.extend_from_slice(&footer);
    let input = DocInput::open(&doc, &id, "").unwrap();
    let meta = TermMetadata {
        doc_start_fp,
        singleton_doc_id: -1,
        ..TermMetadata::EMPTY
    };
    let doc_freq = 3 * BLOCK_SIZE + 3;
    let open = |flags| {
        input
            .lazy_cursor_with_flags(meta, doc_freq, IndexOptions::DocsAndFreqs, false, flags)
            .unwrap()
    };
    let walk = |mut c: LazyDocsCursor<'_>| {
        let mut out = Vec::new();
        let mut empty_in_block_1 = true;
        for target in [0, 3, 200, 300, 520, 700, 769, 770] {
            let d = c.advance(target).unwrap();
            out.push((
                d,
                if d == NO_MORE_DOCS {
                    0
                } else {
                    c.freq().unwrap()
                },
            ));
            if target == 300 {
                empty_in_block_1 = c.level0_impacts().is_empty();
            }
            if d == NO_MORE_DOCS {
                break;
            }
        }
        (out, empty_in_block_1)
    };
    let (with, with_empty) = walk(open(PostingsFlags::Freqs));
    let (without, without_empty) = walk(open(PostingsFlags::FreqsNoImpacts));
    assert_eq!(with, without);
    assert!(!with_empty, "the impacts cursor decoded a block's impacts");
    assert!(without_empty, "the plain one never does");
}

#[test]
fn lazy_cursor_next_doc_after_advance_shallow_moves_to_the_shallow_block() {
    // Two full blocks (docs 0..255 and 256..511). Walk into block 0, then
    // shallow-advance into block 1 and call `next_doc()`.
    //
    // `advance_shallow` reads block 1's header and impacts but leaves its
    // body unpacked, so `block_docs` still holds *block 0*. Answering
    // `next_doc()` out of that buffer hands back doc 1 -- behind the
    // position the shallow move just established, and described by block
    // 1's impacts rather than its own block's. Lucene gates its `nextDoc`
    // on exactly this (`if (doc == level0LastDocID || needsRefilling)`)
    // and lands on the shallow block's first doc, 256.
    let id = [41u8; ID_LENGTH];
    let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
    let doc_start_fp = doc.len() as u64;
    write_full_block(&mut doc, false, 0);
    write_full_block(&mut doc, false, 0);
    doc.extend_from_slice(&footer);

    let input = DocInput::open(&doc, &id, "").unwrap();
    let meta = TermMetadata {
        doc_start_fp,
        singleton_doc_id: -1,
        ..TermMetadata::EMPTY
    };
    let mut cursor = input
        .lazy_cursor(meta, 2 * BLOCK_SIZE, IndexOptions::Docs, false)
        .unwrap();

    assert_eq!(cursor.next_doc().unwrap(), 0);
    assert_eq!(cursor.advance_shallow(300).unwrap(), 2 * BLOCK_SIZE - 1);
    assert_eq!(
        cursor.next_doc().unwrap(),
        BLOCK_SIZE,
        "next_doc must materialize the shallow-positioned block, not answer \
             from the stale previous one"
    );
    // And the walk continues from there, still in order.
    assert_eq!(cursor.next_doc().unwrap(), BLOCK_SIZE + 1);
}

#[test]
fn lazy_cursor_advance_to_no_more_docs_ends_from_any_position() {
    // Two full blocks: end from inside a decoded block, from a block that
    // a shallow move left undecoded, and before the first document.
    let id = [42u8; ID_LENGTH];
    let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
    let doc_start_fp = doc.len() as u64;
    write_full_block(&mut doc, false, 0);
    write_full_block(&mut doc, false, 0);
    doc.extend_from_slice(&footer);
    let input = DocInput::open(&doc, &id, "").unwrap();
    let meta = TermMetadata {
        doc_start_fp,
        singleton_doc_id: -1,
        ..TermMetadata::EMPTY
    };
    let open = || {
        input
            .lazy_cursor(meta, 2 * BLOCK_SIZE, IndexOptions::Docs, false)
            .unwrap()
    };
    let ends = |mut cursor: LazyDocsCursor<'_>| {
        assert_eq!(cursor.advance(NO_MORE_DOCS).unwrap(), NO_MORE_DOCS);
        assert_eq!(cursor.doc_id(), NO_MORE_DOCS);
        assert_eq!(cursor.next_doc().unwrap(), NO_MORE_DOCS);
        assert_eq!(cursor.advance(7).unwrap(), NO_MORE_DOCS);
        assert_eq!(cursor.advance_shallow(7).unwrap(), NO_MORE_DOCS);
    };
    let mut decoded = open();
    assert_eq!(decoded.next_doc().unwrap(), 0);
    ends(decoded);
    let mut shallow = open();
    assert_eq!(shallow.next_doc().unwrap(), 0);
    assert_eq!(shallow.advance_shallow(300).unwrap(), 2 * BLOCK_SIZE - 1);
    ends(shallow);
    ends(open());
}

#[test]
fn a_level1_entry_that_disagrees_with_its_own_skip1_end_fp_is_rejected() {
    // The level-1 entry's `skip1EndFP` short is the byte length of that
    // entry's own metadata, read off the same file as the metadata it
    // measures. Java only asserts it; here a corrupt `.doc` must be a decode
    // error, not a panic. Build a valid
    // `docFreq >= LEVEL1_NUM_DOCS` term's first level-1 entry, then
    // corrupt only that short.
    let id = [42u8; ID_LENGTH];
    let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
    let doc_start_fp = doc.len() as u64;

    // Level-1 entry: docDelta, span length, then (freqs) skip1EndFP,
    // numImpactBytes, impact bytes.
    doc.write_vint(1000); // docDelta
    doc.write_vlong(0); // span byte length (unused by this path)
    let skip1_pos = doc.len();
    doc.write_i16(0); // skip1EndFP placeholder -- patched below
    doc.write_i16(0); // numImpactBytes = 0
    let after = doc.len();
    doc.extend_from_slice(&footer);
    // Honest value first: from just after the skip1EndFP short to `after`.
    let honest = (after - (skip1_pos + 2)) as i16;
    doc[skip1_pos..skip1_pos + 2].copy_from_slice(&honest.to_le_bytes());

    let input = DocInput::open(&doc, &id, "").unwrap();
    let meta = TermMetadata {
        doc_start_fp,
        singleton_doc_id: -1,
        ..TermMetadata::EMPTY
    };
    // Sanity: with the honest length the entry parses (it fails later, on
    // the blocks that are not there, but not on this check).
    let honest_err = input
        .read_postings(meta, LEVEL1_NUM_DOCS, IndexOptions::DocsAndFreqs, false)
        .unwrap_err();
    assert!(
        !format!("{honest_err}").contains("level-1 skip entry"),
        "the honest entry must not trip the skip1EndFP check: {honest_err}"
    );

    // Now lie about it by one byte.
    doc[skip1_pos..skip1_pos + 2].copy_from_slice(&(honest + 1).to_le_bytes());
    let input = DocInput::open(&doc, &id, "").unwrap();
    let err = input
        .read_postings(meta, LEVEL1_NUM_DOCS, IndexOptions::DocsAndFreqs, false)
        .unwrap_err();
    assert!(
        matches!(err, Error::Store(lucene_store::Error::Corrupted(_)))
            && format!("{err}").contains("level-1 skip entry"),
        "expected a Corrupted decode error naming the level-1 entry, got {err:?}"
    );
}

/// A length read off disk must never become an unchecked `as usize`.
///
/// `numImpactBytes` in a level-1 entry is a signed `readShort`; a corrupt
/// `.doc` can put a negative there, and `negative as usize` sign-extends
/// to ~2^64, so the `impact_start + len` that follows overflows -- which
/// in a debug build is a panic, and a panic in a debug build of the FFI
/// takes the JVM down. Java throws instead. Same class as the
/// lazy-cursor tail block.
#[test]
fn a_level1_entry_with_a_negative_impacts_length_is_rejected() {
    let id = [70u8; ID_LENGTH];
    let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
    let doc_start_fp = doc.len() as u64;
    doc.write_vint(1000); // docDelta
    doc.write_vlong(0); // span byte length
    doc.write_i16(0); // skip1EndFP (never reached)
    doc.write_i16(-1); // numImpactBytes: impossible
    doc.extend_from_slice(&footer);

    let input = DocInput::open(&doc, &id, "").unwrap();
    let meta = TermMetadata {
        doc_start_fp,
        singleton_doc_id: -1,
        ..TermMetadata::EMPTY
    };
    let err = input
        .read_postings(meta, LEVEL1_NUM_DOCS, IndexOptions::DocsAndFreqs, false)
        .unwrap_err();
    assert!(
        matches!(err, Error::Store(lucene_store::Error::Corrupted(_)))
            && format!("{err}").contains("level-1 impacts"),
        "expected a Corrupted decode error naming the impacts length, got {err:?}"
    );
}

/// The level-0 header's impacts length is a `vint`, which decodes to a
/// negative from a five-byte varint with the sign bit set -- the same
/// overflow-into-panic shape as the level-1 short above.
#[test]
fn a_level0_header_with_a_negative_impacts_length_is_rejected() {
    let id = [71u8; ID_LENGTH];
    let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
    let doc_start_fp = doc.len() as u64;
    doc.write_vlong(4); // level0NumBytes
    doc.write_i16(256); // docDelta
    doc.write_i16(4); // blockLength
    doc.write_vint(-1); // impacts byte length: impossible
    doc.extend_from_slice(&footer);

    let input = DocInput::open(&doc, &id, "").unwrap();
    let meta = TermMetadata {
        doc_start_fp,
        singleton_doc_id: -1,
        ..TermMetadata::EMPTY
    };
    let err = input
        .read_postings(meta, BLOCK_SIZE, IndexOptions::DocsAndFreqs, false)
        .unwrap_err();
    assert!(
        matches!(err, Error::Store(lucene_store::Error::Corrupted(_))),
        "expected a Corrupted decode error, got {err:?}"
    );
}

/// `blockLength` is a `vlong15`: a corrupt one can be negative, and
/// `position() + (negative as usize)` overflows.
#[test]
fn a_level0_header_with_a_negative_block_length_is_rejected() {
    let id = [72u8; ID_LENGTH];
    let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
    let doc_start_fp = doc.len() as u64;
    doc.write_vlong(4); // level0NumBytes
    doc.write_i16(256); // docDelta
                        // `read_vlong15`: a negative short means "the low 15 bits, plus a
                        // vlong of high bits shifted up by 15". `1 << 48` shifted by 15 sets
                        // the sign bit of the i64.
    doc.write_i16(-32768i16); // 0x8000: low 15 bits are 0, more follows
    doc.write_vlong(1i64 << 48);
    doc.extend_from_slice(&footer);

    let input = DocInput::open(&doc, &id, "").unwrap();
    let meta = TermMetadata {
        doc_start_fp,
        singleton_doc_id: -1,
        ..TermMetadata::EMPTY
    };
    let err = input
        .read_postings(meta, BLOCK_SIZE, IndexOptions::Docs, false)
        .unwrap_err();
    assert!(
        matches!(err, Error::Store(lucene_store::Error::Corrupted(_)))
            && format!("{err}").contains("level-0 block"),
        "expected a Corrupted decode error naming the block length, got {err:?}"
    );
}

/// The lazy cursor can reach its tail block with a `doc_count_left` that
/// is not a legal tail size, because that number descends from `docFreq`
/// -- a term-dictionary value, not one of our own.
///
/// The route: a level-0 header whose `docDelta` overshoots what its body
/// decodes (nothing on the wire ties the two together). `advance` walks
/// off the end of the decoded block, `advance_shallow` returns early
/// because the target is still within the *claimed* extent, and the tail
/// path is entered with a whole block's worth of documents still
/// outstanding. Slicing the fixed 256-entry block array by that must not
/// panic; `Lucene104PostingsReader.refillRemainder` asserts the same
/// bound.
#[test]
fn a_tail_block_larger_than_block_size_is_rejected_rather_than_panicking() {
    let id = [73u8; ID_LENGTH];
    let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
    let doc_start_fp = doc.len() as u64;

    let mut body = Vec::new();
    // Packed at one bit per value, every delta 1: docs 0..255 as an array.
    // (A `bitsPerValue == 0` block is kept as a bit set, which catches the
    // header's lie below inside the block, before the tail is reached.)
    body.write_byte(1);
    body.write_bytes(&[0xFF; 32]);
    doc.write_vlong(4); // level0NumBytes: the two header fields, no metadata region
    doc.write_i16(1000); // docDelta: claims the block reaches doc 999
    doc.write_i16(body.len() as i16);
    doc.write_bytes(&body);
    doc.extend_from_slice(&footer);

    let input = DocInput::open(&doc, &id, "").unwrap();
    let meta = TermMetadata {
        doc_start_fp,
        singleton_doc_id: -1,
        ..TermMetadata::EMPTY
    };
    // 600 documents: one full block, leaving 344 -- more than a tail can
    // hold -- once the block is consumed.
    let mut cursor = input
        .lazy_cursor(meta, 600, IndexOptions::Docs, false)
        .unwrap();
    assert_eq!(cursor.next_doc().unwrap(), 0);
    assert_eq!(cursor.advance(255).unwrap(), 255);
    let err = cursor.advance(500).unwrap_err();
    assert!(
        matches!(err, Error::Store(lucene_store::Error::Corrupted(_)))
            && format!("{err}").contains("tail block"),
        "expected a Corrupted decode error naming the tail block, got {err:?}"
    );
}

/// A negative `total_term_freq` is a corrupt term dictionary, rejected
/// before it sizes or divides anything. A count past `u32::MAX` is a real
/// value (Lucene's `totalTermFreq` is a `long`): it is walked like any
/// other, so here -- one position on disk -- it is the frequency
/// cross-check that rejects it, never a blanket `u32` ceiling.
#[test]
fn total_term_freq_is_checked_for_sign_not_capped_at_u32() {
    let id = [74u8; ID_LENGTH];
    let (mut pos, pos_footer) = pos_header_and_footer(&id);
    let pos_start_fp = pos.len() as u64;
    pos.write_vint(1);
    pos.extend_from_slice(&pos_footer);
    let pos_in = PosInput::open(&pos, &id, "").unwrap();
    let meta = TermMetadata {
        pos_start_fp,
        ..TermMetadata::EMPTY
    };
    for total_term_freq in [-1i64, u32::MAX as i64 + 1, i64::MAX] {
        let err = read_positions_for_docs(
            &pos_in,
            None,
            meta,
            &[1],
            total_term_freq,
            IndexOptions::DocsAndFreqsAndPositions,
            false,
            &[0],
        )
        .unwrap_err();
        let want = if total_term_freq < 0 {
            "negative count"
        } else {
            "disagrees with total_term_freq"
        };
        assert!(
            matches!(err, Error::Store(lucene_store::Error::Corrupted(_)))
                && format!("{err}").contains(want),
            "total_term_freq {total_term_freq}: expected {want:?}, got {err:?}"
        );
        assert!(!matches!(err, Error::Unsupported(_)));
        // The whole-term reader guards the same value the same way.
        assert!(read_positions_and_flat(
            &pos_in,
            None,
            meta,
            &[1],
            total_term_freq,
            IndexOptions::DocsAndFreqsAndPositions,
            false,
        )
        .is_err());
    }
    assert_eq!(wire_count(u32::MAX as i64 + 1, "t").unwrap(), 1usize << 32);
    assert_eq!(stream_len(7).unwrap(), 7);
    assert!(matches!(
        stream_len(u32::MAX as usize + 1),
        Err(Error::Unsupported(_))
    ));
}

/// A tail payload whose length runs past the end of `.pos` must be a
/// decode error, not a multi-gigabyte `resize` driven straight off disk.
#[test]
fn a_tail_payload_longer_than_the_file_is_rejected() {
    let id = [75u8; ID_LENGTH];
    let (mut pos, pos_footer) = pos_header_and_footer(&id);
    let pos_start_fp = pos.len() as u64;
    pos.write_vint((3 << 1) | 1); // posDelta=3, payload length follows
    pos.write_vint(1_000_000); // ... and it is longer than the whole file
    pos.extend_from_slice(&pos_footer);
    let pos_in = PosInput::open(&pos, &id, "").unwrap();
    let meta = TermMetadata {
        pos_start_fp,
        ..TermMetadata::EMPTY
    };
    // Both the whole-term reader and the wanted-documents walk.
    assert!(read_positions_and_flat(
        &pos_in,
        None,
        meta,
        &[1],
        1,
        IndexOptions::DocsAndFreqsAndPositions,
        true,
    )
    .is_err());
    assert!(read_occurrences_for_docs(
        &pos_in,
        None,
        meta,
        &[1],
        1,
        IndexOptions::DocsAndFreqsAndPositions,
        true,
        &[0],
    )
    .is_err());
}

/// A field that does not index positions has no `.pos` data to walk, in
/// either shape.
#[test]
fn wanted_documents_need_a_positions_indexing_field() {
    let id = [76u8; ID_LENGTH];
    let (mut pos, pos_footer) = pos_header_and_footer(&id);
    let pos_start_fp = pos.len() as u64;
    pos.write_vint(1);
    pos.extend_from_slice(&pos_footer);
    let pos_in = PosInput::open(&pos, &id, "").unwrap();
    let meta = TermMetadata {
        pos_start_fp,
        ..TermMetadata::EMPTY
    };
    for options in [IndexOptions::Docs, IndexOptions::DocsAndFreqs] {
        assert!(matches!(
            read_positions_for_docs(&pos_in, None, meta, &[1], 1, options, false, &[0]),
            Err(Error::Unsupported(_))
        ));
        assert!(matches!(
            read_occurrences_for_docs(&pos_in, None, meta, &[1], 1, options, false, &[0]),
            Err(Error::Unsupported(_))
        ));
    }
}

#[test]
fn a_block_body_that_disagrees_with_its_headers_byte_length_is_rejected() {
    // A level-0 header records the body's byte length (`level0NumBytes`);
    // the body's own encoding records nothing that ties it to that. Here
    // the header claims one byte more than the body actually occupies, so
    // after decoding it the reader sits one byte short of where the header
    // says the block ends -- and every following block would decode as
    // garbage. Java only `assert`s this; here both paths that decode a
    // full block must return a decode error.
    let id = [42u8; ID_LENGTH];
    let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
    let doc_start_fp = doc.len() as u64;

    let mut body = Vec::new();
    body.write_byte(0); // bitsPerValue == 0: 256 consecutive docs, 0..255.
    doc.write_vlong(4); // level0NumBytes (honest -- the lie under test is blockLength)
    doc.write_i16(256); // docDelta (honest)
    doc.write_i16(body.len() as i16 + 1); // blockLength, matching the lie
    doc.write_bytes(&body);
    doc.write_byte(0); // the extra byte the header accounts for
    doc.extend_from_slice(&footer);

    let input = DocInput::open(&doc, &id, "").unwrap();
    let meta = TermMetadata {
        doc_start_fp,
        singleton_doc_id: -1,
        ..TermMetadata::EMPTY
    };
    // Eager path.
    let err = input
        .read_postings(meta, BLOCK_SIZE, IndexOptions::Docs, false)
        .unwrap_err();
    assert!(
        matches!(err, Error::Store(lucene_store::Error::Corrupted(_))),
        "expected a Corrupted decode error from read_postings, got {err:?}"
    );
    // Lazy path: `advance` refills the block through the same check.
    let mut cursor = input
        .lazy_cursor(meta, BLOCK_SIZE, IndexOptions::Docs, false)
        .unwrap();
    let err = cursor.advance(10).unwrap_err();
    assert!(
        matches!(err, Error::Store(lucene_store::Error::Corrupted(_))),
        "expected a Corrupted decode error from the cursor, got {err:?}"
    );
}

/// `or_shifted_bits` against a bit-by-bit reference: every offset sign
/// and word alignment, ranges that are empty, word-aligned or cross a
/// word, and bits that land outside the destination (dropped).
#[test]
fn or_shifted_bits_matches_a_bit_by_bit_reference() {
    let src: Vec<u64> = (0..4u64)
        .map(|i| 0x9E37_79B9_7F4A_7C15u64.rotate_left(i as u32 * 13) ^ i)
        .collect();
    for off in [-130i64, -64, -63, -1, 0, 1, 5, 63, 64, 65, 200] {
        for (lo, hi) in [
            (0usize, 0usize),
            (3, 3),
            (0, 64),
            (0, 256),
            (5, 70),
            (64, 128),
            (63, 65),
            (100, 255),
            (7, 999),
        ] {
            let mut got = [0u64; 3];
            or_shifted_bits(&mut got, &src, lo, hi, off);
            let mut want = [0u64; 3];
            for i in lo..hi.min(256) {
                if src[i / 64] >> (i % 64) & 1 == 1 {
                    let d = i as i64 + off;
                    if (0..192).contains(&d) {
                        want[d as usize / 64] |= 1 << (d % 64);
                    }
                }
            }
            assert_eq!(got, want, "off {off}, {lo}..{hi}");
        }
    }
}

/// A level-0 header whose impacts length reaches past the header's own
/// `level0NumBytes`: the header walk that slices the impacts out (only
/// for the block it stops on) must report it, not hand the block body to
/// the impacts decoder as if it were impacts -- a bound read from the
/// wrong bytes would skip blocks that compete.
#[test]
fn a_level0_impacts_length_past_its_header_is_corrupt() {
    let id = [42u8; ID_LENGTH];
    let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
    let doc_start_fp = doc.len() as u64;
    // Two full blocks, so the walk reads headers rather than the tail.
    for _ in 0..2 {
        let mut body = Vec::new();
        body.write_byte(0); // bitsPerValue == 0: 256 consecutive docs.
        body.write_byte(0); // freqs: bitsPerValue 0 ...
        body.write_vint(1); // ... every frequency 1.
        doc.write_vlong(4 + 1 + 2); // docDelta, blockLength, impacts length, 2 impact bytes
        doc.write_i16(256);
        doc.write_i16(body.len() as i16);
        doc.write_vint(40); // impacts length: far past the 2 bytes the header holds
        doc.write_bytes(&[1, 1]);
        doc.write_bytes(&body);
    }
    doc.extend_from_slice(&[0u8; 64]);
    doc.extend_from_slice(&footer);

    let input = DocInput::open(&doc, &id, "").unwrap();
    let meta = TermMetadata {
        doc_start_fp,
        singleton_doc_id: -1,
        ..TermMetadata::EMPTY
    };
    let mut cursor = input
        .lazy_cursor_with_flags(
            meta,
            2 * BLOCK_SIZE,
            IndexOptions::DocsAndFreqs,
            false,
            PostingsFlags::Freqs,
        )
        .unwrap();
    let err = cursor.advance_shallow(300).unwrap_err();
    assert!(
        matches!(&err, Error::Store(lucene_store::Error::Corrupted(m)) if m.contains("impacts")),
        "expected the impacts overrun, got {err:?}"
    );
}

#[test]
fn lazy_cursor_advance_rejects_a_block_body_that_undershoots_its_header() {
    // A level-0 header's `docDelta` and the block body's own deltas are
    // independent on the wire: nothing forces them to agree. Here the
    // header claims the block's last doc is 999 while the body decodes to
    // docs 0..255, so `advance_shallow` stops on the block (999 >= 500)
    // and the post-refill search for the first doc >= 500 finds nothing.
    // That must be a decode error, not an index one past the block buffer.
    let id = [42u8; ID_LENGTH];
    let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
    let doc_start_fp = doc.len() as u64;

    let mut body = Vec::new();
    body.write_byte(0); // bitsPerValue == 0: 256 consecutive docs, 0..255.
    doc.write_vlong(4); // level0NumBytes: the two header fields, no metadata region
    doc.write_i16(1000); // docDelta -- a lie: the body only reaches 255.
    doc.write_i16(body.len() as i16); // blockLength (honest, so the seek lands right)
    doc.write_bytes(&body);
    doc.extend_from_slice(&footer);

    let input = DocInput::open(&doc, &id, "").unwrap();
    let meta = TermMetadata {
        doc_start_fp,
        singleton_doc_id: -1,
        ..TermMetadata::EMPTY
    };
    let mut cursor = input
        .lazy_cursor(meta, BLOCK_SIZE, IndexOptions::Docs, false)
        .unwrap();
    let err = cursor.advance(500).unwrap_err();
    assert!(
        matches!(err, Error::Store(lucene_store::Error::Corrupted(_))),
        "expected a Corrupted decode error, got {err:?}"
    );
}

/// The same lie through a *packed* block, whose body is also made not to
/// ascend (a delta that wraps): docs `0..=99`, `1000`, then `401..=555`.
/// The landing searches count the documents below the target, which is
/// the answer only on an ascending block, so each rechecks what it
/// landed on: the in-block one falls back to the exact scan (first
/// document `>= 450` is `1000`), and the one after a refill reports the
/// block as corrupt rather than returning a document below the target.
#[test]
fn lazy_cursor_advance_on_a_packed_block_that_does_not_ascend_never_lands_short() {
    let mut deltas = [1u32; BLOCK_SIZE as usize];
    deltas[100] = 901; // 99 -> 1000
    deltas[101] = 401u32.wrapping_sub(1000); // 1000 -> 401
    let docs: Vec<i32> = deltas
        .iter()
        .scan(-1i32, |d, &x| {
            *d = d.wrapping_add(x as i32);
            Some(*d)
        })
        .collect();
    assert_eq!(
        (docs[99], docs[100], docs[101], docs[255]),
        (99, 1000, 401, 555)
    );

    let id = [43u8; ID_LENGTH];
    let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
    let doc_start_fp = doc.len() as u64;
    let mut body = Vec::new();
    body.write_byte(32);
    for_util::for_encode(&mut deltas, 32, &mut body);
    doc.write_vlong(4); // level0NumBytes: the two header fields, no metadata region
    doc.write_i16(556); // docDelta: the block's last document, 555
    doc.write_i16(body.len() as i16);
    doc.write_bytes(&body);
    doc.extend_from_slice(&footer);

    let input = DocInput::open(&doc, &id, "").unwrap();
    let meta = TermMetadata {
        doc_start_fp,
        singleton_doc_id: -1,
        ..TermMetadata::EMPTY
    };
    let open = || {
        input
            .lazy_cursor(meta, BLOCK_SIZE, IndexOptions::Docs, false)
            .unwrap()
    };
    let mut cursor = open();
    assert_eq!(cursor.advance(50).unwrap(), 50);
    assert_eq!(cursor.advance(450).unwrap(), 1000);
    let err = open().advance(500).unwrap_err();
    assert!(
        matches!(err, Error::Store(lucene_store::Error::Corrupted(_))),
        "expected a Corrupted decode error, got {err:?}"
    );
}

#[test]
fn lazy_cursor_advance_skips_corrupted_earlier_block_without_decoding_it() {
    // Block 0 (docs 0..256) is deliberately corrupt: a dense bit-set
    // encoding (`bitsPerValue == -4`) with zero bits actually set, which
    // `decode_full_block_body` rejects with `Error::Store(Corrupted)` --
    // see `read_full_block_bitset_encoding_rejects_too_few_set_bits`.
    // Block 1 (docs 256..511) is a normal, valid all-consecutive block.
    // `advance(300)` lands in block 1: if the cursor decoded block 0's
    // body along the way (as the eager `read_postings` path always
    // does), this test would fail with a decode error instead of
    // returning doc 300 -- proving the skip genuinely bypasses
    // `ForUtil`/`PForUtil` decode for a block it can prove is entirely
    // behind the target, not just "returns the right answer by luck".
    let id = [23u8; ID_LENGTH];
    let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
    let doc_start_fp = doc.len() as u64;

    // Corrupt block 0: IndexOptions::Docs (no freq field), docDelta=256
    // (claims last doc 255, consistent with a real all-256-bit block),
    // but the body's bit-set has no bits set at all.
    let mut corrupt_body = Vec::new();
    corrupt_body.write_byte((-4i8) as u8); // 4 longs = 256 bits
    corrupt_body.extend_from_slice(&[0u8; 32]); // none set -- corrupt
    doc.write_vlong(4); // level0NumBytes: the two header fields, no metadata region
    doc.write_i16(BLOCK_SIZE as i16); // docDelta = 256
    doc.write_i16(corrupt_body.len() as i16); // blockLength
    doc.write_bytes(&corrupt_body);

    // Block 1: valid, all-consecutive, no freq field (Docs mode).
    write_full_block(&mut doc, false, 0);

    doc.extend_from_slice(&footer);

    let input = DocInput::open(&doc, &id, "").unwrap();
    let meta = TermMetadata {
        doc_start_fp,
        singleton_doc_id: -1,
        ..TermMetadata::EMPTY
    };
    let mut cursor = input
        .lazy_cursor(meta, 2 * BLOCK_SIZE, IndexOptions::Docs, false)
        .unwrap();

    let result = cursor.advance(300).unwrap();
    assert_eq!(result, 300);
    assert_eq!(cursor.freq(), Some(1));

    // Sanity check the other direction: actually decoding block 0 (a
    // target inside it) must surface the corruption error, confirming
    // the earlier success was really a skip and not an accidental pass.
    let mut cursor2 = input
        .lazy_cursor(meta, 2 * BLOCK_SIZE, IndexOptions::Docs, false)
        .unwrap();
    let err = cursor2.advance(10).unwrap_err();
    assert!(matches!(
        err,
        Error::Store(lucene_store::Error::Corrupted(_))
    ));
}

#[test]
fn lazy_cursor_advance_to_doc_before_current_is_a_no_op() {
    let id = [24u8; ID_LENGTH];
    let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
    let doc_start_fp = doc.len() as u64;
    doc.write_group_vints(&[(3 << 1) | 1, (3 << 1) | 1]); // docs 2, 5 (deltas 3,3 from prev=-1), freq=1 each
    doc.extend_from_slice(&footer);

    let input = DocInput::open(&doc, &id, "").unwrap();
    let meta = TermMetadata {
        doc_start_fp,
        singleton_doc_id: -1,
        ..TermMetadata::EMPTY
    };
    let mut cursor = input
        .lazy_cursor(meta, 2, IndexOptions::DocsAndFreqs, false)
        .unwrap();
    assert_eq!(cursor.advance(5).unwrap(), 5);
    // Advancing "backward" to a target at/before the current doc is a
    // documented no-op, matching `PostingsCursor::advance`'s contract.
    assert_eq!(cursor.advance(3).unwrap(), 5);
    assert_eq!(cursor.advance(5).unwrap(), 5);
}

#[test]
fn lazy_cursor_advance_past_last_doc_returns_no_more_docs() {
    let id = [25u8; ID_LENGTH];
    let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
    let doc_start_fp = doc.len() as u64;
    doc.write_group_vints(&[(3 << 1) | 1, (3 << 1) | 1]); // docs 2, 5 (deltas 3,3 from prev=-1)
    doc.extend_from_slice(&footer);

    let input = DocInput::open(&doc, &id, "").unwrap();
    let meta = TermMetadata {
        doc_start_fp,
        singleton_doc_id: -1,
        ..TermMetadata::EMPTY
    };
    let mut cursor = input
        .lazy_cursor(meta, 2, IndexOptions::DocsAndFreqs, false)
        .unwrap();
    assert_eq!(cursor.advance(100).unwrap(), NO_MORE_DOCS);
    assert_eq!(cursor.freq(), None);
    // Once exhausted, further `next_doc()`/`advance()` calls stay
    // `NO_MORE_DOCS` rather than erroring or wrapping around.
    assert_eq!(cursor.next_doc().unwrap(), NO_MORE_DOCS);
    assert_eq!(cursor.advance(1).unwrap(), NO_MORE_DOCS);
}

#[test]
fn lazy_cursor_next_doc_from_start_walks_in_order() {
    let id = [26u8; ID_LENGTH];
    let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
    let doc_start_fp = doc.len() as u64;
    doc.write_group_vints(&[(3 << 1) | 1, (3 << 1) | 1]); // docs 2, 5 (deltas 3,3 from prev=-1)
    doc.extend_from_slice(&footer);

    let input = DocInput::open(&doc, &id, "").unwrap();
    let meta = TermMetadata {
        doc_start_fp,
        singleton_doc_id: -1,
        ..TermMetadata::EMPTY
    };
    let mut cursor = input
        .lazy_cursor(meta, 2, IndexOptions::DocsAndFreqs, false)
        .unwrap();
    assert_eq!(cursor.doc_id(), -1);
    assert_eq!(cursor.next_doc().unwrap(), 2);
    assert_eq!(cursor.next_doc().unwrap(), 5);
    assert_eq!(cursor.next_doc().unwrap(), NO_MORE_DOCS);
}

#[test]
fn decode_term_metadata_absolute_then_delta_docstart() {
    let mut bytes = Vec::new();
    // absolute: docStartFP delta=10 (l = 10<<1 = 20), docFreq>1 so no singleton vint
    bytes.write_vlong(20);
    // second term in same block: docStartFP delta=5 (l = 5<<1 = 10)
    bytes.write_vlong(10);
    let mut r = SliceInput::new(&bytes);

    let first = decode_term_metadata(
        &mut r,
        2,
        true,
        TermMetadata::EMPTY,
        IndexOptions::DocsAndFreqs,
        false,
        2,
    )
    .unwrap();
    assert_eq!(first.doc_start_fp, 10);
    assert_eq!(first.singleton_doc_id, -1);

    let second = decode_term_metadata(
        &mut r,
        2,
        false,
        first,
        IndexOptions::DocsAndFreqs,
        false,
        2,
    )
    .unwrap();
    assert_eq!(second.doc_start_fp, 15);
}

#[test]
fn decode_term_metadata_absolute_resets_fps_even_with_stale_prev() {
    // `Lucene104PostingsReader.decodeTerm` zeroes docStartFP/posStartFP/
    // payStartFP before applying deltas whenever `absolute` is set --
    // regardless of what `prev` carries in. A caller passing a non-empty
    // `prev` alongside `absolute=true` (e.g. a future multi-block
    // BlockTree reader that doesn't reset `prev_meta` per block) must
    // still get FPs computed as deltas-from-zero, not
    // deltas-from-`prev`.
    let mut bytes = Vec::new();
    // docStartFP delta=7 (l = 7<<1 = 14, docFreq>1 so no singleton vint).
    bytes.write_vlong(14);
    // posStartFP delta=3.
    bytes.write_vlong(3);
    let mut r = SliceInput::new(&bytes);

    let stale_prev = TermMetadata {
        doc_start_fp: 1000,
        pos_start_fp: 2000,
        pay_start_fp: 3000,
        singleton_doc_id: -1,
        last_pos_block_offset: -1,
        skip_offset: -1,
    };
    let decoded = decode_term_metadata(
        &mut r,
        2,
        true,
        stale_prev,
        IndexOptions::DocsAndFreqsAndPositions,
        false,
        1,
    )
    .unwrap();
    assert_eq!(decoded.doc_start_fp, 7, "should be 0 + 7, not 1000 + 7");
    assert_eq!(decoded.pos_start_fp, 3, "should be 0 + 3, not 2000 + 3");
}

#[test]
fn decode_term_metadata_singleton_absolute_then_zigzag_delta() {
    let mut bytes = Vec::new();
    // absolute singleton: docStartFP delta=0 (l=0), then raw vint singletonDocID=7
    bytes.write_vlong(0);
    bytes.write_vint(7);
    // next term: singleton delta of +3 via zigzag, flag bit set
    let zz = lucene_util::zigzag::encode(3);
    bytes.write_vlong(((zz as i64) << 1) | 1);
    let mut r = SliceInput::new(&bytes);

    let first = decode_term_metadata(
        &mut r,
        1,
        true,
        TermMetadata::EMPTY,
        IndexOptions::DocsAndFreqs,
        false,
        2,
    )
    .unwrap();
    assert_eq!(first.singleton_doc_id, 7);

    let second = decode_term_metadata(
        &mut r,
        1,
        false,
        first,
        IndexOptions::DocsAndFreqs,
        false,
        2,
    )
    .unwrap();
    assert_eq!(second.singleton_doc_id, 10);
    assert_eq!(second.doc_start_fp, first.doc_start_fp);
}

#[test]
fn decode_term_metadata_rejects_absolute_singleton_delta() {
    let mut bytes = Vec::new();
    bytes.write_vlong(1); // flag bit set on what must be an absolute decode
    let mut r = SliceInput::new(&bytes);
    let err = decode_term_metadata(
        &mut r,
        1,
        true,
        TermMetadata::EMPTY,
        IndexOptions::DocsAndFreqs,
        false,
        2,
    )
    .unwrap_err();
    assert!(matches!(err, Error::AbsoluteSingletonDelta));
}

#[test]
fn decode_term_metadata_rejects_delta_with_no_previous_singleton() {
    let mut bytes = Vec::new();
    bytes.write_vlong(1); // flag bit set, non-absolute
    let mut r = SliceInput::new(&bytes);
    let err = decode_term_metadata(
        &mut r,
        1,
        false,
        TermMetadata::EMPTY,
        IndexOptions::DocsAndFreqs,
        false,
        2,
    )
    .unwrap_err();
    assert!(matches!(err, Error::NoPreviousSingleton));
}

#[test]
fn decode_term_metadata_with_positions_reads_pos_and_pay_fps() {
    // IndexOptions::DocsAndFreqsAndPositions, no offsets, has_payloads
    // true: posStartFP delta, then payStartFP delta (payloads alone
    // trigger it, per `Lucene104PostingsReader.java:239-242`), then
    // (totalTermFreq > BLOCK_SIZE) a lastPosBlockOffset vlong.
    let mut bytes = Vec::new();
    bytes.write_vlong(4); // docStartFP delta = 2 (l=4, bit0 clear -> 4>>1=2)
    bytes.write_vlong(7); // posStartFP delta
    bytes.write_vlong(11); // payStartFP delta (has_payloads=true)
    bytes.write_vlong(300); // lastPosBlockOffset (totalTermFreq=BLOCK_SIZE+1 > BLOCK_SIZE)
    let mut r = SliceInput::new(&bytes);

    let meta = decode_term_metadata(
        &mut r,
        5,
        true,
        TermMetadata::EMPTY,
        IndexOptions::DocsAndFreqsAndPositions,
        true,
        BLOCK_SIZE as i64 + 1,
    )
    .unwrap();
    assert_eq!(meta.doc_start_fp, 2);
    assert_eq!(meta.pos_start_fp, 7);
    assert_eq!(meta.pay_start_fp, 11);
    assert_eq!(meta.last_pos_block_offset, 300);
}

#[test]
fn decode_term_metadata_with_positions_no_last_pos_block_offset_when_small() {
    // totalTermFreq <= BLOCK_SIZE: no lastPosBlockOffset vlong on the wire.
    let mut bytes = Vec::new();
    bytes.write_vlong(0); // docStartFP delta = 0
    bytes.write_vlong(3); // posStartFP delta
    let mut r = SliceInput::new(&bytes);

    let meta = decode_term_metadata(
        &mut r,
        2,
        true,
        TermMetadata::EMPTY,
        IndexOptions::DocsAndFreqsAndPositions,
        false,
        BLOCK_SIZE as i64,
    )
    .unwrap();
    assert_eq!(meta.pos_start_fp, 3);
    assert_eq!(meta.pay_start_fp, 0);
    assert_eq!(meta.last_pos_block_offset, -1);
    // No bytes left to read (would error if the writer had emitted a
    // pay/lastPosBlockOffset field this decode didn't consume).
    assert!(r.read_vlong().is_err());
}

#[test]
fn decode_term_metadata_offsets_without_payloads_still_reads_pay_fp() {
    // DocsAndFreqsAndPositionsAndOffsets subsumes offsets, so payStartFP
    // is written even when has_payloads=false.
    let mut bytes = Vec::new();
    bytes.write_vlong(0);
    bytes.write_vlong(5); // posStartFP delta
    bytes.write_vlong(9); // payStartFP delta (offsets, not payloads)
    let mut r = SliceInput::new(&bytes);

    let meta = decode_term_metadata(
        &mut r,
        2,
        true,
        TermMetadata::EMPTY,
        IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
        false,
        BLOCK_SIZE as i64,
    )
    .unwrap();
    assert_eq!(meta.pos_start_fp, 5);
    assert_eq!(meta.pay_start_fp, 9);
}

fn pos_header_and_footer(id: &[u8; ID_LENGTH]) -> (Vec<u8>, Vec<u8>) {
    let mut before = Vec::new();
    codec_util::write_index_header(&mut before, POS_CODEC, VERSION_CURRENT, id, "");
    let mut after = Vec::new();
    codec_util::write_footer(&mut after);
    (before, after)
}

fn pay_header_and_footer(id: &[u8; ID_LENGTH]) -> (Vec<u8>, Vec<u8>) {
    let mut before = Vec::new();
    codec_util::write_index_header(&mut before, PAY_CODEC, VERSION_CURRENT, id, "");
    let mut after = Vec::new();
    codec_util::write_footer(&mut after);
    (before, after)
}

/// `level0NumBytes` is the level-0 header's own self-description: the
/// byte length from just after it to the start of the block body. This
/// reader reaches the same position by parsing the fields in between, so
/// a disagreement means the two do not describe the same block -- a
/// corrupt vlong, or a `FieldInfos` whose `has_payloads` does not match
/// the bytes, which decides whether two of those fields are there at all.
///
/// Without the check the block body decodes from the wrong offset and
/// produces plausible garbage silently. `read_level1_entry` has always
/// had the equivalent check (`skip1EndFP`); this is its level-0 twin.
#[test]
fn a_level0_header_whose_num_skip_bytes_disagrees_with_its_fields_is_rejected() {
    let id = [76u8; ID_LENGTH];
    let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
    let doc_start_fp = doc.len() as u64;
    let mut body = Vec::new();
    body.write_byte(0); // bitsPerValue == 0: 256 consecutive docs.
                        // The two header fields are 4 bytes and there is no metadata region
                        // (IndexOptions::Docs), so the honest value is 4.
    doc.write_vlong(5);
    doc.write_i16(BLOCK_SIZE as i16);
    doc.write_i16(body.len() as i16);
    doc.write_bytes(&body);
    doc.extend_from_slice(&footer);

    let input = DocInput::open(&doc, &id, "").unwrap();
    let meta = TermMetadata {
        doc_start_fp,
        singleton_doc_id: -1,
        ..TermMetadata::EMPTY
    };
    let err = input
        .read_postings(meta, BLOCK_SIZE, IndexOptions::Docs, false)
        .expect_err("the header does not describe its own fields");
    assert!(
        format!("{err}").contains("level-0 skip header"),
        "unexpected error: {err}"
    );
}

/// A frequency big enough to outrun `.pos` many times over must be
/// rejected up front, not walked until the file EOFs.
///
/// It is not a panic that is at stake -- every index is bounded -- but an
/// allocation blow-up: each refill yields up to 256 more `Position`
/// records, each with its own payload `Vec`, and a `PForUtil` block can be
/// as little as its own token byte. An allocation failure *aborts*, which
/// no `catch_unwind` at the FFI boundary can intercept, so the ceiling has
/// to be checked before the loop rather than discovered inside it.
#[test]
fn a_frequency_that_dwarfs_the_pos_file_is_rejected_before_the_walk() {
    let id = [77u8; ID_LENGTH];
    let (pos, pos_start_fp) = tail_only_pos(&id, &[1, 2, 3]);
    let pos_in = PosInput::open(&pos, &id, "").unwrap();
    let err = walk_one(
        &pos_in,
        PositionOrigin {
            pos_fp: pos_start_fp,
            pay_fp: 0,
            skip: 0,
        },
        10_000_000,
        Some(pos_start_fp),
        3,
    )
    .expect_err("ten million occurrences do not fit in a handful of bytes");
    assert!(
        format!("{err}").contains("cannot fit in the"),
        "unexpected error: {err}"
    );
}

/// `Lucene104PostingsReader.reset`'s three-way `lastPosBlockFP` rule
/// (`Lucene104PostingsReader.java:526-532`), which is the only thing that
/// separates a full `PForUtil` block from the vint tail once a walk has
/// jumped into the middle of `.pos`.
#[test]
fn last_pos_block_fp_matches_lucene_reset() {
    let meta = TermMetadata {
        pos_start_fp: 1000,
        last_pos_block_offset: 77,
        ..TermMetadata::EMPTY
    };
    // Below one full block: everything is the tail, which therefore
    // starts at the term's own posStartFP.
    assert_eq!(last_pos_block_fp(meta, 5), Some(1000));
    assert_eq!(last_pos_block_fp(meta, BLOCK_SIZE as i64 - 1), Some(1000));
    // Exactly one full block: no tail at all, and Java's `-1` sentinel
    // (a pointer nothing can equal) becomes `None` here.
    assert_eq!(last_pos_block_fp(meta, BLOCK_SIZE as i64), None);
    // Past a full block: posStartFP + lastPosBlockOffset.
    assert_eq!(last_pos_block_fp(meta, BLOCK_SIZE as i64 + 1), Some(1077));
}

/// A `.pos` stream holding `count` single-vint occurrences with no
/// offsets or payloads, plus the term's `posStartFP`.
fn tail_only_pos(id: &[u8; ID_LENGTH], deltas: &[i32]) -> (Vec<u8>, u64) {
    let (mut pos, footer) = pos_header_and_footer(id);
    let pos_start_fp = pos.len() as u64;
    for &d in deltas {
        pos.write_vint(d);
    }
    pos.extend_from_slice(&footer);
    (pos, pos_start_fp)
}

fn positions_only_wants() -> PositionWants {
    PositionWants {
        has_offsets: false,
        has_payloads: false,
        want_offsets: false,
        want_payloads: false,
    }
}

fn walk_one(
    pos_in: &PosInput<'_>,
    origin: PositionOrigin,
    freq: usize,
    last_pos_block: Option<u64>,
    tail_count: usize,
) -> Result<Vec<Position>> {
    let mut sink = FullOccurrences {
        occurrences: Vec::new(),
        doc_starts: Vec::new(),
    };
    walk_document_occurrences(
        pos_in,
        None,
        origin,
        freq,
        last_pos_block,
        tail_count,
        positions_only_wants(),
        &mut WalkScratch::new(),
        &mut sink,
    )?;
    Ok(sink.occurrences)
}

/// The honest case, so the three rejections below cannot pass for the
/// wrong reason: a skip of two occurrences into a five-occurrence tail
/// yields the third and fourth, with the position accumulator restarting
/// at the document's first occurrence.
#[test]
fn a_skip_driven_walk_starts_at_the_occurrence_the_origin_names() {
    let id = [70u8; ID_LENGTH];
    let (pos, pos_start_fp) = tail_only_pos(&id, &[1, 2, 3, 4, 5]);
    let pos_in = PosInput::open(&pos, &id, "").unwrap();
    let got = walk_one(
        &pos_in,
        PositionOrigin {
            pos_fp: pos_start_fp,
            pay_fp: 0,
            skip: 2,
        },
        2,
        Some(pos_start_fp),
        5,
    )
    .unwrap();
    assert_eq!(got.iter().map(|p| p.position).collect::<Vec<_>>(), [3, 7]);
}

/// `skipPositions`' `assert posIn.getFilePointer() != lastPosBlockFP`:
/// the vint tail is the last block there is, so skip data asking to step
/// a whole block past it is a disagreement between `.doc` and the term's
/// `totalTermFreq`, not something to seek blindly on.
#[test]
fn a_skip_that_steps_past_the_vint_tail_is_rejected() {
    let id = [71u8; ID_LENGTH];
    let (pos, pos_start_fp) = tail_only_pos(&id, &[1, 2, 3]);
    let pos_in = PosInput::open(&pos, &id, "").unwrap();
    let err = walk_one(
        &pos_in,
        PositionOrigin {
            pos_fp: pos_start_fp,
            pay_fp: 0,
            skip: BLOCK_SIZE as u64,
        },
        1,
        Some(pos_start_fp),
        3,
    )
    .expect_err("the tail is the last block");
    assert!(
        format!("{err}").contains("step past the last .pos block"),
        "unexpected error: {err}"
    );
}

/// The landing offset inside the block is a file-derived value too: it
/// must be checked against the block's own length, not used to index it.
#[test]
fn a_skip_landing_past_the_blocks_own_length_is_rejected() {
    let id = [72u8; ID_LENGTH];
    let (pos, pos_start_fp) = tail_only_pos(&id, &[1, 2, 3]);
    let pos_in = PosInput::open(&pos, &id, "").unwrap();
    let err = walk_one(
        &pos_in,
        PositionOrigin {
            pos_fp: pos_start_fp,
            pay_fp: 0,
            skip: 5,
        },
        1,
        Some(pos_start_fp),
        3,
    )
    .expect_err("5 occurrences into a 3-occurrence block");
    assert!(
        format!("{err}").contains("occurrences into a 3-occurrence"),
        "unexpected error: {err}"
    );
}

/// A frequency that outruns the occurrences `.pos` actually holds must
/// stop, not spin: the empty tail block is reached, refilled to zero
/// occurrences, and the walk reports the disagreement.
#[test]
fn a_frequency_longer_than_the_position_stream_is_rejected() {
    let id = [73u8; ID_LENGTH];
    let (pos, pos_start_fp) = tail_only_pos(&id, &[1, 2]);
    let pos_in = PosInput::open(&pos, &id, "").unwrap();
    let err = walk_one(
        &pos_in,
        PositionOrigin {
            pos_fp: pos_start_fp,
            pay_fp: 0,
            skip: 0,
        },
        3,
        Some(pos_start_fp),
        2,
    )
    .expect_err("only two occurrences exist");
    assert!(
        format!("{err}").contains("outruns the occurrences"),
        "unexpected error: {err}"
    );
}

/// A `.pos` pointer beyond the file is an EOF, not a panic or a silently
/// truncated address.
#[test]
fn a_position_origin_past_the_end_of_the_pos_file_is_an_error() {
    let id = [74u8; ID_LENGTH];
    let (pos, _pos_start_fp) = tail_only_pos(&id, &[1, 2]);
    let pos_in = PosInput::open(&pos, &id, "").unwrap();
    assert!(walk_one(
        &pos_in,
        PositionOrigin {
            pos_fp: u64::MAX,
            pay_fp: 0,
            skip: 0,
        },
        1,
        None,
        0,
    )
    .is_err());
}

/// `position_origin` adds the current `.doc` block's frequencies to the
/// skip data, so it needs frequencies: a `DocsOnly` cursor skipped them
/// on the wire and must say so rather than summing the `1`s it filled in.
#[test]
fn position_origin_needs_a_freqs_cursor() {
    let id = [75u8; ID_LENGTH];
    let (mut doc, doc_footer) = header_and_footer(DOC_CODEC, &id);
    let doc_start_fp = doc.len() as u64;
    // Two docs in a group-varint tail block, deltas 3 and 4 (doc IDs 2
    // and 6) with freq 1 each -- `(delta << 1) | 1` is the freq-is-one
    // packing `read_tail_block` decodes.
    doc.write_group_vints(&[(3 << 1) | 1, (4 << 1) | 1]);
    doc.extend_from_slice(&doc_footer);
    let doc_in = DocInput::open(&doc, &id, "").unwrap();
    let meta = TermMetadata {
        doc_start_fp,
        ..TermMetadata::EMPTY
    };
    let mut cursor = doc_in
        .lazy_cursor_with_flags(
            meta,
            2,
            IndexOptions::DocsAndFreqsAndPositions,
            false,
            PostingsFlags::DocsOnly,
        )
        .unwrap();
    // Before the first move there is no position to report at all.
    assert!(cursor.position_origin().unwrap().is_none());
    assert_eq!(cursor.next_doc().unwrap(), 2);
    let err = cursor
        .position_origin()
        .expect_err("DocsOnly skipped the frequencies the sum needs");
    assert!(
        format!("{err}").contains("PostingsFlags::Freqs"),
        "unexpected error: {err}"
    );

    // A freqs cursor that never opted into tracking positions stepped
    // over every block's `.pos`/`.pay` skip data, so it has no origin to
    // give either.
    let mut cursor = doc_in
        .lazy_cursor_with_flags(
            meta,
            2,
            IndexOptions::DocsAndFreqsAndPositions,
            false,
            PostingsFlags::Freqs,
        )
        .unwrap();
    assert_eq!(cursor.next_doc().unwrap(), 2);
    let err = cursor
        .position_origin()
        .expect_err("no track_positions, no origin");
    assert!(
        format!("{err}").contains("track_positions"),
        "unexpected error: {err}"
    );
}

#[test]
fn flat_occurrences_exact_total_frequency_boundaries_with_every_extra_stream() {
    for total in [127usize, 128, 129, 255, 256, 257] {
        for has_offsets in [false, true] {
            for has_payloads in [false, true] {
                let id = [62; ID_LENGTH];
                let (mut pos, pos_footer) = pos_header_and_footer(&id);
                let (mut pay, pay_footer) = pay_header_and_footer(&id);
                let meta = TermMetadata {
                    pos_start_fp: pos.len() as u64,
                    pay_start_fp: pay.len() as u64,
                    ..TermMetadata::EMPTY
                };
                for _ in 0..total / BLOCK_SIZE as usize {
                    // Constant PForUtil block: every position delta is one.
                    pos.write_byte(0);
                    pos.write_vint(1);
                    if has_payloads {
                        pay.write_byte(0);
                        pay.write_vint(2); // two payload bytes per occurrence
                        pay.write_vint(2 * BLOCK_SIZE);
                        for _ in 0..BLOCK_SIZE {
                            pay.write_bytes(&[7, 9]);
                        }
                    }
                    if has_offsets {
                        pay.write_byte(0);
                        pay.write_vint(2); // start-offset delta
                        pay.write_byte(0);
                        pay.write_vint(5); // offset length
                    }
                }
                for _ in 0..total % BLOCK_SIZE as usize {
                    if has_payloads {
                        pos.write_vint(3); // position delta one, length changes
                        pos.write_vint(2);
                        pos.write_bytes(&[7, 9]);
                    } else {
                        pos.write_vint(1);
                    }
                    if has_offsets {
                        pos.write_vint(5); // start delta two, length changes
                        pos.write_vint(5);
                    }
                }
                pos.extend_from_slice(&pos_footer);
                pay.extend_from_slice(&pay_footer);
                let pos_in = PosInput::open(&pos, &id, "").unwrap();
                let pay_in = PayInput::open(&pay, &id, "").unwrap();
                let freqs = [(total / 2) as i32, 0, (total - total / 2) as i32];
                let options = if has_offsets {
                    IndexOptions::DocsAndFreqsAndPositionsAndOffsets
                } else {
                    IndexOptions::DocsAndFreqsAndPositions
                };
                let nested = read_positions_and_flat(
                    &pos_in,
                    (has_offsets || has_payloads).then_some(&pay_in),
                    meta,
                    &freqs,
                    total as i64,
                    options,
                    has_payloads,
                )
                .unwrap();
                assert!(nested[1].is_empty());
                for (doc, &freq) in nested.iter().zip(&freqs) {
                    assert_eq!(doc.len(), freq as usize);
                    for (index, occurrence) in doc.iter().enumerate() {
                        assert_eq!(occurrence.position, index as i32 + 1);
                        if has_offsets {
                            assert_eq!(occurrence.start_offset, 2 * (index as i32 + 1));
                            assert_eq!(occurrence.end_offset, 2 * (index as i32 + 1) + 5);
                        }
                        assert_eq!(
                            occurrence.payload,
                            if has_payloads { vec![7, 9] } else { vec![] }
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn read_positions_single_position_no_offsets_no_payloads() {
    // One doc, one occurrence: total_term_freq=1 < BLOCK_SIZE, so it's
    // entirely the vint tail (`refillLastPositionBlock`), no PForUtil
    // blocks at all. code = posDelta (no payload bit-packing since
    // has_payloads=false).
    let id = [20u8; ID_LENGTH];
    let (mut pos, pos_footer) = pos_header_and_footer(&id);
    let pos_start_fp = pos.len() as u64;
    pos.write_vint(42); // posDelta = 42
    pos.extend_from_slice(&pos_footer);

    let pos_in = PosInput::open(&pos, &id, "").unwrap();
    let meta = TermMetadata {
        pos_start_fp,
        ..TermMetadata::EMPTY
    };
    let result = read_positions_and_flat(
        &pos_in,
        None,
        meta,
        &[1],
        1,
        IndexOptions::DocsAndFreqsAndPositions,
        false,
    )
    .unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].len(), 1);
    assert_eq!(result[0][0].position, 42);
    assert_eq!(result[0][0].start_offset, -1);
    assert_eq!(result[0][0].end_offset, -1);
    assert!(result[0][0].payload.is_empty());
}

#[test]
fn read_positions_multiple_positions_with_payload_on_some_occurrences() {
    // Two docs sharing one term: doc0 has 2 occurrences (positions 1
    // and 3, first with a payload, second without -- payload length
    // changes 2 -> 0, both written explicitly since the length changed
    // each time); doc1 has 1 occurrence with no payload, reusing the
    // last (0) length so no length vint is written for it.
    let id = [21u8; ID_LENGTH];
    let (mut pos, pos_footer) = pos_header_and_footer(&id);
    let pos_start_fp = pos.len() as u64;

    // doc0, occurrence 0: posDelta=1, payload length changes to 2 (bit set).
    pos.write_vint((1 << 1) | 1);
    pos.write_vint(2);
    pos.write_bytes(&[0xAA, 0xBB]);
    // doc0, occurrence 1: posDelta=2 (position 1+2=3), payload length changes to 0.
    pos.write_vint((2 << 1) | 1);
    pos.write_vint(0);
    // doc1, occurrence 0: posDelta=5 (fresh accumulator, position=5),
    // payload length unchanged (still 0), so bit is clear.
    pos.write_vint(5 << 1);

    pos.extend_from_slice(&pos_footer);

    let pos_in = PosInput::open(&pos, &id, "").unwrap();
    let meta = TermMetadata {
        pos_start_fp,
        ..TermMetadata::EMPTY
    };
    let result = read_positions_and_flat(
        &pos_in,
        None,
        meta,
        &[2, 1],
        3,
        IndexOptions::DocsAndFreqsAndPositions,
        true,
    )
    .unwrap();
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].len(), 2);
    assert_eq!(result[0][0].position, 1);
    assert_eq!(result[0][0].payload, vec![0xAA, 0xBB]);
    assert_eq!(result[0][1].position, 3);
    assert!(result[0][1].payload.is_empty());
    assert_eq!(result[1].len(), 1);
    assert_eq!(result[1][0].position, 5);
    assert!(result[1][0].payload.is_empty());
}

#[test]
fn read_positions_with_offsets() {
    // One doc, two occurrences, offsets but no payloads: payIn carries
    // the offset start-delta/length vint pairs interleaved with .pos'
    // posDelta vints (no payload bit-packing on the posDelta code since
    // has_payloads=false).
    let id = [22u8; ID_LENGTH];
    let (mut pos, pos_footer) = pos_header_and_footer(&id);
    let pos_start_fp = pos.len() as u64;

    // occurrence 0: posDelta=1 (position=1), offset [0,3) (length 3, bit set).
    pos.write_vint(1);
    pos.write_vint(1); // offset start delta = 0, bit set (length changes)
    pos.write_vint(3);
    // occurrence 1: posDelta=1 (position=2), offset [4,7) (start delta=4
    // from lastStartOffset=0, length still 3 so bit clear, reused).
    pos.write_vint(1);
    pos.write_vint(4 << 1);

    pos.extend_from_slice(&pos_footer);

    let pos_in = PosInput::open(&pos, &id, "").unwrap();
    let meta = TermMetadata {
        pos_start_fp,
        ..TermMetadata::EMPTY
    };
    let result = read_positions_and_flat(
        &pos_in,
        None,
        meta,
        &[2],
        2,
        IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
        false,
    )
    .unwrap();
    assert_eq!(result[0][0].position, 1);
    assert_eq!(result[0][0].start_offset, 0);
    assert_eq!(result[0][0].end_offset, 3);
    assert_eq!(result[0][1].position, 2);
    assert_eq!(result[0][1].start_offset, 4);
    assert_eq!(result[0][1].end_offset, 7);
}

#[test]
fn read_positions_rejects_offsets_without_pay_input() {
    // total_term_freq spans one full block (BLOCK_SIZE), which is what
    // actually requires `.pay` for a field with offsets -- a term whose
    // whole total_term_freq fit in the vint tail wouldn't need it (see
    // `read_positions`'s own doc comment), so this deliberately uses a
    // full-block-sized total to exercise the real requirement.
    let id = [23u8; ID_LENGTH];
    let (mut pos, pos_footer) = pos_header_and_footer(&id);
    pos.write_byte(0);
    pos.write_vint(1);
    pos.extend_from_slice(&pos_footer);
    let pos_in = PosInput::open(&pos, &id, "").unwrap();
    let err = read_positions_and_flat(
        &pos_in,
        None,
        TermMetadata::EMPTY,
        &[BLOCK_SIZE],
        BLOCK_SIZE as i64,
        IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
        false,
    )
    .unwrap_err();
    assert!(matches!(err, Error::Unsupported(_)));
}

#[test]
fn read_positions_rejects_freqs_summing_past_total_term_freq() {
    // `freqs` (decoded independently from `.doc`) claiming more
    // occurrences than `total_term_freq` (decoded from the term
    // dictionary) is corrupted input -- must be a decode error, not an
    // out-of-bounds panic when the re-chop loop runs past the end of the
    // flat `pos_deltas` array.
    let id = [27u8; ID_LENGTH];
    let (mut pos, pos_footer) = pos_header_and_footer(&id);
    let pos_start_fp = pos.len() as u64;
    pos.write_vint(1); // one posDelta -- total_term_freq=1 worth of data
    pos.extend_from_slice(&pos_footer);

    let pos_in = PosInput::open(&pos, &id, "").unwrap();
    let meta = TermMetadata {
        pos_start_fp,
        ..TermMetadata::EMPTY
    };
    // freqs claims 2 occurrences for a single doc, but total_term_freq
    // (and thus the decoded pos_deltas array) only has 1.
    let err = read_positions_and_flat(
        &pos_in,
        None,
        meta,
        &[2],
        1,
        IndexOptions::DocsAndFreqsAndPositions,
        false,
    )
    .unwrap_err();
    assert!(matches!(
        err,
        Error::Store(lucene_store::Error::Corrupted(_))
    ));
}

#[test]
fn read_positions_rejects_freqs_summing_below_total_term_freq() {
    // The reverse mismatch: freqs under-claim occurrences relative to
    // total_term_freq. Must also be a decode error, not a silent partial
    // decode that drops leftover positions.
    let id = [28u8; ID_LENGTH];
    let (mut pos, pos_footer) = pos_header_and_footer(&id);
    let pos_start_fp = pos.len() as u64;
    pos.write_vint(1);
    pos.write_vint(1);
    pos.extend_from_slice(&pos_footer);

    let pos_in = PosInput::open(&pos, &id, "").unwrap();
    let meta = TermMetadata {
        pos_start_fp,
        ..TermMetadata::EMPTY
    };
    let err = read_positions_and_flat(
        &pos_in,
        None,
        meta,
        &[1],
        2,
        IndexOptions::DocsAndFreqsAndPositions,
        false,
    )
    .unwrap_err();
    assert!(matches!(
        err,
        Error::Store(lucene_store::Error::Corrupted(_))
    ));
}

#[test]
fn read_positions_rejects_non_position_index_options() {
    let id = [24u8; ID_LENGTH];
    let (mut pos, pos_footer) = pos_header_and_footer(&id);
    pos.extend_from_slice(&pos_footer);
    let pos_in = PosInput::open(&pos, &id, "").unwrap();
    let err = read_positions_and_flat(
        &pos_in,
        None,
        TermMetadata::EMPTY,
        &[1],
        1,
        IndexOptions::DocsAndFreqs,
        false,
    )
    .unwrap_err();
    assert!(matches!(err, Error::Unsupported(_)));
}

#[test]
fn read_positions_exactly_one_full_block_boundary() {
    // total_term_freq == BLOCK_SIZE (256) exactly: one full PForUtil
    // block, no vint tail at all -- exercises `num_full_blocks == 1,
    // tail_count == 0`. All 256 posDeltas equal 1 (positions 1..=256,
    // one doc's occurrences), payload lengths all equal 0 (still needs a
    // PForUtil block + a zero-length `numBytes` vint on `.pay`, matching
    // what the real writer emits even for an all-empty-payload block).
    let id = [25u8; ID_LENGTH];
    let (mut pos, pos_footer) = pos_header_and_footer(&id);
    let pos_start_fp = pos.len() as u64;
    pos.write_byte(0); // PForUtil token: bitsPerValue=0, numExceptions=0
    pos.write_vint(1); // fill value: posDelta=1 for all 256 occurrences
    pos.extend_from_slice(&pos_footer);

    let (mut pay, pay_footer) = pay_header_and_footer(&id);
    let pay_start_fp = pay.len() as u64;
    pay.write_byte(0); // payloadLengthBuffer PForUtil token: all-equal
    pay.write_vint(0); // fill value: length 0 for all 256
    pay.write_vint(0); // numBytes = 0 (no payload bytes follow)
    pay.extend_from_slice(&pay_footer);

    let pos_in = PosInput::open(&pos, &id, "").unwrap();
    let pay_in = PayInput::open(&pay, &id, "").unwrap();
    let meta = TermMetadata {
        pos_start_fp,
        pay_start_fp,
        ..TermMetadata::EMPTY
    };
    let result = read_positions_and_flat(
        &pos_in,
        Some(&pay_in),
        meta,
        &[BLOCK_SIZE],
        BLOCK_SIZE as i64,
        IndexOptions::DocsAndFreqsAndPositions,
        true,
    )
    .unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].len(), BLOCK_SIZE as usize);
    let expected: Vec<i32> = (1..=BLOCK_SIZE).collect();
    assert_eq!(
        result[0].iter().map(|p| p.position).collect::<Vec<_>>(),
        expected
    );
    assert!(result[0].iter().all(|p| p.payload.is_empty()));
}

/// Recomputes the trailing 8-byte CRC32 of a codec-footer-terminated
/// buffer in place, so a byte flip in the body is not simply "caught" by
/// the checksum -- c15/c19/c25's shape, and `blocktree.rs`'s
/// `resign_footer` twin.
fn resign_footer(buf: &mut [u8]) {
    let len = buf.len();
    let checksum = crc32fast::hash(&buf[..len - 8]) as u64;
    buf[len - 8..].copy_from_slice(&checksum.to_be_bytes());
}

/// Drives every `.doc`-only reader in this module over one term and
/// reports whether any of them refused the bytes. Deliberately does *not*
/// require the three to agree: on a corrupt file they legitimately can
/// disagree, and the point of the sweep is that none of them panics or
/// aborts.
fn exercise_doc_readers(doc: &[u8], id: &[u8; ID_LENGTH], meta: TermMetadata, df: i32) -> bool {
    let Ok(input) = DocInput::open(doc, id, "") else {
        return true;
    };
    let opts = IndexOptions::DocsAndFreqs;
    let mut rejected = false;
    if input.read_postings(meta, df, opts, false).is_err() {
        rejected = true;
    }
    if input
        .read_postings_with_flags(meta, df, opts, false, PostingsFlags::DocsOnly)
        .is_err()
    {
        rejected = true;
    }
    // A full sequential walk of the lazy cursor, then a skipping one --
    // the two reach different code (`refill` vs `advance_shallow`'s
    // seek-past, and `skip_level1_to`'s span jump).
    match input.lazy_cursor(meta, df, opts, false) {
        Err(_) => rejected = true,
        Ok(mut c) => {
            // `df + 1` steps is a bound, not a heuristic: every step
            // either moves within a decoded block or consumes wire, and
            // the term claims `df` documents.
            for _ in 0..=df {
                match c.next_doc() {
                    Ok(NO_MORE_DOCS) => break,
                    Ok(_) => {
                        let _ = c.freq();
                        let _ = c.level0_impacts();
                        let _ = c.level1_impacts();
                    }
                    Err(_) => {
                        rejected = true;
                        break;
                    }
                }
            }
        }
    }
    match input.lazy_cursor(meta, df, opts, false) {
        Err(_) => rejected = true,
        Ok(mut c) => {
            let mut target = 0i32;
            while target < df {
                match c.advance_shallow(target) {
                    Ok(_) => {}
                    Err(_) => {
                        rejected = true;
                        break;
                    }
                }
                match c.advance(target) {
                    Ok(NO_MORE_DOCS) => break,
                    Ok(_) => {}
                    Err(_) => {
                        rejected = true;
                        break;
                    }
                }
                target = target.saturating_add(700);
            }
        }
    }
    rejected
}

/// A `.doc` level-0 header for a positions-indexing field: the same
/// shape [`write_full_block_with_impacts`] writes, plus
/// `readLevel0PosData`'s four sub-fields (`posEndFPDelta`,
/// `posBufferUpto`, `payEndFPDelta`, `payloadByteUpto`) inside the
/// freq-gated region. Needed so a corruption sweep can reach
/// [`read_pos_skip`] and [`LazyDocsCursor::position_origin`], which the
/// no-positions helper never emits a byte of.
fn write_full_block_with_pos_skip(
    out: &mut Vec<u8>,
    pos_end_fp_delta: i64,
    pos_buffer_upto: u8,
    pay_end_fp_delta: i64,
) {
    let mut meta_region = Vec::new();
    let mut impact_bytes = Vec::new();
    write_impacts(&mut impact_bytes, &[Impact { freq: 1, norm: 1 }]);
    meta_region.write_vint(impact_bytes.len() as i32);
    meta_region.write_bytes(&impact_bytes);
    meta_region.write_vlong(pos_end_fp_delta);
    meta_region.write_byte(pos_buffer_upto);
    meta_region.write_vlong(pay_end_fp_delta);
    meta_region.write_vint(0); // payloadByteUpto

    let mut block_body = Vec::new();
    block_body.write_byte(0); // bitsPerValue == 0: all 256 deltas are 1.
    block_body.write_byte(0); // PForUtil freq token: all-equal
    block_body.write_vint(1); // every freq is 1

    out.write_vlong((meta_region.len() + 4) as i64);
    out.write_i16(BLOCK_SIZE as i16); // docDelta via writeVInt15
    out.write_i16((meta_region.len() + block_body.len()) as i16); // blockLength
    out.write_bytes(&meta_region);
    out.write_bytes(&block_body);
}

/// Re-signed single-byte corruption sweep over the `.doc` of a
/// *positions-indexing* term, driven through
/// [`read_occurrences_for_doc`] -- the skip-data path, which is the half
/// [`every_resigned_single_byte_doc_corruption_is_an_error_or_a_clean_decode`]
/// cannot reach because its field indexes no positions.
///
/// Every value this reaches comes off the flipped `.doc`: the level-0
/// header's `posEndFPDelta`/`posBufferUpto`/`payEndFPDelta`, the
/// frequencies the in-block sum adds up, and the `docDelta`/`blockLength`
/// that frame them. `.pos`/`.pay` are left intact so a rejection is
/// attributable to the `.doc` bytes alone.
#[test]
fn every_resigned_single_byte_positional_doc_corruption_is_an_error_or_a_clean_decode() {
    let id = [42u8; ID_LENGTH];
    let (mut pos, pos_footer) = pos_header_and_footer(&id);
    let pos_start_fp = pos.len() as u64;
    for _ in 0..2 {
        pos.write_byte(0); // full block: all 256 posDeltas are 1
        pos.write_vint(1);
    }
    let pos_blocks_len = pos.len() as u64 - pos_start_fp;
    for _ in 0..2 {
        pos.write_vint(2); // vint tail: (posDelta 1 << 1) | keep-length
    }
    pos.extend_from_slice(&pos_footer);

    let (mut pay, pay_footer) = pay_header_and_footer(&id);
    let pay_start_fp = pay.len() as u64;
    for _ in 0..2 {
        pay.write_byte(0); // payloadLengthBuffer: all zero
        pay.write_vint(0);
        pay.write_vint(0); // numBytes == 0
    }
    let pay_block_len = (pay.len() as u64 - pay_start_fp) / 2;
    pay.extend_from_slice(&pay_footer);

    let (mut doc, doc_footer) = header_and_footer(DOC_CODEC, &id);
    let doc_start_fp = doc.len() as u64;
    for _ in 0..2 {
        write_full_block_with_pos_skip(
            &mut doc,
            (pos_blocks_len / 2) as i64,
            0,
            pay_block_len as i64,
        );
    }
    doc.write_group_vints(&[3; 2]); // tail: 2 docs, delta 1, freq 1
    doc.extend_from_slice(&doc_footer);

    let df = 2 * BLOCK_SIZE + 2;
    let ttf = df as i64; // every document has exactly one occurrence
    let meta = TermMetadata {
        doc_start_fp,
        singleton_doc_id: -1,
        pos_start_fp,
        pay_start_fp,
        last_pos_block_offset: pos_blocks_len as i64,
        skip_offset: -1,
    };
    let opts = IndexOptions::DocsAndFreqsAndPositions;
    let pos_in = PosInput::open(&pos, &id, "").unwrap();
    let pay_in = PayInput::open(&pay, &id, "").unwrap();

    let exercise = |doc: &[u8]| -> bool {
        let Ok(doc_in) = DocInput::open(doc, &id, "") else {
            return true;
        };
        let mut rejected = doc_in.read_postings(meta, df, opts, true).is_err();
        for target in [0, 1, 7, 255, 256, 300, 511, 512, 513] {
            match read_occurrences_for_doc(
                &doc_in,
                &pos_in,
                Some(&pay_in),
                meta,
                df,
                ttf,
                opts,
                true,
                target,
            ) {
                Ok(_) => {}
                Err(_) => rejected = true,
            }
        }
        rejected
    };
    assert!(!exercise(&doc));

    let body = doc_start_fp as usize..doc.len() - codec_util::FOOTER_LENGTH;
    let mut total = 0usize;
    let mut rejected = 0usize;
    for off in body {
        for mask in [0x01u8, 0x80] {
            let mut corrupt = doc.clone();
            corrupt[off] ^= mask;
            resign_footer(&mut corrupt);
            total += 1;
            if exercise(&corrupt) {
                rejected += 1;
            }
        }
    }
    // Measured when this was written: 49 of 60 (81.7%). As with the
    // sibling sweep, the bar is "no panic, no abort" plus a floor -- the
    // eleven accepted flips all land in a `posEndFPDelta`/`payEndFPDelta`
    // or an impacts byte, where a different-but-in-range value produces a
    // different-but-well-formed answer the format cannot contradict.
    assert_eq!(total, 60);
    assert!(
        rejected >= 45,
        "only {rejected} of {total} re-signed positional .doc corruptions were rejected"
    );
}

/// Re-signed single-byte corruption sweep over a whole `.doc` term body:
/// flip one bit, **re-sign the codec footer so the checksum still
/// passes**, and require a typed error or a clean decode from every
/// `.doc` reader -- never a panic, and never an allocation abort
/// `catch_unwind` cannot intercept, which through the FFI is a dead JVM.
///
/// The term is deliberately the richest shape this format has: one
/// level-1 skip entry with a real impacts run, the 32 full level-0 blocks
/// it spans (each with its own impacts and its own `PForUtil` freq
/// block), and a group-varint tail. So the sweep reaches
/// `read_level1_entry`'s `skip1EndFP`/`numImpactBytes`, every level-0
/// header field including `level0NumBytes`/`docDelta`/`blockLength`, all
/// three `bitsPerValue` body encodings (a flip of the token byte reaches
/// the packed and bit-set branches as well as the all-consecutive one),
/// and the tail block's own doc-count bound.
#[test]
fn every_resigned_single_byte_doc_corruption_is_an_error_or_a_clean_decode() {
    let id = [41u8; ID_LENGTH];
    let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
    let doc_start_fp = doc.len() as u64;
    let mut span = Vec::new();
    for i in 0..LEVEL1_FACTOR {
        let impacts = vec![Impact {
            freq: 1 + i as i32,
            norm: 2 + i as i64,
        }];
        write_full_block_with_impacts(&mut span, true, 1 + (i as i32 % 5), &impacts);
    }
    write_level1_entry_with_impacts(
        &mut doc,
        LEVEL1_NUM_DOCS,
        &span,
        &[Impact { freq: 9, norm: 4 }],
    );
    doc.extend_from_slice(&span);
    // Tail: 8 documents, delta 1, freq 1 -- `(delta << 1) | 1`.
    doc.write_group_vints(&[3; 8]);
    doc.extend_from_slice(&footer);

    let df = LEVEL1_NUM_DOCS + 8;
    let meta = TermMetadata {
        doc_start_fp,
        singleton_doc_id: -1,
        ..TermMetadata::EMPTY
    };
    // The clean file must decode, or the sweep proves nothing.
    assert!(!exercise_doc_readers(&doc, &id, meta, df));

    let body = doc_start_fp as usize..doc.len() - codec_util::FOOTER_LENGTH;
    let mut total = 0usize;
    let mut rejected = 0usize;
    for off in body {
        for mask in [0x01u8, 0x80] {
            let mut corrupt = doc.clone();
            corrupt[off] ^= mask;
            resign_footer(&mut corrupt);
            total += 1;
            if exercise_doc_readers(&corrupt, &id, meta, df) {
                rejected += 1;
            }
        }
    }

    // Measured when this was written: 589 of 744 (79.2%). The rest decode
    // to a different but self-consistent postings list -- flipping a
    // freq's fill value, or an impacts delta, is a wrong answer the
    // format has no way to notice, which is exactly what the checksum
    // that re-signing defeats exists for. Other files reject far fewer
    // (on the order of 15-85%), so a rate below 100% is the norm.
    //
    // What this pins is that nothing *panics or aborts*, plus a floor so
    // a future change that stops bounding something fails loudly.
    assert_eq!(total, 744);
    assert!(
        rejected >= 570,
        "only {rejected} of {total} re-signed .doc corruptions were rejected"
    );
}

#[test]
fn a_payload_length_block_claiming_a_negative_length_is_rejected_not_a_panic() {
    // `.pay`'s payload-length stream is `PForUtil`-decoded as `u32` and
    // kept as `i32`, so a corrupt block can hand `read_positions` a
    // *negative* length. `negative as usize` sign-extends to ~2^64 and
    // the old `payload_upto + len` either panicked outright ("attempt to
    // add with overflow", debug) or wrapped to a value *below*
    // `payload_upto`, slipped past the `end > payload_bytes.len()` check
    // and panicked in `payload_bytes[start..end]` ("slice index starts at
    // 256 but ends at 255", release). Two full blocks, because the first
    // one has to move `payload_upto` off zero for the wrap to be the
    // interesting case rather than a plain "longer than the file".
    let id = [40u8; ID_LENGTH];
    let (mut pos, pos_footer) = pos_header_and_footer(&id);
    let pos_start_fp = pos.len() as u64;
    for _ in 0..2 {
        pos.write_byte(0); // PForUtil token: all 256 posDeltas equal
        pos.write_vint(1);
    }
    pos.extend_from_slice(&pos_footer);

    let (mut pay, pay_footer) = pay_header_and_footer(&id);
    let pay_start_fp = pay.len() as u64;
    // Block 0: every payload one byte long, 256 payload bytes.
    pay.write_byte(0);
    pay.write_vint(1);
    pay.write_vint(BLOCK_SIZE);
    pay.write_bytes(&[7u8; BLOCK_SIZE as usize]);
    // Block 1: every payload length is 0xFFFF_FFFF, i.e. `-1` as an
    // `i32`, with no payload bytes at all behind it.
    pay.write_byte(0);
    pay.write_vint(-1);
    pay.write_vint(0);
    pay.extend_from_slice(&pay_footer);

    let pos_in = PosInput::open(&pos, &id, "").unwrap();
    let pay_in = PayInput::open(&pay, &id, "").unwrap();
    let meta = TermMetadata {
        pos_start_fp,
        pay_start_fp,
        ..TermMetadata::EMPTY
    };
    let err = read_positions_and_flat(
        &pos_in,
        Some(&pay_in),
        meta,
        &[2 * BLOCK_SIZE],
        2 * BLOCK_SIZE as i64,
        IndexOptions::DocsAndFreqsAndPositions,
        true,
    )
    .unwrap_err();
    assert!(
        matches!(err, Error::Store(lucene_store::Error::Corrupted(ref m)) if m.contains("payload length")),
        "expected a corruption report, got {err:?}"
    );
}

#[test]
fn read_positions_full_block_with_offsets_no_payloads() {
    // Same full-PForUtil-block shape as
    // `read_positions_exactly_one_full_block_boundary`, but for a field
    // with offsets and no payloads: `.pay` carries only the two
    // offset-start-delta/offset-length PForUtil blocks (no payload-length
    // block, no payload bytes), matching `read_positions`'s `has_offsets`
    // branch. A mismatch in the payload/offset `.pay` cursor ordering
    // would silently produce wrong offsets.
    let id = [26u8; ID_LENGTH];
    let (mut pos, pos_footer) = pos_header_and_footer(&id);
    let pos_start_fp = pos.len() as u64;
    pos.write_byte(0); // PForUtil token: bitsPerValue=0, numExceptions=0
    pos.write_vint(1); // fill value: posDelta=1 for all 256 occurrences
    pos.extend_from_slice(&pos_footer);

    let (mut pay, pay_footer) = pay_header_and_footer(&id);
    let pay_start_fp = pay.len() as u64;
    pay.write_byte(0); // offsetStartDeltaBuffer PForUtil token: all-equal
    pay.write_vint(2); // fill value: start delta = 2 for all 256
    pay.write_byte(0); // offsetLengthBuffer PForUtil token: all-equal
    pay.write_vint(5); // fill value: length = 5 for all 256
    pay.extend_from_slice(&pay_footer);

    let pos_in = PosInput::open(&pos, &id, "").unwrap();
    let pay_in = PayInput::open(&pay, &id, "").unwrap();
    let meta = TermMetadata {
        pos_start_fp,
        pay_start_fp,
        ..TermMetadata::EMPTY
    };
    let result = read_positions_and_flat(
        &pos_in,
        Some(&pay_in),
        meta,
        &[BLOCK_SIZE],
        BLOCK_SIZE as i64,
        IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
        false,
    )
    .unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].len(), BLOCK_SIZE as usize);
    for (i, p) in result[0].iter().enumerate() {
        let expected_start = (i as i32 + 1) * 2;
        assert_eq!(p.start_offset, expected_start, "occurrence {i}");
        assert_eq!(p.end_offset, expected_start + 5, "occurrence {i}");
        assert!(p.payload.is_empty());
    }
}

// ---------------------------------------------------------------------
// Min-of-N A/B harness. Criterion is unusable
// on this machine (the same code measured 83/91/129 us across three
// consecutive runs), so this reports the **minimum** of 40 timed
// repetitions, which is the statistic a noisy shared machine does not
// corrupt upward. Run with:
//
//   ARITH_PERF_REPS=40 cargo test --release -p lucene-codecs --lib \
//       arith_gate_perf -- --nocapture
// ---------------------------------------------------------------------

fn perf_doc_file(id: &[u8; ID_LENGTH]) -> (Vec<u8>, u64, i32) {
    let (mut doc, footer) = header_and_footer(DOC_CODEC, id);
    let doc_start_fp = doc.len() as u64;
    let mut span = Vec::new();
    for _ in 0..LEVEL1_FACTOR {
        write_full_block_with_impacts(&mut span, true, 3, &[Impact { freq: 2, norm: 5 }]);
    }
    write_level1_entry_with_impacts(
        &mut doc,
        LEVEL1_NUM_DOCS,
        &span,
        &[Impact { freq: 9, norm: 4 }],
    );
    doc.extend_from_slice(&span);
    doc.write_group_vints(&[3; 8]);
    doc.extend_from_slice(&footer);
    (doc, doc_start_fp, LEVEL1_NUM_DOCS + 8)
}

/// 16 full `.pos` blocks (4 096 occurrences, one per document) with a
/// one-byte payload each, plus the matching `.pay`.
fn perf_pos_pay(id: &[u8; ID_LENGTH]) -> (Vec<u8>, u64, Vec<u8>, u64, usize) {
    const BLOCKS: usize = 16;
    let (mut pos, pos_footer) = pos_header_and_footer(id);
    let pos_start_fp = pos.len() as u64;
    for _ in 0..BLOCKS {
        pos.write_byte(0);
        pos.write_vint(1);
    }
    pos.extend_from_slice(&pos_footer);

    let (mut pay, pay_footer) = pay_header_and_footer(id);
    let pay_start_fp = pay.len() as u64;
    for _ in 0..BLOCKS {
        pay.write_byte(0);
        pay.write_vint(1); // every payload one byte long
        pay.write_vint(BLOCK_SIZE);
        pay.write_bytes(&[9u8; BLOCK_SIZE as usize]);
    }
    pay.extend_from_slice(&pay_footer);
    (
        pos,
        pos_start_fp,
        pay,
        pay_start_fp,
        BLOCKS * BLOCK_SIZE as usize,
    )
}

fn min_of<F: FnMut()>(reps: usize, mut f: F) -> std::time::Duration {
    let mut best = std::time::Duration::MAX;
    for _ in 0..reps {
        let t = std::time::Instant::now();
        f();
        best = best.min(t.elapsed());
    }
    best
}

/// Runs one repetition of each arm under `cargo test` -- enough to keep
/// the harness itself exercised and honest -- and a real measurement when
/// `ARITH_PERF_REPS` asks for one.
#[test]
fn arith_gate_perf() {
    let id = [90u8; ID_LENGTH];
    let (doc, doc_start_fp, df) = perf_doc_file(&id);
    let doc_in = DocInput::open(&doc, &id, "").unwrap();
    let doc_meta = TermMetadata {
        doc_start_fp,
        singleton_doc_id: -1,
        ..TermMetadata::EMPTY
    };
    let opts = IndexOptions::DocsAndFreqs;

    let (pos, pos_start_fp, pay, pay_start_fp, ttf) = perf_pos_pay(&id);
    let pos_in = PosInput::open(&pos, &id, "").unwrap();
    let pay_in = PayInput::open(&pay, &id, "").unwrap();
    let pos_meta = TermMetadata {
        pos_start_fp,
        pay_start_fp,
        ..TermMetadata::EMPTY
    };
    let freqs = vec![1i32; ttf];
    let wanted: Vec<usize> = (0..ttf).step_by(4).collect();
    let popts = IndexOptions::DocsAndFreqsAndPositions;

    // 1 under `cargo test`; set `ARITH_PERF_REPS=40` for a measurement.
    let reps: usize = std::env::var("ARITH_PERF_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);

    let read_postings = min_of(reps, || {
        let p = doc_in.read_postings(doc_meta, df, opts, false).unwrap();
        std::hint::black_box(&p);
    });
    let lazy_walk = min_of(reps, || {
        let mut c = doc_in.lazy_cursor(doc_meta, df, opts, false).unwrap();
        let mut sum = 0i64;
        loop {
            match c.next_doc().unwrap() {
                NO_MORE_DOCS => break,
                d => sum += d as i64,
            }
        }
        std::hint::black_box(sum);
    });
    let wanted_walk = min_of(reps, || {
        let r = read_positions_for_docs(
            &pos_in,
            Some(&pay_in),
            pos_meta,
            &freqs,
            ttf as i64,
            popts,
            true,
            &wanted,
        )
        .unwrap();
        std::hint::black_box(&r);
    });
    let whole_term_positions = min_of(reps, || {
        let r = super::read_positions(
            &pos_in,
            Some(&pay_in),
            pos_meta,
            &freqs,
            ttf as i64,
            popts,
            true,
        )
        .unwrap();
        std::hint::black_box(&r);
    });

    println!(
        "ARITHPERF read_postings={:?} lazy_walk={:?} wanted_walk={:?} whole_term_positions={:?}",
        read_postings, lazy_walk, wanted_walk, whole_term_positions
    );
}
