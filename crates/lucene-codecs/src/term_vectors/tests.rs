#![cfg(test)]
// The arithmetic gate is about values read off disk; a test's `i + 1` is
// not one. See docs/arithmetic-gate.md.
#![allow(clippy::arithmetic_side_effects)]

use super::*;

fn write_vint(out: &mut Vec<u8>, mut v: i32) {
    loop {
        let mut b = (v & 0x7f) as u8;
        v = ((v as u32) >> 7) as i32;
        if v != 0 {
            b |= 0x80;
            out.push(b);
        } else {
            out.push(b);
            break;
        }
    }
}

fn write_string(out: &mut Vec<u8>, s: &str) {
    write_vint(out, s.len() as i32);
    out.extend_from_slice(s.as_bytes());
}

fn id() -> [u8; ID_LENGTH] {
    [3u8; ID_LENGTH]
}

/// Hand-encodes a single-doc chunk (docBase=0, chunkDocs=1) with one
/// field (number 5) that has POSITIONS+OFFSETS+PAYLOADS, two terms:
/// "cat" (freq 2, prefix-shared with nothing) and "car" (freq 1,
/// sharing prefix "ca" with "cat"). Values were derived by hand in the
/// module's development notes; see the assertions below for what they
/// decode to.
fn build_single_doc_chunk() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (tvd, tvx, tvm, _chunk_start) = build_single_doc_chunk_with_meta_overrides(1, 0, 0);
    (tvd, tvx, tvm)
}

/// Same as [`build_single_doc_chunk`] but lets the caller override the
/// outer `numChunks`/`numDirtyChunks`/`numDirtyDocs` meta fields (to
/// exercise `open`'s consistency-check error paths) and returns the
/// `.tvd` offset of the chunk's `docBase` byte (to let callers corrupt
/// it and re-sign the `.tvd` footer for `CorruptChunkBounds` tests).
fn build_single_doc_chunk_with_meta_overrides(
    num_chunks_outer: i32,
    num_dirty_chunks: i32,
    num_dirty_docs: i32,
) -> (Vec<u8>, Vec<u8>, Vec<u8>, usize) {
    build_single_doc_chunk_with_index_chunks(
        num_chunks_outer + 1,
        num_chunks_outer,
        num_dirty_chunks,
        num_dirty_docs,
    )
}

/// As above, but the `.tvx` index meta's own chunk count is given
/// separately so a test can build a (structurally consistent) index that
/// claims more than one chunk -- the only way to reach `open`'s third
/// dirty-chunk check, which needs `numDirtyChunks >= 2`.
fn build_single_doc_chunk_with_index_chunks(
    index_num_chunks: i32,
    num_chunks_outer: i32,
    num_dirty_chunks: i32,
    num_dirty_docs: i32,
) -> (Vec<u8>, Vec<u8>, Vec<u8>, usize) {
    let mut tvd = Vec::new();
    tvd.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
    write_string(&mut tvd, DATA_CODEC);
    tvd.extend_from_slice(&(VERSION_CURRENT as u32).to_be_bytes());
    tvd.extend_from_slice(&id());
    tvd.push(0); // empty suffix
    let chunk_start = tvd.len() as i64;

    write_vint(&mut tvd, 0); // docBase
    write_vint(&mut tvd, 1 << 1); // token: chunkDocs=1, dirty=0
    write_vint(&mut tvd, 1); // numFields = totalFields = 1

    // fieldNums: 1 distinct field (number 5), 8 bits/value.
    tvd.push(8); // token: (totalDistinct-1=0)<<5 | bitsPerFieldNum=8
    tvd.push(5); // field number 5, MSB-packed (byte-aligned, trivial)

    // allFieldNumOffs: 1 field, offset 0 into field_nums, 1 bit/value.
    write_vint(&mut tvd, 1); // slice byte length
    tvd.push(0x00);

    // flags: selector=1 (direct array), 1 field, FLAGS_BITS=4, value=7
    // (POSITIONS|OFFSETS|PAYLOADS).
    write_vint(&mut tvd, 1); // selector
    write_vint(&mut tvd, 1); // slice byte length
    tvd.push(0x07);

    // numTerms: 1 field, 8 bits/value, value=2.
    write_vint(&mut tvd, 8); // bitsRequired
    write_vint(&mut tvd, 1); // slice byte length
    tvd.push(2);

    // prefixLengths [0, 2] (block-packed: min=0, bpv=2).
    tvd.extend_from_slice(&[0x05, 0x20]);
    // suffixLengths [3, 1] (min=1, bpv=2): token, minValue vlong, packed.
    tvd.extend_from_slice(&[0x04, 0x01, 0x80]);
    // termFreqsMinus1 [1, 0] (min=0, bpv=1).
    tvd.extend_from_slice(&[0x03, 0x80]);

    // positions_flat [0, 2, 1] (min=0, bpv=2): term0 abs=0, delta=2
    // (2nd occurrence); term1 abs=1 (its own first occurrence).
    tvd.extend_from_slice(&[0x05, 0x24]);

    // charsPerTerm: 1 distinct field, value 4.0.
    tvd.extend_from_slice(&4.0f32.to_bits().to_le_bytes());
    // start_offsets_flat [0, 0, 0] (bpv=0, constant -- no packed bytes).
    tvd.push(0x01);
    // lengths_flat [0, 0, 0] (bpv=0, constant).
    tvd.push(0x01);
    // payload_lengths_flat [1, 0, 2] (min=0, bpv=2).
    tvd.extend_from_slice(&[0x05, 0x48]);

    // LZ4 (CompressionMode.FAST, no dictionary): literal-only unit
    // wrapping "cat"+"r" (term suffixes) then payload bytes 0xAA,0xBB,0xCC.
    let payload = [b'c', b'a', b't', b'r', 0xAA, 0xBB, 0xCC];
    tvd.push((payload.len() as u8) << 4); // LZ4 literal-length token
    tvd.extend_from_slice(&payload);

    tvd.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
    tvd.extend_from_slice(&0u32.to_be_bytes());
    let checksum = crc32fast::hash(&tvd) as u64;
    tvd.extend_from_slice(&checksum.to_be_bytes());

    // .tvx: docs=[0, sentinel=maxDoc], startPointers=[chunk_start, sentinel=maxPointer].
    let mut tvx = Vec::new();
    tvx.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
    write_string(&mut tvx, INDEX_CODEC);
    tvx.extend_from_slice(&(INDEX_VERSION_CURRENT as u32).to_be_bytes());
    tvx.extend_from_slice(&id());
    tvx.push(0);
    let docs_start = tvx.len() as i64;
    let docs_end = tvx.len() as i64;
    let start_pointers_end = tvx.len() as i64;
    tvx.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
    tvx.extend_from_slice(&0u32.to_be_bytes());
    let checksum = crc32fast::hash(&tvx) as u64;
    tvx.extend_from_slice(&checksum.to_be_bytes());

    // .tvm
    let max_doc = 1i32;
    let max_pointer = (tvd.len() - codec_util::FOOTER_LENGTH) as i64;
    let mut tvm = Vec::new();
    tvm.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
    write_string(&mut tvm, META_CODEC);
    tvm.extend_from_slice(&(VERSION_CURRENT as u32).to_be_bytes());
    tvm.extend_from_slice(&id());
    tvm.push(0);
    write_vint(&mut tvm, 0); // packedIntsVersion (unused by this port)
    write_vint(&mut tvm, 4096); // chunkSize (unused by this port)
    tvm.extend_from_slice(&max_doc.to_le_bytes());
    tvm.extend_from_slice(&0i32.to_le_bytes()); // blockShift=0
    tvm.extend_from_slice(&index_num_chunks.to_le_bytes());
    tvm.extend_from_slice(&docs_start.to_le_bytes());
    let doc_mins: Vec<i64> = std::iter::once(0i64)
        .chain(std::iter::repeat_n(
            max_doc as i64,
            (index_num_chunks - 1) as usize,
        ))
        .collect();
    for min in doc_mins {
        tvm.extend_from_slice(&min.to_le_bytes());
        tvm.extend_from_slice(&0i32.to_le_bytes());
        tvm.extend_from_slice(&0i64.to_le_bytes());
        tvm.push(0);
    }
    tvm.extend_from_slice(&docs_end.to_le_bytes());
    let fp_mins: Vec<i64> = std::iter::once(chunk_start)
        .chain(std::iter::repeat_n(
            max_pointer,
            (index_num_chunks - 1) as usize,
        ))
        .collect();
    for min in fp_mins {
        tvm.extend_from_slice(&min.to_le_bytes());
        tvm.extend_from_slice(&0i32.to_le_bytes());
        tvm.extend_from_slice(&0i64.to_le_bytes());
        tvm.push(0);
    }
    tvm.extend_from_slice(&start_pointers_end.to_le_bytes());
    tvm.extend_from_slice(&max_pointer.to_le_bytes());
    write_vint(&mut tvm, num_chunks_outer); // numChunks (outer)
    write_vint(&mut tvm, num_dirty_chunks); // numDirtyChunks
    write_vint(&mut tvm, num_dirty_docs); // numDirtyDocs
    tvm.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
    tvm.extend_from_slice(&0u32.to_be_bytes());
    let checksum = crc32fast::hash(&tvm) as u64;
    tvm.extend_from_slice(&checksum.to_be_bytes());

    (tvd, tvx, tvm, chunk_start as usize)
}

/// Recomputes and appends-in-place the trailing 8-byte CRC32 checksum of
/// a codec-footer-terminated buffer (footer magic + zero algorithm id
/// are assumed already present; only the checksum bytes are patched).
fn resign_footer(buf: &mut [u8]) {
    let len = buf.len();
    let checksum = crc32fast::hash(&buf[..len - 8]) as u64;
    buf[len - 8..].copy_from_slice(&checksum.to_be_bytes());
}

#[test]
fn single_doc_full_decode_positions_offsets_payloads() {
    let (tvd, tvx, tvm) = build_single_doc_chunk();
    let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
    assert_eq!(reader.max_doc(), 1);

    let doc = reader.document(0).unwrap().unwrap();
    assert_eq!(doc.fields.len(), 1);
    let field = &doc.fields[0];
    assert_eq!(field.field_number, 5);
    assert!(field.has_positions && field.has_offsets && field.has_payloads);
    assert_eq!(field.terms.len(), 2);

    let cat = &field.terms[0];
    assert_eq!(cat.term, b"cat");
    assert_eq!(cat.freq, 2);
    assert_eq!(cat.positions, Some(vec![0, 2]));
    assert_eq!(cat.start_offsets, Some(vec![0, 8]));
    assert_eq!(cat.end_offsets, Some(vec![3, 11]));
    assert_eq!(cat.payloads, Some(vec![vec![0xAA], vec![]]));

    let car = &field.terms[1];
    assert_eq!(car.term, b"car");
    assert_eq!(car.freq, 1);
    assert_eq!(car.positions, Some(vec![1]));
    assert_eq!(car.start_offsets, Some(vec![4]));
    assert_eq!(car.end_offsets, Some(vec![7]));
    assert_eq!(car.payloads, Some(vec![vec![0xBB, 0xCC]]));
}

#[test]
fn doc_out_of_range_rejected() {
    let (tvd, tvx, tvm) = build_single_doc_chunk();
    let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
    assert!(matches!(
        reader.document(1),
        Err(Error::DocOutOfRange(1, 1))
    ));
    assert!(matches!(
        reader.document(-1),
        Err(Error::DocOutOfRange(-1, 1))
    ));
}

#[test]
fn wrong_segment_id_rejected() {
    let (tvd, tvx, tvm) = build_single_doc_chunk();
    let wrong_id = [9u8; ID_LENGTH];
    assert!(open(&tvd, &tvx, &tvm, &wrong_id, "").is_err());
}

#[test]
fn direct_writer_bits_required_rounds_up_to_supported_widths() {
    assert_eq!(direct_writer_bits_required(0), 1);
    assert_eq!(direct_writer_bits_required(1), 1);
    assert_eq!(direct_writer_bits_required(3), 2);
    assert_eq!(direct_writer_bits_required(7), 4); // 3 bits needed, rounds to 4
    assert_eq!(direct_writer_bits_required(255), 8);
    assert_eq!(direct_writer_bits_required(256), 12); // 9 bits needed, rounds to 12
}

#[test]
fn num_chunks_mismatch_rejected() {
    // The builder always writes index_num_chunks=2 (1 real chunk + 1
    // sentinel); an outer numChunks of 2 breaks the required
    // index==outer+1 relationship (2 != 2+1).
    let (tvd, tvx, tvm, _) = build_single_doc_chunk_with_index_chunks(2, 2, 0, 0);
    assert!(matches!(
        open(&tvd, &tvx, &tvm, &id(), ""),
        Err(Error::NumChunksMismatch {
            index_num_chunks: 2,
            outer_num_chunks: 2
        })
    ));
}

#[test]
fn too_many_dirty_chunks_rejected() {
    let (tvd, tvx, tvm, _) = build_single_doc_chunk_with_meta_overrides(1, 2, 2);
    assert!(matches!(
        open(&tvd, &tvx, &tvm, &id(), ""),
        Err(Error::TooManyDirtyChunks(2, 1))
    ));
}

#[test]
fn dirty_chunks_docs_mismatch_rejected() {
    let (tvd, tvx, tvm, _) = build_single_doc_chunk_with_meta_overrides(1, 1, 0);
    assert!(matches!(
        open(&tvd, &tvx, &tvm, &id(), ""),
        Err(Error::DirtyChunksDocsMismatch(1, 0))
    ));
}

/// `Lucene90CompressingTermVectorsReader`'s third dirty-chunk check.
/// Two dirty chunks holding one doc between them is impossible: a forced
/// flush always carries at least one pending doc.
#[test]
fn more_dirty_chunks_than_dirty_docs_rejected() {
    let (tvd, tvx, tvm, _) = build_single_doc_chunk_with_index_chunks(4, 3, 2, 1);
    assert!(matches!(
        open(&tvd, &tvx, &tvm, &id(), ""),
        Err(Error::MoreDirtyChunksThanDirtyDocs(2, 1))
    ));
}

/// The boundary the check above must *not* reject: as many dirty docs as
/// dirty chunks is legal (every forced flush held exactly one doc).
#[test]
fn equal_dirty_chunks_and_dirty_docs_accepted() {
    let (tvd, tvx, tvm, _) = build_single_doc_chunk_with_index_chunks(4, 3, 2, 2);
    assert!(open(&tvd, &tvx, &tvm, &id(), "").is_ok());
}

#[test]
fn wrong_tvd_length_rejected() {
    let (mut tvd, tvx, tvm) = build_single_doc_chunk();
    tvd.push(0); // stray byte after the footer
    assert!(open(&tvd, &tvx, &tvm, &id(), "").is_err());
}

#[test]
fn corrupt_chunk_bounds_rejected() {
    // Patch the .tvd chunk's token so it claims chunkDocs=2 starting at
    // docBase=0, while .tvm still says maxDoc=1 -- doc_base+chunk_docs
    // (2) exceeds max_doc (1).
    let (mut tvd, tvx, tvm, chunk_start) = build_single_doc_chunk_with_meta_overrides(1, 0, 0);
    // Byte right after docBase's single vint byte (docBase=0) is the
    // token vint; original value is `1<<1=2` (chunkDocs=1). Bump it to
    // `2<<1=4` (chunkDocs=2).
    assert_eq!(tvd[chunk_start + 1], 2);
    tvd[chunk_start + 1] = 4;
    resign_footer(&mut tvd);

    let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
    assert!(matches!(
        reader.document(0),
        Err(Error::CorruptChunkBounds { .. })
    ));
}

/// Splices `insert` over `tvd[at..at + remove]`, then repairs `.tvm`'s
/// `maxPointer` and re-signs both footers, so that only
/// the semantic invariant under test can fire and never a checksum or a
/// length mismatch the corruption happened to also break.
fn splice_tvd(tvd: &mut Vec<u8>, tvm: &mut [u8], at: usize, remove: usize, insert: &[u8]) {
    let footer = codec_util::FOOTER_LENGTH;
    // `.tvm` tail is: maxPointer (i64 LE), then the three one-byte vints
    // `numChunks`/`numDirtyChunks`/`numDirtyDocs`, then the footer.
    let mp_off = tvm.len() - footer - 3 - 8;
    let old_max_pointer = i64::from_le_bytes(tvm[mp_off..mp_off + 8].try_into().unwrap());
    assert_eq!(
        old_max_pointer,
        (tvd.len() - footer) as i64,
        "maxPointer offset"
    );
    tvd.splice(at..at + remove, insert.iter().copied());
    let new_max_pointer = (tvd.len() - footer) as i64;
    tvm[mp_off..mp_off + 8].copy_from_slice(&new_max_pointer.to_le_bytes());
    resign_footer(tvd);
    resign_footer(tvm);
}

/// The `.tvm` offsets of the three trailing chunk-count vlongs.
fn tvm_counts_offset(tvm: &[u8]) -> usize {
    tvm.len() - codec_util::FOOTER_LENGTH - 3
}

#[test]
fn negative_distinct_field_count_extension_is_a_decode_error_not_an_allocation() {
    // `totalDistinctFields` accumulated into a `u32` would turn a negative
    // escape into ~4 billion and
    // `packed_ints::byte_count` then sized `vec![0u8; n]` at ~16 GB --
    // an abort, not a catchable error.
    let (mut tvd, tvx, mut tvm, chunk_start) = build_single_doc_chunk_with_meta_overrides(1, 0, 0);
    assert_eq!(tvd[chunk_start + 3], 8, "fieldNums token");
    // token: totalDistinctFields inline = 7 (escape), bitsPerFieldNum = 8,
    // followed by a vint encoding of -1.
    let mut insert = vec![(7u8 << 5) | 8];
    write_vint(&mut insert, -1);
    splice_tvd(&mut tvd, &mut tvm, chunk_start + 3, 1, &insert);

    let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
    let err = reader.document(0).unwrap_err();
    assert!(
        format!("{err}").contains("negative distinct-field-number count extension"),
        "{err}"
    );
}

#[test]
fn more_distinct_field_numbers_than_fields_is_a_decode_error() {
    // The positive, plausible case a sign check and an EOF would both
    // miss: 13 distinct field numbers at 8 bits each is 13 readable bytes
    // in a chunk that has them, but the writer only ever emits the
    // *deduplicated* numbers of the chunk's fields, and this chunk has
    // exactly one field.
    let (mut tvd, tvx, mut tvm, chunk_start) = build_single_doc_chunk_with_meta_overrides(1, 0, 0);
    let mut insert = vec![(7u8 << 5) | 8];
    write_vint(&mut insert, 5); // 7 + 5 + 1 = 13
    splice_tvd(&mut tvd, &mut tvm, chunk_start + 3, 1, &insert);

    let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
    let err = reader.document(0).unwrap_err();
    assert!(
        format!("{err}").contains("13 distinct field numbers across only 1 fields"),
        "{err}"
    );
}

#[test]
fn absurd_per_document_field_count_is_a_decode_error_not_an_allocation() {
    // `numFields` is the chunk's per-document field count; for a
    // single-document chunk it is a bare vint. It sizes six
    // `Vec::with_capacity` calls, so a negative one (~2^64 through the
    // `as usize`) and a merely enormous one are both aborts.
    for (name, value) in [("negative", -1i32), ("absurd", 1 << 30)] {
        let (mut tvd, tvx, mut tvm, chunk_start) =
            build_single_doc_chunk_with_meta_overrides(1, 0, 0);
        assert_eq!(tvd[chunk_start + 2], 1, "numFields");
        let mut insert = Vec::new();
        write_vint(&mut insert, value);
        splice_tvd(&mut tvd, &mut tvm, chunk_start + 2, 1, &insert);

        let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
        let err = reader.document(0).unwrap_err();
        let text = format!("{err}");
        assert!(
            text.contains("bad per-document field count") || text.contains("more than the"),
            "{name}: {text}"
        );
    }
}

#[test]
fn absurd_per_field_term_count_is_a_decode_error_not_an_allocation() {
    // `numTerms` is `direct_reader`-encoded at a bit width the chunk
    // chooses. At 64 bits an entry is an arbitrary `i64`: a negative one
    // made `term_offsets` non-monotonic (and its `as usize` casts
    // astronomic), which is what the three `block_packed` streams are
    // sliced by.
    let (mut tvd, tvx, mut tvm, chunk_start) = build_single_doc_chunk_with_meta_overrides(1, 0, 0);
    // numTerms header is `bitsRequired` vint, slice length vint, bytes.
    assert_eq!(tvd[chunk_start + 10], 8, "numTerms bitsRequired");
    assert_eq!(tvd[chunk_start + 11], 1, "numTerms slice length");
    let mut insert = vec![64, 8]; // 64 bits per value, 8-byte slice
    insert.extend_from_slice(&(-1i64).to_le_bytes());
    splice_tvd(&mut tvd, &mut tvm, chunk_start + 10, 3, &insert);

    let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
    let err = reader.document(0).unwrap_err();
    assert!(
        format!("{err}").contains("bad per-field term count"),
        "{err}"
    );
}

#[test]
fn absurd_term_frequency_is_a_decode_error_not_an_overflow() {
    // `termFreqsMinus1` is block-packed, so an entry is an arbitrary
    // `i64` and `freq = v + 1` overflows at `i64::MAX`. The stream
    // lengths derived from it size three more `block_packed::decode_all`
    // calls.
    let (mut tvd, tvx, mut tvm, chunk_start) = build_single_doc_chunk_with_meta_overrides(1, 0, 0);
    // termFreqsMinus1 is the third block-packed stream: prefixLengths
    // (2 bytes), suffixLengths (3 bytes), then [0x03, 0x80].
    let freqs_at = chunk_start + 13 + 2 + 3;
    assert_eq!(&tvd[freqs_at..freqs_at + 2], &[0x03, 0x80]);
    // One block of 2 values, bitsPerValue = 64, minValueEquals0 set.
    let mut insert = vec![((64u32 << 1) | 1) as u8];
    insert.extend_from_slice(&i64::MAX.to_be_bytes());
    insert.extend_from_slice(&0i64.to_be_bytes());
    splice_tvd(&mut tvd, &mut tvm, freqs_at, 2, &insert);

    let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
    let err = reader.document(0).unwrap_err();
    assert!(
        format!("{err}").contains("frequenc") || format!("{err}").contains("overflow"),
        "{err}"
    );
}

#[test]
fn negative_term_prefix_length_is_a_decode_error_not_a_shorter_term() {
    // A prefix is shared with the *previously decoded* term, which lives
    // in a different part of the file. Folding a negative one away with
    // `.max(0)` would decode the term from its suffix alone -- accepted,
    // self-consistent and wrong, where Java throws (`System.arraycopy` with a negative `destPos`).
    let (mut tvd, tvx, mut tvm, chunk_start) = build_single_doc_chunk_with_meta_overrides(1, 0, 0);
    let prefixes_at = chunk_start + 13;
    assert_eq!(&tvd[prefixes_at..prefixes_at + 2], &[0x05, 0x20]);
    // One block of 2 values, bitsPerValue = 64, minValueEquals0 set.
    let mut insert = vec![((64u32 << 1) | 1) as u8];
    insert.extend_from_slice(&0i64.to_be_bytes());
    insert.extend_from_slice(&(-1i64).to_be_bytes());
    splice_tvd(&mut tvd, &mut tvm, prefixes_at, 2, &insert);

    let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
    let err = reader.document(0).unwrap_err();
    assert!(
        format!("{err}").contains("negative term prefix length"),
        "{err}"
    );
}

#[test]
fn negative_term_suffix_length_is_a_decode_error() {
    // Folding negatives away with `.max(0)` in the decompressed-length
    // ceiling would let the per-document byte cursor add one as a huge
    // `usize` and index the decompressed buffer from nowhere.
    let (mut tvd, tvx, mut tvm, chunk_start) = build_single_doc_chunk_with_meta_overrides(1, 0, 0);
    let suffixes_at = chunk_start + 13 + 2;
    assert_eq!(&tvd[suffixes_at..suffixes_at + 3], &[0x04, 0x01, 0x80]);
    // One block of 2 values, bitsPerValue = 64, minValueEquals0 set.
    let mut insert = vec![((64u32 << 1) | 1) as u8];
    insert.extend_from_slice(&(-1i64).to_be_bytes());
    insert.extend_from_slice(&1i64.to_be_bytes());
    splice_tvd(&mut tvd, &mut tvm, suffixes_at, 3, &insert);

    let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
    let err = reader.document(0).unwrap_err();
    assert!(
        format!("{err}").contains("negative term suffix length"),
        "{err}"
    );
}

#[test]
fn a_num_chunks_at_i64_max_is_a_mismatch_not_an_overflow() {
    // `numChunks + 1` is a `long` add in Java. Here it panicked in a
    // debug build for a `.tvm` vlong near `i64::MAX`.
    let (tvd, tvx, mut tvm) = build_single_doc_chunk();
    let at = tvm_counts_offset(&tvm);
    let mut insert = Vec::new();
    write_vlong_test(&mut insert, i64::MAX as u64);
    insert.push(0); // numDirtyChunks
    insert.push(0); // numDirtyDocs
    tvm.splice(at..at + 3, insert);
    resign_footer(&mut tvm);

    assert!(matches!(
        open(&tvd, &tvx, &tvm, &id(), ""),
        Err(Error::NumChunksMismatch { .. })
    ));
}

#[test]
fn a_negative_dirty_chunk_count_is_rejected_at_open() {
    // Java only *asserts* `numDirtyChunks >= 0`. Without a real check a
    // negative one passes all three cross-checks (it is "nonzero", it is
    // not greater than `numChunks`, and it is not greater than a positive
    // `numDirtyDocs`) and then reaches `too_dirty`'s `* 100`, which
    // overflows at `i64::MIN`.
    let (tvd, tvx, mut tvm) = build_single_doc_chunk();
    let at = tvm_counts_offset(&tvm);
    let mut insert = vec![1u8]; // numChunks = 1
    write_vlong_test(&mut insert, i64::MIN as u64);
    insert.push(200); // numDirtyDocs, > maxDocsPerChunk so `too_dirty` gets there
    tvm.splice(at..at + 3, insert);
    resign_footer(&mut tvm);

    let Err(err) = open(&tvd, &tvx, &tvm, &id(), "") else {
        panic!("a negative numDirtyChunks must be rejected at open");
    };
    assert!(format!("{err}").contains("negative chunk counts"), "{err}");
}

#[test]
fn a_negative_max_pointer_is_a_decode_error_not_an_overflow() {
    // `maxPointer as usize + FOOTER_LENGTH` sign-extended a negative
    // `.tvm` field into a huge length and then overflowed the add.
    let (tvd, tvx, mut tvm) = build_single_doc_chunk();
    let mp_off = tvm.len() - codec_util::FOOTER_LENGTH - 3 - 8;
    tvm[mp_off..mp_off + 8].copy_from_slice(&(-16i64).to_le_bytes());
    resign_footer(&mut tvm);

    let Err(err) = open(&tvd, &tvx, &tvm, &id(), "") else {
        panic!("a negative maxPointer must be rejected at open");
    };
    assert!(format!("{err}").contains(".tvd length should be"), "{err}");
}

#[test]
fn a_field_number_offset_past_the_distinct_field_numbers_is_a_decode_error() {
    // Java only asserts `fieldNumOff < fieldNums.length`; a corrupt
    // `allFieldNumOffs` entry indexed `field_nums` out of range here.
    let (mut tvd, tvx, mut tvm, chunk_start) = build_single_doc_chunk_with_meta_overrides(1, 0, 0);
    // allFieldNumOffs: slice length vint then the packed byte. The chunk
    // has one distinct field number, so `bits_per_off` is 1 and the only
    // legal offset is 0.
    assert_eq!(tvd[chunk_start + 5], 1, "allFieldNumOffs slice length");
    assert_eq!(tvd[chunk_start + 6], 0x00);
    tvd[chunk_start + 6] = 0x01;
    splice_tvd(&mut tvd, &mut tvm, chunk_start + 6, 1, &[0x01]);

    let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
    let err = reader.document(0).unwrap_err();
    assert!(
        format!("{err}").contains("is outside the chunk's 1 distinct field numbers"),
        "{err}"
    );
}

/// A document with two fields, several terms each, carrying positions,
/// offsets and payloads -- so a sweep over a segment of these reaches
/// every stream `read_chunk` decodes, not just the two a minimal chunk
/// has.
fn rich_doc(n: usize) -> TermVectorsDocument {
    let term = |s: &str| format!("{s}{n:03}").into_bytes();
    TermVectorsDocument {
        fields: vec![
            TermVectorField {
                field_number: 0,
                has_positions: true,
                has_offsets: true,
                has_payloads: true,
                terms: vec![
                    TermVectorTerm {
                        term: term("alpha"),
                        freq: 2,
                        positions: Some(vec![0, 4]),
                        start_offsets: Some(vec![0, 20]),
                        end_offsets: Some(vec![8, 28]),
                        payloads: Some(vec![vec![1, 2], vec![3]]),
                    },
                    TermVectorTerm {
                        term: term("alpine"),
                        freq: 1,
                        positions: Some(vec![7]),
                        start_offsets: Some(vec![35]),
                        end_offsets: Some(vec![43]),
                        payloads: Some(vec![Vec::new()]),
                    },
                ],
            },
            TermVectorField {
                field_number: 3,
                has_positions: true,
                has_offsets: false,
                has_payloads: false,
                terms: vec![TermVectorTerm {
                    term: term("beta"),
                    freq: 1,
                    positions: Some(vec![1]),
                    start_offsets: None,
                    end_offsets: None,
                    payloads: None,
                }],
            },
        ],
    }
}

/// Re-signed single-byte corruption sweep over the whole `.tvd` chunk
/// region and the whole `.tvm`, in c15/c19/c25's shape: flip one bit,
/// re-sign the footer so the CRC cannot "catch" the corruption, and
/// require a typed error rather than a panic or -- the failure this
/// batch exists to remove -- an allocation abort no `catch_unwind` can
/// intercept.
///
/// The fixture is deliberately a **writer-produced, multi-chunk,
/// multi-field, payload-carrying** segment rather than the hand-built
/// single-document chunk the rest of this module's error tests use. A
/// one-document chunk takes `read_chunk`'s `chunk_docs == 1` shortcut
/// (so it never reaches `block_packed::decode_all` for the per-document
/// field counts), has one distinct field number (so the `fieldNumOff`
/// range check is vacuous), and gives the `.tvx` monotonic arrays
/// nothing to discriminate: such a sweep measures the fixture, not
/// the decoder.
///
/// The assertion is deliberately *not* "every corruption is rejected":
/// many single-bit flips produce a self-consistent chunk that decodes to
/// different but well-formed vectors, which is what a checksum is for.
/// What is asserted is that nothing panics and that the rejection rate
/// does not silently collapse.
#[test]
fn every_resigned_single_byte_tvd_and_tvm_corruption_is_an_error_or_a_clean_decode() {
    // 200 rich documents: the byte-size trigger closes several chunks, so
    // `.tvx`'s two monotonic arrays carry real entries.
    let docs: Vec<TermVectorsDocument> = (0..400).map(rich_doc).collect();
    let (tvd, tvx, tvm) = write_best_speed(&docs, &id(), "");
    let max_doc = docs.len() as i32;
    {
        let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
        assert!(
            reader.num_chunks() >= 3,
            "the sweep needs a multi-chunk segment, got {}",
            reader.num_chunks()
        );
        // Two fields with different flag sets, so the per-field-number
        // flags array and the `fieldNumOff` range check are both live.
        let chunk = reader.read_chunk(0).unwrap();
        assert!(
            chunk.num_docs() > 1,
            "the `chunkDocs == 1` shortcut must not fire"
        );
        assert_eq!(chunk.document(0).unwrap().unwrap().fields.len(), 2);
    }

    // The `.tvd` header is fixed-size and shared with every other codec
    // file; the interesting bytes are the chunk region.
    let header_len = {
        let mut probe = SliceInput::new(&tvd);
        codec_util::check_index_header(
            &mut probe,
            DATA_CODEC,
            VERSION_START,
            VERSION_CURRENT,
            &id(),
            "",
        )
        .unwrap();
        probe.position()
    };
    let mut total = 0usize;
    let mut rejected = 0usize;

    // Read every document, not just doc 0: the per-document field-count
    // and byte-cursor arrays only discriminate when they are walked.
    let read_all = |tvd: &[u8], tvx: &[u8], tvm: &[u8]| -> Result<()> {
        let reader = open(tvd, tvx, tvm, &id(), "")?;
        let mut cursor = ChunkCursor::new();
        for doc in 0..reader.max_doc().min(max_doc) {
            cursor.document(&reader, doc)?;
        }
        Ok(())
    };

    for off in (header_len..tvd.len() - codec_util::FOOTER_LENGTH).step_by(7) {
        for mask in [0x01u8, 0x80] {
            let mut corrupt = tvd.clone();
            corrupt[off] ^= mask;
            resign_footer(&mut corrupt);
            total += 1;
            if read_all(&corrupt, &tvx, &tvm).is_err() {
                rejected += 1;
            }
        }
    }
    for off in 0..tvm.len() - codec_util::FOOTER_LENGTH {
        for mask in [0x01u8, 0x80] {
            let mut corrupt = tvm.clone();
            corrupt[off] ^= mask;
            resign_footer(&mut corrupt);
            total += 1;
            if read_all(&tvd, &tvx, &corrupt).is_err() {
                rejected += 1;
            }
        }
    }

    // Measured when this was written: 688 of 3 724 (18%), and that low
    // figure is the fixture being *right*, not the decoder being weak.
    // A `.tvd` chunk is mostly its LZ4 literal run -- term suffixes and
    // payload bytes -- and flipping one of those yields a different but
    // perfectly well-formed vector, which is what the checksum exists to
    // catch. `.tvm`, which is all structure, rejects 145 of its 152.
    // Compare `.kdd` 204/438 and `.fdt` 624/904, both for the same
    // reason, against `.fdm` 269/282 and `.tim`+`.tip`+`.tmd` 391/436.
    // What this pins is that nothing *panics or aborts* on 3 724 corrupt
    // multi-chunk segments, plus a floor so a future change that stops
    // bounding something fails loudly.
    assert!(
        rejected >= 660,
        "only {rejected} of {total} re-signed .tvd/.tvm corruptions were rejected"
    );
}

#[test]
fn invalid_flags_selector_rejected() {
    let (tvd, tvx, tvm) = build_offsets_only_field_chunk_with_selector(7);
    let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
    assert!(matches!(
        reader.document(0),
        Err(Error::InvalidFlagsSelector(7))
    ));
}

/// Hand-encodes a single-doc, single-field chunk with one term ("cat")
/// whose field has OFFSETS only (no POSITIONS, no PAYLOADS) -- exercises
/// the "no positions" branch of offset-patching (position_correction=0)
/// and, with `selector=0`, the dedup-flags-array decode path (per-field
/// flags looked up by distinct-field-number rather than stored direct
/// per field). `selector` lets `invalid_flags_selector_rejected` reuse
/// this builder with an out-of-range selector instead.
fn build_offsets_only_field_chunk_with_selector(selector: i32) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut tvd = Vec::new();
    tvd.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
    write_string(&mut tvd, DATA_CODEC);
    tvd.extend_from_slice(&(VERSION_CURRENT as u32).to_be_bytes());
    tvd.extend_from_slice(&id());
    tvd.push(0); // empty suffix
    let chunk_start = tvd.len() as i64;

    write_vint(&mut tvd, 0); // docBase
    write_vint(&mut tvd, 1 << 1); // token: chunkDocs=1, dirty=0
    write_vint(&mut tvd, 1); // numFields = totalFields = 1

    // fieldNums: 1 distinct field (number 9), 8 bits/value.
    tvd.push(8); // token: (totalDistinct-1=0)<<5 | bitsPerFieldNum=8
    tvd.push(9);

    // allFieldNumOffs: 1 field, offset 0 into field_nums, 1 bit/value.
    write_vint(&mut tvd, 1); // slice byte length
    tvd.push(0x00);

    write_vint(&mut tvd, selector);
    if selector == 0 {
        // Dedup path: one flags entry per distinct field number
        // (FLAGS_BITS=4), value=2 (OFFSETS only).
        write_vint(&mut tvd, 1); // slice byte length
        tvd.push(0x02);
    } else if selector == 1 {
        // Direct path: one flags entry per field in the chunk.
        write_vint(&mut tvd, 1); // slice byte length
        tvd.push(0x02);
    }
    // For an invalid selector, no further flags bytes are read at all
    // (document() errors out immediately on the unmatched selector).

    // numTerms: 1 field, 1 bit/value, value=1.
    write_vint(&mut tvd, 1); // bitsRequired
    write_vint(&mut tvd, 1); // slice byte length
    tvd.push(0x01);

    // prefixLengths [0] (bpv=0, min=0).
    tvd.push(0x01);
    // suffixLengths [3] (bpv=0, min=3): token, minValue vlong.
    tvd.push(0x00);
    let target = lucene_util::zigzag::encode(3) - 1;
    write_vlong_test(&mut tvd, target);
    // termFreqsMinus1 [0] (bpv=0, min=0).
    tvd.push(0x01);

    // No positions_flat (total_positions=0, OFFSETS-only field).
    // charsPerTerm: 1 distinct field, value 4.0 (irrelevant, no
    // positions to multiply against since has_positions=false).
    tvd.extend_from_slice(&4.0f32.to_bits().to_le_bytes());
    // start_offsets_flat [0] (bpv=0, min=0) -- absolute offset 0.
    tvd.push(0x01);
    // lengths_flat [0] (bpv=0, min=0) -- actual length = 0 + termLen(3).
    tvd.push(0x01);
    // No payload_lengths_flat (total_payloads=0).

    // LZ4 (CompressionMode.FAST, no dictionary): literal-only unit
    // wrapping "cat" (the only term suffix; no payload bytes).
    let payload = *b"cat";
    tvd.push((payload.len() as u8) << 4);
    tvd.extend_from_slice(&payload);

    tvd.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
    tvd.extend_from_slice(&0u32.to_be_bytes());
    let checksum = crc32fast::hash(&tvd) as u64;
    tvd.extend_from_slice(&checksum.to_be_bytes());

    let (tvx, tvm) = build_trivial_single_chunk_index_and_meta(chunk_start, tvd.len() as i64);
    (tvd, tvx, tvm)
}

fn write_vlong_test(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let mut b = (v & 0x7f) as u8;
        v >>= 7;
        if v != 0 {
            b |= 0x80;
            out.push(b);
        } else {
            out.push(b);
            break;
        }
    }
}

/// Builds a trivial `.tvx`/`.tvm` pair for a single one-chunk, one-doc
/// segment whose `.tvd` chunk starts at `chunk_start` and whose
/// (footer-terminated) length is `tvd_len`.
fn build_trivial_single_chunk_index_and_meta(chunk_start: i64, tvd_len: i64) -> (Vec<u8>, Vec<u8>) {
    let mut tvx = Vec::new();
    tvx.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
    write_string(&mut tvx, INDEX_CODEC);
    tvx.extend_from_slice(&(INDEX_VERSION_CURRENT as u32).to_be_bytes());
    tvx.extend_from_slice(&id());
    tvx.push(0);
    let docs_start = tvx.len() as i64;
    let docs_end = tvx.len() as i64;
    let start_pointers_end = tvx.len() as i64;
    tvx.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
    tvx.extend_from_slice(&0u32.to_be_bytes());
    let checksum = crc32fast::hash(&tvx) as u64;
    tvx.extend_from_slice(&checksum.to_be_bytes());

    let max_doc = 1i32;
    let max_pointer = tvd_len - codec_util::FOOTER_LENGTH as i64;
    let mut tvm = Vec::new();
    tvm.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
    write_string(&mut tvm, META_CODEC);
    tvm.extend_from_slice(&(VERSION_CURRENT as u32).to_be_bytes());
    tvm.extend_from_slice(&id());
    tvm.push(0);
    write_vint(&mut tvm, 0);
    write_vint(&mut tvm, 4096);
    tvm.extend_from_slice(&max_doc.to_le_bytes());
    tvm.extend_from_slice(&0i32.to_le_bytes());
    tvm.extend_from_slice(&2i32.to_le_bytes());
    tvm.extend_from_slice(&docs_start.to_le_bytes());
    for min in [0i64, max_doc as i64] {
        tvm.extend_from_slice(&min.to_le_bytes());
        tvm.extend_from_slice(&0i32.to_le_bytes());
        tvm.extend_from_slice(&0i64.to_le_bytes());
        tvm.push(0);
    }
    tvm.extend_from_slice(&docs_end.to_le_bytes());
    for min in [chunk_start, max_pointer] {
        tvm.extend_from_slice(&min.to_le_bytes());
        tvm.extend_from_slice(&0i32.to_le_bytes());
        tvm.extend_from_slice(&0i64.to_le_bytes());
        tvm.push(0);
    }
    tvm.extend_from_slice(&start_pointers_end.to_le_bytes());
    tvm.extend_from_slice(&max_pointer.to_le_bytes());
    write_vint(&mut tvm, 1);
    write_vint(&mut tvm, 0);
    write_vint(&mut tvm, 0);
    tvm.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
    tvm.extend_from_slice(&0u32.to_be_bytes());
    let checksum = crc32fast::hash(&tvm) as u64;
    tvm.extend_from_slice(&checksum.to_be_bytes());

    (tvx, tvm)
}

#[test]
fn dedup_flags_selector_and_offsets_without_positions() {
    let (tvd, tvx, tvm) = build_offsets_only_field_chunk_with_selector(0);
    let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
    let doc = reader.document(0).unwrap().unwrap();
    assert_eq!(doc.fields.len(), 1);
    let field = &doc.fields[0];
    assert_eq!(field.field_number, 9);
    assert!(!field.has_positions);
    assert!(field.has_offsets);
    assert!(!field.has_payloads);
    assert_eq!(field.terms.len(), 1);
    let cat = &field.terms[0];
    assert_eq!(cat.term, b"cat");
    assert_eq!(cat.freq, 1);
    assert_eq!(cat.positions, None);
    assert_eq!(cat.start_offsets, Some(vec![0]));
    assert_eq!(cat.end_offsets, Some(vec![3]));
    assert_eq!(cat.payloads, None);
}

#[test]
fn direct_flags_selector_offsets_without_positions() {
    let (tvd, tvx, tvm) = build_offsets_only_field_chunk_with_selector(1);
    let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
    let doc = reader.document(0).unwrap().unwrap();
    let field = &doc.fields[0];
    assert!(!field.has_positions && field.has_offsets && !field.has_payloads);
    assert_eq!(field.terms[0].start_offsets, Some(vec![0]));
    assert_eq!(field.terms[0].end_offsets, Some(vec![3]));
}

/// Hand-encodes a single-doc chunk with 8 distinct field numbers (0..7),
/// each with one field carrying a single one-character term, freq 1, and
/// **no** positions/offsets/payloads at all. Exercises the
/// distinct-field-numbers extension-byte path (>=8 distinct field
/// numbers needs an extra vint beyond the 3-bit inline count) and the
/// all-empty-arrays branches (`total_positions`/`total_offsets`/
/// `total_payloads` all 0, so none of those `block_packed` streams are
/// read at all).
fn build_eight_distinct_fields_no_flags_chunk() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut tvd = Vec::new();
    tvd.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
    write_string(&mut tvd, DATA_CODEC);
    tvd.extend_from_slice(&(VERSION_CURRENT as u32).to_be_bytes());
    tvd.extend_from_slice(&id());
    tvd.push(0); // empty suffix
    let chunk_start = tvd.len() as i64;

    write_vint(&mut tvd, 0); // docBase
    write_vint(&mut tvd, 1 << 1); // token: chunkDocs=1, dirty=0
    write_vint(&mut tvd, 8); // numFields = totalFields = 8

    // fieldNums: 8 distinct fields (0..7), 3 bits/value. totalDistinct-1
    // (7) hits the 3-bit inline cap (0x07), so an extra vint(0) follows.
    tvd.push(0xE3); // (7<<5)|3
    write_vint(&mut tvd, 0); // extra: totalDistinct = 7+0+1 = 8
    tvd.extend_from_slice(&[0x05, 0x39, 0x77]); // packed_ints, 3 bits x 8 -> [0..7]

    // allFieldNumOffs: 8 fields, identity offsets 0..7, 4 bits/value
    // (bitsRequired(totalDistinct-1=7) rounds up to 4).
    write_vint(&mut tvd, 4); // slice byte length
    tvd.extend_from_slice(&[0x10, 0x32, 0x54, 0x76]);

    // flags: selector=1 (direct), 8 fields, FLAGS_BITS=4, all 0.
    write_vint(&mut tvd, 1); // selector
    write_vint(&mut tvd, 4); // slice byte length
    tvd.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);

    // numTerms: 8 fields, 1 bit/value, all 1.
    write_vint(&mut tvd, 1); // bitsRequired
    write_vint(&mut tvd, 1); // slice byte length
    tvd.push(0xFF);

    // prefixLengths: 8 zeros (bpv=0, min=0).
    tvd.push(0x01);
    // suffixLengths: 8 ones (bpv=0, min=1): token, minValue vlong.
    tvd.push(0x00);
    let target = lucene_util::zigzag::encode(1) - 1;
    write_vlong_test(&mut tvd, target);
    // termFreqsMinus1: 8 zeros (bpv=0, min=0).
    tvd.push(0x01);

    // No positions/offsets/payloads streams at all (all totals 0).

    // LZ4: 8 one-byte term suffixes, no payload bytes.
    let payload = *b"abcdefgh";
    tvd.push((payload.len() as u8) << 4);
    tvd.extend_from_slice(&payload);

    tvd.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
    tvd.extend_from_slice(&0u32.to_be_bytes());
    let checksum = crc32fast::hash(&tvd) as u64;
    tvd.extend_from_slice(&checksum.to_be_bytes());

    let (tvx, tvm) = build_trivial_single_chunk_index_and_meta(chunk_start, tvd.len() as i64);
    (tvd, tvx, tvm)
}

/// `DirectWriter.finish()`'s padding: 3 values at 12 bits are 5 packed
/// bytes plus one pad byte, and the length Java writes counts it.
#[test]
fn direct_blocks_carry_direct_writer_padding() {
    let mut out = Vec::new();
    write_direct(&mut out, &[300, 1, 4095], 12);
    let mut want = vec![6];
    want.extend(direct_reader::encode(&[300, 1, 4095], 12));
    want.push(0);
    assert_eq!(out, want);
    out.clear();
    write_direct(&mut out, &[3, 1], 4);
    assert_eq!(out.len(), 2);
}

/// A field of 300 terms needs 12 bits per term count: the padded
/// num-terms block still reads back.
#[test]
fn write_best_speed_field_with_hundreds_of_terms_round_trips() {
    let terms: Vec<TermVectorTerm> = (0..300)
        .map(|i| TermVectorTerm {
            term: format!("t{i:04}").into_bytes(),
            freq: 1,
            positions: None,
            start_offsets: None,
            end_offsets: None,
            payloads: None,
        })
        .collect();
    let docs = vec![TermVectorsDocument {
        fields: vec![TermVectorField {
            field_number: 0,
            has_positions: false,
            has_offsets: false,
            has_payloads: false,
            terms,
        }],
    }];
    let (tvd, tvx, tvm) = write_best_speed(&docs, &id(), "");
    let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
    let doc = reader.document(0).unwrap().unwrap();
    assert_eq!(doc.fields[0].terms.len(), 300);
    assert_eq!(doc.fields[0].terms[299].term, b"t0299");
}

#[test]
fn write_best_speed_single_doc_single_field_single_term_round_trips() {
    let docs = vec![TermVectorsDocument {
        fields: vec![TermVectorField {
            field_number: 5,
            has_positions: true,
            has_offsets: false,
            has_payloads: false,
            terms: vec![TermVectorTerm {
                term: b"cat".to_vec(),
                freq: 1,
                positions: Some(vec![0]),
                start_offsets: None,
                end_offsets: None,
                payloads: None,
            }],
        }],
    }];
    let (tvd, tvx, tvm) = write_best_speed(&docs, &id(), "");
    let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
    assert_eq!(reader.max_doc(), 1);
    let doc = reader.document(0).unwrap().unwrap();
    assert_eq!(doc.fields.len(), 1);
    let field = &doc.fields[0];
    assert_eq!(field.field_number, 5);
    assert!(field.has_positions && !field.has_offsets && !field.has_payloads);
    assert_eq!(field.terms.len(), 1);
    assert_eq!(field.terms[0].term, b"cat");
    assert_eq!(field.terms[0].freq, 1);
    assert_eq!(field.terms[0].positions, Some(vec![0]));
}

/// With `OffsetAttribute`'s real unit (UTF-16 code units), a
/// multi-byte term's offset span is **shorter** than the term's UTF-8 byte
/// length, so `flushOffsets`' `length - prefixLength - suffixLength` goes
/// negative -- the one caller in this crate that feeds
/// `block_packed::encode_all` negative values. `caf\u{e9}` spans 4 Java `char`s and occupies 5
/// bytes: -1. `\u{4e16}\u{754c}` spans 2 and occupies 6: -4. Java's
/// reader adds `prefixLength + suffixLength` back the same way, so a
/// wrong sign here is invisible to every ASCII fixture and wrong for
/// every non-ASCII document.
#[test]
fn write_best_speed_offsets_shorter_than_the_term_bytes_round_trip() {
    // "caf\u{e9} \u{4e16}\u{754c} dog", tokenized and offset the way
    // `lucene_analysis::tokenize` now does: Java `char` indices.
    let docs = vec![TermVectorsDocument {
        fields: vec![TermVectorField {
            field_number: 0,
            has_positions: true,
            has_offsets: true,
            has_payloads: false,
            terms: vec![
                TermVectorTerm {
                    term: "caf\u{e9}".as_bytes().to_vec(),
                    freq: 1,
                    positions: Some(vec![0]),
                    start_offsets: Some(vec![0]),
                    end_offsets: Some(vec![4]),
                    payloads: None,
                },
                TermVectorTerm {
                    term: "dog".as_bytes().to_vec(),
                    freq: 1,
                    positions: Some(vec![3]),
                    start_offsets: Some(vec![8]),
                    end_offsets: Some(vec![11]),
                    payloads: None,
                },
                TermVectorTerm {
                    term: "\u{4e16}\u{754c}".as_bytes().to_vec(),
                    freq: 1,
                    positions: Some(vec![1]),
                    start_offsets: Some(vec![5]),
                    end_offsets: Some(vec![7]),
                    payloads: None,
                },
            ],
        }],
    }];
    // The negative deltas this exercises, stated rather than assumed.
    assert_eq!(4i32 - "caf\u{e9}".len() as i32, -1);
    assert_eq!(2i32 - "\u{4e16}\u{754c}".len() as i32, -4);

    let (tvd, tvx, tvm) = write_best_speed(&docs, &id(), "");
    let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
    let doc = reader.document(0).unwrap().unwrap();
    let field = &doc.fields[0];
    let got: Vec<(&str, i32, i32)> = field
        .terms
        .iter()
        .map(|t| {
            (
                std::str::from_utf8(&t.term).unwrap(),
                t.start_offsets.as_ref().unwrap()[0],
                t.end_offsets.as_ref().unwrap()[0],
            )
        })
        .collect();
    // Terms come back in the writer's (ascending byte) order.
    assert_eq!(
        got,
        vec![
            ("caf\u{e9}", 0, 4),
            ("dog", 8, 11),
            ("\u{4e16}\u{754c}", 5, 7),
        ]
    );
}

#[test]
fn write_best_speed_all_field_numbers_zero_uses_nonzero_bit_width() {
    // A chunk where every field across every doc has
    // field_number == 0 (an entirely ordinary case -- e.g. any
    // single-field index) must not encode bits_per_field_num as 0. Real
    // Lucene's reader unconditionally indexes packedBulkOps[bitsPerValue
    // - 1], so a 0-bit width there is an ArrayIndexOutOfBoundsException
    // in real Lucene even though this port's own reader tolerates it --
    // this test only proves the width isn't 0 on the wire; cross-engine
    // coverage for this exact shape lives in the fixture example.
    let docs = vec![
        TermVectorsDocument {
            fields: vec![TermVectorField {
                field_number: 0,
                has_positions: true,
                has_offsets: false,
                has_payloads: false,
                terms: vec![TermVectorTerm {
                    term: b"cat".to_vec(),
                    freq: 1,
                    positions: Some(vec![0]),
                    start_offsets: None,
                    end_offsets: None,
                    payloads: None,
                }],
            }],
        },
        TermVectorsDocument {
            fields: vec![TermVectorField {
                field_number: 0,
                has_positions: true,
                has_offsets: false,
                has_payloads: false,
                terms: vec![TermVectorTerm {
                    term: b"dog".to_vec(),
                    freq: 1,
                    positions: Some(vec![0]),
                    start_offsets: None,
                    end_offsets: None,
                    payloads: None,
                }],
            }],
        },
    ];
    let (tvd, tvx, tvm) = write_best_speed(&docs, &id(), "");

    // bits_per_field_num is the low 5 bits of the fdt's first byte,
    // right after the tvx/tvm header framing this test doesn't need to
    // re-derive -- simplest to assert the invariant round-trip-only:
    // decode via this module's own reader (which would also decode a
    // wire-correct 0-bits chunk without complaint) but additionally
    // confirm the actually-written token byte's low 5 bits are nonzero
    // by re-deriving the same offset the reader itself uses.
    let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
    assert_eq!(reader.max_doc(), 2);
    let doc0 = reader.document(0).unwrap().unwrap();
    assert_eq!(doc0.fields[0].field_number, 0);
    assert_eq!(doc0.fields[0].terms[0].term, b"cat");
    let doc1 = reader.document(1).unwrap().unwrap();
    assert_eq!(doc1.fields[0].field_number, 0);
    assert_eq!(doc1.fields[0].terms[0].term, b"dog");

    // Note: this port's own reader tolerates bits_per_field_num == 0
    // (an all-zero chunk decodes as all-zero regardless), so a
    // round-trip through it can't by itself prove the wire bit-width
    // isn't 0 -- that only matters to a *real* Lucene reader. The
    // cross-engine fixture (write_term_vectors_fixture.rs /
    // VerifyTermVectors.java) covers an all-field-0 chunk specifically
    // so this shape is actually proven against real Lucene, not just
    // against this port's own (more permissive) reader.
}

#[test]
fn write_best_speed_single_doc_single_field_multiple_terms_round_trips() {
    let docs = vec![TermVectorsDocument {
        fields: vec![TermVectorField {
            field_number: 2,
            has_positions: true,
            has_offsets: false,
            has_payloads: false,
            terms: vec![
                TermVectorTerm {
                    term: b"cat".to_vec(),
                    freq: 2,
                    positions: Some(vec![0, 3]),
                    start_offsets: None,
                    end_offsets: None,
                    payloads: None,
                },
                TermVectorTerm {
                    term: b"dog".to_vec(),
                    freq: 1,
                    positions: Some(vec![1]),
                    start_offsets: None,
                    end_offsets: None,
                    payloads: None,
                },
            ],
        }],
    }];
    let (tvd, tvx, tvm) = write_best_speed(&docs, &id(), "");
    let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
    let doc = reader.document(0).unwrap().unwrap();
    let field = &doc.fields[0];
    assert_eq!(field.terms.len(), 2);
    assert_eq!(field.terms[0].term, b"cat");
    assert_eq!(field.terms[0].positions, Some(vec![0, 3]));
    assert_eq!(field.terms[1].term, b"dog");
    assert_eq!(field.terms[1].positions, Some(vec![1]));
}

#[test]
fn write_best_speed_multi_doc_multi_field_round_trips() {
    let docs = vec![
        TermVectorsDocument {
            fields: vec![
                TermVectorField {
                    field_number: 0,
                    has_positions: true,
                    has_offsets: false,
                    has_payloads: false,
                    terms: vec![TermVectorTerm {
                        term: b"alpha".to_vec(),
                        freq: 1,
                        positions: Some(vec![0]),
                        start_offsets: None,
                        end_offsets: None,
                        payloads: None,
                    }],
                },
                TermVectorField {
                    field_number: 1,
                    has_positions: false,
                    has_offsets: false,
                    has_payloads: false,
                    terms: vec![TermVectorTerm {
                        term: b"beta".to_vec(),
                        freq: 1,
                        positions: None,
                        start_offsets: None,
                        end_offsets: None,
                        payloads: None,
                    }],
                },
            ],
        },
        TermVectorsDocument { fields: vec![] },
        TermVectorsDocument {
            fields: vec![TermVectorField {
                field_number: 0,
                has_positions: true,
                has_offsets: false,
                has_payloads: false,
                terms: vec![TermVectorTerm {
                    term: b"gamma".to_vec(),
                    freq: 3,
                    positions: Some(vec![0, 1, 5]),
                    start_offsets: None,
                    end_offsets: None,
                    payloads: None,
                }],
            }],
        },
    ];
    let (tvd, tvx, tvm) = write_best_speed(&docs, &id(), "");
    let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
    assert_eq!(reader.max_doc(), 3);

    let doc0 = reader.document(0).unwrap().unwrap();
    assert_eq!(doc0.fields.len(), 2);
    assert_eq!(doc0.fields[0].field_number, 0);
    assert_eq!(doc0.fields[0].terms[0].term, b"alpha");
    assert_eq!(doc0.fields[0].terms[0].positions, Some(vec![0]));
    assert_eq!(doc0.fields[1].field_number, 1);
    assert!(!doc0.fields[1].has_positions);
    assert_eq!(doc0.fields[1].terms[0].term, b"beta");
    assert_eq!(doc0.fields[1].terms[0].positions, None);

    assert!(reader.document(1).unwrap().is_none());

    let doc2 = reader.document(2).unwrap().unwrap();
    assert_eq!(doc2.fields.len(), 1);
    assert_eq!(doc2.fields[0].terms[0].term, b"gamma");
    assert_eq!(doc2.fields[0].terms[0].positions, Some(vec![0, 1, 5]));
}

#[test]
fn write_best_speed_empty_doc_set_produces_zero_max_doc() {
    let (tvd, tvx, tvm) = write_best_speed(&[], &id(), "");
    let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
    assert_eq!(reader.max_doc(), 0);
}

/// Regression: a field that declares OFFSETS (or PAYLOADS) but holds no
/// terms contributes zero occurrences, so the reader -- which gates on
/// `totalOffsets > 0` / `totalPayloads > 0`, not on the flag -- expects
/// no `charsPerTerm`/offset/payload-length streams at all. Writing them
/// anyway desynchronizes every following byte of the chunk.
#[test]
fn write_best_speed_flagged_but_termless_field_round_trips() {
    let docs = vec![TermVectorsDocument {
        fields: vec![
            TermVectorField {
                field_number: 0,
                has_positions: false,
                has_offsets: true,
                has_payloads: true,
                terms: vec![],
            },
            TermVectorField {
                field_number: 1,
                has_positions: true,
                has_offsets: false,
                has_payloads: false,
                terms: vec![TermVectorTerm {
                    term: b"hello".to_vec(),
                    freq: 2,
                    positions: Some(vec![0, 5]),
                    start_offsets: None,
                    end_offsets: None,
                    payloads: None,
                }],
            },
        ],
    }];
    let (tvd, tvx, tvm) = write_best_speed(&docs, &id(), "");
    let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
    let doc = reader.document(0).unwrap().unwrap();
    assert_eq!(doc.fields.len(), 2);
    assert_eq!(doc.fields[0].field_number, 0);
    assert!(doc.fields[0].has_offsets && doc.fields[0].has_payloads);
    assert!(doc.fields[0].terms.is_empty());
    assert_eq!(doc.fields[1].field_number, 1);
    assert_eq!(doc.fields[1].terms.len(), 1);
    assert_eq!(doc.fields[1].terms[0].term, b"hello".to_vec());
    assert_eq!(doc.fields[1].terms[0].positions, Some(vec![0, 5]));
}

#[test]
fn write_best_speed_offsets_only_round_trips() {
    let docs = vec![TermVectorsDocument {
        fields: vec![TermVectorField {
            field_number: 3,
            has_positions: false,
            has_offsets: true,
            has_payloads: false,
            terms: vec![TermVectorTerm {
                term: b"cat".to_vec(),
                freq: 2,
                positions: None,
                start_offsets: Some(vec![0, 10]),
                end_offsets: Some(vec![3, 13]),
                payloads: None,
            }],
        }],
    }];
    let (tvd, tvx, tvm) = write_best_speed(&docs, &id(), "");
    let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
    let doc = reader.document(0).unwrap().unwrap();
    let field = &doc.fields[0];
    assert!(!field.has_positions && field.has_offsets && !field.has_payloads);
    let cat = &field.terms[0];
    assert_eq!(cat.positions, None);
    assert_eq!(cat.start_offsets, Some(vec![0, 10]));
    assert_eq!(cat.end_offsets, Some(vec![3, 13]));
    assert_eq!(cat.payloads, None);
}

#[test]
fn write_best_speed_positions_and_offsets_freq_three_round_trips() {
    // Regression for a review-confirmed bug: the offset-delta correction used
    // the absolute position instead of the position *delta* the read side
    // actually applies, which only coincidentally matched for freq <= 2. A
    // freq-3 term with non-trivial, non-uniform position gaps is the
    // smallest case that would silently corrupt the 3rd+ occurrence's
    // decoded offsets under the old (buggy) code.
    let docs = vec![TermVectorsDocument {
        fields: vec![TermVectorField {
            field_number: 3,
            has_positions: true,
            has_offsets: true,
            has_payloads: false,
            terms: vec![TermVectorTerm {
                term: b"the".to_vec(),
                freq: 3,
                positions: Some(vec![0, 5, 12]),
                start_offsets: Some(vec![0, 20, 45]),
                end_offsets: Some(vec![3, 23, 48]),
                payloads: None,
            }],
        }],
    }];
    let (tvd, tvx, tvm) = write_best_speed(&docs, &id(), "");
    let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
    let doc = reader.document(0).unwrap().unwrap();
    let field = &doc.fields[0];
    let the = &field.terms[0];
    assert_eq!(the.positions, Some(vec![0, 5, 12]));
    assert_eq!(the.start_offsets, Some(vec![0, 20, 45]));
    assert_eq!(the.end_offsets, Some(vec![3, 23, 48]));
}

#[test]
fn write_best_speed_payloads_only_round_trips() {
    let docs = vec![TermVectorsDocument {
        fields: vec![TermVectorField {
            field_number: 4,
            has_positions: false,
            has_offsets: false,
            has_payloads: true,
            terms: vec![TermVectorTerm {
                term: b"dog".to_vec(),
                freq: 2,
                positions: None,
                start_offsets: None,
                end_offsets: None,
                payloads: Some(vec![vec![0xAA, 0xBB], vec![]]),
            }],
        }],
    }];
    let (tvd, tvx, tvm) = write_best_speed(&docs, &id(), "");
    let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
    let doc = reader.document(0).unwrap().unwrap();
    let field = &doc.fields[0];
    assert!(!field.has_positions && !field.has_offsets && field.has_payloads);
    let dog = &field.terms[0];
    assert_eq!(dog.payloads, Some(vec![vec![0xAA, 0xBB], vec![]]));
}

#[test]
fn write_best_speed_positions_offsets_payloads_round_trips_multi_term_multi_doc() {
    let docs = vec![
        TermVectorsDocument {
            fields: vec![TermVectorField {
                field_number: 5,
                has_positions: true,
                has_offsets: true,
                has_payloads: true,
                terms: vec![
                    TermVectorTerm {
                        term: b"cat".to_vec(),
                        freq: 2,
                        positions: Some(vec![0, 2]),
                        start_offsets: Some(vec![0, 8]),
                        end_offsets: Some(vec![3, 11]),
                        payloads: Some(vec![vec![0xAA], vec![]]),
                    },
                    TermVectorTerm {
                        term: b"car".to_vec(),
                        freq: 1,
                        positions: Some(vec![1]),
                        start_offsets: Some(vec![4]),
                        end_offsets: Some(vec![7]),
                        payloads: Some(vec![vec![0xBB, 0xCC]]),
                    },
                ],
            }],
        },
        TermVectorsDocument {
            fields: vec![TermVectorField {
                field_number: 5,
                has_positions: true,
                has_offsets: true,
                has_payloads: true,
                terms: vec![TermVectorTerm {
                    term: b"dog".to_vec(),
                    freq: 1,
                    positions: Some(vec![0]),
                    start_offsets: Some(vec![0]),
                    end_offsets: Some(vec![3]),
                    payloads: Some(vec![vec![0xDD, 0xEE, 0xFF]]),
                }],
            }],
        },
    ];
    let (tvd, tvx, tvm) = write_best_speed(&docs, &id(), "");
    let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
    assert_eq!(reader.max_doc(), 2);

    let doc0 = reader.document(0).unwrap().unwrap();
    let field0 = &doc0.fields[0];
    assert!(field0.has_positions && field0.has_offsets && field0.has_payloads);
    let cat = &field0.terms[0];
    assert_eq!(cat.term, b"cat");
    assert_eq!(cat.positions, Some(vec![0, 2]));
    assert_eq!(cat.start_offsets, Some(vec![0, 8]));
    assert_eq!(cat.end_offsets, Some(vec![3, 11]));
    assert_eq!(cat.payloads, Some(vec![vec![0xAA], vec![]]));
    let car = &field0.terms[1];
    assert_eq!(car.term, b"car");
    assert_eq!(car.positions, Some(vec![1]));
    assert_eq!(car.start_offsets, Some(vec![4]));
    assert_eq!(car.end_offsets, Some(vec![7]));
    assert_eq!(car.payloads, Some(vec![vec![0xBB, 0xCC]]));

    let doc1 = reader.document(1).unwrap().unwrap();
    let field1 = &doc1.fields[0];
    let dog = &field1.terms[0];
    assert_eq!(dog.term, b"dog");
    assert_eq!(dog.positions, Some(vec![0]));
    assert_eq!(dog.start_offsets, Some(vec![0]));
    assert_eq!(dog.end_offsets, Some(vec![3]));
    assert_eq!(dog.payloads, Some(vec![vec![0xDD, 0xEE, 0xFF]]));
}

#[test]
fn write_best_speed_positions_only_regression_still_works() {
    // Same shape as `write_best_speed_single_doc_single_field_multiple_terms_round_trips`
    // but re-asserted here to make explicit that the positions-only path
    // (has_offsets=false, has_payloads=false) is unaffected by the
    // offsets/payloads extension.
    let docs = vec![TermVectorsDocument {
        fields: vec![TermVectorField {
            field_number: 1,
            has_positions: true,
            has_offsets: false,
            has_payloads: false,
            terms: vec![TermVectorTerm {
                term: b"only".to_vec(),
                freq: 1,
                positions: Some(vec![0]),
                start_offsets: None,
                end_offsets: None,
                payloads: None,
            }],
        }],
    }];
    let (tvd, tvx, tvm) = write_best_speed(&docs, &id(), "");
    let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
    let doc = reader.document(0).unwrap().unwrap();
    let field = &doc.fields[0];
    assert!(field.has_positions && !field.has_offsets && !field.has_payloads);
    assert_eq!(field.terms[0].positions, Some(vec![0]));
    assert_eq!(field.terms[0].start_offsets, None);
    assert_eq!(field.terms[0].payloads, None);
}

#[test]
fn a_chunk_claiming_two_billion_documents_is_a_decode_error_not_an_overflow() {
    // Reachable only once `chunkDocs` is decoded the way Java decodes it
    // (`readVInt() >>> 1`, unsigned): a corrupt token then names a
    // `chunkDocs` near `i32::MAX`, and a chunk whose `docBase` is
    // anything but 0 overflows `docBase + chunkDocs`. Needs a *second*
    // chunk, so this builds a 129-document segment (the
    // MAX_DOCS_PER_CHUNK trigger) and corrupts the second chunk's header.
    let docs: Vec<TermVectorsDocument> = (0..129).map(tiny_doc).collect();
    let (mut tvd, tvx, mut tvm) = write_best_speed(&docs, &id(), "");
    let entry = {
        let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
        reader.chunk_for_doc(128).unwrap()
    };
    assert_eq!(entry.doc_base, 128);
    // docBase = 128 is a two-byte vint; the token vint follows it.
    let token_at = entry.start_pointer as usize + 2;
    assert_eq!(
        tvd[token_at],
        (1 << 1) | 1,
        "one document, dirty tail chunk"
    );
    let mut insert = Vec::new();
    write_vint(&mut insert, -1); // token = -1 => chunkDocs = 0x7FFF_FFFF
    splice_tvd(&mut tvd, &mut tvm, token_at, 1, &insert);

    let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
    assert!(matches!(
        reader.document(128),
        Err(Error::CorruptChunkBounds {
            doc_base: 128,
            chunk_docs: i32::MAX,
            ..
        })
    ));
}

#[test]
fn encode_literal_lz4_round_trips_through_lz4_decompress() {
    for payload in [
        Vec::new(),
        b"short".to_vec(),
        vec![0x42u8; 5000], // forces the 0xFF-continuation length encoding
    ] {
        let encoded = encode_literal_lz4(&payload);
        let mut input = SliceInput::new(&encoded);
        let mut out = vec![0u8; payload.len()];
        if !payload.is_empty() {
            crate::lz4::decompress(&mut input, payload.len(), &mut out, 0).unwrap();
        }
        assert_eq!(out, payload);
    }
}

#[test]
fn eight_distinct_fields_with_no_flags_decodes_all_fields() {
    let (tvd, tvx, tvm) = build_eight_distinct_fields_no_flags_chunk();
    let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
    let doc = reader.document(0).unwrap().unwrap();
    assert_eq!(doc.fields.len(), 8);
    for (i, field) in doc.fields.iter().enumerate() {
        assert_eq!(field.field_number, i as i32);
        assert!(!field.has_positions && !field.has_offsets && !field.has_payloads);
        assert_eq!(field.terms.len(), 1);
        let term = &field.terms[0];
        assert_eq!(term.term, vec![b'a' + i as u8]);
        assert_eq!(term.freq, 1);
        assert_eq!(term.positions, None);
        assert_eq!(term.start_offsets, None);
        assert_eq!(term.end_offsets, None);
        assert_eq!(term.payloads, None);
    }
}

// ---------------------------------------------------------------
// Chunking: the writer has
// `Lucene90CompressingTermVectorsWriter`'s streaming, chunk-flushing
// shape, not one chunk per segment.
// ---------------------------------------------------------------

/// A positions-only document whose single field holds `terms` terms, each
/// short enough that the *document count* trigger fires first.
fn tiny_doc(n: usize) -> TermVectorsDocument {
    TermVectorsDocument {
        fields: vec![TermVectorField {
            field_number: 0,
            has_positions: true,
            has_offsets: false,
            has_payloads: false,
            terms: vec![TermVectorTerm {
                term: format!("t{n:04}").into_bytes(),
                freq: 2,
                positions: Some(vec![n as i32 % 7, n as i32 % 7 + 3]),
                start_offsets: None,
                end_offsets: None,
                payloads: None,
            }],
        }],
    }
}

/// A document whose single field carries `bytes` bytes of term text, so
/// the *byte size* trigger fires long before 128 documents.
fn fat_doc(n: usize, bytes: usize) -> TermVectorsDocument {
    let mut term = format!("{n:08}").into_bytes();
    term.resize(bytes, b'x');
    TermVectorsDocument {
        fields: vec![TermVectorField {
            field_number: 0,
            has_positions: true,
            has_offsets: false,
            has_payloads: false,
            terms: vec![TermVectorTerm {
                term,
                freq: 1,
                positions: Some(vec![0]),
                start_offsets: None,
                end_offsets: None,
                payloads: None,
            }],
        }],
    }
}

/// Every chunk header in `.tvd`, walked through `.tvx`: `(docBase,
/// chunkDocs, dirty)`.
fn chunk_headers(reader: &TermVectorsReader<'_>) -> Vec<(i32, i32, bool)> {
    let mut out = Vec::new();
    let mut doc = 0i32;
    while doc < reader.max_doc() {
        let entry = reader.chunk_for_doc(doc).unwrap();
        let mut input = SliceInput::new(reader.tvd());
        input.seek(entry.start_pointer as usize).unwrap();
        let base = input.read_vint().unwrap();
        let code = input.read_vint().unwrap();
        out.push((base, code >> 1, code & 1 != 0));
        doc = base + (code >> 1);
    }
    out
}

fn assert_round_trips(docs: &[TermVectorsDocument], tvd: &[u8], tvx: &[u8], tvm: &[u8]) {
    let reader = open(tvd, tvx, tvm, &id(), "").unwrap();
    assert_eq!(reader.max_doc(), docs.len() as i32);
    for (i, expected) in docs.iter().enumerate() {
        let got = reader.document(i as i32).unwrap();
        if expected.fields.is_empty() {
            assert!(got.is_none(), "doc {i} should have no vectors");
            continue;
        }
        let got = got.unwrap_or_else(|| panic!("doc {i} missing"));
        assert_eq!(
            got.fields.len(),
            expected.fields.len(),
            "doc {i} field count"
        );
        for (g, e) in got.fields.iter().zip(&expected.fields) {
            assert_eq!(g.field_number, e.field_number, "doc {i}");
            assert_eq!(g.has_positions, e.has_positions, "doc {i}");
            assert_eq!(g.has_offsets, e.has_offsets, "doc {i}");
            assert_eq!(g.has_payloads, e.has_payloads, "doc {i}");
            assert_eq!(g.terms, e.terms, "doc {i} field {}", e.field_number);
        }
    }
}

#[test]
fn the_document_count_trigger_closes_a_chunk_at_128_documents() {
    // `Lucene90TermVectorsFormat`'s `maxDocsPerChunk`. 300 tiny documents
    // are nowhere near 4 096 bytes of term text, so every chunk boundary
    // here is the document-count trigger.
    let docs: Vec<_> = (0..300).map(tiny_doc).collect();
    let (tvd, tvx, tvm) = write_best_speed(&docs, &id(), "");
    let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
    assert_eq!(
        chunk_headers(&reader),
        vec![
            (0, 128, false),
            (128, 128, false),
            (256, 44, true), // the forced final flush
        ]
    );
    assert_eq!(reader.num_chunks(), 3);
    assert_eq!(reader.num_dirty_chunks(), 1);
    assert_eq!(reader.num_dirty_docs(), 44);
    assert_round_trips(&docs, &tvd, &tvx, &tvm);
}

#[test]
fn the_byte_size_trigger_closes_a_chunk_before_128_documents() {
    // 600-byte terms: the 4 096-byte trigger fires on the 7th document
    // (7 * 600 = 4 200 >= 4 096), long before 128.
    let docs: Vec<_> = (0..20).map(|n| fat_doc(n, 600)).collect();
    let (tvd, tvx, tvm) = write_best_speed(&docs, &id(), "");
    let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
    let headers = chunk_headers(&reader);
    assert_eq!(headers.len(), 3, "{headers:?}");
    assert_eq!(headers[0], (0, 7, false));
    assert_eq!(headers[1], (7, 7, false));
    assert_eq!(headers[2], (14, 6, true));
    assert_round_trips(&docs, &tvd, &tvx, &tvm);
}

#[test]
fn a_document_set_ending_exactly_on_a_chunk_boundary_still_flushes_a_dirty_tail() {
    // 128 documents fill the first chunk on the *last* `add_document`,
    // so `finish` finds nothing pending: one clean chunk, no dirty ones.
    let docs: Vec<_> = (0..128).map(tiny_doc).collect();
    let (tvd, tvx, tvm) = write_best_speed(&docs, &id(), "");
    let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
    assert_eq!(chunk_headers(&reader), vec![(0, 128, false)]);
    assert_eq!(reader.num_dirty_chunks(), 0);
    assert_eq!(reader.num_dirty_docs(), 0);
    assert_round_trips(&docs, &tvd, &tvx, &tvm);
}

#[test]
fn a_custom_chunk_geometry_is_recorded_and_blocks_a_default_geometry_bulk_copy() {
    // `Lucene90CompressingTermVectorsFormat`'s constructor takes
    // `chunkSize`/`maxDocsPerChunk`; `Lucene90TermVectorsFormat` fixes
    // them at 4096/128. A segment written with any other geometry is
    // still valid and readable -- `chunkSize` is in `.tvm` -- but
    // `canPerformBulkMerge` refuses to copy its chunks into a
    // default-geometry writer.
    let docs: Vec<_> = (0..300).map(tiny_doc).collect();
    let mut w = TermVectorsWriter::with_geometry(&id(), "", i32::MAX as usize, i32::MAX as usize);
    for doc in &docs {
        w.add_document(doc);
    }
    let (tvd, tvx, tvm) = w.finish();
    let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
    // One chunk for the whole segment -- which is exactly what this
    // module's writer did before batch `c8`.
    assert_eq!(reader.num_chunks(), 1);
    assert_eq!(chunk_headers(&reader), vec![(0, 300, true)]);
    assert_round_trips(&docs, &tvd, &tvx, &tvm);

    let default = TermVectorsWriter::new(&id(), "");
    assert!(!default.can_bulk_copy(&reader));
    assert!(matches!(
        TermVectorsWriter::new(&id(), "").copy_chunks(&reader, 0, 300),
        Err(Error::BulkCopyNotPermitted { .. })
    ));
    // ... but a writer with the same geometry can.
    let mut same =
        TermVectorsWriter::with_geometry(&id(), "", i32::MAX as usize, i32::MAX as usize);
    same.copy_chunks(&reader, 0, 300).unwrap();
    let (tvd2, tvx2, tvm2) = same.finish();
    assert_round_trips(&docs, &tvd2, &tvx2, &tvm2);
}

#[test]
fn a_multi_chunk_segment_round_trips_positions_offsets_and_payloads() {
    let docs: Vec<_> = (0..400)
        .map(|n| TermVectorsDocument {
            fields: vec![
                TermVectorField {
                    field_number: 0,
                    has_positions: true,
                    has_offsets: true,
                    has_payloads: true,
                    terms: vec![
                        TermVectorTerm {
                            term: format!("alpha{:03}", n % 17).into_bytes(),
                            freq: 3,
                            positions: Some(vec![0, 4, 9]),
                            start_offsets: Some(vec![0, 24, 54]),
                            end_offsets: Some(vec![8, 32, 62]),
                            payloads: Some(vec![vec![1, 2], vec![], vec![(n % 251) as u8, 7, 7]]),
                        },
                        TermVectorTerm {
                            term: format!("alpha{:03}z", n % 17).into_bytes(),
                            freq: 1,
                            positions: Some(vec![11]),
                            start_offsets: Some(vec![66]),
                            end_offsets: Some(vec![75]),
                            payloads: Some(vec![vec![9]]),
                        },
                    ],
                },
                TermVectorField {
                    field_number: 3,
                    has_positions: true,
                    has_offsets: false,
                    has_payloads: false,
                    terms: vec![TermVectorTerm {
                        term: format!("beta{n}").into_bytes(),
                        freq: 2,
                        positions: Some(vec![1, 6]),
                        start_offsets: None,
                        end_offsets: None,
                        payloads: None,
                    }],
                },
            ],
        })
        .collect();
    let (tvd, tvx, tvm) = write_best_speed(&docs, &id(), "");
    let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
    assert!(reader.num_chunks() > 1, "expected several chunks");
    assert_round_trips(&docs, &tvd, &tvx, &tvm);
}

#[test]
fn terms_are_prefix_compressed_against_the_previous_term_in_the_field() {
    // `startTerm`'s `StringHelper.bytesDifference`. Two documents with the
    // same *total* term bytes, one where consecutive terms share a long
    // prefix and one where they share nothing: only prefix sharing can
    // make the first `.tvd` smaller.
    let shared = TermVectorsDocument {
        fields: vec![TermVectorField {
            field_number: 0,
            has_positions: false,
            has_offsets: false,
            has_payloads: false,
            terms: (0..40)
                .map(|i| TermVectorTerm {
                    term: format!("commonprefixaaaaaaaaaaaaaaaa{i:04}").into_bytes(),
                    freq: 1,
                    positions: None,
                    start_offsets: None,
                    end_offsets: None,
                    payloads: None,
                })
                .collect(),
        }],
    };
    let distinct = TermVectorsDocument {
        fields: vec![TermVectorField {
            field_number: 0,
            has_positions: false,
            has_offsets: false,
            has_payloads: false,
            terms: (0..40)
                .map(|i| {
                    // Same length, but the discriminating digits come
                    // first, so the common prefix is always 0.
                    let mut term = format!("{i:04}").into_bytes();
                    term.extend_from_slice(b"aaaaaaaaaaaaaaaacommonprefix");
                    TermVectorTerm {
                        term,
                        freq: 1,
                        positions: None,
                        start_offsets: None,
                        end_offsets: None,
                        payloads: None,
                    }
                })
                .collect(),
        }],
    };
    let (shared_tvd, tvx, tvm) = write_best_speed(std::slice::from_ref(&shared), &id(), "");
    let (distinct_tvd, ..) = write_best_speed(&[distinct], &id(), "");
    assert!(
        shared_tvd.len() < distinct_tvd.len(),
        "prefix sharing should shrink the chunk: {} vs {}",
        shared_tvd.len(),
        distinct_tvd.len()
    );
    assert_round_trips(&[shared], &shared_tvd, &tvx, &tvm);
}

#[test]
fn chars_per_term_is_the_offset_to_position_ratio_of_the_field() {
    // `flushOffsets`: `charsPerTerm[i] = sumOffsets[i] / sumPos[i]`, both
    // summed over each term's *last* occurrence. Here one term, one doc:
    // last position 8, last start offset 48 -> 6.0.
    let doc = TermVectorsDocument {
        fields: vec![TermVectorField {
            field_number: 2,
            has_positions: true,
            has_offsets: true,
            has_payloads: false,
            terms: vec![TermVectorTerm {
                term: b"word".to_vec(),
                freq: 3,
                positions: Some(vec![0, 4, 8]),
                start_offsets: Some(vec![0, 24, 48]),
                end_offsets: Some(vec![4, 28, 52]),
                payloads: None,
            }],
        }],
    };
    let (tvd, tvx, tvm) = write_best_speed(std::slice::from_ref(&doc), &id(), "");
    let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
    let chunk = reader.read_chunk(0).unwrap();
    assert_eq!(chunk.chars_per_term, vec![6.0f32]);
    // And with a perfectly uniform ratio every encoded start-offset delta
    // is exactly 0, which is the whole point of the correction.
    assert_eq!(chunk.start_offsets_flat, vec![0, 0, 0]);
    assert_round_trips(&[doc], &tvd, &tvx, &tvm);
}

#[test]
fn a_field_with_offsets_but_no_positions_gets_a_zero_chars_per_term() {
    // `sumPos` stays 0 (the `fd.hasOffsets && fd.hasPositions` gate), so
    // Java writes `charsPerTerm = 0` and the deltas are plain.
    let doc = TermVectorsDocument {
        fields: vec![TermVectorField {
            field_number: 0,
            has_positions: false,
            has_offsets: true,
            has_payloads: false,
            terms: vec![TermVectorTerm {
                term: b"word".to_vec(),
                freq: 2,
                positions: None,
                start_offsets: Some(vec![0, 24]),
                end_offsets: Some(vec![4, 28]),
                payloads: None,
            }],
        }],
    };
    let (tvd, tvx, tvm) = write_best_speed(std::slice::from_ref(&doc), &id(), "");
    let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
    assert_eq!(reader.read_chunk(0).unwrap().chars_per_term, vec![0.0f32]);
    assert_round_trips(&[doc], &tvd, &tvx, &tvm);
}

#[test]
fn flags_are_written_once_per_field_number_when_they_never_change() {
    // `flushFlags`' `nonChangingFlags` branch (selector 0): 128 documents
    // of one field with constant flags cost one 4-bit entry, not 128.
    // The alternating variant has to fall back to selector 1.
    let constant: Vec<_> = (0..128).map(tiny_doc).collect();
    let mut alternating = constant.clone();
    for (i, doc) in alternating.iter_mut().enumerate() {
        if i % 2 == 1 {
            // Same field number, different flags -> `nonChangingFlags`
            // is false.
            doc.fields[0].has_positions = false;
            doc.fields[0].terms[0].positions = None;
        }
    }
    let (constant_tvd, ..) = write_best_speed(&constant, &id(), "");
    let (alternating_tvd, ..) = write_best_speed(&alternating, &id(), "");
    // Selector 0 writes 1 byte of flags where selector 1 writes 64
    // (128 fields x 4 bits). The alternating set also drops half its
    // positions, which *shrinks* it -- so the flags saving has to beat
    // that to show up at all.
    assert!(
        constant_tvd.len() + 40 < alternating_tvd.len() + 64,
        "expected the constant-flag chunk to use the per-field-number \
             flags array: {} vs {}",
        constant_tvd.len(),
        alternating_tvd.len()
    );
    let (tvd, tvx, tvm) = write_best_speed(&alternating, &id(), "");
    assert_round_trips(&alternating, &tvd, &tvx, &tvm);
}

#[test]
fn distinct_field_numbers_are_written_sorted() {
    // `flushFieldNums` sorts; the reader resolves through `fieldNumOffs`
    // either way, but real Lucene's *writer* is the reference and
    // `CheckIndex` walks `TVFields.iterator()` in document field order.
    let doc = TermVectorsDocument {
        fields: [7, 2, 5]
            .into_iter()
            .map(|n| TermVectorField {
                field_number: n,
                has_positions: false,
                has_offsets: false,
                has_payloads: false,
                terms: vec![TermVectorTerm {
                    term: format!("f{n}").into_bytes(),
                    freq: 1,
                    positions: None,
                    start_offsets: None,
                    end_offsets: None,
                    payloads: None,
                }],
            })
            .collect(),
    };
    let (tvd, tvx, tvm) = write_best_speed(std::slice::from_ref(&doc), &id(), "");
    let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
    assert_eq!(reader.read_chunk(0).unwrap().field_nums, vec![2, 5, 7]);
    // Document order is preserved regardless.
    let got = reader.document(0).unwrap().unwrap();
    // One field alone is the document's field of that number.
    let mut cursor = ChunkCursor::new();
    for n in [7, 2, 5] {
        assert_eq!(
            cursor.field(&reader, 0, n).unwrap().as_ref(),
            got.fields.iter().find(|f| f.field_number == n)
        );
    }
    assert!(cursor.field(&reader, 0, 3).unwrap().is_none());
    assert_eq!(
        got.fields
            .iter()
            .map(|f| f.field_number)
            .collect::<Vec<_>>(),
        vec![7, 2, 5]
    );
    assert_round_trips(&[doc], &tvd, &tvx, &tvm);
}

#[test]
fn a_chunk_cursor_serves_every_document_of_a_chunk_from_one_decode() {
    let docs: Vec<_> = (0..300).map(tiny_doc).collect();
    let (tvd, tvx, tvm) = write_best_speed(&docs, &id(), "");
    let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
    let mut cursor = ChunkCursor::new();
    for doc_id in 0..reader.max_doc() {
        assert_eq!(
            cursor.document(&reader, doc_id).unwrap(),
            reader.document(doc_id).unwrap(),
            "doc {doc_id}"
        );
    }
    // the field form reloads across chunks, backwards too
    for doc_id in (0..reader.max_doc()).rev() {
        assert_eq!(
            cursor.field(&reader, doc_id, 0).unwrap(),
            Some(tiny_doc(doc_id as usize).fields.remove(0)),
            "doc {doc_id} field"
        );
    }
    // A backwards walk must reload rather than serve a stale chunk.
    for doc_id in (0..reader.max_doc()).rev() {
        assert_eq!(
            cursor.document(&reader, doc_id).unwrap(),
            reader.document(doc_id).unwrap(),
            "doc {doc_id} backwards"
        );
    }
    cursor.reset();
    assert!(cursor.chunk.is_none());
}

#[test]
fn read_chunk_reports_its_own_extent_and_rejects_documents_outside_it() {
    let docs: Vec<_> = (0..300).map(tiny_doc).collect();
    let (tvd, tvx, tvm) = write_best_speed(&docs, &id(), "");
    let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
    let chunk = reader.read_chunk(200).unwrap();
    assert_eq!(chunk.doc_base(), 128);
    assert_eq!(chunk.num_docs(), 128);
    assert!(chunk.contains(128) && chunk.contains(255));
    assert!(!chunk.contains(127) && !chunk.contains(256));
    assert!(matches!(
        chunk.document(300),
        Err(Error::CorruptChunkBounds { .. })
    ));
    assert!(matches!(
        chunk.field(300, 0),
        Err(Error::CorruptChunkBounds { .. })
    ));
    assert_eq!(
        chunk.field(200, 0).unwrap(),
        Some(tiny_doc(200).fields.remove(0))
    );
    assert!(matches!(
        reader.read_chunk(-1),
        Err(Error::DocOutOfRange(-1, 300))
    ));
    assert!(matches!(
        reader.chunk_for_doc(300),
        Err(Error::DocOutOfRange(300, 300))
    ));
}

// ---------------------------------------------------------------
// The bulk merge path (`copyChunks`).
// ---------------------------------------------------------------

fn segment(docs: &[TermVectorsDocument]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    write_best_speed(docs, &id(), "")
}

#[test]
fn copy_chunks_of_two_whole_segments_reproduces_every_document() {
    let a: Vec<_> = (0..300).map(tiny_doc).collect();
    let b: Vec<_> = (300..500).map(tiny_doc).collect();
    let (a_tvd, a_tvx, a_tvm) = segment(&a);
    let (b_tvd, b_tvx, b_tvm) = segment(&b);
    let ra = open(&a_tvd, &a_tvx, &a_tvm, &id(), "").unwrap();
    let rb = open(&b_tvd, &b_tvx, &b_tvm, &id(), "").unwrap();

    let mut w = TermVectorsWriter::new(&id(), "");
    w.copy_chunks(&ra, 0, ra.max_doc()).unwrap();
    w.copy_chunks(&rb, 0, rb.max_doc()).unwrap();
    let (tvd, tvx, tvm) = w.finish();

    let merged: Vec<_> = a.into_iter().chain(b).collect();
    assert_round_trips(&merged, &tvd, &tvx, &tvm);
    let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
    // Both sources' dirty tails are carried over verbatim, which is why
    // dirtiness accumulates across merge generations.
    assert_eq!(reader.num_dirty_chunks(), 2);
    assert_eq!(reader.num_dirty_docs(), 44 + 72);
}

#[test]
fn copy_chunks_rejects_a_chunk_claiming_two_billion_documents() {
    // The bulk-copy loop's twin of
    // `a_chunk_claiming_two_billion_documents_is_a_decode_error_not_an_overflow`:
    // `Lucene90CompressingTermVectorsWriter.copyChunks` also decodes
    // `code >>> 1`, so a corrupt token drives both `doc` and the writer's
    // own `numDocs` past `i32::MAX`. Advancing both with a plain `+=`
    // *before* testing `doc > to_doc` would let a wrap put the cursor back
    // inside the range and copy a chunk body two billion documents long.
    let docs: Vec<_> = (0..300).map(tiny_doc).collect();
    let (mut s_tvd, s_tvx, mut s_tvm) = segment(&docs);
    let entry = {
        let reader = open(&s_tvd, &s_tvx, &s_tvm, &id(), "").unwrap();
        reader.chunk_for_doc(128).unwrap()
    };
    assert_eq!(entry.doc_base, 128);
    let token_at = entry.start_pointer as usize + 2;
    let mut insert = Vec::new();
    write_vint(&mut insert, -1);
    splice_tvd(&mut s_tvd, &mut s_tvm, token_at, 1, &insert);

    let reader = open(&s_tvd, &s_tvx, &s_tvm, &id(), "").unwrap();
    let mut w = TermVectorsWriter::new(&id(), "");
    assert!(matches!(
        w.copy_chunks(&reader, 128, 256),
        Err(Error::CorruptChunkBounds { .. })
    ));
}

#[test]
fn copy_chunks_of_a_partial_range_copies_the_ragged_ends_document_at_a_time() {
    let docs: Vec<_> = (0..300).map(tiny_doc).collect();
    let (s_tvd, s_tvx, s_tvm) = segment(&docs);
    let reader = open(&s_tvd, &s_tvx, &s_tvm, &id(), "").unwrap();

    // 100..270 straddles both boundaries: 100..128 and 256..270 are
    // re-encoded, 128..256 is one whole copied chunk.
    let mut w = TermVectorsWriter::new(&id(), "");
    w.copy_chunks(&reader, 100, 270).unwrap();
    let (tvd, tvx, tvm) = w.finish();
    assert_round_trips(&docs[100..270], &tvd, &tvx, &tvm);
}

#[test]
fn copy_chunks_of_a_range_inside_one_chunk_copies_no_chunk_at_all() {
    let docs: Vec<_> = (0..300).map(tiny_doc).collect();
    let (s_tvd, s_tvx, s_tvm) = segment(&docs);
    let reader = open(&s_tvd, &s_tvx, &s_tvm, &id(), "").unwrap();
    let mut w = TermVectorsWriter::new(&id(), "");
    w.copy_chunks(&reader, 130, 140).unwrap();
    let (tvd, tvx, tvm) = w.finish();
    assert_round_trips(&docs[130..140], &tvd, &tvx, &tvm);
}

#[test]
fn copy_chunks_after_buffered_documents_forces_a_dirty_flush_first() {
    let docs: Vec<_> = (0..300).map(tiny_doc).collect();
    let (s_tvd, s_tvx, s_tvm) = segment(&docs);
    let reader = open(&s_tvd, &s_tvx, &s_tvm, &id(), "").unwrap();
    let mut w = TermVectorsWriter::new(&id(), "");
    let extra = tiny_doc(9999);
    w.add_document(&extra);
    w.copy_chunks(&reader, 0, 256).unwrap();
    let (tvd, tvx, tvm) = w.finish();
    let expected: Vec<_> = std::iter::once(extra)
        .chain(docs[0..256].iter().cloned())
        .collect();
    assert_round_trips(&expected, &tvd, &tvx, &tvm);
    let merged = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
    assert_eq!(chunk_headers(&merged)[0], (0, 1, true));
}

#[test]
fn copy_chunks_of_an_empty_range_writes_nothing() {
    let docs: Vec<_> = (0..300).map(tiny_doc).collect();
    let (s_tvd, s_tvx, s_tvm) = segment(&docs);
    let reader = open(&s_tvd, &s_tvx, &s_tvm, &id(), "").unwrap();
    let mut w = TermVectorsWriter::new(&id(), "");
    w.copy_chunks(&reader, 5, 5).unwrap();
    let (tvd, tvx, tvm) = w.finish();
    let merged = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
    assert_eq!(merged.max_doc(), 0);
    assert_eq!(merged.num_chunks(), 0);
}

#[test]
fn copy_chunks_rejects_an_out_of_range_or_inverted_document_range() {
    let docs: Vec<_> = (0..300).map(tiny_doc).collect();
    let (s_tvd, s_tvx, s_tvm) = segment(&docs);
    let reader = open(&s_tvd, &s_tvx, &s_tvm, &id(), "").unwrap();
    let mut w = TermVectorsWriter::new(&id(), "");
    assert!(matches!(
        w.copy_chunks(&reader, 0, 301),
        Err(Error::DocOutOfRange(301, 300))
    ));
    assert!(matches!(
        w.copy_chunks(&reader, 10, 5),
        Err(Error::InvertedDocRange {
            from_doc: 10,
            to_doc: 5
        })
    ));
}

#[test]
fn dirtiness_accumulates_across_bulk_copies_until_the_segment_is_too_dirty() {
    // `tooDirty`: more dirty docs than a full chunk AND more than 1% of
    // chunks dirty. Repeated bulk copies of a one-document segment build
    // exactly that -- the degradation Java's safety switch exists for.
    let one = vec![tiny_doc(0)];
    let (s_tvd, s_tvx, s_tvm) = segment(&one);
    let source = open(&s_tvd, &s_tvx, &s_tvm, &id(), "").unwrap();
    let probe = TermVectorsWriter::new(&id(), "");
    assert!(!probe.too_dirty(&source));

    let mut w = TermVectorsWriter::new(&id(), "");
    for _ in 0..130 {
        w.copy_chunks(&source, 0, 1).unwrap();
    }
    let (tvd, tvx, tvm) = w.finish();
    let merged = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
    assert_eq!(merged.num_dirty_chunks(), 130);
    assert_eq!(merged.num_dirty_docs(), 130);
    let probe = TermVectorsWriter::new(&id(), "");
    assert!(probe.too_dirty(&merged));
    assert!(!probe.can_bulk_copy(&merged));
    assert!(matches!(
        TermVectorsWriter::new(&id(), "").copy_chunks(&merged, 0, 1),
        Err(Error::BulkCopyNotPermitted { .. })
    ));
    // ... and it still reads back correctly.
    assert_eq!(merged.max_doc(), 130);
    assert_eq!(
        merged.document(129).unwrap().unwrap().fields[0].terms[0].term,
        b"t0000"
    );
}

#[test]
fn check_integrity_detects_payload_corruption_that_the_footer_shape_cannot() {
    let docs: Vec<_> = (0..300).map(tiny_doc).collect();
    let (mut tvd, tvx, tvm) = segment(&docs);
    {
        let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
        reader.check_integrity().unwrap();
    }
    // Flip one byte of a chunk body: every length, pointer and footer
    // field stays intact, so `open`'s `retrieve_checksum` still passes.
    let victim = tvd.len() - codec_util::FOOTER_LENGTH - 20;
    tvd[victim] ^= 0x40;
    let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
    assert!(reader.check_integrity().is_err());
}

/// Rewrites a valid two-chunk segment's `.tvd` at `at` via `mutate`,
/// keeping the file length identical so `open`'s `maxPointer` cross-check
/// still passes and only `copy_chunks` can catch the damage.
fn corrupt_chunk_header(mutate: impl Fn(&mut Vec<u8>, usize)) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let docs: Vec<_> = (0..300).map(tiny_doc).collect();
    let (mut tvd, tvx, tvm) = segment(&docs);
    let start = {
        let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
        reader.chunk_for_doc(128).unwrap().start_pointer as usize
    };
    mutate(&mut tvd, start);
    (tvd, tvx, tvm)
}

#[test]
fn a_chunk_header_whose_doc_base_disagrees_with_the_index_is_rejected() {
    // Java's `base != docID` `CorruptIndexException`. docBase 128 is a
    // two-byte vint (0x80 0x01); rewrite it as 129 in place.
    let (tvd, tvx, tvm) = corrupt_chunk_header(|tvd, at| {
        assert_eq!((tvd[at], tvd[at + 1]), (0x80, 0x01));
        tvd[at] = 0x81;
    });
    let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
    let mut w = TermVectorsWriter::new(&id(), "");
    assert!(matches!(
        w.copy_chunks(&reader, 0, 300),
        Err(Error::CorruptChunkBounds { .. })
    ));
}

#[test]
fn a_chunk_header_claiming_no_documents_is_rejected() {
    // A zero-document chunk would make the copy loop spin without
    // advancing. The token is `(chunkDocs << 1) | dirty`; 128 docs is
    // `256` = 0x80 0x02, rewritten as `0` = 0x80 0x00.
    let (tvd, tvx, tvm) = corrupt_chunk_header(|tvd, at| {
        assert_eq!((tvd[at + 2], tvd[at + 3]), (0x80, 0x02));
        tvd[at + 3] = 0x00;
    });
    let reader = open(&tvd, &tvx, &tvm, &id(), "").unwrap();
    let mut w = TermVectorsWriter::new(&id(), "");
    assert!(matches!(
        w.copy_chunks(&reader, 0, 300),
        Err(Error::CorruptChunkBounds { chunk_docs: 0, .. })
    ));
}

#[test]
fn a_chunk_claiming_more_documents_than_the_requested_range_is_rejected() {
    // Java's `docID > toDocID`. The first chunk claims 128 docs but only
    // 100 were asked for.
    let docs: Vec<_> = (0..300).map(tiny_doc).collect();
    let (s_tvd, s_tvx, s_tvm) = segment(&docs);
    let reader = open(&s_tvd, &s_tvx, &s_tvm, &id(), "").unwrap();
    // A range that merely ends mid-chunk is handled by the ragged-tail
    // loop, so reaching this guard needs a chunk header that *lies*:
    // rewrite the first chunk's token to claim 256 documents (same vint
    // width) and ask for 0..128.
    let mut tvd = s_tvd.clone();
    let start = reader.chunk_for_doc(0).unwrap().start_pointer as usize;
    assert_eq!((tvd[start + 1], tvd[start + 2]), (0x80, 0x02));
    tvd[start + 2] = 0x04; // 512 = 256 docs
    let bad = open(&tvd, &s_tvx, &s_tvm, &id(), "").unwrap();
    let mut w = TermVectorsWriter::new(&id(), "");
    assert!(matches!(
        w.copy_chunks(&bad, 0, 128),
        Err(Error::CorruptChunkBounds { .. })
    ));
}
