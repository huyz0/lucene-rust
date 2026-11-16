#![cfg(test)]
// The arithmetic gate is about values read off disk; a test's `i + 1` is
// not one. See docs/arithmetic-gate.md.
#![allow(clippy::arithmetic_side_effects)]

use super::*;

/// The unit decoder the reader calls, with this module's error type.
fn decompress_unit(
    mode: Mode,
    input: &mut SliceInput,
    original_length: usize,
    offset: usize,
    length: usize,
    out: &mut Vec<u8>,
) -> Result<()> {
    let compression = mode.compression();
    Ok(compression::decompress(
        compression,
        input,
        original_length,
        offset,
        length,
        out,
    )?)
}

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

/// A single-block, zero-dictionary `LZ4WithPresetDictCompressionMode`
/// unit wrapping `bytes` verbatim (see `decompress_unit`'s doc comment
/// for the interleaved length vints -- they're
/// only used by Java's partial-read optimization).
fn encode_store_unit(bytes: &[u8]) -> Vec<u8> {
    let block = encode_literal_lz4(bytes);
    let mut out = Vec::new();
    write_vint(&mut out, 0); // dictLength = 0
    write_vint(&mut out, bytes.len().max(1) as i32); // blockLength
    write_vint(&mut out, 1); // dict's compressed length
    write_vint(&mut out, block.len() as i32); // block0's compressed length
    out.push(0x00); // dict decompress unit: dictLength=0 -> single empty token
    out.extend(block);
    out
}

fn id() -> [u8; ID_LENGTH] {
    [7u8; ID_LENGTH]
}

/// Builds a valid `.fdt`+`.fdx`+`.fdm` trio for a single chunk containing
/// exactly one document (`doc_bytes`, its already-encoded field entries).
/// A single-doc chunk is the simplest valid framing (`numStoredFields`
/// and length are each a plain vint, no bulk int array); the bulk-array
/// path is exercised directly via `read_bulk_ints` tests instead.
fn build_single_chunk_index(doc_bytes: &[u8]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    build_single_chunk_index_with_meta_overrides(doc_bytes, 1, 0, 0)
}

/// The three tail counters are vlongs on the wire, so they are taken as
/// `i64` here -- a test needs to be able to put a value past `i32` in one.
fn build_single_chunk_index_with_meta_overrides(
    doc_bytes: &[u8],
    num_chunks_outer: i64,
    num_dirty_chunks: i64,
    num_dirty_docs: i64,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    // .fdt
    let mut fdt = Vec::new();
    fdt.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
    write_string(&mut fdt, DATA_CODEC_BEST_SPEED);
    fdt.extend_from_slice(&(VERSION_CURRENT as u32).to_be_bytes());
    fdt.extend_from_slice(&id());
    fdt.push(0); // empty suffix
    let chunk_start = fdt.len() as i64;

    write_vint(&mut fdt, 0); // docBase
    write_vint(&mut fdt, 1 << 2); // token: chunkDocs=1, sliced=0, dirty=0
    write_vint(&mut fdt, 1); // numStoredFields
    write_vint(&mut fdt, doc_bytes.len() as i32); // length
    fdt.extend(encode_store_unit(doc_bytes));
    fdt.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
    fdt.extend_from_slice(&0u32.to_be_bytes());
    let checksum = crc32fast::hash(&fdt) as u64;
    fdt.extend_from_slice(&checksum.to_be_bytes());

    // .fdx: docs array [0] and startPointers array [chunk_start], 1 block each (blockShift=0)
    let mut fdx = Vec::new();
    fdx.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
    write_string(&mut fdx, INDEX_CODEC);
    fdx.extend_from_slice(&(INDEX_VERSION_CURRENT as u32).to_be_bytes());
    fdx.extend_from_slice(&id());
    fdx.push(0);
    let docs_start = fdx.len() as i64;
    // Both arrays are constant (bpv=0), so they need zero bytes here --
    // the constant value itself lives entirely in the .fdm meta below.
    let docs_end = fdx.len() as i64;
    let start_pointers_end = fdx.len() as i64;
    fdx.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
    fdx.extend_from_slice(&0u32.to_be_bytes());
    let checksum = crc32fast::hash(&fdx) as u64;
    fdx.extend_from_slice(&checksum.to_be_bytes());

    // .fdm. The index arrays carry one extra sentinel entry beyond the
    // real chunk count (see `open`'s doc comment on `index_num_chunks`):
    // docs = [0 (chunk 0's docBase), maxDoc (sentinel)], startPointers =
    // [chunk_start, maxPointer (sentinel)]. blockShift=0 -> 1 value/block,
    // so that's 2 blocks each.
    let max_doc = 1i32;
    let max_pointer = (fdt.len() - codec_util::FOOTER_LENGTH) as i64;
    let mut fdm = Vec::new();
    fdm.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
    write_string(&mut fdm, META_CODEC);
    fdm.extend_from_slice(&(VERSION_CURRENT as u32).to_be_bytes());
    fdm.extend_from_slice(&id());
    fdm.push(0);
    write_vint(&mut fdm, 80 * 1024); // chunkSize (unused by the reader beyond framing)
    fdm.extend_from_slice(&max_doc.to_le_bytes());
    fdm.extend_from_slice(&0i32.to_le_bytes()); // blockShift = 0 -> 1 value per block
    fdm.extend_from_slice(&2i32.to_le_bytes()); // index_num_chunks = totalChunks(1) + 1
    fdm.extend_from_slice(&docs_start.to_le_bytes());
    for min in [0i64, max_doc as i64] {
        fdm.extend_from_slice(&min.to_le_bytes());
        fdm.extend_from_slice(&0i32.to_le_bytes()); // avg bits
        fdm.extend_from_slice(&0i64.to_le_bytes()); // offset
        fdm.push(0); // bpv
    }
    fdm.extend_from_slice(&docs_end.to_le_bytes());
    for min in [chunk_start, max_pointer] {
        fdm.extend_from_slice(&min.to_le_bytes());
        fdm.extend_from_slice(&0i32.to_le_bytes());
        fdm.extend_from_slice(&0i64.to_le_bytes());
        fdm.push(0);
    }
    fdm.extend_from_slice(&start_pointers_end.to_le_bytes());
    fdm.extend_from_slice(&max_pointer.to_le_bytes());
    write_vlong(&mut fdm, num_chunks_outer);
    write_vlong(&mut fdm, num_dirty_chunks);
    write_vlong(&mut fdm, num_dirty_docs);
    fdm.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
    fdm.extend_from_slice(&0u32.to_be_bytes());
    let checksum = crc32fast::hash(&fdm) as u64;
    fdm.extend_from_slice(&checksum.to_be_bytes());

    (fdt, fdx, fdm)
}

fn field_bytes(field_number: i32, bits: i64, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let info_and_bits = ((field_number as i64) << TYPE_BITS) | bits;
    write_vlong(&mut out, info_and_bits);
    out.extend_from_slice(payload);
    out
}

fn write_vlong(out: &mut Vec<u8>, mut v: i64) {
    loop {
        let mut b = (v & 0x7f) as u8;
        v = ((v as u64) >> 7) as i64;
        if v != 0 {
            b |= 0x80;
            out.push(b);
        } else {
            out.push(b);
            break;
        }
    }
}

fn string_field_payload(s: &str) -> Vec<u8> {
    let mut out = Vec::new();
    write_string(&mut out, s);
    out
}

#[test]
fn a_captured_chunk_reads_back_its_documents_in_every_mode() {
    let doc = |i: usize, len: usize| Document {
        fields: vec![
            StoredField {
                field_number: 0,
                value: FieldValue::String(format!("id-{i}")),
            },
            StoredField {
                field_number: 1,
                value: FieldValue::Binary(
                    (0..len).map(|j| ((i * 31 + j / 7) % 251) as u8).collect(),
                ),
            },
            StoredField {
                field_number: 2,
                value: FieldValue::Long(i as i64),
            },
        ],
    };
    for mode in [
        Mode::BestSpeed,
        Mode::BestCompression,
        Mode::Zstd(ZstdMode::zstd(3)),
        Mode::Zstd(ZstdMode::zstd_no_dict(3)),
        Mode::ZstdPaged(3),
    ] {
        let mut w = StoredFieldsWriter::new(mode, &[0; ID_LENGTH], "");
        let docs: Vec<Document> = (0..12)
            .map(|i| doc(i, if i % 4 == 3 { 700_000 } else { 300 }))
            .collect();
        let mut kept = Vec::new();
        for d in &docs {
            let at = w.num_docs();
            let big = d.fields.len() == 3
                && matches!(&d.fields[1].value, FieldValue::Binary(b) if b.len() > 1000);
            if big {
                w.capture_chunks();
            }
            w.add_document(d);
            if big {
                let chunks = w.take_captured();
                let chunk = chunks
                    .into_iter()
                    .find(|c| c.contains(at))
                    .expect("a big document closes its chunk");
                kept.push((at, chunk));
            }
        }
        assert_eq!(kept.len(), 3, "{mode:?}");
        for (at, chunk) in &kept {
            assert_eq!(
                &chunk.document(*at).unwrap(),
                &docs[*at as usize],
                "{mode:?}"
            );
            assert!(chunk.document(*at + 1).is_err(), "{mode:?}");
            assert!(chunk.ram_bytes() > 0);
        }
        // Documents sharing the big one's chunk read back too.
        let (at, chunk) = &kept[0];
        for d in 0..*at {
            if chunk.contains(d) {
                assert_eq!(&chunk.document(d).unwrap(), &docs[d as usize], "{mode:?}");
            }
        }
        // The file the writer goes on to finish is unchanged by capturing.
        let mut plain = StoredFieldsWriter::new(mode, &[0; ID_LENGTH], "");
        for d in &docs {
            plain.add_document(d);
        }
        assert_eq!(w.finish(), plain.finish(), "{mode:?}");
    }
    // Chunks compressed against a dictionary are not captured.
    let mut w = StoredFieldsWriter::new(Mode::ZstdPaged(3), &[0; ID_LENGTH], "");
    w.set_zstd_dictionary(&[7; 1024]);
    w.capture_chunks();
    w.add_document(&doc(0, 700_000));
    assert!(w.take_captured().is_empty());
}

#[test]
fn single_doc_single_string_field_round_trips() {
    let doc_bytes = field_bytes(0, TYPE_STRING, &string_field_payload("hello"));
    let (fdt, fdx, fdm) = build_single_chunk_index(&doc_bytes);
    let reader = open(&fdt, &fdx, &fdm, &id(), "").unwrap();
    assert_eq!(reader.max_doc(), 1);
    let doc = reader.document(0).unwrap();
    assert_eq!(doc.fields.len(), 1);
    assert_eq!(doc.fields[0].field_number, 0);
    assert_eq!(doc.fields[0].value, FieldValue::String("hello".to_string()));
}

#[test]
fn doc_out_of_range_rejected() {
    let field = field_bytes(0, TYPE_STRING, &string_field_payload("x"));
    let (fdt, fdx, fdm) = build_single_chunk_index(&field);
    let reader = open(&fdt, &fdx, &fdm, &id(), "").unwrap();
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
fn read_zint_round_trips_small_and_large_values() {
    for v in [0i32, 1, -1, 63, -64, 1_000_000, i32::MIN, i32::MAX] {
        let mut out = Vec::new();
        write_vint(&mut out, lucene_util::zigzag::encode(v as i64) as i32);
        let mut input = SliceInput::new(&out);
        assert_eq!(read_zint(&mut input).unwrap(), v, "value {v}");
    }
}

#[test]
fn read_zfloat_small_integer_and_full_encoding() {
    // small integer: b = (value+1)|0x80
    let mut out = vec![((5i32 + 1) as u8) | 0x80];
    let mut input = SliceInput::new(&out);
    assert_eq!(read_zfloat(&mut input).unwrap(), 5.0);

    // full encoding: positive float, first byte < 0x80 and != 0xFF.
    // Layout is `b<<24 | (readShort()&0xFFFF)<<8 | readByte()`, and
    // `readShort`/`readInt` are little-endian, so the middle two bytes
    // on disk are (bits>>8)&0xFF then (bits>>16)&0xFF -- not the
    // natural big-endian byte order of `bits` itself.
    out.clear();
    let bits = 1.5f32.to_bits();
    out.push((bits >> 24) as u8);
    out.push((bits >> 8) as u8);
    out.push((bits >> 16) as u8);
    out.push(bits as u8);
    let mut input = SliceInput::new(&out);
    assert_eq!(read_zfloat(&mut input).unwrap(), 1.5);

    // negative value: leading 0xFF then a plain little-endian `readInt`.
    out.clear();
    out.push(0xFF);
    out.extend_from_slice(&(-2.5f32).to_bits().to_le_bytes());
    let mut input = SliceInput::new(&out);
    assert_eq!(read_zfloat(&mut input).unwrap(), -2.5);
}

#[test]
fn read_zdouble_small_integer_float_and_full_encoding() {
    let mut out = vec![((3i32 + 1) as u8) | 0x80];
    let mut input = SliceInput::new(&out);
    assert_eq!(read_zdouble(&mut input).unwrap(), 3.0);

    // 0xFE marker + a plain little-endian `readInt` holding float bits.
    out = vec![0xFE];
    out.extend_from_slice(&2.5f32.to_bits().to_le_bytes());
    let mut input = SliceInput::new(&out);
    assert_eq!(read_zdouble(&mut input).unwrap(), 2.5);

    // 0xFF marker + a plain little-endian `readLong` holding double bits.
    out = vec![0xFF];
    out.extend_from_slice(&1.25f64.to_bits().to_le_bytes());
    let mut input = SliceInput::new(&out);
    assert_eq!(read_zdouble(&mut input).unwrap(), 1.25);

    // Full positive-double encoding: `b<<56 | (readInt()&0xFFFFFFFF)<<24
    // | (readShort()&0xFFFF)<<8 | readByte()`, with `readInt`/`readShort`
    // little-endian -- same byte-order subtlety as `read_zfloat`.
    out.clear();
    let bits: u64 = 4607182418800017408; // 1.0f64's bit pattern
    out.push((bits >> 56) as u8);
    out.extend_from_slice(&((bits >> 24) as u32).to_le_bytes());
    out.extend_from_slice(&((bits >> 8) as u16).to_le_bytes());
    out.push(bits as u8);
    let mut input = SliceInput::new(&out);
    assert_eq!(read_zdouble(&mut input).unwrap(), 1.0);
}

#[test]
fn read_tlong_uncompressed_and_scaled_encodings() {
    // uncompressed: header bits 0x00, low 5 bits hold zigzag(value) directly if it fits
    let mut out = vec![lucene_util::zigzag::encode(7) as u8];
    let mut input = SliceInput::new(&out);
    assert_eq!(read_tlong(&mut input).unwrap(), 7);

    // second-scaled: value 5000ms = 5s -> zigzag(5)=10, header = 10 | SECOND_ENCODING
    out = vec![10u8 | SECOND_ENCODING];
    let mut input = SliceInput::new(&out);
    assert_eq!(read_tlong(&mut input).unwrap(), 5000);

    // day-scaled: value = 2 days = 2*86_400_000 ms -> zigzag(2)=4, header=4|DAY_ENCODING
    out = vec![4u8 | DAY_ENCODING];
    let mut input = SliceInput::new(&out);
    assert_eq!(read_tlong(&mut input).unwrap(), 2 * DAY);
}

#[test]
fn read_bulk_ints_all_equal_shape() {
    let mut out = Vec::new();
    out.push(0); // bpv=0 marker
    write_vint(&mut out, 42);
    let mut input = SliceInput::new(&out);
    assert_eq!(read_bulk_ints(&mut input, 5).unwrap(), vec![42i64; 5]);
}

#[test]
fn read_bulk_ints_scalar_tail_for_every_nonzero_width() {
    // count < 128 always takes the scalar tail loop (`read_scalar`),
    // regardless of bpv -- exercise all three non-constant widths.
    let mut out = vec![8u8, 10, 250];
    let mut input = SliceInput::new(&out);
    assert_eq!(read_bulk_ints(&mut input, 2).unwrap(), vec![10, 250]);

    out = vec![16u8];
    out.extend_from_slice(&300u16.to_le_bytes());
    out.extend_from_slice(&40000u16.to_le_bytes());
    input = SliceInput::new(&out);
    assert_eq!(read_bulk_ints(&mut input, 2).unwrap(), vec![300, 40000]);

    out = vec![32u8];
    out.extend_from_slice(&70000i32.to_le_bytes());
    out.extend_from_slice(&(-1i32).to_le_bytes());
    input = SliceInput::new(&out);
    assert_eq!(
        read_bulk_ints(&mut input, 2).unwrap(),
        vec![70000, 0xFFFFFFFF]
    );
}

#[test]
fn read_bulk_ints_transposed_block_matches_java_layout() {
    // 128 sequential values 0..128, bpv=8: verifies the word/lane
    // transposition (see `read_bulk_ints`'s doc comment) against a
    // hand-encoded block using Java's exact readInts8 layout formula.
    let values: Vec<i64> = (0..128).collect();
    let mut out = vec![8u8]; // bpv=8
    let values_per_word = 64 / 8;
    let num_words = 128 / values_per_word;
    for w in 0..num_words {
        let mut word: u64 = 0;
        for lane in 0..values_per_word {
            let pos = w + lane * num_words;
            let shift = (values_per_word - 1 - lane) * 8;
            word |= (values[pos] as u64) << shift;
        }
        out.extend_from_slice(&(word as i64).to_le_bytes());
    }
    let mut input = SliceInput::new(&out);
    assert_eq!(read_bulk_ints(&mut input, 128).unwrap(), values);
}

#[test]
fn read_bulk_ints_unsupported_width_rejected() {
    // Real Lucene's writer (`StoredFieldsInts.writeInts`) only ever emits
    // bpv 0 (constant-fill) or the smallest of 8/16/32 that fits the max
    // value, and Java's own `readInts` throws `IOException("Unsupported
    // number of bits per value: " + bpv)` for every other byte value --
    // confirmed against
    // lucene/core/.../codecs/lucene90/compressing/StoredFieldsInts.java.
    // Sweep the whole invalid space (not just one sentinel) to prove the
    // rejection boundary is exactly "not 0, 8, 16, or 32", matching
    // Java's switch/default exactly.
    for width in 0u16..=255 {
        let bpv = width as u8;
        if matches!(bpv, 0 | 8 | 16 | 32) {
            continue;
        }
        let out = vec![bpv];
        let mut input = SliceInput::new(&out);
        assert!(
            matches!(
                read_bulk_ints(&mut input, 4),
                Err(Error::UnsupportedBulkIntWidth(w)) if w == bpv
            ),
            "expected UnsupportedBulkIntWidth({bpv}) to be rejected"
        );
    }
}

#[test]
fn wrong_segment_id_rejected() {
    let field = field_bytes(0, TYPE_STRING, &string_field_payload("x"));
    let (fdt, fdx, fdm) = build_single_chunk_index(&field);
    let wrong_id = [9u8; ID_LENGTH];
    assert!(open(&fdt, &fdx, &fdm, &wrong_id, "").is_err());
}

#[test]
fn num_chunks_mismatch_rejected() {
    // The test builder always writes index_num_chunks=2 (1 real chunk +
    // 1 sentinel); passing outer=2 breaks the required index=outer+1
    // relationship (2 != 2+1).
    let field = field_bytes(0, TYPE_STRING, &string_field_payload("x"));
    let (fdt, fdx, fdm) = build_single_chunk_index_with_meta_overrides(&field, 2, 0, 0);
    assert!(matches!(
        open(&fdt, &fdx, &fdm, &id(), ""),
        Err(Error::NumChunksMismatch {
            index_num_chunks: 2,
            outer_num_chunks: 2
        })
    ));
}

#[test]
fn too_many_dirty_chunks_rejected() {
    let field = field_bytes(0, TYPE_STRING, &string_field_payload("x"));
    let (fdt, fdx, fdm) = build_single_chunk_index_with_meta_overrides(&field, 1, 2, 2);
    assert!(matches!(
        open(&fdt, &fdx, &fdm, &id(), ""),
        Err(Error::TooManyDirtyChunks(2, 1))
    ));
}

#[test]
fn dirty_chunks_docs_mismatch_rejected() {
    let field = field_bytes(0, TYPE_STRING, &string_field_payload("x"));
    let (fdt, fdx, fdm) = build_single_chunk_index_with_meta_overrides(&field, 1, 1, 0);
    assert!(matches!(
        open(&fdt, &fdx, &fdm, &id(), ""),
        Err(Error::DirtyChunksDocsMismatch(1, 0))
    ));
}

#[test]
fn wrong_fdt_length_rejected() {
    let field = field_bytes(0, TYPE_STRING, &string_field_payload("x"));
    let (mut fdt, fdx, fdm) = build_single_chunk_index(&field);
    fdt.push(0); // append a stray byte after the footer
    assert!(open(&fdt, &fdx, &fdm, &id(), "").is_err());
}

#[test]
fn multi_doc_chunk_round_trips_through_bulk_int_arrays() {
    let doc0 = field_bytes(0, TYPE_STRING, &string_field_payload("aa"));
    let doc1 = field_bytes(1, TYPE_NUMERIC_INT, &{
        let mut p = Vec::new();
        write_vint(&mut p, lucene_util::zigzag::encode(5) as i32);
        p
    });

    // .fdt: a 2-doc chunk. numStoredFields uses the bpv=0 (all-equal)
    // shape (both docs have exactly 1 field); lengths uses bpv=8 scalar
    // bytes (the docs are different lengths), covering both bulk-array
    // shapes in one chunk.
    let mut fdt = Vec::new();
    fdt.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
    write_string(&mut fdt, DATA_CODEC_BEST_SPEED);
    fdt.extend_from_slice(&(VERSION_CURRENT as u32).to_be_bytes());
    fdt.extend_from_slice(&id());
    fdt.push(0);
    let chunk_start = fdt.len() as i64;

    write_vint(&mut fdt, 0); // docBase
    write_vint(&mut fdt, 2 << 2); // chunkDocs=2, sliced=0, dirty=0
    fdt.push(0); // numStoredFields: bpv=0 constant
    write_vint(&mut fdt, 1);
    fdt.push(8); // lengths: bpv=8 (count=2 < 128, so `read_bulk_ints`'s scalar tail loop)
    fdt.push(doc0.len() as u8);
    fdt.push(doc1.len() as u8);
    let payload = [doc0.clone(), doc1.clone()].concat();
    fdt.extend(encode_store_unit(&payload));
    fdt.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
    fdt.extend_from_slice(&0u32.to_be_bytes());
    let checksum = crc32fast::hash(&fdt) as u64;
    fdt.extend_from_slice(&checksum.to_be_bytes());

    let (fdx, fdm) = build_fdx_fdm_for_single_chunk(&fdt, 2, chunk_start);
    let reader = open(&fdt, &fdx, &fdm, &id(), "").unwrap();
    assert_eq!(reader.max_doc(), 2);

    let d0 = reader.document(0).unwrap();
    assert_eq!(d0.fields[0].value, FieldValue::String("aa".to_string()));
    let d1 = reader.document(1).unwrap();
    assert_eq!(d1.fields[0].value, FieldValue::Int(5));
}

#[test]
fn sliced_chunk_end_to_end_through_document() {
    // `sliced` only controls how many independent LZ4WithPresetDict units
    // back the chunk, not their size -- so a small payload with the
    // sliced bit set already exercises `document()`'s sliced branch
    // (one loop iteration, since `remaining` < the 80KB unit size).
    let doc_bytes = field_bytes(0, TYPE_STRING, &string_field_payload("sliced"));

    let mut fdt = Vec::new();
    fdt.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
    write_string(&mut fdt, DATA_CODEC_BEST_SPEED);
    fdt.extend_from_slice(&(VERSION_CURRENT as u32).to_be_bytes());
    fdt.extend_from_slice(&id());
    fdt.push(0);
    let chunk_start = fdt.len() as i64;

    write_vint(&mut fdt, 0); // docBase
    write_vint(&mut fdt, (1 << 2) | 1); // chunkDocs=1, sliced=1
    write_vint(&mut fdt, 1); // numStoredFields
    write_vint(&mut fdt, doc_bytes.len() as i32);
    fdt.extend(encode_store_unit(&doc_bytes));
    fdt.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
    fdt.extend_from_slice(&0u32.to_be_bytes());
    let checksum = crc32fast::hash(&fdt) as u64;
    fdt.extend_from_slice(&checksum.to_be_bytes());

    let (fdx, fdm) = build_fdx_fdm_for_single_chunk(&fdt, 1, chunk_start);
    let reader = open(&fdt, &fdx, &fdm, &id(), "").unwrap();
    let doc = reader.document(0).unwrap();
    assert_eq!(
        doc.fields[0].value,
        FieldValue::String("sliced".to_string())
    );
}

#[test]
fn empty_document_has_no_fields() {
    // A chunk with 2 docs, the first empty (numStoredFields=0, length=0)
    // -- exercises `document()`'s `doc_length == 0` shortcut.
    let doc1 = field_bytes(0, TYPE_STRING, &string_field_payload("x"));

    let mut fdt = Vec::new();
    fdt.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
    write_string(&mut fdt, DATA_CODEC_BEST_SPEED);
    fdt.extend_from_slice(&(VERSION_CURRENT as u32).to_be_bytes());
    fdt.extend_from_slice(&id());
    fdt.push(0);
    let chunk_start = fdt.len() as i64;

    write_vint(&mut fdt, 0); // docBase
    write_vint(&mut fdt, 2 << 2); // chunkDocs=2
    fdt.push(8); // numStoredFields: bpv=8, [0, 1]
    fdt.push(0);
    fdt.push(1);
    fdt.push(8); // lengths: bpv=8, [0, doc1.len()]
    fdt.push(0);
    fdt.push(doc1.len() as u8);
    fdt.extend(encode_store_unit(&doc1));
    fdt.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
    fdt.extend_from_slice(&0u32.to_be_bytes());
    let checksum = crc32fast::hash(&fdt) as u64;
    fdt.extend_from_slice(&checksum.to_be_bytes());

    let (fdx, fdm) = build_fdx_fdm_for_single_chunk(&fdt, 2, chunk_start);
    let reader = open(&fdt, &fdx, &fdm, &id(), "").unwrap();
    let doc0 = reader.document(0).unwrap();
    assert!(doc0.fields.is_empty());
    let doc1_read = reader.document(1).unwrap();
    assert_eq!(
        doc1_read.fields[0].value,
        FieldValue::String("x".to_string())
    );
}

#[test]
fn corrupt_chunk_bounds_rejected() {
    // A chunk header claiming chunkDocs=1 starting at docBase=0, but the
    // .fdx points a doc id (1) at this same chunk -- out of its range.
    let doc_bytes = field_bytes(0, TYPE_STRING, &string_field_payload("x"));
    let (fdt, fdx, fdm) = build_single_chunk_index(&doc_bytes);
    // Patch maxDoc in .fdm (see build_single_chunk_index: a fixed i32 at
    // a known offset) up to 2 so `document(1)` is in-range per `open`'s
    // own doc-count check, but still out of the single real chunk.
    let mut fdm = fdm;
    let max_doc_offset = 4 + 1 + META_CODEC.len() + 4 + ID_LENGTH + 1 + vint_len_test(80 * 1024);
    fdm[max_doc_offset..max_doc_offset + 4].copy_from_slice(&2i32.to_le_bytes());
    // Recompute the meta footer checksum after patching maxDoc: it
    // covers everything up to (not including) the trailing 8-byte
    // checksum field itself (footer magic + algorithm id are covered).
    let checksum_at = fdm.len() - 8;
    let checksum = crc32fast::hash(&fdm[..checksum_at]) as u64;
    fdm[checksum_at..].copy_from_slice(&checksum.to_be_bytes());

    let reader = open(&fdt, &fdx, &fdm, &id(), "").unwrap();
    assert!(matches!(
        reader.document(1),
        Err(Error::CorruptChunkBounds { .. })
    ));
}

fn vint_len_test(mut v: i32) -> usize {
    let mut n = 1;
    while (v as u32) >= 0x80 {
        v = ((v as u32) >> 7) as i32;
        n += 1;
    }
    n
}

#[test]
fn decompress_unit_zero_length_produces_empty_vec() {
    let mut input = SliceInput::new(&[]);
    let mut out = Vec::new();
    decompress_unit(Mode::BestSpeed, &mut input, 0, 0, 0, &mut out).unwrap();
    assert_eq!(out, Vec::<u8>::new());
}

#[test]
fn long_binary_field_exercises_extended_literal_length_encoding() {
    // >270 bytes forces `encode_literal_lz4`'s extended-length loop
    // (literalLen encoded as 0x0F + continuation bytes).
    let long_value = vec![b'q'; 300];
    let mut payload = Vec::new();
    write_vint(&mut payload, long_value.len() as i32);
    payload.extend_from_slice(&long_value);
    let doc_bytes = field_bytes(0, TYPE_BYTE_ARR, &payload);

    let (fdt, fdx, fdm) = build_single_chunk_index(&doc_bytes);
    let reader = open(&fdt, &fdx, &fdm, &id(), "").unwrap();
    let doc = reader.document(0).unwrap();
    assert_eq!(doc.fields[0].value, FieldValue::Binary(long_value));
}

#[test]
fn large_field_number_exercises_vlong_continuation_byte() {
    // fieldNumber=20 -> infoAndBits = 20<<3 = 160, which needs a vlong
    // continuation byte (>127).
    let doc_bytes = field_bytes(20, TYPE_STRING, &string_field_payload("y"));
    let (fdt, fdx, fdm) = build_single_chunk_index(&doc_bytes);
    let reader = open(&fdt, &fdx, &fdm, &id(), "").unwrap();
    let doc = reader.document(0).unwrap();
    assert_eq!(doc.fields[0].field_number, 20);
    assert_eq!(doc.fields[0].value, FieldValue::String("y".to_string()));
}

#[test]
fn sliced_chunk_splits_decompression_across_units() {
    // Force `sliced=true` with a payload spanning two 80KB-equivalent
    // units -- use a tiny "unit size" stand-in isn't possible (it's a
    // hardcoded constant in `document()`), so this test instead directly
    // exercises `decompress_unit` twice back to back the same way
    // `document()`'s sliced branch does, confirming concatenation is
    // correct. (An end-to-end `document()` sliced test would need a
    // payload > 80KB, impractical for a unit test.)
    let part_a = vec![b'a'; 100];
    let part_b = vec![b'b'; 50];
    let mut compressed = Vec::new();
    compressed.extend(encode_store_unit(&part_a));
    compressed.extend(encode_store_unit(&part_b));
    let mut input = SliceInput::new(&compressed);

    let mut out = Vec::new();
    decompress_unit(
        Mode::BestSpeed,
        &mut input,
        part_a.len(),
        0,
        part_a.len(),
        &mut out,
    )
    .unwrap();
    decompress_unit(
        Mode::BestSpeed,
        &mut input,
        part_b.len(),
        0,
        part_b.len(),
        &mut out,
    )
    .unwrap();

    let mut expected = part_a;
    expected.extend(part_b);
    assert_eq!(out, expected);
}

/// Builds a valid `.fdx`+`.fdm` pair for a single chunk of `chunk_docs`
/// documents starting at `chunk_start` in `fdt`, sharing
/// `multi_doc_chunk_round_trips_through_bulk_int_arrays`'s and
/// `build_single_chunk_index`'s index/meta layout.
fn build_fdx_fdm_for_single_chunk(
    fdt: &[u8],
    chunk_docs: i32,
    chunk_start: i64,
) -> (Vec<u8>, Vec<u8>) {
    let mut fdx = Vec::new();
    fdx.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
    write_string(&mut fdx, INDEX_CODEC);
    fdx.extend_from_slice(&(INDEX_VERSION_CURRENT as u32).to_be_bytes());
    fdx.extend_from_slice(&id());
    fdx.push(0);
    let docs_start = fdx.len() as i64;
    let docs_end = fdx.len() as i64;
    let start_pointers_end = fdx.len() as i64;
    fdx.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
    fdx.extend_from_slice(&0u32.to_be_bytes());
    let checksum = crc32fast::hash(&fdx) as u64;
    fdx.extend_from_slice(&checksum.to_be_bytes());

    // Index arrays carry a sentinel entry beyond the 1 real chunk (see
    // `open`'s doc comment on `index_num_chunks`): 2 blocks each.
    let max_pointer = (fdt.len() - codec_util::FOOTER_LENGTH) as i64;
    let mut fdm = Vec::new();
    fdm.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
    write_string(&mut fdm, META_CODEC);
    fdm.extend_from_slice(&(VERSION_CURRENT as u32).to_be_bytes());
    fdm.extend_from_slice(&id());
    fdm.push(0);
    write_vint(&mut fdm, 80 * 1024);
    fdm.extend_from_slice(&chunk_docs.to_le_bytes()); // maxDoc
    fdm.extend_from_slice(&0i32.to_le_bytes()); // blockShift
    fdm.extend_from_slice(&2i32.to_le_bytes()); // index_num_chunks = totalChunks(1) + 1
    fdm.extend_from_slice(&docs_start.to_le_bytes());
    for min in [0i64, chunk_docs as i64] {
        fdm.extend_from_slice(&min.to_le_bytes());
        fdm.extend_from_slice(&0i32.to_le_bytes());
        fdm.extend_from_slice(&0i64.to_le_bytes());
        fdm.push(0);
    }
    fdm.extend_from_slice(&docs_end.to_le_bytes());
    for min in [chunk_start, max_pointer] {
        fdm.extend_from_slice(&min.to_le_bytes());
        fdm.extend_from_slice(&0i32.to_le_bytes());
        fdm.extend_from_slice(&0i64.to_le_bytes());
        fdm.push(0);
    }
    fdm.extend_from_slice(&start_pointers_end.to_le_bytes());
    fdm.extend_from_slice(&max_pointer.to_le_bytes());
    write_vint(&mut fdm, 1);
    write_vint(&mut fdm, 0);
    write_vint(&mut fdm, 0);
    fdm.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
    fdm.extend_from_slice(&0u32.to_be_bytes());
    let checksum = crc32fast::hash(&fdm) as u64;
    fdm.extend_from_slice(&checksum.to_be_bytes());

    (fdx, fdm)
}

/// Decodes one value through [`visit_field`] and hands back what the
/// visitor was given, so tests can assert on the one decode path, which is
/// the visitor's.
fn visit_one(payload: &[u8], bits: i64) -> Result<FieldValue> {
    let mut input = SliceInput::new(payload);
    let mut visitor = DocumentVisitor::all();
    visit_field(&mut input, 0, bits, usize::MAX, &mut visitor)?;
    let mut doc = visitor.into_document();
    Ok(doc.fields.remove(0).value)
}

#[test]
fn every_field_value_type_round_trips_through_visit_field() {
    let mut int_payload = Vec::new();
    write_vint(&mut int_payload, lucene_util::zigzag::encode(-7) as i32);
    assert_eq!(
        visit_one(&int_payload, TYPE_NUMERIC_INT).unwrap(),
        FieldValue::Int(-7)
    );

    // zigzag(7) = 14, fits directly in the header's low 5 bits with no
    // continuation byte and no second/hour/day scale applied.
    let mut long_payload = vec![lucene_util::zigzag::encode(7) as u8];
    assert_eq!(
        visit_one(&long_payload, TYPE_NUMERIC_LONG).unwrap(),
        FieldValue::Long(7)
    );

    let mut bin_payload = Vec::new();
    write_vint(&mut bin_payload, 3);
    bin_payload.extend_from_slice(b"xyz");
    assert_eq!(
        visit_one(&bin_payload, TYPE_BYTE_ARR).unwrap(),
        FieldValue::Binary(b"xyz".to_vec())
    );

    let mut str_payload = Vec::new();
    write_string(&mut str_payload, "hi");
    assert_eq!(
        visit_one(&str_payload, TYPE_STRING).unwrap(),
        FieldValue::String("hi".to_string())
    );

    long_payload = vec![((9i32 + 1) as u8) | 0x80];
    assert_eq!(
        visit_one(&long_payload, TYPE_NUMERIC_FLOAT).unwrap(),
        FieldValue::Float(9.0)
    );

    long_payload = vec![((2i32 + 1) as u8) | 0x80];
    assert_eq!(
        visit_one(&long_payload, TYPE_NUMERIC_DOUBLE).unwrap(),
        FieldValue::Double(2.0)
    );

    assert!(matches!(visit_one(&[], 6), Err(Error::UnknownTypeTag(6))));
}

/// [`skip_field`] must land the cursor exactly where [`visit_field`]
/// does, for every encoding -- a skip that stops one byte short or long
/// misreads every following field of the document rather than failing.
#[test]
fn skip_field_advances_exactly_as_far_as_visit_field() {
    let mut int_payload = Vec::new();
    write_vint(&mut int_payload, lucene_util::zigzag::encode(-7) as i32);
    let mut bin_payload = Vec::new();
    write_vint(&mut bin_payload, 3);
    bin_payload.extend_from_slice(b"xyz");
    let mut str_payload = Vec::new();
    write_string(&mut str_payload, "hi");

    let cases: [(Vec<u8>, i64); 6] = [
        (int_payload, TYPE_NUMERIC_INT),
        (
            vec![lucene_util::zigzag::encode(7) as u8],
            TYPE_NUMERIC_LONG,
        ),
        (bin_payload, TYPE_BYTE_ARR),
        (str_payload, TYPE_STRING),
        (vec![((9i32 + 1) as u8) | 0x80], TYPE_NUMERIC_FLOAT),
        (vec![((2i32 + 1) as u8) | 0x80], TYPE_NUMERIC_DOUBLE),
    ];
    for (payload, bits) in cases {
        let mut read = SliceInput::new(&payload);
        let mut visitor = DocumentVisitor::all();
        visit_field(&mut read, 0, bits, usize::MAX, &mut visitor).unwrap();
        let mut skipped = SliceInput::new(&payload);
        skip_field(&mut skipped, bits).unwrap();
        assert_eq!(
            read.position(),
            skipped.position(),
            "skip_field disagreed with visit_field for bits={bits}"
        );
    }

    let mut unknown = SliceInput::new(&[]);
    assert!(matches!(
        skip_field(&mut unknown, 6),
        Err(Error::UnknownTypeTag(6))
    ));
}

fn paged_docs(n: usize) -> Vec<Document> {
    let words = [
        "merge", "segment", "commit", "index", "query", "field", "stored", "reader", "writer",
        "zstd", "page", "cache", "lucene", "rust", "plugin", "search",
    ];
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    (0..n)
        .map(|i| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let len = 4 + (state >> 58) as usize * if i % 97 == 0 { 400 } else { 3 };
            let mut text = format!("{{\"id\":{i},\"msg\":\"");
            for j in 0..len {
                text.push_str(words[((state >> (j % 50)) as usize + j) % words.len()]);
                text.push(' ');
            }
            text.push_str("\"}");
            Document {
                fields: vec![
                    StoredField {
                        field_number: 0,
                        value: FieldValue::String(text),
                    },
                    StoredField {
                        field_number: 1,
                        value: FieldValue::Long(i as i64),
                    },
                ],
            }
        })
        .collect()
}

/// Every chunk of a paged `.fdt` either starts on a page boundary or
/// ends inside the page it starts in.
fn assert_page_packed(reader: &StoredFieldsReader<'_>) {
    let page = ZSTD_PAGED_PAGE_SIZE as i64;
    for chunk in 0..reader.num_chunks() {
        let start = reader.block_start_pointer(chunk).unwrap();
        let end = if chunk + 1 < reader.num_chunks() {
            reader.block_start_pointer(chunk + 1).unwrap()
        } else {
            reader.max_pointer()
        };
        assert!(
            start % page == 0 || (end - 1) / page == start / page,
            "chunk {chunk} at {start}..{end} crosses a page"
        );
    }
}

#[test]
fn paged_zstd_round_trips_with_and_without_a_dictionary_and_packs_pages() {
    let docs = paged_docs(6000);
    let mut sizes = Vec::new();
    for with_dict in [false, true] {
        let mut w = StoredFieldsWriter::new(Mode::ZstdPaged(3), &id_write(), "");
        if with_dict {
            let samples: Vec<Vec<u8>> = docs[..2000]
                .chunks(20)
                .map(|c| {
                    let mut b = Vec::new();
                    for d in c {
                        serialize_doc_into(d, &mut b);
                    }
                    b
                })
                .collect();
            let refs: Vec<&[u8]> = samples.iter().map(Vec::as_slice).collect();
            let dict = lucene_zstd::train_dictionary(&refs, 16 << 10).unwrap();
            w.set_zstd_dictionary(&dict);
        }
        for d in &docs {
            w.add_document(d);
        }
        let (fdt, fdx, fdm) = w.finish();
        sizes.push(fdt.len());
        let reader = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
        assert_eq!(reader.data_codec(), DATA_CODEC_ZSTD_PAGED);
        assert_eq!(reader.has_zstd_dictionary(), with_dict);
        assert_eq!(reader.max_doc(), docs.len() as i32);
        if with_dict {
            assert_eq!(
                reader.block_start_pointer(0).unwrap() % ZSTD_PAGED_PAGE_SIZE as i64,
                0
            );
        }
        assert_page_packed(&reader);
        for (i, d) in docs.iter().enumerate() {
            let got = reader.document(i as i32).unwrap();
            assert_eq!(got.fields.len(), 2);
            assert_eq!(got.fields[0].value, d.fields[0].value, "doc {i}");
            assert_eq!(got.fields[1].value, d.fields[1].value);
        }
        // A sequential read sees the same documents.
        let chunk = reader.read_chunk(4321).unwrap();
        assert!(chunk.contains(4321));
    }
    assert!(sizes[1] < sizes[0], "a dictionary should help: {sizes:?}");
}

#[test]
fn paged_zstd_streams_the_same_bytes_it_buffers() {
    use lucene_store::Directory;
    let docs = paged_docs(20000);
    let mut buffered = StoredFieldsWriter::new(Mode::ZstdPaged(1), &id_write(), "");
    for d in &docs {
        buffered.add_document(d);
    }
    let (fdt, fdx, fdm) = buffered.finish();
    let root = lucene_util::test_support::TempDir::new("stored-paged-spill");
    let dir = lucene_store::FsDirectory::open(root.path());
    let mut streamed = StoredFieldsWriter::new(Mode::ZstdPaged(1), &id_write(), "");
    streamed.stream_to(dir.create_output("_0.fdt").unwrap());
    for d in &docs {
        streamed.add_document(d);
    }
    let (out, sfdx, sfdm) = streamed.finish_streamed();
    out.expect("streamed").close().unwrap();
    let sfdt = dir.open("_0.fdt").unwrap();
    assert!(sfdt.as_ref() == fdt.as_slice());
    assert_eq!((sfdx, sfdm), (fdx.clone(), fdm.clone()));
    let reader = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
    assert_page_packed(&reader);
    assert_eq!(
        reader.document(19999).unwrap().fields[1].value,
        FieldValue::Long(19999)
    );
}

fn id_write() -> [u8; ID_LENGTH] {
    [4u8; ID_LENGTH]
}

#[test]
fn write_best_speed_single_doc_round_trips_through_own_reader() {
    let docs = vec![Document {
        fields: vec![
            StoredField {
                field_number: 0,
                value: FieldValue::String("hello world".to_string()),
            },
            StoredField {
                field_number: 1,
                value: FieldValue::Int(-42),
            },
            StoredField {
                field_number: 2,
                value: FieldValue::Long(1_234_567_890_123),
            },
            StoredField {
                field_number: 3,
                value: FieldValue::Float(1.5),
            },
            StoredField {
                field_number: 4,
                value: FieldValue::Double(2.25),
            },
            StoredField {
                field_number: 5,
                value: FieldValue::Binary(vec![1, 2, 3, 4, 5]),
            },
        ],
    }];

    let (fdt, fdx, fdm) = write_best_speed(&docs, &id_write(), "");
    let reader = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
    assert_eq!(reader.max_doc(), 1);
    let got = reader.document(0).unwrap();
    assert_eq!(got.fields.len(), docs[0].fields.len());
    for (got_field, want_field) in got.fields.iter().zip(&docs[0].fields) {
        assert_eq!(got_field.field_number, want_field.field_number);
        assert_eq!(got_field.value, want_field.value);
    }
}

/// A wide document, read one field at a time through the visitor: the
/// subset visitor must see exactly the field it asked for, and the
/// values must equal what the whole-document path produces.
#[test]
fn visit_document_takes_only_the_fields_it_asks_for() {
    let docs = vec![wide_document()];
    let (fdt, fdx, fdm) = write_best_speed(&docs, &id_write(), "");
    let reader = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
    let whole = reader.document(0).unwrap();

    for want in &whole.fields {
        let mut visitor = DocumentVisitor::for_fields(&[want.field_number]);
        reader.visit_document(0, &mut visitor).unwrap();
        let got = visitor.into_document();
        assert_eq!(got.fields.len(), 1, "field {}", want.field_number);
        assert_eq!(got.fields[0].field_number, want.field_number);
        assert_eq!(got.fields[0].value, want.value);
    }

    // And a subset spanning both a skipped-then-taken and a
    // taken-then-skipped boundary.
    let mut visitor = DocumentVisitor::for_fields(&[1, 4]);
    reader.visit_document(0, &mut visitor).unwrap();
    let got = visitor.into_document();
    assert_eq!(
        got.fields
            .iter()
            .map(|f| f.field_number)
            .collect::<Vec<_>>(),
        vec![1, 4]
    );
}

/// `Status.STOP` must end the document where it is asked to, leaving
/// every later field undecoded -- and `Status.NO` on the *last* field
/// must take Java's "treat like STOP" shortcut rather than skipping into
/// the end of the buffer.
#[test]
fn visit_document_stops_early_and_skips_the_last_field_without_reading_it() {
    struct StopAfter {
        stop_at: i32,
        seen: Vec<i32>,
        taken: Vec<i32>,
    }
    impl StoredFieldVisitor for StopAfter {
        fn needs_field(&mut self, field_number: i32) -> Result<VisitStatus> {
            self.seen.push(field_number);
            Ok(if field_number == self.stop_at {
                VisitStatus::Stop
            } else {
                VisitStatus::Yes
            })
        }
        fn string_field(&mut self, field_number: i32, _value: &str) -> Result<()> {
            self.taken.push(field_number);
            Ok(())
        }
        fn int_field(&mut self, field_number: i32, _value: i32) -> Result<()> {
            self.taken.push(field_number);
            Ok(())
        }
        fn long_field(&mut self, field_number: i32, _value: i64) -> Result<()> {
            self.taken.push(field_number);
            Ok(())
        }
        fn float_field(&mut self, field_number: i32, _value: f32) -> Result<()> {
            self.taken.push(field_number);
            Ok(())
        }
        fn double_field(&mut self, field_number: i32, _value: f64) -> Result<()> {
            self.taken.push(field_number);
            Ok(())
        }
        fn binary_field(&mut self, field_number: i32, _value: &[u8]) -> Result<()> {
            self.taken.push(field_number);
            Ok(())
        }
    }

    let docs = vec![wide_document()];
    let (fdt, fdx, fdm) = write_best_speed(&docs, &id_write(), "");
    let reader = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();

    let mut visitor = StopAfter {
        stop_at: 2,
        seen: Vec::new(),
        taken: Vec::new(),
    };
    reader.visit_document(0, &mut visitor).unwrap();
    assert_eq!(visitor.seen, vec![0, 1, 2], "STOP did not end the walk");
    assert_eq!(visitor.taken, vec![0, 1]);

    // `NO` on the final field: the loop returns instead of calling
    // `skip_field`, so a truncated final value would not be noticed --
    // which is exactly Java's documented behaviour, and what this pins.
    let mut only_first = DocumentVisitor::for_fields(&[0]);
    reader.visit_document(0, &mut only_first).unwrap();
    assert_eq!(only_first.into_document().fields.len(), 1);
}

/// Six fields, one per encoding -- the shape a "wide document" test
/// needs so that every `skip_field` arm is exercised by a *reader*, not
/// only by the unit test that compares the two cursors.
fn wide_document() -> Document {
    Document {
        fields: vec![
            StoredField {
                field_number: 0,
                value: FieldValue::String("a fairly long stored value".to_string()),
            },
            StoredField {
                field_number: 1,
                value: FieldValue::Int(-42),
            },
            StoredField {
                field_number: 2,
                value: FieldValue::Long(1_234_567_890_123),
            },
            StoredField {
                field_number: 3,
                value: FieldValue::Float(1.5),
            },
            StoredField {
                field_number: 4,
                value: FieldValue::Double(2.25),
            },
            StoredField {
                field_number: 5,
                value: FieldValue::Binary(vec![7u8; 64]),
            },
        ],
    }
}

#[test]
fn write_best_speed_spans_many_chunks_and_every_doc_still_reads_back() {
    // Past `BEST_SPEED_MAX_DOCS_PER_CHUNK`, so the doc-count trigger fires
    // several times and the per-doc arrays cross whole transposed blocks.
    // Docs are small enough that the byte trigger never fires, which pins
    // the chunk boundaries at exact multiples of the doc cap.
    const N: usize = BEST_SPEED_MAX_DOCS_PER_CHUNK * 2 + 37;
    let docs: Vec<Document> = (0..N)
        .map(|i| Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String(format!("doc-{i}")),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::Long(i as i64 * -3),
                },
            ],
        })
        .collect();

    let (fdt, fdx, fdm) = write_best_speed(&docs, &id_write(), "seg");
    let reader = open(&fdt, &fdx, &fdm, &id_write(), "seg").unwrap();
    assert_eq!(reader.max_doc(), N as i32);

    for i in 0..N {
        let doc = reader.document(i as i32).unwrap();
        assert_eq!(doc.fields.len(), 2, "doc {i}");
        assert_eq!(
            doc.fields[0].value,
            FieldValue::String(format!("doc-{i}")),
            "doc {i}"
        );
        assert_eq!(
            doc.fields[1].value,
            FieldValue::Long(i as i64 * -3),
            "doc {i}"
        );
    }
}

/// Each document's chunk range is the chunk its bytes decode from: the
/// chunks tile the `.fdt` between its header and footer, a run of
/// documents in one chunk names it once, and documents outside the
/// segment name none.
#[test]
fn chunk_ranges_name_each_documents_chunk_once() {
    const N: usize = BEST_SPEED_MAX_DOCS_PER_CHUNK * 2 + 37;
    let docs: Vec<Document> = (0..N)
        .map(|i| Document {
            fields: vec![StoredField {
                field_number: 0,
                value: FieldValue::String(format!("doc-{i}")),
            }],
        })
        .collect();
    let (fdt, fdx, fdm) = write_best_speed(&docs, &id_write(), "seg");
    let reader = open(&fdt, &fdx, &fdm, &id_write(), "seg").unwrap();
    let all: Vec<i32> = (0..N as i32).collect();
    let ranges = reader.chunk_ranges(&all).unwrap();
    assert_eq!(ranges.len(), 3);
    for pair in ranges.windows(2) {
        assert_eq!(pair[0].end, pair[1].start);
    }
    assert!(ranges[2].end < fdt.len());
    let cap = BEST_SPEED_MAX_DOCS_PER_CHUNK as i32;
    assert_eq!(reader.chunk_range(0).unwrap(), Some(ranges[0].clone()));
    assert_eq!(reader.chunk_range(cap).unwrap(), Some(ranges[1].clone()));
    assert_eq!(
        reader.chunk_range(N as i32 - 1).unwrap(),
        Some(ranges[2].clone())
    );
    assert_eq!(reader.chunk_range(-1).unwrap(), None);
    assert_eq!(reader.chunk_range(N as i32).unwrap(), None);
    assert_eq!(
        reader
            .chunk_ranges(&[-5, 1, 2, cap + 1, N as i32 + 3])
            .unwrap(),
        vec![ranges[0].clone(), ranges[1].clone()]
    );
    // Bytes in memory have nothing to prefetch.
    reader.prefetch(&all).unwrap();
}

#[test]
fn write_best_speed_closes_a_chunk_on_the_byte_trigger_too() {
    // Each doc carries more than a tenth of the chunk size, so the byte
    // trigger closes chunks long before the doc cap is anywhere near.
    let big = "x".repeat(BEST_SPEED_CHUNK_SIZE / 8);
    let docs: Vec<Document> = (0..40)
        .map(|i| Document {
            fields: vec![StoredField {
                field_number: 0,
                value: FieldValue::String(format!("{i}-{big}")),
            }],
        })
        .collect();

    let (fdt, fdx, fdm) = write_best_speed(&docs, &id_write(), "seg");
    let reader = open(&fdt, &fdx, &fdm, &id_write(), "seg").unwrap();
    assert_eq!(reader.max_doc(), 40);
    for i in 0..40 {
        assert_eq!(
            reader.document(i).unwrap().fields[0].value,
            FieldValue::String(format!("{i}-{big}"))
        );
    }
}

#[test]
fn write_best_speed_multi_doc_round_trips_with_varying_field_counts() {
    let docs = vec![
        Document {
            fields: vec![StoredField {
                field_number: 0,
                value: FieldValue::String("doc0".to_string()),
            }],
        },
        Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("doc1-a".to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::Long(-7),
                },
            ],
        },
        Document { fields: vec![] },
    ];

    let (fdt, fdx, fdm) = write_best_speed(&docs, &id_write(), "seg");
    let reader = open(&fdt, &fdx, &fdm, &id_write(), "seg").unwrap();
    assert_eq!(reader.max_doc(), 3);

    let doc0 = reader.document(0).unwrap();
    assert_eq!(doc0.fields.len(), 1);
    assert_eq!(doc0.fields[0].value, FieldValue::String("doc0".to_string()));

    let doc1 = reader.document(1).unwrap();
    assert_eq!(doc1.fields.len(), 2);
    assert_eq!(
        doc1.fields[0].value,
        FieldValue::String("doc1-a".to_string())
    );
    assert_eq!(doc1.fields[1].value, FieldValue::Long(-7));

    let doc2 = reader.document(2).unwrap();
    assert_eq!(doc2.fields.len(), 0);
}

#[test]
fn write_best_speed_empty_doc_set_produces_zero_max_doc() {
    let (fdt, fdx, fdm) = write_best_speed(&[], &id_write(), "");
    let reader = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
    assert_eq!(reader.max_doc(), 0);
}

#[test]
fn write_best_compression_single_doc_round_trips_through_own_reader() {
    let docs = vec![Document {
        fields: vec![
            StoredField {
                field_number: 0,
                value: FieldValue::String("hello world".to_string()),
            },
            StoredField {
                field_number: 1,
                value: FieldValue::Int(-42),
            },
            StoredField {
                field_number: 2,
                value: FieldValue::Long(1_234_567_890_123),
            },
            StoredField {
                field_number: 3,
                value: FieldValue::Float(1.5),
            },
            StoredField {
                field_number: 4,
                value: FieldValue::Double(2.25),
            },
            StoredField {
                field_number: 5,
                value: FieldValue::Binary(vec![1, 2, 3, 4, 5]),
            },
        ],
    }];

    let (fdt, fdx, fdm) = write_best_compression(&docs, &id_write(), "");
    let reader = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
    assert_eq!(reader.max_doc(), 1);
    let got = reader.document(0).unwrap();
    assert_eq!(got.fields.len(), docs[0].fields.len());
    for (got_field, want_field) in got.fields.iter().zip(&docs[0].fields) {
        assert_eq!(got_field.field_number, want_field.field_number);
        assert_eq!(got_field.value, want_field.value);
    }
}

#[test]
fn write_best_compression_multi_doc_round_trips_with_varying_field_counts() {
    let docs = vec![
        Document {
            fields: vec![StoredField {
                field_number: 0,
                value: FieldValue::String("doc0".to_string()),
            }],
        },
        Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("doc1-a".to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::Long(-7),
                },
            ],
        },
        Document { fields: vec![] },
    ];

    let (fdt, fdx, fdm) = write_best_compression(&docs, &id_write(), "seg");
    let reader = open(&fdt, &fdx, &fdm, &id_write(), "seg").unwrap();
    assert_eq!(reader.max_doc(), 3);

    let doc0 = reader.document(0).unwrap();
    assert_eq!(doc0.fields.len(), 1);
    assert_eq!(doc0.fields[0].value, FieldValue::String("doc0".to_string()));

    let doc1 = reader.document(1).unwrap();
    assert_eq!(doc1.fields.len(), 2);
    assert_eq!(
        doc1.fields[0].value,
        FieldValue::String("doc1-a".to_string())
    );
    assert_eq!(doc1.fields[1].value, FieldValue::Long(-7));

    let doc2 = reader.document(2).unwrap();
    assert_eq!(doc2.fields.len(), 0);
}

#[test]
fn write_best_compression_empty_doc_set_produces_zero_max_doc() {
    let (fdt, fdx, fdm) = write_best_compression(&[], &id_write(), "");
    let reader = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
    assert_eq!(reader.max_doc(), 0);
}

#[test]
fn write_best_compression_large_payload_forces_multiple_sub_blocks() {
    // `dictLength = len / 60`, `blockLength = ceil((len - dictLength) / 10)`
    // (see write_best_compression's doc comment) -- a payload well past a
    // few KB guarantees `blockLength < len`, so `decompress_unit` must
    // walk more than one sub-block to reconstruct the chunk.
    let big_field = "the quick brown fox jumps over the lazy dog ".repeat(2000);
    let docs = vec![
        Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String(big_field.clone()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::Int(7),
                },
            ],
        },
        Document {
            fields: vec![StoredField {
                field_number: 0,
                value: FieldValue::String("second doc".to_string()),
            }],
        },
    ];

    let (fdt, fdx, fdm) = write_best_compression(&docs, &id_write(), "");
    let reader = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
    assert_eq!(reader.max_doc(), 2);

    let doc0 = reader.document(0).unwrap();
    assert_eq!(doc0.fields[0].value, FieldValue::String(big_field));
    assert_eq!(doc0.fields[1].value, FieldValue::Int(7));

    let doc1 = reader.document(1).unwrap();
    assert_eq!(
        doc1.fields[0].value,
        FieldValue::String("second doc".to_string())
    );
}

#[test]
fn write_zint_round_trips_through_read_zint() {
    for v in [0i32, 1, -1, i32::MIN, i32::MAX] {
        let mut out = Vec::new();
        write_zint(&mut out, v);
        let mut input = SliceInput::new(&out);
        assert_eq!(read_zint(&mut input).unwrap(), v, "value {v}");
    }
}

#[test]
fn write_tlong_round_trips_through_read_tlong_across_every_time_scale() {
    // One value per header branch: sub-5-bit, needing a continuation
    // vlong, exact-day, exact-hour (but not day), exact-second (but not
    // hour), and not a whole second at all -- plus the extremes.
    for v in [
        0i64,
        1,
        -1,
        15,
        -16,
        1_000_000_000_000,
        2 * DAY,
        -3 * DAY,
        5 * HOUR,
        7 * SECOND,
        1_234,
        i64::MIN,
        i64::MAX,
    ] {
        let mut out = Vec::new();
        write_tlong(&mut out, v);
        let mut input = SliceInput::new(&out);
        assert_eq!(read_tlong(&mut input).unwrap(), v, "value {v}");
    }
}

/// The whole point of `writeTLong`'s scale headers is that a timestamp
/// with day/hour/second precision costs far fewer bytes than its
/// magnitude would otherwise need. Assert the encoding actually shrinks,
/// not merely that it round-trips (which the always-10-byte encoding this
/// replaced also did).
#[test]
fn write_tlong_uses_the_scale_headers_to_shrink_timestamps() {
    let one_day_ms = DAY;
    let mut scaled = Vec::new();
    write_tlong(&mut scaled, one_day_ms);
    assert_eq!(scaled.len(), 1, "1 day should fit the header's low 5 bits");
    assert_eq!(scaled[0] & DAY_ENCODING_MASK, DAY_ENCODING);

    let mut unscaled = Vec::new();
    write_tlong(&mut unscaled, one_day_ms + 1); // not a whole second
    assert!(
        unscaled.len() > scaled.len(),
        "a non-second-aligned value must not get a scale header"
    );
    assert_eq!(unscaled[0] & DAY_ENCODING_MASK, 0);
}

#[test]
fn write_zfloat_round_trips_through_read_zfloat() {
    for v in [
        0.0f32,
        1.0,
        -1.0,
        125.0,
        126.0,
        1.5,
        -1.5,
        -0.0,
        f32::MIN,
        f32::MAX,
    ] {
        let mut out = Vec::new();
        write_zfloat(&mut out, v);
        let mut input = SliceInput::new(&out);
        let got = read_zfloat(&mut input).unwrap();
        assert_eq!(got, v, "value {v}");
        assert_eq!(
            got.is_sign_negative(),
            v.is_sign_negative(),
            "-0.0 must not collapse to +0.0 ({v})"
        );
    }
    // NaN survives as a NaN (Java canonicalizes it via floatToIntBits).
    let mut out = Vec::new();
    write_zfloat(&mut out, f32::NAN);
    assert!(read_zfloat(&mut SliceInput::new(&out)).unwrap().is_nan());
}

/// Small integral floats must take the one-byte encoding, and larger
/// ones the 4/5-byte ones -- the sizes are the reason this encoding
/// exists, so assert them rather than only the round trip.
#[test]
fn write_zfloat_uses_the_shortest_encoding_per_branch() {
    for (v, expected_len) in [
        (-1.0f32, 1usize),
        (0.0, 1),
        (125.0, 1),
        (126.0, 4), // past the small-integer range, still positive
        (1.5, 4),   // positive, not integral
        (-0.0, 5),  // negative zero is excluded from the small-int case
        (-1.5, 5),  // negative
    ] {
        let mut out = Vec::new();
        write_zfloat(&mut out, v);
        assert_eq!(out.len(), expected_len, "value {v}");
    }
}

#[test]
fn write_zdouble_round_trips_through_read_zdouble() {
    for v in [
        0.0f64,
        1.0,
        -1.0,
        124.0,
        125.0,
        1.5,
        -1.5,
        -0.0,
        f64::MIN,
        f64::MAX,
        std::f64::consts::PI,
    ] {
        let mut out = Vec::new();
        write_zdouble(&mut out, v);
        let mut input = SliceInput::new(&out);
        let got = read_zdouble(&mut input).unwrap();
        assert_eq!(got, v, "value {v}");
        assert_eq!(
            got.is_sign_negative(),
            v.is_sign_negative(),
            "-0.0 must not collapse to +0.0 ({v})"
        );
    }
    let mut out = Vec::new();
    write_zdouble(&mut out, f64::NAN);
    assert!(read_zdouble(&mut SliceInput::new(&out)).unwrap().is_nan());
}

#[test]
fn write_zdouble_uses_the_shortest_encoding_per_branch() {
    for (v, expected_len) in [
        (-1.0f64, 1usize),
        (0.0, 1),
        (124.0, 1),
        (125.0, 5),                 // past small-int, exact as f32
        (1.5, 5),                   // exact as f32
        (-1.5, 5),                  // negative but exact as f32
        (std::f64::consts::PI, 8),  // positive, not f32-exact
        (-std::f64::consts::PI, 9), // negative, not f32-exact
    ] {
        let mut out = Vec::new();
        write_zdouble(&mut out, v);
        assert_eq!(out.len(), expected_len, "value {v}");
    }
}

/// A single document larger than `2 * chunkSize` makes the writer mark
/// the chunk `sliced` and split it into several independent
/// `chunk_size`-plaintext compression units. The reader then has to cut
/// the wanted byte range against each unit's own extent -- the one place
/// two levels of splitting (chunk into units, unit into sub-blocks)
/// compose. No hand-built fixture covers this; only the real writer
/// produces it.
#[test]
fn writer_produced_sliced_chunk_round_trips_through_the_reader() {
    // ~230KB in one document: past 2 * BEST_SPEED_CHUNK_SIZE (160KB), so
    // three units. Compressible-but-not-trivially so, to keep the LZ4
    // sub-blocks from degenerating into single long matches.
    let big: Vec<u8> = (0..230_000u32)
        .map(|i| (i.wrapping_mul(2_654_435_761) >> 24) as u8)
        .collect();
    let docs = vec![
        Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::Binary(big.clone()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::String("trailing marker".to_string()),
                },
            ],
        },
        Document {
            fields: vec![StoredField {
                field_number: 0,
                value: FieldValue::String("next chunk".to_string()),
            }],
        },
    ];

    let (fdt, fdx, fdm) = write_best_speed(&docs, &id_write(), "");

    // Prove the chunk really is sliced, rather than assuming the size
    // arithmetic worked out: re-read the first chunk's token.
    let reader = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
    let mut probe = SliceInput::new(&fdt);
    let block_start = reader.block_start_pointer(0).unwrap();
    probe.seek(block_start as usize).unwrap();
    probe.read_vint().unwrap(); // docBase
    let token = probe.read_vint().unwrap();
    assert_eq!(token & 1, 1, "expected the first chunk to be marked sliced");

    assert_eq!(reader.max_doc(), 2);
    let doc0 = reader.document(0).unwrap();
    assert_eq!(doc0.fields.len(), 2);
    assert_eq!(doc0.fields[0].value, FieldValue::Binary(big));
    assert_eq!(
        doc0.fields[1].value,
        FieldValue::String("trailing marker".to_string())
    );
    assert_eq!(
        reader.document(1).unwrap().fields[0].value,
        FieldValue::String("next chunk".to_string())
    );
}

/// The reader now decompresses only the sub-blocks a document actually
/// intersects (skipping the rest by their recorded compressed length),
/// so the *position* of a document inside the chunk is a real code path:
/// wholly inside the dictionary prefix, spanning the dictionary/first
/// sub-block seam, in the middle, and in the last sub-block.
#[test]
fn every_document_position_inside_a_multi_sub_block_chunk_reads_back() {
    // ~40KB of payload in one chunk: dictLength = 40000/20 = 2000 and
    // ten ~3.8KB sub-blocks, so documents land in all of the above.
    let docs: Vec<Document> = (0..400i32)
        .map(|i| Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String(format!("document number {i} ").repeat(5)),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::Int(i),
                },
            ],
        })
        .collect();

    let (fdt, fdx, fdm) = write_best_speed(&docs, &id_write(), "");
    let reader = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
    assert_eq!(reader.max_doc(), 400);
    for i in 0..400i32 {
        let got = reader.document(i).unwrap();
        assert_eq!(got.fields.len(), 2, "doc {i}");
        assert_eq!(
            got.fields[0].value,
            FieldValue::String(format!("document number {i} ").repeat(5)),
            "doc {i}"
        );
        assert_eq!(got.fields[1].value, FieldValue::Int(i), "doc {i}");
    }
}

/// Same, for BEST_COMPRESSION: DEFLATE's sub-block lengths are
/// interleaved rather than batched up front, so its skip path is a
/// separate branch in `decompress_unit`.
#[test]
fn every_document_position_inside_a_deflate_multi_sub_block_chunk_reads_back() {
    let docs: Vec<Document> = (0..300)
        .map(|i| Document {
            fields: vec![StoredField {
                field_number: 0,
                value: FieldValue::String(format!("deflate doc {i} ").repeat(9)),
            }],
        })
        .collect();

    let (fdt, fdx, fdm) = write_best_compression(&docs, &id_write(), "");
    let reader = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
    for i in 0..300 {
        assert_eq!(
            reader.document(i).unwrap().fields[0].value,
            FieldValue::String(format!("deflate doc {i} ").repeat(9)),
            "doc {i}"
        );
    }
}

/// A `.fdt` whose unit header claims `blockLength == 0` while the unit
/// still has bytes left to cover would spin forever counting blocks (Java
/// divides by it and throws). It must be reported as corruption.
/// `.fdm`'s `chunkSize` is what a `sliced` chunk is divided by; zero
/// would make the reader's unit loop make no progress.
#[test]
fn zero_chunk_size_in_meta_is_rejected() {
    let doc_bytes = field_bytes(0, TYPE_STRING, &string_field_payload("x"));
    let (fdt, fdx, mut fdm) = build_single_chunk_index(&doc_bytes);
    // Rewrite the chunkSize vint (first byte after the index header) to 0
    // and re-checksum. `build_single_chunk_index` writes 80*1024, which is
    // a 3-byte vint, so overwrite all three with a single 0 plus padding
    // the reader will never reach... simpler: rebuild via a targeted patch.
    let header_len = 4 + 1 + META_CODEC.len() + 4 + ID_LENGTH + 1;
    // 80*1024 = 0x14000 -> vint bytes [0x80, 0x80, 0x05]; make it [0x80, 0x80, 0x00] = 0.
    fdm[header_len + 2] = 0x00;
    let body_len = fdm.len() - codec_util::FOOTER_LENGTH;
    let checksum = crc32fast::hash(&fdm[..body_len + 8]) as u64;
    fdm[body_len + 8..].copy_from_slice(&checksum.to_be_bytes());
    assert!(open(&fdt, &fdx, &fdm, &id(), "").is_err());
}

#[test]
fn zero_block_length_in_a_unit_header_is_rejected_not_hung_on() {
    let mut unit = Vec::new();
    write_vint(&mut unit, 0); // dictLength
    write_vint(&mut unit, 0); // blockLength = 0, but originalLength > 0
    let mut input = SliceInput::new(&unit);
    let mut out = Vec::new();
    let err = decompress_unit(Mode::BestSpeed, &mut input, 8, 0, 8, &mut out).unwrap_err();
    assert!(
        matches!(err, Error::Store(lucene_store::Error::Corrupted(_))),
        "unexpected error: {err:?}"
    );
}

/// A dictionary longer than the whole unit is likewise impossible and
/// would otherwise make the block loop's `original_length - plain`
/// underflow.
#[test]
fn dictionary_longer_than_the_unit_is_rejected() {
    let mut unit = Vec::new();
    write_vint(&mut unit, 100); // dictLength > originalLength
    write_vint(&mut unit, 4);
    let mut input = SliceInput::new(&unit);
    let mut out = Vec::new();
    assert!(decompress_unit(Mode::BestSpeed, &mut input, 8, 0, 8, &mut out).is_err());
}

#[test]
fn write_bulk_ints_round_trips_across_whole_transposed_blocks() {
    // Sizes that straddle the 128-value block boundary in every direction:
    // just under one block, exactly one, one plus a tail, and several.
    for count in [127usize, 128, 129, 200, 256, 300] {
        for width_cap in [0xFFi64, 0xFFFF, 0xFFFF_FFFFu32 as i64] {
            // Vary the values so the all-equal shortcut never fires, and
            // make each position distinguishable so a transposition slip
            // shows up as a mismatch rather than a coincidence.
            let values: Vec<i64> = (0..count)
                .map(|k| {
                    if k == 0 {
                        width_cap
                    } else {
                        (k as i64) % width_cap
                    }
                })
                .collect();
            let mut out = Vec::new();
            write_bulk_ints(&mut out, &values);
            let mut input = SliceInput::new(&out);
            assert_eq!(
                read_bulk_ints(&mut input, count).unwrap(),
                values,
                "count={count} width_cap={width_cap}"
            );
        }
    }
}

#[test]
fn write_bulk_ints_all_equal_and_varying_widths_round_trip() {
    for values in [
        vec![5i64, 5, 5, 5],
        vec![1i64, 200, 3, 4],
        vec![1i64, 70000, 3, 4],
        vec![1i64, 4_000_000_000, 3, 4],
    ] {
        let mut out = Vec::new();
        write_bulk_ints(&mut out, &values);
        let mut input = SliceInput::new(&out);
        assert_eq!(read_bulk_ints(&mut input, values.len()).unwrap(), values);
    }
}

#[test]
fn encode_literal_lz4_round_trips_through_lz4_decompress() {
    for payload in [
        Vec::new(),
        b"short".to_vec(),
        vec![0x42; 5000], // forces the 0xFF-continuation length encoding
    ] {
        let encoded = encode_literal_lz4(&payload);
        let mut input = SliceInput::new(&encoded);
        let mut dest = vec![0u8; payload.len()];
        crate::lz4::decompress(&mut input, payload.len(), &mut dest, 0).unwrap();
        assert_eq!(dest, payload);
    }
}

// --- streaming writer: the three merge paths ---

/// `N` small documents, deterministic and self-identifying so a
/// bulk-copy boundary error shows up as a document holding another
/// document's values rather than as a decode error.
fn merge_docs(tag: &str, n: usize) -> Vec<Document> {
    (0..n)
        .map(|i| Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String(format!("{tag}-{i}")),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::Long(i as i64 * 7 - 3),
                },
            ],
        })
        .collect()
}

fn assert_doc_is(reader: &StoredFieldsReader<'_>, doc_id: i32, tag: &str, i: usize) {
    let doc = reader.document(doc_id).unwrap();
    assert_eq!(doc.fields.len(), 2, "doc {doc_id}");
    assert_eq!(
        doc.fields[0].value,
        FieldValue::String(format!("{tag}-{i}")),
        "doc {doc_id} field 0"
    );
    assert_eq!(
        doc.fields[1].value,
        FieldValue::Long(i as i64 * 7 - 3),
        "doc {doc_id} field 1"
    );
}

#[test]
fn copy_chunks_of_two_whole_segments_reproduces_every_document() {
    // Both sources span several full chunks plus a trailing dirty one, so
    // the copy loop runs over real chunk boundaries. Each source is copied
    // whole, which is exactly what an unsorted, deletion-free merge does.
    const N: usize = BEST_SPEED_MAX_DOCS_PER_CHUNK * 2 + 51;
    let a = merge_docs("a", N);
    let b = merge_docs("b", N);
    let (fdt_a, fdx_a, fdm_a) = write_best_speed(&a, &id_write(), "");
    let (fdt_b, fdx_b, fdm_b) = write_best_speed(&b, &id_write(), "");
    let ra = open(&fdt_a, &fdx_a, &fdm_a, &id_write(), "").unwrap();
    let rb = open(&fdt_b, &fdx_b, &fdm_b, &id_write(), "").unwrap();
    assert!(ra.num_chunks() >= 3, "source should span several chunks");

    let mut w = StoredFieldsWriter::new(Mode::BestSpeed, &id_write(), "");
    assert!(w.can_bulk_copy(&ra) && w.can_bulk_copy(&rb));
    w.copy_chunks(&ra, 0, N as i32).unwrap();
    w.copy_chunks(&rb, 0, N as i32).unwrap();
    let (fdt, fdx, fdm) = w.finish();

    let merged = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
    assert_eq!(merged.max_doc(), 2 * N as i32);
    for i in 0..N {
        assert_doc_is(&merged, i as i32, "a", i);
        assert_doc_is(&merged, (N + i) as i32, "b", i);
    }
    // Every chunk was copied, so nothing was recompressed: the merged
    // `.fdt` is the two sources' chunk bodies plus rebased headers.
    assert_eq!(merged.num_chunks(), ra.num_chunks() + rb.num_chunks());
}

#[test]
fn a_streamed_fdt_is_byte_for_byte_the_buffered_one() {
    use lucene_store::Directory;
    // Incompressible values, so the `.fdt` passes `SPILL_BYTES` several
    // times and the chunk pointers cross spills.
    let mut x: u64 = 0x9e37_79b9_7f4a_7c15;
    let docs: Vec<Document> = (0..3000)
        .map(|i| {
            let bytes: Vec<u8> = (0..1500)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    x as u8
                })
                .collect();
            Document {
                fields: vec![
                    StoredField {
                        field_number: 0,
                        value: FieldValue::Binary(bytes),
                    },
                    StoredField {
                        field_number: 1,
                        value: FieldValue::Long(i),
                    },
                ],
            }
        })
        .collect();
    for mode in [Mode::BestSpeed, Mode::BestCompression] {
        let mut buffered = StoredFieldsWriter::new(mode, &id_write(), "");
        for d in &docs {
            buffered.add_document(d);
        }
        let (fdt, fdx, fdm) = buffered.finish();
        assert!(fdt.len() > 3 * SPILL_BYTES, "{} bytes", fdt.len());

        let root = lucene_util::test_support::TempDir::new("stored-spill");
        let dir = lucene_store::FsDirectory::open(root.path());
        let mut streamed = StoredFieldsWriter::new(mode, &id_write(), "");
        streamed.stream_to(dir.create_output("_0.fdt").unwrap());
        for d in &docs {
            streamed.add_document(d);
        }
        let (out, sfdx, sfdm) = streamed.finish_streamed();
        out.expect("streamed").close().unwrap();
        let sfdt = dir.open("_0.fdt").unwrap();
        assert!(sfdt.as_ref() == fdt.as_slice(), "{mode:?} .fdt differs");
        assert_eq!((sfdx, sfdm), (fdx.clone(), fdm.clone()), "{mode:?}");
        let reader = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
        assert_eq!(reader.max_doc(), 3000);
        let last = reader.document(2999).unwrap();
        assert!(matches!(last.fields[1].value, FieldValue::Long(2999)));
        match (&last.fields[0].value, &docs[2999].fields[0].value) {
            (FieldValue::Binary(a), FieldValue::Binary(b)) => assert_eq!(a, b),
            other => panic!("{other:?}"),
        }
    }
}

#[test]
fn copy_chunks_of_a_partial_range_copies_the_ragged_ends_document_at_a_time() {
    // A run that starts and ends mid-chunk -- what a sorted merge produces
    // -- must still reproduce exactly those documents. Java reaches the
    // same two document-at-a-time loops through its `isLoaded(docID)`
    // check; this port tests the chunk-boundary condition directly.
    const N: usize = BEST_SPEED_MAX_DOCS_PER_CHUNK * 3;
    let docs = merge_docs("p", N);
    let (fdt_s, fdx_s, fdm_s) = write_best_speed(&docs, &id_write(), "");
    let src = open(&fdt_s, &fdx_s, &fdm_s, &id_write(), "").unwrap();
    assert_eq!(src.num_chunks(), 3);

    let from = 5i32;
    let to = (BEST_SPEED_MAX_DOCS_PER_CHUNK * 2 + 9) as i32;
    let mut w = StoredFieldsWriter::new(Mode::BestSpeed, &id_write(), "");
    w.copy_chunks(&src, from, to).unwrap();
    let (fdt, fdx, fdm) = w.finish();

    let merged = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
    assert_eq!(merged.max_doc(), to - from);
    for (merged_id, source_id) in (from..to).enumerate() {
        assert_doc_is(&merged, merged_id as i32, "p", source_id as usize);
    }
}

#[test]
fn copy_chunks_of_a_range_inside_one_chunk_copies_no_chunk_at_all() {
    // `from_pointer == to_pointer`: no whole chunk fits, so every document
    // goes through the DOC path and the bulk loop must not run.
    let docs = merge_docs("q", 40);
    let (fdt_s, fdx_s, fdm_s) = write_best_speed(&docs, &id_write(), "");
    let src = open(&fdt_s, &fdx_s, &fdm_s, &id_write(), "").unwrap();
    assert_eq!(src.num_chunks(), 1);

    let mut w = StoredFieldsWriter::new(Mode::BestSpeed, &id_write(), "");
    w.copy_chunks(&src, 10, 20).unwrap();
    let (fdt, fdx, fdm) = w.finish();
    let merged = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
    assert_eq!(merged.max_doc(), 10);
    for (merged_id, source_id) in (10..20).enumerate() {
        assert_doc_is(&merged, merged_id as i32, "q", source_id);
    }
}

#[test]
fn copy_chunks_after_buffered_documents_forces_a_dirty_flush_first() {
    // Java's `if (numBufferedDocs > 0) flush(true);`: a bulk copy appends
    // whole chunks, so anything already buffered has to close as its own
    // (dirty) chunk first, or the copied chunks' `docBase`s would be wrong.
    let src_docs = merge_docs("s", BEST_SPEED_MAX_DOCS_PER_CHUNK + 5);
    let (fdt_s, fdx_s, fdm_s) = write_best_speed(&src_docs, &id_write(), "");
    let src = open(&fdt_s, &fdx_s, &fdm_s, &id_write(), "").unwrap();

    let mut w = StoredFieldsWriter::new(Mode::BestSpeed, &id_write(), "");
    for doc in merge_docs("v", 3).iter() {
        w.add_document(doc);
    }
    w.copy_chunks(&src, 0, src_docs.len() as i32).unwrap();
    let (fdt, fdx, fdm) = w.finish();

    let merged = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
    assert_eq!(merged.max_doc(), (3 + src_docs.len()) as i32);
    for i in 0..3 {
        assert_doc_is(&merged, i as i32, "v", i);
    }
    for i in 0..src_docs.len() {
        assert_doc_is(&merged, (3 + i) as i32, "s", i);
    }
    // The three buffered docs became a dirty chunk of their own, and the
    // source's own trailing dirty chunk came across with its bit set.
    assert!(merged.num_dirty_chunks() >= 2);
    assert!(merged.num_dirty_docs() >= 3 + 5);
}

#[test]
fn serialized_documents_round_trip_through_the_doc_path() {
    // The DOC path: bytes copied without ever being parsed into fields.
    let docs = merge_docs("d", 300);
    let (fdt_s, fdx_s, fdm_s) = write_best_speed(&docs, &id_write(), "");
    let src = open(&fdt_s, &fdx_s, &fdm_s, &id_write(), "").unwrap();

    let mut w = StoredFieldsWriter::new(Mode::BestSpeed, &id_write(), "");
    // Only the even documents, the way a source with deletions merges.
    for i in (0..docs.len()).step_by(2) {
        let d = src.serialized_document(i as i32).unwrap();
        w.add_serialized_document(d.num_stored_fields, &d.bytes);
    }
    let (fdt, fdx, fdm) = w.finish();
    let merged = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
    assert_eq!(merged.max_doc(), 150);
    for (merged_id, source_id) in (0..docs.len()).step_by(2).enumerate() {
        assert_doc_is(&merged, merged_id as i32, "d", source_id);
    }
}

#[test]
fn a_documents_serialized_bytes_are_exactly_what_the_writer_buffered() {
    // `serialized_document` is only safe to copy if it is byte-identical
    // to what `serialize_doc_into` produced -- otherwise the DOC path
    // silently rewrites documents.
    let docs = vec![Document {
        fields: vec![
            StoredField {
                field_number: 2,
                value: FieldValue::String("hello".to_string()),
            },
            StoredField {
                field_number: 9,
                value: FieldValue::Binary(vec![0, 1, 255, 7]),
            },
            StoredField {
                field_number: 0,
                value: FieldValue::Double(-0.5),
            },
        ],
    }];
    let (fdt, fdx, fdm) = write_best_speed(&docs, &id_write(), "");
    let reader = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
    let got = reader.serialized_document(0).unwrap();
    let mut want = Vec::new();
    serialize_doc_into(&docs[0], &mut want);
    assert_eq!(got.bytes, want);
    assert_eq!(got.num_stored_fields, 3);
}

#[test]
fn an_empty_document_serializes_to_no_bytes_and_no_fields() {
    let docs = vec![Document::default(), merge_docs("e", 1).remove(0)];
    let (fdt, fdx, fdm) = write_best_speed(&docs, &id_write(), "");
    let reader = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
    let empty = reader.serialized_document(0).unwrap();
    assert_eq!(empty.num_stored_fields, 0);
    assert!(empty.bytes.is_empty());
    assert!(reader.document(0).unwrap().fields.is_empty());
}

#[test]
fn dirtiness_accumulates_across_bulk_copies_until_the_segment_is_too_dirty() {
    // `tooDirty` is Java's safety switch against a segment's compression
    // ratio degrading forever: every generation of bulk copies carries the
    // previous generation's incomplete chunks across verbatim. A single
    // segment this port flushes can never be too dirty (its one forced
    // flush leaves fewer than `maxDocsPerChunk` dirty docs) -- but a
    // *merged* one accumulates one dirty chunk per bulk-copied source.
    let docs = merge_docs("t", 1);
    let (fdt_s, fdx_s, fdm_s) = write_best_speed(&docs, &id_write(), "");
    let src = open(&fdt_s, &fdx_s, &fdm_s, &id_write(), "").unwrap();
    assert_eq!(src.num_dirty_chunks(), 1);

    let mut w = StoredFieldsWriter::new(Mode::BestSpeed, &id_write(), "");
    assert!(
        !w.too_dirty(&src),
        "a single flushed segment is never dirty"
    );
    let copies = BEST_SPEED_MAX_DOCS_PER_CHUNK + 1;
    for _ in 0..copies {
        w.copy_chunks(&src, 0, 1).unwrap();
    }
    let (fdt, fdx, fdm) = w.finish();
    let merged = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
    assert_eq!(merged.max_doc(), copies as i32);
    assert_eq!(merged.num_dirty_docs(), copies as i64);
    assert_eq!(merged.num_dirty_chunks(), copies as i64);
    for i in 0..copies {
        assert_doc_is(&merged, i as i32, "t", 0);
    }

    let w2 = StoredFieldsWriter::new(Mode::BestSpeed, &id_write(), "");
    assert!(w2.too_dirty(&merged));
    assert!(!w2.can_bulk_copy(&merged));
}

#[test]
fn a_best_compression_source_cannot_be_bulk_copied_into_a_best_speed_writer() {
    // Different compressor, so the chunk payload framing differs: copying
    // it verbatim would produce bytes the merged segment's own reader
    // decodes with the wrong decompressor.
    let docs = merge_docs("c", 20);
    let (fdt, fdx, fdm) = write_best_compression(&docs, &id_write(), "");
    let src = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
    assert_eq!(src.data_codec(), Mode::BestCompression.data_codec());
    assert_eq!(src.chunk_size(), BEST_COMPRESSION_CHUNK_SIZE as i32);

    let w = StoredFieldsWriter::new(Mode::BestSpeed, &id_write(), "");
    assert!(!w.can_bulk_copy(&src));
    // ...but the DOC path is compressor-independent: serialized bytes are
    // the same either way.
    let mut w = w;
    for i in 0..docs.len() {
        let d = src.serialized_document(i as i32).unwrap();
        w.add_serialized_document(d.num_stored_fields, &d.bytes);
    }
    let (fdt, fdx, fdm) = w.finish();
    let merged = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
    assert_eq!(merged.data_codec(), Mode::BestSpeed.data_codec());
    for i in 0..docs.len() {
        assert_doc_is(&merged, i as i32, "c", i);
    }
}

#[test]
fn copy_chunks_rejects_an_out_of_range_document_range() {
    let docs = merge_docs("r", 10);
    let (fdt, fdx, fdm) = write_best_speed(&docs, &id_write(), "");
    let src = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
    let mut w = StoredFieldsWriter::new(Mode::BestSpeed, &id_write(), "");
    assert!(matches!(
        w.copy_chunks(&src, 0, 11),
        Err(Error::DocOutOfRange(_, 10))
    ));
    assert!(matches!(
        w.copy_chunks(&src, 5, 4),
        Err(Error::InvertedDocRange {
            from_doc: 5,
            to_doc: 4
        })
    ));
    // And a writer that cannot legally bulk-copy from this reader at all
    // is refused in release builds too, not only under `debug_assert`.
    let mut deflate_writer = StoredFieldsWriter::new(Mode::BestCompression, &id_write(), "");
    assert!(matches!(
        deflate_writer.copy_chunks(&src, 0, 10),
        Err(Error::BulkCopyNotPermitted {
            same_mode: false,
            ..
        })
    ));
}

#[test]
fn copy_chunks_of_an_empty_range_writes_nothing() {
    let docs = merge_docs("z", 10);
    let (fdt, fdx, fdm) = write_best_speed(&docs, &id_write(), "");
    let src = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
    let mut w = StoredFieldsWriter::new(Mode::BestSpeed, &id_write(), "");
    w.copy_chunks(&src, 3, 3).unwrap();
    let (fdt, fdx, fdm) = w.finish();
    let merged = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
    assert_eq!(merged.max_doc(), 0);
    assert_eq!(merged.num_chunks(), 0);
}

#[test]
fn chunk_for_doc_reports_the_containing_chunks_base_and_offset() {
    const N: usize = BEST_SPEED_MAX_DOCS_PER_CHUNK * 2 + 3;
    let docs = merge_docs("k", N);
    let (fdt, fdx, fdm) = write_best_speed(&docs, &id_write(), "");
    let reader = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
    let cap = BEST_SPEED_MAX_DOCS_PER_CHUNK as i32;
    assert_eq!(reader.chunk_for_doc(0).unwrap().doc_base, 0);
    assert_eq!(reader.chunk_for_doc(cap - 1).unwrap().doc_base, 0);
    assert_eq!(reader.chunk_for_doc(cap).unwrap().doc_base, cap);
    assert_eq!(reader.chunk_for_doc(2 * cap).unwrap().doc_base, 2 * cap);
    // Start pointers are strictly increasing and the last chunk's data
    // ends at `maxPointer`.
    assert!(
        reader.chunk_for_doc(0).unwrap().start_pointer
            < reader.chunk_for_doc(cap).unwrap().start_pointer
    );
    assert!(reader.chunk_for_doc(2 * cap).unwrap().start_pointer < reader.max_pointer());
    assert!(reader.chunk_for_doc(N as i32).is_err());
}

#[test]
fn a_chunk_cursor_serves_every_document_of_a_chunk_from_one_decompression() {
    const N: usize = BEST_SPEED_MAX_DOCS_PER_CHUNK * 2 + 17;
    let docs = merge_docs("cc", N);
    let (fdt, fdx, fdm) = write_best_speed(&docs, &id_write(), "");
    let reader = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();

    let mut cursor = ChunkCursor::new();
    for i in 0..N {
        let (num_stored_fields, bytes) = cursor.document(&reader, i as i32).unwrap();
        assert_eq!(num_stored_fields, 2, "doc {i}");
        // Identical to the random-access read, byte for byte.
        let random = reader.serialized_document(i as i32).unwrap();
        assert_eq!(bytes, random.bytes.as_slice(), "doc {i}");
        let parsed = parse_document(num_stored_fields, bytes).unwrap();
        let direct = reader.document(i as i32).unwrap();
        assert_eq!(parsed.fields.len(), direct.fields.len(), "doc {i}");
        for (a, b) in parsed.fields.iter().zip(&direct.fields) {
            assert_eq!(a.field_number, b.field_number, "doc {i}");
            assert_eq!(a.value, b.value, "doc {i}");
        }
    }

    // Backwards, and jumping across chunks, must reload rather than
    // return a stale chunk's document.
    for i in (0..N).rev().step_by(7) {
        let (_, bytes) = cursor.document(&reader, i as i32).unwrap();
        assert_eq!(
            bytes,
            reader
                .serialized_document(i as i32)
                .unwrap()
                .bytes
                .as_slice(),
            "doc {i}"
        );
    }
    cursor.reset();
    let (_, bytes) = cursor.document(&reader, 0).unwrap();
    assert_eq!(
        bytes,
        reader.serialized_document(0).unwrap().bytes.as_slice()
    );
}

#[test]
fn read_chunk_reports_its_own_extent_and_rejects_documents_outside_it() {
    const N: usize = BEST_SPEED_MAX_DOCS_PER_CHUNK + 40;
    let docs = merge_docs("rc", N);
    let (fdt, fdx, fdm) = write_best_speed(&docs, &id_write(), "");
    let reader = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();

    let first = reader.read_chunk(0).unwrap();
    assert_eq!(first.doc_base(), 0);
    assert_eq!(first.num_docs(), BEST_SPEED_MAX_DOCS_PER_CHUNK as i32);
    assert!(first.contains(0) && first.contains(first.num_docs() - 1));
    assert!(!first.contains(first.num_docs()));
    assert!(first.document(first.num_docs()).is_none());

    let second = reader
        .read_chunk(BEST_SPEED_MAX_DOCS_PER_CHUNK as i32)
        .unwrap();
    assert_eq!(second.doc_base(), BEST_SPEED_MAX_DOCS_PER_CHUNK as i32);
    assert_eq!(second.num_docs(), 40);
    assert!(reader.read_chunk(N as i32).is_err());
}

#[test]
fn a_chunk_holding_only_empty_documents_decompresses_to_nothing() {
    // Every document has zero stored fields, so the chunk's payload is
    // zero bytes -- `read_chunk` must not try to decompress an empty unit.
    let docs = vec![Document::default(); 5];
    let (fdt, fdx, fdm) = write_best_speed(&docs, &id_write(), "");
    let reader = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
    let chunk = reader.read_chunk(0).unwrap();
    assert_eq!(chunk.num_docs(), 5);
    for i in 0..5 {
        let (n, bytes) = chunk.document(i).unwrap();
        assert_eq!(n, 0);
        assert!(bytes.is_empty());
        assert!(parse_document(n, bytes).unwrap().fields.is_empty());
    }
}

/// A two-chunk source plus a mutation applied to its `.fdt`. Every
/// mutation here replaces a vint with another of the **same length**, so
/// `open`'s `maxPointer`-vs-file-length cross-check still passes and the
/// corruption really does have to be caught by `copy_chunks` itself.
fn corrupted_source(mutate: impl Fn(&mut Vec<u8>, usize)) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    const N: usize = BEST_SPEED_MAX_DOCS_PER_CHUNK * 2;
    let docs = merge_docs("x", N);
    let (mut fdt, fdx, fdm) = write_best_speed(&docs, &id_write(), "");
    let second_chunk = {
        let reader = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
        reader
            .chunk_for_doc(BEST_SPEED_MAX_DOCS_PER_CHUNK as i32)
            .unwrap()
            .start_pointer as usize
    };
    mutate(&mut fdt, second_chunk);
    (fdt, fdx, fdm)
}

#[test]
fn a_chunk_header_whose_doc_base_disagrees_with_the_index_is_rejected() {
    // Java's `if (base != docID) throw new CorruptIndexException(...)`.
    // The `.fdx` index and the `.fdt` chunk headers are redundant on
    // purpose: disagreeing is how a bad bulk copy announces itself before
    // it writes a segment that reads back plausible but wrong documents.
    let (fdt, fdx, fdm) = corrupted_source(|fdt, at| {
        // docBase 1024 (vint 0x80 0x08) -> 1025 (vint 0x81 0x08).
        assert_eq!(&fdt[at..at + 2], &[0x80, 0x08]);
        fdt[at] = 0x81;
    });
    let reader = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
    let mut w = StoredFieldsWriter::new(Mode::BestSpeed, &id_write(), "");
    assert!(matches!(
        w.copy_chunks(&reader, 0, reader.max_doc()),
        Err(Error::CorruptChunkBounds { .. })
    ));
}

#[test]
fn a_chunk_header_claiming_no_documents_is_rejected() {
    // Not a case Java has to guard (its own writer never emits it), but a
    // zero-document chunk would make the copy loop spin without ever
    // advancing `docID`.
    let (fdt, fdx, fdm) = corrupted_source(|fdt, at| {
        // Skip the docBase vint, then blank the token to a non-canonical
        // two-byte encoding of 0 so the chunk's length is unchanged.
        let token_at = at + 2;
        assert_eq!(&fdt[token_at..token_at + 2], &[0x80, 0x20]);
        fdt[token_at] = 0x80;
        fdt[token_at + 1] = 0x00;
    });
    let reader = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
    let mut w = StoredFieldsWriter::new(Mode::BestSpeed, &id_write(), "");
    assert!(matches!(
        w.copy_chunks(&reader, 0, reader.max_doc()),
        Err(Error::CorruptChunkBounds { chunk_docs: 0, .. })
    ));
}

#[test]
fn a_chunk_claiming_more_documents_than_the_requested_range_is_rejected() {
    // Java's second `CorruptIndexException`: `docID > toDocID` after a
    // chunk. The first chunk is made to claim 2048 documents instead of
    // 1024, so copying it would run past the end of the requested run.
    const N: usize = BEST_SPEED_MAX_DOCS_PER_CHUNK * 2;
    let docs = merge_docs("y", N);
    let (mut fdt, fdx, fdm) = write_best_speed(&docs, &id_write(), "");
    let first_chunk = {
        let reader = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
        reader.chunk_for_doc(0).unwrap().start_pointer as usize
    };
    // docBase 0 is one byte; the token follows. 1024 docs -> token 4096
    // (vint 0x80 0x20); 2048 docs -> token 8192 (vint 0x80 0x40).
    let token_at = first_chunk + 1;
    assert_eq!(&fdt[token_at..token_at + 2], &[0x80, 0x20]);
    fdt[token_at + 1] = 0x40;

    let reader = open(&fdt, &fdx, &fdm, &id_write(), "").unwrap();
    let mut w = StoredFieldsWriter::new(Mode::BestSpeed, &id_write(), "");
    assert!(matches!(
        w.copy_chunks(&reader, 0, BEST_SPEED_MAX_DOCS_PER_CHUNK as i32),
        Err(Error::CorruptChunkBounds { .. })
    ));
}

// ---------------------------------------------------------------------
// The arithmetic gate (docs/arithmetic-gate.md): every length, count,
// offset and width this format reads off disk, driven past the range a
// Lucene writer could ever have produced. Each of these must fail cleanly rather
// than by panicking, by reserving memory
// proportional to the *claim*, or by accepting the file outright.
// ---------------------------------------------------------------------

/// Recomputes a codec file's trailing CRC so that only the *semantic*
/// invariant under test can fire, never the checksum.
fn resign(file: &mut [u8]) {
    let at = file.len() - 8;
    let checksum = crc32fast::hash(&file[..at]) as u64;
    file[at..].copy_from_slice(&checksum.to_be_bytes());
}

/// Byte offset of `maxDoc` in a `.fdm` built by
/// `build_single_chunk_index`; everything after it is fixed-width until
/// the three tail vlongs.
fn fdm_max_doc_offset() -> usize {
    4 + 1 + META_CODEC.len() + 4 + ID_LENGTH + 1 + vint_len_test(80 * 1024)
}

/// `maxPointer` sits 120 bytes past `maxDoc`: blockShift (4) +
/// indexNumChunks (4) + docsStart (8) + two 21-byte monotonic block
/// headers (42) + docsEnd (8) + another 42 + startPointersEnd (8).
fn fdm_max_pointer_offset() -> usize {
    fdm_max_doc_offset() + 4 + 4 + 4 + 8 + 42 + 8 + 42 + 8
}

#[test]
fn negative_max_pointer_is_a_decode_error_not_an_overflow() {
    // `maxPointer` is a raw i64 and `open` cross-checks it against the
    // `.fdt` length. `(-1i64) as usize + FOOTER_LENGTH` wraps to 15, so
    // before the fix the check was not merely bypassable, it *panicked*
    // in a debug build before it could reject anything.
    let doc_bytes = field_bytes(0, TYPE_STRING, &string_field_payload("x"));
    let (fdt, fdx, mut fdm) = build_single_chunk_index(&doc_bytes);
    let at = fdm_max_pointer_offset();
    fdm[at..at + 8].copy_from_slice(&(-1i64).to_le_bytes());
    resign(&mut fdm);
    assert!(matches!(
        open(&fdt, &fdx, &fdm, &id(), ""),
        Err(Error::Store(lucene_store::Error::Corrupted(_)))
    ));
}

#[test]
fn num_chunks_at_the_top_of_the_vlong_range_is_a_decode_error_not_an_overflow() {
    // `index_num_chunks != num_chunks + 1` overflowed for a `numChunks`
    // vlong of `i64::MAX`. `numChunks` also feeds
    // `direct_monotonic::floor_index` as a search bound, so it is pinned
    // non-negative here too.
    let doc_bytes = field_bytes(0, TYPE_STRING, &string_field_payload("x"));
    let (fdt, fdx, fdm) = build_single_chunk_index_with_meta_overrides(&doc_bytes, i64::MAX, 0, 0);
    assert!(matches!(
        open(&fdt, &fdx, &fdm, &id(), ""),
        Err(Error::NumChunksMismatch { .. })
    ));
}

#[test]
fn negative_max_doc_is_rejected_by_open() {
    // Java takes `numDocs` from `SegmentInfo`; this port takes it from
    // the `.fdm`, and every chunk-bounds check is stated relative to it.
    // A negative one would make `doc_base + chunk_docs > max_doc` vacuous.
    let doc_bytes = field_bytes(0, TYPE_STRING, &string_field_payload("x"));
    let (fdt, fdx, mut fdm) = build_single_chunk_index(&doc_bytes);
    let at = fdm_max_doc_offset();
    fdm[at..at + 4].copy_from_slice(&(-1i32).to_le_bytes());
    resign(&mut fdm);
    assert!(matches!(
        open(&fdt, &fdx, &fdm, &id(), ""),
        Err(Error::Store(lucene_store::Error::Corrupted(_)))
    ));
}

/// A `.fdt` holding one chunk whose header claims `chunk_docs` documents
/// with all-equal (`bpv == 0`) field-count and length arrays, plus a
/// matching `.fdx`/`.fdm` with `max_doc` set as high as the caller likes.
fn single_chunk_with_claimed_counts(
    chunk_docs: i32,
    num_stored_fields: i32,
    length: i32,
    max_doc: i32,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut fdt = Vec::new();
    fdt.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
    write_string(&mut fdt, DATA_CODEC_BEST_SPEED);
    fdt.extend_from_slice(&(VERSION_CURRENT as u32).to_be_bytes());
    fdt.extend_from_slice(&id());
    fdt.push(0);
    let chunk_start = fdt.len() as i64;
    write_vint(&mut fdt, 0); // docBase
    write_vint(&mut fdt, chunk_docs << 2); // token
    if chunk_docs == 1 {
        // The single-document header shape: two bare vints, no bulk
        // arrays at all (`Lucene90CompressingStoredFieldsWriter.flush`).
        write_vint(&mut fdt, num_stored_fields);
        write_vint(&mut fdt, length);
    } else {
        fdt.push(0); // numStoredFields: bpv=0 (all equal)
        write_vint(&mut fdt, num_stored_fields);
        fdt.push(0); // lengths: bpv=0 (all equal)
        write_vint(&mut fdt, length);
    }
    fdt.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
    fdt.extend_from_slice(&0u32.to_be_bytes());
    let checksum = crc32fast::hash(&fdt) as u64;
    fdt.extend_from_slice(&checksum.to_be_bytes());
    let (fdx, fdm) = build_fdx_fdm_for_single_chunk(&fdt, max_doc, chunk_start);
    (fdt, fdx, fdm)
}

#[test]
fn a_chunk_claiming_more_documents_than_the_format_allows_is_rejected() {
    // `chunkDocs` off a chunk token sizes `read_bulk_ints`' `vec![0i64;
    // count]` and the `offsets` reservation, and the only thing that used
    // to bound it was `maxDoc` -- itself a raw i32 off the `.fdm`. With
    // `maxDoc = i32::MAX` a 60-byte file could ask for a 17 GB `Vec`,
    // which *aborts* rather than unwinding. 2 000 000 is used here so the
    // pre-fix behaviour (a 16 MB reservation, then EOF) is survivable
    // enough to observe; the shape is identical at i32::MAX.
    let (fdt, fdx, fdm) = single_chunk_with_claimed_counts(2_000_000, 1, 1, i32::MAX);
    let reader = open(&fdt, &fdx, &fdm, &id(), "").unwrap();
    assert!(matches!(
        reader.document(0),
        Err(Error::CorruptChunkBounds { .. })
    ));
    // ...and the legal ceiling really is the per-mode `maxDocsPerChunk`:
    // one document under it still parses its header.
    let (fdt, fdx, fdm) = single_chunk_with_claimed_counts(
        BEST_SPEED_MAX_DOCS_PER_CHUNK as i32,
        0,
        0,
        BEST_SPEED_MAX_DOCS_PER_CHUNK as i32,
    );
    let reader = open(&fdt, &fdx, &fdm, &id(), "").unwrap();
    assert!(reader.document(0).unwrap().fields.is_empty());
    let (fdt, fdx, fdm) =
        single_chunk_with_claimed_counts(BEST_SPEED_MAX_DOCS_PER_CHUNK as i32 + 1, 0, 0, i32::MAX);
    let reader = open(&fdt, &fdx, &fdm, &id(), "").unwrap();
    assert!(matches!(
        reader.document(0),
        Err(Error::CorruptChunkBounds { .. })
    ));
}

#[test]
fn a_negative_document_length_is_a_decode_error_not_a_backwards_offset() {
    // Only the `bpv == 0` bulk-int shape can carry a negative value (the
    // 8/16/32-bit shapes are masked), and running it into the cumulative
    // `offsets` array would give a *decreasing* offsets array, so `serialized_document`'s
    // `offsets[i + 1] as usize - doc_offset` then underflowed to ~2^64 --
    // which `Vec::with_capacity` turns into a capacity-overflow panic.
    let (fdt, fdx, fdm) = single_chunk_with_claimed_counts(2, 1, -1, 2);
    let reader = open(&fdt, &fdx, &fdm, &id(), "").unwrap();
    assert!(matches!(
        reader.document(0),
        Err(Error::Store(lucene_store::Error::Corrupted(_)))
    ));
    // The single-document header shape reads its length as a bare vint
    // rather than through a bulk array, and needed the same guard.
    let (fdt, fdx, fdm) = single_chunk_with_claimed_counts(1, 1, -1, 1);
    let reader = open(&fdt, &fdx, &fdm, &id(), "").unwrap();
    assert!(matches!(
        reader.document(0),
        Err(Error::LengthFieldCountMismatch { length: -1, .. })
    ));
}

#[test]
fn a_chunk_claiming_more_decompressed_bytes_than_could_be_produced_is_rejected() {
    // Nothing on the wire relates a chunk's decompressed size to the
    // compressed bytes that follow it, so the header can name any size it
    // likes -- and that size flows into `decompress_range`'s output `Vec`
    // and `decompress_unit`'s `vec![0u8; dictLength + blockLength]`. The
    // LZ4 block format expands by at most 255x, so a claim past 255 times
    // the bytes left in the file cannot be produced by *any* input.
    let (fdt, fdx, fdm) = single_chunk_with_claimed_counts(1, 1, 10_000_000, 1);
    let reader = open(&fdt, &fdx, &fdm, &id(), "").unwrap();
    let err = reader.document(0).unwrap_err();
    let Error::Store(lucene_store::Error::Corrupted(msg)) = &err else {
        panic!("unexpected error: {err:?}");
    };
    assert!(msg.contains("decompressed bytes"), "{msg}");

    // The whole-chunk path is where the claim is worst: `read_chunk`
    // reserves the chunk's *entire* declared length in one go, and with
    // up to 1024 documents each declaring an `i32::MAX` length that is a
    // ~2 TB request off a hundred-byte file. Checked in
    // `read_chunk_header`, so both paths are covered by the one check.
    let (fdt, fdx, fdm) = single_chunk_with_claimed_counts(1024, 1, i32::MAX, 1024);
    let reader = open(&fdt, &fdx, &fdm, &id(), "").unwrap();
    let err = reader.read_chunk(0).unwrap_err();
    let Error::Store(lucene_store::Error::Corrupted(msg)) = &err else {
        panic!("unexpected error: {err:?}");
    };
    assert!(msg.contains("decompressed bytes"), "{msg}");
}

#[test]
fn an_oversized_binary_field_length_is_a_decode_error_not_an_allocation() {
    // Java hands its visitor a lazy `StoredFieldDataInput`; this port owns
    // a `Vec<u8>`, so the vint length has to be bounded by the bytes
    // actually left. A negative one would sign-extend to ~2^64.
    let mut payload = Vec::new();
    write_vint(&mut payload, 1_000_000);
    let field = field_bytes(0, TYPE_BYTE_ARR, &payload);
    let (fdt, fdx, fdm) = build_single_chunk_index(&field);
    let reader = open(&fdt, &fdx, &fdm, &id(), "").unwrap();
    assert!(matches!(
        reader.document(0),
        Err(Error::Store(lucene_store::Error::Corrupted(_)))
    ));

    let mut payload = Vec::new();
    write_vint(&mut payload, -1);
    let field = field_bytes(0, TYPE_BYTE_ARR, &payload);
    let (fdt, fdx, fdm) = build_single_chunk_index(&field);
    let reader = open(&fdt, &fdx, &fdm, &id(), "").unwrap();
    assert!(matches!(
        reader.document(0),
        Err(Error::Store(lucene_store::Error::Corrupted(_)))
    ));
}

#[test]
fn an_absurd_stored_field_count_does_not_reserve_a_slot_per_claimed_field() {
    // `numStoredFields` comes off the chunk header's 32-bit bulk int
    // array, read unsigned, so it reaches ~4.3e9 -- and a `StoredField`
    // is 40-odd bytes. Reserving for the claim is a ~170 GB request: the
    // allocator fails, `handle_alloc_error` **aborts**, and no
    // `catch_unwind` at the FFI boundary can intercept that. Reserving
    // for what the bytes could hold costs nothing and errors identically.
    assert!(parse_document(i64::from(u32::MAX), b"").is_err());
    assert!(parse_document(i64::MAX, b"\x00").is_err());
}

#[test]
fn a_negative_sub_block_compressed_length_is_a_decode_error() {
    // The `length == 0` skip path sums every sub-block's compressed
    // length. A negative vint sign-extended to ~2^64 and overflowed the
    // running total, which in release wraps to a *small* skip -- leaving
    // the reader mid-unit, so the next `sliced` unit's framing header is
    // read out of compressed bytes and decodes plausible garbage.
    let mut unit = Vec::new();
    write_vint(&mut unit, 0); // dictLength
    write_vint(&mut unit, 4); // blockLength
    write_vint(&mut unit, -1); // dictionary's compressed length
    write_vint(&mut unit, 1);
    write_vint(&mut unit, 1);
    let mut input = SliceInput::new(&unit);
    let mut out = Vec::new();
    let err = decompress_unit(Mode::BestSpeed, &mut input, 8, 0, 0, &mut out).unwrap_err();
    assert!(
        matches!(err, Error::Store(lucene_store::Error::Corrupted(_))),
        "unexpected error: {err:?}"
    );
}

/// Flips one bit of one of the three files, re-signs all three, and
/// tries to open the segment and read every document out of it. Returns
/// `true` if the mutation was *rejected* (a typed error from `open`, from
/// a read, or from `check_integrity`) rather than silently accepted. A
/// panic, an abort or a reservation proportional to a number just read
/// off disk is a test failure, which is the point.
fn byte_flip_is_rejected(
    files: &(Vec<u8>, Vec<u8>, Vec<u8>),
    which: usize,
    at: usize,
    bit: u8,
) -> bool {
    let (mut fdt, mut fdx, mut fdm) = files.clone();
    {
        let target = match which {
            0 => &mut fdt,
            1 => &mut fdx,
            _ => &mut fdm,
        };
        target[at] ^= bit;
    }
    resign(&mut fdt);
    resign(&mut fdx);
    resign(&mut fdm);
    let Ok(reader) = open(&fdt, &fdx, &fdm, &id_write(), "") else {
        return true;
    };
    let mut rejected = reader.check_integrity().is_err();
    let mut cursor = ChunkCursor::new();
    for doc in 0..reader.max_doc() {
        // The cursor path decompresses each chunk once; the
        // random-access path re-frames the unit per document, so it is
        // sampled at the chunk edges rather than run for every document.
        match cursor.document(&reader, doc) {
            Ok((n, bytes)) => rejected |= parse_document(n, bytes).is_err(),
            Err(_) => rejected = true,
        }
        if doc % 1021 == 0 {
            rejected |= reader.document(doc).is_err();
        }
    }
    rejected
}

fn flip_sweep(files: &(Vec<u8>, Vec<u8>, Vec<u8>), which: usize) -> (usize, usize) {
    let len = match which {
        0 => files.0.len(),
        1 => files.1.len(),
        _ => files.2.len(),
    };
    let (mut rejected, mut total) = (0usize, 0usize);
    // The trailing 16-byte footer is skipped: it is recomputed anyway.
    for at in 0..len - codec_util::FOOTER_LENGTH {
        for bit in [0x01u8, 0x80] {
            total += 1;
            if byte_flip_is_rejected(files, which, at, bit) {
                rejected += 1;
            }
        }
    }
    (rejected, total)
}

#[test]
fn every_single_bit_flip_in_a_re_signed_index_is_handled_without_panicking() {
    // The `.tvd` sweep, applied to the stored-fields `.fdx`/`.fdm`
    // pair. The footers are recomputed after every flip so only the
    // *semantic* invariants can fire -- a checksum catching everything
    // would prove nothing about the decoders. Three chunks, so that the
    // monotonic doc-base and start-pointer arrays actually discriminate
    // (with one chunk every lookup lands on chunk 0 whatever they say).
    let docs = merge_docs("flip", BEST_SPEED_MAX_DOCS_PER_CHUNK * 2 + 52);
    let files = write_best_speed(&docs, &id_write(), "");
    let (fdm_rejected, fdm_total) = flip_sweep(&files, 2);
    let (fdx_rejected, fdx_total) = flip_sweep(&files, 1);
    // Recorded as floors rather than exact counts: a handful of `.fdm`
    // bits genuinely do not matter (the codec version, which `open`
    // accepts across its supported range; `chunkSize`, which only a
    // `sliced` chunk consults; the unused `offset` of a `bpv == 0`
    // monotonic block; `numDirtyDocs`, which Java only sanity-checks).
    assert!(
        fdm_rejected * 100 >= fdm_total * 90,
        ".fdm bit flips rejected: {fdm_rejected}/{fdm_total}"
    );
    assert!(
        fdx_rejected * 100 >= fdx_total * 80,
        ".fdx bit flips rejected: {fdx_rejected}/{fdx_total}"
    );
}

#[test]
fn every_single_bit_flip_in_a_re_signed_data_file_is_handled_without_panicking() {
    // The same sweep over the `.fdt` itself -- chunk headers, the
    // preset-dictionary unit framing, and the compressed payload. A small
    // segment, because every byte is swept: the interesting bytes are the
    // header and framing ones, and the payload is deliberately included
    // so the LZ4 decoder is driven with garbage too.
    //
    // The bar here is *not* a rejection rate. A flip inside a document's
    // compressed payload legitimately decodes to a different but
    // perfectly well-formed document, and Lucene has no way to notice
    // (that is what the checksum this test deliberately repairs is for).
    // The bar is that nothing panics, aborts, or reserves memory
    // proportional to a length it just read.
    let docs = merge_docs("flip", 40);
    let files = write_best_speed(&docs, &id_write(), "");
    let (rejected, total) = flip_sweep(&files, 0);
    // 624/904 at the time of writing; the floor is deliberately far
    // below that, since the exact number tracks the compressor.
    assert!(
        rejected * 100 >= total * 40,
        ".fdt bit flips rejected: {rejected}/{total}"
    );
}

/// Records every callback, so a lazily read document can be compared
/// with the same visitor run over the fully decompressed bytes.
struct Recorder {
    /// Per field position: `true` takes it, `false` says `NO`.
    take: Vec<bool>,
    stop_at: Option<usize>,
    defer_from: usize,
    seen: usize,
    log: Vec<String>,
}

impl StoredFieldVisitor for Recorder {
    fn needs_field(&mut self, n: i32) -> Result<VisitStatus> {
        let i = self.seen;
        self.seen += 1;
        self.log.push(format!("ask {n}"));
        if self.stop_at == Some(i) {
            return Ok(VisitStatus::Stop);
        }
        Ok(if self.take.get(i).copied().unwrap_or(true) {
            VisitStatus::Yes
        } else {
            VisitStatus::No
        })
    }
    fn string_field(&mut self, n: i32, v: &str) -> Result<()> {
        self.log
            .push(format!("str {n} {} {:?}", v.len(), &v[..v.len().min(8)]));
        Ok(())
    }
    fn binary_field(&mut self, n: i32, v: &[u8]) -> Result<()> {
        let sum = v
            .iter()
            .fold(0u64, |a, &b| a.wrapping_mul(31).wrapping_add(u64::from(b)));
        self.log.push(format!("bin {n} {} {sum}", v.len()));
        Ok(())
    }
    fn int_field(&mut self, n: i32, v: i32) -> Result<()> {
        self.log.push(format!("int {n} {v}"));
        Ok(())
    }
    fn long_field(&mut self, n: i32, v: i64) -> Result<()> {
        self.log.push(format!("long {n} {v}"));
        Ok(())
    }
    fn float_field(&mut self, n: i32, v: f32) -> Result<()> {
        self.log.push(format!("float {n} {v}"));
        Ok(())
    }
    fn double_field(&mut self, n: i32, v: f64) -> Result<()> {
        self.log.push(format!("double {n} {v}"));
        Ok(())
    }
    fn defer_values_from(&self) -> usize {
        self.defer_from
    }
    fn deferred_field(&mut self, n: i32, string: bool, len: usize) -> Result<()> {
        self.log.push(format!("deferred {n} {string} {len}"));
        Ok(())
    }
}

#[test]
fn sliced_documents_read_lazily_match_the_whole_document() {
    // Values that straddle compression units in every way: numbers
    // between big values, a value starting right at a unit boundary,
    // strings with multi-byte characters, and documents in sliced
    // chunks next to small ones.
    let big = |n: usize, seed: u64| -> Vec<u8> {
        let mut x = seed | 1;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                b"abcdefgh {}\"\n0123"[(x % 16) as usize]
            })
            .collect()
    };
    let mut docs = Vec::new();
    for d in 0..6u64 {
        let mut fields = vec![
            StoredField {
                field_number: 0,
                value: FieldValue::String(format!("id-{d}")),
            },
            StoredField {
                field_number: 1,
                value: FieldValue::Binary(big(150_000 + d as usize * 37_001, d)),
            },
            StoredField {
                field_number: 2,
                value: FieldValue::Long(-(d as i64) * 1_000_000_007),
            },
            StoredField {
                field_number: 3,
                value: FieldValue::String("é".repeat(40_000 + d as usize)),
            },
            StoredField {
                field_number: 4,
                value: FieldValue::Double(1.5 + d as f64),
            },
            StoredField {
                field_number: 5,
                value: FieldValue::Float(0.25),
            },
            StoredField {
                field_number: 6,
                value: FieldValue::Int(d as i32 - 3),
            },
            StoredField {
                field_number: 7,
                value: FieldValue::Binary(big(300_001, d + 9)),
            },
        ];
        if d % 2 == 1 {
            fields.reverse();
        }
        docs.push(Document { fields });
        docs.push(Document {
            fields: vec![StoredField {
                field_number: 0,
                value: FieldValue::String(format!("small-{d}")),
            }],
        });
    }
    let seg = [9u8; ID_LENGTH];
    for mode in [
        Mode::BestSpeed,
        Mode::BestCompression,
        Mode::ZstdPaged(3),
        Mode::Zstd(ZstdMode::zstd(3)),
    ] {
        let mut w = StoredFieldsWriter::new(mode, &seg, "");
        for doc in &docs {
            w.add_document(doc);
        }
        let (fdt, fdx, fdm) = w.finish();
        let reader = open(&fdt, &fdx, &fdm, &seg, "").unwrap();
        let num_fields = 8;
        let mut plans: Vec<(Vec<bool>, Option<usize>, usize)> = vec![
            (vec![true; num_fields], None, usize::MAX),
            (vec![false; num_fields], None, usize::MAX),
            (vec![true; num_fields], Some(1), usize::MAX),
            (vec![true; num_fields], None, 100_000),
            (vec![true; num_fields], None, 0),
        ];
        for mask in 0..64u32 {
            let take: Vec<bool> = (0..num_fields)
                .map(|i| (mask >> (i % 6)) & 1 == 1)
                .collect();
            plans.push((
                take,
                Some((mask % 9) as usize),
                60_000 * (mask as usize % 3),
            ));
        }
        for doc_id in 0..docs.len() as i32 {
            let whole = reader.serialized_document(doc_id).unwrap();
            for (take, stop_at, defer_from) in &plans {
                let new = || Recorder {
                    take: take.clone(),
                    stop_at: *stop_at,
                    defer_from: *defer_from,
                    seen: 0,
                    log: Vec::new(),
                };
                let mut lazy = new();
                reader.visit_document(doc_id, &mut lazy).unwrap();
                let mut eager = new();
                visit_document(whole.num_stored_fields, &whole.bytes, &mut eager).unwrap();
                assert_eq!(
                    lazy.log, eager.log,
                    "{mode:?} doc {doc_id} {take:?} {stop_at:?}"
                );
            }
            let mut all = DocumentVisitor::all();
            reader.visit_document(doc_id, &mut all).unwrap();
            assert_eq!(
                all.into_document(),
                docs[doc_id as usize],
                "{mode:?} doc {doc_id}"
            );
        }
    }
}

#[test]
fn a_lazily_read_document_that_claims_more_bytes_than_it_has_is_an_error() {
    // A big value whose length vint is corrupted upward: the lazy reader
    // must report it, not read past the document.
    let seg = [4u8; ID_LENGTH];
    let doc = Document {
        fields: vec![
            StoredField {
                field_number: 0,
                value: FieldValue::Binary(vec![7u8; 200_000]),
            },
            StoredField {
                field_number: 1,
                value: FieldValue::Int(1),
            },
        ],
    };
    let mut w = StoredFieldsWriter::new(Mode::ZstdPaged(3), &seg, "");
    w.add_document(&doc);
    let (fdt, fdx, fdm) = w.finish();
    let reader = open(&fdt, &fdx, &fdm, &seg, "").unwrap();
    // The serialized bytes, re-parsed with a visitor that says NO to the
    // big value and YES to the int, work...
    let mut ok = Recorder {
        take: vec![false, true],
        stop_at: None,
        defer_from: usize::MAX,
        seen: 0,
        log: Vec::new(),
    };
    reader.visit_document(0, &mut ok).unwrap();
    assert_eq!(ok.log, vec!["ask 0", "ask 1", "int 1 1"]);
    // ...and a document whose value length is past its end is refused
    // by the length check before any skip.
    let mut bytes = Vec::new();
    bytes.push(1u8); // field 0, TYPE_BYTE_ARR
    bytes.extend_from_slice(&[0xFF, 0xFF, 0x0F]); // length 262143
    bytes.extend_from_slice(&[0u8; 10]);
    let mut v = DocumentVisitor::all();
    assert!(visit_document(1, &bytes, &mut v).is_err());
}

/// A document with `text_len` text bytes among small fields, and a
/// pseudo-random (so barely compressible) body.
fn outsized_doc(text_len: usize, seed: u32) -> Document {
    let mut x = seed.max(1);
    let text: String = (0..text_len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            char::from(b'a' + (x % 26) as u8)
        })
        .collect();
    Document {
        fields: vec![
            StoredField {
                field_number: 0,
                value: FieldValue::Int(7),
            },
            StoredField {
                field_number: 1,
                value: FieldValue::String(text),
            },
            StoredField {
                field_number: 2,
                value: FieldValue::Binary(vec![9; 70]),
            },
            StoredField {
                field_number: 3,
                value: FieldValue::Double(2.5),
            },
        ],
    }
}

/// What the buffered path writes for `docs`: each document serialized
/// whole and handed over as bytes, so none takes the streamed path.
fn buffered_bytes(mode: Mode, docs: &[Document]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut w = StoredFieldsWriter::new(mode, &[3; ID_LENGTH], "");
    for doc in docs {
        let mut bytes = Vec::new();
        serialize_doc_into(doc, &mut bytes);
        w.add_serialized_document(doc.fields.len() as i64, &bytes);
    }
    w.finish()
}

#[test]
fn an_outsized_document_streams_the_bytes_the_buffered_path_writes() {
    for mode in [
        Mode::BestSpeed,
        Mode::BestCompression,
        Mode::Zstd(ZstdMode::zstd(3)),
    ] {
        let chunk = mode.chunk_size() as usize;
        let small = |n: usize| outsized_doc(n, 5 + n as u32);
        // A document whose whole serialization is `total` bytes.
        let exactly = |total: usize| {
            let overhead = serialized_doc_len(&outsized_doc(0, 1));
            outsized_doc(total - overhead, 9)
        };
        let max_docs = mode.max_docs_per_chunk() as usize;
        let tiny = || Document {
            fields: vec![StoredField {
                field_number: 0,
                value: FieldValue::Int(1),
            }],
        };
        // Alone, behind buffered documents, between two outsized ones,
        // on the exact multiple of a slice, and one byte over it.
        // The flag says every outsized document is past twice a slice,
        // so none may grow the chunk buffer.
        let cases: Vec<(bool, Vec<Document>)> = vec![
            (true, vec![small(2 * chunk + 100)]),
            (
                true,
                vec![small(40), small(50), small(2 * chunk + 100), small(60)],
            ),
            (
                true,
                vec![small(3 * chunk), small(2 * chunk + 1), small(10)],
            ),
            (false, vec![small(chunk + chunk / 2), small(2 * chunk + 7)]),
            (
                false,
                vec![
                    small(10),
                    small(2 * chunk),
                    small(chunk - 1),
                    small(chunk + 3),
                ],
            ),
            (true, vec![small(7 * chunk + 123)]),
            // The payload lands on a whole number of slices.
            (true, vec![exactly(2 * chunk)]),
            (true, vec![exactly(3 * chunk), exactly(2 * chunk + 1)]),
            (true, vec![exactly(4 * chunk - 1), small(20)]),
            // One document short of the document-count trigger, and then
            // the outsized document that closes the chunk on its size.
            (
                true,
                std::iter::repeat_with(tiny)
                    .take(max_docs - 1)
                    .chain([small(2 * chunk + 5)])
                    .collect(),
            ),
        ];
        for (all_streamed, docs) in cases {
            let mut w = StoredFieldsWriter::new(mode, &[3; ID_LENGTH], "");
            for doc in &docs {
                w.add_document(doc);
            }
            let held = w.chunk_buf.capacity();
            assert!(
                !all_streamed || held < 2 * chunk,
                "the chunk buffer grew to {held} bytes for {mode:?}: not streamed"
            );
            assert_eq!(
                w.finish(),
                buffered_bytes(mode, &docs),
                "{mode:?}, documents {:?}",
                docs.iter().map(serialized_doc_len).collect::<Vec<_>>()
            );
        }
    }
}

/// `doc` with every string value held as a [`FieldValue::Shared`].
fn shared_copy(doc: &Document) -> Document {
    Document {
        fields: doc
            .fields
            .iter()
            .map(|f| StoredField {
                field_number: f.field_number,
                value: match &f.value {
                    FieldValue::String(s) => FieldValue::Shared(std::sync::Arc::from(&**s)),
                    other => other.clone(),
                },
            })
            .collect(),
    }
}

#[test]
fn a_shared_value_writes_the_bytes_its_string_writes() {
    for mode in [
        Mode::BestSpeed,
        Mode::BestCompression,
        Mode::Zstd(ZstdMode::zstd(3)),
    ] {
        let chunk = mode.chunk_size() as usize;
        // Buffered ones, an outsized one that streams, and one on each
        // side of the size that streams.
        let docs = [
            outsized_doc(30, 1),
            outsized_doc(2 * chunk + 100, 2),
            outsized_doc(chunk, 3),
            outsized_doc(5 * chunk + 9, 4),
            outsized_doc(12, 5),
        ];
        let mut shared = StoredFieldsWriter::new(mode, &[3; ID_LENGTH], "");
        let mut owned = StoredFieldsWriter::new(mode, &[3; ID_LENGTH], "");
        for doc in &docs {
            let copy = shared_copy(doc);
            assert_ne!(&copy, doc, "the copy holds shared values");
            shared.add_document(&copy);
            owned.add_document(doc);
        }
        let mut alone = StoredFieldsWriter::new(mode, &[3; ID_LENGTH], "");
        alone.add_document(&shared_copy(&outsized_doc(5 * chunk + 9, 4)));
        assert!(
            alone.chunk_buf.capacity() < 2 * chunk,
            "a shared outsized value streams too: {} bytes held",
            alone.chunk_buf.capacity()
        );
        assert_eq!(shared.finish(), owned.finish(), "{mode:?}");
    }
}

#[test]
fn a_shared_value_reads_back_as_the_string_it_wrote() {
    let doc = outsized_doc(70_000, 8);
    let mut w = StoredFieldsWriter::new(Mode::BestSpeed, &[3; ID_LENGTH], "");
    w.add_document(&shared_copy(&doc));
    let (fdt, fdx, fdm) = w.finish();
    let reader = open(&fdt, &fdx, &fdm, &[3; ID_LENGTH], "").expect("the segment opens");
    assert_eq!(reader.document(0).unwrap(), doc);
    assert_eq!(FieldValue::Shared("é".into()).as_text(), Some("é"));
    assert_eq!(FieldValue::Int(1).as_text(), None);
}

#[test]
fn a_streamed_document_reads_back() {
    let mode = Mode::BestSpeed;
    let chunk = mode.chunk_size() as usize;
    let docs = [outsized_doc(30, 1), outsized_doc(5 * chunk + 9, 2)];
    let mut w = StoredFieldsWriter::new(mode, &[3; ID_LENGTH], "");
    for doc in &docs {
        w.add_document(doc);
    }
    let (fdt, fdx, fdm) = w.finish();
    let reader = open(&fdt, &fdx, &fdm, &[3; ID_LENGTH], "").expect("the segment opens");
    assert_eq!(reader.document(0).unwrap(), docs[0]);
    assert_eq!(reader.document(1).unwrap(), docs[1]);
}

#[test]
fn a_streamed_document_spills_the_fdt_as_it_goes() {
    let root = lucene_util::test_support::TempDir::new("stored-stream");
    use lucene_store::Directory;
    let dir = lucene_store::FsDirectory::open(root.path());
    let mode = Mode::BestSpeed;
    let docs = [outsized_doc(50, 4), outsized_doc(3_000_000, 3)];
    let mut w = StoredFieldsWriter::new(mode, &[3; ID_LENGTH], "");
    w.stream_to(dir.create_output("x.fdt").unwrap());
    for doc in &docs {
        w.add_document(doc);
    }
    // Compressed output is held a slice's worth at a time, not whole:
    // the data barely compresses, so ~3 MB would be held without spills.
    let held = w.fdt_high_water;
    assert!(
        held < 2 * STREAM_BYTES + 2 * mode.chunk_size() as usize,
        "the .fdt held {held} bytes at once"
    );
    let (out, fdx, fdm) = w.finish_streamed();
    out.expect("streamed").close().unwrap();
    let (fdt_ref, fdx_ref, fdm_ref) = buffered_bytes(mode, &docs);
    assert!(dir.open("x.fdt").unwrap().as_ref() == fdt_ref.as_slice());
    assert_eq!((fdx, fdm), (fdx_ref, fdm_ref));
}
