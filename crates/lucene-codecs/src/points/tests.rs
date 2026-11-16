#![cfg(test)]
// The arithmetic gate is about values read off disk; a test's `i + 1` is
// not one. See docs/arithmetic-gate.md.
#![allow(clippy::arithmetic_side_effects)]

use super::*;

#[test]
fn copy_dim_and_refill_copy_every_width() {
    for len in [1usize, 2, 3, 4, 5, 6, 7, 8, 16] {
        let src: Vec<u8> = (1..=len as u8).collect();
        let mut dst = vec![0u8; len];
        copy_dim(&mut dst, &src);
        assert_eq!(dst, src, "copy_dim, {len} bytes");
        // In place once the length matches; refilled when it does not, or
        // when the measurement arm asks for a fresh allocation.
        for (start, reuse) in [
            (vec![9u8; len], true),
            (vec![9u8; len + 1], true),
            (vec![9u8; len], false),
        ] {
            let mut buf = start;
            refill(&mut buf, &src, reuse);
            assert_eq!(buf, src, "refill, {len} bytes, reuse {reuse}");
        }
    }
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

#[test]
fn continuous_ids_decode() {
    let mut bytes = vec![CONTINUOUS_IDS as u8];
    write_vint(&mut bytes, 100);
    let mut input = SliceInput::new(&bytes);
    assert_eq!(
        read_doc_ids(&mut input, 5).unwrap(),
        vec![100, 101, 102, 103, 104]
    );
}

#[test]
fn bitset_ids_decode() {
    // docs 2, 5, 130 (offsetWords=0, spans 3 64-bit words).
    let mut bytes = vec![BITSET_IDS as u8];
    write_vint(&mut bytes, 0); // offsetWords
    write_vint(&mut bytes, 3); // longLen
    let mut word0 = 0u64;
    word0 |= 1 << 2;
    word0 |= 1 << 5;
    bytes.extend_from_slice(&word0.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    let mut word2 = 0u64;
    word2 |= 1 << (130 - 128);
    bytes.extend_from_slice(&word2.to_le_bytes());
    let mut input = SliceInput::new(&bytes);
    assert_eq!(read_doc_ids(&mut input, 3).unwrap(), vec![2, 5, 130]);
}

#[test]
fn delta_bpv16_even_count() {
    let min = 1000i32;
    let deltas = [0i32, 5, 2, 9]; // docIds = min+delta
    let mut bytes = vec![DELTA_BPV_16 as u8];
    write_vint(&mut bytes, min);
    let half = deltas.len() / 2;
    for i in 0..half {
        let word = ((deltas[i] as u32) << 16) | (deltas[half + i] as u32);
        bytes.extend_from_slice(&(word as i32).to_le_bytes());
    }
    // out[i] = min+deltas[i], out[i+half] = min+deltas[half+i] -- the
    // pairing is (index, index+half) sharing one packed word, not
    // consecutive indices.
    let mut input = SliceInput::new(&bytes);
    assert_eq!(
        read_doc_ids(&mut input, 4).unwrap(),
        vec![1000, 1005, 1002, 1009]
    );
}

#[test]
fn delta_bpv16_odd_count() {
    let min = 10i32;
    let mut bytes = vec![DELTA_BPV_16 as u8];
    write_vint(&mut bytes, min);
    // count=1: half=0, no packed words, then one trailing u16.
    bytes.extend_from_slice(&7u16.to_le_bytes());
    let mut input = SliceInput::new(&bytes);
    assert_eq!(read_doc_ids(&mut input, 1).unwrap(), vec![17]);
}

#[test]
fn bpv32_decode() {
    let mut bytes = vec![BPV_32 as u8];
    for v in [1i32, 1_000_000, 70_000] {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let mut input = SliceInput::new(&bytes);
    assert_eq!(
        read_doc_ids(&mut input, 3).unwrap(),
        vec![1, 1_000_000, 70_000]
    );
}

#[test]
fn unsupported_doc_ids_encoding_rejected() {
    let bytes = [1u8]; // no such marker byte is defined
    let mut input = SliceInput::new(&bytes);
    assert!(matches!(
        read_doc_ids(&mut input, 1),
        Err(Error::UnsupportedDocIdsEncoding(1))
    ));
}

#[test]
fn legacy_delta_vint_decode() {
    // marker 0 (LEGACY_DELTA_VINT): each id is a vint delta from the
    // previous one, starting at 0. Real Lucene 10.5.0 writers never
    // produce this marker (it predates DELTA_BPV_16/BPV_21/BPV_24/
    // BPV_32), so this is a hand-built fixture, not a real-Lucene one.
    let mut bytes = vec![0u8]; // LEGACY_DELTA_VINT
    write_vint(&mut bytes, 5); // doc 0: 0 + 5 = 5
    write_vint(&mut bytes, 3); // doc 1: 5 + 3 = 8
    write_vint(&mut bytes, 100); // doc 2: 8 + 100 = 108
    let mut input = SliceInput::new(&bytes);
    assert_eq!(read_doc_ids(&mut input, 3).unwrap(), vec![5, 8, 108]);
}

#[test]
fn legacy_delta_vint_empty() {
    let bytes = vec![0u8];
    let mut input = SliceInput::new(&bytes);
    assert_eq!(read_doc_ids(&mut input, 0).unwrap(), Vec::<i32>::new());
}

/// A single-leaf field (numLeaves=1): the packed index is just the root
/// FP delta vlong, no split descriptor bytes at all.
#[test]
fn single_leaf_decode_leaf_pointers() {
    let field = PointsField {
        num_dims: 1,
        num_index_dims: 1,
        bytes_per_dim: 4,
        max_points_in_leaf_node: 512,
        num_leaves: 1,
        min_packed_value: vec![0; 4],
        max_packed_value: vec![0; 4],
        point_count: 3,
        doc_count: 3,
        index_start_pointer: 0,
        num_index_bytes: 0,
        bkd_version: BKD_VERSION_CURRENT,
    };
    let mut inner = Vec::new();
    write_vlong(&mut inner, 300_000); // large enough to need vlong continuation bytes
    assert_eq!(decode_leaf_pointers(&inner, &field).unwrap(), vec![300_000]);
}

/// A 3-leaf field (root splits into leaf 2 (left) and an inner node 3
/// that splits into leaves 6/7): exercises the recursive descent,
/// inherited-vs-delta FP baselines, and the `leftNumBytes` skip.
#[test]
fn three_leaf_decode_leaf_pointers() {
    let field = PointsField {
        num_dims: 1,
        num_index_dims: 1,
        bytes_per_dim: 4,
        max_points_in_leaf_node: 512,
        num_leaves: 3,
        min_packed_value: vec![0; 4],
        max_packed_value: vec![0; 4],
        point_count: 3,
        doc_count: 3,
        index_start_pointer: 0,
        num_index_bytes: 0,
        bkd_version: BKD_VERSION_CURRENT,
    };
    // leafNodeOffset=3. node1 (root) is not a leaf (1<3); its children
    // are node2 (leaf, 2>=3? no wait 2<3 so node2 is NOT a leaf either;
    // recompute: leafNodeOffset=3 means leaves are nodeId>=3. node1's
    // children are 2,3. node2<3 -> inner; node3>=3 -> leaf.
    // node2's children are 4,5, both >=3 -> leaves.
    let mut inner = Vec::new();
    write_vlong(&mut inner, 100); // root FP baseline (node1)

    // node1 split descriptor: splitDim=0 (numIndexDims=1 so code%1=0
    // always), prefix=4 (== bytesPerDim so suffix=0, no extra bytes).
    // code = splitDim + numIndexDims*(prefix + (1+bytesPerDim)*firstDiffByteDelta)
    // with numIndexDims=1: code = 0 + 1*(prefix + 5*0) = prefix = 4.
    write_vint(&mut inner, 4);
    // left child (node2) < leafNodeOffset(3) -> leftNumBytes follows.
    // node2's own subtree (below) is 5 bytes; set leftNumBytes=5
    // (unused by this port, but must still be present/consumed).
    write_vint(&mut inner, 5);

    // -- node2's subtree (left of root) --
    // node2 split descriptor (same shape as node1's).
    write_vint(&mut inner, 4);
    // node2's left child (node4) >= leafNodeOffset(3) -> no leftNumBytes.
    // node4 (leaf) inherits node2's FP baseline (100, unchanged).
    // node5 (leaf, right child of node2): FP delta.
    write_vlong(&mut inner, 7); // node5 FP = 100+7=107

    // -- node3 (right child of root, a leaf): FP delta from root's 100.
    write_vlong(&mut inner, 50); // node3 FP = 100+50=150

    assert_eq!(
        decode_leaf_pointers(&inner, &field).unwrap(),
        vec![100, 107, 150]
    );
}

fn field_1d(bytes_per_dim: i32) -> PointsField {
    PointsField {
        num_dims: 1,
        num_index_dims: 1,
        bytes_per_dim,
        max_points_in_leaf_node: 512,
        num_leaves: 1,
        min_packed_value: vec![0; bytes_per_dim as usize],
        max_packed_value: vec![0; bytes_per_dim as usize],
        point_count: 0,
        doc_count: 0,
        index_start_pointer: 0,
        num_index_bytes: 0,
        bkd_version: BKD_VERSION_CURRENT,
    }
}

#[test]
fn leaf_unique_value_all_points_share_one_value() {
    let field = field_1d(2);
    let mut bytes = Vec::new();
    write_vint(&mut bytes, 3); // count
    bytes.push(CONTINUOUS_IDS as u8);
    write_vint(&mut bytes, 5); // docBase -> docs 5,6,7
    write_vint(&mut bytes, 2); // common prefix = full bytesPerDim
    bytes.extend_from_slice(&[0x12, 0x34]);
    bytes.push(0xFF); // compressedDim = -1 (unique)

    let mut input = SliceInput::new(&bytes);
    let mut out = Vec::new();
    read_leaf_block(&mut input, &field, &mut out).unwrap();
    assert_eq!(out.len(), 3);
    for (i, p) in out.iter().enumerate() {
        assert_eq!(p.doc_id, 5 + i as i32);
        assert_eq!(p.packed_value, vec![0x12, 0x34]);
    }
}

#[test]
fn leaf_sparse_low_cardinality_two_runs() {
    let field = field_1d(1);
    let mut bytes = Vec::new();
    write_vint(&mut bytes, 4); // count
    bytes.push(CONTINUOUS_IDS as u8);
    write_vint(&mut bytes, 10); // docBase -> docs 10,11,12,13
    write_vint(&mut bytes, 0); // common prefix = 0
    bytes.push(0xFE); // compressedDim = -2 (sparse)
                      // run 1: length=2, value=0xAA
    write_vint(&mut bytes, 2);
    bytes.push(0xAA);
    // run 2: length=2, value=0xBB
    write_vint(&mut bytes, 2);
    bytes.push(0xBB);

    let mut input = SliceInput::new(&bytes);
    let mut out = Vec::new();
    read_leaf_block(&mut input, &field, &mut out).unwrap();
    assert_eq!(
        out.iter()
            .map(|p| (p.doc_id, p.packed_value[0]))
            .collect::<Vec<_>>(),
        vec![(10, 0xAA), (11, 0xAA), (12, 0xBB), (13, 0xBB)]
    );
}

#[test]
fn leaf_run_length_compressed_dim() {
    let field = field_1d(2);
    let mut bytes = Vec::new();
    write_vint(&mut bytes, 3); // count
    bytes.push(CONTINUOUS_IDS as u8);
    write_vint(&mut bytes, 0); // docBase -> docs 0,1,2
    write_vint(&mut bytes, 0); // common prefix = 0
    bytes.push(0x00); // compressedDim = 0
    bytes.push(0x01); // run byte value (shared high byte)
    bytes.push(3); // runLen = 3 (all points in one run)
    bytes.push(0x11); // point0 low byte
    bytes.push(0x22); // point1 low byte
    bytes.push(0x33); // point2 low byte

    let mut input = SliceInput::new(&bytes);
    let mut out = Vec::new();
    read_leaf_block(&mut input, &field, &mut out).unwrap();
    assert_eq!(
        out.iter()
            .map(|p| p.packed_value.clone())
            .collect::<Vec<_>>(),
        vec![vec![0x01, 0x11], vec![0x01, 0x22], vec![0x01, 0x33]]
    );
}

#[test]
fn leaf_multi_index_dim_skips_min_max_box() {
    let field = PointsField {
        num_dims: 2,
        num_index_dims: 2,
        bytes_per_dim: 1,
        max_points_in_leaf_node: 512,
        num_leaves: 1,
        min_packed_value: vec![0; 2],
        max_packed_value: vec![0; 2],
        point_count: 0,
        doc_count: 0,
        index_start_pointer: 0,
        num_index_bytes: 0,
        bkd_version: BKD_VERSION_CURRENT,
    };
    let mut bytes = Vec::new();
    write_vint(&mut bytes, 1); // count
    bytes.push(CONTINUOUS_IDS as u8);
    write_vint(&mut bytes, 0); // docBase
    write_vint(&mut bytes, 0); // prefix dim0 = 0
    write_vint(&mut bytes, 0); // prefix dim1 = 0
    bytes.push(0x00); // compressedDim = 0 -- comes *before* the box
                      // (real `BKDReader.visitDocValuesWithCardinality`'s current-version
                      // layout, see `read_leaf_block`'s doc comment).
                      // min/max box (2 dims x (min tail + max tail), 1 byte each) -- values
                      // are irrelevant, just skipped -- only present because
                      // compressedDim != -1.
    bytes.extend_from_slice(&[0xEE, 0xEE, 0xEE, 0xEE]);
    bytes.push(0xAA); // run byte -> dim0's only byte
    bytes.push(1); // runLen = 1
    bytes.push(0xBB); // dim1's suffix byte for the one point

    let mut input = SliceInput::new(&bytes);
    let mut out = Vec::new();
    read_leaf_block(&mut input, &field, &mut out).unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].packed_value, vec![0xAA, 0xBB]);
}

#[test]
fn leaf_unsupported_compressed_dim_rejected() {
    let field = field_1d(1);
    let mut bytes = Vec::new();
    write_vint(&mut bytes, 1); // count
    bytes.push(CONTINUOUS_IDS as u8);
    write_vint(&mut bytes, 0);
    write_vint(&mut bytes, 0); // prefix = 0
    bytes.push(5); // compressedDim=5, but numDims=1 -> invalid

    let mut input = SliceInput::new(&bytes);
    let mut out = Vec::new();
    assert!(matches!(
        read_leaf_block(&mut input, &field, &mut out),
        Err(Error::UnsupportedCompressedDim(5))
    ));
}

#[test]
fn leaf_sparse_sub_block_count_mismatch_rejected() {
    let field = field_1d(1);
    let mut bytes = Vec::new();
    write_vint(&mut bytes, 3); // count
    bytes.push(CONTINUOUS_IDS as u8);
    write_vint(&mut bytes, 0);
    write_vint(&mut bytes, 0); // prefix = 0
    bytes.push(0xFE); // compressedDim = -2
    write_vint(&mut bytes, 5); // run length overshoots count(3)
    bytes.push(0xAA);

    let mut input = SliceInput::new(&bytes);
    let mut out = Vec::new();
    assert!(matches!(
        read_leaf_block(&mut input, &field, &mut out),
        Err(Error::SubBlockCountMismatch {
            expected: 3,
            actual: 5
        })
    ));
}

#[test]
fn leaf_compressed_dim_sub_block_count_mismatch_rejected() {
    let field = field_1d(1);
    let mut bytes = Vec::new();
    write_vint(&mut bytes, 3); // count
    bytes.push(CONTINUOUS_IDS as u8);
    write_vint(&mut bytes, 0);
    write_vint(&mut bytes, 0); // prefix = 0
    bytes.push(0x00); // compressedDim = 0
    bytes.push(0xAA); // run byte
    bytes.push(5); // runLen overshoots count(3) -- caught before reading further

    let mut input = SliceInput::new(&bytes);
    let mut out = Vec::new();
    assert!(matches!(
        read_leaf_block(&mut input, &field, &mut out),
        Err(Error::SubBlockCountMismatch {
            expected: 3,
            actual: 5
        })
    ));
}

#[test]
fn bpv21_round_trips() {
    // count=100 makes one_third=32 (nonzero -- exercises the vectorized
    // main loop) with a 4-value remainder split across the triple-pack
    // loop (3 values) and the final scalar tail (1 value).
    let count = 100usize;
    let ids: Vec<i32> = (0..count as i32).map(|i| 1000 + i * 37).collect();
    let mut bytes = vec![BPV_21 as u8];
    write_bpv21_test(&mut bytes, &ids);
    let mut input = SliceInput::new(&bytes);
    assert_eq!(read_doc_ids(&mut input, count).unwrap(), ids);
}

/// BKD version 9's `BPV_24` (`DocIdsWriter.writeScalarInts24`'s layout):
/// eight ids in three little-endian longs, most significant bits first,
/// then short+byte per remaining id. The same bytes under version 10
/// decode to different ids, so the version must be what picks the
/// decoder.
#[test]
fn scalar_bpv24_is_bkd_version_9s_layout() {
    let ids: Vec<i32> = (0..11).map(|i| 0x0A_0000 + i * 0x1_0203).collect();
    let mut bytes = vec![BPV_24 as u8];
    let v = |i: usize| ids[i] as u64;
    let l1 = (v(0) << 40) | (v(1) << 16) | (v(2) >> 8);
    let l2 = ((v(2) & 0xff) << 56) | (v(3) << 32) | (v(4) << 8) | (v(5) >> 16);
    let l3 = ((v(5) & 0xffff) << 48) | (v(6) << 24) | v(7);
    for l in [l1, l2, l3] {
        bytes.extend_from_slice(&l.to_le_bytes());
    }
    for &id in &ids[8..] {
        bytes.extend_from_slice(&((id >> 8) as u16).to_le_bytes());
        bytes.push(id as u8);
    }
    let mut out = Vec::new();
    read_doc_ids_into(&mut SliceInput::new(&bytes), ids.len(), &mut out, 9).unwrap();
    assert_eq!(out, ids);
    let mut vectorized = Vec::new();
    read_doc_ids_into(
        &mut SliceInput::new(&bytes),
        ids.len(),
        &mut vectorized,
        BKD_VERSION_CURRENT,
    )
    .unwrap();
    assert_ne!(vectorized, ids);
    // Too few bytes for the ids it claims.
    let mut out = Vec::new();
    assert!(read_doc_ids_into(&mut SliceInput::new(&bytes[..20]), ids.len(), &mut out, 9).is_err());
}

#[test]
fn bpv24_round_trips() {
    // count=42 makes quarter=10 (nonzero -- exercises the vectorized
    // main loop) with a 2-value remainder for the final scalar tail.
    let count = 42usize;
    let ids: Vec<i32> = (0..count as i32).map(|i| 100_000 + i * 41).collect();
    let mut bytes = vec![BPV_24 as u8];
    write_bpv24_test(&mut bytes, &ids);
    let mut input = SliceInput::new(&bytes);
    assert_eq!(read_doc_ids(&mut input, count).unwrap(), ids);
}

#[test]
fn bitset_ids_count_mismatch_rejected() {
    let mut bytes = vec![BITSET_IDS as u8];
    write_vint(&mut bytes, 0); // offsetWords
    write_vint(&mut bytes, 1); // longLen
    let word = (1u64 << 2) | (1u64 << 5); // 2 bits set
    bytes.extend_from_slice(&word.to_le_bytes());
    let mut input = SliceInput::new(&bytes);
    // Claim 3 expected, but only 2 bits are set.
    assert!(matches!(
        read_doc_ids(&mut input, 3),
        Err(Error::SubBlockCountMismatch {
            expected: 3,
            actual: 2
        })
    ));
}

/// Mirrors `DocIdsWriter.writeDocIds`'s `BPV_21` branch exactly enough to
/// produce bytes `read_bpv21` can decode -- for test purposes only.
fn write_bpv21_test(out: &mut Vec<u8>, ids: &[i32]) {
    let count = ids.len();
    let one_third = floor_to_multiple_of_16(count / 3);
    let num_ints = one_third * 2;
    let mut scratch = vec![0i32; num_ints];
    for i in 0..num_ints {
        scratch[i] = ids[i] << 11;
    }
    for i in 0..one_third {
        let long_idx = i + num_ints;
        scratch[i] |= ids[long_idx] & 0x7FF;
        scratch[i + one_third] |= (ids[long_idx] >> 11) & 0x7FF;
    }
    for &v in &scratch {
        out.extend_from_slice(&v.to_le_bytes());
    }
    let mut i = one_third * 3;
    while i + 2 < count {
        let l = (ids[i] as i64) | ((ids[i + 1] as i64) << 21) | ((ids[i + 2] as i64) << 42);
        out.extend_from_slice(&l.to_le_bytes());
        i += 3;
    }
    while i < count {
        out.extend_from_slice(&(ids[i] as u16).to_le_bytes());
        out.push((ids[i] >> 16) as u8);
        i += 1;
    }
}

/// Mirrors `DocIdsWriter.writeDocIds`'s vectorized `BPV_24` branch.
fn write_bpv24_test(out: &mut Vec<u8>, ids: &[i32]) {
    let count = ids.len();
    let quarter = count / 4;
    let num_ints = quarter * 3;
    let mut scratch = vec![0i32; num_ints];
    for i in 0..num_ints {
        scratch[i] = ids[i] << 8;
    }
    for i in 0..quarter {
        let long_idx = i + num_ints;
        scratch[i] |= ids[long_idx] & 0xFF;
        scratch[i + quarter] |= (ids[long_idx] >> 8) & 0xFF;
        scratch[i + quarter * 2] |= (ids[long_idx] >> 16) & 0xFF;
    }
    for &v in &scratch {
        out.extend_from_slice(&v.to_le_bytes());
    }
    let mut i = quarter * 4;
    while i < count {
        out.extend_from_slice(&(ids[i] as u16).to_le_bytes());
        out.push((ids[i] >> 16) as u8);
        i += 1;
    }
}

fn id() -> [u8; codec_util::ID_LENGTH] {
    [7u8; codec_util::ID_LENGTH]
}

fn write_vint_i32(out: &mut Vec<u8>, v: i32) {
    write_vint(out, v);
}

/// Hand-encodes a minimal valid `.kdm`/`.kdi`/`.kdd` triple with zero
/// fields (meta stream's field loop terminates on the first `-1`).
fn build_empty_points_index() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    fn write_string(out: &mut Vec<u8>, s: &str) {
        write_vint_i32(out, s.len() as i32);
        out.extend_from_slice(s.as_bytes());
    }
    fn index_header(codec: &str, version: i32) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
        write_string(&mut out, codec);
        out.extend_from_slice(&(version as u32).to_be_bytes());
        out.extend_from_slice(&id());
        out.push(0); // empty suffix
        out
    }
    fn footer(buf: &mut Vec<u8>) {
        buf.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
        buf.extend_from_slice(&0u32.to_be_bytes());
        let checksum = crc32fast::hash(buf) as u64;
        buf.extend_from_slice(&checksum.to_be_bytes());
    }

    let mut kdi = index_header(INDEX_CODEC_NAME, VERSION_CURRENT);
    footer(&mut kdi);
    let mut kdd = index_header(DATA_CODEC_NAME, VERSION_CURRENT);
    footer(&mut kdd);
    let mut kdm = index_header(META_CODEC_NAME, VERSION_CURRENT);
    // Field numbers are a plain 4-byte little-endian i32 (`readInt`),
    // not a vint.
    kdm.extend_from_slice(&(-1i32).to_le_bytes()); // field loop terminator, zero fields
    kdm.extend_from_slice(&0i64.to_le_bytes()); // indexLength
    kdm.extend_from_slice(&0i64.to_le_bytes()); // dataLength
    footer(&mut kdm);

    (kdm, kdi, kdd)
}

#[test]
fn merged_empty_fields_and_empty_operations_emit_valid_empty_triples() {
    let dir = lucene_store::ByteBuffersDirectory::new();
    for native_merge in [false, true] {
        for fields in [
            Vec::new(),
            vec![MergePointsField {
                field_number: 3,
                field_name: String::new(),
                num_dims: 1,
                num_index_dims: 1,
                bytes_per_dim: 4,
                sources: vec![PointsMergeSource::default()],
                max_point_count: 2,
            }],
        ] {
            let (meta, index, data) =
                write_merged_with_mode(fields, 4, 1, &dir, "_empty", &id(), "", native_merge)
                    .unwrap();
            let reader = open(&meta, &index, &data, &id(), "").unwrap();
            assert_eq!(reader.field_numbers().count(), 0);
            assert!(meta.len() > codec_util::FOOTER_LENGTH);
            assert!(index.len() > codec_util::FOOTER_LENGTH);
            assert!(data.len() > codec_util::FOOTER_LENGTH);
        }
    }
}

#[test]
fn generic_one_dimensional_merge_sorts_unsorted_source_values() {
    let dir = lucene_store::ByteBuffersDirectory::new();
    let field = MergePointsField {
        field_number: 3,
        field_name: String::new(),
        num_dims: 1,
        num_index_dims: 1,
        bytes_per_dim: 4,
        sources: vec![PointsMergeSource {
            docs: vec![0, 1, 2, 2],
            values: [9u32, 1, 5, 5]
                .into_iter()
                .flat_map(u32::to_be_bytes)
                .collect(),
        }],
        max_point_count: 9,
    };
    let (meta, index, data) =
        write_merged_with_mode(vec![field], 2, 3, &dir, "_generic", &id(), "", false).unwrap();
    let reader = open(&meta, &index, &data, &id(), "").unwrap();
    assert_eq!(reader.field(3).unwrap().point_count, 4);
    let mut actual: Vec<_> = reader
        .decode_all_points(3)
        .unwrap()
        .into_iter()
        .map(|point| (point.doc_id, point.packed_value))
        .collect();
    actual.sort();
    assert_eq!(
        actual,
        vec![
            (0, 9u32.to_be_bytes().to_vec()),
            (1, 1u32.to_be_bytes().to_vec()),
            (2, 5u32.to_be_bytes().to_vec()),
            (2, 5u32.to_be_bytes().to_vec()),
        ]
    );
}

#[test]
fn empty_points_index_opens_with_zero_fields() {
    let (kdm, kdi, kdd) = build_empty_points_index();
    let reader = open(&kdm, &kdi, &kdd, &id(), "").unwrap();
    assert!(reader.field(0).is_none());
}

#[test]
fn wrong_segment_id_rejected() {
    let (kdm, kdi, kdd) = build_empty_points_index();
    let wrong_id = [9u8; codec_util::ID_LENGTH];
    assert!(open(&kdm, &kdi, &kdd, &wrong_id, "").is_err());
}

#[test]
fn illegal_field_number_rejected() {
    let (kdm, kdi, kdd) = build_empty_points_index();
    // `build_empty_points_index` writes the field-loop terminator (-1)
    // as the very first bytes after the index header; splice in an
    // illegal (negative, non -1) field number before it instead.
    let mut header = Vec::new();
    header.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
    write_vint_i32(&mut header, META_CODEC_NAME.len() as i32);
    header.extend_from_slice(META_CODEC_NAME.as_bytes());
    header.extend_from_slice(&(VERSION_CURRENT as u32).to_be_bytes());
    header.extend_from_slice(&id());
    header.push(0);
    assert_eq!(&kdm[..header.len()], header.as_slice());

    let mut patched = header;
    patched.extend_from_slice(&(-5i32).to_le_bytes()); // illegal field number
    patched.extend_from_slice(&0i64.to_le_bytes()); // indexLength
    patched.extend_from_slice(&0i64.to_le_bytes()); // dataLength
    patched.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
    patched.extend_from_slice(&0u32.to_be_bytes());
    let checksum = crc32fast::hash(&patched) as u64;
    patched.extend_from_slice(&checksum.to_be_bytes());

    assert!(matches!(
        open(&patched, &kdi, &kdd, &id(), ""),
        Err(Error::IllegalFieldNumber(-5))
    ));
}

fn long_sortable_bytes(v: i64) -> Vec<u8> {
    // NumericUtils.longToSortableBytes: flip the sign bit, then big-endian.
    ((v ^ i64::MIN) as u64).to_be_bytes().to_vec()
}

#[test]
fn write_then_read_single_leaf_continuous_ids_round_trips() {
    let points: Vec<(i32, Vec<u8>)> = (0..10)
        .map(|i| (i, long_sortable_bytes((i as i64) * 100 - 500)))
        .collect();
    let field = WritePointsField {
        field_number: 3,
        num_dims: 1,
        num_index_dims: 1,
        bytes_per_dim: 8,
        points: points.clone(),
    };
    let (kdm, kdi, kdd) = write(&[field], 512, &id(), "").unwrap();

    let reader = open(&kdm, &kdi, &kdd, &id(), "").unwrap();
    let meta = reader.field(3).unwrap();
    assert_eq!(meta.num_dims, 1);
    assert_eq!(meta.num_index_dims, 1);
    assert_eq!(meta.bytes_per_dim, 8);
    assert_eq!(meta.num_leaves, 1);
    assert_eq!(meta.point_count, 10);
    assert_eq!(meta.doc_count, 10);
    assert_eq!(meta.max_points_in_leaf_node, 512);

    let mut decoded = reader.decode_all_points(3).unwrap();
    decoded.sort_by_key(|p| p.doc_id);
    let mut expected: Vec<Point> = points
        .into_iter()
        .map(|(doc_id, packed_value)| Point {
            doc_id,
            packed_value,
        })
        .collect();
    expected.sort_by_key(|p| p.doc_id);
    assert_eq!(decoded, expected);
}

#[test]
fn write_then_read_single_leaf_non_continuous_ids_round_trips() {
    // Every third doc skips the field, like GenPoints.java's real fixture
    // -- forces the BPV_32 doc-id path instead of CONTINUOUS_IDS.
    let points: Vec<(i32, Vec<u8>)> = (0..30)
        .filter(|i| i % 3 != 0)
        .map(|i| (i, long_sortable_bytes((i as i64) * 7919 - 1_000_000)))
        .collect();
    let field = WritePointsField {
        field_number: 0,
        num_dims: 1,
        num_index_dims: 1,
        bytes_per_dim: 8,
        points: points.clone(),
    };
    let (kdm, kdi, kdd) = write(&[field], 512, &id(), "").unwrap();

    let reader = open(&kdm, &kdi, &kdd, &id(), "").unwrap();
    let meta = reader.field(0).unwrap();
    assert_eq!(meta.point_count, points.len() as i64);
    assert_eq!(meta.doc_count, points.len() as i32);

    let mut decoded = reader.decode_all_points(0).unwrap();
    decoded.sort_by_key(|p| p.doc_id);
    let mut expected: Vec<Point> = points
        .into_iter()
        .map(|(doc_id, packed_value)| Point {
            doc_id,
            packed_value,
        })
        .collect();
    expected.sort_by_key(|p| p.doc_id);
    assert_eq!(decoded, expected);
}

#[test]
fn write_multiple_fields_round_trips() {
    let field_a = WritePointsField {
        field_number: 0,
        num_dims: 1,
        num_index_dims: 1,
        bytes_per_dim: 4,
        points: vec![
            (0, vec![0, 0, 0, 1]),
            (1, vec![0, 0, 0, 2]),
            (2, vec![0, 0, 0, 3]),
        ],
    };
    let field_b = WritePointsField {
        field_number: 1,
        num_dims: 1,
        num_index_dims: 1,
        bytes_per_dim: 8,
        points: vec![(5, long_sortable_bytes(42)), (7, long_sortable_bytes(-1))],
    };
    let (kdm, kdi, kdd) = write(&[field_a.clone(), field_b.clone()], 512, &id(), "").unwrap();

    let reader = open(&kdm, &kdi, &kdd, &id(), "").unwrap();
    assert!(reader.field(0).is_some());
    assert!(reader.field(1).is_some());

    let mut got_a = reader.decode_all_points(0).unwrap();
    got_a.sort_by_key(|p| p.doc_id);
    assert_eq!(
        got_a,
        vec![
            Point {
                doc_id: 0,
                packed_value: vec![0, 0, 0, 1]
            },
            Point {
                doc_id: 1,
                packed_value: vec![0, 0, 0, 2]
            },
            Point {
                doc_id: 2,
                packed_value: vec![0, 0, 0, 3]
            },
        ]
    );

    let mut got_b = reader.decode_all_points(1).unwrap();
    got_b.sort_by_key(|p| p.doc_id);
    assert_eq!(
        got_b,
        vec![
            Point {
                doc_id: 5,
                packed_value: long_sortable_bytes(42)
            },
            Point {
                doc_id: 7,
                packed_value: long_sortable_bytes(-1)
            },
        ]
    );
}

#[test]
fn write_single_point_round_trips() {
    let field = WritePointsField {
        field_number: 0,
        num_dims: 1,
        num_index_dims: 1,
        bytes_per_dim: 8,
        points: vec![(9, long_sortable_bytes(123_456_789))],
    };
    let (kdm, kdi, kdd) = write(&[field], 512, &id(), "").unwrap();
    let reader = open(&kdm, &kdi, &kdd, &id(), "").unwrap();
    let decoded = reader.decode_all_points(0).unwrap();
    assert_eq!(
        decoded,
        vec![Point {
            doc_id: 9,
            packed_value: long_sortable_bytes(123_456_789)
        }]
    );
}

#[test]
fn write_then_read_two_leaves_round_trips() {
    // 8 points, max 4 => exactly 2 leaves (numLeftLeafNodes(2) == 1).
    let points: Vec<(i32, Vec<u8>)> = (0..8)
        .map(|i| (i, long_sortable_bytes((i as i64) * 1000)))
        .collect();
    let field = WritePointsField {
        field_number: 0,
        num_dims: 1,
        num_index_dims: 1,
        bytes_per_dim: 8,
        points: points.clone(),
    };
    let (kdm, kdi, kdd) = write(&[field], 4, &id(), "").unwrap();

    let reader = open(&kdm, &kdi, &kdd, &id(), "").unwrap();
    let meta = reader.field(0).unwrap();
    assert_eq!(meta.num_leaves, 2);
    assert_eq!(meta.point_count, 8);

    let mut decoded = reader.decode_all_points(0).unwrap();
    decoded.sort_by_key(|p| p.doc_id);
    let mut expected: Vec<Point> = points
        .into_iter()
        .map(|(doc_id, packed_value)| Point {
            doc_id,
            packed_value,
        })
        .collect();
    expected.sort_by_key(|p| p.doc_id);
    assert_eq!(decoded, expected);
}

#[test]
fn write_then_read_many_leaves_round_trips() {
    // 173 points (deliberately not a multiple of the leaf size,
    // deliberately not a power of two leaf count), max 4/leaf => 44
    // leaves, forcing several levels of recursion, an unbalanced final
    // level, and both the `numLeftLeafNodes == 1` and `> 1` branches of
    // `pack_index`. Every third doc skips the field (like
    // `GenPoints.java`) so doc ids aren't a trivial consecutive run
    // within every leaf.
    let points: Vec<(i32, Vec<u8>)> = (0..300)
        .filter(|i| i % 3 != 0)
        .map(|i| (i, long_sortable_bytes((i as i64) * 7919 - 1_000_000)))
        .collect();
    let expected_leaves = points.len().div_ceil(4) as i32;
    let field = WritePointsField {
        field_number: 0,
        num_dims: 1,
        num_index_dims: 1,
        bytes_per_dim: 8,
        points: points.clone(),
    };
    let (kdm, kdi, kdd) = write(&[field], 4, &id(), "").unwrap();

    let reader = open(&kdm, &kdi, &kdd, &id(), "").unwrap();
    let meta = reader.field(0).unwrap();
    assert_eq!(meta.num_leaves, expected_leaves);
    assert_eq!(meta.point_count, points.len() as i64);
    assert_eq!(meta.doc_count, points.len() as i32);

    let mut decoded = reader.decode_all_points(0).unwrap();
    decoded.sort_by_key(|p| p.doc_id);
    let mut expected: Vec<Point> = points
        .into_iter()
        .map(|(doc_id, packed_value)| Point {
            doc_id,
            packed_value,
        })
        .collect();
    expected.sort_by_key(|p| p.doc_id);
    assert_eq!(decoded, expected);
}

#[test]
fn write_then_read_2d_multi_leaf_round_trips() {
    // LatLonPoint-shaped: 2 dimensions, 4 bytes each. 100 points, max
    // 4/leaf => 25 leaves, exercising multi-dimension split-dimension
    // selection (widest_dim) across several recursion levels together
    // with multi-leaf packing.
    let points: Vec<(i32, Vec<u8>)> = (0..100i32)
        .map(|i| {
            let lat = (i * 37) % 1000; // narrower range
            let lon = (i * 9973) % 1_000_000; // much wider range
            let mut v = Vec::with_capacity(8);
            v.extend_from_slice(&lat.to_be_bytes());
            v.extend_from_slice(&lon.to_be_bytes());
            (i, v)
        })
        .collect();
    let field = WritePointsField {
        field_number: 0,
        num_dims: 2,
        num_index_dims: 2,
        bytes_per_dim: 4,
        points: points.clone(),
    };
    let (kdm, kdi, kdd) = write(&[field], 4, &id(), "").unwrap();

    let reader = open(&kdm, &kdi, &kdd, &id(), "").unwrap();
    let meta = reader.field(0).unwrap();
    assert_eq!(meta.num_dims, 2);
    assert_eq!(meta.num_index_dims, 2);
    assert_eq!(meta.num_leaves, points.len().div_ceil(4) as i32);
    assert_eq!(meta.point_count, points.len() as i64);

    let mut decoded = reader.decode_all_points(0).unwrap();
    decoded.sort_by_key(|p| p.doc_id);
    let mut expected: Vec<Point> = points
        .into_iter()
        .map(|(doc_id, packed_value)| Point {
            doc_id,
            packed_value,
        })
        .collect();
    expected.sort_by_key(|p| p.doc_id);
    assert_eq!(decoded, expected);
}

/// A 2D field of 100 points, 4 per leaf: (doc, packed) pairs and the
/// reader's files.
#[allow(clippy::type_complexity)]
fn two_d_field() -> (Vec<(i32, Vec<u8>)>, (Vec<u8>, Vec<u8>, Vec<u8>)) {
    let points: Vec<(i32, Vec<u8>)> = (0..100i32)
        .map(|i| {
            let mut v = Vec::with_capacity(8);
            v.extend_from_slice(&((i * 37) % 1000).to_be_bytes());
            v.extend_from_slice(&((i * 9973) % 1_000_000).to_be_bytes());
            (i, v)
        })
        .collect();
    let field = WritePointsField {
        field_number: 0,
        num_dims: 2,
        num_index_dims: 2,
        bytes_per_dim: 4,
        points: points.clone(),
    };
    (points, write(&[field], 4, &id(), "").unwrap())
}

/// Every node of the field's tree, depth first, left before right.
fn all_nodes(reader: &PointsReader<'_>) -> Vec<PointTreeNode> {
    let mut out = Vec::new();
    let mut stack = vec![reader.point_tree(0).unwrap()];
    while let Some(node) = stack.pop() {
        if let Some((l, r)) = reader.point_tree_children(&node).unwrap() {
            stack.push(r);
            stack.push(l);
        }
        out.push(node);
    }
    out
}

#[test]
fn point_tree_cells_contain_their_leaves_points() {
    let (points, (kdm, kdi, kdd)) = two_d_field();
    let reader = open(&kdm, &kdi, &kdd, &id(), "").unwrap();
    let root = reader.point_tree(0).unwrap();
    let meta = reader.field(0).unwrap();
    assert_eq!(root.min_packed(), meta.min_packed_value);
    assert_eq!(root.max_packed(), meta.max_packed_value);
    assert!(!root.is_leaf());
    // Every node: its children split its cell; every leaf's points lie in
    // its cell; the leaves hold every point once, in leaf order.
    let nodes = all_nodes(&reader);
    assert_eq!(nodes.len(), 2 * meta.num_leaves as usize - 1);
    let mut seen = Vec::new();
    let mut leaves = 0;
    for node in &nodes {
        let (min, max) = (node.min_packed(), node.max_packed());
        match reader.point_tree_children(node).unwrap() {
            Some((l, r)) => {
                assert_eq!(l.min_packed(), min);
                assert_eq!(r.max_packed(), max);
                assert!(!node.is_leaf());
            }
            None => {
                assert!(node.is_leaf());
                leaves += 1;
                let mut v = Collect(Vec::new(), Relation::CellCrossesQuery);
                reader.visit_leaf(node, &mut v).unwrap();
                for (doc, packed) in v.0 {
                    for d in 0..2 {
                        let r = d * 4..d * 4 + 4;
                        assert!(packed[r.clone()] >= min[r.clone()]);
                        assert!(packed[r.clone()] <= max[r]);
                    }
                    seen.push(doc);
                }
            }
        }
    }
    assert_eq!(leaves, meta.num_leaves);
    let all: Vec<i32> = reader
        .decode_all_points(0)
        .unwrap()
        .into_iter()
        .map(|p| p.doc_id)
        .collect();
    assert_eq!(seen, all, "leaf order");
    seen.sort_unstable();
    assert_eq!(seen, points.iter().map(|p| p.0).collect::<Vec<_>>());
    // An inner node visits nothing; an unknown field is an error.
    let mut v = Collect(Vec::new(), Relation::CellCrossesQuery);
    reader.visit_leaf(&root, &mut v).unwrap();
    assert!(v.0.is_empty());
    assert!(reader.point_tree(7).is_err());
}

#[test]
fn a_node_of_another_field_is_refused() {
    // Field 0: one dimension of 8 bytes; field 1: two of 4. Their packed
    // values are the same length, their trees are not.
    let one_d = WritePointsField {
        field_number: 0,
        num_dims: 1,
        num_index_dims: 1,
        bytes_per_dim: 8,
        points: (0..100i32)
            .map(|i| (i, i64::from(i * 7919 % 1000).to_be_bytes().to_vec()))
            .collect(),
    };
    let (points, _) = two_d_field();
    let two_d = WritePointsField {
        field_number: 1,
        num_dims: 2,
        num_index_dims: 2,
        bytes_per_dim: 4,
        points,
    };
    // Field 2: one dimension of 4 bytes, shorter packed values.
    let short = WritePointsField {
        field_number: 2,
        num_dims: 1,
        num_index_dims: 1,
        bytes_per_dim: 4,
        points: (0..10i32).map(|i| (i, i.to_be_bytes().to_vec())).collect(),
    };
    let (kdm, kdi, kdd) = write(&[two_d], 4, &id(), "").unwrap();
    let reader = open(&kdm, &kdi, &kdd, &id(), "").unwrap();
    let two_root = reader.point_tree(1).unwrap();
    let two_child = reader.point_tree_children(&two_root).unwrap().unwrap().0;
    // Another segment's reader, where field 1 is a one-dimension field:
    // of 8 bytes (same packed length, other tree), of 4 bytes, or absent.
    let renumber = |mut f: WritePointsField| {
        f.field_number = 1;
        f
    };
    for other in [
        vec![renumber(one_d.clone())],
        vec![renumber(short.clone())],
        vec![short.clone()],
    ] {
        let (kdm, kdi, kdd) = write(&other, 4, &id(), "").unwrap();
        let other = open(&kdm, &kdi, &kdd, &id(), "").unwrap();
        for node in [&two_root, &two_child] {
            assert!(other.point_tree_children(node).is_err());
            let mut v = Collect(Vec::new(), Relation::CellCrossesQuery);
            assert!(other.visit_leaf(node, &mut v).is_err());
            assert!(v.0.is_empty());
        }
    }
    // Its own reader takes it.
    assert!(reader.point_tree_children(&two_child).unwrap().is_some());
}

#[test]
fn count_points_refuses_a_multi_dimension_field() {
    // Its leaves are related by their own boxes, so a crossing cell's
    // leaf may arrive whole through `visit_many`: not a count `pointCount`
    // (one dimension only) ever makes.
    let (_, (kdm, kdi, kdd)) = two_d_field();
    let reader = open(&kdm, &kdi, &kdd, &id(), "").unwrap();
    let e = reader
        .count_points(0, &mut Collect(Vec::new(), Relation::CellCrossesQuery))
        .unwrap_err();
    assert!(
        matches!(
            e,
            Error::MultiDimensionCount {
                field_number: 0,
                num_dims: 2
            }
        ),
        "{e}"
    );
}

#[test]
fn a_multi_dimension_leaf_is_related_by_its_own_box() {
    let (points, (kdm, kdi, kdd)) = two_d_field();
    let reader = open(&kdm, &kdi, &kdd, &id(), "").unwrap();
    let cells: Vec<(Vec<u8>, Vec<u8>)> = all_nodes(&reader)
        .iter()
        .map(|n| (n.min_packed().to_vec(), n.max_packed().to_vec()))
        .collect();
    // Every tree cell crosses; a box that is no tree cell is a leaf's own
    // (`visitDocValuesWithCardinality`'s second look) and answers `leaf`.
    for (leaf, ids, values) in [
        (Relation::CellInsideQuery, points.len(), 0),
        (Relation::CellOutsideQuery, 0, 0),
        (Relation::CellCrossesQuery, 0, points.len()),
    ] {
        let mut v = LeafRelation {
            cells: &cells,
            leaf,
            leaf_calls: 0,
            ids: Vec::new(),
            values: 0,
        };
        reader.intersect(0, &mut v).unwrap();
        assert_eq!(v.leaf_calls, reader.field(0).unwrap().num_leaves as usize);
        assert_eq!((v.ids.len(), v.values), (ids, values), "{leaf:?}");
    }
}

struct LeafRelation<'c> {
    cells: &'c [(Vec<u8>, Vec<u8>)],
    leaf: Relation,
    leaf_calls: usize,
    ids: Vec<i32>,
    values: usize,
}

impl IntersectVisitor for LeafRelation<'_> {
    fn compare(&mut self, min: &[u8], max: &[u8]) -> Relation {
        if self.cells.iter().any(|(a, b)| a == min && b == max) {
            return Relation::CellCrossesQuery;
        }
        self.leaf_calls += 1;
        self.leaf
    }
    fn visit(&mut self, doc_id: i32) {
        self.ids.push(doc_id);
    }
    fn visit_with_value(&mut self, _doc_id: i32, _packed: &[u8]) {
        self.values += 1;
    }
}

struct Collect(Vec<(i32, Vec<u8>)>, Relation);

impl IntersectVisitor for Collect {
    fn compare(&mut self, _min: &[u8], _max: &[u8]) -> Relation {
        self.1
    }
    fn visit(&mut self, _doc_id: i32) {}
    fn visit_with_value(&mut self, doc_id: i32, packed: &[u8]) {
        self.0.push((doc_id, packed.to_vec()));
    }
}

#[test]
fn write_then_read_3d_multi_leaf_round_trips() {
    // 3 dimensions, 2 bytes each, non-continuous doc ids (every third
    // doc skips the field, like the 1D fixtures above).
    let points: Vec<(i32, Vec<u8>)> = (0..150i32)
        .filter(|i| i % 3 != 0)
        .map(|i| {
            let d0 = ((i * 41) % 500) as u16;
            let d1 = ((i * 173) % 30000) as u16;
            let d2 = ((i * 7) % 60000) as u16;
            let mut v = Vec::with_capacity(6);
            v.extend_from_slice(&d0.to_be_bytes());
            v.extend_from_slice(&d1.to_be_bytes());
            v.extend_from_slice(&d2.to_be_bytes());
            (i, v)
        })
        .collect();
    let expected_leaves = points.len().div_ceil(8) as i32;
    let field = WritePointsField {
        field_number: 3,
        num_dims: 3,
        num_index_dims: 3,
        bytes_per_dim: 2,
        points: points.clone(),
    };
    let (kdm, kdi, kdd) = write(&[field], 8, &id(), "").unwrap();

    let reader = open(&kdm, &kdi, &kdd, &id(), "").unwrap();
    let meta = reader.field(3).unwrap();
    assert_eq!(meta.num_dims, 3);
    assert_eq!(meta.num_index_dims, 3);
    assert_eq!(meta.num_leaves, expected_leaves);
    assert_eq!(meta.point_count, points.len() as i64);
    assert_eq!(meta.doc_count, points.len() as i32);

    let mut decoded = reader.decode_all_points(3).unwrap();
    decoded.sort_by_key(|p| p.doc_id);
    let mut expected: Vec<Point> = points
        .into_iter()
        .map(|(doc_id, packed_value)| Point {
            doc_id,
            packed_value,
        })
        .collect();
    expected.sort_by_key(|p| p.doc_id);
    assert_eq!(decoded, expected);
}

#[test]
fn write_then_read_num_index_dims_less_than_num_dims_round_trips() {
    // 3 dimensions, but only the first 2 are index dims -- like a
    // `LatLonShape`-style bounding box where the 3rd dimension is a
    // non-indexed, data-only payload dimension. Multiple leaves, so both
    // the split-dimension selection (must never pick dim 2) and the
    // per-leaf/per-field bounding boxes (must only cover dims 0-1) get
    // exercised, not just single-leaf storage.
    let points: Vec<(i32, Vec<u8>)> = (0..40i32)
        .map(|i| {
            let d0 = ((i * 41) % 500) as u16;
            let d1 = ((i * 173) % 30000) as u16;
            // Data-only dim: deliberately made *wider-ranging* than the
            // two index dims, so a widest-dim implementation that (wrongly)
            // considered all `num_dims` dimensions instead of only the
            // first `num_index_dims` would pick this one instead.
            let d2 = (i * 1777) as u16;
            let mut v = Vec::with_capacity(6);
            v.extend_from_slice(&d0.to_be_bytes());
            v.extend_from_slice(&d1.to_be_bytes());
            v.extend_from_slice(&d2.to_be_bytes());
            (i, v)
        })
        .collect();
    let expected_leaves = points.len().div_ceil(8) as i32;
    let field = WritePointsField {
        field_number: 4,
        num_dims: 3,
        num_index_dims: 2,
        bytes_per_dim: 2,
        points: points.clone(),
    };
    let (kdm, kdi, kdd) = write(&[field], 8, &id(), "").unwrap();

    let reader = open(&kdm, &kdi, &kdd, &id(), "").unwrap();
    let meta = reader.field(4).unwrap();
    assert_eq!(meta.num_dims, 3);
    assert_eq!(meta.num_index_dims, 2);
    assert_eq!(meta.num_leaves, expected_leaves);
    assert_eq!(meta.point_count, points.len() as i64);
    assert_eq!(meta.doc_count, points.len() as i32);
    // min/max packed value is sized for index dims only (2 * 2 bytes),
    // not the full 3 dims.
    assert_eq!(meta.min_packed_value.len(), 4);
    assert_eq!(meta.max_packed_value.len(), 4);

    let mut decoded = reader.decode_all_points(4).unwrap();
    decoded.sort_by_key(|p| p.doc_id);
    let mut expected: Vec<Point> = points
        .into_iter()
        .map(|(doc_id, packed_value)| Point {
            doc_id,
            packed_value,
        })
        .collect();
    expected.sort_by_key(|p| p.doc_id);
    // Every point's full packed value -- including the non-indexed 3rd
    // dimension -- must survive the round trip unchanged.
    assert_eq!(decoded, expected);
}

#[test]
fn write_then_read_three_leaves_unbalanced_round_trips() {
    // 9 points, max 4 => ceil(9/4) = 3 leaves (numLeftLeafNodes(3) ==
    // 2), exercising the same 2-leaves-left/1-leaf-right shape as the
    // hand-built `three_leaf_decode_leaf_pointers` unit test above, but
    // now produced by the writer instead of hand-encoded.
    let points: Vec<(i32, Vec<u8>)> = (0..9).map(|i| (100 + i, vec![(255 - i) as u8])).collect();
    let field = WritePointsField {
        field_number: 7,
        num_dims: 1,
        num_index_dims: 1,
        bytes_per_dim: 1,
        points: points.clone(),
    };
    let (kdm, kdi, kdd) = write(&[field], 4, &id(), "").unwrap();

    let reader = open(&kdm, &kdi, &kdd, &id(), "").unwrap();
    let meta = reader.field(7).unwrap();
    assert_eq!(meta.num_leaves, 3);

    let mut decoded = reader.decode_all_points(7).unwrap();
    decoded.sort_by_key(|p| p.doc_id);
    let mut expected: Vec<Point> = points
        .into_iter()
        .map(|(doc_id, packed_value)| Point {
            doc_id,
            packed_value,
        })
        .collect();
    expected.sort_by_key(|p| p.doc_id);
    assert_eq!(decoded, expected);
}

#[test]
fn write_then_read_exactly_at_max_points_per_leaf_stays_single_leaf() {
    // count == max exactly: must stay a single leaf (num_leaves ==
    // ceil(count / max) == 1), the boundary just below the split trigger.
    let points: Vec<(i32, Vec<u8>)> = (0..4)
        .map(|i| (i, long_sortable_bytes((i as i64) * 1000)))
        .collect();
    let field = WritePointsField {
        field_number: 0,
        num_dims: 1,
        num_index_dims: 1,
        bytes_per_dim: 8,
        points: points.clone(),
    };
    let (kdm, kdi, kdd) = write(&[field], 4, &id(), "").unwrap();
    let reader = open(&kdm, &kdi, &kdd, &id(), "").unwrap();
    let meta = reader.field(0).unwrap();
    assert_eq!(meta.num_leaves, 1);
    assert_eq!(meta.point_count, 4);

    let mut decoded = reader.decode_all_points(0).unwrap();
    decoded.sort_by_key(|p| p.doc_id);
    let mut expected: Vec<Point> = points
        .into_iter()
        .map(|(doc_id, packed_value)| Point {
            doc_id,
            packed_value,
        })
        .collect();
    expected.sort_by_key(|p| p.doc_id);
    assert_eq!(decoded, expected);
}

#[test]
fn write_then_read_one_over_max_points_per_leaf_splits_into_two() {
    // count == max + 1: exactly one point over the threshold must
    // trigger a split into 2 leaves (the classic BKD off-by-one
    // boundary), with the left leaf getting exactly `max` points and the
    // right leaf getting the single leftover point.
    let points: Vec<(i32, Vec<u8>)> = (0..5)
        .map(|i| (i, long_sortable_bytes((i as i64) * 1000)))
        .collect();
    let field = WritePointsField {
        field_number: 0,
        num_dims: 1,
        num_index_dims: 1,
        bytes_per_dim: 8,
        points: points.clone(),
    };
    let (kdm, kdi, kdd) = write(&[field], 4, &id(), "").unwrap();
    let reader = open(&kdm, &kdi, &kdd, &id(), "").unwrap();
    let meta = reader.field(0).unwrap();
    assert_eq!(meta.num_leaves, 2);
    assert_eq!(meta.point_count, 5);

    let mut decoded = reader.decode_all_points(0).unwrap();
    decoded.sort_by_key(|p| p.doc_id);
    let mut expected: Vec<Point> = points
        .into_iter()
        .map(|(doc_id, packed_value)| Point {
            doc_id,
            packed_value,
        })
        .collect();
    expected.sort_by_key(|p| p.doc_id);
    assert_eq!(decoded, expected);
}

#[test]
fn write_then_read_all_points_identical_degenerate_case() {
    // Every point identical in every dimension -- no dimension has any
    // variance to split on. A real BKD tree still must split purely by
    // count (never by value), producing several valid leaves rather
    // than looping forever or panicking. 2 dimensions so widest_dim's
    // all-zero-range tie-break (dimension 0) is actually exercised.
    let value = vec![7u8, 7, 7, 7, 9, 9, 9, 9]; // 2 dims x 4 bytes, identical
    let points: Vec<(i32, Vec<u8>)> = (0..10).map(|i| (i, value.clone())).collect();
    let field = WritePointsField {
        field_number: 0,
        num_dims: 2,
        num_index_dims: 2,
        bytes_per_dim: 4,
        points: points.clone(),
    };
    let (kdm, kdi, kdd) = write(&[field], 4, &id(), "").unwrap();
    let reader = open(&kdm, &kdi, &kdd, &id(), "").unwrap();
    let meta = reader.field(0).unwrap();
    assert_eq!(meta.num_leaves, 3); // ceil(10/4)
    assert_eq!(meta.point_count, 10);

    let mut decoded = reader.decode_all_points(0).unwrap();
    decoded.sort_by_key(|p| p.doc_id);
    let mut expected: Vec<Point> = points
        .into_iter()
        .map(|(doc_id, packed_value)| Point {
            doc_id,
            packed_value,
        })
        .collect();
    expected.sort_by_key(|p| p.doc_id);
    assert_eq!(decoded, expected);
}

#[test]
fn write_then_read_last_dim_only_varies_multi_leaf_round_trips() {
    // Full write/read round-trip of the same shape as
    // `widest_dim_picks_last_dim_when_only_it_varies` above, but through
    // the real writer with enough points to force multiple leaves --
    // proves compute_leaf_plan actually splits on dimension 2 at every
    // level rather than stalling because dims 0/1 look unsplittable.
    let points: Vec<(i32, Vec<u8>)> = (0..40i32).map(|i| (i, vec![5, 5, (i * 3) as u8])).collect();
    let field = WritePointsField {
        field_number: 0,
        num_dims: 3,
        num_index_dims: 3,
        bytes_per_dim: 1,
        points: points.clone(),
    };
    let (kdm, kdi, kdd) = write(&[field], 4, &id(), "").unwrap();
    let reader = open(&kdm, &kdi, &kdd, &id(), "").unwrap();
    let meta = reader.field(0).unwrap();
    assert_eq!(meta.num_leaves, 10); // ceil(40/4)
    assert_eq!(meta.point_count, 40);

    let mut decoded = reader.decode_all_points(0).unwrap();
    decoded.sort_by_key(|p| p.doc_id);
    let mut expected: Vec<Point> = points
        .into_iter()
        .map(|(doc_id, packed_value)| Point {
            doc_id,
            packed_value,
        })
        .collect();
    expected.sort_by_key(|p| p.doc_id);
    assert_eq!(decoded, expected);
}

#[test]
fn write_packed_rejects_values_that_do_not_fill_the_points() {
    for len in [7, 9, 0] {
        let field = PackedPointsField {
            field_number: 3,
            num_dims: 1,
            num_index_dims: 1,
            bytes_per_dim: 4,
            docs: vec![0, 1],
            values: vec![0; len],
        };
        assert!(
            matches!(
                write_packed(vec![field], 512, 2, &id(), ""),
                Err(Error::PackedValuesLength {
                    field_number: 3,
                    points: 2,
                    stride: 4,
                    actual,
                }) if actual == len
            ),
            "values of {len} bytes"
        );
    }
}

#[test]
fn write_rejects_empty_field() {
    let field = WritePointsField {
        field_number: 0,
        num_dims: 1,
        num_index_dims: 1,
        bytes_per_dim: 4,
        points: vec![],
    };
    assert!(matches!(
        write(&[field], 512, &id(), ""),
        Err(Error::EmptyField { field_number: 0 })
    ));
}

#[test]
fn write_rejects_num_index_dims_zero() {
    let field = WritePointsField {
        field_number: 0,
        num_dims: 2,
        num_index_dims: 0,
        bytes_per_dim: 4,
        points: vec![(0, vec![0u8; 8])],
    };
    assert!(matches!(
        write(&[field], 512, &id(), ""),
        Err(Error::InvalidNumIndexDims {
            field_number: 0,
            num_dims: 2,
            num_index_dims: 0,
        })
    ));
}

#[test]
fn write_rejects_num_index_dims_greater_than_num_dims() {
    let field = WritePointsField {
        field_number: 0,
        num_dims: 2,
        num_index_dims: 3,
        bytes_per_dim: 4,
        points: vec![(0, vec![0u8; 8])],
    };
    assert!(matches!(
        write(&[field], 512, &id(), ""),
        Err(Error::InvalidNumIndexDims {
            field_number: 0,
            num_dims: 2,
            num_index_dims: 3,
        })
    ));
}

#[test]
fn write_rejects_wrong_packed_value_length() {
    let field = WritePointsField {
        field_number: 0,
        num_dims: 1,
        num_index_dims: 1,
        bytes_per_dim: 8,
        points: vec![(0, vec![1, 2, 3])],
    };
    assert!(matches!(
        write(&[field], 512, &id(), ""),
        Err(Error::WrongPackedValueLength {
            field_number: 0,
            index: 0,
            expected: 8,
            actual: 3,
        })
    ));
}

#[test]
fn write_then_read_rejects_wrong_segment_id() {
    let field = WritePointsField {
        field_number: 0,
        num_dims: 1,
        num_index_dims: 1,
        bytes_per_dim: 4,
        points: vec![(0, vec![0, 0, 0, 1])],
    };
    let (kdm, kdi, kdd) = write(&[field], 512, &id(), "").unwrap();
    let wrong_id = [9u8; codec_util::ID_LENGTH];
    assert!(open(&kdm, &kdi, &kdd, &wrong_id, "").is_err());
}

// ------------------------------------------------------------------
// Arithmetic-gate controls: values off a `.kdd` leaf block.
// ------------------------------------------------------------------

/// Java's `DocIdsWriter.readInts` decodes into
/// `BKDReaderDocIDSetIterator.docIDs`, a `new int[maxPointsInLeafNode]`
/// allocated once, so a larger `count` throws there before a doc id lands.
/// This port allocates per leaf, so the same vint sized a fresh `Vec`: a
/// *negative* one became `usize::MAX` and `Vec::with_capacity` panicked
/// with "capacity overflow", and a merely huge one is the abort shape.
#[test]
fn leaf_count_outside_max_points_in_leaf_node_is_a_decode_error() {
    let field = field_1d(1);
    for count in [-1i32, field.max_points_in_leaf_node + 1, i32::MAX] {
        let mut bytes = Vec::new();
        write_vint(&mut bytes, count);
        bytes.push(BPV_32 as u8);
        let mut input = SliceInput::new(&bytes);
        let mut out = Vec::new();
        let err = read_leaf_block(&mut input, &field, &mut out).unwrap_err();
        assert!(
            format!("{err}").contains("maxPointsInLeafNode"),
            "count={count}: {err}"
        );
    }
}

/// A leaf common prefix wider than the dimension it belongs to.
/// `BKDWriter` computes it as the common prefix of that dimension's
/// `bytesPerDim` bytes, so no writer emits one; Java's `readBytes` spills
/// into the *next* dimension's bytes, and here the slice bound leaves
/// `scratch_value` outright.
#[test]
fn leaf_common_prefix_outside_bytes_per_dim_is_a_decode_error() {
    let field = field_1d(2);
    for prefix in [3i32, -1] {
        let mut bytes = Vec::new();
        write_vint(&mut bytes, 1); // count
        bytes.push(CONTINUOUS_IDS as u8);
        write_vint(&mut bytes, 0); // docBase
        write_vint(&mut bytes, prefix);
        bytes.extend_from_slice(&[0u8; 8]);
        let mut input = SliceInput::new(&bytes);
        let mut out = Vec::new();
        let err = read_leaf_block(&mut input, &field, &mut out).unwrap_err();
        assert!(
            format!("{err}").contains("past bytesPerDim"),
            "prefix={prefix}: {err}"
        );
    }
}

/// `BKDWriter.writeLeafBlockPackedValues` asserts
/// `commonPrefixLengths[sortedDim] < bytesPerDim` before it can pick a
/// non-negative compressed dimension -- the run-length-compressed byte is
/// the first byte *after* that prefix, so a full-width prefix means there
/// is no such byte. Without the check `compressed_byte_offset` addressed
/// one past `scratch_value`.
#[test]
fn full_width_common_prefix_on_the_compressed_dim_is_a_decode_error() {
    let field = field_1d(2);
    let mut bytes = Vec::new();
    write_vint(&mut bytes, 1); // count
    bytes.push(CONTINUOUS_IDS as u8);
    write_vint(&mut bytes, 0); // docBase
    write_vint(&mut bytes, 2); // common prefix == bytesPerDim
    bytes.extend_from_slice(&[0xAA, 0xBB]);
    bytes.push(0x00); // compressedDim = 0
    bytes.push(0x01); // run byte
    bytes.push(1); // runLen
    let mut input = SliceInput::new(&bytes);
    let mut out = Vec::new();
    let err = read_leaf_block(&mut input, &field, &mut out).unwrap_err();
    assert!(
        format!("{err}").contains("full-width common prefix"),
        "{err}"
    );
}

/// A negative sub-block length in the low-cardinality (`-2`) layout. The
/// `i + length > count` guard was supposed to catch it, but `i + length`
/// overflowed first once `i` had advanced past the first run.
#[test]
fn negative_low_cardinality_run_length_is_a_decode_error() {
    let field = field_1d(1);
    let mut bytes = Vec::new();
    write_vint(&mut bytes, 4); // count
    bytes.push(CONTINUOUS_IDS as u8);
    write_vint(&mut bytes, 0); // docBase
    write_vint(&mut bytes, 0); // common prefix
    bytes.push(0xFE); // compressedDim = -2
    write_vint(&mut bytes, 2); // first run: length 2
    bytes.push(0xAA);
    write_vint(&mut bytes, -1); // second run: negative length
    let mut input = SliceInput::new(&bytes);
    let mut out = Vec::new();
    let err = read_leaf_block(&mut input, &field, &mut out).unwrap_err();
    assert!(
        format!("{err}").contains("negative low-cardinality run length"),
        "{err}"
    );
}

/// `read_bitset_ids`' three unbounded header values: `longLen` sized
/// `vec![0i64; n]` with nothing between it and the allocator, and
/// `offsetWords * 64` overflowed an `i32` on the way to the doc base.
#[test]
fn corrupt_bitset_doc_id_header_is_a_decode_error() {
    // Negative `longLen` -> `vec![0i64; usize::MAX]`.
    let mut bytes = Vec::new();
    bytes.push(BITSET_IDS as u8);
    write_vint(&mut bytes, 0);
    write_vint(&mut bytes, -1);
    let err = read_doc_ids(&mut SliceInput::new(&bytes), 1).unwrap_err();
    assert!(
        format!("{err}").contains("negative bitset doc-id header"),
        "{err}"
    );

    // A plausible-but-absurd `longLen`: 2^28 words is a 2 GB reservation
    // out of a nine-byte block.
    let mut bytes = Vec::new();
    bytes.push(BITSET_IDS as u8);
    write_vint(&mut bytes, 0);
    write_vint(&mut bytes, 1 << 28);
    let err = read_doc_ids(&mut SliceInput::new(&bytes), 1).unwrap_err();
    assert!(format!("{err}").contains("bitset doc ids"), "{err}");

    // `offsetWords * 64` past `i32::MAX`.
    let mut bytes = Vec::new();
    bytes.push(BITSET_IDS as u8);
    write_vint(&mut bytes, i32::MAX);
    write_vint(&mut bytes, 1);
    bytes.extend_from_slice(&1i64.to_le_bytes());
    let err = read_doc_ids(&mut SliceInput::new(&bytes), 1).unwrap_err();
    assert!(format!("{err}").contains("past i32::MAX"), "{err}");
}

/// Java decodes BPV_21's top field as `(int) (l >>> 42)`, an *unsigned*
/// shift over a 22-bit field. A signed `>>` turned a corrupt block's
/// negative word into a negative doc id instead of the in-range value
/// Java produces -- a silently different answer, not a rejected one.
#[test]
fn bpv21_top_field_is_zero_extended_like_java() {
    let mut bytes = Vec::new();
    bytes.push(BPV_21 as u8);
    bytes.extend_from_slice(&(-1i64).to_le_bytes());
    let ids = read_doc_ids(&mut SliceInput::new(&bytes), 3).unwrap();
    assert_eq!(ids, vec![0x1F_FFFF, 0x1F_FFFF, 0x3F_FFFF]);
}

/// A doc-id block whose fixed per-id cost already exceeds the bytes left
/// is refused before its buffer is reserved, not after.
#[test]
fn doc_id_blocks_are_bounded_by_the_bytes_that_remain() {
    for (marker, needle) in [
        (BPV_32, "BPV_32 doc ids"),
        (BPV_24, "BPV_24 doc ids"),
        (BPV_21, "BPV_21 doc ids"),
        (DELTA_BPV_16, "delta-16 doc ids"),
        (LEGACY_DELTA_VINT, "legacy delta-vint doc ids"),
    ] {
        let mut bytes = vec![marker as u8];
        bytes.extend_from_slice(&[0u8; 8]);
        let err = read_doc_ids(&mut SliceInput::new(&bytes), 100_000).unwrap_err();
        assert!(format!("{err}").contains(needle), "{marker}: {err}");
    }
}

/// The reservation cap that keeps a `.kdm`'s `numLeaves` from asking for
/// 17 GB. A well-formed file is unaffected: the `min` picks `numLeaves`.
#[test]
fn leaf_pointer_reservation_is_capped_by_the_packed_index_length() {
    assert_eq!(leaf_pointer_capacity(3, 64), 3);
    assert_eq!(leaf_pointer_capacity(1, 1), 1);
    // 2^31 leaves x 8 bytes = 17 GB, out of a 40-byte packed index.
    assert_eq!(leaf_pointer_capacity(i32::MAX, 40), 41);
}

/// The packed tree's node ids double at every level, so an `i32` runs out
/// after 31 of them. Java lets `nodeID * 2` wrap; a wrapped id compares
/// `< numLeaves` again, and since the walk only stops at a leaf the
/// recursion is then bounded by nothing but the `.kdi`'s length -- i.e. a
/// large enough packed index overflows the *stack*. Checking the multiply
/// caps the depth at 31 instead.
#[test]
fn a_packed_index_deeper_than_the_node_id_space_is_a_decode_error() {
    let mut field = field_1d(1);
    field.num_leaves = i32::MAX;
    // Root FP-delta vlong, then a chain of `(code=0, leftNumBytes=0)`
    // inner nodes at two bytes each -- 63 levels' worth, twice what it
    // takes to drive `node_id` past 2^30.
    let inner_nodes = vec![0u8; 128];
    let err = decode_leaf_pointers(&inner_nodes, &field).unwrap_err();
    assert!(format!("{err}").contains("node id overflows"), "{err}");
}

/// `PointValues.MAX_NUM_BYTES`, the ceiling `FieldInfo` and
/// `FieldType.setDimensions` put on every point field Java can index.
/// `BKDConfig` does not check it, so the read side does not either; the
/// write side refuses what `FieldInfo` would.
#[test]
fn write_rejects_bytes_per_dim_past_max_num_bytes() {
    let field = WritePointsField {
        field_number: 0,
        num_dims: 1,
        num_index_dims: 1,
        bytes_per_dim: MAX_NUM_BYTES + 1,
        points: vec![(0, vec![0u8; (MAX_NUM_BYTES + 1) as usize])],
    };
    match write(&[field], 512, &id(), "") {
        Err(Error::InvalidConfig(msg)) => {
            assert!(msg.contains("MAX_NUM_BYTES"), "{msg}")
        }
        other => panic!("expected InvalidConfig, got {other:?}"),
    }
}
