#![cfg(test)]
// A test's `i + 1` is not a length read off disk; see
// `docs/arithmetic-gate.md`'s "Test code" section.
#![allow(clippy::arithmetic_side_effects)]

use super::*;

// --- Hand-built FST bytes ---------------------------------------------
//
// These unit tests build minimal, hand-crafted FST bodies (list-encoded
// nodes only) to exercise `Fst::get`'s own boundary/error handling --
// the differential tests in `tests/fst_fixtures.rs` cover real,
// Lucene-written FSTs (with genuine arc sharing); see the
// `test-coverage` skill on why both layers are needed.

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

/// Builds a single-key FST body (list-encoded) accepting exactly `key`,
/// with final output `output`, using the classic "chain of arcs, each
/// node has exactly one arc, output lives on the arc immediately
/// preceding acceptance" shape. Bytes are laid out in *forward* address
/// order (index 0 = lowest address) since a `BytesReader` seeks directly
/// by absolute position; only reading direction is reversed.
///
/// Node for byte `key[i]` sits at a higher address than node for
/// `key[i+1]`; the last node (accepting) is the lowest-addressed / first
/// bytes in the array.
/// Appends one arc's bytes to `bytes`, given its fields in *logical*
/// (forward) read order (`flags`, `label`, ...). Because `BytesReader`
/// reads by decrementing its position (`ReverseBytesReader`), the
/// physical array must hold each field's bytes at *decreasing*
/// addresses relative to reading order -- i.e. the logical byte
/// sequence appended here must be pushed in reverse, so that the
/// address of `logical[0]` (the flags byte, returned as this arc's
/// node address) ends up higher than `logical[1]`, and so on. This
/// mirrors how real `FSTCompiler` grows its byte store from high
/// addresses down to low ones.
fn append_arc_logical(bytes: &mut Vec<u8>, logical: &[u8]) -> i64 {
    for &b in logical.iter().rev() {
        bytes.push(b);
    }
    (bytes.len() - 1) as i64
}

/// Builds a minimal, list-encoded (no array/direct-addressing nodes)
/// FST body accepting exactly `key` with final output `output`: one
/// node per key byte, each with exactly one arc (`BIT_LAST_ARC`),
/// explicit (non-`BIT_TARGET_NEXT`) `vlong` targets throughout, and the
/// arc for the last byte carrying `BIT_STOP_NODE | BIT_FINAL_ARC` (plus
/// the final output, if non-empty) instead of a target.
///
/// Nodes are appended starting from the *last* key byte, so each
/// non-final node's target (the already-appended next node's address)
/// is known by the time it's written.
fn build_single_key_fst(key: &[u8], output: &[u8]) -> (Vec<u8>, i64) {
    let mut bytes: Vec<u8> = Vec::new();
    let mut next_target: Option<i64> = None;
    let mut addrs = vec![0i64; key.len()];

    for i in (0..key.len()).rev() {
        let b = key[i];
        let is_last_byte = i == key.len() - 1;

        let mut logical = Vec::new();
        let mut flags = BIT_LAST_ARC;
        if is_last_byte {
            flags |= BIT_FINAL_ARC | BIT_STOP_NODE;
            if !output.is_empty() {
                flags |= BIT_ARC_HAS_FINAL_OUTPUT;
            }
            logical.push(flags);
            logical.push(b);
            if !output.is_empty() {
                write_vint(&mut logical, output.len() as i32);
                logical.extend_from_slice(output);
            }
        } else {
            logical.push(flags);
            logical.push(b);
            write_vlong(
                &mut logical,
                next_target.expect("target known from reverse pass"),
            );
        }

        let addr = append_arc_logical(&mut bytes, &logical);
        addrs[i] = addr;
        next_target = Some(addr);
    }

    let start_node = addrs[0];
    (bytes, start_node)
}

fn fst_from_body(
    bytes: Vec<u8>,
    start_node: i64,
    input_type: InputType,
    empty_output: Option<Vec<u8>>,
) -> Fst<'static> {
    fst_from_body_with_version(bytes, start_node, input_type, empty_output, VERSION_CURRENT)
}

fn fst_from_body_with_version(
    bytes: Vec<u8>,
    start_node: i64,
    input_type: InputType,
    empty_output: Option<Vec<u8>>,
    version: i32,
) -> Fst<'static> {
    Fst {
        metadata: FstMetadata {
            input_type,
            empty_output,
            start_node,
            version,
            num_bytes: bytes.len() as i64,
        },
        bytes: FstBytes::Owned(bytes),
    }
}

/// Appends one `FST.readLabel`-format label to `logical` (an
/// in-construction-order arc byte buffer, later reversed by
/// `append_arc_logical` -- see `build_single_key_fst`'s doc comment),
/// exactly mirroring `read_label`'s three decode branches so a
/// hand-built fixture round-trips through the real decoder.
fn write_label(logical: &mut Vec<u8>, input_type: InputType, version: i32, label: i32) {
    match input_type {
        InputType::Byte1 => {
            debug_assert!((0..=255).contains(&label));
            logical.push(label as u8);
        }
        InputType::Byte2 => {
            debug_assert!((0..=0xFFFF).contains(&label));
            let v = label as u16;
            // `read_label`'s `BYTE2` branch computes
            // `le = (b2 << 8) | b1` then (for legacy versions) byte-swaps
            // `le` to recover the original value -- so writing the
            // *already-swapped* value's low/high bytes here is what
            // makes that swap round-trip back to `v`.
            let sw = if version < VERSION_LITTLE_ENDIAN {
                v.swap_bytes()
            } else {
                v
            };
            logical.push((sw & 0xFF) as u8);
            logical.push((sw >> 8) as u8);
        }
        InputType::Byte4 => write_vint(logical, label),
    }
}

/// `BYTE2`/`BYTE4` analogue of `build_single_key_fst`: a minimal,
/// list-encoded FST accepting exactly one label sequence (`labels`) with
/// final output `output`, using `write_label` for each label instead of
/// `build_single_key_fst`'s raw byte push -- the only difference from
/// `build_single_key_fst`'s own construction beyond that is operating in
/// the `i32` label domain throughout.
fn build_single_label_seq_fst(
    labels: &[i32],
    output: &[u8],
    input_type: InputType,
    version: i32,
) -> (Vec<u8>, i64) {
    let mut bytes: Vec<u8> = Vec::new();
    let mut next_target: Option<i64> = None;
    let mut addrs = vec![0i64; labels.len()];

    for i in (0..labels.len()).rev() {
        let label = labels[i];
        let is_last = i == labels.len() - 1;

        let mut logical = Vec::new();
        let mut flags = BIT_LAST_ARC;
        if is_last {
            flags |= BIT_FINAL_ARC | BIT_STOP_NODE;
            if !output.is_empty() {
                flags |= BIT_ARC_HAS_FINAL_OUTPUT;
            }
            logical.push(flags);
            write_label(&mut logical, input_type, version, label);
            if !output.is_empty() {
                write_vint(&mut logical, output.len() as i32);
                logical.extend_from_slice(output);
            }
        } else {
            logical.push(flags);
            write_label(&mut logical, input_type, version, label);
            write_vlong(
                &mut logical,
                next_target.expect("target known from reverse pass"),
            );
        }

        let addr = append_arc_logical(&mut bytes, &logical);
        addrs[i] = addr;
        next_target = Some(addr);
    }

    let start_node = addrs[0];
    (bytes, start_node)
}

#[test]
fn read_label_decodes_byte2_little_endian() {
    let labels = [0x1234i32, 0x00ff, 0xffff];
    let (bytes, start) =
        build_single_label_seq_fst(&labels, b"v", InputType::Byte2, VERSION_CURRENT);
    let fst = fst_from_body_with_version(bytes, start, InputType::Byte2, None, VERSION_CURRENT);
    assert_eq!(
        fst.get_labels(&labels).unwrap(),
        Some(b"v".to_vec()),
        "BYTE2 little-endian label round-trip"
    );
}

#[test]
fn read_label_decodes_byte2_legacy_big_endian() {
    // `VERSION_START` (6) predates `VERSION_LITTLE_ENDIAN` (8): `BYTE2`
    // labels were stored byte-swapped relative to the current format.
    let labels = [0x1234i32, 0xabcd];
    let (bytes, start) = build_single_label_seq_fst(&labels, b"v", InputType::Byte2, VERSION_START);
    let fst = fst_from_body_with_version(bytes, start, InputType::Byte2, None, VERSION_START);
    assert_eq!(
        fst.get_labels(&labels).unwrap(),
        Some(b"v".to_vec()),
        "BYTE2 legacy big-endian label round-trip"
    );
}

#[test]
fn read_label_decodes_byte4_as_vint() {
    // Includes values spanning `vint`'s 1-, 2-, 3-, 4-, and 5-byte
    // encodings (7, 14, and 21+ significant bits), plus the maximum
    // valid Unicode code point (0x10FFFF, comfortably within `i32`'s
    // range but well past a single byte or even `BYTE2`'s 0xFFFF
    // ceiling).
    let labels = [0x41i32, 0x1f600, 0x10ffff, 0];
    let (bytes, start) =
        build_single_label_seq_fst(&labels, b"v", InputType::Byte4, VERSION_CURRENT);
    let fst = fst_from_body_with_version(bytes, start, InputType::Byte4, None, VERSION_CURRENT);
    assert_eq!(
        fst.get_labels(&labels).unwrap(),
        Some(b"v".to_vec()),
        "BYTE4 vint label round-trip"
    );
}

#[test]
fn get_labels_returns_none_for_absent_label_sequence() {
    let labels = [0x1f600i32, 0x1f601];
    let (bytes, start) =
        build_single_label_seq_fst(&labels, b"v", InputType::Byte4, VERSION_CURRENT);
    let fst = fst_from_body_with_version(bytes, start, InputType::Byte4, None, VERSION_CURRENT);
    assert_eq!(fst.get_labels(&[0x1f600, 0x99]).unwrap(), None);
    assert_eq!(fst.get_labels(&[0x1f600]).unwrap(), None);
}

#[test]
fn get_delegates_to_get_labels_for_byte1() {
    let (bytes, start) = build_single_key_fst(b"cat", b"1");
    let fst = fst_from_body(bytes, start, InputType::Byte1, None);
    let labels: Vec<i32> = b"cat".iter().map(|&b| b as i32).collect();
    assert_eq!(fst.get(b"cat").unwrap(), fst.get_labels(&labels).unwrap());
    assert_eq!(
        fst.seek_exact(b"cat").unwrap(),
        fst.seek_exact_labels(&labels).unwrap()
    );
}

#[test]
fn iter_labels_walks_byte1_fst_same_as_iter() {
    let (bytes, start) = build_single_key_fst(b"cat", b"1");
    let fst = fst_from_body(bytes, start, InputType::Byte1, None);

    let via_iter: Vec<(Vec<u8>, Vec<u8>)> = fst.iter().unwrap().collect::<Result<_>>().unwrap();

    let mut enumerator = fst.iter_labels();
    let mut via_iter_labels = Vec::new();
    while let Some(pair) = enumerator.next_labels() {
        let (key, output) = pair.unwrap();
        via_iter_labels.push((
            key.into_iter().map(|l| l as u8).collect::<Vec<u8>>(),
            output,
        ));
    }

    assert_eq!(via_iter, via_iter_labels);
}

#[test]
fn single_key_found_with_output() {
    let (bytes, start) = build_single_key_fst(b"cat", b"1");
    let fst = fst_from_body(bytes, start, InputType::Byte1, None);
    assert_eq!(fst.get(b"cat").unwrap(), Some(b"1".to_vec()));
}

#[test]
fn single_key_not_present_wrong_label() {
    let (bytes, start) = build_single_key_fst(b"cat", b"1");
    let fst = fst_from_body(bytes, start, InputType::Byte1, None);
    assert_eq!(fst.get(b"dog").unwrap(), None);
}

#[test]
fn single_key_not_present_prefix_of_key() {
    let (bytes, start) = build_single_key_fst(b"cat", b"1");
    let fst = fst_from_body(bytes, start, InputType::Byte1, None);
    // "ca" is a proper prefix that isn't itself accepted.
    assert_eq!(fst.get(b"ca").unwrap(), None);
}

#[test]
fn single_key_not_present_extends_past_key() {
    let (bytes, start) = build_single_key_fst(b"cat", b"1");
    let fst = fst_from_body(bytes, start, InputType::Byte1, None);
    // "cats" walks past the accepting node: the last arc is BIT_STOP_NODE
    // with no outgoing arcs, so target_has_arcs is false.
    assert_eq!(fst.get(b"cats").unwrap(), None);
}

#[test]
fn empty_key_accepted_via_empty_output() {
    let (bytes, start) = build_single_key_fst(b"x", b"ignored");
    let fst = fst_from_body(bytes, start, InputType::Byte1, Some(b"empty-out".to_vec()));
    assert_eq!(fst.get(b"").unwrap(), Some(b"empty-out".to_vec()));
}

#[test]
fn empty_key_rejected_when_fst_does_not_accept_it() {
    let (bytes, start) = build_single_key_fst(b"x", b"1");
    let fst = fst_from_body(bytes, start, InputType::Byte1, None);
    assert_eq!(fst.get(b"").unwrap(), None);
}

#[test]
fn empty_output_with_zero_length_bytes_still_accepts_empty_key() {
    let (bytes, start) = build_single_key_fst(b"x", b"1");
    // emptyOutput != None but zero-length: still "accepts empty string",
    // just with a NO_OUTPUT (empty) final output -- mirrors
    // `getFirstArc`'s `emptyOutput != NO_OUTPUT` check.
    let fst = fst_from_body(bytes, start, InputType::Byte1, Some(Vec::new()));
    assert_eq!(fst.get(b"").unwrap(), Some(Vec::new()));
}

#[test]
fn non_byte1_input_type_is_unsupported_for_get() {
    let (bytes, start) = build_single_key_fst(b"x", b"1");
    let fst = fst_from_body(bytes, start, InputType::Byte2, None);
    assert!(matches!(fst.get(b"x"), Err(Error::Unsupported(_))));
}

// --- `ARCS_FOR_DIRECT_ADDRESSING` fixed-length-arc nodes ---------------
//
// Hand-built (not real-Lucene-written) node in this encoding, to pin down
// `find_target_arc_direct_addressing`/`bit_table_*`'s byte-level field
// layout before/alongside the real-fixture differential test in
// `tests/fst_direct_addressing_fixtures.rs`.

/// Builds a single root node spanning the label range
/// `[first_label, first_label + num_arcs)`, encoded as
/// `ARCS_FOR_DIRECT_ADDRESSING`: a presence bit-table (one bit per label
/// in the range) followed by `first_label`, then one fixed-size arc slot
/// per *present* label only (ascending order), each a final, accepting
/// stop node whose 1-byte output is given by `present`. `present`'s
/// labels must all fall within the range and need not be pre-sorted.
/// Returns `(body_bytes, start_node_address)`.
fn build_direct_addressing_node(
    first_label: u8,
    num_arcs: i32,
    present: &[(u8, u8)],
) -> (Vec<u8>, i64) {
    let num_presence_bytes = num_presence_bytes(num_arcs) as usize;
    let mut presence = vec![0u8; num_presence_bytes];
    for &(label, _) in present {
        let idx = (label - first_label) as usize;
        presence[idx / 8] |= 1 << (idx % 8);
    }

    let mut sorted_present = present.to_vec();
    sorted_present.sort_by_key(|(l, _)| *l);
    let num_present = sorted_present.len();

    let mut logical_arcs: Vec<Vec<u8>> = Vec::with_capacity(num_present);
    for (i, &(_label, output)) in sorted_present.iter().enumerate() {
        let mut logical = Vec::new();
        let mut flags = BIT_FINAL_ARC | BIT_STOP_NODE | BIT_ARC_HAS_FINAL_OUTPUT;
        if i == num_present - 1 {
            flags |= BIT_LAST_ARC;
        }
        logical.push(flags);
        // No label byte: direct-addressing arcs never store the label
        // explicitly, it's implied by position in the range.
        write_vint(&mut logical, 1);
        logical.push(output);
        logical_arcs.push(logical);
    }
    let bytes_per_arc = logical_arcs.iter().map(|a| a.len()).max().unwrap_or(0);

    let mut logical = vec![ARCS_FOR_DIRECT_ADDRESSING];
    write_vint(&mut logical, num_arcs);
    write_vint(&mut logical, bytes_per_arc as i32);
    logical.extend_from_slice(&presence);
    logical.push(first_label);
    for arc in &logical_arcs {
        let mut padded = arc.clone();
        padded.resize(bytes_per_arc, 0u8);
        logical.extend_from_slice(&padded);
    }

    let mut bytes = Vec::new();
    let node_addr = append_arc_logical(&mut bytes, &logical);
    (bytes, node_addr)
}

/// A node header's `numArcs`/`bytesPerArc` are plain `vint`s off disk and
/// `FST.java` trusts both. `numArcs <= 0` then makes `numArcs - 1` and
/// `numArcs - 2` underflow (a debug panic; in release, a negative
/// midpoint that binary-searches the wrong side of the array), and
/// `bytesPerArc == 0` makes every slot address collapse onto one arc.
/// Neither is a header `FSTCompiler` can emit.
#[test]
fn an_array_node_header_with_no_arcs_or_zero_width_slots_is_rejected() {
    for (num_arcs, bpa) in [(0i32, 4i32), (-7, 4), (5, 0), (5, -3)] {
        let mut logical = vec![ARCS_FOR_BINARY_SEARCH];
        write_vint(&mut logical, num_arcs);
        write_vint(&mut logical, bpa);
        logical.extend_from_slice(&[0u8; 64]);
        let mut bytes = Vec::new();
        let addr = append_arc_logical(&mut bytes, &logical);
        let fst = fst_from_body(bytes, addr, InputType::Byte1, None);
        assert!(
            matches!(fst.get(b"a"), Err(Error::Corrupt(_))),
            "numArcs={num_arcs} bytesPerArc={bpa}"
        );
    }
}

/// A fixed-length arc array of `numArcs` slots physically precedes
/// `posArcsStart` in the body, so a header naming more than the bytes
/// below it cannot describe a real node. Without the check the reverse
/// cursor walks to a negative position on every probe -- caught by
/// `read_byte`, but only after the binary search has been driven entirely
/// by garbage.
#[test]
fn an_arc_array_larger_than_the_body_below_it_is_rejected() {
    let mut logical = vec![ARCS_FOR_BINARY_SEARCH];
    write_vint(&mut logical, 1_000_000);
    write_vint(&mut logical, 64);
    let mut bytes = Vec::new();
    let addr = append_arc_logical(&mut bytes, &logical);
    let fst = fst_from_body(bytes, addr, InputType::Byte1, None);
    assert!(matches!(fst.get(b"a"), Err(Error::Corrupt(_))));
}

/// A direct-addressing or continuous node stores no labels: `read_arc`
/// derives each one as `firstLabel + arcIdx`. A `BYTE1` FST whose label
/// range runs past 255 therefore produces a label no byte key can hold --
/// which `labels_to_bytes` must not narrow with a bare `as u8` (260 would
/// become 4, a plausible wrong key in release builds).
#[test]
fn a_label_range_outside_the_input_types_alphabet_is_rejected() {
    // `firstLabel = 250`, a 200-wide range: labels up to 449.
    let (bytes, addr) = build_direct_addressing_node(250, 200, &[(250, 7), (255, 9)]);
    let fst = fst_from_body(bytes, addr, InputType::Byte1, None);
    let err = fst.get(&[250]).unwrap_err();
    assert!(
        format!("{err}").contains("alphabet"),
        "expected the label range to be rejected, got {err}"
    );
}

/// The bit table is rounded up to whole bytes, so its last byte carries up
/// to seven bits past the label range. `FSTCompiler` writes those as zero;
/// a set one would otherwise be returned as an `arcIdx` outside
/// `0..numArcs`, i.e. a label outside the range the header declared.
#[test]
fn a_presence_bit_set_past_the_label_range_is_not_returned_as_an_arc() {
    // Range of 3 labels ('a'..='c') in a one-byte table; bit 5 is padding.
    let (mut bytes, addr) = build_direct_addressing_node(b'a', 3, &[(b'a', 1), (b'c', 2)]);
    // The presence byte is the one holding bits 0 and 2 (value 0b101).
    let at = bytes
        .iter()
        .position(|&b| b == 0b0000_0101)
        .expect("presence byte");
    bytes[at] |= 1 << 5;
    let fst = fst_from_body(bytes, addr, InputType::Byte1, None);
    // Enumeration must stop at the real arcs rather than walking into the
    // padding bit and reporting a fourth, non-existent label.
    let keys: Vec<Vec<u8>> = fst.iter().unwrap().map(|r| r.unwrap().0).collect();
    assert_eq!(keys, vec![b"a".to_vec(), b"c".to_vec()]);
}

/// `BitTableUtil.nextBitSet` returns `-1` for "no next set bit", and `-1`
/// is as far outside `0..numArcs` as an index past the end is. Passed on
/// as an `arcIdx` it makes `read_arc` derive `firstLabel - 1`: an arc one
/// label *below* the range the node declared, and for `firstLabel == 0`
/// exactly `END_LABEL`, i.e. a spurious acceptance of the empty
/// continuation. Nothing downstream catches it -- `presence_index` goes
/// `-1 -> 0`, so the slot address is the perfectly valid `pos_arcs_start`.
///
/// `firstLabel == 0` is the case that makes the consequence loudest, and
/// it is why the byte-flip sweep could not find this: every FST fixture's
/// `firstLabel` is an ASCII letter, and the sweep only asserts "a typed
/// error or a clean decode", which `firstLabel - 1` is.
#[test]
fn an_all_zero_presence_table_does_not_decode_an_arc_below_the_label_range() {
    let (mut bytes, addr) = build_direct_addressing_node(0, 3, &[(0, 1), (2, 2)]);
    let at = bytes
        .iter()
        .position(|&b| b == 0b0000_0101)
        .expect("presence byte");
    bytes[at] = 0;
    let fst = fst_from_body(bytes, addr, InputType::Byte1, None);
    let mut it = fst.iter().unwrap();
    assert!(matches!(it.next(), Some(Err(Error::Corrupt(_)))));
}

/// The **third** call site of the same out-of-domain sentinel.
/// `seek_floor_direct_addressing`'s in-range-but-absent branch
/// passes `bit_table_previous_bit_set`'s result straight into
/// `read_arc_by_direct_addressing`, where Java has only
/// `assert floorIndex != -1;` -- off in production. A presence table whose
/// bit 0 is clear makes it `-1`, and with `first_label == 0` the arc that
/// decodes carries label `-1`, which **is `END_LABEL`**: a plausible,
/// perfectly well-formed wrong answer, which is why a byte-flip sweep
/// walks past it. `first_label == 0` is deliberate here -- every FST
/// fixture in the tree starts at an ASCII letter, where the same bug is
/// merely an off-by-one label and nothing looks wrong.
#[test]
fn a_floor_seek_into_a_gap_with_no_arc_below_it_is_rejected() {
    let (mut bytes, addr) = build_direct_addressing_node(0, 3, &[(0, 1), (2, 2)]);
    let at = bytes
        .iter()
        .position(|&b| b == 0b0000_0101)
        .expect("presence byte");
    // Clear bit 0 only: label 2 is still present, so the node is not
    // empty and the seek reaches the in-range branch rather than the
    // last-arc one the test below covers.
    bytes[at] = 0b0000_0100;
    let fst = fst_from_body(bytes, addr, InputType::Byte1, None);
    let mut it = fst.iter_labels();
    assert!(matches!(it.seek_floor_labels(&[1]), Err(Error::Corrupt(_))));
}

/// The same out-of-domain sentinel through `readLastArcByDirectAddressing`:
/// `countBits - 1` is `-1` for an all-zero table, and a `-1`
/// `presence_index` addresses *above* the arc array, which `read_byte`
/// accepts.
#[test]
fn an_all_zero_presence_table_is_rejected_by_the_last_arc_path_too() {
    let (mut bytes, addr) = build_direct_addressing_node(b'a', 3, &[(b'a', 1), (b'c', 2)]);
    let at = bytes
        .iter()
        .position(|&b| b == 0b0000_0101)
        .expect("presence byte");
    bytes[at] = 0;
    let fst = fst_from_body(bytes, addr, InputType::Byte1, None);
    let mut it = fst.iter_labels();
    assert!(matches!(
        it.seek_floor_labels(&[b'z' as i32]),
        Err(Error::Corrupt(_))
    ));
}

/// An FST body is written bottom-up, so every arc target precedes the node
/// that holds the arc. A body whose targets do not strictly decrease is
/// cyclic, and the enum's descent then never terminates: it grows `arcs`,
/// `outputs` and `labels` one level per step until the OOM killer stops
/// the process. The byte-flip sweep found it as a SIGKILL.
#[test]
fn a_cyclic_arc_target_is_rejected_rather_than_descended_forever() {
    // One node, one arc labelled 'a', whose explicit target is the node
    // itself.
    let mut logical = vec![BIT_LAST_ARC];
    logical.push(b'a');
    // The node's own address is `logical.len() - 1` once appended.
    write_vlong(&mut logical, 0);
    let self_addr = (logical.len() - 1) as i64;
    let mut logical = vec![BIT_LAST_ARC];
    logical.push(b'a');
    write_vlong(&mut logical, self_addr);
    let mut bytes = Vec::new();
    let addr = append_arc_logical(&mut bytes, &logical);
    assert_eq!(addr, self_addr, "the arc must point at its own node");
    let fst = fst_from_body(bytes, addr, InputType::Byte1, None);
    let mut it = fst.iter().unwrap();
    assert!(matches!(it.next(), Some(Err(Error::Corrupt(_)))));
}

/// `FST.readLabel`'s `BYTE4` branch is a bare `readVInt()`, which decodes
/// to any `i32`. Every label a writer emits is non-negative; a negative
/// one makes `targetLabel - firstLabel` and `firstLabel + arcIdx`
/// unbounded subtractions on both ends.
#[test]
fn a_negative_byte4_label_is_rejected() {
    let mut logical = vec![BIT_LAST_ARC | BIT_FINAL_ARC | BIT_STOP_NODE];
    write_vint(&mut logical, -5);
    let mut bytes = Vec::new();
    let addr = append_arc_logical(&mut bytes, &logical);
    let fst = fst_from_body(bytes, addr, InputType::Byte4, None);
    assert!(matches!(fst.get_labels(&[5]), Err(Error::Corrupt(_))));
}

#[test]
fn direct_addressing_node_finds_every_present_label() {
    let first_label = b'a';
    let num_arcs = (b'z' - b'a' + 1) as i32;
    let present = [
        (b'a', 1u8),
        (b'c', 2u8),
        (b'f', 3u8),
        (b'm', 4u8),
        (b'z', 5u8),
    ];
    let (bytes, start) = build_direct_addressing_node(first_label, num_arcs, &present);
    let fst = fst_from_body(bytes, start, InputType::Byte1, None);
    for (label, output) in present {
        assert_eq!(
            fst.get(&[label]).unwrap(),
            Some(vec![output]),
            "label {label} should resolve to {output}"
        );
    }
}

#[test]
fn direct_addressing_node_rejects_absent_labels_in_and_around_range() {
    let first_label = b'a';
    let num_arcs = (b'z' - b'a' + 1) as i32;
    let present = [
        (b'a', 1u8),
        (b'c', 2u8),
        (b'f', 3u8),
        (b'm', 4u8),
        (b'z', 5u8),
    ];
    let (bytes, start) = build_direct_addressing_node(first_label, num_arcs, &present);
    let fst = fst_from_body(bytes, start, InputType::Byte1, None);
    // In-range but not present (bit clear), plus strictly before/after
    // the label range entirely.
    for absent in [b'b', b'd', b'g', b'n', b'y', 0u8, 0xffu8] {
        assert_eq!(
            fst.get(&[absent]).unwrap(),
            None,
            "label {absent} should be absent"
        );
    }
}

#[test]
fn direct_addressing_node_enumerates_every_present_label_in_ascending_order() {
    let first_label = b'a';
    let num_arcs = (b'z' - b'a' + 1) as i32;
    let present = [
        (b'm', 4u8),
        (b'a', 1u8),
        (b'z', 5u8),
        (b'c', 2u8),
        (b'f', 3u8),
    ];
    let (bytes, start) = build_direct_addressing_node(first_label, num_arcs, &present);
    let fst = fst_from_body(bytes, start, InputType::Byte1, None);
    let mut expected: Vec<(Vec<u8>, Vec<u8>)> =
        present.iter().map(|&(l, o)| (vec![l], vec![o])).collect();
    expected.sort();
    let got: Vec<(Vec<u8>, Vec<u8>)> = fst.iter().unwrap().collect::<Result<_>>().unwrap();
    assert_eq!(got, expected);
}

// --- `ARCS_FOR_CONTINUOUS` fixed-length-arc nodes ----------------------
//
// Hand-built (not real-Lucene-written) node in this encoding, to pin down
// `find_target_arc_continuous`'s byte-level field layout before/alongside
// the real-fixture differential test in
// `tests/fst_continuous_fixtures.rs`.

/// Builds a single root node spanning the label range
/// `[first_label, first_label + labels.len())`, encoded as
/// `ARCS_FOR_CONTINUOUS`: no presence bit-table and no per-arc label byte
/// at all (every label in the range is present by construction), just
/// `first_label` followed by one fixed-size arc slot per label in
/// ascending order, each a final, accepting stop node whose 1-byte output
/// is `outputs[i]`. Returns `(body_bytes, start_node_address)`.
fn build_continuous_node(first_label: u8, outputs: &[u8]) -> (Vec<u8>, i64) {
    let num_arcs = outputs.len();
    let mut logical_arcs: Vec<Vec<u8>> = Vec::with_capacity(num_arcs);
    for (i, &output) in outputs.iter().enumerate() {
        let mut logical = Vec::new();
        let mut flags = BIT_FINAL_ARC | BIT_STOP_NODE | BIT_ARC_HAS_FINAL_OUTPUT;
        if i == num_arcs - 1 {
            flags |= BIT_LAST_ARC;
        }
        logical.push(flags);
        // No label byte: continuous arcs never store the label
        // explicitly, it's implied by position in the range.
        write_vint(&mut logical, 1);
        logical.push(output);
        logical_arcs.push(logical);
    }
    let bytes_per_arc = logical_arcs.iter().map(|a| a.len()).max().unwrap_or(0);

    let mut logical = vec![ARCS_FOR_CONTINUOUS];
    write_vint(&mut logical, num_arcs as i32);
    write_vint(&mut logical, bytes_per_arc as i32);
    logical.push(first_label);
    for arc in &logical_arcs {
        let mut padded = arc.clone();
        padded.resize(bytes_per_arc, 0u8);
        logical.extend_from_slice(&padded);
    }

    let mut bytes = Vec::new();
    let node_addr = append_arc_logical(&mut bytes, &logical);
    (bytes, node_addr)
}

#[test]
fn continuous_node_finds_every_present_label() {
    let first_label = b'a';
    let outputs = [1u8, 2, 3, 4, 5];
    let (bytes, start) = build_continuous_node(first_label, &outputs);
    let fst = fst_from_body(bytes, start, InputType::Byte1, None);
    for (i, &output) in outputs.iter().enumerate() {
        let label = first_label + i as u8;
        assert_eq!(
            fst.get(&[label]).unwrap(),
            Some(vec![output]),
            "label {label} should resolve to {output}"
        );
    }
}

#[test]
fn continuous_node_rejects_labels_outside_the_range() {
    let first_label = b'a';
    let outputs = [1u8, 2, 3, 4, 5]; // Covers 'a'..='e'.
    let (bytes, start) = build_continuous_node(first_label, &outputs);
    let fst = fst_from_body(bytes, start, InputType::Byte1, None);
    for absent in [b'`', b'f', b'z', 0u8, 0xffu8] {
        assert_eq!(
            fst.get(&[absent]).unwrap(),
            None,
            "label {absent} should be absent"
        );
    }
}

#[test]
fn continuous_node_enumerates_every_label_in_ascending_order() {
    let first_label = b'a';
    let outputs = [1u8, 2, 3, 4, 5];
    let (bytes, start) = build_continuous_node(first_label, &outputs);
    let fst = fst_from_body(bytes, start, InputType::Byte1, None);
    let expected: Vec<(Vec<u8>, Vec<u8>)> = outputs
        .iter()
        .enumerate()
        .map(|(i, &o)| (vec![first_label + i as u8], vec![o]))
        .collect();
    let got: Vec<(Vec<u8>, Vec<u8>)> = fst.iter().unwrap().collect::<Result<_>>().unwrap();
    assert_eq!(got, expected);
}

// --- `ARCS_FOR_BINARY_SEARCH` fixed-length-arc nodes ------------------
//
// Hand-built (not real-Lucene-written) node in this encoding, to pin
// down `find_target_arc_binary_search`'s byte-level field layout and
// binary-search bounds before/alongside the real-fixture differential
// test in `tests/fst_fixtures.rs`.

/// Builds a single root node with `labels.len()` arcs (ascending,
/// distinct byte labels) encoded as `ARCS_FOR_BINARY_SEARCH`: each arc is
/// a fixed-size slot of `bytes_per_arc` bytes (padded with trailing zero
/// bytes after its real content -- real Lucene pads the same way so
/// every slot can be addressed via `posArcsStart - bytesPerArc * idx`).
/// Every arc is a final, accepting stop node whose 1-byte output is
/// `outputs[i]`. Returns `(body_bytes, start_node_address)`.
fn build_binary_search_node(labels: &[u8], outputs: &[u8]) -> (Vec<u8>, i64) {
    assert_eq!(labels.len(), outputs.len());
    let num_arcs = labels.len();

    // Build each arc's logical (forward-read) bytes first, to find the
    // max length -> bytes_per_arc (every slot is padded to this size).
    let mut logical_arcs: Vec<Vec<u8>> = Vec::with_capacity(num_arcs);
    for (i, &label) in labels.iter().enumerate() {
        let mut logical = Vec::new();
        let mut flags = BIT_FINAL_ARC | BIT_STOP_NODE | BIT_ARC_HAS_FINAL_OUTPUT;
        if i == num_arcs - 1 {
            // Only the highest-labeled arc (the last slot, at the
            // lowest address) is actually the node's last arc --
            // `is_last()` must be false for every other slot so
            // enumeration (`FstEnum`) knows to keep advancing through
            // the node instead of stopping after the first arc it
            // reads. (Real Lucene-written binary-search nodes get this
            // right by construction; `find_target_arc`'s direct
            // binary-search jump would not notice the bit on every slot,
            // but `FstEnum`'s incremental walk does.)
            flags |= BIT_LAST_ARC;
        }
        logical.push(flags);
        logical.push(label);
        write_vint(&mut logical, 1);
        logical.push(outputs[i]);
        logical_arcs.push(logical);
    }
    let bytes_per_arc = logical_arcs.iter().map(|a| a.len()).max().unwrap();

    // Build the *entire* node (header + every fixed-size slot) as one
    // logical (forward-read-order) byte sequence, then push it via
    // `append_arc_logical`'s single whole-blob reversal -- this keeps
    // every field's relative addressing (header bytes, then slot 0,
    // slot 1, ...) consistent with the reverse `BytesReader`'s
    // decreasing-position reads, exactly the way a real node's bytes
    // are laid out. `append_arc_logical` returns the flags byte's
    // address (`node_addr`), matching what `find_target_arc` seeks to.
    let mut logical = vec![ARCS_FOR_BINARY_SEARCH];
    write_vint(&mut logical, num_arcs as i32);
    write_vint(&mut logical, bytes_per_arc as i32);
    for arc in &logical_arcs {
        let mut padded = arc.clone();
        padded.resize(bytes_per_arc, 0u8);
        logical.extend_from_slice(&padded);
    }

    let mut bytes = Vec::new();
    let node_addr = append_arc_logical(&mut bytes, &logical);
    (bytes, node_addr)
}

#[test]
fn binary_search_node_finds_every_label() {
    let labels = *b"acfmz";
    let outputs = [1u8, 2, 3, 4, 5];
    let (bytes, start) = build_binary_search_node(&labels, &outputs);
    let fst = fst_from_body(bytes, start, InputType::Byte1, None);
    for (label, output) in labels.iter().zip(outputs.iter()) {
        assert_eq!(
            fst.get(&[*label]).unwrap(),
            Some(vec![*output]),
            "label {label} should resolve to {output}"
        );
    }
}

#[test]
fn binary_search_node_rejects_absent_labels() {
    let labels = *b"acfmz";
    let outputs = [1u8, 2, 3, 4, 5];
    let (bytes, start) = build_binary_search_node(&labels, &outputs);
    let fst = fst_from_body(bytes, start, InputType::Byte1, None);
    // Before the first label, between two labels, and after the last.
    for absent in [b'0', b'b', b'd', b'g', b'n', 0xffu8] {
        assert_eq!(fst.get(&[absent]).unwrap(), None);
    }
}

#[test]
fn read_rejects_bad_input_type_byte() {
    let mut meta = Vec::new();
    codec_util::write_header(&mut meta, FILE_FORMAT_NAME, VERSION_CURRENT);
    meta.push(0); // no empty output
    meta.push(9); // invalid input type marker
    let mut input = SliceInput::new(&meta);
    assert!(matches!(Fst::read(&mut input), Err(Error::Corrupt(_))));
}

#[test]
fn read_rejects_wrong_codec_header() {
    let mut meta = Vec::new();
    codec_util::write_header(&mut meta, "NotFST", VERSION_CURRENT);
    let mut input = SliceInput::new(&meta);
    assert!(matches!(Fst::read(&mut input), Err(Error::Store(_))));
}

#[test]
fn read_metadata_round_trips_through_real_header_and_body() {
    // Full end-to-end `Fst::read` (metadata + body via one cursor),
    // independent of the real-Lucene differential fixtures: hand-builds
    // the exact byte layout `FSTMetadata.save` produces for a
    // no-empty-output, BYTE1, single-key FST.
    let (body, start_node) = build_single_key_fst(b"ab", b"42");

    let mut file = Vec::new();
    codec_util::write_header(&mut file, FILE_FORMAT_NAME, VERSION_CURRENT);
    file.push(0); // no empty output
    file.push(0); // INPUT_TYPE.BYTE1
    write_vlong(&mut file, start_node);
    write_vlong(&mut file, body.len() as i64);
    file.extend_from_slice(&body);

    let mut input = SliceInput::new(&file);
    let fst = Fst::read(&mut input).unwrap();
    assert_eq!(fst.metadata().input_type, InputType::Byte1);
    assert_eq!(fst.metadata().start_node, start_node);
    assert_eq!(fst.metadata().num_bytes, body.len() as i64);
    assert_eq!(fst.get(b"ab").unwrap(), Some(b"42".to_vec()));
    assert_eq!(fst.get(b"ac").unwrap(), None);
}

#[test]
fn read_rejects_truncated_body() {
    // A numBytes claiming more body than actually follows must fail, not
    // silently succeed with a short/garbage body. It is rejected against
    // the bytes actually left in the file *before* the body `Vec` is
    // sized -- see `read_rejects_a_body_length_no_allocation_could_hold`
    // for why that ordering is the point.
    let mut file = Vec::new();
    codec_util::write_header(&mut file, FILE_FORMAT_NAME, VERSION_CURRENT);
    file.push(0);
    file.push(0);
    write_vlong(&mut file, 0);
    write_vlong(&mut file, 1000); // claims 1000 body bytes, none follow
    let mut input = SliceInput::new(&file);
    assert!(matches!(Fst::read(&mut input), Err(Error::Corrupt(_))));
}

#[test]
fn read_rejects_a_body_length_no_allocation_could_hold() {
    // `numBytes` is a vlong off disk that sizes `vec![0u8; numBytes]`.
    // Unbounded, this is not a recoverable error: the allocation fails
    // and the process *aborts*, which `catch_unwind` cannot intercept and
    // which therefore takes the embedding JVM down with it. Before the
    // bound this reserved ~4 EB; it must now be a typed error.
    let mut file = Vec::new();
    codec_util::write_header(&mut file, FILE_FORMAT_NAME, VERSION_CURRENT);
    file.push(0); // no empty output
    file.push(0); // BYTE1
    write_vlong(&mut file, 0); // startNode
    write_vlong(&mut file, i64::MAX / 2);
    let mut input = SliceInput::new(&file);
    assert!(matches!(Fst::read(&mut input), Err(Error::Corrupt(_))));
}

#[test]
fn read_rejects_an_empty_output_length_no_allocation_could_hold() {
    // Same shape one field earlier: the accepts-empty flag's own vint
    // length sized a `vec![0u8; n]` with nothing bounding it but the
    // `read_bytes` that came after the allocation.
    let mut file = Vec::new();
    codec_util::write_header(&mut file, FILE_FORMAT_NAME, VERSION_CURRENT);
    file.push(1); // accepts the empty string
    write_vint(&mut file, i32::MAX); // ... with a 2 GB output
    let mut input = SliceInput::new(&file);
    assert!(matches!(Fst::read(&mut input), Err(Error::Corrupt(_))));
}

#[test]
fn arc_output_length_is_bounded_by_the_body_below_the_cursor() {
    // `ByteSequenceOutputs.read`'s `int len = in.readVInt()` sizing
    // `vec![0u8; len]`: a negative length sign-extends through `as usize`
    // to ~2^64 and aborts, and a large positive one reserves 2 GB. Both
    // must be a decode error instead.
    for len in [-1i32, i32::MAX] {
        let mut encoded = Vec::new();
        write_vint(&mut encoded, len);
        encoded.reverse();
        let mut r = BytesReader::new(&encoded);
        r.set_position(encoded.len() as i64 - 1);
        assert!(
            matches!(r.read_output(), Err(Error::Corrupt(_))),
            "len={len}"
        );
    }
    // A length that does fit below the cursor still decodes.
    let mut encoded = vec![b'x'; 3];
    write_vint(&mut encoded, 3);
    let mut r = BytesReader::new(&encoded);
    r.set_position(encoded.len() as i64 - 1);
    assert_eq!(r.read_output().unwrap(), vec![b'x'; 3]);
}

#[test]
fn a_vint_longer_than_five_bytes_is_rejected_not_silently_truncated() {
    // `DataInput.readVInt`'s "Invalid vInt detected (too many bits)".
    // Without the bound the loop keeps consuming continuation bytes,
    // returning a value built from bits Java rejects outright -- and the
    // `shift` accumulator itself overflows on a long enough run.
    let encoded = vec![0x80u8; 32];
    let mut r = BytesReader::new(&encoded);
    r.set_position(encoded.len() as i64 - 1);
    assert!(matches!(r.read_vint(), Err(Error::Corrupt(_))));
    let mut r = BytesReader::new(&encoded);
    r.set_position(encoded.len() as i64 - 1);
    assert!(matches!(r.read_vlong(), Err(Error::Corrupt(_))));
}

#[test]
fn typed_output_decoders_reject_a_corrupt_payload_instead_of_panicking() {
    // `PositiveIntOutputs::decode`/`PairOutputs::decode` read a payload
    // that came off the FST body, so a truncated or corrupt one reaches
    // them. Indexing the slice directly (`bytes[idx]`) or shifting by an
    // unbounded amount would panic.
    assert!(PositiveIntOutputs::decode(&[0x80]).is_err());
    assert!(PositiveIntOutputs::decode(&[0x80; 32]).is_err());
    type Pairs = PairOutputs<PositiveIntOutputs, ByteSequenceOutputs>;
    // A first-component length past the end of the payload.
    assert!(Pairs::decode(&[9, 1, 2]).is_err());
    // ... and a negative one, which `as usize` turned into ~2^64.
    let mut bytes = Vec::new();
    write_vint(&mut bytes, -1);
    bytes.push(7);
    assert!(Pairs::decode(&bytes).is_err());
}

#[test]
fn read_borrowed_rejects_truncated_body_same_as_read() {
    // Same malformed bytes as `read_rejects_truncated_body`, through
    // `read_borrowed` -- `SliceInput::slice`'s bounds check must reject
    // this the same way `read_bytes` does for the owned path, not
    // silently return a short/garbage slice.
    let mut file = Vec::new();
    codec_util::write_header(&mut file, FILE_FORMAT_NAME, VERSION_CURRENT);
    file.push(0);
    file.push(0);
    write_vlong(&mut file, 0);
    write_vlong(&mut file, 1000); // claims 1000 body bytes, none follow
    let mut input = SliceInput::new(&file);
    assert!(matches!(
        Fst::read_borrowed(&mut input),
        Err(Error::Store(_))
    ));
}

#[test]
fn read_rejects_negative_num_bytes() {
    // `readVLong`'s bit-shifting can decode a hand-crafted (never
    // written by a real, well-behaved encoder) byte sequence as a
    // negative `i64`; `Fst::read` must reject that outright via its own
    // explicit guard rather than trying `vec![0u8; num_bytes as usize]`
    // with a negative-cast-to-usize length.
    let mut file = Vec::new();
    codec_util::write_header(&mut file, FILE_FORMAT_NAME, VERSION_CURRENT);
    file.push(0);
    file.push(0);
    write_vlong(&mut file, 0);
    write_vlong(&mut file, -1); // decodes to a negative numBytes
    let mut input = SliceInput::new(&file);
    assert!(matches!(Fst::read(&mut input), Err(Error::Corrupt(_))));
}

#[test]
fn read_accepts_empty_string_via_empty_output() {
    // Full `Fst::read` path for the `acceptsEmpty == 1` branch. The
    // empty-string output is `writeFinalOutput`-serialized first
    // (`vint(len)` + payload), *then* the whole serialized buffer is
    // reversed, and *that* is length-prefixed -- see
    // `read_fst_metadata_prefix`. Byte-for-byte identical to what
    // `fixtures/data/fst_empty_key` (a real Lucene `FST.save`) contains;
    // this test pins the layout in isolation, the fixture proves it
    // against Lucene itself.
    let (body, start_node) = build_single_key_fst(b"x", b"ignored");
    let empty_output = b"root-output".to_vec();
    let mut reversed = Vec::new();
    write_vint(&mut reversed, empty_output.len() as i32);
    reversed.extend_from_slice(&empty_output);
    reversed.reverse();

    let mut file = Vec::new();
    codec_util::write_header(&mut file, FILE_FORMAT_NAME, VERSION_CURRENT);
    file.push(1); // accepts empty string
    write_vint(&mut file, reversed.len() as i32);
    file.extend_from_slice(&reversed);
    file.push(0); // INPUT_TYPE.BYTE1
    write_vlong(&mut file, start_node);
    write_vlong(&mut file, body.len() as i64);
    file.extend_from_slice(&body);

    let mut input = SliceInput::new(&file);
    let fst = Fst::read(&mut input).unwrap();
    assert_eq!(fst.metadata().empty_output, Some(empty_output.clone()));
    assert_eq!(fst.get(b"").unwrap(), Some(empty_output));
}

#[test]
fn read_parses_byte2_and_byte4_input_types() {
    for (marker, expected) in [(1u8, InputType::Byte2), (2u8, InputType::Byte4)] {
        let mut file = Vec::new();
        codec_util::write_header(&mut file, FILE_FORMAT_NAME, VERSION_CURRENT);
        file.push(0); // no empty output
        file.push(marker);
        write_vlong(&mut file, 0); // startNode
        write_vlong(&mut file, 0); // numBytes: empty body
        let mut input = SliceInput::new(&file);
        let fst = Fst::read(&mut input).unwrap();
        assert_eq!(fst.metadata().input_type, expected);
    }
}

#[test]
fn get_errors_on_out_of_range_start_node() {
    // A start_node address beyond the body's bounds must surface as an
    // error from the reverse byte cursor, not a panic or silent
    // misdecode -- exercises `BytesReader::read_byte`'s own bounds
    // check directly (real Lucene-written FSTs never do this; this is
    // this decoder's own defensive boundary).
    let fst = fst_from_body(vec![1, 2, 3], 100, InputType::Byte1, None);
    assert!(matches!(fst.get(b"a"), Err(Error::Corrupt(_))));
}

#[test]
fn multi_byte_vint_output_length_round_trips() {
    // Forces `read_vint`'s (and the test-only `write_vint`'s) multi-byte
    // continuation branch: an output of 200 bytes needs a 2-byte vint
    // length prefix (127 is the largest 1-byte vint).
    let big_output = vec![7u8; 200];
    let (bytes, start) = build_single_key_fst(b"k", &big_output);
    let fst = fst_from_body(bytes, start, InputType::Byte1, None);
    assert_eq!(fst.get(b"k").unwrap(), Some(big_output));
}

#[test]
fn multi_byte_vlong_target_round_trips() {
    // Forces `read_vlong`'s multi-byte continuation branch for an
    // explicit arc target: a long enough key chain pushes early node
    // addresses well past 127 (the largest 1-byte vlong value).
    let key: Vec<u8> = (0..100u8).collect(); // 100 distinct bytes, deep chain
    let (bytes, start) = build_single_key_fst(&key, b"end");
    let fst = fst_from_body(bytes, start, InputType::Byte1, None);
    assert_eq!(fst.get(&key).unwrap(), Some(b"end".to_vec()));
    assert!(bytes_len_forces_multi_byte_vlong(&fst));
}

/// Sanity check that the previous test's key is actually long enough to
/// push at least one node address past the 1-byte vlong threshold (127)
/// -- guards against the test silently stopping being meaningful if the
/// key length above is ever shortened.
fn bytes_len_forces_multi_byte_vlong(fst: &Fst) -> bool {
    fst.bytes.len() > 127
}

// --- `output_add` (free function) ---------------------------------

#[test]
fn output_add_covers_all_branches() {
    assert_eq!(output_add(&[], &[]), Vec::<u8>::new());
    assert_eq!(output_add(&[], &[1, 2]), vec![1, 2]);
    assert_eq!(output_add(&[1, 2], &[]), vec![1, 2]);
    assert_eq!(output_add(&[1, 2], &[3, 4]), vec![1, 2, 3, 4]);
}

// --- `Arc` accessors -------------------------------------------------

#[test]
fn arc_accessors() {
    let (bytes, start) = build_single_key_fst(b"q", b"9");
    let fst = fst_from_body(bytes, start, InputType::Byte1, None);
    let mut r = fst.reader();
    let first = fst.first_arc();
    let arc = fst
        .find_target_arc(b'q' as i32, &first, &mut r)
        .unwrap()
        .unwrap();
    assert_eq!(arc.label(), b'q' as i32);
    assert_eq!(arc.output(), &[] as &[u8]);
    assert_eq!(arc.target(), FINAL_END_NODE);
    assert!(arc.is_last());
    assert!(arc.is_final());
    assert_eq!(arc.next_final_output(), b"9");
}

// --- Hand-built multi-arc nodes ---------------------------------------
//
// `build_single_key_fst` only ever emits single-arc nodes with explicit
// vlong targets. Real Lucene-written FSTs (see `tests/fst_fixtures.rs`)
// exercise genuine branching, but happened to never use the
// `BIT_TARGET_NEXT` compaction for a *non-last* sibling arc for this
// fixture's particular key set, nor a final-output-bearing arc that gets
// skipped over mid-scan, nor a non-final stop node, nor a
// zero-length-but-flagged output -- so those `read_arc`/
// `seek_to_next_node`/`find_target_arc` branches need bytes built by
// hand here instead.

/// Builds a two-arc root node: `'a'` (not last, `BIT_TARGET_NEXT`,
/// pointing implicitly at a one-arc child node for `'b'`) and `'z'`
/// (last, itself accepting). Exercises `read_arc`'s `BIT_TARGET_NEXT`
/// non-last branch, which calls `seek_to_next_node` to skip past `'z'`
/// before landing on the `'b'`-node's address.
fn build_target_next_branching_fst(b_output: &[u8], z_output: &[u8]) -> (Vec<u8>, i64) {
    let mut bytes = Vec::new();

    // Child node for 'b' (address = addr_b), reached only via 'a's
    // implicit BIT_TARGET_NEXT target.
    let mut node_b = Vec::new();
    let mut flags_b = BIT_LAST_ARC | BIT_FINAL_ARC | BIT_STOP_NODE;
    if !b_output.is_empty() {
        flags_b |= BIT_ARC_HAS_FINAL_OUTPUT;
    }
    node_b.push(flags_b);
    node_b.push(b'b');
    if !b_output.is_empty() {
        write_vint(&mut node_b, b_output.len() as i32);
        node_b.extend_from_slice(b_output);
    }
    let addr_b = append_arc_logical(&mut bytes, &node_b);

    // Root's 'z' arc (appended directly after node_b, so it stays
    // adjacent -- this is what makes 'a's implicit TARGET_NEXT target
    // land exactly on addr_b once 'z' is fully skipped).
    let mut arc_z = Vec::new();
    let mut flags_z = BIT_LAST_ARC | BIT_FINAL_ARC | BIT_STOP_NODE;
    if !z_output.is_empty() {
        flags_z |= BIT_ARC_HAS_FINAL_OUTPUT;
    }
    arc_z.push(flags_z);
    arc_z.push(b'z');
    if !z_output.is_empty() {
        write_vint(&mut arc_z, z_output.len() as i32);
        arc_z.extend_from_slice(z_output);
    }
    append_arc_logical(&mut bytes, &arc_z);

    // Root's 'a' arc: not last, BIT_TARGET_NEXT, no output of its own --
    // its target is implicit (wherever the byte stream lands right
    // after skipping every remaining sibling, i.e. addr_b).
    let arc_a = vec![BIT_TARGET_NEXT, b'a'];
    let addr_a = append_arc_logical(&mut bytes, &arc_a);

    assert!(addr_b < addr_a); // sanity: node_b really is "below" root
    (bytes, addr_a)
}

#[test]
fn target_next_non_last_arc_reaches_child_via_seek_to_next_node() {
    let (bytes, start) = build_target_next_branching_fst(b"B", b"Z");
    let fst = fst_from_body(bytes, start, InputType::Byte1, None);
    assert_eq!(fst.get(b"ab").unwrap(), Some(b"B".to_vec()));
    assert_eq!(fst.get(b"z").unwrap(), Some(b"Z".to_vec()));
    // 'a' alone is a structural-only arc (not final): present in the
    // FST's arcs but not an accepted key.
    assert_eq!(fst.get(b"a").unwrap(), None);
    assert_eq!(fst.get(b"ac").unwrap(), None);
}

#[test]
fn skips_final_output_of_a_non_matching_sibling_arc_during_scan() {
    // Root arcs: 'a' (itself accepting, has a final output, NOT last)
    // and 'z' (last, accepting). Looking up "z" must skip clean over
    // 'a's final-output bytes (`find_target_arc`'s
    // `BIT_ARC_HAS_FINAL_OUTPUT` skip branch) before reaching 'z'.
    let mut bytes = Vec::new();

    let mut arc_z = vec![
        BIT_LAST_ARC | BIT_FINAL_ARC | BIT_STOP_NODE | BIT_ARC_HAS_FINAL_OUTPUT,
        b'z',
    ];
    write_vint(&mut arc_z, 1);
    arc_z.push(b'Z');
    append_arc_logical(&mut bytes, &arc_z);

    let mut arc_a = vec![
        BIT_FINAL_ARC | BIT_ARC_HAS_FINAL_OUTPUT | BIT_STOP_NODE,
        b'a',
    ];
    write_vint(&mut arc_a, 1);
    arc_a.push(b'A');
    let start = append_arc_logical(&mut bytes, &arc_a);

    let fst = fst_from_body(bytes, start, InputType::Byte1, None);
    assert_eq!(fst.get(b"z").unwrap(), Some(b"Z".to_vec()));
    assert_eq!(fst.get(b"a").unwrap(), Some(b"A".to_vec()));
}

#[test]
fn stop_node_without_final_arc_is_a_non_accepting_dead_end() {
    // A `BIT_STOP_NODE` arc without `BIT_FINAL_ARC` represents Lucene's
    // "non-final dead-end state" (`FST.NON_FINAL_END_NODE`): the label
    // matches structurally but the key isn't accepted there.
    let arc = vec![BIT_LAST_ARC | BIT_STOP_NODE, b'q'];
    let mut bytes = Vec::new();
    let start = append_arc_logical(&mut bytes, &arc);
    let fst = fst_from_body(bytes, start, InputType::Byte1, None);
    assert_eq!(fst.get(b"q").unwrap(), None);
}

#[test]
fn zero_length_flagged_output_is_read_as_empty() {
    // BIT_ARC_HAS_OUTPUT set but the stored vint length is 0: a
    // defensive case a real encoder wouldn't produce (it would just
    // clear the flag), but `read_output`'s `len == 0` branch must still
    // decode it as an empty output rather than misreading subsequent
    // bytes as output content.
    let mut bytes = Vec::new();
    let mut arc = vec![
        BIT_LAST_ARC | BIT_FINAL_ARC | BIT_STOP_NODE | BIT_ARC_HAS_OUTPUT,
        b'r',
    ];
    write_vint(&mut arc, 0); // output length 0, despite the flag being set
    let start = append_arc_logical(&mut bytes, &arc);
    let fst = fst_from_body(bytes, start, InputType::Byte1, None);
    assert_eq!(fst.get(b"r").unwrap(), Some(Vec::new()));
}

// --- `build_fst` (simplified FSTCompiler) -----------------------------
//
// These round-trip *through the existing, unmodified* `Fst::read`/
// `Fst::get` -- the whole point of this builder is that its output is
// usable by the reader that was already written and tested above, not
// just "some bytes this module's own code happens to also understand".

fn assert_round_trips(
    entries: &[(Vec<u8>, Vec<u8>)],
    present: &[(&[u8], &[u8])],
    absent: &[&[u8]],
) {
    let fst = build_fst(entries).unwrap();
    let file = write_fst(&fst);
    let mut input = SliceInput::new(&file);
    let read_back = Fst::read(&mut input).unwrap();

    for (key, expected) in present {
        assert_eq!(
            read_back.get(key).unwrap(),
            Some(expected.to_vec()),
            "key {key:?} should resolve to {expected:?}"
        );
        // Also check the in-memory `Fst` `build_fst` returns directly,
        // independent of the `write_fst`/`Fst::read` serialization round
        // trip.
        assert_eq!(fst.get(key).unwrap(), Some(expected.to_vec()));
    }
    for key in absent {
        assert_eq!(
            read_back.get(key).unwrap(),
            None,
            "key {key:?} should be absent"
        );
        assert_eq!(fst.get(key).unwrap(), None);
    }
}

/// The same 7-key shape as `fixtures/src/GenFst.java`'s fixture (shared
/// prefixes within each group, two disjoint groups, plus a lone
/// single-byte key), so this test can be eyeballed against that fixture's
/// differential test for a sanity cross-check even though this builder
/// doesn't consume real Lucene bytes.
fn seven_key_fixture() -> Vec<(Vec<u8>, Vec<u8>)> {
    vec![
        (b"app".to_vec(), b"1".to_vec()),
        (b"apple".to_vec(), b"2".to_vec()),
        (b"application".to_vec(), b"3".to_vec()),
        (b"banana".to_vec(), b"4".to_vec()),
        (b"band".to_vec(), b"5".to_vec()),
        (b"bandana".to_vec(), b"6".to_vec()),
        (b"z".to_vec(), b"7".to_vec()),
    ]
}

#[test]
fn build_fst_seven_key_fixture_round_trips_through_real_reader() {
    let entries = seven_key_fixture();
    assert_round_trips(
        &entries,
        &[
            (b"app", b"1"),
            (b"apple", b"2"),
            (b"application", b"3"),
            (b"banana", b"4"),
            (b"band", b"5"),
            (b"bandana", b"6"),
            (b"z", b"7"),
        ],
        &[
            b"ap",         // proper prefix of app, not itself accepted
            b"appl",       // proper prefix of apple/application
            b"applicatio", // proper prefix of application
            b"appz",       // diverges mid-key
            b"ban",        // proper prefix of banana/band/bandana
            b"bandanas",   // extends past an accepting node
            b"",           // no empty-string key in this fixture
            b"zz",         // extends past z's accepting stop node
            b"missing-entirely",
        ],
    );
}

#[test]
fn build_fst_single_key() {
    let entries = vec![(b"cat".to_vec(), b"1".to_vec())];
    assert_round_trips(&entries, &[(b"cat", b"1")], &[b"ca", b"cats", b"dog", b""]);
}

#[test]
fn build_fst_empty_key_set_never_accepts_anything() {
    let fst = build_fst(&[]).unwrap();
    assert_eq!(fst.get(b"").unwrap(), None);
    assert_eq!(fst.get(b"anything").unwrap(), None);
}

#[test]
fn build_fst_accepts_the_empty_string_key_via_empty_output() {
    let entries = vec![
        (Vec::new(), b"root".to_vec()),
        (b"x".to_vec(), b"1".to_vec()),
    ];
    assert_round_trips(&entries, &[(b"", b"root"), (b"x", b"1")], &[b"xx", b"y"]);
}

#[test]
fn build_fst_one_key_is_a_proper_prefix_of_another() {
    // "band" is itself accepted *and* has a further child ('a' -> "na")
    // continuing to "bandana" -- exercises the non-stop-node branch of
    // `build_node` where a child is both final and has its own children.
    let entries = vec![
        (b"band".to_vec(), b"short".to_vec()),
        (b"bandana".to_vec(), b"long".to_vec()),
    ];
    assert_round_trips(
        &entries,
        &[(b"band", b"short"), (b"bandana", b"long")],
        &[b"ban", b"bandan", b"bandanas"],
    );
}

#[test]
fn build_fst_rejects_unsorted_input() {
    let entries = vec![(b"b".to_vec(), Vec::new()), (b"a".to_vec(), Vec::new())];
    assert_eq!(
        build_fst(&entries).unwrap_err(),
        BuildError::NotSorted { index: 1 }
    );
}

#[test]
fn build_fst_rejects_duplicate_key() {
    let entries = vec![(b"a".to_vec(), Vec::new()), (b"a".to_vec(), Vec::new())];
    assert_eq!(
        build_fst(&entries).unwrap_err(),
        BuildError::NotSorted { index: 1 }
    );
}

#[test]
fn build_fst_many_keys_forces_multi_byte_vlong_targets() {
    // Enough keys/branching that at least one child node's address
    // exceeds the 1-byte vlong threshold (127), forcing `write_vlong`'s
    // (and `read_vlong`'s, on the read side already covered above)
    // multi-byte continuation branch through the builder's own target
    // encoding.
    let entries: Vec<(Vec<u8>, Vec<u8>)> = (0u16..200)
        .map(|i| {
            let key = format!("key{i:04}").into_bytes();
            let output = i.to_le_bytes().to_vec();
            (key, output)
        })
        .collect();
    let fst = build_fst(&entries).unwrap();
    assert!(fst.metadata().num_bytes > 127);
    for (key, output) in &entries {
        assert_eq!(fst.get(key).unwrap(), Some(output.clone()));
    }
    assert_eq!(fst.get(b"key9999").unwrap(), None);
}

#[test]
fn build_fst_many_keys_round_trips_through_write_fst_and_both_readers() {
    // The 200-key/multi-byte-vlong-target case above only queries the
    // freshly-built in-memory `Fst` directly -- it never serializes via
    // `write_fst` and never re-parses via `Fst::read`/`Fst::read_borrowed`,
    // so the multi-byte vlong target shape was never exercised through
    // either actual reader, only through the builder's own in-memory
    // representation. This closes that gap.
    let entries: Vec<(Vec<u8>, Vec<u8>)> = (0u16..200)
        .map(|i| {
            let key = format!("key{i:04}").into_bytes();
            let output = i.to_le_bytes().to_vec();
            (key, output)
        })
        .collect();
    let built = build_fst(&entries).unwrap();
    let bytes = write_fst(&built);

    let mut owned_input = SliceInput::new(&bytes);
    let read_fst = Fst::read(&mut owned_input).unwrap();
    for (key, output) in &entries {
        assert_eq!(read_fst.get(key).unwrap(), Some(output.clone()));
    }
    assert_eq!(read_fst.get(b"key9999").unwrap(), None);

    let mut borrowed_input = SliceInput::new(&bytes);
    let borrowed_fst = Fst::read_borrowed(&mut borrowed_input).unwrap();
    assert!(borrowed_fst.is_borrowed());
    for (key, output) in &entries {
        assert_eq!(borrowed_fst.get(key).unwrap(), Some(output.clone()));
    }
    assert_eq!(borrowed_fst.get(b"key9999").unwrap(), None);
}

// --- `Fst::read_borrowed` (zero-copy body) ----------------------------

/// Builds one `(codec header + metadata + body)` byte buffer via
/// `build_fst`/`write_fst`, then confirms `Fst::read` (owned copy) and
/// `Fst::read_borrowed` (zero-copy) agree on every present/absent key --
/// same fixture shape `build_fst_seven_key_fixture_round_trips_through_real_reader`
/// already uses for the owned path.
#[test]
fn read_borrowed_matches_read_for_same_bytes() {
    let entries = seven_key_fixture();
    let built = build_fst(&entries).unwrap();
    let file = write_fst(&built);

    let mut owned_input = SliceInput::new(&file);
    let owned = Fst::read(&mut owned_input).unwrap();

    let mut borrowed_input = SliceInput::new(&file);
    let borrowed = Fst::read_borrowed(&mut borrowed_input).unwrap();

    assert!(!owned.is_borrowed());
    assert!(borrowed.is_borrowed());

    let present: &[(&[u8], &[u8])] = &[
        (b"app", b"1"),
        (b"apple", b"2"),
        (b"application", b"3"),
        (b"banana", b"4"),
        (b"band", b"5"),
        (b"bandana", b"6"),
        (b"z", b"7"),
    ];
    for (key, expected) in present {
        assert_eq!(owned.get(key).unwrap(), Some(expected.to_vec()));
        assert_eq!(borrowed.get(key).unwrap(), Some(expected.to_vec()));
    }
    for key in [b"ap".as_slice(), b"appz", b"missing-entirely"] {
        assert_eq!(owned.get(key).unwrap(), None);
        assert_eq!(borrowed.get(key).unwrap(), None);
    }
}

/// Structural proof (not just a runtime output check) that
/// `Fst::read_borrowed` really borrows rather than allocating its own
/// second copy of the body: `FstBytes::Borrowed` holds a `&[u8]`, which
/// (unlike `FstBytes::Owned(Vec<u8>)`) cannot itself own heap
/// allocation -- if the body were copied, this variant would have had
/// to be `Vec<u8>` to hold it. This is the honest, provable half of
/// "no extra full-size copy happens": a `&[u8]` is a pointer+length,
/// never a second buffer.
#[test]
fn read_borrowed_body_is_a_slice_not_a_second_owned_buffer() {
    let entries = seven_key_fixture();
    let built = build_fst(&entries).unwrap();
    let file = write_fst(&built);

    let mut input = SliceInput::new(&file);
    let fst = Fst::read_borrowed(&mut input).unwrap();
    match &fst.bytes {
        FstBytes::Borrowed(slice) => {
            // The slice's address must fall inside `file`'s own
            // allocation -- proof this is a view into the caller's
            // buffer, not a copy living at some independent address.
            let file_start = file.as_ptr() as usize;
            let file_end = file_start + file.len();
            let slice_start = slice.as_ptr() as usize;
            assert!(slice_start >= file_start && slice_start <= file_end);
            assert_eq!(slice.len(), fst.metadata.num_bytes as usize);
        }
        FstBytes::Owned(_) => panic!("read_borrowed must produce FstBytes::Borrowed"),
    }
}

/// End-to-end through this port's real zero-copy `Directory` backend
/// (`MmapDirectory`): writes an FST file to disk, opens it via
/// `MmapDirectory` (an actual OS `mmap(2)`, `lucene_store::directory`'s
/// `Input::Mapped`), and confirms `Fst::read_borrowed` over that mapped
/// buffer resolves keys correctly -- this is the concrete scenario the
/// module doc on `FstBytes` describes: a large FST backed by mmap'd
/// bytes, opened without a second full-size heap copy.
///
/// **`with_read_threshold(_, 0)`, and the variant is asserted.** This
/// FST is a few hundred bytes, i.e. under
/// `lucene_store::directory::SMALL_FILE_READ_THRESHOLD`, so the default
/// backend hands back an `Input::Owned` and this test would go on passing
/// while proving nothing whatsoever about mmap. The
/// assertion is what stops the next threshold change from doing it again.
#[test]
fn read_borrowed_over_a_real_mmap_directory_input() {
    use lucene_store::directory::{Directory, Input, MmapDirectory};
    use lucene_store::DataOutput;

    let entries = seven_key_fixture();
    let built = build_fst(&entries).unwrap();
    let file_bytes = write_fst(&built);

    // A guard, not a bare path: the manual `remove_dir_all` at the end of
    // this test is skipped whenever an assertion above it fires, and a
    // leaked directory here lands in a tmpfs -- i.e. in RAM. See
    // `lucene_util::test_support`.
    let root = lucene_util::test_support::TempDir::new("fst-mmap");

    let dir = MmapDirectory::with_read_threshold(&root, 0);
    let mut out = dir.create_output("fst.bin").unwrap();
    out.write_bytes(&file_bytes);
    out.close().unwrap();

    let mapped = dir.open("fst.bin").unwrap();
    assert!(
        matches!(mapped, Input::Mapped(_)),
        "this test is about the mapped path; it read the file instead: {mapped:?}"
    );
    let mut input = SliceInput::new(&mapped);
    let fst = Fst::read_borrowed(&mut input).unwrap();
    assert!(fst.is_borrowed());
    assert_eq!(fst.get(b"app").unwrap(), Some(b"1".to_vec()));
    assert_eq!(fst.get(b"bandana").unwrap(), Some(b"6".to_vec()));
    assert_eq!(fst.get(b"missing").unwrap(), None);

    // `root` removes itself here (kept if this test is unwinding).
}

#[test]
fn build_fst_long_output_forces_multi_byte_vint_length() {
    // Every other test's outputs are <=4 bytes, so the final-output
    // length's `write_vint` call never exercises its own multi-byte
    // (>127) continuation branch -- distinct from the vlong target
    // encoding the test above already stresses. A 200-byte output
    // forces that branch.
    let long_output: Vec<u8> = (0u16..200).map(|i| (i % 256) as u8).collect();
    let entries: Vec<(Vec<u8>, Vec<u8>)> = vec![
        (b"a".to_vec(), vec![]),
        (b"ab".to_vec(), long_output.clone()),
    ];
    let fst = build_fst(&entries).unwrap();
    assert_eq!(fst.get(b"a").unwrap(), Some(vec![]));
    assert_eq!(fst.get(b"ab").unwrap(), Some(long_output));
    assert_eq!(fst.get(b"abc").unwrap(), None);
}

// --- `Fst::iter` (full ordered enumeration) ---------------------------

fn collect_iter(fst: &Fst) -> Vec<(Vec<u8>, Vec<u8>)> {
    fst.iter()
        .expect("BYTE1 fixture should support iter")
        .collect::<Result<_>>()
        .expect("enumeration should not error")
}

#[test]
fn iter_over_empty_fst_yields_nothing() {
    let fst = build_fst(&[]).unwrap();
    assert_eq!(collect_iter(&fst), Vec::<(Vec<u8>, Vec<u8>)>::new());
}

#[test]
fn iter_over_single_key_fst_yields_that_one_key() {
    let entries = vec![(b"cat".to_vec(), b"1".to_vec())];
    let fst = build_fst(&entries).unwrap();
    assert_eq!(collect_iter(&fst), vec![(b"cat".to_vec(), b"1".to_vec())]);
}

#[test]
fn iter_stays_exhausted_after_yielding_none() {
    // Found in review: `advance()` used `upto == 0` to mean both
    // "not yet started" and "just finished" -- calling `next()` again
    // after it returned `None` re-entered the "not yet started" branch
    // and silently restarted enumeration from the first key instead of
    // staying exhausted, violating the standard Rust `Iterator`
    // contract (an iterator should keep returning `None` once it has).
    let entries = vec![(b"cat".to_vec(), b"1".to_vec())];
    let fst = build_fst(&entries).unwrap();
    let mut it = fst.iter().unwrap();
    assert_eq!(
        it.next().unwrap().unwrap(),
        (b"cat".to_vec(), b"1".to_vec())
    );
    assert!(it.next().is_none());
    // The bug: this second post-exhaustion call would incorrectly
    // yield "cat" again instead of staying `None`.
    assert!(it.next().is_none());
    assert!(it.next().is_none());
}

#[test]
fn iter_over_single_empty_string_key_yields_it() {
    // The only accepted key is the empty string itself (via emptyOutput),
    // with no other real arcs at all -- exercises `read_first_target_arc`'s
    // `follow.target() <= 0` branch (the synthetic END_LABEL arc is also
    // BIT_LAST_ARC, so enumeration must stop immediately after it).
    let entries = vec![(Vec::new(), b"root".to_vec())];
    let fst = build_fst(&entries).unwrap();
    assert_eq!(collect_iter(&fst), vec![(Vec::new(), b"root".to_vec())]);
}

#[test]
fn iter_over_seven_key_fixture_yields_ascending_sorted_order() {
    let entries = seven_key_fixture();
    let fst = build_fst(&entries).unwrap();
    assert_eq!(collect_iter(&fst), entries);
}

#[test]
fn iter_over_empty_string_plus_other_keys() {
    // Empty string accepted *and* the root has further real children --
    // exercises `read_first_target_arc`'s `follow.target() > 0` branch
    // (the synthetic END_LABEL arc is not last, so `read_next_arc` must
    // fall through to the root node's first real arc afterwards).
    let entries = vec![
        (Vec::new(), b"root".to_vec()),
        (b"x".to_vec(), b"1".to_vec()),
    ];
    let fst = build_fst(&entries).unwrap();
    assert_eq!(collect_iter(&fst), entries);
}

#[test]
fn iter_over_binary_search_root_node_yields_ascending_order() {
    let labels = *b"acfmz";
    let outputs = [1u8, 2, 3, 4, 5];
    let (bytes, start) = build_binary_search_node(&labels, &outputs);
    let fst = fst_from_body(bytes, start, InputType::Byte1, None);
    let expected: Vec<(Vec<u8>, Vec<u8>)> = labels
        .iter()
        .zip(outputs.iter())
        .map(|(&l, &o)| (vec![l], vec![o]))
        .collect();
    assert_eq!(collect_iter(&fst), expected);
}

#[test]
fn iter_errors_on_non_byte1_input_type() {
    let (bytes, start) = build_single_key_fst(b"x", b"1");
    let fst = fst_from_body(bytes, start, InputType::Byte2, None);
    assert!(matches!(fst.iter(), Err(Error::Unsupported(_))));
}

// --- Seek support (`seek_exact`/`seek_ceil`/`seek_floor`) --------------
//
// These exercise the hand-built, list-encoded seven-key fixture already
// used above -- `tests/fst_seek_fixtures.rs` covers the same operations
// against real, Lucene-written fixtures spanning all four node encodings
// (list, binary search, direct addressing, continuous).

#[test]
fn fst_seek_exact_on_present_and_absent_keys() {
    let fst = build_fst(&seven_key_fixture()).unwrap();
    assert_eq!(fst.seek_exact(b"app").unwrap(), Some(b"1".to_vec()));
    assert_eq!(fst.seek_exact(b"bandana").unwrap(), Some(b"6".to_vec()));
    assert_eq!(fst.seek_exact(b"appl").unwrap(), None);
    assert_eq!(fst.seek_exact(b"missing").unwrap(), None);

    let mut e = fst.iter().unwrap();
    assert_eq!(
        e.seek_exact(b"z").unwrap(),
        Some((b"z".to_vec(), b"7".to_vec()))
    );
    assert_eq!(e.seek_exact(b"appl").unwrap(), None);
}

#[test]
fn fst_enum_seek_ceil_lands_between_two_keys() {
    let fst = build_fst(&seven_key_fixture()).unwrap();
    let mut e = fst.iter().unwrap();
    // "appl" sits strictly between "app" and "apple".
    assert_eq!(
        e.seek_ceil(b"appl").unwrap(),
        Some((b"apple".to_vec(), b"2".to_vec()))
    );
    // "ban" sits strictly between "bandana"'s prefix and "banana".
    assert_eq!(
        e.seek_ceil(b"ban").unwrap(),
        Some((b"banana".to_vec(), b"4".to_vec()))
    );
}

#[test]
fn fst_enum_seek_ceil_before_first_key_finds_first_key() {
    let fst = build_fst(&seven_key_fixture()).unwrap();
    let mut e = fst.iter().unwrap();
    assert_eq!(
        e.seek_ceil(b"").unwrap(),
        Some((b"app".to_vec(), b"1".to_vec()))
    );
    assert_eq!(
        e.seek_ceil(b"AAA").unwrap(),
        Some((b"app".to_vec(), b"1".to_vec()))
    );
}

#[test]
fn fst_enum_seek_ceil_past_last_key_finds_nothing() {
    let fst = build_fst(&seven_key_fixture()).unwrap();
    let mut e = fst.iter().unwrap();
    assert_eq!(e.seek_ceil(b"zz").unwrap(), None);
    assert_eq!(e.seek_ceil(b"zzzzzzzzzz").unwrap(), None);
}

#[test]
fn fst_enum_seek_ceil_on_exact_key_returns_it() {
    let fst = build_fst(&seven_key_fixture()).unwrap();
    let mut e = fst.iter().unwrap();
    assert_eq!(
        e.seek_ceil(b"banana").unwrap(),
        Some((b"banana".to_vec(), b"4".to_vec()))
    );
}

#[test]
fn fst_enum_seek_floor_lands_between_two_keys() {
    let fst = build_fst(&seven_key_fixture()).unwrap();
    let mut e = fst.iter().unwrap();
    assert_eq!(
        e.seek_floor(b"appl").unwrap(),
        Some((b"app".to_vec(), b"1".to_vec()))
    );
    assert_eq!(
        e.seek_floor(b"bane").unwrap(),
        Some((b"bandana".to_vec(), b"6".to_vec()))
    );
}

#[test]
fn fst_enum_seek_floor_past_last_key_finds_last_key() {
    let fst = build_fst(&seven_key_fixture()).unwrap();
    let mut e = fst.iter().unwrap();
    assert_eq!(
        e.seek_floor(b"zzzzzzzzzz").unwrap(),
        Some((b"z".to_vec(), b"7".to_vec()))
    );
}

#[test]
fn fst_enum_seek_floor_before_first_key_finds_nothing() {
    let fst = build_fst(&seven_key_fixture()).unwrap();
    let mut e = fst.iter().unwrap();
    assert_eq!(e.seek_floor(b"").unwrap(), None);
    assert_eq!(e.seek_floor(b"AAA").unwrap(), None);
}

#[test]
fn fst_enum_seek_floor_on_exact_key_returns_it() {
    let fst = build_fst(&seven_key_fixture()).unwrap();
    let mut e = fst.iter().unwrap();
    assert_eq!(
        e.seek_floor(b"band").unwrap(),
        Some((b"band".to_vec(), b"5".to_vec()))
    );
}

#[test]
fn fst_enum_sequential_seeks_forward_and_backward_use_rewind_prefix() {
    // Exercises `rewind_prefix`'s forward (cmp<0) and backward (cmp>0)
    // branches across successive seeks on the same `FstEnum`, not just a
    // single fresh seek from the start.
    let fst = build_fst(&seven_key_fixture()).unwrap();
    let mut e = fst.iter().unwrap();
    assert_eq!(
        e.seek_ceil(b"application").unwrap(),
        Some((b"application".to_vec(), b"3".to_vec()))
    );
    // Backward: "ap" < "application"'s shared prefix.
    assert_eq!(
        e.seek_ceil(b"ap").unwrap(),
        Some((b"app".to_vec(), b"1".to_vec()))
    );
    // Forward again, past everything already visited.
    assert_eq!(
        e.seek_ceil(b"c").unwrap(),
        Some((b"z".to_vec(), b"7".to_vec()))
    );
    // "z" is the last key -- a plain `next()` after landing on it (via
    // seek) correctly reports the enumeration is exhausted, same as real
    // Lucene's `FSTEnum` (seeking only repositions the enum; it doesn't
    // change what "the next key after this one" means).
    assert!(e.next().is_none());
}

#[test]
fn fst_enum_seek_on_empty_string_key_fixture() {
    let entries = vec![
        (Vec::new(), b"root".to_vec()),
        (b"x".to_vec(), b"1".to_vec()),
    ];
    let fst = build_fst(&entries).unwrap();
    assert_eq!(fst.seek_exact(b"").unwrap(), Some(b"root".to_vec()));

    let mut e = fst.iter().unwrap();
    assert_eq!(
        e.seek_ceil(b"").unwrap(),
        Some((Vec::new(), b"root".to_vec()))
    );
    assert_eq!(
        e.seek_floor(b"").unwrap(),
        Some((Vec::new(), b"root".to_vec()))
    );
    assert_eq!(
        e.seek_ceil(b"w").unwrap(),
        Some((b"x".to_vec(), b"1".to_vec()))
    );
    assert_eq!(
        e.seek_floor(b"w").unwrap(),
        Some((Vec::new(), b"root".to_vec()))
    );
}

#[test]
fn fst_enum_seek_on_empty_fst_finds_nothing() {
    let fst = build_fst(&[]).unwrap();
    let mut e = fst.iter().unwrap();
    assert_eq!(e.seek_ceil(b"anything").unwrap(), None);
    assert_eq!(e.seek_floor(b"anything").unwrap(), None);
    assert_eq!(fst.seek_exact(b"").unwrap(), None);
}

#[test]
fn fst_enum_next_resumes_after_seek_following_full_exhaustion() {
    // `seek_ceil`/`seek_floor`/`seek_exact` must clear
    // `done` (a Rust-only fused-iterator flag with no Java equivalent) so
    // that repositioning a fully-exhausted `FstEnum` and calling `next()`
    // again resumes ordered enumeration instead of short-circuiting.
    let fst = build_fst(&seven_key_fixture()).unwrap();
    let mut e = fst.iter().unwrap();
    while e.next().is_some() {}
    assert_eq!(
        e.seek_ceil(b"ban").unwrap(),
        Some((b"banana".to_vec(), b"4".to_vec()))
    );
    assert_eq!(
        e.next().unwrap().unwrap(),
        (b"band".to_vec(), b"5".to_vec())
    );
    assert_eq!(
        e.next().unwrap().unwrap(),
        (b"bandana".to_vec(), b"6".to_vec())
    );
}

#[test]
fn fst_enum_next_resumes_after_seek_floor_following_full_exhaustion() {
    let fst = build_fst(&seven_key_fixture()).unwrap();
    let mut e = fst.iter().unwrap();
    while e.next().is_some() {}
    assert_eq!(
        e.seek_floor(b"band").unwrap(),
        Some((b"band".to_vec(), b"5".to_vec()))
    );
    assert_eq!(
        e.next().unwrap().unwrap(),
        (b"bandana".to_vec(), b"6".to_vec())
    );
}

#[test]
fn fst_enum_next_resumes_after_seek_exact_following_full_exhaustion() {
    let fst = build_fst(&seven_key_fixture()).unwrap();
    let mut e = fst.iter().unwrap();
    while e.next().is_some() {}
    assert_eq!(
        e.seek_exact(b"banana").unwrap(),
        Some((b"banana".to_vec(), b"4".to_vec()))
    );
    assert_eq!(
        e.next().unwrap().unwrap(),
        (b"band".to_vec(), b"5".to_vec())
    );
}

// --- Suffix sharing / minimization proof tests -------------------------
//
// These prove `build_node`'s `NodeHash` dedup actually collapses
// structurally-identical nodes reached via different prefixes into a
// single shared address, rather than merely producing a correct (but
// possibly non-minimal) FST. A "round-trips correctly" test alone would
// pass even for the old, pre-dedup naive trie builder -- these instead
// inspect the compiled node addresses directly.

/// Walks `prefix` from the FST's root and returns the address of the
/// node reached after consuming it (i.e. the `target()` of the arc for
/// `prefix`'s last byte) -- the same "which node did we land on" question
/// `Fst::get` answers per-byte, just exposed here instead of being
/// discarded after the final `is_final()` check.
fn node_after_prefix(fst: &Fst<'_>, prefix: &[u8]) -> i64 {
    let mut r = fst.reader();
    let mut arc = fst.first_arc();
    for &b in prefix {
        arc = fst
            .find_target_arc(b as i32, &arc, &mut r)
            .unwrap()
            .unwrap_or_else(|| panic!("prefix {prefix:?} not found in FST"));
    }
    arc.target()
}

#[test]
fn build_fst_shares_identical_suffix_node_across_two_prefixes() {
    // "bat" and "cat" share the final "at" suffix: the node reached
    // after 'b' and the node reached after 'c' are structurally
    // identical (single child arc 'a', itself leading to a single child
    // arc 't', final, no output) and must be written to the byte store
    // exactly once and reused, i.e. `node_after_prefix(.., b"b")` and
    // `node_after_prefix(.., b"c")` must be the *same* address.
    let entries = vec![(b"bat".to_vec(), Vec::new()), (b"cat".to_vec(), Vec::new())];
    let fst = build_fst(&entries).unwrap();

    let after_b = node_after_prefix(&fst, b"b");
    let after_c = node_after_prefix(&fst, b"c");
    assert_eq!(
            after_b, after_c,
            "the shared \"at\" suffix must compile to one shared node address, not two independent copies"
        );

    // Sharing a node must not corrupt lookups through either path.
    assert_eq!(fst.get(b"bat").unwrap(), Some(Vec::new()));
    assert_eq!(fst.get(b"cat").unwrap(), Some(Vec::new()));
    assert_eq!(fst.get(b"at").unwrap(), None);
    assert_eq!(fst.get(b"ba").unwrap(), None);
}

#[test]
fn build_fst_shares_classic_mop_pop_stop_top_suffix_node() {
    // Real Lucene's own canonical `FSTCompiler` suffix-sharing example:
    // "mop"/"moth"/"pop"/"star"/"stop"/"top". The nodes reached after
    // "po", "sto", and "to" are each a single final 'p' arc (stop node,
    // no output) and so must all three collapse to the identical shared
    // address, while the node after "mo" (which additionally has a 'th'
    // child) must NOT be that same address.
    let entries = vec![
        (b"mop".to_vec(), Vec::new()),
        (b"moth".to_vec(), Vec::new()),
        (b"pop".to_vec(), Vec::new()),
        (b"star".to_vec(), Vec::new()),
        (b"stop".to_vec(), Vec::new()),
        (b"top".to_vec(), Vec::new()),
    ];
    let fst = build_fst(&entries).unwrap();

    let after_po = node_after_prefix(&fst, b"po");
    let after_sto = node_after_prefix(&fst, b"sto");
    let after_to = node_after_prefix(&fst, b"to");
    let after_mo = node_after_prefix(&fst, b"mo");

    assert_eq!(
        after_po, after_sto,
        "\"po\" and \"sto\" both end in a bare final 'p' node and must share it"
    );
    assert_eq!(
        after_sto, after_to,
        "\"sto\" and \"to\" both end in a bare final 'p' node and must share it"
    );
    assert_ne!(
            after_mo, after_po,
            "\"mo\" has an extra 'th' child so its node must NOT be reused for the plain-'p' shared node"
        );

    // Regression: get/seek/enumerate all still correct against an FST
    // that now has genuinely shared/multiply-referenced nodes.
    for (key, _) in &entries {
        assert_eq!(fst.get(key).unwrap(), Some(Vec::new()), "get({key:?})");
    }
    assert_eq!(collect_iter(&fst), entries);

    let mut e = fst.iter().unwrap();
    assert_eq!(
        e.seek_ceil(b"mos").unwrap(),
        Some((b"moth".to_vec(), Vec::new()))
    );
    let mut e = fst.iter().unwrap();
    assert_eq!(
        e.seek_floor(b"stoz").unwrap(),
        Some((b"stop".to_vec(), Vec::new()))
    );
    let mut e = fst.iter().unwrap();
    assert_eq!(
        e.seek_exact(b"top").unwrap(),
        Some((b"top".to_vec(), Vec::new()))
    );
}

#[test]
fn build_fst_dedup_keeps_compiled_size_sublinear_in_repeated_suffix_count() {
    // A naive (non-deduplicating) trie builder writes one independent
    // copy of the shared tail's arcs per distinct prefix, so its total
    // size grows linearly with `num_prefixes * tail.len()`. With
    // suffix-sharing minimization, the tail's arcs are written once and
    // every prefix's last arc just targets that single shared address,
    // so total size grows roughly like `num_prefixes + tail.len()`
    // instead. Assert the actual compiled size is well under the naive
    // linear bound to prove sharing, not just correctness, occurred.
    const TAIL: &[u8] = b"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"; // 44 bytes
    let prefixes: &[&[u8]] = &[b"aa", b"bb", b"cc", b"dd", b"ee", b"ff", b"gg", b"hh"];

    let mut entries: Vec<(Vec<u8>, Vec<u8>)> = prefixes
        .iter()
        .map(|p| {
            let mut key = p.to_vec();
            key.extend_from_slice(TAIL);
            (key, Vec::new())
        })
        .collect();
    entries.sort();

    let fst = build_fst(&entries).unwrap();

    // Naive bound: each of the 8 prefixes would need its own full,
    // independent copy of every arc down the shared tail. Each arc here
    // is at least 3 bytes (flags + label + 1-byte vlong target), so a
    // fully-unshared build would need well over
    // `prefixes.len() * TAIL.len() * 3` bytes for the tails alone.
    let naive_unshared_lower_bound = prefixes.len() * TAIL.len() * 3;
    assert!(
        (fst.metadata.num_bytes as usize) < naive_unshared_lower_bound,
        "compiled size {} should be far smaller than the naive no-sharing bound {} \
             if suffix sharing actually collapsed the repeated tail to one copy",
        fst.metadata.num_bytes,
        naive_unshared_lower_bound
    );

    // And every key must still resolve correctly through the shared tail.
    for (key, output) in &entries {
        assert_eq!(fst.get(key).unwrap().as_ref(), Some(output));
    }

    // Direct structural check: the node reached after each distinct
    // 2-byte prefix must be the *same* address (all of them lead into
    // the one shared `TAIL` chain).
    let first_addr = node_after_prefix(&fst, prefixes[0]);
    for p in &prefixes[1..] {
        assert_eq!(
            node_after_prefix(&fst, p),
            first_addr,
            "prefix {p:?} should land on the same shared tail node as {:?}",
            prefixes[0]
        );
    }
}

// --- Typed outputs: `PositiveIntOutputs` / `PairOutputs` ---------------

type WeightedEntry = (Vec<u8>, Pair<i64, Vec<u8>>);

#[test]
fn positive_int_outputs_round_trips_values_including_zero() {
    let entries: Vec<(Vec<u8>, i64)> = vec![
        (b"a".to_vec(), 0),
        (b"apple".to_vec(), 1),
        (b"banana".to_vec(), 300),
        (b"grape".to_vec(), i64::MAX),
    ];
    let fst = build_fst_typed::<PositiveIntOutputs>(&entries).unwrap();
    for (key, value) in &entries {
        assert_eq!(
            fst.get_typed::<PositiveIntOutputs>(key).unwrap(),
            Some(*value),
            "get_typed({key:?})"
        );
    }
    assert_eq!(
        fst.get_typed::<PositiveIntOutputs>(b"missing").unwrap(),
        None
    );

    // NO_OUTPUT (0) must encode to zero extra bytes, same as a plain
    // empty `Vec<u8>` output today.
    assert_eq!(PositiveIntOutputs::encode(&0), Vec::<u8>::new());
}

#[test]
fn pair_outputs_round_trips_both_components_over_a_shared_prefix_key_set() {
    // A realistic key set with both shared prefixes ("band"/"banana"/
    // "bandana") and a shared suffix ("cat"/"bat", stressing interaction
    // with `build_node`'s suffix-sharing dedup) -- every key gets a
    // distinct (weight, payload) pair and both components must read
    // back correctly regardless of which nodes ended up shared.
    type Weighted = PairOutputs<PositiveIntOutputs, ByteSequenceOutputs>;
    let entries: Vec<WeightedEntry> = vec![
        (
            b"banana".to_vec(),
            Pair {
                first: 4,
                second: b"fruit".to_vec(),
            },
        ),
        (
            b"band".to_vec(),
            Pair {
                first: 5,
                second: b"music".to_vec(),
            },
        ),
        (
            b"bandana".to_vec(),
            Pair {
                first: 6,
                second: b"headwear".to_vec(),
            },
        ),
        (
            b"bat".to_vec(),
            Pair {
                first: 1,
                second: Vec::new(),
            },
        ),
        (
            b"cat".to_vec(),
            Pair {
                first: 2,
                second: b"meow".to_vec(),
            },
        ),
    ];
    let sorted = {
        let mut e = entries.clone();
        e.sort_by(|a, b| a.0.cmp(&b.0));
        e
    };
    let fst = build_fst_typed::<Weighted>(&sorted).unwrap();

    for (key, value) in &entries {
        let got = fst.get_typed::<Weighted>(key).unwrap();
        assert_eq!(got.as_ref(), Some(value), "get_typed({key:?})");
    }
    assert_eq!(fst.get_typed::<Weighted>(b"ba").unwrap(), None);
}

#[test]
fn pair_outputs_shared_suffix_nodes_keep_distinct_first_components() {
    // Highest-risk interaction with suffix sharing: "bat" and "cat" would
    // share their tail node under plain `ByteSequenceOutputs` (see
    // `build_fst_shares_identical_suffix_node_across_two_prefixes`
    // above), but here their *entire* output lives on that same final
    // arc and differs per key (different `Pair` values) -- so the two
    // subtrees must NOT collapse to one shared node this time, and each
    // key's distinct pair must still read back correctly.
    type Weighted = PairOutputs<PositiveIntOutputs, ByteSequenceOutputs>;
    let entries: Vec<WeightedEntry> = vec![
        (
            b"bat".to_vec(),
            Pair {
                first: 1,
                second: b"flying".to_vec(),
            },
        ),
        (
            b"cat".to_vec(),
            Pair {
                first: 2,
                second: b"purring".to_vec(),
            },
        ),
    ];
    let fst = build_fst_typed::<Weighted>(&entries).unwrap();

    let after_b = node_after_prefix(&fst, b"b");
    let after_c = node_after_prefix(&fst, b"c");
    assert_ne!(
        after_b, after_c,
        "distinct per-key Pair outputs on the shared-shape \"at\" tail must prevent node sharing"
    );

    assert_eq!(
        fst.get_typed::<Weighted>(b"bat").unwrap(),
        Some(Pair {
            first: 1,
            second: b"flying".to_vec()
        })
    );
    assert_eq!(
        fst.get_typed::<Weighted>(b"cat").unwrap(),
        Some(Pair {
            first: 2,
            second: b"purring".to_vec()
        })
    );
}

#[test]
fn pair_outputs_zero_value_encodes_to_no_extra_bytes() {
    type IntPair = PairOutputs<PositiveIntOutputs, PositiveIntOutputs>;
    let zero = <IntPair as Outputs>::zero();
    assert_eq!(IntPair::encode(&zero), Vec::<u8>::new());
    assert_eq!(IntPair::decode(&[]).unwrap(), zero);
}

// --- `IntsRefFstEnum` (`IntsRefFSTEnum`-equivalent) --------------------

#[test]
fn encode_decode_ints_key_round_trips() {
    let key = vec![0, 1, 255, 256, 65536, i32::MAX];
    let encoded = encode_ints_key(&key);
    assert_eq!(encoded.len(), key.len() * 4);
    assert_eq!(decode_ints_key(&encoded).unwrap(), key);
    // A key whose length is not a multiple of four cannot have come from
    // `encode_ints_key`; it is what enumerating a `BYTE1` FST as
    // `IntsRef` keys produces, and it must be an error rather than a
    // silently truncated key.
    assert!(decode_ints_key(&encoded[..encoded.len() - 1]).is_err());
}

#[test]
#[should_panic(expected = "encode_ints_key requires non-negative elements")]
fn encode_ints_key_rejects_a_negative_element_even_in_release_builds() {
    // A real `assert!`, not `debug_assert!` -- must not be compiled out
    // under `--release`, since silently accepting a negative element
    // would corrupt the big-endian ordering invariant this whole
    // IntsRefFstEnum scoping depends on.
    encode_ints_key(&[3, -1, 7]);
}

#[test]
fn encode_ints_key_big_endian_preserves_numeric_order() {
    // Unsigned-byte-lexicographic order over the encoding must match
    // numeric order over the original ints, which is the property that
    // lets ascending byte-key enumeration double as ascending int-key
    // enumeration -- see the `IntsRefFstEnum` section doc.
    let mut values = vec![0i32, 1, 2, 255, 256, 65535, 65536, 1_000_000, i32::MAX];
    let mut encoded: Vec<Vec<u8>> = values.iter().map(|&v| encode_ints_key(&[v])).collect();
    let sorted_encoded = {
        let mut e = encoded.clone();
        e.sort();
        e
    };
    values.sort_unstable();
    encoded.sort();
    assert_eq!(encoded, sorted_encoded);
    assert_eq!(
        encoded,
        values
            .iter()
            .map(|&v| encode_ints_key(&[v]))
            .collect::<Vec<_>>()
    );
}

#[test]
fn ints_ref_fst_enum_over_empty_fst_yields_nothing() {
    let fst = build_fst_from_ints::<ByteSequenceOutputs>(&[]).unwrap();
    let keys: Vec<_> = fst.iter_ints().unwrap().map(|r| r.unwrap().0).collect();
    assert!(keys.is_empty());
}

#[test]
fn ints_ref_fst_enum_over_single_key_fst_yields_that_one_key() {
    let entries = vec![(vec![1i32, 2, 3], b"abc".to_vec())];
    let fst = build_fst_from_ints::<ByteSequenceOutputs>(&entries).unwrap();
    let results: Vec<_> = fst.iter_ints().unwrap().map(|r| r.unwrap()).collect();
    assert_eq!(results, vec![(vec![1, 2, 3], b"abc".to_vec())]);
}

#[test]
fn ints_ref_fst_enum_multi_key_shared_prefix_and_suffix_ascending_order() {
    // Overlapping prefixes (`[1,2,*]`) and a shared suffix (`,9,9]`) to
    // stress enumeration together with suffix-sharing minimization.
    let mut entries = vec![
        (vec![1i32, 2, 100], 10i64),
        (vec![1, 2, 200], 20i64),
        (vec![1, 3, 9, 9], 30i64),
        (vec![5, 3, 9, 9], 40i64),
        (vec![5, 4], 50i64),
    ];
    entries.sort_by(|a, b| a.0.cmp(&b.0));

    let fst = build_fst_from_ints::<PositiveIntOutputs>(&entries).unwrap();
    let results: Vec<_> = fst
        .iter_ints()
        .unwrap()
        .map(|r| {
            let (key, output) = r.unwrap();
            (key, PositiveIntOutputs::decode(&output).unwrap())
        })
        .collect();
    assert_eq!(results, entries);

    // Confirm each key is independently reachable via `get_typed` too,
    // through the same big-endian encoding boundary.
    for (key, value) in &entries {
        assert_eq!(
            fst.get_typed::<PositiveIntOutputs>(&encode_ints_key(key))
                .unwrap(),
            Some(*value)
        );
    }
}

#[test]
fn ints_ref_fst_enum_seek_ceil_floor_exact() {
    let entries = vec![
        (vec![1i32, 2], b"a".to_vec()),
        (vec![1, 2, 3], b"b".to_vec()),
        (vec![1, 5], b"c".to_vec()),
        (vec![9, 9], b"d".to_vec()),
    ];
    let fst = build_fst_from_ints::<ByteSequenceOutputs>(&entries).unwrap();
    let mut e = fst.iter_ints().unwrap();

    // seek_ceil lands on an in-between target's smallest key >= target.
    assert_eq!(
        e.seek_ceil(&[1, 3]).unwrap(),
        Some((vec![1, 5], b"c".to_vec()))
    );
    // seek_floor lands on the largest key <= target.
    assert_eq!(
        e.seek_floor(&[1, 4]).unwrap(),
        Some((vec![1, 2, 3], b"b".to_vec()))
    );
    // seek_exact only succeeds on a real key.
    assert_eq!(
        e.seek_exact(&[9, 9]).unwrap(),
        Some((vec![9, 9], b"d".to_vec()))
    );
    assert_eq!(e.seek_exact(&[9, 8]).unwrap(), None);
    // Past the last key.
    assert_eq!(e.seek_ceil(&[100]).unwrap(), None);
    // Before the first key: no accepted key <= target.
    assert_eq!(e.seek_floor(&[0]).unwrap(), None);
}
