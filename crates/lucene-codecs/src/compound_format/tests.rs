#![cfg(test)]
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

fn build_cfe(id: &[u8; ID_LENGTH], entries: &[(&str, i64, i64)]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
    write_string(&mut out, ENTRY_CODEC);
    out.extend_from_slice(&(VERSION_CURRENT as u32).to_be_bytes());
    out.extend_from_slice(id);
    out.push(0); // empty suffix
    write_vint(&mut out, entries.len() as i32);
    for (name, offset, length) in entries {
        write_string(&mut out, name);
        out.extend_from_slice(&offset.to_le_bytes());
        out.extend_from_slice(&length.to_le_bytes());
    }
    out.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
    out.extend_from_slice(&0u32.to_be_bytes());
    let checksum = crc32fast::hash(&out) as u64;
    out.extend_from_slice(&checksum.to_be_bytes());
    out
}

fn build_cfs(id: &[u8; ID_LENGTH], version: i32, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
    write_string(&mut out, DATA_CODEC);
    out.extend_from_slice(&(version as u32).to_be_bytes());
    out.extend_from_slice(id);
    out.push(0);
    out.extend_from_slice(payload);
    out.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
    out.extend_from_slice(&0u32.to_be_bytes());
    let checksum = crc32fast::hash(&out) as u64;
    out.extend_from_slice(&checksum.to_be_bytes());
    out
}

#[test]
fn empty_entries_parses() {
    let id = [1u8; ID_LENGTH];
    let buf = build_cfe(&id, &[]);
    let entries = parse_entries(&buf, &id).unwrap();
    assert_eq!(entries.version, VERSION_CURRENT);
    assert_eq!(entries.entries.len(), 0);
}

#[test]
fn duplicate_entry_rejected() {
    let id = [1u8; ID_LENGTH];
    let buf = build_cfe(&id, &[(".fnm", 0, 10), (".fnm", 10, 5)]);
    assert!(matches!(
        parse_entries(&buf, &id),
        Err(Error::DuplicateEntry(name)) if name == ".fnm"
    ));
}

#[test]
fn wrong_id_rejected() {
    let id = [1u8; ID_LENGTH];
    let buf = build_cfe(&id, &[]);
    let wrong_id = [2u8; ID_LENGTH];
    assert!(matches!(
        parse_entries(&buf, &wrong_id),
        Err(Error::Store(_))
    ));
}

#[test]
fn open_input_slices_correct_bytes() {
    let id = [1u8; ID_LENGTH];
    let header_len = codec_util::index_header_length(DATA_CODEC, "");
    let payload = b"hello world!";
    let cfe = build_cfe(
        &id,
        &[
            (".fnm", header_len as i64, 5),
            (".si", header_len as i64 + 5, 7),
        ],
    );
    let entries = parse_entries(&cfe, &id).unwrap();
    let cfs = build_cfs(&id, VERSION_CURRENT, payload);

    assert_eq!(
        open_input(&cfs, &entries, ".fnm").unwrap().as_slice(),
        b"hello"
    );
    assert_eq!(
        open_input(&cfs, &entries, ".si").unwrap().as_slice(),
        b" world!"
    );
}

#[test]
fn open_input_unknown_id_rejected() {
    let id = [1u8; ID_LENGTH];
    let entries = parse_entries(&build_cfe(&id, &[]), &id).unwrap();
    let cfs = build_cfs(&id, VERSION_CURRENT, b"");
    assert!(matches!(
        open_input(&cfs, &entries, ".nope"),
        Err(Error::FileNotFound(name, _)) if name == ".nope"
    ));
}

#[test]
fn check_data_header_footer_valid_with_entries() {
    let id = [1u8; ID_LENGTH];
    let header_len = codec_util::index_header_length(DATA_CODEC, "");
    let payload = b"abcdefg";
    let cfe = build_cfe(&id, &[(".fnm", header_len as i64, payload.len() as i64)]);
    let entries = parse_entries(&cfe, &id).unwrap();
    let cfs = build_cfs(&id, VERSION_CURRENT, payload);
    check_data_header_footer(&cfs, &id, &entries).unwrap();
}

#[test]
fn check_data_header_footer_valid_with_no_entries() {
    let id = [1u8; ID_LENGTH];
    let entries = parse_entries(&build_cfe(&id, &[]), &id).unwrap();
    let cfs = build_cfs(&id, VERSION_CURRENT, b"");
    check_data_header_footer(&cfs, &id, &entries).unwrap();
}

#[test]
fn check_data_header_footer_wrong_length_rejected() {
    let id = [1u8; ID_LENGTH];
    let header_len = codec_util::index_header_length(DATA_CODEC, "");
    let cfe = build_cfe(&id, &[(".fnm", header_len as i64, 100)]); // implies far more data than exists
    let entries = parse_entries(&cfe, &id).unwrap();
    let cfs = build_cfs(&id, VERSION_CURRENT, b"short");
    assert!(matches!(
        check_data_header_footer(&cfs, &id, &entries),
        Err(Error::WrongLength { .. })
    ));
}

#[test]
fn check_data_header_footer_wrong_id_rejected() {
    let id = [1u8; ID_LENGTH];
    let entries = parse_entries(&build_cfe(&id, &[]), &id).unwrap();
    let cfs = build_cfs(&id, VERSION_CURRENT, b"");
    let wrong_id = [2u8; ID_LENGTH];
    assert!(matches!(
        check_data_header_footer(&cfs, &wrong_id, &entries),
        Err(Error::Store(_))
    ));
}

#[test]
fn open_input_entry_offset_past_end_of_data_is_error() {
    let id = [1u8; ID_LENGTH];
    // Entry claims bytes far beyond what the (short) data buffer holds.
    let cfe = build_cfe(&id, &[(".fnm", 1000, 5)]);
    let entries = parse_entries(&cfe, &id).unwrap();
    let cfs = build_cfs(&id, VERSION_CURRENT, b"short");
    assert!(matches!(
        open_input(&cfs, &entries, ".fnm"),
        Err(Error::Store(_))
    ));
}

/// `.cfe` offsets/lengths are `i64` straight off disk. A corrupt pair
/// whose sum overflows must land in `WrongLength`, not panic on the
/// `offset + length` addition (which is a debug-build panic in Rust
/// where Java's `long` silently wraps).
#[test]
fn overflowing_entry_extent_is_a_length_error_not_a_panic() {
    let id = [3u8; ID_LENGTH];
    let cfe = build_cfe(&id, &[("f", i64::MAX, i64::MAX)]);
    let entries = parse_entries(&cfe, &id).unwrap();
    let cfs = build_cfs(&id, VERSION_CURRENT, &[]);
    assert!(matches!(
        check_data_header_footer(&cfs, &id, &entries),
        Err(Error::WrongLength { .. })
    ));
}

/// A corrupt entry count reads past the file instead of reserving
/// `i32::MAX` entries.
#[test]
fn huge_entry_count_is_an_error_not_an_allocation() {
    let id = [5u8; ID_LENGTH];
    let mut cfe = build_cfe(&id, &[("f", 0, 0)]);
    let at = codec_util::index_header_length(ENTRY_CODEC, "");
    cfe.splice(at..at + 1, [0xff, 0xff, 0xff, 0xff, 0x07]);
    assert!(parse_entries(&cfe, &id).is_err());
}

/// The mirror case: a negative offset (only reachable from a corrupt
/// `.cfe`) must not slice anything out of the `.cfs`.
#[test]
fn negative_entry_offset_is_rejected_by_open_input() {
    let id = [4u8; ID_LENGTH];
    let cfe = build_cfe(&id, &[("f", -8, 4)]);
    let entries = parse_entries(&cfe, &id).unwrap();
    let cfs = build_cfs(&id, VERSION_CURRENT, &[]);
    assert!(open_input(&cfs, &entries, "f").is_err());
}

/// `lucene_store::codec_util::index_header_length` (`9 + name + ID_LENGTH +
/// 1 + suffix`) agrees with the alternative derivation (`4 + vint_len(name)
/// then name, then 4 + ID_LENGTH + 1`) for every codec name
/// `write_index_header` accepts -- ASCII shorter than 128 bytes, so the
/// name's length prefix is exactly one byte. Pinned here against the bytes
/// `write_index_header` actually emits rather than against a second
/// hand-derived formula.
#[test]
fn index_header_length_matches_the_bytes_write_index_header_emits() {
    let id = [7u8; ID_LENGTH];
    for codec in [DATA_CODEC, ENTRY_CODEC, "A", &"x".repeat(127)] {
        let mut out = Vec::new();
        codec_util::write_index_header(&mut out, codec, VERSION_CURRENT, &id, "");
        assert_eq!(
            out.len(),
            codec_util::index_header_length(codec, ""),
            "index_header_length disagrees with write_index_header for {codec:?}"
        );
    }
}

/// Builds a standalone codec file (header + body + footer) as a real
/// sub-file `write()` would consume -- the same shape produced by e.g.
/// `field_infos::write`/`stored_fields::write_best_speed`.
fn build_sub_file(codec: &str, version: i32, id: &[u8; ID_LENGTH], body: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    codec_util::write_index_header(&mut out, codec, version, id, "");
    out.extend_from_slice(body);
    codec_util::write_footer(&mut out);
    out
}

#[test]
fn write_round_trips_through_reader_with_multiple_sub_files() {
    let id = [42u8; ID_LENGTH];
    // Three distinct sub-files of different sizes/codecs so both the
    // ascending-size ordering and the alignment/offset math are actually
    // exercised (a single sub-file would leave that logic unchecked).
    let fnm = build_sub_file("FieldInfos", 1, &id, b"field infos body bytes");
    let fdt = build_sub_file("StoredFields", 3, &id, &[7u8; 200]);
    let dvd = build_sub_file("DocValues", 0, &id, b"dv");

    let sub_files = vec![
        (".fnm".to_string(), fnm.clone()),
        (".fdt".to_string(), fdt.clone()),
        (".dvd".to_string(), dvd.clone()),
    ];

    let (cfs, cfe) = write(&id, &sub_files).unwrap();

    let entries = parse_entries(&cfe, &id).unwrap();
    assert_eq!(entries.entries.len(), 3);
    check_data_header_footer(&cfs, &id, &entries).unwrap();

    // Every sub-file must come back byte-for-byte identical, including
    // its own header and footer -- a corrupted offset could otherwise
    // still leave the entries table "looking right" while shifting bytes.
    assert_eq!(
        open_input(&cfs, &entries, ".fnm").unwrap().as_slice(),
        fnm.as_slice()
    );
    assert_eq!(
        open_input(&cfs, &entries, ".fdt").unwrap().as_slice(),
        fdt.as_slice()
    );
    assert_eq!(
        open_input(&cfs, &entries, ".dvd").unwrap().as_slice(),
        dvd.as_slice()
    );

    // Ascending-size packing: .dvd (smallest) should be placed before
    // .fnm, which should be placed before .fdt (largest).
    let dvd_offset = entries.get(".dvd").unwrap().offset;
    let fnm_offset = entries.get(".fnm").unwrap().offset;
    let fdt_offset = entries.get(".fdt").unwrap().offset;
    assert!(dvd_offset < fnm_offset);
    assert!(fnm_offset < fdt_offset);

    // Every sub-file start offset is 64-byte aligned.
    for (_, entry) in &entries.entries {
        assert_eq!(entry.offset % ALIGNMENT_BYTES as i64, 0);
    }
}

#[test]
fn write_empty_sub_files_round_trips() {
    let id = [9u8; ID_LENGTH];
    let (cfs, cfe) = write(&id, &[]).unwrap();
    let entries = parse_entries(&cfe, &id).unwrap();
    assert_eq!(entries.entries.len(), 0);
    check_data_header_footer(&cfs, &id, &entries).unwrap();
}

#[test]
fn write_rejects_sub_file_with_wrong_segment_id() {
    let id = [1u8; ID_LENGTH];
    let wrong_id = [2u8; ID_LENGTH];
    let sub_file = build_sub_file("FieldInfos", 1, &wrong_id, b"body");
    let err = write(&id, &[(".fnm".to_string(), sub_file)]).unwrap_err();
    assert!(matches!(err, Error::Store(_)));
}

#[test]
fn write_rejects_sub_file_with_bad_magic() {
    let id = [1u8; ID_LENGTH];
    let mut sub_file = build_sub_file("FieldInfos", 1, &id, b"body");
    sub_file[0] ^= 0xFF; // corrupt the magic
    let err = write(&id, &[(".fnm".to_string(), sub_file)]).unwrap_err();
    assert!(matches!(err, Error::BadSubFileMagic(name) if name == ".fnm"));
}

#[test]
fn write_rejects_sub_file_with_corrupt_footer_checksum() {
    let id = [1u8; ID_LENGTH];
    let mut sub_file = build_sub_file("FieldInfos", 1, &id, b"body");
    let last = sub_file.len() - 1;
    sub_file[last] ^= 0xFF; // corrupt checksum field
    let err = write(&id, &[(".fnm".to_string(), sub_file)]).unwrap_err();
    assert!(matches!(err, Error::Store(_)));
}

#[test]
fn write_rejects_sub_file_too_short_for_footer() {
    let id = [1u8; ID_LENGTH];
    // Valid header but no room left for a footer afterwards.
    let mut sub_file = Vec::new();
    codec_util::write_index_header(&mut sub_file, "FieldInfos", 1, &id, "");
    let err = write(&id, &[(".fnm".to_string(), sub_file)]).unwrap_err();
    assert!(matches!(err, Error::Store(_)));
}

#[test]
fn write_rejects_completely_empty_sub_file_bytes() {
    // Not even a full magic number's worth of bytes -- must fail cleanly
    // (EOF) rather than panic on the initial `read_be_u32`.
    let id = [1u8; ID_LENGTH];
    let err = write(&id, &[(".fnm".to_string(), Vec::new())]).unwrap_err();
    assert!(matches!(err, Error::Store(_)));
}

#[test]
fn write_single_sub_file_round_trips() {
    // Exactly one sub-file: no ordering to exercise, but the header/footer
    // framing and offset math for a lone entry must still be correct.
    let id = [3u8; ID_LENGTH];
    let fnm = build_sub_file("FieldInfos", 1, &id, b"only file");
    let (cfs, cfe) = write(&id, &[(".fnm".to_string(), fnm.clone())]).unwrap();

    let entries = parse_entries(&cfe, &id).unwrap();
    assert_eq!(entries.entries.len(), 1);
    check_data_header_footer(&cfs, &id, &entries).unwrap();
    assert_eq!(
        open_input(&cfs, &entries, ".fnm").unwrap().as_slice(),
        fnm.as_slice()
    );
    assert_eq!(
        entries.get(".fnm").unwrap().offset % ALIGNMENT_BYTES as i64,
        0
    );
}

#[test]
fn write_zero_length_body_sub_file_round_trips() {
    // A sub-file whose own body is empty (header+footer only, e.g. a
    // degenerate all-empty doc-values file) -- its recorded length is
    // just header+footer, and the *next* sub-file's offset must start
    // right after it with no gap or off-by-one, aligned as usual.
    let id = [4u8; ID_LENGTH];
    let empty_dvd = build_sub_file("DocValues", 0, &id, b"");
    let fnm = build_sub_file("FieldInfos", 1, &id, b"non-empty body");

    let sub_files = vec![
        (".dvd".to_string(), empty_dvd.clone()),
        (".fnm".to_string(), fnm.clone()),
    ];
    let (cfs, cfe) = write(&id, &sub_files).unwrap();
    let entries = parse_entries(&cfe, &id).unwrap();
    check_data_header_footer(&cfs, &id, &entries).unwrap();

    let dvd_entry = entries.get(".dvd").unwrap();
    assert_eq!(dvd_entry.length as usize, empty_dvd.len());
    assert_eq!(
        open_input(&cfs, &entries, ".dvd").unwrap().as_slice(),
        empty_dvd.as_slice()
    );
    assert_eq!(
        open_input(&cfs, &entries, ".fnm").unwrap().as_slice(),
        fnm.as_slice()
    );
}

#[test]
fn write_many_sub_files_stresses_entry_count_vint_boundary() {
    // 200 sub-files pushes the `.cfe` entry-count vint from 1 byte to 2
    // bytes (boundary at 128) and stresses the alignment/offset math
    // across many small entries -- a scale no real segment's handful of
    // sub-files would ever reach.
    let id = [5u8; ID_LENGTH];
    let mut sub_files = Vec::new();
    for i in 0..200 {
        let name = format!(".x{i}");
        let body = vec![i as u8; (i % 7) as usize];
        sub_files.push((name, build_sub_file("Misc", 0, &id, &body)));
    }
    let (cfs, cfe) = write(&id, &sub_files).unwrap();
    let entries = parse_entries(&cfe, &id).unwrap();
    assert_eq!(entries.entries.len(), 200);
    check_data_header_footer(&cfs, &id, &entries).unwrap();

    for (name, original) in &sub_files {
        assert_eq!(
            open_input(&cfs, &entries, name).unwrap().as_slice(),
            original.as_slice(),
            "mismatch for {name}"
        );
    }
    for (_, entry) in &entries.entries {
        assert_eq!(entry.offset % ALIGNMENT_BYTES as i64, 0);
    }
}

#[test]
fn write_unusual_sub_file_names_round_trip() {
    // Codec-suffix-shaped names (e.g. `_Lucene104_0.doc`) are what real
    // segments actually produce (see fixtures/data/compound_index's
    // manifest); also check an empty name and a non-ASCII one to be sure
    // the id is treated as an opaque string, not parsed/validated.
    let id = [6u8; ID_LENGTH];
    let names = ["_Lucene104_0.doc", "_Lucene90_0.dvd", "", "héllo.tïm"];
    let sub_files: Vec<(String, Vec<u8>)> = names
        .iter()
        .enumerate()
        .map(|(i, name)| {
            (
                name.to_string(),
                build_sub_file("Misc", 0, &id, format!("body {i}").as_bytes()),
            )
        })
        .collect();

    let (cfs, cfe) = write(&id, &sub_files).unwrap();
    let entries = parse_entries(&cfe, &id).unwrap();
    for (name, original) in &sub_files {
        assert_eq!(
            open_input(&cfs, &entries, name).unwrap().as_slice(),
            original.as_slice(),
            "mismatch for {name:?}"
        );
    }
}

#[test]
fn write_duplicate_sub_file_names_detected_on_read() {
    // `write` itself doesn't dedupe by name (that's a caller-correctness
    // invariant, not something the packer can/should second-guess) but
    // the resulting `.cfe` must still be caught as corrupt/ambiguous by
    // `parse_entries` rather than silently letting the second entry
    // shadow the first.
    let id = [7u8; ID_LENGTH];
    let a = build_sub_file("Misc", 0, &id, b"aaa");
    let b = build_sub_file("Misc", 0, &id, b"bbbbb");
    let (_cfs, cfe) = write(&id, &[(".fnm".to_string(), a), (".fnm".to_string(), b)]).unwrap();

    assert!(matches!(
        parse_entries(&cfe, &id),
        Err(Error::DuplicateEntry(name)) if name == ".fnm"
    ));
}
