//! Differential test against a real `.dvm`/`.dvd` pair whose single NUMERIC
//! field trips `Lucene90DocValuesConsumer.writeValues`'s `doBlocks`
//! varying-bits-per-value split: two full 16384-value blocks with very
//! different value ranges (so per-block widths differ sharply from the
//! whole-field width) plus a trailing partial block.
//! Regenerate with `fixtures/src/GenDocValuesVaryingBpv.java`.
// Test-support code opts out of the arithmetic gate at the file boundary:
// the gate exists for values read off disk in production decode paths, not
// for a fixture builder's own index arithmetic. See
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::{doc_values as ndv, field_infos};

fn dir() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/doc_values_varying_bpv/"
    )
    .to_string()
}

struct Manifest {
    kv: Vec<(String, String)>,
}

impl Manifest {
    fn load() -> Self {
        let text = std::fs::read_to_string(format!("{}manifest.properties", dir()))
            .expect("run fixtures generator first (GenDocValuesVaryingBpv)");
        let kv = text
            .lines()
            .filter_map(|l| l.split_once('='))
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        Manifest { kv }
    }

    fn get(&self, key: &str) -> &str {
        self.kv
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
            .unwrap_or_else(|| panic!("manifest key {key} missing"))
    }
}

fn id_from_hex(hex: &str) -> [u8; 16] {
    let mut id = [0u8; 16];
    for i in 0..16 {
        id[i] = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap();
    }
    id
}

fn load_field_infos(manifest: &Manifest, id: &[u8; 16]) -> field_infos::FieldInfos {
    let buf = std::fs::read(format!("{}{}.raw", dir(), manifest.get("fnm_file_name"))).unwrap();
    field_infos::parse(&buf, id, "").unwrap()
}

/// `Lucene90DocValuesFormat` is wrapped in a `PerFieldDocValuesFormat`, which
/// gives each format instance its own segment-suffix on top of the
/// segment's own (empty) suffix -- derive it from the real filename rather
/// than hardcoding it, since the counter can vary.
fn dv_suffix(manifest: &Manifest) -> String {
    let segment_name = manifest.get("segment_name");
    let name = manifest.get("dvm_file_name");
    name.strip_prefix(&format!("{segment_name}_"))
        .and_then(|s| s.strip_suffix(".dvm"))
        .unwrap_or_else(|| panic!("unexpected dvm file name shape: {name}"))
        .to_string()
}

fn field_number(manifest: &Manifest, field: &str) -> i32 {
    manifest
        .get("field_numbers")
        .split(',')
        .find_map(|kv| {
            let (name, num) = kv.split_once(':').unwrap();
            (name == field).then(|| num.parse().unwrap())
        })
        .unwrap_or_else(|| panic!("field {field} missing from field_numbers"))
}

#[test]
fn parses_real_varying_bpv_numeric_dv_and_matches_lucene_values() {
    let manifest = Manifest::load();
    let id = id_from_hex(manifest.get("id_hex"));
    let fis = load_field_infos(&manifest, &id);
    let meta_buf =
        std::fs::read(format!("{}{}.raw", dir(), manifest.get("dvm_file_name"))).unwrap();
    let data_buf =
        std::fs::read(format!("{}{}.raw", dir(), manifest.get("dvd_file_name"))).unwrap();

    let suffix = dv_suffix(&manifest);
    let (version, parsed) = ndv::parse_meta(&meta_buf, &id, &suffix, &fis).unwrap();
    assert_eq!(version, 2);
    let data_version = ndv::check_data_header_footer(&data_buf, &id, &suffix).unwrap();
    assert_eq!(data_version, version);

    let entry = parsed
        .numeric_entry(field_number(&manifest, "varying_bpv"))
        .unwrap();
    assert!(entry.is_dense());
    // The whole point of this fixture: confirm the writer actually took the
    // `doBlocks` path rather than silently falling back to a single width.
    assert!(
        entry.block_shift.is_some(),
        "fixture's field didn't trip doBlocks -- adjust GenDocValuesVaryingBpv's value shape"
    );

    let expected: Vec<Option<i64>> = manifest
        .get("field.varying_bpv.values")
        .split(',')
        .map(|s| {
            if s == "NONE" {
                None
            } else {
                Some(s.parse().unwrap())
            }
        })
        .collect();

    let max_doc: usize = manifest.get("max_doc").parse().unwrap();
    assert_eq!(expected.len(), max_doc);

    for (doc, &want) in expected.iter().enumerate() {
        let got = ndv::numeric_value(&data_buf, entry, doc as i32).unwrap();
        assert_eq!(got, want, "doc {doc}");
    }

    // `NumericReader` caches the decoded block between calls, where
    // `numeric_value` re-reads the jump table and block header every time.
    // Two implementations of one lookup is the shape that silently diverges,
    // so assert they agree against real Lucene-written data for every document
    // -- and in three access orders, because a cache that is only exercised
    // forwards can be wrong on the first two.
    let mut reader = ndv::NumericReader::new(&data_buf, entry);
    for (doc, &want) in expected.iter().enumerate() {
        assert_eq!(
            reader.value(doc as i32).unwrap(),
            want,
            "forward, doc {doc}"
        );
    }
    let mut reader = ndv::NumericReader::new(&data_buf, entry);
    for (doc, &want) in expected.iter().enumerate().rev() {
        assert_eq!(
            reader.value(doc as i32).unwrap(),
            want,
            "backward, doc {doc}"
        );
    }
    // Strided, so consecutive calls land in different blocks and every one is a
    // cache miss -- the path the forward walk above never takes.
    let mut reader = ndv::NumericReader::new(&data_buf, entry);
    let mut doc = 0usize;
    for _ in 0..expected.len() {
        assert_eq!(
            reader.value(doc as i32).unwrap(),
            expected[doc],
            "strided, doc {doc}"
        );
        doc = (doc + 16_384 + 1) % expected.len();
    }

    // Out-of-range must be the same error the free function raises, not a
    // silent read of whatever the cached block happens to hold.
    assert!(reader.value(max_doc as i32).is_err());
    assert!(reader.value(-1).is_err());
}

/// `NumericReader::fill_window` over varying-bits-per-value columns, dense
/// and sparse, is Lucene's value for every document of every window: whole
/// windows across the blocks, windows that are not aligned or run past the
/// last document, and a window behind the previous one (a rewind).
#[test]
fn windows_of_varying_bpv_values_are_lucenes_values() {
    let manifest = Manifest::load();
    let id = id_from_hex(manifest.get("id_hex"));
    let fis = load_field_infos(&manifest, &id);
    let meta_buf =
        std::fs::read(format!("{}{}.raw", dir(), manifest.get("dvm_file_name"))).unwrap();
    let data_buf =
        std::fs::read(format!("{}{}.raw", dir(), manifest.get("dvd_file_name"))).unwrap();
    let (_, parsed) = ndv::parse_meta(&meta_buf, &id, &dv_suffix(&manifest), &fis).unwrap();
    let max_doc: i32 = manifest.get("max_doc").parse().unwrap();
    for (field, dense) in [("varying_bpv", true), ("sparse_varying_bpv", false)] {
        let entry = parsed
            .numeric_entry(field_number(&manifest, field))
            .unwrap();
        assert_eq!(entry.is_dense(), dense, "{field}");
        assert!(
            entry.block_shift.is_some(),
            "{field} did not split into blocks"
        );
        let expected: Vec<Option<i64>> = manifest
            .get(&format!("field.{field}.values"))
            .split(',')
            .map(|s| (s != "NONE").then(|| s.parse().unwrap()))
            .collect();
        let mut windows: Vec<(i32, usize)> =
            (0..max_doc).step_by(1024).map(|b| (b, 1024)).collect();
        windows.extend([
            (3, 100),
            (16_300, 200),
            (max_doc - 10, 64),
            (0, 64),
            (5, 3000),
        ]);
        let mut reader = ndv::NumericReader::new(&data_buf, entry);
        let mut checked = 0;
        for (start, len) in windows {
            let mut values = vec![i64::MIN; len];
            let mut present = vec![0u64; len.div_ceil(64)];
            reader
                .fill_window(start, &mut values, &mut present)
                .unwrap();
            for i in 0..len {
                let doc = start as usize + i;
                let want = expected.get(doc).copied().flatten();
                let has = present[i >> 6] >> (i & 63) & 1 == 1;
                assert_eq!(
                    has.then_some(values[i]),
                    want,
                    "{field} doc {doc} window {start}+{len}"
                );
                checked += 1;
            }
        }
        assert!(checked > max_doc as usize, "{field}: {checked}");
    }
}

/// The write side: handed the fixture's two columns, the segment id and the
/// per-field suffix, this port's writer produces Lucene's `.dvm` and `.dvd`
/// byte for byte -- `writeValues`' `doBlocks` split included (both fields
/// take it; see the reader tests above).
#[test]
fn varying_bpv_columns_are_written_byte_identical_to_lucene() {
    let manifest = Manifest::load();
    let id = id_from_hex(manifest.get("id_hex"));
    let max_doc: i32 = manifest.get("max_doc").parse().unwrap();
    let parse = |field: &str| -> Vec<Option<i64>> {
        manifest
            .get(&format!("field.{field}.values"))
            .split(',')
            .map(|s| (s != "NONE").then(|| s.parse().unwrap()))
            .collect()
    };
    let dense: Vec<i64> = parse("varying_bpv")
        .into_iter()
        .map(Option::unwrap)
        .collect();
    let sparse: Vec<(i32, i64)> = parse("sparse_varying_bpv")
        .into_iter()
        .enumerate()
        .filter_map(|(doc, v)| v.map(|v| (doc as i32, v)))
        .collect();
    let (meta, data, _) = ndv::write_dense_fields(
        &[
            ndv::DenseField::Numeric(field_number(&manifest, "varying_bpv"), &dense),
            ndv::DenseField::SparseNumeric(field_number(&manifest, "sparse_varying_bpv"), &sparse),
        ],
        max_doc,
        &id,
        &dv_suffix(&manifest),
    )
    .unwrap();
    let want_meta =
        std::fs::read(format!("{}{}.raw", dir(), manifest.get("dvm_file_name"))).unwrap();
    let want_data =
        std::fs::read(format!("{}{}.raw", dir(), manifest.get("dvd_file_name"))).unwrap();
    assert_eq!(meta, want_meta, ".dvm");
    assert_eq!(data.len(), want_data.len(), ".dvd length");
    assert!(data == want_data, ".dvd bytes differ");
}
