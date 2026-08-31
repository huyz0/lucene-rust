//! **A block jump table real Lucene wrote, read by this port.**
//!
//! `c39-codecs-readpath` ported `IndexedDISI`'s block jump table in both
//! directions and proved the *write* side against real Lucene
//! (`VerifySparseNumericDocValues`' block-jump pass, two independent negative
//! controls). The read side had never run over bytes Lucene wrote, for a
//! fixture reason rather than a design one: `IndexedDISI.writeBitSet` emits
//! `jumpTableEntryCount = 0` for anything under two logical 65 536-document
//! blocks, and the largest Java-written index in this tree was 36 000
//! documents. So there was no Java-written table to read (ledger item 23c).
//!
//! That matters because this sweep has twice found a writer and a reader
//! agreeing on a *shared* mistake -- the FST framing bug and the invented
//! `.si` sort encoding -- both in formats where only one direction had been
//! checked.
//!
//! `fixtures/src/GenDisiJumpTable.java` writes 200 000 documents with a
//! numeric doc-values field on every third one (DENSE blocks, four of them)
//! and another on every 20 000th (SPARSE blocks, with the last logical block
//! empty so `flushBlockJumps`' empty-block fill is exercised).
//!
//! The access pattern under test is the **cold seek**:
//! `Lucene90DocValuesProducer`'s single-lookup shape opens a cursor and asks
//! one question, and `IndexedDISI.advanceBlock` only consults the table when
//! the target is at least two blocks ahead. A forward scan never touches it,
//! which is why the scan assertions here are a complement rather than the
//! point.
// Test-support code opts out of the arithmetic gate at the file boundary. See
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::{doc_values as ndv, field_infos};

fn dir() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/disi_jump_table_index/"
    )
    .to_string()
}

struct Manifest {
    kv: Vec<(String, String)>,
}

impl Manifest {
    fn load() -> Self {
        let text = std::fs::read_to_string(format!("{}manifest.properties", dir()))
            .expect("run scripts/gen-fixtures.sh --only GenDisiJumpTable first");
        Manifest {
            kv: text
                .lines()
                .filter_map(|l| l.split_once('='))
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        }
    }

    fn get(&self, key: &str) -> &str {
        self.kv
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
            .unwrap_or_else(|| panic!("manifest key {key} missing"))
    }

    fn field_number(&self, field: &str) -> i32 {
        self.get("field_numbers")
            .split(',')
            .find_map(|kv| {
                let (name, num) = kv.split_once(':').unwrap();
                (name == field).then(|| num.parse().unwrap())
            })
            .unwrap_or_else(|| panic!("field {field} missing from field_numbers"))
    }

    fn probes(&self) -> Vec<i32> {
        self.get("probes")
            .split(',')
            .map(|p| p.parse().unwrap())
            .collect()
    }

    fn probe_values(&self, field: &str) -> Vec<Option<i64>> {
        self.get(&format!("field.{field}.probe_values"))
            .split(',')
            .map(|v| (v != "NONE").then(|| v.parse().unwrap()))
            .collect()
    }
}

fn id_from_hex(hex: &str) -> [u8; 16] {
    let mut id = [0u8; 16];
    for (i, slot) in id.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap();
    }
    id
}

/// `Lucene90DocValuesFormat` is wrapped in a `PerFieldDocValuesFormat`, which
/// gives each format instance its own segment suffix on top of the segment's
/// own (empty) one -- derive it from the real filename.
fn dv_suffix(m: &Manifest) -> String {
    let segment_name = m.get("segment_name");
    let name = m.get("dvm_file_name");
    name.strip_prefix(&format!("{segment_name}_"))
        .and_then(|s| s.strip_suffix(".dvm"))
        .unwrap_or_else(|| panic!("unexpected dvm file name shape: {name}"))
        .to_string()
}

struct Fixture {
    manifest: Manifest,
    data: Vec<u8>,
    parsed: ndv::DocValuesMeta,
}

fn open() -> Fixture {
    let manifest = Manifest::load();
    let id = id_from_hex(manifest.get("id_hex"));
    let fnm = std::fs::read(format!("{}{}.raw", dir(), manifest.get("fnm_file_name"))).unwrap();
    let fis = field_infos::parse(&fnm, &id, "").unwrap();
    let meta = std::fs::read(format!("{}{}.raw", dir(), manifest.get("dvm_file_name"))).unwrap();
    let data = std::fs::read(format!("{}{}.raw", dir(), manifest.get("dvd_file_name"))).unwrap();
    let suffix = dv_suffix(&manifest);
    let (_, parsed) = ndv::parse_meta(&meta, &id, &suffix, &fis).unwrap();
    Fixture {
        manifest,
        data,
        parsed,
    }
}

/// The fixture is only worth anything if Lucene actually emitted a table.
/// Pin that first, so a regenerated corpus that fell below two blocks fails
/// here rather than silently making every test below vacuous.
#[test]
fn real_lucene_emitted_a_block_jump_table_for_both_fields() {
    let f = open();
    let max_doc: i64 = f.manifest.get("max_doc").parse().unwrap();
    assert!(max_doc > 2 * 65536, "under two blocks there is no table");
    for field in ["sparse", "very_sparse"] {
        let entry = f
            .parsed
            .numeric_entry(f.manifest.field_number(field))
            .unwrap_or_else(|| panic!("{field} has a numeric entry"));
        assert!(!entry.is_dense(), "{field} must take the IndexedDISI path");
        assert!(!entry.is_empty_field());
        assert!(
            entry.jump_table_entry_count > 2,
            "{field}: real Lucene wrote jumpTableEntryCount={}, which is not a table \
             spanning several blocks",
            entry.jump_table_entry_count
        );
    }
}

/// One **cold** lookup per probe -- a fresh single-value read, which is
/// `Lucene90DocValuesProducer`'s own shape and the only access pattern where
/// `advanceBlock` consults the table.
#[test]
fn cold_seeks_through_a_java_written_jump_table_match_lucenes_values() {
    let f = open();
    let probes = f.manifest.probes();
    assert!(probes.len() > 20, "the probe set shrank");
    let mut present = 0usize;
    // Each `numeric_value` opens a *fresh* cursor at block 0, so a probe two or
    // more logical blocks in is one `advanceBlock` call that takes the
    // jump-table branch (`block_index >= (self.block >> 16) + 2`).
    let mut used_the_table = 0usize;
    for field in ["sparse", "very_sparse"] {
        let entry = f
            .parsed
            .numeric_entry(f.manifest.field_number(field))
            .unwrap();
        let expected = f.manifest.probe_values(field);
        assert_eq!(expected.len(), probes.len());
        for (&doc, want) in probes.iter().zip(&expected) {
            let got = ndv::numeric_value(&f.data, entry, doc).unwrap();
            assert_eq!(
                got, *want,
                "{field}: doc {doc} -- real Lucene says {want:?}, this port {got:?}"
            );
            if want.is_some() {
                present += 1;
            }
            if doc >> 16 >= 2 {
                used_the_table += 1;
            }
        }
    }
    assert!(present > 10, "too few present probes to prove anything");
    assert!(
        used_the_table > 10,
        "only {used_the_table} probes were two or more blocks in, so `advanceBlock`'s \
         jump-table branch was barely exercised"
    );
}

/// A forward walk over every present document, which never consults the table
/// -- the complement to the cold seeks, pinning that the trailing table bytes
/// do not confuse sequential decoding.
#[test]
fn a_full_scan_of_a_java_written_column_matches_lucenes_cardinality_and_checksum() {
    let f = open();
    let max_doc: i32 = f.manifest.get("max_doc").parse().unwrap();
    for field in ["sparse", "very_sparse"] {
        let entry = f
            .parsed
            .numeric_entry(f.manifest.field_number(field))
            .unwrap();
        let mut reader = ndv::NumericReader::new(&f.data, entry);
        let mut count = 0i64;
        let mut checksum = 0i64;
        for doc in 0..max_doc {
            if let Some(v) = reader.value(doc).unwrap() {
                count += 1;
                checksum = checksum.wrapping_mul(31).wrapping_add(v);
            }
        }
        assert_eq!(
            count,
            f.manifest
                .get(&format!("field.{field}.count"))
                .parse::<i64>()
                .unwrap(),
            "{field}: cardinality"
        );
        assert_eq!(
            checksum,
            f.manifest
                .get(&format!("field.{field}.checksum"))
                .parse::<i64>()
                .unwrap(),
            "{field}: value checksum"
        );
    }
}

/// **The negative control**: perturb the table Lucene wrote, and the answers
/// must change. Without it every assertion above would still pass over a
/// reader that ignored the table entirely and walked the block headers -- and
/// "the reader ignores the table" is precisely the state this fixture exists
/// to rule out.
///
/// Two independent perturbations, because the table's two `i32` halves fail
/// differently: the *index* (the cardinality before the block, which seeds the
/// ordinal) and the *offset* (where the block header starts). c39 ruled both
/// out in the write direction; this is the read direction.
#[test]
fn corrupting_the_java_written_jump_table_changes_the_answer() {
    let f = open();
    let entry = f
        .parsed
        .numeric_entry(f.manifest.field_number("sparse"))
        .unwrap();
    // The table trails the block payloads inside the recorded region, so its
    // first byte is `docsWithFieldLength - 8 * jumpTableEntryCount` in.
    let region_start = entry.docs_with_field_offset as usize;
    let region_end = region_start + entry.docs_with_field_length as usize;
    let table_start = region_end - 8 * entry.jump_table_entry_count as usize;
    // The third block's entry: far enough in that a probe there has to have
    // jumped, and not the sentinel.
    let third_entry = table_start + 2 * 8;

    // A probe well inside the third block, whose value Lucene recorded.
    let probes = f.manifest.probes();
    let values = f.manifest.probe_values("sparse");
    let (&doc, want) = probes
        .iter()
        .zip(&values)
        .find(|(&d, v)| d >> 16 == 2 && v.is_some())
        .expect("a present probe in the third block");
    assert_eq!(
        ndv::numeric_value(&f.data, entry, doc).unwrap(),
        *want,
        "baseline"
    );

    for (label, byte_offset) in [("index", third_entry), ("offset", third_entry + 4)] {
        let mut data = f.data.clone();
        data[byte_offset] ^= 0x11;
        let got = ndv::numeric_value(&data, entry, doc);
        assert!(
            got.is_err() || got.unwrap() != *want,
            "{label}: perturbing the jump table left the answer unchanged, so nothing read it"
        );
    }
}
