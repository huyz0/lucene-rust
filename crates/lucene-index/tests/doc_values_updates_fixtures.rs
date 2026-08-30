//! Differential test against a real index whose doc-values were **updated in
//! place** by an actual `IndexWriter` (`updateNumericDocValue` /
//! `updateBinaryDocValue`), across three update rounds. Regenerate with
//! `fixtures/src/GenDocValuesUpdates.java`.
//!
//! A doc-values update is the one write this port used to do in a format of
//! its own invention. What Lucene actually writes is the updated field's
//! **whole column** in a new generation of ordinary
//! `Lucene90DocValuesFormat` files, named
//! `_<segment>_<base36 gen>_Lucene90_0.{dvm,dvd,dvs}`, plus a `FieldInfos`
//! generation `_<segment>_<base36 gen>.fnm` recording that field's
//! `FieldInfo.docValuesGen`, plus `docValuesGen` and a per-field
//! `dvUpdatesFiles` map in `segments_N`.
//!
//! The fixture deliberately carries three doc-values fields in three different
//! states, because resolving *every* field to the newest generation reads back
//! plausibly and is wrong:
//!
//! | field  | type    | state |
//! |--------|---------|-------|
//! | `val`  | NUMERIC | updated twice -- generation 3, and generation 1's files are gone |
//! | `tag`  | BINARY  | updated once -- generation 2 |
//! | `keep` | NUMERIC | never updated -- generation -1, still the base column |
//!
//! Every expected value in the manifest is what real Lucene's own
//! `DirectoryReader` reads back, so the assertions are against Lucene's
//! answers rather than a second derivation of the format.
// Test-support code opts out of the arithmetic gate at the file boundary:
// the gate exists for values read off disk in production decode paths, not
// for a fixture builder's own index arithmetic. See
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::doc_values;
use lucene_codecs::field_infos::{self, FieldInfos};
use lucene_index::field_updates;
use lucene_index::{segment_info, segment_infos};
use lucene_store::codec_util::ID_LENGTH;

fn dir() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/doc_values_updates_index/"
    )
    .to_string()
}

struct Manifest {
    kv: Vec<(String, String)>,
}

impl Manifest {
    fn load() -> Self {
        let text = std::fs::read_to_string(format!("{}manifest.properties", dir()))
            .expect("run fixtures generator first (GenDocValuesUpdates)");
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

    fn get_i64(&self, key: &str) -> i64 {
        self.get(key).parse().unwrap()
    }

    /// A comma-separated per-doc column; an empty cell means "no value".
    fn column(&self, key: &str) -> Vec<Option<String>> {
        self.get(key)
            .split(',')
            .map(|c| {
                if c.is_empty() {
                    None
                } else {
                    Some(c.to_string())
                }
            })
            .collect()
    }
}

fn segment_id(manifest: &Manifest) -> [u8; ID_LENGTH] {
    let hex = manifest.get("id_hex");
    let mut id = [0u8; ID_LENGTH];
    for (i, slot) in id.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap();
    }
    id
}

fn read(name: &str) -> Vec<u8> {
    std::fs::read(format!("{}{name}", dir())).unwrap_or_else(|e| panic!("read {name}: {e}"))
}

/// `IndexWriter.readFieldInfos`: the generational `.fnm` when the segment has
/// field updates, which is the only one recording the updated fields'
/// `docValuesGen`.
fn newest_field_infos(sci: &segment_infos::SegmentCommitInfo, id: &[u8; ID_LENGTH]) -> FieldInfos {
    assert_ne!(sci.field_infos_gen, -1, "the fixture has field updates");
    let name = field_updates::field_infos_gen_file_name(&sci.segment_name, sci.field_infos_gen);
    let suffix = lucene_util::base36::to_base36(sci.field_infos_gen);
    field_infos::parse(&read(&name), id, &suffix).expect("parse generational .fnm")
}

/// `SegmentDocValuesProducer`: open the `(meta, data)` pair holding
/// `field`'s current column -- its own generation when it has one, the base
/// `.dvm`/`.dvd` otherwise.
fn open_column(
    sci: &segment_infos::SegmentCommitInfo,
    id: &[u8; ID_LENGTH],
    infos: &FieldInfos,
    si_files: &[String],
    field_name: &str,
) -> (doc_values::DocValuesMeta, Vec<u8>, i32) {
    let field = infos
        .fields
        .iter()
        .find(|f| f.name == field_name)
        .unwrap_or_else(|| panic!("no field {field_name}"))
        .clone();
    let number = field.number;

    if field.doc_values_gen != -1 {
        let gen = field.doc_values_gen;
        // The per-field component comes out of the field's own `.fnm`
        // attributes, exactly as `PerFieldDocValuesFormat.FieldsReader` reads
        // it -- not hardcoded, so a fixture written with a different suffix
        // would still resolve.
        let attr = |key: &str| {
            field
                .attributes
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v.clone())
                .unwrap_or_else(|| panic!("{field_name} has no {key} attribute"))
        };
        let per_field = format!(
            "{}_{}",
            attr("PerFieldDocValuesFormat.format"),
            attr("PerFieldDocValuesFormat.suffix")
        );
        let suffix = field_updates::generation_segment_suffix(gen, &per_field);
        let meta_name =
            field_updates::generation_file_name(&sci.segment_name, gen, &per_field, "dvm");
        let data_name =
            field_updates::generation_file_name(&sci.segment_name, gen, &per_field, "dvd");
        let only = FieldInfos {
            fields: vec![field],
        };
        let (_, meta) = doc_values::parse_meta(&read(&meta_name), id, &suffix, &only)
            .unwrap_or_else(|e| panic!("parse {meta_name}: {e}"));
        let _ = number;
        return (meta, read(&data_name), max_doc(sci, id, si_files));
    }

    let meta_name = si_files.iter().find(|f| f.ends_with(".dvm")).unwrap();
    let data_name = si_files.iter().find(|f| f.ends_with(".dvd")).unwrap();
    let suffix = meta_name
        .strip_prefix(&format!("{}_", sci.segment_name))
        .and_then(|s| s.strip_suffix(".dvm"))
        .unwrap();
    let (_, meta) = doc_values::parse_meta(&read(meta_name), id, suffix, infos)
        .unwrap_or_else(|e| panic!("parse {meta_name}: {e}"));
    (meta, read(data_name), max_doc(sci, id, si_files))
}

fn max_doc(
    sci: &segment_infos::SegmentCommitInfo,
    id: &[u8; ID_LENGTH],
    _si_files: &[String],
) -> i32 {
    let si = segment_info::parse(&read(&format!("{}.si", sci.segment_name)), id).unwrap();
    si.doc_count
}

fn load() -> (
    Manifest,
    segment_infos::SegmentCommitInfo,
    [u8; ID_LENGTH],
    FieldInfos,
    Vec<String>,
) {
    let manifest = Manifest::load();
    let id = segment_id(&manifest);
    let segments_file_name = manifest.get("segments_file_name").to_string();
    let generation = segments_file_name
        .strip_prefix("segments_")
        .map(|g| i64::from_str_radix(g, 36).unwrap())
        .unwrap();
    let sis = segment_infos::parse(&read(&segments_file_name), generation).unwrap();
    assert_eq!(sis.segments.len(), 1);
    let sci = sis.segments[0].clone();
    let si = segment_info::parse(&read(&format!("{}.si", sci.segment_name)), &id).unwrap();
    let infos = newest_field_infos(&sci, &id);
    (manifest, sci, id, infos, si.files)
}

#[test]
fn segments_n_records_the_generations_and_files_real_lucene_wrote() {
    let (manifest, sci, _id, _infos, si_files) = load();

    assert_eq!(sci.doc_values_gen, manifest.get_i64("doc_values_gen"));
    assert_eq!(sci.field_infos_gen, manifest.get_i64("field_infos_gen"));
    assert_eq!(
        sci.field_infos_files,
        manifest
            .get("field_infos_files")
            .split(',')
            .map(str::to_string)
            .collect::<Vec<_>>()
    );

    // Per-field doc-values update files, keyed by field number. Two entries:
    // `val` and `tag`. `keep` was never updated and must have none -- a reader
    // that recorded an entry for every doc-values field would still pass every
    // structural check and read `keep` from a file that does not exist.
    let mut expected_fields: Vec<i32> = manifest
        .get("dv_update_fields")
        .split(',')
        .map(|n| n.parse().unwrap())
        .collect();
    expected_fields.sort_unstable();
    let mut got_fields: Vec<i32> = sci.dv_update_files.iter().map(|(n, _)| *n).collect();
    got_fields.sort_unstable();
    assert_eq!(got_fields, expected_fields);

    for &number in &expected_fields {
        let mut got: Vec<String> = sci
            .dv_update_files
            .iter()
            .find(|(n, _)| *n == number)
            .unwrap()
            .1
            .clone();
        got.sort();
        let expected: Vec<String> = manifest
            .get(&format!("dv_update_files.{number}"))
            .split(',')
            .map(str::to_string)
            .collect();
        assert_eq!(got, expected, "field {number}'s update files");
    }

    // `SegmentCommitInfo.files()` must name every generational file: none of
    // them is in the `.si` (they did not exist when it was written), so a
    // deleter or checksum walk that only reads the `.si` skips exactly the
    // files an update round produced.
    let files = sci.files(&si_files);
    for (_, names) in &sci.dv_update_files {
        for name in names {
            assert!(files.contains(name), "{name} missing from files()");
        }
    }
    for name in &sci.field_infos_files {
        assert!(files.contains(name), "{name} missing from files()");
    }
}

#[test]
fn the_generational_field_infos_records_each_fields_own_doc_values_generation() {
    let (manifest, _sci, _id, infos, _si_files) = load();
    for field in &infos.fields {
        let expected = manifest.get_i64(&format!("field_dv_gen.{}", field.name));
        assert_eq!(
            field.doc_values_gen, expected,
            "{}'s docValuesGen in the generational .fnm",
            field.name
        );
        let expected_number = manifest.get_i64(&format!("field_number.{}", field.name)) as i32;
        assert_eq!(field.number, expected_number);
    }
    // Each updated field takes its *own* generation -- `handleDVUpdates` calls
    // `advanceDocValuesGen()` inside its per-field loop, not once around it, so
    // two fields updated in different rounds never share a number.
    let val = infos.fields.iter().find(|f| f.name == "val").unwrap();
    let tag = infos.fields.iter().find(|f| f.name == "tag").unwrap();
    assert_ne!(val.doc_values_gen, tag.doc_values_gen);
}

#[test]
fn every_documents_value_matches_what_real_lucene_reads_back() {
    let (manifest, sci, id, infos, si_files) = load();
    let max = manifest.get_i64("max_doc") as i32;

    for field_name in ["val", "keep"] {
        let (meta, data, max_doc) = open_column(&sci, &id, &infos, &si_files, field_name);
        assert_eq!(max_doc, max);
        let number = infos
            .fields
            .iter()
            .find(|f| f.name == field_name)
            .unwrap()
            .number;
        let entry = meta
            .numeric_entry(number)
            .unwrap_or_else(|| panic!("{field_name} has no NUMERIC entry"));
        let expected = manifest.column(&format!("expected_{field_name}"));
        assert_eq!(expected.len(), max as usize);
        for doc in 0..max {
            let got = doc_values::numeric_value(&data, entry, doc).unwrap();
            let want = expected[doc as usize]
                .as_ref()
                .map(|v| v.parse::<i64>().unwrap());
            assert_eq!(got, want, "{field_name} doc {doc}");
        }
    }

    let (meta, data, _) = open_column(&sci, &id, &infos, &si_files, "tag");
    let number = infos
        .fields
        .iter()
        .find(|f| f.name == "tag")
        .unwrap()
        .number;
    let entry = meta.binary_entry(number).expect("tag has no BINARY entry");
    let expected = manifest.column("expected_tag");
    for doc in 0..max {
        let got = doc_values::binary_value(&data, entry, doc)
            .unwrap()
            .map(|v| String::from_utf8(v.to_vec()).unwrap());
        assert_eq!(got, expected[doc as usize], "tag doc {doc}");
    }
}

#[test]
fn a_never_updated_field_still_resolves_to_the_base_column() {
    // `keep` is at generation -1, so it must come out of the *base*
    // `.dvm`/`.dvd` even though the segment has two update generations. This
    // is the case a reader gets wrong by resolving every field to
    // `SegmentCommitInfo.docValuesGen` instead of `FieldInfo.docValuesGen`.
    let (_manifest, sci, id, infos, si_files) = load();
    let keep = infos.fields.iter().find(|f| f.name == "keep").unwrap();
    assert_eq!(keep.doc_values_gen, -1);

    let (meta, data, max_doc) = open_column(&sci, &id, &infos, &si_files, "keep");
    let entry = meta.numeric_entry(keep.number).unwrap();
    // The base column also still holds `val`'s and `tag`'s superseded values,
    // which is exactly why it is not deleted.
    assert!(meta.numeric_entry(2).is_some(), "base .dvm still holds val");
    for doc in 0..max_doc {
        assert_eq!(
            doc_values::numeric_value(&data, entry, doc).unwrap(),
            Some(1000 + doc as i64)
        );
    }
}
