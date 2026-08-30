//! Differential tests against Java-written `.si` files.
//! Regenerate with fixtures/src/GenSegmentInfo.java.
// Test-support code opts out of the arithmetic gate at the file boundary:
// the gate exists for values read off disk in production decode paths, not
// for a fixture builder's own index arithmetic. See
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use lucene_index::segment_info;

fn fixture_dir() -> String {
    concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/data/").to_string()
}

struct Manifest {
    kv: Vec<(String, String)>,
}

impl Manifest {
    fn load(segment: &str) -> Self {
        let text =
            std::fs::read_to_string(format!("{}{}.manifest.properties", fixture_dir(), segment))
                .expect("run fixtures generator first (GenSegmentInfo)");
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

    fn get_i32(&self, key: &str) -> i32 {
        self.get(key).parse().unwrap()
    }
}

fn id_from_hex(hex: &str) -> [u8; 16] {
    let mut id = [0u8; 16];
    for i in 0..16 {
        id[i] = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap();
    }
    id
}

fn check_segment(segment: &str) {
    let manifest = Manifest::load(segment);
    let buf = std::fs::read(format!("{}{}.si", fixture_dir(), segment)).unwrap();
    let id = id_from_hex(manifest.get("id_hex"));

    let si = segment_info::parse(&buf, &id).unwrap();

    assert_eq!(si.version.major, manifest.get_i32("version_major"));
    assert_eq!(si.version.minor, manifest.get_i32("version_minor"));
    assert_eq!(si.version.bugfix, manifest.get_i32("version_bugfix"));

    if manifest.get_i32("has_min_version") == 1 {
        let mv = si.min_version.expect("expected min_version to be present");
        assert_eq!(mv.major, manifest.get_i32("min_version_major"));
        assert_eq!(mv.minor, manifest.get_i32("min_version_minor"));
        assert_eq!(mv.bugfix, manifest.get_i32("min_version_bugfix"));
    } else {
        assert!(si.min_version.is_none());
    }

    assert_eq!(si.doc_count, manifest.get_i32("doc_count"));
    assert_eq!(
        si.is_compound_file,
        manifest.get_i32("is_compound_file") == 1
    );
    assert_eq!(si.has_blocks, manifest.get_i32("has_blocks") == 1);

    let mut expected_diag: Vec<(String, String)> = manifest
        .get("diagnostics")
        .split(';')
        .filter(|kv| !kv.is_empty())
        .map(|kv| {
            let (k, v) = kv.split_once('=').unwrap();
            (k.to_string(), v.to_string())
        })
        .collect();
    let mut actual_diag = si.diagnostics.clone();
    expected_diag.sort();
    actual_diag.sort();
    assert_eq!(actual_diag, expected_diag);

    let mut expected_attrs: Vec<(String, String)> = manifest
        .get("attributes")
        .split(';')
        .filter(|kv| !kv.is_empty())
        .map(|kv| {
            let (k, v) = kv.split_once('=').unwrap();
            (k.to_string(), v.to_string())
        })
        .collect();
    let mut actual_attrs = si.attributes.clone();
    expected_attrs.sort();
    actual_attrs.sort();
    assert_eq!(actual_attrs, expected_attrs);

    // Lucene99SegmentInfoFormat.write() adds the `.si` file itself to SegmentInfo's
    // file set before writing (see Lucene99SegmentInfoFormat.write, "Only add the
    // file once we've successfully created it"), so the persisted set is the
    // manifest's list plus the `.si` file.
    let manifest_files = manifest.get("files");
    let mut expected_files: Vec<String> = manifest_files
        .split(',')
        .filter(|f| !f.is_empty())
        .map(String::from)
        .chain(std::iter::once(format!("{segment}.si")))
        .collect();
    let mut actual_files = si.files.clone();
    expected_files.sort();
    actual_files.sort();
    assert_eq!(actual_files, expected_files);

    // Byte-exact re-encode. `parse` + `write` are documented as exact
    // inverses; asserting it against real Lucene bytes (rather than only
    // round-tripping our own output) is what actually pins the write path to
    // Lucene's -- it is how the `SegmentInfo.NO == -1` (0xFF, not 0) encoding
    // of `isCompoundFile`/`hasBlocks` was caught.
    assert_eq!(
        segment_info::write(&si, ""),
        buf,
        "re-encoding the Java-written {segment}.si must reproduce it byte for byte"
    );
}

#[test]
fn segment_with_min_version() {
    check_segment("_0");
}

#[test]
fn segment_without_min_version() {
    check_segment("_1");
}

#[test]
fn wrong_segment_id_rejected() {
    let buf = std::fs::read(format!("{}_0.si", fixture_dir())).unwrap();
    let wrong_id = [0u8; 16];
    assert!(segment_info::parse(&buf, &wrong_id).is_err());
}

/// A real-Lucene-written *index-sorted* `.si`. This is the only fixture that
/// exercises `SortFieldProvider`'s byte layout (provider name string, then
/// that provider's own bytestream) rather than the `.si` body's own fields --
/// the encoding this port previously invented from scratch. Two sort fields
/// with opposite `reverse`/missing policies, so neither flag nor the priority
/// order can be silently dropped.
#[test]
fn index_sorted_segment_matches_real_lucene_sort_field_provider_bytes() {
    check_segment("_2");

    let manifest = Manifest::load("_2");
    let buf = std::fs::read(format!("{}_2.si", fixture_dir())).unwrap();
    let id = id_from_hex(manifest.get("id_hex"));
    let si = segment_info::parse(&buf, &id).unwrap();

    let rendered: Vec<String> = si
        .index_sort
        .expect("_2 is an index-sorted segment")
        .iter()
        .map(|f| {
            format!(
                "{}:{}:{}",
                f.field,
                if f.reverse { 1 } else { 0 },
                render_long_sentinel(f)
            )
        })
        .collect();
    assert_eq!(rendered.join(","), manifest.get("index_sort"));
}

/// `GenSegmentInfo`'s manifest spells the two `LONG` sentinels as
/// `first`/`last`, which is all it needs: both sort fields are
/// `SortField(field, LONG, reverse)` with `Long.MIN_VALUE`/`Long.MAX_VALUE`.
/// Anything else is a fixture/model mismatch worth failing loudly on.
fn render_long_sentinel(f: &segment_info::IndexSortField) -> &'static str {
    match &f.kind {
        segment_info::IndexSortKind::Numeric(segment_info::NumericSortKey::Long(Some(v))) => {
            if *v == i64::MIN {
                "first"
            } else if *v == i64::MAX {
                "last"
            } else {
                panic!("unexpected LONG missing value {v}")
            }
        }
        other => panic!("unexpected sort kind {other:?}"),
    }
}

/// The bytes this port *writes* for the same sort must be byte-identical to
/// the ones real Lucene wrote for `_2` -- not merely round-trippable through
/// our own parser. Comparing the two files' sort-field regions directly is
/// the strongest available statement of write-path fidelity without a JVM
/// (`scripts/verify-write-path.sh`'s `VerifySegmentInfo` covers the JVM side).
#[test]
fn our_writer_emits_the_same_sort_field_bytes_as_lucene() {
    let manifest = Manifest::load("_2");
    let java_bytes = std::fs::read(format!("{}_2.si", fixture_dir())).unwrap();
    let id = id_from_hex(manifest.get("id_hex"));
    let si = segment_info::parse(&java_bytes, &id).unwrap();
    let ours = segment_info::write(&si, "");
    assert_eq!(
        ours, java_bytes,
        "re-encoding a Lucene-written index-sorted .si must reproduce it byte for byte"
    );
}
