//! Differential tests against Java-written `.si` files.
//! Regenerate with fixtures/src/GenSegmentInfo.java.

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
    let mut expected_files: Vec<String> = manifest
        .get("files")
        .split(',')
        .map(String::from)
        .chain(std::iter::once(format!("{segment}.si")))
        .collect();
    let mut actual_files = si.files.clone();
    expected_files.sort();
    actual_files.sort();
    assert_eq!(actual_files, expected_files);
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
