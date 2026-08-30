//! Differential test against a real `.dvm`/`.dvd`/`.dvs` triple written by
//! an actual IndexWriter for a NUMERIC field with a doc-values skip index
//! (`NumericDocValuesField.indexedField`, 36000 docs -- comfortably past
//! the 4096-doc base interval size and the 8-interval level-1 grouping
//! threshold, so the fixture exercises both a 1-level and a 2-level skip
//! interval). Regenerate with fixtures/src/GenDocValuesSkipIndex.java.
// Test-support code opts out of the arithmetic gate at the file boundary:
// the gate exists for values read off disk in production decode paths, not
// for a fixture builder's own index arithmetic. See
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::doc_values::{self, DocValuesSkipperMeta, SkipIndexLevelInterval};
use lucene_codecs::field_infos;

fn dir() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/doc_values_skip_index/"
    )
    .to_string()
}

struct Manifest {
    kv: Vec<(String, String)>,
}

impl Manifest {
    fn load() -> Self {
        let text = std::fs::read_to_string(format!("{}manifest.properties", dir()))
            .expect("run fixtures generator first (GenDocValuesSkipIndex)");
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

/// `Lucene90DocValuesFormat` is wrapped in a `PerFieldDocValuesFormat`,
/// which gives each format its own segment-suffix on top of the segment's
/// own (empty) suffix -- derive it from the real filename (same trick as
/// the other doc-values fixture tests).
fn dv_suffix(manifest: &Manifest) -> String {
    let segment_name = manifest.get("segment_name");
    let name = manifest.get("dvm_file_name");
    name.strip_prefix(&format!("{segment_name}_"))
        .and_then(|s| s.strip_suffix(".dvm"))
        .unwrap_or_else(|| panic!("unexpected dvm file name shape: {name}"))
        .to_string()
}

/// Parses `skip.intervals` -- `level_count,minDoc:maxDoc:minVal:maxVal:docCount,...;...`
/// -- into the same shape [`doc_values::parse_skip_index`] returns, so the
/// test can compare structurally instead of string-diffing.
fn expected_intervals(manifest: &Manifest) -> Vec<Vec<SkipIndexLevelInterval>> {
    manifest
        .get("skip.intervals")
        .split(';')
        .map(|interval| {
            let mut parts = interval.split(',');
            let level_count: usize = parts.next().unwrap().parse().unwrap();
            let levels: Vec<SkipIndexLevelInterval> = parts
                .map(|level| {
                    let mut f = level.split(':');
                    let min_doc_id = f.next().unwrap().parse().unwrap();
                    let max_doc_id = f.next().unwrap().parse().unwrap();
                    let min_value = f.next().unwrap().parse().unwrap();
                    let max_value = f.next().unwrap().parse().unwrap();
                    let doc_count = f.next().unwrap().parse().unwrap();
                    SkipIndexLevelInterval {
                        min_doc_id,
                        max_doc_id,
                        min_value,
                        max_value,
                        doc_count,
                    }
                })
                .collect();
            assert_eq!(
                levels.len(),
                level_count,
                "level count mismatch in manifest"
            );
            levels
        })
        .collect()
}

#[test]
fn parses_real_numeric_skip_index_meta_and_intervals() {
    let manifest = Manifest::load();
    let id = id_from_hex(manifest.get("id_hex"));
    let fnm = std::fs::read(format!("{}{}.raw", dir(), manifest.get("fnm_file_name"))).unwrap();
    let fis = field_infos::parse(&fnm, &id, "").unwrap();

    let dvm = std::fs::read(format!("{}{}.raw", dir(), manifest.get("dvm_file_name"))).unwrap();
    let dvs = std::fs::read(format!("{}{}.raw", dir(), manifest.get("dvs_file_name"))).unwrap();
    let suffix = dv_suffix(&manifest);

    let field_number = manifest.get_i32("field_number");
    let (_, parsed) = doc_values::parse_meta(&dvm, &id, &suffix, &fis).unwrap();

    let want_skipper = DocValuesSkipperMeta {
        offset: manifest.get_i64("skip.offset"),
        length: manifest.get_i64("skip.length"),
        min_value: manifest.get_i64("skip.min_value"),
        max_value: manifest.get_i64("skip.max_value"),
        doc_count: manifest.get_i32("skip.doc_count"),
        max_doc_id: manifest.get_i32("skip.max_doc_id"),
        max_value_count: manifest.get_i32("skip.max_value_count"),
    };
    let got_skipper = parsed
        .skipper_meta(field_number)
        .copied()
        .expect("field has a skip index");
    assert_eq!(got_skipper, want_skipper);

    // Sanity: the field's own numeric entry is still readable -- a skip
    // index doesn't replace or shadow the regular per-doc value entry.
    assert!(parsed.numeric_entry(field_number).is_some());

    let decoded = doc_values::parse_skip_index(&dvs, &id, &suffix, &got_skipper).unwrap();
    assert_eq!(decoded.min_value, want_skipper.min_value);
    assert_eq!(decoded.max_value, want_skipper.max_value);
    assert_eq!(decoded.doc_count, want_skipper.doc_count);
    assert_eq!(decoded.max_doc_id, want_skipper.max_doc_id);
    assert_eq!(decoded.max_value_count, want_skipper.max_value_count);

    let want_intervals = expected_intervals(&manifest);
    assert_eq!(decoded.intervals.len(), want_intervals.len());
    // At least one interval must carry more than one level -- otherwise
    // this fixture wouldn't actually be exercising the multi-level branch
    // of the format (see the generator's NUM_DOCS comment).
    assert!(want_intervals.iter().any(|levels| levels.len() > 1));
    for (got, want) in decoded.intervals.iter().zip(want_intervals.iter()) {
        assert_eq!(&got.levels, want);
    }
}

/// Drives [`doc_values::DocValuesSkipper`] across the same real `.dvs`, and
/// cross-checks every interval it reports against the field's actual `.dvd`
/// values: each level-0 interval must bracket its own documents' values and
/// count them exactly, each coarser level must contain the finer one, and
/// walking `advance(maxDocID(0) + 1)` to exhaustion must visit every document
/// with a value exactly once.
#[test]
fn skipper_walks_real_skip_index_and_brackets_the_real_values() {
    let manifest = Manifest::load();
    let id = id_from_hex(manifest.get("id_hex"));
    let fnm = std::fs::read(format!("{}{}.raw", dir(), manifest.get("fnm_file_name"))).unwrap();
    let fis = field_infos::parse(&fnm, &id, "").unwrap();
    let dvm = std::fs::read(format!("{}{}.raw", dir(), manifest.get("dvm_file_name"))).unwrap();
    let dvd = std::fs::read(format!("{}{}.raw", dir(), manifest.get("dvd_file_name"))).unwrap();
    let dvs = std::fs::read(format!("{}{}.raw", dir(), manifest.get("dvs_file_name"))).unwrap();
    let suffix = dv_suffix(&manifest);
    let field_number = manifest.get_i32("field_number");

    let (_, parsed) = doc_values::parse_meta(&dvm, &id, &suffix, &fis).unwrap();
    let skipper_meta = parsed.skipper_meta(field_number).copied().unwrap();
    let numeric = parsed.numeric_entry(field_number).unwrap();
    let index = doc_values::parse_skip_index(&dvs, &id, &suffix, &skipper_meta).unwrap();
    let mut values = doc_values::NumericReader::new(&dvd, numeric);

    let mut skipper = doc_values::DocValuesSkipper::new(&index);
    assert_eq!(skipper.min_doc_id(0), -1, "not advanced yet");
    skipper.advance(0);

    let mut total_docs = 0;
    let mut intervals_seen = 0;
    let mut previous_max_doc = -1;
    while skipper.min_doc_id(0) != doc_values::NO_MORE_DOCS {
        intervals_seen += 1;
        assert!(
            skipper.min_doc_id(0) > previous_max_doc,
            "intervals must not overlap"
        );

        let mut docs_in_interval = 0;
        for doc in skipper.min_doc_id(0)..=skipper.max_doc_id(0) {
            if let Some(v) = values.value(doc).unwrap() {
                docs_in_interval += 1;
                assert!(
                    v >= skipper.min_value(0) && v <= skipper.max_value(0),
                    "doc {doc} value {v} outside level-0 range [{}, {}]",
                    skipper.min_value(0),
                    skipper.max_value(0)
                );
                assert!(v >= skipper.global_min_value() && v <= skipper.global_max_value());
            }
        }
        assert_eq!(docs_in_interval, skipper.doc_count(0));
        total_docs += docs_in_interval;

        // Coarser levels must be non-decreasing in coverage, per
        // `DocValuesSkipper`'s own contract.
        for level in 1..skipper.num_levels() {
            assert!(skipper.min_doc_id(level) <= skipper.min_doc_id(level - 1));
            assert!(skipper.max_doc_id(level) >= skipper.max_doc_id(level - 1));
            assert!(skipper.min_value(level) <= skipper.min_value(level - 1));
            assert!(skipper.max_value(level) >= skipper.max_value(level - 1));
            assert!(skipper.doc_count(level) >= skipper.doc_count(level - 1));
        }

        previous_max_doc = skipper.max_doc_id(0);
        skipper.advance(previous_max_doc + 1);
    }

    assert_eq!(intervals_seen, manifest.get_i32("skip.interval_count"));
    assert_eq!(total_docs, manifest.get_i32("skip.doc_count"));
    assert_eq!(
        skipper.max_doc_id(0),
        doc_values::NO_MORE_DOCS,
        "exhausted on every level"
    );
}

/// `advance_range` must land on the first interval whose value range
/// intersects the query range, skipping every earlier one -- checked against
/// the decoded intervals directly.
#[test]
fn skipper_advance_range_lands_on_the_first_intersecting_interval() {
    let manifest = Manifest::load();
    let id = id_from_hex(manifest.get("id_hex"));
    let fnm = std::fs::read(format!("{}{}.raw", dir(), manifest.get("fnm_file_name"))).unwrap();
    let fis = field_infos::parse(&fnm, &id, "").unwrap();
    let dvm = std::fs::read(format!("{}{}.raw", dir(), manifest.get("dvm_file_name"))).unwrap();
    let dvs = std::fs::read(format!("{}{}.raw", dir(), manifest.get("dvs_file_name"))).unwrap();
    let suffix = dv_suffix(&manifest);
    let field_number = manifest.get_i32("field_number");

    let (_, parsed) = doc_values::parse_meta(&dvm, &id, &suffix, &fis).unwrap();
    let skipper_meta = parsed.skipper_meta(field_number).copied().unwrap();
    let index = doc_values::parse_skip_index(&dvs, &id, &suffix, &skipper_meta).unwrap();

    // A range that only the last interval can hold.
    let last = index.intervals.last().unwrap().levels[0];
    let (lo, hi) = (last.max_value - 1, last.max_value);
    let want = index
        .intervals
        .iter()
        .position(|i| i.levels[0].min_value <= hi && i.levels[0].max_value >= lo)
        .unwrap();

    let mut skipper = doc_values::DocValuesSkipper::new(&index);
    skipper.advance_range(lo, hi);
    assert_eq!(
        skipper.min_doc_id(0),
        index.intervals[want].levels[0].min_doc_id
    );

    // A range no interval can hold leaves the skipper exhausted.
    let mut skipper = doc_values::DocValuesSkipper::new(&index);
    skipper.advance_range(index.max_value + 1, index.max_value + 100);
    assert_eq!(skipper.min_doc_id(0), doc_values::NO_MORE_DOCS);
}
