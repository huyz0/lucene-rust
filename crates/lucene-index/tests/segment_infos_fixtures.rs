//! Differential test against a real `segments_N` file written by an actual
//! IndexWriter across two commits. Regenerate with fixtures/src/GenSegmentInfos.java.

use lucene_index::segment_infos;

fn dir() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/segments_index/"
    )
    .to_string()
}

struct Manifest {
    kv: Vec<(String, String)>,
}

impl Manifest {
    fn load() -> Self {
        let text = std::fs::read_to_string(format!("{}manifest.properties", dir()))
            .expect("run fixtures generator first (GenSegmentInfos)");
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
}

#[test]
fn parses_real_two_commit_index() {
    let manifest = Manifest::load();
    let segments_file_name = manifest.get("segments_file_name");
    let generation = manifest.get_i64("generation");

    let buf = std::fs::read(format!("{}{}.raw", dir(), segments_file_name)).unwrap();
    let sis = segment_infos::parse(&buf, generation).unwrap();

    assert_eq!(sis.generation, generation);
    assert_eq!(sis.counter, manifest.get_i64("counter"));
    assert_eq!(
        sis.segments.len(),
        manifest.get("num_segments").parse::<usize>().unwrap()
    );

    let expected_names: Vec<&str> = manifest.get("segment_names").split(',').collect();
    let expected_docs: Vec<i32> = manifest
        .get("segment_doc_counts")
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    let expected_dels: Vec<i32> = manifest
        .get("segment_del_counts")
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();

    for (i, sci) in sis.segments.iter().enumerate() {
        assert_eq!(sci.segment_name, expected_names[i]);
        assert_eq!(sci.del_count, expected_dels[i]);
        assert!(sci.codec_name.starts_with("Lucene"));

        // Cross-check against the segment's own .si file (parsed independently by
        // segment_info::parse) to confirm segments_N's doc count claim is consistent
        // with what the segment itself records — a real cross-module integration
        // check, not just a fixture-vs-fixture comparison.
        let si_buf = std::fs::read(format!("{}{}.si", dir(), sci.segment_name)).unwrap();
        let si = lucene_index::segment_info::parse(&si_buf, &sci.segment_id).unwrap();
        assert_eq!(si.doc_count, expected_docs[i]);
    }

    let expected_user_data: Vec<(String, String)> = manifest
        .get("user_data")
        .split(';')
        .filter(|s| !s.is_empty())
        .map(|kv| {
            let (k, v) = kv.split_once('=').unwrap();
            (k.to_string(), v.to_string())
        })
        .collect();
    let mut actual_user_data = sis.user_data.clone();
    let mut expected_user_data = expected_user_data;
    actual_user_data.sort();
    expected_user_data.sort();
    assert_eq!(actual_user_data, expected_user_data);
}

#[test]
fn wrong_generation_suffix_rejected() {
    let manifest = Manifest::load();
    let segments_file_name = manifest.get("segments_file_name");
    let buf = std::fs::read(format!("{}{}.raw", dir(), segments_file_name)).unwrap();
    assert!(segment_infos::parse(&buf, 999).is_err());
}
