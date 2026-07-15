//! Closes the loop between task #47's `merge_policy::find_merges` (decide
//! which segments to merge) and the existing `merge::merge_stored_only_segments`
//! (actually merge them): proves `find_merges`' output -- a chosen group of
//! segment names -- can be fed straight into the real merge-execution
//! machinery without any shape mismatch. Not a full automatic
//! merge-triggering pipeline (out of scope per this task's brief), just the
//! "decide, then execute" handoff.

use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::stored_fields::{self, Document, FieldValue, StoredField};
use lucene_index::merge::{merge_stored_only_segments, MergeSource};
use lucene_index::merge_policy::{find_merges, MergePolicyConfig, SegmentStat};
use lucene_index::segment_info::LuceneVersion;
use lucene_index::segment_writer;
use lucene_store::codec_util::ID_LENGTH;
use lucene_store::directory::FsDirectory;

fn version() -> LuceneVersion {
    LuceneVersion {
        major: 10,
        minor: 0,
        bugfix: 0,
    }
}

fn field(name: &str, number: i32) -> FieldInfo {
    FieldInfo {
        name: name.to_string(),
        number,
        store_term_vectors: false,
        omit_norms: false,
        store_payloads: false,
        soft_deletes_field: false,
        parent_field: false,
        index_options: IndexOptions::None,
        doc_values_type: DocValuesType::None,
        doc_values_skip_index_type: DocValuesSkipIndexType::None,
        doc_values_gen: -1,
        attributes: vec![],
        point_dimension_count: 0,
        point_index_dimension_count: 0,
        point_num_bytes: 0,
        vector_dimension: 0,
        vector_encoding: VectorEncoding::Float32,
        vector_similarity_function: VectorSimilarityFunction::Euclidean,
    }
}

fn doc_with(field_number: i32, value: &str) -> Document {
    Document {
        fields: vec![StoredField {
            field_number,
            value: FieldValue::String(value.to_string()),
        }],
    }
}

fn tempdir() -> String {
    let dir = std::env::temp_dir().join(format!(
        "lucene-rust-merge-policy-integration-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir.to_str().unwrap().to_string()
}

#[test]
fn find_merges_output_feeds_directly_into_merge_execution() {
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let fields = vec![field("id", 0)];

    // Three small segments -- with a tight config, find_merges should
    // propose merging all three together.
    let names = ["_0", "_1", "_2"];
    let values = [["a", "b"], ["c", "d"], ["e", "f"]];
    for (i, name) in names.iter().enumerate() {
        segment_writer::flush_stored_only_segment(
            &dir,
            name,
            [(i + 1) as u8; ID_LENGTH],
            "Lucene104",
            version(),
            &fields,
            &[doc_with(0, values[i][0]), doc_with(0, values[i][1])],
            false,
        )
        .unwrap();
    }

    // Step 1: decide. Stats approximated by doc count (no on-disk byte-size
    // computation needed for this small, synthetic case).
    let stats: Vec<SegmentStat> = names
        .iter()
        .map(|name| SegmentStat {
            name: name.to_string(),
            doc_count: 2,
            del_count: 0,
            size_bytes: 2,
        })
        .collect();
    let config = MergePolicyConfig {
        max_merge_at_once: 3,
        segments_per_tier: 1,
        max_merged_segment_size: 1_000_000,
        reclaim_weight: 1.0,
        floor_segment_size: 0,
        force_merge_deletes_pct_allowed: 10.0,
    };
    let groups = find_merges(&stats, &config);
    assert_eq!(groups.len(), 1, "expected one merge group proposed");
    let chosen = &groups[0];
    assert_eq!(chosen.len(), 3, "expected all three segments grouped");

    // Step 2: execute. Feed find_merges' chosen segment names straight into
    // merge_stored_only_segments, re-reading each named segment's files off
    // disk (as a real caller resolving names to on-disk files would).
    let mut fdts = Vec::new();
    let mut fdxs = Vec::new();
    let mut fdms = Vec::new();
    for name in chosen.iter() {
        fdts.push(std::fs::read(std::path::Path::new(&tmp).join(format!("{name}.fdt"))).unwrap());
        fdxs.push(std::fs::read(std::path::Path::new(&tmp).join(format!("{name}.fdx"))).unwrap());
        fdms.push(std::fs::read(std::path::Path::new(&tmp).join(format!("{name}.fdm"))).unwrap());
    }
    let readers: Vec<stored_fields::StoredFieldsReader> = chosen
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let idx = names.iter().position(|n| n == name).unwrap();
            stored_fields::open(
                &fdts[i],
                &fdxs[i],
                &fdms[i],
                &[(idx + 1) as u8; ID_LENGTH],
                "",
            )
            .unwrap()
        })
        .collect();
    let sources: Vec<MergeSource> = readers
        .iter()
        .map(|r| MergeSource::stored_only(&fields, r, None))
        .collect();

    let sci = merge_stored_only_segments(
        &dir,
        &sources,
        "_merged",
        [42u8; ID_LENGTH],
        "Lucene104",
        version(),
    )
    .unwrap();
    assert_eq!(sci.segment_name, "_merged");

    let merged_fdt = std::fs::read(std::path::Path::new(&tmp).join("_merged.fdt")).unwrap();
    let merged_fdx = std::fs::read(std::path::Path::new(&tmp).join("_merged.fdx")).unwrap();
    let merged_fdm = std::fs::read(std::path::Path::new(&tmp).join("_merged.fdm")).unwrap();
    let merged_reader = stored_fields::open(
        &merged_fdt,
        &merged_fdx,
        &merged_fdm,
        &[42u8; ID_LENGTH],
        "",
    )
    .unwrap();
    assert_eq!(merged_reader.max_doc(), 6);

    std::fs::remove_dir_all(&tmp).ok();
}
