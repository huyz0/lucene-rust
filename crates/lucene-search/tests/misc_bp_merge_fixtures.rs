//! M14 T14.3 `misc`, differentially against Lucene 10.5.0:
//! `fixtures/src/GenMiscBpMerge.java`'s `misc/bp_merge.tsv`.
//!
//! `BPReorderingMergePolicy` through the writer over `misc/bp_merge_src`
//! (fifteen segments with deletions): the merged segments' documents in
//! Lucene's order -- reordered by a forced merge and by natural merges,
//! untouched by natural merges below `minNaturalMergeNumDocs`.
#![allow(clippy::arithmetic_side_effects)]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use lucene_index::index_writer::IndexWriter;
use lucene_index::merge_policy::{MergePolicy, TieredMergePolicy};
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::misc::index::{BpIndexReorderer, BpReorderingMergePolicy};
use lucene_search::reader::{LeafReader, StoredFieldVisitor, VisitStatus};
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;

fn data() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/misc")
}

#[derive(Default)]
struct Id(Option<String>);

impl StoredFieldVisitor for Id {
    fn needs_field(&mut self, _n: i32) -> lucene_codecs::stored_fields::Result<VisitStatus> {
        Ok(VisitStatus::Yes)
    }
    fn string_field(&mut self, _n: i32, value: &str) -> lucene_codecs::stored_fields::Result<()> {
        self.0 = Some(value.to_string());
        Ok(())
    }
}

fn policy(min_natural: i32, field: &str) -> BpReorderingMergePolicy {
    let mut reorderer = BpIndexReorderer::new();
    reorderer.set_ram_budget_mb(256.0);
    reorderer.set_min_doc_freq(2).unwrap();
    reorderer.set_min_partition_size(4).unwrap();
    reorderer.set_fields(Some(vec![field.to_string()]));
    let mut tiered = TieredMergePolicy::default();
    tiered
        .compound_file_settings_mut()
        .set_no_cfs_ratio(0.0)
        .unwrap();
    tiered.config.segments_per_tier = 2;
    tiered.config.floor_segment_size = 1048;
    let mut policy = BpReorderingMergePolicy::new(Box::new(tiered), reorderer);
    policy.set_min_natural_merge_num_docs(min_natural).unwrap();
    policy
}

#[test]
fn bp_reordering_merge_policy_merges_as_lucene_does() {
    let tsv = std::fs::read_to_string(data().join("bp_merge.tsv")).unwrap();
    let rows = |tag: &str, config: &str| -> Vec<String> {
        tsv.lines()
            .map(|l| l.split('\t').collect::<Vec<_>>())
            .filter(|c| c[0] == tag && c[1] == config)
            .map(|c| c[3].to_string())
            .collect()
    };
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };

    // (config, minNaturalMergeNumDocs, forced)
    for (config, min_natural, forced) in [
        ("identity", 1, true),
        ("forced", 1, true),
        ("natural", 1, false),
        ("too_small", i32::MAX, false),
    ] {
        let tmp = TempDir::new("bp-merge");
        for entry in std::fs::read_dir(data().join("bp_merge_src")).unwrap() {
            let entry = entry.unwrap();
            if entry.file_name() != "write.lock" {
                std::fs::copy(entry.path(), tmp.path().join(entry.file_name())).unwrap();
            }
        }
        let dir = FsDirectory::open(tmp.path());
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", version).unwrap();
        w.set_pluggable_merge_policy(Some(Arc::new(policy(
            min_natural,
            if config == "identity" { "none" } else { "body" },
        ))));
        if forced {
            w.force_merge(1).unwrap();
        } else {
            w.maybe_merge().unwrap();
        }
        w.close().unwrap();

        let reader = DirectoryReader::open(&dir).unwrap();
        let mut ids = Vec::new();
        for seg in reader.segment_readers() {
            for d in 0..seg.max_doc {
                if seg
                    .live_docs()
                    .is_some_and(|l| !l.get(usize::try_from(d).unwrap()))
                {
                    continue;
                }
                let mut id = Id::default();
                seg.document(d, &mut id).unwrap();
                ids.push(id.0.unwrap());
            }
        }
        assert_eq!(
            reader.segment_readers().len(),
            rows("G", config).len(),
            "{config}: segments"
        );
        assert_eq!(ids, rows("O", config), "{config}: documents");
    }
}

/// `computeDocMap` over the segments' merged view (deleted documents included
/// in the view), in leaf order.
#[test]
fn bp_doc_map_over_a_merged_view_equals_lucenes() {
    use lucene_search::reader::slow_codec::SlowCompositeCodecReaderWrapper;
    use lucene_search::reader::CodecReader;

    let tsv = std::fs::read_to_string(data().join("bp_merge.tsv")).unwrap();
    let want: Vec<i32> = tsv
        .lines()
        .find(|l| l.starts_with("V\t"))
        .unwrap()
        .split('\t')
        .nth(2)
        .unwrap()
        .split(' ')
        .map(|v| v.parse().unwrap())
        .collect();
    let dir = FsDirectory::open(data().join("bp_merge_src"));
    let reader = DirectoryReader::open(&dir).unwrap();
    let leaves: Vec<Arc<dyn CodecReader>> = reader
        .segment_readers()
        .iter()
        .map(|r| Arc::new(r.clone()) as Arc<dyn CodecReader>)
        .collect();
    let merged = SlowCompositeCodecReaderWrapper::wrap(leaves).unwrap();
    let mut reorderer = BpIndexReorderer::new();
    reorderer.set_ram_budget_mb(256.0);
    reorderer.set_min_doc_freq(2).unwrap();
    reorderer.set_min_partition_size(4).unwrap();
    reorderer.set_fields(Some(vec!["body".to_string()]));
    let map = reorderer.compute_doc_map(merged.as_ref()).unwrap();
    let got: Vec<i32> = (0..map.size()).map(|d| map.new_to_old(d)).collect();
    assert_eq!(got, want);
}

#[test]
fn bp_reordering_merge_policy_setters_refuse_out_of_range_values() {
    let mut p = policy(1, "body");
    assert!(p.set_min_natural_merge_num_docs(0).is_err());
    assert!(p
        .set_min_natural_merge_ratio_from_biggest_segment(-0.1)
        .is_err());
    assert!(p
        .set_min_natural_merge_ratio_from_biggest_segment(1.0)
        .is_err());
    assert!(p
        .set_min_natural_merge_ratio_from_biggest_segment(f32::NAN)
        .is_err());
    p.set_min_natural_merge_ratio_from_biggest_segment(0.5)
        .unwrap();
    assert_eq!(p.compound_file_settings().no_cfs_ratio, 0.0);
}
