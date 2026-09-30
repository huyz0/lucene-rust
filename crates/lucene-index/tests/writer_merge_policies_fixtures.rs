//! Differential test for a pluggable merge policy *inside the writer*
//! (`IndexWriter::set_pluggable_merge_policy`) against
//! `fixtures/data/merge_policies/writer_runs.properties`
//! (`GenMergePolicies.writerRuns`).
//!
//! Java ran a real `IndexWriter` with `LogDocMergePolicy` and a
//! `SerialMergeScheduler`, committing batches of `1 + i % 4` documents, then
//! `forceMerge(2)` and `forceMerge(1)`, and recorded every segment's `maxDoc`,
//! in segment order, after each step. The Rust writer replays the same
//! batches under the same policy; the segment layout must match after every
//! step -- which merges ran, in which order, and where each merged segment
//! landed.
//!
//! Regenerate with `scripts/gen-fixtures.sh --only GenMergePolicies`.
#![allow(clippy::arithmetic_side_effects)]

use std::collections::HashMap;
use std::sync::Arc;

use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_index::buffered_updates::Term;
use lucene_index::index_writer::IndexWriter;
use lucene_index::merge_policy::log::LogDocMergePolicy;
use lucene_index::segment_info::{self, LuceneVersion};
use lucene_index::segment_infos;
use lucene_store::directory::Directory;
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;

fn expected() -> HashMap<String, String> {
    std::fs::read_to_string(format!(
        "{}/../../fixtures/data/merge_policies/writer_runs.properties",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap()
    .lines()
    .filter(|l| !l.starts_with('#'))
    .filter_map(|l| l.split_once('='))
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect()
}

fn id_field() -> FieldInfo {
    FieldInfo {
        name: "id".to_string(),
        number: 0,
        store_term_vectors: false,
        omit_norms: true,
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
        vector_encoding: VectorEncoding::Byte,
        vector_similarity_function: VectorSimilarityFunction::Euclidean,
    }
}

fn doc(id: usize) -> Document {
    Document {
        fields: vec![StoredField {
            field_number: 0,
            value: FieldValue::String(id.to_string()),
        }],
    }
}

/// Every segment's `maxDoc` in the latest commit, in segment order.
fn layout(dir: &dyn Directory) -> String {
    let infos = segment_infos::read_latest(dir).unwrap();
    infos
        .segments
        .iter()
        .map(|sci| {
            let bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
            segment_info::parse(&bytes, &sci.segment_id)
                .unwrap()
                .doc_count
                .to_string()
        })
        .collect::<Vec<_>>()
        .join(",")
}

#[test]
fn log_doc_merge_policy_in_the_writer_matches_lucene() {
    let want = expected();
    let commits: usize = want["commits"].parse().unwrap();
    let runs: Vec<&str> = want["runs"].split(',').collect();
    assert_eq!(runs.len(), 3);
    for run in runs {
        let key = |k: &str| want[&format!("run.{run}.{k}")].clone();
        let mut policy = LogDocMergePolicy::new();
        policy
            .set_merge_factor(key("mergeFactor").parse().unwrap())
            .unwrap();
        policy.set_min_merge_docs(key("minMergeDocs").parse().unwrap());

        let tmp = TempDir::new(&format!("writer-merge-{run}"));
        let dir = FsDirectory::open(tmp.path());
        let version = LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        };
        let mut w = IndexWriter::open(&dir, vec![id_field()], "Lucene104", version).unwrap();
        w.set_pluggable_merge_policy(Some(Arc::new(policy)));
        assert!(w.pluggable_merge_policy().is_some());
        let mut id = 0;
        for c in 0..commits {
            for _ in 0..1 + c % 4 {
                w.add_document(doc(id)).unwrap();
                id += 1;
            }
            w.commit().unwrap();
            assert_eq!(
                layout(&dir),
                key(&format!("commit.{c}")),
                "{run} commit {c}"
            );
        }
        w.force_merge(2).unwrap();
        assert_eq!(layout(&dir), key("force2"), "{run} forceMerge(2)");
        w.force_merge(1).unwrap();
        assert_eq!(layout(&dir), key("force1"), "{run} forceMerge(1)");
        // `forceMergeDeletes` asks the same policy; with nothing deleted
        // `LogMergePolicy.findForcedDeletesMerges` specifies no merge.
        w.force_merge_deletes().unwrap();
        w.commit().unwrap();
        assert_eq!(layout(&dir), key("force1"), "{run} forceMergeDeletes");
    }
}

/// Every segment of the latest commit as `maxDoc/delCount`, in segment order.
fn layout_with_deletes(dir: &dyn Directory) -> String {
    let infos = segment_infos::read_latest(dir).unwrap();
    infos
        .segments
        .iter()
        .map(|sci| {
            let bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
            let si = segment_info::parse(&bytes, &sci.segment_id).unwrap();
            format!("{}/{}", si.doc_count, sci.del_count)
        })
        .collect::<Vec<_>>()
        .join(",")
}

/// `forceMergeDeletes` under a pluggable policy, against
/// `fixtures/data/merge_policies/writer_deletes.properties`
/// (`GenMergePolicies.writerDeletes`): six commits of five documents with an
/// indexed `id`, deletes by term leaving four segments with deletes, then
/// `forceMergeDeletes()`. `LogMergePolicy.findForcedDeletesMerges` specifies
/// the runs of adjacent segments with deletes; the layout after it -- which
/// segments merged, and where each merged segment landed -- must be Java's.
#[test]
fn force_merge_deletes_under_log_doc_merge_policy_matches_lucene() {
    let want: HashMap<String, String> = std::fs::read_to_string(format!(
        "{}/../../fixtures/data/merge_policies/writer_deletes.properties",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap()
    .lines()
    .filter(|l| !l.starts_with('#'))
    .filter_map(|l| l.split_once('='))
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect();
    let mut policy = LogDocMergePolicy::new();
    policy
        .set_merge_factor(want["mergeFactor"].parse().unwrap())
        .unwrap();
    policy.set_min_merge_docs(want["minMergeDocs"].parse().unwrap());

    let tmp = TempDir::new("writer-merge-deletes");
    let dir = FsDirectory::open(tmp.path());
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let field = FieldInfo {
        index_options: IndexOptions::Docs,
        ..id_field()
    };
    let mut w = IndexWriter::open(&dir, vec![field], "Lucene104", version).unwrap();
    w.set_postings_field(Some("id")).unwrap();
    w.set_pluggable_merge_policy(Some(Arc::new(policy)));

    let commits: usize = want["commits"].parse().unwrap();
    let per_commit: usize = want["perCommit"].parse().unwrap();
    let mut id = 0;
    for _ in 0..commits {
        for _ in 0..per_commit {
            w.add_document(doc(id)).unwrap();
            id += 1;
        }
        w.commit().unwrap();
    }
    assert_eq!(layout_with_deletes(&dir), want["added"], "after the adds");

    let terms: Vec<Term> = want["deletes"]
        .split(',')
        .map(|x| Term::new("id", x.as_bytes().to_vec()))
        .collect();
    w.delete_documents_by_term(&terms).unwrap();
    w.commit().unwrap();
    assert_eq!(
        layout_with_deletes(&dir),
        want["deleted"],
        "after the deletes"
    );

    w.force_merge_deletes().unwrap();
    w.commit().unwrap();
    assert_eq!(
        layout_with_deletes(&dir),
        want["forceMergeDeletes"],
        "after forceMergeDeletes"
    );
}
