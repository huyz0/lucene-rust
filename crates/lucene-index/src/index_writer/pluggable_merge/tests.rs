#![cfg(test)]
use super::*;
use lucene_codecs::compound_format;
use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::stored_fields::{self, Document, FieldValue, StoredField};
use lucene_store::directory::{Directory, FsDirectory};
use lucene_util::test_support::TempDir;

use crate::segment_info::LuceneVersion;
use crate::segment_infos;

fn field() -> FieldInfo {
    FieldInfo {
        name: "id".to_string(),
        number: 0,
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

fn doc(id: &str) -> Document {
    Document {
        fields: vec![StoredField {
            field_number: 0,
            value: FieldValue::String(id.to_string()),
        }],
    }
}

fn version() -> LuceneVersion {
    LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    }
}

/// Every stored document of every committed segment, read through the
/// compound archive when the segment is one.
fn stored_ids(dir: &FsDirectory) -> (Vec<String>, Vec<bool>) {
    let sis = segment_infos::read_latest(dir).unwrap();
    let mut ids = Vec::new();
    let mut compound = Vec::new();
    for sci in &sis.segments {
        let si_bytes = dir.open(&format!("{}.si", sci.segment_name)).unwrap();
        let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
        compound.push(si.is_compound_file);
        let read = |ext: &str| -> Vec<u8> {
            if si.is_compound_file {
                let cfs = dir.open(&format!("{}.cfs", sci.segment_name)).unwrap();
                let cfe = dir.open(&format!("{}.cfe", sci.segment_name)).unwrap();
                let entries = compound_format::parse_entries(&cfe, &sci.segment_id).unwrap();
                compound_format::check_data_header_footer(&cfs, &sci.segment_id, &entries).unwrap();
                compound_format::open_input(&cfs, &entries, ext)
                    .unwrap()
                    .as_slice()
                    .to_vec()
            } else {
                dir.open(&format!("{}{ext}", sci.segment_name))
                    .unwrap()
                    .to_vec()
            }
        };
        let (fdt, fdx, fdm) = (read(".fdt"), read(".fdx"), read(".fdm"));
        let reader = stored_fields::open(&fdt, &fdx, &fdm, &sci.segment_id, "").unwrap();
        for d in 0..reader.max_doc() {
            if let FieldValue::String(s) = &reader.document(d).unwrap().fields[0].value {
                ids.push(s.clone());
            }
        }
    }
    (ids, compound)
}

/// What a fake `OneMerge` subclass answers: see [`hooked_merges`].
#[derive(Debug)]
enum Answer {
    Fail,
    Readers(usize),
    Order(Vec<i32>),
    HideFirst,
}

impl crate::merge_policy::MergeHooks for Answer {
    fn prepare(
        &self,
        _dir: &dyn lucene_store::Directory,
        sources: &crate::segment_infos::SegmentInfos,
        may_reorder: bool,
    ) -> std::result::Result<crate::merge_policy::PreparedMerge, String> {
        assert!(may_reorder);
        let n = sources.segments.len();
        match self {
            Answer::Fail => Err("no".to_string()),
            Answer::Readers(k) => Ok(crate::merge_policy::PreparedMerge {
                live_docs: vec![None; *k],
                new_to_old: None,
                diagnostics: Vec::new(),
            }),
            Answer::Order(o) => Ok(crate::merge_policy::PreparedMerge {
                live_docs: vec![None; n],
                new_to_old: Some(o.clone()),
                diagnostics: Vec::new(),
            }),
            Answer::HideFirst => {
                let mut live = lucene_util::fixed_bit_set::FixedBitSet::new(1);
                let mut all = vec![None; n];
                // Every source here holds one document; the first is
                // hidden.
                all[0] = Some(live.clone());
                live.set(0);
                all[1] = Some(live);
                Ok(crate::merge_policy::PreparedMerge {
                    live_docs: all,
                    new_to_old: Some((0..n as i32).rev().collect()),
                    diagnostics: Vec::new(),
                })
            }
        }
    }
}

/// A `OneMerge` with hooks: the writer honours the wrapped readers' live
/// documents and the reorder's document order, and refuses a hook that
/// fails, answers for the wrong number of sources, or maps documents
/// that are not a permutation of the merged view.
#[test]
fn hooked_merges() {
    let run = |answer: Answer| {
        let tmp = TempDir::new("pluggable-merge-hooks");
        let dir = FsDirectory::open(&tmp);
        let mut w = IndexWriter::open(&dir, vec![field()], "Lucene104", version()).unwrap();
        for id in ["a", "b", "c"] {
            w.add_document(doc(id)).unwrap();
            w.commit().unwrap();
        }
        let names: Vec<String> = w
            .segment_infos()
            .segments
            .iter()
            .map(|s| s.segment_name.clone())
            .collect();
        let result = w.execute_merge_hooked(&names, Some(Arc::new(answer)));
        let ids = result.as_ref().ok().map(|()| {
            w.commit().unwrap();
            stored_ids(&dir).0
        });
        (result.map_err(|e| e.to_string()), ids)
    };
    let (r, ids) = run(Answer::Order(vec![2, 0, 1]));
    assert!(r.is_ok(), "{r:?}");
    assert_eq!(ids.unwrap(), ["c", "a", "b"]);
    let (_, ids) = run(Answer::HideFirst);
    assert_eq!(ids.unwrap(), ["c", "b"]);
    for (answer, want) in [
        (Answer::Fail, "merge hooks: no"),
        (Answer::Readers(2), "2 readers for 3 sources"),
        (Answer::Order(vec![0, 1]), "covers 2 documents"),
        (Answer::Order(vec![0, 1, 7]), "outside the merged view"),
        (Answer::Order(vec![0, 1, 1]), "appears twice"),
    ] {
        let (r, _) = run(answer);
        let e = r.unwrap_err();
        assert!(e.contains(want), "{e}");
    }
}

/// `useCompoundFile`: a flushed segment is packed, `.si` outside the
/// archive and listing exactly the archive and itself; the loose files
/// are gone. A merge then follows the merge policy: `TieredMergePolicy`'s
/// default `noCFSRatio` (0.1) keeps a merge of the whole index loose, a
/// ratio of 1.0 packs it.
#[test]
fn flushes_and_merges_write_compound_segments_as_java_decides() {
    let tmp = TempDir::new("pluggable-merge-compound");
    let dir = FsDirectory::open(&tmp);
    let mut w = IndexWriter::open(&dir, vec![field()], "Lucene104", version()).unwrap();
    assert!(!w.use_compound_file());
    w.set_use_compound_file(true);
    assert!(w.use_compound_file());
    for (i, id) in ["a", "b", "c", "d"].iter().enumerate() {
        w.add_document(doc(id)).unwrap();
        if i % 2 == 1 {
            w.commit().unwrap();
        }
    }
    let (ids, compound) = stored_ids(&dir);
    assert_eq!(ids, ["a", "b", "c", "d"]);
    assert_eq!(compound, [true, true]);
    let names = dir.list_all().unwrap();
    assert!(names.iter().any(|n| n == "_0.cfs") && names.iter().any(|n| n == "_0.cfe"));
    assert!(!names.iter().any(|n| n == "_0.fdt" || n == "_0.fnm"));
    let si = segment_info::parse(
        &dir.open("_0.si").unwrap(),
        &segment_infos::read_latest(&dir).unwrap().segments[0].segment_id,
    )
    .unwrap();
    assert_eq!(si.files, ["_0.cfs", "_0.cfe", "_0.si"]);

    // The whole index merged is more than 10% of the index: loose.
    w.force_merge(1).unwrap();
    let (ids, compound) = stored_ids(&dir);
    assert_eq!(ids, ["a", "b", "c", "d"]);
    assert_eq!(compound, [false]);

    // A policy with `noCFSRatio` 1.0 packs the merged segment.
    let mut policy = merge_policy::TieredMergePolicy::default();
    policy
        .compound_file_settings_mut()
        .set_no_cfs_ratio(1.0)
        .unwrap();
    w.set_pluggable_merge_policy(Some(Arc::new(policy)));
    w.set_use_compound_file(false);
    w.add_document(doc("e")).unwrap();
    w.commit().unwrap();
    let (_, compound) = stored_ids(&dir);
    assert_eq!(compound.last(), Some(&false), "the flag is off for flushes");
    w.force_merge(1).unwrap();
    let (ids, compound) = stored_ids(&dir);
    assert_eq!(ids, ["a", "b", "c", "d", "e"]);
    assert_eq!(compound, [true]);
}

/// With the flag off and no pluggable policy, a merge stays loose
/// whatever `TieredMergePolicy` would have said.
#[test]
fn a_writer_without_compound_files_merges_loose() {
    let tmp = TempDir::new("pluggable-merge-loose");
    let dir = FsDirectory::open(&tmp);
    let mut w = IndexWriter::open(&dir, vec![field()], "Lucene104", version()).unwrap();
    for id in ["a", "b"] {
        w.add_document(doc(id)).unwrap();
        w.commit().unwrap();
    }
    w.force_merge(1).unwrap();
    let (ids, compound) = stored_ids(&dir);
    assert_eq!(ids, ["a", "b"]);
    assert_eq!(compound, [false]);
}
