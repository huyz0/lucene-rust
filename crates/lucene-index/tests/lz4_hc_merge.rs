//! `IndexWriter::set_merge_lz4_high_compression`: merged `BEST_SPEED` stored
//! fields are smaller, read back the same, record the `.si` attribute, and a
//! second merge copies those chunks as they are.

#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::stored_fields::{self, Document, FieldValue, StoredField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::{self, LuceneVersion};
use lucene_index::segment_infos;
use lucene_index::segment_writer::{LZ4_HC_ATTRIBUTE, LZ4_HC_VALUE};
use lucene_store::directory::{Directory, FsDirectory};
use lucene_util::test_support::TempDir;

fn body_field() -> FieldInfo {
    FieldInfo {
        name: "body".to_string(),
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

fn body(i: usize) -> String {
    format!(
        "{{\"@timestamp\":{},\"clientip\":\"10.{}.{}.{}\",\"request\":\"GET /images/item{}.gif HTTP/1.1\",\"status\":{},\"size\":{}}}",
        893_964_617 + i / 3,
        i % 7,
        (i * 13) % 251,
        (i * 31) % 253,
        (i * 17) % 200,
        [200, 304, 404][i % 3],
        (i * 7919) % 60_000
    )
}

fn version() -> LuceneVersion {
    LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    }
}

/// Every segment's (name, `.fdt` length, has the attribute), and the bodies
/// in doc order.
fn inspect(dir: &dyn Directory) -> (Vec<(String, usize, bool)>, Vec<String>) {
    let sis = segment_infos::read_latest(dir).unwrap();
    let mut segs = Vec::new();
    let mut bodies = Vec::new();
    for sci in &sis.segments {
        let read = |ext: &str| dir.open(&format!("{}.{ext}", sci.segment_name)).unwrap();
        let si = segment_info::parse(&read("si"), &sci.segment_id).unwrap();
        let hc = si
            .attributes
            .iter()
            .any(|(k, v)| k == LZ4_HC_ATTRIBUTE && v == LZ4_HC_VALUE);
        let (fdt, fdx, fdm) = (read("fdt"), read("fdx"), read("fdm"));
        segs.push((sci.segment_name.clone(), fdt.len(), hc));
        let reader = stored_fields::open(&fdt, &fdx, &fdm, &sci.segment_id, "").unwrap();
        for d in 0..reader.max_doc() {
            if let FieldValue::String(s) = &reader.document(d).unwrap().fields[0].value {
                bodies.push(s.clone());
            }
        }
    }
    (segs, bodies)
}

fn build(dir: &FsDirectory, hc: bool) {
    let mut w = IndexWriter::open(dir, vec![body_field()], "Lucene104", version()).unwrap();
    w.set_merge_lz4_high_compression(hc);
    for i in 0..6000 {
        w.add_document(Document {
            fields: vec![StoredField {
                field_number: 0,
                value: FieldValue::String(body(i)),
            }],
        })
        .unwrap();
        if i % 1500 == 1499 {
            w.flush().unwrap();
        }
    }
    w.commit().unwrap();
    w.force_merge(1).unwrap();
    w.commit().unwrap();
}

#[test]
fn merged_stored_fields_are_smaller_and_read_back_the_same() {
    let fast_tmp = TempDir::new("lz4-hc-off");
    let fast_dir = FsDirectory::open(fast_tmp.path());
    build(&fast_dir, false);
    let (fast_segs, fast_bodies) = inspect(&fast_dir);

    let hc_tmp = TempDir::new("lz4-hc-on");
    let hc_dir = FsDirectory::open(hc_tmp.path());
    build(&hc_dir, true);
    let (hc_segs, hc_bodies) = inspect(&hc_dir);

    assert_eq!(fast_segs.len(), 1);
    assert_eq!(hc_segs.len(), 1);
    assert!(
        !fast_segs[0].2,
        "the attribute is only written with the option on"
    );
    assert!(hc_segs[0].2, "{hc_segs:?}");
    assert!(
        hc_segs[0].1 * 100 < fast_segs[0].1 * 95,
        "HC {} vs fast {}",
        hc_segs[0].1,
        fast_segs[0].1
    );
    assert_eq!(hc_bodies, fast_bodies);
    assert_eq!(hc_bodies.len(), 6000);
    assert_eq!(hc_bodies[1234], body(1234));
}

#[test]
fn a_merge_copies_high_compression_chunks_as_they_are() {
    let tmp = TempDir::new("lz4-hc-copy");
    let dir = FsDirectory::open(tmp.path());
    build(&dir, true);
    let (first, _) = inspect(&dir);
    // Add one more small segment and merge again: the big segment's chunks
    // are copied, so the merged .fdt is the old one plus the new docs.
    let mut w = IndexWriter::open(&dir, vec![body_field()], "Lucene104", version()).unwrap();
    w.set_merge_lz4_high_compression(true);
    for i in 6000..6010 {
        w.add_document(Document {
            fields: vec![StoredField {
                field_number: 0,
                value: FieldValue::String(body(i)),
            }],
        })
        .unwrap();
    }
    w.commit().unwrap();
    w.force_merge(1).unwrap();
    w.commit().unwrap();
    let (second, bodies) = inspect(&dir);
    assert_eq!(second.len(), 1);
    assert!(second[0].2);
    assert_eq!(bodies.len(), 6010);
    assert_eq!(bodies[6005], body(6005));
    // The copied chunks keep their bytes, and only the new docs add to them.
    assert!(
        second[0].1 > first[0].1 && second[0].1 < first[0].1 + 2000,
        "{first:?} {second:?}"
    );
}
