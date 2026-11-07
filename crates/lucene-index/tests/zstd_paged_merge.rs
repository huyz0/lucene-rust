//! Paged zstd stored fields through `IndexWriter`: flushed segments carry
//! no dictionary, a merge above the size threshold trains one, and every
//! document reads back the same.

#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::stored_fields::{self, Document, FieldValue, StoredField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::{self, LuceneVersion};
use lucene_index::segment_infos;
use lucene_index::segment_writer::{ZSTD_PAGED_ATTRIBUTE, ZSTD_PAGED_VALUE};
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

fn version() -> LuceneVersion {
    LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    }
}

/// Every segment's (`.fdt` length, has a dictionary, records the
/// attribute, codec), and the bodies in doc order.
fn inspect(dir: &dyn Directory) -> (Vec<(usize, bool, bool, String)>, Vec<String>) {
    let sis = segment_infos::read_latest(dir).unwrap();
    let mut segs = Vec::new();
    let mut bodies = Vec::new();
    for sci in &sis.segments {
        let read = |ext: &str| dir.open(&format!("{}.{ext}", sci.segment_name)).unwrap();
        let si = segment_info::parse(&read("si"), &sci.segment_id).unwrap();
        let attr = si
            .attributes
            .iter()
            .any(|(k, v)| k == ZSTD_PAGED_ATTRIBUTE && v == ZSTD_PAGED_VALUE);
        let (fdt, fdx, fdm) = (read("fdt"), read("fdx"), read("fdm"));
        let reader = stored_fields::open(&fdt, &fdx, &fdm, &sci.segment_id, "").unwrap();
        assert_eq!(reader.data_codec(), stored_fields::DATA_CODEC_ZSTD_PAGED);
        segs.push((fdt.len(), reader.has_zstd_dictionary(), attr, sci.codec_name.clone()));
        for d in 0..reader.max_doc() {
            if let FieldValue::String(s) = &reader.document(d).unwrap().fields[0].value {
                bodies.push(s.clone());
            }
        }
    }
    (segs, bodies)
}

fn body(i: usize) -> String {
    format!(
        "{{\"@timestamp\":{},\"clientip\":\"10.{}.{}.{}\",\"request\":\"GET /images/item{}.gif HTTP/1.1\",\"status\":{},\"size\":{},\"agent\":\"Mozilla/5.0 ({})\"}}",
        893_964_617 + i / 3,
        i % 7,
        (i * 13) % 251,
        (i * 31) % 253,
        (i * 7) % 2000,
        [200, 304, 404][i % 3],
        (i * 7919) % 60_000,
        ["X11; Linux x86_64", "Windows NT 10.0; Win64; x64", "Macintosh; Intel Mac OS X 10_15_7"][i % 3]
    )
}

#[test]
fn a_large_merge_trains_a_dictionary_and_reads_back_the_same() {
    // `ZSTD_PAGED_OUT=DIR` keeps the index there, with every body sorted in
    // `bodies.expected`, for checking another reader (the plugin's Java
    // codec) against it.
    let tmp = TempDir::new("zstd-paged-merge");
    let out = std::env::var_os("ZSTD_PAGED_OUT").map(std::path::PathBuf::from);
    let dir = FsDirectory::open(out.as_deref().unwrap_or(tmp.path()));
    let codec = lucene_codecs::codec::ZstdPagedLucene104Codec::new(3).unwrap();
    let mut w = IndexWriter::open(&dir, vec![body_field()], "Lucene104", version()).unwrap();
    w.set_codec(Some(std::sync::Arc::new(codec))).unwrap();
    // Small enough for a test; the default is for real segments.
    w.set_zstd_dictionary_min_bytes(1 << 20);
    let n = 160_000;
    for i in 0..n {
        w.add_document(Document {
            fields: vec![StoredField {
                field_number: 0,
                value: FieldValue::String(body(i)),
            }],
        })
        .unwrap();
        if i % 40_000 == 39_999 {
            w.flush().unwrap();
        }
    }
    w.commit().unwrap();
    let (flushed, flushed_bodies) = inspect(&dir);
    assert_eq!(flushed.len(), 4);
    assert!(flushed.iter().all(|s| !s.1 && s.2 && s.3 == "LuceneRustZstdPaged104"), "{flushed:?}");
    let flushed_bytes: usize = flushed.iter().map(|s| s.0).sum();
    assert!(flushed_bytes >= 1 << 20, "{flushed_bytes}");

    w.force_merge(1).unwrap();
    w.commit().unwrap();
    let (merged, bodies) = inspect(&dir);
    assert_eq!(merged.len(), 1);
    assert!(merged[0].1 && merged[0].2, "{merged:?}");
    // The merge may order the segments differently; the documents are the same.
    let (mut bodies, mut flushed_bodies) = (bodies, flushed_bodies);
    bodies.sort_unstable();
    flushed_bodies.sort_unstable();
    assert!(bodies == flushed_bodies);
    assert_eq!(bodies.len(), n);

    // A flush after the merge: one segment with a dictionary, one without.
    for i in n..n + 5000 {
        w.add_document(Document {
            fields: vec![StoredField {
                field_number: 0,
                value: FieldValue::String(body(i)),
            }],
        })
        .unwrap();
    }
    w.commit().unwrap();
    let (segs, mut all) = inspect(&dir);
    assert_eq!(segs.iter().map(|s| s.1).collect::<Vec<_>>(), [true, false]);
    if let Some(out) = &out {
        all.sort_unstable();
        std::fs::write(out.join("bodies.expected"), all.join("\n") + "\n").unwrap();
    }
}
