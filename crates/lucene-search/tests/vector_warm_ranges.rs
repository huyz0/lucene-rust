//! `SegmentReader::vector_warm_ranges`: the vector files a vector search
//! reads at random, for a caller that warms them when the first vector
//! search starts rather than when the segment opens.

#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::field_infos::{FieldInfo, VectorEncoding, VectorSimilarityFunction};
use lucene_codecs::stored_fields::{FieldValue, StoredField};
use lucene_index::concurrent_writer::ConcurrentIndexWriter;
use lucene_index::index_writer::{DocumentVector, ExplicitDocument, IndexWriter};
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;

const DIM: usize = 8;
const DOCS: usize = 64;

fn build(dir: &FsDirectory, vectors: bool) {
    let mut w = IndexWriter::open(
        dir,
        Vec::new(),
        "Lucene104",
        LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        },
    )
    .unwrap();
    w.enable_explicit_documents().unwrap();
    w.register_field(FieldInfo::new("id", 0)).unwrap();
    if vectors {
        w.register_field(FieldInfo::new("v", 0).with_vectors(
            DIM as i32,
            VectorEncoding::Float32,
            VectorSimilarityFunction::Euclidean,
        ))
        .unwrap();
    }
    w.set_merge_policy(None);
    let w = ConcurrentIndexWriter::new(w, 1).unwrap();
    let (docs, vecs): (Vec<_>, Vec<_>) = (0..DOCS)
        .map(|id| {
            let mut d = ExplicitDocument::default();
            d.stored.push(StoredField {
                field_number: 0,
                value: FieldValue::String(id.to_string()),
            });
            let v = (0..DIM).map(|c| (id * DIM + c) as f32).collect::<Vec<_>>();
            (d, vec![DocumentVector::float32("v", v)])
        })
        .unzip();
    if vectors {
        w.add_explicit_documents_with_vectors_in(None, docs, vecs)
            .unwrap();
    } else {
        w.add_explicit_documents_in(None, docs).unwrap();
    }
    w.flush().unwrap();
    w.commit().unwrap();
    w.into_writer().unwrap().close().unwrap();
}

#[test]
fn a_segment_with_vectors_names_its_vector_files_and_one_without_names_none() {
    let tmp = TempDir::new("vector-warm-ranges");
    let dir = FsDirectory::open(&tmp);
    build(&dir, true);
    let reader = DirectoryReader::open(&dir).unwrap();
    let ranges = reader.segment_readers()[0].vector_warm_ranges();
    // The flat vectors (`.vec`) hold every vector's bytes, and the graph
    // (`.vex`) is the second file.
    let flat = DOCS * DIM * 4;
    assert!(
        ranges.iter().any(|r| r.len() >= flat),
        "no range holds the {flat} bytes of vectors: {:?}",
        ranges.iter().map(|r| r.len()).collect::<Vec<_>>()
    );
    assert!(ranges.len() >= 2, "the graph is named too");

    let tmp = TempDir::new("vector-warm-ranges-none");
    let dir = FsDirectory::open(&tmp);
    build(&dir, false);
    let reader = DirectoryReader::open(&dir).unwrap();
    assert!(reader.segment_readers()[0].vector_warm_ranges().is_empty());
}
