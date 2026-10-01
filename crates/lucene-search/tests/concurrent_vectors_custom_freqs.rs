//! `DocumentsWriter` with per-thread buffers carrying what rides alongside a
//! document: several threads add documents with KNN vectors and with
//! custom-frequency terms through one
//! [`ConcurrentIndexWriter`](lucene_index::concurrent_writer::ConcurrentIndexWriter),
//! each flushing its own segments, and every committed document must read
//! back with its own vector and its own frequency -- the two travel in the
//! thread's buffer with the document, through the flush's index sort and its
//! segment build, never from another thread's.

// Test-support code opts out of the arithmetic gate at the file boundary:
// see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::collections::BTreeSet;

use lucene_codecs::field_infos::{
    FieldInfo, IndexOptions, VectorEncoding, VectorSimilarityFunction,
};
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_index::concurrent_writer::ConcurrentIndexWriter;
use lucene_index::index_writer::{DocumentVector, IndexWriter, DISABLE_AUTO_FLUSH_MB};
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::reader::{
    IndexReader, LeafReader, PostingsFlags, StoredFieldVisitor, VisitStatus, NO_MORE_DOCS,
};
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;

/// Reads a document's stored `id`.
#[derive(Default)]
struct Id(Option<usize>);

impl StoredFieldVisitor for Id {
    fn needs_field(
        &mut self,
        _field_number: i32,
    ) -> lucene_codecs::stored_fields::Result<VisitStatus> {
        Ok(VisitStatus::Yes)
    }
    fn string_field(
        &mut self,
        _field_number: i32,
        value: &str,
    ) -> lucene_codecs::stored_fields::Result<()> {
        self.0 = Some(value[1..].parse().unwrap());
        Ok(())
    }
}

const THREADS: usize = 4;
const PER_THREAD: usize = 150;

fn fields() -> Vec<FieldInfo> {
    vec![
        FieldInfo::new("id", 0),
        // Norms omitted, as `FeatureField`'s are.
        FieldInfo {
            index_options: IndexOptions::DocsAndCustomFreqs,
            omit_norms: true,
            ..FieldInfo::new("score", 1)
        },
        FieldInfo {
            vector_dimension: 4,
            vector_encoding: VectorEncoding::Float32,
            vector_similarity_function: VectorSimilarityFunction::Euclidean,
            ..FieldInfo::new("v", 2)
        },
    ]
}

fn doc(id: usize) -> Document {
    Document {
        fields: vec![StoredField {
            field_number: 0,
            value: FieldValue::String(format!("d{id}")),
        }],
    }
}

/// Document `id`'s vector: its id first, so a vector names its document.
fn vector(id: usize) -> Vec<f32> {
    vec![id as f32, (id * 2) as f32, 1.0, -(id as f32)]
}

#[test]
fn every_thread_s_vectors_and_frequencies_stay_with_their_documents() {
    let tmp = TempDir::new("concurrent-vectors-custom-freqs");
    let dir = FsDirectory::open(&tmp);
    let mut w = IndexWriter::open(
        &dir,
        fields(),
        "Lucene104",
        LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        },
    )
    .unwrap();
    w.set_custom_freq_postings_field(Some("score")).unwrap();
    w.add_vector_field("v").unwrap();
    w.set_max_buffered_docs(40).unwrap();
    w.set_ram_buffer_size_mb(DISABLE_AUTO_FLUSH_MB).unwrap();
    w.set_merge_policy(None);
    w.set_max_full_flush_merge_wait_millis(0);
    let w = ConcurrentIndexWriter::new(w, THREADS).unwrap();
    std::thread::scope(|s| {
        for t in 0..THREADS {
            let w = &w;
            s.spawn(move || {
                for k in 0..PER_THREAD {
                    let id = t * PER_THREAD + k;
                    // Every other document carries a vector, the rest a
                    // custom frequency: a buffer mixes both, and plain
                    // documents in between.
                    if id.is_multiple_of(2) {
                        w.add_document_with_vectors(
                            doc(id),
                            vec![DocumentVector::float32("v", vector(id))],
                        )
                        .unwrap();
                    } else if id.is_multiple_of(3) {
                        w.add_document(doc(id)).unwrap();
                    } else {
                        w.add_document_with_custom_freq_terms(
                            doc(id),
                            vec![("k".to_string(), id as i32 + 1)],
                        )
                        .unwrap();
                    }
                }
            });
        }
    });
    // A vector of the wrong dimension is refused, as the single writer
    // refuses it.
    assert!(w
        .add_document_with_vectors(doc(0), vec![DocumentVector::float32("v", vec![1.0])])
        .is_err());
    w.commit().unwrap();

    let reader = DirectoryReader::open(&dir).unwrap();
    assert!(reader.segment_readers().len() > THREADS, "several flushes");
    let mut vectors_seen = BTreeSet::new();
    let mut freqs_seen = BTreeSet::new();
    for seg in reader.segment_readers() {
        // Each document's stored id, to check what its vector and its
        // frequency name against.
        let ids: Vec<usize> = (0..seg.max_doc())
            .map(|d| {
                let mut id = Id::default();
                seg.document(d, &mut id).unwrap();
                id.0.unwrap()
            })
            .collect();
        if let Some(values) = seg.float_vector_values("v").unwrap() {
            for ord in 0..values.size() {
                let d = values.ord_to_doc(ord).unwrap() as usize;
                assert_eq!(values.vector_value(ord).unwrap(), vector(ids[d]));
                assert!(vectors_seen.insert(ids[d]));
            }
        }
        if let Some(terms) = seg.terms("score").unwrap() {
            let mut te = terms.iterator().unwrap();
            assert!(te.try_seek_exact(b"k").unwrap());
            let mut pe = te.postings(PostingsFlags::Freqs).unwrap();
            loop {
                let d = pe.next_doc().unwrap();
                if d == NO_MORE_DOCS {
                    break;
                }
                let id = ids[d as usize];
                assert_eq!(pe.freq(), id as i32 + 1, "d{id}'s frequency");
                assert!(freqs_seen.insert(id));
            }
        }
    }
    let all = 0..THREADS * PER_THREAD;
    let want_vectors: BTreeSet<usize> = all.clone().filter(|i| i % 2 == 0).collect();
    let want_freqs: BTreeSet<usize> = all.filter(|i| i % 2 == 1 && i % 3 != 0).collect();
    assert_eq!(vectors_seen, want_vectors);
    assert_eq!(freqs_seen, want_freqs);
    for result in lucene_index::check_index::check_directory(&dir).unwrap() {
        assert!(result.all_passed(), "{:?}", result.failures());
    }
}

/// A buffer none of whose documents gave the custom-frequency field terms
/// -- a thread's buffer of other documents -- flushes a segment that does
/// not claim the field indexed without a term dictionary.
#[test]
fn a_segment_without_custom_frequency_terms_does_not_claim_the_field() {
    let tmp = TempDir::new("custom-freqs-absent");
    let dir = FsDirectory::open(&tmp);
    let mut w = IndexWriter::open(
        &dir,
        fields(),
        "Lucene104",
        LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        },
    )
    .unwrap();
    w.set_custom_freq_postings_field(Some("score")).unwrap();
    w.add_document(doc(1)).unwrap();
    w.commit().unwrap();
    w.add_document_with_custom_freq_terms(doc(2), vec![("k".to_string(), 3)])
        .unwrap();
    w.commit().unwrap();
    w.force_merge(1).unwrap();
    for result in lucene_index::check_index::check_directory(&dir).unwrap() {
        assert!(result.all_passed(), "{:?}", result.failures());
    }
}
