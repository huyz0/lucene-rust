//! `Lucene104Codec` per-field routing and optional same-wire stored-fields writers.
//!
//! Postings, doc-values and vector selection use native format descriptors.
//! An optional stored-fields factory wraps the real document writer lifecycle
//! while retaining the native reader graph and `Lucene104` on-disk formats.
//! Other segment-level formats remain fixed to the native codec.

use std::fmt;

use crate::per_field_doc_values::Lucene90DocValuesFormat;
use crate::per_field_knn_vectors::KnnVectorsFormat;
use crate::per_field_postings::Lucene104PostingsFormat;
use crate::stored_fields_writer::StoredFieldsWriterFactory;

/// `Lucene104Codec` with its `get*FormatForField` methods overridable: a
/// subclass in Java, an implementation of this trait here.
pub trait Lucene104Codec: Send + Sync + fmt::Debug {
    /// `Codec.getName()`, what the `.si`/`segments_N` record.
    fn name(&self) -> &str {
        "Lucene104"
    }

    /// Optional lifecycle wrapper for native stored-fields flushes and merges.
    ///
    /// `None` retains the direct native encoder and its bulk merge paths.
    /// A factory must preserve the native wire format and segment codec name.
    fn stored_fields_writer_factory(&self) -> Option<&dyn StoredFieldsWriterFactory> {
        None
    }

    /// `getPostingsFormatForField(field)`: `Lucene104PostingsFormat()`.
    fn postings_format_for_field(&self, _field: &str) -> Lucene104PostingsFormat {
        Lucene104PostingsFormat::default()
    }

    /// `getDocValuesFormatForField(field)`: `Lucene90DocValuesFormat()`.
    fn doc_values_format_for_field(&self, _field: &str) -> Lucene90DocValuesFormat {
        Lucene90DocValuesFormat::default()
    }

    /// `getKnnVectorsFormatForField(field)`: `Lucene99HnswVectorsFormat()`.
    fn knn_vectors_format_for_field(&self, _field: &str) -> KnnVectorsFormat {
        KnnVectorsFormat::default()
    }
}

/// `new Lucene104Codec()`: every field on the default formats.
#[derive(Debug, Default, Clone, Copy)]
pub struct DefaultLucene104Codec;

impl Lucene104Codec for DefaultLucene104Codec {}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct Routed;

    impl Lucene104Codec for Routed {
        fn knn_vectors_format_for_field(&self, field: &str) -> KnnVectorsFormat {
            if field == "v" {
                KnnVectorsFormat::hnsw(8, 40).unwrap()
            } else {
                KnnVectorsFormat::default()
            }
        }
    }

    #[test]
    fn the_defaults_are_lucene104codecs() {
        let c = DefaultLucene104Codec;
        assert_eq!(c.name(), "Lucene104");
        assert!(c.stored_fields_writer_factory().is_none());
        assert_eq!(
            c.postings_format_for_field("f"),
            Lucene104PostingsFormat::default()
        );
        assert_eq!(
            c.doc_values_format_for_field("f"),
            Lucene90DocValuesFormat::default()
        );
        assert_eq!(
            c.knn_vectors_format_for_field("f"),
            KnnVectorsFormat::default()
        );
        assert_eq!(
            Routed.knn_vectors_format_for_field("v"),
            KnnVectorsFormat::hnsw(8, 40).unwrap()
        );
        assert_eq!(
            Routed.knn_vectors_format_for_field("w"),
            KnnVectorsFormat::default()
        );
    }
}
