//! `IndexWriterConfig.setCodec`'s per-field half: `Lucene104Codec`'s three
//! overridable routing methods -- `getPostingsFormatForField`,
//! `getDocValuesFormatForField`, `getKnnVectorsFormatForField` -- as a trait
//! whose every method defaults to what `Lucene104Codec` answers
//! ([`Lucene104Codec`]).
//!
//! Java's `Codec` also names the segment-level formats (stored fields, norms,
//! points, compound, ...); this port writes exactly `Lucene104Codec`'s, so
//! the codec a writer is configured with is only ever its per-field routing.
//! A codec's formats are the ones this port writes: `Lucene104PostingsFormat`
//! instances, `Lucene90DocValuesFormat` instances and the
//! [`crate::per_field_knn_vectors::KnnVectorsFormat`]s.

use std::fmt;

use crate::per_field_doc_values::Lucene90DocValuesFormat;
use crate::per_field_knn_vectors::KnnVectorsFormat;
use crate::per_field_postings::Lucene104PostingsFormat;

/// `Lucene104Codec` with its `get*FormatForField` methods overridable: a
/// subclass in Java, an implementation of this trait here.
pub trait Lucene104Codec: Send + Sync + fmt::Debug {
    /// `Codec.getName()`, what the `.si`/`segments_N` record.
    fn name(&self) -> &str {
        "Lucene104"
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
