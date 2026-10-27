//! `Lucene104Codec` per-field routing and optional same-wire writers.
//!
//! Postings, doc-values and vector selection use native format descriptors.
//! An optional stored-fields factory wraps the real document writer lifecycle
//! while retaining the native reader graph and `Lucene104` on-disk formats.
//! Optional live-docs, norms, points, postings and term-vector writers add checks
//! to their native output. Other segment-level formats remain fixed to the codec.

use std::fmt;

use crate::live_docs_writer::LiveDocsWriter;
use crate::norms_writer::NormsWriterFactory;
use crate::per_field_doc_values::Lucene90DocValuesFormat;
use crate::per_field_knn_vectors::KnnVectorsFormat;
use crate::per_field_postings::Lucene104PostingsFormat;
use crate::points_writer::PointsWriterFactory;
use crate::postings_consumer::PostingsWriterFactory;
use crate::stored_fields_writer::StoredFieldsWriterFactory;
use crate::term_vectors_writer::TermVectorsWriterFactory;

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

    /// Optional wrapper for native live-docs writes.
    ///
    /// `None` retains the direct native encoder. A writer must preserve the
    /// native wire format and leave generation advancement to the index layer.
    fn live_docs_writer(&self) -> Option<&dyn LiveDocsWriter> {
        None
    }

    /// Optional same-wire consumer for native norms flushes and merges.
    /// `None` retains the direct native encoder and reader graph.
    fn norms_writer_factory(&self) -> Option<&dyn NormsWriterFactory> {
        None
    }

    /// Optional same-wire points writer for native flushes and merges.
    /// `None` retains the direct native encoder and reader graph.
    fn points_writer_factory(&self) -> Option<&dyn PointsWriterFactory> {
        None
    }

    /// Optional same-wire consumer for native Lucene104 postings groups.
    /// Completion groups retain their own native writer.
    fn postings_writer_factory(&self) -> Option<&dyn PostingsWriterFactory> {
        None
    }

    /// Optional same-wire term-vector callbacks for native flushes and merges.
    /// `None` retains the direct native encoder and bulk chunk-copy merge paths.
    fn term_vectors_writer_factory(&self) -> Option<&dyn TermVectorsWriterFactory> {
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

    /// Whether `knnVectorsFormat()` is the format itself rather than a
    /// `PerFieldKnnVectorsFormat` (`ConfigurableMCodec`): every field's
    /// vectors go to one set of files with no codec suffix (`_0.vec`), in the
    /// format [`Self::knn_vectors_format_for_field`] answers (the same for
    /// every field), and no field records the per-field attributes. The
    /// postings twin is [`Lucene104PostingsFormat::direct`].
    fn knn_vectors_format_is_direct(&self) -> bool {
        false
    }

    /// Called just before the writer does `point` with a format this trait
    /// has no writer factory for (doc values, field infos, segment info,
    /// compound files); an error fails that write as the format's own
    /// `IOException` would (`CrankyCodec`'s fake ones). The files themselves
    /// are always the native ones. The default never fails.
    ///
    /// # Errors
    /// The failure to inject.
    fn before_write(&self, _point: WritePoint) -> std::io::Result<()> {
        Ok(())
    }

    /// Shown each doc-values instance's encoded files before the writer
    /// writes them -- at flush, merge and doc-values update alike -- as a
    /// wrapping `DocValuesConsumer` sees each field's values go by
    /// (`AssertingDocValuesConsumer`'s checks). The default looks at nothing.
    ///
    /// # Errors
    /// The files cannot be read back.
    fn check_doc_values(&self, _write: &DocValuesWrite<'_>) -> std::io::Result<()> {
        Ok(())
    }
}

/// One doc-values instance's files, as [`Lucene104Codec::check_doc_values`]
/// is shown them.
#[derive(Debug, Clone, Copy)]
pub struct DocValuesWrite<'a> {
    /// The segment's document count.
    pub max_doc: i32,
    /// The segment's id.
    pub segment_id: &'a [u8; lucene_store::codec_util::ID_LENGTH],
    /// The files' segment suffix (`Asserting_0`, or a generation's).
    pub segment_suffix: &'a str,
    /// The per-field format the instance is (`Asserting`, `Lucene90`).
    pub format_name: &'a str,
    /// The instance's fields.
    pub field_infos: &'a crate::field_infos::FieldInfos,
    /// `.dvm`, `.dvd` and `.dvs`.
    pub dvm: &'a [u8],
    pub dvd: &'a [u8],
    pub dvs: &'a [u8],
}

/// A write [`Lucene104Codec::before_write`] can fail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WritePoint {
    /// `DocValuesFormat.fieldsConsumer(state)`: one per format instance.
    DocValuesConsumer,
    /// `DocValuesConsumer.add*Field`: one per field of the instance.
    DocValuesField,
    /// `DocValuesConsumer.close()`, after the instance's files are written.
    DocValuesClose,
    /// `FieldInfosFormat.write`: a segment's `.fnm`.
    FieldInfos,
    /// `SegmentInfoFormat.write`: a segment's `.si`.
    SegmentInfo,
    /// `CompoundFormat.write`: a segment's `.cfs`/`.cfe`.
    Compound,
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
        assert!(c.live_docs_writer().is_none());
        assert!(c.norms_writer_factory().is_none());
        assert!(c.points_writer_factory().is_none());
        assert!(c.postings_writer_factory().is_none());
        assert!(c.term_vectors_writer_factory().is_none());
        assert!(!c.knn_vectors_format_is_direct());
        assert!(c.before_write(WritePoint::SegmentInfo).is_ok());
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
