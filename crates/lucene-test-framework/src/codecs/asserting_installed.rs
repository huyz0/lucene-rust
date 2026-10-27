//! Java `AssertingCodec` as a whole codec installed in
//! [`lucene_index::index_writer::IndexWriter`] (`IndexWriter::set_codec`):
//! the segment records `Asserting`; every field's postings, doc values and
//! vectors go to one `Asserting` per-field instance whose files are its
//! `Lucene104PostingsFormat`, `Lucene90DocValuesFormat` and
//! `Lucene99HnswVectorsFormat` delegate's (`_0_Asserting_0.tim`,
//! `.dvd`, `.vec`); stored fields, term vectors, norms, live docs, points and
//! postings are written through the asserting writers of
//! [`AssertingStoredFieldsCodec`], [`AssertingTermVectorsCodec`],
//! [`AssertingNormsCodec`], [`AssertingLiveDocsCodec`],
//! [`AssertingPointsCodec`] and [`AssertingPostingsCodec`].
//!
//! [`super::asserting_codec::AssertingCodec`] is the typed composition with
//! Java's delegates, selectors and suppression; this is its default graph,
//! which a reader opens by name (`lucene_codecs::codecs::for_name`, after
//! [`super::register_codecs`]); [`super::asserting_reader`] is its read side.
//! Doc values are encoded natively and checked from their encoded files
//! before they are written ([`Lucene104Codec::check_doc_values`]), at flush,
//! merge and update alike, through `AssertingDocValuesConsumer`'s checks.
//! `AssertingKnnVectorsWriter` checks nothing but non-null arguments, which
//! the types guarantee, so vectors are written natively.

use std::fmt;
use std::sync::Arc;

use lucene_codecs::codec::{DocValuesWrite, Lucene104Codec};
use lucene_codecs::field_infos::{DocValuesType, FieldInfo};
use lucene_codecs::live_docs_writer::LiveDocsWriter;
use lucene_codecs::norms_writer::NormsWriterFactory;
use lucene_codecs::per_field_doc_values::Lucene90DocValuesFormat;
use lucene_codecs::per_field_knn_vectors::KnnVectorsFormat;
use lucene_codecs::per_field_postings::Lucene104PostingsFormat;
use lucene_codecs::points_writer::PointsWriterFactory;
use lucene_codecs::postings_consumer::PostingsWriterFactory;
use lucene_codecs::segment_state::SegmentReadState;
use lucene_codecs::stored_fields_writer::StoredFieldsWriterFactory;
use lucene_codecs::term_vectors_writer::TermVectorsWriterFactory;
use lucene_search::Result as SearchResult;

use super::asserting_doc_values::{
    AssertingDocValuesConsumer, DocValuesConsumer, DocValuesProducer, Files,
    Lucene90DocValuesFormat as Lucene90DocValuesFormatAdapter, ReadState,
};
use super::asserting_live_docs_writer::AssertingLiveDocsCodec;
use super::asserting_norms_writer::AssertingNormsCodec;
use super::asserting_points::AssertingPointsCodec;
use super::asserting_postings::AssertingPostingsCodec;
use super::asserting_stored_fields_writer::AssertingStoredFieldsCodec;
use super::asserting_term_vectors_writer::AssertingTermVectorsCodec;

/// `AssertingCodec`'s SPI name, also the name of its postings, doc-values
/// and vectors formats.
pub const NAME: &str = "Asserting";

/// `TestUtil.getDefaultKnnVectorsFormat()`, `AssertingKnnVectorsFormat`'s
/// delegate.
static DEFAULT_KNN_VECTORS: KnnVectorsFormat = KnnVectorsFormat::Hnsw {
    max_conn: lucene_codecs::hnsw::DEFAULT_MAX_CONN,
    beam_width: lucene_codecs::hnsw::DEFAULT_BEAM_WIDTH,
};

/// A per-field format choice overriding `AssertingCodec`'s.
type Select<T> = Option<Arc<dyn Fn(&str) -> T + Send + Sync>>;

/// `new AssertingCodec()` through IndexWriter, or a subclass of it that
/// overrides `getPostingsFormatForField`, `getDocValuesFormatForField` or
/// `getKnnVectorsFormatForField` ([`Self::with_postings_format`] & co.).
#[derive(Default, Clone)]
pub struct InstalledAssertingCodec {
    stored_fields: AssertingStoredFieldsCodec,
    term_vectors: AssertingTermVectorsCodec,
    norms: AssertingNormsCodec,
    live_docs: AssertingLiveDocsCodec,
    points: AssertingPointsCodec,
    postings: AssertingPostingsCodec,
    postings_for_field: Select<Lucene104PostingsFormat>,
    doc_values_for_field: Select<Lucene90DocValuesFormat>,
    knn_vectors_for_field: Select<KnnVectorsFormat>,
}

impl fmt::Debug for InstalledAssertingCodec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InstalledAssertingCodec")
            .field("postings_for_field", &self.postings_for_field.is_some())
            .field("doc_values_for_field", &self.doc_values_for_field.is_some())
            .field(
                "knn_vectors_for_field",
                &self.knn_vectors_for_field.is_some(),
            )
            .finish()
    }
}

impl InstalledAssertingCodec {
    pub fn new() -> Self {
        Self::default()
    }

    /// `getPostingsFormatForField` overridden: `select(field)` instead of
    /// the `Asserting` postings format.
    pub fn with_postings_format(
        mut self,
        select: impl Fn(&str) -> Lucene104PostingsFormat + Send + Sync + 'static,
    ) -> Self {
        self.postings_for_field = Some(Arc::new(select));
        self
    }

    /// `getDocValuesFormatForField` overridden.
    pub fn with_doc_values_format(
        mut self,
        select: impl Fn(&str) -> Lucene90DocValuesFormat + Send + Sync + 'static,
    ) -> Self {
        self.doc_values_for_field = Some(Arc::new(select));
        self
    }

    /// `getKnnVectorsFormatForField` overridden.
    pub fn with_knn_vectors_format(
        mut self,
        select: impl Fn(&str) -> KnnVectorsFormat + Send + Sync + 'static,
    ) -> Self {
        self.knn_vectors_for_field = Some(Arc::new(select));
        self
    }
}

impl Lucene104Codec for InstalledAssertingCodec {
    fn name(&self) -> &str {
        NAME
    }

    fn stored_fields_writer_factory(&self) -> Option<&dyn StoredFieldsWriterFactory> {
        Some(&self.stored_fields)
    }

    fn live_docs_writer(&self) -> Option<&dyn LiveDocsWriter> {
        Some(&self.live_docs)
    }

    fn norms_writer_factory(&self) -> Option<&dyn NormsWriterFactory> {
        Some(&self.norms)
    }

    fn points_writer_factory(&self) -> Option<&dyn PointsWriterFactory> {
        Some(&self.points)
    }

    fn postings_writer_factory(&self) -> Option<&dyn PostingsWriterFactory> {
        Some(&self.postings)
    }

    fn term_vectors_writer_factory(&self) -> Option<&dyn TermVectorsWriterFactory> {
        Some(&self.term_vectors)
    }

    /// `AssertingCodec.getPostingsFormatForField`: one
    /// `AssertingPostingsFormat` over `TestUtil.getDefaultPostingsFormat()`.
    fn postings_format_for_field(&self, field: &str) -> Lucene104PostingsFormat {
        match &self.postings_for_field {
            Some(select) => select(field),
            None => Lucene104PostingsFormat::default().named(NAME),
        }
    }

    /// `AssertingCodec.getDocValuesFormatForField`: one
    /// `AssertingDocValuesFormat` over `TestUtil.getDefaultDocValuesFormat()`.
    fn doc_values_format_for_field(&self, field: &str) -> Lucene90DocValuesFormat {
        match &self.doc_values_for_field {
            Some(select) => select(field),
            None => Lucene90DocValuesFormat::default().named(NAME),
        }
    }

    /// `AssertingCodec.getKnnVectorsFormatForField`: one
    /// `AssertingKnnVectorsFormat` over `TestUtil.getDefaultKnnVectorsFormat()`.
    fn knn_vectors_format_for_field(&self, field: &str) -> KnnVectorsFormat {
        match &self.knn_vectors_for_field {
            Some(select) => select(field),
            None => KnnVectorsFormat::named(NAME, &DEFAULT_KNN_VECTORS),
        }
    }

    /// `AssertingDocValuesConsumer.add*Field`: each field's values, read
    /// back from the instance's files, go through the consumer's checks.
    fn check_doc_values(&self, write: &DocValuesWrite<'_>) -> std::io::Result<()> {
        use super::asserting_doc_values::DocValuesFormat as _;
        // An overriding selection's own format has no asserting consumer.
        if write.format_name != NAME {
            return Ok(());
        }
        let io = |e: lucene_search::Error| std::io::Error::other(e.to_string());
        let files: Files = [("dvm", write.dvm), ("dvd", write.dvd), ("dvs", write.dvs)]
            .into_iter()
            .map(|(ext, bytes)| (ext.to_string(), bytes.to_vec()))
            .collect();
        let format = Lucene90DocValuesFormatAdapter;
        let producer = format
            .fields_producer(ReadState {
                files: &files,
                segment: SegmentReadState {
                    max_doc: write.max_doc,
                    segment_id: write.segment_id,
                    segment_suffix: write.segment_suffix,
                    field_infos: write.field_infos,
                },
            })
            .map_err(io)?;
        let mut consumer = AssertingDocValuesConsumer::new(Checked, write.max_doc);
        for field in &write.field_infos.fields {
            match field.doc_values_type {
                DocValuesType::Numeric => consumer.add_numeric_field(field, &producer),
                DocValuesType::Binary => consumer.add_binary_field(field, &producer),
                DocValuesType::Sorted => consumer.add_sorted_field(field, &producer),
                DocValuesType::SortedNumeric => consumer.add_sorted_numeric_field(field, &producer),
                DocValuesType::SortedSet => consumer.add_sorted_set_field(field, &producer),
                DocValuesType::None => Ok(()),
            }
            .map_err(io)?;
        }
        Ok(())
    }
}

/// The checked fields' destination: the writer writes the files itself.
struct Checked;

impl DocValuesConsumer for Checked {
    fn add_numeric_field(&mut self, _: &FieldInfo, _: &dyn DocValuesProducer) -> SearchResult<()> {
        Ok(())
    }
    fn add_binary_field(&mut self, _: &FieldInfo, _: &dyn DocValuesProducer) -> SearchResult<()> {
        Ok(())
    }
    fn add_sorted_field(&mut self, _: &FieldInfo, _: &dyn DocValuesProducer) -> SearchResult<()> {
        Ok(())
    }
    fn add_sorted_numeric_field(
        &mut self,
        _: &FieldInfo,
        _: &dyn DocValuesProducer,
    ) -> SearchResult<()> {
        Ok(())
    }
    fn add_sorted_set_field(
        &mut self,
        _: &FieldInfo,
        _: &dyn DocValuesProducer,
    ) -> SearchResult<()> {
        Ok(())
    }
    fn close(&mut self) -> SearchResult<()> {
        Ok(())
    }
}
