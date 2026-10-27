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
//! [`super::register_codecs`]). Reader-side asserting producers are not
//! installed: segments are read through the native graph.

use lucene_codecs::codec::Lucene104Codec;
use lucene_codecs::live_docs_writer::LiveDocsWriter;
use lucene_codecs::norms_writer::NormsWriterFactory;
use lucene_codecs::per_field_doc_values::Lucene90DocValuesFormat;
use lucene_codecs::per_field_knn_vectors::KnnVectorsFormat;
use lucene_codecs::per_field_postings::Lucene104PostingsFormat;
use lucene_codecs::points_writer::PointsWriterFactory;
use lucene_codecs::postings_consumer::PostingsWriterFactory;
use lucene_codecs::stored_fields_writer::StoredFieldsWriterFactory;
use lucene_codecs::term_vectors_writer::TermVectorsWriterFactory;

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

/// `new AssertingCodec()` through IndexWriter.
#[derive(Debug, Default, Clone, Copy)]
pub struct InstalledAssertingCodec {
    stored_fields: AssertingStoredFieldsCodec,
    term_vectors: AssertingTermVectorsCodec,
    norms: AssertingNormsCodec,
    live_docs: AssertingLiveDocsCodec,
    points: AssertingPointsCodec,
    postings: AssertingPostingsCodec,
}

impl InstalledAssertingCodec {
    pub fn new() -> Self {
        Self::default()
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
    fn postings_format_for_field(&self, _field: &str) -> Lucene104PostingsFormat {
        Lucene104PostingsFormat::default().named(NAME)
    }

    /// `AssertingCodec.getDocValuesFormatForField`: one
    /// `AssertingDocValuesFormat` over `TestUtil.getDefaultDocValuesFormat()`.
    fn doc_values_format_for_field(&self, _field: &str) -> Lucene90DocValuesFormat {
        Lucene90DocValuesFormat::default().named(NAME)
    }

    /// `AssertingCodec.getKnnVectorsFormatForField`: one
    /// `AssertingKnnVectorsFormat` over `TestUtil.getDefaultKnnVectorsFormat()`.
    fn knn_vectors_format_for_field(&self, _field: &str) -> KnnVectorsFormat {
        KnnVectorsFormat::named(NAME, &DEFAULT_KNN_VECTORS)
    }
}
