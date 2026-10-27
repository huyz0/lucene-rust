//! Test-only codec adapters and fault injection from `org.apache.lucene.tests.codecs`.
pub mod asserting_codec;
pub mod asserting_doc_values;
pub mod asserting_installed;
pub mod asserting_knn_vectors;
pub mod asserting_live_docs;
pub mod asserting_live_docs_writer;
pub mod asserting_norms;
pub mod asserting_norms_writer;
pub mod asserting_points;
pub mod asserting_postings;
pub mod asserting_reader;
pub mod asserting_stored_fields;
pub mod asserting_stored_fields_writer;
pub mod asserting_term_vectors;
pub mod asserting_term_vectors_writer;
pub mod blockterms;
pub mod bloom;
pub mod cheap_bastard;
pub mod compressing_codec;
pub mod configurable_m;
pub mod cranky_codec;
pub mod cranky_compound;
pub mod cranky_doc_values;
pub mod cranky_field_infos;
pub mod cranky_installed;
pub mod cranky_live_docs;
pub mod cranky_norms;
pub mod cranky_points;
pub mod cranky_postings;
pub mod cranky_segment_info;
pub mod cranky_stored_fields;
pub mod cranky_term_vectors;
pub mod mockrandom;
pub mod ramonly;
pub mod random_installed;
pub mod rot13;

/// Makes the test framework's terms-dictionary postings formats readable by
/// name (`PostingsFormat`'s SPI): `LuceneFixedGap`,
/// `LuceneVarGapFixedInterval`, `LuceneVarGapDocFreqInterval`,
/// `UniformSplitRot13`, `STUniformSplitRot13`,
/// `TestBloomFilteredLucenePostings` and (read-only) `MockRandom`.
/// Idempotent.
pub fn register_postings_formats() {
    use lucene_codecs::terms_dictionary::register;
    register(&blockterms::FIXED_GAP);
    register(&blockterms::VAR_GAP_FIXED_INTERVAL);
    register(&blockterms::VAR_GAP_DOC_FREQ_INTERVAL);
    for format in [
        rot13::UniformSplitRot13PostingsFormat::default().dictionary_format(),
        rot13::StUniformSplitRot13PostingsFormat.dictionary_format(),
    ]
    .into_iter()
    .flatten()
    {
        register(format);
    }
    register(bloom::TestBloomFilteredLucenePostings::dictionary_format());
    register(mockrandom::MockRandomPostingsFormat::read_only());
}

/// The test framework's codecs that keep `Lucene104`'s file formats, by the
/// name `Codec.forName` resolves (the jar's `META-INF/services` entries).
pub const LUCENE104_FILTER_CODECS: &[&str] = &["Asserting", "CheapBastard", "ConfigurableMCodec"];

/// Makes the test framework's codecs readable by name (`Codec`'s SPI), and
/// their postings formats ([`register_postings_formats`]) and
/// `AssertingKnnVectorsFormat` too. Idempotent.
pub fn register_codecs() {
    use lucene_codecs::codec::Lucene104Codec;
    use lucene_codecs::codecs::{lucene104, register, renamed};
    register_postings_formats();
    lucene_codecs::per_field_knn_vectors::register_wrapper(
        asserting_installed::InstalledAssertingCodec::new().knn_vectors_format_for_field(""),
    );
    for name in LUCENE104_FILTER_CODECS {
        register(renamed(lucene104(), name));
    }
}
