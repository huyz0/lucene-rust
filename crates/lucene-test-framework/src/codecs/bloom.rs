//! `TestBloomFilteredLucenePostings`: the default Lucene104 postings
//! wrapped in `BloomFilteringPostingsFormat` with a 1024-byte filter for
//! every field. Its `LowMemoryBloomFactory` keeps even saturated filters;
//! the parent factory's normal downsizing still applies.
//!
//! Java: `fieldsConsumer` and `fieldsProducer` delegate to the bloom
//! format. Rust passes file buffers and postings values instead of a
//! Directory and streaming FieldsConsumer. IndexWriter does not select
//! this test format.

use std::fmt;

use lucene_codecs::bloom::{
    BloomFilterFactory, BloomFilteredFieldsProducer, BloomFilteringPostingsFormat, BloomOutput,
    FuzzySet, Result,
};
use lucene_codecs::field_infos::FieldInfo;
use lucene_codecs::per_field_postings::Lucene104PostingsFormat;
use lucene_codecs::postings_writer::{FieldNorms, FieldPostingsInput, TermSource};
use lucene_codecs::segment_state::{SegmentReadState, SegmentWriteState};

/// Java: `TestBloomFilteredLucenePostings.LowMemoryBloomFactory`.
#[derive(Debug)]
struct LowMemoryBloomFactory;

impl BloomFilterFactory for LowMemoryBloomFactory {
    fn get_set_for_field(
        &self,
        _state: &SegmentWriteState<'_>,
        _info: &FieldInfo,
    ) -> lucene_store::Result<Option<FuzzySet>> {
        FuzzySet::create_set_based_on_max_memory(1024).map(Some)
    }

    fn is_saturated(&self, _bloom_filter: &FuzzySet, _info: &FieldInfo) -> bool {
        false
    }
}

/// Java's test-only bloom format and its fixed low-memory factory.
#[derive(Debug)]
pub struct TestBloomFilteredLucenePostings {
    delegate: BloomFilteringPostingsFormat<LowMemoryBloomFactory>,
}

impl Default for TestBloomFilteredLucenePostings {
    fn default() -> Self {
        Self {
            delegate: BloomFilteringPostingsFormat::with_factory(
                Lucene104PostingsFormat::default(),
                LowMemoryBloomFactory,
            ),
        }
    }
}

impl TestBloomFilteredLucenePostings {
    /// Java's `getName()` (the delegate still writes `BloomFilter` headers).
    pub fn name(&self) -> &'static str {
        "TestBloomFilteredLucenePostings"
    }

    /// Java: `fieldsConsumer(state).write(fields, norms)` and `close()`.
    ///
    /// # Errors
    /// The delegate's postings or bloom writer errors.
    pub fn fields_consumer<T: TermSource>(
        &self,
        inputs: &[FieldPostingsInput<'_, T>],
        norms: &[FieldNorms<'_>],
        state: &SegmentWriteState<'_>,
    ) -> Result<BloomOutput> {
        self.delegate.write(inputs, norms, state)
    }

    /// Java: `fieldsProducer(state)` over the delegate's file buffers.
    ///
    /// # Errors
    /// The delegate's header, checksum, field or dictionary errors.
    pub fn fields_producer(
        &self,
        blm: &[u8],
        tim: &[u8],
        tip: &[u8],
        tmd: &[u8],
        state: &SegmentReadState<'_>,
    ) -> Result<BloomFilteredFieldsProducer> {
        BloomFilteredFieldsProducer::open(blm, tim, tip, tmd, state)
    }
}

impl fmt::Display for TestBloomFilteredLucenePostings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TestBloomFilteredLucenePostings({})", self.delegate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_codecs::field_infos::FieldInfos;

    #[test]
    fn every_field_gets_the_small_filter_even_when_it_is_saturated() {
        let infos = FieldInfos::new(vec![FieldInfo::new("body", 0)]).unwrap();
        let state = SegmentWriteState {
            max_doc: 0,
            segment_id: &[7; 16],
            segment_suffix: "",
            field_infos: &infos,
        };
        let factory = LowMemoryBloomFactory;
        let info = &infos.fields[0];
        let mut set = factory.get_set_for_field(&state, info).unwrap().unwrap();
        assert_eq!(set.bloom_size(), 8191);
        assert_eq!(set.hash_count(), 1);
        for i in 0..100_000 {
            set.add_value(format!("term{i}").as_bytes());
        }
        assert!(set.saturation() > 0.99);
        assert!(!factory.is_saturated(&set, info));
        let format = TestBloomFilteredLucenePostings::default();
        assert_eq!(format.name(), "TestBloomFilteredLucenePostings");
        assert_eq!(format.to_string(), "TestBloomFilteredLucenePostings(BloomFilteringPostingsFormat(PostingsFormat(name=Lucene104)))");
        let state = SegmentReadState {
            max_doc: state.max_doc,
            segment_id: state.segment_id,
            segment_suffix: state.segment_suffix,
            field_infos: &infos,
        };
        assert!(format.fields_producer(&[], &[], &[], &[], &state).is_err());
    }
}
