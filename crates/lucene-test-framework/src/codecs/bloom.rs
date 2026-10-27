//! `TestBloomFilteredLucenePostings`: the default Lucene104 postings
//! wrapped in `BloomFilteringPostingsFormat` with a 1024-byte filter for
//! every field. Its `LowMemoryBloomFactory` keeps even saturated filters;
//! the parent factory's normal downsizing still applies.
//!
//! Java: `fieldsConsumer` and `fieldsProducer` delegate to the bloom
//! format. Rust passes file buffers and postings values instead of a
//! Directory and streaming FieldsConsumer. IndexWriter routes fields to it
//! through [`TestBloomFilteredLucenePostings::dictionary_format`], once
//! [`super::register_postings_formats`] has made it readable by name.

use std::fmt;
use std::sync::LazyLock;

use lucene_codecs::blocktree;
use lucene_codecs::bloom::{
    BloomFilterFactory, BloomFilteredFieldsProducer, BloomFilteringPostingsFormat, BloomOutput,
    Error, FuzzySet, Result,
};
use lucene_codecs::field_infos::FieldInfo;
use lucene_codecs::per_field_postings::Lucene104PostingsFormat;
use lucene_codecs::postings_writer::{
    self, DictionaryField, FieldNorms, FieldPostingsInput, PostingsFiles, TermSource, WriteOptions,
};
use lucene_codecs::segment_state::{SegmentReadState, SegmentWriteState};
use lucene_codecs::terms_dictionary::{ErasedInput, TermsDictionaryFormat};

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

impl TestBloomFilteredLucenePostings {
    /// The format as IndexWriter routes a field to it
    /// (`Lucene104PostingsFormat::dictionary`) and reads its groups back.
    pub fn dictionary_format() -> &'static dyn TermsDictionaryFormat {
        static FORMAT: LazyLock<TestBloomFilteredLucenePostings> =
            LazyLock::new(TestBloomFilteredLucenePostings::default);
        &*FORMAT
    }
}

/// The group's dictionary is the delegate's block tree behind the `.blm`.
impl TermsDictionaryFormat for TestBloomFilteredLucenePostings {
    fn name(&self) -> &'static str {
        TestBloomFilteredLucenePostings::name(self)
    }

    fn extensions(&self) -> &'static [&'static str] {
        &["blm", "tim", "tip", "tmd"]
    }

    fn write(
        &self,
        inputs: &[ErasedInput<'_, '_>],
        norms: &[FieldNorms<'_>],
        _options: &WriteOptions,
        state: &SegmentWriteState<'_>,
    ) -> postings_writer::Result<(PostingsFiles, Vec<Vec<u8>>)> {
        let out = self
            .fields_consumer(inputs, norms, state)
            .map_err(|e| match e {
                Error::Write(e) => e,
                Error::Store(e) => postings_writer::Error::Store(e),
                e => postings_writer::Error::Dictionary(e.to_string()),
            })?;
        let d = out.delegate;
        let postings = PostingsFiles {
            doc: d.doc,
            psm: d.psm,
            pos: d.pos,
            pay: d.pay,
        };
        Ok((postings, vec![out.blm, d.tim, d.tip, d.tmd]))
    }

    fn read(
        &self,
        files: &[&[u8]],
        state: &SegmentReadState<'_>,
    ) -> blocktree::Result<Vec<DictionaryField>> {
        let [blm, tim, tip, tmd] = files else {
            return Err(blocktree::Error::Unsupported(
                "a bloom group needs its .blm, .tim, .tip and .tmd",
            ));
        };
        let producer = self
            .fields_producer(blm, tim, tip, tmd, state)
            .map_err(|e| match e {
                Error::Store(e) => blocktree::Error::Store(e),
                Error::BlockTree(e) => e,
                e => blocktree::Error::ForeignDictionary {
                    format: "TestBloomFilteredLucenePostings",
                    message: e.to_string(),
                },
            })?;
        producer.delegate().dictionary_fields(state.field_infos)
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
