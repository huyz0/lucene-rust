//! The two uniform-split ROT13 test formats. Every encoded part has seven
//! zero prefix bytes followed by its bytes plus 13, wrapping as Java bytes.
//! Decoding skips the prefix without checking it, then subtracts 13.
//!
//! Java's five static volatile flags are shared by both formats. Dictionary
//! and metadata encoder hooks stand in for Java's protected writer overrides.
//! Encoder write-to-output and flag updates are one operation over a Vec;
//! ownership replaces close-on-failure, and files are passed by extension.
//! Readers expose native dictionaries; callers open postings separately.
//! Dictionary-on-heap is recorded as in the underlying uniform-split format
//! (all dictionaries are decoded at open). IndexWriter routing, shared-terms
//! merge-state support and other inherited gaps remain in the parity ledger.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use lucene_codecs::postings_writer::{FieldNorms, FieldPostingsInput, TermSource};
use lucene_codecs::segment_state::{SegmentReadState, SegmentWriteState};
use lucene_codecs::uniformsplit::sharedterms::{
    StUniformSplitPostingsFormat, StUniformSplitTermsReader,
};
use lucene_codecs::uniformsplit::{
    BlockDecoder, BlockEncoder, Error, Result, UniformSplitOutput, UniformSplitPostingsFormat,
    UniformSplitTermsReader, DEFAULT_DELTA_NUM_LINES, DEFAULT_TARGET_NUM_BLOCK_LINES,
};

/// Java: `Rot13CypherTestUtil`, including its untracked block encoder/decoder.
#[derive(Debug, Clone, Copy, Default)]
pub struct Rot13Cypher;

impl Rot13Cypher {
    /// Java: `Rot13CypherTestUtil.encode`.
    ///
    /// # Errors
    /// A length/allocation that cannot hold the encoded bytes.
    pub fn encode(bytes: &[u8]) -> Result<Vec<u8>> {
        let length = bytes
            .len()
            .checked_add(7)
            .ok_or_else(|| Error::IllegalArgument("ROT13 encoded length overflow".into()))?;
        let mut encoded = Vec::new();
        encoded
            .try_reserve_exact(length)
            .map_err(|e| Error::IllegalArgument(format!("ROT13 allocation: {e}")))?;
        encoded.resize(7, 0);
        encoded.extend(bytes.iter().map(|b| b.wrapping_add(13)));
        Ok(encoded)
    }

    /// Java: `Rot13CypherTestUtil.decode`. Prefix values are deliberately ignored.
    ///
    /// # Errors
    /// A block shorter than seven bytes is `Corrupted`, rather than Java's
    /// negative array size or input exception.
    pub fn decode(bytes: &[u8]) -> Result<Vec<u8>> {
        let body = bytes.get(7..).ok_or_else(|| {
            Error::Store(lucene_store::Error::Corrupted(
                "ROT13 block shorter than 7 bytes".into(),
            ))
        })?;
        Ok(body.iter().map(|b| b.wrapping_sub(13)).collect())
    }
}

impl BlockEncoder for Rot13Cypher {
    fn encode(&self, bytes: &[u8]) -> Result<Vec<u8>> {
        Self::encode(bytes)
    }
}

impl BlockDecoder for Rot13Cypher {
    fn decode(&self, bytes: &[u8]) -> Result<Vec<u8>> {
        Self::decode(bytes)
    }
}

/// Snapshot of `UniformSplitRot13PostingsFormat`'s five static flags.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EncodingState {
    pub encoder_called: bool,
    pub decoder_called: bool,
    pub blocks_encoded: bool,
    pub fields_metadata_encoded: bool,
    pub dictionary_encoded: bool,
}

/// Sequentially consistent atomics implement Java's static volatile booleans.
pub struct EncodingFlags {
    encoder_called: AtomicBool,
    decoder_called: AtomicBool,
    blocks_encoded: AtomicBool,
    fields_metadata_encoded: AtomicBool,
    dictionary_encoded: AtomicBool,
}

/// Shared by both formats, as Java's inherited static fields are.
pub static ENCODING_FLAGS: EncodingFlags = EncodingFlags {
    encoder_called: AtomicBool::new(false),
    decoder_called: AtomicBool::new(false),
    blocks_encoded: AtomicBool::new(false),
    fields_metadata_encoded: AtomicBool::new(false),
    dictionary_encoded: AtomicBool::new(false),
};

impl EncodingFlags {
    /// Java: `UniformSplitRot13PostingsFormat.resetEncodingFlags`.
    pub fn reset(&self) {
        self.encoder_called.store(false, Ordering::SeqCst);
        self.decoder_called.store(false, Ordering::SeqCst);
        self.blocks_encoded.store(false, Ordering::SeqCst);
        self.fields_metadata_encoded.store(false, Ordering::SeqCst);
        self.dictionary_encoded.store(false, Ordering::SeqCst);
    }

    pub fn snapshot(&self) -> EncodingState {
        EncodingState {
            encoder_called: self.encoder_called.load(Ordering::SeqCst),
            decoder_called: self.decoder_called.load(Ordering::SeqCst),
            blocks_encoded: self.blocks_encoded.load(Ordering::SeqCst),
            fields_metadata_encoded: self.fields_metadata_encoded.load(Ordering::SeqCst),
            dictionary_encoded: self.dictionary_encoded.load(Ordering::SeqCst),
        }
    }

    // Java's if/read, then flag writes are separate volatile operations,
    // not an atomic exchange (which would strengthen concurrent semantics).
    fn record(&self, flag: &AtomicBool) {
        if self.encoder_called.load(Ordering::SeqCst) {
            flag.store(true, Ordering::SeqCst);
            self.encoder_called.store(false, Ordering::SeqCst);
        }
    }
}

struct TrackedEncoder {
    shared: bool,
}

impl BlockEncoder for TrackedEncoder {
    fn encode(&self, bytes: &[u8]) -> Result<Vec<u8>> {
        let encoded = Rot13Cypher::encode(bytes)?;
        ENCODING_FLAGS.encoder_called.store(true, Ordering::SeqCst);
        Ok(encoded)
    }

    fn encode_dictionary(&self, bytes: &[u8]) -> Result<Vec<u8>> {
        // Java: the anonymous writer's `writeDictionary` override.
        ENCODING_FLAGS.record(&ENCODING_FLAGS.blocks_encoded);
        let encoded = self.encode(bytes)?;
        ENCODING_FLAGS.record(&ENCODING_FLAGS.dictionary_encoded);
        Ok(encoded)
    }

    fn encode_fields_metadata(&self, bytes: &[u8]) -> Result<Vec<u8>> {
        // Java: ST's override records the final block before metadata;
        // ordinary uniform-split has already recorded it at its dictionary.
        if self.shared {
            ENCODING_FLAGS.record(&ENCODING_FLAGS.blocks_encoded);
        }
        let encoded = self.encode(bytes)?;
        ENCODING_FLAGS.record(&ENCODING_FLAGS.fields_metadata_encoded);
        Ok(encoded)
    }
}

struct TrackedDecoder;

impl BlockDecoder for TrackedDecoder {
    fn decode(&self, bytes: &[u8]) -> Result<Vec<u8>> {
        ENCODING_FLAGS.decoder_called.store(true, Ordering::SeqCst);
        Rot13Cypher::decode(bytes)
    }
}

/// Java: `UniformSplitRot13PostingsFormat`, 32 +/- 3 lines per block.
#[derive(Debug, Clone, Copy, Default)]
pub struct UniformSplitRot13PostingsFormat {
    dictionary_on_heap: bool,
}

impl UniformSplitRot13PostingsFormat {
    /// Java's protected constructor's dictionary setting. The Rust codec
    /// always decodes dictionaries at open, so this setting is only recorded.
    pub fn new(dictionary_on_heap: bool) -> Self {
        Self { dictionary_on_heap }
    }
    pub fn name(&self) -> &'static str {
        "UniformSplitRot13"
    }
    pub fn dictionary_on_heap(&self) -> bool {
        self.dictionary_on_heap
    }

    /// Java: `fieldsConsumer(state).write(fields, norms)` and `close()`.
    ///
    /// # Errors
    /// The underlying uniform-split/postings writer's input and encoding errors.
    pub fn fields_consumer<T: TermSource>(
        &self,
        inputs: &[FieldPostingsInput<'_, T>],
        norms: &[FieldNorms<'_>],
        state: &SegmentWriteState<'_>,
    ) -> Result<BTreeMap<String, Vec<u8>>> {
        Ok(files(self.format()?.write(inputs, norms, state)?, false))
    }

    /// Java: `fieldsProducer`; postings inputs are opened separately.
    ///
    /// # Errors
    /// Missing/damaged dictionary files or invalid encoded bytes.
    pub fn fields_producer(
        &self,
        files: &BTreeMap<String, Vec<u8>>,
        state: &SegmentReadState<'_>,
    ) -> Result<UniformSplitTermsReader> {
        self.format()?
            .open(file(files, "ustb")?.to_vec(), file(files, "ustd")?, state)
    }

    fn format(&self) -> Result<UniformSplitPostingsFormat> {
        UniformSplitPostingsFormat::new(
            DEFAULT_TARGET_NUM_BLOCK_LINES,
            DEFAULT_DELTA_NUM_LINES,
            Some(Arc::new(TrackedEncoder { shared: false })),
            Some(Arc::new(TrackedDecoder)),
            self.dictionary_on_heap,
        )
    }
}

/// Java: `STUniformSplitRot13PostingsFormat`, one dictionary shared by fields.
#[derive(Debug, Clone, Copy, Default)]
pub struct StUniformSplitRot13PostingsFormat;

impl StUniformSplitRot13PostingsFormat {
    pub fn name(&self) -> &'static str {
        "STUniformSplitRot13"
    }

    /// Java: `fieldsConsumer(state).write(fields, norms)` and `close()`.
    ///
    /// # Errors
    /// The underlying shared-terms/postings writer's input and encoding errors.
    pub fn fields_consumer<T: TermSource>(
        &self,
        inputs: &[FieldPostingsInput<'_, T>],
        norms: &[FieldNorms<'_>],
        state: &SegmentWriteState<'_>,
    ) -> Result<BTreeMap<String, Vec<u8>>> {
        Ok(files(self.format()?.write(inputs, norms, state)?, true))
    }

    /// Java: `fieldsProducer`; postings inputs are opened separately.
    ///
    /// # Errors
    /// Missing/damaged dictionary files or invalid encoded bytes.
    pub fn fields_producer(
        &self,
        files: &BTreeMap<String, Vec<u8>>,
        state: &SegmentReadState<'_>,
    ) -> Result<StUniformSplitTermsReader> {
        self.format()?.open(
            file(files, "stustb")?.to_vec(),
            file(files, "stustd")?,
            state,
        )
    }

    fn format(&self) -> Result<StUniformSplitPostingsFormat> {
        StUniformSplitPostingsFormat::new(
            DEFAULT_TARGET_NUM_BLOCK_LINES,
            DEFAULT_DELTA_NUM_LINES,
            Some(Arc::new(TrackedEncoder { shared: true })),
            Some(Arc::new(TrackedDecoder)),
            false,
        )
    }
}

fn file<'a>(files: &'a BTreeMap<String, Vec<u8>>, extension: &str) -> Result<&'a [u8]> {
    files
        .get(extension)
        .map(Vec::as_slice)
        .ok_or_else(|| Error::IllegalArgument(format!("missing {extension} file for ROT13")))
}

fn files(output: UniformSplitOutput, shared: bool) -> BTreeMap<String, Vec<u8>> {
    let mut files = BTreeMap::from([
        ("doc".into(), output.postings.doc),
        ("psm".into(), output.postings.psm),
        (if shared { "stustb" } else { "ustb" }.into(), output.ustb),
        (if shared { "stustd" } else { "ustd" }.into(), output.ustd),
    ]);
    if !output.postings.pos.is_empty() {
        files.insert("pos".into(), output.postings.pos);
    }
    if !output.postings.pay.is_empty() {
        files.insert("pay".into(), output.postings.pay);
    }
    files
}

impl fmt::Display for UniformSplitRot13PostingsFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PostingsFormat(name={})", self.name())
    }
}
impl fmt::Display for StUniformSplitRot13PostingsFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PostingsFormat(name={})", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn cypher_roundtrips_every_byte_sequence(bytes in prop::collection::vec(any::<u8>(), 0..8192)) {
            let encoded = Rot13Cypher::encode(&bytes).unwrap();
            prop_assert_eq!(&encoded[..7], &[0; 7]);
            prop_assert_eq!(Rot13Cypher::decode(&encoded).unwrap(), bytes);
        }
    }

    #[test]
    fn malformed_blocks_and_nonzero_prefixes() {
        for length in 0..7 {
            assert!(matches!(
                Rot13Cypher::decode(&vec![0; length]),
                Err(Error::Store(_))
            ));
        }
        assert!(Rot13Cypher::decode(&[99; 7]).unwrap().is_empty());
        assert_eq!(
            Rot13Cypher::decode(&[99, 8, 7, 6, 5, 4, 3, 0, 255]).unwrap(),
            [243, 242]
        );
        assert_eq!(
            Rot13Cypher::encode(&[0, 127, 243, 255]).unwrap(),
            [0, 0, 0, 0, 0, 0, 0, 13, 140, 0, 12]
        );
    }

    #[test]
    fn flags_optional_files_and_writer_errors() {
        use lucene_codecs::field_infos::{FieldInfo, FieldInfos, IndexOptions};
        use lucene_codecs::postings_writer::TermPostings;

        let normal = UniformSplitRot13PostingsFormat::default();
        assert!(!normal.dictionary_on_heap());
        assert!(UniformSplitRot13PostingsFormat::new(true).dictionary_on_heap());
        assert_eq!(normal.to_string(), "PostingsFormat(name=UniformSplitRot13)");
        let shared = StUniformSplitRot13PostingsFormat;
        assert_eq!(
            shared.to_string(),
            "PostingsFormat(name=STUniformSplitRot13)"
        );
        ENCODING_FLAGS.reset();
        assert_eq!(ENCODING_FLAGS.snapshot(), EncodingState::default());
        // No preceding block: the dictionary's hook must leave blocks false.
        assert_eq!(
            TrackedEncoder { shared: false }
                .encode_dictionary(&[255])
                .unwrap(),
            [0, 0, 0, 0, 0, 0, 0, 12]
        );
        assert_eq!(
            ENCODING_FLAGS.snapshot(),
            EncodingState {
                dictionary_encoded: true,
                ..EncodingState::default()
            }
        );
        assert!(TrackedDecoder.decode(&[]).is_err());
        assert!(ENCODING_FLAGS.snapshot().decoder_called);
        ENCODING_FLAGS.reset();
        assert_eq!(ENCODING_FLAGS.snapshot(), EncodingState::default());

        let infos = FieldInfos::new(vec![FieldInfo::new("body", 0)
            .with_index_options(IndexOptions::DocsAndFreqsAndPositionsAndOffsets)])
        .unwrap();
        let id = [7; 16];
        let state = SegmentWriteState {
            max_doc: 1,
            segment_id: &id,
            segment_suffix: "",
            field_infos: &infos,
        };
        let mut terms = vec![TermPostings {
            term: b"a".to_vec(),
            docs: vec![(0, 1)],
            positions: vec![vec![0]],
            offsets: vec![vec![(0, 1)]],
            ..TermPostings::default()
        }];
        for is_shared in [false, true] {
            let input = FieldPostingsInput {
                field_number: 0,
                index_options: IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
                doc_count: 1,
                has_payloads: false,
                terms: &terms,
            };
            ENCODING_FLAGS.reset();
            let files = if is_shared {
                shared.fields_consumer(&[input], &[], &state)
            } else {
                normal.fields_consumer(&[input], &[], &state)
            }
            .unwrap();
            assert!(files.contains_key("pos"));
            assert!(files.contains_key("pay"));
            assert_eq!(
                ENCODING_FLAGS.snapshot(),
                EncodingState {
                    blocks_encoded: true,
                    dictionary_encoded: true,
                    fields_metadata_encoded: true,
                    ..EncodingState::default()
                }
            );
        }
        terms[0].docs[0].1 = 0;
        for is_shared in [false, true] {
            let input = FieldPostingsInput {
                field_number: 0,
                index_options: IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
                doc_count: 1,
                has_payloads: false,
                terms: &terms,
            };
            let result = if is_shared {
                shared.fields_consumer(&[input], &[], &state)
            } else {
                normal.fields_consumer(&[input], &[], &state)
            };
            assert!(matches!(result, Err(Error::Write(_))));
        }
        terms[0].docs[0].1 = 1;
        terms[0].positions.clear();
        terms[0].offsets.clear();
        let infos = FieldInfos::new(vec![
            FieldInfo::new("body", 0).with_index_options(IndexOptions::DocsAndFreqs)
        ])
        .unwrap();
        let state = SegmentWriteState {
            field_infos: &infos,
            ..state
        };
        for is_shared in [false, true] {
            let input = FieldPostingsInput {
                field_number: 0,
                index_options: IndexOptions::DocsAndFreqs,
                doc_count: 1,
                has_payloads: false,
                terms: &terms,
            };
            let files = if is_shared {
                shared.fields_consumer(&[input], &[], &state)
            } else {
                normal.fields_consumer(&[input], &[], &state)
            }
            .unwrap();
            assert!(!files.contains_key("pos"));
            assert!(!files.contains_key("pay"));
        }
    }
}
