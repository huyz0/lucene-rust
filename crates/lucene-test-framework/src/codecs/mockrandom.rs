//! `MockRandomPostingsFormat`: a seeded choice of FST, block tree,
//! block terms (fixed, every-N, doc-frequency or random gaps), or ordinal
//! block tree, all over Lucene104 postings.
//!
//! `.sd` is IndexHeader("MockRandomSeed", 0), a little-endian long seed,
//! and a checked footer. The seed selects the dictionary at reopen; writer
//! settings are kept in its own dictionary files. The constructor consumes
//! the caller's nextLong, then each writer consumes an unused skip-interval
//! draw before its seed. Readers consume the buffer-size draw before choosing.
//!
//! Rust accepts/returns files by extension and exposes the selected native
//! dictionary reader as an enum, rather than Java's FieldsProducer inheritance.
//! Postings inputs are opened and passed separately to its term enums; this
//! constructor checks the seed and dictionary files, not the postings files.
//! Fields must be in name order. Ownership replaces close-on-failure; a failed write
//! returns no files, while Java may leave partial files in its Directory.
//! IndexWriter routes fields to an instance through
//! [`MockRandomPostingsFormat::into_dictionary_format`]; a group is read back
//! by the read-only instance [`super::register_postings_formats`] registers.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{LazyLock, Mutex};

use lucene_codecs::blockterms::{self, IndexTermSelector, TermStats, VariableGapTermsIndexWriter};
use lucene_codecs::blocktree::{self, BlockTreeFields};
use lucene_codecs::blocktreeords::{BlockTreeOrdsPostingsFormat, OrdsBlockTreeTermsReader};
use lucene_codecs::field_infos::FieldInfo;
use lucene_codecs::memory::{FstPostingsFormat, FstTermsReader};
use lucene_codecs::postings_writer::{
    self, DictionaryField, FieldNorms, FieldPostingsInput, PostingsFiles, TermSource, WriteOptions,
};
use lucene_codecs::segment_state::{SegmentReadState, SegmentWriteState};
use lucene_codecs::terms_dictionary::{ErasedInput, TermsDictionaryFormat};
use lucene_store::codec_util;
use lucene_store::data_input::{DataInput, SliceInput};
use lucene_store::data_output::DataOutput;
use lucene_util::java_random::JavaRandom;

use super::blockterms::BlockTermsFormat;
use crate::util::random_numbers::random_int_between;

/// Errors from the selected dictionary or MockRandom's seed file.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
    #[error(transparent)]
    Write(#[from] postings_writer::Error),
    #[error(transparent)]
    BlockTree(#[from] blocktree::Error),
    #[error(transparent)]
    BlockTerms(#[from] blockterms::Error),
    #[error(transparent)]
    Fst(#[from] lucene_codecs::memory::Error),
    #[error(transparent)]
    Ords(#[from] lucene_codecs::blocktreeords::Error),
    #[error("missing {0} file for MockRandom")]
    MissingFile(&'static str),
    #[error("Please use MockRandomPostingsFormat(Random)")]
    ReadOnly,
}

pub type Result<T> = std::result::Result<T, Error>;

/// Java's source random is instance state; Default is the read-only SPI constructor.
#[derive(Debug, Default)]
pub struct MockRandomPostingsFormat {
    seed_random: Option<JavaRandom>,
}

#[derive(Debug, PartialEq, Eq)]
enum WriterChoice {
    Fst,
    BlockTree(usize, usize),
    Fixed(i32),
    EveryN(i32),
    DocFreq(i32, i32),
    RandomGap(i64, i32),
    Ords(usize, usize),
}

impl WriterChoice {
    /// Java: `fieldsConsumer`, after the seed has been saved.
    fn from_seed(seed: i64) -> Self {
        let mut r = JavaRandom::new(seed);
        r.next_int(); // Java's unused buffer-size draw.
        match r.next_int_bounded(4) {
            0 => Self::Fst,
            kind @ (1 | 3) => {
                let min = random_int_between(&mut r, 2, 100) as usize;
                let max = (2 * (min - 1) + r.next_int_bounded(100) as usize).max(2);
                if kind == 1 {
                    Self::BlockTree(min, max)
                } else {
                    Self::Ords(min, max)
                }
            }
            _ => {
                if r.next_boolean() {
                    Self::Fixed(random_int_between(&mut r, 1, 100))
                } else {
                    match r.next_int_bounded(3) {
                        0 => Self::EveryN(random_int_between(&mut r, 1, 100)),
                        1 => Self::DocFreq(
                            random_int_between(&mut r, 2, 100),
                            random_int_between(&mut r, 1, 100),
                        ),
                        _ => Self::RandomGap(r.next_long(), random_int_between(&mut r, 2, 40)),
                    }
                }
            }
        }
    }
}

/// Java's anonymous `IndexTermSelector`: its random is shared across fields.
struct RandomGapSelector {
    random: JavaRandom,
    gap: i32,
}

impl IndexTermSelector for RandomGapSelector {
    fn is_index_term(&mut self, _term: &[u8], _stats: TermStats) -> bool {
        self.random.next_int_bounded(self.gap) == self.gap / 2
    }
    fn new_field(&mut self, _field: &FieldInfo) {}
}

/// The native dictionary reader Java selects in `fieldsProducer`.
pub enum MockRandomFieldsProducer {
    Fst(FstTermsReader),
    BlockTree(BlockTreeFields),
    BlockTerms(blockterms::BlockTermsReader),
    Ords(OrdsBlockTreeTermsReader),
}

impl MockRandomPostingsFormat {
    /// Java: `MockRandomPostingsFormat(Random)`.
    pub fn new(random: &mut JavaRandom) -> Self {
        Self {
            seed_random: Some(JavaRandom::new(random.next_long())),
        }
    }

    pub fn name(&self) -> &'static str {
        "MockRandom"
    }

    /// Java: the skip-interval draw preceding each writer's seed.
    fn next_seed(&mut self, max_doc: i32) -> Result<i64> {
        let r = self.seed_random.as_mut().ok_or(Error::ReadOnly)?;
        let min = if max_doc > 1_000_000 { 3 } else { 2 };
        random_int_between(r, min, 10);
        Ok(r.next_long())
    }

    /// Java: `fieldsConsumer(state).write(fields, norms)` and `close()`.
    ///
    /// # Errors
    /// A read-only constructor, invalid postings or an underlying writer error.
    pub fn fields_consumer<T: TermSource>(
        &mut self,
        inputs: &[FieldPostingsInput<'_, T>],
        norms: &[FieldNorms<'_>],
        state: &SegmentWriteState<'_>,
    ) -> Result<BTreeMap<String, Vec<u8>>> {
        let seed = self.next_seed(state.max_doc)?;
        let mut files = match WriterChoice::from_seed(seed) {
            WriterChoice::Fst => {
                let output = FstPostingsFormat.write(inputs, norms, state)?;
                let mut files = postings_files(output.postings);
                files.insert("tfp".into(), output.tfp);
                files
            }
            WriterChoice::BlockTree(min, max) => {
                let options = state.postings_options(WriteOptions {
                    min_items_in_block: min,
                    max_items_in_block: max,
                    ..WriteOptions::default()
                });
                let output = postings_writer::write_fields_with_options(
                    inputs,
                    norms,
                    &options,
                    state.segment_id,
                    state.segment_suffix,
                )?;
                let mut files = BTreeMap::from([
                    ("doc".into(), output.doc),
                    ("psm".into(), output.psm),
                    ("tim".into(), output.tim),
                    ("tip".into(), output.tip),
                    ("tmd".into(), output.tmd),
                ]);
                if !output.pos.is_empty() {
                    files.insert("pos".into(), output.pos);
                }
                if !output.pay.is_empty() {
                    files.insert("pay".into(), output.pay);
                }
                files
            }
            WriterChoice::Fixed(term_index_interval) => BlockTermsFormat::FixedGap {
                term_index_interval,
            }
            .fields_consumer(inputs, norms, state)?,
            WriterChoice::EveryN(term_index_interval) => BlockTermsFormat::VarGapFixedInterval {
                term_index_interval,
            }
            .fields_consumer(inputs, norms, state)?,
            WriterChoice::DocFreq(doc_freq_threshold, term_index_interval) => {
                BlockTermsFormat::VarGapDocFreqInterval {
                    doc_freq_threshold,
                    term_index_interval,
                }
                .fields_consumer(inputs, norms, state)?
            }
            WriterChoice::RandomGap(seed, gap) => {
                let index = VariableGapTermsIndexWriter::new(
                    state,
                    RandomGapSelector {
                        random: JavaRandom::new(seed),
                        gap,
                    },
                );
                let output = blockterms::write(index, inputs, norms, state)?;
                let mut files = postings_files(output.postings);
                files.insert("tib".into(), output.tib);
                files.insert("tmv".into(), output.index.tmv);
                files.insert("tiv".into(), output.index.tiv);
                files
            }
            WriterChoice::Ords(min, max) => {
                let output =
                    BlockTreeOrdsPostingsFormat::new(min, max)?.write(inputs, norms, state)?;
                let mut files = postings_files(output.postings);
                files.insert("tio".into(), output.tio);
                files.insert("tipo".into(), output.tipo);
                files
            }
        };
        let mut sd = Vec::new();
        codec_util::write_index_header(
            &mut sd,
            "MockRandomSeed",
            0,
            state.segment_id,
            state.segment_suffix,
        );
        sd.write_i64(seed);
        codec_util::write_footer(&mut sd);
        files.insert("sd".into(), sd);
        Ok(files)
    }

    /// Java: `fieldsProducer`, including the buffer draw and reader-only
    /// variable-gap draws (which diverge from the writer's after nextBoolean).
    ///
    /// # Errors
    /// A missing file, damaged seed header/footer, or underlying reader error.
    pub fn fields_producer(
        &self,
        files: &BTreeMap<String, Vec<u8>>,
        state: &SegmentReadState<'_>,
    ) -> Result<MockRandomFieldsProducer> {
        let file = |ext: &'static str| files.get(ext).ok_or(Error::MissingFile(ext));
        let sd = file("sd")?;
        let mut input = SliceInput::new(sd);
        codec_util::check_index_header(
            &mut input,
            "MockRandomSeed",
            0,
            0,
            state.segment_id,
            state.segment_suffix,
        )?;
        let seed = input.read_i64()?;
        codec_util::check_footer(&mut input, sd.len())?;
        let mut random = JavaRandom::new(seed);
        random_int_between(&mut random, 1, 4096);
        Ok(match random.next_int_bounded(4) {
            0 => MockRandomFieldsProducer::Fst(FstTermsReader::open(file("tfp")?, state)?),
            1 => MockRandomFieldsProducer::BlockTree(blocktree::open(
                file("tim")?,
                file("tip")?,
                file("tmd")?,
                state.field_infos,
                state.segment_id,
                state.segment_suffix,
                state.max_doc,
            )?),
            2 => {
                let format = if random.next_boolean() {
                    BlockTermsFormat::fixed_gap()
                } else {
                    match random.next_int_bounded(3) {
                        1 => {
                            random.next_int();
                        }
                        2 => {
                            random.next_long();
                        }
                        _ => {}
                    }
                    BlockTermsFormat::var_gap_fixed_interval()
                };
                MockRandomFieldsProducer::BlockTerms(format.fields_producer(files, state)?)
            }
            _ => MockRandomFieldsProducer::Ords(OrdsBlockTreeTermsReader::open(
                file("tio")?.clone(),
                file("tipo")?,
                state,
            )?),
        })
    }
}

/// Every dictionary file a seed may select, the seed file first.
const DICTIONARY_EXTENSIONS: &[&str] = &[
    "sd", "tfp", "tim", "tip", "tmd", "tib", "tii", "tmv", "tiv", "tio", "tipo",
];

/// A `MockRandomPostingsFormat` as IndexWriter routes fields to it: every
/// flush and merge draws the next writer's seed, so the format sits behind
/// a lock.
#[derive(Debug)]
pub struct RoutedMockRandom(Mutex<MockRandomPostingsFormat>);

impl MockRandomPostingsFormat {
    /// This instance as `Lucene104PostingsFormat::dictionary` takes it. It
    /// is leaked: a routed format lives as long as the writers using it, as
    /// Java's codec instance does.
    pub fn into_dictionary_format(self) -> &'static dyn TermsDictionaryFormat {
        Box::leak(Box::new(RoutedMockRandom(Mutex::new(self))))
    }

    /// The read-only SPI instance, which reads every `MockRandom` group.
    pub fn read_only() -> &'static dyn TermsDictionaryFormat {
        static FORMAT: LazyLock<RoutedMockRandom> =
            LazyLock::new(|| RoutedMockRandom(Mutex::new(MockRandomPostingsFormat::default())));
        &*FORMAT
    }
}

impl TermsDictionaryFormat for RoutedMockRandom {
    fn name(&self) -> &'static str {
        "MockRandom"
    }

    /// The seed picks the dictionary, so only `.sd` is always written.
    fn extensions(&self) -> &'static [&'static str] {
        DICTIONARY_EXTENSIONS
    }

    fn write(
        &self,
        inputs: &[ErasedInput<'_, '_>],
        norms: &[FieldNorms<'_>],
        _options: &WriteOptions,
        state: &SegmentWriteState<'_>,
    ) -> postings_writer::Result<(PostingsFiles, Vec<Vec<u8>>)> {
        let mut format = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let mut files = format
            .fields_consumer(inputs, norms, state)
            .map_err(|e| match e {
                Error::Write(e) => e,
                Error::Store(e) => postings_writer::Error::Store(e),
                e => postings_writer::Error::Dictionary(e.to_string()),
            })?;
        let mut take = |ext: &str| files.remove(ext).unwrap_or_default();
        let postings = PostingsFiles {
            doc: take("doc"),
            psm: take("psm"),
            pos: take("pos"),
            pay: take("pay"),
        };
        let dictionary = DICTIONARY_EXTENSIONS.iter().map(|ext| take(ext)).collect();
        Ok((postings, dictionary))
    }

    fn read(
        &self,
        files: &[&[u8]],
        state: &SegmentReadState<'_>,
    ) -> blocktree::Result<Vec<DictionaryField>> {
        let err = |e: Error| match e {
            Error::Store(e) => blocktree::Error::Store(e),
            Error::BlockTree(e) => e,
            e => blocktree::Error::ForeignDictionary {
                format: "MockRandom",
                message: e.to_string(),
            },
        };
        let files: BTreeMap<String, Vec<u8>> = DICTIONARY_EXTENSIONS
            .iter()
            .zip(files)
            .filter(|(_, bytes)| !bytes.is_empty())
            .map(|(ext, bytes)| (ext.to_string(), bytes.to_vec()))
            .collect();
        let format = self.0.lock().unwrap_or_else(|e| e.into_inner());
        match format.fields_producer(&files, state).map_err(err)? {
            MockRandomFieldsProducer::Fst(r) => r.dictionary_fields().map_err(|e| err(e.into())),
            MockRandomFieldsProducer::BlockTree(f) => f.dictionary_fields(state.field_infos),
            MockRandomFieldsProducer::BlockTerms(r) => {
                r.dictionary_fields().map_err(|e| err(e.into()))
            }
            MockRandomFieldsProducer::Ords(r) => r.dictionary_fields().map_err(|e| err(e.into())),
        }
    }
}

fn postings_files(postings: PostingsFiles) -> BTreeMap<String, Vec<u8>> {
    let mut files = BTreeMap::from([("doc".into(), postings.doc), ("psm".into(), postings.psm)]);
    if !postings.pos.is_empty() {
        files.insert("pos".into(), postings.pos);
    }
    if !postings.pay.is_empty() {
        files.insert("pay".into(), postings.pay);
    }
    files
}

impl fmt::Display for MockRandomPostingsFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PostingsFormat(name=MockRandom)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_codecs::field_infos::{FieldInfos, IndexOptions};
    use lucene_codecs::postings_writer::TermPostings;

    #[test]
    fn constructor_and_skip_threshold_preserve_java_random_stream() {
        let mut read_only = MockRandomPostingsFormat::default();
        assert_eq!(read_only.name(), "MockRandom");
        assert_eq!(read_only.to_string(), "PostingsFormat(name=MockRandom)");
        assert_eq!(
            read_only.next_seed(0).unwrap_err().to_string(),
            "Please use MockRandomPostingsFormat(Random)"
        );
        for max_doc in [0, 1_000_000, 1_000_001] {
            let mut caller = JavaRandom::new(42);
            let mut expected_caller = JavaRandom::new(42);
            let mut expected = JavaRandom::new(expected_caller.next_long());
            let mut format = MockRandomPostingsFormat::new(&mut caller);
            assert_eq!(caller.next_long(), expected_caller.next_long());
            for _ in 0..10 {
                random_int_between(&mut expected, if max_doc > 1_000_000 { 3 } else { 2 }, 10);
                assert_eq!(format.next_seed(max_doc).unwrap(), expected.next_long());
            }
        }
    }

    #[test]
    fn random_gap_keeps_its_stream_across_fields() {
        let mut selector = RandomGapSelector {
            random: JavaRandom::new(12),
            gap: 21,
        };
        let mut expected = JavaRandom::new(12);
        for field in [FieldInfo::new("a", 0), FieldInfo::new("b", 1)] {
            selector.new_field(&field);
            for _ in 0..100 {
                assert_eq!(
                    selector.is_index_term(
                        b"term",
                        TermStats {
                            doc_freq: 1,
                            total_term_freq: 1
                        }
                    ),
                    expected.next_int_bounded(21) == 10
                );
            }
        }
    }

    #[test]
    fn selected_writers_propagate_errors_and_keep_optional_postings_files() {
        let id = [7; 16];
        let infos = FieldInfos::new(vec![FieldInfo::new("body", 0)
            .with_index_options(IndexOptions::DocsAndFreqsAndPositionsAndOffsets)])
        .unwrap();
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
        // The fixture seeds select all seven writer variants on their first call.
        for seed in [0, 1, 2, 4, 8, 17, 121] {
            let mut format = MockRandomPostingsFormat::new(&mut JavaRandom::new(seed));
            let input = FieldPostingsInput {
                field_number: 0,
                index_options: IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
                doc_count: 1,
                has_payloads: false,
                terms: &terms,
            };
            let files = format.fields_consumer(&[input], &[], &state).unwrap();
            assert!(files.contains_key("pay"));
            assert!(files.contains_key("pos"));
        }
        terms[0].docs[0].1 = 0;
        for seed in [0, 1, 2, 4, 8, 17, 121] {
            let mut format = MockRandomPostingsFormat::new(&mut JavaRandom::new(seed));
            let input = FieldPostingsInput {
                field_number: 0,
                index_options: IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
                doc_count: 1,
                has_payloads: false,
                terms: &terms,
            };
            assert!(format.fields_consumer(&[input], &[], &state).is_err());
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
        for seed in [1, 2] {
            let input = FieldPostingsInput {
                field_number: 0,
                index_options: IndexOptions::DocsAndFreqs,
                doc_count: 1,
                has_payloads: false,
                terms: &terms,
            };
            let files = MockRandomPostingsFormat::new(&mut JavaRandom::new(seed))
                .fields_consumer(&[input], &[], &state)
                .unwrap();
            assert!(!files.contains_key("pos"));
            assert!(!files.contains_key("pay"));
        }
    }
}
