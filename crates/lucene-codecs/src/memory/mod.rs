//! Port of lucene-codecs' `org.apache.lucene.codecs.memory`: postings
//! formats whose terms dictionary lives in memory.
//!
//! [`FstPostingsFormat`] (`FSTPostingsFormat`, "FST50") keeps the
//! `Lucene104PostingsFormat` postings and replaces the block tree with one
//! FST per field mapping every term to its statistics and postings
//! metadata ([`FstTermsWriter`], [`FstTermsReader`]):
//!
//! ```text
//! .tfp   IndexHeader("FSTTerms", 2)
//!        IndexHeader("Lucene104PostingsWriterTerms", 0), vint(BLOCK_SIZE)
//!        fields' FSTs, then at dirStart:
//!          vint(numFields)
//!          per field: vint(number), vlong(numTerms),
//!                     [vlong(sumTotalTermFreq), unless IndexOptions.DOCS]
//!                     vlong(sumDocFreq), vint(docCount),
//!                     FST<TermData> (metadata and body)
//!        long(dirStart)
//!        Footer
//! ```
//!
//! Each term's output is [`TermData`]: `encodeTerm(absolute=true)`'s
//! bytes, `docFreq` and `totalTermFreq` ([`FstTermOutputs`]). The postings
//! files are `Lucene104PostingsWriter`'s, under the same codec suffix.
//!
//! [`DirectPostingsFormat`] ("Direct") writes exactly `Lucene104`'s files
//! and, on open, loads every field into memory ([`DirectField`]): all term
//! bytes in one array, each term's postings as one `int[]` (`docFreq <=
//! lowFreqCutoff`) or one array per statistic, and skip pointers over runs
//! of `minSkipCount` or more terms sharing a prefix, which
//! [`DirectIntersectTermsEnum`] follows.
//!
//! # What differs from Java
//!
//! - Module-level, like `IDVersionPostingsFormat` and the bloom filter: the
//!   files go in and come out as bytes, this port's `IndexWriter` routes no
//!   field here and `DirectoryReader` does not open these files.
//! - Fields are written in the order they are given; Java's are in field
//!   name order, which is what a caller passing a flush's or a merge's
//!   fields in that order gets.
//! - `seekExact(BytesRef, TermState)` (the deferred seek) is not ported.
//! - `DirectPostingsFormat` under a merge (`IOContext.MERGE`, where Java
//!   hands back the `Lucene104` reader unloaded) is not modelled; the caller
//!   opens the files with [`crate::blocktree`] for that. `ramBytesUsed` is
//!   not ported, and the postings enums do not take a `reuse` instance.
//! - Where Java would index past an array (a dead automaton state from a
//!   start term the automaton does not accept, `seekExact(ord)` out of
//!   range), this port returns [`Error::Unsupported`] instead; and
//!   `DirectIntersectTermsEnum::next` treats a state whose transitions such
//!   a start term ran past as exhausted, where Java reads past them.
//! - `.psm` is neither read nor validated (the whole-file checksums of
//!   `.doc`/`.pos`/`.pay` stand in for its recorded lengths).
//! - `DirectTermsEnum::next` past the end keeps `ord()` at `size()`; Java's
//!   keeps counting.

mod direct;
mod fst_term_outputs;
mod fst_terms;
#[cfg(test)]
mod tests;

pub use direct::{
    DirectField, DirectFields, DirectFiles, DirectIntersectTermsEnum, DirectPostingsEnum,
    DirectPostingsFormat, DirectTermsEnum, HighFreqDocsEnum, HighFreqPostingsEnum, LowFreqDocsEnum,
    LowFreqDocsEnumNoPos, LowFreqDocsEnumNoTf, LowFreqPostingsEnum, DEFAULT_LOW_FREQ_CUTOFF,
    DEFAULT_MIN_SKIP_COUNT,
};
pub use fst_term_outputs::{FstTermOutputs, TermData};
pub use fst_terms::{
    FstFieldReader, FstIntersectTermsEnum, FstSegmentTermsEnum, FstTermsReader, FstTermsWriter,
    TERMS_CODEC_NAME, TERMS_EXTENSION, TERMS_VERSION_CURRENT, TERMS_VERSION_START,
};

pub use crate::blocktree::SeekStatus;
use crate::blocktree::{POSTINGS_TERMS_CODEC, POSTINGS_VERSION_CURRENT, POSTINGS_VERSION_START};
use crate::postings_writer::{
    self, FieldNorms, FieldPostingsInput, PostingsFiles, TermSource, WriteOptions,
};
use crate::segment_state::{SegmentReadState, SegmentWriteState};

/// What reading or writing these formats fails with.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Headers, footers, checksums and field summaries.
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
    /// A term dictionary FST that does not decode.
    #[error(transparent)]
    Fst(#[from] crate::fst_compiler::FstReadError),
    /// Building a field's FST.
    #[error(transparent)]
    Compile(#[from] crate::fst_compiler::CompileError),
    /// The postings writer's input checks.
    #[error(transparent)]
    Write(#[from] postings_writer::Error),
    /// The block tree under `DirectPostingsFormat`.
    #[error(transparent)]
    BlockTree(#[from] crate::blocktree::Error),
    /// Decoding a term's postings.
    #[error(transparent)]
    Postings(#[from] crate::postings::Error),
    /// Java's `IllegalStateException`/`UnsupportedOperationException`.
    #[error("{0}")]
    Unsupported(&'static str),
}

/// `Result` of this module.
pub type Result<T> = std::result::Result<T, Error>;

fn corrupt(msg: impl Into<String>) -> Error {
    Error::Store(lucene_store::Error::Corrupted(msg.into()))
}

/// `FSTPostingsFormat`: `Lucene104PostingsWriter` postings under an FST
/// terms dictionary.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct FstPostingsFormat;

/// What [`FstPostingsFormat::write`] produces.
#[derive(Debug, Clone, Default)]
pub struct FstOutput {
    /// `.doc`, `.psm`, `.pos`, `.pay`.
    pub postings: PostingsFiles,
    /// `.tfp`.
    pub tfp: Vec<u8>,
}

impl FstPostingsFormat {
    /// `getName()`.
    pub const NAME: &'static str = "FST50";

    /// `toString()`.
    pub fn name(&self) -> &'static str {
        Self::NAME
    }

    /// `fieldsConsumer(state).write(fields, norms)` and `close()`.
    ///
    /// # Errors
    /// What the postings writer rejects in `inputs`, and an FST that cannot
    /// be built.
    pub fn write<T: TermSource>(
        &self,
        inputs: &[FieldPostingsInput<'_, T>],
        norms: &[FieldNorms<'_>],
        state: &SegmentWriteState<'_>,
    ) -> Result<FstOutput> {
        let options = state.postings_options(WriteOptions::default());
        let mut terms = FstTermsWriter::new(state);
        let postings = postings_writer::write_postings(
            inputs,
            norms,
            options.has_prox,
            options.has_payloads_or_offsets,
            state.segment_id,
            state.segment_suffix,
            |field| terms.write_field(field),
        )?;
        Ok(FstOutput {
            postings,
            tfp: terms.finish(),
        })
    }

    /// `fieldsProducer(state)`: the `.tfp`'s fields.
    ///
    /// # Errors
    /// See [`FstTermsReader::open`].
    pub fn open(&self, tfp: &[u8], state: &SegmentReadState<'_>) -> Result<FstTermsReader> {
        FstTermsReader::open(tfp, state)
    }
}

impl FstTermsReader {
    /// Every field's terms with their statistics and postings metadata
    /// ([`crate::terms_dictionary::TermsDictionaryFormat::read`]).
    ///
    /// # Errors
    /// An FST or term metadata that does not decode.
    pub fn dictionary_fields(&self) -> Result<Vec<postings_writer::DictionaryField>> {
        let mut fields = Vec::with_capacity(self.size());
        for name in self.field_names() {
            let Some(terms) = self.terms(name) else {
                continue;
            };
            let info = terms.field_info();
            let mut field = postings_writer::DictionaryField {
                field_number: info.number,
                index_options: info.index_options,
                has_payloads: info.store_payloads,
                doc_count: terms.doc_count(),
                terms: Vec::new(),
            };
            let mut it = terms.iter();
            while let Some(bytes) = it.next()? {
                let bytes = bytes.to_vec();
                field.terms.push(postings_writer::DictionaryTerm {
                    bytes,
                    doc_freq: it.doc_freq(),
                    total_term_freq: it.total_term_freq(),
                    meta: it.term_metadata()?,
                });
            }
            fields.push(field);
        }
        Ok(fields)
    }
}

/// `new FSTPostingsFormat()`, the instance
/// [`crate::terms_dictionary::lookup`] reads `FST50` groups with.
pub static DEFAULT_FORMAT: FstPostingsFormat = FstPostingsFormat;

impl crate::terms_dictionary::TermsDictionaryFormat for FstPostingsFormat {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn extensions(&self) -> &'static [&'static str] {
        &["tfp"]
    }

    fn write(
        &self,
        inputs: &[crate::terms_dictionary::ErasedInput<'_, '_>],
        norms: &[FieldNorms<'_>],
        _options: &crate::postings_writer::WriteOptions,
        state: &SegmentWriteState<'_>,
    ) -> postings_writer::Result<(PostingsFiles, Vec<Vec<u8>>)> {
        let out = FstPostingsFormat::write(self, inputs, norms, state).map_err(|e| match e {
            Error::Write(e) => e,
            Error::Store(e) => postings_writer::Error::Store(e),
            e => postings_writer::Error::Dictionary(e.to_string()),
        })?;
        Ok((out.postings, vec![out.tfp]))
    }

    fn read(
        &self,
        files: &[&[u8]],
        state: &SegmentReadState<'_>,
    ) -> crate::blocktree::Result<Vec<postings_writer::DictionaryField>> {
        let err = |e| match e {
            Error::Store(e) => crate::blocktree::Error::Store(e),
            Error::BlockTree(e) => e,
            Error::Postings(e) => crate::blocktree::Error::Postings(e),
            e => crate::blocktree::Error::ForeignDictionary {
                format: Self::NAME,
                message: e.to_string(),
            },
        };
        let [tfp] = files else {
            return Err(crate::blocktree::Error::Unsupported(
                "an FST50 group needs its .tfp",
            ));
        };
        FstTermsReader::open(tfp, state)
            .and_then(|reader| reader.dictionary_fields())
            .map_err(err)
    }
}
