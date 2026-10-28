//! Port of `OrdsBlockTreeTermsReader` and `OrdsFieldReader`: opens a
//! group's `.tio`/`.tipo` and loads each field's FST terms index.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use lucene_store::codec_util::{self, FOOTER_LENGTH};
use lucene_store::data_input::{DataInput, SliceInput};
use lucene_util::automaton::{AutomatonType, CompiledAutomaton};

use super::outputs::{FstOrdsOutputs, Output};
use super::{
    corrupt, Error, OrdsIntersectTermsEnum, OrdsSegmentTermsEnum, OrdsTermState, Result,
    OUTPUT_FLAGS_NUM_BITS, TERMS_CODEC_NAME, TERMS_INDEX_CODEC_NAME, VERSION_CURRENT,
};
pub(super) use crate::blocktree::{
    POSTINGS_TERMS_CODEC, POSTINGS_VERSION_CURRENT, POSTINGS_VERSION_START,
};
use crate::field_infos::{FieldInfo, IndexOptions};
use crate::fst_util::TypedFst;
use crate::postings::{self, DocInput, Postings, PostingsFlags};
use crate::segment_state::SegmentReadState;

/// `OrdsBlockTreeTermsReader`: the fields of one group, by name.
#[derive(Debug)]
pub struct OrdsBlockTreeTermsReader {
    /// `in`: the whole `.tio`, shared with every field (`parent.in`).
    tio: Arc<[u8]>,
    fields: BTreeMap<String, OrdsFieldReader>,
}

/// `OrdsFieldReader`: one field's statistics and terms index.
pub struct OrdsFieldReader {
    pub(super) info: FieldInfo,
    /// `parent.in`.
    pub(super) tio: Arc<[u8]>,
    num_terms: i64,
    sum_total_term_freq: i64,
    sum_doc_freq: i64,
    doc_count: i32,
    index_start_fp: i64,
    pub(super) root_block_fp: i64,
    pub(super) root_code: Output,
    min_term: Vec<u8>,
    max_term: Vec<u8>,
    pub(super) index: TypedFst<FstOrdsOutputs>,
}

impl fmt::Debug for OrdsFieldReader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OrdsFieldReader")
            .field("field", &self.info.name)
            .field("num_terms", &self.num_terms)
            .field("root_block_fp", &self.root_block_fp)
            .finish()
    }
}

/// `readBytesRef(in)`.
fn read_bytes_ref(input: &mut SliceInput<'_>) -> Result<Vec<u8>> {
    let n = input.read_vint()?;
    let n = usize::try_from(n).map_err(|_| corrupt(format!("negative length {n}")))?;
    if n > input.len().saturating_sub(input.position()) {
        return Err(corrupt(format!("length {n} past the end")));
    }
    let mut bytes = vec![0u8; n];
    input.read_bytes(&mut bytes)?;
    Ok(bytes)
}

/// `seekDir(input)`: the directory offset in the long before the footer.
fn seek_dir(input: &mut SliceInput<'_>, len: usize) -> Result<()> {
    let at = len
        .checked_sub(FOOTER_LENGTH + 8)
        .ok_or_else(|| corrupt("file too short for its directory offset"))?;
    input.seek(at)?;
    let dir = input.read_i64()?;
    let dir = usize::try_from(dir)
        .ok()
        .filter(|&d| d <= at)
        .ok_or_else(|| corrupt(format!("invalid directory offset {dir}")))?;
    Ok(input.seek(dir)?)
}

impl OrdsBlockTreeTermsReader {
    /// `new OrdsBlockTreeTermsReader(postingsReader, state)`: both headers
    /// (their versions equal), the whole `.tipo`'s checksum, the postings
    /// writer's header, the `.tio`'s footer, then each field's summary and
    /// terms index.
    ///
    /// # Errors
    /// `Corrupted` for a bad header, footer or checksum, a field number
    /// `state.field_infos` lacks, a field summary Java refuses, a duplicate
    /// field; [`Error::Fst`] for a terms index that does not decode;
    /// [`Error::Unsupported`] for a postings block size other than
    /// `Lucene104PostingsFormat`'s.
    pub fn open(tio: Vec<u8>, tipo: &[u8], state: &SegmentReadState<'_>) -> Result<Self> {
        let tio: Arc<[u8]> = tio.into();
        let mut input = SliceInput::new(&tio);
        let version = codec_util::check_index_header(
            &mut input,
            TERMS_CODEC_NAME,
            VERSION_CURRENT,
            VERSION_CURRENT,
            state.segment_id,
            state.segment_suffix,
        )?
        .version;
        let mut index_in = SliceInput::new(tipo);
        let index_version = codec_util::check_index_header(
            &mut index_in,
            TERMS_INDEX_CODEC_NAME,
            VERSION_CURRENT,
            VERSION_CURRENT,
            state.segment_id,
            state.segment_suffix,
        )?
        .version;
        if index_version != version {
            return Err(corrupt(format!(
                "mixmatched version files: {version},{index_version}"
            )));
        }
        // `CodecUtil.checksumEntireFile(indexIn)`.
        let index_end = tipo
            .len()
            .checked_sub(FOOTER_LENGTH)
            .ok_or_else(|| corrupt("file too small to contain a codec footer"))?;
        codec_util::check_whole_file_footer(tipo, index_end)?;
        // `Lucene104PostingsReader.init(in, state)`.
        codec_util::check_index_header(
            &mut input,
            POSTINGS_TERMS_CODEC,
            POSTINGS_VERSION_START,
            POSTINGS_VERSION_CURRENT,
            state.segment_id,
            state.segment_suffix,
        )?;
        if input.read_vint()? != postings::BLOCK_SIZE {
            return Err(Error::Unsupported(
                "index-time BLOCK_SIZE differs from read-time BLOCK_SIZE",
            ));
        }
        codec_util::retrieve_checksum(&tio)?;
        seek_dir(&mut input, tio.len())?;
        seek_dir(&mut index_in, tipo.len())?;

        let num_fields = input.read_vint()?;
        if num_fields < 0 {
            return Err(corrupt(format!("invalid numFields: {num_fields}")));
        }
        let mut fields = BTreeMap::new();
        for _ in 0..num_fields {
            let field = input.read_vint()?;
            let num_terms = input.read_vlong()?;
            let code = read_bytes_ref(&mut input)?;
            let root_code = Output::new(code, 0, num_terms);
            let info = state
                .field_infos
                .field_by_number(field)
                .ok_or_else(|| corrupt(format!("invalid field number: {field}")))?;
            let sum_total_term_freq = input.read_vlong()?;
            // Without frequencies sumDocFreq == sumTotalTermFreq, one value.
            let sum_doc_freq = if info.index_options == IndexOptions::Docs {
                sum_total_term_freq
            } else {
                input.read_vlong()?
            };
            let doc_count = input.read_vint()?;
            let min_term = read_bytes_ref(&mut input)?;
            let max_term = read_bytes_ref(&mut input)?;
            if doc_count < 0 || doc_count > state.max_doc {
                return Err(corrupt(format!(
                    "invalid docCount: {doc_count} maxDoc: {}",
                    state.max_doc
                )));
            }
            if sum_doc_freq < i64::from(doc_count) {
                return Err(corrupt(format!(
                    "invalid sumDocFreq: {sum_doc_freq} docCount: {doc_count}"
                )));
            }
            if sum_total_term_freq < sum_doc_freq {
                return Err(corrupt(format!(
                    "invalid sumTotalTermFreq: {sum_total_term_freq} sumDocFreq: {sum_doc_freq}"
                )));
            }
            let index_start_fp = index_in.read_vlong()?;
            let root_block_fp = super::ushr64(
                SliceInput::new(&root_code.bytes).read_vlong()?,
                OUTPUT_FLAGS_NUM_BITS,
            );
            let start = usize::try_from(index_start_fp)
                .ok()
                .filter(|&s| s <= tipo.len())
                .ok_or_else(|| corrupt(format!("invalid indexStartFP {index_start_fp}")))?;
            let (index, _) = TypedFst::<FstOrdsOutputs>::read_prefix(&tipo[start..])?;
            if index.empty_output().is_none() {
                // Java asserts the first arc is final: the root prefix is empty.
                return Err(corrupt("terms index has no root output"));
            }
            let reader = OrdsFieldReader {
                info: info.clone(),
                tio: Arc::clone(&tio),
                num_terms,
                sum_total_term_freq,
                sum_doc_freq,
                doc_count,
                index_start_fp,
                root_block_fp,
                root_code,
                min_term,
                max_term,
                index,
            };
            if fields.insert(info.name.clone(), reader).is_some() {
                return Err(corrupt(format!("duplicate field: {}", info.name)));
            }
        }
        Ok(OrdsBlockTreeTermsReader { tio, fields })
    }

    /// `terms(field)`.
    pub fn terms(&self, field: &str) -> Option<&OrdsFieldReader> {
        self.fields.get(field)
    }

    /// `iterator()`: the field names, in order.
    pub fn field_names(&self) -> impl Iterator<Item = &str> {
        self.fields.keys().map(String::as_str)
    }

    /// `size()`.
    pub fn size(&self) -> usize {
        self.fields.len()
    }

    /// `checkIntegrity()`'s terms dictionary half: the whole `.tio`'s
    /// checksum (the postings' are the caller's, with the `.doc`).
    ///
    /// # Errors
    /// `Corrupted` when the checksum does not match.
    pub fn check_integrity(&self) -> Result<()> {
        let end = self
            .tio
            .len()
            .checked_sub(FOOTER_LENGTH)
            .ok_or_else(|| corrupt("file too small to contain a codec footer"))?;
        codec_util::check_whole_file_footer(&self.tio, end)?;
        Ok(())
    }
}

impl fmt::Display for OrdsBlockTreeTermsReader {
    /// `toString()`, the postings reader by its class name.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "OrdsBlockTreeTermsReader(fields={},delegate=Lucene104PostingsReader)",
            self.fields.len()
        )
    }
}

impl fmt::Display for OrdsFieldReader {
    /// `toString()`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "OrdsBlockTreeTerms(terms={},postings={},positions={},docs={})",
            self.num_terms, self.sum_doc_freq, self.sum_total_term_freq, self.doc_count
        )
    }
}

impl OrdsFieldReader {
    /// The field's `FieldInfo`.
    pub fn field_info(&self) -> &FieldInfo {
        &self.info
    }

    /// `size()`.
    pub fn size(&self) -> i64 {
        self.num_terms
    }

    /// `getSumTotalTermFreq()`.
    pub fn sum_total_term_freq(&self) -> i64 {
        self.sum_total_term_freq
    }

    /// `getSumDocFreq()`.
    pub fn sum_doc_freq(&self) -> i64 {
        self.sum_doc_freq
    }

    /// `getDocCount()`.
    pub fn doc_count(&self) -> i32 {
        self.doc_count
    }

    /// `getMin()`.
    pub fn min(&self) -> &[u8] {
        &self.min_term
    }

    /// `getMax()`.
    pub fn max(&self) -> &[u8] {
        &self.max_term
    }

    /// `hasFreqs()`.
    pub fn has_freqs(&self) -> bool {
        !matches!(
            self.info.index_options,
            IndexOptions::None | IndexOptions::Docs
        )
    }

    /// `hasPositions()`.
    pub fn has_positions(&self) -> bool {
        self.info.index_options.subsumes_positions()
    }

    /// `hasOffsets()`.
    pub fn has_offsets(&self) -> bool {
        self.info.index_options.subsumes_offsets()
    }

    /// `hasPayloads()`.
    pub fn has_payloads(&self) -> bool {
        self.info.store_payloads
    }

    /// `rootCode`: the root block's frame data and the field's ord range.
    pub fn root_code(&self) -> &Output {
        &self.root_code
    }

    /// `rootBlockFP`.
    pub fn root_block_fp(&self) -> i64 {
        self.root_block_fp
    }

    /// `indexStartFP`.
    pub fn index_start_fp(&self) -> i64 {
        self.index_start_fp
    }

    /// `iterator()`: an `OrdsSegmentTermsEnum`, unpositioned.
    pub fn iter(&self) -> OrdsSegmentTermsEnum<'_> {
        OrdsSegmentTermsEnum::new(self)
    }

    /// `intersect(compiled, startTerm)`: the terms `compiled` accepts, after
    /// `start_term` when given.
    ///
    /// # Errors
    /// [`Error::Unsupported`] unless `compiled` is `NORMAL`
    /// (`IllegalArgumentException`: use `CompiledAutomaton.getTermsEnum`);
    /// `Corrupted` for a malformed block or terms index, or a start term
    /// whose prefix `compiled` rejects (Java steps into a dead state).
    pub fn intersect<'a>(
        &'a self,
        compiled: &'a CompiledAutomaton,
        start_term: Option<&[u8]>,
    ) -> Result<OrdsIntersectTermsEnum<'a>> {
        let (AutomatonType::NORMAL, Some(runnable), Some(accessor)) = (
            compiled.automaton_type,
            compiled.byte_runnable(),
            compiled.get_transition_accessor(),
        ) else {
            return Err(Error::Unsupported(
                "please use CompiledAutomaton.getTermsEnum instead",
            ));
        };
        OrdsIntersectTermsEnum::new(
            self,
            runnable,
            accessor,
            compiled.common_suffix_ref.as_deref(),
            start_term,
        )
    }

    /// `postingsReader.postings(fieldInfo, state, reuse, flags)`: a term's
    /// documents and frequencies, read from the group's `.doc` (a `docFreq
    /// == 1` term's one document is in its metadata).
    ///
    /// # Errors
    /// What decoding the `.doc` fails with.
    pub fn postings(
        &self,
        state: &OrdsTermState,
        doc_in: &DocInput<'_>,
        flags: PostingsFlags,
    ) -> Result<Postings> {
        if state.doc_freq == 1 {
            return Ok(postings::singleton_postings(
                state.meta,
                state.total_term_freq,
            )?);
        }
        Ok(doc_in.read_postings_with_flags(
            state.meta,
            state.doc_freq,
            self.info.index_options,
            self.info.store_payloads,
            flags,
        )?)
    }
}
