//! Port of `UniformSplitTermsReader` and `UniformSplitTerms`: opens a
//! group's `.ustb`/`.ustd`, reads the fields' metadata and loads each
//! field's dictionary.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use lucene_store::codec_util::{self, FOOTER_LENGTH};
use lucene_store::data_input::{DataInput, SliceInput};
use lucene_util::automaton::{AutomatonType, CompiledAutomaton};

use super::dictionary::FstDictionary;
use super::field_metadata::{has_freqs, FieldMetadata};
use super::sharedterms::SharedFieldInfos;
use super::{
    corrupt, length, BlockDecoder, BlockReader, Error, IntersectBlockReader, Result,
    UniformSplitTermState, NAME, VERSION_CURRENT, VERSION_ENCODABLE_FIELDS_METADATA, VERSION_START,
};
use crate::blocktree::{POSTINGS_TERMS_CODEC, POSTINGS_VERSION_CURRENT, POSTINGS_VERSION_START};
use crate::field_infos::FieldInfo;
use crate::postings::{self, DocInput, Postings, PostingsFlags};
use crate::segment_state::SegmentReadState;

/// `UniformSplitTermsReader`: the fields of one group, by name.
pub struct UniformSplitTermsReader {
    /// `blockInput`: the whole `.ustb`, shared with every field.
    ustb: Arc<[u8]>,
    /// `version`.
    version: i32,
    /// `fieldToTermsMap`, iterated as `sortedFieldNames`.
    fields: BTreeMap<String, UniformSplitTerms>,
}

impl fmt::Debug for UniformSplitTermsReader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UniformSplitTermsReader")
            .field("version", &self.version)
            .field("fields", &self.fields.keys().collect::<Vec<_>>())
            .finish()
    }
}

/// `UniformSplitTerms`: one field's metadata and dictionary.
pub struct UniformSplitTerms {
    /// `blockInput`.
    pub(crate) ustb: Arc<[u8]>,
    /// `fieldMetadata`.
    pub(crate) metadata: FieldMetadata,
    /// The dictionary `dictionaryBrowserSupplier` reads (`sharedterms`: one
    /// for every field).
    pub(crate) dictionary: Arc<FstDictionary>,
    /// `blockDecoder`.
    pub(crate) decoder: Option<Arc<dyn BlockDecoder>>,
    /// `sharedterms`' `fieldInfos`: set, each block line lists the fields
    /// that have its term, and the field's term state is read from that
    /// list (`STBlockLine.Serializer.readTermStateForField`).
    pub(crate) shared: Option<Arc<SharedFieldInfos>>,
}

impl fmt::Debug for UniformSplitTerms {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UniformSplitTerms")
            .field("field", &self.metadata.field_info.name)
            .field("num_terms", &self.metadata.num_terms)
            .finish()
    }
}

/// `seekFieldsMetadata(indexInput)`: the offset in the long before the
/// footer.
fn seek_fields_metadata(input: &mut SliceInput<'_>, len: usize) -> Result<()> {
    let at = len
        .checked_sub(FOOTER_LENGTH + 8)
        .ok_or_else(|| corrupt("file too short for its fields metadata offset"))?;
    input.seek(at)?;
    let start = input.read_i64()?;
    let start = usize::try_from(start)
        .ok()
        .filter(|&s| s <= at)
        .ok_or_else(|| corrupt(format!("invalid fields metadata offset {start}")))?;
    Ok(input.seek(start)?)
}

impl UniformSplitTermsReader {
    /// `new UniformSplitTermsReader(postingsReader, state, blockDecoder,
    /// dictionaryOnHeap)`: both headers (the `.ustd`'s version the
    /// `.ustb`'s), the whole `.ustd`'s checksum, the postings writer's
    /// header, the `.ustb`'s footer, then every field's metadata and
    /// dictionary.
    ///
    /// # Errors
    /// `Corrupted` for a bad header, footer or checksum, field metadata Java
    /// refuses, a dictionary start outside the `.ustd`; [`Error::Fst`] for a
    /// dictionary that does not decode; [`Error::Unsupported`] for a
    /// postings block size other than `Lucene104PostingsFormat`'s; what the
    /// decoder fails with.
    pub fn open(
        ustb: Vec<u8>,
        ustd: &[u8],
        state: &SegmentReadState<'_>,
        decoder: Option<Arc<dyn BlockDecoder>>,
    ) -> Result<Self> {
        let opened = OpenedFiles::open(ustb, ustd, state, decoder.as_deref(), NAME)?;
        // `fillFieldMap`: a later field of the same name replaces the
        // earlier one, as `HashMap.put` does.
        let mut fields = BTreeMap::new();
        for metadata in opened.metadata {
            let dictionary = read_dictionary(
                ustd,
                opened.ustd_end,
                metadata.dictionary_start_fp,
                &metadata.field_info.name,
                decoder.as_deref(),
            )?;
            fields.insert(
                metadata.field_info.name.clone(),
                UniformSplitTerms {
                    ustb: Arc::clone(&opened.ustb),
                    metadata,
                    dictionary: Arc::new(dictionary),
                    decoder: decoder.clone(),
                    shared: None,
                },
            );
        }
        Ok(UniformSplitTermsReader {
            ustb: opened.ustb,
            version: opened.version,
            fields,
        })
    }

    /// `terms(field)`.
    pub fn terms(&self, field: &str) -> Option<&UniformSplitTerms> {
        self.fields.get(field)
    }

    /// `iterator()`: `sortedFieldNames`.
    pub fn field_names(&self) -> impl Iterator<Item = &str> {
        self.fields.keys().map(String::as_str)
    }

    /// `size()`.
    pub fn size(&self) -> usize {
        self.fields.len()
    }

    /// `version`.
    pub fn version(&self) -> i32 {
        self.version
    }

    /// `checkIntegrity()`'s terms dictionary half: the whole `.ustb`'s
    /// checksum (the `.ustd`'s was checked at open; the postings' are the
    /// caller's).
    ///
    /// # Errors
    /// `Corrupted` when the checksum does not match.
    pub fn check_integrity(&self) -> Result<()> {
        check_blocks_integrity(&self.ustb)
    }
}

/// `checkIntegrity()`: the whole blocks file's checksum.
pub(crate) fn check_blocks_integrity(ustb: &[u8]) -> Result<()> {
    let end = ustb
        .len()
        .checked_sub(FOOTER_LENGTH)
        .ok_or_else(|| corrupt("file too small to contain a codec footer"))?;
    codec_util::check_whole_file_footer(ustb, end)?;
    Ok(())
}

/// `FSTDictionary.read` at a field's `dictionaryStartFP`, inside the
/// dictionary file's checksummed bytes.
pub(crate) fn read_dictionary(
    ustd: &[u8],
    ustd_end: usize,
    dictionary_start_fp: i64,
    field: &str,
    decoder: Option<&dyn BlockDecoder>,
) -> Result<FstDictionary> {
    let dictionary_start = usize::try_from(dictionary_start_fp)
        .ok()
        .filter(|&s| s <= ustd_end)
        .ok_or_else(|| {
            corrupt(format!(
                "invalid dictionary start {dictionary_start_fp} for field {field}"
            ))
        })?;
    FstDictionary::read(&ustd[dictionary_start..ustd_end], decoder)
}

/// What `UniformSplitTermsReader`'s constructor reads before `fillFieldMap`.
pub(crate) struct OpenedFiles {
    /// `blockInput`.
    pub(crate) ustb: Arc<[u8]>,
    /// `version`.
    pub(crate) version: i32,
    /// Where the dictionary file's footer starts.
    pub(crate) ustd_end: usize,
    /// `fieldMetadataCollection`.
    pub(crate) metadata: Vec<FieldMetadata>,
}

impl OpenedFiles {
    /// The constructor up to `fillFieldMap`, for the codec `codec_name`.
    pub(crate) fn open(
        ustb: Vec<u8>,
        ustd: &[u8],
        state: &SegmentReadState<'_>,
        decoder: Option<&dyn BlockDecoder>,
        codec_name: &str,
    ) -> Result<Self> {
        let ustb: Arc<[u8]> = ustb.into();
        let mut input = SliceInput::new(&ustb);
        let version = codec_util::check_index_header(
            &mut input,
            codec_name,
            VERSION_START,
            VERSION_CURRENT,
            state.segment_id,
            state.segment_suffix,
        )?
        .version;
        let mut dictionary_input = SliceInput::new(ustd);
        codec_util::check_index_header(
            &mut dictionary_input,
            codec_name,
            version,
            version,
            state.segment_id,
            state.segment_suffix,
        )?;
        // `CodecUtil.checksumEntireFile(dictionaryInput)`.
        let ustd_end = ustd
            .len()
            .checked_sub(FOOTER_LENGTH)
            .ok_or_else(|| corrupt("file too small to contain a codec footer"))?;
        codec_util::check_whole_file_footer(ustd, ustd_end)?;
        // `Lucene104PostingsReader.init(blockInput, state)`.
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
        codec_util::retrieve_checksum(&ustb)?;
        seek_fields_metadata(&mut input, ustb.len())?;
        // `readFieldsMetadata`.
        let num_fields = input.read_vint()?;
        if num_fields < 0 {
            return Err(corrupt(format!("Illegal number of fields= {num_fields}")));
        }
        let metadata = match decoder {
            Some(decoder) if version >= VERSION_ENCODABLE_FIELDS_METADATA => {
                let encoded_length = input.read_vlong()?;
                if encoded_length < 0 {
                    return Err(corrupt(format!("Illegal encoded length: {encoded_length}")));
                }
                let start = input.position();
                let mut pos = start;
                let encoded = super::read_slice(&ustb, &mut pos, length(encoded_length)?)?;
                let decoded = decoder.decode(encoded)?;
                read_unencoded_fields_metadata(num_fields, &decoded, 0, state)?
            }
            _ => read_unencoded_fields_metadata(num_fields, &ustb, input.position(), state)?,
        };
        Ok(OpenedFiles {
            ustb,
            version,
            ustd_end,
            metadata,
        })
    }
}

/// `readUnencodedFieldsMetadata(numFields, metadataInput, fieldInfos,
/// fieldMetadataReader, maxNumDocs)` from `buf` at `pos`.
fn read_unencoded_fields_metadata(
    num_fields: i32,
    buf: &[u8],
    mut pos: usize,
    state: &SegmentReadState<'_>,
) -> Result<Vec<FieldMetadata>> {
    // Each field's metadata takes at least one byte: no allocation past
    // the input for a corrupt count.
    let capacity = usize::try_from(num_fields)
        .unwrap_or(0)
        .min(buf.len().saturating_sub(pos));
    let mut fields = Vec::with_capacity(capacity);
    for _ in 0..num_fields {
        fields.push(FieldMetadata::read(
            buf,
            &mut pos,
            state.field_infos,
            state.max_doc,
        )?);
    }
    Ok(fields)
}

impl UniformSplitTerms {
    /// The field's `FieldInfo`.
    pub fn field_info(&self) -> &FieldInfo {
        &self.metadata.field_info
    }

    /// `fieldMetadata`.
    pub fn metadata(&self) -> &FieldMetadata {
        &self.metadata
    }

    /// The field's dictionary.
    pub fn dictionary(&self) -> &FstDictionary {
        &self.dictionary
    }

    /// `size()`.
    pub fn size(&self) -> i64 {
        self.metadata.num_terms
    }

    /// `getSumTotalTermFreq()`.
    pub fn sum_total_term_freq(&self) -> i64 {
        self.metadata.sum_total_term_freq
    }

    /// `getSumDocFreq()`.
    pub fn sum_doc_freq(&self) -> i64 {
        self.metadata.sum_doc_freq
    }

    /// `getDocCount()`.
    pub fn doc_count(&self) -> i32 {
        self.metadata.doc_count
    }

    /// `getMin()`: `Terms.getMin`'s default, the first term of a fresh
    /// iterator.
    ///
    /// # Errors
    /// `Corrupted` for a malformed first block.
    pub fn min(&self) -> Result<Option<Vec<u8>>> {
        Ok(self.iter().next()?.map(<[u8]>::to_vec))
    }

    /// `getMax()`: the field's last term.
    pub fn max(&self) -> &[u8] {
        &self.metadata.last_term
    }

    /// `hasFreqs()`.
    pub fn has_freqs(&self) -> bool {
        has_freqs(self.metadata.field_info.index_options)
    }

    /// `hasOffsets()`.
    pub fn has_offsets(&self) -> bool {
        self.metadata.field_info.index_options.subsumes_offsets()
    }

    /// `hasPositions()`.
    pub fn has_positions(&self) -> bool {
        self.metadata.field_info.index_options.subsumes_positions()
    }

    /// `hasPayloads()`.
    pub fn has_payloads(&self) -> bool {
        self.metadata.field_info.store_payloads
    }

    /// `iterator()`: a [`BlockReader`], unpositioned.
    pub fn iter(&self) -> BlockReader<'_> {
        BlockReader::new(self)
    }

    /// `intersect(compiled, startTerm)`: the terms `compiled` accepts, after
    /// `start_term` when given.
    ///
    /// # Errors
    /// [`Error::Unsupported`] unless `compiled` is `NORMAL`
    /// (`checkIntersectAutomatonType`'s `IllegalArgumentException`).
    pub fn intersect<'a>(
        &'a self,
        compiled: &'a CompiledAutomaton,
        start_term: Option<&[u8]>,
    ) -> Result<IntersectBlockReader<'a>> {
        if compiled.automaton_type != AutomatonType::NORMAL {
            return Err(Error::Unsupported(
                "please use CompiledAutomaton.getTermsEnum instead",
            ));
        }
        IntersectBlockReader::new(self, compiled, start_term)
    }

    /// `postingsReader.postings(fieldInfo, state, reuse, flags)`: a term's
    /// documents and frequencies, read from the group's `.doc` (a `docFreq
    /// == 1` term's one document is in its state).
    ///
    /// # Errors
    /// What decoding the `.doc` fails with.
    pub fn postings(
        &self,
        state: &UniformSplitTermState,
        doc_in: &DocInput<'_>,
        flags: PostingsFlags,
    ) -> Result<Postings> {
        field_postings(&self.metadata.field_info, state, doc_in, flags)
    }
}

/// `postingsReader.postings(fieldInfo, state, reuse, flags)`.
pub(crate) fn field_postings(
    field_info: &FieldInfo,
    state: &UniformSplitTermState,
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
        field_info.index_options,
        field_info.store_payloads,
        flags,
    )?)
}
