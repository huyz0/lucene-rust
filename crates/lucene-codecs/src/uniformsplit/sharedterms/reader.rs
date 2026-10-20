//! Port of `STUniformSplitTermsReader`, `STUniformSplitTerms` and
//! `UnionFieldMetadataBuilder`: every field of the group reads the one
//! dictionary and the shared blocks, keeping to its own lines.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use lucene_util::automaton::CompiledAutomaton;

use super::super::field_metadata::FieldMetadata;
use super::super::reader::{check_blocks_integrity, read_dictionary, OpenedFiles};
use super::super::{
    corrupt, BlockDecoder, Error, Result, UniformSplitTermState, UniformSplitTerms,
};
use super::block_line::SharedFieldInfos;
use super::block_reader::{StBlockReader, StIntersectBlockReader, StMergingBlockReader};
use super::NAME;
use crate::field_infos::FieldInfo;
use crate::postings::{DocInput, Postings, PostingsFlags};
use crate::segment_state::SegmentReadState;

/// `UnionFieldMetadataBuilder`: the metadata covering every field's blocks,
/// which the shared dictionary and [`StMergingBlockReader`] read with.
#[derive(Debug, Clone)]
pub struct UnionFieldMetadataBuilder {
    dictionary_start_fp: i64,
    min_start_block_fp: i64,
    max_end_block_fp: i64,
    max_last_term: Option<Vec<u8>>,
}

impl Default for UnionFieldMetadataBuilder {
    fn default() -> Self {
        UnionFieldMetadataBuilder {
            dictionary_start_fp: -1,
            min_start_block_fp: i64::MAX,
            max_end_block_fp: i64::MIN,
            max_last_term: None,
        }
    }
}

impl UnionFieldMetadataBuilder {
    /// `addFieldMetadata(fieldMetadata)`.
    ///
    /// # Errors
    /// `Corrupted` for a `dictionaryStartFP` that differs from the fields'
    /// before (Java asserts).
    pub fn add_field_metadata(&mut self, field: &FieldMetadata) -> Result<&mut Self> {
        if self.dictionary_start_fp != -1 && self.dictionary_start_fp != field.dictionary_start_fp {
            return Err(corrupt(format!(
                "field {} has dictionary start {} where the others have {}",
                field.field_info.name, field.dictionary_start_fp, self.dictionary_start_fp
            )));
        }
        self.dictionary_start_fp = field.dictionary_start_fp;
        self.min_start_block_fp = self.min_start_block_fp.min(field.first_block_start_fp);
        self.max_end_block_fp = self.max_end_block_fp.max(field.last_block_start_fp);
        if self
            .max_last_term
            .as_deref()
            .is_none_or(|max| max < field.last_term.as_slice())
        {
            self.max_last_term = Some(field.last_term.clone());
        }
        Ok(self)
    }

    /// `build()`: a metadata of no field (number `-1`, no statistics).
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] when no field was added (Java's
    /// `IllegalStateException`).
    pub fn build(&self) -> Result<FieldMetadata> {
        let Some(last_term) = self.max_last_term.clone() else {
            return Err(Error::IllegalArgument(
                "no field metadata was provided".into(),
            ));
        };
        let mut union = FieldMetadata::new(FieldInfo::new("", -1), 0);
        union.dictionary_start_fp = self.dictionary_start_fp;
        union.first_block_start_fp = self.min_start_block_fp;
        union.last_block_start_fp = self.max_end_block_fp;
        union.last_term = last_term;
        Ok(union)
    }
}

/// `STUniformSplitTermsReader`: the fields of one group, by name.
pub struct StUniformSplitTermsReader {
    ustb: Arc<[u8]>,
    version: i32,
    fields: BTreeMap<String, StUniformSplitTerms>,
}

impl fmt::Debug for StUniformSplitTermsReader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StUniformSplitTermsReader")
            .field("version", &self.version)
            .field("fields", &self.fields.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl StUniformSplitTermsReader {
    /// The constructor: `UniformSplitTermsReader`'s checks and fields'
    /// metadata under [`NAME`], then `fillFieldMap`: the union metadata and,
    /// at its `dictionaryStartFP`, the one dictionary every field reads.
    ///
    /// # Errors
    /// As [`super::super::UniformSplitTermsReader::open`]; `Corrupted` for
    /// fields whose `dictionaryStartFP` differ.
    pub fn open(
        stustb: Vec<u8>,
        stustd: &[u8],
        state: &SegmentReadState<'_>,
        decoder: Option<Arc<dyn BlockDecoder>>,
    ) -> Result<Self> {
        let opened = OpenedFiles::open(stustb, stustd, state, decoder.as_deref(), NAME)?;
        let mut fields = BTreeMap::new();
        if !opened.metadata.is_empty() {
            let mut builder = UnionFieldMetadataBuilder::default();
            for metadata in &opened.metadata {
                builder.add_field_metadata(metadata)?;
            }
            let union_metadata = builder.build()?;
            let dictionary = Arc::new(read_dictionary(
                stustd,
                opened.ustd_end,
                union_metadata.dictionary_start_fp,
                NAME,
                decoder.as_deref(),
            )?);
            let shared = Arc::new(SharedFieldInfos::new(state.field_infos));
            let terms = |metadata: FieldMetadata| UniformSplitTerms {
                ustb: Arc::clone(&opened.ustb),
                metadata,
                dictionary: Arc::clone(&dictionary),
                decoder: decoder.clone(),
                shared: Some(Arc::clone(&shared)),
            };
            let union = Arc::new(terms(union_metadata));
            for metadata in opened.metadata {
                fields.insert(
                    metadata.field_info.name.clone(),
                    StUniformSplitTerms {
                        terms: terms(metadata),
                        union: Arc::clone(&union),
                    },
                );
            }
        }
        Ok(StUniformSplitTermsReader {
            ustb: opened.ustb,
            version: opened.version,
            fields,
        })
    }

    /// `terms(field)`.
    pub fn terms(&self, field: &str) -> Option<&StUniformSplitTerms> {
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

    /// `checkIntegrity()`'s terms dictionary half: the whole `.stustb`'s
    /// checksum.
    ///
    /// # Errors
    /// `Corrupted` when the checksum does not match.
    pub fn check_integrity(&self) -> Result<()> {
        check_blocks_integrity(&self.ustb)
    }
}

/// `STUniformSplitTerms`: one field of the group.
pub struct StUniformSplitTerms {
    /// The field's metadata over the shared blocks and dictionary.
    terms: UniformSplitTerms,
    /// `unionFieldMetadata`, over the same.
    union: Arc<UniformSplitTerms>,
}

impl fmt::Debug for StUniformSplitTerms {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StUniformSplitTerms")
            .field("field", &self.terms.metadata.field_info.name)
            .finish()
    }
}

impl StUniformSplitTerms {
    /// The field's `FieldInfo`.
    pub fn field_info(&self) -> &FieldInfo {
        self.terms.field_info()
    }

    /// `fieldMetadata`.
    pub fn metadata(&self) -> &FieldMetadata {
        self.terms.metadata()
    }

    /// `unionFieldMetadata`.
    pub fn union_metadata(&self) -> &FieldMetadata {
        self.union.metadata()
    }

    /// `size()`.
    pub fn size(&self) -> i64 {
        self.terms.size()
    }

    /// `getSumTotalTermFreq()`.
    pub fn sum_total_term_freq(&self) -> i64 {
        self.terms.sum_total_term_freq()
    }

    /// `getSumDocFreq()`.
    pub fn sum_doc_freq(&self) -> i64 {
        self.terms.sum_doc_freq()
    }

    /// `getDocCount()`.
    pub fn doc_count(&self) -> i32 {
        self.terms.doc_count()
    }

    /// `getMin()`: the first term of a fresh [`StBlockReader`].
    ///
    /// # Errors
    /// `Corrupted` for a malformed block.
    pub fn min(&self) -> Result<Option<Vec<u8>>> {
        Ok(self.iter().next()?.map(<[u8]>::to_vec))
    }

    /// `getMax()`: the field's last term.
    pub fn max(&self) -> &[u8] {
        self.terms.max()
    }

    /// `hasFreqs()`.
    pub fn has_freqs(&self) -> bool {
        self.terms.has_freqs()
    }

    /// `hasOffsets()`.
    pub fn has_offsets(&self) -> bool {
        self.terms.has_offsets()
    }

    /// `hasPositions()`.
    pub fn has_positions(&self) -> bool {
        self.terms.has_positions()
    }

    /// `hasPayloads()`.
    pub fn has_payloads(&self) -> bool {
        self.terms.has_payloads()
    }

    /// `iterator()`: an [`StBlockReader`], unpositioned.
    pub fn iter(&self) -> StBlockReader<'_> {
        StBlockReader::new(&self.terms)
    }

    /// `intersect(compiled, startTerm)`.
    ///
    /// # Errors
    /// As [`UniformSplitTerms::intersect`].
    pub fn intersect<'a>(
        &'a self,
        compiled: &'a CompiledAutomaton,
        start_term: Option<&[u8]>,
    ) -> Result<StIntersectBlockReader<'a>> {
        Ok(StIntersectBlockReader::new(
            self.terms.intersect(compiled, start_term)?,
        ))
    }

    /// `createMergingBlockReader()`: every field's lines, over the union
    /// metadata.
    pub fn merging_block_reader(&self) -> StMergingBlockReader<'_> {
        StMergingBlockReader::new(&self.union)
    }

    /// `postingsReader.postings(fieldInfo, state, reuse, flags)`.
    ///
    /// # Errors
    /// As [`UniformSplitTerms::postings`].
    pub fn postings(
        &self,
        state: &UniformSplitTermState,
        doc_in: &DocInput<'_>,
        flags: PostingsFlags,
    ) -> Result<Postings> {
        self.terms.postings(state, doc_in, flags)
    }
}
