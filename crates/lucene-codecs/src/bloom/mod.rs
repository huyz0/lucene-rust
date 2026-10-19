//! Port of lucene-codecs' `org.apache.lucene.codecs.bloom`:
//! `BloomFilteringPostingsFormat`, a postings format that wraps another one
//! and keeps, per field, a bloom filter ([`FuzzySet`]) of the field's terms,
//! so `seekExact` on a term the segment does not hold -- the common case of a
//! primary-key lookup across many segments -- is refused from memory without
//! touching the terms dictionary.
//!
//! # The files
//!
//! The delegate's files, unchanged, under the group's codec suffix
//! (`_0_BloomFilter_0.tim`, `.tip`, `.tmd`, `.doc`, ...), plus one more:
//!
//! ```text
//! .blm   IndexHeader("BloomFilter", 3)
//!        String(delegate postings format name)
//!        int(numBlooms)
//!        per bloom: int(fieldNumber), FuzzySet.serialize
//!        Footer
//! ```
//!
//! A field the factory gave no set, or whose set ended more than 90%
//! saturated (`DefaultBloomFilterFactory.isSaturated`), has no entry: its
//! terms are the delegate's, unfiltered. A set is written downsized to the
//! smallest usable size that keeps it at most half saturated.
//!
//! # What differs from Java
//!
//! - The delegate is the postings this port reads and writes: a
//!   `Lucene104PostingsFormat` with any block sizes when writing; any
//!   generation [`crate::postings::PostingsFormat`] names when reading
//!   (only `Lucene104` is exercised against Java).
//!   Another delegate name is [`Error::UnsupportedDelegate`] -- Java's
//!   `PostingsFormat.forName` would find it through SPI.
//! - Java writes the `.blm` entries in `HashMap<FieldInfo, FuzzySet>`
//!   order, and `FieldInfo` hashes by identity, so two runs of the same
//!   flush can order them differently; this port writes them in field-number
//!   order. Each entry is the same bytes either way.
//! - A [`BloomFilterFactory`] is asked with the segment's [`SegmentWriteState`]
//!   and the field's `FieldInfo`, as Java's is.

mod fuzzy_set;

pub use fuzzy_set::{
    get_estimated_number_unique_values_allowing_for_collisions, get_nearest_set_size,
    get_nearest_set_size_for_saturation, ContainsResult, FuzzySet, USABLE_BIT_SET_SIZES,
};

use std::collections::HashMap;
use std::fmt;

use lucene_store::codec_util::{self, FOOTER_LENGTH};
use lucene_store::data_input::{DataInput, SliceInput};
use lucene_store::data_output::DataOutput;

use crate::blocktree::{self, BlockTreeFields, FieldTerms, TermStats, TermsEnum};
use crate::field_infos::{FieldInfo, FieldInfos};
use crate::per_field_postings::{
    Lucene104PostingsFormat, PostingsFormatKind, PER_FIELD_FORMAT_KEY,
};
use crate::postings;
use crate::postings_writer::{self, FieldNorms, FieldPostingsInput, Output, TermSource};
use crate::segment_state::{SegmentReadState, SegmentWriteState};

/// `BloomFilteringPostingsFormat.BLOOM_CODEC_NAME`: the format's name, the
/// `.blm` codec name and the stem of its codec suffix.
pub const BLOOM_CODEC_NAME: &str = "BloomFilter";
/// `BloomFilteringPostingsFormat.VERSION_START`.
pub const VERSION_START: i32 = 3;
/// `BloomFilteringPostingsFormat.VERSION_CURRENT`.
pub const VERSION_CURRENT: i32 = VERSION_START;
/// `BloomFilteringPostingsFormat.BLOOM_EXTENSION`.
pub const BLOOM_EXTENSION: &str = "blm";

/// What reading or writing a bloom-filtered group can fail with.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The `.blm` itself: header, footer, checksum, a filter's sizes.
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
    /// The delegate's terms dictionary.
    #[error(transparent)]
    BlockTree(#[from] blocktree::Error),
    /// The delegate's writer.
    #[error(transparent)]
    Write(#[from] postings_writer::Error),
    /// A delegate this port has no postings format for.
    #[error("unsupported delegate postings format: {0}")]
    UnsupportedDelegate(String),
    /// `fieldsConsumer` on a format constructed without a delegate (Java's
    /// `UnsupportedOperationException`).
    #[error(
        "Error - org.apache.lucene.codecs.bloom.BloomFilteringPostingsFormat has been constructed without a choice of PostingsFormat"
    )]
    NoDelegate,
}

/// `Result` of this module.
pub type Result<T> = std::result::Result<T, Error>;

fn corrupt(msg: impl Into<String>) -> Error {
    Error::Store(lucene_store::Error::Corrupted(msg.into()))
}

/// `BloomFilterFactory`: chooses each field's filter at write time.
pub trait BloomFilterFactory: fmt::Debug + Send + Sync {
    /// `getSetForField(state, info)`: the field's empty filter, or `None`
    /// to leave the field unfiltered.
    ///
    /// # Errors
    /// What creating the set fails with.
    fn get_set_for_field(
        &self,
        state: &SegmentWriteState<'_>,
        info: &FieldInfo,
    ) -> lucene_store::Result<Option<FuzzySet>>;

    /// `downsize(fieldInfo, initialSet)`: the filter as written, `None` to
    /// keep `initial_set` as it is.
    fn downsize(&self, _info: &FieldInfo, initial_set: &FuzzySet) -> Option<FuzzySet> {
        initial_set.downsize(initial_set.target_max_saturation())
    }

    /// `isSaturated(bloomFilter, fieldInfo)`: a saturated filter is not
    /// written.
    fn is_saturated(&self, bloom_filter: &FuzzySet, info: &FieldInfo) -> bool;
}

/// `DefaultBloomFilterFactory`.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct DefaultBloomFilterFactory;

impl BloomFilterFactory for DefaultBloomFilterFactory {
    /// Every document assumed to hold a unique term, at a false-positive
    /// rate of about 10%: `FuzzySet.createOptimalSet(maxDoc, 0.1023f)`.
    fn get_set_for_field(
        &self,
        state: &SegmentWriteState<'_>,
        _info: &FieldInfo,
    ) -> lucene_store::Result<Option<FuzzySet>> {
        FuzzySet::create_optimal_set(state.max_doc, 0.1023).map(Some)
    }

    /// More than 90% of the bits set.
    fn is_saturated(&self, bloom_filter: &FuzzySet, _info: &FieldInfo) -> bool {
        bloom_filter.saturation() > 0.9
    }
}

/// `BloomFilteringPostingsFormat`.
#[derive(Debug)]
pub struct BloomFilteringPostingsFormat<F: BloomFilterFactory = DefaultBloomFilterFactory> {
    delegate: Option<Lucene104PostingsFormat>,
    factory: F,
}

impl Default for BloomFilteringPostingsFormat {
    /// `new BloomFilteringPostingsFormat()`: the reading-only instance SPI
    /// builds, with no delegate to write with.
    fn default() -> Self {
        BloomFilteringPostingsFormat {
            delegate: None,
            factory: DefaultBloomFilterFactory,
        }
    }
}

impl BloomFilteringPostingsFormat {
    /// `new BloomFilteringPostingsFormat(delegatePostingsFormat)`, with the
    /// default factory.
    pub fn new(delegate: Lucene104PostingsFormat) -> Self {
        Self::with_factory(delegate, DefaultBloomFilterFactory)
    }
}

impl<F: BloomFilterFactory> BloomFilteringPostingsFormat<F> {
    /// `new BloomFilteringPostingsFormat(delegatePostingsFormat,
    /// bloomFilterFactory)`.
    pub fn with_factory(delegate: Lucene104PostingsFormat, factory: F) -> Self {
        BloomFilteringPostingsFormat {
            delegate: Some(delegate),
            factory,
        }
    }

    /// `getName()`.
    pub fn name(&self) -> &'static str {
        BLOOM_CODEC_NAME
    }

    /// The format written to and read through.
    pub fn delegate(&self) -> Option<&Lucene104PostingsFormat> {
        self.delegate.as_ref()
    }

    /// `fieldsConsumer(state)` then `write(fields, norms)` and `close()`:
    /// the delegate's files for `inputs` (in the order given, which Java
    /// takes in field-name order) and the `.blm` beside them.
    ///
    /// Every term with a document goes into its field's filter
    /// (`BloomFilteredFieldsConsumer.write`); `inputs` holds live postings
    /// only, so that is every term.
    ///
    /// # Errors
    /// [`Error::NoDelegate`] for the reading-only instance,
    /// [`Error::UnsupportedDelegate`] for a completion delegate, a field
    /// missing from `state.field_infos`, and what the delegate's writer or
    /// the factory fail with.
    pub fn write<T: TermSource>(
        &self,
        inputs: &[FieldPostingsInput<'_, T>],
        norms: &[FieldNorms<'_>],
        state: &SegmentWriteState<'_>,
    ) -> Result<BloomOutput> {
        let delegate = self.delegate.ok_or(Error::NoDelegate)?;
        if delegate.kind != PostingsFormatKind::Lucene104 {
            return Err(Error::UnsupportedDelegate(delegate.name().to_string()));
        }
        // "Delegate must write first."
        let output = postings_writer::write_fields_with_options(
            inputs,
            norms,
            &state.postings_options(delegate.write_options()),
            state.segment_id,
            state.segment_suffix,
        )?;
        let mut blooms: Vec<(&FieldInfo, FuzzySet)> = Vec::new();
        for input in inputs {
            let info = state.field_info(input.field_number).ok_or_else(|| {
                corrupt(format!(
                    "field number {} not in FieldInfos",
                    input.field_number
                ))
            })?;
            let mut bloom_filter: Option<FuzzySet> = None;
            for term in input.terms {
                if bloom_filter.is_none() {
                    match self.factory.get_set_for_field(state, info)? {
                        Some(set) => bloom_filter = Some(set),
                        // Field not bloom'd.
                        None => break,
                    }
                }
                if let Some(set) = bloom_filter.as_mut() {
                    // "Make sure there's at least one doc for this term."
                    if !term.docs().is_empty() {
                        set.add_value(term.term());
                    }
                }
            }
            if let Some(set) = bloom_filter {
                blooms.push((info, set));
            }
        }
        Ok(BloomOutput {
            delegate: output,
            blm: self.write_blm(blooms, delegate.name(), state),
        })
    }

    /// `BloomFilteredFieldsConsumer.close()`: the `.blm` of the fields'
    /// accumulated filters, saturated ones dropped and the rest downsized.
    fn write_blm(
        &self,
        mut blooms: Vec<(&FieldInfo, FuzzySet)>,
        delegate_name: &str,
        state: &SegmentWriteState<'_>,
    ) -> Vec<u8> {
        blooms.retain(|(info, set)| !self.factory.is_saturated(set, info));
        blooms.sort_by_key(|(info, _)| info.number);
        let mut blm = Vec::new();
        codec_util::write_index_header(
            &mut blm,
            BLOOM_CODEC_NAME,
            VERSION_CURRENT,
            state.segment_id,
            state.segment_suffix,
        );
        blm.write_string(delegate_name);
        // At most one entry per field of a segment, far below `i32::MAX`.
        blm.write_i32(i32::try_from(blooms.len()).unwrap_or(i32::MAX));
        for (info, set) in &blooms {
            blm.write_i32(info.number);
            // `saveAppropriatelySizedBloomFilter`.
            match self.factory.downsize(info, set) {
                Some(right_sized) => right_sized.serialize(&mut blm),
                None => set.serialize(&mut blm),
            }
        }
        codec_util::write_footer(&mut blm);
        blm
    }
}

impl<F: BloomFilterFactory> fmt::Display for BloomFilteringPostingsFormat<F> {
    /// `toString()`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.delegate {
            Some(d) => write!(
                f,
                "BloomFilteringPostingsFormat(PostingsFormat(name={}))",
                d.name()
            ),
            None => write!(f, "BloomFilteringPostingsFormat(null)"),
        }
    }
}

/// What [`BloomFilteringPostingsFormat::write`] produces: the delegate's
/// files and the `.blm`.
#[derive(Debug, Clone)]
pub struct BloomOutput {
    /// The delegate's `.doc`/`.pos`/`.pay`/`.psm`/`.tim`/`.tip`/`.tmd`.
    pub delegate: Output,
    /// The `.blm`.
    pub blm: Vec<u8>,
}

/// `BloomFilteredFieldsProducer`: the delegate's fields, and a filter for
/// each field the `.blm` names.
#[derive(Debug)]
pub struct BloomFilteredFieldsProducer {
    delegate: BlockTreeFields,
    delegate_name: String,
    blooms_by_field_name: HashMap<String, FuzzySet>,
}

impl BloomFilteredFieldsProducer {
    /// `new BloomFilteredFieldsProducer(state)`: reads the `.blm`, opening
    /// the delegate it names over `tim`/`tip`/`tmd` (the group's files, under
    /// the same codec suffix).
    ///
    /// # Errors
    /// `Corrupted` for a bad `.blm` header, footer, checksum, filter or field
    /// number; [`Error::UnsupportedDelegate`] for a delegate this port cannot
    /// read; what the delegate's terms dictionary fails with.
    pub fn open(
        blm: &[u8],
        tim: &[u8],
        tip: &[u8],
        tmd: &[u8],
        state: &SegmentReadState<'_>,
    ) -> Result<Self> {
        let mut input = SliceInput::new(blm);
        codec_util::check_index_header(
            &mut input,
            BLOOM_CODEC_NAME,
            VERSION_START,
            VERSION_CURRENT,
            state.segment_id,
            state.segment_suffix,
        )?;
        let delegate_name = input.read_string()?;
        if postings::PostingsFormat::from_name(&delegate_name).is_none() {
            return Err(Error::UnsupportedDelegate(delegate_name));
        }
        // `PostingsFormat.forName(delegate).fieldsProducer(state)`: the
        // delegate decodes as the format the `.blm` names, not as the
        // `BloomFilter` its fields' attributes record.
        let delegate_infos = FieldInfos {
            fields: state
                .field_infos
                .fields
                .iter()
                .map(|f| {
                    let mut f = f.clone();
                    for (k, v) in &mut f.attributes {
                        if k == PER_FIELD_FORMAT_KEY && v == BLOOM_CODEC_NAME {
                            v.clone_from(&delegate_name);
                        }
                    }
                    f
                })
                .collect(),
        };
        let delegate = blocktree::open(
            tim,
            tip,
            tmd,
            &delegate_infos,
            state.segment_id,
            state.segment_suffix,
            state.max_doc,
        )?;
        let num_blooms = input.read_i32()?;
        let mut blooms_by_field_name = HashMap::new();
        for _ in 0..num_blooms {
            let field_num = input.read_i32()?;
            let bloom = FuzzySet::deserialize(&mut input)?;
            let info = state
                .field_infos
                .field_by_number(field_num)
                .ok_or_else(|| corrupt(format!("invalid field number: {field_num}")))?;
            blooms_by_field_name.insert(info.name.clone(), bloom);
        }
        codec_util::check_footer(&mut input, blm.len())?;
        Ok(BloomFilteredFieldsProducer {
            delegate,
            delegate_name,
            blooms_by_field_name,
        })
    }

    /// The delegate's name, as the `.blm` records it.
    pub fn delegate_name(&self) -> &str {
        &self.delegate_name
    }

    /// The delegate's fields.
    pub fn delegate(&self) -> &BlockTreeFields {
        &self.delegate
    }

    /// The filter of `field`, if the `.blm` has one.
    pub fn bloom(&self, field: &str) -> Option<&FuzzySet> {
        self.blooms_by_field_name.get(field)
    }

    /// `terms(field)`: the delegate's terms, filtered when the field has a
    /// filter.
    pub fn terms(&self, field: &str) -> Option<BloomFilteredTerms<'_>> {
        let delegate = self.delegate.field(field)?;
        Some(BloomFilteredTerms {
            delegate,
            filter: self.blooms_by_field_name.get(field),
        })
    }

    /// `iterator()`: the delegate's field names.
    pub fn field_names(&self) -> impl Iterator<Item = &str> {
        self.delegate.iter_fields().map(|(name, _)| name)
    }

    /// `size()`: the delegate's field count.
    pub fn size(&self) -> usize {
        self.delegate.iter_fields().count()
    }
}

impl fmt::Display for BloomFilteredFieldsProducer {
    /// `toString()`, the delegate by name.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BloomFilteredFieldsProducer(fields={},delegate={})",
            self.blooms_by_field_name.len(),
            self.delegate_name
        )
    }
}

/// `BloomFilteredTerms` (and, for a field without a filter, the delegate's
/// `Terms` as Java returns it): every statistic and enum is the delegate's;
/// only [`Self::try_seek_exact`] consults the filter first.
#[derive(Debug, Clone, Copy)]
pub struct BloomFilteredTerms<'a> {
    delegate: &'a FieldTerms,
    filter: Option<&'a FuzzySet>,
}

impl<'a> BloomFilteredTerms<'a> {
    /// The delegate's terms.
    pub fn delegate(&self) -> &'a FieldTerms {
        self.delegate
    }

    /// The field's filter, `None` for an unfiltered field.
    pub fn filter(&self) -> Option<&'a FuzzySet> {
        self.filter
    }

    /// `BloomFilteredTermsEnum.seekExact(text)` and its statistics: `None`
    /// without a dictionary read when the filter says the term is absent.
    ///
    /// # Errors
    /// What the delegate's dictionary fails with.
    pub fn try_seek_exact(&self, term: &[u8]) -> Result<Option<TermStats>> {
        if let Some(filter) = self.filter {
            if filter.contains(term) == ContainsResult::No {
                return Ok(None);
            }
        }
        Ok(self.delegate.try_seek_exact(term)?)
    }

    /// `iterator()`: `next`, `seekCeil` and the rest go to the delegate's
    /// enum unfiltered (`BloomFilteredTermsEnum` forwards them).
    pub fn iter(&self) -> TermsEnum<'a> {
        self.delegate.iter()
    }

    /// `size()`.
    pub fn size(&self) -> i64 {
        self.delegate.num_terms
    }

    /// `getSumTotalTermFreq()`.
    pub fn sum_total_term_freq(&self) -> i64 {
        self.delegate.sum_total_term_freq
    }

    /// `getSumDocFreq()`.
    pub fn sum_doc_freq(&self) -> i64 {
        self.delegate.sum_doc_freq
    }

    /// `getDocCount()`.
    pub fn doc_count(&self) -> i32 {
        self.delegate.doc_count
    }

    /// `getMin()`.
    pub fn min(&self) -> &'a [u8] {
        &self.delegate.min_term
    }

    /// `getMax()`.
    pub fn max(&self) -> &'a [u8] {
        &self.delegate.max_term
    }
}

/// The `.blm`'s own footer checksum, without opening it; returns the
/// length of its payload. Not Java's `checkIntegrity`, which checks the
/// delegate's files only (the `.blm`'s checksum is verified at open).
///
/// # Errors
/// `Corrupted` when the file is shorter than a footer or its checksum fails.
pub fn check_blm_footer(blm: &[u8]) -> Result<u64> {
    let end = blm
        .len()
        .checked_sub(FOOTER_LENGTH)
        .ok_or_else(|| corrupt("file too small to contain a codec footer"))?;
    Ok(codec_util::check_whole_file_footer(blm, end)?)
}

#[cfg(test)]
mod tests;
