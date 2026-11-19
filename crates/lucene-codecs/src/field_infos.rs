//! Port of `org.apache.lucene.codecs.lucene94.Lucene94FieldInfosFormat` (`.fnm`
//! files) — read-only.
//!
//! `.fnm` maps field names to field numbers plus everything the rest of the
//! codec needs to interpret those numbers in other per-field files (postings,
//! doc values, points, vectors): this is why doc values / postings parsing
//! needs `FieldInfos` read first.
//!
//! Wire format (vint/string/map-of-strings per `lucene_store::data_input`;
//! `DocValuesGen` is a plain little-endian i64, everything else a byte or vint;
//! header/footer per `codec_util`):
//! ```text
//! IndexHeader(codec="Lucene94FieldInfos", version in [0, 2], id, suffix)
//! FieldsCount --> vint
//! per field:
//!   FieldName      --> String
//!   FieldNumber    --> vint (must be >= 0)
//!   FieldBits      --> u8 (0x1 term vectors, 0x2 omit norms, 0x4 payloads,
//!                      0x8 soft-deletes field, 0x10 parent field [version>=1],
//!                      0x20 has doc-values-skip-index [version>=2]; no other
//!                      bits may be set for the file's format version)
//!   IndexOptions   --> u8 (0..=5)
//!   DocValuesType  --> u8 (0..=5)
//!   DocValuesSkipIndexType --> u8 (0..=1), only present if version >= 2
//!   DocValuesGen   --> i64 (LE)
//!   Attributes     --> MapOfStrings
//!   PointDimensionCount --> vint; if nonzero, PointIndexDimensionCount (vint)
//!                      and PointNumBytes (vint) follow
//!   VectorDimension --> vint
//!   VectorEncoding  --> u8 (0..=1)
//!   VectorSimilarityFunction --> u8 (0..=3)
//! Footer
//! ```

use lucene_store::codec_util::{self, ID_LENGTH};
use lucene_store::data_input::{DataInput, SliceInput};
use lucene_store::data_output::DataOutput;

const CODEC_NAME: &str = "Lucene94FieldInfos";
const FORMAT_START: i32 = 0;
const FORMAT_PARENT_FIELD: i32 = 1;
const FORMAT_DOCVALUE_SKIPPER: i32 = 2;
const FORMAT_CURRENT: i32 = FORMAT_DOCVALUE_SKIPPER;

const STORE_TERMVECTOR: u8 = 0x1;
const OMIT_NORMS: u8 = 0x2;
const STORE_PAYLOADS: u8 = 0x4;
const SOFT_DELETES_FIELD: u8 = 0x8;
const PARENT_FIELD_FIELD: u8 = 0x10;
const DOCVALUES_SKIPPER: u8 = 0x20;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
    #[error("invalid field number for field: {0}, fieldNumber={1}")]
    InvalidFieldNumber(String, i32),
    #[error("unused bits are set \"{0:#010b}\"")]
    UnusedBitsSet(u8),
    #[error("parent field bit is set but shouldn't \"{0:#010b}\"")]
    ParentFieldBitSetButTooOld(u8),
    #[error("doc values skipper bit is set but shouldn't \"{0:#010b}\"")]
    DocValuesSkipperBitSetButTooOld(u8),
    #[error("invalid IndexOptions byte: {0}")]
    InvalidIndexOptions(u8),
    #[error("invalid docvalues byte: {0}")]
    InvalidDocValuesType(u8),
    #[error("invalid docvaluesskipindex byte: {0}")]
    InvalidDocValuesSkipIndexType(u8),
    #[error("invalid vector encoding: {0}")]
    InvalidVectorEncoding(u8),
    #[error("invalid distance function: {0}")]
    InvalidVectorSimilarityFunction(u8),
    /// Condenses `FieldInfo.checkConsistency`'s many `IllegalArgumentException`
    /// messages into one contextual variant (Rust idiom: one type, rich
    /// message) rather than one enum case per Java throw site.
    #[error("invalid fieldinfo for field '{0}': {1}")]
    Inconsistent(String, &'static str),
    /// Port of the `FieldInfos(FieldInfo[])` constructor's cross-field
    /// `IllegalArgumentException`s -- duplicate names/numbers and more than
    /// one soft-deletes or parent field. Java raises these while *reading*
    /// a `.fnm` (the format's `read` hands the decoded array straight to
    /// that constructor), so a `.fnm` that trips one of them is rejected by
    /// real Lucene and must be rejected here too.
    #[error("invalid fieldinfos: {0}")]
    InvalidFieldInfos(String),
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexOptions {
    None,
    Docs,
    DocsAndFreqs,
    DocsAndFreqsAndPositions,
    DocsAndFreqsAndPositionsAndOffsets,
    DocsAndCustomFreqs,
}

impl IndexOptions {
    fn from_byte(b: u8) -> Result<Self> {
        match b {
            0 => Ok(Self::None),
            1 => Ok(Self::Docs),
            2 => Ok(Self::DocsAndFreqs),
            3 => Ok(Self::DocsAndFreqsAndPositions),
            4 => Ok(Self::DocsAndFreqsAndPositionsAndOffsets),
            5 => Ok(Self::DocsAndCustomFreqs),
            other => Err(Error::InvalidIndexOptions(other)),
        }
    }

    /// Port of `IndexOptions.subsumes(DOCS_AND_FREQS_AND_POSITIONS)`: whether
    /// this option indexes positions (and therefore may store payloads).
    /// `DocsAndCustomFreqs` is special-cased in Java to subsume as if it were
    /// `DocsAndFreqs` — i.e. it does NOT subsume positions.
    pub fn subsumes_positions(self) -> bool {
        matches!(
            self,
            Self::DocsAndFreqsAndPositions | Self::DocsAndFreqsAndPositionsAndOffsets
        )
    }

    /// Port of `IndexOptions.subsumes(DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS)`:
    /// whether this option indexes character offsets.
    pub fn subsumes_offsets(self) -> bool {
        matches!(self, Self::DocsAndFreqsAndPositionsAndOffsets)
    }

    fn to_byte(self) -> u8 {
        match self {
            Self::None => 0,
            Self::Docs => 1,
            Self::DocsAndFreqs => 2,
            Self::DocsAndFreqsAndPositions => 3,
            Self::DocsAndFreqsAndPositionsAndOffsets => 4,
            Self::DocsAndCustomFreqs => 5,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocValuesType {
    None,
    Numeric,
    Binary,
    Sorted,
    SortedSet,
    SortedNumeric,
}

impl DocValuesType {
    fn from_byte(b: u8) -> Result<Self> {
        match b {
            0 => Ok(Self::None),
            1 => Ok(Self::Numeric),
            2 => Ok(Self::Binary),
            3 => Ok(Self::Sorted),
            4 => Ok(Self::SortedSet),
            5 => Ok(Self::SortedNumeric),
            other => Err(Error::InvalidDocValuesType(other)),
        }
    }

    fn to_byte(self) -> u8 {
        match self {
            Self::None => 0,
            Self::Numeric => 1,
            Self::Binary => 2,
            Self::Sorted => 3,
            Self::SortedSet => 4,
            Self::SortedNumeric => 5,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocValuesSkipIndexType {
    None,
    Range,
}

impl DocValuesSkipIndexType {
    fn from_byte(b: u8) -> Result<Self> {
        match b {
            0 => Ok(Self::None),
            1 => Ok(Self::Range),
            other => Err(Error::InvalidDocValuesSkipIndexType(other)),
        }
    }

    /// Port of `DocValuesSkipIndexType.isCompatibleWith`.
    fn is_compatible_with(self, dv_type: DocValuesType) -> bool {
        match self {
            Self::None => true,
            Self::Range => matches!(
                dv_type,
                DocValuesType::Numeric
                    | DocValuesType::SortedNumeric
                    | DocValuesType::Sorted
                    | DocValuesType::SortedSet
            ),
        }
    }

    fn to_byte(self) -> u8 {
        match self {
            Self::None => 0,
            Self::Range => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorEncoding {
    Byte,
    Float32,
}

impl VectorEncoding {
    fn from_byte(b: u8) -> Result<Self> {
        match b {
            0 => Ok(Self::Byte),
            1 => Ok(Self::Float32),
            other => Err(Error::InvalidVectorEncoding(other)),
        }
    }

    fn to_byte(self) -> u8 {
        match self {
            Self::Byte => 0,
            Self::Float32 => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorSimilarityFunction {
    Euclidean,
    DotProduct,
    Cosine,
    MaximumInnerProduct,
}

impl VectorSimilarityFunction {
    fn from_byte(b: u8) -> Result<Self> {
        match b {
            0 => Ok(Self::Euclidean),
            1 => Ok(Self::DotProduct),
            2 => Ok(Self::Cosine),
            3 => Ok(Self::MaximumInnerProduct),
            other => Err(Error::InvalidVectorSimilarityFunction(other)),
        }
    }

    fn to_byte(self) -> u8 {
        match self {
            Self::Euclidean => 0,
            Self::DotProduct => 1,
            Self::Cosine => 2,
            Self::MaximumInnerProduct => 3,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FieldInfo {
    pub name: String,
    pub number: i32,
    pub store_term_vectors: bool,
    pub omit_norms: bool,
    pub store_payloads: bool,
    pub soft_deletes_field: bool,
    pub parent_field: bool,
    pub index_options: IndexOptions,
    pub doc_values_type: DocValuesType,
    pub doc_values_skip_index_type: DocValuesSkipIndexType,
    pub doc_values_gen: i64,
    pub attributes: Vec<(String, String)>,
    pub point_dimension_count: i32,
    pub point_index_dimension_count: i32,
    pub point_num_bytes: i32,
    pub vector_dimension: i32,
    pub vector_encoding: VectorEncoding,
    pub vector_similarity_function: VectorSimilarityFunction,
}

impl FieldInfo {
    /// The starting point of Java's `FieldInfo` constructor: a field with a
    /// name and a number and **no options at all** -- not indexed, no doc
    /// values, no points, no vectors, `dvGen == -1`.
    ///
    /// This is the one `FieldInfo` shape that is trivially consistent, so it
    /// is the only sound seed for the chained `with_*` setters below. Java has
    /// no such staged constructor (it takes all eighteen parameters at once
    /// and validates in the constructor body); the staging exists because
    /// eighteen positional parameters is not a Rust API, and it is closed the
    /// same way Java's is -- by [`Self::checked`], which applies the
    /// constructor's coercion and then `checkConsistency()`.
    ///
    /// `vector_encoding`/`vector_similarity_function` have no "absent" state
    /// on the wire, so a non-vector field carries the same pair
    /// `IndexingChain`'s own non-vector `FieldInfo`s do: `FLOAT32` and
    /// `EUCLIDEAN`.
    pub fn new(name: impl Into<String>, number: i32) -> Self {
        FieldInfo {
            name: name.into(),
            number,
            store_term_vectors: false,
            omit_norms: false,
            store_payloads: false,
            soft_deletes_field: false,
            parent_field: false,
            index_options: IndexOptions::None,
            doc_values_type: DocValuesType::None,
            doc_values_skip_index_type: DocValuesSkipIndexType::None,
            doc_values_gen: -1,
            attributes: Vec::new(),
            point_dimension_count: 0,
            point_index_dimension_count: 0,
            point_num_bytes: 0,
            vector_dimension: 0,
            vector_encoding: VectorEncoding::Float32,
            vector_similarity_function: VectorSimilarityFunction::Euclidean,
        }
    }

    /// `IndexOptions` -- the `indexOptions` constructor parameter.
    pub fn with_index_options(mut self, index_options: IndexOptions) -> Self {
        self.index_options = index_options;
        self
    }

    /// `storeTermVector`.
    pub fn with_store_term_vectors(mut self, store_term_vectors: bool) -> Self {
        self.store_term_vectors = store_term_vectors;
        self
    }

    /// `omitNorms`.
    pub fn with_omit_norms(mut self, omit_norms: bool) -> Self {
        self.omit_norms = omit_norms;
        self
    }

    /// `storePayloads`.
    pub fn with_store_payloads(mut self, store_payloads: bool) -> Self {
        self.store_payloads = store_payloads;
        self
    }

    /// `docValues` + `docValuesSkipIndex` + `dvGen`, which Java validates
    /// against each other (`isCompatibleWith`, and "cannot have a docvalues
    /// update generation without having docvalues"), so they are set together.
    pub fn with_doc_values(
        mut self,
        doc_values_type: DocValuesType,
        skip_index: DocValuesSkipIndexType,
        doc_values_gen: i64,
    ) -> Self {
        self.doc_values_type = doc_values_type;
        self.doc_values_skip_index_type = skip_index;
        self.doc_values_gen = doc_values_gen;
        self
    }

    /// `attributes`.
    pub fn with_attributes(mut self, attributes: Vec<(String, String)>) -> Self {
        self.attributes = attributes;
        self
    }

    /// `pointDimensionCount` + `pointIndexDimensionCount` + `pointNumBytes`,
    /// the triple Java cross-checks (each is meaningless without the others).
    pub fn with_points(
        mut self,
        dimension_count: i32,
        index_dimension_count: i32,
        num_bytes: i32,
    ) -> Self {
        self.point_dimension_count = dimension_count;
        self.point_index_dimension_count = index_dimension_count;
        self.point_num_bytes = num_bytes;
        self
    }

    /// `vectorDimension` + `vectorEncoding` + `vectorSimilarityFunction`, set
    /// together for the same reason [`Self::with_points`] takes a triple.
    pub fn with_vectors(
        mut self,
        dimension: i32,
        encoding: VectorEncoding,
        similarity: VectorSimilarityFunction,
    ) -> Self {
        self.vector_dimension = dimension;
        self.vector_encoding = encoding;
        self.vector_similarity_function = similarity;
        self
    }

    /// `softDeletesField`.
    pub fn with_soft_deletes_field(mut self, soft_deletes_field: bool) -> Self {
        self.soft_deletes_field = soft_deletes_field;
        self
    }

    /// `isParentField`.
    pub fn with_parent_field(mut self, parent_field: bool) -> Self {
        self.parent_field = parent_field;
        self
    }

    /// **The constructor**: `FieldInfo`'s constructor body, in order --
    /// the non-indexed coercion ("for non-indexed fields, leave defaults":
    /// `storeTermVector`/`storePayloads`/`omitNorms` are forced to `false`
    /// when `indexOptions == NONE`), then `checkConsistency()`.
    ///
    /// This is what a `FieldInfo` struct literal skips. Java makes the
    /// inconsistent combinations *unrepresentable*, because the only way to
    /// obtain a `FieldInfo` is through a constructor that throws; a Rust
    /// public-field struct cannot do that, so this method is the door, and
    /// every place that accepts a caller-supplied `FieldInfo`
    /// ([`FieldInfos::new`], `IndexWriter::open`) puts every field through it.
    ///
    /// The coercion is not a convenience: it is the behaviour real Lucene's
    /// *reader* has (`Lucene94FieldInfosFormat.read` builds each `FieldInfo`
    /// through this same constructor), so a `.fnm` whose bits carry
    /// `storeTermVector` on a non-indexed field is a file Lucene opens with
    /// the flag silently cleared -- not a corrupt one.
    pub fn checked(mut self) -> Result<Self> {
        if self.index_options == IndexOptions::None {
            self.store_term_vectors = false;
            self.store_payloads = false;
            self.omit_norms = false;
        }
        self.check_consistency()?;
        Ok(self)
    }

    /// Port of `FieldInfo.checkConsistency` (the subset of invariants that
    /// don't require comparing against sibling fields, which is all Java
    /// checks here too — `verifySameSchema` is a separate, merge-time check
    /// out of scope for this read-only parser).
    ///
    /// Public, as Java's is: a caller that builds a [`FieldInfo`] by struct
    /// literal can ask the same question the constructor asks. Prefer
    /// [`Self::checked`], which also applies the constructor's coercion.
    pub fn check_consistency(&self) -> Result<()> {
        let err = |msg: &'static str| Err(Error::Inconsistent(self.name.clone(), msg));

        if self.index_options != IndexOptions::None {
            if !self.index_options.subsumes_positions() && self.store_payloads {
                return err("indexed field cannot have payloads without positions");
            }
        } else {
            if self.store_term_vectors {
                return err("non-indexed field cannot store term vectors");
            }
            if self.store_payloads {
                return err("non-indexed field cannot store payloads");
            }
            if self.omit_norms {
                return err("non-indexed field cannot omit norms");
            }
        }

        if !self
            .doc_values_skip_index_type
            .is_compatible_with(self.doc_values_type)
        {
            return err("incompatible docValuesSkipIndexType with doc values type");
        }
        if self.doc_values_gen != -1 && self.doc_values_type == DocValuesType::None {
            return err("cannot have a docvalues update generation without having docvalues");
        }

        if self.point_dimension_count < 0 {
            return err("pointDimensionCount must be >= 0");
        }
        if self.point_index_dimension_count < 0 {
            return err("pointIndexDimensionCount must be >= 0");
        }
        if self.point_num_bytes < 0 {
            return err("pointNumBytes must be >= 0");
        }
        if self.point_dimension_count != 0 && self.point_num_bytes == 0 {
            return err("pointNumBytes must be > 0 when pointDimensionCount != 0");
        }
        if self.point_index_dimension_count != 0 && self.point_dimension_count == 0 {
            return err("pointIndexDimensionCount must be 0 when pointDimensionCount=0");
        }
        if self.point_num_bytes != 0 && self.point_dimension_count == 0 {
            return err("pointDimensionCount must be > 0 when pointNumBytes != 0");
        }

        if self.vector_dimension < 0 {
            return err("vectorDimension must be >= 0");
        }

        if self.soft_deletes_field && self.parent_field {
            return err("field can't be used as soft-deletes field and parent document field");
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct FieldInfos {
    pub fields: Vec<FieldInfo>,
}

/// `FieldInfos.hasProx()` over `fields`: some field indexes positions.
pub fn has_prox(fields: &[FieldInfo]) -> bool {
    fields.iter().any(|f| f.index_options.subsumes_positions())
}

/// `FieldInfos.hasPayloads() || hasOffsets()` over `fields`.
pub fn has_payloads_or_offsets(fields: &[FieldInfo]) -> bool {
    fields.iter().any(|f| {
        f.index_options.subsumes_offsets()
            || (f.store_payloads && f.index_options.subsumes_positions())
    })
}

impl FieldInfos {
    /// Port of the `FieldInfos(FieldInfo[])` constructor: every field goes
    /// through [`FieldInfo::checked`] (Java's per-field constructor already
    /// ran by the time this one does, so the invariants hold field by field),
    /// then the cross-field checks in [`Self::check_consistency`].
    ///
    /// This is the door for a caller assembling a field list by hand -- the
    /// gap the sweep recorded as "a caller can still build combinations Java
    /// makes unrepresentable and find out at `parse` time or not at all".
    pub fn new(fields: Vec<FieldInfo>) -> Result<Self> {
        // The cranky field-info benchmark exposed iterator/error machinery
        // even for zero fields. No per-field or cross-field check can fail
        // here; drop any reserved input storage and return an empty value.
        if fields.is_empty() {
            return Ok(Self { fields: Vec::new() });
        }
        let fields = fields
            .into_iter()
            .map(FieldInfo::checked)
            .collect::<Result<Vec<_>>>()?;
        let infos = FieldInfos { fields };
        infos.check_consistency()?;
        Ok(infos)
    }

    pub fn field_by_number(&self, number: i32) -> Option<&FieldInfo> {
        self.fields.iter().find(|f| f.number == number)
    }

    /// Port of `FieldInfos.fieldInfo(String)`.
    pub fn field_by_name(&self, name: &str) -> Option<&FieldInfo> {
        self.fields.iter().find(|f| f.name == name)
    }

    /// Port of `FieldInfos.getSoftDeletesField()`: the name of the single
    /// field flagged as the soft-deletes field, if any.
    pub fn soft_deletes_field(&self) -> Option<&str> {
        self.fields
            .iter()
            .find(|f| f.soft_deletes_field)
            .map(|f| f.name.as_str())
    }

    /// Port of `FieldInfos.getParentField()` (Lucene 9.10+).
    pub fn parent_field(&self) -> Option<&str> {
        self.fields
            .iter()
            .find(|f| f.parent_field)
            .map(|f| f.name.as_str())
    }

    /// Port of the cross-field half of the `FieldInfos(FieldInfo[])`
    /// constructor: every check it makes that `FieldInfo.checkConsistency`
    /// (per-field) cannot. Java performs these *during* `.fnm` reading,
    /// since `Lucene94FieldInfosFormat.read` returns
    /// `new FieldInfos(infos)`; this port therefore runs them at the end of
    /// [`parse`] rather than leaving a `.fnm` real Lucene would reject
    /// silently accepted.
    ///
    /// Java's `hasVectors`/`hasNorms`/... aggregate flags and its
    /// `byNumber`/`byName` lookup arrays are computed in the same pass; they
    /// are omitted here because nothing in this port consumes them (the
    /// accessors above answer the same questions by scanning), which is a
    /// deliberate scope call, not an oversight.
    fn check_consistency(&self) -> Result<()> {
        let mut soft_deletes: Option<&str> = None;
        let mut parent: Option<&str> = None;
        for (i, f) in self.fields.iter().enumerate() {
            if f.number < 0 {
                return Err(Error::InvalidFieldInfos(format!(
                    "illegal field number: {} for field {}",
                    f.number, f.name
                )));
            }
            for previous in &self.fields[..i] {
                if previous.name == f.name {
                    return Err(Error::InvalidFieldInfos(format!(
                        "duplicate field names: {} and {} have: {}",
                        previous.number, f.number, f.name
                    )));
                }
                if previous.number == f.number {
                    return Err(Error::InvalidFieldInfos(format!(
                        "duplicate field numbers: {} and {} have: {}",
                        previous.name, f.name, f.number
                    )));
                }
            }
            if f.soft_deletes_field {
                if let Some(existing) = soft_deletes {
                    if existing != f.name {
                        return Err(Error::InvalidFieldInfos(format!(
                            "multiple soft-deletes fields [{}, {existing}]",
                            f.name
                        )));
                    }
                }
                soft_deletes = Some(&f.name);
            }
            if f.parent_field {
                if let Some(existing) = parent {
                    if existing != f.name {
                        return Err(Error::InvalidFieldInfos(format!(
                            "multiple parent fields [{}, {existing}]",
                            f.name
                        )));
                    }
                }
                parent = Some(&f.name);
            }
        }
        Ok(())
    }
}

/// Parses a whole `.fnm` file already read into memory.
///
/// Both field-infos generations Lucene 10.5.0 reads are accepted, told apart
/// by the codec name in the file's own header: `Lucene94FieldInfos`
/// (versions 0..=2, every codec from `Lucene94` on) and the retired
/// `Lucene90FieldInfos` (version 0, the `Lucene90`..`Lucene92` codecs --
/// `backward_codecs.lucene90.Lucene90FieldInfosFormat`), whose wire format
/// has no vector-encoding byte (every vector field is `FLOAT32`), no
/// doc-values-skip-index byte and no parent-field bit, and whose reader
/// validates none of the flag bits.
pub fn parse(buf: &[u8], segment_id: &[u8; ID_LENGTH], segment_suffix: &str) -> Result<FieldInfos> {
    let generation = match codec_util::peek_codec_name(buf).as_deref() {
        Some(LUCENE90_CODEC_NAME) => Generation::Lucene90,
        _ => Generation::Lucene94,
    };
    parse_generation(buf, segment_id, segment_suffix, generation)
}

/// `Lucene90FieldInfosFormat.CODEC_NAME`.
const LUCENE90_CODEC_NAME: &str = "Lucene90FieldInfos";

/// Which `FieldInfosFormat` wrote a `.fnm`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Generation {
    /// `backward_codecs.lucene90.Lucene90FieldInfosFormat`.
    Lucene90,
    /// `codecs.lucene94.Lucene94FieldInfosFormat`.
    Lucene94,
}

fn parse_generation(
    buf: &[u8],
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
    generation: Generation,
) -> Result<FieldInfos> {
    let mut input = SliceInput::new(buf);

    // `Lucene90FieldInfosFormat` has one version (0). Its fields are read as
    // a `Lucene94` format-0 file would be, minus the vector-encoding byte and
    // the bit validation, both gated on `lucene90` below.
    let lucene90 = generation == Generation::Lucene90;
    let (codec_name, format_current) = if lucene90 {
        (LUCENE90_CODEC_NAME, FORMAT_START)
    } else {
        (CODEC_NAME, FORMAT_CURRENT)
    };
    let header = codec_util::check_index_header(
        &mut input,
        codec_name,
        FORMAT_START,
        format_current,
        segment_id,
        segment_suffix,
    )?;
    let format = header.version;

    // Java's `new FieldInfo[size]` throws on a negative count; a huge one is
    // bounded by the bytes left (every record takes several), so a corrupt
    // count fails on a short read rather than in the allocator.
    let size = input.read_vint()?;
    let size = usize::try_from(size).map_err(|_| {
        lucene_store::Error::Corrupted(format!("negative field count {size} in .fnm"))
    })?;
    let mut fields = Vec::with_capacity(size.min(input.remaining()));

    for _ in 0..size {
        let name = input.read_string()?;
        let number = input.read_vint()?;
        if number < 0 {
            return Err(Error::InvalidFieldNumber(name, number));
        }

        let bits = input.read_byte()?;
        let store_term_vectors = bits & STORE_TERMVECTOR != 0;
        let omit_norms = bits & OMIT_NORMS != 0;
        let store_payloads = bits & STORE_PAYLOADS != 0;
        let soft_deletes_field = bits & SOFT_DELETES_FIELD != 0;
        let parent_field = format >= FORMAT_PARENT_FIELD && bits & PARENT_FIELD_FIELD != 0;

        // `Lucene90FieldInfosFormat.read` reads the four flags it knows and
        // ignores every other bit.
        if !lucene90 {
            if bits & 0xC0 != 0 {
                return Err(Error::UnusedBitsSet(bits));
            }
            if format < FORMAT_PARENT_FIELD && bits & 0xF0 != 0 {
                return Err(Error::ParentFieldBitSetButTooOld(bits));
            }
            if format < FORMAT_DOCVALUE_SKIPPER && bits & DOCVALUES_SKIPPER != 0 {
                return Err(Error::DocValuesSkipperBitSetButTooOld(bits));
            }
        }

        let index_options = IndexOptions::from_byte(input.read_byte()?)?;
        let doc_values_type = DocValuesType::from_byte(input.read_byte()?)?;
        let doc_values_skip_index_type = if format >= FORMAT_DOCVALUE_SKIPPER {
            DocValuesSkipIndexType::from_byte(input.read_byte()?)?
        } else {
            DocValuesSkipIndexType::None
        };
        let doc_values_gen = input.read_i64()?;
        let attributes = input.read_map_of_strings()?;

        let point_dimension_count = input.read_vint()?;
        let (point_index_dimension_count, point_num_bytes) = if point_dimension_count != 0 {
            (input.read_vint()?, input.read_vint()?)
        } else {
            (point_dimension_count, 0)
        };

        let vector_dimension = input.read_vint()?;
        // `Lucene90FieldInfosFormat` predates byte vectors: no encoding byte,
        // every vector field is `VectorEncoding.FLOAT32`.
        let vector_encoding = if lucene90 {
            VectorEncoding::Float32
        } else {
            VectorEncoding::from_byte(input.read_byte()?)?
        };
        let vector_similarity_function = VectorSimilarityFunction::from_byte(input.read_byte()?)?;

        let field = FieldInfo {
            name,
            number,
            store_term_vectors,
            omit_norms,
            store_payloads,
            soft_deletes_field,
            parent_field,
            index_options,
            doc_values_type,
            doc_values_skip_index_type,
            doc_values_gen,
            attributes,
            point_dimension_count,
            point_index_dimension_count,
            point_num_bytes,
            vector_dimension,
            vector_encoding,
            vector_similarity_function,
        };
        // `Lucene94FieldInfosFormat.read` builds each entry with
        // `new FieldInfo(...)` and only then calls `checkConsistency()`, so the
        // constructor's "for non-indexed fields, leave defaults" coercion has
        // already cleared `storeTermVector`/`storePayloads`/`omitNorms` before
        // any check looks at them. A `.fnm` carrying those bits on a
        // non-indexed field is therefore a file real Lucene **opens**, with the
        // bits dropped -- not a corrupt one, so the reader must accept it.
        fields.push(field.checked()?);
    }

    codec_util::check_footer(&mut input, buf.len())?;

    let infos = FieldInfos { fields };
    infos.check_consistency()?;
    Ok(infos)
}

/// Port of `Lucene94FieldInfosFormat.write`: the exact byte-level inverse of
/// [`parse`], always writing the current format version
/// (`FORMAT_DOCVALUE_SKIPPER`) -- this port never needs to emit an older-
/// version `.fnm` file, since it only ever writes fresh segments, never
/// upgrades in place. Fields are written in the order given by `fields`;
/// callers are responsible for field-number uniqueness and `check_consistency`
/// invariants (this function does not re-validate them, matching the parser's
/// stance that a hand-built writer is trusted -- the round-trip tests below
/// exercise this via [`parse`] itself, which does validate).
pub fn write(fields: &[FieldInfo], segment_id: &[u8; ID_LENGTH], segment_suffix: &str) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();

    codec_util::write_index_header(
        &mut out,
        CODEC_NAME,
        FORMAT_CURRENT,
        segment_id,
        segment_suffix,
    );

    out.write_vint(fields.len() as i32);
    for f in fields {
        out.write_string(&f.name);
        out.write_vint(f.number);

        // Java's `FieldInfo` constructor coerces all three of these to `false`
        // for a non-indexed field before anything can read them
        // (`FieldInfo.java:110-114`), which is why `checkConsistency`'s
        // "non-indexed field cannot store term vectors / store payloads /
        // omit norms" can never fire on a `FieldInfo` Java built. A Rust
        // `FieldInfo` is a plain struct with no constructor, so a caller can
        // hand `write` the combination Java makes unrepresentable -- and the
        // bits then land on the wire, where Java's reader coerces them away
        // again but this port's own `parse` rejects them outright
        // (`check_consistency`). The result would be an `IndexWriter` able to
        // write a `.fnm` it could not itself re-open. Coercing here rather
        // than erroring is what Java does, and it is the behaviour that keeps
        // `write` -> `parse` total.
        let indexed = f.index_options != IndexOptions::None;

        let mut bits = 0u8;
        if f.store_term_vectors && indexed {
            bits |= STORE_TERMVECTOR;
        }
        if f.omit_norms && indexed {
            bits |= OMIT_NORMS;
        }
        if f.store_payloads && indexed {
            bits |= STORE_PAYLOADS;
        }
        if f.soft_deletes_field {
            bits |= SOFT_DELETES_FIELD;
        }
        if f.parent_field {
            bits |= PARENT_FIELD_FIELD;
        }
        // No `DOCVALUES_SKIPPER` (0x20) here: `Lucene94FieldInfosFormat.write`
        // never sets it (javap, 10.5); the skip index has its own byte below.
        out.write_byte(bits);

        out.write_byte(f.index_options.to_byte());
        out.write_byte(f.doc_values_type.to_byte());
        out.write_byte(f.doc_values_skip_index_type.to_byte());
        out.write_i64(f.doc_values_gen);
        out.write_map_of_strings(&f.attributes);

        out.write_vint(f.point_dimension_count);
        if f.point_dimension_count != 0 {
            out.write_vint(f.point_index_dimension_count);
            out.write_vint(f.point_num_bytes);
        }

        out.write_vint(f.vector_dimension);
        out.write_byte(f.vector_encoding.to_byte());
        out.write_byte(f.vector_similarity_function.to_byte());
    }

    codec_util::write_footer(&mut out);
    out
}

#[cfg(test)]
mod tests;
