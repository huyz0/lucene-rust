//! Port of `org.apache.lucene.codecs.lucene94.Lucene94FieldInfosFormat` (`.fnm`
//! files) — read-only.
//!
//! `.fnm` maps field names to field numbers plus everything the rest of the
//! codec needs to interpret those numbers in other per-field files (postings,
//! doc values, points, vectors): this is why doc values / postings parsing
//! needs `FieldInfos` read first (see PLAN.md Phase 2).
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
pub fn parse(buf: &[u8], segment_id: &[u8; ID_LENGTH], segment_suffix: &str) -> Result<FieldInfos> {
    let mut input = SliceInput::new(buf);

    let header = codec_util::check_index_header(
        &mut input,
        CODEC_NAME,
        FORMAT_START,
        FORMAT_CURRENT,
        segment_id,
        segment_suffix,
    )?;
    let format = header.version;

    let size = input.read_vint()? as usize;
    let mut fields = Vec::with_capacity(size);

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

        if bits & 0xC0 != 0 {
            return Err(Error::UnusedBitsSet(bits));
        }
        if format < FORMAT_PARENT_FIELD && bits & 0xF0 != 0 {
            return Err(Error::ParentFieldBitSetButTooOld(bits));
        }
        if format < FORMAT_DOCVALUE_SKIPPER && bits & DOCVALUES_SKIPPER != 0 {
            return Err(Error::DocValuesSkipperBitSetButTooOld(bits));
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
        let vector_encoding = VectorEncoding::from_byte(input.read_byte()?)?;
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
        // bits dropped -- not a corrupt one. This port used to reject it, which
        // is how a writer-produced segment became unreadable to its own reader
        // (c23) while real Lucene read it fine.
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
        // (`check_consistency`). The result was an `IndexWriter` able to write
        // a `.fnm` it could not itself re-open, found by c23 running
        // `check_index` over a writer-produced segment. Coercing here rather
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
        if f.doc_values_skip_index_type != DocValuesSkipIndexType::None {
            bits |= DOCVALUES_SKIPPER;
        }
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
mod tests {
    use super::*;

    /// Test-only `.fnm` byte builder, independent of the Java fixtures under
    /// `tests/field_infos_fixtures.rs` (which exercise a real IndexWriter's
    /// output): this covers the parser's own error/consistency handling with
    /// deliberately-invalid field combinations no real Lucene writer would
    /// ever produce.
    struct FieldBuilder {
        name: String,
        number: i32,
        bits: u8,
        index_options: u8,
        doc_values_type: u8,
        doc_values_skip_index_type: Option<u8>, // None => omit (format < 2)
        doc_values_gen: i64,
        point_dimension_count: i32,
        point_index_dimension_count: i32,
        point_num_bytes: i32,
        vector_dimension: i32,
        vector_encoding: u8,
        vector_similarity_function: u8,
    }

    impl FieldBuilder {
        fn valid(name: &str, number: i32) -> Self {
            Self {
                name: name.to_string(),
                number,
                bits: 0,
                index_options: 1, // Docs
                doc_values_type: 0,
                doc_values_skip_index_type: Some(0),
                doc_values_gen: -1,
                point_dimension_count: 0,
                point_index_dimension_count: 0,
                point_num_bytes: 0,
                vector_dimension: 0,
                vector_encoding: 0,
                vector_similarity_function: 0,
            }
        }

        fn build(&self, out: &mut Vec<u8>) {
            write_string(out, &self.name);
            write_vint(out, self.number);
            out.push(self.bits);
            out.push(self.index_options);
            out.push(self.doc_values_type);
            if let Some(skip) = self.doc_values_skip_index_type {
                out.push(skip);
            }
            out.extend_from_slice(&self.doc_values_gen.to_le_bytes());
            write_vint(out, 0); // attributes: empty map
            write_vint(out, self.point_dimension_count);
            if self.point_dimension_count != 0 {
                write_vint(out, self.point_index_dimension_count);
                write_vint(out, self.point_num_bytes);
            }
            write_vint(out, self.vector_dimension);
            out.push(self.vector_encoding);
            out.push(self.vector_similarity_function);
        }
    }

    struct FnmBuilder {
        id: [u8; ID_LENGTH],
        suffix: String,
        format_version: i32,
        fields: Vec<FieldBuilder>,
    }

    impl FnmBuilder {
        fn valid() -> Self {
            Self {
                id: [4u8; ID_LENGTH],
                suffix: String::new(),
                format_version: FORMAT_CURRENT,
                fields: vec![],
            }
        }

        fn build(&self) -> Vec<u8> {
            let mut out = Vec::new();
            out.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
            write_string(&mut out, CODEC_NAME);
            out.extend_from_slice(&(self.format_version as u32).to_be_bytes());
            out.extend_from_slice(&self.id);
            out.push(self.suffix.len() as u8);
            out.extend_from_slice(self.suffix.as_bytes());

            write_vint(&mut out, self.fields.len() as i32);
            for f in &self.fields {
                f.build(&mut out);
            }

            out.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
            out.extend_from_slice(&0u32.to_be_bytes());
            let checksum = crc32fast::hash(&out) as u64;
            out.extend_from_slice(&checksum.to_be_bytes());
            out
        }
    }

    fn write_vint(out: &mut Vec<u8>, mut v: i32) {
        loop {
            let mut b = (v & 0x7f) as u8;
            v = ((v as u32) >> 7) as i32;
            if v != 0 {
                b |= 0x80;
                out.push(b);
            } else {
                out.push(b);
                break;
            }
        }
    }

    fn write_string(out: &mut Vec<u8>, s: &str) {
        write_vint(out, s.len() as i32);
        out.extend_from_slice(s.as_bytes());
    }

    #[test]
    fn empty_field_infos_parses() {
        let b = FnmBuilder::valid();
        let fis = parse(&b.build(), &b.id, &b.suffix).unwrap();
        assert_eq!(fis.fields.len(), 0);
    }

    #[test]
    fn single_plain_field_parses() {
        let mut b = FnmBuilder::valid();
        b.fields.push(FieldBuilder::valid("id", 0));
        let fis = parse(&b.build(), &b.id, &b.suffix).unwrap();
        assert_eq!(fis.fields.len(), 1);
        assert_eq!(fis.fields[0].name, "id");
        assert_eq!(fis.fields[0].index_options, IndexOptions::Docs);
    }

    /// Port of `FieldInfos`' constructor checks -- Java reaches these from
    /// `Lucene94FieldInfosFormat.read` itself (`return new
    /// FieldInfos(infos)`), so a `.fnm` tripping one of them is rejected by
    /// real Lucene, not merely frowned upon.
    #[test]
    fn duplicate_field_names_rejected() {
        let mut b = FnmBuilder::valid();
        b.fields.push(FieldBuilder::valid("dup", 0));
        b.fields.push(FieldBuilder::valid("dup", 1));
        assert!(matches!(
            parse(&b.build(), &b.id, &b.suffix),
            Err(Error::InvalidFieldInfos(msg)) if msg.contains("duplicate field names")
        ));
    }

    #[test]
    fn duplicate_field_numbers_rejected() {
        let mut b = FnmBuilder::valid();
        b.fields.push(FieldBuilder::valid("a", 4));
        b.fields.push(FieldBuilder::valid("b", 4));
        assert!(matches!(
            parse(&b.build(), &b.id, &b.suffix),
            Err(Error::InvalidFieldInfos(msg)) if msg.contains("duplicate field numbers")
        ));
    }

    #[test]
    fn multiple_soft_deletes_fields_rejected() {
        let mut b = FnmBuilder::valid();
        let mut a = FieldBuilder::valid("a", 0);
        a.bits = SOFT_DELETES_FIELD;
        let mut c = FieldBuilder::valid("c", 1);
        c.bits = SOFT_DELETES_FIELD;
        b.fields.push(a);
        b.fields.push(c);
        assert!(matches!(
            parse(&b.build(), &b.id, &b.suffix),
            Err(Error::InvalidFieldInfos(msg)) if msg.contains("multiple soft-deletes fields")
        ));
    }

    #[test]
    fn multiple_parent_fields_rejected() {
        let mut b = FnmBuilder::valid();
        let mut a = FieldBuilder::valid("a", 0);
        a.bits = PARENT_FIELD_FIELD;
        let mut c = FieldBuilder::valid("c", 1);
        c.bits = PARENT_FIELD_FIELD;
        b.fields.push(a);
        b.fields.push(c);
        assert!(matches!(
            parse(&b.build(), &b.id, &b.suffix),
            Err(Error::InvalidFieldInfos(msg)) if msg.contains("multiple parent fields")
        ));
    }

    /// One soft-deletes field and one parent field on *different* fields is
    /// legal (only the same field being both is not -- see
    /// `FieldInfo.checkConsistency`); the accessors report each one.
    #[test]
    fn distinct_soft_deletes_and_parent_fields_accepted() {
        let mut b = FnmBuilder::valid();
        let mut a = FieldBuilder::valid("__soft_deletes", 0);
        a.bits = SOFT_DELETES_FIELD;
        let mut c = FieldBuilder::valid("_parent", 1);
        c.bits = PARENT_FIELD_FIELD;
        b.fields.push(a);
        b.fields.push(c);
        let fis = parse(&b.build(), &b.id, &b.suffix).unwrap();
        assert_eq!(fis.soft_deletes_field(), Some("__soft_deletes"));
        assert_eq!(fis.parent_field(), Some("_parent"));
        assert_eq!(fis.field_by_name("_parent").unwrap().number, 1);
        assert!(fis.field_by_name("nope").is_none());
    }

    #[test]
    fn no_soft_deletes_or_parent_field_reports_none() {
        let mut b = FnmBuilder::valid();
        b.fields.push(FieldBuilder::valid("id", 0));
        let fis = parse(&b.build(), &b.id, &b.suffix).unwrap();
        assert_eq!(fis.soft_deletes_field(), None);
        assert_eq!(fis.parent_field(), None);
    }

    #[test]
    fn negative_field_number_rejected() {
        let mut b = FnmBuilder::valid();
        b.fields.push(FieldBuilder::valid("bad", -1));
        assert!(matches!(
            parse(&b.build(), &b.id, &b.suffix),
            Err(Error::InvalidFieldNumber(name, -1)) if name == "bad"
        ));
    }

    #[test]
    fn unused_bits_rejected() {
        let mut b = FnmBuilder::valid();
        let mut f = FieldBuilder::valid("f", 0);
        f.bits = 0x80; // top bit, always unused
        b.fields.push(f);
        assert!(matches!(
            parse(&b.build(), &b.id, &b.suffix),
            Err(Error::UnusedBitsSet(0x80))
        ));
    }

    #[test]
    fn parent_field_bit_rejected_when_format_too_old() {
        let mut b = FnmBuilder::valid();
        b.format_version = FORMAT_START;
        let mut f = FieldBuilder::valid("f", 0);
        f.bits = PARENT_FIELD_FIELD;
        b.fields.push(f);
        assert!(matches!(
            parse(&b.build(), &b.id, &b.suffix),
            Err(Error::ParentFieldBitSetButTooOld(_))
        ));
    }

    #[test]
    fn parent_field_bit_accepted_at_current_format() {
        let mut b = FnmBuilder::valid();
        let mut f = FieldBuilder::valid("f", 0);
        f.bits = PARENT_FIELD_FIELD;
        b.fields.push(f);
        let fis = parse(&b.build(), &b.id, &b.suffix).unwrap();
        assert!(fis.fields[0].parent_field);
    }

    #[test]
    fn doc_values_skipper_bit_rejected_when_format_too_old() {
        let mut b = FnmBuilder::valid();
        b.format_version = FORMAT_PARENT_FIELD; // < FORMAT_DOCVALUE_SKIPPER
        let mut f = FieldBuilder::valid("f", 0);
        f.bits = DOCVALUES_SKIPPER;
        f.doc_values_skip_index_type = None; // omitted at this format version
        b.fields.push(f);
        assert!(matches!(
            parse(&b.build(), &b.id, &b.suffix),
            Err(Error::DocValuesSkipperBitSetButTooOld(_))
        ));
    }

    #[test]
    fn doc_values_skip_index_range_accepted_with_compatible_type() {
        let mut b = FnmBuilder::valid();
        let mut f = FieldBuilder::valid("f", 0);
        f.doc_values_type = 1; // Numeric
        f.doc_values_skip_index_type = Some(1); // Range
        b.fields.push(f);
        let fis = parse(&b.build(), &b.id, &b.suffix).unwrap();
        assert_eq!(
            fis.fields[0].doc_values_skip_index_type,
            DocValuesSkipIndexType::Range
        );
    }

    #[test]
    fn doc_values_skip_index_range_incompatible_with_none_type_rejected() {
        let mut b = FnmBuilder::valid();
        let mut f = FieldBuilder::valid("f", 0);
        f.doc_values_type = 0; // None
        f.doc_values_skip_index_type = Some(1); // Range: incompatible with None
        b.fields.push(f);
        assert!(matches!(
            parse(&b.build(), &b.id, &b.suffix),
            Err(Error::Inconsistent(_, _))
        ));
    }

    #[test]
    fn invalid_index_options_byte_rejected() {
        let mut b = FnmBuilder::valid();
        let mut f = FieldBuilder::valid("f", 0);
        f.index_options = 6; // out of range
        b.fields.push(f);
        assert!(matches!(
            parse(&b.build(), &b.id, &b.suffix),
            Err(Error::InvalidIndexOptions(6))
        ));
    }

    #[test]
    fn invalid_doc_values_type_byte_rejected() {
        let mut b = FnmBuilder::valid();
        let mut f = FieldBuilder::valid("f", 0);
        f.doc_values_type = 6;
        b.fields.push(f);
        assert!(matches!(
            parse(&b.build(), &b.id, &b.suffix),
            Err(Error::InvalidDocValuesType(6))
        ));
    }

    #[test]
    fn invalid_doc_values_skip_index_byte_rejected() {
        let mut b = FnmBuilder::valid();
        let mut f = FieldBuilder::valid("f", 0);
        f.doc_values_skip_index_type = Some(2); // out of range
        b.fields.push(f);
        assert!(matches!(
            parse(&b.build(), &b.id, &b.suffix),
            Err(Error::InvalidDocValuesSkipIndexType(2))
        ));
    }

    #[test]
    fn invalid_vector_encoding_byte_rejected() {
        let mut b = FnmBuilder::valid();
        let mut f = FieldBuilder::valid("f", 0);
        f.vector_encoding = 2; // out of range
        b.fields.push(f);
        assert!(matches!(
            parse(&b.build(), &b.id, &b.suffix),
            Err(Error::InvalidVectorEncoding(2))
        ));
    }

    #[test]
    fn invalid_vector_similarity_byte_rejected() {
        let mut b = FnmBuilder::valid();
        let mut f = FieldBuilder::valid("f", 0);
        f.vector_similarity_function = 4; // out of range
        b.fields.push(f);
        assert!(matches!(
            parse(&b.build(), &b.id, &b.suffix),
            Err(Error::InvalidVectorSimilarityFunction(4))
        ));
    }

    #[test]
    fn points_field_parses_dimensions() {
        let mut b = FnmBuilder::valid();
        let mut f = FieldBuilder::valid("point", 0);
        f.point_dimension_count = 1;
        f.point_index_dimension_count = 1;
        f.point_num_bytes = 8;
        b.fields.push(f);
        let fis = parse(&b.build(), &b.id, &b.suffix).unwrap();
        assert_eq!(fis.fields[0].point_dimension_count, 1);
        assert_eq!(fis.fields[0].point_num_bytes, 8);
    }

    // --- check_consistency ---

    #[test]
    fn payloads_without_positions_rejected() {
        let mut b = FnmBuilder::valid();
        let mut f = FieldBuilder::valid("f", 0);
        f.index_options = 2; // DocsAndFreqs: no positions
        f.bits = STORE_PAYLOADS;
        b.fields.push(f);
        assert!(matches!(
            parse(&b.build(), &b.id, &b.suffix),
            Err(Error::Inconsistent(_, _))
        ));
    }

    #[test]
    fn payloads_with_positions_accepted() {
        let mut b = FnmBuilder::valid();
        let mut f = FieldBuilder::valid("f", 0);
        f.index_options = 3; // DocsAndFreqsAndPositions
        f.bits = STORE_PAYLOADS;
        b.fields.push(f);
        let fis = parse(&b.build(), &b.id, &b.suffix).unwrap();
        assert!(fis.fields[0].store_payloads);
    }

    /// A non-indexed field carrying any of the three indexed-only bits is a
    /// file real Lucene **reads**: `Lucene94FieldInfosFormat.read` builds each
    /// `FieldInfo` through the constructor, whose "for non-indexed fields,
    /// leave defaults" branch clears all three *before* `checkConsistency()`
    /// runs. This port rejected the file instead, which made a segment its own
    /// writer had produced unopenable (c23). Real Lucene's own answer for the
    /// same bytes is pinned by `fixtures/src/VerifyFieldInfos.java`'s
    /// `noindex_flags` field.
    #[test]
    fn non_indexed_field_has_its_indexed_only_bits_coerced_away_not_rejected() {
        for bits in [
            STORE_TERMVECTOR,
            STORE_PAYLOADS,
            OMIT_NORMS,
            STORE_TERMVECTOR | STORE_PAYLOADS | OMIT_NORMS,
        ] {
            let mut b = FnmBuilder::valid();
            let mut f = FieldBuilder::valid("f", 0);
            f.index_options = 0; // None
            f.bits = bits;
            b.fields.push(f);
            let fis = parse(&b.build(), &b.id, &b.suffix)
                .unwrap_or_else(|e| panic!("bits {bits:#04x} must parse, got {e}"));
            let f = &fis.fields[0];
            assert!(!f.store_term_vectors, "bits {bits:#04x}");
            assert!(!f.store_payloads, "bits {bits:#04x}");
            assert!(!f.omit_norms, "bits {bits:#04x}");
        }
    }

    // --- the constructor (`FieldInfo::new` / `with_*` / `checked`) ---

    #[test]
    fn the_constructor_seed_is_consistent_and_carries_javas_non_vector_defaults() {
        let f = FieldInfo::new("f", 3).checked().unwrap();
        assert_eq!(f.name, "f");
        assert_eq!(f.number, 3);
        assert_eq!(f.index_options, IndexOptions::None);
        assert_eq!(f.doc_values_type, DocValuesType::None);
        assert_eq!(f.doc_values_gen, -1);
        assert_eq!(f.vector_encoding, VectorEncoding::Float32);
        assert_eq!(
            f.vector_similarity_function,
            VectorSimilarityFunction::Euclidean
        );
    }

    #[test]
    fn the_constructor_coerces_the_three_indexed_only_flags_off_a_non_indexed_field() {
        let f = FieldInfo::new("f", 0)
            .with_store_term_vectors(true)
            .with_store_payloads(true)
            .with_omit_norms(true)
            .checked()
            .expect("Java's constructor coerces rather than throwing here");
        assert!(!f.store_term_vectors);
        assert!(!f.store_payloads);
        assert!(!f.omit_norms);
    }

    #[test]
    fn the_constructor_rejects_payloads_without_positions() {
        // The one indexed-field combination the coercion does NOT rescue: the
        // field *is* indexed, so `storePayloads` survives to `checkConsistency`.
        let err = FieldInfo::new("f", 0)
            .with_index_options(IndexOptions::DocsAndFreqs)
            .with_store_payloads(true)
            .checked()
            .unwrap_err();
        assert!(matches!(err, Error::Inconsistent(_, _)), "{err}");
        // ... and accepts it once positions are indexed.
        assert!(FieldInfo::new("f", 0)
            .with_index_options(IndexOptions::DocsAndFreqsAndPositions)
            .with_store_payloads(true)
            .checked()
            .is_ok());
    }

    #[test]
    fn the_constructor_rejects_every_other_check_consistency_violation() {
        let cases: Vec<FieldInfo> = vec![
            // docValuesSkipIndex incompatible with the doc-values type
            FieldInfo::new("f", 0).with_doc_values(
                DocValuesType::None,
                DocValuesSkipIndexType::Range,
                -1,
            ),
            // a docvalues update generation without doc values
            FieldInfo::new("f", 0).with_doc_values(
                DocValuesType::None,
                DocValuesSkipIndexType::None,
                5,
            ),
            FieldInfo::new("f", 0).with_points(-1, 0, 0),
            FieldInfo::new("f", 0).with_points(1, -1, 4),
            FieldInfo::new("f", 0).with_points(1, 1, -1),
            // pointNumBytes must be > 0 when pointDimensionCount != 0
            FieldInfo::new("f", 0).with_points(1, 1, 0),
            // pointIndexDimensionCount must be 0 when pointDimensionCount == 0
            FieldInfo::new("f", 0).with_points(0, 1, 0),
            // pointDimensionCount must be > 0 when pointNumBytes != 0
            FieldInfo::new("f", 0).with_points(0, 0, 4),
            FieldInfo::new("f", 0).with_vectors(
                -1,
                VectorEncoding::Float32,
                VectorSimilarityFunction::Cosine,
            ),
            FieldInfo::new("f", 0)
                .with_soft_deletes_field(true)
                .with_parent_field(true),
        ];
        for case in cases {
            let described = format!("{case:?}");
            assert!(
                matches!(case.checked(), Err(Error::Inconsistent(_, _))),
                "must be rejected: {described}"
            );
        }
    }

    #[test]
    fn the_constructor_setters_each_reach_the_field_they_name() {
        let f = FieldInfo::new("f", 7)
            .with_index_options(IndexOptions::DocsAndFreqsAndPositionsAndOffsets)
            .with_store_term_vectors(true)
            .with_omit_norms(true)
            .with_store_payloads(true)
            .with_doc_values(
                DocValuesType::SortedNumeric,
                DocValuesSkipIndexType::Range,
                4,
            )
            .with_attributes(vec![("k".to_string(), "v".to_string())])
            .with_points(2, 1, 8)
            .with_vectors(
                16,
                VectorEncoding::Byte,
                VectorSimilarityFunction::MaximumInnerProduct,
            )
            .with_soft_deletes_field(true)
            .checked()
            .unwrap();
        assert_eq!(
            f.index_options,
            IndexOptions::DocsAndFreqsAndPositionsAndOffsets
        );
        assert!(f.store_term_vectors && f.omit_norms && f.store_payloads);
        assert_eq!(f.doc_values_type, DocValuesType::SortedNumeric);
        assert_eq!(f.doc_values_skip_index_type, DocValuesSkipIndexType::Range);
        assert_eq!(f.doc_values_gen, 4);
        assert_eq!(f.attributes, vec![("k".to_string(), "v".to_string())]);
        assert_eq!(
            (
                f.point_dimension_count,
                f.point_index_dimension_count,
                f.point_num_bytes
            ),
            (2, 1, 8)
        );
        assert_eq!(f.vector_dimension, 16);
        assert_eq!(f.vector_encoding, VectorEncoding::Byte);
        assert_eq!(
            f.vector_similarity_function,
            VectorSimilarityFunction::MaximumInnerProduct
        );
        assert!(f.soft_deletes_field);
        assert!(
            FieldInfo::new("f", 0)
                .with_parent_field(true)
                .checked()
                .unwrap()
                .parent_field
        );
    }

    #[test]
    fn field_infos_new_runs_both_the_per_field_and_the_cross_field_checks() {
        // Per-field: the payloads-without-positions violation above.
        assert!(matches!(
            FieldInfos::new(vec![FieldInfo::new("f", 0)
                .with_index_options(IndexOptions::DocsAndFreqs)
                .with_store_payloads(true)]),
            Err(Error::Inconsistent(_, _))
        ));
        // Cross-field: duplicate names, duplicate numbers, two soft-deletes
        // fields, two parent fields.
        assert!(matches!(
            FieldInfos::new(vec![FieldInfo::new("f", 0), FieldInfo::new("f", 1)]),
            Err(Error::InvalidFieldInfos(_))
        ));
        assert!(matches!(
            FieldInfos::new(vec![FieldInfo::new("a", 0), FieldInfo::new("b", 0)]),
            Err(Error::InvalidFieldInfos(_))
        ));
        assert!(matches!(
            FieldInfos::new(vec![
                FieldInfo::new("a", 0).with_soft_deletes_field(true),
                FieldInfo::new("b", 1).with_soft_deletes_field(true),
            ]),
            Err(Error::InvalidFieldInfos(_))
        ));
        assert!(matches!(
            FieldInfos::new(vec![
                FieldInfo::new("a", 0).with_parent_field(true),
                FieldInfo::new("b", 1).with_parent_field(true),
            ]),
            Err(Error::InvalidFieldInfos(_))
        ));
        // ... and the coercion is applied on the way through.
        let infos = FieldInfos::new(vec![FieldInfo::new("a", 0)
            .with_omit_norms(true)
            .with_store_term_vectors(true)])
        .unwrap();
        assert!(!infos.fields[0].omit_norms);
        assert!(!infos.fields[0].store_term_vectors);
    }

    #[test]
    fn doc_values_gen_without_doc_values_rejected() {
        let mut b = FnmBuilder::valid();
        let mut f = FieldBuilder::valid("f", 0);
        f.doc_values_type = 0; // None
        f.doc_values_gen = 5;
        b.fields.push(f);
        assert!(matches!(
            parse(&b.build(), &b.id, &b.suffix),
            Err(Error::Inconsistent(_, _))
        ));
    }

    #[test]
    fn point_num_bytes_zero_with_nonzero_dimension_count_rejected() {
        // Bypass the builder's own `if point_dimension_count != 0` write-side
        // branch (which always writes a consistent numBytes) by writing the
        // point fields by hand: dimension count 1, index dim count 1, num
        // bytes 0 — an invariant violation only reachable via crafted bytes.
        let mut b = FnmBuilder::valid();
        b.fields.push(FieldBuilder::valid("f", 0));
        let mut bytes = b.build();
        // Find the point_dimension_count vint (0x00) just before the trailing
        // vector fields + footer, and hand-patch a violation in a fresh
        // buffer instead, since the builder always keeps points consistent.
        bytes.clear();
        let mut out = Vec::new();
        out.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
        write_string(&mut out, CODEC_NAME);
        out.extend_from_slice(&(FORMAT_CURRENT as u32).to_be_bytes());
        out.extend_from_slice(&b.id);
        out.push(0); // empty suffix
        write_vint(&mut out, 1); // one field
        write_string(&mut out, "f");
        write_vint(&mut out, 0); // number
        out.push(0); // bits
        out.push(0); // index options: None
        out.push(0); // doc values type: None
        out.push(0); // doc values skip index: None
        out.extend_from_slice(&(-1i64).to_le_bytes()); // doc values gen
        write_vint(&mut out, 0); // attributes
        write_vint(&mut out, 1); // pointDimensionCount = 1
        write_vint(&mut out, 1); // pointIndexDimensionCount = 1
        write_vint(&mut out, 0); // pointNumBytes = 0 (invalid: must be >0)
        write_vint(&mut out, 0); // vectorDimension
        out.push(0); // vector encoding
        out.push(0); // vector similarity
        out.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
        out.extend_from_slice(&0u32.to_be_bytes());
        let checksum = crc32fast::hash(&out) as u64;
        out.extend_from_slice(&checksum.to_be_bytes());

        assert!(matches!(
            parse(&out, &b.id, &b.suffix),
            Err(Error::Inconsistent(_, _))
        ));
    }

    // --- write() round-trips through parse() ---

    fn sample_field(name: &str, number: i32) -> FieldInfo {
        FieldInfo {
            name: name.to_string(),
            number,
            store_term_vectors: false,
            omit_norms: false,
            store_payloads: false,
            soft_deletes_field: false,
            parent_field: false,
            index_options: IndexOptions::Docs,
            doc_values_type: DocValuesType::None,
            doc_values_skip_index_type: DocValuesSkipIndexType::None,
            doc_values_gen: -1,
            attributes: vec![],
            point_dimension_count: 0,
            point_index_dimension_count: 0,
            point_num_bytes: 0,
            vector_dimension: 0,
            vector_encoding: VectorEncoding::Float32,
            vector_similarity_function: VectorSimilarityFunction::Euclidean,
        }
    }

    #[test]
    fn write_empty_round_trips() {
        let id = [7u8; ID_LENGTH];
        let bytes = write(&[], &id, "");
        let fis = parse(&bytes, &id, "").unwrap();
        assert_eq!(fis.fields.len(), 0);
    }

    #[test]
    fn write_plain_field_round_trips() {
        let id = [7u8; ID_LENGTH];
        let field = sample_field("id", 0);
        let bytes = write(&[field], &id, "");
        let fis = parse(&bytes, &id, "").unwrap();
        assert_eq!(fis.fields.len(), 1);
        assert_eq!(fis.fields[0].name, "id");
        assert_eq!(fis.fields[0].number, 0);
        assert_eq!(fis.fields[0].index_options, IndexOptions::Docs);
    }

    #[test]
    fn write_term_vectors_and_payloads_round_trip() {
        let id = [7u8; ID_LENGTH];
        let mut field = sample_field("with_tv", 1);
        field.store_term_vectors = true;
        field.store_payloads = true;
        field.index_options = IndexOptions::DocsAndFreqsAndPositions;
        let bytes = write(&[field], &id, "sfx");
        let fis = parse(&bytes, &id, "sfx").unwrap();
        assert!(fis.fields[0].store_term_vectors);
        assert!(fis.fields[0].store_payloads);
    }

    #[test]
    fn write_soft_deletes_and_parent_field_round_trip() {
        let id = [7u8; ID_LENGTH];
        let mut soft = sample_field("__soft_deletes", 2);
        soft.soft_deletes_field = true;
        let mut parent = sample_field("__parent", 3);
        parent.parent_field = true;
        let bytes = write(&[soft, parent], &id, "");
        let fis = parse(&bytes, &id, "").unwrap();
        assert!(fis.fields[0].soft_deletes_field);
        assert!(fis.fields[1].parent_field);
    }

    #[test]
    fn write_doc_values_and_skip_index_round_trip() {
        let id = [7u8; ID_LENGTH];
        let mut field = sample_field("num_dv", 0);
        field.doc_values_type = DocValuesType::Numeric;
        field.doc_values_skip_index_type = DocValuesSkipIndexType::Range;
        field.doc_values_gen = 42;
        field.attributes = vec![("k1".to_string(), "v1".to_string())];
        let bytes = write(&[field], &id, "");
        let fis = parse(&bytes, &id, "").unwrap();
        assert_eq!(fis.fields[0].doc_values_type, DocValuesType::Numeric);
        assert_eq!(
            fis.fields[0].doc_values_skip_index_type,
            DocValuesSkipIndexType::Range
        );
        assert_eq!(fis.fields[0].doc_values_gen, 42);
        assert_eq!(
            fis.fields[0].attributes,
            vec![("k1".to_string(), "v1".to_string())]
        );
    }

    #[test]
    fn write_points_field_round_trips() {
        let id = [7u8; ID_LENGTH];
        let mut field = sample_field("point_field", 0);
        field.point_dimension_count = 1;
        field.point_index_dimension_count = 1;
        field.point_num_bytes = 8;
        let bytes = write(&[field], &id, "");
        let fis = parse(&bytes, &id, "").unwrap();
        assert_eq!(fis.fields[0].point_dimension_count, 1);
        assert_eq!(fis.fields[0].point_index_dimension_count, 1);
        assert_eq!(fis.fields[0].point_num_bytes, 8);
    }

    #[test]
    fn write_vector_field_round_trips() {
        let id = [7u8; ID_LENGTH];
        let mut field = sample_field("vector_field", 0);
        field.vector_dimension = 3;
        field.vector_encoding = VectorEncoding::Byte;
        field.vector_similarity_function = VectorSimilarityFunction::Cosine;
        let bytes = write(&[field], &id, "");
        let fis = parse(&bytes, &id, "").unwrap();
        assert_eq!(fis.fields[0].vector_dimension, 3);
        assert_eq!(fis.fields[0].vector_encoding, VectorEncoding::Byte);
        assert_eq!(
            fis.fields[0].vector_similarity_function,
            VectorSimilarityFunction::Cosine
        );
    }

    #[test]
    fn write_multiple_fields_preserve_order() {
        let id = [7u8; ID_LENGTH];
        let fields = vec![
            sample_field("a", 0),
            sample_field("b", 1),
            sample_field("c", 2),
        ];
        let bytes = write(&fields, &id, "");
        let fis = parse(&bytes, &id, "").unwrap();
        assert_eq!(fis.fields.len(), 3);
        assert_eq!(fis.fields[0].name, "a");
        assert_eq!(fis.fields[1].name, "b");
        assert_eq!(fis.fields[2].name, "c");
    }

    #[test]
    fn wrong_id_rejected() {
        let b = FnmBuilder::valid();
        let wrong_id = [9u8; ID_LENGTH];
        assert!(matches!(
            parse(&b.build(), &wrong_id, &b.suffix),
            Err(Error::Store(_))
        ));
    }
}
