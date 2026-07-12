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
    pub(crate) fn subsumes_positions(self) -> bool {
        matches!(
            self,
            Self::DocsAndFreqsAndPositions | Self::DocsAndFreqsAndPositionsAndOffsets
        )
    }

    /// Port of `IndexOptions.subsumes(DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS)`:
    /// whether this option indexes character offsets.
    pub(crate) fn subsumes_offsets(self) -> bool {
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
    /// Port of `FieldInfo.checkConsistency` (the subset of invariants that
    /// don't require comparing against sibling fields, which is all Java
    /// checks here too — `verifySameSchema` is a separate, merge-time check
    /// out of scope for this read-only parser).
    fn check_consistency(&self) -> Result<()> {
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
    pub fn field_by_number(&self, number: i32) -> Option<&FieldInfo> {
        self.fields.iter().find(|f| f.number == number)
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
        field.check_consistency()?;
        fields.push(field);
    }

    codec_util::check_footer(&mut input, buf.len())?;

    Ok(FieldInfos { fields })
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

        let mut bits = 0u8;
        if f.store_term_vectors {
            bits |= STORE_TERMVECTOR;
        }
        if f.omit_norms {
            bits |= OMIT_NORMS;
        }
        if f.store_payloads {
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

    #[test]
    fn non_indexed_field_cannot_store_term_vectors() {
        let mut b = FnmBuilder::valid();
        let mut f = FieldBuilder::valid("f", 0);
        f.index_options = 0; // None
        f.bits = STORE_TERMVECTOR;
        b.fields.push(f);
        assert!(matches!(
            parse(&b.build(), &b.id, &b.suffix),
            Err(Error::Inconsistent(_, _))
        ));
    }

    #[test]
    fn non_indexed_field_cannot_store_payloads() {
        let mut b = FnmBuilder::valid();
        let mut f = FieldBuilder::valid("f", 0);
        f.index_options = 0;
        f.bits = STORE_PAYLOADS;
        b.fields.push(f);
        assert!(matches!(
            parse(&b.build(), &b.id, &b.suffix),
            Err(Error::Inconsistent(_, _))
        ));
    }

    #[test]
    fn non_indexed_field_cannot_omit_norms() {
        let mut b = FnmBuilder::valid();
        let mut f = FieldBuilder::valid("f", 0);
        f.index_options = 0;
        f.bits = OMIT_NORMS;
        b.fields.push(f);
        assert!(matches!(
            parse(&b.build(), &b.id, &b.suffix),
            Err(Error::Inconsistent(_, _))
        ));
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
