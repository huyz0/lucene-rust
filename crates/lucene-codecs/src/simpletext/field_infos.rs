//! `SimpleTextFieldInfosFormat`: a segment's field infos as one `.inf` text
//! file.
//!
//! ```text
//! number of fields N
//!   name <name>                         per field, in FieldInfos order
//!   number <n>
//!   index options <IndexOptions name>
//!   term vectors <bool>
//!   payloads <bool>
//!   norms <bool>                        !omitNorms
//!   doc values <DocValuesType name>
//!   doc values skip index<NONE|RANGE>   (no space before the value)
//!   doc values gen <long>
//!   attributes <count>
//!     key <k>
//!     value <v>
//!   data dimensional count <n>
//!   index dimensional count <n>
//!   dimensional num bytes <n>
//!   vector number of dimensions <n>
//!   vector encoding <BYTE|FLOAT32>
//!   vector similarity <VectorSimilarityFunction name>
//!   soft-deletes <bool>
//!   parent <bool>
//! checksum <20 digits>
//! ```
//!
//! The values are the enums' Java `name()`s. Attributes keep the order they
//! are given in (Java writes its `HashMap`'s order and reads into a fresh
//! one), so a re-write of what was read reproduces the file.

use super::util::{
    self, boolean_str, parse_boolean, parse_integer, parse_long, read_string, read_value,
};
use super::{corrupt, Result};
use crate::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, FieldInfos, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};

/// `SimpleTextFieldInfosFormat.FIELD_INFOS_EXTENSION`.
pub const FIELD_INFOS_EXTENSION: &str = "inf";

/// `NUMFIELDS`.
pub const NUMFIELDS: &[u8] = b"number of fields ";
/// `NAME`.
pub const NAME: &[u8] = b"  name ";
/// `NUMBER`.
pub const NUMBER: &[u8] = b"  number ";
/// `STORETV`.
pub const STORETV: &[u8] = b"  term vectors ";
/// `STORETVPOS` (declared by Java, never written).
pub const STORETVPOS: &[u8] = b"  term vector positions ";
/// `STORETVOFF` (declared by Java, never written).
pub const STORETVOFF: &[u8] = b"  term vector offsets ";
/// `PAYLOADS`.
pub const PAYLOADS: &[u8] = b"  payloads ";
/// `NORMS`.
pub const NORMS: &[u8] = b"  norms ";
/// `DOCVALUES`.
pub const DOCVALUES: &[u8] = b"  doc values ";
/// `DOCVALUES_SKIP_INDEX` -- no trailing space, as Java's.
pub const DOCVALUES_SKIP_INDEX: &[u8] = b"  doc values skip index";
/// `DOCVALUES_GEN`.
pub const DOCVALUES_GEN: &[u8] = b"  doc values gen ";
/// `INDEXOPTIONS`.
pub const INDEXOPTIONS: &[u8] = b"  index options ";
/// `NUM_ATTS`.
pub const NUM_ATTS: &[u8] = b"  attributes ";
/// `ATT_KEY`.
pub const ATT_KEY: &[u8] = b"    key ";
/// `ATT_VALUE`.
pub const ATT_VALUE: &[u8] = b"    value ";
/// `DATA_DIM_COUNT`.
pub const DATA_DIM_COUNT: &[u8] = b"  data dimensional count ";
/// `INDEX_DIM_COUNT`.
pub const INDEX_DIM_COUNT: &[u8] = b"  index dimensional count ";
/// `DIM_NUM_BYTES`.
pub const DIM_NUM_BYTES: &[u8] = b"  dimensional num bytes ";
/// `VECTOR_NUM_DIMS`.
pub const VECTOR_NUM_DIMS: &[u8] = b"  vector number of dimensions ";
/// `VECTOR_ENCODING`.
pub const VECTOR_ENCODING: &[u8] = b"  vector encoding ";
/// `VECTOR_SIMILARITY`.
pub const VECTOR_SIMILARITY: &[u8] = b"  vector similarity ";
/// `SOFT_DELETES`.
pub const SOFT_DELETES: &[u8] = b"  soft-deletes ";
/// `PARENT`.
pub const PARENT: &[u8] = b"  parent ";

/// `IndexFileNames.segmentFileName(segment, suffix, "inf")`.
pub fn field_infos_file_name(segment: &str, suffix: &str) -> String {
    super::segment_file_name(segment, suffix, FIELD_INFOS_EXTENSION)
}

/// `IndexOptions.toString()`.
pub fn index_options_name(o: IndexOptions) -> &'static str {
    match o {
        IndexOptions::None => "NONE",
        IndexOptions::Docs => "DOCS",
        IndexOptions::DocsAndFreqs => "DOCS_AND_FREQS",
        IndexOptions::DocsAndFreqsAndPositions => "DOCS_AND_FREQS_AND_POSITIONS",
        IndexOptions::DocsAndFreqsAndPositionsAndOffsets => {
            "DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS"
        }
        IndexOptions::DocsAndCustomFreqs => "DOCS_AND_CUSTOM_FREQS",
    }
}

/// `DocValuesType.toString()`.
pub fn doc_values_type_name(t: DocValuesType) -> &'static str {
    match t {
        DocValuesType::None => "NONE",
        DocValuesType::Numeric => "NUMERIC",
        DocValuesType::Binary => "BINARY",
        DocValuesType::Sorted => "SORTED",
        DocValuesType::SortedSet => "SORTED_SET",
        DocValuesType::SortedNumeric => "SORTED_NUMERIC",
    }
}

/// `DocValuesSkipIndexType.toString()`.
pub fn skip_index_type_name(t: DocValuesSkipIndexType) -> &'static str {
    match t {
        DocValuesSkipIndexType::None => "NONE",
        DocValuesSkipIndexType::Range => "RANGE",
    }
}

/// `VectorEncoding.name()`.
pub fn vector_encoding_name(e: VectorEncoding) -> &'static str {
    match e {
        VectorEncoding::Byte => "BYTE",
        VectorEncoding::Float32 => "FLOAT32",
    }
}

/// `VectorSimilarityFunction.name()`.
pub fn similarity_name(f: VectorSimilarityFunction) -> &'static str {
    match f {
        VectorSimilarityFunction::Euclidean => "EUCLIDEAN",
        VectorSimilarityFunction::DotProduct => "DOT_PRODUCT",
        VectorSimilarityFunction::Cosine => "COSINE",
        VectorSimilarityFunction::MaximumInnerProduct => "MAXIMUM_INNER_PRODUCT",
    }
}

/// `Enum.valueOf(class, name)` over one of the `*_name` tables.
fn value_of<T: Copy>(
    s: &str,
    what: &str,
    all: &[T],
    name: fn(T) -> &'static str,
) -> Result<T> {
    all.iter().copied().find(|&v| name(v) == s).ok_or_else(|| {
        super::Error::IllegalArgument(format!("No enum constant {what}.{s}"))
    })
}

/// `IndexOptions.valueOf(s)`.
///
/// # Errors
/// [`super::Error::IllegalArgument`] for an unknown name, as Java's.
pub fn index_options(s: &str) -> Result<IndexOptions> {
    value_of(
        s,
        "org.apache.lucene.index.IndexOptions",
        &[
            IndexOptions::None,
            IndexOptions::Docs,
            IndexOptions::DocsAndFreqs,
            IndexOptions::DocsAndFreqsAndPositions,
            IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
            IndexOptions::DocsAndCustomFreqs,
        ],
        index_options_name,
    )
}

/// `docValuesType(dvType)`: `DocValuesType.valueOf`.
///
/// # Errors
/// [`super::Error::IllegalArgument`] for an unknown name.
pub fn doc_values_type(s: &str) -> Result<DocValuesType> {
    value_of(
        s,
        "org.apache.lucene.index.DocValuesType",
        &[
            DocValuesType::None,
            DocValuesType::Numeric,
            DocValuesType::Binary,
            DocValuesType::Sorted,
            DocValuesType::SortedSet,
            DocValuesType::SortedNumeric,
        ],
        doc_values_type_name,
    )
}

/// `docValuesSkipIndexType(s)`: `DocValuesSkipIndexType.valueOf`.
///
/// # Errors
/// [`super::Error::IllegalArgument`] for an unknown name.
pub fn doc_values_skip_index_type(s: &str) -> Result<DocValuesSkipIndexType> {
    value_of(
        s,
        "org.apache.lucene.index.DocValuesSkipIndexType",
        &[DocValuesSkipIndexType::None, DocValuesSkipIndexType::Range],
        skip_index_type_name,
    )
}

/// `vectorEncoding(s)`: `VectorEncoding.valueOf`.
///
/// # Errors
/// [`super::Error::IllegalArgument`] for an unknown name.
pub fn vector_encoding(s: &str) -> Result<VectorEncoding> {
    value_of(
        s,
        "org.apache.lucene.index.VectorEncoding",
        &[VectorEncoding::Byte, VectorEncoding::Float32],
        vector_encoding_name,
    )
}

/// `distanceFunction(s)`: `VectorSimilarityFunction.valueOf`.
///
/// # Errors
/// [`super::Error::IllegalArgument`] for an unknown name.
pub fn distance_function(s: &str) -> Result<VectorSimilarityFunction> {
    value_of(
        s,
        "org.apache.lucene.index.VectorSimilarityFunction",
        &[
            VectorSimilarityFunction::Euclidean,
            VectorSimilarityFunction::DotProduct,
            VectorSimilarityFunction::Cosine,
            VectorSimilarityFunction::MaximumInnerProduct,
        ],
        similarity_name,
    )
}

/// `SimpleTextFieldInfosFormat.read`: the `.inf` file's field infos.
///
/// # Errors
/// `Corrupted` for a line without its prefix, a number that does not parse,
/// a negative field count or a bad checksum; `Eof` for a short file;
/// [`super::Error::IllegalArgument`] for an unknown enum name; and what
/// `new FieldInfos(infos)` refuses ([`super::Error::FieldInfos`]).
pub fn read(data: &[u8]) -> Result<FieldInfos> {
    let mut pos = 0usize;
    let mut scratch = Vec::new();
    let size = parse_integer(read_value(data, &mut pos, &mut scratch, NUMFIELDS)?)?;
    // Java's `new FieldInfo[size]`, sized from the file; a field takes at
    // least one byte per line, so cap the reservation by the file.
    let size = usize::try_from(size).map_err(|_| corrupt(format!("invalid field count: {size}")))?;
    let mut infos = Vec::with_capacity(size.min(data.len()));
    for _ in 0..size {
        let name = read_string(data, &mut pos, &mut scratch, NAME)?;
        let field_number = parse_integer(read_value(data, &mut pos, &mut scratch, NUMBER)?)?;
        let index_options = index_options(&read_string(data, &mut pos, &mut scratch, INDEXOPTIONS)?)?;
        let store_term_vector = parse_boolean(read_value(data, &mut pos, &mut scratch, STORETV)?);
        let store_payloads = parse_boolean(read_value(data, &mut pos, &mut scratch, PAYLOADS)?);
        let omit_norms = !parse_boolean(read_value(data, &mut pos, &mut scratch, NORMS)?);
        let doc_values_type =
            doc_values_type(&read_string(data, &mut pos, &mut scratch, DOCVALUES)?)?;
        let skip_index = doc_values_skip_index_type(&read_string(
            data,
            &mut pos,
            &mut scratch,
            DOCVALUES_SKIP_INDEX,
        )?)?;
        let dv_gen = parse_long(read_value(data, &mut pos, &mut scratch, DOCVALUES_GEN)?)?;
        let num_atts = parse_integer(read_value(data, &mut pos, &mut scratch, NUM_ATTS)?)?;
        let mut atts: Vec<(String, String)> = Vec::new();
        for _ in 0..num_atts {
            let key = read_string(data, &mut pos, &mut scratch, ATT_KEY)?;
            let value = read_string(data, &mut pos, &mut scratch, ATT_VALUE)?;
            // `atts.put(key, value)`: a repeated key keeps its first slot
            // and takes the last value.
            match atts.iter_mut().find(|(k, _)| *k == key) {
                Some(slot) => slot.1 = value,
                None => atts.push((key, value)),
            }
        }
        let dimensional_count =
            parse_integer(read_value(data, &mut pos, &mut scratch, DATA_DIM_COUNT)?)?;
        let index_dimensional_count =
            parse_integer(read_value(data, &mut pos, &mut scratch, INDEX_DIM_COUNT)?)?;
        let dimensional_num_bytes =
            parse_integer(read_value(data, &mut pos, &mut scratch, DIM_NUM_BYTES)?)?;
        let vector_num_dimensions =
            parse_integer(read_value(data, &mut pos, &mut scratch, VECTOR_NUM_DIMS)?)?;
        let encoding =
            vector_encoding(&read_string(data, &mut pos, &mut scratch, VECTOR_ENCODING)?)?;
        let similarity =
            distance_function(&read_string(data, &mut pos, &mut scratch, VECTOR_SIMILARITY)?)?;
        let soft_deletes = parse_boolean(read_value(data, &mut pos, &mut scratch, SOFT_DELETES)?);
        let parent = parse_boolean(read_value(data, &mut pos, &mut scratch, PARENT)?);
        infos.push(
            FieldInfo::new(name, field_number)
                .with_index_options(index_options)
                .with_store_term_vectors(store_term_vector)
                .with_omit_norms(omit_norms)
                .with_store_payloads(store_payloads)
                .with_doc_values(doc_values_type, skip_index, dv_gen)
                .with_attributes(atts)
                .with_points(
                    dimensional_count,
                    index_dimensional_count,
                    dimensional_num_bytes,
                )
                .with_vectors(vector_num_dimensions, encoding, similarity)
                .with_soft_deletes_field(soft_deletes)
                .with_parent_field(parent),
        );
    }
    util::check_footer(data, 0, &mut pos)?;
    Ok(FieldInfos::new(infos)?)
}

/// `SimpleTextFieldInfosFormat.write`: the `.inf` file for `infos`, in
/// their order.
pub fn write(infos: &[FieldInfo]) -> Vec<u8> {
    let mut out = Vec::new();
    util::write_line(&mut out, NUMFIELDS, infos.len().to_string().as_bytes());
    for fi in infos {
        util::write_line(&mut out, NAME, fi.name.as_bytes());
        util::write_line(&mut out, NUMBER, fi.number.to_string().as_bytes());
        util::write_line(
            &mut out,
            INDEXOPTIONS,
            index_options_name(fi.index_options).as_bytes(),
        );
        util::write_line(&mut out, STORETV, boolean_str(fi.store_term_vectors));
        util::write_line(&mut out, PAYLOADS, boolean_str(fi.store_payloads));
        util::write_line(&mut out, NORMS, boolean_str(!fi.omit_norms));
        util::write_line(
            &mut out,
            DOCVALUES,
            doc_values_type_name(fi.doc_values_type).as_bytes(),
        );
        util::write_line(
            &mut out,
            DOCVALUES_SKIP_INDEX,
            skip_index_type_name(fi.doc_values_skip_index_type).as_bytes(),
        );
        util::write_line(&mut out, DOCVALUES_GEN, fi.doc_values_gen.to_string().as_bytes());
        util::write_line(&mut out, NUM_ATTS, fi.attributes.len().to_string().as_bytes());
        for (key, value) in &fi.attributes {
            util::write_line(&mut out, ATT_KEY, key.as_bytes());
            util::write_line(&mut out, ATT_VALUE, value.as_bytes());
        }
        util::write_line(
            &mut out,
            DATA_DIM_COUNT,
            fi.point_dimension_count.to_string().as_bytes(),
        );
        util::write_line(
            &mut out,
            INDEX_DIM_COUNT,
            fi.point_index_dimension_count.to_string().as_bytes(),
        );
        util::write_line(&mut out, DIM_NUM_BYTES, fi.point_num_bytes.to_string().as_bytes());
        util::write_line(
            &mut out,
            VECTOR_NUM_DIMS,
            fi.vector_dimension.to_string().as_bytes(),
        );
        util::write_line(
            &mut out,
            VECTOR_ENCODING,
            vector_encoding_name(fi.vector_encoding).as_bytes(),
        );
        util::write_line(
            &mut out,
            VECTOR_SIMILARITY,
            similarity_name(fi.vector_similarity_function).as_bytes(),
        );
        util::write_line(&mut out, SOFT_DELETES, boolean_str(fi.soft_deletes_field));
        util::write_line(&mut out, PARENT, boolean_str(fi.parent_field));
    }
    util::write_checksum(&mut out);
    out
}
