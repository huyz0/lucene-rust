//! The four retired quantized `KnnVectorsFormat`s as one per-field reader:
//! `Lucene99ScalarQuantizedVectorsFormat` and `Lucene102BinaryQuantizedVectorsFormat`
//! (flat: raw vectors plus quantized codes, no graph) and their HNSW
//! wrappers `Lucene99HnswScalarQuantizedVectorsFormat` and
//! `Lucene102HnswBinaryQuantizedVectorsFormat` (ports of those four format
//! classes' `fieldsReader`).
//!
//! Every one of them keeps the raw vectors in `Lucene99FlatVectorsFormat`'s
//! `.vec`/`.vemf` ([`crate::vectors`]), the codes in its own pair
//! (`.veq`/`.vemq`, [`super::scalar_quantized_vectors`];
//! `.veb`/`.vemb`, [`super::binary_quantized_vectors`]) and, for an HNSW
//! wrapper, the graph in `Lucene99HnswVectorsFormat`'s `.vem`/`.vex`
//! ([`crate::hnsw_vectors`]) -- all under the per-field suffix
//! (`_0_Lucene99HnswScalarQuantizedVectorsFormat_0.veq`).
//!
//! # How each one searches
//!
//! What a KNN search of such a field does is the format's, and the four
//! differ, which [`SearchKind`] records:
//!
//! - the HNSW wrappers are `Lucene99HnswVectorsReader` over the quantized
//!   flat reader: its graph-or-exhaustive choice, scored by the quantized
//!   scorer ([`QuantizedVectorsReader::float_scorer`]);
//! - `Lucene99ScalarQuantizedVectorsFormat` alone inherits
//!   `FlatVectorsReader.search`, which **finds nothing** ("don't scan stored
//!   field data. If we didn't index it, produce no search results") -- a KNN
//!   query over such a field only answers through its exact-search fallback;
//! - `Lucene102BinaryQuantizedVectorsFormat` alone scores every accepted
//!   ordinal, one `collect` and `incVisitedCount(1)` each, with no early
//!   termination check.
//!
//! A `BYTE` field in any of them is not quantized: it is searched and scored
//! as the raw reader does.

use lucene_store::codec_util::ID_LENGTH;

use super::binary_quantized_vectors::{self, Lucene102BinaryQuantizedVectorsReader};
use super::scalar_quantized_vectors::{self, Lucene99ScalarQuantizedVectorsReader};
use crate::direct_monotonic;
use crate::field_infos::FieldInfos;
use crate::hnsw::VectorScorer;
use crate::hnsw_vectors::HnswVectorsReader;
use crate::indexed_disi::DisiCursor;
use crate::vectors::{
    file_region, DocToOrdCursor, Error, FlatVectorsReader, FloatVectorScorer, OrdToDoc, Result,
};

/// `ordToDoc(ord)` over a quantized format's `OrdToDocDISIReaderConfiguration`,
/// whose sparse structures live in that format's own data file.
pub(crate) fn ord_to_doc(map: &OrdToDoc, file: &[u8], size: i32, ord: i32) -> Result<i32> {
    if ord < 0 || ord >= size {
        return Err(Error::OrdOutOfRange(ord, size));
    }
    match map {
        OrdToDoc::Empty | OrdToDoc::Dense => Ok(ord),
        OrdToDoc::Explicit(docs) => docs
            .get(ord as usize)
            .copied()
            .ok_or(Error::OrdOutOfRange(ord, size)),
        OrdToDoc::Sparse {
            addresses_offset,
            addresses_length,
            meta,
            ..
        } => {
            let region =
                file_region(file, *addresses_offset, *addresses_length).ok_or_else(|| {
                    Error::CorruptMeta("ordToDoc addresses region out of bounds".into())
                })?;
            Ok(direct_monotonic::get(region, meta, i64::from(ord))? as i32)
        }
    }
}

/// The doc -> ordinal direction of the same configuration.
pub(crate) fn doc_to_ord<'a>(
    map: &OrdToDoc,
    file: &'a [u8],
    size: i32,
) -> Result<DocToOrdCursor<'a>> {
    match map {
        OrdToDoc::Empty => Ok(DocToOrdCursor::Empty),
        OrdToDoc::Dense => Ok(DocToOrdCursor::Dense { size }),
        OrdToDoc::Explicit(_) => Err(Error::CorruptMeta(
            "explicit ordToDoc table in a quantized field".into(),
        )),
        OrdToDoc::Sparse {
            docs_with_field_offset,
            docs_with_field_length,
            jump_table_entry_count,
            dense_rank_power,
            ..
        } => {
            let region = file_region(file, *docs_with_field_offset, *docs_with_field_length)
                .ok_or_else(|| Error::CorruptMeta("docsWithField region out of bounds".into()))?;
            Ok(DocToOrdCursor::Sparse(Box::new(DisiCursor::new(
                region,
                *dense_rank_power,
                *jump_table_entry_count,
            ))))
        }
    }
}

/// Which retired quantized format a field was written with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizedFormat {
    /// `Lucene99ScalarQuantizedVectorsFormat`: flat.
    Lucene99ScalarQuantized,
    /// `Lucene99HnswScalarQuantizedVectorsFormat`.
    Lucene99HnswScalarQuantized,
    /// `Lucene102BinaryQuantizedVectorsFormat`: flat.
    Lucene102BinaryQuantized,
    /// `Lucene102HnswBinaryQuantizedVectorsFormat`.
    Lucene102HnswBinaryQuantized,
}

/// How a format's reader answers `search(field, target, collector, acceptDocs)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchKind {
    /// `Lucene99HnswVectorsReader.search`: graph walk or bulk-scored scan.
    Hnsw,
    /// `FlatVectorsReader.search`: nothing is collected.
    Nothing,
    /// `Lucene102BinaryQuantizedVectorsReader.search`: every accepted
    /// ordinal, unconditionally.
    ScanAll,
}

impl QuantizedFormat {
    /// Every format, oldest first.
    pub const ALL: [QuantizedFormat; 4] = [
        QuantizedFormat::Lucene99ScalarQuantized,
        QuantizedFormat::Lucene99HnswScalarQuantized,
        QuantizedFormat::Lucene102BinaryQuantized,
        QuantizedFormat::Lucene102HnswBinaryQuantized,
    ];

    /// `KnnVectorsFormat.forName`, over the name
    /// `PerFieldKnnVectorsFormat.format` records.
    pub fn for_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|f| f.name() == name)
    }

    /// `getName()`.
    pub fn name(self) -> &'static str {
        match self {
            QuantizedFormat::Lucene99ScalarQuantized => scalar_quantized_vectors::NAME,
            QuantizedFormat::Lucene99HnswScalarQuantized => scalar_quantized_vectors::HNSW_NAME,
            QuantizedFormat::Lucene102BinaryQuantized => binary_quantized_vectors::NAME,
            QuantizedFormat::Lucene102HnswBinaryQuantized => binary_quantized_vectors::HNSW_NAME,
        }
    }

    /// Whether the format writes a `.vem`/`.vex` graph.
    pub fn has_graph(self) -> bool {
        matches!(
            self,
            QuantizedFormat::Lucene99HnswScalarQuantized
                | QuantizedFormat::Lucene102HnswBinaryQuantized
        )
    }

    /// The quantized metadata and data extensions.
    pub fn quantized_extensions(self) -> (&'static str, &'static str) {
        match self {
            QuantizedFormat::Lucene99ScalarQuantized
            | QuantizedFormat::Lucene99HnswScalarQuantized => (
                scalar_quantized_vectors::META_EXTENSION,
                scalar_quantized_vectors::DATA_EXTENSION,
            ),
            _ => (
                binary_quantized_vectors::META_EXTENSION,
                binary_quantized_vectors::DATA_EXTENSION,
            ),
        }
    }

    /// See [`SearchKind`].
    pub fn search_kind(self) -> SearchKind {
        match self {
            QuantizedFormat::Lucene99ScalarQuantized => SearchKind::Nothing,
            QuantizedFormat::Lucene102BinaryQuantized => SearchKind::ScanAll,
            _ => SearchKind::Hnsw,
        }
    }

    fn is_binary(self) -> bool {
        matches!(
            self,
            QuantizedFormat::Lucene102BinaryQuantized
                | QuantizedFormat::Lucene102HnswBinaryQuantized
        )
    }
}

/// The quantized flat reader a format carries.
#[derive(Debug, Clone)]
pub enum QuantizedReader<'a> {
    Scalar(Lucene99ScalarQuantizedVectorsReader<'a>),
    Binary(Lucene102BinaryQuantizedVectorsReader<'a>),
}

/// One field group's files, as the directory holds them.
#[derive(Debug, Clone, Copy)]
pub struct QuantizedFiles<'a> {
    pub vemf: &'a [u8],
    pub vec: &'a [u8],
    /// `.vemq` or `.vemb`.
    pub quantized_meta: &'a [u8],
    /// `.veq` or `.veb`.
    pub quantized_data: &'a [u8],
    /// `.vem`/`.vex`, for an HNSW format.
    pub graph: Option<(&'a [u8], &'a [u8])>,
}

/// A retired quantized format's `fieldsReader`: the raw flat reader, the
/// quantized one and, for an HNSW wrapper, the graph.
#[derive(Debug, Clone)]
pub struct QuantizedVectorsReader<'a> {
    format: QuantizedFormat,
    flat: FlatVectorsReader<'a>,
    quantized: QuantizedReader<'a>,
    graph: Option<HnswVectorsReader<'a>>,
}

impl<'a> QuantizedVectorsReader<'a> {
    /// Opens every file of one per-field suffix, and cross-checks the
    /// quantized metadata against the segment's `FieldInfos`.
    pub fn open(
        format: QuantizedFormat,
        files: QuantizedFiles<'a>,
        field_infos: &FieldInfos,
        segment_id: &[u8; ID_LENGTH],
        segment_suffix: &str,
    ) -> Result<Self> {
        let flat = FlatVectorsReader::open(files.vemf, files.vec, segment_id, segment_suffix)?;
        let quantized = if format.is_binary() {
            let r = Lucene102BinaryQuantizedVectorsReader::open(
                files.quantized_meta,
                files.quantized_data,
                segment_id,
                segment_suffix,
            )?;
            r.check_field_infos(field_infos)?;
            QuantizedReader::Binary(r)
        } else {
            let r = Lucene99ScalarQuantizedVectorsReader::open(
                files.quantized_meta,
                files.quantized_data,
                segment_id,
                segment_suffix,
            )?;
            r.check_field_infos(field_infos)?;
            QuantizedReader::Scalar(r)
        };
        let graph = match (format.has_graph(), files.graph) {
            (true, Some((vem, vex))) => Some(HnswVectorsReader::open(
                vem,
                vex,
                segment_id,
                segment_suffix,
            )?),
            (true, None) => {
                return Err(Error::CorruptMeta(format!(
                    "{} needs its .vem/.vex graph files",
                    format.name()
                )))
            }
            (false, _) => None,
        };
        Ok(QuantizedVectorsReader {
            format,
            flat,
            quantized,
            graph,
        })
    }

    pub fn format(&self) -> QuantizedFormat {
        self.format
    }

    /// The raw vectors: what `getFloatVectorValues`/`getByteVectorValues`
    /// iterate, and what a merge copies.
    pub fn flat(&self) -> &FlatVectorsReader<'a> {
        &self.flat
    }

    pub fn quantized(&self) -> &QuantizedReader<'a> {
        &self.quantized
    }

    /// `getVectorCount(fieldInfo)` (Lucene 10.6, `branch_10_6`, #16647): the
    /// HNSW wrapper's is `Lucene99HnswVectorsReader`'s (the graph entry's
    /// size), the flat one's its quantized reader's.
    ///
    /// # Errors
    /// [`Error::UnknownField`] for a field this reader does not hold.
    pub fn vector_count(&self, field_number: i32) -> Result<i32> {
        match (&self.graph, &self.quantized) {
            (Some(graph), _) => graph.vector_count(field_number),
            (None, QuantizedReader::Scalar(r)) => r.vector_count(field_number, &self.flat),
            (None, QuantizedReader::Binary(r)) => r.vector_count(field_number, &self.flat),
        }
    }

    /// The HNSW reader, for a graph format.
    pub fn graph(&self) -> Option<&HnswVectorsReader<'a>> {
        self.graph.as_ref()
    }

    /// `checkIntegrity`'s quantized half (the raw and graph files were
    /// checked whole at open).
    pub fn check_integrity(&self) -> Result<()> {
        match &self.quantized {
            QuantizedReader::Scalar(r) => r.check_integrity(),
            QuantizedReader::Binary(r) => r.check_integrity(),
        }
    }

    /// `getRandomVectorScorer(field, float[] target)`, which is also what
    /// `getFloatVectorValues(field).scorer(target)` scores with: quantized,
    /// except for a `Lucene99` field with no quantizer (no vectors), which
    /// Java hands to the raw reader.
    pub fn float_scorer(
        &self,
        field_number: i32,
        target: &[f32],
    ) -> Result<QuantizedFloatScorer<'a>> {
        match &self.quantized {
            QuantizedReader::Scalar(r) => match r.scorer(field_number, target)? {
                Some(s) => Ok(QuantizedFloatScorer::Scalar(s)),
                None => Ok(QuantizedFloatScorer::Raw(
                    self.flat
                        .float_vector_values(field_number)?
                        .scorer(target)?,
                )),
            },
            QuantizedReader::Binary(r) => Ok(QuantizedFloatScorer::Binary(
                r.scorer(field_number, target)?,
            )),
        }
    }
}

/// A float query's scorer over a quantized field.
#[derive(Debug, Clone)]
pub enum QuantizedFloatScorer<'a> {
    Raw(FloatVectorScorer<'a>),
    Scalar(scalar_quantized_vectors::Lucene99ScalarQuantizedScorer<'a>),
    Binary(binary_quantized_vectors::Lucene102BinaryQuantizedScorer<'a>),
}

impl VectorScorer for QuantizedFloatScorer<'_> {
    fn score(&mut self, node: i32) -> Result<f32> {
        match self {
            QuantizedFloatScorer::Raw(s) => s.score(node),
            QuantizedFloatScorer::Scalar(s) => s.score(node),
            QuantizedFloatScorer::Binary(s) => s.score(node),
        }
    }

    fn max_ord(&self) -> i32 {
        match self {
            QuantizedFloatScorer::Raw(s) => s.max_ord(),
            QuantizedFloatScorer::Scalar(s) => s.max_ord(),
            QuantizedFloatScorer::Binary(s) => s.max_ord(),
        }
    }
}

/// Fixture access for the quantized formats' unit tests:
/// `fixtures/data/bwc-quantized/<version>/`, one per-field group at a time.
#[cfg(test)]
pub(crate) mod test_support {
    use std::collections::BTreeMap;

    use lucene_store::codec_util::ID_LENGTH;
    use lucene_store::data_input::{DataInput, SliceInput};

    /// One `PerFieldKnnVectorsFormat` group's files, with the segment id and
    /// suffix read back off its own headers.
    pub(crate) struct Group {
        pub files: BTreeMap<String, Vec<u8>>,
        pub id: [u8; ID_LENGTH],
        pub suffix: String,
    }

    impl Group {
        pub fn file(&self, ext: &str) -> &[u8] {
            self.files
                .get(ext)
                .unwrap_or_else(|| panic!("no .{ext} in {}", self.suffix))
        }
    }

    /// `<segment>_<format>_<n>.*` of `version`'s quantized fixture, or `None`
    /// when there is no such group.
    pub(crate) fn fixture_group(
        version: &str,
        segment: &str,
        format: &str,
        n: usize,
    ) -> Option<Group> {
        let dir = format!(
            "{}/../../fixtures/data/bwc-quantized/{version}",
            env!("CARGO_MANIFEST_DIR")
        );
        let suffix = format!("{format}_{n}");
        let prefix = format!("{segment}_{suffix}.");
        let mut files = BTreeMap::new();
        for entry in std::fs::read_dir(&dir).expect("bwc-quantized fixture") {
            let name = entry.unwrap().file_name().to_string_lossy().to_string();
            if let Some(ext) = name.strip_prefix(&prefix) {
                files.insert(
                    ext.to_string(),
                    std::fs::read(format!("{dir}/{name}")).unwrap(),
                );
            }
        }
        let vec = files.get("vec")?;
        let mut input = SliceInput::new(vec);
        input.read_be_u32().unwrap();
        input.read_string().unwrap();
        input.read_be_u32().unwrap();
        let mut id = [0u8; ID_LENGTH];
        input.read_bytes(&mut id).unwrap();
        Some(Group { files, id, suffix })
    }
}

#[cfg(test)]
mod tests;
