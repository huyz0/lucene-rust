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

    /// The HNSW graph reader, for an HNSW wrapper.
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
mod tests {
    use super::test_support::fixture_group;
    use super::*;
    use crate::field_infos::{FieldInfo, VectorEncoding, VectorSimilarityFunction};

    fn infos_for(flat: &FlatVectorsReader<'_>) -> FieldInfos {
        FieldInfos::new(
            flat.fields()
                .iter()
                .map(|e| {
                    let mut fi = FieldInfo::new(format!("f{}", e.field_number), e.field_number);
                    fi.vector_dimension = e.dimension;
                    fi.vector_encoding = e.encoding;
                    fi.vector_similarity_function = e.similarity;
                    fi
                })
                .collect(),
        )
        .unwrap()
    }

    #[test]
    fn formats_are_named_and_described_as_java_does() {
        for f in QuantizedFormat::ALL {
            assert_eq!(QuantizedFormat::for_name(f.name()), Some(f));
        }
        assert_eq!(QuantizedFormat::for_name("Lucene99HnswVectorsFormat"), None);
        assert!(QuantizedFormat::Lucene99HnswScalarQuantized.has_graph());
        assert!(!QuantizedFormat::Lucene102BinaryQuantized.has_graph());
        assert_eq!(
            QuantizedFormat::Lucene99ScalarQuantized.quantized_extensions(),
            ("vemq", "veq")
        );
        assert_eq!(
            QuantizedFormat::Lucene102HnswBinaryQuantized.quantized_extensions(),
            ("vemb", "veb")
        );
        assert_eq!(
            QuantizedFormat::Lucene99ScalarQuantized.search_kind(),
            SearchKind::Nothing
        );
        assert_eq!(
            QuantizedFormat::Lucene102BinaryQuantized.search_kind(),
            SearchKind::ScanAll
        );
        assert_eq!(
            QuantizedFormat::Lucene102HnswBinaryQuantized.search_kind(),
            SearchKind::Hnsw
        );
    }

    #[test]
    fn every_fixture_group_opens_whole() {
        let mut opened = 0;
        for format in QuantizedFormat::ALL {
            for n in 0..10 {
                let Some(g) = fixture_group("10.2.2", "_0", format.name(), n) else {
                    continue;
                };
                let (meta_ext, data_ext) = format.quantized_extensions();
                let files = QuantizedFiles {
                    vemf: g.file("vemf"),
                    vec: g.file("vec"),
                    quantized_meta: g.file(meta_ext),
                    quantized_data: g.file(data_ext),
                    graph: format.has_graph().then(|| (g.file("vem"), g.file("vex"))),
                };
                let flat =
                    FlatVectorsReader::open(files.vemf, files.vec, &g.id, &g.suffix).unwrap();
                let infos = infos_for(&flat);
                let r =
                    QuantizedVectorsReader::open(format, files, &infos, &g.id, &g.suffix).unwrap();
                assert_eq!(r.format(), format);
                // A quantized entry must name a field of the segment.
                let entries = match r.quantized() {
                    QuantizedReader::Scalar(q) => q.fields().len(),
                    QuantizedReader::Binary(q) => q.fields().len(),
                };
                let unknown = QuantizedVectorsReader::open(
                    format,
                    files,
                    &FieldInfos::new(Vec::new()).unwrap(),
                    &g.id,
                    &g.suffix,
                );
                assert_eq!(unknown.is_err(), entries > 0);
                assert_eq!(r.graph().is_some(), format.has_graph());
                r.check_integrity().unwrap();
                for e in r.flat().fields() {
                    if e.encoding != VectorEncoding::Float32 {
                        continue;
                    }
                    let target = vec![0.25f32; e.dimension as usize];
                    let mut s = r.float_scorer(e.field_number, &target).unwrap();
                    assert_eq!(s.max_ord(), e.size);
                    assert!(s.score(0).unwrap().is_finite());
                    match (&r.quantized, &s) {
                        (QuantizedReader::Scalar(_), QuantizedFloatScorer::Scalar(_))
                        | (QuantizedReader::Binary(_), QuantizedFloatScorer::Binary(_)) => {}
                        other => panic!("{format:?}: {:?}", std::mem::discriminant(other.1)),
                    }
                }
                // Without its graph files an HNSW wrapper does not open.
                if format.has_graph() {
                    let mut no_graph = files;
                    no_graph.graph = None;
                    assert!(QuantizedVectorsReader::open(
                        format, no_graph, &infos, &g.id, &g.suffix
                    )
                    .is_err());
                }
                opened += 1;
            }
        }
        assert!(opened >= 8, "{opened} groups");
    }

    #[test]
    fn a_field_without_a_quantizer_scores_on_the_raw_vectors() {
        // A Lucene99 entry with no vectors has no quantizer: Java hands the
        // query to the raw reader. Built from the 9.12.2 fixture's raw pair
        // plus a hand-made `.vemq` whose one entry is empty.
        use lucene_store::data_output::DataOutput;
        let g = fixture_group(
            "9.12.2",
            "_0",
            super::scalar_quantized_vectors::HNSW_NAME,
            0,
        )
        .unwrap();
        let flat =
            FlatVectorsReader::open(g.file("vemf"), g.file("vec"), &g.id, &g.suffix).unwrap();
        let e = flat.fields()[0].clone();
        let mut meta = Vec::new();
        let mut data = Vec::new();
        let (mc, dc) = (
            super::scalar_quantized_vectors::META_CODEC,
            super::scalar_quantized_vectors::DATA_CODEC,
        );
        lucene_store::codec_util::write_index_header(&mut meta, mc, 1, &g.id, &g.suffix);
        lucene_store::codec_util::write_index_header(&mut data, dc, 1, &g.id, &g.suffix);
        meta.write_i32(e.field_number);
        meta.write_i32(1);
        meta.write_i32(match e.similarity {
            VectorSimilarityFunction::Euclidean => 0,
            VectorSimilarityFunction::DotProduct => 1,
            VectorSimilarityFunction::Cosine => 2,
            VectorSimilarityFunction::MaximumInnerProduct => 3,
        });
        meta.write_vlong(0);
        meta.write_vlong(0);
        meta.write_vint(e.dimension);
        meta.write_i32(0);
        crate::vectors::write_stored_meta(&mut meta, &mut data, &[], 10);
        meta.write_i32(-1);
        lucene_store::codec_util::write_footer(&mut meta);
        lucene_store::codec_util::write_footer(&mut data);
        let r = QuantizedVectorsReader::open(
            QuantizedFormat::Lucene99HnswScalarQuantized,
            QuantizedFiles {
                vemf: g.file("vemf"),
                vec: g.file("vec"),
                quantized_meta: &meta,
                quantized_data: &data,
                graph: Some((g.file("vem"), g.file("vex"))),
            },
            &infos_for(&flat),
            &g.id,
            &g.suffix,
        )
        .unwrap();
        let target = vec![0.5f32; e.dimension as usize];
        let mut s = r.float_scorer(e.field_number, &target).unwrap();
        assert!(matches!(s, QuantizedFloatScorer::Raw(_)));
        let mut raw = flat
            .float_vector_values(e.field_number)
            .unwrap()
            .scorer(&target)
            .unwrap();
        assert_eq!(s.max_ord(), raw.max_ord());
        assert_eq!(
            s.score(3).unwrap().to_bits(),
            raw.score(3).unwrap().to_bits()
        );
    }

    #[test]
    fn ordinal_maps_check_their_bounds() {
        use std::sync::Arc;
        let explicit = OrdToDoc::Explicit(Arc::from(vec![2, 5, 9]));
        assert_eq!(ord_to_doc(&explicit, &[], 3, 1).unwrap(), 5);
        assert!(ord_to_doc(&explicit, &[], 4, 3).is_err());
        assert!(ord_to_doc(&explicit, &[], 3, -1).is_err());
        assert!(doc_to_ord(&explicit, &[], 3).is_err());
        assert_eq!(ord_to_doc(&OrdToDoc::Dense, &[], 3, 2).unwrap(), 2);
        assert!(matches!(
            doc_to_ord(&OrdToDoc::Empty, &[], 0).unwrap(),
            DocToOrdCursor::Empty
        ));
        let sparse = OrdToDoc::Sparse {
            docs_with_field_offset: 100,
            docs_with_field_length: 10,
            jump_table_entry_count: -1,
            dense_rank_power: 9,
            addresses_offset: 100,
            addresses_length: 10,
            meta: crate::direct_monotonic::load_meta(
                &mut lucene_store::data_input::SliceInput::new(&[0u8; 64]),
                1,
                16,
            )
            .unwrap(),
        };
        assert!(ord_to_doc(&sparse, &[0u8; 8], 1, 0).is_err());
        assert!(doc_to_ord(&sparse, &[0u8; 8], 1).is_err());
    }
}
