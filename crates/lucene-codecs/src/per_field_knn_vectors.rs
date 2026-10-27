//! The write half of `org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat`
//! for the KNN vector formats this port writes ([`KnnVectorsFormat`]):
//! `Lucene99HnswVectorsFormat(maxConn, beamWidth)`,
//! `Lucene104HnswScalarQuantizedVectorsFormat(encoding, maxConn, beamWidth)`
//! and the graph-less `Lucene104ScalarQuantizedVectorsFormat(encoding)`.
//!
//! `FieldsWriter.getInstance`: each field goes to the format its caller
//! routes it to (`getKnnVectorsFormatForField`); the first field of a format
//! instance opens that instance's writer under the next suffix number of the
//! format's *name*, so two HNSW instances with different graph parameters
//! are `Lucene99HnswVectorsFormat_0` and `_1`, and a quantized instance
//! beside them `Lucene104HnswBinaryQuantizedVectorsFormat_0` ([`group_fields`]).
//! Each field records its instance in its `PerFieldKnnVectorsFormat.format`/
//! `.suffix` attributes; each instance's files carry the codec suffix
//! `<name>_<n>` ([`Group::codec_suffix`]) in their names and headers.
//!
//! The order the writer first reaches a field in is the caller's: the order
//! a flush's documents first carry each vector field (`IndexingChain` calls
//! `addField` the first time it sees one), the merged `FieldInfos`
//! (field-number order) at a merge.
//!
//! What this does not model: Java keys instances by identity; a format here
//! is its settings, and equal settings share one group.

use lucene_util::quantization::ScalarEncoding;

use crate::hnsw;
use crate::scalar_quantized_vectors;

/// `PerFieldKnnVectorsFormat.PER_FIELD_FORMAT_KEY`.
pub const PER_FIELD_FORMAT_KEY: &str = "PerFieldKnnVectorsFormat.format";
/// `PerFieldKnnVectorsFormat.PER_FIELD_SUFFIX_KEY`.
pub const PER_FIELD_SUFFIX_KEY: &str = "PerFieldKnnVectorsFormat.suffix";
/// `Lucene99HnswVectorsFormat`'s SPI name.
pub const HNSW_NAME: &str = "Lucene99HnswVectorsFormat";

/// A KNN vectors format a field can be routed to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KnnVectorsFormat {
    /// `Lucene99HnswVectorsFormat(maxConn, beamWidth)`: raw vectors
    /// (`.vec`/`.vemf`) and an HNSW graph (`.vem`/`.vex`).
    Hnsw { max_conn: i32, beam_width: i32 },
    /// `Lucene104HnswScalarQuantizedVectorsFormat(encoding, maxConn,
    /// beamWidth)`: the raw vectors, their scalar-quantized codes
    /// (`.veq`/`.vemq`, `FLOAT32` fields only) and a graph.
    HnswScalarQuantized {
        encoding: ScalarEncoding,
        max_conn: i32,
        beam_width: i32,
    },
    /// `Lucene104ScalarQuantizedVectorsFormat(encoding)`: the raw vectors and
    /// their codes, no graph (every search is exhaustive).
    ScalarQuantized { encoding: ScalarEncoding },
    /// A format that wraps `format` without changing its files
    /// (`AssertingKnnVectorsFormat`'s `Asserting`): fields record `name` and
    /// the instance's files carry it (`_0_Asserting_0.vec`). Readers find
    /// the files by their extensions, whatever the name.
    Named {
        name: &'static str,
        format: &'static KnnVectorsFormat,
    },
}

impl Default for KnnVectorsFormat {
    /// `Lucene99HnswVectorsFormat()`.
    fn default() -> Self {
        KnnVectorsFormat::Hnsw {
            max_conn: hnsw::DEFAULT_MAX_CONN,
            beam_width: hnsw::DEFAULT_BEAM_WIDTH,
        }
    }
}

/// Java's `IllegalArgumentException` from a format constructor.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[error("{0}")]
pub struct InvalidFormat(pub String);

fn check_graph(max_conn: i32, beam_width: i32) -> Result<(), InvalidFormat> {
    if max_conn <= 0 || max_conn > hnsw::MAXIMUM_MAX_CONN {
        return Err(InvalidFormat(format!(
            "maxConn must be positive and less than or equal to {}; maxConn={max_conn}",
            hnsw::MAXIMUM_MAX_CONN
        )));
    }
    if beam_width <= 0 || beam_width > hnsw::MAXIMUM_BEAM_WIDTH {
        return Err(InvalidFormat(format!(
            "beamWidth must be positive and less than or equal to {}; beamWidth={beam_width}",
            hnsw::MAXIMUM_BEAM_WIDTH
        )));
    }
    Ok(())
}

impl KnnVectorsFormat {
    /// `new Lucene99HnswVectorsFormat(maxConn, beamWidth)`.
    pub fn hnsw(max_conn: i32, beam_width: i32) -> Result<Self, InvalidFormat> {
        check_graph(max_conn, beam_width)?;
        Ok(KnnVectorsFormat::Hnsw {
            max_conn,
            beam_width,
        })
    }

    /// `new Lucene104HnswScalarQuantizedVectorsFormat(encoding, maxConn,
    /// beamWidth)`.
    pub fn hnsw_scalar_quantized(
        encoding: ScalarEncoding,
        max_conn: i32,
        beam_width: i32,
    ) -> Result<Self, InvalidFormat> {
        check_graph(max_conn, beam_width)?;
        Ok(KnnVectorsFormat::HnswScalarQuantized {
            encoding,
            max_conn,
            beam_width,
        })
    }

    /// `new Lucene104ScalarQuantizedVectorsFormat(encoding)`.
    pub fn scalar_quantized(encoding: ScalarEncoding) -> Self {
        KnnVectorsFormat::ScalarQuantized { encoding }
    }

    /// `format`'s files under a wrapping format's `name`
    /// ([`KnnVectorsFormat::Named`]).
    pub const fn named(name: &'static str, format: &'static KnnVectorsFormat) -> Self {
        KnnVectorsFormat::Named { name, format }
    }

    /// The format whose files this one writes: itself, or a
    /// [`KnnVectorsFormat::Named`] wrapper's delegate.
    pub fn files_format(&self) -> &KnnVectorsFormat {
        match self {
            KnnVectorsFormat::Named { format, .. } => format.files_format(),
            other => other,
        }
    }

    /// `KnnVectorsFormat.getName()`: what `PerFieldKnnVectorsFormat.format`
    /// records and the suffix numbering is per.
    pub fn name(&self) -> &'static str {
        match self {
            KnnVectorsFormat::Named { name, .. } => name,
            KnnVectorsFormat::Hnsw { .. } => HNSW_NAME,
            KnnVectorsFormat::HnswScalarQuantized { .. } => scalar_quantized_vectors::HNSW_NAME,
            KnnVectorsFormat::ScalarQuantized { .. } => scalar_quantized_vectors::NAME,
        }
    }

    /// The graph parameters `(maxConn, beamWidth)`, `None` for a format
    /// without a graph.
    pub fn graph(&self) -> Option<(i32, i32)> {
        match *self.files_format() {
            KnnVectorsFormat::Hnsw {
                max_conn,
                beam_width,
            }
            | KnnVectorsFormat::HnswScalarQuantized {
                max_conn,
                beam_width,
                ..
            } => Some((max_conn, beam_width)),
            KnnVectorsFormat::ScalarQuantized { .. } | KnnVectorsFormat::Named { .. } => None,
        }
    }

    /// The quantization, `None` for plain HNSW.
    pub fn quantization(&self) -> Option<ScalarEncoding> {
        match *self.files_format() {
            KnnVectorsFormat::Hnsw { .. } | KnnVectorsFormat::Named { .. } => None,
            KnnVectorsFormat::HnswScalarQuantized { encoding, .. }
            | KnnVectorsFormat::ScalarQuantized { encoding } => Some(encoding),
        }
    }
}

static WRAPPERS: std::sync::RwLock<Vec<KnnVectorsFormat>> = std::sync::RwLock::new(Vec::new());

/// `KnnVectorsFormat`'s SPI for a [`KnnVectorsFormat::Named`] wrapper: makes
/// a segment's `<name>_<n>` instances readable as their delegate's files. A
/// later registration under a name replaces the earlier one; a format that
/// is not a wrapper is ignored (the built-in formats need none).
pub fn register_wrapper(format: KnnVectorsFormat) {
    if !matches!(format, KnnVectorsFormat::Named { .. }) {
        return;
    }
    let mut wrappers = WRAPPERS.write().unwrap_or_else(|e| e.into_inner());
    wrappers.retain(|w| w.name() != format.name());
    wrappers.push(format);
}

/// `KnnVectorsFormat.forName(name)` among this release's formats: the three
/// built-in ones (a default instance of each; a reader takes the settings
/// from the files) and every registered wrapper ([`register_wrapper`]).
pub fn for_name(name: &str) -> Option<KnnVectorsFormat> {
    let builtin = [
        KnnVectorsFormat::default(),
        KnnVectorsFormat::HnswScalarQuantized {
            encoding: ScalarEncoding::UnsignedByte,
            max_conn: hnsw::DEFAULT_MAX_CONN,
            beam_width: hnsw::DEFAULT_BEAM_WIDTH,
        },
        KnnVectorsFormat::ScalarQuantized {
            encoding: ScalarEncoding::UnsignedByte,
        },
    ];
    if let Some(format) = builtin.into_iter().find(|f| f.name() == name) {
        return Some(format);
    }
    let wrappers = WRAPPERS.read().unwrap_or_else(|e| e.into_inner());
    wrappers.iter().copied().find(|w| w.name() == name)
}

/// One format instance's share of a segment: its format, its suffix number
/// among the instances of the same name, and its fields' numbers in the
/// order the writer reached them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Group {
    pub format: KnnVectorsFormat,
    pub suffix: u32,
    pub field_numbers: Vec<i32>,
    /// The codec's `knnVectorsFormat()` itself, not a per-field instance
    /// ([`direct_group`]): no codec suffix, no per-field attributes.
    pub direct: bool,
}

impl Group {
    /// `PerFieldKnnVectorsFormat.getSuffix(formatName, suffix)`: the codec
    /// suffix this instance's files and headers carry; none for a
    /// [`direct`](Self::direct) group.
    pub fn codec_suffix(&self) -> String {
        if self.direct {
            String::new()
        } else {
            format!("{}_{}", self.format.name(), self.suffix)
        }
    }
}

/// A codec whose `knnVectorsFormat()` is `format` itself: every one of
/// `fields` (`(number, name)`) in one unsuffixed group, none when there are
/// no fields.
pub fn direct_group(fields: &[(i32, &str)], format: KnnVectorsFormat) -> Vec<Group> {
    if fields.is_empty() {
        return Vec::new();
    }
    vec![Group {
        format,
        suffix: 0,
        field_numbers: fields.iter().map(|&(number, _)| number).collect(),
        direct: true,
    }]
}

/// `FieldsWriter.getInstance` over `fields` (`(number, name)`) in the order
/// the writer reaches them: each field joins the group of `format_for(name)`,
/// a new format opening the next group, whose suffix is the number of groups
/// of the same format name opened before it.
pub fn group_fields(
    fields: &[(i32, &str)],
    format_for: &dyn Fn(&str) -> KnnVectorsFormat,
) -> Vec<Group> {
    let mut groups: Vec<Group> = Vec::new();
    for &(number, name) in fields {
        let format = format_for(name);
        match groups.iter_mut().find(|g| g.format == format) {
            Some(group) => group.field_numbers.push(number),
            None => {
                let same_name = groups
                    .iter()
                    .filter(|g| g.format.name() == format.name())
                    .count();
                groups.push(Group {
                    format,
                    suffix: u32::try_from(same_name).unwrap_or(u32::MAX),
                    field_numbers: vec![number],
                    direct: false,
                });
            }
        }
    }
    groups
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suffixes_are_numbered_per_format_name() {
        let small = KnnVectorsFormat::hnsw(8, 50).unwrap();
        let sq =
            KnnVectorsFormat::hnsw_scalar_quantized(ScalarEncoding::UnsignedByte, 16, 100).unwrap();
        let route = |name: &str| match name {
            "a" | "d" => KnnVectorsFormat::default(),
            "b" => sq,
            _ => small,
        };
        let groups = group_fields(&[(0, "b"), (1, "a"), (2, "c"), (3, "d")], &route);
        let got: Vec<(String, Vec<i32>)> = groups
            .iter()
            .map(|g| (g.codec_suffix(), g.field_numbers.clone()))
            .collect();
        assert_eq!(
            got,
            vec![
                (
                    "Lucene104HnswBinaryQuantizedVectorsFormat_0".to_string(),
                    vec![0]
                ),
                ("Lucene99HnswVectorsFormat_0".to_string(), vec![1, 3]),
                ("Lucene99HnswVectorsFormat_1".to_string(), vec![2]),
            ]
        );
        assert!(group_fields(&[], &route).is_empty());
        let direct = direct_group(&[(0, "b"), (2, "c")], small);
        assert_eq!(direct.len(), 1);
        assert_eq!(direct[0].codec_suffix(), "");
        assert_eq!(direct[0].field_numbers, vec![0, 2]);
        assert!(direct_group(&[], small).is_empty());
    }

    #[test]
    fn formats_validate_and_describe_themselves_as_java_does() {
        assert!(KnnVectorsFormat::hnsw(0, 100).is_err());
        assert!(KnnVectorsFormat::hnsw(513, 100).is_err());
        assert!(KnnVectorsFormat::hnsw(16, 0).is_err());
        assert!(KnnVectorsFormat::hnsw(16, 3201).is_err());
        assert!(
            KnnVectorsFormat::hnsw_scalar_quantized(ScalarEncoding::PackedNibble, 0, 1).is_err()
        );
        let d = KnnVectorsFormat::default();
        assert_eq!(d.graph(), Some((16, 100)));
        assert_eq!(d.quantization(), None);
        assert_eq!(d.name(), "Lucene99HnswVectorsFormat");
        let flat = KnnVectorsFormat::scalar_quantized(ScalarEncoding::SevenBit);
        assert_eq!(flat.graph(), None);
        assert_eq!(flat.quantization(), Some(ScalarEncoding::SevenBit));
        assert_eq!(flat.name(), "Lucene104ScalarQuantizedVectorsFormat");
        let sq =
            KnnVectorsFormat::hnsw_scalar_quantized(ScalarEncoding::UnsignedByte, 32, 64).unwrap();
        assert_eq!(sq.graph(), Some((32, 64)));
        assert_eq!(sq.quantization(), Some(ScalarEncoding::UnsignedByte));
    }
}
