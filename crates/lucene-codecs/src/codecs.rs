//! Codec dispatch: which format generation each component of a named codec
//! reads with -- the port of what the `LuceneXXCodec` classes of
//! `lucene-core` (`Lucene104Codec`) and `lucene-backward-codecs`
//! (`Lucene90Codec` .. `Lucene103Codec`) compose.
//!
//! Lucene resolves a segment's codec by the name `segments_N` records
//! (`Codec.forName`), and each codec hard-wires its segment-info, field-infos,
//! stored-fields, term-vectors, norms, points, live-docs and compound formats.
//! Postings, doc values and vectors are per-field (`PerFieldPostingsFormat`
//! and friends): a field's `FieldInfo` attributes name the format that wrote
//! it, so the codec only supplies the *default* for new writes, which a
//! reader never needs.
//!
//! Most of those hard-wired components share a codec name and a version
//! range across every codec from `Lucene90` to `Lucene104`, and their readers
//! dispatch on the `CodecUtil` header alone: `.fnm` (`Lucene90FieldInfos` vs
//! `Lucene94FieldInfos`), `.kdm` (`Lucene90PointsFormat` versions 0 and 1),
//! the postings files (`Lucene90PostingsWriter*`, `Lucene99PostingsWriter*`,
//! ...). The one component whose header cannot tell its generations apart is
//! `.si`: `Lucene90SegmentInfoFormat` and `Lucene99SegmentInfoFormat` both
//! write `IndexHeader("Lucene90SegmentInfo", 0)`, and only the latter carries
//! a `hasBlocks` byte. That is why [`CodecFormats::segment_info`] exists and
//! is looked up by codec name, as `SegmentInfos.readCommit` does.

use std::sync::RwLock;

/// The `.si` format a codec reads its segment infos with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentInfoFormat {
    /// `backward_codecs.lucene90.Lucene90SegmentInfoFormat`: no `hasBlocks`
    /// byte (`Lucene90`..`Lucene95` codecs).
    Lucene90,
    /// `codecs.lucene99.Lucene99SegmentInfoFormat` (`Lucene99` onwards).
    Lucene99,
}

/// The `.fnm` format a codec reads its field infos with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldInfosFormat {
    /// `backward_codecs.lucene90.Lucene90FieldInfosFormat`
    /// (`Lucene90`..`Lucene92` codecs).
    Lucene90,
    /// `codecs.lucene94.Lucene94FieldInfosFormat` (`Lucene94` onwards).
    Lucene94,
}

/// What one named codec composes, as far as reading is concerned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CodecFormats {
    /// `Codec.getName()`, as `segments_N` records it.
    pub name: &'static str,
    pub segment_info: SegmentInfoFormat,
    pub field_infos: FieldInfosFormat,
    /// The codec's default postings format name
    /// (`PerFieldPostingsFormat.format` of a field written with defaults).
    pub default_postings: &'static str,
    /// The codec's default vectors format name
    /// (`PerFieldKnnVectorsFormat.format`).
    pub default_knn_vectors: &'static str,
    /// Extensions of the files a `FilterCodec`'s own formats add to each
    /// segment (`sit`, `sip`), which a reader opens beside the native ones.
    pub extra_files: &'static [&'static str],
}

/// Every codec Lucene 10.5.0 can read an index with, oldest first: the
/// `lucene-backward-codecs` 9.x/10.x codecs plus the current `Lucene104`.
/// Lucene 8 codecs (`Lucene80`..`Lucene87`) are absent on purpose: neither
/// Lucene 10 nor OpenSearch 3.x opens an index created by Lucene 8.
pub const CODECS: &[CodecFormats] = &[
    CodecFormats {
        name: "Lucene90",
        segment_info: SegmentInfoFormat::Lucene90,
        field_infos: FieldInfosFormat::Lucene90,
        default_postings: "Lucene90",
        extra_files: &[],
        default_knn_vectors: "Lucene90HnswVectorsFormat",
    },
    CodecFormats {
        name: "Lucene91",
        segment_info: SegmentInfoFormat::Lucene90,
        field_infos: FieldInfosFormat::Lucene90,
        default_postings: "Lucene90",
        extra_files: &[],
        default_knn_vectors: "Lucene91HnswVectorsFormat",
    },
    CodecFormats {
        name: "Lucene92",
        segment_info: SegmentInfoFormat::Lucene90,
        field_infos: FieldInfosFormat::Lucene90,
        default_postings: "Lucene90",
        // `Lucene92HnswVectorsFormat.NAME` really is lower-case: 9.2 and 9.3
        // name their vector files `_N_lucene92HnswVectorsFormat_0.vec`.
        extra_files: &[],
        default_knn_vectors: "lucene92HnswVectorsFormat",
    },
    CodecFormats {
        name: "Lucene94",
        segment_info: SegmentInfoFormat::Lucene90,
        field_infos: FieldInfosFormat::Lucene94,
        default_postings: "Lucene90",
        extra_files: &[],
        default_knn_vectors: "Lucene94HnswVectorsFormat",
    },
    CodecFormats {
        name: "Lucene95",
        segment_info: SegmentInfoFormat::Lucene90,
        field_infos: FieldInfosFormat::Lucene94,
        default_postings: "Lucene90",
        extra_files: &[],
        default_knn_vectors: "Lucene95HnswVectorsFormat",
    },
    CodecFormats {
        name: "Lucene99",
        segment_info: SegmentInfoFormat::Lucene99,
        field_infos: FieldInfosFormat::Lucene94,
        default_postings: "Lucene99",
        extra_files: &[],
        default_knn_vectors: "Lucene99HnswVectorsFormat",
    },
    CodecFormats {
        name: "Lucene912",
        segment_info: SegmentInfoFormat::Lucene99,
        field_infos: FieldInfosFormat::Lucene94,
        default_postings: "Lucene912",
        extra_files: &[],
        default_knn_vectors: "Lucene99HnswVectorsFormat",
    },
    CodecFormats {
        name: "Lucene100",
        segment_info: SegmentInfoFormat::Lucene99,
        field_infos: FieldInfosFormat::Lucene94,
        default_postings: "Lucene912",
        extra_files: &[],
        default_knn_vectors: "Lucene99HnswVectorsFormat",
    },
    CodecFormats {
        name: "Lucene101",
        segment_info: SegmentInfoFormat::Lucene99,
        field_infos: FieldInfosFormat::Lucene94,
        default_postings: "Lucene101",
        extra_files: &[],
        default_knn_vectors: "Lucene99HnswVectorsFormat",
    },
    CodecFormats {
        name: "Lucene103",
        segment_info: SegmentInfoFormat::Lucene99,
        field_infos: FieldInfosFormat::Lucene94,
        default_postings: "Lucene103",
        extra_files: &[],
        default_knn_vectors: "Lucene99HnswVectorsFormat",
    },
    CodecFormats {
        name: "Lucene104",
        segment_info: SegmentInfoFormat::Lucene99,
        field_infos: FieldInfosFormat::Lucene94,
        default_postings: "Lucene104",
        extra_files: &[],
        default_knn_vectors: "Lucene99HnswVectorsFormat",
    },
];

/// OpenSearch custom-codecs' zstd codecs, `(name, delegate)`: each is a
/// `FilterCodec` over the named Lucene codec that only replaces its stored
/// fields (with [`crate::stored_fields::Mode::Zstd`] formats, which the
/// stored-fields reader recognises by their own data codec names). Every
/// generation the plugin still registers is here, so an index OpenSearch
/// wrote with `index.codec: zstd` or `zstd_no_dict` opens whichever version
/// wrote it. `Lucene95CustomCodec` is 2.9.0's name for both modes.
pub const OPENSEARCH_ZSTD_CODECS: &[(&str, &str)] = &[
    ("Lucene95CustomCodec", "Lucene95"),
    ("ZSTD", "Lucene95"),
    ("ZSTDNODICT", "Lucene95"),
    ("ZSTD99", "Lucene99"),
    ("ZSTDNODICT99", "Lucene99"),
    ("ZSTD912", "Lucene912"),
    ("ZSTDNODICT912", "Lucene912"),
    ("ZSTD101", "Lucene101"),
    ("ZSTDNODICT101", "Lucene101"),
    ("ZSTD103", "Lucene103"),
    ("ZSTDNODICT103", "Lucene103"),
    (ZSTD_CODEC, "Lucene104"),
    (ZSTD_NO_DICT_CODEC, "Lucene104"),
];

/// The OpenSearch k-NN plugin's codecs, `(name, delegate)`: each is a
/// `FilterCodec` over the named Lucene codec. With the lucene engine, a
/// vector field's files are Lucene's own formats (some under k-NN format
/// names, which name the same files), so an index the k-NN plugin wrote
/// opens whichever version wrote it.
pub const OPENSEARCH_KNN_CODECS: &[(&str, &str)] = &[
    ("KNN910Codec", "Lucene91"),
    ("KNN920Codec", "Lucene92"),
    ("KNN940Codec", "Lucene94"),
    ("KNN950Codec", "Lucene95"),
    ("KNN990Codec", "Lucene99"),
    ("KNN9120Codec", "Lucene912"),
    ("KNN10010Codec", "Lucene101"),
    ("KNN1030Codec", "Lucene103"),
    ("KNN1040Codec", "Lucene104"),
];

/// The codec OpenSearch 3.x writes for `index.codec: zstd`.
pub const ZSTD_CODEC: &str = "ZSTD104";
/// The codec OpenSearch 3.x writes for `index.codec: zstd_no_dict`.
pub const ZSTD_NO_DICT_CODEC: &str = "ZSTDNODICT104";

/// [`OPENSEARCH_ZSTD_CODECS`] and [`OPENSEARCH_KNN_CODECS`] as readable codecs.
static OPENSEARCH: std::sync::LazyLock<Vec<CodecFormats>> = std::sync::LazyLock::new(|| {
    OPENSEARCH_ZSTD_CODECS
        .iter()
        .chain(OPENSEARCH_KNN_CODECS)
        .filter_map(|&(name, delegate)| {
            CODECS
                .iter()
                .find(|c| c.name == delegate)
                .map(|c| renamed(c, name))
        })
        .collect()
});

static REGISTERED: RwLock<Vec<&'static CodecFormats>> = RwLock::new(Vec::new());

/// `Codec`'s SPI: makes a codec defined outside this crate (the test
/// framework's `Asserting`, `CheapBastard`, ...) readable by its name. Such a
/// codec composes formats this port reads already; `formats` says which
/// generation, as the built-in entries do. A later registration under a name
/// replaces the earlier one; a built-in name, or one of
/// [`OPENSEARCH_ZSTD_CODECS`] or [`OPENSEARCH_KNN_CODECS`], cannot be replaced.
pub fn register(formats: CodecFormats) {
    if CODECS
        .iter()
        .chain(OPENSEARCH.iter())
        .any(|c| c.name == formats.name)
    {
        return;
    }
    let mut registered = REGISTERED.write().unwrap_or_else(|e| e.into_inner());
    if registered.iter().any(|c| **c == formats) {
        return;
    }
    registered.retain(|c| c.name != formats.name);
    registered.push(Box::leak(Box::new(formats)));
}

/// `formats` under another codec name: what a `FilterCodec` that keeps its
/// delegate's file formats reads with.
pub const fn renamed(formats: &CodecFormats, name: &'static str) -> CodecFormats {
    CodecFormats { name, ..*formats }
}

/// `Codec.forName(name)`, restricted to the codecs this port reads: the
/// built-in ones and those [`register`]ed.
pub fn for_name(name: &str) -> Option<&'static CodecFormats> {
    CODECS
        .iter()
        .chain(OPENSEARCH.iter())
        .find(|c| c.name == name)
        .or_else(|| {
            let registered = REGISTERED.read().unwrap_or_else(|e| e.into_inner());
            registered.iter().copied().find(|c| c.name == name)
        })
}

/// The current codec, `Lucene104`.
pub fn lucene104() -> &'static CodecFormats {
    for_name("Lucene104").expect("the current codec")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_codec_resolves_by_its_own_name() {
        for c in CODECS {
            assert_eq!(for_name(c.name), Some(c));
        }
        assert_eq!(for_name("Lucene87"), None);
        assert_eq!(for_name("lucene104"), None);
    }

    #[test]
    fn a_registered_codec_resolves_and_cannot_shadow_a_builtin() {
        assert_eq!(for_name("RenamedForTest"), None);
        register(renamed(lucene104(), "RenamedForTest"));
        let c = for_name("RenamedForTest").unwrap();
        assert_eq!(c.segment_info, SegmentInfoFormat::Lucene99);
        assert_eq!(c.default_postings, "Lucene104");
        register(renamed(for_name("Lucene95").unwrap(), "Lucene104"));
        assert_eq!(lucene104().segment_info, SegmentInfoFormat::Lucene99);
        register(renamed(for_name("Lucene95").unwrap(), "RenamedForTest"));
        assert_eq!(
            for_name("RenamedForTest").unwrap().segment_info,
            SegmentInfoFormat::Lucene90
        );
    }

    #[test]
    fn opensearch_codecs_read_as_their_delegates() {
        assert_eq!(
            OPENSEARCH.len(),
            OPENSEARCH_ZSTD_CODECS.len() + OPENSEARCH_KNN_CODECS.len()
        );
        for &(name, delegate) in OPENSEARCH_ZSTD_CODECS.iter().chain(OPENSEARCH_KNN_CODECS) {
            let c = for_name(name).unwrap();
            assert_eq!(renamed(c, delegate), *for_name(delegate).unwrap(), "{name}");
        }
        register(renamed(for_name("Lucene95").unwrap(), ZSTD_CODEC));
        assert_eq!(
            for_name(ZSTD_CODEC).unwrap().segment_info,
            SegmentInfoFormat::Lucene99
        );
    }

    #[test]
    fn segment_info_generation_splits_at_lucene99() {
        let si = |n| for_name(n).unwrap().segment_info;
        assert_eq!(si("Lucene95"), SegmentInfoFormat::Lucene90);
        assert_eq!(si("Lucene99"), SegmentInfoFormat::Lucene99);
        let fnm = |n| for_name(n).unwrap().field_infos;
        assert_eq!(fnm("Lucene92"), FieldInfosFormat::Lucene90);
        assert_eq!(fnm("Lucene94"), FieldInfosFormat::Lucene94);
    }
}
