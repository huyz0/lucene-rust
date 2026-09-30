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
        default_knn_vectors: "Lucene90HnswVectorsFormat",
    },
    CodecFormats {
        name: "Lucene91",
        segment_info: SegmentInfoFormat::Lucene90,
        field_infos: FieldInfosFormat::Lucene90,
        default_postings: "Lucene90",
        default_knn_vectors: "Lucene91HnswVectorsFormat",
    },
    CodecFormats {
        name: "Lucene92",
        segment_info: SegmentInfoFormat::Lucene90,
        field_infos: FieldInfosFormat::Lucene90,
        default_postings: "Lucene90",
        // `Lucene92HnswVectorsFormat.NAME` really is lower-case: 9.2 and 9.3
        // name their vector files `_N_lucene92HnswVectorsFormat_0.vec`.
        default_knn_vectors: "lucene92HnswVectorsFormat",
    },
    CodecFormats {
        name: "Lucene94",
        segment_info: SegmentInfoFormat::Lucene90,
        field_infos: FieldInfosFormat::Lucene94,
        default_postings: "Lucene90",
        default_knn_vectors: "Lucene94HnswVectorsFormat",
    },
    CodecFormats {
        name: "Lucene95",
        segment_info: SegmentInfoFormat::Lucene90,
        field_infos: FieldInfosFormat::Lucene94,
        default_postings: "Lucene90",
        default_knn_vectors: "Lucene95HnswVectorsFormat",
    },
    CodecFormats {
        name: "Lucene99",
        segment_info: SegmentInfoFormat::Lucene99,
        field_infos: FieldInfosFormat::Lucene94,
        default_postings: "Lucene99",
        default_knn_vectors: "Lucene99HnswVectorsFormat",
    },
    CodecFormats {
        name: "Lucene912",
        segment_info: SegmentInfoFormat::Lucene99,
        field_infos: FieldInfosFormat::Lucene94,
        default_postings: "Lucene912",
        default_knn_vectors: "Lucene99HnswVectorsFormat",
    },
    CodecFormats {
        name: "Lucene100",
        segment_info: SegmentInfoFormat::Lucene99,
        field_infos: FieldInfosFormat::Lucene94,
        default_postings: "Lucene912",
        default_knn_vectors: "Lucene99HnswVectorsFormat",
    },
    CodecFormats {
        name: "Lucene101",
        segment_info: SegmentInfoFormat::Lucene99,
        field_infos: FieldInfosFormat::Lucene94,
        default_postings: "Lucene101",
        default_knn_vectors: "Lucene99HnswVectorsFormat",
    },
    CodecFormats {
        name: "Lucene103",
        segment_info: SegmentInfoFormat::Lucene99,
        field_infos: FieldInfosFormat::Lucene94,
        default_postings: "Lucene103",
        default_knn_vectors: "Lucene99HnswVectorsFormat",
    },
    CodecFormats {
        name: "Lucene104",
        segment_info: SegmentInfoFormat::Lucene99,
        field_infos: FieldInfosFormat::Lucene94,
        default_postings: "Lucene104",
        default_knn_vectors: "Lucene99HnswVectorsFormat",
    },
];

/// `Codec.forName(name)`, restricted to the codecs this port reads.
pub fn for_name(name: &str) -> Option<&'static CodecFormats> {
    CODECS.iter().find(|c| c.name == name)
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
    fn segment_info_generation_splits_at_lucene99() {
        let si = |n| for_name(n).unwrap().segment_info;
        assert_eq!(si("Lucene95"), SegmentInfoFormat::Lucene90);
        assert_eq!(si("Lucene99"), SegmentInfoFormat::Lucene99);
        let fnm = |n| for_name(n).unwrap().field_infos;
        assert_eq!(fnm("Lucene92"), FieldInfosFormat::Lucene90);
        assert_eq!(fnm("Lucene94"), FieldInfosFormat::Lucene94);
    }
}
