//! `SimpleTextCodec` and `SimpleTextKnnVectorsFormat`: the codec named
//! `SimpleText` and the formats it composes.
//!
//! What differs from Java: this crate has no `Codec` abstraction a codec
//! plugs into -- [`crate::codec::Lucene104Codec`] is only per-field routing
//! and [`crate::codecs`] reads binary formats by codec name -- so
//! [`SimpleTextCodec`] is a thin aggregator. Its accessors name each
//! format (its SPI name where Java's has one, the extensions of the files it
//! writes); the readers and writers are this module's siblings, which take
//! and return bytes. `IndexWriter`/`DirectoryReader` do not route through
//! it. [`SimpleTextKnnVectorsFormat`] hands out the vectors writer and
//! reader.

use super::knn_vectors_reader::SimpleTextKnnVectorsReader;
use super::knn_vectors_writer::{SimpleTextKnnVectorsWriter, META_EXTENSION, VECTOR_EXTENSION};
use super::{
    compound, doc_values_writer, field_infos, fields_writer, live_docs, norms, points_writer,
    segment_info, stored_fields_writer, term_vectors_writer, Result,
};
use crate::field_infos::FieldInfos;

/// `Codec.getName()` of `SimpleTextCodec`.
pub const CODEC_NAME: &str = "SimpleText";

/// `KnnVectorsFormat.DEFAULT_MAX_DIMENSIONS`.
pub const DEFAULT_MAX_DIMENSIONS: i32 = 1024;

/// One of `SimpleTextCodec`'s formats, as far as this crate models one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SimpleTextFormat {
    /// The SPI name of a per-field format (postings, doc values, vectors).
    pub name: Option<&'static str>,
    /// The extensions of the files it writes.
    pub extensions: &'static [&'static str],
}

/// `org.apache.lucene.codecs.simpletext.SimpleTextKnnVectorsFormat`.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SimpleTextKnnVectorsFormat;

impl SimpleTextKnnVectorsFormat {
    /// `getName()`.
    pub const NAME: &'static str = "SimpleTextKnnVectorsFormat";

    /// This format as a [`SimpleTextFormat`].
    pub const FORMAT: SimpleTextFormat = SimpleTextFormat {
        name: Some(Self::NAME),
        extensions: &[VECTOR_EXTENSION, META_EXTENSION],
    };

    /// `fieldsWriter(state)`.
    pub fn fields_writer(&self) -> SimpleTextKnnVectorsWriter {
        SimpleTextKnnVectorsWriter::new()
    }

    /// `fieldsReader(state)` over the `.gri` and `.vec` bytes.
    ///
    /// # Errors
    /// What [`SimpleTextKnnVectorsReader::open`] fails with.
    pub fn fields_reader<'a>(
        &self,
        meta: &[u8],
        vector_data: &'a [u8],
        field_infos: &'a FieldInfos,
    ) -> Result<SimpleTextKnnVectorsReader<'a>> {
        SimpleTextKnnVectorsReader::open(meta, vector_data, field_infos)
    }

    /// `getMaxDimensions(fieldName)`.
    pub fn max_dimensions(&self, _field_name: &str) -> i32 {
        DEFAULT_MAX_DIMENSIONS
    }
}

/// `org.apache.lucene.codecs.simpletext.SimpleTextCodec`.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SimpleTextCodec;

/// A format with no SPI name.
const fn unnamed(extensions: &'static [&'static str]) -> SimpleTextFormat {
    SimpleTextFormat {
        name: None,
        extensions,
    }
}

impl SimpleTextCodec {
    /// `getName()`.
    pub fn name(&self) -> &'static str {
        CODEC_NAME
    }

    /// `postingsFormat()`: `SimpleTextPostingsFormat` ([`fields_writer`]).
    pub fn postings_format(&self) -> SimpleTextFormat {
        SimpleTextFormat {
            name: Some(fields_writer::FORMAT_NAME),
            extensions: &[fields_writer::POSTINGS_EXTENSION],
        }
    }

    /// `storedFieldsFormat()` ([`stored_fields_writer`]).
    pub fn stored_fields_format(&self) -> SimpleTextFormat {
        unnamed(&[stored_fields_writer::FIELDS_EXTENSION])
    }

    /// `termVectorsFormat()` ([`term_vectors_writer`]).
    pub fn term_vectors_format(&self) -> SimpleTextFormat {
        unnamed(&[term_vectors_writer::VECTORS_EXTENSION])
    }

    /// `fieldInfosFormat()` ([`field_infos`]).
    pub fn field_infos_format(&self) -> SimpleTextFormat {
        unnamed(&[field_infos::FIELD_INFOS_EXTENSION])
    }

    /// `segmentInfoFormat()` ([`segment_info`]).
    pub fn segment_info_format(&self) -> SimpleTextFormat {
        unnamed(&[segment_info::SI_EXTENSION])
    }

    /// `normsFormat()` ([`norms`]).
    pub fn norms_format(&self) -> SimpleTextFormat {
        unnamed(&[norms::NORMS_SEG_EXTENSION])
    }

    /// `liveDocsFormat()` ([`live_docs`]).
    pub fn live_docs_format(&self) -> SimpleTextFormat {
        unnamed(&[live_docs::LIVEDOCS_EXTENSION])
    }

    /// `docValuesFormat()`: `SimpleTextDocValuesFormat`, SPI name
    /// `SimpleText` ([`doc_values_writer`]).
    pub fn doc_values_format(&self) -> SimpleTextFormat {
        SimpleTextFormat {
            name: Some(CODEC_NAME),
            extensions: &[doc_values_writer::DOC_VALUES_EXTENSION],
        }
    }

    /// `compoundFormat()` ([`compound`]).
    pub fn compound_format(&self) -> SimpleTextFormat {
        unnamed(&[compound::DATA_EXTENSION])
    }

    /// `pointsFormat()` ([`points_writer`]).
    pub fn points_format(&self) -> SimpleTextFormat {
        unnamed(&[
            points_writer::POINT_EXTENSION,
            points_writer::POINT_INDEX_EXTENSION,
        ])
    }

    /// `knnVectorsFormat()`.
    pub fn knn_vectors_format(&self) -> SimpleTextKnnVectorsFormat {
        SimpleTextKnnVectorsFormat
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field_infos::{FieldInfo, VectorEncoding, VectorSimilarityFunction};

    #[test]
    fn names_every_format() {
        let c = SimpleTextCodec;
        assert_eq!(c.name(), "SimpleText");
        let ext = |f: SimpleTextFormat| f.extensions.join(",");
        assert_eq!(c.postings_format().name, Some("SimpleText"));
        assert_eq!(ext(c.postings_format()), "pst");
        assert_eq!(c.doc_values_format().name, Some("SimpleText"));
        assert_eq!(ext(c.doc_values_format()), "dat");
        assert_eq!(ext(c.stored_fields_format()), "fld");
        assert_eq!(ext(c.term_vectors_format()), "tvc");
        assert_eq!(ext(c.field_infos_format()), "inf");
        assert_eq!(ext(c.segment_info_format()), "si");
        assert_eq!(ext(c.norms_format()), "len");
        assert_eq!(ext(c.live_docs_format()), "liv");
        assert_eq!(ext(c.compound_format()), "scf");
        assert_eq!(ext(c.points_format()), "dim,dii");
        assert_eq!(c.points_format().name, None);
        assert_eq!(
            SimpleTextKnnVectorsFormat::FORMAT,
            SimpleTextFormat {
                name: Some("SimpleTextKnnVectorsFormat"),
                extensions: &["vec", "gri"],
            }
        );
        assert_eq!(c.knn_vectors_format().max_dimensions("f"), 1024);
    }

    #[test]
    fn the_vectors_format_writes_what_it_reads() {
        let infos = FieldInfos::new(vec![FieldInfo::new("v", 0).with_vectors(
            1,
            VectorEncoding::Float32,
            VectorSimilarityFunction::Euclidean,
        )])
        .unwrap();
        let format = SimpleTextCodec.knn_vectors_format();
        let mut w = format.fields_writer();
        w.write_float_field(&infos.fields[0], [(3, &[2.5f32][..])])
            .unwrap();
        let (vec, gri) = w.finish();
        let r = format.fields_reader(&gri, &vec, &infos).unwrap();
        let v = r.float_vector_values("v").unwrap().unwrap();
        assert_eq!(
            (v.vector_value(0), v.ord_to_doc(0)),
            (Some(&[2.5f32][..]), Some(3))
        );
    }
}
