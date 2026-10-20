//! `SimpleTextKnnVectorsWriter` (and `SimpleTextKnnVectorsFormat`'s
//! extensions): every vector field of a segment as one line per vector in
//! `.vec`, and per field its slice of `.vec` and its doc ids in `.gri`.
//!
//! ```text
//! .vec, per field in write order, one line per vector in ord order:
//!   [<Float.toString>, <Float.toString>, ...]   (Arrays.toString(float[]))
//!   [<byte>, <byte>, ...]                       (Arrays.toString(byte[]))
//! checksum <CRC32>
//!
//! .gri, per field:
//!   field-number <n>
//!   field-name <name>
//!   vector-data-offset <offset of its first line in .vec>
//!   vector-data-length <bytes of its lines>
//!   vector-dimension <d>
//!   size <count>
//!   <docID>                                     (count lines, ord order)
//! field-number -1
//! checksum <CRC32>
//! ```
//!
//! What differs from Java: the writer is handed each field's `(docID,
//! vector)`s in ord order -- what Java's `BufferingKnnVectorsWriter` hands
//! `writeField` (doc order on a flush, remapped by an index sort; doc order
//! of the merged segment on a merge) -- and returns both files' bytes. Java's
//! `assert value.length == dimension` is an `IllegalArgument` error, as are
//! values of the other encoding (Java's `addField` picks the kind by it).

use lucene_util::geo::java_float_string;

use super::util::{write, write_checksum, write_newline};
use super::{segment_file_name, Error, Result};
use crate::field_infos::{FieldInfo, VectorEncoding};

/// `SimpleTextKnnVectorsFormat.VECTOR_EXTENSION`.
pub const VECTOR_EXTENSION: &str = "vec";
/// `SimpleTextKnnVectorsFormat.META_EXTENSION`.
pub const META_EXTENSION: &str = "gri";

/// `SimpleTextKnnVectorsWriter.FIELD_NUMBER`.
pub const FIELD_NUMBER: &[u8] = b"field-number ";
/// `SimpleTextKnnVectorsWriter.FIELD_NAME`.
pub const FIELD_NAME: &[u8] = b"field-name ";
/// `SimpleTextKnnVectorsWriter.VECTOR_DATA_OFFSET`.
pub const VECTOR_DATA_OFFSET: &[u8] = b"vector-data-offset ";
/// `SimpleTextKnnVectorsWriter.VECTOR_DATA_LENGTH`.
pub const VECTOR_DATA_LENGTH: &[u8] = b"vector-data-length ";
/// `SimpleTextKnnVectorsWriter.VECTOR_DIMENSION`.
pub const VECTOR_DIMENSION: &[u8] = b"vector-dimension ";
/// `SimpleTextKnnVectorsWriter.SIZE`.
pub const SIZE: &[u8] = b"size ";

/// The `.vec` file's name: `IndexFileNames.segmentFileName(segment, suffix,
/// "vec")`.
pub fn vector_data_file_name(segment: &str, suffix: &str) -> String {
    segment_file_name(segment, suffix, VECTOR_EXTENSION)
}

/// The `.gri` file's name.
pub fn meta_file_name(segment: &str, suffix: &str) -> String {
    segment_file_name(segment, suffix, META_EXTENSION)
}

/// `org.apache.lucene.codecs.simpletext.SimpleTextKnnVectorsWriter`.
#[derive(Debug, Default)]
pub struct SimpleTextKnnVectorsWriter {
    meta: Vec<u8>,
    vector_data: Vec<u8>,
}

impl SimpleTextKnnVectorsWriter {
    /// `new SimpleTextKnnVectorsWriter(state)`: both files empty.
    pub fn new() -> Self {
        Self::default()
    }

    /// `writeField(fieldInfo, FloatVectorValues, maxDoc)`: `vectors` are the
    /// field's `(docID, vector)`s in ord order.
    ///
    /// # Errors
    /// `IllegalArgument` for a vector whose length is not the field's
    /// dimension (Java's `assert`).
    pub fn write_float_field<'v>(
        &mut self,
        field: &FieldInfo,
        vectors: impl IntoIterator<Item = (i32, &'v [f32])>,
    ) -> Result<()> {
        // Java: SimpleTextKnnVectorsWriter.writeField(FieldInfo, FloatVectorValues, int)
        check_encoding(field, VectorEncoding::Float32)?;
        let vector_data_offset = self.vector_data.len();
        let mut doc_ids = Vec::new();
        for (doc_id, value) in vectors {
            check_dimension(field, value.len())?;
            // Java: writeFloatVectorValue -- Arrays.toString(value)
            let parts: Vec<String> = value.iter().map(|&v| java_float_string(v)).collect();
            self.write_array(&parts);
            doc_ids.push(doc_id);
        }
        self.write_meta(field, vector_data_offset, &doc_ids);
        Ok(())
    }

    /// `writeField(fieldInfo, ByteVectorValues, maxDoc)`: `vectors` are the
    /// field's `(docID, vector)`s in ord order, each byte a Java `byte`.
    ///
    /// # Errors
    /// As [`Self::write_float_field`].
    pub fn write_byte_field<'v>(
        &mut self,
        field: &FieldInfo,
        vectors: impl IntoIterator<Item = (i32, &'v [u8])>,
    ) -> Result<()> {
        // Java: SimpleTextKnnVectorsWriter.writeField(FieldInfo, ByteVectorValues, int)
        check_encoding(field, VectorEncoding::Byte)?;
        let vector_data_offset = self.vector_data.len();
        let mut doc_ids = Vec::new();
        for (doc_id, value) in vectors {
            check_dimension(field, value.len())?;
            // Java: writeByteVectorValue -- Arrays.toString(value)
            let parts: Vec<String> = value.iter().map(|&b| (b as i8).to_string()).collect();
            self.write_array(&parts);
            doc_ids.push(doc_id);
        }
        self.write_meta(field, vector_data_offset, &doc_ids);
        Ok(())
    }

    /// `Arrays.toString(...)` then a newline, into `.vec`.
    fn write_array(&mut self, parts: &[String]) {
        write(&mut self.vector_data, b"[");
        write(&mut self.vector_data, parts.join(", ").as_bytes());
        write(&mut self.vector_data, b"]");
        write_newline(&mut self.vector_data);
    }

    /// `writeMeta(field, vectorDataOffset, vectorDataLength, docIds)`.
    fn write_meta(&mut self, field: &FieldInfo, vector_data_offset: usize, doc_ids: &[i32]) {
        // The buffer only grows, so the subtraction never saturates.
        let vector_data_length = self.vector_data.len().saturating_sub(vector_data_offset);
        let meta = &mut self.meta;
        write_field(meta, FIELD_NUMBER, &field.number.to_string());
        write_field(meta, FIELD_NAME, &field.name);
        write_field(meta, VECTOR_DATA_OFFSET, &vector_data_offset.to_string());
        write_field(meta, VECTOR_DATA_LENGTH, &vector_data_length.to_string());
        write_field(meta, VECTOR_DIMENSION, &field.vector_dimension.to_string());
        write_field(meta, SIZE, &doc_ids.len().to_string());
        for doc_id in doc_ids {
            write(meta, doc_id.to_string().as_bytes());
            write_newline(meta);
        }
    }

    /// `finish()` (then `close()`): the `-1` field number and both
    /// checksums. Returns `(vec, gri)`.
    pub fn finish(mut self) -> (Vec<u8>, Vec<u8>) {
        // Java: SimpleTextKnnVectorsWriter.finish
        write_field(&mut self.meta, FIELD_NUMBER, "-1");
        write_checksum(&mut self.meta);
        write_checksum(&mut self.vector_data);
        (self.vector_data, self.meta)
    }
}

/// `writeField(out, fieldName, value)`: the prefix, the value, a newline.
fn write_field(out: &mut Vec<u8>, prefix: &[u8], value: &str) {
    write(out, prefix);
    write(out, value.as_bytes());
    write_newline(out);
}

/// Java's `BufferingKnnVectorsWriter.addField`, which hands the field
/// float or byte values by its encoding: values of the other kind are an
/// `IllegalArgument`.
fn check_encoding(field: &FieldInfo, encoding: VectorEncoding) -> Result<()> {
    if field.vector_encoding != encoding {
        return Err(Error::IllegalArgument(format!(
            "field \"{}\" is {:?}, not {encoding:?}",
            field.name, field.vector_encoding
        )));
    }
    Ok(())
}

/// Java's `assert value.length == vectors.dimension()`.
fn check_dimension(field: &FieldInfo, len: usize) -> Result<()> {
    if i32::try_from(len).ok() != Some(field.vector_dimension) {
        return Err(Error::IllegalArgument(format!(
            "field \"{}\": a vector of {len} dimensions, expected {}",
            field.name, field.vector_dimension
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field_infos::VectorSimilarityFunction;

    fn field(name: &str, number: i32, dims: i32, encoding: VectorEncoding) -> FieldInfo {
        FieldInfo::new(name, number).with_vectors(dims, encoding, VectorSimilarityFunction::Cosine)
    }

    #[test]
    fn writes_javas_lines() {
        let f = field("f\nx", 3, 2, VectorEncoding::Float32);
        let b = field("b", 4, 3, VectorEncoding::Byte);
        let mut w = SimpleTextKnnVectorsWriter::new();
        let v1 = [1.0f32, -0.001];
        let v2 = [1e7f32, 1.5e-5];
        w.write_float_field(&f, [(0, &v1[..]), (5, &v2[..])])
            .unwrap();
        let bv = [0x80u8, 0x7f, 0xff];
        w.write_byte_field(&b, [(2, &bv[..])]).unwrap();
        w.write_float_field(&f, []).unwrap();
        let (vec, gri) = w.finish();
        let vec = String::from_utf8(vec).unwrap();
        let body = "[1.0, -0.001]\n[1.0E7, 1.5E-5]\n[-128, 127, -1]\n";
        assert!(vec.starts_with(body), "{vec}");
        assert_eq!(vec.len(), body.len() + 30);
        let gri = String::from_utf8(gri).unwrap();
        let want = "field-number 3\nfield-name f\\\nx\nvector-data-offset 0\n\
                    vector-data-length 30\nvector-dimension 2\nsize 2\n0\n5\n\
                    field-number 4\nfield-name b\nvector-data-offset 30\n\
                    vector-data-length 16\nvector-dimension 3\nsize 1\n2\n\
                    field-number 3\nfield-name f\\\nx\nvector-data-offset 46\n\
                    vector-data-length 0\nvector-dimension 2\nsize 0\n\
                    field-number -1\nchecksum ";
        assert!(gri.starts_with(want), "{gri}");
        assert_eq!(vector_data_file_name("_0", ""), "_0.vec");
        assert_eq!(meta_file_name("_0", "s"), "_0_s.gri");
    }

    #[test]
    fn an_empty_writer_writes_two_checksums() {
        let (vec, gri) = SimpleTextKnnVectorsWriter::new().finish();
        assert_eq!(vec.len(), 30);
        assert!(gri.starts_with(b"field-number -1\nchecksum "));
    }

    #[test]
    fn refuses_a_vector_of_another_dimension() {
        let f = field("f", 0, 2, VectorEncoding::Float32);
        let mut w = SimpleTextKnnVectorsWriter::new();
        let e = w.write_float_field(&f, [(0, &[1.0f32][..])]).unwrap_err();
        assert!(matches!(e, Error::IllegalArgument(m) if m.contains("1 dimensions")));
        let b = field("b", 1, 2, VectorEncoding::Byte);
        let e = w.write_byte_field(&b, [(0, &[1u8, 2, 3][..])]).unwrap_err();
        assert!(matches!(e, Error::IllegalArgument(m) if m.contains("3 dimensions")));
    }

    #[test]
    fn refuses_values_of_the_other_encoding() {
        let f = field("f", 0, 2, VectorEncoding::Float32);
        let b = field("b", 1, 2, VectorEncoding::Byte);
        let mut w = SimpleTextKnnVectorsWriter::new();
        let e = w.write_byte_field(&f, [(0, &[1u8, 2][..])]).unwrap_err();
        assert!(matches!(e, Error::IllegalArgument(m) if m.contains("not Byte")));
        let e = w
            .write_float_field(&b, [(0, &[1.0f32, 2.0][..])])
            .unwrap_err();
        assert!(matches!(e, Error::IllegalArgument(m) if m.contains("not Float32")));
        // Nothing was written.
        let (vec, gri) = w.finish();
        assert_eq!(vec.len(), 30);
        assert!(gri.starts_with(
            b"field-number -1
"
        ));
    }
}
