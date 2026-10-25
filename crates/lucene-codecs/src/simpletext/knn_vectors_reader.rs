//! `SimpleTextKnnVectorsReader`: the `.gri` field entries at open, then a
//! field's vectors parsed from its slice of the `.vec` whenever its values
//! are asked for (Java reads them all up front, every time), and the search:
//! an exhaustive scan scoring every accepted vector by the field's
//! similarity function into a [`KnnCollect`].
//!
//! What differs from Java: the reader borrows both files' bytes and the
//! segment's [`FieldInfos`]. Every `assert` (a line's prefix, a field number
//! seen twice, a vector line of other than `dimension` values), every
//! `NumberFormatException`, the `NegativeArraySizeException` of a negative
//! `size`, the `NullPointerException` of a field name the field infos do not
//! know, a vector line too short to hold its brackets and a slice outside
//! the `.vec` (Java's `IllegalArgumentException`) is `Corrupted` (or
//! `Eof`); the field-name line's prefix is checked (Java strips its length
//! unchecked). A field whose dimension differs from its entry's is
//! `Corrupted` too (Java's `IllegalStateException`, with Java's message). A
//! search on a field unknown to the field infos or without an entry collects
//! nothing (Java's values are
//! `null` there and its search throws a `NullPointerException`); an accepted
//! doc outside the accept bits is `Corrupted`. Java's `VectorUtil` asserts
//! every score is finite: a float vector holding `NaN`/`Infinity`, a score
//! that overflows and a zero vector in a COSINE field (Java: `NaN`, which
//! sorts first) are `Corrupted`, a zero COSINE target `IllegalArgument`. Vectors are parsed into one
//! flat array grown line by line, never sized from a count off the file.
//! Float text is Java's `Float.parseFloat` without its hexadecimal form
//! ([`super::stored_fields_reader::parse_java_float`]). `getOffHeapByteSize`
//! and the iterator's `cost` are ported; `copy()` (Java returns `this`) and
//! `ramBytesUsed` are not.

use std::collections::HashMap;

use lucene_util::bits::Bits;

use super::knn_vectors_writer::{
    FIELD_NAME, FIELD_NUMBER, SIZE, VECTOR_DATA_LENGTH, VECTOR_DATA_OFFSET, VECTOR_DIMENSION,
};
use super::stored_fields_reader::parse_java_float;
use super::util::{self, parse_integer, parse_long, read_line, read_string, read_value};
use super::{corrupt, Error, Result, NO_MORE_DOCS};
use crate::field_infos::{FieldInfo, FieldInfos, VectorSimilarityFunction};
use crate::hnsw::KnnCollect;

/// One field's `.gri` entry (Java's private `FieldEntry` record).
#[derive(Debug, Clone, PartialEq)]
pub struct FieldEntry {
    /// `vector-dimension`.
    pub dimension: i32,
    /// `vector-data-offset`: where the field's lines start in the `.vec`.
    pub vector_data_offset: i64,
    /// `vector-data-length`: the bytes of its lines.
    pub vector_data_length: i64,
    /// The doc id of each ord.
    pub ord_to_doc: Vec<i32>,
    /// The field's similarity, from the field infos.
    pub similarity_function: VectorSimilarityFunction,
}

impl FieldEntry {
    /// `size()`: the number of vectors.
    pub fn size(&self) -> usize {
        self.ord_to_doc.len()
    }

    /// Decode this entry's vectors from the same `.vec` file. The result
    /// borrows only the entry; decoded elements are owned, as in Java.
    ///
    /// # Errors
    /// `Corrupted` or `Eof` for an invalid slice, dimension or vector line.
    pub fn vector_values<T: VectorElement>(
        &self,
        vector_data: &[u8],
    ) -> Result<SimpleTextVectorValues<'_, T>> {
        SimpleTextVectorValues::new(self, slice(vector_data, self)?)
    }
}

/// A vector element type: Java's `float` (`f32`) or `byte` (`u8`, a Java
/// `byte`'s bits).
pub trait VectorElement: Copy + std::fmt::Debug {
    /// One value of a vector line: `Float.parseFloat(s)`, cast `(byte)` for
    /// bytes.
    ///
    /// # Errors
    /// `Corrupted` for text Java would not parse.
    fn parse(text: &[u8]) -> Result<Self>;

    /// `similarityFunction.compare(vector, target)`.
    fn compare(similarity: VectorSimilarityFunction, vector: &[Self], target: &[Self]) -> f32;

    /// Whether every component is zero (a COSINE score's `0 / 0`).
    fn is_zero(vector: &[Self]) -> bool;
}

impl VectorElement for f32 {
    fn parse(text: &[u8]) -> Result<Self> {
        let v = parse_java_float(text)?;
        if !v.is_finite() {
            // Java scores it, and `VectorUtil` asserts the score is finite.
            return Err(corrupt(format!("SimpleText vectors: non-finite value {v}")));
        }
        Ok(v)
    }

    fn compare(similarity: VectorSimilarityFunction, vector: &[f32], target: &[f32]) -> f32 {
        similarity.score(vector, target)
    }

    fn is_zero(vector: &[f32]) -> bool {
        vector.iter().all(|&v| v == 0.0)
    }
}

impl VectorElement for u8 {
    fn parse(text: &[u8]) -> Result<Self> {
        // Java: `(byte) Float.parseFloat(s)` -- float to int (toward zero,
        // saturating, NaN to 0), then the int's low byte.
        Ok(parse_java_float(text)? as i32 as u8)
    }

    fn compare(similarity: VectorSimilarityFunction, vector: &[u8], target: &[u8]) -> f32 {
        similarity.score_bytes(vector, target)
    }

    fn is_zero(vector: &[u8]) -> bool {
        vector.iter().all(|&v| v == 0)
    }
}

/// `org.apache.lucene.codecs.simpletext.SimpleTextKnnVectorsReader`.
#[derive(Debug, Clone)]
pub struct SimpleTextKnnVectorsReader<'a> {
    field_infos: &'a FieldInfos,
    data_in: &'a [u8],
    field_entries: HashMap<i32, FieldEntry>,
}

impl<'a> SimpleTextKnnVectorsReader<'a> {
    /// `new SimpleTextKnnVectorsReader(readState)` over the `.gri` and `.vec`
    /// bytes: the `.gri` is checksummed, the `.vec` only by
    /// [`Self::check_integrity`].
    ///
    /// # Errors
    /// `Corrupted` or `Eof` for a damaged `.gri`.
    pub fn open(meta: &[u8], vector_data: &'a [u8], field_infos: &'a FieldInfos) -> Result<Self> {
        // Java: SimpleTextKnnVectorsReader.<init>
        let mut pos = 0usize;
        let mut scratch = Vec::new();
        let mut field_entries = HashMap::new();
        let mut field_number = read_int(meta, &mut pos, &mut scratch, FIELD_NUMBER)?;
        while field_number != -1 {
            let field_name = read_string(meta, &mut pos, &mut scratch, FIELD_NAME)?;
            let vector_data_offset = parse_long(read_value(
                meta,
                &mut pos,
                &mut scratch,
                VECTOR_DATA_OFFSET,
            )?)?;
            let vector_data_length = parse_long(read_value(
                meta,
                &mut pos,
                &mut scratch,
                VECTOR_DATA_LENGTH,
            )?)?;
            let dimension = read_int(meta, &mut pos, &mut scratch, VECTOR_DIMENSION)?;
            let size = read_int(meta, &mut pos, &mut scratch, SIZE)?;
            if size < 0 {
                return Err(corrupt(format!(
                    "SimpleText vectors: negative size {size} for field \"{field_name}\""
                )));
            }
            let mut doc_ids = Vec::new();
            for _ in 0..size {
                doc_ids.push(read_int(meta, &mut pos, &mut scratch, b"")?);
            }
            if field_entries.contains_key(&field_number) {
                return Err(corrupt(format!(
                    "SimpleText vectors: field number {field_number} listed twice"
                )));
            }
            let info = field_infos.field_by_name(&field_name).ok_or_else(|| {
                corrupt(format!(
                    "SimpleText vectors: field \"{field_name}\" is not in the field infos"
                ))
            })?;
            field_entries.insert(
                field_number,
                FieldEntry {
                    dimension,
                    vector_data_offset,
                    vector_data_length,
                    ord_to_doc: doc_ids,
                    similarity_function: info.vector_similarity_function,
                },
            );
            field_number = read_int(meta, &mut pos, &mut scratch, FIELD_NUMBER)?;
        }
        util::check_footer(meta, 0, &mut pos)?;
        Ok(SimpleTextKnnVectorsReader {
            field_infos,
            data_in: vector_data,
            field_entries,
        })
    }

    /// The entry the `.gri` holds for field number `number`.
    pub fn field_entry(&self, number: i32) -> Option<&FieldEntry> {
        self.field_entries.get(&number)
    }

    /// Move the parsed, owned `.gri` entries into a segment core. Values
    /// can then borrow these entries and decode from independently owned
    /// file bytes without a self-referential reader.
    pub fn into_field_entries(self) -> HashMap<i32, FieldEntry> {
        self.field_entries
    }

    /// `getFloatVectorValues(field)`.
    ///
    /// # Errors
    /// What [`Self::vector_values`] fails with.
    pub fn float_vector_values(
        &self,
        field: &str,
    ) -> Result<Option<SimpleTextVectorValues<'_, f32>>> {
        self.vector_values(field)
    }

    /// `getByteVectorValues(field)`.
    ///
    /// # Errors
    /// What [`Self::vector_values`] fails with.
    pub fn byte_vector_values(
        &self,
        field: &str,
    ) -> Result<Option<SimpleTextVectorValues<'_, u8>>> {
        self.vector_values(field)
    }

    /// `getFloatVectorValues`/`getByteVectorValues`: every vector of
    /// `field`, parsed from its slice of the `.vec` (as Java, whatever the
    /// field's encoding). `None` for a field the field infos do not know or
    /// the `.gri` does not list.
    ///
    /// # Errors
    /// `IllegalState` (Java's message) for a field without vectors;
    /// `Corrupted` for a dimension other than the entry's, a slice outside
    /// the `.vec` or a damaged vector line; `Eof` for a missing line.
    pub fn vector_values<T: VectorElement>(
        &self,
        field: &str,
    ) -> Result<Option<SimpleTextVectorValues<'_, T>>> {
        // Java: SimpleTextKnnVectorsReader.getFloatVectorValues / getByteVectorValues
        let Some(info) = self.field_infos.field_by_name(field) else {
            return Ok(None);
        };
        let dimension = info.vector_dimension;
        if dimension == 0 {
            return Err(Error::IllegalState(
                "KNN vectors readers should not be called on fields that don't enable KNN vectors",
            ));
        }
        let Some(entry) = self.field_entries.get(&info.number) else {
            return Ok(None);
        };
        if dimension != entry.dimension {
            return Err(corrupt(format!(
                "Inconsistent vector dimension for field=\"{field}\"; {dimension} != {}",
                entry.dimension
            )));
        }
        entry.vector_values(self.data_in).map(Some)
    }

    /// `search(field, target, knnCollector, acceptDocs)` for either target
    /// type: every vector, in ord order, whose doc `accept_docs` accepts
    /// (`None`: every doc), scored against `target` by the field's
    /// similarity, until the collector is early-terminated.
    ///
    /// # Errors
    /// `IllegalArgument` (Java's message) for a target of another
    /// dimension; what [`Self::vector_values`] fails with; `Corrupted` for a
    /// doc outside `accept_docs`.
    pub fn search<T: VectorElement, C: KnnCollect + ?Sized>(
        &self,
        field: &str,
        target: &[T],
        knn_collector: &mut C,
        accept_docs: Option<&dyn Bits>,
    ) -> Result<()> {
        // Java: SimpleTextKnnVectorsReader.search
        let Some(values) = self.vector_values::<T>(field)? else {
            return Ok(());
        };
        if i32::try_from(target.len()).ok() != Some(values.dimension()) {
            return Err(Error::IllegalArgument(format!(
                "vector query dimension: {} differs from field dimension: {}",
                target.len(),
                values.dimension()
            )));
        }
        let vector_similarity = values.entry.similarity_function;
        let cosine = vector_similarity == VectorSimilarityFunction::Cosine;
        if cosine && T::is_zero(target) {
            return Err(Error::IllegalArgument(
                "a zero vector has no COSINE similarity".into(),
            ));
        }
        for (ord, &doc) in values.entry.ord_to_doc.iter().enumerate() {
            if let Some(bits) = accept_docs {
                if !accepted(bits, doc)? {
                    continue;
                }
            }
            if knn_collector.early_terminated() {
                break;
            }
            let Some(vector) = values.vector_value(ord) else {
                break;
            };
            if cosine && T::is_zero(vector) {
                return Err(corrupt(format!(
                    "SimpleText vectors: doc {doc} holds a zero vector in a COSINE field"
                )));
            }
            let score = T::compare(vector_similarity, vector, target);
            if !score.is_finite() {
                return Err(corrupt(format!(
                    "SimpleText vectors: doc {doc} scores {score}"
                )));
            }
            knn_collector.collect(doc, score);
            knn_collector.inc_visited_count(1);
        }
        Ok(())
    }

    /// `checkIntegrity()`: the `.vec`'s checksum; a `.vec` that is only its
    /// checksum line is checked as that line.
    ///
    /// # Errors
    /// `Corrupted` for a footer elsewhere or a wrong checksum.
    pub fn check_integrity(&self) -> Result<()> {
        // Java: SimpleTextKnnVectorsReader.checkIntegrity
        if i64::try_from(self.data_in.len()).ok() == Some(util::FOOTER_LENGTH) {
            let mut pos = 0;
            return util::check_footer(self.data_in, 0, &mut pos);
        }
        util::check_integrity(self.data_in)
    }

    /// `getOffHeapByteSize(fieldInfo)`: `None` for a field the `.gri` does
    /// not list, else empty (every vector is on heap).
    pub fn off_heap_byte_size(&self, field_info: &FieldInfo) -> Option<HashMap<String, i64>> {
        // Java: SimpleTextKnnVectorsReader.getOffHeapByteSize
        self.field_entries
            .get(&field_info.number)
            .map(|_| HashMap::new())
    }
}

/// `readInt(in, field)`: the line, its prefix asserted, `Integer.parseInt`
/// of the rest.
fn read_int(input: &[u8], pos: &mut usize, scratch: &mut Vec<u8>, prefix: &[u8]) -> Result<i32> {
    parse_integer(read_value(input, pos, scratch, prefix)?)
}

/// Whether `bits` accepts `doc`; a doc outside them is corrupt (Java's
/// `FixedBitSet.get` would fail or read another word).
fn accepted(bits: &dyn Bits, doc: i32) -> Result<bool> {
    match usize::try_from(doc) {
        Ok(d) if d < bits.length() => Ok(bits.get(d)),
        _ => Err(corrupt(format!(
            "SimpleText vectors: doc {doc} outside the accepted docs ({})",
            bits.length()
        ))),
    }
}

/// `dataIn.slice("vector-data", offset, length)`.
fn slice<'d>(data: &'d [u8], entry: &FieldEntry) -> Result<&'d [u8]> {
    let (offset, length) = (entry.vector_data_offset, entry.vector_data_length);
    usize::try_from(offset)
        .ok()
        .zip(usize::try_from(length).ok())
        .and_then(|(o, l)| data.get(o..o.checked_add(l)?))
        .ok_or_else(|| {
            corrupt(format!(
                "slice() vector-data out of bounds: offset={offset},length={length},fileLength={}",
                data.len()
            ))
        })
}

/// `SimpleTextFloatVectorValues` / `SimpleTextByteVectorValues`: a field's
/// vectors, all parsed when made.
#[derive(Debug, Clone)]
pub struct SimpleTextVectorValues<'r, T> {
    entry: &'r FieldEntry,
    dimension: usize,
    /// Every vector, `dimension` values each, in ord order.
    values: Vec<T>,
}

impl<'r, T: VectorElement> SimpleTextVectorValues<'r, T> {
    fn new(entry: &'r FieldEntry, input: &[u8]) -> Result<Self> {
        let dimension = usize::try_from(entry.dimension)
            .ok()
            .filter(|&d| d > 0)
            .ok_or_else(|| corrupt(format!("SimpleText vectors: dimension {}", entry.dimension)))?;
        let mut values = Vec::new();
        // Java: readAllVectors
        let mut pos = 0usize;
        let mut scratch = Vec::new();
        for _ in 0..entry.size() {
            read_line(input, &mut pos, &mut scratch)?;
            read_vector(&scratch, dimension, &mut values)?;
        }
        Ok(SimpleTextVectorValues {
            entry,
            dimension,
            values,
        })
    }

    /// `dimension()`.
    pub fn dimension(&self) -> i32 {
        self.entry.dimension
    }

    /// `size()`.
    pub fn size(&self) -> usize {
        self.entry.size()
    }

    /// `vectorValue(ord)`; `None` past the last ord (Java's array index
    /// error).
    pub fn vector_value(&self, ord: usize) -> Option<&[T]> {
        let start = ord.checked_mul(self.dimension)?;
        self.values.get(start..start.checked_add(self.dimension)?)
    }

    /// `ordToDoc(ord)`; `None` past the last ord.
    pub fn ord_to_doc(&self, ord: usize) -> Option<i32> {
        self.entry.ord_to_doc.get(ord).copied()
    }

    /// `iterator()`: `createSparseIterator()`.
    pub fn iterator(&self) -> DocIndexIterator<'r> {
        DocIndexIterator {
            ord_to_doc: &self.entry.ord_to_doc,
            ord: -1,
        }
    }

    /// `scorer(target)`: `None` for a field with no vectors.
    pub fn scorer<'v>(&'v self, target: &'v [T]) -> Option<VectorScorer<'v, 'r, T>> {
        if self.size() == 0 {
            return None;
        }
        Some(VectorScorer {
            values: self,
            target,
            iterator: self.iterator(),
        })
    }
}

/// `KnnVectorValues.createSparseIterator()`: the docs of each ord in turn.
#[derive(Debug, Clone)]
pub struct DocIndexIterator<'r> {
    ord_to_doc: &'r [i32],
    /// -1 before the first doc, [`NO_MORE_DOCS`] after the last.
    ord: i32,
}

impl DocIndexIterator<'_> {
    /// `docID()`.
    // SENTINEL: `-1` = "not positioned yet", as `DocIdSetIterator.docID()`
    // before the first `nextDoc`; `NO_MORE_DOCS` once exhausted.
    pub fn doc_id(&self) -> i32 {
        match self.ord {
            -1 => -1,
            NO_MORE_DOCS => NO_MORE_DOCS,
            ord => usize::try_from(ord)
                .ok()
                .and_then(|o| self.ord_to_doc.get(o))
                .copied()
                .unwrap_or(NO_MORE_DOCS),
        }
    }

    /// `index()`: the current ord.
    pub fn index(&self) -> i32 {
        self.ord
    }

    /// `nextDoc()`.
    pub fn next_doc(&mut self) -> i32 {
        let last = i32::try_from(self.ord_to_doc.len())
            .unwrap_or(i32::MAX)
            .saturating_sub(1);
        self.ord = if self.ord >= last {
            NO_MORE_DOCS
        } else {
            // ARITH: `ord < last <= i32::MAX - 1`, so `ord + 1` fits.
            #[allow(clippy::arithmetic_side_effects)]
            {
                self.ord + 1
            }
        };
        self.doc_id()
    }

    /// `advance(target)`: `slowAdvance`.
    pub fn advance(&mut self, target: i32) -> i32 {
        loop {
            let doc = self.next_doc();
            if doc >= target {
                return doc;
            }
        }
    }

    /// `cost()`: the number of vectors.
    pub fn cost(&self) -> usize {
        self.ord_to_doc.len()
    }
}

/// The `VectorScorer` of `scorer(target)`: the iterator's current vector
/// against the target.
#[derive(Debug, Clone)]
pub struct VectorScorer<'v, 'r, T> {
    values: &'v SimpleTextVectorValues<'r, T>,
    target: &'v [T],
    iterator: DocIndexIterator<'r>,
}

impl<'r, T: VectorElement> VectorScorer<'_, 'r, T> {
    /// `iterator()`.
    pub fn iterator(&mut self) -> &mut DocIndexIterator<'r> {
        &mut self.iterator
    }

    /// `score()`: `None` while the iterator is not on a doc (Java's array
    /// index error).
    pub fn score(&self) -> Option<f32> {
        let ord = usize::try_from(self.iterator.index()).ok()?;
        let vector = self.values.vector_value(ord)?;
        Some(T::compare(
            self.values.entry.similarity_function,
            vector,
            self.target,
        ))
    }
}

/// `readVector(value)`: one `Arrays.toString` line, its brackets skipped
/// unchecked, split on `,` as `String.split` does, each part parsed.
fn read_vector<T: VectorElement>(line: &[u8], dimension: usize, out: &mut Vec<T>) -> Result<()> {
    // Java: new BytesRef(scratch.bytes(), 1, scratch.length() - 2).utf8ToString()
    let inner = line
        .len()
        .checked_sub(1)
        .and_then(|end| line.get(1..end))
        .ok_or_else(|| corrupt("SimpleText vectors: a vector line without its brackets"))?;
    let text = String::from_utf8_lossy(inner);
    // `String.split(",")` drops trailing empty strings -- all of them for
    // ",,", none of "" (which splits to one empty string).
    let kept = text.trim_end_matches(',');
    let parts = if kept.is_empty() && !text.is_empty() {
        0
    } else {
        kept.split(',').count()
    };
    if parts != dimension {
        return Err(corrupt(format!(
            "SimpleText vectors: read {text} when expecting {dimension} floats"
        )));
    }
    if parts > 0 {
        for part in kept.split(',') {
            out.push(T::parse(part.as_bytes())?);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::knn_vectors_writer::SimpleTextKnnVectorsWriter;
    use super::*;
    use crate::field_infos::VectorEncoding;
    use crate::hnsw::KnnCollector;
    use lucene_util::fixed_bit_set::FixedBitSet;

    fn infos() -> FieldInfos {
        FieldInfos::new(vec![
            FieldInfo::new("f", 0).with_vectors(
                2,
                VectorEncoding::Float32,
                VectorSimilarityFunction::Euclidean,
            ),
            FieldInfo::new("b", 1).with_vectors(
                3,
                VectorEncoding::Byte,
                VectorSimilarityFunction::DotProduct,
            ),
            FieldInfo::new("plain", 2),
            FieldInfo::new("absent", 3).with_vectors(
                2,
                VectorEncoding::Float32,
                VectorSimilarityFunction::Cosine,
            ),
        ])
        .unwrap()
    }

    /// `(vec, gri)`: `f` with docs 1, 3, 4 and `b` with doc 0.
    fn files(infos: &FieldInfos) -> (Vec<u8>, Vec<u8>) {
        let mut w = SimpleTextKnnVectorsWriter::new();
        let f = [[0.0f32, 1.0], [2.0, 2.0], [0.0, 1.0]];
        w.write_float_field(
            infos.field_by_name("f").unwrap(),
            [1, 3, 4].into_iter().zip(f.iter().map(|v| &v[..])),
        )
        .unwrap();
        w.write_byte_field(
            infos.field_by_name("b").unwrap(),
            [(0, &[1u8, 0xff, 2][..])],
        )
        .unwrap();
        w.finish()
    }

    fn reseal(mut body: Vec<u8>) -> Vec<u8> {
        let end = body.len().saturating_sub(30);
        body.truncate(end);
        util::write_checksum(&mut body);
        body
    }

    fn edit(file: &[u8], from: &str, to: &str) -> Vec<u8> {
        let text = String::from_utf8(file.to_vec()).unwrap();
        assert!(text.contains(from), "{from}");
        reseal(text.replacen(from, to, 1).into_bytes())
    }

    #[test]
    fn reads_values_iterators_and_scorers() {
        let infos = infos();
        let (vec, gri) = files(&infos);
        let r = SimpleTextKnnVectorsReader::open(&gri, &vec, &infos).unwrap();
        r.check_integrity().unwrap();
        let v = r.float_vector_values("f").unwrap().unwrap();
        assert_eq!((v.dimension(), v.size()), (2, 3));
        assert_eq!(v.vector_value(1), Some(&[2.0f32, 2.0][..]));
        assert_eq!(v.vector_value(3), None);
        assert_eq!(v.ord_to_doc(2), Some(4));
        assert_eq!(v.ord_to_doc(3), None);
        let mut it = v.iterator();
        assert_eq!((it.doc_id(), it.index(), it.cost()), (-1, -1, 3));
        assert_eq!(it.advance(3), 3);
        assert_eq!(it.index(), 1);
        assert_eq!(it.next_doc(), 4);
        assert_eq!(it.next_doc(), NO_MORE_DOCS);
        assert_eq!(it.doc_id(), NO_MORE_DOCS);
        let target = [0.0f32, 1.0];
        let mut s = v.scorer(&target).unwrap();
        assert_eq!(s.score(), None);
        assert_eq!(s.iterator().next_doc(), 1);
        assert_eq!(s.score(), Some(1.0));
        // A byte field read as floats, and the reverse: Java parses either.
        let b = r.byte_vector_values("b").unwrap().unwrap();
        assert_eq!(b.vector_value(0), Some(&[1u8, 0xff, 2][..]));
        let bf = r.float_vector_values("b").unwrap().unwrap();
        assert_eq!(bf.vector_value(0), Some(&[1.0f32, -1.0, 2.0][..]));
        assert!(r.float_vector_values("nope").unwrap().is_none());
        assert!(r.float_vector_values("absent").unwrap().is_none());
        let e = r.float_vector_values("plain").unwrap_err();
        assert!(matches!(e, Error::IllegalState(m) if m.starts_with("KNN vectors readers")));
        let fi = infos.field_by_name("f").unwrap();
        assert_eq!(r.off_heap_byte_size(fi), Some(HashMap::new()));
        assert_eq!(
            r.off_heap_byte_size(infos.field_by_name("absent").unwrap()),
            None
        );
        assert_eq!(r.field_entry(0).unwrap().size(), 3);
        // An empty field has no scorer.
        let empty = FieldEntry {
            dimension: 2,
            vector_data_offset: 0,
            vector_data_length: 0,
            ord_to_doc: Vec::new(),
            similarity_function: VectorSimilarityFunction::Cosine,
        };
        let ev = SimpleTextVectorValues::<f32>::new(&empty, b"").unwrap();
        assert!(ev.scorer(&target).is_none());
        assert_eq!(ev.iterator().next_doc(), NO_MORE_DOCS);
    }

    #[test]
    fn searches_in_ord_order() {
        let infos = infos();
        let (vec, gri) = files(&infos);
        let r = SimpleTextKnnVectorsReader::open(&gri, &vec, &infos).unwrap();
        let mut c = KnnCollector::new(2, u64::MAX);
        r.search("f", &[0.0f32, 1.0], &mut c, None).unwrap();
        assert_eq!(c.visited_count(), 3);
        // Docs 1 and 4 tie: the lower doc first.
        assert_eq!(c.top_docs(), vec![(1, 1.0), (4, 1.0)]);
        let mut odd = FixedBitSet::new(5);
        odd.set(3);
        odd.set(1);
        let mut c = KnnCollector::new(5, u64::MAX);
        r.search("f", &[0.0f32, 1.0], &mut c, Some(&odd)).unwrap();
        assert_eq!(c.top_docs(), vec![(1, 1.0), (3, 1.0 / 6.0)]);
        // The visit limit stops the scan.
        let mut c = KnnCollector::new(5, 1);
        r.search("f", &[0.0f32, 1.0], &mut c, None).unwrap();
        assert_eq!(c.top_docs(), vec![(1, 1.0)]);
        let mut c = KnnCollector::new(5, u64::MAX);
        r.search("b", &[1u8, 1, 1], &mut c, None).unwrap();
        assert_eq!(c.top_docs().len(), 1);
        // No entry: nothing collected.
        let mut c = KnnCollector::new(5, u64::MAX);
        r.search("absent", &[0.0f32, 1.0], &mut c, None).unwrap();
        assert_eq!(c.visited_count(), 0);
        let e = r.search("f", &[0.0f32], &mut c, None).unwrap_err();
        assert!(
            matches!(&e, Error::IllegalArgument(m) if m == "vector query dimension: 1 differs from field dimension: 2"),
            "{e}"
        );
        let short = FixedBitSet::new(2);
        let e = r
            .search("f", &[0.0f32, 1.0], &mut c, Some(&short))
            .unwrap_err();
        assert!(e.to_string().contains("outside the accepted docs"), "{e}");
    }

    #[test]
    fn checks_integrity() {
        let infos = infos();
        let (vec, gri) = files(&infos);
        let empty_vec = SimpleTextKnnVectorsWriter::new().finish().0;
        let r = SimpleTextKnnVectorsReader::open(&gri, &empty_vec, &infos).unwrap();
        r.check_integrity().unwrap();
        let mut bad = vec.clone();
        bad[2] ^= 1;
        let r = SimpleTextKnnVectorsReader::open(&gri, &bad, &infos).unwrap();
        assert!(r.check_integrity().is_err());
        let mut bad = empty_vec.clone();
        bad[12] ^= 1;
        let r = SimpleTextKnnVectorsReader::open(&gri, &bad, &infos).unwrap();
        assert!(r.check_integrity().is_err());
    }

    #[test]
    fn refuses_a_damaged_meta() {
        let infos = infos();
        let (vec, gri) = files(&infos);
        let open = |gri: &[u8]| SimpleTextKnnVectorsReader::open(gri, &vec, &infos).err();
        let cases = [
            ("size 3", "size -1"),
            ("size 3", "size x"),
            ("field-number 1", "field-number 0"),
            ("field-name b", "field-name nope"),
            ("field-name f", "feld-name f"),
            ("field-number -1", "field-number -2"),
            ("\n4\n", "\n+4x\n"),
        ];
        for (from, to) in cases {
            assert!(open(&edit(&gri, from, to)).is_some(), "{to}");
        }
        let mut flipped = gri.clone();
        flipped[3] ^= 1;
        assert!(open(&flipped).is_some());
        assert!(open(&gri[..gri.len() - 1]).is_some());
        assert!(open(b"").is_some());
    }

    #[test]
    fn refuses_damaged_vectors() {
        let infos = infos();
        let (vec, gri) = files(&infos);
        let values = |gri: &[u8], vec: &[u8]| -> Result<()> {
            let r = SimpleTextKnnVectorsReader::open(gri, vec, &infos)?;
            r.float_vector_values("f")?;
            r.byte_vector_values("b")?;
            Ok(())
        };
        values(&gri, &vec).unwrap();
        let gri_cases = [
            ("vector-dimension 2", "vector-dimension 3"),
            ("vector-data-offset 0", "vector-data-offset -1"),
            ("vector-data-length 33", "vector-data-length 9999"),
            ("vector-data-length 33", "vector-data-length -1"),
            ("vector-data-length 33", "vector-data-length 32"),
        ];
        for (from, to) in gri_cases {
            assert!(values(&edit(&gri, from, to), &vec).is_err(), "{to}");
        }
        // Same-length edits, so no offset moves.
        for (from, to, ok) in [
            ("[2.0, 2.0]", "[2.0, x.0]", false),
            ("[2.0, 2.0]", "[2.0,2.0,]", true),
            ("[2.0, 2.0]", "[2.0,2,3.]", false),
            ("[2.0, 2.0]", "[2.0      ", false),
            ("[1, -1, 2]", "[1, -1,  ]", false),
        ] {
            assert_eq!(values(&gri, &edit(&vec, from, to)).is_ok(), ok, "{to}");
        }
        // A line of one or no byte cannot hold its brackets.
        let mut dims = Vec::new();
        assert!(read_vector::<f32>(b"[", 1, &mut dims).is_err());
        assert!(read_vector::<f32>(b"", 1, &mut dims).is_err());
        assert!(read_vector::<f32>(b"[]", 1, &mut dims).is_err());
        assert!(read_vector::<f32>(b"[,,]", 1, &mut dims).is_err());
        read_vector::<f32>(b"x 1.5f,2,]", 2, &mut dims).unwrap();
        assert_eq!(dims, [1.5, 2.0]);
        let mut bytes = Vec::new();
        read_vector::<u8>(b"[300.7, -129, NaN]", 3, &mut bytes).unwrap();
        assert_eq!(bytes, [44, 127, 0]);
        let bad_dim = FieldEntry {
            dimension: -1,
            vector_data_offset: 0,
            vector_data_length: 0,
            ord_to_doc: vec![0],
            similarity_function: VectorSimilarityFunction::Cosine,
        };
        assert!(SimpleTextVectorValues::<f32>::new(&bad_dim, b"[1]\n").is_err());
    }

    /// One COSINE field of each encoding holding doc 0's vector.
    fn cosine_files(f: &[f32], b: &[u8]) -> (FieldInfos, Vec<u8>, Vec<u8>) {
        let infos = FieldInfos::new(vec![
            FieldInfo::new("f", 0).with_vectors(
                2,
                VectorEncoding::Float32,
                VectorSimilarityFunction::Cosine,
            ),
            FieldInfo::new("b", 1).with_vectors(
                2,
                VectorEncoding::Byte,
                VectorSimilarityFunction::Cosine,
            ),
        ])
        .unwrap();
        let mut w = SimpleTextKnnVectorsWriter::new();
        w.write_float_field(&infos.fields[0], [(0, f)]).unwrap();
        w.write_byte_field(&infos.fields[1], [(0, b)]).unwrap();
        let (vec, gri) = w.finish();
        (infos, vec, gri)
    }

    #[test]
    fn refuses_non_finite_values_and_scores() {
        let (infos, vec, gri) = cosine_files(&[f32::NAN, 1.0], &[1, 1]);
        let r = SimpleTextKnnVectorsReader::open(&gri, &vec, &infos).unwrap();
        let e = r.float_vector_values("f").unwrap_err();
        assert!(e.to_string().contains("non-finite value NaN"), "{e}");
        let mut c = KnnCollector::new(1, u64::MAX);
        assert!(r.search("f", &[1.0f32, 1.0], &mut c, None).is_err());
        let (infos, vec, gri) = cosine_files(&[f32::INFINITY, 1.0], &[1, 1]);
        let r = SimpleTextKnnVectorsReader::open(&gri, &vec, &infos).unwrap();
        assert!(r.float_vector_values("f").is_err());
        // A score that overflows: a DOT_PRODUCT of two huge vectors.
        let infos = FieldInfos::new(vec![FieldInfo::new("d", 0).with_vectors(
            1,
            VectorEncoding::Float32,
            VectorSimilarityFunction::DotProduct,
        )])
        .unwrap();
        let mut w = SimpleTextKnnVectorsWriter::new();
        w.write_float_field(&infos.fields[0], [(0, &[3e38f32][..])])
            .unwrap();
        let (vec, gri) = w.finish();
        let r = SimpleTextKnnVectorsReader::open(&gri, &vec, &infos).unwrap();
        let mut c = KnnCollector::new(1, u64::MAX);
        let e = r.search("d", &[3e38f32], &mut c, None).unwrap_err();
        assert!(e.to_string().contains("doc 0 scores inf"), "{e}");
    }

    #[test]
    fn refuses_zero_cosine_vectors() {
        let (infos, vec, gri) = cosine_files(&[0.0, -0.0], &[0, 0]);
        let r = SimpleTextKnnVectorsReader::open(&gri, &vec, &infos).unwrap();
        let mut c = KnnCollector::new(1, u64::MAX);
        let e = r.search("f", &[1.0f32, 0.0], &mut c, None).unwrap_err();
        assert!(
            e.to_string().contains("zero vector in a COSINE field"),
            "{e}"
        );
        let e = r.search("b", &[1u8, 0], &mut c, None).unwrap_err();
        assert!(
            e.to_string().contains("zero vector in a COSINE field"),
            "{e}"
        );
        let (infos, vec, gri) = cosine_files(&[1.0, 0.0], &[1, 0]);
        let r = SimpleTextKnnVectorsReader::open(&gri, &vec, &infos).unwrap();
        let mut c = KnnCollector::new(1, u64::MAX);
        let e = r.search("f", &[0.0f32, 0.0], &mut c, None).unwrap_err();
        assert!(matches!(e, Error::IllegalArgument(_)), "{e}");
        let e = r.search("b", &[0u8, 0], &mut c, None).unwrap_err();
        assert!(matches!(e, Error::IllegalArgument(_)), "{e}");
        r.search("f", &[2.0f32, 0.0], &mut c, None).unwrap();
        assert_eq!(c.top_docs(), [(0, 1.0)]);
    }
}
