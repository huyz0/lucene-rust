//! `SimpleTextDocValuesWriter` (and `SimpleTextDocValuesFormat`'s file
//! name): a segment's doc values as one `.dat` text file (norms reuse it as
//! `.len`, see [`super::norms`]).
//!
//! ```text
//! field <name>
//!   type <NUMERIC|BINARY|SORTED|SORTED_NUMERIC|SORTED_SET>
//!   minalue <min> / maxvalue <max>       NUMERIC, SORTED_NUMERIC (sic)
//!   doccount <docs with a value>
//!   maxvaluecount <most values in a doc>
//!   NUMERIC:  origin <min, or 0 if lower and a doc lacks a value>
//!             pattern <a 0 per digit of max - origin>
//!             per doc: <value - origin, zero-padded>\n<T|F>\n
//!   BINARY, SORTED_NUMERIC (whose value is the CSV of the doc's longs):
//!             maxlength <longest BytesRef.toString of a value>
//!             pattern <a 0 per digit of maxlength>
//!             per doc: length <n, padded>\n<BytesRef.toString, padded
//!                      with spaces to maxlength>\n<T|F>\n
//!   SORTED, SORTED_SET:
//!             numvalues <dictionary size> / maxlength <longest term>
//!             pattern <a 0 per digit of maxlength>
//!             ordpattern <SORTED: a 0 per digit of numvalues + 1;
//!                         SORTED_SET: an X per char of the longest ord CSV>
//!             per term: length <n, padded>\n<raw bytes, unescaped,
//!                       padded with spaces>\n
//!             per doc: SORTED <ord + 1, padded>\n; SORTED_SET <ord CSV,
//!                      padded with spaces>\n
//!   skipdata                             all but BINARY: per 8 docs with a
//!     interval                           value, their doc and value range
//!       minDocID <d> / maxDocID <d> / minValue <v> / maxValue <v> /
//!       docCount <n>
//!   skipend
//! END
//! checksum <20 digits>
//! ```
//!
//! Every record has a fixed width per field, so the reader seeks straight
//! to a document's (or a term's) line.
//!
//! What differs from Java: the writer takes this crate's in-memory columns
//! ([`DocValuesInput`]), not a `DocValuesProducer`, and returns the file
//! from [`SimpleTextDocValuesWriter::finish`]; its `assert`s (a field added
//! twice, the wrong doc-values type, documents out of order or outside the
//! segment, an ord outside the dictionary) are [`Error::IllegalArgument`].

use std::collections::HashSet;

use super::decimal_format;
use super::field_infos::doc_values_type_name;
use super::util::{bytes_ref_to_string, write, write_checksum, write_line, write_newline};
use super::{Error, Result};
use crate::doc_values::{BinaryColumn, SortedNumericColumn, SortedOrdsColumn, SortedSetOrdsColumn};
use crate::field_infos::{DocValuesType, FieldInfo, IndexOptions};

/// `SimpleTextDocValuesFormat`'s extension.
pub const DOC_VALUES_EXTENSION: &str = "dat";
/// `END`.
pub const END: &[u8] = b"END";
/// `FIELD`.
pub const FIELD: &[u8] = b"field ";
/// `TYPE`.
pub const TYPE: &[u8] = b"  type ";
/// `DOCCOUNT`.
pub const DOCCOUNT: &[u8] = b"  doccount ";
/// `ORIGIN`.
pub const ORIGIN: &[u8] = b"  origin ";
/// `MINVALUE` (Java's spelling).
pub const MINVALUE: &[u8] = b"  minalue ";
/// `MAXVALUE`.
pub const MAXVALUE: &[u8] = b"  maxvalue ";
/// `MAXVALUECOUNT`.
pub const MAXVALUECOUNT: &[u8] = b"  maxvaluecount ";
/// `PATTERN`.
pub const PATTERN: &[u8] = b"  pattern ";
/// `LENGTH`.
pub const LENGTH: &[u8] = b"length ";
/// `MAXLENGTH`.
pub const MAXLENGTH: &[u8] = b"  maxlength ";
/// `NUMVALUES`.
pub const NUMVALUES: &[u8] = b"  numvalues ";
/// `ORDPATTERN`.
pub const ORDPATTERN: &[u8] = b"  ordpattern ";
/// `SKIP_DATA`.
pub const SKIP_DATA: &[u8] = b"  skipdata";
/// `SKIP_INTERVAL`.
pub const SKIP_INTERVAL: &[u8] = b"    interval";
/// `SKIP_MIN_DOC_ID`.
pub const SKIP_MIN_DOC_ID: &[u8] = b"      minDocID ";
/// `SKIP_MAX_DOC_ID`.
pub const SKIP_MAX_DOC_ID: &[u8] = b"      maxDocID ";
/// `SKIP_MIN_VALUE`.
pub const SKIP_MIN_VALUE: &[u8] = b"      minValue ";
/// `SKIP_MAX_VALUE`.
pub const SKIP_MAX_VALUE: &[u8] = b"      maxValue ";
/// `SKIP_DOC_COUNT`.
pub const SKIP_DOC_COUNT: &[u8] = b"      docCount ";
/// `SKIP_END`.
pub const SKIP_END: &[u8] = b"  skipend";
/// `SKIP_INTERVAL_SIZE`: documents per skip interval.
pub const SKIP_INTERVAL_SIZE: i32 = 8;

/// `IndexFileNames.segmentFileName(segment, suffix, ext)`.
pub fn doc_values_file_name(segment: &str, suffix: &str, ext: &str) -> String {
    super::segment_file_name(segment, suffix, ext)
}

/// `SimpleTextDocValuesWriter.SkipInterval`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SkipInterval {
    /// The first document with a value in the interval.
    pub min_doc_id: i32,
    /// The last.
    pub max_doc_id: i32,
    /// The smallest value (or ord).
    pub min_value: i64,
    /// The largest.
    pub max_value: i64,
    /// Documents with a value in the interval.
    pub doc_count: i32,
}

/// One field's values in doc order, as the `add*Field` methods consume
/// them (the `DocValuesProducer` Java's take).
#[derive(Debug, Clone, Copy)]
pub enum DocValuesInput<'a> {
    /// `(doc, value)` per document with a value.
    Numeric(&'a [(i32, i64)]),
    /// BINARY.
    Binary(&'a BinaryColumn),
    /// SORTED: the dictionary and each document's ord.
    Sorted(&'a SortedOrdsColumn),
    /// SORTED_NUMERIC.
    SortedNumeric(&'a SortedNumericColumn),
    /// SORTED_SET: the dictionary and each document's ords.
    SortedSet(&'a SortedSetOrdsColumn),
}

/// The skip intervals' running state, shared by every `add*Field`.
#[derive(Debug)]
struct Intervals {
    list: Vec<SkipInterval>,
    doc_count: i32,
    min_doc: i32,
    max_doc: i32,
    min_value: i64,
    max_value: i64,
}

impl Intervals {
    fn new() -> Self {
        Intervals {
            list: Vec::new(),
            doc_count: 0,
            min_doc: -1,
            max_doc: -1,
            min_value: i64::MAX,
            max_value: i64::MIN,
        }
    }

    /// `if (intervalDocCount == 0) intervalMinDocID = doc; intervalMaxDocID
    /// = doc;`
    fn start_doc(&mut self, doc: i32) {
        if self.doc_count == 0 {
            self.min_doc = doc;
        }
        self.max_doc = doc;
    }

    fn value(&mut self, v: i64) {
        self.min_value = self.min_value.min(v);
        self.max_value = self.max_value.max(v);
    }

    /// `intervalDocCount++` and, at [`SKIP_INTERVAL_SIZE`], close the
    /// interval.
    fn end_doc(&mut self) {
        self.doc_count = self.doc_count.saturating_add(1);
        if self.doc_count == SKIP_INTERVAL_SIZE {
            self.close();
        }
    }

    fn close(&mut self) {
        self.list.push(SkipInterval {
            min_doc_id: self.min_doc,
            max_doc_id: self.max_doc,
            min_value: self.min_value,
            max_value: self.max_value,
            doc_count: self.doc_count,
        });
        self.doc_count = 0;
        self.min_value = i64::MAX;
        self.max_value = i64::MIN;
    }

    /// The trailing partial interval, then the list.
    fn finish(mut self) -> Vec<SkipInterval> {
        if self.doc_count > 0 {
            self.close();
        }
        self.list
    }
}

/// `"0" * n`.
fn zeros(n: usize) -> String {
    "0".repeat(n)
}

/// `SimpleTextDocValuesWriter`: the file is built in memory and returned by
/// [`Self::finish`].
#[derive(Debug)]
pub struct SimpleTextDocValuesWriter {
    data: Vec<u8>,
    num_docs: i32,
    fields_seen: HashSet<String>,
}

impl SimpleTextDocValuesWriter {
    /// `new SimpleTextDocValuesWriter(state, ext)` for a segment of
    /// `max_doc` documents.
    pub fn new(max_doc: i32) -> Self {
        SimpleTextDocValuesWriter {
            data: Vec::new(),
            num_docs: max_doc,
            fields_seen: HashSet::new(),
        }
    }

    /// `fieldSeen(field)`: a field may be added once.
    fn field_seen(&mut self, field: &str) -> Result<()> {
        if !self.fields_seen.insert(field.to_string()) {
            return Err(Error::IllegalArgument(format!(
                "field \"{field}\" was added more than once during flush"
            )));
        }
        Ok(())
    }

    /// The `add*Field` matching `values`' type.
    ///
    /// # Errors
    /// What that method fails with.
    pub fn add_field(&mut self, field: &FieldInfo, values: DocValuesInput<'_>) -> Result<()> {
        match values {
            DocValuesInput::Numeric(v) => self.add_numeric_field(field, v),
            DocValuesInput::Binary(v) => self.add_binary_field(field, v),
            DocValuesInput::Sorted(v) => self.add_sorted_field(field, v),
            DocValuesInput::SortedNumeric(v) => self.add_sorted_numeric_field(field, v),
            DocValuesInput::SortedSet(v) => self.add_sorted_set_field(field, v),
        }
    }

    fn check_type(field: &FieldInfo, want: DocValuesType, or_norms: bool) -> Result<()> {
        let norms = or_norms && field.index_options != IndexOptions::None && !field.omit_norms;
        if field.doc_values_type != want && !norms {
            return Err(Error::IllegalArgument(format!(
                "field \"{}\" has doc values type {}, not {}",
                field.name,
                doc_values_type_name(field.doc_values_type),
                doc_values_type_name(want)
            )));
        }
        Ok(())
    }

    /// `docs` strictly ascending within `0..numDocs` (what a doc-values
    /// iterator guarantees).
    fn check_docs(&self, field: &str, docs: impl Iterator<Item = i32>) -> Result<()> {
        let mut last = -1i32;
        for doc in docs {
            if doc <= last || doc >= self.num_docs {
                return Err(Error::IllegalArgument(format!(
                    "field \"{field}\": doc {doc} out of order or outside maxDoc={}",
                    self.num_docs
                )));
            }
            last = doc;
        }
        Ok(())
    }

    fn line(&mut self, prefix: &[u8], value: &str) {
        write_line(&mut self.data, prefix, value.as_bytes());
    }

    /// `addNumericField(field, valuesProducer)` (also norms, see
    /// [`super::norms`]).
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for a field added twice, of another type
    /// without norms, or documents out of order or outside the segment.
    pub fn add_numeric_field(&mut self, field: &FieldInfo, values: &[(i32, i64)]) -> Result<()> {
        // Java: SimpleTextDocValuesWriter.addNumericField
        Self::check_type(field, DocValuesType::Numeric, true)?;
        self.check_docs(&field.name, values.iter().map(|v| v.0))?;
        self.field_seen(&field.name)?;
        self.write_field_entry(field, DocValuesType::Numeric);
        let mut min_value = i64::MAX;
        let mut max_value = i64::MIN;
        let mut intervals = Intervals::new();
        for &(doc, v) in values {
            min_value = min_value.min(v);
            max_value = max_value.max(v);
            intervals.start_doc(doc);
            intervals.value(v);
            intervals.end_doc();
        }
        let num_values = values.len();
        let skip = intervals.finish();
        self.line(MINVALUE, &min_value.to_string());
        self.line(MAXVALUE, &max_value.to_string());
        self.line(DOCCOUNT, &num_values.to_string());
        self.line(MAXVALUECOUNT, if num_values == 0 { "0" } else { "1" });
        if usize::try_from(self.num_docs).ok() != Some(num_values) {
            min_value = min_value.min(0);
            max_value = max_value.max(0);
        }
        self.line(ORIGIN, &min_value.to_string());
        // Two i64s apart never wrap an i128.
        let diff = i128::from(max_value).wrapping_sub(i128::from(min_value));
        let width = diff.to_string().len();
        self.line(PATTERN, &zeros(width));
        let mut next = values.iter().peekable();
        for i in 0..self.num_docs {
            let (value, present) = match next.next_if(|v| v.0 == i) {
                Some(&(_, v)) => (v, true),
                None => (0, false),
            };
            let delta = i128::from(value).wrapping_sub(i128::from(min_value));
            let s = decimal_format::format(delta, width);
            write(&mut self.data, s.as_bytes());
            write_newline(&mut self.data);
            write(&mut self.data, if present { b"T" } else { b"F" });
            write_newline(&mut self.data);
        }
        self.write_skip_data(&skip);
        Ok(())
    }

    /// `addBinaryField(field, valuesProducer)`.
    ///
    /// # Errors
    /// As [`Self::add_numeric_field`], and lengths that do not add up to
    /// the column's bytes.
    pub fn add_binary_field(&mut self, field: &FieldInfo, values: &BinaryColumn) -> Result<()> {
        // Java: SimpleTextDocValuesWriter.addBinaryField
        Self::check_type(field, DocValuesType::Binary, false)?;
        let strings = binary_strings(&field.name, values)?;
        self.check_docs(&field.name, values.docs.iter().copied())?;
        self.field_seen(&field.name)?;
        self.write_field_entry(field, DocValuesType::Binary);
        self.do_add_binary_field(&values.docs, &strings, -1);
        Ok(())
    }

    /// `doAddBinaryField(field, valuesProducer, maxValueCount)` over each
    /// document's `BytesRef.toString()`.
    fn do_add_binary_field(&mut self, docs: &[i32], strings: &[String], max_value_count: i32) {
        // Java: SimpleTextDocValuesWriter.doAddBinaryField
        let max_length = strings.iter().map(String::len).max().unwrap_or(0);
        let doc_count = docs.len();
        self.line(DOCCOUNT, &doc_count.to_string());
        let count = if max_value_count == -1 {
            i32::from(doc_count != 0)
        } else {
            max_value_count
        };
        self.line(MAXVALUECOUNT, &count.to_string());
        self.line(MAXLENGTH, &max_length.to_string());
        let width = max_length.to_string().len();
        self.line(PATTERN, &zeros(width));
        let mut next = docs.iter().zip(strings).peekable();
        for i in 0..self.num_docs {
            let value = next.next_if(|v| *v.0 == i).map(|v| v.1);
            let length = value.map_or(0, String::len);
            write(&mut self.data, LENGTH);
            let n = decimal_format::format(length as i128, width);
            write(&mut self.data, n.as_bytes());
            write_newline(&mut self.data);
            if let Some(v) = value {
                write(&mut self.data, v.as_bytes());
            }
            for _ in length..max_length {
                self.data.push(b' ');
            }
            write_newline(&mut self.data);
            write(&mut self.data, if value.is_some() { b"T" } else { b"F" });
            write_newline(&mut self.data);
        }
    }

    /// `addSortedField(field, valuesProducer)`.
    ///
    /// # Errors
    /// As [`Self::add_numeric_field`], and an ord outside the dictionary or
    /// a column whose docs and ords differ in length.
    pub fn add_sorted_field(&mut self, field: &FieldInfo, values: &SortedOrdsColumn) -> Result<()> {
        // Java: SimpleTextDocValuesWriter.addSortedField
        Self::check_type(field, DocValuesType::Sorted, false)?;
        if values.docs.len() != values.ords.len() {
            return Err(Error::IllegalArgument(format!(
                "field \"{}\": {} docs but {} ords",
                field.name,
                values.docs.len(),
                values.ords.len()
            )));
        }
        check_ords(&field.name, &values.ords, values.dict.len())?;
        self.check_docs(&field.name, values.docs.iter().copied())?;
        self.field_seen(&field.name)?;
        self.write_field_entry(field, DocValuesType::Sorted);
        let mut intervals = Intervals::new();
        for (&doc, &ord) in values.docs.iter().zip(&values.ords) {
            intervals.start_doc(doc);
            intervals.value(ord);
            intervals.end_doc();
        }
        let skip = intervals.finish();
        let doc_count = values.docs.len();
        self.line(DOCCOUNT, &doc_count.to_string());
        self.line(MAXVALUECOUNT, if doc_count == 0 { "0" } else { "1" });
        let value_count = values.dict.len();
        let max_length: i64 = values
            .dict
            .iter()
            .map(|v| v.len() as i64)
            .max()
            .unwrap_or(-1);
        self.line(NUMVALUES, &value_count.to_string());
        self.line(MAXLENGTH, &max_length.to_string());
        let width = max_length.to_string().len();
        self.line(PATTERN, &zeros(width));
        let ord_width = (value_count as u64).saturating_add(1).to_string().len();
        self.line(ORDPATTERN, &zeros(ord_width));
        self.write_terms(&values.dict, width, max_length);
        let mut next = values.docs.iter().zip(&values.ords).peekable();
        for i in 0..self.num_docs {
            let ord = next.next_if(|v| *v.0 == i).map_or(-1, |v| *v.1);
            let s = decimal_format::format(i128::from(ord).wrapping_add(1), ord_width);
            write(&mut self.data, s.as_bytes());
            write_newline(&mut self.data);
        }
        self.write_skip_data(&skip);
        Ok(())
    }

    /// The dictionary: per term its padded length line, then its raw bytes
    /// padded with spaces to `max_length`.
    fn write_terms(&mut self, dict: &[Vec<u8>], width: usize, max_length: i64) {
        for value in dict {
            write(&mut self.data, LENGTH);
            let n = decimal_format::format(value.len() as i128, width);
            write(&mut self.data, n.as_bytes());
            write_newline(&mut self.data);
            self.data.extend_from_slice(value);
            for _ in (value.len() as i64)..max_length {
                self.data.push(b' ');
            }
            write_newline(&mut self.data);
        }
    }

    /// `addSortedNumericField(field, valuesProducer)`: the min and max, then
    /// each document's values as a CSV through `doAddBinaryField`.
    ///
    /// # Errors
    /// As [`Self::add_numeric_field`], and counts that do not add up to the
    /// column's values.
    pub fn add_sorted_numeric_field(
        &mut self,
        field: &FieldInfo,
        values: &SortedNumericColumn,
    ) -> Result<()> {
        // Java: SimpleTextDocValuesWriter.addSortedNumericField
        Self::check_type(field, DocValuesType::SortedNumeric, false)?;
        let per_doc = split_counts(&field.name, &values.docs, &values.counts, &values.values)?;
        self.check_docs(&field.name, values.docs.iter().copied())?;
        self.field_seen(&field.name)?;
        self.write_field_entry(field, DocValuesType::SortedNumeric);
        let mut min_value = i64::MAX;
        let mut max_value = i64::MIN;
        let mut max_value_count = 0usize;
        let mut intervals = Intervals::new();
        for (&doc, vals) in values.docs.iter().zip(&per_doc) {
            max_value_count = max_value_count.max(vals.len());
            intervals.start_doc(doc);
            for &v in *vals {
                min_value = min_value.min(v);
                max_value = max_value.max(v);
                intervals.value(v);
            }
            intervals.end_doc();
        }
        let skip = intervals.finish();
        self.line(MINVALUE, &min_value.to_string());
        self.line(MAXVALUE, &max_value.to_string());
        // The binary view: `BytesRef(csv).toString()` per document.
        let strings: Vec<String> = per_doc
            .iter()
            .map(|vals| {
                let csv: Vec<String> = vals.iter().map(i64::to_string).collect();
                bytes_ref_to_string(csv.join(",").as_bytes())
            })
            .collect();
        let max_value_count = i32::try_from(max_value_count).unwrap_or(i32::MAX);
        self.do_add_binary_field(&values.docs, &strings, max_value_count);
        self.write_skip_data(&skip);
        Ok(())
    }

    /// `addSortedSetField(field, valuesProducer)`.
    ///
    /// # Errors
    /// As [`Self::add_sorted_field`].
    pub fn add_sorted_set_field(
        &mut self,
        field: &FieldInfo,
        values: &SortedSetOrdsColumn,
    ) -> Result<()> {
        // Java: SimpleTextDocValuesWriter.addSortedSetField
        Self::check_type(field, DocValuesType::SortedSet, false)?;
        let per_doc = split_counts(&field.name, &values.docs, &values.counts, &values.ords)?;
        check_ords(&field.name, &values.ords, values.dict.len())?;
        self.check_docs(&field.name, values.docs.iter().copied())?;
        self.field_seen(&field.name)?;
        self.write_field_entry(field, DocValuesType::SortedSet);
        let mut max_value_count = 0usize;
        let mut intervals = Intervals::new();
        for (&doc, ords) in values.docs.iter().zip(&per_doc) {
            max_value_count = max_value_count.max(ords.len());
            intervals.start_doc(doc);
            for &ord in *ords {
                intervals.value(ord);
            }
            intervals.end_doc();
        }
        let skip = intervals.finish();
        self.line(DOCCOUNT, &values.docs.len().to_string());
        self.line(MAXVALUECOUNT, &max_value_count.to_string());
        let max_length = values.dict.iter().map(Vec::len).max().unwrap_or(0);
        self.line(NUMVALUES, &values.dict.len().to_string());
        self.line(MAXLENGTH, &max_length.to_string());
        let width = max_length.to_string().len();
        self.line(PATTERN, &zeros(width));
        let lists: Vec<String> = per_doc
            .iter()
            .map(|ords| {
                let parts: Vec<String> = ords.iter().map(i64::to_string).collect();
                parts.join(",")
            })
            .collect();
        let max_ord_list_length = lists.iter().map(String::len).max().unwrap_or(0);
        self.line(ORDPATTERN, &"X".repeat(max_ord_list_length));
        self.write_terms(&values.dict, width, max_length as i64);
        let mut next = values.docs.iter().zip(&lists).peekable();
        for i in 0..self.num_docs {
            let list = next.next_if(|v| *v.0 == i).map_or("", |v| v.1.as_str());
            write(&mut self.data, list.as_bytes());
            for _ in list.len()..max_ord_list_length {
                self.data.push(b' ');
            }
            write_newline(&mut self.data);
        }
        self.write_skip_data(&skip);
        Ok(())
    }

    /// `writeSkipData(intervals)`.
    fn write_skip_data(&mut self, intervals: &[SkipInterval]) {
        // Java: SimpleTextDocValuesWriter.writeSkipData
        write(&mut self.data, SKIP_DATA);
        write_newline(&mut self.data);
        for interval in intervals {
            write(&mut self.data, SKIP_INTERVAL);
            write_newline(&mut self.data);
            self.line(SKIP_MIN_DOC_ID, &interval.min_doc_id.to_string());
            self.line(SKIP_MAX_DOC_ID, &interval.max_doc_id.to_string());
            self.line(SKIP_MIN_VALUE, &interval.min_value.to_string());
            self.line(SKIP_MAX_VALUE, &interval.max_value.to_string());
            self.line(SKIP_DOC_COUNT, &interval.doc_count.to_string());
        }
        write(&mut self.data, SKIP_END);
        write_newline(&mut self.data);
    }

    /// `writeFieldEntry(field, type)`.
    fn write_field_entry(&mut self, field: &FieldInfo, ty: DocValuesType) {
        write_line(&mut self.data, FIELD, field.name.as_bytes());
        self.line(TYPE, doc_values_type_name(ty));
    }

    /// `close()`: `END` and the checksum.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] when no field was added (Java's `assert
    /// !fieldsSeen.isEmpty()`).
    pub fn finish(mut self) -> Result<Vec<u8>> {
        // Java: SimpleTextDocValuesWriter.close
        if self.fields_seen.is_empty() {
            return Err(Error::IllegalArgument(
                "a doc values file needs at least one field".into(),
            ));
        }
        write(&mut self.data, END);
        write_newline(&mut self.data);
        write_checksum(&mut self.data);
        Ok(self.data)
    }
}

/// Each binary value's `BytesRef.toString()`.
fn binary_strings(field: &str, values: &BinaryColumn) -> Result<Vec<String>> {
    if values.docs.len() != values.lengths.len() {
        return Err(Error::IllegalArgument(format!(
            "field \"{field}\": {} docs but {} lengths",
            values.docs.len(),
            values.lengths.len()
        )));
    }
    let mut out = Vec::with_capacity(values.lengths.len());
    let mut rest = values.bytes.as_slice();
    for &len in &values.lengths {
        let len = usize::try_from(len).unwrap_or(usize::MAX);
        let (v, tail) = rest.split_at_checked(len).ok_or_else(|| {
            Error::IllegalArgument(format!("field \"{field}\": lengths exceed the bytes"))
        })?;
        out.push(bytes_ref_to_string(v));
        rest = tail;
    }
    Ok(out)
}

/// A multi-valued column's values, per document.
fn split_counts<'v>(
    field: &str,
    docs: &[i32],
    counts: &[u32],
    values: &'v [i64],
) -> Result<Vec<&'v [i64]>> {
    let bad = || Error::IllegalArgument(format!("field \"{field}\": counts do not match values"));
    if docs.len() != counts.len() {
        return Err(bad());
    }
    let mut out = Vec::with_capacity(counts.len());
    let mut rest = values;
    for &count in counts {
        let (v, tail) = rest
            .split_at_checked(usize::try_from(count).unwrap_or(usize::MAX))
            .ok_or_else(bad)?;
        out.push(v);
        rest = tail;
    }
    if !rest.is_empty() {
        return Err(bad());
    }
    Ok(out)
}

fn check_ords(field: &str, ords: &[i64], dict_len: usize) -> Result<()> {
    if let Some(ord) = ords
        .iter()
        .find(|&&o| usize::try_from(o).map_or(true, |o| o >= dict_len))
    {
        return Err(Error::IllegalArgument(format!(
            "field \"{field}\": ord {ord} outside a dictionary of {dict_len}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field_infos::DocValuesSkipIndexType;

    fn fi(name: &str, ty: DocValuesType) -> FieldInfo {
        let mut f = FieldInfo::new(name, 0);
        f.doc_values_type = ty;
        f.doc_values_skip_index_type = DocValuesSkipIndexType::None;
        f
    }

    fn refused(r: Result<()>) {
        assert!(matches!(r, Err(Error::IllegalArgument(_))), "{r:?}");
    }

    #[test]
    fn add_field_dispatches_on_the_input() {
        let mut w = SimpleTextDocValuesWriter::new(3);
        let bin = BinaryColumn {
            docs: vec![0],
            lengths: vec![1],
            bytes: b"z".to_vec(),
        };
        let sorted = SortedOrdsColumn {
            dict: vec![b"a".to_vec()],
            docs: vec![1],
            ords: vec![0],
        };
        let snum = SortedNumericColumn {
            docs: vec![2],
            counts: vec![1],
            values: vec![4],
        };
        let sset = SortedSetOrdsColumn {
            dict: vec![b"a".to_vec()],
            docs: vec![0],
            counts: vec![1],
            ords: vec![0],
        };
        w.add_field(
            &fi("n", DocValuesType::Numeric),
            DocValuesInput::Numeric(&[(0, 1)]),
        )
        .unwrap();
        w.add_field(
            &fi("b", DocValuesType::Binary),
            DocValuesInput::Binary(&bin),
        )
        .unwrap();
        w.add_field(
            &fi("s", DocValuesType::Sorted),
            DocValuesInput::Sorted(&sorted),
        )
        .unwrap();
        w.add_field(
            &fi("sn", DocValuesType::SortedNumeric),
            DocValuesInput::SortedNumeric(&snum),
        )
        .unwrap();
        w.add_field(
            &fi("ss", DocValuesType::SortedSet),
            DocValuesInput::SortedSet(&sset),
        )
        .unwrap();
        let data = w.finish().unwrap();
        assert!(data.starts_with(FIELD) && data.windows(END.len()).any(|x| x == END));
        assert_eq!(
            doc_values_file_name("_0", "", DOC_VALUES_EXTENSION),
            "_0.dat"
        );
        assert_eq!(
            doc_values_file_name("_0", "x", DOC_VALUES_EXTENSION),
            "_0_x.dat"
        );
    }

    #[test]
    fn bad_input_is_refused() {
        let mut w = SimpleTextDocValuesWriter::new(3);
        // Wrong type (and not a normed field).
        refused(w.add_numeric_field(&fi("n", DocValuesType::Binary), &[(0, 1)]));
        // Out of order, repeated, outside maxDoc.
        refused(w.add_numeric_field(&fi("n", DocValuesType::Numeric), &[(1, 1), (0, 1)]));
        refused(w.add_numeric_field(&fi("n", DocValuesType::Numeric), &[(1, 1), (1, 1)]));
        refused(w.add_numeric_field(&fi("n", DocValuesType::Numeric), &[(3, 1)]));
        refused(w.add_numeric_field(&fi("n", DocValuesType::Numeric), &[(-1, 1)]));
        w.add_numeric_field(&fi("n", DocValuesType::Numeric), &[])
            .unwrap();
        refused(w.add_numeric_field(&fi("n", DocValuesType::Numeric), &[]));
        let b = fi("b", DocValuesType::Binary);
        refused(w.add_binary_field(
            &b,
            &BinaryColumn {
                docs: vec![0],
                lengths: vec![],
                bytes: vec![],
            },
        ));
        refused(w.add_binary_field(
            &b,
            &BinaryColumn {
                docs: vec![0],
                lengths: vec![2],
                bytes: vec![1],
            },
        ));
        refused(w.add_binary_field(&fi("b", DocValuesType::Numeric), &BinaryColumn::default()));
        let s = fi("s", DocValuesType::Sorted);
        refused(w.add_sorted_field(
            &s,
            &SortedOrdsColumn {
                dict: vec![],
                docs: vec![0],
                ords: vec![],
            },
        ));
        refused(w.add_sorted_field(
            &s,
            &SortedOrdsColumn {
                dict: vec![b"a".to_vec()],
                docs: vec![0],
                ords: vec![1],
            },
        ));
        refused(w.add_sorted_field(
            &s,
            &SortedOrdsColumn {
                dict: vec![b"a".to_vec()],
                docs: vec![0],
                ords: vec![-1],
            },
        ));
        let sn = fi("sn", DocValuesType::SortedNumeric);
        refused(w.add_sorted_numeric_field(
            &sn,
            &SortedNumericColumn {
                docs: vec![0],
                counts: vec![],
                values: vec![],
            },
        ));
        refused(w.add_sorted_numeric_field(
            &sn,
            &SortedNumericColumn {
                docs: vec![0],
                counts: vec![2],
                values: vec![1],
            },
        ));
        refused(w.add_sorted_numeric_field(
            &sn,
            &SortedNumericColumn {
                docs: vec![0],
                counts: vec![1],
                values: vec![1, 2],
            },
        ));
        let ss = fi("ss", DocValuesType::SortedSet);
        refused(w.add_sorted_set_field(
            &ss,
            &SortedSetOrdsColumn {
                dict: vec![],
                docs: vec![0],
                counts: vec![1],
                ords: vec![0],
            },
        ));
        refused(w.add_sorted_set_field(
            &ss,
            &SortedSetOrdsColumn {
                dict: vec![],
                docs: vec![0],
                counts: vec![],
                ords: vec![],
            },
        ));
        assert!(
            SimpleTextDocValuesWriter::new(1).finish().is_err(),
            "no field"
        );
    }

    #[test]
    fn a_normed_field_may_write_numerics() {
        let mut f = fi("body", DocValuesType::None);
        f.index_options = IndexOptions::DocsAndFreqs;
        let mut w = SimpleTextDocValuesWriter::new(2);
        w.add_numeric_field(&f, &[(0, 3), (1, 1)]).unwrap();
        f.omit_norms = true;
        f.name = "omitted".into();
        refused(w.add_numeric_field(&f, &[(0, 3)]));
        w.finish().unwrap();
    }
}
