//! `SimpleTextDocValuesReader`: the `.dat` (and norms' `.len`) written by
//! [`super::doc_values_writer`]. Opening walks each field's header lines and
//! seeks past its fixed-width records (`OneField`); an iterator then seeks
//! straight to document `i`'s record at `dataStart + i * width`, and a
//! dictionary lookup to term `ord`'s at `dataStart + ord * width`.
//!
//! What differs from Java:
//! - The reader borrows the file's bytes, not an `IndexInput`; fields are
//!   looked up by name (Java takes a `FieldInfo` and reads only its name,
//!   and [`SimpleTextDocValuesReader::skipper`] its doc-values type).
//!   Cached headers can be rebound to the same file's independently owned
//!   bytes; `*_from_data` borrows those bytes together with the headers.
//!   Iterators are typed structs, not anonymous `DocValues` subclasses; a
//!   [`Error::IllegalArgument`] stands for Java's `IndexOutOfBoundsException`
//!   (a document or ord outside the field) and for the `NullPointerException`
//!   of an unknown field; `nextDoc` after `NO_MORE_DOCS` stays there (Java's
//!   `docID() + 1` overflows into a negative seek). `advanceExact` outside
//!   `0..maxDoc` is `IllegalArgument` too, where Java just seeks and reads
//!   whatever lies there.
//! - Every `assert` on a line's prefix and every `NumberFormatException`,
//!   `ParseException`, `ClassCastException` (a `NaN` or `∞` where a
//!   `BigDecimal` was expected), `ArithmeticException` (a fraction),
//!   `NegativeArraySizeException`, array index error and seek past the end
//!   (or backwards) is `Corrupted` (or `Eof`); a pattern that is not all `0`s is refused
//!   when an iterator over it is built ([`decimal_format::check_pattern`]).
//!   A file with no field (`END` alone) is `Corrupted` too: Java's reader
//!   accepts it, but its writer never writes one (an `assert`).
//! - `lookupTerm` returns `Ok(ord)` or `Err(insertion point)` instead of
//!   `-(insertionPoint + 1)`; `termsEnum()` is
//!   [`SimpleTextDocValuesTermsEnum`] (Java's `SortedDocValuesTermsEnum` /
//!   `SortedSetDocValuesTermsEnum`, without `docFreq`/`postings`).

use std::collections::HashMap;

use super::decimal_format;
use super::doc_values_writer::{
    SkipInterval, DOCCOUNT, END, FIELD, LENGTH, MAXLENGTH, MAXVALUE, MAXVALUECOUNT, MINVALUE,
    NUMVALUES, ORDPATTERN, ORIGIN, PATTERN, SKIP_DATA, SKIP_DOC_COUNT, SKIP_END, SKIP_INTERVAL,
    SKIP_MAX_DOC_ID, SKIP_MAX_VALUE, SKIP_MIN_DOC_ID, SKIP_MIN_VALUE, TYPE,
};
use super::field_infos::doc_values_type;
use super::util::{
    self, from_bytes_ref_string, parse_integer, parse_long, read_line, read_value, tail,
};
use super::{corrupt, file_pointer, Error, Result, SeekStatus, NO_MORE_DOCS};
use crate::field_infos::DocValuesType;

/// `SimpleTextDocValuesReader.OneField`: one field's header.
#[derive(Debug, Clone, PartialEq)]
pub struct OneField {
    /// The type its `type` line names.
    pub doc_values_type: DocValuesType,
    /// `docCount`.
    pub doc_count: i32,
    /// `dataStartFilePointer`.
    pub data_start: usize,
    /// `pattern`.
    pub pattern: String,
    /// `ordPattern` (SORTED, SORTED_SET).
    pub ord_pattern: String,
    /// `maxLength`.
    pub max_length: i32,
    /// `origin` (NUMERIC).
    pub origin: i64,
    /// `minValue` (NUMERIC, SORTED_NUMERIC).
    pub min_value: i64,
    /// `maxValue` (NUMERIC, SORTED_NUMERIC).
    pub max_value: i64,
    /// `maxValueCount`.
    pub max_value_count: i32,
    /// `numValues` (SORTED, SORTED_SET).
    pub num_values: i64,
    /// `skipIntervals`.
    pub skip_intervals: Vec<SkipInterval>,
}

impl OneField {
    fn new(doc_values_type: DocValuesType) -> Self {
        OneField {
            doc_values_type,
            doc_count: 0,
            data_start: 0,
            pattern: String::new(),
            ord_pattern: String::new(),
            max_length: 0,
            origin: 0,
            min_value: 0,
            max_value: 0,
            max_value_count: 0,
            num_values: 0,
            skip_intervals: Vec::new(),
        }
    }

    /// `SimpleTextDocValuesReader.isSingleValued(field)` (Lucene 10.6,
    /// branch_10_6, #16714): every document has at most one value, by the
    /// writer's recorded `maxValueCount`, so `getSortedNumeric`/`getSortedSet`
    /// expose the column as a singleton (`DocValues.singleton`), as the
    /// default codec does. An unknown count (`-1`, a pre-`maxValueCount`
    /// file) is not single-valued.
    pub fn is_single_valued(&self) -> bool {
        (0..=1).contains(&self.max_value_count)
    }

    /// `pattern.length()` (UTF-16 units).
    fn pattern_len(&self) -> i64 {
        java_len(&self.pattern)
    }

    /// NUMERIC: `1 + pattern.length() + 2`.
    fn numeric_width(&self) -> i64 {
        self.pattern_len().saturating_add(3)
    }

    /// BINARY, SORTED_NUMERIC: `9 + pattern.length() + maxLength + 2`.
    fn binary_width(&self) -> i64 {
        self.pattern_len()
            .saturating_add(i64::from(self.max_length))
            .saturating_add(11)
    }

    /// SORTED, SORTED_SET dictionary: `9 + pattern.length() + maxLength`.
    fn term_width(&self) -> i64 {
        self.pattern_len()
            .saturating_add(i64::from(self.max_length))
            .saturating_add(9)
    }

    /// SORTED, SORTED_SET ord line: `1 + ordPattern.length()`.
    fn ord_width(&self) -> i64 {
        java_len(&self.ord_pattern).saturating_add(1)
    }
}

/// `String.length()`.
fn java_len(s: &str) -> i64 {
    i64::try_from(s.encode_utf16().count()).unwrap_or(i64::MAX)
}

/// `start + a * b` (Java's `long` arithmetic) as a position in `data`.
///
/// A negative `a * b` (a negative width or count read off the file) is
/// refused: a backward seek could re-read an earlier field's lines, and
/// `open` would then never reach `END`.
fn seek(data: &[u8], start: usize, a: i64, b: i64) -> Result<usize> {
    let offset = a
        .checked_mul(b)
        .ok_or_else(|| corrupt("seek position overflows"))?;
    if offset < 0 {
        return Err(corrupt(format!("negative seek offset {offset}")));
    }
    let fp = i64::try_from(start)
        .ok()
        .and_then(|s| s.checked_add(offset))
        .ok_or_else(|| corrupt("seek position overflows"))?;
    file_pointer(data, fp, "seek")
}

/// A `maxlength` below -1 (Java's writer starts it at 0) can make a record
/// width negative; no writer produces one.
fn check_max_length(max_length: i32) -> Result<()> {
    if max_length < -1 {
        return Err(corrupt(format!("maxlength {max_length} below -1")));
    }
    Ok(())
}

/// `SimpleTextDocValuesReader` over the whole file.
#[derive(Debug, Clone)]
pub struct SimpleTextDocValuesReader<'d> {
    data: &'d [u8],
    max_doc: i32,
    fields: HashMap<String, OneField>,
}

impl<'d> SimpleTextDocValuesReader<'d> {
    /// `new SimpleTextDocValuesReader(state, ext)` over a segment of
    /// `max_doc` documents: every field's header (the checksum is
    /// [`Self::check_integrity`]'s, as Java's).
    ///
    /// # Errors
    /// `Corrupted` for a malformed or misplaced line, a number that does
    /// not parse or a seek past the end; `Eof` for a short file.
    pub fn open(data: &'d [u8], max_doc: i32) -> Result<Self> {
        // Java: SimpleTextDocValuesReader.<init>
        let mut fields = HashMap::new();
        let mut scratch = Vec::new();
        let mut pos = 0usize;
        loop {
            read_line(data, &mut pos, &mut scratch)?;
            if scratch == END {
                break;
            }
            let name = String::from_utf8_lossy(
                tail(&scratch, FIELD).ok_or_else(|| bad_line(&scratch, "field"))?,
            )
            .into_owned();
            let ty = String::from_utf8_lossy(read_value(data, &mut pos, &mut scratch, TYPE)?)
                .into_owned();
            let ty = doc_values_type(&ty).map_err(|e| corrupt(e.to_string()))?;
            let mut field = OneField::new(ty);
            if matches!(ty, DocValuesType::Numeric | DocValuesType::SortedNumeric) {
                field.min_value = parse_long(read_value(data, &mut pos, &mut scratch, MINVALUE)?)?;
                field.max_value = parse_long(read_value(data, &mut pos, &mut scratch, MAXVALUE)?)?;
            }
            field.doc_count = parse_integer(read_value(data, &mut pos, &mut scratch, DOCCOUNT)?)?;
            read_line(data, &mut pos, &mut scratch)?;
            if let Some(v) = tail(&scratch, MAXVALUECOUNT) {
                field.max_value_count = parse_integer(v)?;
                read_line(data, &mut pos, &mut scratch)?;
            } else if field.doc_count == 0 {
                field.max_value_count = 0;
            } else if matches!(ty, DocValuesType::Numeric | DocValuesType::Sorted) {
                field.max_value_count = 1;
            } else {
                field.max_value_count = -1;
            }
            let max_doc_l = i64::from(max_doc);
            match ty {
                DocValuesType::Numeric => {
                    field.origin = parse_long(expect(&scratch, ORIGIN)?)?;
                    field.pattern = read_str(data, &mut pos, &mut scratch, PATTERN)?;
                    field.data_start = pos;
                    pos = seek(data, pos, field.numeric_width(), max_doc_l)?;
                    field.skip_intervals = read_skip_data(data, &mut pos, &mut scratch)?;
                }
                DocValuesType::Binary | DocValuesType::SortedNumeric => {
                    field.max_length = parse_integer(expect(&scratch, MAXLENGTH)?)?;
                    check_max_length(field.max_length)?;
                    field.pattern = read_str(data, &mut pos, &mut scratch, PATTERN)?;
                    field.data_start = pos;
                    pos = seek(data, pos, field.binary_width(), max_doc_l)?;
                    if ty == DocValuesType::SortedNumeric {
                        field.skip_intervals = read_skip_data(data, &mut pos, &mut scratch)?;
                    }
                }
                DocValuesType::Sorted | DocValuesType::SortedSet => {
                    field.num_values = parse_long(expect(&scratch, NUMVALUES)?)?;
                    if field.num_values < 0 {
                        return Err(corrupt(format!("negative numvalues {}", field.num_values)));
                    }
                    field.max_length =
                        parse_integer(read_value(data, &mut pos, &mut scratch, MAXLENGTH)?)?;
                    check_max_length(field.max_length)?;
                    field.pattern = read_str(data, &mut pos, &mut scratch, PATTERN)?;
                    field.ord_pattern = read_str(data, &mut pos, &mut scratch, ORDPATTERN)?;
                    field.data_start = pos;
                    let terms = seek(data, pos, field.term_width(), field.num_values)?;
                    pos = seek(data, terms, field.ord_width(), max_doc_l)?;
                    field.skip_intervals = read_skip_data(data, &mut pos, &mut scratch)?;
                }
                DocValuesType::None => return Err(corrupt("doc values type NONE")),
            }
            fields.insert(name, field);
        }
        if fields.is_empty() {
            return Err(corrupt("a doc values file without fields"));
        }
        Ok(SimpleTextDocValuesReader {
            data,
            max_doc,
            fields,
        })
    }

    /// The segment's `maxDoc`.
    pub fn max_doc(&self) -> i32 {
        self.max_doc
    }

    /// Bind the parsed field headers to the same file's bytes under a new
    /// lifetime. An empty binding can retain metadata in a segment core;
    /// the `*_from_data` factories then take the live file bytes explicitly.
    pub fn rebind<'n>(&self, data: &'n [u8]) -> SimpleTextDocValuesReader<'n> {
        SimpleTextDocValuesReader {
            data,
            max_doc: self.max_doc,
            fields: self.fields.clone(),
        }
    }

    /// The field named `name`'s header.
    pub fn field(&self, name: &str) -> Option<&OneField> {
        self.fields.get(name)
    }

    /// `fields.size()`.
    pub fn num_fields(&self) -> usize {
        self.fields.len()
    }

    fn one_field(&self, name: &str) -> Result<&OneField> {
        self.fields
            .get(name)
            .ok_or_else(|| Error::IllegalArgument(format!("no doc values for field \"{name}\"")))
    }

    fn cursor<'r>(&'r self, name: &str, data: &'r [u8]) -> Result<Cursor<'r>> {
        Ok(Cursor {
            data,
            max_doc: self.max_doc,
            field: self.one_field(name)?,
            doc: -1,
            scratch: Vec::new(),
        })
    }

    /// `getNumeric(field)`.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for an unknown field; `Corrupted` for a
    /// pattern that is not all `0`s.
    pub fn numeric(&self, name: &str) -> Result<SimpleTextNumericDocValues<'_>> {
        self.numeric_from_data(name, self.data)
    }

    /// [`Self::numeric`] using the same file's live bytes with cached headers.
    ///
    /// # Errors
    /// As [`Self::numeric`].
    pub fn numeric_from_data<'r>(
        &'r self,
        name: &str,
        data: &'r [u8],
    ) -> Result<SimpleTextNumericDocValues<'r>> {
        // Java: SimpleTextDocValuesReader.getNumeric
        let c = self.cursor(name, data)?;
        decimal_format::check_pattern(&c.field.pattern)?;
        Ok(SimpleTextNumericDocValues { c })
    }

    /// `getBinary(field)`.
    ///
    /// # Errors
    /// As [`Self::numeric`].
    pub fn binary(&self, name: &str) -> Result<SimpleTextBinaryDocValues<'_>> {
        self.binary_from_data(name, self.data)
    }

    /// [`Self::binary`] using the same file's live bytes with cached headers.
    ///
    /// # Errors
    /// As [`Self::binary`].
    pub fn binary_from_data<'r>(
        &'r self,
        name: &str,
        data: &'r [u8],
    ) -> Result<SimpleTextBinaryDocValues<'r>> {
        // Java: SimpleTextDocValuesReader.getBinary
        let c = self.cursor(name, data)?;
        decimal_format::check_pattern(&c.field.pattern)?;
        Ok(SimpleTextBinaryDocValues { c })
    }

    /// `getSorted(field)`.
    ///
    /// # Errors
    /// As [`Self::numeric`], for the pattern and the ord pattern.
    pub fn sorted(&self, name: &str) -> Result<SimpleTextSortedDocValues<'_>> {
        self.sorted_from_data(name, self.data)
    }

    /// [`Self::sorted`] using the same file's live bytes with cached headers.
    ///
    /// # Errors
    /// As [`Self::sorted`].
    pub fn sorted_from_data<'r>(
        &'r self,
        name: &str,
        data: &'r [u8],
    ) -> Result<SimpleTextSortedDocValues<'r>> {
        // Java: SimpleTextDocValuesReader.getSorted
        let c = self.cursor(name, data)?;
        decimal_format::check_pattern(&c.field.pattern)?;
        decimal_format::check_pattern(&c.field.ord_pattern)?;
        Ok(SimpleTextSortedDocValues { c, ord: -1 })
    }

    /// `getSortedNumeric(field)`: the BINARY view of each document's CSV.
    ///
    /// # Errors
    /// As [`Self::numeric`].
    pub fn sorted_numeric(&self, name: &str) -> Result<SimpleTextSortedNumericDocValues<'_>> {
        self.sorted_numeric_from_data(name, self.data)
    }

    /// [`Self::sorted_numeric`] with cached headers and the same file's bytes.
    ///
    /// # Errors
    /// As [`Self::sorted_numeric`].
    pub fn sorted_numeric_from_data<'r>(
        &'r self,
        name: &str,
        data: &'r [u8],
    ) -> Result<SimpleTextSortedNumericDocValues<'r>> {
        // Java: SimpleTextDocValuesReader.getSortedNumeric
        Ok(SimpleTextSortedNumericDocValues {
            binary: self.binary_from_data(name, data)?,
            values: Vec::new(),
            index: 0,
        })
    }

    /// `getSortedSet(field)`.
    ///
    /// # Errors
    /// As [`Self::numeric`].
    pub fn sorted_set(&self, name: &str) -> Result<SimpleTextSortedSetDocValues<'_>> {
        self.sorted_set_from_data(name, self.data)
    }

    /// [`Self::sorted_set`] with cached headers and the same file's bytes.
    ///
    /// # Errors
    /// As [`Self::sorted_set`].
    pub fn sorted_set_from_data<'r>(
        &'r self,
        name: &str,
        data: &'r [u8],
    ) -> Result<SimpleTextSortedSetDocValues<'r>> {
        // Java: SimpleTextDocValuesReader.getSortedSet
        let c = self.cursor(name, data)?;
        decimal_format::check_pattern(&c.field.pattern)?;
        Ok(SimpleTextSortedSetDocValues {
            c,
            ord_list: String::new(),
            parts: Vec::new(),
            index: 0,
        })
    }

    /// `getSkipper(fieldInfo)`, `ty` being the field's doc-values type.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for an unknown field.
    pub fn skipper(&self, name: &str, ty: DocValuesType) -> Result<SimpleTextSkipper<'_>> {
        // Java: SimpleTextDocValuesReader.getSkipper
        Ok(SimpleTextSkipper {
            numeric: matches!(ty, DocValuesType::Numeric | DocValuesType::SortedNumeric),
            field: self.one_field(name)?,
            cursor: -1,
        })
    }

    /// `checkIntegrity()`: every line up to where the checksum line must
    /// start, then the checksum.
    ///
    /// # Errors
    /// `Corrupted` for a footer elsewhere or a wrong checksum.
    pub fn check_integrity(&self) -> Result<()> {
        // Java: SimpleTextDocValuesReader.checkIntegrity
        util::check_integrity(self.data)
    }
}

fn bad_line(line: &[u8], want: &str) -> Error {
    corrupt(format!(
        "SimpleText failure: expected {want} line but got {:?}",
        String::from_utf8_lossy(line)
    ))
}

/// `assert startsWith(prefix)` on the line already read, and its tail.
fn expect<'s>(line: &'s [u8], prefix: &[u8]) -> Result<&'s [u8]> {
    tail(line, prefix).ok_or_else(|| bad_line(line, &String::from_utf8_lossy(prefix)))
}

fn read_str(data: &[u8], pos: &mut usize, scratch: &mut Vec<u8>, prefix: &[u8]) -> Result<String> {
    util::read_string(data, pos, scratch, prefix)
}

/// `readSkipData()`.
fn read_skip_data(
    data: &[u8],
    pos: &mut usize,
    scratch: &mut Vec<u8>,
) -> Result<Vec<SkipInterval>> {
    // Java: SimpleTextDocValuesReader.readSkipData
    read_line(data, pos, scratch)?;
    expect(scratch, SKIP_DATA)?;
    let mut intervals = Vec::new();
    loop {
        read_line(data, pos, scratch)?;
        if tail(scratch, SKIP_END).is_some() {
            return Ok(intervals);
        }
        expect(scratch, SKIP_INTERVAL)?;
        let min_doc_id = parse_integer(read_value(data, pos, scratch, SKIP_MIN_DOC_ID)?)?;
        let max_doc_id = parse_integer(read_value(data, pos, scratch, SKIP_MAX_DOC_ID)?)?;
        let min_value = parse_long(read_value(data, pos, scratch, SKIP_MIN_VALUE)?)?;
        let max_value = parse_long(read_value(data, pos, scratch, SKIP_MAX_VALUE)?)?;
        let doc_count = parse_integer(read_value(data, pos, scratch, SKIP_DOC_COUNT)?)?;
        intervals.push(SkipInterval {
            min_doc_id,
            max_doc_id,
            min_value,
            max_value,
            doc_count,
        });
    }
}

/// `decoder.parse(new String(line, LENGTH.length, …)).intValue()` on a
/// `length` line.
fn parse_length(line: &[u8]) -> Result<i32> {
    let v = expect(line, LENGTH)?;
    Ok(decimal_format::parse(&String::from_utf8_lossy(v))?.int_value())
}

/// A `length` line's value as a byte count.
fn length_bytes(len: i32) -> Result<usize> {
    usize::try_from(len).map_err(|_| corrupt(format!("negative value length {len}")))
}

/// The `len` raw bytes at `*pos`.
fn raw<'d>(data: &'d [u8], pos: &mut usize, len: usize) -> Result<&'d [u8]> {
    let end = pos
        .checked_add(len)
        .filter(|&e| e <= data.len())
        .ok_or(lucene_store::Error::Eof { offset: *pos })?;
    let v = data.get(*pos..end).unwrap_or_default();
    *pos = end;
    Ok(v)
}

/// What every iterator holds: the file, the field, the current document.
#[derive(Debug, Clone)]
struct Cursor<'r> {
    data: &'r [u8],
    max_doc: i32,
    field: &'r OneField,
    doc: i32,
    scratch: Vec<u8>,
}

impl Cursor<'_> {
    /// `if (docID < 0 || docID >= maxDoc) throw IndexOutOfBoundsException`.
    fn check_doc(&self, doc: i32) -> Result<()> {
        if doc < 0 || doc >= self.max_doc {
            return Err(Error::IllegalArgument(format!(
                "docID must be 0 .. {}; got {doc}",
                i64::from(self.max_doc).saturating_sub(1)
            )));
        }
        Ok(())
    }

    /// `docID() + 1`, not past `NO_MORE_DOCS`.
    fn next_target(&self) -> i32 {
        self.doc.saturating_add(1)
    }

    /// The NUMERIC `T`/`F` line of document `i`.
    fn numeric_has(&mut self, i: i32) -> Result<bool> {
        let mut pos = seek(
            self.data,
            self.field.data_start,
            self.field.numeric_width(),
            i64::from(i),
        )?;
        read_line(self.data, &mut pos, &mut self.scratch)?; // data
        read_line(self.data, &mut pos, &mut self.scratch)?; // 'T' or 'F'
        Ok(self.scratch.first() == Some(&b'T'))
    }

    /// The BINARY `T`/`F` line of document `i`, past its value.
    fn binary_has(&mut self, i: i32) -> Result<bool> {
        let data = self.data;
        let mut pos = seek(
            data,
            self.field.data_start,
            self.field.binary_width(),
            i64::from(i),
        )?;
        read_line(data, &mut pos, &mut self.scratch)?;
        let len = length_bytes(parse_length(&self.scratch)?)?;
        raw(data, &mut pos, len)?;
        read_line(data, &mut pos, &mut self.scratch)?; // newline
        read_line(data, &mut pos, &mut self.scratch)?; // 'T' or 'F'
        Ok(self.scratch.first() == Some(&b'T'))
    }

    /// The SORTED/SORTED_SET ord line of document `i`, into `scratch`.
    fn read_ord_line(&mut self, i: i32) -> Result<()> {
        let f = self.field;
        let terms = seek(self.data, f.data_start, f.term_width(), f.num_values)?;
        let mut pos = seek(self.data, terms, f.ord_width(), i64::from(i))?;
        read_line(self.data, &mut pos, &mut self.scratch)
    }

    /// The `advance` loop of every iterator: the first `i >= target` that
    /// `has` accepts, or `NO_MORE_DOCS`.
    fn advance_by(
        &mut self,
        target: i32,
        mut has: impl FnMut(&mut Self, i32) -> Result<bool>,
    ) -> Result<i32> {
        let mut i = target;
        while i < self.max_doc {
            if has(self, i)? {
                self.doc = i;
                return Ok(i);
            }
            i = i.saturating_add(1);
        }
        self.doc = NO_MORE_DOCS;
        Ok(NO_MORE_DOCS)
    }
}

/// `getNumeric`'s `NumericDocValues` (`getNumericNonIterator` over
/// `getNumericDocsWithField`).
#[derive(Debug, Clone)]
pub struct SimpleTextNumericDocValues<'r> {
    c: Cursor<'r>,
}

impl SimpleTextNumericDocValues<'_> {
    /// `docID()`.
    pub fn doc_id(&self) -> i32 {
        self.c.doc
    }

    /// `cost()`: `maxDoc`.
    pub fn cost(&self) -> i64 {
        i64::from(self.c.max_doc)
    }

    /// `nextDoc()`.
    ///
    /// # Errors
    /// `Corrupted`/`Eof` for a record that does not read.
    pub fn next_doc(&mut self) -> Result<i32> {
        self.advance(self.c.next_target())
    }

    /// `advance(target)`.
    ///
    /// # Errors
    /// As [`Self::next_doc`].
    pub fn advance(&mut self, target: i32) -> Result<i32> {
        // Java: SimpleTextDocValuesReader.getNumericDocsWithField.advance
        self.c.advance_by(target, Cursor::numeric_has)
    }

    /// `advanceExact(target)`.
    ///
    /// # Errors
    /// As [`Self::next_doc`]; [`Error::IllegalArgument`] outside the segment.
    pub fn advance_exact(&mut self, target: i32) -> Result<bool> {
        self.c.check_doc(target)?;
        self.c.doc = target;
        self.c.numeric_has(target)
    }

    /// `longValue()`: `origin + delta` of the current document.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] when not on a document; `Corrupted` for a
    /// value that does not parse as an integral `BigDecimal`.
    pub fn long_value(&mut self) -> Result<i64> {
        // Java: SimpleTextDocValuesReader.getNumericNonIterator.apply
        let c = &mut self.c;
        c.check_doc(c.doc)?;
        let mut pos = seek(
            c.data,
            c.field.data_start,
            c.field.numeric_width(),
            i64::from(c.doc),
        )?;
        read_line(c.data, &mut pos, &mut c.scratch)?;
        let bd = decimal_format::parse(&String::from_utf8_lossy(&c.scratch))?;
        read_line(c.data, &mut pos, &mut c.scratch)?; // real or not
        bd.to_long_exact_wrapping(c.field.origin)
    }
}

/// `getBinary`'s `BinaryDocValues`.
#[derive(Debug, Clone)]
pub struct SimpleTextBinaryDocValues<'r> {
    c: Cursor<'r>,
}

impl SimpleTextBinaryDocValues<'_> {
    /// `docID()`.
    pub fn doc_id(&self) -> i32 {
        self.c.doc
    }

    /// `cost()`: `maxDoc`.
    pub fn cost(&self) -> i64 {
        i64::from(self.c.max_doc)
    }

    /// `nextDoc()`.
    ///
    /// # Errors
    /// `Corrupted`/`Eof` for a record that does not read.
    pub fn next_doc(&mut self) -> Result<i32> {
        self.advance(self.c.next_target())
    }

    /// `advance(target)`.
    ///
    /// # Errors
    /// As [`Self::next_doc`].
    pub fn advance(&mut self, target: i32) -> Result<i32> {
        // Java: SimpleTextDocValuesReader.getBinaryDocsWithField.advance
        self.c.advance_by(target, Cursor::binary_has)
    }

    /// `advanceExact(target)`.
    ///
    /// # Errors
    /// As [`Self::next_doc`]; [`Error::IllegalArgument`] outside the segment.
    pub fn advance_exact(&mut self, target: i32) -> Result<bool> {
        self.c.check_doc(target)?;
        self.c.doc = target;
        self.c.binary_has(target)
    }

    /// `binaryValue()`: the current document's value, decoded from its
    /// `BytesRef.toString()`.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] when not on a document or for a value not
    /// in `BytesRef.toString` form; `Corrupted`/`Eof` for a bad record.
    pub fn binary_value(&mut self) -> Result<Vec<u8>> {
        // Java: SimpleTextDocValuesReader.getBinary.apply
        let c = &mut self.c;
        c.check_doc(c.doc)?;
        let data = c.data;
        let mut pos = seek(
            data,
            c.field.data_start,
            c.field.binary_width(),
            i64::from(c.doc),
        )?;
        read_line(data, &mut pos, &mut c.scratch)?;
        let len = length_bytes(parse_length(&c.scratch)?)?;
        let v = raw(data, &mut pos, len)?;
        from_bytes_ref_string(&String::from_utf8_lossy(v))
    }
}

/// `lookupOrd(ord)` of a SORTED/SORTED_SET field: the term's raw bytes.
fn lookup_ord<'d>(
    data: &'d [u8],
    field: &OneField,
    ord: i64,
    scratch: &mut Vec<u8>,
) -> Result<&'d [u8]> {
    if ord < 0 || ord >= field.num_values {
        return Err(Error::IllegalArgument(format!(
            "ord must be 0 .. {}; got {ord}",
            field.num_values.saturating_sub(1)
        )));
    }
    let mut pos = seek(data, field.data_start, field.term_width(), ord)?;
    read_line(data, &mut pos, scratch)?;
    let len = length_bytes(parse_length(scratch)?)?;
    raw(data, &mut pos, len)
}

/// `SortedDocValues.lookupTerm` / `SortedSetDocValues.lookupTerm`: a binary
/// search over `0..count` with `lookupOrd`.
fn lookup_term(
    data: &[u8],
    field: &OneField,
    count: i64,
    key: &[u8],
) -> Result<std::result::Result<i64, i64>> {
    let mut scratch = Vec::new();
    let mut low = 0i64;
    let mut high = count.saturating_sub(1);
    while low <= high {
        // `(low + high) >>> 1`, for 0 <= low <= high.
        let mid = low.saturating_add(high.saturating_sub(low) / 2);
        let term = lookup_ord(data, field, mid, &mut scratch)?;
        match term.cmp(key) {
            std::cmp::Ordering::Less => low = mid.saturating_add(1),
            std::cmp::Ordering::Greater => high = mid.saturating_sub(1),
            std::cmp::Ordering::Equal => return Ok(Ok(mid)),
        }
    }
    Ok(Err(low))
}

/// `getSorted`'s `SortedDocValues`.
#[derive(Debug, Clone)]
pub struct SimpleTextSortedDocValues<'r> {
    c: Cursor<'r>,
    ord: i32,
}

impl<'r> SimpleTextSortedDocValues<'r> {
    /// `docID()`.
    pub fn doc_id(&self) -> i32 {
        self.c.doc
    }

    /// `cost()`: `maxDoc`.
    pub fn cost(&self) -> i64 {
        i64::from(self.c.max_doc)
    }

    /// `(int) ordDecoder.parse(line).longValue() - 1` of document `i`.
    fn read_ord(c: &mut Cursor<'_>, i: i32) -> Result<i32> {
        c.read_ord_line(i)?;
        let parsed = decimal_format::parse(&String::from_utf8_lossy(&c.scratch))?;
        Ok((parsed.long_value() as i32).wrapping_sub(1))
    }

    /// `nextDoc()`.
    ///
    /// # Errors
    /// `Corrupted`/`Eof` for an ord line that does not read.
    pub fn next_doc(&mut self) -> Result<i32> {
        self.advance(self.c.next_target())
    }

    /// `advance(target)`.
    ///
    /// # Errors
    /// As [`Self::next_doc`].
    pub fn advance(&mut self, target: i32) -> Result<i32> {
        // Java: SimpleTextDocValuesReader.getSorted.advance
        let mut ord = self.ord;
        let doc = self.c.advance_by(target, |c, i| {
            ord = Self::read_ord(c, i)?;
            Ok(ord >= 0)
        })?;
        self.ord = ord;
        Ok(doc)
    }

    /// `advanceExact(target)`.
    ///
    /// # Errors
    /// As [`Self::next_doc`]; [`Error::IllegalArgument`] outside the segment.
    pub fn advance_exact(&mut self, target: i32) -> Result<bool> {
        self.c.check_doc(target)?;
        self.c.doc = target;
        self.ord = Self::read_ord(&mut self.c, target)?;
        Ok(self.ord >= 0)
    }

    /// `ordValue()`.
    pub fn ord_value(&self) -> i32 {
        self.ord
    }

    /// `lookupOrd(ord)`.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for an ord outside the dictionary;
    /// `Corrupted`/`Eof` for a bad term record.
    pub fn lookup_ord(&mut self, ord: i32) -> Result<&'r [u8]> {
        lookup_ord(
            self.c.data,
            self.c.field,
            i64::from(ord),
            &mut self.c.scratch,
        )
    }

    /// `getValueCount()`: `(int) numValues`.
    pub fn value_count(&self) -> i32 {
        self.c.field.num_values as i32
    }

    /// `lookupTerm(key)`: `Ok(ord)`, or `Err(insertion point)`.
    ///
    /// # Errors
    /// What [`Self::lookup_ord`] fails with.
    pub fn lookup_term(&self, key: &[u8]) -> Result<std::result::Result<i32, i32>> {
        // SENTINEL-OK: this module's `lookup_term` returns `Ok(ord)` or
        // `Err(insertion point)`; it has no `-1` sentinel.
        let r = lookup_term(
            self.c.data,
            self.c.field,
            i64::from(self.value_count()),
            key,
        )?;
        Ok(r.map(|o| o as i32).map_err(|o| o as i32))
    }

    /// `termsEnum()`.
    pub fn terms_enum(&self) -> SimpleTextDocValuesTermsEnum<'r> {
        SimpleTextDocValuesTermsEnum::new(self.c.data, self.c.field, i64::from(self.value_count()))
    }
}

/// `getSortedNumeric`'s `SortedNumericDocValues`.
#[derive(Debug, Clone)]
pub struct SimpleTextSortedNumericDocValues<'r> {
    binary: SimpleTextBinaryDocValues<'r>,
    values: Vec<i64>,
    index: usize,
}

impl SimpleTextSortedNumericDocValues<'_> {
    /// `docID()`.
    pub fn doc_id(&self) -> i32 {
        self.binary.doc_id()
    }

    /// `cost()`.
    pub fn cost(&self) -> i64 {
        self.binary.cost()
    }

    /// `nextDoc()`.
    ///
    /// # Errors
    /// What the BINARY view fails with; `Corrupted` for a CSV part that is
    /// not a `long`.
    pub fn next_doc(&mut self) -> Result<i32> {
        let doc = self.binary.next_doc()?;
        self.set_current_doc()?;
        Ok(doc)
    }

    /// `advance(target)`.
    ///
    /// # Errors
    /// As [`Self::next_doc`].
    pub fn advance(&mut self, target: i32) -> Result<i32> {
        let doc = self.binary.advance(target)?;
        self.set_current_doc()?;
        Ok(doc)
    }

    /// `advanceExact(target)`.
    ///
    /// # Errors
    /// As [`Self::next_doc`].
    pub fn advance_exact(&mut self, target: i32) -> Result<bool> {
        if self.binary.advance_exact(target)? {
            self.set_current_doc()?;
            return Ok(true);
        }
        Ok(false)
    }

    /// `setCurrentDoc()`: the CSV of the current document, parsed.
    fn set_current_doc(&mut self) -> Result<()> {
        // Java: SimpleTextDocValuesReader.getSortedNumeric.setCurrentDoc
        if self.doc_id() == NO_MORE_DOCS {
            return Ok(());
        }
        let csv = self.binary.binary_value()?;
        let csv = String::from_utf8_lossy(&csv);
        self.values.clear();
        if !csv.is_empty() {
            for s in java_split_comma(&csv) {
                self.values.push(parse_long(s.as_bytes())?);
            }
        }
        self.index = 0;
        Ok(())
    }

    /// `nextValue()`.
    ///
    /// # Errors
    /// [`Error::IllegalState`] past `docValueCount()` values (Java's array
    /// index error).
    pub fn next_value(&mut self) -> Result<i64> {
        let v = *self.values.get(self.index).ok_or(Error::IllegalState(
            "nextValue called more than docValueCount times",
        ))?;
        self.index = self.index.saturating_add(1);
        Ok(v)
    }

    /// `docValueCount()`.
    pub fn doc_value_count(&self) -> usize {
        self.values.len()
    }
}

/// `s.split(",")`: Java drops trailing empty parts.
fn java_split_comma(s: &str) -> Vec<&str> {
    let mut parts: Vec<&str> = s.split(',').collect();
    while parts.last() == Some(&"") {
        parts.pop();
    }
    parts
}

/// `getSortedSet`'s `SortedSetDocValues`.
#[derive(Debug, Clone)]
pub struct SimpleTextSortedSetDocValues<'r> {
    c: Cursor<'r>,
    /// The current document's trimmed ord list (`currentOrds`, unsplit).
    ord_list: String,
    /// `currentOrds`: byte ranges of `ord_list`.
    parts: Vec<(usize, usize)>,
    index: usize,
}

impl<'r> SimpleTextSortedSetDocValues<'r> {
    /// `docID()`.
    pub fn doc_id(&self) -> i32 {
        self.c.doc
    }

    /// `cost()`: `maxDoc`.
    pub fn cost(&self) -> i64 {
        i64::from(self.c.max_doc)
    }

    /// Document `i`'s ord list; `true` (and the list split) when it is not
    /// blank.
    fn load(&mut self, i: i32) -> Result<bool> {
        self.c.read_ord_line(i)?;
        let line = String::from_utf8_lossy(&self.c.scratch);
        let trimmed = line.trim_matches(|ch: char| ch <= ' ');
        if trimmed.is_empty() {
            return Ok(false);
        }
        self.ord_list.clear();
        self.ord_list.push_str(trimmed);
        self.parts.clear();
        let mut start = 0usize;
        for part in java_split_comma(&self.ord_list) {
            let end = start.saturating_add(part.len());
            self.parts.push((start, end));
            start = end.saturating_add(1);
        }
        self.index = 0;
        Ok(true)
    }

    /// `nextDoc()`.
    ///
    /// # Errors
    /// `Corrupted`/`Eof` for an ord line that does not read.
    pub fn next_doc(&mut self) -> Result<i32> {
        self.advance(self.c.next_target())
    }

    /// `advance(target)`.
    ///
    /// # Errors
    /// As [`Self::next_doc`].
    pub fn advance(&mut self, target: i32) -> Result<i32> {
        // Java: SimpleTextDocValuesReader.getSortedSet.advance
        let mut i = target;
        while i < self.c.max_doc {
            if self.load(i)? {
                self.c.doc = i;
                return Ok(i);
            }
            i = i.saturating_add(1);
        }
        self.c.doc = NO_MORE_DOCS;
        Ok(NO_MORE_DOCS)
    }

    /// `advanceExact(target)`.
    ///
    /// # Errors
    /// As [`Self::next_doc`]; [`Error::IllegalArgument`] outside the segment.
    pub fn advance_exact(&mut self, target: i32) -> Result<bool> {
        self.c.check_doc(target)?;
        let has = self.load(target)?;
        self.c.doc = target;
        Ok(has)
    }

    /// `nextOrd()`: `Long.parseLong(currentOrds[currentIndex++])`.
    ///
    /// # Errors
    /// [`Error::IllegalState`] past `docValueCount()` ords; `Corrupted` for
    /// an ord that is not a `long`.
    pub fn next_ord(&mut self) -> Result<i64> {
        let &(start, end) = self.parts.get(self.index).ok_or(Error::IllegalState(
            "nextOrd called more than docValueCount times",
        ))?;
        self.index = self.index.saturating_add(1);
        parse_long(self.ord_list.as_bytes().get(start..end).unwrap_or_default())
    }

    /// `docValueCount()`.
    pub fn doc_value_count(&self) -> usize {
        self.parts.len()
    }

    /// `lookupOrd(ord)`.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for an ord outside the dictionary;
    /// `Corrupted`/`Eof` for a bad term record.
    pub fn lookup_ord(&mut self, ord: i64) -> Result<&'r [u8]> {
        lookup_ord(self.c.data, self.c.field, ord, &mut self.c.scratch)
    }

    /// `getValueCount()`.
    pub fn value_count(&self) -> i64 {
        self.c.field.num_values
    }

    /// `lookupTerm(key)`: `Ok(ord)`, or `Err(insertion point)`.
    ///
    /// # Errors
    /// What [`Self::lookup_ord`] fails with.
    pub fn lookup_term(&self, key: &[u8]) -> Result<std::result::Result<i64, i64>> {
        // SENTINEL-OK: this module's `lookup_term` returns `Ok(ord)` or
        // `Err(insertion point)`; it has no `-1` sentinel.
        lookup_term(self.c.data, self.c.field, self.value_count(), key)
    }

    /// `termsEnum()`.
    pub fn terms_enum(&self) -> SimpleTextDocValuesTermsEnum<'r> {
        SimpleTextDocValuesTermsEnum::new(self.c.data, self.c.field, self.value_count())
    }
}

/// `SortedDocValuesTermsEnum` / `SortedSetDocValuesTermsEnum`: the
/// dictionary in ord order through `lookupOrd`.
#[derive(Debug, Clone)]
pub struct SimpleTextDocValuesTermsEnum<'r> {
    data: &'r [u8],
    field: &'r OneField,
    value_count: i64,
    current_ord: i64,
    term: Vec<u8>,
    scratch: Vec<u8>,
}

impl<'r> SimpleTextDocValuesTermsEnum<'r> {
    fn new(data: &'r [u8], field: &'r OneField, value_count: i64) -> Self {
        SimpleTextDocValuesTermsEnum {
            data,
            field,
            value_count,
            current_ord: -1,
            term: Vec::new(),
            scratch: Vec::new(),
        }
    }

    /// `next()`: the next term, `None` past the last.
    ///
    /// # Errors
    /// What `lookupOrd` fails with.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Result<Option<&[u8]>> {
        self.current_ord = self.current_ord.saturating_add(1);
        if self.current_ord >= self.value_count {
            return Ok(None);
        }
        let t = lookup_ord(self.data, self.field, self.current_ord, &mut self.scratch)?;
        self.term.clear();
        self.term.extend_from_slice(t);
        Ok(Some(&self.term))
    }

    /// `seekCeil(text)`.
    ///
    /// # Errors
    /// What `lookupOrd` fails with.
    pub fn seek_ceil(&mut self, text: &[u8]) -> Result<SeekStatus> {
        // SENTINEL-OK: this module's `lookup_term` returns `Ok(ord)` or
        // `Err(insertion point)`; it has no `-1` sentinel.
        match lookup_term(self.data, self.field, self.value_count, text)? {
            Ok(ord) => {
                self.current_ord = ord;
                self.term.clear();
                self.term.extend_from_slice(text);
                Ok(SeekStatus::Found)
            }
            Err(ord) => {
                self.current_ord = ord;
                if ord == self.value_count {
                    return Ok(SeekStatus::End);
                }
                let t = lookup_ord(self.data, self.field, ord, &mut self.scratch)?;
                self.term.clear();
                self.term.extend_from_slice(t);
                Ok(SeekStatus::NotFound)
            }
        }
    }

    /// `seekExact(text)`.
    ///
    /// # Errors
    /// What `lookupOrd` fails with.
    pub fn seek_exact(&mut self, text: &[u8]) -> Result<bool> {
        // SENTINEL-OK: this module's `lookup_term` returns `Ok(ord)` or
        // `Err(insertion point)`; it has no `-1` sentinel.
        match lookup_term(self.data, self.field, self.value_count, text)? {
            Ok(ord) => {
                self.current_ord = ord;
                self.term.clear();
                self.term.extend_from_slice(text);
                Ok(true)
            }
            Err(_) => Ok(false),
        }
    }

    /// `seekExact(ord)`.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] outside the dictionary.
    pub fn seek_exact_ord(&mut self, ord: i64) -> Result<()> {
        let t = lookup_ord(self.data, self.field, ord, &mut self.scratch)?;
        self.current_ord = ord;
        self.term.clear();
        self.term.extend_from_slice(t);
        Ok(())
    }

    /// `term()`.
    pub fn term(&self) -> &[u8] {
        &self.term
    }

    /// `ord()`.
    pub fn ord(&self) -> i64 {
        self.current_ord
    }
}

/// `getSkipper`'s one-level `DocValuesSkipper` over the field's intervals.
#[derive(Debug, Clone)]
pub struct SimpleTextSkipper<'r> {
    numeric: bool,
    field: &'r OneField,
    /// `cursor`: -1 before the first `advance`.
    cursor: i64,
}

impl SimpleTextSkipper<'_> {
    fn current(&self) -> Option<&SkipInterval> {
        usize::try_from(self.cursor)
            .ok()
            .and_then(|c| self.field.skip_intervals.get(c))
    }

    /// `numLevels()`.
    pub fn num_levels(&self) -> i32 {
        1
    }

    /// `minValue(level)`.
    pub fn min_value_at(&self, _level: i32) -> i64 {
        self.current()
            .map_or_else(|| self.min_value(), |i| i.min_value)
    }

    /// `maxValue(level)`.
    pub fn max_value_at(&self, _level: i32) -> i64 {
        self.current()
            .map_or_else(|| self.max_value(), |i| i.max_value)
    }

    /// `docCount(level)`.
    pub fn doc_count_at(&self, _level: i32) -> i32 {
        self.current()
            .map_or_else(|| self.doc_count(), |i| i.doc_count)
    }

    /// `minValue()`: the field's minimum (0 for ords).
    pub fn min_value(&self) -> i64 {
        if self.numeric {
            self.field.min_value
        } else {
            0
        }
    }

    /// `maxValue()`: the field's maximum (`numValues - 1` for ords).
    pub fn max_value(&self) -> i64 {
        if self.numeric {
            self.field.max_value
        } else {
            self.field.num_values.wrapping_sub(1)
        }
    }

    /// `docCount()`.
    pub fn doc_count(&self) -> i32 {
        self.field.doc_count
    }

    /// `maxValueCount()`.
    pub fn max_value_count(&self) -> i32 {
        self.field.max_value_count
    }

    /// `minDocID(level)`.
    ///
    // SENTINEL: -1 before the first `advance` (Java's), `NO_MORE_DOCS` past
    // the last interval.
    pub fn min_doc_id(&self, _level: i32) -> i32 {
        match self.position() {
            None => -1,
            Some(Some(i)) => i.min_doc_id,
            Some(None) => NO_MORE_DOCS,
        }
    }

    /// `maxDocID(level)`.
    ///
    // SENTINEL: as [`Self::min_doc_id`].
    pub fn max_doc_id(&self, _level: i32) -> i32 {
        match self.position() {
            None => -1,
            Some(Some(i)) => i.max_doc_id,
            Some(None) => NO_MORE_DOCS,
        }
    }

    /// `None` before the first `advance`, else the interval (`None` past the
    /// last).
    fn position(&self) -> Option<Option<&SkipInterval>> {
        if self.cursor == -1 {
            None
        } else {
            Some(self.current())
        }
    }

    /// `advance(target)`: the first interval whose `maxDocID >= target`.
    pub fn advance(&mut self, target: i32) {
        // Java: SimpleTextDocValuesReader.getSkipper.advance
        let intervals = &self.field.skip_intervals;
        if intervals.is_empty() {
            self.cursor = 0;
            return;
        }
        if self.cursor == -1 {
            self.cursor = 0;
        }
        while self.current().is_some_and(|i| i.max_doc_id < target) {
            self.cursor = self.cursor.saturating_add(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::doc_values::{
        BinaryColumn, SortedNumericColumn, SortedOrdsColumn, SortedSetOrdsColumn,
    };
    use crate::field_infos::{DocValuesSkipIndexType as Skip, FieldInfo};
    use crate::simpletext::doc_values_writer::SimpleTextDocValuesWriter;

    const MAX_DOC: i32 = 6;

    fn fi(name: &str, n: i32, ty: DocValuesType, skip: Skip) -> FieldInfo {
        let mut f = FieldInfo::new(name, n);
        f.doc_values_type = ty;
        f.doc_values_skip_index_type = skip;
        f
    }

    /// One field of every type over six documents.
    fn file() -> Vec<u8> {
        let mut w = SimpleTextDocValuesWriter::new(MAX_DOC);
        w.add_numeric_field(
            &fi("num", 0, DocValuesType::Numeric, Skip::Range),
            &[(0, 5), (2, -3), (5, i64::MAX)],
        )
        .unwrap();
        w.add_binary_field(
            &fi("bin", 1, DocValuesType::Binary, Skip::None),
            &BinaryColumn {
                docs: vec![1, 3],
                lengths: vec![3, 0],
                bytes: b"a\nb".to_vec(),
            },
        )
        .unwrap();
        w.add_sorted_field(
            &fi("sorted", 2, DocValuesType::Sorted, Skip::Range),
            &SortedOrdsColumn {
                dict: vec![b"x".to_vec(), b"y".to_vec()],
                docs: vec![0, 4],
                ords: vec![1, 0],
            },
        )
        .unwrap();
        w.add_sorted_numeric_field(
            &fi("snum", 3, DocValuesType::SortedNumeric, Skip::Range),
            &SortedNumericColumn {
                docs: vec![2],
                counts: vec![2],
                values: vec![-1, 7],
            },
        )
        .unwrap();
        w.add_sorted_set_field(
            &fi("sset", 4, DocValuesType::SortedSet, Skip::None),
            &SortedSetOrdsColumn {
                dict: vec![b"p".to_vec(), b"q".to_vec(), b"r".to_vec()],
                docs: vec![0, 5],
                counts: vec![2, 1],
                ords: vec![0, 2, 1],
            },
        )
        .unwrap();
        w.add_numeric_field(&fi("none", 5, DocValuesType::Numeric, Skip::None), &[])
            .unwrap();
        w.finish().unwrap()
    }

    fn text(data: &[u8]) -> String {
        String::from_utf8(data.to_vec()).unwrap()
    }

    /// Lucene 10.6's `isSingleValued` (#16714), off the recorded
    /// `maxValueCount`.
    #[test]
    fn single_valued_fields_follow_max_value_count() {
        let data = file();
        let r = SimpleTextDocValuesReader::open(&data, MAX_DOC).unwrap();
        for (name, single) in [
            ("num", true),
            ("sorted", true),
            ("snum", false),
            ("sset", false),
            ("none", true),
        ] {
            assert_eq!(r.field(name).unwrap().is_single_valued(), single, "{name}");
        }
        let mut unknown = OneField::new(DocValuesType::SortedNumeric);
        unknown.max_value_count = -1;
        assert!(!unknown.is_single_valued());
    }

    #[test]
    fn cached_headers_borrow_live_bytes_for_each_value_type() {
        let data = file();
        let metadata = SimpleTextDocValuesReader::open(&data, MAX_DOC)
            .unwrap()
            .rebind(&[]);
        let mut numeric = metadata.numeric_from_data("num", &data).unwrap();
        assert!(numeric.advance_exact(2).unwrap());
        assert_eq!(numeric.long_value().unwrap(), -3);
        let mut binary = metadata.binary_from_data("bin", &data).unwrap();
        assert!(binary.advance_exact(1).unwrap());
        assert_eq!(binary.binary_value().unwrap(), b"a\nb");
        let mut sorted = metadata.sorted_from_data("sorted", &data).unwrap();
        assert!(sorted.advance_exact(4).unwrap());
        assert_eq!(sorted.lookup_ord(sorted.ord_value()).unwrap(), b"x");
        let mut numeric = metadata.sorted_numeric_from_data("snum", &data).unwrap();
        assert!(numeric.advance_exact(2).unwrap());
        assert_eq!(numeric.doc_value_count(), 2);
        assert_eq!(numeric.next_value().unwrap(), -1);
        assert_eq!(numeric.next_value().unwrap(), 7);
        let mut set = metadata.sorted_set_from_data("sset", &data).unwrap();
        assert!(set.advance_exact(0).unwrap());
        assert_eq!(set.doc_value_count(), 2);
        let ord = set.next_ord().unwrap();
        assert_eq!(set.lookup_ord(ord).unwrap(), b"p");
        let ord = set.next_ord().unwrap();
        assert_eq!(set.lookup_ord(ord).unwrap(), b"r");
        assert!(metadata.numeric("num").unwrap().next_doc().is_err());
        assert!(metadata.check_integrity().is_err());
    }

    #[test]
    fn numeric_iterates_and_advances() {
        let data = file();
        let r = SimpleTextDocValuesReader::open(&data, MAX_DOC).unwrap();
        assert_eq!((r.max_doc(), r.num_fields()), (MAX_DOC, 6));
        r.check_integrity().unwrap();
        let mut it = r.numeric("num").unwrap();
        assert!(it.cost() >= 0);
        let mut got = Vec::new();
        while it.next_doc().unwrap() != NO_MORE_DOCS {
            got.push((it.doc_id(), it.long_value().unwrap()));
        }
        assert_eq!(got, [(0, 5), (2, -3), (5, i64::MAX)]);
        let mut it = r.numeric("num").unwrap();
        assert_eq!(it.advance(3).unwrap(), 5);
        let mut it = r.numeric("num").unwrap();
        assert!(!it.advance_exact(1).unwrap());
        assert!(it.advance_exact(2).unwrap());
        assert_eq!(it.long_value().unwrap(), -3);
        let mut none = r.numeric("none").unwrap();
        assert_eq!(none.next_doc().unwrap(), NO_MORE_DOCS);
        assert!(r.numeric("missing").is_err());
        assert!(r.field("num").is_some() && r.field("missing").is_none());
    }

    #[test]
    fn binary_and_sorted_numeric_values() {
        let data = file();
        let r = SimpleTextDocValuesReader::open(&data, MAX_DOC).unwrap();
        let mut it = r.binary("bin").unwrap();
        assert!(it.cost() >= 0);
        assert_eq!(it.next_doc().unwrap(), 1);
        assert_eq!(it.binary_value().unwrap(), b"a\nb");
        assert_eq!(it.next_doc().unwrap(), 3);
        assert_eq!(it.binary_value().unwrap(), b"");
        assert_eq!(it.next_doc().unwrap(), NO_MORE_DOCS);
        let mut it = r.binary("bin").unwrap();
        assert_eq!(it.advance(2).unwrap(), 3);
        assert_eq!(it.doc_id(), 3);
        let mut it = r.binary("bin").unwrap();
        assert!(!it.advance_exact(2).unwrap());
        assert!(it.advance_exact(3).unwrap());
        assert!(r.binary("missing").is_err());

        let mut s = r.sorted_numeric("snum").unwrap();
        assert!(s.cost() >= 0);
        assert_eq!(s.next_doc().unwrap(), 2);
        assert_eq!(s.doc_value_count(), 2);
        assert_eq!((s.next_value().unwrap(), s.next_value().unwrap()), (-1, 7));
        assert_eq!(s.next_doc().unwrap(), NO_MORE_DOCS);
        let mut s = r.sorted_numeric("snum").unwrap();
        assert_eq!(s.advance(1).unwrap(), 2);
        assert_eq!(s.doc_id(), 2);
        let mut s = r.sorted_numeric("snum").unwrap();
        assert!(!s.advance_exact(1).unwrap());
        assert!(s.advance_exact(2).unwrap());
        assert_eq!(s.next_value().unwrap(), -1);
        assert!(r.sorted_numeric("missing").is_err());
    }

    #[test]
    fn sorted_ords_dictionary_and_terms_enum() {
        let data = file();
        let r = SimpleTextDocValuesReader::open(&data, MAX_DOC).unwrap();
        let mut s = r.sorted("sorted").unwrap();
        assert!(s.cost() >= 0);
        assert_eq!(s.value_count(), 2);
        assert_eq!(s.next_doc().unwrap(), 0);
        assert_eq!(s.ord_value(), 1);
        assert_eq!(s.lookup_ord(1).unwrap(), b"y");
        assert_eq!(s.next_doc().unwrap(), 4);
        assert_eq!((s.doc_id(), s.ord_value()), (4, 0));
        assert_eq!(s.next_doc().unwrap(), NO_MORE_DOCS);
        assert!(s.lookup_ord(-1).is_err() && s.lookup_ord(2).is_err());
        let mut s = r.sorted("sorted").unwrap();
        assert_eq!(s.advance(1).unwrap(), 4);
        let mut s = r.sorted("sorted").unwrap();
        assert!(!s.advance_exact(3).unwrap());
        assert!(s.advance_exact(4).unwrap());
        assert_eq!(s.lookup_term(b"y").unwrap(), Ok(1));
        assert_eq!(s.lookup_term(b"a").unwrap(), Err(0));
        assert_eq!(s.lookup_term(b"xx").unwrap(), Err(1));
        assert_eq!(s.lookup_term(b"z").unwrap(), Err(2));
        let mut te = s.terms_enum();
        assert_eq!(te.next().unwrap(), Some(&b"x"[..]));
        assert_eq!(te.next().unwrap(), Some(&b"y"[..]));
        assert_eq!(te.next().unwrap(), None);
        let mut te = s.terms_enum();
        assert_eq!(te.seek_ceil(b"a").unwrap(), SeekStatus::NotFound);
        assert_eq!((te.ord(), te.term()), (0, &b"x"[..]));
        assert_eq!(te.seek_ceil(b"y").unwrap(), SeekStatus::Found);
        assert_eq!(te.ord(), 1);
        assert_eq!(te.seek_ceil(b"z").unwrap(), SeekStatus::End);
        assert!(te.seek_exact(b"x").unwrap());
        assert!(!te.seek_exact(b"w").unwrap());
        te.seek_exact_ord(1).unwrap();
        assert_eq!(te.term(), b"y");
        assert!(te.seek_exact_ord(2).is_err());
        assert!(r.sorted("missing").is_err());
    }

    #[test]
    fn sorted_set_ords() {
        let data = file();
        let r = SimpleTextDocValuesReader::open(&data, MAX_DOC).unwrap();
        let mut s = r.sorted_set("sset").unwrap();
        assert!(s.cost() >= 0);
        assert_eq!(s.value_count(), 3);
        assert_eq!(s.next_doc().unwrap(), 0);
        assert_eq!(s.doc_value_count(), 2);
        assert_eq!((s.next_ord().unwrap(), s.next_ord().unwrap()), (0, 2));
        assert_eq!(s.next_doc().unwrap(), 5);
        assert_eq!((s.doc_id(), s.next_ord().unwrap()), (5, 1));
        assert_eq!(s.next_doc().unwrap(), NO_MORE_DOCS);
        assert_eq!(s.lookup_ord(2).unwrap(), b"r");
        assert!(s.lookup_ord(3).is_err());
        assert_eq!(s.lookup_term(b"q").unwrap(), Ok(1));
        assert_eq!(s.lookup_term(b"qq").unwrap(), Err(2));
        let mut te = s.terms_enum();
        assert_eq!(te.seek_ceil(b"qq").unwrap(), SeekStatus::NotFound);
        assert_eq!(te.term(), b"r");
        let mut s = r.sorted_set("sset").unwrap();
        assert_eq!(s.advance(1).unwrap(), 5);
        let mut s = r.sorted_set("sset").unwrap();
        assert!(!s.advance_exact(1).unwrap());
        assert!(s.advance_exact(5).unwrap());
        assert!(r.sorted_set("missing").is_err());
    }

    #[test]
    fn skippers_walk_their_intervals() {
        let data = file();
        let r = SimpleTextDocValuesReader::open(&data, MAX_DOC).unwrap();
        let mut s = r.skipper("num", DocValuesType::Numeric).unwrap();
        assert_eq!(s.num_levels(), 1);
        assert_eq!((s.min_doc_id(0), s.max_doc_id(0)), (-1, -1));
        assert_eq!((s.min_value(), s.max_value()), (-3, i64::MAX));
        assert_eq!((s.min_value_at(0), s.max_value_at(0)), (-3, i64::MAX));
        assert_eq!(
            (s.doc_count(), s.doc_count_at(0), s.max_value_count()),
            (3, 3, 1)
        );
        s.advance(0);
        assert_eq!((s.min_doc_id(0), s.max_doc_id(0)), (0, 5));
        assert_eq!(s.doc_count_at(0), 3);
        s.advance(6);
        assert_eq!(
            (s.min_doc_id(0), s.max_doc_id(0)),
            (NO_MORE_DOCS, NO_MORE_DOCS)
        );
        assert_eq!(s.min_value_at(0), -3);
        let s = r.skipper("sorted", DocValuesType::Sorted).unwrap();
        assert_eq!((s.min_value(), s.max_value()), (0, 1));
        assert!(r.skipper("missing", DocValuesType::Numeric).is_err());
    }

    #[test]
    fn malformed_headers_are_corrupt() {
        let data = file();
        let t = text(&data);
        let open = |s: &str| SimpleTextDocValuesReader::open(s.as_bytes(), MAX_DOC).map(|_| ());
        assert!(SimpleTextDocValuesReader::open(b"", MAX_DOC).is_err());
        assert!(open("feld num\n").is_err());
        assert!(open(&t[t.find("END").unwrap()..]).is_err(), "no field");
        assert!(open(&t.replacen("  type NUMERIC", "  type NONE", 1)).is_err());
        assert!(open(&t.replacen("  type NUMERIC", "  type WHAT", 1)).is_err());
        assert!(open(&t.replacen("  minalue ", "  minalue x", 1)).is_err());
        assert!(open(&t.replacen("  origin ", "  orign ", 1)).is_err());
        assert!(open(&t.replacen("  maxlength ", "  maxlength x", 1)).is_err());
        assert!(open(&t.replacen("  numvalues ", "  numvalue ", 1)).is_err());
        assert!(open(&t.replacen("  skipdata", "  skipdat", 1)).is_err());
        assert!(open(&t.replacen("    interval", "    intervl", 1)).is_err());
        assert!(open(&t[..t.len() / 2]).is_err(), "truncated");
        assert!(
            SimpleTextDocValuesReader::open(&data, 1_000_000).is_err(),
            "past the end"
        );
        assert!(SimpleTextDocValuesReader::open(&data, -1).is_err());
        // A width or count that seeks backwards: `bin`'s records would
        // rewind onto `num`'s skip data, and `open` would never reach END.
        let reseal = |s: String| {
            let mut b = s.as_bytes()[..s.rfind("checksum ").unwrap()].to_vec();
            util::write_checksum(&mut b);
            b
        };
        for (from, to) in [
            ("  maxlength 9\n", "  maxlength -99\n"),
            ("  maxlength 9\n", "  maxlength -2\n"),
            ("  numvalues 2\n", "  numvalues -1\n"),
        ] {
            assert!(t.contains(from), "{from}");
            let bad = reseal(t.replacen(from, to, 1));
            assert!(
                SimpleTextDocValuesReader::open(&bad, MAX_DOC).is_err(),
                "{to}"
            );
        }
        assert!(seek(&data, 10, -1, 1).is_err() && seek(&data, 10, i64::MAX, 2).is_err());
        assert_eq!(seek(&data, 10, 2, 3).unwrap(), 16);
        // Without `maxvaluecount` lines: 0 for no documents, 1 for
        // NUMERIC/SORTED, -1 otherwise.
        let stripped: String = t
            .split_inclusive('\n')
            .filter(|l| !l.starts_with("  maxvaluecount "))
            .collect();
        let r = SimpleTextDocValuesReader::open(stripped.as_bytes(), MAX_DOC).unwrap();
        let count = |f: &str| r.field(f).unwrap().max_value_count;
        assert_eq!(
            [
                count("num"),
                count("sorted"),
                count("bin"),
                count("sset"),
                count("none")
            ],
            [1, 1, -1, -1, 0]
        );
    }

    #[test]
    fn malformed_records_are_corrupt() {
        let t = text(&file());
        fn r(s: &str) -> SimpleTextDocValuesReader<'_> {
            SimpleTextDocValuesReader::open(s.as_bytes(), MAX_DOC).unwrap()
        }
        // num's first record follows its pattern line.
        let at = t.find("  pattern ").unwrap();
        let rec = t[at..].find('\n').unwrap() + at + 1;
        let mut bad = t.clone();
        bad.replace_range(rec..=rec, "x");
        let rd = r(&bad);
        let mut it = rd.numeric("num").unwrap();
        assert!(it.next_doc().and_then(|_| it.long_value()).is_err());
        // bin's first length line.
        let bad = t.replacen("\nlength 9", "\nlength x", 1);
        assert_ne!(bad, t);
        let rd = r(&bad);
        let mut it = rd.binary("bin").unwrap();
        assert!(it
            .next_doc()
            .and_then(|_| it.binary_value().map(|_| 0))
            .is_err());
        // An ord list that is not longs.
        let bad = t.replacen("0,2", "0,z", 1);
        let rd = r(&bad);
        let mut it = rd.sorted_set("sset").unwrap();
        assert!(it
            .next_doc()
            .and_then(|_| it.next_ord())
            .and_then(|_| it.next_ord())
            .is_err());
        assert_eq!(java_split_comma("1,,2,,"), ["1", "", "2"]);
        // check_integrity: one changed byte breaks the checksum.
        let flipped = t.replacen("doccount 3", "doccount 4", 1);
        let rd = r(&flipped);
        assert!(rd.check_integrity().is_err());
    }
}
