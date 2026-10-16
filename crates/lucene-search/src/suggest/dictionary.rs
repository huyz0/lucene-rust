//! `FileDictionary`, `DocumentDictionary` and `DocumentValueSourceDictionary`
//! (M13 T13.3a): the [`Dictionary`]s of the suggest
//! package. Each reads its source when
//! [`get_entry_iterator`](crate::spell::Dictionary::get_entry_iterator) is
//! called and replays the entries from memory, so a malformed line or a
//! failed read surfaces there rather than from a later `next()`.
//!
//! Departures from Java:
//!
//! * `FileDictionary`'s field delimiter is a regular expression in Java
//!   (`String.split`); here it is a literal. `\t`, the default, and any
//!   delimiter without regex metacharacters behave the same.
//! * A weight that is neither a `long` nor a `double` (`Double.parseDouble`
//!   also takes a trailing `d`/`f` and surrounding whitespace, which are
//!   handled; hexadecimal floating point is not) is reported as an
//!   `IllegalArgument` carrying `NumberFormatException`'s message.
//! * `DocumentDictionary` reads a [`DirectoryReader`]; it visits segment by
//!   segment, which is the order of Java's global document ids.

use std::collections::BTreeSet;
use std::sync::Arc;

use lucene_codecs::stored_fields::{DocumentVisitor, FieldValue};

use super::input::InputIterator;
use crate::directory_reader::{DirectoryReader, SegmentReader};
use crate::spell::Dictionary;
use crate::values_source::{LongValuesSource, ValuesContext};
use crate::{Error, Result};

/// One entry of a dictionary: `(term, weight, payload, contexts)`.
#[derive(Debug, Clone)]
struct DictEntry {
    term: Vec<u8>,
    weight: i64,
    payload: Vec<u8>,
    contexts: BTreeSet<Vec<u8>>,
}

/// The entries a dictionary read, replayed as an [`InputIterator`].
#[derive(Debug, Clone)]
pub struct EntriesInputIterator {
    entries: std::vec::IntoIter<DictEntry>,
    current: Option<DictEntry>,
    has_payloads: bool,
    has_contexts: bool,
}

impl EntriesInputIterator {
    fn new(entries: Vec<DictEntry>, has_payloads: bool, has_contexts: bool) -> Self {
        Self {
            entries: entries.into_iter(),
            current: None,
            has_payloads,
            has_contexts,
        }
    }
}

impl InputIterator for EntriesInputIterator {
    fn next(&mut self) -> Option<Vec<u8>> {
        self.current = self.entries.next();
        self.current.as_ref().map(|e| e.term.clone())
    }

    fn weight(&self) -> i64 {
        self.current.as_ref().map_or(0, |e| e.weight)
    }

    fn payload(&self) -> Option<&[u8]> {
        if !self.has_payloads {
            return None;
        }
        self.current.as_ref().map(|e| e.payload.as_slice())
    }

    fn has_payloads(&self) -> bool {
        self.has_payloads
    }

    fn contexts(&self) -> Option<&BTreeSet<Vec<u8>>> {
        if !self.has_contexts {
            return None;
        }
        self.current.as_ref().map(|e| &e.contexts)
    }

    fn has_contexts(&self) -> bool {
        self.has_contexts
    }
}

/// `FileDictionary.DEFAULT_FIELD_DELIMITER`.
pub const DEFAULT_FIELD_DELIMITER: &str = "\t";

/// `FileDictionary`: lines of `term`, `term TAB weight` or
/// `term TAB weight TAB payload`.
#[derive(Debug, Clone)]
pub struct FileDictionary {
    text: String,
    field_delimiter: String,
}

impl FileDictionary {
    /// `new FileDictionary(Reader)`: the text, already decoded.
    pub fn from_text(text: impl Into<String>) -> Self {
        Self::from_text_with_delimiter(text, DEFAULT_FIELD_DELIMITER)
    }

    /// `new FileDictionary(Reader, fieldDelimiter)`.
    pub fn from_text_with_delimiter(
        text: impl Into<String>,
        field_delimiter: impl Into<String>,
    ) -> Self {
        Self {
            text: text.into(),
            field_delimiter: field_delimiter.into(),
        }
    }

    /// `new FileDictionary(InputStream)`: UTF-8, strictly decoded as
    /// `IOUtils.getDecodingReader` does.
    ///
    /// # Errors
    /// Bytes that are not UTF-8.
    pub fn from_utf8(bytes: &[u8]) -> Result<Self> {
        Self::from_utf8_with_delimiter(bytes, DEFAULT_FIELD_DELIMITER)
    }

    /// `new FileDictionary(InputStream, fieldDelimiter)`.
    ///
    /// # Errors
    /// Bytes that are not UTF-8.
    pub fn from_utf8_with_delimiter(
        bytes: &[u8],
        field_delimiter: impl Into<String>,
    ) -> Result<Self> {
        let text = std::str::from_utf8(bytes)
            .map_err(|e| Error::IllegalArgument(format!("MalformedInputException: {e}")))?;
        Ok(Self::from_text_with_delimiter(text, field_delimiter))
    }
}

/// `BufferedReader.readLine`: lines end at `\n`, `\r` or `\r\n`; a final
/// terminator does not start another line.
fn read_lines(text: &str) -> Vec<&str> {
    let mut lines = Vec::new();
    let bytes = text.as_bytes();
    let (mut start, mut i) = (0, 0);
    while i < bytes.len() {
        match bytes[i] {
            b'\n' => {
                lines.push(&text[start..i]);
                i += 1;
                start = i;
            }
            b'\r' => {
                lines.push(&text[start..i]);
                i += if bytes.get(i + 1) == Some(&b'\n') {
                    2
                } else {
                    1
                };
                start = i;
            }
            _ => i += 1,
        }
    }
    if start < bytes.len() {
        lines.push(&text[start..]);
    }
    lines
}

/// `String.split(delimiter)` for a literal delimiter: trailing empty strings
/// are dropped, and a line without the delimiter is returned whole.
fn java_split<'a>(line: &'a str, delimiter: &str) -> Vec<&'a str> {
    if delimiter.is_empty() {
        // a zero-width match between every pair of characters
        return line
            .char_indices()
            .map(|(i, c)| &line[i..i + c.len_utf8()])
            .collect();
    }
    if !line.contains(delimiter) {
        return vec![line];
    }
    let mut parts: Vec<&str> = line.split(delimiter).collect();
    while parts.last().is_some_and(|p| p.is_empty()) {
        parts.pop();
    }
    parts
}

/// `Double.parseDouble`, as far as `FileDictionary` needs it.
fn parse_double(s: &str) -> Option<f64> {
    let t = s.trim_matches(|c: char| c <= ' ');
    let t = t
        .strip_suffix(['d', 'D', 'f', 'F'])
        .filter(|rest| rest.ends_with(|c: char| c.is_ascii_digit() || c == '.'))
        .unwrap_or(t);
    match t {
        "NaN" | "+NaN" | "-NaN" => return Some(f64::NAN),
        "Infinity" | "+Infinity" => return Some(f64::INFINITY),
        "-Infinity" => return Some(f64::NEG_INFINITY),
        _ => {}
    }
    // Java spells NaN and Infinity in exactly one case, which the match above
    // took; any other letter but the exponent is not a Java decimal (Rust's
    // `parse` would take `nan`, `inf`, `INFINITY`). Hexadecimal floats are not
    // read.
    if t.chars()
        .any(|c| c.is_ascii_alphabetic() && c != 'e' && c != 'E')
    {
        return None;
    }
    t.parse::<f64>().ok()
}

/// `readWeight`: `Long.parseLong`, else `(long) Double.parseDouble`.
fn read_weight(weight: &str) -> Result<i64> {
    if let Ok(v) = weight.parse::<i64>() {
        return Ok(v);
    }
    parse_double(weight)
        // a saturating cast, NaN to 0, as Java's
        .map(|d| d as i64)
        .ok_or_else(|| {
            // `FloatingDecimal` reports the trimmed input
            let trimmed = weight.trim_matches(|c: char| c <= ' ');
            Error::IllegalArgument(if trimmed.is_empty() {
                "NumberFormatException: empty String".to_string()
            } else {
                format!("NumberFormatException: For input string: \"{trimmed}\"")
            })
        })
}

impl Dictionary for FileDictionary {
    fn get_entry_iterator(&self) -> Result<Box<dyn InputIterator>> {
        let lines = read_lines(&self.text);
        let mut entries = Vec::with_capacity(lines.len());
        let mut has_payloads = false;
        for (n, line) in lines.iter().enumerate() {
            let fields = java_split(line, &self.field_delimiter);
            if fields.len() > 3 {
                return Err(Error::IllegalArgument(
                    "More than 3 fields in one line".into(),
                ));
            }
            if n == 0 {
                // the first line decides whether the dictionary has payloads
                has_payloads = fields.len() == 3;
            }
            let Some(term) = fields.first() else {
                return Err(Error::IllegalArgument(
                    "ArrayIndexOutOfBoundsException: Index 0 out of bounds for length 0".into(),
                ));
            };
            let weight = match fields.get(1) {
                Some(w) => read_weight(w)?,
                None => 1,
            };
            let payload = match fields.get(2) {
                Some(p) if has_payloads => p.as_bytes().to_vec(),
                _ => Vec::new(),
            };
            entries.push(DictEntry {
                term: term.as_bytes().to_vec(),
                weight,
                payload,
                contexts: BTreeSet::new(),
            });
        }
        Ok(Box::new(EntriesInputIterator::new(
            entries,
            has_payloads,
            false,
        )))
    }
}

/// Where a document's weight comes from.
enum WeightSource<'a> {
    /// `DocumentDictionary`: a stored numeric field, else numeric doc values
    /// of the same name.
    Field(Option<&'a str>),
    /// `DocumentValueSourceDictionary`: a `LongValuesSource`.
    Source(&'a Arc<dyn LongValuesSource>),
}

/// `IndexableField.numericValue().longValue()`: `None` for a string or binary.
fn numeric_long(value: &FieldValue) -> Option<i64> {
    match value {
        FieldValue::Int(v) => Some(i64::from(*v)),
        FieldValue::Long(v) => Some(*v),
        FieldValue::Float(v) => Some(*v as i64),
        FieldValue::Double(v) => Some(*v as i64),
        FieldValue::String(_) | FieldValue::Binary(_) => None,
    }
}

/// `Double.toString`/`Float.toString` from Rust's shortest scientific form
/// (`{:e}`): decimal for `1e-3 <= |x| < 1e7`, else `d.dddE<exp>`.
fn java_fp_string(sci: &str) -> String {
    let (neg, body) = match sci.strip_prefix('-') {
        Some(b) => (true, b),
        None => (false, sci),
    };
    let (mantissa, exp) = body.split_once('e').unwrap_or((body, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let mut out = String::from(if neg { "-" } else { "" });
    if (-3..7).contains(&exp) {
        if exp >= 0 {
            let int_len = exp as usize + 1;
            let padded = format!("{digits:0<int_len$}");
            let (int, frac) = padded.split_at(int_len);
            out.push_str(int);
            out.push('.');
            out.push_str(if frac.is_empty() { "0" } else { frac });
        } else {
            out.push_str("0.");
            out.push_str(&"0".repeat((-exp - 1) as usize));
            out.push_str(&digits);
        }
    } else {
        out.push_str(&digits[..1]);
        out.push('.');
        out.push_str(if digits.len() > 1 { &digits[1..] } else { "0" });
        out.push('E');
        out.push_str(&exp.to_string());
    }
    out
}

fn java_double_string(d: f64) -> String {
    if d.is_nan() {
        "NaN".into()
    } else if d.is_infinite() {
        if d > 0.0 { "Infinity" } else { "-Infinity" }.into()
    } else if d == 0.0 {
        if d.is_sign_negative() { "-0.0" } else { "0.0" }.into()
    } else {
        java_fp_string(&format!("{d:e}"))
    }
}

fn java_float_string(f: f32) -> String {
    if f.is_nan() {
        "NaN".into()
    } else if f.is_infinite() {
        if f > 0.0 { "Infinity" } else { "-Infinity" }.into()
    } else if f == 0.0 {
        if f.is_sign_negative() { "-0.0" } else { "0.0" }.into()
    } else {
        java_fp_string(&format!("{f:e}"))
    }
}

/// `binaryValue()`, else `new BytesRef(stringValue())`: a number's
/// `stringValue()` is its `toString()`.
fn bytes_of(value: &FieldValue) -> Vec<u8> {
    match value {
        FieldValue::Binary(b) => b.clone(),
        FieldValue::String(s) => s.as_bytes().to_vec(),
        FieldValue::Int(v) => v.to_string().into_bytes(),
        FieldValue::Long(v) => v.to_string().into_bytes(),
        FieldValue::Float(v) => java_float_string(*v).into_bytes(),
        FieldValue::Double(v) => java_double_string(*v).into_bytes(),
    }
}

fn field_number(sr: &SegmentReader, name: &str) -> Option<i32> {
    sr.field_infos()
        .fields
        .iter()
        .find(|f| f.name == name)
        .map(|f| f.number)
}

/// `DocumentDictionary.DocumentInputIterator`, over every segment of
/// `reader`.
fn document_entries(
    reader: &DirectoryReader,
    field: &str,
    weight: &WeightSource<'_>,
    payload_field: Option<&str>,
    contexts_field: Option<&str>,
) -> Result<Vec<DictEntry>> {
    let mut entries = Vec::new();
    for sr in reader.segment_readers() {
        let wanted = |name: &str| field_number(sr, name);
        let (field_n, weight_n, payload_n, contexts_n) = (
            wanted(field),
            match weight {
                WeightSource::Field(Some(w)) => wanted(w),
                _ => None,
            },
            payload_field.and_then(wanted),
            contexts_field.and_then(wanted),
        );
        let mut numbers: Vec<i32> = [field_n, weight_n, payload_n, contexts_n]
            .into_iter()
            .flatten()
            .collect();
        numbers.sort_unstable();
        numbers.dedup();
        let weight_values = match weight {
            WeightSource::Source(source) => {
                let ctx = ValuesContext::for_reader(sr);
                Some(source.get_values(&ctx, 0, None)?)
            }
            WeightSource::Field(_) => None,
        };
        let mut weight_values = weight_values;
        let dv_entry = match (weight_n, weight) {
            (Some(n), WeightSource::Field(_)) => sr
                .doc_values_for_field(n)
                .and_then(|(meta, data)| meta.numeric_entry(n).map(|e| (e.clone(), data))),
            _ => None,
        };
        // derives the column's addressing once, not per document
        let mut dv_reader = dv_entry
            .as_ref()
            .map(|(entry, data)| lucene_codecs::doc_values::NumericReader::new(data, entry));
        for doc in 0..sr.max_doc {
            if sr.live_docs().is_some_and(|live| !live.get(doc as usize)) {
                continue;
            }
            let mut visitor = DocumentVisitor::for_fields(&numbers);
            let had_stored = sr.visit_stored_document(doc, &mut visitor)?;
            let stored = if had_stored {
                visitor.into_document().fields
            } else {
                Vec::new()
            };
            let values_of = |n: Option<i32>| -> Vec<&FieldValue> {
                n.map_or_else(Vec::new, |n| {
                    stored
                        .iter()
                        .filter(|f| f.field_number == n)
                        .map(|f| &f.value)
                        .collect()
                })
            };
            let term_values = values_of(field_n);
            if term_values.is_empty() {
                continue; // no values in this document
            }
            let payload = if payload_field.is_some() {
                // in case the iterator has payloads configured, an empty value
                // stands for a missing one
                values_of(payload_n)
                    .first()
                    .map(|v| bytes_of(v))
                    .unwrap_or_default()
            } else {
                Vec::new()
            };
            let contexts: BTreeSet<Vec<u8>> = if contexts_field.is_some() {
                values_of(contexts_n).into_iter().map(bytes_of).collect()
            } else {
                BTreeSet::new()
            };
            let weight = match weight {
                WeightSource::Field(_) => match values_of(weight_n).first() {
                    // found weight as stored
                    Some(w) => numeric_long(w).unwrap_or(0),
                    None => match dv_reader.as_mut() {
                        // found weight as a NumericDocValue
                        Some(reader) => reader
                            .value(doc)
                            .map_err(|e| Error::IllegalArgument(e.to_string()))?
                            .unwrap_or(0),
                        None => 0,
                    },
                },
                WeightSource::Source(_) => {
                    let values = weight_values.as_mut().expect("opened above");
                    if values.advance_exact(doc)? {
                        values.long_value()?
                    } else {
                        0
                    }
                }
            };
            for value in term_values {
                entries.push(DictEntry {
                    term: bytes_of(value),
                    weight,
                    payload: payload.clone(),
                    contexts: contexts.clone(),
                });
            }
        }
    }
    Ok(entries)
}

/// `DocumentDictionary`: the terms of a stored field, weighted by a numeric
/// field, with an optional payload field and contexts field.
pub struct DocumentDictionary<'r> {
    reader: &'r DirectoryReader,
    field: String,
    weight_field: Option<String>,
    payload_field: Option<String>,
    contexts_field: Option<String>,
}

impl<'r> DocumentDictionary<'r> {
    /// `new DocumentDictionary(reader, field, weightField)`.
    pub fn new(reader: &'r DirectoryReader, field: &str, weight_field: &str) -> Self {
        Self {
            reader,
            field: field.to_string(),
            weight_field: Some(weight_field.to_string()),
            payload_field: None,
            contexts_field: None,
        }
    }

    /// `new DocumentDictionary(reader, field, weightField, payloadField)`.
    pub fn with_payload(mut self, payload_field: &str) -> Self {
        self.payload_field = Some(payload_field.to_string());
        self
    }

    /// `new DocumentDictionary(reader, field, weightField, payloadField,
    /// contextsField)`.
    pub fn with_contexts(mut self, contexts_field: &str) -> Self {
        self.contexts_field = Some(contexts_field.to_string());
        self
    }
}

impl Dictionary for DocumentDictionary<'_> {
    fn get_entry_iterator(&self) -> Result<Box<dyn InputIterator>> {
        let entries = document_entries(
            self.reader,
            &self.field,
            &WeightSource::Field(self.weight_field.as_deref()),
            self.payload_field.as_deref(),
            self.contexts_field.as_deref(),
        )?;
        Ok(Box::new(EntriesInputIterator::new(
            entries,
            self.payload_field.is_some(),
            self.contexts_field.is_some(),
        )))
    }
}

/// `DocumentValueSourceDictionary`: as [`DocumentDictionary`], weighted by a
/// [`LongValuesSource`].
pub struct DocumentValueSourceDictionary<'r> {
    reader: &'r DirectoryReader,
    field: String,
    weights: Arc<dyn LongValuesSource>,
    payload_field: Option<String>,
    contexts_field: Option<String>,
}

impl<'r> DocumentValueSourceDictionary<'r> {
    /// `new DocumentValueSourceDictionary(reader, field, weightsValueSource)`.
    pub fn new(
        reader: &'r DirectoryReader,
        field: &str,
        weights: Arc<dyn LongValuesSource>,
    ) -> Self {
        Self {
            reader,
            field: field.to_string(),
            weights,
            payload_field: None,
            contexts_field: None,
        }
    }

    /// The `payload` constructor argument.
    pub fn with_payload(mut self, payload_field: &str) -> Self {
        self.payload_field = Some(payload_field.to_string());
        self
    }

    /// The `contexts` constructor argument.
    pub fn with_contexts(mut self, contexts_field: &str) -> Self {
        self.contexts_field = Some(contexts_field.to_string());
        self
    }
}

impl Dictionary for DocumentValueSourceDictionary<'_> {
    fn get_entry_iterator(&self) -> Result<Box<dyn InputIterator>> {
        let entries = document_entries(
            self.reader,
            &self.field,
            &WeightSource::Source(&self.weights),
            self.payload_field.as_deref(),
            self.contexts_field.as_deref(),
        )?;
        Ok(Box::new(EntriesInputIterator::new(
            entries,
            self.payload_field.is_some(),
            self.contexts_field.is_some(),
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_with_an_empty_delimiter_is_per_character() {
        assert_eq!(java_split("aé", ""), vec!["a", "é"]);
    }

    #[test]
    fn numeric_values_truncate_like_a_java_cast() {
        assert_eq!(numeric_long(&FieldValue::Int(-3)), Some(-3));
        assert_eq!(numeric_long(&FieldValue::Long(9)), Some(9));
        assert_eq!(numeric_long(&FieldValue::Float(2.9)), Some(2));
        assert_eq!(numeric_long(&FieldValue::Double(-2.9)), Some(-2));
        assert_eq!(numeric_long(&FieldValue::String("1".into())), None);
        assert_eq!(numeric_long(&FieldValue::Binary(vec![1])), None);
    }

    #[test]
    fn floating_point_strings_follow_java() {
        assert_eq!(java_double_string(f64::NAN), "NaN");
        assert_eq!(java_double_string(f64::INFINITY), "Infinity");
        assert_eq!(java_double_string(f64::NEG_INFINITY), "-Infinity");
        assert_eq!(java_double_string(0.0), "0.0");
        assert_eq!(java_double_string(-0.0), "-0.0");
        assert_eq!(java_double_string(1.5), "1.5");
        assert_eq!(java_double_string(1.0e7), "1.0E7");
        assert_eq!(java_float_string(f32::NAN), "NaN");
        assert_eq!(java_float_string(f32::INFINITY), "Infinity");
        assert_eq!(java_float_string(f32::NEG_INFINITY), "-Infinity");
        assert_eq!(java_float_string(0.0), "0.0");
        assert_eq!(java_float_string(-0.0), "-0.0");
        assert_eq!(java_float_string(2.5), "2.5");
    }
}
