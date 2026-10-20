//! `SimpleTextStoredFieldsReader`: the `.fld` written by
//! [`super::stored_fields_writer`], walked once at open (`readIndex`) for
//! where each document starts, then read a document at a time.
//!
//! What differs from Java:
//! - The reader borrows the file's bytes, not an `IndexInput`; values reach a
//!   [`StoredFieldVisitor`] by field number (this crate's visitor never
//!   holds a `FieldInfo`, so the reader takes no `FieldInfos`).
//! - `Float.parseFloat`/`Double.parseDouble` accept what Java's `toString`
//!   writes and the rest of Java's decimal grammar (`NaN`, `Infinity`, an
//!   `f`/`d` suffix, surrounding whitespace); hexadecimal floating-point
//!   literals are `Corrupted`. Java's `RuntimeException("unknown field
//!   type")`, its `NumberFormatException`s, its `assert`s on each line's
//!   prefix and on the document count, and its array index errors on a
//!   document count above `maxDoc` or a document outside it are `Corrupted`
//!   (`IllegalArgument` for the document number).

use super::stored_fields_writer::{
    DOC, END, FIELD, NAME, TYPE, TYPE_BINARY, TYPE_DOUBLE, TYPE_FLOAT, TYPE_INT, TYPE_LONG,
    TYPE_STRING, VALUE,
};
use super::util::{self, parse_int, parse_integer, parse_long, read_line, read_value, tail};
use super::{corrupt, Error, Result};
use crate::stored_fields::{Document, DocumentVisitor, StoredFieldVisitor, VisitStatus};

/// `SimpleTextStoredFieldsReader` over the whole `.fld`.
#[derive(Debug, Clone)]
pub struct SimpleTextStoredFieldsReader<'d> {
    data: &'d [u8],
    /// docid -> offset in the `.fld`, just past its `doc` line.
    offsets: Vec<usize>,
}

impl<'d> SimpleTextStoredFieldsReader<'d> {
    /// `new SimpleTextStoredFieldsReader(directory, si, fn, context)`:
    /// `readIndex(si.maxDoc())`.
    ///
    /// # Errors
    /// `Corrupted` for a document count other than `max_doc` or a bad
    /// checksum; `Eof` for a file without `END`.
    pub fn open(data: &'d [u8], max_doc: i32) -> Result<Self> {
        // Java: SimpleTextStoredFieldsReader.readIndex
        let size = usize::try_from(max_doc)
            .map_err(|_| Error::IllegalArgument(format!("invalid maxDoc {max_doc}")))?;
        // ALLOC: one offset per `doc N` line, at least 6 bytes each
        // (`doc 0` and its newline), so never more than `data.len() / 6`.
        let mut offsets = Vec::with_capacity(size.min(data.len() / 6));
        let mut scratch = Vec::new();
        let mut pos = 0usize;
        while scratch != END {
            read_line(data, &mut pos, &mut scratch)?;
            if scratch.starts_with(DOC) {
                if offsets.len() == size {
                    return Err(corrupt(format!("more than maxDoc={max_doc} documents")));
                }
                offsets.push(pos);
            }
        }
        util::check_footer(data, 0, &mut pos)?;
        if offsets.len() != size {
            return Err(corrupt(format!(
                "{} documents, expected maxDoc={max_doc}",
                offsets.len()
            )));
        }
        Ok(SimpleTextStoredFieldsReader { data, offsets })
    }

    /// The number of documents (`maxDoc`).
    pub fn max_doc(&self) -> usize {
        self.offsets.len()
    }

    /// `document(n, visitor)`.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for `n` outside the segment; `Corrupted`
    /// for a malformed line or an unknown type; what `visitor` returns.
    pub fn visit_document(&self, n: i32, visitor: &mut dyn StoredFieldVisitor) -> Result<()> {
        // Java: SimpleTextStoredFieldsReader.document
        let mut pos = usize::try_from(n)
            .ok()
            .and_then(|n| self.offsets.get(n).copied())
            .ok_or_else(|| Error::IllegalArgument(format!("document {n} out of bounds")))?;
        let data = self.data;
        let mut scratch = Vec::new();
        loop {
            read_line(data, &mut pos, &mut scratch)?;
            let Some(number) = tail(&scratch, FIELD) else {
                break;
            };
            let field_number = parse_int(number)?;
            read_value(data, &mut pos, &mut scratch, NAME)?;
            let ty = match read_value(data, &mut pos, &mut scratch, TYPE)? {
                t if t == TYPE_STRING => TYPE_STRING,
                t if t == TYPE_BINARY => TYPE_BINARY,
                t if t == TYPE_INT => TYPE_INT,
                t if t == TYPE_LONG => TYPE_LONG,
                t if t == TYPE_FLOAT => TYPE_FLOAT,
                t if t == TYPE_DOUBLE => TYPE_DOUBLE,
                _ => return Err(corrupt("unknown field type")),
            };
            match visitor.needs_field(field_number)? {
                VisitStatus::Yes => {
                    let value = read_value(data, &mut pos, &mut scratch, VALUE)?;
                    read_field(ty, field_number, value, visitor)?;
                }
                VisitStatus::No => {
                    read_value(data, &mut pos, &mut scratch, VALUE)?;
                }
                VisitStatus::Stop => return Ok(()),
            }
        }
        Ok(())
    }

    /// `document(n, new DocumentStoredFieldVisitor())`: every value of
    /// document `n`.
    ///
    /// # Errors
    /// What [`Self::visit_document`] fails with.
    pub fn document(&self, n: i32) -> Result<Document> {
        let mut visitor = DocumentVisitor::all();
        self.visit_document(n, &mut visitor)?;
        Ok(visitor.into_document())
    }
}

/// `readField(type, fieldInfo, visitor)`, after its `value` line.
fn read_field(
    ty: &[u8],
    field_number: i32,
    value: &[u8],
    visitor: &mut dyn StoredFieldVisitor,
) -> Result<()> {
    // Java: SimpleTextStoredFieldsReader.readField
    if ty == TYPE_STRING {
        visitor.string_field(field_number, &String::from_utf8_lossy(value))?;
    } else if ty == TYPE_BINARY {
        visitor.binary_field(field_number, value)?;
    } else if ty == TYPE_INT {
        visitor.int_field(field_number, parse_integer(value)?)?;
    } else if ty == TYPE_LONG {
        visitor.long_field(field_number, parse_long(value)?)?;
    } else if ty == TYPE_FLOAT {
        visitor.float_field(field_number, parse_java_float(value)?)?;
    } else {
        visitor.double_field(field_number, parse_java_double(value)?)?;
    }
    Ok(())
}

/// The decimal text `Float.parseFloat`/`Double.parseDouble` take, in the
/// shape Rust's `parse` takes: trimmed, without an `f`/`d` suffix, with
/// `NaN`/`Infinity` spelled exactly (Rust also takes `inf`, `nan`).
fn java_fp_text(bytes: &[u8]) -> Result<String> {
    let bad = || {
        corrupt(format!(
            "For input string: \"{}\"",
            String::from_utf8_lossy(bytes)
        ))
    };
    let s = std::str::from_utf8(bytes).map_err(|_| bad())?;
    // `String.trim()`: every char <= ' ' at either end.
    let s = s.trim_matches(|c: char| c <= ' ');
    let unsigned = s.strip_prefix(['+', '-']).unwrap_or(s);
    if unsigned == "NaN" || unsigned == "Infinity" {
        return Ok(s.to_string());
    }
    let body = unsigned
        .strip_suffix(['f', 'F', 'd', 'D'])
        .unwrap_or(unsigned);
    let well_formed = !body.is_empty()
        && body
            .bytes()
            .all(|b| b.is_ascii_digit() || matches!(b, b'.' | b'e' | b'E' | b'+' | b'-'))
        && body.bytes().any(|b| b.is_ascii_digit());
    if !well_formed {
        return Err(bad());
    }
    let sign = s.len().saturating_sub(unsigned.len());
    Ok(format!("{}{body}", s.get(..sign).unwrap_or("")))
}

/// `Float.parseFloat(String)`.
///
/// # Errors
/// `Corrupted` for text Java would not parse (its `NumberFormatException`),
/// and for hexadecimal literals.
pub fn parse_java_float(bytes: &[u8]) -> Result<f32> {
    let text = java_fp_text(bytes)?;
    text.parse::<f32>()
        .map_err(|_| corrupt(format!("For input string: \"{text}\"")))
}

/// `Double.parseDouble(String)`.
///
/// # Errors
/// As [`parse_java_float`].
pub fn parse_java_double(bytes: &[u8]) -> Result<f64> {
    let text = java_fp_text(bytes)?;
    text.parse::<f64>()
        .map_err(|_| corrupt(format!("For input string: \"{text}\"")))
}

#[cfg(test)]
mod tests {
    use super::super::stored_fields_writer::write;
    use super::*;
    use crate::field_infos::{FieldInfo, FieldInfos};
    use crate::stored_fields::{FieldValue, StoredField};

    fn file() -> Vec<u8> {
        let infos = FieldInfos::new(vec![FieldInfo::new("a", 0), FieldInfo::new("b", 1)]).unwrap();
        let v = |field_number, value| StoredField {
            field_number,
            value,
        };
        let docs = vec![
            Document {
                fields: vec![
                    v(0, FieldValue::String("s\\\n".into())),
                    v(1, FieldValue::Binary(vec![0, 0x0a])),
                    v(0, FieldValue::Int(i32::MIN)),
                    v(1, FieldValue::Long(5)),
                    v(0, FieldValue::Float(f32::NAN)),
                    v(1, FieldValue::Double(f64::NEG_INFINITY)),
                ],
            },
            Document::default(),
        ];
        write(&docs, &infos).unwrap()
    }

    fn reseal(mut body: Vec<u8>) -> Vec<u8> {
        util::write_checksum(&mut body);
        body
    }

    fn body(data: &[u8]) -> Vec<u8> {
        let at = data.windows(9).rposition(|w| w == b"checksum ").unwrap();
        data[..at].to_vec()
    }

    #[test]
    fn reads_what_the_writer_writes() {
        let data = file();
        let r = SimpleTextStoredFieldsReader::open(&data, 2).unwrap();
        assert_eq!(r.max_doc(), 2);
        let d = r.document(0).unwrap();
        let values: Vec<_> = d
            .fields
            .iter()
            .map(|f| (f.field_number, f.value.clone()))
            .collect();
        assert_eq!(values[0], (0, FieldValue::String("s\\\n".into())));
        assert_eq!(values[1], (1, FieldValue::Binary(vec![0, 0x0a])));
        assert_eq!(values[2], (0, FieldValue::Int(i32::MIN)));
        assert_eq!(values[3], (1, FieldValue::Long(5)));
        assert!(matches!(values[4].1, FieldValue::Float(f) if f.is_nan()));
        assert_eq!(values[5], (1, FieldValue::Double(f64::NEG_INFINITY)));
        assert!(r.document(1).unwrap().fields.is_empty());
        assert!(matches!(r.document(2), Err(Error::IllegalArgument(_))));
        assert!(matches!(r.document(-1), Err(Error::IllegalArgument(_))));
        // A visitor that wants only field 1, then one that stops at once.
        let mut only_b = DocumentVisitor::for_fields(&[1]);
        r.visit_document(0, &mut only_b).unwrap();
        assert_eq!(only_b.document().fields.len(), 3);
        struct Stop;
        impl StoredFieldVisitor for Stop {
            fn needs_field(&mut self, _: i32) -> crate::stored_fields::Result<VisitStatus> {
                Ok(VisitStatus::Stop)
            }
        }
        r.visit_document(0, &mut Stop).unwrap();
    }

    #[test]
    fn document_count_must_be_max_doc() {
        let data = file();
        assert!(SimpleTextStoredFieldsReader::open(&data, 1).is_err());
        assert!(SimpleTextStoredFieldsReader::open(&data, 3).is_err());
        assert!(matches!(
            SimpleTextStoredFieldsReader::open(&data, -1),
            Err(Error::IllegalArgument(_))
        ));
        assert!(SimpleTextStoredFieldsReader::open(&data[..data.len() - 1], 2).is_err());
        assert!(SimpleTextStoredFieldsReader::open(b"doc 0\n", 1).is_err());
    }

    #[test]
    fn damaged_lines_are_corrupted() {
        let data = file();
        let b = body(&data);
        let text = String::from_utf8(b).unwrap();
        for (from, to) in [
            ("type string", "type text"),
            ("    name a", "    nome a"),
            ("value -2147483648", "value -2147483649"),
            ("value 5", "value 5x"),
            ("value NaN", "value NaN?"),
            ("value -Infinity", "value 0x1p3"),
            ("  field 1", "  field x"),
        ] {
            assert!(text.contains(from), "{from}");
            let bad = reseal(text.replacen(from, to, 1).into_bytes());
            let r = SimpleTextStoredFieldsReader::open(&bad, 2).unwrap();
            assert!(r.document(0).is_err(), "{from} -> {to}");
        }
        // A `value` line missing where a skipped value should be.
        let bad = reseal(text.replacen("    value s", "    vxlue s", 1).into_bytes());
        let r = SimpleTextStoredFieldsReader::open(&bad, 2).unwrap();
        let mut none = DocumentVisitor::for_fields(&[]);
        assert!(r.visit_document(0, &mut none).is_err());
    }

    #[test]
    fn java_float_grammar() {
        assert_eq!(parse_java_float(b" 1.5f ").unwrap(), 1.5);
        assert_eq!(parse_java_float(b"+2E3").unwrap(), 2000.0);
        assert_eq!(parse_java_float(b"-Infinity").unwrap(), f32::NEG_INFINITY);
        assert!(parse_java_float(b"NaN").unwrap().is_nan());
        assert_eq!(parse_java_double(b".5d").unwrap(), 0.5);
        assert_eq!(parse_java_double(b"4.9E-324").unwrap(), 4.9e-324);
        for bad in [
            &b""[..],
            b"inf",
            b"nan",
            b"1e",
            b".",
            b"Infinityf",
            b"1-2",
            b"\xff",
            b"1__0",
        ] {
            assert!(parse_java_float(bad).is_err(), "{bad:?}");
            assert!(parse_java_double(bad).is_err(), "{bad:?}");
        }
    }
}
