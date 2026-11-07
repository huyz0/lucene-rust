//! OpenSearch k-NN's derived source (`KNN10010DerivedSourceStoredFieldsFormat`): a segment whose
//! `.si` attribute [`DERIVED_VECTOR_FIELDS`] lists vector fields stores each document's `_source`
//! with those fields' values masked (the number `1`). Reading puts each field's vector, from the
//! segment's vector values, back in place of the masked value (`DerivedSourceVectorTransformer`).
//!
//! The `_source` is spliced, not re-serialized: everything but the replaced values keeps its bytes.
//! A `_source` in another content type (SMILE, CBOR, YAML) is refused ([`stored_fields::Error::DerivedSource`]),
//! so the caller reads the document through Lucene's codec, which handles every content type; one
//! that is not content at all (a no-op tombstone's) is returned unchanged, as stock returns it.

use lucene_codecs::field_infos::{FieldInfo, FieldInfos, VectorEncoding};
use lucene_codecs::stored_fields::{self, StoredFieldVisitor, VisitStatus};

/// The `.si` attribute naming the segment's derived vector fields, comma separated
/// (`DerivedSourceSegmentAttributeParser.DERIVED_SOURCE_FIELD`).
pub const DERIVED_VECTOR_FIELDS: &str = "derived_vector_fields";
/// The same for vector fields inside nested objects (`NESTED_DERIVED_SOURCE_FIELD`).
pub const NESTED_DERIVED_VECTOR_FIELDS: &str = "nested_derived_vector_fields";
/// The field attribute `KNN9120Codec` marks a derived field with (value `true`).
pub const FIELD_ATTRIBUTE: &str = "knn-derived-source-enabled";
/// OpenSearch's `_source` stored field (`SourceFieldMapper.NAME`).
pub const SOURCE_FIELD: &str = "_source";

/// A segment's derived vector fields, resolved against its field infos.
#[derive(Debug, Clone)]
pub struct DerivedFields {
    source_number: i32,
    fields: Vec<DerivedField>,
    nested: bool,
}

#[derive(Debug, Clone)]
struct DerivedField {
    path: String,
    number: i32,
    encoding: VectorEncoding,
}

impl DerivedFields {
    /// The derived fields a segment's `.si` attributes list, `None` when it lists none
    /// (`KNN10010DerivedSourceStoredFieldsFormat.fieldsReader`). A listed field the segment has no
    /// field info for is skipped, as stock skips it.
    pub fn from_attributes(attributes: &[(String, String)], infos: &FieldInfos) -> Option<Self> {
        let names = |key: &str| -> Vec<&str> {
            attributes
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v.split(',').filter(|s| !s.is_empty()).collect())
                .unwrap_or_default()
        };
        let root = names(DERIVED_VECTOR_FIELDS);
        let nested = names(NESTED_DERIVED_VECTOR_FIELDS);
        let resolved = |list: &[&str]| -> Vec<DerivedField> {
            list.iter()
                .filter_map(|name| field_info(infos, name))
                .map(|f| DerivedField {
                    path: f.name.clone(),
                    number: f.number,
                    encoding: f.vector_encoding,
                })
                .collect()
        };
        let mut fields = resolved(&root);
        // `KNN9120Codec`'s segments mark the fields themselves.
        for f in &infos.fields {
            let marked = f
                .attributes
                .iter()
                .any(|(k, v)| k == FIELD_ATTRIBUTE && v == "true");
            if marked && !fields.iter().any(|d| d.number == f.number) {
                fields.push(DerivedField {
                    path: f.name.clone(),
                    number: f.number,
                    encoding: f.vector_encoding,
                });
            }
        }
        let nested_fields = resolved(&nested);
        if fields.is_empty() && nested_fields.is_empty() {
            return None;
        }
        let source_number = infos.field_by_name(SOURCE_FIELD)?.number;
        Some(Self {
            source_number,
            fields,
            nested: !nested_fields.is_empty(),
        })
    }

    /// The `_source` field's number.
    pub fn source_number(&self) -> i32 {
        self.source_number
    }
}

/// `KNN10010DerivedSourceStoredFieldsFormat.getFieldInfo`: the field by its name, else the one
/// field whose name matches ignoring case (one with vectors before one without; two with vectors
/// match none).
fn field_info<'a>(infos: &'a FieldInfos, name: &str) -> Option<&'a FieldInfo> {
    if let Some(f) = infos.field_by_name(name) {
        return Some(f);
    }
    let mut matched: Option<&FieldInfo> = None;
    for candidate in &infos.fields {
        if !candidate.name.eq_ignore_ascii_case(name) {
            continue;
        }
        let has = |f: &FieldInfo| f.vector_dimension > 0;
        match matched {
            None => matched = Some(candidate),
            Some(m) if has(candidate) && !has(m) => matched = Some(candidate),
            Some(m) if has(m) && has(candidate) => return None,
            Some(_) => {}
        }
    }
    matched
}

/// A document's vector for a derived field: `Ok(None)` when the document has none.
pub trait VectorSource {
    /// The float vector of field `number` for the current document.
    fn float_vector(&mut self, number: i32) -> stored_fields::Result<Option<Vec<f32>>>;
    /// The byte vector of field `number` for the current document.
    fn byte_vector(&mut self, number: i32) -> stored_fields::Result<Option<Vec<u8>>>;
}

fn refused(why: &str) -> stored_fields::Error {
    stored_fields::Error::DerivedSource(why.to_string())
}

/// `DerivedSourceVectorTransformer.injectVectors`: `source` with each derived field's non-null
/// value replaced by the document's vector (`[f, ...]` for floats as Java's `Float.toString`
/// prints them, `[i, ...]` for bytes, signed).
///
/// # Errors
/// [`stored_fields::Error::DerivedSource`] for a `_source` in another content type, a segment
/// with nested derived fields, or a vector that cannot be read.
pub fn inject(
    fields: &DerivedFields,
    source: &[u8],
    vectors: &mut dyn VectorSource,
) -> stored_fields::Result<Vec<u8>> {
    if fields.nested {
        return Err(refused(
            "nested derived vector fields are read by the codec",
        ));
    }
    match content_type(source) {
        Content::Json => {}
        Content::Other => return Err(refused("_source is not JSON")),
        Content::None => return Ok(source.to_vec()),
    }
    let paths: Vec<&str> = fields.fields.iter().map(|f| f.path.as_str()).collect();
    let spans = value_spans(source, &paths)?;
    let mut out = Vec::with_capacity(source.len() + spans.len() * 64);
    let mut at = 0;
    for (start, end, which) in spans {
        let field = &fields.fields[which];
        let text = match field.encoding {
            VectorEncoding::Float32 => vectors.float_vector(field.number)?.map(|v| {
                let mut s = String::with_capacity(v.len() * 12 + 2);
                s.push('[');
                for (i, x) in v.iter().enumerate() {
                    if i > 0 {
                        s.push(',');
                    }
                    s.push_str(&lucene_analysis::attributes::java_float_to_string(*x));
                }
                s.push(']');
                s
            }),
            VectorEncoding::Byte => vectors.byte_vector(field.number)?.map(|v| {
                let mut s = String::with_capacity(v.len() * 4 + 2);
                s.push('[');
                for (i, b) in v.iter().enumerate() {
                    if i > 0 {
                        s.push(',');
                    }
                    s.push_str(&i32::from(*b as i8).to_string());
                }
                s.push(']');
                s
            }),
        };
        let Some(text) = text else { continue };
        out.extend_from_slice(&source[at..start]);
        out.extend_from_slice(text.as_bytes());
        at = end;
    }
    out.extend_from_slice(&source[at..]);
    Ok(out)
}

enum Content {
    Json,
    Other,
    None,
}

/// `MediaTypeRegistry.xContentType` as `XContentHelper.convertToMap` detects it: JSON from its
/// opening brace, SMILE, CBOR and YAML from their headers; anything else is not content.
fn content_type(source: &[u8]) -> Content {
    let first = source
        .iter()
        .position(|b| !matches!(b, b' ' | b'\t' | b'\n' | b'\r'));
    match first.map(|i| source[i]) {
        Some(b'{') => Content::Json,
        _ if source.starts_with(b":)\n") || source.starts_with(b"---") => Content::Other,
        // A CBOR map (major type 5), or the self-describe tag.
        Some(0xa0..=0xbf) | Some(0xd9) if first == Some(0) => Content::Other,
        _ => Content::None,
    }
}

/// The byte spans of the non-null values at `paths` (keys joined by `.`, objects only), in order,
/// with the index of the path each one is at.
fn value_spans(source: &[u8], paths: &[&str]) -> stored_fields::Result<Vec<(usize, usize, usize)>> {
    let mut p = Scanner { b: source, i: 0 };
    p.ws();
    if p.peek() != Some(b'{') {
        return Err(refused("_source is not a JSON object"));
    }
    let mut spans = Vec::new();
    let mut path = String::new();
    p.object(&mut path, paths, &mut spans, 0)?;
    p.ws();
    if p.i != source.len() {
        return Err(refused("trailing bytes after the _source object"));
    }
    Ok(spans)
}

const MAX_DEPTH: usize = 512;

struct Scanner<'a> {
    b: &'a [u8],
    i: usize,
}

impl Scanner<'_> {
    fn peek(&self) -> Option<u8> {
        self.b.get(self.i).copied()
    }

    fn ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.i += 1;
        }
    }

    fn eat(&mut self, c: u8) -> stored_fields::Result<()> {
        self.ws();
        if self.peek() == Some(c) {
            self.i += 1;
            Ok(())
        } else {
            Err(refused("malformed JSON _source"))
        }
    }

    fn object(
        &mut self,
        path: &mut String,
        paths: &[&str],
        spans: &mut Vec<(usize, usize, usize)>,
        depth: usize,
    ) -> stored_fields::Result<()> {
        if depth > MAX_DEPTH {
            return Err(refused("_source nests too deep"));
        }
        self.eat(b'{')?;
        self.ws();
        if self.peek() == Some(b'}') {
            self.i += 1;
            return Ok(());
        }
        loop {
            self.ws();
            let key = self.string()?;
            self.eat(b':')?;
            self.ws();
            let base = path.len();
            if base > 0 {
                path.push('.');
            }
            path.push_str(&key);
            let start = self.i;
            match self.peek() {
                Some(b'{') if !paths.contains(&path.as_str()) => {
                    self.object(path, paths, spans, depth + 1)?;
                }
                _ => {
                    self.skip_value(depth + 1)?;
                    if let Some(which) = paths.iter().position(|p| *p == path.as_str()) {
                        if &self.b[start..self.i] != b"null" {
                            spans.push((start, self.i, which));
                        }
                    }
                }
            }
            path.truncate(base);
            self.ws();
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    return Ok(());
                }
                _ => return Err(refused("malformed JSON _source")),
            }
        }
    }

    fn string(&mut self) -> stored_fields::Result<String> {
        if self.peek() != Some(b'"') {
            return Err(refused("malformed JSON _source"));
        }
        self.i += 1;
        let mut out = Vec::new();
        loop {
            let Some(c) = self.peek() else {
                return Err(refused("unterminated JSON string"));
            };
            self.i += 1;
            match c {
                b'"' => break,
                b'\\' => {
                    let Some(e) = self.peek() else {
                        return Err(refused("unterminated JSON string"));
                    };
                    self.i += 1;
                    match e {
                        b'"' | b'\\' | b'/' => out.push(e),
                        b'b' => out.push(8),
                        b'f' => out.push(12),
                        b'n' => out.push(b'\n'),
                        b'r' => out.push(b'\r'),
                        b't' => out.push(b'\t'),
                        b'u' => {
                            let hi = self.hex4()?;
                            let low = (0xD800..0xDC00).contains(&hi)
                                && self.b.get(self.i..self.i + 2) == Some(b"\\u")
                                && self
                                    .b
                                    .get(self.i + 2..self.i + 6)
                                    .and_then(|h| std::str::from_utf8(h).ok())
                                    .and_then(|h| u16::from_str_radix(h, 16).ok())
                                    .is_some_and(|lo| (0xDC00..0xE000).contains(&lo));
                            let code = if low {
                                self.i += 2;
                                let lo = self.hex4()?;
                                0x10000
                                    + ((u32::from(hi) - 0xD800) << 10)
                                    + (u32::from(lo) - 0xDC00)
                            } else {
                                // A lone surrogate is U+FFFD; what follows is read on its own.
                                u32::from(hi)
                            };
                            let ch = char::from_u32(code).unwrap_or('\u{FFFD}');
                            let mut buf = [0u8; 4];
                            out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                        }
                        _ => return Err(refused("bad JSON escape")),
                    }
                }
                _ => out.push(c),
            }
        }
        String::from_utf8(out).map_err(|_| refused("JSON key is not UTF-8"))
    }

    fn hex4(&mut self) -> stored_fields::Result<u16> {
        let digits = self
            .b
            .get(self.i..self.i + 4)
            .and_then(|d| std::str::from_utf8(d).ok())
            .and_then(|d| u16::from_str_radix(d, 16).ok())
            .ok_or_else(|| refused("bad JSON \\u escape"))?;
        self.i += 4;
        Ok(digits)
    }

    fn skip_value(&mut self, depth: usize) -> stored_fields::Result<()> {
        if depth > MAX_DEPTH {
            return Err(refused("_source nests too deep"));
        }
        self.ws();
        match self.peek() {
            Some(b'"') => {
                self.i += 1;
                loop {
                    match self.peek() {
                        None => return Err(refused("unterminated JSON string")),
                        Some(b'"') => {
                            self.i += 1;
                            return Ok(());
                        }
                        Some(b'\\') => self.i += 2,
                        Some(_) => self.i += 1,
                    }
                }
            }
            Some(open @ (b'{' | b'[')) => {
                let close = if open == b'{' { b'}' } else { b']' };
                self.i += 1;
                self.ws();
                if self.peek() == Some(close) {
                    self.i += 1;
                    return Ok(());
                }
                loop {
                    if open == b'{' {
                        self.ws();
                        self.string()?;
                        self.eat(b':')?;
                    }
                    self.skip_value(depth + 1)?;
                    self.ws();
                    match self.peek() {
                        Some(b',') => self.i += 1,
                        Some(c) if c == close => {
                            self.i += 1;
                            return Ok(());
                        }
                        _ => return Err(refused("malformed JSON _source")),
                    }
                }
            }
            Some(_) => {
                let start = self.i;
                while matches!(self.peek(), Some(c) if !matches!(c, b',' | b'}' | b']' | b' ' | b'\t' | b'\n' | b'\r'))
                {
                    self.i += 1;
                }
                if self.i == start {
                    Err(refused("malformed JSON _source"))
                } else {
                    Ok(())
                }
            }
            None => Err(refused("malformed JSON _source")),
        }
    }
}

/// A visitor that hands `inner` the `_source` with the derived fields' vectors put back
/// (`DerivedSourceStoredFieldVisitor`).
pub struct InjectingVisitor<'a> {
    pub inner: &'a mut dyn StoredFieldVisitor,
    pub fields: &'a DerivedFields,
    pub vectors: &'a mut dyn VectorSource,
}

impl StoredFieldVisitor for InjectingVisitor<'_> {
    fn needs_field(&mut self, field_number: i32) -> stored_fields::Result<VisitStatus> {
        self.inner.needs_field(field_number)
    }

    fn string_field(&mut self, field_number: i32, value: &str) -> stored_fields::Result<()> {
        self.inner.string_field(field_number, value)
    }

    fn binary_field(&mut self, field_number: i32, value: &[u8]) -> stored_fields::Result<()> {
        if field_number == self.fields.source_number {
            let injected = inject(self.fields, value, self.vectors)?;
            return self.inner.binary_field(field_number, &injected);
        }
        self.inner.binary_field(field_number, value)
    }

    fn int_field(&mut self, field_number: i32, value: i32) -> stored_fields::Result<()> {
        self.inner.int_field(field_number, value)
    }

    fn long_field(&mut self, field_number: i32, value: i64) -> stored_fields::Result<()> {
        self.inner.long_field(field_number, value)
    }

    fn float_field(&mut self, field_number: i32, value: f32) -> stored_fields::Result<()> {
        self.inner.float_field(field_number, value)
    }

    fn double_field(&mut self, field_number: i32, value: f64) -> stored_fields::Result<()> {
        self.inner.double_field(field_number, value)
    }

    fn document_size(&mut self, serialized_len: usize, num_stored_fields: i64) {
        self.inner.document_size(serialized_len, num_stored_fields);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fixed(Vec<f32>, Vec<u8>);
    impl VectorSource for Fixed {
        fn float_vector(&mut self, _: i32) -> stored_fields::Result<Option<Vec<f32>>> {
            Ok(Some(self.0.clone()))
        }
        fn byte_vector(&mut self, _: i32) -> stored_fields::Result<Option<Vec<u8>>> {
            Ok(Some(self.1.clone()))
        }
    }

    fn fields(paths: &[(&str, VectorEncoding)]) -> DerivedFields {
        DerivedFields {
            source_number: 0,
            fields: paths
                .iter()
                .enumerate()
                .map(|(i, (p, e))| DerivedField {
                    path: (*p).to_string(),
                    number: i as i32 + 1,
                    encoding: *e,
                })
                .collect(),
            nested: false,
        }
    }

    fn run(paths: &[(&str, VectorEncoding)], src: &str) -> stored_fields::Result<String> {
        let f = fields(paths);
        let mut v = Fixed(vec![0.5, -1.0, 1e-5], vec![1, 255]);
        inject(&f, src.as_bytes(), &mut v).map(|b| String::from_utf8(b).unwrap())
    }

    #[test]
    fn masked_values_get_the_vectors_back() {
        let got = run(
            &[("emb", VectorEncoding::Float32)],
            r#"{"tag":"a","emb":1,"n":3}"#,
        )
        .unwrap();
        assert_eq!(got, r#"{"tag":"a","emb":[0.5,-1.0,1.0E-5],"n":3}"#);
        let got = run(&[("b", VectorEncoding::Byte)], r#"{ "b" : 1 }"#).unwrap();
        assert_eq!(got, r#"{ "b" : [1,-1] }"#);
    }

    #[test]
    fn paths_follow_objects_and_dotted_keys() {
        let paths = [("o.emb", VectorEncoding::Float32)];
        assert_eq!(
            run(&paths, r#"{"o":{"x":[1,{"emb":2}],"emb":1}}"#).unwrap(),
            r#"{"o":{"x":[1,{"emb":2}],"emb":[0.5,-1.0,1.0E-5]}}"#
        );
        assert_eq!(
            run(&paths, r#"{"o.emb":1,"s":"o.emb \" }"}"#).unwrap(),
            r#"{"o.emb":[0.5,-1.0,1.0E-5],"s":"o.emb \" }"}"#
        );
    }

    #[test]
    fn null_and_absent_values_stay() {
        let paths = [("emb", VectorEncoding::Float32)];
        assert_eq!(run(&paths, r#"{"emb":null}"#).unwrap(), r#"{"emb":null}"#);
        assert_eq!(run(&paths, r#"{"x":1}"#).unwrap(), r#"{"x":1}"#);
    }

    #[test]
    fn a_source_in_another_content_type_or_malformed_is_refused() {
        let paths = [("emb", VectorEncoding::Float32)];
        for src in [":)\n\u{3}", "---\nemb: 1", r#"{"emb":1"#, r#"{"emb":1} x"#] {
            assert!(
                matches!(
                    run(&paths, src),
                    Err(stored_fields::Error::DerivedSource(_))
                ),
                "{src}"
            );
        }
    }

    #[test]
    fn a_lone_surrogate_in_a_key_does_not_hide_what_follows() {
        let paths = [("\u{FFFD}A", VectorEncoding::Float32)];
        assert_eq!(
            run(&paths, r#"{"\ud800\u0041":1}"#).unwrap(),
            r#"{"\ud800\u0041":[0.5,-1.0,1.0E-5]}"#
        );
    }

    #[test]
    fn a_source_that_is_not_content_is_kept() {
        let paths = [("emb", VectorEncoding::Float32)];
        for src in ["", "[1]", "noop"] {
            assert_eq!(run(&paths, src).unwrap(), src);
        }
    }

    #[test]
    fn fields_resolve_from_attributes_ignoring_case() {
        let infos = FieldInfos {
            fields: vec![
                FieldInfo::new("_source", 0),
                FieldInfo::new("Emb", 1).with_vectors(
                    4,
                    VectorEncoding::Float32,
                    lucene_codecs::field_infos::VectorSimilarityFunction::Euclidean,
                ),
            ],
        };
        let attrs = vec![(DERIVED_VECTOR_FIELDS.to_string(), "emb,missing".to_string())];
        let f = DerivedFields::from_attributes(&attrs, &infos).unwrap();
        assert_eq!(f.fields.len(), 1);
        assert_eq!(f.fields[0].path, "Emb");
        assert!(DerivedFields::from_attributes(&[], &infos).is_none());
    }
}
