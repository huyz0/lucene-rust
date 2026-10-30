//! `significant_text`'s per-document text (`SignificantTextAggregatorFactory`'s
//! `SignificantTextCollectorSource`): the `_source` values at each source
//! field path (`SourceLookup.extractRawValues`), analyzed with the field's
//! index analyzer, optionally through OpenSearch's `DeDuplicatingTokenFilter`
//! (with a `DuplicateByteSequenceSpotter` per owning bucket), and each term
//! once per document.
//!
//! Only what Java would read the same way is answered: a JSON `_source`
//! (not SMILE, CBOR, YAML or a compressed one), without duplicate keys or
//! unpaired surrogates. Anything else is [`Refused`], and the caller runs
//! the aggregation in Java.

use std::collections::{HashMap, HashSet};

use lucene_analysis::factory::CustomAnalyzer;
use lucene_analysis::TokenStream;
use lucene_codecs::stored_fields::{StoredFieldVisitor, VisitStatus};

/// `SourceFieldMapper.NAME`: the stored field holding a document's source.
pub const SOURCE_FIELD: &str = "_source";

/// Reads one document's `_source` bytes (`SourceLookup`'s fields visitor).
pub struct SourceVisitor {
    number: i32,
    pub bytes: Option<Vec<u8>>,
}

impl SourceVisitor {
    pub fn new(number: i32) -> Self {
        SourceVisitor {
            number,
            bytes: None,
        }
    }
}

impl StoredFieldVisitor for SourceVisitor {
    fn needs_field(
        &mut self,
        field_number: i32,
    ) -> lucene_codecs::stored_fields::Result<VisitStatus> {
        Ok(if self.bytes.is_some() {
            VisitStatus::Stop
        } else if field_number == self.number {
            VisitStatus::Yes
        } else {
            VisitStatus::No
        })
    }

    fn binary_field(
        &mut self,
        _field_number: i32,
        value: &[u8],
    ) -> lucene_codecs::stored_fields::Result<()> {
        self.bytes = Some(value.to_vec());
        Ok(())
    }
}

/// A `_source` this module does not read as Java would.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refused(pub String);

// ------------------------------------------------------------------ JSON

/// A parsed JSON value, as OpenSearch's map of the source holds it.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    /// An integer: `Integer`, `Long` or `BigInteger` in Java, all printed
    /// as their decimal digits.
    Int(String),
    /// A floating-point number: a `Double`.
    Float(f64),
    Str(String),
    Arr(Vec<Json>),
    /// In document order; keys are distinct (Java's parser rejects
    /// duplicates).
    Obj(Vec<(String, Json)>),
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
    depth: usize,
}

/// Nesting Java's parser would accept before this one gives up.
const MAX_DEPTH: usize = 1000;

fn refuse<T>(why: &str) -> Result<T, Refused> {
    Err(Refused(why.to_string()))
}

impl Parser<'_> {
    fn ws(&mut self) {
        while let Some(&c) = self.b.get(self.i) {
            if matches!(c, b' ' | b'\t' | b'\n' | b'\r') {
                self.i += 1;
            } else {
                break;
            }
        }
    }

    fn peek(&self) -> Option<u8> {
        self.b.get(self.i).copied()
    }

    fn eat(&mut self, lit: &[u8]) -> Result<(), Refused> {
        if self.b.get(self.i..self.i + lit.len()) == Some(lit) {
            self.i += lit.len();
            Ok(())
        } else {
            refuse("malformed JSON")
        }
    }

    fn value(&mut self) -> Result<Json, Refused> {
        self.ws();
        match self.peek() {
            Some(b'{') => {
                self.depth += 1;
                if self.depth > MAX_DEPTH {
                    return refuse("JSON nested too deep");
                }
                self.i += 1;
                let mut fields: Vec<(String, Json)> = Vec::new();
                let mut seen: HashMap<String, ()> = HashMap::new();
                self.ws();
                if self.peek() == Some(b'}') {
                    self.i += 1;
                } else {
                    loop {
                        self.ws();
                        if self.peek() != Some(b'"') {
                            return refuse("malformed JSON");
                        }
                        let key = self.string()?;
                        if seen.insert(key.clone(), ()).is_some() {
                            return refuse("duplicate JSON key");
                        }
                        self.ws();
                        self.eat(b":")?;
                        let v = self.value()?;
                        fields.push((key, v));
                        self.ws();
                        match self.peek() {
                            Some(b',') => self.i += 1,
                            Some(b'}') => {
                                self.i += 1;
                                break;
                            }
                            _ => return refuse("malformed JSON"),
                        }
                    }
                }
                self.depth -= 1;
                Ok(Json::Obj(fields))
            }
            Some(b'[') => {
                self.depth += 1;
                if self.depth > MAX_DEPTH {
                    return refuse("JSON nested too deep");
                }
                self.i += 1;
                let mut items = Vec::new();
                self.ws();
                if self.peek() == Some(b']') {
                    self.i += 1;
                } else {
                    loop {
                        items.push(self.value()?);
                        self.ws();
                        match self.peek() {
                            Some(b',') => self.i += 1,
                            Some(b']') => {
                                self.i += 1;
                                break;
                            }
                            _ => return refuse("malformed JSON"),
                        }
                    }
                }
                self.depth -= 1;
                Ok(Json::Arr(items))
            }
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') => self.eat(b"true").map(|_| Json::Bool(true)),
            Some(b'f') => self.eat(b"false").map(|_| Json::Bool(false)),
            Some(b'n') => self.eat(b"null").map(|_| Json::Null),
            Some(b'-' | b'0'..=b'9') => self.number(),
            _ => refuse("malformed JSON"),
        }
    }

    fn number(&mut self) -> Result<Json, Refused> {
        let start = self.i;
        if self.peek() == Some(b'-') {
            self.i += 1;
        }
        let digits = |p: &mut Self| {
            let s = p.i;
            while matches!(p.peek(), Some(b'0'..=b'9')) {
                p.i += 1;
            }
            p.i - s
        };
        let int_start = self.i;
        let n = digits(self);
        if n == 0 || (n > 1 && self.b[int_start] == b'0') {
            return refuse("malformed JSON number");
        }
        let mut float = false;
        if self.peek() == Some(b'.') {
            self.i += 1;
            if digits(self) == 0 {
                return refuse("malformed JSON number");
            }
            float = true;
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.i += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.i += 1;
            }
            if digits(self) == 0 {
                return refuse("malformed JSON number");
            }
            float = true;
        }
        let text =
            std::str::from_utf8(&self.b[start..self.i]).map_err(|_| Refused("number".into()))?;
        if float {
            let v: f64 = text
                .parse()
                .map_err(|_| Refused("malformed JSON number".into()))?;
            Ok(Json::Float(v))
        } else {
            // Integer.toString / Long.toString: "-0" prints "0"; a
            // BigInteger (past a long) prints its digits as written.
            Ok(Json::Int(match text.parse::<i64>() {
                Ok(v) => v.to_string(),
                Err(_) => text.to_string(),
            }))
        }
    }

    fn hex4(&mut self) -> Result<u16, Refused> {
        let h = self
            .b
            .get(self.i..self.i + 4)
            .and_then(|h| std::str::from_utf8(h).ok())
            .and_then(|h| u16::from_str_radix(h, 16).ok())
            .ok_or_else(|| Refused("malformed JSON escape".into()))?;
        self.i += 4;
        Ok(h)
    }

    fn string(&mut self) -> Result<String, Refused> {
        self.i += 1; // the opening quote
        let mut out = String::new();
        loop {
            let start = self.i;
            while let Some(&c) = self.b.get(self.i) {
                if c == b'"' || c == b'\\' || c < 0x20 {
                    break;
                }
                self.i += 1;
            }
            out.push_str(
                std::str::from_utf8(&self.b[start..self.i])
                    .map_err(|_| Refused("JSON string is not UTF-8".into()))?,
            );
            match self.peek() {
                Some(b'"') => {
                    self.i += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.i += 1;
                    let e = self
                        .peek()
                        .ok_or_else(|| Refused("malformed JSON".into()))?;
                    self.i += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let u = self.hex4()?;
                            if (0xD800..0xDC00).contains(&u) {
                                if self.b.get(self.i..self.i + 2) != Some(b"\\u") {
                                    return refuse("unpaired surrogate");
                                }
                                self.i += 2;
                                let lo = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&lo) {
                                    return refuse("unpaired surrogate");
                                }
                                let c = 0x10000
                                    + ((u32::from(u) - 0xD800) << 10)
                                    + (u32::from(lo) - 0xDC00);
                                out.push(
                                    char::from_u32(c).ok_or_else(|| Refused("surrogate".into()))?,
                                );
                            } else if (0xDC00..0xE000).contains(&u) {
                                return refuse("unpaired surrogate");
                            } else {
                                out.push(
                                    char::from_u32(u32::from(u))
                                        .ok_or_else(|| Refused("escape".into()))?,
                                );
                            }
                        }
                        _ => return refuse("malformed JSON escape"),
                    }
                }
                _ => return refuse("malformed JSON string"),
            }
        }
    }
}

/// `_source`'s bytes as the map Java would build, or [`Refused`] for a
/// format this module does not read.
pub fn parse_source(bytes: &[u8]) -> Result<Json, Refused> {
    let mut p = Parser {
        b: bytes,
        i: 0,
        depth: 0,
    };
    p.ws();
    if p.peek() != Some(b'{') {
        // SMILE (":)\n"), CBOR, YAML, a compressed source, or not an object.
        return refuse("_source is not a JSON object");
    }
    let v = p.value()?;
    p.ws();
    if p.i != bytes.len() {
        return refuse("trailing bytes after the JSON source");
    }
    Ok(v)
}

/// `String.split("\\.")`: trailing empty strings dropped.
fn split_path(path: &str) -> Vec<&str> {
    if !path.contains('.') {
        return vec![path];
    }
    let mut parts: Vec<&str> = path.split('.').collect();
    while parts.last() == Some(&"") {
        parts.pop();
    }
    parts
}

fn get<'a>(obj: &'a [(String, Json)], key: &str) -> Option<&'a Json> {
    obj.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

/// `XContentMapValues.extractRawValues(path, map)`: the scalars at `path`,
/// through objects and arrays, a dotted key standing for several path
/// elements.
pub fn extract_raw_values<'a>(path: &str, source: &'a Json) -> Vec<&'a Json> {
    let mut out = Vec::new();
    let elements = split_path(path);
    if let (false, Json::Obj(map)) = (elements.is_empty(), source) {
        from_map(&mut out, map, &elements, 0);
    }
    out
}

fn from_map<'a>(
    out: &mut Vec<&'a Json>,
    part: &'a [(String, Json)],
    elements: &[&str],
    index: usize,
) {
    if index == elements.len() {
        return;
    }
    let mut key = elements[index].to_string();
    let mut current = get(part, &key).filter(|v| !matches!(v, Json::Null));
    let mut next = index + 1;
    while current.is_none() && next != elements.len() {
        key.push('.');
        key.push_str(elements[next]);
        current = get(part, &key).filter(|v| !matches!(v, Json::Null));
        next += 1;
    }
    match current {
        None => {}
        Some(Json::Obj(map)) => from_map(out, map, elements, next),
        Some(Json::Arr(list)) => from_list(out, list, elements, next),
        Some(v) => out.push(v),
    }
}

fn from_list<'a>(out: &mut Vec<&'a Json>, part: &'a [Json], elements: &[&str], index: usize) {
    for v in part {
        match v {
            Json::Null => {}
            Json::Obj(map) => from_map(out, map, elements, index),
            Json::Arr(list) => from_list(out, list, elements, index),
            v => out.push(v),
        }
    }
}

/// `obj.toString()` of a scalar from the source map.
pub fn display(v: &Json) -> Option<String> {
    Some(match v {
        Json::Str(s) => s.clone(),
        Json::Int(s) => s.clone(),
        Json::Bool(b) => b.to_string(),
        Json::Float(f) => lucene_util::geo::java_double_string(*f),
        Json::Null | Json::Arr(_) | Json::Obj(_) => return None,
    })
}

// --------------------------------------------------------- de-duplication

/// `DuplicateByteSequenceSpotter`: how often each 6-byte window of a
/// stream of token-hash bytes was seen before, counted to 255.
#[derive(Default)]
pub struct Spotter {
    buffer: [u8; 6],
    filled: bool,
    next: usize,
    counts: HashMap<[u8; 6], u8>,
}

impl Spotter {
    /// `startNewSequence()`.
    pub fn start_new_sequence(&mut self) {
        self.filled = false;
        self.next = 0;
    }

    /// `addByte(b)`: the window ending here's prior sightings.
    pub fn add_byte(&mut self, b: u8) -> i16 {
        self.buffer[self.next] = b;
        self.next += 1;
        if self.next >= 6 {
            self.next = 0;
            self.filled = true;
        }
        if !self.filled {
            return 0;
        }
        let mut key = [0u8; 6];
        for (i, k) in key.iter_mut().enumerate() {
            *k = self.buffer[(self.next + i) % 6];
        }
        let c = self.counts.entry(key).or_insert(0);
        let prior = *c;
        if prior < 255 {
            *c += 1;
        }
        // `(short) (node.add(sequence << 8) - 1)`: a new window's add
        // answers 1, a known one its count before this sighting plus 1.
        i16::from(prior)
    }
}

/// `MurmurHash3.hash128(bytes, 0, len, 0).h1` (x64, 128-bit).
fn murmur_h1(key: &[u8]) -> i64 {
    const C1: u64 = 0x87c3_7b91_1142_53d5;
    const C2: u64 = 0x4cf5_ad43_2745_937f;
    fn fmix(mut k: u64) -> u64 {
        k ^= k >> 33;
        k = k.wrapping_mul(0xff51_afd7_ed55_8ccd);
        k ^= k >> 33;
        k = k.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        k ^= k >> 33;
        k
    }
    let (mut h1, mut h2) = (0u64, 0u64);
    let blocks = key.len() / 16;
    for i in 0..blocks {
        let mut k1 = u64::from_le_bytes(key[i * 16..i * 16 + 8].try_into().unwrap_or([0; 8]));
        let mut k2 = u64::from_le_bytes(key[i * 16 + 8..i * 16 + 16].try_into().unwrap_or([0; 8]));
        k1 = k1.wrapping_mul(C1).rotate_left(31).wrapping_mul(C2);
        h1 ^= k1;
        h1 = h1
            .rotate_left(27)
            .wrapping_add(h2)
            .wrapping_mul(5)
            .wrapping_add(0x52dc_e729);
        k2 = k2.wrapping_mul(C2).rotate_left(33).wrapping_mul(C1);
        h2 ^= k2;
        h2 = h2
            .rotate_left(31)
            .wrapping_add(h1)
            .wrapping_mul(5)
            .wrapping_add(0x3849_5ab5);
    }
    let tail = &key[blocks * 16..];
    let (mut k1, mut k2) = (0u64, 0u64);
    for (i, &b) in tail.iter().enumerate().rev() {
        if i >= 8 {
            k2 ^= u64::from(b) << ((i - 8) * 8);
        } else {
            k1 ^= u64::from(b) << (i * 8);
        }
    }
    if tail.len() > 8 {
        k2 = k2.wrapping_mul(C2).rotate_left(33).wrapping_mul(C1);
        h2 ^= k2;
    }
    if !tail.is_empty() {
        k1 = k1.wrapping_mul(C1).rotate_left(31).wrapping_mul(C2);
        h1 ^= k1;
    }
    h1 ^= key.len() as u64;
    h2 ^= key.len() as u64;
    h1 = h1.wrapping_add(h2);
    h2 = h2.wrapping_add(h1);
    h1 = fmix(h1);
    h2 = fmix(h2);
    h1 = h1.wrapping_add(h2);
    h1 as i64
}

/// `DeDuplicatingTokenFilter`'s `DuplicateTaggingFilter.loadAllTokens`
/// over one value's terms: which are kept (`getNumPriorUsesInASequence() <
/// 1`), the six-token ring buffer replayed as Java runs it.
pub fn dedup_keep(terms: &[Vec<u8>], spotter: &mut Spotter) -> Vec<bool> {
    const W: usize = 6;
    let mut keep = Vec::with_capacity(terms.len());
    let mut states: [Option<usize>; W] = [None; W];
    let mut max: [i16; W] = [0; W];
    let mut cursor = 0usize;
    let mut wrapped = false;
    let record =
        |states: &[Option<usize>; W], max: &mut [i16; W], at: usize, keep: &mut Vec<bool>| {
            let kept = max[at] < 1;
            max[at] = 0;
            if states[at].is_some() {
                keep.push(kept);
            }
        };
    for (t, term) in terms.iter().enumerate() {
        let byte = (murmur_h1(term) & 0xFF) as u8;
        let sightings = spotter.add_byte(byte);
        states[cursor] = Some(t);
        if sightings >= 1 {
            let mut pos = cursor as isize;
            for _ in 0..W {
                if pos < 0 {
                    pos = W as isize - 1;
                }
                let p = pos as usize;
                max[p] = max[p].max(sightings);
                pos -= 1;
            }
        }
        cursor += 1;
        if cursor >= W {
            cursor = 0;
            wrapped = true;
        }
        if wrapped && states[cursor].is_some() {
            record(&states, &mut max, cursor, &mut keep);
        }
    }
    let mut pos = if wrapped { (cursor + 1) % W } else { 0 };
    while pos != cursor {
        record(&states, &mut max, pos, &mut keep);
        pos = (pos + 1) % W;
    }
    keep
}

// ------------------------------------------------------------- the terms

/// What `significant_text` reads of each document.
#[derive(Clone)]
pub struct TextSource {
    /// `source_fields`, or the field itself.
    pub source_fields: Vec<String>,
    /// The field, as `analyzer.tokenStream(fieldType.name(), ...)` names it.
    pub field: String,
    pub analyzer: std::sync::Arc<CustomAnalyzer>,
    /// `filter_duplicate_text`.
    pub dedup: bool,
}

impl std::fmt::Debug for TextSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TextSource")
            .field("source_fields", &self.source_fields)
            .field("field", &self.field)
            .field("dedup", &self.dedup)
            .finish_non_exhaustive()
    }
}

impl TextSource {
    /// One document's terms, each once, in the order Java hands them to
    /// the bucket ords; `spotter` is the owning bucket's when
    /// de-duplicating.
    pub fn terms(
        &self,
        source: Option<&[u8]>,
        mut spotter: Option<&mut Spotter>,
        out: &mut Vec<Vec<u8>>,
    ) -> Result<(), Refused> {
        out.clear();
        let Some(bytes) = source else {
            return Ok(());
        };
        let map = parse_source(bytes)?;
        let mut seen: HashSet<Vec<u8>> = HashSet::new();
        let mut value_terms: Vec<Vec<u8>> = Vec::new();
        for field in &self.source_fields {
            for v in extract_raw_values(field, &map) {
                let text = display(v).ok_or_else(|| Refused("not a scalar".into()))?;
                value_terms.clear();
                let mut ts = self
                    .analyzer
                    .analyzer()
                    .token_stream(&self.field, &text)
                    .map_err(|e| Refused(format!("analysis: {e}")))?;
                let fail = |e: lucene_analysis::AnalysisError| Refused(format!("analysis: {e}"));
                ts.reset().map_err(fail)?;
                while ts.increment_token().map_err(fail)? {
                    // `scratch.copyChars(termAtt)`: the char term's UTF-8.
                    value_terms.push(ts.attributes().term().as_bytes().to_vec());
                }
                ts.end().map_err(fail)?;
                ts.close().map_err(fail)?;
                let keep = match spotter.as_deref_mut() {
                    Some(s) => dedup_keep(&value_terms, s),
                    None => vec![true; value_terms.len()],
                };
                for (term, kept) in value_terms.drain(..).zip(keep) {
                    if kept && seen.insert(term.clone()) {
                        out.push(term);
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_values_print_as_java_prints_the_source_map() {
        let s = parse_source(r#" {"a": {"b": ["x", {"c": "y"}, null, 1.0, -0, 12345678901234567890123, true]}, "d.e": "z", "u": "😀"} "#.as_bytes()).unwrap();
        let shown: Vec<String> = extract_raw_values("a.b", &s)
            .into_iter()
            .filter_map(display)
            .collect();
        assert_eq!(
            shown,
            vec!["x", "1.0", "0", "12345678901234567890123", "true"]
        );
        let shown: Vec<String> = extract_raw_values("a.b.c", &s)
            .into_iter()
            .filter_map(display)
            .collect();
        assert_eq!(
            shown,
            vec!["x", "y", "1.0", "0", "12345678901234567890123", "true"]
        );
        assert_eq!(
            extract_raw_values("d.e", &s)
                .into_iter()
                .filter_map(display)
                .collect::<Vec<_>>(),
            vec!["z"]
        );
        assert_eq!(display(extract_raw_values("u", &s)[0]).unwrap(), "😀");
        assert!(extract_raw_values("nope", &s).is_empty());
    }

    #[test]
    fn sources_java_would_read_differently_are_refused() {
        assert!(parse_source(br#"{"a": 1, "a": 2}"#).is_err());
        assert!(parse_source(br#"{"a": "\ud800"}"#).is_err());
        assert!(parse_source(b":)\n\x05").is_err());
        assert!(parse_source(br#"{"a": 01}"#).is_err());
        assert!(parse_source(br#"{"a": 1} x"#).is_err());
    }

    #[test]
    fn murmur_matches_the_reference_vectors() {
        // MurmurHash3_x64_128 of "" with seed 0 is zero; "hello" is the
        // canonical cc2d...  vector's first half.
        assert_eq!(murmur_h1(b""), 0);
        assert_eq!(murmur_h1(b"hello") as u64, 0xcbd8_a7b3_41bd_9b02);
    }

    const DEDUP_GOLDEN: &str = "\
w0 w2 w0 w1 w3 w1 w2 w2 w0 w3|w0 w2 w0 w1 w3 w1 w2 w2 w0 w3\n\
w0 w0 w8 w8 w0 w6 w7 w5 w8 w7 w2 w9 w11 w9 w10 w10 w7 w6 w10 w0 w2 w5|w0 w0 w8 w8 w0 w6 w7 w5 w8 w7 w2 w9 w11 w9 w10 w10 w7 w6 w10 w0 w2 w5\n\
w11 w0 w1 w1 w0 w11 w1 w11 w1 w11 w6 w7 w11 w1 w7 w1 w8|w11 w0 w1 w1 w0 w11 w1 w11 w1 w11 w6 w7 w11 w1 w7 w1 w8\n\
w0 w1 w3 w3 w0 w2 w2 w1 w3 w1 w0 w3 w2 w3 w0 w2 w1 w1 w0 w1 w1 w1 w2 w1 w2 w1 w3 w1 w2 w3 w0 w0 w0 w3 w3 w1 w3 w1|w0 w1 w3 w3 w0 w2 w2 w1 w3 w1 w0 w3 w2 w3 w0 w2 w1 w1 w0 w1 w1 w1 w2 w1 w2 w1 w3 w1 w2 w3 w0 w0 w0 w3 w3 w1 w3 w1\n\
w3 w3 w11 w3 w9 w0 w8 w4 w6 w4 w0 w1 w11 w7 w9 w2|w3 w3 w11 w3 w9 w0 w8 w4 w6 w4 w0 w1 w11 w7 w9 w2\n\
w2 w2 w4 w9 w7 w3 w6 w10 w7 w2 w3 w6 w0 w2 w7 w8 w0 w10 w3 w1 w10 w4 w8 w10 w4 w0 w8 w9 w2 w11 w1 w5 w11 w1 w11|w2 w2 w4 w9 w7 w3 w6 w10 w7 w2 w3 w6 w0 w2 w7 w8 w0 w10 w3 w1 w10 w4 w8 w10 w4 w0 w8 w9 w2 w11 w1 w5 w11 w1 w11\n\
w1 w1 w1 w2 w1 w1 w1 w3 w0 w1 w3 w2 w3 w3 w2 w3 w1 w2 w1 w3 w2 w0 w1 w3|w1 w1 w1 w2 w1 w1 w1 w3 w0 w1 w3 w2 w3 w3 w2 w3 w1 w2 w1 w3 w2 w0 w1 w3\n\
w8 w2 w9 w0 w10 w9 w0 w0 w1 w10 w2|w8 w2 w9 w0 w10 w9 w0 w0 w1 w10 w2\n\
w2 w7 w6 w5 w0 w9 w0 w3 w2 w9 w2 w10 w6 w11 w5 w11 w1 w6 w9 w8 w6 w8 w5 w1 w6 w7 w10 w8 w6|w2 w7 w6 w5 w0 w9 w0 w3 w2 w9 w2 w10 w6 w11 w5 w11 w1 w6 w9 w8 w6 w8 w5 w1 w6 w7 w10 w8 w6\n\
w2 w1 w3 w2 w2 w1 w2 w3 w2 w1 w0 w1 w2|w2 w1 w3 w2 w2 w1 w2 w3 w2 w1 w0 w1 w2\n\
w4 w1 w10 w7 w9 w1 w10 w3 w6 w3 w7 w6 w5 w6 w6 w6 w11 w10 w0 w9 w1 w11 w6 w7 w9 w9 w5 w9 w9 w3 w11 w3 w3|w4 w1 w10 w7 w9 w1 w10 w3 w6 w3 w7 w6 w5 w6 w6 w6 w11 w10 w0 w9 w1 w11 w6 w7 w9 w9 w5 w9 w9 w3 w11 w3 w3\n\
w2 w9 w0 w4 w4 w7 w5 w11 w3 w6 w2|w2 w9 w0 w4 w4 w7 w5 w11 w3 w6 w2\n\
w3 w0 w2 w1 w0 w0 w3 w2 w2 w1 w3 w3 w3 w2 w3 w1 w3 w2 w0 w0|w3 w0 w2 w1 w0 w0 w3 w2 w2 w1 w3 w3 w3 w2 w3 w1 w3 w2 w0 w0\n\
w2 w11 w5 w3 w7 w5 w1 w11 w1 w6 w6 w0 w9 w1 w1 w10 w8 w5 w2|w2 w11 w5 w3 w7 w5 w1 w11 w1 w6 w6 w0 w9 w1 w1 w10 w8 w5 w2\n\
w2 w11 w2 w7 w1 w9 w8 w3 w1 w6 w1 w8 w6 w5 w5 w6 w6 w8 w5 w5 w1 w8 w8 w1|w2 w11 w2 w7 w1 w9 w8 w3 w1 w6 w1 w8 w6 w5 w5 w6 w6 w8 w5 w5 w1 w8 w8 w1\n\
w2 w3 w2 w0 w3 w1 w3 w1 w1 w2 w1 w0 w2 w0 w1 w1 w3 w1 w2 w2 w0 w2 w1 w1 w2 w0 w2 w3 w2 w1 w2 w1 w3|w2 w3 w2 w0 w3 w1 w3 w1 w1 w2 w1 w0 w2 w0 w1 w2 w1 w1 w2 w0 w2 w3 w2 w1 w2 w1 w3\n\
w10 w9 w8 w10 w1 w2 w5 w4 w6 w8 w0 w1 w6 w11 w8 w1 w11 w9 w8 w10 w8|w10 w9 w8 w10 w1 w2 w5 w4 w6 w8 w0 w1 w6 w11 w8 w1 w11 w9 w8 w10 w8\n\
w7 w8 w9 w2 w0 w4 w1 w1 w7 w11 w10 w5 w4 w5 w6 w10 w4 w3 w1 w0 w4 w6 w0 w11 w1 w2 w1 w0 w5|w7 w8 w9 w2 w0 w4 w1 w1 w7 w11 w10 w5 w4 w5 w6 w10 w4 w3 w1 w0 w4 w6 w0 w11 w1 w2 w1 w0 w5\n\
w1 w2 w1 w3 w0 w1 w3 w0 w2 w1 w1 w1 w2 w1 w2|w1 w2 w1 w3 w0 w1 w3 w0 w2\n\
w6 w2 w3 w5 w6 w6 w8 w1 w9 w6 w6 w6 w4 w2 w5 w7 w4 w1 w10 w1|w6 w2 w3 w5 w6 w6 w8 w1 w9 w6 w6 w6 w4 w2 w5 w7 w4 w1 w10 w1\n\
w3|w3\n\
w0 w1 w0 w2 w1 w0 w2 w2 w0 w1 w3 w2 w2 w3 w1 w3 w2 w2 w2 w3 w0 w1 w2 w2 w1 w3 w2 w2 w1 w2 w1 w0 w2 w2 w1 w0 w3 w2 w0|w0 w1 w0 w2 w1 w0 w2 w2 w0 w1 w3 w2 w2 w3 w1 w3 w2 w2 w2 w3 w0 w1 w2 w1 w0 w2 w2 w1 w0 w3 w2 w0\n\
w11 w2 w6 w6 w3 w8 w10 w11 w1 w10 w10|w11 w2 w6 w6 w3 w8 w10 w11 w1 w10 w10\n\
w5|w5\n\
w3 w1 w3 w2 w1|w3 w1 w3 w2 w1\n\
w3 w3 w2 w11 w6|w3 w3 w2 w11 w6\n\
w6 w0 w6 w10 w10 w10 w5 w6 w3 w1 w3 w4 w1 w1 w2 w2 w6 w6|w6 w0 w6 w10 w10 w10 w5 w6 w3 w1 w3 w4 w1 w1 w2 w2 w6 w6\n\
w0 w2 w1 w3 w0 w2 w1|w0 w2 w1 w3 w0 w2 w1\n\
w2 w7 w7 w7 w0 w9 w7 w4 w1 w5 w0 w9|w2 w7 w7 w7 w0 w9 w7 w4 w1 w5 w0 w9\n\
w8 w0 w6 w11 w5 w8 w1 w2 w6 w11 w10 w0 w2 w6 w11 w2 w8|w8 w0 w6 w11 w5 w8 w1 w2 w6 w11 w10 w0 w2 w6 w11 w2 w8";

    /// `DeDuplicatingTokenFilter` over a `WhitespaceTokenizer`, one
    /// spotter for all lines and `startNewSequence()` after every second
    /// line, as OpenSearch 3.9 ran it: `input|kept` per line.
    #[test]
    fn dedup_matches_opensearch() {
        let mut s = Spotter::default();
        for (n, line) in DEDUP_GOLDEN.lines().enumerate() {
            let (input, kept) = line.split_once('|').unwrap();
            let t: Vec<Vec<u8>> = input
                .split_whitespace()
                .map(|w| w.as_bytes().to_vec())
                .collect();
            let keep = dedup_keep(&t, &mut s);
            let got: Vec<&str> = input
                .split_whitespace()
                .zip(&keep)
                .filter(|(_, &k)| k)
                .map(|(w, _)| w)
                .collect();
            assert_eq!(got.join(" "), kept, "line {n}");
            if n % 2 == 1 {
                s.start_new_sequence();
            }
        }
    }

    #[test]
    fn a_repeated_run_of_six_tokens_is_dropped_the_second_time() {
        let t: Vec<Vec<u8>> = "a b c d e f g a b c d e f g"
            .split(' ')
            .map(|s| s.as_bytes().to_vec())
            .collect();
        let mut s = Spotter::default();
        let keep = dedup_keep(&t, &mut s);
        assert_eq!(keep.len(), t.len());
        assert!(keep[..7].iter().all(|&k| k));
        assert!(keep[7..].iter().any(|&k| !k));
    }
}
