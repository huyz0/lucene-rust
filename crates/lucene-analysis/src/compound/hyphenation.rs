//! `org.apache.lucene.analysis.compound.hyphenation`: Liang's TeX hyphenation
//! (`HyphenationTree`, `Hyphenation`, `Hyphen`) and the FOP XML pattern
//! format (`PatternParser`).
//!
//! Differs:
//! - `HyphenationTree` keeps its patterns, character classes and exceptions
//!   in hash maps where Java uses `TernaryTree`s (with `ByteVector`/
//!   `CharVector` storage). `searchPatterns` visits every pattern that is a
//!   prefix of the word from a position, which is exactly what a lookup of
//!   each such prefix in a map visits, and merges values the same way; the
//!   packed nibble encoding of values is not needed. So a grammar Java
//!   cannot load is loaded here: building Java's `TernaryTree`s throws
//!   `ArrayIndexOutOfBoundsException` or `StackOverflowError` on files of
//!   20,000 and 80,000 random patterns (137 KB and 548 KB), which these
//!   maps load in tens of milliseconds.
//! - The XML is read by a minimal parser of this module's own (elements,
//!   attributes, text, comments, CDATA, the five predefined entities and
//!   character references; a `DOCTYPE` is skipped, as Java resolves only
//!   `hyphenation.dtd` and that DTD declares no entities). It refuses what
//!   is not well-formed in the ways a grammar file goes wrong (an unclosed
//!   or mismatched element, a second root, text outside the root, a bad
//!   entity or attribute); Java's SAX parser checks more (name characters,
//!   duplicate attributes, `<` in attribute values, ...), and a file only it
//!   refuses loads here. A malformed file is an `IllegalArgument` whose
//!   message starts [`MALFORMED`]; character data outside the root element
//!   carries Xerces' own message (`Content is not allowed in prolog.`), and
//!   the factory reports every such error as Java does, an `IOException`
//!   wrapping a `SAXParseException`.
//! - SAX hands `PatternParser.characters` the text in small chunks; here it
//!   is one chunk per text node, which `readToken` walks with an index
//!   instead of deleting what it consumed from the front (the result does
//!   not depend on the chunking).

use std::collections::HashMap;

use crate::char_array_set::WordHash;
use crate::java_character::{is_digit, is_whitespace};
use crate::AnalysisError;

/// `Hyphen`: a discretionary hyphen of an exception (`pre`, `no`, `post`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hyphen {
    /// `preBreak`.
    pub pre_break: Option<String>,
    /// `noBreak`.
    pub no_break: Option<String>,
    /// `postBreak`.
    pub post_break: Option<String>,
}

/// An exception's part: a letter run or a [`Hyphen`] (Java's
/// `ArrayList<Object>` items).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExceptionPart {
    /// A `String` item.
    Text(String),
    /// A `Hyphen` item.
    Hyphen(Hyphen),
}

/// `Hyphenation`: the hyphenation points of a word, `0` and its length
/// included.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hyphenation {
    points: Vec<usize>,
}

impl Hyphenation {
    /// `getHyphenationPoints()`.
    pub fn hyphenation_points(&self) -> &[usize] {
        &self.points
    }

    /// `length()`: the number of hyphenation points.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether there are no points (never, for a built `Hyphenation`).
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

/// `HyphenationTree`.
#[derive(Debug, Default, Clone)]
pub struct HyphenationTree {
    /// Pattern letters -> its inter-letter values (`TernaryTree` + `vspace`,
    /// which packs them two nibbles a byte, each `digit - '0' + 1`, a zero
    /// nibble ending them; kept as `getValues` unpacks them, so a lookup
    /// decodes nothing).
    patterns: HashMap<Vec<u16>, Vec<i8>, WordHash>,
    /// The longest pattern, bounding the prefixes `search_patterns` probes.
    max_pattern_len: usize,
    /// `classmap`: a letter -> its class's first char.
    classmap: HashMap<u16, u16, WordHash>,
    /// `stoplist`: exception word -> its parts.
    stoplist: HashMap<Vec<u16>, Vec<ExceptionPart>, WordHash>,
}

fn units(s: &str) -> Vec<u16> {
    s.encode_utf16().collect()
}

impl HyphenationTree {
    /// `new HyphenationTree()`, empty.
    pub fn new() -> Self {
        Self::default()
    }

    /// `HyphenationCompoundWordTokenFilter.getHyphenationTree(InputSource)`:
    /// a tree loaded from an FOP hyphenation XML document.
    pub fn from_xml(xml: &str) -> Result<Self, AnalysisError> {
        let mut tree = HyphenationTree::new();
        PatternParser::new(&mut tree).parse(xml)?;
        Ok(tree)
    }

    /// `PatternConsumer.addClass(String chargroup)`.
    pub fn add_class(&mut self, chargroup: &str) {
        let u = units(chargroup);
        if let Some(&equiv) = u.first() {
            for &c in &u {
                self.classmap.insert(c, equiv);
            }
        }
    }

    /// `PatternConsumer.addException(String word, ArrayList hyphenated)`.
    pub fn add_exception(&mut self, word: &str, hyphenated: Vec<ExceptionPart>) {
        self.stoplist.insert(units(word), hyphenated);
    }

    /// `PatternConsumer.addPattern(String pattern, String ivalue)`: `ivalue`
    /// is digits, one per inter-letter position.
    pub fn add_pattern(&mut self, pattern: &str, ivalue: &str) {
        let key = units(pattern);
        self.max_pattern_len = self.max_pattern_len.max(key.len());
        self.patterns
            .insert(key, Self::get_values(&Self::pack_values(ivalue)));
    }

    // Java: HyphenationTree.packValues
    fn pack_values(values: &str) -> Vec<u8> {
        let v: Vec<u16> = values.encode_utf16().collect();
        let n = v.len();
        let m = if n & 1 == 1 {
            (n >> 1) + 2
        } else {
            (n >> 1) + 1
        };
        let mut va = vec![0u8; m];
        for (i, &c) in v.iter().enumerate() {
            let nib = (c.wrapping_sub(u16::from(b'0')).wrapping_add(1) & 0x0f) as u8;
            if i & 1 == 1 {
                va[i >> 1] |= nib;
            } else {
                va[i >> 1] = nib << 4;
            }
        }
        va[m - 1] = 0;
        va
    }

    /// The packed nibbles, high then low, up to the first zero byte or zero
    /// low nibble (the loop both `getValues` and `unpackValues` run).
    fn nibbles(packed: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        for &v in packed {
            if v == 0 {
                break;
            }
            out.push(v >> 4);
            if v & 0x0f == 0 {
                break;
            }
            out.push(v & 0x0f);
        }
        out
    }

    // Java: HyphenationTree.getValues (`(byte) (nibble - 1)`)
    fn get_values(packed: &[u8]) -> Vec<i8> {
        Self::nibbles(packed)
            .into_iter()
            .map(|n| (u16::from(n).wrapping_sub(1)) as i8)
            .collect()
    }

    /// `findPattern(String)`: a pattern's values as digits, or `""`
    /// (`unpackValues`).
    pub fn find_pattern(&self, pat: &str) -> String {
        // `unpackValues`' `nibble - 1 + '0'` is `getValues`' value + '0'.
        self.patterns
            .get(&units(pat))
            .map(|values| {
                let u: Vec<u16> = values
                    .iter()
                    .map(|&v| (i16::from(v) as u16).wrapping_add(u16::from(b'0')))
                    .collect();
                String::from_utf16_lossy(&u)
            })
            .unwrap_or_default()
    }

    // Java: HyphenationTree.searchPatterns -- every pattern that is a prefix
    // of word[index..] (up to its null terminator) max-merges its values
    // into il from index on.
    fn search_patterns(&self, word: &[u16], index: usize, il: &mut [i8]) {
        let end = word[index..]
            .iter()
            .position(|&c| c == 0)
            .map_or(word.len(), |p| index + p);
        let longest = (end - index).min(self.max_pattern_len);
        for l in 1..=longest {
            if let Some(values) = self.patterns.get(&word[index..index + l]) {
                for (slot, &v) in il[index..].iter_mut().zip(values) {
                    if v > *slot {
                        *slot = v;
                    }
                }
            }
        }
    }

    /// `hyphenate(char[] w, int offset, int len, int remainCharCount, int
    /// pushCharCount)`: `None` when the word has a non-letter between
    /// letters, is too short, or has no point.
    pub fn hyphenate(&self, w: &[u16], remain: usize, push: usize) -> Option<Hyphenation> {
        let len = w.len();
        let mut word = vec![0u16; len + 3];
        let mut ignore_at_beginning = 0;
        let mut i_length = len;
        let mut end_of_letters = false;
        for i in 1..=len {
            match self.classmap.get(&w[i - 1]) {
                None => {
                    if i == 1 + ignore_at_beginning {
                        ignore_at_beginning += 1;
                    } else {
                        end_of_letters = true;
                    }
                    i_length -= 1;
                }
                Some(&nc) => {
                    if end_of_letters {
                        return None;
                    }
                    word[i - ignore_at_beginning] = nc;
                }
            }
        }
        let len = i_length;
        if len < remain + push {
            return None;
        }
        let mut result = Vec::new();
        if let Some(hw) = self.stoplist.get(&word[1..=len]) {
            let mut j = 0;
            for o in hw {
                if let ExceptionPart::Text(s) = o {
                    j += s.encode_utf16().count();
                    if j >= remain && j < len - push {
                        result.push(j + ignore_at_beginning);
                    }
                }
            }
        } else {
            word[0] = u16::from(b'.');
            word[len + 1] = u16::from(b'.');
            word[len + 2] = 0;
            let mut il = vec![0i8; len + 3];
            for i in 0..=len {
                self.search_patterns(&word, i, &mut il);
            }
            for i in 0..len {
                if il[i + 1] & 1 == 1 && i >= remain && i <= len - push {
                    result.push(i + ignore_at_beginning);
                }
            }
        }
        if result.is_empty() {
            return None;
        }
        let mut points = Vec::with_capacity(result.len() + 2);
        points.push(0);
        points.extend(result);
        points.push(len);
        Some(Hyphenation { points })
    }
}

// ---------------------------------------------------------------- the XML

/// A SAX event of [`parse_xml`].
#[derive(Debug, PartialEq, Eq)]
enum Event {
    Start(String, Vec<(String, String)>),
    End(String),
    Text(String),
}

fn malformed(what: &str) -> AnalysisError {
    AnalysisError::IllegalArgument(format!("{MALFORMED}{what}"))
}

/// How a malformed-XML error's message starts (the rest is Xerces'
/// message where the check is Xerces', the port's own otherwise).
pub const MALFORMED: &str = "malformed hyphenation XML: ";

/// XML's whitespace: space, tab, CR, LF.
fn is_xml_space(s: &str) -> bool {
    s.chars().all(|c| matches!(c, ' ' | '\t' | '\r' | '\n'))
}

/// Xerces' message for character data outside the root element.
fn outside_root(root_closed: bool) -> AnalysisError {
    malformed(if root_closed {
        "Content is not allowed in trailing section."
    } else {
        "Content is not allowed in prolog."
    })
}

/// Decodes `&...;` references.
fn decode_entities(s: &str) -> Result<String, AnalysisError> {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find('&') {
        out.push_str(&rest[..i]);
        let tail = &rest[i + 1..];
        let end = tail
            .find(';')
            .ok_or_else(|| malformed("unterminated entity"))?;
        let name = &tail[..end];
        let ch = match name {
            "lt" => '<',
            "gt" => '>',
            "amp" => '&',
            "quot" => '"',
            "apos" => '\'',
            _ => {
                let cp = if let Some(h) = name.strip_prefix("#x") {
                    u32::from_str_radix(h, 16).ok()
                } else if let Some(d) = name.strip_prefix('#') {
                    d.parse().ok()
                } else {
                    None
                };
                cp.and_then(char::from_u32)
                    .ok_or_else(|| malformed(&format!("unknown entity &{name};")))?
            }
        };
        out.push(ch);
        rest = &tail[end + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

/// The local part of a (possibly prefixed) name (SAX's `localName` with
/// namespace awareness).
fn local(name: &str) -> String {
    name.rsplit(':').next().unwrap_or(name).to_string()
}

/// A minimal non-validating XML reader: the events of `xml`.
fn parse_xml(xml: &str) -> Result<Vec<Event>, AnalysisError> {
    let mut events = Vec::new();
    let mut rest = xml.strip_prefix('\u{FEFF}').unwrap_or(xml);
    // The open elements, innermost last; whether the root has closed.
    let mut open: Vec<&str> = Vec::new();
    let mut root_closed = false;
    while !rest.is_empty() {
        let Some(lt) = rest.find('<') else {
            if !open.is_empty() {
                return Err(malformed("text outside the root element"));
            }
            if !is_xml_space(rest) {
                return Err(outside_root(root_closed));
            }
            break;
        };
        if lt > 0 {
            let text = decode_entities(&rest[..lt])?;
            if !open.is_empty() {
                events.push(Event::Text(text));
            } else if !is_xml_space(&text) {
                return Err(outside_root(root_closed));
            }
        }
        rest = &rest[lt..];
        if let Some(r) = rest.strip_prefix("<!--") {
            let end = r
                .find("-->")
                .ok_or_else(|| malformed("unterminated comment"))?;
            rest = &r[end + 3..];
        } else if let Some(r) = rest.strip_prefix("<![CDATA[") {
            let end = r
                .find("]]>")
                .ok_or_else(|| malformed("unterminated CDATA"))?;
            events.push(Event::Text(r[..end].to_string()));
            rest = &r[end + 3..];
        } else if let Some(r) = rest.strip_prefix("<?") {
            let end = r
                .find("?>")
                .ok_or_else(|| malformed("unterminated declaration"))?;
            rest = &r[end + 2..];
        } else if let Some(r) = rest.strip_prefix("<!DOCTYPE") {
            // Skip to the matching '>', past an internal subset `[...]`.
            let mut in_subset = false;
            let end = r
                .char_indices()
                .find(|&(_, c)| {
                    if c == '[' {
                        in_subset = true;
                    } else if c == ']' {
                        in_subset = false;
                    }
                    c == '>' && !in_subset
                })
                .map(|(i, _)| i)
                .ok_or_else(|| malformed("unterminated DOCTYPE"))?;
            rest = &r[end + 1..];
        } else if let Some(r) = rest.strip_prefix("</") {
            let end = r
                .find('>')
                .ok_or_else(|| malformed("unterminated end tag"))?;
            let name = r[..end].trim_end();
            if open.pop() != Some(name) {
                return Err(malformed("end tag not matching the open element"));
            }
            root_closed = open.is_empty();
            events.push(Event::End(local(name)));
            rest = &r[end + 1..];
        } else {
            // A start tag; quoted '>' may appear in attribute values.
            let r = &rest[1..];
            let mut quote = None;
            let end = r
                .char_indices()
                .find(|&(_, c)| match quote {
                    Some(q) => {
                        if c == q {
                            quote = None;
                        }
                        false
                    }
                    None => {
                        if c == '"' || c == '\'' {
                            quote = Some(c);
                        }
                        c == '>'
                    }
                })
                .map(|(i, _)| i)
                .ok_or_else(|| malformed("unterminated tag"))?;
            let mut inner = &r[..end];
            let empty = inner.ends_with('/');
            if empty {
                inner = &inner[..inner.len() - 1];
            }
            let name_end = inner
                .find(|c: char| c.is_ascii_whitespace())
                .unwrap_or(inner.len());
            let name = &inner[..name_end];
            if name.is_empty() {
                return Err(malformed("empty tag name"));
            }
            if root_closed {
                return Err(malformed("an element after the root element"));
            }
            let mut attrs = Vec::new();
            let mut a = inner[name_end..].trim_start();
            while !a.is_empty() {
                let eq = a
                    .find('=')
                    .ok_or_else(|| malformed("attribute without value"))?;
                let key = a[..eq].trim();
                let v = a[eq + 1..].trim_start();
                let q = v.chars().next().filter(|&c| c == '"' || c == '\'');
                let q = q.ok_or_else(|| malformed("unquoted attribute"))?;
                let close = v[1..]
                    .find(q)
                    .ok_or_else(|| malformed("unterminated attribute"))?;
                attrs.push((local(key), decode_entities(&v[1..1 + close])?));
                a = v[close + 2..].trim_start();
            }
            events.push(Event::Start(local(name), attrs));
            if empty {
                events.push(Event::End(local(name)));
                root_closed = open.is_empty();
            } else {
                open.push(name);
            }
            rest = &r[end + 1..];
        }
    }
    if !open.is_empty() || !root_closed {
        return Err(malformed("unclosed element"));
    }
    Ok(events)
}

const ELEM_CLASSES: u8 = 1;
const ELEM_EXCEPTIONS: u8 = 2;
const ELEM_PATTERNS: u8 = 3;
const ELEM_HYPHEN: u8 = 4;

/// `PatternParser`: feeds an FOP hyphenation document to a
/// [`HyphenationTree`] (`PatternConsumer`).
pub struct PatternParser<'a> {
    consumer: &'a mut HyphenationTree,
    curr_element: u8,
    token: Vec<u16>,
    exception: Vec<ExceptionPart>,
    hyphen_char: u16,
}

fn is_ws(c: u16) -> bool {
    is_whitespace(u32::from(c))
}

impl<'a> PatternParser<'a> {
    /// `new PatternParser(PatternConsumer)`.
    pub fn new(consumer: &'a mut HyphenationTree) -> Self {
        PatternParser {
            consumer,
            curr_element: 0,
            token: Vec::new(),
            exception: Vec::new(),
            hyphen_char: u16::from(b'-'),
        }
    }

    /// `parse(InputSource)`.
    pub fn parse(&mut self, xml: &str) -> Result<(), AnalysisError> {
        for e in parse_xml(xml)? {
            match e {
                Event::Start(name, attrs) => self.start_element(&name, &attrs),
                Event::End(_) => self.end_element(),
                Event::Text(t) => self.characters(&t),
            }
        }
        Ok(())
    }

    // Java: PatternParser.readToken
    //
    // Java deletes what it consumed from the front of `chars`; this advances
    // `pos` past it instead (Java's chunks are SAX's small buffers, this
    // parser's one chunk is the whole text).
    fn read_token(&mut self, chars: &[u16], pos: &mut usize) -> Option<String> {
        let rest = &chars[*pos..];
        let lead = rest.iter().take_while(|&&c| is_ws(c)).count();
        if lead > 0 {
            *pos += lead;
            if !self.token.is_empty() {
                return Some(String::from_utf16_lossy(&std::mem::take(&mut self.token)));
            }
        }
        let rest = &chars[*pos..];
        let word_end = rest.iter().position(|&c| is_ws(c));
        let i = word_end.unwrap_or(rest.len());
        self.token.extend_from_slice(&rest[..i]);
        *pos += i;
        if word_end.is_some() {
            return Some(String::from_utf16_lossy(&std::mem::take(&mut self.token)));
        }
        // Java appends the (now empty) rest again.
        None
    }

    /// `getPattern(String)`: the word without its digits.
    fn get_pattern(word: &str) -> String {
        word.chars().filter(|&c| !is_digit(c as u32)).collect()
    }

    /// `getInterletterValues(String)`: a digit per inter-letter position.
    fn get_interletter_values(pat: &str) -> String {
        let word: Vec<u16> = pat.encode_utf16().chain([u16::from(b'a')]).collect();
        let mut il = String::new();
        let mut i = 0;
        while i < word.len() {
            let c = word[i];
            if is_digit(u32::from(c)) {
                il.push(char::from_u32(u32::from(c)).unwrap_or('0'));
                i += 1;
            } else {
                il.push('0');
            }
            i += 1;
        }
        il
    }

    // Java: PatternParser.normalizeException
    fn normalize_exception(&self, ex: Vec<ExceptionPart>) -> Vec<ExceptionPart> {
        let mut res = Vec::new();
        for item in ex {
            match item {
                ExceptionPart::Text(s) => {
                    let mut buf = Vec::new();
                    for c in s.encode_utf16() {
                        if c != self.hyphen_char {
                            buf.push(c);
                        } else {
                            res.push(ExceptionPart::Text(String::from_utf16_lossy(&buf)));
                            buf.clear();
                            res.push(ExceptionPart::Hyphen(Hyphen {
                                pre_break: Some(String::from_utf16_lossy(&[self.hyphen_char])),
                                no_break: None,
                                post_break: None,
                            }));
                        }
                    }
                    if !buf.is_empty() {
                        res.push(ExceptionPart::Text(String::from_utf16_lossy(&buf)));
                    }
                }
                h => res.push(h),
            }
        }
        res
    }

    // Java: PatternParser.getExceptionWord
    fn exception_word(ex: &[ExceptionPart]) -> String {
        ex.iter()
            .map(|p| match p {
                ExceptionPart::Text(s) => s.as_str(),
                ExceptionPart::Hyphen(h) => h.no_break.as_deref().unwrap_or(""),
            })
            .collect()
    }

    fn add_exception_word(&mut self, word: String) {
        let mut ex = std::mem::take(&mut self.exception);
        ex.push(ExceptionPart::Text(word));
        let ex = self.normalize_exception(ex);
        self.consumer
            .add_exception(&Self::exception_word(&ex), ex.clone());
        self.exception = ex;
    }

    // Java: PatternParser.startElement
    fn start_element(&mut self, local: &str, attrs: &[(String, String)]) {
        let attr = |n: &str| attrs.iter().find(|(k, _)| k == n).map(|(_, v)| v.clone());
        match local {
            "hyphen-char" => {
                if let Some(h) = attr("value") {
                    let u = units(&h);
                    if u.len() == 1 {
                        self.hyphen_char = u[0];
                    }
                }
            }
            "classes" => self.curr_element = ELEM_CLASSES,
            "patterns" => self.curr_element = ELEM_PATTERNS,
            "exceptions" => {
                self.curr_element = ELEM_EXCEPTIONS;
                self.exception = Vec::new();
            }
            "hyphen" => {
                if !self.token.is_empty() {
                    self.exception
                        .push(ExceptionPart::Text(String::from_utf16_lossy(&self.token)));
                }
                self.exception.push(ExceptionPart::Hyphen(Hyphen {
                    pre_break: attr("pre"),
                    no_break: attr("no"),
                    post_break: attr("post"),
                }));
                self.curr_element = ELEM_HYPHEN;
            }
            _ => {}
        }
        self.token.clear();
    }

    // Java: PatternParser.endElement
    fn end_element(&mut self) {
        if !self.token.is_empty() {
            let word = String::from_utf16_lossy(&self.token);
            match self.curr_element {
                ELEM_CLASSES => self.consumer.add_class(&word),
                ELEM_EXCEPTIONS => self.add_exception_word(word),
                ELEM_PATTERNS => self.consumer.add_pattern(
                    &Self::get_pattern(&word),
                    &Self::get_interletter_values(&word),
                ),
                _ => {}
            }
            if self.curr_element != ELEM_HYPHEN {
                self.token.clear();
            }
        }
        self.curr_element = if self.curr_element == ELEM_HYPHEN {
            ELEM_EXCEPTIONS
        } else {
            0
        };
    }

    // Java: PatternParser.characters
    fn characters(&mut self, text: &str) {
        let chars = units(text);
        let mut pos = 0;
        while let Some(word) = self.read_token(&chars, &mut pos) {
            match self.curr_element {
                ELEM_CLASSES => self.consumer.add_class(&word),
                ELEM_EXCEPTIONS => {
                    self.add_exception_word(word);
                    self.exception.clear();
                }
                ELEM_PATTERNS => self.consumer.add_pattern(
                    &Self::get_pattern(&word),
                    &Self::get_interletter_values(&word),
                ),
                _ => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const XML: &str = r#"<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE hyphenation-info SYSTEM "hyphenation.dtd" [ <!ENTITY x "y"> ]>
<!-- test -->
<hyphenation-info>
<hyphen-char value="="/>
<hyphen-min before="2" after="2"/>
<classes>
aA bB dD eE fF hH iI kK lL nN oO rR sS tT uU
</classes>
<exceptions>
ta=ble <hyphen pre="k" no="c" post="k"/>
</exceptions>
<patterns>
1ba 1fe a1b b1e 4b1l .ab4 &#x6c;1d <![CDATA[n1d]]>
</patterns>
</hyphenation-info>
"#;

    fn u(s: &str) -> Vec<u16> {
        s.encode_utf16().collect()
    }

    #[test]
    fn parses_patterns_classes_and_exceptions() {
        let t = HyphenationTree::from_xml(XML).unwrap();
        assert_eq!(t.find_pattern("ba"), "100");
        assert_eq!(t.find_pattern("bl"), "410");
        assert_eq!(t.find_pattern(".ab"), "0004");
        assert_eq!(t.find_pattern("ld"), "010");
        assert_eq!(t.find_pattern("nd"), "010");
        assert_eq!(t.find_pattern("zz"), "");
        assert!(t.stoplist.contains_key(&u("table")));
        assert_eq!(t.classmap.get(&u16::from(b'A')), Some(&u16::from(b'a')));
    }

    #[test]
    fn values_pack_like_java() {
        assert_eq!(HyphenationTree::pack_values("0102"), vec![0x12, 0x13, 0]);
        assert_eq!(HyphenationTree::pack_values("010"), vec![0x12, 0x10, 0]);
        assert_eq!(HyphenationTree::get_values(&[0x12, 0x10, 0]), vec![0, 1, 0]);
        // A digit whose nibble wraps to zero ends the values early, as in Java.
        assert_eq!(
            HyphenationTree::get_values(&HyphenationTree::pack_values("0?1")),
            vec![0]
        );
    }

    #[test]
    fn hyphenates_by_patterns_and_exceptions() {
        let t = HyphenationTree::from_xml(XML).unwrap();
        let pts = |w: &str| {
            t.hyphenate(&u(w), 1, 1)
                .map(|h| h.hyphenation_points().to_vec())
        };
        assert_eq!(pts("table"), Some(vec![0, 2, 5]));
        assert_eq!(pts("rabe"), Some(vec![0, 2, 3, 4]));
        assert_eq!(pts("xrabe"), Some(vec![0, 3, 4, 4]));
        assert_eq!(pts("ra-be"), None);
        assert_eq!(pts("a"), None);
        assert_eq!(pts("tttt"), None);
        assert!(t
            .hyphenate(&u("rabe"), 1, 1)
            .is_some_and(|h| h.len() == 4 && !h.is_empty()));
    }

    #[test]
    fn reads_one_large_text_in_linear_time() {
        // One text node of 900k units: SAX hands Java's `characters` small
        // chunks, but this parser hands it the whole text, so a token must
        // not cost a copy of the rest of it (that was minutes, not
        // milliseconds).
        let xml = format!(
            "<hyphenation-info><patterns>{}a1b</patterns></hyphenation-info>",
            "b1c ".repeat(225_000)
        );
        let start = std::time::Instant::now();
        let t = HyphenationTree::from_xml(&xml).unwrap();
        let took = start.elapsed();
        assert_eq!(t.find_pattern("bc"), "010");
        assert_eq!(t.find_pattern("ab"), "010");
        assert!(took < std::time::Duration::from_secs(30), "{took:?}");
    }

    #[test]
    fn rejects_malformed_xml() {
        for bad in [
            "<a>",
            "<a></b></a></c>",
            "<a b=c></a>",
            "<a b=\"c></a>",
            "<a>&bogus;</a>",
            "<!-- x",
            "<a><![CDATA[x</a>",
            "<?x",
            "<!DOCTYPE x",
            "</a",
            "<a",
            "< ></>",
            "<a b></a>",
            "<a>&x</a>",
            "x",
            // What SAX refuses as not well-formed: a mismatched end tag, a
            // second root, no root at all.
            "<a></b>",
            "<a><b></a></b>",
            "<a></a><b></b>",
            "<a/><b/>",
            "",
            " <!-- only a comment --> ",
        ] {
            assert!(HyphenationTree::from_xml(bad).is_err(), "{bad}");
        }
        assert!(HyphenationTree::from_xml("<a x:b='1'>&lt;&gt;&amp;&quot;&apos;&#65;</a>").is_ok());
        // Character data outside the root: Xerces' two messages.
        let msg = |x: &str| HyphenationTree::from_xml(x).err().unwrap().to_string();
        for (x, want) in [
            ("x<a/>", "Content is not allowed in prolog."),
            ("x", "Content is not allowed in prolog."),
            ("<a/>x", "Content is not allowed in trailing section."),
            (
                "<a/>x<!-- c -->",
                "Content is not allowed in trailing section.",
            ),
        ] {
            assert!(
                msg(x).ends_with(&format!("{MALFORMED}{want}")),
                "{x}: {}",
                msg(x)
            );
        }
        assert!(HyphenationTree::from_xml(" \r\n\t<a/> \n").is_ok());
    }
}
