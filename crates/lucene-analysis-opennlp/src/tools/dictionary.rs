//! OpenNLP's dictionaries: `Dictionary` (the abbreviations of the sentence
//! detector and tokenizer, the POS tagger's n-gram dictionary) and
//! `POSDictionary` (the tag dictionary), both read by
//! `DictionaryEntryPersistor` from XML, and `DictionaryLemmatizer`'s
//! tab-separated word list.
//!
//! Differs: a case-insensitive `Dictionary` is a `HashSet` of
//! `StringList`s hashed on their lower-cased text and compared with
//! `String.compareToIgnoreCase`; here two entries are the same when their
//! lower-cased tokens are -- which differs only for strings equal ignoring
//! case whose `Character.toLowerCase` forms differ (`ϑ`/`θ`), which Java's
//! hash set does not find either unless their hashes collide. A
//! case-insensitive tag dictionary holding one word in two cases keeps the
//! later entry; Java keeps whichever its `HashMap` iterates last.

use std::collections::{HashMap, HashSet};

use lucene_analysis::factory::{FactoryError, JavaException};

use super::text::{self, JString};
use super::xml::{self, Element, Node};

/// One dictionary entry as the persistor hands it on: its tokens and its
/// attributes.
struct Entry {
    tokens: Vec<JString>,
    attributes: Vec<(String, JString)>,
}

fn invalid_format(message: impl Into<String>) -> FactoryError {
    FactoryError::new(
        JavaException::Io,
        format!("InvalidFormatException: {}", message.into()),
    )
}

/// `DictionaryEntryPersistor.create`: the entries of the XML dictionary,
/// each handed to `insert`, and the dictionary's `case_sensitive` flag
/// (`true` unless the root says otherwise).
fn read_entries(
    bytes: &[u8],
    insert: &mut dyn FnMut(Entry) -> Result<(), FactoryError>,
) -> Result<bool, FactoryError> {
    let root = xml::parse(bytes).map_err(|e| match e.kind {
        JavaException::Io => invalid_format(format!(
            "The profile data stream has an invalid format! {}",
            e.message
        )),
        _ => e,
    })?;
    let mut h = Handler {
        case_sensitive: true,
        inside_token: false,
        token: Vec::new(),
        tokens: Vec::new(),
        attributes: None,
        insert,
    };
    h.walk(&root)?;
    Ok(h.case_sensitive)
}

/// `DictionaryContenthandler`, driven by a walk of the element tree in
/// document order (SAX's events).
struct Handler<'a> {
    case_sensitive: bool,
    inside_token: bool,
    token: JString,
    tokens: Vec<JString>,
    attributes: Option<Vec<(String, JString)>>,
    insert: &'a mut dyn FnMut(Entry) -> Result<(), FactoryError>,
}

impl Handler<'_> {
    fn walk(&mut self, el: &Element) -> Result<(), FactoryError> {
        let local = xml::local_name(&el.name);
        let attributes = || {
            el.attributes
                .iter()
                .filter(|(n, _)| n != "xmlns" && !n.starts_with("xmlns:"))
                .map(|(n, v)| (xml::local_name(n).to_string(), v.clone()))
                .collect::<Vec<_>>()
        };
        match local {
            "dictionary" => {
                if let Some((_, v)) = attributes().iter().find(|(n, _)| n == "case_sensitive") {
                    self.case_sensitive = text::parse_boolean(&text::to_string(v));
                }
                self.attributes = None;
            }
            "entry" => self.attributes = Some(attributes()),
            "token" => self.inside_token = true,
            _ => {}
        }
        for child in &el.children {
            match child {
                Node::Element(e) => self.walk(e)?,
                Node::Text(t) => {
                    if self.inside_token {
                        self.token.extend_from_slice(t);
                    }
                }
            }
        }
        match local {
            "token" => {
                self.tokens.push(text::trim(&self.token).to_vec());
                self.token.clear();
                self.inside_token = false;
            }
            "entry" => {
                if self.tokens.is_empty() {
                    // `new StringList(tokens)`
                    return Err(FactoryError::illegal_argument("tokens must not be empty"));
                }
                let entry = Entry {
                    tokens: std::mem::take(&mut self.tokens),
                    attributes: self.attributes.take().unwrap_or_default(),
                };
                (self.insert)(entry)?;
            }
            _ => {}
        }
        Ok(())
    }
}

/// `opennlp.tools.dictionary.Dictionary`.
#[derive(Debug, Clone, Default)]
pub struct Dictionary {
    case_sensitive: bool,
    /// The entries as first inserted, in insertion order.
    entries: Vec<Vec<JString>>,
    keys: HashSet<Vec<JString>>,
}

impl Dictionary {
    /// `new Dictionary(InputStream)`.
    pub fn read(bytes: &[u8]) -> Result<Dictionary, FactoryError> {
        let mut entries = Vec::new();
        let case_sensitive = read_entries(bytes, &mut |e| {
            entries.push(e.tokens);
            Ok(())
        })?;
        let mut d = Dictionary {
            case_sensitive,
            ..Dictionary::default()
        };
        for e in entries {
            d.put(e);
        }
        Ok(d)
    }

    fn key(&self, tokens: &[JString]) -> Vec<JString> {
        if self.case_sensitive {
            tokens.to_vec()
        } else {
            tokens.iter().map(|t| text::to_lower_case(t)).collect()
        }
    }

    /// `put(StringList)`: a `HashSet` add (an equal entry is kept as first
    /// added).
    pub fn put(&mut self, tokens: Vec<JString>) {
        if self.keys.insert(self.key(&tokens)) {
            self.entries.push(tokens);
        }
    }

    /// `contains(StringList)`.
    pub fn contains(&self, tokens: &[JString]) -> bool {
        self.keys.contains(&self.key(tokens))
    }

    /// `asStringSet().contains(str)`: the one-token entry `s`.
    pub fn contains_string(&self, s: &[u16]) -> bool {
        self.contains(&[s.to_vec()])
    }

    /// Every entry's first token (`asStringSet()`'s iteration; the
    /// abbreviation scan of `SentenceDetectorME.isAcceptableBreak`).
    pub fn first_tokens(&self) -> impl Iterator<Item = &JString> {
        self.entries.iter().map(|e| &e[0])
    }
}

/// `String.split(String)` on a one-character separator: Java's fast path,
/// trailing empty strings dropped.
pub fn java_split(s: &[u16], sep: u16) -> Vec<JString> {
    let mut parts: Vec<JString> = s.split(|&c| c == sep).map(<[u16]>::to_vec).collect();
    // an input without the separator is itself, even when empty
    if parts.len() > 1 {
        while parts.last().is_some_and(Vec::is_empty) {
            parts.pop();
        }
    }
    parts
}

/// `opennlp.tools.postag.POSDictionary`: a word's admissible tags.
#[derive(Debug, Clone, Default)]
pub struct PosDictionary {
    case_sensitive: bool,
    tags: HashMap<JString, Vec<JString>>,
}

impl PosDictionary {
    /// `POSDictionary.create(InputStream)`.
    pub fn read(bytes: &[u8]) -> Result<PosDictionary, FactoryError> {
        let mut words = Vec::new();
        let case_sensitive = read_entries(bytes, &mut |e| {
            let tags = e
                .attributes
                .iter()
                .find(|(n, _)| n == "tags")
                .map(|(_, v)| v.clone())
                .ok_or_else(|| {
                    FactoryError::new(
                        JavaException::NullPointer,
                        "an entry of the tag dictionary has no tags attribute",
                    )
                })?;
            if e.tokens.len() != 1 {
                return Err(invalid_format(format!(
                    "The profile data stream has an invalid format! Each entry must have exactly one token! [{}]",
                    e.tokens.iter().map(|t| text::to_string(t)).collect::<Vec<_>>().join(",")
                )));
            }
            let word = e.tokens.into_iter().next().unwrap_or_default();
            words.push((word, java_split(&tags, 0x20)));
            Ok(())
        })?;
        let mut d = PosDictionary {
            case_sensitive,
            tags: HashMap::new(),
        };
        for (word, tags) in words {
            let word = if case_sensitive {
                word
            } else {
                text::to_lower_case(&word)
            };
            d.tags.insert(word, tags);
        }
        Ok(d)
    }

    /// `getTags(word)`.
    pub fn tags(&self, word: &[u16]) -> Option<&[JString]> {
        if self.case_sensitive {
            self.tags.get(word)
        } else {
            self.tags.get(&text::to_lower_case(word))
        }
        .map(Vec::as_slice)
    }
}

/// `opennlp.tools.lemmatizer.DictionaryLemmatizer`: `word<TAB>tag<TAB>lemma`
/// lines (alternative lemmas separated by `#`), looked up by the
/// lower-cased word and its tag.
#[derive(Debug, Clone, Default)]
pub struct DictionaryLemmatizer {
    map: HashMap<(JString, JString), Vec<JString>>,
}

impl DictionaryLemmatizer {
    /// `new DictionaryLemmatizer(InputStream)` (UTF-8, malformed bytes
    /// replaced, as `InputStreamReader`). A line with fewer than three
    /// fields is Java's `ArrayIndexOutOfBoundsException`.
    pub fn read(bytes: &[u8]) -> Result<DictionaryLemmatizer, FactoryError> {
        let content = String::from_utf8_lossy(bytes);
        let mut map = HashMap::new();
        for line in java_lines(&content) {
            let elems = java_split(&text::units(line), 0x09);
            let Some(lemmas) = elems.get(2) else {
                return Err(FactoryError::new(
                    JavaException::ArrayIndexOutOfBounds,
                    format!("Index 2 out of bounds for length {}", elems.len()),
                ));
            };
            let lemmas = java_split(lemmas, 0x23);
            map.insert((elems[0].clone(), elems[1].clone()), lemmas);
        }
        Ok(DictionaryLemmatizer { map })
    }

    /// `lemmatize(tokens, postags)`: each word's first lemma, `"O"` for a
    /// word the dictionary lacks.
    pub fn lemmatize(&self, tokens: &[JString], tags: &[JString]) -> Vec<JString> {
        tokens
            .iter()
            .zip(tags)
            .map(|(word, tag)| {
                let key = (text::string_to_lower_case(word), tag.clone());
                match self.map.get(&key).and_then(|v| v.first()) {
                    Some(l) => l.clone(),
                    None => text::units("O"),
                }
            })
            .collect()
    }
}

/// `BufferedReader.readLine()`'s lines: ended by `\n`, `\r` or `\r\n`, the
/// last one only if non-empty.
// ARITH: offsets within the string, from `find`.
#[allow(clippy::arithmetic_side_effects)]
fn java_lines(s: &str) -> Vec<&str> {
    let mut lines = Vec::new();
    let mut rest = s;
    while !rest.is_empty() {
        match rest.find(['\n', '\r']) {
            Some(i) => {
                lines.push(&rest[..i]);
                let skip = if rest[i..].starts_with("\r\n") { 2 } else { 1 };
                rest = &rest[i + skip..];
            }
            None => {
                lines.push(rest);
                rest = "";
            }
        }
    }
    lines
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arithmetic_side_effects)] // test code
    use super::*;

    fn u(s: &str) -> JString {
        text::units(s)
    }

    #[test]
    fn dictionaries() {
        let d = Dictionary::read(
            br#"<?xml version="1.0"?><dictionary case_sensitive="false"><entry><token> Dr. </token></entry><entry><token>dr.</token></entry><entry a="b"><token>e.g.</token><token>x</token></entry><other/></dictionary>"#,
        )
        .unwrap();
        assert!(d.contains_string(&u("DR.")) && !d.contains_string(&u("e.G.")));
        assert!(d.contains(&[u("E.G."), u("X")]));
        assert_eq!(
            d.first_tokens().cloned().collect::<Vec<_>>(),
            vec![u("Dr."), u("e.g.")]
        );
        let cs = Dictionary::read(b"<dictionary><entry><token>Dr.</token></entry></dictionary>")
            .unwrap();
        assert!(cs.contains_string(&u("Dr.")) && !cs.contains_string(&u("dr.")));
        let ns = Dictionary::read(
            b"<d:dictionary xmlns:d=\"x\" d:case_sensitive=\"false\"><d:entry><d:token>A<b>c</b></d:token></d:entry></d:dictionary>",
        )
        .unwrap();
        assert!(ns.contains_string(&u("ac")));
        let e = Dictionary::read(b"<dictionary><entry/></dictionary>").unwrap_err();
        assert_eq!(e.message, "tokens must not be empty");
        let e = Dictionary::read(b"<dictionary>").unwrap_err();
        assert!(
            e.message.starts_with("InvalidFormatException"),
            "{}",
            e.message
        );
        let e = Dictionary::read(b"<!DOCTYPE x><dictionary/>").unwrap_err();
        assert_eq!(e.kind, JavaException::UnsupportedOperation);
    }

    #[test]
    fn tag_dictionaries() {
        let d = PosDictionary::read(
            br#"<dictionary case_sensitive="false"><entry tags="DT"><token>The</token></entry><entry tags="VBD VBP  VB  "><token>read</token></entry></dictionary>"#,
        )
        .unwrap();
        assert_eq!(d.tags(&u("THE")).unwrap(), &[u("DT")]);
        assert_eq!(
            d.tags(&u("read")).unwrap(),
            &[u("VBD"), u("VBP"), u(""), u("VB")]
        );
        assert!(d.tags(&u("x")).is_none());
        let cs = PosDictionary::read(
            br#"<dictionary><entry tags="NN"><token>Dog</token></entry></dictionary>"#,
        )
        .unwrap();
        assert!(cs.tags(&u("dog")).is_none() && cs.tags(&u("Dog")).is_some());
        let e = PosDictionary::read(b"<dictionary><entry><token>a</token></entry></dictionary>")
            .unwrap_err();
        assert_eq!(e.kind, JavaException::NullPointer);
        let e = PosDictionary::read(
            br#"<dictionary><entry tags="A"><token>a</token><token>b</token></entry></dictionary>"#,
        )
        .unwrap_err();
        assert!(
            e.message.contains("exactly one token! [a,b]"),
            "{}",
            e.message
        );
    }

    #[test]
    fn lemma_dictionaries() {
        let d = DictionaryLemmatizer::read(
            b"dogs\tNNS\tdog\nsaw\tNN\tsaw#see\r\nx\tY\t\tz\rlast\tT\tl",
        )
        .unwrap();
        assert_eq!(
            d.lemmatize(
                &[u("Dogs"), u("saw"), u("cat"), u("x"), u("LAST")],
                &[u("NNS"), u("NN"), u("NN"), u("Y"), u("T")]
            ),
            vec![u("dog"), u("saw"), u("O"), u(""), u("l")]
        );
        for bad in [&b"a\tb"[..], b"a\tb\t", b"\n"] {
            let e = DictionaryLemmatizer::read(bad).unwrap_err();
            assert_eq!(e.kind, JavaException::ArrayIndexOutOfBounds);
        }
        assert_eq!(java_split(&[], 9), vec![Vec::<u16>::new()]);
        assert_eq!(java_split(&u("\t\t"), 9), Vec::<JString>::new());
        assert_eq!(java_lines(""), Vec::<&str>::new());
    }
}
