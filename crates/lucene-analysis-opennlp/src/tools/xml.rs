//! The XML a model package holds -- `DictionaryEntryPersistor`'s
//! dictionaries (`.dictionary`, `.tagdict`), read with the JDK's SAX parser,
//! and the feature generator descriptor (`generator.featuregen`), read into
//! a DOM -- parsed into a small element tree.
//!
//! Well-formed XML 1.0 over UTF-8 (a byte-order mark skipped): the XML
//! declaration, comments, processing instructions, CDATA sections, the five
//! predefined entities and character references; line ends normalised to
//! `\n` and white space in attribute values to spaces, as the parser does.
//!
//! Differs: a document type declaration, a declared encoding other than
//! UTF-8 (or US-ASCII, its subset) and a namespace prefix are not read --
//! the first two are refused as `UnsupportedOperationException`s; a prefix
//! is dropped from element and attribute names (the SAX local name), where
//! the JDK first checks it is bound.

use lucene_analysis::factory::{FactoryError, JavaException};

use super::text::{self, JString};

/// A node of the tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Node {
    /// An element.
    Element(Element),
    /// Character data (text and CDATA sections, adjacent ones joined, as
    /// `Text.getWholeText()` reads them).
    Text(JString),
}

/// An element: its qualified name, attributes and children.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Element {
    /// The name as written (`getTagName()`).
    pub name: String,
    /// The attributes, in document order, `xmlns` declarations included.
    pub attributes: Vec<(String, JString)>,
    /// The child nodes.
    pub children: Vec<Node>,
}

/// The local part of a qualified name.
pub fn local_name(name: &str) -> &str {
    name.rsplit(':').next().unwrap_or(name)
}

impl Element {
    /// `getAttribute(name)`: the value, or `""` when absent (as the DOM).
    pub fn attribute(&self, name: &str) -> Option<&JString> {
        self.attributes
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v)
    }

    /// The child elements, in order.
    pub fn elements(&self) -> impl Iterator<Item = &Element> {
        self.children.iter().filter_map(|n| match n {
            Node::Element(e) => Some(e),
            Node::Text(_) => None,
        })
    }
}

fn syntax(message: impl Into<String>) -> FactoryError {
    FactoryError::new(
        JavaException::Io,
        format!("SAXParseException: {}", message.into()),
    )
}

/// Parses `bytes` into its document element.
pub fn parse(bytes: &[u8]) -> Result<Element, FactoryError> {
    let bytes = bytes.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(bytes);
    let text = std::str::from_utf8(bytes).map_err(|e| {
        syntax(format!(
            "Invalid byte of UTF-8 sequence at {}",
            e.valid_up_to()
        ))
    })?;
    // line-end normalisation
    let text = text.replace("\r\n", "\n").replace('\r', "\n");
    let mut p = Parser {
        s: text.as_bytes(),
        src: &text,
        pos: 0,
    };
    p.prolog()?;
    let root = p.element(0)?;
    p.misc()?;
    if p.pos < p.s.len() {
        return Err(syntax("Content is not allowed in trailing section."));
    }
    Ok(root)
}

/// Deepest element nesting accepted (the JDK's parser has no limit of its
/// own, but recursion here needs one).
const MAX_DEPTH: usize = 256;

struct Parser<'a> {
    s: &'a [u8],
    src: &'a str,
    pos: usize,
}

fn is_space(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | b'\r')
}

fn is_name_byte(c: u8) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, b'_' | b':' | b'-' | b'.') || c >= 0x80
}

impl<'a> Parser<'a> {
    fn rest(&self) -> &'a [u8] {
        &self.s[self.pos..]
    }

    // ARITH: `pos < len` is checked first.
    #[allow(clippy::arithmetic_side_effects)]
    fn skip_space(&mut self) {
        while self.pos < self.s.len() && is_space(self.s[self.pos]) {
            self.pos += 1;
        }
    }

    // ARITH: the token was matched at `pos`.
    #[allow(clippy::arithmetic_side_effects)]
    fn expect(&mut self, token: &[u8]) -> Result<(), FactoryError> {
        if self.rest().starts_with(token) {
            self.pos += token.len();
            Ok(())
        } else {
            Err(syntax(format!(
                "expected \"{}\" at offset {}",
                String::from_utf8_lossy(token),
                self.pos
            )))
        }
    }

    /// Up to (and past) `end`, returning what came before it.
    // ARITH: offsets from `find` within the document.
    #[allow(clippy::arithmetic_side_effects)]
    fn until(&mut self, end: &str) -> Result<&'a str, FactoryError> {
        let rest = &self.src[self.pos..];
        let i = rest.find(end).ok_or_else(|| {
            syntax(format!(
                "XML document structures must start and end within the same entity (no \"{end}\")"
            ))
        })?;
        self.pos += i + end.len();
        Ok(&rest[..i])
    }

    // ARITH: offsets within the document.
    #[allow(clippy::arithmetic_side_effects)]
    fn name(&mut self) -> Result<String, FactoryError> {
        let start = self.pos;
        while self.pos < self.s.len() && is_name_byte(self.s[self.pos]) {
            self.pos += 1;
        }
        if self.pos == start
            || self.s[start].is_ascii_digit()
            || matches!(self.s[start], b'-' | b'.')
        {
            return Err(syntax(format!("a name was expected at offset {start}")));
        }
        Ok(self.src[start..self.pos].to_string())
    }

    /// The XML declaration, then comments, processing instructions and
    /// white space.
    // ARITH: offsets within the declaration, from `find`.
    #[allow(clippy::arithmetic_side_effects)]
    fn prolog(&mut self) -> Result<(), FactoryError> {
        if self.rest().starts_with(b"<?xml") && self.s.get(5).is_some_and(|&c| is_space(c)) {
            let decl = self.until("?>")?;
            if let Some(i) = decl.find("encoding") {
                let v = decl[i + 8..]
                    .trim_start()
                    .trim_start_matches('=')
                    .trim_start();
                let quote = v.chars().next().unwrap_or('"');
                let enc = v
                    .get(quote.len_utf8()..)
                    .and_then(|rest| rest.split(quote).next())
                    .unwrap_or("");
                if !enc.eq_ignore_ascii_case("UTF-8")
                    && !enc.eq_ignore_ascii_case("UTF8")
                    && !enc.eq_ignore_ascii_case("US-ASCII")
                    && !enc.eq_ignore_ascii_case("ASCII")
                {
                    return Err(FactoryError::new(
                        JavaException::UnsupportedOperation,
                        format!("XML encoding {enc} is not supported by this port (UTF-8 is)"),
                    ));
                }
            }
        }
        self.misc()?;
        if self.rest().starts_with(b"<!DOCTYPE") {
            return Err(FactoryError::new(
                JavaException::UnsupportedOperation,
                "an XML document type declaration is not supported by this port",
            ));
        }
        Ok(())
    }

    /// Comments, processing instructions and white space.
    // ARITH: the matched marker's length.
    #[allow(clippy::arithmetic_side_effects)]
    fn misc(&mut self) -> Result<(), FactoryError> {
        loop {
            self.skip_space();
            if self.rest().starts_with(b"<!--") {
                self.pos += 4;
                self.until("-->")?;
            } else if self.rest().starts_with(b"<?") {
                self.pos += 2;
                self.until("?>")?;
            } else {
                return Ok(());
            }
        }
    }

    // ARITH: offsets within the document, nesting below `MAX_DEPTH`.
    #[allow(clippy::arithmetic_side_effects)]
    fn element(&mut self, depth: usize) -> Result<Element, FactoryError> {
        if depth > MAX_DEPTH {
            return Err(syntax("elements nested too deeply"));
        }
        self.expect(b"<")?;
        let name = self.name()?;
        let mut el = Element {
            name,
            ..Element::default()
        };
        loop {
            let had_space = self.rest().first().is_some_and(|&c| is_space(c));
            self.skip_space();
            if self.rest().starts_with(b"/>") {
                self.pos += 2;
                return Ok(el);
            }
            if self.rest().starts_with(b">") {
                self.pos += 1;
                break;
            }
            if !had_space {
                return Err(syntax(format!(
                    "Element type \"{}\" must be followed by either attribute specifications, \">\" or \"/>\".",
                    el.name
                )));
            }
            let attr = self.name()?;
            self.skip_space();
            self.expect(b"=")?;
            self.skip_space();
            let quote = *self
                .rest()
                .first()
                .ok_or_else(|| syntax("attribute value expected"))?;
            if quote != b'"' && quote != b'\'' {
                return Err(syntax("Open quote is expected for attribute"));
            }
            self.pos += 1;
            let raw = self.until(if quote == b'"' { "\"" } else { "'" })?;
            if raw.contains('<') {
                return Err(syntax(
                    "The value of attribute must not contain the '<' character.",
                ));
            }
            let mut value = Vec::new();
            decode_into(raw, &mut value)?;
            for u in &mut value {
                if matches!(*u, 0x09 | 0x0A | 0x0D) {
                    *u = 0x20;
                }
            }
            if el.attributes.iter().any(|(n, _)| *n == attr) {
                return Err(syntax(format!(
                    "Attribute \"{attr}\" was already specified"
                )));
            }
            el.attributes.push((attr, value));
        }
        // content
        loop {
            let rest = self.rest();
            if rest.is_empty() {
                return Err(syntax(format!(
                    "The element type \"{}\" must be terminated by the matching end-tag",
                    el.name
                )));
            }
            if rest.starts_with(b"</") {
                self.pos += 2;
                let end = self.name()?;
                if end != el.name {
                    return Err(syntax(format!(
                        "The element type \"{}\" must be terminated by the matching end-tag \"</{}>\".",
                        el.name, el.name
                    )));
                }
                self.skip_space();
                self.expect(b">")?;
                el.children
                    .retain(|n| !matches!(n, Node::Text(t) if t.is_empty()));
                return Ok(el);
            } else if rest.starts_with(b"<!--") {
                self.pos += 4;
                self.until("-->")?;
                // a comment ends a text node
                el.children.push(Node::Text(Vec::new()));
            } else if rest.starts_with(b"<![CDATA[") {
                self.pos += 9;
                let data = self.until("]]>")?;
                push_text(&mut el.children, &text::units(data));
            } else if rest.starts_with(b"<?") {
                self.pos += 2;
                self.until("?>")?;
            } else if rest.starts_with(b"<") {
                let child = self.element(depth + 1)?;
                el.children.push(Node::Element(child));
            } else {
                let start = self.pos;
                while self.pos < self.s.len() && self.s[self.pos] != b'<' {
                    self.pos += 1;
                }
                let mut units = Vec::new();
                decode_into(&self.src[start..self.pos], &mut units)?;
                push_text(&mut el.children, &units);
            }
        }
    }
}

fn push_text(children: &mut Vec<Node>, units: &[u16]) {
    match children.last_mut() {
        // adjacent character data is one text node; an empty node is the
        // marker a comment leaves, which the new text node replaces
        Some(Node::Text(t)) => t.extend_from_slice(units),
        _ => children.push(Node::Text(units.to_vec())),
    }
}

/// Character data with its entity and character references resolved.
// ARITH: offsets within the text, from `find`.
#[allow(clippy::arithmetic_side_effects)]
fn decode_into(raw: &str, out: &mut JString) -> Result<(), FactoryError> {
    let mut rest = raw;
    while let Some(i) = rest.find('&') {
        out.extend(rest[..i].encode_utf16());
        let after = &rest[i + 1..];
        let end = after.find(';').ok_or_else(|| {
            syntax("The entity name must immediately follow the '&' in the entity reference.")
        })?;
        let name = &after[..end];
        let cp = match name {
            "lt" => Some(0x3C),
            "gt" => Some(0x3E),
            "amp" => Some(0x26),
            "quot" => Some(0x22),
            "apos" => Some(0x27),
            _ => {
                let num = if let Some(h) = name.strip_prefix("#x") {
                    u32::from_str_radix(h, 16).ok()
                } else if let Some(d) = name.strip_prefix('#') {
                    d.parse::<u32>().ok()
                } else {
                    return Err(syntax(format!(
                        "The entity \"{name}\" was referenced, but not declared."
                    )));
                };
                Some(
                    num.filter(|&c| char::from_u32(c).is_some() && c != 0)
                        .ok_or_else(|| {
                            syntax(format!(
                                "A character reference &{name}; is not a legal XML character."
                            ))
                        })?,
                )
            }
        };
        if let Some(cp) = cp {
            text::push_code_point(out, cp);
        }
        rest = &after[end + 1..];
    }
    out.extend(rest.encode_utf16());
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arithmetic_side_effects)] // test code
    use super::*;

    fn t(s: &str) -> JString {
        text::units(s)
    }

    #[test]
    fn documents() {
        let doc = parse(
            "\u{FEFF}<?xml version=\"1.0\" encoding='UTF-8'?>\n<!-- c --><?pi x?>\r\n<a x=\"1\tz\" y='&lt;&#65;&#x42;'>t1<!--c-->t2<![CDATA[<c>]]>&amp;<b/><c k=\"v\" >in</c ></a>\n<!-- after -->"
                .as_bytes(),
        )
        .unwrap();
        assert_eq!(doc.name, "a");
        assert_eq!(doc.attribute("x"), Some(&t("1 z")));
        assert_eq!(doc.attribute("y"), Some(&t("<AB")));
        assert_eq!(doc.attribute("z"), None);
        assert_eq!(
            doc.children,
            vec![
                Node::Text(t("t1")),
                Node::Text(t("t2<c>&")),
                Node::Element(Element {
                    name: "b".into(),
                    ..Element::default()
                }),
                Node::Element(Element {
                    name: "c".into(),
                    attributes: vec![("k".into(), t("v"))],
                    children: vec![Node::Text(t("in"))],
                }),
            ]
        );
        assert_eq!(doc.elements().count(), 2);
        assert_eq!(local_name("ns:entry"), "entry");
        assert_eq!(local_name("entry"), "entry");
        assert_eq!(
            parse(b"<a>&#128512;</a>").unwrap().children,
            vec![Node::Text(t("😀"))]
        );
        assert_eq!(
            parse(b"<?xml version=\"1.0\" encoding=\"US-ASCII\"?><a/>")
                .unwrap()
                .name,
            "a"
        );
    }

    #[test]
    fn malformed_documents() {
        for bad in [
            &b"<a>"[..],
            b"<a></b>",
            b"<a x=1/>",
            b"<a x=\"1\" x=\"2\"/>",
            b"<a x=\"<\"/>",
            b"<a>&foo;</a>",
            b"<a>&#0;</a>",
            b"<a>&#xZZ;</a>",
            b"<a>& b</a>",
            b"<a/>junk",
            b"",
            b"<1a/>",
            b"<a\"/>",
            b"<a x=\"1\"y=\"2\"/>",
            b"<a x",
            b"<a><!-- x</a>",
            b"\xff<a/>",
        ] {
            let e = parse(bad).unwrap_err();
            assert_eq!(
                e.kind,
                JavaException::Io,
                "{:?}",
                String::from_utf8_lossy(bad)
            );
        }
        let deep = "<a>".repeat(300);
        assert!(parse(deep.as_bytes()).is_err());
        for refused in [
            &b"<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?><a/>"[..],
            "<?xml version=\"1.0\" encoding=é?><a/>".as_bytes(),
            b"<!DOCTYPE a><a/>",
        ] {
            assert_eq!(
                parse(refused).unwrap_err().kind,
                JavaException::UnsupportedOperation
            );
        }
    }
}
