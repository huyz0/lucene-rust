//! The document tree `org.apache.lucene.queryparser.xml` walks: what
//! `DocumentBuilderFactory` makes with `FEATURE_SECURE_PROCESSING`, not
//! namespace-aware, read event by event by `quick-xml`.
//!
//! Kept: elements (with their qualified names and attributes), text, CDATA
//! sections (which Java's `DOMUtils.getText` skips, a `CDATA_SECTION_NODE`
//! being neither text nor an element), and entity references, expanded into
//! text. Dropped: comments and processing instructions, which no builder
//! reads. A document type declaration is refused: Java expands an internal
//! subset's entities and refuses external ones, and this reader expands
//! neither (see the module's `Differs`).

use quick_xml::escape::{resolve_predefined_entity, unescape};
use quick_xml::events::Event;
use quick_xml::Reader;

/// An element: its qualified name, its attributes in document order, and
/// its children.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Element {
    pub name: String,
    pub attributes: Vec<(String, String)>,
    pub children: Vec<Node>,
}

/// A child of an element.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Node {
    Element(Element),
    /// Character data, entity references included.
    Text(String),
    /// A `CDATA_SECTION_NODE`: kept, but not part of `getText`.
    CData(String),
}

impl Element {
    /// `Element.getAttribute(name)`: the value, or `""` when absent.
    pub fn attribute(&self, name: &str) -> &str {
        self.attributes
            .iter()
            .find(|(n, _)| n == name)
            .map_or("", |(_, v)| v.as_str())
    }

    /// The element children, in order (`getFirstChild`/`getNextSibling`
    /// over `ELEMENT_NODE`s).
    pub fn elements(&self) -> impl Iterator<Item = &Element> {
        self.children.iter().filter_map(|n| match n {
            Node::Element(e) => Some(e),
            _ => None,
        })
    }

    /// `DOMUtils.getText`: the character data of this element and its
    /// descendants, CDATA sections left out.
    pub fn text(&self) -> String {
        let mut out = String::new();
        self.collect_text(&mut out);
        out
    }

    fn collect_text(&self, out: &mut String) {
        for child in &self.children {
            match child {
                Node::Text(t) => out.push_str(t),
                Node::Element(e) => e.collect_text(out),
                Node::CData(_) => {}
            }
        }
    }
}

/// `DocumentBuilder.parse` of `xml`, returning the document element.
///
/// # Errors
/// The message of the first malformed or refused construct.
pub fn parse(xml: &str) -> Result<Element, String> {
    // XML 1.0's `Char` production, which `quick-xml` does not enforce
    if let Some(c) = xml.chars().find(|&c| !is_xml_char(c)) {
        return Err(invalid_char(c));
    }
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(false);
    let mut stack: Vec<Element> = Vec::new();
    let mut root: Option<Element> = None;
    loop {
        let event = reader.read_event().map_err(syntax)?;
        match event {
            Event::Start(start) => stack.push(element(&start)?),
            Event::Empty(start) => {
                let e = element(&start)?;
                attach(&mut stack, &mut root, e)?;
            }
            Event::End(_) => {
                let e = stack.pop().ok_or("unexpected end tag")?;
                attach(&mut stack, &mut root, e)?;
            }
            Event::Text(text) => {
                let raw = text.xml10_content().map_err(syntax)?;
                let decoded = unescape(&raw).map_err(syntax)?;
                if stack.is_empty() {
                    if !decoded.trim_matches(is_xml_space).is_empty() {
                        return Err("content is not allowed in trailing section".to_string());
                    }
                } else {
                    push_text(&mut stack, Node::Text(decoded.into_owned()));
                }
            }
            Event::GeneralRef(entity) => {
                let name = entity.decode().map_err(syntax)?;
                let text = match entity.resolve_char_ref().map_err(syntax)? {
                    Some(c) if !is_xml_char(c) => return Err(invalid_char(c)),
                    Some(c) => c.to_string(),
                    None => resolve_predefined_entity(&name)
                        .ok_or_else(|| {
                            format!("The entity \"{name}\" was referenced, but not declared.")
                        })?
                        .to_string(),
                };
                if stack.is_empty() {
                    return Err("content is not allowed in trailing section".to_string());
                }
                push_text(&mut stack, Node::Text(text));
            }
            Event::CData(cdata) => {
                let text = cdata.xml10_content().map_err(syntax)?.into_owned();
                if stack.is_empty() {
                    return Err("content is not allowed in trailing section".to_string());
                }
                if let Some(top) = stack.last_mut() {
                    top.children.push(Node::CData(text));
                }
            }
            Event::DocType(_) => return Err("DOCTYPE is not supported".to_string()),
            Event::Eof => break,
            Event::Decl(_) | Event::Comment(_) | Event::PI(_) => {}
        }
    }
    if !stack.is_empty() {
        return Err("XML document structures must start and end within the same entity.".into());
    }
    root.ok_or_else(|| "Premature end of file.".to_string())
}

/// XML 1.0's `Char`: tab, newline, carriage return, and the code points from
/// U+0020 on, less U+FFFE and U+FFFF (a `str` holds no surrogates).
fn is_xml_char(c: char) -> bool {
    matches!(c, '\t' | '\n' | '\r') || (c >= ' ' && c != '\u{fffe}' && c != '\u{ffff}')
}

fn invalid_char(c: char) -> String {
    format!(
        "An invalid XML character (Unicode: 0x{:x}) was found.",
        u32::from(c)
    )
}

/// XML 1.0's attribute-value normalisation for a CDATA attribute: line ends
/// to `\n`, then every literal tab, newline and carriage return to a space;
/// references are expanded after, so `&#9;` stays a tab.
fn normalize_attribute(raw: &[u8]) -> Result<String, String> {
    let raw = std::str::from_utf8(raw).map_err(|e| e.to_string())?;
    let spaced: String = raw
        .replace("\r\n", "\n")
        .chars()
        .map(|c| {
            if matches!(c, '\t' | '\n' | '\r') {
                ' '
            } else {
                c
            }
        })
        .collect();
    Ok(unescape(&spaced).map_err(syntax)?.into_owned())
}

fn is_xml_space(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r')
}

fn element(start: &quick_xml::events::BytesStart<'_>) -> Result<Element, String> {
    let name = std::str::from_utf8(start.name().as_ref())
        .map_err(|e| e.to_string())?
        .to_string();
    let mut attributes = Vec::new();
    for attr in start.attributes() {
        let attr = attr.map_err(syntax)?;
        // `<` may not appear in an attribute value, escaped or not
        if attr.value.contains(&b'<') {
            return Err("The value of an attribute must not contain the '<' character.".into());
        }
        let key = std::str::from_utf8(attr.key.as_ref())
            .map_err(|e| e.to_string())?
            .to_string();
        let value = normalize_attribute(&attr.value)?;
        attributes.push((key, value));
    }
    Ok(Element {
        name,
        attributes,
        children: Vec::new(),
    })
}

/// Hangs a finished element under the open one, or makes it the document
/// element.
fn attach(stack: &mut [Element], root: &mut Option<Element>, e: Element) -> Result<(), String> {
    match stack.last_mut() {
        Some(parent) => {
            parent.children.push(Node::Element(e));
            Ok(())
        }
        None if root.is_none() => {
            *root = Some(e);
            Ok(())
        }
        None => {
            Err("The markup in the document following the root element must be well-formed.".into())
        }
    }
}

fn push_text(stack: &mut [Element], node: Node) {
    if let Some(top) = stack.last_mut() {
        top.children.push(node);
    }
}

fn syntax(e: impl std::fmt::Display) -> String {
    e.to_string()
}
