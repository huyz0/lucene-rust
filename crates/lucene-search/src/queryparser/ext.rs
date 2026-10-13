//! `org.apache.lucene.queryparser.ext`: the extendable query parser. A field
//! name carrying the extension delimiter (`title:ext` in a query string
//! `title\:ext:text`, the `:` escaped to stay inside the field) hands its
//! text to the [`ParserExtension`] registered under the key after the
//! delimiter, which builds the query.

use std::collections::HashMap;
use std::sync::Arc;

use lucene_analysis::Analyzer;

use super::classic::{QueryParser, QueryParserError};
use super::ParsedQuery;

/// `ParserExtension`: the query an [`ExtensionQuery`] stands for (`None` is
/// Java's `null`).
pub trait ParserExtension: Send + Sync {
    /// `parse(query)`.
    ///
    /// # Errors
    /// A `ParseException` ([`QueryParserError::is_parse_exception`]) is
    /// reported by the parser as `Cannot parse '...': ...`; any other error
    /// passes through.
    fn parse(&self, query: &ExtensionQuery<'_>) -> Result<Option<ParsedQuery>, QueryParserError>;
}

/// `ExtensionQuery`: what an extension is asked to parse.
pub struct ExtensionQuery<'a> {
    /// `topLevelParser()`: the parser the extension was found by, for
    /// parsing a sub-query with its settings.
    pub top_level_parser: &'a QueryParser,
    /// `field()`: the field, the extension key cut off.
    pub field: &'a str,
    /// `rawQueryString()`: the text after the field.
    pub raw_query_string: &'a str,
}

/// `Extensions.DEFAULT_EXTENSION_FIELD_DELIMITER`.
pub const DEFAULT_EXTENSION_FIELD_DELIMITER: char = ':';

/// `Extensions`: the registered extensions, and the delimiter that splits
/// a field name from an extension key.
pub struct Extensions {
    extensions: HashMap<String, Arc<dyn ParserExtension>>,
    delimiter: char,
}

impl Default for Extensions {
    fn default() -> Self {
        Self::new()
    }
}

impl Extensions {
    /// `Extensions()`: the delimiter is `:`.
    pub fn new() -> Self {
        Self::with_delimiter(DEFAULT_EXTENSION_FIELD_DELIMITER)
    }

    /// `Extensions(extensionFieldDelimiter)`.
    pub fn with_delimiter(delimiter: char) -> Self {
        Extensions {
            extensions: HashMap::new(),
            delimiter,
        }
    }

    /// `add(key, extension)`.
    pub fn add(&mut self, key: impl Into<String>, extension: Arc<dyn ParserExtension>) {
        self.extensions.insert(key.into(), extension);
    }

    /// `getExtension(key)`.
    pub fn extension(&self, key: &str) -> Option<&Arc<dyn ParserExtension>> {
        self.extensions.get(key)
    }

    /// `getExtensionFieldDelimiter()`.
    pub fn extension_field_delimiter(&self) -> char {
        self.delimiter
    }

    /// `splitExtensionField(defaultField, field)`: the field and the
    /// extension key after the delimiter (`None` when there is no delimiter);
    /// a delimiter first in the field leaves the default field.
    pub fn split_extension_field<'a>(
        &self,
        default_field: &'a str,
        field: &'a str,
    ) -> (&'a str, Option<&'a str>) {
        let Some(index) = field.find(self.delimiter) else {
            return (field, None);
        };
        let index_field = if index == 0 {
            default_field
        } else {
            &field[..index]
        };
        (
            index_field,
            Some(&field[index + self.delimiter.len_utf8()..]),
        )
    }

    /// `escapeExtensionField(extfield)`.
    pub fn escape_extension_field(&self, extfield: &str) -> String {
        QueryParser::escape(extfield)
    }

    /// `buildExtensionField(extensionKey)`.
    pub fn build_extension_field(&self, extension_key: &str) -> String {
        self.build_extension_field_for(extension_key, "")
    }

    /// `buildExtensionField(extensionKey, field)`: `field`, the delimiter
    /// and the key, escaped.
    pub fn build_extension_field_for(&self, extension_key: &str, field: &str) -> String {
        let mut raw = String::from(field);
        raw.push(self.delimiter);
        raw.push_str(extension_key);
        self.escape_extension_field(&raw)
    }
}

/// `ExtendableQueryParser`: a [`QueryParser`] whose field queries go to
/// [`Extensions`] first.
pub struct ExtendableQueryParser;

impl ExtendableQueryParser {
    /// `ExtendableQueryParser(f, a)`: no extension is registered.
    pub fn without_extensions(
        field: impl Into<String>,
        analyzer: impl Into<Arc<Analyzer>>,
    ) -> QueryParser {
        Self::with_extensions(field, analyzer, Extensions::new())
    }

    /// `ExtendableQueryParser(f, a, ext)`.
    pub fn with_extensions(
        field: impl Into<String>,
        analyzer: impl Into<Arc<Analyzer>>,
        extensions: Extensions,
    ) -> QueryParser {
        QueryParser::new(field, analyzer).with_extensions(Arc::new(extensions))
    }
}
