//! `org.apache.lucene.queryparser.xml`: queries written as XML documents,
//! through `CoreParser` and its builders.
//!
//! [`CoreParser`] is the core set (`TermQuery`, `TermsQuery`,
//! `MatchAllDocsQuery`, `BooleanQuery`, `RangeQuery`, `PointRangeQuery`,
//! `DisjunctionMaxQuery`, `UserQuery`, `ConstantScoreQuery`, and the span
//! queries). [`CorePlusQueriesParser`] adds `LikeThisQuery` (a
//! [`crate::mlt::MoreLikeThisQuery`]) and [`CorePlusExtensionsParser`]
//! `FuzzyLikeThisQuery` too (a [`crate::fuzzy_like_this::FuzzyLikeThisQuery`]).
//! A caller cannot register its own builders as Java's `addQueryBuilder`
//! does.
//!
//! **Differs:** a `FuzzyLikeThisQuery`'s clause order follows the JVM's
//! `BytesRef` hash seed, which Rust has no clock-seeded copy of:
//! [`CorePlusExtensionsParser::with_hash_seed`] sets it (default `0`).
//!
//! **Differs:** a document type declaration is refused (Java accepts an
//! internal subset and refuses external entities); the XML reader is
//! `quick-xml`, whose syntax-error text is not Xerces'. Java's
//! `ParserException` text for a syntax error is therefore the reader's,
//! prefixed as Java prefixes it.

pub(crate) mod builders;
pub(crate) mod dom;
#[cfg(test)]
mod tests;

use std::sync::Arc;

use lucene_analysis::Analyzer;

use crate::queryparser::classic::{QueryParser, QueryParserError};
use crate::queryparser::ParsedQuery;
use builders::{parser_error, Ancestors, Builders, Level};

/// `org.apache.lucene.queryparser.xml.ParserException`, the class of every
/// error the parser reports as Java would.
pub const PARSER_EXCEPTION: &str = "org.apache.lucene.queryparser.xml.ParserException";

/// `CoreParser`: parses an XML query document.
pub struct CoreParser {
    builders: Builders,
}

impl CoreParser {
    /// `new CoreParser(defaultField, analyzer)`: a `UserQuery` makes its own
    /// classic parser on its field (or `default_field`).
    pub fn new(default_field: Option<&str>, analyzer: impl Into<Arc<Analyzer>>) -> Self {
        CoreParser {
            builders: Builders {
                default_field: default_field.map(str::to_string),
                analyzer: analyzer.into(),
                parser: None,
                level: Level::Core,
                hash_seed: 0,
            },
        }
    }

    /// `new CoreParser(analyzer, parser)`: every `UserQuery` goes through
    /// `parser`.
    pub fn with_parser(analyzer: impl Into<Arc<Analyzer>>, parser: QueryParser) -> Self {
        CoreParser {
            builders: Builders {
                default_field: None,
                analyzer: analyzer.into(),
                parser: Some(parser),
                level: Level::Core,
                hash_seed: 0,
            },
        }
    }

    /// `CoreParser.parse(xmlStream)`: the query of the document element.
    ///
    /// # Errors
    /// `ParserException` for a document that is not well-formed, refused, or
    /// not a query the builders make; the runtime exceptions a builder's
    /// parse or constructor throws, as Java lets them through.
    pub fn parse(&self, xml: &str) -> Result<ParsedQuery, QueryParserError> {
        let root =
            dom::parse(xml).map_err(|m| parser_error(format!("Error parsing XML stream: {m}")))?;
        let ancestors: Ancestors<'_> = Vec::new();
        self.builders
            .query(&root, &ancestors)
            .map_err(java_exception)
    }
}

/// `CorePlusQueriesParser`: the core builders and `LikeThisQuery`, whose
/// fields default to `contents`.
pub struct CorePlusQueriesParser {
    core: CoreParser,
}

impl CorePlusQueriesParser {
    /// `new CorePlusQueriesParser(defaultField, analyzer)`.
    pub fn new(default_field: Option<&str>, analyzer: impl Into<Arc<Analyzer>>) -> Self {
        let mut core = CoreParser::new(default_field, analyzer);
        core.builders.level = Level::PlusQueries;
        CorePlusQueriesParser { core }
    }

    /// `new CorePlusQueriesParser(analyzer, parser)`.
    pub fn with_parser(analyzer: impl Into<Arc<Analyzer>>, parser: QueryParser) -> Self {
        let mut core = CoreParser::with_parser(analyzer, parser);
        core.builders.level = Level::PlusQueries;
        CorePlusQueriesParser { core }
    }

    /// `parse(xmlStream)`, as [`CoreParser::parse`].
    ///
    /// # Errors
    /// As [`CoreParser::parse`].
    pub fn parse(&self, xml: &str) -> Result<ParsedQuery, QueryParserError> {
        self.core.parse(xml)
    }
}

/// `CorePlusExtensionsParser`: [`CorePlusQueriesParser`]'s builders and
/// `FuzzyLikeThisQuery`.
pub struct CorePlusExtensionsParser {
    core: CoreParser,
}

impl CorePlusExtensionsParser {
    /// `new CorePlusExtensionsParser(defaultField, analyzer)`.
    pub fn new(default_field: Option<&str>, analyzer: impl Into<Arc<Analyzer>>) -> Self {
        let mut core = CoreParser::new(default_field, analyzer);
        core.builders.level = Level::PlusExtensions;
        CorePlusExtensionsParser { core }
    }

    /// `new CorePlusExtensionsParser(analyzer, parser)`.
    pub fn with_parser(analyzer: impl Into<Arc<Analyzer>>, parser: QueryParser) -> Self {
        let mut core = CoreParser::with_parser(analyzer, parser);
        core.builders.level = Level::PlusExtensions;
        CorePlusExtensionsParser { core }
    }

    /// This parser with `FuzzyLikeThisQuery`s rewritten under the JVM hash
    /// seed `seed` (`StringHelper.GOOD_FAST_HASH_SEED`; see the module's
    /// `Differs`).
    pub fn with_hash_seed(mut self, seed: i32) -> Self {
        self.core.builders.hash_seed = seed;
        self
    }

    /// `parse(xmlStream)`, as [`CoreParser::parse`].
    ///
    /// # Errors
    /// As [`CoreParser::parse`].
    pub fn parse(&self, xml: &str) -> Result<ParsedQuery, QueryParserError> {
        self.core.parse(xml)
    }
}

/// The class a runtime exception carries in Java: `java.lang` for the
/// `IllegalArgumentException` and `NullPointerException` the builders throw
/// through the query constructors, `org.apache.lucene.search` for a
/// `TooManyClauses`.
fn java_exception(mut e: QueryParserError) -> QueryParserError {
    e.class = match e.class {
        "IllegalArgumentException" => "java.lang.IllegalArgumentException",
        "NullPointerException" => "java.lang.NullPointerException",
        "NumberFormatException" => "java.lang.NumberFormatException",
        "TooManyClauses" => "org.apache.lucene.search.TooManyClauses",
        other => other,
    };
    e
}
