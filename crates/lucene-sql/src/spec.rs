//! What a SQL table over an index looks like: its columns, how each is read,
//! and how a search column's text is analysed.
//!
//! An index does not say which of its fields are keys, keywords or text, so the
//! table is declared once ([`TableSpec`]) and checked against the segments when
//! it is opened ([`crate::Snapshot::new`]).

use std::fmt;
use std::sync::Arc;

use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};

use crate::error::{Error, Result};

/// The virtual column a `match_text` or `knn` predicate scores into.
pub const SCORE_COLUMN: &str = "_score";

/// How one column is read.
#[derive(Debug, Clone, PartialEq)]
pub enum ColumnKind {
    /// A string field that is stored and indexed as one term (Lucene's
    /// `StringField` with `Store.YES`): read from the stored fields, and
    /// `column = 'x'` is a term lookup.
    Key,
    /// A `long` doc-values field.
    Long,
    /// A single-valued keyword: sorted doc values, also indexed under the same
    /// name, so `column = 'x'` is a term lookup and `GROUP BY column` runs on
    /// the aggregation kernels.
    Keyword,
    /// An analysed text field, searched by `match_text(column, '...')`. It has
    /// no values to read.
    Text,
    /// A float vector field of `dim` dimensions, searched by
    /// `knn(column, '0.1,0.2,...')`. It has no values to read.
    Vector {
        /// Dimensions.
        dim: usize,
    },
}

impl ColumnKind {
    /// Whether the column's values can be projected.
    pub fn is_readable(&self) -> bool {
        matches!(
            self,
            ColumnKind::Key | ColumnKind::Long | ColumnKind::Keyword
        )
    }

    /// Whether `column = 'x'` can be a term lookup.
    pub fn is_term_indexed(&self) -> bool {
        matches!(self, ColumnKind::Key | ColumnKind::Keyword)
    }

    fn data_type(&self) -> DataType {
        match self {
            ColumnKind::Key | ColumnKind::Keyword | ColumnKind::Text => DataType::Utf8,
            ColumnKind::Long => DataType::Int64,
            ColumnKind::Vector { dim } => DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                i32::try_from(*dim).unwrap_or(i32::MAX),
            ),
        }
    }
}

/// One column of the table.
#[derive(Debug, Clone, PartialEq)]
pub struct Column {
    /// The field's name; the column's name in SQL.
    pub name: String,
    /// How it is read.
    pub kind: ColumnKind,
}

/// Turns the text of a `match_text` predicate into the terms searched: what the
/// index's analyzer did to the field when it was written.
#[derive(Clone)]
pub struct Analyzer(Arc<AnalyzeFn>);

type AnalyzeFn = dyn Fn(&str) -> Vec<String> + Send + Sync;

impl Analyzer {
    /// An analyzer from `f`.
    pub fn new(f: impl Fn(&str) -> Vec<String> + Send + Sync + 'static) -> Self {
        Self(Arc::new(f))
    }

    /// Splits on whitespace and lowercases: what a whitespace analyzer with a
    /// lowercase filter indexed.
    pub fn whitespace_lowercase() -> Self {
        Self::new(|text| text.split_whitespace().map(str::to_lowercase).collect())
    }

    /// The terms of `text`.
    pub fn terms(&self, text: &str) -> Vec<String> {
        (self.0)(text)
    }
}

impl fmt::Debug for Analyzer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Analyzer")
    }
}

/// The table a SQL statement queries: a name and its columns. A `_score` column
/// (`Float32`) is always the last.
#[derive(Debug, Clone)]
pub struct TableSpec {
    name: String,
    columns: Vec<Column>,
    analyzer: Analyzer,
}

impl TableSpec {
    /// An empty table called `name` (lower case, as DataFusion folds
    /// unquoted identifiers).
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            columns: Vec::new(),
            analyzer: Analyzer::whitespace_lowercase(),
        }
    }

    fn with(mut self, name: &str, kind: ColumnKind) -> Self {
        self.columns.push(Column {
            name: name.to_string(),
            kind,
        });
        self
    }

    /// Adds a [`ColumnKind::Key`] column.
    #[must_use]
    pub fn key(self, name: &str) -> Self {
        self.with(name, ColumnKind::Key)
    }

    /// Adds a [`ColumnKind::Long`] column.
    #[must_use]
    pub fn long(self, name: &str) -> Self {
        self.with(name, ColumnKind::Long)
    }

    /// Adds a [`ColumnKind::Keyword`] column.
    #[must_use]
    pub fn keyword(self, name: &str) -> Self {
        self.with(name, ColumnKind::Keyword)
    }

    /// Adds a [`ColumnKind::Text`] column.
    #[must_use]
    pub fn text(self, name: &str) -> Self {
        self.with(name, ColumnKind::Text)
    }

    /// Adds a [`ColumnKind::Vector`] column.
    #[must_use]
    pub fn vector(self, name: &str, dim: usize) -> Self {
        self.with(name, ColumnKind::Vector { dim })
    }

    /// Sets how `match_text` analyses its text.
    #[must_use]
    pub fn analyzer(mut self, analyzer: Analyzer) -> Self {
        self.analyzer = analyzer;
        self
    }

    /// The table's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The declared columns, without `_score`.
    pub fn columns(&self) -> &[Column] {
        &self.columns
    }

    /// The analyzer of `match_text`.
    pub fn text_analyzer(&self) -> &Analyzer {
        &self.analyzer
    }

    /// Checks the declaration: lower-case unique names, none called `_score`.
    ///
    /// # Errors
    /// [`Error::Unsupported`] naming the offending column.
    pub fn validate(&self) -> Result<()> {
        if self.name.is_empty() || self.name != self.name.to_lowercase() {
            return Err(Error::unsupported(format!(
                "table name `{}` must be non-empty and lower case",
                self.name
            )));
        }
        for (i, c) in self.columns.iter().enumerate() {
            if c.name.is_empty() || c.name == SCORE_COLUMN {
                return Err(Error::unsupported(format!(
                    "column name `{}` is not allowed",
                    c.name
                )));
            }
            if self.columns[..i].iter().any(|o| o.name == c.name) {
                return Err(Error::unsupported(format!(
                    "column `{}` is declared twice",
                    c.name
                )));
            }
            if let ColumnKind::Vector { dim } = c.kind {
                if dim == 0 || i32::try_from(dim).is_err() {
                    return Err(Error::unsupported(format!(
                        "vector column `{}` has {dim} dimensions",
                        c.name
                    )));
                }
            }
        }
        Ok(())
    }

    /// The index of the column called `name` (the `_score` column is
    /// [`Self::score_index`]).
    pub fn index_of(&self, name: &str) -> Option<usize> {
        if name == SCORE_COLUMN {
            return Some(self.score_index());
        }
        self.columns.iter().position(|c| c.name == name)
    }

    /// The index of the `_score` column: after every declared one.
    pub fn score_index(&self) -> usize {
        self.columns.len()
    }

    /// The Arrow schema: every declared column, then `_score`.
    pub fn arrow_schema(&self) -> SchemaRef {
        let mut fields: Vec<Field> = self
            .columns
            .iter()
            .map(|c| Field::new(&c.name, c.kind.data_type(), true))
            .collect();
        fields.push(Field::new(SCORE_COLUMN, DataType::Float32, true));
        Arc::new(Schema::new(fields))
    }

    /// The columns a `SELECT *` reads: those whose values can be read.
    pub fn readable_columns(&self) -> Vec<usize> {
        self.columns
            .iter()
            .enumerate()
            .filter(|(_, c)| c.kind.is_readable())
            .map(|(i, _)| i)
            .chain(std::iter::once(self.score_index()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> TableSpec {
        TableSpec::new("t")
            .key("id")
            .long("price")
            .keyword("cat")
            .text("body")
            .vector("vec", 4)
    }

    #[test]
    fn schema_has_a_column_per_declaration_and_a_score() {
        let s = spec();
        s.validate().unwrap();
        let schema = s.arrow_schema();
        let names: Vec<_> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(names, ["id", "price", "cat", "body", "vec", "_score"]);
        assert_eq!(schema.field(1).data_type(), &DataType::Int64);
        assert_eq!(schema.field(5).data_type(), &DataType::Float32);
        assert!(matches!(
            schema.field(4).data_type(),
            DataType::FixedSizeList(_, 4)
        ));
        assert_eq!(s.index_of("cat"), Some(2));
        assert_eq!(s.index_of("_score"), Some(5));
        assert_eq!(s.index_of("nope"), None);
        assert_eq!(s.readable_columns(), vec![0, 1, 2, 5]);
        assert_eq!(s.name(), "t");
        assert_eq!(s.columns().len(), 5);
    }

    #[test]
    fn column_kinds_say_what_they_can_do() {
        assert!(ColumnKind::Key.is_term_indexed());
        assert!(ColumnKind::Keyword.is_term_indexed());
        assert!(!ColumnKind::Long.is_term_indexed());
        assert!(!ColumnKind::Text.is_readable());
        assert!(!ColumnKind::Vector { dim: 2 }.is_readable());
        assert!(ColumnKind::Long.is_readable());
    }

    #[test]
    fn validation_rejects_bad_declarations() {
        for bad in [
            TableSpec::new("T"),
            TableSpec::new(""),
            TableSpec::new("t").long("_score"),
            TableSpec::new("t").long(""),
            TableSpec::new("t").long("a").key("a"),
            TableSpec::new("t").vector("v", 0),
        ] {
            assert!(bad.validate().is_err());
        }
    }

    #[test]
    fn analyzers_split_and_lowercase_or_follow_a_custom_function() {
        let a = Analyzer::whitespace_lowercase();
        assert_eq!(a.terms("Hello  WORLD"), ["hello", "world"]);
        let c = Analyzer::new(|t| vec![t.to_string()]);
        assert_eq!(c.terms("A b"), ["A b"]);
        assert_eq!(format!("{c:?}"), "Analyzer");
        let s = TableSpec::new("t").analyzer(c);
        assert_eq!(s.text_analyzer().terms("X"), ["X"]);
    }
}
