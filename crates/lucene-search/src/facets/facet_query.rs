//! `FacetQuery` and `MultiFacetQuery`: the drill-down term of one category,
//! and of any of several categories of one dim, as plain queries.
//!
//! Java's two classes extend `TermQuery` and `TermInSetQuery`; here each
//! wraps the query it is and converts into a [`Clause`].
//!
//! Differs: Java's `IllegalArgumentException` for an empty path component is
//! an `Err`.

use std::fmt;

use crate::explain::describe_clause;
use crate::facets::{path_to_string, FacetsConfig, DEFAULT_INDEX_FIELD_NAME};
use crate::query::{Clause, TermInSetQuery, TermQuery};
use crate::{Error, Result};

/// `FacetsConfig.pathToString(dim, path)`, its refusal an `IllegalArgument`.
fn path_term(dim: &str, path: &[&str]) -> Result<String> {
    path_to_string(dim, path).map_err(|e| Error::IllegalArgument(e.to_string()))
}

/// `FacetQuery`: a `TermQuery` over the drill-down term of `dim/path`.
#[derive(Debug, Clone, PartialEq)]
pub struct FacetQuery(TermQuery);

impl FacetQuery {
    /// `new FacetQuery(config, dim, path...)`: in the dim's index field.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for an empty path component.
    pub fn new(config: &FacetsConfig, dim: &str, path: &[&str]) -> Result<Self> {
        let field = &config.dim_config(dim).index_field_name;
        Ok(FacetQuery(TermQuery::new(
            field.as_str(),
            path_term(dim, path)?,
        )))
    }

    /// `new FacetQuery(dim, path...)`: in the default index field
    /// (`DEFAULT_DIM_CONFIG`).
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for an empty path component.
    pub fn with_default_config(dim: &str, path: &[&str]) -> Result<Self> {
        Ok(FacetQuery(TermQuery::new(
            DEFAULT_INDEX_FIELD_NAME,
            path_term(dim, path)?,
        )))
    }

    /// The `TermQuery` this is.
    pub fn query(&self) -> &TermQuery {
        &self.0
    }

    /// `toString(field)`: the term alone when it is in `field`.
    pub fn to_string_with_field(&self, field: &str) -> String {
        if self.0.field == field {
            String::from_utf8_lossy(&self.0.term).into_owned()
        } else {
            self.to_string()
        }
    }
}

impl From<FacetQuery> for Clause {
    fn from(q: FacetQuery) -> Clause {
        Clause::Term(q.0)
    }
}

/// `toString()`: the term query's.
impl fmt::Display for FacetQuery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&describe_clause(&Clause::Term(self.0.clone())))
    }
}

/// `MultiFacetQuery`: a `TermInSetQuery` over the drill-down terms of several
/// categories of one dim.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiFacetQuery(TermInSetQuery);

impl MultiFacetQuery {
    /// `new MultiFacetQuery(config, dim, paths...)`: in the dim's index field.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for an empty path component.
    pub fn new(config: &FacetsConfig, dim: &str, paths: &[&[&str]]) -> Result<Self> {
        let field = &config.dim_config(dim).index_field_name;
        Self::build(field, dim, paths)
    }

    /// `new MultiFacetQuery(dim, paths...)`: in the default index field.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for an empty path component.
    pub fn with_default_config(dim: &str, paths: &[&[&str]]) -> Result<Self> {
        Self::build(DEFAULT_INDEX_FIELD_NAME, dim, paths)
    }

    /// `toTerms(dim, paths)` into a `TermInSetQuery`, whose terms are sorted
    /// and distinct (`TermInSetQuery`'s `PrefixCodedTerms`).
    fn build(field: &str, dim: &str, paths: &[&[&str]]) -> Result<Self> {
        let mut terms: Vec<Vec<u8>> = Vec::with_capacity(paths.len());
        for path in paths {
            terms.push(path_term(dim, path)?.into_bytes());
        }
        terms.sort();
        terms.dedup();
        Ok(MultiFacetQuery(TermInSetQuery::new(field, terms)))
    }

    /// The `TermInSetQuery` this is.
    pub fn query(&self) -> &TermInSetQuery {
        &self.0
    }

    /// `toString(field)`: `TermInSetQuery` names its field whatever `field`
    /// is.
    pub fn to_string_with_field(&self, _field: &str) -> String {
        self.to_string()
    }
}

impl From<MultiFacetQuery> for Clause {
    fn from(q: MultiFacetQuery) -> Clause {
        Clause::TermInSet(q.0)
    }
}

/// `toString()`: the term-in-set query's.
impl fmt::Display for MultiFacetQuery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&describe_clause(&Clause::TermInSet(self.0.clone())))
    }
}

#[cfg(test)]
mod tests;
