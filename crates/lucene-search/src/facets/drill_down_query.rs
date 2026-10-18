//! `DrillDownQuery`: a base query narrowed to documents in chosen facet
//! categories, one `FILTER` clause per dimension (the values of one dim are
//! OR'd, the dims AND'd).
//!
//! Java's `DrillDownQuery` is a `Query` that rewrites to the `BooleanQuery`
//! [`DrillDownQuery::boolean_query`] builds; here it is a builder whose
//! [`DrillDownQuery::rewrite`] hands that `BooleanQuery` to a search, and
//! [`crate::facets::DrillSideways`] reads its parts.
//!
//! Differs: `add(dim, path)` returns Java's `IllegalArgumentException` for an
//! empty path component as an `Err` (and adds nothing); `term` returns the
//! `TermQuery` over Java's `Term`. `clone()` and `hashCode` are Rust's own.
//! `equals` is structural: Java compares its dims' `BooleanQuery.Builder`s,
//! which have no `equals`, so two separately built queries with a dim are
//! never equal there while a clone (sharing the builders) is. `clone()` and
//! the filter constructor deep-copy the dims, where Java's share their
//! builders (an `add` to the copy also changes the original).

use std::fmt;

use crate::explain::describe_clause;
use crate::facets::{path_to_string, FacetsConfig};
use crate::query::{BooleanQuery, Clause, MatchAllDocsQuery, TermQuery};
use crate::{Error, Result};

/// `DrillDownQuery`.
#[derive(Debug, Clone)]
pub struct DrillDownQuery<'c> {
    config: &'c FacetsConfig,
    base_query: Option<Clause>,
    /// Each dim's values, `SHOULD` clauses, in the order the dims were first
    /// added.
    dim_queries: Vec<BooleanQuery>,
    /// `drillDownDims`: the dims, in the order they were first added (a
    /// dim's position is its index in [`Self::dim_queries`]).
    dims: Vec<String>,
}

impl<'c> DrillDownQuery<'c> {
    /// `DrillDownQuery.term(field, dim, path)`: the drill-down term of a
    /// category, in index field `field`.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for an empty path component.
    pub fn term(field: &str, dim: &str, path: &[&str]) -> Result<TermQuery> {
        let text = path_to_string(dim, path).map_err(|e| Error::IllegalArgument(e.to_string()))?;
        Ok(TermQuery::new(field, text))
    }

    /// `new DrillDownQuery(config)`: no base query (a pure browse).
    pub fn new(config: &'c FacetsConfig) -> Self {
        DrillDownQuery {
            config,
            base_query: None,
            dim_queries: Vec::new(),
            dims: Vec::new(),
        }
    }

    /// `new DrillDownQuery(config, baseQuery)`.
    pub fn with_base(config: &'c FacetsConfig, base_query: Clause) -> Self {
        DrillDownQuery {
            base_query: Some(base_query),
            ..Self::new(config)
        }
    }

    /// `DrillDownQuery(config, filter, other)`: `other`'s drill-downs over the
    /// base `+(other's base, or *:*) #filter`.
    pub(crate) fn with_filter(
        config: &'c FacetsConfig,
        filter: Clause,
        other: &DrillDownQuery<'_>,
    ) -> Self {
        let base = other
            .base_query
            .clone()
            .unwrap_or_else(|| Clause::MatchAllDocs(MatchAllDocsQuery::new(0)));
        DrillDownQuery {
            config,
            base_query: Some(Clause::Boolean(Box::new(BooleanQuery {
                must: vec![base],
                filter: vec![filter],
                ..Default::default()
            }))),
            dim_queries: other.dim_queries.clone(),
            dims: other.dims.clone(),
        }
    }

    /// `add(dim, path...)`: drills down to the category `dim/path` (OR'd with
    /// the dim's other values), matched by its term in the dim's index field.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for an empty path component.
    pub fn add(&mut self, dim: &str, path: &[&str]) -> Result<()> {
        let field = &self.config.dim_config(dim).index_field_name;
        let term = Self::term(field, dim, path)?;
        self.add_query(dim, Clause::Term(term));
        Ok(())
    }

    /// `add(dim, subQuery)`: drills down on `dim` with a query of the
    /// caller's (OR'd with the dim's other values).
    pub fn add_query(&mut self, dim: &str, sub_query: Clause) {
        let index = match self.dims.iter().position(|d| d == dim) {
            Some(i) => i,
            None => {
                self.dims.push(dim.to_string());
                self.dim_queries.push(BooleanQuery::default());
                self.dim_queries.len() - 1
            }
        };
        self.dim_queries[index].should.push(sub_query);
    }

    /// `getBaseQuery()`.
    pub fn base_query(&self) -> Option<&Clause> {
        self.base_query.as_ref()
    }

    /// `getDrillDownQueries()`: each dim's query, in dim order.
    pub fn drill_down_queries(&self) -> Vec<Clause> {
        self.dim_queries
            .iter()
            .map(|q| Clause::Boolean(Box::new(q.clone())))
            .collect()
    }

    /// `getDims()`'s keys, in order: a dim's position is its index here.
    pub fn dims(&self) -> &[String] {
        &self.dims
    }

    /// The config the dims' index fields come from.
    pub fn config(&self) -> &'c FacetsConfig {
        self.config
    }

    /// `getBooleanQuery()`: `+base #dim1 #dim2 ...`.
    pub fn boolean_query(&self) -> BooleanQuery {
        BooleanQuery {
            must: self.base_query.iter().cloned().collect(),
            filter: self.drill_down_queries(),
            ..Default::default()
        }
    }

    /// `rewrite(searcher)`: [`Self::boolean_query`], or `*:*` when it has no
    /// clause (no base and no dim).
    pub fn rewrite(&self) -> BooleanQuery {
        match self.rewrite_clause() {
            Clause::Boolean(b) => *b,
            all => BooleanQuery {
                must: vec![all],
                ..Default::default()
            },
        }
    }

    /// [`Self::rewrite`] as the query Java returns: `MatchAllDocsQuery` when
    /// there is no clause, else the boolean query.
    pub fn rewrite_clause(&self) -> Clause {
        let q = self.boolean_query();
        if q.must.is_empty() && q.filter.is_empty() {
            Clause::MatchAllDocs(MatchAllDocsQuery::new(0))
        } else {
            Clause::Boolean(Box::new(q))
        }
    }
}

/// `equals`: the base query and the dims' queries (not the config), compared
/// structurally (Java compares the dims' builders by identity; module doc).
impl PartialEq for DrillDownQuery<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.base_query == other.base_query && self.dim_queries == other.dim_queries
    }
}

/// `toString()`: the boolean query's.
impl fmt::Display for DrillDownQuery<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let q = Clause::Boolean(Box::new(self.boolean_query()));
        f.write_str(&describe_clause(&q))
    }
}

#[cfg(test)]
mod tests;
