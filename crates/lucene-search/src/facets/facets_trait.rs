//! `Facets`: the common API of every facet implementation.
//!
//! Java's `null` `FacetResult` is `None`; a `topN` is an `i32`, as in Java,
//! so a non-positive one is refused with Java's message rather than being
//! unrepresentable.

use super::facet_result::{FacetResult, FacetValue};
use crate::{Error, Result};

/// `Facets`.
pub trait Facets {
    /// `getAllChildren(dim, path...)`: every child of the path with a value,
    /// in the implementation's own order; `None` when the path is unknown or
    /// nothing was counted.
    fn get_all_children(&self, dim: &str, path: &[&str]) -> Result<Option<FacetResult>>;

    /// `getTopChildren(topN, dim, path...)`: the `top_n` best children of the
    /// path; `None` when the path is unknown or has no child with a value.
    fn get_top_children(&self, top_n: i32, dim: &str, path: &[&str])
        -> Result<Option<FacetResult>>;

    /// `getSpecificValue(dim, path...)`: the value of one path; `-1` when the
    /// path is unknown.
    fn get_specific_value(&self, dim: &str, path: &[&str]) -> Result<FacetValue>;

    /// `getAllDims(topN)`: every dim with a value, each with its `top_n` best
    /// children.
    fn get_all_dims(&self, top_n: i32) -> Result<Vec<FacetResult>>;

    /// `getTopDims(topNDims, topNChildren)`: the `top_n_dims` best dims, each
    /// with its `top_n_children` best children. The default is Java's: the
    /// head of [`Self::get_all_dims`]; a negative `top_n_dims` fails as
    /// `List.subList` does.
    fn get_top_dims(&self, top_n_dims: i32, top_n_children: i32) -> Result<Vec<FacetResult>> {
        let mut all = self.get_all_dims(top_n_children)?;
        let Ok(keep) = usize::try_from(top_n_dims) else {
            return Err(Error::IllegalArgument(format!(
                "fromIndex(0) > toIndex({top_n_dims})"
            )));
        };
        all.truncate(keep);
        Ok(all)
    }
}

/// `Facets.validateTopN(topN)`.
pub fn validate_top_n(top_n: i32) -> Result<()> {
    if top_n <= 0 {
        return Err(Error::IllegalArgument(format!(
            "topN must be > 0 (got: {top_n})"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
