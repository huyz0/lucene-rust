//! `MultiFacets`: dims routed to the [`Facets`] that counted them.

use super::facet_result::{FacetResult, FacetValue};
use super::facets_trait::{validate_top_n, Facets};
use crate::{Error, Result};

/// `MultiFacets`: each dim of `dim_to_facets` answered by its own
/// [`Facets`], every other dim by the default one (or refused when there is
/// none).
///
/// Java takes a `Map`; [`MultiFacets::get_all_dims`] walks it in its
/// iteration order, which here is the order of the entries given.
///
/// Differs: Java's `getAllDims` adds a mapped dim's `null` `getTopChildren`
/// result to its list; a Rust list has no `null`, so that entry is left out.
pub struct MultiFacets<'f> {
    dim_to_facets: Vec<(String, &'f dyn Facets)>,
    default_facets: Option<&'f dyn Facets>,
}

impl<'f> MultiFacets<'f> {
    /// `new MultiFacets(dimToFacets, defaultFacets)`.
    pub fn new(
        dim_to_facets: Vec<(String, &'f dyn Facets)>,
        default_facets: Option<&'f dyn Facets>,
    ) -> Self {
        MultiFacets {
            dim_to_facets,
            default_facets,
        }
    }

    fn facets_for(&self, dim: &str) -> Result<&'f dyn Facets> {
        if let Some((_, f)) = self.dim_to_facets.iter().find(|(d, _)| d == dim) {
            return Ok(*f);
        }
        self.default_facets
            .ok_or_else(|| Error::IllegalArgument(format!("invalid dim \"{dim}\"")))
    }
}

impl Facets for MultiFacets<'_> {
    fn get_all_children(&self, dim: &str, path: &[&str]) -> Result<Option<FacetResult>> {
        self.facets_for(dim)?.get_all_children(dim, path)
    }

    fn get_top_children(
        &self,
        top_n: i32,
        dim: &str,
        path: &[&str],
    ) -> Result<Option<FacetResult>> {
        validate_top_n(top_n)?;
        self.facets_for(dim)?.get_top_children(top_n, dim, path)
    }

    fn get_specific_value(&self, dim: &str, path: &[&str]) -> Result<FacetValue> {
        self.facets_for(dim)?.get_specific_value(dim, path)
    }

    fn get_all_dims(&self, top_n: i32) -> Result<Vec<FacetResult>> {
        validate_top_n(top_n)?;
        let mut results = Vec::new();
        for (dim, facets) in &self.dim_to_facets {
            if let Some(r) = facets.get_top_children(top_n, dim, &[])? {
                results.push(r);
            }
        }
        if let Some(default) = self.default_facets {
            for r in default.get_all_dims(top_n)? {
                if !self.dim_to_facets.iter().any(|(d, _)| *d == r.dim) {
                    results.push(r);
                }
            }
        }
        Ok(results)
    }
}

#[cfg(test)]
mod tests;
