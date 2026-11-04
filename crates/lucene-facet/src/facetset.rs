//! The `org.apache.lucene.facet.facetset` package: documents carrying sets
//! of numeric tuples ([`FacetSetsField`], a `BINARY` doc value), counted by
//! the [`FacetSetMatcher`]s their tuples match.
//!
//! A [`FacetSetDecoder`] turns the packed bytes back into `long`s:
//! [`decode_longs`] for [`LongFacetSet`] and [`DoubleFacetSet`],
//! [`decode_ints`] for [`IntFacetSet`] and [`FloatFacetSet`].
//!
//! Differs: Java's `FacetSet` subclasses are told apart by class (the
//! "same type" check of `FacetSetsField.create`); here by
//! [`FacetSet::class_name`], which names the Java class.

mod facet_set;
mod matchers;
mod matching_facet_sets_counts;

pub use facet_set::{
    decode_ints, decode_longs, DoubleFacetSet, FacetSet, FacetSetDecoder, FacetSetsField,
    FloatFacetSet, IntFacetSet, LongFacetSet,
};
pub use matchers::{DimRange, ExactFacetSetMatcher, FacetSetMatcher, RangeFacetSetMatcher};
pub use matching_facet_sets_counts::MatchingFacetSetsCounts;
