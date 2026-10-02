//! The pure-geometry half of Lucene 10.5.0's `lucene-spatial-extras`
//! (`org.apache.lucene.spatial`): the Spatial4j-to-spatial3d bridge
//! ([`spatial4j`]), the spatial prefix trees and their cells
//! ([`prefix_tree`]) and, to come, the query arguments (`query`).
//!
//! What indexes and searches -- the `SpatialStrategy`s, their queries,
//! value sources and facet counters -- lives in `lucene_search::spatial`,
//! the lowest crate that has both fields and queries.

#![forbid(unsafe_code)]

pub mod prefix_tree;
pub mod spatial4j;
