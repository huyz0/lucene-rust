//! Ports of Spatial4j (`org.locationtech.spatial4j`: shapes, contexts, WKT,
//! geohash, distance calculators) and of the parts of Lucene's
//! `lucene-spatial-extras` that do not touch an index (`org.apache.lucene.spatial`:
//! prefix trees, `SpatialArgs`, the Geo3D shape adapters).
//!
//! The two modules refer to each other (a Spatial4j context builds Geo3D
//! shapes and S2 cells, the prefix trees are built on Spatial4j shapes), so
//! they are one crate. Index-facing strategies live in `lucene-spatial-extras`.

#![forbid(unsafe_code)]

pub mod spatial4j;
pub mod spatial_extras;
