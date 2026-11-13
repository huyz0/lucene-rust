//! OpenSearch's shard-level aggregations over a query's matching documents:
//! numeric metrics ([`metric_aggs`]), bucket aggregations and their
//! sub-aggregations ([`bucket_aggs`]) with the per-segment result cache
//! ([`bucket_aggs::segment_cache`]), the `terms` aggregation on keyword fields
//! ([`terms_agg`]), `significant_text` ([`significant_text`]), and the
//! `percentiles` and `cardinality` sketches ([`tdigest`], [`cardinality_sketch`]).
//!
//! Sits above `lucene-search`, whose readers, queries, global ordinals
//! ([`lucene_search::global_ords`]) and segment matching
//! ([`lucene_search::segment_match`]) it reads through.

#![forbid(unsafe_code)]

pub mod bucket_aggs;
pub mod cardinality_sketch;
pub mod metric_aggs;
pub mod significant_text;
pub mod tdigest;
pub mod terms_agg;
