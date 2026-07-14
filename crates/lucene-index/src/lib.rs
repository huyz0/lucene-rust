#![forbid(unsafe_code)]
//! lucene-index: see /PLAN.md for scope.

pub mod check_index;
pub mod deletes;
pub mod index_writer;
pub mod indexing_chain;
pub mod merge;
pub mod merge_policy;
pub mod points_delete;
pub mod segment_info;
pub mod segment_infos;
pub mod segment_writer;
pub mod term_delete;
pub mod update_document;
