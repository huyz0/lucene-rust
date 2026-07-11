#![forbid(unsafe_code)]
//! lucene-codecs: see /PLAN.md for scope.

mod block_packed;
pub mod compound_format;
pub mod direct_monotonic;
pub mod direct_reader;
pub mod doc_values;
pub mod field_infos;
pub mod indexed_disi;
pub mod live_docs;
mod lz4;
pub mod norms;
mod packed_ints;
pub mod stored_fields;
pub mod term_vectors;
pub mod terms_dict;
