#![forbid(unsafe_code)]
//! lucene-codecs: see /PLAN.md for scope.

mod block_packed;
pub mod blocktree;
pub mod compound_format;
mod deflate;
pub mod direct_monotonic;
pub mod direct_reader;
pub mod doc_values;
pub mod doc_values_updates;
pub mod field_infos;
mod for_util;
pub mod fst;
pub mod fuzzy;
pub mod indexed_disi;
pub mod live_docs;
mod lz4;
pub mod norms;
mod packed_ints;
pub mod points;
pub mod postings;
pub mod postings_writer;
pub mod regexp;
pub mod stored_fields;
pub mod suggest;
pub mod term_vectors;
pub mod terms_dict;
pub mod vectors;
pub mod wildcard;
