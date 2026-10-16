//! `org.apache.lucene.monitor` (M13 T13.5): the percolator. Queries are
//! registered once; a document batch is matched against all of them,
//! narrowed by a presearcher over each query's term tree.
//!
//! T13.5a -- the query analysis the presearcher is built from: the term tree
//! ([`tree`]) and the analyzer and decomposer ([`analyzer`]).

pub mod analyzer;
pub mod facade;
pub mod index;
pub mod presearcher;
pub mod tree;

pub use analyzer::{QueryAnalyzer, QueryDecomposer};
pub use tree::{QueryTree, Term, TermWeightor, ANYTOKEN, ANYTOKEN_FIELD};
