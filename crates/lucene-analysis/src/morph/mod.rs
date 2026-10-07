//! analysis-common's `org.apache.lucene.analysis.morph`: the Viterbi lattice
//! and dictionary readers Kuromoji (`lucene-analysis-kuromoji`) and Nori
//! (`lucene-analysis-nori`) are built on (M12 T12.1; deferred from M11).
//!
//! - [`resource`] -- the `DataInput`/`CodecUtil.checkHeader` reads the
//!   dictionary files need.
//! - [`token_info_fst`] -- `TokenInfoFST`: the read path of an
//!   `FST<Long>` (`FST.findTargetArc` over every node encoding) with the
//!   root-arc cache, and the trie a user dictionary compiles to.
//! - [`binary_dictionary`], [`character_definition`],
//!   [`connection_costs`] -- `BinaryDictionary`'s target map and entry
//!   buffer, `CharacterDefinition`, `ConnectionCosts`.
//! - [`viterbi`] -- `Viterbi`, its `Position` and `WrappedPositionArray`;
//!   [`viterbi_nbest`] -- `ViterbiNBest` and its `Lattice`;
//!   [`graphviz`] -- `GraphvizFormatter`.
//! - [`token`] -- `Token`, `TokenType`, `MorphData`.
//!
//! Every file here is read off disk (a caller may load its own
//! dictionaries), so the module carries the arithmetic gate
//! (`docs/arithmetic-gate.md`) although the crate does not: a length, count
//! or offset from a dictionary reaches no `+`, index or allocation unbounded.
//! Java's `int` cost arithmetic (which wraps) is spelled `wrapping_*`.
//!
//! The dictionary *builders* (`BinaryDictionaryWriter`,
//! `CharacterDefinitionWriter`, `ConnectionCostsWriter`,
//! `DictionaryEntryWriter`) are not ported: M12 loads dictionaries, it does
//! not build them.
#![deny(clippy::arithmetic_side_effects)]

pub mod binary_dictionary;
pub mod character_definition;
pub mod connection_costs;
pub mod graphviz;
pub mod resource;
pub mod token;
pub mod token_info_fst;
pub mod viterbi;
pub mod viterbi_nbest;

pub use binary_dictionary::BinaryDictionary;
pub use character_definition::CharacterDefinition;
pub use connection_costs::ConnectionCosts;
pub use graphviz::GraphvizFormatter;
pub use token::{MorphData, MorphToken, Token, TokenType};
pub use token_info_fst::{FstArc, TokenInfoFst};
pub use viterbi::{Position, Viterbi, ViterbiLang, WrappedPositionArray};
pub use viterbi_nbest::{Lattice, NBestState};
