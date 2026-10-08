//! A port of the parts of ICU4J 77.1 (`com.ibm.icu`) that Lucene's
//! `analysis-icu` module calls, reading ICU's own binary data files.
//!
//! ICU4J is Copyright (C) 2016 and later Unicode, Inc. and others, under
//! the Unicode License v3 (`NOTICE`; the licence text is in the root
//! `LICENSE`). Each module names the ICU4J class it ports; the algorithms
//! and data formats are ICU's, the Rust is this project's.
//!
//! - [`binary`]: `ICUBinary`, the data file header.
//! - [`code_point_trie`]: `CodePointTrie` (read-only).
//! - [`normalizer2_impl`], [`normalizer2`]: `Normalizer2Impl`,
//!   `Norm2AllModes`, `Normalizer2`, `FilteredNormalizer2`.
//! - [`unicode_set`]: `UnicodeSet` and its pattern syntax.
//! - [`rbbi_data`], [`rbbi`]: `RBBIDataWrapper`, `RuleBasedBreakIterator`.
//! - [`break_engines`], [`tries`]: the dictionary break engines and the
//!   `BytesTrie`/`CharsTrie` their dictionaries are.
//! - [`char_iter`]: `CharacterIterator` and `CharacterIteration`.
//! - [`coll`]: collation (`RuleBasedCollator` sort keys).
//! - [`trie2`]: `Trie2_32`/`Trie2_16`, the older code point trie.
//! - [`uprops`]: character properties and their aliases.
//! - [`utf16`]: Java's surrogate-pair conventions.

pub mod binary;
pub mod break_engines;
pub mod char_iter;
pub mod code_point_trie;
pub mod coll;
pub mod normalizer2;
pub mod normalizer2_impl;
pub mod rbbi;
pub mod rbbi_data;
pub mod translit;
pub mod trie2;
pub mod tries;
pub mod ucase;
pub mod unicode_set;
pub mod uprops;
pub mod utf16;
