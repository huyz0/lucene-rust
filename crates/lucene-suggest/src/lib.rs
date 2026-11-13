#![forbid(unsafe_code)]
//! `org.apache.lucene.search.suggest` and `org.apache.lucene.search.spell`
//! (`lucene-suggest`): the suggester API, the completion postings format and
//! the spell checkers.
//!
//! - [`lookup`] (`Lookup`, `LookupResult`, `LookupPriorityQueue`) and
//!   [`input`] (`InputIterator` and its buffering and sorting wrappers).
//! - [`dictionary`] (`FileDictionary`, `DocumentDictionary`).
//! - [`tst`] (`TSTLookup`), [`wfst`] (`WFSTCompletionLookup`) and
//!   [`fst_completion`] (`FSTCompletion` and its lookup and sorters).
//! - [`analyzing`] (`AnalyzingSuggester`, `FuzzySuggester`,
//!   `SuggestStopFilter`), [`suggest_stop_factory`] and [`free_text`]
//!   (`FreeTextSuggester`).
//! - [`fst_util`] (`FSTUtil`) and [`document`] (the completion lookup).

pub mod analyzing;
pub mod completion_query;
pub mod dictionary;
pub mod document;
pub mod free_text;
pub mod fst_completion;
pub mod fst_util;
pub mod infix;
pub mod input;
pub mod lookup;
pub mod spell;
pub mod suggest_field;
pub mod suggest_stop_factory;
pub mod tst;
pub mod wfst;
