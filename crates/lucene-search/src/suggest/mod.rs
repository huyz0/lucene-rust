//! `org.apache.lucene.search.suggest` (`lucene-suggest`), M13 T13.3: the
//! suggester API and the completion postings format. T13.3a is the base layer:
//! [`lookup`] (`Lookup`, `LookupResult`, `LookupPriorityQueue`) and [`input`]
//! (`InputIterator` and its buffering and sorting wrappers); T13.3b adds [`tst`]
//! (`TSTLookup`); T13.3b also adds [`wfst`] (`WFSTCompletionLookup`) and
//! [`fst_completion`] (`FSTCompletion` and its lookup and sorters), and T13.3a's
//! [`dictionary`] (`FileDictionary`, `DocumentDictionary`); T13.3c adds [`analyzing`] (`AnalyzingSuggester`, `FuzzySuggester`, `SuggestStopFilter`) and [`suggest_stop_factory`] and [`free_text`] (`FreeTextSuggester`); T13.3d adds [`fst_util`] (`FSTUtil`) and [`document`] (the completion lookup).

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
pub mod suggest_field;
pub mod suggest_stop_factory;
pub mod tst;
pub mod wfst;
