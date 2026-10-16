//! `org.apache.lucene.search.suggest` (`lucene-suggest`), M13 T13.3: the
//! suggester API and the completion postings format. T13.3a is the base layer:
//! [`lookup`] (`Lookup`, `LookupResult`, `LookupPriorityQueue`) and [`input`]
//! (`InputIterator` and its buffering and sorting wrappers); T13.3b adds [`tst`]
//! (`TSTLookup`); T13.3d adds [`fst_util`] (`FSTUtil`) and [`document`] (the completion lookup).

pub mod document;
pub mod fst_util;
pub mod input;
pub mod lookup;
pub mod tst;
