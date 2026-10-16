//! `org.apache.lucene.search.suggest` (`lucene-suggest`), M13 T13.3: the
//! suggester API and the completion postings format. T13.3a is the base layer:
//! [`lookup`] (`Lookup`, `LookupResult`, `LookupPriorityQueue`) and [`input`]
//! (`InputIterator` and its buffering and sorting wrappers).

pub mod input;
pub mod lookup;
