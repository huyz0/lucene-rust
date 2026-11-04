//! The `lucene-sandbox` queries that are cases of the core query tree
//! ([`crate::extended_query::ExtendedQuery`]): `CoveringQuery`,
//! `TermAutomatonQuery` and `PhraseWildcardQuery`. The rest of the module is
//! the `lucene-sandbox` crate, whose `search` package re-exports these.

pub mod search;
