//! `org.apache.lucene.sandbox.search`'s queries that the core query tree
//! runs.

pub(crate) mod covering;
pub(crate) mod phrase_wildcard;
pub(crate) mod term_automaton;

pub use crate::extended_query::CombinedFieldQuery;
pub use covering::CoveringQuery;

pub use phrase_wildcard::{
    PhraseWildcardBuilder, PhraseWildcardQuery, PhraseWildcardTerm, ResolvedPhraseWildcard,
};
pub use term_automaton::{GraphToken, TermAutomatonQuery, TokenStreamToTermAutomatonQuery};
