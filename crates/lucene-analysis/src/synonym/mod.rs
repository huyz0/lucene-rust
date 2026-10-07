//! `org.apache.lucene.analysis.synonym`: [`SynonymMap`] (built directly or
//! parsed from Solr or WordNet rule files), [`SynonymGraphFilter`] and the
//! deprecated [`SynonymFilter`].
//!
//! The factories (`SynonymGraphFilterFactory`, `SynonymFilterFactory`) are
//! T11.7's; `word2vec` (a model-driven synonym source) is `lucene-search`'s
//! `word2vec` module, as its provider is an HNSW graph.

mod solr_synonym_parser;
mod synonym_filter;
mod synonym_graph_filter;
mod synonym_map;
mod wordnet_synonym_parser;

pub use solr_synonym_parser::SolrSynonymParser;
pub use synonym_filter::SynonymFilter;
pub use synonym_graph_filter::{SynonymGraphFilter, TYPE_SYNONYM};
pub use synonym_map::{
    NodeId, SynonymEntry, SynonymMap, SynonymMapBuilder, SynonymParseError, SynonymParserBase,
    WORD_SEPARATOR,
};
pub use wordnet_synonym_parser::WordnetSynonymParser;
