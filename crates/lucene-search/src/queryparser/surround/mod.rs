//! `org.apache.lucene.queryparser.surround`: the surround query language
//! (`W`, `N`, `OR`, `AND`, `NOT` operators with distances, `term*` prefixes,
//! `te?m*` truncations, `^weights`, `field:` prefixes), its parser
//! ([`parser`]) and its query tree and rewrite ([`query`]).

mod constants;
pub mod parser;
pub mod query;
mod token_manager;
