#![forbid(unsafe_code)]
//! lucene-analysis-opennlp: Lucene's `analysis-opennlp` module (M12 T12.5)
//! -- `OpenNLPTokenizer` (sentence detection, then tokenization, each token
//! with its sentence index), `OpenNLPPOSFilter` (the part-of-speech tag as
//! the token type), `OpenNLPChunkerFilter` (the chunk tag as the type),
//! `OpenNLPLemmatizerFilter` (dictionary and/or statistical lemmas) and
//! their factories -- over a port of the Apache OpenNLP 2.5.3 inference
//! runtime they call ([`tools`]), reading OpenNLP's own model packages.
//!
//! **Why a port and not a JVM call.** The models are zips of a manifest,
//! dictionaries and maxent/perceptron parameters that inference only
//! evaluates: sums of feature weights, `StrictMath.exp`, a beam search. The
//! port reads the same files and computes the same doubles (fdlibm's `exp`
//! and `log`, Java's `PriorityQueue` order for equal scores), so it tags as
//! OpenNLP does, which `tests/opennlp_fixtures.rs` checks against models
//! trained by OpenNLP itself. Training is not ported: Lucene never trains.
//!
//! Refused (`UnsupportedOperationException`): `QN` and `NaiveBayes` event
//! models, feature generators beyond the POS tagger's defaults, and user
//! tool-factory or serializer classes -- each named in its error.
//! `NLPNERTaggerOp` (no Lucene filter calls it) is not ported.

pub mod factory;
pub mod filters;
pub mod ops;
pub mod tokenizer;
pub mod tools;

pub use factory::{
    register_factories, OpenNlpChunkerFilterFactory, OpenNlpLemmatizerFilterFactory,
    OpenNlpPosFilterFactory, OpenNlpTokenizerFactory,
};
pub use filters::{OpenNlpChunkerFilter, OpenNlpLemmatizerFilter, OpenNlpPosFilter};
pub use ops::LemmatizerOp;
pub use tokenizer::{open_nlp_tokenizer, OpenNlpSentenceBreakIterator, OpenNlpTokenizer};
