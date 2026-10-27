//! The port of Apache OpenNLP 2.5.3's inference runtime that Lucene's
//! module calls (`opennlp-tools`): model packages ([`model`] over [`zip`],
//! [`properties`], [`xml`] and [`dictionary`]), event models and the beam
//! search ([`ml`]), and the six tools -- [`sentdetect`], [`tokenize`],
//! [`postag`] (with [`featuregen`]), [`chunker`], [`lemmatizer`] and
//! [`namefind`].

pub mod chunker;
pub mod dictionary;
pub mod featuregen;
pub mod lemmatizer;
pub mod ml;
pub mod model;
pub mod namefind;
pub mod postag;
pub mod properties;
pub mod sentdetect;
mod supplementary_scripts;
pub mod text;
pub mod tokenize;
pub mod xml;
pub mod zip;
