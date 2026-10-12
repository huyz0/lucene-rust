//! `org.apache.lucene.analysis.opennlp.tools.NLPLemmatizerOp`: the
//! dictionary lemmatizer first, the statistical one for the words the
//! dictionary lacks (or alone), a word without a lemma kept as it is.
//! (The other `NLP*Op`s are the tools themselves here: Lucene's wrap one
//! OpenNLP call each.)

use std::sync::Arc;

use lucene_analysis::AnalysisError;

use crate::tools::dictionary::DictionaryLemmatizer;
use crate::tools::lemmatizer::Lemmatizer;
use crate::tools::text::{self, JString};

/// `NLPLemmatizerOp`: at least one of the two lemmatizers.
pub struct LemmatizerOp {
    dictionary: Option<Arc<DictionaryLemmatizer>>,
    lemmatizer: Option<Lemmatizer>,
}

impl LemmatizerOp {
    /// `new NLPLemmatizerOp(dictionaryLemmatizer, lemmatizerModel)`.
    pub fn new(
        dictionary: Option<Arc<DictionaryLemmatizer>>,
        lemmatizer: Option<Lemmatizer>,
    ) -> Self {
        LemmatizerOp {
            dictionary,
            lemmatizer,
        }
    }

    /// `lemmatize(words, postags)`.
    pub fn lemmatize(
        &mut self,
        words: &[JString],
        tags: &[JString],
    ) -> Result<Vec<JString>, AnalysisError> {
        let underscore = text::units("_");
        let Some(dictionary) = &self.dictionary else {
            let Some(me) = &mut self.lemmatizer else {
                return Err(AnalysisError::IllegalState(
                    "NullPointerException: NLPLemmatizerOp has no lemmatizer".into(),
                ));
            };
            let mut lemmas = me.lemmatize(words, tags)?;
            for (lemma, word) in lemmas.iter_mut().zip(words) {
                if *lemma == underscore {
                    lemma.clone_from(word);
                }
            }
            return Ok(lemmas);
        };
        let mut lemmas = dictionary.lemmatize(words, tags);
        let mut max_ent: Option<Vec<JString>> = None;
        let o = text::units("O");
        for i in 0..lemmas.len() {
            if lemmas[i] != o {
                continue;
            }
            // not in the dictionary: the statistical lemma if there is one
            lemmas[i] = match &mut self.lemmatizer {
                Some(me) => {
                    if max_ent.is_none() {
                        max_ent = Some(me.lemmatize(words, tags)?);
                    }
                    let m = max_ent.as_ref().map(|m| &m[i]);
                    match m {
                        Some(l) if *l != underscore => l.clone(),
                        _ => words[i].clone(),
                    }
                }
                None => words[i].clone(),
            };
        }
        Ok(lemmas)
    }
}
