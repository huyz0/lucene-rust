//! `opennlp.tools.postag`: `POSTaggerME.tag` over a loaded `POSModel`, as
//! Lucene's `NLPPOSTaggerOp` creates it (`POSTagFormat.PENN`): a beam
//! search over the maxent model, each position described by
//! `ConfigurablePOSContextGenerator` (the model's feature generator
//! descriptor; OpenNLP 1.8 and later) or `DefaultPOSContextGenerator`
//! (older models, with their n-gram dictionary), outcomes restricted by
//! the tag dictionary (`DefaultPOSSequenceValidator`), and the tags mapped
//! to Penn Treebank tags when the model's look like Universal Dependencies
//! ones (`POSTagFormatMapper`).

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use lucene_analysis::factory::FactoryError;
use lucene_analysis::AnalysisError;

use super::dictionary::{Dictionary, PosDictionary};
use super::featuregen::{self, Generator};
use super::ml::{BeamSearch, MaxentModel, SequenceContext};
use super::model::{BaseModel, Component, Version};
use super::text::{self, JString};

/// OpenNLP's `pos-default-features.xml`: the descriptor of a model that
/// has none.
const DEFAULT_FEATURES: &[u8] = include_bytes!("../resources/pos-default-features.xml");

/// A loaded `POSModel`.
#[derive(Debug, Clone)]
pub struct PosModel {
    base: Arc<BaseModel>,
    tag_dictionary: Option<Arc<PosDictionary>>,
    ngram_dictionary: Option<Arc<Dictionary>>,
    descriptor: Option<Arc<Vec<u8>>>,
}

impl PosModel {
    /// `new POSModel(InputStream)`.
    pub fn load(bytes: &[u8]) -> Result<PosModel, FactoryError> {
        let base = BaseModel::load(Component::PosTagger, bytes)?;
        Ok(PosModel {
            tag_dictionary: base.pos_dictionary("tags.tagdict"),
            ngram_dictionary: base.dictionary("ngram.dictionary"),
            descriptor: base.bytes("generator.featuregen"),
            base: Arc::new(base),
        })
    }
}

/// `Cache<String, String[]>`: a context generator's contexts of the
/// current sentence by position and previous two tags, at most `capacity`,
/// the eldest dropped first (`Cache` is an insertion-ordered
/// `LinkedHashMap`, so a hit does not refresh an entry).
#[derive(Debug, Clone)]
struct ContextCache {
    capacity: usize,
    sentence: Option<u64>,
    map: HashMap<JString, Vec<JString>>,
    order: VecDeque<JString>,
}

impl ContextCache {
    /// `cacheSize > 0` makes one.
    fn new(capacity: usize) -> Option<ContextCache> {
        (capacity > 0).then(|| ContextCache {
            capacity,
            sentence: None,
            map: HashMap::new(),
            order: VecDeque::new(),
        })
    }

    /// The cached contexts of `key` for `sentence` (another sentence
    /// clears the cache, as Java's identity check on the token array).
    fn get(&mut self, sentence: u64, key: &[u16]) -> Option<&Vec<JString>> {
        if self.sentence == Some(sentence) {
            self.map.get(key)
        } else {
            self.map.clear();
            self.order.clear();
            self.sentence = Some(sentence);
            None
        }
    }

    /// `LinkedHashMap.put` in insertion order: a key already present keeps
    /// its place.
    fn put(&mut self, key: JString, contexts: Vec<JString>) {
        if let Some(v) = self.map.get_mut(&key) {
            *v = contexts;
            return;
        }
        self.order.push_back(key.clone());
        self.map.insert(key, contexts);
        if self.map.len() > self.capacity {
            if let Some(eldest) = self.order.pop_front() {
                self.map.remove(&eldest);
            }
        }
    }
}

/// The two `POSContextGenerator`s.
#[derive(Debug, Clone)]
enum ContextGenerator {
    /// `DefaultPOSContextGenerator`, over the n-gram dictionary.
    Default(Option<Arc<Dictionary>>),
    /// `ConfigurablePOSContextGenerator`, over the descriptor's generators.
    Configurable(Generator),
}

/// `POSTaggerME` with `POSTagFormat.PENN`.
#[derive(Debug, Clone)]
pub struct PosTagger {
    model: Arc<MaxentModel>,
    generator: ContextGenerator,
    cache: Option<ContextCache>,
    tag_dictionary: Option<Arc<PosDictionary>>,
    beam: BeamSearch,
    /// The Penn tag of each outcome (`convertTags`), or the outcome itself.
    penn: Vec<JString>,
    sentence: u64,
}

/// `POSTagFormatMapper.CONVERSION_TABLE_PENN_TO_UD`'s keys.
const PENN_TAGS: [&str; 47] = [
    "#", "$", "''", ",", "-LRB-", "-RRB-", ".", ":", "AFX", "CC", "CD", "DT", "EX", "FW", "HYPH",
    "IN", "JJ", "JJR", "JJS", "LS", "MD", "NIL", "NN", "NNP", "NNPS", "NNS", "PDT", "POS", "PRP",
    "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",
    "WDT", "WP", "WP$", "WRB",
];

/// `POSTagFormatMapper.CONVERSION_TABLE_UD_TO_PENN`.
const UD_TO_PENN: [(&str, &str); 17] = [
    ("ADJ", "JJ"),
    ("ADP", "IN"),
    ("ADV", "RB"),
    ("AUX", "MD"),
    ("CCONJ", "CC"),
    ("DET", "DT"),
    ("INTJ", "UH"),
    ("NOUN", "NN"),
    ("NUM", "CD"),
    ("PART", "RP"),
    ("PRON", "PRP"),
    ("PROPN", "NNP"),
    ("PUNCT", "."),
    ("SCONJ", "IN"),
    ("SYM", "SYM"),
    ("VERB", "VB"),
    ("X", "FW"),
];

/// The tags a tagger reports for each outcome: `POSTagFormatMapper`
/// guesses the model's format from its outcomes (more UD than Penn names:
/// UD, mapped to Penn, `?` for an unknown one; otherwise as they are).
fn penn_tags(model: &MaxentModel) -> Vec<JString> {
    let names: Vec<&str> = (0..model.num_outcomes())
        .map(|i| model.outcome_name(i))
        .collect();
    let ud = names
        .iter()
        .filter(|n| UD_TO_PENN.iter().any(|(u, _)| u == *n))
        .count();
    let penn = names.iter().filter(|n| PENN_TAGS.contains(n)).count();
    (0..model.num_outcomes())
        .map(|i| {
            if ud > penn {
                let mapped = UD_TO_PENN
                    .iter()
                    .find(|(u, _)| *u == names[i])
                    .map_or("?", |(_, p)| p);
                text::units(mapped)
            } else {
                model.outcome(i).to_vec()
            }
        })
        .collect()
}

impl PosTagger {
    /// `new POSTaggerME(model, POSTagFormat.PENN)`: the beam size read from
    /// the manifest, the context generator built.
    pub fn new(model: &PosModel) -> Result<PosTagger, AnalysisError> {
        Self::with_format(model, true)
    }

    /// `new POSTaggerME(model, POSTagFormatMapper.guessFormat(model))`, as
    /// the name finder's `POSTaggerNameFeatureGenerator` makes it: the
    /// format asked for is the one guessed, so the model's own tags.
    pub fn new_untranslated(model: &PosModel) -> Result<PosTagger, AnalysisError> {
        Self::with_format(model, false)
    }

    fn with_format(model: &PosModel, penn: bool) -> Result<PosTagger, AnalysisError> {
        let beam_size = model.base.beam_size(3)?;
        let version: Version = model.base.version();
        let generator = if version.major == 1 && version.minor < 8 {
            ContextGenerator::Default(model.ngram_dictionary.clone())
        } else {
            let descriptor = model
                .descriptor
                .as_deref()
                .map_or(DEFAULT_FEATURES, |d| d.as_slice());
            ContextGenerator::Configurable(featuregen::create_with(
                descriptor,
                &*model.base,
                featuregen::Owner::PosTagger,
            )?)
        };
        let maxent = model.base.maxent_model();
        Ok(PosTagger {
            penn: if penn {
                penn_tags(&maxent)
            } else {
                maxent.outcomes().to_vec()
            },
            model: maxent,
            generator,
            cache: ContextCache::new(usize::try_from(beam_size).unwrap_or(0)),
            tag_dictionary: model.tag_dictionary.clone(),
            beam: BeamSearch::new(beam_size),
            sentence: 0,
        })
    }

    /// `tag(sentence)`: a Penn tag per word.
    pub fn tag(&mut self, words: &[JString]) -> Result<Vec<JString>, AnalysisError> {
        self.sentence = self.sentence.wrapping_add(1);
        let model = Arc::clone(&self.model);
        let mut ctx = PosContext {
            tagger: self,
            words,
            error: None,
        };
        let outcomes = model_best(&model, words.len(), &mut ctx)?;
        if let Some(e) = ctx.error {
            return Err(e);
        }
        Ok(outcomes
            .into_iter()
            .map(|o| self.penn[o as usize].clone())
            .collect())
    }
}

fn model_best(
    model: &MaxentModel,
    len: usize,
    ctx: &mut PosContext<'_>,
) -> Result<Vec<u32>, AnalysisError> {
    let beam = ctx.tagger.beam.clone();
    beam.best_sequence(model, len, ctx)
}

/// One `tag` call's context generator and validator.
struct PosContext<'a> {
    tagger: &'a mut PosTagger,
    words: &'a [JString],
    error: Option<AnalysisError>,
}

impl SequenceContext for PosContext<'_> {
    fn context(
        &mut self,
        index: usize,
        prior: &[u32],
        outcomes: &[JString],
        features: &mut Vec<JString>,
    ) {
        let tags: Vec<JString> = prior
            .iter()
            .map(|&o| outcomes[o as usize].clone())
            .collect();
        let tagprev = index.checked_sub(1).and_then(|i| tags.get(i));
        let tagprevprev = index.checked_sub(2).and_then(|i| tags.get(i));
        // `index + tagprev + tagprevprev`, Java's string concatenation
        let null = text::units("null");
        let key = text::concat(&[
            &text::int_units(index),
            tagprev.unwrap_or(&null),
            tagprevprev.unwrap_or(&null),
        ]);
        let sentence = self.tagger.sentence;
        if let Some(cache) = &mut self.tagger.cache {
            if let Some(cached) = cache.get(sentence, &key) {
                features.extend_from_slice(cached);
                return;
            }
        }
        match &mut self.tagger.generator {
            ContextGenerator::Configurable(g) => {
                if let Err(e) = g.create_features(features, self.words, index, &tags, sentence) {
                    self.error.get_or_insert(e);
                }
            }
            ContextGenerator::Default(dict) => {
                default_context(
                    features,
                    self.words,
                    index,
                    tagprev,
                    tagprevprev,
                    dict.as_deref(),
                );
            }
        }
        if let Some(cache) = &mut self.tagger.cache {
            cache.put(key, features.clone());
        }
    }

    fn valid(&self, i: usize, _prior: &[u32], outcome: u32, outcomes: &[JString]) -> bool {
        match &self.tagger.tag_dictionary {
            None => true,
            Some(d) => match d.tags(&self.words[i]) {
                None => true,
                Some(tags) => tags.contains(&outcomes[outcome as usize]),
            },
        }
    }
}

/// `DefaultPOSContextGenerator.getContext` (without its cache).
// ARITH: token indices of the sentence and lengths below 5.
#[allow(clippy::arithmetic_side_effects)]
fn default_context(
    e: &mut Vec<JString>,
    tokens: &[JString],
    index: usize,
    tagprev: Option<&JString>,
    tagprevprev: Option<&JString>,
    dict: Option<&Dictionary>,
) {
    const SE: &str = "*SE*";
    const SB: &str = "*SB*";
    let cat = |k: &str, v: &[u16]| text::concat(&[&text::units(k), v]);
    let lex = &tokens[index];
    let (next, nextnext) = if tokens.len() > index + 1 {
        let nn = tokens
            .get(index + 2)
            .cloned()
            .unwrap_or_else(|| text::units(SE));
        (tokens[index + 1].clone(), Some(nn))
    } else {
        (text::units(SE), None)
    };
    let (prev, prevprev) = if index >= 1 {
        let pp = if index >= 2 {
            tokens[index - 2].clone()
        } else {
            text::units(SB)
        };
        (tokens[index - 1].clone(), Some(pp))
    } else {
        (text::units(SB), None)
    };
    e.push(text::units("default"));
    e.push(cat("w=", lex));
    if !dict.is_some_and(|d| d.contains(std::slice::from_ref(lex))) {
        for li in 0..4 {
            e.push(cat("suf=", &lex[lex.len().saturating_sub(li + 1)..]));
        }
        for li in 0..4 {
            e.push(cat("pre=", &lex[..(li + 1).min(lex.len())]));
        }
        if lex.contains(&u16::from(b'-')) {
            e.push(text::units("h"));
        }
        if lex
            .iter()
            .any(|c| (u16::from(b'A')..=u16::from(b'Z')).contains(c))
        {
            e.push(text::units("c"));
        }
        if lex
            .iter()
            .any(|c| (u16::from(b'0')..=u16::from(b'9')).contains(c))
        {
            e.push(text::units("d"));
        }
    }
    e.push(cat("p=", &prev));
    if let Some(t) = tagprev {
        e.push(cat("t=", t));
    }
    if let Some(pp) = prevprev {
        e.push(cat("pp=", &pp));
        if let (Some(t2), Some(t)) = (tagprevprev, tagprev) {
            let mut f = cat("t2=", t2);
            f.push(u16::from(b','));
            f.extend_from_slice(t);
            e.push(f);
        }
    }
    e.push(cat("n=", &next));
    if let Some(nn) = nextnext {
        e.push(cat("nn=", &nn));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `Cache`'s insertion order: a re-put key keeps its place, a hit does
    /// not refresh one, and another sentence starts empty.
    #[test]
    fn context_cache_is_an_insertion_ordered_map() {
        let k = text::units;
        let mut c = ContextCache::new(2).unwrap();
        assert!(c.get(1, &k("a")).is_none());
        c.put(k("a"), vec![k("1")]);
        c.put(k("b"), vec![k("2")]);
        c.put(k("a"), vec![k("3")]);
        assert_eq!(c.get(1, &k("a")), Some(&vec![k("3")]));
        c.put(k("c"), vec![k("4")]);
        assert!(c.get(1, &k("a")).is_none());
        assert!(c.get(1, &k("b")).is_some() && c.get(1, &k("c")).is_some());
        assert!(c.get(2, &k("b")).is_none());
        assert!(ContextCache::new(0).is_none());
    }
}
