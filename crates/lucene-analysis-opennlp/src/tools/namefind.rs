//! `opennlp.tools.namefind`: `NameFinderME.find` over a loaded
//! `TokenNameFinderModel`, as Lucene's `NLPNERTaggerOp` makes it: a beam
//! search over the maxent model, each position described by
//! `DefaultNameContextGenerator` (the model's feature generator descriptor,
//! or OpenNLP's default one, then the previous two outcomes), outcomes
//! restricted by the sequence codec's validator, and the outcomes decoded
//! into typed name spans whose probability is the mean of their tokens'.
//! After every sentence the adaptive generators learn its outcomes, until
//! `clearAdaptiveData`.
//!
//! The sequence codec is the one the manifest names (`BioCodec` or
//! `BilouCodec`); a class Java cannot load there makes it fall back to
//! `BioCodec`, as here any other name does. Not ported: `find` with
//! additional context (`NLPNERTaggerOp` never passes any, so the
//! `AdditionalContextFeatureGenerator` window `NameFinderME` appends adds
//! no feature).

use std::sync::Arc;

use lucene_analysis::factory::FactoryError;
use lucene_analysis::AnalysisError;

use super::featuregen::{self, Generator, Owner, Prior};
use super::ml::{BeamSearch, MaxentModel, SequenceContext};
use super::model::{Artifact, BaseModel, Component};
use super::text::{self, JString};

/// OpenNLP's `ner-default-features.xml`: the descriptor of a model that
/// has none.
const DEFAULT_FEATURES: &[u8] = include_bytes!("../resources/ner-default-features.xml");

/// The manifest property naming the sequence codec.
const SEQUENCE_CODEC: &str = "sequenceCodecImplName";

/// A loaded `TokenNameFinderModel`.
#[derive(Debug, Clone)]
pub struct NameFinderModel {
    base: Arc<BaseModel>,
}

impl NameFinderModel {
    /// `new TokenNameFinderModel(InputStream)`.
    pub fn load(bytes: &[u8]) -> Result<NameFinderModel, FactoryError> {
        Ok(NameFinderModel {
            base: Arc::new(BaseModel::load(Component::NameFinder, bytes)?),
        })
    }

    /// The model package (its artifacts are the feature generators'
    /// resources).
    pub fn base(&self) -> &BaseModel {
        &self.base
    }

    /// `getFactory().createSequenceCodec()`.
    pub fn sequence_codec(&self) -> SequenceCodec {
        match self.base.manifest_property(SEQUENCE_CODEC) {
            Some("opennlp.tools.namefind.BilouCodec") => SequenceCodec::Bilou,
            _ => SequenceCodec::Bio,
        }
    }
}

/// A name: `opennlp.tools.util.Span` with its type and probability.
#[derive(Debug, Clone, PartialEq)]
pub struct NameSpan {
    /// The first token.
    pub start: usize,
    /// The token after the last.
    pub end: usize,
    /// The name type (`person` of `person-start`), if the outcome has one.
    pub kind: Option<JString>,
    /// The mean probability of its tokens' outcomes.
    pub prob: f64,
}

/// The two `SequenceCodec<String>`s.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceCodec {
    /// `BioCodec`: `start`, `cont` and `other` outcomes.
    Bio,
    /// `BilouCodec`: `start`, `cont`, `last`, `unit` and `other`.
    Bilou,
}

fn ends_with(s: &[u16], suffix: &str) -> bool {
    let suffix = text::units(suffix);
    s.ends_with(&suffix)
}

/// Java's line terminators, which `.` does not match.
fn is_line_terminator(c: u16) -> bool {
    matches!(c, 0x0A | 0x0D | 0x85 | 0x2028 | 0x2029)
}

/// `extractNameType(outcome)`: group 1 of `(.+)-\w+` matched against the
/// whole outcome -- what precedes the last `-` when an ASCII word follows
/// it.
// ARITH: `i` is the index of a unit of the outcome.
#[allow(clippy::arithmetic_side_effects)]
pub fn extract_name_type(outcome: &[u16]) -> Option<JString> {
    let i = outcome.iter().rposition(|&c| c == u16::from(b'-'))?;
    let (kind, word) = (&outcome[..i], &outcome[i + 1..]);
    let is_word = |c: &u16| u8::try_from(*c).is_ok_and(|b| b.is_ascii_alphanumeric() || b == b'_');
    if kind.is_empty() || kind.iter().any(|&c| is_line_terminator(c)) {
        return None;
    }
    if word.is_empty() || !word.iter().all(is_word) {
        return None;
    }
    Some(kind.to_vec())
}

impl SequenceCodec {
    /// `decode(outcomes)`: the name spans, without probabilities.
    // ARITH: token indices of the sequence.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn decode(self, c: &[&[u16]]) -> Vec<NameSpan> {
        let span = |start: usize, end: usize, outcome: &[u16]| NameSpan {
            start,
            end,
            kind: extract_name_type(outcome),
            prob: 0.0,
        };
        let mut spans = Vec::new();
        let mut open: Option<(usize, usize)> = None;
        match self {
            SequenceCodec::Bio => {
                for (li, tag) in c.iter().enumerate() {
                    if ends_with(tag, "start") {
                        if let Some((start, end)) = open {
                            spans.push(span(start, end, c[li - 1]));
                        }
                        open = Some((li, li + 1));
                    } else if ends_with(tag, "cont") {
                        if let Some((_, end)) = &mut open {
                            *end = li + 1;
                        }
                    } else if ends_with(tag, "other") {
                        if let Some((start, end)) = open.take() {
                            spans.push(span(start, end, c[li - 1]));
                        }
                    }
                }
                if let (Some((start, end)), Some(last)) = (open, c.last()) {
                    spans.push(span(start, end, last));
                }
            }
            SequenceCodec::Bilou => {
                for (li, tag) in c.iter().enumerate() {
                    if ends_with(tag, "start") {
                        open = Some((li, li + 1));
                    } else if ends_with(tag, "cont") {
                        if let Some((_, end)) = &mut open {
                            *end = li + 1;
                        }
                    } else if ends_with(tag, "last") {
                        if let Some((start, end)) = open.take() {
                            spans.push(span(start, end + 1, c[li - 1]));
                        }
                    } else if ends_with(tag, "unit") {
                        spans.push(span(li, li + 1, tag));
                    }
                }
            }
        }
        spans
    }

    /// The codec's `SequenceValidator`: `NameFinderSequenceValidator` or
    /// `BilouNameFinderSequenceValidator`.
    pub fn valid(self, prior: &[&[u16]], outcome: &[u16]) -> bool {
        let last = prior.last().copied();
        // a continuation of the previous name, of its type
        let continues = |last: &[u16]| -> bool {
            let previous = extract_name_type(last);
            let current = extract_name_type(outcome);
            match (previous, current) {
                (None, None) => true,
                (p, Some(c)) => p.is_some_and(|p| p == c),
                (Some(_), None) => false,
            }
        };
        match self {
            SequenceCodec::Bio => {
                if !ends_with(outcome, "cont") {
                    return true;
                }
                match last {
                    None => false,
                    Some(l) if ends_with(l, "other") => false,
                    Some(l) if ends_with(l, "cont") || ends_with(l, "start") => continues(l),
                    Some(_) => true,
                }
            }
            SequenceCodec::Bilou => {
                if ends_with(outcome, "cont") || ends_with(outcome, "last") {
                    match last {
                        None => return false,
                        Some(l) if ends_with(l, "other") || ends_with(l, "unit") => return false,
                        // after a last only a start, other or unit
                        Some(l) if ends_with(l, "last") => return false,
                        Some(l) if ends_with(l, "cont") || ends_with(l, "start") => {
                            let previous = extract_name_type(l);
                            let current = extract_name_type(outcome);
                            if previous.is_some() || current.is_some() {
                                return continues(l);
                            }
                        }
                        Some(_) => {}
                    }
                }
                if let Some(l) = last {
                    if ends_with(outcome, "start")
                        || ends_with(outcome, "other")
                        || ends_with(outcome, "unit")
                    {
                        return !ends_with(l, "start") && !ends_with(l, "cont");
                    }
                }
                true
            }
        }
    }
}

/// `NameFinderME`.
#[derive(Debug, Clone)]
pub struct NameFinder {
    model: Arc<MaxentModel>,
    codec: SequenceCodec,
    beam: BeamSearch,
    generator: Generator,
    sentence: u64,
    probs: Vec<f64>,
}

impl NameFinder {
    /// `new NameFinderME(model)`: the codec, the beam size from the
    /// manifest, then the context generator from the descriptor.
    pub fn new(model: &NameFinderModel) -> Result<NameFinder, AnalysisError> {
        let codec = model.sequence_codec();
        let beam_size = model.base.beam_size(3)?;
        let descriptor = match model.base.artifact("generator.featuregen") {
            None => DEFAULT_FEATURES,
            Some(Artifact::Bytes(b)) => b.as_slice(),
            Some(other) => {
                return Err(AnalysisError::IllegalState(format!(
                    "ClassCastException: the generator descriptor is a {}, not a byte[]",
                    other.describe()
                )))
            }
        };
        let generator = featuregen::create_with(descriptor, &*model.base, Owner::NameFinder)?;
        Ok(NameFinder {
            model: model.base.maxent_model(),
            codec,
            beam: BeamSearch::new(beam_size),
            generator,
            sentence: 0,
            probs: Vec::new(),
        })
    }

    /// `find(tokens)`: the names of a sentence; its outcomes then feed the
    /// adaptive generators.
    // ARITH: a span's tokens lie within the sentence, its end past its start.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn find(&mut self, tokens: &[JString]) -> Result<Vec<NameSpan>, AnalysisError> {
        self.sentence = self.sentence.wrapping_add(1);
        let model = Arc::clone(&self.model);
        let beam = self.beam.clone();
        let mut ctx = NameContext {
            finder: self,
            tokens,
            error: None,
        };
        let result = beam.best_sequence_with_probs(&model, tokens.len(), &mut ctx);
        if let Some(e) = ctx.error.take() {
            return Err(e);
        }
        let (best, probs) = result?;
        let outcomes: Vec<JString> = best
            .iter()
            .map(|&o| model.outcome(o as usize).to_vec())
            .collect();
        self.generator.update_adaptive_data(tokens, &outcomes);
        let refs: Vec<&[u16]> = outcomes.iter().map(Vec::as_slice).collect();
        let mut spans = self.codec.decode(&refs);
        for span in &mut spans {
            // `probs(spans)`: the mean over the span's tokens
            let sum: f64 = probs[span.start..span.end].iter().fold(0.0, |p, q| p + q);
            span.prob = sum / (span.end - span.start) as f64;
        }
        self.probs = probs;
        Ok(spans)
    }

    /// `clearAdaptiveData()`: what the generators learnt forgotten (call it
    /// between documents).
    pub fn clear_adaptive_data(&mut self) {
        self.generator.clear_adaptive_data();
    }

    /// `probs()`: each token's outcome probability in the last sentence.
    pub fn probs(&self) -> &[f64] {
        &self.probs
    }
}

/// One `find` call's context generator and validator.
struct NameContext<'a> {
    finder: &'a mut NameFinder,
    tokens: &'a [JString],
    error: Option<AnalysisError>,
}

impl SequenceContext for NameContext<'_> {
    /// `DefaultNameContextGenerator.getContext`: the generators' features,
    /// then the previous outcome (`po`), with the word (`pow`) and its
    /// class (`powf`), and the one before (`ppo`).
    // ARITH: `index` is a position of the sentence, guarded before `- 1`/`- 2`.
    #[allow(clippy::arithmetic_side_effects)]
    fn context(
        &mut self,
        index: usize,
        prior: &[u32],
        outcomes: &[JString],
        features: &mut Vec<JString>,
    ) {
        // The previous outcomes by id (no copy of the sequence per call).
        let preds = Prior::Ids {
            ids: prior,
            outcomes,
        };
        let sentence = self.finder.sentence;
        if let Err(e) = self.finder.generator.create_features_with(
            features,
            self.tokens,
            index,
            preds,
            sentence,
        ) {
            self.error.get_or_insert(e);
        }
        let other = text::units("other");
        let pred = |i: usize| &outcomes[prior[i] as usize][..];
        let po = if index > 0 { pred(index - 1) } else { &other };
        let ppo = if index > 1 { pred(index - 2) } else { &other };
        let token = &self.tokens[index];
        let mut f = |parts: &[&[u16]]| features.push(text::concat(parts));
        f(&[&text::units("po="), po]);
        f(&[&text::units("pow="), po, &[u16::from(b',')], token]);
        f(&[
            &text::units("powf="),
            po,
            &[u16::from(b',')],
            &text::units(featuregen::token_feature(token)),
        ]);
        f(&[&text::units("ppo="), ppo]);
    }

    fn valid(&self, _i: usize, prior: &[u32], outcome: u32, outcomes: &[JString]) -> bool {
        let prior: Vec<&[u16]> = prior
            .iter()
            .map(|&o| outcomes[o as usize].as_slice())
            .collect();
        self.finder.codec.valid(&prior, &outcomes[outcome as usize])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u(s: &str) -> JString {
        text::units(s)
    }

    #[test]
    fn name_types() {
        assert_eq!(extract_name_type(&u("person-start")), Some(u("person")));
        assert_eq!(extract_name_type(&u("a-b-cont")), Some(u("a-b")));
        assert_eq!(extract_name_type(&u("start")), None);
        assert_eq!(extract_name_type(&u("-start")), None);
        assert_eq!(extract_name_type(&u("x-")), None);
        assert_eq!(extract_name_type(&u("x-st.art")), None);
        assert_eq!(extract_name_type(&u("x\ny-start")), None);
        assert_eq!(extract_name_type(&u("é-start_1")), Some(u("é")));
    }

    fn decode(codec: SequenceCodec, tags: &str) -> Vec<(usize, usize, Option<String>)> {
        let tags: Vec<JString> = tags.split(' ').map(u).collect();
        let refs: Vec<&[u16]> = tags.iter().map(Vec::as_slice).collect();
        codec
            .decode(&refs)
            .into_iter()
            .map(|s| (s.start, s.end, s.kind.map(|k| text::to_string(&k))))
            .collect()
    }

    #[test]
    fn codecs_decode() {
        let p = |s: usize, e: usize, k: &str| (s, e, Some(k.to_string()));
        assert_eq!(
            decode(
                SequenceCodec::Bio,
                "p-start p-cont other l-start p-start x-cont"
            ),
            vec![p(0, 2, "p"), p(3, 4, "l"), p(4, 6, "x")]
        );
        assert_eq!(
            decode(SequenceCodec::Bio, "cont other start"),
            vec![(2, 3, None)]
        );
        assert_eq!(
            decode(
                SequenceCodec::Bilou,
                "p-start p-cont p-last other l-unit p-start l-last last cont q-start x"
            ),
            vec![p(0, 3, "p"), p(4, 5, "l"), p(5, 7, "p")]
        );
    }

    fn valid(codec: SequenceCodec, prior: &str, outcome: &str) -> bool {
        let prior: Vec<JString> = prior.split(' ').filter(|s| !s.is_empty()).map(u).collect();
        let refs: Vec<&[u16]> = prior.iter().map(Vec::as_slice).collect();
        codec.valid(&refs, &u(outcome))
    }

    #[test]
    fn validators() {
        use SequenceCodec::{Bilou, Bio};
        assert!(valid(Bio, "", "p-start") && !valid(Bio, "", "p-cont"));
        assert!(!valid(Bio, "other", "p-cont"));
        assert!(valid(Bio, "p-start", "p-cont") && !valid(Bio, "p-start", "l-cont"));
        assert!(!valid(Bio, "start", "p-cont") && !valid(Bio, "p-start", "cont"));
        assert!(valid(Bio, "start", "cont") && valid(Bio, "x", "p-cont"));
        assert!(!valid(Bilou, "", "p-cont") && !valid(Bilou, "", "p-last"));
        assert!(!valid(Bilou, "other", "p-last") && !valid(Bilou, "p-unit", "p-cont"));
        assert!(!valid(Bilou, "p-last", "p-cont"));
        assert!(valid(Bilou, "p-start", "p-last") && !valid(Bilou, "p-cont", "l-last"));
        assert!(valid(Bilou, "start", "cont") && valid(Bilou, "x", "p-cont"));
        assert!(!valid(Bilou, "p-start", "other") && !valid(Bilou, "p-cont", "l-unit"));
        assert!(valid(Bilou, "p-last", "l-start") && valid(Bilou, "", "other"));
        assert!(valid(Bilou, "p-last", "p-x"));
    }
}
