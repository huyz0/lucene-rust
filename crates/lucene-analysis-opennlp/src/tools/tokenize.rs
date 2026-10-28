//! `opennlp.tools.tokenize`: `TokenizerME.tokenizePos` over a loaded
//! `TokenizerModel` -- white-space tokens, each of two or more characters
//! that the alphanumeric pattern does not take whole split where the maxent
//! model says `T` between two characters (`DefaultTokenContextGenerator`
//! describing each position), an abbreviation kept whole.

use std::sync::Arc;

use lucene_analysis::factory::FactoryError;
use lucene_analysis::util::JavaPattern;
use lucene_analysis::AnalysisError;

use super::dictionary::Dictionary;
use super::ml::MaxentModel;
use super::model::{BaseModel, Component};
use super::sentdetect::Span;
use super::text::{self, is_whitespace, JString};

/// `opennlp.tools.tokenize.lang.Factory.DEFAULT_ALPHANUMERIC`.
const DEFAULT_ALPHANUMERIC: &str = "^[A-Za-z0-9]+$";

/// A loaded `TokenizerModel`.
#[derive(Debug, Clone)]
pub struct TokenizerModel {
    model: Arc<MaxentModel>,
    abbreviations: Option<Arc<Dictionary>>,
    use_alphanumeric: bool,
    alphanumeric: String,
}

impl TokenizerModel {
    /// `new TokenizerModel(InputStream)`.
    pub fn load(bytes: &[u8]) -> Result<TokenizerModel, FactoryError> {
        let base = BaseModel::load(Component::Tokenizer, bytes)?;
        Ok(TokenizerModel {
            model: base.maxent_model(),
            abbreviations: base.dictionary("abbreviations.dictionary"),
            use_alphanumeric: base
                .manifest_property("useAlphaNumericOptimization")
                .is_some_and(text::parse_boolean),
            // getAlphaNumericPattern: the manifest's, else (no language
            // code on a loaded model) the default
            alphanumeric: base
                .manifest_property("alphaNumericPattern")
                .unwrap_or(DEFAULT_ALPHANUMERIC)
                .to_string(),
        })
    }
}

/// `TokenizerME`.
#[derive(Debug, Clone)]
pub struct Tokenizer {
    model: TokenizerModel,
    alphanumeric: JavaPattern,
    /// Per outcome, whether its name is `TokenizerME.SPLIT` (Java compares the best
    /// outcome's *name*: a model without the outcome never splits, and a
    /// repeated name splits on either index).
    split: Vec<bool>,
    probs: Vec<f64>,
    features: Vec<JString>,
}

impl Tokenizer {
    /// `new TokenizerME(model)`: the alphanumeric pattern compiled (Java's
    /// `PatternSyntaxException` for a bad one).
    pub fn new(model: TokenizerModel) -> Result<Self, AnalysisError> {
        let alphanumeric = JavaPattern::compile(&model.alphanumeric)?;
        let split = (0..model.model.num_outcomes())
            .map(|i| model.model.outcome_name(i) == "T")
            .collect();
        let probs = vec![0.0; model.model.num_outcomes()];
        Ok(Tokenizer {
            model,
            alphanumeric,
            split,
            probs,
            features: Vec::new(),
        })
    }

    /// `tokenizePos(d)`: the tokens' spans.
    // ARITH: positions within the text; `j` grows by at most the token's dot count.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn tokenize_pos(&mut self, d: &[u16]) -> Vec<Span> {
        let mut tokens = Vec::new();
        for ws in whitespace_tokens(d) {
            let tok = &d[ws.start..ws.end];
            if tok.len() < 2
                || (self.model.use_alphanumeric && self.alphanumeric.matches(&text::to_string(tok)))
            {
                tokens.push(ws);
                continue;
            }
            let mut start = ws.start;
            let end = ws.end;
            let mut j = ws.start + 1;
            while j < end {
                self.features.clear();
                context(
                    tok,
                    j - ws.start,
                    self.model.abbreviations.as_deref(),
                    &mut self.features,
                );
                self.model.model.eval(&self.features, &mut self.probs);
                if self.split[MaxentModel::best_outcome(&self.probs)] {
                    if self.is_acceptable_abbreviation(tok) {
                        tokens.push(span(start, end));
                        let dots = tok.iter().filter(|&&c| c == u16::from(b'.')).count();
                        j += dots;
                        start = j + 1;
                    } else {
                        tokens.push(span(start, j));
                        start = j;
                    }
                }
                j += 1;
            }
            if start < end {
                tokens.push(span(start, end));
            }
        }
        tokens
    }

    /// `isAcceptableAbbreviation(s)`.
    fn is_acceptable_abbreviation(&self, s: &[u16]) -> bool {
        self.model
            .abbreviations
            .as_ref()
            .is_some_and(|d| d.contains_string(s))
    }
}

/// `new Span(s, e)`: never a start past the end here (Java's constructor
/// would throw) -- an abbreviation's split moves `start` to `j + 1` with
/// `j` advanced past its dots, and the next span either starts there and
/// ends at the token's end or at a later `j`.
fn span(start: usize, end: usize) -> Span {
    Span { start, end }
}

/// `WhitespaceTokenizer.tokenizePos` (new lines not kept).
fn whitespace_tokens(d: &[u16]) -> Vec<Span> {
    let mut tokens = Vec::new();
    let mut start = None;
    for (i, &c) in d.iter().enumerate() {
        if is_whitespace(c) {
            if let Some(s) = start.take() {
                tokens.push(Span { start: s, end: i });
            }
        } else if start.is_none() {
            start = Some(i);
        }
    }
    if let Some(s) = start {
        tokens.push(Span {
            start: s,
            end: d.len(),
        });
    }
    tokens
}

/// `DefaultTokenContextGenerator.createContext(sentence, index)`.
// ARITH: positions within the token (`index < len`).
#[allow(clippy::arithmetic_side_effects)]
fn context(
    sentence: &[u16],
    index: usize,
    abbreviations: Option<&Dictionary>,
    preds: &mut Vec<JString>,
) {
    let cat = |key: &str, rest: &[u16]| text::concat(&[&text::units(key), rest]);
    preds.push(cat("p=", &sentence[..index]));
    preds.push(cat("s=", &sentence[index..]));
    if index > 0 {
        add_char_preds("p1", sentence[index - 1], preds);
        if index > 1 {
            add_char_preds("p2", sentence[index - 2], preds);
            preds.push(cat("p21=", &sentence[index - 2..index]));
        } else {
            preds.push(text::units("p2=bok"));
        }
        preds.push(cat("p1f1=", &sentence[index - 1..=index]));
    } else {
        preds.push(text::units("p1=bok"));
    }
    add_char_preds("f1", sentence[index], preds);
    if index + 1 < sentence.len() {
        add_char_preds("f2", sentence[index + 1], preds);
        preds.push(cat("f12=", &sentence[index..index + 2]));
    } else {
        preds.push(text::units("f2=bok"));
    }
    if sentence[0] == u16::from(b'&') && sentence[sentence.len() - 1] == u16::from(b';') {
        preds.push(text::units("cc"));
    }
    if index == sentence.len() - 1 && abbreviations.is_some_and(|d| d.contains_string(sentence)) {
        preds.push(text::units("pabb"));
    }
}

/// `DefaultTokenContextGenerator.addCharPreds`.
fn add_char_preds(key: &str, c: u16, preds: &mut Vec<JString>) {
    let mut value = text::units(key);
    value.extend_from_slice(&[u16::from(b'='), c]);
    preds.push(value);
    let mut with = |suffix: &str| preds.push(text::units(&format!("{key}{suffix}")));
    if text::is_letter(c) {
        with("_alpha");
        if text::is_upper_case(c) {
            with("_caps");
        }
    } else if text::is_digit(c) {
        with("_num");
    } else if is_whitespace(c) {
        with("_ws");
    } else {
        match c {
            0x2E | 0x3F | 0x21 => with("_eos"),
            0x60 | 0x22 | 0x27 => with("_quote"),
            0x5B | 0x7B | 0x28 => with("_lp"),
            0x5D | 0x7D | 0x29 => with("_rp"),
            _ => {}
        }
    }
}
