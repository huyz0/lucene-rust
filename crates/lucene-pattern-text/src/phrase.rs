//! A phrase on a pattern-text field: the conjunction of its terms on the
//! docs-only message field, each candidate verified against its message.
//!
//! Most candidates are decided by their template alone, once per template
//! ordinal and segment:
//!
//! - a template with no variables is the whole message, so the phrase is
//!   checked once against its tokens;
//! - a template whose static text between two variables holds the phrase
//!   matches every document that uses it.
//!
//! Only the rest read the stored variables, rebuild the message and check it.
//! That is what keeps a phrase as fast as a positional phrase on logs, where
//! most phrases sit in the static part of a message.

use std::collections::HashMap;
use std::sync::Arc;

use lucene_codecs::stored_fields::{StoredFieldVisitor, VisitStatus};
use lucene_search::directory_reader::SegmentReader;
use lucene_search::function::FunctionMatchQuery;
use lucene_search::query::{BooleanQuery, Clause, TermQuery};
use lucene_search::reader::{LeafReader, SortedDocValues};
use lucene_search::values_source::{
    BoxDoubleValues, DoubleValues, DoubleValuesSource, ValuesContext,
};
use lucene_search::{Error, Result};

use crate::fields::{ARGS_SUFFIX, TEMPLATE_SUFFIX};
use crate::format::{decode_args, join, VARIABLE};

/// The field's analyzer, as the verifier runs it: `text`'s tokens appended
/// to `out` as `(position, term)`, positions from 0 in token order (so a
/// removed stopword leaves a gap, as in the indexed field).
pub type Tokenizer = dyn Fn(&str, &mut Vec<(i32, String)>) + Send + Sync;

/// A phrase's analyzed terms with their positions relative to the first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhraseTokens {
    terms: Vec<(i32, String)>,
}

impl PhraseTokens {
    /// `tokens` as the analyzer produced them (any first position).
    pub fn new(tokens: Vec<(i32, String)>) -> Self {
        let base = tokens.first().map_or(0, |t| t.0);
        Self {
            terms: tokens
                .into_iter()
                .map(|(p, t)| (p.saturating_sub(base), t))
                .collect(),
        }
    }

    /// The terms, in phrase order.
    pub fn terms(&self) -> impl Iterator<Item = &str> {
        self.terms.iter().map(|(_, t)| t.as_str())
    }

    /// Whether `tokens` (positions ascending) hold the phrase at some start.
    pub fn found_in(&self, tokens: &[(i32, String)]) -> bool {
        let Some((_, first)) = self.terms.first() else {
            return true;
        };
        tokens.iter().any(|(start, t)| {
            t == first
                && self.terms.iter().skip(1).all(|(rel, want)| {
                    let at = start.saturating_add(*rel);
                    let lo = tokens.partition_point(|(p, _)| *p < at);
                    tokens[lo..]
                        .iter()
                        .take_while(|(p, _)| *p == at)
                        .any(|(_, have)| have == want)
                })
        })
    }
}

/// `field:"phrase"` on a pattern-text field: the phrase's terms as `MUST`
/// clauses (scored as `match_only_text` scores them) and a
/// [`PatternPhraseSource`] filter. A phrase of one term is that term.
pub fn pattern_phrase_query(
    field: &str,
    phrase: PhraseTokens,
    tokenizer: Arc<Tokenizer>,
) -> Clause {
    let mut q = BooleanQuery::new();
    for t in phrase.terms() {
        q.must
            .push(Clause::Term(TermQuery::new(field, t.as_bytes())));
    }
    if phrase.terms.len() > 1 {
        let source = PatternPhraseSource::new(field, phrase, tokenizer);
        q.filter
            .push(Clause::from(FunctionMatchQuery::with_match_cost(
                Arc::new(source),
                Arc::new(|v: f64| v != 0.0),
                MATCH_COST,
            )));
    }
    Clause::Boolean(Box::new(q))
}

/// The verifier's `TwoPhaseIterator.matchCost()`: a doc-values read, and for
/// some documents a stored-fields read and an analysis.
const MATCH_COST: f32 = 100.0;

/// 1 for a document whose message holds the phrase, 0 otherwise.
pub struct PatternPhraseSource {
    field: String,
    phrase: PhraseTokens,
    tokenizer: Arc<Tokenizer>,
}

impl std::fmt::Debug for PatternPhraseSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.describe())
    }
}

impl PatternPhraseSource {
    /// The verifier of `phrase` on pattern-text field `field`.
    pub fn new(field: impl Into<String>, phrase: PhraseTokens, tokenizer: Arc<Tokenizer>) -> Self {
        Self {
            field: field.into(),
            phrase,
            tokenizer,
        }
    }

    /// What `template` alone says: `Some` when it settles every document
    /// that uses it.
    fn template_verdict(&self, template: &str, tokens: &mut Vec<(i32, String)>) -> Option<bool> {
        for part in template.split(VARIABLE) {
            tokens.clear();
            (self.tokenizer)(part, tokens);
            if self.phrase.found_in(tokens) {
                return Some(true);
            }
        }
        // Without slots the one part is the whole message.
        (!template.contains(VARIABLE)).then_some(false)
    }
}

impl DoubleValuesSource for PatternPhraseSource {
    fn get_values<'c>(
        &self,
        ctx: &ValuesContext<'c>,
        leaf: usize,
        _scores: Option<BoxDoubleValues<'c>>,
    ) -> Result<BoxDoubleValues<'c>> {
        let reader = ctx.leaf_reader(leaf)?;
        let templates = reader.sorted_doc_values(&format!("{}{TEMPLATE_SUFFIX}", self.field))?;
        let args_number = reader
            .field_infos()
            .field_by_name(&format!("{}{ARGS_SUFFIX}", self.field))
            .map(|f| f.number);
        Ok(Box::new(Values {
            source: self.clone_parts(),
            reader,
            templates,
            args_number,
            verdicts: HashMap::new(),
            tokens: Vec::new(),
            value: 0.0,
        }))
    }

    fn needs_scores(&self) -> bool {
        false
    }

    fn is_cacheable(&self, _ctx: &ValuesContext<'_>, _leaf: usize) -> bool {
        true
    }

    fn describe(&self) -> String {
        let terms: Vec<&str> = self.phrase.terms().collect();
        format!("pattern_phrase({}:\"{}\")", self.field, terms.join(" "))
    }

    fn equality_key(&self) -> String {
        let terms: Vec<String> = self
            .phrase
            .terms
            .iter()
            .map(|(p, t)| format!("{p}:{t}"))
            .collect();
        format!("pattern_phrase({}:{})", self.field, terms.join(","))
    }
}

impl PatternPhraseSource {
    fn clone_parts(&self) -> Arc<PatternPhraseSource> {
        Arc::new(PatternPhraseSource {
            field: self.field.clone(),
            phrase: self.phrase.clone(),
            tokenizer: Arc::clone(&self.tokenizer),
        })
    }
}

struct Values<'c> {
    source: Arc<PatternPhraseSource>,
    reader: &'c SegmentReader,
    templates: Option<Box<dyn SortedDocValues + 'c>>,
    args_number: Option<i32>,
    /// Per template ordinal: `Some` when the template settles it.
    verdicts: HashMap<i32, Option<bool>>,
    tokens: Vec<(i32, String)>,
    value: f64,
}

/// Collects one stored binary field's bytes.
struct ArgsVisitor {
    number: i32,
    bytes: Option<Vec<u8>>,
}

impl StoredFieldVisitor for ArgsVisitor {
    fn needs_field(
        &mut self,
        field_number: i32,
    ) -> lucene_codecs::stored_fields::Result<VisitStatus> {
        Ok(if self.bytes.is_some() {
            VisitStatus::Stop
        } else if field_number == self.number {
            VisitStatus::Yes
        } else {
            VisitStatus::No
        })
    }

    fn binary_field(
        &mut self,
        _field_number: i32,
        value: &[u8],
    ) -> lucene_codecs::stored_fields::Result<()> {
        self.bytes = Some(value.to_vec());
        Ok(())
    }
}

fn corrupt(e: crate::Error) -> Error {
    Error::IllegalState(e.to_string())
}

impl Values<'_> {
    fn verify(&mut self, doc: i32, template: &str) -> Result<bool> {
        let mut args = Vec::new();
        let mut visitor;
        if let Some(number) = self.args_number {
            visitor = ArgsVisitor {
                number,
                bytes: None,
            };
            let _ = self.reader.visit_stored_document(doc, &mut visitor)?;
            if let Some(b) = &visitor.bytes {
                decode_args(b, &mut args).map_err(corrupt)?;
            }
        }
        let message = join(template, &args).map_err(corrupt)?;
        self.tokens.clear();
        (self.source.tokenizer)(&message, &mut self.tokens);
        Ok(self.source.phrase.found_in(&self.tokens))
    }
}

impl DoubleValues for Values<'_> {
    fn advance_exact(&mut self, doc: i32) -> Result<bool> {
        let Some(templates) = self.templates.as_mut() else {
            return Ok(false);
        };
        if !templates.advance_exact(doc)? {
            return Ok(false);
        }
        let ord = templates.ord_value();
        let known = self.verdicts.get(&ord).copied();
        let (verdict, template) = match known {
            Some(Some(v)) => (Some(v), None),
            _ => {
                let bytes = templates.lookup_ord(ord)?;
                let template = String::from_utf8(bytes).map_err(|_| {
                    Error::IllegalState("pattern_text template is not UTF-8".into())
                })?;
                let verdict = match known {
                    Some(v) => v,
                    None => {
                        let mut tokens = std::mem::take(&mut self.tokens);
                        let v = self.source.template_verdict(&template, &mut tokens);
                        self.tokens = tokens;
                        self.verdicts.insert(ord, v);
                        v
                    }
                };
                (verdict, Some(template))
            }
        };
        let matched = match (verdict, template) {
            (Some(v), _) => v,
            (None, Some(t)) => self.verify(doc, &t)?,
            (None, None) => false,
        };
        self.value = if matched { 1.0 } else { 0.0 };
        Ok(true)
    }

    fn double_value(&mut self) -> Result<f64> {
        Ok(self.value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(s: &str) -> Vec<(i32, String)> {
        s.split_whitespace()
            .enumerate()
            .map(|(i, t)| (i as i32, t.to_lowercase()))
            .collect()
    }

    fn tokenizer() -> Arc<Tokenizer> {
        Arc::new(|text: &str, out: &mut Vec<(i32, String)>| out.extend(toks(text)))
    }

    #[test]
    fn phrase_positions_must_line_up() {
        let p = PhraseTokens::new(vec![(3, "failed".into()), (4, "password".into())]);
        assert!(p.found_in(&toks("x failed password for root")));
        assert!(!p.found_in(&toks("password failed")));
        assert!(!p.found_in(&toks("failed x password")));
        let gap = PhraseTokens::new(vec![(0, "a".into()), (2, "c".into())]);
        assert!(gap.found_in(&toks("a b c")));
        assert!(!gap.found_in(&toks("a c")));
        assert!(PhraseTokens::new(vec![]).found_in(&[]));
    }

    #[test]
    fn templates_settle_static_phrases_and_defer_the_rest() {
        let p = PhraseTokens::new(toks("connection closed by"));
        let s = PatternPhraseSource::new("m", p, tokenizer());
        let mut t = Vec::new();
        assert_eq!(
            s.template_verdict("\u{2} Connection closed by \u{2}", &mut t),
            Some(true)
        );
        assert_eq!(
            s.template_verdict("connection closed by peer", &mut t),
            Some(true)
        );
        assert_eq!(
            s.template_verdict("connection reset by peer", &mut t),
            Some(false)
        );
        // the phrase crosses a slot: only the message can tell
        assert_eq!(s.template_verdict("connection \u{2} by", &mut t), None);
        assert_eq!(s.template_verdict("\u{2}", &mut t), None);
        assert!(s.describe().contains("connection closed by"));
        assert!(s.equality_key().contains("1:closed"));
    }
}
