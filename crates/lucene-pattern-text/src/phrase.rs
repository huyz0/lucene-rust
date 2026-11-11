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
    doc_values_cacheable, BoxDoubleValues, DoubleValues, DoubleValuesSource, ValuesContext,
};
use lucene_search::{Error, Result};

use crate::fields::{ARGS_SUFFIX, TEMPLATE_SUFFIX};
use crate::format::{decode_args, join, VARIABLE};

/// The field's analyzer, as the verifier runs it: `text`'s tokens appended
/// to `out` as `(position, term)`, positions non-decreasing (only relative
/// positions matter, so a removed stopword leaves the same gap as in the
/// indexed field).
///
/// The template shortcut tokenizes the static text between two variables on
/// its own, so it needs an analyzer that never joins tokens across a space:
/// the tokens of part of a message, between two spaces, must be the tokens
/// the whole message has there. Standard, whitespace, simple and pattern
/// tokenizers with per-token filters (lowercase, stop, stemmers, ASCII
/// folding) meet this; shingles, word-delimiter catenation or a keyword
/// tokenizer do not, and need [`PatternPhraseSource::whole_messages_only`].
pub type Tokenizer = dyn Fn(&str, &mut Vec<(i32, String)>) + Send + Sync;

/// A phrase's analyzed terms by position, relative to the first: each
/// position matches any of its terms (stacked synonyms), as Lucene's
/// `MultiPhraseQuery` does.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhraseTokens {
    positions: Vec<(i32, Vec<String>)>,
}

impl PhraseTokens {
    /// `tokens` as the analyzer produced them (positions non-decreasing, any
    /// first position).
    pub fn new(tokens: Vec<(i32, String)>) -> Self {
        let base = tokens.first().map_or(0, |t| t.0);
        let mut positions: Vec<(i32, Vec<String>)> = Vec::new();
        for (p, t) in tokens {
            let rel = p.saturating_sub(base);
            match positions.last_mut() {
                Some((last, terms)) if *last == rel => terms.push(t),
                _ => positions.push((rel, vec![t])),
            }
        }
        Self { positions }
    }

    /// The terms at each position, in phrase order.
    pub fn positions(&self) -> impl Iterator<Item = &[String]> {
        self.positions.iter().map(|(_, t)| t.as_slice())
    }

    /// Whether `tokens` (positions non-decreasing) hold the phrase at some
    /// start.
    pub fn found_in(&self, tokens: &[(i32, String)]) -> bool {
        let Some((_, first)) = self.positions.first() else {
            return true;
        };
        let at = |pos: i32, want: &[String]| {
            let lo = tokens.partition_point(|(p, _)| *p < pos);
            tokens[lo..]
                .iter()
                .take_while(|(p, _)| *p == pos)
                .any(|(_, have)| want.contains(have))
        };
        tokens.iter().any(|(start, t)| {
            first.contains(t)
                && self
                    .positions
                    .iter()
                    .skip(1)
                    .all(|(rel, want)| at(start.saturating_add(*rel), want))
        })
    }
}

/// `field:"phrase"` on a pattern-text field: each position's terms as a
/// `FILTER` clause and a [`PatternPhraseSource`] filter, so every hit scores
/// 0 as a `match_only_text` phrase does (OpenSearch's `SourceFieldMatchQuery`
/// over a filter-only conjunction). A phrase of one position needs no check.
pub fn pattern_phrase_query(
    field: &str,
    phrase: PhraseTokens,
    verifier: PatternPhraseSource,
) -> Clause {
    debug_assert_eq!(verifier.field, field);
    let mut q = BooleanQuery::new();
    for terms in phrase.positions() {
        let mut leg: Vec<Clause> = terms
            .iter()
            .map(|t| Clause::Term(TermQuery::new(field, t.as_bytes())))
            .collect();
        q.filter.push(if leg.len() == 1 {
            leg.remove(0)
        } else {
            Clause::Boolean(Box::new(BooleanQuery::new().with_should(leg)))
        });
    }
    if phrase.positions.len() > 1 {
        q.filter
            .push(Clause::from(FunctionMatchQuery::with_match_cost(
                Arc::new(verifier),
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
    analyzer_key: String,
    template_parts: bool,
}

impl std::fmt::Debug for PatternPhraseSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.describe())
    }
}

impl PatternPhraseSource {
    /// The verifier of `phrase` on pattern-text field `field`, analyzing
    /// messages with `tokenizer`. `analyzer_key` names the analyzer, so two
    /// verifiers with the same phrase and different analyzers are not equal
    /// (and not one query-cache entry).
    pub fn new(
        field: impl Into<String>,
        phrase: PhraseTokens,
        tokenizer: Arc<Tokenizer>,
        analyzer_key: impl Into<String>,
    ) -> Self {
        Self {
            field: field.into(),
            phrase,
            tokenizer,
            analyzer_key: analyzer_key.into(),
            template_parts: true,
        }
    }

    /// Never decides from a template's static parts alone, for an analyzer
    /// that can join tokens across spaces (see [`Tokenizer`]). A template
    /// without variables is still decided once.
    #[must_use]
    pub fn whole_messages_only(mut self) -> Self {
        self.template_parts = false;
        self
    }

    /// What `template` alone says: `Some` when it settles every document
    /// that uses it.
    fn template_verdict(&self, template: &str, tokens: &mut Vec<(i32, String)>) -> Option<bool> {
        if !template.contains(VARIABLE) {
            tokens.clear();
            (self.tokenizer)(template, tokens);
            return Some(self.phrase.found_in(tokens));
        }
        if self.template_parts {
            for part in template.split(VARIABLE) {
                tokens.clear();
                (self.tokenizer)(part, tokens);
                if self.phrase.found_in(tokens) {
                    return Some(true);
                }
            }
        }
        None
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
            visitor: ArgsVisitor {
                number: -1,
                bytes: Vec::new(),
                found: false,
            },
            message: String::new(),
            value: 0.0,
        }))
    }

    fn needs_scores(&self) -> bool {
        false
    }

    fn is_cacheable(&self, ctx: &ValuesContext<'_>, leaf: usize) -> bool {
        doc_values_cacheable(ctx, leaf, &format!("{}{TEMPLATE_SUFFIX}", self.field))
    }

    fn describe(&self) -> String {
        let terms: Vec<String> = self.phrase.positions().map(|t| t.join("|")).collect();
        format!("pattern_phrase({}:\"{}\")", self.field, terms.join(" "))
    }

    fn equality_key(&self) -> String {
        let terms: Vec<String> = self
            .phrase
            .positions
            .iter()
            .map(|(p, t)| format!("{p}:{}", t.join("|")))
            .collect();
        format!(
            "pattern_phrase({}:{},{},{})",
            self.field,
            terms.join(","),
            self.analyzer_key,
            self.template_parts
        )
    }
}

impl PatternPhraseSource {
    fn clone_parts(&self) -> Arc<PatternPhraseSource> {
        Arc::new(PatternPhraseSource {
            field: self.field.clone(),
            phrase: self.phrase.clone(),
            tokenizer: Arc::clone(&self.tokenizer),
            analyzer_key: self.analyzer_key.clone(),
            template_parts: self.template_parts,
        })
    }
}

/// What one template ordinal says about its documents.
enum Verdict {
    /// Every document matches, or none does.
    Settled(bool),
    /// Each document's message decides; the template, decoded once.
    Check(String),
}

struct Values<'c> {
    source: Arc<PatternPhraseSource>,
    reader: &'c SegmentReader,
    templates: Option<Box<dyn SortedDocValues + 'c>>,
    args_number: Option<i32>,
    verdicts: HashMap<i32, Verdict>,
    tokens: Vec<(i32, String)>,
    visitor: ArgsVisitor,
    message: String,
    value: f64,
}

/// Collects one stored binary field's bytes into a reused buffer.
struct ArgsVisitor {
    number: i32,
    bytes: Vec<u8>,
    found: bool,
}

impl StoredFieldVisitor for ArgsVisitor {
    fn needs_field(
        &mut self,
        field_number: i32,
    ) -> lucene_codecs::stored_fields::Result<VisitStatus> {
        Ok(if self.found {
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
        self.bytes.clear();
        self.bytes.extend_from_slice(value);
        self.found = true;
        Ok(())
    }
}

fn corrupt(e: crate::Error) -> Error {
    Error::IllegalState(e.to_string())
}

/// Whether document `doc`'s message holds the phrase: its variables read
/// into `visitor`, the message rebuilt into `message` and analyzed.
#[allow(clippy::too_many_arguments)]
fn verify(
    source: &PatternPhraseSource,
    reader: &SegmentReader,
    args_number: Option<i32>,
    visitor: &mut ArgsVisitor,
    message: &mut String,
    tokens: &mut Vec<(i32, String)>,
    doc: i32,
    template: &str,
) -> Result<bool> {
    let mut args = Vec::new();
    visitor.found = false;
    if let Some(number) = args_number {
        visitor.number = number;
        let _ = reader.visit_stored_document(doc, visitor)?;
        if visitor.found {
            decode_args(&visitor.bytes, &mut args).map_err(corrupt)?;
        }
    }
    *message = join(template, &args).map_err(corrupt)?;
    tokens.clear();
    (source.tokenizer)(message, tokens);
    Ok(source.phrase.found_in(tokens))
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
        if !self.verdicts.contains_key(&ord) {
            let template = String::from_utf8(templates.lookup_ord(ord)?)
                .map_err(|_| Error::IllegalState("pattern_text template is not UTF-8".into()))?;
            let v = match self.source.template_verdict(&template, &mut self.tokens) {
                Some(v) => Verdict::Settled(v),
                None => Verdict::Check(template),
            };
            self.verdicts.insert(ord, v);
        }
        let matched = match self.verdicts.get(&ord) {
            Some(Verdict::Settled(v)) => *v,
            Some(Verdict::Check(template)) => verify(
                &self.source,
                self.reader,
                self.args_number,
                &mut self.visitor,
                &mut self.message,
                &mut self.tokens,
                doc,
                template,
            )?,
            None => false,
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
        let s = PatternPhraseSource::new("m", p.clone(), tokenizer(), "ws");
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
        assert!(format!("{s:?}").starts_with("pattern_phrase(m:"));
        let other = PatternPhraseSource::new("m", p.clone(), tokenizer(), "std");
        assert_ne!(s.equality_key(), other.equality_key());
        let whole = PatternPhraseSource::new("m", p, tokenizer(), "ws").whole_messages_only();
        assert_eq!(
            whole.template_verdict("\u{2} connection closed by \u{2}", &mut t),
            None
        );
        assert_eq!(
            whole.template_verdict("connection closed by peer", &mut t),
            Some(true)
        );
    }

    #[test]
    fn stacked_terms_match_any_of_them() {
        let p = PhraseTokens::new(vec![
            (0, "fast".into()),
            (0, "quick".into()),
            (1, "fox".into()),
        ]);
        assert_eq!(p.positions().count(), 2);
        assert!(p.found_in(&toks("quick fox")));
        assert!(p.found_in(&toks("fast fox")));
        assert!(!p.found_in(&toks("slow fox")));
    }
}
