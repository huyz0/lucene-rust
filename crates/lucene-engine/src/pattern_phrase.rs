//! The plugin's phrase query over a `pattern_text` field, as node `47` of the
//! query tree ([`crate::queries`]).
//!
//! The phrase's terms are a filter-only conjunction, and each candidate is
//! verified against its message as [`lucene_pattern_text::PatternPhraseSource`]
//! verifies it: once per template where the template settles it, otherwise by
//! rebuilding the message from the template and the stored variables and
//! running the field's index analyzer over it. Every hit scores 0, as the
//! plugin's Java `PatternPhraseQuery` and a `match_only_text` phrase score.
//!
//! The node, after its kind byte:
//!
//! | field | layout |
//! |---|---|
//! | `field` | string |
//! | analyzer | bytes: the index analyzer's spec ([`crate::analysis`]) |
//! | `template_parts` | `u8`: 1 to decide from a template's static text, 0 not to |
//! | `general` | `u8`: 0 for the exact phrase below; 1 for a phrase with slop or a prefix: its approximation (a query-tree node) and the phrase itself (a node), run over each message |
//! | positions (`general` 0) | `i32` count, at least 1, then per position `position: i32` (non-decreasing), `n: i32` (at least 1) and `n` terms as bytes |

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use lucene_analysis::TokenStream;
use lucene_pattern_text::{pattern_phrase_query, PatternPhraseSource, PhraseTokens, Tokenizer};
use lucene_search::query::Clause;

use crate::nodes::invalid;
use crate::query_blob::Cursor;
use crate::status::Status;

pub const NODE_PATTERN_PHRASE: u8 = 47;

/// At most this many terms in one phrase (the plugin's `MAX_PATTERN_TERMS`
/// keeps a longer one in Java).
pub const MAX_TERMS: usize = 1 << 16;

pub fn string(c: &mut Cursor<'_>) -> std::result::Result<String, Status> {
    Ok(std::str::from_utf8(c.bytes()?)
        .map_err(|_| Status::InvalidUtf8)?
        .to_string())
}

pub fn decode(
    c: &mut Cursor<'_>,
    depth: usize,
    nodes: &mut usize,
) -> std::result::Result<Clause, Status> {
    let field = string(c)?;
    let spec = c.bytes()?;
    let template_parts = match c.u8()? {
        0 => false,
        1 => true,
        f => {
            return Err(invalid(format!(
                "query tree: pattern phrase template flag {f} (expected 0..=1)"
            )))
        }
    };
    let general = c.u8()?;
    if general > 1 {
        return Err(invalid(format!(
            "query tree: pattern phrase general flag {general} (expected 0..=1)"
        )));
    }
    if general == 1 {
        let approximation = crate::query_blob::decode_node(c, depth.saturating_add(1), nodes)?;
        let positional = crate::query_blob::decode_node(c, depth.saturating_add(1), nodes)?;
        return general_phrase(field, spec, approximation, positional);
    }
    let count = c.length()?;
    if count == 0 {
        return Err(invalid("query tree: pattern phrase without terms"));
    }
    let mut tokens: Vec<(i32, String)> = Vec::new();
    let mut last = i32::MIN;
    for _ in 0..count {
        let pos = c.i32()?;
        if pos < last {
            return Err(invalid("query tree: pattern phrase positions out of order"));
        }
        last = pos;
        let n = c.length()?;
        if n == 0 {
            return Err(invalid("query tree: pattern phrase position without terms"));
        }
        if tokens.len().saturating_add(n) > MAX_TERMS {
            return Err(invalid(format!(
                "query tree: pattern phrase with more than {MAX_TERMS} terms"
            )));
        }
        for _ in 0..n {
            tokens.push((pos, string(c)?));
        }
    }
    let (tokenizer, key) = tokenizer_of(&field, spec)?;
    let phrase = PhraseTokens::new(tokens);
    let mut verifier = PatternPhraseSource::new(&field, phrase.clone(), tokenizer, key);
    if !template_parts {
        verifier = verifier.whole_messages_only();
    }
    Ok(pattern_phrase_query(&field, phrase, verifier))
}

/// A phrase with slop or a prefix: each candidate's tokens are indexed in a
/// one-document memory index and `positional` is run over it, as the
/// plugin's Java verifier does.
fn general_phrase(
    field: String,
    spec: &[u8],
    approximation: Clause,
    positional: Clause,
) -> std::result::Result<Clause, Status> {
    let (tokenizer, key) = tokenizer_of(&field, spec)?;
    let description = format!("{positional:?}");
    let name = field.clone();
    let matcher: Arc<lucene_pattern_text::Positional> =
        Arc::new(move |tokens: &[(i32, String)]| {
            let mut stream = lucene_index::document::FieldTokens::default();
            let mut previous = -1i32;
            for (pos, term) in tokens {
                let mut t = lucene_index::document::FieldToken::new(term.as_bytes(), 0, 0);
                t.position_increment = pos.saturating_sub(previous);
                previous = *pos;
                stream.tokens.push(t);
            }
            let mut memory = lucene_search::memory::MemoryIndex::new();
            if memory.add_field_tokens(&name, &stream, 0, 1).is_err() {
                return false;
            }
            // A prefix phrase expands over this message's own terms.
            let query = match &positional {
                Clause::Extended(e) => match e.as_ref() {
                    lucene_search::extended_query::ExtendedQuery::MultiPhrasePrefix(q) => {
                        let terms = memory
                            .field_terms()
                            .into_iter()
                            .find(|(f, _)| *f == name)
                            .map(|(_, t)| t)
                            .unwrap_or_default();
                        q.rewrite_over(&terms)
                    }
                    _ => positional.clone(),
                },
                _ => positional.clone(),
            };
            memory.any_match(&query).unwrap_or(false)
        });
    let verifier = PatternPhraseSource::positional(&field, description, matcher, tokenizer, key);
    Ok(lucene_pattern_text::pattern_positional_query(
        approximation,
        verifier,
    ))
}

/// The field's index analyzer as a message tokenizer, and a key naming it.
fn tokenizer_of(field: &str, spec: &[u8]) -> std::result::Result<(Arc<Tokenizer>, String), Status> {
    let analyzer = crate::analysis::analyzer(spec)?;
    let mut hasher = DefaultHasher::new();
    spec.hash(&mut hasher);
    let key = format!("{:016x}", hasher.finish());
    let name = field.to_string();
    // A message the analyzer refuses keeps the tokens read before the
    // failure, so its document can only miss, never match wrongly.
    let tokenizer: Arc<Tokenizer> = Arc::new(move |text: &str, out: &mut Vec<(i32, String)>| {
        let Ok(mut ts) = analyzer.analyzer().token_stream(&name, text) else {
            return;
        };
        let mut pos = -1i32;
        if ts.reset().is_err() {
            return;
        }
        while let Ok(true) = ts.increment_token() {
            let a = ts.attributes();
            pos = pos.saturating_add(a.position_increment());
            out.push((pos, a.term().to_string()));
        }
        let _ = ts.end();
        let _ = ts.close();
    });
    Ok((tokenizer, key))
}
