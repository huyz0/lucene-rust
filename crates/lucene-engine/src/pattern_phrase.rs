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
//! | positions | `i32` count, at least 1, then per position `position: i32` (non-decreasing), `n: i32` (at least 1) and `n` terms as bytes |

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
pub const MAX_TERMS: usize = 1024;

pub fn string(c: &mut Cursor<'_>) -> std::result::Result<String, Status> {
    Ok(std::str::from_utf8(c.bytes()?)
        .map_err(|_| Status::InvalidUtf8)?
        .to_string())
}

pub fn decode(c: &mut Cursor<'_>) -> std::result::Result<Clause, Status> {
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
    let analyzer = crate::analysis::analyzer(spec)?;
    let mut hasher = DefaultHasher::new();
    spec.hash(&mut hasher);
    let key = format!("{:016x}", hasher.finish());
    let name = field.clone();
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
    let phrase = PhraseTokens::new(tokens);
    let mut verifier = PatternPhraseSource::new(&field, phrase.clone(), tokenizer, key);
    if !template_parts {
        verifier = verifier.whole_messages_only();
    }
    Ok(pattern_phrase_query(&field, phrase, verifier))
}
