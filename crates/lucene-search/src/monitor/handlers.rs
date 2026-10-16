//! `org.apache.lucene.monitor.CustomQueryHandler`, `RegexpQueryHandler` and
//! `SuffixingNGramTokenFilter`: how a query no term names -- a regular
//! expression -- is made indexable.
//!
//! `RegexpQueryHandler` indexes a regexp under its longest literal run plus a
//! suffix (`fooXX` for `/foo.*bar/`) and a wildcard token. At match time the
//! batch's terms are expanded by the n-gram filter into every suffix and
//! infix gram, each with that suffix, so a document that contains the literal
//! run produces the same gram, and a token too long to expand produces the
//! wildcard token, which every regexp is indexed under.

use std::collections::HashSet;

use crate::function::java_double;
use crate::monitor::tree::{QueryTree, Term, TermWeightor};
use crate::query::Clause;

/// `CustomQueryHandler`.
pub trait CustomQueryHandler: Send + Sync {
    /// `handleQuery(query, termWeightor)`: the tree for a query the analyzer
    /// found no terms in, or `None` to leave it to the next handler.
    fn handle_query(&self, query: &Clause, weightor: &TermWeightor) -> Option<QueryTree>;

    /// `wrapTermStream(field, in)`: a field's batch terms as the presearcher
    /// asks about them; Java's default passes them through.
    fn wrap_term_stream(&self, _field: &str, terms: Vec<Vec<u8>>) -> Vec<Vec<u8>> {
        terms
    }
}

/// `RegexpQueryHandler.DEFAULT_NGRAM_SUFFIX`.
pub const DEFAULT_NGRAM_SUFFIX: &str = "XX";
/// `RegexpQueryHandler.DEFAULT_MAX_TOKEN_SIZE`.
pub const DEFAULT_MAX_TOKEN_SIZE: usize = 30;
/// `RegexpQueryHandler.DEFAULT_WILDCARD_TOKEN`.
pub const DEFAULT_WILDCARD_TOKEN: &str = "__WILDCARD__";

/// `RegexpQueryHandler`.
#[derive(Debug, Clone)]
pub struct RegexpQueryHandler {
    ngram_suffix: String,
    wildcard_token: String,
    max_token_size: usize,
    excluded_fields: Vec<String>,
}

impl Default for RegexpQueryHandler {
    /// `new RegexpQueryHandler()`.
    fn default() -> Self {
        Self::new(
            DEFAULT_NGRAM_SUFFIX,
            DEFAULT_MAX_TOKEN_SIZE,
            DEFAULT_WILDCARD_TOKEN,
            Vec::new(),
        )
    }
}

impl RegexpQueryHandler {
    /// `new RegexpQueryHandler(ngramSuffix, maxTokenSize, wildcardToken,
    /// excludedFields)`.
    pub fn new(
        ngram_suffix: impl Into<String>,
        max_token_size: usize,
        wildcard_token: impl Into<String>,
        excluded_fields: Vec<String>,
    ) -> Self {
        RegexpQueryHandler {
            ngram_suffix: ngram_suffix.into(),
            wildcard_token: wildcard_token.into(),
            max_token_size,
            excluded_fields,
        }
    }

    /// `new RegexpQueryHandler(maxTokenSize)`.
    pub fn with_max_token_size(max_token_size: usize) -> Self {
        Self::new(
            DEFAULT_NGRAM_SUFFIX,
            max_token_size,
            DEFAULT_WILDCARD_TOKEN,
            Vec::new(),
        )
    }
}

impl CustomQueryHandler for RegexpQueryHandler {
    fn wrap_term_stream(&self, field: &str, terms: Vec<Vec<u8>>) -> Vec<Vec<u8>> {
        if self.excluded_fields.iter().any(|f| f == field) {
            return terms;
        }
        suffixing_ngrams(
            &terms,
            &self.ngram_suffix,
            &self.wildcard_token,
            self.max_token_size,
        )
    }

    fn handle_query(&self, q: &Clause, weightor: &TermWeightor) -> Option<QueryTree> {
        let Clause::Regexp(query) = q else {
            return None;
        };
        // `query.toString("")`: `field:/regexp/`
        let rep = format!("{}:/{}/", query.field, query.pattern);
        let regexp = parse_out_regexp(&rep);
        let selected = select_longest_substring(regexp);
        let term = Term {
            field: query.field.clone(),
            bytes: format!("{selected}{}", self.ngram_suffix).into_bytes(),
        };
        let weight = weightor.apply(&term);
        let description = format!(
            "WILDCARD_NGRAM[{}:{}]^{}",
            term.field,
            String::from_utf8_lossy(&term.bytes),
            java_double(weight)
        );
        let wildcard = Term {
            field: term.field.clone(),
            bytes: self.wildcard_token.clone().into_bytes(),
        };
        Some(QueryTree::Custom {
            weight,
            terms: vec![term, wildcard],
            description,
        })
    }
}

/// `parseOutRegexp(rep)`: the text between the first `/` after the first `:`
/// and the last `/`.
fn parse_out_regexp(rep: &str) -> &str {
    let field_sep = rep.find(':').unwrap_or(0);
    let first_slash = rep[field_sep..].find('/').map_or(0, |i| field_sep + i);
    let last_slash = rep.rfind('/').unwrap_or(rep.len());
    rep.get(first_slash + 1..last_slash).unwrap_or("")
}

/// `selectLongestSubstring(regexp)`: the longest piece of `regexp.split("\\.|\\*|.\\?")`
/// -- the pieces between a `.`, a `*`, or any character followed by `?` --
/// the first of equals, by UTF-16 length.
fn select_longest_substring(regexp: &str) -> String {
    let chars: Vec<char> = regexp.chars().collect();
    let mut best = String::new();
    let mut best_len = 0usize;
    let mut piece = String::new();
    let mut i = 0;
    let consider = |piece: &mut String, best: &mut String, best_len: &mut usize| {
        let len = piece.encode_utf16().count();
        if len > *best_len {
            *best_len = len;
            *best = std::mem::take(piece);
        } else {
            piece.clear();
        }
    };
    while i < chars.len() {
        // the alternatives in order: `\.`, `\*`, `.\?`
        if chars[i] == '.' || chars[i] == '*' {
            consider(&mut piece, &mut best, &mut best_len);
            i += 1;
        } else if chars.get(i + 1) == Some(&'?') {
            consider(&mut piece, &mut best, &mut best_len);
            i += 2;
        } else {
            piece.push(chars[i]);
            i += 1;
        }
    }
    consider(&mut piece, &mut best, &mut best_len);
    best
}

/// `SuffixingNGramTokenFilter` over one field's terms: each term, then (for a
/// term no longer than `max_token_length`) its suffix and infix grams with
/// `suffix` appended, longest first, each gram once; a longer term is followed
/// by `wildcard` instead. Grams already seen in the stream are skipped, and
/// the first repeated suffix gram ends a term.
///
/// The filter's counters mix UTF-16 lengths with code point counts, so a term
/// with a supplementary character is not expanded; that is kept.
pub fn suffixing_ngrams(
    terms: &[Vec<u8>],
    suffix: &str,
    wildcard: &str,
    max_token_length: usize,
) -> Vec<Vec<u8>> {
    let mut out: Vec<Vec<u8>> = Vec::new();
    let mut seen_suffixes: HashSet<String> = HashSet::new();
    let mut seen_infixes: HashSet<String> = HashSet::new();
    for raw in terms {
        let text = String::from_utf8_lossy(raw).into_owned();
        out.push(text.clone().into_bytes());
        let chars: Vec<char> = text.chars().collect();
        let term_len = i64::try_from(text.encode_utf16().count()).unwrap_or(i64::MAX);
        let code_points = i64::try_from(chars.len()).unwrap_or(i64::MAX);
        if usize::try_from(term_len).unwrap_or(usize::MAX) > max_token_length {
            out.push(wildcard.as_bytes().to_vec());
            continue;
        }
        let (mut gram, mut pos) = (term_len, 0i64);
        loop {
            if gram == 0 {
                pos += 1;
                gram = term_len - pos;
            }
            if gram >= 0 && pos + gram <= code_points {
                let start = usize::try_from(pos).unwrap_or(0);
                let end = start + usize::try_from(gram).unwrap_or(0);
                let mut g: String = chars[start..end].iter().collect();
                g.push_str(suffix);
                if gram == term_len - pos && !seen_suffixes.insert(g.clone()) {
                    break;
                }
                if !seen_infixes.insert(g.clone()) {
                    gram = 0;
                    continue;
                }
                out.push(g.into_bytes());
                gram -= 1;
            } else {
                break;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::RegexpQuery;

    fn regexp(pattern: &str) -> Clause {
        Clause::Regexp(RegexpQuery {
            field: "f".into(),
            pattern: pattern.into(),
        })
    }

    #[test]
    fn the_longest_literal_run_is_chosen_and_the_first_of_equals() {
        assert_eq!(select_longest_substring("foo.*bar"), "foo");
        assert_eq!(select_longest_substring("ab.cd"), "ab");
        assert_eq!(select_longest_substring("a.bcd"), "bcd");
        assert_eq!(select_longest_substring("ab?cde"), "cde");
        assert_eq!(select_longest_substring(""), "");
        assert_eq!(select_longest_substring("..."), "");
    }

    #[test]
    fn only_a_regexp_is_handled_and_an_excluded_field_is_not_expanded() {
        let h = RegexpQueryHandler::default();
        let w = TermWeightor::default();
        assert!(h
            .handle_query(
                &Clause::Term(crate::query::TermQuery::new("f", b"x".to_vec())),
                &w
            )
            .is_none());
        let tree = h.handle_query(&regexp("foo.*"), &w).unwrap();
        assert!(tree
            .to_string_at(0)
            .unwrap()
            .starts_with("WILDCARD_NGRAM[f:fooXX]^"));
        let terms = vec![b"ab".to_vec()];
        assert_eq!(h.wrap_term_stream("f", terms.clone()).len(), 5);
        let excluded = RegexpQueryHandler::new("XX", 30, "W", vec!["f".into()]);
        assert_eq!(excluded.wrap_term_stream("f", terms.clone()), terms);
        let short = RegexpQueryHandler::with_max_token_size(1);
        let out = short.wrap_term_stream("f", terms);
        assert_eq!(out, vec![b"ab".to_vec(), b"__WILDCARD__".to_vec()]);
    }

    #[test]
    fn the_default_handler_passes_terms_through() {
        struct Nothing;
        impl CustomQueryHandler for Nothing {
            fn handle_query(&self, _: &Clause, _: &TermWeightor) -> Option<QueryTree> {
                None
            }
        }
        assert_eq!(
            Nothing.wrap_term_stream("f", vec![b"a".to_vec()]),
            vec![b"a".to_vec()]
        );
    }
}
