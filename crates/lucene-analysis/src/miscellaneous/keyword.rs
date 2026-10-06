//! Keyword marking and the filters that read or repeat it:
//! `KeywordMarkerFilter` (`SetKeywordMarkerFilter`,
//! `PatternKeywordMarkerFilter`), `KeywordRepeatFilter`,
//! `RemoveDuplicatesTokenFilter`, `StemmerOverrideFilter`.

use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;

use crate::attributes::{AttributeSource, State};
use crate::token_stream::{TokenFilter, TokenStream};
use crate::util::java_regex::JavaPattern;
use crate::{simple_to_lowercase, AnalysisError, CharArraySet};

/// `KeywordMarkerFilter.isKeyword()`.
pub trait KeywordTest: Send {
    /// Whether the current token is a keyword.
    fn is_keyword(&mut self, attributes: &AttributeSource) -> bool;
}

/// `SetKeywordMarkerFilter`'s test: the term is in the set.
pub struct InSet(Arc<CharArraySet>);

impl KeywordTest for InSet {
    fn is_keyword(&mut self, a: &AttributeSource) -> bool {
        self.0.contains(a.term())
    }
}

/// `PatternKeywordMarkerFilter`'s test: the pattern matches the whole term.
pub struct MatchesPattern(JavaPattern);

impl KeywordTest for MatchesPattern {
    fn is_keyword(&mut self, a: &AttributeSource) -> bool {
        self.0.matches(a.term())
    }
}

/// `org.apache.lucene.analysis.miscellaneous.KeywordMarkerFilter`: sets the
/// keyword flag (never clears it) where the test says so.
pub struct KeywordMarkerFilter<I, K> {
    input: I,
    test: K,
}

/// `org.apache.lucene.analysis.miscellaneous.SetKeywordMarkerFilter`.
pub type SetKeywordMarkerFilter<I> = KeywordMarkerFilter<I, InSet>;
/// `org.apache.lucene.analysis.miscellaneous.PatternKeywordMarkerFilter`.
pub type PatternKeywordMarkerFilter<I> = KeywordMarkerFilter<I, MatchesPattern>;

impl<I: TokenStream, K: KeywordTest> KeywordMarkerFilter<I, K> {
    /// A marker over any [`KeywordTest`] (Java: a subclass).
    pub fn with_test(input: I, test: K) -> Self {
        KeywordMarkerFilter { input, test }
    }
}

impl<I: TokenStream> KeywordMarkerFilter<I, InSet> {
    /// `new SetKeywordMarkerFilter(TokenStream, CharArraySet)`.
    pub fn new(input: I, keywords: Arc<CharArraySet>) -> Self {
        Self::with_test(input, InSet(keywords))
    }
}

impl<I: TokenStream> KeywordMarkerFilter<I, MatchesPattern> {
    /// `new PatternKeywordMarkerFilter(TokenStream, Pattern)`.
    pub fn with_pattern(input: I, pattern: JavaPattern) -> Self {
        Self::with_test(input, MatchesPattern(pattern))
    }
}

impl<I: TokenStream, K: KeywordTest> TokenFilter for KeywordMarkerFilter<I, K> {
    crate::filter_input!();
    // Java: KeywordMarkerFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if !self.input.increment_token()? {
            return Ok(false);
        }
        if self.test.is_keyword(self.input.attributes()) {
            self.input.attributes_mut().set_keyword(true);
        }
        Ok(true)
    }
}

/// `org.apache.lucene.analysis.miscellaneous.KeywordRepeatFilter`: every
/// token twice, first marked keyword, then (same position) not.
pub struct KeywordRepeatFilter<I> {
    input: I,
    state: Option<State>,
}

impl<I: TokenStream> KeywordRepeatFilter<I> {
    /// `new KeywordRepeatFilter(TokenStream)`.
    pub fn new(input: I) -> Self {
        KeywordRepeatFilter { input, state: None }
    }
}

impl<I: TokenStream> TokenFilter for KeywordRepeatFilter<I> {
    crate::filter_input!();
    // Java: KeywordRepeatFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if let Some(state) = self.state.take() {
            let a = self.input.attributes_mut();
            a.restore_state(&state);
            a.set_position_increment(0)?;
            a.set_keyword(false);
            return Ok(true);
        }
        if self.input.increment_token()? {
            self.state = Some(self.input.attributes().capture_state());
            self.input.attributes_mut().set_keyword(true);
            return Ok(true);
        }
        Ok(false)
    }

    fn reset_filter(&mut self) -> Result<(), AnalysisError> {
        self.input.reset()?;
        self.state = None;
        Ok(())
    }
}

/// `org.apache.lucene.analysis.miscellaneous.RemoveDuplicatesTokenFilter`:
/// drops a token whose term already occurred at the same position.
pub struct RemoveDuplicatesTokenFilter<I> {
    input: I,
    previous: HashSet<String>,
}

impl<I: TokenStream> RemoveDuplicatesTokenFilter<I> {
    /// `new RemoveDuplicatesTokenFilter(TokenStream)`.
    pub fn new(input: I) -> Self {
        RemoveDuplicatesTokenFilter {
            input,
            previous: HashSet::new(),
        }
    }
}

impl<I: TokenStream> TokenFilter for RemoveDuplicatesTokenFilter<I> {
    crate::filter_input!();
    // Java: RemoveDuplicatesTokenFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        while self.input.increment_token()? {
            let a = self.input.attributes();
            if a.position_increment() > 0 {
                self.previous.clear();
            }
            let duplicate = a.position_increment() == 0 && self.previous.contains(a.term());
            self.previous.insert(a.term().to_string());
            if !duplicate {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn reset_filter(&mut self) -> Result<(), AnalysisError> {
        self.input.reset()?;
        self.previous.clear();
        Ok(())
    }
}

/// `StemmerOverrideFilter.StemmerOverrideMap`: term -> stem.
///
/// Differs: Java compiles the map into an `FST<BytesRef>` keyed by code
/// point; this is a `HashMap` over the same keys (lowercased by
/// `Character.toLowerCase` per code point under `ignoreCase`), which answers
/// every lookup the same way -- the FST is never serialised.
#[derive(Debug, Clone, Default)]
pub struct StemmerOverrideMap {
    map: HashMap<String, String>,
    ignore_case: bool,
}

impl StemmerOverrideMap {
    /// `get(char[], int, ...)`: the stem for `term`, if any.
    pub fn get(&self, term: &str) -> Option<&str> {
        if self.ignore_case {
            let lower: String = term.chars().map(simple_to_lowercase).collect();
            self.map.get(&lower).map(String::as_str)
        } else {
            self.map.get(term).map(String::as_str)
        }
    }
}

/// `StemmerOverrideFilter.Builder`.
#[derive(Debug, Clone, Default)]
pub struct StemmerOverrideBuilder {
    map: StemmerOverrideMap,
}

impl StemmerOverrideBuilder {
    /// `new Builder(boolean ignoreCase)`.
    pub fn new(ignore_case: bool) -> Self {
        StemmerOverrideBuilder {
            map: StemmerOverrideMap {
                map: HashMap::new(),
                ignore_case,
            },
        }
    }

    /// `add(input, output)`: `false` (and ignored) if `input` was already
    /// added -- the first mapping wins, as `BytesRefHash.add` makes it.
    pub fn add(&mut self, input: &str, output: &str) -> bool {
        let key = if self.map.ignore_case {
            input.chars().map(simple_to_lowercase).collect()
        } else {
            input.to_string()
        };
        if self.map.map.contains_key(&key) {
            return false;
        }
        self.map.map.insert(key, output.to_string());
        true
    }

    /// `build()`.
    pub fn build(self) -> Arc<StemmerOverrideMap> {
        Arc::new(self.map)
    }
}

/// `org.apache.lucene.analysis.miscellaneous.StemmerOverrideFilter`: replaces
/// a non-keyword term found in the map with its stem and marks it keyword.
pub struct StemmerOverrideFilter<I> {
    input: I,
    map: Arc<StemmerOverrideMap>,
}

impl<I: TokenStream> StemmerOverrideFilter<I> {
    /// `new StemmerOverrideFilter(TokenStream, StemmerOverrideMap)`.
    pub fn new(input: I, map: Arc<StemmerOverrideMap>) -> Self {
        StemmerOverrideFilter { input, map }
    }
}

impl<I: TokenStream> TokenFilter for StemmerOverrideFilter<I> {
    crate::filter_input!();
    // Java: StemmerOverrideFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if !self.input.increment_token()? {
            return Ok(false);
        }
        let a = self.input.attributes_mut();
        if !a.is_keyword() {
            if let Some(stem) = self.map.get(a.term()) {
                let stem = stem.to_string();
                a.set_term(&stem);
                a.set_keyword(true);
            }
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::canned::{render, Canned};

    fn keyed(ts: &mut dyn TokenStream) -> Vec<(String, bool, i32)> {
        let mut out = Vec::new();
        crate::token_stream::consume(ts, |a| {
            out.push((a.term().to_string(), a.is_keyword(), a.position_increment()))
        })
        .unwrap();
        out
    }

    #[test]
    fn markers_and_repeat() {
        let set = Arc::new(CharArraySet::from_words(["b"], false));
        let mut f = SetKeywordMarkerFilter::new(Canned::parse("a:0:1:1:1 b:2:3:1:1"), set);
        assert_eq!(
            keyed(&mut f),
            vec![("a".into(), false, 1), ("b".into(), true, 1)]
        );
        let p = JavaPattern::compile("[a-z]+ing").unwrap();
        let mut f = PatternKeywordMarkerFilter::with_pattern(
            Canned::parse("sing:0:4:1:1 singer:5:11:1:1"),
            p,
        );
        assert_eq!(
            keyed(&mut f),
            vec![("sing".into(), true, 1), ("singer".into(), false, 1)]
        );
        let mut f = KeywordRepeatFilter::new(Canned::parse("a:0:1:1:1"));
        assert_eq!(
            keyed(&mut f),
            vec![("a".into(), true, 1), ("a".into(), false, 0)]
        );
    }

    #[test]
    fn duplicates_and_overrides() {
        let mut f = RemoveDuplicatesTokenFilter::new(Canned::parse(
            "a:0:1:1:1 a:0:1:0:1 b:0:1:0:1 a:2:3:1:1|3|0",
        ));
        assert_eq!(render(&mut f), "a:0:1:1:1 b:0:1:0:1 a:2:3:1:1|3|0");
        let mut b = StemmerOverrideBuilder::new(true);
        assert!(b.add("Running", "run"));
        assert!(!b.add("running", "ran"), "first mapping wins");
        let map = b.build();
        let mut c = Canned::parse("RUNNING:0:7:1:1 running:8:15:1:1 x:16:17:1:1");
        c.set_keywords(&[false, true, false]);
        let mut f = StemmerOverrideFilter::new(c, map);
        assert_eq!(
            keyed(&mut f),
            vec![
                ("run".into(), true, 1),
                ("running".into(), true, 1),
                ("x".into(), false, 1)
            ]
        );
        let mut b = StemmerOverrideBuilder::new(false);
        b.add("A", "b");
        assert_eq!(b.build().get("a"), None);
    }
}
