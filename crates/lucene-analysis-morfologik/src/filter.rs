//! `MorfologikFilter` and `MorphosyntacticTagsAttribute`
//! (`org.apache.lucene.analysis.morfologik`).

use std::sync::Arc;

use lucene_analysis::attributes::State;
use lucene_analysis::java_character;
use lucene_analysis::token_stream::{TokenFilter, TokenStream};
use lucene_analysis::{AnalysisError, CustomAttribute};

use crate::dictionary::{Dictionary, DictionaryLookup, WordData};

/// `MorphosyntacticTagsAttribute`: the tags of the current lemma (`None`
/// when cleared, as Java's `null`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MorphosyntacticTagsAttribute {
    /// `getTags()`.
    pub tags: Option<Vec<String>>,
}

impl CustomAttribute for MorphosyntacticTagsAttribute {
    fn clear(&mut self) {
        self.tags = None;
    }
}

/// `MorfologikFilter.lemmaSplitter.split(tag)`: `+` and `|` separate tags;
/// trailing empty tags are dropped, as `Pattern.split` drops them.
fn split_tags(tag: &str) -> Vec<String> {
    let mut parts: Vec<String> = tag.split(['+', '|']).map(str::to_string).collect();
    while parts.last().is_some_and(String::is_empty) {
        parts.pop();
    }
    parts
}

/// `MorfologikFilter.toLowercase`: `Character.toLowerCase` per code point.
fn to_lowercase(units: &[u16]) -> Vec<u16> {
    let mut out = Vec::with_capacity(units.len());
    for r in char::decode_utf16(units.iter().copied()) {
        let cp = r.map_or_else(|e| u32::from(e.unpaired_surrogate()), u32::from);
        java_character::push_utf16(&mut out, java_character::to_lower_case(cp));
    }
    out
}

/// `MorfologikFilter`: each term found in the dictionary (as it is, or
/// lower-cased) becomes its lemmas, the first in its place and the rest at
/// the same position, each with its tags; a keyword or unknown term passes
/// with its tags cleared.
pub struct MorfologikFilter<I> {
    input: I,
    lookup: DictionaryLookup,
    lemmas: Vec<WordData>,
    lemma_index: usize,
    current: Option<State>,
}

impl<I: TokenStream> MorfologikFilter<I> {
    /// `new MorfologikFilter(in, dictionary)`.
    pub fn new(input: I, dictionary: Arc<Dictionary>) -> Self {
        let mut f = MorfologikFilter {
            input,
            lookup: DictionaryLookup::new(dictionary),
            lemmas: Vec::new(),
            lemma_index: 0,
            current: None,
        };
        f.input
            .attributes_mut()
            .add_custom::<MorphosyntacticTagsAttribute>();
        f
    }

    // Java: MorfologikFilter.lookupSurfaceForm
    fn lookup(&mut self, word: &[u16]) -> Result<bool, AnalysisError> {
        self.lemmas = self.lookup.lookup(word)?;
        self.lemma_index = 0;
        Ok(!self.lemmas.is_empty())
    }

    // Java: MorfologikFilter.popNextLemma
    fn pop_next_lemma(&mut self) {
        let lemma = &self.lemmas[self.lemma_index];
        self.lemma_index = self.lemma_index.saturating_add(1);
        let attrs = self.input.attributes_mut();
        match &lemma.stem {
            Some(stem) => attrs.set_term_utf16(stem),
            // CharTermAttribute.append(null) appends "null".
            None => attrs.set_term("null"),
        }
        let tags = match &lemma.tag {
            Some(tag) => split_tags(&String::from_utf16_lossy(tag)),
            None => Vec::new(),
        };
        attrs.add_custom::<MorphosyntacticTagsAttribute>().tags = Some(tags);
    }
}

impl<I: TokenStream> TokenFilter for MorfologikFilter<I> {
    type Input = I;
    fn input(&self) -> &I {
        &self.input
    }
    fn input_mut(&mut self) -> &mut I {
        &mut self.input
    }

    // Java: MorfologikFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if self.lemma_index < self.lemmas.len() {
            if let Some(state) = &self.current {
                self.input.attributes_mut().restore_state(state);
            }
            self.input.attributes_mut().set_position_increment(0)?;
            self.pop_next_lemma();
            return Ok(true);
        }
        if !self.input.increment_token()? {
            return Ok(false);
        }
        let attrs = self.input.attributes();
        let found = if attrs.is_keyword() {
            false
        } else {
            let term: Vec<u16> = attrs.term().encode_utf16().collect();
            self.lookup(&term)? || self.lookup(&to_lowercase(&term))?
        };
        if found {
            self.current = Some(self.input.attributes().capture_state());
            self.pop_next_lemma();
        } else {
            self.input
                .attributes_mut()
                .add_custom::<MorphosyntacticTagsAttribute>()
                .tags = None;
        }
        Ok(true)
    }

    fn reset_filter(&mut self) -> Result<(), AnalysisError> {
        self.lemma_index = 0;
        self.lemmas.clear();
        self.input.reset()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tags_split_as_java() {
        assert_eq!(
            split_tags("subst:sg:nom:m1+subst:pl|adj"),
            ["subst:sg:nom:m1", "subst:pl", "adj"]
        );
        assert_eq!(split_tags("+a"), ["", "a"]);
        assert!(split_tags("++").is_empty());
        assert_eq!(split_tags("x"), ["x"]);
        assert_eq!(
            to_lowercase(&"ÀB𐐀".encode_utf16().collect::<Vec<_>>()),
            "àb𐐨".encode_utf16().collect::<Vec<_>>()
        );
        let mut t = MorphosyntacticTagsAttribute { tags: Some(vec![]) };
        CustomAttribute::clear(&mut t);
        assert_eq!(t.tags, None);
    }
}
