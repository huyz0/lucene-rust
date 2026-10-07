//! `MorfologikFilter` and `MorphosyntacticTagsAttribute`
//! (`org.apache.lucene.analysis.morfologik`).

use std::sync::Arc;

use lucene_analysis::attributes::State;
use lucene_analysis::java_character;
use lucene_analysis::token_stream::{TokenFilter, TokenStream};
use lucene_analysis::{AnalysisError, CustomAttribute};

use crate::dictionary::{Dictionary, DictionaryLookup};

/// `MorphosyntacticTagsAttribute`: the tags of the current lemma (`None`
/// when cleared, as Java's `null`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MorphosyntacticTagsAttribute {
    /// `getTags()`.
    pub tags: Option<Vec<String>>,
}

impl CustomAttribute for MorphosyntacticTagsAttribute {
    fn impl_class(&self) -> &'static str {
        "org.apache.lucene.analysis.morfologik.MorphosyntacticTagsAttributeImpl"
    }
    fn clear(&mut self) {
        self.tags = None;
    }

    /// `MorphosyntacticTagsAttributeImpl.reflectWith`: the tag list.
    fn reflect(
        &self,
        reflector: &mut dyn FnMut(&'static str, &'static str, lucene_analysis::AttrValue<'_>),
    ) {
        let tags = self.tags.as_ref().map(|t| {
            t.iter()
                .map(|s| std::borrow::Cow::Borrowed(s.as_str()))
                .collect()
        });
        reflector(
            "org.apache.lucene.analysis.morfologik.MorphosyntacticTagsAttribute",
            "tags",
            lucene_analysis::AttrValue::List(tags),
        );
    }
}

/// `MorfologikFilter.lemmaSplitter.split(tag)` into `out`, reusing its
/// strings: `+` and `|` separate tags; trailing empty tags are dropped, as
/// `Pattern.split` drops them.
fn split_tags_into(tag: &str, out: &mut Vec<String>) {
    let mut n = 0usize;
    for part in tag.split(['+', '|']) {
        match out.get_mut(n) {
            Some(s) => {
                s.clear();
                s.push_str(part);
            }
            None => out.push(part.to_string()),
        }
        n = n.saturating_add(1);
    }
    out.truncate(n);
    while out.last().is_some_and(String::is_empty) {
        out.pop();
    }
}

/// `MorfologikFilter.toLowercase`: `Character.toLowerCase` per code point,
/// into `out` (cleared first).
fn to_lowercase_into(term: &str, out: &mut String) {
    out.clear();
    if term.is_ascii() {
        out.push_str(term);
        out.make_ascii_lowercase();
        return;
    }
    for c in term.chars() {
        let cp = java_character::to_lower_case(u32::from(c));
        // A code point's lower case is a code point, never a surrogate.
        out.push(char::from_u32(cp).unwrap_or(char::REPLACEMENT_CHARACTER));
    }
}

/// `MorfologikFilter`: each term found in the dictionary (as it is, or
/// lower-cased) becomes its lemmas, the first in its place and the rest at
/// the same position, each with its tags; a keyword or unknown term passes
/// with its tags cleared.
pub struct MorfologikFilter<I> {
    input: I,
    lookup: DictionaryLookup,
    lemmas: usize,
    lemma_index: usize,
    current: Option<State>,
    lower: String,
}

impl<I: TokenStream> MorfologikFilter<I> {
    /// `new MorfologikFilter(in, dictionary)`.
    pub fn new(input: I, dictionary: Arc<Dictionary>) -> Self {
        let mut f = MorfologikFilter {
            input,
            lookup: DictionaryLookup::new(dictionary),
            lemmas: 0,
            lemma_index: 0,
            current: None,
            lower: String::new(),
        };
        f.input
            .attributes_mut()
            .add_custom::<MorphosyntacticTagsAttribute>();
        f
    }

    // Java: MorfologikFilter.popNextLemma
    fn pop_next_lemma(&mut self) {
        let (stem, tag) = self.lookup.form(self.lemma_index).unwrap_or_default();
        self.lemma_index = self.lemma_index.saturating_add(1);
        let attrs = self.input.attributes_mut();
        // CharTermAttribute.append(null) appends "null".
        attrs.set_term(stem.unwrap_or("null"));
        let tags = attrs
            .add_custom::<MorphosyntacticTagsAttribute>()
            .tags
            .get_or_insert_with(Vec::new);
        match tag {
            Some(tag) => split_tags_into(tag, tags),
            None => tags.clear(),
        }
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
        if self.lemma_index < self.lemmas {
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
        self.lemma_index = 0;
        self.lemmas = 0;
        if !attrs.is_keyword() {
            // Java: lookupSurfaceForm(termAtt) ||
            // lookupSurfaceForm(toLowercase(termAtt)). A lookup that finds
            // nothing leaves the reused forms alone, so a term that is its
            // own lower case is not looked up twice.
            let term = attrs.term();
            self.lemmas = self.lookup.lookup_str(term)?;
            if self.lemmas == 0 {
                to_lowercase_into(term, &mut self.lower);
                if self.lower != term {
                    self.lemmas = self.lookup.lookup_str(&self.lower)?;
                }
            }
        }
        if self.lemmas > 0 {
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
        self.lemmas = 0;
        self.input.reset()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split_tags(tag: &str) -> Vec<String> {
        let mut out = vec!["stale".to_string(); 4];
        split_tags_into(tag, &mut out);
        out
    }

    fn lower(term: &str) -> String {
        let mut out = "stale".to_string();
        to_lowercase_into(term, &mut out);
        out
    }

    #[test]
    fn tags_reflect_as_java_list() {
        let mut a = lucene_analysis::AttributeSource::new();
        a.add_custom::<MorphosyntacticTagsAttribute>().tags =
            Some(vec!["subst:sg".into(), "adj".into()]);
        let mut out = Vec::new();
        a.reflect_custom(&mut |c, k, v| out.push(format!("{c}#{k}={v}")));
        assert_eq!(
            out,
            ["org.apache.lucene.analysis.morfologik.MorphosyntacticTagsAttribute#tags=[subst:sg, adj]"]
        );
        // restoreState's error names Java's implementation class.
        let e = lucene_analysis::AttributeSource::new()
            .try_restore_state(&a.capture_state())
            .unwrap_err()
            .to_string();
        assert!(
            e.contains(
                "type org.apache.lucene.analysis.morfologik.MorphosyntacticTagsAttributeImpl that"
            ),
            "{e}"
        );
        a.clear_attributes();
        assert!(a.reflect_as_string(false).ends_with("tags=null"));
    }

    #[test]
    fn tags_split_as_java() {
        assert_eq!(
            split_tags("subst:sg:nom:m1+subst:pl|adj"),
            ["subst:sg:nom:m1", "subst:pl", "adj"]
        );
        assert_eq!(split_tags("+a"), ["", "a"]);
        assert!(split_tags("++").is_empty());
        assert_eq!(split_tags("x"), ["x"]);
        assert_eq!(split_tags("a|b|c|d|e"), ["a", "b", "c", "d", "e"]);
        assert_eq!(lower("ÀB𐐀"), "àb𐐨");
        assert_eq!(lower("AbC"), "abc");
        let mut t = MorphosyntacticTagsAttribute { tags: Some(vec![]) };
        CustomAttribute::clear(&mut t);
        assert_eq!(t.tags, None);
    }
}
