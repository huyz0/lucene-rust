//! `org.apache.lucene.search.vectorhighlight` (`lucene-highlighter`): the
//! `FastVectorHighlighter`. It reads a field's term vector (positions and
//! offsets) instead of re-analysing the text: the occurrences of the query's
//! terms ([`FieldTermStack`]) are grouped into the phrases the query asks for
//! ([`FieldPhraseList`]), those into fragments ([`FragListBuilder`],
//! [`FieldFragList`]), and the fragments are cut from the stored value and
//! marked up ([`FragmentsBuilder`]).
//!
//! Deviations from Java:
//!
//! - A `BooleanQuery` is flattened in the order `MUST`, `FILTER`, `SHOULD`:
//!   this port's `BooleanQuery` keeps its clauses by kind, not in the order
//!   they were added, and that order picks the number (so the colour) of each
//!   term or phrase.
//! - `FragmentsBuilder` is one type ordered by [`FragmentsOrder`], not a
//!   class hierarchy, and a fragments builder works on a copy of the
//!   fragments, so the same [`FieldFragList`] can be used again.
//! - A multi-term query is expanded against the terms of the index with the
//!   same matching as the classic highlighter's (`expand_clause`), up to
//!   1024 terms.

mod field_phrase_list;
mod field_query;
mod field_term_stack;
mod frag_list;
mod fragments;

pub use field_phrase_list::{FieldPhraseList, Toffs, WeightedPhraseInfo};
pub use field_query::{FieldQuery, QueryPhraseMap};
pub use field_term_stack::{FieldTermStack, TermInfo};
pub use frag_list::{
    BaseFragListBuilder, FieldFragList, FragListBuilder, FragListKind, SimpleFragListBuilder,
    SingleFragListBuilder, SubInfo, WeightedFragInfo, WeightedFragListBuilder, MARGIN_DEFAULT,
};
pub use fragments::{
    BoundaryScanner, BreakIteratorBoundaryScanner, FragmentsBuilder, FragmentsOrder,
    SimpleBoundaryScanner, COLORED_POST_TAGS, COLORED_PRE_TAGS,
};

use super::{Encoder, Error, Result};
use crate::directory_reader::DirectoryReader;
use crate::query::Clause;

/// `FastVectorHighlighter`.
pub struct FastVectorHighlighter {
    phrase_highlight: bool,
    field_match: bool,
    frag_list_builder: Box<dyn FragListBuilder>,
    fragments_builder: FragmentsBuilder,
    phrase_limit: usize,
}

impl Default for FastVectorHighlighter {
    fn default() -> Self {
        Self::new(Self::DEFAULT_PHRASE_HIGHLIGHT, Self::DEFAULT_FIELD_MATCH)
    }
}

impl FastVectorHighlighter {
    /// `DEFAULT_PHRASE_HIGHLIGHT`.
    pub const DEFAULT_PHRASE_HIGHLIGHT: bool = true;
    /// `DEFAULT_FIELD_MATCH`.
    pub const DEFAULT_FIELD_MATCH: bool = true;

    /// `FastVectorHighlighter(phraseHighlight, fieldMatch)`: a
    /// [`SimpleFragListBuilder`] and score-ordered fragments.
    pub fn new(phrase_highlight: bool, field_match: bool) -> Self {
        Self::with_builders(
            phrase_highlight,
            field_match,
            Box::new(SimpleFragListBuilder::new()),
            FragmentsBuilder::score_order(),
        )
    }

    /// `FastVectorHighlighter(phraseHighlight, fieldMatch, fragListBuilder,
    /// fragmentsBuilder)`.
    pub fn with_builders(
        phrase_highlight: bool,
        field_match: bool,
        frag_list_builder: Box<dyn FragListBuilder>,
        fragments_builder: FragmentsBuilder,
    ) -> Self {
        Self {
            phrase_highlight,
            field_match,
            frag_list_builder,
            fragments_builder,
            phrase_limit: usize::MAX,
        }
    }

    /// `getFieldQuery(query, reader)`.
    ///
    /// # Errors
    /// A multi-term query whose terms cannot be read.
    pub fn get_field_query(
        &self,
        query: &Clause,
        reader: Option<&DirectoryReader>,
    ) -> Result<FieldQuery> {
        FieldQuery::new(query, reader, self.phrase_highlight, self.field_match)
    }

    /// `getBestFragment(fieldQuery, reader, docId, fieldName, fragCharSize)`.
    ///
    /// # Errors
    /// A document the reader lacks, a `frag_char_size` too small for the
    /// fragment list builder, or offsets outside the stored text.
    pub fn get_best_fragment(
        &mut self,
        field_query: &FieldQuery,
        reader: &DirectoryReader,
        doc: i32,
        field: &str,
        frag_char_size: i32,
    ) -> Result<Option<String>> {
        let list = self.field_frag_list(
            &*self.frag_list_builder,
            field_query,
            reader,
            doc,
            &[field],
            false,
            frag_char_size,
        )?;
        self.fragments_builder
            .create_fragment(reader, doc, field, &list)
    }

    /// `getBestFragments(fieldQuery, reader, docId, fieldName, fragCharSize,
    /// maxNumFragments)`: `None` for a document without the field stored.
    ///
    /// # Errors
    /// As [`Self::get_best_fragment`], and a negative `max_num_fragments`.
    pub fn get_best_fragments(
        &mut self,
        field_query: &FieldQuery,
        reader: &DirectoryReader,
        doc: i32,
        field: &str,
        frag_char_size: i32,
        max_num_fragments: i32,
    ) -> Result<Option<Vec<String>>> {
        let list = self.field_frag_list(
            &*self.frag_list_builder,
            field_query,
            reader,
            doc,
            &[field],
            false,
            frag_char_size,
        )?;
        self.fragments_builder
            .create_fragments(reader, doc, field, &list, max_num_fragments)
    }

    /// `getBestFragment(..., fragListBuilder, fragmentsBuilder, preTags,
    /// postTags, encoder)`.
    ///
    /// # Errors
    /// As [`Self::get_best_fragment`].
    #[allow(clippy::too_many_arguments)]
    pub fn get_best_fragment_with(
        &self,
        field_query: &FieldQuery,
        reader: &DirectoryReader,
        doc: i32,
        field: &str,
        frag_char_size: i32,
        frag_list_builder: &dyn FragListBuilder,
        fragments_builder: &mut FragmentsBuilder,
        pre_tags: &[String],
        post_tags: &[String],
        encoder: &dyn Encoder,
    ) -> Result<Option<String>> {
        let list = self.field_frag_list(
            frag_list_builder,
            field_query,
            reader,
            doc,
            &[field],
            false,
            frag_char_size,
        )?;
        fragments_builder
            .create_fragment_with(reader, doc, field, &list, pre_tags, post_tags, encoder)
    }

    /// `getBestFragments(..., maxNumFragments, fragListBuilder,
    /// fragmentsBuilder, preTags, postTags, encoder)`.
    ///
    /// # Errors
    /// As [`Self::get_best_fragments`].
    #[allow(clippy::too_many_arguments)]
    pub fn get_best_fragments_with(
        &self,
        field_query: &FieldQuery,
        reader: &DirectoryReader,
        doc: i32,
        field: &str,
        frag_char_size: i32,
        max_num_fragments: i32,
        frag_list_builder: &dyn FragListBuilder,
        fragments_builder: &mut FragmentsBuilder,
        pre_tags: &[String],
        post_tags: &[String],
        encoder: &dyn Encoder,
    ) -> Result<Option<Vec<String>>> {
        let list = self.field_frag_list(
            frag_list_builder,
            field_query,
            reader,
            doc,
            &[field],
            false,
            frag_char_size,
        )?;
        fragments_builder.create_fragments_with(
            reader,
            doc,
            field,
            &list,
            max_num_fragments,
            pre_tags,
            post_tags,
            encoder,
        )
    }

    /// `getBestFragments(..., storedField, matchedFields, ...)`: the matches
    /// found in several fields (each one's term vector) marked up in the
    /// stored value of `stored_field`.
    ///
    /// # Errors
    /// As [`Self::get_best_fragments`], and an empty `matched_fields`.
    #[allow(clippy::too_many_arguments)]
    pub fn get_best_fragments_matched(
        &self,
        field_query: &FieldQuery,
        reader: &DirectoryReader,
        doc: i32,
        stored_field: &str,
        matched_fields: &[&str],
        frag_char_size: i32,
        max_num_fragments: i32,
        frag_list_builder: &dyn FragListBuilder,
        fragments_builder: &mut FragmentsBuilder,
        pre_tags: &[String],
        post_tags: &[String],
        encoder: &dyn Encoder,
    ) -> Result<Option<Vec<String>>> {
        if matched_fields.is_empty() {
            return Err(Error::IllegalArgument(
                "matchedFields must contain at least on field name.".to_string(),
            ));
        }
        let list = self.field_frag_list(
            frag_list_builder,
            field_query,
            reader,
            doc,
            matched_fields,
            true,
            frag_char_size,
        )?;
        fragments_builder.create_fragments_with(
            reader,
            doc,
            stored_field,
            &list,
            max_num_fragments,
            pre_tags,
            post_tags,
            encoder,
        )
    }

    /// `getFieldFragList`: one field's, or the merge of several fields'.
    #[allow(clippy::too_many_arguments)] // Java's signature, one for one
    fn field_frag_list(
        &self,
        frag_list_builder: &dyn FragListBuilder,
        field_query: &FieldQuery,
        reader: &DirectoryReader,
        doc: i32,
        fields: &[&str],
        merge: bool,
        frag_char_size: i32,
    ) -> Result<FieldFragList> {
        let mut lists = Vec::with_capacity(fields.len());
        for field in fields {
            let mut stack = FieldTermStack::new(reader, doc, field, field_query)?;
            lists.push(FieldPhraseList::new(
                &mut stack,
                field_query,
                self.phrase_limit,
            ));
        }
        let phrases = match lists.as_mut_slice() {
            [only] if !merge => std::mem::take(only),
            _ => FieldPhraseList::merge(&lists),
        };
        frag_list_builder.create_field_frag_list(&phrases, frag_char_size)
    }

    /// `isPhraseHighlight()`.
    pub fn is_phrase_highlight(&self) -> bool {
        self.phrase_highlight
    }

    /// `isFieldMatch()`.
    pub fn is_field_match(&self) -> bool {
        self.field_match
    }

    /// `getPhraseLimit()`.
    pub fn phrase_limit(&self) -> usize {
        self.phrase_limit
    }

    /// `setPhraseLimit(phraseLimit)`: a limit of zero or less finds no phrase.
    pub fn set_phrase_limit(&mut self, phrase_limit: i32) {
        self.phrase_limit = usize::try_from(phrase_limit).unwrap_or(0);
    }
}
