//! `org.apache.lucene.analysis.ja.JapanesePartOfSpeechStopFilter`: drops
//! the tokens whose part of speech is a stop tag.
//!
//! Differs: Java's `Set<String>` of stop tags is a [`CharArraySet`]
//! (case-sensitive, as the factory and the analyzer load it): one probe per
//! token, with the token's part of speech borrowed from the dictionary and
//! hashed by the set's word hash rather than SipHash.

use std::sync::Arc;

use lucene_analysis::attributes::AttributeSource;
use lucene_analysis::{CharArraySet, FilteringTokenFilter, TokenStream};

use crate::attributes::PartOfSpeechAttribute;

/// The predicate of [`JapanesePartOfSpeechStopFilter`].
pub struct StopTags(Arc<CharArraySet>);

impl lucene_analysis::token_stream::Accept for StopTags {
    // Java: JapanesePartOfSpeechStopFilter.accept
    fn accept(&mut self, atts: &AttributeSource) -> Result<bool, lucene_analysis::AnalysisError> {
        let pos = atts
            .custom::<PartOfSpeechAttribute>()
            .and_then(PartOfSpeechAttribute::part_of_speech);
        Ok(pos.is_none_or(|p| !self.0.contains(p)))
    }
}

/// `JapanesePartOfSpeechStopFilter`.
pub type JapanesePartOfSpeechStopFilter<I> = FilteringTokenFilter<I, StopTags>;

/// `new JapanesePartOfSpeechStopFilter(input, stopTags)`.
pub fn japanese_part_of_speech_stop_filter<I: TokenStream>(
    mut input: I,
    stop_tags: Arc<CharArraySet>,
) -> JapanesePartOfSpeechStopFilter<I> {
    input.attributes_mut().add_custom::<PartOfSpeechAttribute>();
    FilteringTokenFilter::new(input, StopTags(stop_tags))
}
