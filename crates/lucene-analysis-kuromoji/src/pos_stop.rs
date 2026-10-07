//! `org.apache.lucene.analysis.ja.JapanesePartOfSpeechStopFilter`: drops
//! the tokens whose part of speech is a stop tag.

use std::collections::HashSet;
use std::sync::Arc;

use lucene_analysis::attributes::AttributeSource;
use lucene_analysis::{FilteringTokenFilter, TokenStream};

use crate::attributes::PartOfSpeechAttribute;

/// The predicate of [`JapanesePartOfSpeechStopFilter`].
pub struct StopTags(Arc<HashSet<String>>);

impl lucene_analysis::token_stream::Accept for StopTags {
    // Java: JapanesePartOfSpeechStopFilter.accept
    fn accept(&mut self, atts: &AttributeSource) -> Result<bool, lucene_analysis::AnalysisError> {
        let pos = atts
            .custom::<PartOfSpeechAttribute>()
            .and_then(PartOfSpeechAttribute::part_of_speech);
        Ok(pos.is_none_or(|p| !self.0.contains(&p)))
    }
}

/// `JapanesePartOfSpeechStopFilter`.
pub type JapanesePartOfSpeechStopFilter<I> = FilteringTokenFilter<I, StopTags>;

/// `new JapanesePartOfSpeechStopFilter(input, stopTags)`.
pub fn japanese_part_of_speech_stop_filter<I: TokenStream>(
    mut input: I,
    stop_tags: Arc<HashSet<String>>,
) -> JapanesePartOfSpeechStopFilter<I> {
    input.attributes_mut().add_custom::<PartOfSpeechAttribute>();
    FilteringTokenFilter::new(input, StopTags(stop_tags))
}
