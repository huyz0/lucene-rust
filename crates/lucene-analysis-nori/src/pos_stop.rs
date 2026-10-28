//! `org.apache.lucene.analysis.ko.KoreanPartOfSpeechStopFilter`: drops the
//! tokens whose left part of speech is a stop tag.

use std::collections::HashSet;
use std::sync::{Arc, LazyLock};

use lucene_analysis::attributes::AttributeSource;
use lucene_analysis::{FilteringTokenFilter, TokenStream};

use crate::attributes::PartOfSpeechAttribute;
use crate::pos::Tag;

/// `KoreanPartOfSpeechStopFilter.DEFAULT_STOP_TAGS`: endings, particles,
/// adverbs and determiners, spaces, brackets, separators, affixes and the
/// unknown tags.
pub fn default_stop_tags() -> Arc<HashSet<Tag>> {
    static TAGS: LazyLock<Arc<HashSet<Tag>>> = LazyLock::new(|| {
        Arc::new(HashSet::from([
            Tag::Ep,
            Tag::Ef,
            Tag::Ec,
            Tag::Etn,
            Tag::Etm,
            Tag::Ic,
            Tag::Jks,
            Tag::Jkc,
            Tag::Jkg,
            Tag::Jko,
            Tag::Jkb,
            Tag::Jkv,
            Tag::Jkq,
            Tag::Jx,
            Tag::Jc,
            Tag::Mag,
            Tag::Maj,
            Tag::Mm,
            Tag::Sp,
            Tag::Ssc,
            Tag::Sso,
            Tag::Sc,
            Tag::Se,
            Tag::Xpn,
            Tag::Xsa,
            Tag::Xsn,
            Tag::Xsv,
            Tag::Una,
            Tag::Na,
            Tag::Vsv,
        ]))
    });
    Arc::clone(&TAGS)
}

/// The predicate of [`KoreanPartOfSpeechStopFilter`]: the stop tags as a
/// bit per `Tag` ordinal (48 of them), so the per-token test is a shift
/// rather than a hash of the tag.
pub struct StopTags(u64);

// `Vsv` is the last tag: every ordinal fits the mask.
const _: () = assert!((Tag::Vsv as u32) < 64);

impl StopTags {
    fn new(tags: &HashSet<Tag>) -> Self {
        StopTags(tags.iter().fold(0, |bits, &t| bits | (1u64 << t as u32)))
    }
}

impl lucene_analysis::token_stream::Accept for StopTags {
    // Java: KoreanPartOfSpeechStopFilter.accept
    fn accept(&mut self, atts: &AttributeSource) -> Result<bool, lucene_analysis::AnalysisError> {
        let left = atts
            .custom::<PartOfSpeechAttribute>()
            .and_then(PartOfSpeechAttribute::left_pos);
        Ok(left.is_none_or(|t| self.0 & (1u64 << t as u32) == 0))
    }
}

/// `KoreanPartOfSpeechStopFilter`.
pub type KoreanPartOfSpeechStopFilter<I> = FilteringTokenFilter<I, StopTags>;

/// `new KoreanPartOfSpeechStopFilter(input, stopTags)`.
pub fn korean_part_of_speech_stop_filter<I: TokenStream>(
    mut input: I,
    stop_tags: Arc<HashSet<Tag>>,
) -> KoreanPartOfSpeechStopFilter<I> {
    input.attributes_mut().add_custom::<PartOfSpeechAttribute>();
    FilteringTokenFilter::new(input, StopTags::new(&stop_tags))
}
