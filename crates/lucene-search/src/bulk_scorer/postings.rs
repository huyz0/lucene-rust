//! Native postings for `TermScorer` and `PhraseScorer`: binary blocks or SimpleText's line
//! reader and up to four impacts levels. No text-to-binary conversion.
//! Differs: text impacts are copied after a fallible positioning call so
//! the scorer's bound getters remain infallible.

use lucene_codecs::postings::{
    Impact, LazyDocsCursor, Level0Impacts, PositionsCursor, NO_MORE_DOCS,
};
use lucene_codecs::simpletext::fields_reader::SimpleTextImpactsEnum;

use crate::{blocktree, Result};

// The binary scorer already keeps LazyDocsCursor inline. Preserve that
// allocation and access pattern while introducing the text alternative.
#[allow(clippy::large_enum_variant)]
pub(crate) enum TermPostings<'a> {
    Binary(LazyDocsCursor<'a>),
    Text(Box<TextPostings<'a>>),
}

impl<'a> From<LazyDocsCursor<'a>> for TermPostings<'a> {
    fn from(cursor: LazyDocsCursor<'a>) -> Self {
        Self::Binary(cursor)
    }
}

pub(crate) struct TextPostings<'a> {
    cursor: SimpleTextImpactsEnum<'a>,
    has_freqs: bool,
    levels: Vec<(i32, Vec<Impact>)>,
}

fn text_error(error: lucene_codecs::simpletext::Error) -> crate::Error {
    crate::directory_reader::Error::SimpleText(error).into()
}

impl TextPostings<'_> {
    fn refresh(&mut self) -> Result<()> {
        let impacts = self.cursor.impacts().map_err(text_error)?;
        self.levels
            .resize_with(impacts.num_levels(), || (NO_MORE_DOCS, Vec::new()));
        for (level, (end, pairs)) in self.levels.iter_mut().enumerate() {
            *end = impacts.doc_id_up_to(level).expect("existing impacts level");
            pairs.clear();
            pairs.extend_from_slice(impacts.impacts(level).expect("existing impacts level"));
        }
        Ok(())
    }

    fn postings(&self) -> &lucene_codecs::simpletext::fields_reader::SimpleTextPostings<'_> {
        match &self.cursor {
            SimpleTextImpactsEnum::Slow(p) | SimpleTextImpactsEnum::Skip(p) => p,
        }
    }
}

impl<'a> TermPostings<'a> {
    pub(crate) fn text(cursor: SimpleTextImpactsEnum<'a>, has_freqs: bool) -> Result<Self> {
        let mut text = TextPostings {
            cursor,
            has_freqs,
            levels: Vec::new(),
        };
        text.refresh()?;
        Ok(Self::Text(Box::new(text)))
    }

    pub(crate) fn doc_id(&self) -> i32 {
        match self {
            Self::Binary(c) => c.doc_id(),
            Self::Text(c) => c.postings().doc_id(),
        }
    }

    pub(crate) fn has_freqs(&self) -> bool {
        match self {
            Self::Binary(c) => c.has_freqs(),
            Self::Text(c) => c.has_freqs,
        }
    }

    pub(crate) fn binary_docs_only_impacts(&self) -> bool {
        matches!(self, Self::Binary(_)) && !self.has_freqs()
    }

    pub(crate) fn freq(&self) -> Option<i32> {
        match self {
            Self::Binary(c) => c.freq(),
            Self::Text(c) => (c.postings().doc_id() >= 0 && c.postings().doc_id() != NO_MORE_DOCS)
                .then(|| c.postings().freq()),
        }
    }

    pub(crate) fn next_doc(&mut self) -> Result<i32> {
        match self {
            Self::Binary(c) => Ok(c.next_doc().map_err(blocktree::Error::Postings)?),
            Self::Text(c) => {
                if c.postings().doc_id() == NO_MORE_DOCS {
                    return Ok(NO_MORE_DOCS);
                }
                let doc = c.cursor.postings().next_doc().map_err(text_error)?;
                c.refresh()?;
                Ok(doc)
            }
        }
    }

    pub(crate) fn advance(&mut self, target: i32) -> Result<i32> {
        match self {
            Self::Binary(c) => Ok(c.advance(target).map_err(blocktree::Error::Postings)?),
            Self::Text(c) => {
                // Search's binary cursor permits a target at or behind its
                // current document; every format keeps that same contract.
                if c.postings().doc_id() >= target {
                    return Ok(c.postings().doc_id());
                }
                let doc = c.cursor.postings().advance(target).map_err(text_error)?;
                c.refresh()?;
                Ok(doc)
            }
        }
    }

    pub(crate) fn advance_shallow(&mut self, target: i32) -> Result<i32> {
        match self {
            Self::Binary(c) => Ok(c
                .advance_shallow(target)
                .map_err(blocktree::Error::Postings)?),
            Self::Text(c) => {
                c.cursor.advance_shallow(target).map_err(text_error)?;
                c.refresh()?;
                Ok(c.levels[0].0)
            }
        }
    }

    pub(crate) fn num_levels(&self) -> usize {
        match self {
            Self::Binary(c) => {
                if c.level1_last_doc_id() != NO_MORE_DOCS && !c.level1_impacts().is_empty() {
                    2
                } else {
                    1
                }
            }
            Self::Text(c) => c.levels.len(),
        }
    }

    pub(crate) fn doc_id_up_to(&self, level: usize) -> i32 {
        match self {
            Self::Binary(c) => match level {
                0 => c.level0_last_doc_id(),
                1 => c.level1_last_doc_id(),
                _ => NO_MORE_DOCS,
            },
            Self::Text(c) => c.levels.get(level).map_or(NO_MORE_DOCS, |l| l.0),
        }
    }

    pub(crate) fn level0_last_doc_id(&self) -> i32 {
        self.doc_id_up_to(0)
    }

    pub(crate) fn level0_impacts_view(&self) -> Level0Impacts<'_> {
        match self {
            Self::Binary(c) => c.level0_impacts_view(),
            Self::Text(c) => Level0Impacts::Decoded(&c.levels[0].1),
        }
    }

    pub(crate) fn impacts(&mut self, level: usize) -> &[Impact] {
        match self {
            Self::Binary(c) => match level {
                0 => c.level0_impacts(),
                1 => c.level1_impacts(),
                _ => &[],
            },
            Self::Text(c) => c.levels.get(level).map_or(&[], |l| l.1.as_slice()),
        }
    }

    pub(crate) fn level0_impacts(&mut self) -> &[Impact] {
        self.impacts(0)
    }

    pub(crate) fn advance_exact_in_bits(&mut self, target: i32) -> Option<bool> {
        match self {
            Self::Binary(c) => c.advance_exact_in_bits(target),
            Self::Text(_) => None,
        }
    }

    pub(crate) fn doc_id_run_end(&self) -> i32 {
        match self {
            Self::Binary(c) => c.doc_id_run_end(),
            // Java's default DocIdSetIterator.docIDRunEnd().
            Self::Text(c) => c.postings().doc_id().saturating_add(1),
        }
    }

    pub(crate) fn next_postings_at_most(
        &mut self,
        up_to: i32,
        max: usize,
        docs: &mut Vec<i32>,
        freqs: &mut Vec<i32>,
    ) -> Result<()> {
        match self {
            Self::Binary(c) => Ok(c
                .next_postings_at_most(up_to, max, docs, freqs)
                .map_err(blocktree::Error::Postings)?),
            Self::Text(c) => {
                // Java: PostingsEnum.nextPostings, at most 16 documents.
                docs.clear();
                freqs.clear();
                let p = c.cursor.postings();
                while p.doc_id() >= 0 && p.doc_id() < up_to && docs.len() < max.clamp(1, 16) {
                    docs.push(p.doc_id());
                    freqs.push(p.freq());
                    p.next_doc().map_err(text_error)?;
                }
                c.refresh()?;
                Ok(())
            }
        }
    }
}

/// Native positional postings for PhraseWeight: the binary cursor stays
/// inline, while text retains its line reader and all skip levels.
#[allow(clippy::large_enum_variant)]
pub(crate) enum PhrasePostings<'a> {
    Binary(PositionsCursor<'a>),
    Text(Box<TextPostings<'a>>),
}

impl<'a> From<PositionsCursor<'a>> for PhrasePostings<'a> {
    fn from(cursor: PositionsCursor<'a>) -> Self {
        Self::Binary(cursor)
    }
}

impl<'a> PhrasePostings<'a> {
    pub(crate) fn text(cursor: SimpleTextImpactsEnum<'a>) -> Result<Self> {
        let mut text = TextPostings {
            cursor,
            has_freqs: true,
            levels: Vec::new(),
        };
        text.refresh()?;
        Ok(Self::Text(Box::new(text)))
    }

    pub(crate) fn doc_id(&self) -> i32 {
        match self {
            Self::Binary(c) => c.doc_id(),
            Self::Text(c) => c.postings().doc_id(),
        }
    }

    pub(crate) fn freq(&self) -> i32 {
        match self {
            Self::Binary(c) => c.freq(),
            Self::Text(c) => c.postings().freq(),
        }
    }

    pub(crate) fn next_doc(&mut self) -> Result<i32> {
        match self {
            Self::Binary(c) => Ok(c.next_doc().map_err(blocktree::Error::Postings)?),
            Self::Text(c) => {
                if c.postings().doc_id() == NO_MORE_DOCS {
                    return Ok(NO_MORE_DOCS);
                }
                let doc = c.cursor.postings().next_doc().map_err(text_error)?;
                c.refresh()?;
                Ok(doc)
            }
        }
    }

    pub(crate) fn advance(&mut self, target: i32) -> Result<i32> {
        match self {
            Self::Binary(c) => Ok(c.advance(target).map_err(blocktree::Error::Postings)?),
            Self::Text(c) => {
                if c.postings().doc_id() >= target {
                    return Ok(c.postings().doc_id());
                }
                let doc = c.cursor.postings().advance(target).map_err(text_error)?;
                c.refresh()?;
                Ok(doc)
            }
        }
    }

    pub(crate) fn next_position(&mut self) -> Result<i32> {
        match self {
            Self::Binary(c) => Ok(c.next_position().map_err(blocktree::Error::Postings)?),
            Self::Text(c) => c.cursor.postings().next_position().map_err(text_error),
        }
    }

    pub(crate) fn positions_into(&mut self, out: &mut Vec<i32>) -> Result<()> {
        match self {
            Self::Binary(c) => Ok(c.positions_into(out).map_err(blocktree::Error::Postings)?),
            Self::Text(c) => {
                for _ in 0..c.postings().freq() {
                    out.push(c.cursor.postings().next_position().map_err(text_error)?);
                }
                Ok(())
            }
        }
    }

    pub(crate) fn advance_shallow(&mut self, target: i32) -> Result<()> {
        match self {
            Self::Binary(c) => {
                c.docs_mut()
                    .advance_shallow(target)
                    .map_err(blocktree::Error::Postings)?;
            }
            Self::Text(c) => {
                c.cursor.advance_shallow(target).map_err(text_error)?;
                c.refresh()?;
            }
        }
        Ok(())
    }

    pub(crate) fn num_levels(&self) -> usize {
        match self {
            Self::Binary(c) => {
                if !c.docs().has_freqs() || c.docs().level1_last_doc_id() == NO_MORE_DOCS {
                    1
                } else {
                    2
                }
            }
            Self::Text(c) => c.levels.len(),
        }
    }

    pub(crate) fn doc_id_up_to(&self, level: usize) -> i32 {
        match self {
            Self::Binary(c) => {
                if !c.docs().has_freqs() {
                    return NO_MORE_DOCS;
                }
                match level {
                    0 => c.docs().level0_last_doc_id(),
                    1 => c.docs().level1_last_doc_id(),
                    _ => NO_MORE_DOCS,
                }
            }
            Self::Text(c) => c.levels.get(level).map_or(NO_MORE_DOCS, |l| l.0),
        }
    }

    pub(crate) fn level_impacts(&mut self, level: usize, out: &mut Vec<Impact>) {
        out.clear();
        match self {
            Self::Binary(c) => {
                let c = c.docs_mut();
                if !c.has_freqs() {
                    out.push(Impact { freq: 1, norm: 1 });
                } else if level == 0 && c.level0_last_doc_id() != NO_MORE_DOCS {
                    out.extend_from_slice(c.level0_impacts());
                } else if level == 1 {
                    out.extend_from_slice(c.level1_impacts());
                } else {
                    out.push(Impact {
                        freq: i32::MAX,
                        norm: 1,
                    });
                }
            }
            Self::Text(c) => {
                if let Some((_, pairs)) = c.levels.get(level) {
                    out.extend_from_slice(pairs);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_codecs::simpletext::{self as text, fields_reader::SimpleTextFieldsReader};
    use std::{fs, path::PathBuf};

    fn root(variant: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/data/codecs_simpletext/postings")
            .join(variant)
    }

    fn unhex(value: &str) -> Vec<u8> {
        value
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
            .collect()
    }

    fn reader(variant: &str) -> SimpleTextFieldsReader {
        read_with_damage(variant, false)
    }

    fn read_with_damage(variant: &str, damaged: bool) -> SimpleTextFieldsReader {
        let root = root(variant);
        let index = root.join("index");
        let si = fs::read_dir(&index)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .find(|path| path.extension().is_some_and(|ext| ext == "si"))
            .unwrap();
        let bytes = fs::read(&si).unwrap();
        let at = 9usize.saturating_add(usize::from(bytes[4]));
        let id: [u8; 16] = bytes[at..at.saturating_add(16)].try_into().unwrap();
        let name = si.file_stem().unwrap().to_str().unwrap();
        let infos = lucene_codecs::field_infos::parse(
            &fs::read(index.join(format!("{name}.fnm"))).unwrap(),
            &id,
            "",
        )
        .unwrap();
        let meta = fs::read_to_string(root.join("meta.tsv")).unwrap();
        let max_doc = meta
            .lines()
            .next()
            .unwrap()
            .split('\t')
            .nth(1)
            .unwrap()
            .parse()
            .unwrap();
        let mut data = fs::read(index.join(format!("{name}_SimpleText_0.pst"))).unwrap();
        if damaged {
            let label = b"        skipDoc ";
            let at = data
                .windows(label.len())
                .position(|bytes| bytes == label)
                .unwrap()
                .saturating_add(label.len());
            data[at] = b'x';
            let footer = data
                .windows(b"checksum ".len())
                .rposition(|bytes| bytes == b"checksum ")
                .unwrap();
            data.truncate(footer);
            text::util::write_checksum(&mut data);
        }
        SimpleTextFieldsReader::open(
            data,
            &lucene_codecs::segment_state::SegmentReadState {
                max_doc,
                segment_id: &id,
                segment_suffix: "SimpleText_0",
                field_infos: &infos,
            },
        )
        .unwrap()
    }

    fn impacts_text(cursor: &mut TermPostings<'_>) -> String {
        let mut output = cursor.num_levels().to_string();
        for level in 0..cursor.num_levels() {
            let end = cursor.doc_id_up_to(level);
            let pairs = cursor
                .impacts(level)
                .iter()
                .map(|i| format!("{}-{}", i.freq, i.norm))
                .collect::<Vec<_>>()
                .join(".");
            output.push_str(&format!("/{end}:{pairs}"));
        }
        output
    }

    #[test]
    fn native_text_impacts_match_every_java_level_and_seek() {
        let mut max_levels = 0;
        for variant in ["flush", "merged"] {
            let reader = reader(variant);
            let manifest = fs::read_to_string(root(variant).join("impacts.tsv")).unwrap();
            for line in manifest.lines() {
                let row: Vec<_> = line.split('\t').collect();
                let terms = reader.terms(row[0]).unwrap().unwrap();
                let mut te = terms.iterator();
                assert!(te.seek_exact(&unhex(row[1])).unwrap());
                let mut cursor =
                    TermPostings::text(te.impacts(text::FREQS).unwrap(), terms.has_freqs())
                        .unwrap();
                assert_eq!(cursor.doc_id(), -1);
                assert_eq!(cursor.freq(), None);
                assert_eq!(format!("i={}", impacts_text(&mut cursor)), row[2]);
                for step in &row[3..] {
                    let (operation, expected) = step.split_once('=').unwrap();
                    let target = operation[1..].parse().unwrap();
                    let got = if operation.starts_with('s') {
                        let doc = cursor.doc_id();
                        let end = cursor.advance_shallow(target).unwrap();
                        assert_eq!(cursor.doc_id(), doc, "shallow movement moved postings");
                        assert_eq!(end, cursor.doc_id_up_to(0));
                        impacts_text(&mut cursor)
                    } else {
                        let doc = cursor.advance(target).unwrap();
                        if doc == NO_MORE_DOCS {
                            assert_eq!(cursor.freq(), None);
                            "-".into()
                        } else {
                            assert_eq!(cursor.advance(doc).unwrap(), doc);
                            assert_eq!(cursor.advance(0).unwrap(), doc);
                            assert_eq!(cursor.doc_id_run_end(), doc.saturating_add(1));
                            assert_eq!(cursor.advance_exact_in_bits(doc), None);
                            format!(
                                "{doc}:{}={}",
                                cursor.freq().unwrap(),
                                impacts_text(&mut cursor)
                            )
                        }
                    };
                    max_levels = max_levels.max(cursor.num_levels());
                    assert_eq!(got, expected, "{variant}: {line}, {step}");
                }
                assert_eq!(cursor.doc_id_up_to(4), NO_MORE_DOCS);
                assert!(cursor.impacts(4).is_empty());
                assert!(matches!(
                    cursor.level0_impacts_view(),
                    Level0Impacts::Decoded(_)
                ));
            }
        }
        assert_eq!(max_levels, 4, "exercise all SimpleText skip levels");
    }

    #[test]
    fn native_text_batches_match_java_documents_and_frequencies() {
        let reader = reader("merged");
        for field in reader.field_names() {
            let terms = reader.terms(field).unwrap().unwrap();
            let manifest =
                fs::read_to_string(root("merged").join(format!("postings.{field}.tsv"))).unwrap();
            for row in manifest.lines().take(25) {
                let row: Vec<_> = row.split('\t').collect();
                let mut te = terms.iterator();
                assert!(te.seek_exact(&unhex(row[0])).unwrap());
                let expected: Vec<(i32, i32)> = row[3..]
                    .iter()
                    .map(|doc| {
                        let mut parts = doc.split(':');
                        (
                            parts.next().unwrap().parse().unwrap(),
                            parts.next().unwrap().parse().unwrap(),
                        )
                    })
                    .collect();
                for limit in [1, 3, 16, usize::MAX] {
                    let slow = SimpleTextImpactsEnum::Slow(te.postings(text::FREQS).unwrap());
                    let mut cursor = TermPostings::text(slow, terms.has_freqs()).unwrap();
                    assert_eq!(cursor.has_freqs(), terms.has_freqs());
                    let mut actual = Vec::new();
                    cursor.next_doc().unwrap();
                    let (mut docs, mut freqs) = (Vec::new(), Vec::new());
                    cursor
                        .next_postings_at_most(0, limit, &mut docs, &mut freqs)
                        .unwrap();
                    assert!(docs.is_empty());
                    loop {
                        cursor
                            .next_postings_at_most(NO_MORE_DOCS, limit, &mut docs, &mut freqs)
                            .unwrap();
                        if docs.is_empty() {
                            break;
                        }
                        assert!(docs.len() <= limit.min(16));
                        actual.extend(docs.iter().copied().zip(freqs.iter().copied()));
                    }
                    assert_eq!(actual, expected, "{field}, {}", row[0]);
                    assert_eq!(cursor.doc_id(), NO_MORE_DOCS);
                    assert_eq!(cursor.next_doc().unwrap(), NO_MORE_DOCS);
                    assert_eq!(cursor.advance(NO_MORE_DOCS).unwrap(), NO_MORE_DOCS);
                }
            }
        }
    }

    #[test]
    fn native_text_skip_errors_reach_the_scorer() {
        let reader = read_with_damage("merged", true);
        let terms = reader.terms("bin").unwrap().unwrap();
        let mut te = terms.iterator();
        assert!(te.next().unwrap().is_some());
        for operation in 0..3 {
            let mut cursor = TermPostings::text(te.impacts(text::FREQS).unwrap(), false).unwrap();
            let result = match operation {
                0 => cursor.advance_shallow(1),
                1 => cursor.advance(1),
                _ => cursor.next_doc(),
            };
            assert!(result.unwrap_err().to_string().contains('x'));
        }
    }

    #[test]
    fn native_text_term_leg_uses_all_four_max_score_levels() {
        use crate::bulk_scorer::TermLeg;
        use crate::similarities::{
            ClassicSimilarity, CollectionStatistics, Similarity, TermStatistics,
        };
        let reader = reader("merged");
        let terms = reader.terms("cat").unwrap().unwrap();
        let mut te = terms.iterator();
        assert!(te.seek_exact(b"c0").unwrap());
        let cost = i64::from(te.doc_freq());
        let score = crate::similarity::do_score(1.0, 1.0, crate::similarity::UNNORMED_NORM_INVERSE);
        let mut cursor = TermPostings::text(te.impacts(text::FREQS).unwrap(), true).unwrap();
        cursor.advance_shallow(41).unwrap();
        assert_eq!(cursor.num_levels(), 4);
        let ends: Vec<_> = (0..4).map(|level| cursor.doc_id_up_to(level)).collect();
        assert_eq!(ends, [57, 57, 153, 500]); // Java's cat:c0 trace.
        let mut leg = TermLeg::scoring(cursor, 1.0, None, cost, 1.0);
        for &end in &ends {
            assert_eq!(leg.max_score(end), score);
            assert_eq!(leg.max_score(end), score); // Cached bound.
        }
        assert_eq!(leg.skip_up_to(score), -1);
        assert_eq!(leg.skip_up_to(score.next_up()), 500);
        leg.set_min_competitive_score(score.next_up());
        let mut doc = leg.impacts_advance(41).unwrap();
        while doc != NO_MORE_DOCS {
            assert!(doc > 500, "all four finite levels must be skipped");
            doc = leg.impacts_next_doc().unwrap();
        }
        let scorer = ClassicSimilarity::default().scorer(
            "cat",
            1.0,
            &CollectionStatistics {
                max_doc: 1461,
                doc_count: 1461,
                sum_total_term_freq: 1461,
                sum_doc_freq: 1461,
            },
            &[TermStatistics {
                doc_freq: cost,
                total_term_freq: cost,
            }],
        );
        let expected = scorer.score(1.0, 1);
        let mut cursor = TermPostings::text(te.impacts(text::FREQS).unwrap(), true).unwrap();
        cursor.advance_shallow(41).unwrap();
        let mut leg = TermLeg::scoring_sim(cursor, scorer, None, cost);
        for &end in &ends {
            assert_eq!(leg.max_score(end), expected);
            assert_eq!(leg.max_score(end), expected);
        }
        assert_eq!(leg.skip_up_to(expected), -1);
        assert_eq!(leg.skip_up_to(expected.next_up()), 500);
    }

    /// The two-phase bulk path must keep Java's postings, scores and top-k
    /// when live documents, window ends and published thresholds interact.
    #[test]
    // ARITH: all document arithmetic is bounded by this fixture's 1461 docs;
    // the predicate's divisor is the nonzero constant 3.
    #[allow(clippy::arithmetic_side_effects)]
    fn native_text_two_phase_windows_match_filtered_java_postings() {
        use crate::bulk_scorer::{score_term_two_phase_window, TermLeg, TwoPhaseRun};
        use crate::collector::{ScoringCollector, TopDocsCollector};
        use crate::exec::Scorer;
        use lucene_util::fixed_bit_set::FixedBitSet;

        struct Predicate {
            doc: i32,
            max_doc: i32,
            reject_all: bool,
            batches: usize,
        }
        impl Scorer for Predicate {
            fn doc_id(&self) -> i32 {
                self.doc
            }
            fn next_doc(&mut self) -> Result<i32> {
                self.advance(self.doc.saturating_add(1))
            }
            fn advance(&mut self, target: i32) -> Result<i32> {
                self.doc = if target >= self.max_doc {
                    NO_MORE_DOCS
                } else {
                    target
                };
                Ok(self.doc)
            }
            fn cost(&self) -> i64 {
                i64::from(self.max_doc)
            }
            fn two_phase(&self) -> bool {
                true
            }
            fn matches(&mut self) -> Result<bool> {
                Ok(!self.reject_all && self.doc % 3 != 1)
            }
            fn score(&mut self) -> Result<f32> {
                panic!("a filter must never be scored")
            }
            fn max_score(&mut self, _: i32) -> Result<f32> {
                Ok(0.0)
            }
            fn batch_matches(&self) -> bool {
                true
            }
            fn matches_batch(&mut self, docs: &[i32], keep: &mut [bool]) -> Result<()> {
                self.batches += 1;
                for (&doc, keep) in docs.iter().zip(keep) {
                    self.advance(doc)?;
                    *keep = self.matches()?;
                }
                Ok(())
            }
        }

        let reader = reader("merged");
        let max_doc = 1461;
        let mut mixed = FixedBitSet::new(max_doc as usize);
        for doc in 0..max_doc {
            if doc % 5 != 0 {
                mixed.set(doc as usize);
            }
        }
        let empty = FixedBitSet::new(max_doc as usize);
        for (field, term) in [("cat", b"c0".as_slice()), ("tag", b"t0"), ("body", b"a")] {
            let terms = reader.terms(field).unwrap().unwrap();
            let mut te = terms.iterator();
            assert!(te.seek_exact(term).unwrap());
            let cost = i64::from(te.doc_freq());
            let manifest =
                fs::read_to_string(root("merged").join(format!("postings.{field}.tsv"))).unwrap();
            let row: Vec<_> = manifest
                .lines()
                .find(|row| unhex(row.split('\t').next().unwrap()) == term)
                .unwrap()
                .split('\t')
                .collect();
            let postings: Vec<(i32, f32)> = row[3..]
                .iter()
                .map(|posting| {
                    let mut parts = posting.split(':');
                    let doc = parts.next().unwrap().parse().unwrap();
                    let freq = parts.next().unwrap().parse::<f32>().unwrap();
                    (
                        doc,
                        crate::similarity::do_score(
                            1.0,
                            freq,
                            crate::similarity::UNNORMED_NORM_INVERSE,
                        ),
                    )
                })
                .collect();
            for live in [None, Some(&mixed), Some(&empty)] {
                for reject_all in [false, true] {
                    for exhaustive in [false, true] {
                        for start in [0, 43] {
                            let threshold = if exhaustive { u64::MAX } else { 0 };
                            let mut expected =
                                TopDocsCollector::with_total_hits_threshold(3, threshold);
                            for &(doc, score) in &postings {
                                if doc >= start
                                    && !reject_all
                                    && doc % 3 != 1
                                    && live.is_none_or(|bits| bits.get_doc(doc))
                                {
                                    expected.collect(doc, score);
                                }
                            }
                            let cursor = TermPostings::text(
                                te.impacts(text::FREQS).unwrap(),
                                terms.has_freqs(),
                            )
                            .unwrap();
                            let mut leg = TermLeg::scoring(cursor, 1.0, None, cost, f32::MAX);
                            let mut filter = Predicate {
                                doc: -1,
                                max_doc,
                                reject_all,
                                batches: 0,
                            };
                            let mut run = TwoPhaseRun::default();
                            let mut actual =
                                TopDocsCollector::with_total_hits_threshold(3, threshold);
                            let mut min = start;
                            for max in [44, 50, 501, NO_MORE_DOCS] {
                                let next = score_term_two_phase_window(
                                    &mut leg,
                                    &mut filter,
                                    &mut run,
                                    live,
                                    &mut actual,
                                    min,
                                    max,
                                )
                                .unwrap();
                                assert!(next >= max);
                                min = max;
                            }
                            let bits = |collector: &TopDocsCollector| {
                                collector
                                    .top_docs()
                                    .iter()
                                    .map(|hit| (hit.doc_id, hit.score.to_bits()))
                                    .collect::<Vec<_>>()
                            };
                            assert_eq!(
                                bits(&actual),
                                bits(&expected),
                                "{field}, start {start}, exhaustive {exhaustive}, reject {reject_all}"
                            );
                            if exhaustive {
                                assert_eq!(actual.total_hits(), expected.total_hits());
                            }
                            if live.is_some_and(|bits| std::ptr::eq(bits, &empty)) {
                                assert_eq!(filter.batches, 0);
                            } else {
                                assert!(filter.batches > 0);
                            }
                        }
                    }
                }
            }
        }
    }
}
