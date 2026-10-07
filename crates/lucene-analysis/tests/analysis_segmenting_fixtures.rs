//! M12 T12.6: the JDK's sentence `BreakIterator` and
//! `SegmentingTokenizerBase` against `fixtures/src/GenAnalysisSegmenting.java`.

mod support;

use lucene_analysis::util::sentence_break::{class_of, following_boundary, Class};
use lucene_analysis::util::{Segmenter, SegmentingBase, SegmentingTokenizer};
use lucene_analysis::{java_character, AnalysisError, Analyzer};
use support::{chain, comps, unesc};

const CLASSES: [Class; 13] = [
    Class::Other,
    Class::Space,
    Class::Term,
    Class::Close,
    Class::Quote,
    Class::Digit,
    Class::Period,
    Class::Upper,
    Class::Lower,
    Class::Danda,
    Class::Para,
    Class::Ignore,
    Class::Done,
];

fn read(file: &str) -> String {
    std::fs::read_to_string(support::data_dir("analysis_segmenting") + file).unwrap()
}

/// Every code point both JDKs class alike has the JDK's class.
#[test]
fn every_code_point_has_the_jdks_class() {
    let mut checked = 0u32;
    for line in read("sentence_classes.txt").lines() {
        let f: Vec<&str> = line.split(' ').collect();
        let lo = u32::from_str_radix(f[0], 16).unwrap();
        let hi = u32::from_str_radix(f[1], 16).unwrap();
        let class = CLASSES[f[2].parse::<usize>().unwrap()];
        for cp in lo..=hi {
            assert_eq!(class_of(cp), class, "U+{cp:04X}");
            checked += 1;
        }
    }
    assert!(checked > 1_100_000, "{checked}");
}

/// The texts and 20,000 seeded strings over every class: every boundary.
#[test]
fn sentence_boundaries_match_the_jdk() {
    let mut n = 0;
    for line in read("sentences.txt").lines() {
        let (units, bounds) = line.split_once('\t').unwrap();
        let text: Vec<u16> = units
            .split(' ')
            .filter(|u| !u.is_empty())
            .map(|u| u16::from_str_radix(u, 16).unwrap())
            .collect();
        let expected: Vec<usize> = bounds
            .split(' ')
            .filter(|b| !b.is_empty())
            .map(|b| b.parse().unwrap())
            .collect();
        let mut actual = Vec::new();
        let mut pos = 0;
        while let Some(b) = following_boundary(&text, pos) {
            actual.push(b);
            pos = b;
        }
        assert_eq!(actual, expected, "{units}");
        n += 1;
    }
    assert_eq!(n, 20_009);
}

/// The generator's `WholeSentenceTokenizer`.
#[derive(Default)]
struct WholeSentence {
    bounds: Option<(usize, usize)>,
}

impl Segmenter for WholeSentence {
    fn set_next_sentence(
        &mut self,
        _: &SegmentingBase,
        start: usize,
        end: usize,
    ) -> Result<(), AnalysisError> {
        self.bounds = Some((start, end));
        Ok(())
    }
    fn increment_word(&mut self, base: &mut SegmentingBase) -> Result<bool, AnalysisError> {
        let Some((start, end)) = self.bounds.take() else {
            return Ok(false);
        };
        emit(base, start, end, None)
    }
}

/// The generator's `SentenceAndWordTokenizer`.
struct SentenceAndWord {
    sentence_end: usize,
    word_end: usize,
    pos_boost: i32,
}

impl Default for SentenceAndWord {
    fn default() -> Self {
        SentenceAndWord {
            sentence_end: 0,
            word_end: 0,
            pos_boost: -1,
        }
    }
}

impl Segmenter for SentenceAndWord {
    fn set_next_sentence(
        &mut self,
        _: &SegmentingBase,
        start: usize,
        end: usize,
    ) -> Result<(), AnalysisError> {
        self.word_end = start;
        self.sentence_end = end;
        self.pos_boost += 10;
        Ok(())
    }
    fn increment_word(&mut self, base: &mut SegmentingBase) -> Result<bool, AnalysisError> {
        let word = |u: u16| java_character::is_letter_or_digit(u32::from(u));
        let mut start = self.word_end;
        while start < self.sentence_end && !word(base.buffer()[start]) {
            start += 1;
        }
        if start == self.sentence_end {
            return Ok(false);
        }
        let mut end = start + 1;
        while end < self.sentence_end && word(base.buffer()[end]) {
            end += 1;
        }
        self.word_end = end;
        let boost = std::mem::replace(&mut self.pos_boost, 0);
        emit(base, start, end, Some(boost))
    }
    fn reset(&mut self) {
        *self = Self::default();
    }
}

fn emit(
    base: &mut SegmentingBase,
    start: usize,
    end: usize,
    boost: Option<i32>,
) -> Result<bool, AnalysisError> {
    let (s, e) = (
        base.correct_offset(base.offset() + start as i32),
        base.correct_offset(base.offset() + end as i32),
    );
    let term = base.buffer()[start..end].to_vec();
    let a = base.attributes_mut();
    a.clear_attributes();
    a.set_term_utf16(&term);
    a.set_offset(s, e)?;
    if let Some(b) = boost {
        let inc = a.position_increment() + b;
        a.set_position_increment(inc)?;
    }
    Ok(true)
}

fn build(name: &str) -> Option<Analyzer> {
    Some(match name {
        "whole_sentence" => chain(|| comps(SegmentingTokenizer::new(WholeSentence::default()))),
        "sentence_words" => chain(|| comps(SegmentingTokenizer::new(SentenceAndWord::default()))),
        _ => return None,
    })
}

#[test]
fn segmenting_tokenizers_match_lucene() {
    let texts: Vec<String> = read("texts.txt").lines().map(unesc).collect();
    assert!(texts.iter().any(|t| t.encode_utf16().count() > 3 * 1024));
    let names = support::fixture_names("analysis_segmenting");
    assert_eq!(
        support::check_chains_over("analysis_segmenting", &texts, &names, build),
        2
    );
}
