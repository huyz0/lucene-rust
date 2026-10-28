//! `opennlp.tools.sentdetect`: `SentenceDetectorME.sentPosDetect` over a
//! loaded `SentenceModel` -- `DefaultEndOfSentenceScanner` finds the
//! candidate end-of-sentence characters, `DefaultSDContextGenerator`
//! describes each, the maxent model decides (`s` splits), and
//! `isAcceptableBreak` vetoes a break inside an abbreviation.

use std::sync::Arc;

use lucene_analysis::factory::FactoryError;

use super::dictionary::Dictionary;
use super::ml::MaxentModel;
use super::model::{BaseModel, Component};
use super::text::{self, is_whitespace, JString};

/// `opennlp.tools.util.Span`'s offsets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    /// The start.
    pub start: usize,
    /// The end (exclusive).
    pub end: usize,
}

impl Span {
    /// `Span.trim(text)`: white space (`StringUtil.isWhitespace`) removed
    /// from both ends.
    // ARITH: offsets within the span.
    #[allow(clippy::arithmetic_side_effects)]
    fn trim(self, s: &[u16]) -> Span {
        let mut start = self.start;
        while start < self.end && is_whitespace(s[start]) {
            start += 1;
        }
        let mut end = self.end;
        while end > self.start && is_whitespace(s[end - 1]) {
            end -= 1;
        }
        if start > end {
            Span {
                start: self.start,
                end: self.start,
            }
        } else {
            Span { start, end }
        }
    }
}

/// `opennlp.tools.sentdetect.lang.Factory.defaultEosCharacters`.
const DEFAULT_EOS: [u16; 3] = [b'.' as u16, b'!' as u16, b'?' as u16];

/// A loaded `SentenceModel`.
#[derive(Debug, Clone)]
pub struct SentenceModel {
    model: Arc<MaxentModel>,
    abbreviations: Option<Arc<Dictionary>>,
    eos: JString,
    use_token_end: bool,
}

impl SentenceModel {
    /// `new SentenceModel(InputStream)`.
    pub fn load(bytes: &[u8]) -> Result<SentenceModel, FactoryError> {
        let base = BaseModel::load(Component::SentenceDetector, bytes)?;
        // getEOSCharacters: the manifest's, else (the factory's language
        // code being unset for a loaded model) the default ones
        let eos = match base.manifest_property("eosCharacters") {
            Some(e) if !e.is_empty() => text::units(e),
            _ => DEFAULT_EOS.to_vec(),
        };
        let use_token_end = base
            .manifest_property("useTokenEnd")
            .is_some_and(text::parse_boolean);
        Ok(SentenceModel {
            model: base.maxent_model(),
            abbreviations: base.dictionary("abbreviations.dictionary"),
            eos,
            use_token_end,
        })
    }
}

/// `SentenceDetectorME`.
#[derive(Debug, Clone)]
pub struct SentenceDetector {
    model: SentenceModel,
    /// Per outcome, whether its name is `SentenceDetectorME.SPLIT` (Java compares the best
    /// outcome's *name*: a model without the outcome never splits, and a
    /// repeated name splits on either index).
    split: Vec<bool>,
    probs: Vec<f64>,
    features: Vec<JString>,
}

impl SentenceDetector {
    /// `new SentenceDetectorME(model)`.
    pub fn new(model: SentenceModel) -> Self {
        let split = (0..model.model.num_outcomes())
            .map(|i| model.model.outcome_name(i) == "s")
            .collect();
        let probs = vec![0.0; model.model.num_outcomes()];
        SentenceDetector {
            model,
            split,
            probs,
            features: Vec::new(),
        }
    }

    fn is_eos(&self, c: u16) -> bool {
        self.model.eos.contains(&c)
    }

    /// `sentPosDetect(s)`: the sentences' spans.
    // ARITH: positions within the text plus one.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn sent_pos_detect(&mut self, s: &[u16]) -> Vec<Span> {
        let first_ws = |mut pos: usize| {
            while pos < s.len() && !is_whitespace(s[pos]) {
                pos += 1;
            }
            pos
        };
        let first_non_ws = |mut pos: usize| {
            while pos < s.len() && is_whitespace(s[pos]) {
                pos += 1;
            }
            pos
        };
        let enders: Vec<usize> = (0..s.len()).filter(|&i| self.is_eos(s[i])).collect();
        let mut positions: Vec<usize> = Vec::with_capacity(enders.len());
        let mut index = 0;
        for (i, &cint) in enders.iter().enumerate() {
            let fws = first_ws(cint + 1);
            if i + 1 < enders.len() && enders[i + 1] < fws {
                continue;
            }
            if positions.last().is_some_and(|&last| cint < last) {
                continue;
            }
            self.features.clear();
            self.context(s, cint);
            self.model.model.eval(&self.features, &mut self.probs);
            let best = MaxentModel::best_outcome(&self.probs);
            if self.split[best] && self.is_acceptable_break(s, index, cint) {
                if index != cint {
                    positions.push(if self.model.use_token_end {
                        first_non_ws(first_ws(cint + 1))
                    } else {
                        first_non_ws(cint + 1)
                    });
                }
                index = cint + 1;
            }
        }
        if positions.is_empty() {
            let span = Span {
                start: 0,
                end: s.len(),
            }
            .trim(s);
            return if span.end > span.start {
                vec![span]
            } else {
                Vec::new()
            };
        }
        let mut spans = Vec::with_capacity(positions.len() + 1);
        let mut start = 0;
        for &p in &positions {
            // never empty: it holds the end-of-sentence character at
            // `cint`, which no earlier position passes
            spans.push(Span { start, end: p }.trim(s));
            start = p;
        }
        let last = positions[positions.len() - 1];
        if last != s.len() {
            spans.push(
                Span {
                    start: last,
                    end: s.len(),
                }
                .trim(s),
            );
        }
        spans
    }

    /// `isAcceptableBreak(s, fromIndex, candidateIndex)`: no abbreviation
    /// found from `from` covers the candidate. An abbreviation that is not
    /// found at all counts as at -1 (`indexOf`), so a candidate within its
    /// length of the start is refused, as in Java.
    // ARITH: positions and lengths of the in-memory text, far below `isize::MAX`.
    #[allow(clippy::arithmetic_side_effects)]
    fn is_acceptable_break(&self, s: &[u16], from: usize, candidate: usize) -> bool {
        let Some(dict) = &self.model.abbreviations else {
            return true;
        };
        let candidate = candidate as isize;
        for token in dict.first_tokens() {
            let position = text::index_of(s, token, from).map_or(-1, |p| p as isize);
            if position + (token.len() as isize) < candidate || position > candidate {
                continue;
            }
            return false;
        }
        true
    }

    /// `DefaultSDContextGenerator.getContext(sb, position)` into
    /// `self.features`.
    // ARITH: positions within the text (`position < len`).
    #[allow(clippy::arithmetic_side_effects)]
    fn context(&mut self, sb: &[u16], position: usize) {
        let last_index = sb.len() - 1;
        let feats = &mut self.features;
        if position > 0 && is_whitespace(sb[position - 1]) {
            feats.push(text::units("sp"));
        }
        if position < last_index && is_whitespace(sb[position + 1]) {
            feats.push(text::units("sn"));
        }
        let eos_char = sb[position];
        feats.push(match eos_char {
            0x0A => text::units("eos=<LF>"),
            0x0D => text::units("eos=<CR>"),
            c => text::concat(&[&text::units("eos="), &[c]]),
        });
        let mut prefix_start = previous_space_index(sb, position);
        let mut c = position;
        while c > prefix_start + 1 {
            c -= 1;
            if self.model.eos.contains(&sb[c]) {
                prefix_start = c;
                break;
            }
        }
        let prefix = text::trim(&sb[prefix_start..position]);
        let prev_start = previous_space_index(sb, prefix_start);
        let previous = text::trim(&sb[prev_start..prefix_start]);
        let mut suffix_end = next_space_index(sb, position, last_index);
        let mut c = position + 1;
        while c < suffix_end {
            if self.model.eos.contains(&sb[c]) {
                suffix_end = c;
                break;
            }
            c += 1;
        }
        let next_end = next_space_index(sb, suffix_end + 1, last_index + 1);
        let (suffix, next): (&[u16], &[u16]) = if position == last_index {
            (&[], &[])
        } else {
            (
                text::trim(&sb[position + 1..suffix_end]),
                text::trim(&sb[(suffix_end + 1).min(next_end)..next_end]),
            )
        };
        let abbreviation = |s: &[u16]| {
            self.model
                .abbreviations
                .as_ref()
                .is_some_and(|d| d.contains_string(s))
        };
        let feats = &mut self.features;
        // collectFeatures
        feats.push(text::concat(&[&text::units("x="), prefix]));
        if !prefix.is_empty() {
            feats.push(text::int_units(prefix.len()));
            if text::is_upper_case(prefix[0]) {
                feats.push(text::units("xcap"));
            }
            if abbreviation(&text::concat(&[prefix, &[eos_char]])) {
                feats.push(text::units("xabbrev"));
            }
        }
        for (key, value, cap, abbrev) in [
            ("v=", previous, "vcap", "vabbrev"),
            ("s=", suffix, "scap", "sabbrev"),
            ("n=", next, "ncap", "nabbrev"),
        ] {
            feats.push(text::concat(&[&text::units(key), value]));
            if !value.is_empty() {
                if text::is_upper_case(value[0]) {
                    feats.push(text::units(cap));
                }
                if abbreviation(value) {
                    feats.push(text::units(abbrev));
                }
            }
        }
    }
}

/// `DefaultSDContextGenerator.previousSpaceIndex`.
// ARITH: positions within the text, `seek > 0` before stepping back.
#[allow(clippy::arithmetic_side_effects)]
fn previous_space_index(sb: &[u16], seek: usize) -> usize {
    let mut seek = seek as isize;
    loop {
        seek -= 1;
        if !(seek > 0 && !is_whitespace(sb[seek as usize])) {
            break;
        }
    }
    if seek > 0 && is_whitespace(sb[seek as usize]) {
        let mut seek = seek as usize;
        while seek > 0 && is_whitespace(sb[seek - 1]) {
            seek -= 1;
        }
        return seek;
    }
    0
}

/// `DefaultSDContextGenerator.nextSpaceIndex`.
// ARITH: positions within the text plus one.
#[allow(clippy::arithmetic_side_effects)]
fn next_space_index(sb: &[u16], seek: usize, last_index: usize) -> usize {
    let mut seek = seek + 1;
    while seek < last_index {
        if is_whitespace(sb[seek]) {
            while sb.len() > seek + 1 && is_whitespace(sb[seek + 1]) {
                seek += 1;
            }
            return seek;
        }
        seek += 1;
    }
    last_index
}
