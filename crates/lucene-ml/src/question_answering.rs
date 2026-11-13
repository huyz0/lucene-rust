//! ml-commons' `QUESTION_ANSWERING` function for the sentence highlighting
//! model (`model_type: sentence_highlighting`, `QuestionAnsweringModel` with
//! its `SentenceHighlightingQATranslator`): the context is cut into
//! sentences, the question and context are tokenized as a pair, each context
//! token is tagged with the sentence its word belongs to, and the model
//! answers with the indices of the relevant sentences.
//!
//! A context longer than the token limit is read in chunks: the main
//! encoding, then each of its overflowing encodings, and a sentence found
//! relevant in any chunk is a highlight.
//!
//! Offsets are UTF-16 code units, as Java's `String` indexes them.

use tokenizers::{
    PaddingParams, PaddingStrategy, Tokenizer, TruncationDirection, TruncationParams,
    TruncationStrategy,
};

use crate::torchscript::{Data, Input, TorchScript};
use crate::{MlError, Result};

/// The marker for a token no sentence owns.
const IGNORE_TOKEN_ID: i64 = -100;

/// `DefaultSentenceSegmenter`'s abbreviations, whose dot does not end a
/// sentence.
const ABBREVIATIONS: [&str; 15] = [
    "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "Ltd.", "Co.", "Inc.", "i.e.", "e.g.",
    "etc.", "vs.", "fig.",
];

/// A sentence of the context: its text (trimmed) and where it sits in the
/// context, in UTF-16 units.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sentence {
    /// The sentence, trimmed.
    pub text: String,
    /// Where it starts in the context.
    pub start: usize,
    /// Where it ends in the context.
    pub end: usize,
}

/// A relevant sentence, as the model's answer lists it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Highlight {
    /// The sentence's text.
    pub text: String,
    /// Its index among the context's sentences.
    pub position: usize,
    /// Where it starts in the context, in UTF-16 units.
    pub start: usize,
    /// Where it ends in the context, in UTF-16 units.
    pub end: usize,
}

/// How the tokenizer cuts a long context, from the model's configuration
/// (`token_max_length`, `token_overlap_stride`, `with_overflowing_tokens`,
/// `padding`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Config {
    /// The most tokens of one chunk.
    pub max_length: usize,
    /// The tokens a chunk shares with the one before it.
    pub stride: usize,
    /// Whether a long context is read past its first chunk.
    pub overflowing: bool,
    /// Whether a chunk is padded.
    pub padding: bool,
}

impl Default for Config {
    fn default() -> Config {
        Config {
            max_length: 512,
            stride: 128,
            overflowing: true,
            padding: false,
        }
    }
}

/// A deployed sentence highlighting model.
pub struct HighlighterModel {
    tokenizer: Tokenizer,
    module: Box<dyn TorchScript>,
    overflowing: bool,
}

impl HighlighterModel {
    /// A TorchScript highlighter, already loaded as `module`, with the
    /// tokenizer file `tokenizer_json`.
    pub fn torchscript(
        module: Box<dyn TorchScript>,
        tokenizer_json: &[u8],
        config: Config,
    ) -> Result<HighlighterModel> {
        let mut tokenizer =
            Tokenizer::from_bytes(tokenizer_json).map_err(|e| MlError::Tokenizer(e.to_string()))?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                direction: TruncationDirection::Right,
                max_length: config.max_length,
                strategy: TruncationStrategy::OnlySecond,
                stride: config.stride,
            }))
            .map_err(|e| MlError::Tokenizer(e.to_string()))?;
        let padding = config.padding.then(|| PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            ..tokenizer.get_padding().cloned().unwrap_or_default()
        });
        tokenizer.with_padding(padding);
        Ok(HighlighterModel {
            tokenizer,
            module,
            overflowing: config.overflowing,
        })
    }

    /// The sentences of `context` that answer `question`, in context order.
    pub fn highlight(&self, question: &str, context: &str) -> Result<Vec<Highlight>> {
        let sentences = segment(context);
        let words = word_sentence_ids(&sentences, context);
        let encoding = self
            .tokenizer
            .encode((question, context), true)
            .map_err(|e| MlError::Tokenizer(e.to_string()))?;
        let chunks = if self.overflowing {
            encoding.get_overflowing().len()
        } else {
            0
        };
        let mut relevant = vec![false; sentences.len()];
        for chunk in 0..=chunks {
            let e = match chunk.checked_sub(1) {
                None => &encoding,
                Some(i) => &encoding.get_overflowing()[i],
            };
            let widen = |v: &[u32]| v.iter().map(|&x| i64::from(x)).collect::<Vec<_>>();
            let ids = widen(e.get_ids());
            let mask = widen(e.get_attention_mask());
            let types = widen(e.get_type_ids());
            let tagged = token_sentence_ids(&types, e.get_word_ids(), &words);
            let outputs = self.module.forward(&[
                Input {
                    name: "input_ids",
                    values: &ids,
                },
                Input {
                    name: "attention_mask",
                    values: &mask,
                },
                Input {
                    name: "token_type_ids",
                    values: &types,
                },
                Input {
                    name: "sentence_ids",
                    values: &tagged,
                },
            ])?;
            for o in outputs {
                let values: Vec<i64> = match o.data {
                    Data::I64(v) => v,
                    #[allow(clippy::cast_possible_truncation)]
                    Data::F32(v) => v.into_iter().map(|x| x as i64).collect(),
                };
                for v in values {
                    if let Ok(i) = usize::try_from(v) {
                        if let Some(r) = relevant.get_mut(i) {
                            *r = true;
                        }
                    }
                }
            }
        }
        Ok(sentences
            .into_iter()
            .enumerate()
            .filter(|(i, _)| relevant[*i])
            .map(|(position, s)| Highlight {
                text: s.text,
                position,
                start: s.start,
                end: s.end,
            })
            .collect())
    }
}

/// Each token's sentence: `-100` for the question, the specials and any
/// token past the words, else the sentence of the whitespace-separated word
/// whose number is the token's word id.
fn token_sentence_ids(types: &[i64], word_ids: &[Option<u32>], words: &[i64]) -> Vec<i64> {
    let mut out = vec![IGNORE_TOKEN_ID; types.len()];
    let start = types.iter().position(|&t| t == 1).unwrap_or(types.len());
    for (slot, word_id) in out.iter_mut().zip(word_ids).skip(start) {
        let sentence = word_id
            .and_then(|w| usize::try_from(w).ok())
            .and_then(|w| words.get(w));
        if let Some(&s) = sentence {
            *slot = s;
        }
    }
    out
}

/// For each piece of `context` split on whitespace (as Java's `split`
/// returns them), the sentence that holds it; a piece no sentence holds
/// belongs to the first.
fn word_sentence_ids(sentences: &[Sentence], context: &str) -> Vec<i64> {
    let units: Vec<u16> = context.encode_utf16().collect();
    let lengths = split_lengths(&units, is_space);
    let mut ids = vec![0_i64; lengths.len()];
    let mut starts = Vec::with_capacity(lengths.len());
    let mut at = 0_usize;
    for &len in &lengths {
        starts.push(at);
        at = at.saturating_add(len).saturating_add(1);
    }
    for (n, s) in sentences.iter().enumerate() {
        for (w, &len) in lengths.iter().enumerate() {
            let start = starts[w];
            if start >= s.start && start.saturating_add(len) <= s.end {
                ids[w] = i64::try_from(n).unwrap_or(i64::MAX);
            }
        }
    }
    ids
}

/// The lengths of the pieces `units.split(regex-of-runs-of-delim)` returns
/// in Java: a leading empty piece stays, trailing empty ones go, and text
/// without a delimiter is one piece.
// ARITH: every index is below the slice's length, which fits in usize with room
// for the one-past steps taken here.
#[allow(clippy::arithmetic_side_effects)]
fn split_lengths(units: &[u16], delim: fn(u16) -> bool) -> Vec<usize> {
    let mut pieces = Vec::new();
    let mut piece_start = 0_usize;
    let mut i = 0_usize;
    let mut matched = false;
    while i < units.len() {
        if delim(units[i]) {
            let run_start = i;
            while i < units.len() && delim(units[i]) {
                i += 1;
            }
            pieces.push(run_start - piece_start);
            piece_start = i;
            matched = true;
        } else {
            i += 1;
        }
    }
    if !matched {
        return vec![units.len()];
    }
    pieces.push(units.len() - piece_start);
    while pieces.last() == Some(&0) {
        pieces.pop();
    }
    pieces
}

/// Java's `\s`.
fn is_space(u: u16) -> bool {
    matches!(u, 0x20 | 0x09 | 0x0A | 0x0B | 0x0C | 0x0D)
}

/// Java's `\w`.
fn is_word(u: u16) -> bool {
    u == 0x5F
        || (0x30..=0x39).contains(&u)
        || (0x41..=0x5A).contains(&u)
        || (0x61..=0x7A).contains(&u)
}

fn is_upper(u: u16) -> bool {
    (0x41..=0x5A).contains(&u)
}

/// Splits `context` into sentences as `DefaultSentenceSegmenter` does with
/// its default configuration (abbreviations and quotes handled, whitespace
/// not preserved).
pub fn segment(context: &str) -> Vec<Sentence> {
    let original: Vec<u16> = context.encode_utf16().collect();
    let handled = handle_quotes(&handle_abbreviations(&original));
    let mut sentences = Vec::new();
    let mut cursor = 0_usize;
    for piece in split_sentences(&handled) {
        let start = match index_of(&original, piece, cursor) {
            Some(i) => i,
            None => approximate_position(&original, piece, cursor),
        };
        let end = start.saturating_add(piece.len());
        sentences.push(Sentence {
            text: String::from_utf16_lossy(trim(piece)),
            start,
            end,
        });
        cursor = end;
    }
    sentences
}

/// Java's `String.trim`.
// ARITH: every index is below the slice's length, which fits in usize with room
// for the one-past steps taken here.
#[allow(clippy::arithmetic_side_effects)]
fn trim(s: &[u16]) -> &[u16] {
    let from = s.iter().position(|&u| u > 0x20).unwrap_or(s.len());
    let to = s.iter().rposition(|&u| u > 0x20).map_or(from, |i| i + 1);
    &s[from..to]
}

/// Java's `String.indexOf(needle, from)`.
// ARITH: every index is below the slice's length, which fits in usize with room
// for the one-past steps taken here.
#[allow(clippy::arithmetic_side_effects)]
fn index_of(haystack: &[u16], needle: &[u16], from: usize) -> Option<usize> {
    if needle.is_empty() {
        return Some(from.min(haystack.len()));
    }
    if from >= haystack.len() {
        return None;
    }
    haystack[from..]
        .windows(needle.len())
        .position(|w| w == needle)
        .map(|i| i + from)
}

/// `findApproximatePosition`: where the piece's first ten units sit at or
/// after `cursor`, else `cursor`.
fn approximate_position(text: &[u16], piece: &[u16], cursor: usize) -> usize {
    if piece.is_empty() {
        return cursor;
    }
    index_of(text, &piece[..piece.len().min(10)], cursor).unwrap_or(cursor)
}

/// Each abbreviation, followed by whitespace and preceded by a word
/// boundary, has its dots turned to `@@@` and its whitespace to one space, in the list's
/// order.
// ARITH: every index is below the slice's length, which fits in usize with room
// for the one-past steps taken here.
#[allow(clippy::arithmetic_side_effects)]
fn handle_abbreviations(text: &[u16]) -> Vec<u16> {
    let mut text = text.to_vec();
    for abbreviation in ABBREVIATIONS {
        let needle: Vec<u16> = abbreviation.encode_utf16().collect();
        let replacement: Vec<u16> = format!("{} ", abbreviation.replace('.', "@@@"))
            .encode_utf16()
            .collect();
        let mut out = Vec::with_capacity(text.len());
        let mut changed = false;
        let mut i = 0_usize;
        while i < text.len() {
            let end = i + needle.len();
            let boundary = i == 0 || !is_word(text[i - 1]);
            if boundary && end < text.len() && text[i..end] == needle[..] && is_space(text[end]) {
                let mut after = end;
                while after < text.len() && is_space(text[after]) {
                    after += 1;
                }
                out.extend_from_slice(&replacement);
                changed = true;
                i = after;
            } else {
                out.push(text[i]);
                i += 1;
            }
        }
        if changed {
            text = out;
        }
    }
    text
}

/// `([.!?])"` becomes `"\1`.
// ARITH: every index is below the slice's length, which fits in usize with room
// for the one-past steps taken here.
#[allow(clippy::arithmetic_side_effects)]
fn handle_quotes(text: &[u16]) -> Vec<u16> {
    let mut out = Vec::with_capacity(text.len());
    let mut i = 0_usize;
    while i < text.len() {
        if matches!(text[i], 0x2E | 0x21 | 0x3F) && text.get(i + 1) == Some(&0x22) {
            out.push(0x22);
            out.push(text[i]);
            i += 2;
        } else {
            out.push(text[i]);
            i += 1;
        }
    }
    out
}

/// Splits at whitespace that follows `.`, `!` or `?`, unless the text
/// before it ends like `a.b.`, `aB.` or `aB.C.`.
// ARITH: every index is below the slice's length, which fits in usize with room
// for the one-past steps taken here.
#[allow(clippy::arithmetic_side_effects)]
fn split_sentences(text: &[u16]) -> Vec<&[u16]> {
    let mut pieces = Vec::new();
    let mut piece_start = 0_usize;
    let mut i = 1_usize;
    let mut matched = false;
    while i < text.len() {
        if is_space(text[i])
            && matches!(text[i - 1], 0x2E | 0x21 | 0x3F)
            && !abbreviation_end(text, i)
        {
            let split = i;
            while i < text.len() && is_space(text[i]) {
                i += 1;
            }
            pieces.push(&text[piece_start..split]);
            piece_start = i;
            matched = true;
        } else {
            i += 1;
        }
    }
    if !matched {
        return vec![text];
    }
    pieces.push(&text[piece_start..]);
    while pieces.last().is_some_and(|p| p.is_empty()) {
        pieces.pop();
    }
    pieces
}

/// Whether the units before `at` match one of the sentence pattern's
/// negative lookbehinds: `\w\.\w.`, `\w[A-Z]\.` or `\w[A-Z]\.[A-Z]\.`.
fn abbreviation_end(text: &[u16], at: usize) -> bool {
    let before = |n: usize| at.checked_sub(n).map(|i| text[i]);
    let dotted = matches!(
        (before(4), before(3), before(2)),
        (Some(a), Some(0x2E), Some(c)) if is_word(a) && is_word(c)
    );
    let initial = matches!(
        (before(3), before(2), before(1)),
        (Some(a), Some(b), Some(0x2E)) if is_word(a) && is_upper(b)
    );
    let initials = matches!(
        (before(5), before(4), before(3), before(2), before(1)),
        (Some(a), Some(b), Some(0x2E), Some(d), Some(0x2E)) if is_word(a) && is_upper(b) && is_upper(d)
    );
    dotted || initial || initials
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts(context: &str) -> Vec<String> {
        segment(context).into_iter().map(|s| s.text).collect()
    }

    #[test]
    fn splits_after_sentence_ends() {
        let s = segment("The sun is out. Is it warm? Yes!");
        assert_eq!(
            texts("The sun is out. Is it warm? Yes!"),
            vec!["The sun is out.", "Is it warm?", "Yes!"]
        );
        assert_eq!((s[0].start, s[0].end), (0, 15));
        assert_eq!((s[1].start, s[1].end), (16, 27));
    }

    #[test]
    fn keeps_abbreviations_inside_a_sentence_and_marks_them() {
        assert_eq!(
            texts("Dr. Smith came. He left."),
            vec!["Dr@@@ Smith came.", "He left."]
        );
        assert_eq!(texts("See e.g. this one."), vec!["See e@@@g@@@ this one."]);
    }

    #[test]
    fn keeps_initials_and_dotted_words_together() {
        assert_eq!(
            texts("It was J.R. Smith. Done."),
            vec!["It was J.R. Smith.", "Done."]
        );
        assert_eq!(texts("a.b. c"), vec!["a.b. c"]);
    }

    #[test]
    fn moves_sentence_marks_behind_a_closing_quote() {
        assert_eq!(
            texts("He said \"go.\" Then left."),
            vec!["He said \"go\".", "Then left."]
        );
    }

    #[test]
    fn drops_trailing_whitespace_and_keeps_an_unmatched_text_whole() {
        assert_eq!(texts("One. "), vec!["One."]);
        assert_eq!(texts("no ending"), vec!["no ending"]);
        assert_eq!(texts(""), vec![""]);
    }

    #[test]
    fn counts_in_utf16_units() {
        let s = segment("a\u{1F600}. b.");
        assert_eq!((s[1].start, s[1].end), (5, 7));
    }

    #[test]
    fn words_take_the_sentence_that_holds_them() {
        let context = "The sun is out. Is it warm?";
        let ids = word_sentence_ids(&segment(context), context);
        assert_eq!(ids, vec![0, 0, 0, 0, 1, 1, 1]);
    }

    #[test]
    fn splitting_matches_java_for_edge_cases() {
        let lens = |s: &str| split_lengths(&s.encode_utf16().collect::<Vec<_>>(), is_space);
        assert_eq!(lens(""), vec![0]);
        assert_eq!(lens("   "), Vec::<usize>::new());
        assert_eq!(lens(" a  bc "), vec![0, 1, 2]);
        assert_eq!(lens("ab"), vec![2]);
    }

    #[test]
    fn tokens_take_their_words_sentence() {
        let types = [0, 0, 1, 1, 1, 1];
        let words = [Some(0), None, Some(0), Some(1), Some(5), None];
        assert_eq!(
            token_sentence_ids(&types, &words, &[7, 8]),
            vec![-100, -100, 7, 8, -100, -100]
        );
    }

    struct Fixed(Vec<i64>);

    impl TorchScript for Fixed {
        fn forward(&self, inputs: &[Input<'_>]) -> Result<Vec<crate::torchscript::TorchOutput>> {
            assert_eq!(
                inputs.iter().map(|i| i.name).collect::<Vec<_>>(),
                [
                    "input_ids",
                    "attention_mask",
                    "token_type_ids",
                    "sentence_ids"
                ]
            );
            Ok(vec![crate::torchscript::TorchOutput {
                name: String::new(),
                group: 0,
                shape: vec![self.0.len()],
                data: Data::I64(self.0.clone()),
            }])
        }
    }

    fn word_level() -> Vec<u8> {
        br#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],
            "normalizer":null,"pre_tokenizer":{"type":"Whitespace"},
            "post_processor":{"type":"TemplateProcessing",
              "single":[{"Sequence":{"id":"A","type_id":0}}],
              "pair":[{"Sequence":{"id":"A","type_id":0}},{"Sequence":{"id":"B","type_id":1}}],
              "special_tokens":{}},
            "decoder":null,
            "model":{"type":"WordLevel","vocab":{"[UNK]":0,"a":1,"b":2},"unk_token":"[UNK]"}}"#
            .to_vec()
    }

    #[test]
    fn answers_with_the_sentences_the_model_names() {
        let model = HighlighterModel::torchscript(
            Box::new(Fixed(vec![1, 7])),
            &word_level(),
            Config::default(),
        )
        .unwrap();
        let hits = model.highlight("a", "a b. b a. a a.").unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(
            hits[0],
            Highlight {
                text: "b a.".into(),
                position: 1,
                start: 5,
                end: 9
            }
        );
    }

    #[test]
    fn answers_nothing_when_no_sentence_is_named() {
        let model = HighlighterModel::torchscript(
            Box::new(Fixed(vec![-1, 9])),
            &word_level(),
            Config::default(),
        )
        .unwrap();
        assert!(model.highlight("a", "a b.").unwrap().is_empty());
    }

    #[test]
    fn reads_every_chunk_of_a_long_context() {
        let config = Config {
            max_length: 6,
            stride: 1,
            ..Config::default()
        };
        let model =
            HighlighterModel::torchscript(Box::new(Fixed(vec![0])), &word_level(), config).unwrap();
        let context = vec!["a"; 20].join(" ");
        assert_eq!(model.highlight("b", &context).unwrap().len(), 1);
    }
}
