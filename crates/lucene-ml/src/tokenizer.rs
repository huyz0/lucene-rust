//! A Hugging Face `tokenizer.json`, configured as ml-commons' translators get
//! it from DJL's `HuggingFaceTokenizer.builder().optPadding(true)`.
//!
//! DJL's defaults then are: special tokens added, truncation `LONGEST_FIRST`
//! at the file's own truncation length when it sets one (else 512) and never
//! past 512, the file's stride, and padding to the longest sequence of a
//! batch.

use tokenizers::{
    PaddingParams, PaddingStrategy, Tokenizer, TruncationDirection, TruncationParams,
    TruncationStrategy,
};

use crate::{MlError, Result};

/// DJL's `modelMaxLength`, the cap on any truncation length.
pub const MODEL_MAX_LENGTH: usize = 512;

/// One text's tokens, as the model inputs take them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Encoded {
    /// `input_ids`.
    pub ids: Vec<i64>,
    /// `attention_mask`.
    pub attention_mask: Vec<i64>,
    /// `token_type_ids`.
    pub type_ids: Vec<i64>,
}

/// Parses `json` (a `tokenizer.json`) and applies DJL's truncation and
/// padding.
pub fn load(json: &[u8]) -> Result<Tokenizer> {
    let mut tokenizer =
        Tokenizer::from_bytes(json).map_err(|e| MlError::Tokenizer(e.to_string()))?;
    let (max_length, stride) = match tokenizer.get_truncation() {
        Some(t) => (t.max_length.min(MODEL_MAX_LENGTH), t.stride),
        None => (MODEL_MAX_LENGTH, 0),
    };
    tokenizer
        .with_truncation(Some(TruncationParams {
            direction: TruncationDirection::Right,
            max_length,
            strategy: TruncationStrategy::LongestFirst,
            stride,
        }))
        .map_err(|e| MlError::Tokenizer(e.to_string()))?;
    let padding = match tokenizer.get_padding() {
        Some(p) => PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            ..p.clone()
        },
        None => PaddingParams::default(),
    };
    tokenizer.with_padding(Some(padding));
    Ok(tokenizer)
}

/// Encodes one text with special tokens.
pub fn encode(tokenizer: &Tokenizer, text: &str) -> Result<Encoded> {
    encoded(tokenizer.encode(text, true))
}

/// Encodes the pair `first`, `second` as one sequence with special tokens,
/// as DJL's `encode(text, textPair)` does for a cross-encoder.
pub fn encode_pair(tokenizer: &Tokenizer, first: &str, second: &str) -> Result<Encoded> {
    encoded(tokenizer.encode((first, second), true))
}

fn encoded(encoding: tokenizers::Result<tokenizers::Encoding>) -> Result<Encoded> {
    let encoding = encoding.map_err(|e| MlError::Tokenizer(e.to_string()))?;
    let widen = |v: &[u32]| v.iter().map(|&x| i64::from(x)).collect::<Vec<_>>();
    Ok(Encoded {
        ids: widen(encoding.get_ids()),
        attention_mask: widen(encoding.get_attention_mask()),
        type_ids: widen(encoding.get_type_ids()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word_level(truncation: &str) -> Vec<u8> {
        format!(
            r#"{{"version":"1.0","truncation":{truncation},"padding":null,"added_tokens":[],
            "normalizer":null,"pre_tokenizer":{{"type":"Whitespace"}},
            "post_processor":{{"type":"TemplateProcessing",
              "single":[{{"SpecialToken":{{"id":"[CLS]","type_id":0}}}},{{"Sequence":{{"id":"A","type_id":0}}}}],
              "pair":[{{"Sequence":{{"id":"A","type_id":0}}}},{{"Sequence":{{"id":"B","type_id":1}}}}],
              "special_tokens":{{"[CLS]":{{"id":"[CLS]","ids":[1],"tokens":["[CLS]"]}}}}}},
            "decoder":null,
            "model":{{"type":"WordLevel","vocab":{{"[UNK]":0,"[CLS]":1,"a":2,"b":3}},"unk_token":"[UNK]"}}}}"#
        )
        .into_bytes()
    }

    #[test]
    fn adds_special_tokens_and_masks() {
        let t = load(&word_level("null")).unwrap();
        let e = encode(&t, "a b zz").unwrap();
        assert_eq!(e.ids, vec![1, 2, 3, 0]);
        assert_eq!(e.attention_mask, vec![1; 4]);
        assert_eq!(e.type_ids, vec![0; 4]);
    }

    #[test]
    fn truncates_at_512_without_a_file_length() {
        let t = load(&word_level("null")).unwrap();
        let text = vec!["a"; 600].join(" ");
        assert_eq!(encode(&t, &text).unwrap().ids.len(), MODEL_MAX_LENGTH);
    }

    #[test]
    fn keeps_a_shorter_file_length_and_caps_a_longer_one() {
        let short = r#"{"direction":"Right","max_length":3,"strategy":"LongestFirst","stride":0}"#;
        let t = load(&word_level(short)).unwrap();
        assert_eq!(encode(&t, "a b a b").unwrap().ids, vec![1, 2, 3]);
        let long =
            r#"{"direction":"Right","max_length":4096,"strategy":"LongestFirst","stride":0}"#;
        let t = load(&word_level(long)).unwrap();
        let text = vec!["b"; 900].join(" ");
        assert_eq!(encode(&t, &text).unwrap().ids.len(), MODEL_MAX_LENGTH);
    }

    #[test]
    fn encodes_a_pair_with_its_type_ids() {
        let t = load(&word_level("null")).unwrap();
        let e = encode_pair(&t, "a", "b b").unwrap();
        assert_eq!(e.ids, vec![2, 3, 3]);
        assert_eq!(e.type_ids, vec![0, 1, 1]);
    }

    #[test]
    fn refuses_a_malformed_file() {
        assert!(matches!(load(b"{"), Err(MlError::Tokenizer(_))));
    }
}
