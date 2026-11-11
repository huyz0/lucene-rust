//! The `bert-uncased` and `mbert-uncased` analyzers of ml-commons
//! (`HFModelTokenizer`): a Hugging Face tokenizer read as a token stream.
//!
//! Each word piece of the text becomes one token, in order, with its
//! vocabulary id, its offsets in Unicode scalar values (as DJL reports
//! them, so a character outside the BMP counts once) and its weight
//! from the analyzer's `idf.json` (1.0 when the piece has none). Special
//! tokens are not added and the text is never cut: ml-commons encodes
//! without special tokens and walks the overflowing encodings too, so the
//! stream covers the whole text either way.

use std::collections::HashMap;

use tokenizers::Tokenizer;

use crate::{MlError, Result};

/// One token of the stream.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub id: u32,
    pub token: String,
    pub start: u32,
    pub end: u32,
    pub weight: f32,
}

pub struct HfAnalyzer {
    tokenizer: Tokenizer,
    weights: HashMap<String, f32>,
}

impl HfAnalyzer {
    /// The analyzer of `tokenizer.json` and, when given, `idf.json` bytes.
    pub fn load(tokenizer_json: &[u8], idf_json: Option<&[u8]>) -> Result<HfAnalyzer> {
        let mut tokenizer =
            Tokenizer::from_bytes(tokenizer_json).map_err(|e| MlError::Tokenizer(e.to_string()))?;
        tokenizer
            .with_truncation(None)
            .map_err(|e| MlError::Tokenizer(e.to_string()))?;
        tokenizer.with_padding(None);
        let weights = match idf_json {
            Some(json) => serde_json::from_slice::<HashMap<String, f32>>(json)
                .map_err(|e| MlError::Model(format!("cannot read idf.json: {e}")))?,
            None => HashMap::new(),
        };
        Ok(HfAnalyzer { tokenizer, weights })
    }

    /// The tokens of `text`; none for an empty text.
    pub fn analyze(&self, text: &str) -> Result<Vec<Token>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        let encoding = self
            .tokenizer
            .encode_char_offsets(text, false)
            .map_err(|e| MlError::Tokenizer(e.to_string()))?;
        let at = |c: usize| u32::try_from(c).unwrap_or(u32::MAX);
        let mut out = Vec::with_capacity(encoding.get_ids().len());
        for ((&id, token), &(start, end)) in encoding
            .get_ids()
            .iter()
            .zip(encoding.get_tokens())
            .zip(encoding.get_offsets())
        {
            out.push(Token {
                id,
                token: token.clone(),
                start: at(start),
                end: at(end),
                weight: self.weights.get(token).copied().unwrap_or(1.0),
            });
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOKENIZER: &str = r#"{"version":"1.0","truncation":{"direction":"Right","max_length":2,"strategy":"LongestFirst","stride":0},
        "padding":null,
        "added_tokens":[{"id":0,"content":"[CLS]","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],
        "normalizer":{"type":"Lowercase"},"pre_tokenizer":{"type":"Whitespace"},
        "post_processor":{"type":"TemplateProcessing",
          "single":[{"SpecialToken":{"id":"[CLS]","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],
          "pair":[{"Sequence":{"id":"A","type_id":0}}],
          "special_tokens":{"[CLS]":{"id":"[CLS]","ids":[0],"tokens":["[CLS]"]}}},
        "decoder":null,
        "model":{"type":"WordLevel","vocab":{"[CLS]":0,"[UNK]":1,"hello":2,"world":3,"😀":4},"unk_token":"[UNK]"}}"#;

    #[test]
    fn streams_every_piece_in_order_without_special_tokens_or_truncation() {
        let a = HfAnalyzer::load(TOKENIZER.as_bytes(), Some(br#"{"hello": 2.5}"#)).unwrap();
        let t = a.analyze("Hello 😀 world hello").unwrap();
        let ids: Vec<u32> = t.iter().map(|t| t.id).collect();
        assert_eq!(ids, vec![2, 4, 3, 2]);
        assert_eq!(t[0].token, "hello");
        assert_eq!((t[0].start, t[0].end, t[0].weight), (0, 5, 2.5));
        assert_eq!(
            (t[1].start, t[1].end),
            (6, 7),
            "one scalar value, two UTF-16 units"
        );
        assert_eq!((t[2].start, t[2].end, t[2].weight), (8, 13, 1.0));
        assert_eq!((t[3].start, t[3].end), (14, 19));
        assert!(a.analyze("").unwrap().is_empty());
    }

    #[test]
    fn without_idf_every_weight_is_one_and_a_bad_idf_is_refused() {
        let a = HfAnalyzer::load(TOKENIZER.as_bytes(), None).unwrap();
        assert!(a.analyze("world").unwrap().iter().all(|t| t.weight == 1.0));
        assert!(matches!(
            HfAnalyzer::load(TOKENIZER.as_bytes(), Some(b"[1]")),
            Err(MlError::Model(_))
        ));
        assert!(matches!(
            HfAnalyzer::load(b"{", None),
            Err(MlError::Tokenizer(_))
        ));
    }
}
