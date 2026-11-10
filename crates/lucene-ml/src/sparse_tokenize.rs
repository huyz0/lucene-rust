//! ml-commons' `SPARSE_TOKENIZE` models (`SparseTokenizerModel`): no network,
//! only a tokenizer and, optionally, `idf.json`'s weight per token. A text's
//! sparse vector is its distinct token ids, each decoded back to its token
//! (special tokens decode to nothing and are dropped) and weighted by the
//! token's idf, 1.0 when the file has none.

use std::collections::HashMap;

use tokenizers::Tokenizer;

use crate::{tokenizer, MlError, Result};

/// One entry of a text's sparse vector.
#[derive(Debug, Clone, PartialEq)]
pub struct Weight {
    /// The token id (`sparse_embedding_format: TOKEN_ID` keys by it).
    pub id: u32,
    /// The decoded token (the default `WORD` format keys by it).
    pub token: String,
    /// The token's idf weight.
    pub weight: f32,
}

/// A deployed sparse tokenizer.
pub struct SparseTokenizer {
    tokenizer: Tokenizer,
    idf: HashMap<String, f32>,
}

impl SparseTokenizer {
    /// Loads `tokenizer_json` (a `tokenizer.json`) and `idf_json` (an
    /// `idf.json` object of token to weight), if the model has one.
    pub fn load(tokenizer_json: &[u8], idf_json: Option<&[u8]>) -> Result<SparseTokenizer> {
        let tokenizer = tokenizer::load(tokenizer_json)?;
        let idf = match idf_json {
            Some(json) => serde_json::from_slice::<HashMap<String, f32>>(json)
                .map_err(|e| MlError::Model(format!("cannot read idf.json: {e}")))?,
            None => HashMap::new(),
        };
        Ok(SparseTokenizer { tokenizer, idf })
    }

    /// `text`'s sparse vector, in first-occurrence order of its tokens.
    pub fn encode(&self, text: &str) -> Result<Vec<Weight>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| MlError::Tokenizer(e.to_string()))?;
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::new();
        for &id in encoding.get_ids() {
            if !seen.insert(id) {
                continue;
            }
            let token = self
                .tokenizer
                .decode(&[id], true)
                .map_err(|e| MlError::Tokenizer(e.to_string()))?;
            if token.is_empty() {
                continue;
            }
            let weight = self.idf.get(&token).copied().unwrap_or(1.0);
            out.push(Weight { id, token, weight });
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOKENIZER: &str = r#"{"version":"1.0","truncation":null,"padding":null,
        "added_tokens":[{"id":0,"content":"[CLS]","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],
        "normalizer":null,"pre_tokenizer":{"type":"Whitespace"},
        "post_processor":{"type":"TemplateProcessing",
          "single":[{"SpecialToken":{"id":"[CLS]","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],
          "pair":[{"Sequence":{"id":"A","type_id":0}}],
          "special_tokens":{"[CLS]":{"id":"[CLS]","ids":[0],"tokens":["[CLS]"]}}},
        "decoder":null,
        "model":{"type":"WordLevel","vocab":{"[CLS]":0,"[UNK]":1,"hello":2,"world":3},"unk_token":"[UNK]"}}"#;

    #[test]
    fn weighs_distinct_tokens_by_idf_and_drops_special_tokens() {
        let t = SparseTokenizer::load(TOKENIZER.as_bytes(), Some(br#"{"hello": 2.5}"#)).unwrap();
        let w = t.encode("hello world hello").unwrap();
        assert_eq!(
            w,
            vec![
                Weight {
                    id: 2,
                    token: "hello".into(),
                    weight: 2.5
                },
                Weight {
                    id: 3,
                    token: "world".into(),
                    weight: 1.0
                },
            ]
        );
        assert!(t.encode("").unwrap().is_empty());
    }

    #[test]
    fn without_idf_every_weight_is_one_and_a_bad_idf_is_refused() {
        let t = SparseTokenizer::load(TOKENIZER.as_bytes(), None).unwrap();
        assert!(t.encode("world").unwrap().iter().all(|w| w.weight == 1.0));
        assert!(matches!(
            SparseTokenizer::load(TOKENIZER.as_bytes(), Some(b"[1]")),
            Err(MlError::Model(_))
        ));
    }
}
