//! `opennlp.tools.lemmatizer`: `LemmatizerME.lemmatize` over a loaded
//! `LemmatizerModel` -- a beam search (`BeamSize`, default 3) predicting
//! each word's shortest edit script (`DefaultLemmatizerContextGenerator`),
//! then `StringUtil.decodeShortestEditScript` applying it to the
//! lower-cased word (an empty lemma is `_`).

use std::sync::Arc;

use lucene_analysis::factory::FactoryError;
use lucene_analysis::AnalysisError;

use super::ml::{BeamSearch, MaxentModel, SequenceContext};
use super::model::{BaseModel, Component};
use super::text::{self, JString};

/// A loaded `LemmatizerModel`.
#[derive(Debug, Clone)]
pub struct LemmatizerModel {
    base: Arc<BaseModel>,
}

impl LemmatizerModel {
    /// `new LemmatizerModel(InputStream)`.
    pub fn load(bytes: &[u8]) -> Result<LemmatizerModel, FactoryError> {
        Ok(LemmatizerModel {
            base: Arc::new(BaseModel::load(Component::Lemmatizer, bytes)?),
        })
    }
}

/// `LemmatizerME`.
#[derive(Debug, Clone)]
pub struct Lemmatizer {
    model: Arc<MaxentModel>,
    beam: BeamSearch,
}

impl Lemmatizer {
    /// `new LemmatizerME(model)`.
    pub fn new(model: &LemmatizerModel) -> Result<Lemmatizer, AnalysisError> {
        Ok(Lemmatizer {
            beam: BeamSearch::new(model.base.beam_size(3)?),
            model: model.base.maxent_model(),
        })
    }

    /// `lemmatize(toks, tags)`.
    pub fn lemmatize(
        &mut self,
        toks: &[JString],
        tags: &[JString],
    ) -> Result<Vec<JString>, AnalysisError> {
        let mut ctx = LemmaContext { toks, tags };
        let ses = self.beam.best_sequence(&self.model, toks.len(), &mut ctx)?;
        toks.iter()
            .zip(ses)
            .map(|(tok, s)| {
                let lemma = decode_shortest_edit_script(
                    &text::string_to_lower_case(tok),
                    self.model.outcome(s as usize),
                )?;
                Ok(if lemma.is_empty() {
                    text::units("_")
                } else {
                    lemma
                })
            })
            .collect()
    }
}

struct LemmaContext<'a> {
    toks: &'a [JString],
    tags: &'a [JString],
}

impl SequenceContext for LemmaContext<'_> {
    // ARITH: suffix and prefix lengths below 8, subtracted with saturation.
    #[allow(clippy::arithmetic_side_effects)]
    fn context(&mut self, index: usize, prior: &[u32], outcomes: &[JString], f: &mut Vec<JString>) {
        let cat = |k: &str, v: &[u16]| text::concat(&[&text::units(k), v]);
        let lex = &self.toks[index];
        let p_1 = match index.checked_sub(1).and_then(|i| prior.get(i)) {
            Some(&o) => cat("p_1=", &outcomes[o as usize]),
            None => text::units("p_1=bos"),
        };
        let w0 = cat("w0=", lex);
        let t0 = cat("t0=", self.tags.get(index).map_or(&[][..], Vec::as_slice));
        f.push(w0.clone());
        f.push(t0.clone());
        f.push(p_1.clone());
        f.push(text::concat(&[&p_1, &t0]));
        f.push(text::concat(&[&p_1, &w0]));
        // getSuffixes/getPrefixes leave index 0 unset: "suf=null", "pre=null"
        f.push(text::units("suf=null"));
        for li in 1..7 {
            f.push(cat("suf=", &lex[lex.len().saturating_sub(li + 1)..]));
        }
        f.push(text::units("pre=null"));
        for li in 1..5 {
            f.push(cat("pre=", &lex[..(li + 1).min(lex.len())]));
        }
        if lex.contains(&u16::from(b'-')) {
            f.push(text::units("h"));
        }
        if lex
            .iter()
            .any(|c| (u16::from(b'A')..=u16::from(b'Z')).contains(c))
        {
            f.push(text::units("c"));
        }
        if lex
            .iter()
            .any(|c| (u16::from(b'0')..=u16::from(b'9')).contains(c))
        {
            f.push(text::units("d"));
        }
    }

    fn valid(&self, _: usize, _: &[u32], _: u32, _: &[JString]) -> bool {
        true
    }
}

/// `StringBuffer.reverse()`: the units reversed, a surrogate pair kept in
/// order.
// ARITH: `i + 1 < len`.
#[allow(clippy::arithmetic_side_effects)]
fn java_reverse(v: &mut [u16]) {
    v.reverse();
    let mut i = 0;
    while i + 1 < v.len() {
        if (0xDC00..=0xDFFF).contains(&v[i]) && (0xD800..=0xDBFF).contains(&v[i + 1]) {
            v.swap(i, i + 1);
            i += 1;
        }
        i += 1;
    }
}

/// `StringUtil.decodeShortestEditScript(wordForm, permutations)`: the
/// script's `R<i><from><to>`, `I<i><c>` and `D<i><c>` operations applied to
/// the reversed word (an index out of range gives the word back
/// unchanged; any other letter is skipped). Java's
/// `NumberFormatException` for an index that is not a digit, and
/// `StringIndexOutOfBoundsException` for a script cut short.
// ARITH: script offsets bounded by the script's length (each read is checked).
#[allow(clippy::arithmetic_side_effects)]
pub fn decode_shortest_edit_script(word: &[u16], ses: &[u16]) -> Result<JString, AnalysisError> {
    let cut = || {
        AnalysisError::IllegalState(format!(
            "StringIndexOutOfBoundsException: the edit script {} ends inside an operation",
            text::to_string(ses)
        ))
    };
    let at = |i: usize| ses.get(i).copied().ok_or_else(cut);
    let digit = |i: usize| -> Result<usize, AnalysisError> {
        let c = at(i)?;
        text::parse_int(&[c])
            .and_then(|d| usize::try_from(d).ok())
            .ok_or_else(|| {
                AnalysisError::IllegalArgument(format!(
                    "NumberFormatException: For input string: \"{}\"",
                    text::to_string(&[c])
                ))
            })
    };
    let mut lemma = word.to_vec();
    java_reverse(&mut lemma);
    let mut p = 0;
    while p < ses.len() {
        let op = ses[p];
        p += 1;
        match op {
            0x52 => {
                // 'R'
                let index = digit(p)?;
                let replace = at(p + 1)?;
                let with = at(p + 2)?;
                if lemma.len() <= index {
                    return Ok(word.to_vec());
                }
                if lemma[index] == replace {
                    lemma[index] = with;
                }
                p += 3;
            }
            0x49 => {
                // 'I'
                let index = digit(p)?;
                let insert = at(p + 1)?;
                if lemma.len() < index {
                    return Ok(word.to_vec());
                }
                lemma.insert(index, insert);
                p += 2;
            }
            0x44 => {
                // 'D'
                let index = digit(p)?;
                if lemma.len() <= index {
                    return Ok(word.to_vec());
                }
                lemma.remove(index);
                p += 2;
            }
            _ => {}
        }
    }
    java_reverse(&mut lemma);
    Ok(lemma)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arithmetic_side_effects)] // test code
    use super::*;

    fn d(word: &str, ses: &str) -> Result<String, AnalysisError> {
        decode_shortest_edit_script(&text::units(word), &text::units(ses))
            .map(|l| text::to_string(&l))
    }

    #[test]
    fn edit_scripts() {
        // dogs -> dog: reversed "sgod", delete index 0 's'
        assert_eq!(d("dogs", "D0s").unwrap(), "dog");
        // ran -> run: reversed "nar", replace index 1 'a' with 'u'
        assert_eq!(d("ran", "R1au").unwrap(), "run");
        assert_eq!(d("ran", "R1xu").unwrap(), "ran");
        // insert
        assert_eq!(d("go", "I0e").unwrap(), "goe");
        assert_eq!(d("dogs", "O").unwrap(), "dogs");
        assert_eq!(d("ab", "D5x").unwrap(), "ab");
        assert_eq!(d("ab", "R5xy").unwrap(), "ab");
        assert_eq!(d("ab", "I5x").unwrap(), "ab");
        assert_eq!(d("ab", "I٢x").unwrap(), "xab");
        assert!(matches!(
            d("ab", "Dx"),
            Err(AnalysisError::IllegalArgument(_))
        ));
        assert!(matches!(
            d("ab", "R0a"),
            Err(AnalysisError::IllegalState(_))
        ));
        assert!(matches!(d("ab", "I"), Err(AnalysisError::IllegalState(_))));
        // surrogate pairs survive the reversals
        assert_eq!(d("a😀", "D2x").unwrap(), "😀");
        assert_eq!(d("a😀b", "D0b").unwrap(), "a😀");
        let mut v = vec![0xDC00, 0xD800];
        java_reverse(&mut v);
        assert_eq!(v, vec![0xD800, 0xDC00]);
    }
}
