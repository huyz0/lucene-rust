//! `org.apache.lucene.analysis.ja.JapaneseHiraganaUppercaseFilter` and
//! `JapaneseKatakanaUppercaseFilter` (with `JapaneseFilterUtil`'s map):
//! small kana to their normal-size letters.

use lucene_analysis::{AnalysisError, TokenFilter, TokenStream};

/// `JapaneseHiraganaUppercaseFilter.LETTER_MAPPINGS`.
fn hiragana(c: char) -> Option<char> {
    Some(match c {
        'ぁ' => 'あ',
        'ぃ' => 'い',
        'ぅ' => 'う',
        'ぇ' => 'え',
        'ぉ' => 'お',
        'っ' => 'つ',
        'ゃ' => 'や',
        'ゅ' => 'ゆ',
        'ょ' => 'よ',
        'ゎ' => 'わ',
        'ゕ' => 'か',
        'ゖ' => 'け',
        _ => return None,
    })
}

/// `JapaneseKatakanaUppercaseFilter.LETTER_MAPPINGS`.
fn katakana(c: char) -> Option<char> {
    Some(match c {
        'ァ' => 'ア',
        'ィ' => 'イ',
        'ゥ' => 'ウ',
        'ェ' => 'エ',
        'ォ' => 'オ',
        'ヵ' => 'カ',
        'ㇰ' => 'ク',
        'ヶ' => 'ケ',
        'ㇱ' => 'シ',
        'ㇲ' => 'ス',
        'ッ' => 'ツ',
        'ㇳ' => 'ト',
        'ㇴ' => 'ヌ',
        'ㇵ' => 'ハ',
        'ㇶ' => 'ヒ',
        'ㇷ' => 'フ',
        'ㇸ' => 'ヘ',
        'ㇹ' => 'ホ',
        'ㇺ' => 'ム',
        'ャ' => 'ヤ',
        'ュ' => 'ユ',
        'ョ' => 'ヨ',
        'ㇻ' => 'ラ',
        'ㇼ' => 'リ',
        'ㇽ' => 'ル',
        'ㇾ' => 'レ',
        'ㇿ' => 'ロ',
        'ヮ' => 'ワ',
        _ => return None,
    })
}

/// `JapaneseHiraganaUppercaseFilter`.
pub struct JapaneseHiraganaUppercaseFilter<I> {
    input: I,
}

impl<I: TokenStream> JapaneseHiraganaUppercaseFilter<I> {
    /// `new JapaneseHiraganaUppercaseFilter(input)`.
    pub fn new(input: I) -> Self {
        JapaneseHiraganaUppercaseFilter { input }
    }
}

impl<I: TokenStream> TokenFilter for JapaneseHiraganaUppercaseFilter<I> {
    type Input = I;
    fn input(&self) -> &I {
        &self.input
    }
    fn input_mut(&mut self) -> &mut I {
        &mut self.input
    }
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if !self.input.increment_token()? {
            return Ok(false);
        }
        let atts = self.input.attributes_mut();
        if atts.term().chars().any(|c| hiragana(c).is_some()) {
            let mapped: String = atts
                .term()
                .chars()
                .map(|c| hiragana(c).unwrap_or(c))
                .collect();
            atts.set_term(&mapped);
        }
        Ok(true)
    }
}

/// `JapaneseKatakanaUppercaseFilter`: also ㇷ゚ (ㇷ and the combining
/// semi-voiced mark) to プ.
pub struct JapaneseKatakanaUppercaseFilter<I> {
    input: I,
}

impl<I: TokenStream> JapaneseKatakanaUppercaseFilter<I> {
    /// `new JapaneseKatakanaUppercaseFilter(input)`.
    pub fn new(input: I) -> Self {
        JapaneseKatakanaUppercaseFilter { input }
    }
}

impl<I: TokenStream> TokenFilter for JapaneseKatakanaUppercaseFilter<I> {
    type Input = I;
    fn input(&self) -> &I {
        &self.input
    }
    fn input_mut(&mut self) -> &mut I {
        &mut self.input
    }
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if !self.input.increment_token()? {
            return Ok(false);
        }
        let atts = self.input.attributes_mut();
        if atts.term().chars().any(|c| katakana(c).is_some()) {
            let mut out = String::with_capacity(atts.term().len());
            let mut it = atts.term().chars().peekable();
            while let Some(c) = it.next() {
                if c == 'ㇷ' && it.peek() == Some(&'\u{309A}') {
                    // ㇷ゚detected, replace it by プ.
                    it.next();
                    out.push('プ');
                } else {
                    out.push(katakana(c).unwrap_or(c));
                }
            }
            atts.set_term(&out);
        }
        Ok(true)
    }
}
