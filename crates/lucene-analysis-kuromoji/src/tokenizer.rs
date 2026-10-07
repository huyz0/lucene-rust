//! `org.apache.lucene.analysis.ja.JapaneseTokenizer`: Japanese text
//! segmented by the Viterbi search over IPADIC (and an optional user
//! dictionary), in normal, search or extended mode, with n-best output.

use std::sync::Arc;

use lucene_analysis::attributes::AttributeSource;
use lucene_analysis::morph::viterbi::forward;
use lucene_analysis::morph::{ConnectionCosts, GraphvizFormatter, Viterbi};
use lucene_analysis::reader::{CharReader, StrReader};
use lucene_analysis::token_stream::{TokenStream, Tokenizer, TokenizerInput};
use lucene_analysis::AnalysisError;

use crate::attributes::{
    BaseFormAttribute, InflectionAttribute, PartOfSpeechAttribute, ReadingAttribute,
};
use crate::dict::{self, TokenInfoDictionary, UnknownDictionary, UserDictionary};
use crate::token::Token;
use crate::viterbi::JaViterbi;

/// `JapaneseTokenizer.Mode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Mode {
    /// `NORMAL`: ordinary segmentation, no decomposition of compounds.
    Normal,
    /// `SEARCH`: segmentation for search -- long compounds are decomposed,
    /// the compound kept as a synonym (`DEFAULT_MODE`).
    #[default]
    Search,
    /// `EXTENDED`: search mode, unknown words as unigrams.
    Extended,
}

impl Mode {
    /// `Mode.valueOf(name)` (`IllegalArgumentException` for another name).
    pub fn value_of(name: &str) -> Result<Mode, AnalysisError> {
        match name {
            "NORMAL" => Ok(Mode::Normal),
            "SEARCH" => Ok(Mode::Search),
            "EXTENDED" => Ok(Mode::Extended),
            _ => Err(AnalysisError::IllegalArgument(format!(
                "No enum constant org.apache.lucene.analysis.ja.JapaneseTokenizer.Mode.{name}"
            ))),
        }
    }
}

/// `JapaneseTokenizer`.
pub struct JapaneseTokenizer {
    atts: AttributeSource,
    input: TokenizerInput,
    viterbi: Viterbi<Token>,
    lang: JaViterbi,
    /// Position of the last token returned (posInc 0 or 1).
    last_token_pos: i32,
}

impl JapaneseTokenizer {
    /// `new JapaneseTokenizer(userDictionary, discardPunctuation, mode)`:
    /// compound tokens discarded.
    pub fn new(
        user_dictionary: Option<Arc<UserDictionary>>,
        discard_punctuation: bool,
        mode: Mode,
    ) -> Self {
        Self::with_options(user_dictionary, discard_punctuation, true, mode)
    }

    /// `new JapaneseTokenizer(userDictionary, discardPunctuation,
    /// discardCompoundToken, mode)` over the default dictionaries.
    pub fn with_options(
        user_dictionary: Option<Arc<UserDictionary>>,
        discard_punctuation: bool,
        discard_compound_token: bool,
        mode: Mode,
    ) -> Self {
        Self::with_dictionaries(
            TokenInfoDictionary::instance(),
            UnknownDictionary::instance(),
            dict::ConnectionCosts::instance(),
            user_dictionary,
            discard_punctuation,
            discard_compound_token,
            mode,
        )
    }

    /// `new JapaneseTokenizer(factory, systemDictionary, unkDictionary,
    /// connectionCosts, userDictionary, discardPunctuation,
    /// discardCompoundToken, mode)`.
    pub fn with_dictionaries(
        system_dictionary: Arc<TokenInfoDictionary>,
        unk_dictionary: Arc<UnknownDictionary>,
        connection_costs: Arc<ConnectionCosts>,
        user_dictionary: Option<Arc<UserDictionary>>,
        discard_punctuation: bool,
        discard_compound_token: bool,
        mode: Mode,
    ) -> Self {
        let fst = Arc::clone(system_dictionary.fst());
        let user_fst = user_dictionary.as_ref().map(|u| Arc::clone(u.fst()));
        let (search_mode, extended_mode, output_compounds) = match mode {
            Mode::Search => (true, false, !discard_compound_token),
            Mode::Extended => (true, true, !discard_compound_token),
            Mode::Normal => (false, false, false),
        };
        let lang = JaViterbi::new(
            system_dictionary,
            unk_dictionary,
            user_dictionary,
            discard_punctuation,
            search_mode,
            extended_mode,
            output_compounds,
        );
        let mut viterbi = Viterbi::new(fst, user_fst, connection_costs);
        viterbi.reset_buffer();
        viterbi.reset_state();
        let mut atts = AttributeSource::new();
        // Java's addAttribute order (after the core attributes).
        atts.add_custom::<BaseFormAttribute>();
        atts.add_custom::<PartOfSpeechAttribute>();
        atts.add_custom::<ReadingAttribute>();
        atts.add_custom::<InflectionAttribute>();
        JapaneseTokenizer {
            atts,
            input: TokenizerInput::new(),
            viterbi,
            lang,
            last_token_pos: -1,
        }
    }

    /// `setGraphvizFormatter(dotOut)`: from now on every backtrace is
    /// recorded; [`Self::graphviz`] gives the text.
    pub fn set_graphviz_formatter(&mut self, dot_out: GraphvizFormatter) {
        self.lang.dot_out = Some(dot_out);
    }

    /// The formatter set by [`Self::set_graphviz_formatter`].
    pub fn graphviz(&self) -> Option<&GraphvizFormatter> {
        self.lang.dot_out.as_ref()
    }

    /// `setNBestCost(value)`.
    pub fn set_n_best_cost(&mut self, value: i32) {
        self.lang.nbest.set_n_best_cost(&mut self.viterbi, value);
    }

    /// `probeDelta(inText, requiredToken)`.
    fn probe_delta(&mut self, in_text: &str, required_token: &str) -> Result<i32, AnalysisError> {
        let in_units: Vec<u16> = in_text.encode_utf16().collect();
        let req: Vec<u16> = required_token.encode_utf16().collect();
        let Some(start) = find(&in_units, &req) else {
            // -1 when no requiredToken.
            return Ok(-1);
        };
        let start = i32::try_from(start).unwrap_or(i32::MAX);
        let end = start.wrapping_add(i32::try_from(req.len()).unwrap_or(i32::MAX));
        let mut delta = i32::MAX;
        let save = self.lang.nbest.n_best_cost();
        self.set_reader(Box::new(StrReader::new(in_text.to_string())))?;
        let run = (|| {
            self.reset()?;
            self.set_n_best_cost(1);
            let mut prev_root_base = -1;
            while self.increment_token()? {
                let root = self.lang.nbest.lattice_root_base().unwrap_or(-1);
                if root != prev_root_base {
                    prev_root_base = root;
                    delta = delta.min(self.lang.nbest.probe_delta(start, end).unwrap_or(i32::MAX));
                }
            }
            Ok::<(), AnalysisError>(())
        })();
        // reset & end; setReader & close
        let ended = self.end();
        let closed = self.close();
        self.set_n_best_cost(save);
        run?;
        ended?;
        closed?;
        Ok(if delta == i32::MAX { -1 } else { delta })
    }

    /// `calcNBestCost(examples)`: the largest cost delta that brings each
    /// `text-token` example's token (examples separated by `/`) into the
    /// n-best output.
    pub fn calc_n_best_cost(&mut self, examples: &str) -> Result<i32, AnalysisError> {
        let mut max_delta = 0;
        for example in examples.split('/') {
            if example.is_empty() {
                continue;
            }
            let pair: Vec<&str> = example.split('-').collect();
            let pair = trim_trailing_empty(pair);
            if pair.len() != 2 {
                return Err(AnalysisError::IllegalState(format!(
                    "RuntimeException: Unexpected example form: {example} (expected two '-')"
                )));
            }
            max_delta = max_delta.max(self.probe_delta(pair[0], pair[1])?);
        }
        Ok(max_delta)
    }

    /// `incrementToken()`'s attribute half: `token` to the attributes.
    fn set_token(&mut self, token: Token) -> Result<(), AnalysisError> {
        let base = token.base();
        let (start, end) = (base.start_offset, base.end_offset);
        let pos_len = base.pos_len;
        self.atts.clear_attributes();
        self.atts.set_term_utf16(base.surface());
        let (cs, ce) = (
            self.input.correct_offset(start),
            self.input.correct_offset(end),
        );
        self.atts.set_offset(cs, ce)?;
        if start == self.last_token_pos {
            self.atts.set_position_increment(0)?;
            self.atts.set_position_length(pos_len)?;
        } else if self.viterbi.output_nbest {
            // The position length is always calculated if outputNBest is
            // true.
            self.atts.set_position_increment(1)?;
            self.atts.set_position_length(pos_len)?;
        } else {
            self.atts.set_position_increment(1)?;
            self.atts.set_position_length(1)?;
        }
        self.last_token_pos = start;
        let token = Arc::new(token);
        self.atts.add_custom::<BaseFormAttribute>().token = Some(Arc::clone(&token));
        self.atts.add_custom::<PartOfSpeechAttribute>().token = Some(Arc::clone(&token));
        self.atts.add_custom::<ReadingAttribute>().token = Some(Arc::clone(&token));
        self.atts.add_custom::<InflectionAttribute>().token = Some(token);
        Ok(())
    }
}

/// `String.split`'s trailing-empty-string removal.
fn trim_trailing_empty(mut v: Vec<&str>) -> Vec<&str> {
    while v.last().is_some_and(|s| s.is_empty()) {
        v.pop();
    }
    v
}

/// `String.indexOf(String)` on UTF-16 units.
fn find(hay: &[u16], needle: &[u16]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    hay.windows(needle.len()).position(|w| w == needle)
}

impl TokenStream for JapaneseTokenizer {
    fn attributes(&self) -> &AttributeSource {
        &self.atts
    }
    fn attributes_mut(&mut self) -> &mut AttributeSource {
        &mut self.atts
    }
    /// A source, not a wrapper: no conditional wrapper below it.
    fn conditional_root(&mut self) -> Option<&mut dyn std::any::Any> {
        None
    }
    fn as_tokenizer(&mut self) -> Option<&mut dyn Tokenizer> {
        Some(self)
    }

    fn increment_token(&mut self) -> Result<bool, AnalysisError> {
        // forward() can return w/o producing any new tokens, when the
        // tokens it had produced were entirely punctuation. So we loop here
        // until we get a real token or we end:
        while self.viterbi.pending.is_empty() {
            if self.viterbi.end {
                return Ok(false);
            }
            // Push Viterbi forward some more:
            let reader = self.input.reader()?;
            forward(&mut self.viterbi, &mut self.lang, reader)?;
        }
        let Some(token) = self.viterbi.pending.pop() else {
            return Ok(false);
        };
        self.set_token(token)?;
        Ok(true)
    }

    fn reset(&mut self) -> Result<(), AnalysisError> {
        self.input.reset();
        self.viterbi.reset_buffer();
        self.viterbi.reset_state();
        self.last_token_pos = -1;
        Ok(())
    }

    fn end(&mut self) -> Result<(), AnalysisError> {
        self.atts.end_attributes();
        // Set final offset
        let final_offset = self.input.correct_offset(self.viterbi.pos);
        self.atts.set_offset(final_offset, final_offset)
    }

    fn close(&mut self) -> Result<(), AnalysisError> {
        let r = self.input.close();
        self.viterbi.reset_buffer();
        r
    }
}

impl Tokenizer for JapaneseTokenizer {
    fn set_reader(&mut self, input: Box<dyn CharReader>) -> Result<(), AnalysisError> {
        self.input.set_reader(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modes_and_n_best_examples() {
        assert_eq!(Mode::value_of("NORMAL").unwrap(), Mode::Normal);
        assert_eq!(Mode::value_of("EXTENDED").unwrap(), Mode::Extended);
        assert!(Mode::value_of("normal").is_err());
        assert_eq!(Mode::default(), Mode::Search);
        let mut t = JapaneseTokenizer::new(None, true, Mode::Normal);
        assert!(t.graphviz().is_none());
        assert_eq!(t.calc_n_best_cost("//").unwrap(), 0);
        assert_eq!(t.calc_n_best_cost("abc-xyz").unwrap(), 0);
        assert!(t.calc_n_best_cost("a-b-c").is_err());
        assert_eq!(find(&[1, 2, 3], &[]), Some(0));
        assert_eq!(trim_trailing_empty(vec!["a", ""]), ["a"]);
        // incrementToken before reset: Java's ILLEGAL_STATE_READER.
        assert!(t.increment_token().is_err());
    }
}
