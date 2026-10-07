//! `org.apache.lucene.analysis.ko.KoreanTokenizer`: Korean text segmented
//! by the Viterbi search over mecab-ko-dic (and an optional user
//! dictionary), compounds decompounded per [`DecompoundMode`].

use std::sync::Arc;

use lucene_analysis::attributes::AttributeSource;
use lucene_analysis::morph::viterbi::forward;
use lucene_analysis::morph::{ConnectionCosts, GraphvizFormatter, Viterbi};
use lucene_analysis::reader::CharReader;
use lucene_analysis::token_stream::{TokenStream, Tokenizer, TokenizerInput};
use lucene_analysis::AnalysisError;

use crate::attributes::{PartOfSpeechAttribute, ReadingAttribute};
use crate::dict::{self, TokenInfoDictionary, UnknownDictionary, UserDictionary};
use crate::token::Token;
use crate::viterbi::{DecompoundMode, KoViterbi};

/// `KoreanTokenizer`.
pub struct KoreanTokenizer {
    atts: AttributeSource,
    input: TokenizerInput,
    viterbi: Viterbi<Token>,
    lang: KoViterbi,
}

impl Default for KoreanTokenizer {
    /// `new KoreanTokenizer()`: no user dictionary, `DISCARD`, no unknown
    /// unigrams, punctuation discarded.
    fn default() -> Self {
        Self::new(None, DecompoundMode::Discard, false, true)
    }
}

impl KoreanTokenizer {
    /// `new KoreanTokenizer(factory, userDictionary, mode,
    /// outputUnknownUnigrams, discardPunctuation)` over the default
    /// dictionaries.
    pub fn new(
        user_dictionary: Option<Arc<UserDictionary>>,
        mode: DecompoundMode,
        output_unknown_unigrams: bool,
        discard_punctuation: bool,
    ) -> Self {
        Self::with_dictionaries(
            TokenInfoDictionary::instance(),
            UnknownDictionary::instance(),
            dict::ConnectionCosts::instance(),
            user_dictionary,
            mode,
            output_unknown_unigrams,
            discard_punctuation,
        )
    }

    /// `new KoreanTokenizer(factory, systemDictionary, unkDictionary,
    /// connectionCosts, userDictionary, mode, outputUnknownUnigrams,
    /// discardPunctuation)`.
    pub fn with_dictionaries(
        system_dictionary: Arc<TokenInfoDictionary>,
        unk_dictionary: Arc<UnknownDictionary>,
        connection_costs: Arc<ConnectionCosts>,
        user_dictionary: Option<Arc<UserDictionary>>,
        mode: DecompoundMode,
        output_unknown_unigrams: bool,
        discard_punctuation: bool,
    ) -> Self {
        let fst = Arc::clone(system_dictionary.fst());
        let user_fst = user_dictionary.as_ref().map(|u| Arc::clone(u.fst()));
        let lang = KoViterbi::new(
            system_dictionary,
            unk_dictionary,
            user_dictionary,
            discard_punctuation,
            mode,
            output_unknown_unigrams,
        );
        let mut viterbi = Viterbi::new(fst, user_fst, connection_costs);
        viterbi.enable_space_penalty_factor = true;
        viterbi.output_longest_user_entry_only = true;
        viterbi.reset_buffer();
        viterbi.reset_state();
        let mut atts = AttributeSource::new();
        // Java's addAttribute order (after the core attributes).
        atts.add_custom::<PartOfSpeechAttribute>();
        atts.add_custom::<ReadingAttribute>();
        KoreanTokenizer {
            atts,
            input: TokenizerInput::new(),
            viterbi,
            lang,
        }
    }

    /// `setGraphvizFormatter(dotOut)`.
    pub fn set_graphviz_formatter(&mut self, dot_out: GraphvizFormatter) {
        self.lang.dot_out = Some(dot_out);
    }

    /// The formatter set by [`Self::set_graphviz_formatter`].
    pub fn graphviz(&self) -> Option<&GraphvizFormatter> {
        self.lang.dot_out.as_ref()
    }
}

impl TokenStream for KoreanTokenizer {
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
        // parse() is able to return w/o producing any new tokens, when the
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
        let base = token.base();
        let (start, end) = (base.start_offset, base.end_offset);
        let (pos_incr, pos_len) = (base.pos_incr, base.pos_len);
        self.atts.clear_attributes();
        self.atts.set_term_utf16(base.surface());
        let (cs, ce) = (
            self.input.correct_offset(start),
            self.input.correct_offset(end),
        );
        self.atts.set_offset(cs, ce)?;
        let token = Arc::new(token);
        self.atts.add_custom::<PartOfSpeechAttribute>().token = Some(Arc::clone(&token));
        self.atts.add_custom::<ReadingAttribute>().token = Some(token);
        self.atts.set_position_increment(pos_incr)?;
        self.atts.set_position_length(pos_len)?;
        Ok(true)
    }

    fn reset(&mut self) -> Result<(), AnalysisError> {
        self.input.reset();
        self.viterbi.reset_buffer();
        self.viterbi.reset_state();
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

impl Tokenizer for KoreanTokenizer {
    fn set_reader(&mut self, input: Box<dyn CharReader>) -> Result<(), AnalysisError> {
        self.input.set_reader(input)
    }
}
