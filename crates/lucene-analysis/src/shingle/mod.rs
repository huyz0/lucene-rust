//! `org.apache.lucene.analysis.shingle`: `ShingleFilter` and
//! `FixedShingleFilter`.

use std::collections::VecDeque;

use crate::attributes::{AttributeSource, State};
use crate::graph_token_filter::GraphTokenFilter;
use crate::token_stream::{TokenFilter, TokenStream};
use crate::AnalysisError;

/// `ShingleFilter.DEFAULT_FILLER_TOKEN`.
pub const DEFAULT_FILLER_TOKEN: &str = "_";
/// `ShingleFilter.DEFAULT_MAX_SHINGLE_SIZE`.
pub const DEFAULT_MAX_SHINGLE_SIZE: i32 = 2;
/// `ShingleFilter.DEFAULT_MIN_SHINGLE_SIZE`.
pub const DEFAULT_MIN_SHINGLE_SIZE: i32 = 2;
/// `ShingleFilter.DEFAULT_TOKEN_TYPE`.
pub const DEFAULT_TOKEN_TYPE: &str = "shingle";
/// `ShingleFilter.DEFAULT_TOKEN_SEPARATOR`.
pub const DEFAULT_TOKEN_SEPARATOR: &str = " ";

/// `ShingleFilter.CircularSequence`.
#[derive(Debug, Clone, Copy)]
struct CircularSequence {
    value: i32,
    previous_value: i32,
    min_value: i32,
}

impl CircularSequence {
    fn new(min_value: i32) -> Self {
        CircularSequence {
            value: min_value,
            previous_value: min_value,
            min_value,
        }
    }

    fn advance(&mut self, min_shingle: i32, max_shingle: i32) {
        self.previous_value = self.value;
        if self.value == 1 {
            self.value = min_shingle;
        } else if self.value == max_shingle {
            self.reset();
        } else {
            self.value += 1;
        }
    }

    fn reset(&mut self) {
        self.value = self.min_value;
        self.previous_value = self.min_value;
    }

    fn at_min_value(&self) -> bool {
        self.value == self.min_value
    }
}

/// `ShingleFilter.InputWindowToken`.
struct InputWindowToken {
    att_source: AttributeSource,
    is_filler: bool,
}

/// `org.apache.lucene.analysis.shingle.ShingleFilter`: word n-grams
/// ("shingles") of `minShingleSize..=maxShingleSize` tokens, optionally with
/// the unigrams, filling position gaps with `fillerToken`.
pub struct ShingleFilter<I> {
    input: I,
    input_window: VecDeque<InputWindowToken>,
    gram_size: CircularSequence,
    gram_builder: String,
    token_type: String,
    token_separator: String,
    filler_token: String,
    output_unigrams: bool,
    output_unigrams_if_no_shingles: bool,
    max_shingle_size: i32,
    min_shingle_size: i32,
    num_filler_tokens_to_insert: i32,
    next_input_stream_token: Option<AttributeSource>,
    is_next_input_stream_token: bool,
    is_output_here: bool,
    no_shingle_output: bool,
    end_state: Option<State>,
    exhausted: bool,
    /// Window tokens shifted out, reused by the next captures (Java's
    /// `getNextToken(target)` recycles the shifted-out token the same way).
    spare: Vec<AttributeSource>,
}

/// `captureState` into a recycled source when there is one.
fn capture(spare: &mut Vec<AttributeSource>, a: &AttributeSource) -> AttributeSource {
    match spare.pop() {
        Some(mut t) => {
            t.clone_from(a);
            t
        }
        None => a.clone(),
    }
}

impl<I: TokenStream> ShingleFilter<I> {
    /// `new ShingleFilter(TokenStream, int minShingleSize, int maxShingleSize)`.
    pub fn new(
        input: I,
        min_shingle_size: i32,
        max_shingle_size: i32,
    ) -> Result<Self, AnalysisError> {
        let mut f = ShingleFilter {
            input,
            input_window: VecDeque::new(),
            gram_size: CircularSequence::new(1),
            gram_builder: String::new(),
            token_type: DEFAULT_TOKEN_TYPE.to_string(),
            token_separator: DEFAULT_TOKEN_SEPARATOR.to_string(),
            filler_token: DEFAULT_FILLER_TOKEN.to_string(),
            output_unigrams: true,
            output_unigrams_if_no_shingles: false,
            max_shingle_size: 0,
            min_shingle_size: 0,
            num_filler_tokens_to_insert: 0,
            next_input_stream_token: None,
            is_next_input_stream_token: false,
            is_output_here: false,
            no_shingle_output: true,
            end_state: None,
            exhausted: false,
            spare: Vec::new(),
        };
        f.set_max_shingle_size(max_shingle_size)?;
        f.set_min_shingle_size(min_shingle_size)?;
        Ok(f)
    }

    fn new_sequence(&self) -> CircularSequence {
        CircularSequence::new(if self.output_unigrams {
            1
        } else {
            self.min_shingle_size
        })
    }

    /// `setTokenType`.
    pub fn set_token_type(&mut self, token_type: &str) {
        self.token_type = token_type.to_string();
    }

    /// `setOutputUnigrams`.
    pub fn set_output_unigrams(&mut self, output_unigrams: bool) {
        self.output_unigrams = output_unigrams;
        self.gram_size = self.new_sequence();
    }

    /// `setOutputUnigramsIfNoShingles`.
    pub fn set_output_unigrams_if_no_shingles(&mut self, v: bool) {
        self.output_unigrams_if_no_shingles = v;
    }

    /// `setMaxShingleSize`.
    pub fn set_max_shingle_size(&mut self, max: i32) -> Result<(), AnalysisError> {
        if max < 2 {
            return Err(AnalysisError::IllegalArgument(
                "Max shingle size must be >= 2".into(),
            ));
        }
        self.max_shingle_size = max;
        Ok(())
    }

    /// `setMinShingleSize`.
    pub fn set_min_shingle_size(&mut self, min: i32) -> Result<(), AnalysisError> {
        if min < 2 {
            return Err(AnalysisError::IllegalArgument(
                "Min shingle size must be >= 2".into(),
            ));
        }
        if min > self.max_shingle_size {
            return Err(AnalysisError::IllegalArgument(
                "Min shingle size must be <= max shingle size".into(),
            ));
        }
        self.min_shingle_size = min;
        self.gram_size = self.new_sequence();
        Ok(())
    }

    /// `setTokenSeparator` (`None` is `""`).
    pub fn set_token_separator(&mut self, sep: Option<&str>) {
        self.token_separator = sep.unwrap_or("").to_string();
    }

    /// `setFillerToken` (`None` is `""`).
    pub fn set_filler_token(&mut self, filler: Option<&str>) {
        self.filler_token = filler.unwrap_or("").to_string();
    }

    /// A filler: `source`'s attributes with an empty-width offset at its
    /// start and the filler term.
    fn filler_from(&self, source: &AttributeSource) -> Result<InputWindowToken, AnalysisError> {
        let mut a = source.clone();
        let start = a.start_offset();
        a.set_offset(start, start)?;
        a.set_term(&self.filler_token);
        Ok(InputWindowToken {
            att_source: a,
            is_filler: true,
        })
    }

    // Java: ShingleFilter.getNextToken (a recycled `target` is a new value
    // here; Java's copyTo overwrites every attribute it holds)
    fn get_next_token(&mut self) -> Result<Option<InputWindowToken>, AnalysisError> {
        if self.num_filler_tokens_to_insert > 0 {
            let next = self
                .next_input_stream_token
                .clone()
                .expect("set with the filler count");
            self.num_filler_tokens_to_insert -= 1;
            return Ok(Some(self.filler_from(&next)?));
        }
        if self.is_next_input_stream_token {
            self.is_next_input_stream_token = false;
            let next = self
                .next_input_stream_token
                .clone()
                .expect("set with the flag");
            return Ok(Some(InputWindowToken {
                att_source: next,
                is_filler: false,
            }));
        }
        if self.exhausted {
            return Ok(None);
        }
        if self.input.increment_token()? {
            let a = self.input.attributes();
            let inc = a.position_increment();
            if inc > 1 {
                self.num_filler_tokens_to_insert = (inc - 1).min(self.max_shingle_size - 1);
                self.next_input_stream_token = Some(a.capture_state());
                self.is_next_input_stream_token = true;
                let filler = self.filler_from(a)?;
                self.num_filler_tokens_to_insert -= 1;
                return Ok(Some(filler));
            }
            return Ok(Some(InputWindowToken {
                att_source: capture(&mut self.spare, a),
                is_filler: false,
            }));
        }
        self.exhausted = true;
        self.input.end()?;
        let a = self.input.attributes();
        self.end_state = Some(a.capture_state());
        self.num_filler_tokens_to_insert = a.position_increment().min(self.max_shingle_size - 1);
        if self.num_filler_tokens_to_insert > 0 {
            // Java: a fresh source holding only a term and the end offset.
            let mut next = AttributeSource::new();
            let end = a.end_offset();
            next.set_offset(end, end)?;
            self.next_input_stream_token = Some(next);
            return self.get_next_token();
        }
        Ok(None)
    }

    // Java: ShingleFilter.shiftInputWindow
    fn shift_input_window(&mut self) -> Result<(), AnalysisError> {
        if let Some(t) = self.input_window.pop_front() {
            self.spare.push(t.att_source);
        }
        while (self.input_window.len() as i32) < self.max_shingle_size {
            match self.get_next_token()? {
                Some(t) => self.input_window.push_back(t),
                None => break,
            }
        }
        if self.output_unigrams_if_no_shingles
            && self.no_shingle_output
            && self.gram_size.min_value > 1
            && (self.input_window.len() as i32) < self.min_shingle_size
        {
            self.gram_size.min_value = 1;
        }
        self.gram_size.reset();
        self.is_output_here = false;
        Ok(())
    }
}

impl<I: TokenStream> TokenFilter for ShingleFilter<I> {
    crate::filter_input!();

    // Java: ShingleFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        let mut token_available = false;
        let mut built_gram_size = 0;
        if self.gram_size.at_min_value() || (self.input_window.len() as i32) < self.gram_size.value
        {
            self.shift_input_window()?;
            self.gram_builder.clear();
        } else {
            built_gram_size = self.gram_size.previous_value;
        }
        if self.input_window.len() as i32 >= self.gram_size.value {
            let mut is_all_filler = true;
            let mut next_end_offset = 0;
            let (min, max) = (self.min_shingle_size, self.max_shingle_size);
            let mut gram_num = 1;
            let mut iter = self.input_window.iter();
            while built_gram_size < self.gram_size.value {
                let Some(next) = iter.next() else { break };
                if built_gram_size < gram_num {
                    if built_gram_size > 0 {
                        self.gram_builder.push_str(&self.token_separator);
                    }
                    self.gram_builder.push_str(next.att_source.term());
                    built_gram_size += 1;
                }
                if is_all_filler && next.is_filler {
                    if gram_num == self.gram_size.value {
                        self.gram_size.advance(min, max);
                    }
                } else {
                    is_all_filler = false;
                }
                next_end_offset = next.att_source.end_offset();
                gram_num += 1;
            }
            if !is_all_filler && built_gram_size == self.gram_size.value {
                let first = self
                    .input_window
                    .front()
                    .expect("non-empty window")
                    .att_source
                    .clone();
                let a = self.input.attributes_mut();
                a.restore_state(&first);
                a.set_position_increment(if self.is_output_here { 0 } else { 1 })?;
                a.set_term(&self.gram_builder);
                if self.gram_size.value > 1 {
                    a.set_token_type(self.token_type.clone());
                    self.no_shingle_output = false;
                }
                let start = a.start_offset();
                a.set_offset(start, next_end_offset)?;
                if self.output_unigrams {
                    a.set_position_length(built_gram_size)?;
                } else {
                    a.set_position_length(1.max(built_gram_size - self.min_shingle_size + 1))?;
                }
                self.is_output_here = true;
                self.gram_size.advance(min, max);
                token_available = true;
            }
        }
        Ok(token_available)
    }

    // Java: ShingleFilter.end
    fn end_filter(&mut self) -> Result<(), AnalysisError> {
        match &self.end_state {
            Some(s) if self.exhausted => {
                let s = s.clone();
                self.input.attributes_mut().restore_state(&s);
                Ok(())
            }
            _ => self.input.end(),
        }
    }

    // Java: ShingleFilter.reset
    fn reset_filter(&mut self) -> Result<(), AnalysisError> {
        self.input.reset()?;
        self.gram_size.reset();
        self.input_window.clear();
        self.next_input_stream_token = None;
        self.is_next_input_stream_token = false;
        self.num_filler_tokens_to_insert = 0;
        self.is_output_here = false;
        self.no_shingle_output = true;
        self.exhausted = false;
        self.end_state = None;
        if self.output_unigrams_if_no_shingles && !self.output_unigrams {
            self.gram_size.min_value = self.min_shingle_size;
        }
        Ok(())
    }
}

/// `FixedShingleFilter.MAX_SHINGLE_SIZE`.
const MAX_FIXED_SHINGLE_SIZE: i32 = 4;

/// `org.apache.lucene.analysis.shingle.FixedShingleFilter`: shingles of
/// exactly `shingleSize` positions over a token graph.
pub struct FixedShingleFilter<I> {
    graph: GraphTokenFilter<I>,
    shingle_size: i32,
    token_separator: String,
    filler_token: String,
}

impl<I: TokenStream> FixedShingleFilter<I> {
    /// `new FixedShingleFilter(TokenStream, int shingleSize)`: `" "`, `"_"`.
    pub fn new(input: I, shingle_size: i32) -> Result<Self, AnalysisError> {
        Self::with_separator(input, shingle_size, " ", "_")
    }

    /// `new FixedShingleFilter(TokenStream, int, String tokenSeparator, String fillerToken)`.
    pub fn with_separator(
        input: I,
        shingle_size: i32,
        sep: &str,
        filler: &str,
    ) -> Result<Self, AnalysisError> {
        if shingle_size <= 1 || shingle_size > MAX_FIXED_SHINGLE_SIZE {
            return Err(AnalysisError::IllegalArgument(format!(
                "Shingle size must be between 2 and {MAX_FIXED_SHINGLE_SIZE}, got {shingle_size}"
            )));
        }
        Ok(FixedShingleFilter {
            graph: GraphTokenFilter::new(input),
            shingle_size,
            token_separator: sep.to_string(),
            filler_token: filler.to_string(),
        })
    }
}

impl<I: TokenStream> TokenFilter for FixedShingleFilter<I> {
    type Input = GraphTokenFilter<I>;

    fn input(&self) -> &Self::Input {
        &self.graph
    }

    fn input_mut(&mut self) -> &mut Self::Input {
        &mut self.graph
    }

    // Java: FixedShingleFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        let (mut shingle_pos_inc, mut start_offset, mut end_offset);
        let mut buffer = String::new();
        'outer: loop {
            if !self.graph.increment_graph()? {
                if !self.graph.increment_base_token()? {
                    return Ok(false);
                }
                shingle_pos_inc = self.graph.attributes().position_increment();
            } else {
                shingle_pos_inc = 0;
            }
            let a = self.graph.attributes();
            start_offset = a.start_offset();
            end_offset = a.end_offset();
            buffer.clear();
            buffer.push_str(a.term());
            let mut i = 1;
            while i < self.shingle_size {
                if !self.graph.increment_graph_token()? {
                    let trailing = self.graph.trailing_positions();
                    if i + trailing < self.shingle_size {
                        continue 'outer;
                    }
                    while i < self.shingle_size {
                        buffer.push_str(&self.token_separator);
                        buffer.push_str(&self.filler_token);
                        i += 1;
                    }
                    break;
                }
                let a = self.graph.attributes();
                let mut pos_inc = a.position_increment();
                if pos_inc > 1 {
                    if i + pos_inc > self.shingle_size {
                        while i < self.shingle_size {
                            buffer.push_str(&self.token_separator);
                            buffer.push_str(&self.filler_token);
                            i += 1;
                        }
                        break;
                    }
                    while pos_inc > 1 {
                        buffer.push_str(&self.token_separator);
                        buffer.push_str(&self.filler_token);
                        pos_inc -= 1;
                        i += 1;
                    }
                }
                buffer.push_str(&self.token_separator);
                buffer.push_str(a.term());
                end_offset = a.end_offset();
                i += 1;
            }
            break;
        }
        let a = self.graph.attributes_mut();
        a.clear_attributes();
        a.set_offset(start_offset, end_offset)?;
        a.set_position_increment(shingle_pos_inc)?;
        a.set_term(&buffer);
        a.set_token_type("shingle");
        Ok(true)
    }
}

/// `org.apache.lucene.analysis.shingle.ShingleAnalyzerWrapper`: the
/// delegate's chain behind a configured [`ShingleFilter`]. Build the
/// [`crate::Analyzer`] with [`Self::into_analyzer`] (the delegate's reuse
/// strategy, as Java's constructor passes up).
pub struct ShingleAnalyzerWrapper {
    delegate: crate::Analyzer,
    min_shingle_size: i32,
    max_shingle_size: i32,
    token_separator: String,
    output_unigrams: bool,
    output_unigrams_if_no_shingles: bool,
    filler_token: Option<String>,
}

impl ShingleAnalyzerWrapper {
    /// `new ShingleAnalyzerWrapper(Analyzer, int minShingleSize, int
    /// maxShingleSize)`: separator `" "`, unigrams output, filler `"_"`.
    pub fn new(
        delegate: crate::Analyzer,
        min_shingle_size: i32,
        max_shingle_size: i32,
    ) -> Result<Self, AnalysisError> {
        Self::with_options(
            delegate,
            min_shingle_size,
            max_shingle_size,
            Some(DEFAULT_TOKEN_SEPARATOR),
            true,
            false,
            Some(DEFAULT_FILLER_TOKEN),
        )
    }

    /// The seven-argument constructor (`None` separator is `""`).
    pub fn with_options(
        delegate: crate::Analyzer,
        min_shingle_size: i32,
        max_shingle_size: i32,
        token_separator: Option<&str>,
        output_unigrams: bool,
        output_unigrams_if_no_shingles: bool,
        filler_token: Option<&str>,
    ) -> Result<Self, AnalysisError> {
        let bad = |m: &str| Err(AnalysisError::IllegalArgument(m.to_string()));
        if max_shingle_size < 2 {
            return bad("Max shingle size must be >= 2");
        }
        if min_shingle_size < 2 {
            return bad("Min shingle size must be >= 2");
        }
        if min_shingle_size > max_shingle_size {
            return bad("Min shingle size must be <= max shingle size");
        }
        Ok(ShingleAnalyzerWrapper {
            delegate,
            min_shingle_size,
            max_shingle_size,
            token_separator: token_separator.unwrap_or("").to_string(),
            output_unigrams,
            output_unigrams_if_no_shingles,
            filler_token: filler_token.map(str::to_string),
        })
    }

    /// The [`crate::Analyzer`], with the delegate's reuse strategy.
    pub fn into_analyzer(self) -> crate::Analyzer {
        let strategy = self.delegate.reuse_strategy();
        crate::Analyzer::with_reuse_strategy(self, strategy)
    }
}

impl crate::AnalyzerWrapper for ShingleAnalyzerWrapper {
    fn wrapped_analyzer(&self, _field_name: &str) -> &crate::Analyzer {
        &self.delegate
    }

    // Java: ShingleAnalyzerWrapper.wrapComponents
    fn wrap_components(
        &self,
        _field_name: &str,
        components: crate::TokenStreamComponents,
    ) -> crate::TokenStreamComponents {
        let (source, sink) = components.into_parts();
        let mut f = ShingleFilter::new(sink, self.min_shingle_size, self.max_shingle_size)
            .expect("shingle sizes checked in the constructor");
        f.set_token_separator(Some(&self.token_separator));
        f.set_output_unigrams(self.output_unigrams);
        f.set_output_unigrams_if_no_shingles(self.output_unigrams_if_no_shingles);
        f.set_filler_token(self.filler_token.as_deref());
        crate::TokenStreamComponents::from_parts(source, Box::new(f))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::canned::{render, Canned};

    #[test]
    fn shingles() {
        let mut f =
            ShingleFilter::new(Canned::parse("a:0:1:1:1 b:2:3:1:1 c:4:5:1:1|5|0"), 2, 3).unwrap();
        assert_eq!(
            render(&mut f),
            "a:0:1:1:1 a b:0:3:0:2 a b c:0:5:0:3 b:2:3:1:1 b c:2:5:0:2 c:4:5:1:1|5|0"
        );
        let mut f = ShingleFilter::new(Canned::parse("a:0:1:1:1 c:4:5:2:1|7|1"), 2, 2).unwrap();
        f.set_output_unigrams(false);
        f.set_token_separator(Some("+"));
        f.set_filler_token(Some("*"));
        f.set_token_type("sh");
        assert_eq!(render(&mut f), "a+*:0:4:1:1 *+c:4:5:1:1 c+*:4:7:1:1|7|1");
        let mut f = ShingleFilter::new(Canned::parse("a:0:1:1:1|1|0"), 3, 3).unwrap();
        f.set_output_unigrams(false);
        f.set_output_unigrams_if_no_shingles(true);
        f.set_token_separator(None);
        f.set_filler_token(None);
        assert_eq!(render(&mut f), "a:0:1:1:1|1|0");
        assert!(ShingleFilter::new(Canned::parse(""), 1, 2).is_err());
        assert!(ShingleFilter::new(Canned::parse(""), 2, 1).is_err());
        assert!(ShingleFilter::new(Canned::parse(""), 3, 2).is_err());
        let mut f = ShingleFilter::new(Canned::parse("a:0:1:1:1|4|0"), 2, 2).unwrap();
        f.reset().unwrap();
        f.end().unwrap();
        assert_eq!(f.attributes().end_offset(), 4);
    }

    #[test]
    fn fixed_shingles() {
        let mut f =
            FixedShingleFilter::new(Canned::parse("a:0:1:1:1 b:2:3:1:1 c:4:5:1:1|5|0"), 2).unwrap();
        assert_eq!(render(&mut f), "a b:0:3:1:1 b c:2:5:1:1|5|0");
        let mut f =
            FixedShingleFilter::new(Canned::parse("a:0:1:1:1 c:4:5:3:1 d:6:7:1:1|9|1"), 3).unwrap();
        assert_eq!(render(&mut f), "a _ _:0:1:1:1 c d _:4:7:3:1|9|1");
        assert!(FixedShingleFilter::new(Canned::parse(""), 5).is_err());
    }

    #[test]
    fn shingle_analyzer_wrapper_configures_the_filter() {
        use crate::core_analysis::WhitespaceAnalyzer;
        let terms = |a: &crate::Analyzer, text: &str| {
            let mut ts = a.token_stream("f", text).unwrap();
            let mut out = Vec::new();
            crate::token_stream::consume(&mut ts, |x| out.push(x.term().to_string())).unwrap();
            out
        };
        let ws = || crate::Analyzer::new(WhitespaceAnalyzer::default());
        let a = ShingleAnalyzerWrapper::new(ws(), 2, 2)
            .unwrap()
            .into_analyzer();
        assert_eq!(terms(&a, "a b"), vec!["a", "a b", "b"]);
        let a = ShingleAnalyzerWrapper::with_options(ws(), 2, 3, None, false, true, None)
            .unwrap()
            .into_analyzer();
        assert_eq!(terms(&a, "a b c"), vec!["ab", "abc", "bc"]);
        assert_eq!(terms(&a, "z"), vec!["z"]);
        for (min, max) in [(2, 1), (1, 2), (3, 2)] {
            assert!(ShingleAnalyzerWrapper::new(ws(), min, max).is_err());
        }
    }
}
