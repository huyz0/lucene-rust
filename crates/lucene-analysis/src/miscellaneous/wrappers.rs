//! `EmptyTokenStream`, `ConcatenatingTokenStream`, `LimitTokenCountAnalyzer`,
//! `PerFieldAnalyzerWrapper`.

use std::collections::HashMap;

use super::LimitTokenCountFilter;
use crate::analyzer::{Analyzer, AnalyzerWrapper, DelegatingAnalyzerWrapper, ReuseStrategy};
use crate::attributes::AttributeSource;
use crate::token_stream::TokenStream;
use crate::{AnalysisError, TokenStreamComponents};

/// `org.apache.lucene.analysis.miscellaneous.EmptyTokenStream`: a stream
/// with no tokens.
#[derive(Debug, Default)]
pub struct EmptyTokenStream {
    atts: AttributeSource,
}

impl EmptyTokenStream {
    /// `new EmptyTokenStream()`.
    pub fn new() -> Self {
        Self::default()
    }
}

impl TokenStream for EmptyTokenStream {
    /// A source, not a wrapper: no conditional wrapper below it.
    fn conditional_root(&mut self) -> Option<&mut dyn std::any::Any> {
        None
    }
    fn attributes(&self) -> &AttributeSource {
        &self.atts
    }
    fn attributes_mut(&mut self) -> &mut AttributeSource {
        &mut self.atts
    }
    fn increment_token(&mut self) -> Result<bool, AnalysisError> {
        Ok(false)
    }
}

/// `org.apache.lucene.analysis.miscellaneous.ConcatenatingTokenStream`: the
/// tokens of each source in turn, the offsets of each shifted by the sum of
/// the earlier sources' final end offsets, and the first token of each later
/// source's position increment raised by the previous source's final one.
///
/// Java merges the sources' attribute *classes* and fails on a clash; every
/// [`AttributeSource`] here has the same fixed set, so `copyTo` is a clone.
pub struct ConcatenatingTokenStream {
    sources: Vec<Box<dyn TokenStream>>,
    atts: AttributeSource,
    current_source: usize,
    offset_increment: i32,
    initial_position_increment: i32,
}

impl ConcatenatingTokenStream {
    /// `new ConcatenatingTokenStream(TokenStream...)`. Java's
    /// `combineSources` reads `sources[0]` and throws
    /// `ArrayIndexOutOfBoundsException` on none; that is an
    /// `IllegalArgument` here.
    pub fn new(sources: Vec<Box<dyn TokenStream>>) -> Result<Self, AnalysisError> {
        let atts = match sources.first() {
            Some(s) => s.attributes().clone(),
            None => {
                return Err(AnalysisError::IllegalArgument(
                    "ConcatenatingTokenStream needs at least one source".into(),
                ))
            }
        };
        Ok(ConcatenatingTokenStream {
            sources,
            atts,
            current_source: 0,
            offset_increment: 0,
            initial_position_increment: 1,
        })
    }
}

impl TokenStream for ConcatenatingTokenStream {
    /// Java's concatenation reads independent sources, none of them a
    /// conditional filter's delegate chain: no wrapper below it.
    fn conditional_root(&mut self) -> Option<&mut dyn std::any::Any> {
        None
    }
    fn attributes(&self) -> &AttributeSource {
        &self.atts
    }
    fn attributes_mut(&mut self) -> &mut AttributeSource {
        &mut self.atts
    }

    // Java: ConcatenatingTokenStream.incrementToken
    fn increment_token(&mut self) -> Result<bool, AnalysisError> {
        let mut new_source = false;
        while !self.sources[self.current_source].increment_token()? {
            if self.current_source + 1 >= self.sources.len() {
                return Ok(false);
            }
            let src = &mut self.sources[self.current_source];
            src.end()?;
            self.initial_position_increment = src.attributes().position_increment();
            self.offset_increment = self
                .offset_increment
                .wrapping_add(src.attributes().end_offset());
            self.current_source += 1;
            new_source = true;
        }
        self.atts
            .clone_from(self.sources[self.current_source].attributes());
        let (start, end) = (self.atts.start_offset(), self.atts.end_offset());
        self.atts.set_offset(
            start.wrapping_add(self.offset_increment),
            end.wrapping_add(self.offset_increment),
        )?;
        if new_source {
            let inc = self.atts.position_increment();
            self.atts
                .set_position_increment(inc.wrapping_add(self.initial_position_increment))?;
        }
        Ok(true)
    }

    // Java: ConcatenatingTokenStream.end
    fn end(&mut self) -> Result<(), AnalysisError> {
        let src = &mut self.sources[self.current_source];
        src.end()?;
        let final_offset = src
            .attributes()
            .end_offset()
            .wrapping_add(self.offset_increment);
        let final_pos_inc = src.attributes().position_increment();
        self.atts.end_attributes();
        self.atts.set_offset(final_offset, final_offset)?;
        self.atts.set_position_increment(final_pos_inc)
    }

    // Java: ConcatenatingTokenStream.reset -- `initialPositionIncrement` is
    // not reset, as in Java.
    fn reset(&mut self) -> Result<(), AnalysisError> {
        for s in &mut self.sources {
            s.reset()?;
        }
        self.current_source = 0;
        self.offset_increment = 0;
        Ok(())
    }

    // Java: IOUtils.close(sources) -- every source is closed, the first
    // failure reported.
    fn close(&mut self) -> Result<(), AnalysisError> {
        let mut first = Ok(());
        for s in &mut self.sources {
            let r = s.close();
            if first.is_ok() {
                first = r;
            }
        }
        first
    }
}

/// `org.apache.lucene.analysis.miscellaneous.LimitTokenCountAnalyzer`: the
/// delegate's chain behind a [`LimitTokenCountFilter`]. Build the
/// [`Analyzer`] with [`Self::into_analyzer`] (Java's constructor passes the
/// delegate's reuse strategy up).
pub struct LimitTokenCountAnalyzer {
    delegate: Analyzer,
    max_token_count: i32,
    consume_all_tokens: bool,
}

impl LimitTokenCountAnalyzer {
    /// `new LimitTokenCountAnalyzer(Analyzer, int, boolean)`. Java checks
    /// `maxTokenCount` when the filter is built, on the first `tokenStream`;
    /// the check is made here, once.
    pub fn new(
        delegate: Analyzer,
        max_token_count: i32,
        consume_all_tokens: bool,
    ) -> Result<Self, AnalysisError> {
        if max_token_count < 1 {
            return Err(AnalysisError::IllegalArgument(
                "maxTokenCount must be greater than zero".into(),
            ));
        }
        Ok(LimitTokenCountAnalyzer {
            delegate,
            max_token_count,
            consume_all_tokens,
        })
    }

    /// The [`Analyzer`], with the delegate's reuse strategy.
    pub fn into_analyzer(self) -> Analyzer {
        let strategy = self.delegate.reuse_strategy();
        Analyzer::with_reuse_strategy(self, strategy)
    }
}

impl AnalyzerWrapper for LimitTokenCountAnalyzer {
    fn wrapped_analyzer(&self, _field_name: &str) -> &Analyzer {
        &self.delegate
    }

    fn wrap_components(
        &self,
        _field_name: &str,
        components: TokenStreamComponents,
    ) -> TokenStreamComponents {
        let (source, sink) = components.into_parts();
        let filter =
            LimitTokenCountFilter::new(sink, self.max_token_count, self.consume_all_tokens)
                .expect("maxTokenCount checked in new");
        TokenStreamComponents::from_parts(source, Box::new(filter))
    }
}

/// `org.apache.lucene.analysis.miscellaneous.PerFieldAnalyzerWrapper`: each
/// field's own analyzer, or the default. Build the [`Analyzer`] with
/// [`Self::into_analyzer`] (`PER_FIELD_REUSE_STRATEGY`, as in Java).
pub struct PerFieldAnalyzerWrapper {
    default_analyzer: Analyzer,
    field_analyzers: HashMap<String, Analyzer>,
}

impl PerFieldAnalyzerWrapper {
    /// `new PerFieldAnalyzerWrapper(Analyzer, Map<String, Analyzer>)`.
    pub fn new(default_analyzer: Analyzer, field_analyzers: HashMap<String, Analyzer>) -> Self {
        PerFieldAnalyzerWrapper {
            default_analyzer,
            field_analyzers,
        }
    }

    /// The [`Analyzer`] (a `DelegatingAnalyzerWrapper`).
    pub fn into_analyzer(self) -> Analyzer {
        Analyzer::delegating(self, ReuseStrategy::PerField)
    }
}

impl DelegatingAnalyzerWrapper for PerFieldAnalyzerWrapper {
    fn wrapped_analyzer(&self, field_name: &str) -> &Analyzer {
        self.field_analyzers
            .get(field_name)
            .unwrap_or(&self.default_analyzer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core_analysis::{SimpleAnalyzer, WhitespaceAnalyzer};
    use std::sync::Arc;

    use crate::util::canned::render;
    use crate::util::WhitespaceTokenizer;
    use crate::CharArraySet;
    use crate::StrReader;
    use crate::Tokenizer;

    fn ws(text: &str) -> Box<dyn TokenStream> {
        let mut t = WhitespaceTokenizer::new();
        t.set_reader(Box::new(StrReader::new(text))).unwrap();
        Box::new(t)
    }

    fn terms(a: &Analyzer, field: &str, text: &str) -> Vec<String> {
        let mut ts = a.token_stream(field, text).unwrap();
        let mut out = Vec::new();
        crate::token_stream::consume(&mut ts, |x| out.push(x.term().to_string())).unwrap();
        out
    }

    #[test]
    fn empty_token_stream_has_no_tokens() {
        assert_eq!(render(&mut EmptyTokenStream::new()), "|0|0");
    }

    #[test]
    fn concatenating_shifts_offsets_and_positions() {
        // Lucene 10.5.0: "a b " + "c" + "d  e" -> a:0:1:1 b:2:3:1 c:4:5:1
        // d:5:6:1 e:8:9:1, end 9, posInc 0.
        let mut c = ConcatenatingTokenStream::new(vec![ws("a b "), ws("c"), ws("d  e")]).unwrap();
        assert_eq!(
            render(&mut c),
            "a:0:1:1:1 b:2:3:1:1 c:4:5:1:1 d:5:6:1:1 e:8:9:1:1|9|0"
        );
    }

    fn stop_of(text: &str) -> Box<dyn TokenStream> {
        let of = Arc::new(CharArraySet::from_words(["of"], false));
        Box::new(crate::StopFilter::new(ws(text), of))
    }

    #[test]
    fn concatenating_carries_the_final_position_increment() {
        // Lucene 10.5.0, a source ending in a removed stopword (final
        // posInc 1) then "y": y's increment is 1 + 1.
        let mut c = ConcatenatingTokenStream::new(vec![stop_of("x of "), ws("y")]).unwrap();
        assert_eq!(render(&mut c), "x:0:1:1:1 y:5:6:2:1|6|0");
        // An EmptyTokenStream between them ends with posInc 0 and offset 0.
        let mut c = ConcatenatingTokenStream::new(vec![
            stop_of("x of "),
            Box::new(EmptyTokenStream::new()),
            ws("of y"),
        ])
        .unwrap();
        assert_eq!(render(&mut c), "x:0:1:1:1 of:5:7:1:1 y:8:9:1:1|9|0");
        c.close().unwrap();
        // The last source's final increment is the stream's.
        let mut c = ConcatenatingTokenStream::new(vec![ws("p"), stop_of("of")]).unwrap();
        assert_eq!(render(&mut c), "p:0:1:1:1|3|1");
    }

    #[test]
    fn concatenating_needs_a_source() {
        assert!(matches!(
            ConcatenatingTokenStream::new(Vec::new()),
            Err(AnalysisError::IllegalArgument(_))
        ));
    }

    #[test]
    fn limit_token_count_analyzer_limits_each_stream() {
        let a =
            LimitTokenCountAnalyzer::new(Analyzer::new(WhitespaceAnalyzer::default()), 2, false)
                .unwrap()
                .into_analyzer();
        assert_eq!(terms(&a, "f", "a b c d"), vec!["a", "b"]);
        assert_eq!(terms(&a, "f", "e"), vec!["e"]);
        let all = LimitTokenCountAnalyzer::new(Analyzer::new(SimpleAnalyzer), 1, true)
            .unwrap()
            .into_analyzer();
        assert_eq!(terms(&all, "f", "X y z"), vec!["x"]);
        assert!(matches!(
            LimitTokenCountAnalyzer::new(Analyzer::new(SimpleAnalyzer), 0, false),
            Err(AnalysisError::IllegalArgument(_))
        ));
    }

    #[test]
    fn per_field_wrapper_picks_the_field_analyzer() {
        let mut fields = HashMap::new();
        fields.insert("simple".to_string(), Analyzer::new(SimpleAnalyzer));
        let a = PerFieldAnalyzerWrapper::new(Analyzer::new(WhitespaceAnalyzer::default()), fields)
            .into_analyzer();
        assert_eq!(terms(&a, "simple", "Foo-Bar"), vec!["foo", "bar"]);
        assert_eq!(terms(&a, "other", "Foo-Bar"), vec!["Foo-Bar"]);
        assert_eq!(a.reuse_strategy(), ReuseStrategy::PerField);
    }
}
