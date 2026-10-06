//! `org.apache.lucene.analysis.util` (analysis-common's helpers), plus
//! lucene-core's `RollingBuffer`, which the graph filters share.

pub mod char_tokenizer;
pub mod rolling_buffer;

pub use char_tokenizer::{
    from_separator_char_predicate, CharTokenizer, JavaLetter, LetterTokenizer, NotJavaWhitespace,
    NotUnicodeWhitespace, TokenChar, UnicodeWhitespaceTokenizer, WhitespaceTokenizer,
};
pub use rolling_buffer::{Resettable, RollingBuffer};

/// Test support: Lucene test-framework's `CannedTokenStream`, written as the
/// compact `term:start:end:posInc:posLen ...|finalOffset|finalPosInc` spec the
/// unit tests carry next to the output Java's filter gave for it.
#[cfg(test)]
pub(crate) mod canned {
    use crate::attributes::AttributeSource;
    use crate::token_stream::TokenStream;
    use crate::AnalysisError;

    /// `CannedTokenStream`.
    pub(crate) struct Canned {
        atts: AttributeSource,
        tokens: Vec<AttributeSource>,
        upto: usize,
        final_offset: i32,
        final_inc: i32,
    }

    impl Canned {
        /// Parses the spec (see the module docs).
        pub(crate) fn parse(spec: &str) -> Self {
            let mut parts = spec.split('|');
            let toks = parts.next().unwrap().trim();
            let final_offset = parts.next().map_or(0, |s| s.parse().unwrap());
            let final_inc = parts.next().map_or(0, |s| s.parse().unwrap());
            let tokens = toks
                .split(' ')
                .filter(|t| !t.is_empty())
                .map(|t| {
                    let f: Vec<&str> = t.rsplitn(5, ':').collect();
                    let mut a = AttributeSource::new();
                    a.set_term(f[4]);
                    a.set_offset(f[3].parse().unwrap(), f[2].parse().unwrap())
                        .unwrap();
                    a.set_position_increment(f[1].parse().unwrap()).unwrap();
                    a.set_position_length(f[0].parse().unwrap()).unwrap();
                    a
                })
                .collect();
            Canned {
                atts: AttributeSource::new(),
                tokens,
                upto: 0,
                final_offset,
                final_inc,
            }
        }
    }

    impl TokenStream for Canned {
        fn attributes(&self) -> &AttributeSource {
            &self.atts
        }
        fn attributes_mut(&mut self) -> &mut AttributeSource {
            &mut self.atts
        }
        fn increment_token(&mut self) -> Result<bool, AnalysisError> {
            match self.tokens.get(self.upto) {
                Some(t) => {
                    self.atts.restore_state(t);
                    self.upto += 1;
                    Ok(true)
                }
                None => Ok(false),
            }
        }
        fn reset(&mut self) -> Result<(), AnalysisError> {
            self.upto = 0;
            Ok(())
        }
        fn end(&mut self) -> Result<(), AnalysisError> {
            self.atts.end_attributes();
            self.atts.set_position_increment(self.final_inc)?;
            self.atts.set_offset(self.final_offset, self.final_offset)
        }
    }

    /// Runs `ts` and renders its output in the spec format.
    pub(crate) fn render(ts: &mut dyn TokenStream) -> String {
        let mut out = Vec::new();
        let end = crate::token_stream::consume(ts, |a| {
            out.push(format!(
                "{}:{}:{}:{}:{}",
                a.term(),
                a.start_offset(),
                a.end_offset(),
                a.position_increment(),
                a.position_length()
            ))
        })
        .unwrap();
        format!(
            "{}|{}|{}",
            out.join(" "),
            end.end_offset(),
            end.position_increment()
        )
    }
}
