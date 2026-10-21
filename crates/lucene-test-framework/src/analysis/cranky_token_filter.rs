//! `CrankyTokenFilter`: throws fake I/O errors from `incrementToken`,
//! `end`, `reset` or `close` at random -- for testing that a consumer
//! survives a failing analysis chain.

use lucene_analysis::{AnalysisError, TokenFilter, TokenStream};
use lucene_util::java_random::JavaRandom;

/// `org.apache.lucene.tests.analysis.CrankyTokenFilter`.
pub struct CrankyTokenFilter<I> {
    input: I,
    random: JavaRandom,
    thing_to_do: i32,
}

impl<I: TokenStream> CrankyTokenFilter<I> {
    /// `CrankyTokenFilter(input, random)`.
    pub fn new(input: I, random: JavaRandom) -> Self {
        CrankyTokenFilter {
            input,
            random,
            thing_to_do: 0,
        }
    }

    fn fake(what: &str) -> AnalysisError {
        AnalysisError::Io(format!("Fake IOException from TokenStream.{what}()"))
    }
}

impl<I: TokenStream> TokenFilter for CrankyTokenFilter<I> {
    type Input = I;

    fn input(&self) -> &I {
        &self.input
    }

    fn input_mut(&mut self) -> &mut I {
        &mut self.input
    }

    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if self.thing_to_do == 0 && self.random.next_boolean() {
            return Err(Self::fake("incrementToken"));
        }
        self.input.increment_token()
    }

    fn end_filter(&mut self) -> Result<(), AnalysisError> {
        self.input.end()?;
        if self.thing_to_do == 1 && self.random.next_boolean() {
            return Err(Self::fake("end"));
        }
        Ok(())
    }

    fn reset_filter(&mut self) -> Result<(), AnalysisError> {
        self.input.reset()?;
        self.thing_to_do = self.random.next_int_bounded(100);
        if self.thing_to_do == 2 && self.random.next_boolean() {
            return Err(Self::fake("reset"));
        }
        Ok(())
    }

    fn close_filter(&mut self) -> Result<(), AnalysisError> {
        self.input.close()?;
        if self.thing_to_do == 3 && self.random.next_boolean() {
            return Err(Self::fake("close"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::mock_tokenizer::MockTokenizer;
    use lucene_analysis::Tokenizer;

    /// Runs the whole workflow once; the fake error's method, if any.
    fn once(f: &mut CrankyTokenFilter<MockTokenizer>) -> Option<String> {
        f.input_mut().set_enable_checks(false);
        f.input_mut()
            .set_reader(Box::new(lucene_analysis::StrReader::new("a b")))
            .unwrap();
        let run = (|| {
            f.reset()?;
            while f.increment_token()? {}
            f.end()
        })();
        let closed = f.close();
        run.and(closed).err().map(|e| e.to_string())
    }

    #[test]
    fn every_fault_fires() {
        let mut f = CrankyTokenFilter::new(MockTokenizer::default(), JavaRandom::new(42));
        let mut seen = std::collections::BTreeSet::new();
        for _ in 0..5000 {
            if let Some(e) = once(&mut f) {
                assert!(e.contains("Fake IOException from TokenStream."), "{e}");
                seen.insert(e);
            }
        }
        for what in ["incrementToken", "end", "reset", "close"] {
            assert!(
                seen.iter().any(|e| e.contains(&format!(".{what}()"))),
                "{what} never failed: {seen:?}"
            );
        }
    }
}
