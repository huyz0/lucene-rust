//! `org.apache.lucene.analysis.commongrams`: `CommonGramsFilter` and
//! `CommonGramsQueryFilter`.

use std::sync::Arc;

use crate::attributes::State;
use crate::token_stream::{TokenFilter, TokenStream};
use crate::{AnalysisError, CharArraySet};

/// `CommonGramsFilter.GRAM_TYPE`.
pub const GRAM_TYPE: &str = "gram";
const SEPARATOR: char = '_';

/// `org.apache.lucene.analysis.commongrams.CommonGramsFilter`: a bigram
/// (`the_quick`) stacked on every token next to a common word.
pub struct CommonGramsFilter<I> {
    input: I,
    common_words: Option<Arc<CharArraySet>>,
    buffer: String,
    last_start_offset: i32,
    last_was_common: bool,
    saved_state: Option<State>,
}

impl<I: TokenStream> CommonGramsFilter<I> {
    /// `new CommonGramsFilter(TokenStream, CharArraySet commonWords)`.
    pub fn new(input: I, common_words: Option<Arc<CharArraySet>>) -> Self {
        CommonGramsFilter {
            input,
            common_words,
            buffer: String::new(),
            last_start_offset: 0,
            last_was_common: false,
            saved_state: None,
        }
    }

    fn is_common(&self) -> bool {
        let term = self.input.attributes().term();
        self.common_words.as_ref().is_some_and(|w| w.contains(term))
    }

    // Java: CommonGramsFilter.saveTermBuffer
    fn save_term_buffer(&mut self) {
        let a = self.input.attributes();
        self.buffer.clear();
        self.buffer.push_str(a.term());
        self.buffer.push(SEPARATOR);
        self.last_start_offset = a.start_offset();
        self.last_was_common = self.is_common();
    }

    // Java: CommonGramsFilter.gramToken
    fn gram_token(&mut self) -> Result<(), AnalysisError> {
        let a = self.input.attributes_mut();
        self.buffer.push_str(a.term());
        let end_offset = a.end_offset();
        a.clear_attributes();
        a.set_term(&self.buffer);
        a.set_position_increment(0)?;
        a.set_position_length(2)?;
        a.set_offset(self.last_start_offset, end_offset)?;
        a.set_token_type(GRAM_TYPE);
        self.buffer.clear();
        Ok(())
    }
}

impl<I: TokenStream> TokenFilter for CommonGramsFilter<I> {
    crate::filter_input!();

    // Java: CommonGramsFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if let Some(saved) = self.saved_state.take() {
            self.input.attributes_mut().restore_state(&saved);
            self.save_term_buffer();
            return Ok(true);
        } else if !self.input.increment_token()? {
            return Ok(false);
        }
        if self.last_was_common || (self.is_common() && !self.buffer.is_empty()) {
            self.saved_state = Some(self.input.attributes().capture_state());
            self.gram_token()?;
            return Ok(true);
        }
        self.save_term_buffer();
        Ok(true)
    }

    fn reset_filter(&mut self) -> Result<(), AnalysisError> {
        self.input.reset()?;
        self.last_was_common = false;
        self.saved_state = None;
        self.buffer.clear();
        Ok(())
    }
}

/// `org.apache.lucene.analysis.commongrams.CommonGramsQueryFilter`: keeps the
/// grams, and the unigrams no gram covers.
pub struct CommonGramsQueryFilter<I> {
    input: CommonGramsFilter<I>,
    previous: Option<State>,
    previous_type: Option<String>,
    exhausted: bool,
}

impl<I: TokenStream> CommonGramsQueryFilter<I> {
    /// `new CommonGramsQueryFilter(CommonGramsFilter)`.
    pub fn new(input: CommonGramsFilter<I>) -> Self {
        CommonGramsQueryFilter {
            input,
            previous: None,
            previous_type: None,
            exhausted: false,
        }
    }

    fn is_gram_type(&self) -> bool {
        self.input.attributes().token_type() == GRAM_TYPE
    }

    fn unstack_gram(&mut self) -> Result<(), AnalysisError> {
        if self.is_gram_type() {
            let a = self.input.attributes_mut();
            a.set_position_increment(1)?;
            a.set_position_length(1)?;
        }
        Ok(())
    }
}

impl<I: TokenStream> TokenFilter for CommonGramsQueryFilter<I> {
    type Input = CommonGramsFilter<I>;

    fn input(&self) -> &Self::Input {
        &self.input
    }

    fn input_mut(&mut self) -> &mut Self::Input {
        &mut self.input
    }

    // Java: CommonGramsQueryFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        while !self.exhausted && self.input.increment_token()? {
            let current = self.input.attributes().capture_state();
            if self.previous.is_some() && !self.is_gram_type() {
                let prev = self.previous.replace(current).expect("checked");
                self.input.attributes_mut().restore_state(&prev);
                self.previous_type = Some(self.input.attributes().token_type().to_string());
                self.unstack_gram()?;
                return Ok(true);
            }
            self.previous = Some(current);
        }
        self.exhausted = true;
        if self.previous.is_none() || self.previous_type.as_deref() == Some(GRAM_TYPE) {
            return Ok(false);
        }
        let prev = self.previous.take().expect("checked");
        self.input.attributes_mut().restore_state(&prev);
        self.unstack_gram()?;
        Ok(true)
    }

    fn reset_filter(&mut self) -> Result<(), AnalysisError> {
        self.input.reset()?;
        self.previous = None;
        self.previous_type = None;
        self.exhausted = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::canned::{render, Canned};

    fn common() -> Option<Arc<CharArraySet>> {
        Some(Arc::new(CharArraySet::from_words(["the", "of"], false)))
    }

    const IN: &str = "the:0:3:1:1 quick:4:9:1:1 of:10:12:1:1 x:13:14:1:1|14|0";

    #[test]
    fn index_and_query_grams() {
        let mut f = CommonGramsFilter::new(Canned::parse(IN), common());
        assert_eq!(
            render(&mut f),
            "the:0:3:1:1 the_quick:0:9:0:2 quick:4:9:1:1 quick_of:4:12:0:2 of:10:12:1:1 of_x:10:14:0:2 x:13:14:1:1|14|0"
        );
        let mut f =
            CommonGramsQueryFilter::new(CommonGramsFilter::new(Canned::parse(IN), common()));
        assert_eq!(
            render(&mut f),
            "the_quick:0:9:1:1 quick_of:4:12:1:1 of_x:10:14:1:1|14|0"
        );
        let mut f = CommonGramsQueryFilter::new(CommonGramsFilter::new(
            Canned::parse("a:0:1:1:1 b:2:3:1:1|3|0"),
            None,
        ));
        assert_eq!(render(&mut f), "a:0:1:1:1 b:2:3:1:1|3|0");
        let mut f = CommonGramsQueryFilter::new(CommonGramsFilter::new(
            Canned::parse("the:0:3:1:1|3|0"),
            common(),
        ));
        assert_eq!(render(&mut f), "the:0:3:1:1|3|0");
    }
}
