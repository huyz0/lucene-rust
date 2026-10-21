//! `MockSynonymFilter`: a hard-coded synonym graph -- "dogs" also emits
//! "dog", and "guinea pig" also emits the two-position "cavy".

use std::collections::VecDeque;

use lucene_analysis::attributes::AttributeSource;
use lucene_analysis::{AnalysisError, TokenFilter, TokenStream};

/// `org.apache.lucene.tests.analysis.MockSynonymFilter`.
pub struct MockSynonymFilter<I> {
    input: I,
    token_queue: VecDeque<AttributeSource>,
    end_of_input: bool,
}

impl<I: TokenStream> MockSynonymFilter<I> {
    /// `MockSynonymFilter(input)`.
    pub fn new(input: I) -> Self {
        MockSynonymFilter {
            input,
            token_queue: VecDeque::new(),
            end_of_input: false,
        }
    }

    /// `addSynonym`: the current token rewritten as the synonym, queued.
    fn add_synonym(
        &mut self,
        text: &str,
        pos_len: i32,
        end_offset: i32,
    ) -> Result<(), AnalysisError> {
        let atts = self.input.attributes_mut();
        atts.set_term(text);
        atts.set_position_increment(0)?;
        atts.set_position_length(pos_len)?;
        let start = atts.start_offset();
        atts.set_offset(start, end_offset)?;
        atts.set_token_type("SYNONYM");
        self.token_queue.push_back(atts.capture_state());
        Ok(())
    }

    /// `addSynonymAndRestoreOrigToken`.
    fn add_synonym_and_restore_orig_token(
        &mut self,
        text: &str,
        pos_len: i32,
        end_offset: i32,
    ) -> Result<(), AnalysisError> {
        let orig = self.input.attributes().capture_state();
        self.add_synonym(text, pos_len, end_offset)?;
        self.input.attributes_mut().restore_state(&orig);
        Ok(())
    }
}

impl<I: TokenStream> TokenFilter for MockSynonymFilter<I> {
    type Input = I;

    fn input(&self) -> &I {
        &self.input
    }

    fn input_mut(&mut self) -> &mut I {
        &mut self.input
    }

    fn reset_filter(&mut self) -> Result<(), AnalysisError> {
        self.input.reset()?;
        self.token_queue.clear();
        self.end_of_input = false;
        Ok(())
    }

    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if let Some(state) = self.token_queue.pop_front() {
            self.input.attributes_mut().restore_state(&state);
            return Ok(true);
        }
        if !self.end_of_input && self.input.increment_token()? {
            let term = self.input.attributes().term().to_string();
            if term == "dogs" {
                let end = self.input.attributes().end_offset();
                self.add_synonym_and_restore_orig_token("dog", 1, end)?;
            } else if term == "guinea" {
                let first_saved = self.input.attributes().capture_state();
                if self.input.increment_token()? {
                    let next = self.input.attributes().term().to_string();
                    if next == "pig" {
                        let second_saved = self.input.attributes().capture_state();
                        let second_end = self.input.attributes().end_offset();
                        self.input.attributes_mut().restore_state(&first_saved);
                        self.add_synonym("cavy", 2, second_end)?;
                        self.token_queue.push_back(second_saved);
                    } else if next == "dogs" {
                        self.token_queue
                            .push_back(self.input.attributes().capture_state());
                        let end = self.input.attributes().end_offset();
                        self.add_synonym("dog", 1, end)?;
                    }
                } else {
                    self.end_of_input = true;
                }
                self.input.attributes_mut().restore_state(&first_saved);
            }
            Ok(true)
        } else {
            self.end_of_input = true;
            Ok(false)
        }
    }
}
