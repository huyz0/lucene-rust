//! `org.apache.lucene.analysis.util.ElisionFilter`.

use std::sync::Arc;

use crate::token_stream::{TokenFilter, TokenStream};
use crate::{AnalysisError, CharArraySet};

/// `org.apache.lucene.analysis.util.ElisionFilter`: drops an article and the
/// first `'` or `’` after it (`l'avion` -> `avion`) when the part before the
/// apostrophe is in the set.
pub struct ElisionFilter<I> {
    input: I,
    articles: Arc<CharArraySet>,
}

impl<I: TokenStream> ElisionFilter<I> {
    /// `new ElisionFilter(TokenStream, CharArraySet)`.
    pub fn new(input: I, articles: Arc<CharArraySet>) -> Self {
        ElisionFilter { input, articles }
    }
}

impl<I: TokenStream> TokenFilter for ElisionFilter<I> {
    crate::filter_input!();
    // Java: ElisionFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if !self.input.increment_token()? {
            return Ok(false);
        }
        let a = self.input.attributes_mut();
        let term = a.term();
        if let Some(index) = term.find(['\'', '\u{2019}']) {
            if self.articles.contains(&term[..index]) {
                let apostrophe_len = term[index..].chars().next().map_or(1, char::len_utf8);
                let rest = term[index + apostrophe_len..].to_string();
                a.set_term(&rest);
            }
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::canned::{render, Canned};

    #[test]
    fn elides_listed_articles() {
        let articles = Arc::new(CharArraySet::from_words(["l", "qu"], true));
        let mut c = Canned::parse("x:0:1:1:1 y:2:3:1:1 z:4:5:1:1 w:6:7:1:1");
        c.set_terms(&["L'avion", "qu\u{2019}il", "aujourd'hui", "plain"]);
        let mut f = ElisionFilter::new(c, articles);
        assert_eq!(
            render(&mut f),
            "avion:0:1:1:1 il:2:3:1:1 aujourd'hui:4:5:1:1 plain:6:7:1:1|0|0"
        );
    }
}
