//! OpenSearch's `plain` highlighter (`PlainHighlighter`) over one value of a
//! field: the classic [`Highlighter`] with OpenSearch's `CustomQueryScorer`
//! ([`QueryScorer::opensearch`]), a `SimpleHTMLFormatter` of the field's
//! tags, the `default` or `html` encoder and the field's fragmenter, the
//! value re-analyzed with the field's index analyzer and read whole
//! (`setMaxDocCharsToAnalyze(Integer.MAX_VALUE)`). What OpenSearch does
//! around it -- loading the values, the `max_analyzed_offset` checks,
//! ordering the fragments of every value and picking how many to show --
//! stays with the caller; the `no_match_size` summary's cut is
//! [`PlainHighlighter::no_match_end`].
//!
//! OpenSearch keeps one highlighter per field for a whole fetch phase, and
//! its `SimpleSpanFragmenter` does not reset `waitForPos` between texts: a
//! span read to the end of one value keeps the next value (of the hit, or of
//! the next hit) from starting a fragment until that position.
//! [`PlainHighlighter::fragments_carrying`] takes and gives back that state.

use std::cell::Cell;
use std::rc::Rc;
use std::sync::Arc;

use lucene_analysis::{Analyzer, AttributeSource, TokenStream};
use lucene_search::query::Clause;

use crate::{
    DefaultEncoder, DynStream, Encoder, Fragmenter, Highlighter, NullFragmenter, QueryScorer,
    Result, SimpleFragmenter, SimpleHtmlEncoder, SimpleHtmlFormatter, SimpleSpanFragmenter,
};

/// A `SimpleSpanFragmenter` whose `waitForPos` is read back after each token.
struct Carrying {
    inner: SimpleSpanFragmenter,
    wait_for_pos: Rc<Cell<i32>>,
}

impl Fragmenter for Carrying {
    fn start(&mut self, original_text: &str) {
        self.inner.start(original_text);
    }

    fn is_new_fragment(&mut self, token: &AttributeSource) -> bool {
        let new = self.inner.is_new_fragment(token);
        self.wait_for_pos.set(self.inner.wait_for_pos());
        new
    }
}

/// The `fragmenter` option, as `PlainHighlighter` picks it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlainFragmenter {
    /// `number_of_fragments: 0`: the value whole (`NullFragmenter`).
    Null,
    /// `span`, the default: `SimpleSpanFragmenter` of `fragment_size`.
    Span(i32),
    /// `simple`: `SimpleFragmenter` of `fragment_size`.
    Simple(i32),
}

/// A field's highlighting options.
#[derive(Debug, Clone)]
pub struct PlainOptions {
    /// The field: the analyzer's, and with `require_field_match` the only
    /// one whose query terms count.
    pub field: String,
    pub require_field_match: bool,
    pub fragmenter: PlainFragmenter,
    /// `number_of_fragments`; 0 reads each value as one fragment.
    pub number_of_fragments: i32,
    pub pre_tag: String,
    pub post_tag: String,
    /// The `html` encoder (`SimpleHTMLEncoder`), else `DefaultEncoder`.
    pub html: bool,
}

/// One field's `plain` highlighter.
pub struct PlainHighlighter {
    options: PlainOptions,
    query: Clause,
    analyzer: Arc<Analyzer>,
}

/// A fragment that scored: its score and marked-up text.
#[derive(Debug, Clone, PartialEq)]
pub struct PlainFragment {
    pub score: f32,
    pub text: String,
}

impl PlainHighlighter {
    /// A highlighter of `query` over values `analyzer` analyzes as
    /// `options.field`.
    pub fn new(options: PlainOptions, query: Clause, analyzer: Arc<Analyzer>) -> Self {
        PlainHighlighter {
            options,
            query,
            analyzer,
        }
    }

    /// The options.
    pub fn options(&self) -> &PlainOptions {
        &self.options
    }

    /// `getBestTextFragments(tokenStream, text, false, n)` over one value,
    /// `n` being `number_of_fragments` (1 when it is 0): the fragments that
    /// score above 0, in the highlighter's order.
    ///
    /// # Errors
    /// The analysis chain's; [`crate::Error::InvalidTokenOffsets`] when a
    /// token reaches past the text.
    pub fn fragments(&self, text: &str) -> Result<Vec<PlainFragment>> {
        self.fragments_carrying(text, &mut -1)
    }

    /// [`Self::fragments`] with the span fragmenter's `waitForPos` as the
    /// previous text left it (-1 for none), updated to what this one leaves.
    ///
    /// # Errors
    /// As [`Self::fragments`].
    pub fn fragments_carrying(
        &self,
        text: &str,
        wait_for_pos: &mut i32,
    ) -> Result<Vec<PlainFragment>> {
        let o = &self.options;
        let field = o.require_field_match.then_some(o.field.as_str());
        let mut scorer = QueryScorer::opensearch(&self.query, field);
        scorer.set_expand_multi_term_query(true);
        let carried = Rc::new(Cell::new(*wait_for_pos));
        let fragmenter: Box<dyn Fragmenter> = match o.fragmenter {
            PlainFragmenter::Null => Box::new(NullFragmenter),
            PlainFragmenter::Span(size) => {
                let mut inner = SimpleSpanFragmenter::with_size(&scorer, size);
                inner.set_wait_for_pos(*wait_for_pos);
                Box::new(Carrying {
                    inner,
                    wait_for_pos: Rc::clone(&carried),
                })
            }
            PlainFragmenter::Simple(size) => Box::new(SimpleFragmenter::new(size)),
        };
        let formatter = SimpleHtmlFormatter::new(o.pre_tag.clone(), o.post_tag.clone());
        let encoder: Box<dyn Encoder> = if o.html {
            Box::new(SimpleHtmlEncoder)
        } else {
            Box::new(DefaultEncoder)
        };
        let mut highlighter = Highlighter::with_encoder(formatter, encoder, scorer);
        highlighter.set_text_fragmenter(fragmenter);
        highlighter.set_max_doc_chars_to_analyze(i32::MAX);
        let n = usize::try_from(o.number_of_fragments.max(1)).unwrap_or(1);
        let stream = DynStream::new(self.analyzer.token_stream(&o.field, text)?);
        let fragments = highlighter.get_best_text_fragments(stream, text, false, n)?;
        *wait_for_pos = carried.get();
        Ok(fragments
            .iter()
            .filter(|f| f.score() > 0.0)
            .map(|f| PlainFragment {
                score: f.score(),
                text: f.to_string(),
            })
            .collect())
    }

    /// `findGoodEndForNoHighlightExcerpt(noMatchSize, analyzer, field,
    /// contents)`: where the `no_match_size` summary of `text` ends -- at
    /// `no_match_size` when a token ends there, else at the end of the last
    /// token before it, or of the last token at all; -1 when no token ends
    /// before it.
    ///
    /// # Errors
    /// The analysis chain's.
    pub fn no_match_end(&self, no_match_size: i32, text: &str) -> Result<i32> {
        let mut stream = self.analyzer.token_stream(&self.options.field, text)?;
        let mut end = -1;
        stream.reset()?;
        while stream.increment_token()? {
            let token_end = stream.attributes().end_offset();
            if token_end >= no_match_size {
                if token_end == no_match_size {
                    end = no_match_size;
                }
                stream.close()?;
                return Ok(end);
            }
            end = token_end;
        }
        stream.end()?;
        stream.close()?;
        Ok(end)
    }
}

#[cfg(test)]
mod tests;
