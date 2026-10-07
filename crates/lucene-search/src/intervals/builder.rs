//! `IntervalBuilder` and `Intervals.analyzedText`: an [`IntervalsSource`]
//! from analyzed text -- a term, a phrase, an ordered or unordered
//! conjunction with a gap limit, synonyms as disjunctions, and a token
//! graph cut at its articulation points with each side path a phrase.
//!
//! The token stream is read once into its tokens' attributes and replayed,
//! as Java replays it through a `CachingTokenFilter`
//! ([`crate::query_builder`] does the same).

use lucene_analysis::{
    AnalysisError, Analyzer, AttributeSource, GraphTokenStreamFiniteStrings, TokenStream,
};

use super::{Intervals, IntervalsSource};
use crate::extended_query::MAX_CLAUSE_COUNT;
use crate::{Error, Result};

fn analysis(e: AnalysisError) -> Error {
    Error::IllegalArgument(format!("Error analyzing interval text: {e}"))
}

/// `IntervalBuilder.NO_INTERVALS`.
fn no_intervals() -> IntervalsSource {
    Intervals::no_intervals("No terms in analyzed text")
}

/// `Intervals.analyzedText(text, analyzer, field, maxGaps, ordered)`.
///
/// # Errors
/// The analyzer's, and an expansion past the clause limit.
pub fn analyzed_text(
    text: &str,
    analyzer: &Analyzer,
    field: &str,
    max_gaps: i32,
    ordered: bool,
) -> Result<IntervalsSource> {
    let mut ts = analyzer.token_stream(field, text).map_err(analysis)?;
    analyzed_tokens(&mut ts, max_gaps, ordered)
}

/// `Intervals.analyzedText(tokenStream, maxGaps, ordered)`
/// (`IntervalBuilder.analyzeText`).
///
/// # Errors
/// The stream's, and an expansion past the clause limit.
pub fn analyzed_tokens(
    stream: &mut dyn TokenStream,
    max_gaps: i32,
    ordered: bool,
) -> Result<IntervalsSource> {
    let tokens = read_tokens(stream).map_err(analysis)?;
    // Phase 1: count the tokens, and whether any is a synonym or spans
    // several positions.
    let num_tokens = tokens.len();
    let has_synonyms = tokens.iter().any(|t| t.position_increment() == 0);
    let is_graph = tokens.iter().any(|t| t.position_length() > 1);
    // Phase 2: a term, a graph, a phrase with synonyms, or a plain phrase.
    if num_tokens == 0 {
        Ok(no_intervals())
    } else if num_tokens == 1 {
        Ok(Intervals::term(tokens[0].term_bytes().to_vec()))
    } else if is_graph {
        combine_sources(analyze_graph(&tokens)?, max_gaps, ordered)
    } else if has_synonyms {
        analyze_synonyms(&tokens, max_gaps, ordered)
    } else {
        combine_sources(analyze_terms(&tokens), max_gaps, ordered)
    }
}

/// `combineSources`.
fn combine_sources(
    sources: Vec<IntervalsSource>,
    max_gaps: i32,
    ordered: bool,
) -> Result<IntervalsSource> {
    if sources.is_empty() {
        return Ok(no_intervals());
    }
    if sources.len() == 1 {
        return Ok(sources.into_iter().next().unwrap_or_else(no_intervals));
    }
    if max_gaps == 0 && ordered {
        return Intervals::phrase(sources);
    }
    let inner = if ordered {
        Intervals::ordered(sources)
    } else {
        Intervals::unordered(sources)
    };
    if max_gaps == -1 {
        return Ok(inner);
    }
    Intervals::maxgaps(max_gaps, inner)
}

/// `analyzeTerms`: each token a term, extended back over the positions
/// before it.
fn analyze_terms(tokens: &[AttributeSource]) -> Vec<IntervalsSource> {
    tokens
        .iter()
        .map(|t| {
            let preceding_spaces = t.position_increment().wrapping_sub(1);
            extend(Intervals::term(t.term_bytes().to_vec()), preceding_spaces)
        })
        .collect()
}

/// `extend(source, precedingSpaces)`.
fn extend(source: IntervalsSource, preceding_spaces: i32) -> IntervalsSource {
    if preceding_spaces == 0 {
        return source;
    }
    Intervals::extend(source, preceding_spaces, 0)
}

/// `analyzeSynonyms`: the terms stacked at a position become a disjunction.
fn analyze_synonyms(
    tokens: &[AttributeSource],
    max_gaps: i32,
    ordered: bool,
) -> Result<IntervalsSource> {
    let mut terms = Vec::new();
    let mut synonyms: Vec<IntervalsSource> = Vec::new();
    let mut spaces = 0i32;
    for t in tokens {
        let pos_inc = t.position_increment();
        if pos_inc > 0 {
            if synonyms.len() == 1 {
                terms.push(extend(synonyms.remove(0), spaces));
            } else if synonyms.len() > 1 {
                terms.push(extend(
                    Intervals::or(std::mem::take(&mut synonyms))?,
                    spaces,
                ));
            }
            synonyms.clear();
            spaces = pos_inc.wrapping_sub(1);
        }
        synonyms.push(Intervals::term(t.term_bytes().to_vec()));
    }
    if synonyms.len() == 1 {
        terms.push(extend(synonyms.remove(0), spaces));
    } else {
        terms.push(extend(Intervals::or(synonyms)?, spaces));
    }
    combine_sources(terms, max_gaps, ordered)
}

/// `analyzeGraph`: the graph cut at its articulation points; a segment
/// with side paths is a disjunction of its paths' phrases, one without is
/// its terms.
fn analyze_graph(tokens: &[AttributeSource]) -> Result<Vec<IntervalsSource>> {
    let graph = GraphTokenStreamFiniteStrings::new(&mut Replay::new(tokens)).map_err(analysis)?;
    let mut clauses = Vec::new();
    let points = graph.articulation_points().map_err(analysis)?;
    let mut last_state = 0;
    for i in 0..=points.len() {
        let start = last_state;
        let end = points.get(i).copied().unwrap_or(-1);
        last_state = end;
        if graph.has_side_path(start) {
            let mut paths = Vec::new();
            for path in graph.finite_strings_between(start, end).map_err(analysis)? {
                let phrase = combine_sources(analyze_terms(path.tokens()), 0, true)?;
                if paths.len() >= MAX_CLAUSE_COUNT {
                    return Err(Error::InvalidQuery(format!(
                        "maxClauseCount is set to {MAX_CLAUSE_COUNT}"
                    )));
                }
                paths.push(phrase);
            }
            if !paths.is_empty() {
                clauses.push(Intervals::or(paths)?);
            }
        } else if let Some(path) = graph
            .finite_strings_between(start, end)
            .map_err(analysis)?
            .into_iter()
            .next()
        {
            clauses.extend(analyze_terms(path.tokens()));
        }
    }
    Ok(clauses)
}

/// Every token's attributes, `source` reset, consumed and ended.
fn read_tokens(
    source: &mut dyn TokenStream,
) -> std::result::Result<Vec<AttributeSource>, AnalysisError> {
    source.reset()?;
    let mut tokens = Vec::new();
    while source.increment_token()? {
        tokens.push(source.attributes().clone());
    }
    source.end()?;
    Ok(tokens)
}

/// The cached tokens replayed as a stream.
struct Replay<'t> {
    atts: AttributeSource,
    tokens: &'t [AttributeSource],
    upto: usize,
}

impl<'t> Replay<'t> {
    fn new(tokens: &'t [AttributeSource]) -> Self {
        Self {
            atts: AttributeSource::new(),
            tokens,
            upto: 0,
        }
    }
}

impl TokenStream for Replay<'_> {
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
    fn increment_token(&mut self) -> std::result::Result<bool, AnalysisError> {
        let Some(t) = self.tokens.get(self.upto) else {
            return Ok(false);
        };
        self.atts = t.clone();
        self.upto += 1;
        Ok(true)
    }
    fn reset(&mut self) -> std::result::Result<(), AnalysisError> {
        self.upto = 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(spec: &[(&str, i32, i32)]) -> Vec<AttributeSource> {
        spec.iter()
            .map(|&(t, inc, len)| {
                let mut a = AttributeSource::new();
                a.set_term(t);
                a.set_position_increment(inc).unwrap();
                a.set_position_length(len).unwrap();
                a
            })
            .collect()
    }

    fn built(spec: &[(&str, i32, i32)], max_gaps: i32, ordered: bool) -> String {
        let toks = tokens(spec);
        let mut replay = Replay::new(&toks);
        assert!(replay.conditional_root().is_none());
        analyzed_tokens(&mut replay, max_gaps, ordered)
            .unwrap()
            .to_string()
    }

    /// Synonyms, holes and token graphs, each against what Lucene 10.5.0's
    /// `Intervals.analyzedText(tokenStream, maxGaps, ordered)` printed for the
    /// same canned tokens (a throwaway `CannedTokenStream` program; no
    /// analyzer this port has makes stacked or multi-position tokens).
    #[test]
    fn synonyms_holes_and_graphs_build_as_lucene() {
        let car = [("fast", 1, 1), ("quick", 0, 1), ("car", 1, 1)];
        assert_eq!(built(&car, 0, true), "BLOCK(or(fast,quick),car)");
        let hole = [("a", 1, 1), ("b", 2, 1), ("c", 0, 1)];
        assert_eq!(built(&hole, -1, false), "UNORDERED(a,EXTEND(or(b,c),1,0))");
        let ny = [("ny", 1, 2), ("new", 0, 1), ("york", 1, 1), ("city", 1, 1)];
        assert_eq!(
            built(&ny, 0, true),
            "or(BLOCK(new,york,city),BLOCK(ny,city))"
        );
        assert_eq!(
            built(&ny, 2, false),
            "or(MAXGAPS/2(UNORDERED(BLOCK(new,york),city)),MAXGAPS/2(UNORDERED(ny,city)))"
        );
        let after_hole = [("x", 1, 1), ("ny", 2, 2), ("new", 0, 1), ("york", 1, 1)];
        assert_eq!(
            built(&after_hole, -1, true),
            "ORDERED(x,or(BLOCK(EXTEND(new,1,0),york),EXTEND(ny,1,0)))"
        );
        let tail = [("a", 1, 1), ("b", 1, 1), ("c", 0, 1)];
        assert_eq!(built(&tail, 3, true), "MAXGAPS/3(ORDERED(a,or(b,c)))");
        assert_eq!(built(&[], 0, true), "NOMATCH(No terms in analyzed text)");
        assert_eq!(built(&[("solo", 1, 1)], 0, true), "solo");
    }
}
