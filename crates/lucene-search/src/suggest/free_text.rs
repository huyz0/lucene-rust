//! `org.apache.lucene.search.suggest.analyzing.FreeTextSuggester` (M13
//! T13.3c): a statistical n-gram language model for free text. Every 1..=`grams`
//! token n-gram of the suggestions goes into one FST whose output is the
//! n-gram's count (stored as `Long.MAX_VALUE - count`); a lookup completes the
//! last tokens of the query with the highest order model that has seen them,
//! backing off to lower orders with a constant `ALPHA` penalty.
//!
//! Java counts the n-grams with a temporary `IndexWriter` over a `body` field;
//! here the analyzed tokens are counted in memory, which gives the same terms
//! (in unsigned byte order) with the same total term frequencies.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::sync::Arc;

use lucene_analysis::shingle::ShingleFilter;
use lucene_analysis::{
    AnalysisError, Analyzer, AnalyzerWrapper, TokenStream, TokenStreamComponents,
};
use lucene_codecs::fst::{InputType, PositiveIntOutputs};
use lucene_codecs::fst_compiler::{FstCompilerBuilder, FstReadError};
use lucene_codecs::fst_util::{get, to_ints_ref, Arc as FstArc, TopNSearcher, TypedFst};
use lucene_store::codec_util::{check_header, write_header};
use lucene_store::{DataInput, DataOutput, Error, Result, SliceInput, VecDataOutput};

use super::input::InputIterator;
use super::lookup::{
    charsequence_compare, illegal_state, unsupported_operation, Lookup, LookupResult,
};

type NgramFst = TypedFst<PositiveIntOutputs>;

/// `CODEC_NAME`.
pub const CODEC_NAME: &str = "freetextsuggest";
/// `VERSION_START` = `VERSION_CURRENT`.
pub const VERSION_START: i32 = 0;
/// `DEFAULT_GRAMS`: a bigram model.
pub const DEFAULT_GRAMS: i32 = 2;
/// `ALPHA`: the backoff smoothing constant.
pub const ALPHA: f64 = 0.4;
/// `DEFAULT_SEPARATOR`.
pub const DEFAULT_SEPARATOR: u8 = 0x1e;
/// `IndexWriter.MAX_TERM_LENGTH`.
const MAX_TERM_LENGTH: usize = 32766;

fn fst_err(e: FstReadError) -> Error {
    Error::Corrupted(e.0)
}

fn analysis_err(e: AnalysisError) -> Error {
    Error::IllegalArgument(e.to_string())
}

/// `BytesRef.toString()`: `[61 62]`.
fn bytes_ref_string(bytes: &[u8]) -> String {
    let hex: Vec<String> = bytes.iter().map(|b| format!("{b:x}")).collect();
    format!("[{}]", hex.join(" "))
}

/// The anonymous `AnalyzerWrapper` of `addShingles`: the wrapped analyzer's
/// chain with a `ShingleFilter(2, grams)` joined by the separator.
struct Shingles {
    other: Arc<Analyzer>,
    grams: i32,
    separator: String,
}

impl AnalyzerWrapper for Shingles {
    fn wrapped_analyzer(&self, _field_name: &str) -> &Analyzer {
        &self.other
    }

    fn wrap_components(
        &self,
        _field_name: &str,
        components: TokenStreamComponents,
    ) -> TokenStreamComponents {
        let (source, sink) = components.into_parts();
        // `new_for_grams` has refused `grams < 2`
        let mut shingles =
            ShingleFilter::new(sink, 2, self.grams).expect("grams checked by the constructor");
        shingles.set_token_separator(Some(&self.separator));
        TokenStreamComponents::from_parts(source, Box::new(shingles))
    }
}

/// `FreeTextSuggester`.
pub struct FreeTextSuggester {
    fst: Option<NgramFst>,
    index_analyzer: Arc<Analyzer>,
    query_analyzer: Arc<Analyzer>,
    tot_tokens: i64,
    grams: i32,
    separator: u8,
    count: u64,
}

impl FreeTextSuggester {
    /// `new FreeTextSuggester(analyzer)`: one analyzer, a bigram model.
    pub fn new(analyzer: Arc<Analyzer>) -> Self {
        Self::with_analyzers(analyzer.clone(), analyzer)
    }

    /// `new FreeTextSuggester(indexAnalyzer, queryAnalyzer)`.
    pub fn with_analyzers(index_analyzer: Arc<Analyzer>, query_analyzer: Arc<Analyzer>) -> Self {
        Self::with_options(
            index_analyzer,
            query_analyzer,
            DEFAULT_GRAMS,
            DEFAULT_SEPARATOR,
        )
        .expect("the defaults are valid")
    }

    /// `new FreeTextSuggester(indexAnalyzer, queryAnalyzer, grams, separator)`.
    pub fn with_options(
        index_analyzer: Arc<Analyzer>,
        query_analyzer: Arc<Analyzer>,
        grams: i32,
        separator: u8,
    ) -> Result<Self> {
        if grams < 1 {
            return Err(Error::IllegalArgument("grams must be >= 1".into()));
        }
        if separator & 0x80 != 0 {
            return Err(Error::IllegalArgument(
                "separator must be simple ascii character".into(),
            ));
        }
        Ok(Self {
            fst: None,
            index_analyzer: Self::add_shingles(index_analyzer, grams, separator),
            query_analyzer: Self::add_shingles(query_analyzer, grams, separator),
            tot_tokens: 0,
            grams,
            separator,
            count: 0,
        })
    }

    /// `addShingles(other)`.
    fn add_shingles(other: Arc<Analyzer>, grams: i32, separator: u8) -> Arc<Analyzer> {
        if grams == 1 {
            return other;
        }
        let strategy = other.reuse_strategy();
        Arc::new(Analyzer::with_reuse_strategy(
            Shingles {
                other,
                grams,
                separator: char::from(separator).to_string(),
            },
            strategy,
        ))
    }

    /// `get(key)`: Java throws `UnsupportedOperationException`.
    pub fn get(&self, _key: &str) -> Result<Option<i64>> {
        Err(unsupported_operation("null"))
    }

    /// `countGrams(token)`.
    fn count_grams(&self, token: &[u8]) -> i32 {
        let separators = token.iter().filter(|&&b| b == self.separator).count();
        i32::try_from(separators)
            .unwrap_or(i32::MAX)
            .saturating_add(1)
    }

    /// `encodeWeight`.
    fn encode_weight(ngram_count: i64) -> i64 {
        i64::MAX - ngram_count
    }

    /// `decodeWeight`: Java narrows the difference to an `int`.
    fn decode_weight(output: i64) -> i64 {
        i64::from((i64::MAX.wrapping_sub(output)) as i32)
    }

    /// `lookupPrefix`: the summed output along `prefix`, with the arc it ends
    /// on, or `None` when the model never saw the prefix.
    fn lookup_prefix(fst: &NgramFst, prefix: &[u8]) -> Result<Option<(i64, FstArc<i64>)>> {
        let mut arc = fst.first_arc();
        let mut reader = fst.bytes_reader();
        let mut output = 0i64;
        for &b in prefix {
            match fst
                .find_target_arc(i32::from(b), &arc, &mut reader)
                .map_err(fst_err)?
            {
                None => return Ok(None),
                Some(next) => {
                    output = output.wrapping_add(*next.output());
                    arc = next;
                }
            }
        }
        Ok(Some((output, arc)))
    }

    /// The `body` terms of `surface` with their counts added to `terms`.
    fn count_terms(&self, surface: &[u8], terms: &mut BTreeMap<Vec<u8>, i64>) -> Result<()> {
        let text = String::from_utf8_lossy(surface);
        let mut ts = self
            .index_analyzer
            .token_stream("", &text)
            .map_err(analysis_err)?;
        let mut run = || -> std::result::Result<(), AnalysisError> {
            ts.reset()?;
            while ts.increment_token()? {
                let term = ts.attributes().term_bytes();
                if term.len() > MAX_TERM_LENGTH {
                    return Err(AnalysisError::IllegalArgument(format!(
                        "Document contains at least one immense term in field=\"body\" (whose UTF8 encoding is longer than the max length {MAX_TERM_LENGTH})"
                    )));
                }
                *terms.entry(term.to_vec()).or_insert(0) += 1;
            }
            ts.end()?;
            ts.close()
        };
        run().map_err(analysis_err)
    }
}

impl Lookup for FreeTextSuggester {
    fn get_count(&self) -> Result<u64> {
        Ok(self.count)
    }

    fn build(&mut self, input: &mut dyn InputIterator) -> Result<()> {
        if input.has_payloads() {
            return Err(Error::IllegalArgument(
                "this suggester doesn't support payloads".into(),
            ));
        }
        if input.has_contexts() {
            return Err(Error::IllegalArgument(
                "this suggester doesn't support contexts".into(),
            ));
        }
        self.tot_tokens = 0;
        let mut terms: BTreeMap<Vec<u8>, i64> = BTreeMap::new();
        let mut new_count = 0u64;
        while let Some(surface) = input.next() {
            self.count_terms(&surface, &mut terms)?;
            new_count += 1;
        }
        if terms.is_empty() {
            return Err(Error::IllegalArgument(
                "need at least one suggestion".into(),
            ));
        }
        // Move all ngrams into an FST
        let mut compiler = FstCompilerBuilder::new(InputType::Byte1).build::<PositiveIntOutputs>();
        let mut tot_tokens = 0i64;
        for (term, freq) in &terms {
            let ngram_count = self.count_grams(term);
            if ngram_count > self.grams {
                return Err(Error::IllegalArgument(format!(
                    "tokens must not contain separator byte; got token={} but gramCount={}, which is greater than expected max ngram size={}",
                    bytes_ref_string(term),
                    ngram_count,
                    self.grams
                )));
            }
            if ngram_count == 1 {
                tot_tokens += freq;
            }
            compiler
                .add(&to_ints_ref(term), Self::encode_weight(*freq))
                .map_err(|e| Error::IllegalArgument(e.to_string()))?;
        }
        self.tot_tokens = tot_tokens;
        self.fst = compiler.compile().map(NgramFst::from_compiled);
        self.count = new_count;
        Ok(())
    }

    fn store(&self, output: &mut dyn DataOutput) -> Result<bool> {
        let Some(fst) = &self.fst else {
            return Err(illegal_state("Lookup not supported at this time"));
        };
        let mut header = VecDataOutput::new();
        write_header(&mut header, CODEC_NAME, VERSION_START);
        output.write_bytes(&header.into_inner());
        output.write_vlong(self.count as i64);
        output.write_byte(self.separator);
        output.write_vint(self.grams);
        output.write_vlong(self.tot_tokens);
        output.write_bytes(&fst.save());
        Ok(true)
    }

    fn load(&mut self, input: &mut dyn DataInput) -> Result<bool> {
        let mut rest = vec![0u8; input.remaining()];
        input.read_bytes(&mut rest)?;
        let mut fields = SliceInput::new(&rest);
        check_header(&mut fields, CODEC_NAME, VERSION_START, VERSION_START)?;
        let count = fields.read_vlong()?;
        let separator_orig = fields.read_byte()?;
        if separator_orig != self.separator {
            return Err(illegal_state(format!(
                "separator={} is incorrect: original model was built with separator={}",
                self.separator as i8, separator_orig as i8
            )));
        }
        let grams_orig = fields.read_vint()?;
        if grams_orig != self.grams {
            return Err(illegal_state(format!(
                "grams={} is incorrect: original model was built with grams={}",
                self.grams, grams_orig
            )));
        }
        let tot_tokens = fields.read_vlong()?;
        let fst = NgramFst::read(&rest[fields.position()..]).map_err(fst_err)?;
        self.count = count as u64;
        self.tot_tokens = tot_tokens;
        self.fst = Some(fst);
        Ok(true)
    }

    fn lookup(
        &self,
        key: &str,
        contexts: Option<&BTreeSet<Vec<u8>>>,
        _only_more_popular: bool,
        num: usize,
    ) -> Result<Vec<LookupResult>> {
        if contexts.is_some() {
            return Err(Error::IllegalArgument(
                "this suggester doesn't support contexts".into(),
            ));
        }
        let Some(fst) = &self.fst else {
            return Err(illegal_state("Lookup not supported at this time"));
        };
        let grams = self.grams as usize;
        let separator = self.separator;

        let mut last_tokens: Vec<Option<Vec<u8>>> = vec![None; grams];
        let mut max_end_offset = -1;
        let mut saw_real_token = false;
        let (end_pos_inc, end_offset) = {
            let mut ts = self
                .query_analyzer
                .token_stream("", key)
                .map_err(analysis_err)?;
            let mut run = || -> Result<(i32, i32)> {
                ts.reset().map_err(analysis_err)?;
                // Run full analysis, but save only the last 1gram, last 2gram, etc.
                while ts.increment_token().map_err(analysis_err)? {
                    let atts = ts.attributes();
                    let token_bytes = atts.term_bytes();
                    saw_real_token |= !token_bytes.is_empty();
                    let gram_count = atts.position_length();
                    // Safety: make sure the recalculated count "agrees"
                    if self.count_grams(token_bytes) != gram_count {
                        return Err(Error::IllegalArgument(format!(
                            "tokens must not contain separator byte; got token={} but gramCount={} does not match recalculated count={}",
                            bytes_ref_string(token_bytes),
                            gram_count,
                            self.count_grams(token_bytes)
                        )));
                    }
                    max_end_offset = max_end_offset.max(atts.end_offset());
                    let slot = usize::try_from(gram_count.saturating_sub(1))
                        .ok()
                        .filter(|&i| i < grams)
                        .ok_or_else(|| {
                            Error::IllegalArgument(format!("gramCount={gram_count} out of range"))
                        })?;
                    last_tokens[slot] = Some(token_bytes.to_vec());
                }
                ts.end().map_err(analysis_err)?;
                let atts = ts.attributes();
                let end = (atts.position_increment(), atts.end_offset());
                ts.close().map_err(analysis_err)?;
                Ok(end)
            };
            run()?
        };

        if !saw_real_token {
            return Err(Error::IllegalArgument(
                "no tokens produced by analyzer, or the only tokens were empty strings".into(),
            ));
        }

        // Carefully fill last tokens with _ tokens; ShingleFilter apparently
        // won't emit "only hole" tokens. This is also true if the input is
        // the empty string (no tokens, so `max_end_offset` is still -1).
        let last_token_ended = end_offset > max_end_offset || end_pos_inc > 0;
        if last_token_ended {
            // If the user hit space after the last token, "upgrade" all
            // tokens: "foo " suggests all bigrams starting with foo, and not
            // any unigrams starting with "foo".
            for i in (1..grams).rev() {
                let Some(mut token) = last_tokens[i - 1].clone() else {
                    continue;
                };
                token.push(separator);
                last_tokens[i] = Some(token);
            }
            last_tokens[0] = Some(Vec::new());
        }

        if num == 0 {
            return Ok(Vec::new()); // Java's `TopNSearcher` throws for a depth of 0
        }

        // Try highest order models first, and if they return results, return
        // that; else, fall back.
        let mut backoff = 1.0f64;
        let mut results: Vec<LookupResult> = Vec::with_capacity(num);
        // A given suffix is added once, from the highest order model that saw
        // it; lower order models skip it.
        let mut seen: HashSet<Vec<u8>> = HashSet::new();

        for gram in (0..grams).rev() {
            let Some(token) = last_tokens[gram].as_mut() else {
                continue;
            };
            // Don't make unigram predictions from the empty string
            if token.is_empty() && !key.is_empty() {
                continue;
            }
            if end_pos_inc > 0 && i64::try_from(gram).unwrap_or(i64::MAX) <= i64::from(end_pos_inc)
            {
                // Skip hole-only predictions
                break;
            }

            // match the prefix portion exactly
            let Some((prefix_output, arc)) = Self::lookup_prefix(fst, token)? else {
                // This model never saw this prefix, e.g. the trigram model
                // never saw context "purple mushroom"
                backoff *= ALPHA;
                continue;
            };

            // Denominator for computing scores from the current model's
            // predictions
            let mut context_count = self.tot_tokens;
            let mut last_token_fragment: Option<Vec<u8>> = None;
            if let Some(i) = token.iter().rposition(|&b| b == separator) {
                let context = &token[..i];
                let output = get(fst, &to_ints_ref(context))
                    .map_err(fst_err)?
                    .ok_or_else(|| {
                        // Java's `decodeWeight(null)`: a filler `_` can leave a
                        // seen n-gram whose context was never indexed
                        Error::Io(std::io::Error::other(
                            "NullPointerException: Cannot invoke \"java.lang.Long.longValue()\" because \"output\" is null",
                        ))
                    })?;
                context_count = Self::decode_weight(output);
                last_token_fragment = Some(token[i + 1..].to_vec());
            }
            let final_last_token = last_token_fragment.unwrap_or_else(|| token.clone());

            // Because several models live in one FST (1gram, 2gram, 3gram),
            // the search is restricted to the current model: the separator
            // arcs are not followed, and completions whose last token a
            // higher order model already predicted are rejected. Up to
            // `seen.len()` paths may be rejected, so the queue is that much
            // deeper.
            let completions = {
                let mut searcher = TopNSearcher::new(
                    fst,
                    num,
                    num.saturating_add(seen.len()),
                    |a: &i64, b: &i64| a.cmp(b),
                );
                searcher.skip_label = Some(i32::from(separator));
                let seen_ref = &seen;
                let last = &final_last_token;
                searcher.accept_result = Some(Box::new(move |path| {
                    let mut candidate = last.clone();
                    candidate.extend(path.input.iter().map(|&l| l as u8));
                    !seen_ref.contains(&candidate)
                }));
                // this search is initialized with a single start node, so an
                // empty input path is right
                searcher
                    .add_start_paths(&arc, prefix_output, true, Vec::new())
                    .map_err(fst_err)?;
                searcher.search().map_err(fst_err)?
            };

            let prefix_length = token.len();
            for completion in completions.top_n {
                token.truncate(prefix_length);
                token.extend(completion.input.iter().map(|&l| l as u8));

                // Skip this path if a higher order model already
                // saw/predicted its last token
                let last_token = match token.iter().rposition(|&b| b == separator) {
                    Some(i) => token[i + 1..].to_vec(),
                    None => token.clone(),
                };
                if seen.contains(&last_token) {
                    continue;
                }
                seen.insert(last_token);
                let value =
                    (i64::MAX as f64 * backoff * (Self::decode_weight(completion.output) as f64)
                        / (context_count as f64)) as i64;
                results.push(LookupResult::new(
                    String::from_utf8_lossy(token).into_owned(),
                    value,
                ));
            }
            backoff *= ALPHA;
        }

        results.sort_by(|a, b| {
            b.value
                .cmp(&a.value)
                // Tie break by UTF16 sort order
                .then_with(|| charsequence_compare(&a.key, &b.key))
        });
        results.truncate(num);
        Ok(results)
    }
}
