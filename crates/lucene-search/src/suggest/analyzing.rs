//! `org.apache.lucene.search.suggest.analyzing.AnalyzingSuggester` and
//! `FuzzySuggester` (M13 T13.3c): a weighted FST over the *analyzed* form of
//! each suggestion, so "Ghost  Christmas" finds "the ghost of christmas".
//!
//! Each surface form is analyzed into an automaton (a graph analyzer may give
//! several paths), every finite string of it becomes an FST input
//! `analyzed ++ 0 ++ dedup`, and the output pairs the cost
//! (`Integer.MAX_VALUE - weight`) with the surface form (and, when there are
//! payloads, `0x1F ++ payload`). A lookup analyzes the query into an automaton,
//! intersects it with the FST, and runs a top-N search from the nodes it
//! reaches.
//!
//! Differs from Java: the build sorts in memory (`OfflineSorter` spills to
//! `tempDir`) with a *stable* sort on `(analyzed, cost, surface)`, so entries
//! equal on those keys keep their input order where Java's order is
//! undefined; `lookup` with `num == 0` returns nothing where Java asserts;
//! `load` reads the rest of its input for the FST (see
//! [`super::fst_completion`]).

use std::collections::{BTreeSet, HashSet};
use std::sync::Arc;

use lucene_analysis::{
    AnalysisError, Analyzer, CharArraySet, State, TokenFilter, TokenStream, TokenStreamToAutomaton,
    HOLE, POS_SEP,
};
use lucene_codecs::fst::{ByteSequenceOutputs, InputType, Pair, PairOutputs, PositiveIntOutputs};
use lucene_codecs::fst_compiler::{FstCompilerBuilder, FstOutputs, FstReadError};
use lucene_codecs::fst_util::{to_ints_ref, TopNSearcher, TypedFst};
use lucene_store::{DataInput, DataOutput, Error, Result, SliceInput};
use lucene_util::automaton::automata;
use lucene_util::automaton::operations::{determinize, topo_sort_states, union};
use lucene_util::automaton::{
    Automaton, Builder, FiniteStringsIterator, LevenshteinAutomata, LimitedFiniteStringsIterator,
    Transition, TransitionAccessor, Utf32ToUtf8, DEFAULT_DETERMINIZE_WORK_LIMIT,
};

use super::fst_util::{intersect_prefix_paths, Path};
use super::input::InputIterator;
use super::lookup::{unsupported_operation, Lookup, LookupResult};

/// `AnalyzingSuggester.EXACT_FIRST`: a surface form equal to the query comes
/// first, whatever its weight.
pub const EXACT_FIRST: i32 = 1;
/// `AnalyzingSuggester.PRESERVE_SEP`: token separators are kept, so "a b" does
/// not match "ab".
pub const PRESERVE_SEP: i32 = 2;

/// The separator `replaceSep` keeps (`TokenStreamToAutomaton.POS_SEP`).
const SEP_LABEL: i32 = POS_SEP;
/// The label that ends an analyzed form in the FST.
const END_BYTE: i32 = 0;
const PAYLOAD_SEP: u8 = 0x1f;
/// `Short.MAX_VALUE - 2`.
const MAX_LENGTH: usize = 32765;

type Output = Pair<i64, Vec<u8>>;
type SuggestOutputs = PairOutputs<PositiveIntOutputs, ByteSequenceOutputs>;
type SuggestFst = TypedFst<SuggestOutputs>;

fn illegal(e: impl std::fmt::Display) -> Error {
    Error::IllegalArgument(e.to_string())
}

fn fst_err(e: FstReadError) -> Error {
    Error::Corrupted(e.0)
}

/// `FuzzySuggester`'s settings.
#[derive(Debug, Clone, Copy)]
struct FuzzyOptions {
    max_edits: i32,
    transpositions: bool,
    non_fuzzy_prefix: i32,
    min_fuzzy_length: i32,
    unicode_aware: bool,
}

/// `AnalyzingSuggester`.
pub struct AnalyzingSuggester {
    fst: Option<SuggestFst>,
    index_analyzer: Arc<Analyzer>,
    query_analyzer: Arc<Analyzer>,
    exact_first: bool,
    preserve_sep: bool,
    max_surface_forms_per_analyzed_form: i32,
    max_graph_expansions: i32,
    /// Highest number of analyzed paths seen for one surface form; kept
    /// across builds, as in Java.
    max_analyzed_paths_for_one_input: i32,
    has_payloads: bool,
    preserve_position_increments: bool,
    count: u64,
    fuzzy: Option<FuzzyOptions>,
}

/// One analyzed form of a surface form, ready to sort.
struct Entry {
    analyzed: Vec<u8>,
    cost: i32,
    surface: Vec<u8>,
    payload: Vec<u8>,
}

impl AnalyzingSuggester {
    /// `new AnalyzingSuggester(tempDir, prefix, analyzer)`: `EXACT_FIRST |
    /// PRESERVE_SEP`, 256 surface forms, no graph limit, position increments
    /// preserved.
    pub fn new(analyzer: Arc<Analyzer>) -> Self {
        Self::with_analyzers(analyzer.clone(), analyzer)
    }

    /// `new AnalyzingSuggester(tempDir, prefix, indexAnalyzer, queryAnalyzer)`.
    pub fn with_analyzers(index_analyzer: Arc<Analyzer>, query_analyzer: Arc<Analyzer>) -> Self {
        Self::with_options(
            index_analyzer,
            query_analyzer,
            EXACT_FIRST | PRESERVE_SEP,
            256,
            -1,
            true,
        )
        .expect("the default options are valid")
    }

    /// The full constructor.
    ///
    /// # Errors
    /// `IllegalArgument` for the options Java rejects.
    pub fn with_options(
        index_analyzer: Arc<Analyzer>,
        query_analyzer: Arc<Analyzer>,
        options: i32,
        max_surface_forms_per_analyzed_form: i32,
        max_graph_expansions: i32,
        preserve_position_increments: bool,
    ) -> Result<Self> {
        if options & !(EXACT_FIRST | PRESERVE_SEP) != 0 {
            return Err(illegal(format!(
                "options should only contain EXACT_FIRST and PRESERVE_SEP; got {options}"
            )));
        }
        if max_surface_forms_per_analyzed_form <= 0 || max_surface_forms_per_analyzed_form > 256 {
            return Err(illegal(format!(
                "maxSurfaceFormsPerAnalyzedForm must be > 0 and < 256 (got: {max_surface_forms_per_analyzed_form})"
            )));
        }
        if max_graph_expansions < 1 && max_graph_expansions != -1 {
            return Err(illegal(format!(
                "maxGraphExpansions must -1 (no limit) or > 0 (got: {max_graph_expansions})"
            )));
        }
        Ok(Self {
            fst: None,
            index_analyzer,
            query_analyzer,
            exact_first: options & EXACT_FIRST != 0,
            preserve_sep: options & PRESERVE_SEP != 0,
            max_surface_forms_per_analyzed_form,
            max_graph_expansions,
            max_analyzed_paths_for_one_input: 0,
            has_payloads: false,
            preserve_position_increments,
            count: 0,
            fuzzy: None,
        })
    }

    /// `get(key)`: Java throws `UnsupportedOperationException`.
    pub fn get(&self, _key: &str) -> Result<Option<i64>> {
        Err(unsupported_operation("null"))
    }

    /// `getTokenStreamToAutomaton()`.
    fn token_stream_to_automaton(&self) -> TokenStreamToAutomaton {
        let mut tsta = TokenStreamToAutomaton::new();
        tsta.set_preserve_position_increments(self.preserve_position_increments);
        tsta.set_final_offset_gap_as_hole(true);
        if let Some(f) = &self.fuzzy {
            tsta.set_unicode_arcs(f.unicode_aware);
        }
        tsta
    }

    /// `replaceSep`: a position separator becomes [`SEP_LABEL`] when kept and
    /// an epsilon otherwise; a hole is dropped.
    fn replace_sep(&self, a: &Automaton) -> Result<Automaton> {
        let mut result = Builder::new();
        result.copy_states(a);
        let order = topo_sort_states(a).map_err(illegal)?;
        let mut t = Transition::new();
        for &state in order.iter().rev() {
            let count = a.init_transition(state, &mut t);
            for _ in 0..count {
                a.get_next_transition(&mut t);
                if t.min == POS_SEP {
                    if self.preserve_sep {
                        result.add_transition_label(state, t.dest, SEP_LABEL);
                    } else {
                        result.add_epsilon(state, t.dest);
                    }
                } else if t.min == HOLE {
                    result.add_epsilon(state, t.dest);
                } else {
                    result.add_transition(state, t.dest, t.min, t.max);
                }
            }
        }
        Ok(result.finish())
    }

    /// `convertAutomaton`: identity, but a unicode-aware `FuzzySuggester`
    /// turns code point arcs into UTF-8 byte arcs.
    fn convert_automaton(&self, a: Automaton) -> Result<Automaton> {
        match &self.fuzzy {
            Some(f) if f.unicode_aware => {
                let utf8 = Utf32ToUtf8::new().convert(&a);
                determinize(&utf8, DEFAULT_DETERMINIZE_WORK_LIMIT).map_err(illegal)
            }
            _ => Ok(a),
        }
    }

    /// `toAutomaton(surfaceForm, ts2a)`.
    fn to_automaton(&self, surface: &[u8], ts2a: &TokenStreamToAutomaton) -> Result<Automaton> {
        let text = String::from_utf8_lossy(surface);
        let automaton = {
            let mut ts = self
                .index_analyzer
                .token_stream("", &text)
                .map_err(illegal)?;
            ts2a.to_automaton(&mut ts).map_err(illegal)?
        };
        let automaton = self.replace_sep(&automaton)?;
        self.convert_automaton(automaton)
    }

    /// `toLookupAutomaton(key)`.
    fn to_lookup_automaton(&self, key: &str) -> Result<Automaton> {
        let automaton = {
            let mut ts = self.query_analyzer.token_stream("", key).map_err(illegal)?;
            self.token_stream_to_automaton()
                .to_automaton(&mut ts)
                .map_err(illegal)?
        };
        let automaton = self.replace_sep(&automaton)?;
        determinize(&automaton, DEFAULT_DETERMINIZE_WORK_LIMIT).map_err(illegal)
    }

    /// `FuzzySuggester.toLevenshteinAutomata`.
    fn to_levenshtein_automata(
        &self,
        f: &FuzzyOptions,
        automaton: &Automaton,
    ) -> Result<Automaton> {
        let mut subs = Vec::new();
        let mut strings = FiniteStringsIterator::new(automaton);
        while let Some(string) = strings.next_string().map_err(illegal)? {
            let len = i32::try_from(string.len()).unwrap_or(i32::MAX);
            if len <= f.non_fuzzy_prefix || len < f.min_fuzzy_length {
                subs.push(automata::make_string_ints(&string));
            } else {
                let split = usize::try_from(f.non_fuzzy_prefix).map_err(illegal)?;
                let (prefix, rest) = string.split_at(split);
                let alpha_max = if f.unicode_aware { 0x10FFFF } else { 255 };
                let lev =
                    LevenshteinAutomata::with_alphabet(rest.to_vec(), alpha_max, f.transpositions)
                        .map_err(illegal)?;
                // `UnicodeUtil.newString(ints, offset, nonFuzzyPrefix)`: each
                // int a code point (a byte, when not unicode aware)
                let prefix: String = prefix
                    .iter()
                    .map(|&c| {
                        u32::try_from(c)
                            .ok()
                            .and_then(char::from_u32)
                            .unwrap_or('\u{FFFD}')
                    })
                    .collect();
                let a = lev
                    .to_automaton_with_prefix(f.max_edits, &prefix)
                    .ok_or_else(|| illegal(format!("maxEdits {} is not supported", f.max_edits)))?;
                subs.push(a);
            }
        }
        Ok(match subs.len() {
            0 => automata::make_empty(),
            1 => subs.swap_remove(0),
            _ => {
                let refs: Vec<&Automaton> = subs.iter().collect();
                determinize(&union(&refs), DEFAULT_DETERMINIZE_WORK_LIMIT).map_err(illegal)?
            }
        })
    }

    /// `getFullPrefixPaths`: the exact prefix paths, or for a fuzzy suggester
    /// everything within the edit distance.
    fn full_prefix_paths(
        &self,
        prefix_paths: Vec<Path<Output>>,
        lookup_automaton: &Automaton,
        fst: &SuggestFst,
    ) -> Result<Vec<Path<Output>>> {
        let Some(f) = &self.fuzzy else {
            return Ok(prefix_paths);
        };
        let lev = self.convert_automaton(self.to_levenshtein_automata(f, lookup_automaton)?)?;
        intersect_prefix_paths(&lev, fst)
    }

    /// `getLookupResult(output1, output2, spare)`.
    fn lookup_result(&self, output: &Output) -> LookupResult {
        let weight = i64::from(decode_weight(output.first));
        if self.has_payloads {
            let sep = output
                .second
                .iter()
                .position(|&b| b == PAYLOAD_SEP)
                .unwrap_or(output.second.len());
            let payload = output.second.get(sep + 1..).unwrap_or_default().to_vec();
            LookupResult::with_all(
                String::from_utf8_lossy(&output.second[..sep]).into_owned(),
                None,
                weight,
                Some(payload),
                None,
            )
        } else {
            LookupResult::new(String::from_utf8_lossy(&output.second).into_owned(), weight)
        }
    }
}

/// `sameSurfaceForm(key, output2)`.
fn same_surface_form(has_payloads: bool, key: &[u8], output2: &[u8]) -> bool {
    if has_payloads {
        // output2 has at least the payload separator
        key.len() < output2.len() && output2.starts_with(key) && output2[key.len()] == PAYLOAD_SEP
    } else {
        key == output2
    }
}

/// `decodeWeight`: cost to weight.
fn decode_weight(encoded: i64) -> i32 {
    (i64::from(i32::MAX).wrapping_sub(encoded)) as i32
}

/// `encodeWeight`: weight to cost.
fn encode_weight(value: i64) -> Result<i32> {
    if !(0..=i64::from(i32::MAX)).contains(&value) {
        return Err(unsupported_operation(format!(
            "cannot encode value: {value}"
        )));
    }
    Ok(i32::MAX - value as i32)
}

impl Lookup for AnalyzingSuggester {
    fn get_count(&self) -> Result<u64> {
        Ok(self.count)
    }

    fn build(&mut self, input: &mut dyn InputIterator) -> Result<()> {
        if input.has_contexts() {
            return Err(illegal("this suggester doesn't support contexts"));
        }
        self.has_payloads = input.has_payloads();
        let ts2a = self.token_stream_to_automaton();
        let mut entries: Vec<Entry> = Vec::new();
        let mut new_count = 0u64;
        while let Some(surface) = input.next() {
            let automaton = self.to_automaton(&surface, &ts2a)?;
            let mut strings =
                LimitedFiniteStringsIterator::new(&automaton, self.max_graph_expansions)
                    .map_err(illegal)?;
            while let Some(string) = strings.next_string().map_err(illegal)? {
                let analyzed: Vec<u8> = string.iter().map(|&l| l as u8).collect();
                if analyzed.len() > MAX_LENGTH {
                    return Err(illegal(format!(
                        "cannot handle analyzed forms > {MAX_LENGTH} in length (got {})",
                        analyzed.len()
                    )));
                }
                let cost = encode_weight(input.weight())?;
                let payload = if self.has_payloads {
                    if surface.len() > MAX_LENGTH {
                        return Err(illegal(format!(
                            "cannot handle surface form > {MAX_LENGTH} in length (got {})",
                            surface.len()
                        )));
                    }
                    if surface.contains(&PAYLOAD_SEP) {
                        return Err(illegal(
                            "surface form cannot contain unit separator character U+001F; this character is reserved",
                        ));
                    }
                    input.payload().unwrap_or_default().to_vec()
                } else {
                    Vec::new()
                };
                entries.push(Entry {
                    analyzed,
                    cost,
                    surface: surface.clone(),
                    payload,
                });
                new_count += 1;
            }
            self.max_analyzed_paths_for_one_input =
                self.max_analyzed_paths_for_one_input.max(strings.size());
        }

        // Sort all input/output pairs (required by FST.Builder):
        entries.sort_by(|a, b| {
            a.analyzed
                .cmp(&b.analyzed)
                .then(a.cost.cmp(&b.cost))
                .then_with(|| a.surface.cmp(&b.surface))
        });

        let mut compiler = FstCompilerBuilder::new(InputType::Byte1).build::<SuggestOutputs>();
        // Used to remove duplicate surface forms (but we still index the
        // highest-weight one); cleared on a new analyzed form, so at most 256
        // entries.
        let mut seen_surface_forms: HashSet<&[u8]> = HashSet::new();
        let mut previous_analyzed: Option<&[u8]> = None;
        let mut dedup = 0i32;
        for e in &entries {
            match previous_analyzed {
                None => {
                    previous_analyzed = Some(&e.analyzed);
                    seen_surface_forms.insert(&e.surface);
                }
                Some(previous) if previous == e.analyzed.as_slice() => {
                    dedup += 1;
                    if dedup >= self.max_surface_forms_per_analyzed_form {
                        // More than maxSurfaceFormsPerAnalyzedForm dups: skip the rest
                        continue;
                    }
                    if !seen_surface_forms.insert(&e.surface) {
                        continue;
                    }
                }
                Some(_) => {
                    dedup = 0;
                    previous_analyzed = Some(&e.analyzed);
                    seen_surface_forms.clear();
                    seen_surface_forms.insert(&e.surface);
                }
            }
            // NOTE: must be byte 0 so we sort before whatever is next
            let mut key = e.analyzed.clone();
            key.push(0);
            key.push(dedup as u8);
            let mut second = e.surface.clone();
            if self.has_payloads {
                second.push(PAYLOAD_SEP);
                second.extend_from_slice(&e.payload);
            }
            compiler
                .add(
                    &to_ints_ref(&key),
                    Pair {
                        first: i64::from(e.cost),
                        second,
                    },
                )
                .map_err(illegal)?;
        }
        self.fst = compiler.compile().map(SuggestFst::from_compiled);
        self.count = new_count;
        Ok(())
    }

    fn store(&self, output: &mut dyn DataOutput) -> Result<bool> {
        output.write_vlong(self.count as i64);
        let Some(fst) = &self.fst else {
            return Ok(false);
        };
        output.write_bytes(&fst.save());
        output.write_vint(self.max_analyzed_paths_for_one_input);
        output.write_byte(u8::from(self.has_payloads));
        Ok(true)
    }

    fn load(&mut self, input: &mut dyn DataInput) -> Result<bool> {
        self.count = input.read_vlong()? as u64;
        let mut rest = vec![0u8; input.remaining()];
        input.read_bytes(&mut rest)?;
        let (fst, consumed) = SuggestFst::read_prefix(&rest).map_err(fst_err)?;
        let mut tail = SliceInput::new(&rest[consumed..]);
        self.max_analyzed_paths_for_one_input = tail.read_vint()?;
        self.has_payloads = tail.read_byte()? == 1;
        self.fst = Some(fst);
        Ok(true)
    }

    fn lookup(
        &self,
        key: &str,
        contexts: Option<&BTreeSet<Vec<u8>>>,
        only_more_popular: bool,
        num: usize,
    ) -> Result<Vec<LookupResult>> {
        if only_more_popular {
            return Err(illegal(
                "this suggester only works with onlyMorePopular=false",
            ));
        }
        if contexts.is_some() {
            return Err(illegal("this suggester doesn't support contexts"));
        }
        let Some(fst) = &self.fst else {
            return Ok(Vec::new());
        };
        for c in key.chars() {
            if c == '\u{1e}' {
                return Err(illegal(
                    "lookup key cannot contain HOLE character U+001E; this character is reserved",
                ));
            }
            if c == '\u{1f}' {
                return Err(illegal(
                    "lookup key cannot contain unit separator character U+001F; this character is reserved",
                ));
            }
        }
        if num == 0 {
            return Ok(Vec::new());
        }
        let utf8_key = key.as_bytes();
        let lookup_automaton = self.to_lookup_automaton(key)?;
        let mut reader = fst.bytes_reader();
        let mut results: Vec<LookupResult> = Vec::new();

        let mut prefix_paths =
            intersect_prefix_paths(&self.convert_automaton(lookup_automaton.clone())?, fst)?;
        let max_surface = self.max_surface_forms_per_analyzed_form as usize;
        let weight_cmp = |a: &Output, b: &Output| a.first.cmp(&b.first);

        if self.exact_first {
            let mut exact = Vec::new();
            for path in &prefix_paths {
                if let Some(arc) = fst
                    .find_target_arc(END_BYTE, &path.fst_node, &mut reader)
                    .map_err(fst_err)?
                {
                    // This node has END_BYTE arc leaving, meaning it's an
                    // "exact" match:
                    exact.push((arc, path));
                }
            }
            let limit = exact.len().saturating_mul(max_surface);
            let mut searcher = TopNSearcher::new(fst, limit, limit, weight_cmp);
            // NOTE: we could almost get away with only using the first start
            // node; the catch is maxSurfaceFormsPerAnalyzedForm having pruned
            // the exact match from one of them.
            for (arc, path) in &exact {
                searcher
                    .add_start_paths(
                        arc,
                        SuggestOutputs::add(&path.output, arc.output()),
                        false,
                        path.input.clone(),
                    )
                    .map_err(fst_err)?;
            }
            let completions = searcher.search().map_err(fst_err)?;
            // every "exactly the same analyzed form" path, then a linear scan
            // for the one whose surface form is the query
            for completion in &completions.top_n {
                if same_surface_form(self.has_payloads, utf8_key, &completion.output.second) {
                    results.push(self.lookup_result(&completion.output));
                    break;
                }
            }
            if results.len() == num {
                return Ok(results); // that was quick
            }
        }

        let has_payloads = self.has_payloads;
        let exact_first = self.exact_first;
        let depth = num.saturating_mul(self.max_analyzed_paths_for_one_input as usize);
        let mut searcher = TopNSearcher::new(fst, num - results.len(), depth, weight_cmp);
        let mut seen: HashSet<Vec<u8>> = HashSet::new();
        searcher.accept_result = Some(Box::new(move |path| {
            // Dedup: when the input analyzes to a graph we can get duplicate
            // surface forms:
            if !seen.insert(path.output.second.clone()) {
                return false;
            }
            // In exactFirst mode, don't accept any paths matching the surface
            // form: the first search already found it
            !(exact_first && same_surface_form(has_payloads, utf8_key, &path.output.second))
        }));

        prefix_paths = self.full_prefix_paths(prefix_paths, &lookup_automaton, fst)?;
        for path in &prefix_paths {
            searcher
                .add_start_paths(
                    &path.fst_node,
                    path.output.clone(),
                    true,
                    path.input.clone(),
                )
                .map_err(fst_err)?;
        }
        let completions = searcher.search().map_err(fst_err)?;
        for completion in &completions.top_n {
            results.push(self.lookup_result(&completion.output));
            if results.len() == num {
                // In the exactFirst=true case the search may produce one extra path
                break;
            }
        }
        Ok(results)
    }
}

/// `FuzzySuggester`: an [`AnalyzingSuggester`] whose query may be misspelt by
/// up to `max_edits` edits past a non-fuzzy prefix.
pub struct FuzzySuggester {
    inner: AnalyzingSuggester,
}

impl FuzzySuggester {
    /// `FuzzySuggester.DEFAULT_UNICODE_AWARE`.
    pub const DEFAULT_UNICODE_AWARE: bool = false;
    /// `FuzzySuggester.DEFAULT_MIN_FUZZY_LENGTH`.
    pub const DEFAULT_MIN_FUZZY_LENGTH: i32 = 3;
    /// `FuzzySuggester.DEFAULT_NON_FUZZY_PREFIX`.
    pub const DEFAULT_NON_FUZZY_PREFIX: i32 = 1;
    /// `FuzzySuggester.DEFAULT_MAX_EDITS`.
    pub const DEFAULT_MAX_EDITS: i32 = 1;
    /// `FuzzySuggester.DEFAULT_TRANSPOSITIONS`.
    pub const DEFAULT_TRANSPOSITIONS: bool = true;

    /// `new FuzzySuggester(tempDir, prefix, analyzer)`.
    pub fn new(analyzer: Arc<Analyzer>) -> Self {
        Self::with_analyzers(analyzer.clone(), analyzer)
    }

    /// `new FuzzySuggester(tempDir, prefix, indexAnalyzer, queryAnalyzer)`.
    pub fn with_analyzers(index_analyzer: Arc<Analyzer>, query_analyzer: Arc<Analyzer>) -> Self {
        Self::with_options(
            index_analyzer,
            query_analyzer,
            EXACT_FIRST | PRESERVE_SEP,
            256,
            -1,
            true,
            Self::DEFAULT_MAX_EDITS,
            Self::DEFAULT_TRANSPOSITIONS,
            Self::DEFAULT_NON_FUZZY_PREFIX,
            Self::DEFAULT_MIN_FUZZY_LENGTH,
            Self::DEFAULT_UNICODE_AWARE,
        )
        .expect("the default options are valid")
    }

    /// The full constructor.
    ///
    /// # Errors
    /// `IllegalArgument` for the options Java rejects.
    #[allow(clippy::too_many_arguments)]
    pub fn with_options(
        index_analyzer: Arc<Analyzer>,
        query_analyzer: Arc<Analyzer>,
        options: i32,
        max_surface_forms_per_analyzed_form: i32,
        max_graph_expansions: i32,
        preserve_position_increments: bool,
        max_edits: i32,
        transpositions: bool,
        non_fuzzy_prefix: i32,
        min_fuzzy_length: i32,
        unicode_aware: bool,
    ) -> Result<Self> {
        let mut inner = AnalyzingSuggester::with_options(
            index_analyzer,
            query_analyzer,
            options,
            max_surface_forms_per_analyzed_form,
            max_graph_expansions,
            preserve_position_increments,
        )?;
        if !(0..=lucene_util::automaton::MAXIMUM_SUPPORTED_DISTANCE).contains(&max_edits) {
            return Err(illegal(format!(
                "maxEdits must be between 0 and {}",
                lucene_util::automaton::MAXIMUM_SUPPORTED_DISTANCE
            )));
        }
        if non_fuzzy_prefix < 0 {
            return Err(illegal(format!(
                "nonFuzzyPrefix must not be >= 0 (got {non_fuzzy_prefix})"
            )));
        }
        if min_fuzzy_length < 0 {
            return Err(illegal(format!(
                "minFuzzyLength must not be >= 0 (got {min_fuzzy_length})"
            )));
        }
        inner.fuzzy = Some(FuzzyOptions {
            max_edits,
            transpositions,
            non_fuzzy_prefix,
            min_fuzzy_length,
            unicode_aware,
        });
        Ok(Self { inner })
    }

    /// `get(key)`: Java throws `UnsupportedOperationException`.
    pub fn get(&self, key: &str) -> Result<Option<i64>> {
        self.inner.get(key)
    }
}

impl Lookup for FuzzySuggester {
    fn get_count(&self) -> Result<u64> {
        self.inner.get_count()
    }
    fn build(&mut self, input: &mut dyn InputIterator) -> Result<()> {
        self.inner.build(input)
    }
    fn store(&self, output: &mut dyn DataOutput) -> Result<bool> {
        self.inner.store(output)
    }
    fn load(&mut self, input: &mut dyn DataInput) -> Result<bool> {
        self.inner.load(input)
    }
    fn lookup(
        &self,
        key: &str,
        contexts: Option<&BTreeSet<Vec<u8>>>,
        only_more_popular: bool,
        num: usize,
    ) -> Result<Vec<LookupResult>> {
        self.inner.lookup(key, contexts, only_more_popular, num)
    }
}

/// `SuggestStopFilter`: a [`StopFilter`](lucene_analysis::StopFilter) for
/// queries typed so far. A stop word that ends the input with no separator
/// after it may still be growing into another word, so it is kept (as a
/// keyword); one followed by a separator is dropped.
pub struct SuggestStopFilter<I: TokenStream> {
    input: I,
    stop_words: Arc<CharArraySet>,
    end_state: Option<State>,
}

impl<I: TokenStream> SuggestStopFilter<I> {
    /// `new SuggestStopFilter(input, stopWords)`.
    pub fn new(input: I, stop_words: Arc<CharArraySet>) -> Self {
        Self {
            input,
            stop_words,
            end_state: None,
        }
    }

    /// Adds the positions skipped so far to the current token's increment.
    fn add_skipped(&mut self, skipped: i32) -> std::result::Result<(), AnalysisError> {
        let atts = self.input.attributes_mut();
        let inc = atts.position_increment();
        atts.set_position_increment(skipped.saturating_add(inc))
    }
}

impl<I: TokenStream> TokenFilter for SuggestStopFilter<I> {
    type Input = I;

    fn input(&self) -> &I {
        &self.input
    }

    fn input_mut(&mut self) -> &mut I {
        &mut self.input
    }

    fn reset_filter(&mut self) -> std::result::Result<(), AnalysisError> {
        self.input.reset()?;
        self.end_state = None;
        Ok(())
    }

    fn end_filter(&mut self) -> std::result::Result<(), AnalysisError> {
        match &self.end_state {
            None => self.input.end(),
            // we already called end() from increment() when the stream was
            // complete, so we do not call it again here
            Some(state) => {
                let state = state.clone();
                self.input.attributes_mut().restore_state(&state);
                Ok(())
            }
        }
    }

    fn increment(&mut self) -> std::result::Result<bool, AnalysisError> {
        if self.end_state.is_some() || !self.input.increment_token()? {
            return Ok(false);
        }
        let mut skipped_positions = 0i32;
        loop {
            let atts = self.input.attributes();
            if !self.stop_words.contains(atts.term()) {
                // Not a stopword; return the current token:
                self.add_skipped(skipped_positions)?;
                return Ok(true);
            }
            let pos_inc = atts.position_increment();
            let end_offset = atts.end_offset();
            // This token may be a stopword, if it's not the end:
            let saved = atts.capture_state();
            if self.input.increment_token()? {
                // It was a stopword; skip it
                skipped_positions = skipped_positions.saturating_add(pos_inc);
                continue;
            }
            self.input.attributes_mut().clear_attributes();
            self.input.end()?;
            self.end_state = Some(self.input.attributes().capture_state());
            if self.input.attributes().end_offset() > end_offset {
                // There was a token separator after the stopword, so it was one
                return Ok(false);
            }
            // No token separator after the final token that looked like a
            // stop-word; don't filter it:
            self.input.attributes_mut().restore_state(&saved);
            self.add_skipped(skipped_positions)?;
            self.input.attributes_mut().set_keyword(true);
            return Ok(true);
        }
    }
}
