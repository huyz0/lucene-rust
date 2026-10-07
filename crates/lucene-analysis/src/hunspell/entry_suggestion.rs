//! `WordFormGenerator.compress` ("munch") and its `EntrySuggestion`: the
//! dictionary entries (stems with flags) that would generate a given list of
//! words, found by Java's best-first search over candidate stems and flag
//! sets.
//!
//! The search order is Java's, so the answer is too: `java.util.PriorityQueue`'s
//! binary heap is reproduced ([`JavaPriorityQueue`]), maps and sets iterate
//! in Java's insertion order (`LinkedHashMap`, `LinkedHashSet`), and the
//! `visited` set compares states as Java's `Map.equals` does (by content).

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fmt;

use super::dictionary::{DictEntry, AFFIX_FLAG};
use super::word_form_generator::WordFormGenerator;
use super::{HunspellError, FLAG_UNSET};

/// `org.apache.lucene.analysis.hunspell.EntrySuggestion`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntrySuggestion {
    /// `getEntriesToEdit()`: existing entries whose flags would change.
    pub entries_to_edit: Vec<DictEntry>,
    /// `getEntriesToAdd()`: new entries.
    pub entries_to_add: Vec<DictEntry>,
    /// `getExtraGeneratedWords()`: words the entries generate beyond those asked for.
    pub extra_generated: Vec<String>,
}

/// Java's `List.toString`.
fn list<T: fmt::Display>(items: &[T]) -> String {
    let parts: Vec<String> = items.iter().map(ToString::to_string).collect();
    format!("[{}]", parts.join(", "))
}

impl fmt::Display for EntrySuggestion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "EntrySuggestion{{toEdit={}, toAdd={}, extra={}}}",
            list(&self.entries_to_edit),
            list(&self.entries_to_add),
            list(&self.extra_generated)
        )
    }
}

/// `FlagSet`: a set of flags, kept sorted (Java's `CharHashSet`, compared as
/// a set).
type FlagSet = Vec<u16>;

/// `Set<FlagSet>` in insertion order (`LinkedHashSet`, or `Set.of`).
type FlagSets = Vec<FlagSet>;

/// `Map<String, Set<FlagSet>>` in insertion order (`LinkedHashMap`).
type StemToFlags = Vec<(String, FlagSets)>;

/// `FlagSet.flatten`: the union, sorted.
fn flatten(sets: &[FlagSet]) -> FlagSet {
    let mut all: FlagSet = sets.iter().flatten().copied().collect();
    all.sort_unstable();
    all.dedup();
    all
}

/// A map's content, for `Map.equals`/`hashCode` (the `visited` set).
fn content_key(m: &StemToFlags) -> Vec<(String, Vec<FlagSet>)> {
    let mut k: Vec<(String, Vec<FlagSet>)> = m
        .iter()
        .map(|(s, f)| {
            let mut f = f.clone();
            f.sort();
            (s.clone(), f)
        })
        .collect();
    k.sort();
    k
}

/// `WordCompressor.State`.
#[derive(Debug, Clone)]
struct State {
    stem_to_flags: StemToFlags,
    under_generated: usize,
    over_generated: usize,
    /// The most requested forms adding only flags to this state could reach.
    potential_coverage: usize,
}

/// `solutionFitness`: more potential coverage first, then fewer stems,
/// fewer missing words, fewer extra words.
fn fitness(a: &State, b: &State) -> Ordering {
    b.potential_coverage
        .cmp(&a.potential_coverage)
        .then(a.stem_to_flags.len().cmp(&b.stem_to_flags.len()))
        .then(a.under_generated.cmp(&b.under_generated))
        .then(a.over_generated.cmp(&b.over_generated))
}

/// `java.util.PriorityQueue` with a comparator: the same binary heap, so
/// equal elements leave in Java's order.
struct JavaPriorityQueue<T, F: Fn(&T, &T) -> Ordering> {
    queue: Vec<T>,
    cmp: F,
}

impl<T, F: Fn(&T, &T) -> Ordering> JavaPriorityQueue<T, F> {
    fn new(cmp: F) -> Self {
        JavaPriorityQueue {
            queue: Vec::new(),
            cmp,
        }
    }

    /// `offer`: `siftUp`.
    fn offer(&mut self, x: T) {
        self.queue.push(x);
        let mut k = self.queue.len() - 1;
        while k > 0 {
            let parent = (k - 1) >> 1;
            if (self.cmp)(&self.queue[k], &self.queue[parent]) != Ordering::Less {
                break;
            }
            self.queue.swap(k, parent);
            k = parent;
        }
    }

    /// `poll`: the last element moved to the root, then `siftDown`.
    fn poll(&mut self) -> Option<T> {
        let n = self.queue.len().checked_sub(1)?;
        self.queue.swap(0, n);
        let result = self.queue.pop();
        let half = n >> 1;
        let mut k = 0;
        while k < half {
            let mut child = 2 * k + 1;
            let right = child + 1;
            if right < n && (self.cmp)(&self.queue[child], &self.queue[right]) == Ordering::Greater
            {
                child = right;
            }
            if (self.cmp)(&self.queue[k], &self.queue[child]) != Ordering::Greater {
                break;
            }
            self.queue.swap(k, child);
            k = child;
        }
        result
    }
}

/// `WordFormGenerator.WordCompressor`.
struct WordCompressor<'g, 'd> {
    generator: &'g WordFormGenerator<'d>,
    forbidden: &'g HashSet<String>,
    word_set: HashSet<String>,
    existing_stems: HashSet<String>,
    stem_to_possible_flags: HashMap<String, FlagSets>,
    /// `stemsToForms`, in insertion order.
    stems_to_forms: Vec<(String, Vec<String>)>,
    expansion_cache: HashMap<(String, Vec<FlagSet>), Vec<String>>,
}

impl WordFormGenerator<'_> {
    /// `compress(words, forbidden, checkCanceled)`: entries generating every
    /// word in `words` (dictionary format and case) and none in
    /// `forbidden`, or `None` when no combination does.
    ///
    /// # Errors
    /// `IllegalArgument` when `words` and `forbidden` intersect.
    pub fn compress(
        &self,
        words: &[&str],
        forbidden: &HashSet<String>,
    ) -> Result<Option<EntrySuggestion>, HunspellError> {
        if words.is_empty() {
            return Ok(None);
        }
        if words.iter().any(|w| forbidden.contains(*w)) {
            return Err(HunspellError::IllegalArgument(
                "'words' and 'forbidden' shouldn't intersect".to_string(),
            ));
        }
        Ok(WordCompressor::new(self, words, forbidden).compress())
    }
}

fn index_of(stems: &[(String, Vec<String>)], stem: &str) -> Option<usize> {
    stems.iter().position(|(s, _)| s == stem)
}

impl<'g, 'd> WordCompressor<'g, 'd> {
    fn new(
        generator: &'g WordFormGenerator<'d>,
        words: &[&str],
        forbidden: &'g HashSet<String>,
    ) -> Self {
        let mut c = WordCompressor {
            generator,
            forbidden,
            word_set: words.iter().map(|w| w.to_string()).collect(),
            existing_stems: HashSet::new(),
            stem_to_possible_flags: HashMap::new(),
            stems_to_forms: Vec::new(),
            expansion_cache: HashMap::new(),
        };
        let d = generator.dictionary;
        for &word in words {
            c.stem_to_possible_flags
                .entry(word.to_string())
                .or_default();
            c.register_stem(word, word);
            let units: Vec<u16> = word.encode_utf16().collect();
            let mut found: Vec<(String, FlagSet)> = Vec::new();
            generator
                .stemmer
                .remove_affixes_with_candidates(&units, &mut |cand| {
                    let mut flags: FlagSet = [
                        cand.outer_prefix,
                        cand.inner_prefix,
                        cand.outer_suffix,
                        cand.inner_suffix,
                    ]
                    .into_iter()
                    .filter(|&a| a >= 0)
                    .map(|a| d.affix_data(a, AFFIX_FLAG))
                    .collect();
                    flags.sort_unstable();
                    flags.dedup();
                    found.push((String::from_utf16_lossy(cand.word), flags));
                    true
                });
            for (candidate, flags) in found {
                let generates_forbidden = !forbidden.is_empty()
                    && c.all_generated_for(&candidate, std::slice::from_ref(&flags))
                        .iter()
                        .any(|w| forbidden.contains(w));
                if !generates_forbidden {
                    c.register_stem(&candidate, word);
                    let possible = c.stem_to_possible_flags.entry(candidate).or_default();
                    if !possible.contains(&flags) {
                        possible.push(flags);
                    }
                }
            }
        }
        c.existing_stems = c
            .stems_to_forms
            .iter()
            .map(|(s, _)| s)
            .filter(|s| d.lookup_entries(s).is_some())
            .cloned()
            .collect();
        c
    }

    /// `registerStem`: `stemsToForms[stem] += word`.
    fn register_stem(&mut self, stem: &str, word: &str) {
        let i = match index_of(&self.stems_to_forms, stem) {
            Some(i) => i,
            None => {
                self.stems_to_forms.push((stem.to_string(), Vec::new()));
                self.stems_to_forms.len() - 1
            }
        };
        let forms = &mut self.stems_to_forms[i].1;
        if !forms.iter().any(|f| f == word) {
            forms.push(word.to_string());
        }
    }

    fn forms_of(&self, stem: &str) -> &[String] {
        index_of(&self.stems_to_forms, stem).map_or(&[], |i| &self.stems_to_forms[i].1)
    }

    /// `allGenerated(StemWithFlags)`: the words `stem` with the union of
    /// `flags` generates, cached.
    fn all_generated_for(&mut self, stem: &str, flags: &[FlagSet]) -> Vec<String> {
        let mut key_flags = flags.to_vec();
        key_flags.sort();
        key_flags.dedup();
        let key = (stem.to_string(), key_flags);
        if let Some(words) = self.expansion_cache.get(&key) {
            return words.clone();
        }
        // `getAllWordForms(stem, toFlagString(flags))`, without the round
        // trip through a string (a supplementary UTF-8 flag's lone surrogate
        // survives in a Java string, not in a Rust one).
        let encoded = flatten(flags);
        let words: Vec<String> = self
            .generator
            .word_forms_of_flags(stem, self.to_flag_string(&encoded), encoded)
            .into_iter()
            .map(|w| w.word)
            .collect();
        self.expansion_cache.insert(key, words.clone());
        words
    }

    /// `allGenerated(Map)`: every entry's words, in map order.
    fn all_generated(&mut self, stem_to_flags: &StemToFlags) -> Vec<String> {
        let mut out = Vec::new();
        for (stem, flags) in stem_to_flags {
            out.extend(self.all_generated_for(stem, flags));
        }
        out
    }

    /// `toFlagString`: `printFlags` of the sorted flags.
    fn to_flag_string(&self, flags: &FlagSet) -> String {
        self.generator.dictionary.flag_parsing.print_flags(flags)
    }

    fn compress(mut self) -> Option<EntrySuggestion> {
        // `stemSorter`, reversed: existing stems first, then by more forms;
        // a stable sort keeps `stemsToForms`' order among equals.
        let mut sorted_stems: Vec<String> =
            self.stems_to_forms.iter().map(|(s, _)| s.clone()).collect();
        sorted_stems.sort_by(|a, b| {
            let key = |s: &String| (self.existing_stems.contains(s), self.forms_of(s).len());
            key(b).cmp(&key(a))
        });
        let mut queue = JavaPriorityQueue::new(fitness);
        let mut visited: HashSet<Vec<(String, Vec<FlagSet>)>> = HashSet::new();
        queue.offer(State {
            stem_to_flags: Vec::new(),
            under_generated: self.word_set.len(),
            over_generated: 0,
            potential_coverage: 0,
        });
        let mut result = None;
        while let Some(state) = queue.poll() {
            if state.under_generated == 0 {
                result = Some(state);
                break;
            }
            for stem in &sorted_stems {
                if state.stem_to_flags.iter().any(|(s, _)| s == stem) {
                    continue;
                }
                let mut with_stem = state.stem_to_flags.clone();
                with_stem.push((stem.clone(), Vec::new()));
                if visited.insert(content_key(&with_stem)) {
                    if let Some(next) = self.new_state(with_stem) {
                        if state.under_generated > next.under_generated
                            || next.potential_coverage > state.potential_coverage
                        {
                            queue.offer(next);
                        }
                    }
                }
            }
            if state.potential_coverage < self.word_set.len() {
                // No flags until the entries can potentially cover every form.
                continue;
            }
            for (i, (stem, current)) in state.stem_to_flags.iter().enumerate() {
                let possible = self
                    .stem_to_possible_flags
                    .get(stem)
                    .cloned()
                    .unwrap_or_default();
                for flags in possible {
                    if current.contains(&flags) {
                        continue;
                    }
                    let mut with_flags = state.stem_to_flags.clone();
                    with_flags[i].1.push(flags);
                    if visited.insert(content_key(&with_flags)) {
                        if let Some(next) = self.new_state(with_flags) {
                            if state.under_generated > next.under_generated {
                                queue.offer(next);
                            }
                        }
                    }
                }
            }
        }
        result.map(|r| self.suggestion_of(&r))
    }

    /// `newState`: `None` when the entries generate a forbidden word.
    fn new_state(&mut self, stem_to_flags: StemToFlags) -> Option<State> {
        let generated: HashSet<String> = self.all_generated(&stem_to_flags).into_iter().collect();
        let mut over_generated = 0;
        for s in &generated {
            if self.forbidden.contains(s) {
                return None;
            }
            if !self.word_set.contains(s) {
                over_generated += 1;
            }
        }
        let potential: HashSet<&String> = stem_to_flags
            .iter()
            .flat_map(|(s, _)| self.forms_of(s))
            .collect();
        let potential_coverage = potential.len();
        let under_generated = self
            .word_set
            .iter()
            .filter(|w| !generated.contains(*w))
            .count();
        Some(State {
            stem_to_flags,
            under_generated,
            over_generated,
            potential_coverage,
        })
    }

    fn suggestion_of(&mut self, state: &State) -> EntrySuggestion {
        let mut s = EntrySuggestion {
            entries_to_edit: Vec::new(),
            entries_to_add: Vec::new(),
            extra_generated: Vec::new(),
        };
        for (stem, flags) in &state.stem_to_flags {
            self.add_entry(&mut s, stem, &flatten(flags));
        }
        let mut extras = self.all_generated(&state.stem_to_flags);
        // `distinct().sorted()`: `String.compareTo`, UTF-16 order.
        extras.sort_by(|a, b| a.encode_utf16().cmp(b.encode_utf16()));
        extras.dedup();
        let forbidden_word = self.generator.dictionary.forbiddenword;
        for extra in extras {
            if self.word_set.contains(&extra) || self.existing_stems.contains(&extra) {
                continue;
            }
            if self.forbidden.contains(&extra) && forbidden_word != FLAG_UNSET {
                self.add_entry(&mut s, &extra, &vec![forbidden_word]);
            } else {
                s.extra_generated.push(extra);
            }
        }
        s
    }

    /// `addEntry`: to `toEdit` for an existing stem, else to `toAdd`.
    fn add_entry(&self, s: &mut EntrySuggestion, stem: &str, flags: &FlagSet) {
        let entry = DictEntry {
            stem: stem.to_string(),
            flags: self.to_flag_string(flags),
            morphological_data: String::new(),
        };
        if self.existing_stems.contains(stem) {
            s.entries_to_edit.push(entry);
        } else {
            s.entries_to_add.push(entry);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn priority_queue_is_javas_heap() {
        // Equal keys leave in heap order, not insertion order.
        let mut q = JavaPriorityQueue::new(|a: &(i32, char), b: &(i32, char)| a.0.cmp(&b.0));
        for x in [(1, 'a'), (1, 'b'), (0, 'c'), (1, 'd'), (1, 'e')] {
            q.offer(x);
        }
        let order: String = std::iter::from_fn(|| q.poll()).map(|x| x.1).collect();
        assert_eq!(order, "cedab");
        assert!(q.poll().is_none());
    }

    #[test]
    fn suggestion_prints_as_java() {
        let s = EntrySuggestion {
            entries_to_edit: vec![],
            entries_to_add: vec![DictEntry {
                stem: "a".into(),
                flags: "B".into(),
                morphological_data: String::new(),
            }],
            extra_generated: vec!["x".into(), "y".into()],
        };
        assert_eq!(
            s.to_string(),
            "EntrySuggestion{toEdit=[], toAdd=[a/B], extra=[x, y]}"
        );
    }
}
