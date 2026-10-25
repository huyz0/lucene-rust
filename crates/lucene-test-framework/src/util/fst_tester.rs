//! `FSTTester`: builds an FST from sorted input/output pairs, then checks
//! that every pair is accepted with its output, that random walks only find
//! added inputs, and that `IntsRefFSTEnum`'s `next`,
//! `seekCeil`, `seekFloor` and `seekExact` land where a binary search over
//! the pairs says they should.
//!
//! Every random draw happens in Java's order, so a seed reproduces Java's
//! walk. Rust's typed output algebra has no Java `instanceof List` test:
//! [`FstTester::do_test`] adds scalar outputs, and
//! [`FstTester::do_test_lists`] expands `ListOfOutputs` pairs into one scalar
//! compiler addition per value, preserving their order.
//!
//! Java's off-heap build (`dataOutput(indexOutput)`) writes the same bytes
//! the on-heap build saves, so it is modelled as a save/read round trip.
//! There is no randomized IOContext, but the on-heap save path retains its
//! three random draws before saving.

use std::collections::HashMap;

use lucene_codecs::fst::InputType;
use lucene_codecs::fst_compiler::{
    CompileError, FstCompiler, FstCompilerBuilder, FstOutputs, FstReadError,
};
use lucene_codecs::fst_util::{self, IntsRefFstEnum, TypedFst, END_LABEL};
use lucene_search::misc::{ListOfOutputs, ListOutput};
use lucene_util::java_random::JavaRandom;

use super::lucene_test_case::at_least;
use super::test_util::{next_int, random_realistic_unicode_string};

/// `FSTTester.getRandomString(random)`.
pub fn get_random_string(r: &mut JavaRandom) -> String {
    if r.next_boolean() {
        random_realistic_unicode_string(r)
    } else {
        simple_random_string(r)
    }
}

/// `FSTTester.simpleRandomString(r)`: up to 9 chars from `a..=f`.
pub fn simple_random_string(r: &mut JavaRandom) -> String {
    let end = r.next_int_bounded(10);
    (0..end)
        .map(|_| char::from(u8::try_from(next_int(r, 97, 102)).unwrap_or(b'a')))
        .collect()
}

/// `FSTTester.toIntsRef(s, inputMode)`: UTF-8 bytes (mode 0) or code points
/// (any other mode).
pub fn to_ints_ref(s: &str, input_mode: i32) -> Vec<i32> {
    if input_mode == 0 {
        fst_util::to_ints_ref(s.as_bytes())
    } else {
        // A `char` is at most 0x10FFFF.
        s.chars().map(|c| c as i32).collect()
    }
}

/// `FSTTester.InputOutput<T>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputOutput<V> {
    pub input: Vec<i32>,
    pub output: V,
}

/// `FSTTester<T>`.
pub struct FstTester<'r, O: FstOutputs> {
    random: &'r mut JavaRandom,
    multiplier: i32,
    input_mode: i32,
    pairs: Vec<InputOutput<O::Value>>,
    /// `nodeCount`, set by [`Self::do_test`].
    pub node_count: i64,
    /// `arcCount`, set by [`Self::do_test`].
    pub arc_count: i64,
}

impl<'r, O: FstOutputs> FstTester<'r, O> {
    /// `new FSTTester<>(random, dir, inputMode, pairs, outputs)`; `pairs`
    /// must be sorted by input, as Java's callers sort them. `multiplier` is
    /// `RANDOM_MULTIPLIER`, which `atLeast` scales by.
    pub fn new(
        random: &'r mut JavaRandom,
        multiplier: i32,
        input_mode: i32,
        pairs: Vec<InputOutput<O::Value>>,
    ) -> Self {
        FstTester {
            random,
            multiplier,
            input_mode,
            pairs,
            node_count: 0,
            arc_count: 0,
        }
    }

    /// The pairs under test.
    pub fn pairs(&self) -> &[InputOutput<O::Value>] {
        &self.pairs
    }

    /// `run(fst, term, null)`: the output `term` maps to, or `None`.
    fn run(fst: &TypedFst<O>, term: &[i32]) -> Result<Option<O::Value>, FstReadError> {
        let mut arc = fst.first_arc();
        let mut output = O::no_output();
        let mut reader = fst.bytes_reader();
        for i in 0..=term.len() {
            let label = term.get(i).copied().unwrap_or(END_LABEL);
            match fst.find_target_arc(label, &arc, &mut reader)? {
                None => return Ok(None),
                Some(next) => arc = next,
            }
            output = O::add(&output, arc.output());
        }
        Ok(Some(output))
    }

    /// `randomAcceptedWord(fst, in)`.
    fn random_accepted_word(
        &mut self,
        fst: &TypedFst<O>,
        input: &mut Vec<i32>,
        arcs: &mut Vec<fst_util::Arc<O::Value>>,
    ) -> Result<O::Value, FstReadError> {
        let mut arc = fst.first_arc();
        input.clear();
        let mut output = O::no_output();
        let mut reader = fst.bytes_reader();
        loop {
            let follow = arc.clone();
            fst.read_first_target_arc(&follow, &mut arc, &mut reader)?;
            arcs.push(arc.clone());
            while !arc.is_last() {
                fst.read_next_arc(&mut arc, &mut reader)?;
                arcs.push(arc.clone());
            }
            let pick = self.random.next_int_bounded(len_i32(arcs.len()));
            arc = arcs.swap_remove(index(pick));
            arcs.clear();
            output = O::add(&output, arc.output());
            if arc.label() == END_LABEL {
                break;
            }
            input.push(arc.label());
        }
        Ok(output)
    }

    /// `doTest()`: builds the FST, verifies it, and returns it (`None` when
    /// no pair was added). Assertion failures panic, as JUnit's do.
    pub fn do_test(&mut self) -> Result<Option<TypedFst<O>>, FstReadError> {
        self.do_test_with(|compiler, pair| compiler.add(&pair.input, pair.output.clone()))
    }

    /// The add policy is local to this test utility; production output
    /// algebras need no test-specific extension or runtime type inspection.
    fn do_test_with(
        &mut self,
        mut add: impl FnMut(&mut FstCompiler<O>, &InputOutput<O::Value>) -> Result<(), CompileError>,
    ) -> Result<Option<TypedFst<O>>, FstReadError> {
        let input_type = if self.input_mode == 0 {
            InputType::Byte1
        } else {
            InputType::Byte4
        };
        let use_off_heap = self.random.next_boolean();
        let mut compiler = FstCompilerBuilder::new(input_type).build::<O>();
        for pair in &self.pairs {
            if let Err(e) = add(&mut compiler, pair) {
                panic!("add({:?}) failed: {e}", pair.input);
            }
        }
        let mut fst = compiler.compile().map(TypedFst::from_compiled);
        if let Some(built) = fst.take() {
            // Off heap, Java reads the FST back from the file it was written
            // to; on heap, it round-trips through `save` half the time.
            let save_reload = use_off_heap || self.random.next_boolean();
            if save_reload && !use_off_heap {
                // Java: LuceneTestCase.newIOContext(random, IOContext.DEFAULT).
                self.random.next_int_bounded(4192);
                self.random.next_int_bounded(512);
                self.random.next_int_bounded(3);
            }
            fst = Some(if save_reload {
                TypedFst::read(&built.save())?
            } else {
                built
            });
        }
        self.verify_unpruned(fst.as_ref())?;
        let stats = compiler.stats();
        self.node_count = stats.node_count;
        self.arc_count = stats.arc_count;
        Ok(fst)
    }

    /// `verifyUnPruned(inputMode, fst)`.
    fn verify_unpruned(&mut self, fst: Option<&TypedFst<O>>) -> Result<(), FstReadError> {
        if self.pairs.is_empty() {
            assert!(fst.is_none(), "no pairs, yet an FST was built");
            return Ok(());
        }
        let fst = fst.expect("pairs were added but no FST was built");

        // Every pair is accepted, and `next()` steps through them in order.
        {
            let mut fst_enum = IntsRefFstEnum::new(fst);
            for pair in &self.pairs {
                let output = Self::run(fst, &pair.input)?;
                let output =
                    output.unwrap_or_else(|| panic!("term {:?} is not accepted", pair.input));
                assert_eq!(pair.output, output, "output of {:?}", pair.input);
                let (term, out) = fst_enum.next()?.expect("fstEnum ended early");
                assert_eq!(pair.input, term, "fstEnum.next()");
                assert_eq!(&pair.output, out, "fstEnum.next() output");
            }
            assert!(
                fst_enum.next()?.is_none(),
                "fstEnum.next() past the last pair"
            );
        }

        let terms_map: HashMap<Vec<i32>, O::Value> = self
            .pairs
            .iter()
            .map(|p| (p.input.clone(), p.output.clone()))
            .collect();

        // Random walks find only added inputs, with their outputs.
        let mut scratch = Vec::new();
        // Retain allocation across walks; every outgoing arc is still decoded
        // in Java order, and no decoded arc or output survives a walk.
        let mut arcs = Vec::new();
        let num = at_least(self.random, self.multiplier, 500);
        for _ in 0..num {
            let output = self.random_accepted_word(fst, &mut scratch, &mut arcs)?;
            let expected = terms_map
                .get(scratch.as_slice())
                .unwrap_or_else(|| panic!("accepted word {scratch:?} is not valid"));
            assert_eq!(expected, &output, "output of accepted word {scratch:?}");
        }

        self.verify_seeks(fst, &terms_map)?;
        self.verify_mixed(fst, &terms_map)
    }

    /// `binarySearch(pairs, term)` in `Collections.binarySearch`'s encoding.
    fn search(&self, term: &[i32]) -> i32 {
        match self
            .pairs
            .binary_search_by(|p| p.input.as_slice().cmp(term))
        {
            Ok(i) => len_i32(i),
            Err(i) => -len_i32(i) - 1,
        }
    }

    /// The `IntsRefFSTEnum.seek` section: seeks to absent and present terms.
    fn verify_seeks(
        &mut self,
        fst: &TypedFst<O>,
        terms_map: &HashMap<Vec<i32>, O::Value>,
    ) -> Result<(), FstReadError> {
        let mut fst_enum = IntsRefFstEnum::new(fst);
        let num = at_least(self.random, self.multiplier, 100);
        let size = len_i32(self.pairs.len());
        for _ in 0..num {
            if self.random.next_boolean() {
                // Seek to a term that does not exist.
                loop {
                    let term = to_ints_ref(&get_random_string(self.random), self.input_mode);
                    let mut pos = self.search(&term);
                    if pos >= 0 {
                        continue;
                    }
                    debug_assert!(!terms_map.contains_key(term.as_slice()));
                    pos = -(pos + 1);
                    let target = &term;
                    let result = if self.random.next_int_bounded(3) == 0 {
                        pos = -1;
                        fst_enum.seek_exact(target)?
                    } else if self.random.next_boolean() {
                        pos -= 1;
                        fst_enum.seek_floor(target)?
                    } else {
                        fst_enum.seek_ceil(target)?
                    };
                    if pos != -1 && pos < size {
                        let want = &self.pairs[index(pos)];
                        let (got, out) = result
                            .unwrap_or_else(|| panic!("got null but expected {:?}", want.input));
                        assert_eq!(want.input, got, "seek to {term:?}");
                        assert_eq!(&want.output, out, "seek output for {term:?}");
                    } else {
                        assert!(result.is_none(), "expected null seeking {term:?}");
                    }
                    break;
                }
            } else {
                // Seek to a term that exists.
                let pick = self.random.next_int_bounded(size);
                let pair = &self.pairs[index(pick)];
                let target = &pair.input;
                let result = if self.random.next_int_bounded(3) == 2 {
                    fst_enum.seek_exact(target)?
                } else if self.random.next_boolean() {
                    fst_enum.seek_floor(target)?
                } else {
                    fst_enum.seek_ceil(target)?
                };
                let (got, out) = result.expect("seek to an existing term returned null");
                assert_eq!(pair.input, got, "seek to existing term");
                assert_eq!(&pair.output, out, "seek output for existing term");
            }
        }
        Ok(())
    }

    /// The "mixed next/seek" section.
    fn verify_mixed(
        &mut self,
        fst: &TypedFst<O>,
        terms_map: &HashMap<Vec<i32>, O::Value>,
    ) -> Result<(), FstReadError> {
        let size = len_i32(self.pairs.len());
        let num = at_least(self.random, self.multiplier, 100);
        for _ in 0..num {
            let mut fst_enum = IntsRefFstEnum::new(fst);
            let mut upto: i32 = -1;
            loop {
                let is_done;
                if upto == size - 1 || self.random.next_boolean() {
                    upto += 1;
                    is_done = fst_enum.next()?.is_none();
                } else if upto != -1
                    && f64::from(upto) < 0.75 * f64::from(size)
                    && self.random.next_boolean()
                {
                    let mut found = None;
                    for _ in 0..10 {
                        let term = to_ints_ref(&get_random_string(self.random), self.input_mode);
                        if !terms_map.contains_key(term.as_slice())
                            && term.as_slice() > self.pairs[index(upto)].input.as_slice()
                        {
                            let pos = self.search(&term);
                            debug_assert!(pos < 0);
                            upto = -(pos + 1);
                            let target = &term;
                            found = Some(if self.random.next_boolean() {
                                upto -= 1;
                                assert!(upto != -1);
                                fst_enum.seek_floor(target)?.is_none()
                            } else {
                                fst_enum.seek_ceil(target)?.is_none()
                            });
                            break;
                        }
                    }
                    match found {
                        Some(done) => is_done = done,
                        None => continue,
                    }
                } else {
                    let inc = self.random.next_int_bounded(size - upto - 1);
                    upto += inc;
                    if upto == -1 {
                        upto = 0;
                    }
                    let target = &self.pairs[index(upto)].input;
                    is_done = if self.random.next_boolean() {
                        fst_enum.seek_ceil(target)?.is_none()
                    } else {
                        fst_enum.seek_floor(target)?.is_none()
                    };
                }
                if upto == size {
                    assert!(is_done, "enum not exhausted after the last pair");
                    break;
                }
                assert!(!is_done, "enum exhausted at pair {upto}");
                let want = &self.pairs[index(upto)];
                // SENTINEL-OK: IntsRefFstEnum::current returns Option, checked
                // by expect; current() in the diagnostics is not an i32 call.
                let (got, out) = fst_enum.current().expect("current() after a hit");
                assert_eq!(want.input, got, "current() at pair {upto}");
                assert_eq!(&want.output, out, "current() output at pair {upto}");
            }
        }
        Ok(())
    }
}

impl<O: FstOutputs> FstTester<'_, ListOfOutputs<O>> {
    /// Java: `FSTTester.doTest`'s `pair.output instanceof List` branch.
    /// Add each list value as a scalar to the same input, letting the
    /// compiler merge final outputs. Scalar pairs still add once. Empty and
    /// one-element lists fail verification, as they do in Java: callers use
    /// a scalar for one output and a list for two or more.
    pub fn do_test_lists(&mut self) -> Result<Option<TypedFst<ListOfOutputs<O>>>, FstReadError> {
        self.do_test_with(|compiler, pair| {
            match &pair.output {
                ListOutput::One(value) => {
                    compiler.add(&pair.input, ListOutput::One(value.clone()))?;
                }
                ListOutput::Many(values) => {
                    for value in values {
                        compiler.add(&pair.input, ListOutput::One(value.clone()))?;
                    }
                }
            }
            Ok(())
        })
    }
}

/// A pair count or position as Java's `int`.
fn len_i32(n: usize) -> i32 {
    i32::try_from(n).expect("more pairs than an int counts")
}

/// A non-negative Java `int` index.
fn index(i: i32) -> usize {
    usize::try_from(i).expect("negative index")
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_codecs::fst::{ByteSequenceOutputs, PositiveIntOutputs};
    use lucene_codecs::fst_compiler::NoOutputs;

    fn sorted_pairs<V: Clone>(
        r: &mut JavaRandom,
        mode: i32,
        n: usize,
        mut out: impl FnMut(&mut JavaRandom, usize) -> V,
    ) -> Vec<InputOutput<V>> {
        let mut inputs: Vec<Vec<i32>> = (0..n)
            .map(|_| to_ints_ref(&get_random_string(r), mode))
            .collect();
        inputs.sort();
        inputs.dedup();
        inputs
            .into_iter()
            .enumerate()
            .map(|(i, input)| InputOutput {
                input,
                output: out(r, i),
            })
            .collect()
    }

    #[test]
    fn byte_inputs_with_int_outputs_verify() {
        for seed in 0..6 {
            let mut r = JavaRandom::new(seed);
            let mut pairs = sorted_pairs(&mut r, 0, 60, |r, _| i64::from(r.next_int_bounded(5)));
            // Outputs must not decrease for a positive-int FST to be useful,
            // but any non-negative value is legal.
            pairs.iter_mut().for_each(|p| p.output = p.output.abs());
            let mut t = FstTester::<PositiveIntOutputs>::new(&mut r, 1, 0, pairs);
            let fst = t.do_test().unwrap();
            assert!(fst.is_some());
            assert!(t.node_count > 0 && t.arc_count > 0);
            assert!(!t.pairs().is_empty());
        }
    }

    #[test]
    fn code_point_inputs_and_no_outputs_verify() {
        for seed in 0..4 {
            let mut r = JavaRandom::new(seed);
            let pairs = sorted_pairs(&mut r, 1, 40, |_, _| ());
            let mut t = FstTester::<NoOutputs>::new(&mut r, 1, 1, pairs);
            assert!(t.do_test().unwrap().is_some());
        }
    }

    #[test]
    fn code_point_inputs_with_outputs_verify_all_enum_operations() {
        for seed in 0..6 {
            let mut random = JavaRandom::new(seed + 81);
            let pairs = sorted_pairs(&mut random, 1, 60, |_, i| i as i64 * 17);
            let mut tester = FstTester::<PositiveIntOutputs>::new(&mut random, 1, 1, pairs);
            assert!(tester.do_test().unwrap().is_some());
        }
    }

    #[test]
    fn byte_sequence_outputs_verify() {
        let mut r = JavaRandom::new(42);
        let pairs = sorted_pairs(&mut r, 0, 50, |r, i| {
            let mut b = vec![0u8; i % 3];
            r.next_bytes(&mut b);
            b
        });
        let mut t = FstTester::<ByteSequenceOutputs>::new(&mut r, 1, 0, pairs);
        assert!(t.do_test().unwrap().is_some());
    }

    #[test]
    fn no_pairs_build_no_fst() {
        let mut r = JavaRandom::new(1);
        let mut t = FstTester::<NoOutputs>::new(&mut r, 1, 0, Vec::new());
        assert!(t.do_test().unwrap().is_none());
    }

    #[test]
    fn empty_input_alone_is_accepted() {
        let mut r = JavaRandom::new(3);
        let pairs = vec![InputOutput {
            input: Vec::new(),
            output: 7i64,
        }];
        let mut t = FstTester::<PositiveIntOutputs>::new(&mut r, 1, 0, pairs);
        assert!(t.do_test().unwrap().is_some());
    }

    #[test]
    fn list_pairs_expand_scalars_before_shared_prefix_compilation() {
        for mode in [0, 1] {
            let expected = [
                ("", ListOutput::Many(vec![0, 9, 0])),
                ("a", ListOutput::Many(vec![5, 1])),
                ("ab", ListOutput::One(0)),
                ("abc", ListOutput::Many(vec![1, 0, 1, 1 << 40])),
                ("\u{10000}", ListOutput::Many(vec![0, 7, 0])),
            ];
            let pairs = expected
                .iter()
                .map(|(term, output)| InputOutput {
                    input: to_ints_ref(term, mode),
                    output: output.clone(),
                })
                .collect();
            let mut random = JavaRandom::new(42);
            let mut tester =
                FstTester::<ListOfOutputs<PositiveIntOutputs>>::new(&mut random, 1, mode, pairs);
            let fst = tester.do_test_lists().unwrap().unwrap();
            let mut enumeration = IntsRefFstEnum::new(&fst);
            for (term, output) in expected {
                let (actual_term, actual_output) = enumeration.next().unwrap().unwrap();
                assert_eq!(actual_term, to_ints_ref(term, mode));
                assert_eq!(actual_output, &output);
            }
            assert!(enumeration.next().unwrap().is_none());
        }
    }

    #[test]
    fn invalid_empty_and_singleton_lists_fail_instead_of_becoming_scalar_pairs() {
        for values in [Vec::new(), vec![7]] {
            let mut random = JavaRandom::new(1);
            let pairs = vec![InputOutput {
                input: to_ints_ref("a", 0),
                output: ListOutput::Many(values),
            }];
            let mut tester =
                FstTester::<ListOfOutputs<PositiveIntOutputs>>::new(&mut random, 1, 0, pairs);
            assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                tester.do_test_lists().unwrap();
            }))
            .is_err());
        }
    }

    #[test]
    #[should_panic(expected = "is not accepted")]
    fn a_wrong_pair_is_caught() {
        // Verify against pairs the FST was not built from: the tester's
        // checks must fire.
        let mut r = JavaRandom::new(5);
        let pairs = vec![
            InputOutput {
                input: to_ints_ref("ab", 0),
                output: (),
            },
            InputOutput {
                input: to_ints_ref("cd", 0),
                output: (),
            },
        ];
        let mut t = FstTester::<NoOutputs>::new(&mut r, 1, 0, pairs);
        let fst = t.do_test().unwrap();
        t.pairs.push(InputOutput {
            input: to_ints_ref("zz", 0),
            output: (),
        });
        t.verify_unpruned(fst.as_ref()).unwrap();
    }

    #[test]
    #[should_panic(expected = "output of")]
    fn a_wrong_output_is_caught() {
        let mut r = JavaRandom::new(5);
        let pairs = vec![InputOutput {
            input: to_ints_ref("ab", 0),
            output: 3i64,
        }];
        let mut t = FstTester::<PositiveIntOutputs>::new(&mut r, 1, 0, pairs);
        let fst = t.do_test().unwrap();
        t.pairs[0].output = 4;
        t.verify_unpruned(fst.as_ref()).unwrap();
    }

    #[test]
    fn strings_match_java_shapes() {
        let mut r = JavaRandom::new(9);
        for _ in 0..50 {
            let s = simple_random_string(&mut r);
            assert!(s.len() < 10 && s.bytes().all(|b| (b'a'..=b'f').contains(&b)));
        }
        assert_eq!(to_ints_ref("\u{10400}a", 1), vec![0x10400, 97]);
        assert_eq!(to_ints_ref("\u{e9}", 0), vec![0xC3, 0xA9]);
    }
}
