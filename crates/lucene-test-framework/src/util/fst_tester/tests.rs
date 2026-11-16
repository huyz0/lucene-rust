#![cfg(test)]
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
