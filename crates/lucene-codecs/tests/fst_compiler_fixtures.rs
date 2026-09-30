//! Differential test for the `FSTCompiler` port against Lucene 10.5.0:
//! replays every `fixtures/data/fst_compiler/*.txt` case (written by
//! `fixtures/src/GenFstCompiler.java`) through
//! `lucene_codecs::fst_compiler::FstCompiler` with the same configuration and
//! the same `add` calls, and requires `FST.save(out, out)`'s bytes exactly,
//! plus Java's node and arc counts. Byte-sequence FSTs are also read back
//! through this crate's `Fst` reader, which must return every key's output.
#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::fst::{
    ByteSequenceOutputs, Fst, InputType, Pair, PairOutputs, PositiveIntOutputs,
};
use lucene_codecs::fst_compiler::{
    CharSequenceOutputs, CompiledFst, FstCompiler, FstCompilerBuilder, FstOutputs,
    IntSequenceOutputs, NoOutputs,
};
use lucene_store::data_input::SliceInput;

fn dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/fst_compiler")
}

fn unhex(s: &str) -> Vec<u8> {
    if s == "-" {
        return Vec::new();
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

fn csv(s: &str) -> Vec<i64> {
    if s == "-" {
        return Vec::new();
    }
    s.split(',').map(|x| x.parse().unwrap()).collect()
}

struct Case {
    name: String,
    outputs: String,
    input_type: InputType,
    ram_mb: f64,
    allow_fixed: bool,
    da_factor: f32,
    version: i32,
    adds: Vec<(Vec<i32>, String)>,
    stats: (i64, i64),
    fst: Option<Vec<u8>>,
}

fn parse(name: &str, text: &str) -> Case {
    let mut lines = text.lines();
    let cfg: Vec<&str> = lines.next().unwrap().split(' ').collect();
    assert_eq!(cfg[0], "config");
    let input_type = match cfg[2] {
        "BYTE1" => InputType::Byte1,
        "BYTE2" => InputType::Byte2,
        _ => InputType::Byte4,
    };
    let mut case = Case {
        name: name.to_string(),
        outputs: cfg[1].to_string(),
        input_type,
        ram_mb: cfg[3].parse().unwrap(),
        allow_fixed: cfg[4] == "true",
        da_factor: cfg[5].parse().unwrap(),
        version: cfg[6].parse().unwrap(),
        adds: Vec::new(),
        stats: (0, 0),
        fst: None,
    };
    for line in lines {
        let p: Vec<&str> = line.split(' ').collect();
        match p[0] {
            "k" => {
                let key: Vec<i32> = if input_type == InputType::Byte1 {
                    unhex(p[1]).into_iter().map(i32::from).collect()
                } else {
                    csv(p[1]).into_iter().map(|v| v as i32).collect()
                };
                case.adds.push((key, p[2].to_string()));
            }
            "stats" => case.stats = (p[1].parse().unwrap(), p[2].parse().unwrap()),
            "fst" => case.fst = (p[1] != "-").then(|| unhex(p[1])),
            other => panic!("{name}: unknown line {other}"),
        }
    }
    case
}

fn compile<O: FstOutputs>(
    case: &Case,
    value: impl Fn(&str) -> O::Value,
) -> (Option<CompiledFst<O::Value>>, (i64, i64)) {
    let mut c: FstCompiler<O> = FstCompilerBuilder::new(case.input_type)
        .suffix_ram_limit_mb(case.ram_mb)
        .unwrap()
        .allow_fixed_length_arcs(case.allow_fixed)
        .direct_addressing_max_oversizing_factor(case.da_factor)
        .version(case.version)
        .unwrap()
        .build();
    for (key, v) in &case.adds {
        c.add(key, value(v)).unwrap();
    }
    let fst = c.compile();
    let s = c.stats();
    (fst, (s.node_count, s.arc_count))
}

fn check<O: FstOutputs>(case: &Case, value: impl Fn(&str) -> O::Value) -> Option<Vec<u8>> {
    let (fst, stats) = compile::<O>(case, value);
    let saved = fst.map(|f| f.save::<O>());
    assert_eq!(stats, case.stats, "{}: node/arc counts", case.name);
    match (&saved, &case.fst) {
        (Some(got), Some(want)) => {
            if got != want {
                let at = got.iter().zip(want).position(|(a, b)| a != b);
                panic!(
                    "{}: saved FST differs (len {} vs Java {}, first difference at {at:?})",
                    case.name,
                    got.len(),
                    want.len()
                );
            }
        }
        (None, None) => {}
        _ => panic!("{}: one side produced no FST", case.name),
    }
    saved
}

#[test]
fn fst_compiler_matches_lucene_byte_for_byte() {
    let mut names: Vec<_> = std::fs::read_dir(dir())
        .expect("run scripts/gen-fixtures.sh --only GenFstCompiler")
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    names.sort();
    let mut checked = 0;
    let mut read_back = 0;
    for name in &names {
        let text = std::fs::read_to_string(dir().join(name)).unwrap();
        let case = parse(name, &text);
        match case.outputs.as_str() {
            "bytes" => {
                let saved = check::<ByteSequenceOutputs>(&case, unhex);
                // The crate's reader over the compiler's bytes returns every
                // key's output.
                if let Some(saved) = saved {
                    let fst = Fst::read(&mut SliceInput::new(&saved)).unwrap();
                    for (key, v) in &case.adds {
                        let key: Vec<u8> = key.iter().map(|&l| l as u8).collect();
                        assert_eq!(fst.get(&key).unwrap(), Some(unhex(v)), "{name}");
                        read_back += 1;
                    }
                }
            }
            "long" => {
                check::<PositiveIntOutputs>(&case, |s| s.parse().unwrap());
            }
            "ints" => {
                check::<IntSequenceOutputs>(&case, |s| {
                    csv(s).into_iter().map(|v| v as i32).collect()
                });
            }
            "chars" => {
                check::<CharSequenceOutputs>(&case, |s| {
                    csv(s).into_iter().map(|v| v as u16).collect()
                });
            }
            "none" => {
                check::<NoOutputs>(&case, |_| ());
            }
            "pair" => {
                check::<PairOutputs<PositiveIntOutputs, ByteSequenceOutputs>>(&case, |s| {
                    let (a, b) = s.split_once('|').unwrap();
                    Pair {
                        first: a.parse().unwrap(),
                        second: unhex(b),
                    }
                });
            }
            other => panic!("{name}: outputs {other}"),
        }
        checked += 1;
    }
    assert!(checked >= 20, "only {checked} cases");
    assert!(read_back >= 2000, "read back {read_back}");
}
