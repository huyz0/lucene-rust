//! Java-produced integer-label enumeration traces, including reused-cursor seeks.
#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::fst::PositiveIntOutputs;
use lucene_codecs::fst_util::{IntsRefFstEnum, TypedFst};

fn labels(text: &str) -> Vec<i32> {
    if text == "-" {
        Vec::new()
    } else {
        text.split(',').map(|s| s.parse().unwrap()).collect()
    }
}

fn formatted(pair: Option<(&[i32], &i64)>) -> String {
    pair.map_or_else(
        || "null".into(),
        |(input, output)| {
            let key = if input.is_empty() {
                "-".into()
            } else {
                input
                    .iter()
                    .map(i32::to_string)
                    .collect::<Vec<_>>()
                    .join(",")
            };
            format!("{key}:{output}")
        },
    )
}

#[test]
fn integer_enumeration_matches_java_traces() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/data/ints_ref_fst_enum");
    for name in ["list", "binary", "direct", "continuous", "byte2"] {
        let bytes = std::fs::read(dir.join(format!("{name}.fst"))).unwrap();
        let fst = TypedFst::<PositiveIntOutputs>::read(&bytes).unwrap();
        let mut cursor = IntsRefFstEnum::new(&fst);
        assert!(cursor.current().is_none());
        let trace = std::fs::read_to_string(dir.join(format!("{name}.txt"))).unwrap();
        let mut checked = 0;
        for (line, text) in trace.lines().enumerate() {
            let columns: Vec<_> = text.split(' ').collect();
            let target = labels(columns[1]);
            let result = match columns[0] {
                "key" => continue,
                "next" => cursor.next().unwrap(),
                "current" => cursor.current(),
                "exact" => cursor.seek_exact(&target).unwrap(),
                "ceil" => cursor.seek_ceil(&target).unwrap(),
                "floor" => cursor.seek_floor(&target).unwrap(),
                other => panic!("unknown operation {other}"),
            };
            assert_eq!(formatted(result), columns[2], "{name}:{} {text}", line + 1);
            checked += 1;
        }
        assert!(checked > 10_000, "trace did not exercise enough operations");
        // A saved Java FST with a truncated body must not be accepted.
        assert!(TypedFst::<PositiveIntOutputs>::read(&bytes[..bytes.len() - 1]).is_err());
    }
}
