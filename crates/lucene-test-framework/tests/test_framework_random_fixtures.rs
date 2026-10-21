//! Differential test against `fixtures/src/testframework/GenTestFrameworkRandom.java`:
//! for 200 seeds, every generator in `util` draws what Lucene 10.5.0's test
//! framework draws from the same `java.util.Random` seed -- numbers,
//! strings, English, `atLeast`/`rarely`, random regexps and automata (their
//! shape and accepted strings), and `LineFileDocs` over the gzip excerpt and
//! its plain copy.
//!
//! Regenerate with `scripts/gen-test-framework-fixtures.sh --only TestFrameworkRandom`.

use std::fs;
use std::path::PathBuf;

use lucene_test_framework::util::automaton_test_util::{
    random_automaton, random_regexp, RandomAcceptedStrings,
};
use lucene_test_framework::util::english::{int_to_english, long_to_english};
use lucene_test_framework::util::line_file_docs::{default_excerpt, gunzip, LineFileDocs};
use lucene_test_framework::util::lucene_test_case::{at_least, java_string_hash, rarely, usually};
use lucene_test_framework::util::random_numbers::{
    random_from, random_int_between, random_long_between,
};
use lucene_test_framework::util::test_util::*;
use lucene_util::automaton::TransitionAccessor;
use lucene_util::java_random::JavaRandom;
use lucene_util::test_support::TempDir;

fn fixture() -> String {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/testframework/data/random.txt");
    fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

fn units_hex(units: &[u16]) -> String {
    if units.is_empty() {
        return "-".into();
    }
    units.iter().map(|u| format!("{u:04x}")).collect()
}

fn str_hex(s: &str) -> String {
    units_hex(&s.encode_utf16().collect::<Vec<_>>())
}

fn bytes_hex(b: &[u8]) -> String {
    if b.is_empty() {
        return "-".into();
    }
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// `BigInteger(buffer).toByteArray()`: the shortest two's-complement form.
fn minimal_twos_complement(mut b: &[u8]) -> &[u8] {
    while b.len() > 1 && ((b[0] == 0 && b[1] & 0x80 == 0) || (b[0] == 0xFF && b[1] & 0x80 != 0)) {
        b = &b[1..];
    }
    b
}

fn ints(a: &[i32]) -> String {
    if a.is_empty() {
        return "-".into();
    }
    a.iter().map(i32::to_string).collect::<Vec<_>>().join(",")
}

/// The lines Rust produces for one seed, in the generator's order.
fn seed_lines(seed: i64, line_docs: bool, plain: &std::path::Path) -> Vec<String> {
    let mut out = vec![format!("seed {seed}")];
    let mut line = |cat: &str, name: &str, v: String| out.push(format!("{cat} {name} {v}"));
    let base = seed * 31;

    let r = &mut JavaRandom::new(base);
    line("num", "nextInt", next_int(r, -100, 100).to_string());
    line("num", "nextIntWide", next_int(r, 0, i32::MAX).to_string());
    line("num", "nextLong", next_long(r, -1000, 1000).to_string());
    line(
        "num",
        "nextLongFull",
        next_long(r, i64::MIN, i64::MAX).to_string(),
    );
    line(
        "num",
        "nextLongBig",
        next_long(r, -5, i64::MAX / 3).to_string(),
    );
    line(
        "num",
        "randomIntBetween",
        random_int_between(r, -5, 5).to_string(),
    );
    line(
        "num",
        "randomIntBetweenFull",
        random_int_between(r, i32::MIN, i32::MAX).to_string(),
    );
    line(
        "num",
        "randomLongBetween",
        random_long_between(r, -10, 1 << 40).to_string(),
    );
    line(
        "num",
        "randomLongBetweenFull",
        random_long_between(r, i64::MIN, i64::MAX).to_string(),
    );
    line(
        "num",
        "randomLongBetweenHigh",
        random_long_between(r, i64::MAX - 3, i64::MAX).to_string(),
    );
    let big = next_big_integer(r, 20);
    line(
        "num",
        "nextBigInteger",
        bytes_hex(minimal_twos_complement(&big)),
    );
    line(
        "num",
        "randomFrom",
        random_from(r, &["a", "b", "c", "d", "e", "f", "g"]).to_string(),
    );
    line("num", "atLeast", at_least(r, 1, 10).to_string());
    for _ in 0..20 {
        line("num", "rarely", rarely(r, false, 1).to_string());
    }
    line("num", "usually", usually(r, false, 1).to_string());

    let r = &mut JavaRandom::new(base + 1);
    line("str", "simple", str_hex(&random_simple_string(r)));
    line("str", "simpleMax", str_hex(&random_simple_string_max(r, 7)));
    line(
        "str",
        "simpleBetween",
        str_hex(&random_simple_string_between(r, 3, 12)),
    );
    line(
        "str",
        "simpleRange",
        units_hex(&random_simple_string_range(
            r,
            u16::from(b'k'),
            u16::from(b'q'),
            9,
        )),
    );
    line("str", "unicode", str_hex(&random_unicode_string(r)));
    line(
        "str",
        "unicodeMax",
        str_hex(&random_unicode_string_max(r, 50)),
    );
    let mut fixed = [0u16; 20];
    random_fixed_length_unicode_string(r, &mut fixed, 0, 20);
    line("str", "fixedLength", units_hex(&fixed));
    line("str", "regexpish", str_hex(&random_regexpish_string(r)));
    line(
        "str",
        "regexpishMax",
        str_hex(&random_regexpish_string_max(r, 40)),
    );
    line("str", "htmlish", str_hex(&random_htmlish_string(r, 8)));
    line(
        "str",
        "realistic",
        str_hex(&random_realistic_unicode_string(r)),
    );
    line(
        "str",
        "realisticMax",
        str_hex(&random_realistic_unicode_string_max(r, 30)),
    );
    line(
        "str",
        "realisticBetween",
        str_hex(&random_realistic_unicode_string_between(r, 5, 30)),
    );
    line(
        "str",
        "fixedByteLength",
        str_hex(&random_fixed_byte_length_unicode_string(r, 25)),
    );
    line("str", "binaryTerm", bytes_hex(&random_binary_term(r)));
    line(
        "str",
        "binaryTermLen",
        bytes_hex(&random_binary_term_len(r, 7)),
    );
    line(
        "str",
        "analysisSimple",
        str_hex(&random_analysis_string(r, 30, true)),
    );
    line(
        "str",
        "analysis",
        str_hex(&random_analysis_string(r, 30, false)),
    );
    line(
        "str",
        "subString",
        str_hex(&random_sub_string(r, 10, false)),
    );
    line(
        "str",
        "subStringSimple",
        str_hex(&random_sub_string(r, 10, true)),
    );
    line(
        "str",
        "recase",
        str_hex(&randomly_recase_code_points(
            r,
            "Hello Wörld, ÀÉÎ ßtraße 42",
        )),
    );

    let r = &mut JavaRandom::new(base + 2);
    line("eng", "int", str_hex(&int_to_english(r.next_int())));
    line(
        "eng",
        "small",
        str_hex(&int_to_english(r.next_int_bounded(2000) - 1000)),
    );
    line("eng", "long", str_hex(&long_to_english(r.next_long())));

    let r = &mut JavaRandom::new(base + 3);
    line("aut", "regexp", str_hex(&random_regexp(r)));
    let a = random_automaton(r);
    let transitions: i32 = (0..a.get_num_states())
        .map(|s| a.get_num_transitions(s))
        .sum();
    line(
        "aut",
        "automaton",
        format!(
            "{} {transitions} {}",
            a.get_num_states(),
            a.is_deterministic()
        ),
    );
    match RandomAcceptedStrings::new(&a) {
        Err(_) => line("aut", "accepted", "ERR".into()),
        Ok(ras) => {
            for _ in 0..3 {
                match ras.get_random_accepted_string(r) {
                    Ok(s) => line("aut", "accepted", ints(&s)),
                    Err(_) => {
                        line("aut", "accepted", "ERR".into());
                        break;
                    }
                }
            }
        }
    }

    if line_docs {
        for (kind, k, path) in [
            ("gz", 4, default_excerpt()),
            ("plain", 5, plain.to_path_buf()),
        ] {
            let r = &mut JavaRandom::new(base + k);
            let mut docs = LineFileDocs::with_path(r, &path).expect("open line docs");
            for _ in 0..60 {
                let d = docs.next_doc().expect("next line doc");
                line(
                    "docs",
                    kind,
                    format!(
                        "{} {} {} {} {} {}",
                        d.id,
                        d.id,
                        d.page_views,
                        str_hex(&d.title),
                        d.date,
                        java_string_hash(&d.body)
                    ),
                );
            }
        }
    }
    out
}

/// The excerpt, gunzipped, as the generator's plain-file copy.
fn plain_copy(dir: &TempDir) -> PathBuf {
    let text = gunzip(&fs::read(default_excerpt()).unwrap()).unwrap();
    let p = dir.path().join("europarl.lines.excerpt.txt");
    fs::write(&p, text).unwrap();
    p
}

#[test]
fn generators_draw_what_lucene_draws() {
    let dir = TempDir::new("tf-random");
    let plain = plain_copy(&dir);
    let want = fixture();
    let mut want_lines = want.lines();
    let mut seeds = 0;
    for seed in 0..200 {
        for got in seed_lines(seed, seed < 40, &plain) {
            let expected = want_lines
                .next()
                .unwrap_or_else(|| panic!("fixture ends before {got:?}"));
            assert_eq!(got, expected, "seed {seed}");
        }
        seeds += 1;
    }
    assert!(
        want_lines.next().is_none(),
        "fixture has lines past seed 199"
    );
    assert_eq!(seeds, 200);
}
