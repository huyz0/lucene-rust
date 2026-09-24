//! Differential test for the regexp term-dictionary walk against real
//! Lucene's `IntersectTermsEnum`.
//!
//! `fixtures/src/GenRegexpIntersect.java` indexes a 65 000-term dictionary
//! whose block-tree layout (written by Lucene itself) has floor blocks,
//! nested sub-blocks, multi-byte UTF-8 and ill-formed binary terms, then
//! records, per pattern, what `RegexpQuery.getCompiled().getTermsEnum(terms)`
//! enumerates: how many terms, an order-sensitive hash of them, their summed
//! `docFreq`, and the first and last. This port's own terms writer emits one
//! leaf block per field, so this is the only test whose dictionary has the
//! structure the lockstep walk's floor skip and sub-block descent navigate.
// Test-support code opts out of the arithmetic gate at the file boundary:
// see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::blocktree;
use lucene_codecs::field_infos;
use lucene_codecs::regexp::RegexpPattern;

fn dir() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/regexp_intersect_index/"
    )
    .to_string()
}

fn manifest() -> std::collections::HashMap<String, String> {
    std::fs::read_to_string(format!("{}manifest.properties", dir()))
        .expect("run fixtures generator first (GenRegexpIntersect)")
        .lines()
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

/// As `GenRegexpIntersect` writes a term: lowercase hex, `EMPTY` for the
/// empty term.
fn hex(bytes: &[u8]) -> String {
    if bytes.is_empty() {
        return "EMPTY".to_string();
    }
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn fnv(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

#[test]
fn regexp_walk_matches_real_lucene_intersect_terms_enum() {
    let m = manifest();
    let read = |key: &str| std::fs::read(format!("{}{}", dir(), m[key])).unwrap();
    let id_hex = &m["id_hex"];
    let mut id = [0u8; 16];
    for (i, slot) in id.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&id_hex[i * 2..i * 2 + 2], 16).unwrap();
    }
    let tim_name = &m["tim_file_name"];
    // `_0_Lucene104_0.tim` -> `Lucene104_0`.
    let suffix = tim_name
        .trim_end_matches(".tim")
        .splitn(3, '_')
        .nth(2)
        .unwrap()
        .to_string();
    let max_doc: i32 = m["max_doc"].parse().unwrap();
    let field_infos = field_infos::parse(&read("fnm_file_name"), &id, "").expect("parse .fnm");
    let fields = blocktree::open(
        &read("tim_file_name"),
        &read("tip_file_name"),
        &read("tmd_file_name"),
        &field_infos,
        &id,
        &suffix,
        max_doc,
    )
    .expect("open blocktree");
    let field = fields.field("body").expect("body");

    let cases = std::fs::read_to_string(format!("{}cases.tsv", dir())).unwrap();
    let mut failures = Vec::new();
    let mut checked = 0;
    // Twice over: dictionary-wide walks in the first pass make the field
    // build its term n-gram index, so the second pass answers every
    // pattern that forces literal text from the index instead -- both paths
    // are held to the same ground truth.
    for line in cases.lines().chain(cases.lines()) {
        let cols: Vec<&str> = line.split('\t').collect();
        let (pattern, count, hash, df_sum, first, last) =
            (cols[0], cols[1], cols[2], cols[3], cols[4], cols[5]);
        let p = RegexpPattern::parse(pattern).expect("parse");
        for (walk, lazy) in [("eager", false), ("lazy", true)] {
            // The lazy automaton has no totality check (it is never built for
            // a pattern that small), so `ALL` patterns are the eager walk's.
            if lazy && (pattern == ".*" || pattern == "@") {
                continue;
            }
            let terms: Box<dyn Iterator<Item = _>> = if lazy {
                Box::new(
                    field
                        .regexp_intersect_states_lazy(
                            &p,
                            lucene_codecs::automaton::MAX_LAZY_SET_ENTRIES,
                        )
                        .expect("every fixture pattern builds an NFA"),
                )
            } else {
                Box::new(field.regexp_intersect_states(&p))
            };
            let mut h = 0xcbf29ce484222325u64;
            let (mut n, mut dfs) = (0u64, 0i64);
            let (mut got_first, mut got_last) = ("-".to_string(), "-".to_string());
            for r in terms {
                let (term, seeked) = r.expect("walk");
                h = fnv(h, &(term.len() as u32).to_le_bytes());
                h = fnv(h, &term);
                if n == 0 {
                    got_first = hex(&term);
                }
                got_last = hex(&term);
                n += 1;
                dfs += i64::from(seeked.stats.doc_freq);
            }
            let got = format!("{n}\t{h:x}\t{dfs}\t{got_first}\t{got_last}");
            let want = format!("{count}\t{hash}\t{df_sum}\t{first}\t{last}");
            if got != want {
                failures.push(format!("{walk} {pattern:?}: got {got}, Lucene {want}"));
            }
        }
        checked += 1;
    }
    assert!(checked > 30, "fixture looks truncated: {checked}");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The fixture's segment opened from the given `.tip`/`.tim` bytes.
fn open_with(tip: &[u8], tim: &[u8]) -> Result<blocktree::BlockTreeFields, blocktree::Error> {
    let m = manifest();
    let read = |key: &str| std::fs::read(format!("{}{}", dir(), m[key])).unwrap();
    let mut id = [0u8; 16];
    for (i, slot) in id.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&m["id_hex"][i * 2..i * 2 + 2], 16).unwrap();
    }
    let suffix = m["tim_file_name"]
        .trim_end_matches(".tim")
        .splitn(3, '_')
        .nth(2)
        .unwrap()
        .to_string();
    let field_infos = field_infos::parse(&read("fnm_file_name"), &id, "").expect("parse .fnm");
    blocktree::open(
        tim,
        tip,
        &read("tmd_file_name"),
        &field_infos,
        &id,
        &suffix,
        m["max_doc"].parse().unwrap(),
    )
}

/// A lazily determinized walk that runs out of its state budget ends with an
/// error -- `TooComplexToDeterminizeException`'s place -- not a wrong answer.
#[test]
fn a_lazy_walk_out_of_budget_fails_the_query() {
    let m = manifest();
    let read = |key: &str| std::fs::read(format!("{}{}", dir(), m[key])).unwrap();
    let fields = open_with(&read("tip_file_name"), &read("tim_file_name")).unwrap();
    let field = fields.field("body").unwrap();
    let p = RegexpPattern::parse(".*z").unwrap();
    let last = field
        .regexp_intersect_states_lazy(&p, 8)
        .unwrap()
        .last()
        .expect("the walk yields its error");
    let err = last.expect_err("an 8-entry budget cannot cover `.*z`");
    assert!(format!("{err}").contains("too complex"), "{err}");
}

/// Single-byte corruptions of the term index and the terms file, sampled
/// across both: every walk ends -- with terms or with an error -- and none
/// panics. A corrupt index can point the lockstep walk's floor skips and
/// sub-block descents anywhere; before the walk refused to load a block twice,
/// `check_index`'s equivalent sweep spun for ten hours on one of these.
#[test]
fn every_walk_over_a_corrupted_dictionary_ends() {
    let m = manifest();
    let read = |key: &str| std::fs::read(format!("{}{}", dir(), m[key])).unwrap();
    let (tip, tim) = (read("tip_file_name"), read("tim_file_name"));
    let patterns: Vec<RegexpPattern> = [".*z", "t1[0-9]", "[a-z][0-9]{2}", ".*1.*2.*"]
        .iter()
        .map(|p| RegexpPattern::parse(p).unwrap())
        .collect();
    let mut walked = 0;
    for (which, original) in [("tip", &tip), ("tim", &tim)] {
        // The body, not the header or footer, which `open` itself checks.
        let body = 40..original.len().saturating_sub(16);
        let step = (body.len() / 150).max(1);
        for off in body.step_by(step) {
            for mask in [0x01u8, 0xFF] {
                let mut bytes = original.clone();
                bytes[off] ^= mask;
                let fields = if which == "tip" {
                    open_with(&bytes, &tim)
                } else {
                    open_with(&tip, &bytes)
                };
                let Ok(fields) = fields else { continue };
                let Some(field) = fields.field("body") else {
                    continue;
                };
                for p in &patterns {
                    // Ends, one way or the other; the count is irrelevant.
                    for r in field.regexp_intersect_states(p) {
                        if r.is_err() {
                            break;
                        }
                    }
                    walked += 1;
                }
            }
        }
    }
    assert!(
        walked > 500,
        "the sweep walked too few corrupted dictionaries: {walked}"
    );
}
