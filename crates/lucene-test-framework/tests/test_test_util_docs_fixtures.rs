//! Differential test against `fixtures/src/testframework/GenTestFrameworkTestUtilDocs.java`:
//! [`docs`] asks the terms enum for the postings features Lucene 10.5.0's
//! `TestUtil.docs` does for 200 seeds and three caller flags, drawing as
//! much; [`string_to_char_sequence`] draws its one `nextInt(5)`; and
//! [`bytes_ref_to_string`] prints valid UTF-8 as Java does.
//!
//! Regenerate with `scripts/gen-test-framework-fixtures.sh --only TestFrameworkTestUtilDocs`.

use std::fs;
use std::path::PathBuf;

use lucene_codecs::blocktree::SeekStatus;
use lucene_search::reader::{PostingsEnum, PostingsFlags, TermsEnum};
use lucene_test_framework::util::test_util::{bytes_ref_to_string, docs, string_to_char_sequence};
use lucene_util::java_random::JavaRandom;

/// Refuses every postings request, naming the flags it was asked for.
struct Recording;

impl TermsEnum for Recording {
    fn next(&mut self) -> lucene_search::Result<Option<&[u8]>> {
        Ok(None)
    }
    fn term(&self) -> Option<&[u8]> {
        None
    }
    fn try_seek_ceil(&mut self, _: &[u8]) -> lucene_search::Result<SeekStatus> {
        Ok(SeekStatus::End)
    }
    fn doc_freq(&mut self) -> lucene_search::Result<i32> {
        Ok(0)
    }
    fn total_term_freq(&mut self) -> lucene_search::Result<i64> {
        Ok(0)
    }
    fn postings(&mut self, flags: PostingsFlags) -> lucene_search::Result<Box<dyn PostingsEnum>> {
        Err(lucene_search::Error::Unsupported(format!("{flags:?}")))
    }
}

fn flags(name: &str) -> PostingsFlags {
    match name {
        "None" => PostingsFlags::None,
        "Freqs" => PostingsFlags::Freqs,
        "All" => PostingsFlags::All,
        other => panic!("caller flags {other}"),
    }
}

fn unhex(s: &str) -> Vec<u8> {
    let s = s.strip_prefix('-').unwrap();
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

#[test]
fn test_util_helpers_match_lucene() {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/testframework/data/test_util_docs.txt");
    let fixture = fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    let mut counts = [0; 3];
    for line in fixture.lines() {
        let (kind, rest) = line.split_once(' ').unwrap();
        match kind {
            "docs" => {
                let parts: Vec<&str> = rest.split(' ').collect();
                let seed: i64 = parts[0].parse().unwrap();
                let mut r = JavaRandom::new(seed);
                let Err(lucene_search::Error::Unsupported(asked)) =
                    docs(&mut r, &mut Recording, flags(parts[1]))
                else {
                    panic!("the stub refuses");
                };
                assert_eq!(asked, parts[2], "{line}");
                assert_eq!(r.next_int().to_string(), parts[3], "{line}: draws");
                counts[0] += 1;
            }
            "chars" => {
                let (seed, next) = rest.split_once(' ').unwrap();
                let mut r = JavaRandom::new(seed.parse().unwrap());
                assert_eq!(string_to_char_sequence("h\u{e9}llo", &mut r), "h\u{e9}llo");
                assert_eq!(r.next_int().to_string(), next, "{line}");
                counts[1] += 1;
            }
            "bytes" => {
                let (hex, expected) = rest.split_once(' ').unwrap();
                assert_eq!(bytes_ref_to_string(Some(&unhex(hex))), expected, "{line}");
                counts[2] += 1;
            }
            other => panic!("unknown fixture line {other}"),
        }
    }
    assert_eq!(counts, [600, 20, 3]);
}
