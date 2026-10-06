//! M11 T11.3, opt-in: every Snowball stemmer over the Snowball project's
//! own test vocabularies.
//!
//! `snowball-data` (github.com/snowballstem/snowball-data) holds, per
//! language, `voc.txt` and the stems the reference implementation gives
//! (`output.txt`). Lucene 10.5.0 pins no data commit (its `snowball.gradle`
//! pins only the compiler, `34f3612e`); this uses `f08c4d63`, the last data
//! commit before that compiler commit, where Lucene's 30 stemmers give
//! `output.txt` word for word (9.2 million Arabic words, 90,729 Greek, ...).
//! The vocabularies are not redistributed (Greek's and French's are partly
//! CC BY-SA, Arabic's GPL-3.0): `scripts/check-snowball-vocabulary.sh`
//! fetches them pinned by SHA-256 and runs this with `SNOWBALL_DATA` naming
//! the directory. Without it this test checks nothing and says so.

use lucene_analysis::snowball::SnowballStemmer;

#[test]
fn every_stemmer_reproduces_snowball_data() {
    let Ok(dir) = std::env::var("SNOWBALL_DATA") else {
        eprintln!("SNOWBALL_DATA is not set: skipped (scripts/check-snowball-vocabulary.sh)");
        return;
    };
    let mut total = 0usize;
    let mut failures = Vec::new();
    for name in SnowballStemmer::names() {
        let lang = name.to_lowercase();
        let read = |file: &str| {
            let path = format!("{dir}/{lang}/{file}");
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"))
        };
        let (voc, output) = (read("voc.txt"), read("output.txt"));
        let (voc, output): (Vec<&str>, Vec<&str>) =
            (voc.lines().collect(), output.lines().collect());
        assert_eq!(
            voc.len(),
            output.len(),
            "{lang}: voc.txt and output.txt differ in length"
        );
        let mut s = SnowballStemmer::for_name(name).expect("a shipped stemmer");
        let mut wrong = 0usize;
        let mut first = Vec::new();
        for (word, want) in voc.iter().zip(&output) {
            s.set_current(word);
            s.stem();
            let got = s.current();
            if got != *want {
                wrong += 1;
                if first.len() < 5 {
                    first.push(format!("{word} -> {got} (snowball-data: {want})"));
                }
            }
        }
        eprintln!("{name}: {} words, {wrong} differ", voc.len());
        total += voc.len();
        if wrong > 0 {
            failures.push(format!(
                "{name}: {wrong} of {} differ, first {first:?}",
                voc.len()
            ));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
    assert!(total > 10_000_000, "{total} words");
}
