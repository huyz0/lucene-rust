//! M11 T11.6: the compound package against
//! `fixtures/src/GenAnalysisCompound.java`: a toy hyphenation grammar's
//! points for words (`points.words`), and the dictionary and hyphenation
//! decompounders with their options over
//! `fixtures/corpus/analysis-compound.txt`, row for row.

mod support;

use std::sync::{Arc, LazyLock};

use lucene_analysis::compound::{
    CompoundSizes, DictionaryCompoundWordTokenFilter, HyphenationCompoundWordTokenFilter,
    HyphenationTree,
};
use lucene_analysis::util::WhitespaceTokenizer;
use lucene_analysis::{Analyzer, CharArraySet, LowerCaseFilter, StandardTokenizer};
use support::{chain, comps, unesc};

const DICT: &[&str] = &[
    "rind", "fleisch", "donau", "dampf", "schiff", "fahrt", "kapitän", "fuss", "ball", "pumpe",
    "haus", "tür", "schloss", "garten", "zaun", "bahn", "hof", "kinder", "wagen", "see", "sonne",
    "blume", "brot", "kasten", "wurst", "ufer", "kaffee", "abend", "essen", "weiß", "kühl",
    "schrank", "bahnhof",
];

static TREE: LazyLock<Arc<HyphenationTree>> = LazyLock::new(|| {
    let xml = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/corpus/hyphenation-test.xml"
    ))
    .unwrap();
    Arc::new(HyphenationTree::from_xml(&xml).unwrap())
});

fn dict() -> Arc<CharArraySet> {
    Arc::new(CharArraySet::from_words(DICT, true))
}

fn sizes(min_word: usize, min_sub: usize, max_sub: usize, longest: bool) -> CompoundSizes {
    CompoundSizes {
        min_word_size: min_word,
        min_subword_size: min_sub,
        max_subword_size: max_sub,
        only_longest_match: longest,
    }
}

fn build(name: &str) -> Option<Analyzer> {
    let ws_lower = || LowerCaseFilter::new(WhitespaceTokenizer::new());
    let tree = || Arc::clone(&TREE);
    let d = sizes(5, 2, 15, false);
    Some(match name {
        "ws_lower_dict" => chain(move || {
            comps(DictionaryCompoundWordTokenFilter::dictionary(
                ws_lower(),
                dict(),
            ))
        }),
        "ws_dict_cased" => chain(|| {
            comps(DictionaryCompoundWordTokenFilter::dictionary(
                WhitespaceTokenizer::new(),
                dict(),
            ))
        }),
        "ws_lower_dict_longest" => chain(move || {
            comps(DictionaryCompoundWordTokenFilter::dictionary_with(
                ws_lower(),
                dict(),
                sizes(5, 2, 15, true),
                false,
            ))
        }),
        "ws_lower_dict_no_subwords" => chain(move || {
            comps(DictionaryCompoundWordTokenFilter::dictionary_with(
                ws_lower(),
                dict(),
                d,
                true,
            ))
        }),
        "ws_lower_dict_sizes" => chain(move || {
            comps(DictionaryCompoundWordTokenFilter::dictionary_with(
                ws_lower(),
                dict(),
                sizes(8, 3, 5, false),
                false,
            ))
        }),
        "std_lower_dict" => chain(|| {
            comps(DictionaryCompoundWordTokenFilter::dictionary(
                LowerCaseFilter::new(StandardTokenizer::new()),
                dict(),
            ))
        }),
        "ws_lower_hyph" => chain(move || {
            comps(HyphenationCompoundWordTokenFilter::hyphenation(
                ws_lower(),
                tree(),
                None,
                d,
                false,
                false,
            ))
        }),
        "ws_hyph_cased" => chain(move || {
            comps(HyphenationCompoundWordTokenFilter::hyphenation(
                WhitespaceTokenizer::new(),
                tree(),
                None,
                d,
                false,
                false,
            ))
        }),
        "ws_lower_hyph_dict" => chain(move || {
            comps(HyphenationCompoundWordTokenFilter::hyphenation(
                ws_lower(),
                tree(),
                Some(dict()),
                d,
                false,
                false,
            ))
        }),
        "ws_lower_hyph_dict_longest" => chain(move || {
            comps(HyphenationCompoundWordTokenFilter::hyphenation(
                ws_lower(),
                tree(),
                Some(dict()),
                sizes(5, 2, 15, true),
                false,
                false,
            ))
        }),
        "ws_lower_hyph_dict_nosub" => chain(move || {
            comps(HyphenationCompoundWordTokenFilter::hyphenation(
                ws_lower(),
                tree(),
                Some(dict()),
                d,
                true,
                false,
            ))
        }),
        "ws_lower_hyph_dict_nooverlap" => chain(move || {
            comps(HyphenationCompoundWordTokenFilter::hyphenation(
                ws_lower(),
                tree(),
                Some(dict()),
                d,
                false,
                true,
            ))
        }),
        "ws_lower_hyph_sizes" => chain(move || {
            comps(HyphenationCompoundWordTokenFilter::hyphenation(
                ws_lower(),
                tree(),
                None,
                sizes(6, 3, 6, false),
                false,
                false,
            ))
        }),
        _ => return None,
    })
}

#[test]
fn hyphenation_points_match_lucene() {
    let text =
        std::fs::read_to_string(support::data_dir("analysis_compound") + "points.words").unwrap();
    let mut n = 0;
    for row in text.lines() {
        let f: Vec<&str> = row.split('\t').collect();
        let (remain, push, word) = (f[0].parse().unwrap(), f[1].parse().unwrap(), unesc(f[2]));
        let units: Vec<u16> = word.encode_utf16().collect();
        let got = match TREE.hyphenate(&units, remain, push) {
            None => "-".to_string(),
            Some(h) => h
                .hyphenation_points()
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(","),
        };
        assert_eq!(got, f[3], "{remain} {push} {word}");
        n += 1;
    }
    assert!(n > 150, "{n}");
}

#[test]
fn decompounders_match_lucene_token_for_token() {
    let mut names = support::fixture_names("analysis_compound");
    names.retain(|n| n != "points");
    let checked =
        support::check_chains("analysis_compound", "analysis-compound.txt", &names, build);
    assert_eq!(checked, 13);
}

static RANDOM: LazyLock<Arc<HyphenationTree>> = LazyLock::new(|| {
    let xml =
        std::fs::read_to_string(support::data_dir("analysis_compound") + "random.xml").unwrap();
    Arc::new(HyphenationTree::from_xml(&xml).unwrap())
});

/// The random grammar's points: 3000 patterns, at the scale where Java's
/// `TernaryTree`s grow and balance.
#[test]
fn random_grammar_points_match_lucene() {
    let text =
        std::fs::read_to_string(support::data_dir("analysis_compound") + "random_points.words")
            .unwrap();
    let mut n = 0;
    for row in text.lines() {
        let f: Vec<&str> = row.split('\t').collect();
        let (remain, push, word) = (f[0].parse().unwrap(), f[1].parse().unwrap(), unesc(f[2]));
        let units: Vec<u16> = word.encode_utf16().collect();
        let got = match RANDOM.hyphenate(&units, remain, push) {
            None => "-".to_string(),
            Some(h) => h
                .hyphenation_points()
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(","),
        };
        assert_eq!(got, f[3], "{remain} {push} {word:?}");
        n += 1;
    }
    assert_eq!(n, 4000);
}

/// 32 random configurations of both decompounders over the random grammar
/// and a random dictionary.
#[test]
fn random_decompounder_configurations_match_lucene() {
    let dir = support::data_dir("analysis_compound");
    let lines: Vec<String> = std::fs::read_to_string(dir.clone() + "random_lines.txt")
        .unwrap()
        .lines()
        .map(unesc)
        .collect();
    let (mut words, mut configs) = (Vec::new(), Vec::new());
    for row in std::fs::read_to_string(dir + "random_configs.txt")
        .unwrap()
        .lines()
    {
        let f: Vec<String> = row.split('\t').map(str::to_string).collect();
        match f[0].as_str() {
            "D" => words.push(f[1].clone()),
            _ => configs.push(f[1..].to_vec()),
        }
    }
    let dict = Arc::new(CharArraySet::from_words(&words, true));
    let expected = support::prefixed_rows("analysis_compound", "random_chains.rows", 1);
    assert_eq!(expected.len(), configs.len());
    for (c, (id, rows)) in configs.iter().zip(&expected) {
        assert_eq!(&c[0], id);
        let n = |i: usize| c[i].parse::<usize>().unwrap();
        let b = |i: usize| c[i] == "true";
        let s = sizes(n(1), n(2), n(3), b(4));
        let (no_sub, no_overlap, use_dict) = (b(5), b(6), b(7));
        let (d, t) = (Arc::clone(&dict), Arc::clone(&RANDOM));
        let a = if b(8) {
            chain(move || {
                comps(HyphenationCompoundWordTokenFilter::hyphenation(
                    WhitespaceTokenizer::new(),
                    Arc::clone(&t),
                    use_dict.then(|| Arc::clone(&d)),
                    s,
                    no_sub,
                    no_overlap,
                ))
            })
        } else {
            chain(move || {
                comps(DictionaryCompoundWordTokenFilter::dictionary_with(
                    WhitespaceTokenizer::new(),
                    Arc::clone(&d),
                    s,
                    no_sub,
                ))
            })
        };
        support::check_rows(&format!("config {c:?}"), &a, &lines, rows);
    }
    assert_eq!(configs.len(), 32);
}
