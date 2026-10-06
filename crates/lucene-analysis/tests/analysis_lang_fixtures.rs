//! M11 T11.6: the per-language packages against
//! `fixtures/src/GenAnalysisLanguages.java`: every language analyzer (and a
//! few filter chains) over `fixtures/corpus/analysis-lang.txt` row for row,
//! and `lang.words` -- each stemmer and normalizer over words built from its
//! own suffixes, prefixes and characters -- word for word.

mod support;

use std::collections::BTreeMap;

use lucene_analysis::lang::*;
use lucene_analysis::token_stream::TokenStream;
use lucene_analysis::util::WhitespaceTokenizer;
use lucene_analysis::{Analyzer, CharArraySet, KeywordTokenizer};
use support::{chain, chain_cf, comps, esc, unesc};

fn set(words: &[&str]) -> CharArraySet {
    CharArraySet::from_words(words, false)
}

fn build(name: &str) -> Option<Analyzer> {
    Some(match name {
        "arabic" => Analyzer::new(ar::ArabicAnalyzer::default()),
        "arabic_excl" => Analyzer::new(ar::ArabicAnalyzer::with_exclusions(
            &ar::DEFAULT_STOP_SET,
            &set(&["الأطفال"]),
        )),
        "armenian" => Analyzer::new(hy::ArmenianAnalyzer::default()),
        "basque" => Analyzer::new(eu::BasqueAnalyzer::default()),
        "bengali" => Analyzer::new(bn::BengaliAnalyzer::default()),
        "brazilian" => Analyzer::new(br::BrazilianAnalyzer::default()),
        "bulgarian" => Analyzer::new(bg::BulgarianAnalyzer::default()),
        "catalan" => Analyzer::new(ca::CatalanAnalyzer::default()),
        "czech" => Analyzer::new(cz::CzechAnalyzer::default()),
        "danish" => Analyzer::new(da::DanishAnalyzer::default()),
        "dutch" => Analyzer::new(nl::DutchAnalyzer::default()),
        "dutch_no_dict" => Analyzer::new(nl::DutchAnalyzer::with_options(
            &nl::DEFAULT_STOP_SET,
            &set(&["kinderen"]),
            &BTreeMap::new(),
        )),
        "estonian" => Analyzer::new(et::EstonianAnalyzer::default()),
        "finnish" => Analyzer::new(fi::FinnishAnalyzer::default()),
        "french" => Analyzer::new(fr::FrenchAnalyzer::default()),
        "french_excl" => Analyzer::new(fr::FrenchAnalyzer::with_exclusions(
            &fr::DEFAULT_STOP_SET,
            &set(&["enfants", "châteaux"]),
        )),
        "galician" => Analyzer::new(gl::GalicianAnalyzer::default()),
        "german" => Analyzer::new(de::GermanAnalyzer::default()),
        "german_excl" => Analyzer::new(de::GermanAnalyzer::with_exclusions(
            &de::DEFAULT_STOP_SET,
            &set(&["häuser"]),
        )),
        "greek" => Analyzer::new(el::GreekAnalyzer::default()),
        "hindi" => Analyzer::new(hi::HindiAnalyzer::default()),
        "hungarian" => Analyzer::new(hu::HungarianAnalyzer::default()),
        "indonesian" => Analyzer::new(id::IndonesianAnalyzer::default()),
        "irish" => Analyzer::new(ga::IrishAnalyzer::default()),
        "italian" => Analyzer::new(it::ItalianAnalyzer::default()),
        "latvian" => Analyzer::new(lv::LatvianAnalyzer::default()),
        "lithuanian" => Analyzer::new(lt::LithuanianAnalyzer::default()),
        "nepali" => Analyzer::new(ne::NepaliAnalyzer::default()),
        "norwegian" => Analyzer::new(no::NorwegianAnalyzer::default()),
        "persian" => Analyzer::new(fa::PersianAnalyzer::default()),
        "portuguese" => Analyzer::new(pt::PortugueseAnalyzer::default()),
        "romanian" => Analyzer::new(ro::RomanianAnalyzer::default()),
        "russian" => Analyzer::new(ru::RussianAnalyzer::default()),
        "serbian" => Analyzer::new(sr::SerbianAnalyzer::default()),
        "sorani" => Analyzer::new(ckb::SoraniAnalyzer::default()),
        "spanish" => Analyzer::new(es::SpanishAnalyzer::default()),
        "swedish" => Analyzer::new(sv::SwedishAnalyzer::default()),
        "swedish_excl" => Analyzer::new(sv::SwedishAnalyzer::with_exclusions(
            &sv::DEFAULT_STOP_SET,
            &set(&["barnen"]),
        )),
        "tamil" => Analyzer::new(ta::TamilAnalyzer::default()),
        "telugu" => Analyzer::new(te::TeluguAnalyzer::default()),
        "turkish" => Analyzer::new(tr::TurkishAnalyzer::default()),
        "ws_turkish_lowercase_apostrophe" => chain(|| {
            comps(tr::TurkishLowerCaseFilter::new(tr::ApostropheFilter::new(
                WhitespaceTokenizer::new(),
            )))
        }),
        "ws_irish_lowercase" => {
            chain(|| comps(ga::IrishLowerCaseFilter::new(WhitespaceTokenizer::new())))
        }
        "ws_german_normalization_minimal" => chain(|| {
            comps(de::GermanMinimalStemFilter::new(
                de::GermanNormalizationFilter::new(WhitespaceTokenizer::new()),
            ))
        }),
        "persian_char_filter" => chain_cf(
            |r| Box::new(fa::PersianCharFilter::new(r)),
            || comps(WhitespaceTokenizer::new()),
        ),
        _ => return None,
    })
}

/// The generator's `filters()`, each over a `KeywordTokenizer`.
fn filter(name: &str, k: KeywordTokenizer) -> Option<Box<dyn TokenStream>> {
    use lucene_analysis::lang::no::{NorwegianLightStemmer, NorwegianMinimalStemmer};
    Some(match name {
        "GermanLight" => Box::new(de::GermanLightStemFilter::new(k)),
        "GermanMinimal" => Box::new(de::GermanMinimalStemFilter::new(k)),
        "German" => Box::new(de::GermanStemFilter::new(k)),
        "GermanNormalization" => Box::new(de::GermanNormalizationFilter::new(k)),
        "FrenchLight" => Box::new(fr::FrenchLightStemFilter::new(k)),
        "FrenchMinimal" => Box::new(fr::FrenchMinimalStemFilter::new(k)),
        "SpanishLight" => Box::new(es::SpanishLightStemFilter::new(k)),
        "SpanishMinimal" => Box::new(es::SpanishMinimalStemFilter::new(k)),
        "SpanishPlural" => Box::new(es::SpanishPluralStemFilter::new(k)),
        "ItalianLight" => Box::new(it::ItalianLightStemFilter::new(k)),
        "PortugueseLight" => Box::new(pt::PortugueseLightStemFilter::new(k)),
        "PortugueseMinimal" => Box::new(pt::PortugueseMinimalStemFilter::new(k)),
        "Portuguese" => Box::new(pt::PortugueseStemFilter::new(k)),
        "Galician" => Box::new(gl::GalicianStemFilter::new(k)),
        "GalicianMinimal" => Box::new(gl::GalicianMinimalStemFilter::new(k)),
        "SwedishLight" => Box::new(sv::SwedishLightStemFilter::new(k)),
        "SwedishMinimal" => Box::new(sv::SwedishMinimalStemFilter::new(k)),
        "NorwegianLight" => Box::new(no::NorwegianLightStemFilter::new(k)),
        "NorwegianLightNynorsk" => Box::new(no::NorwegianLightStemFilter::with_stemmer(
            k,
            NorwegianLightStemmer::new(no::NYNORSK).unwrap(),
        )),
        "NorwegianLightBoth" => Box::new(no::NorwegianLightStemFilter::with_stemmer(
            k,
            NorwegianLightStemmer::new(no::BOKMAAL | no::NYNORSK).unwrap(),
        )),
        "NorwegianMinimal" => Box::new(no::NorwegianMinimalStemFilter::new(k)),
        "NorwegianMinimalNynorsk" => Box::new(no::NorwegianMinimalStemFilter::with_stemmer(
            k,
            NorwegianMinimalStemmer::new(no::NYNORSK).unwrap(),
        )),
        "NorwegianNormalization" => Box::new(no::NorwegianNormalizationFilter::new(k)),
        "FinnishLight" => Box::new(fi::FinnishLightStemFilter::new(k)),
        "HungarianLight" => Box::new(hu::HungarianLightStemFilter::new(k)),
        "RussianLight" => Box::new(ru::RussianLightStemFilter::new(k)),
        "Bulgarian" => Box::new(bg::BulgarianStemFilter::new(k)),
        "Czech" => Box::new(cz::CzechStemFilter::new(k)),
        "Latvian" => Box::new(lv::LatvianStemFilter::new(k)),
        "Indonesian" => Box::new(id::IndonesianStemFilter::new(k)),
        "IndonesianInflectional" => Box::new(id::IndonesianStemFilter::with_stemmer(
            k,
            id::IndonesianStemmer {
                stem_derivational: false,
            },
        )),
        "Hindi" => Box::new(hi::HindiStemFilter::new(k)),
        "HindiNormalization" => Box::new(hi::HindiNormalizationFilter::new(k)),
        "Bengali" => Box::new(bn::BengaliStemFilter::new(k)),
        "BengaliNormalization" => Box::new(bn::BengaliNormalizationFilter::new(k)),
        "Telugu" => Box::new(te::TeluguStemFilter::new(k)),
        "TeluguNormalization" => Box::new(te::TeluguNormalizationFilter::new(k)),
        "Sorani" => Box::new(ckb::SoraniStemFilter::new(k)),
        "SoraniNormalization" => Box::new(ckb::SoraniNormalizationFilter::new(k)),
        "Arabic" => Box::new(ar::ArabicStemFilter::new(k)),
        "ArabicNormalization" => Box::new(ar::ArabicNormalizationFilter::new(k)),
        "Persian" => Box::new(fa::PersianStemFilter::new(k)),
        "PersianNormalization" => Box::new(fa::PersianNormalizationFilter::new(k)),
        "IndicNormalization" => Box::new(in_::IndicNormalizationFilter::new(k)),
        "RomanianNormalization" => Box::new(ro::RomanianNormalizationFilter::new(k)),
        "IrishLowerCase" => Box::new(ga::IrishLowerCaseFilter::new(k)),
        "TurkishLowerCase" => Box::new(tr::TurkishLowerCaseFilter::new(k)),
        "Apostrophe" => Box::new(tr::ApostropheFilter::new(k)),
        "Greek" => Box::new(el::GreekStemFilter::new(k)),
        "GreekLowerCase" => Box::new(el::GreekLowerCaseFilter::new(k)),
        "Brazilian" => Box::new(br::BrazilianStemFilter::new(k)),
        "SerbianNormalization" => Box::new(sr::SerbianNormalizationFilter::new(k)),
        "SerbianNormalizationRegular" => Box::new(sr::SerbianNormalizationRegularFilter::new(k)),
        _ => return None,
    })
}

#[test]
fn language_analyzers_match_lucene_token_for_token() {
    let mut names = support::fixture_names("analysis_lang");
    names.retain(|n| n != "lang");
    let checked = support::check_chains("analysis_lang", "analysis-lang.txt", &names, build);
    assert!(checked >= 40, "{checked}");
}

#[test]
fn stemmers_and_normalizers_match_lucene_word_for_word() {
    let text = std::fs::read_to_string(support::data_dir("analysis_lang") + "lang.words").unwrap();
    let mut filters: BTreeMap<String, Box<dyn TokenStream>> = BTreeMap::new();
    let mut n = 0;
    for row in text.lines() {
        let f: Vec<&str> = row.split('\t').collect();
        let (name, word, expected) = (f[0], unesc(f[1]), f[2]);
        let ts = filters.entry(name.to_string()).or_insert_with(|| {
            filter(name, KeywordTokenizer::new()).unwrap_or_else(|| panic!("no filter {name}"))
        });
        let tok = ts
            .as_tokenizer()
            .expect("a keyword tokenizer at the source");
        tok.set_reader(Box::new(lucene_analysis::reader::StrReader::new(&word)))
            .unwrap();
        ts.reset().unwrap();
        let got = if ts.increment_token().unwrap() {
            esc(ts.attributes().term())
        } else {
            "<none>".to_string()
        };
        ts.end().unwrap();
        ts.close().unwrap();
        assert_eq!(
            support::normalise_expected(expected),
            got,
            "{name}({})",
            esc(&word)
        );
        n += 1;
    }
    assert!(n > 15_000, "{n} words");
}

#[test]
fn analyzers_normalize_like_lucene() {
    let text =
        std::fs::read_to_string(support::data_dir("analysis_lang") + "normalize.words").unwrap();
    let mut n = 0;
    for row in text.lines() {
        let f: Vec<&str> = row.split('\t').collect();
        let a = build(f[0]).unwrap_or_else(|| panic!("no chain {}", f[0]));
        let text = unesc(f[1]);
        let got = a.normalize("f", &text).unwrap();
        assert_eq!(support::hex(Some(&got)), f[2], "{}({text})", f[0]);
        n += 1;
    }
    assert!(n > 400, "{n}");
}
