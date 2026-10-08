//! Unit tests of the factories' own boundaries: accessors, factories used
//! before `inform`, error mappings the configuration fixtures cannot reach.

use std::sync::Arc;

use super::args::JavaArgs;
use super::conditional::FactoryCondition;
use super::filters_resource::{java_split, wrapped_runtime};
use super::*;
use crate::reader::CharReader;
use crate::token_stream::TokenStream;
use crate::{AnalysisError, KeywordTokenizer, StrReader};

fn build<F: FactoryClass>(pairs: &[(&str, &str)]) -> Result<F, FactoryError> {
    F::from_args(&mut JavaArgs::from_pairs(pairs))
}

fn input(text: &str) -> Box<dyn TokenStream> {
    let mut t = crate::util::WhitespaceTokenizer::new();
    crate::token_stream::Tokenizer::set_reader(&mut t, Box::new(StrReader::new(text))).unwrap();
    Box::new(t)
}

fn terms(mut ts: Box<dyn TokenStream>) -> Result<Vec<String>, AnalysisError> {
    ts.reset()?;
    let mut out = Vec::new();
    while ts.increment_token()? {
        out.push(ts.attributes().term().to_string());
    }
    ts.end()?;
    ts.close()?;
    Ok(out)
}

fn loader() -> MapResourceLoader {
    MapResourceLoader::new()
        .with("words.txt", "foo\nbar\n")
        .with("types.txt", "<NUM>\n")
        .with("rules.txt", "- => ALPHA\n")
        .with("odd.txt", "no arrow\n")
        .with("bad-type.txt", "- => VOWEL\n")
        .with("stems.txt", "foo\tf\n")
        .with("empty.txt", "\n")
}

#[test]
fn exception_names() {
    let all = [
        (
            JavaException::IllegalArgument,
            "java.lang.IllegalArgumentException",
        ),
        (JavaException::Io, "java.io.IOException"),
        (
            JavaException::AlreadySet,
            "org.apache.lucene.util.SetOnce$AlreadySetException",
        ),
        (
            JavaException::AlreadyClosed,
            "org.apache.lucene.store.AlreadyClosedException",
        ),
        (
            JavaException::TooComplexToDeterminize,
            "org.apache.lucene.util.automaton.TooComplexToDeterminizeException",
        ),
        (
            JavaException::IllformedLocale,
            "java.util.IllformedLocaleException",
        ),
        (
            JavaException::NoSuchElement,
            "java.util.NoSuchElementException",
        ),
        (
            JavaException::MalformedInput,
            "java.nio.charset.MalformedInputException",
        ),
        (
            JavaException::PatternSyntax,
            "java.util.regex.PatternSyntaxException",
        ),
        (
            JavaException::ArrayIndexOutOfBounds,
            "java.lang.ArrayIndexOutOfBoundsException",
        ),
        (
            JavaException::UnsupportedEncoding,
            "java.io.UnsupportedEncodingException",
        ),
        (
            JavaException::MissingResource,
            "java.util.MissingResourceException",
        ),
        (
            JavaException::IllegalIcuArgument,
            "com.ibm.icu.util.IllegalIcuArgumentException",
        ),
    ];
    for (k, name) in all {
        assert_eq!(k.qualified_name(), name);
    }
    let e = FactoryError::from(lucene_util::automaton::AutomatonError::IllegalState(
        "s".into(),
    ));
    assert_eq!(e.kind, JavaException::IllegalState);
    let w = wrapped_runtime(&FactoryError::io("gone"));
    assert_eq!(w.kind, JavaException::Runtime);
    assert_eq!(w.message, "java.io.IOException: gone");
    assert_eq!(java_split("", ","), vec![""]);
    assert_eq!(java_split("a,,b,,", ","), vec!["a", "", "b"]);
    assert_eq!(java_split(",,", ","), Vec::<&str>::new());
}

#[test]
fn accessors_and_base_mut() {
    let mut stop = build::<StopFilterFactory>(&[("ignoreCase", "true")]).unwrap();
    assert!(stop.is_ignore_case());
    assert!(stop.words().is_none());
    stop.base_mut().set_explicit_lucene_match_version(true);
    let mut lower = build::<LowerCaseFilterFactory>(&[]).unwrap();
    lower.base_mut().set_explicit_lucene_match_version(true);
    assert!(lower.base().is_explicit_lucene_match_version());
    for f in [
        &mut build::<LimitTokenCountFilterFactory>(&[("maxTokenCount", "1")]).unwrap()
            as &mut dyn TokenFilterFactory,
        &mut build::<NGramFilterFactory>(&[("minGramSize", "1"), ("maxGramSize", "1")]).unwrap(),
        &mut build::<WordDelimiterGraphFilterFactory>(&[]).unwrap(),
        &mut build::<SynonymGraphFilterFactory>(&[("synonyms", "x")]).unwrap(),
        &mut build::<ProtectedTermFilterFactory>(&[("protected", "x")]).unwrap(),
        &mut TermPredicateFactory::new(|_| true),
    ] {
        let class = f.base().class_name();
        assert_eq!(f.base_mut().class_name(), class);
    }
    let mut t = build::<TypeTokenFilterFactory>(&[("types", "types.txt")]).unwrap();
    assert!(t.stop_types().is_none());
    t.inform(&loader()).unwrap();
    assert_eq!(t.stop_types().unwrap(), ["<NUM>"]);
    let k = build::<KeywordMarkerFilterFactory>(&[("ignoreCase", "true")]).unwrap();
    assert!(k.is_ignore_case());
    let s = build::<StemmerOverrideFilterFactory>(&[("ignoreCase", "true")]).unwrap();
    assert!(s.is_ignore_case());
    let p =
        build::<ProtectedTermFilterFactory>(&[("protected", "x"), ("ignoreCase", "true")]).unwrap();
    assert!(p.is_ignore_case());
    assert!(p.protected_terms().is_none());
}

#[test]
fn factories_used_before_inform() {
    let not_informed = |f: &dyn TokenFilterFactory| f.create(input("a")).err().unwrap();
    for f in [
        &build::<StopFilterFactory>(&[]).unwrap() as &dyn TokenFilterFactory,
        &build::<HyphenationCompoundWordTokenFilterFactory>(&[("hyphenator", "h")]).unwrap(),
        &build::<TypeTokenFilterFactory>(&[("types", "t")]).unwrap(),
        &build::<HunspellStemFilterFactory>(&[("dictionary", "d")]).unwrap(),
        &build::<PatternTypingFilterFactory>(&[("patternFile", "p")]).unwrap(),
        &build::<DelimitedPayloadTokenFilterFactory>(&[("encoder", "float")]).unwrap(),
        &build::<SnowballPorterFilterFactory>(&[]).unwrap(),
        &build::<ElisionFilterFactory>(&[]).unwrap(),
    ] {
        assert!(matches!(not_informed(f), AnalysisError::IllegalState(_)));
    }
    // The ones Java lets through return their input.
    for f in [
        &build::<KeepWordFilterFactory>(&[]).unwrap() as &dyn TokenFilterFactory,
        &build::<DictionaryCompoundWordTokenFilterFactory>(&[("dictionary", "d")]).unwrap(),
        &build::<StemmerOverrideFilterFactory>(&[]).unwrap(),
        &build::<SynonymFilterFactory>(&[("synonyms", "s")]).unwrap(),
        &build::<KeywordMarkerFilterFactory>(&[]).unwrap(),
    ] {
        assert_eq!(terms(f.create(input("a b")).unwrap()).unwrap(), ["a", "b"]);
    }
    let e = build::<ElisionFilterFactory>(&[]).unwrap();
    assert_eq!(terms(e.normalize(input("l'a"))).unwrap(), ["l'a"]);
    let mut p = build::<ProtectedTermFilterFactory>(&[
        ("protected", "words.txt"),
        ("wrappedFilters", "lowercase"),
    ])
    .unwrap();
    assert!(p.create(input("A")).is_err());
    p.inform(&loader()).unwrap();
    assert_eq!(
        terms(p.create(input("A foo")).unwrap()).unwrap(),
        ["a", "foo"]
    );
    p.set_inner_filters(Vec::new());
    assert_eq!(terms(p.create(input("A")).unwrap()).unwrap(), ["A"]);
    let mut w = TermPredicateFactory::new(|t| t == "x");
    assert_eq!(terms(w.create(input("x")).unwrap()).unwrap(), ["x"]);
    w.inform(&loader()).unwrap();
    assert!(w.as_conditional().is_some());
}

/// A factory that drops its input, which no real factory does.
struct Dropping(FactoryBase);

impl AnalysisFactory for Dropping {
    fn base(&self) -> &FactoryBase {
        &self.0
    }
    fn base_mut(&mut self) -> &mut FactoryBase {
        &mut self.0
    }
}

impl TokenFilterFactory for Dropping {
    fn create(&self, _input: Box<dyn TokenStream>) -> Result<Box<dyn TokenStream>, AnalysisError> {
        Ok(Box::new(KeywordTokenizer::new()))
    }
}

#[test]
fn conditional_chains_report_broken_inner_filters() {
    let base = || FactoryBase::new("x.Dropping", &mut JavaArgs::new()).unwrap();
    let mut w = TermPredicateFactory::new(|_| true);
    w.set_inner_filters(vec![Box::new(Dropping(base()))]);
    assert!(matches!(
        w.create(input("a")).err().unwrap(),
        AnalysisError::IllegalState(_)
    ));
    // An inner factory whose filter cannot be built.
    let mut w = TermPredicateFactory::new(|_| true);
    let broken = build::<LimitTokenCountFilterFactory>(&[("maxTokenCount", "0")]).unwrap();
    w.set_inner_filters(vec![Box::new(broken)]);
    assert!(w.create(input("a")).is_err());
    // The delegate forwards to the chain it wraps.
    let mut w = TermPredicateFactory::new(|t| t.len() > 1);
    w.set_inner_filters(vec![Box::new(
        build::<UpperCaseFilterFactory>(&[]).unwrap(),
    )]);
    let mut ts = w.create(input("ab c")).unwrap();
    assert!(ts.as_tokenizer().is_some());
    assert!(ts.conditional_root().is_none());
    assert_eq!(terms(ts).unwrap(), ["AB", "c"]);
    let cond =
        FactoryCondition::NotProtected(Arc::new(crate::CharArraySet::from_words(["a"], false)));
    assert!(matches!(cond, FactoryCondition::NotProtected(_)));
}

#[test]
fn argument_errors_the_fixtures_do_not_reach() {
    assert!(
        build::<CapitalizationFilterFactory>(&[("minWordLength", "-1")])
            .unwrap()
            .create(input("a"))
            .is_err()
    );
    assert!(
        build::<CodepointCountFilterFactory>(&[("min", "3"), ("max", "1")])
            .unwrap()
            .create(input("a"))
            .is_err()
    );
    assert!(build::<ConcatenateGraphFilterFactory>(&[("maxGraphExpansions", "x")]).is_err());
    assert!(build::<DelimitedTermFrequencyTokenFilterFactory>(&[("delimiter", "ab")]).is_err());
    assert!(build::<FingerprintFilterFactory>(&[("maxOutputTokenSize", "x")]).is_err());
    let e =
        build::<TruncateTokenFilterFactory>(&[("truncateAfterChars", "1"), ("prefixLength", "1")])
            .err()
            .unwrap();
    assert_eq!(
        e.message,
        "Can only give one of the following parameters: [truncateAfterCodePoints, truncateAfterChars, prefixLength]"
    );
    assert!(build::<DateRecognizerFilterFactory>(&[("locale", "de")]).is_ok());
    let e = build::<DateRecognizerFilterFactory>(&[("locale", "en-Latn")])
        .err()
        .unwrap();
    assert_eq!(e.kind, JavaException::UnsupportedOperation);
    let e = build::<DateRecognizerFilterFactory>(&[("locale", "en_US!")])
        .err()
        .unwrap();
    assert_eq!(e.kind, JavaException::IllformedLocale);
    assert!(build::<DateRecognizerFilterFactory>(&[("locale", "EN")]).is_ok());
    for key in ["minWordSize", "minSubwordSize", "maxSubwordSize"] {
        assert!(build::<DictionaryCompoundWordTokenFilterFactory>(&[
            ("dictionary", "d"),
            (key, "x")
        ])
        .is_err());
        // Java checks the sizes when the filter is built, not the factory.
        let mut f = build::<DictionaryCompoundWordTokenFilterFactory>(&[
            ("dictionary", "words.txt"),
            (key, "-1"),
        ])
        .unwrap();
        f.inform(&loader()).unwrap();
        let e = f.create(input("a")).err().unwrap();
        assert!(
            matches!(&e, AnalysisError::IllegalArgument(m) if *m == format!("{key} cannot be negative")),
            "{e}"
        );
    }
    let mut h = build::<HunspellStemFilterFactory>(&[("dictionary", "words.txt")]).unwrap();
    assert_eq!(
        h.inform(&loader()).err().unwrap().kind,
        JavaException::NullPointer
    );
    let mut so = build::<StemmerOverrideFilterFactory>(&[("dictionary", ",")]).unwrap();
    so.inform(&loader()).unwrap();
    assert_eq!(terms(so.create(input("foo")).unwrap()).unwrap(), ["foo"]);
    for (file, message) in [
        ("odd.txt", "Invalid Mapping Rule : [no arrow]"),
        (
            "bad-type.txt",
            "Invalid Mapping Rule : [- => VOWEL]. Illegal type.",
        ),
    ] {
        let mut w = build::<WordDelimiterGraphFilterFactory>(&[("types", file)]).unwrap();
        assert_eq!(w.inform(&loader()).err().unwrap().message, message);
    }
    let mut w = build::<WordDelimiterGraphFilterFactory>(&[("adjustOffsets", "true")]).unwrap();
    w.inform(&loader()).unwrap();
    let mut pt = build::<PatternTypingFilterFactory>(&[("patternFile", "words.txt")]).unwrap();
    assert_eq!(
        pt.inform(&loader()).err().unwrap().kind,
        JavaException::StringIndexOutOfBounds
    );
}

#[test]
fn hunspell_errors_keep_their_classes() {
    use super::filters_resource::hunspell_error_for_tests as map;
    use crate::hunspell::HunspellError as H;
    for (e, kind) in [
        (
            H::IllegalArgument("a".into()),
            JavaException::IllegalArgument,
        ),
        (H::NumberFormat("a".into()), JavaException::NumberFormat),
        (
            H::IndexOutOfBounds("a".into()),
            JavaException::ArrayIndexOutOfBounds,
        ),
        (
            H::UnsupportedCharset("a".into()),
            JavaException::UnsupportedOperation,
        ),
        (H::UnmappableCharacter("a".into()), JavaException::Io),
        (
            H::NegativeArraySize("a".into()),
            JavaException::IllegalState,
        ),
        (
            H::Parse {
                message: "m".into(),
                line: 1,
            },
            JavaException::Io,
        ),
    ] {
        assert_eq!(map(e).kind, kind);
    }
}

#[test]
fn char_reader_boxes_compose() {
    let f = build::<PersianCharFilterFactory>(&[]).unwrap();
    let r: Box<dyn CharReader> = f.create(Box::new(StrReader::new("a")));
    let mut r = f.normalize(r);
    assert_eq!(crate::reader::read_to_string(&mut *r).unwrap(), "a");
}

/// A tokenizer is a source: no conditional filter's wrapper lies below it,
/// whatever the factory (`TokenStream::conditional_root` has no default, so
/// each one answers for itself).
#[test]
fn every_tokenizer_is_a_conditional_source() {
    let mut built = 0;
    for name in spi::available_tokenizers() {
        let pairs: &[(&str, &str)] = match name {
            "pattern" | "simplePattern" | "simplePatternSplit" => &[("pattern", "a")],
            _ => &[],
        };
        let Ok(factory) = spi::tokenizer_for_name(name, &mut JavaArgs::from_pairs(pairs)) else {
            continue;
        };
        let Ok(mut ts) = factory.create() else {
            continue;
        };
        assert!(ts.conditional_root().is_none(), "{name}");
        built += 1;
    }
    assert!(built >= 13, "{built}");
}

/// `Locale.Builder.setLanguageTag` over tags of every shape, against what
/// JDK 21 answered (a throwaway `Locale.Builder` program): well-formed, or
/// `IllformedLocaleException`'s message.
#[test]
fn language_tags_parse_as_the_jdk_does() {
    let cases: &[(&str, Option<&str>)] = &[
        ("en", None),
        ("en-", Some("Empty subtag [at index 3]")),
        ("-en", Some("Empty subtag [at index 0]")),
        ("e", Some("Invalid subtag: e [at index 0]")),
        ("x", Some("Incomplete privateuse [at index 0]")),
        ("x-", Some("Incomplete privateuse [at index 0]")),
        ("en-x", Some("Incomplete privateuse [at index 3]")),
        ("en-x-a", None),
        ("en-a", Some("Incomplete extension 'a' [at index 3]")),
        ("en-a-bb", None),
        ("en-a-bb-a-cc", None),
        ("en-a-bb-x", Some("Incomplete privateuse [at index 8]")),
        ("en-a-x-1", Some("Incomplete extension 'a' [at index 3]")),
        ("123", Some("Invalid subtag: 123 [at index 0]")),
        ("en-123", None),
        ("en-US", None),
        ("en-USA", None),
        ("en-us-us", Some("Invalid subtag: us [at index 6]")),
        ("en--US", Some("Empty subtag [at index 3]")),
        ("en-Latn", None),
        ("en-Latn-US", None),
        ("en-Latn-US-posix", None),
        ("en-1abc", None),
        ("en-abcd", None),
        ("en-abcde", None),
        ("en-abc", None),
        ("en-abc-def-ghi", None),
        (
            "en-abc-def-ghi-jkl",
            Some("Invalid subtag: jkl [at index 15]"),
        ),
        ("abcdefghi", Some("Invalid subtag: abcdefghi [at index 0]")),
        (
            "en-abcdefghi",
            Some("Invalid subtag: abcdefghi [at index 3]"),
        ),
        ("i-klingon", None),
        ("en-GB-oed", None),
        ("zh-min-nan", None),
        ("art-lojban", None),
        ("en-x-abcdefghi", Some("Incomplete privateuse [at index 3]")),
        (
            "en-x-toolongsubtag",
            Some("Incomplete privateuse [at index 3]"),
        ),
        ("en-US-posix-posix", None),
        ("en-a-bb-a-cc", None),
        ("en-a-bb-b-cc", None),
        ("en US", Some("Invalid subtag: en US [at index 0]")),
        ("en_US", Some("Invalid subtag: en_US [at index 0]")),
        ("en-ü", Some("Invalid subtag: ü [at index 3]")),
        ("ü", Some("Invalid subtag: ü [at index 0]")),
        ("", Some("Empty subtag [at index 0]")),
        ("EN-us", None),
        ("en-a-bb-x-x", None),
        ("x-a", None),
        ("x-a-", Some("Empty subtag [at index 4]")),
        ("qaa", None),
        ("sgn-be-fr", None),
        ("en-u-ca-japanese", None),
        ("en-u-ca", None),
        ("en-u-xx-yyy", None),
        ("en-t-de", None),
        ("en-t-de-u-ca-gregory", None),
        ("en-US-u-co-phonebk-x-a", None),
        ("en-a-b", Some("Incomplete extension 'a' [at index 3]")),
        ("en-1234", None),
        ("en-12345", None),
        ("en-001", None),
        ("en-0", Some("Invalid subtag: 0 [at index 3]")),
        ("-", Some("Empty subtag [at index 0]")),
    ];
    for (tag, want) in cases {
        assert_eq!(
            super::filters_misc::language_tag_error(tag).as_deref(),
            *want,
            "{tag}"
        );
    }
}

/// A size taken from configuration that Java's heap could not hold is
/// Java's `OutOfMemoryError` (an error here), never an aborted process:
/// MinHash's `hashCount * bucketCount` sets, the n-gram tokenizers'
/// `2 * maxGram + 1024` buffers.
#[test]
fn hostile_sizes_are_errors_not_aborts() {
    let oom = |e: AnalysisError| matches!(e, AnalysisError::IllegalArgument(m) if m.starts_with("OutOfMemoryError"));
    for pairs in [
        &[("hashCount", "1000000000")][..],
        &[("bucketCount", "2000000000")][..],
        &[("hashCount", "100000"), ("bucketCount", "100000")][..],
    ] {
        let f = build::<MinHashFilterFactory>(pairs).unwrap();
        assert!(oom(f.create(input("a")).err().unwrap()), "{pairs:?}");
    }
    for name in ["nGram", "edgeNGram"] {
        let mut args = JavaArgs::from_pairs(&[("minGramSize", "1"), ("maxGramSize", "1000000000")]);
        let f = spi::tokenizer_for_name(name, &mut args).unwrap();
        assert!(oom(f.create().err().unwrap()), "{name}");
    }
    // Sizes a heap holds still build.
    let f = build::<MinHashFilterFactory>(&[("hashCount", "2"), ("bucketCount", "3")]).unwrap();
    assert!(f.create(input("a")).is_ok());
}
