//! Boundary and error handling of the analyzing suggesters (M13 T13.3c) that
//! the Java-fixture differential tests do not reach: limits, unsupported
//! inputs, corrupt saved state.

use std::collections::BTreeSet;
use std::sync::Arc;

use lucene_analysis::Analyzer;
use lucene_search::suggest::analyzing::{AnalyzingSuggester, FuzzySuggester};
use lucene_search::suggest::input::InputIterator;
use lucene_search::suggest::lookup::Lookup;
use lucene_store::{SliceInput, VecDataOutput};

/// One phrase, optionally with a payload and contexts.
struct One {
    surface: String,
    payload: Option<Vec<u8>>,
    contexts: Option<BTreeSet<Vec<u8>>>,
    taken: bool,
}

fn one(surface: &str) -> One {
    One {
        surface: surface.to_string(),
        payload: None,
        contexts: None,
        taken: false,
    }
}

impl InputIterator for One {
    fn next(&mut self) -> Option<Vec<u8>> {
        if std::mem::replace(&mut self.taken, true) {
            return None;
        }
        Some(self.surface.as_bytes().to_vec())
    }
    fn weight(&self) -> i64 {
        1
    }
    fn payload(&self) -> Option<&[u8]> {
        self.payload.as_deref()
    }
    fn has_payloads(&self) -> bool {
        self.payload.is_some()
    }
    fn contexts(&self) -> Option<&BTreeSet<Vec<u8>>> {
        self.contexts.as_ref()
    }
    fn has_contexts(&self) -> bool {
        self.contexts.is_some()
    }
}

fn analyzer() -> Arc<Analyzer> {
    Arc::new(Analyzer::standard(None))
}

fn message(e: lucene_store::Error) -> String {
    e.to_string()
}

#[test]
fn an_analyzed_form_over_the_length_limit_is_refused() {
    let mut s = AnalyzingSuggester::new(analyzer());
    let long = "x ".repeat(17000);
    let err = message(s.build(&mut one(&long)).unwrap_err());
    assert!(
        err.contains("cannot handle analyzed forms > 32765 in length (got 34000)"),
        "{err}"
    );
}

#[test]
fn a_surface_form_over_the_length_limit_is_refused_when_it_carries_a_payload() {
    // trailing spaces are not tokens, so the analyzed form stays short
    let a = analyzer();
    let surface = format!("xy{}", " ".repeat(33000));
    let mut s = AnalyzingSuggester::new(a.clone());
    let mut phrase = one(&surface);
    phrase.payload = Some(b"p".to_vec());
    let err = message(s.build(&mut phrase).unwrap_err());
    assert!(
        err.contains("cannot handle surface form > 32765 in length (got 33002)"),
        "{err}"
    );
    // without a payload the surface form is as long as it likes
    let mut s = AnalyzingSuggester::new(a);
    s.build(&mut one(&surface)).unwrap();
    assert_eq!(s.get_count().unwrap(), 1);
}

#[test]
fn contexts_are_refused_at_build_and_a_phrase_with_no_tokens_is_not_indexed() {
    let mut s = AnalyzingSuggester::new(analyzer());
    let mut phrase = one("a b");
    phrase.contexts = Some(BTreeSet::new());
    let err = message(s.build(&mut phrase).unwrap_err());
    assert!(err.contains("doesn't support contexts"), "{err}");

    // punctuation analyzes to no tokens, so no analyzed form: nothing is
    // indexed (and an FST with no entries is not kept)
    let mut s = AnalyzingSuggester::new(analyzer());
    s.build(&mut one("!!!")).unwrap();
    assert_eq!(s.get_count().unwrap(), 0);
    assert!(s.lookup("", None, false, 3).unwrap().is_empty());
    assert!(!s.store(&mut VecDataOutput::new()).unwrap());
}

#[test]
fn a_saved_suggester_that_is_cut_short_is_refused() {
    let mut s = FuzzySuggester::new(analyzer());
    s.build(&mut one("hello world")).unwrap();
    let mut out = VecDataOutput::new();
    assert!(s.store(&mut out).unwrap());
    let bytes = out.into_inner();
    for cut in [0, 1, 4, bytes.len() / 2, bytes.len() - 1] {
        let mut fresh = FuzzySuggester::new(analyzer());
        assert!(
            fresh.load(&mut SliceInput::new(&bytes[..cut])).is_err(),
            "cut at {cut}"
        );
    }
    let mut fresh = FuzzySuggester::new(analyzer());
    assert!(fresh.load(&mut SliceInput::new(&bytes)).unwrap());
    assert_eq!(fresh.get_count().unwrap(), 1);
    let hits = fresh.lookup("helo", None, false, 3).unwrap();
    assert_eq!(hits[0].key, "hello world");
}

#[test]
fn an_unbuilt_suggester_answers_nothing_and_a_rebuild_replaces_the_data() {
    let mut s = AnalyzingSuggester::new(analyzer());
    assert_eq!(s.get_count().unwrap(), 0);
    assert!(s.lookup("a", None, false, 3).unwrap().is_empty());
    s.build(&mut one("first phrase")).unwrap();
    s.build(&mut one("second phrase")).unwrap();
    assert_eq!(s.get_count().unwrap(), 1);
    assert!(s.lookup("first", None, false, 3).unwrap().is_empty());
    assert_eq!(s.lookup("second", None, false, 3).unwrap().len(), 1);
}

#[test]
fn fuzzy_edits_of_zero_and_unicode_prefixes_match_exactly() {
    let mut s = FuzzySuggester::with_options(
        analyzer(),
        analyzer(),
        3,
        256,
        -1,
        true,
        0,
        true,
        0,
        0,
        true,
    )
    .unwrap();
    s.build(&mut one("日本語 text")).unwrap();
    assert_eq!(s.lookup("日", None, false, 3).unwrap().len(), 1);
    assert!(s.lookup("本", None, false, 3).unwrap().is_empty());
}

mod factory {
    use lucene_analysis::factory::args::JavaArgs;
    use lucene_analysis::factory::spi;
    use lucene_analysis::factory::{
        AnalysisFactory, FactoryClass, JavaException, MapResourceLoader, TokenFilterFactory,
    };
    use lucene_analysis::{KeywordTokenizer, TokenStream};
    use lucene_search::suggest::suggest_stop_factory::{
        register_factories, SuggestStopFilterFactory,
    };

    fn build(pairs: &[(&str, &str)]) -> SuggestStopFilterFactory {
        SuggestStopFilterFactory::from_args(&mut JavaArgs::from_pairs(pairs)).unwrap()
    }

    fn input() -> Box<dyn TokenStream> {
        Box::new(KeywordTokenizer::new())
    }

    #[test]
    fn arguments_default_stop_words_and_resources() {
        assert!(
            SuggestStopFilterFactory::from_args(&mut JavaArgs::from_pairs(&[("x", "1")])).is_err()
        );
        let mut f = build(&[("ignoreCase", "true")]);
        assert!(f.is_ignore_case());
        assert!(f.is_resource_loader_aware());
        assert!(f.stop_words().is_none());
        assert!(f
            .create(input())
            .err()
            .unwrap()
            .to_string()
            .contains("not informed"));
        f.inform(&MapResourceLoader::new()).unwrap();
        assert!(f.stop_words().unwrap().contains("The"));
        assert!(f.create(input()).is_ok());
        assert!(f.base().class_name().ends_with("SuggestStopFilterFactory"));
        assert!(f.base_mut().class_name().ends_with("Factory"));

        // a format without words
        let mut f = build(&[("format", "snowball")]);
        let err = f.inform(&MapResourceLoader::new()).unwrap_err();
        assert_eq!(err.kind, JavaException::IllegalArgument);
        assert!(err
            .to_string()
            .contains("'format' can not be specified w/o an explicit 'words' file: snowball"));
    }

    #[test]
    fn word_files_in_either_format() {
        let loader = MapResourceLoader::new()
            .with("stops.txt", b"foo\nbar\n")
            .with("snow.txt", b"baz | a comment\n");
        let mut f = build(&[("words", "stops.txt")]);
        f.inform(&loader).unwrap();
        assert!(f.stop_words().unwrap().contains("foo"));
        assert!(!f.stop_words().unwrap().contains("the"));
        let mut f = build(&[("words", "snow.txt"), ("format", "SNOWBALL")]);
        f.inform(&loader).unwrap();
        assert!(f.stop_words().unwrap().contains("baz"));
        let mut f = build(&[("words", "stops.txt"), ("format", "csv")]);
        let err = f.inform(&loader).unwrap_err();
        assert!(err
            .to_string()
            .contains("Unknown 'format' specified for 'words' file: csv"));
        assert!(build(&[("words", "missing.txt")]).inform(&loader).is_err());
    }

    #[test]
    fn registered_under_its_spi_name() {
        register_factories().unwrap();
        register_factories().unwrap();
        assert!(spi::lookup_token_filter("suggestStop").is_ok());
    }
}
