#![cfg(test)]
use super::*;
use crate::token_stream::consume;
use crate::{StandardAnalyzer, StandardTokenizer};
use std::sync::atomic::{AtomicUsize, Ordering};

fn terms(a: &Analyzer, field: &str, text: &str) -> Vec<String> {
    let mut ts = a.token_stream(field, text).unwrap();
    let mut out = Vec::new();
    consume(&mut ts, |x| out.push(x.term().to_string())).unwrap();
    out
}

#[test]
fn a_shared_text_stream_yields_the_same_tokens_as_a_borrowed_one() {
    let text = "The quick brown 😀 fox, jumps over the lazy dog's 世界 back";
    let shared: Arc<str> = Arc::from(text);
    let a = Analyzer::new(StandardAnalyzer::new());
    let mut want = Vec::new();
    let mut ts = a.token_stream("f", text).unwrap();
    consume(&mut ts, |x| {
        want.push((x.term().to_string(), x.start_offset(), x.end_offset()))
    })
    .unwrap();
    drop(ts);
    // twice: the second run reuses the components the first returned
    for _ in 0..2 {
        let mut got = Vec::new();
        let mut ts = a.token_stream_shared("f", &shared).unwrap();
        consume(&mut ts, |x| {
            got.push((x.term().to_string(), x.start_offset(), x.end_offset()))
        })
        .unwrap();
        drop(ts);
        assert_eq!(got, want);
    }
    // the stream let go of its clone once dropped
    assert_eq!(Arc::strong_count(&shared), 1);
    // and a borrowed stream after a shared one still works
    assert_eq!(terms(&a, "f", "a b"), vec!["a", "b"]);
}

/// Counts `create_components` calls; per-field gaps.
struct Counting(Arc<AtomicUsize>);

impl AnalyzerDefinition for Counting {
    fn create_components(&self, _f: &str) -> Result<TokenStreamComponents, AnalysisError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(TokenStreamComponents::new(StandardTokenizer::new()))
    }
    fn position_increment_gap(&self, field: &str) -> i32 {
        if field == "wide" {
            100
        } else {
            0
        }
    }
}

#[test]
fn components_are_reused_per_strategy() {
    let n = Arc::new(AtomicUsize::new(0));
    let a = Analyzer::new(Counting(Arc::clone(&n)));
    assert_eq!(a.reuse_strategy(), ReuseStrategy::Global);
    assert_eq!(terms(&a, "f", "a b"), vec!["a", "b"]);
    assert_eq!(terms(&a, "g", "c"), vec!["c"]);
    assert_eq!(
        n.load(Ordering::SeqCst),
        1,
        "global: one set for every field"
    );
    // two live streams need two sets
    let s1 = a.token_stream("f", "x").unwrap();
    let s2 = a.token_stream("f", "y").unwrap();
    drop((s1, s2));
    assert_eq!(n.load(Ordering::SeqCst), 2);

    let n = Arc::new(AtomicUsize::new(0));
    let a = Analyzer::with_reuse_strategy(Counting(Arc::clone(&n)), ReuseStrategy::PerField);
    terms(&a, "f", "a");
    terms(&a, "g", "a");
    terms(&a, "f", "a");
    assert_eq!(n.load(Ordering::SeqCst), 2, "per field: one set per field");
    assert_eq!(a.position_increment_gap_for_field("wide"), 100);
    assert_eq!(a.position_increment_gap_for_field("f"), 0);
    assert_eq!(a.offset_gap_for_field("f"), 1);
}

#[test]
fn close_makes_the_analyzer_unusable() {
    let a = Analyzer::new(StandardAnalyzer::new());
    a.close();
    let err = a.token_stream("f", "x").err().unwrap();
    assert!(matches!(err, AnalysisError::AlreadyClosed(_)), "{err}");
    // a stream dropped after close does not resurrect the pool
    let b = Analyzer::new(StandardAnalyzer::new());
    let ts = b.token_stream("f", "x").unwrap();
    b.close();
    drop(ts);
    assert!(b.token_stream("f", "x").is_err());
}

/// Components are made once per thread and reused: Java's thread local.
#[test]
fn components_are_reused_per_thread() {
    let made = Arc::new(AtomicUsize::new(0));
    let a = Arc::new(Analyzer::new(Counting(Arc::clone(&made))));
    for text in ["one", "two", "three"] {
        assert_eq!(terms(&a, "f", text), vec![text]);
    }
    assert_eq!(made.load(Ordering::SeqCst), 1);
    let other = Arc::clone(&a);
    std::thread::spawn(move || {
        for text in ["four", "five"] {
            assert_eq!(terms(&other, "f", text), vec![text]);
        }
    })
    .join()
    .unwrap();
    assert_eq!(
        made.load(Ordering::SeqCst),
        2,
        "one set for the other thread"
    );
    assert_eq!(terms(&a, "f", "six"), vec!["six"]);
    assert_eq!(made.load(Ordering::SeqCst), 2, "this thread's set was kept");
    // two streams alive at once use two sets
    let (mut x, mut y) = (
        a.token_stream("f", "x").unwrap(),
        a.token_stream("f", "y").unwrap(),
    );
    x.reset().unwrap();
    y.reset().unwrap();
    assert!(x.increment_token().unwrap() && y.increment_token().unwrap());
    assert_eq!(made.load(Ordering::SeqCst), 3);
}

/// A thread's entries for analyzers that are gone are forgotten, and a
/// closed analyzer's are dropped at once.
#[test]
fn a_threads_pools_forget_dropped_and_closed_analyzers() {
    for _ in 0..50 {
        let a = Analyzer::new(Counting(Arc::new(AtomicUsize::new(0))));
        assert_eq!(terms(&a, "f", "x"), vec!["x"]);
    }
    let keep = Analyzer::new(Counting(Arc::new(AtomicUsize::new(0))));
    assert_eq!(terms(&keep, "f", "y"), vec!["y"]);
    let live = THREAD_POOLS.with(|p| p.borrow().len());
    assert!(live <= 2, "{live} pools kept on this thread");
    keep.close();
    let after_close = THREAD_POOLS.with(|p| {
        p.borrow()
            .iter()
            .filter(|e| e.owner.strong_count() > 0 && e.owner.as_ptr() == Arc::as_ptr(&keep.alive))
            .count()
    });
    assert_eq!(after_close, 0);
    assert!(keep.token_stream("f", "z").is_err());
}

#[test]
fn an_abandoned_stream_is_closed_and_reusable() {
    let a = Analyzer::new(StandardAnalyzer::new());
    {
        let mut ts = a.token_stream("f", "one two").unwrap();
        ts.reset().unwrap();
        assert!(ts.increment_token().unwrap());
        // dropped mid-stream, never closed
    }
    assert_eq!(terms(&a, "f", "three"), vec!["three"]);
}

/// Emits its input's first token, then `extra` empty ones.
struct Repeat {
    input: Box<dyn TokenStream>,
    n: usize,
    extra: usize,
}

impl crate::TokenFilter for Repeat {
    type Input = Box<dyn TokenStream>;
    fn input(&self) -> &Self::Input {
        &self.input
    }
    fn input_mut(&mut self) -> &mut Self::Input {
        &mut self.input
    }
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        self.n += 1;
        if self.n == 1 {
            return self.input.increment_token();
        }
        Ok(self.n <= 1 + self.extra)
    }
}

struct Norm(usize);

impl AnalyzerDefinition for Norm {
    fn create_components(&self, _f: &str) -> Result<TokenStreamComponents, AnalysisError> {
        Ok(TokenStreamComponents::new(StandardTokenizer::new()))
    }
    fn normalize(&self, _f: &str, input: Box<dyn TokenStream>) -> Box<dyn TokenStream> {
        if self.0 == 0 {
            Box::new(crate::FilteringTokenFilter::new(
                input,
                |_: &AttributeSource| false,
            ))
        } else {
            Box::new(Repeat {
                input,
                n: 0,
                extra: self.0,
            })
        }
    }
}

#[test]
fn normalize_runs_the_normalize_chain_over_one_token() {
    let a = Analyzer::new(StandardAnalyzer::new());
    assert_eq!(a.normalize("f", "HeLLo World").unwrap(), b"hello world");
    assert_eq!(a.normalize("f", "").unwrap(), b"");
    // a normalize chain that drops the token, or adds one, is an error
    let err = Analyzer::new(Norm(0)).normalize("f", "x").unwrap_err();
    assert!(err.to_string().contains("but got 0"), "{err}");
    let err = Analyzer::new(Norm(1)).normalize("f", "x").unwrap_err();
    assert!(err.to_string().contains("but got 2+"), "{err}");
}

/// Uppercases ASCII letters.
struct Upper(Box<dyn CharReader>);

impl crate::CharFilter for Upper {
    fn input(&self) -> &dyn CharReader {
        &*self.0
    }
    fn input_mut(&mut self) -> &mut dyn CharReader {
        &mut *self.0
    }
    fn read_filtered(&mut self, buf: &mut [u16]) -> Result<usize, AnalysisError> {
        let n = self.0.read(buf)?;
        for u in &mut buf[..n] {
            if (u16::from(b'a')..=u16::from(b'z')).contains(u) {
                *u -= 32;
            }
        }
        Ok(n)
    }
    fn correct(&self, off: i32) -> i32 {
        off
    }
}

/// Lowercases field "lower"'s tokens; uppercases field "shout"'s reader.
struct PerField {
    inner: Analyzer,
}

impl AnalyzerWrapper for PerField {
    fn wrapped_analyzer(&self, _f: &str) -> &Analyzer {
        &self.inner
    }
    fn wrap_components(&self, field: &str, c: TokenStreamComponents) -> TokenStreamComponents {
        if field != "lower" {
            return c;
        }
        let (source, sink) = c.into_parts();
        TokenStreamComponents::from_parts(source, Box::new(LowerCaseFilter::new(sink)))
    }
    fn wrap_reader(&self, field: &str, r: Box<dyn CharReader>) -> Box<dyn CharReader> {
        if field == "shout" {
            Box::new(Upper(r))
        } else {
            r
        }
    }
    fn wrap_reader_for_normalization(
        &self,
        field: &str,
        r: Box<dyn CharReader>,
    ) -> Box<dyn CharReader> {
        self.wrap_reader(field, r)
    }
    fn wrap_token_stream_for_normalization(
        &self,
        _f: &str,
        input: Box<dyn TokenStream>,
    ) -> Box<dyn TokenStream> {
        input
    }
}

#[test]
fn analyzer_wrapper_wraps_components_readers_and_gaps() {
    let inner = Analyzer::new(Counting(Arc::new(AtomicUsize::new(0))));
    let a = Analyzer::with_reuse_strategy(PerField { inner }, ReuseStrategy::PerField);
    assert_eq!(terms(&a, "lower", "AbC d"), vec!["abc", "d"]);
    assert_eq!(terms(&a, "other", "AbC"), vec!["AbC"]);
    assert_eq!(terms(&a, "shout", "AbC"), vec!["ABC"]);
    assert_eq!(a.position_increment_gap_for_field("wide"), 100);
    assert_eq!(a.offset_gap_for_field("x"), 1);
    assert_eq!(a.normalize("shout", "ab").unwrap(), b"AB");
    assert_eq!(a.normalize("x", "ab").unwrap(), b"ab");
}

struct Delegate {
    a: Analyzer,
    b: Analyzer,
}

impl DelegatingAnalyzerWrapper for Delegate {
    fn wrapped_analyzer(&self, field: &str) -> &Analyzer {
        if field == "b" {
            &self.b
        } else {
            &self.a
        }
    }
}

#[test]
fn delegating_wrapper_reuses_the_wrapped_components() {
    let na = Arc::new(AtomicUsize::new(0));
    let d = Analyzer::delegating(
        Delegate {
            a: Analyzer::new(Counting(Arc::clone(&na))),
            b: Analyzer::new(StandardAnalyzer::new()),
        },
        ReuseStrategy::PerField,
    );
    assert_eq!(terms(&d, "a", "X y"), vec!["X", "y"]);
    assert_eq!(terms(&d, "a", "z"), vec!["z"]);
    assert_eq!(terms(&d, "b", "X y"), vec!["x", "y"]);
    assert_eq!(na.load(Ordering::SeqCst), 1);
    assert_eq!(d.position_increment_gap_for_field("wide"), 100);
    assert_eq!(d.normalize("b", "Q").unwrap(), b"q");
    // its own components, asked for directly, are the wrapped analyzer's
    let mut c = d.create_components("b").unwrap();
    c.set_reader(Box::new(StrReader::new("Hi"))).unwrap();
    let mut out = Vec::new();
    consume(c.token_stream(), |x| out.push(x.term().to_string())).unwrap();
    assert_eq!(out, vec!["hi"]);
}

struct NoTokenizer;

impl AnalyzerDefinition for NoTokenizer {
    fn create_components(&self, _f: &str) -> Result<TokenStreamComponents, AnalysisError> {
        Ok(TokenStreamComponents::new(StringTokenStream::new(
            "x".into(),
            1,
        )))
    }
}

#[test]
fn components_without_a_tokenizer_cannot_take_a_reader() {
    let err = Analyzer::new(NoTokenizer)
        .token_stream("f", "x")
        .err()
        .unwrap();
    assert!(err.to_string().contains("no Tokenizer source"), "{err}");
}

#[test]
fn materialising_helpers_and_builders() {
    let a = Analyzer::standard(None)
        .with_position_increment_gap(7)
        .with_offset_gap(2);
    assert_eq!(a.position_increment_gap(), 7);
    assert_eq!(a.offset_gap(), 2);
    let s = a.try_analyze_stream("Fox  ").unwrap();
    assert_eq!(s.tokens.len(), 1);
    assert_eq!(s.final_offset, 5);
    let mut seen = Vec::new();
    let end = a
        .try_for_each_token("A b", |t, s, e, i| seen.push((t.to_string(), s, e, i)))
        .unwrap();
    assert_eq!(
        seen,
        vec![("a".to_string(), 0, 1, 1), ("b".to_string(), 2, 3, 1)]
    );
    assert_eq!(end, (0, 3));
    // builders on a definition-backed analyzer are no-ops
    let d = Analyzer::new(StandardAnalyzer::new())
        .with_ascii_folding()
        .with_stemming()
        .with_snowball_stemming()
        .with_synonyms(HashMap::new())
        .with_bidirectional_synonyms(HashMap::new());
    assert_eq!(d.analyze("Café")[0].term, "café");
    // keyword normalize is the identity
    assert_eq!(Analyzer::keyword().normalize("f", "AB").unwrap(), b"AB");
    let k = Analyzer::keyword();
    let mut ts = k.token_stream("f", "q").unwrap();
    assert!(ts.as_tokenizer().is_some());
    ts.close().unwrap();
}
