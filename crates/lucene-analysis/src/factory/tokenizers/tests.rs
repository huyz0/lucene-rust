#![cfg(test)]
use super::*;

fn build<F: FactoryClass>(pairs: &[(&str, &str)]) -> Result<F, FactoryError> {
    F::from_args(&mut JavaArgs::from_pairs(pairs))
}

fn terms(f: &dyn TokenizerFactory, text: &str) -> Result<Vec<String>, AnalysisError> {
    let mut t = f.create()?;
    let tok = t.as_tokenizer().expect("a tokenizer");
    tok.set_reader(Box::new(crate::StrReader::new(text)))?;
    t.reset()?;
    let mut out = Vec::new();
    while t.increment_token()? {
        out.push(t.attributes().term().to_string());
    }
    t.end()?;
    t.close()?;
    Ok(out)
}

#[test]
fn every_tokenizer_builds_and_runs() {
    let text = "Hello wörld/a/b 12 foo@bar.com";
    let cases: Vec<(Box<dyn TokenizerFactory>, usize)> = vec![
        (
            Box::new(build::<StandardTokenizerFactory>(&[("maxTokenLength", "3")]).unwrap()),
            10,
        ),
        (Box::new(build::<ClassicTokenizerFactory>(&[]).unwrap()), 6),
        (
            Box::new(build::<UAX29URLEmailTokenizerFactory>(&[]).unwrap()),
            6,
        ),
        (Box::new(build::<KeywordTokenizerFactory>(&[]).unwrap()), 1),
        (
            Box::new(build::<LetterTokenizerFactory>(&[("maxTokenLen", "2")]).unwrap()),
            14,
        ),
        (
            Box::new(build::<WhitespaceTokenizerFactory>(&[("rule", "unicode")]).unwrap()),
            4,
        ),
        (
            Box::new(build::<WhitespaceTokenizerFactory>(&[]).unwrap()),
            4,
        ),
        (
            Box::new(build::<EdgeNGramTokenizerFactory>(&[("maxGramSize", "2")]).unwrap()),
            2,
        ),
        (Box::new(build::<NGramTokenizerFactory>(&[]).unwrap()), 59),
        (
            Box::new(build::<PathHierarchyTokenizerFactory>(&[("reverse", "true")]).unwrap()),
            3,
        ),
        (
            Box::new(build::<PathHierarchyTokenizerFactory>(&[("replace", "|")]).unwrap()),
            3,
        ),
        (
            Box::new(build::<PatternTokenizerFactory>(&[("pattern", "\\s+")]).unwrap()),
            4,
        ),
        (
            Box::new(build::<SimplePatternTokenizerFactory>(&[("pattern", "[a-z]+")]).unwrap()),
            8,
        ),
        (
            Box::new(build::<SimplePatternSplitTokenizerFactory>(&[("pattern", " ")]).unwrap()),
            4,
        ),
        (
            Box::new(build::<WikipediaTokenizerFactory>(&[("untokenizedTypes", "b, i")]).unwrap()),
            6,
        ),
    ];
    for (f, n) in cases {
        let got = terms(&*f, text).unwrap();
        assert_eq!(got.len(), n, "{}: {got:?}", f.base().class_name());
    }
    let k = build::<KeywordTokenizerFactory>(&[("maxTokenLen", "10")]).unwrap();
    assert_eq!(k.max_token_len(), 10);
    // Thai configures as in Java and, without an installed word
    // iterator, refuses to create its tokenizer.
    let thai = build::<ThaiTokenizerFactory>(&[]).unwrap();
    let e = thai.create().err().unwrap();
    assert_eq!(
        e,
        AnalysisError::UnsupportedOperation(crate::lang::th::UNSUPPORTED.into())
    );
    let e = build::<ThaiTokenizerFactory>(&[("x", "1")]).err().unwrap();
    assert_eq!(e.message, "Unknown parameters: {x=1}");
}

#[test]
fn javas_argument_errors() {
    let e = build::<KeywordTokenizerFactory>(&[("maxTokenLen", "0")])
        .err()
        .unwrap();
    assert_eq!(
        e.message,
        "maxTokenLen must be greater than 0 and less than 1048576 passed: 0"
    );
    let e = build::<WhitespaceTokenizerFactory>(&[("rule", "x")])
        .err()
        .unwrap();
    assert_eq!(
        e.message,
        "Configuration Error: 'rule' value must be one of [java, unicode]"
    );
    let e = build::<ThaiTokenizerFactory>(&[("x", "1")]).err().unwrap();
    assert_eq!(e.message, "Unknown parameters: {x=1}");
    let e = build::<SimplePatternTokenizerFactory>(&[("pattern", "(")])
        .err()
        .unwrap();
    assert_eq!(e.java_class(), "IllegalArgumentException");
    let e = build::<SimplePatternTokenizerFactory>(&[
        (
            "pattern",
            "(a|b)*a(a|b)(a|b)(a|b)(a|b)(a|b)(a|b)(a|b)(a|b)(a|b)(a|b)",
        ),
        ("determinizeWorkLimit", "10"),
    ])
    .err()
    .unwrap();
    assert_eq!(e.java_class(), "TooComplexToDeterminizeException");
    let e = build::<PatternTokenizerFactory>(&[]).err().unwrap();
    assert_eq!(
        e.message,
        "Configuration Error: missing parameter 'pattern'"
    );
    let p = build::<PatternTokenizerFactory>(&[("pattern", "a"), ("group", "3")]).unwrap();
    assert!(p.create().is_err());
    let w = build::<WikipediaTokenizerFactory>(&[("tokenOutput", "7")]).unwrap();
    assert!(w.create().is_err());
    for e in [
        build::<StandardTokenizerFactory>(&[("x", "1")]).err(),
        build::<ClassicTokenizerFactory>(&[("x", "1")]).err(),
        build::<UAX29URLEmailTokenizerFactory>(&[("x", "1")]).err(),
        build::<LetterTokenizerFactory>(&[("x", "1")]).err(),
        build::<EdgeNGramTokenizerFactory>(&[("x", "1")]).err(),
        build::<NGramTokenizerFactory>(&[("x", "1")]).err(),
        build::<PathHierarchyTokenizerFactory>(&[("x", "1")]).err(),
        build::<SimplePatternSplitTokenizerFactory>(&[("pattern", "a"), ("x", "1")]).err(),
    ] {
        assert_eq!(e.unwrap().message, "Unknown parameters: {x=1}");
    }
}
