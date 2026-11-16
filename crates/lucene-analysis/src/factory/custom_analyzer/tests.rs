#![cfg(test)]
use super::*;
use crate::factory::{LowerCaseFilterFactory, MapResourceLoader, WhitespaceTokenizerFactory};

fn terms(a: &Analyzer, text: &str) -> Vec<String> {
    let mut ts = a.token_stream("f", text).unwrap();
    ts.reset().unwrap();
    let mut out = Vec::new();
    while ts.increment_token().unwrap() {
        out.push(ts.attributes().term().to_string());
    }
    ts.end().unwrap();
    ts.close().unwrap();
    out
}

#[test]
fn builds_by_name_and_class() {
    let a = CustomAnalyzer::builder()
        .with_default_match_version(Version::LUCENE_10_0_0)
        .unwrap()
        .add_char_filter("htmlStrip", &[])
        .unwrap()
        .with_tokenizer("whitespace", &["maxTokenLen", "10"])
        .unwrap()
        .add_token_filter("lowercase", &[])
        .unwrap()
        .add_token_filter_class::<crate::factory::ASCIIFoldingFilterFactory>(&[])
        .unwrap()
        .with_position_increment_gap(100)
        .unwrap()
        .with_offset_gap(5)
        .unwrap()
        .build()
        .unwrap();
    assert_eq!(terms(&a, "<b>Héllo</b> WORLD"), vec!["hello", "world"]);
    assert_eq!(a.position_increment_gap_for_field("f"), 100);
    assert_eq!(a.offset_gap_for_field("f"), 5);
    assert_eq!(
        a.normalize("f", "<i>ÀB</i>").unwrap(),
        b"<i>ab</i>".to_vec()
    );
    assert_eq!(
        a.to_string(),
        "CustomAnalyzer(org.apache.lucene.analysis.charfilter.HTMLStripCharFilterFactory,\
             org.apache.lucene.analysis.core.WhitespaceTokenizerFactory,\
             org.apache.lucene.analysis.core.LowerCaseFilterFactory,\
             org.apache.lucene.analysis.miscellaneous.ASCIIFoldingFilterFactory)"
    );
    assert_eq!(a.char_filter_factories().len(), 1);
    assert_eq!(a.token_filter_factories().len(), 2);
    assert_eq!(
        a.tokenizer_factory().base().lucene_match_version(),
        Version::LUCENE_10_0_0
    );
    let b = CustomAnalyzer::builder()
        .with_tokenizer_class::<WhitespaceTokenizerFactory>(&[])
        .unwrap()
        .add_char_filter_class::<crate::factory::CJKWidthCharFilterFactory>(&[])
        .unwrap()
        .build()
        .unwrap();
    assert_eq!(terms(b.analyzer(), "Ａ b"), vec!["A", "b"]);
    assert_eq!(b.analyzer().position_increment_gap_for_field("f"), 0);
    assert_eq!(b.into_analyzer().offset_gap_for_field("f"), 1);
}

#[test]
fn javas_builder_errors() {
    let e = CustomAnalyzer::builder().build().err().unwrap();
    assert_eq!(e.message, "You have to set at least a tokenizer.");
    let e = CustomAnalyzer::builder()
        .with_tokenizer("whitespace", &[])
        .unwrap()
        .with_tokenizer("keyword", &[])
        .err()
        .unwrap();
    assert_eq!(e.java_class(), "AlreadySetException");
    let e = CustomAnalyzer::builder()
        .with_tokenizer("whitespace", &[])
        .unwrap()
        .with_default_match_version(Version::LATEST)
        .err()
        .unwrap();
    assert_eq!(e.java_class(), "IllegalStateException");
    let b = CustomAnalyzer::builder()
        .with_default_match_version(Version::LATEST)
        .unwrap();
    assert!(b.with_default_match_version(Version::LATEST).is_err());
    assert!(CustomAnalyzer::builder()
        .with_position_increment_gap(-1)
        .is_err());
    assert!(CustomAnalyzer::builder().with_offset_gap(-1).is_err());
    let b = CustomAnalyzer::builder().with_offset_gap(1).unwrap();
    assert!(b.with_offset_gap(1).is_err());
    let b = CustomAnalyzer::builder()
        .with_position_increment_gap(1)
        .unwrap();
    assert!(b.with_position_increment_gap(1).is_err());
    let e = CustomAnalyzer::builder()
        .with_tokenizer("whitespace", &["x"])
        .err()
        .unwrap();
    assert_eq!(
        e.message,
        "Key-value pairs expected, so the number of params must be even."
    );
    let e = CustomAnalyzer::builder()
        .when("lowercase", &[])
        .err()
        .unwrap();
    assert_eq!(
        e.message,
        "TokenFilterFactory lowercase is not a ConditionalTokenFilterFactory"
    );
    let e = CustomAnalyzer::builder()
        .when_factory(Box::new(
            LowerCaseFilterFactory::from_args(&mut JavaArgs::new()).unwrap(),
        ))
        .endwhen()
        .err()
        .unwrap();
    assert!(e
        .message
        .ends_with("is not a ConditionalTokenFilterFactory"));
    assert!(CustomAnalyzer::builder_with_dir("/nonexistent/dir").is_err());
    assert!(CustomAnalyzer::builder()
        .add_char_filter("nope", &[])
        .is_err());
    assert!(CustomAnalyzer::builder()
        .add_token_filter("nope", &["a"])
        .is_err());
    assert!(CustomAnalyzer::builder()
        .add_char_filter("nope", &["a"])
        .is_err());
}

#[test]
fn conditions() {
    let loader = MapResourceLoader::new().with("p.txt", "FOO\n");
    let a = CustomAnalyzer::builder_with_loader(Box::new(loader))
        .with_tokenizer("whitespace", &[])
        .unwrap()
        .when("protectedTerm", &["protected", "p.txt"])
        .unwrap()
        .add_token_filter("lowercase", &[])
        .unwrap()
        .add_token_filter_class::<crate::factory::ReverseStringFilterFactory>(&[])
        .unwrap()
        .endwhen()
        .unwrap()
        .when_term(|t| t.len() > 3)
        .add_token_filter("uppercase", &[])
        .unwrap()
        .endwhen()
        .unwrap()
        .build()
        .unwrap();
    assert_eq!(terms(&a, "FOO BAR Quux"), vec!["FOO", "rab", "XUUQ"]);
    let e = CustomAnalyzer::builder()
        .when_term(|_| true)
        .add_token_filter("nope", &[])
        .err()
        .unwrap();
    assert_eq!(e.java_class(), "IllegalArgumentException");
    assert!(CustomAnalyzer::builder().when("nope", &[]).is_err());
    assert!(CustomAnalyzer::builder()
        .when("protectedTerm", &["x"])
        .is_err());
    let c = CustomAnalyzer::builder().when_term(|_| true);
    assert!(c.add_token_filter("lowercase", &["x"]).is_err());
    let c = CustomAnalyzer::builder().when_term(|_| true);
    assert!(c
        .add_token_filter_class::<LowerCaseFilterFactory>(&["x"])
        .is_err());
}
