//! Lucene's `TestToken`, ported.
//!
//! `testTokenAttributeFactory` is not ported: it checks that
//! `Token.TOKEN_ATTRIBUTE_FACTORY` makes a `Token` implement a stream's
//! attributes, and the port's attributes are a fixed struct with no
//! factory (see `analysis::token`). `assertCloneIsEqual` and
//! `assertCopyIsEqual` compare hash codes as well; here equal values hash
//! equally by `Hash`'s contract, checked with the std hasher.

use std::collections::hash_map::DefaultHasher;
use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};

use lucene_analysis::attributes::AttributeSource;
use lucene_test_framework::analysis::Token;

fn hash(t: &Token) -> u64 {
    let mut h = DefaultHasher::new();
    t.hash(&mut h);
    h.finish()
}

fn assert_clone_is_equal(t: &Token) -> Token {
    let clone = t.clone();
    assert_eq!(&clone, t, "Clone must be equal");
    assert_eq!(hash(&clone), hash(t), "Clone's hashcode must be equal");
    clone
}

fn assert_copy_is_equal(t: &Token) -> Token {
    let mut atts = AttributeSource::new();
    t.copy_to(&mut atts).unwrap();
    let copy = Token::from_attributes(&atts);
    assert_eq!(&copy, t, "Copied instance must be equal");
    assert_eq!(
        hash(&copy),
        hash(t),
        "Copied instance's hashcode must be equal"
    );
    copy
}

#[test]
fn test_ctor() {
    let t = Token::new("hello", 0, 0);
    assert_eq!(t.start_offset(), 0);
    assert_eq!(t.end_offset(), 0);
    assert_eq!(t.position_increment(), 1);
    assert_eq!(t.position_length(), 1);
    assert_eq!(t.to_string(), "hello");
    assert_eq!(t.token_type(), "word");
    assert_eq!(t.flags(), 0);
    assert!(t.payload().is_none());
}

#[test]
fn test_clone() {
    let mut t = Token::default();
    t.set_offset(0, 5).unwrap();
    t.set_term("hello");
    let copy = assert_clone_is_equal(&t);
    assert_eq!(t.to_string(), copy.to_string());
    t.set_payload(Some(vec![1, 2, 3, 4]));
    let copy = assert_clone_is_equal(&t);
    assert_eq!(copy.payload(), Some(&[1u8, 2, 3, 4][..]));
}

#[test]
fn test_copy_to() {
    let t = Token::default();
    let copy = assert_copy_is_equal(&t);
    assert_eq!(t.to_string(), "");
    assert_eq!(copy.to_string(), "");
    let mut t = Token::default();
    t.set_offset(0, 5).unwrap();
    t.set_term("hello");
    let copy = assert_copy_is_equal(&t);
    assert_eq!(t.to_string(), copy.to_string());
    t.set_payload(Some(vec![1, 2, 3, 4]));
    let copy = assert_copy_is_equal(&t);
    assert_eq!(copy.payload(), Some(&[1u8, 2, 3, 4][..]));
}

#[test]
fn test_attribute_reflection() {
    let mut t = Token::new("foobar", 6, 22);
    t.set_flags(8);
    t.set_position_increment(3);
    t.set_position_length(11);
    t.set_term_frequency(42);
    let mut got = BTreeMap::new();
    t.reflect_with(&mut |class, key, value| {
        assert!(got
            .insert(format!("{class}#{key}"), value.to_string())
            .is_none());
    });
    let p = "org.apache.lucene.analysis.tokenattributes.";
    let want: BTreeMap<String, String> = [
        ("CharTermAttribute#term", "foobar"),
        ("TermToBytesRefAttribute#bytes", "[66 6f 6f 62 61 72]"),
        ("OffsetAttribute#startOffset", "6"),
        ("OffsetAttribute#endOffset", "22"),
        ("PositionIncrementAttribute#positionIncrement", "3"),
        ("PositionLengthAttribute#positionLength", "11"),
        ("PayloadAttribute#payload", "null"),
        ("TypeAttribute#type", "word"),
        ("FlagsAttribute#flags", "8"),
        ("TermFrequencyAttribute#termFrequency", "42"),
    ]
    .into_iter()
    .map(|(k, v)| (format!("{p}{k}"), v.to_string()))
    .collect();
    assert_eq!(got, want);
}
