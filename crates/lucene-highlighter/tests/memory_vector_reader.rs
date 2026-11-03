//! MemoryIndex's reader built straight off a term vector against the one
//! its replay through `TokenStreamFromTermVector` builds.

use lucene_codecs::term_vectors::{TermVectorField, TermVectorTerm};
use lucene_highlighter::TokenStreamFromTermVector;
use lucene_index::document::{FieldToken, FieldTokens};
use lucene_search::memory::MemoryIndex;

fn token(text: &str, inc: i32, start: i32, end: i32) -> FieldToken {
    FieldToken {
        position_increment: inc,
        ..FieldToken::new(text.as_bytes().to_vec(), start, end)
    }
}

fn stream(tokens: Vec<FieldToken>, final_offset: i32) -> FieldTokens {
    FieldTokens {
        tokens,
        final_position_increment: 0,
        final_offset,
        end_attributes: None,
    }
}

// ---------------------------------------------------------------------------
// The reader off a term vector
// ---------------------------------------------------------------------------

fn tv_term(term: &[u8], positions: &[i32], offsets: &[(i32, i32)]) -> TermVectorTerm {
    TermVectorTerm {
        term: term.to_vec(),
        freq: i32::try_from(positions.len()).unwrap(),
        positions: Some(positions.to_vec()),
        start_offsets: Some(offsets.iter().map(|o| o.0).collect()),
        end_offsets: Some(offsets.iter().map(|o| o.1).collect()),
        payloads: None,
    }
}

fn tv_field(terms: Vec<TermVectorTerm>) -> TermVectorField {
    TermVectorField {
        field_number: 3,
        has_positions: true,
        has_offsets: true,
        has_payloads: false,
        terms,
    }
}

/// The reader the stream replay builds, as text: `None` when the stream
/// refuses the vector.
fn replayed(vector: &TermVectorField) -> Option<String> {
    use lucene_analysis::TokenStream;
    let mut ts = TokenStreamFromTermVector::new(vector, -1).ok()?;
    ts.reset().ok()?;
    let mut tokens = Vec::new();
    while ts.increment_token().ok()? {
        let a = ts.attributes();
        tokens.push(token(
            std::str::from_utf8(a.term_bytes()).ok()?,
            a.position_increment(),
            a.start_offset(),
            a.end_offset(),
        ));
    }
    ts.end().ok()?;
    let final_offset = ts.attributes().end_offset();
    let mut mi = MemoryIndex::with_options(true, false);
    mi.add_field_tokens("tv", &stream(tokens, final_offset), 0, 1)
        .ok()?;
    let r = mi.create_reader().ok()?;
    Some(r.debug_form())
}

fn direct(vector: &TermVectorField) -> Option<String> {
    MemoryIndex::vector_reader("tv", vector)
        .unwrap()
        .map(|r| r.debug_form())
}

/// The reader built off a term vector is the one its replay as a token
/// stream builds -- terms, postings, statistics and norm -- and every case
/// it does not cover is left to the replay.
#[test]
fn a_vector_reader_is_the_replayed_reader() {
    let long_span = (0, i32::from(i16::MAX) + 100);
    let plain = [
        // overlapping tokens (two terms at one position), gaps, repeats
        vec![
            tv_term(b"brown", &[1, 5], &[(4, 9), (30, 35)]),
            tv_term(b"fast", &[1], &[(4, 9)]),
            tv_term(b"quick", &[0, 0, 7], &[(0, 3), (0, 5), (40, 45)]),
        ],
        vec![tv_term("ölmühle".as_bytes(), &[2], &[(1, 8)])],
        // a span longer than Short.MAX_VALUE is cut
        vec![tv_term(b"x", &[0], &[long_span])],
    ];
    for terms in plain {
        let v = tv_field(terms);
        assert!(direct(&v).is_some(), "{v:?}");
        assert_eq!(direct(&v), replayed(&v), "{v:?}");
    }
    let one = || tv_field(vec![tv_term(b"a", &[0], &[(0, 1)])]);
    let mut no_positions = one();
    no_positions.has_positions = false;
    no_positions.terms[0].positions = None;
    let mut no_offsets = one();
    no_offsets.has_offsets = false;
    let mut zero_freq = one();
    zero_freq.terms[0].freq = 0;
    let mut short = one();
    short.terms[0].freq = 2;
    let refused = [
        no_positions,
        no_offsets,
        zero_freq,
        short,
        tv_field(Vec::new()),
        tv_field(vec![tv_term(b"a", &[-1], &[(0, 1)])]),
        tv_field(vec![tv_term(b"a", &[0], &[(-1, 1)])]),
        tv_field(vec![tv_term(b"a", &[0], &[(5, 1)])]),
        tv_field(vec![tv_term(b"a", &[3, 1], &[(0, 1), (2, 3)])]),
        tv_field(vec![tv_term(b"a", &[1, 1], &[(4, 5), (2, 3)])]),
        tv_field(vec![tv_term(&[0xff, 0xfe], &[0], &[(0, 1)])]),
        tv_field(vec![tv_term(&[b'a'; 40_000], &[0], &[(0, 1)])]),
    ];
    for v in refused {
        assert!(
            direct(&v).is_none(),
            "{:?}",
            v.terms.first().map(|t| t.freq)
        );
    }
}
