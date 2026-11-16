#![cfg(test)]
use super::*;
use crate::analysis::canned::CannedTokenStream;
use crate::analysis::token::Token;
use lucene_analysis::token_stream::consume;

fn terms<T: TokenStream>(ts: &mut T) -> Vec<(String, i32, i32, i32, i32)> {
    let mut out = Vec::new();
    consume(ts, |a| {
        out.push((
            a.term().to_string(),
            a.position_increment(),
            a.position_length(),
            a.start_offset(),
            a.end_offset(),
        ))
    })
    .unwrap();
    out
}

fn canned() -> CannedTokenStream {
    CannedTokenStream::new(vec![
        Token::new("a", 0, 1),
        Token::new("b", 2, 3),
        Token::with_pos_inc("c", 2, 4, 5),
        Token::new("d", 6, 7),
    ])
}

#[test]
fn random_lookahead_passes_tokens_through() {
    let expected = terms(&mut canned());
    for seed in 0..50 {
        let mut f = MockRandomLookaheadTokenFilter::new(&mut JavaRandom::new(seed), canned());
        assert_eq!(terms(&mut f), expected, "seed {seed}");
        // reset replays the same draws
        assert_eq!(terms(&mut f), expected, "seed {seed}");
    }
}

#[test]
fn graph_filter_inserts_consistent_tokens_and_replays() {
    let mut inserted = 0;
    for seed in 0..50 {
        let mut f = MockGraphTokenFilter::new(&mut JavaRandom::new(seed), canned());
        let first = terms(&mut f);
        assert_eq!(terms(&mut f), first, "seed {seed}");
        inserted += first.len() - 4;
        let originals: Vec<_> = first
            .iter()
            .filter(|t| t.0.len() == 1 && t.0 <= "d".into())
            .collect();
        assert!(originals.len() >= 4);
    }
    assert!(inserted > 0, "no seed inserted a token");
}

#[test]
fn graph_filter_refuses_before_reset_and_after_close() {
    let mut f = MockGraphTokenFilter::new(&mut JavaRandom::new(1), canned());
    assert!(matches!(
        f.increment_token(),
        Err(AnalysisError::IllegalState(_))
    ));
    f.reset().unwrap();
    assert!(f.increment_token().unwrap());
    f.close().unwrap();
    let e = f.increment_token().unwrap_err();
    assert_eq!(
        e,
        AnalysisError::IllegalState("incrementToken called in wrong state!".into())
    );
}

#[test]
fn position_bookkeeping() {
    let mut p: Position<u8> = Position {
        extra: 3,
        ..Position::default()
    };
    p.add(AttributeSource::new());
    let _ = p.next_state();
    p.reset();
    assert_eq!(
        (
            p.input_tokens.len(),
            p.next_read,
            p.start_offset,
            p.end_offset,
            p.extra
        ),
        (0, 0, -1, -1, 0)
    );
}

#[test]
#[should_panic(expected = "prev startOffset=0 vs new startOffset=1")]
fn inconsistent_input_offsets_trip_the_assertion() {
    let ts = CannedTokenStream::new(vec![
        Token::new("a", 0, 1),
        Token::with_pos_inc("b", 0, 1, 1),
    ]);
    let mut la: LookaheadTokenFilter<_> = LookaheadTokenFilter::new(ts);
    la.reset().unwrap();
    while la.peek_token().unwrap() {}
}

#[test]
#[should_panic(expected = "prev endOffset=1 vs new endOffset=2")]
fn inconsistent_end_offsets_trip_the_assertion() {
    let ts = CannedTokenStream::new(vec![
        Token::new("a", 0, 1),
        Token::with_pos_inc("b", 0, 0, 2),
    ]);
    let mut la: LookaheadTokenFilter<_> = LookaheadTokenFilter::new(ts);
    la.reset().unwrap();
    while la.peek_token().unwrap() {}
}

/// Inserts a token whose end offset does not match the lookahead's.
struct BadInsert;
impl<I: TokenStream> LookaheadHooks<I, ()> for BadInsert {
    fn after_position(&mut self, la: &mut LookaheadTokenFilter<I>) -> Result<(), AnalysisError> {
        if la.output_pos == 0 && !la.end {
            la.peek_token()?;
            la.insert_token();
            let atts = la.attributes_mut();
            atts.clear_attributes();
            atts.set_term("x");
            atts.set_position_increment(0)?;
            atts.set_offset(0, 99)?;
        }
        Ok(())
    }
}

#[test]
#[should_panic(expected = "offsetAtt.endOffset=99 vs expected=1")]
fn an_inconsistent_insert_trips_the_assertion() {
    let mut la: LookaheadTokenFilter<_> = LookaheadTokenFilter::new(canned());
    la.reset().unwrap();
    let mut hooks = BadInsert;
    while la.next_token(&mut hooks).unwrap() {}
}

#[test]
fn no_hooks_is_a_pass_through() {
    let expected = terms(&mut canned());
    let mut la: LookaheadTokenFilter<_> = LookaheadTokenFilter::new(canned());
    la.reset().unwrap();
    let mut got = 0;
    while la.next_token(&mut NoHooks).unwrap() {
        got += 1;
    }
    assert_eq!(got, expected.len());
}
