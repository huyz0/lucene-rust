#![cfg(test)]
use super::*;
use crate::token_stream::consume;

struct Canned {
    atts: AttributeSource,
    tokens: Vec<(&'static str, i32, i32, i32)>,
    upto: usize,
    end: (i32, i32),
}

impl TokenStream for Canned {
    /// A source, not a wrapper: no conditional wrapper below it.
    fn conditional_root(&mut self) -> Option<&mut dyn std::any::Any> {
        None
    }
    fn attributes(&self) -> &AttributeSource {
        &self.atts
    }
    fn attributes_mut(&mut self) -> &mut AttributeSource {
        &mut self.atts
    }
    fn increment_token(&mut self) -> Result<bool, AnalysisError> {
        self.atts.clear_attributes();
        let Some(&(t, inc, len, end)) = self.tokens.get(self.upto) else {
            return Ok(false);
        };
        self.upto += 1;
        self.atts.set_term(t);
        self.atts.set_position_increment(inc)?;
        self.atts.set_position_length(len)?;
        self.atts.set_offset(0, end)?;
        Ok(true)
    }
    fn reset(&mut self) -> Result<(), AnalysisError> {
        self.upto = 0;
        Ok(())
    }
    fn end(&mut self) -> Result<(), AnalysisError> {
        self.atts.end_attributes();
        self.atts.set_position_increment(self.end.0)?;
        self.atts.set_offset(self.end.1, self.end.1)
    }
}

fn canned(tokens: Vec<(&'static str, i32, i32, i32)>, end: (i32, i32)) -> Canned {
    Canned {
        atts: AttributeSource::new(),
        tokens,
        upto: 0,
        end,
    }
}

/// Every accepted string, labels as chars (small acyclic automata only).
fn strings(a: &Automaton) -> Vec<String> {
    fn walk(a: &Automaton, s: i32, cur: &mut String, out: &mut Vec<String>) {
        if a.is_accept(s) {
            out.push(cur.clone());
        }
        for t in &a.get_sorted_transitions()[s as usize] {
            for l in t.min..=t.max {
                let c = match l {
                    POS_SEP => '|',
                    HOLE => '_',
                    l => char::from_u32(l as u32).unwrap(),
                };
                cur.push(c);
                walk(a, t.dest, cur, out);
                cur.pop();
            }
        }
    }
    let mut out = Vec::new();
    walk(a, 0, &mut String::new(), &mut out);
    out.sort();
    out
}

#[test]
fn linear_stream() {
    let mut ts = canned(vec![("ab", 1, 1, 2), ("c", 1, 1, 4)], (0, 4));
    let a = token_stream_to_automaton(&mut ts).unwrap();
    assert_eq!(strings(&a), vec!["ab|c"]);
}

#[test]
fn synonyms_holes_and_trailing_holes() {
    // "wifi" spanning "wi fi", then a hole (stopword), then "x", then two
    // trailing removed positions.
    let mut ts = canned(
        vec![
            ("wifi", 1, 2, 5),
            ("wi", 0, 1, 2),
            ("fi", 1, 1, 5),
            ("x", 2, 1, 9),
        ],
        (2, 12),
    );
    let a = token_stream_to_automaton(&mut ts).unwrap();
    assert_eq!(strings(&a), vec!["wifi|_|x|_|_", "wi|fi|_|x|_|_"]);
    let mut conv = TokenStreamToAutomaton::default();
    conv.set_preserve_position_increments(false);
    let a = conv.to_automaton(&mut ts).unwrap();
    assert_eq!(strings(&a), vec!["wifi|x", "wi|fi|x"]);
}

#[test]
fn final_offset_gap_unicode_arcs_and_change_token() {
    let mut ts = canned(vec![("é", 1, 1, 1)], (0, 5));
    let a = token_stream_to_automaton(&mut ts).unwrap();
    // two UTF-8 byte arcs
    assert_eq!(a.get_num_states(), 3);
    let mut conv = TokenStreamToAutomaton::new();
    conv.set_unicode_arcs(true);
    conv.set_final_offset_gap_as_hole(true);
    let a = conv.to_automaton(&mut ts).unwrap();
    assert_eq!(strings(&a), vec!["é|_"]);
    let mut conv = TokenStreamToAutomaton::new();
    conv.set_change_token(|b| b.iter().map(|&x| x.to_ascii_uppercase()).collect());
    let mut ts = canned(vec![("ab", 1, 1, 1)], (0, 2));
    let a = conv.to_automaton(&mut ts).unwrap();
    assert_eq!(strings(&a), vec!["AB"]);
    let mut ts = canned(vec![("a", 0, 1, 1)], (0, 2));
    assert!(conv.to_automaton(&mut ts).is_err());
}

#[test]
fn builder_merges_adjacent_ranges_and_sorts() {
    let mut b = AutomatonBuilder::new();
    let s0 = b.create_state();
    let s1 = b.create_state();
    let s2 = b.create_state();
    b.add_transition_label(s0, s1, b'c' as i32);
    b.add_transition_label(s0, s1, b'a' as i32);
    b.add_transition_label(s0, s1, b'b' as i32);
    b.add_transition_label(s0, s2, b'a' as i32);
    b.add_transition(s0, s1, b'x' as i32, b'z' as i32);
    b.set_accept(s1, true);
    assert!(b.is_accept(s1));
    assert_eq!(b.get_num_states(), 3);
    let a = b.finish();
    let got: Vec<(i32, i32, i32)> = a.get_sorted_transitions()[s0 as usize]
        .iter()
        .map(|t| (t.dest, t.min, t.max))
        .collect();
    assert_eq!(got, vec![(2, 97, 97), (1, 97, 99), (1, 120, 122)]);
    assert_eq!(a.get_total_num_transitions(), 3);
}

#[test]
fn automaton_round_trips_to_a_token_graph() {
    // 0 -a-> 1 -b-> 2(accept); 0 -c-> 2
    let mut b = AutomatonBuilder::new();
    for _ in 0..3 {
        b.create_state();
    }
    b.add_transition_label(0, 1, b'a' as i32);
    b.add_transition_label(1, 2, b'b' as i32);
    b.add_transition_label(0, 2, b'c' as i32);
    b.set_accept(2, true);
    let a = b.finish();
    let mut ts = automaton_to_token_stream(&a).unwrap();
    let mut toks = Vec::new();
    let end = consume(&mut ts, |x| {
        toks.push((
            x.term().to_string(),
            x.position_increment(),
            x.position_length(),
            x.start_offset(),
            x.end_offset(),
        ))
    })
    .unwrap();
    assert_eq!(
        toks,
        vec![
            ("a".to_string(), 1, 1, 0, 1),
            ("c".to_string(), 0, 2, 0, 2),
            ("b".to_string(), 1, 1, 1, 2),
        ]
    );
    assert_eq!((end.start_offset(), end.position_increment()), (2, 0));
}

#[test]
fn cycles_are_rejected() {
    let mut b = AutomatonBuilder::new();
    b.create_state();
    b.create_state();
    b.add_transition_label(0, 1, 1);
    b.add_transition_label(1, 0, 1);
    let err = automaton_to_token_stream(&b.finish()).err().unwrap();
    assert!(err.to_string().contains("Start node has incoming edges"));
    let mut b = AutomatonBuilder::new();
    for _ in 0..3 {
        b.create_state();
    }
    b.add_transition_label(0, 1, 1);
    b.add_transition_label(1, 2, 1);
    b.add_transition_label(2, 1, 1);
    let err = automaton_to_token_stream(&b.finish()).err().unwrap();
    assert!(err.to_string().contains("Cycle found"));
    // intermediate accept adds an edge to the terminal layer
    let mut b = AutomatonBuilder::new();
    for _ in 0..3 {
        b.create_state();
    }
    b.add_transition_label(0, 1, b'a' as i32);
    b.add_transition_label(1, 2, b'b' as i32);
    b.set_accept(1, true);
    b.set_accept(2, true);
    let mut ts = automaton_to_token_stream(&b.finish()).unwrap();
    let mut n = 0;
    consume(&mut ts, |_| n += 1).unwrap();
    assert_eq!(n, 3);
    assert!(automaton_to_token_stream(&Automaton::default()).is_ok());
}
