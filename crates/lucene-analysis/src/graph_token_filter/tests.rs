#![cfg(test)]
use super::*;
use crate::token_stream::consume;

/// A canned stream: (term, posInc, posLen).
struct Canned {
    atts: AttributeSource,
    tokens: Vec<(&'static str, i32, i32)>,
    upto: usize,
    end_inc: i32,
}

impl Canned {
    fn new(tokens: Vec<(&'static str, i32, i32)>, end_inc: i32) -> Self {
        Canned {
            atts: AttributeSource::new(),
            tokens,
            upto: 0,
            end_inc,
        }
    }
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
        let Some(&(t, inc, len)) = self.tokens.get(self.upto) else {
            return Ok(false);
        };
        let off = self.upto as i32;
        self.upto += 1;
        self.atts.set_term(t);
        self.atts.set_position_increment(inc)?;
        self.atts.set_position_length(len)?;
        self.atts.set_offset(off, off + 1)?;
        Ok(true)
    }
    fn reset(&mut self) -> Result<(), AnalysisError> {
        self.upto = 0;
        Ok(())
    }
    fn end(&mut self) -> Result<(), AnalysisError> {
        self.atts.end_attributes();
        self.atts.set_position_increment(self.end_inc)?;
        let n = self.tokens.len() as i32;
        self.atts.set_offset(n, n)
    }
}

/// Emits, per base token, every path of up to `depth` tokens joined by
/// `_` -- the shape of a graph-aware shingle filter.
struct Paths<I> {
    graph: GraphTokenFilter<I>,
    depth: usize,
    pending: Vec<String>,
}

impl<I: TokenStream> TokenFilter for Paths<I> {
    type Input = GraphTokenFilter<I>;
    fn input(&self) -> &Self::Input {
        &self.graph
    }
    fn input_mut(&mut self) -> &mut Self::Input {
        &mut self.graph
    }
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        loop {
            if let Some(p) = self.pending.pop() {
                self.graph.attributes_mut().set_term(&p);
                return Ok(true);
            }
            if !self.graph.increment_base_token()? {
                return Ok(false);
            }
            let mut paths = Vec::new();
            loop {
                let mut path = vec![self.graph.attributes().term().to_string()];
                while path.len() < self.depth && self.graph.increment_graph_token()? {
                    path.push(self.graph.attributes().term().to_string());
                }
                paths.push(path.join("_"));
                if !self.graph.increment_graph()? {
                    break;
                }
            }
            paths.reverse();
            self.pending = paths;
        }
    }
}

fn terms(ts: &mut dyn TokenStream) -> (Vec<String>, AttributeSource) {
    let mut out = Vec::new();
    let end = consume(ts, |a| out.push(a.term().to_string())).unwrap();
    (out, end)
}

#[test]
fn walks_every_path_through_a_graph() {
    // "wi fi network" with "wifi" spanning two positions.
    let canned = Canned::new(
        vec![
            ("wifi", 1, 2),
            ("wi", 0, 1),
            ("fi", 1, 1),
            ("network", 1, 1),
        ],
        3,
    );
    let mut f = Paths {
        graph: GraphTokenFilter::new(canned),
        depth: 2,
        pending: vec![],
    };
    let (t, end) = terms(&mut f);
    assert_eq!(t, vec!["wifi_network", "wi_fi", "fi_network", "network"]);
    assert_eq!(end.position_increment(), 3);
    assert_eq!(end.end_offset(), 4);
    assert_eq!(f.graph.trailing_positions(), 3);
    assert!(f.graph.cached_token_count() <= 4);
    // end() again replays the stored trailing values.
    f.graph.end_filter().unwrap();
    assert_eq!(f.graph.attributes().position_increment(), 3);
    // reusable after reset
    let (t2, _) = terms(&mut f);
    assert_eq!(t2.len(), 4);
}

#[test]
fn plain_use_passes_tokens_through() {
    let canned = Canned::new(vec![("a", 1, 1), ("b", 1, 1)], 0);
    let mut g = GraphTokenFilter::new(canned);
    let (t, end) = terms(&mut g);
    assert_eq!(t, vec!["a", "b"]);
    assert_eq!(end.position_increment(), 0);
    // no graph before a base token
    let canned = Canned::new(vec![], 0);
    let mut g = GraphTokenFilter::new(canned);
    g.reset_filter().unwrap();
    assert!(!g.increment_graph().unwrap());
    assert!(!g.increment_graph_token().unwrap());
    assert!(!g.increment_base_token().unwrap());
}

#[test]
fn too_many_cached_tokens_is_an_error() {
    // A base token spanning 200 positions forces 200 look-ahead tokens.
    let mut toks = vec![("big", 1, 200)];
    toks.extend(std::iter::repeat_n(("x", 1, 1), 200));
    let mut g = GraphTokenFilter::new(Canned::new(toks, 0));
    g.reset_filter().unwrap();
    assert!(g.increment_base_token().unwrap());
    let err = g.increment_graph_token().unwrap_err();
    assert_eq!(
        err.to_string(),
        "illegal state: Too many cached tokens (> 100)"
    );
}

#[test]
fn too_many_graph_paths_is_an_error() {
    // 32 stacked tokens at each of 3 positions: 32^2 paths of depth 3
    // (and 96 cached tokens, under the cache limit).
    let mut toks = Vec::new();
    for _pos in 0..3 {
        for k in 0..32 {
            toks.push(("t", i32::from(k == 0), 1));
        }
    }
    let mut g = GraphTokenFilter::new(Canned::new(toks, 0));
    g.reset_filter().unwrap();
    assert!(g.increment_base_token().unwrap());
    let mut err = None;
    'outer: for _ in 0..5000 {
        while g.increment_graph_token().unwrap() {}
        match g.increment_graph() {
            Ok(true) => {}
            Ok(false) => break 'outer,
            Err(e) => {
                err = Some(e);
                break 'outer;
            }
        }
    }
    let err = err.expect("more than 1000 paths");
    assert_eq!(
        err.to_string(),
        "illegal state: Too many graph paths (> 1000)"
    );
}
