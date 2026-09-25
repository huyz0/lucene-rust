//! The S-expression query grammar of `fixtures/src/GenMixedBooleanScoring.java`
//! (and `BenchRunner.java`'s `sexpr` kind), with the field supplied by the
//! query file's field column:
//!
//! ```text
//!   (t TERM)  (b MSM (+ Q) (# Q) (? Q) (- Q) ...)  (boost F Q)  (const Q)
//!   (dismax TIE Q...)
//! ```

use lucene_search::query::{BoostQuery, ConstantScoreQuery, DisjunctionMaxQuery};
use lucene_search::{BooleanQuery, Clause, PhraseQuery, TermQuery};

struct Tokens {
    toks: Vec<String>,
    at: usize,
}

impl Tokens {
    fn next(&mut self) -> String {
        self.at += 1;
        self.toks[self.at - 1].clone()
    }
    fn peek(&self) -> &str {
        &self.toks[self.at]
    }
    fn expect(&mut self, t: &str) {
        let got = self.next();
        assert_eq!(got, t, "sexpr: expected {t}");
    }
}

pub fn parse(field: &str, text: &str) -> Clause {
    let mut t = Tokens {
        toks: text
            .replace('(', " ( ")
            .replace(')', " ) ")
            .split_whitespace()
            .map(str::to_string)
            .collect(),
        at: 0,
    };
    clause(field, &mut t)
}

fn clause(field: &str, t: &mut Tokens) -> Clause {
    t.expect("(");
    let op = t.next();
    let q = match op.as_str() {
        // `(t word)` searches the query's field; `(t title:word)` another one.
        "t" => {
            let w = t.next();
            let (f, w) = w.split_once(':').unwrap_or((field, &w));
            Clause::Term(TermQuery::new(f, w.as_bytes().to_vec()))
        }
        "p" | "ps" => {
            let slop: u32 = if op == "ps" { t.next().parse().expect("slop") } else { 0 };
            let mut words = Vec::new();
            while t.peek() != ")" {
                words.push(t.next());
            }
            Clause::Phrase(PhraseQuery::new(field, words).with_slop(slop))
        }
        "boost" => {
            let f: f32 = t.next().parse().expect("boost");
            BoostQuery::new(clause(field, t), f).into()
        }
        "const" => ConstantScoreQuery::new(clause(field, t), 1.0).into(),
        "dismax" => {
            let tie: f32 = t.next().parse().expect("tie");
            let mut ds = Vec::new();
            while t.peek() == "(" {
                ds.push(clause(field, t));
            }
            DisjunctionMaxQuery::new(ds, tie).into()
        }
        "b" => {
            let mut b = BooleanQuery::new();
            b.minimum_should_match = t.next().parse().expect("msm");
            while t.peek() == "(" {
                t.expect("(");
                let occur = t.next();
                let c = clause(field, t);
                match occur.as_str() {
                    "+" => b.must.push(c),
                    "#" => b.filter.push(c),
                    "?" => b.should.push(c),
                    "-" => b.must_not.push(c),
                    other => panic!("sexpr: bad occur {other}"),
                }
                t.expect(")");
            }
            Clause::Boolean(Box::new(b))
        }
        other => panic!("sexpr: unknown op {other}"),
    };
    t.expect(")");
    q
}

/// A boolean as is; anything else as the lone `MUST` clause of one.
pub fn root(clause: Clause) -> BooleanQuery {
    match clause {
        Clause::Boolean(b) => *b,
        other => {
            let mut b = BooleanQuery::new();
            b.must.push(other);
            b
        }
    }
}
