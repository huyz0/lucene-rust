//! The S-expression query grammar of `fixtures/src/GenMixedBooleanScoring.java`
//! (and `BenchRunner.java`'s `sexpr` kind), with the field supplied by the
//! query file's field column:
//!
//! ```text
//!   (t TERM)  (b MSM (+ Q) (# Q) (? Q) (- Q) ...)  (boost F Q)  (const Q)
//!   (dismax TIE Q...)
//! ```

use lucene_search::query::{
    BoostQuery, ConstantScoreQuery, DisjunctionMaxQuery, PrefixQuery, RegexpQuery, TermInSetQuery,
    WildcardQuery,
};
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
        "r" => {
            let f = t.next();
            let min: i64 = t.next().parse().expect("min");
            let max: i64 = t.next().parse().expect("max");
            Clause::PointsRange(lucene_search::query::PointsRangeQuery::new(f, min, max))
        }
        "all" => Clause::MatchAllDocs(lucene_search::query::MatchAllDocsQuery::new(0)),
        "pre" => Clause::Prefix(PrefixQuery::new(field, t.next().into_bytes())),
        "wc" => Clause::Wildcard(WildcardQuery::new(field, t.next().into_bytes())),
        "re" => Clause::Regexp(RegexpQuery::new(field, t.next())),
        "ts" => {
            let mut terms = Vec::new();
            while t.peek() != ")" {
                terms.push(t.next().into_bytes());
            }
            Clause::TermInSet(TermInSetQuery::new(field, terms))
        }
        "p" | "ps" => {
            let slop: u32 = if op == "ps" {
                t.next().parse().expect("slop")
            } else {
                0
            };
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

/// A sort in `GenSortedSearch`'s grammar: comma-separated keys, `score`,
/// `doc` or `FIELD:TYPE:SELECTOR:ORDER:MISSING` (`asc`/`desc`; `last`,
/// `first` or `none`).
pub fn sort(spec: &str) -> Vec<lucene_search::top_field::SortField> {
    use lucene_search::top_field::{Selector, SortField, SortType};
    spec.split(',')
        .map(|k| match k {
            "score" => SortField::score(),
            "doc" => SortField::doc(),
            _ => {
                let p: Vec<&str> = k.split(':').collect();
                let ty = match p[1] {
                    "long" => SortType::Long,
                    "int" => SortType::Int,
                    "double" => SortType::Double,
                    "float" => SortType::Float,
                    other => panic!("sort type {other}"),
                };
                let reverse = p[3] == "desc";
                let high = (p[4] == "last") != reverse;
                let missing = match (p[4], ty) {
                    ("none", _) => 0,
                    (_, SortType::Long) => {
                        if high {
                            i64::MAX
                        } else {
                            i64::MIN
                        }
                    }
                    (_, SortType::Int) => i64::from(if high { i32::MAX } else { i32::MIN }),
                    // doubleToSortableLong / floatToSortableInt of +-infinity.
                    (_, SortType::Double) => {
                        let bits = if high {
                            f64::INFINITY
                        } else {
                            f64::NEG_INFINITY
                        }
                        .to_bits() as i64;
                        bits ^ ((bits >> 63) & 0x7fff_ffff_ffff_ffff)
                    }
                    (_, _) => {
                        let bits = if high {
                            f32::INFINITY
                        } else {
                            f32::NEG_INFINITY
                        }
                        .to_bits() as i32;
                        i64::from(bits ^ ((bits >> 31) & 0x7fff_ffff))
                    }
                };
                SortField {
                    field: p[0].to_string(),
                    ty,
                    reverse,
                    selector: if p[2] == "max" {
                        Selector::Max
                    } else {
                        Selector::Min
                    },
                    missing,
                }
            }
        })
        .collect()
}
