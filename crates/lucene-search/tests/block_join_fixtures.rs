//! M10 T10.2, differentially against Lucene 10.5.0: `fixtures/src/GenBlockJoin.java`.
//!
//! A four-segment index of two-level blocks (grandchildren, children,
//! parents; empty and single-child blocks), with whole blocks deleted in two
//! segments and lone children, grandchildren and parents in a third. Every
//! recorded search -- `ToParentBlockJoinQuery` in each score mode,
//! `ToChildBlockJoinQuery`, `ParentChildrenBlockJoinQuery`,
//! `ParentsChildrenBlockJoinQuery`, two levels deep, alone (the bulk scorer)
//! and inside booleans (the scorer), boosted and constant-scored -- must
//! return Lucene's hits with the same score bits, or fail where Lucene threw;
//! every `ToParentBlockJoinSortField` search the same documents and sort
//! values.

// Test fixtures' own arithmetic -- see `docs/arithmetic-gate.md`'s "Test code".
#![allow(clippy::arithmetic_side_effects)]

use std::collections::HashMap;
use std::sync::Arc;

use lucene_search::directory_reader::DirectoryReader;
use lucene_search::field_norms::FieldNorms;
use lucene_search::index_searcher::{IndexSearcher, SegmentNorms};
use lucene_search::join::{
    BitSetProducer, JoinMissing, JoinSortType, ParentChildrenBlockJoinQuery,
    ParentsChildrenBlockJoinQuery, QueryBitSetProducer, ScoreMode, ToChildBlockJoinQuery,
    ToParentBlockJoinQuery, ToParentBlockJoinSortField,
};
use lucene_search::query::{BooleanQuery, BoostQuery, Clause, ConstantScoreQuery, TermQuery};
use lucene_store::FsDirectory;

fn data() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/block_join")
}

fn term(field: &str, value: &str) -> Clause {
    Clause::Term(TermQuery::new(field, value.as_bytes().to_vec()))
}

fn bool_of(clauses: Vec<(&str, Clause)>) -> BooleanQuery {
    let mut b = BooleanQuery::default();
    for (occ, c) in clauses {
        match occ {
            "must" => b.must.push(c),
            "should" => b.should.push(c),
            "filter" => b.filter.push(c),
            "not" => b.must_not.push(c),
            other => panic!("occur {other}"),
        }
    }
    b
}

/// The four parent and child filters of `GenBlockJoin`.
struct Filters(HashMap<&'static str, Arc<dyn BitSetProducer>>);

impl Filters {
    fn new() -> Self {
        let q = |c: Clause| -> Arc<dyn BitSetProducer> {
            Arc::new(QueryBitSetProducer::new(BooleanQuery {
                must: vec![c],
                ..Default::default()
            }))
        };
        let mut m: HashMap<&'static str, Arc<dyn BitSetProducer>> = HashMap::new();
        m.insert("P0", q(term("type", "parent")));
        m.insert(
            "P1",
            Arc::new(QueryBitSetProducer::new(bool_of(vec![
                ("should", term("type", "parent")),
                ("should", term("type", "child")),
            ]))),
        );
        m.insert("KIDS", q(term("type", "child")));
        m.insert("GRANDS", q(term("type", "grand")));
        Filters(m)
    }

    fn get(&self, name: &str) -> Arc<dyn BitSetProducer> {
        Arc::clone(self.0.get(name).unwrap_or_else(|| panic!("filter {name}")))
    }
}

/// A recursive-descent reader of `GenBlockJoin`'s query specs.
struct Parser<'s> {
    s: &'s str,
    at: usize,
    filters: &'s Filters,
}

impl<'s> Parser<'s> {
    fn peek(&self) -> u8 {
        self.s.as_bytes()[self.at]
    }

    fn eat(&mut self, c: u8) {
        assert_eq!(self.peek(), c, "at {} of {}", self.at, self.s);
        self.at += 1;
    }

    /// Up to the next `,`, `(` or `)`.
    fn word(&mut self) -> &'s str {
        let start = self.at;
        while !matches!(self.peek(), b',' | b'(' | b')') {
            self.at += 1;
        }
        &self.s[start..self.at]
    }

    fn mode(name: &str) -> ScoreMode {
        match name {
            "None" => ScoreMode::None,
            "Avg" => ScoreMode::Avg,
            "Max" => ScoreMode::Max,
            "Total" => ScoreMode::Total,
            "Min" => ScoreMode::Min,
            other => panic!("score mode {other}"),
        }
    }

    fn query(&mut self) -> Clause {
        let head = self.word();
        self.eat(b'(');
        let q = match head {
            "t" => {
                let f = self.word();
                self.eat(b',');
                let v = self.word();
                term(f, v)
            }
            "bool" => {
                let mut clauses = Vec::new();
                loop {
                    let start = self.at;
                    while self.peek() != b':' {
                        self.at += 1;
                    }
                    let occ = &self.s[start..self.at];
                    self.at += 1;
                    clauses.push((occ, self.query()));
                    if self.peek() == b')' {
                        break;
                    }
                    self.eat(b',');
                }
                Clause::Boolean(Box::new(bool_of(clauses)))
            }
            "boost" => {
                let f: f32 = self.word().parse().expect("boost");
                self.eat(b',');
                Clause::Boost(Box::new(BoostQuery::new(self.query(), f)))
            }
            "cs" => Clause::ConstantScore(Box::new(ConstantScoreQuery::new(self.query(), 1.0))),
            "tp" => {
                let mode = Self::mode(self.word());
                self.eat(b',');
                let f = self.filters.get(self.word());
                self.eat(b',');
                ToParentBlockJoinQuery::new(self.query(), f, mode).into()
            }
            "tc" => {
                let f = self.filters.get(self.word());
                self.eat(b',');
                ToChildBlockJoinQuery::new(self.query(), f).into()
            }
            "pc" => {
                let f = self.filters.get(self.word());
                self.eat(b',');
                let doc: i32 = self.word().parse().expect("doc");
                self.eat(b',');
                ParentChildrenBlockJoinQuery::new(f, self.query(), doc).into()
            }
            "pcs" => {
                let f = self.filters.get(self.word());
                self.eat(b',');
                let limit: i32 = self.word().parse().expect("limit");
                self.eat(b',');
                let parent = self.query();
                self.eat(b',');
                let child = self.query();
                ParentsChildrenBlockJoinQuery::new(f, parent, child, limit)
                    .expect("limit")
                    .into()
            }
            other => panic!("query head {other} in {}", self.s),
        };
        self.eat(b')');
        q
    }
}

fn parse(spec: &str, filters: &Filters) -> Clause {
    let mut p = Parser {
        s: spec,
        at: 0,
        filters,
    };
    let q = p.query();
    assert_eq!(p.at, spec.len(), "trailing input in {spec}");
    q
}

fn missing(spec: &str) -> Option<JoinMissing> {
    let (kind, rest) = spec.split_at(1);
    match (spec, kind) {
        ("null", _) => None,
        ("first", _) => Some(JoinMissing::StringFirst),
        ("last", _) => Some(JoinMissing::StringLast),
        (_, "l") => Some(JoinMissing::Long(rest.parse().unwrap())),
        (_, "i") => Some(JoinMissing::Int(rest.parse().unwrap())),
        (_, "f") => Some(JoinMissing::Float(f32::from_bits(
            u32::from_str_radix(rest, 16).unwrap(),
        ))),
        (_, "d") => Some(JoinMissing::Double(f64::from_bits(
            u64::from_str_radix(rest, 16).unwrap(),
        ))),
        _ => panic!("missing value {spec}"),
    }
}

/// A sort value as Lucene printed it, in this port's encoding: the
/// comparable long (`floatToSortableInt`, `doubleToSortableLong`), or a term.
fn expected_value(v: &str) -> (Option<i64>, Option<Option<Vec<u8>>>) {
    let (kind, rest) = v.split_at(1);
    match (v, kind) {
        ("null", _) => (None, Some(None)),
        (_, "l") | (_, "i") => (Some(rest.parse().unwrap()), None),
        (_, "f") => {
            let bits = u32::from_str_radix(rest, 16).unwrap() as i32;
            (Some(i64::from(bits ^ ((bits >> 31) & 0x7fff_ffff))), None)
        }
        (_, "d") => {
            let bits = u64::from_str_radix(rest, 16).unwrap() as i64;
            (Some(bits ^ ((bits >> 63) & 0x7fff_ffff_ffff_ffff)), None)
        }
        (_, "s") => (None, Some(Some(rest.as_bytes().to_vec()))),
        _ => panic!("sort value {v}"),
    }
}

#[test]
fn block_join_queries_match_lucene_bit_for_bit() {
    let dir = data();
    let text = std::fs::read_to_string(dir.join("searches.tsv"))
        .expect("run scripts/gen-fixtures.sh --only GenBlockJoin");
    let reader = DirectoryReader::open(&FsDirectory::open(dir.join("index"))).unwrap();
    assert_eq!(reader.segment_readers().len(), 4);
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let owned = reader.field_norms_by_field(&["body".to_string()]);
    let norms: Vec<SegmentNorms<'_, '_>> = owned
        .iter()
        .map(|m: &HashMap<String, FieldNorms<'_>>| Some(m))
        .collect();
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    let filters = Filters::new();

    let (mut cases, mut errors, mut failures) = (0, 0, Vec::new());
    for line in text.lines() {
        let [kind, spec, want] = line.split('\t').collect::<Vec<_>>()[..] else {
            panic!("bad line {line}");
        };
        cases += 1;
        if kind == "sort" {
            let got = sorted(&searcher, &reader, spec, &filters);
            check(&mut failures, &mut errors, line, want, got, |got| {
                let want: SortHits = want
                    .split(' ')
                    .filter(|s| *s != "-")
                    .map(|h| {
                        let (d, v) = h.split_once(':').unwrap();
                        (d.parse().unwrap(), expected_value(v))
                    })
                    .collect();
                *got == want
            });
            continue;
        }
        let clause = parse(spec, &filters);
        let n = if kind == "all" { 100_000 } else { 10 };
        let query = BooleanQuery {
            must: vec![clause],
            ..Default::default()
        };
        let got = searcher.search(&query, n).map(|top| {
            top.score_docs
                .iter()
                .map(|h| (h.doc, h.score.to_bits()))
                .collect::<Vec<_>>()
        });
        check(&mut failures, &mut errors, line, want, got, |got| {
            let want: Vec<(i32, u32)> = want
                .split(' ')
                .filter(|s| *s != "-")
                .map(|h| {
                    let (d, b) = h.split_once(':').unwrap();
                    (d.parse().unwrap(), u32::from_str_radix(b, 16).unwrap())
                })
                .collect();
            *got == want
        });
    }
    assert!(
        failures.is_empty(),
        "{} of {cases} searches differ from Lucene:\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert!(cases >= 900, "{cases} searches");
    assert!(errors >= 18, "{errors} expected errors");
}

/// Compares one search with Lucene's answer: an error where Lucene threw,
/// else `same(got)`.
fn check<T: std::fmt::Debug>(
    failures: &mut Vec<String>,
    errors: &mut usize,
    line: &str,
    want: &str,
    got: lucene_search::Result<T>,
    same: impl FnOnce(&T) -> bool,
) {
    match (want.starts_with("ERR "), got) {
        (true, Err(_)) => *errors += 1,
        (true, Ok(got)) => failures.push(format!("{line}\n  expected an error, got {got:?}")),
        (false, Err(e)) => failures.push(format!("{line}\n  error {e}")),
        (false, Ok(got)) => {
            if !same(&got) {
                let shown = format!("{got:?}");
                failures.push(format!(
                    "{}\n  got {}",
                    &line[..line.len().min(400)],
                    &shown[..shown.len().min(400)]
                ));
            }
        }
    }
}

type SortHits = Vec<(i32, (Option<i64>, Option<Option<Vec<u8>>>))>;

fn sorted(
    searcher: &IndexSearcher<'_, '_>,
    reader: &DirectoryReader,
    spec: &str,
    filters: &Filters,
) -> lucene_search::Result<SortHits> {
    let inner = spec
        .strip_prefix("sort(")
        .and_then(|s| s.strip_suffix(')'))
        .expect("sort spec");
    let parts: Vec<&str> = inner.splitn(9, ',').collect();
    let ty = match parts[1] {
        "LONG" => JoinSortType::Long,
        "INT" => JoinSortType::Int,
        "FLOAT" => JoinSortType::Float,
        "DOUBLE" => JoinSortType::Double,
        "STRING" => JoinSortType::String,
        other => panic!("type {other}"),
    };
    let sf = ToParentBlockJoinSortField::new(
        parts[0],
        ty,
        parts[2] == "true",
        parts[3] == "true",
        missing(parts[4]),
        missing(parts[5]),
        filters.get(parts[6]),
        filters.get(parts[7]),
    )?;
    let query = BooleanQuery {
        must: vec![parse(parts[8], filters)],
        ..Default::default()
    };
    let top = searcher.search_sorted(
        reader.segment_readers(),
        &query,
        50,
        &[sf.sort_field()],
        None,
    )?;
    Ok(top
        .hits
        .into_iter()
        .map(|h| {
            let v = if ty == JoinSortType::String {
                (None, Some(h.terms.into_iter().next().flatten()))
            } else {
                (Some(h.values[0]), None)
            };
            (h.doc, v)
        })
        .collect())
}
