//! M14.f, differentially against Lucene 10.5.0:
//! `fixtures/src/GenSandboxQueries.java`.
//!
//! lucene-sandbox's scoring queries -- `CombinedFieldQuery` over several
//! terms, `CoveringQuery` (constant and per-document minimums),
//! `TermAutomatonQuery` (any-term transitions, loops, rewrites to term and
//! phrase queries), `TokenStreamToTermAutomatonQuery` and
//! `PhraseWildcardQuery` (prefix, wildcard and fuzzy positions under shared
//! expansion budgets) -- over a three-segment index with deletions: each
//! query's total hits and top 20 with their score bits (`searches.tsv`), and
//! the errors Java throws building or searching the bad ones
//! (`errors.tsv`).

// Test fixtures' own arithmetic -- see `docs/arithmetic-gate.md`'s "Test code".
#![allow(clippy::arithmetic_side_effects)]

use lucene_search::directory_reader::DirectoryReader;
use lucene_search::extended_query::{MultiTermQuery, MultiTermSource, RewriteMethod};
use lucene_search::index_searcher::{IndexSearcher, SegmentNorms};
use lucene_search::query::{
    BooleanQuery, BoostQuery, Clause, FuzzyQuery, PrefixQuery, RegexpQuery, TermQuery,
    WildcardQuery,
};
use lucene_search::sandbox::search::{
    CombinedFieldQuery, CoveringQuery, GraphToken, PhraseWildcardBuilder, TermAutomatonQuery,
    TokenStreamToTermAutomatonQuery,
};
use lucene_search::values_source::{
    long_constant, long_from_long_field, BoxDoubleValues, BoxLongValues, LongValuesSource,
    ValuesContext,
};
use lucene_search::{Error, Result};
use lucene_store::FsDirectory;

fn data() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/sandbox_queries")
}

/// `GenSandboxQueries.Parser`.
struct Parser<'s> {
    tok: Vec<&'s str>,
    pos: usize,
}

impl<'s> Parser<'s> {
    fn new(spec: &'s str) -> Self {
        Self {
            tok: spec.split(' ').collect(),
            pos: 0,
        }
    }

    fn next(&mut self) -> &'s str {
        self.pos += 1;
        self.tok[self.pos - 1]
    }

    fn int(&mut self) -> i32 {
        self.next().parse().unwrap()
    }

    fn query(&mut self) -> Result<Clause> {
        Ok(match self.next() {
            "T" => {
                let f = self.next();
                Clause::Term(TermQuery::new(f, self.next()))
            }
            "B" => {
                let n = self.int();
                let mut should = Vec::new();
                for _ in 0..n {
                    should.push(self.query()?);
                }
                Clause::Boolean(Box::new(BooleanQuery::new().with_should(should)))
            }
            "boost" => {
                let x: f32 = self.next().parse().unwrap();
                Clause::Boost(Box::new(BoostQuery::new(self.query()?, x)))
            }
            "CFS" => {
                let n = self.int();
                let terms: Vec<Vec<u8>> = (0..n).map(|_| self.next().as_bytes().to_vec()).collect();
                let m = self.int();
                let mut fields = Vec::new();
                for _ in 0..m {
                    let f = self.next().to_string();
                    fields.push((f, self.next().parse::<f32>().unwrap()));
                }
                Clause::from(CombinedFieldQuery::with_terms(terms, fields)?)
            }
            "COV" => {
                let kind = self.next();
                let arg = self.next();
                let min = if kind == "c" {
                    long_constant(arg.parse().unwrap())
                } else {
                    long_from_long_field(arg)
                };
                let n = self.int();
                let mut qs = Vec::new();
                for _ in 0..n {
                    qs.push(self.query()?);
                }
                Clause::from(CoveringQuery::new(qs, min)?)
            }
            "TAQ" => {
                let mut q = TermAutomatonQuery::new(self.next());
                for _ in 0..self.int() {
                    q.create_state();
                }
                let accept = self.next();
                if accept != "-" {
                    for a in accept.split(',') {
                        q.set_accept(a.parse().unwrap(), true)?;
                    }
                }
                for _ in 0..self.int() {
                    let t: Vec<&str> = self.next().splitn(3, '>').collect();
                    let (s, d) = (t[0].parse().unwrap(), t[1].parse().unwrap());
                    if t[2] == "*" {
                        q.add_any_transition(s, d);
                    } else {
                        q.add_transition(s, d, t[2]);
                    }
                }
                q.finish()?;
                Clause::from(q)
            }
            "TS" => {
                let f = self.next();
                let preserve = self.next() == "1";
                let n = self.int();
                let tokens: Vec<GraphToken> = (0..n)
                    .map(|_| {
                        let t: Vec<&str> = self.next().split('/').collect();
                        GraphToken::new(t[0], t[1].parse().unwrap(), t[2].parse().unwrap())
                    })
                    .collect();
                let mut ts = TokenStreamToTermAutomatonQuery::new();
                ts.set_preserve_position_increments(preserve);
                Clause::from(ts.to_query(f, tokens)?)
            }
            "PW" => {
                let f = self.next();
                let max = self.int();
                let slop = self.int();
                let n = self.int();
                let mut b = PhraseWildcardBuilder::new(f, usize::try_from(max).unwrap());
                b.set_slop(slop)?;
                for _ in 0..n {
                    let (kind, text) = self.next().split_once(':').unwrap();
                    match kind {
                        "t" => {
                            b.add_term(text);
                        }
                        "p" => {
                            b.add_multi_term(&Clause::Prefix(PrefixQuery::new(f, text)))?;
                        }
                        "w" => {
                            b.add_multi_term(&Clause::Wildcard(WildcardQuery::new(f, text)))?;
                        }
                        "f" => {
                            let (t, e) = text.split_once('~').unwrap();
                            let q = FuzzyQuery::new(f, t).with_max_edits(e.parse().unwrap());
                            b.add_multi_term(&Clause::Fuzzy(q))?;
                        }
                        other => panic!("{other}"),
                    }
                }
                Clause::from(b.build())
            }
            other => panic!("{other}"),
        })
    }
}

fn hits(searcher: &IndexSearcher<'_, '_>, q: Clause) -> Result<String> {
    let td = searcher.search(
        &BooleanQuery {
            must: vec![q],
            ..Default::default()
        },
        20,
    )?;
    let mut b = format!("{}\t", td.total_hits.value);
    for sd in &td.score_docs {
        b.push_str(&format!("{}:{:x},", sd.doc, sd.score.to_bits()));
    }
    Ok(b)
}

/// Java's exception class and message.
fn java(e: &Error) -> String {
    match e {
        Error::InvalidQuery(m) | Error::IllegalArgument(m) => {
            format!("IllegalArgumentException: {m}")
        }
        Error::IllegalState(m) => format!("IllegalStateException: {m}"),
        other => format!("{other:?}"),
    }
}

fn open<T>(f: impl FnOnce(&IndexSearcher<'_, '_>) -> T) -> T {
    let reader = DirectoryReader::open(&FsDirectory::open(data().join("index"))).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let fields: Vec<String> = ["body", "title"].iter().map(|f| f.to_string()).collect();
    let owned = reader.field_norms_by_field(&fields);
    let norms: Vec<SegmentNorms<'_, '_>> = owned.iter().map(Some).collect();
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    f(&searcher)
}

#[test]
fn sandbox_queries_score_as_lucene_does() {
    let text = std::fs::read_to_string(data().join("searches.tsv")).unwrap();
    open(|searcher| {
        let mut bad = 0;
        for line in text.lines() {
            let (spec, want) = line.split_once('\t').unwrap();
            let got = Parser::new(spec)
                .query()
                .and_then(|q| hits(searcher, q))
                .unwrap_or_else(|e| format!("error {e:?}"));
            if got != want {
                bad += 1;
                eprintln!("{spec}\n  java {want}\n  rust {got}");
            }
        }
        assert_eq!(bad, 0, "{bad} of {} searches differ", text.lines().count());
    });
}

#[test]
fn sandbox_query_errors_are_lucenes() {
    let text = std::fs::read_to_string(data().join("errors.tsv")).unwrap();
    open(|searcher| {
        let mut bad = 0;
        for line in text.lines() {
            let (spec, want) = line.split_once('\t').unwrap();
            let got = match Parser::new(spec).query().and_then(|q| hits(searcher, q)) {
                Ok(_) => "OK".to_string(),
                Err(e) => java(&e),
            };
            if got != want {
                bad += 1;
                eprintln!("{spec}\n  java {want}\n  rust {got}");
            }
        }
        assert_eq!(bad, 0, "{bad} error lines differ");
    });
}

fn must(q: Clause) -> BooleanQuery {
    BooleanQuery {
        must: vec![q],
        ..Default::default()
    }
}

/// What a query's top 20 and total come to, for comparing two of them.
fn same(searcher: &IndexSearcher<'_, '_>, a: Clause, b: Clause) {
    assert_eq!(
        hits(searcher, a.clone()).unwrap(),
        hits(searcher, b.clone()).unwrap(),
        "{a:?}\n{b:?}"
    );
}

/// A minimum that reads scores, which `CoveringQuery` refuses.
struct ScoreReading;

impl LongValuesSource for ScoreReading {
    fn get_values<'c>(
        &self,
        _ctx: &ValuesContext<'c>,
        _leaf: usize,
        _scores: Option<BoxDoubleValues<'c>>,
    ) -> Result<BoxLongValues<'c>> {
        unreachable!()
    }
    fn needs_scores(&self) -> bool {
        true
    }
    fn is_cacheable(&self, _ctx: &ValuesContext<'_>, _leaf: usize) -> bool {
        false
    }
    fn describe(&self) -> String {
        "score".into()
    }
}

#[test]
fn covering_query_edges() {
    let t = |w: &str| Clause::Term(TermQuery::new("body", w));
    let too_many: Vec<Clause> = (0..1025).map(|i| t(&format!("x{i}"))).collect();
    assert!(matches!(
        CoveringQuery::new(too_many, long_constant(1)),
        Err(Error::InvalidQuery(m)) if m == "too many clauses"
    ));
    assert_eq!(
        java(&CoveringQuery::new([t("w1")], std::sync::Arc::new(ScoreReading)).unwrap_err()),
        "IllegalArgumentException: The minimum number of matches may not depend on the score."
    );
    // A multiset compared regardless of order, with its minimum's description.
    let a = CoveringQuery::new([t("w1"), t("w2")], long_from_long_field("mm")).unwrap();
    let b = CoveringQuery::new([t("w2"), t("w1")], long_from_long_field("mm")).unwrap();
    let c = CoveringQuery::new([t("w2"), t("w1")], long_from_long_field("other")).unwrap();
    assert_eq!(a, b);
    assert_ne!(a, c);
    assert!(format!("{a:?}").starts_with("CoveringQuery(queries=["));
    assert!(a.rewrite().is_none());
    open(|searcher| {
        // As a filter (no scores), and over terms no segment has.
        let q = Clause::from(a.clone());
        let filtered = BooleanQuery {
            filter: vec![q.clone()],
            ..Default::default()
        };
        let total = searcher.search(&must(q), 1000).unwrap().total_hits.value;
        assert_eq!(searcher.count(&filtered).unwrap(), total);
        assert!(total > 0);
        let none = CoveringQuery::new([t("zz"), t("yy")], long_from_long_field("mm")).unwrap();
        assert_eq!(hits(searcher, none.into()).unwrap(), "0\t");
    });
}

#[test]
fn phrase_wildcard_query_edges() {
    let mut b = PhraseWildcardBuilder::new("body", 100);
    assert_eq!(
        java(
            &b.add_field_term(&TermQuery::new("title", "w1"))
                .unwrap_err()
        ),
        "IllegalArgumentException: Term field \"title\" cannot be different from the \
         PhraseWildcardQuery field \"body\""
    );
    assert_eq!(
        java(
            &b.add_multi_term(&Clause::Prefix(PrefixQuery::new("title", "w")))
                .unwrap_err()
        ),
        "IllegalArgumentException: PrefixQuery field \"title\" cannot be different from the \
         PhraseWildcardQuery field \"body\""
    );
    assert!(b
        .add_multi_term(&Clause::Term(TermQuery::new("body", "w1")))
        .is_err());
    assert!(b
        .add_multi_term(&Clause::from(
            lucene_search::extended_query::SynonymQuery::new("body", [("w1", 1.0)]).unwrap()
        ))
        .is_err());
    open(|searcher| {
        let pw = |terms: &[Clause], first: Option<&str>, opt: bool, slop: i32| {
            let mut b = PhraseWildcardBuilder::with_segment_optimization("body", 100, opt);
            b.set_slop(slop).unwrap();
            if let Some(f) = first {
                b.add_field_term(&TermQuery::new("body", f)).unwrap();
            }
            for t in terms {
                b.add_multi_term(t).unwrap();
            }
            Clause::from(b.build())
        };
        // One position rewrites to the query it was given, rewrite method
        // and all (`phraseTerms.get(0).getQuery()`).
        let scoring = Clause::from(MultiTermQuery::new(
            MultiTermSource::Prefix(PrefixQuery::new("body", "w1")),
            RewriteMethod::ScoringBoolean,
        ));
        let mut one = PhraseWildcardBuilder::new("body", 100);
        one.add_multi_term(&scoring).unwrap();
        assert_eq!(one.build().resolve(searcher.segments()).unwrap(), scoring);
        let regexp = Clause::Regexp(RegexpQuery::new("body", "w1."));
        let wildcard = Clause::Wildcard(WildcardQuery::new("body", "w1?"));
        let multi = Clause::from(MultiTermQuery::new(
            MultiTermSource::Wildcard(WildcardQuery::new("body", "w1?")),
            RewriteMethod::default(),
        ));
        // The regexp, wildcard and generic multi-term positions expand alike.
        same(
            searcher,
            pw(std::slice::from_ref(&regexp), Some("w0"), true, 0),
            pw(std::slice::from_ref(&wildcard), Some("w0"), true, 0),
        );
        same(
            searcher,
            pw(std::slice::from_ref(&multi), Some("w0"), true, 1),
            pw(std::slice::from_ref(&wildcard), Some("w0"), true, 1),
        );
        // Without the segment optimisation, the same hits.
        same(
            searcher,
            pw(std::slice::from_ref(&wildcard), Some("w0"), false, 0),
            pw(std::slice::from_ref(&wildcard), Some("w0"), true, 0),
        );
        same(
            searcher,
            pw(&[wildcard.clone(), regexp.clone()], None, false, 2),
            pw(&[wildcard.clone(), regexp.clone()], None, true, 2),
        );
        // One position: that query itself.
        same(
            searcher,
            pw(std::slice::from_ref(&wildcard), None, true, 0),
            multi.clone(),
        );
        // A fuzzy position at zero edits is its term.
        let fuzzy0 = Clause::Fuzzy(FuzzyQuery::new("body", "w1").with_max_edits(0));
        let mut phrase = PhraseWildcardBuilder::new("body", 100);
        phrase.add_term("w0").add_term("w1");
        same(
            searcher,
            pw(&[fuzzy0], Some("w0"), true, 0),
            phrase.build().into(),
        );
        // A field without positions.
        let mut b = PhraseWildcardBuilder::new("id", 100);
        b.add_term("1").add_term("2");
        assert_eq!(
            java(&hits(searcher, b.build().into()).unwrap_err()),
            "IllegalStateException: field \"id\" was indexed without position data; cannot \
             run PhraseWildcardQuery"
        );
        // A field no segment has.
        let mut b = PhraseWildcardBuilder::new("nope", 100);
        b.add_term("1")
            .add_multi_term(&Clause::Prefix(PrefixQuery::new("nope", "x")))
            .unwrap();
        assert_eq!(hits(searcher, b.build().into()).unwrap(), "0\t");
    });
}
