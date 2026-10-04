//! M10 T10.3, differentially against Lucene 10.5.0:
//! `fixtures/src/GenQueryTimeJoin.java`.
//!
//! A four-segment index of from and to documents with deletions in two
//! segments. Every recorded `JoinUtil.createJoinQuery` -- terms joins
//! (single- and multi-valued from fields, single- and multi-term to fields),
//! numeric joins (long, int, float and double points, single- and
//! multi-valued), global-ordinal joins (with and without min/max), in every
//! score mode -- is built here from the same from query over the same index
//! and searched alone (every hit) and boosted inside a boolean (the top
//! ten): the same hits with the same score bits, or an error where Lucene
//! threw.

// Test fixtures' own arithmetic -- see `docs/arithmetic-gate.md`'s "Test code".
#![allow(clippy::arithmetic_side_effects)]

use std::collections::HashMap;
use std::sync::Arc;

use lucene_search::directory_reader::DirectoryReader;
use lucene_search::field_norms::FieldNorms;
use lucene_search::index_searcher::{IndexSearcher, SegmentNorms};
use lucene_search::join::{
    create_global_ordinals_join_query, create_join_query, create_numeric_join_query, ordinal_map,
    NumericType, ScoreMode,
};
use lucene_search::ordinal_map::OrdinalMap;
use lucene_search::query::{BooleanQuery, BoostQuery, Clause, MatchAllDocsQuery, TermQuery};
use lucene_store::FsDirectory;

fn data() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/query_time_join")
}

fn term(field: &str, value: &str) -> Clause {
    Clause::Term(TermQuery::new(field, value.as_bytes().to_vec()))
}

/// A reader of the generator's query specs.
struct Parser<'s> {
    s: &'s str,
    at: usize,
}

impl<'s> Parser<'s> {
    fn peek(&self) -> u8 {
        self.s.as_bytes()[self.at]
    }

    fn eat(&mut self, c: u8) {
        assert_eq!(self.peek(), c, "at {} of {}", self.at, self.s);
        self.at += 1;
    }

    fn word(&mut self) -> &'s str {
        let start = self.at;
        while !matches!(self.peek(), b',' | b'(' | b')') {
            self.at += 1;
        }
        &self.s[start..self.at]
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
            "all" => Clause::MatchAllDocs(MatchAllDocsQuery::new(0)),
            "bool" => {
                let mut b = BooleanQuery::default();
                loop {
                    let start = self.at;
                    while self.peek() != b':' {
                        self.at += 1;
                    }
                    let occ = &self.s[start..self.at];
                    self.at += 1;
                    let c = self.query();
                    match occ {
                        "must" => b.must.push(c),
                        "should" => b.should.push(c),
                        "filter" => b.filter.push(c),
                        "not" => b.must_not.push(c),
                        other => panic!("occur {other}"),
                    }
                    if self.peek() == b')' {
                        break;
                    }
                    self.eat(b',');
                }
                Clause::Boolean(Box::new(b))
            }
            other => panic!("query head {other} in {}", self.s),
        };
        self.eat(b')');
        q
    }
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

/// The join query a spec names, built over `searcher`.
fn build(
    spec: &str,
    searcher: &IndexSearcher<'_, '_>,
    gj: &Arc<OrdinalMap>,
) -> lucene_search::Result<Clause> {
    let (head, rest) = spec.split_once('(').expect("join spec");
    let inner = rest.strip_suffix(')').expect("join spec");
    let fields = match head {
        "terms" => 5,
        "num" => 6,
        _ => 4,
    };
    let mut parts = inner.splitn(fields, ',');
    let mut next = || parts.next().expect("spec part");
    match head {
        "terms" => {
            let (from, mv, to, m) = (next(), next() == "true", next(), mode(next()));
            let q = Parser { s: next(), at: 0 }.query();
            create_join_query(from, mv, to, &q, searcher, m)
        }
        "num" => {
            let (from, mv, to) = (next(), next() == "true", next());
            let ty = match next() {
                "Long" => NumericType::Long,
                "Integer" => NumericType::Int,
                "Float" => NumericType::Float,
                "Double" => NumericType::Double,
                other => panic!("type {other}"),
            };
            let m = mode(next());
            let q = Parser { s: next(), at: 0 }.query();
            create_numeric_join_query(from, mv, to, ty, &q, searcher, m)
        }
        "gord" | "gord1" => {
            let m = mode(next());
            let min: i32 = next().parse().unwrap();
            let max: i32 = next().parse().unwrap();
            let mut p = Parser { s: next(), at: 0 };
            let from = p.query();
            p.eat(b',');
            let to = p.query();
            create_global_ordinals_join_query(
                "gj",
                &from,
                &to,
                searcher,
                m,
                (head == "gord").then(|| Arc::clone(gj)),
                min,
                max,
            )
        }
        "gord-nomap" => create_global_ordinals_join_query(
            "gj",
            &term("type", "from"),
            &Clause::MatchAllDocs(MatchAllDocsQuery::new(0)),
            searcher,
            ScoreMode::None,
            None,
            0,
            i32::MAX,
        ),
        other => panic!("join {other}"),
    }
}

/// `doc:scorebits ..`, or `-` for none.
fn scored_hits(want: &str) -> Vec<(i32, u32)> {
    want.split(' ')
        .filter(|s| *s != "-")
        .map(|h| {
            let (d, b) = h.split_once(':').unwrap();
            (d.parse().unwrap(), u32::from_str_radix(b, 16).unwrap())
        })
        .collect()
}

fn search(
    searcher: &IndexSearcher<'_, '_>,
    clause: Clause,
    n: usize,
) -> lucene_search::Result<Vec<(i32, u32)>> {
    let query = BooleanQuery {
        must: vec![clause],
        ..Default::default()
    };
    Ok(searcher
        .search(&query, n)?
        .score_docs
        .iter()
        .map(|h| (h.doc, h.score.to_bits()))
        .collect())
}

#[test]
fn query_time_joins_match_lucene_bit_for_bit() {
    let dir = data();
    let text = std::fs::read_to_string(dir.join("searches.tsv"))
        .expect("run scripts/gen-fixtures.sh --only GenQueryTimeJoin");
    let reader = DirectoryReader::open(&FsDirectory::open(dir.join("index"))).unwrap();
    assert_eq!(reader.segment_readers().len(), 4);
    let mut opened = reader.open_segments().unwrap();
    opened.open_points().unwrap();
    let segments = opened.as_open_segments();
    let owned = reader.field_norms_by_field(&["body".to_string()]);
    let norms: Vec<SegmentNorms<'_, '_>> = owned
        .iter()
        .map(|m: &HashMap<String, FieldNorms<'_>>| Some(m))
        .collect();
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    let gj = ordinal_map(&searcher, "gj").unwrap();
    // The first segment alone, as `new IndexSearcher(leafReader)`: its own
    // statistics, its own average field length.
    let first: HashMap<String, FieldNorms<'_>> = reader.segment_readers()[0]
        .field_norms("body")
        .map(|n| ("body".to_string(), n))
        .into_iter()
        .collect();
    let first_norms: Vec<SegmentNorms<'_, '_>> = vec![Some(&first)];
    let one = IndexSearcher::new(&segments[..1], &first_norms).unwrap();

    let (mut cases, mut errors, mut failures) = (0, 0, Vec::new());
    let mut kinds: HashMap<String, usize> = HashMap::new();
    for line in text.lines() {
        let [kind, spec, want] = line.split('\t').collect::<Vec<_>>()[..] else {
            panic!("bad line {line}");
        };
        cases += 1;
        let (join_spec, word) = match spec.rsplit_once('|') {
            Some((j, w)) if kind == "bool" => (j, Some(w)),
            _ => (spec, None),
        };
        *kinds
            .entry(join_spec.split('(').next().unwrap().to_string())
            .or_default() += 1;
        let on = if kind == "one" { &one } else { &searcher };
        let got = build(join_spec, on, &gj).and_then(|join| match word {
            None => {
                let hits = search(on, join.clone(), 100_000)?;
                // `count` runs the join without scores.
                let count = on.count(&BooleanQuery {
                    must: vec![join],
                    ..Default::default()
                })?;
                assert_eq!(count, hits.len() as u64, "count of {spec}");
                Ok(hits)
            }
            Some(w) => {
                let q = BooleanQuery {
                    must: vec![Clause::Boost(Box::new(BoostQuery::new(join, 2.0)))],
                    filter: vec![term("type", "to")],
                    should: vec![term("body", w)],
                    ..Default::default()
                };
                Ok(searcher
                    .search(&q, 10)?
                    .score_docs
                    .iter()
                    .map(|h| (h.doc, h.score.to_bits()))
                    .collect())
            }
        });
        match (want.starts_with("ERR "), got) {
            (true, Err(_)) => errors += 1,
            (true, Ok(got)) => failures.push(format!("{line}\n  Lucene threw; got {got:?}")),
            (false, Err(e)) => failures.push(format!("{line}\n  error {e}")),
            (false, Ok(got)) => {
                let want = scored_hits(want);
                if got != want {
                    failures.push(format!("{kind}\t{spec}\n  want {want:?}\n  got  {got:?}"));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {cases} searches differ from Lucene:\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert!(cases >= 800, "{cases} searches");
    assert!(errors >= 4, "{errors} expected errors");
    assert!(kinds.get("gord1").copied().unwrap_or(0) >= 25, "{kinds:?}");
    for k in ["terms", "num", "gord"] {
        assert!(kinds.get(k).copied().unwrap_or(0) >= 100, "{k}: {kinds:?}");
    }
}

/// The paths the recorded searches do not reach: explanations, the doc-values
/// getters' refusals and singletons, the term-set enum over a real terms
/// enum, the leaf collector's segment walk, and misuse of the to-fields.
#[test]
fn query_time_join_edges() {
    use lucene_search::join::seeking_term_set_terms_enum;
    use lucene_search::leaf_collector::{search_segments, SegmentCollector};
    use lucene_search::multi_segment::OpenSegment;
    use lucene_search::reader::doc_values::{
        get_numeric, get_sorted, get_sorted_numeric, get_sorted_set,
    };
    use lucene_search::reader::{LeafReader, TermsEnum};

    let dir = data();
    let reader = DirectoryReader::open(&FsDirectory::open(dir.join("index"))).unwrap();
    let mut opened = reader.open_segments().unwrap();
    opened.open_points().unwrap();
    let segments = opened.as_open_segments();
    let owned = reader.field_norms_by_field(&["body".to_string()]);
    let norms: Vec<SegmentNorms<'_, '_>> = owned.iter().map(Some).collect();
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    let from = term("type", "from");
    let all = |c: Clause| BooleanQuery {
        must: vec![c],
        ..Default::default()
    };

    // `explain` gives the score a search gives.
    for (join, desc) in [
        (
            create_join_query("fkm", true, "pkm", &from, &searcher, ScoreMode::Max).unwrap(),
            "Score based on join value k",
        ),
        (
            create_numeric_join_query(
                "nfk",
                false,
                "npl",
                NumericType::Long,
                &from,
                &searcher,
                ScoreMode::Total,
            )
            .unwrap(),
            "A match",
        ),
    ] {
        let hits = search(&searcher, join.clone(), 5).unwrap();
        assert!(!hits.is_empty());
        for (doc, bits) in hits {
            let e = searcher.explain(&all(join.clone()), doc).unwrap();
            assert_eq!(e.value.to_bits(), bits, "{e:?}");
            assert!(e.description.starts_with(desc), "{e:?}");
        }
        // A from document is never a match.
        let f = search(&searcher, term("type", "from"), 1).unwrap()[0].0;
        let miss = searcher.explain(&all(join.clone()), f).unwrap();
        assert!(!miss.matched, "{miss:?}");
    }
    // A to-field no segment has.
    for join in [
        create_join_query("fk", false, "nofield", &from, &searcher, ScoreMode::Avg).unwrap(),
        create_numeric_join_query(
            "nfk",
            false,
            "nofield",
            NumericType::Long,
            &from,
            &searcher,
            ScoreMode::Avg,
        )
        .unwrap(),
    ] {
        assert!(search(&searcher, join.clone(), 10).unwrap().is_empty());
        assert!(!searcher.explain(&all(join), 0).unwrap().matched);
    }
    let gord = create_global_ordinals_join_query(
        "gj",
        &from,
        &term("type", "to"),
        &searcher,
        ScoreMode::Max,
        Some(ordinal_map(&searcher, "gj").unwrap()),
        0,
        i32::MAX,
    )
    .unwrap();
    assert!(searcher.explain(&all(gord), 0).is_err());

    // The to-field must be one-dimensional points of the type's width.
    for (to, ty) in [("pk", NumericType::Long), ("npl", NumericType::Int)] {
        let join =
            create_numeric_join_query("nfk", false, to, ty, &from, &searcher, ScoreMode::Max)
                .unwrap();
        let err = search(&searcher, join, 10).unwrap_err().to_string();
        assert!(err.contains("was indexed with"), "{err}");
    }
    // An empty from side.
    let none = create_join_query(
        "fk",
        false,
        "pk",
        &term("body", "nosuchword"),
        &searcher,
        ScoreMode::None,
    )
    .unwrap();
    assert!(search(&searcher, none, 10).unwrap().is_empty());
    // No segments, and a single segment without the join field.
    let empty = IndexSearcher::new(&[], &[]).unwrap();
    let q = create_global_ordinals_join_query(
        "gj",
        &from,
        &from,
        &empty,
        ScoreMode::None,
        None,
        0,
        i32::MAX,
    )
    .unwrap();
    assert!(matches!(q, Clause::MatchNoDocs(_)));
    let one = IndexSearcher::new(&segments[..1], &norms[..1]).unwrap();
    let q =
        create_global_ordinals_join_query("nope", &from, &from, &one, ScoreMode::Avg, None, 0, 9)
            .unwrap();
    assert!(matches!(q, Clause::MatchNoDocs(_)));

    // A to-query without a scorer in any segment.
    for mode in [ScoreMode::None, ScoreMode::Max] {
        let q = create_global_ordinals_join_query(
            "gj",
            &from,
            &term("body", "nosuchword"),
            &searcher,
            mode,
            Some(ordinal_map(&searcher, "gj").unwrap()),
            0,
            i32::MAX,
        )
        .unwrap();
        assert!(search(&searcher, q, 10).unwrap().is_empty());
    }
    // A global-ordinals join run over a reader it was not built for: the
    // segments are not the ordinal map's (Java's `IllegalStateException`),
    // and with scores a segment without the join field matches nothing.
    {
        let other_dir = data().join("../block_join/index");
        let other = DirectoryReader::open(&FsDirectory::open(other_dir)).unwrap();
        let other_opened = other.open_segments().unwrap();
        let other_segments = other_opened.as_open_segments();
        let other_norms: Vec<SegmentNorms<'_, '_>> = other_segments.iter().map(|_| None).collect();
        let foreign = IndexSearcher::new(&other_segments, &other_norms).unwrap();
        let map = Some(ordinal_map(&searcher, "gj").unwrap());
        let match_all = Clause::MatchAllDocs(MatchAllDocsQuery::new(0));
        let q = create_global_ordinals_join_query(
            "gj",
            &from,
            &match_all,
            &searcher,
            ScoreMode::None,
            map.clone(),
            0,
            i32::MAX,
        )
        .unwrap();
        let err = search(&foreign, q, 10).unwrap_err().to_string();
        assert!(err.contains("different index reader"), "{err}");
        let q = create_global_ordinals_join_query(
            "gj",
            &from,
            &match_all,
            &searcher,
            ScoreMode::Max,
            map,
            0,
            i32::MAX,
        )
        .unwrap();
        assert!(search(&foreign, q, 10).unwrap().is_empty());
    }
    // A scoring numeric join over segments whose points were not opened.
    {
        let unopened = reader.open_segments().unwrap();
        let bare = unopened.as_open_segments();
        let bare_searcher = IndexSearcher::new(&bare, &norms).unwrap();
        let join = create_numeric_join_query(
            "nfk",
            false,
            "npl",
            NumericType::Long,
            &from,
            &bare_searcher,
            ScoreMode::Max,
        )
        .unwrap();
        assert!(search(&bare_searcher, join, 10).is_err());
    }

    // The doc-values getters.
    let seg = &reader.segment_readers()[0];
    let msg = |r: lucene_search::Result<()>| r.unwrap_err().to_string();
    assert!(msg(get_sorted(seg, "nfk").map(|_| ()))
        .contains("unexpected docvalues type NUMERIC for field 'nfk' (expected=SORTED). Re-index"));
    assert!(msg(get_sorted_set(seg, "nfk").map(|_| ()))
        .contains("(expected one of [SORTED, SORTED_SET])"));
    assert!(msg(get_numeric(seg, "fk").map(|_| ())).contains("(expected=NUMERIC)"));
    assert!(msg(get_sorted_numeric(seg, "fk").map(|_| ()))
        .contains("(expected one of [SORTED_NUMERIC, NUMERIC])"));
    assert!(msg(get_sorted(seg, "body").map(|_| ())).contains("type NONE"));
    let mut empty = get_sorted(seg, "nosuchfield").unwrap();
    assert_eq!(empty.next_doc().unwrap(), i32::MAX);
    let mut single = get_sorted(seg, "fk").unwrap();
    let mut set = get_sorted_set(seg, "fk").unwrap();
    assert_eq!(set.value_count(), i64::from(single.value_count()));
    let mut docs = 0;
    while set.next_doc().unwrap() != i32::MAX {
        let doc = set.doc_id();
        assert!(single.advance_exact(doc).unwrap());
        assert_eq!(set.doc_value_count(), 1);
        let ord = set.next_ord().unwrap();
        assert_eq!(ord, i64::from(single.ord_value()));
        assert!(set.next_ord().is_err(), "one ordinal per document");
        assert_eq!(
            set.lookup_ord(ord).unwrap(),
            single.lookup_ord(single.ord_value()).unwrap()
        );
        docs += 1;
    }
    assert!(docs > 0 && set.cost() > 0);
    assert!(set.advance(0).unwrap() == i32::MAX);
    assert!(set.lookup_ord(i64::MAX).is_err());
    let mut set = get_sorted_set(seg, "fk").unwrap();
    let first = set.advance(0).unwrap();
    assert!(set.advance_exact(first).unwrap());
    let mut num = get_numeric(seg, "nfk").unwrap();
    let mut sn = get_sorted_numeric(seg, "nfk").unwrap();
    let mut first = true;
    while sn.next_doc().unwrap() != i32::MAX {
        let doc = sn.doc_id();
        assert!(num.advance_exact(doc).unwrap());
        assert_eq!(sn.doc_value_count(), 1);
        assert_eq!(sn.next_value().unwrap(), num.long_value());
        assert!(sn.next_value().is_err());
        if first {
            first = false;
            let mut again = get_sorted_numeric(seg, "nfk").unwrap();
            assert_eq!(again.advance(doc).unwrap(), doc);
            assert!(again.advance_exact(doc).unwrap());
            assert!(again.cost() > 0);
        }
    }

    // `SeekingTermSetTermsEnum` over the segment's terms of `pk`.
    let terms = seg.terms("pk").unwrap().unwrap();
    let want: Vec<Vec<u8>> = ["k1", "k10", "k5", "k99", "zz"]
        .iter()
        .map(|t| t.as_bytes().to_vec())
        .collect();
    let mut it = seeking_term_set_terms_enum(terms.iterator().unwrap(), want.clone().into());
    let mut found = Vec::new();
    while let Some(t) = it.next().unwrap() {
        found.push(t.to_vec());
        assert!(it.doc_freq().unwrap() > 0);
    }
    let mut present = Vec::new();
    let mut all_terms = terms.iterator().unwrap();
    while let Some(t) = all_terms.next().unwrap() {
        if want.iter().any(|w| w.as_slice() == t) {
            present.push(t.to_vec());
        }
    }
    assert_eq!(found, present);
    assert!(!found.is_empty());

    // The leaf collector enters and finishes every segment, matching or not,
    // and stops at the first error.
    struct Walk {
        entered: Vec<usize>,
        finished: usize,
        docs: usize,
        fail_at: Option<usize>,
    }
    impl<'a> SegmentCollector<'a> for Walk {
        fn score_mode(&self) -> lucene_search::collector::ScoreMode {
            lucene_search::collector::ScoreMode::CompleteNoScores
        }
        fn set_next_reader(
            &mut self,
            ord: usize,
            _leaf: &OpenSegment<'a>,
        ) -> lucene_search::Result<()> {
            self.entered.push(ord);
            Ok(())
        }
        fn collect(&mut self, _doc: i32, _score: f32) -> lucene_search::Result<()> {
            self.docs += 1;
            if Some(self.docs) == self.fail_at {
                return Err(lucene_search::Error::IllegalState("stop".into()));
            }
            Ok(())
        }
        fn finish(&mut self) -> lucene_search::Result<()> {
            self.finished += 1;
            Ok(())
        }
    }
    let mut w = Walk {
        entered: Vec::new(),
        finished: 0,
        docs: 0,
        fail_at: None,
    };
    search_segments(&searcher, &all(term("id", "f0")), &mut w).unwrap();
    assert_eq!(w.entered, vec![0, 1, 2, 3]);
    assert_eq!(w.finished, 4);
    assert!(w.docs <= 1);
    let mut w = Walk {
        entered: Vec::new(),
        finished: 0,
        docs: 0,
        fail_at: Some(3),
    };
    assert!(search_segments(&searcher, &all(from.clone()), &mut w).is_err());

    // A collector per slice, each walking its own slice's segments.
    let mut sliced = IndexSearcher::new(&segments, &norms).unwrap();
    sliced.set_slices(vec![vec![2, 0], vec![3, 1]]).unwrap();
    let walks = lucene_search::leaf_collector::search_slices(&sliced, &all(from.clone()), || {
        Ok(Walk {
            entered: Vec::new(),
            finished: 0,
            docs: 0,
            fail_at: None,
        })
    })
    .unwrap();
    assert_eq!(walks.len(), 2);
    assert_eq!(walks[0].entered, vec![0, 2]);
    assert_eq!(walks[1].entered, vec![1, 3]);
    let total: usize = walks.iter().map(|w| w.docs).sum();
    assert_eq!(total as u64, searcher.count(&all(from.clone())).unwrap());
    let mut w = Walk {
        entered: Vec::new(),
        finished: 0,
        docs: 0,
        fail_at: None,
    };
    assert!(
        lucene_search::leaf_collector::search_slice(&searcher, &all(from), &[9], &mut w).is_err()
    );
    let none = sliced.set_slices(Vec::new());
    assert!(none.is_ok());
    let walks = lucene_search::leaf_collector::search_slices(&sliced, &all(term("x", "y")), || {
        Ok(Walk {
            entered: Vec::new(),
            finished: 0,
            docs: 0,
            fail_at: None,
        })
    })
    .unwrap();
    assert_eq!(walks.len(), 1, "no slices: one empty collector");
}
