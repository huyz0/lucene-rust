#![allow(clippy::arithmetic_side_effects)]
//! **Keyword sorts and the doc-values terms dictionary, against real Lucene.**
//!
//! `fixtures/src/GenKeywordSort.java` writes three segments with deletions and
//! four keyword columns (a sparse single-valued `SORTED_SET`, a multi-valued
//! one, a dense `SORTED`, and one without postings), then records:
//!
//! * every segment's `lookupTerm` for present, absent, in-between and
//!   out-of-range keys, and `lookupOrd` for sampled ordinals -- the random
//!   access `TermsDict` gives the keyword comparator;
//! * `TopFieldCollector`'s answer for six queries by thirteen keyword sorts
//!   (both directions, `STRING_FIRST`/`STRING_LAST`, `MIN`/`MAX` selectors,
//!   beside numeric, score and document keys), two page sizes, two thresholds,
//!   and the next page after the last hit and after its values alone.

use std::collections::HashMap;

use lucene_codecs::doc_values::SortedSetKind;
use lucene_codecs::terms_dict::{TermsDict, TermsDictEntry};
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::field_norms::FieldNorms;
use lucene_search::query::{MatchAllDocsQuery, PointsRangeQuery};
use lucene_search::top_field::{search_sorted, FieldDoc, Selector, SortField, SortType};
use lucene_search::{BooleanQuery, Clause, PhraseQuery, TermQuery};
use lucene_store::FsDirectory;

fn fixture_dir() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/keyword_sort_index"
    )
    .to_string()
}

fn manifest() -> HashMap<String, String> {
    let text = std::fs::read_to_string(format!("{}/manifest.properties", fixture_dir()))
        .expect("run scripts/gen-fixtures.sh --only GenKeywordSort");
    text.lines()
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

/// The terms dictionary of `field` in `reader`'s segment `seg`.
fn terms_entry<'r>(
    reader: &'r DirectoryReader,
    seg: usize,
    field: &str,
) -> (&'r [u8], &'r TermsDictEntry) {
    let r = &reader.segment_readers()[seg];
    let info = r
        .field_infos()
        .fields
        .iter()
        .find(|f| f.name == field)
        .unwrap();
    let (meta, data) = r.doc_values_for_field(info.number).unwrap();
    let entry = if let Some(e) = meta.sorted_set_entry(info.number) {
        match &e.kind {
            SortedSetKind::Single(s) => &s.terms,
            SortedSetKind::Multi { terms, .. } => terms,
        }
    } else {
        &meta.sorted_entry(info.number).unwrap().terms
    };
    (data, entry)
}

#[test]
fn the_terms_dictionary_answers_lookups_as_lucene_does() {
    let m = manifest();
    let reader = DirectoryReader::open(&FsDirectory::open(fixture_dir())).expect("open");
    let mut checked = 0;
    for (k, v) in &m {
        if let Some(rest) = k.strip_prefix("ord.") {
            let mut p = rest.splitn(3, '.');
            let seg: usize = p.next().unwrap().parse().unwrap();
            let field = p.next().unwrap();
            let ord: i64 = p.next().unwrap().parse().unwrap();
            let (data, entry) = terms_entry(&reader, seg, field);
            let mut dict = TermsDict::open(data, entry).unwrap();
            assert_eq!(dict.seek_ord(ord).unwrap(), unhex(v).as_slice(), "{k}");
            checked += 1;
        }
    }
    assert!(checked > 1000, "{checked} ordinals");
    let probes: usize = m["probe_count"].parse().unwrap();
    // One dictionary per (segment, field), reused across its probes, as a
    // comparator reuses its segment's: seeks go backwards and forwards.
    let mut failures = Vec::new();
    for i in 0..probes {
        let p: Vec<&str> = m[&format!("probe.{i}")].split(':').collect();
        let (seg, field, key, want): (usize, &str, Vec<u8>, i64) = (
            p[0].parse().unwrap(),
            p[1],
            unhex(p[2]),
            p[3].parse().unwrap(),
        );
        let (data, entry) = terms_entry(&reader, seg, field);
        let mut dict = TermsDict::open(data, entry).unwrap();
        let got = dict.lookup_term(&key).unwrap();
        if got != want {
            failures.push(format!(
                "seg {seg} {field} {key:?}: got {got}, Lucene {want}"
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {probes}:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

fn sort(spec: &str) -> Vec<SortField> {
    spec.split(',')
        .map(|k| {
            let p: Vec<&str> = k.split(':').collect();
            let ty = match p[1] {
                "string" => SortType::String,
                "score" => SortType::Score,
                "doc" => SortType::Doc,
                "int" => SortType::Int,
                "long" => SortType::Long,
                other => panic!("type {other}"),
            };
            SortField {
                field: p[0].to_string(),
                ty,
                selector: if p[2] == "max" {
                    Selector::Max
                } else {
                    Selector::Min
                },
                reverse: p[3] == "true",
                missing: p[4].parse().unwrap(),
            }
        })
        .collect()
}

fn field_doc(s: &str, keyword: bool) -> FieldDoc {
    let mut parts = s.split(':');
    let doc = parts.next().unwrap().parse().unwrap();
    let mut values = Vec::new();
    let mut terms = Vec::new();
    for v in parts {
        if v == "n" {
            values.push(0);
            terms.push(None);
        } else if let Some(h) = v.strip_prefix('x') {
            values.push(0);
            terms.push(Some(unhex(h)));
        } else {
            values.push(v.parse().unwrap());
            terms.push(None);
        }
    }
    if !keyword {
        terms.clear();
    }
    FieldDoc { doc, values, terms }
}

fn parse(toks: &[String], at: &mut usize) -> Clause {
    let mut next = || {
        *at += 1;
        toks[*at - 1].clone()
    };
    assert_eq!(next(), "(");
    let op = next();
    let q = match op.as_str() {
        "all" => Clause::MatchAllDocs(MatchAllDocsQuery::new(0)),
        "t" => Clause::Term(TermQuery::new("body", next().into_bytes())),
        "p" => {
            let mut words = Vec::new();
            while toks[*at] != ")" {
                *at += 1;
                words.push(toks[*at - 1].clone());
            }
            Clause::Phrase(PhraseQuery::new("body", words))
        }
        "r" => {
            let min = next().parse().unwrap();
            let max = next().parse().unwrap();
            Clause::PointsRange(PointsRangeQuery::new("r", min, max))
        }
        "b" => {
            let mut b = BooleanQuery::new();
            b.minimum_should_match = next().parse().unwrap();
            while toks[*at] == "(" {
                *at += 1;
                let occur = toks[*at].clone();
                *at += 1;
                let c = parse(toks, at);
                match occur.as_str() {
                    "+" => b.must.push(c),
                    "#" => b.filter.push(c),
                    "?" => b.should.push(c),
                    "-" => b.must_not.push(c),
                    other => panic!("occur {other}"),
                }
                *at += 1;
            }
            Clause::Boolean(Box::new(b))
        }
        other => panic!("op {other}"),
    };
    *at += 1;
    q
}

fn query(text: &str) -> BooleanQuery {
    let toks: Vec<String> = text
        .replace('(', " ( ")
        .replace(')', " ) ")
        .split_whitespace()
        .map(str::to_string)
        .collect();
    match parse(&toks, &mut 0) {
        Clause::Boolean(b) => *b,
        other => {
            let mut b = BooleanQuery::new();
            b.must.push(other);
            b
        }
    }
}

#[test]
fn keyword_sorts_match_real_lucene() {
    let m = manifest();
    let reader = DirectoryReader::open(&FsDirectory::open(fixture_dir())).expect("open");
    assert_eq!(reader.segment_readers().len(), 3);
    let mut opened = reader.open_segments().expect("open postings");
    opened.open_points().expect("open points");
    let segments = opened.as_open_segments();
    let owned: Vec<HashMap<String, FieldNorms<'_>>> = reader
        .field_norms("body")
        .into_iter()
        .map(|n| n.into_iter().map(|n| ("body".to_string(), n)).collect())
        .collect();
    let norms: Vec<Option<&HashMap<String, FieldNorms<'_>>>> = owned.iter().map(Some).collect();
    let runs: usize = m["run_count"].parse().unwrap();
    assert!(runs > 800);
    let mut failures = Vec::new();
    for r in 0..runs {
        let k = |s: &str| m[&format!("run.{r}.{s}")].clone();
        let text = k("query");
        let spec = k("sort");
        let keys = sort(&spec);
        let top_n: usize = k("top_n").parse().unwrap();
        let threshold: u64 = match k("threshold").as_str() {
            "max" => u64::MAX,
            n => n.parse().unwrap(),
        };
        let after = m.get(&format!("run.{r}.after")).map(|s| field_doc(s, true));
        let got = search_sorted(
            &segments,
            reader.segment_readers(),
            &query(&text),
            &norms,
            &keys,
            top_n,
            threshold,
            after.as_ref(),
        )
        .unwrap_or_else(|e| panic!("{text} by {spec}: {e}"));
        let want: Vec<FieldDoc> = match k("hits").as_str() {
            "" => Vec::new(),
            h => h.split(',').map(|s| field_doc(s, true)).collect(),
        };
        let total: u64 = k("total").parse().unwrap();
        let gte = k("relation") == "gte";
        let got_gte =
            got.total.relation == lucene_search::collector::TotalHitsRelation::GreaterThanOrEqualTo;
        let total_ok = if gte {
            got_gte && got.total.value > threshold
        } else {
            !got_gte && got.total.value == total
        };
        if got.hits != want || !total_ok {
            failures.push(format!(
                "run {r}: {text} by {spec}, top {top_n}, threshold {threshold}, after {after:?}\n  got    {:?} total {} gte {got_gte}\n  Lucene {:?} total {total} gte {gte}",
                got.hits.iter().take(3).collect::<Vec<_>>(),
                got.total.value,
                want.iter().take(3).collect::<Vec<_>>(),
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {runs} runs disagree with Lucene:\n{}",
        failures.len(),
        failures
            .iter()
            .take(10)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn the_terms_dictionary_reports_what_it_cannot_answer() {
    let reader = DirectoryReader::open(&FsDirectory::open(fixture_dir())).expect("open");
    let (data, entry) = terms_entry(&reader, 0, "k");
    let mut dict = TermsDict::open(data, entry).unwrap();
    assert_eq!(dict.ord(), -1, "before any seek");
    let size = dict.size();
    let last = dict.seek_ord(size - 1).unwrap().to_vec();
    assert_eq!((dict.ord(), dict.term()), (size - 1, last.as_slice()));
    // An ordinal outside the dictionary is an error, not a term.
    assert!(dict.seek_ord(-1).is_err());
    assert!(dict.seek_ord(size).is_err());
    // Past the last term: the end, and an insertion point after every ordinal.
    let mut past = last.clone();
    past.push(0xff);
    assert_eq!(dict.lookup_term(&past).unwrap(), -size - 1);
    // An entry without its random-access index cannot be opened for it.
    let mut bare = entry.clone();
    bare.index = None;
    assert!(TermsDict::open(data, &bare).is_err());
    // An empty dictionary ends at once.
    let mut empty = entry.clone();
    empty.terms_dict_size = 0;
    let mut dict = TermsDict::open(data, &empty).unwrap();
    assert_eq!(dict.lookup_term(b"anything").unwrap(), -1);
}
