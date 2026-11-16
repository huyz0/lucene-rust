#![cfg(test)]
use super::*;

/// `(text, weight, contexts)` rows.
/// A test row: text, weight, contexts.
type TestRow = (String, i64, Option<BTreeSet<Vec<u8>>>);

struct Rows {
    rows: Vec<TestRow>,
    next: usize,
}

impl InputIterator for Rows {
    fn next(&mut self) -> Option<Vec<u8>> {
        let row = self.rows.get(self.next)?;
        self.next += 1;
        Some(row.0.as_bytes().to_vec())
    }
    fn weight(&self) -> i64 {
        self.rows[self.next - 1].1
    }
    fn payload(&self) -> Option<&[u8]> {
        None
    }
    fn has_payloads(&self) -> bool {
        false
    }
    fn contexts(&self) -> Option<&BTreeSet<Vec<u8>>> {
        self.rows[self.next - 1].2.as_ref()
    }
    fn has_contexts(&self) -> bool {
        true
    }
}

const WORDS: [&str; 12] = [
    "apple", "apply", "ape", "banana", "band", "bandit", "cat", "catalog", "dog", "dot", "éclair",
    "écho",
];

fn rows(n: usize) -> Vec<TestRow> {
    let mut x = 0x2545_f491_4f6c_dd1du64;
    let mut next = move |m: usize| {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x % m as u64) as usize
    };
    (0..n)
        .map(|i| {
            let words = 1 + next(4);
            let text: Vec<&str> = (0..words).map(|_| WORDS[next(WORDS.len())]).collect();
            let contexts = match next(3) {
                0 => None,
                1 => Some(BTreeSet::from([b"red".to_vec()])),
                _ => Some(BTreeSet::from([b"red".to_vec(), b"blue".to_vec()])),
            };
            // weights collide, so ties keep insertion order
            (format!("{} {i}", text.join(" ")), next(50) as i64, contexts)
        })
        .collect()
}

fn suggester(n: usize) -> AnalyzingInfixSuggester {
    let analyzer = Arc::new(Analyzer::standard(None));
    let mut s = AnalyzingInfixSuggester::with_options(
        analyzer.clone(),
        analyzer,
        4,
        false,
        true,
        false,
        false,
    )
    .unwrap();
    s.build(&mut Rows {
        rows: rows(n),
        next: 0,
    })
    .unwrap();
    s
}

fn term(field: Field, t: &str) -> Query {
    Query::Term(field, t.as_bytes().to_vec())
}

fn leaves() -> Vec<Query> {
    vec![
        term(Field::Text, "apple"),
        term(Field::Text, "dot"),
        term(Field::Text, "nowhere"),
        term(Field::TextGrams, "ba"),
        term(Field::TextGrams, "é"),
        term(Field::Contexts, "blue"),
        Query::Prefix(b"band".to_vec()),
        Query::Prefix(b"zzzz".to_vec()),
        Query::Prefix(b"cata".to_vec()),
        Query::Term(Field::Text, vec![0xff]),
        Query::Prefix(vec![0xff]),
    ]
}

#[test]
fn top_hits_are_the_first_of_all_matches_for_every_query_shape() {
    // past the parallel inversion and analysis thresholds
    let s = suggester(PARALLEL_INVERSION_DOCS + 500);
    let snap = &s.searcher.as_ref().unwrap().base;
    // the parallel inversion is the sequential one
    let inv = Inverted::of(&snap.docs, 0);
    assert_eq!(inv.text.len(), snap.text.len());
    for (k, v) in &inv.text {
        assert_eq!(snap.text.get(*k), Some(v), "{k}");
    }
    assert_eq!(inv.grams.len(), snap.grams.len());
    for (k, v) in &inv.grams {
        assert_eq!(snap.grams.get(*k), Some(v), "{k}");
    }
    assert_eq!(inv.contexts.len(), snap.contexts.len());

    let n = leaves().len();
    let mut queries = Vec::new();
    for a in 0..n {
        for b in 0..n {
            for (oa, ob) in [
                (Occur::Must, Occur::Must),
                (Occur::Must, Occur::MustNot),
                (Occur::Must, Occur::Should),
                (Occur::Should, Occur::Should),
                (Occur::Filter, Occur::Must),
                (Occur::MustNot, Occur::MustNot),
            ] {
                let (la, lb) = (leaves().swap_remove(a), leaves().swap_remove(b));
                queries.push(Query::Bool(vec![(oa, la), (ob, lb)]));
            }
        }
        // a lone clause, and one beside a context sub-query
        queries.push(Query::Bool(vec![(Occur::Must, leaves().swap_remove(a))]));
        queries.push(Query::Bool(vec![
            (Occur::Must, leaves().swap_remove(a)),
            (
                Occur::Must,
                Query::Bool(vec![
                    (Occur::Should, term(Field::Contexts, "red")),
                    (Occur::Should, term(Field::Contexts, "green")),
                ]),
            ),
            (Occur::MustNot, term(Field::Contexts, "blue")),
        ]));
    }
    queries.push(Query::Bool(Vec::new()));
    for q in &queries {
        let all = snap.matching(q);
        for num in [1, 7, 100, usize::MAX] {
            let want: Vec<usize> = all.iter().copied().take(num).collect();
            assert_eq!(snap.top(q, num, &[], usize::MAX), want, "num {num}");
            // every third document dead, the documents cut at `end`
            let dead: Vec<u64> = (0..snap.docs.len().div_ceil(64))
                .map(|w| {
                    (0..64)
                        .filter(|b| (w * 64 + b) % 3 == 0)
                        .fold(0u64, |m, b| m | 1 << b)
                })
                .collect();
            for end in [0, 5, snap.docs.len() / 2] {
                let want: Vec<usize> = all
                    .iter()
                    .copied()
                    .filter(|&d| d < end && d % 3 != 0)
                    .take(num)
                    .collect();
                assert_eq!(snap.top(q, num, &dead, end), want, "num {num} end {end}");
            }
        }
        for d in [0, 1, all.first().copied().unwrap_or(2)] {
            assert_eq!(snap.doc_matches(d, q), all.contains(&d));
        }
    }
    assert!(!snap.doc_matches(snap.docs.len(), &term(Field::Text, "apple")));
}

#[test]
fn a_rare_prefix_falls_back_to_its_postings() {
    // the only "bandit" documents are the last ones: the scan's budget
    // (the prefix's postings) runs out before it reaches them
    let s = suggester(300);
    let snap = &s.searcher.as_ref().unwrap().base;
    let q = Query::Bool(vec![(Occur::Must, Query::Prefix(b"bandi".to_vec()))]);
    let all = snap.matching(&q);
    assert!(!all.is_empty());
    assert_eq!(snap.top(&q, 1000, &[], usize::MAX), all.to_vec());
    assert_eq!(snap.top(&q, 1, &[], usize::MAX), all[..1].to_vec());
}

#[test]
fn edge_ngrams_are_code_point_prefixes() {
    let s = suggester(0);
    let doc = s
        .build_document("éclair ab".as_bytes(), None, 1, None)
        .unwrap();
    let grams: Vec<&str> = doc.grams.iter().map(String::as_str).collect();
    assert_eq!(grams, ["a", "ab", "é", "éc", "écl", "écla"]);
}

#[test]
fn update_replaces_every_copy_and_refresh_skips_an_unchanged_view() {
    let mut s = suggester(50);
    let first = s.searcher.as_ref().unwrap().base.docs.as_ptr();
    s.refresh().unwrap();
    assert_eq!(s.searcher.as_ref().unwrap().base.docs.as_ptr(), first);

    // a model of the writer: (text, weight) in insertion order
    let mut model: Vec<(String, i64)> = rows(50).into_iter().map(|r| (r.0, r.1)).collect();
    s.add(b"dup", None, 3, None).unwrap();
    s.add(b"dup", None, 4, None).unwrap();
    model.push(("dup".into(), 3));
    model.push(("dup".into(), 4));
    let texts: Vec<String> = model.iter().map(|m| m.0.clone()).collect();
    for round in 0..3000usize {
        let text = &texts[round * 7 % texts.len()];
        let weight = (round % 40) as i64;
        s.update(text.as_bytes(), None, weight, None).unwrap();
        model.retain(|(t, _)| t != text);
        model.push((text.clone(), weight));
    }
    // the holes were dropped along the way
    assert!(s.deleted < s.docs.len());
    let live: Vec<(String, i64)> = s
        .docs
        .iter()
        .flatten()
        .map(|d| (d.text.clone(), d.weight))
        .collect();
    assert_eq!(live, model);
    s.refresh().unwrap();
    let mut sorted = model.clone();
    sorted.sort_by_key(|m| std::cmp::Reverse(m.1));
    let view: Vec<(String, i64)> = s
        .searcher
        .as_ref()
        .unwrap()
        .live_docs()
        .into_iter()
        .map(|d| (d.text.clone(), d.weight))
        .collect();
    assert_eq!(view, sorted);
    assert_eq!(s.get_count().unwrap(), 51);
    let hits = s.lookup_with("dup", None, 5, true, false).unwrap();
    assert_eq!(hits.len(), 1);

    // a closed suggester reopened by `add` sees its documents again
    s.close();
    s.add(b"late", None, 1, None).unwrap();
    s.refresh().unwrap();
    assert_eq!(s.get_count().unwrap(), 52);
}

/// Every lookup shape the test below asks, as `Debug` text.
fn lookups(s: &AnalyzingInfixSuggester) -> Vec<String> {
    let red = BTreeSet::from([b"red".to_vec()]);
    let not_blue = BTreeMap::from([(b"blue".to_vec(), Occur::MustNot)]);
    let red_and_blue = BTreeMap::from([
        (b"red".to_vec(), Occur::Must),
        (b"blue".to_vec(), Occur::Should),
    ]);
    let mut out = Vec::new();
    for key in [
        "a", "ap", "appl", "apple", "apple ", "band", "ba ca", "cat dog", "é", "écl", "zzz", "dup",
    ] {
        for num in [1, 4, 30] {
            for all in [true, false] {
                out.push(format!("{:?}", s.lookup_with(key, None, num, all, true)));
                out.push(format!(
                    "{:?}",
                    s.lookup_with(key, Some(&red), num, all, false)
                ));
                out.push(format!(
                    "{:?}",
                    s.lookup_context_info(key, &not_blue, num, all, false)
                ));
                out.push(format!(
                    "{:?}",
                    s.lookup_context_info(key, &red_and_blue, num, all, false)
                ));
            }
        }
    }
    out.push(format!("{:?}", s.get_count()));
    out
}

/// An incremental refresh (dead base documents, an added run, compaction)
/// reads as a view made again from every document: the same lookups, in
/// the same order, under random interleaved adds, updates, refreshes,
/// commits and closes.
#[test]
fn incremental_refreshes_read_as_full_rebuilds() {
    let base_rows = rows(1500);
    let mut s = suggester(1500);
    let mut model = suggester(1500);
    let mut x = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move |m: usize| {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x % m as u64) as usize
    };
    let mut texts: Vec<String> = base_rows.iter().map(|r| r.0.clone()).collect();
    let (mut incremental, mut compactions) = (0, 0);
    for step in 0..1200 {
        let op = next(100);
        let weight = next(60) as i64;
        let contexts = match next(3) {
            0 => None,
            1 => Some(BTreeSet::from([b"red".to_vec()])),
            _ => Some(BTreeSet::from([b"blue".to_vec()])),
        };
        let payload = (next(4) == 0).then(|| format!("p{step}").into_bytes());
        if op < 40 {
            // an update of a text already there (often one just added)
            let text = if next(3) == 0 {
                texts[texts.len() - 1 - next(texts.len().min(20))].clone()
            } else {
                texts[next(texts.len())].clone()
            };
            for t in [&mut s, &mut model] {
                t.update(
                    text.as_bytes(),
                    contexts.as_ref(),
                    weight,
                    payload.as_deref(),
                )
                .unwrap();
            }
        } else if op < 70 {
            let words = 1 + next(3);
            let mut text: Vec<&str> = (0..words).map(|_| WORDS[next(WORDS.len())]).collect();
            if next(10) == 0 {
                text = vec!["dup"];
            }
            let text = format!(
                "{} {}",
                text.join(" "),
                if text == ["dup"] { 0 } else { step }
            );
            texts.push(text.clone());
            for t in [&mut s, &mut model] {
                t.add(
                    text.as_bytes(),
                    contexts.as_ref(),
                    weight,
                    payload.as_deref(),
                )
                .unwrap();
            }
        } else if op < 90 {
            let before = s.searcher.as_ref().map(|v| v.base.docs.as_ptr());
            s.refresh().unwrap();
            model.rebuild = true;
            model.refresh().unwrap();
            let view = s.searcher.as_ref().unwrap();
            if before == Some(view.base.docs.as_ptr()) {
                incremental += usize::from(!view.added.docs.is_empty() || view.base_dead > 0);
            } else {
                compactions += 1;
            }
            assert_eq!(model.searcher.as_ref().unwrap().base_dead, 0);
            let docs = |v: &View| -> Vec<(String, i64, u64)> {
                v.live_docs()
                    .into_iter()
                    .map(|d| (d.text.clone(), d.weight, d.seq))
                    .collect()
            };
            assert_eq!(
                docs(view),
                docs(model.searcher.as_ref().unwrap()),
                "step {step}"
            );
        } else if op < 97 {
            // lookups between refreshes see the last view only
            assert_eq!(lookups(&s), lookups(&model), "step {step}");
        } else if op < 99 {
            s.commit().unwrap();
            model.commit().unwrap();
        } else {
            s.close();
            model.close();
            // reopened by the next write; a lookup needs a view again
            for t in [&mut s, &mut model] {
                t.add(b"reopened", None, 5, None).unwrap();
                t.refresh().unwrap();
            }
        }
    }
    s.refresh().unwrap();
    model.rebuild = true;
    model.refresh().unwrap();
    assert_eq!(lookups(&s), lookups(&model));
    // both paths ran
    assert!(incremental > 20, "{incremental}");
    assert!(compactions > 2, "{compactions}");
}
