#![cfg(test)]
use super::*;
use crate::tree::TermWeightor;
use lucene_search::query::{PrefixQuery, TermQuery};

struct Rng(u64);

impl Rng {
    fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % n as u64) as usize
    }
}

const WORDS: [&str; 12] = [
    "fox", "dog", "cat", "owl", "elk", "bee", "ant", "yak", "emu", "gnu", "ram", "hen",
];

fn term(field: &str, text: &str) -> Clause {
    Clause::Term(TermQuery::new(field, text.as_bytes().to_vec()))
}

/// A random query over two fields: a term, a prefix (any-token), or a
/// boolean of terms with every occurrence.
fn random_query(rng: &mut Rng, depth: usize) -> Clause {
    let field = ["f", "g"][rng.below(2)];
    match rng.below(if depth == 0 { 2 } else { 4 }) {
        0 => term(field, WORDS[rng.below(WORDS.len())]),
        1 if rng.below(4) == 0 => Clause::Prefix(PrefixQuery::new(field, b"z".to_vec())),
        1 => term(field, WORDS[rng.below(WORDS.len())]),
        _ => {
            let mut b = BooleanQuery::new();
            for _ in 0..1 + rng.below(3) {
                let c = random_query(rng, depth - 1);
                match rng.below(4) {
                    0 => b.must.push(c),
                    1 => b.filter.push(c),
                    2 if !b.should.is_empty() || !b.must.is_empty() => b.must_not.push(c),
                    _ => b.should.push(c),
                }
            }
            Clause::Boolean(Box::new(b))
        }
    }
}

fn random_batch(rng: &mut Rng) -> BatchTerms {
    let mut batch = BatchTerms::new();
    for field in ["f", "g"] {
        let mut terms: Vec<Vec<u8>> = (0..rng.below(4))
            .map(|_| WORDS[rng.below(WORDS.len())].as_bytes().to_vec())
            .collect();
        terms.sort_unstable();
        terms.dedup();
        batch.insert(field.to_string(), terms);
    }
    batch
}

fn ids(entries: &[&CachedEntry]) -> Vec<String> {
    entries.iter().map(|e| e.cache_id.clone()).collect()
}

/// Every live entry `matches_presearcher` selects, entry by entry.
fn brute_force<'a>(index: &'a QueryIndex, query: &Clause) -> Vec<&'a CachedEntry> {
    index
        .slots
        .iter()
        .flatten()
        .filter(|(indexed, _)| matches_presearcher(query, indexed))
        .map(|(_, e)| e)
        .collect()
}

#[test]
fn the_inverted_selection_equals_the_entry_by_entry_one() {
    let presearchers = [
        TermFilteredPresearcher::default(),
        TermFilteredPresearcher::multipass(3, 0.0, TermWeightor::default(), Vec::new()),
        TermFilteredPresearcher::new(TermWeightor::default(), vec!["lang".to_string()]),
    ];
    for (p, presearcher) in presearchers.into_iter().enumerate() {
        let mut rng = Rng(0x9e37_79b9 + p as u64);
        let mut index = QueryIndex::new(presearcher);
        for round in 0..6 {
            // registrations that replace earlier ids, and deletions, so
            // slots empty out and the index compacts
            let queries: Vec<MonitorQuery> = (0..60)
                .map(|_| {
                    let id = format!("q{}", rng.below(120));
                    let mut md = BTreeMap::new();
                    if rng.below(2) == 0 {
                        md.insert("lang".to_string(), Some(["en", "de"][rng.below(2)].into()));
                    }
                    MonitorQuery::new(id, random_query(&mut rng, 2), None, md).unwrap()
                })
                .collect();
            index.register(queries).unwrap();
            let gone: Vec<String> = (0..10).map(|_| format!("q{}", rng.below(120))).collect();
            index.delete(&gone).unwrap();
            assert_eq!(
                index.disjunct_count(),
                index.slots.iter().flatten().count(),
                "round {round}"
            );
            for _ in 0..40 {
                let batch = random_batch(&mut rng);
                let filter = match rng.below(3) {
                    0 => None,
                    _ => {
                        let mut values = BatchDocFreqs::new();
                        values.insert(
                            "lang".to_string(),
                            vec![(["en", "de"][rng.below(2)].as_bytes().to_vec(), 1)],
                        );
                        index.batch_filter(&values, 1).unwrap()
                    }
                };
                let query = index
                    .presearcher
                    .build_query(&batch, &AcceptAll, filter.clone());
                let want = ids(&brute_force(&index, &query));
                let got = ids(&index.select(&batch, &AcceptAll, filter.clone()));
                assert_eq!(got, want, "presearcher {p} round {round} {query:?}");
                let debug: Vec<String> = index
                    .select_debug(&batch, &AcceptAll, filter)
                    .iter()
                    .map(|(e, _)| e.cache_id.clone())
                    .collect();
                assert_eq!(debug, want);
            }
        }
    }
}

#[test]
fn replacing_and_deleting_keep_ids_and_compaction_keeps_order() {
    let mut index = QueryIndex::new(TermFilteredPresearcher::default());
    let mq =
        |id: &str, t: &str| MonitorQuery::new(id, term("f", t), None, BTreeMap::new()).unwrap();
    index
        .register((0..100).map(|i| mq(&format!("q{i:03}"), "fox")).collect())
        .unwrap();
    // replace the even ids: over half the slots empty, so it compacts
    index
        .register(
            (0..100)
                .step_by(2)
                .map(|i| mq(&format!("q{i:03}"), "fox"))
                .collect(),
        )
        .unwrap();
    index
        .register(
            (0..100)
                .step_by(3)
                .map(|i| mq(&format!("q{i:03}"), "fox"))
                .collect(),
        )
        .unwrap();
    assert_eq!(index.slots.len(), 184);
    // the odd ids not a multiple of three: 217 slots for 100 entries,
    // so the index compacts
    index
        .register(
            (0..100)
                .filter(|i| i % 2 == 1 && i % 3 != 0)
                .map(|i| mq(&format!("q{i:03}"), "fox"))
                .collect(),
        )
        .unwrap();
    assert_eq!(index.disjunct_count(), 100);
    assert_eq!(index.slots.len(), 100);
    let mut batch = BatchTerms::new();
    batch.insert("f".into(), vec![b"fox".to_vec()]);
    let got: Vec<String> = index
        .select(&batch, &AcceptAll, None)
        .iter()
        .map(|e| e.query_id.clone())
        .collect();
    // registration order: each id where its last registration put it
    let mut want: Vec<String> = Vec::new();
    let last = |i: usize| {
        if i.is_multiple_of(3) {
            2
        } else if i.is_multiple_of(2) {
            1
        } else {
            3
        }
    };
    for round in 1..=3 {
        want.extend(
            (0..100)
                .filter(|&i| last(i) == round)
                .map(|i| format!("q{i:03}")),
        );
    }
    assert_eq!(got, want);
    assert_eq!(index.query_ids().len(), 100);
    assert!(index.get_query("q050").is_some());
    index.delete(&["q050".to_string()]).unwrap();
    assert!(index.get_query("q050").is_none());
    assert_eq!(index.cache_stats().queries, 99);
}
