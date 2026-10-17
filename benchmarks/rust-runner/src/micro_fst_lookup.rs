//! M13 T13.3b's pairs (`benchmarks/micro/java/FstLookupMicro.java` is the
//! Java twin): `GenSuggestFst`'s 324 weighted terms
//! (`fixtures/data/suggest_fst/terms.tsv`). `wfst_lookup`/`fstc_lookup`: top
//! 10 for every 1-3 character prefix of a term (WFST exact first;
//! FSTCompletion with ten buckets, higher weights first), units: lookups;
//! `wfst_build`/`fstc_build`: a build over all terms, units: terms.

use std::collections::{BTreeSet, HashSet};
use std::hint::black_box;
use std::time::Duration;

use lucene_search::suggest::fst_completion::FstCompletionLookup;
use lucene_search::suggest::input::InputIterator;
use lucene_search::suggest::lookup::Lookup;
use lucene_search::suggest::wfst::WfstCompletionLookup;

use super::measure;

/// The terms with their weights, as an `InputIterator`.
struct Entries<'a> {
    rows: &'a [(String, i64)],
    next: usize,
}

impl InputIterator for Entries<'_> {
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
        None
    }
    fn has_contexts(&self) -> bool {
        false
    }
}

fn entries(rows: &[(String, i64)]) -> Entries<'_> {
    Entries { rows, next: 0 }
}

/// Every 1-3 character prefix of a term, first-seen order (`LinkedHashSet`).
fn prefixes(rows: &[(String, i64)]) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for (t, _) in rows {
        let ends = t.char_indices().skip(1).map(|(i, _)| i);
        for end in ends.chain(std::iter::once(t.len())).take(3) {
            let p = t[..end].to_string();
            if seen.insert(p.clone()) {
                out.push(p);
            }
        }
    }
    out
}

fn lookups(name: &str, lk: &dyn Lookup, prefixes: &[String], flag: bool, w: Duration, m: Duration) {
    measure(name, w, m, || {
        for p in prefixes {
            black_box(lk.lookup(p, None, flag, 10).unwrap().len());
        }
        prefixes.len() as u64
    });
}

pub fn bench_fst_lookup(w: Duration, m: Duration) {
    let rows: Vec<(String, i64)> = std::fs::read_to_string("fixtures/data/suggest_fst/terms.tsv")
        .unwrap()
        .lines()
        .map(|l| {
            let (t, wt) = l.split_once('\t').unwrap();
            (t.to_string(), wt.parse().unwrap())
        })
        .collect();
    let prefixes = prefixes(&rows);
    let mut wfst = WfstCompletionLookup::with_exact_first(true);
    wfst.build(&mut entries(&rows)).unwrap();
    let mut fstc = FstCompletionLookup::with_options(10, true);
    fstc.build(&mut entries(&rows)).unwrap();
    lookups("wfst_lookup", &wfst, &prefixes, false, w, m);
    lookups("fstc_lookup", &fstc, &prefixes, true, w, m);
    measure("wfst_build", w, m, || {
        let mut b = WfstCompletionLookup::with_exact_first(true);
        b.build(&mut entries(&rows)).unwrap();
        black_box(b.get_count().unwrap());
        rows.len() as u64
    });
    measure("fstc_build", w, m, || {
        let mut b = FstCompletionLookup::with_options(10, true);
        b.build(&mut entries(&rows)).unwrap();
        black_box(b.get_count().unwrap());
        rows.len() as u64
    });
}
