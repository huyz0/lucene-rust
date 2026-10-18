//! M13 T13.3b (`suggest`) against Lucene 10.5.0 (`fixtures/src/GenSuggest.java`):
//! `TSTLookup`'s answers to every short prefix of a 324-entry corpus, its
//! `get`, and the bytes of `store()`, both ways.

use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

use lucene_search::suggest::input::InputIterator;
use lucene_search::suggest::lookup::{Lookup, LookupResult};
use lucene_search::suggest::tst::TstLookup;
use lucene_store::{SliceInput, VecDataOutput};

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/suggest")
}

/// `AnalysisRows.esc` undone.
fn unesc(s: &str) -> String {
    let mut out: Vec<u16> = Vec::new();
    let mut it = s.chars();
    while let Some(c) = it.next() {
        if c != '\\' {
            let mut buf = [0u16; 2];
            out.extend_from_slice(c.encode_utf16(&mut buf));
            continue;
        }
        match it.next() {
            Some('u') => {
                let h: String = it.by_ref().take(4).collect();
                out.push(u16::from_str_radix(&h, 16).unwrap());
            }
            Some('t') => out.push(9),
            Some('n') => out.push(10),
            Some('r') => out.push(13),
            Some(o) => out.push(o as u16),
            None => out.push(u16::from(b'\\')),
        }
    }
    String::from_utf16_lossy(&out)
}

/// The weighted terms of `terms.tsv`, in input order.
struct Terms {
    rows: Vec<(String, i64)>,
    next: usize,
}

impl InputIterator for Terms {
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

fn terms() -> Terms {
    let rows = fs::read_to_string(dir().join("terms.tsv"))
        .unwrap()
        .lines()
        .map(|l| {
            let (t, w) = l.split_once('\t').unwrap();
            (unesc(t), w.parse().unwrap())
        })
        .collect();
    Terms { rows, next: 0 }
}

fn built() -> TstLookup {
    let mut tst = TstLookup::new();
    tst.build(&mut terms()).unwrap();
    tst
}

fn rendered(res: &[LookupResult]) -> String {
    res.iter()
        .map(|r| format!("{}\u{1e}{}", r.key, r.value))
        .collect::<Vec<_>>()
        .join("\u{1f}")
}

#[test]
fn tst_lookup_answers_as_lucenes_does() {
    let tst = built();
    assert_eq!(tst.get_count().unwrap(), 324);
    let text = fs::read_to_string(dir().join("tst.tsv")).unwrap();
    let (mut checked, mut bad) = (0, Vec::new());
    for line in text.lines() {
        let c: Vec<&str> = line.split('\t').collect();
        let prefix = unesc(c[0]);
        let more = c[1] == "1";
        let num: usize = c[2].parse().unwrap();
        let got = match tst.lookup_no_contexts(&prefix, more, num) {
            Ok(res) => format!("R\t{}", rendered(&res)),
            // the fixture records the exception class; any error will do
            Err(_) => "X".to_string(),
        };
        let want = if c[3] == "R" {
            format!("R\t{}", unesc(c[4]))
        } else {
            "X".to_string()
        };
        checked += 1;
        if got != want {
            bad.push(format!(
                "{prefix:?} more={more} num={num}\n  want {want:?}\n  got  {got:?}"
            ));
        }
    }
    assert!(
        bad.is_empty(),
        "{} of {checked} differ:\n{}",
        bad.len(),
        bad.iter().take(5).cloned().collect::<Vec<_>>().join("\n")
    );
    assert_eq!(checked, 792);
}

#[test]
fn tst_get_equals_lucenes() {
    let tst = built();
    let text = fs::read_to_string(dir().join("tst_get.tsv")).unwrap();
    let mut checked = 0;
    for line in text.lines() {
        let (k, v) = line.split_once('\t').unwrap();
        let key = unesc(k);
        let want: Option<i64> = if v == "-" {
            None
        } else {
            Some(v.parse().unwrap())
        };
        assert_eq!(tst.get(&key).unwrap(), want, "get({key:?})");
        checked += 1;
    }
    assert_eq!(checked, 234);
}

#[test]
fn tst_store_writes_lucenes_bytes_and_loads_them() {
    let tst = built();
    let mut out = VecDataOutput::new();
    assert!(tst.store(&mut out).unwrap());
    let want = fs::read(dir().join("tst.bin")).unwrap();
    assert_eq!(out.into_inner(), want, "store() bytes");

    let mut loaded = TstLookup::new();
    assert!(loaded.load(&mut SliceInput::new(&want)).unwrap());
    assert_eq!(loaded.get_count().unwrap(), 324);
    // Java's own reload: the keys with a supplementary character are gone
    let text = fs::read_to_string(dir().join("tst_reload.tsv")).unwrap();
    let mut lost = 0;
    for line in text.lines() {
        let (k, v) = line.split_once('\t').unwrap();
        let key = unesc(k);
        let want: Option<i64> = if v == "-" {
            None
        } else {
            Some(v.parse().unwrap())
        };
        assert_eq!(loaded.get(&key).unwrap(), want, "after load: get({key:?})");
        lost += usize::from(want.is_none() && tst.get(&key).unwrap().is_some());
    }
    assert!(
        lost > 0,
        "the corpus should include keys that do not survive"
    );
    for prefix in ["ca", "x", "\u{65e5}", "a"] {
        assert_eq!(
            rendered(&loaded.lookup_no_contexts(prefix, true, 5).unwrap()),
            rendered(&tst.lookup_no_contexts(prefix, true, 5).unwrap()),
            "after load: {prefix:?}"
        );
    }
}
