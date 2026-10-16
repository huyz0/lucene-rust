//! M13 T13.3a/b against Lucene 10.5.0 (`fixtures/src/GenSuggestFst.java`):
//! `WFSTCompletionLookup`, `FSTCompletionLookup`, `FileDictionary`,
//! `DocumentDictionary` and `DocumentValueSourceDictionary` answer as Lucene's
//! do, and the two lookups write and read Lucene's bytes.

use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

use lucene_search::directory_reader::DirectoryReader;
use lucene_search::spell::Dictionary;
use lucene_search::suggest::dictionary::{
    DocumentDictionary, DocumentValueSourceDictionary, FileDictionary,
};
use lucene_search::suggest::fst_completion::FstCompletionLookup;
use lucene_search::suggest::input::{InputIterator, JavaRandom};
use lucene_search::suggest::lookup::{Lookup, LookupResult};
use lucene_search::suggest::wfst::WfstCompletionLookup;
use lucene_search::values_source::{long_constant, long_from_long_field};
use lucene_store::{FsDirectory, SliceInput, VecDataOutput};

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/suggest_fst")
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

fn terms_of(rows: &[(&str, i64)]) -> Terms {
    Terms {
        rows: rows.iter().map(|(t, w)| (t.to_string(), *w)).collect(),
        next: 0,
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

fn rendered(res: &[LookupResult]) -> String {
    res.iter()
        .map(|r| format!("{}\u{1e}{}", r.key, r.value))
        .collect::<Vec<_>>()
        .join("\u{1f}")
}

/// Every row of a lookup answer file against `lk`; the row count.
fn check_lookups(lk: &dyn Lookup, file: &str) -> usize {
    let text = fs::read_to_string(dir().join(file)).unwrap();
    let (mut checked, mut bad) = (0, Vec::new());
    for line in text.lines() {
        let c: Vec<&str> = line.split('\t').collect();
        let prefix = unesc(c[0]);
        let flag = c[1] == "1";
        let num: usize = c[2].parse().unwrap();
        let got = match lk.lookup_no_contexts(&prefix, flag, num) {
            Ok(res) => format!("R\t{}", rendered(&res)),
            Err(_) => "X".to_string(),
        };
        let want = if c[3] == "R" {
            format!("R\t{}", unesc(c.get(4).copied().unwrap_or("")))
        } else {
            "X".to_string()
        };
        checked += 1;
        if got != want {
            bad.push(format!(
                "{file}: {prefix:?} flag={flag} num={num}\n  want {want:?}\n  got  {got:?}"
            ));
        }
    }
    assert!(
        bad.is_empty(),
        "{} of {checked} differ:\n{}",
        bad.len(),
        bad.iter().take(5).cloned().collect::<Vec<_>>().join("\n")
    );
    checked
}

fn check_gets(file: &str, get: impl Fn(&str) -> Option<i64>) -> usize {
    let text = fs::read_to_string(dir().join(file)).unwrap();
    let mut checked = 0;
    for line in text.lines() {
        let (k, v) = line.split_once('\t').unwrap();
        let key = unesc(k);
        let want: Option<i64> = if v == "-" {
            None
        } else {
            Some(v.parse().unwrap())
        };
        assert_eq!(get(&key), want, "{file}: get({key:?})");
        checked += 1;
    }
    checked
}

fn stored(lk: &dyn Lookup) -> (bool, Vec<u8>) {
    let mut out = VecDataOutput::new();
    let ok = lk.store(&mut out).unwrap();
    (ok, out.into_inner())
}

fn wfst(exact_first: bool) -> WfstCompletionLookup {
    let mut w = WfstCompletionLookup::with_exact_first(exact_first);
    w.build(&mut terms()).unwrap();
    w
}

#[test]
fn wfst_answers_as_lucenes_does() {
    for (tag, exact) in [("exact", true), ("plain", false)] {
        let w = wfst(exact);
        let count: u64 = fs::read_to_string(dir().join(format!("wfst_{tag}.count")))
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        assert_eq!(w.get_count().unwrap(), count);
        assert_eq!(check_lookups(&w, &format!("wfst_{tag}.tsv")), 396);
        let n = check_gets(&format!("wfst_get_{tag}.tsv"), |k| {
            w.get(k).unwrap().map(i64::from)
        });
        assert!(n > 200);
    }
}

#[test]
fn wfst_store_writes_lucenes_bytes_and_loads_them() {
    for (tag, exact) in [("exact", true), ("plain", false)] {
        let w = wfst(exact);
        let (ok, bytes) = stored(&w);
        assert!(ok);
        assert_eq!(
            bytes,
            fs::read(dir().join(format!("wfst_{tag}.bin"))).unwrap(),
            "store() bytes ({tag})"
        );
        let mut loaded = WfstCompletionLookup::with_exact_first(exact);
        assert!(loaded.load(&mut SliceInput::new(&bytes)).unwrap());
        assert_eq!(loaded.get_count().unwrap(), w.get_count().unwrap());
        assert_eq!(
            check_lookups(&loaded, &format!("wfst_{tag}_reload.tsv")),
            396
        );
    }
}

#[test]
fn wfst_empty_and_edge_cases() {
    let mut empty = WfstCompletionLookup::new();
    empty.build(&mut terms_of(&[])).unwrap();
    let (ok, bytes) = stored(&empty);
    assert!(!ok);
    assert_eq!(bytes, fs::read(dir().join("wfst_empty.bin")).unwrap());
    assert_eq!(
        fs::read_to_string(dir().join("wfst_empty.stored")).unwrap(),
        "false\n"
    );
    let got = format!(
        "R\t{}\n",
        rendered(&empty.lookup_no_contexts("a", false, 3).unwrap())
    );
    assert_eq!(
        got,
        fs::read_to_string(dir().join("wfst_empty.lookup")).unwrap()
    );
    assert_eq!(empty.get("a").unwrap(), None);

    // the heaviest duplicate wins; 0 and Integer.MAX_VALUE round trip
    let mut w = WfstCompletionLookup::new();
    w.build(&mut terms_of(&[
        ("dup", 5),
        ("dup", 9),
        ("dup", 1),
        ("zero", 0),
        ("max", i64::from(i32::MAX)),
        ("dupx", 9),
    ]))
    .unwrap();
    for line in fs::read_to_string(dir().join("wfst_edge.tsv"))
        .unwrap()
        .lines()
    {
        let c: Vec<&str> = line.split('\t').collect();
        if c[0] == "count" {
            assert_eq!(w.get_count().unwrap().to_string(), c[1]);
            continue;
        }
        let got = format!(
            "R\t{}",
            rendered(
                &w.lookup_no_contexts(c[0], false, c[1].parse().unwrap())
                    .unwrap()
            )
        );
        assert_eq!(got, format!("R\t{}", unesc(c[3])), "{line:?}");
    }
}

fn error_rows(file: &str) -> Vec<(String, Vec<String>)> {
    fs::read_to_string(dir().join(file))
        .unwrap()
        .lines()
        .map(|l| {
            let c: Vec<String> = l.split('\t').map(String::from).collect();
            (c[0].clone(), c[1..].to_vec())
        })
        .collect()
}

/// The Java message of an error row, `X <class> <message>` or `<class> <message>`.
fn message(cols: &[String]) -> String {
    let rest: Vec<&String> = cols.iter().skip_while(|c| *c == "X").collect();
    unesc(rest[1])
}

#[test]
fn wfst_refuses_what_lucene_refuses() {
    let w = wfst(true);
    let ctx: BTreeSet<Vec<u8>> = [b"c".to_vec()].into_iter().collect();
    for (case, cols) in error_rows("wfst_errors.tsv") {
        let want = message(&cols);
        let err = match case.as_str() {
            "only_more_popular" => w.lookup_no_contexts("ca", true, 3).unwrap_err(),
            "contexts" => w.lookup("ca", Some(&ctx), false, 3).unwrap_err(),
            c => {
                let weight: i64 = c.strip_prefix("weight_").unwrap().parse().unwrap();
                WfstCompletionLookup::new()
                    .build(&mut terms_of(&[("a", weight)]))
                    .unwrap_err()
            }
        };
        assert!(err.to_string().contains(&want), "{case}: {err} vs {want}");
    }
}

fn fstc(buckets: i32, exact: bool) -> FstCompletionLookup {
    let mut f = FstCompletionLookup::with_options(buckets, exact);
    f.build(&mut terms()).unwrap();
    f
}

const FSTC_CASES: [(&str, i32, bool); 3] =
    [("b10e", 10, true), ("b3n", 3, false), ("b1e", 1, true)];

#[test]
fn fst_completion_answers_as_lucenes_does() {
    for (tag, buckets, exact) in FSTC_CASES {
        let f = fstc(buckets, exact);
        let count: u64 = fs::read_to_string(dir().join(format!("fstc_{tag}.count")))
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        assert_eq!(f.get_count().unwrap(), count);
        assert_eq!(check_lookups(&f, &format!("fstc_{tag}.tsv")), 792);
        check_gets(&format!("fstc_get_{tag}.tsv"), |k| f.get(k).unwrap());
    }
}

#[test]
fn fst_completion_store_writes_lucenes_bytes_and_loads_them() {
    for (tag, buckets, exact) in FSTC_CASES {
        let f = fstc(buckets, exact);
        let (ok, bytes) = stored(&f);
        assert!(ok);
        assert_eq!(
            bytes,
            fs::read(dir().join(format!("fstc_{tag}.bin"))).unwrap(),
            "store() bytes ({tag})"
        );
        let mut loaded = FstCompletionLookup::with_options(buckets, exact);
        assert!(loaded.load(&mut SliceInput::new(&bytes)).unwrap());
        assert_eq!(loaded.get_count().unwrap(), f.get_count().unwrap());
        assert_eq!(
            check_lookups(&loaded, &format!("fstc_{tag}_reload.tsv")),
            792
        );
    }
    let mut empty = FstCompletionLookup::new();
    empty.build(&mut terms_of(&[])).unwrap();
    let (ok, bytes) = stored(&empty);
    assert!(!ok);
    assert_eq!(bytes, fs::read(dir().join("fstc_empty.bin")).unwrap());
    assert_eq!(
        fs::read_to_string(dir().join("fstc_empty.stored")).unwrap(),
        "false\n"
    );
    assert!(empty.lookup_no_contexts("a", true, 3).unwrap().is_empty());
}

/// Mixed-sign weights: the big-endian weight prefix sorts as unsigned bytes,
/// so negative weights take the highest buckets, as in Lucene.
#[test]
fn fst_completion_buckets_mixed_sign_weights_as_lucene_does() {
    let rows = [
        ("neg", -5),
        ("negone", -1),
        ("nil", 0),
        ("nine", 9),
        ("ninety", 90),
        ("nmin", i64::from(i32::MIN)),
        ("nmax", i64::from(i32::MAX)),
        ("nope", 3),
    ];
    let mut f = FstCompletionLookup::with_options(4, true);
    f.build(&mut terms_of(&rows)).unwrap();
    assert!(check_lookups(&f, "fstc_signed.tsv") > 0);
    assert_eq!(check_gets("fstc_get_signed.tsv", |k| f.get(k).unwrap()), 11);
    let (ok, bytes) = stored(&f);
    assert!(ok);
    assert_eq!(bytes, fs::read(dir().join("fstc_signed.bin")).unwrap());
}

#[test]
fn fst_completion_refuses_what_lucene_refuses() {
    use lucene_search::suggest::fst_completion::{FstCompletionBuilder, InMemorySorter};
    let f = fstc(10, true);
    let ctx: BTreeSet<Vec<u8>> = [b"c".to_vec()].into_iter().collect();
    let mut builder =
        FstCompletionBuilder::with_sorter(4, Box::new(InMemorySorter::natural()), 5.0).unwrap();
    for (case, cols) in error_rows("fstc_errors.tsv") {
        let want = message(&cols);
        let err = if case == "contexts" {
            f.lookup("a", Some(&ctx), false, 3).unwrap_err()
        } else if let Some(w) = case.strip_prefix("weight_") {
            FstCompletionLookup::new()
                .build(&mut terms_of(&[("a", w.parse().unwrap())]))
                .unwrap_err()
        } else if let Some(b) = case.strip_prefix("buckets_") {
            FstCompletionBuilder::with_sorter(
                b.parse().unwrap(),
                Box::new(InMemorySorter::natural()),
                5.0,
            )
            .err()
            .unwrap()
        } else {
            let b: i32 = case.strip_prefix("add_bucket_").unwrap().parse().unwrap();
            builder.add(b"x", b).unwrap_err()
        };
        assert!(err.to_string().contains(&want), "{case}: {err} vs {want}");
    }
}

/// `dictRows`' text for a Rust dictionary: the `P` row, then a `T` row per
/// entry, or an `X` row for an error.
fn dict_rows(d: &dyn Dictionary) -> Vec<String> {
    let mut it = match d.get_entry_iterator() {
        Ok(it) => it,
        Err(e) => return vec![format!("X\t{e}")],
    };
    let hex = |b: &[u8]| {
        if b.is_empty() {
            ".".to_string()
        } else {
            b.iter().map(|x| format!("{x:02x}")).collect()
        }
    };
    let mut rows = vec![format!(
        "P\t{}\t{}",
        u8::from(it.has_payloads()),
        u8::from(it.has_contexts())
    )];
    while let Some(term) = it.next() {
        let payload = if it.has_payloads() {
            hex(it.payload().unwrap())
        } else {
            "-".into()
        };
        let contexts = if it.has_contexts() {
            let set: BTreeSet<String> = it.contexts().unwrap().iter().map(|c| hex(c)).collect();
            if set.is_empty() {
                ".".to_string()
            } else {
                set.into_iter().collect::<Vec<_>>().join(",")
            }
        } else {
            "-".into()
        };
        rows.push(format!(
            "T\t{}\t{}\t{payload}\t{contexts}",
            String::from_utf8_lossy(&term),
            it.weight()
        ));
    }
    rows
}

/// A `.tsv` of dictionary rows: the terms unescaped; an `X` row is kept as
/// its message so the caller can look for it in the Rust error.
fn dict_expected(file: &str) -> Vec<String> {
    fs::read_to_string(dir().join(file))
        .unwrap()
        .lines()
        .map(|l| {
            let c: Vec<&str> = l.split('\t').collect();
            match c[0] {
                "T" => format!("T\t{}\t{}\t{}\t{}", unesc(c[1]), c[2], c[3], c[4]),
                "X" => format!("X\t{}", unesc(c[2])),
                _ => l.to_string(),
            }
        })
        .collect()
}

fn assert_dict(got: &[String], want: &[String], what: &str) {
    if want[0].starts_with("X\t") {
        assert!(
            got[0].starts_with("X\t") && got[0].contains(&want[0][2..]),
            "{what}: want {want:?}, got {got:?}"
        );
    } else {
        assert_eq!(got, want, "{what}");
    }
}

#[test]
fn file_dictionary_reads_as_lucenes_does() {
    let mut checked = 0;
    for name in [
        "dict_payload",
        "dict_plain",
        "dict_spaces",
        "dict_empty",
        "dict_only_term",
        "dict_bad4",
        "dict_badweight",
        "dict_badfirst",
        "dict_alltabs",
        "dict_lowernan",
        "dict_upperinf",
        "dict_padbad",
        "dict_emptyweight",
    ] {
        let bytes = fs::read(dir().join(format!("{name}.txt"))).unwrap();
        let d = FileDictionary::from_utf8(&bytes).unwrap();
        assert_dict(&dict_rows(&d), &dict_expected(&format!("{name}.tsv")), name);
        checked += 1;
    }
    let bytes = fs::read(dir().join("dict_semicolon.txt")).unwrap();
    let d = FileDictionary::from_utf8_with_delimiter(&bytes, ";").unwrap();
    assert_dict(
        &dict_rows(&d),
        &dict_expected("dict_semicolon.tsv"),
        "semicolon",
    );
    assert_eq!(checked, 13);
}

#[test]
fn file_dictionary_refuses_bytes_that_are_not_utf8() {
    assert!(FileDictionary::from_utf8(&[b'a', 0xff, b'\n']).is_err());
    // from_text takes already decoded text
    let d = FileDictionary::from_text("a\t2\r\nb\t3\n");
    assert_eq!(
        dict_rows(&d),
        vec!["P\t0\t0", "T\ta\t2\t-\t-", "T\tb\t3\t-\t-"]
    );
}

fn index() -> DirectoryReader {
    DirectoryReader::open(&FsDirectory::open(dir().join("index"))).unwrap()
}

#[test]
fn document_dictionaries_read_as_lucenes_do() {
    let r = index();
    let cases: Vec<(&str, Box<dyn Dictionary + '_>)> = vec![
        (
            "docdict_weight",
            Box::new(DocumentDictionary::new(&r, "title", "weight")),
        ),
        (
            "docdict_dv",
            Box::new(DocumentDictionary::new(&r, "title", "w2")),
        ),
        (
            "docdict_full",
            Box::new(
                DocumentDictionary::new(&r, "title", "weight")
                    .with_payload("payload")
                    .with_contexts("ctx"),
            ),
        ),
        (
            "docdict_none",
            Box::new(DocumentDictionary::new(&r, "title", "missing")),
        ),
        (
            "docdict_nofield",
            Box::new(DocumentDictionary::new(&r, "nofield", "weight")),
        ),
        (
            "docvs_dv",
            Box::new(DocumentValueSourceDictionary::new(
                &r,
                "title",
                long_from_long_field("w2"),
            )),
        ),
        (
            "docvs_full",
            Box::new(
                DocumentValueSourceDictionary::new(&r, "title", long_constant(5))
                    .with_payload("payload")
                    .with_contexts("ctx"),
            ),
        ),
    ];
    for (name, d) in cases {
        let want = dict_expected(&format!("{name}.tsv"));
        let got = dict_rows(d.as_ref());
        assert_dict(&got, &want, name);
        assert!(got.len() > 1 || name == "docdict_nofield", "{name}");
    }
}

#[test]
fn java_random_matches_java_util_random() {
    let text = fs::read_to_string(dir().join("random.tsv")).unwrap();
    let mut rows = 0;
    for line in text.lines() {
        let c: Vec<i64> = line.split('\t').map(|v| v.parse().unwrap()).collect();
        let mut r = JavaRandom::new(c[0]);
        let got: Vec<i64> = (0..6)
            .map(|_| i64::from(r.next_int(c[1] as i32).unwrap()))
            .collect();
        assert_eq!(got, c[2..], "seed {} bound {}", c[0], c[1]);
        rows += 1;
    }
    assert_eq!(rows, 50);
}

#[test]
fn unsorted_input_iterator_is_a_shuffle_of_its_source() {
    use lucene_search::suggest::input::UnsortedInputIterator;
    let rows: Vec<(&str, i64)> = (0..50)
        .map(|i| (["a", "b", "c", "d", "e"][i % 5], i as i64))
        .collect();
    let mut seen = Vec::new();
    let mut it = UnsortedInputIterator::with_seed(&mut terms_of(&rows), 7);
    while let Some(t) = it.next() {
        seen.push((String::from_utf8(t).unwrap(), it.weight()));
    }
    let mut sorted = seen.clone();
    sorted.sort_by_key(|(_, w)| *w);
    let want: Vec<(String, i64)> = rows.iter().map(|(t, w)| (t.to_string(), *w)).collect();
    assert_eq!(sorted, want, "every entry once, weight with its term");
    assert_ne!(seen, want, "in a different order");
    // the same seed gives the same order
    let mut again = UnsortedInputIterator::with_seed(&mut terms_of(&rows), 7);
    let mut second = Vec::new();
    while let Some(t) = again.next() {
        second.push((String::from_utf8(t).unwrap(), again.weight()));
    }
    assert_eq!(seen, second);
    assert!(!it.has_payloads() && !it.has_contexts());
    assert!(it.payload().is_none() && it.contexts().is_none());
}
