//! M14 T14.3 `misc`, differentially against Lucene 10.5.0:
//! `fixtures/src/GenMiscLazy.java`'s `misc/lazy.tsv`.
//!
//! For each document of a two-segment index of stored fields, a
//! `LazyDocument` is handed fields in two groups and read in between; Lucene
//! recorded which lazy fields were loaded after each step and what every value
//! reads as (string, number, binary). The port must load the same fields at
//! the same steps and read the same values.
#![allow(clippy::arithmetic_side_effects)]

use std::collections::BTreeMap;

use lucene_index::document::Number;
use lucene_search::directory_reader::{DirectoryReader, SegmentReader};
use lucene_search::misc::lazy_document::{LazyDocument, LazyField};
use lucene_store::FsDirectory;

const GROUP_A: [&str; 3] = ["s", "i", "b"];
const GROUP_B: [&str; 4] = ["l", "f", "d", "other"];

/// `(field, index, loaded)` for every lazy field handed out, in order.
type Stage = Vec<(String, usize, bool)>;

fn data() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/misc")
}

fn hex(b: &[u8]) -> String {
    if b.is_empty() {
        return "+".into();
    }
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn num(n: Option<Number>) -> String {
    match n {
        None => "-".into(),
        Some(Number::Int(v)) => format!("I:{v}"),
        Some(Number::Long(v)) => format!("J:{v}"),
        Some(Number::Float(v)) => format!("F:{}", v.to_bits() as i32),
        Some(Number::Double(v)) => format!("D:{}", v.to_bits() as i64),
    }
}

/// Every lazy field handed out so far, per field name in the order handed out.
struct Handed<'a, 'r> {
    fields: Vec<(String, Vec<LazyField<'a, 'r>>)>,
}

impl Handed<'_, '_> {
    fn stage(&self) -> Stage {
        let mut out = Vec::new();
        for (name, list) in &self.fields {
            for (i, f) in list.iter().enumerate() {
                out.push((name.clone(), i, f.has_been_loaded()));
            }
        }
        out
    }

    fn read_first(&self, name: &str) {
        let list = &self.fields.iter().find(|(n, _)| n == name).unwrap().1;
        if let Some(f) = list.first() {
            f.string_value().unwrap();
        }
    }
}

type Values = BTreeMap<(i32, String, usize), (String, String, String)>;

/// Hands `lazy` one field per value `doc` has of each of `names`, as the Java
/// generator did.
fn hand<'a, 'r>(
    handed: &mut Handed<'a, 'r>,
    lazy: &'a LazyDocument<'r>,
    (segment, doc): (&SegmentReader, i32),
    values: &Values,
    names: &[&str],
) {
    for name in names {
        let n = values
            .keys()
            .filter(|(d, f, _)| *d == doc && f == name)
            .count();
        let info = segment.field_infos().field_by_name(name).unwrap();
        let list = (0..n).map(|_| lazy.get_field(info)).collect();
        handed.fields.push((name.to_string(), list));
    }
}

#[test]
fn lazy_fields_load_as_lucene_does() {
    let tsv = std::fs::read_to_string(data().join("lazy.tsv")).unwrap();
    let reader = DirectoryReader::open(&FsDirectory::open(data().join("lazy"))).unwrap();
    // (doc, stage) -> Java's loaded flags; (doc, field, idx) -> string, number, binary.
    let mut stages: BTreeMap<(i32, i32), Stage> = BTreeMap::new();
    let mut values: Values = BTreeMap::new();
    for line in tsv.lines() {
        let c: Vec<&str> = line.split('\t').collect();
        match c[0] {
            "L" => stages
                .entry((c[1].parse().unwrap(), c[2].parse().unwrap()))
                .or_default()
                .push((c[3].to_string(), c[4].parse().unwrap(), c[5] == "1")),
            "V" => {
                values.insert(
                    (
                        c[1].parse().unwrap(),
                        c[2].to_string(),
                        c[3].parse().unwrap(),
                    ),
                    (c[4].to_string(), c[5].to_string(), c[6].to_string()),
                );
            }
            other => panic!("unknown row {other}"),
        }
    }
    let max_doc: i32 = reader.segment_readers().iter().map(|s| s.max_doc).sum();
    assert_eq!(max_doc, 16);
    let mut loaded_siblings = 0;
    for doc in 0..max_doc {
        let mut base = 0;
        let mut segment = None;
        for s in reader.segment_readers() {
            if doc < base + s.max_doc {
                segment = Some(s);
                break;
            }
            base += s.max_doc;
        }
        let segment = segment.unwrap();
        let lazy = LazyDocument::new(&reader, doc);
        let mut handed = Handed { fields: Vec::new() };
        hand(&mut handed, &lazy, (segment, doc), &values, &GROUP_A);
        assert_eq!(handed.stage(), stages[&(doc, 1)], "doc {doc} stage 1");
        handed.read_first(GROUP_A[0]);
        assert_eq!(handed.stage(), stages[&(doc, 2)], "doc {doc} stage 2");
        hand(&mut handed, &lazy, (segment, doc), &values, &GROUP_B);
        assert_eq!(handed.stage(), stages[&(doc, 3)], "doc {doc} stage 3");
        handed.read_first(GROUP_B[0]);
        assert_eq!(handed.stage(), stages[&(doc, 4)], "doc {doc} stage 4");
        loaded_siblings += handed
            .stage()
            .iter()
            .filter(|(_, i, loaded)| *i > 0 && *loaded)
            .count();
        for (name, list) in &handed.fields {
            for (i, f) in list.iter().enumerate() {
                let got = (
                    f.string_value().unwrap().unwrap_or_else(|| "-".into()),
                    num(f.numeric_value().unwrap()),
                    f.binary_value().unwrap().map_or("-".into(), |b| hex(&b)),
                );
                assert_eq!(
                    &got,
                    &values[&(doc, name.clone(), i)],
                    "doc {doc} {name}[{i}]"
                );
            }
        }
        assert_eq!(handed.stage(), stages[&(doc, 5)], "doc {doc} stage 5");
    }
    assert!(
        loaded_siblings > 0,
        "a read loads the field's other values too"
    );
}

/// Java's assertion "field value was not lazy loaded": more lazy fields were
/// handed out for a name than the document has values.
#[test]
fn a_lazy_field_beyond_the_documents_values_is_refused() {
    let reader = DirectoryReader::open(&FsDirectory::open(data().join("lazy"))).unwrap();
    let segment = &reader.segment_readers()[0];
    let info = segment.field_infos().field_by_name("l").unwrap();
    let lazy = LazyDocument::new(&reader, 0);
    // No document has three values of `l` (GenMiscLazy stores at most two).
    let fields: Vec<_> = (0..3).map(|_| lazy.get_field(info)).collect();
    assert!(fields.iter().all(|f| !f.has_been_loaded()));
    let err = fields[2].string_value().unwrap_err();
    assert!(
        matches!(&err, lucene_search::Error::IllegalState(m) if m == "field value was not lazy loaded"),
        "{err:?}"
    );
    assert_eq!(fields[0].name(), "l");
}
