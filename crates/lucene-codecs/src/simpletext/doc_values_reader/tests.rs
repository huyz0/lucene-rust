#![cfg(test)]
use super::*;
use crate::doc_values::{BinaryColumn, SortedNumericColumn, SortedOrdsColumn, SortedSetOrdsColumn};
use crate::field_infos::{DocValuesSkipIndexType as Skip, FieldInfo};
use crate::simpletext::doc_values_writer::SimpleTextDocValuesWriter;

const MAX_DOC: i32 = 6;

fn fi(name: &str, n: i32, ty: DocValuesType, skip: Skip) -> FieldInfo {
    let mut f = FieldInfo::new(name, n);
    f.doc_values_type = ty;
    f.doc_values_skip_index_type = skip;
    f
}

/// One field of every type over six documents.
fn file() -> Vec<u8> {
    let mut w = SimpleTextDocValuesWriter::new(MAX_DOC);
    w.add_numeric_field(
        &fi("num", 0, DocValuesType::Numeric, Skip::Range),
        &[(0, 5), (2, -3), (5, i64::MAX)],
    )
    .unwrap();
    w.add_binary_field(
        &fi("bin", 1, DocValuesType::Binary, Skip::None),
        &BinaryColumn {
            docs: vec![1, 3],
            lengths: vec![3, 0],
            bytes: b"a\nb".to_vec(),
        },
    )
    .unwrap();
    w.add_sorted_field(
        &fi("sorted", 2, DocValuesType::Sorted, Skip::Range),
        &SortedOrdsColumn {
            dict: vec![b"x".to_vec(), b"y".to_vec()],
            docs: vec![0, 4],
            ords: vec![1, 0],
        },
    )
    .unwrap();
    w.add_sorted_numeric_field(
        &fi("snum", 3, DocValuesType::SortedNumeric, Skip::Range),
        &SortedNumericColumn {
            docs: vec![2],
            counts: vec![2],
            values: vec![-1, 7],
        },
    )
    .unwrap();
    w.add_sorted_set_field(
        &fi("sset", 4, DocValuesType::SortedSet, Skip::None),
        &SortedSetOrdsColumn {
            dict: vec![b"p".to_vec(), b"q".to_vec(), b"r".to_vec()],
            docs: vec![0, 5],
            counts: vec![2, 1],
            ords: vec![0, 2, 1],
        },
    )
    .unwrap();
    w.add_numeric_field(&fi("none", 5, DocValuesType::Numeric, Skip::None), &[])
        .unwrap();
    w.finish().unwrap()
}

fn text(data: &[u8]) -> String {
    String::from_utf8(data.to_vec()).unwrap()
}

/// Lucene 10.6's `isSingleValued` (#16714), off the recorded
/// `maxValueCount`.
#[test]
fn single_valued_fields_follow_max_value_count() {
    let data = file();
    let r = SimpleTextDocValuesReader::open(&data, MAX_DOC).unwrap();
    for (name, single) in [
        ("num", true),
        ("sorted", true),
        ("snum", false),
        ("sset", false),
        ("none", true),
    ] {
        assert_eq!(r.field(name).unwrap().is_single_valued(), single, "{name}");
    }
    let mut unknown = OneField::new(DocValuesType::SortedNumeric);
    unknown.max_value_count = -1;
    assert!(!unknown.is_single_valued());
}

#[test]
fn cached_headers_borrow_live_bytes_for_each_value_type() {
    let data = file();
    let metadata = SimpleTextDocValuesReader::open(&data, MAX_DOC)
        .unwrap()
        .rebind(&[]);
    let mut numeric = metadata.numeric_from_data("num", &data).unwrap();
    assert!(numeric.advance_exact(2).unwrap());
    assert_eq!(numeric.long_value().unwrap(), -3);
    let mut binary = metadata.binary_from_data("bin", &data).unwrap();
    assert!(binary.advance_exact(1).unwrap());
    assert_eq!(binary.binary_value().unwrap(), b"a\nb");
    let mut sorted = metadata.sorted_from_data("sorted", &data).unwrap();
    assert!(sorted.advance_exact(4).unwrap());
    assert_eq!(sorted.lookup_ord(sorted.ord_value()).unwrap(), b"x");
    let mut numeric = metadata.sorted_numeric_from_data("snum", &data).unwrap();
    assert!(numeric.advance_exact(2).unwrap());
    assert_eq!(numeric.doc_value_count(), 2);
    assert_eq!(numeric.next_value().unwrap(), -1);
    assert_eq!(numeric.next_value().unwrap(), 7);
    let mut set = metadata.sorted_set_from_data("sset", &data).unwrap();
    assert!(set.advance_exact(0).unwrap());
    assert_eq!(set.doc_value_count(), 2);
    let ord = set.next_ord().unwrap();
    assert_eq!(set.lookup_ord(ord).unwrap(), b"p");
    let ord = set.next_ord().unwrap();
    assert_eq!(set.lookup_ord(ord).unwrap(), b"r");
    assert!(metadata.numeric("num").unwrap().next_doc().is_err());
    assert!(metadata.check_integrity().is_err());
}

#[test]
fn numeric_iterates_and_advances() {
    let data = file();
    let r = SimpleTextDocValuesReader::open(&data, MAX_DOC).unwrap();
    assert_eq!((r.max_doc(), r.num_fields()), (MAX_DOC, 6));
    r.check_integrity().unwrap();
    let mut it = r.numeric("num").unwrap();
    assert!(it.cost() >= 0);
    let mut got = Vec::new();
    while it.next_doc().unwrap() != NO_MORE_DOCS {
        got.push((it.doc_id(), it.long_value().unwrap()));
    }
    assert_eq!(got, [(0, 5), (2, -3), (5, i64::MAX)]);
    let mut it = r.numeric("num").unwrap();
    assert_eq!(it.advance(3).unwrap(), 5);
    let mut it = r.numeric("num").unwrap();
    assert!(!it.advance_exact(1).unwrap());
    assert!(it.advance_exact(2).unwrap());
    assert_eq!(it.long_value().unwrap(), -3);
    let mut none = r.numeric("none").unwrap();
    assert_eq!(none.next_doc().unwrap(), NO_MORE_DOCS);
    assert!(r.numeric("missing").is_err());
    assert!(r.field("num").is_some() && r.field("missing").is_none());
}

#[test]
fn binary_and_sorted_numeric_values() {
    let data = file();
    let r = SimpleTextDocValuesReader::open(&data, MAX_DOC).unwrap();
    let mut it = r.binary("bin").unwrap();
    assert!(it.cost() >= 0);
    assert_eq!(it.next_doc().unwrap(), 1);
    assert_eq!(it.binary_value().unwrap(), b"a\nb");
    assert_eq!(it.next_doc().unwrap(), 3);
    assert_eq!(it.binary_value().unwrap(), b"");
    assert_eq!(it.next_doc().unwrap(), NO_MORE_DOCS);
    let mut it = r.binary("bin").unwrap();
    assert_eq!(it.advance(2).unwrap(), 3);
    assert_eq!(it.doc_id(), 3);
    let mut it = r.binary("bin").unwrap();
    assert!(!it.advance_exact(2).unwrap());
    assert!(it.advance_exact(3).unwrap());
    assert!(r.binary("missing").is_err());

    let mut s = r.sorted_numeric("snum").unwrap();
    assert!(s.cost() >= 0);
    assert_eq!(s.next_doc().unwrap(), 2);
    assert_eq!(s.doc_value_count(), 2);
    assert_eq!((s.next_value().unwrap(), s.next_value().unwrap()), (-1, 7));
    assert_eq!(s.next_doc().unwrap(), NO_MORE_DOCS);
    let mut s = r.sorted_numeric("snum").unwrap();
    assert_eq!(s.advance(1).unwrap(), 2);
    assert_eq!(s.doc_id(), 2);
    let mut s = r.sorted_numeric("snum").unwrap();
    assert!(!s.advance_exact(1).unwrap());
    assert!(s.advance_exact(2).unwrap());
    assert_eq!(s.next_value().unwrap(), -1);
    assert!(r.sorted_numeric("missing").is_err());
}

#[test]
fn sorted_ords_dictionary_and_terms_enum() {
    let data = file();
    let r = SimpleTextDocValuesReader::open(&data, MAX_DOC).unwrap();
    let mut s = r.sorted("sorted").unwrap();
    assert!(s.cost() >= 0);
    assert_eq!(s.value_count(), 2);
    assert_eq!(s.next_doc().unwrap(), 0);
    assert_eq!(s.ord_value(), 1);
    assert_eq!(s.lookup_ord(1).unwrap(), b"y");
    assert_eq!(s.next_doc().unwrap(), 4);
    assert_eq!((s.doc_id(), s.ord_value()), (4, 0));
    assert_eq!(s.next_doc().unwrap(), NO_MORE_DOCS);
    assert!(s.lookup_ord(-1).is_err() && s.lookup_ord(2).is_err());
    let mut s = r.sorted("sorted").unwrap();
    assert_eq!(s.advance(1).unwrap(), 4);
    let mut s = r.sorted("sorted").unwrap();
    assert!(!s.advance_exact(3).unwrap());
    assert!(s.advance_exact(4).unwrap());
    assert_eq!(s.lookup_term(b"y").unwrap(), Ok(1));
    assert_eq!(s.lookup_term(b"a").unwrap(), Err(0));
    assert_eq!(s.lookup_term(b"xx").unwrap(), Err(1));
    assert_eq!(s.lookup_term(b"z").unwrap(), Err(2));
    let mut te = s.terms_enum();
    assert_eq!(te.next().unwrap(), Some(&b"x"[..]));
    assert_eq!(te.next().unwrap(), Some(&b"y"[..]));
    assert_eq!(te.next().unwrap(), None);
    let mut te = s.terms_enum();
    assert_eq!(te.seek_ceil(b"a").unwrap(), SeekStatus::NotFound);
    assert_eq!((te.ord(), te.term()), (0, &b"x"[..]));
    assert_eq!(te.seek_ceil(b"y").unwrap(), SeekStatus::Found);
    assert_eq!(te.ord(), 1);
    assert_eq!(te.seek_ceil(b"z").unwrap(), SeekStatus::End);
    assert!(te.seek_exact(b"x").unwrap());
    assert!(!te.seek_exact(b"w").unwrap());
    te.seek_exact_ord(1).unwrap();
    assert_eq!(te.term(), b"y");
    assert!(te.seek_exact_ord(2).is_err());
    assert!(r.sorted("missing").is_err());
}

#[test]
fn sorted_set_ords() {
    let data = file();
    let r = SimpleTextDocValuesReader::open(&data, MAX_DOC).unwrap();
    let mut s = r.sorted_set("sset").unwrap();
    assert!(s.cost() >= 0);
    assert_eq!(s.value_count(), 3);
    assert_eq!(s.next_doc().unwrap(), 0);
    assert_eq!(s.doc_value_count(), 2);
    assert_eq!((s.next_ord().unwrap(), s.next_ord().unwrap()), (0, 2));
    assert_eq!(s.next_doc().unwrap(), 5);
    assert_eq!((s.doc_id(), s.next_ord().unwrap()), (5, 1));
    assert_eq!(s.next_doc().unwrap(), NO_MORE_DOCS);
    assert_eq!(s.lookup_ord(2).unwrap(), b"r");
    assert!(s.lookup_ord(3).is_err());
    assert_eq!(s.lookup_term(b"q").unwrap(), Ok(1));
    assert_eq!(s.lookup_term(b"qq").unwrap(), Err(2));
    let mut te = s.terms_enum();
    assert_eq!(te.seek_ceil(b"qq").unwrap(), SeekStatus::NotFound);
    assert_eq!(te.term(), b"r");
    let mut s = r.sorted_set("sset").unwrap();
    assert_eq!(s.advance(1).unwrap(), 5);
    let mut s = r.sorted_set("sset").unwrap();
    assert!(!s.advance_exact(1).unwrap());
    assert!(s.advance_exact(5).unwrap());
    assert!(r.sorted_set("missing").is_err());
}

#[test]
fn skippers_walk_their_intervals() {
    let data = file();
    let r = SimpleTextDocValuesReader::open(&data, MAX_DOC).unwrap();
    let mut s = r.skipper("num", DocValuesType::Numeric).unwrap();
    assert_eq!(s.num_levels(), 1);
    assert_eq!((s.min_doc_id(0), s.max_doc_id(0)), (-1, -1));
    assert_eq!((s.min_value(), s.max_value()), (-3, i64::MAX));
    assert_eq!((s.min_value_at(0), s.max_value_at(0)), (-3, i64::MAX));
    assert_eq!(
        (s.doc_count(), s.doc_count_at(0), s.max_value_count()),
        (3, 3, 1)
    );
    s.advance(0);
    assert_eq!((s.min_doc_id(0), s.max_doc_id(0)), (0, 5));
    assert_eq!(s.doc_count_at(0), 3);
    s.advance(6);
    assert_eq!(
        (s.min_doc_id(0), s.max_doc_id(0)),
        (NO_MORE_DOCS, NO_MORE_DOCS)
    );
    assert_eq!(s.min_value_at(0), -3);
    let s = r.skipper("sorted", DocValuesType::Sorted).unwrap();
    assert_eq!((s.min_value(), s.max_value()), (0, 1));
    assert!(r.skipper("missing", DocValuesType::Numeric).is_err());
}

#[test]
fn malformed_headers_are_corrupt() {
    let data = file();
    let t = text(&data);
    let open = |s: &str| SimpleTextDocValuesReader::open(s.as_bytes(), MAX_DOC).map(|_| ());
    assert!(SimpleTextDocValuesReader::open(b"", MAX_DOC).is_err());
    assert!(open("feld num\n").is_err());
    assert!(open(&t[t.find("END").unwrap()..]).is_err(), "no field");
    assert!(open(&t.replacen("  type NUMERIC", "  type NONE", 1)).is_err());
    assert!(open(&t.replacen("  type NUMERIC", "  type WHAT", 1)).is_err());
    assert!(open(&t.replacen("  minalue ", "  minalue x", 1)).is_err());
    assert!(open(&t.replacen("  origin ", "  orign ", 1)).is_err());
    assert!(open(&t.replacen("  maxlength ", "  maxlength x", 1)).is_err());
    assert!(open(&t.replacen("  numvalues ", "  numvalue ", 1)).is_err());
    assert!(open(&t.replacen("  skipdata", "  skipdat", 1)).is_err());
    assert!(open(&t.replacen("    interval", "    intervl", 1)).is_err());
    assert!(open(&t[..t.len() / 2]).is_err(), "truncated");
    assert!(
        SimpleTextDocValuesReader::open(&data, 1_000_000).is_err(),
        "past the end"
    );
    assert!(SimpleTextDocValuesReader::open(&data, -1).is_err());
    // A width or count that seeks backwards: `bin`'s records would
    // rewind onto `num`'s skip data, and `open` would never reach END.
    let reseal = |s: String| {
        let mut b = s.as_bytes()[..s.rfind("checksum ").unwrap()].to_vec();
        util::write_checksum(&mut b);
        b
    };
    for (from, to) in [
        ("  maxlength 9\n", "  maxlength -99\n"),
        ("  maxlength 9\n", "  maxlength -2\n"),
        ("  numvalues 2\n", "  numvalues -1\n"),
    ] {
        assert!(t.contains(from), "{from}");
        let bad = reseal(t.replacen(from, to, 1));
        assert!(
            SimpleTextDocValuesReader::open(&bad, MAX_DOC).is_err(),
            "{to}"
        );
    }
    assert!(seek(&data, 10, -1, 1).is_err() && seek(&data, 10, i64::MAX, 2).is_err());
    assert_eq!(seek(&data, 10, 2, 3).unwrap(), 16);
    // Without `maxvaluecount` lines: 0 for no documents, 1 for
    // NUMERIC/SORTED, -1 otherwise.
    let stripped: String = t
        .split_inclusive('\n')
        .filter(|l| !l.starts_with("  maxvaluecount "))
        .collect();
    let r = SimpleTextDocValuesReader::open(stripped.as_bytes(), MAX_DOC).unwrap();
    let count = |f: &str| r.field(f).unwrap().max_value_count;
    assert_eq!(
        [
            count("num"),
            count("sorted"),
            count("bin"),
            count("sset"),
            count("none")
        ],
        [1, 1, -1, -1, 0]
    );
}

#[test]
fn malformed_records_are_corrupt() {
    let t = text(&file());
    fn r(s: &str) -> SimpleTextDocValuesReader<'_> {
        SimpleTextDocValuesReader::open(s.as_bytes(), MAX_DOC).unwrap()
    }
    // num's first record follows its pattern line.
    let at = t.find("  pattern ").unwrap();
    let rec = t[at..].find('\n').unwrap() + at + 1;
    let mut bad = t.clone();
    bad.replace_range(rec..=rec, "x");
    let rd = r(&bad);
    let mut it = rd.numeric("num").unwrap();
    assert!(it.next_doc().and_then(|_| it.long_value()).is_err());
    // bin's first length line.
    let bad = t.replacen("\nlength 9", "\nlength x", 1);
    assert_ne!(bad, t);
    let rd = r(&bad);
    let mut it = rd.binary("bin").unwrap();
    assert!(it
        .next_doc()
        .and_then(|_| it.binary_value().map(|_| 0))
        .is_err());
    // An ord list that is not longs.
    let bad = t.replacen("0,2", "0,z", 1);
    let rd = r(&bad);
    let mut it = rd.sorted_set("sset").unwrap();
    assert!(it
        .next_doc()
        .and_then(|_| it.next_ord())
        .and_then(|_| it.next_ord())
        .is_err());
    assert_eq!(java_split_comma("1,,2,,"), ["1", "", "2"]);
    // check_integrity: one changed byte breaks the checksum.
    let flipped = t.replacen("doccount 3", "doccount 4", 1);
    let rd = r(&flipped);
    assert!(rd.check_integrity().is_err());
}
