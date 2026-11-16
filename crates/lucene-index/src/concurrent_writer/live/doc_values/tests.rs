#![cfg(test)]
use lucene_codecs::field_infos::{DocValuesSkipIndexType, FieldInfo};
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;

use super::*;
use crate::concurrent_writer::ConcurrentIndexWriter;
use crate::index_writer::IndexWriter;
use crate::segment_info::LuceneVersion;

fn value(field_number: i32, value: FieldValue) -> StoredField {
    StoredField {
        field_number,
        value,
    }
}

/// Doc values written from columns are byte for byte the ones written
/// from the documents, for every type, dense or sparse, and over any
/// run of the buffer.
#[test]
fn columns_write_the_documents_doc_values() {
    let tmp = TempDir::new("live-dv-columns");
    let dir = FsDirectory::open(tmp.path());
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", version).unwrap();
    w.enable_explicit_documents().unwrap();
    let mut field = |name: &str, kind: DocValuesType, skip: bool| {
        let skip = if skip {
            DocValuesSkipIndexType::Range
        } else {
            DocValuesSkipIndexType::None
        };
        w.register_field(FieldInfo::new(name, 0).with_doc_values(kind, skip, -1))
            .unwrap()
    };
    let num = field("num", DocValuesType::Numeric, true);
    let sparse = field("sparse", DocValuesType::Numeric, false);
    let multi = field("multi", DocValuesType::SortedNumeric, false);
    let bin = field("bin", DocValuesType::Binary, false);
    let kw = field("kw", DocValuesType::Sorted, true);
    let tags = field("tags", DocValuesType::SortedSet, false);
    let w = ConcurrentIndexWriter::new(w, 1).unwrap();
    let cfg = w.cfg();

    let words = ["delta", "alpha", "echo", "bravo", "charlie"];
    let docs: Vec<Vec<StoredField>> = (0..300i64)
        .map(|i| {
            let mut d = vec![value(num, FieldValue::Long(i * 7 - 1000))];
            if i % 3 == 0 {
                d.push(value(sparse, FieldValue::Long(i)));
            }
            for k in 0..(i % 4) {
                d.push(value(multi, FieldValue::Long((i * 31 + k * 17) % 50 - 20)));
            }
            if i % 5 != 1 {
                d.push(value(bin, FieldValue::Binary(format!("b{i}").into_bytes())));
            }
            d.push(value(
                kw,
                FieldValue::String(words[(i % 5) as usize].to_string()),
            ));
            for k in 0..(i % 3) {
                let w = words[((i + k * 2) % 5) as usize];
                d.push(value(tags, FieldValue::Binary(w.as_bytes().to_vec())));
                d.push(value(tags, FieldValue::Binary(w.as_bytes().to_vec())));
            }
            d
        })
        .collect();
    let mut columns = DocValuesColumns::default();
    for (doc, values) in docs.iter().enumerate() {
        columns.record(i32::try_from(doc).unwrap(), values, |n| {
            cfg.field_by_number(n).map(|f| f.doc_values_type)
        });
    }
    let mut compared = 0;
    for (start, end) in [(0usize, 300usize), (0, 128), (37, 211), (299, 300), (5, 6)] {
        let run: Vec<&[StoredField]> = docs[start..end].iter().map(Vec::as_slice).collect();
        let prebuilt = columns
            .range(i32::try_from(start).unwrap(), i32::try_from(end).unwrap())
            .unwrap();
        // A field none of the run's documents has keeps its empty column.
        let present: Vec<i32> = prebuilt.iter().map(|(n, _)| *n).collect();
        let all = [num, sparse, multi, bin, kw, tags];
        assert!(
            all.iter().all(|n| present.contains(n)),
            "documents {start}..{end}"
        );
        let (decoded, from_columns) = cfg.doc_values_both_ways(&run, &prebuilt);
        assert_eq!(decoded, from_columns, "documents {start}..{end}");
        // Copied out first and built later, the same files.
        let built: Vec<(i32, PrebuiltDocValues)> = columns
            .runs(i32::try_from(start).unwrap(), i32::try_from(end).unwrap())
            .unwrap()
            .iter()
            .map(|(n, r)| (*n, r.build().unwrap()))
            .collect();
        let (_, from_runs) = cfg.doc_values_both_ways(&run, &built);
        assert_eq!(decoded, from_runs, "documents {start}..{end}, built later");
        compared += 1;
    }
    assert_eq!(compared, 5);
}

/// Terms past a block's end are found by id, in a copy taken before
/// and after more arrive.
#[test]
fn a_term_list_finds_terms_across_blocks() {
    let mut terms = Terms::default();
    let n = TERM_BLOCK * 2 + 5;
    for k in 0..n {
        assert_eq!(
            terms.id(format!("t{k}").as_bytes()),
            Some(u32::try_from(k).unwrap())
        );
    }
    assert_eq!(terms.id(b"t3"), Some(3));
    let copy = terms.list.clone();
    terms.id(b"later");
    assert_eq!(copy.len(), n);
    for k in [0, TERM_BLOCK - 1, TERM_BLOCK, 2 * TERM_BLOCK, n - 1] {
        let id = u32::try_from(k).unwrap();
        assert_eq!(copy.get(id), format!("t{k}").as_bytes());
        assert_eq!(terms.list.get(id), format!("t{k}").as_bytes());
    }
    assert_eq!(terms.list.get(u32::try_from(n).unwrap()), b"later");
}

#[test]
fn a_value_of_the_wrong_kind_refuses_the_columns() {
    let mut columns = DocValuesColumns::default();
    let kind = |_| Some(DocValuesType::Numeric);
    columns.record(0, &[value(1, FieldValue::Long(3))], kind);
    assert!(columns.range(0, 1).is_some());
    columns.record(1, &[value(1, FieldValue::Binary(vec![1]))], kind);
    assert!(columns.range(0, 2).is_none());
    let mut columns = DocValuesColumns::default();
    columns.record(
        0,
        &[value(1, FieldValue::Long(3)), value(1, FieldValue::Long(4))],
        kind,
    );
    assert!(
        matches!(
            columns.range(0, 1).as_deref(),
            Some([(1, PrebuiltDocValues::Numeric(v))]) if v == &[3]
        ),
        "a NUMERIC field keeps a document's first value"
    );
}
