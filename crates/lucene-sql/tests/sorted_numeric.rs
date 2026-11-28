//! A long column over a field written as sorted numeric, which is how
//! OpenSearch writes every number and date: one value per document reads as
//! the plain numeric column does; several values are refused by name.

mod common;

use std::sync::Arc;

use common::{rows, sorted};
use lucene_index::document::{Document, KeywordField, SortedNumericDocValuesField, Store};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_sql::{Snapshot, SqlConfig, SqlEngine, TableSpec};
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;

fn engine(n: usize, multi_at: Option<usize>) -> (TempDir, SqlEngine) {
    let dir = TempDir::new("sql-sn");
    {
        let fs = FsDirectory::open(dir.path_str());
        let version = LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        };
        let mut w = IndexWriter::open(&fs, Vec::new(), "Lucene104", version).unwrap();
        for i in 0..n {
            let mut doc = Document::new();
            doc.add(KeywordField::new("cat", format!("c{}", i % 4), Store::No));
            // Every fifth document has no value: the field is sparse.
            if i % 5 != 0 {
                doc.add(SortedNumericDocValuesField::new("price", i as i64 * 3));
            }
            if multi_at == Some(i) {
                doc.add(SortedNumericDocValuesField::new("price", 7));
            }
            w.add_fields_document(&doc).unwrap();
            if (i + 1) % (n / 2) == 0 {
                w.commit().unwrap();
            }
        }
        w.commit().unwrap();
    }
    let reader = DirectoryReader::open(&FsDirectory::open(dir.path_str())).unwrap();
    let spec = TableSpec::new("t").keyword("cat").long("price");
    let snapshot = Arc::new(Snapshot::new(reader, spec).unwrap());
    let config = SqlConfig {
        worker_threads: 0,
        ..SqlConfig::default()
    };
    (dir, SqlEngine::new(snapshot, &config).unwrap())
}

fn expected(n: usize) -> Vec<Option<i64>> {
    (0..n)
        .map(|i| (i % 5 != 0).then_some(i as i64 * 3))
        .collect()
}

#[test]
fn one_value_per_document_reads_like_a_numeric_field() {
    let (_dir, e) = engine(100, None);
    let r = rows(
        &e.query_blocking("SELECT price FROM t WHERE cat = 'c1' ORDER BY price LIMIT 3")
            .unwrap(),
    );
    assert_eq!(r, vec![vec!["3"], vec!["27"], vec!["39"]]);

    let values = expected(100);
    let missing = values.iter().filter(|v| v.is_none()).count();
    let r = rows(
        &e.query_blocking("SELECT count(*) FROM t WHERE price IS NULL")
            .unwrap(),
    );
    assert_eq!(r, vec![vec![missing.to_string()]]);
    let big = values.iter().flatten().filter(|&&v| v > 100).count();
    let r = rows(
        &e.query_blocking("SELECT count(*) FROM t WHERE price > 100")
            .unwrap(),
    );
    assert_eq!(r, vec![vec![big.to_string()]]);
}

#[test]
fn aggregates_over_it_match_a_scan() {
    let (_dir, e) = engine(100, None);
    let values = expected(100);
    let mut want = Vec::new();
    for c in 0..4 {
        let group: Vec<i64> = (0..100)
            .filter(|i| i % 4 == c)
            .filter_map(|i| values[i])
            .collect();
        want.push(vec![
            format!("c{c}"),
            group.iter().sum::<i64>().to_string(),
            group.len().to_string(),
            group.iter().min().unwrap().to_string(),
            group.iter().max().unwrap().to_string(),
        ]);
    }
    let sql = "SELECT cat, sum(price), count(price), min(price), max(price) FROM t GROUP BY cat";
    assert_eq!(sorted(rows(&e.query_blocking(sql).unwrap())), want);
    assert_eq!(sorted(rows(&e.standard_blocking(sql).unwrap())), want);
}

#[test]
fn several_values_in_a_document_are_refused_by_column_name() {
    let (_dir, e) = engine(100, Some(8));
    let err = e.query_blocking("SELECT price FROM t").unwrap_err();
    assert!(err.to_string().contains("price"), "{err}");
}
