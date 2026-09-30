//! Differential test for `lucene_search::segment_order::SegmentOrder` against
//! `fixtures/data/segment_order/manifest.properties`
//! (`fixtures/src/GenSegmentOrder.java`): for every sort -- the skip-index
//! path and the points path, `LONG`/`INT`/`FLOAT`/`DOUBLE`, ascending and
//! descending, with and without a missing value, an absent field and a
//! non-numeric key -- the reordered reader's segments must come in Java's
//! order, with doc bases that follow it.
//!
//! Regenerate with `scripts/gen-fixtures.sh --only GenSegmentOrder`.

use std::collections::HashMap;

use lucene_search::directory_reader::DirectoryReader;
use lucene_search::segment_order::SegmentOrder;
use lucene_search::top_field::{SortField, SortType};
use lucene_store::FsDirectory;

fn fixtures() -> String {
    format!(
        "{}/../../fixtures/data/segment_order",
        env!("CARGO_MANIFEST_DIR")
    )
}

#[test]
fn segment_order_matches_lucene() {
    let want: HashMap<String, String> =
        std::fs::read_to_string(format!("{}/manifest.properties", fixtures()))
            .unwrap()
            .lines()
            .filter(|l| !l.starts_with('#'))
            .filter_map(|l| l.split_once('='))
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
    let dir = FsDirectory::open(format!("{}/index", fixtures()));
    let reader = DirectoryReader::open(&dir).unwrap();
    let count: usize = want["count"].parse().unwrap();
    assert!(count >= 30);
    for i in 0..count {
        let spec: Vec<&str> = want[&format!("sort.{i}")].split(';').collect();
        let ty = match spec[1] {
            "LONG" => SortType::Long,
            "INT" => SortType::Int,
            "FLOAT" => SortType::Float,
            "DOUBLE" => SortType::Double,
            _ => SortType::String,
        };
        let reverse = spec[2] == "true";
        let mut sf = SortField::numeric(spec[0], ty, reverse);
        let missing_set = spec[3] != "-";
        if missing_set {
            sf.missing = spec[3].parse().unwrap();
        }
        let reordered = SegmentOrder::from_sort(&sf, missing_set).reorder(&reader);
        let names: Vec<&str> = reordered
            .segment_readers()
            .iter()
            .map(|s| s.segment_name.as_str())
            .collect();
        assert_eq!(
            names.join(","),
            want[&format!("order.{i}")],
            "sort {i}: {spec:?}"
        );

        // The view is a reader in its own right: doc bases follow the new
        // order, and it holds the same documents.
        let mut base = 0;
        for s in reordered.segment_readers() {
            assert_eq!(s.doc_base, base, "sort {i}");
            base += s.max_doc;
        }
        assert_eq!(reordered.max_doc(), reader.max_doc());
        assert_eq!(reordered.num_docs(), reader.num_docs());
    }
}
