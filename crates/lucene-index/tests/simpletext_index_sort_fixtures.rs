//! Typed SimpleText segment infos against all four real Lucene sort providers.

use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

use lucene_index::segment_info::{self, IndexSortKind, NumericSortKey};

fn unhex(s: &str) -> Vec<u8> {
    s.as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

#[test]
fn every_simpletext_sort_provider_decodes_and_rebuilds_javas_segment_info() {
    let root =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/simpletext_index_sort");
    let manifest = fs::read_to_string(root.join("cases.tsv")).unwrap();
    let mut kinds = BTreeSet::new();
    assert_eq!(manifest.lines().count(), 184);
    for (ordinal, row) in manifest.lines().enumerate() {
        let row: Vec<_> = row.split('\t').collect();
        let id = unhex(row[1]).try_into().unwrap();
        let bytes = fs::read(root.join(format!("{}.si", row[0]))).unwrap();
        let mut info = segment_info::parse_simpletext(&bytes, row[0], &id).unwrap();
        assert_eq!(
            segment_info::describe_index_sort(
                (!info.index_sort.is_empty()).then_some(info.index_sort.as_slice())
            ),
            String::from_utf8(unhex(row[2])).unwrap(),
            "{}",
            row[0]
        );
        assert_eq!(info.doc_count, 7);
        assert_eq!(info.version.to_string(), row[3]);
        assert_eq!(
            info.min_version
                .map(|v| v.to_string())
                .unwrap_or_else(|| "null".into()),
            row[4]
        );
        assert_eq!(info.min_version.is_some(), ordinal % 2 == 0);
        assert_eq!(info.is_compound_file, ordinal % 2 == 0);
        assert_eq!(info.has_blocks, ordinal % 3 == 0);
        assert_eq!(info.diagnostics, [("source".into(), "fixture".into())]);
        assert_eq!(info.attributes, [("note".into(), "line\nbreak\\".into())]);
        for field in &info.index_sort {
            kinds.insert(match &field.kind {
                IndexSortKind::Numeric(NumericSortKey::Int(_)) => "int",
                IndexSortKind::Numeric(NumericSortKey::Long(_)) => "long",
                IndexSortKind::Numeric(NumericSortKey::Float(_)) => "float",
                IndexSortKind::Numeric(NumericSortKey::Double(_)) => "double",
                IndexSortKind::String(_) => "string",
                IndexSortKind::SortedNumeric { .. } => "sortednumeric",
                IndexSortKind::SortedSet { .. } => "sortedset",
                IndexSortKind::Binary(_) => "binary",
            });
        }
        assert_eq!(
            segment_info::write_simpletext(&mut info).unwrap(),
            bytes,
            "{}",
            row[0]
        );
    }
    assert_eq!(kinds.len(), 8);
}
