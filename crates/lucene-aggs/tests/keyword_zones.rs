#![allow(clippy::arithmetic_side_effects)]
//! Keyword `terms` aggregations over `Adaptive90` ordinals (blocks and
//! per-1024-document statistics) come out exactly as over a stock Lucene90
//! copy of the same documents: a clustered keyword (whole windows fold), one
//! with a run of documents lacking it, a multi-valued one, `missing`, a
//! second segment, and the merged segment; and the clustered keyword's
//! windows are the ones answered from the statistics.

use std::collections::HashMap;
use std::sync::Arc;

use lucene_aggs::bucket_aggs::{aggregate_tree, zone_windows_folded, AggNode, Globals, TermsOrder};
use lucene_aggs::metric_aggs::{Source, ValueKind, NEED_ALL, NEED_COUNT, NEED_SUM};
use lucene_codecs::field_infos::{DocValuesSkipIndexType, DocValuesType, FieldInfo};
use lucene_codecs::per_field_doc_values::Lucene90DocValuesFormat;
use lucene_codecs::stored_fields::{FieldValue, StoredField};
use lucene_index::index_writer::{ExplicitDocument, ExplicitFields, IndexWriter};
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::query::MatchAllDocsQuery;
use lucene_search::{BooleanQuery, Clause};
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;

const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};
const BYTES: i32 = 0;
const LEVEL: i32 = 1;
const GAPPY: i32 = 2;
const MULTI: i32 = 3;
const DOCS: usize = 12_000;

fn kw(field_number: i32, v: &str) -> StoredField {
    StoredField {
        field_number,
        value: FieldValue::Binary(v.as_bytes().to_vec()),
    }
}

fn doc(i: usize) -> ExplicitDocument {
    let mut dv = vec![
        StoredField {
            field_number: BYTES,
            value: FieldValue::Long(100 + (i as i64 * 7) % 500),
        },
        // Runs of 2500 documents of one level, so most windows hold one.
        kw(LEVEL, &format!("level-{}", (i / 2500 + i / 5000) % 5)),
        kw(MULTI, &format!("m{}", i % 3)),
    ];
    if !(2_000..5_000).contains(&i) {
        dv.push(kw(GAPPY, &format!("g{}", (i / 1500) % 3)));
    }
    if i.is_multiple_of(7) {
        dv.push(kw(MULTI, &format!("m{}", 3 + i % 2)));
    }
    ExplicitDocument {
        stored: Vec::new(),
        fields: ExplicitFields {
            doc_values: dv,
            ..ExplicitFields::default()
        },
    }
}

fn build(dir: &FsDirectory, format: Lucene90DocValuesFormat, merge: bool) {
    let mut w = IndexWriter::open(dir, Vec::new(), "Lucene104", VERSION).unwrap();
    w.enable_explicit_documents().unwrap();
    w.set_default_doc_values_format(format);
    let dv = |name: &str, number, t| {
        FieldInfo::new(name, number).with_doc_values(t, DocValuesSkipIndexType::None, -1)
    };
    w.register_field(dv("bytes", BYTES, DocValuesType::Numeric))
        .unwrap();
    for (n, f) in [("level", LEVEL), ("gappy", GAPPY), ("multi", MULTI)] {
        w.register_field(dv(n, f, DocValuesType::SortedSet))
            .unwrap();
    }
    w.add_explicit_documents((0..DOCS / 2).map(doc).collect())
        .unwrap();
    w.commit().unwrap();
    w.add_explicit_documents((DOCS / 2..DOCS).map(doc).collect())
        .unwrap();
    w.commit().unwrap();
    if merge {
        w.force_merge(1).unwrap();
        w.commit().unwrap();
    }
}

fn terms(field: &str, missing: Option<&str>, subs: Vec<AggNode>) -> AggNode {
    AggNode::Terms {
        field: field.into(),
        missing: missing.map(|m| m.as_bytes().to_vec()),
        accepted: None,
        shard_size: 100,
        order: TermsOrder::CountDesc,
        zero_buckets: false,
        subs,
    }
}

fn metric(needs: u8) -> AggNode {
    AggNode::Metric {
        field: "bytes".into(),
        missing: None,
        kind: ValueKind::Long,
        source: Source::DocValues,
        needs,
        script: None,
    }
}

/// Every tree over every keyword, run over `dir`; the results, rendered.
fn run(dir: &FsDirectory) -> Vec<String> {
    let reader = DirectoryReader::open(dir).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let mut ords = HashMap::new();
    for f in ["level", "gappy", "multi"] {
        ords.insert(
            f.to_string(),
            Arc::new(
                lucene_search::global_ords::GlobalOrds::build(reader.segment_readers(), f).unwrap(),
            ),
        );
    }
    let globals = Globals::new(&ords);
    let mut q = BooleanQuery::new();
    q.filter = vec![Clause::MatchAllDocs(MatchAllDocsQuery::new(i32::MAX))];
    let slices = vec![(0..segments.len()).collect::<Vec<_>>()];
    let trees = [
        terms("level", None, vec![]),
        terms("level", None, vec![metric(NEED_COUNT | NEED_SUM)]),
        terms("level", Some("none"), vec![metric(NEED_ALL)]),
        terms("gappy", None, vec![metric(NEED_ALL)]),
        terms("gappy", Some("none"), vec![metric(NEED_COUNT | NEED_SUM)]),
        terms("multi", None, vec![metric(NEED_ALL)]),
    ];
    trees
        .iter()
        .map(|t| {
            let r = aggregate_tree(
                &segments,
                reader.segment_readers(),
                &q,
                std::slice::from_ref(t),
                &globals,
                &slices,
                None,
            )
            .unwrap();
            format!("{r:?}")
        })
        .collect()
}

#[test]
fn keyword_terms_fold_whole_windows_exactly() {
    for merge in [false, true] {
        let stock_tmp = TempDir::new("kw-stock");
        let adaptive_tmp = TempDir::new("kw-adaptive");
        let stock = FsDirectory::open(stock_tmp.path());
        let adaptive = FsDirectory::open(adaptive_tmp.path());
        build(&stock, Lucene90DocValuesFormat::default(), merge);
        build(&adaptive, Lucene90DocValuesFormat::rust_adaptive(), merge);
        let before = zone_windows_folded();
        let expected = run(&stock);
        assert_eq!(zone_windows_folded(), before, "stock has no statistics");
        let got = run(&adaptive);
        assert_eq!(expected, got, "merge: {merge}");
        assert!(
            zone_windows_folded() > before + 5,
            "clustered keyword windows fold (merge: {merge})"
        );
    }
}
