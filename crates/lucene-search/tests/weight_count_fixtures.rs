//! **Cross-engine ground truth for `Weight.count(LeafReaderContext)`** --
//! `IndexSearcher.count(query)`'s answers, recorded from real Lucene 10.5.0 by
//! `fixtures/src/AppendCountManifest.java` and re-derived here through this
//! port's own shortcuts.
//!
//! Why it needs five fixture indexes: the shortcut has five distinct arms and
//! no single committed index reaches more than two of them.
//!
//! | index | what only it can show |
//! |---|---|
//! | `blocktree_index` | no deletions, so `TermQuery` answers from `docFreq` |
//! | `live_docs_index` | deletions, so `docFreq` is *not* the answer |
//! | `norms_index` | a normed field on every document -- `rewrite` to `*:*` |
//! | `doc_values_index` | doc values with no doc count to shortcut from |
//! | `doc_values_skip_index` | the `DocValuesSkipper.docCount()` arm |
//!
//! The `live_docs_index` line is the one that matters most: `id:1` names a
//! **deleted** document whose `docFreq` is still 1, so a port that takes the
//! shortcut unconditionally reports 1 where Lucene reports 0. Nothing in this
//! tree could have caught that before, because every other scoring fixture is
//! deletion-free.

use lucene_search::directory_reader::DirectoryReader;
use lucene_search::query::TermQuery;
use lucene_search::weight_count::{
    count_field_exists_leaf, count_match_all_docs, count_term_query, count_term_query_shortcut,
    field_exists_rewrites_to_match_all_docs,
};
use lucene_store::FsDirectory;

fn fixture(name: &str) -> String {
    format!("{}/../../fixtures/data/{name}", env!("CARGO_MANIFEST_DIR"))
}

struct Manifest(Vec<(String, String)>);

impl Manifest {
    fn load(index: &str) -> Self {
        let text = std::fs::read_to_string(format!("{}/manifest.properties", fixture(index)))
            .expect("run scripts/gen-fixtures.sh first");
        Manifest(
            text.lines()
                .filter_map(|l| l.split_once('='))
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        )
    }

    fn get(&self, key: &str) -> &str {
        self.0
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
            .unwrap_or_else(|| panic!("manifest key {key} missing -- re-run AppendCountManifest"))
    }

    fn num(&self, key: &str) -> i64 {
        self.get(key).parse().unwrap()
    }
}

/// One-segment fixtures, all of them: the `DirectoryReader` is opened and its
/// single leaf handed back with the postings input already resolved.
fn open_leaf<R>(
    index: &str,
    body: impl FnOnce(
        &lucene_search::directory_reader::SegmentReader,
        &lucene_search::multi_segment::OpenSegment<'_>,
    ) -> R,
) -> R {
    let reader = DirectoryReader::open(&FsDirectory::open(fixture(index))).expect("open reader");
    assert_eq!(
        reader.segment_readers().len(),
        1,
        "{index} is expected to be a single-segment fixture"
    );
    let opened = reader.open_segments().expect("open segments");
    let segments = opened.as_open_segments();
    body(&reader.segment_readers()[0], &segments[0])
}

#[test]
fn term_query_counts_match_real_lucene_without_deletions() {
    let m = Manifest::load("blocktree_index");
    open_leaf("blocktree_index", |seg, open| {
        assert_eq!(seg.num_docs() as i64, m.num("count.numDocs"));
        assert_eq!(seg.max_doc as i64, m.num("count.maxDoc"));

        for (key, field, term) in [
            ("count.term.body.cat", "body", "cat"),
            ("count.term.big.everywhere", "big", "everywhere"),
            ("count.term.absent", "body", "zzz-no-such-term"),
            ("count.term.absentfield", "no-such-field", "cat"),
        ] {
            let query = TermQuery::new(field, term);
            // The shortcut fires -- `.doc` is never touched, which is what
            // makes this cheaper than collecting.
            let shortcut = count_term_query_shortcut(open.fields, None, &query);
            assert_eq!(
                shortcut,
                Some(m.num(key)),
                "{key}: the docFreq shortcut must be real Lucene's own count"
            );
            // And the full entry point agrees, with and without `.doc`.
            assert_eq!(
                count_term_query(open.fields, None, None, &query).unwrap(),
                m.num(key)
            );
            assert_eq!(
                count_term_query(open.fields, open.doc_in, None, &query).unwrap(),
                m.num(key)
            );
        }

        assert_eq!(
            count_match_all_docs(seg.max_doc, seg.live_docs()),
            m.num("count.matchall")
        );
    });
}

/// The deletions half. `id:1` has `docFreq == 1` and is deleted, so the
/// shortcut must decline and the scan must answer 0 -- a port that shortcuts
/// unconditionally reports a deleted document as a hit.
#[test]
fn term_query_counts_match_real_lucene_with_deletions() {
    let m = Manifest::load("live_docs_index");
    open_leaf("live_docs_index", |seg, open| {
        assert_eq!(seg.num_docs() as i64, m.num("count.numDocs"));
        assert_ne!(
            seg.num_docs(),
            seg.max_doc,
            "this fixture is only useful because it has deletions"
        );

        for (key, term) in [("count.term.id.1", "1"), ("count.term.id.0", "0")] {
            let query = TermQuery::new("id", term);
            assert_eq!(
                count_term_query_shortcut(open.fields, seg.live_docs(), &query),
                None,
                "{key}: `hasDeletions()` forbids the docFreq shortcut"
            );
            assert_eq!(
                count_term_query(open.fields, open.doc_in, seg.live_docs(), &query).unwrap(),
                m.num(key),
                "{key}: the scan must agree with real Lucene"
            );
            // And the shortcut, taken anyway, would have been wrong for the
            // deleted document -- which is the whole reason Java gates it.
            if term == "1" {
                assert_eq!(
                    count_term_query_shortcut(open.fields, None, &query),
                    Some(1),
                    "docFreq counts the deleted document"
                );
            }
        }

        assert_eq!(
            count_match_all_docs(seg.max_doc, seg.live_docs()),
            m.num("count.matchall")
        );
    });
}

/// `FieldExistsQuery`'s `count` and `rewrite`, over the three sources and both
/// the complete and partial cases, against real Lucene's own answers.
#[test]
fn field_exists_counts_and_rewrites_match_real_lucene() {
    // (index, field, whether the port can answer without a scan)
    let cases: [(&str, &str); 6] = [
        // Norms on 4 of 8959 documents: the norms branch declines.
        ("blocktree_index", "body"),
        // Norms on every document: `rewrite` collapses to `*:*`.
        ("norms_index", "body"),
        ("norms_index", "sparse_body"),
        // Doc values with no terms, no points and no skipper: nothing to
        // count from, even for the field on every document.
        ("doc_values_index", "varying"),
        ("doc_values_index", "sparse"),
        // A skip index, so `DocValuesSkipper.docCount()` answers directly.
        ("doc_values_skip_index", "skip_numeric"),
    ];

    for (index, field) in cases {
        let m = Manifest::load(index);
        open_leaf(index, |seg, open| {
            let leaf = seg.field_exists_leaf(field, None).expect("resolve leaf");
            let expected = m.num(&format!("count.fieldexists.{field}"));
            let rewrites = m.get(&format!("rewrite.fieldexists.{field}")) == "*:*";

            // These fixtures are single-segment, so the reader-wide pair the
            // norms branch reads is this leaf's own.
            assert_eq!(
                field_exists_rewrites_to_match_all_docs(
                    &[leaf],
                    leaf.terms_doc_count,
                    leaf.max_doc
                ),
                rewrites,
                "{index}/{field}: rewrite decision differs from real Lucene"
            );

            match count_field_exists_leaf(&leaf) {
                Some(count) => assert_eq!(
                    count, expected,
                    "{index}/{field}: shortcut count differs from real Lucene"
                ),
                None => {
                    // Java's `-1`: the query has to be run. Do that, and check
                    // the scan lands on Lucene's number too -- otherwise a
                    // shortcut that wrongly declines would pass this test.
                    let mut docs = lucene_search::VecCollector::default();
                    let source = leaf.source.expect("a field with no source cannot decline");
                    match source {
                        lucene_search::doc_value_query::FieldExistsSource::Norms => {
                            let norms = seg.field_norms(field).expect("norms for a normed field");
                            lucene_search::doc_value_query::search_field_exists_norms(
                                &norms,
                                seg.live_docs(),
                                seg.max_doc,
                                &mut docs,
                            )
                            .unwrap();
                        }
                        lucene_search::doc_value_query::FieldExistsSource::DocValues => {
                            let number = seg
                                .field_infos()
                                .fields
                                .iter()
                                .find(|f| f.name == field)
                                .expect("the field is in .fnm")
                                .number;
                            let (meta, data) = seg
                                .doc_values_for_field(number)
                                .expect("a doc-values field has a .dvm/.dvd");
                            let dv = lucene_search::doc_value_query::doc_values_field(meta, number)
                                .expect("the field has a doc-values entry");
                            lucene_search::doc_value_query::search_field_exists(
                                data,
                                dv,
                                seg.live_docs(),
                                seg.max_doc,
                                &mut docs,
                            )
                            .unwrap();
                        }
                        lucene_search::doc_value_query::FieldExistsSource::Vectors => {
                            unreachable!("no vector field is in this test's case list")
                        }
                    }
                    assert_eq!(
                        docs.docs.len() as i64,
                        expected,
                        "{index}/{field}: the scan must agree with real Lucene"
                    );
                    let _ = open;
                }
            }
        });
    }
}

/// A field absent from a segment is Java's `fieldInfo == null`: `count` is 0
/// and the rewrite is blocked, with no error.
#[test]
fn a_field_absent_from_the_segment_counts_zero_and_blocks_the_rewrite() {
    open_leaf("norms_index", |seg, _open| {
        let leaf = seg
            .field_exists_leaf("no-such-field", None)
            .expect("an absent field is not an error");
        assert_eq!(leaf.source, None);
        assert_eq!(count_field_exists_leaf(&leaf), Some(0));
        assert!(!field_exists_rewrites_to_match_all_docs(
            &[leaf],
            None,
            leaf.max_doc
        ));
    });
}
