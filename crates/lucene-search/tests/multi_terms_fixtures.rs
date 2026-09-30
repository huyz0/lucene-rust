//! Differential test for `lucene_search::multi_terms` against
//! `fixtures/data/multi_terms/manifest.properties`
//! (`fixtures/src/GenMultiTerms.java`): over four segments, one of them
//! missing a field and one with a deleted document, the indexed field names
//! (`MultiFields`), each field's summed statistics and min/max
//! (`MultiTerms`), every term with its `docFreq`/`totalTermFreq` and
//! postings (`MultiTermsEnum`, `MultiPostingsEnum`), `seekCeil` from a set of
//! targets, and `advance` over a term spanning three segments must all be
//! Java's.
//!
//! Regenerate with `scripts/gen-fixtures.sh --only GenMultiTerms`.

use std::collections::HashMap;

use lucene_codecs::blocktree::SeekStatus;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::multi_terms::{indexed_fields, MultiTerms, NO_MORE_DOCS};
use lucene_search::reader::PostingsFlags;
use lucene_store::FsDirectory;

fn status(s: SeekStatus) -> &'static str {
    match s {
        SeekStatus::Found => "FOUND",
        SeekStatus::NotFound => "NOT_FOUND",
        SeekStatus::End => "END",
    }
}

fn utf8(b: &[u8]) -> String {
    String::from_utf8(b.to_vec()).unwrap()
}

#[test]
fn multi_terms_match_lucene() {
    let root = format!(
        "{}/../../fixtures/data/multi_terms",
        env!("CARGO_MANIFEST_DIR")
    );
    let want: HashMap<String, String> =
        std::fs::read_to_string(format!("{root}/manifest.properties"))
            .unwrap()
            .lines()
            .filter(|l| !l.starts_with('#'))
            .filter_map(|l| l.split_once('='))
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
    let dir = FsDirectory::open(format!("{root}/index"));
    let reader = DirectoryReader::open(&dir).unwrap();

    assert_eq!(indexed_fields(&reader).join(","), want["fields"]);

    for field in ["body", "tag", "id", "absent"] {
        let key = |k: &str| want[&format!("{field}.{k}")].clone();
        let Some(terms) = MultiTerms::get_terms(&reader, field).unwrap() else {
            assert_eq!(key("terms"), "null", "{field}");
            continue;
        };
        assert_eq!(
            terms.sum_total_term_freq().to_string(),
            key("sum_total_term_freq")
        );
        assert_eq!(terms.sum_doc_freq().to_string(), key("sum_doc_freq"));
        assert_eq!(terms.doc_count().to_string(), key("doc_count"));
        assert_eq!(utf8(&terms.min().unwrap().unwrap()), key("min"));
        assert_eq!(utf8(&terms.max().unwrap().unwrap()), key("max"));
        assert_eq!(terms.size(), -1);

        let mut te = terms.iterator().unwrap();
        let mut all = Vec::new();
        while let Some(t) = te.next().unwrap() {
            let t = utf8(t);
            let df = te.doc_freq().unwrap();
            let ttf = te.total_term_freq().unwrap();
            let mut pe = te.postings(PostingsFlags::Freqs).unwrap();
            let mut docs = Vec::new();
            loop {
                let doc = pe.next_doc().unwrap();
                if doc == NO_MORE_DOCS {
                    break;
                }
                docs.push(format!("{doc}/{}", pe.freq()));
            }
            all.push(format!("{t}:{df}:{ttf}:{}", docs.join(" ")));
        }
        assert_eq!(all.join(","), key("terms"), "{field} terms");
        assert!(te.next().unwrap().is_none(), "stays exhausted");

        let mut seeks = Vec::new();
        for target in ["", "a", "fox", "fp", "lazz", "the", "zz", "blue", "c"] {
            let mut s = terms.iterator().unwrap();
            let st = s.try_seek_ceil(target.as_bytes()).unwrap();
            let at = if st == SeekStatus::End {
                String::new()
            } else {
                format!("{}/{}", utf8(s.term().unwrap()), s.doc_freq().unwrap())
            };
            seeks.push(format!("{target}>{}>{at}", status(st)));
        }
        assert_eq!(seeks.join(","), key("seek_ceil"), "{field} seekCeil");
    }

    let terms = MultiTerms::get_terms(&reader, "body").unwrap().unwrap();
    let mut te = terms.iterator().unwrap();
    assert!(te.try_seek_exact(b"fox").unwrap());
    let mut adv = Vec::new();
    for target in [1, 4, 5, 8, 20] {
        let mut pe = te.postings(PostingsFlags::Freqs).unwrap();
        let doc = pe.advance(target).unwrap();
        adv.push(if doc == NO_MORE_DOCS {
            format!("{target}>end")
        } else {
            format!("{target}>{doc}/{}", pe.freq())
        });
    }
    assert_eq!(adv.join(","), want["body.fox.advance"]);
    // After a seek, `next` carries on from the sought term.
    assert_eq!(te.next().unwrap(), Some(&b"lazy"[..]));
    assert!(!te.try_seek_exact(b"fp").unwrap());
}
