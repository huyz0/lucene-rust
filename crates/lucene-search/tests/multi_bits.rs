//! `MultiBits::live_docs` over Java's `fixtures/data/merge_policies/commits_index`
//! (`GenMergePolicies.commitsIndex`): its latest commit has three segments,
//! the first with document `a` deleted, and Java recorded `maxDoc=4`,
//! `numDocs=3` for it (`commits.properties`). The top-level bits must agree
//! with those counts, with each segment's own live docs, and with the stored
//! ids; a commit without deletions has none.

use lucene_codecs::stored_fields::FieldValue;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::multi_bits::MultiBits;
use lucene_store::FsDirectory;

#[test]
fn multi_bits_answers_for_every_segment() {
    let dir = FsDirectory::open(format!(
        "{}/../../fixtures/data/merge_policies/commits_index",
        env!("CARGO_MANIFEST_DIR")
    ));
    let commits = DirectoryReader::list_commits(&dir).unwrap();

    let first = DirectoryReader::open_commit(&dir, &commits[0]).unwrap();
    assert!(MultiBits::live_docs(&first).is_none(), "no deletions yet");

    let reader = DirectoryReader::open_commit(&dir, &commits[2]).unwrap();
    let bits = MultiBits::live_docs(&reader).unwrap();
    assert_eq!(bits.len(), 4);
    let live: Vec<i32> = (0..bits.len()).filter(|&d| bits.get(d)).collect();
    assert_eq!(live.len(), 3);
    assert_eq!(live.len() as i32, reader.num_docs());
    for s in reader.segment_readers() {
        for local in 0..s.max_doc {
            let expected = s
                .live_docs()
                .is_none_or(|l| l.get(usize::try_from(local).unwrap()));
            assert_eq!(bits.get(s.doc_base + local), expected);
        }
    }
    let ids: Vec<String> = live
        .iter()
        .map(|&d| {
            let s = reader
                .segment_readers()
                .iter()
                .rev()
                .find(|s| s.doc_base <= d)
                .unwrap();
            match &s.stored_document(d - s.doc_base).unwrap().unwrap().fields[0].value {
                FieldValue::String(id) => id.clone(),
                other => panic!("{other:?}"),
            }
        })
        .collect();
    assert_eq!(ids, ["b", "c", "d"]);
}
