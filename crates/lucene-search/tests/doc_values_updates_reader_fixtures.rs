//! `DirectoryReader` over a real index whose doc-values were **updated in
//! place** by an actual `IndexWriter`. Regenerate with
//! `fixtures/src/GenDocValuesUpdates.java`.
//!
//! `crates/lucene-index/tests/doc_values_updates_fixtures.rs` proves the
//! *format* is decoded correctly by resolving the generations by hand. This
//! proves the **reader** does it: `SegmentReader::open` must read the
//! generational `.fnm` (not the base one) to learn each field's
//! `FieldInfo.docValuesGen`, and `doc_values_for_field` must then serve an
//! updated field from its own generation while a never-updated field keeps
//! coming out of the base column -- `SegmentDocValuesProducer`'s per-field
//! producer map.
//!
//! Reading the base `.fnm`, or resolving every field to
//! `SegmentCommitInfo.docValuesGen`, produces a reader that opens cleanly and
//! answers with superseded values. Nothing structural catches that; only
//! asserting the values does.

use lucene_codecs::doc_values;
use lucene_search::directory_reader::DirectoryReader;
use lucene_store::FsDirectory;

fn fixture_dir() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/doc_values_updates_index/"
    )
    .to_string()
}

fn manifest_column(key: &str) -> Vec<Option<String>> {
    let text = std::fs::read_to_string(format!("{}manifest.properties", fixture_dir()))
        .expect("run fixtures generator first (GenDocValuesUpdates)");
    let line = text
        .lines()
        .find_map(|l| l.strip_prefix(&format!("{key}=")))
        .unwrap_or_else(|| panic!("manifest key {key} missing"));
    line.split(',')
        .map(|c| {
            if c.is_empty() {
                None
            } else {
                Some(c.to_string())
            }
        })
        .collect()
}

#[test]
fn the_reader_serves_each_field_from_its_own_doc_values_generation() {
    let dir = FsDirectory::open(fixture_dir());
    let reader = DirectoryReader::open(&dir).expect("open the updated index");
    assert_eq!(reader.segment_readers().len(), 1);
    let seg = &reader.segment_readers()[0];

    // The reader must be on the *generational* FieldInfos: the base one
    // reports every field at generation -1.
    let val = seg
        .field_infos()
        .fields
        .iter()
        .find(|f| f.name == "val")
        .expect("field val");
    let tag = seg
        .field_infos()
        .fields
        .iter()
        .find(|f| f.name == "tag")
        .expect("field tag");
    let keep = seg
        .field_infos()
        .fields
        .iter()
        .find(|f| f.name == "keep")
        .expect("field keep");
    assert_ne!(val.doc_values_gen, -1);
    assert_ne!(tag.doc_values_gen, -1);
    assert_ne!(
        val.doc_values_gen, tag.doc_values_gen,
        "each updated field takes its own generation"
    );
    assert_eq!(
        keep.doc_values_gen, -1,
        "a never-updated field stays on the base column"
    );

    let expected_val = manifest_column("expected_val");
    let expected_keep = manifest_column("expected_keep");
    let expected_tag = manifest_column("expected_tag");
    assert_eq!(expected_val.len(), seg.max_doc as usize);

    for (name, number, expected) in [
        ("val", val.number, &expected_val),
        ("keep", keep.number, &expected_keep),
    ] {
        let (meta, data) = seg
            .doc_values_for_field(number)
            .unwrap_or_else(|| panic!("no doc values resolved for {name}"));
        let entry = meta.numeric_entry(number).expect("NUMERIC entry");
        for doc in 0..seg.max_doc {
            let got = doc_values::numeric_value(data, entry, doc).unwrap();
            let want = expected[doc as usize]
                .as_ref()
                .map(|v| v.parse::<i64>().unwrap());
            assert_eq!(got, want, "{name} doc {doc}");
        }
    }

    let (meta, data) = seg.doc_values_for_field(tag.number).expect("tag resolved");
    let entry = meta.binary_entry(tag.number).expect("BINARY entry");
    for doc in 0..seg.max_doc {
        let got = doc_values::binary_value(data, entry, doc)
            .unwrap()
            .map(|v| String::from_utf8(v.to_vec()).unwrap());
        assert_eq!(got, expected_tag[doc as usize], "tag doc {doc}");
    }
}

#[test]
fn the_base_column_is_still_what_an_un_updated_field_reads_and_is_still_superseded_for_the_rest() {
    // The base `.dvm`/`.dvd` are never rewritten, so they still hold `val`'s
    // and `tag`'s *original* values. `doc_values_meta`/`doc_values_data` are
    // deliberately still that base pair -- what must differ is what
    // `doc_values_for_field` returns for a field that has a generation.
    let dir = FsDirectory::open(fixture_dir());
    let reader = DirectoryReader::open(&dir).expect("open the updated index");
    let seg = &reader.segment_readers()[0];
    let val = seg
        .field_infos()
        .fields
        .iter()
        .find(|f| f.name == "val")
        .unwrap();

    let base_meta = seg.doc_values_meta().expect("a base .dvm");
    let base_data = seg.doc_values_data().expect("a base .dvd");
    let base_entry = base_meta
        .numeric_entry(val.number)
        .expect("the base column still describes val");
    // The first document the fixture's reset round did *not* touch, so its
    // base value and its generation value are both present and differ.
    let expected = manifest_column("expected_val");
    let doc = (0..seg.max_doc)
        .find(|d| expected[*d as usize].is_some())
        .expect("some document still has a value");
    assert_eq!(
        doc_values::numeric_value(base_data, base_entry, doc).unwrap(),
        Some(doc as i64),
        "the base column is untouched by an update (it still holds doc {doc}'s original value)"
    );

    let (gen_meta, gen_data) = seg.doc_values_for_field(val.number).unwrap();
    let gen_entry = gen_meta.numeric_entry(val.number).unwrap();
    assert_eq!(
        doc_values::numeric_value(gen_data, gen_entry, doc).unwrap(),
        expected[doc as usize].as_ref().map(|v| v.parse().unwrap()),
        "and the generation is what a field with one resolves to"
    );
    assert_ne!(
        doc_values::numeric_value(gen_data, gen_entry, doc).unwrap(),
        Some(doc as i64),
        "the two must actually differ, or this test proves nothing"
    );

    // ...and a document the last round *reset* reads back as having no value
    // through the generation, while the base still holds its original.
    let reset_doc = (0..seg.max_doc)
        .find(|d| expected[*d as usize].is_none())
        .expect("some document was reset");
    assert_eq!(
        doc_values::numeric_value(gen_data, gen_entry, reset_doc).unwrap(),
        None
    );
    assert_eq!(
        doc_values::numeric_value(base_data, base_entry, reset_doc).unwrap(),
        Some(reset_doc as i64)
    );
}
