//! M13 T13.3d against Lucene 10.5.0 (`fixtures/src/GenCompletion.java`): a
//! suggest field built by a real `IndexWriter` with `Completion104PostingsFormat`.
//! Fed the same documents, the Rust writer produces the same `.cmp` and `.lkp`
//! bytes, and the reader opens Lucene's.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use lucene_codecs::completion::{
    decode_weight, java_hash_map_order, CompletionFieldWriter, CompletionFields,
    CompletionFilesWriter, NrtOutputs, TermDoc,
};
use lucene_codecs::fst_util::get_bytes;
use lucene_store::codec_util::ID_LENGTH;
use lucene_store::DataOutput;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/completion")
}

fn segment_id() -> [u8; ID_LENGTH] {
    let hex = fs::read_to_string(dir().join("id.hex")).unwrap();
    let mut id = [0u8; ID_LENGTH];
    for (b, pair) in id.iter_mut().zip(hex.trim().as_bytes().chunks(2)) {
        *b = u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap();
    }
    id
}

/// `SuggestField`'s payload: `vint len, surface, vint weight + 1, type`.
fn payload(surface: &str, weight: i64) -> Vec<u8> {
    let mut out = Vec::new();
    out.write_vint(surface.len() as i32);
    out.extend_from_slice(surface.as_bytes());
    out.write_vint((weight as i32).wrapping_add(1));
    out.push(0);
    out
}

/// The postings of the field: term (the surface form itself, under the
/// keyword analyzer) to its documents, terms in byte order.
fn postings() -> BTreeMap<Vec<u8>, Vec<TermDoc>> {
    let mut terms: BTreeMap<Vec<u8>, Vec<TermDoc>> = BTreeMap::new();
    for line in fs::read_to_string(dir().join("docs.tsv")).unwrap().lines() {
        let c: Vec<&str> = line.split('\t').collect();
        let doc_id: i32 = c[0].parse().unwrap();
        let weight: i64 = c[2].parse().unwrap();
        terms
            .entry(c[1].as_bytes().to_vec())
            .or_default()
            .push(TermDoc {
                doc_id,
                payloads: vec![payload(c[1], weight)],
            });
    }
    terms
}

fn suffix() -> String {
    fs::read_to_string(dir().join("suffix.txt"))
        .unwrap()
        .trim()
        .to_string()
}

#[test]
fn rust_writes_the_same_completion_files_as_lucene() {
    let mut field = CompletionFieldWriter::new();
    for (term, docs) in postings() {
        field.write_term(&term, &docs).unwrap();
    }
    let mut files =
        CompletionFilesWriter::new("Completion104", "Lucene104", segment_id(), &suffix());
    files.add_field("suggest", 0, field).unwrap();
    let (lkp, cmp) = files.finish();
    assert_eq!(cmp, fs::read(dir().join("suggest.cmp")).unwrap(), ".cmp");
    assert_eq!(lkp, fs::read(dir().join("suggest.lkp")).unwrap(), ".lkp");
}

#[test]
fn rust_reads_lucenes_completion_files() {
    let fields = CompletionFields::open(
        "Completion104",
        &segment_id(),
        &suffix(),
        &fs::read(dir().join("suggest.cmp")).unwrap(),
        fs::read(dir().join("suggest.lkp")).unwrap(),
    )
    .unwrap();
    assert_eq!(fields.delegate_format, "Lucene104");
    assert_eq!(fields.fields.len(), 1);
    let meta = fields.fields[0].1;
    assert_eq!((meta.min_weight, meta.max_weight), (0, i64::from(i32::MAX)));
    let s = fields.suggester(0).unwrap().unwrap();
    // "light": weight 0, the last document
    let ndocs = postings().values().map(Vec::len).sum::<usize>() as i32;
    let out = get_bytes::<NrtOutputs>(&s.fst, b"light\x00\x00")
        .unwrap()
        .unwrap();
    assert_eq!(decode_weight(out.first), 0);
    assert_eq!(out.second.last().copied(), Some((ndocs - 3) as u8));
    assert_eq!(s.max_analyzed_paths_per_output, 30);
}

/// `hashmap.tsv`: the order Java's `HashMap` iterated each key list in,
/// including a table resized by the 13th, 25th and 49th keys and buckets of
/// eight to ten colliding keys ("Aa" and "BB" hash alike) that resize a
/// table below 64 slots instead of treeifying.
#[test]
fn field_table_order_is_javas_hash_map_order() {
    let text = fs::read_to_string(dir().join("hashmap.tsv")).unwrap();
    let mut checked = 0;
    for line in text.lines() {
        let (keys, want) = line.split_once("\t").unwrap();
        let keys: Vec<&str> = keys.split("\u{1f}").collect();
        let want: Vec<&str> = want.split("\u{1f}").collect();
        let got: Vec<&str> = java_hash_map_order(&keys)
            .into_iter()
            .map(|i| keys[i])
            .collect();
        assert_eq!(got, want, "keys {keys:?}");
        checked += 1;
    }
    assert_eq!(checked, 9);
}

#[test]
fn damaged_lucene_files_are_refused() {
    let cmp = fs::read(dir().join("suggest.cmp")).unwrap();
    let lkp = fs::read(dir().join("suggest.lkp")).unwrap();
    let open = |cmp: &[u8], lkp: Vec<u8>| {
        CompletionFields::open("Completion104", &segment_id(), &suffix(), cmp, lkp)
    };
    assert!(open(&cmp, lkp.clone()).is_ok());
    let mut flipped = cmp.clone();
    let mid = flipped.len() / 2;
    flipped[mid] ^= 0x40;
    assert!(open(&flipped, lkp.clone()).is_err(), "a flipped .cmp byte");
    assert!(
        open(&cmp, lkp[..lkp.len() / 2].to_vec()).is_err(),
        "a truncated .lkp"
    );
    assert!(
        open(&cmp[..cmp.len() - 1], lkp).is_err(),
        "a truncated .cmp"
    );
}
