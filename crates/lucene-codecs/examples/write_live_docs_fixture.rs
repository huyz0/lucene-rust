//! Writes a `live_docs::write`-produced `.liv` file plus a manifest to the
//! directory given as the first CLI argument.
//!
//! Reverse-direction differential test (Rust writes, Java reads), same
//! division of labor as `write_norms_fixture.rs`: `fixtures/src/
//! VerifyLiveDocs.java` opens the result through real Lucene's
//! `Lucene90LiveDocsFormat` with a hand-built `SegmentCommitInfo`, so this
//! slice doesn't also need a `.si`/`segments_N` writer.
//!
//! Three segments cover the distinct cases this simple, always-dense format
//! actually has: no deletions, some deletions, and a doc count that isn't a
//! multiple of 64 (the bitset's last-word partial-bits edge case).
//!
//! Run: `cargo run -p lucene-codecs --example write_live_docs_fixture -- <dir>`

use lucene_codecs::live_docs;
use lucene_store::{DataOutput, Directory, FsDirectory};
use lucene_util::fixed_bit_set::FixedBitSet;
use std::io::Write;

const SEGMENT_ID: [u8; 16] = *b"rustwrittenliv01";

struct Segment {
    name: &'static str,
    max_doc: usize,
    del_gen: i64,
    deleted_doc_ids: &'static [usize],
}

fn main() {
    let out_dir = std::env::args()
        .nth(1)
        .expect("usage: write_live_docs_fixture <output-dir>");
    std::fs::create_dir_all(&out_dir).unwrap();

    let segments = [
        // All docs live -- no deletions at all, still a valid (if
        // pointless in practice) `.liv` shape to verify.
        Segment {
            name: "_0",
            max_doc: 5,
            del_gen: 1,
            deleted_doc_ids: &[],
        },
        // A typical case: some docs deleted, doc count a multiple of 64
        // wouldn't by itself prove anything about the partial-word case, so
        // this one is intentionally not doc-count-aligned to 64 either.
        Segment {
            name: "_1",
            max_doc: 5,
            del_gen: 3,
            deleted_doc_ids: &[1, 3],
        },
        // Word-boundary edge case: 130 docs -> 3 words, last word holds
        // only 2 live-bit slots (129 % 64 == 1 relevant bit... concretely
        // bits2words(130) == 3, last word covers doc ids 128-129 only).
        // Delete doc 129 (last doc, in the partial last word) and doc 64
        // (first doc of a full middle word) to exercise both a full and a
        // partial word with deletions.
        Segment {
            name: "_2",
            max_doc: 130,
            del_gen: 2,
            deleted_doc_ids: &[64, 129],
        },
    ];

    let dir = FsDirectory::open(&out_dir);
    let mut manifest = std::fs::File::create(format!("{out_dir}/manifest.properties")).unwrap();
    writeln!(manifest, "id_hex={}", hex(&SEGMENT_ID)).unwrap();
    writeln!(
        manifest,
        "segments={}",
        segments
            .iter()
            .map(|s| s.name)
            .collect::<Vec<_>>()
            .join(",")
    )
    .unwrap();

    for seg in &segments {
        let mut live_docs = FixedBitSet::new(seg.max_doc);
        for i in 0..seg.max_doc {
            live_docs.set(i);
        }
        for &doc_id in seg.deleted_doc_ids {
            live_docs.clear(doc_id);
        }
        let expected_del_count = seg.deleted_doc_ids.len();

        let bytes = live_docs::write(&live_docs, &SEGMENT_ID, seg.del_gen, expected_del_count)
            .expect("live docs write");

        // Mirrors `IndexFileNames.fileNameFromGeneration(name, "liv", gen)`:
        // gen == 0 would be the bare `<name>.liv`, but `.liv` generations
        // always start at 1 in practice (a fresh segment has no `.liv` file
        // at all -- gen -1 -- until its first deletion), so every case here
        // uses the `_<base36(gen)>` suffix form.
        let file_name = format!(
            "{}_{}.liv",
            seg.name,
            lucene_util::base36::to_base36(seg.del_gen)
        );
        let mut out = dir.create_output(&file_name).unwrap();
        out.write_bytes(&bytes);
        out.close().unwrap();
        dir.sync(std::slice::from_ref(&file_name)).unwrap();

        writeln!(manifest, "{}.liv_file_name={file_name}", seg.name).unwrap();
        writeln!(manifest, "{}.max_doc={}", seg.name, seg.max_doc).unwrap();
        writeln!(manifest, "{}.del_gen={}", seg.name, seg.del_gen).unwrap();
        writeln!(manifest, "{}.del_count={expected_del_count}", seg.name).unwrap();
        let deleted: Vec<String> = seg.deleted_doc_ids.iter().map(|d| d.to_string()).collect();
        writeln!(
            manifest,
            "{}.deleted_doc_ids={}",
            seg.name,
            deleted.join(",")
        )
        .unwrap();
    }

    println!("wrote live docs fixture to {out_dir}");
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
