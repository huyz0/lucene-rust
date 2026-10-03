//! Writes two block-join indexes through this port's `IndexWriter` -- one
//! unsorted, one index-sorted on the parents' `rank` -- for `VerifyJoin` to
//! open with real Lucene 10.5.0: `CheckIndex`, `CheckJoinIndex`, every live
//! block's children through `ToChildBlockJoinQuery` and its parent through
//! `ToParentBlockJoinQuery`, then Lucene's own `IndexWriter` appending blocks
//! and force-merging the index, and the checks again.
//!
//! Each index is a seeded stream of block adds (zero to five children and a
//! parent), block updates (`updateDocuments` on the `block` term every
//! document of a block carries), whole-block deletes, commits and natural
//! merges under an aggressive merge policy, with the parent field `_parent`
//! (`IndexWriterConfig.setParentField`), which the sorted index needs to move
//! whole blocks. `<out>/<variant>/blocks.tsv` lists the live blocks the index
//! must hold: `block`, `version`, `children`, `rank`.
//!
//! Usage: `write_block_join_fixture <output-dir>`.
#![allow(clippy::arithmetic_side_effects)]

use std::collections::BTreeMap;
use std::fmt::Write as _;

use lucene_index::buffered_updates::Term;
use lucene_index::document::{Document, NumericDocValuesField, Store, StringField};
use lucene_index::index_writer::{IndexWriter, MergePolicyConfig};
use lucene_index::segment_info::{IndexSortField, LuceneVersion};
use lucene_store::FsDirectory;

/// xorshift64*.
struct Rng(u64);

impl Rng {
    fn below(&mut self, n: u64) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) % n
    }
}

/// Must match `VerifyJoin.java`'s reading of the ids.
fn block(b: u64, v: u64, children: u64, rank: i64) -> Vec<Document> {
    let mut docs = Vec::new();
    for j in 0..children {
        let mut d = Document::new();
        d.add(StringField::new("id", format!("c{b}.{v}.{j}"), Store::Yes));
        d.add(StringField::new("block", b.to_string(), Store::No));
        d.add(StringField::new("type", "child", Store::No));
        docs.push(d);
    }
    let mut p = Document::new();
    p.add(StringField::new("id", format!("p{b}.{v}"), Store::Yes));
    p.add(StringField::new("block", b.to_string(), Store::No));
    p.add(StringField::new("type", "parent", Store::No));
    p.add(NumericDocValuesField::new("rank", rank));
    docs.push(p);
    docs
}

fn write(out: &std::path::Path, seed: u64, sorted: bool) {
    let index = out.join("index");
    std::fs::create_dir_all(&index).expect("create index dir");
    let dir = FsDirectory::open(&index);
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", version).expect("open writer");
    w.set_parent_field(Some("_parent")).expect("parent field");
    if sorted {
        w.set_index_sort(Some(&[IndexSortField::long("rank", false, None)]))
            .expect("index sort");
    }
    w.set_max_buffered_docs(17).expect("buffer");
    w.set_merge_policy(Some(MergePolicyConfig {
        max_merge_at_once: 3,
        segments_per_tier: 2,
        floor_segment_size: 1,
        ..MergePolicyConfig::default()
    }));
    let mut rng = Rng(seed);
    let mut live: BTreeMap<u64, (u64, u64, i64)> = BTreeMap::new();
    let mut next = 0u64;
    for _ in 0..1500 {
        match rng.below(100) {
            0..=59 => {
                let (children, rank) = (rng.below(6), rng.below(40) as i64);
                w.add_fields_documents(&block(next, 0, children, rank))
                    .expect("add");
                live.insert(next, (0, children, rank));
                next += 1;
            }
            60..=79 if !live.is_empty() => {
                let b = *live
                    .keys()
                    .nth(rng.below(live.len() as u64) as usize)
                    .expect("b");
                let v = live[&b].0 + 1;
                let (children, rank) = (rng.below(6), rng.below(40) as i64);
                w.update_fields_documents(
                    Term::new("block", b.to_string()),
                    &block(b, v, children, rank),
                )
                .expect("update");
                live.insert(b, (v, children, rank));
            }
            80..=91 if !live.is_empty() => {
                let b = *live
                    .keys()
                    .nth(rng.below(live.len() as u64) as usize)
                    .expect("b");
                w.delete_documents_by_term(&[Term::new("block", b.to_string())])
                    .expect("delete");
                live.remove(&b);
            }
            _ => {
                w.commit().expect("commit");
                w.maybe_merge().expect("merge");
            }
        }
    }
    w.commit().expect("commit");
    let mut tsv = String::new();
    for (b, (v, children, rank)) in &live {
        writeln!(tsv, "{b}\t{v}\t{children}\t{rank}").expect("format");
    }
    std::fs::write(out.join("blocks.tsv"), tsv).expect("write blocks.tsv");
}

fn main() {
    let out = std::env::args()
        .nth(1)
        .expect("usage: write_block_join_fixture <output-dir>");
    let out = std::path::Path::new(&out);
    let _ = std::fs::remove_dir_all(out);
    write(&out.join("unsorted"), 0x5eed_0001, false);
    write(&out.join("sorted"), 0x5eed_0002, true);
}
