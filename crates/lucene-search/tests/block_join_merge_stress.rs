//! Block integrity under merges -- M10's named risk: a merge that splits or
//! reorders a document block breaks every join silently. Seeded streams of
//! block adds, block updates (`updateDocuments` on a term every document of
//! the block carries), whole-block deletes, flushes, commits and natural
//! merges under an aggressive merge policy, unsorted and index-sorted (a
//! parent field moving whole blocks), then `CheckJoinIndex` over the result,
//! `CheckIndex`, and every live block read back whole, in order, exactly
//! once -- the parents of a sorted segment in key order.

// Test fixtures' own arithmetic -- see `docs/arithmetic-gate.md`'s "Test code".
#![allow(clippy::arithmetic_side_effects)]

use std::collections::BTreeMap;

use lucene_codecs::stored_fields::FieldValue;
use lucene_index::buffered_updates::Term;
use lucene_index::document::{Document, NumericDocValuesField, Store, StringField};
use lucene_index::index_writer::{IndexWriter, MergePolicyConfig};
use lucene_index::segment_info::{IndexSortField, LuceneVersion};
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::join::{check_join_index, QueryBitSetProducer};
use lucene_search::query::{BooleanQuery, Clause, TermQuery};
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;

const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};

/// xorshift64*: a seeded stream, no dependency.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// Version `v` of block `b`: `children` children `c{b}.{v}.{j}` then the
/// parent `p{b}.{v}`, every document tagged `block:{b}`; the parent carries
/// `rank`.
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

/// What the index must hold: each live block's version, child count and rank.
type Model = BTreeMap<u64, (u64, u64, i64)>;

fn run(seed: u64, sorted: bool, ops: usize) -> usize {
    let tmp = TempDir::new(&format!("block-merge-stress-{seed}-{sorted}"));
    let dir = FsDirectory::open(tmp.path());
    let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
    let mut model = Model::new();
    let mut commits = 0usize;
    {
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
        w.set_parent_field(Some("_parent")).unwrap();
        if sorted {
            w.set_index_sort(Some(&[IndexSortField::long(
                "rank",
                rng.below(2) == 0,
                None,
            )]))
            .unwrap();
        }
        w.set_max_buffered_docs(5 + rng.below(30) as i32).unwrap();
        w.set_merge_policy(Some(MergePolicyConfig {
            max_merge_at_once: 2 + rng.below(3) as usize,
            segments_per_tier: 2,
            floor_segment_size: 1,
            ..MergePolicyConfig::default()
        }));
        let mut next_block = 0u64;
        for _ in 0..ops {
            match rng.below(100) {
                0..=59 => {
                    let (children, rank) = (rng.below(6), rng.below(50) as i64);
                    w.add_fields_documents(&block(next_block, 0, children, rank))
                        .unwrap();
                    model.insert(next_block, (0, children, rank));
                    next_block += 1;
                }
                60..=79 if !model.is_empty() => {
                    let b = *model
                        .keys()
                        .nth(rng.below(model.len() as u64) as usize)
                        .unwrap();
                    let (v, _, _) = model[&b];
                    let (children, rank) = (rng.below(6), rng.below(50) as i64);
                    w.update_fields_documents(
                        Term::new("block", b.to_string()),
                        &block(b, v + 1, children, rank),
                    )
                    .unwrap();
                    model.insert(b, (v + 1, children, rank));
                }
                80..=91 if !model.is_empty() => {
                    let b = *model
                        .keys()
                        .nth(rng.below(model.len() as u64) as usize)
                        .unwrap();
                    w.delete_documents_by_term(&[Term::new("block", b.to_string())])
                        .unwrap();
                    model.remove(&b);
                }
                _ => {
                    w.commit().unwrap();
                    w.maybe_merge().unwrap();
                    commits += 1;
                }
            }
        }
        w.commit().unwrap();
        w.maybe_merge().unwrap();
        w.commit().unwrap();
    }

    for r in lucene_index::check_index::check_directory(&dir).unwrap() {
        assert!(
            r.all_passed(),
            "seed {seed}: {}: {:?}",
            r.segment_name,
            r.failures()
        );
    }
    let reader = DirectoryReader::open(&dir).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let parents = QueryBitSetProducer::new(BooleanQuery {
        must: vec![Clause::Term(TermQuery::new("type", b"parent".to_vec()))],
        ..Default::default()
    });
    if !model.is_empty() {
        check_join_index(&segments, &parents)
            .unwrap_or_else(|e| panic!("seed {seed} sorted={sorted}: {e}"));
    }

    // Every live document, block by block.
    let mut seen = Model::new();
    for seg in reader.segment_readers() {
        let mut children: Vec<String> = Vec::new();
        let mut last_rank: Option<i64> = None;
        for doc in 0..seg.max_doc {
            if seg.live_docs().is_some_and(|l| !l.get_doc(doc)) {
                continue;
            }
            let stored = seg.stored_document(doc).unwrap().unwrap();
            let FieldValue::String(id) = &stored.fields[0].value else {
                panic!("an id");
            };
            if let Some(rest) = id.strip_prefix('c') {
                children.push(rest.to_string());
                continue;
            }
            let (b, v) = id[1..].split_once('.').unwrap();
            let (b, v): (u64, u64) = (b.parse().unwrap(), v.parse().unwrap());
            let (mv, mc, rank) = *model
                .get(&b)
                .unwrap_or_else(|| panic!("seed {seed}: block {b} is not live"));
            assert_eq!(v, mv, "seed {seed}: block {b}'s version");
            let want: Vec<String> = (0..mc).map(|j| format!("{b}.{v}.{j}")).collect();
            assert_eq!(
                children, want,
                "seed {seed}: block {b}'s children, in order"
            );
            children.clear();
            assert!(seen.insert(b, (v, mc, rank)).is_none(), "block {b} twice");
            if seg.index_sort().is_some() {
                let reverse = seg.index_sort().unwrap()[0].reverse;
                if let Some(prev) = last_rank {
                    assert!(
                        if reverse { prev >= rank } else { prev <= rank },
                        "seed {seed}: parents out of sort order in {}",
                        seg.segment_name
                    );
                }
                last_rank = Some(rank);
            }
        }
        assert!(
            children.is_empty(),
            "seed {seed}: children after the last parent"
        );
    }
    assert_eq!(seen, model, "seed {seed}: every live block exactly once");
    assert!(
        reader.segment_readers().len() < commits.max(4),
        "merges ran"
    );
    commits
}

#[test]
fn blocks_survive_many_merges_unsorted() {
    for seed in 0..12 {
        run(seed, false, 700);
    }
}

#[test]
fn blocks_survive_many_merges_index_sorted() {
    for seed in 100..112 {
        run(seed, true, 700);
    }
}
