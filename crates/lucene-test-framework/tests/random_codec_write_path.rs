//! The test framework's `RandomCodec` installed in
//! [`lucene_index::index_writer::IndexWriter`] (`RandomCodec::installed`):
//! for each of the fixture's seeds, every segment records `Asserting`, each
//! field's postings and doc values sit in the files of the candidate
//! instance `Math.abs(perFieldSeed ^ name.hashCode()) % size` picks
//! (equal-settings instances under their own suffixes), and the index reads
//! back through the asserting producers, flushed and merged: every term,
//! doc value and point range. `VerifyRandomCodec` (verify-write-path) has
//! Lucene write the same documents with the same codec and compares the
//! files and the points bytes.
// Test-support code opts out of the arithmetic gate at the file boundary:
// see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

#[path = "../examples/write_random_codec_fixture.rs"]
mod fixture;

use std::collections::{BTreeMap, BTreeSet};

use lucene_codecs::stored_fields::{self, StoredFieldVisitor, VisitStatus};
use lucene_search::reader::{IndexReader, IntersectVisitor, PostingsFlags, Relation, NO_MORE_DOCS};
use lucene_store::{Directory, FsDirectory};
use lucene_test_framework::codecs::asserting_reader::AssertingCodecReader;
use lucene_test_framework::index::random_codec::java_string_hash;
use lucene_util::test_support::TempDir;

use fixture::{deleted, point, PER_SEGMENT, SEEDS, SEGMENTS};

const TOTAL: usize = PER_SEGMENT * SEGMENTS;

/// The stored id of a document.
#[derive(Default)]
struct IdVisitor(Option<usize>);

impl StoredFieldVisitor for IdVisitor {
    fn needs_field(&mut self, _field_number: i32) -> stored_fields::Result<VisitStatus> {
        Ok(VisitStatus::Yes)
    }
    // `id` is the only stored field.
    fn string_field(&mut self, _field_number: i32, value: &str) -> stored_fields::Result<()> {
        self.0 = Some(value[1..].parse().unwrap());
        Ok(())
    }
}

/// Collects the documents inside a box of `p`.
struct Inside {
    lo: [i32; 3],
    hi: [i32; 3],
    docs: Vec<i32>,
}

fn decode(packed: &[u8], dim: usize) -> i32 {
    let bytes: [u8; 4] = packed[dim * 4..dim * 4 + 4].try_into().unwrap();
    (u32::from_be_bytes(bytes) ^ 0x8000_0000) as i32
}

impl IntersectVisitor for Inside {
    fn compare(&mut self, min: &[u8], max: &[u8]) -> Relation {
        let mut inside = true;
        for d in 0..3 {
            let (lo, hi) = (decode(min, d), decode(max, d));
            if hi < self.lo[d] || lo > self.hi[d] {
                return Relation::CellOutsideQuery;
            }
            inside &= lo >= self.lo[d] && hi <= self.hi[d];
        }
        if inside {
            Relation::CellInsideQuery
        } else {
            Relation::CellCrossesQuery
        }
    }
    fn visit(&mut self, doc: i32) {
        self.docs.push(doc);
    }
    fn visit_with_value(&mut self, doc: i32, value: &[u8]) {
        if (0..3).all(|d| (self.lo[d]..=self.hi[d]).contains(&decode(value, d))) {
            self.docs.push(doc);
        }
    }
}

/// The candidate index `RandomCodec` picks for `field` among `len`.
fn pick(seed: i32, field: &str, len: usize) -> usize {
    let units: Vec<u16> = field.encode_utf16().collect();
    ((seed ^ java_string_hash(&units)).wrapping_abs() % len as i32) as usize
}

/// The per-field file stems (`Lucene90_1`) one segment must have: each
/// selected candidate instance numbered among its name's instances in the
/// order its first field is written (postings by name, doc values `n`, `s`,
/// `b` as the fixture adds them).
fn expected_stems(seed: i64) -> BTreeSet<String> {
    let codec = fixture::codec(seed);
    let s = codec.per_field_seed();
    let mut stems = BTreeSet::new();
    let mut number = |names: Vec<(&'static str, usize)>| {
        let mut seen: Vec<usize> = Vec::new();
        let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
        for (name, index) in names {
            if !seen.contains(&index) {
                seen.push(index);
                let n = counts.entry(name).or_default();
                stems.insert(format!("{name}_{n}"));
                *n += 1;
            }
        }
    };
    let postings = codec.postings_formats();
    number(
        ["body", "code", "id", "kind"]
            .iter()
            .map(|f| {
                let i = pick(s, f, postings.len());
                (postings[i].name(), i)
            })
            .collect(),
    );
    let dv = codec.doc_values_formats();
    number(
        ["n", "s", "b"]
            .iter()
            .map(|f| {
                let i = pick(s, f, dv.len());
                (dv[i].name(), i)
            })
            .collect(),
    );
    stems
}

fn stems(dir: &dyn Directory, segment: &str) -> BTreeSet<String> {
    dir.list_all()
        .unwrap()
        .into_iter()
        .filter_map(|f| {
            let rest = f.strip_prefix(&format!("{segment}_"))?;
            let (stem, _) = rest.rsplit_once('.')?;
            // `_0_1.liv` is a deletes generation, not a format instance.
            stem.contains('_').then(|| stem.to_string())
        })
        .collect()
}

/// Reads the whole index back through the asserting producers.
fn read_back(dir: &dyn Directory, segments: usize) {
    let reader = AssertingCodecReader::open(dir).unwrap();
    assert_eq!(reader.leaves().len(), segments);
    let live_count = (0..TOTAL).filter(|&i| !deleted(i)).count();
    assert_eq!(reader.num_docs() as usize, live_count);
    let mut kinds: BTreeMap<String, usize> = BTreeMap::new();
    let mut codes: BTreeMap<String, usize> = BTreeMap::new();
    let mut bodies: BTreeMap<String, usize> = BTreeMap::new();
    let boxes = [
        ([0, -50, 0], [96, 50, 1000]),
        ([10, -10, 100], [40, 20, 600]),
        ([50, 0, 0], [60, 50, 999]),
    ];
    let mut in_boxes = [0usize; 3];
    for ctx in reader.leaves() {
        let leaf = ctx.reader;
        leaf.check_integrity().unwrap();
        let live = |doc: i32| leaf.live_docs().is_none_or(|bits| bits.get(doc as usize));
        let ids: Vec<usize> = (0..leaf.max_doc())
            .map(|doc| {
                let mut v = IdVisitor::default();
                leaf.document(doc, &mut v).unwrap();
                v.0.unwrap()
            })
            .collect();
        let mut n = leaf.numeric_doc_values("n").unwrap().expect("n");
        let mut s = leaf.sorted_doc_values("s").unwrap().expect("s");
        let mut b = leaf.binary_doc_values("b").unwrap().expect("b");
        for (doc, &i) in ids.iter().enumerate() {
            let doc = doc as i32;
            assert!(n.advance_exact(doc).unwrap());
            assert_eq!(n.long_value(), i as i64);
            assert!(s.advance_exact(doc).unwrap());
            assert_eq!(
                s.lookup_ord(s.ord_value()).unwrap(),
                format!("s{}", i % 13).into_bytes()
            );
            assert!(b.advance_exact(doc).unwrap());
            assert_eq!(b.binary_value(), format!("b{i}").as_bytes());
            assert_eq!(live(doc), !deleted(i), "d{i}");
        }
        for (field, counts) in [
            ("kind", &mut kinds),
            ("code", &mut codes),
            ("body", &mut bodies),
        ] {
            let terms = leaf.terms(field).unwrap().expect(field);
            let mut te = terms.iterator().unwrap();
            while let Some(term) = te.next().unwrap() {
                let term = String::from_utf8(term.to_vec()).unwrap();
                let mut postings = te.postings(PostingsFlags::Freqs).unwrap();
                loop {
                    let doc = postings.next_doc().unwrap();
                    if doc == NO_MORE_DOCS {
                        break;
                    }
                    if live(doc) {
                        *counts.entry(term.clone()).or_default() += 1;
                    }
                }
            }
        }
        let values = leaf.point_values("p").unwrap().expect("p");
        for (k, &(lo, hi)) in boxes.iter().enumerate() {
            let mut inside = Inside {
                lo,
                hi,
                docs: Vec::new(),
            };
            values.intersect(&mut inside).unwrap();
            in_boxes[k] += inside.docs.iter().filter(|&&d| live(d)).count();
        }
    }
    let count = |p: &dyn Fn(usize) -> bool| (0..TOTAL).filter(|&i| !deleted(i) && p(i)).count();
    for k in 0..11 {
        assert_eq!(kinds[&format!("k{k}")], count(&|i| i % 11 == k), "k{k}");
    }
    for k in 0..3 {
        assert_eq!(codes[&format!("c{k}")], count(&|i| i % 3 == k), "c{k}");
    }
    for k in 0..7 {
        assert_eq!(bodies[&format!("x{k}")], count(&|i| i % 7 == k), "x{k}");
    }
    assert_eq!(bodies["common"], live_count);
    for (k, &(lo, hi)) in boxes.iter().enumerate() {
        let expected = count(&|i| {
            let p = point(i);
            (0..3).all(|d| (lo[d]..=hi[d]).contains(&p[d]))
        });
        assert_eq!(in_boxes[k], expected, "box {k}");
    }
}

#[test]
fn every_seed_routes_its_fields_and_reads_back_flushed_and_merged() {
    for seed in SEEDS {
        let tmp = TempDir::new("random-codec-write-path");
        for (sub, merge) in [("flushed", false), ("merged", true)] {
            let path = tmp.path().join(sub);
            std::fs::create_dir_all(&path).unwrap();
            let dir = FsDirectory::open(&path);
            fixture::write(&dir, seed, merge);
            let segments: &[&str] = if merge { &["_2"] } else { &["_0", "_1"] };
            for segment in segments {
                assert_eq!(
                    stems(&dir, segment),
                    expected_stems(seed),
                    "{seed} {segment}"
                );
            }
            read_back(&dir, segments.len());
        }
    }
}

/// The seeds cover every postings candidate, both `Lucene90` instances side
/// by side, and two `Lucene104` instances in one segment (the four postings
/// fields' hashes differ in their low bits, so each seed's four candidates
/// all get a field).
#[test]
fn the_seeds_cover_every_candidate_and_twin_instances() {
    let mut postings = BTreeSet::new();
    let mut twins = BTreeSet::new();
    for seed in SEEDS {
        for stem in expected_stems(seed) {
            let (name, n) = stem.rsplit_once('_').unwrap();
            postings.insert(name.to_string());
            if n == "1" {
                twins.insert(name.to_string());
            }
        }
    }
    for name in [
        "Lucene104",
        "FST50",
        "Direct",
        "TestBloomFilteredLucenePostings",
        "MockRandom",
        "BlockTreeOrds",
        "LuceneFixedGap",
        "LuceneVarGapFixedInterval",
        "LuceneVarGapDocFreqInterval",
        "Asserting",
        "Lucene90",
    ] {
        assert!(postings.contains(name), "no seed routes a field to {name}");
    }
    assert!(twins.contains("Lucene90"), "{twins:?}");
    assert!(twins.contains("Lucene104"), "{twins:?}");
}
