//! lucene-codecs' SimpleText points (`.dim`/`.dii`) against Java:
//! `GenCodecsSimpleTextCodec`'s two indexes (flushed, and force-merged into
//! a compound file) record what Java's reader makes of every points field --
//! its stats, a pre-order walk of the tree (cells, sizes), the hashes of
//! `visitDocValues` and `visitDocIDs`, and boxes through `intersect` (the
//! relations returned, each visit) and `estimatePointCount` -- and what each
//! `writeField` was handed (`pin.SEG.tsv`).
//!
//! Here the Rust reader opens Java's files and agrees with every manifest,
//! the Rust writer rebuilds both files byte for byte from Java's inputs, and
//! damaged copies of Java's bytes are refused.

// Test-support arithmetic over small in-memory counters; see
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::fs;
use std::path::{Path, PathBuf};

use lucene_codecs::field_infos::FieldInfos;
use lucene_codecs::points::{IntersectVisitor, Relation};
use lucene_codecs::simpletext::bkd_reader::{SimpleTextBkdReader, SimpleTextPointTree};
use lucene_codecs::simpletext::compound::{self, SimpleTextCompoundReader};
use lucene_codecs::simpletext::field_infos as st_infos;
use lucene_codecs::simpletext::points_reader::SimpleTextPointsReader;
use lucene_codecs::simpletext::points_writer::{
    points_file_name, points_index_file_name, SimpleTextPointsWriter,
};
use lucene_codecs::simpletext::{util, Error};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/codecs_simpletext/codec")
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

fn unhex_str(s: &str) -> String {
    String::from_utf8(unhex(s)).unwrap()
}

fn rows(path: &Path) -> Vec<Vec<String>> {
    fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{path:?}: {e} (run GenCodecsSimpleTextCodec)"))
        .lines()
        .map(|l| l.split('\t').map(str::to_string).collect())
        .collect()
}

/// `text` with its checksum line replaced by a valid one: semantic damage.
fn reseal(text: &str) -> Vec<u8> {
    let mut out = text.rsplit_once("checksum ").unwrap().0.as_bytes().to_vec();
    util::write_checksum(&mut out);
    out
}

/// A segment with points: its name, maxDoc, field infos, `.dim` and `.dii`
/// (from the directory or its compound file), and its manifests' directory.
struct Seg {
    name: String,
    max_doc: usize,
    infos: FieldInfos,
    dim: Vec<u8>,
    dii: Vec<u8>,
    dir: PathBuf,
}

fn segments(case: &str) -> Vec<Seg> {
    let dir = root().join(case);
    let index = dir.join("index");
    let mut out = Vec::new();
    for r in rows(&dir.join("segments.tsv")) {
        let name = r[0].clone();
        if !dir.join(format!("points.{name}.tsv")).exists() {
            continue;
        }
        let scf = index.join(compound::compound_file_name(&name));
        let core = |file: &str| -> Vec<u8> {
            if scf.exists() {
                let scf = fs::read(&scf).unwrap();
                let r = SimpleTextCompoundReader::open(&scf, &name).unwrap();
                r.open_input(&scf, file).unwrap().to_vec()
            } else {
                fs::read(index.join(file)).unwrap()
            }
        };
        let infos = st_infos::read(&core(&st_infos::field_infos_file_name(&name, ""))).unwrap();
        out.push(Seg {
            max_doc: r[1].parse().unwrap(),
            infos,
            dim: core(&points_file_name(&name, "")),
            dii: core(&points_index_file_name(&name, "")),
            dir: dir.clone(),
            name,
        });
    }
    out
}

/// Both cases' segments with points.
fn all_segments() -> Vec<Seg> {
    let segs: Vec<Seg> = ["flush", "merged"]
        .iter()
        .flat_map(|c| segments(c))
        .collect();
    assert!(segs.len() >= 3, "flush _0/_1 and the merged segment");
    segs
}

/// One field's block of `points.SEG.tsv` or `pin.SEG.tsv`.
struct Block {
    head: Vec<String>,
    rows: Vec<Vec<String>>,
}

fn blocks(path: &Path) -> Vec<Block> {
    let mut out: Vec<Block> = Vec::new();
    for r in rows(path) {
        if r[0] == "field" {
            out.push(Block {
                head: r,
                rows: Vec::new(),
            });
        } else {
            out.last_mut().unwrap().rows.push(r);
        }
    }
    out
}

/// `GenCodecsSimpleTextCodec.walk`.
fn walk(out: &mut Vec<String>, t: &mut SimpleTextPointTree<'_, '_>, depth: usize) {
    let leaf = !t.move_to_child().unwrap();
    if !leaf {
        assert!(t.move_to_parent());
    }
    // A clone is rooted here: same cell and size, and cannot climb.
    let mut c = t.clone_tree();
    assert_eq!(c.min_packed_value(), t.min_packed_value());
    assert_eq!(c.size().unwrap(), t.size().unwrap());
    assert!(!c.move_to_parent());
    assert!(!c.move_to_sibling().unwrap());
    assert_eq!(c.is_leaf_node(), leaf);
    out.push(format!(
        "node\t{depth}\t{}\t{}\t{}\t{}",
        hex(t.min_packed_value()),
        hex(t.max_packed_value()),
        t.size().unwrap(),
        if leaf { "1" } else { "0" }
    ));
    if t.move_to_child().unwrap() {
        loop {
            walk(out, t, depth + 1);
            if !t.move_to_sibling().unwrap() {
                break;
            }
        }
        assert!(t.move_to_parent());
    }
}

/// Every point handed over, and the manifest's `all` hash.
#[derive(Default)]
struct All {
    points: Vec<Vec<u8>>,
    count: i64,
    hash: i64,
    grown: usize,
}

impl IntersectVisitor for All {
    fn compare(&mut self, _: &[u8], _: &[u8]) -> Relation {
        Relation::CellCrossesQuery
    }
    fn visit(&mut self, _: i32) {
        panic!("visit(doc) from visitDocValues");
    }
    fn visit_with_value(&mut self, doc: i32, packed: &[u8]) {
        self.points.push(packed.to_vec());
        self.count += 1;
        self.hash = self
            .hash
            .wrapping_mul(1_000_003)
            .wrapping_add(i64::from(doc));
        for &b in packed {
            self.hash = self.hash.wrapping_mul(31).wrapping_add(i64::from(b));
        }
    }
    fn grow(&mut self, count: usize) {
        self.grown += count;
    }
}

/// The manifest's `ids` hash.
#[derive(Default)]
struct Ids {
    count: i64,
    hash: i64,
    grown: Vec<usize>,
}

impl IntersectVisitor for Ids {
    fn compare(&mut self, _: &[u8], _: &[u8]) -> Relation {
        panic!("compare from visitDocIDs");
    }
    fn visit(&mut self, doc: i32) {
        self.count += 1;
        self.hash = self
            .hash
            .wrapping_mul(1_000_003)
            .wrapping_add(i64::from(doc));
    }
    fn visit_with_value(&mut self, _: i32, _: &[u8]) {
        panic!("visit(doc, value) from visitDocIDs");
    }
    fn grow(&mut self, count: usize) {
        self.grown.push(count);
    }
}

/// `GenCodecsSimpleTextCodec.BoxVisitor`.
struct BoxVisitor {
    lo: Vec<u8>,
    hi: Vec<u8>,
    idims: usize,
    bpd: usize,
    relations: String,
    visits: i64,
    value_visits: i64,
    matches: i64,
    hash: i64,
}

impl BoxVisitor {
    fn new(lo: Vec<u8>, hi: Vec<u8>, idims: usize, bpd: usize) -> Self {
        BoxVisitor {
            lo,
            hi,
            idims,
            bpd,
            relations: String::new(),
            visits: 0,
            value_visits: 0,
            matches: 0,
            hash: 0,
        }
    }
}

impl IntersectVisitor for BoxVisitor {
    fn compare(&mut self, min: &[u8], max: &[u8]) -> Relation {
        let mut crosses = false;
        for d in 0..self.idims {
            let r = d * self.bpd..(d + 1) * self.bpd;
            if min[r.clone()] > self.hi[r.clone()] || max[r.clone()] < self.lo[r.clone()] {
                self.relations.push('O');
                return Relation::CellOutsideQuery;
            }
            crosses |= min[r.clone()] < self.lo[r.clone()] || max[r.clone()] > self.hi[r];
        }
        self.relations.push(if crosses { 'C' } else { 'I' });
        if crosses {
            Relation::CellCrossesQuery
        } else {
            Relation::CellInsideQuery
        }
    }
    fn visit(&mut self, doc: i32) {
        self.visits += 1;
        self.hash = self
            .hash
            .wrapping_mul(1_000_003)
            .wrapping_add(i64::from(doc));
    }
    fn visit_with_value(&mut self, doc: i32, packed: &[u8]) {
        self.value_visits += 1;
        let m = (0..self.idims).all(|d| {
            let r = d * self.bpd..(d + 1) * self.bpd;
            packed[r.clone()] >= self.lo[r.clone()] && packed[r.clone()] <= self.hi[r]
        });
        if m {
            self.matches += 1;
        }
        self.hash = self
            .hash
            .wrapping_mul(1_000_003)
            .wrapping_add(2 * i64::from(doc) + i64::from(m));
    }
}

/// Checks one field's tree against its `points.SEG.tsv` block.
fn check_field(v: &SimpleTextBkdReader<'_>, b: &Block) {
    let h = &b.head;
    let what = unhex_str(&h[1]);
    assert_eq!(v.num_dimensions().to_string(), h[2], "{what}");
    assert_eq!(v.num_index_dimensions().to_string(), h[3], "{what}");
    assert_eq!(v.bytes_per_dimension().to_string(), h[4], "{what}");
    assert_eq!(v.size().to_string(), h[5], "{what}");
    assert_eq!(v.doc_count().to_string(), h[6], "{what}");
    assert_eq!(hex(v.min_packed_value()), h[7], "{what}");
    assert_eq!(hex(v.max_packed_value()), h[8], "{what}");
    let mut got = Vec::new();
    walk(&mut got, &mut v.point_tree(), 0);
    let mut lines = b.rows.iter().map(|r| r.join("\t"));
    for g in &got {
        assert_eq!(Some(g.clone()), lines.next(), "{what}: walk");
    }
    let mut all = All::default();
    v.point_tree().visit_doc_values(&mut all).unwrap();
    assert_eq!(
        format!("all\t{}\t{}", all.count, all.hash),
        lines.next().unwrap(),
        "{what}"
    );
    assert_eq!(all.grown, all.points.len(), "{what}: grow per leaf");
    let mut ids = Ids::default();
    v.point_tree().visit_doc_ids(&mut ids).unwrap();
    assert_eq!(
        format!("ids\t{}\t{}", ids.count, ids.hash),
        lines.next().unwrap(),
        "{what}"
    );
    assert_eq!(
        ids.grown,
        vec![usize::try_from(v.size()).unwrap()],
        "{what}: one grow"
    );
    let mut boxes = 0;
    for line in lines {
        let r: Vec<&str> = line.split('\t').collect();
        assert_eq!(r[0], "box", "{what}");
        let idims = v.num_index_dimensions();
        let bpd = v.bytes_per_dimension();
        let mut e = BoxVisitor::new(unhex(r[1]), unhex(r[2]), idims, bpd);
        let estimate = v.estimate_point_count(&mut e).unwrap();
        let mut bv = BoxVisitor::new(unhex(r[1]), unhex(r[2]), idims, bpd);
        v.intersect(&mut bv).unwrap();
        let got = format!(
            "box\t{}\t{}\t{estimate}\t{}\t{}\t{}\t{}\t{}",
            r[1], r[2], bv.relations, bv.visits, bv.value_visits, bv.matches, bv.hash
        );
        assert_eq!(got, line, "{what}");
        boxes += 1;
    }
    assert_eq!(boxes, 9, "{what}");
}

#[test]
fn reads_java_points() {
    let mut fields = 0;
    let mut multi_leaf = 0;
    for seg in all_segments() {
        let r = SimpleTextPointsReader::open(&seg.dim, &seg.dii).unwrap();
        r.check_integrity().unwrap();
        let manifest = blocks(&seg.dir.join(format!("points.{}.tsv", seg.name)));
        assert_eq!(r.field_names().count(), manifest.len(), "{}", seg.name);
        for b in &manifest {
            let name = unhex_str(&b.head[1]);
            let v = r.values(&seg.infos, &name).unwrap().unwrap();
            check_field(v, b);
            fields += 1;
            if v.leaf_block_fps().len() > 1 {
                multi_leaf += 1;
            }
        }
    }
    assert!(fields >= 20, "{fields} fields");
    assert!(multi_leaf >= 8, "{multi_leaf} multi-leaf trees");
}

#[test]
fn values_refuses_what_java_refuses() {
    let seg = all_segments().remove(0);
    let r = SimpleTextPointsReader::open(&seg.dim, &seg.dii).unwrap();
    let e = r.values(&seg.infos, "no such field").unwrap_err();
    assert!(
        matches!(&e, Error::IllegalArgument(m) if m == "field=\"no such field\" is unrecognized")
    );
    let plain = seg
        .infos
        .fields
        .iter()
        .find(|f| f.point_dimension_count == 0)
        .unwrap();
    let e = r.values(&seg.infos, &plain.name).unwrap_err();
    assert!(
        matches!(&e, Error::IllegalArgument(m) if m.ends_with("did not index points")),
        "{e}"
    );
    assert!(r.field("no such field").is_none());
}

#[test]
fn rebuilds_java_bytes() {
    for seg in all_segments() {
        let mut w = SimpleTextPointsWriter::new(&seg.name, seg.max_doc, None);
        for b in blocks(&seg.dir.join(format!("pin.{}.tsv", seg.name))) {
            let name = unhex_str(&b.head[1]);
            let fi = seg.infos.field_by_name(&name).unwrap();
            let size: u64 = b.head[2].parse().unwrap();
            let points: Vec<(Vec<u8>, i32)> = b
                .rows
                .iter()
                .map(|r| (unhex(&r[2]), r[1].parse().unwrap()))
                .collect();
            w.write_field(fi, size, points.iter().map(|(v, d)| (v.as_slice(), *d)))
                .unwrap();
        }
        let (dim, dii) = w.finish();
        assert!(dim == seg.dim, "{}: .dim differs", seg.name);
        assert!(dii == seg.dii, "{}: .dii differs", seg.name);
    }
}

/// Opens, checks integrity and walks every field fully.
fn read_all(dim: &[u8], dii: &[u8]) -> bool {
    let full = || -> Result<(), Error> {
        let r = SimpleTextPointsReader::open(dim, dii)?;
        r.check_integrity()?;
        let names: Vec<String> = r.field_names().map(str::to_string).collect();
        for name in names {
            let v = r.field(&name).unwrap();
            v.point_tree().visit_doc_values(&mut All::default())?;
            v.point_tree().visit_doc_ids(&mut Ids::default())?;
            let idims = v.num_index_dimensions();
            let bpd = v.bytes_per_dimension();
            let lo = v.min_packed_value().to_vec();
            let mut bv = BoxVisitor::new(lo.clone(), lo, idims, bpd);
            v.intersect(&mut bv)?;
            v.estimate_point_count(&mut bv)?;
        }
        Ok(())
    };
    full().is_ok()
}

fn refuses_raw_damage(data: &[u8], read: impl Fn(&[u8]) -> bool) {
    assert!(read(data), "the intact file reads");
    let mut flipped = data.to_vec();
    flipped[data.len() / 2] ^= 0x01;
    assert!(!read(&flipped), "a flipped byte");
    for cut in [1, 2, 10, 29, data.len() / 2] {
        assert!(!read(&data[..data.len() - cut]), "truncated by {cut}");
    }
    assert!(!read(&[]), "empty");
}

#[test]
fn refuses_damaged_files() {
    for seg in all_segments() {
        refuses_raw_damage(&seg.dim, |d| read_all(d, &seg.dii));
        refuses_raw_damage(&seg.dii, |d| read_all(&seg.dim, d));
    }
}

/// `seg`'s `.dim` with the first `from` after `after` replaced by `to`,
/// resealed.
fn edit(text: &str, after: &str, from: &str, to: &str) -> Vec<u8> {
    let start = text.find(after).unwrap();
    let at = start + text[start..].find(from).unwrap();
    reseal(&format!("{}{to}{}", &text[..at], &text[at + from.len()..]))
}

#[test]
fn refuses_semantic_damage() {
    let seg = all_segments().remove(0);
    let text = String::from_utf8(seg.dim.clone()).unwrap();
    let dii = &seg.dii;
    // The first field's index starts at the first `num data dims` line.
    let idx = "num data dims ";
    let block_fp = text[text.find(idx).unwrap()..]
        .lines()
        .find(|l| l.starts_with("  block fp "))
        .unwrap()
        .to_string();
    let cases: Vec<(&str, Vec<u8>)> = vec![
        (
            "a bad config",
            edit(&text, idx, "bytes per dim ", "bytes per dim 9"),
        ),
        (
            "more data dims",
            edit(&text, idx, "num data dims ", "num data dims 1"),
        ),
        (
            "one point a leaf",
            edit(&text, idx, "max leaf points 512", "max leaf points 1"),
        ),
        (
            "negative index count",
            edit(&text, idx, "index count ", "index count -"),
        ),
        (
            // Not the zero-leaf check (bkd_reader's `refuses_a_bad_index`
            // covers that): the split line no longer parses.
            "a malformed index count",
            edit(&text, idx, "index count ", "index count 0\nx"),
        ),
        (
            "a block fp past the end",
            edit(&text, idx, &block_fp, "  block fp 99999999"),
        ),
        (
            "a negative block fp",
            edit(&text, idx, &block_fp, "  block fp -1"),
        ),
        (
            "negative split count",
            edit(&text, idx, "split count ", "split count -"),
        ),
        (
            "a long min",
            edit(&text, idx, "min value [", "min value [0 "),
        ),
        (
            "a long split value",
            edit(&text, idx, "  split value [", "  split value [0 "),
        ),
        (
            "a bad line",
            edit(&text, idx, "point count ", "pint count "),
        ),
        (
            "a long value",
            edit(&text, "block count", "  block value [", "  block value [0 "),
        ),
        (
            "a negative block",
            edit(&text, "", "block count ", "block count -1\n"),
        ),
        ("a bad doc", edit(&text, "", "  doc ", "  doc x")),
    ];
    for (what, dim) in cases {
        assert!(!read_all(&dim, dii), "{what}");
    }
    // An index pointer outside the `.dim`, and a negative field count
    // (Java's loop then reads no field, but the checksum line is not next).
    let dii_text = String::from_utf8(dii.clone()).unwrap();
    let fp_line = dii_text
        .lines()
        .find(|l| l.starts_with("  field fp ") && !l.starts_with("  field fp name "))
        .unwrap();
    for (from, to) in [
        (fp_line, "  field fp 9999999999"),
        ("field count ", "field count -"),
    ] {
        let d = edit(&dii_text, "", from, to);
        assert!(!read_all(&seg.dim, &d), "{to}");
    }
}
