//! lucene-codecs' SimpleText vectors (`.vec`/`.gri`) against Java:
//! `GenCodecsSimpleTextCodec`'s two indexes (flushed, and force-merged into
//! a compound file) record what Java's reader makes of every vector field
//! (`knn.SEG.tsv`): each ord's doc and vector, then searches through
//! `TopKnnCollector` with and without accept bits and visit limits, their
//! hits as doc and score bits.
//!
//! Here the Rust reader opens Java's files and agrees with every manifest
//! bit for bit, the Rust writer rebuilds both files byte for byte from the
//! manifests' vectors, and damaged copies of Java's bytes are refused.

// Test-support arithmetic over small in-memory counters; see
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::fs;
use std::path::{Path, PathBuf};

use lucene_codecs::field_infos::{FieldInfos, VectorEncoding};
use lucene_codecs::hnsw::KnnCollector;
use lucene_codecs::simpletext::compound::{self, SimpleTextCompoundReader};
use lucene_codecs::simpletext::field_infos as st_infos;
use lucene_codecs::simpletext::knn_vectors_reader::SimpleTextKnnVectorsReader;
use lucene_codecs::simpletext::knn_vectors_writer::{
    meta_file_name, vector_data_file_name, SimpleTextKnnVectorsWriter,
};
use lucene_codecs::simpletext::{util, Error};
use lucene_util::bits::Bits;
use lucene_util::fixed_bit_set::FixedBitSet;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/codecs_simpletext/codec")
}

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

fn floats(s: &str) -> Vec<f32> {
    (0..s.len())
        .step_by(8)
        .map(|i| f32::from_bits(u32::from_str_radix(&s[i..i + 8], 16).unwrap()))
        .collect()
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

/// A segment with vectors: its name, maxDoc, field infos, `.vec`, `.gri`,
/// live docs and its manifests' directory.
struct Seg {
    name: String,
    max_doc: usize,
    infos: FieldInfos,
    vec: Vec<u8>,
    gri: Vec<u8>,
    live: Option<FixedBitSet>,
    dir: PathBuf,
}

fn segments(case: &str) -> Vec<Seg> {
    let dir = root().join(case);
    let index = dir.join("index");
    let mut out = Vec::new();
    for r in rows(&dir.join("segments.tsv")) {
        let name = r[0].clone();
        if !dir.join(format!("knn.{name}.tsv")).exists() {
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
        let live_path = dir.join(format!("live.{name}.tsv"));
        let live = live_path.exists().then(|| {
            let r = rows(&live_path);
            let mut bits = FixedBitSet::new(r[0][0].parse().unwrap());
            for (d, c) in r[1][0].chars().enumerate() {
                if c == '1' {
                    bits.set(d);
                }
            }
            bits
        });
        out.push(Seg {
            max_doc: r[1].parse().unwrap(),
            infos,
            vec: core(&vector_data_file_name(&name, "")),
            gri: core(&meta_file_name(&name, "")),
            live,
            dir: dir.clone(),
            name,
        });
    }
    out
}

fn all_segments() -> Vec<Seg> {
    let segs: Vec<Seg> = ["flush", "merged"]
        .iter()
        .flat_map(|c| segments(c))
        .collect();
    assert_eq!(segs.len(), 3, "flush _0/_1 and the merged segment");
    segs
}

/// One field's block of `knn.SEG.tsv`.
struct Block {
    head: Vec<String>,
    rows: Vec<Vec<String>>,
}

fn blocks(seg: &Seg) -> Vec<Block> {
    let mut out: Vec<Block> = Vec::new();
    for r in rows(&seg.dir.join(format!("knn.{}.tsv", seg.name))) {
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

fn field_name(b: &Block) -> String {
    String::from_utf8(unhex(&b.head[1])).unwrap()
}

/// `(doc, scoreBits)`s, the visited count and whether it terminated early.
type Hits = (Vec<(i32, u32)>, u64, bool);

fn search(r: &SimpleTextKnnVectorsReader<'_>, seg: &Seg, b: &Block, q: &[String]) -> Hits {
    let name = field_name(b);
    let k: usize = q[1].parse().unwrap();
    let limit: u64 = q[2].parse().unwrap();
    let mut odd = FixedBitSet::new(seg.max_doc);
    for d in (1..seg.max_doc).step_by(2) {
        odd.set(d);
    }
    let bits: Option<&dyn Bits> = match q[3].as_str() {
        "none" => None,
        "live" => seg.live.as_ref().map(|b| b as &dyn Bits),
        "odd" => Some(&odd),
        other => panic!("{other}"),
    };
    let mut c = KnnCollector::new(k, limit);
    if b.head[2] == "BYTE" {
        r.search(&name, &unhex(&q[4]), &mut c, bits).unwrap();
    } else {
        r.search(&name, &floats(&q[4]), &mut c, bits).unwrap();
    }
    let (visited, early) = (c.visited_count(), c.early_terminated());
    let hits = c
        .top_docs()
        .into_iter()
        .map(|(d, s)| (d, s.to_bits()))
        .collect();
    (hits, visited, early)
}

/// Every `Float.toString` shape of the generator's `TEXT_FLOATS` reaches a
/// Java `.vec`, so none of the table is dead.
#[test]
fn java_vectors_hold_every_float_text_shape() {
    let text: String = all_segments()
        .iter()
        .map(|seg| String::from_utf8_lossy(&seg.vec).into_owned())
        .collect();
    for shape in [
        "-0.0,",
        "1.4E-45",
        "1.1754944E-38",
        "1.0E7",
        "9999999.0",
        "0.001,",
        "9.9999E-4",
        "1.0E-4",
        "-1.5,",
        "1.0E10",
        "100.0",
        "8.5E-44",
        "123456.79",
    ] {
        assert!(text.contains(shape), "no {shape} in Java's .vec");
    }
}

#[test]
fn reads_java_vectors_and_searches_bit_for_bit() {
    let (mut fields, mut queries, mut early_queries) = (0, 0, 0);
    for seg in all_segments() {
        let r = SimpleTextKnnVectorsReader::open(&seg.gri, &seg.vec, &seg.infos).unwrap();
        r.check_integrity().unwrap();
        for b in blocks(&seg) {
            let name = field_name(&b);
            let fi = seg.infos.field_by_name(&name).unwrap();
            let encoding = match fi.vector_encoding {
                VectorEncoding::Byte => "BYTE",
                VectorEncoding::Float32 => "FLOAT32",
            };
            assert_eq!(b.head[2], encoding, "{name}");
            assert_eq!(
                b.head[3],
                st_infos::similarity_name(fi.vector_similarity_function)
            );
            let (dim, size): (i32, usize) =
                (b.head[4].parse().unwrap(), b.head[5].parse().unwrap());
            let mut want = Vec::new();
            for row in b.rows.iter().filter(|r| r[0] == "v") {
                want.push((row[2].parse::<i32>().unwrap(), row[3].clone()));
            }
            assert_eq!(want.len(), size);
            if encoding == "BYTE" {
                let v = r.byte_vector_values(&name).unwrap().unwrap();
                assert_eq!((v.dimension(), v.size()), (dim, size));
                for (ord, (doc, hex)) in want.iter().enumerate() {
                    assert_eq!(v.ord_to_doc(ord), Some(*doc));
                    assert_eq!(v.vector_value(ord).unwrap(), unhex(hex), "{name} {ord}");
                }
            } else {
                let v = r.float_vector_values(&name).unwrap().unwrap();
                assert_eq!((v.dimension(), v.size()), (dim, size));
                for (ord, (doc, hex)) in want.iter().enumerate() {
                    assert_eq!(v.ord_to_doc(ord), Some(*doc));
                    let got: Vec<u32> = v
                        .vector_value(ord)
                        .unwrap()
                        .iter()
                        .map(|f| f.to_bits())
                        .collect();
                    let want: Vec<u32> = floats(hex).iter().map(|f| f.to_bits()).collect();
                    assert_eq!(got, want, "{name} {ord}");
                }
            }
            // The queries: each "q" row, its "hit" rows, its "visited" row.
            let mut i = 0;
            let qrows: Vec<&Vec<String>> = b.rows.iter().filter(|r| r[0] != "v").collect();
            while i < qrows.len() {
                let q = qrows[i];
                assert_eq!(q[0], "q");
                i += 1;
                let mut want_hits = Vec::new();
                while qrows[i][0] == "hit" {
                    let h = qrows[i];
                    want_hits.push((
                        h[1].parse().unwrap(),
                        u32::from_str_radix(&h[2], 16).unwrap(),
                    ));
                    i += 1;
                }
                let visited = qrows[i];
                i += 1;
                let (hits, count, early) = search(&r, &seg, &b, q);
                assert_eq!(hits, want_hits, "{}/{name}: {q:?}", seg.name);
                assert_eq!(count.to_string(), visited[1], "{name}: {q:?}");
                assert_eq!(early, visited[2] == "GREATER_THAN_OR_EQUAL_TO");
                queries += 1;
                early_queries += usize::from(early);
            }
            fields += 1;
        }
        // A field the field infos know with no entry in this segment.
        for fi in seg.infos.fields.iter().filter(|f| f.vector_dimension > 0) {
            if r.field_entry(fi.number).is_none() {
                assert!(r.float_vector_values(&fi.name).unwrap().is_none());
            }
        }
    }
    assert!(fields >= 30, "{fields} fields");
    assert!(
        queries >= 180 && early_queries >= 30,
        "{queries} {early_queries}"
    );
}

#[test]
fn rebuilds_java_bytes() {
    for seg in all_segments() {
        let mut w = SimpleTextKnnVectorsWriter::new();
        for b in blocks(&seg) {
            let fi = seg.infos.field_by_name(&field_name(&b)).unwrap();
            let v: Vec<(i32, &String)> = b
                .rows
                .iter()
                .filter(|r| r[0] == "v")
                .map(|r| (r[2].parse().unwrap(), &r[3]))
                .collect();
            if b.head[2] == "BYTE" {
                let vs: Vec<(i32, Vec<u8>)> = v.iter().map(|(d, h)| (*d, unhex(h))).collect();
                w.write_byte_field(fi, vs.iter().map(|(d, x)| (*d, x.as_slice())))
                    .unwrap();
            } else {
                let vs: Vec<(i32, Vec<f32>)> = v.iter().map(|(d, h)| (*d, floats(h))).collect();
                w.write_float_field(fi, vs.iter().map(|(d, x)| (*d, x.as_slice())))
                    .unwrap();
            }
        }
        let (vec, gri) = w.finish();
        assert!(vec == seg.vec, "{}: .vec differs", seg.name);
        assert!(gri == seg.gri, "{}: .gri differs", seg.name);
    }
}

/// Opens, checks integrity and reads every listed field's vectors.
fn read_all(seg: &Seg, gri: &[u8], vec: &[u8]) -> bool {
    let full = || -> Result<(), Error> {
        let r = SimpleTextKnnVectorsReader::open(gri, vec, &seg.infos)?;
        r.check_integrity()?;
        for fi in seg.infos.fields.iter().filter(|f| f.vector_dimension > 0) {
            if fi.vector_encoding == VectorEncoding::Byte {
                r.byte_vector_values(&fi.name)?;
            } else {
                r.float_vector_values(&fi.name)?;
            }
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
        refuses_raw_damage(&seg.vec, |d| read_all(&seg, &seg.gri, d));
        refuses_raw_damage(&seg.gri, |d| read_all(&seg, d, &seg.vec));
    }
}

/// `text` with the first `from` replaced by `to`, resealed.
fn edit(text: &str, from: &str, to: &str) -> Vec<u8> {
    assert!(text.contains(from), "{from}");
    reseal(&text.replacen(from, to, 1))
}

#[test]
fn refuses_semantic_damage() {
    let seg = all_segments().remove(0);
    let gri = String::from_utf8(seg.gri.clone()).unwrap();
    let vec = String::from_utf8(seg.vec.clone()).unwrap();
    let first_vector = vec.lines().next().unwrap().to_string();
    let gri_cases = [
        ("vector-dimension 4", "vector-dimension 5"),
        ("vector-data-offset 0", "vector-data-offset -1"),
        ("vector-data-length ", "vector-data-length 99999999"),
        ("size 144", "size -144"),
        ("size 144", "size 145"),
        ("field-name fvec", "field-name nope"),
        ("field-number 28", "field-number 27"),
        ("field-number 28", "field-numbr 28"),
        ("\n7\n", "\nseven\n"),
    ];
    for (from, to) in gri_cases {
        assert!(!read_all(&seg, &edit(&gri, from, to), &seg.vec), "{to}");
    }
    // Vector lines: one value short, one over, one that is not a float.
    let short = first_vector.rsplit_once(',').unwrap().0.to_string() + "]";
    let long = first_vector.replace(']', ", 1.0]");
    let not_float = first_vector.replacen('.', "x", 1);
    for to in [short, long, not_float] {
        let v = edit(&vec, &first_vector, &to);
        // Keep every offset: the edited line's length change moves only the
        // first field's length and later offsets, so test the parse alone.
        let delta = to.len() as i64 - first_vector.len() as i64;
        let mut g = gri.clone();
        let len_line = g
            .lines()
            .find(|l| l.starts_with("vector-data-length "))
            .unwrap()
            .to_string();
        let len: i64 = len_line["vector-data-length ".len()..].parse().unwrap();
        g = g.replacen(&len_line, &format!("vector-data-length {}", len + delta), 1);
        let shifted: Vec<String> = g
            .lines()
            .map(|l| match l.strip_prefix("vector-data-offset ") {
                Some(o) if o != "0" => {
                    format!("vector-data-offset {}", o.parse::<i64>().unwrap() + delta)
                }
                _ => l.to_string(),
            })
            .collect();
        let g = reseal(&(shifted.join("\n") + "\n"));
        assert!(!read_all(&seg, &g, &v), "{to}");
    }
}
