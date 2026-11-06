//! OpenSearch custom-codecs' `zstd` and `zstd_no_dict` stored fields.
//!
//! `fixtures/data/opensearch_zstd/{zstd,zstd_no_dict}` were written by
//! OpenSearch 3.9.0's own `Zstd104Codec(3)` and `ZstdNoDict104Codec(6)`
//! (`scripts/zstd/ZstdStoredFields.java write`), and each `.tsv` beside them
//! is what that program's `dump` printed reading them back. This test reads
//! the same indexes and must print the same dump; then it writes the same
//! documents with [`ZstdLucene104Codec`] and must print it again.
//! `scripts/check-zstd-stored-fields.sh` runs the Java dump over the indexes
//! this test writes.

// Test-side arithmetic on generated values, not on index bytes.
#![allow(clippy::arithmetic_side_effects)]

use std::sync::Arc;

use lucene_codecs::codec::ZstdLucene104Codec;
use lucene_codecs::field_infos::{FieldInfo, IndexOptions};
use lucene_codecs::stored_fields::{FieldValue, StoredField, ZstdMode};
use lucene_index::index_writer::{
    ExplicitDocument, ExplicitFields, IndexWriter, InvertedField, InvertedTerm,
};
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_store::directory::FsDirectory;
use lucene_store::Directory;
use lucene_util::test_support::TempDir;

const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 2,
};

fn fixture(name: &str) -> String {
    format!(
        "{}/../../fixtures/data/opensearch_zstd/{name}",
        env!("CARGO_MANIFEST_DIR")
    )
}

fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The dump `ZstdStoredFields.java dump` prints.
fn dump(dir: &dyn Directory) -> String {
    let reader = DirectoryReader::open(dir).unwrap();
    let infos = lucene_index::segment_infos::read_latest(dir).unwrap();
    let mut out = String::new();
    for (seg, commit) in reader.segment_readers().iter().zip(&infos.segments) {
        assert_eq!(seg.segment_name, commit.segment_name);
        let compound = dir
            .list_all()
            .unwrap()
            .contains(&format!("{}.cfs", seg.segment_name));
        let si = dir.open(&format!("{}.si", seg.segment_name)).unwrap();
        let si = lucene_index::segment_info::parse(&si, &commit.segment_id).unwrap();
        assert_eq!(si.is_compound_file, compound);
        let mode = si
            .attributes
            .iter()
            .find(|(k, _)| k.ends_with("StoredFieldsFormat.mode"))
            .map_or("-".to_string(), |(k, v)| format!("{k}={v}"));
        out += &format!(
            "segment\t{}\t{}\t{mode}\t{}\n",
            seg.segment_name, commit.codec_name, seg.max_doc
        );
        let infos = seg.field_infos();
        for d in 0..seg.max_doc {
            let doc = seg.stored_document(d).unwrap().unwrap_or_default();
            let mut render = String::new();
            for f in &doc.fields {
                let name = &infos
                    .fields
                    .iter()
                    .find(|i| i.number == f.field_number)
                    .unwrap()
                    .name;
                let (t, v) = match &f.value {
                    FieldValue::String(s) => ("s", hex(s.as_bytes())),
                    FieldValue::Binary(b) => ("b", hex(b)),
                    FieldValue::Int(i) => ("i", i.to_string()),
                    FieldValue::Long(l) => ("l", l.to_string()),
                    FieldValue::Float(x) => ("f", format!("{:x}", x.to_bits())),
                    FieldValue::Double(x) => ("d", format!("{:x}", x.to_bits())),
                };
                render += &format!("{name}:{t}:{v};");
            }
            out += &format!(
                "{d}\t{}\t{}\t{:x}\n",
                doc.fields.len(),
                render.len(),
                fnv(render.as_bytes())
            );
        }
    }
    out
}

#[test]
fn reads_what_opensearch_wrote() {
    for name in ["zstd", "zstd_no_dict"] {
        let dir = FsDirectory::open(fixture(name));
        let want = std::fs::read_to_string(fixture(&format!("{name}.tsv"))).unwrap();
        assert_eq!(dump(&dir), want, "{name}");
    }
}

/// The documents `ZstdStoredFields.java` writes, drawn from the same LCG.
struct Docs {
    state: u64,
}

const WORDS: [&str; 21] = [
    "lucene",
    "zstd",
    "segment",
    "stored",
    "field",
    "chunk",
    "the",
    "of",
    "and",
    "index",
    "opensearch",
    "rust",
    "dictionary",
    "block",
    "frame",
    "\u{65e5}\u{672c}\u{8a9e}",
    "gr\u{fc}\u{df}e",
    "{\"k\":",
    "1234",
    "-",
    "\n",
];

impl Docs {
    fn next(&mut self, bound: u64) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.state >> 33) % bound
    }

    /// Java's `text(len)`: words until the UTF-16 length reaches `len`.
    fn text(&mut self, len: u64) -> String {
        let (mut s, mut utf16) = (String::new(), 0u64);
        while utf16 < len {
            let w = WORDS[self.next(WORDS.len() as u64) as usize];
            s += w;
            s.push(' ');
            utf16 += w.encode_utf16().count() as u64 + 1;
        }
        s
    }
}

struct Fields {
    id: i32,
    title: i32,
    body: i32,
    bin: i32,
    n: i32,
    l: i32,
    f: i32,
    d: i32,
    tag: i32,
}

fn document(docs: &mut Docs, i: u64, f: &Fields) -> ExplicitDocument {
    let mut stored = Vec::new();
    let mut add = |field_number, value| {
        stored.push(StoredField {
            field_number,
            value,
        })
    };
    if i % 113 != 0 {
        let title_len = 10 + docs.next(150);
        add(f.title, FieldValue::String(docs.text(title_len)));
        let body_len = if i == 777 { 1_300_000 } else { docs.next(3000) };
        add(f.body, FieldValue::String(docs.text(body_len)));
        let bin_len = docs.next(64);
        add(
            f.bin,
            FieldValue::Binary((0..bin_len).map(|_| docs.next(256) as u8).collect()),
        );
        add(f.n, FieldValue::Int(docs.next(1 << 30) as i32 - (1 << 29)));
        add(f.l, FieldValue::Long((docs.next(1 << 30) as i64) << 20));
        add(f.f, FieldValue::Float(docs.next(100_000) as f32 / 7.0));
        add(f.d, FieldValue::Double(docs.next(100_000) as f64 / 13.0));
        for _ in 0..i % 4 {
            add(f.tag, FieldValue::String(format!("t{}", docs.next(50))));
        }
    }
    ExplicitDocument {
        stored,
        fields: ExplicitFields {
            inverted: vec![InvertedField {
                field_number: f.id,
                terms: vec![InvertedTerm {
                    term: i.to_string().into_bytes(),
                    freq: 1,
                    positions: Vec::new(),
                    offsets: Vec::new(),
                    payloads: Vec::new(),
                }]
                .into(),
                norm: None,
            }],
            ..ExplicitFields::default()
        },
    }
}

/// Writes `ZstdStoredFields.java write`'s two segments with `mode`, then
/// returns the dump.
fn write(dir: &FsDirectory, mode: ZstdMode) -> String {
    let codec = ZstdLucene104Codec::new(mode).unwrap();
    let mut w = IndexWriter::open(dir, Vec::new(), "Lucene104", VERSION).unwrap();
    w.set_codec(Some(Arc::new(codec))).unwrap();
    w.enable_explicit_documents().unwrap();
    let mut reg = |name: &str, indexed: bool| {
        let info = FieldInfo::new(name, 0);
        let info = if indexed {
            info.with_index_options(IndexOptions::Docs)
                .with_omit_norms(true)
        } else {
            info
        };
        w.register_field(info).unwrap()
    };
    let f = Fields {
        id: reg("id", true),
        title: reg("title", false),
        body: reg("body", false),
        bin: reg("bin", false),
        n: reg("n", false),
        l: reg("l", false),
        f: reg("f", false),
        d: reg("d", false),
        tag: reg("tag", false),
    };
    let mut docs = Docs { state: 42 };
    for (compound, range) in [(false, 0..1200), (true, 1200..1700)] {
        w.set_use_compound_file(compound);
        let batch = range.map(|i| document(&mut docs, i, &f)).collect();
        w.add_explicit_documents(batch).unwrap();
        w.commit().unwrap();
    }
    dump(dir)
}

/// Copies the index in `from` to `$ZSTD_WRITER_OUT/<name>` when set, for
/// `scripts/check-zstd-stored-fields.sh`.
fn copy_out(from: &std::path::Path, name: &str) {
    if let Ok(out) = std::env::var("ZSTD_WRITER_OUT") {
        let to = std::path::Path::new(&out).join(name);
        std::fs::create_dir_all(&to).unwrap();
        for e in std::fs::read_dir(from).unwrap() {
            let e = e.unwrap();
            if e.file_name() != "write.lock" {
                std::fs::copy(e.path(), to.join(e.file_name())).unwrap();
            }
        }
    }
}

#[test]
fn writes_what_opensearch_reads() {
    for (name, mode) in [
        ("zstd", ZstdMode::zstd(3)),
        ("zstd_no_dict", ZstdMode::zstd_no_dict(6)),
    ] {
        let tmp = TempDir::new("opensearch-zstd");
        let dir = FsDirectory::open(tmp.path());
        let want = std::fs::read_to_string(fixture(&format!("{name}.tsv"))).unwrap();
        assert_eq!(write(&dir, mode), want, "{name}");
        copy_out(tmp.path(), name);

        // A merge recompresses or bulk-copies into one zstd segment.
        let codec = ZstdLucene104Codec::new(mode).unwrap();
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
        w.set_codec(Some(Arc::new(codec))).unwrap();
        w.force_merge(1).unwrap();
        w.commit().unwrap();
        drop(w);
        let merged = dump(&dir);
        let mut lines = merged.lines();
        let header = lines.next().unwrap();
        assert!(
            header.contains(&format!("\t{}\t", codec_name(mode))),
            "{header}"
        );
        let docs: Vec<&str> = lines.collect();
        let original: Vec<&str> = want.lines().filter(|l| !l.starts_with("segment")).collect();
        assert_eq!(docs.len(), original.len());
        for (a, b) in docs.iter().zip(&original) {
            // Merged doc ids are renumbered; the contents stay.
            assert_eq!(a.split_once('\t').unwrap().1, b.split_once('\t').unwrap().1);
        }
        copy_out(tmp.path(), &format!("{name}_merged"));
    }
}

fn codec_name(mode: ZstdMode) -> &'static str {
    if mode.dictionary {
        "ZSTD104"
    } else {
        "ZSTDNODICT104"
    }
}
