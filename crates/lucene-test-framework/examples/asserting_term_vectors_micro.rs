//! Paired actual asserting wrapper, inherited helpers, real bytes and full values.
use std::hint::black_box;
use std::io;
use std::time::{Duration, Instant};

use lucene_codecs::field_infos::{FieldInfo, FieldInfos};
use lucene_store::{ByteBuffersDirectory, Directory, SliceInput};
use lucene_test_framework::codecs::asserting_term_vectors::*;
use lucene_test_framework::codecs::cranky_term_vectors::*;

const N: u64 = 256;
const POS: &[u8] = &[3, 2, 0, 10, 4, 1, 0];
const REAL_POS: &[u8] = &[3, 2, 0, 10, 4, 0];
const OFF: &[u8] = &[0, 2, 1, 3, 0, 0];
#[derive(Default)]
struct Digest(u64);
impl Digest {
    fn mix(&mut self, n: i64) {
        self.0 = self.0.wrapping_mul(31).wrapping_add(n as u64);
    }
    fn bytes(&mut self, b: &[u8]) {
        self.mix(b.len() as i64);
        for &n in b {
            self.mix(i64::from(n));
        }
    }
}
impl TermVectorsWriter for Digest {
    fn start_document(&mut self, n: i32) -> io::Result<()> {
        self.mix(1);
        self.mix(i64::from(n));
        Ok(())
    }
    fn finish_document(&mut self) -> io::Result<()> {
        self.mix(2);
        Ok(())
    }
    fn start_field(&mut self, f: &FieldInfo, n: i32, p: bool, o: bool, y: bool) -> io::Result<()> {
        self.mix(3);
        self.bytes(f.name.as_bytes());
        self.mix(i64::from(f.number));
        self.mix(i64::from(n));
        self.mix(i64::from(p));
        self.mix(i64::from(o));
        self.mix(i64::from(y));
        Ok(())
    }
    fn finish_field(&mut self) -> io::Result<()> {
        self.mix(4);
        Ok(())
    }
    fn start_term(&mut self, t: &[u8], n: i32) -> io::Result<()> {
        self.mix(5);
        self.bytes(t);
        self.mix(i64::from(n));
        Ok(())
    }
    fn finish_term(&mut self) -> io::Result<()> {
        self.mix(6);
        Ok(())
    }
    fn add_position(&mut self, p: i32, s: i32, e: i32, y: Option<&[u8]>) -> io::Result<()> {
        self.mix(7);
        self.mix(i64::from(p));
        self.mix(i64::from(s));
        self.mix(i64::from(e));
        if let Some(payload) = y {
            self.bytes(payload);
        } else {
            self.mix(-1);
        }
        Ok(())
    }
    fn finish(&mut self, n: i32) -> io::Result<()> {
        self.mix(8);
        self.mix(i64::from(n));
        Ok(())
    }
    fn close(&mut self) -> io::Result<()> {
        self.mix(9);
        Ok(())
    }
    fn ram_bytes_used(&self) -> usize {
        37
    }
}
struct ProbeFormat;
impl TermVectorsFormat for ProbeFormat {
    type Writer = Digest;
    type Reader = OwnedSimpleTextReader;
    fn vectors_writer(&mut self, _: &VectorState<'_>) -> io::Result<Digest> {
        Ok(Digest::default())
    }
    fn vectors_reader(&mut self, _: &VectorState<'_>) -> io::Result<OwnedSimpleTextReader> {
        panic!("unused")
    }
}
fn document(w: &mut impl TermVectorsWriter, positions: &[u8], f: &FieldInfo) {
    w.start_document(1).unwrap();
    w.start_field(f, 1, true, true, true).unwrap();
    w.start_term(&[0, 10, 92, 255], 3).unwrap();
    w.add_prox(
        3,
        Some(&mut SliceInput::new(positions)),
        Some(&mut SliceInput::new(OFF)),
    )
    .unwrap();
    w.finish_term().unwrap();
    w.finish_field().unwrap();
    w.finish_document().unwrap();
}
fn consume(r: &impl TermVectorsReader, doc: i32, d: &mut Digest) {
    let document = r.get(doc).unwrap().unwrap_or_default();
    d.start_document(document.len() as i32).unwrap();
    for (name, field) in document {
        d.start_field(
            &FieldInfo::new(name, 7),
            field.terms.len() as i32,
            field.has_positions,
            field.has_offsets,
            field.has_payloads,
        )
        .unwrap();
        for term in field.terms {
            d.start_term(&term.term, term.freq).unwrap();
            for i in 0..term.freq as usize {
                let p = term.positions.as_ref().unwrap()[i];
                let s = term.start_offsets.as_ref().unwrap()[i];
                let e = term.end_offsets.as_ref().unwrap()[i];
                let y = term
                    .payloads
                    .as_ref()
                    .and_then(|v| v.get(i))
                    .filter(|v| !v.is_empty())
                    .map(Vec::as_slice);
                d.add_position(p, s, e, y).unwrap();
            }
            d.finish_term().unwrap();
        }
        d.finish_field().unwrap();
    }
    d.finish_document().unwrap();
}
struct Bench {
    source: ByteBuffersDirectory,
    callbacks: AssertingTermVectorsFormat<ProbeFormat>,
    simple: AssertingTermVectorsFormat<OwnedSimpleTextVectorsFormat>,
    reader: AssertingTermVectorsReader<OwnedSimpleTextReader>,
    field: FieldInfo,
    infos: FieldInfos,
}
impl Bench {
    fn new() -> Self {
        let source = ByteBuffersDirectory::new();
        let bytes =
            std::fs::read("fixtures/testframework/data/asserting_term_vectors.tvc").unwrap();
        let reader =
            AssertingTermVectorsReader::new(OwnedSimpleTextReader::open(bytes, 2).unwrap());
        Self {
            source,
            callbacks: AssertingTermVectorsFormat::new(ProbeFormat),
            simple: AssertingTermVectorsFormat::new(OwnedSimpleTextVectorsFormat),
            reader,
            field: FieldInfo::new("f", 0),
            infos: FieldInfos::new(vec![FieldInfo::new("f", 7)]).unwrap(),
        }
    }
    fn batch(&mut self, name: &str) -> i64 {
        let mut digest = Digest::default();
        let s = VectorState {
            directory: &self.source,
            segment: "_atv",
            max_doc: 2,
        };
        if name == "validation" {
            let mut w = self.callbacks.vectors_writer(&s).unwrap();
            for _ in 0..N {
                document(&mut w, POS, &self.field);
            }
            w.finish(N as i32).unwrap();
            w.close().unwrap();
            digest.mix(w.delegate().0 as i64);
        } else {
            for _ in 0..N {
                match name {
                    "merge" => {
                        let mut w = self.callbacks.vectors_writer(&s).unwrap();
                        let sources = [
                            MergeSource {
                                reader: Some(&self.reader),
                                doc_map: &[2, -1],
                            },
                            MergeSource {
                                reader: None,
                                doc_map: &[0],
                            },
                            MergeSource {
                                reader: Some(&self.reader),
                                doc_map: &[1, 3],
                            },
                        ];
                        let mut abort = || Ok(());
                        let mut state = MergeState {
                            sources: &sources,
                            field_infos: &self.infos,
                            needs_index_sort: true,
                            check_aborted: &mut abort,
                        };
                        let count = w.merge(&mut state).unwrap();
                        w.close().unwrap();
                        digest.mix(i64::from(count));
                        digest.mix(w.delegate().0 as i64);
                    }
                    "write" => {
                        let dir = ByteBuffersDirectory::new();
                        let state = VectorState {
                            directory: &dir,
                            segment: "_atv",
                            max_doc: 2,
                        };
                        let mut w = self.simple.vectors_writer(&state).unwrap();
                        document(&mut w, REAL_POS, &self.field);
                        w.start_document(0).unwrap();
                        w.finish_document().unwrap();
                        w.finish(2).unwrap();
                        w.close().unwrap();
                        digest.bytes(dir.open("_atv.tvc").unwrap().as_ref());
                    }
                    "read" => {
                        let mut r = self.reader.clone_reader().unwrap();
                        r.check_integrity().unwrap();
                        consume(&r, 0, &mut digest);
                        consume(&r, 1, &mut digest);
                        r.close().unwrap();
                    }
                    _ => unreachable!(),
                }
            }
        }
        black_box(digest.0 as i64)
    }
    fn run(&mut self, name: &str, ms: u64) -> (u64, Duration) {
        let expected = self.batch(name);
        let start = Instant::now();
        let mut batches = 0;
        loop {
            assert_eq!(black_box(self.batch(name)), expected);
            batches += 1;
            if start.elapsed() >= Duration::from_millis(ms) {
                break;
            }
        }
        (batches, start.elapsed())
    }
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let warmup = args.get(1).map_or(1000, |v| v.parse().unwrap());
    let measure = args.get(2).map_or(1500, |v| v.parse().unwrap());
    let mut b = Bench::new();
    // Only the asserting writer's shallow state, excluding embedded delegate.
    let bytes =
        std::mem::size_of::<AssertingTermVectorsWriter<Digest>>() - std::mem::size_of::<Digest>();
    println!("#memory\tasserting_term_vectors_writer_wrapper\t{bytes}");
    for name in ["validation", "merge", "write", "read"] {
        println!(
            "#check\tasserting_term_vectors_{name}\t{}\t{N}",
            b.batch(name)
        );
        b.run(name, warmup);
        let (n, t) = b.run(name, measure);
        let count = n * N;
        println!(
            "asserting_term_vectors_{name}\t{:.3}\t{count}",
            t.as_nanos() as f64 / count as f64
        );
    }
}
