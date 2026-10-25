//! Paired with CrankyTermVectorsMicro: actual callbacks, real .tvc writes/reads.
use lucene_codecs::field_infos::{FieldInfo, FieldInfos};
use lucene_store::{ByteBuffersDirectory, Directory, SliceInput};
use lucene_test_framework::codecs::cranky_term_vectors::*;
use lucene_util::java_random::JavaRandom;
use std::cell::RefCell;
use std::hint::black_box;
use std::io;
use std::rc::Rc;
use std::time::{Duration, Instant};
const POS: &[u8] = &[3, 2, 0, 10, 4, 1, 0];
const OFF: &[u8] = &[0, 2, 1, 3, 0, 0];
const N: u64 = 256;
#[derive(Default)]
struct Digest(u64);
impl Digest {
    fn mix(&mut self, n: i64) {
        self.0 = self.0.wrapping_mul(31).wrapping_add(n as u64);
    }
    fn bytes(&mut self, b: &[u8]) {
        self.mix(b.len() as i64);
        for &b in b {
            self.mix(i64::from(b));
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
        if let Some(y) = y {
            self.bytes(y);
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
        0
    }
}
fn consume(reader: &SimpleTextVectorsReader, doc: i32, d: &mut Digest) {
    let fields = reader.get(doc).unwrap().unwrap_or_default();
    d.start_document(fields.len() as i32).unwrap();
    for (name, f) in fields {
        d.start_field(
            &FieldInfo::new(name, 7),
            f.terms.len() as i32,
            f.has_positions,
            f.has_offsets,
            f.has_payloads,
        )
        .unwrap();
        for t in f.terms {
            d.start_term(&t.term, t.freq).unwrap();
            if f.has_positions || f.has_offsets {
                for i in 0..t.freq as usize {
                    d.add_position(
                        t.positions.as_ref().map(|v| v[i]).unwrap_or(-1),
                        t.start_offsets.as_ref().map(|v| v[i]).unwrap_or(-1),
                        t.end_offsets.as_ref().map(|v| v[i]).unwrap_or(-1),
                        t.payloads
                            .as_ref()
                            .map(|v| v[i].as_slice())
                            .filter(|p| !p.is_empty()),
                    )
                    .unwrap();
                }
            }
            d.finish_term().unwrap();
        }
        d.finish_field().unwrap();
    }
    d.finish_document().unwrap();
}
struct Bench {
    random: Rc<RefCell<JavaRandom>>,
    guard: CrankyTermVectorsWriter<Digest>,
    format: CrankyTermVectorsFormat<SimpleTextVectorsFormat>,
    reader: SimpleTextVectorsReader,
    infos: FieldInfos,
    field: FieldInfo,
}
impl Bench {
    fn new() -> Self {
        let random = Rc::new(RefCell::new(JavaRandom::new(42)));
        let bytes = std::fs::read("fixtures/testframework/data/cranky_term_vectors.tvc").unwrap();
        Self {
            guard: CrankyTermVectorsWriter::new(Digest::default(), random.clone()),
            format: CrankyTermVectorsFormat::new(SimpleTextVectorsFormat, random.clone()),
            random,
            reader: SimpleTextVectorsReader::open(bytes, 2).unwrap(),
            infos: FieldInfos::new(vec![FieldInfo::new("f", 7)]).unwrap(),
            field: FieldInfo::new("f", 0),
        }
    }
    fn batch(&mut self, case: &str) -> i64 {
        self.random.borrow_mut().set_seed(42);
        self.guard.delegate_mut().0 = 0;
        let mut digest = Digest::default();
        for _ in 0..N {
            match case {
                "prox" => {
                    if let Err(e) = self.guard.add_prox(
                        3,
                        Some(&mut SliceInput::new(POS)),
                        Some(&mut SliceInput::new(OFF)),
                    ) {
                        assert!(e.to_string().starts_with("Fake IOException"));
                        self.guard.delegate_mut().mix(99);
                    }
                }
                "merge" => {
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
                    match self.guard.merge(&mut state) {
                        Ok(n) => self.guard.delegate_mut().mix(i64::from(n)),
                        Err(e) => {
                            assert!(e.to_string().starts_with("Fake IOException"));
                            self.guard.delegate_mut().mix(99);
                        }
                    }
                }
                "write" => {
                    self.random.borrow_mut().set_seed(42);
                    let dir = ByteBuffersDirectory::new();
                    let state = VectorState {
                        directory: &dir,
                        segment: "_ctv",
                        max_doc: 2,
                    };
                    let mut w = self.format.vectors_writer(&state).unwrap();
                    w.start_document(1).unwrap();
                    w.start_field(&self.field, 1, true, true, true).unwrap();
                    w.start_term(&[0, 10, 92, 255], 3).unwrap();
                    w.add_prox(
                        3,
                        Some(&mut SliceInput::new(POS)),
                        Some(&mut SliceInput::new(OFF)),
                    )
                    .unwrap();
                    w.finish_term().unwrap();
                    w.finish_field().unwrap();
                    w.finish_document().unwrap();
                    w.start_document(0).unwrap();
                    w.finish_document().unwrap();
                    w.finish(2).unwrap();
                    w.close().unwrap();
                    digest.bytes(dir.open("_ctv.tvc").unwrap().as_ref());
                }
                "read" => {
                    consume(&self.reader, 0, &mut digest);
                    consume(&self.reader, 1, &mut digest);
                }
                _ => unreachable!(),
            }
        }
        if case == "prox" || case == "merge" {
            digest.0 = self.guard.delegate().0;
        }
        digest.mix(i64::from(self.random.borrow_mut().next_int()));
        digest.mix(self.random.borrow_mut().next_long());
        black_box(digest.0 as i64)
    }
    fn run(&mut self, case: &str, ms: u64) -> (u64, Duration) {
        let expected = self.batch(case);
        let start = Instant::now();
        let mut batches = 0;
        while start.elapsed() < Duration::from_millis(ms) {
            assert_eq!(black_box(self.batch(case)), expected);
            batches += 1;
        }
        (batches, start.elapsed())
    }
}
fn main() {
    let mut b = Bench::new();
    // Format + writer wrapper overhead and shared Rc<RefCell<JavaRandom>> once;
    // excludes delegate storage, inputs, decoded documents and I/O transients.
    let bytes = std::mem::size_of_val(&b.format) + std::mem::size_of_val(&b.guard)
        - std::mem::size_of::<Digest>()
        + 2 * std::mem::size_of::<usize>()
        + std::mem::size_of::<RefCell<JavaRandom>>();
    println!("#memory\tcranky_term_vectors_wrappers_rng\t{bytes}");
    for case in ["prox", "merge", "write", "read"] {
        let name = format!("cranky_term_vectors_{case}");
        println!("#check\t{name}\t{}\t{N}", b.batch(case));
        b.run(case, 1000);
        let (n, time) = b.run(case, 1500);
        let operations = n * N;
        println!(
            "{name}\t{:.3}\t{operations}",
            time.as_nanos() as f64 / operations as f64
        );
    }
}
