//! What one large stored value costs a reader that does not want it.
//!
//! A document of `_id` (16 bytes) then `_source` (`MB` MiB of JSON-ish
//! text), the shape an OpenSearch document has. Each arm reads the
//! document a few times and reports the minimum:
//!
//! - `id only`: a visitor that takes the first field and stops, which is
//!   what a `_source: false` fetch asks for;
//! - `skip source`: takes the first field and says `NO` to the rest;
//! - `whole`: copies every value out, as a full fetch does.
//!
//! ```text
//! MB=100 cargo run --release -p lucene-codecs --example stored_large_field
//! ```
// A measurement harness's own arithmetic. See `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::hint::black_box;
use std::time::Instant;

use lucene_codecs::stored_fields::{
    self, Document, FieldValue, Mode, StoredField, StoredFieldVisitor, StoredFieldsWriter,
    VisitStatus,
};
use lucene_store::codec_util::ID_LENGTH;

const SEG_ID: [u8; ID_LENGTH] = [5u8; ID_LENGTH];

struct Take {
    stop_after_first: bool,
    seen: usize,
    bytes: usize,
}

impl StoredFieldVisitor for Take {
    fn needs_field(&mut self, n: i32) -> stored_fields::Result<VisitStatus> {
        Ok(match (n, self.stop_after_first, self.seen) {
            (0, _, _) => VisitStatus::Yes,
            (_, true, _) => VisitStatus::Stop,
            (_, false, _) => VisitStatus::No,
        })
    }
    fn binary_field(&mut self, _: i32, v: &[u8]) -> stored_fields::Result<()> {
        self.seen += 1;
        self.bytes += v.len();
        Ok(())
    }
}

struct Copy(Vec<u8>);

impl StoredFieldVisitor for Copy {
    fn needs_field(&mut self, _: i32) -> stored_fields::Result<VisitStatus> {
        Ok(VisitStatus::Yes)
    }
    fn binary_field(&mut self, _: i32, v: &[u8]) -> stored_fields::Result<()> {
        self.0.extend_from_slice(v);
        Ok(())
    }
}

fn source(mb: usize) -> Vec<u8> {
    let mut s = Vec::with_capacity(mb << 20);
    let mut x = 0x9e37_79b9_7f4a_7c15u64;
    s.extend_from_slice(b"{\"body\":\"");
    while s.len() < (mb << 20) - 2 {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        let word = [
            "alpha", "beta", "gamma", "delta", "error", "warn", "info", "trace",
        ][(x & 7) as usize];
        s.extend_from_slice(word.as_bytes());
        s.extend_from_slice(format!(" {} ", x % 100_000).as_bytes());
    }
    s.truncate((mb << 20) - 2);
    s.extend_from_slice(b"\"}");
    s
}

fn time(reps: usize, mut f: impl FnMut() -> usize) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..reps {
        let t = Instant::now();
        black_box(f());
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    best
}

/// `SMALL=1`: many 2 KB documents, the common case, read at random.
fn small() {
    let src = source(1);
    for mode in [Mode::BestSpeed, Mode::BestCompression, Mode::ZstdPaged(3)] {
        let mut w = StoredFieldsWriter::new(mode, &SEG_ID, "");
        let n = 20_000;
        for i in 0..n {
            let at = (i * 37) % ((1 << 20) - 2048);
            w.add_document(&Document {
                fields: vec![
                    StoredField {
                        field_number: 0,
                        value: FieldValue::Binary(format!("id-{i:012}").into_bytes()),
                    },
                    StoredField {
                        field_number: 1,
                        value: FieldValue::Binary(src[at..at + 2000].to_vec()),
                    },
                    StoredField {
                        field_number: 2,
                        value: FieldValue::Long(i as i64),
                    },
                ],
            });
        }
        let (fdt, fdx, fdm) = w.finish();
        let r = stored_fields::open(&fdt, &fdx, &fdm, &SEG_ID, "").expect("open");
        let reads = 20_000;
        let mut best_all = f64::MAX;
        let mut best_id = f64::MAX;
        for _ in 0..7 {
            let mut x = 12345u64;
            let t = Instant::now();
            for _ in 0..reads {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let mut v = Copy(Vec::new());
                r.visit_document(((x >> 33) % n as u64) as i32, &mut v)
                    .unwrap();
                black_box(v.0.len());
            }
            best_all = best_all.min(t.elapsed().as_secs_f64() * 1e9 / reads as f64);
            let t = Instant::now();
            for _ in 0..reads {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let mut v = Take {
                    stop_after_first: true,
                    seen: 0,
                    bytes: 0,
                };
                r.visit_document(((x >> 33) % n as u64) as i32, &mut v)
                    .unwrap();
                black_box(v.bytes);
            }
            best_id = best_id.min(t.elapsed().as_secs_f64() * 1e9 / reads as f64);
        }
        println!(
            "small {mode:?}: whole {:.0} ns/doc | id only {:.0} ns/doc",
            best_all, best_id
        );
    }
}

fn main() {
    if std::env::var_os("SMALL").is_some() {
        return small();
    }
    let mb: usize = std::env::var("MB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100);
    let src = source(mb);
    for mode in [Mode::BestSpeed, Mode::BestCompression, Mode::ZstdPaged(3)] {
        let mut w = StoredFieldsWriter::new(mode, &SEG_ID, "");
        for i in 0..3 {
            w.add_document(&Document {
                fields: vec![
                    StoredField {
                        field_number: 0,
                        value: FieldValue::Binary(format!("id-{i:012}").into_bytes()),
                    },
                    StoredField {
                        field_number: 1,
                        value: FieldValue::Binary(src.clone()),
                    },
                ],
            });
        }
        let (fdt, fdx, fdm) = w.finish();
        let r = stored_fields::open(&fdt, &fdx, &fdm, &SEG_ID, "").expect("open");
        let reps = 5;
        let serialized = time(reps, || r.serialized_document(1).unwrap().bytes.len());
        let id_only = time(reps, || {
            let mut v = Take {
                stop_after_first: true,
                seen: 0,
                bytes: 0,
            };
            r.visit_document(1, &mut v).unwrap();
            v.bytes
        });
        let skip = time(reps, || {
            let mut v = Take {
                stop_after_first: false,
                seen: 0,
                bytes: 0,
            };
            r.visit_document(1, &mut v).unwrap();
            v.bytes
        });
        let whole = time(reps, || {
            let mut v = Copy(Vec::new());
            r.visit_document(1, &mut v).unwrap();
            v.0.len()
        });
        println!(
            "{mode:?}: fdt {} MiB | decompress doc {serialized:.1} ms | id only {id_only:.2} ms | \
             skip source {skip:.2} ms | whole copy {whole:.1} ms",
            fdt.len() >> 20
        );
    }
}
