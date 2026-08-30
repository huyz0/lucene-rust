//! What a `StoredFieldVisitor` saves over materialising the whole
//! `Document` (ledger item 24).
//!
//! `StoredFieldsReader::document` decodes and allocates every field of a
//! document. Java hands a `StoredFieldVisitor` one field at a time and calls
//! `skipField` for the ones it says `NO` to, so retrieving one field of a
//! wide document costs that field plus a length vint per other field --
//! never a `String`/`Vec` per other field.
//!
//! Both arms run over the same `.fdt`, alternating, in one process, and every
//! figure is a **min of N repetitions** -- criterion reported 83/91/129 µs
//! for identical code on this host (`docs/sweep/m2/c24-arith-codecs.md`).
//!
//! The documents are built here rather than read from a corpus because the
//! benchmark corpus stores almost nothing (`GenCorpus` indexes every field
//! `Store.NO`), and the point of this measurement is *width*: how the cost of
//! one field scales with how many others the document carries.
//!
//! ```text
//! cargo run --release -p lucene-codecs --example stored_fields_visitor
//! ```
// A measurement harness's own arithmetic. See `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::hint::black_box;
use std::time::Instant;

use lucene_codecs::stored_fields::{
    self, Document, DocumentVisitor, FieldValue, StoredField, StoredFieldsReader,
};
use lucene_store::codec_util::ID_LENGTH;

const SEG_ID: [u8; ID_LENGTH] = [3u8; ID_LENGTH];
const NUM_DOCS: i32 = 4096;
/// Read with a stride so consecutive reads land in different chunks, which is
/// what a top-`k` result set looks like.
const STRIDE: i32 = 401;

/// `width` string fields per document, each a realistic sentence.
fn corpus(width: i32) -> Vec<Document> {
    (0..NUM_DOCS)
        .map(|doc| Document {
            fields: (0..width)
                .map(|f| StoredField {
                    field_number: f,
                    value: FieldValue::String(format!(
                        "field {f} of document {doc}: a stored value long enough to cost a real \
                         allocation when it is decoded"
                    )),
                })
                .collect(),
        })
        .collect()
}

fn whole_document(reader: &StoredFieldsReader<'_>, doc: i32) -> u64 {
    reader.document(doc).expect("document").fields.len() as u64
}

fn one_field(reader: &StoredFieldsReader<'_>, doc: i32, field: i32) -> u64 {
    let mut visitor = DocumentVisitor::for_fields(&[field]);
    reader.visit_document(doc, &mut visitor).expect("visit");
    visitor.document().fields.len() as u64
}

fn run(width: i32, reps: usize) {
    let docs = corpus(width);
    let (fdt, fdx, fdm) = stored_fields::write_best_speed(&docs, &SEG_ID, "");
    let reader = stored_fields::open(&fdt, &fdx, &fdm, &SEG_ID, "").expect("open");

    // The last field, so the visitor's `NO` path has to skip every other
    // field of the document to reach it -- the worst case for the visitor and
    // the one that matters, since the "don't skipField on the last field"
    // shortcut only helps when the *unwanted* field is last.
    let wanted = width - 1;

    let mut whole = u128::MAX;
    let mut single = u128::MAX;
    let mut sink = 0u64;
    for _ in 0..reps {
        let t = Instant::now();
        let mut n = 0u64;
        let mut doc = 0i32;
        for _ in 0..NUM_DOCS {
            doc = (doc + STRIDE) % NUM_DOCS;
            n += whole_document(&reader, doc);
        }
        whole = whole.min(t.elapsed().as_nanos());
        sink ^= n;

        let t = Instant::now();
        let mut n = 0u64;
        let mut doc = 0i32;
        for _ in 0..NUM_DOCS {
            doc = (doc + STRIDE) % NUM_DOCS;
            n += one_field(&reader, doc, wanted);
        }
        single = single.min(t.elapsed().as_nanos());
        sink ^= n;
    }
    black_box(sink);

    let per = |ns: u128| ns as f64 / NUM_DOCS as f64;
    println!(
        "{width:>3} fields/doc   document(): {:>8.3} us/doc   visit one field: {:>8.3} us/doc   \
         {:.2}x",
        per(whole) / 1000.0,
        per(single) / 1000.0,
        whole as f64 / single.max(1) as f64
    );
}

fn main() {
    let reps: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(40);
    println!("{NUM_DOCS} documents, one field of each retrieved, min of {reps}");
    for width in [1, 4, 16, 64] {
        run(width, reps);
    }
}
