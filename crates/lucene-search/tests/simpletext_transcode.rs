//! `simpletext_transcode::to_simpletext`: every segment IndexWriter wrote
//! (flushed, compound and not, with deletions; then merged) re-encoded as
//! SimpleText, under a `segments_N` naming the SimpleText codec, reads back
//! through [`DirectoryReader`]'s SimpleText path exactly as the native
//! segment does: stored fields, postings with positions, offsets and
//! payloads, norms, all five doc-values types, term vectors, points, float
//! and byte vectors and live docs.
// Test-support code opts out of the arithmetic gate at the file boundary:
// see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::fmt::Write as _;

use lucene_index::buffered_updates::Term;
use lucene_index::document::{
    BinaryDocValuesField, Document, Field, FieldToken, FieldTokens, FieldType, IndexOptions,
    IntPoint, KnnByteVectorField, KnnFloatVectorField, NumericDocValuesField, SortedDocValuesField,
    SortedNumericDocValuesField, SortedSetDocValuesField, Store, StringField, TextField,
    VectorSimilarityFunction,
};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::{self, LuceneVersion};
use lucene_index::segment_infos;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::reader::{IntersectVisitor, LeafReader, PostingsFlags, Relation, NO_MORE_DOCS};
use lucene_search::simpletext_transcode::to_simpletext;
use lucene_store::{DataOutput, Directory, FsDirectory};
use lucene_util::test_support::TempDir;

fn version() -> LuceneVersion {
    LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    }
}

fn doc(i: usize) -> Document {
    let mut d = Document::new();
    d.add(StringField::new("id", format!("d{i}"), Store::Yes));
    let mut body = FieldType::copy_of(&TextField::type_stored());
    body.set_store_term_vectors(true).unwrap();
    body.set_store_term_vector_positions(true).unwrap();
    body.set_store_term_vector_offsets(true).unwrap();
    body.freeze();
    d.add(
        Field::from_string(
            "body",
            format!("w{} x{} common t{i} w{}", i % 5, i % 7, i % 3),
            body,
        )
        .unwrap(),
    );
    // Positions, offsets and payloads from a hand-built stream.
    let mut tok = FieldType::new();
    tok.set_index_options(IndexOptions::DocsAndFreqsAndPositionsAndOffsets)
        .unwrap();
    tok.set_tokenized(true).unwrap();
    tok.freeze();
    let tokens = (0..1 + i % 3)
        .map(|k| {
            let mut t =
                FieldToken::new(format!("p{}", (i + k) % 4), 3 * k as i32, 3 * k as i32 + 2);
            t.payload = (k % 2 == 0).then(|| vec![k as u8, i as u8]);
            t
        })
        .collect();
    d.add(
        Field::from_token_stream(
            "tok",
            FieldTokens {
                tokens,
                final_position_increment: 0,
                final_offset: 9,
                end_attributes: None,
            },
            tok,
        )
        .unwrap(),
    );
    if i % 4 != 1 {
        d.add(IntPoint::new("pt", &[i as i32, -(i as i32) * 3]).unwrap());
        d.add(NumericDocValuesField::new("num", i as i64 * 7 - 50));
        d.add(SortedDocValuesField::new("sorted", format!("s{}", i % 6)));
        d.add(BinaryDocValuesField::new("bin", vec![i as u8; i % 5]));
    }
    d.add(SortedNumericDocValuesField::new("snum", i as i64 % 9));
    d.add(SortedNumericDocValuesField::new("snum", -(i as i64)));
    d.add(SortedSetDocValuesField::new("sset", format!("a{}", i % 4)));
    d.add(SortedSetDocValuesField::new("sset", format!("b{}", i % 3)));
    if i % 3 != 2 {
        d.add(
            KnnFloatVectorField::new(
                "vf",
                vec![i as f32 * 0.5, 1.0, -(i as f32), 0.25],
                VectorSimilarityFunction::Euclidean,
            )
            .unwrap(),
        );
        d.add(
            KnnByteVectorField::new(
                "vb",
                vec![i as u8, 255 - i as u8, 7],
                VectorSimilarityFunction::DotProduct,
            )
            .unwrap(),
        );
    }
    d
}

fn write(dir: &dyn Directory, merge: bool) {
    let mut w = IndexWriter::open(dir, Vec::new(), "Lucene104", version()).unwrap();
    w.set_max_full_flush_merge_wait_millis(0);
    for seg in 0..2 {
        w.set_use_compound_file(seg == 1);
        for i in seg * 60..(seg + 1) * 60 {
            w.add_fields_document(&doc(i)).unwrap();
        }
        w.commit().unwrap();
    }
    let terms: Vec<Term> = (0..120)
        .filter(|i| i % 11 == 4)
        .map(|i| Term::new("id", format!("d{i}")))
        .collect();
    w.delete_documents_by_term(&terms).unwrap();
    w.commit().unwrap();
    if merge {
        w.force_merge(1).unwrap();
        w.commit().unwrap();
    }
}

/// Transcodes every segment of `from` into `to`, with a SimpleText
/// `segments_N`.
fn transcode(from: &dyn Directory, to: &dyn Directory) {
    let mut infos = segment_infos::read_latest(from).unwrap();
    for sci in &mut infos.segments {
        let si = segment_info::parse_for_codec(
            &from.open(&format!("{}.si", sci.segment_name)).unwrap(),
            &sci.segment_id,
            &sci.codec_name,
        )
        .unwrap();
        let text = to_simpletext(from, sci, &si).unwrap();
        let put = |name: &str, bytes: &[u8]| {
            let mut out = to.create_output(name).unwrap();
            out.write_bytes(bytes);
            out.close().unwrap();
        };
        for (name, bytes) in &text.files {
            put(name, bytes);
        }
        put(&format!("{}.si", sci.segment_name), &text.si);
        if let Some((name, bytes)) = &text.live_docs {
            put(name, bytes);
        }
        sci.codec_name = "SimpleText".into();
    }
    segment_infos::write(&infos, to).unwrap();
}

struct All(Vec<(i32, Vec<u8>)>);

impl IntersectVisitor for All {
    fn compare(&mut self, _: &[u8], _: &[u8]) -> Relation {
        Relation::CellCrossesQuery
    }
    fn visit(&mut self, _: i32) {}
    fn visit_with_value(&mut self, doc: i32, value: &[u8]) {
        self.0.push((doc, value.to_vec()));
    }
}

/// Everything a reader serves of a segment, as text.
fn dump(reader: &dyn LeafReader, max_doc: i32) -> String {
    let mut out = String::new();
    let infos = reader.field_infos().clone();
    writeln!(
        out,
        "live {:?}",
        reader
            .live_docs()
            .map(|b| (0..max_doc).map(|d| b.get(d as usize)).collect::<Vec<_>>())
    )
    .unwrap();
    for doc in 0..max_doc {
        let mut visitor = lucene_codecs::stored_fields::DocumentVisitor::all();
        reader.document(doc, &mut visitor).unwrap();
        writeln!(out, "stored {doc} {:?}", visitor.into_document()).unwrap();
        writeln!(out, "tv {doc} {:?}", reader.term_vectors(doc).unwrap()).unwrap();
    }
    for field in &infos.fields {
        let name = &field.name;
        writeln!(
            out,
            "field {name} {:?} {:?} {}",
            field.index_options, field.doc_values_type, field.store_payloads
        )
        .unwrap();
        if let Some(terms) = reader.terms(name).unwrap() {
            writeln!(
                out,
                " stats {} {} {} {}",
                terms.size(),
                terms.sum_doc_freq(),
                terms.sum_total_term_freq(),
                terms.doc_count()
            )
            .unwrap();
            let mut it = terms.iterator().unwrap();
            while let Some(term) = it.next().unwrap() {
                write!(out, " term {term:?}").unwrap();
                let mut p = it.postings(PostingsFlags::All).unwrap();
                while p.next_doc().unwrap() != NO_MORE_DOCS {
                    write!(out, " {}x{}", p.doc_id(), p.freq()).unwrap();
                    if field.index_options.subsumes_positions() {
                        for _ in 0..p.freq() {
                            let pos = p.next_position().unwrap();
                            write!(
                                out,
                                "[{pos} {} {} {:?}]",
                                p.start_offset(),
                                p.end_offset(),
                                p.payload()
                            )
                            .unwrap();
                        }
                    }
                }
                writeln!(out).unwrap();
            }
        }
        if let Some(mut n) = reader.norm_values(name).unwrap() {
            while n.next_doc().unwrap() != NO_MORE_DOCS {
                write!(out, " norm {}={}", n.doc_id(), n.long_value()).unwrap();
            }
            writeln!(out).unwrap();
        }
        if let Some(mut v) = reader.numeric_doc_values(name).unwrap() {
            while v.next_doc().unwrap() != NO_MORE_DOCS {
                write!(out, " {}={}", v.doc_id(), v.long_value()).unwrap();
            }
            writeln!(out).unwrap();
        }
        if let Some(mut v) = reader.binary_doc_values(name).unwrap() {
            while v.next_doc().unwrap() != NO_MORE_DOCS {
                write!(out, " {}={:?}", v.doc_id(), v.binary_value()).unwrap();
            }
            writeln!(out).unwrap();
        }
        if let Some(mut v) = reader.sorted_doc_values(name).unwrap() {
            while v.next_doc().unwrap() != NO_MORE_DOCS {
                let ord = v.ord_value();
                write!(
                    out,
                    " {}={ord}:{:?}",
                    v.doc_id(),
                    v.lookup_ord(ord).unwrap()
                )
                .unwrap();
            }
            writeln!(out).unwrap();
        }
        if let Some(mut v) = reader.sorted_numeric_doc_values(name).unwrap() {
            while v.next_doc().unwrap() != NO_MORE_DOCS {
                write!(out, " {}=", v.doc_id()).unwrap();
                for _ in 0..v.doc_value_count() {
                    write!(out, "{},", v.next_value().unwrap()).unwrap();
                }
            }
            writeln!(out).unwrap();
        }
        if let Some(mut v) = reader.sorted_set_doc_values(name).unwrap() {
            while v.next_doc().unwrap() != NO_MORE_DOCS {
                write!(out, " {}=", v.doc_id()).unwrap();
                for _ in 0..v.doc_value_count() {
                    let ord = v.next_ord().unwrap();
                    write!(out, "{ord}:{:?},", v.lookup_ord(ord).unwrap()).unwrap();
                }
            }
            writeln!(out).unwrap();
        }
        if let Some(p) = reader.point_values(name).unwrap() {
            let mut all = All(Vec::new());
            p.intersect(&mut all).unwrap();
            all.0.sort();
            writeln!(
                out,
                " points {} {} {:?} {:?} {:?}",
                p.size(),
                p.doc_count(),
                p.min_packed_value(),
                p.max_packed_value(),
                all.0
            )
            .unwrap();
        }
        if let Some(v) = reader.float_vector_values(name).unwrap() {
            for ord in 0..v.size() {
                write!(
                    out,
                    " {}={:?}",
                    v.ord_to_doc(ord).unwrap(),
                    v.vector_value(ord).unwrap()
                )
                .unwrap();
            }
            writeln!(out).unwrap();
        }
        if let Some(v) = reader.byte_vector_values(name).unwrap() {
            for ord in 0..v.size() {
                write!(
                    out,
                    " {}={:?}",
                    v.ord_to_doc(ord).unwrap(),
                    v.vector_value(ord).unwrap()
                )
                .unwrap();
            }
            writeln!(out).unwrap();
        }
    }
    out
}

fn check(merge: bool) {
    let tmp = TempDir::new("simpletext-transcode");
    let native_path = tmp.path().join("native");
    let text_path = tmp.path().join("text");
    std::fs::create_dir_all(&native_path).unwrap();
    std::fs::create_dir_all(&text_path).unwrap();
    let (native, text) = (
        FsDirectory::open(&native_path),
        FsDirectory::open(&text_path),
    );
    write(&native, merge);
    transcode(&native, &text);

    let files = text.list_all().unwrap();
    for ext in [
        "pst", "fld", "inf", "dat", "len", "tvc", "dim", "dii", "vec", "gri", "liv",
    ] {
        assert!(
            files.iter().any(|f| f.ends_with(&format!(".{ext}"))) || (merge && ext == "liv"),
            "no .{ext}: {files:?}"
        );
    }
    if !merge {
        assert!(files.iter().any(|f| f.ends_with(".scf")), "{files:?}");
    }

    let (a, b) = (
        DirectoryReader::open(&native).unwrap(),
        DirectoryReader::open(&text).unwrap(),
    );
    assert_eq!(a.segment_readers().len(), b.segment_readers().len());
    for (x, y) in a.segment_readers().iter().zip(b.segment_readers()) {
        assert_eq!(x.max_doc, y.max_doc);
        let (dx, dy) = (dump(x, x.max_doc), dump(y, y.max_doc));
        for (lx, ly) in dx.lines().zip(dy.lines()) {
            assert_eq!(lx, ly);
        }
        assert_eq!(dx, dy);
    }
}

#[test]
fn flushed_segments_read_back_alike() {
    check(false);
}

#[test]
fn a_merged_segment_reads_back_alike() {
    check(true);
}
