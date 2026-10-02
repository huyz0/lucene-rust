//! Writes `GenGeoPoints`' corpus through this port's `IndexWriter` -- the
//! same documents (`fixtures/data/geo_points/docs.tsv`), commits and deletes
//! -- for `VerifyGeoPoints` to open with real Lucene 10.5.0: `CheckIndex`,
//! then every query of `queries.tsv` replayed through Lucene's own
//! `LatLonPoint`/`LatLonDocValuesField`/`XYPointField`/`XYDocValuesField`
//! queries, sorts and `nearest`, which must answer exactly as they answered
//! over Lucene's own index.
//!
//! `LatLonPoint` and `XYPointField` are two-dimension points, so this is the
//! multi-dimension BKD writer's real-Lucene check as well as the geo
//! encodings'.
//!
//! Usage: `write_geo_points_fixture <output-dir>`.
#![allow(clippy::arithmetic_side_effects)]

use lucene_index::buffered_updates::Term;
use lucene_index::document::{
    Document, LatLonDocValuesField, LatLonPoint, Store, StringField, XYDocValuesField, XYPointField,
};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_store::FsDirectory;

fn main() {
    let out = std::env::args()
        .nth(1)
        .expect("usage: write_geo_points_fixture <output-dir>");
    let fixtures =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/geo_points");
    let read = |f: &str| std::fs::read_to_string(fixtures.join(f)).expect("GenGeoPoints fixtures");
    let _ = std::fs::remove_dir_all(&out);
    std::fs::create_dir_all(&out).expect("create output dir");
    let fs = FsDirectory::open(std::path::Path::new(&out));
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w = IndexWriter::open(&fs, Vec::new(), "Lucene104", version).expect("open writer");
    for (i, line) in read("docs.tsv").lines().enumerate() {
        let mut p = line.split('\t');
        let id = p.next().expect("id");
        let mut doc = Document::new();
        doc.add(StringField::new("id", id, Store::Yes));
        for spec in p {
            let (field, ab) = spec.split_once(':').expect("field:a,b");
            let (a, b) = ab.split_once(',').expect("a,b");
            if field == "xy" {
                let (x, y): (f32, f32) = (a.parse().expect("x"), b.parse().expect("y"));
                doc.add(XYPointField::new("xy", x, y).expect("xy"));
                doc.add(XYDocValuesField::new("xy", x, y).expect("xy"));
            } else {
                let (lat, lon): (f64, f64) = (a.parse().expect("lat"), b.parse().expect("lon"));
                doc.add(LatLonPoint::new(field, lat, lon).expect("lat/lon"));
                doc.add(LatLonDocValuesField::new(field, lat, lon).expect("lat/lon"));
            }
        }
        w.add_fields_document(&doc).expect("add");
        if matches!(i + 1, 2000 | 4000 | 6000 | 6060) {
            w.commit().expect("commit");
        }
    }
    for id in read("deletes.tsv").lines() {
        w.delete_documents_by_term(&[Term::new("id", id)])
            .expect("delete");
    }
    w.commit().expect("commit");
}
