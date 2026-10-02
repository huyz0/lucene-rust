//! Writes `GenGeo3dPoints`' corpus through this port's `IndexWriter` -- the
//! same documents (`fixtures/data/geo3d_points/docs.tsv`), commits and
//! deletes -- for `VerifyGeo3D` to open with real Lucene 10.5.0:
//! `CheckIndex`, then every `Geo3DPoint` query and `Geo3DDocValuesField`
//! sort of `queries.tsv` replayed through Lucene's own spatial3d over this
//! index and over Lucene's own, which must answer alike.
//!
//! `Geo3DPoint` is a three-dimension point, so this is also the BKD
//! writer's real-Lucene check for three indexed dimensions.
//!
//! Usage: `write_geo3d_points_fixture <output-dir>`.
#![allow(clippy::arithmetic_side_effects)]

use lucene_index::buffered_updates::Term;
use lucene_index::document::geo3d::from_degrees;
use lucene_index::document::{Document, Geo3DDocValuesField, Geo3DPoint, Store, StringField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_store::FsDirectory;
use lucene_util::spatial3d::{GeoPoint, PlanetModel};

fn main() {
    let out = std::env::args()
        .nth(1)
        .expect("usage: write_geo3d_points_fixture <output-dir>");
    let fixtures =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/geo3d_points");
    let read =
        |f: &str| std::fs::read_to_string(fixtures.join(f)).expect("GenGeo3dPoints fixtures");
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
            let (field, ab) = spec.split_once(':').expect("field:lat,lon");
            let (a, b) = ab.split_once(',').expect("lat,lon");
            let (lat, lon): (f64, f64) = (a.parse().expect("lat"), b.parse().expect("lon"));
            let pm = if field == "s" {
                PlanetModel::sphere()
            } else {
                PlanetModel::wgs84()
            };
            // `GenGeo3dPoints.addFields`: `p` a point, `pd` a doc value,
            // `s` both, on the sphere.
            if field != "pd" {
                doc.add(Geo3DPoint::with_planet_model(field, &pm, lat, lon).expect("point"));
            }
            if field != "p" {
                let g = GeoPoint::from_lat_lon(&pm, from_degrees(lat), from_degrees(lon))
                    .expect("point");
                doc.add(Geo3DDocValuesField::new(field, &g, &pm).expect("doc value"));
            }
        }
        w.add_fields_document(&doc).expect("add");
        if matches!(i + 1, 1500 | 3000 | 4500 | 4550) {
            w.commit().expect("commit");
        }
    }
    for id in read("deletes.tsv").lines() {
        w.delete_documents_by_term(&[Term::new("id", id)])
            .expect("delete");
    }
    w.commit().expect("commit");
}
