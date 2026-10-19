//! lucene-sandbox's document package and multi-range queries,
//! differentially against Lucene 10.5.0: `fixtures/src/GenSandboxDocument.java`
//! built documents from field specs (`docs.tsv`), recorded every sandbox
//! field's packed value (`packed.tsv`), `HalfFloatPoint`'s and
//! `BigIntegerPoint`'s encodings (`half.tsv`, `big.tsv`), and ran the
//! sandbox's queries (`queries.tsv`), `FloatPointNearestNeighbor`
//! (`nearest.tsv`) and the argument checks (`errors.tsv`) over the index it
//! wrote. This builds the same fields and queries with
//! `lucene_search::sandbox` and requires the same bytes, hits, distances
//! and messages -- over Java's index, and over an index this port writes
//! from the same specs.

#![allow(clippy::arithmetic_side_effects)]

use lucene_index::buffered_updates::Term;
use lucene_index::document::*;
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::collector::ScoreDoc;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::document::{self as dq, Boosted, DocumentQuery};
use lucene_search::multi_segment::OpenSegment;
use lucene_search::sandbox::document::{
    BigIntegerPoint, DoublePointMultiRangeBuilder, FloatPointMultiRangeBuilder,
    FloatPointNearestNeighbor, HalfFloatPoint, IntPointMultiRangeBuilder, LatLonBoundingBox,
    LongPointMultiRangeBuilder,
};
use lucene_search::sandbox::search::{SortedNumericStabbingBuilder, SortedSetStabbingBuilder};
use lucene_search::Result;
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;

const NAME: &str = "sandbox_document";

fn root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/data")
        .join(NAME)
}

fn read(file: &str) -> String {
    std::fs::read_to_string(root().join(file))
        .expect("run scripts/gen-fixtures.sh --only GenSandboxDocument")
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn list<T: std::str::FromStr>(s: &str) -> Vec<T>
where
    T::Err: std::fmt::Debug,
{
    s.split(',').map(|v| v.parse().unwrap()).collect()
}

/// `GenSandboxDocument.field(spec)`.
fn field(spec: &str) -> Box<dyn IndexableField> {
    let a: Vec<&str> = spec.split(' ').collect();
    let f = a.get(1).copied().unwrap_or("");
    match a[0] {
        "id" => Box::new(StringField::new("id", a[1], Store::No)),
        "HalfFloatPoint" => Box::new(HalfFloatPoint::new(f, &list::<f32>(a[2])).unwrap()),
        "BigIntegerPoint" => Box::new(BigIntegerPoint::new(f, &list::<i128>(a[2])).unwrap()),
        "LatLonBoundingBox" => {
            let d = list::<f64>(a[2]);
            Box::new(LatLonBoundingBox::new(f, d[0], d[1], d[2], d[3]).unwrap())
        }
        "IntPoint" => Box::new(IntPoint::new(f, &list::<i32>(a[2])).unwrap()),
        "LongPoint" => Box::new(LongPoint::new(f, &list::<i64>(a[2])).unwrap()),
        "FloatPoint" => Box::new(FloatPoint::new(f, &list::<f32>(a[2])).unwrap()),
        "DoublePoint" => Box::new(DoublePoint::new(f, &list::<f64>(a[2])).unwrap()),
        "NumericDocValuesField" => {
            let v = a[2].parse().unwrap();
            Box::new(if a[3] == "1" {
                NumericDocValuesField::indexed_field(f, v)
            } else {
                NumericDocValuesField::new(f, v)
            })
        }
        "SortedNumericDocValuesField" => {
            let v = a[2].parse().unwrap();
            Box::new(if a[3] == "1" {
                SortedNumericDocValuesField::indexed_field(f, v)
            } else {
                SortedNumericDocValuesField::new(f, v)
            })
        }
        "SortedDocValuesField" => Box::new(if a[3] == "1" {
            SortedDocValuesField::indexed_field(f, a[2])
        } else {
            SortedDocValuesField::new(f, a[2])
        }),
        "SortedSetDocValuesField" => Box::new(if a[3] == "1" {
            SortedSetDocValuesField::indexed_field(f, a[2])
        } else {
            SortedSetDocValuesField::new(f, a[2])
        }),
        other => panic!("unknown field spec {other}"),
    }
}

fn docs() -> Vec<Vec<String>> {
    docs_in("docs.tsv")
}

fn docs_in(file: &str) -> Vec<Vec<String>> {
    read(file)
        .lines()
        .map(|l| l.split('\t').skip(1).map(str::to_string).collect())
        .collect()
}

/// The ranges of a multi-range spec: `lo:hi;lo:hi`.
fn ranges(a: &[&str], at: usize) -> Vec<String> {
    a.get(at)
        .map(|s| s.split(';').map(str::to_string).collect())
        .unwrap_or_default()
}

fn bounds<T: std::str::FromStr>(r: &str) -> (Vec<T>, Vec<T>)
where
    T::Err: std::fmt::Debug,
{
    let (lo, hi) = r.split_once(':').unwrap();
    (list(lo), list(hi))
}

/// `GenSandboxDocument.query(spec)`.
fn query(spec: &str) -> Result<Box<dyn DocumentQuery>> {
    let a: Vec<&str> = spec.split(' ').collect();
    let f = a.get(1).copied().unwrap_or("");
    let dims = || a[2].parse::<usize>().unwrap();
    Ok(match a[0] {
        "boost" => Box::new(Boosted::new(
            query(&a[2..].join(" "))?,
            a[1].parse().unwrap(),
        )),
        "HalfRange" => Box::new(HalfFloatPoint::new_range_query_nd(
            f,
            &list::<f32>(a[2]),
            &list::<f32>(a[3]),
        )?),
        "HalfExact" => Box::new(HalfFloatPoint::new_exact_query(f, a[2].parse().unwrap())?),
        "HalfSet" | "HalfSetList" => Box::new(HalfFloatPoint::new_set_query(
            f,
            &a.get(2).map(|s| list::<f32>(s)).unwrap_or_default(),
        )?),
        "BigRange" => Box::new(BigIntegerPoint::new_range_query_nd(
            f,
            &list::<i128>(a[2]),
            &list::<i128>(a[3]),
        )?),
        "BigRange1" => Box::new(BigIntegerPoint::new_range_query(
            f,
            a[2].parse().unwrap(),
            a[3].parse().unwrap(),
        )?),
        "BigExact" => Box::new(BigIntegerPoint::new_exact_query(f, a[2].parse().unwrap())?),
        "BigSet" => Box::new(BigIntegerPoint::new_set_query(
            f,
            &a.get(2).map(|s| list::<i128>(s)).unwrap_or_default(),
        )?),
        "BBox" => {
            let d = list::<f64>(a[3]);
            Box::new(match a[2] {
                "INTERSECTS" => LatLonBoundingBox::new_intersects_query(f, d[0], d[1], d[2], d[3])?,
                "WITHIN" => LatLonBoundingBox::new_within_query(f, d[0], d[1], d[2], d[3])?,
                "CONTAINS" => LatLonBoundingBox::new_contains_query(f, d[0], d[1], d[2], d[3])?,
                "CROSSES" => LatLonBoundingBox::new_crosses_query(f, d[0], d[1], d[2], d[3])?,
                other => panic!("{other}"),
            })
        }
        "IntMR" => {
            let mut b = IntPointMultiRangeBuilder::new(f, dims())?;
            for r in ranges(&a, 3) {
                let (lo, hi) = bounds::<i32>(&r);
                b.add(&lo, &hi)?;
            }
            Box::new(b.build())
        }
        "LongMR" => {
            let mut b = LongPointMultiRangeBuilder::new(f, dims())?;
            for r in ranges(&a, 3) {
                let (lo, hi) = bounds::<i64>(&r);
                b.add(&lo, &hi)?;
            }
            Box::new(b.build())
        }
        "FloatMR" => {
            let mut b = FloatPointMultiRangeBuilder::new(f, dims())?;
            for r in ranges(&a, 3) {
                let (lo, hi) = bounds::<f32>(&r);
                b.add(&lo, &hi)?;
            }
            Box::new(b.build())
        }
        "DoubleMR" => {
            let mut b = DoublePointMultiRangeBuilder::new(f, dims())?;
            for r in ranges(&a, 3) {
                let (lo, hi) = bounds::<f64>(&r);
                b.add(&lo, &hi)?;
            }
            Box::new(b.build())
        }
        "RawMR" => {
            let mut b = IntPointMultiRangeBuilder::new(f, dims())?;
            for r in ranges(&a, 3) {
                let (lo, hi) = bounds::<i32>(&r);
                b.add_packed(IntPoint::pack(&lo).unwrap(), IntPoint::pack(&hi).unwrap());
            }
            Box::new(b.build())
        }
        "SNMR" => {
            let mut b = SortedNumericStabbingBuilder::new(f);
            for r in ranges(&a, 2) {
                let (lo, hi) = r.split_once(':').unwrap();
                b.add(lo.parse().unwrap(), hi.parse().unwrap());
            }
            b.build()
        }
        "SSMR" => {
            let mut b = SortedSetStabbingBuilder::new(f);
            for r in ranges(&a, 2) {
                match r.split_once(':') {
                    None => b.add_single(r.as_bytes()),
                    Some((lo, hi)) => b.add(lo.as_bytes(), hi.as_bytes()),
                };
            }
            b.build()
        }
        other => panic!("unknown query spec {other}"),
    })
}

/// `GenSandboxDocument.hits(searcher, query)`.
fn hits(segments: &[OpenSegment<'_>], q: &dyn DocumentQuery) -> Result<String> {
    let td = dq::search_top_docs(segments, q, 100_000)?;
    let docs: &[ScoreDoc] = &td.score_docs;
    let mut out = td.total_hits.value.to_string();
    let constant = docs.iter().all(|d| d.score == docs[0].score);
    if constant {
        let mut ids: Vec<i32> = docs.iter().map(|d| d.doc_id).collect();
        ids.sort_unstable();
        out.push_str(&format!(
            "\tC:{}",
            if ids.is_empty() {
                "0".to_string()
            } else {
                format!("{:x}", docs[0].score.to_bits())
            }
        ));
        let mut i = 0;
        while i < ids.len() {
            let mut j = i;
            while j + 1 < ids.len() && ids[j + 1] == ids[j] + 1 {
                j += 1;
            }
            out.push_str(&format!("\t{}-{}", ids[i], ids[j]));
            i = j + 1;
        }
    } else {
        out.push_str("\tS");
        for d in docs {
            out.push_str(&format!("\t{}:{:x}", d.doc_id, d.score.to_bits()));
        }
    }
    Ok(out)
}

fn check(dir: &std::path::Path) -> usize {
    let reader = DirectoryReader::open(&FsDirectory::open(dir)).expect("open");
    let opened = reader.open_segments().expect("open postings");
    let segments = opened.as_open_segments();
    let mut failures = Vec::new();
    let mut n = 0;
    for line in read("queries.tsv").lines() {
        let (spec, want) = line.split_once('\t').unwrap();
        let got = query(spec)
            .and_then(|q| hits(&segments, q.as_ref()))
            .unwrap_or_else(|e| format!("error: {e}"));
        if got != want {
            let short = |s: &str| s.chars().take(300).collect::<String>();
            failures.push(format!(
                "{spec}\n  java: {}\n  rust: {}",
                short(want),
                short(&got)
            ));
        }
        n += 1;
    }
    for line in read("errors.tsv").lines() {
        let (spec, want) = line.split_once('\t').unwrap();
        let got = if let Some(rest) = spec.strip_prefix("nearest ") {
            let (n, origin) = rest.split_once(' ').unwrap();
            FloatPointNearestNeighbor::nearest(&segments, "v", n.parse().unwrap(), &list(origin))
                .map(|_| ())
        } else {
            query(spec).and_then(|q| hits(&segments, q.as_ref()).map(|_| ()))
        };
        let ok = match (&got, want) {
            (Ok(()), "OK") => true,
            // Java's runtime failures (a `null` tree, an empty set's
            // `getFirst()`) are errors here; their messages are this port's.
            (Err(_), "NullPointerException" | "NoSuchElementException") => true,
            (Err(e), msg) => {
                let s = e.to_string();
                s.strip_prefix("document query: ").unwrap_or(&s) == msg
            }
            _ => false,
        };
        if !ok {
            failures.push(format!("{spec}\n  java: {want}\n  rust: {got:?}"));
        }
        n += 1;
    }
    for line in read("nearest.tsv").lines() {
        let mut p = line.split('\t');
        let spec = p.next().unwrap();
        let want: Vec<&str> = p.collect();
        let (top_n, origin) = spec.split_once(' ').unwrap();
        let r = FloatPointNearestNeighbor::nearest(
            &segments,
            "v",
            top_n.parse().unwrap(),
            &list::<f32>(origin),
        )
        .unwrap();
        let mut got = vec![r.total_hits.to_string()];
        got.extend(r.hits.iter().map(|(d, f)| format!("{d}:{:x}", f.to_bits())));
        if got != want {
            failures.push(format!("nearest {spec}\n  java: {want:?}\n  rust: {got:?}"));
        }
        n += 1;
    }
    assert!(
        failures.is_empty(),
        "{} of {n} checks differ:\n{}",
        failures.len(),
        failures.join("\n")
    );
    n
}

#[test]
fn every_query_matches_lucene_on_lucenes_index() {
    let n = check(&root().join("index"));
    assert!(n > 150, "{n} checks");
}

/// The same documents, written by this port: every commit and delete
/// `GenSandboxDocument.main` made, in order.
#[test]
fn every_query_matches_lucene_on_this_ports_index() {
    let tmp = TempDir::new("sandbox-document-write");
    let fs = FsDirectory::open(tmp.path());
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w = IndexWriter::open(&fs, Vec::new(), "Lucene104", version).unwrap();
    for (i, specs) in docs().into_iter().enumerate() {
        let mut doc = Document::new();
        for s in &specs {
            doc.add_boxed(field(s));
        }
        w.add_fields_document(&doc).unwrap();
        if (i + 1) % 200 == 0 {
            w.commit().unwrap();
        }
    }
    for i in (0..200).step_by(13) {
        w.delete_documents_by_term(&[Term::new("id", i.to_string())])
            .unwrap();
        w.delete_documents_by_term(&[Term::new("id", (400 + i).to_string())])
            .unwrap();
    }
    w.commit().unwrap();
    drop(w);
    let n = check(tmp.path());
    assert!(n > 150, "{n} checks");
}

#[test]
fn every_field_packs_what_lucene_packs() {
    let mut n = 0;
    for line in read("packed.tsv").lines() {
        let (spec, want) = line.split_once('\t').unwrap();
        let f = field(spec);
        assert_eq!(hex(&f.binary_value().unwrap()), want, "{spec}");
        n += 1;
    }
    assert!(n > 500, "{n} fields");
}

#[test]
fn half_float_point_encodes_as_lucene() {
    let mut n = 0;
    for line in read("half.tsv").lines() {
        let p: Vec<&str> = line.split('\t').collect();
        let bits = |s: &str| f32::from_bits(u32::from_str_radix(s, 16).unwrap());
        let v = bits(p[0]);
        let s = HalfFloatPoint::half_float_to_sortable_short(v);
        let got = [
            s.to_string(),
            format!(
                "{:x}",
                HalfFloatPoint::sortable_short_to_half_float(s).to_bits()
            ),
            format!("{:x}", HalfFloatPoint::next_up(v).to_bits()),
            format!("{:x}", HalfFloatPoint::next_down(v).to_bits()),
            hex(&HalfFloatPoint::encode_dimension(v)),
            format!(
                "{:x}",
                HalfFloatPoint::decode_dimension(&HalfFloatPoint::encode_dimension(v)).to_bits()
            ),
        ];
        assert_eq!(got.as_slice(), &p[1..], "{line}");
        n += 1;
    }
    assert!(n > 5000, "{n} values");
}

#[test]
fn big_integer_point_encodes_as_lucene() {
    for line in read("big.tsv").lines() {
        let (v, want) = line.split_once('\t').unwrap();
        let v: i128 = v.parse().unwrap();
        assert_eq!(hex(&BigIntegerPoint::encode_dimension(v)), want, "{v}");
        let mut b = [0u8; 16];
        for (i, x) in b.iter_mut().enumerate() {
            *x = u8::from_str_radix(&want[2 * i..2 * i + 2], 16).unwrap();
        }
        assert_eq!(BigIntegerPoint::decode_dimension(&b), v);
    }
}

/// Over an index too big for one BKD leaf -- so the walk expands inner cells
/// and prunes them -- `FloatPointNearestNeighbor` returns exactly the
/// `topN` live points nearest the origin (by distance, then document), with
/// ties on small integer coordinates; `MultiRangeQuery` and the doc-values
/// multi-range queries take their every-document and argument paths.
#[test]
fn nearest_and_multi_ranges_over_a_multi_leaf_index() {
    use lucene_search::sandbox::search::{
        ByteRange, LongRange, MultiRangeQueryBuilder, SortedNumericDocValuesMultiRangeQuery,
        SortedSetDocValuesMultiRangeQuery,
    };
    let tmp = TempDir::new("sandbox-nearest");
    let fs = FsDirectory::open(tmp.path());
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w = IndexWriter::open(&fs, Vec::new(), "Lucene104", version).unwrap();
    let mut seed = 0x2026_1019u64;
    let mut next = |n: u64| {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 33) % n) as f32
    };
    // (doc, point) of every point, deleted documents' included.
    let mut points: Vec<(i32, [f32; 3])> = Vec::new();
    let n_docs = 6000;
    for i in 0..n_docs {
        let mut doc = Document::new();
        doc.add_boxed(Box::new(StringField::new("id", i.to_string(), Store::No)));
        doc.add_boxed(Box::new(IntPoint::new("all", &[7]).unwrap()));
        doc.add_boxed(Box::new(StringField::new("s", "x", Store::No)));
        if i % 7 != 0 {
            for _ in 0..1 + (i % 3 == 0) as usize {
                let p = [next(40), next(40), next(40)];
                doc.add_boxed(Box::new(FloatPoint::new("v", &p).unwrap()));
                points.push((i, p));
            }
        }
        if i % 5 == 0 {
            doc.add_boxed(Box::new(LongPoint::new("l", &[i64::from(i)]).unwrap()));
        }
        // Skip-indexed: on every document, and on every fourth (two values).
        let v = i64::from(i % 1000);
        doc.add_boxed(Box::new(SortedNumericDocValuesField::indexed_field(
            "dx", v,
        )));
        if i % 4 == 0 && i < 5900 {
            for x in [v, v + 1] {
                doc.add_boxed(Box::new(SortedNumericDocValuesField::indexed_field(
                    "gx", x,
                )));
            }
        }
        doc.add_boxed(Box::new(SortedSetDocValuesField::new(
            "ss",
            format!("t{:03}", i % 500),
        )));
        w.add_fields_document(&doc).unwrap();
        if i == 3999 {
            w.commit().unwrap();
        }
    }
    let deleted = |i: i32| i % 11 == 0;
    for i in (0..n_docs).filter(|&i| deleted(i)) {
        w.delete_documents_by_term(&[Term::new("id", i.to_string())])
            .unwrap();
    }
    w.commit().unwrap();
    drop(w);
    let reader = DirectoryReader::open(&FsDirectory::open(tmp.path())).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    assert_eq!(segments.len(), 2);
    let doc_count = points
        .iter()
        .map(|p| p.0)
        .collect::<std::collections::BTreeSet<_>>()
        .len() as i64;
    for (top_n, origin) in [
        (1, [0.0, 0.0, 0.0]),
        (10, [20.0, 20.0, 20.0]),
        (25, [3.0, 39.0, 17.0]),
        (100, [-5.0, 50.0, 10.0]),
        (7000, [10.5, 10.5, 10.5]),
    ] {
        let mut want: Vec<(f64, i32)> = points
            .iter()
            .filter(|(d, _)| !deleted(*d))
            .map(|(d, p)| {
                let s: f64 = p
                    .iter()
                    .zip(origin)
                    .map(|(&a, b)| (f64::from(a) - f64::from(b)).powi(2))
                    .sum();
                (s, *d)
            })
            .collect();
        want.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        want.truncate(top_n as usize);
        let got = FloatPointNearestNeighbor::nearest(&segments, "v", top_n, &origin).unwrap();
        assert_eq!(got.total_hits, doc_count);
        let want: Vec<(i32, f32)> = want.iter().map(|&(s, d)| (d, s.sqrt() as f32)).collect();
        assert_eq!(got.hits, want, "top {top_n} of {origin:?}");
    }
    // A shorter origin reads the first dimensions only.
    let got = FloatPointNearestNeighbor::nearest(&segments, "v", 3, &[1.0]).unwrap();
    assert_eq!(got.hits.len(), 3);
    // No such field, no points, the wrong width or too many dimensions.
    let none = FloatPointNearestNeighbor::nearest(&segments, "nope", 3, &[1.0]).unwrap();
    assert_eq!((none.total_hits, none.hits.len()), (0, 0));
    let none = FloatPointNearestNeighbor::nearest(&segments, "s", 3, &[1.0]).unwrap();
    assert_eq!(none.hits.len(), 0);
    assert!(FloatPointNearestNeighbor::nearest(&segments, "l", 3, &[1.0]).is_err());
    assert!(FloatPointNearestNeighbor::nearest(&segments, "v", 3, &[1.0; 4]).is_err());
    assert!(FloatPointNearestNeighbor::nearest(&segments, "v", 0, &[1.0]).is_err());

    // `MultiRangeQuery`: every document inside answers without a walk.
    let count = |q: &dyn DocumentQuery| {
        dq::search_top_docs(&segments, q, 100_000)
            .unwrap()
            .total_hits
            .value
    };
    let live = (0..n_docs).filter(|&i| !deleted(i)).count() as u64;
    let mut b = MultiRangeQueryBuilder::new("all", 4, 1).unwrap();
    b.add(
        IntPoint::pack(&[0]).unwrap(),
        IntPoint::pack(&[10]).unwrap(),
    );
    assert_eq!(count(&b.build()), live);
    let mut b = MultiRangeQueryBuilder::new("l", 8, 1).unwrap();
    b.add(
        LongPoint::pack(&[0]).unwrap(),
        LongPoint::pack(&[99]).unwrap(),
    );
    b.add(
        LongPoint::pack(&[5000]).unwrap(),
        LongPoint::pack(&[5004]).unwrap(),
    );
    let want = (0..n_docs)
        .filter(|&i| i % 5 == 0 && !deleted(i) && (i <= 99 || (5000..=5004).contains(&i)))
        .count() as u64;
    assert_eq!(count(&b.build()), want);
    for (field, bytes, dims) in [("nope", 4, 1), ("s", 4, 1)] {
        let mut b = MultiRangeQueryBuilder::new(field, bytes, dims).unwrap();
        b.add(vec![0; 4], vec![1; 4]);
        assert_eq!(count(&b.build()), 0);
    }
    let search = |q: &dyn DocumentQuery| dq::search_top_docs(&segments, q, 10).map(|_| ());
    let mut b = MultiRangeQueryBuilder::new("v", 4, 1).unwrap();
    b.add(vec![0; 4], vec![1; 4]);
    assert!(search(&b.build())
        .unwrap_err()
        .to_string()
        .contains("numIndexDimensions=3"));
    let mut b = MultiRangeQueryBuilder::new("v", 8, 3).unwrap();
    b.add(vec![0; 24], vec![1; 24]);
    assert!(search(&b.build())
        .unwrap_err()
        .to_string()
        .contains("bytesPerDim=4"));

    // Past 1,000 hits a top-10 search stops early, with the first ten hits.
    let early = |q: &dyn DocumentQuery, want: usize| {
        let all = dq::search_all(&segments, q).unwrap();
        assert_eq!(all.len(), want);
        let top: Vec<i32> = dq::search_top_docs(&segments, q, 10)
            .unwrap()
            .score_docs
            .iter()
            .map(|d| d.doc_id)
            .collect();
        let first: Vec<i32> = all.iter().take(10).map(|d| d.doc_id).collect();
        assert_eq!(top, first);
    };
    let live_where = |f: &dyn Fn(i32) -> bool| (0..n_docs).filter(|&i| !deleted(i) && f(i)).count();
    let mut b = MultiRangeQueryBuilder::new("all", 4, 1).unwrap();
    b.add(
        IntPoint::pack(&[0]).unwrap(),
        IntPoint::pack(&[10]).unwrap(),
    );
    early(&b.build(), live_where(&|_| true));
    let mut b = MultiRangeQueryBuilder::new("l", 8, 1).unwrap();
    b.add(
        LongPoint::pack(&[0]).unwrap(),
        LongPoint::pack(&[9999]).unwrap(),
    );
    early(&b.build(), live_where(&|i| i % 5 == 0));
    let ranges = [LongRange::new(0, 99), LongRange::new(500, 599)];
    let q = SortedNumericDocValuesMultiRangeQuery::new("dx", &ranges);
    early(
        &q,
        live_where(&|i| i % 1000 < 100 || (500..600).contains(&(i % 1000))),
    );
    let q = SortedSetDocValuesMultiRangeQuery::new(
        "ss",
        vec![
            ByteRange::new("t000", "t249"),
            ByteRange::new("t300", "t349"),
        ],
    );
    early(
        &q,
        live_where(&|i| i % 500 < 250 || (300..350).contains(&(i % 500))),
    );
    // A sparse skip-indexed field read stale (see `StaleCursor`) runs ahead
    // of its documents and off the end of its values: no match here, an
    // `EOFException` in Java (which `GenSandboxDocument`'s large index avoids
    // with a dense tail), so this only checks the walk ends.
    let q = SortedNumericDocValuesMultiRangeQuery::new("gx", &ranges);
    assert!(!dq::search_all(&segments, &q).unwrap().is_empty());

    // The doc-values queries on fields this index lacks match nothing.
    let q = SortedNumericDocValuesMultiRangeQuery::new(
        "nope",
        &[LongRange::new(1, 2), LongRange::new(4, 5)],
    );
    assert_eq!(count(&q), 0);
    let q = SortedSetDocValuesMultiRangeQuery::new(
        "nope",
        vec![ByteRange::new("a", "b"), ByteRange::single("c")],
    );
    assert_eq!(count(&q), 0);
}

/// `IndexSearcher.count(q)`'s collector: every match, no scores.
#[derive(Default)]
struct Count(u64);

impl lucene_search::collector::ScoringCollector for Count {
    fn collect(&mut self, _doc_id: i32, _score: f32) {
        self.0 += 1;
    }
    fn score_mode(&self) -> lucene_search::collector::ScoreMode {
        lucene_search::collector::ScoreMode::CompleteNoScores
    }
}

/// `GenSandboxDocument.large`'s queries: `search(q, 10)` (whose total hits
/// show where the bulk scorer stopped), `count(q)` and every hit -- the
/// three ways Java's bulk scorers confirm and mask a stale-read candidate.
fn check_large(dir: &std::path::Path) -> usize {
    use lucene_search::collector::{LeafCollector, TotalHitsRelation};
    let reader = DirectoryReader::open(&FsDirectory::open(dir)).expect("open");
    let opened = reader.open_segments().expect("open postings");
    let segments = opened.as_open_segments();
    assert_eq!(segments.len(), 1);
    let mut failures = Vec::new();
    let mut n = 0;
    for line in read("large/queries.tsv").lines() {
        let (spec, want) = line.split_once('\t').unwrap();
        let q = query(spec).unwrap();
        let top = dq::search_top_docs(&segments, q.as_ref(), 10).unwrap();
        let relation = match top.total_hits.relation {
            TotalHitsRelation::EqualTo => "EQUAL_TO",
            TotalHitsRelation::GreaterThanOrEqualTo => "GREATER_THAN_OR_EQUAL_TO",
        };
        let docs: String = top
            .score_docs
            .iter()
            .map(|d| format!("{},", d.doc_id))
            .collect();
        let mut count = Count::default();
        for leaf in &segments {
            let mut lc = LeafCollector::new(&mut count, leaf.doc_base);
            q.score_leaf(leaf, 1.0, &mut lc).unwrap();
        }
        let got = format!(
            "{}\t{relation}\t{docs}\t{}\t{}",
            top.total_hits.value,
            count.0,
            hits(&segments, q.as_ref()).unwrap()
        );
        if got != want {
            let short = |s: &str| s.chars().take(300).collect::<String>();
            failures.push(format!(
                "{spec}\n  java: {}\n  rust: {}",
                short(want),
                short(&got)
            ));
        }
        n += 1;
    }
    assert!(
        failures.is_empty(),
        "{} of {n} checks differ:\n{}",
        failures.len(),
        failures.join("\n")
    );
    n
}

#[test]
fn large_segment_queries_match_lucene_on_lucenes_index() {
    let n = check_large(&root().join("large/index"));
    assert!(n >= 15, "{n} checks");
}

/// `GenSandboxDocument.large`'s documents, written by this port.
#[test]
fn large_segment_queries_match_lucene_on_this_ports_index() {
    let tmp = TempDir::new("sandbox-document-large");
    let fs = FsDirectory::open(tmp.path());
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w = IndexWriter::open(&fs, Vec::new(), "Lucene104", version).unwrap();
    for specs in docs_in("large/docs.tsv") {
        let mut doc = Document::new();
        for s in &specs {
            doc.add_boxed(field(s));
        }
        w.add_fields_document(&doc).unwrap();
    }
    w.commit().unwrap();
    for i in (0..9000).step_by(9) {
        w.delete_documents_by_term(&[Term::new("id", i.to_string())])
            .unwrap();
    }
    w.commit().unwrap();
    drop(w);
    let n = check_large(tmp.path());
    assert!(n >= 15, "{n} checks");
}
