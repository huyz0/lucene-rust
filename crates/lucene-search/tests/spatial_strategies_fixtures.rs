//! spatial-extras' strategies, differentially against Lucene 10.5.0:
//! `fixtures/src/GenSpatialStrategies.java` indexed a seeded corpus through
//! every strategy of `SpatialExtrasCorpus` (RPT over geohash, quad pruned
//! and not, packed quad, S2 and a planar quad; RPT points-only; the
//! term-query strategy; BBox geodetic and planar; point-vector; serialized
//! doc values with Spatial4j's and Geo3D's codecs; composite optimized and
//! not; the date-range strategy) across four segments with deletions, and
//! recorded every field each shape makes and Lucene's answer to every
//! question: each strategy x operation x query shape, value sources over
//! every document, heatmaps, date facets.
//!
//! This builds the same strategies, requires the same fields (tokens byte
//! for byte), and the same answers -- over Lucene's index and over the index
//! this port writes from the same documents.

#![allow(clippy::arithmetic_side_effects)]

#[path = "common/spatial_corpus.rs"]
mod spatial_corpus;

use std::sync::Arc;

use lucene_index::buffered_updates::Term;
use lucene_index::document::{DocValuesType, IndexOptions, IndexableField, Number};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::document::{self as dq};
use lucene_search::index_searcher::IndexSearcher;
use lucene_search::multi_segment::OpenSegment;
use lucene_search::spatial::bbox::BBoxOverlapRatioValueSource;
use lucene_search::spatial::make_recip_distance_value_source;
use lucene_search::spatial::util::{CachingDoubleValueSource, ShapeAreaValueSource};
use lucene_search::values_source::{DoubleValuesSource, ValuesContext};
use lucene_store::FsDirectory;
use lucene_util::fixed_bit_set::FixedBitSet;
use lucene_util::spatial_extras::prefix_tree::UnitNRShape;
use lucene_util::spatial_extras::query::{SpatialArgs, SpatialOperation};
use lucene_util::test_support::TempDir;

fn root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/spatial_strategies")
}

use spatial_corpus::{corpus, Corpus};

fn read(file: &str) -> String {
    std::fs::read_to_string(root().join(file))
        .expect("run scripts/gen-fixtures.sh --only GenSpatialStrategies")
}

fn esc(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('\t', "\\t")
        .replace('\n', "\\n")
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn h(v: f64) -> String {
    format!("{:x}", v.to_bits())
}

/// `SpatialExtrasCorpus.tokens`.
fn tokens(tokens: &[String]) -> String {
    if tokens.len() <= 24 {
        return tokens.join(",");
    }
    let mut hash: u64 = 0xcbf29ce484222325;
    for t in tokens {
        for i in (0..t.len()).step_by(2) {
            hash ^= u64::from_str_radix(&t[i..i + 2], 16).unwrap();
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash ^= 0xff;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("#{}:{}:{:x}", tokens.len(), tokens[..3].join(","), hash)
}

fn index_options_ordinal(o: IndexOptions) -> u8 {
    match o {
        IndexOptions::None => 0,
        IndexOptions::Docs => 1,
        IndexOptions::DocsAndFreqs | IndexOptions::DocsAndCustomFreqs => 2,
        IndexOptions::DocsAndFreqsAndPositions => 3,
        IndexOptions::DocsAndFreqsAndPositionsAndOffsets => 4,
    }
}

fn doc_values_ordinal(t: DocValuesType) -> u8 {
    match t {
        DocValuesType::None => 0,
        DocValuesType::Numeric => 1,
        DocValuesType::Binary => 2,
        DocValuesType::Sorted => 3,
        DocValuesType::SortedNumeric => 4,
        DocValuesType::SortedSet => 5,
    }
}

/// `SpatialExtrasCorpus.describe`.
fn describe(f: &dyn IndexableField) -> String {
    let t = f.field_type();
    let mut sb = format!("{}|", f.name());
    sb.push(if t.stored() { 's' } else { '-' });
    sb.push(if t.tokenized() { 't' } else { '-' });
    sb.push(if t.omit_norms() { 'n' } else { '-' });
    sb.push_str(&format!(
        "|{}|{}|{}x{}|",
        index_options_ordinal(t.index_options()),
        doc_values_ordinal(t.doc_values_type()),
        t.point_dimension_count(),
        t.point_num_bytes()
    ));
    let plain =
        f.numeric_value().is_none() && f.binary_value().is_none() && f.string_value().is_none();
    if plain && t.index_options() != IndexOptions::None {
        let analyzer = lucene_analysis::Analyzer::standard(None);
        let ts = f.token_stream(&analyzer).unwrap().unwrap();
        let list: Vec<String> = ts.tokens.iter().map(|t| hex(&t.term)).collect();
        sb.push_str(&tokens(&list));
    } else {
        sb.push('-');
    }
    sb.push('|');
    match f.numeric_value() {
        None => sb.push('-'),
        Some(Number::Double(d)) => sb.push_str(&format!("D{}", h(d))),
        Some(Number::Long(l)) => sb.push_str(&format!("L{l:x}")),
        Some(other) => sb.push_str(&format!("{other:?}")),
    }
    sb.push('|');
    match f.binary_value() {
        None => sb.push('-'),
        Some(b) => sb.push_str(&hex(&b)),
    }
    sb.push('|');
    match f.string_value() {
        None => sb.push('-'),
        Some(s) => sb.push_str(&esc(&s)),
    }
    sb
}

/// A failure as the generator writes it: `E`, Java's class and message.
fn err(e: &lucene_search::Error) -> String {
    use lucene_search::Error as E;
    let (class, msg) = match e {
        E::Spatial(s) => (s.java_class().to_string(), s.to_string()),
        E::IllegalArgument(m) | E::DocumentQuery(m) => {
            ("java.lang.IllegalArgumentException".to_string(), m.clone())
        }
        E::IllegalState(m) => ("java.lang.IllegalStateException".to_string(), m.clone()),
        other => ("rust".to_string(), other.to_string()),
    };
    format!("E\t{class}\t{}", esc(&msg))
}

/// An answer as compared: a `ClassCastException`'s message is the JVM's
/// (module and loader names), so only its class is.
fn normalised(answer: &str) -> String {
    if answer.starts_with("E\tjava.lang.ClassCastException") {
        return "E\tjava.lang.ClassCastException".into();
    }
    answer.to_string()
}

fn hex_bits(max_doc: i32, docs: &[i32]) -> String {
    let mut b = vec![0u8; ((max_doc + 7) / 8) as usize];
    for &d in docs {
        b[(d >> 3) as usize] |= 1 << (d & 7);
    }
    while b.last() == Some(&0) {
        b.pop();
    }
    hex(&b)
}

/// One opened index and its answers.
struct Index<'a> {
    segments: Vec<OpenSegment<'a>>,
    reader: &'a DirectoryReader,
}

impl Index<'_> {
    fn mask(&self) -> FixedBitSet {
        let max_doc = self.reader.max_doc();
        let mut bits = FixedBitSet::new(max_doc as usize);
        for (d, id) in self.ids().iter().enumerate() {
            if id % 3 != 0 {
                bits.set(d);
            }
        }
        bits
    }

    /// Every global doc's `id` (stored).
    fn ids(&self) -> Vec<i32> {
        let mut out = Vec::new();
        for r in self.reader.segment_readers() {
            for d in 0..r.max_doc {
                // `id` is each document's first stored field, its only string
                let doc = r.stored_document(d).unwrap().unwrap();
                let id = doc
                    .fields
                    .iter()
                    .find_map(|f| match &f.value {
                        lucene_codecs::stored_fields::FieldValue::String(s) => Some(s.clone()),
                        _ => None,
                    })
                    .unwrap();
                out.push(id.parse().unwrap());
            }
        }
        out
    }
}

fn values(
    segments: &[OpenSegment<'_>],
    src: &Arc<dyn DoubleValuesSource>,
) -> lucene_search::Result<String> {
    let norms = vec![None; segments.len()];
    let searcher = IndexSearcher::new(segments, &norms)?;
    let ctx = ValuesContext::new(&searcher);
    let mut out = Vec::new();
    for (leaf, seg) in segments.iter().enumerate() {
        let mut v = src.get_values(&ctx, leaf, None)?;
        for d in 0..seg.max_doc.unwrap() {
            out.push(if v.advance_exact(d)? {
                h(v.double_value()?)
            } else {
                "-".to_string()
            });
        }
    }
    Ok(format!("V\t{}", out.join(",")))
}

fn answer(c: &Corpus, ix: &Index<'_>, a: &[&str]) -> lucene_search::Result<String> {
    let st = &c.s[a[1]];
    match a[0] {
        "s" => Ok(esc(&st.to_string())),
        "q" => {
            let mut args = SpatialArgs::new(SpatialOperation::get(a[2])?, c.shape(a[3])?);
            if a[4] != "-" {
                args.set_dist_err_pct(Some(a[4].parse().unwrap()));
            }
            let q = st.make_query(&args)?;
            let mut hits = dq::search_all(&ix.segments, q.as_ref())?;
            hits.sort_by_key(|h| h.doc_id);
            let max_doc = ix.reader.max_doc();
            if hits.iter().all(|h| h.score == 1.0) {
                let ids: Vec<i32> = hits.iter().map(|h| h.doc_id).collect();
                return Ok(format!("C\t{}\t{}", ids.len(), hex_bits(max_doc, &ids)));
            }
            let parts: Vec<String> = hits
                .iter()
                .map(|h| format!("{}:{:x}", h.doc_id, h.score.to_bits()))
                .collect();
            Ok(format!("S\t{}\t{}", hits.len(), parts.join(",")))
        }
        "v" => {
            let src: Arc<dyn DoubleValuesSource> = match a[2] {
                "dist" => {
                    let p = point_of(c, a[3]);
                    st.make_distance_value_source(&p, a[4].parse().unwrap())?
                }
                "recip" => make_recip_distance_value_source(&**st, &c.shape(a[3])?)?,
                "overlap" => {
                    let shape = c.shape(a[3])?;
                    let rect = shape.bounding_box()?;
                    Arc::new(BBoxOverlapRatioValueSource::new(
                        c.bb[a[1]].make_shape_value_source(),
                        st.spatial_context().is_geo(),
                        rect,
                        a[4].parse().unwrap(),
                        a[5].parse().unwrap(),
                    )?)
                }
                "overlap2" => {
                    let rect = c.shape(a[3])?.bounding_box()?;
                    c.bb[a[1]].make_overlap_ratio_value_source(rect, a[4].parse().unwrap())?
                }
                "area" => {
                    let src = match c.bb.get(a[1]) {
                        Some(b) => b.make_shape_value_source(),
                        None => c.sdv[a[1]].make_shape_value_source(),
                    };
                    Arc::new(ShapeAreaValueSource::new(
                        src,
                        st.spatial_context().clone(),
                        a[3] == "true",
                        a[4].parse().unwrap(),
                    ))
                }
                "cached" => {
                    let p = point_of(c, a[3]);
                    Arc::new(CachingDoubleValueSource::new(
                        st.make_distance_value_source(&p, a[4].parse().unwrap())?,
                    ))
                }
                other => panic!("unknown source {other}"),
            };
            values(&ix.segments, &src)
        }
        "h" => {
            let shape = if a[2] == "-" {
                None
            } else {
                Some(c.shape(a[2])?)
            };
            let mask = ix.mask();
            let accept = (a[5] == "mask").then_some(&mask);
            let pt = st.as_prefix_tree().expect("a prefix-tree strategy");
            let hm = pt.calc_facets(
                &ix.segments,
                accept,
                shape.as_ref(),
                a[3].parse().unwrap(),
                a[4].parse().unwrap(),
            )?;
            let counts: Vec<String> = hm.counts.iter().map(i32::to_string).collect();
            Ok(format!(
                "H\t{}\t{}\t{}\t{}",
                hm.columns,
                hm.rows,
                esc(&hm.region.to_string()),
                counts.join(",")
            ))
        }
        "f" => {
            let mask = ix.mask();
            let accept = (a[4] == "mask").then_some(&mask);
            let unit = |s: &str| -> lucene_search::Result<UnitNRShape> {
                let shape = c.shape(s)?;
                Ok(shape
                    .as_any()
                    .downcast_ref::<UnitNRShape>()
                    .unwrap()
                    .clone())
            };
            let f =
                c.dr.calc_facets_between(&ix.segments, accept, &unit(a[2])?, &unit(a[3])?)?;
            Ok(format!("F\t{}", esc(&f.to_string())))
        }
        "fr" => {
            let f =
                c.dr.calc_facets(&ix.segments, None, &c.shape(a[2])?, a[3].parse().unwrap())?;
            Ok(format!("F\t{}", esc(&f.to_string())))
        }
        other => panic!("unknown question {other}"),
    }
}

/// A `POINT(x y)` spec's point, made as the WKT reader makes it (`pointXY`
/// of the text's numbers: a Geo3D point keeps its own coordinates, which a
/// round trip through `getX()`/`getY()` would not).
fn point_of(c: &Corpus, spec: &str) -> Arc<dyn lucene_util::spatial4j::Point> {
    let (family, text) = spec.split_once(':').unwrap();
    let inner = text
        .strip_prefix("POINT(")
        .and_then(|t| t.strip_suffix(')'))
        .expect("a POINT");
    let (x, y) = inner.split_once(' ').unwrap();
    c.context_of(family)
        .point_xy(x.parse().unwrap(), y.parse().unwrap())
        .unwrap()
}

fn check_queries(dir: &std::path::Path) -> usize {
    let c = corpus();
    let reader = DirectoryReader::open(&FsDirectory::open(dir)).expect("open");
    let mut opened = reader.open_segments().expect("open segments");
    opened.open_points().expect("open points");
    let segments = opened.as_open_segments();
    assert_eq!(segments.len(), 4, "segments");
    let ix = Index {
        segments,
        reader: &reader,
    };
    let text = read("queries.tsv");
    let mut failures = Vec::new();
    let mut n = 0;
    for line in text.lines() {
        let (lhs, want) = line.split_once("\t=>\t").expect("a record");
        let a: Vec<&str> = lhs.split('\t').collect();
        let got = match answer(&c, &ix, &a) {
            Ok(s) => s,
            Err(e) => err(&e),
        };
        n += 1;
        if normalised(&got) != normalised(want) {
            failures.push(format!(
                "{lhs}\n   java: {}\n   rust: {}",
                want.chars().take(400).collect::<String>(),
                got.chars().take(400).collect::<String>()
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {n} answers differ:\n{}",
        failures.len(),
        failures[..failures.len().min(15)].join("\n")
    );
    n
}

fn write_rust_index(dir: &std::path::Path) {
    let c = corpus();
    let fs = FsDirectory::open(dir);
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w = IndexWriter::open(&fs, Vec::new(), "Lucene104", version).unwrap();
    for (i, line) in read("docs.tsv").lines().enumerate() {
        w.add_fields_document(&c.document(line)).unwrap();
        if matches!(i + 1, 120 | 240 | 360) {
            w.commit().unwrap();
        }
    }
    for id in read("deletes.tsv").lines() {
        w.delete_documents_by_term(&[Term::new("id", id)]).unwrap();
    }
    w.commit().unwrap();
}

#[test]
fn every_spatial_strategy_makes_the_fields_lucene_makes() {
    let c = corpus();
    let docs = read("docs.tsv");
    let fields = read("fields.tsv");
    let mut failures = Vec::new();
    let mut n = 0;
    for (doc, want) in docs.lines().zip(fields.lines()) {
        let mut p = doc.split('\t');
        let mut got = p.next().unwrap().to_string();
        for f in p {
            let (name, spec) = f.split_once('=').unwrap();
            for field in c.fields(name, spec).unwrap() {
                got.push('\t');
                got.push_str(&describe(&*field));
                n += 1;
            }
        }
        if got != want {
            let g: Vec<&str> = got.split('\t').collect();
            let w: Vec<&str> = want.split('\t').collect();
            let first = g
                .iter()
                .zip(&w)
                .position(|(a, b)| a != b)
                .unwrap_or(g.len().min(w.len()));
            failures.push(format!(
                "doc {}: field {first}\n   java: {}\n   rust: {}",
                g[0],
                w.get(first).unwrap_or(&"<none>"),
                g.get(first).unwrap_or(&"<none>")
            ));
        }
    }
    assert!(n > 4000, "only {n} fields");
    assert!(
        failures.is_empty(),
        "{} documents differ:\n{}",
        failures.len(),
        failures[..failures.len().min(10)].join("\n")
    );
    // every strategy indexed something
    let names: Vec<&str> = c.order.iter().map(|(n, _)| *n).collect();
    for name in names.iter().filter(|n| **n != "cmpn") {
        assert!(docs.contains(&format!("\t{name}=")), "{name}");
    }
}

#[test]
fn every_spatial_answer_matches_lucene_on_lucenes_index() {
    let n = check_queries(&root().join("index"));
    assert!(n >= 1200, "{n} answers");
}

#[test]
fn every_spatial_answer_matches_lucene_on_this_ports_index() {
    let tmp = TempDir::new("spatial-strategies-write");
    write_rust_index(tmp.path());
    let n = check_queries(tmp.path());
    assert!(n >= 1200, "{n} answers");
}
