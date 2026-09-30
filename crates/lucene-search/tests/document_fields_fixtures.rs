//! The document package, differentially against Lucene 10.5.0:
//! `fixtures/src/GenDocumentFields.java` built its documents from field
//! specs (`docs.tsv`), recorded what the indexing chain reads from every
//! field (`facets.tsv`), and ran every field-level query, sort and value
//! source over the index it wrote. This builds the same fields with
//! `lucene_index::document`, the same queries with `lucene_search::document`,
//! and requires the same bytes, hits, scores and values -- over Java's index,
//! and over an index this port writes from the same specs.

#![allow(clippy::arithmetic_side_effects)]

use std::collections::BTreeMap;
use std::net::IpAddr;

use lucene_analysis::Analyzer;
use lucene_index::buffered_updates::Term;
use lucene_index::document::*;
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::collector::ScoreDoc;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::document::{self as dq, DocumentQuery};
use lucene_search::multi_segment::OpenSegment;
use lucene_search::query::MatchAllDocsQuery;
use lucene_search::top_field::search_sorted;
use lucene_search::{BooleanQuery, Clause};
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;

fn root(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/data")
        .join(name)
}

fn read(name: &str, file: &str) -> String {
    std::fs::read_to_string(root(name).join(file))
        .expect("run scripts/gen-fixtures.sh --only GenDocumentFields")
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

fn list<T: std::str::FromStr>(s: &str) -> Vec<T>
where
    T::Err: std::fmt::Debug,
{
    s.split(',').map(|v| v.parse().unwrap()).collect()
}

fn store(s: &str) -> Store {
    if s == "Y" {
        Store::Yes
    } else {
        Store::No
    }
}

/// The field type with offsets `GenDocumentFields.CUSTOM` builds.
fn custom_type() -> FieldType {
    let mut ft = FieldType::new();
    ft.set_index_options(IndexOptions::DocsAndFreqsAndPositionsAndOffsets)
        .unwrap();
    ft.set_tokenized(true).unwrap();
    ft.frozen()
}

/// `GenDocumentFields.field(spec)`, with this port's types.
fn field(spec: &str) -> Box<dyn IndexableField> {
    let a: Vec<&str> = spec.split(' ').collect();
    let f = a.get(1).copied().unwrap_or("");
    match a[0] {
        "id" => Box::new(StringField::new("id", a[1], Store::Yes)),
        "text" => Box::new(TextField::new(f, a[3].replace('_', " "), store(a[2]))),
        "custom" => Box::new(Field::from_string(f, a[2].replace('_', " "), custom_type()).unwrap()),
        "stored" => Box::new(Field::stored(
            f,
            match a[2] {
                "int" => StoredValue::Int(a[3].parse().unwrap()),
                "long" => StoredValue::Long(a[3].parse().unwrap()),
                "float" => StoredValue::Float(a[3].parse().unwrap()),
                "double" => StoredValue::Double(a[3].parse().unwrap()),
                "string" => StoredValue::String(a[3].to_string()),
                "bytes" => StoredValue::Binary(unhex(a[3])),
                other => panic!("stored {other}"),
            },
        )),
        "IntField" => Box::new(IntField::new(f, a[2].parse().unwrap(), store(a[3]))),
        "LongField" => Box::new(LongField::new(f, a[2].parse().unwrap(), store(a[3]))),
        "FloatField" => Box::new(FloatField::new(f, a[2].parse().unwrap(), store(a[3]))),
        "DoubleField" => Box::new(DoubleField::new(f, a[2].parse().unwrap(), store(a[3]))),
        "IntPoint" => Box::new(IntPoint::new(f, &list::<i32>(a[2])).unwrap()),
        "LongPoint" => Box::new(LongPoint::new(f, &list::<i64>(a[2])).unwrap()),
        "FloatPoint" => Box::new(FloatPoint::new(f, &list::<f32>(a[2])).unwrap()),
        "DoublePoint" => Box::new(DoublePoint::new(f, &list::<f64>(a[2])).unwrap()),
        "BinaryPoint" => {
            let dims: Vec<Vec<u8>> = a[2].split(',').map(unhex).collect();
            let refs: Vec<&[u8]> = dims.iter().map(Vec::as_slice).collect();
            Box::new(BinaryPoint::new(f, &refs).unwrap())
        }
        "InetAddressPoint" => Box::new(InetAddressPoint::new(f, a[2].parse().unwrap())),
        "IntRange" => Box::new(IntRange::new(f, &list::<i32>(a[2]), &list::<i32>(a[3])).unwrap()),
        "LongRange" => Box::new(LongRange::new(f, &list::<i64>(a[2]), &list::<i64>(a[3])).unwrap()),
        "FloatRange" => {
            Box::new(FloatRange::new(f, &list::<f32>(a[2]), &list::<f32>(a[3])).unwrap())
        }
        "DoubleRange" => {
            Box::new(DoubleRange::new(f, &list::<f64>(a[2]), &list::<f64>(a[3])).unwrap())
        }
        "InetAddressRange" => Box::new(
            InetAddressRange::new(f, a[2].parse().unwrap(), a[3].parse().unwrap()).unwrap(),
        ),
        "IntRangeDocValuesField" => Box::new(
            IntRangeDocValuesField::new(f, &list::<i32>(a[2]), &list::<i32>(a[3])).unwrap(),
        ),
        "LongRangeDocValuesField" => Box::new(
            LongRangeDocValuesField::new(f, &list::<i64>(a[2]), &list::<i64>(a[3])).unwrap(),
        ),
        "FloatRangeDocValuesField" => Box::new(
            FloatRangeDocValuesField::new(f, &list::<f32>(a[2]), &list::<f32>(a[3])).unwrap(),
        ),
        "DoubleRangeDocValuesField" => Box::new(
            DoubleRangeDocValuesField::new(f, &list::<f64>(a[2]), &list::<f64>(a[3])).unwrap(),
        ),
        "KeywordField" => Box::new(KeywordField::new(f, a[2], store(a[3]))),
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
        "BinaryDocValuesField" => Box::new(BinaryDocValuesField::new(f, unhex(a[2]))),
        "FeatureField" => Box::new(FeatureField::new(f, a[2], a[3].parse().unwrap()).unwrap()),
        "LateInteractionField" => {
            let v: Vec<Vec<f32>> = a[2].split(';').map(list::<f32>).collect();
            Box::new(LateInteractionField::new(f, &v).unwrap())
        }
        other => panic!("unknown field spec {other}"),
    }
}

fn skip_name(t: DocValuesSkipIndexType) -> &'static str {
    match t {
        DocValuesSkipIndexType::None => "NONE",
        DocValuesSkipIndexType::Range => "RANGE",
    }
}

/// `GenDocumentFields.facets(field)`.
fn facets(f: &dyn IndexableField, analyzer: &Analyzer) -> String {
    let t = f.field_type();
    let ty = format!(
        "{},{},{},{},{},{},{},{},{},{}",
        t.stored() as u8,
        t.tokenized() as u8,
        index_options_name(t.index_options()),
        t.omit_norms() as u8,
        doc_values_type_name(t.doc_values_type()),
        skip_name(t.doc_values_skip_index_type()),
        t.point_dimension_count(),
        t.point_index_dimension_count(),
        t.point_num_bytes(),
        t.vector_dimension()
    );
    let binary = f.binary_value().map_or("-".to_string(), |b| hex(&b));
    let numeric = match f.numeric_value() {
        None => "-".to_string(),
        Some(Number::Int(v)) => format!("I:{v}"),
        Some(Number::Long(v)) => format!("L:{v}"),
        Some(Number::Float(v)) => format!("F:{:x}", v.to_bits()),
        Some(Number::Double(v)) => format!("D:{:x}", v.to_bits()),
    };
    let stored = if !t.stored() {
        "-".to_string()
    } else {
        match f.stored_value().expect("a stored field has a stored value") {
            StoredValue::Int(v) => format!("int:{v}"),
            StoredValue::Long(v) => format!("long:{v}"),
            StoredValue::Float(v) => format!("float:{:x}", v.to_bits()),
            StoredValue::Double(v) => format!("double:{:x}", v.to_bits()),
            StoredValue::String(s) => format!("string:{}", hex(s.as_bytes())),
            StoredValue::Binary(b) => format!("binary:{}", hex(&b)),
        }
    };
    let indexed = t.index_options() != IndexOptions::None;
    let inv = if !indexed {
        "-"
    } else {
        match f.invertable_type() {
            InvertableType::Binary => "BINARY",
            InvertableType::TokenStream => "TOKEN_STREAM",
        }
    };
    let tokens = if !indexed || f.invertable_type() == InvertableType::Binary {
        "-".to_string()
    } else {
        let ts = f.token_stream(analyzer).unwrap().unwrap();
        let toks: Vec<String> = ts
            .tokens
            .iter()
            .map(|k| {
                format!(
                    "{}/{}/{}/{}/{}",
                    hex(&k.term),
                    k.position_increment,
                    k.start_offset,
                    k.end_offset,
                    k.term_frequency
                )
            })
            .collect();
        format!(
            "{}|{}/{}",
            toks.join(";"),
            ts.final_position_increment,
            ts.final_offset
        )
    };
    format!(
        "{}\t{ty}\t{binary}\t{numeric}\t{stored}\t{inv}\t{tokens}",
        f.name()
    )
}

/// `docs.tsv`: `(id, specs)` in document order.
fn docs() -> Vec<(i32, Vec<String>)> {
    read("document_fields", "docs.tsv")
        .lines()
        .map(|l| {
            let mut p = l.split('\t');
            let id = p.next().unwrap().parse().unwrap();
            (id, p.map(str::to_string).collect())
        })
        .collect()
}

#[test]
fn every_field_indexes_what_lucene_indexes() {
    let docs: BTreeMap<i32, Vec<String>> = docs().into_iter().collect();
    let analyzer = Analyzer::standard(None);
    let mut checked = 0;
    let mut failures = Vec::new();
    for line in read("document_fields", "facets.tsv").lines() {
        let mut p = line.splitn(3, '\t');
        let id: i32 = p.next().unwrap().parse().unwrap();
        let idx: usize = p.next().unwrap().parse().unwrap();
        let want = p.next().unwrap();
        let spec = &docs[&id][idx];
        let got = facets(field(spec).as_ref(), &analyzer);
        if got != want {
            failures.push(format!("{spec}\n  java: {want}\n  rust: {got}"));
        }
        checked += 1;
    }
    assert!(checked > 4000, "{checked} fields");
    assert!(
        failures.is_empty(),
        "{} of {checked} differ:\n{}",
        failures.len(),
        failures[..failures.len().min(20)].join("\n")
    );
}

fn floats(s: &str) -> Vec<f32> {
    list(s)
}
fn doubles(s: &str) -> Vec<f64> {
    list(s)
}
fn ints(s: &str) -> Vec<i32> {
    list(s)
}
fn longs(s: &str) -> Vec<i64> {
    list(s)
}
fn ip(s: &str) -> IpAddr {
    s.parse().unwrap()
}
fn bound(s: &str) -> Option<&[u8]> {
    (s != "*").then_some(s.as_bytes())
}

fn b<Q: DocumentQuery + 'static>(
    q: std::result::Result<Q, lucene_search::Error>,
) -> Box<dyn DocumentQuery> {
    Box::new(q.unwrap())
}

/// `GenDocumentFields.query(spec)`, with this port's factories.
fn query(spec: &str) -> Box<dyn DocumentQuery> {
    let a: Vec<&str> = spec.split(' ').collect();
    let f = a[1];
    let bs = |s: &str| -> Vec<Vec<u8>> { s.split(',').map(|v| v.as_bytes().to_vec()).collect() };
    match a[0] {
        "IntPoint.newRangeQuery" => b(dq::int_point::new_range_query(
            f,
            a[2].parse().unwrap(),
            a[3].parse().unwrap(),
        )),
        "IntPoint.newExactQuery" => b(dq::int_point::new_exact_query(f, a[2].parse().unwrap())),
        "IntPoint.newSetQuery" => b(dq::int_point::new_set_query(f, &ints(a[2]))),
        "IntPoint.newRangeQueryND" => b(dq::int_point::new_range_query_nd(
            f,
            &ints(a[2]),
            &ints(a[3]),
        )),
        "LongPoint.newRangeQuery" => b(dq::long_point::new_range_query(
            f,
            a[2].parse().unwrap(),
            a[3].parse().unwrap(),
        )),
        "LongPoint.newExactQuery" => b(dq::long_point::new_exact_query(f, a[2].parse().unwrap())),
        "LongPoint.newSetQuery" => b(dq::long_point::new_set_query(f, &longs(a[2]))),
        "FloatPoint.newRangeQuery" => b(dq::float_point::new_range_query(
            f,
            a[2].parse().unwrap(),
            a[3].parse().unwrap(),
        )),
        "FloatPoint.newRangeQueryExclusive" => b(dq::float_point::new_range_query(
            f,
            FloatPoint::next_up(a[2].parse().unwrap()),
            FloatPoint::next_down(a[3].parse().unwrap()),
        )),
        "FloatPoint.newExactQuery" => b(dq::float_point::new_exact_query(f, a[2].parse().unwrap())),
        "FloatPoint.newSetQuery" => b(dq::float_point::new_set_query(f, &floats(a[2]))),
        "DoublePoint.newRangeQuery" => b(dq::double_point::new_range_query(
            f,
            a[2].parse().unwrap(),
            a[3].parse().unwrap(),
        )),
        "DoublePoint.newRangeQueryExclusive" => b(dq::double_point::new_range_query(
            f,
            DoublePoint::next_up(a[2].parse().unwrap()),
            DoublePoint::next_down(a[3].parse().unwrap()),
        )),
        "DoublePoint.newExactQuery" => {
            b(dq::double_point::new_exact_query(f, a[2].parse().unwrap()))
        }
        "DoublePoint.newSetQuery" => b(dq::double_point::new_set_query(f, &doubles(a[2]))),
        "DoublePoint.newRangeQueryND" => b(dq::double_point::new_range_query_nd(
            f,
            &doubles(a[2]),
            &doubles(a[3]),
        )),
        "BinaryPoint.newRangeQueryND" => {
            let lo: Vec<Vec<u8>> = a[2].split(',').map(unhex).collect();
            let hi: Vec<Vec<u8>> = a[3].split(',').map(unhex).collect();
            let lo: Vec<&[u8]> = lo.iter().map(Vec::as_slice).collect();
            let hi: Vec<&[u8]> = hi.iter().map(Vec::as_slice).collect();
            b(dq::binary_point::new_range_query_nd(f, &lo, &hi))
        }
        "BinaryPoint.newRangeQuery" => b(dq::binary_point::new_range_query(
            f,
            &unhex(a[2]),
            &unhex(a[3]),
        )),
        "BinaryPoint.newExactQuery" => b(dq::binary_point::new_exact_query(f, &unhex(a[2]))),
        "BinaryPoint.newSetQuery" => {
            let v: Vec<Vec<u8>> = a[2].split(',').map(unhex).collect();
            let v: Vec<&[u8]> = v.iter().map(Vec::as_slice).collect();
            dq::binary_point::new_set_query(f, &v).unwrap()
        }
        "InetAddressPoint.newExactQuery" => b(dq::inet_address_point::new_exact_query(f, ip(a[2]))),
        "InetAddressPoint.newPrefixQuery" => b(dq::inet_address_point::new_prefix_query(
            f,
            ip(a[2]),
            a[3].parse().unwrap(),
        )),
        "InetAddressPoint.newRangeQuery" => b(dq::inet_address_point::new_range_query(
            f,
            ip(a[2]),
            ip(a[3]),
        )),
        "InetAddressPoint.newSetQuery" => {
            let v: Vec<IpAddr> = a[2].split(',').map(ip).collect();
            b(dq::inet_address_point::new_set_query(f, &v))
        }
        "IntField.newRangeQuery" => b(dq::int_field::new_range_query(
            f,
            a[2].parse().unwrap(),
            a[3].parse().unwrap(),
        )),
        "IntField.newExactQuery" => b(dq::int_field::new_exact_query(f, a[2].parse().unwrap())),
        "IntField.newSetQuery" => b(dq::int_field::new_set_query(f, &ints(a[2]))),
        "LongField.newRangeQuery" => b(dq::long_field::new_range_query(
            f,
            a[2].parse().unwrap(),
            a[3].parse().unwrap(),
        )),
        "LongField.newExactQuery" => b(dq::long_field::new_exact_query(f, a[2].parse().unwrap())),
        "LongField.newSetQuery" => b(dq::long_field::new_set_query(f, &longs(a[2]))),
        "FloatField.newRangeQuery" => b(dq::float_field::new_range_query(
            f,
            a[2].parse().unwrap(),
            a[3].parse().unwrap(),
        )),
        "FloatField.newExactQuery" => b(dq::float_field::new_exact_query(f, a[2].parse().unwrap())),
        "FloatField.newSetQuery" => b(dq::float_field::new_set_query(f, &floats(a[2]))),
        "DoubleField.newRangeQuery" => b(dq::double_field::new_range_query(
            f,
            a[2].parse().unwrap(),
            a[3].parse().unwrap(),
        )),
        "DoubleField.newExactQuery" => {
            b(dq::double_field::new_exact_query(f, a[2].parse().unwrap()))
        }
        "DoubleField.newSetQuery" => b(dq::double_field::new_set_query(f, &doubles(a[2]))),
        "IntRange.newIntersectsQuery" => b(dq::int_range::new_intersects_query(
            f,
            &ints(a[2]),
            &ints(a[3]),
        )),
        "IntRange.newWithinQuery" => {
            b(dq::int_range::new_within_query(f, &ints(a[2]), &ints(a[3])))
        }
        "IntRange.newContainsQuery" => b(dq::int_range::new_contains_query(
            f,
            &ints(a[2]),
            &ints(a[3]),
        )),
        "IntRange.newCrossesQuery" => b(dq::int_range::new_crosses_query(
            f,
            &ints(a[2]),
            &ints(a[3]),
        )),
        "LongRange.newIntersectsQuery" => b(dq::long_range::new_intersects_query(
            f,
            &longs(a[2]),
            &longs(a[3]),
        )),
        "LongRange.newWithinQuery" => b(dq::long_range::new_within_query(
            f,
            &longs(a[2]),
            &longs(a[3]),
        )),
        "LongRange.newContainsQuery" => b(dq::long_range::new_contains_query(
            f,
            &longs(a[2]),
            &longs(a[3]),
        )),
        "LongRange.newCrossesQuery" => b(dq::long_range::new_crosses_query(
            f,
            &longs(a[2]),
            &longs(a[3]),
        )),
        "FloatRange.newIntersectsQuery" => b(dq::float_range::new_intersects_query(
            f,
            &floats(a[2]),
            &floats(a[3]),
        )),
        "FloatRange.newWithinQuery" => b(dq::float_range::new_within_query(
            f,
            &floats(a[2]),
            &floats(a[3]),
        )),
        "FloatRange.newContainsQuery" => b(dq::float_range::new_contains_query(
            f,
            &floats(a[2]),
            &floats(a[3]),
        )),
        "FloatRange.newCrossesQuery" => b(dq::float_range::new_crosses_query(
            f,
            &floats(a[2]),
            &floats(a[3]),
        )),
        "DoubleRange.newIntersectsQuery" => b(dq::double_range::new_intersects_query(
            f,
            &doubles(a[2]),
            &doubles(a[3]),
        )),
        "DoubleRange.newWithinQuery" => b(dq::double_range::new_within_query(
            f,
            &doubles(a[2]),
            &doubles(a[3]),
        )),
        "DoubleRange.newContainsQuery" => b(dq::double_range::new_contains_query(
            f,
            &doubles(a[2]),
            &doubles(a[3]),
        )),
        "DoubleRange.newCrossesQuery" => b(dq::double_range::new_crosses_query(
            f,
            &doubles(a[2]),
            &doubles(a[3]),
        )),
        "InetAddressRange.newIntersectsQuery" => b(dq::inet_address_range::new_intersects_query(
            f,
            ip(a[2]),
            ip(a[3]),
        )),
        "InetAddressRange.newWithinQuery" => b(dq::inet_address_range::new_within_query(
            f,
            ip(a[2]),
            ip(a[3]),
        )),
        "InetAddressRange.newContainsQuery" => b(dq::inet_address_range::new_contains_query(
            f,
            ip(a[2]),
            ip(a[3]),
        )),
        "InetAddressRange.newCrossesQuery" => b(dq::inet_address_range::new_crosses_query(
            f,
            ip(a[2]),
            ip(a[3]),
        )),
        "IntRangeDocValuesField.newSlowIntersectsQuery" => b(
            dq::int_range_doc_values_field::new_slow_intersects_query(f, &ints(a[2]), &ints(a[3])),
        ),
        "LongRangeDocValuesField.newSlowIntersectsQuery" => {
            b(dq::long_range_doc_values_field::new_slow_intersects_query(
                f,
                &longs(a[2]),
                &longs(a[3]),
            ))
        }
        "FloatRangeDocValuesField.newSlowIntersectsQuery" => {
            b(dq::float_range_doc_values_field::new_slow_intersects_query(
                f,
                &floats(a[2]),
                &floats(a[3]),
            ))
        }
        "DoubleRangeDocValuesField.newSlowIntersectsQuery" => b(
            dq::double_range_doc_values_field::new_slow_intersects_query(
                f,
                &doubles(a[2]),
                &doubles(a[3]),
            ),
        ),
        "NumericDocValuesField.newSlowRangeQuery" => {
            Box::new(dq::numeric_doc_values_field::new_slow_range_query(
                f,
                a[2].parse().unwrap(),
                a[3].parse().unwrap(),
            ))
        }
        "NumericDocValuesField.newSlowExactQuery" => Box::new(
            dq::numeric_doc_values_field::new_slow_exact_query(f, a[2].parse().unwrap()),
        ),
        "NumericDocValuesField.newSlowSetQuery" => Box::new(
            dq::numeric_doc_values_field::new_slow_set_query(f, &longs(a[2])),
        ),
        "SortedNumericDocValuesField.newSlowRangeQuery" => {
            Box::new(dq::sorted_numeric_doc_values_field::new_slow_range_query(
                f,
                a[2].parse().unwrap(),
                a[3].parse().unwrap(),
            ))
        }
        "SortedNumericDocValuesField.newSlowExactQuery" => Box::new(
            dq::sorted_numeric_doc_values_field::new_slow_exact_query(f, a[2].parse().unwrap()),
        ),
        "SortedNumericDocValuesField.newSlowSetQuery" => Box::new(
            dq::sorted_numeric_doc_values_field::new_slow_set_query(f, &longs(a[2])),
        ),
        "SortedDocValuesField.newSlowRangeQuery" => {
            Box::new(dq::sorted_doc_values_field::new_slow_range_query(
                f,
                bound(a[2]),
                bound(a[3]),
                a[4] == "true",
                a[5] == "true",
            ))
        }
        "SortedDocValuesField.newSlowExactQuery" => Box::new(
            dq::sorted_doc_values_field::new_slow_exact_query(f, a[2].as_bytes()),
        ),
        "SortedDocValuesField.newSlowSetQuery" => {
            let v = bs(a[2]);
            let v: Vec<&[u8]> = v.iter().map(Vec::as_slice).collect();
            Box::new(dq::sorted_doc_values_field::new_slow_set_query(f, &v))
        }
        "SortedSetDocValuesField.newSlowRangeQuery" => {
            Box::new(dq::sorted_set_doc_values_field::new_slow_range_query(
                f,
                bound(a[2]),
                bound(a[3]),
                a[4] == "true",
                a[5] == "true",
            ))
        }
        "SortedSetDocValuesField.newSlowExactQuery" => Box::new(
            dq::sorted_set_doc_values_field::new_slow_exact_query(f, a[2].as_bytes()),
        ),
        "SortedSetDocValuesField.newSlowSetQuery" => {
            let v = bs(a[2]);
            let v: Vec<&[u8]> = v.iter().map(Vec::as_slice).collect();
            Box::new(dq::sorted_set_doc_values_field::new_slow_set_query(f, &v))
        }
        "KeywordField.newExactQuery" => {
            Box::new(dq::keyword_field::new_exact_query(f, a[2].as_bytes()))
        }
        "KeywordField.newSetQuery" => {
            let v = bs(a[2]);
            let v: Vec<&[u8]> = v.iter().map(Vec::as_slice).collect();
            Box::new(dq::keyword_field::new_set_query(f, &v))
        }
        "FeatureField.newLinearQuery" => {
            dq::feature_field::new_linear_query(f, a[2], a[3].parse().unwrap()).unwrap()
        }
        "FeatureField.newLogQuery" => {
            dq::feature_field::new_log_query(f, a[2], a[3].parse().unwrap(), a[4].parse().unwrap())
                .unwrap()
        }
        "FeatureField.newSaturationQuery" => dq::feature_field::new_saturation_query(
            f,
            a[2],
            a[3].parse().unwrap(),
            a[4].parse().unwrap(),
        )
        .unwrap(),
        "FeatureField.newSaturationQueryAuto" => {
            dq::feature_field::new_saturation_query_auto(f, a[2]).unwrap()
        }
        "FeatureField.newSigmoidQuery" => dq::feature_field::new_sigmoid_query(
            f,
            a[2],
            a[3].parse().unwrap(),
            a[4].parse().unwrap(),
            a[5].parse().unwrap(),
        )
        .unwrap(),
        "LongField.newDistanceFeatureQuery" => dq::long_field::new_distance_feature_query(
            f,
            a[2].parse().unwrap(),
            a[3].parse().unwrap(),
            a[4].parse().unwrap(),
        )
        .unwrap(),
        other => panic!("unknown query {other}"),
    }
}

/// `GenDocumentFields.hits(searcher, query)`.
fn hits(segments: &[OpenSegment<'_>], q: &dyn DocumentQuery) -> String {
    let td = dq::search_top_docs(segments, q, 100_000).unwrap();
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
    out
}

fn check_queries(name: &str, dir: &std::path::Path) -> usize {
    let reader = DirectoryReader::open(&FsDirectory::open(dir)).expect("open");
    let opened = reader.open_segments().expect("open postings");
    let segments = opened.as_open_segments();
    let mut failures = Vec::new();
    let mut n = 0;
    for line in read(name, "queries.tsv").lines() {
        let (spec, want) = line.split_once('\t').unwrap();
        let got = hits(&segments, query(spec).as_ref());
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
        "{name}: {} of {n} queries differ:\n{}",
        failures.len(),
        failures.join("\n")
    );
    n
}

#[test]
fn every_query_matches_lucene_on_lucenes_index() {
    let n = check_queries("document_fields", &root("document_fields").join("index"));
    assert!(n > 100, "{n} queries");
}

#[test]
fn index_sorted_skipper_queries_match_lucene() {
    for name in ["document_fields_sorted_num", "document_fields_sorted_kw"] {
        let n = check_queries(name, &root(name).join("index"));
        assert!(n >= 10, "{name}: {n} queries");
    }
}

/// The same documents, written by this port: every commit and delete
/// `GenDocumentFields.main` made, in order.
fn write_rust_index(dir: &std::path::Path) {
    let fs = FsDirectory::open(dir);
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w = IndexWriter::open(&fs, Vec::new(), "Lucene104", version).unwrap();
    for (i, (_, specs)) in docs().into_iter().enumerate() {
        let mut doc = Document::new();
        for s in &specs {
            doc.add_boxed(field(s));
        }
        w.add_fields_document(&doc).unwrap();
        if (i + 1) % 300 == 0 {
            w.commit().unwrap();
        }
    }
    let mut terms = Vec::new();
    for i in (0..300).step_by(17) {
        terms.push(Term::new("id", i.to_string()));
        terms.push(Term::new("id", (600 + i).to_string()));
    }
    for t in terms {
        w.delete_documents_by_term(&[t]).unwrap();
    }
    w.commit().unwrap();
}

#[test]
fn every_query_matches_lucene_on_this_ports_index() {
    let tmp = TempDir::new("document-fields-write");
    write_rust_index(tmp.path());
    let n = check_queries("document_fields", tmp.path());
    assert!(n > 100, "{n} queries");
}

fn match_all() -> BooleanQuery {
    BooleanQuery {
        must: vec![Clause::MatchAllDocs(MatchAllDocsQuery::new(i32::MAX))],
        ..BooleanQuery::default()
    }
}

#[test]
fn sort_fields_order_as_lucene_orders() {
    let dir = root("document_fields").join("index");
    let reader = DirectoryReader::open(&FsDirectory::open(&dir)).expect("open");
    let mut opened = reader.open_segments().expect("open postings");
    opened.open_points().expect("open points");
    let segments = opened.as_open_segments();
    let norms: Vec<
        Option<&std::collections::HashMap<String, lucene_search::field_norms::FieldNorms<'_>>>,
    > = segments.iter().map(|_| None).collect();
    let mut failures = Vec::new();
    for line in read("document_fields", "sorts.tsv").lines() {
        let mut p = line.split('\t');
        let spec = p.next().unwrap();
        let total: u64 = p.next().unwrap().parse().unwrap();
        let want: Vec<String> = p.map(str::to_string).collect();
        let a: Vec<&str> = spec.split(' ').collect();
        let got: Vec<String> = if a[0] == "FeatureField" {
            dq::feature_field::new_feature_sort(a[1], a[2])
                .search(&segments, &dq::MatchAllDocs, 40)
                .unwrap()
                .into_iter()
                .map(|(d, v)| format!("{d}:F:{:x}", v.to_bits()))
                .collect()
        } else {
            let reverse = a[2] == "true";
            let missing = a.get(4);
            let sort = match a[0] {
                "KeywordField" => dq::keyword_field::new_sort_field(
                    a[1],
                    reverse,
                    if a[3] == "MIN" {
                        dq::SortedSetSelector::Min
                    } else {
                        dq::SortedSetSelector::Max
                    },
                ),
                kind => {
                    let sel = if a[3] == "MIN" {
                        dq::NumericSelector::Min
                    } else {
                        dq::NumericSelector::Max
                    };
                    match (kind, missing) {
                        ("IntField", None) => dq::int_field::new_sort_field(a[1], reverse, sel),
                        ("IntField", Some(m)) => dq::int_field::new_sort_field_with_missing(
                            a[1],
                            reverse,
                            sel,
                            m.parse().unwrap(),
                        ),
                        ("LongField", None) => dq::long_field::new_sort_field(a[1], reverse, sel),
                        ("LongField", Some(m)) => dq::long_field::new_sort_field_with_missing(
                            a[1],
                            reverse,
                            sel,
                            m.parse().unwrap(),
                        ),
                        ("FloatField", None) => dq::float_field::new_sort_field(a[1], reverse, sel),
                        ("FloatField", Some(m)) => dq::float_field::new_sort_field_with_missing(
                            a[1],
                            reverse,
                            sel,
                            m.parse().unwrap(),
                        ),
                        ("DoubleField", None) => {
                            dq::double_field::new_sort_field(a[1], reverse, sel)
                        }
                        ("DoubleField", Some(m)) => dq::double_field::new_sort_field_with_missing(
                            a[1],
                            reverse,
                            sel,
                            m.parse().unwrap(),
                        ),
                        other => panic!("sort {other:?}"),
                    }
                }
            };
            let ty = sort.ty;
            let td = search_sorted(
                &segments,
                reader.segment_readers(),
                &match_all(),
                &norms,
                &[sort],
                40,
                u64::MAX,
                None,
            )
            .unwrap();
            assert_eq!(td.total.value, total, "{spec}");
            td.hits
                .iter()
                .map(|h| {
                    use lucene_search::top_field::SortType;
                    let v = h.values[0];
                    let value = match ty {
                        SortType::Int => format!("I:{v}"),
                        SortType::Long => format!("L:{v}"),
                        SortType::Float => {
                            format!("F:{:x}", sortable_int_to_float(v as i32).to_bits())
                        }
                        SortType::Double => format!("D:{:x}", sortable_long_to_double(v).to_bits()),
                        _ => match &h.terms[0] {
                            Some(t) => format!("B:{}", hex(t)),
                            None => "null".to_string(),
                        },
                    };
                    format!("{}:{value}", h.doc)
                })
                .collect()
        };
        if got != want {
            failures.push(format!("{spec}\n  java: {want:?}\n  rust: {got:?}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn feature_values_match_lucene() {
    let dir = root("document_fields").join("index");
    let reader = DirectoryReader::open(&FsDirectory::open(&dir)).expect("open");
    let opened = reader.open_segments().expect("open postings");
    let segments = opened.as_open_segments();
    let mut got = String::new();
    for feature in ["pagerank", "popularity", "freshness", "missing"] {
        let src = dq::feature_field::new_double_values("feat", feature);
        assert!(!src.needs_scores());
        for leaf in &segments {
            let mut values = src.get_values(leaf).unwrap();
            let max_doc = leaf.max_doc.unwrap();
            for d in 0..max_doc {
                if values.advance_exact(d) {
                    got.push_str(&format!(
                        "{}\t{feature}\t{:x}\n",
                        leaf.doc_base + d,
                        values.double_value().to_bits()
                    ));
                }
            }
        }
    }
    assert_eq!(got, read("document_fields", "values.tsv"));
}

#[test]
fn date_tools_matches_lucene() {
    let res = |s: &str| match s {
        "year" => Resolution::Year,
        "month" => Resolution::Month,
        "day" => Resolution::Day,
        "hour" => Resolution::Hour,
        "minute" => Resolution::Minute,
        "second" => Resolution::Second,
        "millisecond" => Resolution::Millisecond,
        other => panic!("{other}"),
    };
    let mut failures = Vec::new();
    for line in read("document_fields", "dates.tsv").lines() {
        let c: Vec<&str> = line.split('\t').collect();
        match c[0] {
            "t" => {
                let t: i64 = c[1].parse().unwrap();
                let r = res(c[2]);
                let s = DateTools::time_to_string(t, r);
                let rounded = DateTools::round(t, r);
                if s != c[3] || rounded.to_string() != c[4] {
                    failures.push(format!("{line} -> {s} {rounded}"));
                }
            }
            "s" => {
                let got =
                    DateTools::string_to_time(c[1]).map_or("ERR".to_string(), |t| t.to_string());
                if got != c[2] {
                    failures.push(format!("{line} -> {got}"));
                }
            }
            other => panic!("{other}"),
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Everything an index holds, field by field, in a comparable text form:
/// field infos, live docs, stored fields, doc values, points, postings
/// (with positions and offsets), norms and vectors.
fn dump(dir: &std::path::Path) -> Vec<String> {
    use lucene_codecs::doc_values as dv;
    use lucene_codecs::field_infos::DocValuesType as Dvt;
    let fs = FsDirectory::open(dir);
    let reader = DirectoryReader::open(&fs).expect("open");
    let opened = reader.open_segments().expect("open postings");
    let segments = opened.as_open_segments();
    let mut out = Vec::new();
    for (si, (r, seg)) in reader.segment_readers().iter().zip(&segments).enumerate() {
        out.push(format!("segment {si} max_doc {}", r.max_doc));
        let mut infos = r.field_infos().fields.clone();
        infos.sort_by(|a, b| a.name.cmp(&b.name));
        for f in &infos {
            out.push(format!(
                "fi {} #{} {:?} omit={} {:?} {:?} points={}/{}/{} vec={}/{:?}/{:?}",
                f.name,
                f.number,
                f.index_options,
                f.omit_norms,
                f.doc_values_type,
                f.doc_values_skip_index_type,
                f.point_dimension_count,
                f.point_index_dimension_count,
                f.point_num_bytes,
                f.vector_dimension,
                f.vector_encoding,
                f.vector_similarity_function
            ));
        }
        let name_of = |n: i32| infos.iter().find(|f| f.number == n).unwrap().name.clone();
        for doc in 0..r.max_doc {
            let live = r.live_docs().is_none_or(|b| b.get_doc(doc));
            let stored = r
                .stored_document(doc)
                .unwrap()
                .map(|d| {
                    d.fields
                        .iter()
                        .map(|f| format!("{}={:?}", name_of(f.field_number), f.value))
                        .collect::<Vec<_>>()
                        .join(" ")
                })
                .unwrap_or_default();
            out.push(format!("doc {doc} live={live} {stored}"));
        }
        for f in &infos {
            if f.doc_values_type == Dvt::None {
                continue;
            }
            let Some((meta, data)) = r.doc_values_for_field(f.number) else {
                out.push(format!("dv {} missing", f.name));
                continue;
            };
            let term = |terms: &lucene_codecs::terms_dict::TermsDictEntry, ord: i64| {
                let mut d = lucene_codecs::terms_dict::TermsDict::open(data, terms).unwrap();
                hex(d.seek_ord(ord).unwrap())
            };
            for doc in 0..r.max_doc {
                let v = match f.doc_values_type {
                    Dvt::Numeric => format!(
                        "{:?}",
                        dv::numeric_value(data, meta.numeric_entry(f.number).unwrap(), doc)
                            .unwrap()
                    ),
                    Dvt::Binary => format!(
                        "{:?}",
                        dv::binary_value(data, meta.binary_entry(f.number).unwrap(), doc)
                            .unwrap()
                            .map(hex)
                    ),
                    Dvt::Sorted => {
                        let e = meta.sorted_entry(f.number).unwrap();
                        format!(
                            "{:?}",
                            dv::sorted_ord(data, e, doc)
                                .unwrap()
                                .map(|o| term(&e.terms, o))
                        )
                    }
                    Dvt::SortedNumeric => format!(
                        "{:?}",
                        dv::sorted_numeric_values(
                            data,
                            meta.sorted_numeric_entry(f.number).unwrap(),
                            doc
                        )
                        .unwrap()
                    ),
                    Dvt::SortedSet => {
                        let e = meta.sorted_set_entry(f.number).unwrap();
                        let (ords, terms): (Vec<i64>, _) = match &e.kind {
                            dv::SortedSetKind::Single(s) => (
                                dv::sorted_ord(data, s, doc).unwrap().into_iter().collect(),
                                &s.terms,
                            ),
                            dv::SortedSetKind::Multi { ords, terms } => {
                                (dv::sorted_numeric_values(data, ords, doc).unwrap(), terms)
                            }
                        };
                        format!(
                            "{:?}",
                            ords.iter().map(|&o| term(terms, o)).collect::<Vec<_>>()
                        )
                    }
                    Dvt::None => unreachable!(),
                };
                out.push(format!("dv {} {doc} {v}", f.name));
            }
            out.push(format!(
                "dv-skip {} {:?}",
                f.name,
                r.doc_values_skip_index(f.number).unwrap().map(|s| (
                    s.min_value,
                    s.max_value,
                    s.doc_count
                ))
            ));
        }
        let points = r.points_reader().unwrap();
        for f in &infos {
            if f.point_dimension_count == 0 {
                continue;
            }
            let mut pts: Vec<(i32, String)> = points
                .decode_all_points(f.number)
                .unwrap()
                .into_iter()
                .map(|p| (p.doc_id, hex(&p.packed_value)))
                .collect();
            pts.sort();
            out.push(format!("points {} {pts:?}", f.name));
        }
        let mut names: Vec<(&str, &lucene_codecs::blocktree::FieldTerms)> =
            seg.fields.iter_fields().collect();
        names.sort_by(|a, b| a.0.cmp(b.0));
        for (name, terms) in names {
            let mut it = terms.iter();
            let mut all = Vec::new();
            while let Some((t, stats)) = it.next() {
                all.push((t.to_vec(), stats));
            }
            for (t, stats) in all {
                let p = terms.postings(&t, seg.doc_in).unwrap().unwrap();
                let pos = match seg.pos_in {
                    Some(pos_in)
                        if infos
                            .iter()
                            .any(|f| f.name == name && f.index_options.subsumes_positions()) =>
                    {
                        terms
                            .positions(&t, seg.doc_in, pos_in, seg.pay_in)
                            .unwrap()
                            .map(|p| format!("{p:?}"))
                            .unwrap_or_default()
                    }
                    _ => String::new(),
                };
                out.push(format!(
                    "term {name}:{} df={} ttf={} docs={:?} freqs={:?} {pos}",
                    hex(&t),
                    stats.doc_freq,
                    stats.total_term_freq,
                    p.docs,
                    p.freqs
                ));
            }
        }
        for f in &infos {
            if let Some(n) = r.field_norms(&f.name) {
                let mut c = n.cursor();
                let norms: Vec<Option<i64>> =
                    (0..r.max_doc).map(|d| c.norm_long(d).unwrap()).collect();
                out.push(format!("norms {} {norms:?}", f.name));
            }
        }
        // Vectors: the flat `.vemf`/`.vec` pair.
        let suffix = "Lucene99HnswVectorsFormat_0";
        if let (Ok(meta), Ok(data)) = (
            std::fs::read(dir.join(format!("{}_{suffix}.vemf", r.segment_name))),
            std::fs::read(dir.join(format!("{}_{suffix}.vec", r.segment_name))),
        ) {
            let flat = lucene_codecs::vectors::FlatVectorsReader::open(
                &meta,
                &data,
                &r.segment_id(),
                suffix,
            )
            .unwrap();
            for f in &infos {
                if f.vector_dimension == 0 {
                    continue;
                }
                let v = flat.float_vector_values(f.number).unwrap();
                let all: Vec<(i32, Vec<f32>)> = (0..v.size())
                    .map(|o| (v.ord_to_doc(o).unwrap(), v.vector(o).unwrap()))
                    .collect();
                out.push(format!("vectors {} {all:?}", f.name));
            }
        }
    }
    out
}

fn assert_same_index(java: &std::path::Path, rust: &std::path::Path) {
    let (a, b) = (dump(java), dump(rust));
    let cut = |s: &String| s.chars().take(400).collect::<String>();
    let diffs: Vec<String> = a
        .iter()
        .zip(&b)
        .filter(|(x, y)| x != y)
        .take(20)
        .map(|(x, y)| format!("  java: {}\n  rust: {}", cut(x), cut(y)))
        .collect();
    assert!(
        diffs.is_empty() && a.len() == b.len(),
        "{} vs {} lines; first differences:\n{}",
        a.len(),
        b.len(),
        diffs.join("\n")
    );
    assert!(a.len() > 50, "{} lines: nothing compared", a.len());
}

#[test]
fn this_ports_index_holds_what_lucenes_holds() {
    let tmp = TempDir::new("document-fields-dump");
    write_rust_index(tmp.path());
    assert_same_index(&root("document_fields").join("index"), tmp.path());
}

// ---------------------------------------------------------------------------
// The columnar batch API.

use lucene_index::document::column::{
    BatchColumn, BytesRefValuesCursor, Column, ColumnBatch, Density, LongTupleCursor,
    LongValuesCursor, NumericKind, ObjectTupleCursor, OrdinalsCursor, OrdinalsTupleCursor,
    StoredType, VecBytesValuesCursor, VecLongTupleCursor, VecLongValuesCursor,
    VecObjectTupleCursor, VecOrdinalsCursor, VecOrdinalsTupleCursor,
};

/// `GenDocumentColumns.type(recipe)`.
fn column_type(recipe: &str) -> FieldType {
    let mut t = FieldType::new();
    match recipe {
        "int_point_sndv_stored" | "float_point_sndv_stored" => {
            t.set_dimensions(1, 4).unwrap();
            t.set_doc_values_type(DocValuesType::SortedNumeric).unwrap();
            t.set_stored(true).unwrap();
        }
        "numeric_dv_skip" => {
            t.set_doc_values_type(DocValuesType::Numeric).unwrap();
            t.set_doc_values_skip_index_type(DocValuesSkipIndexType::Range)
                .unwrap();
        }
        "long_point_stored" => {
            t.set_dimensions(1, 8).unwrap();
            t.set_stored(true).unwrap();
        }
        "keyword_stored" => {
            t.set_index_options(IndexOptions::Docs).unwrap();
            t.set_omit_norms(true).unwrap();
            t.set_tokenized(false).unwrap();
            t.set_doc_values_type(DocValuesType::SortedSet).unwrap();
            t.set_stored(true).unwrap();
        }
        "binary_dv" => t.set_doc_values_type(DocValuesType::Binary).unwrap(),
        "point_2x2" => t.set_dimensions(2, 2).unwrap(),
        "text_stored" => {
            t.set_index_options(IndexOptions::DocsAndFreqsAndPositions)
                .unwrap();
            t.set_stored(true).unwrap();
        }
        "sorted_dv" => t.set_doc_values_type(DocValuesType::Sorted).unwrap(),
        "sorted_set_dv" => t.set_doc_values_type(DocValuesType::SortedSet).unwrap(),
        "text_offsets" => t
            .set_index_options(IndexOptions::DocsAndFreqsAndPositionsAndOffsets)
            .unwrap(),
        "vector3" => t
            .set_vector_attributes(
                3,
                VectorEncoding::Float32,
                VectorSimilarityFunction::Euclidean,
            )
            .unwrap(),
        other => panic!("recipe {other}"),
    }
    t.frozen()
}

struct TLong {
    col: Column,
    kind: NumericKind,
    cells: Vec<(i32, i64)>,
    dense: Vec<i64>,
}
impl lucene_index::document::column::LongColumn for TLong {
    fn column(&self) -> &Column {
        &self.col
    }
    fn numeric_kind(&self) -> NumericKind {
        self.kind
    }
    fn tuples(&self) -> Box<dyn LongTupleCursor + '_> {
        Box::new(VecLongTupleCursor::new(&self.cells))
    }
    fn values(&self) -> lucene_index::document::Result<Box<dyn LongValuesCursor + '_>> {
        Ok(Box::new(VecLongValuesCursor::new(&self.dense)))
    }
}

struct TBinary {
    col: Column,
    stored: StoredType,
    cells: Vec<(i32, Vec<u8>)>,
    dense: Vec<Vec<u8>>,
}
impl lucene_index::document::column::BinaryColumn for TBinary {
    fn column(&self) -> &Column {
        &self.col
    }
    fn stored_type(&self) -> StoredType {
        self.stored
    }
    fn tuples(&self) -> Box<dyn ObjectTupleCursor<Vec<u8>> + '_> {
        Box::new(VecObjectTupleCursor::new(&self.cells))
    }
    fn values(&self) -> lucene_index::document::Result<Box<dyn BytesRefValuesCursor + '_>> {
        Ok(Box::new(VecBytesValuesCursor::new(&self.dense)))
    }
}

struct TDict {
    col: Column,
    dict: Vec<Vec<u8>>,
    cells: Vec<(i32, i32)>,
    dense: Vec<i32>,
}
impl lucene_index::document::column::DictionaryColumn for TDict {
    fn column(&self) -> &Column {
        &self.col
    }
    fn dictionary(&self) -> &[Vec<u8>] {
        &self.dict
    }
    fn tuples(&self) -> Box<dyn OrdinalsTupleCursor + '_> {
        Box::new(VecOrdinalsTupleCursor::new(&self.cells))
    }
    fn values(&self) -> lucene_index::document::Result<Box<dyn OrdinalsCursor + '_>> {
        Ok(Box::new(VecOrdinalsCursor::new(&self.dense)))
    }
}

struct TTokens {
    col: Column,
    cells: Vec<(i32, FieldTokens)>,
}
impl lucene_index::document::column::TokenStreamColumn for TTokens {
    fn column(&self) -> &Column {
        &self.col
    }
    fn tuples(&self) -> Box<dyn ObjectTupleCursor<FieldTokens> + '_> {
        Box::new(VecObjectTupleCursor::new(&self.cells))
    }
}

struct TVector {
    col: Column,
    cells: Vec<(i32, Vec<f32>)>,
}
impl lucene_index::document::column::VectorColumn<Vec<f32>> for TVector {
    fn column(&self) -> &Column {
        &self.col
    }
    fn tuples(&self) -> Box<dyn ObjectTupleCursor<Vec<f32>> + '_> {
        Box::new(VecObjectTupleCursor::new(&self.cells))
    }
}

enum TCol {
    L(TLong),
    B(TBinary),
    D(TDict),
    T(TTokens),
    V(TVector),
}

struct TBatch {
    num_docs: usize,
    cols: Vec<TCol>,
}

impl ColumnBatch for TBatch {
    fn num_docs(&self) -> usize {
        self.num_docs
    }
    fn columns(&self) -> Vec<BatchColumn<'_>> {
        self.cols
            .iter()
            .map(|c| match c {
                TCol::L(c) => BatchColumn::Long(c),
                TCol::B(c) => BatchColumn::Binary(c),
                TCol::D(c) => BatchColumn::Dictionary(c),
                TCol::T(c) => BatchColumn::TokenStream(c),
                TCol::V(c) => BatchColumn::FloatVector(c),
            })
            .collect()
    }
}

/// `columns.tsv`, as batches.
fn batches() -> Vec<TBatch> {
    let analyzer = Analyzer::standard(None);
    let mut out: Vec<TBatch> = Vec::new();
    for line in read("document_columns", "columns.tsv").lines() {
        let c: Vec<&str> = line.split('\t').collect();
        if c[0] == "batch" {
            out.push(TBatch {
                num_docs: c[1].parse().unwrap(),
                cols: Vec::new(),
            });
            continue;
        }
        let density = if c[2] == "DENSE" {
            Density::Dense
        } else {
            Density::Sparse
        };
        let col = Column::new(c[1], column_type(c[3]), density);
        let cells: Vec<(i32, &str)> = c[5..]
            .iter()
            .map(|cell| {
                let (d, v) = cell.split_once(':').unwrap();
                (d.parse().unwrap(), v)
            })
            .collect();
        let t = match c[0] {
            "L" => {
                let cells: Vec<(i32, i64)> = cells
                    .iter()
                    .map(|(d, v)| (*d, v.parse().unwrap()))
                    .collect();
                TCol::L(TLong {
                    col,
                    kind: match c[4] {
                        "INT" => NumericKind::Int,
                        "LONG" => NumericKind::Long,
                        "FLOAT" => NumericKind::Float,
                        _ => NumericKind::Double,
                    },
                    dense: cells.iter().map(|c| c.1).collect(),
                    cells,
                })
            }
            "B" => {
                let cells: Vec<(i32, Vec<u8>)> =
                    cells.iter().map(|(d, v)| (*d, unhex(v))).collect();
                TCol::B(TBinary {
                    col,
                    stored: if c[4] == "STRING" {
                        StoredType::String
                    } else {
                        StoredType::Binary
                    },
                    dense: cells.iter().map(|c| c.1.clone()).collect(),
                    cells,
                })
            }
            "D" => {
                let cells: Vec<(i32, i32)> = cells
                    .iter()
                    .map(|(d, v)| (*d, v.parse().unwrap()))
                    .collect();
                TCol::D(TDict {
                    col,
                    dict: c[4].split(',').map(unhex).collect(),
                    dense: cells.iter().map(|c| c.1).collect(),
                    cells,
                })
            }
            "T" => TCol::T(TTokens {
                col,
                cells: cells
                    .iter()
                    .map(|(d, v)| (*d, analyzer.analyze_stream(&v.replace('_', " ")).into()))
                    .collect(),
            }),
            "V" => TCol::V(TVector {
                col,
                cells: cells.iter().map(|(d, v)| (*d, list::<f32>(v))).collect(),
            }),
            other => panic!("column kind {other}"),
        };
        out.last_mut().unwrap().cols.push(t);
    }
    out
}

#[test]
fn a_column_batch_indexes_what_lucenes_add_batch_indexes() {
    let tmp = TempDir::new("document-columns");
    {
        let fs = FsDirectory::open(tmp.path());
        let version = LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        };
        let mut w = IndexWriter::open(&fs, Vec::new(), "Lucene104", version).unwrap();
        for b in batches() {
            w.add_batch(&b).unwrap();
            w.commit().unwrap();
        }
    }
    assert_same_index(&root("document_columns").join("index"), tmp.path());
}
