//! Index-level SimpleText routing over real Java flush/update/delete and
//! compound/merge indexes from `GenCodecsSimpleTextCodec.java`.

#![allow(clippy::arithmetic_side_effects)]

use std::fs;
use std::path::PathBuf;

use lucene_search::directory_reader::DirectoryReader;
use lucene_search::reader::{
    BinaryDocValues, DocValuesIterator, NumericDocValues, SortedDocValues, SortedNumericDocValues,
    SortedSetDocValues,
};
use lucene_store::FsDirectory;
use std::collections::BTreeMap;

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn unhex(value: &str) -> Vec<u8> {
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

fn fixture(variant: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/data/codecs_simpletext/codec")
        .join(variant)
}

#[test]
fn simpletext_field_existence_matches_java_sources_counts_and_rewrites() {
    use lucene_search::directory_reader::ExistsDocs;
    use lucene_search::weight_count::{
        count_field_exists_leaf, field_exists_rewrites_to_match_all_docs,
    };

    let mut checked = 0;
    let mut vectors = 0;
    let mut deleted_matches = 0;
    let data = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data");
    for variant in ["flush", "merged", "clean", "deleted"] {
        let focused = matches!(variant, "clean" | "deleted");
        let root = if focused {
            data.join("simpletext_field_exists").join(variant)
        } else {
            fixture(variant)
        };
        let reader = DirectoryReader::open(&FsDirectory::open(root.join("index"))).unwrap();
        for segment in reader.segment_readers() {
            let path = if focused {
                root.join("exists.tsv")
            } else {
                data.join("simpletext_directory_reader")
                    .join(variant)
                    .join(format!("exists.{}.tsv", segment.segment_name))
            };
            let manifest = fs::read_to_string(path).unwrap();
            let mut lines = manifest.lines();
            while let Some(line) = lines.next() {
                let row: Vec<_> = line.split('\t').collect();
                let field = String::from_utf8(unhex(row[1])).unwrap();
                let context = format!("{variant} {} {field}", segment.segment_name);
                if row[0] == "unsupported" {
                    assert!(
                        matches!(
                            segment.field_exists_docs(&field),
                            Err(lucene_search::Error::FieldExistsUnsupported(_))
                        ),
                        "{context}"
                    );
                    assert!(
                        matches!(
                            segment.field_exists_leaf(&field, None),
                            Err(lucene_search::Error::FieldExistsUnsupported(_))
                        ),
                        "{context}"
                    );
                    continue;
                }
                assert_eq!(row[0], "field");
                let leaf = segment.field_exists_leaf(&field, None).unwrap();
                if focused {
                    use lucene_search::doc_value_query::FieldExistsSource;
                    let source = match field.as_str() {
                        "norms_dv" | "dense_norms" | "empty_norms" => {
                            Some(FieldExistsSource::Norms)
                        }
                        "float_dv" | "byte_dv" | "vector_terms" => Some(FieldExistsSource::Vectors),
                        "points_dv" | "terms_dv" | "dense_skip" => {
                            Some(FieldExistsSource::DocValues)
                        }
                        "absent-probe" => None,
                        _ => panic!("unexpected supported field {field}"),
                    };
                    assert_eq!(leaf.source, source, "{context}");
                }
                assert_eq!(
                    count_field_exists_leaf(&leaf).unwrap_or(-1),
                    row[2].parse::<i64>().unwrap(),
                    "{context}"
                );
                let parse_docs = |line: &str, label: &str| -> Vec<i32> {
                    let (kind, docs) = line.split_once('\t').unwrap();
                    assert_eq!(kind, label);
                    if docs.is_empty() {
                        Vec::new()
                    } else {
                        docs.split(',').map(|doc| doc.parse().unwrap()).collect()
                    }
                };
                let expected_raw = parse_docs(lines.next().unwrap(), "raw");
                let expected_live = parse_docs(lines.next().unwrap(), "live");
                let raw: Vec<_> = match segment.field_exists_docs(&field).unwrap() {
                    ExistsDocs::None => Vec::new(),
                    ExistsDocs::All => (0..segment.max_doc).collect(),
                    ExistsDocs::Bits(bits) => (0..segment.max_doc)
                        .filter(|&doc| bits.get_doc(doc))
                        .collect(),
                };
                assert_eq!(raw, expected_raw, "{context}");
                let live: Vec<_> = raw
                    .iter()
                    .copied()
                    .filter(|&doc| segment.live_docs().is_none_or(|bits| bits.get_doc(doc)))
                    .collect();
                assert_eq!(live, expected_live, "{context}");
                deleted_matches += raw.len() - live.len();
                let rewrite = lines.next().unwrap().strip_prefix("rewrite\t").unwrap();
                if rewrite != "unsupported" {
                    let leaves: Vec<_> = reader
                        .segment_readers()
                        .iter()
                        .map(|segment| segment.field_exists_leaf(&field, None).unwrap())
                        .collect();
                    let terms_count = leaves
                        .iter()
                        .filter_map(|leaf| leaf.terms_doc_count)
                        .reduce(|a, b| a + b);
                    assert_eq!(
                        field_exists_rewrites_to_match_all_docs(
                            &leaves,
                            terms_count,
                            reader.max_doc()
                        )
                        .to_string(),
                        rewrite,
                        "{context}"
                    );
                }
                checked += 1;
                vectors += usize::from(
                    leaf.source == Some(lucene_search::doc_value_query::FieldExistsSource::Vectors),
                );
            }
        }
    }
    assert!(checked > 100);
    assert!(vectors > 20);
    assert!(deleted_matches > 0);
}

fn check_metadata(variant: &str) {
    let root = fixture(variant);
    let manifest = fs::read_to_string(root.join("segments.tsv")).unwrap();
    let rows: Vec<Vec<&str>> = manifest
        .lines()
        .map(|line| line.split('\t').collect())
        .collect();
    let reader = DirectoryReader::open(&FsDirectory::open(root.join("index"))).unwrap();
    assert_eq!(reader.segment_readers().len(), rows.len());
    let mut max_doc = 0;
    let mut live_doc_count = 0;
    for (segment, row) in reader.segment_readers().iter().zip(rows) {
        let count: i32 = row[1].parse().unwrap();
        let deleted: i32 = row[2].parse().unwrap();
        assert_eq!(segment.segment_name, row[0]);
        assert_eq!(segment.doc_base, max_doc);
        assert_eq!(segment.max_doc, count);
        assert_eq!(segment.num_docs(), count - deleted);
        if deleted == 0 {
            assert!(segment.live_docs().is_none());
        } else {
            let expected = fs::read_to_string(root.join(format!("live.{}.tsv", row[0]))).unwrap();
            let mut lines = expected.lines();
            assert_eq!(lines.next().unwrap().parse::<i32>().unwrap(), count);
            let bits = lines.next().unwrap().as_bytes();
            assert_eq!(bits.len(), usize::try_from(count).unwrap());
            let actual = segment.live_docs().unwrap();
            assert_eq!(actual.len(), bits.len());
            for (doc, &bit) in bits.iter().enumerate() {
                assert_eq!(
                    actual.get(doc),
                    bit == b'1',
                    "{variant} {} doc {doc}",
                    row[0]
                );
            }
        }
        max_doc += count;
        live_doc_count += count - deleted;
    }
    assert_eq!(reader.max_doc(), max_doc);
    assert_eq!(reader.num_docs(), live_doc_count);
}

#[test]
fn simpletext_flush_with_generational_updates_and_deletes_opens_as_java() {
    check_metadata("flush");
}

#[test]
fn simpletext_compound_merge_with_updates_and_deletes_opens_as_java() {
    check_metadata("merged");
}

/// Both search entry points open SimpleText segments: their postings, doc
/// values and points as the binary forms the scorer tree reads
/// (`tests/simpletext_search_fixtures.rs` compares searches with Lucene's).
#[test]
fn search_routing_opens_both_entry_points() {
    for variant in ["flush", "merged"] {
        let reader =
            DirectoryReader::open(&FsDirectory::open(fixture(variant).join("index"))).unwrap();
        let opened = reader.open_segments().unwrap();
        assert_eq!(
            opened.as_open_segments().len(),
            reader.segment_readers().len()
        );
        for (segment, open) in reader
            .segment_readers()
            .iter()
            .zip(opened.as_open_segments())
        {
            assert!(open.doc_in.is_some());
            assert!(open.fields.field("freqs").is_some());
            assert!(segment.doc_values_meta().is_some());
            let points = segment
                .with_open_segment(segment.doc_base, |seg| {
                    assert!(seg.fields.field("offs").is_some() && seg.pay_in.is_some());
                    let points = seg.points.unwrap();
                    let number = segment.field_infos().field_by_name("ipt").unwrap().number;
                    Ok(points.reader.field(number).map(|f| f.point_count))
                })
                .unwrap();
            assert!(points.is_some_and(|count| count > 0), "{variant}");
            assert!(segment.points_files().is_some());
        }
    }
}

#[test]
fn simpletext_stored_values_and_checksums_match_java_through_directory_reader() {
    use lucene_codecs::stored_fields::FieldValue;
    for variant in ["flush", "merged"] {
        let root = fixture(variant);
        let reader = DirectoryReader::open(&FsDirectory::open(root.join("index"))).unwrap();
        for segment in reader.segment_readers() {
            let mut actual = Vec::new();
            for doc in 0..segment.max_doc {
                actual.push(format!("doc\t{doc}"));
                for field in segment.stored_document(doc).unwrap().unwrap().fields {
                    let value = match field.value {
                        FieldValue::String(value) => format!("string\t{}", hex(value.as_bytes())),
                        FieldValue::Binary(value) => format!("binary\t{}", hex(&value)),
                        FieldValue::Int(value) => format!("int\t{value}"),
                        FieldValue::Long(value) => format!("long\t{value}"),
                        FieldValue::Float(value) => format!("float\t{:08x}", value.to_bits()),
                        FieldValue::Double(value) => format!("double\t{:016x}", value.to_bits()),
                    };
                    actual.push(format!("{}\t{value}", field.field_number));
                }
            }
            let expected =
                fs::read_to_string(root.join(format!("stored.{}.tsv", segment.segment_name)))
                    .unwrap();
            let expected: Vec<_> = expected
                .lines()
                .map(|line| line.split('\t').take(3).collect::<Vec<_>>().join("\t"))
                .collect();
            assert_eq!(actual, expected, "{variant} {}", segment.segment_name);
            segment.check_integrity().unwrap();
            assert!(segment.stored_document(-1).is_err());
            assert!(segment.stored_document(segment.max_doc).is_err());
        }
    }
}

#[test]
fn simpletext_terms_stats_seeks_and_all_postings_match_java_through_leaf_reader() {
    use lucene_codecs::blocktree::SeekStatus;
    use lucene_search::reader::{LeafReader, PostingsFlags, NO_MORE_DOCS};
    for variant in ["flush", "merged"] {
        let root = fixture(variant);
        let expected_root = root
            .join("../../../simpletext_directory_reader")
            .join(variant);
        let reader = DirectoryReader::open(&FsDirectory::open(root.join("index"))).unwrap();
        let segments = fs::read_to_string(expected_root.join("segments.tsv")).unwrap();
        assert_eq!(
            reader
                .segment_readers()
                .iter()
                .map(|s| s.segment_name.as_str())
                .collect::<Vec<_>>(),
            segments.lines().collect::<Vec<_>>()
        );
        for segment in reader.segment_readers() {
            let manifest = fs::read_to_string(
                expected_root.join(format!("terms.{}.tsv", segment.segment_name)),
            )
            .unwrap();
            let rows: Vec<Vec<_>> = manifest
                .lines()
                .map(|line| line.split('\t').collect())
                .collect();
            let mut at = 0;
            while at < rows.len() {
                let row = &rows[at];
                assert_eq!(row[0], "field");
                let field = String::from_utf8(unhex(row[2])).unwrap();
                assert_eq!(
                    segment
                        .field_infos()
                        .field_by_name(&field)
                        .unwrap()
                        .number
                        .to_string(),
                    row[1]
                );
                let terms = segment.terms(&field).unwrap().unwrap();
                assert_eq!(terms.size().to_string(), row[3]);
                assert_eq!(terms.sum_total_term_freq().to_string(), row[4]);
                assert_eq!(terms.sum_doc_freq().to_string(), row[5]);
                assert_eq!(terms.doc_count().to_string(), row[6]);
                assert_eq!(
                    segment.field_stats(&field).unwrap(),
                    Some((row[4].parse().unwrap(), row[6].parse().unwrap()))
                );
                assert_eq!(terms.has_freqs().to_string(), row[7]);
                assert_eq!(terms.has_positions().to_string(), row[8]);
                assert_eq!(terms.has_offsets().to_string(), row[9]);
                assert_eq!(terms.has_payloads().to_string(), row[10]);
                let mut cursor = terms.iterator().unwrap();
                assert!(cursor.term().is_none());
                assert!(cursor.doc_freq().is_err());
                assert!(cursor.postings(PostingsFlags::All).is_err());
                at += 1;
                while rows[at][0] == "term" {
                    let row = &rows[at];
                    assert_eq!(hex(cursor.next().unwrap().unwrap()), row[1]);
                    assert_eq!(cursor.doc_freq().unwrap().to_string(), row[2]);
                    assert_eq!(cursor.total_term_freq().unwrap().to_string(), row[3]);
                    assert_eq!(
                        segment
                            .doc_freq(&field, &unhex(row[1]))
                            .unwrap()
                            .to_string(),
                        row[2]
                    );
                    let mut postings = cursor.postings(PostingsFlags::All).unwrap();
                    let mut docs_only = cursor.postings(PostingsFlags::None).unwrap();
                    at += 1;
                    while rows[at][0] == "doc" {
                        let row = &rows[at];
                        let doc: i32 = row[1].parse().unwrap();
                        assert_eq!(postings.next_doc().unwrap(), doc);
                        assert_eq!(postings.freq().to_string(), row[2]);
                        assert_eq!(docs_only.next_doc().unwrap(), doc);
                        assert_eq!(docs_only.freq(), 1);
                        assert_eq!(docs_only.next_position().unwrap(), -1);
                        at += 1;
                        while rows[at][0] == "pos" {
                            let row = &rows[at];
                            assert_eq!(postings.next_position().unwrap().to_string(), row[1]);
                            assert_eq!(postings.start_offset().to_string(), row[2]);
                            assert_eq!(postings.end_offset().to_string(), row[3]);
                            assert_eq!(
                                postings.payload().map(hex).unwrap_or_else(|| "-".into()),
                                row[4]
                            );
                            at += 1;
                        }
                    }
                    assert_eq!(rows[at][0], "endterm");
                    assert_eq!(postings.next_doc().unwrap(), NO_MORE_DOCS);
                    assert_eq!(docs_only.next_doc().unwrap(), NO_MORE_DOCS);
                    at += 1;
                }
                assert!(cursor.next().unwrap().is_none());
                assert!(cursor.term().is_none());
                while rows[at][0] == "ceil" {
                    let row = &rows[at];
                    let status = cursor.try_seek_ceil(&unhex(row[1])).unwrap();
                    let description = match status {
                        SeekStatus::Found => "FOUND",
                        SeekStatus::NotFound => "NOT_FOUND",
                        SeekStatus::End => "END",
                    };
                    assert_eq!(description, row[2]);
                    assert_eq!(cursor.term().map(hex).unwrap_or_else(|| "-".into()), row[3]);
                    if status == SeekStatus::Found {
                        assert!(cursor.try_seek_exact(&unhex(row[1])).unwrap());
                    }
                    at += 1;
                }
                assert_eq!(rows[at][0], "endfield");
                at += 1;
            }
            assert!(segment.terms("missing-field").unwrap().is_none());
            assert_eq!(segment.doc_freq("missing-field", b"missing").unwrap(), 0);
        }
    }
}

fn numeric_value(cursor: &mut dyn NumericDocValues) -> String {
    cursor.long_value().to_string()
}
fn binary_value(cursor: &mut dyn BinaryDocValues) -> String {
    hex(cursor.binary_value())
}
fn sorted_value(cursor: &mut dyn SortedDocValues) -> String {
    cursor.ord_value().to_string()
}
fn sorted_numeric_value(cursor: &mut dyn SortedNumericDocValues) -> String {
    let values: Vec<_> = (0..cursor.doc_value_count())
        .map(|_| cursor.next_value().unwrap().to_string())
        .collect();
    assert!(cursor.next_value().is_err());
    values.join(",")
}
fn sorted_set_value(cursor: &mut dyn SortedSetDocValues) -> String {
    let values: Vec<_> = (0..cursor.doc_value_count())
        .map(|_| cursor.next_ord().unwrap().to_string())
        .collect();
    assert!(cursor.next_ord().is_err());
    values.join(",")
}

fn check_values<C: DocValuesIterator + ?Sized>(
    cursor: &mut C,
    expected: &BTreeMap<i32, String>,
    max_doc: i32,
    mode: usize,
    mut value: impl FnMut(&mut C) -> String,
) {
    use lucene_search::reader::NO_MORE_DOCS;
    assert_eq!(cursor.doc_id(), -1);
    match mode {
        0 => {
            for (&doc, expected) in expected {
                assert_eq!(cursor.next_doc().unwrap(), doc);
                assert_eq!(cursor.doc_id(), doc);
                assert_eq!(&value(cursor), expected);
            }
            assert_eq!(cursor.next_doc().unwrap(), NO_MORE_DOCS);
            assert_eq!(cursor.doc_id(), NO_MORE_DOCS);
            assert_eq!(cursor.next_doc().unwrap(), NO_MORE_DOCS);
        }
        1 => {
            for target in (0..max_doc).step_by(7) {
                if target <= cursor.doc_id() {
                    continue;
                }
                let next = expected.range(target..).next();
                let actual = cursor.advance(target).unwrap();
                assert_eq!(actual, next.map_or(NO_MORE_DOCS, |(&doc, _)| doc));
                assert_eq!(cursor.doc_id(), actual);
                if let Some((_, expected)) = next {
                    assert_eq!(&value(cursor), expected);
                } else {
                    break;
                }
            }
        }
        2 => {
            for doc in 0..max_doc {
                assert_eq!(
                    cursor.advance_exact(doc).unwrap(),
                    expected.contains_key(&doc)
                );
                assert_eq!(cursor.doc_id(), doc);
                if let Some(expected) = expected.get(&doc) {
                    assert_eq!(&value(cursor), expected);
                }
            }
            assert!(cursor.advance_exact(-1).is_err());
            assert!(cursor.advance_exact(max_doc).is_err());
        }
        _ => unreachable!(),
    }
}

#[test]
fn simpletext_all_doc_values_generations_and_norms_match_java_leaf_reader() {
    use lucene_search::reader::LeafReader;
    for variant in ["flush", "merged"] {
        let root = fixture(variant);
        let expected_root = root
            .join("../../../simpletext_directory_reader")
            .join(variant);
        let reader = DirectoryReader::open(&FsDirectory::open(root.join("index"))).unwrap();
        for segment in reader.segment_readers() {
            let manifest = fs::read_to_string(
                expected_root.join(format!("values.{}.tsv", segment.segment_name)),
            )
            .unwrap();
            let rows: Vec<Vec<_>> = manifest
                .lines()
                .map(|line| line.split('\t').collect())
                .collect();
            let mut at = 0;
            while at < rows.len() {
                let row = &rows[at];
                assert_eq!(row[0], "field");
                let field = String::from_utf8(unhex(row[1])).unwrap();
                let ty = row[2];
                if ty != "NORMS" {
                    assert_eq!(
                        segment
                            .field_infos()
                            .field_by_name(&field)
                            .unwrap()
                            .doc_values_gen
                            .to_string(),
                        row[3]
                    );
                }
                let cost: i64 = row[4].parse().unwrap();
                at += 1;
                let mut dictionary = Vec::new();
                while rows[at][0] == "ord" {
                    assert_eq!(rows[at][1].parse::<usize>().unwrap(), dictionary.len());
                    dictionary.push(unhex(rows[at][2]));
                    at += 1;
                }
                if rows[at][0] == "skipall" {
                    let row = &rows[at];
                    let mut skipper = segment.doc_values_skipper(&field).unwrap().unwrap();
                    assert_eq!(skipper.num_levels(), 1);
                    assert_eq!((skipper.min_doc_id(0), skipper.max_doc_id(0)), (-1, -1));
                    assert_eq!(skipper.global_min_value().to_string(), row[1]);
                    assert_eq!(skipper.global_max_value().to_string(), row[2]);
                    assert_eq!(skipper.global_doc_count().to_string(), row[3]);
                    assert_eq!(skipper.max_value_count().to_string(), row[4]);
                    at += 1;
                    let mut target = 0;
                    while rows[at][0] == "skip" {
                        let row = &rows[at];
                        skipper.advance(target);
                        assert_eq!(skipper.min_doc_id(0).to_string(), row[1]);
                        assert_eq!(skipper.max_doc_id(0).to_string(), row[2]);
                        assert_eq!(skipper.min_value(0).to_string(), row[3]);
                        assert_eq!(skipper.max_value(0).to_string(), row[4]);
                        assert_eq!(skipper.doc_count(0).to_string(), row[5]);
                        target = skipper.max_doc_id(0) + 1;
                        at += 1;
                    }
                    skipper.advance(target);
                    assert_eq!(skipper.min_doc_id(0), lucene_search::reader::NO_MORE_DOCS);
                    assert_eq!(skipper.max_doc_id(0), lucene_search::reader::NO_MORE_DOCS);
                } else if ty != "NORMS" {
                    assert!(segment.doc_values_skipper(&field).unwrap().is_none());
                }
                let mut expected = BTreeMap::new();
                while rows[at][0] == "d" {
                    expected.insert(rows[at][1].parse::<i32>().unwrap(), rows[at][2].to_owned());
                    at += 1;
                }
                assert_eq!(rows[at][0], "endfield");
                at += 1;
                if ty == "NORMS" {
                    let norms = segment.field_norms(&field).unwrap().unwrap();
                    let mut cursor = norms.cursor();
                    for doc in (0..segment.max_doc).chain((0..segment.max_doc).rev()) {
                        let value = expected
                            .get(&doc)
                            .map(|value| value.parse::<i64>().unwrap());
                        assert_eq!(cursor.norm_long(doc).unwrap(), value);
                        assert_eq!(cursor.has_norm(doc).unwrap(), value.is_some());
                    }
                    let reader_norms = reader.field_norms(&field).unwrap();
                    let by_field = reader
                        .field_norms_by_field(std::slice::from_ref(&field))
                        .unwrap();
                    let avg = reader.avg_field_length(&field).unwrap().unwrap();
                    for ((segment, norms), by_field) in reader
                        .segment_readers()
                        .iter()
                        .zip(reader_norms)
                        .zip(by_field)
                    {
                        if let Some(norms) = norms {
                            assert_eq!(norms.avg_field_length.to_bits(), avg.to_bits());
                            assert_eq!(by_field[&field].avg_field_length.to_bits(), avg.to_bits());
                            for doc in 0..segment.max_doc {
                                assert_eq!(
                                    norms.cursor().norm_long(doc).unwrap(),
                                    by_field[&field].cursor().norm_long(doc).unwrap()
                                );
                            }
                        }
                    }
                }
                for mode in 0..3 {
                    macro_rules! check {
                        ($getter:ident, $read:ident) => {{
                            let mut cursor = segment.$getter(&field).unwrap().unwrap();
                            assert_eq!(cursor.cost(), cost);
                            check_values(
                                cursor.as_mut(),
                                &expected,
                                segment.max_doc,
                                mode,
                                |cursor| $read(cursor),
                            );
                        }};
                    }
                    match ty {
                        "NUMERIC" => check!(numeric_doc_values, numeric_value),
                        "BINARY" => check!(binary_doc_values, binary_value),
                        "SORTED" => check!(sorted_doc_values, sorted_value),
                        "SORTED_NUMERIC" => check!(sorted_numeric_doc_values, sorted_numeric_value),
                        "SORTED_SET" => check!(sorted_set_doc_values, sorted_set_value),
                        "NORMS" => check!(norm_values, numeric_value),
                        _ => panic!("unknown doc-values type {ty}"),
                    }
                }
                // Lucene 10.6 (#16714): a SimpleText SORTED_NUMERIC column
                // with at most one value per document is a singleton.
                if ty == "SORTED_NUMERIC" {
                    let single = expected.values().all(|v| !v.contains(','));
                    assert_eq!(segment.sorted_numeric_is_singleton(&field), single);
                }
                macro_rules! check_dictionary {
                    ($getter:ident, $ord:ty) => {{
                        let mut cursor = segment.$getter(&field).unwrap().unwrap();
                        assert_eq!(cursor.value_count(), dictionary.len() as $ord);
                        for (ord, term) in dictionary.iter().enumerate() {
                            assert_eq!(cursor.lookup_ord(ord as $ord).unwrap(), *term);
                            assert_eq!(cursor.lookup_term(term).unwrap(), ord as $ord);
                        }
                        for target in [&b""[..], &b"\0"[..], &b"absent-probe"[..], &b"\xff"[..]] {
                            let expected = match dictionary
                                .binary_search_by(|term| term.as_slice().cmp(target))
                            {
                                Ok(ord) => ord as $ord,
                                Err(insert) => -(insert as $ord) - 1,
                            };
                            assert_eq!(cursor.lookup_term(target).unwrap(), expected);
                        }
                        assert!(cursor.lookup_ord(-1).is_err());
                        assert!(cursor.lookup_ord(dictionary.len() as $ord).is_err());
                    }};
                }
                match ty {
                    "SORTED" => check_dictionary!(sorted_doc_values, i32),
                    "SORTED_SET" => check_dictionary!(sorted_set_doc_values, i64),
                    _ => assert!(dictionary.is_empty()),
                }
                // A real field with another type is absent from each typed accessor.
                if ty != "NORMS" {
                    if ty != "NUMERIC" {
                        assert!(segment.numeric_doc_values(&field).unwrap().is_none());
                    }
                    if ty != "BINARY" {
                        assert!(segment.binary_doc_values(&field).unwrap().is_none());
                    }
                    if ty != "SORTED" {
                        assert!(segment.sorted_doc_values(&field).unwrap().is_none());
                    }
                    if ty != "SORTED_NUMERIC" {
                        assert!(segment.sorted_numeric_doc_values(&field).unwrap().is_none());
                    }
                    if ty != "SORTED_SET" {
                        assert!(segment.sorted_set_doc_values(&field).unwrap().is_none());
                    }
                }
            }
            assert!(segment.numeric_doc_values("absent").unwrap().is_none());
            assert!(segment.binary_doc_values("absent").unwrap().is_none());
            assert!(segment.sorted_doc_values("absent").unwrap().is_none());
            assert!(segment
                .sorted_numeric_doc_values("absent")
                .unwrap()
                .is_none());
            assert!(segment.sorted_set_doc_values("absent").unwrap().is_none());
            assert!(segment.norm_values("absent").unwrap().is_none());
            assert!(segment.norm_values("num").unwrap().is_none());
        }
    }
}

#[test]
fn simpletext_reader_wide_norms_score_like_java_bm25_and_classic() {
    use lucene_search::similarities::{
        Bm25Similarity, ClassicSimilarity, CollectionStatistics, Similarity, TermStatistics,
    };
    for variant in ["flush", "merged"] {
        let root = fixture(variant);
        let expected_root = root
            .join("../../../simpletext_directory_reader")
            .join(variant);
        let reader = DirectoryReader::open(&FsDirectory::open(root.join("index"))).unwrap();
        let mut compared = 0;
        for (segment_index, segment) in reader.segment_readers().iter().enumerate() {
            let manifest = fs::read_to_string(
                expected_root.join(format!("scores.{}.tsv", segment.segment_name)),
            )
            .unwrap();
            let rows: Vec<Vec<_>> = manifest
                .lines()
                .map(|line| line.split('\t').collect())
                .collect();
            let mut at = 0;
            while at < rows.len() {
                let row = &rows[at];
                assert_eq!(row[0], "field");
                let field = String::from_utf8(unhex(row[1])).unwrap();
                let collection = CollectionStatistics::new(
                    row[2].parse().unwrap(),
                    row[3].parse().unwrap(),
                    row[4].parse().unwrap(),
                    row[5].parse().unwrap(),
                )
                .unwrap();
                let avg =
                    (collection.sum_total_term_freq as f64 / collection.doc_count as f64) as f32;
                assert_eq!(
                    reader.avg_field_length(&field).unwrap().unwrap().to_bits(),
                    avg.to_bits()
                );
                let norms = reader.field_norms(&field).unwrap();
                let norms = norms[segment_index].as_ref().unwrap();
                assert_eq!(norms.avg_field_length.to_bits(), avg.to_bits());
                let mut cursor = norms.cursor();
                at += 1;
                while rows[at][0] == "term" {
                    let row = &rows[at];
                    let stats =
                        TermStatistics::new(row[2].parse().unwrap(), row[3].parse().unwrap())
                            .unwrap();
                    assert_eq!(
                        reader.doc_freq(&field, &unhex(row[1])).unwrap(),
                        stats.doc_freq
                    );
                    let bm25 = Bm25Similarity::default().scorer(&field, 1.0, &collection, &[stats]);
                    let classic =
                        ClassicSimilarity::default().scorer(&field, 1.0, &collection, &[stats]);
                    let weight =
                        lucene_search::similarity::idf(stats.doc_freq, collection.doc_count);
                    at += 1;
                    let mut docs = Vec::new();
                    let mut freqs = Vec::new();
                    let mut expected_fast = Vec::new();
                    while rows[at][0] == "doc" {
                        let row = &rows[at];
                        let doc = row[1].parse().unwrap();
                        let freq: i32 = row[2].parse().unwrap();
                        let raw = cursor.norm_long(doc).unwrap().unwrap_or(1);
                        assert_eq!(raw.to_string(), row[3]);
                        let expected = u32::from_str_radix(row[4], 16).unwrap();
                        let expected_classic = u32::from_str_radix(row[5], 16).unwrap();
                        assert_eq!(bm25.score(freq as f32, raw).to_bits(), expected);
                        assert_eq!(classic.score(freq as f32, raw).to_bits(), expected_classic);
                        assert_eq!(
                            lucene_search::similarity::do_score(
                                weight,
                                freq as f32,
                                cursor.norm_inverse(doc).unwrap()
                            )
                            .to_bits(),
                            expected
                        );
                        docs.push(doc);
                        freqs.push(freq);
                        expected_fast.push(expected);
                        compared += 1;
                        at += 1;
                    }
                    let mut inverses = Vec::new();
                    cursor.norm_inverse_batch(&docs, &mut inverses).unwrap();
                    let mut scores = Vec::new();
                    lucene_search::similarity::do_score_batch(
                        weight,
                        &freqs,
                        &inverses,
                        &mut scores,
                    );
                    assert_eq!(
                        scores
                            .iter()
                            .map(|score| score.to_bits())
                            .collect::<Vec<_>>(),
                        expected_fast
                    );
                    assert_eq!(rows[at][0], "endterm");
                    at += 1;
                }
                assert_eq!(rows[at][0], "endfield");
                at += 1;
            }
        }
        assert!(compared > 100, "the fixture must score nontrivial postings");
        assert!(reader.avg_field_length("absent").unwrap().is_none());
        assert!(reader
            .field_norms_by_field(&["absent".into()])
            .unwrap()
            .iter()
            .all(|norms| norms.is_empty()));
    }
}

#[test]
fn simpletext_term_vectors_match_java_through_leaf_reader() {
    use lucene_search::reader::LeafReader;
    for variant in ["flush", "merged"] {
        let root = fixture(variant);
        let reader = DirectoryReader::open(&FsDirectory::open(root.join("index"))).unwrap();
        for segment in reader.segment_readers() {
            let mut actual = Vec::new();
            for doc in 0..segment.max_doc {
                actual.push(format!("doc\t{doc}"));
                let Some(vectors) = segment.term_vectors(doc).unwrap() else {
                    actual.push("none".to_owned());
                    continue;
                };
                for field in vectors.fields {
                    let info = segment
                        .field_infos()
                        .field_by_number(field.field_number)
                        .unwrap();
                    actual.push(format!(
                        "field\t{}\t{}\t{}\t{}\t{}",
                        hex(info.name.as_bytes()),
                        field.has_positions,
                        field.has_offsets,
                        field.has_payloads,
                        field.terms.len()
                    ));
                    for term in field.terms {
                        actual.push(format!("term\t{}\t{}", hex(&term.term), term.freq));
                        if field.has_positions || field.has_offsets {
                            for i in 0..usize::try_from(term.freq).unwrap() {
                                let pos = term.positions.as_ref().map_or(-1, |values| values[i]);
                                let start =
                                    term.start_offsets.as_ref().map_or(-1, |values| values[i]);
                                let end = term.end_offsets.as_ref().map_or(-1, |values| values[i]);
                                let payload = term
                                    .payloads
                                    .as_ref()
                                    .map(|values| values[i].as_slice())
                                    .filter(|p| !p.is_empty())
                                    .map_or_else(|| "-".to_owned(), hex);
                                actual.push(format!("pos\t{pos}\t{start}\t{end}\t{payload}"));
                            }
                        }
                    }
                }
            }
            let manifest =
                fs::read_to_string(root.join(format!("tv.{}.tsv", segment.segment_name))).unwrap();
            let expected: Vec<_> = manifest
                .lines()
                .filter(|line| !line.starts_with("ceil\t"))
                .map(str::to_owned)
                .collect();
            assert_eq!(actual, expected, "{variant} {}", segment.segment_name);
            assert!(segment.term_vectors(-1).is_err());
            assert!(segment.term_vectors(segment.max_doc).is_err());
        }
    }
}

struct PointBox {
    lo: Vec<u8>,
    hi: Vec<u8>,
    dims: usize,
    width: usize,
    relations: String,
    visits: i64,
    value_visits: i64,
    matches: i64,
    hash: i64,
}
impl PointBox {
    fn new(lo: Vec<u8>, hi: Vec<u8>, dims: usize, width: usize) -> Self {
        Self {
            lo,
            hi,
            dims,
            width,
            relations: String::new(),
            visits: 0,
            value_visits: 0,
            matches: 0,
            hash: 0,
        }
    }
}
impl lucene_search::reader::IntersectVisitor for PointBox {
    fn compare(&mut self, min: &[u8], max: &[u8]) -> lucene_search::reader::Relation {
        use lucene_search::reader::Relation;
        let mut crosses = false;
        for dim in 0..self.dims {
            let range = dim * self.width..(dim + 1) * self.width;
            if min[range.clone()] > self.hi[range.clone()]
                || max[range.clone()] < self.lo[range.clone()]
            {
                self.relations.push('O');
                return Relation::CellOutsideQuery;
            }
            crosses |=
                min[range.clone()] < self.lo[range.clone()] || max[range.clone()] > self.hi[range];
        }
        self.relations.push(if crosses { 'C' } else { 'I' });
        if crosses {
            Relation::CellCrossesQuery
        } else {
            Relation::CellInsideQuery
        }
    }
    fn visit(&mut self, doc: i32) {
        self.visits += 1;
        self.hash = self
            .hash
            .wrapping_mul(1_000_003)
            .wrapping_add(i64::from(doc));
    }
    fn visit_with_value(&mut self, doc: i32, value: &[u8]) {
        self.value_visits += 1;
        let matches = (0..self.dims).all(|dim| {
            let range = dim * self.width..(dim + 1) * self.width;
            value[range.clone()] >= self.lo[range.clone()] && value[range.clone()] <= self.hi[range]
        });
        self.matches += i64::from(matches);
        self.hash = self
            .hash
            .wrapping_mul(1_000_003)
            .wrapping_add(2 * i64::from(doc) + i64::from(matches));
    }
}

#[test]
fn simpletext_point_bounds_intersections_and_estimates_match_java_leaf_reader() {
    use lucene_search::reader::LeafReader;
    for variant in ["flush", "merged"] {
        let root = fixture(variant);
        let reader = DirectoryReader::open(&FsDirectory::open(root.join("index"))).unwrap();
        for segment in reader.segment_readers() {
            let manifest =
                fs::read_to_string(root.join(format!("points.{}.tsv", segment.segment_name)))
                    .unwrap();
            let rows: Vec<Vec<_>> = manifest
                .lines()
                .map(|line| line.split('\t').collect())
                .collect();
            let mut at = 0;
            while at < rows.len() {
                let header = &rows[at];
                assert_eq!(header[0], "field");
                let name = String::from_utf8(unhex(header[1])).unwrap();
                let points = segment.point_values(&name).unwrap().unwrap();
                assert_eq!(points.num_dimensions().to_string(), header[2]);
                assert_eq!(points.num_index_dimensions().to_string(), header[3]);
                assert_eq!(points.bytes_per_dimension().to_string(), header[4]);
                assert_eq!(points.size().to_string(), header[5]);
                assert_eq!(points.doc_count().to_string(), header[6]);
                assert_eq!(hex(points.min_packed_value()), header[7]);
                assert_eq!(hex(points.max_packed_value()), header[8]);
                at += 1;
                let mut boxes = 0;
                while at < rows.len() && rows[at][0] != "field" {
                    let row = &rows[at];
                    if row[0] == "box" {
                        let visitor = || {
                            PointBox::new(
                                unhex(row[1]),
                                unhex(row[2]),
                                points.num_index_dimensions() as usize,
                                points.bytes_per_dimension() as usize,
                            )
                        };
                        assert_eq!(
                            points
                                .estimate_point_count(&mut visitor())
                                .unwrap()
                                .to_string(),
                            row[3]
                        );
                        let mut actual = visitor();
                        points.intersect(&mut actual).unwrap();
                        assert_eq!(actual.relations, row[4]);
                        assert_eq!(actual.visits.to_string(), row[5]);
                        assert_eq!(actual.value_visits.to_string(), row[6]);
                        assert_eq!(actual.matches.to_string(), row[7]);
                        assert_eq!(actual.hash.to_string(), row[8]);
                        boxes += 1;
                    }
                    at += 1;
                }
                assert_eq!(boxes, 9);
            }
            assert!(segment.point_values("missing").unwrap().is_none());
            assert!(segment.point_values("num").unwrap().is_none());
        }
    }
}

#[test]
fn simpletext_float_and_byte_vectors_match_java_leaf_reader() {
    use lucene_search::reader::LeafReader;
    for variant in ["flush", "merged"] {
        let root = fixture(variant);
        let reader = DirectoryReader::open(&FsDirectory::open(root.join("index"))).unwrap();
        for segment in reader.segment_readers() {
            let manifest =
                fs::read_to_string(root.join(format!("knn.{}.tsv", segment.segment_name))).unwrap();
            let rows: Vec<Vec<_>> = manifest
                .lines()
                .map(|line| line.split('\t').collect())
                .collect();
            let mut at = 0;
            while at < rows.len() {
                let header = &rows[at];
                assert_eq!(header[0], "field");
                let name = String::from_utf8(unhex(header[1])).unwrap();
                let dimension: usize = header[4].parse().unwrap();
                let size: i32 = header[5].parse().unwrap();
                at += 1;
                let mut expected = Vec::new();
                while at < rows.len() && rows[at][0] != "field" {
                    if rows[at][0] == "v" {
                        expected.push(&rows[at]);
                    }
                    at += 1;
                }
                assert_eq!(expected.len(), size as usize);
                match header[2] {
                    "FLOAT32" => {
                        assert!(segment.byte_vector_values(&name).unwrap().is_none());
                        let values = segment.float_vector_values(&name).unwrap().unwrap();
                        assert_eq!(values.dimension(), dimension);
                        assert_eq!(values.size(), size);
                        for row in expected {
                            let ord = row[1].parse().unwrap();
                            assert_eq!(values.ord_to_doc(ord).unwrap().to_string(), row[2]);
                            let actual: Vec<_> = values
                                .vector_value(ord)
                                .unwrap()
                                .iter()
                                .flat_map(|v| v.to_bits().to_be_bytes())
                                .collect();
                            assert_eq!(hex(&actual), row[3]);
                        }
                        for ord in [-1, size] {
                            assert!(values.vector_value(ord).is_err());
                            assert!(values.ord_to_doc(ord).is_err());
                        }
                    }
                    "BYTE" => {
                        assert!(segment.float_vector_values(&name).unwrap().is_none());
                        let values = segment.byte_vector_values(&name).unwrap().unwrap();
                        assert_eq!(values.dimension(), dimension);
                        assert_eq!(values.size(), size);
                        for row in expected {
                            let ord = row[1].parse().unwrap();
                            assert_eq!(values.ord_to_doc(ord).unwrap().to_string(), row[2]);
                            assert_eq!(hex(&values.vector_value(ord).unwrap()), row[3]);
                        }
                        for ord in [-1, size] {
                            assert!(values.vector_value(ord).is_err());
                            assert!(values.ord_to_doc(ord).is_err());
                        }
                    }
                    other => panic!("unknown encoding {other}"),
                }
            }
            for field in ["missing", "num"] {
                assert!(segment.float_vector_values(field).unwrap().is_none());
                assert!(segment.byte_vector_values(field).unwrap().is_none());
            }
        }
    }
}
