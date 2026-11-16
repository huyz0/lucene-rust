#![cfg(test)]
use super::*;

fn infos() -> Vec<FieldInfo> {
    vec![
        FieldInfo::new("body\nx", 0)
            .with_index_options(IndexOptions::DocsAndFreqsAndPositionsAndOffsets)
            .with_store_term_vectors(true)
            .with_store_payloads(true)
            .with_attributes(vec![("k\\".into(), "v\n".into()), ("a".into(), "b".into())]),
        FieldInfo::new("dv", 3)
            .with_doc_values(
                DocValuesType::SortedNumeric,
                DocValuesSkipIndexType::Range,
                2,
            )
            .with_points(3, 2, 4)
            .with_vectors(
                8,
                VectorEncoding::Byte,
                VectorSimilarityFunction::MaximumInnerProduct,
            )
            .with_soft_deletes_field(true),
        FieldInfo::new("p", 1)
            .with_index_options(IndexOptions::Docs)
            .with_omit_norms(true)
            .with_parent_field(true),
    ]
}

fn reseal(body: &str) -> Vec<u8> {
    let mut out = body.as_bytes().to_vec();
    util::write_checksum(&mut out);
    out
}

fn body() -> String {
    let out = String::from_utf8(write(&infos())).unwrap();
    out.rsplit_once("checksum ").unwrap().0.to_string()
}

#[test]
fn round_trip_in_number_order() {
    let out = write(&infos());
    let back = read(&out).unwrap();
    // Java's `FieldInfos` iterates by number, in `write` and after `read`.
    let mut want = infos();
    want.sort_by_key(|fi| fi.number);
    assert_eq!(format!("{:?}", back.fields), format!("{:?}", want));
    assert_eq!(write(&back.fields), out);
    assert_eq!(write(&want), out);
    let text = String::from_utf8(out).unwrap();
    assert!(text.find("  name p\n") < text.find("  name dv\n"), "{text}");
    assert!(text.contains("  doc values skip indexRANGE\n"), "{text}");
    assert!(text.contains("  norms false\n"));
    assert_eq!(field_infos_file_name("_0", ""), "_0.inf");
    assert_eq!(field_infos_file_name("_0", "1"), "_0_1.inf");
}

#[test]
fn every_enum_name_round_trips() {
    for o in [
        IndexOptions::None,
        IndexOptions::Docs,
        IndexOptions::DocsAndFreqs,
        IndexOptions::DocsAndFreqsAndPositions,
        IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
        IndexOptions::DocsAndCustomFreqs,
    ] {
        assert_eq!(index_options(index_options_name(o)).unwrap(), o);
    }
    for t in [
        DocValuesType::None,
        DocValuesType::Numeric,
        DocValuesType::Binary,
        DocValuesType::Sorted,
        DocValuesType::SortedSet,
        DocValuesType::SortedNumeric,
    ] {
        assert_eq!(doc_values_type(doc_values_type_name(t)).unwrap(), t);
    }
    for e in [VectorEncoding::Byte, VectorEncoding::Float32] {
        assert_eq!(vector_encoding(vector_encoding_name(e)).unwrap(), e);
    }
    for f in [
        VectorSimilarityFunction::Euclidean,
        VectorSimilarityFunction::DotProduct,
        VectorSimilarityFunction::Cosine,
        VectorSimilarityFunction::MaximumInnerProduct,
    ] {
        assert_eq!(distance_function(similarity_name(f)).unwrap(), f);
    }
    assert_eq!(
        doc_values_skip_index_type("NONE").unwrap(),
        DocValuesSkipIndexType::None
    );
    let err = index_options("docs").unwrap_err().to_string();
    assert_eq!(
        err,
        "No enum constant org.apache.lucene.index.IndexOptions.docs"
    );
}

#[test]
fn repeated_attribute_keys_collapse_and_negative_counts_read_none() {
    let b = body().replacen(
        "  attributes 2\n",
        "  attributes 3\n    key a\n    value first\n",
        1,
    );
    let fis = read(&reseal(&b)).unwrap();
    assert_eq!(
        fis.fields[0].attributes,
        [
            ("a".to_string(), "b".to_string()),
            ("k\\".into(), "v\n".into())
        ]
    );
    // Java's `new HashMap<>()` loops zero times on a negative count.
    let b = body().replacen(
        "  attributes 2\n    key k\\\\\n    value v\\\n\n    key a\n    value b\n",
        "  attributes -4\n",
        1,
    );
    assert!(read(&reseal(&b)).unwrap().fields[0].attributes.is_empty());
}

#[test]
fn damaged_files_are_refused() {
    let b = body();
    for (from, to) in [
        ("number of fields 3", "number of fields -1"),
        ("number of fields 3", "number of fields 4"),
        ("  number 3", "  number x"),
        ("  index options DOCS\n", "  index options DOC\n"),
        ("  doc values SORTED_NUMERIC", "  doc values SORTED_NUMBER"),
        (
            "  doc values skip indexRANGE",
            "  doc values skip index RANGE",
        ),
        ("  doc values gen 2", "  doc values gen 2.0"),
        ("  vector encoding BYTE", "  vector encoding INT8"),
        (
            "  vector similarity MAXIMUM_INNER_PRODUCT",
            "  vector similarity DOT",
        ),
        (
            "  data dimensional count 3",
            "  data dimensional count three",
        ),
        ("  index dimensional count 2", "  index dimensional count"),
        ("  dimensional num bytes 4", "  dimensional num bytes 0"),
        (
            "  vector number of dimensions 8",
            "  vector number of dimensions -8",
        ),
        ("  soft-deletes", "  soft deletes"),
        ("  parent true", "  parent"),
        ("  name p", "  name dv"),
        ("  number 1", "  number 3"),
        ("  number 1", "  number -1"),
        ("  payloads true", "  payload true"),
    ] {
        assert!(b.contains(from), "{from}");
        assert!(read(&reseal(&b.replacen(from, to, 1))).is_err(), "{to}");
    }
    let out = write(&infos());
    assert!(read(&out[..out.len() - 1]).is_err());
    let mut flipped = out.clone();
    flipped[20] ^= 1;
    assert!(read(&flipped).is_err());
}
