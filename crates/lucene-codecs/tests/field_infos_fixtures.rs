//! Differential test against a real `.fnm` file written by an actual
//! IndexWriter (7 fields of varied shapes + a soft-deletes field introduced
//! by a later doc-values update). Regenerate with fixtures/src/GenFieldInfos.java.

use lucene_codecs::field_infos::{self, DocValuesType, IndexOptions, VectorSimilarityFunction};

fn dir() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/field_infos_index/"
    )
    .to_string()
}

struct Manifest {
    kv: Vec<(String, String)>,
}

impl Manifest {
    fn load() -> Self {
        let text = std::fs::read_to_string(format!("{}manifest.properties", dir()))
            .expect("run fixtures generator first (GenFieldInfos)");
        let kv = text
            .lines()
            .filter_map(|l| l.split_once('='))
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        Manifest { kv }
    }

    fn get(&self, key: &str) -> &str {
        self.kv
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
            .unwrap_or_else(|| panic!("manifest key {key} missing"))
    }

    fn get_i32(&self, key: &str) -> i32 {
        self.get(key).parse().unwrap()
    }

    fn get_bool(&self, key: &str) -> bool {
        self.get(key).parse().unwrap()
    }
}

fn id_from_hex(hex: &str) -> [u8; 16] {
    let mut id = [0u8; 16];
    for i in 0..16 {
        id[i] = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap();
    }
    id
}

fn expected_index_options(java_name: &str) -> IndexOptions {
    match java_name {
        "NONE" => IndexOptions::None,
        "DOCS" => IndexOptions::Docs,
        "DOCS_AND_FREQS" => IndexOptions::DocsAndFreqs,
        "DOCS_AND_FREQS_AND_POSITIONS" => IndexOptions::DocsAndFreqsAndPositions,
        "DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS" => {
            IndexOptions::DocsAndFreqsAndPositionsAndOffsets
        }
        "DOCS_AND_CUSTOM_FREQS" => IndexOptions::DocsAndCustomFreqs,
        other => panic!("unknown IndexOptions: {other}"),
    }
}

fn expected_doc_values_type(java_name: &str) -> DocValuesType {
    match java_name {
        "NONE" => DocValuesType::None,
        "NUMERIC" => DocValuesType::Numeric,
        "BINARY" => DocValuesType::Binary,
        "SORTED" => DocValuesType::Sorted,
        "SORTED_SET" => DocValuesType::SortedSet,
        "SORTED_NUMERIC" => DocValuesType::SortedNumeric,
        other => panic!("unknown DocValuesType: {other}"),
    }
}

fn expected_similarity(java_name: &str) -> VectorSimilarityFunction {
    match java_name {
        "EUCLIDEAN" => VectorSimilarityFunction::Euclidean,
        "DOT_PRODUCT" => VectorSimilarityFunction::DotProduct,
        "COSINE" => VectorSimilarityFunction::Cosine,
        "MAXIMUM_INNER_PRODUCT" => VectorSimilarityFunction::MaximumInnerProduct,
        other => panic!("unknown VectorSimilarityFunction: {other}"),
    }
}

#[test]
fn parses_real_fnm_with_varied_fields_and_soft_deletes() {
    let manifest = Manifest::load();
    let fnm_file_name = manifest.get("fnm_file_name");
    let segment_suffix = manifest.get("segment_suffix");
    let id = id_from_hex(manifest.get("id_hex"));

    let buf = std::fs::read(format!("{}{}.raw", dir(), fnm_file_name)).unwrap();
    let fis = field_infos::parse(&buf, &id, segment_suffix).unwrap();

    assert_eq!(fis.fields.len(), manifest.get_i32("field_count") as usize);

    let field_order: Vec<&str> = manifest.get("field_order").split(',').collect();
    assert_eq!(fis.fields.len(), field_order.len());

    for (i, name) in field_order.iter().enumerate() {
        let f = &fis.fields[i];
        assert_eq!(&f.name, name, "field order mismatch at index {i}");

        let prefix = format!("field.{name}.");
        assert_eq!(f.number, manifest.get_i32(&format!("{prefix}number")));
        assert_eq!(
            f.index_options,
            expected_index_options(manifest.get(&format!("{prefix}index_options")))
        );
        assert_eq!(
            f.doc_values_type,
            expected_doc_values_type(manifest.get(&format!("{prefix}doc_values_type")))
        );
        assert_eq!(
            f.store_term_vectors,
            manifest.get_bool(&format!("{prefix}has_term_vectors"))
        );
        assert_eq!(
            f.soft_deletes_field,
            manifest.get_bool(&format!("{prefix}is_soft_deletes"))
        );
        assert_eq!(
            f.point_dimension_count,
            manifest.get_i32(&format!("{prefix}point_dimension_count"))
        );
        assert_eq!(
            f.point_num_bytes,
            manifest.get_i32(&format!("{prefix}point_num_bytes"))
        );
        assert_eq!(
            f.vector_dimension,
            manifest.get_i32(&format!("{prefix}vector_dimension"))
        );
        assert_eq!(
            f.vector_similarity_function,
            expected_similarity(manifest.get(&format!("{prefix}vector_similarity")))
        );
    }

    // Cross-check the one field this fixture specifically exists to exercise:
    // a field introduced by a later doc-values-update generation, not present
    // in the segment's original flush.
    let soft = fis
        .fields
        .iter()
        .find(|f| f.name == "__soft_deletes")
        .unwrap();
    assert!(soft.soft_deletes_field);
    assert_eq!(soft.doc_values_type, DocValuesType::Numeric);
}

#[test]
fn wrong_segment_id_rejected() {
    let manifest = Manifest::load();
    let fnm_file_name = manifest.get("fnm_file_name");
    let segment_suffix = manifest.get("segment_suffix");
    let buf = std::fs::read(format!("{}{}.raw", dir(), fnm_file_name)).unwrap();
    let wrong_id = [0u8; 16];
    assert!(field_infos::parse(&buf, &wrong_id, segment_suffix).is_err());
}

#[test]
fn wrong_segment_suffix_rejected() {
    let manifest = Manifest::load();
    let fnm_file_name = manifest.get("fnm_file_name");
    let id = id_from_hex(manifest.get("id_hex"));
    let buf = std::fs::read(format!("{}{}.raw", dir(), fnm_file_name)).unwrap();
    assert!(field_infos::parse(&buf, &id, "wrong-suffix").is_err());
}
