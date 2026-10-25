//! Java leaf searches with finite budgets, on the existing Java graph fixture.
use std::{fs, path::PathBuf};

use lucene_codecs::field_infos::{VectorEncoding, VectorSimilarityFunction};
use lucene_index::buffered_updates::Term;
use lucene_index::document::{
    Document, KnnByteVectorField, KnnFloatVectorField, Store, StringField,
};
use lucene_search::collector::TotalHitsRelation;
use lucene_search::directory_reader::DirectoryReader;
use lucene_store::{ByteBuffersDirectory, Directory, FsDirectory};
use lucene_test_framework::index::base_index_file_format::plain_writer;
use lucene_test_framework::index::base_knn_vectors_format::{
    byte_values, float_values, search_nearest_byte, search_nearest_float,
};

#[test]
fn java_leaf_visits_relations_and_score_bits_match() {
    let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures");
    let dir = FsDirectory::open(fixtures.join("data/vectors_index"));
    let reader = DirectoryReader::open(&dir).unwrap();
    assert_eq!(reader.segment_readers().len(), 1);
    let leaf = &reader.segment_readers()[0];
    let manifest = fs::read_to_string(fixtures.join("testframework/data/knn-visits.tsv")).unwrap();
    let mut comparisons = 0;
    for line in manifest.lines() {
        let row: Vec<_> = line.split('\t').collect();
        let field = row[0];
        let ord = row[1].parse().unwrap();
        let k = row[2].parse().unwrap();
        let limit = row[3].parse().unwrap();
        let info = leaf.field_infos().field_by_name(field).unwrap();
        let result = if info.vector_encoding == VectorEncoding::Float32 {
            let target = float_values(leaf, field).unwrap().vector(ord).unwrap();
            search_nearest_float(&dir, leaf, field, &target, k, limit)
        } else {
            let values = byte_values(leaf, field).unwrap();
            let target = values.vector(ord).unwrap();
            search_nearest_byte(&dir, leaf, field, target, k, limit)
        };
        assert_eq!(
            result.total_hits.value,
            row[4].parse::<u64>().unwrap(),
            "{field}/{ord}/{k}/{limit}"
        );
        assert_eq!(
            result.total_hits.relation,
            match row[5] {
                "EQUAL_TO" => TotalHitsRelation::EqualTo,
                "GREATER_THAN_OR_EQUAL_TO" => TotalHitsRelation::GreaterThanOrEqualTo,
                relation => panic!("unknown relation {relation}"),
            }
        );
        let expected: Vec<_> = row[6..]
            .iter()
            .map(|hit| {
                let (doc, bits) = hit.split_once(':').unwrap();
                (
                    doc.parse::<i32>().unwrap(),
                    bits.parse::<i32>().unwrap() as u32,
                )
            })
            .collect();
        let actual: Vec<_> = result
            .score_docs
            .iter()
            .map(|hit| (hit.doc_id, hit.score.to_bits()))
            .collect();
        assert_eq!(actual, expected, "{field}/{ord}/{k}/{limit}");
        comparisons += 1;
    }
    assert_eq!(comparisons, 268);
}

#[test]
fn java_empty_leaf_searches_match() {
    let dir = ByteBuffersDirectory::new();
    let mut writer = plain_writer(&dir);
    let mut vector = Document::new();
    vector.add(StringField::new("id", "0", Store::No));
    vector.add(
        KnnFloatVectorField::new("float", vec![1.0, 0.0], VectorSimilarityFunction::Euclidean)
            .unwrap(),
    );
    vector.add(
        KnnByteVectorField::new("byte", vec![1, 0], VectorSimilarityFunction::Euclidean).unwrap(),
    );
    writer.add_fields_document(&vector).unwrap();
    let mut survivor = Document::new();
    survivor.add(StringField::new("id", "1", Store::No));
    writer.add_fields_document(&survivor).unwrap();
    writer.commit().unwrap();
    writer
        .delete_documents_by_term(&[Term::new("id", "0")])
        .unwrap();
    writer.commit().unwrap();
    writer.force_merge(1).unwrap();
    writer.commit().unwrap();
    let reader = DirectoryReader::open(&dir).unwrap();
    assert_eq!(reader.segment_readers().len(), 1);
    let leaf = &reader.segment_readers()[0];
    assert_eq!(float_values(leaf, "float").unwrap().size(), 0);
    assert_eq!(byte_values(leaf, "byte").unwrap().size(), 0);
    let manifest = include_str!("../../../fixtures/testframework/data/knn-empty.tsv");
    for line in manifest.lines() {
        let row: Vec<_> = line.split('\t').collect();
        let k = row[2].parse().unwrap();
        let limit = row[3].parse().unwrap();
        let result = if row[0] == "float" {
            search_nearest_float(&dir, leaf, row[1], &[1.0, 0.0], k, limit)
        } else {
            search_nearest_byte(&dir, leaf, row[1], &[1, 0], k, limit)
        };
        assert_eq!(
            result.total_hits.value,
            row[4].parse::<u64>().unwrap(),
            "{line}"
        );
        assert_eq!(row[5], "EQUAL_TO");
        assert_eq!(
            result.total_hits.relation,
            TotalHitsRelation::EqualTo,
            "{line}"
        );
        assert!(result.score_docs.is_empty(), "{line}");
    }
    assert_eq!(manifest.lines().count(), 36);
}

#[test]
fn compound_graph_searches_translate_deleted_float_and_byte_ordinals() {
    let dir = ByteBuffersDirectory::new();
    let mut writer = plain_writer(&dir);
    writer.set_use_compound_file(true);
    for id in 0..120 {
        let mut doc = Document::new();
        doc.add(StringField::new("id", id.to_string(), Store::No));
        doc.add(
            KnnFloatVectorField::new(
                "float",
                vec![id as f32, 0.0],
                VectorSimilarityFunction::Euclidean,
            )
            .unwrap(),
        );
        doc.add(
            KnnByteVectorField::new(
                "byte",
                vec![id as u8, 0],
                VectorSimilarityFunction::Euclidean,
            )
            .unwrap(),
        );
        writer.add_fields_document(&doc).unwrap();
    }
    writer.commit().unwrap();
    writer
        .delete_documents_by_term(&[Term::new("id", "0")])
        .unwrap();
    writer.commit().unwrap();
    let reader = DirectoryReader::open(&dir).unwrap();
    assert_eq!(reader.segment_readers().len(), 1);
    let leaf = &reader.segment_readers()[0];
    assert!(dir
        .list_all()
        .unwrap()
        .contains(&format!("{}.cfs", leaf.segment_name)));
    assert!(leaf.live_docs().is_some());
    for result in [
        search_nearest_float(&dir, leaf, "float", &[0.0, 0.0], 4, i32::MAX as u64),
        search_nearest_byte(&dir, leaf, "byte", &[0, 0], 4, i32::MAX as u64),
    ] {
        assert_eq!(result.total_hits.relation, TotalHitsRelation::EqualTo);
        assert!(result.total_hits.value <= 120);
        assert_eq!(
            result
                .score_docs
                .iter()
                .map(|h| h.doc_id)
                .collect::<Vec<_>>(),
            [1, 2, 3, 4]
        );
        for hit in &result.score_docs {
            let expected = 1.0 / (1.0 + (hit.doc_id * hit.doc_id) as f32);
            assert_eq!(hit.score.to_bits(), expected.to_bits());
        }
    }
}
