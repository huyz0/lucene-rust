//! `LateInteractionField` values Java wrote (`fixtures/src/GenDocumentFields.java`,
//! field `li`) score as `MaxSim` over the tokens Java encoded.
#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::doc_values::BinaryReader;
use lucene_codecs::field_infos::DocValuesType;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::values_source::decode_multi_vector;
use lucene_store::FsDirectory;
use lucene_tensor::{decode_into_blocks, MaxSimQuery};

fn root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/document_fields")
}

/// The raw `MaxSim`, with the scalar kernel's per-token `mul_add` order.
fn naive(query: &[Vec<f32>], doc: &[Vec<f32>]) -> f32 {
    query
        .iter()
        .map(|q| {
            doc.iter()
                .map(|d| {
                    q.iter()
                        .zip(d)
                        .fold(0.0f32, |acc, (a, b)| a.mul_add(*b, acc))
                })
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .sum()
}

#[test]
fn javas_late_interaction_values_score_as_max_sim() {
    let tsv = std::fs::read_to_string(root().join("docs.tsv"))
        .expect("run scripts/gen-fixtures.sh --only GenDocumentFields");
    let mut want: Vec<Vec<Vec<f32>>> = tsv
        .lines()
        .flat_map(|l| l.split('\t'))
        .filter_map(|f| f.strip_prefix("LateInteractionField li "))
        .map(|v| {
            v.split(';')
                .map(|t| t.split(',').map(|x| x.parse().unwrap()).collect())
                .collect()
        })
        .collect();
    assert!(!want.is_empty());

    let reader = DirectoryReader::open(&FsDirectory::open(root().join("index"))).expect("open");
    let queries = [
        MaxSimQuery::new(&[vec![1.0, 0.5, -0.25]]).unwrap(),
        MaxSimQuery::new(&[
            vec![-1.0, 2.0, 0.5],
            vec![0.0, 0.0, 1.0],
            vec![3.0, -1.0, 0.0],
        ])
        .unwrap(),
    ];
    let query_rows = [
        vec![vec![1.0, 0.5, -0.25]],
        vec![
            vec![-1.0, 2.0, 0.5],
            vec![0.0, 0.0, 1.0],
            vec![3.0, -1.0, 0.0],
        ],
    ];
    let mut got = Vec::new();
    let mut scratch = Vec::new();
    let mut blocks = Vec::new();
    for seg in reader.segment_readers() {
        let Some(info) = seg.field_infos().field_by_name("li") else {
            continue;
        };
        assert_eq!(info.doc_values_type, DocValuesType::Binary);
        let (meta, data) = seg.doc_values_for_field(info.number).unwrap();
        let mut column = BinaryReader::new(data, meta.binary_entry(info.number).unwrap());
        for doc in 0..seg.max_doc {
            let Some(value) = column.value(doc).unwrap() else {
                continue;
            };
            let tokens = decode_multi_vector(value).unwrap();
            assert_eq!(
                decode_into_blocks(value, &mut blocks).unwrap(),
                (3, tokens.len())
            );
            for (q, rows) in queries.iter().zip(&query_rows) {
                let s = q.score(value, &mut scratch).unwrap();
                assert_eq!(s.to_bits(), naive(rows, &tokens).to_bits(), "doc {doc}");
            }
            got.push(tokens);
        }
    }
    // deleted documents keep their values, so every encoded one is here
    let key = |v: &Vec<Vec<f32>>| format!("{v:?}");
    want.sort_by_key(key);
    got.sort_by_key(key);
    assert_eq!(got, want);
}
