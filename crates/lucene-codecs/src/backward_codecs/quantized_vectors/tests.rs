#![cfg(test)]
use super::test_support::fixture_group;
use super::*;
use crate::field_infos::{FieldInfo, VectorEncoding, VectorSimilarityFunction};

fn infos_for(flat: &FlatVectorsReader<'_>) -> FieldInfos {
    FieldInfos::new(
        flat.fields()
            .iter()
            .map(|e| {
                let mut fi = FieldInfo::new(format!("f{}", e.field_number), e.field_number);
                fi.vector_dimension = e.dimension;
                fi.vector_encoding = e.encoding;
                fi.vector_similarity_function = e.similarity;
                fi
            })
            .collect(),
    )
    .unwrap()
}

#[test]
fn formats_are_named_and_described_as_java_does() {
    for f in QuantizedFormat::ALL {
        assert_eq!(QuantizedFormat::for_name(f.name()), Some(f));
    }
    assert_eq!(QuantizedFormat::for_name("Lucene99HnswVectorsFormat"), None);
    assert!(QuantizedFormat::Lucene99HnswScalarQuantized.has_graph());
    assert!(!QuantizedFormat::Lucene102BinaryQuantized.has_graph());
    assert_eq!(
        QuantizedFormat::Lucene99ScalarQuantized.quantized_extensions(),
        ("vemq", "veq")
    );
    assert_eq!(
        QuantizedFormat::Lucene102HnswBinaryQuantized.quantized_extensions(),
        ("vemb", "veb")
    );
    assert_eq!(
        QuantizedFormat::Lucene99ScalarQuantized.search_kind(),
        SearchKind::Nothing
    );
    assert_eq!(
        QuantizedFormat::Lucene102BinaryQuantized.search_kind(),
        SearchKind::ScanAll
    );
    assert_eq!(
        QuantizedFormat::Lucene102HnswBinaryQuantized.search_kind(),
        SearchKind::Hnsw
    );
}

#[test]
fn every_fixture_group_opens_whole() {
    let mut opened = 0;
    for format in QuantizedFormat::ALL {
        for n in 0..10 {
            let Some(g) = fixture_group("10.2.2", "_0", format.name(), n) else {
                continue;
            };
            let (meta_ext, data_ext) = format.quantized_extensions();
            let files = QuantizedFiles {
                vemf: g.file("vemf"),
                vec: g.file("vec"),
                quantized_meta: g.file(meta_ext),
                quantized_data: g.file(data_ext),
                graph: format.has_graph().then(|| (g.file("vem"), g.file("vex"))),
            };
            let flat = FlatVectorsReader::open(files.vemf, files.vec, &g.id, &g.suffix).unwrap();
            let infos = infos_for(&flat);
            let r = QuantizedVectorsReader::open(format, files, &infos, &g.id, &g.suffix).unwrap();
            assert_eq!(r.format(), format);
            // `getVectorCount` (Lucene 10.6, #16647): the metadata's
            // count, through the graph or the quantized reader; a field
            // the quantized reader lacks falls to the raw one.
            for e in flat.fields() {
                let quantized = match r.quantized() {
                    QuantizedReader::Scalar(q) => q.vector_count(e.field_number, &flat),
                    QuantizedReader::Binary(q) => q.vector_count(e.field_number, &flat),
                };
                assert_eq!(quantized.unwrap(), e.size);
                assert_eq!(r.vector_count(e.field_number).unwrap(), e.size);
            }
            let raw_only = match r.quantized() {
                QuantizedReader::Scalar(q) => q.vector_count(999, &flat),
                QuantizedReader::Binary(q) => q.vector_count(999, &flat),
            };
            assert!(matches!(raw_only, Err(Error::UnknownField(999))));
            assert!(matches!(r.vector_count(999), Err(Error::UnknownField(999))));
            // A quantized entry must name a field of the segment.
            let entries = match r.quantized() {
                QuantizedReader::Scalar(q) => q.fields().len(),
                QuantizedReader::Binary(q) => q.fields().len(),
            };
            let unknown = QuantizedVectorsReader::open(
                format,
                files,
                &FieldInfos::new(Vec::new()).unwrap(),
                &g.id,
                &g.suffix,
            );
            assert_eq!(unknown.is_err(), entries > 0);
            assert_eq!(r.graph().is_some(), format.has_graph());
            r.check_integrity().unwrap();
            for e in r.flat().fields() {
                if e.encoding != VectorEncoding::Float32 {
                    continue;
                }
                let target = vec![0.25f32; e.dimension as usize];
                let mut s = r.float_scorer(e.field_number, &target).unwrap();
                assert_eq!(s.max_ord(), e.size);
                assert!(s.score(0).unwrap().is_finite());
                match (&r.quantized, &s) {
                    (QuantizedReader::Scalar(_), QuantizedFloatScorer::Scalar(_))
                    | (QuantizedReader::Binary(_), QuantizedFloatScorer::Binary(_)) => {}
                    other => panic!("{format:?}: {:?}", std::mem::discriminant(other.1)),
                }
            }
            // Without its graph files an HNSW wrapper does not open.
            if format.has_graph() {
                let mut no_graph = files;
                no_graph.graph = None;
                assert!(
                    QuantizedVectorsReader::open(format, no_graph, &infos, &g.id, &g.suffix)
                        .is_err()
                );
            }
            opened += 1;
        }
    }
    assert!(opened >= 8, "{opened} groups");
}

#[test]
fn a_field_without_a_quantizer_scores_on_the_raw_vectors() {
    // A Lucene99 entry with no vectors has no quantizer: Java hands the
    // query to the raw reader. Built from the 9.12.2 fixture's raw pair
    // plus a hand-made `.vemq` whose one entry is empty.
    use lucene_store::data_output::DataOutput;
    let g = fixture_group(
        "9.12.2",
        "_0",
        super::scalar_quantized_vectors::HNSW_NAME,
        0,
    )
    .unwrap();
    let flat = FlatVectorsReader::open(g.file("vemf"), g.file("vec"), &g.id, &g.suffix).unwrap();
    let e = flat.fields()[0].clone();
    let mut meta = Vec::new();
    let mut data = Vec::new();
    let (mc, dc) = (
        super::scalar_quantized_vectors::META_CODEC,
        super::scalar_quantized_vectors::DATA_CODEC,
    );
    lucene_store::codec_util::write_index_header(&mut meta, mc, 1, &g.id, &g.suffix);
    lucene_store::codec_util::write_index_header(&mut data, dc, 1, &g.id, &g.suffix);
    meta.write_i32(e.field_number);
    meta.write_i32(1);
    meta.write_i32(match e.similarity {
        VectorSimilarityFunction::Euclidean => 0,
        VectorSimilarityFunction::DotProduct => 1,
        VectorSimilarityFunction::Cosine => 2,
        VectorSimilarityFunction::MaximumInnerProduct => 3,
    });
    meta.write_vlong(0);
    meta.write_vlong(0);
    meta.write_vint(e.dimension);
    meta.write_i32(0);
    crate::vectors::write_stored_meta(&mut meta, &mut data, 0, &[], 10);
    meta.write_i32(-1);
    lucene_store::codec_util::write_footer(&mut meta);
    lucene_store::codec_util::write_footer(&mut data);
    let r = QuantizedVectorsReader::open(
        QuantizedFormat::Lucene99HnswScalarQuantized,
        QuantizedFiles {
            vemf: g.file("vemf"),
            vec: g.file("vec"),
            quantized_meta: &meta,
            quantized_data: &data,
            graph: Some((g.file("vem"), g.file("vex"))),
        },
        &infos_for(&flat),
        &g.id,
        &g.suffix,
    )
    .unwrap();
    let target = vec![0.5f32; e.dimension as usize];
    let mut s = r.float_scorer(e.field_number, &target).unwrap();
    assert!(matches!(s, QuantizedFloatScorer::Raw(_)));
    let mut raw = flat
        .float_vector_values(e.field_number)
        .unwrap()
        .scorer(&target)
        .unwrap();
    assert_eq!(s.max_ord(), raw.max_ord());
    assert_eq!(
        s.score(3).unwrap().to_bits(),
        raw.score(3).unwrap().to_bits()
    );
}

#[test]
fn ordinal_maps_check_their_bounds() {
    use std::sync::Arc;
    let explicit = OrdToDoc::Explicit(Arc::from(vec![2, 5, 9]));
    assert_eq!(ord_to_doc(&explicit, &[], 3, 1).unwrap(), 5);
    assert!(ord_to_doc(&explicit, &[], 4, 3).is_err());
    assert!(ord_to_doc(&explicit, &[], 3, -1).is_err());
    assert!(doc_to_ord(&explicit, &[], 3).is_err());
    assert_eq!(ord_to_doc(&OrdToDoc::Dense, &[], 3, 2).unwrap(), 2);
    assert!(matches!(
        doc_to_ord(&OrdToDoc::Empty, &[], 0).unwrap(),
        DocToOrdCursor::Empty
    ));
    let sparse = OrdToDoc::Sparse {
        docs_with_field_offset: 100,
        docs_with_field_length: 10,
        jump_table_entry_count: -1,
        dense_rank_power: 9,
        addresses_offset: 100,
        addresses_length: 10,
        meta: crate::direct_monotonic::load_meta(
            &mut lucene_store::data_input::SliceInput::new(&[0u8; 64]),
            1,
            16,
        )
        .unwrap(),
    };
    assert!(ord_to_doc(&sparse, &[0u8; 8], 1, 0).is_err());
    assert!(doc_to_ord(&sparse, &[0u8; 8], 1).is_err());
}
