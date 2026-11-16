#![cfg(test)]
// ARITH: test fixtures of a few thousand vectors.
#![allow(clippy::arithmetic_side_effects)]

use super::*;
use crate::hnsw::{HnswGraphBuilder, HnswGraphView, DEFAULT_RAND_SEED};

/// Deterministic pseudo-random unit vectors.
fn vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    (0..n)
        .map(|_| {
            let mut v: Vec<f32> = (0..dim)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5
                })
                .collect();
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.iter_mut().for_each(|x| *x /= norm);
            v
        })
        .collect()
}

fn filled(vs: &[Vec<f32>], similarity: VectorSimilarityFunction) -> BufferGraph {
    let g = BufferGraph::new(
        VectorEncoding::Float32,
        similarity,
        vs[0].len(),
        16,
        100,
        DEFAULT_RAND_SEED,
    )
    .unwrap();
    for (i, v) in vs.iter().enumerate() {
        // Every other document carries the field.
        g.push(2 * i as i32, BufferVector::Float32(Arc::new(v.clone())))
            .unwrap();
    }
    g
}

fn edges(g: &OnHeapHnswGraph) -> Vec<Vec<Vec<i32>>> {
    g.graph
        .iter()
        .map(|levels| levels.iter().map(|a| a.nodes().to_vec()).collect())
        .collect()
}

fn exact(vs: &[Vec<f32>], q: &[f32], k: usize, keep: impl Fn(usize) -> bool) -> Vec<i32> {
    let mut scored: Vec<(i32, f32)> = vs
        .iter()
        .enumerate()
        .filter(|(i, _)| keep(*i))
        .map(|(i, v)| {
            (
                2 * i as i32,
                VectorSimilarityFunction::DotProduct.score(q, v),
            )
        })
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    scored.into_iter().take(k).map(|(d, _)| d).collect()
}

/// The graph a buffer ends with is the flush builder's, for both
/// similarities that score differently, whenever the inserts ran.
#[test]
fn the_graph_is_the_flush_builders() {
    let vs = vectors(1500, 24);
    for similarity in [
        VectorSimilarityFunction::DotProduct,
        VectorSimilarityFunction::Cosine,
        VectorSimilarityFunction::Euclidean,
    ] {
        let g = filled(&vs, similarity);
        assert!(g.schedule());
        assert!(!g.schedule(), "scheduled once until the run ends");
        g.insert_batch(700).unwrap();
        // The flush's builder, over the flat file the flush writes.
        let id = *b"buffervectors001";
        let field = crate::vectors::FlatVectorsField {
            field_number: 0,
            similarity,
            dimension: 24,
            docs: (0..1500).map(|i| 2 * i).collect(),
            values: crate::vectors::FieldVectorData::Float32(
                vs.iter().flatten().copied().collect(),
            ),
        };
        let (vec, vemf) = crate::vectors::write_flat_vectors(&[field], 3000, &id, "").unwrap();
        let flat = crate::vectors::FlatVectorsReader::open(&vemf, &vec, &id, "").unwrap();
        let values = flat.float_vector_values(0).unwrap();
        let built = HnswGraphBuilder::new(values.ord_scorer(), 16, 100, DEFAULT_RAND_SEED)
            .unwrap()
            .build(1500)
            .unwrap();
        let ours = g.graph(1500).unwrap();
        assert_eq!(edges(&ours), edges(&built), "{similarity:?}");
        assert_eq!(ours.entry_node(), built.entry_node());
        assert_eq!(ours.num_levels(), built.num_levels());
        g.insert_pending(64).unwrap();
        assert_eq!(g.inserted(), 1500);
        assert!(!g.schedule());
    }
}

/// Searches find what an exact scan finds, over the graph, the
/// vectors not inserted yet, a reader's shorter count, and a filter.
#[test]
fn searches_match_an_exact_scan() {
    let n = 6000;
    let vs = vectors(n, 96);
    let g = filled(&vs, VectorSimilarityFunction::DotProduct);
    g.insert_batch(4500).unwrap();
    let queries = vectors(30, 96);
    let mut found = 0usize;
    let mut wanted = 0usize;
    for (qi, q) in queries.iter().enumerate() {
        let query = BufferQuery::Float32(q.clone());
        for (count, keep) in [(n, None), (5000, None), (n, Some(3usize)), (n, Some(500))] {
            let accept = keep.map(|m| {
                let mut bits = FixedBitSet::new(n);
                (0..n)
                    .filter(|i| (i + qi).is_multiple_of(m))
                    .for_each(|i| bits.set(i));
                bits
            });
            let hits = g.search(&query, 10, count as i32, accept.as_ref()).unwrap();
            let truth = exact(&vs[..count], q, 10, |i| {
                keep.is_none_or(|m| (i + qi).is_multiple_of(m))
            });
            assert!(hits.windows(2).all(|w| w[0].1 >= w[1].1));
            assert!(hits.iter().all(|&(d, _)| (d as usize) < 2 * count));
            if let Some(m) = keep {
                assert!(hits
                    .iter()
                    .all(|&(d, _)| (d as usize / 2 + qi).is_multiple_of(m)));
            }
            assert_eq!(hits.len(), truth.len());
            found += hits.iter().filter(|(d, _)| truth.contains(d)).count();
            wanted += truth.len();
        }
    }
    let recall = found as f64 / wanted as f64;
    assert!(recall > 0.95, "recall {recall}");
}

/// A small buffer, or no vectors at all, is scored exactly; a query
/// of the wrong shape is refused.
#[test]
fn small_and_bad_searches() {
    let vs = vectors(50, 8);
    let g = filled(&vs, VectorSimilarityFunction::DotProduct);
    let q = BufferQuery::Float32(vs[7].clone());
    let hits = g.search(&q, 3, 50, None).unwrap();
    assert_eq!(
        hits.iter().map(|h| h.0).collect::<Vec<_>>(),
        exact(&vs, &vs[7], 3, |_| true)
    );
    assert!(g.search(&q, 3, 0, None).unwrap().is_empty());
    assert!(g
        .search(&BufferQuery::Float32(vec![0.0; 3]), 3, 50, None)
        .is_err());
    assert!(g
        .search(&BufferQuery::Byte(vec![0; 8]), 3, 50, None)
        .is_err());
    assert!(g
        .push(100, BufferVector::Float32(Arc::new(vec![0.0; 3])))
        .is_err());
    assert!(g
        .push(100, BufferVector::Byte(Arc::new(vec![0; 8])))
        .is_err());
    assert!(g.graph(51).is_err());
    assert_eq!(g.doc(3), Some(6));
    assert_eq!(g.doc(50), None);
    assert_eq!(g.params(), (16, 100));
    assert!(!g.is_empty());
    assert!(format!("{g:?}").contains("vectors: 50"));
}

/// Byte vectors score as the flat byte scorer does.
#[test]
fn byte_vectors() {
    let g = BufferGraph::new(
        VectorEncoding::Byte,
        VectorSimilarityFunction::Euclidean,
        4,
        8,
        20,
        DEFAULT_RAND_SEED,
    )
    .unwrap();
    for i in 0..100u8 {
        g.push(i32::from(i), BufferVector::Byte(Arc::new(vec![i, i, 0, 1])))
            .unwrap();
    }
    g.insert_pending(10).unwrap();
    let hits = g
        .search(&BufferQuery::Byte(vec![40, 40, 0, 1]), 2, 100, None)
        .unwrap();
    assert_eq!(hits[0], (40, 1.0));
    assert_eq!(hits[1].0, 39);
}

/// A byte field's graph is the flush builder's over the flat byte
/// scorer too.
#[test]
fn byte_graph_is_the_flush_builders() {
    let n = 800usize;
    let bytes: Vec<Vec<u8>> = vectors(n, 12)
        .iter()
        .map(|v| v.iter().map(|x| (x * 120.0) as i8 as u8).collect())
        .collect();
    for similarity in [
        VectorSimilarityFunction::Euclidean,
        VectorSimilarityFunction::DotProduct,
    ] {
        let g = BufferGraph::new(
            VectorEncoding::Byte,
            similarity,
            12,
            8,
            40,
            DEFAULT_RAND_SEED,
        )
        .unwrap();
        for (i, v) in bytes.iter().enumerate() {
            g.push(i as i32, BufferVector::Byte(Arc::new(v.clone())))
                .unwrap();
        }
        let id = *b"buffervectors002";
        let field = crate::vectors::FlatVectorsField {
            field_number: 0,
            similarity,
            dimension: 12,
            docs: (0..n as i32).collect(),
            values: crate::vectors::FieldVectorData::Byte(
                bytes.iter().flatten().copied().collect(),
            ),
        };
        let (vec, vemf) = crate::vectors::write_flat_vectors(&[field], n as i32, &id, "").unwrap();
        let flat = crate::vectors::FlatVectorsReader::open(&vemf, &vec, &id, "").unwrap();
        let values = flat.byte_vector_values(0).unwrap();
        let built = HnswGraphBuilder::new(values.ord_scorer(), 8, 40, DEFAULT_RAND_SEED)
            .unwrap()
            .build(n as i32)
            .unwrap();
        let ours = g.graph(n as i32).unwrap();
        assert_eq!(edges(&ours), edges(&built), "{similarity:?}");
        assert_eq!(g.shape(), (VectorEncoding::Byte, similarity, 12));
    }
}

/// A flush-only graph is the flush builder's too, float and byte, and
/// hands its graph over once.
#[test]
fn a_flush_graph_is_the_flush_builders() {
    let vs = vectors(1200, 24);
    for similarity in [
        VectorSimilarityFunction::DotProduct,
        VectorSimilarityFunction::Cosine,
        VectorSimilarityFunction::Euclidean,
    ] {
        let g = FlushGraph::new(
            VectorEncoding::Float32,
            similarity,
            24,
            16,
            100,
            DEFAULT_RAND_SEED,
        )
        .unwrap();
        assert!(g.is_empty());
        for (i, v) in vs.iter().enumerate() {
            g.add(3 * i as i32, BufferVector::Float32(Arc::new(v.clone())))
                .unwrap();
        }
        assert_eq!(g.len(), 1200);
        assert_eq!(g.docs()[7], 21);
        assert_eq!(g.params(), (16, 100));
        assert_eq!(g.shape(), (VectorEncoding::Float32, similarity, 24));
        assert!(g.ram_bytes() > 1200 * 33 * 8);
        let id = *b"buffervectors003";
        let field = crate::vectors::FlatVectorsField {
            field_number: 0,
            similarity,
            dimension: 24,
            docs: (0..1200).collect(),
            values: crate::vectors::FieldVectorData::Float32(
                vs.iter().flatten().copied().collect(),
            ),
        };
        let (vec, vemf) = crate::vectors::write_flat_vectors(&[field], 1200, &id, "").unwrap();
        let flat = crate::vectors::FlatVectorsReader::open(&vemf, &vec, &id, "").unwrap();
        let values = flat.float_vector_values(0).unwrap();
        let built = HnswGraphBuilder::new(values.ord_scorer(), 16, 100, DEFAULT_RAND_SEED)
            .unwrap()
            .build(1200)
            .unwrap();
        assert!(g.graph(1199).is_err(), "not every vector");
        let ours = g.graph(1200).unwrap();
        assert_eq!(edges(&ours), edges(&built), "{similarity:?}");
        assert_eq!(ours.entry_node(), built.entry_node());
        assert!(g.graph(1200).is_err(), "taken once");
        assert!(g
            .add(9999, BufferVector::Float32(Arc::new(vs[0].clone())))
            .is_err());
    }

    let bytes: Vec<Vec<u8>> = vectors(500, 12)
        .iter()
        .map(|v| v.iter().map(|x| (x * 120.0) as i8 as u8).collect())
        .collect();
    let g = FlushGraph::new(
        VectorEncoding::Byte,
        VectorSimilarityFunction::Euclidean,
        12,
        8,
        40,
        DEFAULT_RAND_SEED,
    )
    .unwrap();
    for (i, v) in bytes.iter().enumerate() {
        g.add(i as i32, BufferVector::Byte(Arc::new(v.clone())))
            .unwrap();
    }
    let id = *b"buffervectors004";
    let field = crate::vectors::FlatVectorsField {
        field_number: 0,
        similarity: VectorSimilarityFunction::Euclidean,
        dimension: 12,
        docs: (0..500).collect(),
        values: crate::vectors::FieldVectorData::Byte(bytes.iter().flatten().copied().collect()),
    };
    let (vec, vemf) = crate::vectors::write_flat_vectors(&[field], 500, &id, "").unwrap();
    let flat = crate::vectors::FlatVectorsReader::open(&vemf, &vec, &id, "").unwrap();
    let values = flat.byte_vector_values(0).unwrap();
    let built = HnswGraphBuilder::new(values.ord_scorer(), 8, 40, DEFAULT_RAND_SEED)
        .unwrap()
        .build(500)
        .unwrap();
    assert_eq!(edges(&g.graph(500).unwrap()), edges(&built));
}

/// A flush-only graph refuses a vector of another shape, and one
/// retired takes no more.
#[test]
fn a_flush_graph_refuses_bad_vectors_and_retires() {
    let g = FlushGraph::new(
        VectorEncoding::Float32,
        VectorSimilarityFunction::DotProduct,
        4,
        8,
        20,
        DEFAULT_RAND_SEED,
    )
    .unwrap();
    assert!(g
        .add(0, BufferVector::Float32(Arc::new(vec![0.0; 3])))
        .is_err());
    assert!(g.add(0, BufferVector::Byte(Arc::new(vec![0; 4]))).is_err());
    g.add(0, BufferVector::Float32(Arc::new(vec![1.0, 0.0, 0.0, 0.0])))
        .unwrap();
    assert!(format!("{g:?}").contains("FlushGraph"));
    g.retire();
    assert!(g
        .add(1, BufferVector::Float32(Arc::new(vec![0.0, 1.0, 0.0, 0.0])))
        .is_err());
    assert!(g.graph(1).is_err());
    assert!(FlushGraph::new(
        VectorEncoding::Byte,
        VectorSimilarityFunction::DotProduct,
        4,
        0,
        20,
        DEFAULT_RAND_SEED
    )
    .is_err());
}

/// A failed graph is never handed to a flush and schedules no more
/// inserts.
#[test]
fn a_failed_graph_is_refused() {
    let vs = vectors(100, 8);
    let g = filled(&vs, VectorSimilarityFunction::DotProduct);
    assert!(g.graph(99).is_err(), "not every vector");
    g.fail();
    assert!(!g.schedule());
    assert!(g.insert_pending(10).is_err());
    assert!(g.graph(100).is_err());
    // What was published stays searchable, the rest scored exactly.
    let hits = g
        .search(&BufferQuery::Float32(vs[3].clone()), 1, 100, None)
        .unwrap();
    assert_eq!(hits[0].0, 6);
}

/// Searches read the graph while the inserter grows it.
#[test]
fn searches_while_inserting() {
    let n = 5000;
    let vs = vectors(n, 64);
    let g = filled(&vs, VectorSimilarityFunction::DotProduct);
    std::thread::scope(|s| {
        s.spawn(|| g.insert_pending(32).unwrap());
        for q in vectors(50, 64) {
            let hits = g
                .search(&BufferQuery::Float32(q.clone()), 5, n as i32, None)
                .unwrap();
            assert_eq!(hits.len(), 5);
        }
    });
    let _ = g.graph(n as i32).unwrap();
}
