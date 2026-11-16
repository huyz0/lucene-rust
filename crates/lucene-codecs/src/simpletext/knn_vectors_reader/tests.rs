#![cfg(test)]
use super::super::knn_vectors_writer::SimpleTextKnnVectorsWriter;
use super::*;
use crate::field_infos::VectorEncoding;
use crate::hnsw::KnnCollector;
use lucene_util::fixed_bit_set::FixedBitSet;

fn infos() -> FieldInfos {
    FieldInfos::new(vec![
        FieldInfo::new("f", 0).with_vectors(
            2,
            VectorEncoding::Float32,
            VectorSimilarityFunction::Euclidean,
        ),
        FieldInfo::new("b", 1).with_vectors(
            3,
            VectorEncoding::Byte,
            VectorSimilarityFunction::DotProduct,
        ),
        FieldInfo::new("plain", 2),
        FieldInfo::new("absent", 3).with_vectors(
            2,
            VectorEncoding::Float32,
            VectorSimilarityFunction::Cosine,
        ),
    ])
    .unwrap()
}

/// `(vec, gri)`: `f` with docs 1, 3, 4 and `b` with doc 0.
fn files(infos: &FieldInfos) -> (Vec<u8>, Vec<u8>) {
    let mut w = SimpleTextKnnVectorsWriter::new();
    let f = [[0.0f32, 1.0], [2.0, 2.0], [0.0, 1.0]];
    w.write_float_field(
        infos.field_by_name("f").unwrap(),
        [1, 3, 4].into_iter().zip(f.iter().map(|v| &v[..])),
    )
    .unwrap();
    w.write_byte_field(
        infos.field_by_name("b").unwrap(),
        [(0, &[1u8, 0xff, 2][..])],
    )
    .unwrap();
    w.finish()
}

fn reseal(mut body: Vec<u8>) -> Vec<u8> {
    let end = body.len().saturating_sub(30);
    body.truncate(end);
    util::write_checksum(&mut body);
    body
}

fn edit(file: &[u8], from: &str, to: &str) -> Vec<u8> {
    let text = String::from_utf8(file.to_vec()).unwrap();
    assert!(text.contains(from), "{from}");
    reseal(text.replacen(from, to, 1).into_bytes())
}

#[test]
fn reads_values_iterators_and_scorers() {
    let infos = infos();
    let (vec, gri) = files(&infos);
    let r = SimpleTextKnnVectorsReader::open(&gri, &vec, &infos).unwrap();
    r.check_integrity().unwrap();
    let v = r.float_vector_values("f").unwrap().unwrap();
    assert_eq!((v.dimension(), v.size()), (2, 3));
    assert_eq!(v.vector_value(1), Some(&[2.0f32, 2.0][..]));
    assert_eq!(v.vector_value(3), None);
    assert_eq!(v.ord_to_doc(2), Some(4));
    assert_eq!(v.ord_to_doc(3), None);
    let mut it = v.iterator();
    assert_eq!((it.doc_id(), it.index(), it.cost()), (-1, -1, 3));
    assert_eq!(it.advance(3), 3);
    assert_eq!(it.index(), 1);
    assert_eq!(it.next_doc(), 4);
    assert_eq!(it.next_doc(), NO_MORE_DOCS);
    assert_eq!(it.doc_id(), NO_MORE_DOCS);
    let target = [0.0f32, 1.0];
    let mut s = v.scorer(&target).unwrap();
    assert_eq!(s.score(), None);
    assert_eq!(s.iterator().next_doc(), 1);
    assert_eq!(s.score(), Some(1.0));
    // A byte field read as floats, and the reverse: Java parses either.
    let b = r.byte_vector_values("b").unwrap().unwrap();
    assert_eq!(b.vector_value(0), Some(&[1u8, 0xff, 2][..]));
    let bf = r.float_vector_values("b").unwrap().unwrap();
    assert_eq!(bf.vector_value(0), Some(&[1.0f32, -1.0, 2.0][..]));
    assert!(r.float_vector_values("nope").unwrap().is_none());
    assert!(r.float_vector_values("absent").unwrap().is_none());
    let e = r.float_vector_values("plain").unwrap_err();
    assert!(matches!(e, Error::IllegalState(m) if m.starts_with("KNN vectors readers")));
    let fi = infos.field_by_name("f").unwrap();
    assert_eq!(r.off_heap_byte_size(fi), Some(HashMap::new()));
    assert_eq!(
        r.off_heap_byte_size(infos.field_by_name("absent").unwrap()),
        None
    );
    assert_eq!(r.field_entry(0).unwrap().size(), 3);
    // An empty field has no scorer.
    let empty = FieldEntry {
        dimension: 2,
        vector_data_offset: 0,
        vector_data_length: 0,
        ord_to_doc: Vec::new(),
        similarity_function: VectorSimilarityFunction::Cosine,
    };
    let ev = SimpleTextVectorValues::<f32>::new(&empty, b"").unwrap();
    assert!(ev.scorer(&target).is_none());
    assert_eq!(ev.iterator().next_doc(), NO_MORE_DOCS);
}

#[test]
fn searches_in_ord_order() {
    let infos = infos();
    let (vec, gri) = files(&infos);
    let r = SimpleTextKnnVectorsReader::open(&gri, &vec, &infos).unwrap();
    let mut c = KnnCollector::new(2, u64::MAX);
    r.search("f", &[0.0f32, 1.0], &mut c, None).unwrap();
    assert_eq!(c.visited_count(), 3);
    // Docs 1 and 4 tie: the lower doc first.
    assert_eq!(c.top_docs(), vec![(1, 1.0), (4, 1.0)]);
    let mut odd = FixedBitSet::new(5);
    odd.set(3);
    odd.set(1);
    let mut c = KnnCollector::new(5, u64::MAX);
    r.search("f", &[0.0f32, 1.0], &mut c, Some(&odd)).unwrap();
    assert_eq!(c.top_docs(), vec![(1, 1.0), (3, 1.0 / 6.0)]);
    // The visit limit stops the scan.
    let mut c = KnnCollector::new(5, 1);
    r.search("f", &[0.0f32, 1.0], &mut c, None).unwrap();
    assert_eq!(c.top_docs(), vec![(1, 1.0)]);
    let mut c = KnnCollector::new(5, u64::MAX);
    r.search("b", &[1u8, 1, 1], &mut c, None).unwrap();
    assert_eq!(c.top_docs().len(), 1);
    // No entry: nothing collected.
    let mut c = KnnCollector::new(5, u64::MAX);
    r.search("absent", &[0.0f32, 1.0], &mut c, None).unwrap();
    assert_eq!(c.visited_count(), 0);
    let e = r.search("f", &[0.0f32], &mut c, None).unwrap_err();
    assert!(
        matches!(&e, Error::IllegalArgument(m) if m == "vector query dimension: 1 differs from field dimension: 2"),
        "{e}"
    );
    let short = FixedBitSet::new(2);
    let e = r
        .search("f", &[0.0f32, 1.0], &mut c, Some(&short))
        .unwrap_err();
    assert!(e.to_string().contains("outside the accepted docs"), "{e}");
}

#[test]
fn checks_integrity() {
    let infos = infos();
    let (vec, gri) = files(&infos);
    let empty_vec = SimpleTextKnnVectorsWriter::new().finish().0;
    let r = SimpleTextKnnVectorsReader::open(&gri, &empty_vec, &infos).unwrap();
    r.check_integrity().unwrap();
    let mut bad = vec.clone();
    bad[2] ^= 1;
    let r = SimpleTextKnnVectorsReader::open(&gri, &bad, &infos).unwrap();
    assert!(r.check_integrity().is_err());
    let mut bad = empty_vec.clone();
    bad[12] ^= 1;
    let r = SimpleTextKnnVectorsReader::open(&gri, &bad, &infos).unwrap();
    assert!(r.check_integrity().is_err());
}

#[test]
fn refuses_a_damaged_meta() {
    let infos = infos();
    let (vec, gri) = files(&infos);
    let open = |gri: &[u8]| SimpleTextKnnVectorsReader::open(gri, &vec, &infos).err();
    let cases = [
        ("size 3", "size -1"),
        ("size 3", "size x"),
        ("field-number 1", "field-number 0"),
        ("field-name b", "field-name nope"),
        ("field-name f", "feld-name f"),
        ("field-number -1", "field-number -2"),
        ("\n4\n", "\n+4x\n"),
    ];
    for (from, to) in cases {
        assert!(open(&edit(&gri, from, to)).is_some(), "{to}");
    }
    let mut flipped = gri.clone();
    flipped[3] ^= 1;
    assert!(open(&flipped).is_some());
    assert!(open(&gri[..gri.len() - 1]).is_some());
    assert!(open(b"").is_some());
}

#[test]
fn refuses_damaged_vectors() {
    let infos = infos();
    let (vec, gri) = files(&infos);
    let values = |gri: &[u8], vec: &[u8]| -> Result<()> {
        let r = SimpleTextKnnVectorsReader::open(gri, vec, &infos)?;
        r.float_vector_values("f")?;
        r.byte_vector_values("b")?;
        Ok(())
    };
    values(&gri, &vec).unwrap();
    let gri_cases = [
        ("vector-dimension 2", "vector-dimension 3"),
        ("vector-data-offset 0", "vector-data-offset -1"),
        ("vector-data-length 33", "vector-data-length 9999"),
        ("vector-data-length 33", "vector-data-length -1"),
        ("vector-data-length 33", "vector-data-length 32"),
    ];
    for (from, to) in gri_cases {
        assert!(values(&edit(&gri, from, to), &vec).is_err(), "{to}");
    }
    // Same-length edits, so no offset moves.
    for (from, to, ok) in [
        ("[2.0, 2.0]", "[2.0, x.0]", false),
        ("[2.0, 2.0]", "[2.0,2.0,]", true),
        ("[2.0, 2.0]", "[2.0,2,3.]", false),
        ("[2.0, 2.0]", "[2.0      ", false),
        ("[1, -1, 2]", "[1, -1,  ]", false),
    ] {
        assert_eq!(values(&gri, &edit(&vec, from, to)).is_ok(), ok, "{to}");
    }
    // A line of one or no byte cannot hold its brackets.
    let mut dims = Vec::new();
    assert!(read_vector::<f32>(b"[", 1, &mut dims).is_err());
    assert!(read_vector::<f32>(b"", 1, &mut dims).is_err());
    assert!(read_vector::<f32>(b"[]", 1, &mut dims).is_err());
    assert!(read_vector::<f32>(b"[,,]", 1, &mut dims).is_err());
    read_vector::<f32>(b"x 1.5f,2,]", 2, &mut dims).unwrap();
    assert_eq!(dims, [1.5, 2.0]);
    let mut bytes = Vec::new();
    read_vector::<u8>(b"[300.7, -129, NaN]", 3, &mut bytes).unwrap();
    assert_eq!(bytes, [44, 127, 0]);
    let bad_dim = FieldEntry {
        dimension: -1,
        vector_data_offset: 0,
        vector_data_length: 0,
        ord_to_doc: vec![0],
        similarity_function: VectorSimilarityFunction::Cosine,
    };
    assert!(SimpleTextVectorValues::<f32>::new(&bad_dim, b"[1]\n").is_err());
}

/// One COSINE field of each encoding holding doc 0's vector.
fn cosine_files(f: &[f32], b: &[u8]) -> (FieldInfos, Vec<u8>, Vec<u8>) {
    let infos = FieldInfos::new(vec![
        FieldInfo::new("f", 0).with_vectors(
            2,
            VectorEncoding::Float32,
            VectorSimilarityFunction::Cosine,
        ),
        FieldInfo::new("b", 1).with_vectors(
            2,
            VectorEncoding::Byte,
            VectorSimilarityFunction::Cosine,
        ),
    ])
    .unwrap();
    let mut w = SimpleTextKnnVectorsWriter::new();
    w.write_float_field(&infos.fields[0], [(0, f)]).unwrap();
    w.write_byte_field(&infos.fields[1], [(0, b)]).unwrap();
    let (vec, gri) = w.finish();
    (infos, vec, gri)
}

#[test]
fn refuses_non_finite_values_and_scores() {
    let (infos, vec, gri) = cosine_files(&[f32::NAN, 1.0], &[1, 1]);
    let r = SimpleTextKnnVectorsReader::open(&gri, &vec, &infos).unwrap();
    let e = r.float_vector_values("f").unwrap_err();
    assert!(e.to_string().contains("non-finite value NaN"), "{e}");
    let mut c = KnnCollector::new(1, u64::MAX);
    assert!(r.search("f", &[1.0f32, 1.0], &mut c, None).is_err());
    let (infos, vec, gri) = cosine_files(&[f32::INFINITY, 1.0], &[1, 1]);
    let r = SimpleTextKnnVectorsReader::open(&gri, &vec, &infos).unwrap();
    assert!(r.float_vector_values("f").is_err());
    // A score that overflows: a DOT_PRODUCT of two huge vectors.
    let infos = FieldInfos::new(vec![FieldInfo::new("d", 0).with_vectors(
        1,
        VectorEncoding::Float32,
        VectorSimilarityFunction::DotProduct,
    )])
    .unwrap();
    let mut w = SimpleTextKnnVectorsWriter::new();
    w.write_float_field(&infos.fields[0], [(0, &[3e38f32][..])])
        .unwrap();
    let (vec, gri) = w.finish();
    let r = SimpleTextKnnVectorsReader::open(&gri, &vec, &infos).unwrap();
    let mut c = KnnCollector::new(1, u64::MAX);
    let e = r.search("d", &[3e38f32], &mut c, None).unwrap_err();
    assert!(e.to_string().contains("doc 0 scores inf"), "{e}");
}

#[test]
fn refuses_zero_cosine_vectors() {
    let (infos, vec, gri) = cosine_files(&[0.0, -0.0], &[0, 0]);
    let r = SimpleTextKnnVectorsReader::open(&gri, &vec, &infos).unwrap();
    let mut c = KnnCollector::new(1, u64::MAX);
    let e = r.search("f", &[1.0f32, 0.0], &mut c, None).unwrap_err();
    assert!(
        e.to_string().contains("zero vector in a COSINE field"),
        "{e}"
    );
    let e = r.search("b", &[1u8, 0], &mut c, None).unwrap_err();
    assert!(
        e.to_string().contains("zero vector in a COSINE field"),
        "{e}"
    );
    let (infos, vec, gri) = cosine_files(&[1.0, 0.0], &[1, 0]);
    let r = SimpleTextKnnVectorsReader::open(&gri, &vec, &infos).unwrap();
    let mut c = KnnCollector::new(1, u64::MAX);
    let e = r.search("f", &[0.0f32, 0.0], &mut c, None).unwrap_err();
    assert!(matches!(e, Error::IllegalArgument(_)), "{e}");
    let e = r.search("b", &[0u8, 0], &mut c, None).unwrap_err();
    assert!(matches!(e, Error::IllegalArgument(_)), "{e}");
    r.search("f", &[2.0f32, 0.0], &mut c, None).unwrap();
    assert_eq!(c.top_docs(), [(0, 1.0)]);
}
