#![cfg(test)]
use super::*;
use crate::document::{IndexOptions, VectorEncoding, VectorSimilarityFunction};
use crate::index_writer::IndexWriter;
use crate::segment_info::LuceneVersion;
use lucene_store::directory::FsDirectory;
use lucene_util::test_support::TempDir;

fn ft(f: impl FnOnce(&mut FieldType)) -> FieldType {
    let mut t = FieldType::new();
    f(&mut t);
    t.frozen()
}

struct L {
    c: Column,
    kind: NumericKind,
    tuples: Vec<(i32, i64)>,
    dense: Vec<i64>,
}
impl LongColumn for L {
    fn column(&self) -> &Column {
        &self.c
    }
    fn numeric_kind(&self) -> NumericKind {
        self.kind
    }
    fn tuples(&self) -> Box<dyn LongTupleCursor + '_> {
        Box::new(VecLongTupleCursor::new(&self.tuples))
    }
    fn values(&self) -> Result<Box<dyn LongValuesCursor + '_>> {
        Ok(Box::new(VecLongValuesCursor::new(&self.dense)))
    }
}

struct B {
    c: Column,
    stored: StoredType,
    tuples: Vec<(i32, Vec<u8>)>,
}
impl BinaryColumn for B {
    fn column(&self) -> &Column {
        &self.c
    }
    fn stored_type(&self) -> StoredType {
        self.stored
    }
    fn tuples(&self) -> Box<dyn ObjectTupleCursor<Vec<u8>> + '_> {
        Box::new(VecObjectTupleCursor::new(&self.tuples))
    }
}

struct D {
    c: Column,
    dict: Vec<Vec<u8>>,
    tuples: Vec<(i32, i32)>,
}
impl DictionaryColumn for D {
    fn column(&self) -> &Column {
        &self.c
    }
    fn dictionary(&self) -> &[Vec<u8>] {
        &self.dict
    }
    fn tuples(&self) -> Box<dyn OrdinalsTupleCursor + '_> {
        Box::new(VecOrdinalsTupleCursor::new(&self.tuples))
    }
}

struct T {
    c: Column,
    tuples: Vec<(i32, FieldTokens)>,
}
impl TokenStreamColumn for T {
    fn column(&self) -> &Column {
        &self.c
    }
    fn tuples(&self) -> Box<dyn ObjectTupleCursor<FieldTokens> + '_> {
        Box::new(VecObjectTupleCursor::new(&self.tuples))
    }
}

struct V {
    c: Column,
    tuples: Vec<(i32, Vec<u8>)>,
}
impl VectorColumn<Vec<u8>> for V {
    fn column(&self) -> &Column {
        &self.c
    }
    fn tuples(&self) -> Box<dyn ObjectTupleCursor<Vec<u8>> + '_> {
        Box::new(VecObjectTupleCursor::new(&self.tuples))
    }
}

struct Batch<'a> {
    n: usize,
    cols: Vec<BatchColumn<'a>>,
}
impl ColumnBatch for Batch<'_> {
    fn num_docs(&self) -> usize {
        self.n
    }
    fn columns(&self) -> Vec<BatchColumn<'_>> {
        self.cols.clone()
    }
}

fn numeric() -> FieldType {
    ft(|t| t.set_doc_values_type(DocValuesType::Numeric).unwrap())
}

fn long(name: &str, t: FieldType, density: Density, tuples: Vec<(i32, i64)>) -> L {
    L {
        c: Column::new(name, t, density),
        kind: NumericKind::Long,
        dense: tuples.iter().map(|t| t.1).collect(),
        tuples,
    }
}

fn err(batch: &dyn ColumnBatch) -> String {
    batch_documents(batch).unwrap_err().to_string()
}

struct FV {
    c: Column,
    tuples: Vec<(i32, Vec<f32>)>,
}
impl VectorColumn<Vec<f32>> for FV {
    fn column(&self) -> &Column {
        &self.c
    }
    fn tuples(&self) -> Box<dyn ObjectTupleCursor<Vec<f32>> + '_> {
        Box::new(VecObjectTupleCursor::new(&self.tuples))
    }
}

fn vector_type(encoding: VectorEncoding, sim: VectorSimilarityFunction) -> FieldType {
    ft(|t| t.set_vector_attributes(2, encoding, sim).unwrap())
}

/// Lucene 10.6's `TestColumnBatchVectorColumn.testFloatVectorValueValidation`
/// (#16732): a column refuses what `KnnFloatVectorField` refuses.
#[test]
fn float_vector_values_are_validated() {
    let euclidean = vector_type(VectorEncoding::Float32, VectorSimilarityFunction::Euclidean);
    let cosine = vector_type(VectorEncoding::Float32, VectorSimilarityFunction::Cosine);
    for (density, n, ty, tuples, want) in [
        (
            Density::Dense,
            2,
            &euclidean,
            vec![(0, vec![1.0, 2.0]), (1, vec![3.0, f32::NAN])],
            "VectorColumn \"v\" has non-finite value at vector[1]=NaN at batch doc 1",
        ),
        (
            Density::Dense,
            2,
            &euclidean,
            vec![(0, vec![1.0, 2.0]), (1, vec![3.0, f32::INFINITY])],
            "non-finite value at vector[1]=Infinity at batch doc 1",
        ),
        (
            Density::Sparse,
            3,
            &euclidean,
            vec![(0, vec![1.0, 2.0]), (2, vec![f32::NEG_INFINITY, 4.0])],
            "non-finite value at vector[0]=-Infinity at batch doc 2",
        ),
        (
            Density::Dense,
            2,
            &cosine,
            vec![(0, vec![1.0, 2.0]), (1, vec![0.0, -0.0])],
            "VectorColumn \"v\" has a zero vector at batch doc 1, which is not allowed with \
                 cosine similarity function",
        ),
    ] {
        let v = FV {
            c: Column::new("v", ty.clone(), density),
            tuples,
        };
        let e = err(&Batch {
            n,
            cols: vec![BatchColumn::FloatVector(&v)],
        });
        assert!(e.contains(want), "{e}");
    }
    // A zero vector is fine without cosine.
    let v = FV {
        c: Column::new("v", euclidean, Density::Dense),
        tuples: vec![(0, vec![0.0, 0.0])],
    };
    assert!(batch_documents(&Batch {
        n: 1,
        cols: vec![BatchColumn::FloatVector(&v)],
    })
    .is_ok());
}

/// Lucene 10.6's `TestColumnBatchVectorColumn.testByteVectorValueValidation`
/// (#16732).
#[test]
fn byte_vector_values_are_validated() {
    let cosine = vector_type(VectorEncoding::Byte, VectorSimilarityFunction::Cosine);
    let v = V {
        c: Column::new("v", cosine.clone(), Density::Dense),
        tuples: vec![(0, vec![0, 0]), (1, vec![1, 2])],
    };
    let e = err(&Batch {
        n: 2,
        cols: vec![BatchColumn::ByteVector(&v)],
    });
    assert!(e.contains("zero vector at batch doc 0"), "{e}");
    let v = V {
        c: Column::new("v", cosine, Density::Dense),
        tuples: vec![(0, vec![1, 0])],
    };
    assert!(batch_documents(&Batch {
        n: 1,
        cols: vec![BatchColumn::ByteVector(&v)],
    })
    .is_ok());
}

#[test]
fn columns_are_validated_as_process_batch_validates_them() {
    let none = long("a", FieldType::new(), Density::Sparse, vec![]);
    let e = err(&Batch {
        n: 1,
        cols: vec![BatchColumn::Long(&none)],
    });
    assert!(e.contains("must have a non-NONE"), "{e}");
    let two_dims = long(
        "a",
        ft(|t| t.set_dimensions(2, 8).unwrap()),
        Density::Sparse,
        vec![],
    );
    assert!(err(&Batch {
        n: 1,
        cols: vec![BatchColumn::Long(&two_dims)]
    })
    .contains("1-dimensional"));
    let mut int = long(
        "a",
        ft(|t| t.set_dimensions(1, 8).unwrap()),
        Density::Sparse,
        vec![],
    );
    int.kind = NumericKind::Int;
    assert!(err(&Batch {
        n: 1,
        cols: vec![BatchColumn::Long(&int)]
    })
    .contains("pointNumBytes=4"));
    let indexed = long(
        "a",
        ft(|t| t.set_index_options(IndexOptions::Docs).unwrap()),
        Density::Sparse,
        vec![],
    );
    assert!(err(&Batch {
        n: 1,
        cols: vec![BatchColumn::Long(&indexed)]
    })
    .contains("inverted"));
    let sorted = long(
        "a",
        ft(|t| t.set_doc_values_type(DocValuesType::Sorted).unwrap()),
        Density::Sparse,
        vec![],
    );
    assert!(err(&Batch {
        n: 1,
        cols: vec![BatchColumn::Long(&sorted)]
    })
    .contains("incompatible"));
    let bin_num = B {
        c: Column::new("b", numeric(), Density::Sparse),
        stored: StoredType::Binary,
        tuples: vec![],
    };
    assert!(err(&Batch {
        n: 1,
        cols: vec![BatchColumn::Binary(&bin_num)]
    })
    .contains("use a LongColumn"));
    let bin_stored = B {
        c: Column::new("b", ft(|t| t.set_stored(true).unwrap()), Density::Sparse),
        stored: StoredType::Long,
        tuples: vec![],
    };
    assert!(err(&Batch {
        n: 1,
        cols: vec![BatchColumn::Binary(&bin_stored)]
    })
    .contains("storedType"));
    let dict = |t: FieldType, dict: Vec<Vec<u8>>| D {
        c: Column::new("d", t, Density::Sparse),
        dict,
        tuples: vec![(0, 5)],
    };
    let sorted_dv = ft(|t| t.set_doc_values_type(DocValuesType::Sorted).unwrap());
    for (d, want) in [
        (dict(sorted_dv.clone(), vec![]), "must not be empty"),
        (dict(sorted_dv.clone(), vec![vec![0; 40000]]), "too large"),
        (dict(numeric(), vec![vec![1]]), "use a LongColumn"),
        (
            dict(
                ft(|t| t.set_doc_values_type(DocValuesType::Binary).unwrap()),
                vec![vec![1]],
            ),
            "BINARY",
        ),
        (
            dict(
                ft(|t| {
                    t.set_dimensions(1, 1).unwrap();
                    t.set_doc_values_type(DocValuesType::Sorted).unwrap();
                }),
                vec![vec![1]],
            ),
            "does not support points",
        ),
        (
            dict(sorted_dv.clone(), vec![vec![1]]),
            "ordinal 5 is out of range",
        ),
    ] {
        let e = err(&Batch {
            n: 1,
            cols: vec![BatchColumn::Dictionary(&d)],
        });
        assert!(e.contains(want), "{e}");
    }
    let tokens = |t: FieldType| T {
        c: Column::new("t", t, Density::Sparse),
        tuples: vec![],
    };
    let t = tokens(ft(|t| t.set_stored(true).unwrap()));
    assert!(err(&Batch {
        n: 1,
        cols: vec![BatchColumn::TokenStream(&t)]
    })
    .contains("tokenized == true"));
    let t = tokens(ft(|t| {
        t.set_index_options(IndexOptions::Docs).unwrap();
        t.set_stored(true).unwrap();
    }));
    assert!(err(&Batch {
        n: 1,
        cols: vec![BatchColumn::TokenStream(&t)]
    })
    .contains("inverted-only"));
    let vec_type = ft(|t| {
        t.set_vector_attributes(
            2,
            VectorEncoding::Byte,
            VectorSimilarityFunction::DotProduct,
        )
        .unwrap()
    });
    let v = V {
        c: Column::new("v", ft(|t| t.set_stored(true).unwrap()), Density::Sparse),
        tuples: vec![],
    };
    assert!(err(&Batch {
        n: 1,
        cols: vec![BatchColumn::ByteVector(&v)]
    })
    .contains("vectorDimension() > 0"));
    let v = V {
        c: Column::new(
            "v",
            ft(|t| {
                t.set_vector_attributes(
                    2,
                    VectorEncoding::Byte,
                    VectorSimilarityFunction::DotProduct,
                )
                .unwrap();
                t.set_stored(true).unwrap();
            }),
            Density::Sparse,
        ),
        tuples: vec![],
    };
    assert!(err(&Batch {
        n: 1,
        cols: vec![BatchColumn::ByteVector(&v)]
    })
    .contains("vector-only"));
    for (tuples, want) in [
        (
            vec![(0, vec![1u8, 2]), (0, vec![1, 2])],
            "strictly increasing",
        ),
        (vec![(0, vec![1u8])], "expected dimension 2"),
        (vec![(3, vec![1u8, 2])], "out of range"),
    ] {
        let v = V {
            c: Column::new("v", vec_type.clone(), Density::Sparse),
            tuples,
        };
        let e = err(&Batch {
            n: 1,
            cols: vec![BatchColumn::ByteVector(&v)],
        });
        assert!(e.contains(want), "{e}");
    }
    // One feature per field across the batch's columns.
    let a = long("x", numeric(), Density::Sparse, vec![]);
    let b = long("x", numeric(), Density::Sparse, vec![]);
    let e = err(&Batch {
        n: 1,
        cols: vec![BatchColumn::Long(&a), BatchColumn::Long(&b)],
    });
    assert!(e.contains("[doc values]"), "{e}");
    // Doc ids in range and not going backwards; dense counts matching.
    let back = long("x", numeric(), Density::Sparse, vec![(1, 1), (0, 1)]);
    assert!(err(&Batch {
        n: 2,
        cols: vec![BatchColumn::Long(&back)]
    })
    .contains("out-of-order"));
    let neg = long("x", numeric(), Density::Sparse, vec![(-1, 1)]);
    assert!(err(&Batch {
        n: 2,
        cols: vec![BatchColumn::Long(&neg)]
    })
    .contains("out of range"));
    let short = long("x", numeric(), Density::Dense, vec![(0, 1)]);
    assert!(err(&Batch {
        n: 2,
        cols: vec![BatchColumn::Long(&short)]
    })
    .contains("provided 1 values"));
    assert_eq!(
        ColumnValidation::feature_names(
            ColumnValidation::FEATURE_INVERSION
                | ColumnValidation::FEATURE_STORED
                | ColumnValidation::FEATURE_POINTS
                | ColumnValidation::FEATURE_VECTOR
        ),
        "[inversion, stored, points, vectors]"
    );
}

#[test]
fn sparse_columns_without_values_have_no_dense_view() {
    struct OnlyTuples(Column);
    impl LongColumn for OnlyTuples {
        fn column(&self) -> &Column {
            &self.0
        }
        fn tuples(&self) -> Box<dyn LongTupleCursor + '_> {
            Box::new(VecLongTupleCursor::new(&[]))
        }
    }
    impl BinaryColumn for OnlyTuples {
        fn column(&self) -> &Column {
            &self.0
        }
        fn tuples(&self) -> Box<dyn ObjectTupleCursor<Vec<u8>> + '_> {
            Box::new(VecObjectTupleCursor::new(&[]))
        }
    }
    impl DictionaryColumn for OnlyTuples {
        fn column(&self) -> &Column {
            &self.0
        }
        fn dictionary(&self) -> &[Vec<u8>] {
            &[]
        }
        fn tuples(&self) -> Box<dyn OrdinalsTupleCursor + '_> {
            Box::new(VecOrdinalsTupleCursor::new(&[]))
        }
    }
    let o = OnlyTuples(Column::new("o", numeric(), Density::Dense));
    assert_eq!(LongColumn::numeric_kind(&o), NumericKind::Long);
    assert!(LongColumn::values(&o).is_err());
    assert_eq!(BinaryColumn::stored_type(&o), StoredType::Binary);
    assert!(BinaryColumn::values(&o).is_err());
    assert_eq!(DictionaryColumn::stored_type(&o), StoredType::Binary);
    assert!(DictionaryColumn::values(&o).is_err());
    assert_eq!(o.0.density(), Density::Dense);
    for k in [
        NumericKind::Int,
        NumericKind::Long,
        NumericKind::Float,
        NumericKind::Double,
    ] {
        let _ = k.stored_type();
    }
    // The in-memory cursors past their end.
    let mut c = VecLongValuesCursor::new(&[]);
    assert_eq!((c.size(), c.next_long()), (0, 0));
    let mut c = VecBytesValuesCursor::new(&[]);
    assert_eq!((c.size(), c.next_value()), (0, Vec::new()));
    let mut c = VecOrdinalsCursor::new(&[]);
    assert_eq!((c.size(), c.next_ord()), (0, -1));
    let mut c = VecOrdinalsTupleCursor::new(&[]);
    assert_eq!((c.ord_value(), c.next_doc()), (-1, NO_MORE_DOCS));
    let c = VecLongTupleCursor::new(&[]);
    assert_eq!(c.long_value(), 0);
    let c = VecObjectTupleCursor::<Vec<u8>>::new(&[]);
    assert_eq!(c.value(), Vec::<u8>::new());
}

#[test]
fn adapters_view_a_cell_as_a_field() {
    let analyzer = Analyzer::standard(None);
    let stored_long = ft(|t| {
        t.set_stored(true).unwrap();
        t.set_dimensions(1, 4).unwrap();
    });
    let c = Column::new("n", stored_long, Density::Dense);
    let f = LongColumnAdapter::new(&c, NumericKind::Float, i64::from(1_065_353_216));
    assert_eq!(f.stored_value(), Some(StoredValue::Float(1.0)));
    assert_eq!(f.binary_value().unwrap().len(), 4);
    assert_eq!(f.numeric_value(), Some(Number::Long(1_065_353_216)));
    assert!(f.token_stream(&analyzer).unwrap().is_none());
    assert_eq!(f.name(), "n");
    assert_eq!(
        LongColumnAdapter::new(&c, NumericKind::Int, 5).stored_value(),
        Some(StoredValue::Int(5))
    );
    let unstored = Column::new("n", numeric(), Density::Dense);
    assert_eq!(
        LongColumnAdapter::new(&unstored, NumericKind::Long, 1).stored_value(),
        None
    );
    let text = ft(|t| {
        t.set_index_options(IndexOptions::DocsAndFreqsAndPositions)
            .unwrap();
        t.set_stored(true).unwrap();
    });
    let c = Column::new("t", text, Density::Sparse);
    let b = BinaryColumnAdapter::new(&c, StoredType::String, Arc::from(&b"Hello World"[..]));
    assert_eq!(b.string_value().as_deref(), Some("Hello World"));
    assert_eq!(b.invertable_type(), InvertableType::TokenStream);
    assert_eq!(b.token_stream(&analyzer).unwrap().unwrap().tokens.len(), 2);
    assert_eq!(
        b.stored_value(),
        Some(StoredValue::String("Hello World".into()))
    );
    let t = BinaryColumnAdapter::from_tokens(&c, FieldTokens::default());
    assert_eq!(t.binary_value(), None);
    assert_eq!(t.stored_value(), None);
    assert_eq!(
        t.token_stream(&analyzer).unwrap(),
        Some(FieldTokens::default())
    );
    let kw = ft(|t| {
        t.set_index_options(IndexOptions::Docs).unwrap();
        t.set_tokenized(false).unwrap();
        t.set_stored(true).unwrap();
    });
    let c = Column::new("k", kw, Density::Sparse);
    let k = BinaryColumnAdapter::new(&c, StoredType::Binary, Arc::from(&b"k"[..]));
    assert_eq!(k.invertable_type(), InvertableType::Binary);
    assert_eq!(k.string_value(), None);
    assert!(k.token_stream(&analyzer).unwrap().is_none());
    assert_eq!(k.stored_value(), Some(StoredValue::Binary(b"k".to_vec())));
    assert_eq!(k.name(), "k");
}

#[test]
fn a_batch_writes_and_updates() {
    let tmp = TempDir::new("column-batch");
    let dir = FsDirectory::open(tmp.path());
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", version).unwrap();
    let id = D {
        c: Column::new(
            "id",
            ft(|t| {
                t.set_index_options(IndexOptions::Docs).unwrap();
                t.set_tokenized(false).unwrap();
                t.set_omit_norms(true).unwrap();
            }),
            Density::Dense,
        ),
        dict: vec![b"a".to_vec(), b"b".to_vec()],
        tuples: vec![(0, 0), (1, 1)],
    };
    struct Dense<'a>(&'a D, Vec<i32>);
    impl DictionaryColumn for Dense<'_> {
        fn column(&self) -> &Column {
            &self.0.c
        }
        fn dictionary(&self) -> &[Vec<u8>] {
            &self.0.dict
        }
        fn tuples(&self) -> Box<dyn OrdinalsTupleCursor + '_> {
            Box::new(VecOrdinalsTupleCursor::new(&self.0.tuples))
        }
        fn values(&self) -> Result<Box<dyn OrdinalsCursor + '_>> {
            Ok(Box::new(VecOrdinalsCursor::new(&self.1)))
        }
    }
    let dense = Dense(&id, vec![0, 1]);
    let v = V {
        c: Column::new(
            "v",
            ft(|t| {
                t.set_vector_attributes(
                    2,
                    VectorEncoding::Byte,
                    VectorSimilarityFunction::DotProduct,
                )
                .unwrap()
            }),
            Density::Sparse,
        ),
        tuples: vec![(1, vec![1, 2])],
    };
    // A second column for `id`, claiming a different feature.
    let id_values = long("id", numeric(), Density::Dense, vec![(0, 1), (1, 2)]);
    let batch = Batch {
        n: 2,
        cols: vec![
            BatchColumn::Dictionary(&dense),
            BatchColumn::ByteVector(&v),
            BatchColumn::Long(&id_values),
        ],
    };
    w.add_batch(&batch).unwrap();
    w.update_batch(crate::buffered_updates::Term::new("id", "a"), &batch)
        .unwrap();
    w.commit().unwrap();
    assert_eq!(w.committed_doc_count().unwrap(), 4);
    let deleted: i32 = w.segment_infos().segments.iter().map(|s| s.del_count).sum();
    assert_eq!(deleted, 1, "the update deleted the first batch's \"a\"");
    assert_eq!(batch.columns()[0].column().name(), "id");
    assert_eq!(batch.columns()[1].column().name(), "v");
}

/// `processBatch` over the column kinds the other tests leave out: a
/// dense binary column (one value a document, in order), a sparse one
/// with a repeated doc id (two values of that document), a token-stream
/// column and a float-vector column -- each cell its document's field.
#[test]
fn binary_token_and_float_vector_columns_become_their_documents_fields() {
    struct DenseBinary {
        c: Column,
        values: Vec<Vec<u8>>,
    }
    impl BinaryColumn for DenseBinary {
        fn column(&self) -> &Column {
            &self.c
        }
        fn stored_type(&self) -> StoredType {
            StoredType::String
        }
        fn tuples(&self) -> Box<dyn ObjectTupleCursor<Vec<u8>> + '_> {
            Box::new(VecObjectTupleCursor::new(&[]))
        }
        fn values(&self) -> Result<Box<dyn BytesRefValuesCursor + '_>> {
            Ok(Box::new(VecBytesValuesCursor::new(&self.values)))
        }
    }
    struct FloatVectors {
        c: Column,
        tuples: Vec<(i32, Vec<f32>)>,
    }
    impl VectorColumn<Vec<f32>> for FloatVectors {
        fn column(&self) -> &Column {
            &self.c
        }
        fn tuples(&self) -> Box<dyn ObjectTupleCursor<Vec<f32>> + '_> {
            Box::new(VecObjectTupleCursor::new(&self.tuples))
        }
    }
    let stored = ft(|t| t.set_stored(true).unwrap());
    let title = DenseBinary {
        c: Column::new("title", stored.clone(), Density::Dense),
        values: vec![b"t0".to_vec(), b"t1".to_vec(), b"t2".to_vec()],
    };
    let tags = B {
        c: Column::new("tag", stored, Density::Sparse),
        stored: StoredType::Binary,
        tuples: vec![(1, b"x".to_vec()), (1, b"y".to_vec()), (2, b"z".to_vec())],
    };
    let text = ft(|t| t.set_index_options(IndexOptions::DocsAndFreqs).unwrap());
    let body = T {
        c: Column::new("body", text, Density::Sparse),
        tuples: vec![(0, FieldTokens::default())],
    };
    let vec_type = ft(|t| {
        t.set_vector_attributes(
            2,
            VectorEncoding::Float32,
            VectorSimilarityFunction::Euclidean,
        )
        .unwrap()
    });
    let vectors = FloatVectors {
        c: Column::new("vec", vec_type, Density::Sparse),
        tuples: vec![(0, vec![1.0, 2.0]), (2, vec![3.0, 4.0])],
    };
    let batch = Batch {
        n: 3,
        cols: vec![
            BatchColumn::Binary(&title),
            BatchColumn::Binary(&tags),
            BatchColumn::TokenStream(&body),
            BatchColumn::FloatVector(&vectors),
        ],
    };
    let docs = batch_documents(&batch).unwrap();
    assert_eq!(docs.len(), 3);
    let stored = |d: usize, name: &str| -> Vec<Option<StoredValue>> {
        docs[d]
            .get_fields_named(name)
            .iter()
            .map(|f| f.stored_value())
            .collect()
    };
    for d in 0..3 {
        assert_eq!(
            stored(d, "title"),
            vec![Some(StoredValue::String(format!("t{d}")))]
        );
    }
    assert!(stored(0, "tag").is_empty());
    assert_eq!(
        stored(1, "tag"),
        vec![
            Some(StoredValue::Binary(b"x".to_vec())),
            Some(StoredValue::Binary(b"y".to_vec()))
        ]
    );
    assert_eq!(docs[0].get_fields_named("body").len(), 1);
    assert!(docs[1].get_fields_named("body").is_empty());
    let vector = |d: usize| docs[d].get_field("vec").and_then(|f| f.vector_value());
    assert_eq!(vector(0), Some(VectorValue::Float32(vec![1.0, 2.0].into())));
    assert_eq!(vector(1), None);
    assert_eq!(vector(2), Some(VectorValue::Float32(vec![3.0, 4.0].into())));

    // A dense column must have a value for every document.
    let short = DenseBinary {
        c: Column::new("title", ft(|t| t.set_stored(true).unwrap()), Density::Dense),
        values: vec![b"only".to_vec()],
    };
    let e = err(&Batch {
        n: 3,
        cols: vec![BatchColumn::Binary(&short)],
    });
    assert!(e.contains("title"), "{e}");
}
