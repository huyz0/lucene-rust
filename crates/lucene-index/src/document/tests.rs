// Test fixtures' own arithmetic, not values read off disk -- see
// `docs/arithmetic-gate.md`'s "Test code" section.
#![allow(clippy::arithmetic_side_effects)]

use super::*;
use crate::index_writer::{IndexWriter, VectorValue};
use crate::segment_info::LuceneVersion;
use crate::similarity::{FieldInvertState, NormSimilarity};
use lucene_store::directory::{Directory, FsDirectory};
use lucene_store::ByteBuffersDirectory;
use lucene_util::test_support::TempDir;

const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};

fn writer<'d>(dir: &'d FsDirectory) -> IndexWriter<'d> {
    IndexWriter::open(dir, Vec::new(), "Lucene104", VERSION).unwrap()
}

fn doc(fields: Vec<Box<dyn IndexableField>>) -> Document {
    let mut d = Document::new();
    for f in fields {
        d.add_boxed(f);
    }
    d
}

#[test]
fn numbers_convert_as_java_does() {
    assert_eq!(Number::Int(-3).long_value(), -3);
    assert_eq!(Number::Long(5).int_value(), 5);
    assert_eq!(Number::Float(2.9).long_value(), 2);
    assert_eq!(Number::Double(-2.9).int_value(), -2);
    assert_eq!(Number::Double(f64::NAN).long_value(), 0);
    assert_eq!(Number::Float(1e30).int_value(), i32::MAX);
    assert_eq!(Number::Int(7).int_value(), 7);
    assert_eq!(Number::Long(1).long_value(), 1);
    assert_eq!(
        [
            Number::Int(1),
            Number::Long(2),
            Number::Float(1.5),
            Number::Double(2.5)
        ]
        .map(|n| n.to_string()),
        ["1", "2", "1.5", "2.5"]
    );
}

#[test]
fn field_types_freeze_and_validate() {
    let mut ft = FieldType::default();
    assert!(ft.tokenized());
    ft.set_stored(true).unwrap();
    ft.set_store_term_vectors(true).unwrap();
    ft.set_store_term_vector_offsets(true).unwrap();
    ft.set_store_term_vector_positions(true).unwrap();
    ft.set_store_term_vector_payloads(true).unwrap();
    ft.set_omit_norms(true).unwrap();
    ft.set_index_options(IndexOptions::Docs).unwrap();
    ft.set_doc_values_type(DocValuesType::Sorted).unwrap();
    ft.set_doc_values_skip_index_type(DocValuesSkipIndexType::Range)
        .unwrap();
    ft.set_dimensions(2, 4).unwrap();
    ft.set_vector_attributes(3, VectorEncoding::Byte, VectorSimilarityFunction::Cosine)
        .unwrap();
    assert_eq!(ft.put_attribute("k", "v").unwrap(), None);
    assert_eq!(ft.put_attribute("k", "w").unwrap(), Some("v".into()));
    assert_eq!(ft.attributes().unwrap()["k"], "w");
    assert!(ft.stored() && ft.store_term_vectors() && ft.store_term_vector_offsets());
    assert!(ft.store_term_vector_positions() && ft.store_term_vector_payloads());
    assert_eq!(ft.point_index_dimension_count(), 2);
    assert_eq!(ft.vector_dimension(), 3);
    assert_eq!(ft.vector_encoding(), VectorEncoding::Byte);
    assert_eq!(
        ft.vector_similarity_function(),
        VectorSimilarityFunction::Cosine
    );
    assert_eq!(
        ft.to_string(),
        "stored,indexed,tokenized,termVector,termVectorOffsets,termVectorPosition,\
         termVectorPayloads,omitNorms,indexOptions=DOCS,pointDimensionCount=2,\
         pointIndexDimensionCount=2,pointNumBytes=4,docValuesType=SORTED"
    );
    let copy = FieldType::copy_of(&ft.clone().frozen());
    assert!(!copy.is_frozen());
    ft.freeze();
    assert!(matches!(ft.set_stored(false), Err(Error::IllegalState(_))));
    assert!(ft.set_dimensions(1, 1).is_err());
    assert!(ft
        .set_vector_attributes(1, VectorEncoding::Byte, VectorSimilarityFunction::Cosine)
        .is_err());
    assert!(ft.put_attribute("a", "b").is_err());

    let mut t = FieldType::new();
    for (d, i, b) in [
        (-1, 0, 0),
        (17, 1, 1),
        (1, -1, 1),
        (2, 3, 1),
        (9, 9, 1),
        (1, 1, -1),
        (1, 1, 17),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 0),
    ] {
        assert!(t.set_dimensions_with_index(d, i, b).is_err(), "{d} {i} {b}");
    }
    t.set_dimensions(0, 0).unwrap();
    assert!(t
        .set_vector_attributes(
            0,
            VectorEncoding::Float32,
            VectorSimilarityFunction::Euclidean
        )
        .is_err());
    t.set_index_options(IndexOptions::DocsAndFreqsAndPositions)
        .unwrap();
    t.set_tokenized(false).unwrap();
    assert_eq!(t.to_string(), "indexed");
    assert_eq!(FieldType::new().to_string(), "");
    for o in [
        IndexOptions::None,
        IndexOptions::Docs,
        IndexOptions::DocsAndFreqs,
        IndexOptions::DocsAndFreqsAndPositions,
        IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
        IndexOptions::DocsAndCustomFreqs,
    ] {
        assert!(!index_options_name(o).is_empty());
    }
    for d in [
        DocValuesType::None,
        DocValuesType::Numeric,
        DocValuesType::Binary,
        DocValuesType::Sorted,
        DocValuesType::SortedSet,
        DocValuesType::SortedNumeric,
    ] {
        assert!(!doc_values_type_name(d).is_empty());
    }
    assert!(index_options_subsumes(
        IndexOptions::DocsAndCustomFreqs,
        IndexOptions::DocsAndFreqs
    ));
    assert!(!index_options_subsumes(
        IndexOptions::DocsAndCustomFreqs,
        IndexOptions::DocsAndFreqsAndPositions
    ));
}

#[test]
fn fields_check_their_constructor_arguments() {
    let text = TextField::type_stored();
    let unindexed = FieldType::new();
    assert!(Field::from_string("f", "v", unindexed.clone()).is_err());
    assert!(Field::from_reader("f", "v", text.clone()).is_err());
    let mut untok = TextField::type_not_stored().clone();
    untok = FieldType::copy_of(&untok);
    untok.set_tokenized(false).unwrap();
    assert!(Field::from_reader("f", "v", untok.clone()).is_err());
    assert!(Field::from_token_stream("f", FieldTokens::default(), untok.clone()).is_err());
    assert!(Field::from_token_stream("f", FieldTokens::default(), text.clone()).is_err());
    assert!(
        Field::from_bytes("f", vec![1], text.clone()).is_err(),
        "tokenized"
    );
    let mut offsets = FieldType::copy_of(&untok);
    offsets
        .set_index_options(IndexOptions::DocsAndFreqsAndPositionsAndOffsets)
        .unwrap();
    assert!(Field::from_bytes("f", vec![1], offsets).is_err());
    assert!(Field::from_bytes("f", vec![1], unindexed).is_err());
    assert!(Field::from_bytes("f", vec![1], untok).is_ok());
}

#[test]
fn field_values_and_setters() {
    let mut f = Field::stored("s", StoredValue::String("abc".into()));
    assert_eq!(f.string_value().as_deref(), Some("abc"));
    f.set_string_value("xyz").unwrap();
    assert_eq!(f.stored_value(), Some(StoredValue::String("xyz".into())));
    assert!(f.set_bytes_value(vec![1]).is_err());
    assert!(f.set_int_value(1).is_err());
    assert!(f.set_long_value(1).is_err());
    assert!(f.set_float_value(1.0).is_err());
    assert!(f.set_double_value(1.0).is_err());
    assert!(f.set_token_stream(FieldTokens::default()).is_err());
    assert!(matches!(f.data(), FieldData::String(_)));
    let mut b = Field::stored("b", StoredValue::Binary(vec![1]));
    b.set_bytes_value(vec![2]).unwrap();
    assert_eq!(b.binary_value().unwrap().as_ref(), &[2]);
    assert!(b.set_string_value("x").is_err());
    let mut i = Field::stored("i", StoredValue::Int(1));
    i.set_int_value(2).unwrap();
    assert_eq!(i.numeric_value(), Some(Number::Int(2)));
    assert_eq!(i.string_value().as_deref(), Some("2"));
    let mut l = Field::stored("l", StoredValue::Long(1));
    l.set_long_value(3).unwrap();
    assert_eq!(l.stored_value(), Some(StoredValue::Long(3)));
    assert_eq!(l.string_value().as_deref(), Some("3"));
    let mut fl = Field::stored("f", StoredValue::Float(1.0));
    fl.set_float_value(0.5).unwrap();
    assert_eq!(fl.stored_value(), Some(StoredValue::Float(0.5)));
    assert_eq!(fl.string_value().as_deref(), Some("0.5"));
    let mut d = Field::stored("d", StoredValue::Double(1.0));
    d.set_double_value(1e10).unwrap();
    assert_eq!(d.numeric_value(), Some(Number::Double(1e10)));
    assert_eq!(d.string_value().as_deref(), Some("1.0E10"));
    let mut t = TextField::from_token_stream("t", FieldTokens::default());
    t.set_token_stream(FieldTokens {
        final_offset: 3,
        ..FieldTokens::default()
    })
    .unwrap();
    assert!(t.set_string_value("x").is_err());
    assert_eq!(t.stored_value(), None, "not stored");
    assert!(t.string_value().is_none());
    let r = TextField::from_reader("r", "some text");
    assert_eq!(r.stored_value(), None);
    // Java's `Float.toString`/`Double.toString` shapes.
    assert_eq!(java_float_string(f32::NAN), "NaN");
    assert_eq!(java_double_string(f64::NEG_INFINITY), "-Infinity");
    assert_eq!(java_double_string(f64::INFINITY), "Infinity");
    assert_eq!(java_double_string(1.25e-5), "1.25E-5");
    assert_eq!(java_float_string(2e7), "2.0E7");
    assert_eq!(java_double_string(0.0), "0.0");
}

#[test]
fn token_streams_follow_field_tokenstream() {
    let analyzer = Analyzer::standard(None);
    let text = TextField::new("t", "Quick Fox", Store::No);
    let ts = text.token_stream(&analyzer).unwrap().unwrap();
    assert_eq!(ts.tokens.len(), 2);
    assert_eq!(ts.tokens[0].term, b"quick");
    let reader = TextField::from_reader("t", "a b");
    assert_eq!(
        reader
            .token_stream(&analyzer)
            .unwrap()
            .unwrap()
            .tokens
            .len(),
        2
    );
    let pre = TextField::from_token_stream(
        "t",
        FieldTokens {
            tokens: vec![FieldToken::new("x", 0, 1)],
            ..FieldTokens::default()
        },
    );
    assert_eq!(
        pre.token_stream(&analyzer).unwrap().unwrap().tokens[0].term,
        b"x"
    );
    // Untokenized: the whole string, or the bytes, as one token.
    let mut untok = FieldType::new();
    untok.set_index_options(IndexOptions::Docs).unwrap();
    untok.set_tokenized(false).unwrap();
    let s = Field::from_string("u", "héllo", untok.clone()).unwrap();
    let ts = s.token_stream(&analyzer).unwrap().unwrap();
    assert_eq!(ts.tokens[0].end_offset, 5, "UTF-16 units");
    assert_eq!(ts.final_offset, 5);
    let b = Field::from_bytes("u", vec![0xff, 0], untok.clone()).unwrap();
    assert_eq!(
        b.token_stream(&analyzer).unwrap().unwrap().tokens[0].term,
        vec![0xff, 0]
    );
    let n = Field::raw("u", untok.clone(), FieldData::Long(12));
    assert_eq!(
        n.token_stream(&analyzer).unwrap().unwrap().tokens[0].term,
        b"12"
    );
    let none = Field::raw("u", untok, FieldData::TokenStream(FieldTokens::default()));
    assert!(none.token_stream(&analyzer).is_err());
    // A tokenized number is analyzed from its decimal form; a tokenized
    // bytes value cannot be.
    let num = Field::raw("n", TextField::type_not_stored(), FieldData::Int(42));
    assert_eq!(
        num.token_stream(&analyzer).unwrap().unwrap().tokens[0].term,
        b"42"
    );
    let bytes = Field::raw("n", TextField::type_not_stored(), FieldData::Bytes(vec![1]));
    assert!(bytes.token_stream(&analyzer).is_err());
    // Not indexed: no stream.
    assert!(Field::stored("s", StoredValue::Int(1))
        .token_stream(&analyzer)
        .unwrap()
        .is_none());
    let stream: FieldTokens = analyzer.analyze_stream("a b").into();
    assert_eq!(stream.tokens[1].position_increment, 1);
}

#[test]
fn string_fields_are_one_binary_term() {
    let analyzer = Analyzer::standard(None);
    let mut s = StringField::new("s", "v", Store::Yes);
    assert_eq!(s.invertable_type(), InvertableType::Binary);
    assert_eq!(s.binary_value().unwrap().as_ref(), b"v");
    assert_eq!(s.string_value().as_deref(), Some("v"));
    s.set_string_value("w").unwrap();
    assert_eq!(s.stored_value(), Some(StoredValue::String("w".into())));
    assert!(s.set_bytes_value(vec![1]).is_err());
    assert_eq!(
        s.token_stream(&analyzer).unwrap().unwrap().tokens[0].term,
        b"w"
    );
    assert_eq!(s.name(), "s");
    assert!(s.field_type().omit_norms());
    let mut b = StringField::from_bytes("s", vec![9], Store::Yes);
    assert_eq!(b.string_value(), None);
    b.set_bytes_value(vec![8]).unwrap();
    assert_eq!(b.stored_value(), Some(StoredValue::Binary(vec![8])));
    assert!(b.set_string_value("x").is_err());
    assert_eq!(
        b.token_stream(&analyzer).unwrap().unwrap().tokens[0].term,
        vec![8]
    );
    assert_eq!(StringField::new("s", "v", Store::No).stored_value(), None);
    assert!(!StringField::type_not_stored().stored());
}

#[test]
fn documents_hold_their_fields_in_order() {
    let mut d = Document::new();
    d.add(StringField::new("a", "1", Store::Yes));
    d.add(Field::stored("b", StoredValue::Binary(vec![1])));
    d.add(StringField::new("a", "2", Store::No));
    d.add(Field::stored("b", StoredValue::Binary(vec![2])));
    assert_eq!(d.get("a").as_deref(), Some("1"));
    assert_eq!(d.get_values("a"), vec!["1".to_string(), "2".to_string()]);
    assert_eq!(d.get_binary_value("b"), Some(vec![1]));
    assert_eq!(d.get_binary_values("b"), vec![vec![1], vec![2]]);
    assert_eq!(d.get_fields_named("a").len(), 2);
    assert_eq!(d.get_field("b").unwrap().name(), "b");
    assert!(d.get_field("zz").is_none());
    assert_eq!(d.get("zz"), None);
    d.remove_field("a");
    assert_eq!(d.get("a").as_deref(), Some("2"));
    d.remove_fields("b");
    assert_eq!(d.fields().len(), 1);
    d.remove_field("zz");
    d.clear();
    assert!(d.fields().is_empty());
}

#[test]
fn a_document_inverts_as_the_indexing_chain_does() {
    let tmp = TempDir::new("document-invert");
    let dir = FsDirectory::open(tmp.path());
    let mut w = writer(&dir);
    w.enable_explicit_documents().unwrap();
    let mut offsets = FieldType::copy_of(&TextField::type_not_stored());
    offsets
        .set_index_options(IndexOptions::DocsAndFreqsAndPositionsAndOffsets)
        .unwrap();
    // Two values of one text field: the second starts after the analyzer's
    // gaps (0 positions, 1 offset).
    let d = doc(vec![
        Box::new(Field::from_string("t", "a b a", offsets.clone()).unwrap()),
        Box::new(Field::from_string("t", "b", offsets).unwrap()),
        Box::new(KeywordField::new("k", "x", Store::Yes)),
        Box::new(IntField::new("i", 3, Store::No)),
        Box::new(FeatureField::new("feat", "pr", 2.0).unwrap()),
    ]);
    let (e, vectors) = w.invert_fields_document(&d).unwrap();
    assert!(vectors.is_empty());
    let t = e
        .fields
        .inverted
        .iter()
        .find(|f| f.field_number == 0)
        .unwrap();
    let a = t.terms.iter().find(|x| x.term == b"a").unwrap();
    assert_eq!((a.freq, a.positions.clone()), (2, vec![0, 2]));
    assert_eq!(a.offsets, vec![(0, 1), (4, 5)]);
    let b = t.terms.iter().find(|x| x.term == b"b").unwrap();
    assert_eq!(b.positions, vec![1, 3]);
    assert_eq!(b.offsets, vec![(2, 3), (6, 7)]);
    assert!(t.norm.is_some());
    let feat = e
        .fields
        .inverted
        .iter()
        .find(|f| f.field_number == 3)
        .unwrap();
    assert_eq!(feat.terms[0].freq, (2.0f32.to_bits() >> 15) as i32);
    assert_eq!(feat.norm, None);
    assert_eq!(e.stored.len(), 1);
    assert_eq!(e.fields.doc_values.len(), 2);
    assert_eq!(e.fields.points.len(), 1);
    assert!(!e.fields.is_empty());
    assert!(crate::index_writer::ExplicitFields::default().is_empty());
    w.add_fields_document(&d).unwrap();
    w.update_fields_documents(crate::buffered_updates::Term::new("k", "x"), &[d])
        .unwrap();
    w.commit().unwrap();
}

#[test]
fn inversion_refuses_what_lucene_refuses() {
    let tmp = TempDir::new("document-refuse");
    let dir = FsDirectory::open(tmp.path());
    let mut w = writer(&dir);
    w.enable_explicit_documents().unwrap();
    let refuse = |w: &mut IndexWriter<'_>, fields: Vec<Box<dyn IndexableField>>| {
        w.invert_fields_document(&doc(fields))
            .unwrap_err()
            .to_string()
    };
    // Two schemas for one name in one document.
    let e = refuse(
        &mut w,
        vec![
            Box::new(IntField::new("x", 1, Store::No)),
            Box::new(LongField::new("x", 1, Store::No)),
        ],
    );
    assert!(e.contains("Inconsistency"), "{e}");
    let e = refuse(
        &mut w,
        vec![
            Box::new(NumericDocValuesField::new("y", 1)),
            Box::new(NumericDocValuesField::indexed_field("y", 1)),
        ],
    );
    assert!(e.contains("skip index"), "{e}");
    let e = refuse(
        &mut w,
        vec![
            Box::new(StringField::new("z", "a", Store::No)),
            Box::new(TextField::new("z", "a", Store::No)),
        ],
    );
    assert!(e.contains("Inconsistency"), "{e}");
    let e = refuse(
        &mut w,
        vec![
            Box::new(
                KnnFloatVectorField::new("v", vec![1.0], VectorSimilarityFunction::Cosine).unwrap(),
            ),
            Box::new(
                KnnFloatVectorField::new("v", vec![1.0, 2.0], VectorSimilarityFunction::Cosine)
                    .unwrap(),
            ),
        ],
    );
    assert!(e.contains("vector dimension"), "{e}");
    // Term vector flags on an unindexed field; a skip index without doc
    // values.
    for flag in 0..4 {
        let mut ft = FieldType::new();
        ft.set_stored(true).unwrap();
        match flag {
            0 => ft.set_store_term_vectors(true).unwrap(),
            1 => ft.set_store_term_vector_positions(true).unwrap(),
            2 => ft.set_store_term_vector_offsets(true).unwrap(),
            _ => ft.set_store_term_vector_payloads(true).unwrap(),
        }
        let e = refuse(
            &mut w,
            vec![Box::new(Field::raw("tv", ft, FieldData::Int(1)))],
        );
        assert!(e.contains("not indexed"), "{e}");
    }
    let mut ft = FieldType::new();
    ft.set_doc_values_skip_index_type(DocValuesSkipIndexType::Range)
        .unwrap();
    ft.set_stored(true).unwrap();
    let e = refuse(
        &mut w,
        vec![Box::new(Field::raw("sk", ft, FieldData::Int(1)))],
    );
    assert!(e.contains("without doc values"), "{e}");
    // One value only for NUMERIC/BINARY/SORTED.
    let e = refuse(
        &mut w,
        vec![
            Box::new(SortedDocValuesField::new("sd", "a")),
            Box::new(SortedDocValuesField::new("sd", "b")),
        ],
    );
    assert!(e.contains("appears more than once"), "{e}");
    // A stored field without a stored value; a doc value without a value; a
    // point without bytes.
    let mut stored = FieldType::new();
    stored.set_stored(true).unwrap();
    let e = refuse(
        &mut w,
        vec![Box::new(Field::raw(
            "st",
            stored,
            FieldData::Reader("x".into()),
        ))],
    );
    assert!(e.contains("null value"), "{e}");
    let mut dv = FieldType::new();
    dv.set_doc_values_type(DocValuesType::Numeric).unwrap();
    let e = refuse(
        &mut w,
        vec![Box::new(Field::raw(
            "nv",
            dv,
            FieldData::String("x".into()),
        ))],
    );
    assert!(e.contains("null value not allowed"), "{e}");
    let mut dv = FieldType::new();
    dv.set_doc_values_type(DocValuesType::Binary).unwrap();
    let e = refuse(
        &mut w,
        vec![Box::new(Field::raw("bv", dv, FieldData::Int(1)))],
    );
    assert!(e.contains("null value not allowed"), "{e}");
    let mut pt = FieldType::new();
    pt.set_dimensions(1, 4).unwrap();
    let e = refuse(
        &mut w,
        vec![Box::new(Field::raw("pt", pt, FieldData::Int(1)))],
    );
    assert!(e.contains("no binary value"), "{e}");
    let mut vt = FieldType::new();
    vt.set_vector_attributes(2, VectorEncoding::Float32, VectorSimilarityFunction::Cosine)
        .unwrap();
    let e = refuse(
        &mut w,
        vec![Box::new(Field::raw("vt", vt, FieldData::Int(1)))],
    );
    assert!(e.contains("no vector value"), "{e}");
    // A BINARY-inverted field must have bytes and be untokenized, without
    // positions.
    #[derive(Debug)]
    struct NoBytes(&'static str, FieldType);
    impl IndexableField for NoBytes {
        fn name(&self) -> &str {
            self.0
        }
        fn field_type(&self) -> &FieldType {
            &self.1
        }
        fn invertable_type(&self) -> InvertableType {
            InvertableType::Binary
        }
        fn token_stream(&self, _analyzer: &Analyzer) -> Result<Option<FieldTokens>> {
            Ok(None)
        }
    }
    let e = refuse(
        &mut w,
        vec![Box::new(NoBytes("nb", StringField::type_not_stored()))],
    );
    assert!(e.contains("null for binaryValue"), "{e}");
    #[derive(Debug)]
    struct Tokenized(FieldType);
    impl IndexableField for Tokenized {
        fn name(&self) -> &str {
            "tk"
        }
        fn field_type(&self) -> &FieldType {
            &self.0
        }
        fn binary_value(&self) -> Option<Cow<'_, [u8]>> {
            Some(Cow::Borrowed(b"x"))
        }
        fn invertable_type(&self) -> InvertableType {
            InvertableType::Binary
        }
        fn token_stream(&self, _analyzer: &Analyzer) -> Result<Option<FieldTokens>> {
            Ok(None)
        }
    }
    let e = refuse(
        &mut w,
        vec![Box::new(Tokenized(TextField::type_not_stored()))],
    );
    assert!(e.contains("must produce a non-null TokenStream"), "{e}");
    let e = refuse(
        &mut w,
        vec![Box::new(NoBytes("nb2", TextField::type_not_stored()))],
    );
    assert!(e.contains("null for binaryValue"), "{e}");
    // An indexed TOKEN_STREAM field with no stream.
    #[derive(Debug)]
    struct NoStream(FieldType);
    impl IndexableField for NoStream {
        fn name(&self) -> &str {
            "ns"
        }
        fn field_type(&self) -> &FieldType {
            &self.0
        }
        fn token_stream(&self, _analyzer: &Analyzer) -> Result<Option<FieldTokens>> {
            Ok(None)
        }
    }
    let e = refuse(
        &mut w,
        vec![Box::new(NoStream(TextField::type_not_stored()))],
    );
    assert!(e.contains("produced no token stream"), "{e}");
}

fn tokens(toks: Vec<(&str, i32, i32, i32, i32)>) -> FieldTokens {
    FieldTokens {
        tokens: toks
            .into_iter()
            .map(|(t, inc, s, e, f)| FieldToken {
                term: t.as_bytes().to_vec(),
                start_offset: s,
                end_offset: e,
                position_increment: inc,
                term_frequency: f,
                payload: None,
            })
            .collect(),
        final_position_increment: 0,
        final_offset: 0,
        end_attributes: None,
    }
}

#[test]
fn token_streams_are_checked_as_invert_token_stream_checks_them() {
    let tmp = TempDir::new("document-tokens");
    let dir = FsDirectory::open(tmp.path());
    let mut w = writer(&dir);
    w.enable_explicit_documents().unwrap();
    let with = |opts: IndexOptions| {
        let mut ft = FieldType::copy_of(&TextField::type_not_stored());
        ft.set_index_options(opts).unwrap();
        ft.frozen()
    };
    let mut run = |name: &str, opts: IndexOptions, t: FieldTokens| {
        let f = Field::from_token_stream(name, t, with(opts)).unwrap();
        w.invert_fields_document(&doc(vec![Box::new(f)]))
    };
    let pos = IndexOptions::DocsAndFreqsAndPositions;
    let e = run("a", pos, tokens(vec![("x", 0, 0, 1, 1)]))
        .unwrap_err()
        .to_string();
    assert!(e.contains("first position increment must be > 0"), "{e}");
    let e = run(
        "a",
        pos,
        tokens(vec![("x", 1, 0, 1, 1), ("y", -1, 0, 1, 1)]),
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("position increment must be >= 0"), "{e}");
    let e = run(
        "a",
        pos,
        tokens(vec![
            ("x", 1, 0, 1, 1),
            ("y", 1, 0, 1, 1),
            ("z", i32::MAX, 0, 1, 1),
        ]),
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("overflowed"), "{e}");
    let e = run("a", pos, tokens(vec![("x", i32::MAX - 100, 0, 1, 1)]))
        .unwrap_err()
        .to_string();
    assert!(e.contains("is too large"), "{e}");
    let e = run("a", pos, tokens(vec![("x", 1, 5, 6, 1), ("y", 1, 1, 2, 1)]))
        .unwrap_err()
        .to_string();
    assert!(e.contains("offsets must not go backwards"), "{e}");
    let e = run("a", pos, tokens(vec![("x", 1, 0, 1, 2)]))
        .unwrap_err()
        .to_string();
    assert!(e.contains("cannot index positions"), "{e}");
    let e = run("a", pos, tokens(vec![("x", 1, 0, 1, 1), ("x", 1, 0, 1, 2)]))
        .unwrap_err()
        .to_string();
    assert!(e.contains("cannot index positions"), "{e}");
    let long = "x".repeat(super::indexing::MAX_TERM_LENGTH + 1);
    let e = run("a", pos, tokens(vec![(&long, 1, 0, 1, 1)]))
        .unwrap_err()
        .to_string();
    assert!(e.contains("immense term"), "{e}");
    // Overlaps are counted; a repeated term in a DOCS field is one posting.
    let (e, _) = run(
        "b",
        IndexOptions::Docs,
        tokens(vec![("x", 1, 0, 1, 1), ("x", 0, 0, 1, 1)]),
    )
    .unwrap();
    assert_eq!(e.fields.inverted[0].terms.len(), 1);
    let e = run(
        "b",
        IndexOptions::Docs,
        tokens(vec![("x", 1, 0, 1, 1), ("x", 1, 0, 1, 5)]),
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("must index term freq"), "{e}");
    // Custom frequencies accumulate, DOCS_AND_CUSTOM_FREQS refuses a repeat.
    let (e, _) = run(
        "c",
        IndexOptions::DocsAndFreqs,
        tokens(vec![("x", 1, 0, 1, 3), ("x", 1, 0, 1, 4)]),
    )
    .unwrap();
    assert_eq!(e.fields.inverted[0].terms[0].freq, 7);
    let e = run(
        "d",
        IndexOptions::DocsAndCustomFreqs,
        tokens(vec![("x", 1, 0, 1, 3), ("x", 1, 0, 1, 4)]),
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("duplicate termdoc"), "{e}");
    let (e, _) = run(
        "d",
        IndexOptions::DocsAndCustomFreqs,
        tokens(vec![("x", 1, 0, 1, 3), ("y", 1, 0, 1, 4)]),
    )
    .unwrap();
    assert_eq!(e.fields.inverted[0].terms.len(), 2);
    let e = run(
        "e",
        IndexOptions::DocsAndFreqs,
        tokens(vec![("x", 1, 0, 1, i32::MAX), ("y", 1, 0, 1, 2)]),
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("too many tokens"), "{e}");
    let e = run(
        "e",
        IndexOptions::DocsAndFreqs,
        tokens(vec![("x", 1, 0, 1, i32::MAX), ("x", 1, 0, 0, i32::MAX)]),
    );
    assert!(e.is_err());
    // An empty token stream: the field is present with a zero norm.
    let (e, _) = run("f", pos, FieldTokens::default()).unwrap();
    assert_eq!(e.fields.inverted[0].norm, Some(0));
}

/// `FieldSchema.raiseNotSame` for each facet of a schema two instances of
/// one field name can disagree on.
#[test]
fn one_field_name_cannot_carry_two_schemas() {
    let tmp = TempDir::new("document-schema-conflicts");
    let dir = FsDirectory::open(tmp.path());
    let mut w = writer(&dir);
    w.enable_explicit_documents().unwrap();
    let float = |sim| -> Box<dyn IndexableField> {
        Box::new(KnnFloatVectorField::new("v", vec![1.0], sim).unwrap())
    };
    let cases: Vec<(&str, Vec<Box<dyn IndexableField>>)> = vec![
        (
            "point dimension",
            vec![
                Box::new(IntPoint::new("p", &[1]).unwrap()),
                Box::new(IntPoint::new("p", &[1, 2]).unwrap()),
            ],
        ),
        (
            "point num bytes",
            vec![
                Box::new(IntPoint::new("p", &[1]).unwrap()),
                Box::new(LongPoint::new("p", &[1]).unwrap()),
            ],
        ),
        (
            "vector encoding",
            vec![
                float(VectorSimilarityFunction::Cosine),
                Box::new(
                    KnnByteVectorField::new("v", vec![1], VectorSimilarityFunction::Cosine)
                        .unwrap(),
                ),
            ],
        ),
        (
            "vector similarity function",
            vec![
                float(VectorSimilarityFunction::Cosine),
                float(VectorSimilarityFunction::Euclidean),
            ],
        ),
        (
            "doc values type",
            vec![
                Box::new(NumericDocValuesField::new("d", 1)),
                Box::new(SortedNumericDocValuesField::new("d", 1)),
            ],
        ),
    ];
    for (what, fields) in cases {
        let e = w
            .invert_fields_document(&doc(fields))
            .expect_err(what)
            .to_string();
        assert!(e.contains(what), "{what}: {e}");
    }
}

/// The schema facets `updateDocFieldSchema` checks that the earlier conflict
/// test does not reach: index options, omit norms, term vectors, skip index.
#[test]
fn one_field_name_cannot_change_its_index_options_or_norms() {
    let tmp = TempDir::new("document-schema-facets");
    let dir = FsDirectory::open(tmp.path());
    let mut w = writer(&dir);
    w.enable_explicit_documents().unwrap();
    let raw = |opts: IndexOptions, omit: bool, tv: bool| -> Box<dyn IndexableField> {
        let mut ft = FieldType::copy_of(&StringField::type_not_stored());
        ft.set_index_options(opts).unwrap();
        ft.set_omit_norms(omit).unwrap();
        ft.set_store_term_vectors(tv).unwrap();
        Box::new(Field::raw(
            "k",
            ft.frozen(),
            FieldData::String("v".to_string()),
        ))
    };
    let cases: Vec<(&str, Vec<Box<dyn IndexableField>>)> = vec![
        (
            "index options",
            vec![
                raw(IndexOptions::Docs, true, false),
                raw(IndexOptions::DocsAndFreqs, true, false),
            ],
        ),
        (
            "omit norms",
            vec![
                raw(IndexOptions::Docs, false, false),
                raw(IndexOptions::Docs, true, false),
            ],
        ),
        (
            "store term vector",
            vec![
                raw(IndexOptions::Docs, true, false),
                raw(IndexOptions::Docs, true, true),
            ],
        ),
        (
            "doc values skip index type",
            vec![
                Box::new(NumericDocValuesField::new("d", 1)),
                Box::new(NumericDocValuesField::indexed_field("d", 1)),
            ],
        ),
    ];
    for (what, fields) in cases {
        let e = w
            .invert_fields_document(&doc(fields))
            .expect_err(what)
            .to_string();
        assert!(e.contains(what), "{what}: {e}");
    }
}

#[test]
fn a_zero_norm_for_a_non_empty_field_is_refused() {
    #[derive(Debug)]
    struct Zero;
    impl NormSimilarity for Zero {
        fn compute_norm(&self, _field: &str, _state: &FieldInvertState) -> i64 {
            0
        }
    }
    let tmp = TempDir::new("document-zero-norm");
    let dir = FsDirectory::open(tmp.path());
    let mut w = writer(&dir);
    w.set_similarity(Some(std::sync::Arc::new(Zero)));
    let d = doc(vec![Box::new(TextField::new("t", "a", Store::No))]);
    assert!(matches!(
        w.add_fields_document(&d),
        Err(crate::index_writer::Error::ZeroNorm(_))
    ));
}

/// `TestIndexWriterExceptions.testPerFieldFinishExceptionDoesNotCorruptStoredFields`
/// (GITHUB#16553: Java left the stored-fields frame open when
/// `PerField.finish` threw). The port buffers a document's stored values
/// only once it is fully inverted, so a refused document leaves no stored
/// frame behind: the segment's stored fields are the neighbours', in order,
/// and the commit checks clean.
#[test]
fn a_zero_norm_refusal_leaves_the_stored_fields_whole() {
    use lucene_codecs::stored_fields::{self, FieldValue};
    #[derive(Debug)]
    struct ZeroForBad;
    impl NormSimilarity for ZeroForBad {
        fn compute_norm(&self, field: &str, state: &FieldInvertState) -> i64 {
            if field == "bad" {
                0
            } else {
                crate::similarity::default_compute_norm(true, state)
            }
        }
    }
    let tmp = TempDir::new("document-zero-norm-stored");
    let dir = FsDirectory::open(tmp.path());
    let mut w = writer(&dir);
    w.set_similarity(Some(std::sync::Arc::new(ZeroForBad)));
    let with_id = |id: &str, bad: bool| {
        let mut d = doc(vec![Box::new(StringField::new("id", id, Store::Yes))]);
        d.add(TextField::new(
            if bad { "bad" } else { "text" },
            "hello world",
            Store::No,
        ));
        d
    };
    w.add_fields_document(&with_id("a", false)).unwrap();
    assert!(matches!(
        w.add_fields_document(&with_id("b", true)),
        Err(crate::index_writer::Error::ZeroNorm(_))
    ));
    w.add_fields_document(&with_id("c", false)).unwrap();
    let infos = w.commit().unwrap().clone();
    w.close().unwrap();
    for r in crate::check_index::check_directory(&dir).unwrap() {
        assert!(r.all_passed(), "{:?}", r.failures());
    }
    let sci = &infos.segments[0];
    let read = |ext: &str| dir.open(&format!("{}.{ext}", sci.segment_name)).unwrap();
    let (fdt, fdx, fdm) = (read("fdt"), read("fdx"), read("fdm"));
    let reader = stored_fields::open(&fdt, &fdx, &fdm, &sci.segment_id, "").unwrap();
    let ids: Vec<FieldValue> = (0..reader.max_doc())
        .map(|d| reader.document(d).unwrap().fields[0].value.clone())
        .collect();
    assert_eq!(
        ids,
        [
            FieldValue::String("a".into()),
            FieldValue::String("c".into())
        ]
    );
}

#[test]
fn over_long_stored_strings_are_refused() {
    // Checked by length, so the test needs no 700 MB string.
    let max = super::indexing::MAX_STORED_STRING_LENGTH;
    assert!(super::indexing::test_check_stored_string("h", max).is_ok());
    let e = super::indexing::test_check_stored_string("h", max + 1)
        .unwrap_err()
        .to_string();
    assert!(e.contains("too large"), "{e}");
}

#[test]
fn vector_fields_index_through_the_explicit_path() {
    let tmp = TempDir::new("document-vectors");
    let dir = FsDirectory::open(tmp.path());
    let mut w = writer(&dir);
    for i in 0..3 {
        let mut d = Document::new();
        d.add(StringField::new("id", i.to_string(), Store::Yes));
        if i != 1 {
            d.add(
                KnnFloatVectorField::new(
                    "v",
                    vec![i as f32, 1.0],
                    VectorSimilarityFunction::Euclidean,
                )
                .unwrap(),
            );
            d.add(
                KnnByteVectorField::new(
                    "b",
                    vec![i as u8, 7],
                    VectorSimilarityFunction::DotProduct,
                )
                .unwrap(),
            );
        }
        w.add_fields_document(&d).unwrap();
    }
    w.commit().unwrap();
    let files: Vec<String> = std::fs::read_dir(tmp.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    assert!(
        files.iter().any(|f| f.ends_with(".vec")),
        "vector files written: {files:?}"
    );
    // The writer refuses a vector of the wrong shape.
    let bad = KnnFloatVectorField::with_type(
        "v",
        vec![1.0, 2.0, 3.0],
        KnnFloatVectorField::new("x", vec![1.0, 2.0], VectorSimilarityFunction::Euclidean)
            .unwrap()
            .field_type()
            .clone(),
    );
    let mut d = Document::new();
    d.add(bad);
    assert!(w.add_fields_document(&d).is_err());
    let wrong = KnnByteVectorField::with_type(
        "v",
        vec![1, 2],
        KnnFloatVectorField::new("x", vec![1.0, 2.0], VectorSimilarityFunction::Euclidean)
            .unwrap()
            .field_type()
            .clone(),
    );
    let mut d = Document::new();
    d.add(wrong);
    assert!(w.add_fields_document(&d).is_err());
    let _ = VectorValue::Byte(vec![]);
}

/// `testIllegalDimChangeTwoWriters`, `testIllegalSimilarityFunctionChange`
/// (a different segment) and `testIllegalDimChangeViaAddIndexesDirectory`:
/// a field's vector shape cannot change once the index has it, and the
/// refusal is `FieldInfo.verifySameVectorOptions`' message.
#[test]
fn a_changed_vector_schema_is_refused_with_javas_message() {
    let vector_doc = |dim: usize, sim| {
        let mut d = Document::new();
        d.add(KnnFloatVectorField::new("f", vec![0.5; dim], sim).unwrap());
        d
    };
    let tmp = TempDir::new("document-vector-schema");
    let dir = FsDirectory::open(tmp.path());
    let mut w = writer(&dir);
    w.add_fields_document(&vector_doc(4, VectorSimilarityFunction::DotProduct))
        .unwrap();
    w.commit().unwrap();
    let e = w
        .add_fields_document(&vector_doc(6, VectorSimilarityFunction::DotProduct))
        .unwrap_err();
    assert!(
        e.to_string().contains(
            "cannot change field \"f\" from vector dimension=4, vector encoding=FLOAT32, vector \
             similarity function=DOT_PRODUCT to inconsistent vector dimension=6, vector \
             encoding=FLOAT32, vector similarity function=DOT_PRODUCT"
        ),
        "{e}"
    );
    let e = w
        .add_fields_document(&vector_doc(4, VectorSimilarityFunction::Euclidean))
        .unwrap_err();
    assert!(
        e.to_string()
            .contains("to inconsistent vector dimension=4, vector encoding=FLOAT32, vector similarity function=EUCLIDEAN"),
        "{e}"
    );
    drop(w);
    // A second writer starts from the index's schema (`getFieldNumberMap`).
    let mut w = writer(&dir);
    let e = w
        .add_fields_document(&vector_doc(2, VectorSimilarityFunction::DotProduct))
        .unwrap_err();
    assert!(
        e.to_string()
            .contains("from vector dimension=4, vector encoding=FLOAT32"),
        "{e}"
    );
    drop(w);
    // The same field arriving through `addIndexes`.
    let tmp2 = TempDir::new("document-vector-schema-2");
    let dir2 = FsDirectory::open(tmp2.path());
    let mut w2 = writer(&dir2);
    w2.add_fields_document(&vector_doc(6, VectorSimilarityFunction::DotProduct))
        .unwrap();
    let src: &dyn Directory = &dir;
    let e = w2.add_indexes_merged(&[src]).unwrap_err();
    assert!(
        e.to_string().contains(
            "cannot change field \"f\" from vector dimension=6, vector encoding=FLOAT32, vector \
             similarity function=DOT_PRODUCT to inconsistent vector dimension=4"
        ),
        "{e}"
    );
    let e = w2.add_indexes(&[src]).unwrap_err();
    assert!(e.to_string().contains("from vector dimension=6"), "{e}");
    w2.commit().unwrap();
    drop(w2);
    // Two sources that disagree on a field the writer lacks: the first
    // registers it (`globalFieldNumberMap.addOrGet`), the second is refused.
    let tmp3 = TempDir::new("document-vector-schema-3");
    let dir3 = FsDirectory::open(tmp3.path());
    let mut w3 = writer(&dir3);
    let e = w3.add_indexes(&[src, &dir2]).unwrap_err();
    assert!(
        e.to_string().contains(
            "cannot change field \"f\" from vector dimension=4, vector encoding=FLOAT32, vector \
             similarity function=DOT_PRODUCT to inconsistent vector dimension=6"
        ),
        "{e}"
    );
    assert!(w3.segment_infos().segments.is_empty(), "nothing was added");
    // Either source alone is fine.
    w3.add_indexes(&[&dir2]).unwrap();
}

/// `testIllegalDimensionTooLarge`: a vector wider than the format's 1024
/// dimensions is refused when its field is first registered, before and
/// after the field exists, and after a flush.
#[test]
fn a_vector_wider_than_the_format_allows_is_refused() {
    let tmp = TempDir::new("document-vector-dims");
    let dir = FsDirectory::open(tmp.path());
    let mut w = writer(&dir);
    let vector_doc = |dim: usize| {
        let mut d = Document::new();
        d.add(
            KnnFloatVectorField::new("f", vec![0.5; dim], VectorSimilarityFunction::DotProduct)
                .unwrap(),
        );
        d
    };
    let too_large = "Field [f] vector's dimensions must be <= [1024]; got 1025";
    let e = w.add_fields_document(&vector_doc(1025)).unwrap_err();
    assert!(e.to_string().contains(too_large), "{e}");
    w.add_fields_document(&vector_doc(2)).unwrap();
    let e = w.add_fields_document(&vector_doc(1025)).unwrap_err();
    assert!(e.to_string().contains(too_large), "{e}");
    w.commit().unwrap();
    let e = w.add_fields_document(&vector_doc(1025)).unwrap_err();
    assert!(e.to_string().contains(too_large), "{e}");
    // The limit itself is allowed, in a field of its own.
    let mut d = Document::new();
    d.add(
        KnnFloatVectorField::new("g", vec![0.5; 1024], VectorSimilarityFunction::DotProduct)
            .unwrap(),
    );
    w.add_fields_document(&d).unwrap();
    w.commit().unwrap();
}

#[test]
fn default_accessors_and_value_type_names() {
    let p = IntPoint::new("p", &[1]).unwrap();
    assert_eq!(p.string_value(), None);
    assert_eq!(p.stored_value(), None);
    assert_eq!(p.vector_value(), None);
    assert_eq!(p.invertable_type(), InvertableType::TokenStream);
    let reader = Field::from_reader("r", "text", TextField::type_not_stored()).unwrap();
    let mut r = reader.clone();
    assert!(r
        .set_string_value("x")
        .unwrap_err()
        .to_string()
        .contains("from Reader"));
    for (data, class) in [
        (FieldData::Int(1), "Integer"),
        (FieldData::Long(1), "Long"),
        (FieldData::Float(1.0), "Float"),
        (FieldData::Double(1.0), "Double"),
        (FieldData::Bytes(vec![]), "BytesRef"),
        (
            FieldData::TokenStream(FieldTokens::default()),
            "TokenStream",
        ),
    ] {
        let mut f = Field::raw("f", FieldType::new(), data);
        let e = f.set_string_value("x").unwrap_err().to_string();
        assert!(e.contains(&format!("from {class} ")), "{e}");
    }
    let mut ok = FieldType::new();
    ok.set_dimensions_with_index(0, 0, 0).unwrap();
}

/// An analyzer whose chain fails refuses the document, as Java's
/// `IOException` out of `invertTokenStream` does; `None` returns to the
/// standard analyzer.
#[test]
fn a_failing_analyzer_refuses_the_document() {
    struct Broken;
    impl lucene_analysis::AnalyzerDefinition for Broken {
        fn create_components(
            &self,
            _field: &str,
        ) -> std::result::Result<
            lucene_analysis::TokenStreamComponents,
            lucene_analysis::AnalysisError,
        > {
            Err(lucene_analysis::AnalysisError::IllegalState(
                "broken".into(),
            ))
        }
    }
    let tmp = TempDir::new("document-broken-analyzer");
    let dir = FsDirectory::open(tmp.path());
    let mut w = writer(&dir);
    w.set_analyzer(Some(std::sync::Arc::new(lucene_analysis::Analyzer::new(
        Broken,
    ))));
    let d = doc(vec![Box::new(TextField::new("t", "a b", Store::No))]);
    let e = w.add_fields_document(&d).unwrap_err().to_string();
    assert!(e.contains("broken"), "{e}");
    w.set_analyzer(None);
    w.add_fields_document(&d).unwrap();
}

/// The state `computeNorm` is handed: the position and offset after the
/// trailing gaps, the largest frequency, `end()`'s attributes -- and for a
/// `DOCS` field a `maxTermFrequency` of 1 only while a term is new to the
/// buffered segment.
#[test]
fn the_invert_state_carries_what_java_hands_compute_norm() {
    #[derive(Debug, Default)]
    struct Record(std::sync::Mutex<Vec<(String, FieldInvertState)>>);
    impl NormSimilarity for Record {
        fn compute_norm(&self, field: &str, state: &FieldInvertState) -> i64 {
            self.0
                .lock()
                .unwrap()
                .push((field.to_string(), state.clone()));
            1
        }
    }
    let tmp = TempDir::new("document-invert-state");
    let dir = FsDirectory::open(tmp.path());
    let mut w = writer(&dir);
    let sim = std::sync::Arc::new(Record::default());
    w.set_similarity(Some(sim.clone()));
    w.set_position_increment_gap(10);
    let mut docs_only = FieldType::new();
    docs_only.set_index_options(IndexOptions::Docs).unwrap();
    for _ in 0..2 {
        let d = doc(vec![
            Box::new(TextField::new("t", "a b a", Store::No)),
            Box::new(TextField::new("t", "c ", Store::No)),
            Box::new(Field::from_string("d", "x x", docs_only.clone()).unwrap()),
        ]);
        w.add_fields_document(&d).unwrap();
    }
    let states = sim.0.lock().unwrap();
    let t = &states[0].1;
    assert_eq!(states[0].0, "t");
    // "a b a" at 0..2, gap 10, "c" at 13, gap 10.
    assert_eq!((t.position, t.max_term_frequency, t.length), (23, 2, 4));
    // Offsets: 5 + 1 + 2 + 1.
    assert_eq!(t.offset, 9);
    let end = t.attribute_source.as_ref().unwrap();
    assert_eq!((end.end_offset(), end.position_increment()), (2, 0));
    assert_eq!(states[1].0, "d");
    assert_eq!(states[1].1.max_term_frequency, 1, "x is new to the segment");
    assert_eq!(states[3].1.max_term_frequency, 0, "x is not any more");
}

/// A text field type storing term vectors with the given axes.
fn vector_type(opts: IndexOptions, positions: bool, offsets: bool, payloads: bool) -> FieldType {
    let mut ft = FieldType::copy_of(&TextField::type_not_stored());
    ft.set_index_options(opts).unwrap();
    ft.set_store_term_vectors(true).unwrap();
    ft.set_store_term_vector_positions(positions).unwrap();
    ft.set_store_term_vector_offsets(offsets).unwrap();
    ft.set_store_term_vector_payloads(payloads).unwrap();
    ft.frozen()
}

fn vector_term(
    term: &str,
    freq: i32,
    positions: Option<Vec<i32>>,
    offsets: Option<Vec<(i32, i32)>>,
    payloads: Option<Vec<&[u8]>>,
) -> lucene_codecs::term_vectors::TermVectorTerm {
    lucene_codecs::term_vectors::TermVectorTerm {
        term: term.as_bytes().to_vec(),
        freq,
        positions,
        start_offsets: offsets.as_ref().map(|o| o.iter().map(|o| o.0).collect()),
        end_offsets: offsets.as_ref().map(|o| o.iter().map(|o| o.1).collect()),
        payloads: payloads.map(|p| p.into_iter().map(<[u8]>::to_vec).collect()),
    }
}

/// `TermVectorsConsumerPerField`: a vector's axes are its field type's, not
/// the postings' index options; instances of a field continue one another's
/// positions and offsets; payloads are kept only once one is non-empty; a
/// field without a term has no vector. A flush writes each document's
/// vectors in field-name order into a segment `CheckIndex` accepts.
#[test]
fn term_vectors_are_recorded_as_term_vectors_consumer_records_them() {
    let dir = ByteBuffersDirectory::new();
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    w.enable_explicit_documents().unwrap();
    w.set_position_increment_gap(10);
    // `DOCS` postings, vectors with every axis.
    let all = vector_type(IndexOptions::Docs, true, true, true);
    let mut first = tokens(vec![("b", 1, 0, 1, 1), ("a", 1, 2, 3, 1)]);
    first.tokens[1].payload = Some(b"p".to_vec());
    first.final_offset = 3;
    let second = tokens(vec![("b", 1, 0, 1, 1)]);
    let only_offsets = vector_type(IndexOptions::DocsAndFreqs, false, true, false);
    let d0 = doc(vec![
        Box::new(Field::from_token_stream("t", first, all.clone()).unwrap()),
        Box::new(Field::from_token_stream("t", second, all.clone()).unwrap()),
        Box::new(Field::from_string("o", "x y x", only_offsets.clone()).unwrap()),
    ]);
    let (e, _) = w.invert_fields_document(&d0).unwrap();
    let t = e
        .fields
        .term_vectors
        .iter()
        .find(|v| v.field_number == 0)
        .unwrap();
    assert!(t.has_positions && t.has_offsets && t.has_payloads);
    // "b a" at 0..1, gap 10, "b" at 12; offsets 3 + 1 on.
    assert_eq!(
        t.terms,
        [
            vector_term("a", 1, Some(vec![1]), Some(vec![(2, 3)]), Some(vec![b"p"])),
            vector_term(
                "b",
                2,
                Some(vec![0, 12]),
                Some(vec![(0, 1), (4, 5)]),
                Some(vec![b"", b""])
            ),
        ]
    );
    let o = e
        .fields
        .term_vectors
        .iter()
        .find(|v| v.field_number == 1)
        .unwrap();
    assert!(!o.has_positions && o.has_offsets && !o.has_payloads);
    assert_eq!(
        o.terms,
        [
            vector_term("x", 2, None, Some(vec![(0, 1), (4, 5)]), None),
            vector_term("y", 1, None, Some(vec![(2, 3)]), None),
        ]
    );
    // No payload seen: none kept. No token: no vector, but still inverted.
    let d1 = doc(vec![
        Box::new(Field::from_string("t", "c", all.clone()).unwrap()),
        Box::new(Field::from_string("o", "", only_offsets.clone()).unwrap()),
    ]);
    let (e1, _) = w.invert_fields_document(&d1).unwrap();
    assert_eq!(e1.fields.term_vectors.len(), 1);
    assert_eq!(
        e1.fields.term_vectors[0].terms,
        [vector_term("c", 1, Some(vec![0]), Some(vec![(0, 1)]), None)]
    );
    assert!(!e1.fields.term_vectors[0].has_payloads);
    assert_eq!(e1.fields.inverted.len(), 2);

    let mut stored = FieldType::new();
    stored.set_stored(true).unwrap();
    let stored_only = doc(vec![Box::new(
        Field::from_string("s", "v", stored.frozen()).unwrap(),
    )]);
    for d in [&d0, &d1, &stored_only] {
        w.add_fields_document(d).unwrap();
    }
    let infos = w.commit().unwrap().clone();
    for r in crate::check_index::check_directory(&dir).unwrap() {
        assert!(r.all_passed(), "{:?}", r.failures());
    }
    let sci = &infos.segments[0];
    let read = |ext: &str| dir.open(&format!("{}.{ext}", sci.segment_name)).unwrap();
    let (tvd, tvx, tvm) = (read("tvd"), read("tvx"), read("tvm"));
    let reader = lucene_codecs::term_vectors::open(&tvd, &tvx, &tvm, &sci.segment_id, "").unwrap();
    assert_eq!(reader.max_doc(), 3);
    let doc0 = reader.document(0).unwrap().unwrap();
    let mut expected = e.fields.term_vectors.clone();
    // Name order: "o" (number 1) before "t" (number 0).
    expected.sort_by_key(|v| -v.field_number);
    assert_eq!(doc0.fields, expected);
    assert_eq!(
        reader.document(1).unwrap().unwrap().fields,
        e1.fields.term_vectors
    );
    assert!(reader
        .document(2)
        .unwrap()
        .is_none_or(|d| d.fields.is_empty()));
}

/// A document's vector fields in `String.compareTo` order: U+1F600 (the
/// surrogate pair 0xD83D 0xDE00) before U+FF61, though its UTF-8 sorts after.
#[test]
fn term_vector_fields_are_written_in_utf16_name_order() {
    let dir = ByteBuffersDirectory::new();
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    let ft = vector_type(IndexOptions::DocsAndFreqs, false, false, false);
    w.add_fields_document(&doc(vec![
        Box::new(Field::from_string("\u{ff61}", "a", ft.clone()).unwrap()),
        Box::new(Field::from_string("\u{1f600}", "b", ft).unwrap()),
    ]))
    .unwrap();
    let infos = w.commit().unwrap().clone();
    let sci = &infos.segments[0];
    let read = |ext: &str| dir.open(&format!("{}.{ext}", sci.segment_name)).unwrap();
    let (tvd, tvx, tvm) = (read("tvd"), read("tvx"), read("tvm"));
    let reader = lucene_codecs::term_vectors::open(&tvd, &tvx, &tvm, &sci.segment_id, "").unwrap();
    let terms: Vec<Vec<u8>> = reader
        .document(0)
        .unwrap()
        .unwrap()
        .fields
        .iter()
        .map(|f| f.terms[0].term.clone())
        .collect();
    assert_eq!(terms, [b"b".to_vec(), b"a".to_vec()]);
}

/// A segment whose documents invert a term-vector field into no term still
/// gets the vector files, as `TermVectorsConsumer` opens its writer then.
#[test]
fn a_vector_field_without_terms_still_writes_the_vector_files() {
    let dir = ByteBuffersDirectory::new();
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    let ft = vector_type(IndexOptions::DocsAndFreqsAndPositions, true, false, false);
    w.add_fields_document(&doc(vec![
        Box::new(Field::from_string("t", "", ft).unwrap()),
        Box::new(StringField::new("id", "0", Store::No)),
    ]))
    .unwrap();
    w.commit().unwrap();
    for r in crate::check_index::check_directory(&dir).unwrap() {
        assert!(r.all_passed(), "{:?}", r.failures());
    }
    let files = dir.list_all().unwrap();
    assert!(files.iter().any(|f| f.ends_with(".tvd")), "{files:?}");
}

/// `TermVectorsConsumerPerField.start` and `getTermFreq`'s refusals.
#[test]
fn term_vector_settings_are_checked_as_java_checks_them() {
    let dir = ByteBuffersDirectory::new();
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    w.enable_explicit_documents().unwrap();
    let refuse = |w: &mut IndexWriter<'_>, fields: Vec<Box<dyn IndexableField>>| {
        w.invert_fields_document(&doc(fields))
            .unwrap_err()
            .to_string()
    };
    let pos = IndexOptions::DocsAndFreqsAndPositions;
    let text = |name: &str, ft: &FieldType| -> Box<dyn IndexableField> {
        Box::new(Field::from_string(name, "a", ft.clone()).unwrap())
    };
    let e = refuse(
        &mut w,
        vec![text("p", &vector_type(pos, false, false, true))],
    );
    assert!(
        e.contains("cannot index term vector payloads without term vector positions"),
        "{e}"
    );
    // Axes without vectors.
    for (i, what) in ["offsets", "positions", "payloads"].iter().enumerate() {
        let mut ft = FieldType::copy_of(&TextField::type_not_stored());
        match i {
            0 => ft.set_store_term_vector_offsets(true).unwrap(),
            1 => ft.set_store_term_vector_positions(true).unwrap(),
            _ => ft.set_store_term_vector_payloads(true).unwrap(),
        }
        let e = refuse(&mut w, vec![text(&format!("n{i}"), &ft)]);
        assert!(
            e.contains(&format!(
                "cannot index term vector {what} when term vectors are not indexed"
            )),
            "{e}"
        );
    }
    // A later instance changing an axis.
    let base = vector_type(pos, true, true, true);
    for (i, (changed, what)) in [
        (
            vector_type(pos, false, true, false),
            "storeTermVectorPositions",
        ),
        (
            vector_type(pos, true, false, true),
            "storeTermVectorOffsets",
        ),
        (
            vector_type(pos, true, true, false),
            "storeTermVectorPayloads",
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let name = format!("c{i}");
        let e = refuse(&mut w, vec![text(&name, &base), text(&name, &changed)]);
        assert!(e.contains(&format!("({what} changed for field")), "{e}");
    }
    // Custom term frequencies with vector positions or offsets.
    let freqs = IndexOptions::DocsAndFreqs;
    for (i, (ft, what)) in [
        (vector_type(freqs, true, false, false), "positions"),
        (vector_type(freqs, false, true, false), "offsets"),
    ]
    .into_iter()
    .enumerate()
    {
        let f =
            Field::from_token_stream(format!("f{i}"), tokens(vec![("a", 1, 0, 1, 2)]), ft).unwrap();
        let e = refuse(&mut w, vec![Box::new(f)]);
        assert!(
            e.contains(&format!(
                "cannot index term vector {what} while using custom TermFrequencyAttribute"
            )),
            "{e}"
        );
    }
    // Without either axis the frequencies add up.
    let plain = vector_type(freqs, false, false, false);
    let f = Field::from_token_stream(
        "g",
        tokens(vec![("a", 1, 0, 1, 2), ("a", 1, 2, 3, 3)]),
        plain,
    )
    .unwrap();
    let (e, _) = w.invert_fields_document(&doc(vec![Box::new(f)])).unwrap();
    assert_eq!(
        e.fields.term_vectors[0].terms,
        [vector_term("a", 5, None, None, None)]
    );
}

/// `testDeleteAllVectorDocs`: a merge whose every vector document was
/// deleted keeps the field, empty, as `KnnVectorsWriter.merge` writes every
/// vector field of the merged `FieldInfos`.
#[test]
fn a_merge_keeps_a_vector_field_whose_every_vector_was_deleted() {
    let tmp = TempDir::new("document-vector-all-deleted");
    let dir = FsDirectory::open(tmp.path());
    let mut w = writer(&dir);
    let mut d0 = Document::new();
    d0.add(StringField::new("id", "0", Store::No));
    d0.add(
        KnnFloatVectorField::new(
            "v",
            vec![2.0, 3.0, 5.0, 6.0],
            VectorSimilarityFunction::DotProduct,
        )
        .unwrap(),
    );
    let mut d1 = Document::new();
    d1.add(StringField::new("id", "1", Store::No));
    w.add_fields_document(&d0).unwrap();
    w.add_fields_document(&d1).unwrap();
    w.commit().unwrap();
    w.delete_documents_by_term(&[crate::buffered_updates::Term::new("id", "0")])
        .unwrap();
    w.commit().unwrap();
    w.force_merge(1).unwrap();
    let infos = w.commit().unwrap().clone();
    assert_eq!(infos.segments.len(), 1);
    let fields = crate::index_writer::segment_field_infos(&dir, &infos.segments[0]).unwrap();
    let v = fields
        .fields
        .iter()
        .find(|f| f.name == "v")
        .expect("field v");
    assert_eq!(v.vector_dimension, 4);
    let results = crate::check_index::check_directory(&dir).unwrap();
    for r in &results {
        assert!(r.all_passed(), "{:?}", r.failures());
    }
    let segment = &results[1].stats;
    assert_eq!((segment.vector_fields, segment.vector_values), (1, 0));
}

/// `TestIndexSorting.testFlushAfterInconsistentSchema` (GITHUB#15125:
/// Java's flush-time sort comparators hit a `PerField` whose `fieldInfo` a
/// refused document had left `null`). The port registers a document's
/// fields only once its whole schema is consistent and buffers nothing of a
/// refused one, so the sort field has no half-made entry to trip over:
/// every flush, sorted or not, keeps exactly the accepted documents.
#[test]
fn a_flush_after_an_inconsistent_schema_keeps_the_accepted_documents() {
    use crate::segment_info::{IndexSortField, IndexSortKind, StringMissingValue};
    use lucene_codecs::stored_fields::{self, FieldValue};
    // Java draws the sort and the extra fields at random; walk the corners.
    for (sorted, extra) in [(false, 0), (true, 0), (false, 40), (true, 40)] {
        let tmp = TempDir::new("document-inconsistent-schema-sort");
        let dir = FsDirectory::open(tmp.path());
        let mut w = writer(&dir);
        w.enable_explicit_documents().unwrap();
        if sorted {
            w.set_index_sort(Some(&[IndexSortField {
                field: "host.name".to_string(),
                reverse: false,
                kind: IndexSortKind::String(StringMissingValue::Last),
            }]))
            .unwrap();
        }
        let add_extra_fields = |d: &mut Document| {
            for i in 0..extra {
                if i % 2 == 0 {
                    d.add(NumericDocValuesField::new(format!("extra-dv-{i}"), i));
                }
                if i % 3 != 0 {
                    d.add(LongField::new(format!("extra-point-{i}"), i, Store::No));
                }
                if i % 5 != 1 {
                    d.add(KeywordField::new(
                        format!("extra-posting-{i}"),
                        i.to_string(),
                        Store::No,
                    ));
                }
            }
        };
        let id = |v: &str| StringField::new("id", v, Store::Yes);
        // first segment
        let mut d = doc(vec![Box::new(id("1"))]);
        d.add(SortedDocValuesField::new("host.name", b"h2".to_vec()));
        d.add(LongField::new("@timestamp", 1, Store::No));
        add_extra_fields(&mut d);
        w.add_fields_document(&d).unwrap();
        w.commit().unwrap();
        // second segment: "@timestamp" as NUMERIC against SORTED_NUMERIC
        let mut d = doc(vec![Box::new(id("2"))]);
        d.add(NumericDocValuesField::new("@timestamp", 2));
        add_extra_fields(&mut d);
        d.add(SortedDocValuesField::new("host.name", b"h1".to_vec()));
        let e = w.add_fields_document(&d).unwrap_err().to_string();
        assert!(e.contains("@timestamp"), "{e}");
        w.commit().unwrap();
        // third segment
        let mut d = doc(vec![Box::new(id("3"))]);
        add_extra_fields(&mut d);
        d.add(LongField::new("@timestamp", 3, Store::No));
        d.add(SortedDocValuesField::new("host.name", b"h1".to_vec()));
        w.add_fields_document(&d).unwrap();
        let infos = w.commit().unwrap().clone();
        w.close().unwrap();
        for r in crate::check_index::check_directory(&dir).unwrap() {
            assert!(r.all_passed(), "{:?}", r.failures());
        }
        let mut ids = Vec::new();
        for sci in &infos.segments {
            assert_eq!(
                sci.del_count, 0,
                "a refused document leaves no deleted slot"
            );
            let read = |ext: &str| dir.open(&format!("{}.{ext}", sci.segment_name)).unwrap();
            let (fdt, fdx, fdm) = (read("fdt"), read("fdx"), read("fdm"));
            let reader = stored_fields::open(&fdt, &fdx, &fdm, &sci.segment_id, "").unwrap();
            for doc in 0..reader.max_doc() {
                ids.push(reader.document(doc).unwrap().fields[0].value.clone());
            }
        }
        assert_eq!(
            ids,
            [
                FieldValue::String("1".into()),
                FieldValue::String("3".into())
            ],
            "sorted: {sorted}, extra fields: {extra}"
        );
    }
}
