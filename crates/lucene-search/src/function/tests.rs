//! Unit tests of the function module: Java's number formatting and parsing,
//! `MutableValue`, the range matchers, the typed bases' derived getters,
//! every source's accessors, sorts and error paths, and the queries' scorers
//! over the `GenFunction` fixture index (whose values and scores
//! `tests/function_fixtures.rs` checks against Lucene).

// Test fixtures' own arithmetic -- see `docs/arithmetic-gate.md`'s "Test code".
#![allow(clippy::arithmetic_side_effects)]

use std::collections::HashMap;
use std::sync::Arc;

use super::docvalues::{self, Bool, BoolDocValues, Double, DoubleDocValues, Int, Long, Str};
use super::valuesource::*;
use super::*;
use crate::directory_reader::DirectoryReader;
use crate::index_searcher::{IndexSearcher, SegmentNorms};
use crate::query::TermQuery;
use crate::query::{BooleanQuery, BoostQuery, Clause, ConstantScoreQuery, DisjunctionMaxQuery};
use crate::values_source as dvs;

fn index() -> DirectoryReader {
    let dir =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/function/index");
    DirectoryReader::open(&lucene_store::FsDirectory::open(dir)).unwrap()
}

fn term(w: &str) -> Clause {
    Clause::Term(TermQuery::new("body", w.as_bytes().to_vec()))
}

fn must(c: Clause) -> BooleanQuery {
    BooleanQuery {
        must: vec![c],
        ..Default::default()
    }
}

#[test]
fn java_number_formatting_and_parsing() {
    assert_eq!(java_float(1.0), "1.0");
    assert_eq!(java_float(1.0e-5), "1.0E-5");
    assert_eq!(java_float(1.5e10), "1.5E10");
    assert_eq!(java_float(f32::NAN), "NaN");
    assert_eq!(java_float(f32::NEG_INFINITY), "-Infinity");
    assert_eq!(java_float(f32::from_bits(29)), "4.1E-44");
    assert_eq!(java_float(-f32::from_bits(21)), "-2.9E-44");
    assert_eq!(java_float(f32::from_bits(0x0001_2345)), "1.04488E-40");
    assert_eq!(java_double(f64::INFINITY), "Infinity");
    assert_eq!(java_double(-2.5e-7), "-2.5E-7");
    assert_eq!(parse_java_float(" 1.5f ").unwrap(), 1.5);
    assert_eq!(parse_java_float("-Infinity").unwrap(), f32::NEG_INFINITY);
    assert!(parse_java_float("+NaN").unwrap().is_nan());
    assert!(parse_java_float("inf").is_err());
    assert!(parse_java_float("").is_err());
    assert!(parse_java_float("1e").is_err());
    assert_eq!(parse_java_double("2D").unwrap(), 2.0);
    assert!(parse_java_double("-NaN").unwrap().is_nan());
    assert_eq!(parse_java_double("Infinity").unwrap(), f64::INFINITY);
    assert!(parse_java_double("x").is_err());
    assert!(parse_java_double("1.2.3").is_err());
    assert_eq!(parse_java_int("-7").unwrap(), -7);
    assert!(parse_java_int("7.0").is_err());
    assert_eq!(parse_java_long("9000000000").unwrap(), 9_000_000_000);
    assert!(parse_java_long("x").is_err());
    assert_eq!(java_float_array(&[1.0, -0.5]), "[1.0, -0.5]");
    assert_eq!(java_byte_array(&[1, 255]), "[1, -1]");
    for (o, s) in [
        (ObjectVal::Null, "null"),
        (ObjectVal::Bool(true), "true"),
        (ObjectVal::Int(3), "3"),
        (ObjectVal::Long(4), "4"),
        (ObjectVal::Float(1.0), "1.0"),
        (ObjectVal::Double(2.0), "2.0"),
        (ObjectVal::Str("x".into()), "x"),
    ] {
        assert_eq!(o.to_string(), s);
    }
}

#[test]
fn mutable_values_follow_java() {
    use std::cmp::Ordering;
    use std::collections::HashSet;
    let f = |value, exists| MutableValue::Float { value, exists };
    assert_eq!(f(1.0, true), f(1.0, true));
    assert_ne!(f(f32::NAN, true), f(f32::NAN, true));
    assert_ne!(f(0.0, true), f(-0.0, true));
    assert_ne!(f(1.0, true), f(1.0, false));
    let d = |value, exists| MutableValue::Double { value, exists };
    assert_eq!(d(1.0, true), d(1.0, true));
    assert_ne!(
        d(1.0, true),
        MutableValue::Int {
            value: 1,
            exists: true
        }
    );
    let all = [
        MutableValue::Bool {
            value: true,
            exists: true,
        },
        MutableValue::Int {
            value: 1,
            exists: true,
        },
        MutableValue::Long {
            value: 2,
            exists: true,
        },
        f(1.5, true),
        d(2.5, true),
        MutableValue::Str {
            value: b"s".to_vec(),
            exists: true,
        },
        MutableValue::Date {
            value: 9,
            exists: true,
        },
    ];
    let set: HashSet<MutableValue> = all.iter().cloned().collect();
    assert_eq!(set.len(), all.len());
    let shown: Vec<String> = all.iter().map(ToString::to_string).collect();
    assert_eq!(shown, ["true", "1", "2", "1.5", "2.5", "s", "9"]);
    for v in &all {
        assert!(v.exists());
        assert_eq!(v.duplicate(), *v);
        assert_eq!(v.compare_same_type(v), Some(Ordering::Equal));
    }
    assert_eq!(f(1.0, false).to_string(), "(null)");
    assert_eq!(f(1.0, false).to_object(), ObjectVal::Null);
    assert_eq!(all[0].compare_same_type(&all[1]), None);
    let b = |value, exists| MutableValue::Bool { value, exists };
    assert_eq!(
        b(true, true).compare_same_type(&b(false, true)),
        Some(Ordering::Greater)
    );
    assert_eq!(
        b(false, true).compare_same_type(&b(true, true)),
        Some(Ordering::Less)
    );
    assert_eq!(
        b(true, false).compare_same_type(&b(true, true)),
        Some(Ordering::Less)
    );
    assert_eq!(
        b(true, true).compare_same_type(&b(true, false)),
        Some(Ordering::Greater)
    );
    assert_eq!(
        f(f32::NAN, true).compare_same_type(&f(1.0, true)),
        Some(Ordering::Greater)
    );
    assert_eq!(
        f(-0.0, true).compare_same_type(&f(0.0, true)),
        Some(Ordering::Less)
    );
    assert_eq!(
        d(-1.0, true).compare_same_type(&d(f64::NAN, true)),
        Some(Ordering::Less)
    );
    let s = |v: &[u8]| MutableValue::Str {
        value: v.to_vec(),
        exists: true,
    };
    assert_eq!(s(b"a").compare_same_type(&s(b"b")), Some(Ordering::Less));
    assert_eq!(
        MutableValue::Date {
            value: 1,
            exists: true
        }
        .compare_same_type(&MutableValue::Date {
            value: 1,
            exists: false
        }),
        Some(Ordering::Greater)
    );
}

#[test]
fn range_matchers_compare_as_each_base_does() {
    struct V(f32);
    impl FunctionValues for V {
        fn float_val(&mut self, _d: i32) -> Result<f32> {
            Ok(self.0)
        }
        fn double_val(&mut self, _d: i32) -> Result<f64> {
            Ok(f64::from(self.0))
        }
        fn int_val(&mut self, _d: i32) -> Result<i32> {
            Ok(self.0 as i32)
        }
        fn long_val(&mut self, _d: i32) -> Result<i64> {
            Ok(self.0 as i64)
        }
        fn ord_val(&mut self, _d: i32) -> Result<i32> {
            Ok(self.0 as i32)
        }
        fn exists(&mut self, _d: i32) -> Result<bool> {
            Ok(self.0 >= 0.0)
        }
        fn to_string_doc(&mut self, _d: i32) -> Result<String> {
            Ok("v".into())
        }
    }
    let m = RangeMatcher::float(Some("1"), Some("3"), false, true).unwrap();
    assert!(!m.matches(&mut V(1.0), 0).unwrap());
    assert!(m.matches(&mut V(3.0), 0).unwrap());
    assert!(!m.matches(&mut V(-1.0), 0).unwrap());
    let m = RangeMatcher::float(None, None, true, false).unwrap();
    assert!(m.matches(&mut V(1e30), 0).unwrap());
    let m = RangeMatcher::double(Some("1"), Some("3"), true, false).unwrap();
    assert!(m.matches(&mut V(1.0), 0).unwrap() && !m.matches(&mut V(3.0), 0).unwrap());
    let m = RangeMatcher::double(None, Some("2"), false, true).unwrap();
    assert!(m.matches(&mut V(2.0), 0).unwrap());
    assert!(RangeMatcher::double(Some("x"), None, true, true).is_err());
    assert!(RangeMatcher::float(None, Some("y"), true, true).is_err());
    assert_eq!(
        RangeMatcher::int(Some(i32::MAX), Some(i32::MIN), false, false),
        RangeMatcher::Int {
            lower: i32::MAX,
            upper: i32::MIN
        }
    );
    assert_eq!(
        RangeMatcher::long(Some(1), Some(5), false, false),
        RangeMatcher::Long { lower: 2, upper: 4 }
    );
    assert_eq!(
        RangeMatcher::long(Some(i64::MAX), Some(i64::MIN), false, false),
        RangeMatcher::Long {
            lower: i64::MAX,
            upper: i64::MIN
        }
    );
    assert!(RangeMatcher::Long { lower: 2, upper: 4 }
        .matches(&mut V(3.0), 0)
        .unwrap());
    assert!(RangeMatcher::Ord { lower: 1, upper: 2 }
        .matches(&mut V(2.0), 0)
        .unwrap());
    // `FunctionValues`' own defaults.
    let mut v = V(2.5);
    assert!(v.bool_val(0).unwrap());
    assert!(v.str_val(0).is_err());
    assert!(v.bytes_val(0, &mut Vec::new()).is_err());
    assert_eq!(v.object_val(0).unwrap(), ObjectVal::Float(2.5));
    assert_eq!(v.cost(), 100.0);
    assert!(v.num_ord().is_err() && v.byte_val(0).is_err() && v.short_val(0).is_err());
    assert!(v.float_vector_val(0).is_err() && v.byte_vector_val(0).is_err());
    let mut mv = v.new_value();
    v.fill_value(0, &mut mv).unwrap();
    assert_eq!(mv.to_string(), "2.5");
    assert!(v.byte_vals(0, &mut []).is_err() && v.short_vals(0, &mut []).is_err());
    assert!(v.float_vals(0, &mut []).is_err() && v.int_vals(0, &mut []).is_err());
    assert!(v.long_vals(0, &mut []).is_err() && v.double_vals(0, &mut []).is_err());
    assert!(v.str_vals(0, &mut []).is_err());
    assert_eq!(v.explain(0).unwrap().value, 2.5);
    assert!(matches!(
        v.range_matcher(Some("1"), None, true, true).unwrap(),
        RangeMatcher::Float { .. }
    ));
}

#[test]
fn typed_bases_derive_javas_getters() {
    struct D(f64);
    impl DoubleDocValues for D {
        fn description(&self) -> String {
            "d".into()
        }
        fn double_val(&mut self, _d: i32) -> Result<f64> {
            Ok(self.0)
        }
    }
    let mut d = Double(D(300.7));
    assert_eq!(d.byte_val(0).unwrap(), 44);
    assert_eq!(d.short_val(0).unwrap(), 300);
    assert!(d.bool_val(0).unwrap());
    assert_eq!(d.object_val(0).unwrap(), ObjectVal::Double(300.7));
    let mut b = Vec::new();
    assert!(d.bytes_val(0, &mut b).unwrap());
    assert_eq!(b, b"300.7");
    assert!(matches!(
        d.range_matcher(Some("1"), Some("2"), true, true).unwrap(),
        RangeMatcher::Double { .. }
    ));
    let mut mv = d.new_value();
    d.fill_value(0, &mut mv).unwrap();
    assert_eq!(mv.to_string(), "300.7");
    assert_eq!(d.cost(), 100.0);

    struct B(bool);
    impl BoolDocValues for B {
        fn description(&self) -> String {
            "b".into()
        }
        fn bool_val(&mut self, _d: i32) -> Result<bool> {
            Ok(self.0)
        }
        fn exists(&mut self, _d: i32) -> Result<bool> {
            Ok(false)
        }
    }
    let mut b = Bool(B(true));
    assert_eq!((b.byte_val(0).unwrap(), b.short_val(0).unwrap()), (1, 1));
    assert_eq!((b.int_val(0).unwrap(), b.long_val(0).unwrap()), (1, 1));
    assert_eq!(
        (b.float_val(0).unwrap(), b.double_val(0).unwrap()),
        (1.0, 1.0)
    );
    assert_eq!(b.object_val(0).unwrap(), ObjectVal::Null);
    assert_eq!(b.to_string_doc(0).unwrap(), "b=true");
    let mut buf = Vec::new();
    assert!(b.bytes_val(0, &mut buf).unwrap());
    let mut mv = b.new_value();
    b.fill_value(0, &mut mv).unwrap();
    assert_eq!(mv.to_string(), "(null)");
    assert!(b
        .range_matcher(Some("0"), None, true, true)
        .is_ok_and(|m| matches!(m, RangeMatcher::Float { .. })));
    assert_eq!(b.cost(), 100.0);
    let mut f = Bool(B(false));
    assert_eq!(
        (f.float_val(0).unwrap(), f.double_val(0).unwrap()),
        (0.0, 0.0)
    );

    struct S;
    impl docvalues::StrDocValues for S {
        fn description(&self) -> String {
            "s".into()
        }
        fn str_val(&mut self, _d: i32) -> Result<Option<String>> {
            Ok(Some("x".into()))
        }
    }
    let mut s = Str(S);
    assert!(s.byte_val(0).is_err() && s.short_val(0).is_err() && s.float_val(0).is_err());
    assert!(s.int_val(0).is_err() && s.long_val(0).is_err() && s.double_val(0).is_err());
    assert!(s.bool_val(0).unwrap());
    assert_eq!(s.object_val(0).unwrap(), ObjectVal::Str("x".into()));
    assert_eq!(s.to_string_doc(0).unwrap(), "s='x'");
    let mut mv = s.new_value();
    s.fill_value(0, &mut mv).unwrap();
    assert_eq!(mv.to_string(), "x");
    assert!(s.range_matcher(None, None, true, true).is_ok());
    assert_eq!(s.cost(), 100.0);

    struct L;
    impl docvalues::LongDocValues for L {
        fn description(&self) -> String {
            "l".into()
        }
        fn long_val(&mut self, _d: i32) -> Result<i64> {
            Ok(-129)
        }
    }
    let mut l = Long(L);
    assert_eq!(
        (l.byte_val(0).unwrap(), l.short_val(0).unwrap()),
        (127, -129)
    );
    assert!(l.bool_val(0).unwrap());
    let mut buf = Vec::new();
    assert!(l.bytes_val(0, &mut buf).unwrap());
    assert!(l.range_matcher(Some("x"), None, true, true).is_err());
    assert_eq!(l.cost(), 100.0);

    struct I;
    impl docvalues::IntDocValues for I {
        fn description(&self) -> String {
            "i".into()
        }
        fn int_val(&mut self, _d: i32) -> Result<i32> {
            Ok(300)
        }
    }
    let mut i = Int(I);
    assert_eq!((i.byte_val(0).unwrap(), i.short_val(0).unwrap()), (44, 300));
    let mut buf = Vec::new();
    assert!(i.bytes_val(0, &mut buf).unwrap());
    assert!(i.range_matcher(None, Some("x"), true, true).is_err());
    assert_eq!(i.cost(), 100.0);

    struct F(f32);
    impl docvalues::FloatDocValues for F {
        fn description(&self) -> String {
            "f".into()
        }
        fn float_val(&mut self, _d: i32) -> Result<f32> {
            Ok(self.0)
        }
    }
    let mut f = docvalues::Float(F(-1.5));
    assert_eq!((f.byte_val(0).unwrap(), f.short_val(0).unwrap()), (-1, -1));
    let mut buf = Vec::new();
    assert!(f.bytes_val(0, &mut buf).unwrap());
    assert_eq!(f.cost(), 100.0);
    assert_eq!(docvalues::i2b(300), 44);
    assert_eq!(docvalues::opt_str(None), "null");
}

/// Every source's description, accessors and sort, over the fixture.
#[test]
fn sources_describe_sort_and_refuse_as_java() {
    let reader = index();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let norms: Vec<SegmentNorms<'_, '_>> = segments.iter().map(|_| None).collect();
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    let top = TopLevel::of_searcher(&searcher, None);
    assert_eq!(top.max_doc().unwrap(), 80);
    assert!(top.num_docs().unwrap() < 80);

    let i: Arc<dyn ValueSource> = Arc::new(IntFieldSource::new("i"));
    let srcs: Vec<Arc<dyn ValueSource>> = vec![
        Arc::new(MultiValuedIntFieldSource::with_missing(
            "mi",
            NumericSelector::Max,
            Some(3),
        )),
        Arc::new(MultiValuedLongFieldSource::with_missing(
            "ml",
            NumericSelector::Min,
            Some(4),
        )),
        Arc::new(MultiValuedFloatFieldSource::with_missing(
            "mf",
            NumericSelector::Max,
            Some(1.5),
        )),
        Arc::new(MultiValuedDoubleFieldSource::with_missing(
            "md",
            NumericSelector::Max,
            Some(2.5),
        )),
        Arc::new(LongFieldSource::new("l")),
        Arc::clone(&i),
        Arc::new(SortedSetFieldSource::new("ss")),
        Arc::new(SortedSetFieldSource::with_selector("ss", SetSelector::Max)),
        Arc::new(SortedSetFieldSource::with_selector(
            "ss",
            SetSelector::MiddleMin,
        )),
        Arc::new(SortedSetFieldSource::with_selector(
            "ss",
            SetSelector::MiddleMax,
        )),
    ];
    for s in &srcs {
        let f = s.native_sort_field(true).unwrap();
        assert!(f.reverse);
    }
    assert!(FloatFieldSource::new("f")
        .native_sort_field(false)
        .is_none());
    assert!(DoubleFieldSource::new("d")
        .native_sort_field(false)
        .is_none());
    for (s, name) in [
        (&IntFieldSource::new("a") as &dyn FieldCacheSource, "a"),
        (&LongFieldSource::new("b"), "b"),
        (&FloatFieldSource::new("c"), "c"),
        (&DoubleFieldSource::new("d"), "d"),
        (
            &MultiValuedIntFieldSource::new("e", NumericSelector::Min),
            "e",
        ),
        (
            &MultiValuedLongFieldSource::new("f", NumericSelector::Min),
            "f",
        ),
        (
            &MultiValuedFloatFieldSource::new("g", NumericSelector::Min),
            "g",
        ),
        (
            &MultiValuedDoubleFieldSource::new("h", NumericSelector::Min),
            "h",
        ),
        (
            &EnumFieldSource::new("i", HashMap::new(), HashMap::new()),
            "i",
        ),
        (&BytesRefFieldSource::new("j"), "j"),
        (&SortedSetFieldSource::new("k"), "k"),
        (&JoinDocFreqValueSource::new("l", "q"), "l"),
    ] {
        assert_eq!(s.field(), name);
    }
    // Constants' accessors.
    let c = ConstValueSource::new(2.5);
    assert_eq!(
        (
            c.get_int(),
            c.get_long(),
            c.get_float(),
            c.get_double(),
            c.get_bool()
        ),
        (2, 2, 2.5, 2.5, true)
    );
    assert_eq!(c.get_number(), ObjectVal::Float(2.5));
    let d = DoubleConstValueSource::new(-3.75);
    assert_eq!(
        (
            d.get_int(),
            d.get_long(),
            d.get_float(),
            d.get_double(),
            d.get_bool()
        ),
        (-3, -3, -3.75, -3.75, true)
    );
    assert_eq!(d.get_number(), ObjectVal::Double(-3.75));
    assert_eq!(LiteralValueSource::new("x").value(), "x");
    assert!(ConstKnnFloatValueSource::new(vec![f32::NAN]).is_err());
    let q = QueryValueSource::new(term("red"), 1.5);
    assert_eq!((q.query(), q.default_value()), (&term("red"), 1.5));
    let v = VectorValueSource::new(vec![Arc::clone(&i), Arc::clone(&i)]);
    assert_eq!(
        (v.dimension(), v.get_sources().len(), v.name()),
        (2, 2, "vector")
    );
    for (n, want) in [
        (
            DocFreqValueSource::new("f", "v", "f", "v").name(),
            "docfreq",
        ),
        (IDFValueSource::new("f", "v", "f", "v").name(), "idf"),
        (
            TermFreqValueSource::new("f", "v", "f", "v").name(),
            "termfreq",
        ),
        (TFValueSource::new("f", "v", "f", "v").name(), "tf"),
        (
            TotalTermFreqValueSource::new("f", "v", "f", "v").name(),
            "totaltermfreq",
        ),
        (
            SumTotalTermFreqValueSource::new("f").name(),
            "sumtotaltermfreq",
        ),
        (NumDocsValueSource::new().name(), "numdocs"),
        (MaxDocValueSource::new().name(), "maxdoc"),
        (NormValueSource::new("f").name(), "norm"),
    ] {
        assert_eq!(n, want);
    }
    let sum = SumFloatFunction::new(vec![Arc::clone(&i)]);
    assert_eq!(format!("{:?}", &sum as &dyn ValueSource), "sum(int(i))");
    let simple =
        SimpleFloatFunction::new("neg", Arc::clone(&i), Arc::new(|d, v| Ok(-v.float_val(d)?)));
    assert_eq!(
        (SingleFunction::name(&simple), simple.source().description()),
        ("neg", "int(i)".to_string())
    );
    let def = DefFunction::new(vec![Arc::clone(&i)]);
    assert_eq!(
        (MultiFunction::name(&def), def.function_sources().len()),
        ("def", 1)
    );
    assert!(same_source(
        &sum,
        &SumFloatFunction::new(vec![Arc::clone(&i)])
    ));
    assert!(!same_source(&sum, &def));

    // Unweighted sources refuse, as Java's context lookups do.
    let empty = FunctionContext::new();
    assert_eq!(format!("{empty:?}"), "FunctionContext(0 entries)");
    let leaf = ValueLeaf::of_searcher(&searcher, 0, None).unwrap();
    // def() of nothing: Java's index -1, an error rather than a panic.
    let mut none = DefFunction::new(Vec::new())
        .get_values(&empty, &leaf)
        .unwrap();
    assert!(none.float_val(0).is_err() && !none.exists(0).unwrap());
    assert!(ValueLeaf::of_searcher(&searcher, 9, None).is_err());
    let unweighted: Vec<Arc<dyn ValueSource>> = vec![
        Arc::new(DocFreqValueSource::new("body", "red", "body", "red")),
        Arc::new(TotalTermFreqValueSource::new("body", "red", "body", "red")),
        Arc::new(SumTotalTermFreqValueSource::new("body")),
        Arc::new(NumDocsValueSource),
        Arc::new(MaxDocValueSource),
        Arc::new(ScaleFloatFunction::new(Arc::clone(&i), 0.0, 1.0)),
        Arc::new(JoinDocFreqValueSource::new("k", "body")),
    ];
    for s in &unweighted {
        assert!(matches!(
            s.get_values(&empty, &leaf),
            Err(Error::IllegalState(_))
        ));
    }
    // With the classic similarity, `idf` needs `createWeight`'s state too.
    let classic = crate::similarities::ClassicSimilarity::default();
    let mut cs = IndexSearcher::new(&segments, &norms).unwrap();
    cs.set_similarity(&classic);
    let cleaf = ValueLeaf::of_searcher(&cs, 0, None).unwrap();
    assert!(IDFValueSource::new("body", "red", "body", "red")
        .get_values(&empty, &cleaf)
        .is_err());

    // Out-of-order reads.
    let fcx = FunctionContext::create(i.as_ref(), &top).unwrap();
    for s in [
        Arc::clone(&i),
        Arc::new(BytesRefFieldSource::new("s")) as Arc<dyn ValueSource>,
        Arc::new(BytesRefFieldSource::new("b")),
        Arc::new(FloatKnnVectorFieldSource::new("fv")),
        Arc::new(QueryValueSource::new(term("red"), 0.0)),
    ] {
        let mut v = s.get_values(&fcx, &leaf).unwrap();
        v.exists(5).unwrap();
        assert!(
            matches!(v.exists(2), Err(Error::IllegalArgument(_))),
            "{}",
            s.description()
        );
    }
    let mut v = NormValueSource::new("body")
        .get_values(&fcx, &cleaf)
        .unwrap();
    v.float_val(5).unwrap();
    assert!(v.float_val(2).is_err());
    // `termfreq`, sent backwards, starts over.
    let mut tf = TermFreqValueSource::new("body", "red", "body", "red")
        .get_values(&fcx, &leaf)
        .unwrap();
    let late = tf.int_val(20).unwrap();
    let early = tf.int_val(0).unwrap();
    assert_eq!(
        (tf.int_val(20).unwrap(), tf.int_val(0).unwrap()),
        (late, early)
    );
    // A joindf read backwards.
    let j = JoinDocFreqValueSource::new("k", "body");
    let jfcx = FunctionContext::create(&j, &top).unwrap();
    let mut jv = j.get_values(&jfcx, &leaf).unwrap();
    jv.int_val(4).unwrap();
    assert!(jv.int_val(1).is_err());
    // The `numOrd` of a terms index.
    let mut sv = BytesRefFieldSource::new("s")
        .get_values(&fcx, &leaf)
        .unwrap();
    assert!(sv.num_ord().unwrap() > 0);
    let mut fill = sv.new_value();
    sv.fill_value(0, &mut fill).unwrap();
    // A field of another doc-values type.
    assert!(SortedSetFieldSource::new("i")
        .get_values(&fcx, &leaf)
        .is_err());
    assert!(BytesRefFieldSource::new("i")
        .get_values(&fcx, &leaf)
        .is_err());
    // A leaf of a reader without a reader.
    let mut bare = leaf.clone();
    bare.ctx.reader = None;
    assert!(bare.reader().is_err());
    // Vector getters of the other kind.
    let mut fv = FloatKnnVectorFieldSource::new("fv")
        .get_values(&fcx, &leaf)
        .unwrap();
    assert!(fv.byte_vector_val(0).is_err());
    let mut bv = ByteKnnVectorFieldSource::new("bv")
        .get_values(&fcx, &leaf)
        .unwrap();
    assert!(bv.float_vector_val(0).is_err());
    let mut ev = ByteKnnVectorFieldSource::new("nosuch")
        .get_values(&fcx, &leaf)
        .unwrap();
    assert!(ev.float_vector_val(0).is_err() && ev.byte_vector_val(0).unwrap().is_none());
    assert!(ByteKnnVectorFieldSource::new("fv")
        .get_values(&fcx, &leaf)
        .is_err());
}

/// The queries' scorers, explanations and reader-wide preparation, over the
/// fixture.
#[test]
fn queries_score_explain_and_prepare() {
    let reader = index();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let owned = reader.field_norms_by_field(&["body".to_string()]);
    let norms: Vec<SegmentNorms<'_, '_>> = owned.iter().map(Some).collect();
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    let i: Arc<dyn ValueSource> = Arc::new(IntFieldSource::new("i"));
    let fq = Clause::from(FunctionQuery::new(Arc::clone(&i)));
    let fr = Clause::from(FunctionRangeQuery::new(
        Arc::clone(&i),
        Some("1"),
        None,
        true,
        true,
    ));
    let fm = Clause::from(FunctionMatchQuery::with_match_cost(
        dvs::from_int_field("i"),
        Arc::new(|v| v > 3.0),
        7.0,
    ));
    let fs = Clause::from(FunctionScoreQuery::boost_by_value(
        term("red"),
        as_double_values_source(Arc::clone(&i)),
    ));
    // Every position a function query can hold, for the statistics pass.
    let q = BooleanQuery {
        must: vec![Clause::Boost(Box::new(BoostQuery::new(fq.clone(), 2.0)))],
        should: vec![Clause::DisjunctionMax(Box::new(DisjunctionMaxQuery::new(
            vec![fs.clone(), term("blue")],
            0.1,
        )))],
        filter: vec![Clause::ConstantScore(Box::new(ConstantScoreQuery::new(
            fr.clone(),
            1.0,
        )))],
        must_not: vec![fm.clone()],
        ..Default::default()
    };
    let mut found = Vec::new();
    collect_functions(&q, &mut found);
    assert_eq!(found.len(), 4);
    let global = crate::multi_segment::global_boolean_stats(&segments, &q).unwrap();
    assert!(!global.functions().is_empty() && !global.is_empty());
    let shown = format!("{:?}", global.functions());
    assert_eq!(shown, "FunctionStats(1 contexts, 2 values sources)");
    assert_eq!(global.functions(), &global.functions().clone());
    assert_ne!(global.functions(), &FunctionStats::default());
    let none = crate::multi_segment::global_boolean_stats(&segments, &must(term("red"))).unwrap();
    assert!(none.functions().is_empty());
    let hits = searcher.search(&q, 100).unwrap();
    assert!(!hits.score_docs.is_empty());

    // Queries' equality and printing.
    assert_eq!(fq, fq.clone());
    assert_eq!(fr, fr.clone());
    assert_eq!(fm, fm.clone());
    assert_eq!(fs, fs.clone());
    let crate::query::Clause::Extended(e) = &fm else {
        panic!()
    };
    assert_eq!(e.name(), "FunctionMatchQuery");
    let fsq = FunctionScoreQuery::new(term("red"), dvs::constant(1.0));
    assert_eq!(
        (fsq.wrapped_query(), fsq.source().describe()),
        (&term("red"), "constant(1.0)".to_string())
    );
    assert_eq!(
        FunctionQuery::new(Arc::clone(&i))
            .value_source()
            .description(),
        "int(i)"
    );

    // Searched one segment at a time: without a statistics pass the function
    // query is refused (one segment is not the index); a caller whose index
    // is that one segment prepares over it.
    for seg in &segments {
        for q in [must(fs.clone()), must(fq.clone())] {
            let mut c = crate::collector::TopDocsCollector::new(100);
            let e = crate::search_boolean_query_scored_segment(seg, &q, None, None, &mut c);
            assert!(matches!(e, Err(crate::Error::IllegalState(_))), "{e:?}");
            let one =
                crate::multi_segment::global_function_stats(std::slice::from_ref(seg), &q, None)
                    .unwrap();
            assert!(one.is_some());
            let mut c = crate::collector::TopDocsCollector::new(100);
            crate::search_boolean_query_scored_segment(seg, &q, None, one.as_ref(), &mut c)
                .unwrap();
        }
    }

    // Cacheability: never for a function or range query; a match or score
    // query's follows its source.
    let r = segments[0].reader;
    assert!(!crate::segment_cacheable::is_cacheable(&fq, r));
    assert!(!crate::segment_cacheable::is_cacheable(&fr, r));
    assert!(crate::segment_cacheable::is_cacheable(&fm, r));
    assert!(!crate::segment_cacheable::is_cacheable(&fs, r));
    assert!(!crate::segment_cacheable::is_cacheable(&fm, None));
    for (c, name) in [
        (&fq, "FunctionQuery"),
        (&fr, "FunctionRangeQuery"),
        (&fm, "FunctionMatchQuery"),
        (&fs, "FunctionScoreQuery"),
    ] {
        let Clause::Extended(e) = c else {
            panic!("{name} is an extended clause")
        };
        assert_eq!(e.name(), name);
    }

    // Explanations of boosted and filtered function queries.
    for c in [&fq, &fr, &fm, &fs] {
        let boosted = must(Clause::Boost(Box::new(BoostQuery::new(c.clone(), 1.5))));
        for doc in [0, 3, 40] {
            let e = searcher.explain(&boosted, doc).unwrap();
            assert!(!e.description.is_empty());
        }
    }
    let filtered = BooleanQuery {
        must: vec![term("red")],
        filter: vec![fs.clone()],
        ..Default::default()
    };
    assert!(searcher.explain(&filtered, 0).is_ok());
    // A boosted non-function query keeps the general explanation.
    let other = must(Clause::Boost(Box::new(BoostQuery::new(term("red"), 2.0))));
    assert!(searcher.explain(&other, 0).is_ok());

    // `ValueSourceScorer`'s match-everything form.
    let fcx = FunctionContext::new();
    let leaf = ValueLeaf::of_searcher(&searcher, 0, None).unwrap();
    let vals = ConstValueSource::new(f32::NEG_INFINITY)
        .get_values(&fcx, &leaf)
        .unwrap();
    let mut s = ValueSourceScorer::all(vals, 3);
    assert!(s.matches_doc(1).unwrap());
    assert_eq!(s.match_cost_of(), 0.0);
    assert_eq!(s.score_doc(1).unwrap(), -f32::MAX);

    // The wrapped sources' explanations and long values.
    let w = as_double_values_source(Arc::new(ConstValueSource::new(2.0)));
    let vctx = crate::values_source::ValuesContext::new(&searcher);
    assert!(!w.is_cacheable(&vctx, 0) && w.needs_scores());
    let e = w
        .explain(&vctx, 0, 0, &crate::explain::Explanation::match_(1.0, "s"))
        .unwrap();
    assert_eq!(e.value, 2.0);
    let l = as_long_values_source(Arc::new(ConstValueSource::new(2.0)));
    assert!(!l.needs_scores() && !l.is_cacheable(&vctx, 0));
    assert_eq!(l.describe(), "const(2.0)");
    let mut lv = l.get_values(&vctx, 0, None).unwrap();
    assert!(lv.advance_exact(0).unwrap());
    assert_eq!(lv.long_value().unwrap(), 2);
    let reader_only = crate::values_source::ValuesContext::for_reader(r.unwrap());
    assert!(w.get_values(&reader_only, 0, None).is_err());
    // A wrapped source unwraps; any other is wrapped.
    let back = from_double_values_source(Arc::clone(&w));
    assert_eq!(back.description(), "const(2.0)");
    let fd = from_double_values_source(dvs::from_int_field("i"));
    let fdcx =
        FunctionContext::create(fd.as_ref(), &TopLevel::of_searcher(&searcher, None)).unwrap();
    let mut fv = fd.get_values(&fdcx, &leaf).unwrap();
    assert_eq!(fv.to_string_doc(0).unwrap(), fd.description());
}

/// The bridges, sorts, reader functions and scorers' remaining paths.
#[test]
fn bridges_sorts_and_scorer_edges() {
    use crate::values_source::{DoubleValuesSource, ValuesContext};
    let reader = index();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let owned = reader.field_norms_by_field(&["body".to_string()]);
    let norms: Vec<SegmentNorms<'_, '_>> = owned.iter().map(Some).collect();
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    let vctx = ValuesContext::new(&searcher);
    let top = TopLevel::of_searcher(&searcher, None);
    let i: Arc<dyn ValueSource> = Arc::new(IntFieldSource::new("i"));
    assert_eq!(
        format!("{:?}", FunctionQuery::new(Arc::clone(&i))),
        "int(i)"
    );

    // A value source reading the scores it is wrapped with: the score.
    let from_scores = from_double_values_source(dvs::scores());
    let scored =
        must(FunctionScoreQuery::new(term("red"), as_double_values_source(from_scores)).into());
    let plain = searcher.search(&must(term("red")), 100).unwrap();
    let via = searcher.search(&scored, 100).unwrap();
    assert_eq!(plain.score_docs, via.score_docs);

    // `boostByQuery`'s sources, and their explanations of a non-match.
    let bq = FunctionScoreQuery::boost_by_query(term("red"), term("blue"), 2.0);
    assert!(bq.source.needs_searcher() && bq.source.queries().len() == 1);
    assert!(!bq.source.is_cacheable(&vctx, 0));
    let miss = crate::explain::Explanation::no_match("x");
    assert!(!bq.source.explain(&vctx, 0, 0, &miss).unwrap().matched);
    let rewritten = bq.source.rewrite(&top).unwrap();
    assert!(rewritten.is_none());
    let bv = FunctionScoreQuery::boost_by_value(
        term("red"),
        crate::function::index_reader_functions::max_doc(),
    );
    assert!(bv.source.rewrite(&top).unwrap().is_some());

    // The index-reader functions refuse before their rewrite.
    let irfs: Vec<Arc<dyn DoubleValuesSource>> = vec![
        index_reader_functions::doc_freq("body", b"red"),
        index_reader_functions::max_doc(),
        index_reader_functions::num_docs(),
        index_reader_functions::num_deleted_docs(),
        index_reader_functions::total_term_freq("body", b"red"),
        index_reader_functions::sum_doc_freq("body"),
        index_reader_functions::doc_count("body"),
    ];
    for s in &irfs {
        assert!(matches!(
            s.get_values(&vctx, 0, None),
            Err(Error::Unsupported(_))
        ));
        assert!(!s.needs_scores() && !s.is_cacheable(&vctx, 0));
        let r = s.rewrite(&top).unwrap().unwrap();
        assert!(!r.needs_scores() && !r.is_cacheable(&vctx, 0));
        assert_eq!(r.describe(), s.describe());
    }
    let sttf = index_reader_functions::sum_total_term_freq("body");
    assert!(sttf.get_values(&vctx, 0, None).is_err());
    assert!(!sttf.needs_scores() && !sttf.is_cacheable(&vctx, 0));
    let r = sttf.rewrite(&top).unwrap().unwrap();
    assert!(!r.needs_scores() && !r.is_cacheable(&vctx, 0));
    let tf = index_reader_functions::term_freq("body", b"red");
    assert!(!tf.needs_scores() && tf.is_cacheable(&vctx, 0));
    let mut v = tf.get_values(&vctx, 0, None).unwrap();
    let mut found = false;
    for d in 0..20 {
        if v.advance_exact(d).unwrap() {
            found = true;
            assert!(v.double_value().unwrap() >= 1.0);
        }
    }
    assert!(found);
    assert!(!v.advance_exact(3).unwrap() || v.advance_exact(3).unwrap());
    assert!(index_reader_functions::term_freq("nosuch", b"x")
        .get_values(&vctx, 0, None)
        .is_ok());
    assert!(index_reader_functions::term_freq("body", b"nosuch")
        .get_values(&vctx, 0, None)
        .is_ok());
    // Conversions forward their rewrites.
    let l = dvs::to_long_values_source(index_reader_functions::max_doc());
    assert!(l.rewrite(&top).unwrap().is_some());
    assert!(dvs::long_constant(3).rewrite(&top).unwrap().is_none());
    // Explanations of a constant and of a query's scores.
    let c = dvs::constant(1.5);
    assert_eq!(c.explain(&vctx, 0, 0, &miss).unwrap().double_value(), 1.5);
    let q = dvs::from_clause(term("red"));
    let e = q.explain(&vctx, 0, 0, &miss).unwrap();
    assert!(!e.description.is_empty());

    // A query's scores through a two-phase query, read by a function query.
    let phrase = Clause::Phrase(crate::query::PhraseQuery::new(
        "body",
        vec![b"red".to_vec(), b"blue".to_vec()],
    ));
    let fq = must(FunctionQuery::new(Arc::new(QueryValueSource::new(phrase.clone(), 0.0))).into());
    assert!(searcher.search(&fq, 10).is_ok());
    let fsq = must(FunctionScoreQuery::new(phrase.clone(), dvs::from_clause(phrase)).into());
    assert!(searcher.search(&fsq, 10).is_ok());

    // Sorting by a value source, before and after the rewrite.
    let sort = vec![sort_field(Arc::clone(&i) as Arc<dyn ValueSource>, false)];
    let composite: Arc<dyn ValueSource> =
        Arc::new(SumFloatFunction::new(vec![Arc::clone(&i), Arc::clone(&i)]));
    let csort = vec![
        sort_field(composite, true),
        crate::top_field::SortField::doc(),
    ];
    let all = must(Clause::MatchAllDocs(crate::query::MatchAllDocsQuery::new(
        0,
    )));
    let readers = reader.segment_readers();
    assert!(searcher
        .search_sorted(readers, &all, 5, &sort, None)
        .is_ok());
    assert!(searcher
        .search_sorted(readers, &all, 5, &csort, None)
        .is_err());
    let rewritten = crate::top_field::rewrite_sort(&csort, &vctx).unwrap();
    let hits = searcher
        .search_sorted(readers, &all, 5, &rewritten, None)
        .unwrap();
    assert_eq!(hits.hits.len(), 5);

    // The scorers' remaining paths, over the function queries' reader-wide
    // preparation.
    let fqq = FunctionQuery::new(Arc::clone(&i));
    let frq = FunctionRangeQuery::new(Arc::clone(&i), None, None, true, true);
    let fmq = FunctionMatchQuery::new(dvs::from_int_field("i"), Arc::new(|v| v > 0.0));
    let absent = FunctionScoreQuery::new(term("nosuch"), dvs::constant(1.0));
    let two_phase = FunctionScoreQuery::new(
        FunctionRangeQuery::new(Arc::clone(&i), Some("1"), None, true, true),
        dvs::constant(2.0),
    );
    let nested = Clause::Boost(Box::new(BoostQuery::new(
        Clause::Boost(Box::new(BoostQuery::new(
            Clause::from(FunctionQuery::new(Arc::new(ConstValueSource::new(
                f32::NAN,
            )))),
            2.0,
        ))),
        3.0,
    )));
    let every = crate::query::BooleanQuery::new().with_should([
        Clause::from(fqq.clone()),
        Clause::from(frq.clone()),
        Clause::from(fmq.clone()),
        Clause::from(absent.clone()),
        Clause::from(two_phase.clone()),
        nested.clone(),
    ]);
    let prepared = crate::multi_segment::global_function_stats(&segments, &every, None)
        .unwrap()
        .unwrap();
    assert!(
        crate::multi_segment::global_function_stats(&segments, &all, None)
            .unwrap()
            .is_none(),
        "nothing to prepare"
    );
    // A leaf reached without the preparation refuses the function query
    // rather than read its one segment as the index.
    let unprepared = leaf_context(&segments[0], None, None, None);
    for e in [
        crate::exec::function::function_query(&unprepared, &fqq, 1.0).err(),
        crate::exec::function::function_score(
            &unprepared,
            &two_phase,
            1.0,
            crate::exec::Mode::Complete,
            false,
        )
        .err(),
    ] {
        let e = e.expect("refused").to_string();
        assert!(e.contains("without its reader-wide preparation"), "{e}");
    }
    let lc = leaf_context(&segments[0], None, Some(&prepared), None);
    let mut bare = lc;
    bare.max_doc = None;
    bare.reader = None;
    assert!(crate::exec::function::function_query(&bare, &fqq, 1.0).is_err());
    let mut s = crate::exec::function::function_query(&lc, &fqq, 1.0)
        .unwrap()
        .unwrap();
    assert_eq!(s.max_score(10).unwrap(), f32::INFINITY);
    let mut s = crate::exec::function::function_range(&lc, &frq)
        .unwrap()
        .unwrap();
    assert_eq!(s.max_score(10).unwrap(), f32::INFINITY);
    let s = crate::exec::function::function_match(&lc, &fmq, 2.0, crate::exec::Mode::TopScores)
        .unwrap()
        .unwrap();
    assert_eq!(s.match_cost(), DEFAULT_MATCH_COST);
    // A query-backed source in a running search: no matches is no values;
    // a two-phase query is checked once per document.
    let vctx = crate::values_source::ValuesContext::for_leaf(lc);
    let mut none = dvs::from_clause(term("nosuch"))
        .get_values(&vctx, 0, None)
        .unwrap();
    assert!(!none.advance_exact(0).unwrap());
    let ranged = dvs::from_clause(
        FunctionRangeQuery::new(Arc::clone(&i), Some("1"), None, true, true).into(),
    );
    let mut v = ranged.get_values(&vctx, 0, None).unwrap();
    let mut matched = 0;
    for d in 0..10 {
        let first = v.advance_exact(d).unwrap();
        assert_eq!(v.advance_exact(d).unwrap(), first);
        matched += i32::from(first);
    }
    assert!(matched > 0 && matched < 10);
    assert!(crate::exec::function::function_score(
        &lc,
        &absent,
        1.0,
        crate::exec::Mode::Complete,
        false
    )
    .unwrap()
    .is_none());
    let mut s = crate::exec::function::function_score(
        &lc,
        &two_phase,
        1.0,
        crate::exec::Mode::Complete,
        false,
    )
    .unwrap()
    .unwrap();
    assert!(s.two_phase() && s.match_cost() > 0.0);
    s.next_doc().unwrap();
    s.matches().unwrap();
    // Explaining: not a function query, or a function score over a query
    // absent from the segment, or a `NaN` value.
    assert!(crate::exec::function::explain(
        &lc,
        &crate::extended_query::ExtendedQuery::Synonym(
            crate::extended_query::SynonymQuery::new("body", [(b"red".to_vec(), 1.0)]).unwrap()
        ),
        0
    )
    .unwrap()
    .is_none());
    for c in [
        term("red"),
        Clause::Boost(Box::new(BoostQuery::new(term("red"), 2.0))),
    ] {
        assert!(crate::exec::function::explain_boosted(
            lc.fields, lc.doc_in, None, None, None, None, None, None, &c, 1.0, 0
        )
        .unwrap()
        .is_none());
    }
    let e = crate::exec::function::explain_boosted(
        lc.fields,
        lc.doc_in,
        None,
        None,
        None,
        None,
        None,
        Some(&prepared),
        &nested,
        1.0,
        0,
    )
    .unwrap()
    .unwrap();
    assert_eq!(e.details[1].value, 6.0);
    let nan = must(
        FunctionScoreQuery::new(
            Clause::MatchAllDocs(crate::query::MatchAllDocsQuery::new(0)),
            as_double_values_source(Arc::new(ConstValueSource::new(f32::NAN))),
        )
        .into(),
    );
    let e = searcher.explain(&nan, 0).unwrap();
    assert!(e.to_string().contains("NaN is an illegal score"));
    let fnan = must(FunctionQuery::new(Arc::new(ConstValueSource::new(-1.0))).into());
    // Java computes the truncated explanation for its value alone.
    let e = searcher.explain(&fnan, 0).unwrap();
    assert_eq!((e.value, e.details[0].value), (0.0, -1.0));
    assert!(searcher.explain(&must(absent.into()), 0).is_ok());
    // `Math.max`/`min` of signed zeros.
    use super::valuesource::functions::{java_max_f32, java_min_f32};
    assert!(java_max_f32(-0.0, 0.0).is_sign_positive());
    assert!(java_min_f32(0.0, -0.0).is_sign_negative());
    // `RangeMapFloatFunction`'s constant form.
    let rm = RangeMapFloatFunction::with_constants(Arc::clone(&i), 0.0, 5.0, 1.0, Some(2.0));
    assert_eq!(
        rm.description(),
        "map(int(i),0.0,5.0,const(1.0),const(2.0))"
    );
    let rm = RangeMapFloatFunction::with_constants(Arc::clone(&i), 0.0, 5.0, 1.0, None);
    assert_eq!(rm.description(), "map(int(i),0.0,5.0,const(1.0),null)");
}

/// The query strings explanations print, and the similarity accessors the
/// function sources and explanations read.
#[test]
fn query_strings_and_similarity_accessors() {
    use crate::explain::describe_clause;
    use crate::similarities::{Bm25Similarity, ClassicSimilarity, PerFieldSimilarity, Similarity};
    let i: Arc<dyn ValueSource> = Arc::new(IntFieldSource::new("i"));
    let fq = Clause::from(FunctionQuery::new(Arc::clone(&i)));
    let fm = Clause::from(FunctionMatchQuery::new(
        dvs::from_int_field("i"),
        Arc::new(|_| true),
    ));
    let fs = Clause::from(FunctionScoreQuery::new(term("red"), dvs::constant(1.0)));
    assert_eq!(describe_clause(&fq), "int(i)");
    assert_eq!(describe_clause(&fm), "FunctionMatchQuery(double(i))");
    assert_eq!(
        describe_clause(&fs),
        "FunctionScoreQuery(body:red, scored by constant(1.0))"
    );
    let syn = Clause::Extended(Box::new(crate::extended_query::ExtendedQuery::Synonym(
        crate::extended_query::SynonymQuery::new("body", [(b"red".to_vec(), 1.0)]).unwrap(),
    )));
    assert_eq!(describe_clause(&syn), "SynonymQuery");
    let mut b = BooleanQuery {
        should: vec![term("red"), term("blue")],
        ..Default::default()
    };
    assert_eq!(
        describe_clause(&Clause::Boolean(Box::new(b.clone()))),
        "body:red body:blue"
    );
    b.minimum_should_match = 1;
    assert_eq!(
        describe_clause(&Clause::Boolean(Box::new(b))),
        "(body:red body:blue)~1"
    );
    assert_eq!(describe_clause(&term("red")), "body:red");
    // Every caller prints a boolean as `BooleanQuery.toString` does: no
    // outer parentheses to strip, so its first and last clauses keep theirs.
    let dm = Clause::DisjunctionMax(Box::new(crate::query::DisjunctionMaxQuery::new(
        [term("a"), term("b")],
        0.0,
    )));
    let sub = Clause::Boolean(Box::new(BooleanQuery {
        should: vec![term("c"), term("d")],
        ..Default::default()
    }));
    let edges = Clause::Boolean(Box::new(BooleanQuery {
        should: vec![dm.clone(), sub.clone()],
        ..Default::default()
    }));
    assert_eq!(
        format!(
            "{:?}",
            FunctionScoreQuery::new(edges.clone(), dvs::constant(1.0))
        ),
        "FunctionScoreQuery((body:a | body:b) (body:c body:d), scored by constant(1.0))"
    );
    let only_dm = Clause::Boolean(Box::new(BooleanQuery {
        should: vec![dm],
        ..Default::default()
    }));
    assert_eq!(
        format!("{:?}", FunctionScoreQuery::new(only_dm, dvs::constant(1.0))),
        "FunctionScoreQuery((body:a | body:b), scored by constant(1.0))"
    );

    let classic = ClassicSimilarity::default();
    assert!(classic.as_tfidf("f").is_some() && classic.shared().is_some());
    assert_eq!(classic.as_tfidf("f").unwrap().tf(4.0), 2.0);
    let bm25 = Bm25Similarity::default();
    assert!(bm25.as_tfidf("f").is_none() && bm25.shared().is_none());
    let per = PerFieldSimilarity::new(Arc::new(bm25)).with_field("t", Arc::new(classic));
    assert!(per.as_tfidf("t").is_some() && per.as_tfidf("f").is_none());
    assert!(per.shared().is_some());
}

#[test]
fn value_source_group_selector_states() {
    use crate::grouping::{GroupSelector, GroupState, SearchGroup, ValueSourceGroupSelector};
    let reader = index();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let source: Arc<dyn ValueSource> = Arc::new(IntFieldSource::new("i"));
    let norms: Vec<SegmentNorms<'_, '_>> = segments.iter().map(|_| None).collect();
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    let top = TopLevel::of_searcher(&searcher, None);
    let max_doc = reader.segment_readers()[0].max_doc;
    let context = Arc::new(FunctionContext::create(source.as_ref(), &top).unwrap());
    let mut sel = ValueSourceGroupSelector::new(Arc::clone(&source), Arc::clone(&context));
    // Moved before a segment: Java's NullPointerException.
    assert!(matches!(
        sel.advance_to(0, 1.0),
        Err(crate::Error::IllegalState(_))
    ));
    // Without the searcher's leaves the segment stands alone.
    sel.set_next_reader(0, &segments[0]).unwrap();
    let mut present = None;
    let mut missing = None;
    for doc in 0..max_doc {
        match sel.advance_to(doc, 1.0).unwrap() {
            GroupState::Accept => {
                present.get_or_insert((doc, sel.current_value().cloned().unwrap()));
            }
            GroupState::Skip => {
                assert!(sel.current_value().is_none());
                missing.get_or_insert(doc);
            }
        }
    }
    let (doc, value) = present.unwrap();
    let missing = missing.expect("the fixture has documents without i");
    // The second pass keeps only the chosen groups; the empty group is
    // accepted once it is one of them. Values are read forward, so each
    // pass enters the segment again.
    sel.set_groups(&[SearchGroup {
        group_value: Some(value.clone()),
        sort_values: Vec::new(),
    }]);
    sel.set_next_reader(0, &segments[0]).unwrap();
    let mut others = 0;
    for d in 0..max_doc {
        let state = sel.advance_to(d, 1.0).unwrap();
        if d == doc {
            assert_eq!(state, GroupState::Accept);
        } else if d == missing {
            assert_eq!(state, GroupState::Skip);
        } else if state == GroupState::Skip && sel.current_value().is_some() {
            others += 1;
        }
    }
    assert!(others > 0);
    sel.set_groups(&[SearchGroup {
        group_value: None,
        sort_values: Vec::new(),
    }]);
    sel.set_next_reader(0, &segments[0]).unwrap();
    for d in 0..max_doc {
        let state = sel.advance_to(d, 1.0).unwrap();
        if d == doc {
            assert_eq!(state, GroupState::Skip);
        } else if d == missing {
            assert_eq!(state, GroupState::Accept);
        }
    }
}

/// A boolean's explanation takes its value from the boolean's scorer
/// (`BooleanWeight.explain`): that scorer runs under the searcher's
/// similarity, so the value is the score the search gives -- with term,
/// `FieldExistsQuery` and `DocAndScoreQuery` clauses (which explain through
/// that scorer) and a function query that needs a TFIDF similarity.
#[test]
fn a_booleans_explanation_is_its_score_under_the_searchers_similarity() {
    let reader = index();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let owned = reader.field_norms_by_field(&["body".to_string()]);
    let norms: Vec<SegmentNorms<'_, '_>> = owned.iter().map(Some).collect();
    let classic = crate::similarities::ClassicSimilarity::default();
    let mut cs = IndexSearcher::new(&segments, &norms).unwrap();
    cs.set_similarity(&classic);
    let bm25 = IndexSearcher::new(&segments, &norms).unwrap();
    let bases: Vec<i32> = segments.iter().map(|s| s.doc_base).collect();
    let fixed = crate::extended_query::DocAndScoreQuery::new(
        vec![(0, 1.5), (7, 0.25), (29, 2.0), (61, 0.75)],
        &bases,
    );
    let tf = Clause::from(FunctionQuery::new(Arc::new(TFValueSource::new(
        "body",
        "red",
        "body",
        b"red".to_vec(),
    ))));
    let shapes = [
        BooleanQuery {
            must: vec![term("red")],
            should: vec![Clause::Exists(crate::query::FieldExistsQuery::new("i"))],
            ..Default::default()
        },
        BooleanQuery {
            should: vec![term("blue"), Clause::from(fixed)],
            ..Default::default()
        },
        BooleanQuery {
            must: vec![term("red")],
            should: vec![tf],
            ..Default::default()
        },
    ];
    for (searcher, name) in [(&cs, "classic"), (&bm25, "bm25")] {
        for (n, q) in shapes.iter().enumerate() {
            let hits = searcher.search(q, 100);
            if name == "bm25" && n == 2 {
                // `tf()` needs a TFIDF similarity, as in Java.
                assert!(hits.is_err());
                continue;
            }
            let hits = hits.unwrap();
            assert!(hits.score_docs.len() > 1, "{name} {n}");
            for h in &hits.score_docs {
                let e = searcher.explain(q, h.doc).unwrap();
                assert!(e.matched, "{name} {n} doc {}", h.doc);
                assert_eq!(
                    e.value.to_bits(),
                    h.score.to_bits(),
                    "{name} {n} doc {}: {e}",
                    h.doc
                );
            }
        }
    }
}

/// Two sources' vectors of different lengths: Java's `VectorUtil` throws
/// `IllegalArgumentException("vector dimensions differ: a!=b")` from the
/// similarity (`FloatVectorSimilarityFunction.func`'s own `assert` is off in
/// production), for float and byte vectors alike.
#[test]
fn vector_similarities_refuse_vectors_of_different_lengths() {
    use lucene_codecs::field_infos::VectorSimilarityFunction as Sim;
    let reader = index();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let leaf = ValueLeaf::of_segment(&segments[0]);
    let fcx = FunctionContext::new();
    let f3: Arc<dyn ValueSource> =
        Arc::new(ConstKnnFloatValueSource::new(vec![1.0, 2.0, 3.0]).unwrap());
    let f2: Arc<dyn ValueSource> = Arc::new(ConstKnnFloatValueSource::new(vec![1.0, 2.0]).unwrap());
    let b3: Arc<dyn ValueSource> = Arc::new(ConstKnnByteVectorValueSource::new(vec![1, 2, 3]));
    let b4: Arc<dyn ValueSource> = Arc::new(ConstKnnByteVectorValueSource::new(vec![1, 2, 3, 4]));
    for sim in [
        Sim::Euclidean,
        Sim::DotProduct,
        Sim::Cosine,
        Sim::MaximumInnerProduct,
    ] {
        for (s, want) in [
            (
                FloatVectorSimilarityFunction::new(sim, Arc::clone(&f3), Arc::clone(&f2)),
                "3!=2",
            ),
            (
                ByteVectorSimilarityFunction::new(sim, Arc::clone(&b3), Arc::clone(&b4)),
                "3!=4",
            ),
        ] {
            let mut v = s.get_values(&fcx, &leaf).unwrap();
            for r in [v.float_val(0).map(f64::from), v.double_val(0)] {
                match r {
                    Err(Error::IllegalArgument(m)) => {
                        assert_eq!(m, format!("vector dimensions differ: {want}"), "{sim:?}")
                    }
                    other => panic!("{sim:?}: {other:?}"),
                }
            }
            assert!(v.str_val(0).is_err());
        }
        // Equal lengths still compare.
        let same = FloatVectorSimilarityFunction::new(sim, Arc::clone(&f3), Arc::clone(&f3));
        assert!(same.get_values(&fcx, &leaf).unwrap().float_val(0).is_ok());
    }
}

/// `collect_functions` walks each function's scored queries once: a chain
/// of `query()` sources nested `n` deep yields `n` functions, not one per
/// path through the chain (which grew as `2^n`).
#[test]
fn nested_function_queries_are_collected_once_each() {
    let mut c = term("red");
    for _ in 0..20 {
        c = Clause::from(FunctionQuery::new(Arc::new(QueryValueSource::new(c, 0.0))));
    }
    // And through a values source's query and a function score's own query.
    let fsq = FunctionScoreQuery::new(c, dvs::from_clause(term("blue")));
    let q = must(Clause::from(fsq));
    let mut found = Vec::new();
    collect_functions(&q, &mut found);
    assert_eq!(found.len(), 21);
}

/// The batched function paths against the document-at-a-time ones.
mod batches {
    // The batched function paths against the document-at-a-time ones they
    // stand in for: every value source's batch getters against its
    // per-document getters, and whole searches with the batches on and off
    // (`exec::batches_on`), over an index this port writes -- two segments,
    // one past `NumericColumn::get_batch`'s widest window, sparse and dense
    // columns, deleted documents, and terms long enough for impacts to skip
    // blocks. The batches' agreement with Lucene is
    // `tests/function_fixtures.rs` and the benchmark digests.
    use std::sync::Arc;

    use lucene_index::buffered_updates::Term;
    use lucene_index::document::{self as d, Document, IndexableField, Store};
    use lucene_index::index_writer::IndexWriter;
    use lucene_index::segment_info::LuceneVersion;
    use lucene_store::FsDirectory;
    use lucene_util::test_support::TempDir;

    use super::super::valuesource::*;
    use super::super::*;
    use crate::directory_reader::DirectoryReader;
    use crate::index_searcher::{IndexSearcher, SegmentNorms};
    use crate::query::{BooleanQuery, BoostQuery, Clause, TermQuery};
    use crate::top_docs::TopDocs;
    use crate::values_source::{self as dvs, ValuesContext};

    const VERSION: LuceneVersion = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };

    /// Two segments (5 000 and 300 documents), every eleventh document of the
    /// first deleted. `i` is sparse (missing on every seventh), `n` and `f`
    /// dense, `g` a sparse float column; `body` holds `a` everywhere, `b` on
    /// every third (twice on every fifth), `c` on three in seven; `s` a sparse
    /// `SORTED` column.
    fn index(tmp: &TempDir) -> DirectoryReader {
        let dir = FsDirectory::open(tmp.path());
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
        for (seg, count) in [(0, 5000i64), (1, 300)] {
            for i in 0..count {
                let mut doc = Document::new();
                let mut body = String::from("a");
                if i % 3 == 0 {
                    body.push_str(" b");
                    if i % 5 == 0 {
                        body.push_str(" b");
                    }
                }
                if i % 7 < 3 {
                    body.push_str(" c");
                }
                let fields: Vec<Box<dyn IndexableField>> = vec![
                    Box::new(d::StringField::new("id", format!("{seg}-{i}"), Store::No)),
                    Box::new(d::TextField::new("body", body, Store::No)),
                    Box::new(d::NumericDocValuesField::new("n", i * 7919 % 100_000)),
                    Box::new(d::NumericDocValuesField::new(
                        "f",
                        i64::from(((i * 13 % 997) as f32 / 10.0 - 5.0).to_bits()),
                    )),
                ];
                for f in fields {
                    doc.add_boxed(f);
                }
                if i % 7 != 0 {
                    doc.add_boxed(Box::new(d::NumericDocValuesField::new("i", i * 37 % 1000)));
                }
                if i % 5 != 0 {
                    doc.add_boxed(Box::new(d::SortedDocValuesField::new(
                        "s",
                        format!("w{}", i % 13).into_bytes(),
                    )));
                }
                if i % 4 != 0 {
                    doc.add_boxed(Box::new(d::NumericDocValuesField::new(
                        "g",
                        i64::from(((i % 50) as f32 * 1.5).to_bits()),
                    )));
                }
                w.add_fields_document(&doc).unwrap();
            }
            w.commit().unwrap();
        }
        let deleted: Vec<Term> = (0..5000)
            .step_by(11)
            .map(|i| Term::new("id", format!("0-{i}")))
            .collect();
        w.delete_documents_by_term(&deleted).unwrap();
        w.commit().unwrap();
        drop(w);
        DirectoryReader::open(&dir).unwrap()
    }

    fn composite() -> Arc<dyn ValueSource> {
        Arc::new(SumFloatFunction::new(vec![
            Arc::new(ProductFloatFunction::new(vec![
                Arc::new(IntFieldSource::new("i")),
                Arc::new(ConstValueSource::new(0.5)),
            ])),
            Arc::new(LinearFloatFunction::new(
                Arc::new(FloatFieldSource::new("f")),
                2.0,
                1.0,
            )),
            Arc::new(ReciprocalFloatFunction::new(
                Arc::new(LongFieldSource::new("n")),
                0.001,
                10.0,
                1.0,
            )),
        ]))
    }

    fn sources() -> Vec<Arc<dyn ValueSource>> {
        vec![
            Arc::new(IntFieldSource::new("i")),
            Arc::new(LongFieldSource::new("n")),
            Arc::new(FloatFieldSource::new("f")),
            Arc::new(FloatFieldSource::new("g")),
            Arc::new(IntFieldSource::new("nosuch")),
            Arc::new(DoubleFieldSource::new("n")),
            composite(),
            Arc::new(ProductFloatFunction::new(vec![
                Arc::new(IntFieldSource::new("i")),
                Arc::new(FloatFieldSource::new("g")),
            ])),
            Arc::new(DivFloatFunction::new(
                Arc::new(FloatFieldSource::new("f")),
                Arc::new(IntFieldSource::new("i")),
            )),
            Arc::new(MaxFloatFunction::new(vec![
                Arc::new(IntFieldSource::new("i")),
                Arc::new(FloatFieldSource::new("g")),
            ])),
            Arc::new(BytesRefFieldSource::new("s")),
        ]
    }

    /// The batches a scorer may hand its values: aligned runs, runs from an
    /// odd start, every third document, single documents, and one batch whose
    /// span passes the widest window.
    fn patterns(max_doc: i32) -> Vec<Vec<Vec<i32>>> {
        let all: Vec<i32> = (0..max_doc).collect();
        let chunks =
            |docs: &[i32], n: usize| docs.chunks(n).map(<[i32]>::to_vec).collect::<Vec<_>>();
        let thirds: Vec<i32> = (0..max_doc).step_by(3).collect();
        let mut odd = vec![(0..5).collect::<Vec<i32>>()];
        odd.extend(chunks(&all[5..], 128));
        let mut wide = vec![vec![0, 1, max_doc - 1]];
        if max_doc > 4100 {
            wide = vec![vec![0, 4500, max_doc - 1]];
        }
        vec![
            chunks(&all, 64),
            odd,
            chunks(&thirds, 100),
            chunks(&all, 1),
            wide,
        ]
    }

    fn bits(r: &Result<f32>) -> std::result::Result<u32, String> {
        match r {
            Ok(v) => Ok(v.to_bits()),
            Err(e) => Err(e.to_string()),
        }
    }

    #[test]
    fn value_batches_read_what_the_per_document_getters_read() {
        let tmp = TempDir::new("function-batches-values");
        let reader = index(&tmp);
        let opened = reader.open_segments().unwrap();
        let segments = opened.as_open_segments();
        let norms: Vec<SegmentNorms<'_, '_>> = segments.iter().map(|_| None).collect();
        let searcher = IndexSearcher::new(&segments, &norms).unwrap();
        let top = TopLevel::of_searcher(&searcher, None);
        for src in sources() {
            let fcx = FunctionContext::create(src.as_ref(), &top).unwrap();
            for (leaf, seg) in segments.iter().enumerate() {
                let vleaf = ValueLeaf::of_searcher(&searcher, leaf, None).unwrap();
                let max_doc = seg.reader.unwrap().max_doc;
                let mut one = src.get_values(&fcx, &vleaf).unwrap();
                let floats: Vec<Result<f32>> = (0..max_doc).map(|d| one.float_val(d)).collect();
                let mut one = src.get_values(&fcx, &vleaf).unwrap();
                let doubles: Vec<Result<f64>> = (0..max_doc).map(|d| one.double_val(d)).collect();
                for pattern in patterns(max_doc) {
                    let what = format!("{} leaf {leaf}", src.description());
                    let mut fv = src.get_values(&fcx, &vleaf).unwrap();
                    let mut dv = src.get_values(&fcx, &vleaf).unwrap();
                    for batch in &pattern {
                        let mut out = vec![0.0f32; batch.len()];
                        let got = fv.float_val_batch(batch, &mut out);
                        let want: Vec<&Result<f32>> =
                            batch.iter().map(|&d| &floats[d as usize]).collect();
                        match got {
                            Ok(()) => {
                                for (o, w) in out.iter().zip(&want) {
                                    assert_eq!(Ok(o.to_bits()), bits(w), "{what} floats");
                                }
                            }
                            Err(_) => assert!(want.iter().any(|w| w.is_err()), "{what}"),
                        }
                        let mut out = vec![0.0f64; batch.len()];
                        let got = dv.double_val_batch(batch, &mut out);
                        match got {
                            Ok(()) => {
                                for (o, &doc) in out.iter().zip(batch) {
                                    let w = doubles[doc as usize].as_ref().unwrap();
                                    assert_eq!(o.to_bits(), w.to_bits(), "{what} doubles");
                                }
                            }
                            Err(_) => assert!(
                                batch.iter().any(|&d| doubles[d as usize].is_err()),
                                "{what}"
                            ),
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn range_batches_match_and_score_as_the_scorer_does() {
        let tmp = TempDir::new("function-batches-ranges");
        let reader = index(&tmp);
        let opened = reader.open_segments().unwrap();
        let segments = opened.as_open_segments();
        let norms: Vec<SegmentNorms<'_, '_>> = segments.iter().map(|_| None).collect();
        let searcher = IndexSearcher::new(&segments, &norms).unwrap();
        let top = TopLevel::of_searcher(&searcher, None);
        for src in sources() {
            let fcx = FunctionContext::create(src.as_ref(), &top).unwrap();
            for (leaf, seg) in segments.iter().enumerate() {
                let vleaf = ValueLeaf::of_searcher(&searcher, leaf, None).unwrap();
                let max_doc = seg.reader.unwrap().max_doc;
                let mut probe = src.get_values(&fcx, &vleaf).unwrap();
                let own = probe.range_matcher(Some("100"), Some("500"), true, false);
                let mut ranges: Vec<Option<RangeMatcher>> = vec![
                    None,
                    Some(RangeMatcher::float(Some("-2.5"), Some("40"), false, true).unwrap()),
                    Some(RangeMatcher::double(Some("1"), None, true, true).unwrap()),
                    Some(RangeMatcher::int(Some(100), Some(500), false, true)),
                    Some(RangeMatcher::long(None, Some(50_000), true, false)),
                    Some(RangeMatcher::Ord { lower: 0, upper: 3 }),
                ];
                if let Ok(r) = own {
                    ranges.push(Some(r));
                }
                for range in &ranges {
                    let what = format!("{} leaf {leaf} {range:?}", src.description());
                    // Per document, as `ValueSourceScorer` asks: scored, and
                    // matched only (a filter's).
                    let all: Vec<i32> = (0..max_doc).collect();
                    let expect = |scored: bool| {
                        let mut one = src.get_values(&fcx, &vleaf).unwrap();
                        let mut want = (Vec::new(), Vec::new());
                        let values = scored.then_some(&mut want.1);
                        let ok = range_batch_per_doc(
                            one.as_mut(),
                            range.as_ref(),
                            &all,
                            &mut want.0,
                            values,
                        );
                        (ok.is_ok(), want)
                    };
                    for (pattern, scored) in [(0, true), (1, false), (2, true)] {
                        let (want_ok, want) = expect(scored);
                        let batches = &patterns(max_doc)[pattern];
                        let mut v = src.get_values(&fcx, &vleaf).unwrap();
                        let mut got = (Vec::new(), Vec::new());
                        let mut ok = Ok(());
                        for batch in batches {
                            let values = scored.then_some(&mut got.1);
                            ok = v.range_batch(range.as_ref(), batch, &mut got.0, values);
                            if ok.is_err() {
                                break;
                            }
                        }
                        if pattern == 2 {
                            // Every third document: the per-document matches
                            // among them.
                            if want_ok && ok.is_ok() {
                                let thirds: Vec<i32> =
                                    want.0.iter().copied().filter(|d| d % 3 == 0).collect();
                                assert_eq!(got.0, thirds, "{what}");
                            }
                            continue;
                        }
                        assert_eq!(ok.is_ok(), want_ok, "{what}");
                        if ok.is_ok() {
                            assert_eq!(got.0, want.0, "{what}");
                            if scored {
                                let a: Vec<u32> = got.1.iter().map(|f| f.to_bits()).collect();
                                let b: Vec<u32> = want.1.iter().map(|f| f.to_bits()).collect();
                                assert_eq!(a, b, "{what}");
                            } else {
                                assert!(got.1.is_empty());
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn a_column_batch_refuses_documents_behind_the_last_one_asked() {
        let tmp = TempDir::new("function-batches-order");
        let reader = index(&tmp);
        let opened = reader.open_segments().unwrap();
        let segments = opened.as_open_segments();
        let norms: Vec<SegmentNorms<'_, '_>> = segments.iter().map(|_| None).collect();
        let searcher = IndexSearcher::new(&segments, &norms).unwrap();
        let top = TopLevel::of_searcher(&searcher, None);
        let vleaf = ValueLeaf::of_searcher(&searcher, 0, None).unwrap();
        for field in ["i", "n", "f", "g"] {
            let src = FloatFieldSource::new(field);
            let fcx = FunctionContext::create(&src, &top).unwrap();
            let mut v = src.get_values(&fcx, &vleaf).unwrap();
            v.float_val(100).unwrap();
            let mut out = [0.0f32; 2];
            assert!(matches!(
                v.float_val_batch(&[50, 51], &mut out),
                Err(Error::IllegalArgument(_))
            ));
            // A window batch leaves the column where its last `get` would.
            v.float_val_batch(&(128..192).collect::<Vec<_>>(), &mut [0.0; 64])
                .unwrap();
            assert!(v.exists(191).is_ok());
            assert!(v.exists(190).is_err());
        }
    }

    #[test]
    fn wrapped_sources_batch_only_without_a_score_reader() {
        let tmp = TempDir::new("function-batches-wrapped");
        let reader = index(&tmp);
        let opened = reader.open_segments().unwrap();
        let segments = opened.as_open_segments();
        let norms: Vec<SegmentNorms<'_, '_>> = segments.iter().map(|_| None).collect();
        let searcher = IndexSearcher::new(&segments, &norms).unwrap();
        let ctx = ValuesContext::new(&searcher);
        let plain = as_double_values_source(composite());
        let mut v = plain.get_values(&ctx, 1, None).unwrap();
        assert!(v.batch_capable());
        let docs: Vec<i32> = (0..300).collect();
        let (mut out, mut has) = (vec![0.0; 300], vec![false; 300]);
        v.fill_batch(&docs, &[0.0; 300], &mut out, &mut has)
            .unwrap();
        let mut one = plain.get_values(&ctx, 1, None).unwrap();
        for (&d, (&o, &h)) in docs.iter().zip(out.iter().zip(&has)) {
            assert!(h && one.advance_exact(d).unwrap());
            assert_eq!(o.to_bits(), one.double_value().unwrap().to_bits());
        }
        // A source reading the scores through the wrapper's view: per document.
        let scored = as_double_values_source(Arc::new(ProductFloatFunction::new(vec![
            from_double_values_source(dvs::scores()),
            Arc::new(IntFieldSource::new("i")),
        ])));
        let v = scored.get_values(&ctx, 1, None).unwrap();
        assert!(!v.batch_capable());
    }

    fn term(w: &str) -> Clause {
        Clause::Term(TermQuery::new("body", w.as_bytes().to_vec()))
    }

    fn queries() -> Vec<BooleanQuery> {
        let one = |c: Clause| BooleanQuery {
            must: vec![c],
            ..Default::default()
        };
        let filtered = |w: &str, f: Clause| BooleanQuery {
            must: vec![term(w)],
            filter: vec![f],
            ..Default::default()
        };
        let frange = |src: Arc<dyn ValueSource>, lo: Option<&str>, hi: Option<&str>| -> Clause {
            FunctionRangeQuery::new(src, lo, hi, true, false).into()
        };
        let int_i = || -> Arc<dyn ValueSource> { Arc::new(IntFieldSource::new("i")) };
        let float_f = || -> Arc<dyn ValueSource> { Arc::new(FloatFieldSource::new("f")) };
        let mut out = Vec::new();
        for w in ["a", "b", "c"] {
            out.push(one(FunctionScoreQuery::new(
                term(w),
                dvs::from_float_field("f"),
            )
            .into()));
            out.push(one(FunctionScoreQuery::new(
                term(w),
                dvs::from_int_field("i"),
            )
            .into()));
            out.push(one(FunctionScoreQuery::boost_by_value(
                term(w),
                as_double_values_source(composite()),
            )
            .into()));
            out.push(one(FunctionScoreQuery::boost_by_value(
                term(w),
                dvs::from_float_field("g"),
            )
            .into()));
            out.push(one(FunctionScoreQuery::new(
                term(w),
                as_double_values_source(composite()),
            )
            .into()));
            out.push(filtered(w, frange(int_i(), Some("100"), Some("500"))));
            out.push(filtered(w, frange(float_f(), Some("-1"), None)));
            out.push(filtered(w, frange(composite(), None, Some("300"))));
            out.push(filtered(
                w,
                FunctionMatchQuery::new(dvs::from_int_field("i"), Arc::new(|v| v > 500.0)).into(),
            ));
            out.push(filtered(
                w,
                FunctionMatchQuery::new(
                    as_double_values_source(composite()),
                    Arc::new(|v| v < 200.0),
                )
                .into(),
            ));
            out.push(filtered(
                w,
                Clause::Boost(Box::new(BoostQuery {
                    inner: Box::new(
                        FunctionMatchQuery::new(dvs::from_float_field("g"), Arc::new(|v| v > 10.0))
                            .into(),
                    ),
                    boost: 2.0,
                })),
            ));
            out.push(BooleanQuery {
                should: vec![term(w), frange(int_i(), Some("900"), None)],
                ..Default::default()
            });
            out.push(one(FunctionQuery::new(Arc::new(TermFreqValueSource::new(
                "body",
                w,
                "body",
                w.as_bytes(),
            )))
            .into()));
        }
        for src in [
            int_i(),
            float_f(),
            composite(),
            Arc::new(FloatFieldSource::new("g")),
        ] {
            out.push(one(FunctionQuery::new(Arc::clone(&src)).into()));
            out.push(one(Clause::Boost(Box::new(BoostQuery {
                inner: Box::new(FunctionQuery::new(Arc::clone(&src)).into()),
                boost: 0.5,
            }))));
            out.push(one(frange(Arc::clone(&src), Some("2"), Some("300"))));
            out.push(one(frange(src, None, None)));
        }
        out
    }

    fn run(searcher: &IndexSearcher<'_, '_>, q: &BooleanQuery, n: usize, off: bool) -> String {
        crate::exec::tests::BATCHES_OFF.with(|c| c.set(off));
        let r = searcher.search(q, n);
        crate::exec::tests::BATCHES_OFF.with(|c| c.set(false));
        match r {
            Ok(TopDocs {
                total_hits,
                score_docs,
            }) => {
                let hits: Vec<(i32, u32)> = score_docs
                    .iter()
                    .map(|d| (d.doc, d.score.to_bits()))
                    .collect();
                format!("{total_hits:?} {hits:?}")
            }
            Err(e) => format!("error {e}"),
        }
    }

    #[test]
    fn searches_collect_the_same_hits_with_and_without_batches() {
        let tmp = TempDir::new("function-batches-searches");
        let reader = index(&tmp);
        let opened = reader.open_segments().unwrap();
        let mut segments = opened.as_open_segments();
        for seg in &mut segments {
            seg.cache = None;
        }
        let owned = reader.field_norms_by_field(&["body".to_string()]);
        let norms: Vec<SegmentNorms<'_, '_>> = owned.iter().map(Some).collect();
        let searcher = IndexSearcher::new(&segments, &norms).unwrap();
        for q in queries() {
            for n in [1, 10, 2000] {
                let on = run(&searcher, &q, n, false);
                let off = run(&searcher, &q, n, true);
                assert_eq!(on, off, "{q:?} top {n}");
                assert!(!on.starts_with("error"), "{q:?}: {on}");
            }
        }
    }
}

/// The wrapped query's score as the value, handing the weight's boost to the
/// wrapped query (`boosts_wrapped_query`), as OpenSearch's function score does.
struct BoostsWrapped;

impl crate::values_source::DoubleValuesSource for BoostsWrapped {
    fn get_values<'c>(
        &self,
        ctx: &crate::values_source::ValuesContext<'c>,
        leaf: usize,
        scores: Option<crate::values_source::BoxDoubleValues<'c>>,
    ) -> Result<crate::values_source::BoxDoubleValues<'c>> {
        dvs::scores().get_values(ctx, leaf, scores)
    }
    fn needs_scores(&self) -> bool {
        true
    }
    fn is_cacheable(&self, _: &crate::values_source::ValuesContext<'_>, _: usize) -> bool {
        false
    }
    fn describe(&self) -> String {
        "boosts_wrapped".into()
    }
    fn boosts_wrapped_query(&self) -> bool {
        true
    }
}

#[test]
fn a_source_may_take_the_boost_onto_its_wrapped_query() {
    let reader = index();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let owned = reader.field_norms_by_field(&["body".to_string()]);
    let norms: Vec<SegmentNorms<'_, '_>> = owned.iter().map(Some).collect();
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    assert!(!dvs::scores().boosts_wrapped_query());
    let boosted = |source: Arc<dyn crate::values_source::DoubleValuesSource>| {
        let fsq = Clause::from(FunctionScoreQuery::new(
            BooleanQuery {
                must: vec![term("red")],
                should: vec![term("blue")],
                ..Default::default()
            },
            source,
        ));
        let q = must(Clause::Boost(Box::new(BoostQuery::new(fsq, 3.0))));
        let hits = searcher.search(&q, 100).unwrap().score_docs;
        let explained: Vec<f32> = hits
            .iter()
            .map(|h| searcher.explain(&q, h.doc).unwrap().value)
            .collect();
        (hits, explained)
    };
    let plain = searcher
        .search(
            &must(Clause::Boost(Box::new(BoostQuery::new(
                Clause::Boolean(Box::new(BooleanQuery {
                    must: vec![term("red")],
                    should: vec![term("blue")],
                    ..Default::default()
                })),
                3.0,
            )))),
            100,
        )
        .unwrap()
        .score_docs;
    // Lucene's: the value times the boost; the wrapped query's own: its boosted
    // score. For the scores source both are the boosted wrapped score, bit for
    // bit only where `(float) (3 * s)` equals the score of the boosted query.
    let (lucene, _) = boosted(dvs::scores());
    let (own, explained) = boosted(Arc::new(BoostsWrapped));
    assert_eq!(own.len(), plain.len());
    for ((a, b), e) in own.iter().zip(&plain).zip(&explained) {
        assert_eq!((a.doc, a.score.to_bits()), (b.doc, b.score.to_bits()));
        assert_eq!(e.to_bits(), a.score.to_bits());
    }
    assert_eq!(lucene.len(), own.len());
}
