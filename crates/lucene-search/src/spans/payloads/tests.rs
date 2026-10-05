use super::*;

/// `Math.max`/`Math.min` on floats: NaN wins, `-0.0` is below `0.0`.
#[test]
fn the_functions_combine_as_java_floats_do() {
    assert!(java_max(f32::NAN, 1.0).is_nan());
    assert!(java_max(1.0, f32::NAN).is_nan());
    assert!(java_min(f32::NAN, 1.0).is_nan());
    assert!(java_min(1.0, f32::NAN).is_nan());
    assert!(java_max(-0.0, 0.0).is_sign_positive());
    assert!(java_max(0.0, -0.0).is_sign_positive());
    assert!(java_min(-0.0, 0.0).is_sign_negative());
    assert!(java_min(0.0, -0.0).is_sign_negative());
    assert_eq!(java_max(2.0, 3.0), 3.0);
    assert_eq!(java_min(2.0, 3.0), 2.0);

    use PayloadFunction::*;
    // The first payload seeds min and max; sum and average add up.
    assert_eq!(Min.current_score(0, 9.0, 4.0), 4.0);
    assert_eq!(Min.current_score(1, 9.0, 4.0), 4.0);
    assert_eq!(Min.current_score(1, 3.0, 4.0), 3.0);
    assert_eq!(Max.current_score(0, 9.0, 4.0), 4.0);
    assert_eq!(Max.current_score(1, 9.0, 4.0), 9.0);
    assert_eq!(Sum.current_score(3, 9.0, 4.0), 13.0);
    assert_eq!(Average.current_score(3, 9.0, 4.0), 13.0);
    // No payload seen scores 1; the average divides by the count.
    for f in [Min, Max, Sum, Average] {
        assert_eq!(f.doc_score(0, 7.0), 1.0, "{f:?}");
    }
    assert_eq!(Average.doc_score(4, 10.0), 2.5);
    assert_eq!(Sum.doc_score(4, 10.0), 10.0);
    assert_eq!(
        Max.explain(2, 5.0).to_string(),
        "5.0 = MaxPayloadFunction.docScore()\n"
    );
    assert_eq!(Min.name(), "MinPayloadFunction");
    assert_eq!(Sum.name(), "SumPayloadFunction");
    assert_eq!(Average.name(), "AveragePayloadFunction");
}

/// `PayloadSpanScorer`: a negative or NaN function score counts as 0, and
/// the explanation says which.
#[test]
fn negative_and_nan_payload_scores_are_truncated() {
    assert_eq!(payload_score(PayloadFunction::Sum, 2, -3.0), 0.0);
    assert_eq!(payload_score(PayloadFunction::Sum, 2, f32::NAN), 0.0);
    assert_eq!(payload_score(PayloadFunction::Sum, 2, 3.0), 3.0);
    let neg = payload_explanation(PayloadFunction::Sum, 2, -3.0);
    assert_eq!(
        neg.to_string(),
        "0.0 = truncated score, max of:\n  0.0 = minimum score\n  -3.0 = SumPayloadFunction.docScore()\n"
    );
    let nan = payload_explanation(PayloadFunction::Max, 1, f32::NAN);
    assert_eq!(nan.value, 0.0);
    assert!(nan.description.starts_with("payload score, computed as"));
    assert_eq!(nan.details.len(), 1);
    let plain = payload_explanation(PayloadFunction::Min, 0, 0.0);
    assert_eq!(plain.value, 1.0);
}

/// `FLOAT_DECODER` reads the first byte as a signed `byte`; a caller's
/// decoder is equal only to itself.
#[test]
fn decoders_read_and_compare_as_java_does() {
    let d = PayloadDecoder::Float;
    assert_eq!(d.compute_payload_factor(None), 1.0);
    assert_eq!(d.compute_payload_factor(Some(&[])), 1.0);
    assert_eq!(d.compute_payload_factor(Some(&[3, 9])), 3.0);
    assert_eq!(d.compute_payload_factor(Some(&[0xfe])), -2.0);
    let f: Arc<DecodeFn> = Arc::new(|p: Option<&[u8]>| p.map_or(0.5, |b| b.len() as f32));
    let c = PayloadDecoder::Custom(Arc::clone(&f));
    assert_eq!(c.compute_payload_factor(Some(&[1, 2])), 2.0);
    assert_eq!(c.compute_payload_factor(None), 0.5);
    assert_eq!(c, PayloadDecoder::Custom(f));
    assert_ne!(c, PayloadDecoder::Custom(Arc::new(|_: Option<&[u8]>| 0.0)));
    assert_ne!(c, d);
    assert_eq!(d, PayloadDecoder::Float);
    assert_eq!(format!("{d:?}"), "FLOAT_DECODER");
    assert_eq!(format!("{c:?}"), "PayloadDecoder");
}

/// `PayloadMatcherFactory`: equality on the bytes whatever the type; ints
/// and floats big-endian; strings by UTF-16 unit, as `String.compareTo`.
#[test]
fn matchers_compare_as_the_factorys_do() {
    use MatchOperation::*;
    use PayloadType::*;
    let i = |v: i32| v.to_be_bytes().to_vec();
    let f = |v: f32| v.to_be_bytes().to_vec();
    for t in [Int, Float, String] {
        assert!(compare_payload(t, Eq, b"ab", b"ab").unwrap());
        assert!(!compare_payload(t, Eq, b"ab", b"abc").unwrap());
    }
    // `comparePayload(source = threshold, payload = value)`.
    assert!(compare_payload(Int, Gt, &i(3), &i(4)).unwrap());
    assert!(!compare_payload(Int, Gt, &i(3), &i(3)).unwrap());
    assert!(compare_payload(Int, Gte, &i(3), &i(3)).unwrap());
    assert!(compare_payload(Int, Lt, &i(3), &i(-7)).unwrap());
    assert!(compare_payload(Int, Lte, &i(-7), &i(-7)).unwrap());
    assert!(!compare_payload(Int, Lte, &i(-8), &i(-7)).unwrap());
    assert!(compare_payload(Float, Gt, &f(0.5), &f(0.75)).unwrap());
    assert!(compare_payload(Float, Gte, &f(0.5), &f(0.5)).unwrap());
    assert!(compare_payload(Float, Lt, &f(0.5), &f(-1.0)).unwrap());
    assert!(compare_payload(Float, Lte, &f(0.5), &f(0.5)).unwrap());
    assert!(!compare_payload(Float, Lt, &f(0.5), &f(f32::NAN)).unwrap());
    // Too short for an int or a float: Java's read is out of bounds.
    assert!(compare_payload(Int, Gt, &i(3), &[1, 2]).is_err());
    assert!(compare_payload(Float, Lt, &[0], &f(1.0)).is_err());
    // Strings: "b" > "ab"; U+00E9 sorts after ASCII; a supplementary
    // character's surrogates sort before U+FFFD-and-above BMP characters,
    // which a code-point order would not.
    assert!(compare_payload(String, Gt, b"ab", b"b").unwrap());
    assert!(compare_payload(String, Gte, b"b", b"b").unwrap());
    assert!(compare_payload(String, Lt, b"b", b"ab").unwrap());
    assert!(compare_payload(String, Lte, b"b", b"b").unwrap());
    assert!(compare_payload(String, Gt, b"z", "\u{e9}".as_bytes()).unwrap());
    assert!(compare_payload(String, Lt, "\u{ff5e}".as_bytes(), "\u{1f600}".as_bytes()).unwrap());
    // Malformed UTF-8 decodes to U+FFFD, as `new String(bytes, UTF_8)` does.
    assert!(compare_payload(String, Gt, b"z", &[0xff]).unwrap());
}

/// The two queries' `toString`s and the enums' names.
#[test]
fn the_queries_print_as_java_does() {
    let inner = SpanNode::term("f", "a");
    let check = SpanPayloadCheckQuery::with(
        inner.clone(),
        vec![Some(b"x".to_vec()), None],
        PayloadType::Int,
        MatchOperation::Gte,
    );
    assert_eq!(
        check.to_string(),
        "SpanPayloadCheckQuery(f:a, payloadRef: x;null;, payloadType:INT;, operation:GTE;)"
    );
    let plain = SpanPayloadCheckQuery::new(inner.clone(), vec![Some(vec![0xff, 0x01])]);
    assert_eq!(plain.payload_type, PayloadType::String);
    assert_eq!(plain.operation, MatchOperation::Eq);
    assert_eq!(
        plain.to_string(),
        "SpanPayloadCheckQuery(f:a, payloadRef: [ff 1];, payloadType:STRING;, operation:EQ;)"
    );
    let score = PayloadScoreQuery::new(
        inner,
        PayloadFunction::Average,
        PayloadDecoder::Float,
        false,
    );
    assert_eq!(
        score.to_string(),
        "PayloadScoreQuery(f:a, function: AveragePayloadFunction, includeSpanScore: false)"
    );
    let names: Vec<std::string::String> = [
        MatchOperation::Eq,
        MatchOperation::Gt,
        MatchOperation::Gte,
        MatchOperation::Lt,
        MatchOperation::Lte,
    ]
    .iter()
    .map(|o| o.to_string())
    .collect();
    assert_eq!(names, ["EQ", "GT", "GTE", "LT", "LTE"]);
    assert_eq!(PayloadType::Float.to_string(), "FLOAT");
}
