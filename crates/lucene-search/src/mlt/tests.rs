#![cfg(test)]
use super::*;

/// Java's `String.hashCode` and a `HashMap`'s iteration order, as
/// `new HashMap<String, ?>()` filled with these keys iterates them
/// (printed by Lucene's JDK for the same keys).
/// `Field.stringValue()` of a loaded stored field: numbers print as
/// `Integer`/`Long`/`Float`/`Double.toString`, binary is `null`.
#[test]
fn stored_numbers_read_as_their_java_strings() {
    use lucene_codecs::stored_fields::FieldValue;
    let s = |v| stored_string_value(&v);
    assert_eq!(s(FieldValue::String("a b".into())).as_deref(), Some("a b"));
    assert_eq!(s(FieldValue::Int(-42)).as_deref(), Some("-42"));
    assert_eq!(
        s(FieldValue::Long(1 << 40)).as_deref(),
        Some("1099511627776")
    );
    assert_eq!(s(FieldValue::Float(3.0)).as_deref(), Some("3.0"));
    assert_eq!(s(FieldValue::Float(1.0e-5)).as_deref(), Some("1.0E-5"));
    assert_eq!(s(FieldValue::Double(1.0e10)).as_deref(), Some("1.0E10"));
    assert_eq!(s(FieldValue::Double(0.1)).as_deref(), Some("0.1"));
    assert_eq!(s(FieldValue::Binary(vec![1])), None);
}

#[test]
fn hash_order_is_a_java_hash_maps() {
    assert_eq!(java_string_hash("apple"), 93029210);
    assert_eq!(java_string_hash(""), 0);
    assert_eq!(java_string_hash("Aa"), java_string_hash("BB"));
    let keys = ["red", "blue", "green", "fast", "slow"];
    let order: Vec<&str> = java_hash_order(&keys).iter().map(|&i| keys[i]).collect();
    // `new HashMap<>()` filled in `keys`' order, `keySet()` on JDK 25.
    assert_eq!(order, ["red", "green", "fast", "blue", "slow"]);
    // 14 keys: past the 12-entry threshold, at capacity 32.
    let greek = [
        "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
        "lambda", "mu", "nu", "xi",
    ];
    let order: Vec<&str> = java_hash_order(&greek).iter().map(|&i| greek[i]).collect();
    assert_eq!(
        order,
        [
            "zeta", "iota", "nu", "delta", "mu", "theta", "epsilon", "xi", "lambda", "eta",
            "alpha", "kappa", "beta", "gamma"
        ]
    );
    // Colliding keys keep their insertion order.
    let same = ["Aa", "BB"];
    assert_eq!(java_hash_order(&same), [0, 1]);
    let rev = ["BB", "Aa"];
    assert_eq!(java_hash_order(&rev), [0, 1]);
    // Past 12 entries the table doubles.
    let many: Vec<String> = (0..40).map(|i| format!("w{i}")).collect();
    let refs: Vec<&str> = many.iter().map(String::as_str).collect();
    let order = java_hash_order(&refs);
    let mut sorted = order.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, (0..40).collect::<Vec<_>>());
}

#[test]
fn the_queue_keeps_the_highest_scores_and_pops_lowest_first() {
    let mut q = FreqQueue::new(2);
    for (w, s) in [("a", 1.0), ("b", 3.0)] {
        q.add(ScoreTerm {
            word: w.into(),
            top_field: "f".into(),
            score: s,
        });
    }
    let top = q.top_mut().unwrap();
    assert_eq!(top.word, "a");
    *top = ScoreTerm {
        word: "c".into(),
        top_field: "f".into(),
        score: 2.0,
    };
    q.update_top();
    assert_eq!(MoreLikeThis::interesting(q, 1), ["c"]);
}
fn reader() -> DirectoryReader {
    let dir = std::path::Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/mlt/index"
    ));
    DirectoryReader::open(&lucene_store::FsDirectory::open(dir)).unwrap()
}

/// The reader's and a searcher's term statistics agree; a field list
/// left `None` is every indexed field; settings Java refuses are refused.
#[test]
fn statistics_settings_and_refusals() {
    let reader = reader();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let norms = vec![None; segments.len()];
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    for (field, term) in [
        ("body", "river"),
        ("tv", "stone"),
        ("body", "nosuch"),
        ("x", "y"),
    ] {
        assert_eq!(
            TermStatsReader::doc_freq(&reader, field, term.as_bytes()).unwrap(),
            TermStatsReader::doc_freq(&searcher, field, term.as_bytes()).unwrap(),
            "{field}:{term}"
        );
        assert_eq!(
            TermStatsReader::doc_count(&reader, field).unwrap(),
            TermStatsReader::doc_count(&searcher, field).unwrap()
        );
    }
    assert_eq!(
        TermStatsReader::max_doc(&reader),
        TermStatsReader::max_doc(&searcher)
    );

    let mut m = MoreLikeThis::new(&reader);
    m.field_names = None;
    m.min_term_freq = 1;
    m.min_doc_freq = 1;
    let analyzer = Analyzer::standard(None);
    m.analyzer = Some(&analyzer);
    let q = m.like_doc(3).unwrap();
    assert!(!q.should.is_empty());
    let fields = m.field_names.clone().unwrap();
    assert!(fields.contains(&"body".to_string()) && fields.contains(&"id".to_string()));
    assert!(m.set_max_doc_freq_pct(i32::MAX).is_err());
    assert!(m.like_doc(10_000).is_err(), "a document outside the reader");
    m.max_query_terms = 0;
    assert!(m.retrieve_interesting_terms(3).unwrap().is_empty());

    // Statistics alone: no documents, no field list to resolve.
    let mut over = MoreLikeThis::over(&searcher);
    over.field_names = None;
    assert!(over.like_doc(0).is_err());
    assert!(over.like_fields(&[("body", vec!["river".into()])]).is_err());
    over.field_names = Some(vec!["body".into()]);
    assert!(over.like_texts("body", &["river"]).is_err(), "no analyzer");
    over.analyzer = Some(&analyzer);
    over.min_doc_freq = 1;
    over.min_term_freq = 1;
    over.min_word_len = 5;
    over.max_word_len = 5;
    over.stop_words = Some(["stone".to_string()].into_iter().collect());
    let words = over
        .retrieve_interesting_terms_of_text("river stone light house river garden", "body")
        .unwrap();
    // Five letters exactly, and not a stop word.
    assert!(!words.is_empty());
    assert!(
        words.iter().all(|w| w.len() == 5 && w != "stone"),
        "{words:?}"
    );
    over.max_num_tokens_parsed = 1;
    assert_eq!(
        over.retrieve_interesting_terms_of_text("river light", "body")
            .unwrap(),
        ["river"]
    );
}

/// `MoreLikeThisQuery`: `toString`, equality (the analyzer by identity),
/// its rewrite's minimum and settings.
#[test]
fn the_query_rewrites_prints_and_compares() {
    let reader = reader();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let norms = vec![None; segments.len()];
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    let analyzer = std::sync::Arc::new(Analyzer::standard(None));
    let mut q = MoreLikeThisQuery::new(
        "river stone river light",
        vec!["body".into()],
        std::sync::Arc::clone(&analyzer),
        "body",
    );
    assert_eq!(q.to_string(), "like:river stone river light");
    assert_eq!(q, q.clone());
    let other = MoreLikeThisQuery::new(
        "river stone river light",
        vec!["body".into()],
        std::sync::Arc::new(Analyzer::standard(None)),
        "body",
    );
    assert_ne!(q, other, "another analyzer");
    assert!(format!("{q:?}").starts_with("MoreLikeThisQuery"));
    q.min_doc_freq = 1;
    q.percent_terms_to_match = 1.0;
    q.stop_words = Some(["light".to_string()].into_iter().collect());
    let Clause::Boolean(b) = q.rewrite(&searcher).unwrap() else {
        panic!("a boolean")
    };
    assert_eq!(b.minimum_should_match, b.should.len());
    assert!(b.should.len() >= 2);
    let clause = Clause::from(q);
    assert!(matches!(clause, Clause::Extended(_)));
}
