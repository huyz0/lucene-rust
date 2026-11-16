#![cfg(test)]
use super::*;

#[test]
fn iteration_follows_javas_buckets() {
    // Java: HashMap.newHashMap(4) = 8 buckets; "b" (98) and "j" (106)
    // share bucket 2, in insertion order; "a" is bucket 1.
    let map = JavaArgs::from_pairs(&[("j", "1"), ("a", "2"), ("b", "3")]);
    assert_eq!(map.to_string(), "{a=2, j=1, b=3}");
    assert_eq!(map.keys(), vec!["a", "j", "b"]);
    assert_eq!(JavaArgs::new().to_string(), "{}");
}

#[test]
fn resizing_keeps_javas_order() {
    // new HashMap<>() holds 12 entries in 16 buckets, then doubles.
    let mut map = JavaArgs::new();
    let keys: Vec<String> = (0..30).map(|i| format!("key{i}")).collect();
    for k in &keys {
        map.put(k, "v");
    }
    assert_eq!(map.len(), 30);
    assert_eq!(map.table.len(), 64);
    let order = map.keys();
    let mut sorted_by_bucket = keys.clone();
    sorted_by_bucket.sort_by_key(|k| spread(k) as usize & 63);
    assert_eq!(order, sorted_by_bucket);
    assert_eq!(map.put("key3", "w"), Some("v".to_string()));
    assert_eq!(map.put_if_absent("key3", "x"), Some("w".to_string()));
    assert_eq!(map.get("key3"), Some("w"));
    assert_eq!(map.remove("key3"), Some("w".to_string()));
    assert_eq!(map.remove("key3"), None);
    assert!(!map.contains_key("key3"));
    assert_eq!(JavaArgs::with_expected(0).initial_capacity, 1);
    let mut one = JavaArgs::with_expected(0);
    one.put("a", "1");
    one.put("b", "2");
    assert_eq!(one.len(), 2);
    assert!(JavaArgs::default().get("x").is_none());
}

#[test]
fn a_ninth_colliding_key_resizes_a_small_table() {
    // These nine keys share a bucket of 16 (and of 32 the low four bits
    // agree); Java's putVal resizes on the ninth (treeifyBin under 64
    // buckets), long before 0.75 of the table is used.
    let keys = [
        "k18", "k29", "k90", "k140", "k151", "k162", "k173", "k184", "k195",
    ];
    let pairs: Vec<(&str, &str)> = keys.iter().map(|k| (*k, "v")).collect();
    let map = JavaArgs::from_pairs(&pairs);
    assert_eq!(map.table.len(), 32);
    assert_eq!(
        map.to_string(),
        "{k90=v, k18=v, k29=v, k140=v, k151=v, k162=v, k173=v, k184=v, k195=v}"
    );
}

#[test]
fn string_hash_is_javas() {
    assert_eq!(java_string_hash(""), 0);
    assert_eq!(java_string_hash("ignoreCase"), 880_063_522);
    // A supplementary character hashes as its two surrogates.
    assert_eq!(java_string_hash("\u{1F600}"), 0xD83D * 31 + 0xDE00);
}

#[test]
fn integers_parse_like_integer_parse_int() {
    assert_eq!(parse_java_int("42").unwrap(), 42);
    assert_eq!(parse_java_int("+7").unwrap(), 7);
    assert_eq!(parse_java_int("-2147483648").unwrap(), i32::MIN);
    assert_eq!(parse_java_int("\u{0663}\u{0664}").unwrap(), 34);
    for bad in ["", "-", "+", "2147483648", "1.5", " 1", "x"] {
        let e = parse_java_int(bad).unwrap_err();
        assert_eq!(e.kind, JavaException::NumberFormat);
        assert_eq!(e.message, format!("For input string: \"{bad}\""));
    }
}

#[test]
fn floats_and_booleans() {
    assert_eq!(parse_java_float(" 1.5 ").unwrap(), 1.5);
    assert_eq!(parse_java_float("").unwrap_err().message, "empty String");
    assert_eq!(
        parse_java_float(" x ").unwrap_err().message,
        "For input string: \"x\""
    );
    // FloatingDecimal's own message for a second point in the leading run.
    for (s, want) in [
        ("4.0.0", "multiple points"),
        ("-..", "multiple points"),
        ("1x.0.0", "For input string: \"1x.0.0\""),
    ] {
        assert_eq!(parse_java_float(s).unwrap_err().message, want, "{s}");
    }
    assert!(parse_java_boolean("TRUE"));
    assert!(!parse_java_boolean("yes"));
    assert!(java_equals_ignore_case("Straße", "STRAßE"));
    assert!(!java_equals_ignore_case("a", "ab"));
}

#[test]
fn helpers_consume_their_keys() {
    let mut a = JavaArgs::from_pairs(&[
        ("n", "3"),
        ("b", "True"),
        ("f", "2.5"),
        ("c", "|"),
        ("cc", "ab"),
        ("s", " x, y ,x "),
        ("e", " , "),
        ("r", "first"),
        ("p", "a+"),
        ("bad", "("),
    ]);
    assert_eq!(require_int(&mut a, "n").unwrap(), 3);
    assert!(require_boolean(&mut a, "b").unwrap());
    assert_eq!(require_float(&mut a, "f").unwrap(), 2.5);
    assert_eq!(get_char(&mut a, "c", b'x' as u16).unwrap(), b'|' as u16);
    assert_eq!(
        get_char(&mut a, "cc", 0).unwrap_err().message,
        "cc should be a char. \"ab\" is invalid"
    );
    let set = get_set(&mut a, "s").unwrap();
    assert_eq!(set.into_iter().collect::<Vec<_>>(), vec!["x", "y"]);
    assert!(get_set(&mut a, "e").is_none());
    assert!(get_set(&mut a, "e").is_none());
    assert_eq!(
        get_one_of(&mut a, "r", &["all", "first"], None, true).unwrap(),
        Some("first".to_string())
    );
    assert_eq!(
        get_one_of(&mut a, "r", &["all"], Some("all"), true).unwrap(),
        Some("all".to_string())
    );
    assert!(get_pattern(&mut a, "p", "X").is_ok());
    assert_eq!(
        get_pattern(&mut a, "bad", "PatternTokenizerFactory")
            .unwrap_err()
            .message,
        "Configuration Error: 'bad' can not be parsed in PatternTokenizerFactory"
    );
    // Patterns Java compiles and the port refuses are not Java's error.
    for (p, why) in [
        ("a\\X", "grapheme clusters"),
        ("\\N{LATIN SMALL LETTER A}", "character names"),
        ("(?c)a", "CANON_EQ"),
    ] {
        let mut u = JavaArgs::from_pairs(&[("p", p)]);
        let e = get_pattern(&mut u, "p", "PatternTokenizerFactory").unwrap_err();
        assert_eq!(e.kind, JavaException::UnsupportedOperation, "{p}");
        assert!(e.message.contains(why), "{p}: {}", e.message);
    }
    assert!(a.is_empty());
    assert_eq!(
        require(&mut a, "n").unwrap_err().message,
        "Configuration Error: missing parameter 'n'"
    );
    assert_eq!(get_int(&mut a, "n", 9).unwrap(), 9);
    assert!(get_boolean(&mut a, "b", true));
    assert_eq!(get_float(&mut a, "f", 0.5).unwrap(), 0.5);
    assert_eq!(get_or(&mut a, "x", "d"), "d");
    assert!(get(&mut a, "x").is_none());
    assert_eq!(get_char(&mut a, "c", 7).unwrap(), 7);
    assert!(reject_unknown(&a).is_ok());
    a.put("x", "1");
    assert_eq!(
        reject_unknown(&a).unwrap_err().message,
        "Unknown parameters: {x=1}"
    );
    assert_eq!(get_int(&mut a, "x", 0).unwrap(), 1);
}

#[test]
fn allowed_values() {
    let mut a = JavaArgs::from_pairs(&[("r", "ALL"), ("q", "ALL"), ("z", "")]);
    assert_eq!(
        require_one_of(&mut a, "r", &["all", "first"], false).unwrap(),
        "ALL"
    );
    assert_eq!(
        require_one_of(&mut a, "q", &["all", "first"], true)
            .unwrap_err()
            .message,
        "Configuration Error: 'q' value must be one of [all, first]"
    );
    assert_eq!(
        require_char(&mut a, "z").unwrap_err().kind,
        JavaException::StringIndexOutOfBounds
    );
    a.put("z", "ab");
    assert_eq!(require_char(&mut a, "z").unwrap(), u16::from(b'a'));
    assert_eq!(unit_char(0x41), 'A');
    assert_eq!(unit_char(0xD800), char::REPLACEMENT_CHARACTER);
}

#[test]
fn split_at_matches_string_split() {
    assert_eq!(split_at(',', None), Vec::<String>::new());
    assert_eq!(split_at(',', Some("")), vec![""]);
    assert_eq!(split_at(',', Some("a,b")), vec!["a", "b"]);
    assert_eq!(split_at(',', Some("a,,")), vec!["a"]);
    assert_eq!(split_at(',', Some(",a")), vec!["", "a"]);
    assert_eq!(split_at(',', Some(",,")), Vec::<String>::new());
    assert_eq!(split_at(',', Some("a\\,b,c")), vec!["a,b", "c"]);
    assert_eq!(split_at('.', Some("f.k")), vec!["f", "k"]);
    assert_eq!(
        split_file_names(Some("x.txt, y.txt")),
        vec!["x.txt", " y.txt"]
    );
}
