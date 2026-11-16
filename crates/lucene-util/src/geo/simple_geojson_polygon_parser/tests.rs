#![cfg(test)]
use super::*;

fn parse(s: &str) -> Result<Vec<Polygon>, GeoError> {
    SimpleGeoJSONPolygonParser::new(s).parse()
}

#[test]
fn polygons() {
    let p = parse(r#"{"type": "Polygon", "coordinates": [[[100.0, 0.0], [101.0, 0.0], [101.0, 1.0], [100.0, 1.0], [100.0, 0.0]]]}"#).unwrap();
    assert_eq!(p.len(), 1);
    assert_eq!(p[0].poly_lon(1), 101.0);
    let p = parse(r#"{"type": "Feature", "geometry": {"type": "MultiPolygon", "coordinates": [[[[0,0],[1,0],[1,1],[0,0]]], [[[5,5],[6,5],[6,6],[5,5]], [[5.1,5.1],[5.2,5.1],[5.2,5.2],[5.1,5.1]]]]}, "properties": {"a": "b\\c", "n": null, "t": true, "f": false, "x": 1e3}}"#).unwrap();
    assert_eq!(p.len(), 2);
    assert_eq!(p[1].num_holes(), 1);
    assert_eq!(
        Polygon::from_geojson(r#"{"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1],[0,0]]],}"#)
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn errors() {
    let e = |s: &str| parse(s).unwrap_err();
    assert!(
        matches!(e(r#"{"type": "Polygon"}"#), GeoError::Parse { ref message, .. } if message.starts_with("did not see any polygon coordinates"))
    );
    assert!(e(r#"{"coordinates": [[[0,0]]]}"#)
        .to_string()
        .starts_with("did not see type: Polygon or MultiPolygon"));
    assert!(matches!(
        e(r#"{"type": "Polygon", "coordinates": []}"#),
        GeoError::IndexOutOfBounds(_)
    ));
    assert!(matches!(
        e(r#"{"type": "Polygon", "coordinates": null}"#),
        GeoError::NullPointer(_)
    ));
    assert!(e(r#"{"type": "Polygon", "coordinates": 5}"#)
        .to_string()
        .starts_with("coordinates should be an array, but got: class java.lang.Double"));
    assert!(e(r#"{"type": 5}"#)
        .to_string()
        .starts_with("type should be a string, but got: 5.0"));
    assert!(e(r#"{"type": "Point"}"#)
        .to_string()
        .starts_with("can only handle type"));
    assert!(e(r#"{"a": "\uzzzz"}"#)
        .to_string()
        .starts_with("For input string: \"zzzz\""));
    assert!(e(r#"{"a": "\q"}"#)
        .to_string()
        .starts_with("unsupported string escape"));
    assert!(e(r#"{"a": tru}"#)
        .to_string()
        .starts_with("expected \"true\" but got"));
    assert!(e(r#"{"a": nul"#)
        .to_string()
        .starts_with("expected \"null\" but hit EOF"));
    assert!(e(r#"{"a": --}"#)
        .to_string()
        .starts_with("could not parse number as double"));
    assert!(e(r#"{"a": [1 2]}"#)
        .to_string()
        .starts_with("expected ',' separating list items"));
    assert!(e(r#"{"a": [1,"#)
        .to_string()
        .starts_with("hit EOF while parsing array"));
    assert!(e(r#"{"a": [1"#)
        .to_string()
        .starts_with("hit EOF while reading array"));
    assert!(e(r#"{"a": [x]}"#)
        .to_string()
        .starts_with("expected another array or number"));
    assert!(e(r#"{"a": x}"#)
        .to_string()
        .starts_with("expected array, object, string or literal value"));
    assert!(e(r#"{"a": 1 "b": 2}"#)
        .to_string()
        .starts_with("expected , but got \""));
    assert!(e(r#"{"a": 1} x"#)
        .to_string()
        .starts_with("unexpected character 'x'"));
    assert!(e(r#"{"crs": {"properties": {"href": "x"}}}"#)
        .to_string()
        .starts_with("cannot handle linked crs"));
    assert!(e(r#"{"crs": {"properties": {"name": 1}}}"#)
        .to_string()
        .starts_with("crs.properties.name should be a string"));
    assert!(e(r#"{"crs": {"properties": {"name": "x"}}}"#)
        .to_string()
        .starts_with("crs must be CRS84"));
    assert!(e(r#"{"type": "MultiPolygon", "coordinates": [5]}"#)
        .to_string()
        .starts_with("elements of coordinates array should be an array"));
    assert!(e(r#"{"type": "Polygon", "coordinates": [5]}"#)
        .to_string()
        .starts_with("first element of polygon array"));
    assert!(
        e(r#"{"type": "Polygon", "coordinates": [[[0,0],[1,0],[1,1],[0,0]], 5]}"#)
            .to_string()
            .starts_with("elements of coordinates array must be an array")
    );
    assert!(e(r#"{"type": "Polygon", "coordinates": [[5]]}"#)
        .to_string()
        .starts_with("elements of coordinates array must [lat, lon] array, but got: 5.0"));
    assert!(e(r#"{"type": "Polygon", "coordinates": [[[1]]]}"#)
        .to_string()
        .contains("wrong element count: [1.0]"));
    assert!(e(r#"{"type": "Polygon", "coordinates": [[["a", 1]]]}"#)
        .to_string()
        .contains("first element is not a Double: a"));
    assert!(e(r#"{"type": "Polygon", "coordinates": [[[1, [2]]]]}"#)
        .to_string()
        .contains("second element is not a Double: [2.0]"));
    assert!(e(r#"{"type": "Polygon", "type": "Polygon", "coordinates": [[[0,0],[1,0],[1,1],[0,0]]], "coordinates": [[[0,0],[1,0],[1,1],[0,0]]]}"#).to_string().starts_with("only one Polygon"));
    assert!(e("").to_string().starts_with("expected '{' but got EOF"));
    assert!(e("[").to_string().starts_with("expected '{' but got '['"));
    assert!(e("{\"a")
        .to_string()
        .starts_with("hit EOF inside string literal"));
    assert!(e("{\"a\\")
        .to_string()
        .starts_with("hit EOF inside string literal"));
    assert!(e("{\"a\\u12")
        .to_string()
        .starts_with("hit EOF inside string literal"));
    assert!(e("{\"a\": 1,").to_string().starts_with("unexpected EOF"));
    let long = format!("{{\"{}\": x}}", "k".repeat(60));
    assert!(e(&long).to_string().contains("\n..."));
}

#[test]
fn unicode_escape_quirk() {
    // `A` appends "65" and then re-reads "0041".
    let r = parse(
        r#"{"type": "Polygon", "coordinates": [[[0,0],[1,0],[1,1],[0,0]]], "p": {"x": "A"}}"#,
    );
    assert!(r.is_ok());
    assert_eq!(
        parse_int_16(&"-0ff".encode_utf16().collect::<Vec<_>>()).unwrap(),
        -255
    );
    assert_eq!(
        parse_int_16(&"+０Ａ".encode_utf16().collect::<Vec<_>>()).unwrap(),
        10
    );
    assert!(parse_int_16(&"-".encode_utf16().collect::<Vec<_>>()).is_err());
    assert_eq!(
        Value::Bool(true).java_class().unwrap(),
        "class java.lang.Boolean"
    );
    assert_eq!(
        Value::Str("s".into()).java_class().unwrap(),
        "class java.lang.String"
    );
    assert_eq!(
        Value::List(vec![Value::Null]).java_class().unwrap(),
        "class java.util.ArrayList"
    );
    assert_eq!(
        Value::List(vec![Value::Null, Value::Bool(false)]).java_string(),
        "[null, false]"
    );
}

/// Nesting past `MAX_NESTING` is a parse error, not a stack overflow
/// (which would abort the process).
#[test]
fn deep_nesting_is_an_error() {
    let deep = "[".repeat(1_000_000);
    let e = parse(&format!(r#"{{"type": "Polygon", "coordinates": {deep}"#)).unwrap_err();
    let GeoError::Parse { message, offset } = e else {
        panic!("{e:?}")
    };
    assert!(
        message.starts_with("java.lang.StackOverflowError"),
        "{message}"
    );
    // at the array opening level MAX_NESTING + 1 (the object is level 1)
    assert_eq!(offset as usize, 35 + MAX_NESTING as usize - 1);
    let objects = r#"{"a":"#.repeat(100_000);
    let e = parse(&objects).unwrap_err();
    assert!(e.to_string().starts_with("java.lang.StackOverflowError"));
    // just under the cap, the parser gets through and fails on content
    let ok_depth = MAX_NESTING as usize - 1;
    let s = format!(
        r#"{{"type": "Polygon", "coordinates": {}{}}}"#,
        "[".repeat(ok_depth),
        "]".repeat(ok_depth)
    );
    let e = parse(&s).unwrap_err();
    assert!(!e.to_string().contains("StackOverflow"), "{e}");
    // a parse that fails deep inside drops its partial values
    let s = format!(
        r#"{{"type": "Polygon", "coordinates": {}1.0,x"#,
        "[".repeat(ok_depth)
    );
    assert!(parse(&s).is_err());
}
