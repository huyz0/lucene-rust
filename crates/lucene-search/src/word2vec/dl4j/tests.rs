#![cfg(test)]
use super::*;

fn u(s: &str) -> Vec<u16> {
    s.encode_utf16().collect()
}

#[test]
fn base64_as_java_decodes_it() {
    assert_eq!(base64_decode(&u("aGVsbG8=")), Some(b"hello".to_vec()));
    assert_eq!(base64_decode(&u("aGVsbG8")), Some(b"hello".to_vec()));
    assert_eq!(base64_decode(&u("aGk=")).as_deref(), Some(&b"hi"[..]));
    assert_eq!(base64_decode(&u("aGk")).as_deref(), Some(&b"hi"[..]));
    assert_eq!(base64_decode(&u("")), Some(Vec::new()));
    assert_eq!(base64_decode(&u("a")), None);
    assert_eq!(base64_decode(&u("aGk==")), None);
    assert_eq!(base64_decode(&u("aGVs=")), None);
    assert_eq!(base64_decode(&u("aG=k")), None);
    assert_eq!(base64_decode(&u("a-b_")), None);
    assert_eq!(base64_decode(&u("+/+/")), Some(vec![0xFB, 0xFF, 0xBF]));
    assert_eq!(base64_decode(&[0x41, 0x100, 0x41, 0x41]), None);
}

#[test]
fn floats_and_ints_as_java_parses_them() {
    assert_eq!(java_parse_float(" 1.5 "), Some(1.5));
    assert_eq!(java_parse_float("-2e3f"), Some(-2000.0));
    assert_eq!(java_parse_float(".5D"), Some(0.5));
    assert_eq!(java_parse_float("5."), Some(5.0));
    assert_eq!(java_parse_float("1E+2"), Some(100.0));
    assert!(java_parse_float("NaN").unwrap().is_nan());
    assert_eq!(java_parse_float("-Infinity"), Some(f32::NEG_INFINITY));
    for bad in [
        "", ".", "1..2", "inf", "nan", "1e", "1e+", "0x1p3", "1f2", "--1", "1 2",
    ] {
        assert_eq!(java_parse_float(bad), None, "{bad}");
    }
    assert_eq!(java_parse_int("-12"), Some(-12));
    assert_eq!(java_parse_int("+7"), Some(7));
    for bad in ["", "-", "1.0", "99999999999", " 1"] {
        assert_eq!(java_parse_int(bad), None, "{bad}");
    }
}

/// The model of a whole text, fed in one chunk.
fn parse_model(text: &str) -> Result<Word2VecModel> {
    let mut p = ModelParser::default();
    p.feed(text.as_bytes())?;
    p.finish()
}

#[test]
fn lines_and_fields_as_java_splits_them() {
    // `\n`, `\r\n` and `\r` end lines, however the bytes are chunked.
    let text = "3 1\naaa 1\r\nbbb 2\rccc 3";
    for chunk in [1, 2, 3, 100] {
        let mut p = ModelParser::default();
        for c in text.as_bytes().chunks(chunk) {
            p.feed(c).unwrap();
        }
        let m = p.finish().unwrap();
        assert_eq!(
            (m.term(0), m.term(1), m.term(2)),
            (&b"aaa"[..], &b"bbb"[..], &b"ccc"[..])
        );
    }
    // A final line break ends the last line; one more is an empty line.
    assert_eq!(parse_model("1 1\naaa 1\n").unwrap().loaded(), 1);
    assert!(parse_model("2 1\naaa 1\n\n")
        .unwrap_err()
        .to_string()
        .contains("found vector of size 0"));
    assert_eq!(split_space("a b  "), ["a", "b"]);
    assert_eq!(split_space(" a"), ["", "a"]);
    assert!(split_space("").is_empty());
}

#[test]
fn malformed_models_are_refused() {
    let err = |t: &str| parse_model(t).unwrap_err().to_string();
    assert!(err("x 2\n").contains("\"x\""));
    assert!(err("1\n").contains("\"\""));
    assert!(err("1 2\nab 1 2\n").contains("begin 0, end 3, length 2"));
    assert!(err("1 2\nB64 1 2\n").contains("begin 4, end 3"));
    assert!(err("1 2\nB64:!! 1 2\n").contains("Illegal base64"));
    assert!(err("1 2\nabc 1\n").contains("Declared vectors of size 2 but found vector of size 1"));
    assert!(err("1 2\nabc 1 x\n").contains("\"x\""));
    let m = parse_model("2 2\nb64:YQ== 1 0\nB64:Yg 0 1\n").unwrap();
    assert_eq!((m.term(0), m.term(1)), (&b"a"[..], &b"b"[..]));
    let m = parse_model("1 1\nword 2\n").unwrap();
    assert_eq!((m.term(0), m.vector(0)), (&b"word"[..], &[1.0f32][..]));
    assert!(err("").contains("\"\""));
}

#[test]
fn over_long_lines_are_refused() {
    let err = |t: &[u8]| {
        let mut p = ModelParser::default();
        p.feed(t)
            .and_then(|()| p.finish().map(|_| ()))
            .unwrap_err()
            .to_string()
    };
    // The header: 1 MiB.
    assert!(err(&vec![b'1'; MAX_LINE + 1]).contains("over 1048576 bytes"));
    // A term line: 1 MiB and 64 bytes a dimension, however it is fed.
    let mut p = ModelParser::default();
    p.feed(b"1 2\nxyz ").unwrap();
    let fill = vec![b'1'; MAX_LINE + 2 * MAX_LINE_PER_DIMENSION - 4];
    p.feed(&fill).unwrap();
    let e = p.feed(b"1").unwrap_err().to_string();
    assert!(e.contains("over 1048704 bytes"), "{e}");
    // At the limit, the line is read (and here refused for its float).
    let mut p = ModelParser::default();
    p.feed(b"1 2\nxyz ").unwrap();
    p.feed(&fill).unwrap();
    assert!(p
        .finish()
        .unwrap_err()
        .to_string()
        .contains("Declared vectors"));
}

#[test]
fn zips_without_a_model_or_malformed() {
    let missing = read_dl4j_model(&[]).unwrap_err().to_string();
    assert!(missing.contains("'syn0' file is missing"), "{missing}");
    // A stored entry named `other`, then the end.
    let mut z = vec![0x50, 0x4B, 3, 4, 20, 0, 0, 0, 0, 0];
    z.extend_from_slice(&[0; 8]);
    z.extend_from_slice(&1u32.to_le_bytes());
    z.extend_from_slice(&1u32.to_le_bytes());
    z.extend_from_slice(&5u16.to_le_bytes());
    z.extend_from_slice(&0u16.to_le_bytes());
    z.extend_from_slice(b"otherX");
    assert!(read_dl4j_model(&z)
        .unwrap_err()
        .to_string()
        .contains("missing"));
    let mut lzma = z.clone();
    lzma[8] = 14;
    assert!(read_dl4j_model(&lzma)
        .unwrap_err()
        .to_string()
        .contains("method 14"));
    let mut deflate = z.clone();
    deflate[8] = 8;
    assert!(read_dl4j_model(&deflate)
        .unwrap_err()
        .to_string()
        .contains("invalid deflated"));
    assert!(read_dl4j_model(&z[..29])
        .unwrap_err()
        .to_string()
        .contains("truncated"));
    let mut long = z.clone();
    long[18] = 9;
    assert!(read_dl4j_model(&long)
        .unwrap_err()
        .to_string()
        .contains("truncated"));
    let mut name = z.clone();
    name[26] = 200;
    assert!(read_dl4j_model(&name)
        .unwrap_err()
        .to_string()
        .contains("truncated"));
}
