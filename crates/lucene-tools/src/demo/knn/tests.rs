#![cfg(test)]
use super::*;
use lucene_store::directory::FsDirectory;
use lucene_util::test_support::TempDir;

fn dict(text: &str) -> (TempDir, Result<KnnVectorDict, Error>) {
    let tmp = TempDir::new("tools-knn-dict");
    let glove = tmp.path().join("glove.txt");
    std::fs::write(&glove, text).unwrap();
    let dir = FsDirectory::open(tmp.path().join("dict"));
    std::fs::create_dir_all(tmp.path().join("dict")).unwrap();
    let built =
        KnnVectorDict::build(&glove, &dir, "d").and_then(|()| KnnVectorDict::open(&dir, "d"));
    (tmp, built)
}

#[test]
fn the_first_line_is_written_but_not_looked_up() {
    let (_tmp, d) = dict("first 1 0\nalpha 0 2\r\nbeta 3 4 \n");
    let d = d.unwrap();
    assert_eq!(d.dimension(), 2);
    let mut out = vec![0u8; 8];
    d.get(b"alpha", &mut out).unwrap();
    assert_eq!(
        out,
        [0.0f32, 1.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect::<Vec<_>>()
    );
    d.get(b"first", &mut out).unwrap();
    assert_eq!(out, vec![0; 8]);
    d.get(b"beta", &mut out).unwrap();
    assert_eq!(
        out,
        [0.6f32, 0.8]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect::<Vec<_>>()
    );
    assert!(d.ram_bytes_used() > 3 * 8);
    let e = d.get(b"beta", &mut [0u8; 4]).unwrap_err().to_string();
    assert_eq!(
        e,
        "java.lang.IllegalArgumentException: the output array must be of length 8, got 4"
    );
}

#[test]
fn javas_build_errors() {
    let err = |text: &str| dict(text).1.unwrap_err().to_string();
    assert_eq!(
        err("a 1 0\nb 1\n"),
        "java.lang.IllegalStateException: different field count at line 1 got 2 when expecting 3"
    );
    assert_eq!(
        err("a 1 0\nb 1 x\n"),
        "java.lang.NumberFormatException: For input string: \"x\""
    );
    assert_eq!(
        err("a 1 0\nb 0 0\n"),
        "java.lang.IllegalArgumentException: Cannot normalize a zero-length vector"
    );
    assert!(err("a 1 0\nc 1 0\nb 1 0\n").starts_with("java.lang.IllegalArgumentException"));
    assert_eq!(err("a 1 0\n"), "java.lang.NullPointerException");
    assert_eq!(err(""), "java.lang.NullPointerException");
    assert!(err("a \u{0} 1\n").starts_with("java.lang.NumberFormatException"));
    // not UTF-8: an `IOException`, which `IndexFiles` catches
    let tmp = TempDir::new("tools-knn-utf8");
    let glove = tmp.path().join("glove.txt");
    std::fs::write(&glove, b"a 1 0\n\xff 1 0\n").unwrap();
    let dir = FsDirectory::open(tmp.path());
    let e = KnnVectorDict::build(&glove, &dir, "d").unwrap_err();
    assert!(e.is::<std::io::Error>());
    assert_eq!(e.to_string(), "Input length = 1");
}

#[test]
fn outputs_are_created_new_before_the_text_is_read() {
    let tmp = TempDir::new("tools-knn-create-new");
    let glove = tmp.path().join("glove.txt");
    std::fs::write(&glove, "a 1 0\nb 1\n").unwrap();
    let dir = FsDirectory::open(tmp.path().join("dict"));
    std::fs::create_dir_all(tmp.path().join("dict")).unwrap();
    // a failed build leaves both files behind, empty
    assert!(KnnVectorDict::build(&glove, &dir, "d").is_err());
    for ext in ["bin", "fst"] {
        assert_eq!(dir.file_length(&format!("d.{ext}")).unwrap(), 0, "{ext}");
    }
    // and the next build over them fails on the first, `.bin`
    let e = KnnVectorDict::build(&glove, &dir, "d").unwrap_err();
    assert_eq!(e.downcast_ref::<FileAlreadyExists>().unwrap().0, "d.bin");
    // only `.fst` there: `.bin` is created, then `.fst` fails
    dir.delete_file("d.bin").unwrap();
    let e = KnnVectorDict::build(&glove, &dir, "d").unwrap_err();
    assert_eq!(e.to_string(), "d.fst");
    assert_eq!(dir.file_length("d.bin").unwrap(), 0);
}

#[test]
fn floats_parse_as_java_parses_them() {
    assert_eq!(
        parse_float(" x	").unwrap_err().to_string(),
        "java.lang.NumberFormatException: For input string: \"x\""
    );
    assert_eq!(parse_float(" 1.5f ").unwrap(), 1.5);
    assert_eq!(parse_float("2e1").unwrap(), 20.0);
    assert_eq!(parse_float("3D").unwrap(), 3.0);
    assert!(parse_float("NaN").unwrap().is_nan());
    assert_eq!(parse_float("-Infinity").unwrap(), f32::NEG_INFINITY);
    assert_eq!(parse_float("+Infinity").unwrap(), f32::INFINITY);
    assert!(parse_float("inf").is_err());
    assert!(parse_float("1e").is_err());
    assert_eq!(
        parse_float("").unwrap_err().to_string(),
        "java.lang.NumberFormatException: empty String"
    );
}

#[test]
fn a_damaged_bin_is_refused() {
    let tmp = TempDir::new("tools-knn-bad");
    let dir = FsDirectory::open(tmp.path());
    let (_t, d) = dict("first 1 0\nalpha 0 2\n");
    drop(d);
    let glove = tmp.path().join("g.txt");
    std::fs::write(&glove, "first 1 0\nalpha 0 2\n").unwrap();
    KnnVectorDict::build(&glove, &dir, "d").unwrap();
    std::fs::write(tmp.path().join("d.bin"), [0u8, 0, 0, 0, 0, 3, 0, 0, 0]).unwrap();
    assert_eq!(
        KnnVectorDict::open(&dir, "d").unwrap_err().to_string(),
        "java.lang.IllegalStateException: vector file size 9 is not consonant with the \
             vector dimension 3"
    );
    std::fs::write(tmp.path().join("d.bin"), [0u8, 0]).unwrap();
    assert!(KnnVectorDict::open(&dir, "d")
        .unwrap_err()
        .to_string()
        .starts_with("java.io.EOFException"));
    // an ordinal past the vectors
    std::fs::write(tmp.path().join("d.bin"), [2u8, 0, 0, 0]).unwrap();
    let d = KnnVectorDict::open(&dir, "d").unwrap();
    assert!(d.get(b"alpha", &mut [0u8; 8]).is_err());
}

#[test]
fn an_embedding_sums_and_normalizes_the_known_words() {
    let (_tmp, d) = dict("first 1 0\nalpha 0 2\nbeta 3 4\n");
    let d = d.unwrap();
    let e = DemoEmbeddings::new(&d);
    assert_eq!(
        e.compute_embedding("ALPHA unknown").unwrap(),
        vec![0.0, 1.0]
    );
    assert_eq!(
        e.compute_embedding("nothing known").unwrap(),
        vec![0.0, 0.0]
    );
    let both = e.compute_embedding("alpha beta").unwrap();
    assert!((both[0] * both[0] + both[1] * both[1] - 1.0).abs() < 1e-6);
}

#[test]
fn lines_split_as_javas_reader_splits_them() {
    let lines: Vec<&str> = java_lines("a\rb\r\nc\n\nd").collect();
    assert_eq!(lines, ["a", "b", "c", "", "d"]);
    assert_eq!(split_fields("a b  "), ["a", "b"]);
    assert_eq!(split_fields(""), [""]);
    let dict = KnnVectorDict {
        fst: TypedFst::read(
            &TypedFst::<PositiveIntOutputs>::from_compiled({
                let mut c = FstCompilerBuilder::new(InputType::Byte1).build::<PositiveIntOutputs>();
                c.add(&[97], 1).unwrap();
                c.compile().unwrap()
            })
            .save(),
        )
        .unwrap(),
        fst_len: 0,
        vectors: Vec::new(),
        dimension: 1,
    };
    assert!(format!("{dict:?}").contains("dimension: 1"));
}
