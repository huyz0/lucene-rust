#![cfg(test)]
use super::*;

const ID: [u8; 16] = *b"segment-info-ids";

fn info() -> SimpleTextSegmentInfo {
    SimpleTextSegmentInfo {
        name: "_4".into(),
        version: Version::LATEST,
        min_version: None,
        doc_count: 3,
        is_compound_file: true,
        has_blocks: false,
        diagnostics: vec![("os".into(), "a\nb".into()), ("x".into(), String::new())],
        attributes: vec![("k".into(), "v".into())],
        files: vec!["_4.scf".into()],
        id: ID,
        index_sort: vec![SortFieldEntry {
            provider: "SortField".into(),
            description: "<long: \"n\">!".into(),
            bytes: vec![0, 0x0a, 0xff],
        }],
    }
}

fn text(si: &mut SimpleTextSegmentInfo) -> String {
    String::from_utf8(write(si).unwrap()).unwrap()
}

fn reseal(body: &str) -> Vec<u8> {
    let mut out = body.as_bytes().to_vec();
    util::write_checksum(&mut out);
    out
}

fn body(full: &str) -> String {
    full.rsplit_once("checksum ").unwrap().0.to_string()
}

#[test]
fn round_trip_adds_the_si_file_once() {
    let mut si = info();
    let bytes = write(&mut si).unwrap();
    assert_eq!(si.files, ["_4.scf", "_4.si"]);
    let back = read(&bytes, "_4", &ID).unwrap();
    assert_eq!(back, si);
    let mut again = back.clone();
    assert_eq!(write(&mut again).unwrap(), bytes);
    let t = String::from_utf8(bytes).unwrap();
    assert!(t.contains("    min version null\n"));
    assert!(t.contains("    id [73 65 67"), "{t}");
    assert!(t.contains("      bytes [0 a ff]\n"), "{t}");
    assert_eq!(segment_info_file_name("_4"), "_4.si");
}

#[test]
fn repeated_keys_and_files_collapse() {
    let full = text(&mut info());
    let b = body(&full)
        .replace(
            "    diagnostics 2\n",
            "    diagnostics 3\n      key os\n      value z\n",
        )
        .replace("    files 2\n", "    files 3\n      file _4.si\n");
    let si = read(&reseal(&b), "_4", &ID).unwrap();
    assert_eq!(si.diagnostics[0], ("os".to_string(), "a\nb".to_string()));
    assert_eq!(si.diagnostics.len(), 2);
    assert_eq!(si.files, ["_4.si", "_4.scf"]);
}

#[test]
fn min_version_and_bad_versions() {
    let mut si = info();
    si.min_version = Some(Version::parse("9.12.1").unwrap());
    let full = text(&mut si);
    assert!(full.contains("    min version 9.12.1\n"));
    assert_eq!(
        read(full.as_bytes(), "_4", &ID).unwrap().min_version,
        si.min_version
    );
    let b = body(&full);
    for (from, to) in [
        ("    version 10.5.2", "    version 10.x"),
        ("    min version 9.12.1", "    min version 9"),
        ("    version 10.5.2", "    version 256.0.0"),
    ] {
        let err = read(&reseal(&b.replacen(from, to, 1)), "_4", &ID).unwrap_err();
        assert!(
            err.to_string().contains("unable to parse version string"),
            "{err}"
        );
    }
}

#[test]
fn damaged_files_are_refused() {
    let full = text(&mut info());
    let b = body(&full);
    let cases = [
        ("    number of documents 3", "    number of documents x"),
        ("    diagnostics 2", "    diagnostics -1"),
        ("    attributes 1", "    attributes -1"),
        ("    files 2", "    files -1"),
        ("    sort 1", "    sort -1"),
        ("    uses compound", "    used compound"),
        ("      name SortField", "      name NoSuchSort"),
        ("      bytes [0 a ff]", "      bytes [0 a fg]"),
        ("      bytes [0 a ff]", "      bytes 0 a ff"),
        ("      file _4.scf", "      file 4.scf"),
        ("    id [73", "    id [74"),
    ];
    for (from, to) in cases {
        assert!(b.contains(from), "{from}");
        assert!(
            read(&reseal(&b.replacen(from, to, 1)), "_4", &ID).is_err(),
            "{to}"
        );
    }
    // The id must be the one segments_N names, and 16 bytes long.
    assert!(read(full.as_bytes(), "_4", b"other-segment-id")
        .unwrap_err()
        .to_string()
        .contains("file mismatch, expected: "));
    let short = b.replacen("    id [", "    id [1 ", 1);
    assert!(read(&reseal(&short), "_4", &ID).is_err());
    let mut long_id = ID.to_vec();
    long_id.push(1);
    let long = b.replacen("]\n    sort", " 1]\n    sort", 1);
    assert!(matches!(
        read(&reseal(&long), "_4", &long_id),
        Err(Error::IllegalArgument(_))
    ));
    // A flipped checksum digit; truncation.
    let mut flipped = full.clone().into_bytes();
    let last = flipped.len() - 2;
    flipped[last] = if flipped[last] == b'0' { b'1' } else { b'0' };
    assert!(read(&flipped, "_4", &ID).is_err());
    assert!(read(&full.as_bytes()[..full.len() / 2], "_4", &ID).is_err());
}

#[test]
fn codec_file_pattern() {
    for ok in ["_0.si", "_a1_Lucene_0.doc", "_z.", "_0_.x"] {
        assert!(matches_codec_file_pattern(ok), "{ok}");
    }
    for bad in ["0.si", "_.si", "_A.si", "_0", "_0_x", "_0-x.y", ""] {
        assert!(!matches_codec_file_pattern(bad), "{bad}");
    }
    // Java's `.` matches no line terminator, in either `.*`.
    for t in ['\n', '\r', '\u{85}', '\u{2028}', '\u{2029}'] {
        assert!(!matches_codec_file_pattern(&format!("_0.s{t}i")), "{t:?}");
        assert!(!matches_codec_file_pattern(&format!("_0_{t}.si")), "{t:?}");
    }
    assert!(matches_codec_file_pattern("_0_\u{e9}.s\u{2027}"));
    // An id that is not 16 bytes, refused after the footer as Java's
    // `new SegmentInfo` does, with `Arrays.toString`'s signed decimals.
    let full = text(&mut info());
    let short = body(&full).replace("    id [73 65 67", "    id [ff 65 67");
    let short = short.replacen(" 73]\n", "]\n", 1);
    let id = from_bytes_ref_string(
        short
            .lines()
            .find_map(|l| l.strip_prefix("    id "))
            .unwrap(),
    )
    .unwrap();
    assert_eq!(id.len(), 15);
    let err = read(&reseal(&short), "_4", &id).unwrap_err().to_string();
    assert!(err.starts_with("invalid id: [-1, 101, 103, "), "{err}");
    assert!(read(&reseal(&short)[..10], "_4", &id).is_err());
    let mut si = info();
    si.files.push("bad".into());
    assert!(matches!(write(&mut si), Err(Error::IllegalArgument(_))));
    let mut si = info();
    si.name = "x".into();
    assert!(matches!(write(&mut si), Err(Error::IllegalArgument(_))));
}
