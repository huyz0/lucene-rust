//! Differential tests: verify Java-written codec header/footer framing.
//! Regenerate with fixtures/src/GenCodecUtil.java (see that file's doc comment).

use lucene_store::codec_util::{self, ID_LENGTH};
use lucene_store::data_input::SliceInput;

const CODEC: &str = "LuceneRustFixture";
const VERSION: i32 = 3;
const SUFFIX: &str = "seg1";

fn fixture_dir() -> String {
    concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/data/").to_string()
}

fn read_id_hex() -> [u8; ID_LENGTH] {
    let manifest = std::fs::read_to_string(format!("{}manifest.properties", fixture_dir()))
        .expect("run fixtures generator first (GenCodecUtil)");
    let hex = manifest
        .lines()
        .find_map(|l| l.strip_prefix("id_hex="))
        .expect("id_hex missing from manifest");
    let mut id = [0u8; ID_LENGTH];
    for i in 0..ID_LENGTH {
        id[i] = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap();
    }
    id
}

#[test]
fn plain_header_and_footer_verify() {
    let buf = std::fs::read(format!("{}plain.bin", fixture_dir())).unwrap();
    let mut input = SliceInput::new(&buf);
    let header = codec_util::check_header(&mut input, CODEC, VERSION, VERSION).unwrap();
    assert_eq!(header.version, VERSION);

    let payload_start = input.position();
    let payload_end = buf.len() - codec_util::FOOTER_LENGTH;
    let payload = input.slice(payload_start, payload_end).unwrap();
    assert_eq!(
        std::str::from_utf8(payload).unwrap(),
        "hello lucene-rust codec header/footer fixture payload"
    );

    input.seek(payload_end).unwrap();
    codec_util::check_footer(&mut input, buf.len()).unwrap();
}

#[test]
fn index_header_and_footer_verify() {
    let buf = std::fs::read(format!("{}indexed.bin", fixture_dir())).unwrap();
    let expected_id = read_id_hex();
    let mut input = SliceInput::new(&buf);
    let header =
        codec_util::check_index_header(&mut input, CODEC, VERSION, VERSION, &expected_id, SUFFIX)
            .unwrap();
    assert_eq!(header.version, VERSION);
    assert_eq!(header.id, expected_id);
    assert_eq!(header.suffix, SUFFIX);

    let payload_end = buf.len() - codec_util::FOOTER_LENGTH;
    input.seek(payload_end).unwrap();
    codec_util::check_footer(&mut input, buf.len()).unwrap();
}

#[test]
fn wrong_codec_name_rejected() {
    let buf = std::fs::read(format!("{}plain.bin", fixture_dir())).unwrap();
    let mut input = SliceInput::new(&buf);
    let err = codec_util::check_header(&mut input, "WrongCodecName", VERSION, VERSION).unwrap_err();
    assert!(matches!(err, lucene_store::Error::Corrupted(_)));
}

#[test]
fn version_out_of_range_rejected() {
    let buf = std::fs::read(format!("{}plain.bin", fixture_dir())).unwrap();
    let mut input = SliceInput::new(&buf);
    assert!(codec_util::check_header(&mut input, CODEC, VERSION + 1, VERSION + 5).is_err());
    let mut input = SliceInput::new(&buf);
    assert!(codec_util::check_header(&mut input, CODEC, 0, VERSION - 1).is_err());
}

#[test]
fn corrupted_payload_fails_checksum() {
    let buf = std::fs::read(format!("{}corrupt_checksum.bin", fixture_dir())).unwrap();
    let mut input = SliceInput::new(&buf);
    codec_util::check_header(&mut input, CODEC, VERSION, VERSION).unwrap();
    let payload_end = buf.len() - codec_util::FOOTER_LENGTH;
    input.seek(payload_end).unwrap();
    let err = codec_util::check_footer(&mut input, buf.len()).unwrap_err();
    assert!(matches!(err, lucene_store::Error::Corrupted(_)));
}

#[test]
fn whole_file_convenience_helpers() {
    let buf = std::fs::read(format!("{}plain.bin", fixture_dir())).unwrap();
    let header = codec_util::check_whole_file_header(&buf, CODEC, VERSION, VERSION).unwrap();
    assert_eq!(header.version, VERSION);
    let payload_end = buf.len() - codec_util::FOOTER_LENGTH;
    codec_util::check_whole_file_footer(&buf, payload_end).unwrap();
}
