//! Resource bytes produced by Lucene 10.5.0's StringMockResourceLoader.

use lucene_analysis::factory::ResourceLoader;
use lucene_test_framework::util::string_mock_resource_loader::StringMockResourceLoader;

fn units(hex: &str) -> String {
    if hex == "-" {
        return String::new();
    }
    let units: Vec<u16> = (0..hex.len())
        .step_by(4)
        .map(|i| u16::from_str_radix(&hex[i..i + 4], 16).unwrap())
        .collect();
    String::from_utf16(&units).unwrap()
}

fn bytes(hex: &str) -> Vec<u8> {
    if hex == "-" {
        return Vec::new();
    }
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
        .collect()
}

#[test]
fn string_mock_resource_loader_matches_java() {
    let fixture = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/testframework/data/string_mock_resource_loader.tsv"
    ));
    let mut cases = 0;
    for line in fixture.lines() {
        let fields: Vec<_> = line.split('\t').collect();
        assert_eq!(fields.len(), 3);
        let loader = StringMockResourceLoader::new(units(fields[0]));
        let name = units(fields[1]);
        let mut first = loader.open_resource(&name).unwrap();
        let expected = bytes(fields[2]);
        assert_eq!(first, expected);
        first.clear();
        assert_eq!(loader.open_resource(&name).unwrap(), expected);
        cases += 1;
    }
    assert_eq!(cases, 28);
}
