//! The actual Java GeoTestUtil polygon oracle, including its raw-f64 boundaries.
use lucene_test_framework::geo::geo_test_util::contains_slowly;
use lucene_util::geo::Polygon;

fn coordinates(bits: &str) -> Vec<f64> {
    bits.split(',')
        .map(|value| f64::from_bits(value.parse().unwrap()))
        .collect()
}

#[test]
fn actual_java_contains_slowly_agrees_on_boundaries_precision_and_hole_errors() {
    let fixture = include_str!("../../../fixtures/testframework/data/geo_contains_slowly.tsv");
    let mut polygons: Vec<Polygon> = Vec::new();
    let mut queries = 0;
    let mut errors = 0;
    for line in fixture.lines().filter(|line| !line.starts_with('#')) {
        let fields: Vec<_> = line.split('\t').collect();
        assert_eq!(fields.len(), 6, "{line}");
        let id: usize = fields[1].parse().unwrap();
        match fields[0] {
            "P" => {
                assert_eq!(id, polygons.len(), "{line}");
                let holes = if fields[5] == "-" {
                    vec![]
                } else {
                    fields[5]
                        .split(',')
                        .map(|value| polygons[value.parse::<usize>().unwrap()].clone())
                        .collect()
                };
                polygons.push(
                    Polygon::new(&coordinates(fields[3]), &coordinates(fields[4]), holes).unwrap(),
                );
            }
            "Q" => {
                let lat = f64::from_bits(fields[3].parse().unwrap());
                let lon = f64::from_bits(fields[4].parse().unwrap());
                let actual = match contains_slowly(&polygons[id], lat, lon) {
                    Ok(value) => value.to_string(),
                    Err(message) => {
                        errors += 1;
                        format!("error:{message}")
                    }
                };
                assert_eq!(actual, fields[5], "{line}");
                queries += 1;
            }
            _ => panic!("unknown fixture row: {line}"),
        }
    }
    assert_eq!(polygons.len(), 16);
    assert_eq!(queries, 8009);
    assert_eq!(errors, 500);
}
