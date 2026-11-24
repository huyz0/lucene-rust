//! A `date_histogram` rounding in an IANA region zone, against Java's
//! `JavaTime*Rounding` classes (`fixtures/src/GenRegionZoneRounding.java`):
//! every row is a date with the bucket Java gives it and the start of the next.

use lucene_aggs::bucket_aggs::{DateRounding, DateUnit, RegionZone, RoundingKind};

fn kind(text: &str) -> RoundingKind {
    if let Some(ms) = text.strip_prefix("fixed:") {
        return RoundingKind::Interval(ms.parse().unwrap());
    }
    RoundingKind::Unit(match text {
        "second" => DateUnit::Second,
        "minute" => DateUnit::Minute,
        "hour" => DateUnit::Hour,
        "day" => DateUnit::Day,
        "week" => DateUnit::Week,
        "month" => DateUnit::Month,
        "quarter" => DateUnit::Quarter,
        "year" => DateUnit::Year,
        other => panic!("unit {other}"),
    })
}

#[test]
fn region_zone_rounding_agrees_with_java() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/region_zone_rounding/rounding.txt"
    );
    let text = std::fs::read_to_string(path).unwrap();
    let mut rows = 0usize;
    let mut bad = Vec::new();
    for line in text.lines() {
        let f: Vec<&str> = line.split(' ').collect();
        let zone = RegionZone::named(f[0]).unwrap();
        let rounding = DateRounding::in_region(kind(f[1]), zone, f[2].parse().unwrap());
        let utc: i64 = f[3].parse().unwrap();
        let (round, next): (i64, i64) = (f[4].parse().unwrap(), f[5].parse().unwrap());
        rows += 1;
        let got = (rounding.round(utc), rounding.bucket(utc).map(|b| b.1));
        // Java's own next value is not past the date for a few St. John's days
        // (a 00:01 transition): no bucket range, so callers round each date.
        let expect = (round, (next > utc).then_some(next));
        if got != expect {
            bad.push(format!("{line} -> {got:?}"));
        }
    }
    assert!(rows > 30_000, "{rows} rows");
    assert!(
        bad.is_empty(),
        "{} of {rows} rows differ, first: {:#?}",
        bad.len(),
        &bad[..bad.len().min(12)]
    );
}
