#![cfg(test)]
use super::*;
use crate::api::test_support::{call, t, text, vm_call};

fn st(class: &str, name: &str, args: &[Value]) -> String {
    match static_lookup(class, name, args.len()) {
        Some(f) => text(vm_call(f, args)),
        None => format!("no static {class}.{name}/{}", args.len()),
    }
}

fn l(x: i64) -> Value {
    Value::Long(x)
}

fn i(x: i32) -> Value {
    Value::Int(x)
}

fn field(class: &str, name: &str) -> Value {
    static_field(class, name).expect("a constant")
}

#[test]
fn durations_from_every_factory() {
    let cases: Vec<(&str, Vec<Value>, &str)> = vec![
        ("ofDays", vec![l(2)], "PT48H"),
        ("ofHours", vec![l(25)], "PT25H"),
        ("ofMinutes", vec![l(90)], "PT1H30M"),
        ("ofSeconds", vec![l(3725)], "PT1H2M5S"),
        ("ofSeconds", vec![l(1), l(500_000_000)], "PT1.5S"),
        ("ofMillis", vec![l(1500)], "PT1.5S"),
        ("ofNanos", vec![l(1)], "PT0.000000001S"),
    ];
    for (name, args, want) in cases {
        assert_eq!(st(DURATION, name, &args), want, "Duration.{name}");
    }
    assert_eq!(text(Ok(field(DURATION, "ZERO"))), "PT0S");
    assert_eq!(
        st(DURATION, "of", &[l(3), field(CHRONO_UNIT, "MINUTES")]),
        "PT3M"
    );
    let a = vm_call(static_lookup(DURATION, "ofSeconds", 1).unwrap(), &[l(10)]).unwrap();
    assert_eq!(st(DURATION, "from", std::slice::from_ref(&a)), "PT10S");
}

#[test]
fn instants_and_zoned_values_from_factories() {
    assert_eq!(
        st(INSTANT, "ofEpochMilli", &[l(1500)]),
        "1970-01-01T00:00:01.500Z"
    );
    assert_eq!(
        st(INSTANT, "ofEpochMilli", &[l(-1)]),
        "1969-12-31T23:59:59.999Z"
    );
    assert_eq!(
        st(INSTANT, "ofEpochSecond", &[l(60)]),
        "1970-01-01T00:01:00Z"
    );
    assert_eq!(
        st(INSTANT, "ofEpochSecond", &[l(0), l(1_500_000_000)]),
        "1970-01-01T00:00:01.500Z"
    );
    assert_eq!(text(Ok(field(INSTANT, "EPOCH"))), "1970-01-01T00:00:00Z");
    assert!(text(Ok(field(INSTANT, "MIN"))).starts_with("-1000000000-01-01"));
    assert!(text(Ok(field(INSTANT, "MAX"))).starts_with("+1000000000-12-31"));
    assert!(static_field(INSTANT, "NOPE").is_none());
    let epoch = field(INSTANT, "EPOCH");
    assert_eq!(
        st(INSTANT, "from", std::slice::from_ref(&epoch)),
        "1970-01-01T00:00:00Z"
    );
    let utc = field(ZONE_OFFSET, "UTC");
    assert_eq!(
        st(ZONED, "ofInstant", &[epoch.clone(), utc.clone()]),
        "1970-01-01T00:00Z"
    );
    let z = vm_call(
        static_lookup(ZONED, "of", 8).unwrap(),
        &[i(2024), i(2), i(29), i(13), i(45), i(30), i(5), utc.clone()],
    )
    .unwrap();
    assert_eq!(text(Ok(z.clone())), "2024-02-29T13:45:30.000000005Z");
    assert_eq!(
        st(ZONED, "from", std::slice::from_ref(&z)),
        "2024-02-29T13:45:30.000000005Z"
    );
    assert!(st(
        ZONED,
        "of",
        &[i(2023), i(2), i(29), i(0), i(0), i(0), i(0), utc.clone()]
    )
    .contains("DateTimeException"));
    assert!(st(
        ZONED,
        "of",
        &[i(2023), i(1), i(1), i(24), i(0), i(0), i(0), utc]
    )
    .contains("DateTimeException"));
    assert_eq!(
        st(
            DURATION,
            "between",
            &[epoch.clone(), field(INSTANT, "EPOCH")]
        ),
        "PT0S"
    );
    assert!(st(DURATION, "between", &[i(1), i(2)]).contains("Temporal"));
}

#[test]
fn offsets() {
    let cases: Vec<(&str, Vec<Value>, &str)> = vec![
        ("ofHours", vec![i(5)], "+05:00"),
        ("ofHours", vec![i(-8)], "-08:00"),
        ("ofHoursMinutes", vec![i(5), i(30)], "+05:30"),
        ("ofHoursMinutes", vec![i(-5), i(-30)], "-05:30"),
        ("ofHoursMinutesSeconds", vec![i(1), i(2), i(3)], "+01:02:03"),
        ("ofTotalSeconds", vec![i(0)], "Z"),
        ("ofTotalSeconds", vec![i(3600)], "+01:00"),
    ];
    for (name, args, want) in cases {
        assert_eq!(st(ZONE_OFFSET, name, &args), want, "ZoneOffset.{name}");
    }
    assert_eq!(text(Ok(field(ZONE_OFFSET, "UTC"))), "Z");
    assert_eq!(text(Ok(field(ZONE_OFFSET, "MAX"))), "+18:00");
    assert_eq!(text(Ok(field(ZONE_OFFSET, "MIN"))), "-18:00");
    assert!(static_field(ZONE_OFFSET, "NOPE").is_none());
    for bad in [
        st(ZONE_OFFSET, "ofHours", &[i(19)]),
        st(ZONE_OFFSET, "ofHoursMinutes", &[i(1), i(60)]),
        st(ZONE_OFFSET, "ofHoursMinutes", &[i(1), i(-30)]),
        st(ZONE_OFFSET, "ofHoursMinutes", &[i(-1), i(30)]),
        st(ZONE_OFFSET, "ofHoursMinutes", &[i(0), i(30)]).replace("+00:30", "ok"),
        st(ZONE_OFFSET, "ofHoursMinutesSeconds", &[i(0), i(5), i(-5)]),
        st(ZONE_OFFSET, "ofHoursMinutesSeconds", &[i(18), i(1), i(0)]),
        st(ZONE_OFFSET, "ofTotalSeconds", &[i(64_801)]),
    ] {
        assert!(bad.contains("DateTimeException") || bad == "ok", "{bad}");
    }
    let o = vm_call(static_lookup(ZONE_OFFSET, "ofHours", 1).unwrap(), &[i(2)]).unwrap();
    assert_eq!(st(ZONE_OFFSET, "from", &[o]), "+02:00");
    assert!(st(ZONE_OFFSET, "from", &[i(1)]).contains("TemporalAccessor"));
}

#[test]
fn enums_by_number_name_and_values() {
    assert_eq!(st(DAY_OF_WEEK, "of", &[i(1)]), "MONDAY");
    assert_eq!(st(DAY_OF_WEEK, "of", &[i(7)]), "SUNDAY");
    assert!(st(DAY_OF_WEEK, "of", &[i(8)]).contains("Invalid value for DayOfWeek: 8"));
    assert_eq!(
        st(DAY_OF_WEEK, "valueOf", &[Value::str("FRIDAY")]),
        "FRIDAY"
    );
    assert!(st(DAY_OF_WEEK, "valueOf", &[Value::str("FUNDAY")]).contains("No enum constant"));
    assert_eq!(
        st(DAY_OF_WEEK, "from", &[field(DAY_OF_WEEK, "TUESDAY")]),
        "TUESDAY"
    );
    assert!(st(DAY_OF_WEEK, "from", &[i(1)]).contains("TemporalAccessor"));
    assert_eq!(st(MONTH, "of", &[i(12)]), "DECEMBER");
    assert!(st(MONTH, "of", &[i(13)]).contains("Invalid value for MonthOfYear: 13"));
    assert_eq!(st(MONTH, "valueOf", &[Value::str("MARCH")]), "MARCH");
    assert_eq!(st(MONTH, "from", &[field(MONTH, "MAY")]), "MAY");
    assert!(st(MONTH, "from", &[i(1)]).contains("TemporalAccessor"));
    let count =
        |class: &str| match vm_call(static_lookup(class, "values", 0).unwrap(), &[]).unwrap() {
            Value::Array(a) => a.borrow().items.len(),
            _ => 0,
        };
    assert_eq!(count(DAY_OF_WEEK), 7);
    assert_eq!(count(MONTH), 12);
    assert_eq!(count(CHRONO_UNIT), 16);
    assert_eq!(count(CHRONO_FIELD), 30);
    assert_eq!(st(CHRONO_UNIT, "valueOf", &[Value::str("DAYS")]), "Days");
    assert_eq!(st(CHRONO_FIELD, "valueOf", &[Value::str("YEAR")]), "Year");
    assert!(st(CHRONO_UNIT, "valueOf", &[Value::str("EONS")]).contains("No enum constant"));
    assert!(st(CHRONO_FIELD, "valueOf", &[Value::str("EONS")]).contains("No enum constant"));
    assert!(static_lookup("java.time.Nope", "of", 1).is_none());
    assert!(static_field("java.time.Nope", "X").is_none());
    let _ = (call, t);
}
