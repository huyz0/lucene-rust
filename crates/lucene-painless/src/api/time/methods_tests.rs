#![cfg(test)]
//! Instance methods of the `java.time` values, called by name the way
//! compiled code does, against what Java prints.
use super::*;
use crate::api::test_support::{t, text, vm_call};

fn l(x: i64) -> Value {
    Value::Long(x)
}

fn i(x: i32) -> Value {
    Value::Int(x)
}

fn field(class: &str, name: &str) -> Value {
    static_field(class, name).expect("a constant")
}

fn unit(name: &str) -> Value {
    field(CHRONO_UNIT, name)
}

fn zdt() -> Value {
    vm_call(
        static_lookup(ZONED, "of", 8).unwrap(),
        &[
            i(2024),
            i(2),
            i(29),
            i(13),
            i(45),
            i(30),
            i(5),
            field(ZONE_OFFSET, "UTC"),
        ],
    )
    .unwrap()
}

fn dur(secs: i64) -> Value {
    vm_call(static_lookup(DURATION, "ofSeconds", 1).unwrap(), &[l(secs)]).unwrap()
}

fn inst(millis: i64) -> Value {
    vm_call(
        static_lookup(INSTANT, "ofEpochMilli", 1).unwrap(),
        &[l(millis)],
    )
    .unwrap()
}

fn offset(hours: i32) -> Value {
    vm_call(
        static_lookup(ZONE_OFFSET, "ofHours", 1).unwrap(),
        &[i(hours)],
    )
    .unwrap()
}

fn check_all(recv: &Value, cases: &[(&str, Vec<Value>, &str)]) {
    for (name, args, want) in cases {
        assert_eq!(t(recv, name, args), *want, "{name}");
    }
}

#[test]
fn zoned_getters() {
    check_all(
        &zdt(),
        &[
            ("getYear", vec![], "2024"),
            ("getMonthValue", vec![], "2"),
            ("getDayOfMonth", vec![], "29"),
            ("getDayOfYear", vec![], "60"),
            ("getHour", vec![], "13"),
            ("getMinute", vec![], "45"),
            ("getSecond", vec![], "30"),
            ("getNano", vec![], "5"),
            ("getMonth", vec![], "FEBRUARY"),
            ("getDayOfWeek", vec![], "THURSDAY"),
            ("getOffset", vec![], "Z"),
            ("getZone", vec![], "Z"),
            ("toEpochSecond", vec![], "1709214330"),
            ("toInstant", vec![], "2024-02-29T13:45:30.000000005Z"),
        ],
    );
}

#[test]
fn zoned_plus_and_minus() {
    check_all(
        &zdt(),
        &[
            ("plusNanos", vec![l(5)], "2024-02-29T13:45:30.000000010Z"),
            ("plusSeconds", vec![l(30)], "2024-02-29T13:46:00.000000005Z"),
            ("plusMinutes", vec![l(15)], "2024-02-29T14:00:30.000000005Z"),
            ("plusHours", vec![l(11)], "2024-03-01T00:45:30.000000005Z"),
            ("plusDays", vec![l(1)], "2024-03-01T13:45:30.000000005Z"),
            ("plusWeeks", vec![l(1)], "2024-03-07T13:45:30.000000005Z"),
            ("plusMonths", vec![l(12)], "2025-02-28T13:45:30.000000005Z"),
            ("plusYears", vec![l(1)], "2025-02-28T13:45:30.000000005Z"),
            ("minusYears", vec![l(4)], "2020-02-29T13:45:30.000000005Z"),
            ("minusMonths", vec![l(1)], "2024-01-29T13:45:30.000000005Z"),
            ("minusWeeks", vec![l(1)], "2024-02-22T13:45:30.000000005Z"),
            ("minusDays", vec![l(29)], "2024-01-31T13:45:30.000000005Z"),
            ("minusHours", vec![l(14)], "2024-02-28T23:45:30.000000005Z"),
            (
                "minusMinutes",
                vec![l(45)],
                "2024-02-29T13:00:30.000000005Z",
            ),
            (
                "minusSeconds",
                vec![l(31)],
                "2024-02-29T13:44:59.000000005Z",
            ),
            ("minusNanos", vec![l(5)], "2024-02-29T13:45:30Z"),
            (
                "plus",
                vec![l(2), unit("DAYS")],
                "2024-03-02T13:45:30.000000005Z",
            ),
            (
                "minus",
                vec![l(2), unit("DAYS")],
                "2024-02-27T13:45:30.000000005Z",
            ),
            (
                "plus",
                vec![l(1), unit("DECADES")],
                "2034-02-28T13:45:30.000000005Z",
            ),
            (
                "plus",
                vec![l(1), unit("CENTURIES")],
                "2124-02-29T13:45:30.000000005Z",
            ),
            (
                "plus",
                vec![l(1), unit("MILLENNIA")],
                "3024-02-29T13:45:30.000000005Z",
            ),
            (
                "plus",
                vec![l(1), unit("HALF_DAYS")],
                "2024-03-01T01:45:30.000000005Z",
            ),
            (
                "plus",
                vec![l(1), unit("MILLIS")],
                "2024-02-29T13:45:30.001000005Z",
            ),
            (
                "plus",
                vec![l(1), unit("MICROS")],
                "2024-02-29T13:45:30.000001005Z",
            ),
            ("plus", vec![dur(90)], "2024-02-29T13:47:00.000000005Z"),
            ("minus", vec![dur(30)], "2024-02-29T13:45:00.000000005Z"),
        ],
    );
    assert!(t(&zdt(), "plus", &[l(1), unit("FOREVER")]).contains("Unsupported unit"));
    assert!(t(&zdt(), "plusYears", &[l(i64::MAX)]).contains("Exception"));
}

#[test]
fn zoned_with_and_truncate() {
    check_all(
        &zdt(),
        &[
            ("withHour", vec![i(1)], "2024-02-29T01:45:30.000000005Z"),
            ("withMinute", vec![i(0)], "2024-02-29T13:00:30.000000005Z"),
            ("withSecond", vec![i(0)], "2024-02-29T13:45:00.000000005Z"),
            ("withNano", vec![i(0)], "2024-02-29T13:45:30Z"),
            (
                "withDayOfMonth",
                vec![i(1)],
                "2024-02-01T13:45:30.000000005Z",
            ),
            (
                "withDayOfMonth",
                vec![i(29)],
                "2024-02-29T13:45:30.000000005Z",
            ),
            (
                "withDayOfYear",
                vec![i(1)],
                "2024-01-01T13:45:30.000000005Z",
            ),
            (
                "withDayOfYear",
                vec![i(60)],
                "2024-02-29T13:45:30.000000005Z",
            ),
            ("withMonth", vec![i(1)], "2024-01-29T13:45:30.000000005Z"),
            ("withMonth", vec![i(2)], "2024-02-29T13:45:30.000000005Z"),
            ("withYear", vec![i(2023)], "2023-02-28T13:45:30.000000005Z"),
            ("withYear", vec![i(2024)], "2024-02-29T13:45:30.000000005Z"),
            ("truncatedTo", vec![unit("HOURS")], "2024-02-29T13:00Z"),
            ("truncatedTo", vec![unit("DAYS")], "2024-02-29T00:00Z"),
            (
                "truncatedTo",
                vec![unit("NANOS")],
                "2024-02-29T13:45:30.000000005Z",
            ),
            (
                "withFixedOffsetZone",
                vec![],
                "2024-02-29T13:45:30.000000005Z",
            ),
            (
                "withEarlierOffsetAtOverlap",
                vec![],
                "2024-02-29T13:45:30.000000005Z",
            ),
            (
                "withLaterOffsetAtOverlap",
                vec![],
                "2024-02-29T13:45:30.000000005Z",
            ),
            (
                "withZoneSameInstant",
                vec![offset(2)],
                "2024-02-29T15:45:30.000000005+02:00",
            ),
            (
                "withZoneSameLocal",
                vec![offset(2)],
                "2024-02-29T13:45:30.000000005+02:00",
            ),
        ],
    );
    for (name, bad) in [
        ("withHour", 24),
        ("withMinute", 60),
        ("withSecond", 60),
        ("withNano", -1),
        ("withDayOfMonth", 32),
        ("withDayOfYear", 400),
        ("withMonth", 13),
    ] {
        assert!(
            t(&zdt(), name, &[i(bad)]).starts_with("java.time.DateTimeException"),
            "{name}"
        );
    }
}

#[test]
fn zoned_comparisons_and_until() {
    let later = zdt();
    let earlier = t(&later, "minusDays", &[l(1)]);
    assert!(earlier.starts_with("2024-02-28"));
    let e = call_value(&later, "minusDays", &[l(1)]);
    check_all(
        &later,
        &[
            ("isAfter", vec![e.clone()], "true"),
            ("isBefore", vec![e.clone()], "false"),
            ("isEqual", vec![e.clone()], "false"),
            ("isEqual", vec![later.clone()], "true"),
            ("compareTo", vec![later.clone()], "0"),
            (
                "until",
                vec![call_value(&later, "plusDays", &[l(3)]), unit("DAYS")],
                "3",
            ),
            (
                "until",
                vec![call_value(&later, "plusHours", &[l(5)]), unit("HOURS")],
                "5",
            ),
            (
                "until",
                vec![call_value(&later, "plusMonths", &[l(14)]), unit("YEARS")],
                "1",
            ),
        ],
    );
    assert!(
        t(&later, "compareTo", std::slice::from_ref(&e))
            .parse::<i32>()
            .unwrap()
            > 0
    );
    assert!(
        t(&e, "compareTo", std::slice::from_ref(&later))
            .parse::<i32>()
            .unwrap()
            < 0
    );
    // Same instant, different offset: ordered by local time, then zone id.
    let shifted = call_value(&later, "withZoneSameInstant", &[offset(2)]);
    assert_eq!(t(&later, "isEqual", std::slice::from_ref(&shifted)), "true");
    assert_ne!(t(&later, "compareTo", &[shifted]), "0");
    assert!(t(&later, "isAfter", &[i(1)]).contains("Exception"));
}

fn call_value(recv: &Value, name: &str, args: &[Value]) -> Value {
    crate::api::test_support::call(recv, name, args).expect(name)
}

#[test]
fn instant_methods() {
    let a = inst(1_500);
    check_all(
        &a,
        &[
            ("getEpochSecond", vec![], "1"),
            ("getNano", vec![], "500000000"),
            ("toEpochMilli", vec![], "1500"),
            ("plusSeconds", vec![l(1)], "1970-01-01T00:00:02.500Z"),
            ("minusSeconds", vec![l(1)], "1970-01-01T00:00:00.500Z"),
            ("plusMillis", vec![l(500)], "1970-01-01T00:00:02Z"),
            ("minusMillis", vec![l(500)], "1970-01-01T00:00:01Z"),
            ("plusNanos", vec![l(1)], "1970-01-01T00:00:01.500000001Z"),
            ("minusNanos", vec![l(1)], "1970-01-01T00:00:01.499999999Z"),
            ("plus", vec![dur(60)], "1970-01-01T00:01:01.500Z"),
            ("minus", vec![dur(1)], "1970-01-01T00:00:00.500Z"),
            (
                "plus",
                vec![l(1), unit("MINUTES")],
                "1970-01-01T00:01:01.500Z",
            ),
            (
                "plus",
                vec![l(1), unit("HOURS")],
                "1970-01-01T01:00:01.500Z",
            ),
            (
                "plus",
                vec![l(1), unit("HALF_DAYS")],
                "1970-01-01T12:00:01.500Z",
            ),
            ("plus", vec![l(1), unit("DAYS")], "1970-01-02T00:00:01.500Z"),
            (
                "plus",
                vec![l(2), unit("MILLIS")],
                "1970-01-01T00:00:01.502Z",
            ),
            (
                "plus",
                vec![l(2), unit("MICROS")],
                "1970-01-01T00:00:01.500002Z",
            ),
            (
                "plus",
                vec![l(2), unit("NANOS")],
                "1970-01-01T00:00:01.500000002Z",
            ),
            (
                "plus",
                vec![l(2), unit("SECONDS")],
                "1970-01-01T00:00:03.500Z",
            ),
            (
                "minus",
                vec![l(1), unit("DAYS")],
                "1969-12-31T00:00:01.500Z",
            ),
            ("truncatedTo", vec![unit("SECONDS")], "1970-01-01T00:00:01Z"),
            (
                "truncatedTo",
                vec![unit("NANOS")],
                "1970-01-01T00:00:01.500Z",
            ),
            ("atZone", vec![offset(1)], "1970-01-01T01:00:01.500+01:00"),
            ("isAfter", vec![inst(0)], "true"),
            ("isBefore", vec![inst(0)], "false"),
            ("compareTo", vec![inst(1_500)], "0"),
            ("until", vec![inst(5_500), unit("SECONDS")], "4"),
            ("until", vec![inst(1_000_500), unit("MILLIS")], "999000"),
        ],
    );
    assert!(t(&a, "plus", &[l(1), unit("MONTHS")]).contains("Unsupported unit"));
    assert!(t(&a, "compareTo", &[inst(0)]).parse::<i32>().unwrap() > 0);
    assert!(t(&a, "compareTo", &[i(1)]).contains("Exception"));
}

#[test]
fn duration_methods() {
    let d = dur(3725);
    check_all(
        &d,
        &[
            ("getSeconds", vec![], "3725"),
            ("getNano", vec![], "0"),
            ("isNegative", vec![], "false"),
            ("isZero", vec![], "false"),
            ("toDays", vec![], "0"),
            ("toHours", vec![], "1"),
            ("toMinutes", vec![], "62"),
            ("toMillis", vec![], "3725000"),
            ("toNanos", vec![], "3725000000000"),
            ("negated", vec![], "PT-1H-2M-5S"),
            ("multipliedBy", vec![l(2)], "PT2H4M10S"),
            ("multipliedBy", vec![l(0)], "PT0S"),
            ("multipliedBy", vec![l(1)], "PT1H2M5S"),
            ("dividedBy", vec![l(5)], "PT12M25S"),
            ("dividedBy", vec![l(1)], "PT1H2M5S"),
            (
                "dividedBy",
                vec![l(0)],
                "java.lang.ArithmeticException: Cannot divide by zero",
            ),
            ("plus", vec![dur(5)], "PT1H2M10S"),
            ("minus", vec![dur(5)], "PT1H2M"),
            ("plus", vec![l(1), unit("DAYS")], "PT25H2M5S"),
            ("plus", vec![l(1500), unit("MILLIS")], "PT1H2M6.5S"),
            ("plus", vec![l(1), unit("MINUTES")], "PT1H3M5S"),
            ("minus", vec![l(1), unit("HOURS")], "PT2M5S"),
            ("plusDays", vec![l(1)], "PT25H2M5S"),
            ("plusHours", vec![l(1)], "PT2H2M5S"),
            ("plusMinutes", vec![l(1)], "PT1H3M5S"),
            ("plusSeconds", vec![l(1)], "PT1H2M6S"),
            ("plusMillis", vec![l(500)], "PT1H2M5.5S"),
            ("plusNanos", vec![l(1)], "PT1H2M5.000000001S"),
            ("minusDays", vec![l(1)], "PT-22H-57M-55S"),
            ("minusHours", vec![l(1)], "PT2M5S"),
            ("minusMinutes", vec![l(2)], "PT1H5S"),
            ("minusSeconds", vec![l(5)], "PT1H2M"),
            ("minusMillis", vec![l(500)], "PT1H2M4.5S"),
            ("minusNanos", vec![l(1)], "PT1H2M4.999999999S"),
            ("withSeconds", vec![l(1)], "PT1S"),
            ("withNanos", vec![i(500_000_000)], "PT1H2M5.5S"),
            ("compareTo", vec![dur(3725)], "0"),
            ("get", vec![unit("SECONDS")], "3725"),
            ("get", vec![unit("NANOS")], "0"),
        ],
    );
    assert!(t(&d, "withNanos", &[i(-1)]).contains("DateTimeException"));
    assert!(t(&d, "get", &[unit("DAYS")]).contains("Unsupported unit"));
    assert!(t(&d, "plus", &[l(1), unit("MONTHS")]).contains("estimated duration"));
    assert_eq!(t(&t_value(&d, "negated"), "abs", &[]), "PT1H2M5S");
    assert_eq!(t(&t_value(&d, "negated"), "isNegative", &[]), "true");
    assert_eq!(t(&dur(0), "isZero", &[]), "true");
}

fn t_value(recv: &Value, name: &str) -> Value {
    call_value(recv, name, &[])
}

#[test]
fn offset_methods() {
    let o = offset(5);
    check_all(
        &o,
        &[
            ("getTotalSeconds", vec![], "18000"),
            ("getId", vec![], "+05:00"),
            ("normalized", vec![], "+05:00"),
            ("compareTo", vec![offset(3)], "-7200"),
        ],
    );
    assert_eq!(text(Ok(o)), "+05:00");
}
