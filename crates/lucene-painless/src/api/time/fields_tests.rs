#![cfg(test)]
//! `ChronoField`, `ChronoUnit`, `Month` and `DayOfWeek` values and the
//! `get`/`getLong`/`isSupported` members the temporals share.
use super::*;
use crate::api::test_support::{call, t, vm_call};

fn l(x: i64) -> Value {
    Value::Long(x)
}

fn i(x: i32) -> Value {
    Value::Int(x)
}

fn f(name: &str) -> Value {
    static_field(CHRONO_FIELD, name).unwrap_or_else(|| {
        vm_call(
            static_lookup(CHRONO_FIELD, "valueOf", 1).unwrap(),
            &[Value::str(name)],
        )
        .unwrap()
    })
}

fn u(name: &str) -> Value {
    vm_call(
        static_lookup(CHRONO_UNIT, "valueOf", 1).unwrap(),
        &[Value::str(name)],
    )
    .unwrap()
}

fn month(n: i32) -> Value {
    vm_call(static_lookup(MONTH, "of", 1).unwrap(), &[i(n)]).unwrap()
}

fn day(n: i32) -> Value {
    vm_call(static_lookup(DAY_OF_WEEK, "of", 1).unwrap(), &[i(n)]).unwrap()
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
            static_field(ZONE_OFFSET, "UTC").unwrap(),
        ],
    )
    .unwrap()
}

#[test]
fn every_field_of_a_zoned_value() {
    let z = zdt();
    let want = [
        ("NANO_OF_SECOND", "5"),
        ("NANO_OF_DAY", "49530000000005"),
        ("MICRO_OF_SECOND", "0"),
        ("MICRO_OF_DAY", "49530000000"),
        ("MILLI_OF_SECOND", "0"),
        ("MILLI_OF_DAY", "49530000"),
        ("SECOND_OF_MINUTE", "30"),
        ("SECOND_OF_DAY", "49530"),
        ("MINUTE_OF_HOUR", "45"),
        ("MINUTE_OF_DAY", "825"),
        ("HOUR_OF_AMPM", "1"),
        ("CLOCK_HOUR_OF_AMPM", "1"),
        ("HOUR_OF_DAY", "13"),
        ("CLOCK_HOUR_OF_DAY", "13"),
        ("AMPM_OF_DAY", "1"),
        ("DAY_OF_WEEK", "4"),
        ("ALIGNED_DAY_OF_WEEK_IN_MONTH", "1"),
        ("ALIGNED_DAY_OF_WEEK_IN_YEAR", "4"),
        ("DAY_OF_MONTH", "29"),
        ("DAY_OF_YEAR", "60"),
        ("EPOCH_DAY", "19782"),
        ("ALIGNED_WEEK_OF_MONTH", "5"),
        ("ALIGNED_WEEK_OF_YEAR", "9"),
        ("MONTH_OF_YEAR", "2"),
        ("PROLEPTIC_MONTH", "24289"),
        ("YEAR_OF_ERA", "2024"),
        ("YEAR", "2024"),
        ("ERA", "1"),
        ("INSTANT_SECONDS", "1709214330"),
        ("OFFSET_SECONDS", "0"),
    ];
    assert_eq!(want.len(), 30);
    for (idx, (name, v)) in want.iter().enumerate() {
        let field = f(name);
        assert_eq!(t(&field, "name", &[]), *name);
        assert_eq!(t(&field, "ordinal", &[]), idx.to_string());
        assert_eq!(
            t(&z, "getLong", std::slice::from_ref(&field)),
            *v,
            "getLong {name}"
        );
        assert_eq!(t(&z, "isSupported", std::slice::from_ref(&field)), "true");
        assert_eq!(
            t(&field, "getFrom", std::slice::from_ref(&z)),
            *v,
            "getFrom {name}"
        );
        assert_eq!(t(&field, "isSupportedBy", std::slice::from_ref(&z)), "true");
        let got = t(&z, "get", &[field]);
        match *name {
            "NANO_OF_DAY" | "MICRO_OF_DAY" | "EPOCH_DAY" | "PROLEPTIC_MONTH"
            | "INSTANT_SECONDS" => {
                assert!(got.contains("use getLong() instead"), "{name}: {got}");
            }
            _ => assert_eq!(got, *v, "get {name}"),
        }
    }
}

#[test]
fn midnight_and_bce_fields() {
    let midnight = vm_call(
        static_lookup(ZONED, "of", 8).unwrap(),
        &[
            i(-5),
            i(1),
            i(1),
            i(0),
            i(0),
            i(0),
            i(0),
            static_field(ZONE_OFFSET, "UTC").unwrap(),
        ],
    )
    .unwrap();
    let v = |name: &str| t(&midnight, "getLong", &[f(name)]);
    assert_eq!(v("CLOCK_HOUR_OF_AMPM"), "12");
    assert_eq!(v("CLOCK_HOUR_OF_DAY"), "24");
    assert_eq!(v("HOUR_OF_AMPM"), "0");
    assert_eq!(v("AMPM_OF_DAY"), "0");
    assert_eq!(v("YEAR_OF_ERA"), "6");
    assert_eq!(v("YEAR"), "-5");
    assert_eq!(v("ERA"), "0");
}

#[test]
fn field_members() {
    let h = f("HOUR_OF_DAY");
    assert_eq!(t(&h, "isTimeBased", &[]), "true");
    assert_eq!(t(&h, "isDateBased", &[]), "false");
    let y = f("YEAR");
    assert_eq!(t(&y, "isDateBased", &[]), "true");
    assert_eq!(t(&y, "isTimeBased", &[]), "false");
    assert_eq!(t(&h, "checkValidValue", &[l(23)]), "23");
    assert_eq!(
        t(&h, "checkValidValue", &[l(24)]),
        "java.time.DateTimeException: Invalid value for HourOfDay (valid values 0 - 23): 24"
    );
    assert_eq!(t(&h, "checkValidIntValue", &[l(5)]), "5");
    assert!(t(&h, "checkValidIntValue", &[l(99)]).contains("Invalid value"));
    assert_eq!(
        t(&h, "compareTo", std::slice::from_ref(&y)),
        (12 - 26).to_string()
    );
    assert!(t(&h, "compareTo", &[i(1)]).contains("Exception"));
    assert_eq!(t(&h, "toString", &[]), "HourOfDay");
    assert!(t(&h, "isSupportedBy", &[i(1)]).contains("Exception"));
    assert_eq!(t(&zdt(), "isSupported", &[Value::Null]), "false");
    assert!(t(&zdt(), "getLong", &[i(1)]).contains("Exception"));
}

#[test]
fn instants_answer_four_fields() {
    let inst = vm_call(
        static_lookup(INSTANT, "ofEpochSecond", 2).unwrap(),
        &[l(7), l(123_456_789)],
    )
    .unwrap();
    for (name, v) in [
        ("NANO_OF_SECOND", "123456789"),
        ("MICRO_OF_SECOND", "123456"),
        ("MILLI_OF_SECOND", "123"),
        ("INSTANT_SECONDS", "7"),
    ] {
        assert_eq!(t(&inst, "getLong", &[f(name)]), v, "{name}");
        assert!(!t(&inst, "get", &[f(name)]).is_empty());
        assert_eq!(t(&inst, "isSupported", &[f(name)]), "true");
    }
    assert!(t(&inst, "getLong", &[f("YEAR")]).contains("Unsupported field: Year"));
    assert_eq!(t(&inst, "isSupported", &[f("YEAR")]), "false");
}

#[test]
fn single_field_temporals() {
    let d = day(3);
    assert_eq!(t(&d, "getLong", &[f("DAY_OF_WEEK")]), "3");
    assert_eq!(t(&d, "get", &[f("DAY_OF_WEEK")]), "3");
    assert_eq!(t(&d, "isSupported", &[f("DAY_OF_WEEK")]), "true");
    assert_eq!(t(&d, "isSupported", &[f("YEAR")]), "false");
    assert!(t(&d, "getLong", &[f("YEAR")]).contains("Unsupported field"));
    let m = month(11);
    assert_eq!(t(&m, "getLong", &[f("MONTH_OF_YEAR")]), "11");
    assert_eq!(t(&m, "isSupported", &[f("MONTH_OF_YEAR")]), "true");
    assert_eq!(t(&m, "isSupported", &[f("YEAR")]), "false");
    let o = vm_call(static_lookup(ZONE_OFFSET, "ofHours", 1).unwrap(), &[i(2)]).unwrap();
    assert_eq!(t(&o, "getLong", &[f("OFFSET_SECONDS")]), "7200");
    assert_eq!(t(&o, "isSupported", &[f("OFFSET_SECONDS")]), "true");
    assert_eq!(t(&o, "isSupported", &[f("YEAR")]), "false");
    assert!(t(&o, "getLong", &[f("YEAR")]).contains("Unsupported field"));
}

#[test]
fn month_members() {
    let feb = month(2);
    for (name, args, want) in [
        ("getValue", vec![], "2"),
        ("name", vec![], "FEBRUARY"),
        ("ordinal", vec![], "1"),
        ("plus", vec![l(11)], "JANUARY"),
        ("minus", vec![l(2)], "DECEMBER"),
        ("length", vec![Value::Bool(true)], "29"),
        ("length", vec![Value::Bool(false)], "28"),
        ("maxLength", vec![], "29"),
        ("minLength", vec![], "28"),
        ("firstMonthOfQuarter", vec![], "JANUARY"),
        ("firstDayOfYear", vec![Value::Bool(false)], "32"),
        ("compareTo", vec![month(5)], "-3"),
    ] {
        assert_eq!(t(&feb, name, &args), want, "{name}");
    }
    let firsts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335];
    for (n, first) in firsts.iter().enumerate() {
        let m = month(n as i32 + 1);
        let leap = i32::from(n >= 2);
        assert_eq!(
            t(&m, "firstDayOfYear", &[Value::Bool(false)]),
            first.to_string()
        );
        assert_eq!(
            t(&m, "firstDayOfYear", &[Value::Bool(true)]),
            (first + leap).to_string()
        );
    }
    assert_eq!(t(&month(11), "firstMonthOfQuarter", &[]), "OCTOBER");
    assert!(t(&feb, "compareTo", &[day(1)]).contains("Exception"));
    assert!(call(&feb, "nope", &[]).is_err() || t(&feb, "nope", &[]).contains("nope"));
}

#[test]
fn day_of_week_members() {
    let thu = day(4);
    for (name, args, want) in [
        ("getValue", vec![], "4"),
        ("name", vec![], "THURSDAY"),
        ("ordinal", vec![], "3"),
        ("plus", vec![l(4)], "MONDAY"),
        ("plus", vec![l(-5)], "SATURDAY"),
        ("minus", vec![l(4)], "SUNDAY"),
        ("minus", vec![l(-5)], "TUESDAY"),
        ("compareTo", vec![day(1)], "3"),
    ] {
        assert_eq!(t(&thu, name, &args), want, "{name}");
    }
    assert!(t(&thu, "compareTo", &[month(1)]).contains("Exception"));
}

#[test]
fn unit_members() {
    let days = u("DAYS");
    for (name, args, want) in [
        ("name", vec![], "DAYS"),
        ("ordinal", vec![], "7"),
        ("getDuration", vec![], "PT24H"),
        ("isDateBased", vec![], "true"),
        ("isTimeBased", vec![], "false"),
        ("isDurationEstimated", vec![], "true"),
        ("compareTo", vec![u("HOURS")], "2"),
    ] {
        assert_eq!(t(&days, name, &args), want, "{name}");
    }
    let hours = u("HOURS");
    assert_eq!(t(&hours, "isTimeBased", &[]), "true");
    assert_eq!(t(&hours, "isDurationEstimated", &[]), "false");
    let a = zdt();
    let b = call(&a, "plusDays", &[l(10)]).unwrap();
    assert_eq!(t(&days, "between", &[a.clone(), b.clone()]), "10");
    assert_eq!(t(&u("WEEKS"), "between", &[a.clone(), b]), "1");
    let i0 = vm_call(static_lookup(INSTANT, "ofEpochSecond", 1).unwrap(), &[l(0)]).unwrap();
    let i1 = vm_call(
        static_lookup(INSTANT, "ofEpochSecond", 1).unwrap(),
        &[l(120)],
    )
    .unwrap();
    assert_eq!(t(&u("MINUTES"), "between", &[i0, i1]), "2");
    assert!(t(&days, "between", &[i(1), i(2)]).contains("Exception"));
    assert!(t(&days, "compareTo", &[i(1)]).contains("Exception"));
}

#[test]
fn mixed_temporal_end_points() {
    let z = zdt();
    let inst = call(&z, "toInstant", &[]).unwrap();
    let later = call(&inst, "plusSeconds", &[l(10)]).unwrap();
    assert_eq!(t(&inst, "until", &[later, u("SECONDS")]), "10");
    // An Instant's end may be a ZonedDateTime; a zoned value's may not be an Instant.
    assert_eq!(t(&inst, "until", &[z.clone(), u("SECONDS")]), "0");
    assert!(t(&z, "until", &[inst, u("SECONDS")]).contains("Unable to obtain ZonedDateTime"));
    assert!(t(&z, "until", &[i(1), u("SECONDS")]).contains("Exception"));
}
