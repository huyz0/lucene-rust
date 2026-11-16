#![cfg(test)]
use super::*;

fn at(secs: i64, nanos: i64) -> Zoned {
    Zoned::of_instant(Inst::of(secs, nanos).unwrap(), 0).unwrap()
}

#[test]
fn calendar_round_trips() {
    for d in [-800_000, -1, 0, 1, 59, 60, 11_016, 2_932_896, 100_000_000] {
        let (y, m, dd) = civil_from_days(d);
        assert_eq!(days_from_civil(y, m, dd), d);
    }
    assert_eq!(civil_from_days(0), (1970, 1, 1));
    assert_eq!(civil_from_days(19_782), (2024, 2, 29));
}

#[test]
fn prints_as_java_does() {
    assert_eq!(at(0, 0).string(), "1970-01-01T00:00Z");
    assert_eq!(
        at(1_700_000_005, 120_000_000).string(),
        "2023-11-14T22:13:25.120Z"
    );
    assert_eq!(at(-1, 0).string(), "1969-12-31T23:59:59Z");
    assert_eq!(Inst { secs: 0, nanos: 0 }.string(), "1970-01-01T00:00:00Z");
    assert_eq!(
        Inst {
            secs: 1_700_000_005,
            nanos: 123_000
        }
        .string(),
        "2023-11-14T22:13:25.000123Z"
    );
    assert_eq!(
        Dur {
            secs: 3725,
            nanos: 500_000_000
        }
        .string(),
        "PT1H2M5.5S"
    );
    assert_eq!(
        Dur {
            secs: -1,
            nanos: 500_000_000
        }
        .string(),
        "PT-0.5S"
    );
    assert_eq!(Dur::ZERO.string(), "PT0S");
    assert_eq!(offset_string(-3600 * 5 - 1800), "-05:30");
}

#[test]
fn month_arithmetic_clamps_the_day() {
    let jan31 = Zoned::make(days_from_civil(2023, 1, 31), 0, 0).unwrap();
    let feb = jan31.plus_months(1).unwrap();
    assert_eq!(feb.ymd(), (2023, 2, 28));
    let leap = Zoned::make(days_from_civil(2024, 2, 29), 0, 0).unwrap();
    assert_eq!(leap.plus_years(1).unwrap().ymd(), (2025, 2, 28));
    assert_eq!(leap.plus_years(-1).unwrap().ymd(), (2023, 2, 28));
}

#[test]
fn until_counts_whole_units() {
    let a = Zoned::make(days_from_civil(2023, 1, 31), 0, 0).unwrap();
    let b = Zoned::make(days_from_civil(2023, 2, 28), 0, 0).unwrap();
    assert_eq!(a.until(b, Unit(9)).unwrap(), 0);
    assert_eq!(a.until(b, Unit(7)).unwrap(), 28);
    assert_eq!(a.until(b, Unit(8)).unwrap(), 4);
    let c = Zoned::make(days_from_civil(2024, 1, 31), 1, 0).unwrap();
    assert_eq!(a.until(c, Unit(10)).unwrap(), 1);
}

#[test]
fn truncation_rejects_big_units() {
    let z = at(1_700_000_005, 5);
    assert_eq!(z.truncated(Unit(4)).unwrap().string(), "2023-11-14T22:13Z");
    assert_eq!(z.truncated(Unit(7)).unwrap().string(), "2023-11-14T00:00Z");
    assert!(z.truncated(Unit(8)).is_err());
}

#[test]
fn invalid_values_name_the_field() {
    let e = date_of(2023, 2, 30).unwrap_err();
    assert_eq!(e.message().as_deref(), Some("Invalid date 'FEBRUARY 30'"));
    let e = date_of(2023, 13, 1).unwrap_err();
    assert_eq!(
        e.message().as_deref(),
        Some("Invalid value for MonthOfYear (valid values 1 - 12): 13")
    );
}

struct NoEnv;

impl crate::vm::Env for NoEnv {
    fn internal(&mut self, _name: &str, _args: &[Value]) -> Result<Value, Throw> {
        unreachable!()
    }
}

fn call(recv: &Value, name: &str, args: &[Value]) -> Result<Value, Throw> {
    let mut env = NoEnv;
    let mut vm = Vm::new(std::rc::Rc::new(crate::vm::Program::default()), &mut env);
    crate::api::dynamic::call(&mut vm, recv, name, args)
}

fn text(r: Result<Value, Throw>) -> String {
    r.map(|v| v.java_to_string())
        .unwrap_or_else(|e| format!("{}: {}", e.class(), e.message().unwrap_or_default()))
}

fn stat(class: &str, name: &str, args: &[Value]) -> Result<Value, Throw> {
    let mut env = NoEnv;
    let mut vm = Vm::new(std::rc::Rc::new(crate::vm::Program::default()), &mut env);
    static_lookup(class, name, args.len()).expect("a static")(&mut vm, args)
}

fn unit(name: &str) -> Value {
    static_field(CHRONO_UNIT, name).unwrap()
}

fn zoned() -> Value {
    utc_from_millis(1_700_000_005_120).unwrap()
}

#[test]
fn values_match_java() {
    let z = zoned();
    let p = |name: &str, a: i64| text(call(&z, name, &[Value::Long(a)]));
    assert_eq!(z.java_to_string(), "2023-11-14T22:13:25.120Z");
    assert_eq!(p("plusMonths", 3), "2024-02-14T22:13:25.120Z");
    assert_eq!(p("minusYears", 1), "2022-11-14T22:13:25.120Z");
    assert_eq!(p("plusWeeks", 2), "2023-11-28T22:13:25.120Z");
    assert_eq!(p("plusNanos", 5), "2023-11-14T22:13:25.120000005Z");
    let t = |name: &str, v: i32| text(call(&z, name, &[Value::Int(v)]));
    assert_eq!(t("withDayOfYear", 100), "2023-04-10T22:13:25.120Z");
    assert_eq!(t("withMonth", 2), "2023-02-14T22:13:25.120Z");
    assert_eq!(
        t("withHour", 25),
        "java.time.DateTimeException: Invalid value for HourOfDay (valid values 0 - 23): 25"
    );
    assert_eq!(
        t("withDayOfMonth", 40),
        "java.time.DateTimeException: Invalid value for DayOfMonth (valid values 1 - 28/31): 40"
    );
    assert_eq!(
        text(call(&z, "truncatedTo", &[unit("HOURS")])),
        "2023-11-14T22:00Z"
    );
    assert_eq!(text(call(&z, "getDayOfWeek", &[])), "TUESDAY");
    assert_eq!(text(call(&z, "getMonth", &[])), "NOVEMBER");
    assert_eq!(text(call(&z, "getDayOfYear", &[])), "318");
    let later = call(&z, "plusDays", &[Value::Long(40)]).unwrap();
    let later = call(&later, "plusHours", &[Value::Long(3)]).unwrap();
    let between =
        |u: &str, a: &Value, b: &Value| text(call(&unit(u), "between", &[a.clone(), b.clone()]));
    assert_eq!(between("DAYS", &z, &later), "40");
    assert_eq!(between("MONTHS", &z, &later), "1");
    assert_eq!(between("HOURS", &z, &later), "963");
    let plus_nanos = call(&z, "plusNanos", &[Value::Long(1_999_999)]).unwrap();
    assert_eq!(between("MILLIS", &z, &plus_nanos), "1");
    let back = call(&z, "minusYears", &[Value::Long(3)]).unwrap();
    assert_eq!(between("YEARS", &z, &back), "-3");
    let five = call(&z, "plusNanos", &[Value::Long(5)]).unwrap();
    assert_eq!(text(call(&z, "compareTo", &[five])), "-5");
    assert_eq!(z.java_hash(), -1_526_894_067);
    let plus_five = call(
        &z,
        "withZoneSameInstant",
        &[stat(ZONE_OFFSET, "ofHours", &[Value::Int(5)]).unwrap()],
    )
    .unwrap();
    assert_eq!(plus_five.java_to_string(), "2023-11-15T03:13:25.120+05:00");
    assert_eq!(plus_five.java_hash(), 120_808_593);
}

#[test]
fn instants_and_durations_match_java() {
    let i =
        |s: i64, n: i64| stat(INSTANT, "ofEpochSecond", &[Value::Long(s), Value::Long(n)]).unwrap();
    assert_eq!(i(-1, 5).java_to_string(), "1969-12-31T23:59:59.000000005Z");
    assert_eq!(
        static_field(INSTANT, "MIN").unwrap().java_to_string(),
        "-1000000000-01-01T00:00:00Z"
    );
    assert_eq!(
        static_field(INSTANT, "MAX").unwrap().java_to_string(),
        "+1000000000-12-31T23:59:59.999999999Z"
    );
    assert_eq!(
        i(-62_167_219_201, 0).java_to_string(),
        "-0001-12-31T23:59:59Z"
    );
    assert_eq!(
        i(253_402_300_800, 0).java_to_string(),
        "+10000-01-01T00:00:00Z"
    );
    assert_eq!(
        i(-100_000_000_000_000, 0).java_to_string(),
        "-3166904-02-24T14:13:20Z"
    );
    assert_eq!(i(5, 0).java_hash(), 5);
    let d =
        |s: i64, n: i64| stat(DURATION, "ofSeconds", &[Value::Long(s), Value::Long(n)]).unwrap();
    assert_eq!(d(-90, 500).java_to_string(), "PT-1M-29.9999995S");
    assert_eq!(d(5, 7).java_hash(), 362);
    let ms = stat(DURATION, "ofMillis", &[Value::Long(-1500)]).unwrap();
    assert_eq!(ms.java_to_string(), "PT-1.5S");
    let h = stat(DURATION, "ofHours", &[Value::Long(25)]).unwrap();
    let h = call(&h, "plusMinutes", &[Value::Long(3)]).unwrap();
    assert_eq!(h.java_to_string(), "PT25H3M");
    let by7 = call(&d(100, 0), "dividedBy", &[Value::Long(7)]).unwrap();
    assert_eq!(by7.java_to_string(), "PT14.285714285S");
    let ms = stat(DURATION, "ofMillis", &[Value::Long(1500)]).unwrap();
    let x3 = call(&ms, "multipliedBy", &[Value::Long(3)]).unwrap();
    assert_eq!(x3.java_to_string(), "PT4.5S");
    assert_eq!(
        stat(
            ZONE_OFFSET,
            "ofHoursMinutes",
            &[Value::Int(-3), Value::Int(-30)]
        )
        .unwrap()
        .java_to_string(),
        "-03:30"
    );
    assert_eq!(text(call(&i(-2, 0), "toEpochMilli", &[])), "-2000");
    let ms_neg = stat(INSTANT, "ofEpochMilli", &[Value::Long(-1500)]).unwrap();
    assert_eq!(text(call(&ms_neg, "toEpochMilli", &[])), "-1500");
    let until = call(&i(90_061, 0), "until", &[i(0, 0), unit("HOURS")]).unwrap();
    assert_eq!(until.java_to_string(), "-25");
    assert_eq!(
        text(call(&i(0, 0), "until", &[i(5, 0), unit("WEEKS")])),
        "java.time.temporal.UnsupportedTemporalTypeException: Unsupported unit: Weeks"
    );
}

#[test]
fn enums_and_statics() {
    let day = |n: i32| stat(DAY_OF_WEEK, "of", &[Value::Int(n)]).unwrap();
    assert_eq!(day(1).java_to_string(), "MONDAY");
    assert_eq!(
        text(stat(DAY_OF_WEEK, "of", &[Value::Int(8)])),
        "java.time.DateTimeException: Invalid value for DayOfWeek: 8"
    );
    assert_eq!(text(call(&day(1), "minus", &[Value::Long(3)])), "FRIDAY");
    assert_eq!(text(call(&day(5), "compareTo", &[day(1)])), "4");
    assert_eq!(text(stat(MONTH, "valueOf", &[Value::str("MAY")])), "MAY");
    assert_eq!(
        text(stat(MONTH, "valueOf", &[Value::str("MAYBE")])),
        "java.lang.IllegalArgumentException: No enum constant java.time.Month.MAYBE"
    );
    let feb = static_field(MONTH, "FEBRUARY").unwrap();
    assert_eq!(text(call(&feb, "length", &[Value::Bool(true)])), "29");
    assert_eq!(text(call(&feb, "minLength", &[])), "28");
    assert_eq!(text(call(&feb, "plus", &[Value::Long(11)])), "JANUARY");
    assert_eq!(
        text(call(&feb, "firstDayOfYear", &[Value::Bool(true)])),
        "32"
    );
    let nov = static_field(MONTH, "NOVEMBER").unwrap();
    assert_eq!(text(call(&nov, "firstMonthOfQuarter", &[])), "OCTOBER");
    assert_eq!(text(call(&unit("DAYS"), "isDateBased", &[])), "true");
    assert_eq!(text(call(&unit("HOURS"), "isTimeBased", &[])), "true");
    assert_eq!(text(call(&unit("HALF_DAYS"), "getDuration", &[])), "PT12H");
    assert_eq!(unit("HALF_DAYS").java_to_string(), "HalfDays");
    assert!(static_field(MONTH, "SMARCH").is_none());
    let all = stat(CHRONO_UNIT, "values", &[]).unwrap();
    assert!(all.java_to_string().len() > 10);
    assert_eq!(
        text(stat(
            ZONED,
            "of",
            &[
                Value::Int(2023),
                Value::Int(2),
                Value::Int(30),
                Value::Int(0),
                Value::Int(0),
                Value::Int(0),
                Value::Int(0),
                static_field(ZONE_OFFSET, "UTC").unwrap(),
            ]
        )),
        "java.time.DateTimeException: Invalid date 'FEBRUARY 30'"
    );
    let leap = stat(
        ZONED,
        "of",
        &[
            Value::Int(2024),
            Value::Int(2),
            Value::Int(29),
            Value::Int(1),
            Value::Int(2),
            Value::Int(3),
            Value::Int(4),
            static_field(ZONE_OFFSET, "UTC").unwrap(),
        ],
    )
    .unwrap();
    assert_eq!(leap.java_to_string(), "2024-02-29T01:02:03.000000004Z");
    assert_eq!(
        text(call(&leap, "plusYears", &[Value::Long(1)])),
        "2025-02-28T01:02:03.000000004Z"
    );
}

/// Lists what Painless resolves on each class and the runtime lacks:
/// `cargo test -p lucene-painless --lib missing_members -- --ignored --nocapture`.
#[test]
#[ignore]
fn missing_members() {
    let Ok(list) = std::fs::read_to_string("/tmp/members.txt") else {
        return;
    };
    let mut env = NoEnv;
    let mut vm = Vm::new(std::rc::Rc::new(crate::vm::Program::default()), &mut env);
    for line in list.lines() {
        let mut f = line.split(' ');
        let (class, name, arity) = (f.next().unwrap(), f.next().unwrap(), f.next().unwrap());
        let arity: usize = arity.parse().unwrap();
        let recv = sample(class).unwrap();
        let found =
            match crate::api::dynamic::try_call(&mut vm, &recv, name, &vec![Value::Null; arity]) {
                None => false,
                Some(Err(t)) => t.class() != crate::api::UNSUPPORTED,
                Some(Ok(_)) => true,
            };
        if !found {
            println!("MISSING {class} {name}/{arity}");
        }
    }
}
