//! `java.time`: `ZonedDateTime`, `Instant`, `Duration`, `ZoneOffset`,
//! `DayOfWeek`, `Month` and `ChronoUnit`, with Java's results and its
//! exceptions' messages.
//!
//! A date field's values are `ZonedDateTime`s in UTC, so a zone here is a
//! fixed offset: the proleptic ISO calendar needs no zone rules. A member
//! the runtime lacks answers `None`, and the static check keeps a script
//! that uses one off the native path.

use std::any::Any;
use std::cmp::Ordering;

use crate::api::{int_arg, long_arg, npe};
use crate::object::{downcast, JObject};
use crate::value::Value;
use crate::vm::{ApiFn, Throw, Vm};

type R = Result<Value, Throw>;

const NANOS: i64 = 1_000_000_000;
const DAY_NANOS: i64 = 86_400 * NANOS;
const MIN_DAY: i64 = -365_243_219_162;
const MAX_DAY: i64 = 365_241_780_471;
const MIN_SECOND: i64 = -31_557_014_167_219_200;
const MAX_SECOND: i64 = 31_556_889_864_403_199;
const SECONDS_0000_TO_1970: i64 = 62_167_219_200;
const SECONDS_PER_10000_YEARS: i64 = 315_569_520_000;
const MAX_OFFSET: i32 = 18 * 3600;

const ZONED: &str = "java.time.ZonedDateTime";
const INSTANT: &str = "java.time.Instant";
const DURATION: &str = "java.time.Duration";
const ZONE_OFFSET: &str = "java.time.ZoneOffset";
const DAY_OF_WEEK: &str = "java.time.DayOfWeek";
const MONTH: &str = "java.time.Month";
const CHRONO_UNIT: &str = "java.time.temporal.ChronoUnit";
const CHRONO_FIELD: &str = "java.time.temporal.ChronoField";

fn date_time(message: impl Into<String>) -> Throw {
    Throw::new("java.time.DateTimeException", message)
}

fn unsupported_unit(unit: Unit) -> Throw {
    Throw::new(
        "java.time.temporal.UnsupportedTemporalTypeException",
        format!("Unsupported unit: {}", unit.title()),
    )
}

fn overflow() -> Throw {
    Throw::new("java.lang.ArithmeticException", "long overflow")
}

fn bad(v: &Value, to: &str) -> Throw {
    if v.is_null() {
        npe()
    } else {
        Throw::cast_failure(&v.class_name(), to)
    }
}

fn fits(v: i128) -> Result<i64, Throw> {
    i64::try_from(v).map_err(|_| overflow())
}

fn invalid(field: &str, range: &str, value: i64) -> Throw {
    date_time(format!(
        "Invalid value for {field} (valid values {range}): {value}"
    ))
}

fn check(field: &str, range: &str, lo: i64, hi: i64, value: i64) -> Result<(), Throw> {
    if (lo..=hi).contains(&value) {
        Ok(())
    } else {
        Err(invalid(field, range, value))
    }
}

fn check_year(y: i64) -> Result<(), Throw> {
    check(
        "Year",
        "-999999999 - 999999999",
        -999_999_999,
        999_999_999,
        y,
    )
}

fn check_time(h: i64, mi: i64, s: i64, n: i64) -> Result<(), Throw> {
    check("HourOfDay", "0 - 23", 0, 23, h)?;
    check("MinuteOfHour", "0 - 59", 0, 59, mi)?;
    check("SecondOfMinute", "0 - 59", 0, 59, s)?;
    check("NanoOfSecond", "0 - 999999999", 0, 999_999_999, n)
}

// The proleptic ISO calendar.

fn is_leap(y: i64) -> bool {
    y % 4 == 0 && (y % 100 != 0 || y % 400 == 0)
}

fn month_len(y: i64, m: i64) -> i64 {
    match m {
        2 => {
            if is_leap(y) {
                29
            } else {
                28
            }
        }
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// `LocalDate.of`: validated, with Java's messages.
fn date_of(y: i64, m: i64, d: i64) -> Result<i64, Throw> {
    check_year(y)?;
    check("MonthOfYear", "1 - 12", 1, 12, m)?;
    check("DayOfMonth", "1 - 28/31", 1, 31, d)?;
    if d > 28 && d > month_len(y, m) {
        return Err(date_time(if d == 29 {
            format!("Invalid date 'February 29' as '{y}' is not a leap year")
        } else {
            format!("Invalid date '{} {d}'", Month(m as u8 - 1).name())
        }));
    }
    Ok(days_from_civil(y, m, d))
}

fn year_day_of(days: i64) -> i64 {
    let (y, _, _) = civil_from_days(days);
    days - days_from_civil(y, 1, 1) + 1
}

fn date_string(days: i64) -> String {
    let (y, m, d) = civil_from_days(days);
    let year = if y.abs() < 1000 {
        if y < 0 {
            format!("-{:04}", -y)
        } else {
            format!("{y:04}")
        }
    } else if y > 9999 {
        format!("+{y}")
    } else {
        y.to_string()
    };
    format!("{year}-{m:02}-{d:02}")
}

fn time_string(nod: i64) -> String {
    let (h, mi) = (nod / (3600 * NANOS), nod / (60 * NANOS) % 60);
    let (s, n) = (nod / NANOS % 60, nod % NANOS);
    let mut out = format!("{h:02}:{mi:02}");
    if s > 0 || n > 0 {
        out += &format!(":{s:02}");
        if n > 0 {
            out += &fraction(n);
        }
    }
    out
}

/// `.` and the nanos in groups of three digits, as `LocalTime` prints them.
fn fraction(n: i64) -> String {
    if n % 1_000_000 == 0 {
        format!(".{:03}", n / 1_000_000)
    } else if n % 1000 == 0 {
        format!(".{:06}", n / 1000)
    } else {
        format!(".{n:09}")
    }
}

fn offset_string(off: i32) -> String {
    if off == 0 {
        return "Z".into();
    }
    let abs = off.abs();
    let (h, m, s) = (abs / 3600, abs / 60 % 60, abs % 60);
    let mut out = format!("{}{h:02}:{m:02}", if off < 0 { '-' } else { '+' });
    if s != 0 {
        out += &format!(":{s:02}");
    }
    out
}

/// A unit of `ChronoUnit`, by ordinal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Unit(u8);

const UNIT_NAMES: [&str; 16] = [
    "NANOS",
    "MICROS",
    "MILLIS",
    "SECONDS",
    "MINUTES",
    "HOURS",
    "HALF_DAYS",
    "DAYS",
    "WEEKS",
    "MONTHS",
    "YEARS",
    "DECADES",
    "CENTURIES",
    "MILLENNIA",
    "ERAS",
    "FOREVER",
];

const UNIT_TITLES: [&str; 16] = [
    "Nanos",
    "Micros",
    "Millis",
    "Seconds",
    "Minutes",
    "Hours",
    "HalfDays",
    "Days",
    "Weeks",
    "Months",
    "Years",
    "Decades",
    "Centuries",
    "Millennia",
    "Eras",
    "Forever",
];

const NANOS_U: u8 = 0;
const SECONDS_U: u8 = 3;
const DAYS_U: u8 = 7;

impl Unit {
    fn title(self) -> &'static str {
        UNIT_TITLES[usize::from(self.0)]
    }

    fn duration(self) -> Dur {
        let (secs, nanos) = match self.0 {
            0 => (0, 1),
            1 => (0, 1000),
            2 => (0, 1_000_000),
            3 => (1, 0),
            4 => (60, 0),
            5 => (3600, 0),
            6 => (43_200, 0),
            7 => (86_400, 0),
            8 => (7 * 86_400, 0),
            9 => (31_556_952 / 12, 0),
            10 => (31_556_952, 0),
            11 => (315_569_520, 0),
            12 => (3_155_695_200, 0),
            13 => (31_556_952_000, 0),
            14 => (31_556_952_000_000_000, 0),
            _ => (i64::MAX, 999_999_999),
        };
        Dur { secs, nanos }
    }

    fn date_based(self) -> bool {
        (DAYS_U..15).contains(&self.0)
    }
}

/// `Instant`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Inst {
    secs: i64,
    nanos: i32,
}

impl Inst {
    fn of(secs: i64, nanos: i64) -> Result<Inst, Throw> {
        let secs = secs
            .checked_add(nanos.div_euclid(NANOS))
            .ok_or_else(overflow)?;
        if !(MIN_SECOND..=MAX_SECOND).contains(&secs) {
            return Err(date_time("Instant exceeds minimum or maximum instant"));
        }
        Ok(Inst {
            secs,
            nanos: nanos.rem_euclid(NANOS) as i32,
        })
    }

    /// `self` moved by `secs` seconds and `nanos` nanoseconds.
    fn plus(self, secs: i128, nanos: i128) -> Result<Inst, Throw> {
        let secs = i128::from(self.secs) + i128::from(fits(secs)?);
        let total = i128::from(self.nanos) + nanos;
        let secs = fits(secs + total.div_euclid(i128::from(NANOS)))?;
        Inst::of(secs, total.rem_euclid(i128::from(NANOS)) as i64)
    }

    fn total_nanos(self) -> i128 {
        i128::from(self.secs) * i128::from(NANOS) + i128::from(self.nanos)
    }

    fn string(self) -> String {
        let local = |epoch: i64| {
            let days = epoch.div_euclid(86_400);
            let nod = epoch.rem_euclid(86_400) * NANOS;
            (days, nod)
        };
        let mut out = String::new();
        if self.secs >= -SECONDS_0000_TO_1970 {
            let zero = self.secs - SECONDS_PER_10000_YEARS + SECONDS_0000_TO_1970;
            let hi = zero.div_euclid(SECONDS_PER_10000_YEARS) + 1;
            let lo = zero.rem_euclid(SECONDS_PER_10000_YEARS);
            let (days, nod) = local(lo - SECONDS_0000_TO_1970);
            if hi > 0 {
                out += &format!("+{hi}");
            }
            out += &format!("{}T{}", date_string(days), time_string(nod));
            if nod / NANOS % 60 == 0 {
                out += ":00";
            }
        } else {
            let zero = self.secs + SECONDS_0000_TO_1970;
            let hi = zero / SECONDS_PER_10000_YEARS;
            let lo = zero % SECONDS_PER_10000_YEARS;
            let (days, nod) = local(lo - SECONDS_0000_TO_1970);
            let mut text = format!("{}T{}", date_string(days), time_string(nod));
            if nod / NANOS % 60 == 0 {
                text += ":00";
            }
            if hi < 0 {
                if civil_from_days(days).0 == -10_000 {
                    text = format!("{}{}", hi - 1, &text[2..]);
                } else if lo == 0 {
                    text = format!("{hi}{text}");
                } else {
                    text.insert_str(1, &hi.abs().to_string());
                }
            }
            out += &text;
        }
        if self.nanos != 0 {
            out += &fraction(i64::from(self.nanos));
        }
        out.push('Z');
        out
    }
}

/// `Duration`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dur {
    secs: i64,
    nanos: i32,
}

impl Dur {
    const ZERO: Dur = Dur { secs: 0, nanos: 0 };

    /// `Duration.ofSeconds(secs, nanoAdjustment)`.
    fn of(secs: i128, nano_adjustment: i128) -> Result<Dur, Throw> {
        let n = i128::from(NANOS);
        Ok(Dur {
            secs: fits(secs + nano_adjustment.div_euclid(n))?,
            nanos: nano_adjustment.rem_euclid(n) as i32,
        })
    }

    fn total_nanos(self) -> i128 {
        i128::from(self.secs) * i128::from(NANOS) + i128::from(self.nanos)
    }

    /// `Duration` of `total` nanoseconds that must fit its seconds.
    fn of_total(total: i128, show: impl FnOnce() -> String) -> Result<Dur, Throw> {
        let n = i128::from(NANOS);
        i64::try_from(total.div_euclid(n))
            .map(|secs| Dur {
                secs,
                nanos: total.rem_euclid(n) as i32,
            })
            .map_err(|_| {
                Throw::new(
                    "java.lang.ArithmeticException",
                    format!("Exceeds capacity of Duration: {}", show()),
                )
            })
    }

    fn plus(self, secs: i128, nanos: i128) -> Result<Dur, Throw> {
        if secs == 0 && nanos == 0 {
            return Ok(self);
        }
        let n = i128::from(NANOS);
        let epoch = fits(i128::from(self.secs) + i128::from(fits(secs)?))?;
        let epoch = fits(i128::from(epoch) + nanos / n)?;
        Dur::of(i128::from(epoch), i128::from(self.nanos) + nanos % n)
    }

    fn string(self) -> String {
        if self == Dur::ZERO {
            return "PT0S".into();
        }
        let total = if self.secs < 0 && self.nanos > 0 {
            self.secs + 1
        } else {
            self.secs
        };
        let (hours, minutes, secs) = (total / 3600, (total % 3600) / 60, total % 60);
        let mut out = String::from("PT");
        if hours != 0 {
            out += &format!("{hours}H");
        }
        if minutes != 0 {
            out += &format!("{minutes}M");
        }
        if secs == 0 && self.nanos == 0 && out.len() > 2 {
            return out;
        }
        if self.secs < 0 && self.nanos > 0 {
            if secs == 0 {
                out += "-0";
            } else {
                out += &secs.to_string();
            }
        } else {
            out += &secs.to_string();
        }
        if self.nanos > 0 {
            let pos = out.len();
            let n = i64::from(self.nanos);
            out += &(if self.secs < 0 {
                2 * NANOS - n
            } else {
                n + NANOS
            })
            .to_string();
            while out.ends_with('0') {
                out.pop();
            }
            out.replace_range(pos..=pos, ".");
        }
        out.push('S');
        out
    }

    fn to_nanos(self) -> Result<i64, Throw> {
        let (s, n) = self.adjusted();
        s.checked_mul(NANOS)
            .and_then(|x| x.checked_add(n))
            .ok_or_else(overflow)
    }

    fn to_millis(self) -> Result<i64, Throw> {
        let (s, n) = self.adjusted();
        s.checked_mul(1000)
            .and_then(|x| x.checked_add(n / 1_000_000))
            .ok_or_else(overflow)
    }

    /// Seconds and nanos with the nanos carrying the sign, as `toNanos` and
    /// `toMillis` read them.
    fn adjusted(self) -> (i64, i64) {
        if self.secs < 0 {
            (self.secs.wrapping_add(1), i64::from(self.nanos) - NANOS)
        } else {
            (self.secs, i64::from(self.nanos))
        }
    }

    /// The seconds with nine decimals, as `toBigDecimalSeconds` prints.
    fn decimal(total: i128) -> String {
        let n = i128::from(NANOS);
        format!(
            "{}{}.{:09}",
            if total < 0 { "-" } else { "" },
            total.abs() / n,
            total.abs() % n
        )
    }
}

/// `ZonedDateTime` in a fixed offset: the local date (epoch day) and
/// nanosecond of day, and the offset in seconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Zoned {
    days: i64,
    nod: i64,
    off: i32,
}

impl Zoned {
    fn make(days: i64, nod: i64, off: i32) -> Result<Zoned, Throw> {
        if !(MIN_DAY..=MAX_DAY).contains(&days) {
            return Err(invalid("EpochDay", "-365243219162 - 365241780471", days));
        }
        Ok(Zoned { days, nod, off })
    }

    fn of_instant(i: Inst, off: i32) -> Result<Zoned, Throw> {
        let local = i128::from(i.secs) + i128::from(off);
        let days = local.div_euclid(86_400);
        let sod = local.rem_euclid(86_400) as i64;
        Zoned::make(
            i64::try_from(days).map_err(|_| overflow())?,
            sod * NANOS + i64::from(i.nanos),
            off,
        )
    }

    fn instant(self) -> Result<Inst, Throw> {
        let secs =
            i128::from(self.days) * 86_400 + i128::from(self.nod / NANOS) - i128::from(self.off);
        Inst::of(fits(secs)?, self.nod % NANOS)
    }

    fn epoch_second(self) -> i64 {
        self.days * 86_400 + self.nod / NANOS - i64::from(self.off)
    }

    fn nano(self) -> i64 {
        self.nod % NANOS
    }

    fn ymd(self) -> (i64, i64, i64) {
        civil_from_days(self.days)
    }

    fn day_of_week(self) -> DayOfWeek {
        DayOfWeek((self.days + 3).rem_euclid(7) as u8)
    }

    fn plus_nanos(self, n: i128) -> Result<Zoned, Throw> {
        if n == 0 {
            return Ok(self);
        }
        let total = i128::from(self.nod) + n;
        let day = i128::from(DAY_NANOS);
        let days = i128::from(self.days) + total.div_euclid(day);
        Zoned::make(fits(days)?, total.rem_euclid(day) as i64, self.off)
    }

    fn plus_days(self, n: i64) -> Result<Zoned, Throw> {
        if n == 0 {
            return Ok(self);
        }
        Zoned::make(
            self.days.checked_add(n).ok_or_else(overflow)?,
            self.nod,
            self.off,
        )
    }

    fn with_date(self, days: i64) -> Result<Zoned, Throw> {
        Zoned::make(days, self.nod, self.off)
    }

    fn plus_months(self, n: i64) -> Result<Zoned, Throw> {
        if n == 0 {
            return Ok(self);
        }
        let (y, m, d) = self.ymd();
        let count = y.wrapping_mul(12).wrapping_add(m - 1).wrapping_add(n);
        self.resolve_year_month(count.div_euclid(12), count.rem_euclid(12) + 1, d)
    }

    fn plus_years(self, n: i64) -> Result<Zoned, Throw> {
        if n == 0 {
            return Ok(self);
        }
        let (y, m, d) = self.ymd();
        self.resolve_year_month(y.wrapping_add(n), m, d)
    }

    /// `LocalDate.resolvePreviousValid`.
    fn resolve_year_month(self, y: i64, m: i64, d: i64) -> Result<Zoned, Throw> {
        check_year(y)?;
        self.with_date(days_from_civil(y, m, d.min(month_len(y, m))))
    }

    fn plus_era(self, n: i64) -> Result<Zoned, Throw> {
        let (y, _, _) = self.ymd();
        let era = i64::from(y >= 1);
        let to = era.checked_add(n).ok_or_else(overflow)?;
        check("Era", "0 - 1", 0, 1, to)?;
        if to == era {
            return Ok(self);
        }
        self.with_year(1 - y)
    }

    fn with_year(self, y: i64) -> Result<Zoned, Throw> {
        let (cur, m, d) = self.ymd();
        if cur == y {
            return Ok(self);
        }
        self.resolve_year_month(y, m, d)
    }

    fn plus_unit(self, n: i64, unit: Unit) -> Result<Zoned, Throw> {
        let times = |k: i64| n.checked_mul(k).ok_or_else(overflow);
        match unit.0 {
            0..=6 => {
                let per: i128 = match unit.0 {
                    0 => 1,
                    1 => 1000,
                    2 => 1_000_000,
                    3 => i128::from(NANOS),
                    4 => 60 * i128::from(NANOS),
                    5 => 3600 * i128::from(NANOS),
                    _ => 43_200 * i128::from(NANOS),
                };
                self.plus_nanos(i128::from(n) * per)
            }
            7 => self.plus_days(n),
            8 => self.plus_days(times(7)?),
            9 => self.plus_months(n),
            10 => self.plus_years(n),
            11 => self.plus_years(times(10)?),
            12 => self.plus_years(times(100)?),
            13 => self.plus_years(times(1000)?),
            14 => self.plus_era(n),
            _ => Err(unsupported_unit(unit)),
        }
    }

    fn minus_unit(self, n: i64, unit: Unit) -> Result<Zoned, Throw> {
        if n == i64::MIN {
            self.plus_unit(i64::MAX, unit)?.plus_unit(1, unit)
        } else {
            self.plus_unit(-n, unit)
        }
    }

    fn plus_amount(self, d: Dur, sign: i64) -> Result<Zoned, Throw> {
        let mut z = self;
        if d.secs != 0 {
            z = if sign > 0 {
                z.plus_unit(d.secs, Unit(SECONDS_U))?
            } else {
                z.minus_unit(d.secs, Unit(SECONDS_U))?
            };
        }
        if d.nanos != 0 {
            z = if sign > 0 {
                z.plus_unit(i64::from(d.nanos), Unit(NANOS_U))?
            } else {
                z.minus_unit(i64::from(d.nanos), Unit(NANOS_U))?
            };
        }
        Ok(z)
    }

    fn truncated(self, unit: Unit) -> Result<Zoned, Throw> {
        if unit.0 == NANOS_U {
            return Ok(self);
        }
        let dur = truncation_nanos(unit)?;
        Ok(Zoned {
            nod: self.nod / dur * dur,
            ..self
        })
    }

    fn with_field(self, name: &str, v: i64) -> Result<Zoned, Throw> {
        let (y, m, d) = self.ymd();
        let nod_of = |h: i64, mi: i64, s: i64, n: i64| (h * 3600 + mi * 60 + s) * NANOS + n;
        let (h, mi, s, n) = (
            self.nod / (3600 * NANOS),
            self.nod / (60 * NANOS) % 60,
            self.nod / NANOS % 60,
            self.nod % NANOS,
        );
        match name {
            "withHour" => {
                check("HourOfDay", "0 - 23", 0, 23, v)?;
                Ok(Zoned {
                    nod: nod_of(v, mi, s, n),
                    ..self
                })
            }
            "withMinute" => {
                check("MinuteOfHour", "0 - 59", 0, 59, v)?;
                Ok(Zoned {
                    nod: nod_of(h, v, s, n),
                    ..self
                })
            }
            "withSecond" => {
                check("SecondOfMinute", "0 - 59", 0, 59, v)?;
                Ok(Zoned {
                    nod: nod_of(h, mi, v, n),
                    ..self
                })
            }
            "withNano" => {
                check("NanoOfSecond", "0 - 999999999", 0, 999_999_999, v)?;
                Ok(Zoned {
                    nod: nod_of(h, mi, s, v),
                    ..self
                })
            }
            "withDayOfMonth" => {
                if d == v {
                    return Ok(self);
                }
                self.with_date(date_of(y, m, v)?)
            }
            "withDayOfYear" => {
                if self.days - days_from_civil(y, 1, 1) + 1 == v {
                    return Ok(self);
                }
                check_year(y)?;
                check("DayOfYear", "1 - 365/366", 1, 366, v)?;
                if v == 366 && !is_leap(y) {
                    return Err(date_time(format!(
                        "Invalid date 'DayOfYear 366' as '{y}' is not a leap year"
                    )));
                }
                self.with_date(days_from_civil(y, 1, 1) + v - 1)
            }
            "withMonth" => {
                if m == v {
                    return Ok(self);
                }
                check("MonthOfYear", "1 - 12", 1, 12, v)?;
                self.resolve_year_month(y, v, d)
            }
            _ => {
                if y != v {
                    check_year(v)?;
                }
                self.with_year(v)
            }
        }
    }

    fn with_offset_same_local(self, off: i32) -> Zoned {
        Zoned { off, ..self }
    }

    /// `until` as `ZonedDateTime` computes it: a date unit on the local
    /// date and time, a time unit on the instants.
    fn until(self, end: Zoned, unit: Unit) -> Result<i64, Throw> {
        let end = Zoned::of_instant(end.instant()?, self.off)?;
        if unit.date_based() {
            let mut end_day = end.days;
            if end_day > self.days && end.nod < self.nod {
                end_day -= 1;
            } else if end_day < self.days && end.nod > self.nod {
                end_day += 1;
            }
            return date_until(self.days, end_day, unit);
        }
        let diff = (i128::from(end.epoch_second()) - i128::from(self.epoch_second()))
            * i128::from(NANOS)
            + i128::from(end.nano() - self.nano());
        match unit.0 {
            0 => fits(diff),
            1..=6 => fits(diff / i128::from(unit.duration().to_nanos()?)),
            _ => Err(unsupported_unit(unit)),
        }
    }

    fn string(self) -> String {
        format!(
            "{}T{}{}",
            date_string(self.days),
            time_string(self.nod),
            offset_string(self.off)
        )
    }

    fn hash(self) -> i32 {
        let (y, m, d) = self.ymd();
        let y = y as i32;
        let date = (y & 0xFFFF_F800_u32 as i32)
            ^ ((y << 11)
                .wrapping_add((m as i32) << 6)
                .wrapping_add(d as i32));
        let time = (self.nod ^ ((self.nod as u64) >> 32) as i64) as i32;
        (date ^ time) ^ self.off ^ self.off.rotate_left(3)
    }
}

/// The nanoseconds `truncatedTo` rounds down to.
fn truncation_nanos(unit: Unit) -> Result<i64, Throw> {
    let d = unit.duration();
    if d.secs > 86_400 {
        return Err(Throw::new(
            "java.time.temporal.UnsupportedTemporalTypeException",
            "Unit is too large to be used for truncation",
        ));
    }
    let nanos = d.to_nanos()?;
    if DAY_NANOS % nanos != 0 {
        return Err(Throw::new(
            "java.time.temporal.UnsupportedTemporalTypeException",
            "Unit must divide into a standard day without remainder",
        ));
    }
    Ok(nanos)
}

/// `LocalDate.until(end, unit)` for a date unit.
fn date_until(start: i64, end: i64, unit: Unit) -> Result<i64, Throw> {
    let months = || {
        let packed = |day: i64| {
            let (y, m, d) = civil_from_days(day);
            (y * 12 + m - 1) * 32 + d
        };
        (packed(end) - packed(start)) / 32
    };
    Ok(match unit.0 {
        7 => end - start,
        8 => (end - start) / 7,
        9 => months(),
        10 => months() / 12,
        11 => months() / 120,
        12 => months() / 1200,
        13 => months() / 12_000,
        14 => i64::from(civil_from_days(end).0 >= 1) - i64::from(civil_from_days(start).0 >= 1),
        _ => return Err(unsupported_unit(unit)),
    })
}

impl Inst {
    /// `Instant.until`.
    fn until(self, end: Inst, unit: Unit) -> Result<i64, Throw> {
        let nanos = end.total_nanos() - self.total_nanos();
        let secs = i128::from(end.secs) - i128::from(self.secs);
        let unit_secs = |n: i128| fits(secs / n);
        match unit.0 {
            0 => fits(nanos),
            1 => fits(nanos).map(|n| n / 1000),
            2 => Ok(fits(end.total_millis() - self.total_millis())?),
            3 => unit_secs(1),
            4 => unit_secs(60),
            5 => unit_secs(3600),
            6 => unit_secs(43_200),
            7 => unit_secs(86_400),
            _ => Err(unsupported_unit(unit)),
        }
    }

    /// `Instant.toEpochMilli` without the overflow check.
    fn total_millis(self) -> i128 {
        i128::from(self.secs) * 1000 + i128::from(self.nanos) / 1_000_000
    }
}

macro_rules! host {
    ($t:ty) => {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn java_equals(&self, other: &dyn JObject) -> bool {
            other.as_any().downcast_ref::<$t>() == Some(self)
        }
    };
}

fn comparable(class: &str, own: &str, temporal: bool) -> bool {
    class == own
        || matches!(class, "java.lang.Object" | "java.lang.Comparable")
        || (temporal
            && matches!(
                class,
                "java.time.temporal.Temporal" | "java.time.temporal.TemporalAccessor"
            ))
}

impl JObject for Zoned {
    fn class_name(&self) -> &str {
        ZONED
    }

    host!(Zoned);

    fn instance_of(&self, class: &str) -> bool {
        comparable(class, ZONED, true) || class == "java.time.chrono.ChronoZonedDateTime"
    }

    fn java_hash(&self) -> i32 {
        self.hash()
    }

    fn java_to_string(&self) -> String {
        self.string()
    }

    fn call(&self, _vm: &mut Vm, this: &Value, name: &str, args: &[Value]) -> Option<R> {
        let z = *self;
        let int = |v: i64| Ok(Value::Int(v as i32));
        let zoned = |r: Result<Zoned, Throw>| r.map(Value::object);
        Some(match (name, args) {
            ("getYear", []) => int(z.ymd().0),
            ("getMonthValue", []) => int(z.ymd().1),
            ("getDayOfMonth", []) => int(z.ymd().2),
            ("getDayOfYear", []) => int(year_day_of(z.days)),
            ("getHour", []) => int(z.nod / (3600 * NANOS)),
            ("getMinute", []) => int(z.nod / (60 * NANOS) % 60),
            ("getSecond", []) => int(z.nod / NANOS % 60),
            ("getNano", []) => int(z.nano()),
            ("getMonth", []) => Ok(month_value(Month(z.ymd().1 as u8 - 1))),
            ("getDayOfWeek" | "getDayOfWeekEnum", []) => Ok(day_value(z.day_of_week())),
            ("getOffset" | "getZone", []) => Ok(Value::object(Offset(z.off))),
            ("toInstant", []) => z.instant().map(Value::object),
            ("toEpochSecond", []) => Ok(Value::Long(z.epoch_second())),
            ("plus", [amount]) => duration_arg(amount).and_then(|a| zoned(z.plus_amount(a, 1))),
            ("minus", [amount]) => duration_arg(amount).and_then(|a| zoned(z.plus_amount(a, -1))),
            ("plus", [n, unit]) => both(n, unit).and_then(|(n, u)| zoned(z.plus_unit(n, u))),
            ("minus", [n, unit]) => both(n, unit).and_then(|(n, u)| zoned(z.minus_unit(n, u))),
            (
                "plusNanos" | "plusSeconds" | "plusMinutes" | "plusHours" | "plusDays"
                | "plusWeeks" | "plusMonths" | "plusYears",
                [n],
            ) => long_arg(n).and_then(|n| zoned(z.plus_unit(n, named_unit(&name[4..])))),
            (
                "minusNanos" | "minusSeconds" | "minusMinutes" | "minusHours" | "minusDays"
                | "minusWeeks" | "minusMonths" | "minusYears",
                [n],
            ) => long_arg(n).and_then(|n| zoned(z.minus_unit(n, named_unit(&name[5..])))),
            (
                "withHour" | "withMinute" | "withSecond" | "withNano" | "withDayOfMonth"
                | "withDayOfYear" | "withMonth" | "withYear",
                [v],
            ) => int_arg(v).and_then(|v| zoned(z.with_field(name, i64::from(v)))),
            ("truncatedTo", [unit]) => unit_arg(unit).and_then(|u| zoned(z.truncated(u))),
            ("withZoneSameInstant", [zone]) => offset_arg(zone)
                .and_then(|o| z.instant().and_then(|i| zoned(Zoned::of_instant(i, o.0)))),
            ("withZoneSameLocal", [zone]) => {
                offset_arg(zone).map(|o| Value::object(z.with_offset_same_local(o.0)))
            }
            (
                "withFixedOffsetZone" | "withEarlierOffsetAtOverlap" | "withLaterOffsetAtOverlap",
                [],
            ) => Ok(Value::object(z)),
            ("isAfter", [o]) => zoned_arg(o).map(|o| Value::Bool(order(z, o) == Ordering::Greater)),
            ("isBefore", [o]) => zoned_arg(o).map(|o| Value::Bool(order(z, o) == Ordering::Less)),
            ("isEqual", [o]) => zoned_arg(o).map(|o| Value::Bool(order(z, o) == Ordering::Equal)),
            ("compareTo", [o]) => zoned_arg(o).map(|o| Value::Int(compare_zoned(z, o))),
            ("until", [end, unit]) => unit_arg(unit)
                .and_then(|u| z.until(end_zoned(end)?, u))
                .map(Value::Long),
            _ => return temporal_call(this, name, args),
        })
    }

    fn compare_to(&self, other: &dyn JObject) -> Option<i32> {
        other
            .as_any()
            .downcast_ref::<Zoned>()
            .map(|o| compare_zoned(*self, *o))
    }
}

/// The instants' order, nanos breaking a tie in a second.
fn order(a: Zoned, b: Zoned) -> Ordering {
    (a.epoch_second(), a.nano()).cmp(&(b.epoch_second(), b.nano()))
}

/// `ChronoZonedDateTime.compareTo`.
fn compare_zoned(a: Zoned, b: Zoned) -> i32 {
    let by = |o: Ordering| o as i32;
    let cmp = by(a.epoch_second().cmp(&b.epoch_second()));
    if cmp != 0 {
        return cmp;
    }
    let cmp = (a.nano() - b.nano()) as i32;
    if cmp != 0 {
        return cmp;
    }
    // The local date-time: year, month and day by difference, then time.
    let ((ay, am, ad), (by_, bm, bd)) = (a.ymd(), b.ymd());
    let cmp = (ay - by_) as i32;
    if cmp != 0 {
        return cmp;
    }
    let cmp = (am - bm) as i32;
    if cmp != 0 {
        return cmp;
    }
    let cmp = (ad - bd) as i32;
    if cmp != 0 {
        return cmp;
    }
    let cmp = by(a.nod.cmp(&b.nod));
    if cmp != 0 {
        return cmp;
    }
    // Zone ids, which are the offsets' text.
    java_str_cmp(&offset_string(a.off), &offset_string(b.off))
}

fn java_str_cmp(a: &str, b: &str) -> i32 {
    crate::api::string::java_compare(a, b)
}

impl JObject for Inst {
    fn class_name(&self) -> &str {
        INSTANT
    }

    host!(Inst);

    fn instance_of(&self, class: &str) -> bool {
        comparable(class, INSTANT, true)
    }

    fn java_hash(&self) -> i32 {
        ((self.secs ^ ((self.secs as u64) >> 32) as i64) as i32)
            .wrapping_add(51_i32.wrapping_mul(self.nanos))
    }

    fn java_to_string(&self) -> String {
        self.string()
    }

    fn call(&self, _vm: &mut Vm, this: &Value, name: &str, args: &[Value]) -> Option<R> {
        let i = *self;
        let inst = |r: Result<Inst, Throw>| r.map(Value::object);
        let n = i128::from(NANOS);
        Some(match (name, args) {
            ("getEpochSecond", []) => Ok(Value::Long(i.secs)),
            ("getNano", []) => Ok(Value::Int(i.nanos)),
            ("toEpochMilli", []) => fits(i.total_millis()).map(Value::Long),
            ("plusSeconds", [v]) => long_arg(v).and_then(|v| inst(i.plus(i128::from(v), 0))),
            ("minusSeconds", [v]) => long_arg(v).and_then(|v| inst(i.plus(-i128::from(v), 0))),
            ("plusMillis", [v]) => {
                long_arg(v).and_then(|v| inst(i.plus(0, i128::from(v) * 1_000_000)))
            }
            ("minusMillis", [v]) => {
                long_arg(v).and_then(|v| inst(i.plus(0, -i128::from(v) * 1_000_000)))
            }
            ("plusNanos", [v]) => long_arg(v).and_then(|v| inst(i.plus(0, i128::from(v)))),
            ("minusNanos", [v]) => long_arg(v).and_then(|v| inst(i.plus(0, -i128::from(v)))),
            ("plus", [a]) => duration_arg(a).and_then(|d| inst(instant_amount(i, d, 1))),
            ("minus", [a]) => duration_arg(a).and_then(|d| inst(instant_amount(i, d, -1))),
            ("plus", [v, u]) => both(v, u).and_then(|(v, u)| inst(instant_plus_unit(i, v, u, 1))),
            ("minus", [v, u]) => both(v, u).and_then(|(v, u)| inst(instant_plus_unit(i, v, u, -1))),
            ("truncatedTo", [u]) => unit_arg(u).and_then(|u| {
                if u.0 == NANOS_U {
                    return Ok(Value::object(i));
                }
                let dur = i128::from(truncation_nanos(u)?);
                let nod = i128::from(i.secs % 86_400) * n + i128::from(i.nanos);
                inst(i.plus(0, nod.div_euclid(dur) * dur - nod))
            }),
            ("atZone", [zone]) => {
                offset_arg(zone).and_then(|o| Zoned::of_instant(i, o.0).map(Value::object))
            }
            ("isAfter", [o]) => instant_arg(o).map(|o| Value::Bool(i.cmp_to(o) > 0)),
            ("isBefore", [o]) => instant_arg(o).map(|o| Value::Bool(i.cmp_to(o) < 0)),
            ("compareTo", [o]) => instant_arg(o).map(|o| Value::Int(i.cmp_to(o))),
            ("until", [end, unit]) => unit_arg(unit)
                .and_then(|u| i.until(end_instant(end)?, u))
                .map(Value::Long),
            _ => return temporal_call(this, name, args),
        })
    }

    fn compare_to(&self, other: &dyn JObject) -> Option<i32> {
        other
            .as_any()
            .downcast_ref::<Inst>()
            .map(|o| self.cmp_to(*o))
    }
}

impl Inst {
    /// `Instant.compareTo`.
    fn cmp_to(self, other: Inst) -> i32 {
        match self.secs.cmp(&other.secs) {
            Ordering::Equal => self.nanos - other.nanos,
            o => o as i32,
        }
    }
}

fn instant_amount(i: Inst, d: Dur, sign: i128) -> Result<Inst, Throw> {
    let mut i = i;
    if d.secs != 0 {
        i = instant_plus_unit(i, d.secs, Unit(SECONDS_U), sign)?;
    }
    if d.nanos != 0 {
        i = instant_plus_unit(i, i64::from(d.nanos), Unit(NANOS_U), sign)?;
    }
    Ok(i)
}

fn instant_plus_unit(i: Inst, n: i64, unit: Unit, sign: i128) -> Result<Inst, Throw> {
    let n = i128::from(n) * sign;
    match unit.0 {
        0 => i.plus(0, n),
        1 => i.plus(n / 1_000_000, n % 1_000_000 * 1000),
        2 => i.plus(0, n * 1_000_000),
        3 => i.plus(n, 0),
        4 => i.plus(n * 60, 0),
        5 => i.plus(n * 3600, 0),
        6 => i.plus(n * 43_200, 0),
        7 => i.plus(n * 86_400, 0),
        _ => Err(unsupported_unit(unit)),
    }
}

impl JObject for Dur {
    fn class_name(&self) -> &str {
        DURATION
    }

    host!(Dur);

    fn instance_of(&self, class: &str) -> bool {
        comparable(class, DURATION, false) || class == "java.time.temporal.TemporalAmount"
    }

    fn java_hash(&self) -> i32 {
        ((self.secs ^ ((self.secs as u64) >> 32) as i64) as i32)
            .wrapping_add(51_i32.wrapping_mul(self.nanos))
    }

    fn java_to_string(&self) -> String {
        self.string()
    }

    fn call(&self, _vm: &mut Vm, _this: &Value, name: &str, args: &[Value]) -> Option<R> {
        let d = *self;
        let dur = |r: Result<Dur, Throw>| r.map(Value::object);
        Some(match (name, args) {
            ("getSeconds", []) => Ok(Value::Long(d.secs)),
            ("getNano", []) => Ok(Value::Int(d.nanos)),
            ("isNegative", []) => Ok(Value::Bool(d.secs < 0)),
            ("isZero", []) => Ok(Value::Bool(d == Dur::ZERO)),
            ("toDays", []) => Ok(Value::Long(d.secs / 86_400)),
            ("toHours", []) => Ok(Value::Long(d.secs / 3600)),
            ("toMinutes", []) => Ok(Value::Long(d.secs / 60)),
            ("toMillis", []) => d.to_millis().map(Value::Long),
            ("toNanos", []) => d.to_nanos().map(Value::Long),
            ("negated", []) => dur(scaled(d, -1)),
            ("abs", []) => dur(if d.secs < 0 { scaled(d, -1) } else { Ok(d) }),
            ("multipliedBy", [v]) => long_arg(v).and_then(|v| dur(scaled(d, v))),
            ("dividedBy", [v]) => long_arg(v).and_then(|v| match v {
                0 => Err(Throw::new(
                    "java.lang.ArithmeticException",
                    "Cannot divide by zero",
                )),
                1 => Ok(Value::object(d)),
                v => {
                    let q = d.total_nanos() / i128::from(v);
                    dur(Dur::of_total(q, || Dur::decimal(q)))
                }
            }),
            ("plus", [a]) => {
                duration_arg(a).and_then(|a| dur(d.plus(i128::from(a.secs), i128::from(a.nanos))))
            }
            ("minus", [a]) => {
                duration_arg(a).and_then(|a| dur(d.plus(-i128::from(a.secs), -i128::from(a.nanos))))
            }
            ("plus", [v, u]) => {
                both(v, u).and_then(|(v, u)| dur(duration_plus_unit(d, i128::from(v), u)))
            }
            ("minus", [v, u]) => {
                both(v, u).and_then(|(v, u)| dur(duration_plus_unit(d, -i128::from(v), u)))
            }
            (
                "plusDays" | "plusHours" | "plusMinutes" | "plusSeconds" | "plusMillis"
                | "plusNanos",
                [v],
            ) => long_arg(v).and_then(|v| dur(duration_plus_named(d, &name[4..], i128::from(v)))),
            (
                "minusDays" | "minusHours" | "minusMinutes" | "minusSeconds" | "minusMillis"
                | "minusNanos",
                [v],
            ) => long_arg(v).and_then(|v| dur(duration_plus_named(d, &name[5..], -i128::from(v)))),
            ("withSeconds", [v]) => long_arg(v).map(|v| Value::object(Dur { secs: v, ..d })),
            ("withNanos", [v]) => int_arg(v).and_then(|v| {
                check(
                    "NanoOfSecond",
                    "0 - 999999999",
                    0,
                    999_999_999,
                    i64::from(v),
                )?;
                Ok(Value::object(Dur { nanos: v, ..d }))
            }),
            ("compareTo", [o]) => duration_arg(o).map(|o| Value::Int(d.cmp_to(o))),
            ("get", [u]) => unit_arg(u).and_then(|u| match u.0 {
                SECONDS_U => Ok(Value::Long(d.secs)),
                NANOS_U => Ok(Value::Long(i64::from(d.nanos))),
                _ => Err(unsupported_unit(u)),
            }),
            _ => return None,
        })
    }

    fn compare_to(&self, other: &dyn JObject) -> Option<i32> {
        other
            .as_any()
            .downcast_ref::<Dur>()
            .map(|o| self.cmp_to(*o))
    }
}

impl Dur {
    /// `Duration.compareTo`.
    fn cmp_to(self, other: Dur) -> i32 {
        match self.secs.cmp(&other.secs) {
            Ordering::Equal => self.nanos - other.nanos,
            o => o as i32,
        }
    }
}

/// `Duration.multipliedBy`.
fn scaled(d: Dur, by: i64) -> Result<Dur, Throw> {
    match by {
        0 => Ok(Dur::ZERO),
        1 => Ok(d),
        by => {
            let total = d.total_nanos() * i128::from(by);
            Dur::of_total(total, || Dur::decimal(total))
        }
    }
}

fn duration_plus_unit(d: Dur, n: i128, unit: Unit) -> Result<Dur, Throw> {
    match unit.0 {
        DAYS_U => d.plus(n * 86_400, 0),
        0 => d.plus(0, n),
        1 => d.plus(n / (1_000_000 * 1000) * 1000, n % (1_000_000 * 1000) * 1000),
        2 => d.plus(0, n * 1_000_000),
        u if u < DAYS_U => d.plus(n * i128::from(unit.duration().secs), 0),
        _ => Err(Throw::new(
            "java.time.temporal.UnsupportedTemporalTypeException",
            "Unit must not have an estimated duration",
        )),
    }
}

fn duration_plus_named(d: Dur, unit: &str, n: i128) -> Result<Dur, Throw> {
    match unit {
        "Days" => d.plus(n * 86_400, 0),
        "Hours" => d.plus(n * 3600, 0),
        "Minutes" => d.plus(n * 60, 0),
        "Seconds" => d.plus(n, 0),
        "Millis" => d.plus(n / 1000, n % 1000 * 1_000_000),
        _ => d.plus(0, n),
    }
}

/// `ZoneOffset`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Offset(i32);

impl JObject for Offset {
    fn class_name(&self) -> &str {
        ZONE_OFFSET
    }

    host!(Offset);

    fn instance_of(&self, class: &str) -> bool {
        comparable(class, ZONE_OFFSET, false)
            || matches!(
                class,
                "java.time.ZoneId" | "java.time.temporal.TemporalAccessor"
            )
    }

    fn java_hash(&self) -> i32 {
        self.0
    }

    fn java_to_string(&self) -> String {
        offset_string(self.0)
    }

    fn call(&self, _vm: &mut Vm, this: &Value, name: &str, args: &[Value]) -> Option<R> {
        Some(match (name, args) {
            ("getTotalSeconds", []) => Ok(Value::Int(self.0)),
            ("getId", []) => Ok(Value::string(offset_string(self.0))),
            ("normalized", []) => Ok(Value::object(*self)),
            ("compareTo", [o]) => offset_arg(o).map(|o| Value::Int(o.0 - self.0)),
            _ => return temporal_call(this, name, args),
        })
    }

    fn compare_to(&self, other: &dyn JObject) -> Option<i32> {
        other
            .as_any()
            .downcast_ref::<Offset>()
            .map(|o| o.0 - self.0)
    }
}

/// `DayOfWeek`, by ordinal (Monday is 0).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DayOfWeek(u8);

const DAY_NAMES: [&str; 7] = [
    "MONDAY",
    "TUESDAY",
    "WEDNESDAY",
    "THURSDAY",
    "FRIDAY",
    "SATURDAY",
    "SUNDAY",
];

const MONTH_NAMES: [&str; 12] = [
    "JANUARY",
    "FEBRUARY",
    "MARCH",
    "APRIL",
    "MAY",
    "JUNE",
    "JULY",
    "AUGUST",
    "SEPTEMBER",
    "OCTOBER",
    "NOVEMBER",
    "DECEMBER",
];

impl DayOfWeek {
    fn name(self) -> &'static str {
        DAY_NAMES[usize::from(self.0)]
    }
}

impl JObject for DayOfWeek {
    fn class_name(&self) -> &str {
        DAY_OF_WEEK
    }

    host!(DayOfWeek);

    fn instance_of(&self, class: &str) -> bool {
        comparable(class, DAY_OF_WEEK, false)
            || matches!(
                class,
                "java.lang.Enum" | "java.time.temporal.TemporalAccessor"
            )
    }

    fn java_hash(&self) -> i32 {
        i32::from(self.0) + 1
    }

    fn java_to_string(&self) -> String {
        self.name().into()
    }

    fn call(&self, _vm: &mut Vm, this: &Value, name: &str, args: &[Value]) -> Option<R> {
        let o = i64::from(self.0);
        Some(match (name, args) {
            ("getValue", []) => Ok(Value::Int(o as i32 + 1)),
            ("name", []) => Ok(Value::str(self.name())),
            ("ordinal", []) => Ok(Value::Int(o as i32)),
            ("plus", [n]) => long_arg(n).map(|n| day_value(DayOfWeek(((o + n % 7 + 7) % 7) as u8))),
            ("minus", [n]) => {
                long_arg(n).map(|n| day_value(DayOfWeek(((o + -(n % 7) + 7) % 7) as u8)))
            }
            ("compareTo", [e]) => enum_ordinal(e, DAY_OF_WEEK).map(|p| Value::Int(o as i32 - p)),
            _ => return temporal_call(this, name, args),
        })
    }

    fn compare_to(&self, other: &dyn JObject) -> Option<i32> {
        other
            .as_any()
            .downcast_ref::<DayOfWeek>()
            .map(|p| i32::from(self.0) - i32::from(p.0))
    }
}

/// `Month`, by ordinal (January is 0).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Month(u8);

impl Month {
    fn name(self) -> &'static str {
        MONTH_NAMES[usize::from(self.0)]
    }

    fn length(self, leap: bool) -> i32 {
        month_len(if leap { 2000 } else { 2001 }, i64::from(self.0) + 1) as i32
    }
}

impl JObject for Month {
    fn class_name(&self) -> &str {
        MONTH
    }

    host!(Month);

    fn instance_of(&self, class: &str) -> bool {
        comparable(class, MONTH, false)
            || matches!(
                class,
                "java.lang.Enum" | "java.time.temporal.TemporalAccessor"
            )
    }

    fn java_hash(&self) -> i32 {
        i32::from(self.0) + 101
    }

    fn java_to_string(&self) -> String {
        self.name().into()
    }

    fn call(&self, _vm: &mut Vm, this: &Value, name: &str, args: &[Value]) -> Option<R> {
        let o = i64::from(self.0);
        Some(match (name, args) {
            ("getValue", []) => Ok(Value::Int(o as i32 + 1)),
            ("name", []) => Ok(Value::str(self.name())),
            ("ordinal", []) => Ok(Value::Int(o as i32)),
            ("plus", [n]) => {
                long_arg(n).map(|n| month_value(Month(((o + n % 12 + 12) % 12) as u8)))
            }
            ("minus", [n]) => {
                long_arg(n).map(|n| month_value(Month(((o + -(n % 12) + 12) % 12) as u8)))
            }
            ("length", [leap]) => crate::api::bool_arg(leap).map(|l| Value::Int(self.length(l))),
            ("maxLength", []) => Ok(Value::Int(self.length(true))),
            ("minLength", []) => Ok(Value::Int(self.length(false))),
            ("firstMonthOfQuarter", []) => Ok(month_value(Month((self.0 / 3) * 3))),
            ("firstDayOfYear", [leap]) => crate::api::bool_arg(leap).map(|l| {
                let leap = i32::from(l);
                Value::Int(match self.0 {
                    0 => 1,
                    1 => 32,
                    2 => 60 + leap,
                    3 => 91 + leap,
                    4 => 121 + leap,
                    5 => 152 + leap,
                    6 => 182 + leap,
                    7 => 213 + leap,
                    8 => 244 + leap,
                    9 => 274 + leap,
                    10 => 305 + leap,
                    _ => 335 + leap,
                })
            }),
            ("compareTo", [e]) => enum_ordinal(e, MONTH).map(|p| Value::Int(o as i32 - p)),
            _ => return temporal_call(this, name, args),
        })
    }

    fn compare_to(&self, other: &dyn JObject) -> Option<i32> {
        other
            .as_any()
            .downcast_ref::<Month>()
            .map(|p| i32::from(self.0) - i32::from(p.0))
    }
}

impl JObject for Unit {
    fn class_name(&self) -> &str {
        CHRONO_UNIT
    }

    host!(Unit);

    fn instance_of(&self, class: &str) -> bool {
        comparable(class, CHRONO_UNIT, false)
            || matches!(class, "java.lang.Enum" | "java.time.temporal.TemporalUnit")
    }

    fn java_hash(&self) -> i32 {
        i32::from(self.0) + 201
    }

    fn java_to_string(&self) -> String {
        self.title().into()
    }

    fn call(&self, _vm: &mut Vm, _this: &Value, name: &str, args: &[Value]) -> Option<R> {
        let u = *self;
        Some(match (name, args) {
            ("name", []) => Ok(Value::str(UNIT_NAMES[usize::from(u.0)])),
            ("ordinal", []) => Ok(Value::Int(i32::from(u.0))),
            ("getDuration", []) => Ok(Value::object(u.duration())),
            ("isDateBased", []) => Ok(Value::Bool(u.date_based())),
            ("isTimeBased", []) => Ok(Value::Bool(u.0 < DAYS_U)),
            ("isDurationEstimated", []) => Ok(Value::Bool(u.0 >= DAYS_U)),
            ("between", [a, b]) => temporal_until(a, b, u).map(Value::Long),
            ("compareTo", [e]) => {
                enum_ordinal(e, CHRONO_UNIT).map(|p| Value::Int(i32::from(u.0) - p))
            }
            _ => return None,
        })
    }

    fn compare_to(&self, other: &dyn JObject) -> Option<i32> {
        other
            .as_any()
            .downcast_ref::<Unit>()
            .map(|p| i32::from(self.0) - i32::from(p.0))
    }
}

/// A `ChronoField`, by ordinal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Field(u8);

const FIELD_NAMES: [&str; 30] = [
    "NANO_OF_SECOND",
    "NANO_OF_DAY",
    "MICRO_OF_SECOND",
    "MICRO_OF_DAY",
    "MILLI_OF_SECOND",
    "MILLI_OF_DAY",
    "SECOND_OF_MINUTE",
    "SECOND_OF_DAY",
    "MINUTE_OF_HOUR",
    "MINUTE_OF_DAY",
    "HOUR_OF_AMPM",
    "CLOCK_HOUR_OF_AMPM",
    "HOUR_OF_DAY",
    "CLOCK_HOUR_OF_DAY",
    "AMPM_OF_DAY",
    "DAY_OF_WEEK",
    "ALIGNED_DAY_OF_WEEK_IN_MONTH",
    "ALIGNED_DAY_OF_WEEK_IN_YEAR",
    "DAY_OF_MONTH",
    "DAY_OF_YEAR",
    "EPOCH_DAY",
    "ALIGNED_WEEK_OF_MONTH",
    "ALIGNED_WEEK_OF_YEAR",
    "MONTH_OF_YEAR",
    "PROLEPTIC_MONTH",
    "YEAR_OF_ERA",
    "YEAR",
    "ERA",
    "INSTANT_SECONDS",
    "OFFSET_SECONDS",
];

const FIELD_TITLES: [&str; 30] = [
    "NanoOfSecond",
    "NanoOfDay",
    "MicroOfSecond",
    "MicroOfDay",
    "MilliOfSecond",
    "MilliOfDay",
    "SecondOfMinute",
    "SecondOfDay",
    "MinuteOfHour",
    "MinuteOfDay",
    "HourOfAmPm",
    "ClockHourOfAmPm",
    "HourOfDay",
    "ClockHourOfDay",
    "AmPmOfDay",
    "DayOfWeek",
    "AlignedDayOfWeekInMonth",
    "AlignedDayOfWeekInYear",
    "DayOfMonth",
    "DayOfYear",
    "EpochDay",
    "AlignedWeekOfMonth",
    "AlignedWeekOfYear",
    "MonthOfYear",
    "ProlepticMonth",
    "YearOfEra",
    "Year",
    "Era",
    "InstantSeconds",
    "OffsetSeconds",
];

/// Each field's valid range: smallest and largest minimum and maximum, as
/// `ValueRange` holds them.
const FIELD_RANGES: [(i64, i64, i64, i64); 30] = [
    (0, 0, 999_999_999, 999_999_999),
    (0, 0, 86_400 * NANOS - 1, 86_400 * NANOS - 1),
    (0, 0, 999_999, 999_999),
    (0, 0, 86_400 * 1_000_000 - 1, 86_400 * 1_000_000 - 1),
    (0, 0, 999, 999),
    (0, 0, 86_400 * 1000 - 1, 86_400 * 1000 - 1),
    (0, 0, 59, 59),
    (0, 0, 86_399, 86_399),
    (0, 0, 59, 59),
    (0, 0, 1439, 1439),
    (0, 0, 11, 11),
    (1, 1, 12, 12),
    (0, 0, 23, 23),
    (1, 1, 24, 24),
    (0, 0, 1, 1),
    (1, 1, 7, 7),
    (1, 1, 7, 7),
    (1, 1, 7, 7),
    (1, 1, 28, 31),
    (1, 1, 365, 366),
    (MIN_DAY, MIN_DAY, MAX_DAY, MAX_DAY),
    (1, 1, 4, 5),
    (1, 1, 53, 53),
    (1, 1, 12, 12),
    (
        -11_999_999_988,
        -11_999_999_988,
        11_999_999_999,
        11_999_999_999,
    ),
    (1, 1, 999_999_999, 1_000_000_000),
    (-999_999_999, -999_999_999, 999_999_999, 999_999_999),
    (0, 0, 1, 1),
    (i64::MIN, i64::MIN, i64::MAX, i64::MAX),
    (-64_800, -64_800, 64_800, 64_800),
];

const DAY_OF_WEEK_F: u8 = 15;
const MONTH_OF_YEAR_F: u8 = 23;
const INSTANT_SECONDS_F: u8 = 28;
const OFFSET_SECONDS_F: u8 = 29;

impl Field {
    fn title(self) -> &'static str {
        FIELD_TITLES[usize::from(self.0)]
    }

    fn range_text(self) -> String {
        let (a, b, c, d) = FIELD_RANGES[usize::from(self.0)];
        let both = |x: i64, y: i64| {
            if x == y {
                x.to_string()
            } else {
                format!("{x}/{y}")
            }
        };
        format!("{} - {}", both(a, b), both(c, d))
    }

    fn check(self, v: i64) -> Result<i64, Throw> {
        let (lo, _, _, hi) = FIELD_RANGES[usize::from(self.0)];
        if (lo..=hi).contains(&v) {
            Ok(v)
        } else {
            Err(invalid(self.title(), &self.range_text(), v))
        }
    }

    fn unsupported(self) -> Throw {
        Throw::new(
            "java.time.temporal.UnsupportedTemporalTypeException",
            format!("Unsupported field: {}", self.title()),
        )
    }

    fn use_get_long(self) -> Throw {
        Throw::new(
            "java.time.temporal.UnsupportedTemporalTypeException",
            format!(
                "Invalid field '{}' for get() method, use getLong() instead",
                self.title()
            ),
        )
    }
}

impl JObject for Field {
    fn class_name(&self) -> &str {
        CHRONO_FIELD
    }

    host!(Field);

    fn instance_of(&self, class: &str) -> bool {
        comparable(class, CHRONO_FIELD, false)
            || matches!(class, "java.lang.Enum" | "java.time.temporal.TemporalField")
    }

    fn java_hash(&self) -> i32 {
        i32::from(self.0) + 301
    }

    fn java_to_string(&self) -> String {
        self.title().into()
    }

    fn call(&self, _vm: &mut Vm, _this: &Value, name: &str, args: &[Value]) -> Option<R> {
        let f = *self;
        Some(match (name, args) {
            ("name", []) => Ok(Value::str(FIELD_NAMES[usize::from(f.0)])),
            ("ordinal", []) => Ok(Value::Int(i32::from(f.0))),
            ("isDateBased", []) => Ok(Value::Bool((DAY_OF_WEEK_F..=27).contains(&f.0))),
            ("isTimeBased", []) => Ok(Value::Bool(f.0 < DAY_OF_WEEK_F)),
            ("checkValidValue", [v]) => long_arg(v).and_then(|v| f.check(v)).map(Value::Long),
            ("checkValidIntValue", [v]) => long_arg(v).and_then(|v| {
                let v = f.check(v)?;
                i32::try_from(v)
                    .map(Value::Int)
                    .map_err(|_| date_time(format!("Invalid value for {}: {v}", f.title())))
            }),
            ("getFrom", [t]) => temporal_long(t, f).map(Value::Long),
            ("isSupportedBy", [t]) => temporal_supports(t, f).map(Value::Bool),
            ("compareTo", [e]) => field_arg(e).map(|o| Value::Int(i32::from(f.0) - i32::from(o.0))),
            _ => return None,
        })
    }

    fn compare_to(&self, other: &dyn JObject) -> Option<i32> {
        other
            .as_any()
            .downcast_ref::<Field>()
            .map(|p| i32::from(self.0) - i32::from(p.0))
    }
}

fn field_arg(v: &Value) -> Result<Field, Throw> {
    downcast::<Field>(v)
        .copied()
        .ok_or_else(|| bad(v, "java.time.temporal.TemporalField"))
}

/// `temporal.getLong(field)` for the temporals the runtime has.
fn temporal_long(t: &Value, f: Field) -> Result<i64, Throw> {
    if let Some(z) = downcast::<Zoned>(t) {
        return Ok(z.chrono(f));
    }
    if let Some(i) = downcast::<Inst>(t) {
        return i.chrono(f);
    }
    if let Some(d) = downcast::<DayOfWeek>(t) {
        return single_field(f, DAY_OF_WEEK_F, i64::from(d.0) + 1);
    }
    if let Some(m) = downcast::<Month>(t) {
        return single_field(f, MONTH_OF_YEAR_F, i64::from(m.0) + 1);
    }
    if let Some(o) = downcast::<Offset>(t) {
        return single_field(f, OFFSET_SECONDS_F, i64::from(o.0));
    }
    Err(bad(t, "java.time.temporal.TemporalAccessor"))
}

fn single_field(f: Field, only: u8, v: i64) -> Result<i64, Throw> {
    if f.0 == only {
        Ok(v)
    } else {
        Err(f.unsupported())
    }
}

fn temporal_supports(t: &Value, f: Field) -> Result<bool, Throw> {
    if downcast::<Zoned>(t).is_some() {
        return Ok(true);
    }
    if downcast::<Inst>(t).is_some() {
        return Ok(matches!(f.0, 0 | 2 | 4 | INSTANT_SECONDS_F));
    }
    if downcast::<DayOfWeek>(t).is_some() {
        return Ok(f.0 == DAY_OF_WEEK_F);
    }
    if downcast::<Month>(t).is_some() {
        return Ok(f.0 == MONTH_OF_YEAR_F);
    }
    if downcast::<Offset>(t).is_some() {
        return Ok(f.0 == OFFSET_SECONDS_F);
    }
    Err(bad(t, "java.time.temporal.TemporalAccessor"))
}

impl Zoned {
    /// `getLong(field)`.
    fn chrono(self, f: Field) -> i64 {
        let (y, m, d) = self.ymd();
        let doy = year_day_of(self.days);
        let hour = self.nod / (3600 * NANOS);
        let nano = self.nano();
        match f.0 {
            0 => nano,
            1 => self.nod,
            2 => nano / 1000,
            3 => self.nod / 1000,
            4 => nano / 1_000_000,
            5 => self.nod / 1_000_000,
            6 => self.nod / NANOS % 60,
            7 => self.nod / NANOS,
            8 => self.nod / (60 * NANOS) % 60,
            9 => self.nod / (60 * NANOS),
            10 => hour % 12,
            11 => {
                if hour % 12 == 0 {
                    12
                } else {
                    hour % 12
                }
            }
            12 => hour,
            13 => {
                if hour == 0 {
                    24
                } else {
                    hour
                }
            }
            14 => hour / 12,
            15 => i64::from(self.day_of_week().0) + 1,
            16 => (d - 1) % 7 + 1,
            17 => (doy - 1) % 7 + 1,
            18 => d,
            19 => doy,
            20 => self.days,
            21 => (d - 1) / 7 + 1,
            22 => (doy - 1) / 7 + 1,
            23 => m,
            24 => y * 12 + m - 1,
            25 => {
                if y >= 1 {
                    y
                } else {
                    1 - y
                }
            }
            26 => y,
            27 => i64::from(y >= 1),
            INSTANT_SECONDS_F => self.epoch_second(),
            _ => i64::from(self.off),
        }
    }

    /// `get(field)`: the int fields only.
    fn field_int(self, f: Field) -> Result<i64, Throw> {
        match f.0 {
            INSTANT_SECONDS_F | 1 | 3 | 20 | 24 => Err(f.use_get_long()),
            _ => Ok(self.chrono(f)),
        }
    }
}

impl Inst {
    fn chrono(self, f: Field) -> Result<i64, Throw> {
        match f.0 {
            0 => Ok(i64::from(self.nanos)),
            2 => Ok(i64::from(self.nanos / 1000)),
            4 => Ok(i64::from(self.nanos / 1_000_000)),
            INSTANT_SECONDS_F => Ok(self.secs),
            _ => Err(f.unsupported()),
        }
    }
}

fn int_of(f: Field, v: i64) -> Result<Value, Throw> {
    i32::try_from(v).map(Value::Int).map_err(|_| {
        date_time(format!(
            "Invalid value for {} (valid values {}): {v}",
            f.title(),
            f.range_text()
        ))
    })
}

/// `get(TemporalField)`, `getLong` and `isSupported` shared by the
/// temporals.
fn temporal_call(this: &Value, name: &str, args: &[Value]) -> Option<R> {
    Some(match (name, args) {
        ("get", [f]) => field_arg(f).and_then(|f| match downcast::<Zoned>(this) {
            Some(z) => z.field_int(f).and_then(|v| int_of(f, v)),
            None => temporal_long(this, f).and_then(|v| int_of(f, v)),
        }),
        ("getLong", [f]) => field_arg(f)
            .and_then(|f| temporal_long(this, f))
            .map(Value::Long),
        ("isSupported", [f]) => match f {
            Value::Null => Ok(Value::Bool(false)),
            f => field_arg(f)
                .and_then(|f| temporal_supports(this, f))
                .map(Value::Bool),
        },
        _ => return None,
    })
}

/// `a.until(b, unit)` for the temporals the runtime has.
fn temporal_until(a: &Value, b: &Value, unit: Unit) -> Result<i64, Throw> {
    if let Some(a) = downcast::<Zoned>(a) {
        return a.until(end_zoned(b)?, unit);
    }
    if let Some(a) = downcast::<Inst>(a) {
        return a.until(end_instant(b)?, unit);
    }
    Err(bad(a, "java.time.temporal.Temporal"))
}

/// An end point of `ZonedDateTime.until`: `ZonedDateTime.from`.
fn end_zoned(v: &Value) -> Result<Zoned, Throw> {
    match downcast::<Zoned>(v) {
        Some(z) => Ok(*z),
        None => match downcast::<Inst>(v) {
            Some(i) => Err(date_time(format!(
                "Unable to obtain ZonedDateTime from TemporalAccessor: {} of type java.time.Instant",
                i.string()
            ))),
            None => Err(bad(v, "java.time.temporal.Temporal")),
        },
    }
}

/// An end point of `Instant.until`: `Instant.from`.
fn end_instant(v: &Value) -> Result<Inst, Throw> {
    match downcast::<Inst>(v) {
        Some(i) => Ok(*i),
        None => match downcast::<Zoned>(v) {
            Some(z) => z.instant(),
            None => Err(bad(v, "java.time.temporal.Temporal")),
        },
    }
}

fn both(n: &Value, unit: &Value) -> Result<(i64, Unit), Throw> {
    Ok((long_arg(n)?, unit_arg(unit)?))
}

fn zoned_arg(v: &Value) -> Result<Zoned, Throw> {
    downcast::<Zoned>(v)
        .copied()
        .ok_or_else(|| bad(v, "java.time.chrono.ChronoZonedDateTime"))
}

fn instant_arg(v: &Value) -> Result<Inst, Throw> {
    downcast::<Inst>(v).copied().ok_or_else(|| bad(v, INSTANT))
}

fn duration_arg(v: &Value) -> Result<Dur, Throw> {
    downcast::<Dur>(v)
        .copied()
        .ok_or_else(|| bad(v, "java.time.temporal.TemporalAmount"))
}

fn offset_arg(v: &Value) -> Result<Offset, Throw> {
    downcast::<Offset>(v)
        .copied()
        .ok_or_else(|| bad(v, "java.time.ZoneId"))
}

fn unit_arg(v: &Value) -> Result<Unit, Throw> {
    downcast::<Unit>(v)
        .copied()
        .ok_or_else(|| bad(v, "java.time.temporal.TemporalUnit"))
}

fn enum_ordinal(v: &Value, class: &str) -> Result<i32, Throw> {
    match (
        downcast::<DayOfWeek>(v),
        downcast::<Month>(v),
        downcast::<Unit>(v),
    ) {
        (Some(d), _, _) if class == DAY_OF_WEEK => Ok(i32::from(d.0)),
        (_, Some(m), _) if class == MONTH => Ok(i32::from(m.0)),
        (_, _, Some(u)) if class == CHRONO_UNIT => Ok(i32::from(u.0)),
        _ => Err(bad(v, "java.lang.Enum")),
    }
}

fn named_unit(plural: &str) -> Unit {
    Unit(match plural {
        "Nanos" => 0,
        "Seconds" => 3,
        "Minutes" => 4,
        "Hours" => 5,
        "Days" => 7,
        "Weeks" => 8,
        "Months" => 9,
        _ => 10,
    })
}

thread_local! {
    static DAYS: Vec<Value> = (0..7).map(|i| Value::object(DayOfWeek(i))).collect();
    static MONTHS: Vec<Value> = (0..12).map(|i| Value::object(Month(i))).collect();
    static UNITS: Vec<Value> = (0..16).map(|i| Value::object(Unit(i))).collect();
    static FIELDS: Vec<Value> = (0..30).map(|i| Value::object(Field(i))).collect();
}

/// The shared constant of a day (enum constants are one object each).
pub fn day_value(d: DayOfWeek) -> Value {
    DAYS.with(|c| c[usize::from(d.0)].clone())
}

pub fn month_value(m: Month) -> Value {
    MONTHS.with(|c| c[usize::from(m.0)].clone())
}

fn field_value(f: Field) -> Value {
    FIELDS.with(|c| c[usize::from(f.0)].clone())
}

fn unit_value(u: Unit) -> Value {
    UNITS.with(|c| c[usize::from(u.0)].clone())
}

/// A `ZonedDateTime` of `epoch_millis` or `epoch_nanos` in UTC, as a date
/// field's values read.
pub fn utc_from_millis(ms: i64) -> Result<Value, Throw> {
    let inst = Inst::of(ms.div_euclid(1000), ms.rem_euclid(1000) * 1_000_000)?;
    Zoned::of_instant(inst, 0).map(Value::object)
}

/// A `ZonedDateTime` of nanoseconds since the epoch in UTC.
pub fn utc_from_nanos(ns: i64) -> Result<Value, Throw> {
    let inst = Inst::of(ns.div_euclid(NANOS), ns.rem_euclid(NANOS))?;
    Zoned::of_instant(inst, 0).map(Value::object)
}

/// Samples of each class, for the static check's probes.
pub fn sample(class: &str) -> Option<Value> {
    Some(match class {
        ZONED => Value::object(Zoned {
            days: 0,
            nod: 0,
            off: 0,
        }),
        INSTANT => Value::object(Inst { secs: 0, nanos: 0 }),
        DURATION => Value::object(Dur::ZERO),
        DAY_OF_WEEK => day_value(DayOfWeek(0)),
        MONTH => month_value(Month(0)),
        ZONE_OFFSET => Value::object(Offset(0)),
        CHRONO_UNIT => unit_value(Unit(0)),
        CHRONO_FIELD => field_value(Field(0)),
        _ => return None,
    })
}

/// Static fields.
pub fn static_field(class: &str, name: &str) -> Option<Value> {
    Some(match class {
        DAY_OF_WEEK => day_value(DayOfWeek(DAY_NAMES.iter().position(|n| *n == name)? as u8)),
        MONTH => month_value(Month(MONTH_NAMES.iter().position(|n| *n == name)? as u8)),
        CHRONO_UNIT => unit_value(Unit(UNIT_NAMES.iter().position(|n| *n == name)? as u8)),
        CHRONO_FIELD => field_value(Field(FIELD_NAMES.iter().position(|n| *n == name)? as u8)),
        ZONE_OFFSET => Value::object(match name {
            "UTC" => Offset(0),
            "MIN" => Offset(-MAX_OFFSET),
            "MAX" => Offset(MAX_OFFSET),
            _ => return None,
        }),
        INSTANT => Value::object(match name {
            "EPOCH" => Inst { secs: 0, nanos: 0 },
            "MIN" => Inst {
                secs: MIN_SECOND,
                nanos: 0,
            },
            "MAX" => Inst {
                secs: MAX_SECOND,
                nanos: 999_999_999,
            },
            _ => return None,
        }),
        DURATION if name == "ZERO" => Value::object(Dur::ZERO),
        _ => return None,
    })
}

fn offset_of(hours: i64, minutes: i64, seconds: i64) -> Result<Value, Throw> {
    let range = |what: &str, v: i64, lo: i64, hi: i64| {
        if (lo..=hi).contains(&v) {
            Ok(())
        } else {
            Err(date_time(format!(
                "Zone offset {what} not in valid range: value {v} is not in the range {lo} to {hi}"
            )))
        }
    };
    range("hours", hours, -18, 18)?;
    range("minutes", minutes, -59, 59)?;
    range("seconds", seconds, -59, 59)?;
    if hours > 0 {
        if minutes < 0 || seconds < 0 {
            return Err(date_time(
                "Zone offset minutes and seconds must be positive because hours is positive",
            ));
        }
    } else if hours < 0 {
        if minutes > 0 || seconds > 0 {
            return Err(date_time(
                "Zone offset minutes and seconds must be negative because hours is negative",
            ));
        }
    } else if (minutes > 0 && seconds < 0) || (minutes < 0 && seconds > 0) {
        return Err(date_time(
            "Zone offset minutes and seconds must have the same sign",
        ));
    }
    let total = hours * 3600 + minutes * 60 + seconds;
    if total.abs() > i64::from(MAX_OFFSET) {
        return Err(date_time(
            "Zone offset not in valid range: -18:00 to +18:00",
        ));
    }
    Ok(Value::object(Offset(total as i32)))
}

fn enum_of(class: &str, name: &str, names: &[&str]) -> Result<usize, Throw> {
    names.iter().position(|n| *n == name).ok_or_else(|| {
        Throw::illegal_arg(format!(
            "No enum constant {}.{name}",
            class.replace('$', ".")
        ))
    })
}

/// Static methods.
pub fn static_lookup(class: &str, name: &str, arity: usize) -> Option<ApiFn> {
    Some(match (class, name, arity) {
        (ZONED, "ofInstant", 2) => |_vm, a| {
            let (i, o) = (instant_arg(&a[0])?, offset_arg(&a[1])?);
            Zoned::of_instant(i, o.0).map(Value::object)
        },
        (ZONED, "of", 8) => |_vm, a| {
            let n = |k: usize| int_arg(&a[k]).map(i64::from);
            let (y, mo, d) = (n(0)?, n(1)?, n(2)?);
            let days = date_of(y, mo, d)?;
            let (h, mi, s, ns) = (n(3)?, n(4)?, n(5)?, n(6)?);
            check_time(h, mi, s, ns)?;
            let o = offset_arg(&a[7])?;
            Zoned::make(days, (h * 3600 + mi * 60 + s) * NANOS + ns, o.0).map(Value::object)
        },
        (ZONED, "from", 1) => |_vm, a| zoned_arg(&a[0]).map(Value::object),
        (INSTANT, "ofEpochMilli", 1) => |_vm, a| {
            let ms = long_arg(&a[0])?;
            Inst::of(ms.div_euclid(1000), ms.rem_euclid(1000) * 1_000_000).map(Value::object)
        },
        (INSTANT, "ofEpochSecond", 1) => |_vm, a| Inst::of(long_arg(&a[0])?, 0).map(Value::object),
        (INSTANT, "ofEpochSecond", 2) => |_vm, a| {
            let (s, n) = (long_arg(&a[0])?, long_arg(&a[1])?);
            let secs = s.checked_add(n.div_euclid(NANOS)).ok_or_else(overflow)?;
            Inst::of(secs, n.rem_euclid(NANOS)).map(Value::object)
        },
        (INSTANT, "from", 1) => |_vm, a| end_instant(&a[0]).map(Value::object),
        (DURATION, "ofDays", 1) => {
            |_vm, a| Dur::of(i128::from(long_arg(&a[0])?) * 86_400, 0).map(Value::object)
        }
        (DURATION, "ofHours", 1) => {
            |_vm, a| Dur::of(i128::from(long_arg(&a[0])?) * 3600, 0).map(Value::object)
        }
        (DURATION, "ofMinutes", 1) => {
            |_vm, a| Dur::of(i128::from(long_arg(&a[0])?) * 60, 0).map(Value::object)
        }
        (DURATION, "ofSeconds", 1) => {
            |_vm, a| Dur::of(i128::from(long_arg(&a[0])?), 0).map(Value::object)
        }
        (DURATION, "ofSeconds", 2) => |_vm, a| {
            Dur::of(i128::from(long_arg(&a[0])?), i128::from(long_arg(&a[1])?)).map(Value::object)
        },
        (DURATION, "ofMillis", 1) => |_vm, a| {
            let ms = long_arg(&a[0])?;
            Dur::of(i128::from(ms / 1000), i128::from(ms % 1000) * 1_000_000).map(Value::object)
        },
        (DURATION, "ofNanos", 1) => {
            |_vm, a| Dur::of(0, i128::from(long_arg(&a[0])?)).map(Value::object)
        }
        (DURATION, "of", 2) => |_vm, a| {
            let (n, u) = (long_arg(&a[0])?, unit_arg(&a[1])?);
            duration_plus_unit(Dur::ZERO, i128::from(n), u).map(Value::object)
        },
        (DURATION, "from", 1) => |_vm, a| duration_arg(&a[0]).map(Value::object),
        (DURATION, "between", 2) => |_vm, a| duration_between(&a[0], &a[1]),
        (DAY_OF_WEEK, "of", 1) => |_vm, a| {
            let d = int_arg(&a[0])?;
            if (1..=7).contains(&d) {
                Ok(day_value(DayOfWeek(d as u8 - 1)))
            } else {
                Err(date_time(format!("Invalid value for DayOfWeek: {d}")))
            }
        },
        (DAY_OF_WEEK, "from", 1) => {
            |_vm, a| match (downcast::<DayOfWeek>(&a[0]), downcast::<Zoned>(&a[0])) {
                (Some(d), _) => Ok(day_value(*d)),
                (_, Some(z)) => Ok(day_value(z.day_of_week())),
                _ => Err(bad(&a[0], "java.time.temporal.TemporalAccessor")),
            }
        }
        (DAY_OF_WEEK, "valueOf", 1) => |_vm, a| {
            let n = crate::api::str_arg(&a[0])?;
            enum_of(DAY_OF_WEEK, &n, &DAY_NAMES).map(|i| day_value(DayOfWeek(i as u8)))
        },
        (DAY_OF_WEEK, "values", 0) => |_vm, _a| {
            Ok(Value::array(
                DAY_OF_WEEK,
                (0..7).map(|i| day_value(DayOfWeek(i))).collect(),
            ))
        },
        (MONTH, "of", 1) => |_vm, a| {
            let m = int_arg(&a[0])?;
            if (1..=12).contains(&m) {
                Ok(month_value(Month(m as u8 - 1)))
            } else {
                Err(date_time(format!("Invalid value for MonthOfYear: {m}")))
            }
        },
        (MONTH, "from", 1) => |_vm, a| match (downcast::<Month>(&a[0]), downcast::<Zoned>(&a[0])) {
            (Some(m), _) => Ok(month_value(*m)),
            (_, Some(z)) => Ok(month_value(Month(z.ymd().1 as u8 - 1))),
            _ => Err(bad(&a[0], "java.time.temporal.TemporalAccessor")),
        },
        (MONTH, "valueOf", 1) => |_vm, a| {
            let n = crate::api::str_arg(&a[0])?;
            enum_of(MONTH, &n, &MONTH_NAMES).map(|i| month_value(Month(i as u8)))
        },
        (MONTH, "values", 0) => |_vm, _a| {
            Ok(Value::array(
                MONTH,
                (0..12).map(|i| month_value(Month(i))).collect(),
            ))
        },
        (CHRONO_UNIT, "valueOf", 1) => |_vm, a| {
            let n = crate::api::str_arg(&a[0])?;
            enum_of(CHRONO_UNIT, &n, &UNIT_NAMES).map(|i| unit_value(Unit(i as u8)))
        },
        (CHRONO_UNIT, "values", 0) => |_vm, _a| {
            Ok(Value::array(
                CHRONO_UNIT,
                (0..16).map(|i| unit_value(Unit(i))).collect(),
            ))
        },
        (CHRONO_FIELD, "valueOf", 1) => |_vm, a| {
            let n = crate::api::str_arg(&a[0])?;
            enum_of(CHRONO_FIELD, &n, &FIELD_NAMES).map(|i| field_value(Field(i as u8)))
        },
        (CHRONO_FIELD, "values", 0) => |_vm, _a| {
            Ok(Value::array(
                CHRONO_FIELD,
                (0..30).map(|i| field_value(Field(i))).collect(),
            ))
        },
        (ZONE_OFFSET, "ofHours", 1) => |_vm, a| offset_of(i64::from(int_arg(&a[0])?), 0, 0),
        (ZONE_OFFSET, "ofHoursMinutes", 2) => {
            |_vm, a| offset_of(i64::from(int_arg(&a[0])?), i64::from(int_arg(&a[1])?), 0)
        }
        (ZONE_OFFSET, "ofHoursMinutesSeconds", 3) => |_vm, a| {
            offset_of(
                i64::from(int_arg(&a[0])?),
                i64::from(int_arg(&a[1])?),
                i64::from(int_arg(&a[2])?),
            )
        },
        (ZONE_OFFSET, "ofTotalSeconds", 1) => |_vm, a| {
            let s = int_arg(&a[0])?;
            if s.abs() > MAX_OFFSET {
                Err(date_time(
                    "Zone offset not in valid range: -18:00 to +18:00",
                ))
            } else {
                Ok(Value::object(Offset(s)))
            }
        },
        (ZONE_OFFSET, "from", 1) => {
            |_vm, a| match (downcast::<Offset>(&a[0]), downcast::<Zoned>(&a[0])) {
                (Some(o), _) => Ok(Value::object(*o)),
                (_, Some(z)) => Ok(Value::object(Offset(z.off))),
                _ => Err(bad(&a[0], "java.time.temporal.TemporalAccessor")),
            }
        }
        _ => return None,
    })
}

/// `Duration.between`.
fn duration_between(a: &Value, b: &Value) -> R {
    let (start, end) = if let Some(z) = downcast::<Zoned>(a) {
        (z.instant()?, end_zoned(b)?.instant()?)
    } else if let Some(i) = downcast::<Inst>(a) {
        (*i, end_instant(b)?)
    } else {
        return Err(bad(a, "java.time.temporal.Temporal"));
    };
    let total = end.total_nanos() - start.total_nanos();
    Dur::of_total(total, || Dur::decimal(total)).map(Value::object)
}

#[cfg(test)]
mod statics_tests;
#[cfg(test)]
mod tests;
