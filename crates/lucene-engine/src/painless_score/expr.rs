//! A Lucene expression (`lang: expression`) on a segment: the expression is
//! `lucene-expressions`' compile of its source, each of its variables bound as
//! `ExpressionScriptEngine` binds it (the score, a param, or a numeric field's
//! values under a `MultiValueMode`), and the segment's [`ScoreEnv`] reads the
//! fields.

use lucene_expressions::js::{compile, Expression};
use lucene_util::geo::GeoEncodingUtils;

use super::*;
use crate::function_score::{numeric, Numeric};

/// `MultiValueMode`: how a field's several values in a document become one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Min,
    Max,
    Avg,
    Median,
    Sum,
}

/// What one of an expression's variables is.
#[derive(Debug, Clone, PartialEq)]
pub enum ExprBinding {
    /// `_score`.
    Score,
    /// A param, bound as a constant.
    Const(f64),
    /// A numeric field's value (`doc['f'].value`, `.min`, `.sum`, ...),
    /// its stored longs read as `numeric` says.
    Field {
        field: usize,
        numeric: Numeric,
        mode: Mode,
    },
    /// `_value`: the field value an aggregation maps (`0` outside one).
    Value,
    /// `doc['f'].empty`: `1` without a value, else `0`.
    Empty(usize),
    /// `doc['f'].length`: the document's value count.
    Count(usize),
    /// A date's `Calendar` field (`getYear()`, ...): UTC, Gregorian with
    /// the Julian cutover, of the field's first value.
    Calendar {
        field: usize,
        numeric: Numeric,
        calendar_field: u8,
    },
    /// A date's Joda `ReadableDateTime` field (`.date.dayOfWeek`, ...): UTC
    /// ISO, of the field's first value.
    Joda {
        field: usize,
        numeric: Numeric,
        part: u8,
    },
    /// A `geo_point`'s first point: its latitude, longitude, or (as stock's
    /// `GeoEmptyValueSource` does) `1` where there is one.
    GeoLat(usize),
    GeoLon(usize),
    GeoEmpty(usize),
}

/// A compiled expression and what each of its variables reads.
pub struct ExprSource {
    expr: Expression,
    /// Per variable of `expr`, in its order.
    bindings: Vec<(String, ExprBinding)>,
}

impl std::fmt::Debug for ExprSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExprSource")
            .field("source", &self.expr.source_text)
            .field("bindings", &self.bindings)
            .finish()
    }
}

impl ExprSource {
    /// The expression of `source`, then its variables' bindings off `c`
    /// (each a name, a kind and what the kind takes), over `fields` fields.
    pub fn decode(
        source: &str,
        fields: usize,
        c: &mut Cursor<'_>,
    ) -> std::result::Result<Self, Status> {
        let expr = compile(source).map_err(|e| invalid(format!("query tree: expression: {e}")))?;
        let n = c.length()?;
        let mut bindings = Vec::with_capacity(n.min(c.remaining())); // alloc-ok: bounded by the blob
        for _ in 0..n {
            let name = string(c)?;
            let field = |c: &mut Cursor<'_>| -> std::result::Result<usize, Status> {
                let i = usize::try_from(c.i32()?)
                    .ok()
                    .filter(|i| *i < fields)
                    .ok_or_else(|| invalid("query tree: expression binding's field"))?;
                Ok(i)
            };
            let binding = match c.u8()? {
                0 => ExprBinding::Score,
                1 => ExprBinding::Const(c.f64()?),
                2 => {
                    let i = field(c)?;
                    let num = numeric(c)?;
                    let mode = match c.u8()? {
                        0 => Mode::Min,
                        1 => Mode::Max,
                        2 => Mode::Avg,
                        3 => Mode::Median,
                        4 => Mode::Sum,
                        other => {
                            return Err(invalid(format!(
                                "query tree: expression mode {other} (expected 0..=4)"
                            )))
                        }
                    };
                    ExprBinding::Field {
                        field: i,
                        numeric: num,
                        mode,
                    }
                }
                3 => ExprBinding::Empty(field(c)?),
                4 => ExprBinding::Count(field(c)?),
                5 => {
                    let i = field(c)?;
                    let num = numeric(c)?;
                    let calendar_field = c.u8()?;
                    if !matches!(calendar_field, 1 | 2 | 5 | 11 | 12 | 13) {
                        return Err(invalid("query tree: expression calendar field"));
                    }
                    ExprBinding::Calendar {
                        field: i,
                        numeric: num,
                        calendar_field,
                    }
                }
                6 => {
                    let i = field(c)?;
                    let num = numeric(c)?;
                    let part = c.u8()?;
                    if part > 17 {
                        return Err(invalid("query tree: expression date part"));
                    }
                    ExprBinding::Joda {
                        field: i,
                        numeric: num,
                        part,
                    }
                }
                7 => ExprBinding::GeoLat(field(c)?),
                8 => ExprBinding::GeoLon(field(c)?),
                9 => ExprBinding::GeoEmpty(field(c)?),
                10 => ExprBinding::Value,
                other => {
                    return Err(invalid(format!(
                        "query tree: expression binding kind {other} (expected 0..=10)"
                    )))
                }
            };
            bindings.push((name, binding));
        }
        let names: Vec<&str> = bindings.iter().map(|(n, _)| n.as_str()).collect();
        if expr
            .variables
            .iter()
            .map(String::as_str)
            .ne(names.iter().copied())
        {
            return Err(invalid(
                "query tree: expression bindings do not match its variables",
            ));
        }
        Ok(ExprSource { expr, bindings })
    }

    /// What the expression is, for [`Spec::key`].
    pub fn key(&self) -> String {
        format!("{}:{:?}", self.expr.source_text, self.bindings)
    }
}

/// An expression on one segment.
pub(super) struct ExprLeaf {
    spec: Arc<Spec>,
}

impl ExprLeaf {
    pub(super) fn new(spec: &Arc<Spec>) -> Self {
        ExprLeaf {
            spec: Arc::clone(spec),
        }
    }

    fn source(&self) -> &ExprSource {
        self.spec.expr.as_ref().expect("an expression spec")
    }

    /// The value of variable `b` at `env.doc`.
    fn variable(&self, env: &mut ScoreEnv<'_>, b: &ExprBinding) -> Result<f64> {
        Ok(match b {
            // A sort expression is given no scorer: its `_score` reads `0`.
            ExprBinding::Score => {
                if self.spec.needs_score && !self.spec.sort {
                    env.score
                } else {
                    0.0
                }
            }
            ExprBinding::Const(v) => *v,
            ExprBinding::Value => match env.agg_value {
                Value::Double(d) => d,
                _ => 0.0,
            },
            ExprBinding::Field {
                field,
                numeric,
                mode,
            } => {
                let cur = env.load(*field).map_err(|t| runtime_failure(&t))?;
                pick(*numeric, &cur.items[..cur.count], *mode)
            }
            ExprBinding::Empty(field) => {
                let cur = env.load(*field).map_err(|t| runtime_failure(&t))?;
                f64::from(u8::from(cur.count == 0))
            }
            // `GeoEmptyValueSource` is `1` where the document has a point and
            // positions nowhere else, so the expression reads `0` for none.
            ExprBinding::GeoEmpty(field) => {
                let cur = env.load(*field).map_err(|t| runtime_failure(&t))?;
                f64::from(u8::from(cur.count != 0))
            }
            ExprBinding::Count(field) => {
                let cur = env.load(*field).map_err(|t| runtime_failure(&t))?;
                cur.count as f64
            }
            ExprBinding::Calendar {
                field,
                numeric,
                calendar_field,
            } => {
                let cur = env.load(*field).map_err(|t| runtime_failure(&t))?;
                let ms = pick(*numeric, &cur.items[..cur.count], Mode::Min) as i64;
                f64::from(calendar::field(ms, *calendar_field))
            }
            ExprBinding::Joda {
                field,
                numeric,
                part,
            } => {
                let cur = env.load(*field).map_err(|t| runtime_failure(&t))?;
                let ms = pick(*numeric, &cur.items[..cur.count], Mode::Min) as i64;
                f64::from(calendar::joda(ms, *part))
            }
            ExprBinding::GeoLat(field) => {
                let cur = env.load(*field).map_err(|t| runtime_failure(&t))?;
                match cur.items[..cur.count].first() {
                    Some(enc) => GeoEncodingUtils::decode_latitude((*enc >> 32) as i32),
                    None => 0.0,
                }
            }
            ExprBinding::GeoLon(field) => {
                let cur = env.load(*field).map_err(|t| runtime_failure(&t))?;
                match cur.items[..cur.count].first() {
                    Some(enc) => GeoEncodingUtils::decode_longitude(*enc as i32),
                    None => 0.0,
                }
            }
        })
    }

    /// The expression's value at `env.doc`.
    pub(super) fn run_value(&self, env: &mut ScoreEnv<'_>) -> Result<f64> {
        let src = self.source();
        const INLINE: usize = 8;
        let n = src.bindings.len();
        if n <= INLINE {
            let mut cols = [0.0f64; INLINE];
            for (slot, (_, b)) in cols.iter_mut().zip(&src.bindings) {
                *slot = self.variable(env, b)?;
            }
            return src.expr.evaluate_row(&cols[..n]);
        }
        let mut cols = Vec::with_capacity(n); // alloc-ok: more variables than the inline row
        for (_, b) in &src.bindings {
            cols.push(self.variable(env, b)?);
        }
        src.expr.evaluate_row(&cols)
    }

    /// The values (a filter's `1` and `0`) of `docs`, scores checked as
    /// `ScriptScorer` checks them, in document order.
    pub(super) fn fill_batch(
        &self,
        env: &mut ScoreEnv<'_>,
        docs: &[i32],
        scores: &[f32],
        out: &mut [f64],
        has: &mut [bool],
    ) -> Result<()> {
        let src = self.source();
        let n = docs.len();
        let vars = src.bindings.len();
        let mut cols = vec![0.0; vars.saturating_mul(n)]; // alloc-ok: the batch's columns
        for (j, &doc) in docs.iter().enumerate() {
            env.doc = doc;
            env.score = f64::from(scores[j]);
            for (i, (_, b)) in src.bindings.iter().enumerate() {
                cols[i.saturating_mul(n).saturating_add(j)] = self.variable(env, b)?;
            }
        }
        src.expr.evaluate_columns(&cols, &mut out[..n])?;
        if self.spec.is_score() {
            for (k, o) in out[..n].iter_mut().enumerate() {
                *o = f64::from(checked(*o as f32, docs[k])?);
            }
        }
        has[..n].fill(true);
        Ok(())
    }
}

/// `MultiValueMode.pick` over a document's values (a field's stored longs,
/// in ascending order, read as `numeric` says), `0` without one.
fn pick(numeric: Numeric, items: &[i64], mode: Mode) -> f64 {
    let at = |i: usize| numeric.decode(items[i]);
    let n = items.len();
    if n == 0 {
        return 0.0;
    }
    let mid = n / 2;
    match mode {
        Mode::Min => at(0),
        Mode::Max => at(n.saturating_sub(1)),
        Mode::Sum => (0..n).fold(0.0, |s, i| s + at(i)),
        Mode::Avg => (0..n).fold(0.0, |s, i| s + at(i)) / n as f64,
        Mode::Median => {
            if n.is_multiple_of(2) {
                (at(mid.saturating_sub(1)) + at(mid)) / 2.0
            } else {
                at(mid)
            }
        }
    }
}

/// Date fields of an epoch-millisecond instant, in UTC: `java.util.Calendar`'s
/// (`GregorianCalendar`, with the Julian calendar before 1582-10-15) and
/// Joda's `ISOChronology`.
mod calendar {
    // Every operand is a day number or a time of day derived from an `i64`
    // of milliseconds, so none of the sums and products here can overflow.
    #![allow(clippy::arithmetic_side_effects)]

    const MS_PER_DAY: i64 = 86_400_000;
    /// 1582-10-15T00:00Z, `GregorianCalendar`'s cutover.
    const CUTOVER: i64 = -12_219_292_800_000;

    /// The proleptic Gregorian date of day `z` (days since 1970-01-01).
    fn gregorian(z: i64) -> (i64, i64, i64) {
        let z = z + 719_468;
        let era = z.div_euclid(146_097);
        let doe = z.rem_euclid(146_097);
        let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let d = doy - (153 * mp + 2) / 5 + 1;
        let m = if mp < 10 { mp + 3 } else { mp - 9 };
        let y = yoe + era * 400 + i64::from(m <= 2);
        (y, m, d)
    }

    /// The Julian calendar date of day `z`.
    fn julian(z: i64) -> (i64, i64, i64) {
        let c = z + 2_440_588 + 32_082;
        let d = (4 * c + 3).div_euclid(1461);
        let e = c - (1461 * d).div_euclid(4);
        let m = (5 * e + 2) / 153;
        let day = e - (153 * m + 2) / 5 + 1;
        let month = m + 3 - 12 * (m / 10);
        let year = d - 4800 + m / 10;
        (year, month, day)
    }

    /// The day (since 1970-01-01) of a proleptic Gregorian date.
    fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
        let y = y - i64::from(m <= 2);
        let era = y.div_euclid(400);
        let yoe = y.rem_euclid(400);
        let mp = if m > 2 { m - 3 } else { m + 9 };
        let doy = (153 * mp + 2) / 5 + d - 1;
        let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        era * 146_097 + doe - 719_468
    }

    /// `Calendar.get(field)` for `YEAR` (1), `MONTH` (2), `DAY_OF_MONTH` (5),
    /// `HOUR_OF_DAY` (11), `MINUTE` (12) and `SECOND` (13).
    pub fn field(ms: i64, field: u8) -> i32 {
        let day = ms.div_euclid(MS_PER_DAY);
        let of_day = ms.rem_euclid(MS_PER_DAY);
        let (y, m, d) = if ms < CUTOVER {
            julian(day)
        } else {
            gregorian(day)
        };
        let v = match field {
            // The year of its era: a year before 1 is `1 - year` BC.
            1 => {
                if y >= 1 {
                    y
                } else {
                    1 - y
                }
            }
            2 => m - 1,
            5 => d,
            11 => of_day / 3_600_000,
            12 => of_day / 60_000 % 60,
            _ => of_day / 1000 % 60,
        };
        v as i32
    }

    fn weeks_in_year(y: i64) -> i64 {
        let p =
            |y: i64| (y + y.div_euclid(4) - y.div_euclid(100) + y.div_euclid(400)).rem_euclid(7);
        if p(y) == 4 || p(y - 1) == 3 {
            53
        } else {
            52
        }
    }

    /// A Joda `ReadableDateTime` getter on `ISOChronology` in UTC: `part` is
    /// the index of `centuryOfEra`, `dayOfMonth`, `dayOfWeek`, `dayOfYear`,
    /// `era`, `hourOfDay`, `millisOfDay`, `millisOfSecond`, `minuteOfDay`,
    /// `minuteOfHour`, `monthOfYear`, `secondOfDay`, `secondOfMinute`,
    /// `weekOfWeekyear`, `weekyear`, `year`, `yearOfCentury`, `yearOfEra`.
    pub fn joda(ms: i64, part: u8) -> i32 {
        let day = ms.div_euclid(MS_PER_DAY);
        let of_day = ms.rem_euclid(MS_PER_DAY);
        let (y, m, d) = gregorian(day);
        let year_of_era = if y > 0 { y } else { 1 - y };
        let dow = (day + 3).rem_euclid(7) + 1;
        let doy = day - days_from_civil(y, 1, 1) + 1;
        let (week, weekyear) = {
            let w = (doy - dow + 10) / 7;
            if w < 1 {
                (weeks_in_year(y - 1), y - 1)
            } else if w > weeks_in_year(y) {
                (1, y + 1)
            } else {
                (w, y)
            }
        };
        let v = match part {
            0 => year_of_era / 100,
            1 => d,
            2 => dow,
            3 => doy,
            4 => i64::from(y > 0),
            5 => of_day / 3_600_000,
            6 => of_day,
            7 => of_day % 1000,
            8 => of_day / 60_000,
            9 => of_day / 60_000 % 60,
            10 => m,
            11 => of_day / 1000,
            12 => of_day / 1000 % 60,
            13 => week,
            14 => weekyear,
            15 => y,
            16 => year_of_era % 100,
            _ => year_of_era,
        };
        v as i32
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn fields_of_known_instants() {
            // 2024-02-29T13:45:59Z, a Thursday, day 60 of the year, ISO week 9.
            let ms = 1_709_214_359_000;
            assert_eq!(field(ms, 1), 2024);
            assert_eq!(field(ms, 2), 1);
            assert_eq!(field(ms, 5), 29);
            assert_eq!(field(ms, 11), 13);
            assert_eq!(field(ms, 12), 45);
            assert_eq!(field(ms, 13), 59);
            assert_eq!(joda(ms, 2), 4);
            assert_eq!(joda(ms, 3), 60);
            assert_eq!(joda(ms, 13), 9);
            assert_eq!(joda(ms, 14), 2024);
            assert_eq!(joda(ms, 16), 24);
            assert_eq!(joda(ms, 0), 20);
        }

        #[test]
        fn the_calendar_switches_to_julian_before_the_cutover() {
            // 1582-10-04 is the day before 1582-10-15.
            let before = CUTOVER - MS_PER_DAY;
            assert_eq!(
                (field(before, 1), field(before, 2), field(before, 5)),
                (1582, 9, 4)
            );
            assert_eq!(
                (field(CUTOVER, 1), field(CUTOVER, 2), field(CUTOVER, 5)),
                (1582, 9, 15)
            );
            // Joda stays proleptic Gregorian: that same day is 1582-10-14.
            assert_eq!(
                (joda(before, 15), joda(before, 10), joda(before, 1)),
                (1582, 10, 14)
            );
        }

        #[test]
        fn iso_weeks_cross_years() {
            // 2021-01-03 (a Sunday) is week 53 of 2020; 2018-12-31 is week 1 of 2019.
            let jan3 = days_from_civil(2021, 1, 3) * MS_PER_DAY;
            assert_eq!((joda(jan3, 13), joda(jan3, 14)), (53, 2020));
            let dec31 = days_from_civil(2018, 12, 31) * MS_PER_DAY;
            assert_eq!((joda(dec31, 13), joda(dec31, 14)), (1, 2019));
        }

        #[test]
        fn before_year_one() {
            // 1 BC: Calendar's YEAR is 1; Joda's year is 0, its era BC.
            let ms = days_from_civil(0, 6, 1) * MS_PER_DAY;
            assert_eq!(joda(ms, 15), 0);
            assert_eq!(joda(ms, 4), 0);
            assert_eq!(joda(ms, 17), 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn name(b: &mut Vec<u8>, s: &str) {
        b.extend_from_slice(&(s.len() as i32).to_le_bytes());
        b.extend_from_slice(s.as_bytes());
    }

    #[test]
    fn pick_follows_multi_value_mode() {
        let v = [1_i64, 2, 4, 9];
        assert_eq!(pick(Numeric::Long, &v, Mode::Min), 1.0);
        assert_eq!(pick(Numeric::Long, &v, Mode::Max), 9.0);
        assert_eq!(pick(Numeric::Long, &v, Mode::Sum), 16.0);
        assert_eq!(pick(Numeric::Long, &v, Mode::Avg), 4.0);
        assert_eq!(pick(Numeric::Long, &v, Mode::Median), 3.0);
        assert_eq!(pick(Numeric::Long, &v[..3], Mode::Median), 2.0);
        assert_eq!(pick(Numeric::Long, &[], Mode::Sum), 0.0);
        assert_eq!(pick(Numeric::Scaled(0.01), &[250], Mode::Min), 2.5);
    }

    #[test]
    fn decode_reads_bindings_in_variable_order() {
        let mut b = Vec::new();
        b.extend_from_slice(&2_i32.to_le_bytes());
        name(&mut b, "_score");
        b.push(0);
        name(&mut b, "doc['l'].value");
        b.push(2);
        b.extend_from_slice(&0_i32.to_le_bytes());
        b.push(0); // Numeric::Long
        b.push(4); // sum
        let mut c = Cursor::new(&b);
        let src = ExprSource::decode("_score + doc['l'].value", 1, &mut c).unwrap();
        assert_eq!(src.bindings[0].1, ExprBinding::Score);
        assert_eq!(
            src.bindings[1].1,
            ExprBinding::Field {
                field: 0,
                numeric: Numeric::Long,
                mode: Mode::Sum
            }
        );
        assert!(src.key().contains("_score"));
        assert!(format!("{src:?}").contains("ExprSource"));
    }

    #[test]
    fn decode_refuses_what_the_plugin_would_not_send() {
        // Bindings that are not the expression's variables.
        let mut b = Vec::new();
        b.extend_from_slice(&1_i32.to_le_bytes());
        name(&mut b, "x");
        b.push(0);
        assert!(ExprSource::decode("y + 1", 0, &mut Cursor::new(&b)).is_err());
        // A field past the spec's.
        let mut b = Vec::new();
        b.extend_from_slice(&1_i32.to_le_bytes());
        name(&mut b, "x");
        b.push(3);
        b.extend_from_slice(&5_i32.to_le_bytes());
        assert!(ExprSource::decode("x", 1, &mut Cursor::new(&b)).is_err());
        // An unknown kind, mode, calendar field and date part.
        for tail in [
            vec![42u8],
            vec![2, 0, 0, 0, 0, 0, 9],
            vec![5, 0, 0, 0, 0, 0, 3],
            vec![6, 0, 0, 0, 0, 0, 40],
        ] {
            let mut b = Vec::new();
            b.extend_from_slice(&1_i32.to_le_bytes());
            name(&mut b, "x");
            b.extend_from_slice(&tail);
            assert!(
                ExprSource::decode("x", 1, &mut Cursor::new(&b)).is_err(),
                "{tail:?}"
            );
        }
        // Source that does not compile.
        assert!(ExprSource::decode("1 +", 0, &mut Cursor::new(&0_i32.to_le_bytes())).is_err());
    }
}
