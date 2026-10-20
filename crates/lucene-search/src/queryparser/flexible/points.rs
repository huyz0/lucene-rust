//! `standard.config.PointsConfig`, its `PointsConfigListener` (the parser's
//! map, [`super::StandardQueryParser::set_points_config_map`]) and
//! `NumberDateFormat`: how a field indexed as points reads its terms and
//! range bounds -- the `NumberFormat` that parses them and the `Number`
//! class they become.
//!
//! The formats are the two Java offers per locale: `NumberFormat`'s number
//! and integer instances ([`DecimalFormat`], over the generated locale
//! table) and `new NumberDateFormat(DateFormat.getDateInstance(SHORT,
//! locale))`, a date read as its epoch milliseconds.
//!
//! Differs: Java takes any `NumberFormat`; here those three. A date format
//! is the short date instance only, of a locale whose calendar is Gregorian
//! (as the date range bounds, `date_range.rs`), in a fixed-offset default
//! zone.

use lucene_analysis::miscellaneous::{DateLocale, DecimalFormat, JavaNumber};

use crate::queryparser::classic::QueryParserError;
use crate::queryparser::date_range::DateParsing;

/// `PointsConfig.getType()`: `Integer`, `Long`, `Float` or `Double`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointType {
    Int,
    Long,
    Float,
    Double,
}

/// A point term's value, of its config's type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum PointValue {
    Int(i32),
    Long(i64),
    Float(f32),
    Double(f64),
}

impl PointType {
    /// `number.intValue()`, `longValue()`, `floatValue()` or `doubleValue()`.
    pub(crate) fn convert(self, n: JavaNumber) -> PointValue {
        match self {
            PointType::Int => PointValue::Int(n.int_value()),
            PointType::Long => PointValue::Long(n.long_value()),
            PointType::Float => PointValue::Float(n.float_value()),
            PointType::Double => PointValue::Double(n.double_value()),
        }
    }
}

/// `new NumberDateFormat(DateFormat.getDateInstance(DateFormat.SHORT,
/// locale))`: a date's epoch milliseconds, parsed and formatted in the
/// locale's lenient short date pattern, in the zone that was
/// `TimeZone.getDefault()`.
#[derive(Debug, Clone, PartialEq)]
pub struct NumberDateFormat {
    dates: DateParsing,
}

impl NumberDateFormat {
    /// The short date instance of `locale` in a default zone
    /// `default_time_zone` milliseconds from GMT; `None` for a locale whose
    /// calendar is not Gregorian (Thai's Buddhist), which this port does not
    /// compute.
    pub fn short_date_instance(locale: DateLocale, default_time_zone: i32) -> Option<Self> {
        let mut dates = DateParsing::default();
        dates.set_locale(locale);
        dates.default_time_zone = default_time_zone;
        dates.is_supported().then_some(NumberDateFormat { dates })
    }

    /// The time (milliseconds since the epoch) a two-digit year's century
    /// is chosen from: `None` (the default) reads the system clock.
    pub fn set_date_clock(&mut self, now_millis: Option<i64>) {
        self.dates.now = now_millis;
    }
}

/// The `NumberFormat` of a [`PointsConfig`].
#[derive(Debug, Clone, PartialEq)]
pub enum PointsFormat {
    /// `NumberFormat.getNumberInstance(locale)` or `getIntegerInstance(locale)`.
    Decimal(DecimalFormat),
    /// `NumberDateFormat`.
    Date(NumberDateFormat),
}

impl PointsFormat {
    /// `parse(text)`: `None` where Java throws `ParseException`.
    pub(crate) fn parse(&self, text: &str) -> Option<JavaNumber> {
        match self {
            PointsFormat::Decimal(f) => f.parse(text),
            PointsFormat::Date(f) => f.dates.parse(text).map(JavaNumber::Long),
        }
    }

    /// `getClass().getCanonicalName()`.
    pub(crate) fn class_name(&self) -> &'static str {
        match self {
            PointsFormat::Decimal(_) => "java.text.DecimalFormat",
            PointsFormat::Date(_) => {
                "org.apache.lucene.queryparser.flexible.standard.config.NumberDateFormat"
            }
        }
    }

    /// `format(value)` (`Format.format(Object)`): an `Integer` or `Long` as
    /// a `long`, a `Float` or `Double` as a `double`, a date of its
    /// `longValue()`; `IllegalArgumentException` for no value.
    pub(crate) fn format(&self, value: Option<PointValue>) -> Result<String, QueryParserError> {
        let illegal = |what: &str| {
            QueryParserError::runtime_class(
                "java.lang.IllegalArgumentException",
                format!("Cannot format given Object as a {what}"),
            )
        };
        match self {
            PointsFormat::Decimal(f) => match value {
                None => Err(illegal("Number")),
                Some(PointValue::Int(v)) => Ok(f.format_long(i64::from(v))),
                Some(PointValue::Long(v)) => Ok(f.format_long(v)),
                Some(PointValue::Float(v)) => Ok(f.format_double(f64::from(v))),
                Some(PointValue::Double(v)) => Ok(f.format_double(v)),
            },
            PointsFormat::Date(f) => {
                let time = match value {
                    None => return Err(illegal("Date")),
                    Some(PointValue::Int(v)) => i64::from(v),
                    Some(PointValue::Long(v)) => v,
                    Some(PointValue::Float(v)) => v as i64,
                    Some(PointValue::Double(v)) => v as i64,
                };
                Ok(f.dates.format(time))
            }
        }
    }
}

/// `PointsConfig`: a points field's format and type.
#[derive(Debug, Clone, PartialEq)]
pub struct PointsConfig {
    format: PointsFormat,
    point_type: PointType,
}

impl PointsConfig {
    /// `new PointsConfig(format, type)`.
    pub fn new(format: PointsFormat, point_type: PointType) -> Self {
        PointsConfig { format, point_type }
    }

    /// `getNumberFormat()`.
    pub fn number_format(&self) -> &PointsFormat {
        &self.format
    }

    /// `getType()`.
    pub fn point_type(&self) -> PointType {
        self.point_type
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn locale(tag: &str) -> DateLocale {
        DateLocale::for_language_tag(tag).unwrap()
    }

    #[test]
    fn formats_and_types() {
        let us = PointsFormat::Decimal(DecimalFormat::number_instance(locale("en-US")));
        assert_eq!(us.class_name(), "java.text.DecimalFormat");
        assert_eq!(us.format(Some(PointValue::Int(-5))).unwrap(), "-5");
        assert_eq!(us.format(Some(PointValue::Long(1000))).unwrap(), "1,000");
        assert_eq!(us.format(Some(PointValue::Float(1.1))).unwrap(), "1.1");
        assert_eq!(us.format(Some(PointValue::Double(0.5))).unwrap(), "0.5");
        let e = us.format(None).unwrap_err();
        assert_eq!(e.message, "Cannot format given Object as a Number");
        assert_eq!(
            PointType::Int.convert(JavaNumber::Long(1 << 32 | 7)),
            PointValue::Int(7)
        );
        assert_eq!(
            PointType::Long.convert(JavaNumber::Double(2.5)),
            PointValue::Long(2)
        );
        assert_eq!(
            PointType::Float.convert(JavaNumber::Long(3)),
            PointValue::Float(3.0)
        );
        assert_eq!(
            PointType::Double.convert(JavaNumber::Long(3)),
            PointValue::Double(3.0)
        );

        let mut date =
            NumberDateFormat::short_date_instance(locale("en-US"), -5 * 3_600_000).unwrap();
        date.set_date_clock(Some(0));
        let date = PointsFormat::Date(date);
        assert!(date.class_name().ends_with(".NumberDateFormat"));
        // 2020-01-02T00:00-05:00
        let jan2 = 1_577_941_200_000_i64;
        assert_eq!(date.parse("1/2/2020"), Some(JavaNumber::Long(jan2)));
        assert_eq!(date.parse("x"), None);
        assert_eq!(date.format(Some(PointValue::Long(jan2))).unwrap(), "1/2/20");
        assert_eq!(
            date.format(Some(PointValue::Double(jan2 as f64))).unwrap(),
            "1/2/20"
        );
        assert_eq!(
            date.format(Some(PointValue::Float(0.0))).unwrap(),
            "12/31/69"
        );
        assert_eq!(date.format(Some(PointValue::Int(0))).unwrap(), "12/31/69");
        // the year of its era (3 BC, Julian), and the extremes (JDK 25's)
        let bc = -62_198_755_200_000_i64 - 366 * 86_400_000;
        assert_eq!(date.format(Some(PointValue::Long(bc))).unwrap(), "1/1/03");
        assert_eq!(
            date.format(Some(PointValue::Long(i64::MAX))).unwrap(),
            "8/17/94"
        );
        assert_eq!(
            date.format(Some(PointValue::Long(i64::MIN))).unwrap(),
            "12/2/55"
        );
        let fa = NumberDateFormat::short_date_instance(locale("fa"), -5 * 3_600_000).unwrap();
        assert_eq!(
            PointsFormat::Date(fa)
                .format(Some(PointValue::Long(bc)))
                .unwrap(),
            "۳/۱/۱"
        );
        assert_eq!(
            date.format(None).unwrap_err().message,
            "Cannot format given Object as a Date"
        );
        assert!(NumberDateFormat::short_date_instance(locale("th-TH"), 0).is_none());

        let config = PointsConfig::new(us.clone(), PointType::Long);
        assert_eq!(config.number_format(), &us);
        assert_eq!(config.point_type(), PointType::Long);
    }
}
