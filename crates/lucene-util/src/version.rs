//! Port of `org.apache.lucene.util.Version`: `major.minor.bugfix(.prerelease)`
//! with Lucene's range checks, `parse`/`parseLeniently`, `onOrAfter` over the
//! packed `encodedValue`, and the named constants.
//!
//! Java's `Integer.parseInt` also accepts non-ASCII Unicode digits; this port
//! accepts ASCII digits only (a version string with, say, Arabic-Indic
//! digits parses in Java and is rejected here).

use std::fmt;

/// A Lucene version. Equality and ordering are over the encoded value, as in
/// Java.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Version {
    encoded: i32,
}

/// A failed `parse` / `parseLeniently`, with Java's `ParseException` message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VersionParseError(pub String);

impl fmt::Display for VersionParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for VersionParseError {}

macro_rules! versions {
    ($($name:ident = $maj:expr, $min:expr, $bug:expr;)*) => {
        impl Version {
            $(
                #[doc = concat!("`Version.", stringify!($name), "`.")]
                pub const $name: Version = Version::constant($maj, $min, $bug);
            )*

            /// Every named `LUCENE_X_Y_Z` constant, oldest first, with its Java name.
            pub const ALL: &'static [(&'static str, Version)] = &[
                $((stringify!($name), Version::$name),)*
            ];
        }
    };
}

versions! {
    LUCENE_9_0_0 = 9, 0, 0;
    LUCENE_9_1_0 = 9, 1, 0;
    LUCENE_9_2_0 = 9, 2, 0;
    LUCENE_9_3_0 = 9, 3, 0;
    LUCENE_9_4_0 = 9, 4, 0;
    LUCENE_9_4_1 = 9, 4, 1;
    LUCENE_9_4_2 = 9, 4, 2;
    LUCENE_9_5_0 = 9, 5, 0;
    LUCENE_9_6_0 = 9, 6, 0;
    LUCENE_9_7_0 = 9, 7, 0;
    LUCENE_9_8_0 = 9, 8, 0;
    LUCENE_9_9_0 = 9, 9, 0;
    LUCENE_9_9_1 = 9, 9, 1;
    LUCENE_9_9_2 = 9, 9, 2;
    LUCENE_9_10_0 = 9, 10, 0;
    LUCENE_9_11_0 = 9, 11, 0;
    LUCENE_9_11_1 = 9, 11, 1;
    LUCENE_9_12_0 = 9, 12, 0;
    LUCENE_9_12_1 = 9, 12, 1;
    LUCENE_9_12_2 = 9, 12, 2;
    LUCENE_9_12_3 = 9, 12, 3;
    LUCENE_9_12_4 = 9, 12, 4;
    LUCENE_10_0_0 = 10, 0, 0;
    LUCENE_10_1_0 = 10, 1, 0;
    LUCENE_10_2_0 = 10, 2, 0;
    LUCENE_10_2_1 = 10, 2, 1;
    LUCENE_10_2_2 = 10, 2, 2;
    LUCENE_10_3_0 = 10, 3, 0;
    LUCENE_10_3_1 = 10, 3, 1;
    LUCENE_10_3_2 = 10, 3, 2;
    LUCENE_10_4_0 = 10, 4, 0;
    LUCENE_10_5_0 = 10, 5, 0;
}

impl Version {
    /// `Version.LATEST`.
    pub const LATEST: Version = Version::LUCENE_10_5_0;
    /// `Version.LUCENE_CURRENT`.
    pub const LUCENE_CURRENT: Version = Version::LATEST;
    /// `Version.MIN_SUPPORTED_MAJOR`.
    pub const MIN_SUPPORTED_MAJOR: i32 = Version::LATEST.major() - 1;

    const fn constant(major: i32, minor: i32, bugfix: i32) -> Version {
        Version {
            encoded: (major << 18) | (minor << 10) | (bugfix << 2),
        }
    }

    /// The private `Version(major, minor, bugfix, prerelease)` constructor's
    /// checks; the error is Java's `IllegalArgumentException` message.
    pub fn new(major: i32, minor: i32, bugfix: i32, prerelease: i32) -> Result<Version, String> {
        if !(0..=255).contains(&major) {
            return Err(format!("Illegal major version: {major}"));
        }
        if !(0..=255).contains(&minor) {
            return Err(format!("Illegal minor version: {minor}"));
        }
        if !(0..=255).contains(&bugfix) {
            return Err(format!("Illegal bugfix version: {bugfix}"));
        }
        if !(0..=2).contains(&prerelease) {
            return Err(format!("Illegal prerelease version: {prerelease}"));
        }
        if prerelease != 0 && (minor != 0 || bugfix != 0) {
            return Err(format!(
                "Prerelease version only supported with major release (got prerelease: \
                 {prerelease}, minor: {minor}, bugfix: {bugfix})"
            ));
        }
        Ok(Version {
            encoded: (major << 18) | (minor << 10) | (bugfix << 2) | prerelease,
        })
    }

    /// `Version.fromBits(major, minor, bugfix)`.
    pub fn from_bits(major: i32, minor: i32, bugfix: i32) -> Result<Version, String> {
        Version::new(major, minor, bugfix, 0)
    }

    /// `Version.major`.
    pub const fn major(self) -> i32 {
        (self.encoded >> 18) & 0xff
    }
    /// `Version.minor`.
    pub const fn minor(self) -> i32 {
        (self.encoded >> 10) & 0xff
    }
    /// `Version.bugfix`.
    pub const fn bugfix(self) -> i32 {
        (self.encoded >> 2) & 0xff
    }
    /// `Version.prerelease`.
    pub const fn prerelease(self) -> i32 {
        self.encoded & 0x03
    }
    /// The packed `encodedValue` (also Java's `hashCode`).
    pub const fn encoded_value(self) -> i32 {
        self.encoded
    }

    /// `Version.onOrAfter(other)`.
    pub fn on_or_after(self, other: Version) -> bool {
        self.encoded >= other.encoded
    }

    /// `Version.parse`: `major.minor[.bugfix[.prerelease]]`.
    pub fn parse(version: &str) -> Result<Version, VersionParseError> {
        let form = || {
            VersionParseError(format!(
                "Version is not in form major.minor.bugfix(.prerelease) (got: {version})"
            ))
        };
        let num = |what: &str, token: &str| {
            parse_java_int(token).ok_or_else(|| {
                VersionParseError(format!(
                    "Failed to parse {what} version from \"{token}\" (got: {version})"
                ))
            })
        };
        // StrictStringTokenizer: a plain split, empty tokens kept.
        let mut tokens = version.split('.');
        let major = num("major", tokens.next().ok_or_else(form)?)?;
        let minor = num("minor", tokens.next().ok_or_else(form)?)?;
        let mut bugfix = 0;
        let mut prerelease = 0;
        if let Some(t) = tokens.next() {
            bugfix = num("bugfix", t)?;
            if let Some(t) = tokens.next() {
                prerelease = num("prerelease", t)?;
                if prerelease == 0 {
                    return Err(VersionParseError(format!(
                        "Invalid value 0 for prerelease; should be 1 or 2 (got: {version})"
                    )));
                }
                if tokens.next().is_some() {
                    return Err(form());
                }
            }
        }
        Version::new(major, minor, bugfix, prerelease).map_err(|e| {
            VersionParseError(format!("failed to parse version string \"{version}\": {e}"))
        })
    }

    /// `Version.parseLeniently`: also `LATEST`, `LUCENE_CURRENT`,
    /// `LUCENE_X_Y_Z`, `LUCENE_X_Y` and `LUCENE_XY`, case-insensitively.
    pub fn parse_leniently(version: &str) -> Result<Version, VersionParseError> {
        let upper = version.to_uppercase();
        if upper == "LATEST" || upper == "LUCENE_CURRENT" {
            return Ok(Version::LATEST);
        }
        let rewritten = lenient_rewrite(&upper);
        Version::parse(&rewritten).map_err(|pe| {
            VersionParseError(format!(
                "failed to parse lenient version string \"{version}\": {}",
                pe.0
            ))
        })
    }
}

/// The three `replaceFirst` patterns of `parseLeniently`, applied in order
/// (`\d` is ASCII-only in Java regexes by default).
fn lenient_rewrite(s: &str) -> String {
    let Some(rest) = s.strip_prefix("LUCENE_") else {
        return s.to_string();
    };
    let parts: Vec<&str> = rest.split('_').collect();
    let digits = |p: &str| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit());
    if parts.iter().all(|p| digits(p)) {
        match parts.len() {
            3 => return format!("{}.{}.{}", parts[0], parts[1], parts[2]),
            2 => return format!("{}.{}.0", parts[0], parts[1]),
            1 if parts[0].len() == 2 => return format!("{}.{}.0", &rest[..1], &rest[1..]),
            _ => {}
        }
    }
    s.to_string()
}

/// Java's `Integer.parseInt` over ASCII: optional sign, at least one digit,
/// no overflow.
fn parse_java_int(s: &str) -> Option<i32> {
    let digits = s.strip_prefix(['+', '-']).unwrap_or(s);
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    s.parse().ok()
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major(), self.minor(), self.bugfix())?;
        if self.prerelease() != 0 {
            write!(f, ".{}", self.prerelease())?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_and_order() {
        assert_eq!(Version::LATEST.to_string(), "10.5.0");
        assert_eq!(Version::MIN_SUPPORTED_MAJOR, 9);
        assert!(Version::LATEST.on_or_after(Version::LUCENE_9_12_4));
        assert!(!Version::LUCENE_9_0_0.on_or_after(Version::LUCENE_9_1_0));
        assert_eq!(Version::LUCENE_CURRENT, Version::LATEST);
        assert_eq!(Version::from_bits(10, 5, 0), Ok(Version::LATEST));
        assert_eq!(Version::LATEST.encoded_value(), (10 << 18) | (5 << 10));
        assert_eq!(Version::LUCENE_9_12_4.prerelease(), 0);
    }

    #[test]
    fn parse_accepts_and_rejects_like_java() {
        assert_eq!(Version::parse("10.5.0"), Ok(Version::LATEST));
        assert_eq!(Version::parse("10.5"), Ok(Version::LATEST));
        assert_eq!(Version::parse("+10.5"), Ok(Version::LATEST));
        let pre = Version::parse("11.0.0.2").unwrap();
        assert_eq!((pre.major(), pre.prerelease()), (11, 2));
        assert_eq!(pre.to_string(), "11.0.0.2");
        let err = |s: &str| Version::parse(s).unwrap_err().0;
        assert_eq!(
            err("10"),
            "Version is not in form major.minor.bugfix(.prerelease) (got: 10)"
        );
        assert_eq!(
            err("x.1"),
            "Failed to parse major version from \"x\" (got: x.1)"
        );
        assert_eq!(
            err("1."),
            "Failed to parse minor version from \"\" (got: 1.)"
        );
        assert_eq!(
            err("1.2.b"),
            "Failed to parse bugfix version from \"b\" (got: 1.2.b)"
        );
        assert_eq!(
            err("1.0.0.q"),
            "Failed to parse prerelease version from \"q\" (got: 1.0.0.q)"
        );
        assert!(err("1.0.0.0").starts_with("Invalid value 0 for prerelease"));
        assert!(err("1.0.0.1.1").starts_with("Version is not in form"));
        assert_eq!(
            err("256.0"),
            "failed to parse version string \"256.0\": Illegal major version: 256"
        );
        assert!(err("1.-1").ends_with("Illegal minor version: -1"));
        assert!(err("1.1.300").ends_with("Illegal bugfix version: 300"));
        assert!(err("1.0.0.3").ends_with("Illegal prerelease version: 3"));
        assert!(err("1.1.0.1").contains("Prerelease version only supported"));
        assert!(err("99999999999.0").starts_with("Failed to parse major"));
        assert!(err("-.0").starts_with("Failed to parse major"));
    }

    #[test]
    fn parse_leniently_forms() {
        let ok = |s: &str| Version::parse_leniently(s).unwrap();
        assert_eq!(ok("latest"), Version::LATEST);
        assert_eq!(ok("LUCENE_CURRENT"), Version::LATEST);
        assert_eq!(ok("lucene_10_5_0"), Version::LATEST);
        assert_eq!(ok("LUCENE_10_5"), Version::LATEST);
        assert_eq!(ok("LUCENE_95").to_string(), "9.5.0");
        assert_eq!(ok("9.12.4"), Version::LUCENE_9_12_4);
        let e = Version::parse_leniently("LUCENE_1_2_3_4").unwrap_err().0;
        assert!(
            e.starts_with("failed to parse lenient version string \"LUCENE_1_2_3_4\""),
            "{e}"
        );
        assert!(Version::parse_leniently("LUCENE_123").is_err());
        assert!(Version::parse_leniently("LUCENE_1_x").is_err());
        assert!(Version::parse_leniently("foo").is_err());
        assert_eq!(VersionParseError("m".into()).to_string(), "m");
    }
}
