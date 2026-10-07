//! A factory's argument map and `AbstractAnalysisFactory`'s parsing helpers
//! (`require`, `get`, `getInt`, `getFloat`, `getBoolean`, `getChar`,
//! `getSet`, `getPattern`, `splitFileNames`, `splitAt`).
//!
//! # The map
//!
//! Java factories take a `Map<String, String>` and consume it: each helper
//! `remove`s the key it reads, and whatever is left is an error whose message
//! prints the map (`"Unknown parameters: " + args`). That message, like
//! `ProtectedTermFilterFactory`'s walk over the leftovers, follows the map's
//! iteration order, which for the `java.util.HashMap` every caller builds
//! (`CustomAnalyzer.Builder`'s `paramsToMap`, `new HashMap<>()`) is its
//! bucket order. [`JavaArgs`] reproduces it: Java's `String.hashCode`, the
//! spread `h ^ (h >>> 16)`, power-of-two tables grown at a 0.75 load
//! factor, insertion order within a bucket, and `treeifyBin`'s resize when
//! a bucket reaches nine entries in a table under 64 buckets. It does not
//! model the tree such a bucket becomes in a table of 64 or more (its root
//! moves to the front of the bucket's order): that takes nine keys whose
//! spread hashes agree in their low six bits, and the map keeps listing
//! that bucket in insertion order.
//!
//! # Java's number and boolean parsing
//!
//! `Integer.parseInt` takes an optional sign and `Character.digit(c, 10)`
//! digits -- every BMP decimal digit, not only ASCII -- and rejects overflow;
//! `Float.parseFloat` is [`crate::payloads::parse_java_float`] (no
//! hexadecimal literals); `Boolean.parseBoolean` is `"true"` ignoring case,
//! anything else `false`.

use std::collections::BTreeSet;

use super::{FactoryError, JavaException};
use crate::java_character;
use crate::util::JavaPattern;

/// `java.util.HashMap.DEFAULT_INITIAL_CAPACITY`.
const DEFAULT_INITIAL_CAPACITY: usize = 16;

/// `HashMap.TREEIFY_THRESHOLD`: a bucket holding more entries than this
/// after an insertion is treeified -- or, in a table under
/// [`MIN_TREEIFY_CAPACITY`], the table resized instead.
const TREEIFY_THRESHOLD: usize = 8;

/// `HashMap.MIN_TREEIFY_CAPACITY`.
const MIN_TREEIFY_CAPACITY: usize = 64;

/// `String.hashCode()`: `s[0]*31^(n-1) + ... + s[n-1]` over UTF-16 units,
/// wrapping.
pub fn java_string_hash(s: &str) -> i32 {
    s.encode_utf16()
        .fold(0i32, |h, u| h.wrapping_mul(31).wrapping_add(i32::from(u)))
}

/// `HashMap.hash(key)`: the hash code with its high half folded down.
fn spread(s: &str) -> u32 {
    let h = java_string_hash(s) as u32;
    h ^ (h >> 16)
}

/// `HashMap.tableSizeFor`: the smallest power of two at or above `cap`
/// (at least 1).
fn table_size_for(cap: usize) -> usize {
    cap.max(1).next_power_of_two()
}

/// A `java.util.HashMap<String, String>` that iterates in Java's order (see
/// the module docs).
#[derive(Debug, Clone, Default)]
pub struct JavaArgs {
    /// The buckets; empty until the first insertion (Java allocates lazily).
    table: Vec<Vec<(String, String)>>,
    /// The capacity the first insertion allocates (`threshold` before the
    /// table exists).
    initial_capacity: usize,
    len: usize,
}

impl JavaArgs {
    /// `new HashMap<>()`.
    pub fn new() -> Self {
        JavaArgs {
            table: Vec::new(),
            initial_capacity: DEFAULT_INITIAL_CAPACITY,
            len: 0,
        }
    }

    /// `HashMap.newHashMap(numMappings)`: room for `num_mappings` entries
    /// without a resize.
    pub fn with_expected(num_mappings: usize) -> Self {
        // calculateHashMapCapacity: (int) ceil(numMappings / 0.75)
        let cap = num_mappings.saturating_mul(4).div_ceil(3);
        JavaArgs {
            table: Vec::new(),
            initial_capacity: table_size_for(cap),
            len: 0,
        }
    }

    /// `CustomAnalyzer.Builder.paramsToMap(String...)`: a map sized for the
    /// pairs plus one (the `luceneMatchVersion` a builder may add), filled in
    /// order.
    pub fn from_pairs<K: AsRef<str>, V: AsRef<str>>(pairs: &[(K, V)]) -> Self {
        let mut map = Self::with_expected(pairs.len() + 1);
        for (k, v) in pairs {
            map.put(k.as_ref(), v.as_ref());
        }
        map
    }

    /// `size()`.
    pub fn len(&self) -> usize {
        self.len
    }

    /// `isEmpty()`.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn bucket(&self, key: &str) -> Option<usize> {
        if self.table.is_empty() {
            return None;
        }
        Some(spread(key) as usize & (self.table.len() - 1))
    }

    /// `get(key)`.
    pub fn get(&self, key: &str) -> Option<&str> {
        let b = self.bucket(key)?;
        self.table[b]
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    /// `containsKey(key)`.
    pub fn contains_key(&self, key: &str) -> bool {
        self.get(key).is_some()
    }

    /// `put(key, value)`: the previous value, if any.
    pub fn put(&mut self, key: &str, value: &str) -> Option<String> {
        self.insert(key, value, true)
    }

    /// `putIfAbsent(key, value)`: the present value, if any.
    pub fn put_if_absent(&mut self, key: &str, value: &str) -> Option<String> {
        self.insert(key, value, false)
    }

    fn insert(&mut self, key: &str, value: &str, replace: bool) -> Option<String> {
        if self.table.is_empty() {
            self.table = vec![Vec::new(); self.initial_capacity];
        }
        let b = spread(key) as usize & (self.table.len() - 1);
        if let Some(slot) = self.table[b].iter_mut().find(|(k, _)| k == key) {
            let old = slot.1.clone();
            if replace {
                slot.1 = value.to_string();
            }
            return Some(old);
        }
        self.table[b].push((key.to_string(), value.to_string()));
        // putVal: `binCount >= TREEIFY_THRESHOLD - 1` -> treeifyBin, which
        // resizes a small table (a large one's tree is not modelled).
        if self.table[b].len() > TREEIFY_THRESHOLD && self.table.len() < MIN_TREEIFY_CAPACITY {
            self.resize();
        }
        self.len += 1;
        // threshold = (int) (capacity * 0.75)
        if self.len > self.table.len() * 3 / 4 {
            self.resize();
        }
        None
    }

    /// `resize()`: doubles the table. Java splits each bucket into a low and
    /// a high list keeping their order, which is what re-inserting the
    /// entries bucket by bucket produces.
    fn resize(&mut self) {
        let cap = self.table.len() * 2;
        let old = std::mem::replace(&mut self.table, vec![Vec::new(); cap]);
        for (k, v) in old.into_iter().flatten() {
            let b = spread(&k) as usize & (cap - 1);
            self.table[b].push((k, v));
        }
    }

    /// `remove(key)`: the removed value.
    pub fn remove(&mut self, key: &str) -> Option<String> {
        let b = self.bucket(key)?;
        let i = self.table[b].iter().position(|(k, _)| k == key)?;
        self.len -= 1;
        Some(self.table[b].remove(i).1)
    }

    /// The entries in Java's iteration order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.table
            .iter()
            .flatten()
            .map(|(k, v)| (k.as_str(), v.as_str()))
    }

    /// The keys in Java's iteration order.
    pub fn keys(&self) -> Vec<String> {
        self.iter().map(|(k, _)| k.to_string()).collect()
    }
}

/// `AbstractMap.toString()`: `{k1=v1, k2=v2}` in iteration order.
impl std::fmt::Display for JavaArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("{")?;
        for (i, (k, v)) in self.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{k}={v}")?;
        }
        f.write_str("}")
    }
}

// ------------------------------------------------------------ the helpers

/// The `"Unknown parameters: " + args` check every factory ends with.
pub fn reject_unknown(args: &JavaArgs) -> Result<(), FactoryError> {
    reject_unknown_with(args, "Unknown parameters: ")
}

/// [`reject_unknown`] with the prefix the few factories that spell it
/// differently use (`"Unknown parameter(s): "`).
pub fn reject_unknown_with(args: &JavaArgs, prefix: &str) -> Result<(), FactoryError> {
    if args.is_empty() {
        Ok(())
    } else {
        Err(FactoryError::illegal_argument(format!("{prefix}{args}")))
    }
}

fn missing(name: &str) -> FactoryError {
    FactoryError::illegal_argument(format!("Configuration Error: missing parameter '{name}'"))
}

/// `Collection.toString()` of a list: `[a, b]`.
fn list_string(values: &[&str]) -> String {
    format!("[{}]", values.join(", "))
}

/// `String.equalsIgnoreCase`: unit by unit, equal after `toUpperCase` or
/// after `toLowerCase` of the uppercased units.
pub fn java_equals_ignore_case(a: &str, b: &str) -> bool {
    let a: Vec<u16> = a.encode_utf16().collect();
    let b: Vec<u16> = b.encode_utf16().collect();
    a.len() == b.len()
        && a.iter().zip(&b).all(|(&x, &y)| {
            if x == y {
                return true;
            }
            let (ux, uy) = (
                java_character::to_upper_case(u32::from(x)),
                java_character::to_upper_case(u32::from(y)),
            );
            ux == uy || java_character::to_lower_case(ux) == java_character::to_lower_case(uy)
        })
}

fn one_of(
    name: &str,
    s: String,
    allowed: &[&str],
    case_sensitive: bool,
) -> Result<String, FactoryError> {
    let ok = allowed.iter().any(|a| {
        if case_sensitive {
            s == *a
        } else {
            java_equals_ignore_case(&s, a)
        }
    });
    if ok {
        Ok(s)
    } else {
        Err(FactoryError::illegal_argument(format!(
            "Configuration Error: '{name}' value must be one of {}",
            list_string(allowed)
        )))
    }
}

/// `require(args, name)`.
pub fn require(args: &mut JavaArgs, name: &str) -> Result<String, FactoryError> {
    args.remove(name).ok_or_else(|| missing(name))
}

/// `require(args, name, allowedValues, caseSensitive)`.
pub fn require_one_of(
    args: &mut JavaArgs,
    name: &str,
    allowed: &[&str],
    case_sensitive: bool,
) -> Result<String, FactoryError> {
    let s = require(args, name)?;
    one_of(name, s, allowed, case_sensitive)
}

/// `get(args, name)`.
pub fn get(args: &mut JavaArgs, name: &str) -> Option<String> {
    args.remove(name)
}

/// `get(args, name, defaultVal)`.
pub fn get_or(args: &mut JavaArgs, name: &str, default: &str) -> String {
    args.remove(name).unwrap_or_else(|| default.to_string())
}

/// `get(args, name, allowedValues, defaultVal, caseSensitive)`.
pub fn get_one_of(
    args: &mut JavaArgs,
    name: &str,
    allowed: &[&str],
    default: Option<&str>,
    case_sensitive: bool,
) -> Result<Option<String>, FactoryError> {
    match args.remove(name) {
        None => Ok(default.map(str::to_string)),
        Some(s) => one_of(name, s, allowed, case_sensitive).map(Some),
    }
}

/// `NumberFormatException.forInputString(s, 10)`.
fn number_format(s: &str) -> FactoryError {
    FactoryError::new(
        JavaException::NumberFormat,
        format!("For input string: \"{s}\""),
    )
}

/// `Integer.parseInt(s)`: an optional sign and `Character.digit(c, 10)`
/// digits (any BMP decimal digit), no overflow.
pub fn parse_java_int(s: &str) -> Result<i32, FactoryError> {
    let units: Vec<u16> = s.encode_utf16().collect();
    let (negative, digits) = match units.first() {
        Some(&u) if u == u16::from(b'-') => (true, &units[1..]),
        Some(&u) if u == u16::from(b'+') => (false, &units[1..]),
        _ => (false, &units[..]),
    };
    if digits.is_empty() {
        return Err(number_format(s));
    }
    // Accumulated negatively, as Java does, so i32::MIN parses.
    let mut result: i32 = 0;
    for &u in digits {
        let d =
            java_character::decimal_digit_value(u32::from(u)).ok_or_else(|| number_format(s))?;
        result = result
            .checked_mul(10)
            .and_then(|r| r.checked_sub(d as i32))
            .ok_or_else(|| number_format(s))?;
    }
    if negative {
        Ok(result)
    } else {
        result.checked_neg().ok_or_else(|| number_format(s))
    }
}

/// `Float.parseFloat(s)`.
pub fn parse_java_float(s: &str) -> Result<f32, FactoryError> {
    let trimmed = s.trim_matches(|c: char| c <= ' ');
    if trimmed.is_empty() {
        return Err(FactoryError::new(
            JavaException::NumberFormat,
            "empty String",
        ));
    }
    crate::payloads::parse_java_float(trimmed).map_err(|_| number_format(trimmed))
}

/// `Boolean.parseBoolean(s)`.
pub fn parse_java_boolean(s: &str) -> bool {
    java_equals_ignore_case(s, "true")
}

/// `requireInt(args, name)`.
pub fn require_int(args: &mut JavaArgs, name: &str) -> Result<i32, FactoryError> {
    parse_java_int(&require(args, name)?)
}

/// `getInt(args, name, defaultVal)`.
pub fn get_int(args: &mut JavaArgs, name: &str, default: i32) -> Result<i32, FactoryError> {
    match args.remove(name) {
        None => Ok(default),
        Some(s) => parse_java_int(&s),
    }
}

/// `requireBoolean(args, name)`.
pub fn require_boolean(args: &mut JavaArgs, name: &str) -> Result<bool, FactoryError> {
    Ok(parse_java_boolean(&require(args, name)?))
}

/// `getBoolean(args, name, defaultVal)`.
pub fn get_boolean(args: &mut JavaArgs, name: &str, default: bool) -> bool {
    args.remove(name)
        .map_or(default, |s| parse_java_boolean(&s))
}

/// `requireFloat(args, name)`.
pub fn require_float(args: &mut JavaArgs, name: &str) -> Result<f32, FactoryError> {
    parse_java_float(&require(args, name)?)
}

/// `getFloat(args, name, defaultVal)`.
pub fn get_float(args: &mut JavaArgs, name: &str, default: f32) -> Result<f32, FactoryError> {
    match args.remove(name) {
        None => Ok(default),
        Some(s) => parse_java_float(&s),
    }
}

/// `requireChar(args, name)`: the value's first UTF-16 unit. An empty value
/// is Java's `StringIndexOutOfBoundsException`.
pub fn require_char(args: &mut JavaArgs, name: &str) -> Result<u16, FactoryError> {
    let s = require(args, name)?;
    s.encode_utf16().next().ok_or_else(|| {
        FactoryError::new(
            JavaException::StringIndexOutOfBounds,
            "Index 0 out of bounds for length 0",
        )
    })
}

/// `getChar(args, name, defaultValue)`: a value must be exactly one UTF-16
/// unit.
pub fn get_char(args: &mut JavaArgs, name: &str, default: u16) -> Result<u16, FactoryError> {
    match args.remove(name) {
        None => Ok(default),
        Some(s) => {
            let mut units = s.encode_utf16();
            match (units.next(), units.next()) {
                (Some(u), None) => Ok(u),
                _ => Err(FactoryError::illegal_argument(format!(
                    "{name} should be a char. \"{s}\" is invalid"
                ))),
            }
        }
    }
}

/// A UTF-16 unit as a `char`: [`get_char`] only returns a whole BMP scalar
/// (a lone surrogate cannot come from a Rust string).
pub fn unit_char(u: u16) -> char {
    char::from_u32(u32::from(u)).unwrap_or(char::REPLACEMENT_CHARACTER)
}

/// Java's default `\s`: `[ \t\n\x0B\f\r]`.
pub(crate) fn is_java_space(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\u{0B}' | '\u{0C}' | '\r')
}

/// `getSet(args, name)`: the `[^,\s]+` items of the value, as a set; `None`
/// when the key is absent, and also when the value holds no item.
pub fn get_set(args: &mut JavaArgs, name: &str) -> Option<BTreeSet<String>> {
    let s = args.remove(name)?;
    let set: BTreeSet<String> = s
        .split(|c: char| c == ',' || is_java_space(c))
        .filter(|item| !item.is_empty())
        .map(str::to_string)
        .collect();
    (!set.is_empty()).then_some(set)
}

/// `getPattern(args, name)`: a pattern Java refuses is an
/// `IllegalArgumentException` naming the factory's simple class name; one
/// Java compiles and the regex shim cannot ([`crate::util::java_regex`]'s
/// limits) is an `UnsupportedOperationException` with the shim's reason.
pub fn get_pattern(
    args: &mut JavaArgs,
    name: &str,
    simple_class_name: &str,
) -> Result<JavaPattern, FactoryError> {
    let s = require(args, name)?;
    JavaPattern::compile(&s).map_err(|e| match unsupported_pattern(&e) {
        Some(err) => err,
        None => FactoryError::illegal_argument(format!(
            "Configuration Error: '{name}' can not be parsed in {simple_class_name}"
        )),
    })
}

/// The shim's refusal of a pattern Java compiles, as an
/// `UnsupportedOperationException` carrying its reason; `None` for a
/// pattern Java refuses too.
pub(crate) fn unsupported_pattern(e: &crate::AnalysisError) -> Option<FactoryError> {
    match e {
        crate::AnalysisError::IllegalArgument(m) if crate::util::java_regex::is_unsupported(e) => {
            Some(FactoryError::new(
                JavaException::UnsupportedOperation,
                m.clone(),
            ))
        }
        _ => None,
    }
}

/// `splitFileNames(fileNames)`: [`split_at`] on `,`.
pub fn split_file_names(file_names: Option<&str>) -> Vec<String> {
    split_at(',', file_names)
}

/// `splitAt(separator, list)`: `list.split("(?<!\\\\)[sep]")` -- trailing
/// empty items dropped, as `String.split` does -- with each `\` before a
/// separator removed. `None` is an empty list.
pub fn split_at(separator: char, list: Option<&str>) -> Vec<String> {
    let Some(list) = list else {
        return Vec::new();
    };
    let mut items = Vec::new();
    let mut current = String::new();
    let mut prev_backslash = false;
    for c in list.chars() {
        if c == separator && !prev_backslash {
            items.push(std::mem::take(&mut current));
        } else {
            current.push(c);
        }
        prev_backslash = c == '\\';
    }
    items.push(current);
    // String.split drops trailing empty strings, unless nothing matched.
    if items.len() > 1 {
        while items.len() > 1 && items.last().is_some_and(String::is_empty) {
            items.pop();
        }
        if items.len() == 1 && items[0].is_empty() {
            items.clear();
        }
    }
    let escaped = format!("\\{separator}");
    items
        .into_iter()
        .map(|item| item.replace(&escaped, &separator.to_string()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iteration_follows_javas_buckets() {
        // Java: HashMap.newHashMap(4) = 8 buckets; "b" (98) and "j" (106)
        // share bucket 2, in insertion order; "a" is bucket 1.
        let map = JavaArgs::from_pairs(&[("j", "1"), ("a", "2"), ("b", "3")]);
        assert_eq!(map.to_string(), "{a=2, j=1, b=3}");
        assert_eq!(map.keys(), vec!["a", "j", "b"]);
        assert_eq!(JavaArgs::new().to_string(), "{}");
    }

    #[test]
    fn resizing_keeps_javas_order() {
        // new HashMap<>() holds 12 entries in 16 buckets, then doubles.
        let mut map = JavaArgs::new();
        let keys: Vec<String> = (0..30).map(|i| format!("key{i}")).collect();
        for k in &keys {
            map.put(k, "v");
        }
        assert_eq!(map.len(), 30);
        assert_eq!(map.table.len(), 64);
        let order = map.keys();
        let mut sorted_by_bucket = keys.clone();
        sorted_by_bucket.sort_by_key(|k| spread(k) as usize & 63);
        assert_eq!(order, sorted_by_bucket);
        assert_eq!(map.put("key3", "w"), Some("v".to_string()));
        assert_eq!(map.put_if_absent("key3", "x"), Some("w".to_string()));
        assert_eq!(map.get("key3"), Some("w"));
        assert_eq!(map.remove("key3"), Some("w".to_string()));
        assert_eq!(map.remove("key3"), None);
        assert!(!map.contains_key("key3"));
        assert_eq!(JavaArgs::with_expected(0).initial_capacity, 1);
        let mut one = JavaArgs::with_expected(0);
        one.put("a", "1");
        one.put("b", "2");
        assert_eq!(one.len(), 2);
        assert!(JavaArgs::default().get("x").is_none());
    }

    #[test]
    fn a_ninth_colliding_key_resizes_a_small_table() {
        // These nine keys share a bucket of 16 (and of 32 the low four bits
        // agree); Java's putVal resizes on the ninth (treeifyBin under 64
        // buckets), long before 0.75 of the table is used.
        let keys = [
            "k18", "k29", "k90", "k140", "k151", "k162", "k173", "k184", "k195",
        ];
        let pairs: Vec<(&str, &str)> = keys.iter().map(|k| (*k, "v")).collect();
        let map = JavaArgs::from_pairs(&pairs);
        assert_eq!(map.table.len(), 32);
        assert_eq!(
            map.to_string(),
            "{k90=v, k18=v, k29=v, k140=v, k151=v, k162=v, k173=v, k184=v, k195=v}"
        );
    }

    #[test]
    fn string_hash_is_javas() {
        assert_eq!(java_string_hash(""), 0);
        assert_eq!(java_string_hash("ignoreCase"), 880_063_522);
        // A supplementary character hashes as its two surrogates.
        assert_eq!(java_string_hash("\u{1F600}"), 0xD83D * 31 + 0xDE00);
    }

    #[test]
    fn integers_parse_like_integer_parse_int() {
        assert_eq!(parse_java_int("42").unwrap(), 42);
        assert_eq!(parse_java_int("+7").unwrap(), 7);
        assert_eq!(parse_java_int("-2147483648").unwrap(), i32::MIN);
        assert_eq!(parse_java_int("\u{0663}\u{0664}").unwrap(), 34);
        for bad in ["", "-", "+", "2147483648", "1.5", " 1", "x"] {
            let e = parse_java_int(bad).unwrap_err();
            assert_eq!(e.kind, JavaException::NumberFormat);
            assert_eq!(e.message, format!("For input string: \"{bad}\""));
        }
    }

    #[test]
    fn floats_and_booleans() {
        assert_eq!(parse_java_float(" 1.5 ").unwrap(), 1.5);
        assert_eq!(parse_java_float("").unwrap_err().message, "empty String");
        assert_eq!(
            parse_java_float(" x ").unwrap_err().message,
            "For input string: \"x\""
        );
        assert!(parse_java_boolean("TRUE"));
        assert!(!parse_java_boolean("yes"));
        assert!(java_equals_ignore_case("Straße", "STRAßE"));
        assert!(!java_equals_ignore_case("a", "ab"));
    }

    #[test]
    fn helpers_consume_their_keys() {
        let mut a = JavaArgs::from_pairs(&[
            ("n", "3"),
            ("b", "True"),
            ("f", "2.5"),
            ("c", "|"),
            ("cc", "ab"),
            ("s", " x, y ,x "),
            ("e", " , "),
            ("r", "first"),
            ("p", "a+"),
            ("bad", "("),
        ]);
        assert_eq!(require_int(&mut a, "n").unwrap(), 3);
        assert!(require_boolean(&mut a, "b").unwrap());
        assert_eq!(require_float(&mut a, "f").unwrap(), 2.5);
        assert_eq!(get_char(&mut a, "c", b'x' as u16).unwrap(), b'|' as u16);
        assert_eq!(
            get_char(&mut a, "cc", 0).unwrap_err().message,
            "cc should be a char. \"ab\" is invalid"
        );
        let set = get_set(&mut a, "s").unwrap();
        assert_eq!(set.into_iter().collect::<Vec<_>>(), vec!["x", "y"]);
        assert!(get_set(&mut a, "e").is_none());
        assert!(get_set(&mut a, "e").is_none());
        assert_eq!(
            get_one_of(&mut a, "r", &["all", "first"], None, true).unwrap(),
            Some("first".to_string())
        );
        assert_eq!(
            get_one_of(&mut a, "r", &["all"], Some("all"), true).unwrap(),
            Some("all".to_string())
        );
        assert!(get_pattern(&mut a, "p", "X").is_ok());
        assert_eq!(
            get_pattern(&mut a, "bad", "PatternTokenizerFactory")
                .unwrap_err()
                .message,
            "Configuration Error: 'bad' can not be parsed in PatternTokenizerFactory"
        );
        // Patterns Java compiles and the shim cannot are not Java's error.
        for (p, why) in [
            ("(a)\\1", "backreferences"),
            ("a(?=b)", "lookaround"),
            ("(?>ab)", "atomic groups"),
            ("a*+", "possessive quantifiers"),
            ("a\\Z", "an escape"),
        ] {
            let mut u = JavaArgs::from_pairs(&[("p", p)]);
            let e = get_pattern(&mut u, "p", "PatternTokenizerFactory").unwrap_err();
            assert_eq!(e.kind, JavaException::UnsupportedOperation, "{p}");
            assert!(e.message.contains(why), "{p}: {}", e.message);
        }
        assert!(a.is_empty());
        assert_eq!(
            require(&mut a, "n").unwrap_err().message,
            "Configuration Error: missing parameter 'n'"
        );
        assert_eq!(get_int(&mut a, "n", 9).unwrap(), 9);
        assert!(get_boolean(&mut a, "b", true));
        assert_eq!(get_float(&mut a, "f", 0.5).unwrap(), 0.5);
        assert_eq!(get_or(&mut a, "x", "d"), "d");
        assert!(get(&mut a, "x").is_none());
        assert_eq!(get_char(&mut a, "c", 7).unwrap(), 7);
        assert!(reject_unknown(&a).is_ok());
        a.put("x", "1");
        assert_eq!(
            reject_unknown(&a).unwrap_err().message,
            "Unknown parameters: {x=1}"
        );
        assert_eq!(get_int(&mut a, "x", 0).unwrap(), 1);
    }

    #[test]
    fn allowed_values() {
        let mut a = JavaArgs::from_pairs(&[("r", "ALL"), ("q", "ALL"), ("z", "")]);
        assert_eq!(
            require_one_of(&mut a, "r", &["all", "first"], false).unwrap(),
            "ALL"
        );
        assert_eq!(
            require_one_of(&mut a, "q", &["all", "first"], true)
                .unwrap_err()
                .message,
            "Configuration Error: 'q' value must be one of [all, first]"
        );
        assert_eq!(
            require_char(&mut a, "z").unwrap_err().kind,
            JavaException::StringIndexOutOfBounds
        );
        a.put("z", "ab");
        assert_eq!(require_char(&mut a, "z").unwrap(), u16::from(b'a'));
        assert_eq!(unit_char(0x41), 'A');
        assert_eq!(unit_char(0xD800), char::REPLACEMENT_CHARACTER);
    }

    #[test]
    fn split_at_matches_string_split() {
        assert_eq!(split_at(',', None), Vec::<String>::new());
        assert_eq!(split_at(',', Some("")), vec![""]);
        assert_eq!(split_at(',', Some("a,b")), vec!["a", "b"]);
        assert_eq!(split_at(',', Some("a,,")), vec!["a"]);
        assert_eq!(split_at(',', Some(",a")), vec!["", "a"]);
        assert_eq!(split_at(',', Some(",,")), Vec::<String>::new());
        assert_eq!(split_at(',', Some("a\\,b,c")), vec!["a,b", "c"]);
        assert_eq!(split_at('.', Some("f.k")), vec!["f", "k"]);
        assert_eq!(
            split_file_names(Some("x.txt, y.txt")),
            vec!["x.txt", " y.txt"]
        );
    }
}
