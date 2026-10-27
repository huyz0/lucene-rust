//! `LuceneTestCase`'s randomized configuration: the per-test seed, the
//! `RANDOM_MULTIPLIER`/`TEST_NIGHTLY` knobs, `atLeast`, `rarely`, `usually`,
//! the line that reproduces a failure, and `newField`/`newStringField`/
//! `newTextField` ([`new_field`]: Java's random stored/term-vector upgrade,
//! remembered per field name in a [`FieldTypes`]).
//!
//! Java gets its seeds from randomizedtesting's runner, which seeds each test
//! method from the master seed (`-Dtests.seed`). There is no runner here, so
//! [`LuceneTestCase::new`] does the same job in a few lines: the master seed
//! comes from `TESTS_SEED` (hex, as Lucene prints it; a fresh one from the
//! clock otherwise), and each test's seed is the master seed XOR the Java
//! `String.hashCode` of the test's name -- so two tests never share a stream
//! and one test's stream does not move when another is added. The seed is
//! not randomizedtesting's mix, so a Java seed does not reproduce a Rust run
//! or vice versa; the generators themselves draw what Java draws from a
//! given [`JavaRandom`], which is what the differential tests compare.
//!
//! Environment knobs (Java's system properties):
//!
//! | Rust env | Java property | default |
//! |---|---|---|
//! | `TESTS_SEED` | `tests.seed` | from the clock |
//! | `TESTS_NIGHTLY` | `tests.nightly` | `false` |
//! | `TESTS_MULTIPLIER` | `tests.multiplier` | `2` nightly, else `1` |
//! | `TESTS_LINEDOCSFILE` | `tests.linedocsfile` | the committed excerpt ([`super::line_file_docs`]) |
//!
//! When a test panics while its [`LuceneTestCase`] is alive, the drop prints
//! the reproduce line to stderr, the way Lucene prints `NOTE: reproduce with`.

use std::collections::HashMap;

use lucene_index::document::{Field, FieldType, IndexOptions, Store, StringField, TextField};
use lucene_util::java_random::JavaRandom;
use lucene_util::strict_math;

use super::test_util;

/// `LuceneTestCase.DEFAULT_LINE_DOCS_FILE`.
pub const DEFAULT_LINE_DOCS_FILE: &str = "europarl.lines.txt.gz";

/// Java `String.hashCode` of `s`, over its UTF-16 code units.
pub fn java_string_hash(s: &str) -> i32 {
    s.encode_utf16()
        .fold(0i32, |h, c| h.wrapping_mul(31).wrapping_add(i32::from(c)))
}

/// Parses a seed as Lucene prints it: hex, optionally `MASTER:METHOD` (the
/// master part is used). `None` when it is not hex.
pub fn parse_seed(s: &str) -> Option<i64> {
    let master = s.split(':').next().unwrap_or("").trim();
    if master.is_empty() {
        return None;
    }
    u64::from_str_radix(master, 16).ok().map(|v| v as i64)
}

/// Formats a seed the way Lucene prints it: upper-case hex.
pub fn format_seed(seed: i64) -> String {
    format!("{:X}", seed as u64)
}

/// The randomized configuration of one test: its name, seeds, knobs and
/// random stream. Create it first thing in a test; keep it alive for the
/// test's duration so a panic prints the reproduce line.
#[derive(Debug)]
pub struct LuceneTestCase {
    name: String,
    master_seed: i64,
    seed: i64,
    nightly: bool,
    multiplier: i32,
    random: JavaRandom,
    /// `fieldToType`: the type [`Self::new_field`] settled on per name.
    field_types: FieldTypes,
    /// `cargo test ... -- --exact <path>`: what reruns this test alone.
    command: String,
}

impl LuceneTestCase {
    /// The configuration for test `name`, from the environment.
    ///
    /// # Panics
    ///
    /// When `TESTS_SEED` or `TESTS_MULTIPLIER` is set but does not parse:
    /// silently running a different seed than the one asked for would make
    /// the reproduce line a lie.
    pub fn new(name: &str) -> Self {
        let env = |k: &str| std::env::var(k).ok();
        Self::from_vars(
            name,
            env("TESTS_SEED").as_deref(),
            env("TESTS_NIGHTLY").as_deref(),
            env("TESTS_MULTIPLIER").as_deref(),
        )
    }

    /// [`Self::new`] over explicit variable values (`None` = unset).
    pub fn from_vars(
        name: &str,
        seed: Option<&str>,
        nightly: Option<&str>,
        multiplier: Option<&str>,
    ) -> Self {
        let master_seed = match seed {
            Some(s) => parse_seed(s).unwrap_or_else(|| panic!("TESTS_SEED is not hex: {s:?}")),
            None => clock_seed(),
        };
        let nightly = nightly.is_some_and(parse_bool);
        let multiplier = match multiplier {
            Some(s) => s
                .trim()
                .parse()
                .unwrap_or_else(|_| panic!("TESTS_MULTIPLIER is not an int: {s:?}")),
            None => default_multiplier(nightly),
        };
        Self::with_config(name, master_seed, nightly, multiplier)
    }

    /// The configuration for test `name` with explicit knobs, ignoring the
    /// environment.
    pub fn with_config(name: &str, master_seed: i64, nightly: bool, multiplier: i32) -> Self {
        let seed = master_seed ^ i64::from(java_string_hash(name));
        Self {
            name: name.to_string(),
            master_seed,
            seed,
            nightly,
            multiplier,
            random: JavaRandom::new(seed),
            field_types: FieldTypes::default(),
            command: cargo_test_command(
                std::env::var("CARGO_PKG_NAME").ok().as_deref(),
                std::env::current_exe().ok().as_deref(),
                std::thread::current().name(),
                name,
            ),
        }
    }

    /// `LuceneTestCase.random()`: this test's stream.
    pub fn random(&mut self) -> &mut JavaRandom {
        &mut self.random
    }

    /// The test's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The master seed (`TESTS_SEED`).
    pub fn master_seed(&self) -> i64 {
        self.master_seed
    }

    /// This test's own seed, the one [`Self::random`] started from.
    pub fn seed(&self) -> i64 {
        self.seed
    }

    /// `TEST_NIGHTLY`.
    pub fn nightly(&self) -> bool {
        self.nightly
    }

    /// `RANDOM_MULTIPLIER`.
    pub fn multiplier(&self) -> i32 {
        self.multiplier
    }

    /// `atLeast(i)` on this test's stream.
    pub fn at_least(&mut self, i: i32) -> i32 {
        at_least(&mut self.random, self.multiplier, i)
    }

    /// `rarely()` on this test's stream.
    pub fn rarely(&mut self) -> bool {
        rarely(&mut self.random, self.nightly, self.multiplier)
    }

    /// `usually()` on this test's stream.
    pub fn usually(&mut self) -> bool {
        !self.rarely()
    }

    /// `newField(name, value, type)` on this test's stream and field types.
    pub fn new_field(&mut self, name: &str, value: FieldValue<'_>, ft: &FieldType) -> Field {
        new_field(&mut self.random, &mut self.field_types, name, value, ft)
    }

    /// `newStringField(name, value, stored)` on this test's stream.
    pub fn new_string_field(&mut self, name: &str, value: FieldValue<'_>, stored: Store) -> Field {
        new_string_field(&mut self.random, &mut self.field_types, name, value, stored)
    }

    /// `newTextField(name, value, stored)` on this test's stream.
    pub fn new_text_field(&mut self, name: &str, value: &str, stored: Store) -> Field {
        new_text_field(&mut self.random, &mut self.field_types, name, value, stored)
    }

    /// The command line that reruns this test with this configuration.
    pub fn reproduce_line(&self) -> String {
        let mut line = format!("TESTS_SEED={}", format_seed(self.master_seed));
        if self.nightly {
            line.push_str(" TESTS_NIGHTLY=true");
        }
        if self.multiplier != default_multiplier(self.nightly) {
            line.push_str(&format!(" TESTS_MULTIPLIER={}", self.multiplier));
        }
        line.push(' ');
        line.push_str(&self.command);
        line
    }
}

/// The `cargo test` invocation that runs one test alone: `-p <package>` and
/// `--lib` or `--test <target>` from the package cargo ran (`CARGO_PKG_NAME`)
/// and the test binary (`<target>-<hash>`), and libtest's full test path from
/// the thread it names after the test. Without them (not run by cargo, or
/// off the test's thread), the bare `cargo test -- --exact <name>`.
fn cargo_test_command(
    package: Option<&str>,
    exe: Option<&std::path::Path>,
    thread: Option<&str>,
    name: &str,
) -> String {
    let path = thread
        .filter(|t| *t != "main" && t.rsplit("::").next() == Some(name))
        .unwrap_or(name);
    let target = exe
        .and_then(|e| e.file_stem())
        .and_then(|s| s.to_str())
        .map(|s| s.rsplit_once('-').map_or(s, |(stem, _)| stem));
    match (package, target) {
        (Some(p), Some(t)) if t == p.replace('-', "_") => {
            format!("cargo test -p {p} --lib -- --exact {path}")
        }
        (Some(p), Some(t)) => format!("cargo test -p {p} --test {t} -- --exact {path}"),
        _ => format!("cargo test -- --exact {path}"),
    }
}

impl Drop for LuceneTestCase {
    fn drop(&mut self) {
        if std::thread::panicking() {
            eprintln!("NOTE: reproduce with: {}", self.reproduce_line());
        }
    }
}

/// The environment's `TEST_NIGHTLY` and `RANDOM_MULTIPLIER`, read once: the
/// static knobs Java's `rarely(random)`/`atLeast(random, i)` consult.
pub fn knobs() -> (bool, i32) {
    static KNOBS: std::sync::OnceLock<(bool, i32)> = std::sync::OnceLock::new();
    *KNOBS.get_or_init(|| {
        let t = LuceneTestCase::new("knobs");
        (t.nightly(), t.multiplier())
    })
}

/// `LuceneTestCase.defaultRandomMultiplier()`: 2 nightly, else 1.
pub fn default_multiplier(nightly: bool) -> i32 {
    if nightly {
        2
    } else {
        1
    }
}

/// `Boolean.parseBoolean`: "true" in any case; everything else is false.
fn parse_bool(v: &str) -> bool {
    v.trim().eq_ignore_ascii_case("true")
}

fn clock_seed() -> i64 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    (nanos as u64 ^ (nanos >> 64) as u64) as i64
}

/// `LuceneTestCase.atLeast(random, i)`: `i * multiplier` up to half again
/// as many.
pub fn at_least(r: &mut JavaRandom, multiplier: i32, i: i32) -> i32 {
    let min = i.wrapping_mul(multiplier);
    let max = min.wrapping_add(min / 2);
    test_util::next_int(r, min, max)
}

/// `LuceneTestCase.rarely(random)`: true about 1% of the time (5% nightly),
/// more with a larger multiplier, never more than 20%.
pub fn rarely(r: &mut JavaRandom, nightly: bool, multiplier: i32) -> bool {
    let mut p: i32 = if nightly { 5 } else { 1 };
    // `p += (p * Math.log(RANDOM_MULTIPLIER))`: the compound assignment
    // narrows the double sum back to an int.
    p = (f64::from(p) + f64::from(p) * strict_math::log(f64::from(multiplier))) as i32;
    let min = 100 - p.min(20);
    r.next_int_bounded(100) >= min
}

/// `LuceneTestCase.usually(random)`.
pub fn usually(r: &mut JavaRandom, nightly: bool, multiplier: i32) -> bool {
    !rarely(r, nightly, multiplier)
}

/// `LuceneTestCase.fieldToType`: the type [`new_field`] chose for each
/// field name, so every instance of a name is indexed alike. Java keeps one
/// per suite (cleared after it); here a test owns one ([`LuceneTestCase`]
/// carries its own).
#[derive(Debug, Default, Clone)]
pub struct FieldTypes(HashMap<String, FieldType>);

impl FieldTypes {
    /// The type chosen for `name`, if one was.
    pub fn get(&self, name: &str) -> Option<&FieldType> {
        self.0.get(name)
    }

    /// `fieldToType.clear()`.
    pub fn clear(&mut self) {
        self.0.clear();
    }
}

/// A [`new_field`] value: Java's `Object value`, a `String` or a `BytesRef`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldValue<'a> {
    /// A `String`.
    String(&'a str),
    /// A `BytesRef`.
    Bytes(&'a [u8]),
}

/// `newField(random, name, value, type)`: the type chosen for `name`
/// before, or `ft` randomly upgraded -- stored when it is not (one
/// `nextBoolean`), and for an indexed type without term vectors (and not
/// `DOCS_AND_CUSTOM_FREQS`), term vectors when a `nextBoolean` says so,
/// then positions, payloads (with positions) and offsets (string values
/// only) each by a `nextBoolean`, as Java draws them. The type is frozen
/// and remembered for `name`.
///
/// # Panics
/// When the type cannot carry the value (Java's `IllegalArgumentException`
/// from the `Field` constructor).
pub fn new_field(
    random: &mut JavaRandom,
    types: &mut FieldTypes,
    name: &str,
    value: FieldValue<'_>,
    ft: &FieldType,
) -> Field {
    if let Some(prev) = types.get(name) {
        // always use the same fieldType for the same field name
        return create_field(name, value, prev.clone());
    }
    let mut new_type = FieldType::copy_of(ft);
    let set = |r: lucene_index::document::Result<()>| r.expect("an unfrozen copy");
    if !new_type.stored() && random.next_boolean() {
        set(new_type.set_stored(true)); // randomly store it
    }
    if new_type.index_options() != IndexOptions::None
        && new_type.index_options() != IndexOptions::DocsAndCustomFreqs
        && !new_type.store_term_vectors()
        && random.next_boolean()
    {
        set(new_type.set_store_term_vectors(true));
        if !new_type.store_term_vector_positions() {
            set(new_type.set_store_term_vector_positions(random.next_boolean()));
            if new_type.store_term_vector_positions() && !new_type.store_term_vector_payloads() {
                set(new_type.set_store_term_vector_payloads(random.next_boolean()));
            }
        }
        // offsets are disallowed on binary fields
        if matches!(value, FieldValue::String(_)) && !new_type.store_term_vector_offsets() {
            set(new_type.set_store_term_vector_offsets(random.next_boolean()));
        }
    }
    new_type.freeze();
    types.0.insert(name.to_string(), new_type.clone());
    create_field(name, value, new_type)
}

fn create_field(name: &str, value: FieldValue<'_>, ft: FieldType) -> Field {
    match value {
        FieldValue::String(s) => Field::from_string(name, s, ft),
        FieldValue::Bytes(b) => Field::from_bytes(name, b, ft),
    }
    .unwrap_or_else(|e| panic!("newField({name}): {e}"))
}

/// `newStringField(random, name, value, stored)`: [`new_field`] over
/// `StringField`'s type.
pub fn new_string_field(
    random: &mut JavaRandom,
    types: &mut FieldTypes,
    name: &str,
    value: FieldValue<'_>,
    stored: Store,
) -> Field {
    let ft = match stored {
        Store::Yes => StringField::type_stored(),
        Store::No => StringField::type_not_stored(),
    };
    new_field(random, types, name, value, &ft)
}

/// `newTextField(random, name, value, stored)`: [`new_field`] over
/// `TextField`'s type.
pub fn new_text_field(
    random: &mut JavaRandom,
    types: &mut FieldTypes,
    name: &str,
    value: &str,
    stored: Store,
) -> Field {
    let ft = match stored {
        Store::Yes => TextField::type_stored(),
        Store::No => TextField::type_not_stored(),
    };
    new_field(random, types, name, FieldValue::String(value), &ft)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn string_hash_is_javas() {
        assert_eq!(java_string_hash(""), 0);
        assert_eq!(java_string_hash("hello"), 99_162_322);
        // A non-BMP character hashes as its two surrogates.
        assert_eq!(
            java_string_hash("\u{10400}"),
            0xD801i32.wrapping_mul(31).wrapping_add(0xDC00)
        );
    }

    #[test]
    fn seeds_parse_and_print_as_hex() {
        assert_eq!(parse_seed("DEADBEEF"), Some(0xDEAD_BEEF));
        assert_eq!(parse_seed("deadbeef:CAFE"), Some(0xDEAD_BEEF));
        assert_eq!(parse_seed("FFFFFFFFFFFFFFFF"), Some(-1));
        assert_eq!(parse_seed(""), None);
        assert_eq!(parse_seed("xyz"), None);
        assert_eq!(format_seed(-1), "FFFFFFFFFFFFFFFF");
        assert_eq!(format_seed(0xABC), "ABC");
    }

    #[test]
    fn config_derives_the_seed_from_the_name() {
        let t = LuceneTestCase::with_config("testFoo", 42, false, 1);
        assert_eq!(t.seed(), 42 ^ i64::from(java_string_hash("testFoo")));
        assert_eq!(t.name(), "testFoo");
        assert_eq!(t.master_seed(), 42);
        assert!(!t.nightly());
        assert_eq!(t.multiplier(), 1);
        // Run by cargo, on a thread libtest named after this test, whose
        // last segment is not `testFoo`: the bare name, scoped to this lib.
        assert_eq!(
            t.reproduce_line(),
            "TESTS_SEED=2A cargo test -p lucene-test-framework --lib -- --exact testFoo"
        );
        let n = LuceneTestCase::with_config("t", 1, true, 3);
        assert!(n
            .reproduce_line()
            .starts_with("TESTS_SEED=1 TESTS_NIGHTLY=true TESTS_MULTIPLIER=3 cargo test -p "));
        let d = LuceneTestCase::with_config("t", 1, true, 2);
        assert!(!d.reproduce_line().contains("MULTIPLIER"));
    }

    #[test]
    fn the_command_names_package_target_and_path() {
        use std::path::Path;
        let exe = Path::new("target/debug/deps/test_fixed_bit_set-0123abcd.exe");
        assert_eq!(
            cargo_test_command(Some("lucene-x"), Some(exe), Some("test_set"), "test_set"),
            "cargo test -p lucene-x --test test_fixed_bit_set -- --exact test_set"
        );
        let lib = Path::new("deps/lucene_x-99");
        assert_eq!(
            cargo_test_command(Some("lucene-x"), Some(lib), Some("a::tests::f"), "f"),
            "cargo test -p lucene-x --lib -- --exact a::tests::f"
        );
        // A thread named after another test, or `main`: the name given.
        assert_eq!(
            cargo_test_command(Some("lucene-x"), Some(lib), Some("main"), "f"),
            "cargo test -p lucene-x --lib -- --exact f"
        );
        assert_eq!(
            cargo_test_command(None, Some(lib), Some("a::f"), "f"),
            "cargo test -- --exact a::f"
        );
        assert_eq!(
            cargo_test_command(Some("p"), None, None, "f"),
            "cargo test -- --exact f"
        );
    }

    #[test]
    fn this_tests_own_line_reruns_it() {
        let t = LuceneTestCase::with_config("this_tests_own_line_reruns_it", 1, false, 1);
        assert_eq!(
            t.reproduce_line(),
            "TESTS_SEED=1 cargo test -p lucene-test-framework --lib -- --exact util::lucene_test_case::tests::this_tests_own_line_reruns_it"
        );
    }

    #[test]
    fn vars_set_the_knobs() {
        let t = LuceneTestCase::from_vars("v", Some("ff"), Some(" TRUE "), None);
        assert_eq!(t.master_seed(), 255);
        assert!(t.nightly());
        assert_eq!(t.multiplier(), 2);
        let t = LuceneTestCase::from_vars("v", None, Some("yes"), Some(" 4 "));
        assert!(!t.nightly());
        assert_eq!(t.multiplier(), 4);
        // The environment path: whatever the run was given, it parses.
        let _ = LuceneTestCase::new("envtest");
    }

    #[test]
    #[should_panic(expected = "TESTS_SEED is not hex")]
    fn a_bad_seed_is_refused() {
        LuceneTestCase::from_vars("v", Some("nope"), None, None);
    }

    #[test]
    #[should_panic(expected = "TESTS_MULTIPLIER is not an int")]
    fn a_bad_multiplier_is_refused() {
        LuceneTestCase::from_vars("v", Some("1"), None, Some("x"));
    }

    #[test]
    fn helpers_draw_from_the_test_stream() {
        let mut t = LuceneTestCase::with_config("x", 7, false, 1);
        let mut twin = JavaRandom::new(t.seed());
        assert_eq!(t.random().next_int(), twin.next_int());
        let a = t.at_least(10);
        assert!((10..=15).contains(&a));
        assert_eq!(a, at_least(&mut twin, 1, 10));
        assert_eq!(t.rarely(), rarely(&mut twin, false, 1));
        assert_eq!(t.usually(), usually(&mut twin, false, 1));
    }

    #[test]
    fn rarely_is_bounded_by_the_multiplier() {
        let mut r = JavaRandom::new(5);
        // Nightly with a huge multiplier saturates at 20%.
        let hits = (0..10_000).filter(|_| rarely(&mut r, true, 1000)).count();
        assert!((1500..2500).contains(&hits), "{hits}");
        let hits = (0..10_000).filter(|_| rarely(&mut r, false, 1)).count();
        assert!(hits < 300, "{hits}");
    }

    #[test]
    fn drop_while_panicking_prints_the_reproduce_line() {
        let r = std::panic::catch_unwind(|| {
            let _t = LuceneTestCase::with_config("boom", 1, false, 1);
            panic!("expected");
        });
        assert!(r.is_err());
    }
}
