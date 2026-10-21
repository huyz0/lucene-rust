#![forbid(unsafe_code)]
//! lucene-test-framework: a port of Lucene 10.5.0's `lucene/test-framework`
//! (`org.apache.lucene.tests.*`), M14.h.
//!
//! Randomized-test infrastructure (seeds, `rarely`/`atLeast`, random string
//! and number generators, line docs) and the reusable `Base*TestCase` suites
//! that check an implementation against a model. It is test code: every
//! crate that uses it does so through `[dev-dependencies]`, and nothing in a
//! production build depends on it (the architecture skill's rule).
//!
//! Randomness is [`lucene_util::java_random::JavaRandom`], so a generator
//! given the same seed as its Java twin (`fixtures/src/testframework/`)
//! draws the same stream and produces the same value -- which is what the
//! differential tests under `tests/` check. JUnit's per-method seeding is
//! not reproduced: [`util::lucene_test_case::LuceneTestCase`] derives each
//! test's seed from `TESTS_SEED` and the test's name, and prints the line
//! that reproduces a failure.
//!
//! Modules follow the Java packages: [`util`] is `org.apache.lucene.tests.util`,
//! [`analysis`] `org.apache.lucene.tests.analysis` (the mock analysis chain
//! and `BaseTokenStreamTestCase`), [`store`] `org.apache.lucene.tests.store`,
//! [`mockfile`] `org.apache.lucene.tests.mockfile`, [`search`]
//! `org.apache.lucene.tests.search` and [`geo`] `org.apache.lucene.tests.geo`.

pub mod analysis;
pub mod geo;
pub mod mockfile;
pub mod search;
pub mod store;
pub mod util;
