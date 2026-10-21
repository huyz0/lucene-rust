//! `MockReaderWrapper`: a reader that spoon-feeds its consumer (returns
//! fewer units than asked for) and can throw a fake exception after a given
//! number of units.

use std::sync::{Arc, Mutex};

use lucene_analysis::reader::CharReader;
use lucene_analysis::AnalysisError;
use lucene_util::java_random::JavaRandom;

use crate::util::test_util;

/// The fake exception's message.
const EVIL_MESSAGE: &str = "fake exception now!";

/// `org.apache.lucene.tests.analysis.MockReaderWrapper`.
///
/// Java shares the caller's `Random`; the wrapper holds it behind a mutex,
/// so the caller keeps drawing from the same stream.
pub struct MockReaderWrapper {
    input: Box<dyn CharReader>,
    random: Arc<Mutex<JavaRandom>>,
    exc_at_char: i32,
    read_so_far: i32,
    throw_exc_next: bool,
}

impl MockReaderWrapper {
    /// `MockReaderWrapper(random, in)`.
    pub fn new(random: Arc<Mutex<JavaRandom>>, input: Box<dyn CharReader>) -> Self {
        MockReaderWrapper {
            input,
            random,
            exc_at_char: -1,
            read_so_far: 0,
            throw_exc_next: false,
        }
    }

    /// `throwExcAfterChar(charUpto)`: the fake exception once `char_upto`
    /// units have been read. Call it before reading.
    pub fn throw_exc_after_char(&mut self, char_upto: i32) {
        self.exc_at_char = char_upto;
        assert_eq!(self.read_so_far, 0, "only call this on init");
    }

    /// `throwExcNext()`: the fake exception on the next read.
    pub fn throw_exc_next(&mut self) {
        self.throw_exc_next = true;
    }

    /// `isMyEvilException`: whether `e` is this wrapper's fake exception.
    pub fn is_my_evil_exception(e: &AnalysisError) -> bool {
        matches!(e, AnalysisError::Io(m) if m == EVIL_MESSAGE)
    }
}

impl CharReader for MockReaderWrapper {
    fn read(&mut self, buf: &mut [u16]) -> Result<usize, AnalysisError> {
        if self.throw_exc_next || (self.exc_at_char != -1 && self.read_so_far >= self.exc_at_char) {
            return Err(AnalysisError::Io(EVIL_MESSAGE.to_string()));
        }
        let len = buf.len() as i32;
        let real_len = if len <= 1 {
            len
        } else {
            // Spoon-feed: intentionally maybe return less than the consumer asked for
            let mut r = self.random.lock().unwrap_or_else(|p| p.into_inner());
            test_util::next_int(&mut r, 1, len)
        };
        let read = if self.exc_at_char != -1 {
            let left = self.exc_at_char - self.read_so_far;
            let n = self.input.read(&mut buf[..real_len.min(left) as usize])?;
            self.read_so_far += n as i32;
            n
        } else {
            self.input.read(&mut buf[..real_len as usize])?
        };
        Ok(read)
    }

    fn close(&mut self) -> Result<(), AnalysisError> {
        self.input.close()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_analysis::reader::{read_to_string, StrReader};

    fn shared(seed: i64) -> Arc<Mutex<JavaRandom>> {
        Arc::new(Mutex::new(JavaRandom::new(seed)))
    }

    #[test]
    fn spoon_feeds_but_delivers_everything() {
        let mut r = MockReaderWrapper::new(shared(3), Box::new(StrReader::new("hello world")));
        let mut buf = [0u16; 8];
        let n = r.read(&mut buf).unwrap();
        assert!((1..=8).contains(&n));
        let rest = read_to_string(&mut r).unwrap();
        assert_eq!(rest.len() + n, 11);
        r.close().unwrap();
    }

    #[test]
    fn throws_after_the_given_char_and_on_demand() {
        let mut r = MockReaderWrapper::new(shared(1), Box::new(StrReader::new("abcdef")));
        r.throw_exc_after_char(3);
        let mut got = 0;
        let e = loop {
            let mut one = [0u16; 1];
            match r.read(&mut one) {
                Ok(n) => got += n,
                Err(e) => break e,
            }
        };
        assert_eq!(got, 3);
        assert!(MockReaderWrapper::is_my_evil_exception(&e));
        assert!(!MockReaderWrapper::is_my_evil_exception(
            &AnalysisError::Io("other".into())
        ));
        let mut r = MockReaderWrapper::new(shared(1), Box::new(StrReader::new("abc")));
        r.throw_exc_next();
        assert!(r.read(&mut [0u16; 4]).is_err());
    }
}
