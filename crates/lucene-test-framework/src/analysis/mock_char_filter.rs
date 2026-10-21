//! `MockCharFilter`: doubles every character whose value `% 10` is the
//! filter's remainder, recording an offset correction for each inserted
//! copy -- a char filter that changes lengths, for testing offset
//! correction.

use std::collections::BTreeMap;

use lucene_analysis::java_character::{is_high_surrogate, is_low_surrogate};
use lucene_analysis::reader::{CharFilter, CharReader};
use lucene_analysis::AnalysisError;

/// `org.apache.lucene.tests.analysis.MockCharFilter`.
pub struct MockCharFilter {
    input: Box<dyn CharReader>,
    remainder: i32,
    current_offset: i32,
    delta: i32,
    /// `-1`: none.
    buffered_ch: i32,
    corrections: BTreeMap<i32, i32>,
}

impl MockCharFilter {
    /// `MockCharFilter(in, remainder)`.
    ///
    /// # Errors
    /// `IllegalArgument` when `remainder` is not in `0..10`.
    pub fn new(input: Box<dyn CharReader>, remainder: i32) -> Result<Self, AnalysisError> {
        if !(0..10).contains(&remainder) {
            return Err(AnalysisError::IllegalArgument(format!(
                "invalid remainder parameter (must be 0..10): {remainder}"
            )));
        }
        Ok(MockCharFilter {
            input,
            remainder,
            current_offset: -1,
            delta: 0,
            buffered_ch: -1,
            corrections: BTreeMap::new(),
        })
    }

    /// `read()`: one unit, `-1` at the end.
    fn read_one(&mut self) -> Result<i32, AnalysisError> {
        // we have a buffered character, add an offset correction and return it
        if self.buffered_ch >= 0 {
            let ch = self.buffered_ch;
            self.buffered_ch = -1;
            self.current_offset += 1;
            self.corrections.insert(self.current_offset, self.delta - 1);
            self.delta -= 1;
            return Ok(ch);
        }
        // otherwise actually read one
        let mut c = [0u16; 1];
        if self.input.read(&mut c)? == 0 {
            return Ok(-1);
        }
        let ch = i32::from(c[0]);
        self.current_offset += 1;
        if (ch % 10) != self.remainder || is_high_surrogate(c[0]) || is_low_surrogate(c[0]) {
            return Ok(ch);
        }
        // we will double this character, so buffer it.
        self.buffered_ch = ch;
        Ok(ch)
    }
}

impl CharFilter for MockCharFilter {
    fn input(&self) -> &dyn CharReader {
        &*self.input
    }

    fn input_mut(&mut self) -> &mut dyn CharReader {
        &mut *self.input
    }

    fn read_filtered(&mut self, buf: &mut [u16]) -> Result<usize, AnalysisError> {
        let mut num_read = 0;
        for slot in buf.iter_mut() {
            let c = self.read_one()?;
            if c == -1 {
                break;
            }
            *slot = c as u16;
            num_read += 1;
        }
        Ok(num_read)
    }

    /// `correct`: the last correction at or before `current_off`.
    fn correct(&self, current_off: i32) -> i32 {
        let ret = match self.corrections.range(..=current_off).next_back() {
            None => current_off,
            Some((_, diff)) => current_off + diff,
        };
        assert!(
            ret >= 0,
            "currentOff={current_off},diff={}",
            ret - current_off
        );
        ret
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_analysis::reader::{read_to_string, StrReader};

    #[test]
    fn doubles_the_matching_characters_and_corrects_offsets() {
        // 'a' is 97: remainder 7 doubles it
        let mut f = MockCharFilter::new(Box::new(StrReader::new("aba")), 7).unwrap();
        assert_eq!(read_to_string(&mut f).unwrap(), "aabaa");
        assert_eq!(f.correct_offset(0), 0);
        assert_eq!(f.correct_offset(2), 1);
        assert_eq!(f.correct_offset(5), 3);
        assert_eq!(f.read(&mut []).unwrap(), 0);
        f.close().unwrap();
        // surrogates are never doubled (0xd83d % 10 == 7)
        let mut f = MockCharFilter::new(Box::new(StrReader::new("\u{1F600}")), 7).unwrap();
        assert_eq!(read_to_string(&mut f).unwrap(), "\u{1F600}");
    }

    #[test]
    fn rejects_a_bad_remainder() {
        for r in [-1, 10] {
            assert!(MockCharFilter::new(Box::new(StrReader::new("")), r)
                .err()
                .unwrap()
                .to_string()
                .contains("invalid remainder parameter (must be 0..10)"));
        }
    }

    #[test]
    #[should_panic(expected = "currentOff=0,diff=-1")]
    fn a_negative_correction_is_an_assertion() {
        let mut f = MockCharFilter::new(Box::new(StrReader::new("")), 0).unwrap();
        f.corrections.insert(0, -1);
        f.correct(0);
    }
}
