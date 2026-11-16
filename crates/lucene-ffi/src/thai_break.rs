//! The JVM's Thai word iterator for `ThaiTokenizer`.
//!
//! Lucene's tokenizer cuts Thai runs with `BreakIterator.getWordInstance(th)`
//! over the JDK's own dictionary, so the tokens only agree with a Java node's
//! when the same iterator draws the boundaries. The plugin registers an
//! upcall that returns one sentence's boundaries from that iterator;
//! [`ffi_thai_set_break_upcall`] installs it as the word breaker of the Thai
//! tokenizer's `jvmBreaker` form.

use std::sync::OnceLock;

use lucene_analysis::lang::th::{install_jvm_word_breaker, ThaiWordBreaker, DONE};

use crate::error::{guard, FfiStatus};

/// The upcall: the sentence's UTF-16 units `text[..len]`; writes the word
/// boundaries (ascending, starting with 0 and ending with `len`) to
/// `out[..cap]` and returns how many there are, which may exceed `cap` (the
/// caller retries with room), or a negative number on failure.
pub type ThaiBreakUpcall =
    unsafe extern "C" fn(text: *const u16, len: i32, out: *mut i32, cap: i32) -> i32;

static UPCALL: OnceLock<ThaiBreakUpcall> = OnceLock::new();

/// Boundaries of one sentence, walked like a `BreakIterator`.
struct JvmBreaker {
    bounds: Vec<i32>,
    at: usize,
}

impl JvmBreaker {
    fn boxed() -> Box<dyn ThaiWordBreaker> {
        Box::new(JvmBreaker {
            bounds: vec![0],
            at: 0,
        })
    }
}

impl ThaiWordBreaker for JvmBreaker {
    fn set_text(&mut self, text: &[u16]) {
        self.at = 0;
        self.bounds.clear();
        let Some(upcall) = UPCALL.get() else {
            self.bounds.push(0);
            return;
        };
        let len = i32::try_from(text.len()).unwrap_or(i32::MAX);
        let mut cap = 64usize.max(text.len() / 2);
        loop {
            self.bounds.resize(cap, 0);
            // SAFETY: `text` and `bounds` are valid for the lengths passed.
            let n = unsafe {
                upcall(
                    text.as_ptr(),
                    len,
                    self.bounds.as_mut_ptr(),
                    i32::try_from(cap).unwrap_or(i32::MAX),
                )
            };
            match usize::try_from(n) {
                Ok(n) if n <= cap => {
                    self.bounds.truncate(n);
                    break;
                }
                Ok(n) => cap = n,
                // The JVM could not break the text: the whole sentence is one word.
                Err(_) => {
                    self.bounds.clear();
                    self.bounds.extend([0, len]);
                    break;
                }
            }
        }
        if self.bounds.is_empty() {
            self.bounds.push(0);
        }
    }

    fn current(&self) -> i32 {
        self.bounds[self.at]
    }

    fn next(&mut self) -> i32 {
        match self.bounds.get(self.at.saturating_add(1)) {
            Some(&b) => {
                self.at = self.at.saturating_add(1);
                b
            }
            None => DONE,
        }
    }
}

/// Registers the JVM's Thai word-break upcall for the Thai tokenizer's
/// `jvmBreaker` form. Only the first registration counts.
///
/// # Safety
/// `upcall` must be a function with [`ThaiBreakUpcall`]'s contract that stays
/// callable for the life of the process, from any thread.
#[no_mangle]
pub unsafe extern "C" fn ffi_thai_set_break_upcall(upcall: Option<ThaiBreakUpcall>) -> i32 {
    guard(|| {
        let upcall = upcall.ok_or(FfiStatus::NullPointer)?;
        if UPCALL.set(upcall).is_ok() {
            install_jvm_word_breaker(JvmBreaker::boxed);
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stand-in for the JVM: breaks after every third unit.
    unsafe extern "C" fn every_third(text: *const u16, len: i32, out: *mut i32, cap: i32) -> i32 {
        let _ = text;
        let mut n = 0;
        let mut p = 0;
        loop {
            if n < cap {
                // SAFETY: `n < cap`.
                unsafe { *out.add(n as usize) = p };
            }
            n = n.saturating_add(1);
            if p == len {
                return n;
            }
            p = p.saturating_add(3).min(len);
        }
    }

    #[test]
    fn boundaries_come_back_in_order_even_past_the_first_buffer() {
        assert!(UPCALL.set(every_third).is_ok());
        let mut b = JvmBreaker {
            bounds: Vec::new(),
            at: 0,
        };
        let text = vec![0u16; 400];
        b.set_text(&text);
        assert_eq!(b.current(), 0);
        let mut seen = vec![0];
        loop {
            let n = b.next();
            if n == DONE {
                break;
            }
            seen.push(n);
        }
        assert_eq!(seen.len(), 135);
        assert_eq!(*seen.last().unwrap(), 400);
        assert_eq!(b.current(), 400);
        b.set_text(&[]);
        assert_eq!((b.current(), b.next()), (0, DONE));
    }
}
