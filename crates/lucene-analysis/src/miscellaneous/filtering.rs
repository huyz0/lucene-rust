//! The `miscellaneous` package's `FilteringTokenFilter`s: `LengthFilter`,
//! `CodepointCountFilter`, `KeepWordFilter`, `DropIfFlaggedFilter`.

use std::sync::Arc;

use crate::attributes::AttributeSource;
use crate::token_stream::{Accept, FilteringTokenFilter, TokenFilter, TokenStream};
use crate::{AnalysisError, CharArraySet};

/// A Java `FilteringTokenFilter` subclass: a struct over
/// [`FilteringTokenFilter`] with its `accept()` as `$accept`.
macro_rules! filtering_filter {
    ($(#[$doc:meta])* $name:ident, $accept:ty) => {
        $(#[$doc])*
        pub struct $name<I> {
            inner: FilteringTokenFilter<I, $accept>,
        }

        impl<I: TokenStream> TokenFilter for $name<I> {
            type Input = FilteringTokenFilter<I, $accept>;
            fn input(&self) -> &Self::Input {
                &self.inner
            }
            fn input_mut(&mut self) -> &mut Self::Input {
                &mut self.inner
            }
            fn increment(&mut self) -> Result<bool, AnalysisError> {
                self.inner.increment_token()
            }
        }
    };
}

/// The `min`/`max` checks `LengthFilter` and `CodepointCountFilter` share.
fn check_range(min: i32, max: i32) -> Result<(), AnalysisError> {
    if min < 0 {
        return Err(AnalysisError::IllegalArgument(
            "minimum length must be greater than or equal to zero".into(),
        ));
    }
    if min > max {
        return Err(AnalysisError::IllegalArgument(
            "maximum length must not be greater than minimum length".into(),
        ));
    }
    Ok(())
}

/// `LengthFilter.accept()`: the term's UTF-16 length in `[min, max]`.
pub struct LengthAccept {
    min: i32,
    max: i32,
}

impl Accept for LengthAccept {
    fn accept(&mut self, a: &AttributeSource) -> Result<bool, AnalysisError> {
        let len = i32::try_from(a.term_utf16_len()).unwrap_or(i32::MAX);
        Ok(len >= self.min && len <= self.max)
    }
}

filtering_filter!(
    /// `org.apache.lucene.analysis.miscellaneous.LengthFilter`.
    LengthFilter,
    LengthAccept
);

impl<I: TokenStream> LengthFilter<I> {
    /// `new LengthFilter(TokenStream, int min, int max)`.
    pub fn new(input: I, min: i32, max: i32) -> Result<Self, AnalysisError> {
        check_range(min, max)?;
        Ok(LengthFilter {
            inner: FilteringTokenFilter::new(input, LengthAccept { min, max }),
        })
    }
}

/// `CodepointCountFilter.accept()`, with Java's UTF-16 bounds shortcut.
pub struct CodepointCountAccept {
    min: i32,
    max: i32,
}

impl Accept for CodepointCountAccept {
    fn accept(&mut self, a: &AttributeSource) -> Result<bool, AnalysisError> {
        let max32 = i32::try_from(a.term_utf16_len()).unwrap_or(i32::MAX);
        let min32 = max32 >> 1;
        if min32 >= self.min && max32 <= self.max {
            Ok(true)
        } else if min32 > self.max || max32 < self.min {
            Ok(false)
        } else {
            let len = i32::try_from(a.term().chars().count()).unwrap_or(i32::MAX);
            Ok(len >= self.min && len <= self.max)
        }
    }
}

filtering_filter!(
    /// `org.apache.lucene.analysis.miscellaneous.CodepointCountFilter`.
    CodepointCountFilter,
    CodepointCountAccept
);

impl<I: TokenStream> CodepointCountFilter<I> {
    /// `new CodepointCountFilter(TokenStream, int min, int max)`.
    pub fn new(input: I, min: i32, max: i32) -> Result<Self, AnalysisError> {
        check_range(min, max)?;
        Ok(CodepointCountFilter {
            inner: FilteringTokenFilter::new(input, CodepointCountAccept { min, max }),
        })
    }
}

/// `KeepWordFilter.accept()`: the term is in the set.
pub struct KeepWordAccept(Arc<CharArraySet>);

impl Accept for KeepWordAccept {
    fn accept(&mut self, a: &AttributeSource) -> Result<bool, AnalysisError> {
        Ok(self.0.contains(a.term()))
    }
}

filtering_filter!(
    /// `org.apache.lucene.analysis.miscellaneous.KeepWordFilter`.
    KeepWordFilter,
    KeepWordAccept
);

impl<I: TokenStream> KeepWordFilter<I> {
    /// `new KeepWordFilter(TokenStream, CharArraySet)`.
    pub fn new(input: I, words: Arc<CharArraySet>) -> Self {
        KeepWordFilter {
            inner: FilteringTokenFilter::new(input, KeepWordAccept(words)),
        }
    }
}

/// `DropIfFlaggedFilter.accept()`: not every bit of `dropFlags` set.
pub struct DropIfFlaggedAccept(i32);

impl Accept for DropIfFlaggedAccept {
    fn accept(&mut self, a: &AttributeSource) -> Result<bool, AnalysisError> {
        Ok((a.flags() & self.0) != self.0)
    }
}

filtering_filter!(
    /// `org.apache.lucene.analysis.miscellaneous.DropIfFlaggedFilter`.
    DropIfFlaggedFilter,
    DropIfFlaggedAccept
);

impl<I: TokenStream> DropIfFlaggedFilter<I> {
    /// `new DropIfFlaggedFilter(TokenStream, int dropFlags)`.
    pub fn new(input: I, drop_flags: i32) -> Self {
        DropIfFlaggedFilter {
            inner: FilteringTokenFilter::new(input, DropIfFlaggedAccept(drop_flags)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::canned::{render, Canned};

    #[test]
    fn length_and_codepoint_bounds() {
        let mut f = LengthFilter::new(
            Canned::parse("a:0:1:1:1 abc:2:5:1:1 abcdef:6:12:1:1|12|0"),
            2,
            4,
        )
        .unwrap();
        assert_eq!(render(&mut f), "abc:2:5:2:1|12|1");
        assert!(LengthFilter::new(Canned::parse(""), -1, 2).is_err());
        assert!(LengthFilter::new(Canned::parse(""), 3, 2).is_err());
        // Two emoji: UTF-16 length 4, two code points; one emoji and 'a': 3.
        let mut f = CodepointCountFilter::new(
            Canned::parse("😀😀:0:4:1:1 x:4:5:1:1 a😀:5:8:1:1 abcdefgh:8:16:1:1|16|0"),
            2,
            2,
        )
        .unwrap();
        assert_eq!(render(&mut f), "😀😀:0:4:1:1 a😀:5:8:2:1|16|1");
        assert!(CodepointCountFilter::new(Canned::parse(""), 2, 1).is_err());
    }

    #[test]
    fn keep_words_and_flags() {
        let words = Arc::new(CharArraySet::from_words(["b"], true));
        let mut f = KeepWordFilter::new(Canned::parse("a:0:1:1:1 B:2:3:1:1|3|0"), words);
        assert_eq!(render(&mut f), "B:2:3:2:1|3|0");
        let mut c = Canned::parse("a:0:1:1:1 b:2:3:1:1|3|0");
        c.set_flags(&[3, 1]);
        let mut f = DropIfFlaggedFilter::new(c, 3);
        assert_eq!(render(&mut f), "b:2:3:2:1|3|0");
    }
}
