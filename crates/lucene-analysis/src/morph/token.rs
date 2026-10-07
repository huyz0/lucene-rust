//! `morph.Token`, `morph.TokenType` and `morph.MorphData`.

use std::sync::Arc;

/// `org.apache.lucene.analysis.morph.TokenType`: which dictionary a lattice
/// node came from. The declaration order is Java's ordinal order, which
/// `ViterbiNBest.fixupPendingList` sorts by.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum TokenType {
    /// `KNOWN`: the system dictionary.
    #[default]
    Known,
    /// `UNKNOWN`: the unknown-word dictionary (by character class).
    Unknown,
    /// `USER`: the user dictionary.
    User,
}

impl TokenType {
    /// `TokenType.name()`.
    pub fn name(self) -> &'static str {
        match self {
            TokenType::Known => "KNOWN",
            TokenType::Unknown => "UNKNOWN",
            TokenType::User => "USER",
        }
    }
}

/// `org.apache.lucene.analysis.morph.MorphData` (and `Dictionary`'s three
/// defaults, which forward to it): the connection ids and cost of a word.
pub trait MorphData: Send + Sync {
    /// `getLeftId(morphId)`.
    fn left_id(&self, morph_id: i32) -> i32;
    /// `getRightId(morphId)`.
    fn right_id(&self, morph_id: i32) -> i32;
    /// `getWordCost(morphId)`.
    fn word_cost(&self, morph_id: i32) -> i32;
}

/// `org.apache.lucene.analysis.morph.Token`: a span of a backtraced
/// fragment. The fragment (`surfaceForm`, the `char[]` Java copies out of
/// the rolling buffer once per backtrace) is shared by the tokens cut from
/// it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    /// `surfaceForm`.
    pub surface_form: Arc<[u16]>,
    /// `offset`: the token's start in [`Self::surface_form`].
    pub offset: i32,
    /// `length`, in UTF-16 units.
    pub length: i32,
    /// `startOffset`.
    pub start_offset: i32,
    /// `endOffset`.
    pub end_offset: i32,
    /// `posIncr`.
    pub pos_incr: i32,
    /// `posLen`.
    pub pos_len: i32,
    /// `type`.
    pub token_type: TokenType,
}

impl Token {
    /// `new Token(surfaceForm, offset, length, startOffset, endOffset,
    /// type)`: position increment and length 1.
    pub fn new(
        surface_form: Arc<[u16]>,
        offset: i32,
        length: i32,
        start_offset: i32,
        end_offset: i32,
        token_type: TokenType,
    ) -> Self {
        Token {
            surface_form,
            offset,
            length,
            start_offset,
            end_offset,
            pos_incr: 1,
            pos_len: 1,
            token_type,
        }
    }

    /// The token's units: `surfaceForm[offset..offset + length]` (empty if
    /// the span does not lie inside the fragment).
    pub fn surface(&self) -> &[u16] {
        let start = usize::try_from(self.offset).unwrap_or(0);
        let len = usize::try_from(self.length).unwrap_or(0);
        start
            .checked_add(len)
            .and_then(|end| self.surface_form.get(start..end))
            .unwrap_or(&[])
    }

    /// `getSurfaceFormString()`.
    pub fn surface_form_string(&self) -> String {
        String::from_utf16_lossy(self.surface())
    }
}

/// A language's token type, seen as the base [`Token`] it extends.
pub trait MorphToken: Send {
    /// The base token.
    fn base(&self) -> &Token;
    /// The base token, mutably.
    fn base_mut(&mut self) -> &mut Token;
}

impl MorphToken for Token {
    fn base(&self) -> &Token {
        self
    }
    fn base_mut(&mut self) -> &mut Token {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_spans_and_types() {
        let frag: Arc<[u16]> = "abcd".encode_utf16().collect();
        let mut t = Token::new(frag.clone(), 1, 2, 5, 7, TokenType::User);
        assert_eq!(t.surface_form_string(), "bc");
        assert_eq!((t.pos_incr, t.pos_len), (1, 1));
        t.base_mut().pos_len = 3;
        assert_eq!(t.base().pos_len, 3);
        let outside = Token::new(frag, 3, 9, 0, 0, TokenType::Known);
        assert!(outside.surface().is_empty());
        assert_eq!(
            [TokenType::Known, TokenType::Unknown, TokenType::User].map(TokenType::name),
            ["KNOWN", "UNKNOWN", "USER"]
        );
        assert!(TokenType::Known < TokenType::User);
        assert_eq!(TokenType::default(), TokenType::Known);
    }
}
