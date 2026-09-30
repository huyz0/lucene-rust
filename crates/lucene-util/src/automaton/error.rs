//! `TooComplexToDeterminizeException` and the other exceptions the automaton
//! API throws, as one error type.

use std::fmt;

/// `TooComplexToDeterminizeException`: determinizing would need more than
/// `determinize_work_limit` effort (Lucene counts effort as the summed size
/// of every NFA state set it expands, against `10 * workLimit`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TooComplexToDeterminize {
    /// `getAutomaton().getNumStates()` of the input.
    pub num_states: i32,
    /// `getAutomaton().getNumTransitions()` of the input.
    pub num_transitions: i32,
    /// `getDeterminizeWorkLimit()`.
    pub determinize_work_limit: i32,
    /// `getRegExp().getOriginalString()` when a [`crate::automaton::RegExp`]
    /// rethrew it (Lucene's `(RegExp, cause)` constructor).
    pub regexp: Option<String>,
}

impl fmt::Display for TooComplexToDeterminize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.regexp {
            Some(re) => write!(
                f,
                "Determinizing {re} would require more than {} effort.",
                self.determinize_work_limit
            ),
            None => write!(
                f,
                "Determinizing automaton with {} states and {} transitions would require more than {} effort.",
                self.num_states, self.num_transitions, self.determinize_work_limit
            ),
        }
    }
}

impl std::error::Error for TooComplexToDeterminize {}

/// Every exception the automaton API can throw.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum AutomatonError {
    /// `TooComplexToDeterminizeException`.
    #[error("{0}")]
    TooComplex(TooComplexToDeterminize),
    /// `IllegalArgumentException` (also `NumberFormatException`, its
    /// subclass), with Java's message.
    #[error("{0}")]
    IllegalArgument(String),
    /// `IllegalStateException`, with Java's message.
    #[error("{0}")]
    IllegalState(String),
}

impl From<TooComplexToDeterminize> for AutomatonError {
    fn from(e: TooComplexToDeterminize) -> Self {
        AutomatonError::TooComplex(e)
    }
}

pub(crate) fn illegal_argument<T>(msg: impl Into<String>) -> Result<T, AutomatonError> {
    Err(AutomatonError::IllegalArgument(msg.into()))
}
