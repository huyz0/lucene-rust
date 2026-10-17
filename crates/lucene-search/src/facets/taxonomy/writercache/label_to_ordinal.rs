//! `LabelToOrdinal`: the abstract label -> ordinal map with an ordinal
//! counter. (No implementation of it ships in Lucene 10.5.0's main code;
//! its own test supplies one.)

use super::super::FacetLabel;

/// `LabelToOrdinal`. Java's abstract class holds the `counter`; here an
/// implementation exposes it through [`LabelToOrdinal::counter_mut`] and
/// gets `getMaxOrdinal`/`getNextOrdinal` as provided methods.
pub trait LabelToOrdinal {
    /// `INVALID_ORDINAL`: what [`LabelToOrdinal::get_ordinal`] returns for a
    /// label it does not hold.
    const INVALID_ORDINAL: i32 = -2;

    /// The `counter` field.
    fn counter_mut(&mut self) -> &mut i32;

    /// `getMaxOrdinal()`.
    fn max_ordinal(&mut self) -> i32 {
        *self.counter_mut()
    }

    /// `getNextOrdinal()`: the counter, then increments it.
    fn next_ordinal(&mut self) -> i32 {
        let counter = self.counter_mut();
        let ord = *counter;
        *counter = counter.saturating_add(1);
        ord
    }

    /// `addLabel(label, ordinal)`.
    fn add_label(&mut self, label: &FacetLabel, ordinal: i32);

    /// `getOrdinal(label)`: the ordinal, or [`LabelToOrdinal::INVALID_ORDINAL`].
    fn get_ordinal(&self, label: &FacetLabel) -> i32;
}

#[cfg(test)]
mod tests;
