//! `java.util.TreeSet` under a comparator the caller passes per call.

use std::cmp::Ordering;

/// A `TreeSet` under a comparator that the elements' state decides: kept
/// sorted, an element comparing equal to one already in is not added, and
/// `remove` takes out the element comparing equal (as Java's does) -- both
/// faithful to `TreeSet`, including where a comparator that is not a total
/// order makes it drop elements.
///
/// The items sit in a ring buffer, so taking the first (a merge's queue
/// drained best first) is O(1), and an insertion moves the shorter side.
#[derive(Debug, Default, Clone)]
pub struct TreeSet<T> {
    items: std::collections::VecDeque<T>,
}

impl<T: Copy + PartialEq> TreeSet<T> {
    /// An empty set.
    pub fn new() -> Self {
        Self {
            items: std::collections::VecDeque::new(),
        }
    }

    /// `items` added one by one in order -- each comparing equal to an
    /// earlier one dropped, as `add` drops it -- built by one stable sort
    /// rather than an insertion each: the same set for any `cmp` that is a
    /// total preorder.
    pub fn from_items(mut items: Vec<T>, cmp: &mut dyn FnMut(&T, &T) -> Ordering) -> Self {
        // Stable: of the items comparing equal the earliest comes first, and
        // is the one `add` would have kept.
        items.sort_by(|a, b| cmp(a, b));
        items.dedup_by(|later, kept| cmp(kept, later) == Ordering::Equal);
        Self {
            items: items.into(),
        }
    }

    fn search(
        &self,
        x: &T,
        cmp: &mut dyn FnMut(&T, &T) -> Ordering,
    ) -> std::result::Result<usize, usize> {
        self.items.binary_search_by(|e| cmp(e, x))
    }

    /// `add(x)`: whether it was added.
    pub fn add(&mut self, x: T, cmp: &mut dyn FnMut(&T, &T) -> Ordering) -> bool {
        match self.search(&x, cmp) {
            Ok(_) => false,
            Err(i) => {
                self.items.insert(i, x);
                true
            }
        }
    }

    /// `remove(x)`: the element comparing equal to `x`.
    pub fn remove(&mut self, x: &T, cmp: &mut dyn FnMut(&T, &T) -> Ordering) -> bool {
        match self.search(x, cmp) {
            Ok(i) => {
                self.items.remove(i);
                true
            }
            Err(_) => false,
        }
    }

    /// `first()`: the least element.
    pub fn first(&self) -> Option<T> {
        self.items.front().copied()
    }

    /// `last()`: the greatest element.
    pub fn last(&self) -> Option<T> {
        self.items.back().copied()
    }

    /// `pollFirst()`: removes and returns the least element.
    pub fn poll_first(&mut self) -> Option<T> {
        self.items.pop_front()
    }

    /// `pollLast()`: removes and returns the greatest element.
    pub fn poll_last(&mut self) -> Option<T> {
        self.items.pop_back()
    }

    /// `isEmpty()`.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// `size()`.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// The elements in order.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.items.iter()
    }
}
