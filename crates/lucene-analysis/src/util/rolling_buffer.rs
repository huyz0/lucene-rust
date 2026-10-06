//! `org.apache.lucene.util.RollingBuffer`: a window of positions, created on
//! demand by [`RollingBuffer::get`] and released from the front by
//! [`RollingBuffer::free_before`].
//!
//! Java recycles its slot objects through a circular array, calling
//! `reset()` on each one freed; the port keeps the live window in a
//! `VecDeque` and recycles freed slots through a free list, resetting them as
//! Java does. `getMaxPos` and `getBufferSize` report what Java's do.

use std::collections::VecDeque;

/// `RollingBuffer.Resettable`.
pub trait Resettable: Default {
    /// Return to the freshly created state.
    fn reset(&mut self);
}

/// `RollingBuffer<T>`.
pub struct RollingBuffer<T: Resettable> {
    window: VecDeque<T>,
    /// The position of `window[0]`.
    first_pos: i32,
    free: Vec<T>,
}

impl<T: Resettable> Default for RollingBuffer<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Resettable> RollingBuffer<T> {
    /// `new RollingBuffer()`.
    pub fn new() -> Self {
        RollingBuffer {
            window: VecDeque::new(),
            first_pos: 0,
            free: Vec::new(),
        }
    }

    /// `reset()`: every live slot reset and released; positions restart at 0.
    pub fn reset(&mut self) {
        while let Some(mut t) = self.window.pop_front() {
            t.reset();
            self.free.push(t);
        }
        self.first_pos = 0;
    }

    fn next_pos(&self) -> i32 {
        self.first_pos + self.window.len() as i32
    }

    /// `get(int pos)`: creates every position up to `pos`; `pos` must not have
    /// been freed.
    pub fn get(&mut self, pos: i32) -> &mut T {
        while pos >= self.next_pos() {
            let t = self.free.pop().unwrap_or_default();
            self.window.push_back(t);
        }
        assert!(
            pos >= self.first_pos,
            "pos={pos} nextPos={} count={}",
            self.next_pos(),
            self.window.len()
        );
        &mut self.window[(pos - self.first_pos) as usize]
    }

    /// `getMaxPos()`: the highest position created so far.
    pub fn max_pos(&self) -> i32 {
        self.next_pos() - 1
    }

    /// `getBufferSize()`: how many positions are live.
    pub fn buffer_size(&self) -> usize {
        self.window.len()
    }

    /// `freeBefore(int pos)`: releases every position below `pos`.
    pub fn free_before(&mut self, pos: i32) {
        while self.first_pos < pos {
            let Some(mut t) = self.window.pop_front() else {
                // Java asserts toFree <= count; positions past the window
                // were never created, so there is nothing more to free.
                self.first_pos = pos;
                return;
            };
            t.reset();
            self.free.push(t);
            self.first_pos += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default, Debug)]
    struct Slot(i32);

    impl Resettable for Slot {
        fn reset(&mut self) {
            self.0 = 0;
        }
    }

    #[test]
    fn window_grows_frees_and_recycles() {
        let mut b: RollingBuffer<Slot> = RollingBuffer::new();
        assert_eq!(b.max_pos(), -1);
        b.get(3).0 = 7;
        assert_eq!(b.max_pos(), 3);
        assert_eq!(b.buffer_size(), 4);
        b.free_before(2);
        assert_eq!(b.buffer_size(), 2);
        assert_eq!(b.get(3).0, 7);
        b.get(5).0 = 1;
        assert_eq!(b.get(4).0, 0, "a recycled slot comes back reset");
        b.free_before(10);
        assert_eq!(b.buffer_size(), 0);
        b.reset();
        assert_eq!(b.max_pos(), -1);
        b.get(0).0 = 2;
        b.reset();
        assert_eq!(b.get(0).0, 0);
    }

    #[test]
    #[should_panic(expected = "pos=0")]
    fn reading_a_freed_position_panics() {
        let mut b: RollingBuffer<Slot> = RollingBuffer::new();
        b.get(2);
        b.free_before(1);
        b.get(0);
    }
}
