//! randomizedtesting 2.8.1's `generators.RandomNumbers` and
//! `generators.RandomPicks`: the bounded draws Lucene's `TestUtil` builds on.
//!
//! Not a Lucene class -- `com.carrotsearch.randomizedtesting` is the runner
//! Lucene's tests use -- but `TestUtil.nextInt` is `randomIntBetween`, so a
//! port that must draw what Java draws needs these, operation for operation.

use lucene_util::java_random::JavaRandom;

/// `RandomNumbers.randomIntBetween(r, min, max)`: uniform in `min..=max`.
///
/// # Panics
///
/// When `max < min`, as Java's assertion does.
pub fn random_int_between(r: &mut JavaRandom, min: i32, max: i32) -> i32 {
    assert!(max >= min, "max must be >= min: {min}, {max}");
    let range = i64::from(max) - i64::from(min);
    if range < i64::from(i32::MAX) {
        min.wrapping_add(r.next_int_bounded(1 + range as i32))
    } else {
        // `toIntExact(min + nextLong(r, 1 + range))`: in range by construction.
        (i64::from(min) + next_long(r, 1 + range)) as i32
    }
}

/// `RandomNumbers.randomLongBetween(r, min, max)`: uniform in `min..=max`,
/// the full `i64` range included.
///
/// # Panics
///
/// When `max < min`.
pub fn random_long_between(r: &mut JavaRandom, min: i64, max: i64) -> i64 {
    assert!(max >= min, "max must be >= min: {min}, {max}");
    let mut range = max.wrapping_sub(min);
    if range < 0 {
        range = range.wrapping_sub(i64::MAX);
        if range == i64::MIN {
            // Full spectrum.
            r.next_long()
        } else {
            let first = r.next_long() & i64::MAX;
            let second = if range == i64::MAX {
                r.next_long() & i64::MAX
            } else {
                next_long(r, range + 1)
            };
            min.wrapping_add(first).wrapping_add(second)
        }
    } else {
        let second = if range == i64::MAX {
            r.next_long() & i64::MAX
        } else {
            next_long(r, range + 1)
        };
        min.wrapping_add(second)
    }
}

/// `RandomNumbers.nextLong(rnd, n)`: uniform in `0..n`, by rejection.
///
/// # Panics
///
/// When `n <= 0`, as Java throws `IllegalArgumentException`.
pub fn next_long(r: &mut JavaRandom, n: i64) -> i64 {
    assert!(n > 0, "n <= 0: {n}");
    let mut value = r.next_long();
    let range = n - 1;
    if n & range == 0 {
        value &= range;
    } else {
        // `for (long u = value >>> 1; u + range - (value = u % n) < 0L;)
        //    u = rnd.nextLong() >>> 1;`
        let mut u = ((value as u64) >> 1) as i64;
        loop {
            value = u % n;
            if u.wrapping_add(range).wrapping_sub(value) >= 0 {
                break;
            }
            u = ((r.next_long() as u64) >> 1) as i64;
        }
    }
    value
}

/// `RandomPicks.randomFrom(r, list)`: one element, by `nextInt(size)`.
///
/// # Panics
///
/// On an empty slice (Java: `IllegalArgumentException`).
pub fn random_from<'a, T>(r: &mut JavaRandom, items: &'a [T]) -> &'a T {
    assert!(
        !items.is_empty(),
        "Can't pick a random object from an empty list."
    );
    let len = i32::try_from(items.len()).expect("a pickable list fits an int");
    &items[r.next_int_bounded(len) as usize]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int_between_covers_its_bounds_and_the_full_range() {
        let mut r = JavaRandom::new(1);
        for _ in 0..1000 {
            let v = random_int_between(&mut r, -3, 3);
            assert!((-3..=3).contains(&v));
        }
        assert_eq!(random_int_between(&mut r, 7, 7), 7);
        // range == i32::MAX takes the `nextLong` branch.
        for _ in 0..100 {
            let v = random_int_between(&mut r, 0, i32::MAX);
            assert!(v >= 0);
            let _ = random_int_between(&mut r, i32::MIN, i32::MAX);
        }
    }

    #[test]
    fn long_between_takes_every_branch() {
        let mut r = JavaRandom::new(2);
        for _ in 0..200 {
            let v = random_long_between(&mut r, -5, 5);
            assert!((-5..=5).contains(&v));
            let _ = random_long_between(&mut r, i64::MIN, i64::MAX);
            let w = random_long_between(&mut r, i64::MIN, 1);
            assert!(w <= 1);
            let x = random_long_between(&mut r, 0, i64::MAX);
            assert!(x >= 0);
            let y = random_long_between(&mut r, -1, i64::MAX - 1);
            assert!(y >= -1);
        }
    }

    #[test]
    fn next_long_rejects_and_masks() {
        let mut r = JavaRandom::new(3);
        for n in [1i64, 2, 3, 1 << 40, (1 << 62) + 1, i64::MAX] {
            for _ in 0..50 {
                let v = next_long(&mut r, n);
                assert!((0..n).contains(&v));
            }
        }
    }

    #[test]
    #[should_panic(expected = "n <= 0")]
    fn next_long_refuses_zero() {
        next_long(&mut JavaRandom::new(1), 0);
    }

    #[test]
    #[should_panic(expected = "max must be >= min")]
    fn int_between_refuses_an_empty_range() {
        random_int_between(&mut JavaRandom::new(1), 2, 1);
    }

    #[test]
    #[should_panic(expected = "max must be >= min")]
    fn long_between_refuses_an_empty_range() {
        random_long_between(&mut JavaRandom::new(1), 2, 1);
    }

    #[test]
    fn random_from_picks_by_next_int() {
        let mut r = JavaRandom::new(4);
        let items = ["a", "b", "c"];
        let mut twin = JavaRandom::new(4);
        for _ in 0..20 {
            let want = items[twin.next_int_bounded(3) as usize];
            assert_eq!(*random_from(&mut r, &items), want);
        }
    }

    #[test]
    #[should_panic(expected = "empty list")]
    fn random_from_refuses_an_empty_slice() {
        let empty: [u8; 0] = [];
        random_from(&mut JavaRandom::new(1), &empty);
    }
}
