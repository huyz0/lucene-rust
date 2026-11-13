//! Allocation that reports failure as a status instead of aborting.

use crate::status::{set_last_error, Status};

/// `Vec::with_capacity(len)`, except that a length no allocator can serve is
/// [`Status::InvalidArgument`] with a message.
///
/// A length a caller supplied cannot be trusted: `Vec::with_capacity` aborts
/// the process on failure, which would take the whole host node down rather
/// than returning a status code. `try_reserve_exact` returns the failure
/// instead, keeping every failure a status code.
pub fn try_with_capacity<T>(len: usize) -> Result<Vec<T>, Status> {
    let mut out = Vec::new();
    out.try_reserve_exact(len).map_err(|_| {
        set_last_error(format!(
            "cannot allocate {len} x {} bytes for a caller-supplied length",
            std::mem::size_of::<T>()
        ));
        Status::InvalidArgument
    })?;
    Ok(out)
}

/// Copies `src` into a freshly allocated `Vec<T>` via [`try_with_capacity`],
/// so an allocation failure is a status code rather than a process abort.
pub fn try_to_vec<T: Copy>(src: &[T]) -> Result<Vec<T>, Status> {
    let mut out = try_with_capacity(src.len())?;
    out.extend_from_slice(src);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn try_with_capacity_allocates_a_reasonable_length() {
        let v = try_with_capacity::<u64>(8).unwrap();
        assert!(v.is_empty());
        assert!(v.capacity() >= 8);
    }

    #[test]
    fn try_with_capacity_rejects_an_absurd_length_instead_of_aborting() {
        // The exact shape a Java caller's negative `int` takes once widened to
        // `usize`: far larger than any allocator can serve. `Vec::with_capacity`
        // would abort the process here; this must be a status code.
        let got = try_with_capacity::<u64>(usize::MAX / 4);
        assert_eq!(got.err(), Some(Status::InvalidArgument));
    }

    #[test]
    fn try_to_vec_copies_the_slice() {
        assert_eq!(try_to_vec(&[1i32, 2, 3]).unwrap(), vec![1, 2, 3]);
        assert!(try_to_vec::<i32>(&[]).unwrap().is_empty());
    }
}
