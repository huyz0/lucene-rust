//! `org.apache.lucene.misc.util.fst`: `UpToTwoPositiveIntOutputs` and
//! `ListOfOutputs`, FST outputs that let one input map to more than one
//! value (by adding the same input to the compiler several times in a row).
//!
//! Java's `Outputs<Object>` holds a `Long` or a `TwoLongs` (or a `List`);
//! here each is a value enum, and [`FstOutputs`]'s static methods stand for
//! the instance ones. `doShare` -- instance state in Java -- is a const
//! generic, because the trait has no `self`.

use std::fmt;
use std::marker::PhantomData;

use lucene_codecs::fst_compiler::{FstOutputs, FstReadError, ReverseReader};

/// `DataOutput.writeVInt`.
fn write_vint(out: &mut Vec<u8>, v: i32) {
    write_vlong(out, i64::from(v as u32));
}

/// `DataOutput.writeVLong` of the non-negative `v`.
fn write_vlong(out: &mut Vec<u8>, v: i64) {
    let mut v = v as u64;
    while v & !0x7f != 0 {
        out.push((v & 0x7f) as u8 | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

// ---------------------------------------------------------------------------
// UpToTwoPositiveIntOutputs
// ---------------------------------------------------------------------------

/// An output of [`UpToTwoPositiveIntOutputs`]: Java's `Long` or `TwoLongs`
/// (`first` is the first value added for the input, `second` the second).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UpToTwoOutput {
    /// A single value; `One(0)` is `NO_OUTPUT`.
    One(i64),
    /// `TwoLongs(first, second)`.
    Two(i64, i64),
}

impl UpToTwoOutput {
    /// `get(v)`: a single value (0 being `NO_OUTPUT`).
    pub fn of(v: i64) -> Self {
        Self::One(v)
    }

    /// `get(first, second)`.
    pub fn two(first: i64, second: i64) -> Self {
        debug_assert!(first >= 0 && second >= 0);
        Self::Two(first, second)
    }
}

impl fmt::Display for UpToTwoOutput {
    /// `outputToString`: a `Long`, or the `TwoLongs` record's `toString`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::One(v) => write!(f, "{v}"),
            Self::Two(a, b) => write!(f, "TwoLongs[first={a}, second={b}]"),
        }
    }
}

/// `UpToTwoPositiveIntOutputs`: each output is one or two non-negative longs,
/// stored with one bit stolen from each. 0 is reserved for "no output". With
/// `SHARE`, `common` of two outputs is their minimum, else they must be equal
/// to share anything (`getSingleton(doShare)`).
#[derive(Debug, Clone, Copy, Default)]
pub struct UpToTwoPositiveIntOutputs<const SHARE: bool>;

impl<const SHARE: bool> UpToTwoPositiveIntOutputs<SHARE> {
    /// A single output, which `common`, `subtract` and a non-final `add`
    /// always see; Java's `(Long)` cast. `TwoLongs` only ever ends an input.
    fn single(o: &UpToTwoOutput) -> i64 {
        match o {
            UpToTwoOutput::One(v) => *v,
            UpToTwoOutput::Two(..) => {
                debug_assert!(false, "a TwoLongs is only ever a final output");
                0
            }
        }
    }
}

impl<const SHARE: bool> FstOutputs for UpToTwoPositiveIntOutputs<SHARE> {
    type Value = UpToTwoOutput;

    fn no_output() -> UpToTwoOutput {
        UpToTwoOutput::One(0)
    }

    fn common(a: &UpToTwoOutput, b: &UpToTwoOutput) -> UpToTwoOutput {
        let (a, b) = (Self::single(a), Self::single(b));
        UpToTwoOutput::One(if a == 0 || b == 0 {
            0
        } else if SHARE {
            a.min(b)
        } else if a == b {
            a
        } else {
            0
        })
    }

    fn subtract(output: &UpToTwoOutput, inc: &UpToTwoOutput) -> UpToTwoOutput {
        let (output, inc) = (Self::single(output), Self::single(inc));
        debug_assert!(output >= inc);
        UpToTwoOutput::One(if inc == 0 {
            output
        } else if output == inc {
            0
        } else {
            output.wrapping_sub(inc)
        })
    }

    fn add(prefix: &UpToTwoOutput, output: &UpToTwoOutput) -> UpToTwoOutput {
        let prefix = Self::single(prefix);
        match output {
            UpToTwoOutput::One(o) => UpToTwoOutput::One(if prefix == 0 {
                *o
            } else if *o == 0 {
                prefix
            } else {
                prefix.wrapping_add(*o)
            }),
            UpToTwoOutput::Two(first, second) => {
                UpToTwoOutput::Two(first.wrapping_add(prefix), second.wrapping_add(prefix))
            }
        }
    }

    fn write(value: &UpToTwoOutput, out: &mut Vec<u8>) {
        match value {
            UpToTwoOutput::One(v) => write_vlong(out, v.wrapping_shl(1)),
            UpToTwoOutput::Two(first, second) => {
                write_vlong(out, first.wrapping_shl(1) | 1);
                write_vlong(out, *second);
            }
        }
    }

    fn read(input: &mut ReverseReader<'_>) -> Result<UpToTwoOutput, FstReadError> {
        let code = input.read_vlong()?;
        // `code >>> 1`.
        let first = (code as u64).wrapping_shr(1) as i64;
        if code & 1 == 0 {
            Ok(UpToTwoOutput::One(first))
        } else {
            Ok(UpToTwoOutput::Two(first, input.read_vlong()?))
        }
    }

    /// `merge(first, second)`: the two single values as `TwoLongs`. `None`
    /// (Java's `ClassCastException`) when either is already a pair.
    fn merge(first: &UpToTwoOutput, second: &UpToTwoOutput) -> Option<UpToTwoOutput> {
        match (first, second) {
            (UpToTwoOutput::One(a), UpToTwoOutput::One(b)) => Some(UpToTwoOutput::Two(*a, *b)),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// ListOfOutputs
// ---------------------------------------------------------------------------

/// An output of [`ListOfOutputs`]: one value of the wrapped outputs, or
/// (only ever as an input's final output) the several values added for it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ListOutput<T> {
    /// A single value.
    One(T),
    /// Two or more, in the order they were added.
    Many(Vec<T>),
}

impl<T: Clone> ListOutput<T> {
    /// `asList(output)`.
    pub fn as_list(&self) -> Vec<T> {
        match self {
            Self::One(v) => vec![v.clone()],
            Self::Many(v) => v.clone(),
        }
    }
}

impl<T: fmt::Display> fmt::Display for ListOutput<T> {
    /// `outputToString`: the value, or `[a, b, ...]`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::One(v) => write!(f, "{v}"),
            Self::Many(list) => {
                f.write_str("[")?;
                for (i, v) in list.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{v}")?;
                }
                f.write_str("]")
            }
        }
    }
}

/// `ListOfOutputs<T>`: wraps `O`, so one input can map to several of its
/// values. It cannot wrap itself.
#[derive(Debug, Clone, Copy, Default)]
pub struct ListOfOutputs<O>(PhantomData<O>);

impl<O: FstOutputs> ListOfOutputs<O> {
    /// The single value `common` and `subtract` see; a list is only ever a
    /// final output (Java's unchecked `(T)` cast).
    fn single(o: &ListOutput<O::Value>) -> O::Value {
        match o {
            ListOutput::One(v) => v.clone(),
            ListOutput::Many(list) => {
                debug_assert!(false, "a list is only ever a final output");
                list.first().cloned().unwrap_or_else(O::no_output)
            }
        }
    }
}

impl<O: FstOutputs> FstOutputs for ListOfOutputs<O> {
    type Value = ListOutput<O::Value>;

    fn no_output() -> Self::Value {
        ListOutput::One(O::no_output())
    }

    fn common(a: &Self::Value, b: &Self::Value) -> Self::Value {
        ListOutput::One(O::common(&Self::single(a), &Self::single(b)))
    }

    fn subtract(output: &Self::Value, inc: &Self::Value) -> Self::Value {
        ListOutput::One(O::subtract(&Self::single(output), &Self::single(inc)))
    }

    fn add(prefix: &Self::Value, output: &Self::Value) -> Self::Value {
        let prefix = Self::single(prefix);
        match output {
            ListOutput::One(o) => ListOutput::One(O::add(&prefix, o)),
            ListOutput::Many(list) => {
                ListOutput::Many(list.iter().map(|o| O::add(&prefix, o)).collect())
            }
        }
    }

    /// An arc's output, which is never a list.
    fn write(value: &Self::Value, out: &mut Vec<u8>) {
        O::write(&Self::single(value), out);
    }

    /// The count, then each value: a single output is a list of one.
    fn write_final_output(value: &Self::Value, out: &mut Vec<u8>) {
        match value {
            ListOutput::One(v) => {
                write_vint(out, 1);
                O::write(v, out);
            }
            ListOutput::Many(list) => {
                write_vint(out, list.len() as i32);
                for v in list {
                    O::write(v, out);
                }
            }
        }
    }

    fn read(input: &mut ReverseReader<'_>) -> Result<Self::Value, FstReadError> {
        Ok(ListOutput::One(O::read(input)?))
    }

    fn read_final_output(input: &mut ReverseReader<'_>) -> Result<Self::Value, FstReadError> {
        let count = input.read_vint()?;
        if count == 1 {
            return Ok(ListOutput::One(O::read(input)?));
        }
        let count = usize::try_from(count)
            .map_err(|_| FstReadError(format!("negative output count {count}")))?;
        // The count comes off disk: grow as values are read, not up front.
        let mut list = Vec::new();
        for _ in 0..count {
            list.push(O::read(input)?);
        }
        Ok(ListOutput::Many(list))
    }

    /// The values of both, first then second, as one list.
    fn merge(first: &Self::Value, second: &Self::Value) -> Option<Self::Value> {
        let mut list = first.as_list();
        list.extend(second.as_list());
        Some(ListOutput::Many(list))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_codecs::fst::PositiveIntOutputs;

    type Two<const S: bool> = UpToTwoPositiveIntOutputs<S>;
    type List = ListOfOutputs<PositiveIntOutputs>;

    fn one(v: i64) -> UpToTwoOutput {
        UpToTwoOutput::of(v)
    }

    /// A reader over bytes `write` produced: the FST body is read backwards
    /// from its last byte, and the compiler stores each node reversed.
    fn reader(forward: &[u8]) -> (Vec<u8>, i64) {
        (
            forward.iter().rev().copied().collect(),
            forward.len() as i64 - 1,
        )
    }

    #[test]
    fn up_to_two_common_shares_the_minimum_or_only_equal_values() {
        assert_eq!(Two::<true>::common(&one(5), &one(9)), one(5));
        assert_eq!(Two::<false>::common(&one(5), &one(9)), one(0));
        assert_eq!(Two::<false>::common(&one(7), &one(7)), one(7));
        assert_eq!(Two::<true>::common(&one(0), &one(9)), one(0));
        assert_eq!(Two::<true>::common(&one(9), &one(0)), one(0));
    }

    #[test]
    fn up_to_two_subtract_and_add_are_inverse_and_shift_pairs() {
        assert_eq!(Two::<true>::subtract(&one(9), &one(0)), one(9));
        assert_eq!(Two::<true>::subtract(&one(9), &one(9)), one(0));
        assert_eq!(Two::<true>::subtract(&one(9), &one(4)), one(5));
        assert_eq!(Two::<true>::add(&one(0), &one(3)), one(3));
        assert_eq!(Two::<true>::add(&one(4), &one(0)), one(4));
        assert_eq!(Two::<true>::add(&one(4), &one(3)), one(7));
        assert_eq!(
            Two::<true>::add(&one(4), &UpToTwoOutput::two(1, 2)),
            UpToTwoOutput::two(5, 6)
        );
    }

    #[test]
    fn up_to_two_round_trips_through_its_wire_form() {
        for v in [
            one(0),
            one(1),
            one(300),
            UpToTwoOutput::two(0, 0),
            UpToTwoOutput::two(5, 70_000),
        ] {
            let mut bytes = Vec::new();
            Two::<true>::write(&v, &mut bytes);
            let (body, last) = reader(&bytes);
            let mut r = ReverseReader::new(&body);
            r.set_position(last);
            assert_eq!(Two::<true>::read(&mut r).unwrap(), v);
            assert_eq!(r.position(), -1);
        }
    }

    #[test]
    fn up_to_two_merges_two_singles_only() {
        assert_eq!(
            Two::<true>::merge(&one(3), &one(4)),
            Some(UpToTwoOutput::two(3, 4))
        );
        assert_eq!(Two::<true>::merge(&UpToTwoOutput::two(1, 2), &one(4)), None);
        assert_eq!(
            UpToTwoOutput::two(1, 2).to_string(),
            "TwoLongs[first=1, second=2]"
        );
        assert_eq!(one(8).to_string(), "8");
    }

    #[test]
    fn list_merge_concatenates_in_order() {
        let a = ListOutput::One(1i64);
        let b = ListOutput::One(2i64);
        let ab = List::merge(&a, &b).unwrap();
        assert_eq!(ab, ListOutput::Many(vec![1, 2]));
        let abc = List::merge(&ab, &ListOutput::One(3)).unwrap();
        assert_eq!(abc, ListOutput::Many(vec![1, 2, 3]));
        assert_eq!(abc.as_list(), vec![1, 2, 3]);
        assert_eq!(a.as_list(), vec![1]);
        assert_eq!(abc.to_string(), "[1, 2, 3]");
        assert_eq!(a.to_string(), "1");
    }

    #[test]
    fn list_add_pushes_the_prefix_onto_every_value() {
        let many = ListOutput::Many(vec![1i64, 2]);
        assert_eq!(
            List::add(&ListOutput::One(10), &many),
            ListOutput::Many(vec![11, 12])
        );
        assert_eq!(
            List::add(&ListOutput::One(10), &ListOutput::One(5)),
            ListOutput::One(15)
        );
        assert_eq!(List::no_output(), ListOutput::One(0));
        assert_eq!(
            List::common(&ListOutput::One(4), &ListOutput::One(6)),
            ListOutput::One(4)
        );
        assert_eq!(
            List::subtract(&ListOutput::One(6), &ListOutput::One(4)),
            ListOutput::One(2)
        );
    }

    #[test]
    fn list_final_outputs_carry_a_count() {
        for v in [
            ListOutput::One(7i64),
            ListOutput::Many(vec![1, 2, 3]),
            ListOutput::Many(Vec::new()),
        ] {
            let mut bytes = Vec::new();
            List::write_final_output(&v, &mut bytes);
            let (body, last) = reader(&bytes);
            let mut r = ReverseReader::new(&body);
            r.set_position(last);
            assert_eq!(List::read_final_output(&mut r).unwrap(), v);
        }
        // An arc output is the bare value.
        let mut bytes = Vec::new();
        List::write(&ListOutput::One(300), &mut bytes);
        let (body, last) = reader(&bytes);
        let mut r = ReverseReader::new(&body);
        r.set_position(last);
        assert_eq!(List::read(&mut r).unwrap(), ListOutput::One(300));
    }

    #[test]
    fn a_negative_count_is_corrupt_and_a_short_list_is_truncated() {
        let mut bytes = Vec::new();
        write_vint(&mut bytes, -1);
        let (body, last) = reader(&bytes);
        let mut r = ReverseReader::new(&body);
        r.set_position(last);
        let err = List::read_final_output(&mut r).unwrap_err();
        assert!(err.0.contains("negative output count"), "{err}");
        let mut bytes = Vec::new();
        write_vint(&mut bytes, 5);
        write_vlong(&mut bytes, 1);
        let (body, last) = reader(&bytes);
        let mut r = ReverseReader::new(&body);
        r.set_position(last);
        assert!(List::read_final_output(&mut r).is_err());
    }
}
