#![cfg(test)]
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
