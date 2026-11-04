use super::*;

/// `TestCharBlockArray.testArray`: appends of every kind across block
/// boundaries read back as the same characters.
#[test]
fn appends_span_blocks() {
    let mut array = CharBlockArray::with_block_size(4);
    assert!(array.is_empty());
    let mut expected = String::new();
    array.append_str("abc").unwrap();
    expected.push_str("abc");
    array.append_char(u16::from(b'd')).unwrap();
    expected.push('d');
    array
        .append_units(&"efghij".encode_utf16().collect::<Vec<_>>())
        .unwrap();
    expected.push_str("efghij");
    array.append_str("\u{10000}é").unwrap();
    expected.push_str("\u{10000}é");
    array.append_str("").unwrap();
    assert_eq!(array.len(), expected.encode_utf16().count());
    assert!(!array.is_empty());
    assert_eq!(array.to_string(), expected);
    let units: Vec<u16> = expected.encode_utf16().collect();
    for (i, u) in units.iter().enumerate() {
        assert_eq!(array.char_at(i), Some(*u), "char {i}");
    }
    assert_eq!(array.char_at(units.len()), None);
    assert_eq!(array.sub_sequence(2, 9).unwrap(), "cdefghi");
    assert_eq!(array.sub_sequence(4, 4).unwrap(), "");
    assert_eq!(array.sub_sequence_units(0, units.len()).unwrap(), units);
    assert_eq!(array.sub_sequence(3, 2), None);
    assert_eq!(array.sub_sequence(0, units.len() + 1), None);
    assert_eq!(array.block_index(9), 2);
    assert_eq!(array.index_in_block(9), 1);
}

#[test]
fn defaults() {
    let array = CharBlockArray::default();
    assert_eq!(array.block_index(32 * 1024), 1);
    assert_eq!(array.to_string(), "");
    let mut zero = CharBlockArray::with_block_size(0);
    zero.append_str("xy").unwrap();
    assert_eq!(zero.to_string(), "xy");
    assert_eq!(zero.block_index(1), 1);
}

/// `addBlock` past 2 GB of characters: a block size of 2^30 holds one
/// block, and a second would pass the limit.
#[test]
fn more_than_two_gb_refused() {
    let mut array = CharBlockArray::with_block_size(1 << 30);
    // Pretend the first block is full without allocating it.
    array.length = 1 << 30;
    array.blocks[0].chars = Vec::new();
    array.block_size = 0;
    assert!(array.current_full());
    array.block_size = 1 << 30;
    let err = array.add_block().unwrap_err();
    assert!(
        matches!(err, Error::IllegalState(ref m) if m == "cannot store more than 2 GB in CharBlockArray")
    );
}
