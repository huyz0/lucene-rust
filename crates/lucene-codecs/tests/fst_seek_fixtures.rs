//! Differential tests for `Fst::seek_exact`/`FstEnum::seek_ceil`/
//! `FstEnum::seek_floor` against real, Lucene-written FSTs -- reusing the
//! same fixtures the existing `fst_*_fixtures.rs` files already use for
//! plain `get`/`iter` coverage, so these seek variants are exercised against
//! genuine bytes for every node encoding this port decodes:
//!
//! - `fixtures/data/fst/` -- list-encoded nodes only, but with real branching
//!   and shared prefixes (`app`/`apple`/`application`/`banana`/`band`/
//!   `bandana`/`z`).
//! - `fixtures/data/fst_binary_search/` -- an `ARCS_FOR_BINARY_SEARCH` root.
//! - `fixtures/data/fst_direct_addressing/` -- an `ARCS_FOR_DIRECT_ADDRESSING`
//!   root with a deliberate gap ('g') inside the label range.
//! - `fixtures/data/fst_continuous/` -- an `ARCS_FOR_CONTINUOUS` root (no
//!   gaps at all within its label range).
//! - `fixtures/data/fst_seek_non_root_array_node/` -- root stays
//!   list-encoded; array-encoded nodes (one of each of the three fixed-
//!   length-arc encodings) sit one level below it, exercising the
//!   recurse-past-a-non-root-array-node backtracking paths.
//! - `fixtures/data/fst_seek_floor_backtrack_{binary_search,direct_addressing,
//!   continuous}/` -- the root itself is array-encoded (one fixture per
//!   encoding) *and* one of its labels has its own array-encoded (continuous)
//!   child. `seek_floor`'s `find_next_floor_arc_binary_search`/
//!   `_direct_addressing`/`_continuous` are only reachable from
//!   `backtrack_to_floor_arc` re-reading a *parent* node that is itself
//!   array-encoded -- these fixtures are the only ones with such a parent.

use lucene_codecs::fst::Fst;
use lucene_store::data_input::SliceInput;

fn load_fst(dir: &str) -> Fst<'static> {
    let path = format!(
        "{}/../../fixtures/data/{dir}/fst.bin",
        env!("CARGO_MANIFEST_DIR")
    );
    let buf = std::fs::read(&path)
        .unwrap_or_else(|e| panic!("run fixtures generator first ({dir}): {e}"));
    let mut input = SliceInput::new(&buf);
    Fst::read(&mut input).expect("decode real Lucene-written FST")
}

fn kv(k: &[u8], v: &[u8]) -> (Vec<u8>, Vec<u8>) {
    (k.to_vec(), v.to_vec())
}

// --- `fixtures/data/fst/`: list-encoded, branching, shared prefixes -------

#[test]
fn seek_exact_hits_present_keys_in_the_branching_list_fixture() {
    let fst = load_fst("fst");
    for (key, output) in [
        (&b"app"[..], &b"1"[..]),
        (b"apple", b"2"),
        (b"application", b"3"),
        (b"banana", b"4"),
        (b"band", b"5"),
        (b"bandana", b"6"),
        (b"z", b"26"),
    ] {
        assert_eq!(
            fst.seek_exact(key).unwrap(),
            Some(output.to_vec()),
            "key={key:?}"
        );
        let mut e = fst.iter().unwrap();
        assert_eq!(
            e.seek_exact(key).unwrap(),
            Some(kv(key, output)),
            "FstEnum::seek_exact key={key:?}"
        );
    }
}

#[test]
fn seek_exact_misses_absent_keys_in_the_branching_list_fixture() {
    let fst = load_fst("fst");
    for key in [
        &b""[..],
        b"a",
        b"appl",
        b"applies",
        b"ban",
        b"bandanas",
        b"cat",
        b"zz",
    ] {
        assert_eq!(fst.seek_exact(key).unwrap(), None, "key={key:?}");
        let mut e = fst.iter().unwrap();
        assert_eq!(
            e.seek_exact(key).unwrap(),
            None,
            "FstEnum::seek_exact key={key:?}"
        );
    }
}

#[test]
fn seek_ceil_lands_between_two_keys_in_the_branching_list_fixture() {
    let fst = load_fst("fst");
    let mut e = fst.iter().unwrap();
    // "appl" sits strictly between "app" and "apple".
    assert_eq!(e.seek_ceil(b"appl").unwrap(), Some(kv(b"apple", b"2")));
    // "ban" sits strictly between "band"'s prefix and "banana".
    assert_eq!(e.seek_ceil(b"ban").unwrap(), Some(kv(b"banana", b"4")));
    // "bane" sits strictly between "bandana" and "z".
    assert_eq!(e.seek_ceil(b"bane").unwrap(), Some(kv(b"z", b"26")));
}

#[test]
fn seek_ceil_before_first_key_finds_first_key() {
    let fst = load_fst("fst");
    let mut e = fst.iter().unwrap();
    assert_eq!(e.seek_ceil(b"").unwrap(), Some(kv(b"app", b"1")));
    assert_eq!(e.seek_ceil(b"AAA").unwrap(), Some(kv(b"app", b"1")));
}

#[test]
fn seek_ceil_past_the_last_key_finds_nothing() {
    let fst = load_fst("fst");
    let mut e = fst.iter().unwrap();
    assert_eq!(e.seek_ceil(b"zz").unwrap(), None);
    assert_eq!(e.seek_ceil(b"zzzzzzzzzzzzzz").unwrap(), None);
}

#[test]
fn seek_floor_lands_between_two_keys_in_the_branching_list_fixture() {
    let fst = load_fst("fst");
    let mut e = fst.iter().unwrap();
    assert_eq!(e.seek_floor(b"appl").unwrap(), Some(kv(b"app", b"1")));
    assert_eq!(
        e.seek_floor(b"bandanas").unwrap(),
        Some(kv(b"bandana", b"6"))
    );
    assert_eq!(e.seek_floor(b"zz").unwrap(), Some(kv(b"z", b"26")));
}

#[test]
fn seek_floor_before_the_first_key_finds_nothing() {
    let fst = load_fst("fst");
    let mut e = fst.iter().unwrap();
    assert_eq!(e.seek_floor(b"").unwrap(), None);
    assert_eq!(e.seek_floor(b"AAA").unwrap(), None);
}

// --- `fixtures/data/fst_binary_search/`: `ARCS_FOR_BINARY_SEARCH` root ----

#[test]
fn seek_ceil_and_floor_around_gaps_in_the_binary_search_fixture() {
    let fst = load_fst("fst_binary_search");
    // Present single-byte keys, ascending: 01 28 50 78 a0 c8 f0.
    let mut e = fst.iter().unwrap();

    assert_eq!(e.seek_exact(&[0x28]).unwrap(), Some(kv(&[0x28], b"out1")));
    assert_eq!(e.seek_exact(&[0x14]).unwrap(), None);

    // 0x14 sits strictly between 0x01 and 0x28.
    assert_eq!(e.seek_ceil(&[0x14]).unwrap(), Some(kv(&[0x28], b"out1")));
    assert_eq!(e.seek_floor(&[0x14]).unwrap(), Some(kv(&[0x01], b"out0")));

    // Before the first key.
    assert_eq!(e.seek_ceil(&[0x00]).unwrap(), Some(kv(&[0x01], b"out0")));
    assert_eq!(e.seek_floor(&[0x00]).unwrap(), None);

    // Past the last key.
    assert_eq!(e.seek_ceil(&[0xff]).unwrap(), None);
    assert_eq!(e.seek_floor(&[0xff]).unwrap(), Some(kv(&[0xf0], b"out6")));
}

// --- `fixtures/data/fst_direct_addressing/`: `ARCS_FOR_DIRECT_ADDRESSING`
// root with a deliberate gap ('g') inside the label range ------------------

#[test]
fn seek_ceil_and_floor_around_the_gap_in_the_direct_addressing_fixture() {
    let fst = load_fst("fst_direct_addressing");
    // Present: a..f, h (0x61-0x66, 0x68); absent 'g' (0x67) is a genuine gap
    // *inside* the label range, not merely outside it.
    let mut e = fst.iter().unwrap();

    assert_eq!(e.seek_exact(b"f").unwrap(), Some(kv(b"f", b"out5")));
    assert_eq!(e.seek_exact(b"g").unwrap(), None);

    // 'g' is absent but inside the range -- ceil should skip straight to 'h'.
    assert_eq!(e.seek_ceil(b"g").unwrap(), Some(kv(b"h", b"out6")));
    assert_eq!(e.seek_floor(b"g").unwrap(), Some(kv(b"f", b"out5")));

    // Before the first present label ('`' = 0x60, just below 'a').
    assert_eq!(e.seek_ceil(b"`").unwrap(), Some(kv(b"a", b"out0")));
    assert_eq!(e.seek_floor(b"`").unwrap(), None);

    // Past the last present label.
    assert_eq!(e.seek_ceil(&[0xff]).unwrap(), None);
    assert_eq!(e.seek_floor(&[0xff]).unwrap(), Some(kv(b"h", b"out6")));
}

// --- `fixtures/data/fst_continuous/`: `ARCS_FOR_CONTINUOUS` root (no gaps) -

#[test]
fn seek_ceil_and_floor_at_the_edges_of_the_continuous_fixture() {
    let fst = load_fst("fst_continuous");
    // Present: a..g (0x61-0x67), fully contiguous -- no presence bit-table
    // at all, so there's no "gap inside the range" case to test here, only
    // range-boundary behavior.
    let mut e = fst.iter().unwrap();

    assert_eq!(e.seek_exact(b"d").unwrap(), Some(kv(b"d", b"out3")));
    assert_eq!(e.seek_exact(b"h").unwrap(), None);

    // Just past the last present label.
    assert_eq!(e.seek_ceil(b"h").unwrap(), None);
    assert_eq!(e.seek_floor(b"h").unwrap(), Some(kv(b"g", b"out6")));

    // Just before the first present label.
    assert_eq!(e.seek_ceil(&[0x60]).unwrap(), Some(kv(b"a", b"out0")));
    assert_eq!(e.seek_floor(&[0x60]).unwrap(), None);

    // A mid-range label lands exactly (no gaps to skip over).
    assert_eq!(e.seek_ceil(b"d").unwrap(), Some(kv(b"d", b"out3")));
    assert_eq!(e.seek_floor(b"d").unwrap(), Some(kv(b"d", b"out3")));
}

// --- `fixtures/data/fst_seek_non_root_array_node/`: array-encoded nodes one
// level *below* the root (root itself stays list-encoded with 3 arcs: 'B',
// 'C', 'D') -- every prior fixture above only ever put its array node at the
// root, so seeking into/across these never exercised the "recurse past a
// non-root array-encoded node" branches (e.g. `read_last_target_arc`'s array
// branch, `find_next_floor_arc_*`). Keys are `<prefix><label>`: 'B' groups
// widely-spaced labels forced into ARCS_FOR_BINARY_SEARCH, 'D' groups a..f,h
// (gap at 'g') forced into ARCS_FOR_DIRECT_ADDRESSING, 'C' groups fully
// contiguous a..g forced into ARCS_FOR_CONTINUOUS. -----------------------

#[test]
fn seek_within_the_non_root_binary_search_node() {
    let fst = load_fst("fst_seek_non_root_array_node");
    let mut e = fst.iter().unwrap();

    assert_eq!(
        e.seek_exact(&[b'B', 0x28]).unwrap(),
        Some(kv(&[b'B', 0x28], b"bs1"))
    );
    assert_eq!(e.seek_exact(&[b'B', 0x14]).unwrap(), None);

    // 0x14 sits strictly between 'B'+0x01 and 'B'+0x28.
    assert_eq!(
        e.seek_ceil(&[b'B', 0x14]).unwrap(),
        Some(kv(&[b'B', 0x28], b"bs1"))
    );
    assert_eq!(
        e.seek_floor(&[b'B', 0x14]).unwrap(),
        Some(kv(&[b'B', 0x01], b"bs0"))
    );

    // Past the last label under 'B' -- ceil must cross over into 'C'.
    assert_eq!(e.seek_ceil(&[b'B', 0xff]).unwrap(), Some(kv(b"Ca", b"cs0")));
    assert_eq!(
        e.seek_floor(&[b'B', 0xff]).unwrap(),
        Some(kv(&[b'B', 0xf0], b"bs6"))
    );
}

#[test]
fn seek_within_the_non_root_direct_addressing_node_around_its_gap() {
    let fst = load_fst("fst_seek_non_root_array_node");
    let mut e = fst.iter().unwrap();

    assert_eq!(e.seek_exact(b"Df").unwrap(), Some(kv(b"Df", b"da5")));
    assert_eq!(e.seek_exact(b"Dg").unwrap(), None);

    // 'g' is absent but inside the range -- ceil skips straight to 'h'.
    assert_eq!(e.seek_ceil(b"Dg").unwrap(), Some(kv(b"Dh", b"da6")));
    assert_eq!(e.seek_floor(b"Dg").unwrap(), Some(kv(b"Df", b"da5")));

    // Before the first label under 'D' -- ceil must cross back from 'C'.
    assert_eq!(e.seek_ceil(&[b'C', 0xff]).unwrap(), Some(kv(b"Da", b"da0")));
    assert_eq!(
        e.seek_floor(&[b'D', 0x00]).unwrap(),
        Some(kv(b"Cg", b"cs6"))
    );
}

#[test]
fn seek_within_the_non_root_continuous_node_at_its_edges() {
    let fst = load_fst("fst_seek_non_root_array_node");
    let mut e = fst.iter().unwrap();

    assert_eq!(e.seek_exact(b"Cd").unwrap(), Some(kv(b"Cd", b"cs3")));
    assert_eq!(e.seek_exact(b"Ch").unwrap(), None);

    // Just past the last present label under 'C' -- ceil crosses into 'D'.
    assert_eq!(e.seek_ceil(b"Ch").unwrap(), Some(kv(b"Da", b"da0")));
    assert_eq!(e.seek_floor(b"Ch").unwrap(), Some(kv(b"Cg", b"cs6")));

    // A mid-range label lands exactly (no gaps to skip over).
    assert_eq!(e.seek_ceil(b"Cd").unwrap(), Some(kv(b"Cd", b"cs3")));
    assert_eq!(e.seek_floor(b"Cd").unwrap(), Some(kv(b"Cd", b"cs3")));
}

#[test]
fn seek_past_every_key_in_the_non_root_array_node_fixture_finds_nothing() {
    let fst = load_fst("fst_seek_non_root_array_node");
    let mut e = fst.iter().unwrap();
    assert_eq!(e.seek_ceil(&[0xff, 0xff]).unwrap(), None);
    assert_eq!(e.seek_floor(&[0x00]).unwrap(), None);
}

// --- `fixtures/data/fst_seek_floor_backtrack_*/`: root is itself array-
// encoded, and one of its labels (120 / 'd' / 'd' respectively) has its own
// ARCS_FOR_CONTINUOUS child -- exercising `find_next_floor_arc_*` via
// `backtrack_to_floor_arc` re-reading an array-encoded *parent*. ------------

#[test]
fn seek_floor_backtracks_into_a_binary_search_root_from_its_continuous_child() {
    let fst = load_fst("fst_seek_floor_backtrack_binary_search");
    let mut e = fst.iter().unwrap();
    // Root labels: 1, 40, 80, 120(->child), 160, 200, 240. Target's second
    // byte (0x00) is below the child's first label ('a' = 0x61), forcing a
    // backtrack from the child node up into the binary-search-encoded root
    // to find the floor arc there (80, the label just below 120).
    assert_eq!(
        e.seek_floor(&[120, 0x00]).unwrap(),
        Some(kv(&[80], b"out2"))
    );
}

#[test]
fn seek_floor_backtracks_into_a_direct_addressing_root_from_its_continuous_child() {
    let fst = load_fst("fst_seek_floor_backtrack_direct_addressing");
    let mut e = fst.iter().unwrap();
    // Root labels: a, b, c, d(->child), e, f, h (gap at 'g'). Backtracking
    // from d's child up into the direct-addressing-encoded root must find
    // the floor arc 'c', the label just below 'd'.
    assert_eq!(e.seek_floor(b"d\x00").unwrap(), Some(kv(b"c", b"out2")));
}

#[test]
fn seek_floor_backtracks_into_a_continuous_root_from_its_continuous_child() {
    let fst = load_fst("fst_seek_floor_backtrack_continuous");
    let mut e = fst.iter().unwrap();
    // Root labels: a, b, c, d(->child), e, f, g, fully contiguous. Same
    // backtrack, but now the parent (root) is itself ARCS_FOR_CONTINUOUS.
    assert_eq!(e.seek_floor(b"d\x00").unwrap(), Some(kv(b"c", b"out2")));
}
