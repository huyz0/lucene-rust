//! Seek/enumeration coverage over the **off-heap, zero-copy** `Fst::read_borrowed`
//! path -- closes a real gap found while investigating "FST off-heap seek path
//! validation": `fst_seek_fixtures.rs` (this crate's differential seek tests)
//! and `fst.rs`'s in-module seek unit tests both exercise `seek_ceil`/
//! `seek_floor`/`seek_exact`/`Fst::iter` exhaustively across every node
//! encoding (list, binary search, direct addressing, continuous) and
//! backtracking scenario -- but every single one of them builds its `Fst` via
//! `Fst::read` (the owned-copy constructor). Before this file, **no test
//! anywhere exercised seek or enumeration against a `Fst` opened via
//! `Fst::read_borrowed`** -- the only `read_borrowed` tests in `fst.rs`
//! (`read_borrowed_matches_read_for_same_bytes`,
//! `read_borrowed_over_a_real_mmap_directory_input`,
//! `read_borrowed_body_is_a_slice_not_a_second_owned_buffer`) only ever call
//! `Fst::get`/`Fst::get_typed` -- a single-arc-chain descent, not the
//! push/backtrack stack `FstEnum` maintains.
//!
//! This file proves `Fst::read` (owned) and `Fst::read_borrowed` (borrowed,
//! zero-copy) produce IDENTICAL seek/enumeration results over the same real,
//! Lucene-written bytes spanning every node encoding this port decodes,
//! including the non-root array-node and floor-backtrack fixtures that stress
//! backtracking specifically -- and includes one test that opens the borrowed
//! `Fst` over a REAL `MmapDirectory`-backed file (mirroring
//! `read_borrowed_over_a_real_mmap_directory_input`'s pattern) and performs
//! seek/enumeration against that mmap-backed slice, not just an in-memory
//! byte buffer.

use lucene_codecs::fst::Fst;
use lucene_store::data_input::SliceInput;

fn load_bytes(dir: &str) -> Vec<u8> {
    let path = format!(
        "{}/../../fixtures/data/{dir}/fst.bin",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read(&path).unwrap_or_else(|e| panic!("run fixtures generator first ({dir}): {e}"))
}

fn load_owned(dir: &str) -> Fst<'static> {
    let buf = load_bytes(dir);
    let mut input = SliceInput::new(&buf);
    Fst::read(&mut input).expect("decode real Lucene-written FST (owned)")
}

fn load_borrowed(buf: &[u8]) -> Fst<'_> {
    let mut input = SliceInput::new(buf);
    let fst = Fst::read_borrowed(&mut input).expect("decode real Lucene-written FST (borrowed)");
    assert!(fst.is_borrowed(), "expected a borrowed Fst");
    fst
}

fn kv(k: &[u8], v: &[u8]) -> (Vec<u8>, Vec<u8>) {
    (k.to_vec(), v.to_vec())
}

fn collect_iter(fst: &Fst) -> Vec<(Vec<u8>, Vec<u8>)> {
    fst.iter()
        .expect("iter should be supported")
        .collect::<Result<_, _>>()
        .expect("enumeration should not error")
}

/// Every fixture directory this crate's `fst_seek_fixtures.rs` (and the
/// non-seek `fst_*_fixtures.rs` files) already validate against `Fst::read`,
/// covering every node encoding (list/binary-search/direct-addressing/
/// continuous) plus the non-root-array-node and floor-backtrack scenarios.
const FIXTURE_DIRS: &[&str] = &[
    "fst",
    "fst_binary_search",
    "fst_direct_addressing",
    "fst_continuous",
    "fst_seek_non_root_array_node",
    "fst_seek_floor_backtrack_binary_search",
    "fst_seek_floor_backtrack_direct_addressing",
    "fst_seek_floor_backtrack_continuous",
];

/// For every fixture spanning every node encoding, `Fst::iter`'s full ordered
/// enumeration must be byte-identical whether the `Fst` came from `Fst::read`
/// (owned) or `Fst::read_borrowed` (borrowed/zero-copy) -- this also
/// implicitly stress-tests that minimization/suffix-sharing interacts
/// correctly with the off-heap path, since enumeration walks every arc in the
/// FST, not just the ones a handful of point lookups would touch.
#[test]
fn iter_agrees_between_owned_and_borrowed_across_every_node_encoding() {
    for dir in FIXTURE_DIRS {
        let bytes = load_bytes(dir);
        let owned = load_owned(dir);
        let borrowed = load_borrowed(&bytes);

        let owned_entries = collect_iter(&owned);
        let borrowed_entries = collect_iter(&borrowed);
        assert_eq!(
            owned_entries, borrowed_entries,
            "iter() mismatch between owned/borrowed for fixture {dir}"
        );
        assert!(
            !owned_entries.is_empty(),
            "fixture {dir} unexpectedly empty"
        );
    }
}

/// `seek_ceil`/`seek_floor`/`seek_exact` -- both the stateless `Fst` variants
/// and the stateful `FstEnum` variants -- must agree between owned and
/// borrowed for every key actually present in each fixture, replaying the
/// exact same targets `fst_seek_fixtures.rs` uses against `Fst::read`.
#[test]
fn seek_exact_agrees_between_owned_and_borrowed_for_present_keys() {
    let cases: &[(&str, &[&[u8]])] = &[
        (
            "fst",
            &[
                b"app",
                b"apple",
                b"application",
                b"banana",
                b"band",
                b"bandana",
                b"z",
            ],
        ),
        ("fst_binary_search", &[&[0x01], &[0x28], &[0x50], &[0xf0]]),
        ("fst_direct_addressing", &[b"a", b"c", b"f", b"h"]),
        ("fst_continuous", &[b"a", b"d", b"g"]),
        (
            "fst_seek_non_root_array_node",
            &[&[b'B', 0x01][..], b"Da", b"Df", b"Ca", b"Cd"],
        ),
    ];

    for (dir, keys) in cases {
        let bytes = load_bytes(dir);
        let owned = load_owned(dir);
        let borrowed = load_borrowed(&bytes);

        for key in *keys {
            let owned_result = owned.seek_exact(key).unwrap();
            let borrowed_result = borrowed.seek_exact(key).unwrap();
            assert_eq!(
                owned_result, borrowed_result,
                "Fst::seek_exact mismatch fixture={dir} key={key:?}"
            );
            assert!(
                owned_result.is_some(),
                "expected key={key:?} present in fixture {dir}"
            );

            let mut oe = owned.iter().unwrap();
            let mut be = borrowed.iter().unwrap();
            assert_eq!(
                oe.seek_exact(key).unwrap(),
                be.seek_exact(key).unwrap(),
                "FstEnum::seek_exact mismatch fixture={dir} key={key:?}"
            );
        }
    }
}

/// `seek_ceil`/`seek_floor` land-between-keys and edge-of-range scenarios,
/// replayed identically over both owned and borrowed `Fst`s, across list,
/// binary-search, direct-addressing and continuous root encodings -- this is
/// the "more complex traversal" the task calls out: each of these walks a
/// push/pop stack across levels, sometimes backtracking to a previous arc or
/// ascending to the parent, which a single `get()` call never does.
#[test]
fn seek_ceil_and_floor_agree_between_owned_and_borrowed_across_encodings() {
    let targets: &[(&str, &[&[u8]])] = &[
        ("fst", &[b"appl", b"ban", b"bane", b"", b"AAA", b"zz"]),
        ("fst_binary_search", &[&[0x14][..], &[0x00], &[0xff]]),
        ("fst_direct_addressing", &[b"g", b"`", &[0xff]]),
        ("fst_continuous", &[b"h", &[0x60], b"d"]),
        (
            "fst_seek_non_root_array_node",
            &[&[b'B', 0x14][..], &[b'B', 0xff], b"Dg", b"Ch"],
        ),
    ];

    for (dir, keys) in targets {
        let bytes = load_bytes(dir);
        let owned = load_owned(dir);
        let borrowed = load_borrowed(&bytes);

        for key in *keys {
            let mut oe_ceil = owned.iter().unwrap();
            let mut be_ceil = borrowed.iter().unwrap();
            assert_eq!(
                oe_ceil.seek_ceil(key).unwrap(),
                be_ceil.seek_ceil(key).unwrap(),
                "seek_ceil mismatch fixture={dir} key={key:?}"
            );

            let mut oe_floor = owned.iter().unwrap();
            let mut be_floor = borrowed.iter().unwrap();
            assert_eq!(
                oe_floor.seek_floor(key).unwrap(),
                be_floor.seek_floor(key).unwrap(),
                "seek_floor mismatch fixture={dir} key={key:?}"
            );
        }
    }
}

/// The floor-backtrack fixtures specifically stress `seek_floor` ascending
/// from an array-encoded child back up into its (also array-encoded) parent
/// to find the floor arc there -- these are the sharpest edge cases in the
/// whole seek surface for exactly the kind of backward/upward access pattern
/// a zero-copy borrowed slice could subtly mishandle (e.g. if any part of the
/// borrowed-path implementation assumed forward-only access). Confirms owned
/// and borrowed agree on the precise expected result, not just on each other.
#[test]
fn seek_floor_backtrack_scenarios_agree_between_owned_and_borrowed() {
    type Case = (&'static str, &'static [u8], (&'static [u8], &'static [u8]));
    let cases: &[Case] = &[
        (
            "fst_seek_floor_backtrack_binary_search",
            &[120, 0x00],
            (&[80], b"out2"),
        ),
        (
            "fst_seek_floor_backtrack_direct_addressing",
            b"d\x00",
            (b"c", b"out2"),
        ),
        (
            "fst_seek_floor_backtrack_continuous",
            b"d\x00",
            (b"c", b"out2"),
        ),
    ];

    for (dir, target, expected) in cases {
        let bytes = load_bytes(dir);
        let owned = load_owned(dir);
        let borrowed = load_borrowed(&bytes);

        let expected_kv = Some(kv(expected.0, expected.1));

        let mut oe = owned.iter().unwrap();
        assert_eq!(oe.seek_floor(target).unwrap(), expected_kv, "owned {dir}");

        let mut be = borrowed.iter().unwrap();
        assert_eq!(
            be.seek_floor(target).unwrap(),
            expected_kv,
            "borrowed {dir}"
        );
    }
}

/// Same seek/enumeration surface, but the borrowed `Fst` is opened over a
/// REAL `MmapDirectory`-backed file (an actual OS `mmap(2)`), not just an
/// in-memory `Vec<u8>` slice -- mirrors `fst.rs`'s
/// `read_borrowed_over_a_real_mmap_directory_input` pattern but drives the
/// full seek/enumeration API surface (not just `get`) against the mapped
/// bytes, including a backtracking `seek_floor` call.
#[test]
fn seek_and_enumerate_over_a_real_mmap_directory_backed_borrowed_fst() {
    use lucene_store::directory::{Directory, MmapDirectory};

    let file_bytes = load_bytes("fst");

    let mut root = std::env::temp_dir();
    root.push(format!(
        "lucene-rust-fst-borrowed-seek-mmap-test-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&root).unwrap();

    let dir = MmapDirectory::open(&root);
    {
        use lucene_store::DataOutput;
        let mut out = dir.create_output("fst.bin").unwrap();
        out.write_bytes(&file_bytes);
        out.close().unwrap();
    }

    let mapped = dir.open("fst.bin").unwrap();
    let mut input = SliceInput::new(&mapped);
    let fst = Fst::read_borrowed(&mut input).unwrap();
    assert!(fst.is_borrowed());

    // Plain seek_exact through the stateless wrapper.
    assert_eq!(fst.seek_exact(b"app").unwrap(), Some(b"1".to_vec()));
    assert_eq!(fst.seek_exact(b"missing").unwrap(), None);

    // Stateful FstEnum: seek_ceil landing strictly between two keys.
    let mut e = fst.iter().unwrap();
    assert_eq!(
        e.seek_ceil(b"appl").unwrap(),
        Some((b"apple".to_vec(), b"2".to_vec()))
    );
    // seek_floor from the same stateful enum -- exercises backtracking to a
    // previous arc/level from wherever seek_ceil left the stack.
    assert_eq!(
        e.seek_floor(b"bandanas").unwrap(),
        Some((b"bandana".to_vec(), b"6".to_vec()))
    );
    // seek_ceil past the last key.
    assert_eq!(e.seek_ceil(b"zzzzzzzzzzzz").unwrap(), None);

    // Full ordered enumeration off a fresh FstEnum over the same mapped bytes.
    let it = fst.iter().unwrap();
    let mut all = Vec::new();
    for entry in it {
        all.push(entry.unwrap());
    }
    assert_eq!(
        all,
        vec![
            kv(b"app", b"1"),
            kv(b"apple", b"2"),
            kv(b"application", b"3"),
            kv(b"banana", b"4"),
            kv(b"band", b"5"),
            kv(b"bandana", b"6"),
            kv(b"z", b"26"),
        ]
    );

    std::fs::remove_dir_all(&root).ok();
}
