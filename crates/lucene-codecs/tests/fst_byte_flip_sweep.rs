//! Single-byte corruption sweep over every real-Lucene-written FST fixture in
//! the tree, plus a larger builder-produced one.
//!
//! The arithmetic gate's third audit step (`docs/arithmetic-gate.md`, "A
//! module is not audited when the lint goes quiet"): flip each byte in turn
//! and require a typed error or a clean decode from every read path -- never a
//! panic, never an allocation abort, never a hang.
//!
//! Unlike the `.fdm`/`.tvd`/`.vemf` sweeps, there is nothing to **re-sign**
//! here: an FST file carries a `CodecUtil.writeHeader` and no footer at all
//! (`FSTMetadata.save` writes no checksum), so a flipped body byte already
//! reaches the decoder on its own merits. Flips inside the header are swept
//! too and are expected to be rejected by the header check.
//!
//! The corpus spans every node encoding this port decodes -- list-encoded
//! (`fst`, `fst_deep_trie`, both multi-node and multi-level),
//! `ARCS_FOR_BINARY_SEARCH`, `ARCS_FOR_DIRECT_ADDRESSING`,
//! `ARCS_FOR_CONTINUOUS` -- plus the `BYTE2`/`BYTE4` label widths and the
//! empty-key metadata branch. A single-node FST would leave the array node
//! headers, the presence bit-table and the `bytesPerArc` slot arithmetic
//! untouched, which measures the fixture rather than the decoder.
// Test-support code opts out of the arithmetic gate at the file boundary; see
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::fst::{self, Fst, InputType, PositiveIntOutputs};
use lucene_store::data_input::SliceInput;

const CORPUS: &[&str] = &[
    "fst",
    "fst_binary_search",
    "fst_continuous",
    "fst_deep_trie",
    "fst_direct_addressing",
    "fst_byte2",
    "fst_byte4",
    "fst_empty_key",
    "fst_seek_floor_backtrack_binary_search",
    "fst_seek_floor_backtrack_continuous",
    "fst_seek_floor_backtrack_direct_addressing",
    "fst_seek_non_root_array_node",
];

fn fixture(name: &str) -> Vec<u8> {
    let path = format!(
        "{}/../../fixtures/data/{name}/fst.bin",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read(&path).unwrap_or_else(|e| panic!("run the fixtures generator first: {path}: {e}"))
}

/// Probe keys chosen to drive every seek branch regardless of what the
/// fixture actually holds: before the first label, inside the range, past the
/// last, and a few multi-byte descents.
const PROBES: &[&[u8]] = &[
    b"",
    b"\x00",
    b"a",
    b"ab",
    b"abcaa",
    b"abd",
    b"g",
    b"z",
    b"\xff",
    b"\xff\xff\xff\xff",
];

/// Everything a caller can do with an `Fst`, so a flip anywhere has to
/// surface through one of them.
fn walk_everything(buf: &[u8]) -> fst::Result<()> {
    let mut input = SliceInput::new(buf);
    let owned = Fst::read(&mut input)?;

    // The zero-copy path parses the same metadata and must agree. It borrows
    // `buf` directly: a `Box::leak` here would be ~130 MB across the sweep's
    // ~40 000 invocations, and leaking by construction is the wrong shape in a
    // test whose whole purpose is "nothing reserves memory proportional to a
    // number it just read".
    let mut borrowed_in = SliceInput::new(buf);
    let borrowed = Fst::read_borrowed(&mut borrowed_in)?;

    for fst in [&owned, &borrowed] {
        // `Fst::get`/`Fst::iter` are `BYTE1`-only by contract; the `BYTE2`
        // and `BYTE4` fixtures go through the label-domain API instead, and
        // a flip that *changes* the declared input type legitimately swaps
        // which of the two answers.
        let byte1 = fst.metadata().input_type == InputType::Byte1;
        for probe in PROBES {
            let labels: Vec<i32> = probe.iter().map(|&b| b as i32).collect();
            fst.get_labels(&labels)?;
            let mut e = fst.iter_labels();
            e.seek_ceil_labels(&labels)?;
            let mut e = fst.iter_labels();
            e.seek_floor_labels(&labels)?;
            let mut e = fst.iter_labels();
            e.seek_exact_labels(&labels)?;
            if !byte1 {
                continue;
            }
            fst.get(probe)?;
            fst.seek_exact(probe)?;
            // The fixtures store `ByteSequenceOutputs` payloads, so a typed
            // decode of them legitimately fails; what matters here is that it
            // reports rather than panics.
            let _ = fst.get_typed::<PositiveIntOutputs>(probe);
            let mut e = fst.iter()?;
            e.seek_ceil(probe)?;
            let mut e = fst.iter()?;
            e.seek_floor(probe)?;
            let mut e = fst.iter()?;
            e.seek_exact(probe)?;
        }

        // A full ascending enumeration, in the label domain so it covers the
        // `BYTE2`/`BYTE4` fixtures too. Capped: a body whose arc targets have
        // been corrupted into a cycle would otherwise walk forever, and a
        // hang is a worse outcome than an abort (no timeout catches it). The
        // cap is far above any fixture's key count.
        let mut it = fst.iter_labels();
        let mut seen = 0usize;
        while let Some(item) = it.next_labels() {
            item?;
            seen += 1;
            assert!(seen <= 100_000, "enumeration did not terminate");
        }
        if !byte1 {
            continue;
        }
        let it = fst.iter()?;
        let mut seen = 0usize;
        for item in it {
            item?;
            seen += 1;
            assert!(seen <= 100_000, "enumeration did not terminate");
        }

        // The same walk through the `IntsRef` key domain. These fixtures are
        // `BYTE1` FSTs, so a key whose length is not a multiple of four is a
        // legitimate typed error rather than a decode failure -- what the
        // sweep is checking here is that it is an error and not a panic.
        let ints = fst.iter_ints()?;
        let mut seen = 0usize;
        for item in ints {
            if item.is_err() {
                break;
            }
            seen += 1;
            assert!(seen <= 100_000, "int enumeration did not terminate");
        }
        let mut ints = fst.iter_ints()?;
        let _ = ints.seek_ceil(&[97, 98]);
        let _ = ints.seek_floor(&[97, 98]);
        let _ = ints.seek_exact(&[97, 98]);
    }
    Ok(())
}

fn sweep(name: &str, buf: &[u8]) -> (usize, usize) {
    walk_everything(buf).unwrap_or_else(|e| panic!("{name}: the fixture itself must decode: {e}"));
    let mut flipped = 0usize;
    let mut rejected = 0usize;
    for at in 0..buf.len() {
        for bit in 0..8u8 {
            let mut patched = buf.to_vec();
            patched[at] ^= 1 << bit;
            flipped += 1;
            if walk_everything(&patched).is_err() {
                rejected += 1;
            }
        }
    }
    (rejected, flipped)
}

#[test]
fn every_single_byte_fst_corruption_is_an_error_or_a_clean_decode() {
    let mut total = (0usize, 0usize);
    for name in CORPUS {
        let buf = fixture(name);
        let (rejected, flipped) = sweep(name, &buf);
        eprintln!(
            "{name}: {rejected}/{flipped} rejected ({} bytes)",
            buf.len()
        );
        total.0 += rejected;
        total.1 += flipped;
    }
    eprintln!("FST byte-flip sweep: {}/{} rejected", total.0, total.1);
    // A low rate is not automatically a gap (see `docs/arithmetic-gate.md`):
    // a flipped *label* byte names a different but perfectly well-formed key,
    // and a flipped *output* byte a different but well-formed output -- both
    // are clean decodes of a different FST, not misses. The bar this sweep
    // enforces is the one that matters: nothing panics, nothing aborts,
    // nothing fails to terminate. The rate is asserted only as a regression
    // tripwire against the 37% measured at c31.
    assert!(
        total.0 * 10 > total.1 * 3,
        "only {}/{} flips rejected",
        total.0,
        total.1
    );
}

/// The fixtures above are all under 100 bytes, which is enough to cover every
/// *node encoding* but not enough to exercise a body with hundreds of nodes
/// and multi-byte `vlong` targets. This builds one with this port's own
/// `build_fst` and sweeps it the same way.
#[test]
fn every_single_byte_corruption_of_a_large_built_fst_is_an_error_or_a_clean_decode() {
    let entries: Vec<(Vec<u8>, Vec<u8>)> = (0u32..400)
        .map(|i| {
            (
                format!("term{i:05}").into_bytes(),
                format!("out{}", i * 7).into_bytes(),
            )
        })
        .collect();
    let fst = fst::build_fst(&entries).expect("entries are sorted");
    let buf = fst::write_fst(&fst);
    assert!(buf.len() > 1000, "fixture must have many nodes");
    let (rejected, flipped) = sweep("built", &buf);
    eprintln!(
        "built FST byte-flip sweep: {rejected}/{flipped} rejected ({} bytes)",
        buf.len()
    );
    // Lower than the fixture corpus's rate by construction: this FST's body
    // is mostly 400 terms' worth of label and output *payload*, where a flip
    // is a different well-formed key or output rather than a structural
    // error. Tripwire against the 12.9% measured at c31.
    assert!(
        rejected * 100 > flipped * 10,
        "only {rejected}/{flipped} rejected"
    );
}
