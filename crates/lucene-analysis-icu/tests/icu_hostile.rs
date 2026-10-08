//! Corrupt ICU data never panics: every truncation and a byte-flip sweep of
//! the `.nrm` files a caller may hand to `Normalizer2::from_data` (Java's
//! `Normalizer2.getInstance(InputStream, ...)`) either fails to load with a
//! typed error or normalizes text without panicking.
#![allow(clippy::arithmetic_side_effects)] // test code: no value read off disk

use lucene_analysis_icu::icu4j::normalizer2::{Mode, Normalizer2};

const NFC: &[u8] = include_bytes!("../src/resources/nfc.nrm");
const UTR30: &[u8] = include_bytes!("../src/resources/utr30.nrm");

fn texts() -> Vec<Vec<u16>> {
    [
        "Résumé ﬁ ÅΩ ǅ ㎞ 각 ᄀ\u{1161}\u{11a8} a\u{301}\u{316}\u{334} \u{f73}\u{f71}",
        "\u{1d15e}\u{1d165}\u{1d16d} \u{11099}\u{110ba} \u{1100}\u{1161} \u{ac00}\u{11a8}",
        "plain ascii text",
        "\u{e9}\u{e8}\u{301}\u{300}\u{327}\u{328}e\u{301}\u{302}\u{303}",
    ]
    .iter()
    .map(|s| s.encode_utf16().collect())
    .collect()
}

fn exercise(bytes: &[u8]) -> usize {
    let mut n = 0;
    for mode in [
        Mode::Compose,
        Mode::Decompose,
        Mode::Fcd,
        Mode::ComposeContiguous,
    ] {
        let Ok(norm) = Normalizer2::from_data(bytes, mode) else {
            continue;
        };
        for t in texts() {
            n += norm.normalize(&t).len();
            n += norm.span_quick_check_yes(&t);
            let _ = norm.quick_check(&t);
            let mut first = t[..t.len() / 2].to_vec();
            norm.normalize_second_and_append(&mut first, &t[t.len() / 2..]);
            n += first.len();
            for &c in t.iter().take(8) {
                let c = i32::from(c);
                n += usize::from(norm.has_boundary_before(c) && norm.has_boundary_after(c));
                n += norm.get_decomposition(c).map_or(0, |d| d.len());
                n += norm.get_raw_decomposition(c).map_or(0, |d| d.len());
                n += usize::from(norm.compose_pair(c, 0x301) > 0);
            }
        }
    }
    n
}

#[test]
fn truncated_nrm_files_never_panic() {
    for data in [NFC, UTR30] {
        let mut failed = 0;
        for len in (0..data.len()).step_by(97).chain([data.len() - 1]) {
            failed += usize::from(Normalizer2::from_data(&data[..len], Mode::Compose).is_err());
            exercise(&data[..len]);
        }
        // Everything short of the small-FCD bitmap at the end fails to load.
        assert!(failed >= (data.len() - 0x100) / 97, "{failed}");
    }
}

#[test]
fn flipped_nrm_bytes_never_panic() {
    for data in [NFC, UTR30] {
        let mut loaded = 0;
        let positions = (0..200).chain((200..data.len()).step_by(101));
        for at in positions {
            for flip in [0xffu8, 0x01] {
                let mut b = data.to_vec();
                b[at] ^= flip;
                if Normalizer2::from_data(&b, Mode::Compose).is_ok() {
                    loaded += 1;
                }
                exercise(&b);
            }
        }
        assert!(loaded > 300, "{loaded}");
    }
}
