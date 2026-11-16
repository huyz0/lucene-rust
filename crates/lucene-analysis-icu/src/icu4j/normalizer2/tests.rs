#![cfg(test)]
use super::*;
use crate::icu4j::utf16::units;

#[test]
fn filtered_quick_yes_follows_the_inner_check() {
    let set = UnicodeSet::from_pattern("[^\\u00e9]").unwrap();
    let f = Normalizer2::filtered(Normalizer2::nfc(), set).unwrap();
    assert!(f.quick_yes_utf8("abc"));
    // A combining mark fails the per-character loop: not known.
    assert!(!f.quick_yes_utf8("e\u{301}"));
    assert_eq!(f.quick_check(&units("abc")), QuickCheck::Yes);
}

fn s(v: &[u16]) -> String {
    String::from_utf16(v).unwrap()
}

#[test]
fn builtin_forms() {
    let nfc = Normalizer2::nfc();
    let nfd = Normalizer2::get_instance("nfc", Mode::Decompose).unwrap();
    assert_eq!(s(&nfc.normalize(&units("e\u{301}"))), "\u{e9}");
    assert_eq!(s(&nfd.normalize(&units("\u{e9}"))), "e\u{301}");
    assert_eq!(s(&Normalizer2::nfkc().normalize(&units("\u{fb01}"))), "fi");
    assert_eq!(
        s(&Normalizer2::nfkc_casefold().normalize(&units("ABC"))),
        "abc"
    );
    assert_eq!(nfc.quick_check(&units("abc")), QuickCheck::Yes);
    assert_eq!(nfc.quick_check(&units("e\u{301}")), QuickCheck::Maybe);
    assert_eq!(nfc.quick_check(&units("\u{212b}")), QuickCheck::No);
    assert_eq!(nfd.quick_check(&units("\u{e9}")), QuickCheck::No);
    assert_eq!(nfd.quick_check(&units("e")), QuickCheck::Yes);
    assert!(nfc.is_normalized(&units("\u{e9}")));
    assert!(!nfd.is_normalized(&units("\u{e9}")));
    for name in ["nfkc_scf", "uts46"] {
        assert!(Normalizer2::get_instance(name, Mode::Compose).is_ok());
    }
    let e = Normalizer2::get_instance("bogus", Mode::Compose).unwrap_err();
    assert_eq!(e.kind(), IcuErrorKind::MissingResource);
    // Hangul composes and decomposes algorithmically.
    assert_eq!(
        s(&nfc.normalize(&units("\u{1100}\u{1161}\u{11a8}"))),
        "\u{ac01}"
    );
    assert_eq!(
        s(&nfd.normalize(&units("\u{ac01}"))),
        "\u{1100}\u{1161}\u{11a8}"
    );
    // Canonical reordering.
    assert_eq!(
        s(&nfd.normalize(&units("a\u{301}\u{316}"))),
        "a\u{316}\u{301}"
    );
    let fcd = Normalizer2::get_instance("nfc", Mode::Fcd).unwrap();
    assert_eq!(
        s(&fcd.normalize(&units("a\u{301}\u{316}"))),
        "a\u{316}\u{301}"
    );
    assert_eq!(fcd.span_quick_check_yes(&units("abc")), 3);
    let fcc = Normalizer2::get_instance("nfc", Mode::ComposeContiguous).unwrap();
    assert_eq!(s(&fcc.normalize(&units("e\u{301}"))), "\u{e9}");
    assert!(fcc.is_normalized(&units("\u{e9}")));
}

#[test]
fn append_forms() {
    for (mode, expect) in [
        (Mode::Compose, "\u{e9}"),
        (Mode::ComposeContiguous, "\u{e9}"),
        (Mode::Decompose, "e\u{301}"),
        (Mode::Fcd, "e\u{301}"),
    ] {
        let n = Normalizer2::get_instance("nfc", mode).unwrap();
        let mut first = units("e");
        n.normalize_second_and_append(&mut first, &units("\u{301}"));
        assert_eq!(s(&first), expect, "{mode:?}");
        let mut first = units("e");
        n.append(&mut first, &units("\u{301}"));
        // append() still normalizes across the boundary.
        assert_eq!(s(&first), expect, "{mode:?}");
    }
}

#[test]
fn noop_and_filtered() {
    let noop = Normalizer2::noop();
    assert_eq!(noop.normalize(&units("\u{e9}")), units("\u{e9}"));
    let mut f = units("a");
    noop.normalize_second_and_append(&mut f, &units("b"));
    assert_eq!(s(&f), "ab");
    assert_eq!(noop.quick_check(&units("x")), QuickCheck::Yes);
    assert_eq!(noop.span_quick_check_yes(&units("xy")), 2);
    assert!(noop.is_normalized(&units("x")));
    assert!(noop.has_boundary_before(1) && noop.has_boundary_after(1) && noop.is_inert(1));
    assert_eq!(noop.get_decomposition(0xe9), None);
    assert_eq!(noop.get_raw_decomposition(0xe9), None);
    assert_eq!(noop.compose_pair(0x65, 0x301), -1);
    assert_eq!(noop.get_combining_class(0x301), 0);

    // NFD everything but U+00E9.
    let set = UnicodeSet::from_pattern("[^\\u00e9]").unwrap();
    let nfd = Normalizer2::get_instance("nfc", Mode::Decompose).unwrap();
    let f = Normalizer2::filtered(nfd.clone(), set).unwrap();
    assert_eq!(s(&f.normalize(&units("\u{e9}\u{e8}"))), "\u{e9}e\u{300}");
    assert_eq!(f.quick_check(&units("\u{e9}")), QuickCheck::Yes);
    assert_eq!(f.quick_check(&units("\u{e8}")), QuickCheck::No);
    assert_eq!(f.span_quick_check_yes(&units("\u{e9}a\u{e8}")), 2);
    assert_eq!(f.span_quick_check_yes(&units("\u{e9}a")), 2);
    assert!(f.is_normalized(&units("\u{e9}a")));
    assert!(!f.is_normalized(&units("\u{e8}")));
    assert!(f.has_boundary_before(0xe9) && f.has_boundary_after(0xe9) && f.is_inert(0xe9));
    assert!(f.has_boundary_before(0x61) && f.has_boundary_after(0x61) && f.is_inert(0x61));
    assert_eq!(f.get_decomposition(0xe9), None);
    assert_eq!(f.get_decomposition(0xe8), Some(units("e\u{300}")));
    assert_eq!(f.get_raw_decomposition(0xe9), None);
    assert_eq!(f.get_raw_decomposition(0xe8), Some(units("e\u{300}")));
    assert_eq!(f.compose_pair(0x65, 0x300), 0xe8);
    assert_eq!(f.compose_pair(0xe9, 0x300), -1);
    assert_eq!(f.get_combining_class(0x300), 230);
    assert_eq!(f.get_combining_class(0xe9), 0);
    let mut first = Vec::new();
    f.normalize_second_and_append(&mut first, &units("\u{e8}"));
    assert_eq!(s(&first), "e\u{300}");
    let mut first = Vec::new();
    f.append(&mut first, &units("\u{e8}"));
    assert_eq!(s(&first), "\u{e8}");
    let mut first = units("e");
    f.normalize_second_and_append(&mut first, &units("\u{300}\u{e9}\u{e8}"));
    assert_eq!(s(&first), "e\u{300}\u{e9}e\u{300}");
    let mut first = units("\u{e9}e");
    f.normalize_second_and_append(&mut first, &units("\u{300}"));
    assert_eq!(s(&first), "\u{e9}e\u{300}");
    let mut first = units("\u{e9}");
    f.append(&mut first, &units("\u{e9}x"));
    assert_eq!(s(&first), "\u{e9}\u{e9}x");
    let strings = UnicodeSet::from_pattern("[{ab}]").unwrap();
    assert!(Normalizer2::filtered(nfd, strings).is_err());
}

#[test]
fn decompositions_and_pairs() {
    let nfc = Normalizer2::nfc();
    assert_eq!(nfc.get_decomposition(0xe9), Some(units("e\u{301}")));
    assert_eq!(nfc.get_decomposition(0x61), None);
    assert_eq!(
        nfc.get_decomposition(0xac01),
        Some(units("\u{1100}\u{1161}\u{11a8}"))
    );
    assert_eq!(
        nfc.get_raw_decomposition(0xac01),
        Some(units("\u{ac00}\u{11a8}"))
    );
    assert_eq!(
        nfc.get_raw_decomposition(0xac00),
        Some(units("\u{1100}\u{1161}"))
    );
    assert_eq!(
        nfc.get_raw_decomposition(0x1e69),
        Some(units("\u{1e63}\u{307}"))
    );
    assert_eq!(nfc.get_raw_decomposition(0x61), None);
    assert_eq!(nfc.get_raw_decomposition(0x212b), Some(units("\u{c5}")));
    let nfkc = Normalizer2::nfkc();
    assert_eq!(nfkc.get_raw_decomposition(0xfb01), Some(units("fi")));
    assert_eq!(nfkc.get_decomposition(0x1e9b), Some(units("s\u{307}")));
    assert_eq!(
        nfkc.get_raw_decomposition(0x1e9b),
        Some(units("\u{17f}\u{307}"))
    );
    assert_eq!(nfc.compose_pair(0x65, 0x301), 0xe9);
    assert_eq!(nfc.compose_pair(0x1100, 0x1161), 0xac00);
    assert_eq!(nfc.compose_pair(0xac00, 0x11a8), 0xac01);
    assert_eq!(nfc.compose_pair(0xac00, 0x11a7), -1);
    assert_eq!(nfc.compose_pair(0x1100, 0x61), -1);
    assert_eq!(nfc.compose_pair(0x61, 0x62), -1);
    assert_eq!(nfc.compose_pair(0x65, -5), -1);
    assert_eq!(nfc.compose_pair(0x62, 0x307), 0x1e03);
    assert_eq!(nfc.get_combining_class(0x301), 230);
    assert!(nfc.has_boundary_before(0x61) && !nfc.has_boundary_before(0x301));
    assert!(!nfc.has_boundary_after(0x65) && nfc.is_inert(0x2e));
    let nfd = Normalizer2::get_instance("nfc", Mode::Decompose).unwrap();
    assert!(nfd.has_boundary_after(0x65) && nfd.is_inert(0x2e));
    let fcd = Normalizer2::get_instance("nfc", Mode::Fcd).unwrap();
    assert!(fcd.is_inert(0x61) && fcd.has_boundary_before(0x61) && fcd.has_boundary_after(0x61));
    let fcc = Normalizer2::get_instance("nfc", Mode::ComposeContiguous).unwrap();
    assert!(fcc.has_boundary_after(0x2e) && fcc.is_inert(0x2e));
    assert!(Normalizer2::from_data(&[1, 2, 3], Mode::Compose).is_err());
    assert!(Normalizer2::from_data(NFC_DATA, Mode::Compose).is_ok());
}
