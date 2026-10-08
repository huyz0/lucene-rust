//! Corrupt collation data never panics: the root `ucadata.icu` and a
//! tailoring's `%%CollationBin` with each index word rewritten, truncated
//! or byte-flipped either fail to read with a typed error or read; resource
//! bundles flipped byte by byte answer lookups without panicking.
#![allow(clippy::arithmetic_side_effects)] // test code: no value read off disk

use lucene_analysis_icu::icu4j::coll::data::{self, CollationTailoring};
use lucene_analysis_icu::icu4j::coll::res::{pack_file, ResReader};

fn header_size(b: &[u8]) -> usize {
    usize::from(u16::from_be_bytes([b[0], b[1]]))
}

fn root_bytes() -> &'static [u8] {
    pack_file("ucadata.icu").unwrap()
}

fn tailoring_bytes(bundle: &str, kind: &str) -> &'static [u8] {
    let r = ResReader::new(pack_file(bundle).unwrap()).unwrap();
    let colls = r.table_get(r.root(), "collations").unwrap();
    let t = r.table_get(colls, kind).unwrap();
    r.binary(r.table_get(t, "%%CollationBin").unwrap()).unwrap()
}

fn root() -> CollationTailoring {
    data::read(None, root_bytes()).unwrap()
}

/// Each index word set to values around and beyond its neighbours.
fn index_mutations(bytes: &[u8]) -> Vec<Vec<u8>> {
    let h = header_size(bytes);
    let count = i32::from_be_bytes([bytes[h], bytes[h + 1], bytes[h + 2], bytes[h + 3]]);
    let mut out = Vec::new();
    for i in 0..count.min(20) as usize {
        let at = h + 4 * i;
        let v = i32::from_be_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]);
        for nv in [
            -1,
            0,
            1,
            v + 4,
            v - 4,
            v + 8,
            v - 1,
            v + 1,
            i32::MAX,
            bytes.len() as i32,
        ] {
            let mut b = bytes.to_vec();
            b[at..at + 4].copy_from_slice(&nv.to_be_bytes());
            out.push(b);
        }
    }
    out
}

#[test]
fn rewritten_root_indexes_never_panic() {
    let bytes = root_bytes();
    let mut failed = 0;
    let mut total = 0;
    for b in index_mutations(bytes) {
        total += 1;
        failed += usize::from(data::read(None, &b).is_err());
    }
    assert!(failed > total / 3, "{failed} of {total}");
    // A tailoring may not be read as the root, nor the root over itself
    // with a different UCA version.
    let mut b = bytes.to_vec();
    b[21] ^= 0x10;
    let base = root();
    assert!(data::read(Some(&base), &b).is_err());
}

#[test]
fn rewritten_tailoring_indexes_never_panic() {
    let base = root();
    for (bundle, kind) in [
        ("de.res", "phonebook"),
        ("ja.res", "standard"),
        ("zh.res", "pinyin"),
        ("da.res", "standard"),
        ("sr.res", "standard"),
    ] {
        let bytes = tailoring_bytes(bundle, kind);
        assert!(data::read(Some(&base), bytes).is_ok(), "{bundle}");
        let mut failed = 0;
        for b in index_mutations(bytes) {
            failed += usize::from(data::read(Some(&base), &b).is_err());
        }
        assert!(failed > 0, "{bundle}");
        for len in (0..bytes.len()).step_by(61) {
            assert!(data::read(Some(&base), &bytes[..len]).is_err() || len > 64);
        }
    }
}

#[test]
fn truncated_and_flipped_root_never_panics() {
    let bytes = root_bytes();
    let mut failed = 0;
    for len in (0..bytes.len()).step_by(4099) {
        failed += usize::from(data::read(None, &bytes[..len]).is_err());
    }
    assert!(failed > 10, "{failed}");
    let h = header_size(bytes);
    for at in (h..h + 400).chain((h + 400..bytes.len()).step_by(7919)) {
        let mut b = bytes.to_vec();
        b[at] ^= 0xa5;
        let _ = data::read(None, &b);
    }
}

#[test]
fn flipped_bundles_answer_lookups() {
    for name in ["de.res", "root.res", "res_index.res", "translit/root.res"] {
        let bytes = pack_file(name).unwrap();
        let h = header_size(bytes);
        let mut readable = 0;
        let positions: Vec<usize> = (0..h + 160)
            .chain((h + 160..bytes.len()).step_by(bytes.len() / 300 + 1))
            .collect();
        for at in positions {
            let mut b = bytes.to_vec();
            b[at] ^= 0x5a;
            let b: &'static [u8] = Box::leak(b.into_boxed_slice());
            let Ok(r) = ResReader::new(b) else {
                continue;
            };
            readable += 1;
            let root = r.root();
            let _ = r.no_fallback();
            for (key, item) in r.table_entries(root).into_iter().take(40) {
                let _ = r.string(item);
                let _ = r.binary(item);
                let _ = r.table_get(root, &key);
                for (_, inner) in r.table_entries(item).into_iter().take(10) {
                    let _ = r.string(inner);
                    let _ = r.binary(inner);
                    let _ = ResReader::is_table(inner);
                }
            }
            for key in [
                "collations",
                "Version",
                "%%ALIAS",
                "%%Parent",
                "InstalledLocales",
                "zz",
            ] {
                if let Some(x) = r.table_get(root, key) {
                    let _ = r.string(x);
                    let _ = r.table_get(x, "standard");
                }
            }
        }
        assert!(readable > 0, "{name}");
    }
}
