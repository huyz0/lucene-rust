#![cfg(test)]
// Test fixtures' own arithmetic -- see `docs/arithmetic-gate.md`'s "Test code".
#![allow(clippy::arithmetic_side_effects)]
use super::*;
use crate::data_output::DataOutput;
use lucene_util::test_support::TempDir;

fn write(dir: &dyn Directory, name: &str, bytes: &[u8]) {
    let mut out = dir.create_output(name).unwrap();
    out.write_bytes(bytes);
    out.close().unwrap();
}

fn pattern(len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| (i.wrapping_mul(31) ^ (i >> 9)) as u8)
        .collect()
}

#[test]
fn reads_match_the_file() {
    let tmp = TempDir::new("paged-fs");
    let dir = PagedFsDirectory::open(tmp.path());
    // Not a whole number of pages, so the last page is short.
    for &len in &[0usize, 10, 4096, 64 << 10, (64 << 10) + 1, 1_000_003] {
        let name = format!("f{len}.dat");
        let bytes = pattern(len);
        write(&dir, &name, &bytes);
        let input = dir.open(&name).unwrap();
        assert_eq!(&input[..], &bytes[..], "len {len}");
        assert_eq!(dir.file_length(&name).unwrap(), len as u64);
        // Out-of-order touches fault pages in independently.
        if len > 0 {
            let mid = len / 2;
            assert_eq!(input[mid], bytes[mid]);
            assert_eq!(input[len - 1], bytes[len - 1]);
        }
    }
}

#[test]
fn will_need_and_prefetch_fill_ahead() {
    let tmp = TempDir::new("paged-fs-prefetch");
    let dir = PagedFsDirectory::open(tmp.path());
    let bytes = pattern(3 << 20);
    write(&dir, "big.dat", &bytes);
    let input = dir.open("big.dat").unwrap();
    input.will_need(1 << 20, 512 << 10).unwrap();
    crate::prefetch(&[&input[2 << 20..(2 << 20) + 100]]).unwrap();
    if dir.is_paged() {
        assert!(crate::is_prefetchable(&input));
        assert!(crate::is_resident(&input[1 << 20..(1 << 20) + 10]));
        assert!(!crate::is_resident(&input[(7 << 18)..(7 << 18) + 10]));
    }
    assert_eq!(&input[..], &bytes[..]);
}

#[test]
fn pinned_files_are_read_whole() {
    let tmp = TempDir::new("paged-fs-pinned");
    let dir = PagedFsDirectory::open(tmp.path());
    let bytes = pattern(1 << 20);
    write(&dir, "_0.tip", &bytes);
    let before = whole_file_bytes();
    let input = dir.open("_0.tip").unwrap();
    assert!(!crate::paged_input::is_paged(&input));
    assert_eq!(&input[..], &bytes[..]);
    assert!(whole_file_bytes() >= before + (1 << 20));
    // A second open shares the bytes read at the first.
    let again = dir.open("_0.tip").unwrap();
    assert_eq!(again.as_ptr(), input.as_ptr());
}

/// A plain file's read failure is recorded under its full path, where
/// a call on its shard looks for it.
#[test]
fn an_unreadable_page_is_recorded_under_the_shard() {
    if crate::paged_input::Pager::global().is_none() {
        return;
    }
    let tmp = TempDir::new("paged-fs-unreadable");
    let root = tmp.path().to_string_lossy().to_string();
    let dir = PagedFsDirectory::open(tmp.path());
    write(&dir, "_0.doc", &pattern(4 << 20));
    let input = dir.open("_0.doc").unwrap();
    let since = crate::read_failures::epoch();
    // Cut the file under the open input: its last pages cannot be read.
    let f = std::fs::OpenOptions::new()
        .write(true)
        .open(tmp.path().join("_0.doc"))
        .unwrap();
    f.set_len(1 << 20).unwrap();
    crate::read_failures::set_checked(true);
    assert_eq!(input[3 << 20], 0);
    crate::read_failures::set_checked(false);
    let recorded = crate::read_failures::failure_under(&root, since).expect("recorded");
    assert!(recorded.contains("_0.doc"), "{recorded}");
}

#[test]
fn opens_of_one_file_share_its_pages() {
    if crate::paged_input::Pager::global().is_none() {
        return;
    }
    let tmp = TempDir::new("paged-fs-shared");
    let dir = PagedFsDirectory::open(tmp.path());
    let bytes = pattern(4 << 20);
    write(&dir, "_0.doc", &bytes);
    let a = dir.open("_0.doc").unwrap();
    let b = PagedFsDirectory::open(tmp.path()).open("_0.doc").unwrap();
    assert_eq!(a.as_ptr(), b.as_ptr());
    assert_eq!(b[3 << 20], bytes[3 << 20]);
    assert_eq!(a[3 << 20], bytes[3 << 20]);
    drop((a, b));
    // A file rewritten under the same name is a new file.
    let other = pattern(5 << 20);
    dir.delete_file("_0.doc").unwrap();
    write(&dir, "_0.doc", &other);
    let c = dir.open("_0.doc").unwrap();
    assert_eq!(&c[..], &other[..]);
}

#[test]
fn a_file_too_large_to_stay_resident_gets_a_page_from_its_advice() {
    use crate::read_advice::ReadAdvice;
    assert_eq!(super::small_page("xyz", ReadAdvice::Random), 16 << 10);
    assert_eq!(super::small_page("xyz", ReadAdvice::Normal), 16 << 10);
    assert_eq!(super::small_page("xyz", ReadAdvice::Auto), 16 << 10);
    assert_eq!(super::small_page("xyz", ReadAdvice::WillNeed), 16 << 10);
    // Vector data stays small when no advice says so.
    assert_eq!(super::small_page("vec", ReadAdvice::Normal), 4 << 10);
    // A scan keeps the large page: the caller takes the smaller of this and
    // the configured page.
    assert_eq!(super::small_page("doc", ReadAdvice::Sequential), usize::MAX);
    // The graph and quantized codes are advised random but keep 16 KiB.
    let c = PagedFsConfig::default();
    assert_eq!(c.page_size_for("_0_X_0.vex", 60 << 20, 100 << 20), 16 << 10);
    assert_eq!(c.page_size_for("_0_X_0.vec", 60 << 20, 100 << 20), 4 << 10);
}

#[test]
fn adaptive_pages_and_pins_follow_the_budget() {
    let c = PagedFsConfig::default();
    let budget = 100 << 20;
    // Fits: the default page; too large to stay resident: small pages.
    assert_eq!(c.page_size_for("_0.doc", 10 << 20, budget), c.page_size);
    assert_eq!(c.page_size_for("_0.doc", 60 << 20, budget), 16 << 10);
    assert_eq!(c.page_size_for("_0_X_0.vec", 60 << 20, budget), 4 << 10);
    // Search structures are pinned while small against the budget.
    assert!(c.pins("_0_X_0.vex", 12 << 20, budget));
    assert!(!c.pins("_0_X_0.vex", 13 << 20, budget));
    assert!(c.pins("_0.tip", 1 << 30, budget));
    assert!(!c.pins("_0.doc", 1, budget));
    // Explicit settings win; without `adaptive` sizes are fixed.
    let explicit = PagedFsConfig {
        page_sizes: vec![("vec".into(), 32 << 10)],
        ..PagedFsConfig::default()
    };
    assert_eq!(explicit.page_size_for("_0.vec", 1 << 30, budget), 32 << 10);
    let fixed = PagedFsConfig {
        adaptive: false,
        ..PagedFsConfig::default()
    };
    assert_eq!(
        fixed.page_size_for("_0.vec", 1 << 30, budget),
        fixed.page_size
    );
    assert!(!fixed.pins("_0.vex", 1, budget));
}

#[test]
fn vectors_only_maps_the_other_files() {
    let tmp = TempDir::new("paged-fs-vectors-only");
    let dir = PagedFsDirectory::with_config(tmp.path(), PagedFsConfig::vectors_only());
    let bytes = pattern(1 << 20);
    write(&dir, "_0.doc", &bytes);
    write(&dir, "_0_X_0.vec", &bytes);
    let doc = dir.open("_0.doc").unwrap();
    let vec = dir.open("_0_X_0.vec").unwrap();
    assert!(matches!(doc, Input::Mapped(_)));
    if dir.is_paged() {
        assert!(crate::is_prefetchable(&vec));
        assert!(!crate::is_prefetchable(&doc));
    }
    assert_eq!(&doc[..], &bytes[..]);
    assert_eq!(&vec[..], &bytes[..]);
}

#[test]
fn buffered_reads_match_too() {
    let tmp = TempDir::new("paged-fs-buffered");
    let dir = PagedFsDirectory::with_config(
        tmp.path(),
        PagedFsConfig {
            direct_io: false,
            page_size: 16 << 10,
            ..PagedFsConfig::default()
        },
    );
    let bytes = pattern(777_777);
    write(&dir, "b.dat", &bytes);
    assert_eq!(&dir.open("b.dat").unwrap()[..], &bytes[..]);
    assert_eq!(dir.read_at("b.dat", 1000, 50).unwrap(), &bytes[1000..1050]);
}

#[test]
fn aligned_files_get_whole_blocks_per_page() {
    if crate::paged_input::Pager::global().is_none() {
        return;
    }
    let tmp = TempDir::new("paged-fs-aligned");
    let dir = PagedFsDirectory::with_config(
        tmp.path(),
        PagedFsConfig {
            page_size: 4 << 10,
            adaptive: false,
            ..PagedFsConfig::default()
        },
    );
    let bytes = pattern(1 << 20);
    write(&dir, "_0.fdt", &bytes);
    let input = dir.open_aligned("_0.fdt", 16 << 10).unwrap();
    assert_eq!(&input[..], &bytes[..]);
    assert!(dir.open_aligned("_0.fdt", 12 << 10).is_err());
}

#[test]
fn open_file_survives_delete() {
    let tmp = TempDir::new("paged-fs-delete");
    let dir = PagedFsDirectory::open(tmp.path());
    let bytes = pattern(1 << 20);
    write(&dir, "d.dat", &bytes);
    let input = dir.open("d.dat").unwrap();
    dir.delete_file("d.dat").unwrap();
    assert_eq!(&input[..], &bytes[..]);
}

/// A test encoding: a 37-byte header (so units start unaligned), then
/// units of `UNIT` bytes XOR 0x5A followed by a 4-byte sum of them.
const UNIT: usize = 4096;
const HEADER: usize = 37;

fn sum(index: u64, plain: &[u8]) -> u32 {
    plain.iter().fold(index as u32, |a, &b| {
        a.wrapping_mul(31).wrapping_add(u32::from(b))
    })
}

fn encode(plain: &[u8]) -> Vec<u8> {
    let mut out = vec![7u8; HEADER];
    for (i, unit) in plain.chunks(UNIT).enumerate() {
        out.extend(unit.iter().map(|b| b ^ 0x5A));
        out.extend_from_slice(&sum(i as u64, unit).to_le_bytes());
    }
    out
}

struct SumCodec {
    plain_len: u64,
}

impl crate::page_cache::PageCodec for SumCodec {
    fn layout(&self) -> crate::page_cache::UnitLayout {
        crate::page_cache::UnitLayout {
            origin: HEADER as u64,
            stored_unit: UNIT as u64 + 4,
            plain_unit: UNIT as u64,
            plain_len: self.plain_len,
        }
    }

    fn decode(&self, first_unit: u64, stored: &[u8], out: &mut [u8]) -> Result<()> {
        let mut at = 0;
        for (i, unit) in stored.chunks(UNIT + 4).enumerate() {
            let n = unit.len() - 4;
            let plain = &mut out[at..at + n];
            for (o, b) in plain.iter_mut().zip(&unit[..n]) {
                *o = b ^ 0x5A;
            }
            let want = u32::from_le_bytes(unit[n..].try_into().unwrap());
            if sum(first_unit + i as u64, plain) != want {
                return Err(crate::error::Error::Corrupted(format!(
                    "unit {} failed its check",
                    first_unit + i as u64
                )));
            }
            at += n;
        }
        assert_eq!(at, out.len(), "the stored units decode to exactly `out`");
        Ok(())
    }
}

struct SumDecoder;

impl PageDecoder for SumDecoder {
    fn codec(
        &self,
        name: &str,
        stored_len: u64,
        dir: &dyn Directory,
    ) -> Result<std::sync::Arc<dyn crate::page_cache::PageCodec>> {
        assert_eq!(dir.read_at(name, 0, HEADER)?, vec![7u8; HEADER]);
        let body = stored_len - HEADER as u64;
        let units = body.div_ceil(UNIT as u64 + 4);
        Ok(std::sync::Arc::new(SumCodec {
            plain_len: body - 4 * units,
        }))
    }
}

fn decoded_dir(tmp: &TempDir, page: usize, pinned: bool) -> PagedFsDirectory {
    PagedFsDirectory::with_config(
        tmp.path(),
        PagedFsConfig {
            page_size: page,
            adaptive: false,
            pinned_extensions: if pinned {
                vec!["fdt".into()]
            } else {
                Vec::new()
            },
            ..PagedFsConfig::default()
        },
    )
}

/// The bytes read at open for the decoder's header hold the first
/// page: it is installed then, not read again. A first page that does
/// not decode is left to the pager, whose reads of it fail.
#[test]
fn a_decoded_open_installs_the_first_page_it_read() {
    if crate::paged_input::Pager::global().is_none() {
        return;
    }
    let tmp = TempDir::new("paged-fs-decoded-head");
    let dir = decoded_dir(&tmp, UNIT, false);
    let plain = pattern(40 * UNIT);
    write(&dir, "_h.fdt", &encode(&plain));
    let input = dir.open_decoded("_h.fdt", &SumDecoder).unwrap().unwrap();
    assert!(crate::is_resident(&input[..1]));
    assert!(!crate::is_resident(&input[20 * UNIT..20 * UNIT + 1]));
    assert!(input[..] == plain[..]);
    let mut raw = encode(&plain);
    raw[HEADER + 100] ^= 1;
    write(&dir, "_b.fdt", &raw);
    let bad = dir.open_decoded("_b.fdt", &SumDecoder).unwrap().unwrap();
    assert!(!crate::is_resident(&bad[..1]));
    assert!(bad.will_need(0, 1).is_err());
}

/// Decoded files read back through the pager at every tail length and
/// with pages of several units, pinned files read whole, and the
/// stored bytes are never what a reader sees.
#[test]
fn decoded_files_read_back_page_by_page() {
    if crate::paged_input::Pager::global().is_none() {
        return;
    }
    let tmp = TempDir::new("paged-fs-decoded");
    for (page, pinned) in [(UNIT, false), (3 * UNIT, false), (UNIT, true)] {
        let dir = decoded_dir(&tmp, page, pinned);
        for len in [0usize, 1, UNIT - 1, UNIT, UNIT + 1, 10 * UNIT + 5, 300_000] {
            let plain = pattern(len);
            let name = format!("_{len}_{page}_{pinned}.fdt");
            write(&dir, &name, &encode(&plain));
            let input = dir.open_decoded(&name, &SumDecoder).unwrap().unwrap();
            assert_eq!(input.len(), len);
            // Out of order first, then all of it.
            for at in (0..len).step_by(UNIT * 2 + 7).rev() {
                assert_eq!(input[at], plain[at], "{name}[{at}]");
            }
            assert!(input[..] == plain[..], "{name}");
            input.will_need(0, len).unwrap();
        }
    }
    // Where units are not whole OS pages the caller decodes instead.
    struct Odd;
    impl PageDecoder for Odd {
        fn codec(
            &self,
            _: &str,
            _: u64,
            _: &dyn Directory,
        ) -> Result<std::sync::Arc<dyn crate::page_cache::PageCodec>> {
            struct C;
            impl crate::page_cache::PageCodec for C {
                fn layout(&self) -> crate::page_cache::UnitLayout {
                    crate::page_cache::UnitLayout {
                        origin: 0,
                        stored_unit: 100,
                        plain_unit: 100,
                        plain_len: 1000,
                    }
                }
                fn decode(&self, _: u64, _: &[u8], _: &mut [u8]) -> Result<()> {
                    unreachable!()
                }
            }
            Ok(std::sync::Arc::new(C))
        }
    }
    let dir = decoded_dir(&tmp, UNIT, false);
    write(&dir, "odd.fdt", &[0u8; 1000]);
    assert!(dir.open_decoded("odd.fdt", &Odd).is_none());
}

/// A unit that fails its check fails announced reads with an error and
/// reads as zeros otherwise, recorded under the file's path until the
/// input is dropped; the process carries on. Truncation is reported the
/// same way.
#[test]
fn a_bad_unit_fails_reads_with_an_error_not_an_abort() {
    // A closed file's record goes once the pager's last hold on it
    // (a read in flight) ends.
    let cleared = |root: &str| {
        (0..200).any(|_| {
            let gone =
                crate::read_failures::failure_under(root, crate::read_failures::epoch()).is_none();
            if !gone {
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            gone
        })
    };
    if crate::paged_input::Pager::global().is_none() {
        return;
    }
    let tmp = TempDir::new("paged-fs-decoded-bad");
    let root = tmp.path().to_string_lossy().to_string();
    let dir = decoded_dir(&tmp, UNIT, false);
    let plain = pattern(20 * UNIT);
    write(&dir, "_1.fdt", &encode(&plain));
    let input = dir.open_decoded("_1.fdt", &SumDecoder).unwrap().unwrap();
    assert_eq!(input[3 * UNIT], plain[3 * UNIT]);
    assert_eq!(
        crate::read_failures::failure_under(&root, crate::read_failures::epoch()),
        None
    );
    // Corrupt unit 7 on disk under the open input.
    let path = tmp.path().join("_1.fdt");
    let mut raw = std::fs::read(&path).unwrap();
    raw[HEADER + 7 * (UNIT + 4) + 100] ^= 1;
    std::fs::write(&path, &raw).unwrap();
    assert_eq!(&input[7 * UNIT..7 * UNIT + 16], &[0u8; 16]);
    let recorded = crate::read_failures::failure_under(&root, crate::read_failures::epoch())
        .expect("recorded");
    assert!(recorded.contains("unit 7 failed its check"), "{recorded}");
    let err = input.will_need(0, 1).unwrap_err();
    assert!(matches!(err, crate::error::Error::Corrupted(_)), "{err}");
    assert!(crate::prefetch(&[&input[..1]]).is_err());
    // Other pages still read.
    assert_eq!(input[12 * UNIT + 9], plain[12 * UNIT + 9]);
    drop(input);
    assert!(cleared(&root));

    // A unit corrupt at open time fails a pinned (whole) read.
    let pinned = decoded_dir(&tmp, UNIT, true);
    assert!(matches!(
        pinned.open_decoded("_1.fdt", &SumDecoder).unwrap(),
        Err(crate::error::Error::Corrupted(_))
    ));
    // A file truncated under an open input: its missing units fail too.
    std::fs::write(&path, encode(&plain)).unwrap();
    let input = dir.open_decoded("_1.fdt", &SumDecoder).unwrap().unwrap();
    let f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
    f.set_len((HEADER + 15 * (UNIT + 4)) as u64).unwrap();
    assert!(input.will_need(17 * UNIT, 1).is_err());
    assert_eq!(input[18 * UNIT], 0);
    assert!(crate::read_failures::failure_under(&root, crate::read_failures::epoch()).is_some());
    drop(input);
    assert!(cleared(&root));
}

/// A container of three members: filler, a vector file starting on a page,
/// and a vector file starting off one.
fn container(dir: &dyn Directory) -> (Vec<u8>, u64, u64) {
    let vectors = pattern(300 << 10);
    let mut bytes = pattern(5000);
    bytes.resize(8192, 0);
    let vec_at = bytes.len() as u64;
    bytes.extend_from_slice(&vectors);
    write(dir, "_0.cfs", &bytes);
    (vectors, vec_at, vectors_len())
}

fn vectors_len() -> u64 {
    300 << 10
}

#[test]
fn a_vector_member_is_read_as_a_file_of_its_own() {
    let tmp = TempDir::new("paged-fs-member");
    let dir = PagedFsDirectory::with_config(tmp.path(), PagedFsConfig::vectors_only());
    let (vectors, at, len) = container(&dir);
    let whole = dir.open("_0.cfs").unwrap();
    // The archive itself is mapped under `vectors_only`; its vector member
    // is paged on its own, with the bytes the archive holds there.
    assert!(matches!(whole, Input::Mapped(_)));
    let member = dir.open_member("_0.cfs", at, len, "_0_X_0.vec").unwrap();
    if dir.is_paged() {
        let member = member.expect("a vector member is paged");
        assert!(crate::is_prefetchable(&member));
        assert_eq!(member.len() as u64, len);
        assert_eq!(&member[..], &vectors[..]);
        // The same member opened again shares its pages.
        let again = dir
            .open_member("_0.cfs", at, len, "_0_X_1.vec")
            .unwrap()
            .unwrap();
        assert_eq!(&again[10..20], &vectors[10..20]);
    } else {
        assert!(member.is_none());
    }
    // Not a vector file, not on a page, or not inside the container: the
    // caller slices the archive (or, inside it, is refused).
    assert!(dir
        .open_member("_0.cfs", at, len, "_0.fdt")
        .unwrap()
        .is_none());
    assert!(dir
        .open_member("_0.cfs", at + 1, len - 1, "_0_X_0.vec")
        .unwrap()
        .is_none());
    assert!(dir
        .open_member("_0.cfs", at, 0, "_0_X_0.vec")
        .unwrap()
        .is_none());
    if dir.is_paged() {
        assert!(dir
            .open_member("_0.cfs", at, len + 4096, "_0_X_0.vec")
            .is_err());
    }
}

#[test]
fn a_member_gets_the_page_size_and_residency_of_its_kind() {
    let tmp = TempDir::new("paged-fs-member-kind");
    let dir = PagedFsDirectory::open(tmp.path());
    let (vectors, at, len) = container(&dir);
    if !dir.is_paged() {
        return;
    }
    // A graph member small against the budget is read whole, as a loose
    // `.vex` is; the same bytes as a `.vec` are demand paged.
    let budget = crate::paged_input::Pager::global().unwrap().budget();
    assert!(dir.config.pins("_0.vex", len, budget) || len > budget / 8);
    let vec = dir
        .open_member("_0.cfs", at, len, "_0_X_0.vec")
        .unwrap()
        .unwrap();
    assert_eq!(&vec[..], &vectors[..]);
    let vex = dir
        .open_member("_0.cfs", at, len, "_0_X_0.vex")
        .unwrap()
        .unwrap();
    assert_eq!(&vex[..], &vectors[..]);
}
