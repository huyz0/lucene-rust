//! The pager's `SIGBUS` handler (faults served on the faulting thread)
//! passes every other `SIGBUS` to the handler installed before it, and
//! takes faults back when another handler replaces it. One process, so
//! its handlers are this test's alone.
#![cfg(target_os = "linux")]
#![allow(clippy::arithmetic_side_effects)]

use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};

use lucene_store::{Directory, PagedFsConfig, PagedFsDirectory};

static EARLIER: AtomicUsize = AtomicUsize::new(0);
static LATER: AtomicUsize = AtomicUsize::new(0);

/// Maps an anonymous page over the faulting page, so the access runs again
/// and reads zeros.
fn patch(info: *mut libc::siginfo_t) {
    // SAFETY: a SA_SIGINFO handler's `info` is valid; the address is
    // rounded down to its page and remapped in place.
    unsafe {
        let addr = ((*info).si_addr() as usize) & !4095;
        libc::mmap(
            addr as *mut libc::c_void,
            4096,
            libc::PROT_READ,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_FIXED,
            -1,
            0,
        );
    }
}

extern "C" fn earlier(_: libc::c_int, info: *mut libc::siginfo_t, _: *mut libc::c_void) {
    EARLIER.fetch_add(1, Ordering::SeqCst);
    patch(info);
}

extern "C" fn later(_: libc::c_int, info: *mut libc::siginfo_t, _: *mut libc::c_void) {
    LATER.fetch_add(1, Ordering::SeqCst);
    patch(info);
}

fn install(handler: extern "C" fn(libc::c_int, *mut libc::siginfo_t, *mut libc::c_void)) {
    // SAFETY: installs a SA_SIGINFO handler with an empty mask.
    unsafe {
        let mut action: libc::sigaction = std::mem::zeroed();
        action.sa_sigaction = handler as *const () as usize;
        action.sa_flags = libc::SA_SIGINFO;
        libc::sigemptyset(&mut action.sa_mask);
        assert_eq!(
            libc::sigaction(libc::SIGBUS, &action, std::ptr::null_mut()),
            0
        );
    }
}

/// A `SIGBUS` at an address no pager owns: a mapped page past the end of
/// a truncated file.
fn foreign_sigbus(dir: &std::path::Path) {
    let path = dir.join("truncated");
    std::fs::write(&path, vec![7u8; 8192]).unwrap();
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&path)
        .unwrap();
    use std::os::fd::AsRawFd;
    // SAFETY: maps the file's two pages; the mapping is leaked (the
    // handler remaps its second page).
    let base = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            8192,
            libc::PROT_READ,
            libc::MAP_SHARED,
            file.as_raw_fd(),
            0,
        )
    } as *const u8;
    assert_ne!(base as *mut libc::c_void, libc::MAP_FAILED);
    file.set_len(100).unwrap();
    // SAFETY: the second page lies past the file's end: the read raises
    // SIGBUS, which a handler resolves by mapping zeros there.
    let v = unsafe { std::ptr::read_volatile(base.add(4096)) };
    assert_eq!(v, 0);
}

fn paged_file(dir: &std::path::Path, name: &str) -> Vec<u8> {
    let bytes: Vec<u8> = (0..(1usize << 20)).map(|i| (i * 31 % 251) as u8).collect();
    let mut f = std::fs::File::create(dir.join(name)).unwrap();
    f.write_all(&bytes).unwrap();
    f.sync_all().unwrap();
    bytes
}

#[test]
fn foreign_faults_reach_the_earlier_handler_and_pager_faults_survive_a_later_one() {
    let tmp = lucene_util::test_support::TempDir::new("pager-signal-chain");
    install(earlier);
    let Some(pager) = lucene_store::paged_input::Pager::global() else {
        return;
    };
    if !pager.serves_on_thread() {
        return;
    }
    let config = PagedFsConfig {
        page_size: 16 << 10,
        adaptive: false,
        readahead: 0,
        pinned_extensions: Vec::new(),
        ..PagedFsConfig::default()
    };
    let dir = PagedFsDirectory::with_config(tmp.path(), config);
    let want = paged_file(tmp.path(), "a.doc");
    let a = dir.open("a.doc").unwrap();
    assert!(a[..] == want[..], "the pager served a.doc");

    foreign_sigbus(tmp.path());
    assert_eq!(
        EARLIER.load(Ordering::SeqCst),
        1,
        "chained to the earlier handler"
    );

    // Another library installs its handler over the pager's: the next open
    // puts the pager's back in front, chaining that one.
    install(later);
    let want_b = paged_file(tmp.path(), "b.doc");
    let b = dir.open("b.doc").unwrap();
    assert!(b[..] == want_b[..], "the pager served b.doc");
    assert_eq!(
        LATER.load(Ordering::SeqCst),
        0,
        "pager faults stay the pager's"
    );
    foreign_sigbus(tmp.path());
    assert_eq!(
        LATER.load(Ordering::SeqCst),
        1,
        "chained to the later handler"
    );
    assert_eq!(EARLIER.load(Ordering::SeqCst), 1);
}
