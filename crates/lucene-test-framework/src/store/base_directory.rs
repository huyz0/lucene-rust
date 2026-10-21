//! `BaseDirectoryTestCase`: the contract every `Directory` implementation
//! must keep, run over `get_directory(path)`.
//!
//! Each Java scenario is a method of the same name. An input here is a byte
//! buffer read through [`SliceInput`], so the read-side scenarios check the
//! directory's bytes and `SliceInput`'s cursor (`seek`, `slice_input`,
//! EOF) over them; `RandomAccessInput` reads are little-endian reads at an
//! offset of the buffer.
//!
//! Not ported, because the Rust shape makes them unrepresentable:
//! `testDetectClose` (a directory has no `close`; a dropped one cannot be
//! used), `testDoubleCloseOutput`/`testDoubleCloseInput` (closing consumes
//! the output; an input is dropped), the immutability checks of
//! `testSetOfStrings`/`testMapOfStrings` (the reads return owned values),
//! `testGroupVIntOverflow`'s `ArithmeticException` (group vints are `u32`
//! by signature), and `testIsLoaded`/`testIsLoadedOnSlice` (no `isLoaded`).
//!
//! Differs: an out-of-bounds `slice` is an EOF error rather than an
//! `IllegalArgumentException`, and a negative offset is unrepresentable;
//! `testPendingDeletions` installs its virus checker through
//! [`BaseDirectoryTestCase::with_fs_faults`] (Java wraps the path's
//! filesystem) and is skipped without one, as Java skips a directory that
//! is not an `FSDirectory`.

use std::collections::BTreeSet;
use std::io;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier};

use lucene_search::directory_reader::DirectoryReader;
use lucene_store::directory::FsFaults;
use lucene_store::{DataInput, DataOutput, Directory, Error, IndexOutput, SliceInput};
use lucene_util::java_random::JavaRandom;
use lucene_util::test_support::TempDir;

use crate::mockfile::mock_fs::{MockFs, MockFsConfig};
use crate::util::lucene_test_case::LuceneTestCase;
use crate::util::random_numbers::random_int_between;
use crate::util::test_util::{
    next_int, next_long, random_simple_string, random_simple_string_between,
};

type FaultedDirectory<D> = Box<dyn Fn(&Path, Arc<dyn FsFaults>) -> D>;

/// `BaseDirectoryTestCase`, over `get_directory(path)`.
pub struct BaseDirectoryTestCase<D, F> {
    get_directory: F,
    with_faults: Option<FaultedDirectory<D>>,
}

/// The `io::ErrorKind` of an I/O error.
fn io_kind(e: &Error) -> Option<io::ErrorKind> {
    match e {
        Error::Io(e) => Some(e.kind()),
        _ => None,
    }
}

fn is_eof(e: &Error) -> bool {
    matches!(e, Error::Eof { .. })
}

/// `IndexFileNames.segmentFileName(segment, suffix, ext)`.
fn segment_file_name(segment: &str, suffix: &str, ext: &str) -> String {
    if suffix.is_empty() {
        format!("{segment}.{ext}")
    } else {
        format!("{segment}_{suffix}.{ext}")
    }
}

fn random_bytes(r: &mut JavaRandom, len: usize) -> Vec<u8> {
    let mut bytes = vec![0u8; len];
    r.next_bytes(&mut bytes);
    bytes
}

/// `value(idx)` of `testCopyBytes`.
fn value(idx: i32) -> u8 {
    ((idx % 256) * (1 + idx / 256)) as u8
}

fn write_file(dir: &dyn Directory, name: &str, bytes: &[u8]) {
    let mut out = dir.create_output(name).unwrap();
    out.write_bytes(bytes);
    out.close().unwrap();
}

impl<D: Directory, F: Fn(&Path) -> D> BaseDirectoryTestCase<D, F> {
    /// The suite over `get_directory`.
    pub fn new(get_directory: F) -> Self {
        Self {
            get_directory,
            with_faults: None,
        }
    }

    /// How to build the directory over a filesystem with `faults` under it
    /// (Java's `getDirectory(addVirusChecker(path))`): what
    /// `testPendingDeletions` needs.
    pub fn with_fs_faults(mut self, f: impl Fn(&Path, Arc<dyn FsFaults>) -> D + 'static) -> Self {
        self.with_faults = Some(Box::new(f));
        self
    }

    fn dir(&self, temp: &TempDir) -> D {
        (self.get_directory)(temp.path())
    }

    /// `testCopyFrom`: both directions between this directory and a
    /// `newDirectory()`.
    pub fn test_copy_from(&self, tc: &mut LuceneTestCase) {
        let temp = TempDir::new("testCopy");
        let source = self.dir(&temp);
        let dest = tc.new_directory();
        Self::run_copy_from(tc, &source, dest.as_ref());
        let source = tc.new_directory();
        let temp = TempDir::new("testCopyDestination");
        let dest = self.dir(&temp);
        Self::run_copy_from(tc, source.as_ref(), &dest);
    }

    fn run_copy_from(tc: &mut LuceneTestCase, source: &dyn Directory, dest: &dyn Directory) {
        let bytes = random_bytes(tc.random(), 20000);
        write_file(source, "foobar", &bytes);
        dest.copy_from(source, "foobar", "foobaz").unwrap();
        assert!(dest.list_all().unwrap().iter().any(|f| f == "foobaz"));
        assert_eq!(&*dest.open("foobaz").unwrap(), &bytes[..]);
    }

    /// `testRename`.
    pub fn test_rename(&self, tc: &mut LuceneTestCase) {
        let temp = TempDir::new("testRename");
        let dir = self.dir(&temp);
        let num_bytes = tc.random().next_int_bounded(20000) as usize;
        let bytes = random_bytes(tc.random(), num_bytes);
        write_file(&dir, "foobar", &bytes);
        dir.rename("foobar", "foobaz").unwrap();
        let input = dir.open("foobaz").unwrap();
        assert_eq!(input.len(), num_bytes);
        assert_eq!(&*input, &bytes[..]);
    }

    /// `testDeleteFile`: deleting a missing file is `NoSuchFile`.
    pub fn test_delete_file(&self, _tc: &mut LuceneTestCase) {
        let temp = TempDir::new("testDeleteFile");
        let dir = self.dir(&temp);
        let file = "foo.txt".to_string();
        assert!(!dir.list_all().unwrap().contains(&file));
        dir.create_output(&file).unwrap().close().unwrap();
        assert!(dir.list_all().unwrap().contains(&file));
        dir.delete_file(&file).unwrap();
        assert!(!dir.list_all().unwrap().contains(&file));
        let e = dir.delete_file(&file).unwrap_err();
        assert_eq!(io_kind(&e), Some(io::ErrorKind::NotFound), "{e}");
    }

    /// One value written, then read back from a file of `len` bytes.
    fn round_trip(
        &self,
        name: &str,
        len: usize,
        write: impl FnOnce(&mut dyn DataOutput),
        read: impl FnOnce(&mut SliceInput<'_>),
    ) {
        let temp = TempDir::new(name);
        let dir = self.dir(&temp);
        let mut out = dir.create_output(name).unwrap();
        write(&mut out);
        out.close().unwrap();
        let input = dir.open(name).unwrap();
        assert_eq!(input.len(), len);
        let mut input = SliceInput::new(&input);
        read(&mut input);
        assert_eq!(input.position(), len);
    }

    /// `testByte`.
    pub fn test_byte(&self, _tc: &mut LuceneTestCase) {
        self.round_trip(
            "byte",
            1,
            |o| o.write_byte(128),
            |i| {
                assert_eq!(i.read_byte().unwrap(), 128);
            },
        );
    }

    /// `testShort`.
    pub fn test_short(&self, _tc: &mut LuceneTestCase) {
        self.round_trip(
            "short",
            2,
            |o| o.write_i16(-20),
            |i| {
                assert_eq!(i.read_i16().unwrap(), -20);
            },
        );
    }

    /// `testInt`.
    pub fn test_int(&self, _tc: &mut LuceneTestCase) {
        self.round_trip(
            "int",
            4,
            |o| o.write_i32(-500),
            |i| {
                assert_eq!(i.read_i32().unwrap(), -500);
            },
        );
    }

    /// `testLong`.
    pub fn test_long(&self, _tc: &mut LuceneTestCase) {
        self.round_trip(
            "long",
            8,
            |o| o.write_i64(-5000),
            |i| {
                assert_eq!(i.read_i64().unwrap(), -5000);
            },
        );
    }

    /// `testAlignedLittleEndianLongs` and `testUnalignedLittleEndianLongs`
    /// (`padding` leading bytes).
    fn little_endian_longs(&self, name: &str, padding: usize) {
        self.round_trip(
            name,
            24 + padding,
            |o| {
                for _ in 0..padding {
                    o.write_byte(2);
                }
                o.write_i64(3);
                o.write_i64(i64::MAX);
                o.write_i64(-3);
            },
            |i| {
                for _ in 0..padding {
                    assert_eq!(i.read_byte().unwrap(), 2);
                }
                let mut l = [0i64; 4];
                i.read_i64s(&mut l[1..]).unwrap();
                assert_eq!(l, [0, 3, i64::MAX, -3]);
            },
        );
    }

    /// `testAlignedLittleEndianLongs`.
    pub fn test_aligned_little_endian_longs(&self, _tc: &mut LuceneTestCase) {
        self.little_endian_longs("littleEndianLongs", 0);
    }

    /// `testUnalignedLittleEndianLongs`.
    pub fn test_unaligned_little_endian_longs(&self, _tc: &mut LuceneTestCase) {
        self.little_endian_longs("littleEndianLongs", 1);
    }

    /// The `*Underflow` scenarios: `length` values of `width` bytes from
    /// `offset`, in a file too short for them, are an EOF.
    fn underflow(
        &self,
        tc: &mut LuceneTestCase,
        name: &str,
        width: i32,
        read: impl FnOnce(&mut SliceInput<'_>, usize) -> lucene_store::Result<()>,
    ) {
        let temp = TempDir::new(name);
        let dir = self.dir(&temp);
        let offset = tc.random().next_int_bounded(width);
        let length = next_int(tc.random(), 1, 16);
        let short = next_int(tc.random(), 1, width);
        let b = random_bytes(tc.random(), (offset + length * width - short) as usize);
        write_file(&dir, name, &b);
        let input = dir.open(name).unwrap();
        let mut input = SliceInput::new(&input);
        input.seek(offset as usize).unwrap();
        let e = read(&mut input, length as usize).unwrap_err();
        assert!(is_eof(&e), "{e}");
    }

    /// `testLittleEndianLongsUnderflow`.
    pub fn test_little_endian_longs_underflow(&self, tc: &mut LuceneTestCase) {
        self.underflow(tc, "littleEndianLongs", 8, |i, n| {
            i.read_i64s(&mut vec![0; n])
        });
    }

    /// `testAlignedInts` and `testUnalignedInts`; `floats` reads them as
    /// `float` bits (`testAlignedFloats`, `testUnalignedFloats`).
    fn ints(&self, name: &str, padding: usize, floats: bool) {
        let values: [u32; 3] = if floats {
            [3f32.to_bits(), f32::MAX.to_bits(), (-3f32).to_bits()]
        } else {
            [3, i32::MAX as u32, -3i32 as u32]
        };
        self.round_trip(
            name,
            12 + padding,
            |o| {
                for _ in 0..padding {
                    o.write_byte(2);
                }
                for v in values {
                    o.write_i32(v as i32);
                }
            },
            |i| {
                for _ in 0..padding {
                    assert_eq!(i.read_byte().unwrap(), 2);
                }
                let mut read = [0u32; 4];
                i.read_u32s_le(&mut read[1..]).unwrap();
                if floats {
                    let ff = read.map(f32::from_bits);
                    assert_eq!(ff, [0.0, 3.0, f32::MAX, -3.0]);
                } else {
                    assert_eq!(read.map(|v| v as i32), [0, 3, i32::MAX, -3]);
                }
            },
        );
    }

    /// `testAlignedInts`.
    pub fn test_aligned_ints(&self, _tc: &mut LuceneTestCase) {
        self.ints("Ints", 0, false);
    }

    /// `testUnalignedInts`.
    pub fn test_unaligned_ints(&self, tc: &mut LuceneTestCase) {
        let padding = tc.random().next_int_bounded(3) as usize + 1;
        self.ints("Ints", padding, false);
    }

    /// `testIntsUnderflow`.
    pub fn test_ints_underflow(&self, tc: &mut LuceneTestCase) {
        self.underflow(tc, "Ints", 4, |i, n| i.read_u32s_le(&mut vec![0; n]));
    }

    /// `testAlignedFloats`.
    pub fn test_aligned_floats(&self, _tc: &mut LuceneTestCase) {
        self.ints("Floats", 0, true);
    }

    /// `testUnalignedFloats`.
    pub fn test_unaligned_floats(&self, tc: &mut LuceneTestCase) {
        let padding = tc.random().next_int_bounded(3) as usize + 1;
        self.ints("Floats", padding, true);
    }

    /// `testFloatsUnderflow`.
    pub fn test_floats_underflow(&self, tc: &mut LuceneTestCase) {
        self.underflow(tc, "Floats", 4, |i, n| i.read_u32s_le(&mut vec![0; n]));
    }

    /// `testString`.
    pub fn test_string(&self, _tc: &mut LuceneTestCase) {
        self.round_trip(
            "string",
            7,
            |o| o.write_string("hello!"),
            |i| {
                assert_eq!(i.read_string().unwrap(), "hello!");
            },
        );
    }

    /// `testVInt`.
    pub fn test_vint(&self, _tc: &mut LuceneTestCase) {
        self.round_trip(
            "vint",
            2,
            |o| o.write_vint(500),
            |i| {
                assert_eq!(i.read_vint().unwrap(), 500);
            },
        );
    }

    /// `testVLong`.
    pub fn test_vlong(&self, _tc: &mut LuceneTestCase) {
        self.round_trip(
            "vlong",
            9,
            |o| o.write_vlong(i64::MAX),
            |i| {
                assert_eq!(i.read_vlong().unwrap(), i64::MAX);
            },
        );
    }

    /// Writes `values` with `write`, then reads them back to the end.
    fn round_trip_all<T: Copy + PartialEq + std::fmt::Debug>(
        &self,
        name: &str,
        values: &[T],
        write: impl Fn(&mut dyn DataOutput, T),
        read: impl Fn(&mut SliceInput<'_>) -> T,
    ) {
        let temp = TempDir::new(name);
        let dir = self.dir(&temp);
        let mut out = dir.create_output(name).unwrap();
        for &v in values {
            write(&mut out, v);
        }
        out.close().unwrap();
        let input = dir.open(name).unwrap();
        let mut input = SliceInput::new(&input);
        for &v in values {
            assert_eq!(read(&mut input), v);
        }
        assert_eq!(input.position(), input.len());
    }

    /// `testZInt`.
    pub fn test_zint(&self, tc: &mut LuceneTestCase) {
        let r = tc.random();
        let n = r.next_int_bounded(10) as usize;
        let mut ints = Vec::with_capacity(n);
        for _ in 0..n {
            ints.push(match r.next_int_bounded(3) {
                0 => r.next_int(),
                1 => {
                    if r.next_boolean() {
                        i32::MIN
                    } else {
                        i32::MAX
                    }
                }
                _ => (if r.next_boolean() { -1 } else { 1 }) * r.next_int_bounded(1024),
            });
        }
        self.round_trip_all(
            "zint",
            &ints,
            |o, v| o.write_zint(v),
            |i| i.read_zint().unwrap(),
        );
    }

    /// `testZLong`.
    pub fn test_zlong(&self, tc: &mut LuceneTestCase) {
        let r = tc.random();
        let n = r.next_int_bounded(10) as usize;
        let mut longs = Vec::with_capacity(n);
        for _ in 0..n {
            longs.push(match r.next_int_bounded(3) {
                0 => r.next_long(),
                1 => {
                    if r.next_boolean() {
                        i64::MIN
                    } else {
                        i64::MAX
                    }
                }
                _ => (if r.next_boolean() { -1 } else { 1 }) * i64::from(r.next_int_bounded(1024)),
            });
        }
        self.round_trip_all(
            "zlong",
            &longs,
            |o, v| o.write_zlong(v),
            |i| i.read_zlong().unwrap(),
        );
    }

    /// `testSetOfStrings`.
    pub fn test_set_of_strings(&self, _tc: &mut LuceneTestCase) {
        let sets: [Vec<String>; 3] = [
            vec!["test1".into(), "test2".into()],
            vec![],
            vec!["test3".into()],
        ];
        let temp = TempDir::new("testSetOfStrings");
        let dir = self.dir(&temp);
        let mut out = dir.create_output("stringset").unwrap();
        for set in &sets {
            out.write_set_of_strings(set);
        }
        out.close().unwrap();
        let input = dir.open("stringset").unwrap();
        let mut input = SliceInput::new(&input);
        for set in &sets {
            let got: BTreeSet<String> = input.read_set_of_strings().unwrap().into_iter().collect();
            assert_eq!(got, set.iter().cloned().collect());
        }
        assert_eq!(input.position(), input.len());
    }

    /// `testMapOfStrings`.
    pub fn test_map_of_strings(&self, _tc: &mut LuceneTestCase) {
        let pair = |k: &str, v: &str| (k.to_string(), v.to_string());
        let maps = [
            vec![pair("test1", "value1"), pair("test2", "value2")],
            vec![],
            vec![pair("key", "value")],
        ];
        let temp = TempDir::new("testMapOfStrings");
        let dir = self.dir(&temp);
        let mut out = dir.create_output("stringmap").unwrap();
        for map in &maps {
            out.write_map_of_strings(map);
        }
        out.close().unwrap();
        let input = dir.open("stringmap").unwrap();
        let mut input = SliceInput::new(&input);
        for map in &maps {
            let mut got = input.read_map_of_strings().unwrap();
            got.sort();
            assert_eq!(&got, map);
        }
        assert_eq!(input.position(), input.len());
    }

    /// `testChecksum`: the output's running CRC and the file's agree with
    /// a CRC32 of the bytes.
    pub fn test_checksum(&self, tc: &mut LuceneTestCase) {
        let num_bytes = tc.random().next_int_bounded(20000) as usize;
        let bytes = random_bytes(tc.random(), num_bytes);
        let expected = u64::from(crc32fast::hash(&bytes));
        let temp = TempDir::new("testChecksum");
        let dir = self.dir(&temp);
        let mut out = dir.create_output("checksum").unwrap();
        out.write_bytes(&bytes);
        assert_eq!(out.checksum(), expected);
        out.close().unwrap();
        let input = dir.open("checksum").unwrap();
        assert_eq!(u64::from(crc32fast::hash(&input)), expected);
    }

    /// `testThreadSafetyInListAll`: a writer creates files while a reader
    /// lists and opens them; a file still being written may refuse to open
    /// (`AccessDenied`), nothing else may fail.
    pub fn test_thread_safety_in_list_all(&self, tc: &mut LuceneTestCase) {
        let temp = TempDir::new("testThreadSafety");
        let dir = self.dir(&temp);
        let max = random_int_between(tc.random(), 500, 1000);
        let mut rnd = JavaRandom::new(tc.random().next_long());
        let stop = AtomicBool::new(false);
        std::thread::scope(|s| {
            s.spawn(|| {
                // Java's `finally`: a panicking writer still stops the reader.
                struct Stop<'a>(&'a AtomicBool);
                impl Drop for Stop<'_> {
                    fn drop(&mut self) {
                        self.0.store(true, Ordering::SeqCst);
                    }
                }
                let _stop = Stop(&stop);
                for i in 0..max {
                    let file_name = format!("file-{i}");
                    let out = dir.create_output(&file_name);
                    std::thread::yield_now();
                    out.unwrap().close().unwrap();
                    assert!(dir.list_all().unwrap().contains(&file_name));
                }
            });
            s.spawn(|| {
                while !stop.load(Ordering::SeqCst) {
                    let files: Vec<String> = dir
                        .list_all()
                        .unwrap()
                        .into_iter()
                        .filter(|f| !MockFs::is_extra(f))
                        .collect();
                    if files.is_empty() {
                        continue;
                    }
                    loop {
                        let file = &files[rnd.next_int_bounded(files.len() as i32) as usize];
                        match dir.open(file) {
                            Ok(input) => drop(input),
                            Err(e) if io_kind(&e) == Some(io::ErrorKind::PermissionDenied) => {}
                            Err(e) => panic!("Something went wrong when opening: {file}: {e}"),
                        }
                        if rnd.next_int_bounded(3) == 0 {
                            break;
                        }
                    }
                }
            });
        });
    }

    /// `testFileExistsInListAfterCreated` (LUCENE-1468).
    pub fn test_file_exists_in_list_after_created(&self, _tc: &mut LuceneTestCase) {
        let temp = TempDir::new("testFileExistsInListAfterCreated");
        let dir = self.dir(&temp);
        dir.create_output("file").unwrap().close().unwrap();
        assert!(dir.list_all().unwrap().contains(&"file".to_string()));
    }

    /// `testSeekToEOFThenBack` (LUCENE-2852).
    pub fn test_seek_to_eof_then_back(&self, _tc: &mut LuceneTestCase) {
        let temp = TempDir::new("testSeekToEOFThenBack");
        let dir = self.dir(&temp);
        let buffer_length = 1024;
        let mut bytes = vec![0u8; 3 * buffer_length];
        write_file(&dir, "out", &bytes);
        let input = dir.open("out").unwrap();
        let mut i = SliceInput::new(&input);
        i.seek(2 * buffer_length - 1).unwrap();
        i.seek(3 * buffer_length).unwrap();
        i.seek(buffer_length).unwrap();
        i.read_bytes(&mut bytes[..2 * buffer_length]).unwrap();
    }

    /// `testIllegalEOF` (LUCENE-1196).
    pub fn test_illegal_eof(&self, _tc: &mut LuceneTestCase) {
        let temp = TempDir::new("testIllegalEOF");
        let dir = self.dir(&temp);
        write_file(&dir, "out", &[0u8; 1024]);
        let input = dir.open("out").unwrap();
        SliceInput::new(&input).seek(1024).unwrap();
    }

    /// `testSeekPastEOF`: past the end is an EOF; to the end is not, but
    /// reading there is.
    pub fn test_seek_past_eof(&self, tc: &mut LuceneTestCase) {
        let temp = TempDir::new("testSeekPastEOF");
        let dir = self.dir(&temp);
        let len = tc.random().next_int_bounded(2048) as usize;
        write_file(&dir, "out", &vec![0u8; len]);
        let input = dir.open("out").unwrap();
        let mut i = SliceInput::new(&input);
        let past = len + random_int_between(tc.random(), 1, 2048) as usize;
        assert!(is_eof(&i.seek(past).unwrap_err()));
        i.seek(len).unwrap();
        assert!(is_eof(&i.read_byte().unwrap_err()));
        assert!(is_eof(&i.read_bytes(&mut [0u8; 1]).unwrap_err()));
    }

    /// `testSliceOutOfBounds`.
    pub fn test_slice_out_of_bounds(&self, tc: &mut LuceneTestCase) {
        let temp = TempDir::new("testSliceOutOfBounds");
        let dir = self.dir(&temp);
        let len = tc.random().next_int_bounded(2040) as u64 + 8;
        write_file(&dir, "out", &vec![0u8; len as usize]);
        let input = dir.open("out").unwrap();
        let i = SliceInput::new(&input);
        assert!(i.slice_input("slice1", 0, len + 1).is_err());
        let slice = i.slice_input("slice3", 4, len / 2).unwrap();
        assert!(slice.slice_input("slice3sub", 1, len / 2).is_err());
        assert!(i.slice_input("slice4", i64::MAX as u64 - 1, 10).is_err());
        assert!(i.slice_input("slice5", u64::MAX, 10).is_err());
    }

    /// `testNoDir` (LUCENE-3382): opening a reader over a directory that
    /// does not exist fails (`NoSuchFile` or `IndexNotFound`).
    pub fn test_no_dir(&self, _tc: &mut LuceneTestCase) {
        let temp = TempDir::new("doesnotexist");
        let path = temp.path().join("gone");
        let dir = (self.get_directory)(&path);
        assert!(DirectoryReader::open(&dir).is_err());
    }

    /// `testCopyBytes`: a file copied byte by byte and in chunks reads back
    /// the same way.
    pub fn test_copy_bytes(&self, tc: &mut LuceneTestCase) {
        let temp = TempDir::new("testCopyBytes");
        let dir = self.dir(&temp);
        let r = tc.random();
        let mut out = dir.create_output("test").unwrap();
        let mut bytes = vec![0u8; next_int(r, 1, 77777) as usize];
        let size = next_int(r, 1, 1777777);
        let mut byte_upto = 0;
        for upto in 0..size {
            bytes[byte_upto] = value(upto);
            byte_upto += 1;
            if byte_upto == bytes.len() {
                out.write_bytes(&bytes);
                byte_upto = 0;
            }
        }
        out.write_bytes(&bytes[..byte_upto]);
        assert_eq!(out.file_pointer(), size as u64);
        out.close().unwrap();
        assert_eq!(dir.file_length("test").unwrap(), size as u64);

        // copy from test -> test2
        let input = dir.open("test").unwrap();
        let mut input = SliceInput::new(&input);
        let mut out = dir.create_output("test2").unwrap();
        let mut upto = 0;
        while upto < size {
            if r.next_boolean() {
                out.write_byte(input.read_byte().unwrap());
                upto += 1;
            } else {
                let chunk = next_int(r, 1, bytes.len() as i32).min(size - upto);
                let mut buf = vec![0u8; chunk as usize];
                input.read_bytes(&mut buf).unwrap();
                out.write_bytes(&buf);
                upto += chunk;
            }
        }
        assert_eq!(upto, size);
        out.close().unwrap();

        // verify
        let in2 = dir.open("test2").unwrap();
        let mut in2 = SliceInput::new(&in2);
        let mut upto = 0;
        while upto < size {
            if r.next_boolean() {
                assert_eq!(in2.read_byte().unwrap(), value(upto));
                upto += 1;
            } else {
                let limit = next_int(r, 1, bytes.len() as i32).min(size - upto) as usize;
                in2.read_bytes(&mut bytes[..limit]).unwrap();
                for &b in &bytes[..limit] {
                    assert_eq!(b, value(upto));
                    upto += 1;
                }
            }
        }
        dir.delete_file("test").unwrap();
        dir.delete_file("test2").unwrap();
    }

    /// `testCopyBytesWithThreads` (LUCENE-3541): ten threads copy the tail
    /// of one open input at once.
    pub fn test_copy_bytes_with_threads(&self, tc: &mut LuceneTestCase) {
        let temp = TempDir::new("testCopyBytesWithThreads");
        let d = self.dir(&temp);
        let header_len = 100;
        let n = random_int_between(tc.random(), header_len as i32 + 1, 10000) as usize;
        let data = random_bytes(tc.random(), n);
        write_file(&d, "data", &data);
        let input = d.open("data").unwrap();
        write_file(&d, "header", &input[..header_len]);
        let threads = 10;
        let start = Barrier::new(threads);
        std::thread::scope(|s| {
            for i in 0..threads {
                let (d, start, src) = (&d, &start, &input[header_len..]);
                s.spawn(move || {
                    start.wait();
                    write_file(d, &format!("copy{i}"), src);
                });
            }
        });
        for i in 0..threads {
            let copied = d.open(&format!("copy{i}")).unwrap();
            let mut data_copy = data[..header_len].to_vec();
            data_copy.extend_from_slice(&copied);
            assert_eq!(data_copy, data);
        }
        assert_eq!(&*d.open("header").unwrap(), &data[..header_len]);
    }

    /// `testFsyncDoesntCreateNewFiles`: syncing a file deleted behind the
    /// directory's back fails and creates nothing. Skipped (as Java's
    /// `assumeTrue`) for a directory not on a filesystem.
    pub fn test_fsync_doesnt_create_new_files(&self, _tc: &mut LuceneTestCase) {
        let temp = TempDir::new("nocreate");
        let fsdir = self.dir(&temp);
        let Some(path) = fsdir.fs_directory_path().map(Path::to_path_buf) else {
            return;
        };
        let mut out = fsdir.create_output("afile").unwrap();
        out.write_string("boo");
        out.close().unwrap();
        std::fs::remove_file(path.join("afile")).unwrap();
        let file_count = fsdir.list_all().unwrap().len();
        let e = fsdir.sync(&["afile".to_string()]).unwrap_err();
        assert_eq!(io_kind(&e), Some(io::ErrorKind::NotFound), "{e}");
        assert_eq!(fsdir.list_all().unwrap().len(), file_count);
    }

    /// The random-access scenarios: `count` values of `width` bytes written
    /// by `write`, read back at every offset by `read`, from slices, sub
    /// slices and padded copies.
    fn random_access<T: Copy + PartialEq + std::fmt::Debug>(
        &self,
        tc: &mut LuceneTestCase,
        name: &str,
        values: &[T],
        width: usize,
        write: impl Fn(&mut dyn DataOutput, T),
        read: impl Fn(&[u8], usize) -> T,
    ) {
        let temp = TempDir::new(name);
        let dir = self.dir(&temp);
        let mut out = dir.create_output(name).unwrap();
        for &v in values {
            write(&mut out, v);
        }
        out.close().unwrap();
        let input = dir.open(name).unwrap();
        let slice = &input[..];
        assert_eq!(slice.len(), values.len() * width);
        for (i, &v) in values.iter().enumerate() {
            assert_eq!(read(slice, i * width), v);
        }
        for i in 1..values.len() {
            let subslice = &input[i * width..];
            assert_eq!(subslice.len(), input.len() - i * width);
            for (j, &v) in values[i..].iter().enumerate() {
                assert_eq!(read(subslice, j * width), v);
            }
        }
        for i in 0..7 {
            let padded_name = format!("{name}-{i}");
            let mut o = dir.create_output(&padded_name).unwrap();
            o.write_bytes(&random_bytes(tc.random(), i));
            o.write_bytes(&input);
            o.close().unwrap();
            let padded = dir.open(&padded_name).unwrap();
            let whole = &padded[i..];
            assert_eq!(whole.len(), padded.len() - i);
            for (j, &v) in values.iter().enumerate() {
                assert_eq!(read(whole, j * width), v);
            }
        }
    }

    /// `testRandomLong`.
    pub fn test_random_long(&self, tc: &mut LuceneTestCase) {
        let num = next_int(tc.random(), 50, 3000);
        let longs: Vec<i64> = (0..num)
            .map(|_| next_long(tc.random(), i64::MIN, i64::MAX))
            .collect();
        self.random_access(
            tc,
            "longs",
            &longs,
            8,
            |o, v| o.write_i64(v),
            |b, at| i64::from_le_bytes(b[at..at + 8].try_into().unwrap()),
        );
    }

    /// `testRandomInt`.
    pub fn test_random_int(&self, tc: &mut LuceneTestCase) {
        let num = next_int(tc.random(), 50, 3000);
        let ints: Vec<i32> = (0..num).map(|_| tc.random().next_int()).collect();
        self.random_access(
            tc,
            "ints",
            &ints,
            4,
            |o, v| o.write_i32(v),
            |b, at| i32::from_le_bytes(b[at..at + 4].try_into().unwrap()),
        );
    }

    /// `testRandomShort`.
    pub fn test_random_short(&self, tc: &mut LuceneTestCase) {
        let num = next_int(tc.random(), 50, 3000);
        let shorts: Vec<i16> = (0..num).map(|_| tc.random().next_int() as i16).collect();
        self.random_access(
            tc,
            "shorts",
            &shorts,
            2,
            |o, v| o.write_i16(v),
            |b, at| i16::from_le_bytes(b[at..at + 2].try_into().unwrap()),
        );
    }

    /// `testRandomByte`: every sub-range of every slice reads back.
    pub fn test_random_byte(&self, tc: &mut LuceneTestCase) {
        let num = if tc.nightly() {
            next_int(tc.random(), 1000, 3000)
        } else {
            next_int(tc.random(), 50, 1000)
        };
        let bytes = random_bytes(tc.random(), num as usize);
        let temp = TempDir::new("testBytes");
        let dir = self.dir(&temp);
        let mut out = dir.create_output("bytes").unwrap();
        for &b in &bytes {
            out.write_byte(b);
        }
        out.close().unwrap();
        let input = dir.open("bytes").unwrap();
        Self::assert_bytes(tc, &input, &bytes, 0);
        for offset in 1..bytes.len() {
            let subslice = &input[offset..];
            assert_eq!(subslice.len(), input.len() - offset);
            Self::assert_bytes(tc, subslice, &bytes, offset);
        }
        for i in 1..7 {
            let name = format!("bytes-{i}");
            let mut o = dir.create_output(&name).unwrap();
            o.write_bytes(&random_bytes(tc.random(), i));
            o.write_bytes(&input);
            o.close().unwrap();
            let padded = dir.open(&name).unwrap();
            Self::assert_bytes(tc, &padded[i..], &bytes, 0);
        }
    }

    /// `assertBytes(slice, bytes, bytesOffset)`.
    fn assert_bytes(tc: &mut LuceneTestCase, slice: &[u8], bytes: &[u8], bytes_offset: usize) {
        let to_read = bytes.len() - bytes_offset;
        for i in 0..to_read {
            assert_eq!(slice[i], bytes[bytes_offset + i]);
            // Java's random destination offset moves the stream: draw it.
            let _ = tc.random().next_int_bounded(1000);
            assert_eq!(&slice[..i], &bytes[bytes_offset..bytes_offset + i]);
            assert_eq!(&slice[i..to_read], &bytes[bytes_offset + i..]);
        }
    }

    /// `testSliceOfSlice`.
    pub fn test_slice_of_slice(&self, tc: &mut LuceneTestCase) {
        let num = if tc.nightly() {
            next_int(tc.random(), 250, 2500)
        } else {
            next_int(tc.random(), 50, 250)
        } as usize;
        let bytes = random_bytes(tc.random(), num);
        let temp = TempDir::new("sliceOfSlice");
        let dir = self.dir(&temp);
        write_file(&dir, "bytes", &bytes);
        let input = dir.open("bytes").unwrap();
        let mut input = SliceInput::new(&input);
        // seek to a random spot should not impact slicing.
        input
            .seek(next_long(tc.random(), 0, input.len() as i64) as usize)
            .unwrap();
        for i in (0..num).step_by(16) {
            let mut slice1 = input
                .slice_input("slice1", i as u64, (num - i) as u64)
                .unwrap();
            assert_eq!(slice1.position(), 0);
            assert_eq!(slice1.len(), num - i);
            slice1
                .seek(next_long(tc.random(), 0, slice1.len() as i64) as usize)
                .unwrap();
            for j in (0..slice1.len()).step_by(16) {
                let mut slice2 = slice1
                    .slice_input("slice2", j as u64, (num - i - j) as u64)
                    .unwrap();
                // clone shouldn't impact slice data: a copy of the cursor.
                let _ = tc.random().next_boolean();
                assert_eq!(slice2.position(), 0);
                assert_eq!(slice2.len(), num - i - j);
                let mut data = vec![0u8; num];
                data[..i + j].copy_from_slice(&bytes[..i + j]);
                if tc.random().next_boolean() {
                    slice2.read_bytes(&mut data[i + j..]).unwrap();
                } else {
                    let seek = next_long(tc.random(), 0, slice2.len() as i64) as usize;
                    slice2.seek(seek).unwrap();
                    slice2.read_bytes(&mut data[i + j + seek..]).unwrap();
                    slice2.seek(0).unwrap();
                    slice2.read_bytes(&mut data[i + j..i + j + seek]).unwrap();
                }
                assert_eq!(data, bytes);
            }
        }
    }

    /// `testLargeWrites`.
    pub fn test_large_writes(&self, tc: &mut LuceneTestCase) {
        let temp = TempDir::new("largeWrites");
        let dir = self.dir(&temp);
        let mut os = dir.create_output("testBufferStart.txt").unwrap();
        let large_buf = random_bytes(tc.random(), 2048);
        let current_pos = os.file_pointer();
        os.write_bytes(&large_buf);
        assert_eq!(os.file_pointer(), current_pos + large_buf.len() as u64);
        os.close().unwrap();
    }

    /// `testIndexOutputToString` (LUCENE-6084).
    pub fn test_index_output_to_string(&self, _tc: &mut LuceneTestCase) {
        let temp = TempDir::new("toString");
        let dir = self.dir(&temp);
        let out = dir.create_output("camelCase.txt").unwrap();
        assert!(format!("{out:?}").contains("camelCase.txt"), "{out:?}");
        out.close().unwrap();
    }

    /// `testCreateTempOutput`.
    pub fn test_create_temp_output(&self, tc: &mut LuceneTestCase) {
        let temp = TempDir::new("tempOutput");
        let dir = self.dir(&temp);
        let iters = tc.at_least(50);
        let mut names = Vec::new();
        for iter in 0..iters {
            let mut out = dir.create_temp_output("foo", "bar").unwrap();
            names.push(out.name().to_string());
            out.write_vint(iter);
            out.close().unwrap();
        }
        for (iter, name) in names.iter().enumerate() {
            let input = dir.open(name).unwrap();
            assert_eq!(SliceInput::new(&input).read_vint().unwrap(), iter as i32);
        }
        let files: BTreeSet<String> = dir
            .list_all()
            .unwrap()
            .into_iter()
            .filter(|f| !MockFs::is_extra(f))
            .collect();
        assert_eq!(files, names.into_iter().collect());
    }

    /// `testCreateOutputForExistingFile`: `FileAlreadyExists`.
    pub fn test_create_output_for_existing_file(&self, _tc: &mut LuceneTestCase) {
        let temp = TempDir::new("existing");
        let dir = self.dir(&temp);
        dir.create_output("file").unwrap().close().unwrap();
        let e = dir.create_output("file").unwrap_err();
        assert_eq!(io_kind(&e), Some(io::ErrorKind::AlreadyExists), "{e}");
        dir.delete_file("file").unwrap();
        dir.create_output("file").unwrap().close().unwrap();
    }

    /// `testSeekToEndOfFile`.
    pub fn test_seek_to_end_of_file(&self, _tc: &mut LuceneTestCase) {
        let temp = TempDir::new("seekEnd");
        let dir = self.dir(&temp);
        write_file(&dir, "a", &[0u8; 1024]);
        let input = dir.open("a").unwrap();
        let mut i = SliceInput::new(&input);
        i.seek(100).unwrap();
        assert_eq!(i.position(), 100);
        i.seek(1024).unwrap();
        assert_eq!(i.position(), 1024);
    }

    /// `testSeekBeyondEndOfFile`.
    pub fn test_seek_beyond_end_of_file(&self, _tc: &mut LuceneTestCase) {
        let temp = TempDir::new("seekBeyond");
        let dir = self.dir(&temp);
        write_file(&dir, "a", &[0u8; 1024]);
        let input = dir.open("a").unwrap();
        let mut i = SliceInput::new(&input);
        i.seek(100).unwrap();
        assert_eq!(i.position(), 100);
        assert!(is_eof(&i.seek(1025).unwrap_err()));
    }

    /// `testPendingDeletions`: a delete the virus checker refuses becomes a
    /// pending delete, and the file is gone to every later call.
    pub fn test_pending_deletions(&self, tc: &mut LuceneTestCase) {
        let Some(with_faults) = &self.with_faults else {
            return; // "we can only install VirusCheckingFS on an FSDirectory"
        };
        let temp = TempDir::new("pending");
        let fs: Arc<dyn FsFaults> =
            MockFs::new(MockFsConfig::default().with_virus_checker(tc.random()));
        let dir = with_faults(temp.path(), fs);
        // Keep trying until virus checker refuses to delete:
        let file_name = loop {
            let segment = random_simple_string_between(tc.random(), 1, 6);
            let suffix = random_simple_string(tc.random());
            let candidate = segment_file_name(&segment, &suffix, "test");
            if dir.list_all().unwrap().contains(&candidate) {
                continue;
            }
            dir.create_output(&candidate).unwrap().close().unwrap();
            dir.delete_file(&candidate).unwrap();
            if !dir.pending_deletions().unwrap().is_empty() {
                break candidate;
            }
        };
        assert!(!dir.list_all().unwrap().contains(&file_name));
        let not_found = |r: lucene_store::Result<()>| {
            let e = r.unwrap_err();
            assert_eq!(io_kind(&e), Some(io::ErrorKind::NotFound), "{e}");
        };
        not_found(dir.file_length(&file_name).map(drop));
        not_found(dir.rename(&file_name, "file2"));
        not_found(dir.delete_file(&file_name));
        not_found(dir.open(&file_name).map(drop));
    }

    /// `testListAllIsSorted`.
    pub fn test_list_all_is_sorted(&self, tc: &mut LuceneTestCase) {
        let temp = TempDir::new("sorted");
        let dir = self.dir(&temp);
        let count = tc.at_least(20) as usize;
        let mut names = BTreeSet::new();
        while names.len() < count {
            let segment = random_simple_string_between(tc.random(), 1, 6);
            let suffix = random_simple_string(tc.random());
            let name = segment_file_name(&segment, &suffix, "test");
            if tc.random().next_int_bounded(5) == 1 {
                let out = dir.create_temp_output(&name, "foo").unwrap();
                names.insert(out.name().to_string());
                out.close().unwrap();
            } else if !names.contains(&name) {
                let out = dir.create_output(&name).unwrap();
                names.insert(out.name().to_string());
                out.close().unwrap();
            }
        }
        let actual = dir.list_all().unwrap();
        let mut expected = actual.clone();
        expected.sort();
        assert_eq!(actual, expected);
    }

    /// `testDataTypes`.
    pub fn test_data_types(&self, _tc: &mut LuceneTestCase) {
        let values = [43u32, 12345, 123456, 1234567890];
        let temp = TempDir::new("testDataTypes");
        let dir = self.dir(&temp);
        let mut out = dir.create_output("test").unwrap();
        out.write_byte(43);
        out.write_i16(12345);
        out.write_i32(1234567890);
        out.write_group_vints(&values);
        out.write_i64(1234567890123456789);
        out.close().unwrap();
        let input = dir.open("test").unwrap();
        let mut i = SliceInput::new(&input);
        assert_eq!(i.read_byte().unwrap(), 43);
        assert_eq!(i.read_i16().unwrap(), 12345);
        assert_eq!(i.read_i32().unwrap(), 1234567890);
        let mut restored = [0u64; 4];
        i.read_group_vints(&mut restored).unwrap();
        assert_eq!(restored, values.map(u64::from));
        assert_eq!(i.read_i64().unwrap(), 1234567890123456789);
    }

    /// `testGroupVIntOverflow`, minus the `ArithmeticException` (a value
    /// past `u32` is unrepresentable): `2^31` survives the round trip.
    pub fn test_group_vint_overflow(&self, tc: &mut LuceneTestCase) {
        let temp = TempDir::new("testGroupVIntOverflow");
        let dir = self.dir(&temp);
        let size = 32;
        let mut values = [0u32; 32];
        values[0] = 1 << 31;
        for i in 0..size {
            if tc.random().next_boolean() {
                values[i] = values[0];
            }
        }
        let limit = next_int(tc.random(), 1, size as i32 - 1) as usize;
        let mut out = dir.create_output("test").unwrap();
        out.write_group_vints(&values[..limit]);
        out.close().unwrap();
        let input = dir.open("test").unwrap();
        let mut restore = vec![0u64; limit];
        SliceInput::new(&input)
            .read_group_vints(&mut restore)
            .unwrap();
        for i in 0..limit {
            assert_eq!(restore[i], u64::from(values[i]));
        }
    }

    /// `testGroupVInt`.
    pub fn test_group_vint(&self, tc: &mut LuceneTestCase) {
        let temp = TempDir::new("testGroupVInt");
        let dir = self.dir(&temp);
        // test fallback to default implementation of readGroupVInt
        Self::do_test_group_vint(tc, &dir, 5, 1, 6, 8);
        // use more iterations to covers all bpv
        let iterations = tc.at_least(100);
        Self::do_test_group_vint(tc, &dir, iterations, 1, 31, 128);
    }

    /// `doTestGroupVInt(dir, iterations, minBpv, maxBpv, maxNumValues)`:
    /// group vints decode to what plain vints of the same values do.
    pub fn do_test_group_vint(
        tc: &mut LuceneTestCase,
        dir: &dyn Directory,
        iterations: i32,
        min_bpv: i32,
        max_bpv: i32,
        max_num_values: i32,
    ) {
        let mut values = vec![0u32; max_num_values as usize];
        let mut num_values_array = Vec::with_capacity(iterations as usize);
        let mut group_out = dir.create_output("group-varint").unwrap();
        let mut vint_out = dir.create_output("vint").unwrap();
        for _ in 0..iterations {
            let bpv = next_int(tc.random(), min_bpv, max_bpv);
            let n = next_int(tc.random(), 1, max_num_values) as usize;
            num_values_array.push(n);
            let max_value = ((1u64 << bpv) - 1) as i32;
            for v in values.iter_mut().take(n) {
                *v = random_int_between(tc.random(), 0, max_value) as u32;
                vint_out.write_vint(*v as i32);
            }
            group_out.write_group_vints(&values[..n]);
        }
        group_out.close().unwrap();
        vint_out.close().unwrap();
        let group_in = dir.open("group-varint").unwrap();
        let vint_in = dir.open("vint").unwrap();
        let mut group_in = SliceInput::new(&group_in);
        let mut vint_in = SliceInput::new(&vint_in);
        for &n in &num_values_array {
            let mut decoded = vec![0u64; n];
            group_in.read_group_vints(&mut decoded).unwrap();
            for v in decoded {
                assert_eq!(i64::from(vint_in.read_vint().unwrap()), v as i64);
            }
        }
        dir.delete_file("group-varint").unwrap();
        dir.delete_file("vint").unwrap();
    }

    /// `testPrefetch`.
    pub fn test_prefetch(&self, tc: &mut LuceneTestCase) {
        self.do_test_prefetch(tc, 0);
    }

    /// `testPrefetchOnSlice`.
    pub fn test_prefetch_on_slice(&self, tc: &mut LuceneTestCase) {
        let start = next_int(tc.random(), 1, 1024) as usize;
        self.do_test_prefetch(tc, start);
    }

    /// `doTestPrefetch(startOffset)`: prefetches never change what a read
    /// at any offset returns.
    fn do_test_prefetch(&self, tc: &mut LuceneTestCase, start_offset: usize) {
        let temp = TempDir::new("prefetch");
        let dir = self.dir(&temp);
        let total_length = start_offset + next_int(tc.random(), 16384, 65536) as usize;
        let arr = random_bytes(tc.random(), total_length);
        write_file(&dir, "temp.bin", &arr);
        let orig = dir.open("temp.bin").unwrap();
        let orig = SliceInput::new(&orig);
        let mut input = if start_offset == 0 {
            orig.slice_input("clone", 0, total_length as u64).unwrap()
        } else {
            orig.slice_input(
                "slice",
                start_offset as u64,
                (total_length - start_offset) as u64,
            )
            .unwrap()
        };
        let len = input.len();
        let mut temp_buf = [0u8; 2048];
        for _ in 0..10_000 {
            let r = tc.random();
            let offset = next_int(r, 0, len as i32 - 1) as usize;
            if r.next_boolean() {
                let _prefetch_length = next_long(r, 1, (len - offset) as i64);
                input.prefetch(offset);
            }
            input.seek(offset).unwrap();
            assert_eq!(input.position(), offset);
            match r.next_int_bounded(100) {
                0 => assert_eq!(input.read_byte().unwrap(), arr[start_offset + offset]),
                1 => {
                    if len - offset >= 8 {
                        let at = start_offset + offset;
                        let want = i64::from_le_bytes(arr[at..at + 8].try_into().unwrap());
                        assert_eq!(input.read_i64().unwrap(), want);
                    }
                }
                _ => {
                    let read_length =
                        next_int(r, 1, temp_buf.len().min(len - offset) as i32) as usize;
                    input.read_bytes(&mut temp_buf[..read_length]).unwrap();
                    let at = start_offset + offset;
                    assert_eq!(&temp_buf[..read_length], &arr[at..at + read_length]);
                }
            }
        }
    }
}

/// Declares one `#[test]` per [`BaseDirectoryTestCase`] scenario, each
/// running `$suite` (an expression building the suite) under a
/// [`LuceneTestCase`] named after it -- what subclassing
/// `BaseDirectoryTestCase` does in Java. `testThreadSafetyInListAll` runs
/// only when `$slow_list_all` is false or the run is nightly (Java's
/// `TestMockDirectoryWrapper` marks it `@Nightly`).
#[macro_export]
macro_rules! base_directory_tests {
    ($suite:expr) => {
        $crate::base_directory_tests!($suite, slow_list_all = false);
    };
    ($suite:expr, slow_list_all = $slow:expr) => {
        #[test]
        fn test_thread_safety_in_list_all() {
            let mut tc = $crate::util::lucene_test_case::LuceneTestCase::new(
                "test_thread_safety_in_list_all",
            );
            if !$slow || tc.nightly() {
                ($suite).test_thread_safety_in_list_all(&mut tc);
            }
        }
        $crate::base_directory_tests!(@each $suite;
            test_copy_from, test_rename, test_delete_file, test_byte, test_short, test_int,
            test_long, test_aligned_little_endian_longs, test_unaligned_little_endian_longs,
            test_little_endian_longs_underflow, test_aligned_ints, test_unaligned_ints,
            test_ints_underflow, test_aligned_floats, test_unaligned_floats,
            test_floats_underflow, test_string, test_vint, test_vlong, test_zint, test_zlong,
            test_set_of_strings, test_map_of_strings, test_checksum,
            test_file_exists_in_list_after_created, test_seek_to_eof_then_back,
            test_illegal_eof, test_seek_past_eof, test_slice_out_of_bounds, test_no_dir,
            test_copy_bytes, test_copy_bytes_with_threads, test_fsync_doesnt_create_new_files,
            test_random_long, test_random_int, test_random_short, test_random_byte,
            test_slice_of_slice, test_large_writes, test_index_output_to_string,
            test_create_temp_output, test_create_output_for_existing_file,
            test_seek_to_end_of_file, test_seek_beyond_end_of_file, test_pending_deletions,
            test_list_all_is_sorted, test_data_types, test_group_vint_overflow,
            test_group_vint, test_prefetch, test_prefetch_on_slice);
    };
    (@each $suite:expr; $($name:ident),*) => {$(
        #[test]
        fn $name() {
            ($suite).$name(&mut $crate::util::lucene_test_case::LuceneTestCase::new(stringify!($name)));
        }
    )*};
}
