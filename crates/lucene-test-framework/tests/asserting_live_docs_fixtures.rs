//! Actual Java assertions, iterators, masks and Lucene90 bytes.
use std::cell::Cell;
use std::io;
use std::panic::{catch_unwind, AssertUnwindSafe};

use lucene_store::data_output::DataOutput;
use lucene_store::directory::Directory;
use lucene_store::ByteBuffersDirectory;
use lucene_test_framework::codecs::asserting_live_docs::{
    AssertingBits, AssertingLiveDocs, AssertingLiveDocsFormat, DenseLiveDocs, LiveDocs,
};
use lucene_test_framework::codecs::cranky_live_docs::{
    LiveDocsFormat, LiveDocsInfo, Lucene90LiveDocsFormat,
};
use lucene_util::bits::{Bits, MatchAllBits, NO_MORE_DOCS};
use lucene_util::fixed_bit_set::FixedBitSet;

const ID: [u8; 16] = *b"0123456789abcdef";
fn info(length: usize, generation: i64, deleted: usize) -> LiveDocsInfo<'static> {
    LiveDocsInfo {
        name: "_ald",
        segment_id: &ID,
        max_doc: length,
        del_gen: generation,
        next_del_gen: generation + 1,
        del_count: deleted,
    }
}
fn bits(text: &str) -> FixedBitSet {
    let mut bits = FixedBitSet::new(if text == "-" { 0 } else { text.len() });
    for (i, c) in text.chars().enumerate() {
        if c == '1' {
            bits.set(i);
        }
    }
    bits
}
fn bit_string(bits: &impl Bits) -> String {
    if bits.length() == 0 {
        return "-".into();
    }
    (0..bits.length())
        .map(|i| if bits.get(i) { '1' } else { '0' })
        .collect()
}
fn hex(text: &str) -> Vec<u8> {
    text.as_bytes()
        .chunks_exact(2)
        .map(|b| u8::from_str_radix(std::str::from_utf8(b).unwrap(), 16).unwrap())
        .collect()
}

#[test]
fn java_real_bytes_masks_and_iterators() {
    let fixture = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/testframework/data/asserting_live_docs.tsv"
    ));
    let mut current = None;
    let mut cases = 0;
    for line in fixture.lines() {
        let f: Vec<_> = line.split('\t').collect();
        match f[0] {
            "valid" => {
                let length = f[2].parse().unwrap();
                let generation = f[3].parse::<i64>().unwrap();
                let deleted = f[4].parse().unwrap();
                let dir = ByteBuffersDirectory::new();
                let format = AssertingLiveDocsFormat::new(Lucene90LiveDocsFormat);
                let before = info(length, generation - 1, 0);
                format
                    .write_live_docs(&bits(f[5]), &dir, &before, deleted)
                    .unwrap();
                let expected = hex(f[7]);
                assert_eq!(dir.open(f[6]).unwrap().as_ref(), expected.as_slice());
                let after = info(length, generation, deleted);
                let read = format.read_live_docs(&dir, &after).unwrap();
                assert_eq!(bit_string(&read), f[5]);
                assert_eq!(read.deleted_count() as usize, deleted);
                let mut files = vec!["existing".into()];
                format.files(&after, &mut files).unwrap();
                assert_eq!(files, vec!["existing", f[6]]);
                let mut corrupt = expected;
                *corrupt.last_mut().unwrap() ^= 1;
                dir.delete_file(f[6]).unwrap();
                let mut output = dir.create_output(f[6]).unwrap();
                output.write_bytes(&corrupt);
                output.close().unwrap();
                assert!(format.read_live_docs(&dir, &after).is_err());
                current = Some(read);
                cases += 1;
            }
            "iterator" => {
                let view = current.as_ref().unwrap();
                let iterator = || {
                    if f[2] == "live" {
                        view.live_docs_iterator()
                    } else {
                        view.deleted_docs_iterator()
                    }
                };
                let mut it = iterator();
                assert_eq!(it.doc_id(), -1);
                assert_eq!(it.cost(), f[3].parse::<i64>().unwrap());
                for expected in f[4].split(',') {
                    let doc = it.next_doc().unwrap();
                    assert_eq!(doc, expected.parse::<i32>().unwrap());
                    assert_eq!(it.doc_id(), doc);
                }
                for probe in f[5].split(',') {
                    let values: Vec<i32> = probe.split(':').map(|s| s.parse().unwrap()).collect();
                    let mut it = iterator();
                    let doc = it.advance(values[0]).unwrap();
                    assert_eq!(doc, values[1]);
                    let next = if doc == NO_MORE_DOCS {
                        doc
                    } else {
                        it.next_doc().unwrap()
                    };
                    assert_eq!(next, values[2]);
                }
            }
            "mask" => {
                let mut mask = bits(f[3]);
                let outcome = catch_unwind(AssertUnwindSafe(|| {
                    current
                        .as_ref()
                        .unwrap()
                        .apply_mask(&mut mask, f[2].parse().unwrap())
                }));
                assert_eq!(if outcome.is_ok() { "ok" } else { "assert" }, f[4]);
                assert_eq!(bit_string(&mask), f[5]);
            }
            _ => {}
        }
    }
    assert_eq!(cases, 28);
}

struct Probe {
    length: Cell<usize>,
    fail: Cell<bool>,
    reads: Cell<usize>,
    writes: Cell<usize>,
    listings: Cell<usize>,
}
impl LiveDocsFormat for &Probe {
    fn read_live_docs(&self, _: &dyn Directory, _: &LiveDocsInfo<'_>) -> io::Result<FixedBitSet> {
        self.reads.set(self.reads.get() + 1);
        if self.fail.get() {
            return Err(io::Error::other("delegate read"));
        }
        let mut result = FixedBitSet::new(self.length.get());
        for i in 0..result.len() {
            result.set(i);
        }
        Ok(result)
    }
    fn write_live_docs(
        &self,
        _: &FixedBitSet,
        _: &dyn Directory,
        _: &LiveDocsInfo<'_>,
        _: usize,
    ) -> io::Result<()> {
        self.writes.set(self.writes.get() + 1);
        if self.fail.get() {
            Err(io::Error::other("delegate write"))
        } else {
            Ok(())
        }
    }
    fn files(&self, _: &LiveDocsInfo<'_>, files: &mut Vec<String>) -> io::Result<()> {
        self.listings.set(self.listings.get() + 1);
        files.push("delegate-file".into());
        if self.fail.get() {
            Err(io::Error::other("delegate files"))
        } else {
            Ok(())
        }
    }
}

#[test]
fn java_negative_contracts_and_callback_order() {
    let fixture = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/testframework/data/asserting_live_docs.tsv"
    ));
    let probe = Probe {
        length: Cell::new(2),
        fail: Cell::new(false),
        reads: Cell::new(0),
        writes: Cell::new(0),
        listings: Cell::new(0),
    };
    let format = AssertingLiveDocsFormat::new(&probe);
    let dir = ByteBuffersDirectory::new();
    let mut skipped_null = 0;
    for line in fixture
        .lines()
        .filter(|line| line.starts_with("negative\t"))
    {
        let f: Vec<_> = line.split('\t').collect();
        if f[1] == "format_read_null" {
            // The delegate's owned FixedBitSet return cannot be null.
            skipped_null += 1;
            continue;
        }
        let outcome = catch_unwind(AssertUnwindSafe(|| -> io::Result<()> {
            let plain = AssertingBits::new(MatchAllBits(2));
            match f[1] {
                "bits_negative_index" => {
                    plain.get_signed(-1);
                }
                "bits_upper_bound" => {
                    plain.get(2);
                }
                "bits_negative_offset" => plain.apply_mask_signed(&mut bits("1"), -1),
                "bits_past_end_mask_delegate" => {
                    let mut mask = bits("1");
                    plain.apply_mask(&mut mask, 10);
                    assert!(mask.get(0));
                }
                "live_negative_count" => {
                    AssertingLiveDocs::new(CountProbe(-1)).deleted_count();
                }
                "live_count_past_length" => {
                    AssertingLiveDocs::new(CountProbe(3)).deleted_count();
                }
                "format_write_length" => {
                    format.write_live_docs(&bits("1"), &dir, &info(2, -1, 0), 0)?
                }
                "format_write_count" => {
                    format.write_live_docs(&bits("11"), &dir, &info(2, -1, 0), 1)?
                }
                "format_read_length" => {
                    probe.length.set(1);
                    format.read_live_docs(&dir, &info(2, 1, 0))?;
                }
                "format_read_count" => {
                    probe.length.set(2);
                    format.read_live_docs(&dir, &info(2, 1, 1))?;
                }
                "format_write_delegate_io" => {
                    probe.fail.set(true);
                    format.write_live_docs(&bits("11"), &dir, &info(2, -1, 0), 0)?;
                }
                "format_read_delegate_io" => {
                    format.read_live_docs(&dir, &info(2, 1, 0))?;
                }
                "format_files_delegate_io" => {
                    let mut files = vec![];
                    let result = format.files(&info(2, 1, 0), &mut files);
                    assert_eq!(files, vec!["delegate-file"]);
                    result?;
                }
                other => panic!("unknown case {other}"),
            }
            Ok(())
        }));
        let status = match outcome {
            Err(_) => "assert",
            Ok(Err(error)) => {
                assert_eq!(error.to_string(), f[3]);
                "io"
            }
            Ok(Ok(())) => "ok",
        };
        assert_eq!(status, f[2], "{}", f[1]);
        if f[4] != "-" {
            assert_eq!(
                probe.reads.get() + skipped_null,
                f[4].parse::<usize>().unwrap()
            );
            assert_eq!(probe.writes.get(), f[5].parse::<usize>().unwrap());
            assert_eq!(probe.listings.get(), f[6].parse::<usize>().unwrap());
        }
    }
}

struct CountProbe(i32);
impl Bits for CountProbe {
    fn get(&self, _: usize) -> bool {
        true
    }
    fn length(&self) -> usize {
        2
    }
}
impl LiveDocs for CountProbe {
    fn deleted_count(&self) -> i32 {
        self.0
    }
    fn live_docs_iterator(&self) -> Box<dyn lucene_search::reader::DocIdSetIterator + '_> {
        Box::new(Empty(17))
    }
    fn deleted_docs_iterator(&self) -> Box<dyn lucene_search::reader::DocIdSetIterator + '_> {
        Box::new(Empty(23))
    }
}
struct Empty(i64);
impl lucene_search::reader::DocIdSetIterator for Empty {
    fn doc_id(&self) -> i32 {
        NO_MORE_DOCS
    }
    fn next_doc(&mut self) -> lucene_search::Result<i32> {
        Ok(NO_MORE_DOCS)
    }
    fn advance(&mut self, _: i32) -> lucene_search::Result<i32> {
        Ok(NO_MORE_DOCS)
    }
    fn cost(&self) -> i64 {
        self.0
    }
}

#[test]
fn generic_wrappers_forward_masks_and_iterators() {
    let plain = AssertingBits::new(MatchAllBits(2));
    assert_eq!(plain.length(), 2);
    assert!(plain.get_signed(1));
    let mut mask = bits("10");
    plain.apply_mask_signed(&mut mask, 0);
    assert_eq!(bit_string(&mask), "10");
    let wrapped = AssertingLiveDocs::new(CountProbe(0));
    assert_eq!(wrapped.live_docs_iterator().cost(), 17);
    assert_eq!(wrapped.deleted_docs_iterator().cost(), 23);
    assert_eq!(wrapped.deleted_count(), 0);
    let dense = DenseLiveDocs::new(bits("10"));
    assert_eq!(dense.deleted_count(), 1);
}
