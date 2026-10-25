//! Actual Java CrankyLiveDocsFormat outcomes, RNG continuation and .liv bytes.
use std::cell::{Cell, RefCell};
use std::io;
use std::rc::Rc;

use lucene_store::data_output::DataOutput;
use lucene_store::directory::Directory;
use lucene_store::ByteBuffersDirectory;
use lucene_test_framework::codecs::cranky_live_docs::{
    CrankyLiveDocsFormat, LiveDocsFormat, LiveDocsInfo, Lucene90LiveDocsFormat,
};
use lucene_util::fixed_bit_set::FixedBitSet;
use lucene_util::java_random::JavaRandom;

const ID: [u8; 16] = *b"0123456789abcdef";
fn info(max_doc: usize, del_gen: i64, del_count: usize) -> LiveDocsInfo<'static> {
    LiveDocsInfo {
        name: "_0",
        segment_id: &ID,
        max_doc,
        del_gen,
        next_del_gen: if del_gen == -1 { 1 } else { del_gen + 1 },
        del_count,
    }
}

#[derive(Default)]
struct Probe {
    reads: Cell<usize>,
    writes: Cell<usize>,
    listings: Cell<usize>,
}
impl LiveDocsFormat for &Probe {
    fn read_live_docs(&self, _: &dyn Directory, _: &LiveDocsInfo<'_>) -> io::Result<FixedBitSet> {
        self.reads.set(self.reads.get() + 1);
        Ok(FixedBitSet::new(1))
    }
    fn write_live_docs(
        &self,
        _: &FixedBitSet,
        _: &dyn Directory,
        _: &LiveDocsInfo<'_>,
        _: usize,
    ) -> io::Result<()> {
        self.writes.set(self.writes.get() + 1);
        if self.writes.get().is_multiple_of(7) {
            Err(io::Error::other("delegate write"))
        } else {
            Ok(())
        }
    }
    fn files(&self, _: &LiveDocsInfo<'_>, files: &mut Vec<String>) -> io::Result<()> {
        self.listings.set(self.listings.get() + 1);
        files.push("delegate-file".into());
        Ok(())
    }
}

#[test]
fn java_sequences_and_real_live_docs_match() {
    let fixture = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/testframework/data/cranky_live_docs.tsv"
    ));
    let mut sequences = 0;
    let mut byte_cases = 0;
    let mut injected = 0;
    for line in fixture.lines() {
        let f: Vec<_> = line.split('\t').collect();
        let dir = ByteBuffersDirectory::new();
        if f[0] == "sequence" {
            let random = Rc::new(RefCell::new(JavaRandom::new(f[1].parse().unwrap())));
            let probe = Probe::default();
            let formats = [
                CrankyLiveDocsFormat::new(&probe, random.clone()),
                CrankyLiveDocsFormat::new(&probe, random.clone()),
            ];
            let info = info(1, -1, 0);
            for (i, expected) in f[2].chars().enumerate() {
                if i.is_multiple_of(13) {
                    random.borrow_mut().next_boolean();
                }
                let format = &formats[i & 1];
                assert_eq!(format.read_live_docs(&dir, &info).unwrap().len(), 1);
                let mut files = vec!["existing".into()];
                format.files(&info, &mut files).unwrap();
                assert_eq!(files, ["existing", "delegate-file"]);
                let actual = match format.write_live_docs(&FixedBitSet::new(1), &dir, &info, 0) {
                    Ok(()) => 'O',
                    Err(error) if error.to_string() == "delegate write" => 'D',
                    Err(error) => {
                        assert_eq!(
                            error.to_string(),
                            "Fake IOException from LiveDocsFormat.writeLiveDocs()"
                        );
                        injected += 1;
                        'F'
                    }
                };
                assert_eq!(actual, expected, "seed={} operation={i}", f[1]);
            }
            assert_eq!(
                random.borrow_mut().next_long(),
                f[3].parse::<i64>().unwrap()
            );
            assert_eq!(probe.reads.get(), f[4].parse::<usize>().unwrap());
            assert_eq!(probe.writes.get(), f[5].parse::<usize>().unwrap());
            assert_eq!(probe.listings.get(), f[6].parse::<usize>().unwrap());
            sequences += 1;
        } else {
            assert_eq!(f[0], "bytes");
            let max_doc = f[1].parse::<usize>().unwrap();
            let generation = f[2].parse::<i64>().unwrap();
            let mismatch = f[3] == "true";
            let mut bits = FixedBitSet::new(max_doc);
            for i in 0..max_doc - 1 {
                bits.set(i);
            }
            let random = Rc::new(RefCell::new(JavaRandom::new(42)));
            let format = CrankyLiveDocsFormat::new(Lucene90LiveDocsFormat, random.clone());
            let before = info(
                max_doc,
                if generation == 1 { -1 } else { generation - 1 },
                0,
            );
            assert_eq!(
                format
                    .write_live_docs(&bits, &dir, &before, usize::from(!mismatch))
                    .is_err(),
                mismatch
            );
            let expected: Vec<_> = (0..f[5].len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&f[5][i..i + 2], 16).unwrap())
                .collect();
            assert_eq!(&*dir.open(f[4]).unwrap(), expected);
            let after = info(max_doc, generation, 1);
            let mut files = vec!["existing".into()];
            format.files(&after, &mut files).unwrap();
            assert_eq!(files, ["existing", f[4]]);
            assert_eq!(
                format.read_live_docs(&dir, &after).unwrap().words(),
                bits.words()
            );
            let mut damaged = expected;
            *damaged.last_mut().unwrap() ^= 1;
            dir.delete_file(f[4]).unwrap();
            let mut output = dir.create_output(f[4]).unwrap();
            output.write_bytes(&damaged);
            output.close().unwrap();
            assert!(format.read_live_docs(&dir, &after).is_err());
            assert_eq!(
                random.borrow_mut().next_long(),
                f[6].parse::<i64>().unwrap()
            );
            byte_cases += 1;
        }
    }
    assert_eq!(sequences, 7);
    assert_eq!(byte_cases, 12);
    assert!(injected > 0);
}
