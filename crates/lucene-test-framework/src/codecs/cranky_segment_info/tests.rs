#![cfg(test)]
use super::*;
use lucene_index::segment_info::LuceneVersion;
use lucene_store::ByteBuffersDirectory;

fn info() -> SegmentInfo {
    SegmentInfo {
        id: *b"0123456789abcdef",
        version: LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        },
        min_version: None,
        doc_count: 7,
        is_compound_file: false,
        has_blocks: false,
        diagnostics: vec![],
        files: vec![],
        attributes: vec![],
        index_sort: None,
    }
}

struct Fails {
    calls: usize,
    random: Rc<RefCell<JavaRandom>>,
}
impl SegmentInfoFormatDelegate for Fails {
    fn read(
        &mut self,
        _: &dyn Directory,
        _: &str,
        _: &[u8; 16],
    ) -> segment_info::Result<SegmentInfo> {
        assert!(self.random.try_borrow_mut().is_ok());
        self.calls += 1;
        Err(lucene_store::Error::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "delegate read failure",
        ))
        .into())
    }
    fn write(
        &mut self,
        _: &dyn Directory,
        _: &str,
        _: &mut SegmentInfo,
    ) -> segment_info::Result<()> {
        assert!(self.random.try_borrow_mut().is_ok());
        self.calls += 1;
        Err(lucene_store::Error::Io(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "delegate write failure",
        ))
        .into())
    }
}

#[test]
fn exact_draws_and_delegate_failures_preserve_io_errors() {
    let random = Rc::new(RefCell::new(JavaRandom::new(42)));
    let mut expected = JavaRandom::new(42);
    let delegate = Fails {
        calls: 0,
        random: random.clone(),
    };
    let mut cranky = CrankySegmentInfoFormat::new(delegate, random.clone());
    let dir = ByteBuffersDirectory::new();
    let mut attempts = 0;
    let mut injected = 0;
    for _ in 0..1000 {
        let read = cranky.read(&dir, "_base", &info().id).unwrap_err();
        assert!(
            matches!(read, segment_info::Error::Store(lucene_store::Error::Io(ref e)) if e.kind() == std::io::ErrorKind::NotFound && e.to_string() == "delegate read failure")
        );
        attempts += 1;
        let fail = expected.next_int_bounded(100) == 0;
        let err = cranky.write(&dir, "_base", &mut info()).unwrap_err();
        if fail {
            injected += 1;
            assert_eq!(
                err.to_string(),
                "Fake IOException from SegmentInfoFormat.write()"
            );
        } else {
            attempts += 1;
            assert!(
                matches!(err, segment_info::Error::Store(lucene_store::Error::Io(ref e)) if e.kind() == std::io::ErrorKind::PermissionDenied && e.to_string() == "delegate write failure")
            );
        }
    }
    assert!(injected > 0);
    assert_eq!(cranky.delegate().calls, attempts);
    cranky.delegate_mut().calls = 123;
    assert_eq!(cranky.delegate().calls, 123);
    assert_eq!(random.borrow_mut().next_int(), expected.next_int());
    assert!(dir.list_all().unwrap().is_empty());
}

#[test]
fn real_codec_round_trip_and_create_failure() {
    let dir = ByteBuffersDirectory::new();
    let mut codec = Lucene99SegmentInfoFormat;
    let mut written = info();
    codec.write(&dir, "_base", &mut written).unwrap();
    assert_eq!(written.files, ["_base.si"]);
    let bytes = dir.open("_base.si").unwrap();
    assert_eq!(
        &*bytes,
        include_bytes!("../../../../../fixtures/testframework/data/cranky_segment_info.si")
    );
    let read = codec.read(&dir, "_base", &written.id).unwrap();
    assert_eq!(read.doc_count, 7);
    assert_eq!(read.id, written.id);
    assert_eq!(read.files, written.files);
    assert!(codec.read(&dir, "_base", &[0; 16]).is_err());
    assert!(codec.read(&dir, "_absent", &written.id).is_err());
    let mut invalid = info();
    assert!(codec.write(&dir, "_base", &mut invalid).is_err());
    assert!(invalid.files.is_empty());
}

#[test]
fn wrappers_share_one_live_random_source() {
    let random = Rc::new(RefCell::new(JavaRandom::new(17)));
    let mut first = CrankySegmentInfoFormat::new(
        Fails {
            calls: 0,
            random: random.clone(),
        },
        random.clone(),
    );
    let mut second = CrankySegmentInfoFormat::new(
        Fails {
            calls: 0,
            random: random.clone(),
        },
        random.clone(),
    );
    let mut expected = JavaRandom::new(17);
    let dir = ByteBuffersDirectory::new();
    for _ in 0..100 {
        for format in [&mut first, &mut second] {
            let err = format.write(&dir, "_s", &mut info()).unwrap_err();
            let expected_fake = expected.next_int_bounded(100) == 0;
            assert_eq!(
                err.to_string().starts_with("Fake IOException"),
                expected_fake
            );
        }
    }
    assert_eq!(random.borrow_mut().next_long(), expected.next_long());
}
