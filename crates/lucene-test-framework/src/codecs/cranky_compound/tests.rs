#![cfg(test)]
use super::*;
use lucene_store::ByteBuffersDirectory;
use std::io;

const ID: [u8; 16] = *b"0123456789abcdef";
fn info<'a>(files: &'a [String]) -> CompoundInfo<'a> {
    CompoundInfo {
        name: "_test",
        id: &ID,
        files,
    }
}
struct Failing {
    calls: usize,
    random: Rc<RefCell<JavaRandom>>,
    fail: bool,
}
impl CompoundFormat for Failing {
    fn get_compound_reader<'d>(
        &mut self,
        _: &'d dyn Directory,
        info: &CompoundInfo<'_>,
    ) -> lucene_store::Result<CompoundReader<'d>> {
        assert!(self.random.try_borrow_mut().is_ok());
        assert_eq!((info.name, *info.id, info.files.len()), ("_test", ID, 0));
        self.calls += 1;
        Err(lucene_store::Error::Io(io::Error::new(
            io::ErrorKind::NotFound,
            "delegate read failure",
        )))
    }
    fn write(&mut self, _: &dyn Directory, info: &CompoundInfo<'_>) -> compound_format::Result<()> {
        assert!(self.random.try_borrow_mut().is_ok());
        assert_eq!((info.name, *info.id, info.files.len()), ("_test", ID, 0));
        self.calls += 1;
        if self.fail {
            Err(lucene_store::Error::Io(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "delegate write failure",
            ))
            .into())
        } else {
            Ok(())
        }
    }
}
#[test]
fn delegate_identity_draws_and_shared_random_are_preserved() {
    let dir = ByteBuffersDirectory::new();
    let random = Rc::new(RefCell::new(JavaRandom::new(42)));
    let mut formats = [
        CrankyCompoundFormat::new(
            Failing {
                calls: 0,
                random: random.clone(),
                fail: true,
            },
            random.clone(),
        ),
        CrankyCompoundFormat::new(
            Failing {
                calls: 0,
                random: random.clone(),
                fail: true,
            },
            random.clone(),
        ),
    ];
    let mut expected = JavaRandom::new(42);
    let mut calls = 0;
    let mut injected = 0;
    for op in 0..1000 {
        let format = &mut formats[op % 2];
        let error = format
            .get_compound_reader(&dir, &info(&[]))
            .err()
            .expect("delegate must fail");
        assert!(
            matches!(error, lucene_store::Error::Io(ref e) if e.kind() == io::ErrorKind::NotFound && e.to_string() == "delegate read failure")
        );
        calls += 1;
        let fail = expected.next_int_bounded(100) == 0;
        let error = format.write(&dir, &info(&[])).unwrap_err();
        if fail {
            injected += 1;
            assert_eq!(error.to_string(), FAKE_WRITE_ERROR);
        } else {
            calls += 1;
            assert!(
                matches!(error, compound_format::Error::Store(lucene_store::Error::Io(ref e)) if e.kind() == io::ErrorKind::PermissionDenied && e.to_string() == "delegate write failure")
            );
        }
    }
    assert!(injected > 0);
    assert_eq!(
        formats.iter().map(|f| f.delegate().calls).sum::<usize>(),
        calls
    );
    assert_eq!(random.borrow_mut().next_int(), expected.next_int());
    formats[0].delegate_mut().fail = false;
    while expected.next_int_bounded(100) == 0 {
        assert!(formats[0].write(&dir, &info(&[])).is_err());
    }
    formats[0].write(&dir, &info(&[])).unwrap();
    assert_eq!(random.borrow_mut().next_long(), expected.next_long());
    assert!(dir.list_all().unwrap().is_empty());
}
#[test]
fn injection_precedes_any_native_output_creation() {
    let seed = (0..10000)
        .find(|&seed| JavaRandom::new(seed).next_int_bounded(100) == 0)
        .unwrap();
    let random = Rc::new(RefCell::new(JavaRandom::new(seed)));
    let mut format = CrankyCompoundFormat::new(Lucene90CompoundFormat, random);
    let dir = ByteBuffersDirectory::new();
    let files = vec!["_test.missing".into()];
    assert_eq!(
        format.write(&dir, &info(&files)).unwrap_err().to_string(),
        FAKE_WRITE_ERROR
    );
    assert!(dir.list_all().unwrap().is_empty());
}
#[test]
fn native_output_creation_and_cleanup_follow_java_order() {
    let mut format = Lucene90CompoundFormat;
    let dir = ByteBuffersDirectory::new();
    dir.create_output("_test.cfs").unwrap().close().unwrap();
    assert!(
        matches!(format.write(&dir, &info(&[])), Err(compound_format::Error::Store(lucene_store::Error::Io(ref e))) if e.kind() == io::ErrorKind::AlreadyExists)
    );
    assert_eq!(dir.list_all().unwrap(), ["_test.cfs"]);
    let dir = ByteBuffersDirectory::new();
    dir.create_output("_test.cfe").unwrap().close().unwrap();
    assert!(
        matches!(format.write(&dir, &info(&[])), Err(compound_format::Error::Store(lucene_store::Error::Io(ref e))) if e.kind() == io::ErrorKind::AlreadyExists)
    );
    assert_eq!(dir.open("_test.cfs").unwrap().len(), 0);
    let dir = ByteBuffersDirectory::new();
    let files = vec!["_test.missing".into()];
    assert!(
        matches!(format.write(&dir, &info(&files)), Err(compound_format::Error::Store(lucene_store::Error::Io(ref e))) if e.kind() == io::ErrorKind::NotFound)
    );
    // Both memory outputs must have been explicitly closed despite failure.
    assert!(!dir.open("_test.cfs").unwrap().is_empty());
    assert!(!dir.open("_test.cfe").unwrap().is_empty());
    assert!(format.get_compound_reader(&dir, &info(&files)).is_err());
}
#[test]
fn empty_archive_uses_the_actual_reader_and_is_read_only() {
    let dir = ByteBuffersDirectory::new();
    let mut format = Lucene90CompoundFormat;
    format.write(&dir, &info(&[])).unwrap();
    let reader = format.get_compound_reader(&dir, &info(&[])).unwrap();
    assert!(reader.member_files().is_empty());
    reader.check_integrity().unwrap();
    assert!(reader.create_output("_test.new").is_err());
    assert!(reader.open("_test.missing").is_err());
    assert!(format
        .get_compound_reader(
            &dir,
            &CompoundInfo {
                id: &[0; 16],
                ..info(&[])
            }
        )
        .is_err());
}
