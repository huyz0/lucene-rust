#![cfg(test)]
use super::*;
use lucene_store::index_output::{CloseVerdict, OutputFaults, WriteVerdict};
use lucene_store::ByteBuffersDirectory;

struct Recording {
    calls: usize,
    fail: bool,
}
impl NormsConsumer for Recording {
    fn add_norms_field(&mut self, field: NormsField<'_>) -> norms::Result<()> {
        assert!(matches!(field, NormsField::Dense(7, [3])));
        self.calls += 1;
        if self.fail {
            Err(error("delegate add"))
        } else {
            Ok(())
        }
    }
    fn close(&mut self) -> norms::Result<()> {
        self.calls += 1;
        if self.fail {
            Err(error("delegate close"))
        } else {
            Ok(())
        }
    }
}

#[test]
fn close_draws_after_success_and_never_after_delegate_failure() {
    let random = Rc::new(RefCell::new(JavaRandom::new(42)));
    let mut consumer = CrankyNormsConsumer::new(
        Recording {
            calls: 0,
            fail: true,
        },
        random.clone(),
    );
    assert_eq!(consumer.close().unwrap_err().to_string(), "delegate close");
    assert_eq!(consumer.delegate().calls, 1);
    let mut expected = JavaRandom::new(42);
    assert_eq!(
        random.borrow().clone().next_int(),
        expected.clone().next_int()
    );
    consumer.delegate_mut().fail = false;
    consumer.close().unwrap();
    expected.next_int_bounded(100);
    assert_eq!(random.borrow_mut().next_int(), expected.next_int());
}

#[test]
fn guards_preserve_order_and_exact_errors() {
    let failing_seed = (0..10000)
        .find(|&s| JavaRandom::new(s).next_int_bounded(100) == 0)
        .unwrap();
    let random = Rc::new(RefCell::new(JavaRandom::new(failing_seed)));
    let mut consumer = CrankyNormsConsumer::new(
        Recording {
            calls: 0,
            fail: true,
        },
        random.clone(),
    );
    assert_eq!(
        consumer
            .add_norms_field(NormsField::Dense(7, &[3]))
            .unwrap_err()
            .to_string(),
        FAKE_ADD_ERROR
    );
    assert_eq!(consumer.delegate().calls, 0);
    random.borrow_mut().set_seed(failing_seed);
    consumer.delegate_mut().fail = false;
    assert_eq!(consumer.close().unwrap_err().to_string(), FAKE_CLOSE_ERROR);
    assert_eq!(consumer.delegate().calls, 1);
    random.borrow_mut().set_seed(42);
    consumer.delegate_mut().fail = true;
    assert_eq!(
        consumer
            .add_norms_field(NormsField::Dense(7, &[3]))
            .unwrap_err()
            .to_string(),
        "delegate add"
    );
    consumer.delegate_mut().fail = false;
    consumer
        .add_norms_field(NormsField::Dense(7, &[3]))
        .unwrap();
    assert_eq!(consumer.delegate().calls, 3);
}

struct CloseFailure;
impl OutputFaults for CloseFailure {
    fn before_write(&mut self, _: &str, _: usize) -> WriteVerdict {
        WriteVerdict::Write
    }
    fn on_close(&mut self, _: &str) -> CloseVerdict {
        CloseVerdict::FailAfterFlush(std::io::Error::other("output close"))
    }
}

struct CountClose(std::sync::Arc<std::sync::atomic::AtomicUsize>);
impl OutputFaults for CountClose {
    fn before_write(&mut self, _: &str, _: usize) -> WriteVerdict {
        WriteVerdict::Write
    }
    fn on_close(&mut self, _: &str) -> CloseVerdict {
        self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        CloseVerdict::FailAfterFlush(std::io::Error::other("cleanup close"))
    }
}

#[test]
fn real_delegate_close_failure_closes_both_outputs_without_drawing() {
    let dir = ByteBuffersDirectory::new();
    let state = NormsState {
        directory: &dir,
        segment: "_s",
        id: &[0; 16],
        suffix: "",
        max_doc: 1,
    };
    let mut concrete = Lucene90NormsFormat.norms_consumer(&state).unwrap();
    concrete
        .outputs
        .as_mut()
        .unwrap()
        .0
        .set_faults(Box::new(CloseFailure));
    concrete
        .add_norms_field(NormsField::Dense(0, &[17]))
        .unwrap();
    let random = Rc::new(RefCell::new(JavaRandom::new(42)));
    let mut consumer = CrankyNormsConsumer::new(concrete, random.clone());
    assert_eq!(consumer.close().unwrap_err().to_string(), "output close");
    assert_eq!(
        random.borrow_mut().next_int(),
        JavaRandom::new(42).next_int()
    );
    assert!(dir.open("_s.nvm").is_ok());
    assert!(dir.open("_s.nvd").is_ok());
    consumer.close().unwrap();
}

#[test]
fn open_guards_precede_directory_creation_and_creation_errors_propagate() {
    let dir = ByteBuffersDirectory::new();
    let state = NormsState {
        directory: &dir,
        segment: "_s",
        id: &[0; 16],
        suffix: "",
        max_doc: 1,
    };
    let seed = (0..10000)
        .find(|&s| JavaRandom::new(s).next_int_bounded(100) == 0)
        .unwrap();
    let random = Rc::new(RefCell::new(JavaRandom::new(seed)));
    let mut format = CrankyNormsFormat::new(Lucene90NormsFormat, random.clone());
    assert_eq!(
        format.norms_consumer(&state).err().unwrap().to_string(),
        FAKE_OPEN_ERROR
    );
    assert!(dir.list_all().unwrap().is_empty());
    random.borrow_mut().set_seed(0);
    let _consumer = format.norms_consumer(&state).unwrap();
    assert!(format.norms_consumer(&state).is_err());
    assert!(format.norms_producer(&state).is_err());
    let other = ByteBuffersDirectory::new();
    other.create_output("_s.nvm").unwrap().close().unwrap();
    assert!(format
        .delegate_mut()
        .norms_consumer(&NormsState {
            directory: &other,
            ..state
        })
        .is_err());
    assert!(other.list_all().unwrap().contains(&"_s.nvd".to_string()));
}

#[test]
fn malformed_owned_values_fail_at_close_and_close_remains_idempotent() {
    for fault in [false, true] {
        let dir = ByteBuffersDirectory::new();
        let state = NormsState {
            directory: &dir,
            segment: "_s",
            id: &[0; 16],
            suffix: "",
            max_doc: 1,
        };
        let mut consumer = Lucene90NormsFormat.norms_consumer(&state).unwrap();
        let closes = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        if fault {
            let (data, meta) = consumer.outputs.as_mut().unwrap();
            data.set_faults(Box::new(CountClose(closes.clone())));
            meta.set_faults(Box::new(CountClose(closes.clone())));
        }
        consumer.add_norms_field(NormsField::Dense(0, &[])).unwrap();
        assert!(consumer
            .close()
            .unwrap_err()
            .to_string()
            .contains("values.len()"));
        // Both outputs are published and readable even though encoding failed.
        assert!(dir.open("_s.nvm").unwrap().is_empty());
        assert!(dir.open("_s.nvd").unwrap().is_empty());
        assert_eq!(
            closes.load(std::sync::atomic::Ordering::SeqCst),
            if fault { 2 } else { 0 }
        );
        consumer.close().unwrap();
        assert_eq!(
            consumer
                .add_norms_field(NormsField::Sparse(0, &[]))
                .unwrap_err()
                .to_string(),
            "closed consumer"
        );
    }
}
