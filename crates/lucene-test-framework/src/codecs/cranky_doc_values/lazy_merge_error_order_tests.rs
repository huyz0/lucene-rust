#![cfg(test)]
use super::*;

struct Callbacks {
    events: RefCell<Vec<String>>,
    random: Rc<RefCell<JavaRandom>>,
}
impl Callbacks {
    fn record(&self, event: impl std::fmt::Display) {
        // Inspect continuation without consuming the wrapper's shared RNG.
        let continuation = self.random.borrow().clone().next_int();
        self.events
            .borrow_mut()
            .push(format!("{event}:{continuation}"));
    }
}
struct Source {
    index: usize,
    fails: bool,
    calls: Rc<Callbacks>,
}
impl Source {
    fn getter(&self, kind: &str) -> Result<()> {
        self.calls.record(format!("getter:{}:{kind}", self.index));
        if self.fails {
            Err(failure(format!("source {} getter failure", self.index)))
        } else {
            Ok(())
        }
    }
}
impl DocValuesProducer for Source {
    fn get_numeric(&self, _: &FieldInfo) -> Result<Cow<'_, NumericColumn>> {
        self.getter("numeric")?;
        Ok(Cow::Owned(vec![(0, 17)]))
    }
    fn get_binary(&self, _: &FieldInfo) -> Result<Cow<'_, BinaryColumn>> {
        self.getter("binary")?;
        Ok(Cow::Owned(BinaryColumn {
            docs: vec![0],
            lengths: vec![2],
            bytes: vec![0, 255],
        }))
    }
    fn get_sorted_numeric(&self, _: &FieldInfo) -> Result<Cow<'_, SortedNumericColumn>> {
        self.getter("sortedNumeric")?;
        Ok(Cow::Owned(SortedNumericColumn {
            docs: vec![0],
            counts: vec![2],
            values: vec![3, 3],
        }))
    }
    fn check_integrity(&self) -> Result<()> {
        self.calls.record(format!("integrity:{}", self.index));
        Ok(())
    }
    fn close(&mut self) -> Result<()> {
        Ok(())
    }
}
struct RequestingConsumer(Rc<Callbacks>);
impl RequestingConsumer {
    fn request(
        &self,
        kind: &str,
        field: &FieldInfo,
        producer: &dyn DocValuesProducer,
    ) -> Result<()> {
        self.0.record(format!("delegate:{kind}"));
        // Unlike the differential trace's identity-only consumer, this
        // delegate requests the mapped getter and observes its source errors.
        read_column(producer, field)?;
        Ok(())
    }
}
impl DocValuesConsumer for RequestingConsumer {
    fn add_numeric_field(&mut self, f: &FieldInfo, p: &dyn DocValuesProducer) -> Result<()> {
        self.request("numeric", f, p)
    }
    fn add_binary_field(&mut self, f: &FieldInfo, p: &dyn DocValuesProducer) -> Result<()> {
        self.request("binary", f, p)
    }
    fn add_sorted_numeric_field(&mut self, f: &FieldInfo, p: &dyn DocValuesProducer) -> Result<()> {
        self.request("sortedNumeric", f, p)
    }
    fn add_sorted_field(&mut self, _: &FieldInfo, _: &dyn DocValuesProducer) -> Result<()> {
        Err(failure("unexpected sorted add"))
    }
    fn add_sorted_set_field(&mut self, _: &FieldInfo, _: &dyn DocValuesProducer) -> Result<()> {
        Err(failure("unexpected sorted set add"))
    }
    fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

#[test]
fn lazy_source_getter_errors_follow_the_guard_and_preserve_callback_rng_order() {
    for (kind, name, operation) in [
        (DocValuesType::Numeric, "numeric", "addNumericField"),
        (DocValuesType::Binary, "binary", "addBinaryField"),
        (
            DocValuesType::SortedNumeric,
            "sortedNumeric",
            "addSortedNumericField",
        ),
    ] {
        for failing_source in [0, 1] {
            // Seed 42 passes the first guard; seed 18 fails it. The latter
            // proves getters remain lazy, even when they would throw.
            for seed in [42, 18] {
                let random = Rc::new(RefCell::new(JavaRandom::new(seed)));
                let calls = Rc::new(Callbacks {
                    events: RefCell::new(Vec::new()),
                    random: random.clone(),
                });
                let infos = FieldInfos::new(vec![FieldInfo::new("f", 0).with_doc_values(
                    kind,
                    DocValuesSkipIndexType::None,
                    -1,
                )])
                .unwrap();
                let first = Source {
                    index: 0,
                    fails: failing_source == 0,
                    calls: calls.clone(),
                };
                let second = Source {
                    index: 1,
                    fails: failing_source == 1,
                    calls: calls.clone(),
                };
                let sources = [
                    MergeSource {
                        producer: Some(&first),
                        field_infos: &infos,
                        max_doc: 1,
                        doc_map: &[0],
                        live_docs: None,
                    },
                    MergeSource {
                        producer: Some(&second),
                        field_infos: &infos,
                        max_doc: 1,
                        doc_map: &[1],
                        live_docs: None,
                    },
                ];
                let abort = || {
                    calls.record("abort");
                    Ok(())
                };
                let merge = MergeState {
                    sources: &sources,
                    field_infos: &infos,
                    check_aborted: &abort,
                };
                let mut consumer =
                    CrankyDocValuesConsumer::new(RequestingConsumer(calls.clone()), random.clone());
                let before = JavaRandom::new(seed).next_int();
                let mut expected_random = JavaRandom::new(seed);
                let fails_guard = expected_random.next_int_bounded(100) == 0;
                assert_eq!(fails_guard, seed == 18);
                let after = expected_random.next_int();
                let mut expected = vec![
                    format!("abort:{before}"),
                    format!("integrity:0:{before}"),
                    format!("abort:{before}"),
                    format!("integrity:1:{before}"),
                ];
                let error = consumer.merge(&merge).unwrap_err().to_string();
                if fails_guard {
                    assert_eq!(
                        error,
                        format!("Fake IOException from DocValuesConsumer.{operation}()")
                    );
                } else {
                    expected.push(format!("delegate:{name}:{after}"));
                    expected.push(format!("getter:0:{name}:{after}"));
                    if failing_source == 1 {
                        expected.push(format!("getter:1:{name}:{after}"));
                    }
                    assert_eq!(error, format!("source {failing_source} getter failure"));
                }
                assert_eq!(
                    *calls.events.borrow(),
                    expected,
                    "{name}, source {failing_source}, seed {seed}"
                );
                assert_eq!(
                    random.borrow_mut().next_int(),
                    after,
                    "getter failure must not consume another RNG draw"
                );
            }
        }
    }
}
