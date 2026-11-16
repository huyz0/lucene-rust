#![cfg(test)]
use super::*;

#[test]
fn shared_settings_own_custom_input_and_keep_distinct_stable_formats() {
    let codec = {
        let name = String::from("CustomOwnedData");
        let suffix = String::from("private_suffix_é");
        CompressingCodec::with_suffix(&name, &suffix, None, 37, 3, 22).unwrap()
    };
    assert_eq!(codec.name(), "CustomOwnedData");
    assert_eq!(
        codec.term_vectors_format().0.segment_suffix(),
        "private_suffix_é"
    );
    assert!(Rc::ptr_eq(
        &codec.stored_fields_format().0,
        &codec.term_vectors_format().0
    ));
    assert_ne!(
        codec.stored_fields_format() as *const _ as *const (),
        codec.term_vectors_format() as *const _ as *const (),
    );
    assert_eq!(
        codec.owned_string_bytes(),
        "CustomOwnedData".len() + "private_suffix_é".len()
    );
    let standard = CodecKind::Fast.defaults();
    assert!(codec.shared_settings_bytes() > standard.shared_settings_bytes());
    for kind in CodecKind::ALL {
        let codec = kind.configured(37, 3, true, 22).unwrap();
        assert_eq!(codec.owned_string_bytes(), 0);
        assert_eq!(
            codec.stored_fields_format().0.format_name(),
            kind.format_name()
        );
        assert_eq!(
            codec.term_vectors_format().0.segment_suffix(),
            kind.suffix()
        );
        if cfg!(target_pointer_width = "64") {
            let mode = if matches!(
                kind,
                CodecKind::DeflateWithPreset | CodecKind::Lz4WithPreset
            ) {
                let (layout, _) = std::alloc::Layout::new::<[usize; 2]>()
                    .extend(std::alloc::Layout::new::<CompressionMode>())
                    .unwrap();
                layout.pad_to_align().size()
            } else {
                0
            };
            let retained = std::mem::size_of_val(&codec)
                + std::mem::size_of_val(codec.stored_fields_format())
                + std::mem::size_of_val(codec.term_vectors_format())
                + codec.shared_settings_bytes()
                + mode;
            assert!(retained <= 112 + if mode == 0 { 0 } else { 16 });
        }
    }
}

#[test]
fn constructor_boundaries_null_mode_and_validation_precedence() {
    let empty = CompressingCodec::new("", None, 1, 1, 2).unwrap();
    assert_eq!(empty.name(), "");
    assert!(empty.kind().is_none());
    assert!(empty.stored_fields_format().0.compression_mode.is_none());
    assert!(empty.to_string().contains("compressionMode=null"));
    for name in ["a".repeat(127), "A09".into()] {
        let codec = CompressingCodec::with_suffix(
            &name,
            "unrestricted_suffix-é",
            None,
            i32::MAX,
            i32::MAX,
            22,
        )
        .unwrap();
        assert_eq!(codec.name(), name);
        assert_eq!(
            codec.term_vectors_format().0.segment_suffix(),
            "unrestricted_suffix-é"
        );
    }
    for name in ["a".repeat(128), "🦀".repeat(64)] {
        let error = CompressingCodec::new(&name, None, 0, 0, 0).unwrap_err();
        assert!(error
            .to_string()
            .ends_with("is too long (must be < 128 chars)."));
    }
    let error = CompressingCodec::new("bad-name", None, 0, 0, 0).unwrap_err();
    assert_eq!(
        error.to_string(),
        "Illegal service name: 'bad-name' must be simple ascii alphanumeric."
    );
    for (chunk, docs, shift, message) in [
        (0, 0, 0, "chunkSize must be >= 1"),
        (1, 0, 0, "maxDocsPerChunk must be >= 1"),
        (1, 1, 1, "blockSize must be in 2-22, got 1"),
    ] {
        let error = CompressingCodec::new("valid", None, chunk, docs, shift).unwrap_err();
        assert_eq!(error.to_string(), message);
    }
}
#[test]
fn selected_objects_and_mode_identities_survive_repeated_access() {
    for kind in CodecKind::ALL {
        let codec = kind.defaults();
        let second = kind.defaults();
        assert!(std::ptr::eq(
            codec.stored_fields_format(),
            codec.stored_fields_format()
        ));
        assert!(std::ptr::eq(
            codec.term_vectors_format(),
            codec.term_vectors_format()
        ));
        let a = codec
            .stored_fields_format()
            .0
            .compression_mode
            .as_ref()
            .unwrap();
        let b = codec
            .term_vectors_format()
            .0
            .compression_mode
            .as_ref()
            .unwrap();
        let c = second
            .stored_fields_format()
            .0
            .compression_mode
            .as_ref()
            .unwrap();
        assert!(a.same_instance(b));
        assert_eq!(
            a.same_instance(c),
            !matches!(
                kind,
                CodecKind::DeflateWithPreset | CodecKind::Lz4WithPreset
            )
        );
        assert_eq!(
            a.mode().to_string(),
            codec.stored_fields_format().0.mode_display()
        );
        assert_eq!(codec.kind(), Some(kind));
        assert!(!kind.class_name().is_empty());
    }
    let a = ModeHandle::new(CompressionMode::Fast);
    let b = ModeHandle::new(CompressionMode::Dummy);
    assert!(!a.same_instance(&b));
}
#[test]
fn actual_random_source_matches_direct_java_random_calls() {
    let mut random = JavaRandom::new(42);
    let mut control = JavaRandom::new(42);
    assert_eq!(
        RandomSource::next_boolean(&mut random),
        control.next_boolean()
    );
    assert_eq!(
        RandomSource::next_int_bounded(&mut random, 6),
        control.next_int_bounded(6)
    );
    let codec = CompressingCodec::random_instance(&mut random).unwrap();
    assert!(codec.stored_fields_format().0.chunk_size >= 10);
    let codec = CompressingCodec::reasonable_instance(&mut random).unwrap();
    assert!((8192..=131072).contains(&codec.stored_fields_format().0.chunk_size));
    match CompressingCodec::random_with_suffix(&mut random, true) {
        Ok(codec) => assert!(!codec.stored_fields_format().0.segment_suffix().is_empty()),
        Err(error) => assert!(error
            .to_string()
            .starts_with("blockSize must be in 2-22, got ")),
    }
}
#[test]
fn callback_identity_and_errors_pass_through_without_touching_selected_formats() {
    struct Delegate {
        handle: Rc<()>,
        error: Rc<str>,
        calls: std::cell::Cell<usize>,
    }
    impl CodecDelegate for Delegate {
        type Component = Rc<()>;
        type Error = Rc<str>;
        fn component(&self, kind: ComponentKind) -> Result<Self::Component, Self::Error> {
            self.calls.set(self.calls.get() + 1);
            if kind == ComponentKind::Points {
                Err(self.error.clone())
            } else {
                Ok(self.handle.clone())
            }
        }
    }
    let codec = CodecKind::Fast.defaults();
    let stored = codec.stored_fields_format() as *const _;
    let vectors = codec.term_vectors_format() as *const _;
    let codec = codec.map_delegate(|_| Delegate {
        handle: Rc::new(()),
        error: Rc::from("original error"),
        calls: std::cell::Cell::new(0),
    });
    assert_eq!(stored, codec.stored_fields_format() as *const _);
    assert_eq!(vectors, codec.term_vectors_format() as *const _);
    let handle = codec.component(ComponentKind::DocValues).unwrap();
    assert!(Rc::ptr_eq(&handle, &codec.delegate().handle));
    let error = codec.component(ComponentKind::Points).unwrap_err();
    assert!(Rc::ptr_eq(&error, &codec.delegate().error));
    assert_eq!(codec.delegate().calls.get(), 2);
    assert!(codec.to_string().starts_with(codec.name()));
    assert_eq!(codec.delegate().calls.get(), 2);
}
#[test]
fn all_selector_arms_precede_validation() {
    struct Selected {
        value: i32,
        calls: usize,
    }
    impl RandomSource for Selected {
        fn next_boolean(&mut self) -> bool {
            panic!("explicit overload must not draw a boolean")
        }
        fn next_int_bounded(&mut self, bound: i32) -> i32 {
            assert_eq!(bound, 6);
            self.calls += 1;
            self.value
        }
    }
    for (value, kind) in (0..6).zip(CodecKind::ALL) {
        let mut random = Selected { value, calls: 0 };
        let codec = CompressingCodec::random_explicit(&mut random, 37, 3, true, 10).unwrap();
        assert_eq!(codec.kind(), Some(kind));
        let error = CompressingCodec::random_explicit(&mut random, 0, 0, true, 0).unwrap_err();
        assert_eq!(error.to_string(), "chunkSize must be >= 1");
        assert_eq!(random.calls, 2);
    }
}
