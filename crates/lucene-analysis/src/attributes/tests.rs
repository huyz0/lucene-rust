#![cfg(test)]
use super::*;

fn dirty() -> AttributeSource {
    let mut a = AttributeSource::new();
    a.set_term("Foo");
    a.set_bytes_term(Some(vec![1, 2]));
    a.set_offset(3, 7).unwrap();
    a.set_position_increment(4).unwrap();
    a.set_position_length(2).unwrap();
    a.set_token_type("<NUM>");
    a.set_flags(9);
    a.set_keyword(true);
    a.set_payload(Some(vec![5]));
    a.set_term_frequency(6).unwrap();
    a.set_sentence_index(8);
    a
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct Tags(Option<Vec<String>>);

impl CustomAttribute for Tags {
    fn clear(&mut self) {
        self.0 = None;
    }
    fn reflect(&self, r: &mut dyn FnMut(&'static str, &'static str, AttrValue<'_>)) {
        let tags = self
            .0
            .as_ref()
            .map(|t| t.iter().map(|s| Cow::Borrowed(s.as_str())).collect());
        r("x.Tags", "tags", AttrValue::List(tags));
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct Sticky(u32);

impl CustomAttribute for Sticky {
    fn clear(&mut self) {
        self.0 = 0;
    }
    fn reflect(&self, r: &mut dyn FnMut(&'static str, &'static str, AttrValue<'_>)) {
        r("x.Sticky", "n", AttrValue::Int(self.0 as i32));
        r("x.Sticky", "s", AttrValue::Str(None));
    }
    fn end(&mut self) {
        self.0 = 99;
    }
}

#[test]
fn clearing_parks_the_binary_term_buffer_for_the_next_one() {
    let mut a = AttributeSource::new();
    a.bytes_term_mut().extend_from_slice(&[1u8; 100]);
    let capacity = a.bytes_term_mut().capacity();
    a.clear_attributes();
    // unset again, and equal to a source that never had one
    assert_eq!(a.bytes_term(), None);
    assert_eq!(a, AttributeSource::new());
    // the buffer comes back empty with its capacity
    let b = a.bytes_term_mut();
    assert!(b.is_empty() && b.capacity() == capacity);
    b.push(7);
    assert_eq!(a.bytes_term(), Some(&[7u8][..]));
    // a copy sets, and a copy of an unset source unsets, the term
    let mut c = AttributeSource::new();
    c.clone_from(&a);
    assert_eq!(c.bytes_term(), Some(&[7u8][..]));
    c.clone_from(&AttributeSource::new());
    assert_eq!(c.bytes_term(), None);
    a.set_bytes_term(None);
    assert_eq!(a.bytes_term(), None);
}

/// Restoring with a term is restoring, then setting the term.
#[test]
fn restore_state_with_term_is_restore_then_set_term() {
    let mut state = dirty();
    state.add_custom::<Tags>().0 = Some(vec!["t".into()]);
    state.set_token_type(String::from("<OWNED>"));
    for start in [AttributeSource::new(), dirty()] {
        let mut want = start.clone();
        want.restore_state(&state);
        want.set_term("gram");
        let mut got = start;
        got.restore_state_with_term(&state, "gram");
        assert_eq!(got, want);
        // again, over its own buffers and the same (static) type
        got.set_token_type(DEFAULT_TYPE);
        want.set_token_type(DEFAULT_TYPE);
        let other = got.clone();
        got.restore_state_with_term(&other, "x");
        want.set_term("x");
        assert_eq!(got, want);
    }
}

#[test]
fn custom_attributes_follow_the_lifecycle() {
    let mut a = AttributeSource::new();
    assert!(a.custom::<Tags>().is_none());
    a.add_custom::<Tags>().0 = Some(vec!["x".into()]);
    assert_eq!(a.custom::<Tags>(), Some(&Tags(Some(vec!["x".into()]))));
    let state = a.capture_state();
    a.add_custom::<Tags>().0 = None;
    a.add_custom::<Sticky>().0 = 5;
    assert_ne!(a, state);
    a.restore_state(&state);
    assert_eq!(a.custom::<Tags>(), state.custom::<Tags>());
    assert_eq!(a.custom::<Sticky>(), Some(&Sticky(5)));
    // A state holding an attribute this source lacks adds it; Java's
    // check refuses it.
    let mut b = AttributeSource::new();
    let e = b.try_restore_state(&state).unwrap_err().to_string();
    assert!(
        e.contains("State contains AttributeImpl of type") && e.contains("Tags"),
        "{e}"
    );
    assert!(b.custom::<Tags>().is_none());
    // Java copies the state's attributes in order up to the missing one.
    let mut d = dirty();
    d.add_custom::<Sticky>().0 = 7;
    d.add_custom::<Tags>().0 = Some(vec!["t".into()]);
    let partial = d.capture_state();
    let mut target = AttributeSource::new();
    target.add_custom::<Sticky>();
    assert!(target.try_restore_state(&partial).is_err());
    assert_eq!(target.term(), "Foo");
    assert_eq!(target.custom::<Sticky>(), Some(&Sticky(7)));
    assert!(target.custom::<Tags>().is_none());
    b.restore_state(&state);
    assert_eq!(b.custom::<Tags>(), state.custom::<Tags>());
    b.try_restore_state(&state).unwrap();
    // clone_from is an exact copy: an attribute the source lacks goes.
    let mut c = a.clone();
    c.clone_from(&state);
    assert_eq!(c, state);
    assert!(c.custom::<Sticky>().is_none());
    c.clone_from(&state);
    assert_eq!(c, state);
    a.clear_attributes();
    assert_eq!(a.custom::<Tags>(), Some(&Tags(None)));
    a.end_attributes();
    assert_eq!(a.custom::<Sticky>(), Some(&Sticky(99)));
    assert_eq!(a.clone(), a);
    assert!(format!("{a:?}").contains("Sticky"));
}

#[test]
fn reflection_lists_core_then_custom_attributes() {
    let mut a = dirty();
    a.add_custom::<Sticky>().0 = 3;
    a.add_custom::<Tags>().0 = Some(vec!["a".into(), "b".into()]);
    assert_eq!(
        a.reflect_as_string(false),
        "term=Foo,bytes=[1 2],startOffset=3,endOffset=7,positionIncrement=4,positionLength=2,\
             type=<NUM>,termFrequency=6,flags=9,keyword=true,payload=[5],sentences=8,boost=1.0,\
             n=3,s=null,tags=[a, b]"
    );
    let mut custom = Vec::new();
    a.reflect_custom(&mut |c, k, v| custom.push(format!("{c}#{k}={v}")));
    assert_eq!(
        custom,
        ["x.Sticky#n=3", "x.Sticky#s=null", "x.Tags#tags=[a, b]"]
    );
    let fresh = AttributeSource::new().reflect_as_string(true);
    assert!(
        fresh.starts_with("org.apache.lucene.analysis.tokenattributes.CharTermAttribute#term=,")
    );
    assert!(fresh.contains("PayloadAttribute#payload=null"), "{fresh}");
    a.set_boost(2.5);
    a.add_custom::<Tags>().0 = None;
    let s = a.reflect_as_string(false);
    assert!(s.contains("boost=2.5") && s.ends_with("tags=null"), "{s}");
    assert_eq!(AttrValue::Float(1e8).to_string(), "1.0E8");
    assert!(CustomAttribute::impl_class(&Sticky(0)).ends_with("Sticky"));
}

#[test]
fn float_values_print_as_java_float_to_string() {
    // `Float.toString(Float.intBitsToFloat(bits))` under JDK 21 and 25
    // (identical); the same rule matched 3,070,588 values -- every
    // pattern below 70,000, powers of ten +-3 ulps and three million
    // seeded random patterns -- when it was written.
    for (bits, java) in [
        (0x00000000, "0.0"),
        (0x80000000, "-0.0"),
        (0x00000001, "1.4E-45"),
        (0x00000002, "2.8E-45"),
        (0x00000003, "4.2E-45"),
        (0x00000010, "2.2E-44"),
        (0x000fffff, "1.469367E-39"),
        (0x00800000, "1.1754944E-38"),
        (0x7f7fffff, "3.4028235E38"),
        (0x7f800000, "Infinity"),
        (0xff800000, "-Infinity"),
        (0x7fc00000, "NaN"),
        (0x3a83126e, "9.999999E-4"),
        (0x3a83126f, "0.001"),
        (0x3a831270, "0.0010000002"),
        (0x4b18967f, "9999999.0"),
        (0x4b189680, "1.0E7"),
        (0x4b189681, "1.0000001E7"),
        (0x3727c5ac, "1.0E-5"),
        (0x501502f9, "1.0E10"),
        (0x4cbebc20, "1.0E8"),
        (0x3f800000, "1.0"),
        (0x40200000, "2.5"),
        (0x3dcccccd, "0.1"),
        (0x42c80000, "100.0"),
        (0xbf000000, "-0.5"),
        (0x3c23d70a, "0.01"),
        (0x47f12065, "123456.79"),
        (0x7f7fc99e, "3.4E38"),
        (0xc88469f4, "-271183.62"),
        (0x3dcccccd, "0.1"),
        (0x3f7fffff, "0.99999994"),
        (0xbed4fc71, "-0.41598848"),
        (0x4b189680, "1.0E7"),
        (0x3a83126f, "0.001"),
    ] {
        let v = f32::from_bits(bits);
        assert_eq!(java_float_to_string(v), java, "{bits:#x}");
        assert_eq!(AttrValue::Float(v).to_string(), java);
    }
    assert_eq!(split_scientific("7"), (b"7".to_vec(), 0));
}

/// The fast path agrees with the exact expansion: for each value, the digits Java's rule
/// picks from the exact expansion, printed the same way.
#[test]
fn fast_float_digits_are_used_only_where_they_are_javas() {
    fn exact(v: f32) -> String {
        let a = v.abs();
        let (digits, _) = split_scientific(&format!("{a:e}"));
        let (mut d, e) = closest_digits(a, digits.len().max(2)).unwrap();
        while d.len() > 1 && d.last() == Some(&b'0') {
            d.pop();
        }
        // Render through the fast function's layout with the exact digits.
        let text = format!(
            "{}.{}e{e}",
            char::from(d[0]),
            std::str::from_utf8(&d[1..]).unwrap()
        );
        let back: f32 = text.parse().unwrap();
        assert_eq!(back, a, "{text}");
        String::from_utf8(d).unwrap()
    }
    let digits_of = |s: &str| -> String {
        let m = s.split('E').next().unwrap().trim_start_matches('-');
        let d: String = m.chars().filter(char::is_ascii_digit).collect();
        let d = d.trim_start_matches('0').trim_end_matches('0');
        if d.is_empty() {
            "0".into()
        } else {
            d.into()
        }
    };
    let mut x = 0x9e37_79b9_7f4a_7c15u64;
    let check = |bits: u32| {
        let v = f32::from_bits(bits);
        if !v.is_finite() || v == 0.0 {
            return;
        }
        let fast = java_float_to_string(v);
        let want = exact(v);
        let want = want.trim_end_matches('0');
        assert_eq!(
            digits_of(&fast),
            if want.is_empty() { "0" } else { want },
            "{bits:#x} {fast}"
        );
        assert_eq!(fast.parse::<f32>().unwrap(), v, "{bits:#x} {fast}");
    };
    for _ in 0..300_000 {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        check((x >> 32) as u32);
    }
    // Values from vectors: uniform in [-1, 1] at six decimals, and every float near 1.
    for i in 0..200_000u32 {
        check(((f64::from(i) / 100_000.0 - 1.0) as f32).to_bits());
        check(0x3f80_0000 - 100_000 + i);
    }
    // Subnormals and ties.
    for bits in (1..20_000).chain([0x4884_69f4, 0xc884_69f4]) {
        check(bits);
    }
}

#[test]
fn clear_restores_every_java_clear_value() {
    let mut a = dirty();
    a.clear_attributes();
    assert_eq!(a, AttributeSource::new());
    assert_eq!(a.token_type(), "word");
    assert_eq!(a.position_increment(), 1);
    assert_eq!(a.term_frequency(), 1);
}

#[test]
fn end_is_clear_with_a_zero_increment() {
    let mut a = dirty();
    a.end_attributes();
    let mut want = AttributeSource::new();
    want.position_increment = 0;
    assert_eq!(a, want);
}

#[test]
fn capture_and_restore_round_trip() {
    let a = dirty();
    let s = a.capture_state();
    let mut b = AttributeSource::new();
    b.restore_state(&s);
    assert_eq!(a, b);
    assert_eq!(b.payload(), Some(&[5u8][..]));
    assert_eq!(b.bytes_term(), Some(&[1u8, 2][..]));
    assert_eq!(b.flags(), 9);
    assert!(b.is_keyword());
    assert_eq!(b.sentence_index(), 8);
    assert_eq!(b.position_length(), 2);
    assert_eq!((b.start_offset(), b.end_offset()), (3, 7));
}

#[test]
fn term_bytes_prefers_the_binary_term() {
    let mut a = AttributeSource::new();
    a.set_term("é");
    assert_eq!(a.term_bytes(), "é".as_bytes());
    a.set_bytes_term(Some(vec![0xff]));
    assert_eq!(a.term_bytes(), &[0xff]);
    a.set_bytes_term(None);
    assert_eq!(a.term_bytes(), "é".as_bytes());
}

#[test]
fn setters_validate_like_java() {
    let mut a = AttributeSource::new();
    assert!(a.set_offset(-1, 0).is_err());
    assert!(a.set_offset(5, 4).is_err());
    assert!(a.set_position_increment(-1).is_err());
    assert!(a.set_position_length(0).is_err());
    assert!(a.set_term_frequency(0).is_err());
    assert!(a.set_position_increment(0).is_ok());
    let msg = a.set_position_length(0).unwrap_err().to_string();
    assert!(
        msg.contains("Position length must be 1 or greater; got 0"),
        "{msg}"
    );
}

#[test]
fn utf16_terms_decode_and_replace_lone_surrogates() {
    let mut a = AttributeSource::new();
    a.set_term_utf16(&[b'a' as u16, b'b' as u16]);
    assert_eq!(a.term(), "ab");
    a.set_term_utf16(&[0xD83D, 0xDE00, 0x00E9]);
    assert_eq!(a.term(), "😀é");
    assert_eq!(a.term_utf16_len(), 3);
    a.set_term_utf16(&[0xD800, b'x' as u16]);
    assert_eq!(a.term(), "\u{FFFD}x");
    a.term_mut().push('y');
    assert_eq!(a.term(), "\u{FFFD}xy");
}
