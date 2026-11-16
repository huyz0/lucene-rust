#![cfg(test)]
use super::*;

const SIMPLETEXT: &[u8] = include_bytes!("../../../../fixtures/data/simpletext_index_sort/_1.si");
const SIMPLETEXT_ID: [u8; ID_LENGTH] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

proptest::proptest! {
    #[test]
    fn simpletext_sort_round_trips_arbitrary_names_and_numeric_sentinels(
        name in proptest::collection::vec(proptest::prelude::any::<char>(), 0..25),
        reverse in proptest::prelude::any::<bool>(),
        missing in proptest::option::of(proptest::prelude::any::<i64>()),
    ) {
        let mut info = parse_simpletext(SIMPLETEXT, "_1", &SIMPLETEXT_ID).unwrap();
        info.index_sort = vec![IndexSortField::long(name.into_iter().collect::<String>(), reverse, missing)];
        let bytes = write_simpletext(&mut info).unwrap();
        let read = parse_simpletext(&bytes, "_1", &SIMPLETEXT_ID).unwrap();
        proptest::prop_assert_eq!(read, info);
    }
}

#[test]
fn simpletext_sort_bytes_determine_the_type_and_must_be_fully_consumed() {
    use lucene_codecs::simpletext::segment_info as simpletext;
    let mut raw = simpletext::read(SIMPLETEXT, "_1", &SIMPLETEXT_ID).unwrap();
    raw.index_sort[0].description = "a description the reader must ignore".into();
    let bytes = simpletext::write(&mut raw).unwrap();
    let typed = parse_simpletext(&bytes, "_1", &SIMPLETEXT_ID).unwrap();
    assert!(matches!(
        typed.index_sort[0].kind,
        IndexSortKind::Numeric(NumericSortKey::Int(None))
    ));

    raw.index_sort[0].bytes.push(0);
    let bytes = simpletext::write(&mut raw).unwrap();
    assert!(matches!(
        parse_simpletext(&bytes, "_1", &SIMPLETEXT_ID),
        Err(Error::Store(lucene_store::Error::Corrupted(message)))
            if message.contains("trailing bytes")
    ));
    raw.index_sort[0].bytes.clear();
    let bytes = simpletext::write(&mut raw).unwrap();
    assert!(matches!(
        parse_simpletext(&bytes, "_1", &SIMPLETEXT_ID),
        Err(Error::Store(lucene_store::Error::Eof { .. }))
    ));
    assert!(parse_simpletext(SIMPLETEXT, "_1", &[0; ID_LENGTH]).is_err());
    assert!(parse_simpletext(&SIMPLETEXT[..20], "_1", &SIMPLETEXT_ID).is_err());
}

#[test]
fn simpletext_writer_adds_its_file_once_and_preserves_prerelease_versions() {
    let mut info = parse_simpletext(SIMPLETEXT, "_1", &SIMPLETEXT_ID).unwrap();
    info.files.retain(|file| file != "_1.si");
    let bytes = write_simpletext(&mut info).unwrap();
    assert_eq!(info.files.iter().filter(|file| *file == "_1.si").count(), 1);
    assert_eq!(write_simpletext(&mut info).unwrap(), bytes);
    info.version = lucene_util::version::Version::new(10, 0, 0, 2).unwrap();
    info.min_version = Some(lucene_util::version::Version::new(9, 0, 0, 1).unwrap());
    let bytes = write_simpletext(&mut info).unwrap();
    let read = parse_simpletext(&bytes, "_1", &SIMPLETEXT_ID).unwrap();
    assert_eq!(read.version.prerelease(), 2);
    assert_eq!(read.min_version.unwrap().prerelease(), 1);
    info.name = "bad segment".into();
    assert!(write_simpletext(&mut info).is_err());
}

#[test]
fn writer_diagnostics_are_javas_keys_then_the_details() {
    let v = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let d = writer_diagnostics(SOURCE_MERGE, v, &[("mergeFactor", "3".to_string())]);
    let keys: Vec<&str> = d.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(
        keys,
        [
            "source",
            "lucene.version",
            "os",
            "os.arch",
            "os.version",
            "java.runtime.version",
            "java.vendor",
            "timestamp",
            "mergeFactor"
        ]
    );
    assert_eq!(d[0].1, "merge");
    assert_eq!(d[1].1, "10.5.0");
    assert_eq!(d[6].1, "lucene-rust");
    assert!(d[7].1.parse::<u64>().unwrap() > 1_600_000_000_000);
    assert_eq!(d[8].1, "3");
    assert!(!d[4].1.is_empty());
}

#[test]
fn os_name_and_arch_are_the_jvms_spelling() {
    for (os, name) in [
        ("linux", "Linux"),
        ("macos", "Mac OS X"),
        ("windows", "Windows"),
        ("freebsd", "FreeBSD"),
        ("solaris", "solaris"),
    ] {
        assert_eq!(java_os_name(os), name);
    }
    assert_eq!(java_os_arch("linux", "x86_64"), "amd64");
    assert_eq!(java_os_arch("windows", "x86_64"), "amd64");
    assert_eq!(java_os_arch("macos", "x86_64"), "x86_64");
    assert_eq!(java_os_arch("linux", "x86"), "x86");
    assert_eq!(java_os_arch("linux", "aarch64"), "aarch64");
}

/// A 9.0.0 segment (`Lucene90` codec, `Lucene90SegmentInfoFormat`) reads
/// only by its codec: the same bytes through `Lucene99SegmentInfoFormat`
/// take the first diagnostics byte for `hasBlocks` and misread the rest.
#[test]
fn a_lucene90_si_is_read_by_its_codec_name() {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/data/bwc/9.0.0/_0.si");
    let bytes = std::fs::read(path).unwrap();
    // The id the index header carries, after magic, name and version.
    let name_len = bytes[4] as usize;
    let mut id = [0u8; ID_LENGTH];
    id.copy_from_slice(&bytes[9 + name_len..9 + name_len + ID_LENGTH]);
    let si = parse_for_codec(&bytes, &id, "Lucene90").unwrap();
    assert_eq!(si.doc_count, 3000);
    assert!(!si.has_blocks);
    assert!(si.files.iter().any(|f| f == "_0.fnm"));
    assert_eq!(
        si.version,
        LuceneVersion {
            major: 9,
            minor: 0,
            bugfix: 0
        }
    );
    let misread = parse_for_codec(&bytes, &id, "Lucene99");
    assert!(!misread.is_ok_and(|s| s.files == si.files));
    assert!(matches!(
        parse_for_codec(&bytes, &id, "Lucene87"),
        Err(Error::UnknownCodec(n)) if n == "Lucene87"
    ));
}

/// Test-only `.si` byte builder: independent of the Java fixtures under
/// `tests/segment_info_fixtures.rs` (which exercise real Lucene-written
/// bytes) — this covers the parser's own corruption/error handling, which
/// needs deliberately-invalid inputs a real Lucene codec would never write.
struct SiBuilder {
    id: [u8; ID_LENGTH],
    has_min_version: u8,
    doc_count: i32,
    is_compound_file: u8,
    has_blocks: u8,
    num_sort_fields: i32,
    /// Raw bytes for the single sort field's payload, written verbatim
    /// after `num_sort_fields` when it's `1` -- lets tests hand-build
    /// both well-formed and deliberately-corrupt sort-field encodings.
    sort_field_bytes: Vec<u8>,
}

impl SiBuilder {
    fn valid() -> Self {
        Self {
            id: [1u8; ID_LENGTH],
            has_min_version: 0,
            doc_count: 5,
            is_compound_file: 1,
            has_blocks: 0,
            num_sort_fields: 0,
            sort_field_bytes: Vec::new(),
        }
    }

    /// The bytes real Lucene's `SortField.Provider` writes for a
    /// single-valued sort: provider name, field, `SortField.Type` enum
    /// name, reverse flag, missing-present flag, and the missing value in
    /// that type's own encoding.
    fn plain_sort_field_bytes(
        field: &str,
        type_name: &str,
        reverse: i32,
        missing: Option<&[u8]>,
    ) -> Vec<u8> {
        let mut b = Vec::new();
        write_string(&mut b, PROVIDER_SORT_FIELD);
        write_string(&mut b, field);
        write_string(&mut b, type_name);
        b.extend_from_slice(&reverse.to_le_bytes());
        match missing {
            Some(bytes) => {
                b.extend_from_slice(&1i32.to_le_bytes());
                b.extend_from_slice(bytes);
            }
            None => b.extend_from_slice(&0i32.to_le_bytes()),
        }
        b
    }

    /// A LONG sort field with the sort-first/sort-last sentinel this
    /// port's own writer emits.
    fn long_sort_field_bytes(field: &str, reverse: bool, missing_last: bool) -> Vec<u8> {
        let sentinel = if missing_last { i64::MAX } else { i64::MIN };
        Self::plain_sort_field_bytes(
            field,
            TYPE_LONG,
            if reverse { 1 } else { 0 },
            Some(&sentinel.to_le_bytes()),
        )
    }

    fn build(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
        write_string(&mut out, CODEC_NAME);
        out.extend_from_slice(&(VERSION_CURRENT as u32).to_be_bytes());
        out.extend_from_slice(&self.id);
        out.push(0); // empty suffix

        out.extend_from_slice(&10i32.to_le_bytes()); // version major
        out.extend_from_slice(&0i32.to_le_bytes()); // minor
        out.extend_from_slice(&0i32.to_le_bytes()); // bugfix
        out.push(self.has_min_version);
        if self.has_min_version == 1 {
            out.extend_from_slice(&9i32.to_le_bytes());
            out.extend_from_slice(&0i32.to_le_bytes());
            out.extend_from_slice(&0i32.to_le_bytes());
        }
        out.extend_from_slice(&self.doc_count.to_le_bytes());
        out.push(self.is_compound_file);
        out.push(self.has_blocks);
        write_vint(&mut out, 0); // diagnostics: empty map
        write_vint(&mut out, 0); // files: empty set
        write_vint(&mut out, 0); // attributes: empty map
        write_vint(&mut out, self.num_sort_fields);
        out.extend_from_slice(&self.sort_field_bytes);

        out.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
        out.extend_from_slice(&0u32.to_be_bytes());
        let checksum = crc32fast::hash(&out) as u64;
        out.extend_from_slice(&checksum.to_be_bytes());
        out
    }
}

fn write_vint(out: &mut Vec<u8>, mut v: i32) {
    loop {
        let mut b = (v & 0x7f) as u8;
        v = ((v as u32) >> 7) as i32;
        if v != 0 {
            b |= 0x80;
            out.push(b);
        } else {
            out.push(b);
            break;
        }
    }
}

fn write_string(out: &mut Vec<u8>, s: &str) {
    write_vint(out, s.len() as i32);
    out.extend_from_slice(s.as_bytes());
}

#[test]
fn valid_segment_info_parses() {
    let b = SiBuilder::valid();
    let si = parse(&b.build(), &b.id).unwrap();
    assert_eq!(si.doc_count, 5);
    assert!(si.is_compound_file);
    assert!(!si.has_blocks);
    assert!(si.min_version.is_none());
}

#[test]
fn min_version_present_is_parsed() {
    let mut b = SiBuilder::valid();
    b.has_min_version = 1;
    let si = parse(&b.build(), &b.id).unwrap();
    let mv = si.min_version.unwrap();
    assert_eq!((mv.major, mv.minor, mv.bugfix), (9, 0, 0));
}

#[test]
fn illegal_has_min_version_byte_rejected() {
    let b = SiBuilder::valid();
    let mut bytes = b.build();
    // has_min_version byte sits right after the 3 SegVersion i32s (12 bytes)
    // in the payload, following the fixed-size index header.
    let header_len =
        codec_util::CODEC_MAGIC.to_be_bytes().len() + 1 + CODEC_NAME.len() + 4 + ID_LENGTH + 1;
    let has_min_version_offset = header_len + 12;
    bytes[has_min_version_offset] = 7; // neither 0 nor 1
    assert!(matches!(
        parse(&bytes, &b.id),
        Err(Error::IllegalHasMinVersion(7))
    ));
}

#[test]
fn negative_doc_count_rejected() {
    let mut b = SiBuilder::valid();
    b.doc_count = -1;
    assert!(matches!(
        parse(&b.build(), &b.id),
        Err(Error::InvalidDocCount(-1))
    ));
}

/// A `SortField(field, LONG, reverse)` with the `Long.MIN_VALUE`/
/// `Long.MAX_VALUE` sentinel this port's own sorted writers emit.
fn long_sort(field: &str, reverse: bool, missing_last: bool) -> IndexSortField {
    IndexSortField::long(
        field,
        reverse,
        Some(if missing_last { i64::MAX } else { i64::MIN }),
    )
}

#[test]
fn two_field_sort_count_parses_both_fields() {
    let mut b = SiBuilder::valid();
    b.num_sort_fields = 2;
    let mut bytes = SiBuilder::long_sort_field_bytes("price", false, true);
    bytes.extend(SiBuilder::long_sort_field_bytes("timestamp", true, false));
    b.sort_field_bytes = bytes;
    let si = parse(&b.build(), &b.id).unwrap();
    let fields = si.index_sort.unwrap();
    assert_eq!(
        fields,
        vec![
            long_sort("price", false, true),
            long_sort("timestamp", true, false)
        ]
    );
}

#[test]
fn three_field_sort_round_trips_with_mixed_policies() {
    let mut si = sample_si();
    si.index_sort = Some(vec![
        long_sort("a", false, false),
        long_sort("b", true, true),
        long_sort("c", false, true),
    ]);
    let bytes = write(&si, "");
    let parsed = parse(&bytes, &si.id).unwrap();
    let fields = parsed.index_sort.unwrap();
    assert_eq!(fields, si.index_sort.unwrap());
}

#[test]
fn negative_sort_field_count_rejected() {
    let mut b = SiBuilder::valid();
    b.num_sort_fields = -1;
    assert!(matches!(
        parse(&b.build(), &b.id),
        Err(Error::InvalidSortFieldCount(-1))
    ));
}

/// `numSortFields` is a vint that sized `Vec<IndexSortField>` — a
/// multi-`String` struct — with no bound at all. `i32::MAX` of them is a
/// hundreds-of-gigabytes reservation, and an allocation failure is an
/// **abort**: `catch_unwind` cannot intercept it, so it takes the JVM down
/// through the FFI rather than surfacing as a corrupt-index error. Java
/// allocates the same unbounded `SortField[]` but gets a catchable
/// `OutOfMemoryError`.
#[test]
fn absurd_sort_field_count_errors_instead_of_reserving_for_it() {
    let mut b = SiBuilder::valid();
    b.num_sort_fields = i32::MAX;
    assert!(matches!(
        parse(&b.build(), &b.id),
        Err(Error::InvalidSortFieldCount(i32::MAX))
    ));
}

#[test]
fn single_numeric_sort_field_parses() {
    let mut b = SiBuilder::valid();
    b.num_sort_fields = 1;
    b.sort_field_bytes = SiBuilder::long_sort_field_bytes("price", false, true);
    let si = parse(&b.build(), &b.id).unwrap();
    assert_eq!(
        si.index_sort.unwrap(),
        vec![long_sort("price", false, true)]
    );
}

#[test]
fn unknown_sort_field_type_name_rejected() {
    let mut b = SiBuilder::valid();
    b.num_sort_fields = 1;
    b.sort_field_bytes =
        SiBuilder::plain_sort_field_bytes("price", "NOT_A_TYPE", 0, Some(&0i64.to_le_bytes()));
    assert!(matches!(
        parse(&b.build(), &b.id),
        Err(Error::UnknownSortFieldType(t)) if t == "NOT_A_TYPE"
    ));
}

#[test]
fn unknown_sort_field_provider_rejected() {
    let mut b = SiBuilder::valid();
    b.num_sort_fields = 1;
    let mut bytes = Vec::new();
    write_string(&mut bytes, "MyCustomSortField");
    b.sort_field_bytes = bytes;
    assert!(matches!(
        parse(&b.build(), &b.id),
        Err(Error::UnknownSortFieldProvider(p)) if p == "MyCustomSortField"
    ));
}

/// A `reverse` int other than 1 means "ascending" in every one of Java's
/// own `readSortField` implementations (`in.readInt() == 1`) -- we must
/// not be stricter than the format and reject a file Java reads.
#[test]
fn non_one_reverse_int_reads_as_ascending() {
    let mut b = SiBuilder::valid();
    b.num_sort_fields = 1;
    b.sort_field_bytes =
        SiBuilder::plain_sort_field_bytes("price", TYPE_LONG, 7, Some(&i64::MIN.to_le_bytes()));
    let si = parse(&b.build(), &b.id).unwrap();
    assert!(!si.index_sort.unwrap()[0].reverse);
}

/// **Read direction.** A numeric sort with *no* missing
/// value is what `new SortField("f", Type.LONG)` produces when nobody calls
/// `setMissingValue` -- the most ordinary sort there is. Java sorts such a
/// document as if it held `0`.
#[test]
fn a_numeric_sort_with_no_missing_value_parses_and_compares_as_zero() {
    let mut b = SiBuilder::valid();
    b.num_sort_fields = 1;
    b.sort_field_bytes = SiBuilder::plain_sort_field_bytes("price", TYPE_LONG, 0, None);
    let sf = &parse(&b.build(), &b.id).unwrap().index_sort.unwrap()[0];
    assert_eq!(sf.kind, IndexSortKind::Numeric(NumericSortKey::Long(None)));
    let cmp = SortKeyComparator::new(sf);
    // A missing document compares as 0: above -1, below 1.
    assert_eq!(cmp.compare(None, Some(-1)), std::cmp::Ordering::Greater);
    assert_eq!(cmp.compare(None, Some(1)), std::cmp::Ordering::Less);
    assert_eq!(cmp.compare(None, Some(0)), std::cmp::Ordering::Equal);
}

/// An arbitrary numeric missing value (Java allows any long) is carried
/// through, sentinel and all, since `(first|last)` cannot hold it.
#[test]
fn an_arbitrary_numeric_missing_value_round_trips() {
    let mut si = sample_si();
    si.index_sort = Some(vec![IndexSortField::long("price", true, Some(42))]);
    let bytes = write(&si, "");
    let parsed = parse(&bytes, &si.id).unwrap();
    assert_eq!(parsed.index_sort, si.index_sort);
    let sf = &parsed.index_sort.unwrap()[0];
    let cmp = SortKeyComparator::new(sf);
    // Descending, so a missing (42) document sorts *before* 100 and
    // *after* 7 -- the sentinel is compared like any other value.
    assert_eq!(cmp.compare(None, Some(100)), std::cmp::Ordering::Greater);
    assert_eq!(cmp.compare(None, Some(7)), std::cmp::Ordering::Less);
}

/// Every `SortField.Type`/missing-value combination the plain provider
/// can hold, round-tripped through `write` -> `parse` including the
/// no-missing-value form.
#[test]
fn every_numeric_type_and_missing_form_round_trips() {
    let kinds = [
        NumericSortKey::Int(None),
        NumericSortKey::Int(Some(i32::MIN)),
        NumericSortKey::Int(Some(-7)),
        NumericSortKey::Int(Some(i32::MAX)),
        NumericSortKey::Long(None),
        NumericSortKey::Long(Some(i64::MIN)),
        NumericSortKey::Long(Some(i64::MAX)),
        NumericSortKey::Float(None),
        NumericSortKey::Float(Some(f32::NEG_INFINITY)),
        NumericSortKey::Float(Some(-1.5)),
        NumericSortKey::Float(Some(f32::INFINITY)),
        NumericSortKey::Double(None),
        NumericSortKey::Double(Some(f64::NEG_INFINITY)),
        NumericSortKey::Double(Some(2.25)),
        NumericSortKey::Double(Some(f64::INFINITY)),
    ];
    for kind in kinds {
        for reverse in [false, true] {
            let sf = IndexSortField {
                field: "f".to_string(),
                reverse,
                kind: IndexSortKind::Numeric(kind),
            };
            let mut si = sample_si();
            si.index_sort = Some(vec![sf.clone()]);
            let parsed = parse(&write(&si, ""), &si.id).unwrap();
            assert_eq!(parsed.index_sort, Some(vec![sf]), "{kind:?}");
        }
    }
}

/// The STRING form of the plain provider, whose missing-value encoding
/// is `writeInt(1)` for `STRING_FIRST` and `writeInt(0)` for
/// `STRING_LAST` -- the *opposite* way round from the sorted-set marker,
/// which is exactly the kind of detail a shared helper would have got
/// wrong.
#[test]
fn the_string_sort_field_round_trips_all_three_missing_forms() {
    for missing in [
        StringMissingValue::None,
        StringMissingValue::First,
        StringMissingValue::Last,
    ] {
        let sf = IndexSortField {
            field: "name".to_string(),
            reverse: true,
            kind: IndexSortKind::String(missing),
        };
        let mut si = sample_si();
        si.index_sort = Some(vec![sf.clone()]);
        let parsed = parse(&write(&si, ""), &si.id).unwrap();
        assert_eq!(parsed.index_sort, Some(vec![sf]), "{missing:?}");
    }
}

/// Both selector enums, every ordinal, both providers -- and the
/// `BinarySortField` provider, which has no selector at all.
#[test]
fn every_selector_and_provider_round_trips() {
    let mut fields = Vec::new();
    for selector in [SortedNumericSelector::Min, SortedNumericSelector::Max] {
        fields.push(IndexSortField {
            field: format!("sn{selector:?}"),
            reverse: true,
            kind: IndexSortKind::SortedNumeric {
                key: NumericSortKey::Int(Some(3)),
                selector,
            },
        });
    }
    for selector in [
        SortedSetSelector::Min,
        SortedSetSelector::Max,
        SortedSetSelector::MiddleMin,
        SortedSetSelector::MiddleMax,
    ] {
        fields.push(IndexSortField {
            field: format!("ss{selector:?}"),
            reverse: false,
            kind: IndexSortKind::SortedSet {
                selector,
                missing: StringMissingValue::Last,
            },
        });
    }
    for missing in [
        StringMissingValue::None,
        StringMissingValue::First,
        StringMissingValue::Last,
    ] {
        fields.push(IndexSortField {
            field: format!("bin{missing:?}"),
            reverse: true,
            kind: IndexSortKind::Binary(missing),
        });
    }
    let mut si = sample_si();
    si.index_sort = Some(fields.clone());
    let parsed = parse(&write(&si, ""), &si.id).unwrap();
    assert_eq!(parsed.index_sort, Some(fields));
}

/// The exact provider byte streams Java writes, decoded -- hand-built
/// rather than produced by [`write()`], so the two sides cannot agree on a
/// misreading of the format.
#[test]
fn sorted_numeric_sorted_set_and_binary_providers_all_decode() {
    // SortedNumericSortField: field, type, reverse, selector, hasMissing, value
    let mut sn = Vec::new();
    write_string(&mut sn, PROVIDER_SORTED_NUMERIC);
    write_string(&mut sn, "sn");
    write_string(&mut sn, TYPE_LONG);
    sn.extend_from_slice(&1i32.to_le_bytes()); // reverse
    sn.extend_from_slice(&1i32.to_le_bytes()); // selector MAX
    sn.extend_from_slice(&1i32.to_le_bytes()); // hasMissing
    sn.extend_from_slice(&i64::MAX.to_le_bytes());

    // SortedSetSortField: field, reverse, selector, missing marker
    let mut ss = Vec::new();
    write_string(&mut ss, PROVIDER_SORTED_SET);
    write_string(&mut ss, "ss");
    ss.extend_from_slice(&0i32.to_le_bytes());
    ss.extend_from_slice(&2i32.to_le_bytes()); // selector MIDDLE_MIN
    ss.extend_from_slice(&1i32.to_le_bytes()); // STRING_FIRST

    // BinarySortField: field, reverse, missing marker
    let mut bin = Vec::new();
    write_string(&mut bin, PROVIDER_BINARY);
    write_string(&mut bin, "bin");
    bin.extend_from_slice(&0i32.to_le_bytes());
    bin.extend_from_slice(&2i32.to_le_bytes()); // STRING_LAST

    let mut b = SiBuilder::valid();
    b.num_sort_fields = 3;
    let mut bytes = sn;
    bytes.extend(ss);
    bytes.extend(bin);
    b.sort_field_bytes = bytes;

    let fields = parse(&b.build(), &b.id).unwrap().index_sort.unwrap();
    assert_eq!(
        fields,
        vec![
            IndexSortField {
                field: "sn".to_string(),
                reverse: true,
                kind: IndexSortKind::SortedNumeric {
                    key: NumericSortKey::Long(Some(i64::MAX)),
                    selector: SortedNumericSelector::Max,
                },
            },
            IndexSortField {
                field: "ss".to_string(),
                reverse: false,
                kind: IndexSortKind::SortedSet {
                    selector: SortedSetSelector::MiddleMin,
                    missing: StringMissingValue::First,
                },
            },
            IndexSortField {
                field: "bin".to_string(),
                reverse: false,
                kind: IndexSortKind::Binary(StringMissingValue::Last),
            },
        ]
    );
}

#[test]
fn out_of_range_selector_ordinal_rejected() {
    for (provider, extra_int) in [
        (PROVIDER_SORTED_SET, true),
        (PROVIDER_SORTED_NUMERIC, false),
    ] {
        let mut bytes = Vec::new();
        write_string(&mut bytes, provider);
        write_string(&mut bytes, "f");
        if provider == PROVIDER_SORTED_NUMERIC {
            write_string(&mut bytes, TYPE_LONG);
        }
        bytes.extend_from_slice(&0i32.to_le_bytes()); // reverse
        bytes.extend_from_slice(&9i32.to_le_bytes()); // no such selector
        if extra_int {
            bytes.extend_from_slice(&1i32.to_le_bytes());
        }
        let mut b = SiBuilder::valid();
        b.num_sort_fields = 1;
        b.sort_field_bytes = bytes;
        assert!(
            matches!(
                parse(&b.build(), &b.id),
                Err(Error::UnknownSortSelector { selector: 9, .. })
            ),
            "{provider}"
        );
    }
}

/// A `SortField.Type` that cannot be an index sort at all -- Java's own
/// `IndexWriterConfig.setIndexSort` refuses it because
/// `getIndexSorter()` is null, so no real `.si` holds one, and
/// `SortField.serialize` throws on a missing value for it.
#[test]
fn a_type_that_cannot_be_an_index_sort_is_rejected() {
    for type_name in TYPES_NOT_SORTABLE_ON_DISK {
        let mut b = SiBuilder::valid();
        b.num_sort_fields = 1;
        b.sort_field_bytes =
            SiBuilder::plain_sort_field_bytes("f", type_name, 0, Some(&0i64.to_le_bytes()[..4]));
        assert!(
            matches!(
                parse(&b.build(), &b.id),
                Err(Error::UnsupportedSortField { .. })
            ),
            "{type_name}"
        );
    }
}

/// A `SortedNumericSortField` whose type is `STRING` is an
/// `AssertionError` inside Java's own `serialize`, so it is refused here
/// too rather than lowered onto something.
#[test]
fn a_sorted_numeric_sort_field_with_a_string_type_is_rejected() {
    let mut bytes = Vec::new();
    write_string(&mut bytes, PROVIDER_SORTED_NUMERIC);
    write_string(&mut bytes, "f");
    write_string(&mut bytes, TYPE_STRING);
    bytes.extend_from_slice(&0i32.to_le_bytes());
    bytes.extend_from_slice(&0i32.to_le_bytes());
    bytes.extend_from_slice(&0i32.to_le_bytes());
    let mut b = SiBuilder::valid();
    b.num_sort_fields = 1;
    b.sort_field_bytes = bytes;
    assert!(matches!(
        parse(&b.build(), &b.id),
        Err(Error::UnsupportedSortField { .. })
    ));
}

/// Java's two marker readers fall through to "no missing value" for any
/// int that is neither 1 nor 2, so being stricter would refuse a file
/// Lucene reads.
#[test]
fn an_unknown_string_missing_marker_reads_as_no_missing_value() {
    let mut bytes = Vec::new();
    write_string(&mut bytes, PROVIDER_BINARY);
    write_string(&mut bytes, "bin");
    bytes.extend_from_slice(&0i32.to_le_bytes());
    bytes.extend_from_slice(&77i32.to_le_bytes());
    let mut b = SiBuilder::valid();
    b.num_sort_fields = 1;
    b.sort_field_bytes = bytes;
    let fields = parse(&b.build(), &b.id).unwrap().index_sort.unwrap();
    assert_eq!(
        fields[0].kind,
        IndexSortKind::Binary(StringMissingValue::None)
    );
}

/// `key_comparison` is the whole comparator contract, so every kind's
/// answer is pinned: which `compare` and which sentinel.
#[test]
fn key_comparison_names_the_right_compare_and_sentinel_per_kind() {
    use std::cmp::Ordering;
    let sf = |kind| IndexSortField {
        field: "f".to_string(),
        reverse: false,
        kind,
    };
    assert_eq!(
        sf(IndexSortKind::Numeric(NumericSortKey::Int(Some(5)))).key_comparison(),
        (SortKeyKind::Int, 5)
    );
    assert_eq!(
        sf(IndexSortKind::Numeric(NumericSortKey::Long(None))).key_comparison(),
        (SortKeyKind::Long, 0)
    );
    assert_eq!(
        sf(IndexSortKind::Numeric(NumericSortKey::Float(Some(1.0)))).key_comparison(),
        (SortKeyKind::Float, 1.0f32.to_bits() as i32 as i64)
    );
    assert_eq!(
        sf(IndexSortKind::Numeric(NumericSortKey::Double(Some(-1.0)))).key_comparison(),
        (SortKeyKind::Double, (-1.0f64).to_bits() as i64)
    );
    // `None` behaves like `First` in the comparator even though the two
    // are distinguishable on disk.
    for missing in [StringMissingValue::None, StringMissingValue::First] {
        assert_eq!(
            sf(IndexSortKind::String(missing)).key_comparison(),
            (SortKeyKind::Ordinal, i32::MIN as i64)
        );
    }
    assert_eq!(
        sf(IndexSortKind::SortedSet {
            selector: SortedSetSelector::Max,
            missing: StringMissingValue::Last,
        })
        .key_comparison(),
        (SortKeyKind::Ordinal, i32::MAX as i64)
    );
    // A BinarySortField compares ranks with StringSorter's sentinels.
    assert_eq!(
        sf(IndexSortKind::Binary(StringMissingValue::Last)).key_comparison(),
        (SortKeyKind::Ordinal, i32::MAX as i64)
    );
    assert_eq!(
        sf(IndexSortKind::Binary(StringMissingValue::None)).key_comparison(),
        (SortKeyKind::Ordinal, i32::MIN as i64)
    );

    // INT compares the low 32 bits, so a value whose 64-bit form is
    // larger can still be the smaller INT.
    let int_cmp = SortKeyComparator::new(&sf(IndexSortKind::Numeric(NumericSortKey::Int(None))));
    assert_eq!(
        int_cmp.compare(Some(0xFFFF_FFFF), Some(1)),
        Ordering::Less,
        "0xFFFFFFFF is -1 as an int"
    );
}

/// `Float.compare`/`Double.compare`, not `f32::total_cmp`: every NaN is
/// equal and greater than `+Infinity`, and `-0.0 < 0.0`.
#[test]
fn float_and_double_sorts_use_javas_compare() {
    use std::cmp::Ordering;
    let f = IndexSortField {
        field: "f".to_string(),
        reverse: false,
        kind: IndexSortKind::Numeric(NumericSortKey::Float(None)),
    };
    let cmp = SortKeyComparator::new(&f);
    let bits = |v: f32| Some(v.to_bits() as i32 as i64);
    // The ordinary cases first: `Float.compare`'s `<` and `>` arms, which
    // are the ones that make a FLOAT sort a *float* sort -- comparing the
    // same 64 bits as longs would order -1.5 above 1.5.
    assert_eq!(cmp.compare(bits(-1.5), bits(1.5)), Ordering::Less);
    assert_eq!(cmp.compare(bits(1.5), bits(-1.5)), Ordering::Greater);
    assert_eq!(cmp.compare(bits(1.5), bits(1.5)), Ordering::Equal);
    assert_eq!(cmp.compare(bits(-0.0), bits(0.0)), Ordering::Less);
    assert_eq!(
        cmp.compare(bits(f32::NAN), bits(f32::INFINITY)),
        Ordering::Greater
    );
    assert_eq!(
        cmp.compare(bits(f32::NAN), bits(-f32::NAN)),
        Ordering::Equal,
        "Java canonicalizes every NaN"
    );

    let d = IndexSortField {
        field: "d".to_string(),
        reverse: true,
        kind: IndexSortKind::Numeric(NumericSortKey::Double(None)),
    };
    let cmp = SortKeyComparator::new(&d);
    let dbits = |v: f64| Some(v.to_bits() as i64);
    // Reversed, so `<` becomes `Greater`.
    assert_eq!(cmp.compare(dbits(-2.5), dbits(2.5)), Ordering::Greater);
    assert_eq!(cmp.compare(dbits(2.5), dbits(-2.5)), Ordering::Less);
    assert_eq!(cmp.compare(dbits(-0.0), dbits(0.0)), Ordering::Greater);
    assert_eq!(
        cmp.compare(dbits(f64::NAN), dbits(-f64::NAN)),
        Ordering::Equal
    );
    // The sentinel for a no-missing-value DOUBLE sort is `0.0`, and
    // `reverse` applies to it.
    assert_eq!(cmp.compare(None, dbits(1.0)), Ordering::Greater);
}

/// `describe_index_sort` is Java's `Sort.toString()`, and the fixture
/// manifests compare it against Lucene's own output
/// (`tests/index_sort_wide_fixtures.rs`,
/// `examples/write_segment_info_fixture.rs` -> `VerifySegmentInfo`). This
/// pins every branch of it -- including the ones no fixture happens to
/// use -- so a kind added later cannot render as something Java never
/// prints.
#[test]
fn describe_index_sort_renders_every_kind_the_way_java_does() {
    let sf = |field: &str, reverse: bool, kind| IndexSortField {
        field: field.to_string(),
        reverse,
        kind,
    };
    assert_eq!(describe_index_sort(None), "<none>");
    let all = vec![
        sf(
            "i",
            true,
            IndexSortKind::Numeric(NumericSortKey::Int(Some(-7))),
        ),
        sf(
            "l",
            false,
            IndexSortKind::Numeric(NumericSortKey::Long(None)),
        ),
        sf(
            "f",
            false,
            IndexSortKind::Numeric(NumericSortKey::Float(Some(-1.5))),
        ),
        // An *integral* float, where Java prints `1.0` and Rust's
        // `Display` prints `1`.
        sf(
            "f2",
            false,
            IndexSortKind::Numeric(NumericSortKey::Float(Some(1.0))),
        ),
        sf(
            "d",
            true,
            IndexSortKind::Numeric(NumericSortKey::Double(Some(2.25))),
        ),
        sf(
            "d2",
            false,
            IndexSortKind::Numeric(NumericSortKey::Double(Some(-3.0))),
        ),
        sf("s", true, IndexSortKind::String(StringMissingValue::Last)),
        sf("s2", false, IndexSortKind::String(StringMissingValue::None)),
        sf(
            "sn",
            false,
            IndexSortKind::SortedNumeric {
                key: NumericSortKey::Int(Some(9)),
                selector: SortedNumericSelector::Max,
            },
        ),
        sf(
            "sn2",
            false,
            IndexSortKind::SortedNumeric {
                key: NumericSortKey::Double(None),
                selector: SortedNumericSelector::Min,
            },
        ),
        sf(
            "ss",
            true,
            IndexSortKind::SortedSet {
                selector: SortedSetSelector::MiddleMin,
                missing: StringMissingValue::First,
            },
        ),
        sf(
            "ss2",
            false,
            IndexSortKind::SortedSet {
                selector: SortedSetSelector::MiddleMax,
                missing: StringMissingValue::None,
            },
        ),
        sf(
            "ss3",
            false,
            IndexSortKind::SortedSet {
                selector: SortedSetSelector::Max,
                missing: StringMissingValue::Last,
            },
        ),
        sf("b", false, IndexSortKind::Binary(StringMissingValue::None)),
        sf("b2", true, IndexSortKind::Binary(StringMissingValue::First)),
    ];
    assert_eq!(
        describe_index_sort(Some(&all)),
        "<int: \"i\">! missingValue=-7,\
             <long: \"l\">,\
             <float: \"f\"> missingValue=-1.5,\
             <float: \"f2\"> missingValue=1.0,\
             <double: \"d\">! missingValue=2.25,\
             <double: \"d2\"> missingValue=-3.0,\
             <string: \"s\">! missingValue=SortField.STRING_LAST,\
             <string: \"s2\">,\
             <sortednumeric: \"sn\"> missingValue=9 selector=MAX type=INT,\
             <sortednumeric: \"sn2\"> selector=MIN type=DOUBLE,\
             <sortedset: \"ss\">! missingValue=SortField.STRING_FIRST selector=MIDDLE_MIN,\
             <sortedset: \"ss2\"> selector=MIDDLE_MAX,\
             <sortedset: \"ss3\"> missingValue=SortField.STRING_LAST selector=MAX,\
             <binary: \"b\">,\
             <binary: \"b2\">! missingValue=SortField.STRING_FIRST"
    );
    // `Float.toString`/`Double.toString` on the shapes Rust renders
    // differently: infinities and NaN keep Java's spelling.
    assert_eq!(java_float_string(f32::INFINITY), "Infinity");
    assert_eq!(java_float_string(f32::NEG_INFINITY), "-Infinity");
    assert_eq!(java_double_string(f64::NEG_INFINITY), "-Infinity");
    assert_eq!(java_double_string(f64::INFINITY), "Infinity");
    assert_eq!(java_double_string(f64::NAN), "NaN");
}

/// A **SORTED_NUMERIC** FLOAT/DOUBLE column holds
/// `NumericUtils.floatToSortableInt`, not the raw bits a NUMERIC one
/// holds, and `SortedNumericSelector.wrap` undoes that with
/// `sortableFloatBits` before `FloatSorter` ever sees a value.
///
/// Comparing the stored form as raw bits instead reverses the whole
/// negative half of the ordering -- `-1.0f` is stored as `0xC07FFFFF`,
/// which reads back as `-3.99999` -- so this pins the two kinds against
/// each other on the same values.
#[test]
fn a_sorted_numeric_float_sort_undoes_the_sortable_encoding() {
    use std::cmp::Ordering;
    let sorted_numeric = |key| IndexSortField {
        field: "f".to_string(),
        reverse: false,
        kind: IndexSortKind::SortedNumeric {
            key,
            selector: SortedNumericSelector::Min,
        },
    };

    let cmp = SortKeyComparator::new(&sorted_numeric(NumericSortKey::Float(None)));
    // What `FloatField` stores: `floatToSortableInt`.
    let stored = |v: f32| Some(float_to_sortable_int(v) as i64);
    for (a, b) in [(-2.0f32, -1.0), (-1.0, 0.0), (0.0, 1.0), (1.0, 2.0)] {
        assert_eq!(
            cmp.compare(stored(a), stored(b)),
            Ordering::Less,
            "{a} < {b}"
        );
    }
    // The sentinel is in the same space, and a no-missing-value sort
    // compares a document with no value as `0.0`.
    assert_eq!(cmp.compare(None, stored(-1.0)), Ordering::Greater);
    assert_eq!(cmp.compare(None, stored(1.0)), Ordering::Less);
    // With an explicit sentinel it lands where that sentinel does.
    let cmp = SortKeyComparator::new(&sorted_numeric(NumericSortKey::Float(Some(-5.0))));
    assert_eq!(cmp.compare(None, stored(-1.0)), Ordering::Less);

    let cmp = SortKeyComparator::new(&sorted_numeric(NumericSortKey::Double(Some(-5.0))));
    let stored = |v: f64| Some(double_to_sortable_long(v));
    assert_eq!(cmp.compare(stored(-2.0), stored(-1.0)), Ordering::Less);
    assert_eq!(cmp.compare(None, stored(-1.0)), Ordering::Less);

    // And the NUMERIC kind over the *same* field is the other encoding:
    // raw bits, so the sortable form would be read as a different value
    // entirely. This is the assertion that fails if the two kinds are
    // conflated.
    let raw = SortKeyComparator::new(&IndexSortField {
        field: "f".to_string(),
        reverse: false,
        kind: IndexSortKind::Numeric(NumericSortKey::Float(None)),
    });
    assert_eq!(
        raw.compare(
            Some(float_to_sortable_int(-2.0) as i64),
            Some(float_to_sortable_int(-1.0) as i64)
        ),
        Ordering::Greater,
        "read as raw bits the sortable encoding of -2.0 is larger, which is the bug"
    );
}

/// A FLOAT/DOUBLE missing value survives the `sortableInt`/`sortableLong`
/// encoding Java writes it in -- including a NaN, which Java
/// canonicalizes on the way out.
#[test]
fn float_missing_values_round_trip_through_the_sortable_encoding() {
    for v in [0.0f32, -0.0, 1.5, -1.5, f32::MIN, f32::MAX, f32::EPSILON] {
        assert_eq!(
            sortable_int_to_float(float_to_sortable_int(v)).to_bits(),
            v.to_bits(),
            "{v}"
        );
    }
    assert!(sortable_int_to_float(float_to_sortable_int(f32::NAN)).is_nan());
    for v in [0.0f64, -0.0, 1.5, -1.5, f64::MIN, f64::MAX] {
        assert_eq!(
            sortable_long_to_double(double_to_sortable_long(v)).to_bits(),
            v.to_bits(),
            "{v}"
        );
    }
    assert!(sortable_long_to_double(double_to_sortable_long(f64::NAN)).is_nan());
}

/// Illegal `Version.fromBits` components (each is encoded into one byte)
/// must be rejected, not silently carried into later comparisons.
#[test]
fn out_of_range_version_component_rejected() {
    let b = SiBuilder::valid();
    let mut bytes = b.build();
    let header_len =
        codec_util::CODEC_MAGIC.to_be_bytes().len() + 1 + CODEC_NAME.len() + 4 + ID_LENGTH + 1;
    // minor version sits at header_len + 4
    bytes[header_len + 4..header_len + 8].copy_from_slice(&999i32.to_le_bytes());
    assert!(matches!(
        parse(&bytes, &b.id),
        Err(Error::IllegalVersion {
            which: "minor",
            value: 999
        })
    ));
}

#[test]
fn wrong_id_rejected_with_store_error() {
    let b = SiBuilder::valid();
    let wrong_id = [9u8; ID_LENGTH];
    assert!(matches!(parse(&b.build(), &wrong_id), Err(Error::Store(_))));
}

// --- write() round-trips through parse() ---

fn sample_si() -> SegmentInfo {
    SegmentInfo {
        id: [3u8; ID_LENGTH],
        version: LuceneVersion {
            major: 10,
            minor: 0,
            bugfix: 0,
        },
        min_version: None,
        doc_count: 42,
        is_compound_file: false,
        has_blocks: false,
        diagnostics: vec![],
        files: vec![],
        attributes: vec![],
        index_sort: None,
    }
}

#[test]
fn write_minimal_round_trips() {
    let si = sample_si();
    let bytes = write(&si, "");
    let parsed = parse(&bytes, &si.id).unwrap();
    assert_eq!(parsed.doc_count, 42);
    assert!(!parsed.is_compound_file);
    assert!(!parsed.has_blocks);
    assert!(parsed.min_version.is_none());
    assert_eq!(parsed.version, si.version);
    assert!(parsed.diagnostics.is_empty());
    assert!(parsed.files.is_empty());
    assert!(parsed.attributes.is_empty());
}

#[test]
fn write_compound_file_round_trips() {
    let mut si = sample_si();
    si.is_compound_file = true;
    si.has_blocks = true;
    let bytes = write(&si, "");
    let parsed = parse(&bytes, &si.id).unwrap();
    assert!(parsed.is_compound_file);
    assert!(parsed.has_blocks);
}

#[test]
fn write_min_version_round_trips() {
    let mut si = sample_si();
    si.min_version = Some(LuceneVersion {
        major: 9,
        minor: 12,
        bugfix: 0,
    });
    let bytes = write(&si, "");
    let parsed = parse(&bytes, &si.id).unwrap();
    let mv = parsed.min_version.unwrap();
    assert_eq!((mv.major, mv.minor, mv.bugfix), (9, 12, 0));
}

#[test]
fn write_diagnostics_files_attributes_round_trip() {
    let mut si = sample_si();
    si.diagnostics = vec![
        ("source".to_string(), "flush".to_string()),
        ("os".to_string(), "Linux".to_string()),
    ];
    si.files = vec!["_0.fdt".to_string(), "_0.fdx".to_string()];
    si.attributes = vec![(
        "Lucene90StoredFieldsFormat.mode".to_string(),
        "BEST_SPEED".to_string(),
    )];
    let bytes = write(&si, "");
    let parsed = parse(&bytes, &si.id).unwrap();
    assert_eq!(parsed.diagnostics, si.diagnostics);
    assert_eq!(parsed.files, si.files);
    assert_eq!(parsed.attributes, si.attributes);
}

#[test]
fn write_with_segment_suffix_round_trips() {
    let si = sample_si();
    let bytes = write(&si, "suffix1");
    let parsed = parse(&bytes, &si.id);
    // parse() with the wrong (empty) suffix must fail; the exact suffix must match.
    assert!(parsed.is_err());
    let parsed_ok = {
        let mut input = SliceInput::new(&bytes);
        codec_util::check_index_header(
            &mut input,
            CODEC_NAME,
            VERSION_START,
            VERSION_CURRENT,
            &si.id,
            "suffix1",
        )
    };
    assert!(parsed_ok.is_ok());
}

#[test]
fn write_zero_doc_count_round_trips() {
    let mut si = sample_si();
    si.doc_count = 0;
    let bytes = write(&si, "");
    let parsed = parse(&bytes, &si.id).unwrap();
    assert_eq!(parsed.doc_count, 0);
}

#[test]
fn write_no_index_sort_round_trips_as_none() {
    let si = sample_si();
    assert!(si.index_sort.is_none());
    let bytes = write(&si, "");
    let parsed = parse(&bytes, &si.id).unwrap();
    assert!(parsed.index_sort.is_none());
}

#[test]
fn write_ascending_index_sort_round_trips() {
    let mut si = sample_si();
    si.index_sort = Some(vec![long_sort("timestamp", false, false)]);
    let parsed = parse(&write(&si, ""), &si.id).unwrap();
    assert_eq!(parsed.index_sort, si.index_sort);
}

#[test]
fn write_descending_index_sort_with_missing_last_round_trips() {
    let mut si = sample_si();
    si.index_sort = Some(vec![long_sort("score", true, true)]);
    let parsed = parse(&write(&si, ""), &si.id).unwrap();
    assert_eq!(parsed.index_sort, si.index_sort);
}
