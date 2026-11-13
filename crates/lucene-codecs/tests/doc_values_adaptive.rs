//! The `Adaptive90` doc-values format end to end at the codec layer: what
//! [`doc_values::write_fields_with_format`] writes under it reads back value
//! for value through every reader path, a Lucene90 file never carries an
//! adaptive value array, and the format falls back to Lucene90's encodings
//! where they are smaller.
#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::doc_values::{
    self, numeric_value, DenseField, DocValuesMeta, Lucene90DocValuesFormat, NumericReader,
    SortedNumericReader,
};
use lucene_codecs::field_infos::{DocValuesSkipIndexType, DocValuesType, FieldInfo, FieldInfos};

const ID: [u8; 16] = [7; 16];

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn sortable(x: f64) -> i64 {
    let bits = x.to_bits() as i64;
    bits ^ ((bits >> 63) & 0x7fff_ffff_ffff_ffff)
}

fn infos(types: &[DocValuesType], skip: bool) -> FieldInfos {
    FieldInfos {
        fields: types
            .iter()
            .enumerate()
            .map(|(i, &t)| {
                FieldInfo::new(format!("f{i}"), i as i32).with_doc_values(
                    t,
                    if skip {
                        DocValuesSkipIndexType::Range
                    } else {
                        DocValuesSkipIndexType::None
                    },
                    -1,
                )
            })
            .collect(),
    }
}

fn write(
    fields: &[DenseField<'_>],
    max_doc: i32,
    format: Lucene90DocValuesFormat,
    skip: &[i32],
) -> (Vec<u8>, Vec<u8>) {
    let suffix = format!("{}_0", format.name());
    let (meta, data, _) =
        doc_values::write_fields_with_format(fields, skip, max_doc, &ID, &suffix, format).unwrap();
    (meta, data)
}

fn parse(
    meta: &[u8],
    data: &[u8],
    format: Lucene90DocValuesFormat,
    fis: &FieldInfos,
) -> DocValuesMeta {
    let suffix = format!("{}_0", format.name());
    doc_values::check_data_header_footer(data, &ID, &suffix).unwrap();
    doc_values::parse_meta(meta, &ID, &suffix, fis).unwrap().1
}

fn timestamps(n: usize, seed: u64) -> Vec<i64> {
    let mut r = Rng(seed);
    let mut t = 1_700_000_000_000i64;
    (0..n)
        .map(|_| {
            t += (r.next() % 3000) as i64;
            t
        })
        .collect()
}

fn amounts(n: usize, seed: u64) -> Vec<i64> {
    let mut r = Rng(seed);
    (0..n)
        .map(|_| sortable((r.next() % 100_000) as f64 / 100.0))
        .collect()
}

#[test]
fn dense_numeric_reads_back_through_every_path() {
    let fis = infos(&[DocValuesType::Numeric, DocValuesType::Numeric], false);
    let n = 5000;
    let ts = timestamps(n, 1);
    let am = amounts(n, 2);
    let fields = [DenseField::Numeric(0, &ts), DenseField::Numeric(1, &am)];
    let (meta, data) = write(
        &fields,
        n as i32,
        Lucene90DocValuesFormat::rust_adaptive(),
        &[],
    );
    let (lmeta, ldata) = write(&fields, n as i32, Lucene90DocValuesFormat::default(), &[]);
    assert!(
        data.len() * 2 < ldata.len(),
        "{} vs {}",
        data.len(),
        ldata.len()
    );

    let parsed = parse(&meta, &data, Lucene90DocValuesFormat::rust_adaptive(), &fis);
    let lparsed = parse(&lmeta, &ldata, Lucene90DocValuesFormat::default(), &fis);
    for (field, want) in [(0, &ts), (1, &am)] {
        let entry = parsed.numeric_entry(field).unwrap();
        assert!(entry.adaptive.is_some());
        assert!(lparsed.numeric_entry(field).unwrap().adaptive.is_none());
        // One value at a time, in order and backwards.
        let mut r = NumericReader::new(&data, entry);
        for (doc, &v) in want.iter().enumerate() {
            assert_eq!(r.value(doc as i32).unwrap(), Some(v));
        }
        let mut r = NumericReader::new(&data, entry);
        for doc in (0..n).rev().step_by(7) {
            assert_eq!(r.value(doc as i32).unwrap(), Some(want[doc]));
            assert_eq!(
                numeric_value(&data, entry, doc as i32).unwrap(),
                Some(want[doc])
            );
            assert_eq!(r.value_at_index(doc as i64).unwrap(), want[doc]);
        }
        assert!(numeric_value(&data, entry, n as i32).is_err());
        // In bulk, from an unaligned start.
        let mut got = Vec::new();
        NumericReader::new(&data, entry)
            .for_each_value(3, n as i32, |d, v| got.push((d, v)))
            .unwrap();
        assert_eq!(got.len(), n - 3);
        assert!(got.iter().all(|&(d, v)| want[d as usize] == v));
        for start in [0, 1, 127, 128, 4990] {
            let mut values = vec![0i64; 200];
            let mut present = vec![0u64; 4];
            NumericReader::new(&data, entry)
                .fill_window(start, &mut values, &mut present)
                .unwrap();
            for i in 0..200 {
                let doc = start as usize + i;
                let has = present[i / 64] >> (i % 64) & 1 == 1;
                assert_eq!(has, doc < n, "doc {doc}");
                if has {
                    assert_eq!(values[i], want[doc]);
                }
            }
        }
    }
}

#[test]
fn sparse_and_multi_valued_fields_read_back() {
    let fis = infos(
        &[
            DocValuesType::Numeric,
            DocValuesType::SortedNumeric,
            DocValuesType::SortedNumeric,
        ],
        false,
    );
    let n = 3000i32;
    let ts = timestamps(n as usize, 3);
    let sparse: Vec<(i32, i64)> = (0..n)
        .filter(|d| d % 3 != 0)
        .map(|d| (d, ts[d as usize]))
        .collect();
    let multi: Vec<Vec<i64>> = (0..n as usize)
        .map(|d| (0..(d % 3 + 1)).map(|k| ts[d] + k as i64 * 1000).collect())
        .collect();
    let sparse_multi: Vec<(i32, Vec<i64>)> = (0..n)
        .filter(|d| d % 5 != 0)
        .map(|d| (d, vec![ts[d as usize], ts[d as usize] + 1]))
        .collect();
    let fields = [
        DenseField::SparseNumeric(0, &sparse),
        DenseField::SortedNumeric(1, &multi),
        DenseField::SparseSortedNumeric(2, &sparse_multi),
    ];
    let format = Lucene90DocValuesFormat::rust_adaptive();
    let (meta, data) = write(&fields, n, format, &[0, 1, 2]);
    let parsed = parse(
        &meta,
        &data,
        format,
        &infos(
            &[
                DocValuesType::Numeric,
                DocValuesType::SortedNumeric,
                DocValuesType::SortedNumeric,
            ],
            true,
        ),
    );
    let _ = fis;

    let entry = parsed.numeric_entry(0).unwrap();
    assert!(entry.adaptive.is_some());
    let mut r = NumericReader::new(&data, entry);
    for d in 0..n {
        let want = (d % 3 != 0).then(|| ts[d as usize]);
        assert_eq!(r.value(d).unwrap(), want, "doc {d}");
    }
    let mut values = vec![0i64; 128];
    let mut present = vec![0u64; 2];
    NumericReader::new(&data, entry)
        .fill_window(64, &mut values, &mut present)
        .unwrap();
    for i in 0..128 {
        let d = 64 + i;
        assert_eq!(present[i / 64] >> (i % 64) & 1 == 1, d % 3 != 0);
        if d % 3 != 0 {
            assert_eq!(values[i], ts[d]);
        }
    }

    for (field, want) in [
        (
            1,
            (0..n)
                .map(|d| Some(multi[d as usize].clone()))
                .collect::<Vec<_>>(),
        ),
        (
            2,
            (0..n)
                .map(|d| (d % 5 != 0).then(|| vec![ts[d as usize], ts[d as usize] + 1]))
                .collect(),
        ),
    ] {
        let entry = parsed
            .sorted_numeric
            .iter()
            .find(|e| e.field_number == field)
            .unwrap();
        assert!(entry.numeric.adaptive.is_some());
        let mut r = SortedNumericReader::new(&data, entry);
        let mut out = Vec::new();
        for d in 0..n {
            r.values(d, &mut out).unwrap();
            assert_eq!(
                want[d as usize].clone().unwrap_or_default(),
                out,
                "field {field} doc {d}"
            );
        }
        let mut seen = 0;
        SortedNumericReader::new(&data, entry)
            .for_each_doc(0, n, |d, v| {
                assert_eq!(Some(v.to_vec()), want[d as usize]);
                seen += 1;
            })
            .unwrap();
        assert_eq!(seen, want.iter().filter(|w| w.is_some()).count());
    }
}

#[test]
fn lucene90_encodings_win_where_they_are_smaller() {
    let fis = infos(&[DocValuesType::Numeric], false);
    let format = Lucene90DocValuesFormat::rust_adaptive();
    let mut r = Rng(9);
    // Constant; few distinct, widely spread values (Lucene90's table packs
    // them at 2 bits a value); fewer values than one block.
    let columns: [Vec<i64>; 3] = [
        vec![42; 2000],
        (0..2000)
            .map(|_| [7, 1 << 40, -(1 << 50), 3][(r.next() % 4) as usize])
            .collect(),
        vec![1, 5, 9],
    ];
    for values in &columns {
        let fields = [DenseField::Numeric(0, values)];
        let (meta, data) = write(&fields, values.len() as i32, format, &[]);
        let (_, ldata) = write(
            &fields,
            values.len() as i32,
            Lucene90DocValuesFormat::default(),
            &[],
        );
        // The same bytes as Lucene90 wrote, under a segment suffix 2 bytes
        // longer ("Adaptive90" against "Lucene90").
        assert_eq!(data.len(), ldata.len() + 2);
        let parsed = parse(&meta, &data, format, &fis);
        let entry = parsed.numeric_entry(0).unwrap();
        assert!(entry.adaptive.is_none());
        let mut reader = NumericReader::new(&data, entry);
        for (d, &v) in values.iter().enumerate() {
            assert_eq!(reader.value(d as i32).unwrap(), Some(v));
        }
    }
}

#[test]
fn corrupt_adaptive_values_are_errors_not_panics() {
    let fis = infos(&[DocValuesType::Numeric], false);
    let ts = timestamps(1000, 5);
    let fields = [DenseField::Numeric(0, &ts)];
    let format = Lucene90DocValuesFormat::rust_adaptive();
    let (meta, data) = write(&fields, 1000, format, &[]);
    let parsed = parse(&meta, &data, format, &fis);
    let entry = parsed.numeric_entry(0).unwrap();
    // A flipped byte in a block header.
    let mut broken = data.clone();
    broken[entry.values_offset as usize] = 0xff;
    let mut reader = NumericReader::new(&broken, entry);
    assert!(reader.value(0).is_err() || reader.value(0).unwrap() != Some(ts[0]));
    // A descriptor pointing past the file.
    let mut entry = entry.clone();
    entry.adaptive = Some(Box::new(doc_values::AdaptiveEntry::new(
        data.len() as i64 + 5,
    )));
    assert!(NumericReader::new(&data, &entry).value(0).is_err());
    assert!(numeric_value(&data, &entry, 0).is_err());
}

#[test]
fn per_document_value_arrays_carry_window_statistics() {
    use lucene_codecs::adaptive_zones::ZONE_SIZE;
    let n = 3 * ZONE_SIZE + 100;
    let ts = timestamps(n, 11);
    let single: Vec<Vec<i64>> = ts.iter().map(|&t| vec![t]).collect();
    let multi: Vec<Vec<i64>> = ts.iter().map(|&t| vec![t, t + 1]).collect();
    let sparse: Vec<(i32, i64)> = (0..n as i32)
        .filter(|d| d % 3 != 0)
        .map(|d| (d, ts[d as usize]))
        .collect();
    let every: Vec<(i32, i64)> = (0..n as i32).map(|d| (d, ts[d as usize])).collect();
    let fields = [
        DenseField::Numeric(0, &ts),
        DenseField::SortedNumeric(1, &single),
        DenseField::SortedNumeric(2, &multi),
        DenseField::SparseNumeric(3, &sparse),
        DenseField::SparseNumeric(4, &every),
    ];
    let format = Lucene90DocValuesFormat::rust_adaptive();
    let (meta, data) = write(&fields, n as i32, format, &[]);
    let parsed = parse(
        &meta,
        &data,
        format,
        &infos(
            &[
                DocValuesType::Numeric,
                DocValuesType::SortedNumeric,
                DocValuesType::SortedNumeric,
                DocValuesType::Numeric,
                DocValuesType::Numeric,
            ],
            false,
        ),
    );
    let sorted = |f: i32| {
        &parsed
            .sorted_numeric
            .iter()
            .find(|e| e.field_number == f)
            .unwrap()
            .numeric
    };
    let entries = [
        (parsed.numeric_entry(0).unwrap(), true),
        (sorted(1), true),
        (sorted(2), false),
        (parsed.numeric_entry(3).unwrap(), false),
        (parsed.numeric_entry(4).unwrap(), true),
    ];
    for (f, (entry, zoned)) in entries.into_iter().enumerate() {
        assert_eq!(entry.adaptive.as_ref().unwrap().zones, zoned, "field {f}");
        let r = NumericReader::new(&data, entry);
        if !zoned {
            assert_eq!(r.window_stats(0).unwrap(), None, "field {f}");
            continue;
        }
        for (z, chunk) in ts.chunks(ZONE_SIZE).enumerate() {
            let (stats, count) = r.window_stats((z * ZONE_SIZE) as i32).unwrap().unwrap();
            assert_eq!(count, chunk.len());
            assert_eq!(stats.min, *chunk.iter().min().unwrap());
            assert_eq!(stats.max, *chunk.iter().max().unwrap());
            assert_eq!(stats.sum, Some(chunk.iter().sum::<i64>()));
        }
        // Off a zone boundary, past the end, or negative: no statistics.
        for start in [1, 100, ZONE_SIZE as i32 + 64, (4 * ZONE_SIZE) as i32, -1] {
            assert_eq!(r.window_stats(start).unwrap(), None, "start {start}");
        }
        // The values read back as before.
        let mut r = NumericReader::new(&data, entry);
        for d in (0..n).step_by(97) {
            assert_eq!(r.value(d as i32).unwrap(), Some(ts[d]));
        }
    }

    // A Lucene90 entry, and an adaptive one too short for a zone, have none.
    let (lmeta, ldata) = write(
        &fields[..1],
        n as i32,
        Lucene90DocValuesFormat::default(),
        &[],
    );
    let fis = infos(&[DocValuesType::Numeric], false);
    let lparsed = parse(&lmeta, &ldata, Lucene90DocValuesFormat::default(), &fis);
    let lentry = lparsed.numeric_entry(0).unwrap();
    assert_eq!(
        NumericReader::new(&ldata, lentry).window_stats(0).unwrap(),
        None
    );
    let short = &ts[..ZONE_SIZE - 1];
    let (smeta, sdata) = write(
        &[DenseField::Numeric(0, short)],
        short.len() as i32,
        format,
        &[],
    );
    let sparsed = parse(&smeta, &sdata, format, &fis);
    let sentry = sparsed.numeric_entry(0).unwrap();
    assert!(!sentry.adaptive.as_ref().unwrap().zones);
    assert_eq!(
        NumericReader::new(&sdata, sentry).window_stats(0).unwrap(),
        None
    );

    // A zone section cut short is an error, not a panic: field 0 alone puts
    // it last before the footer.
    let (zmeta, zdata) = write(&fields[..1], n as i32, format, &[]);
    let zparsed = parse(&zmeta, &zdata, format, &fis);
    let zentry = zparsed.numeric_entry(0).unwrap();
    assert!(zentry.adaptive.as_ref().unwrap().zones);
    let cut = &zdata[..zdata.len() - 16 - 20];
    assert!(NumericReader::new(cut, zentry).window_stats(0).is_err());
}
