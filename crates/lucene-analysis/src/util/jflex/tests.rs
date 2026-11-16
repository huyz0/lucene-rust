#![cfg(test)]
use super::*;
use crate::reader::StrReader;

fn zlib(raw: &[u8]) -> Vec<u8> {
    miniz_oxide::deflate::compress_to_vec_zlib(raw, 6)
}

fn table(out: &mut Vec<u8>, name: &str, values: &[i32]) {
    out.push(name.len() as u8);
    out.extend_from_slice(name.as_bytes());
    out.extend_from_slice(&(values.len() as u32).to_le_bytes());
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

/// A one-state scanner whose every code point is a one-unit match of
/// action 1, and whose class map sends everything to class 0.
fn tiny() -> Vec<u8> {
    let mut raw = b"JFLX".to_vec();
    table(&mut raw, "ZZ_LEXSTATE", &[0, 0]);
    table(&mut raw, "ZZ_CMAP_TOP", &vec![0; 0x110000 >> 8]);
    table(&mut raw, "ZZ_CMAP_BLOCKS", &[0; 256]);
    table(&mut raw, "ZZ_ACTION", &[0, 1]);
    table(&mut raw, "ZZ_ROWMAP", &[0, 1]);
    table(&mut raw, "ZZ_TRANS", &[1, -1]);
    table(&mut raw, "ZZ_ATTRIBUTE", &[0, 9]);
    zlib(&raw)
}

#[test]
fn corrupt_tables_are_errors() {
    assert!(JFlexTables::load(b"not zlib").is_err());
    assert!(JFlexTables::load(&zlib(b"XXXX")).is_err());
    assert!(JFlexTables::load(&zlib(b"JFLX\x02")).is_err());
    let mut raw = b"JFLX".to_vec();
    table(&mut raw, "ZZ_WRONG", &[0]);
    assert!(JFlexTables::load(&zlib(&raw)).is_err());
    let mut raw = b"JFLX".to_vec();
    for name in TABLE_NAMES {
        table(&mut raw, name, &[0]);
    }
    assert!(JFlexTables::load(&zlib(&raw)).is_err(), "bad shapes");
    assert!(JFlexTables::load(&tiny()).is_ok());
    // A `ZZ_CMAP_TOP` entry past `ZZ_CMAP_BLOCKS`.
    let mut raw = b"JFLX".to_vec();
    table(&mut raw, "ZZ_LEXSTATE", &[0, 0]);
    let mut top = vec![0; 0x110000 >> 8];
    top[1] = 256;
    table(&mut raw, "ZZ_CMAP_TOP", &top);
    table(&mut raw, "ZZ_CMAP_BLOCKS", &[0; 256]);
    table(&mut raw, "ZZ_ACTION", &[0, 1]);
    table(&mut raw, "ZZ_ROWMAP", &[0, 1]);
    table(&mut raw, "ZZ_TRANS", &[1, -1]);
    table(&mut raw, "ZZ_ATTRIBUTE", &[0, 9]);
    let e = JFlexTables::load(&zlib(&raw)).unwrap_err().to_string();
    assert!(e.contains("ZZ_CMAP_BLOCKS"), "{e}");
}

/// A dropped scanner's buffer serves the next scanner of its size on
/// the thread (a few are kept, grown ones are not), and a reused
/// buffer scans as a fresh one does.
#[test]
fn dropped_buffers_are_reused() {
    let tables: &'static JFlexTables = Box::leak(Box::new(JFlexTables::load(&tiny()).unwrap()));
    let scan_all = |s: &mut JFlexScanner, text: &str| {
        let mut r = StrReader::new(text);
        let mut n = 0;
        while s.scan(&mut r).unwrap() != Scan::Eof {
            n += 1;
        }
        n
    };
    let scanners: Vec<JFlexScanner> = (0..SPARE_BUFFERS + 2)
        .map(|_| JFlexScanner::new(tables, 7))
        .collect();
    drop(scanners);
    let spare = || SPARE.with(|s| s.borrow().iter().filter(|b| b.len() == 7).count());
    assert_eq!(spare(), SPARE_BUFFERS);
    let mut s = JFlexScanner::new(tables, 7);
    assert_eq!(spare(), SPARE_BUFFERS - 1);
    assert_eq!(scan_all(&mut s, "abcdefghij"), 10);
    drop(s);
    let mut s = JFlexScanner::new(tables, 7);
    assert_eq!(scan_all(&mut s, "xy"), 2);
    // a buffer that no longer has its size is not kept
    s.buffer.push(0);
    let before = spare();
    drop(s);
    assert_eq!(spare(), before);
}

#[test]
fn scanner_reads_across_refills_and_surrogates() {
    let tables: &'static JFlexTables = Box::leak(Box::new(JFlexTables::load(&tiny()).unwrap()));
    // A fixed 3-unit buffer: pairs straddle refills.
    let mut s = JFlexScanner::new(tables, 3);
    let mut r = StrReader::new("a😀b😀");
    let mut units = Vec::new();
    loop {
        match s.scan(&mut r).unwrap() {
            Scan::Eof => break,
            Scan::Action(a) => {
                assert_eq!(a, 1);
                units.push(s.text().to_vec());
                assert_eq!(s.yycharat(0), s.text()[0]);
            }
        }
    }
    assert_eq!(units.len(), 4);
    assert_eq!(String::from_utf16(&units.concat()).unwrap(), "a😀b😀");
    assert!(s.at_eof());
    assert_eq!(
        s.yychar, 6,
        "at the end, yychar is the input length in UTF-16 units"
    );
    assert!(s.yypushback(5).is_err());
    s.yypushback(0).unwrap();
    s.yyclose();
    s.set_buffer_size(8);
    s.yyreset();
    assert_eq!(s.buffer.len(), 8);
    s.set_buffer_size(2);
    s.buffer = vec![0; 9];
    s.yyreset();
    assert_eq!(s.buffer.len(), 2, "yyreset shrinks a grown buffer");
    // The growing skeleton doubles its buffer.
    let mut g = JFlexScanner::with_growth(tables, 2, true);
    let mut r = StrReader::new("😀😀😀");
    let mut n = 0;
    while let Scan::Action(_) = g.scan(&mut r).unwrap() {
        n += 1;
    }
    assert_eq!(n, 3);
}

/// A reader that hands out one unit per `read`.
struct OneAtATime(StrReader);

impl CharReader for OneAtATime {
    fn read(&mut self, buf: &mut [u16]) -> Result<usize, AnalysisError> {
        let n = buf.len().min(1);
        self.0.read(&mut buf[..n])
    }
}

fn all_units(s: &mut JFlexScanner, r: &mut dyn CharReader) -> String {
    let mut units = Vec::new();
    while let Scan::Action(_) = s.scan(r).unwrap() {
        units.extend_from_slice(s.text());
    }
    String::from_utf16(&units).unwrap()
}

#[test]
fn refill_edge_cases() {
    let tables: &'static JFlexTables = Box::leak(Box::new(JFlexTables::load(&tiny()).unwrap()));
    // A high surrogate read last is held back for the next refill.
    let mut s = JFlexScanner::new(tables, 2);
    assert_eq!(all_units(&mut s, &mut StrReader::new("a😀b")), "a😀b");
    // A one-unit refill that reads a lone high surrogate holds it back.
    let mut s = JFlexScanner::new(tables, 2);
    assert_eq!(all_units(&mut s, &mut StrReader::new("ab😀c")), "ab😀c");
    // A short read ending in a high surrogate reads its low half.
    let mut s = JFlexScanner::new(tables, 8);
    assert_eq!(
        all_units(&mut s, &mut OneAtATime(StrReader::new("😀x"))),
        "😀x"
    );
    let mut s = JFlexScanner::with_growth(tables, 8, true);
    assert_eq!(
        all_units(&mut s, &mut OneAtATime(StrReader::new("x😀"))),
        "x😀"
    );
}

#[test]
fn offsets_by_code_points() {
    let tables: &'static JFlexTables = Box::leak(Box::new(JFlexTables::load(&tiny()).unwrap()));
    let mut s = JFlexScanner::new(tables, 16);
    s.buffer[..5].copy_from_slice(&"a😀b😀".encode_utf16().collect::<Vec<_>>()[..5]);
    s.end_read = 5;
    assert_eq!(s.offset_by_code_points(0, 2), 3);
    assert_eq!(s.offset_by_code_points(0, 9), 5);
    assert_eq!(s.offset_by_code_points(3, -1), 1);
    assert_eq!(s.offset_by_code_points(4, -9), 0);
    assert_eq!(s.yylength(), 0);
}
