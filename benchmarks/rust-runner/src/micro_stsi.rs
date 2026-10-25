//! `SimpleTextSiMicro.java`'s twin: typed .si read/write over all 184 fixtures.
use std::hint::black_box;
use std::time::Duration;

use lucene_index::segment_info;

fn unhex(value: &str) -> Vec<u8> {
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

pub fn bench(warmup: Duration, measure: Duration, root: &str) {
    let manifest = std::fs::read_to_string(format!("{root}/cases.tsv")).unwrap();
    let mut cases = Vec::new();
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for line in manifest.lines() {
        let row: Vec<_> = line.split('\t').collect();
        let id = unhex(row[1]).try_into().unwrap();
        let bytes = std::fs::read(format!("{root}/{}.si", row[0])).unwrap();
        let mut info = segment_info::parse_simpletext(&bytes, row[0], &id).unwrap();
        assert_eq!(
            segment_info::describe_index_sort(
                (!info.index_sort.is_empty()).then_some(info.index_sort.as_slice())
            ),
            String::from_utf8(unhex(row[2])).unwrap()
        );
        let encoded = segment_info::write_simpletext(&mut info).unwrap();
        assert_eq!(encoded, bytes);
        hash = (hash ^ encoded.len() as u64).wrapping_mul(0x0000_0100_0000_01b3);
        for byte in encoded {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
        }
        cases.push((row[0].to_string(), id, bytes, info));
    }
    println!("#check\tcx_st_si_read\t{hash:016x}\t1");
    println!("#check\tcx_st_si_write\t{hash:016x}\t1");
    super::measure("cx_st_si_read", warmup, measure, || {
        for (name, id, bytes, _) in &cases {
            black_box(segment_info::parse_simpletext(black_box(bytes), name, id).unwrap());
        }
        cases.len() as u64
    });
    let count = cases.len() as u64;
    super::measure("cx_st_si_write", warmup, measure, || {
        for (_, _, _, info) in &mut cases {
            black_box(segment_info::write_simpletext(black_box(info)).unwrap());
        }
        count
    });
}
