#![cfg(test)]
#![allow(clippy::arithmetic_side_effects)]

use super::*;

fn text(len: usize, seed: usize) -> Vec<u8> {
    let words: [&[u8]; 6] = [
        b"lucene ", b"zstd ", b"block ", b"index ", b"stored ", b"field ",
    ];
    let mut out = Vec::with_capacity(len);
    let mut i = seed;
    while out.len() < len {
        out.extend_from_slice(words[(i * 7 + i / 3) % words.len()]);
        i += 1;
    }
    out.truncate(len);
    out
}

#[test]
fn frames_round_trip_with_and_without_dictionaries() {
    let dict = text(4096, 1);
    let prepared_c = EncoderDictionary::new(&dict, 5).unwrap();
    let prepared_d = DecoderDictionary::new(&dict);
    let mut c = Compressor::new(5).unwrap();
    assert_eq!(c.level(), 5);
    for len in [1, 100, 10_000, 200_000] {
        let src = text(len, 3);
        let mut frames = vec![b'x'];
        let a = c.compress(&src, &mut frames).unwrap();
        let b = c.compress_with_dict(&dict, &src, &mut frames).unwrap();
        let p = c
            .compress_with_prepared(&prepared_c, &src, &mut frames)
            .unwrap();
        assert_eq!(frames.len(), 1 + a + b + p);
        let (fa, rest) = frames[1..].split_at(a);
        let (fb, fp) = rest.split_at(b);
        let mut d = Decompressor::new();
        for (frame, dict) in [
            (fa, Dictionary::None),
            (fb, Dictionary::Bytes(&dict)),
            (fp, Dictionary::Prepared(&prepared_d)),
            (fp, Dictionary::Bytes(&dict)),
        ] {
            let mut out = b"prefix".to_vec();
            d.decompress(frame, dict, len, &mut out).unwrap();
            assert_eq!(&out[..6], b"prefix");
            assert_eq!(&out[6..], &src[..]);
            let mut slice = vec![0u8; len + 9];
            assert_eq!(d.decompress_to_slice(frame, dict, &mut slice).unwrap(), len);
            assert_eq!(&slice[..len], &src[..]);
        }
    }
}

#[test]
fn the_expected_length_is_enforced() {
    let src = text(5000, 0);
    let mut frame = Vec::new();
    Compressor::new(DEFAULT_LEVEL)
        .unwrap()
        .compress(&src, &mut frame)
        .unwrap();
    let mut d = Decompressor::new();
    let mut out = vec![7u8];
    for wrong in [0, 4999, 5001] {
        let e = d
            .decompress(&frame, Dictionary::None, wrong, &mut out)
            .unwrap_err();
        assert_eq!(
            e,
            Error::Length {
                expected: wrong as u64,
                actual: 5000
            }
        );
        assert_eq!(out, [7]);
    }
    // Spare capacity beyond the expected length does not hide a longer frame.
    let mut roomy = Vec::with_capacity(1 << 16);
    d.decompress(&frame, Dictionary::None, 5000, &mut roomy)
        .unwrap();
    assert_eq!(roomy, src);
    let mut headerless = Vec::with_capacity(compress_bound(src.len()));
    let mut cctx = CCtx::create();
    cctx.set_parameter(zstd_safe::CParameter::ContentSizeFlag(false))
        .unwrap();
    cctx.compress2(&mut headerless, &src).unwrap();
    let mut roomy = Vec::with_capacity(1 << 16);
    assert_eq!(
        d.decompress(&headerless, Dictionary::None, 4000, &mut roomy)
            .unwrap_err(),
        Error::Length {
            expected: 4000,
            actual: 5000
        }
    );
    assert!(roomy.is_empty());
}

#[test]
fn a_frame_without_its_dictionary_or_corrupted_fails_cleanly() {
    let dict = text(2048, 9);
    let src = text(3000, 9);
    let mut frame = Vec::new();
    Compressor::new(1)
        .unwrap()
        .compress_with_dict(&dict, &src, &mut frame)
        .unwrap();
    let mut d = Decompressor::new();
    let mut out = Vec::new();
    assert!(
        d.decompress(&frame, Dictionary::None, 3000, &mut out)
            .is_err()
            || out != src
    );
    let mut out = Vec::new();
    assert!(matches!(
        d.decompress(
            &frame[..frame.len() / 2],
            Dictionary::Bytes(&dict),
            3000,
            &mut out
        ),
        Err(Error::Zstd(_))
    ));
    assert!(out.is_empty());
    let mut small = [0u8; 10];
    assert!(d
        .decompress_to_slice(&frame, Dictionary::Bytes(&dict), &mut small)
        .is_err());
}

#[test]
fn levels_are_checked() {
    assert!(levels().contains(&DEFAULT_LEVEL));
    let too_high = *levels().end() + 1;
    assert_eq!(
        Compressor::new(too_high).unwrap_err(),
        Error::Level(too_high)
    );
    assert!(EncoderDictionary::new(b"d", too_high).is_err());
    assert!(Error::Level(99).to_string().contains("99"));
    let c = Compressor::new(-5).unwrap();
    assert!(format!("{c:?}").contains("-5"));
    assert!(format!("{:?}", Decompressor::default()).contains("Decompressor"));
    assert!(format!("{:?}", DecoderDictionary::new(b"d")).contains("Decoder"));
    assert!(format!("{:?}", EncoderDictionary::new(b"d", 1).unwrap()).contains("Encoder"));
}

#[test]
fn the_thread_decompressor_is_reused_and_nesting_is_safe() {
    let src = text(1000, 2);
    let mut frame = Vec::new();
    Compressor::new(DEFAULT_LEVEL)
        .unwrap()
        .compress(&src, &mut frame)
        .unwrap();
    let out = with_decompressor(|d| {
        let inner = with_decompressor(|d2| {
            let mut o = Vec::new();
            d2.decompress(&frame, Dictionary::None, 1000, &mut o)
                .unwrap();
            o
        });
        let mut o = Vec::new();
        d.decompress(&frame, Dictionary::None, 1000, &mut o)
            .unwrap();
        assert_eq!(o, inner);
        o
    });
    assert_eq!(out, src);
}

#[cfg(feature = "dict-builder")]
#[test]
fn a_trained_dictionary_compresses_small_blocks_better() {
    let samples: Vec<Vec<u8>> = (0..2000)
        .map(|i| {
            format!(
                "{{\"id\":{i},\"user\":\"user{}\",\"status\":\"ok\",\"tags\":[\"a\",\"b\"]}}",
                i % 37
            )
            .into_bytes()
        })
        .collect();
    let refs: Vec<&[u8]> = samples.iter().map(Vec::as_slice).collect();
    let dict = train_dictionary(&refs, 4096).unwrap();
    let e = EncoderDictionary::new(&dict, 3).unwrap();
    let dd = DecoderDictionary::new(&dict);
    let mut c = Compressor::new(3).unwrap();
    let (mut plain, mut with) = (Vec::new(), Vec::new());
    c.compress(&samples[5], &mut plain).unwrap();
    c.compress_with_prepared(&e, &samples[5], &mut with)
        .unwrap();
    assert!(with.len() < plain.len());
    let mut out = Vec::new();
    Decompressor::new()
        .decompress(&with, Dictionary::Prepared(&dd), samples[5].len(), &mut out)
        .unwrap();
    assert_eq!(out, samples[5]);
    assert!(train_dictionary(&[], 4096).is_err());
}
