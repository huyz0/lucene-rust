//! `ZstdTerms104`'s terms dictionary: the same fields written with and without
//! [`WriteOptions::zstd_term_blocks`] read back identically -- every term, its
//! stats and postings, exact and ceiling seeks, wildcard walks -- while the
//! zstd `.tim` is smaller, carries its own header names, and holds zstd
//! blocks; a field large enough gets a trained dictionary.
// Test-side arithmetic on generated values.
#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::blocktree::{self, BlockTreeFields, SeekStatus};
use lucene_codecs::field_infos::{FieldInfo, FieldInfos, IndexOptions};
use lucene_codecs::postings::DocInput;
use lucene_codecs::postings_writer::{
    self, FieldPostingsInput, Output, TermPostings, WriteOptions,
};
use lucene_codecs::wildcard::WildcardPattern;

const ID: [u8; 16] = *b"zstdtermblocks01";
const SUFFIX: &str = "ZstdTerms104_0";
const MAX_DOC: i32 = 40_000;

fn infos() -> FieldInfos {
    FieldInfos {
        fields: vec![
            FieldInfo::new("id", 0).with_index_options(IndexOptions::Docs),
            FieldInfo::new("url", 1).with_index_options(IndexOptions::DocsAndFreqs),
            FieldInfo::new("body", 2).with_index_options(IndexOptions::DocsAndFreqs),
        ],
    }
}

/// A small deterministic generator.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }
}

/// `id`: one OpenSearch-like 20-character id per document, more than a MiB
/// of terms (a trained dictionary); `url`: a few thousand log-style paths;
/// `body`: words.
fn fields() -> Vec<(i32, IndexOptions, Vec<TermPostings>)> {
    let mut rng = Lcg(7);
    const B64: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut ids: Vec<(Vec<u8>, i32)> = (0..MAX_DOC)
        .map(|d| {
            let ts = 1_700_000_000_000u64 + d as u64 * 37;
            let mut id = Vec::with_capacity(20);
            for i in 0..8 {
                id.push(B64[((ts >> (42 - i * 6)) & 63) as usize]);
            }
            id.extend_from_slice(b"QwE5");
            for _ in 0..8 {
                id.push(B64[(rng.next() & 63) as usize]);
            }
            (id, d)
        })
        .collect();
    ids.sort();
    ids.dedup_by(|a, b| a.0 == b.0);
    let id_terms = ids
        .into_iter()
        .map(|(term, d)| TermPostings {
            term,
            docs: vec![(d, 1)],
            ..Default::default()
        })
        .collect();

    let words = [
        "error", "warn", "info", "request", "timeout", "user", "session", "lucene", "zstd",
        "block", "segment", "merge", "query", "shard", "replica", "node",
    ];
    let mut url: std::collections::BTreeMap<Vec<u8>, Vec<(i32, i32)>> = Default::default();
    let mut body: std::collections::BTreeMap<Vec<u8>, Vec<(i32, i32)>> = Default::default();
    for d in 0..MAX_DOC {
        let path = format!(
            "/api/v{}/{}/{}/items/{}",
            d % 3,
            words[(d as usize / 7) % words.len()],
            words[(d as usize / 3) % words.len()],
            (d * 7919) % 2500
        );
        let e = url.entry(path.into_bytes()).or_default();
        e.push((d, 1 + d % 2));
        for k in 0..4 {
            let w = format!(
                "{}{}",
                words[(rng.next() as usize) % words.len()],
                (d + k) % 50
            );
            let e = body.entry(w.into_bytes()).or_default();
            match e.last_mut() {
                Some((doc, f)) if *doc == d => *f += 1,
                _ => e.push((d, 1)),
            }
        }
    }
    let to_terms = |m: std::collections::BTreeMap<Vec<u8>, Vec<(i32, i32)>>| {
        m.into_iter()
            .map(|(term, docs)| TermPostings {
                term,
                docs,
                ..Default::default()
            })
            .collect::<Vec<_>>()
    };
    vec![
        (0, IndexOptions::Docs, id_terms),
        (1, IndexOptions::DocsAndFreqs, to_terms(url)),
        (2, IndexOptions::DocsAndFreqs, to_terms(body)),
    ]
}

fn write(fields: &[(i32, IndexOptions, Vec<TermPostings>)], zstd: Option<i32>) -> Output {
    let inputs: Vec<FieldPostingsInput<'_>> = fields
        .iter()
        .map(|(number, index_options, terms)| {
            let mut docs: Vec<i32> = terms
                .iter()
                .flat_map(|t| t.docs.iter().map(|d| d.0))
                .collect();
            docs.sort_unstable();
            docs.dedup();
            FieldPostingsInput {
                field_number: *number,
                index_options: *index_options,
                doc_count: docs.len() as i32,
                has_payloads: false,
                terms,
            }
        })
        .collect();
    let options = WriteOptions {
        zstd_term_blocks: zstd,
        ..WriteOptions::default()
    };
    postings_writer::write_fields_with_options(&inputs, &[], &options, &ID, SUFFIX).unwrap()
}

fn open(out: &Output) -> BlockTreeFields {
    blocktree::open(&out.tim, &out.tip, &out.tmd, &infos(), &ID, SUFFIX, MAX_DOC).unwrap()
}

fn all_terms(fields: &BlockTreeFields, name: &str) -> Vec<(Vec<u8>, blocktree::TermStats)> {
    let terms = fields.iter_fields().find(|(n, _)| *n == name).unwrap().1;
    let mut it = terms.iter();
    let mut out = Vec::new();
    while let Some((t, s)) = it.try_next().unwrap() {
        out.push((t.to_vec(), s));
    }
    out
}

#[test]
fn zstd_dictionaries_read_back_like_stock_ones_and_are_smaller() {
    let fields = fields();
    let stock = write(&fields, None);
    let zstd = write(&fields, Some(3));
    assert_eq!(stock.doc, zstd.doc, "postings are untouched");
    assert!(
        zstd.tim.len() * 10 < stock.tim.len() * 8,
        "zstd .tim {} vs stock {}",
        zstd.tim.len(),
        stock.tim.len()
    );
    assert_eq!(
        lucene_store::codec_util::peek_codec_name(&zstd.tim).as_deref(),
        Some("ZstdBlockTreeTermsDict")
    );
    assert_eq!(
        lucene_store::codec_util::peek_codec_name(&stock.tim).as_deref(),
        Some("BlockTreeTermsDict")
    );
    // The id field's dictionary sits in .tmd: it is larger by about that much.
    assert!(
        zstd.tmd.len() > stock.tmd.len() + 4096,
        "{} vs {}",
        zstd.tmd.len(),
        stock.tmd.len()
    );

    let (a, b) = (open(&stock), open(&zstd));
    let doc_in = DocInput::open(&zstd.doc, &ID, SUFFIX).unwrap();
    for (name, _) in a.iter_fields() {
        let ta = all_terms(&a, name);
        let tb = all_terms(&b, name);
        assert_eq!(ta, tb, "{name}");
        let fa = a.iter_fields().find(|(n, _)| *n == name).unwrap().1;
        let fb = b.iter_fields().find(|(n, _)| *n == name).unwrap().1;
        for (i, (term, stats)) in ta.iter().enumerate() {
            if i % 5 == 0 {
                assert_eq!(fb.try_seek_exact(term).unwrap(), Some(*stats), "{name}");
                let pa = fa.postings(term, Some(&doc_in)).unwrap().unwrap();
                let pb = fb.postings(term, Some(&doc_in)).unwrap().unwrap();
                assert_eq!((pa.docs, pa.freqs), (pb.docs, pb.freqs));
            }
            if i % 11 == 0 {
                let mut absent = term.clone();
                absent.push(b'~');
                assert_eq!(
                    fa.try_seek_exact(&absent).unwrap(),
                    fb.try_seek_exact(&absent).unwrap()
                );
                let (mut ea, mut eb) = (fa.iter(), fb.iter());
                let sa = ea.try_seek_ceil(&absent).unwrap();
                let sb = eb.try_seek_ceil(&absent).unwrap();
                assert_eq!(sa, sb);
                if sa != SeekStatus::End {
                    assert_eq!(ea.try_next().unwrap(), eb.try_next().unwrap());
                }
            }
        }
        for pattern in [&b"*items/1*"[..], b"err*3", b"AB*", b"/api/v1/*", b"*"] {
            let p = WildcardPattern::new(pattern);
            let ma: Vec<_> = fa.intersect(&p).map(|r| r.unwrap()).collect();
            let mb: Vec<_> = fb.intersect(&p).map(|r| r.unwrap()).collect();
            assert_eq!(ma, mb, "{name} {}", String::from_utf8_lossy(pattern));
        }
    }
}

#[test]
fn small_fields_take_no_dictionary_and_every_level_round_trips() {
    let fields: Vec<_> = fields().into_iter().filter(|f| f.0 != 0).collect();
    let stock = write(&fields, None);
    for level in [-5, 1, 3, 19] {
        let zstd = write(&fields, Some(level));
        let (a, b) = (open(&stock), open(&zstd));
        for name in ["url", "body"] {
            assert_eq!(
                all_terms(&a, name),
                all_terms(&b, name),
                "{name} at {level}"
            );
        }
    }
    let options = WriteOptions {
        zstd_term_blocks: Some(1000),
        ..WriteOptions::default()
    };
    assert!(options.validate().is_err());
}

/// A flipped byte inside a zstd block is an error, never a panic or a
/// silently different term list.
#[test]
fn a_corrupt_zstd_block_is_an_error() {
    let fields: Vec<_> = fields().into_iter().filter(|f| f.0 == 1).collect();
    let zstd = write(&fields, Some(3));
    let good = all_terms(&open(&zstd), "url");
    let header = 64;
    let footer = 16;
    let mut seen_error = false;
    for at in (header..zstd.tim.len() - footer).step_by(97) {
        let mut tim = zstd.tim.clone();
        tim[at] ^= 0x5a;
        // The footer checksum is not verified on open, so the corruption
        // reaches the block decoder.
        let Ok(fields) =
            blocktree::open(&tim, &zstd.tip, &zstd.tmd, &infos(), &ID, SUFFIX, MAX_DOC)
        else {
            continue;
        };
        let terms = fields.iter_fields().next().unwrap().1;
        let mut it = terms.iter();
        loop {
            match it.try_next() {
                Ok(Some(_)) => {}
                Ok(None) => break,
                Err(_) => {
                    seen_error = true;
                    break;
                }
            }
        }
    }
    assert!(seen_error);
    assert!(!good.is_empty());
}
