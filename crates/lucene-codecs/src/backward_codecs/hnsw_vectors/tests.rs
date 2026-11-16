#![cfg(test)]
// The arithmetic gate is about values read off disk; a test's `i + 1` is
// not one. See docs/arithmetic-gate.md.
#![allow(clippy::arithmetic_side_effects)]
use super::*;

/// The five retired fixture indices of `fixtures/data/bwc` and the format
/// each one's vector fields use.
const VERSIONS: [(&str, RetiredHnswFormat); 5] = [
    ("9.0.0", RetiredHnswFormat::Lucene90),
    ("9.1.0", RetiredHnswFormat::Lucene91),
    ("9.3.0", RetiredHnswFormat::Lucene92),
    ("9.4.2", RetiredHnswFormat::Lucene94),
    ("9.8.0", RetiredHnswFormat::Lucene95),
];

struct Files {
    vem: Vec<u8>,
    vec: Vec<u8>,
    vex: Vec<u8>,
    id: [u8; ID_LENGTH],
    suffix: String,
}

/// Segment `_1` (200 vectors a field) of `version`, with the segment id
/// and suffix read back off the `.vem` header itself.
fn files(version: &str, format: RetiredHnswFormat) -> Files {
    let dir = format!(
        "{}/../../fixtures/data/bwc/{version}/",
        env!("CARGO_MANIFEST_DIR")
    );
    let read = |ext: &str| {
        std::fs::read(format!("{dir}_1_{}_0.{ext}", format.name())).expect("bwc fixture")
    };
    let vem = read("vem");
    let mut input = SliceInput::new(&vem);
    input.read_be_u32().unwrap();
    input.read_string().unwrap();
    input.read_be_u32().unwrap();
    let mut id = [0u8; ID_LENGTH];
    input.read_bytes(&mut id).unwrap();
    let len = input.read_byte().unwrap() as usize;
    let mut suffix = vec![0u8; len];
    input.read_bytes(&mut suffix).unwrap();
    Files {
        vec: read("vec"),
        vex: read("vex"),
        vem,
        id,
        suffix: String::from_utf8(suffix).unwrap(),
    }
}

fn open<'a>(
    format: RetiredHnswFormat,
    f: &'a Files,
    vem: &[u8],
) -> Result<RetiredHnswVectorsReader<'a>> {
    RetiredHnswVectorsReader::open(format, vem, &f.vec, &f.vex, &f.id, &f.suffix)
}

/// Reads everything a reader can serve: every vector, both directions
/// of the ordinal map, every node's neighbours on every level, and a
/// search per field.
fn walk_everything(r: &RetiredHnswVectorsReader<'_>) -> Result<()> {
    for entry in r.flat().fields().to_vec() {
        let n = entry.field_number;
        let graph = r.graph(n)?;
        let mut out = Vec::new();
        for level in 0..graph.num_levels() {
            for node in graph.sorted_nodes_on_level(level)? {
                graph.neighbors_into(level, node, &mut out)?;
            }
        }
        match entry.encoding {
            VectorEncoding::Float32 => {
                let v = r.flat().float_vector_values(n)?;
                let mut cursor = v.doc_to_ord()?;
                for ord in 0..v.size() {
                    v.vector(ord)?;
                    let doc = v.ord_to_doc(ord)?;
                    if doc < 0 || cursor.ordinal(doc)? != Some(ord) {
                        return corrupt("doc_to_ord disagrees with ord_to_doc");
                    }
                }
                let q = vec![0.25f32; v.dimension()];
                r.search(n, &mut v.scorer(&q)?, 10, u64::MAX, None, None)?;
            }
            VectorEncoding::Byte => {
                let v = r.flat().byte_vector_values(n)?;
                for ord in 0..v.size() {
                    v.vector(ord)?;
                    v.ord_to_doc(ord)?;
                }
                let q = vec![3u8; v.dimension()];
                r.search(n, &mut v.scorer(&q)?, 10, u64::MAX, None, None)?;
            }
        }
    }
    Ok(())
}

#[test]
fn format_names_round_trip_and_keep_their_spelling() {
    for f in RetiredHnswFormat::ALL {
        assert_eq!(RetiredHnswFormat::for_name(f.name()), Some(f));
    }
    assert_eq!(
        RetiredHnswFormat::Lucene92.name(),
        "lucene92HnswVectorsFormat"
    );
    assert_eq!(
        RetiredHnswFormat::Lucene94.codec_prefix(),
        "lucene94HnswVectorsFormat"
    );
    assert_eq!(
        RetiredHnswFormat::for_name("Lucene99HnswVectorsFormat"),
        None
    );
    assert_eq!(
        RetiredHnswFormat::for_name("Lucene92HnswVectorsFormat"),
        None
    );
    assert!(RetiredHnswFormat::Lucene90.neighbors_sorted());
    assert!(!RetiredHnswFormat::Lucene91.neighbors_sorted());
    assert!(RetiredHnswFormat::Lucene95.neighbors_sorted());
}

#[test]
fn every_retired_fixture_opens_walks_and_searches() {
    for (version, format) in VERSIONS {
        let f = files(version, format);
        let r = open(format, &f, &f.vem).unwrap_or_else(|e| panic!("{version}: {e}"));
        assert_eq!(r.format(), format);
        assert_eq!(r.hits_are_ordinals(), format == RetiredHnswFormat::Lucene90);
        walk_everything(&r).unwrap_or_else(|e| panic!("{version}: {e}"));
        let fields = r.flat().fields();
        assert!(!fields.is_empty());
        for e in fields {
            assert_eq!(e.size, 200, "{version}: every other of 400 docs has one");
            assert_eq!(r.graph(e.field_number).unwrap().size(), 200);
            // `getVectorCount` (Lucene 10.6, #16647).
            assert_eq!(r.vector_count(e.field_number).unwrap(), 200);
        }
        assert!(matches!(r.graph(999), Err(Error::UnknownField(999))));
        assert!(matches!(r.vector_count(999), Err(Error::UnknownField(999))));
    }
}

/// Graph shape per format: `Lucene90` is one level with an unknown `M`;
/// the others are hierarchical, and a seek off a level is an error.
#[test]
fn graphs_report_their_shape_and_reject_bad_seeks() {
    for (version, format) in VERSIONS {
        let f = files(version, format);
        let r = open(format, &f, &f.vem).unwrap();
        let n = r.flat().fields()[0].field_number;
        let g = r.graph(n).unwrap();
        let mut out = Vec::new();
        assert!(g.neighbors_into(0, -1, &mut out).is_err(), "{version}");
        assert!(g.neighbors_into(0, 200, &mut out).is_err(), "{version}");
        assert!(
            g.neighbors_into(g.num_levels(), 0, &mut out).is_err(),
            "{version}"
        );
        assert!(
            g.sorted_nodes_on_level(g.num_levels()).is_err(),
            "{version}"
        );
        if format == RetiredHnswFormat::Lucene90 {
            assert_eq!(
                (g.num_levels(), g.entry_node(), g.max_conn()),
                (1, 0, UNKNOWN_MAX_CONN)
            );
        } else {
            assert!(g.num_levels() > 1, "{version}");
            assert!(g.max_conn() >= 16, "{version}");
            let top = g.sorted_nodes_on_level(g.num_levels() - 1).unwrap();
            assert_eq!(g.entry_node(), top[0]);
            // A node that is not on the top level cannot be sought there.
            let absent = (0..200).find(|x| !top.contains(x)).unwrap();
            assert!(g
                .neighbors_into(g.num_levels() - 1, absent, &mut out)
                .is_err());
        }
    }
}

#[test]
fn search_edge_cases() {
    for (version, format) in VERSIONS {
        let f = files(version, format);
        let r = open(format, &f, &f.vem).unwrap();
        let n = r
            .flat()
            .fields()
            .iter()
            .find(|e| e.encoding == VectorEncoding::Float32)
            .unwrap()
            .field_number;
        let v = r.flat().float_vector_values(n).unwrap();
        let q = vec![0.5f32; v.dimension()];
        let mut scorer = v.scorer(&q).unwrap();
        // k = 0 collects nothing.
        assert!(r
            .search(n, &mut scorer, 0, u64::MAX, None, None)
            .unwrap()
            .0
            .is_empty());
        // A short accept set is a caller error, not a panic.
        let short = FixedBitSet::new(10);
        assert!(matches!(
            r.search(n, &mut scorer, 5, u64::MAX, Some(&short), None),
            Err(Error::InvalidGraphParameter(_))
        ));
        // Accepting only even ordinals returns only even ordinals.
        let mut even = FixedBitSet::new(200);
        for o in (0..200).step_by(2) {
            even.set(o);
        }
        let (hits, early) = r
            .search(n, &mut scorer, 5, u64::MAX, Some(&even), None)
            .unwrap();
        assert!(!early);
        assert_eq!(hits.len(), 5, "{version}");
        assert!(hits.iter().all(|(o, _)| o % 2 == 0), "{version}: {hits:?}");
        assert!(hits.windows(2).all(|w| w[0].1 >= w[1].1));
        // A visit limit of one stops the walk early.
        let (_, early) = r.search(n, &mut scorer, 5, 1, None, None).unwrap();
        assert!(early, "{version}");
        // Seeds reach the hierarchical formats; Lucene90 ignores them.
        let (seeded, _) = r
            .search(n, &mut scorer, 5, u64::MAX, None, Some(&[1, 2, 3]))
            .unwrap();
        assert_eq!(seeded.len(), 5);
    }
}

/// Every single-bit flip of each retired `.vem` (and of the two smallest
/// `.vex` files), re-signed so the checksum passes, either decodes
/// cleanly or is rejected with an error -- never a panic.
#[test]
fn every_resigned_single_byte_corruption_is_an_error_or_a_clean_decode() {
    fn repack(buf: &[u8], at: usize, bit: u8) -> Vec<u8> {
        let mut body = buf[..buf.len() - codec_util::FOOTER_LENGTH].to_vec();
        body[at] ^= 1 << bit;
        codec_util::write_footer(&mut body);
        body
    }
    let mut meta_rejected = 0usize;
    let mut meta_flipped = 0usize;
    for (version, format) in VERSIONS {
        let f = files(version, format);
        for at in 0..f.vem.len() - codec_util::FOOTER_LENGTH {
            for bit in [0u8, 7] {
                let vem = repack(&f.vem, at, bit);
                meta_flipped += 1;
                let bad = open(format, &f, &vem)
                    .and_then(|r| walk_everything(&r))
                    .is_err();
                meta_rejected += usize::from(bad);
            }
        }
        if matches!(
            format,
            RetiredHnswFormat::Lucene90 | RetiredHnswFormat::Lucene95
        ) {
            for at in 0..f.vex.len() - codec_util::FOOTER_LENGTH {
                let vex = repack(&f.vex, at, 0);
                let _ =
                    RetiredHnswVectorsReader::open(format, &f.vem, &f.vec, &vex, &f.id, &f.suffix)
                        .and_then(|r| walk_everything(&r));
            }
        }
        let _ = version;
    }
    assert!(
        meta_rejected > meta_flipped / 2,
        "only {meta_rejected} of {meta_flipped} .vem flips rejected"
    );
}

#[test]
fn mismatched_headers_and_short_files_are_rejected() {
    let (version, format) = VERSIONS[1];
    let f = files(version, format);
    // The wrong format for the files.
    assert!(open(RetiredHnswFormat::Lucene90, &f, &f.vem).is_err());
    // A data file that is not this format's.
    assert!(
        RetiredHnswVectorsReader::open(format, &f.vem, &f.vex, &f.vec, &f.id, &f.suffix).is_err()
    );
    // A wrong segment id.
    assert!(RetiredHnswVectorsReader::open(
        format,
        &f.vem,
        &f.vec,
        &f.vex,
        &[0; ID_LENGTH],
        &f.suffix
    )
    .is_err());
    // Truncation.
    assert!(open(format, &f, &f.vem[..f.vem.len() - 1]).is_err());
    assert!(
        RetiredHnswVectorsReader::open(format, &f.vem, &f.vec[..8], &f.vex, &f.id, &f.suffix)
            .is_err()
    );
    assert!(check_region(-1, 1, 10).is_err());
    assert!(check_region(5, 6, 10).is_err());
    assert!(check_region(5, 5, 10).is_ok());
}
