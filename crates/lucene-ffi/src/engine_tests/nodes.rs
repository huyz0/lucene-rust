//! The `nodes` tests that run through the exported entry points.

#[cfg(test)]
pub(crate) mod tests {
    use crate::jvm_reader::tests::run_limit;
    use crate::jvm_reader::{ffi_close_jvm_reader, ffi_open_jvm_reader};

    use lucene_engine::nodes::*;
    use lucene_engine::query_blob::Cursor;
    use lucene_engine::status::Status;
    use lucene_index::document::{
        Document, NumericDocValuesField, SortedNumericDocValuesField, SortedSetDocValuesField,
        Store, StringField, TextField,
    };
    use lucene_index::index_writer::IndexWriter;
    use lucene_index::segment_info::LuceneVersion;

    use lucene_search::join::BitSetProducer;

    use lucene_search::query::{BooleanQuery, Clause};

    use lucene_store::FsDirectory;

    use lucene_util::fixed_bit_set::FixedBitSet;
    use lucene_util::test_support::TempDir;

    use std::sync::Arc;

    pub use lucene_engine::query_blob::test_support::Blob;

    const VERSION: LuceneVersion = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };

    pub const BODIES: [&str; 6] = [
        "alpha beta gamma",
        "beta alpha delta alpha",
        "gamma gamma alpha",
        "delta beta",
        "alpha delta beta gamma alpha",
        "epsilon",
    ];

    /// Twelve blocks in two segments, OpenSearch's nested layout: block `i`
    /// has `i % 3` nested `comments` before its root, which carries `body`,
    /// `title`, `_primary_term`, and the numeric and keyword doc values the
    /// function scores read (`pos` = `i + 1`, `sd` a double, `sf` a float,
    /// `si` two integers on even blocks, `kw`).
    pub fn index(tag: &str) -> TempDir {
        let tmp = TempDir::new(tag);
        let dir = FsDirectory::open(tmp.path());
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
        for i in 0..12usize {
            let mut docs = Vec::new();
            for j in 0..i % 3 {
                let mut c = Document::new();
                c.add(StringField::new("_nested_path", "comments", Store::No));
                c.add(TextField::new(
                    "c_body",
                    if (i + j) % 2 == 0 {
                        "alpha"
                    } else {
                        "beta alpha"
                    },
                    Store::No,
                ));
                docs.push(c);
            }
            let mut d = Document::new();
            d.add(StringField::new("id", i.to_string(), Store::Yes));
            d.add(TextField::new("body", BODIES[i % 6], Store::No));
            d.add(TextField::new("title", BODIES[(i + 2) % 6], Store::No));
            d.add(NumericDocValuesField::new("_primary_term", 1));
            d.add(SortedNumericDocValuesField::new("pos", i as i64 + 1));
            let sd = (i as f64 - 5.5) * 3.25;
            let bits = sd.to_bits() as i64;
            d.add(SortedNumericDocValuesField::new(
                "sd",
                bits ^ ((bits >> 63) & i64::MAX),
            ));
            let sf = (i as f32) * 0.75 - 2.0;
            let fbits = sf.to_bits() as i32;
            d.add(SortedNumericDocValuesField::new(
                "sf",
                i64::from(fbits ^ ((fbits >> 31) & i32::MAX)),
            ));
            if i % 2 == 0 {
                d.add(SortedNumericDocValuesField::new("si", i as i64));
                d.add(SortedNumericDocValuesField::new("si", 3 * i as i64 + 1));
            }
            d.add(SortedSetDocValuesField::new(
                "kw",
                format!("k{}", i % 4).into_bytes(),
            ));
            docs.push(d);
            w.add_fields_documents(&docs).unwrap();
            if i == 5 {
                w.commit().unwrap();
            }
        }
        w.commit().unwrap();
        tmp
    }

    /// The index opened as the plugin opens a searcher's reader.
    pub fn open(tmp: &TempDir) -> u64 {
        crate::register_codecs();
        let dir = FsDirectory::open(tmp.path());
        let reader = lucene_search::directory_reader::DirectoryReader::open(&dir).unwrap();
        let name = std::fs::read_dir(tmp.path())
            .unwrap()
            .filter_map(|e| e.ok()?.file_name().into_string().ok())
            .filter(|n| n.starts_with("segments_"))
            .max()
            .unwrap();
        let generation = i64::from_str_radix(&name["segments_".len()..], 36).unwrap();
        let bytes = std::fs::read(tmp.path().join(&name)).unwrap();
        let max_docs: Vec<i32> = reader.segment_readers().iter().map(|s| s.max_doc).collect();
        let path = tmp.path().to_str().unwrap();
        let mut handle = 0u64;
        // SAFETY: live buffers of the stated lengths.
        let rc = unsafe {
            ffi_open_jvm_reader(
                path.as_ptr().cast(),
                path.len(),
                bytes.as_ptr(),
                bytes.len(),
                generation,
                0,
                max_docs.as_ptr(),
                max_docs.len(),
                std::ptr::null(),
                std::ptr::null(),
                &mut handle,
            )
        };
        assert_eq!(rc, 0, "{}", lucene_engine::status::last_error());
        handle
    }

    /// `(hits, total)` of `blob`, top 20, counted exactly.
    pub fn search(h: u64, blob: &Blob) -> (Vec<(i32, f32)>, i64) {
        let (hits, total, _) = run_limit(h, &blob.0, 20, i64::MAX)
            .unwrap_or_else(|rc| panic!("status {rc}: {}", lucene_engine::status::last_error()));
        (hits, total)
    }

    /// The status a blob fails with, and the last error.
    pub fn fails(h: u64, blob: &Blob) -> (i32, String) {
        match run_limit(h, &blob.0, 5, i64::MAX) {
            Ok(r) => panic!("expected an error, got {r:?}"),
            Err(rc) => (rc, lucene_engine::status::last_error()),
        }
    }

    fn decode(blob: &Blob) -> std::result::Result<Clause, Status> {
        let mut c = Cursor::new(&blob.0[1..]);
        let mut nodes = 0;
        lucene_engine::query_blob::decode_node(&mut c, 0, &mut nodes)
    }

    /// The roots: documents with `_primary_term`.
    fn roots(h: u64) -> Vec<i32> {
        let mut d: Vec<i32> = search(h, &Blob::tree().exists("_primary_term"))
            .0
            .into_iter()
            .map(|(d, _)| d)
            .collect();
        d.sort_unstable();
        d
    }

    fn nested(mode: u8, child: Blob) -> Blob {
        Blob::tree()
            .u8(NODE_TO_PARENT)
            .u8(mode)
            .exists("_primary_term")
            .u8(1)
            .i32(0)
            .i32(2)
            .u8(0)
            .raw(&child)
            .u8(1)
            .term("_nested_path", "comments")
    }

    #[test]
    fn a_nested_query_finds_the_roots_of_matching_children_in_every_score_mode() {
        let tmp = index("jvm-nodes-nested");
        let h = open(&tmp);
        let roots = roots(h);
        assert_eq!(roots.len(), 12);
        // Blocks 1, 2, 4, 5, 7, 8, 10, 11 have children, all with "alpha".
        let mut scores = Vec::new();
        for mode in 0..=4u8 {
            let (hits, total) = search(h, &nested(mode, Blob::default().term("c_body", "alpha")));
            assert_eq!(total, 8, "mode {mode}");
            assert!(
                hits.iter().all(|(d, _)| roots.contains(d)),
                "mode {mode}: {hits:?}"
            );
            scores.push(hits.iter().map(|&(_, s)| s).fold(0.0f32, f32::max));
        }
        // None scores 0; Total of two children is above Max.
        assert_eq!(scores[0], 0.0);
        assert!(scores[3] > scores[2], "{scores:?}");
        // "beta" is on half the children only.
        let (_, beta) = search(h, &nested(2, Blob::default().term("c_body", "beta")));
        assert!(beta > 0 && beta < 8, "{beta}");
        // Equal filters share their cached sets (and the cache survives a second run).
        let again = search(h, &nested(1, Blob::default().term("c_body", "alpha")));
        assert_eq!(again.1, 8);
        // A child query that matches a root is Lucene's IllegalStateException.
        let unfiltered = Blob::tree()
            .u8(NODE_TO_PARENT)
            .u8(1)
            .exists("_primary_term")
            .u8(5);
        let (rc, msg) = fails(h, &unfiltered);
        assert_eq!(rc, Status::Search.code(), "{msg}");
        ffi_close_jvm_reader(h);
    }

    #[test]
    fn the_parent_cache_is_bounded_by_bytes_and_by_the_open_segments() {
        let p = SharedParents {
            key: "aa".into(),
            query: BooleanQuery::default(),
        };
        assert_eq!(p.key(), "parents(aa)");
        // A cache of its own: the process-wide one is shared by parallel tests.
        let mut c = ParentsCache::default();
        let seg = |n: &str| [n.as_bytes()[0]; 16];
        let set = |bits: usize| Some(Arc::new(FixedBitSet::new(bits)));
        c.insert(seg("a"), "f".into(), set(8 * 1024));
        c.insert(seg("a"), "g".into(), None);
        c.insert(seg("b"), "f".into(), set(8 * 2048));
        assert_eq!((c.entries, c.bytes), (3, 3072));
        assert!(c.get(&seg("a"), "g").is_some_and(|b| b.is_none()));
        assert!(c.get(&seg("a"), "h").is_none());
        // Replacing an entry accounts for the old set.
        c.insert(seg("b"), "f".into(), set(8 * 1024));
        assert_eq!((c.entries, c.bytes), (3, 2048));
        // Only the open segments' sets survive a reader closing.
        c.retain(&[seg("b")].into_iter().collect());
        assert_eq!((c.entries, c.bytes), (1, 1024));
        // A set that would pass the byte cap clears the cache first.
        c.insert(
            seg("c"),
            "f".into(),
            set(8 * (MAX_CACHED_PARENT_BYTES - 512)),
        );
        assert_eq!(c.entries, 1);
        assert!(c.get(&seg("b"), "f").is_none());
        c.clear();
        assert_eq!((c.entries, c.bytes), (0, 0));
    }

    /// The index opened with block 1 (its child, document 1, and root,
    /// document 2, in the first segment) deleted.
    fn open_without_block_1(tmp: &TempDir) -> u64 {
        let dir = FsDirectory::open(tmp.path());
        let reader = lucene_search::directory_reader::DirectoryReader::open(&dir).unwrap();
        let name = std::fs::read_dir(tmp.path())
            .unwrap()
            .filter_map(|e| e.ok()?.file_name().into_string().ok())
            .filter(|n| n.starts_with("segments_"))
            .max()
            .unwrap();
        let generation = i64::from_str_radix(&name["segments_".len()..], 36).unwrap();
        let bytes = std::fs::read(tmp.path().join(&name)).unwrap();
        let max_docs: Vec<i32> = reader.segment_readers().iter().map(|s| s.max_doc).collect();
        let words: Vec<u64> = vec![
            ((1u64 << max_docs[0]) - 1) & !0b110,
            (1u64 << max_docs[1]) - 1,
        ];
        let counts = [1usize, 1];
        let path = tmp.path().to_str().unwrap();
        let mut handle = 0u64;
        // SAFETY: live buffers of the stated lengths.
        let rc = unsafe {
            ffi_open_jvm_reader(
                path.as_ptr().cast(),
                path.len(),
                bytes.as_ptr(),
                bytes.len(),
                generation,
                0,
                max_docs.as_ptr(),
                max_docs.len(),
                words.as_ptr(),
                counts.as_ptr(),
                &mut handle,
            )
        };
        assert_eq!(rc, 0, "{}", lucene_engine::status::last_error());
        handle
    }

    #[test]
    fn parent_sets_are_shared_across_readers_and_dropped_with_their_segments() {
        let tmp = index("jvm-nodes-cache");
        let dir = FsDirectory::open(tmp.path());
        let reader = lucene_search::directory_reader::DirectoryReader::open(&dir).unwrap();
        let segments: Vec<SegmentKey> = reader
            .segment_readers()
            .iter()
            .map(|r| r.segment_id())
            .collect();
        let filter = lucene_engine::query_blob::hex(&Blob::default().exists("_primary_term").0);
        let cached = |s: &SegmentKey| lock(parents_cache()).get(s, &filter);
        assert!(segments.iter().all(|s| cached(s).is_none()));
        let h = open(&tmp);
        let blob = nested(2, Blob::default().term("c_body", "alpha"));
        let (first, total) = search(h, &blob);
        assert_eq!(total, 8);
        // Both segments' sets are cached, with the parents the filter matches.
        for s in &segments {
            let bits = cached(s).expect("cached").expect("parents");
            assert_eq!(bits.cardinality(), 6);
        }
        // A reader over the same segments with block 1 deleted reuses the sets
        // (deletions are not in them) and no longer finds that block.
        let h2 = open_without_block_1(&tmp);
        let set = cached(&segments[0]).unwrap().unwrap();
        let (hits, total) = search(h2, &blob);
        assert_eq!(total, 7);
        assert!(hits.iter().all(|&(d, _)| d != 2), "{hits:?}");
        assert!(first.iter().any(|&(d, _)| d == 2), "{first:?}");
        assert!(Arc::ptr_eq(&set, &cached(&segments[0]).unwrap().unwrap()));
        // Closing one reader keeps the sets the other still reads; closing the
        // last drops them.
        assert_eq!(ffi_close_jvm_reader(h), 0);
        assert!(segments.iter().all(|s| cached(s).is_some()));
        assert_eq!(ffi_close_jvm_reader(h2), 0);
        assert!(segments.iter().all(|s| cached(s).is_none()));
        // The process-wide counts track the cache (other tests may hold sets).
        let (entries, bytes) = parent_cache_stats();
        assert!(entries > 0 || bytes == 0);
    }

    fn span(tag: u8) -> Blob {
        Blob::default().u8(tag)
    }

    fn st(t: &str) -> Blob {
        span(SPAN_TERM).str("body").str(t)
    }

    fn spans(blob: Blob) -> Blob {
        Blob::tree().u8(NODE_SPAN).raw(&blob)
    }

    fn total(h: u64, blob: Blob) -> i64 {
        search(h, &blob).1
    }

    #[test]
    fn span_queries_decode_into_every_span_class() {
        let tmp = index("jvm-nodes-spans");
        let h = open(&tmp);
        // "alpha" is in bodies 0, 1, 2, 4 of every six.
        assert_eq!(total(h, spans(st("alpha"))), 8);
        let near = |slop: i32, ordered: u8| {
            span(SPAN_NEAR)
                .i32(slop)
                .u8(ordered)
                .i32(2)
                .raw(&st("alpha"))
                .raw(&st("beta"))
        };
        // "alpha beta" adjacent in order: bodies 0 only ("alpha beta gamma").
        assert_eq!(total(h, spans(near(0, 1))), 2);
        // Unordered within 1: bodies 0, 1 ("beta alpha"), 4 ("delta beta" ... "alpha delta beta").
        assert!(total(h, spans(near(1, 0))) >= 4);
        let or = span(SPAN_OR).i32(2).raw(&st("epsilon")).raw(&st("delta"));
        assert_eq!(total(h, spans(or)), 8);
        assert_eq!(
            total(h, spans(span(SPAN_FIRST).i32(1).raw(&st("alpha")))),
            4
        );
        assert_eq!(
            total(
                h,
                spans(span(SPAN_POSITION_RANGE).i32(1).i32(2).raw(&st("alpha")))
            ),
            2
        );
        let not = span(SPAN_NOT)
            .i32(1)
            .i32(1)
            .raw(&st("alpha"))
            .raw(&st("beta"));
        assert!(total(h, spans(not)) > 0);
        let big = span(SPAN_NEAR)
            .i32(5)
            .u8(0)
            .i32(2)
            .raw(&st("alpha"))
            .raw(&st("gamma"));
        let containing = span(SPAN_CONTAINING).raw(&big).raw(&st("beta"));
        assert!(total(h, spans(containing)) > 0);
        let within = span(SPAN_WITHIN).raw(&big).raw(&st("beta"));
        assert!(total(h, spans(within)) > 0);
        let masked = span(SPAN_NEAR).i32(3).u8(0).i32(2).raw(&st("alpha")).raw(
            &span(SPAN_FIELD_MASKING)
                .str("body")
                .raw(&span(SPAN_TERM).str("title").str("gamma")),
        );
        assert!(total(h, spans(masked)) > 0);
        // Malformed: an unknown tag, an empty near, clauses of two fields.
        let (rc, msg) = fails(h, &spans(span(99)));
        assert_eq!(rc, Status::InvalidArgument.code());
        assert!(msg.contains("unknown span tag 99"), "{msg}");
        let (_, msg) = fails(h, &spans(span(SPAN_OR).i32(0)));
        assert!(msg.contains("at least 1"), "{msg}");
        let two_fields = span(SPAN_OR)
            .i32(2)
            .raw(&st("a"))
            .raw(&span(SPAN_TERM).str("title").str("b"));
        let (_, msg) = fails(h, &spans(two_fields));
        assert!(msg.contains("span query"), "{msg}");
        let (_, msg) = fails(
            h,
            &spans(
                span(SPAN_NOT)
                    .i32(0)
                    .i32(0)
                    .raw(&st("a"))
                    .raw(&span(SPAN_TERM).str("title").str("b")),
            ),
        );
        assert!(msg.contains("span query"), "{msg}");
        let (_, msg) = fails(
            h,
            &spans(
                span(SPAN_CONTAINING)
                    .raw(&st("a"))
                    .raw(&span(SPAN_TERM).str("title").str("b")),
            ),
        );
        assert!(msg.contains("span query"), "{msg}");
        let (_, msg) = fails(
            h,
            &spans(
                span(SPAN_WITHIN)
                    .raw(&st("a"))
                    .raw(&span(SPAN_TERM).str("title").str("b")),
            ),
        );
        assert!(msg.contains("span query"), "{msg}");
        let (_, msg) = fails(
            h,
            &spans(
                span(SPAN_NEAR)
                    .i32(0)
                    .u8(1)
                    .i32(2)
                    .raw(&st("a"))
                    .raw(&span(SPAN_TERM).str("title").str("b")),
            ),
        );
        assert!(msg.contains("span query"), "{msg}");
        // Too deep: 40 nested span_first.
        let mut deep = st("alpha");
        for _ in 0..40 {
            deep = span(SPAN_FIRST).i32(9).raw(&deep);
        }
        let (_, msg) = fails(h, &spans(deep));
        assert!(msg.contains("nesting depth"), "{msg}");
        // Not UTF-8.
        let bad = spans(span(SPAN_TERM).bytes(&[0xff]).str("x"));
        assert_eq!(fails(h, &bad).0, Status::InvalidUtf8.code());
        ffi_close_jvm_reader(h);
    }

    fn src(tag: u8) -> Blob {
        Blob::default().u8(tag)
    }

    fn it(t: &str) -> Blob {
        src(SOURCE_TERM).str(t)
    }

    fn intervals(source: Blob) -> Blob {
        Blob::tree()
            .u8(NODE_INTERVAL)
            .str("body")
            .u8(0)
            .f32(1.0)
            .raw(&source)
    }

    fn list(tag: u8, items: &[Blob]) -> Blob {
        let mut b = src(tag).i32(items.len() as i32);
        for i in items {
            b = b.raw(i);
        }
        b
    }

    #[test]
    fn interval_sources_decode_into_every_source_class() {
        let tmp = index("jvm-nodes-intervals");
        let h = open(&tmp);
        assert_eq!(total(h, intervals(it("alpha"))), 8);
        // Sigmoid scoring.
        let sig = Blob::tree()
            .u8(NODE_INTERVAL)
            .str("body")
            .u8(1)
            .f32(2.0)
            .f32(0.5)
            .raw(&it("alpha"));
        assert_eq!(search(h, &sig).1, 8);
        let ab = [it("alpha"), it("beta")];
        assert_eq!(total(h, intervals(list(SOURCE_BLOCK, &ab))), 2);
        assert!(
            total(
                h,
                intervals(src(SOURCE_DISJUNCTION).u8(1).i32(2).raw(&ab[0]).raw(&ab[1]))
            ) >= 8
        );
        assert!(total(h, intervals(list(SOURCE_ORDERED, &ab))) > 0);
        assert!(total(h, intervals(list(SOURCE_UNORDERED, &ab))) > 0);
        // Two "alpha"s: bodies 1 and 4.
        let rep = src(SOURCE_REPEATING).i32(2).u8(1).raw(&it("alpha"));
        assert_eq!(total(h, intervals(rep)), 4);
        let rep2 = src(SOURCE_REPEATING).i32(2).u8(2).raw(&it("alpha"));
        assert_eq!(total(h, intervals(rep2.clone())), 4);
        let rep0 = src(SOURCE_REPEATING).i32(2).u8(0).raw(&it("alpha"));
        assert_eq!(total(h, intervals(rep0)), 4);
        let gaps = src(SOURCE_FILTERED)
            .u8(0)
            .i32(0)
            .raw(&list(SOURCE_UNORDERED, &ab));
        assert!(total(h, intervals(gaps)) > 0);
        let width = src(SOURCE_FILTERED)
            .u8(1)
            .i32(2)
            .raw(&list(SOURCE_UNORDERED, &ab));
        assert!(total(h, intervals(width)) > 0);
        assert_eq!(
            total(
                h,
                intervals(src(SOURCE_EXTENDED).i32(1).i32(1).raw(&it("alpha")))
            ),
            8
        );
        assert!(total(h, intervals(src(SOURCE_OFFSET).u8(1).raw(&it("beta")))) > 0);
        assert!(total(h, intervals(src(SOURCE_OFFSET).u8(0).raw(&it("beta")))) > 0);
        assert!(
            total(
                h,
                intervals(src(SOURCE_FIXED_FIELD).str("title").raw(&it("gamma")))
            ) > 0
        );
        assert_eq!(total(h, intervals(src(SOURCE_NO_MATCH).str("nothing"))), 0);
        let big = list(SOURCE_UNORDERED, &ab);
        for tag in SOURCE_CONTAINING..=SOURCE_NON_OVERLAPPING {
            let (a, b) = if tag == SOURCE_CONTAINED_BY {
                (it("gamma"), big.clone())
            } else {
                (big.clone(), it("gamma"))
            };
            search(h, &intervals(src(tag).raw(&a).raw(&b)));
        }
        let msm = src(SOURCE_MIN_SHOULD_MATCH)
            .i32(2)
            .i32(3)
            .raw(&it("alpha"))
            .raw(&it("beta"))
            .raw(&it("epsilon"));
        assert!(total(h, intervals(msm)) > 0);
        // Multi-term sources: every automaton type.
        let mt = |b: Blob| src(SOURCE_MULTI_TERM).i32(16).str("p").raw(&b);
        assert_eq!(
            total(h, intervals(mt(Blob::default().u8(AUTOMATON_NONE)))),
            0
        );
        assert_eq!(
            total(
                h,
                intervals(mt(Blob::default().u8(AUTOMATON_SINGLE).str("delta")))
            ),
            6
        );
        // `al*` over bytes: a -> l -> anything.
        let normal = Blob::default()
            .u8(AUTOMATON_NORMAL)
            .i32(3)
            .u8(0)
            .i32(1)
            .i32(1)
            .i32(i32::from(b'a'))
            .i32(i32::from(b'a'))
            .u8(0)
            .i32(1)
            .i32(2)
            .i32(i32::from(b'l'))
            .i32(i32::from(b'l'))
            .u8(1)
            .i32(1)
            .i32(2)
            .i32(0)
            .i32(255);
        assert_eq!(total(h, intervals(mt(normal))), 8);
        // Every term: more than 2 expansions is Lucene's error.
        let all = src(SOURCE_MULTI_TERM).i32(2).str("p").u8(AUTOMATON_ALL);
        let (rc, msg) = fails(h, &intervals(all));
        assert_eq!(rc, Status::Search.code(), "{msg}");
        ffi_close_jvm_reader(h);
    }

    #[test]
    fn malformed_interval_nodes_are_refused() {
        let tmp = index("jvm-nodes-intervals-bad");
        let h = open(&tmp);
        let bad = |b: Blob, want: &str| {
            let (rc, msg) = fails(h, &b);
            assert_eq!(rc, Status::InvalidArgument.code(), "{msg}");
            assert!(msg.contains(want), "{msg} lacks {want}");
        };
        bad(intervals(src(99)), "unknown interval source tag 99");
        bad(
            intervals(src(SOURCE_REPEATING).i32(0).u8(0).raw(&it("a"))),
            "0 copies",
        );
        // Each copy is an iterator and its cached positions: bounded as the
        // clauses Java's `Intervals.ordered` could have deduplicated.
        bad(
            intervals(src(SOURCE_REPEATING).i32(5000).u8(0).raw(&it("a"))),
            "5000 copies",
        );
        bad(
            intervals(src(SOURCE_REPEATING).i32(i32::MAX).u8(0).raw(&it("a"))),
            "copies",
        );
        bad(
            intervals(src(SOURCE_REPEATING).i32(2).u8(7).raw(&it("a"))),
            "repeating source name 7",
        );
        bad(
            intervals(src(SOURCE_FILTERED).u8(5).i32(1).raw(&it("a"))),
            "interval filter 5",
        );
        bad(intervals(list(SOURCE_ORDERED, &[])), "at least 1");
        bad(
            intervals(src(SOURCE_MIN_SHOULD_MATCH).i32(3).i32(1).raw(&it("a"))),
            "minimum should match 3",
        );
        bad(
            intervals(src(SOURCE_MIN_SHOULD_MATCH).i32(0).i32(1).raw(&it("a"))),
            "minimum should match 0",
        );
        let mt = |b: Blob| src(SOURCE_MULTI_TERM).i32(16).str("p").raw(&b);
        bad(
            intervals(src(SOURCE_MULTI_TERM).i32(5000).str("p").u8(0)),
            "maxExpansions [5000]",
        );
        bad(
            intervals(mt(Blob::default().u8(9))),
            "unknown automaton type 9",
        );
        bad(
            intervals(mt(Blob::default().u8(AUTOMATON_NORMAL).i32(0))),
            "automaton of 0 states",
        );
        bad(
            intervals(mt(Blob::default()
                .u8(AUTOMATON_NORMAL)
                .i32(1)
                .u8(1)
                .i32(1)
                .i32(5)
                .i32(0)
                .i32(1))),
            "transition 0 -> 5",
        );
        bad(
            intervals(mt(Blob::default()
                .u8(AUTOMATON_NORMAL)
                .i32(1)
                .u8(1)
                .i32(1)
                .i32(0)
                .i32(3)
                .i32(1))),
            "over 3..=1",
        );
        // Two transitions from one state over the same byte: not deterministic.
        let nfa = Blob::default()
            .u8(AUTOMATON_NORMAL)
            .i32(2)
            .u8(0)
            .i32(2)
            .i32(1)
            .i32(97)
            .i32(97)
            .i32(0)
            .i32(97)
            .i32(97)
            .u8(1)
            .i32(0);
        bad(intervals(mt(nfa)), "deterministic");
        let scoring = |kind: u8, pivot: f32| {
            Blob::tree()
                .u8(NODE_INTERVAL)
                .str("body")
                .u8(kind)
                .f32(pivot)
                .f32(1.0)
                .raw(&it("a"))
        };
        bad(scoring(7, 1.0), "unknown interval scoring 7");
        bad(scoring(0, -1.0), "pivot must be > 0");
        let mut deep = it("alpha");
        for _ in 0..40 {
            deep = src(SOURCE_EXTENDED).i32(0).i32(0).raw(&deep);
        }
        bad(intervals(deep), "nesting depth");
        assert_eq!(
            fails(
                h,
                &intervals(src(SOURCE_FIXED_FIELD).bytes(&[0xfe]).raw(&it("a")))
            )
            .0,
            Status::InvalidUtf8.code()
        );
        assert_eq!(
            fails(h, &intervals(src(SOURCE_NO_MATCH).bytes(&[0xfe]))).0,
            Status::InvalidUtf8.code()
        );
        assert_eq!(
            fails(
                h,
                &Blob::tree()
                    .u8(NODE_INTERVAL)
                    .bytes(&[0xfe])
                    .u8(0)
                    .f32(1.0)
                    .raw(&it("a"))
            )
            .0,
            Status::InvalidUtf8.code()
        );
        ffi_close_jvm_reader(h);
    }

    fn combined(fields: &[(&str, f32)], term: &str) -> Blob {
        let mut b = Blob::tree()
            .u8(NODE_COMBINED_FIELD)
            .str(term)
            .i32(fields.len() as i32);
        for (f, w) in fields {
            b = b.str(f).f32(*w);
        }
        b
    }

    #[test]
    fn combined_fields_and_the_other_node_errors() {
        let tmp = index("jvm-nodes-combined");
        let h = open(&tmp);
        let (hits, total) = search(h, &combined(&[("body", 1.0), ("title", 2.0)], "gamma"));
        assert!(
            total == 6 && hits.iter().all(|&(_, s)| s > 0.0),
            "{total} {hits:?}"
        );
        let (rc, msg) = fails(h, &combined(&[("body", 0.5)], "gamma"));
        assert_eq!(rc, Status::InvalidArgument.code());
        assert!(msg.contains("weight must be greater"), "{msg}");
        assert!(fails(h, &combined(&[], "gamma")).1.contains("at least 1"));
        assert_eq!(
            fails(
                h,
                &Blob::tree()
                    .u8(NODE_COMBINED_FIELD)
                    .str("t")
                    .i32(1)
                    .bytes(&[0xff])
                    .f32(1.0)
            )
            .0,
            Status::InvalidUtf8.code()
        );
        // An unknown score mode, and kinds past the last.
        let (_, msg) = fails(h, &Blob::tree().u8(NODE_TO_PARENT).u8(9));
        assert!(msg.contains("score mode 9"), "{msg}");
        let (_, msg) = fails(h, &Blob::tree().u8(lucene_engine::queries::NODE_LAST + 1));
        assert!(msg.contains("unknown node kind"), "{msg}");
        // The decode on its own: the nodes are extended clauses.
        assert!(matches!(decode(&spans(st("a"))), Ok(Clause::Extended(_))));
        ffi_close_jvm_reader(h);
    }

    /// The fuzz target's extended-query and term-range seeds are well-formed blobs.
    #[test]
    fn the_fuzz_seeds_decode() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/fuzz/seeds/jvm_search");
        let mut seen = 0;
        for e in std::fs::read_dir(dir).unwrap() {
            let e = e.unwrap();
            let name = e.file_name().to_string_lossy().to_string();
            if name.starts_with("extended_") || name == "term_range" {
                let data = std::fs::read(e.path()).unwrap();
                lucene_engine::query_blob::decode_request(&data[2..]).unwrap_or_else(|s| {
                    panic!(
                        "{:?}: {s:?} {}",
                        e.path(),
                        lucene_engine::status::last_error()
                    )
                });
                seen += 1;
            }
        }
        assert_eq!(seen, 6);
    }
}
