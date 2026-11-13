use super::*;
use crate::jvm_reader::tests::open_fixture;
use lucene_engine::analysis::test_support::spec;

/// `fixtures/corpus/matchhighlight-docs.tsv`'s first document, the value of
/// its `offs`, `tv` and `both` fields in `fixtures/data/uhighlight/index`.
const D0: &str = "The quick brown fox jumps over the lazy dog. The fox is quick and the dog \
is lazy, so the fox jumps again and again over the same lazy dog until the evening comes.";

fn put_bytes(out: &mut Vec<u8>, b: &[u8]) {
    put_i32(out, b.len() as i32);
    out.extend_from_slice(b);
}

fn standard() -> Vec<u8> {
    spec(&[], ("standard", &[]), &[("lowercase", &[])], &[])
}

fn term_query(field: &str, term: &str) -> Vec<u8> {
    let mut q = vec![lucene_engine::query_blob::QUERY_TERM];
    put_bytes(&mut q, field.as_bytes());
    put_bytes(&mut q, term.as_bytes());
    q
}

fn tree(node: impl FnOnce(&mut Vec<u8>)) -> Vec<u8> {
    let mut q = vec![lucene_engine::query_blob::QUERY_TREE];
    node(&mut q);
    q
}

/// A request's fields, OpenSearch's defaults where they have one.
struct Req {
    field: &'static str,
    query: Vec<u8>,
    source: u8,
    analyzers: Vec<(&'static str, Vec<u8>)>,
    offset_limit: i32,
    breaks: u8,
    max_len: i32,
    max_passages: i32,
    no_match_size: i32,
    html: bool,
    require_field_match: bool,
    matched: Vec<&'static str>,
    max_analyzed_offset: i32,
}

impl Req {
    fn new(field: &'static str, query: Vec<u8>, source: u8) -> Self {
        Req {
            field,
            query,
            source,
            analyzers: vec![(field, standard())],
            offset_limit: -1,
            breaks: 2,
            max_len: 100,
            max_passages: 5,
            no_match_size: 0,
            html: false,
            require_field_match: true,
            matched: Vec::new(),
            max_analyzed_offset: 1_000_000,
        }
    }

    fn bytes(&self) -> Vec<u8> {
        let mut out = vec![REQUEST_VERSION];
        put_bytes(&mut out, self.field.as_bytes());
        put_bytes(&mut out, &self.query);
        out.push(self.source);
        put_i32(&mut out, self.analyzers.len() as i32);
        for (f, s) in &self.analyzers {
            put_bytes(&mut out, f.as_bytes());
            put_bytes(&mut out, s);
        }
        put_i32(&mut out, self.offset_limit);
        out.push(self.breaks);
        put_i32(&mut out, self.max_len);
        put_i32(&mut out, self.max_passages);
        put_i32(&mut out, self.no_match_size);
        put_bytes(&mut out, b"<em>");
        put_bytes(&mut out, b"</em>");
        out.push(u8::from(self.html));
        out.push(u8::from(self.require_field_match));
        put_i32(&mut out, self.matched.len() as i32);
        for m in &self.matched {
            put_bytes(&mut out, m.as_bytes());
        }
        put_i32(&mut out, self.max_analyzed_offset);
        out
    }
}

fn utf16(s: &str) -> Vec<u16> {
    s.encode_utf16().collect()
}

/// The status and the snippets with text of a result.
fn parse(out: &[u8]) -> (u8, Vec<(String, bool)>) {
    if out[0] != STATUS_SNIPPETS {
        assert_eq!(out.len(), 1);
        return (out[0], Vec::new());
    }
    let mut c = Cursor::new(&out[1..]);
    let n = c.length().unwrap();
    let mut snippets = Vec::new();
    for _ in 0..n {
        let score = f32::from_bits(c.i32().unwrap() as u32);
        let highlighted = c.u8().unwrap() != 0;
        // a no-match summary's passage is scored NaN, as Lucene's
        assert_eq!(score.is_nan(), !highlighted);
        let len = c.length().unwrap();
        let mut units = Vec::with_capacity(len);
        for _ in 0..len {
            let lo = c.u8().unwrap();
            let hi = c.u8().unwrap();
            units.push(u16::from_le_bytes([lo, hi]));
        }
        let text = String::from_utf16(&units).unwrap();
        if !text.trim().is_empty() {
            snippets.push((text, highlighted));
        }
    }
    assert_eq!(c.remaining(), 0);
    (STATUS_SNIPPETS, snippets)
}

fn run(handle: u64, req: &Req, doc: i32, content: &str) -> (u8, Vec<(String, bool)>) {
    parse(&highlight(handle, &req.bytes(), 0, doc, &utf16(content)).unwrap())
}

#[test]
fn every_offset_source_highlights_alike() {
    let h = open_fixture("uhighlight/index");
    let q = |f| term_query(f, "fox");
    let analysis = run(h, &Req::new("offs", q("offs"), 0), 0, D0);
    assert_eq!(analysis.0, STATUS_SNIPPETS);
    assert!(!analysis.1.is_empty());
    assert!(analysis
        .1
        .iter()
        .all(|(s, hl)| *hl && s.contains("<em>fox</em>")));
    assert_eq!(run(h, &Req::new("offs", q("offs"), 1), 0, D0), analysis);
    assert_eq!(run(h, &Req::new("both", q("both"), 2), 0, D0), analysis);
    assert_eq!(run(h, &Req::new("tv", q("tv"), 3), 0, D0), analysis);
    // the same request again: its cached highlighter
    assert_eq!(run(h, &Req::new("tv", q("tv"), 3), 0, D0), analysis);
}

#[test]
fn the_ffi_entry_point_returns_the_result_bytes() {
    let h = open_fixture("uhighlight/index");
    let req = Req::new("offs", term_query("offs", "fox"), 1).bytes();
    let content = utf16(D0);
    let mut ptr = std::ptr::null_mut();
    let mut len = 0usize;
    // SAFETY: valid buffers for the call.
    let rc = unsafe {
        ffi_jvm_highlight_unified_alloc(
            h,
            req.as_ptr(),
            req.len(),
            0,
            0,
            content.as_ptr(),
            content.len(),
            &mut ptr,
            &mut len,
        )
    };
    assert_eq!(rc, 0, "{}", crate::error::last_error());
    // SAFETY: the buffer the call gave.
    let out = unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec();
    // SAFETY: freed once, with its length.
    unsafe { crate::ffm_bridge::ffi_jvm_free_bytes(ptr, len) };
    assert_eq!(
        parse(&out),
        run(h, &Req::new("offs", term_query("offs", "fox"), 1), 0, D0)
    );
    // a null content with a length is refused
    // SAFETY: the null pointer is what is tested.
    let rc = unsafe {
        ffi_jvm_highlight_unified_alloc(
            h,
            req.as_ptr(),
            req.len(),
            0,
            0,
            std::ptr::null(),
            3,
            &mut ptr,
            &mut len,
        )
    };
    assert_ne!(rc, 0);
}

#[test]
fn a_query_on_another_field_is_a_noop_unless_a_summary_is_asked() {
    let h = open_fixture("uhighlight/index");
    let req = Req::new("offs", term_query("pos", "fox"), 1);
    assert_eq!(run(h, &req, 0, D0).0, STATUS_NOOP);
    let mut req = Req::new("offs", term_query("pos", "fox"), 1);
    req.no_match_size = 20;
    let (status, snippets) = run(h, &req, 0, D0);
    assert_eq!(status, STATUS_SNIPPETS);
    assert_eq!(snippets, [("The quick brown fox jumps".to_string(), false)]);
    // with require_field_match off, the other field's query counts
    let mut req = Req::new("offs", term_query("pos", "fox"), 1);
    req.require_field_match = false;
    let (_, snippets) = run(h, &req, 0, D0);
    assert!(snippets.iter().any(|(s, _)| s.contains("<em>fox</em>")));
}

#[test]
fn analyzed_offset_limits_answer_with_their_statuses() {
    let h = open_fixture("uhighlight/index");
    let mut req = Req::new("offs", term_query("offs", "fox"), 0);
    req.offset_limit = 50;
    req.max_analyzed_offset = 10;
    assert_eq!(run(h, &req, 0, D0).0, STATUS_OFFSET_LIMIT);
    let mut req = Req::new("offs", term_query("offs", "fox"), 0);
    req.max_analyzed_offset = 10;
    assert_eq!(run(h, &req, 0, D0).0, STATUS_TOO_LONG);
    // postings do not re-analyze, so the length does not matter
    let mut req = Req::new("offs", term_query("offs", "fox"), 1);
    req.max_analyzed_offset = 10;
    assert_eq!(run(h, &req, 0, D0).0, STATUS_SNIPPETS);
    // max_analyzer_offset: only the tokens that start before it
    let mut req = Req::new("offs", term_query("offs", "fox"), 0);
    req.offset_limit = 20;
    req.breaks = 0;
    req.max_passages = ALL;
    let (_, snippets) = run(h, &req, 0, D0);
    assert_eq!(snippets.len(), 1);
    assert_eq!(snippets[0].0.matches("<em>").count(), 1);
}

const ALL: i32 = i32::MAX - 1;

#[test]
fn re_analysis_without_its_analyzer_needs_the_jvm() {
    let h = open_fixture("uhighlight/index");
    let mut req = Req::new("offs", term_query("offs", "fox"), 0);
    req.analyzers.clear();
    assert_eq!(run(h, &req, 0, D0).0, STATUS_NEEDS_ANALYZER);
    // a multi-term query on postings re-analyzes
    let prefix = || {
        tree(|q| {
            q.push(9);
            put_bytes(q, b"offs");
            put_bytes(q, b"fo");
        })
    };
    let mut req = Req::new("offs", prefix(), 1);
    req.analyzers.clear();
    assert_eq!(run(h, &req, 0, D0).0, STATUS_NEEDS_ANALYZER);
    let (status, snippets) = run(h, &Req::new("offs", prefix(), 1), 0, D0);
    assert_eq!(status, STATUS_SNIPPETS);
    assert!(snippets.iter().any(|(s, _)| s.contains("<em>fox</em>")));
    // a masked field is re-analyzed with its own analyzer
    let mut req = Req::new("offs", term_query("pos", "fox"), 0);
    req.matched = vec!["pos"];
    assert_eq!(run(h, &req, 0, D0).0, STATUS_NEEDS_ANALYZER);
    req.analyzers.push(("pos", standard()));
    let (status, snippets) = run(h, &req, 0, D0);
    assert_eq!(status, STATUS_SNIPPETS);
    assert!(snippets.iter().any(|(s, _)| s.contains("<em>fox</em>")));
}

#[test]
fn phrases_with_their_own_positions_and_multi_phrases_decode() {
    let h = open_fixture("uhighlight/index");
    let phrase = tree(|q| {
        q.push(lucene_engine::queries::NODE_PHRASE_POSITIONS);
        put_bytes(q, b"offs");
        put_i32(q, 0);
        put_i32(q, 2);
        put_i32(q, 0);
        put_bytes(q, b"lazy");
        put_i32(q, 1);
        put_bytes(q, b"dog");
    });
    let (_, snippets) = run(h, &Req::new("offs", phrase, 1), 0, D0);
    assert!(snippets[0].0.contains("<em>lazy</em> <em>dog</em>"));
    let multi = tree(|q| {
        q.push(lucene_engine::queries::NODE_MULTI_PHRASE);
        put_bytes(q, b"offs");
        put_i32(q, 0);
        put_i32(q, 2);
        put_i32(q, 0);
        put_i32(q, 2);
        put_bytes(q, b"lazy");
        put_bytes(q, b"brown");
        put_i32(q, 1);
        put_i32(q, 1);
        put_bytes(q, b"dog");
    });
    let (_, snippets) = run(h, &Req::new("offs", multi, 1), 0, D0);
    assert!(snippets[0].0.contains("<em>lazy</em> <em>dog</em>"));
    assert!(!snippets[0].0.contains("<em>brown</em>"));
    let prefix = tree(|q| {
        q.push(lucene_engine::queries::NODE_MULTI_PHRASE_PREFIX);
        put_bytes(q, b"offs");
        put_i32(q, 0);
        put_i32(q, 50);
        put_i32(q, 2);
        put_i32(q, 0);
        put_i32(q, 1);
        put_bytes(q, b"lazy");
        put_i32(q, 1);
        put_i32(q, 1);
        put_bytes(q, b"do");
    });
    let (_, snippets) = run(h, &Req::new("offs", prefix, 1), 0, D0);
    assert!(snippets[0].0.contains("<em>lazy</em> <em>dog</em>"));
    // a phrase prefix of no positions
    let empty = tree(|q| {
        q.push(lucene_engine::queries::NODE_MULTI_PHRASE_PREFIX);
        put_bytes(q, b"offs");
        put_i32(q, 0);
        put_i32(q, 50);
        put_i32(q, 0);
    });
    assert!(highlight(h, &Req::new("offs", empty, 1).bytes(), 0, 0, &utf16(D0)).is_err());
    for bad in [
        // no positions
        tree(|q| {
            q.push(lucene_engine::queries::NODE_MULTI_PHRASE);
            put_bytes(q, b"offs");
            put_i32(q, 0);
            put_i32(q, 0);
        }),
        // a negative slop
        tree(|q| {
            q.push(lucene_engine::queries::NODE_MULTI_PHRASE);
            put_bytes(q, b"offs");
            put_i32(q, -1);
            put_i32(q, 0);
        }),
        // positions out of order
        tree(|q| {
            q.push(lucene_engine::queries::NODE_MULTI_PHRASE);
            put_bytes(q, b"offs");
            put_i32(q, 0);
            put_i32(q, 2);
            put_i32(q, 1);
            put_i32(q, 1);
            put_bytes(q, b"dog");
            put_i32(q, 0);
            put_i32(q, 1);
            put_bytes(q, b"lazy");
        }),
        // trailing bytes after the tree
        tree(|q| {
            q.push(9);
            put_bytes(q, b"offs");
            put_bytes(q, b"fo");
            q.push(0);
        }),
    ] {
        assert!(highlight(h, &Req::new("offs", bad, 1).bytes(), 0, 0, &utf16(D0)).is_err());
    }
}

#[test]
fn every_option_decodes() {
    let h = open_fixture("uhighlight/index");
    for (breaks, max_len) in [(0, 0), (1, 0), (2, 30), (3, 0)] {
        let mut req = Req::new("offs", term_query("offs", "fox"), 1);
        req.breaks = breaks;
        req.max_len = max_len;
        req.html = true;
        let (status, snippets) = run(h, &req, 0, D0);
        assert_eq!(status, STATUS_SNIPPETS);
        assert!(!snippets.is_empty());
    }
}

#[test]
fn malformed_requests_and_hits_are_refused() {
    let h = open_fixture("uhighlight/index");
    let good = Req::new("offs", term_query("offs", "fox"), 1);
    let content = utf16(D0);
    let refuse = |bytes: Vec<u8>| {
        assert!(highlight(h, &bytes, 0, 0, &content).is_err());
    };
    let mut b = good.bytes();
    b[0] = 9;
    refuse(b);
    let mut b = good.bytes();
    b.push(0);
    refuse(b);
    let mut r = Req::new("offs", term_query("offs", "fox"), 7);
    refuse(r.bytes());
    r.source = 1;
    r.breaks = 9;
    refuse(r.bytes());
    r.breaks = 2;
    r.offset_limit = -5;
    refuse(r.bytes());
    r.offset_limit = -1;
    r.analyzers = vec![("offs", vec![0xff])];
    refuse(r.bytes());
    refuse(good.bytes()[..10].to_vec());
    let bytes = good.bytes();
    assert!(highlight(h, &bytes, 5, 0, &content).is_err());
    assert!(highlight(h, &bytes, -1, 0, &content).is_err());
    assert!(highlight(h, &bytes, 0, 1 << 20, &content).is_err());
    assert!(highlight(h, &bytes, 0, -1, &content).is_err());
    assert!(highlight(u64::MAX, &bytes, 0, 0, &content).is_err());
}

#[test]
fn the_per_thread_cache_keeps_the_latest_requests() {
    let h = open_fixture("uhighlight/index");
    let expected = run(h, &Req::new("offs", term_query("offs", "fox"), 1), 0, D0);
    for n in 0..(CACHED_PER_THREAD as i32 + 2) {
        let mut req = Req::new("offs", term_query("offs", "fox"), 1);
        req.no_match_size = n;
        assert_eq!(run(h, &req, 0, D0).1, expected.1);
    }
    CACHED.with(|c| assert_eq!(c.borrow().len(), CACHED_PER_THREAD));
    // an older entry moves to the end when used again
    let mut req = Req::new("offs", term_query("offs", "fox"), 1);
    req.no_match_size = 5;
    run(h, &req, 0, D0);
    let key = req.bytes();
    CACHED.with(|c| assert_eq!(&*c.borrow().last().unwrap().0, &key[..]));
}

#[test]
fn a_visit_only_node_highlights_its_phrase_as_terms() {
    let h = open_fixture("uhighlight/index");
    let wrapped = tree(|q| {
        q.push(lucene_engine::queries::NODE_VISIT_ONLY);
        q.push(lucene_engine::queries::NODE_PHRASE_POSITIONS);
        put_bytes(q, b"offs");
        put_i32(q, 0);
        put_i32(q, 2);
        put_i32(q, 0);
        put_bytes(q, b"lazy");
        put_i32(q, 1);
        put_bytes(q, b"dog");
    });
    let (_, snippets) = run(h, &Req::new("offs", wrapped, 1), 0, D0);
    assert!(
        snippets
            .iter()
            .any(|(s, _)| s.contains("<em>dog</em> is <em>lazy</em>")),
        "{snippets:?}"
    );
}

/// A visited record's terms, behind a guard the field passes, inside a
/// `MUST` sub-visitor.
fn visited(guard: Option<&[u8]>, kind_override: Option<u8>) -> Vec<u8> {
    tree(|q| {
        q.push(lucene_engine::queries::NODE_VISITED);
        put_i32(q, 1);
        q.push(2); // sub
        q.push(0); // MUST
        put_i32(q, 2);
        q.push(kind_override.unwrap_or(0)); // terms
        match guard {
            Some(g) => {
                q.push(1);
                put_bytes(q, g);
            }
            None => q.push(0),
        }
        put_i32(q, 2);
        put_bytes(q, b"offs");
        put_bytes(q, b"lazy");
        put_bytes(q, b"offs");
        put_bytes(q, b"dog");
        q.push(3); // a node handed over whole
        q.push(lucene_engine::queries::NODE_PHRASE_POSITIONS);
        put_bytes(q, b"offs");
        put_i32(q, 0);
        put_i32(q, 1);
        put_i32(q, 0);
        put_bytes(q, b"fox");
    })
}

#[test]
fn a_visited_record_highlights_what_its_visit_consumed() {
    let h = open_fixture("uhighlight/index");
    let (status, snippets) = run(h, &Req::new("offs", visited(Some(b"offs"), None), 1), 0, D0);
    assert_eq!(status, STATUS_SNIPPETS);
    let all: String = snippets.iter().map(|(s, _)| s.as_str()).collect();
    for word in ["dog", "lazy", "fox"] {
        assert!(all.contains(&format!("<em>{word}</em>")), "{snippets:?}");
    }
    // the term collector accepts every field: a guard on any field, or none,
    // consumes the same terms
    for guard in [None, Some(&b"tv"[..])] {
        assert_eq!(
            run(h, &Req::new("offs", visited(guard, None), 1), 0, D0),
            (status, snippets.clone())
        );
    }
}

#[test]
fn a_visited_record_with_an_unknown_event_is_refused() {
    let h = open_fixture("uhighlight/index");
    let req = Req::new("offs", visited(None, Some(9)), 1).bytes();
    assert!(highlight(h, &req, 0, 0, &utf16(D0)).is_err());
}
