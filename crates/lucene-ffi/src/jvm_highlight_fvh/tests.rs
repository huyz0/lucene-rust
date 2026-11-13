use super::*;
use crate::jvm_highlight::decode_clause;
use crate::jvm_reader::tests::open_fixture;
use lucene_highlighter::vectorhighlight::FastVectorHighlighter;

/// `fixtures/data/uhighlight/index`'s first document's `tv` and `both`
/// values (term vectors with positions and offsets).
const D0: &str = "The quick brown fox jumps over the lazy dog. The fox is quick and the dog \
is lazy, so the fox jumps again and again over the same lazy dog until the evening comes.";

fn put_bytes(out: &mut Vec<u8>, b: &[u8]) {
    put_i32(out, b.len() as i32);
    out.extend_from_slice(b);
}

fn utf16(s: &str) -> Vec<u16> {
    s.encode_utf16().collect()
}

/// A trie node; `subs` already encoded.
fn node(out: &mut Vec<u8>, terminal: bool, slop: i32, number: i32, subs: &[(&str, Vec<u8>)]) {
    out.push(u8::from(terminal));
    put_i32(out, slop);
    put_i32(out, 1.0f32.to_bits() as i32);
    put_i32(out, number);
    put_i32(out, subs.len() as i32);
    for (term, sub) in subs {
        put_bytes(out, term.as_bytes());
        out.extend_from_slice(sub);
    }
}

fn leaf(number: i32) -> Vec<u8> {
    let mut out = Vec::new();
    node(&mut out, true, 0, number, &[]);
    out
}

/// The options a request is built with.
struct Req {
    field: &'static str,
    matched: Vec<&'static str>,
    terms: Vec<&'static str>,
    /// `lazy dog`, the phrase, besides the terms.
    phrase: bool,
    margin: Option<i32>,
    /// Overrides the frag list kind byte.
    frag_kind: Option<u8>,
    frag_char_size: i32,
    max_fragments: i32,
    score_order: bool,
    scanner: u8,
    no_match_size: i32,
}

impl Req {
    fn new(field: &'static str, terms: &[&'static str]) -> Self {
        Self {
            field,
            matched: Vec::new(),
            terms: terms.to_vec(),
            phrase: false,
            margin: Some(-1),
            frag_kind: None,
            frag_char_size: 40,
            max_fragments: 3,
            score_order: false,
            scanner: 0,
            no_match_size: 0,
        }
    }

    fn bytes(&self) -> Vec<u8> {
        let mut out = vec![REQUEST_VERSION];
        put_bytes(&mut out, self.field.as_bytes());
        put_i32(&mut out, self.matched.len() as i32);
        for f in &self.matched {
            put_bytes(&mut out, f.as_bytes());
        }
        // field_match off: one root map and term set for every field
        out.push(0);
        put_i32(&mut out, 1);
        out.push(0);
        let mut subs: Vec<(&str, Vec<u8>)> = self
            .terms
            .iter()
            .enumerate()
            .map(|(i, t)| (*t, leaf(i as i32)))
            .collect();
        if self.phrase {
            let mut lazy = Vec::new();
            node(
                &mut lazy,
                false,
                0,
                0,
                &[("dog", leaf(self.terms.len() as i32))],
            );
            subs.push(("lazy", lazy));
        }
        node(&mut out, false, 0, 0, &subs);
        put_i32(&mut out, 1);
        out.push(0);
        let mut set = self.terms.clone();
        if self.phrase {
            set.extend(["lazy", "dog"]);
        }
        put_i32(&mut out, set.len() as i32);
        for t in set {
            put_bytes(&mut out, t.as_bytes());
        }
        match self.margin {
            None => {
                out.push(0);
                put_i32(&mut out, -1);
            }
            Some(m) => {
                out.push(self.frag_kind.unwrap_or(1));
                put_i32(&mut out, m);
            }
        }
        put_i32(&mut out, i32::MAX);
        put_i32(&mut out, self.frag_char_size);
        put_i32(&mut out, self.max_fragments);
        out.push(u8::from(self.score_order));
        out.push(0);
        out.push(1);
        out.push(self.scanner);
        put_i32(&mut out, 20);
        let chars = utf16(".,!? \t\n");
        put_i32(&mut out, chars.len() as i32);
        for c in chars {
            out.extend_from_slice(&c.to_le_bytes());
        }
        put_i32(&mut out, 1);
        put_bytes(&mut out, b"<em>");
        put_i32(&mut out, 1);
        put_bytes(&mut out, b"</em>");
        out.push(0);
        put_i32(&mut out, self.no_match_size);
        out
    }
}

fn parse(out: &[u8]) -> (u8, Vec<String>) {
    let mut c = Cursor::new(out);
    let status = c.u8().unwrap();
    if status != STATUS_FRAGMENTS {
        return (status, Vec::new());
    }
    let n = c.i32().unwrap();
    let mut frags = Vec::new();
    for _ in 0..n {
        let len = c.i32().unwrap() as usize;
        let mut units = Vec::with_capacity(len);
        for _ in 0..len {
            let lo = c.u8().unwrap();
            let hi = c.u8().unwrap();
            units.push(u16::from_le_bytes([lo, hi]));
        }
        frags.push(String::from_utf16(&units).unwrap());
    }
    assert_eq!(c.remaining(), 0);
    (status, frags)
}

fn run(h: u64, req: &Req, values: &[&str]) -> (u8, Vec<String>) {
    let values: Vec<Vec<u16>> = values.iter().map(|v| utf16(v)).collect();
    parse(&highlight(h, &req.bytes(), 0, 0, &values).unwrap())
}

fn term_query(field: &str, term: &str) -> Vec<u8> {
    let mut q = vec![lucene_engine::query_blob::QUERY_TERM];
    put_bytes(&mut q, field.as_bytes());
    put_bytes(&mut q, term.as_bytes());
    q
}

#[test]
fn a_term_gives_what_the_library_highlighter_gives() {
    let h = open_fixture("uhighlight/index");
    let (status, frags) = run(h, &Req::new("tv", &["fox"]), &[D0]);
    assert_eq!(status, STATUS_FRAGMENTS);
    assert!(!frags.is_empty());
    assert!(
        frags.iter().all(|f| f.contains("<em>fox</em>")),
        "{frags:?}"
    );
    // the same fragments as the reader-backed highlighter's
    let entry = jvm_reader::lookup(h, "test").unwrap();
    let clause = decode_clause(&term_query("tv", "fox")).unwrap();
    let fvh = FastVectorHighlighter::new(true, false);
    let fq = fvh.get_field_query(&clause, Some(&entry.reader)).unwrap();
    let mut builder = FragmentsBuilder::simple();
    let expected = fvh
        .get_best_fragments_with(
            &fq,
            &entry.reader,
            0,
            "tv",
            40,
            3,
            &SimpleFragListBuilder::new(),
            &mut builder,
            &["<em>".to_string()],
            &["</em>".to_string()],
            &DefaultEncoder,
        )
        .unwrap()
        .unwrap();
    assert_eq!(frags, expected);
    // a cached request answers alike
    assert_eq!(run(h, &Req::new("tv", &["fox"]), &[D0]), (status, frags));
}

#[test]
fn phrases_score_order_and_single_fragments() {
    let h = open_fixture("uhighlight/index");
    let mut req = Req::new("tv", &["fox"]);
    req.phrase = true;
    req.score_order = true;
    let (_, frags) = run(h, &req, &[D0]);
    assert!(
        frags.iter().any(|f| f.contains("<em>lazy dog</em>")),
        "{frags:?}"
    );
    req.margin = None;
    req.frag_char_size = i32::MAX;
    req.max_fragments = i32::MAX;
    let (_, whole) = run(h, &req, &[D0]);
    assert_eq!(whole.len(), 1);
    assert!(
        whole[0].starts_with("The quick brown <em>fox</em>"),
        "{whole:?}"
    );
    for scanner in [1, 2] {
        req.scanner = scanner;
        let (status, f) = run(h, &req, &[D0]);
        assert_eq!(status, STATUS_FRAGMENTS);
        assert_eq!(f.len(), 1);
    }
}

#[test]
fn matched_fields_merge_their_matches() {
    let h = open_fixture("uhighlight/index");
    let mut req = Req::new("tv", &["fox", "dog"]);
    req.matched = vec!["tv", "both"];
    let (status, frags) = run(h, &req, &[D0]);
    assert_eq!(status, STATUS_FRAGMENTS);
    assert!(frags.iter().any(|f| f.contains("<em>fox</em>")));
}

#[test]
fn no_match_gives_the_summary_and_none_without_values() {
    let h = open_fixture("uhighlight/index");
    let mut req = Req::new("tv", &["melon"]);
    assert_eq!(run(h, &req, &[D0]), (STATUS_FRAGMENTS, Vec::new()));
    req.no_match_size = 12;
    let (_, frags) = run(h, &req, &[D0]);
    assert_eq!(frags, vec!["The quick brown".to_string()]);
    assert_eq!(run(h, &req, &[]), (STATUS_FRAGMENTS, Vec::new()));
    // a field with no term vector in the segment matches nothing
    let other = Req::new("nothing_here", &["fox"]);
    assert_eq!(run(h, &other, &[D0]), (STATUS_FRAGMENTS, Vec::new()));
}

#[test]
fn what_java_throws_on_runs_the_stock_highlighter() {
    let h = open_fixture("uhighlight/index");
    // a negative margin: SimpleFragListBuilder refuses it
    let mut req = Req::new("tv", &["fox"]);
    req.margin = Some(-5);
    assert_eq!(run(h, &req, &[D0]).0, STATUS_STOCK);
    // offsets past the values: StringIndexOutOfBoundsException
    assert_eq!(
        run(h, &Req::new("tv", &["fox"]), &["short"]).0,
        STATUS_STOCK
    );
    // a fragment holding half a surrogate pair
    let mut text: Vec<u16> = utf16(D0);
    text[17] = 0xD83D;
    let values = vec![text];
    let out = highlight(h, &Req::new("tv", &["fox"]).bytes(), 0, 0, &values).unwrap();
    assert_eq!(out[0], STATUS_STOCK);
}

#[test]
fn the_ffi_entry_point_returns_the_result_bytes() {
    let h = open_fixture("uhighlight/index");
    let req = Req::new("tv", &["fox"]).bytes();
    let first = utf16(D0);
    let second = utf16("a second value with a fox");
    let content: Vec<u16> = first.iter().chain(&second).copied().collect();
    let lengths = [first.len() as i32, second.len() as i32];
    let mut ptr = std::ptr::null_mut();
    let mut len = 0usize;
    // SAFETY: valid buffers for the call.
    let rc = unsafe {
        ffi_jvm_highlight_fvh_alloc(
            h,
            req.as_ptr(),
            req.len(),
            0,
            0,
            content.as_ptr(),
            content.len(),
            lengths.as_ptr(),
            2,
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
        run(
            h,
            &Req::new("tv", &["fox"]),
            &[D0, "a second value with a fox"]
        )
    );
    // lengths that overrun the content are refused
    let bad = [first.len() as i32 + 100];
    // SAFETY: valid buffers; the lengths are what is tested.
    let rc = unsafe {
        ffi_jvm_highlight_fvh_alloc(
            h,
            req.as_ptr(),
            req.len(),
            0,
            0,
            content.as_ptr(),
            content.len(),
            bad.as_ptr(),
            1,
            &mut ptr,
            &mut len,
        )
    };
    assert_ne!(rc, 0);
}

#[test]
fn malformed_requests_and_hits_are_refused() {
    let h = open_fixture("uhighlight/index");
    let good = Req::new("tv", &["fox"]).bytes();
    let values = vec![utf16(D0)];
    let mut versioned = good.clone();
    versioned[0] = 9;
    assert!(highlight(h, &versioned, 0, 0, &values).is_err());
    let mut trailing = good.clone();
    trailing.push(0);
    assert!(highlight(h, &trailing, 0, 0, &values).is_err());
    assert!(highlight(h, &good[..good.len() - 3], 0, 0, &values).is_err());
    assert!(highlight(h, &good, 7, 0, &values).is_err());
    assert!(highlight(h, &good, 0, -1, &values).is_err());
    assert!(highlight(h, &good, 0, 1 << 20, &values).is_err());
    assert!(highlight(u64::MAX, &good, 0, 0, &values).is_err());
}

#[test]
fn bad_kinds_counts_and_deep_tries_are_refused() {
    let h = open_fixture("uhighlight/index");
    let values = vec![utf16(D0)];
    let mut req = Req::new("tv", &["fox"]);
    req.scanner = 9;
    assert!(highlight(h, &req.bytes(), 0, 0, &values).is_err());
    // a negative count of matched fields
    let mut neg = vec![REQUEST_VERSION];
    put_bytes(&mut neg, b"tv");
    put_i32(&mut neg, -1);
    assert!(highlight(h, &neg, 0, 0, &values).is_err());
    // a count past the request's end
    let mut over = vec![REQUEST_VERSION];
    put_bytes(&mut over, b"tv");
    put_i32(&mut over, 1 << 20);
    assert!(highlight(h, &over, 0, 0, &values).is_err());
    // a frag list kind it does not know
    let mut kind = Req::new("tv", &["fox"]);
    kind.frag_kind = Some(7);
    assert!(highlight(h, &kind.bytes(), 0, 0, &values).is_err());
    // a trie deeper than the bound
    let mut deep = vec![REQUEST_VERSION];
    put_bytes(&mut deep, b"tv");
    put_i32(&mut deep, 0);
    deep.push(0);
    put_i32(&mut deep, 1);
    deep.push(0);
    for _ in 0..=MAX_DEPTH + 1 {
        deep.push(0);
        put_i32(&mut deep, 0);
        put_i32(&mut deep, 0);
        put_i32(&mut deep, 0);
        put_i32(&mut deep, 1);
        put_bytes(&mut deep, b"t");
    }
    assert!(highlight(h, &deep, 0, 0, &values).is_err());
}
