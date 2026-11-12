use super::*;
use crate::jvm_analysis::tests::spec;
use crate::jvm_reader;

const TEXT: &str = "The quick brown fox jumps over the lazy dog. A second sentence mentions \
the fox again & it ends here! Third <b>sentence</b> without the animal.";

fn put_bytes(out: &mut Vec<u8>, b: &[u8]) {
    put_i32(out, b.len() as i32);
    out.extend_from_slice(b);
}

fn term_query(field: &str, term: &str) -> Vec<u8> {
    let mut q = vec![jvm_reader::QUERY_TERM];
    put_bytes(&mut q, field.as_bytes());
    put_bytes(&mut q, term.as_bytes());
    q
}

fn request(query: &[u8], fragmenter: u8, size: i32, nfrags: i32) -> Vec<u8> {
    let mut out = vec![REQUEST_VERSION];
    put_bytes(&mut out, b"body");
    put_bytes(&mut out, query);
    put_bytes(
        &mut out,
        &spec(&[], ("standard", &[]), &[("lowercase", &[])], &[]),
    );
    out.push(1);
    out.push(fragmenter);
    put_i32(&mut out, size);
    put_i32(&mut out, nfrags);
    put_bytes(&mut out, b"<em>");
    put_bytes(&mut out, b"</em>");
    out.push(0);
    out
}

fn utf16(s: &str) -> Vec<u16> {
    s.encode_utf16().collect()
}

/// The fragments' texts and the summary's end.
fn decode_result(b: &[u8]) -> (Vec<(f32, String)>, i32) {
    let mut c = Cursor::new(b);
    assert_eq!(c.u8().unwrap(), STATUS_FRAGMENTS);
    let n = c.i32().unwrap();
    let mut out = Vec::new();
    for _ in 0..n {
        let score = f32::from_bits(c.i32().unwrap() as u32);
        let len = c.i32().unwrap() as usize;
        let mut units = Vec::new();
        for _ in 0..len {
            let lo = c.u8().unwrap() as u16;
            let hi = c.u8().unwrap() as u16;
            units.push(lo | hi << 8);
        }
        out.push((score, String::from_utf16(&units).unwrap()));
    }
    let end = c.i32().unwrap();
    let wait = c.i32().unwrap();
    assert!(wait >= -1);
    assert_eq!(c.remaining(), 0);
    (out, end)
}

/// Through the C entry point: `values` and an optional summary.
fn call(
    request: &[u8],
    values: &[&str],
    no_match_size: i32,
    summary: Option<&str>,
) -> Result<(Vec<(f32, String)>, i32), i32> {
    let units: Vec<Vec<u16>> = values.iter().map(|v| utf16(v)).collect();
    let content: Vec<u16> = units.concat();
    let lengths: Vec<i32> = units.iter().map(|u| u.len() as i32).collect();
    let summary_units = summary.map(utf16);
    let (sp, sl) = match &summary_units {
        Some(s) => (s.as_ptr(), s.len() as i64),
        None => (std::ptr::null(), -1),
    };
    let mut ptr = std::ptr::null_mut();
    let mut len = 0usize;
    // SAFETY: every buffer is live and sized as passed.
    let rc = unsafe {
        ffi_jvm_highlight_plain_alloc(
            request.as_ptr(),
            request.len(),
            if content.is_empty() {
                std::ptr::null()
            } else {
                content.as_ptr()
            },
            content.len(),
            if lengths.is_empty() {
                std::ptr::null()
            } else {
                lengths.as_ptr()
            },
            lengths.len(),
            no_match_size,
            sp,
            sl,
            -1,
            &mut ptr,
            &mut len,
        )
    };
    if rc != 0 {
        return Err(rc);
    }
    // SAFETY: the library handed back `len` bytes at `ptr`.
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec();
    // SAFETY: owned by the library, freed once.
    unsafe { crate::ffm_bridge::ffi_jvm_free_bytes(ptr, len) };
    Ok(decode_result(&bytes))
}

#[test]
fn each_value_gives_its_fragments_in_order() {
    let req = request(&term_query("body", "fox"), 1, 20, 3);
    let (frags, end) = call(&req, &[TEXT, "no match here", "a fox"], 30, Some(TEXT)).unwrap();
    let texts: Vec<&str> = frags.iter().map(|(_, t)| t.as_str()).collect();
    assert_eq!(
        texts,
        [
            "The quick brown <em>fox</em>",
            " sentence mentions the <em>fox</em>",
            "a <em>fox</em>"
        ]
    );
    assert!(frags.iter().all(|(s, _)| *s > 0.0));
    // there are fragments: no summary
    assert_eq!(end, -1);
    // the cache serves the same request again
    assert_eq!(call(&req, &["a fox"], 0, None).unwrap().0.len(), 1);
}

#[test]
fn no_fragment_cuts_the_summary() {
    let req = request(&term_query("title", "fox"), 0, 100, 0);
    let (frags, end) = call(&req, &[TEXT], 30, Some(TEXT)).unwrap();
    assert!(frags.is_empty());
    assert_eq!(&TEXT[..end as usize], "The quick brown fox jumps over");
    assert_eq!(call(&req, &[TEXT], 0, Some(TEXT)).unwrap().1, -1);
    assert_eq!(call(&req, &[TEXT], 30, None).unwrap().1, -1);
    assert_eq!(call(&req, &[], 30, None).unwrap(), (vec![], -1));
    // the simple fragmenter
    let req = request(&term_query("body", "fox"), 2, 20, 3);
    assert_eq!(call(&req, &[TEXT], 0, None).unwrap().0.len(), 2);
}

#[test]
fn bad_requests_and_buffers_are_refused() {
    let good = request(&term_query("body", "fox"), 1, 100, 5);
    let mut bad_version = good.clone();
    bad_version[0] = 9;
    let mut bad_fragmenter = good.clone();
    let at = bad_fragmenter.len() - (4 + 4 + 4 + 4 + 4 + 5 + 1) - 1;
    bad_fragmenter[at] = 7;
    let mut trailing = good.clone();
    trailing.push(0);
    for bad in [bad_version, bad_fragmenter, trailing, vec![]] {
        assert!(call(&bad, &["a fox"], 0, None).is_err());
    }
    // lengths that overrun or do not cover the content
    let content = utf16("a fox");
    for lengths in [vec![9], vec![2], vec![-1]] {
        let mut ptr = std::ptr::null_mut();
        let mut len = 0usize;
        // SAFETY: the buffers are live; the lengths are what is tested.
        let rc = unsafe {
            ffi_jvm_highlight_plain_alloc(
                good.as_ptr(),
                good.len(),
                content.as_ptr(),
                content.len(),
                lengths.as_ptr(),
                1,
                0,
                std::ptr::null(),
                -1,
                -1,
                &mut ptr,
                &mut len,
            )
        };
        assert_ne!(rc, 0);
    }
    // null buffers with lengths
    let mut ptr = std::ptr::null_mut();
    let mut len = 0usize;
    // SAFETY: null pointers with non-zero lengths, refused before any read.
    let rc = unsafe {
        ffi_jvm_highlight_plain_alloc(
            good.as_ptr(),
            good.len(),
            std::ptr::null(),
            3,
            std::ptr::null(),
            1,
            0,
            std::ptr::null(),
            2,
            -1,
            &mut ptr,
            &mut len,
        )
    };
    assert_ne!(rc, 0);
    // SAFETY: as above, the lengths pointer null with a count.
    let rc = unsafe {
        ffi_jvm_highlight_plain_alloc(
            good.as_ptr(),
            good.len(),
            content.as_ptr(),
            content.len(),
            std::ptr::null(),
            1,
            0,
            std::ptr::null(),
            -1,
            -1,
            &mut ptr,
            &mut len,
        )
    };
    assert_ne!(rc, 0);
}

fn fuzzy_query(term: &str, max_edits: u8) -> Vec<u8> {
    let mut q = vec![jvm_reader::QUERY_TREE, crate::jvm_queries::NODE_FUZZY];
    put_bytes(&mut q, b"body");
    put_bytes(&mut q, term.as_bytes());
    q.push(max_edits);
    put_i32(&mut q, 0);
    q.push(1);
    q
}

#[test]
fn a_fuzzy_weights_each_term_by_its_similarity() {
    // "fox" exactly scores 1, "fix" one edit off scores 1 - 1/3
    let req = request(&fuzzy_query("fox", 1), 2, 8, 5);
    let (frags, _) = call(&req, &["a fox and a fix"], 0, None).unwrap();
    let got: Vec<(f32, &str)> = frags.iter().map(|(s, t)| (*s, t.as_str())).collect();
    assert_eq!(got.len(), 2);
    assert_eq!(got[0].1, "a <em>fox</em>");
    assert_eq!(got[1].1, " and a <em>fix</em>");
    assert!((got[0].0 - 1.0).abs() < 1e-6, "{got:?}");
    assert!((got[1].0 - 2.0 / 3.0).abs() < 1e-6, "{got:?}");
    // more edits than a fuzzy takes
    assert!(call(
        &request(&fuzzy_query("fox", 3), 2, 8, 5),
        &["a fox"],
        0,
        None
    )
    .is_err());
}

#[test]
fn the_span_state_is_carried_between_calls() {
    let req = request(&term_query("body", "fox"), 1, 10, 5);
    let out = highlight(&req, &[&utf16("a fox")], 0, None, 7).unwrap();
    // the state goes on as the fragmenter leaves it: past "a fox"'s 2 tokens, still 7
    assert_eq!(
        i32::from_le_bytes(out[out.len() - 4..].try_into().unwrap()),
        7
    );
    let out = highlight(&req, &[&utf16("a fox")], 0, None, -1).unwrap();
    assert_eq!(
        i32::from_le_bytes(out[out.len() - 4..].try_into().unwrap()),
        -1
    );
}
