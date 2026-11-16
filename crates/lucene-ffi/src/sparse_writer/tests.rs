#![cfg(test)]
use super::*;

fn attrs(pairs: &[(&str, &str)]) -> Vec<u8> {
    let mut out = (pairs.len() as u32).to_le_bytes().to_vec();
    for (k, v) in pairs {
        for s in [k, v] {
            out.extend_from_slice(&(s.len() as u32).to_le_bytes());
            out.extend_from_slice(s.as_bytes());
        }
    }
    out
}

fn forward(pairs: &[(i32, f32)]) -> Vec<u8> {
    let mut out = Vec::new();
    for (t, w) in pairs {
        out.extend_from_slice(&t.to_be_bytes());
        out.extend_from_slice(&w.to_be_bytes());
    }
    out
}

unsafe fn open(name: &str, number: i32, a: &[u8], max_doc: i32) -> (i32, u64) {
    let mut h = 0u64;
    // SAFETY: valid buffers.
    let rc = unsafe {
        ffi_sparse_clusterer_open(
            name.as_ptr(),
            name.len(),
            number,
            a.as_ptr(),
            a.len(),
            max_doc,
            &mut h,
        )
    };
    (rc, h)
}

#[test]
fn attributes_decode_and_refuse_bad_layouts() {
    let a = attrs(&[("k", "v"), ("x", "")]);
    assert_eq!(
        decode_attributes(&a).unwrap(),
        vec![("k".into(), "v".into()), ("x".into(), String::new())]
    );
    assert!(decode_attributes(&a[..a.len() - 1]).is_err());
    let mut long = a.clone();
    long.push(0);
    assert!(decode_attributes(&long).is_err());
    assert!(decode_attributes(&[1, 0, 0, 0, 1, 0, 0, 0, 0xFF, 0, 0, 0, 0]).is_err());
}

#[test]
fn segment_files_match_the_rust_writer_s() {
    let field_attrs = [
        ("sparse_vector_field", "true"),
        ("approximate_threshold", "4"),
        ("cluster_ratio", "0.5"),
    ];
    let docs: Vec<Vec<(i32, f32)>> = (0..30)
        .map(|d| vec![(d % 3, 0.5 + d as f32 / 30.0), (7, 1.25), (100 + d, 0.1)])
        .collect();
    let id = [9u8; ID_LENGTH];

    // The same field through the Rust writer's clusterer.
    let info = FieldInfo::new("emb", 2).with_attributes(
        field_attrs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect(),
    );
    let mut direct = FieldClusterer::new(&info, 31).unwrap();
    for (d, v) in docs.iter().enumerate() {
        if d != 5 {
            direct.add(d as i32, &forward(v)).unwrap();
        }
    }
    let (want_sit, want_sip) = lucene_sparse::postings::write(&id, &[direct.finish(&id)]);

    let a = attrs(&field_attrs);
    // SAFETY: valid buffers.
    let (rc, h) = unsafe { open("emb", 2, &a, 31) };
    assert_eq!(rc, 0);
    for (d, v) in docs.iter().enumerate() {
        if d == 5 {
            continue;
        }
        let bytes = forward(v);
        // SAFETY: valid buffers.
        let rc = unsafe { ffi_sparse_clusterer_add(h, d as i32, bytes.as_ptr(), bytes.len()) };
        assert_eq!(rc, 0);
    }
    // Out of order is refused.
    // SAFETY: valid buffers.
    let rc = unsafe { ffi_sparse_clusterer_add(h, 3, std::ptr::null(), 0) };
    assert_eq!(rc, FfiStatus::InvalidArgument.code());

    let mut ptr = std::ptr::null_mut();
    let mut len = 0usize;
    // SAFETY: valid buffers.
    let rc = unsafe { ffi_sparse_postings_write_alloc(&h, 1, id.as_ptr(), &mut ptr, &mut len) };
    assert_eq!(rc, 0);
    // SAFETY: the call above left `len` bytes at `ptr`.
    let out = unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec();
    // SAFETY: the buffer came from the call above.
    unsafe { crate::ffm_bridge::ffi_jvm_free_bytes(ptr, len) };
    let sit_len = u64::from_le_bytes(out[..8].try_into().unwrap()) as usize;
    assert_eq!(&out[8..8 + sit_len], want_sit.as_slice());
    assert_eq!(&out[8 + sit_len..], want_sip.as_slice());
    // The write took the handle.
    assert_eq!(
        ffi_sparse_clusterer_close(h),
        FfiStatus::InvalidHandle.code()
    );
}

#[test]
fn no_fields_still_write_a_pair_and_bad_handles_fail() {
    let id = [1u8; ID_LENGTH];
    let mut ptr = std::ptr::null_mut();
    let mut len = 0usize;
    // SAFETY: valid buffers.
    let rc = unsafe {
        ffi_sparse_postings_write_alloc(std::ptr::null(), 0, id.as_ptr(), &mut ptr, &mut len)
    };
    assert_eq!(rc, 0);
    assert!(len > 8);
    // SAFETY: the buffer came from the call above.
    unsafe { crate::ffm_bridge::ffi_jvm_free_bytes(ptr, len) };

    let bogus = 12345u64;
    // SAFETY: valid buffers.
    let rc = unsafe { ffi_sparse_postings_write_alloc(&bogus, 1, id.as_ptr(), &mut ptr, &mut len) };
    assert_eq!(rc, FfiStatus::InvalidHandle.code());
    // SAFETY: valid buffers.
    let rc = unsafe { ffi_sparse_clusterer_add(bogus, 0, std::ptr::null(), 0) };
    assert_eq!(rc, FfiStatus::InvalidHandle.code());

    let bad = attrs(&[("sparse_vector_field", "true"), ("n_postings", "x")]);
    // SAFETY: valid buffers.
    let (rc, _) = unsafe { open("emb", 0, &bad, 10) };
    assert_eq!(rc, FfiStatus::InvalidArgument.code());
    let ok = attrs(&[("sparse_vector_field", "true")]);
    // SAFETY: valid buffers.
    let (rc, h) = unsafe { open("emb", 0, &ok, 10) };
    assert_eq!(rc, 0);
    assert_eq!(ffi_sparse_clusterer_close(h), 0);
}
