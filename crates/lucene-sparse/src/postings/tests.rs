#![cfg(test)]
use super::*;
use lucene_store::DataOutput;

const ID: [u8; ID_LENGTH] = [7; ID_LENGTH];

fn header(out: &mut Vec<u8>) -> i64 {
    codec_util::write_index_header(out, CODEC_NAME, VERSION, &ID, "");
    out.len() as i64
}

fn finish(mut out: Vec<u8>, start: i64) -> Vec<u8> {
    out.write_i64(start);
    codec_util::write_footer(&mut out);
    out
}

/// `.sit` with one field (number 3) and the given (term, offset) pairs.
fn sit(terms: &[(&str, i64)]) -> Vec<u8> {
    let mut out = Vec::new();
    let start = header(&mut out);
    out.write_vint(1);
    out.write_vint(3);
    out.write_vlong(terms.len() as i64);
    for (t, o) in terms {
        out.write_vint(t.len() as i32);
        out.write_bytes(t.as_bytes());
        out.write_vlong(*o);
    }
    finish(out, start)
}

/// `.sip` whose body is `body`, and the offset the body starts at.
fn sip(body: impl FnOnce(&mut Vec<u8>)) -> (Arc<Input>, i64) {
    let mut out = Vec::new();
    let start = header(&mut out);
    body(&mut out);
    (Arc::new(Input::external(finish(out, start))), start)
}

#[test]
fn reads_clusters_with_and_without_summaries() {
    let (sip, start) = sip(|o| {
        o.write_vlong(2);
        // Cluster 1: docs 1 and 4, skippable, summary {2: 9}.
        o.write_vlong(2);
        o.write_vint(1);
        o.write_byte(10);
        o.write_vint(4);
        o.write_byte(200);
        o.write_byte(0);
        o.write_vlong(1);
        o.write_vint(2);
        o.write_byte(9);
        // Cluster 2: doc 6, not skippable, no summary.
        o.write_vlong(1);
        o.write_vint(6);
        o.write_byte(3);
        o.write_byte(1);
        o.write_vlong(0);
        // An empty term.
        o.write_vlong(0);
    });
    let empty_at = sip.len() as i64 - codec_util::FOOTER_LENGTH as i64 - 8 - 1;
    let p = ClusteredPostings::open(&sit(&[("12", start), ("13", empty_at)]), sip, &ID).unwrap();
    assert!(p.has_field(3));
    assert!(!p.has_field(4));
    let c = p.clusters(3, b"12").unwrap().unwrap();
    assert_eq!(c.len(), 2);
    assert_eq!(c[0].docs, vec![1, 4]);
    assert_eq!(c[0].weights, vec![10, 200]);
    assert!(!c[0].should_not_skip);
    assert_eq!(c[0].summary.as_ref().unwrap().tokens(), &[2]);
    assert!(c[1].should_not_skip);
    assert_eq!(c[1].summary, None);
    assert_eq!(p.clusters(3, b"13").unwrap(), None);
    assert_eq!(p.clusters(3, b"99").unwrap(), None);
    assert_eq!(p.clusters(4, b"12").unwrap(), None);

    // The cursor reads the same clusters in place.
    let mut cursor = p.cursor(3, b"12").unwrap().unwrap();
    let mut docs = vec![99];
    let first = cursor.next().unwrap().unwrap();
    assert!(!first.should_not_skip);
    assert_eq!(first.doc_count(), 2);
    first.docs_into(&mut docs).unwrap();
    assert_eq!(docs, vec![1, 4]);
    let mut dense = vec![0u8; 3];
    dense[2] = 5;
    assert_eq!(first.summary_dot(&dense).unwrap(), 45);
    // A query shorter than the summary's tokens stops the scan.
    assert_eq!(first.summary_dot(&[1, 1]).unwrap(), 0);
    let second = cursor.next().unwrap().unwrap();
    assert!(second.should_not_skip);
    second.docs_into(&mut docs).unwrap();
    assert_eq!(docs, vec![6]);
    assert_eq!(second.summary_dot(&dense).unwrap(), 0);
    assert!(cursor.next().unwrap().is_none());
    assert!(cursor.next().unwrap().is_none());
    assert!(p.cursor(3, b"13").unwrap().is_none());
    assert!(p.cursor(3, b"99").unwrap().is_none());
    assert!(p.cursor(4, b"12").unwrap().is_none());
}

#[test]
fn a_segment_without_clustered_fields_has_no_field_count() {
    let mut out = Vec::new();
    let start = header(&mut out);
    let sit = finish(out, start);
    let (sip, _) = sip(|_| {});
    let p = ClusteredPostings::open(&sit, sip, &ID).unwrap();
    assert!(!p.has_field(0));
}

#[test]
fn rejects_corrupt_files() {
    let (good_sip, start) = sip(|o| o.write_vlong(0));
    // Wrong segment id.
    assert!(ClusteredPostings::open(&sit(&[]), good_sip.clone(), &[1; ID_LENGTH]).is_err());
    // Too short to hold the trailing long and footer.
    assert!(directory_offset(&[0; 10], "x").is_err());
    // A body offset past the end.
    let mut out = Vec::new();
    header(&mut out);
    let bad = finish(out, 1 << 40);
    assert!(ClusteredPostings::open(&bad, good_sip.clone(), &ID).is_err());
    // A term running past the end of the dictionary.
    let mut out = Vec::new();
    let s = header(&mut out);
    out.write_vint(1);
    out.write_vint(3);
    out.write_vlong(1);
    out.write_vint(1000);
    assert!(ClusteredPostings::open(&finish(out, s), good_sip.clone(), &ID).is_err());
    // A postings offset outside `.sip`.
    let p = ClusteredPostings::open(&sit(&[("1", 1 << 40)]), good_sip, &ID).unwrap();
    assert!(p.clusters(3, b"1").is_err());
    assert!(p.cursor(3, b"1").is_err());
    // Counts larger than what is left.
    for body in [
        |o: &mut Vec<u8>| {
            o.write_vlong(1);
            o.write_vlong(1 << 30);
        },
        |o: &mut Vec<u8>| {
            o.write_vlong(1);
            o.write_vlong(0);
            o.write_byte(0);
            o.write_vlong(1 << 30);
        },
        |o: &mut Vec<u8>| {
            o.write_vlong(1);
            o.write_vlong(0);
            o.write_byte(0);
            o.write_vlong(1);
            o.write_vint(70000);
            o.write_byte(1);
        },
    ] {
        let (sip, start) = sip(body);
        let p = ClusteredPostings::open(&sit(&[("1", start)]), sip, &ID).unwrap();
        assert!(p.clusters(3, b"1").is_err());
        assert!(p.cursor(3, b"1").unwrap().unwrap().next().is_err());
    }
    let _ = start;
}
