#![cfg(test)]
use super::*;

/// **The one-pass match read marks what collecting the matches marks.**
/// A dense range over the terms fixture (deletions in three of its four
/// segments) is a constant-scored bit set, copied a word at a time;
/// with the scorer moved past its first documents only the rest are
/// copied; a term goes through the bulk scorer's batches.
#[test]
fn one_pass_match_reads_mark_what_collecting_marks() {
    let dir = lucene_store::FsDirectory::open(std::path::Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/terms_aggs_index"
    )));
    let reader = crate::directory_reader::DirectoryReader::open(&dir).unwrap();
    let mut opened = reader.open_segments().unwrap();
    opened.open_points().unwrap();
    let segments = opened.as_open_segments();
    let filter = |c: Clause| BooleanQuery {
        filter: vec![c],
        ..Default::default()
    };
    let queries = [
        filter(Clause::PointsRange(crate::query::PointsRangeQuery::new(
            "r",
            0,
            i64::MAX,
        ))),
        filter(Clause::Term(crate::query::TermQuery::new(
            "body",
            b"w0".to_vec(),
        ))),
    ];
    let mut copied = 0;
    for seg in &segments {
        let ctx = exec::LeafContext {
            fields: seg.fields,
            doc_in: seg.doc_in,
            pos_in: seg.pos_in,
            pay_in: seg.pay_in,
            live_docs: seg.live_docs,
            points: seg.points,
            norms: None,
            global: None,
            max_doc: seg.max_doc,
            cache: seg.cache,
            reader: seg.reader,
            similarity: seg.similarity,
        };
        let max_doc = seg.reader.unwrap().max_doc;
        for q in &queries {
            let clause = lone_clause(q);
            let (mut words, mut docs, mut buf, mut want_words) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            let want = segment_matches(&ctx, q, &clause, seg.live_docs, &mut buf).unwrap();
            let got = segment_match_read(
                &ctx,
                q,
                &clause,
                seg.live_docs,
                max_doc,
                &mut words,
                &mut docs,
            )
            .unwrap();
            let (Some(want), Some((got, count))) = (want, got) else {
                panic!("{q:?} matches in every segment");
            };
            let want_docs: Vec<i32> = want.unwrap().to_vec();
            assert_eq!(count, want_docs.len() as u64);
            match (
                got,
                column_read(Some(&want_docs), seg.live_docs, max_doc, &mut want_words),
            ) {
                (ColumnRead::Seek(g), ColumnRead::Seek(w)) => assert_eq!(g, w),
                (ColumnRead::Stream(Accept::Marked(g)), ColumnRead::Stream(Accept::Marked(w))) => {
                    assert_eq!(g, w)
                }
                _ => panic!("{q:?}: different reads"),
            }
            // The bit set copied with its scorer moved on: only the
            // documents from there are marked.
            let mut bulk = exec::bulk_boolean(&ctx, q, 1.0, Mode::NoScores)
                .unwrap()
                .unwrap();
            let exec::Bulk::Scorer(scorer, _) = &mut bulk else {
                continue;
            };
            let target = scorer.advance(70).unwrap();
            let mut marks = vec![0u64; usize::try_from(max_doc).unwrap().div_ceil(64)];
            if !copy_constant_bits(&bulk, seg.live_docs, &mut marks) {
                continue;
            }
            copied += 1;
            for &d in &want_docs {
                let set = marks[d as usize >> 6] >> (d & 63) & 1 == 1;
                assert_eq!(set, d >= target, "doc {d}, scorer on {target}");
            }
            let marked: u32 = marks.iter().map(|w| w.count_ones()).sum();
            assert_eq!(
                marked as usize,
                want_docs.iter().filter(|&&d| d >= target).count()
            );
        }
    }
    assert!(copied > 0, "no segment's range was a bit set");
    // Any other bulk scorer is not copied.
    let ctx_seg = &segments[0];
    let ctx = exec::LeafContext {
        fields: ctx_seg.fields,
        doc_in: ctx_seg.doc_in,
        pos_in: ctx_seg.pos_in,
        pay_in: ctx_seg.pay_in,
        live_docs: ctx_seg.live_docs,
        points: ctx_seg.points,
        norms: None,
        global: None,
        max_doc: ctx_seg.max_doc,
        cache: ctx_seg.cache,
        reader: ctx_seg.reader,
        similarity: ctx_seg.similarity,
    };
    let bulk = exec::bulk_boolean(&ctx, &queries[1], 1.0, Mode::NoScores)
        .unwrap()
        .unwrap();
    assert!(!copy_constant_bits(&bulk, None, &mut [0u64; 4]));
}

#[test]
fn a_match_all_is_seen_through_its_wrappers() {
    let all = || Clause::MatchAllDocs(crate::query::MatchAllDocsQuery::new(0));
    let term = || Clause::Term(crate::TermQuery::new("f", b"x".to_vec()));
    assert!(matches_everything(&all()));
    assert!(matches_everything(&Clause::ConstantScore(Box::new(
        crate::query::ConstantScoreQuery::new(all(), 1.0)
    ))));
    assert!(matches_everything(&Clause::Boost(Box::new(
        crate::query::BoostQuery::new(all(), 2.0)
    ))));
    let mut b = BooleanQuery::new();
    b.filter.push(all());
    b.must.push(all());
    assert!(matches_everything(&Clause::Boolean(Box::new(b.clone()))));
    b.must_not.push(term());
    assert!(!matches_everything(&Clause::Boolean(Box::new(b))));
    let mut b = BooleanQuery::new();
    b.should.push(term());
    b.should.push(all());
    assert!(matches_everything(&Clause::Boolean(Box::new(b.clone()))));
    b.minimum_should_match = 2;
    assert!(!matches_everything(&Clause::Boolean(Box::new(b))));
    let mut b = BooleanQuery::new();
    b.must.push(all());
    b.filter.push(term());
    assert!(!matches_everything(&Clause::Boolean(Box::new(b))));
    assert!(!matches_everything(&term()));
}
