/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.lucene.search.FieldDoc;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.SortedNumericSortField;
import org.apache.lucene.search.SortedSetSortField;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopFieldDocs;
import org.apache.lucene.search.TotalHits;
import org.opensearch.common.lucene.search.TopDocsAndMaxScore;
import org.opensearch.search.SearchHit;
import org.opensearch.search.aggregations.Aggregator;
import org.opensearch.search.aggregations.AggregatorBase;
import org.opensearch.search.aggregations.InternalAggregation;
import org.opensearch.search.aggregations.metrics.InternalTopHits;
import org.opensearch.search.fetch.FetchPhase;
import org.opensearch.search.fetch.FetchSearchResult;
import org.opensearch.search.internal.SubSearchContext;
import org.opensearch.search.sort.SortAndFormats;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

import static org.lucenerust.opensearch.NativeAggregationTree.get;

/**
 * {@code top_hits} ({@code TopHitsAggregator}) as a native tree node: the native side runs each
 * bucket's {@code TopScoreDocCollector} over the scored matches -- or, with a sort, its {@code
 * TopFieldCollector} (exact counts, as {@code TopFieldCollectorManager(sort, topN, null,
 * Integer.MAX_VALUE, false)} is) beside a {@code MaxScoreCollector} for {@code track_scores} -- and
 * hands back its top documents, sort values and hit count; the rest is {@code
 * TopHitsAggregator.buildAggregation} as Java runs it: the fetch phase over the aggregator's own
 * sub-search context, each hit given its shard, score and sort values. A rescore, a size Java would
 * refuse, a sort key other than the score, the document or a plain numeric or keyword field (a
 * {@code mode}, nested or geo-distance key), and {@code track_scores} under a {@code global}
 * (whose {@code populateScores} scores the main query, not the one collected) fall back.
 */
final class NativeTopHits {
    private NativeTopHits() {}

    static final byte TOP_HITS = 12;

    private static final String CLASS = "org.opensearch.search.aggregations.metrics.TopHitsAggregator";

    /** The node for a built {@code top_hits} aggregator, or null. */
    static NativeAggregationTree.Node node(AggregatorBase agg) throws ReflectiveOperationException {
        if (agg.getClass().getName().equals(CLASS) == false) {
            return null;
        }
        if (!(get(agg, "subSearchContext") instanceof SubSearchContext sub) || !(get(agg, "fetchPhase") instanceof FetchPhase fetch)) {
            return null;
        }
        SortAndFormats sort = sub.sort();
        if (sort == null && agg.context().rescore() != null && agg.context().rescore().isEmpty() == false) {
            return null;
        }
        // TopHitsAggregator.collect's topN: TopScoreDocCollectorManager refuses one below 1.
        long topN = Math.min((long) sub.from() + sub.size(), sub.searcher().getIndexReader().maxDoc());
        if (topN < 1) {
            return null;
        }
        byte[] sortBlob = null;
        if (sort != null) {
            for (SortField f : sort.sort.getSort()) {
                if (plainKey(f) == false) {
                    return null;
                }
            }
            if (sub.trackScores()) {
                for (Aggregator p = agg.parent(); p != null; p = p.parent()) {
                    if (p.getClass().getName().equals("org.opensearch.search.aggregations.bucket.global.GlobalAggregator")) {
                        return null;
                    }
                }
            }
            SortEncoder.Encoded encoded = SortEncoder.encode(sort.sort, null, sub.trackScores());
            if (encoded.fallbackReason() != null) {
                return null;
            }
            sortBlob = encoded.blob();
        }
        return new Node(agg, sub, fetch, (int) topN, sortBlob);
    }

    /** A key the native fed collector sorts by: the score, the document, a numeric or keyword field. */
    private static boolean plainKey(SortField f) {
        if (f.getClass() == SortField.class) {
            return f.getType() == SortField.Type.SCORE || f.getType() == SortField.Type.DOC;
        }
        return f.getClass() == SortedNumericSortField.class || f.getClass() == SortedSetSortField.class;
    }

    static final class Node extends NativeAggregationTree.Node {
        final SubSearchContext sub;
        final FetchPhase fetch;
        final int topN;
        /** The sort's blob ({@link SortEncoder}), or null by score. */
        final byte[] sortBlob;

        Node(AggregatorBase agg, SubSearchContext sub, FetchPhase fetch, int topN, byte[] sortBlob) {
            super(agg, List.of());
            this.sub = sub;
            this.fetch = fetch;
            this.topN = topN;
            this.sortBlob = sortBlob;
        }

        @Override
        void write(ByteArrayOutputStream out) {
            out.write(TOP_HITS);
            NativeAggregations.writeInt(out, topN);
            if (sortBlob == null) {
                out.write(0);
            } else {
                out.write(1);
                NativeAggregations.writeInt(out, sortBlob.length);
                out.write(sortBlob, 0, sortBlob.length);
            }
        }

        @Override
        List<InternalAggregation> read(ByteBuffer in, int n) {
            count(in, n);
            List<InternalAggregation> out = new ArrayList<>(n);
            for (int i = 0; i < n; i++) {
                if (in.get() == 0) {
                    // A bucket that collected nothing.
                    out.add(agg.buildEmptyAggregation());
                    continue;
                }
                long total = in.getLong();
                if (sortBlob != null) {
                    out.add(readSorted(in, total));
                    continue;
                }
                int count = in.getInt();
                if (count < 0 || count > topN) {
                    throw new IllegalStateException("lucene-rust: " + count + " top hits for a size of " + topN);
                }
                ScoreDoc[] docs = new ScoreDoc[count];
                for (int d = 0; d < count; d++) {
                    int doc = in.getInt();
                    docs[d] = new ScoreDoc(doc, Float.intBitsToFloat(in.getInt()));
                }
                out.add(build(new TopDocs(new TotalHits(total, TotalHits.Relation.EQUAL_TO), docs)));
            }
            return out;
        }

        /** A sorted bucket's hits, after its total: {@code TopFieldCollector.topDocs()} and the max score. */
        private InternalTopHits readSorted(ByteBuffer in, long total) {
            float maxScore = Float.intBitsToFloat(in.getInt());
            int count = in.getInt();
            if (count < 0 || count > topN) {
                throw new IllegalStateException("lucene-rust: " + count + " top hits for a size of " + topN);
            }
            SortField[] fields = sub.sort().sort.getSort();
            int[] docs = new int[count];
            float[] scores = new float[count];
            long[] values = new long[count * fields.length];
            for (int d = 0; d < count; d++) {
                docs[d] = in.getInt();
            }
            for (int d = 0; d < count; d++) {
                scores[d] = Float.intBitsToFloat(in.getInt());
            }
            for (int v = 0; v < values.length; v++) {
                values[v] = in.getLong();
            }
            int len = in.getInt();
            if (len < 0 || len > in.remaining()) {
                throw new IllegalStateException("lucene-rust: " + len + " bytes of top hits' sort terms");
            }
            byte[] terms = new byte[len];
            in.get(terms);
            FieldDoc[] hits = SortEncoder.hits(fields, count, docs, values, terms);
            for (int d = 0; d < count; d++) {
                // TopFieldCollector.populateScores, under track_scores (NaN otherwise, as collected).
                hits[d].score = scores[d];
            }
            return build(new TopFieldDocs(new TotalHits(total, TotalHits.Relation.EQUAL_TO), hits, fields), maxScore);
        }

        /** {@code TopHitsAggregator.buildAggregation} from the collected top documents, by score. */
        private InternalTopHits build(TopDocs topDocs) {
            return build(topDocs, topDocs.scoreDocs.length > 0 ? topDocs.scoreDocs[0].score : Float.NaN);
        }

        /** {@code TopHitsAggregator.buildAggregation} from the collected top documents and max score. */
        private InternalTopHits build(TopDocs topDocs, float maxScore) {
            TopDocsAndMaxScore topDocsAndMaxScore = new TopDocsAndMaxScore(topDocs, maxScore);
            sub.queryResult().topDocs(topDocsAndMaxScore, sub.sort() == null ? null : sub.sort().formats);
            int[] docIdsToLoad = new int[topDocs.scoreDocs.length];
            for (int i = 0; i < topDocs.scoreDocs.length; i++) {
                docIdsToLoad[i] = topDocs.scoreDocs[i].doc;
            }
            sub.docIdsToLoad(docIdsToLoad, 0, docIdsToLoad.length);
            fetch.execute(sub, "fetch_top_hits_aggregation[" + agg.name() + "]");
            FetchSearchResult fetchResult = sub.fetchResult();
            SearchHit[] internalHits = fetchResult.fetchResult().hits().getHits();
            for (int i = 0; i < internalHits.length; i++) {
                internalHits[i].shard(sub.shardTarget());
                internalHits[i].score(topDocs.scoreDocs[i].score);
                if (topDocs.scoreDocs[i] instanceof FieldDoc fieldDoc) {
                    internalHits[i].sortValues(fieldDoc.fields, sub.sort().formats);
                }
            }
            return new InternalTopHits(agg.name(), sub.from(), sub.size(), topDocsAndMaxScore, fetchResult.hits(), agg.metadata());
        }
    }
}
