/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.opensearch.common.lucene.search.TopDocsAndMaxScore;
import org.opensearch.search.SearchHit;
import org.opensearch.search.aggregations.AggregatorBase;
import org.opensearch.search.aggregations.InternalAggregation;
import org.opensearch.search.aggregations.metrics.InternalTopHits;
import org.opensearch.search.fetch.FetchPhase;
import org.opensearch.search.fetch.FetchSearchResult;
import org.opensearch.search.internal.SubSearchContext;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

import static org.lucenerust.opensearch.NativeAggregationTree.get;

/**
 * {@code top_hits} by score ({@code TopHitsAggregator} without a sort or rescore) as a native tree
 * node: the native side runs each bucket's {@code TopScoreDocCollector} over the scored matches and
 * hands back its top documents and hit count; the rest is {@code TopHitsAggregator.buildAggregation}
 * as Java runs it -- the fetch phase over the aggregator's own sub-search context, each hit given its
 * shard and score. A sort, a rescore, or a size Java would refuse falls back.
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
        if (sub.sort() != null || (agg.context().rescore() != null && agg.context().rescore().isEmpty() == false)) {
            return null;
        }
        // TopHitsAggregator.collect's topN: TopScoreDocCollectorManager refuses one below 1.
        long topN = Math.min((long) sub.from() + sub.size(), sub.searcher().getIndexReader().maxDoc());
        if (topN < 1) {
            return null;
        }
        return new Node(agg, sub, fetch, (int) topN);
    }

    static final class Node extends NativeAggregationTree.Node {
        final SubSearchContext sub;
        final FetchPhase fetch;
        final int topN;

        Node(AggregatorBase agg, SubSearchContext sub, FetchPhase fetch, int topN) {
            super(agg, List.of());
            this.sub = sub;
            this.fetch = fetch;
            this.topN = topN;
        }

        @Override
        void write(ByteArrayOutputStream out) {
            out.write(TOP_HITS);
            NativeAggregations.writeInt(out, topN);
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

        /** {@code TopHitsAggregator.buildAggregation} from the collected top documents, by score. */
        private InternalTopHits build(TopDocs topDocs) {
            float maxScore = topDocs.scoreDocs.length > 0 ? topDocs.scoreDocs[0].score : Float.NaN;
            TopDocsAndMaxScore topDocsAndMaxScore = new TopDocsAndMaxScore(topDocs, maxScore);
            sub.queryResult().topDocs(topDocsAndMaxScore, null);
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
            }
            return new InternalTopHits(agg.name(), sub.from(), sub.size(), topDocsAndMaxScore, fetchResult.hits(), agg.metadata());
        }
    }
}
