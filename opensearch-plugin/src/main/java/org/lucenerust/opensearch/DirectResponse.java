/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.function.Function;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.DocumentStoredFieldVisitor;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.util.BytesRef;
import org.opensearch.common.xcontent.XContentHelper;
import org.opensearch.common.xcontent.json.JsonXContent;
import org.opensearch.core.common.bytes.BytesArray;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.mapper.Uid;
import org.opensearch.search.fetch.subphase.FetchSourceContext;

/** The JSON a direct search answers with: what a search response carries for a single shard, hit by hit. */
final class DirectResponse {
    /** What the request asked of the response. */
    record Shape(FetchSourceContext fetch, boolean totalAsInt, boolean ids) {
        boolean source() {
            return fetch == null || fetch.fetchSource();
        }

        Function<Map<String, ?>, Map<String, Object>> filter() {
            return fetch != null && fetch.fetchSource() && (fetch.includes().length > 0 || fetch.excludes().length > 0) ? fetch.getFilter() : null;
        }
    }

    /** The hits found: documents, scores (or the sort members) and the counts the native search returned. */
    record Hits(int[] docs, float[] scores, long[] counts, String[] sort, String maxScore, String aggregations, DirectFields fields) {
        Hits(int[] docs, float[] scores, long[] counts, String[] sort, String maxScore) {
            this(docs, scores, counts, sort, maxScore, null, null);
        }

        Hits(int[] docs, float[] scores, long[] counts, String[] sort, String maxScore, String aggregations) {
            this(docs, scores, counts, sort, maxScore, aggregations, null);
        }
    }

    private DirectResponse() {}

    /** The response's JSON, or null when a hit holds something a direct search leaves to OpenSearch. */
    static String write(String index, IndexReader reader, long start, int from, int size, long countLimit, Hits hits, Shape shape) throws IOException {
        long[] counts = hits.counts();
        int found = (int) counts[0];
        StringBuilder out = new StringBuilder(512 + Math.max(found - from, 0) * 512);
        out.append("{\"took\":").append((System.nanoTime() - start) / 1_000_000)
            .append(",\"timed_out\":false,\"_shards\":{\"total\":1,\"successful\":1,\"skipped\":0,\"failed\":0},\"hits\":{");
        if (countLimit == 0 && shape.totalAsInt()) {
            out.append("\"total\":-1,");
        } else if (countLimit > 0) {
            boolean lowerBound = counts[2] == 1 || counts[1] > countLimit;
            long value = lowerBound ? countLimit : counts[1];
            if (shape.totalAsInt()) {
                out.append("\"total\":").append(value).append(',');
            } else {
                out.append("\"total\":{\"value\":").append(value).append(",\"relation\":\"").append(lowerBound ? "gte" : "eq").append("\"},");
            }
        }
        out.append("\"max_score\":").append(maxScore(hits, found, size)).append(",\"hits\":[");
        String[] fields = hits.fields() == null ? null : hits.fields().render(reader, hits.docs(), from, found);
        for (int i = from; i < found; i++) {
            if (i > from && out.charAt(out.length() - 1) != '[') {
                out.append(',');
            }
            if (hit(out, index, reader, hits, i, shape, fields == null ? null : fields[i - from]) == false) {
                return null;
            }
        }
        out.append("]}");
        if (hits.aggregations() != null) {
            out.append(',').append(hits.aggregations());
        }
        out.append('}');
        return out.toString();
    }

    private static String maxScore(Hits hits, int found, int size) {
        if (found == 0 || size == 0) {
            return "null";
        }
        return hits.sort() == null ? Float.toString(hits.scores()[0]) : hits.maxScore();
    }

    private static boolean hit(StringBuilder out, String index, IndexReader reader, Hits hits, int i, Shape shape, String fields) throws IOException {
        Document doc = null;
        BytesRef id = null;
        BytesRef sourceBytes = null;
        if (shape.ids() || shape.source()) {
            DocumentStoredFieldVisitor visitor = shape.source()
                ? new DocumentStoredFieldVisitor("_id", "_source", "_routing", "_ignored")
                : new DocumentStoredFieldVisitor("_id", "_routing", "_ignored");
            reader.storedFields().document(hits.docs()[i], visitor);
            doc = visitor.getDocument();
            id = doc.getBinaryValue("_id");
            sourceBytes = doc.getBinaryValue("_source");
            if (id == null || sourceBytes == null && shape.source()) {
                return false;
            }
        }
        out.append("{\"_index\":");
        json(out, index);
        if (shape.ids()) {
            out.append(",\"_id\":");
            json(out, Uid.decodeId(id.bytes, id.offset, id.length));
        }
        out.append(",\"_score\":").append(hits.scores() == null ? "null" : Float.toString(hits.scores()[i]));
        String routing = shape.ids() ? doc.get("_routing") : null;
        if (routing != null) {
            out.append(",\"_routing\":");
            json(out, routing);
        }
        String[] ignored = shape.ids() ? doc.getValues("_ignored") : new String[0];
        if (ignored.length > 0) {
            out.append(",\"_ignored\":[");
            for (int k = 0; k < ignored.length; k++) {
                if (k > 0) {
                    out.append(',');
                }
                json(out, ignored[k]);
            }
            out.append(']');
        }
        if (fields != null) {
            out.append(",\"fields\":").append(fields);
        }
        if (shape.source()) {
            out.append(",\"_source\":").append(source(sourceBytes, shape.filter()));
        }
        if (hits.sort() != null) {
            out.append(',').append(hits.sort()[i]);
        }
        out.append('}');
        return true;
    }

    /** The stored source, filtered by {@code _source} includes and excludes as the fetch phase does. */
    private static String source(BytesRef bytes, Function<Map<String, ?>, Map<String, Object>> filter) throws IOException {
        if (filter == null) {
            return new String(bytes.bytes, bytes.offset, bytes.length, StandardCharsets.UTF_8);
        }
        Map<String, Object> map = XContentHelper.convertToMap(new BytesArray(bytes.bytes, bytes.offset, bytes.length), true, MediaTypeRegistry.JSON).v2();
        try (XContentBuilder builder = JsonXContent.contentBuilder()) {
            builder.map(filter.apply(map));
            return builder.toString();
        }
    }

    /** {@code s} as a JSON string literal. */
    static void json(StringBuilder out, String s) {
        out.append('"');
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"' -> out.append("\\\"");
                case '\\' -> out.append("\\\\");
                case '\n' -> out.append("\\n");
                case '\r' -> out.append("\\r");
                case '\t' -> out.append("\\t");
                default -> {
                    if (c < 0x20) {
                        out.append(String.format("\\u%04x", (int) c));
                    } else {
                        out.append(c);
                    }
                }
            }
        }
        out.append('"');
    }
}
