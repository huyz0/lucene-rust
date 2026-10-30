/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.lucene.search.FieldDoc;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.search.grouping.CollapseTopFieldDocs;
import org.apache.lucene.util.BytesRef;
import org.opensearch.common.lucene.search.TopDocsAndMaxScore;
import org.opensearch.index.mapper.KeywordFieldMapper;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.mapper.NumberFieldMapper;
import org.opensearch.search.collapse.CollapseContext;
import org.opensearch.search.internal.SearchContext;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;

/**
 * Field collapsing ({@code collapse}) run natively: what OpenSearch's {@code
 * CollapsingTopDocsCollectorContext} does with a {@code CollapsingTopDocsCollector} (one per slice
 * of a concurrent search, merged by {@link CollapseTopFieldDocs#merge}), with the collector in
 * {@code crates/lucene-search/src/top_field/collapse.rs}.
 *
 * <p>The collapse blob is the concrete field ({@code int} length, UTF-8) and its kind: {@link #NUMERIC} for
 * a {@code NumberFieldType} ({@code CollapsingDocValuesSource.Numeric}), {@link #KEYWORD} for a
 * {@code KeywordFieldType} ({@code .Keyword}) -- the two {@link CollapseContext#createTopDocs}
 * accepts.
 */
final class NativeCollapse {
    static final byte NUMERIC = 0;
    static final byte KEYWORD = 1;
    private static final byte VALUE_MISSING = 0;
    private static final byte VALUE_LONG = 1;
    private static final byte VALUE_BYTES = 2;

    private NativeCollapse() {}

    /** The collapse blob for {@code ctx}'s collapse, or null when its field is neither kind. */
    static byte[] blob(CollapseContext collapse) {
        MappedFieldType type = collapse.getFieldType();
        byte kind;
        if (type != null && type.unwrap() instanceof KeywordFieldMapper.KeywordFieldType) {
            kind = KEYWORD;
        } else if (type != null && type.unwrap() instanceof NumberFieldMapper.NumberFieldType) {
            kind = NUMERIC;
        } else {
            return null;
        }
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        // The doc values of the field an alias names, as CollapsingDocValuesSource reads them.
        byte[] name = type.name().getBytes(StandardCharsets.UTF_8);
        writeInt(out, name.length);
        out.writeBytes(name);
        out.write(kind);
        return out.toByteArray();
    }

    /** The sort the collapse collector orders groups by: the request's, else by score. */
    static Sort sort(SearchContext ctx) {
        return ctx.sort() == null ? Sort.RELEVANCE : ctx.sort().sort;
    }

    /** {@code CollapsingTopDocsCollectorContext.sortByScore}: the max score is the first hit's. */
    static boolean sortByScore(SearchContext ctx) {
        return ctx.sort() == null || SortField.FIELD_SCORE.equals(ctx.sort().sort.getSort()[0]);
    }

    /**
     * The native answer ({@code search_collapsed_blobs}) as {@code CollapsingTopDocsCollectorContext}
     * stores it: one slice's top docs as its collector gives them, several merged by {@link
     * CollapseTopFieldDocs#merge} into {@code numHits}; the max score the first hit's when the score
     * leads the sort, else the slices' tracked ones combined in slice order, NaN untracked.
     */
    static TopDocsAndMaxScore decode(SearchContext ctx, byte[] encoded, int numHits) {
        Sort sort = sort(ctx);
        SortField[] fields = sort.getSort();
        String field = ctx.collapse().getFieldName();
        int scorePos = -1;
        for (int i = 0; i < fields.length; i++) {
            if (fields[i].getType() == SortField.Type.SCORE) {
                scorePos = i;
                break;
            }
        }
        ByteBuffer in = ByteBuffer.wrap(encoded).order(ByteOrder.LITTLE_ENDIAN);
        int slices = in.getInt();
        CollapseTopFieldDocs[] parts = new CollapseTopFieldDocs[slices];
        float maxScore = Float.NaN;
        for (int s = 0; s < slices; s++) {
            long total = in.getLong();
            float sliceMax = Float.intBitsToFloat(in.getInt());
            int n = in.getInt();
            ScoreDoc[] hits = new ScoreDoc[n];
            Object[] values = new Object[n];
            for (int i = 0; i < n; i++) {
                int doc = in.getInt();
                Object[] row = new Object[fields.length];
                for (int k = 0; k < fields.length; k++) {
                    if (SortEncoder.type(fields[k]) == SortEncoder.STRING) {
                        int len = in.getInt();
                        if (len >= 0) {
                            byte[] term = new byte[len];
                            in.get(term);
                            row[k] = new BytesRef(term);
                        }
                    } else {
                        row[k] = SortEncoder.value(fields[k], in.getLong());
                    }
                }
                float score = scorePos == -1 ? Float.NaN : (float) row[scorePos];
                hits[i] = new FieldDoc(doc, score, row);
                values[i] = switch (in.get()) {
                    case VALUE_MISSING -> null;
                    case VALUE_LONG -> in.getLong();
                    case VALUE_BYTES -> {
                        byte[] term = new byte[in.getInt()];
                        in.get(term);
                        yield new BytesRef(term);
                    }
                    default -> throw new IllegalArgumentException("collapse value kind");
                };
            }
            parts[s] = new CollapseTopFieldDocs(field, new TotalHits(total, TotalHits.Relation.EQUAL_TO), hits, fields, values);
            // The reduce's MaxScoreCollector loop: NaN until the first, Math.max after.
            maxScore = Float.isNaN(maxScore) ? sliceMax : Math.max(maxScore, sliceMax);
        }
        // The terminated flag (terminated(encoded)) ends the answer.
        if (in.remaining() != 1) {
            throw new IllegalArgumentException("collapse result: trailing bytes");
        }
        CollapseTopFieldDocs top = parts.length == 1 ? parts[0] : CollapseTopFieldDocs.merge(sort, 0, numHits, parts);
        if (sortByScore(ctx)) {
            maxScore = top.scoreDocs.length > 0 ? top.scoreDocs[0].score : Float.NaN;
        } else if (ctx.trackScores() == false) {
            maxScore = Float.NaN;
        }
        return new TopDocsAndMaxScore(top, maxScore);
    }

    /** Whether a {@code terminate_after} search ended early: the answer's last byte. */
    static boolean terminated(byte[] encoded) {
        return encoded[encoded.length - 1] != 0;
    }

    /** Whether a native failure is the collector refusing the field, which Lucene raises itself. */
    static boolean refused(String message) {
        return message != null
            && (message.contains("the collapse field must be single valued") || message.contains("unexpected doc values type"));
    }

    private static void writeInt(ByteArrayOutputStream out, int v) {
        for (int i = 0; i < 4; i++) {
            out.write(v >>> (8 * i));
        }
    }
}
