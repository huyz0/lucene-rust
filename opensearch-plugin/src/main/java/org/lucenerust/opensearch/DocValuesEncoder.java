/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.lucene.search.LongValuesSource;
import org.apache.lucene.search.NumericDocValuesRangeQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.util.BytesRef;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.Collection;
import java.util.List;

/**
 * The Lucene queries OpenSearch builds over doc values, as the nodes {@code
 * crates/lucene-ffi/src/jvm_queries.rs} reads:
 *
 * <ul>
 *   <li>the doc-values-only queries of a field that is not indexed: {@code
 *       SortedNumericDocValuesRangeQuery} (a number or date {@code range}/{@code term}, node 29),
 *       {@code SortedNumericDocValuesSetQuery} ({@code terms}, node 30), {@code
 *       SortedSetDocValuesRangeQuery} (a {@code keyword} or {@code ip} {@code range}/{@code term},
 *       node 31) and the sandbox's {@code SortedSetDocValuesMultiRangeQuery} ({@code ip} {@code
 *       terms}, node 32);
 *   <li>the sandbox's {@code CoveringQuery} ({@code terms_set}), its minimum a constant or
 *       OpenSearch's {@code minimum_should_match_field} over plain integral doc values (node 33);
 *   <li>{@code LongDistanceFeatureQuery} and {@code LatLonPointDistanceFeatureQuery} ({@code
 *       distance_feature} on a {@code date} or {@code geo_point}, nodes 34 and 35).
 * </ul>
 *
 * <p>Every class is Lucene's own, matched by name (most are package-private) and read through
 * {@link Reflect}: the fields their constructors were given, which their {@code equals} compares.
 * Falls back: a {@code terms_set} minimum from a script ({@code covering_script}) or from field
 * data other than plain integral doc values ({@code covering_field_data}).
 */
final class DocValuesEncoder {
    static final byte NODE_DV_NUMERIC_RANGE = 29;
    static final byte NODE_DV_NUMERIC_SET = 30;
    static final byte NODE_DV_ORD_RANGE = 31;
    static final byte NODE_DV_MULTI_RANGE = 32;
    static final byte NODE_COVERING = 33;
    static final byte NODE_LONG_DISTANCE_FEATURE = 34;
    static final byte NODE_GEO_DISTANCE_FEATURE = 35;

    /** What {@link #node} returns for a query that is none of these. */
    static final String NOT_DOC_VALUES = "not_doc_values";

    private static final String DOCUMENT = "org.apache.lucene.document.";
    private static final String SANDBOX = "org.apache.lucene.sandbox.search.";

    private DocValuesEncoder() {}

    /**
     * Appends {@code q} when it is one of these queries: null when it was written, a fallback reason
     * when it cannot be, {@link #NOT_DOC_VALUES} when it is none of them.
     */
    static String node(Query q, ByteArrayOutputStream out, ExtendedEncoder.Child child) {
        try {
            switch (q.getClass().getName()) {
                case DOCUMENT + "SortedNumericDocValuesRangeQuery" -> {
                    NumericDocValuesRangeQuery r = (NumericDocValuesRangeQuery) q;
                    out.write(NODE_DV_NUMERIC_RANGE);
                    writeString(out, r.getField());
                    writeLong(out, r.lowerValue());
                    writeLong(out, r.upperValue());
                    return null;
                }
                case DOCUMENT + "SortedNumericDocValuesSetQuery" -> {
                    return numericSet(q, out);
                }
                case DOCUMENT + "SortedSetDocValuesRangeQuery" -> {
                    BytesRef lower = (BytesRef) ExtendedEncoder.field(q, "lowerValue");
                    BytesRef upper = (BytesRef) ExtendedEncoder.field(q, "upperValue");
                    // The same flags as NODE_TERM_RANGE: 1 a lower bound follows, 2 an upper, 4 and
                    // 8 they are inclusive. A null bound is an open end.
                    int flags = (lower != null ? 1 : 0)
                        | (upper != null ? 2 : 0)
                        | ((boolean) ExtendedEncoder.field(q, "lowerInclusive") ? 4 : 0)
                        | ((boolean) ExtendedEncoder.field(q, "upperInclusive") ? 8 : 0);
                    out.write(NODE_DV_ORD_RANGE);
                    writeString(out, (String) ExtendedEncoder.field(q, "field"));
                    out.write(flags);
                    if (lower != null) {
                        writeBytes(out, lower);
                    }
                    if (upper != null) {
                        writeBytes(out, upper);
                    }
                    return null;
                }
                case SANDBOX + "SortedSetDocValuesMultiRangeQuery" -> {
                    List<?> ranges = (List<?>) ExtendedEncoder.field(q, "rangeClauses");
                    if (ranges.isEmpty()) {
                        return "clause_SortedSetDocValuesMultiRangeQuery";
                    }
                    out.write(NODE_DV_MULTI_RANGE);
                    writeString(out, (String) ExtendedEncoder.field(q, "fieldName"));
                    writeInt(out, ranges.size());
                    for (Object r : ranges) {
                        writeBytes(out, (BytesRef) ExtendedEncoder.field(r, "lower"));
                        writeBytes(out, (BytesRef) ExtendedEncoder.field(r, "upper"));
                    }
                    return null;
                }
                case SANDBOX + "CoveringQuery" -> {
                    return covering(q, out, child);
                }
                case DOCUMENT + "LongDistanceFeatureQuery" -> {
                    out.write(NODE_LONG_DISTANCE_FEATURE);
                    writeString(out, (String) ExtendedEncoder.field(q, "field"));
                    writeLong(out, (long) ExtendedEncoder.field(q, "origin"));
                    writeLong(out, (long) ExtendedEncoder.field(q, "pivotDistance"));
                    return null;
                }
                case DOCUMENT + "LatLonPointDistanceFeatureQuery" -> {
                    out.write(NODE_GEO_DISTANCE_FEATURE);
                    writeString(out, (String) ExtendedEncoder.field(q, "field"));
                    writeLong(out, Double.doubleToRawLongBits((double) ExtendedEncoder.field(q, "originLat")));
                    writeLong(out, Double.doubleToRawLongBits((double) ExtendedEncoder.field(q, "originLon")));
                    writeLong(out, Double.doubleToRawLongBits((double) ExtendedEncoder.field(q, "pivotDistance")));
                    return null;
                }
                default -> {
                    return NOT_DOC_VALUES;
                }
            }
        } catch (ReflectiveOperationException | ClassCastException e) {
            return "extended_reflect";
        }
    }

    /** {@code DocValuesLongHashSet.MISSING}: the empty slot, and the value {@code hasMissingValue} records. */
    private static final long MISSING = Long.MIN_VALUE;

    /** A set query's numbers: the hash table's used slots, and {@link #MISSING} when it holds it. */
    private static String numericSet(Query q, ByteArrayOutputStream out) throws ReflectiveOperationException {
        Object set = ExtendedEncoder.field(q, "numbers");
        long[] table = (long[]) ExtendedEncoder.field(set, "table");
        boolean hasMissing = (boolean) ExtendedEncoder.field(set, "hasMissingValue");
        int count = hasMissing ? 1 : 0;
        for (long v : table) {
            if (v != MISSING) {
                count++;
            }
        }
        out.write(NODE_DV_NUMERIC_SET);
        writeString(out, (String) ExtendedEncoder.field(q, "field"));
        writeInt(out, count);
        if (hasMissing) {
            writeLong(out, MISSING);
        }
        for (long v : table) {
            if (v != MISSING) {
                writeLong(out, v);
            }
        }
        return null;
    }

    /**
     * {@code CoveringQuery}: its queries in the multiset's own order (the order the scorer sums
     * them in), then the minimum. A constant one is rewritten away before a search sees it; OpenSearch's
     * {@code minimum_should_match_field} reads each document's first value of the field.
     */
    private static String covering(Query q, ByteArrayOutputStream out, ExtendedEncoder.Child child)
        throws ReflectiveOperationException {
        Collection<?> queries = (Collection<?>) ExtendedEncoder.field(q, "queries");
        LongValuesSource min = (LongValuesSource) ExtendedEncoder.field(q, "minimumNumberMatch");
        String field = null;
        if (min instanceof LongValuesSource.ConstantLongValuesSource == false) {
            String name = min.getClass().getName();
            if (name.equals("org.opensearch.index.query.TermsSetQueryBuilder$ScriptLongValueSource")) {
                return "covering_script";
            }
            if (name.equals("org.opensearch.index.query.TermsSetQueryBuilder$FieldValuesSource") == false) {
                return "covering_" + min.getClass().getSimpleName();
            }
            Object data = ExtendedEncoder.field(min, "fieldData");
            // Plain sorted-numeric doc values of an integral type, which getLongValues hands over
            // as they are stored.
            if (ExtendedEncoder.numericType(data) != 0) {
                return "covering_field_data";
            }
            field = (String) ExtendedEncoder.field(min, "fieldName");
        }
        out.write(NODE_COVERING);
        writeInt(out, queries.size());
        for (Object sub : queries) {
            String reason = child.node((Query) sub);
            if (reason != null) {
                return reason;
            }
        }
        if (field == null) {
            out.write(0);
            writeLong(out, ((LongValuesSource.ConstantLongValuesSource) min).getValue());
        } else {
            out.write(1);
            writeString(out, field);
        }
        return null;
    }

    private static void writeString(ByteArrayOutputStream out, String s) {
        writeBytes(out, s.getBytes(StandardCharsets.UTF_8));
    }

    private static void writeBytes(ByteArrayOutputStream out, BytesRef b) {
        writeInt(out, b.length);
        out.write(b.bytes, b.offset, b.length);
    }

    private static void writeBytes(ByteArrayOutputStream out, byte[] b) {
        writeInt(out, b.length);
        out.writeBytes(b);
    }

    private static void writeLong(ByteArrayOutputStream out, long v) {
        writeInt(out, (int) v);
        writeInt(out, (int) (v >>> 32));
    }

    private static void writeInt(ByteArrayOutputStream out, int v) {
        out.write(v);
        out.write(v >>> 8);
        out.write(v >>> 16);
        out.write(v >>> 24);
    }
}
