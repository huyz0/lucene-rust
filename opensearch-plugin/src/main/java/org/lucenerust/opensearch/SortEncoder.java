/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.lucene.search.FieldDoc;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.SortedNumericSelector;
import org.apache.lucene.search.SortedNumericSortField;
import org.apache.lucene.search.SortedSetSelector;
import org.apache.lucene.search.SortedSetSortField;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.NumericUtils;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;

/**
 * Translates a search's {@link Sort} (and its {@code search_after} {@link FieldDoc}) into the sort
 * blob {@code decode_sort} in {@code crates/lucene-ffi/src/jvm_reader.rs} reads, or says why it
 * cannot -- and turns the native search's per-hit values back into the {@link FieldDoc#fields}
 * Lucene's {@code TopFieldCollector} would have produced.
 *
 * <p>Encodable keys, in any number and order (read path R4):
 *
 * <ul>
 *   <li>the score ({@link SortField.Type#SCORE}, either direction) and the document ({@link
 *       SortField.Type#DOC});
 *   <li>a {@link SortedNumericSortField} of type {@code LONG}, {@code INT}, {@code DOUBLE} or {@code
 *       FLOAT} with the {@code MIN} or {@code MAX} selector -- what OpenSearch builds for a {@code
 *       long}, {@code integer}, {@code short}, {@code byte}, {@code double}, {@code float} or {@code
 *       date} field sorted with {@code mode} {@code min}/{@code max} (the default);
 *   <li>a {@link SortedSetSortField} with the {@code MIN} or {@code MAX} selector and {@code
 *       STRING_FIRST}/{@code STRING_LAST} -- what OpenSearch builds for a {@code keyword} field
 *       sorted with {@code mode} {@code min}/{@code max} and {@code missing} {@code _first}/{@code
 *       _last}.
 * </ul>
 *
 * <p>Values cross the boundary as {@code long}s, each key's comparable form: the value for {@code
 * LONG}/{@code INT}, {@link NumericUtils#doubleToSortableLong} and {@link
 * NumericUtils#floatToSortableInt} for {@code DOUBLE}/{@code FLOAT} (which round-trip exactly), the
 * document id for {@code DOC}, and the float's bits for the score. A keyword key's values are
 * its terms instead, as {@link BytesRef}s ({@code null} for a document without one), both ways.
 */
public final class SortEncoder {
    static final byte SCORE = 0;
    static final byte DOC = 1;
    static final byte LONG = 2;
    static final byte INT = 3;
    static final byte DOUBLE = 4;
    static final byte FLOAT = 5;
    static final byte STRING = 6;
    static final byte REVERSE = 1;
    static final byte MAX = 2;
    /** Blob options: track the max score over every match (track_scores). */
    static final byte TRACK_MAX_SCORE = 1;
    static final byte TERMINATE_AFTER = 2;
    static final byte COUNT_SEGMENTS = 4;
    /** {@code MAX_SORT_KEYS} in {@code jvm_reader.rs}. */
    static final int MAX_KEYS = 16;

    /** An encoded sort, or the reason there is none (exactly one is non-null). */
    public record Encoded(byte[] blob, String fallbackReason) {}

    private SortEncoder() {}

    // getOptimizeSortWithIndexedData is deprecated in 10.5.0 but still read by every comparator
    // (it disables skipping), so a sort that set it must not run on the always-skipping native side.
    public static Encoded encode(Sort sort, FieldDoc after) {
        return encode(sort, after, false);
    }

    /**
     * {@link #encode(Sort, FieldDoc)}, asking also for the max score over every match -- what
     * OpenSearch's {@code MaxScoreCollector} computes for {@code track_scores} when the score does
     * not lead the sort.
     */
    @SuppressWarnings("deprecation")
    public static Encoded encode(Sort sort, FieldDoc after, boolean trackMaxScore) {
        return encode(sort, after, trackMaxScore, new int[0][]);
    }

    /**
     * {@link #encode(Sort, FieldDoc, boolean)} for a concurrent segment search: {@code slices} as
     * {@link NativeAggregations#slices} returns them, each searched by its own collector and the
     * hits merged, as Lucene's concurrent {@code TopFieldCollectorManager} does.
     */
    @SuppressWarnings("deprecation")
    public static Encoded encode(Sort sort, FieldDoc after, boolean trackMaxScore, int[][] slices) {
        return encode(sort, after, trackMaxScore, slices, 0, false);
    }

    /**
     * {@link #encode(Sort, FieldDoc, boolean, int[][])} behind OpenSearch's {@code terminate_after}
     * ({@code terminateAfter} documents let through, {@code 0} for none): the native search collects
     * the query's first {@code terminateAfter} matches in index order, as a sequential search's
     * EarlyTerminatingCollector does, and keeps the top hits among them. With {@code
     * countSegments} the total is a {@code size: 0} search's instead: TotalHitCountCollector's,
     * which takes a whole segment's {@code Weight.count} where there is one.
     */
    @SuppressWarnings("deprecation")
    public static Encoded encode(
        Sort sort,
        FieldDoc after,
        boolean trackMaxScore,
        int[][] slices,
        int terminateAfter,
        boolean countSegments
    ) {
        SortField[] fields = sort.getSort();
        if (fields.length == 0 || fields.length > MAX_KEYS) {
            return new Encoded(null, "sort_keys");
        }
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        out.write(fields.length);
        for (SortField f : fields) {
            byte type = type(f);
            if (type < 0) {
                return new Encoded(null, "sort_" + f.getClass().getSimpleName() + "_" + f.getType());
            }
            if (f.getOptimizeSortWithIndexedData() == false) {
                // The native comparator always skips with points; this one was told not to.
                return new Encoded(null, "sort_unoptimized");
            }
            byte flags = f.getReverse() ? REVERSE : 0;
            if (f instanceof SortedNumericSortField sn && sn.getSelector() == SortedNumericSelector.Type.MAX) {
                flags |= MAX;
            }
            if (f instanceof SortedSetSortField ss && ss.getSelector() == SortedSetSelector.Type.MAX) {
                flags |= MAX;
            }
            out.write(type);
            out.write(flags);
            if (type != SCORE && type != DOC) {
                byte[] name = f.getField().getBytes(StandardCharsets.UTF_8);
                writeInt(out, name.length);
                out.writeBytes(name);
                Object missing = f.getMissingValue();
                if (type == STRING) {
                    // TermOrdValComparator: last only for STRING_LAST, first otherwise.
                    writeLong(out, missing == SortField.STRING_LAST ? 1 : 0);
                } else {
                    // Lucene's numeric comparators treat an unset missing value as 0.
                    writeLong(out, missing == null ? 0 : comparable(type, missing));
                }
            }
        }
        if (after == null) {
            out.write(0);
        } else {
            if (after.fields == null || after.fields.length != fields.length) {
                return new Encoded(null, "search_after_fields");
            }
            out.write(1);
            writeInt(out, after.doc);
            for (int i = 0; i < fields.length; i++) {
                if (type(fields[i]) == STRING) {
                    // A missing term is a value of its own (TermOrdValComparator's null top).
                    if (after.fields[i] == null) {
                        out.write(0);
                    } else if (after.fields[i] instanceof BytesRef b) {
                        out.write(1);
                        writeInt(out, b.length);
                        out.write(b.bytes, b.offset, b.length);
                    } else {
                        return new Encoded(null, "search_after_type");
                    }
                    continue;
                }
                if (after.fields[i] == null) {
                    return new Encoded(null, "search_after_null");
                }
                writeLong(out, comparable(type(fields[i]), after.fields[i]));
            }
        }
        out.write((trackMaxScore ? TRACK_MAX_SCORE : 0) | (terminateAfter > 0 ? TERMINATE_AFTER : 0) | (countSegments ? COUNT_SEGMENTS : 0));
        if (terminateAfter > 0) {
            writeInt(out, terminateAfter);
        }
        NativeAggregations.writeSlices(out, slices);
        return new Encoded(out.toByteArray(), null);
    }

    /** The blob type of {@code f}, or -1 when it has none. */
    static byte type(SortField f) {
        if (f.getClass() == SortField.class) {
            return switch (f.getType()) {
                case SCORE -> SCORE;
                case DOC -> DOC;
                default -> -1;
            };
        }
        if (f.getClass() == SortedNumericSortField.class) {
            return switch (((SortedNumericSortField) f).getNumericType()) {
                case LONG -> LONG;
                case INT -> INT;
                case DOUBLE -> DOUBLE;
                case FLOAT -> FLOAT;
                default -> -1;
            };
        }
        if (f.getClass() == SortedSetSortField.class) {
            // MIN and MAX only: the MIDDLE_* selectors pick a median the native side does not.
            return switch (((SortedSetSortField) f).getSelector()) {
                case MIN, MAX -> STRING;
                default -> -1;
            };
        }
        return -1;
    }

    /** Whether any key of {@code fields} is a keyword key, whose terms come back as bytes. */
    static boolean hasTerms(SortField[] fields) {
        for (SortField f : fields) {
            if (type(f) == STRING) {
                return true;
            }
        }
        return false;
    }

    /**
     * The native search's {@code n} hits as {@link FieldDoc}s: {@code values} holds one {@code
     * long} per key per hit, and {@code terms} (null when no key is a keyword key) each keyword
     * key's term per hit, in order, as a little-endian {@code int} length ({@code -1} for none) and
     * the bytes.
     */
    static FieldDoc[] hits(SortField[] fields, int n, int[] docs, long[] values, byte[] terms) {
        FieldDoc[] hits = new FieldDoc[n];
        int pos = 0;
        for (int i = 0; i < n; i++) {
            Object[] row = new Object[fields.length];
            for (int k = 0; k < fields.length; k++) {
                if (type(fields[k]) != STRING) {
                    row[k] = value(fields[k], values[i * fields.length + k]);
                    continue;
                }
                int len = (terms[pos] & 0xff) | (terms[pos + 1] & 0xff) << 8 | (terms[pos + 2] & 0xff) << 16
                    | (terms[pos + 3] & 0xff) << 24;
                pos += 4;
                if (len >= 0) {
                    row[k] = new BytesRef(java.util.Arrays.copyOfRange(terms, pos, pos + len));
                    pos += len;
                }
            }
            hits[i] = new FieldDoc(docs[i], Float.NaN, row);
        }
        return hits;
    }

    /** A sort value as the native side compares it. */
    static long comparable(byte type, Object v) {
        return switch (type) {
            case SCORE -> Float.floatToIntBits(((Number) v).floatValue());
            case DOC, INT -> ((Number) v).intValue();
            case LONG -> ((Number) v).longValue();
            case DOUBLE -> NumericUtils.doubleToSortableLong(((Number) v).doubleValue());
            case FLOAT -> NumericUtils.floatToSortableInt(((Number) v).floatValue());
            default -> throw new IllegalArgumentException("sort type " + type);
        };
    }

    /** The {@link FieldDoc#fields} entry Lucene's comparator returns for a native value. */
    static Object value(SortField f, long v) {
        return switch (type(f)) {
            case SCORE -> Float.intBitsToFloat((int) v);
            case DOC, INT -> (int) v;
            case LONG -> v;
            case DOUBLE -> NumericUtils.sortableLongToDouble(v);
            case FLOAT -> NumericUtils.sortableIntToFloat((int) v);
            default -> throw new IllegalArgumentException(f.toString());
        };
    }

    private static void writeInt(ByteArrayOutputStream out, int v) {
        for (int i = 0; i < 4; i++) {
            out.write(v >>> (8 * i));
        }
    }

    private static void writeLong(ByteArrayOutputStream out, long v) {
        for (int i = 0; i < 8; i++) {
            out.write((int) (v >>> (8 * i)));
        }
    }
}
