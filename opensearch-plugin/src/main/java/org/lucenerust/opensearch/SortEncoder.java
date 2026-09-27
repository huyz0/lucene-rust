/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.FieldDoc;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.SortedNumericSelector;
import org.apache.lucene.search.SortedNumericSortField;
import org.apache.lucene.search.SortedSetSelector;
import org.apache.lucene.search.SortedSetSortField;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.NumericUtils;
import org.opensearch.index.fielddata.IndexFieldData;
import org.opensearch.index.fielddata.IndexNumericFieldData;

import java.io.ByteArrayOutputStream;
import java.lang.reflect.Field;
import java.util.Arrays;
import java.util.List;
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
    /** Key flags' mode field: OpenSearch's {@code MultiValueMode} for its own comparator sources. */
    static final byte MODE_SUM = 4;
    static final byte MODE_AVG = 8;
    static final byte MODE_MEDIAN = 12;
    /**
     * Key flags: a nested key -- its root filter and inner query (query blobs) and {@code
     * max_children} follow its missing value.
     */
    static final byte NESTED_KEY = 16;
    /** Blob options: track the max score over every match (track_scores). */
    static final byte TRACK_MAX_SCORE = 1;
    static final byte TERMINATE_AFTER = 2;
    static final byte COUNT_SEGMENTS = 4;
    /** Blob options: per-segment index-sort prefix flags follow the slices. */
    static final byte INDEX_SORTED = 8;
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
        return encode(sort, after, trackMaxScore, slices, terminateAfter, countSegments, null);
    }

    /**
     * {@link #encode(Sort, FieldDoc, boolean, int[][], int, boolean)} with, per segment, whether
     * the sort is a prefix of its index sort ({@link #indexSorted}; null for none): such a
     * segment's documents come in the sort's order, and the collector ends it at the first one that
     * does not compete, as {@code TopFieldCollector} does.
     */
    @SuppressWarnings("deprecation")
    public static Encoded encode(
        Sort sort,
        FieldDoc after,
        boolean trackMaxScore,
        int[][] slices,
        int terminateAfter,
        boolean countSegments,
        boolean[] indexSorted
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
            Mode mode = mode(f);
            if (mode != null) {
                flags |= mode.flags();
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
                } else if (mode != null) {
                    // What the comparator source hands its comparator: `missingObject`.
                    writeLong(out, comparable(type, mode.missing()));
                    if (mode.parents() != null) {
                        writeInt(out, mode.parents().length);
                        out.writeBytes(mode.parents());
                        writeInt(out, mode.children().length);
                        out.writeBytes(mode.children());
                        writeInt(out, mode.maxChildren());
                    }
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
        out.write(
            (trackMaxScore ? TRACK_MAX_SCORE : 0) | (terminateAfter > 0 ? TERMINATE_AFTER : 0) | (countSegments ? COUNT_SEGMENTS : 0)
                | (indexSorted != null ? INDEX_SORTED : 0)
        );
        if (terminateAfter > 0) {
            writeInt(out, terminateAfter);
        }
        NativeAggregations.writeSlices(out, slices);
        if (indexSorted != null) {
            writeInt(out, indexSorted.length);
            for (boolean b : indexSorted) {
                out.write(b ? 1 : 0);
            }
        }
        return new Encoded(out.toByteArray(), null);
    }

    /** The blob type of {@code f}, or -1 when it has none. */
    static byte type(SortField f) {
        if (f.getClass() == SortField.class) {
            return switch (f.getType()) {
                case SCORE -> SCORE;
                case DOC -> DOC;
                case CUSTOM -> {
                    Mode mode = mode(f);
                    yield mode == null ? -1 : mode.type();
                }
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

    /**
     * A numeric key OpenSearch sorts with its own comparator source ({@code LongValuesComparatorSource}
     * and the {@code Int}, {@code Double} and {@code Float} ones) -- which it does for {@code mode}
     * {@code sum}, {@code avg} and {@code median}: its blob type, its mode flags and the missing value
     * its comparator is given.
     */
    record Mode(byte type, byte flags, Object missing, byte[] parents, byte[] children, int maxChildren) {}

    private static final String SOURCES = "org.opensearch.index.fielddata.fieldcomparator.";
    private static final Field SORT_MODE = field(IndexFieldData.XFieldComparatorSource.class, "sortMode");
    private static final Field MISSING = field(IndexFieldData.XFieldComparatorSource.class, "missingValue");
    private static final Field NESTED = field(IndexFieldData.XFieldComparatorSource.class, "nested");
    private static final Field SKIPPING = field(IndexFieldData.XFieldComparatorSource.class, "enableSkipping");

    /**
     * {@code f}'s {@link Mode}, or null when it is not such a key or not one the native comparator
     * reproduces: skipping disabled (the native comparator always skips with points), a converting
     * source ({@code numeric_type}), a comparator whose type is not the field's own (the native
     * side reads the stored values in the key's encoding), or a nested key whose filters the native
     * side cannot run or whose mode has no nested pick ({@code median}). A nested key ({@link
     * #NESTED_KEY}) is sent with
     * its root filter and inner query; the native side does not skip with it, so the searcher runs
     * one only where Lucene's comparator would not skip either ({@link #hasNested}).
     */
    static Mode mode(SortField f) {
        if (f.getClass() != SortField.class || f.getType() != SortField.Type.CUSTOM) {
            return null;
        }
        if (!(f.getComparatorSource() instanceof IndexFieldData.XFieldComparatorSource src)) {
            return null;
        }
        String name = src.getClass().getName();
        byte type = switch (name.startsWith(SOURCES) ? name.substring(SOURCES.length()) : "") {
            case "LongValuesComparatorSource" -> LONG;
            case "IntValuesComparatorSource" -> INT;
            case "DoubleValuesComparatorSource" -> DOUBLE;
            case "FloatValuesComparatorSource" -> FLOAT;
            default -> -1;
        };
        if (type < 0 || SORT_MODE == null || MISSING == null || NESTED == null || SKIPPING == null) {
            return null;
        }
        try {
            if (SKIPPING.getBoolean(src) == false) {
                return null;
            }
            Object nested = NESTED.get(src);
            byte[] parents = null;
            byte[] children = null;
            int maxChildren = Integer.MAX_VALUE;
            if (nested != null) {
                // XFieldComparatorSource.Nested: the root documents' BitSetProducer (the query
                // it wraps), the inner query, the searcher that rewrites it, max_children.
                if (!(nested instanceof IndexFieldData.XFieldComparatorSource.Nested n)) {
                    return null;
                }
                Object producer = NativeAggregationTree.get(n, "rootFilter");
                org.apache.lucene.search.IndexSearcher searcher = (org.apache.lucene.search.IndexSearcher) NativeAggregationTree.get(
                    n,
                    "searcher"
                );
                if (!(NativeAggregationTree.get(producer, "query") instanceof org.apache.lucene.search.Query parentQuery)) {
                    return null;
                }
                parents = QueryEncoder.encode(searcher.rewrite(parentQuery), field -> true).blob();
                children = QueryEncoder.encode(searcher.rewrite(n.getInnerQuery()), field -> true).blob();
                if (parents == null || children == null) {
                    return null;
                }
                if (n.getNestedSort() != null) {
                    maxChildren = n.getNestedSort().getMaxChildren();
                }
            }
            Field data = field(src.getClass(), "indexFieldData");
            if (data == null || !(data.get(src) instanceof IndexNumericFieldData ifd)) {
                return null;
            }
            boolean own = switch (ifd.getNumericType()) {
                case LONG, DATE -> type == LONG;
                case INT, SHORT, BYTE -> type == INT;
                case DOUBLE -> type == DOUBLE;
                case FLOAT -> type == FLOAT;
                default -> false;
            };
            if (own == false) {
                return null;
            }
            if (type == LONG) {
                Field converter = field(src.getClass(), "converter");
                if (converter == null || converter.get(src) != null) {
                    return null;
                }
            }
            byte flags = switch (((Enum<?>) SORT_MODE.get(src)).name()) {
                case "MIN" -> 0;
                case "MAX" -> MAX;
                // SUM stays Lucene's: its comparator skips with the points of the single values,
                // which a sum can pass, so which documents it keeps depends on where skipping
                // starts. An average or median lies between a document's least and greatest value,
                // so the same skipping is exact for them, and the native answer is Lucene's. A
                // nested key never skips natively and runs only where Lucene's does not either.
                case "SUM" -> nested != null ? MODE_SUM : -1;
                case "AVG" -> MODE_AVG;
                // MultiValueMode.MEDIAN has no nested pick.
                case "MEDIAN" -> nested != null ? -1 : MODE_MEDIAN;
                default -> -1;
            };
            if (flags < 0) {
                return null;
            }
            if (nested != null) {
                flags |= NESTED_KEY;
            }
            return new Mode(type, flags, src.missingObject(MISSING.get(src), f.getReverse()), parents, children, maxChildren);
        } catch (ReflectiveOperationException | RuntimeException | java.io.IOException e) {
            return null;
        }
    }

    private static Field field(Class<?> c, String name) {
        try {
            Field f = c.getDeclaredField(name);
            f.setAccessible(true);
            return f;
        } catch (ReflectiveOperationException | RuntimeException e) {
            return null;
        }
    }

    /**
     * Per leaf of {@code reader}, whether {@code sort} is a prefix of the leaf's index sort --
     * {@code TopFieldCollector.canEarlyTerminateOnPrefix}, field by field with {@link
     * SortField#equals} -- or null when no leaf's is.
     */
    static boolean[] indexSorted(IndexReader reader, Sort sort) {
        List<LeafReaderContext> leaves = reader.leaves();
        boolean[] out = new boolean[leaves.size()];
        boolean any = false;
        SortField[] search = sort.getSort();
        for (int i = 0; i < out.length; i++) {
            Sort index = leaves.get(i).reader().getMetaData().sort();
            if (index == null || search.length > index.getSort().length) {
                continue;
            }
            out[i] = Arrays.asList(search).equals(Arrays.asList(index.getSort()).subList(0, search.length));
            any |= out[i];
        }
        return any ? out : null;
    }

    /** Whether any key of {@code fields} is a nested one ({@link #NESTED_KEY}). */
    static boolean hasNested(SortField[] fields) {
        for (SortField f : fields) {
            Mode m = mode(f);
            if (m != null && m.parents() != null) {
                return true;
            }
        }
        return false;
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
