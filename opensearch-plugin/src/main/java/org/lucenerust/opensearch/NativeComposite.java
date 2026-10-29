/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.lucene.index.DocValues;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.BoostQuery;
import org.apache.lucene.search.ConstantScoreQuery;
import org.apache.lucene.search.IndexOrDocValuesQuery;
import org.apache.lucene.search.PointRangeQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.Sort;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.NumericUtils;
import org.opensearch.common.Rounding;
import org.opensearch.index.mapper.DocCountFieldMapper;
import org.opensearch.index.mapper.KeywordFieldMapper;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.search.DocValueFormat;
import org.opensearch.search.aggregations.AggregatorBase;
import org.opensearch.search.aggregations.InternalAggregation;
import org.opensearch.search.aggregations.InternalAggregations;
import org.opensearch.search.aggregations.bucket.composite.CompositeValuesSourceConfig;
import org.opensearch.search.aggregations.bucket.missing.MissingOrder;
import org.opensearch.search.internal.SearchContext;

import java.io.ByteArrayOutputStream;
import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.lucenerust.opensearch.NativeAggregationTree.get;
import static org.lucenerust.opensearch.NativeAggregationTree.writeLong;
import static org.lucenerust.opensearch.NativeAggregationTree.writeString;

/**
 * A top-level {@code composite} aggregation as a native tree node: its sources read off the built
 * {@code CompositeAggregator} (their parsed {@code after} values included), and its shard result
 * built as the aggregator's {@code buildAggregations} builds it.
 */
final class NativeComposite {
    private static final String COMPOSITE = "org.opensearch.search.aggregations.bucket.composite.";

    private NativeComposite() {}

    /** One source: its field, how it reads values, and its order. */
    private record Source(String field, byte values, byte valueKind, double interval, long[] rounding, int reverseMul,
        boolean missingBucket, MissingOrder missingOrder, Object after) {}

    /** The node for a built {@code CompositeAggregator}, or null when one of its sources is not supported. */
    static NativeAggregationTree.Node node(SearchContext ctx, AggregatorBase agg, List<NativeAggregationTree.Node> subs)
        throws Exception {
        if (agg.getClass().getName().equals(COMPOSITE + "CompositeAggregator") == false || CTORS_OK == false) {
            return null;
        }
        IndexReader reader = ctx.searcher().getIndexReader();
        CompositeValuesSourceConfig[] configs = (CompositeValuesSourceConfig[]) get(agg, "sourceConfigs");
        for (LeafReaderContext leaf : reader.leaves()) {
            // Documents counting as many, or an index sort led by the first source's field (which
            // the aggregator walks, buildIndexSortPrefix): not native.
            Sort sort = leaf.reader().getMetaData().sort();
            if (leaf.reader().getFieldInfos().fieldInfo(DocCountFieldMapper.NAME) != null
                || (sort != null && configs.length > 0 && configs[0].fieldType() != null
                    && sort.getSort()[0].getField().equals(configs[0].fieldType().name()))) {
                return null;
            }
        }
        // OpenSearch's filter rewrite answers a lone date_histogram source from the points
        // (FilterRewriteOptimizationContext): faster than reading every document.
        // Both early-terminating paths below are slower shapes: native only under
        // `native_shapes: all`.
        boolean allShapes = RustQueryPhaseSearcher.allShapes(ctx);
        Object rewrite = get(agg, "filterRewriteOptimizationContext");
        if (allShapes == false && rewrite != null && Boolean.TRUE.equals(get(rewrite, "canOptimize"))) {
            return null;
        }
        Object[] sources = (Object[]) get(agg, "sources");
        boolean hasAfter = get(agg, "rawAfterKey") != null;
        List<Source> out = new ArrayList<>(configs.length);
        for (int i = 0; i < configs.length; i++) {
            Source s = source(configs[i], sources[i], hasAfter);
            if (s == null) {
                return null;
            }
            out.add(s);
        }
        // Per segment: whether the aggregator reads the first source sorted (its points or terms),
        // which counts a document once per distinct first-source value. A range query on that field
        // also bounds the values read: not native.
        Method producer = method(sources[0].getClass(), "createSortedDocsProducerOrNull", IndexReader.class, Query.class);
        if (producer == null) {
            return null;
        }
        List<LeafReaderContext> leaves = reader.leaves();
        int size = (int) get(agg, "size");
        boolean[] sortedLead = new boolean[leaves.size()];
        boolean earlyTerminated = false;
        for (int i = 0; i < leaves.size(); i++) {
            Object p = producer.invoke(sources[0], leaves.get(i).reader(), ctx.query());
            if (p != null) {
                // A range query on that field also bounds the values the points read sees.
                if (p.getClass().getSimpleName().equals("PointsSortedDocsProducer") && unwrap(ctx.query()) instanceof PointRangeQuery) {
                    return null;
                }
                // The sorted read stops once the queue is full and the first source's values pass
                // its worst key: past `size` distinct values it visits only the documents of the
                // first ones, where the native pass reads every document. A keyword's values are
                // counted; a points field's are not, so it stays on Lucene.
                if (allShapes == false
                    && (p.getClass().getSimpleName().equals("PointsSortedDocsProducer")
                        || DocValues.getSortedSet(leaves.get(i).reader(), configs[0].fieldType().name()).getValueCount() > size)) {
                    return null;
                }
                sortedLead[i] = true;
                earlyTerminated = true;
            }
        }
        return new Node(agg, subs, out, size, hasAfter, sortedLead, earlyTerminated);
    }

    /** {@code LongValuesSource.extractQuery}. */
    private static Query unwrap(Query q) {
        if (q instanceof BoostQuery b) {
            return unwrap(b.getQuery());
        }
        if (q instanceof IndexOrDocValuesQuery i) {
            return unwrap(i.getIndexQuery());
        }
        if (q instanceof ConstantScoreQuery c) {
            return unwrap(c.getQuery());
        }
        return q;
    }

    private static Source source(CompositeValuesSourceConfig config, Object source, boolean hasAfter) throws Exception {
        MappedFieldType type = config.fieldType();
        if (config.hasScript() || type == null) {
            return null;
        }
        Object vs = config.valuesSource();
        int reverseMul = config.reverseMul();
        boolean missingBucket = config.missingBucket();
        MissingOrder missingOrder = config.missingOrder();
        Object after = hasAfter ? get(source, "afterValue") : null;
        if (hasAfter && after == null && missingBucket == false) {
            return null;
        }
        String cls = source.getClass().getSimpleName();
        String vsCls = vs.getClass().getSimpleName();
        byte kind = NativeAggregations.valueKind(type);
        String field = type.name();
        switch (cls) {
            case "GlobalOrdinalValuesSource":
                if (type instanceof KeywordFieldMapper.KeywordFieldType == false) {
                    return null;
                }
                return new Source(field, (byte) 0, (byte) 0, 0, null, reverseMul, missingBucket, missingOrder, after);
            case "LongValuesSource":
                if (kind != NativeAggregations.LONG) {
                    return null;
                }
                if (vsCls.equals("RoundingValuesSource")) {
                    long[] r = NativeAggregationTree.roundingCode((Rounding) get(vs, "rounding"));
                    return r == null ? null : new Source(field, (byte) 4, kind, 0, r, reverseMul, missingBucket, missingOrder, after);
                }
                return new Source(field, (byte) 1, kind, 0, null, reverseMul, missingBucket, missingOrder, after);
            case "DoubleValuesSource":
                if (kind < 0) {
                    return null;
                }
                if (vsCls.equals("HistogramValuesSource")) {
                    double interval = (double) get(vs, "interval");
                    return new Source(field, (byte) 3, kind, interval, null, reverseMul, missingBucket, missingOrder, after);
                }
                if (kind == NativeAggregations.LONG) {
                    return null;
                }
                return new Source(field, (byte) 2, kind, 0, null, reverseMul, missingBucket, missingOrder, after);
            default:
                return null;
        }
    }

    /** A method {@code c} or a superclass declares, made accessible; or null. */
    private static Method method(Class<?> c, String name, Class<?>... args) {
        for (Class<?> k = c; k != null; k = k.getSuperclass()) {
            try {
                Method m = k.getDeclaredMethod(name, args);
                m.setAccessible(true);
                return m;
            } catch (NoSuchMethodException e) {
                // a superclass may declare it
            } catch (RuntimeException e) {
                return null;
            }
        }
        return null;
    }

    static final class Node extends NativeAggregationTree.Node {
        final List<Source> sources;
        final int size;
        final boolean hasAfter;
        final boolean[] sortedLead;
        final boolean earlyTerminated;

        Node(AggregatorBase agg, List<NativeAggregationTree.Node> subs, List<Source> sources, int size, boolean hasAfter,
            boolean[] sortedLead, boolean earlyTerminated) {
            super(agg, subs);
            this.sources = sources;
            this.size = size;
            this.hasAfter = hasAfter;
            this.sortedLead = sortedLead;
            this.earlyTerminated = earlyTerminated;
        }

        @Override
        void write(ByteArrayOutputStream out) {
            out.write(NativeAggregationTree.COMPOSITE);
            NativeAggregations.writeInt(out, size);
            out.write(sources.size());
            for (Source s : sources) {
                writeString(out, s.field());
                out.write(s.values());
                switch (s.values()) {
                    case 2 -> out.write(s.valueKind());
                    case 3 -> {
                        out.write(s.valueKind());
                        writeLong(out, Double.doubleToRawLongBits(s.interval()));
                    }
                    case 4 -> NativeAggregationTree.writeRounding(out, s.rounding());
                    default -> {}
                }
                out.write((s.reverseMul() == -1 ? 1 : 0) | (s.missingBucket() ? 2 : 0));
                out.write(switch (s.missingOrder()) {
                    case FIRST -> 1;
                    case LAST -> 2;
                    default -> 0;
                });
            }
            out.write(hasAfter ? 1 : 0);
            if (hasAfter) {
                for (Source s : sources) {
                    Object a = s.after();
                    if (a == null) {
                        out.write(0);
                    } else if (a instanceof BytesRef b) {
                        out.write(2);
                        NativeAggregationTree.writeBytes(out, b);
                    } else if (a instanceof Double d) {
                        out.write(1);
                        writeLong(out, NumericUtils.doubleToSortableLong(d));
                    } else {
                        out.write(1);
                        writeLong(out, (Long) a);
                    }
                }
            }
            NativeAggregations.writeInt(out, sortedLead.length);
            for (boolean b : sortedLead) {
                out.write(b ? 1 : 0);
            }
            writeSubs(out);
        }

        @Override
        @SuppressWarnings("unchecked")
        List<InternalAggregation> read(ByteBuffer in, int n) throws ReflectiveOperationException {
            if (n != 1) {
                throw new IllegalStateException("composite under " + n + " buckets");
            }
            int m = in.getInt();
            long[] docs = new long[m];
            List<Object> keys = new ArrayList<>(m);
            for (int b = 0; b < m; b++) {
                docs[b] = in.getLong();
                Comparable<?>[] values = new Comparable<?>[sources.size()];
                for (int i = 0; i < values.length; i++) {
                    values[i] = switch (in.get()) {
                        case 0 -> null;
                        case 1 -> in.getLong();
                        case 2 -> in.getDouble();
                        case 3 -> {
                            byte[] t = new byte[in.getInt()];
                            in.get(t);
                            yield new BytesRef(t);
                        }
                        default -> throw new IllegalStateException("composite value tag");
                    };
                }
                keys.add(KEY_CTOR.newInstance((Object) values));
            }
            List<InternalAggregations> subAggs = readSubs(in, m);
            List<String> names = (List<String>) get(agg, "sourceNames");
            List<DocValueFormat> formats = (List<DocValueFormat>) get(agg, "formats");
            int[] reverseMuls = (int[]) get(agg, "reverseMuls");
            MissingOrder[] missingOrders = (MissingOrder[]) get(agg, "missingOrders");
            List<Object> buckets = new ArrayList<>(m);
            for (int b = 0; b < m; b++) {
                buckets.add(BUCKET_CTOR.newInstance(names, formats, keys.get(b), reverseMuls, missingOrders, docs[b], subAggs.get(b)));
            }
            Object last = m > 0 ? keys.get(m - 1) : null;
            return List.of(
                (InternalAggregation) COMPOSITE_CTOR.newInstance(
                    agg.name(),
                    size,
                    names,
                    formats,
                    buckets,
                    last,
                    reverseMuls,
                    missingOrders,
                    earlyTerminated,
                    agg.metadata()
                )
            );
        }
    }

    private static final Class<?> KEY = cls(COMPOSITE + "CompositeKey");
    private static final Class<?> BUCKET = cls(COMPOSITE + "InternalComposite$InternalBucket");
    private static final Constructor<?> KEY_CTOR = ctor(KEY, Comparable[].class);
    private static final Constructor<?> BUCKET_CTOR = ctor(
        BUCKET,
        List.class,
        List.class,
        KEY,
        int[].class,
        MissingOrder[].class,
        long.class,
        InternalAggregations.class
    );
    private static final Constructor<?> COMPOSITE_CTOR = ctor(
        cls(COMPOSITE + "InternalComposite"),
        String.class,
        int.class,
        List.class,
        List.class,
        List.class,
        KEY,
        int[].class,
        MissingOrder[].class,
        boolean.class,
        Map.class
    );
    private static final boolean CTORS_OK = KEY_CTOR != null && BUCKET_CTOR != null && COMPOSITE_CTOR != null;

    private static Class<?> cls(String c) {
        try {
            return Class.forName(c);
        } catch (ReflectiveOperationException | RuntimeException e) {
            return Object.class;
        }
    }

    private static Constructor<?> ctor(Class<?> c, Class<?>... args) {
        try {
            Constructor<?> k = c.getDeclaredConstructor(args);
            k.setAccessible(true);
            return k;
        } catch (ReflectiveOperationException | RuntimeException e) {
            return null;
        }
    }
}
