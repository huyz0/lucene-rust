/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FilterCodecReader;
import org.apache.lucene.index.FilterDirectoryReader;
import org.apache.lucene.index.FilterLeafReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.StandardDirectoryReader;
import org.apache.lucene.store.ByteBuffersDataOutput;
import org.apache.lucene.store.ByteBuffersIndexOutput;
import org.apache.lucene.store.Directory;
import org.apache.lucene.util.Bits;
import org.lucenerust.opensearch.store.NativeStores;
import org.lucenerust.opensearch.store.RustEncryptedDirectory;
import org.apache.lucene.util.FixedBitSet;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * One native reader per Java searcher reader, opened on first use and closed when the Java reader
 * closes.
 *
 * <p>The native reader is built from exactly what the Java reader sees -- its leaves' own {@link
 * SegmentCommitInfo}s, in leaf order, and each leaf's live docs -- so a native doc ID names the same
 * document as a Java one. An OpenSearch searcher is an NRT reader: its newest segments are not in
 * any commit, and its deletions (hard and soft) are in memory, so opening the latest {@code
 * segments_N} instead would be wrong in both.
 *
 * <p>Lifecycle: the native handle is closed from the Java reader's {@link
 * IndexReader.CacheHelper} closed-listener -- when OpenSearch releases a searcher's reader, the
 * native side releases its mapped segment files with it, so a merged-away segment's files can be
 * deleted. A refresh reuses the previous native reader's unchanged segments.
 */
public final class NativeReaders {
    private static final Logger logger = LogManager.getLogger(NativeReaders.class);

    /** A native handle, or the reason there is none (cached, so a failing reader fails once). */
    public record Acquired(long handle, String fallbackReason) {}

    /**
     * The postings formats the Rust reader decodes: Lucene 10.5.0's default and every default an
     * older Lucene an OpenSearch 2.x/3.x cluster can hold segments of wrote ({@code Lucene90} for
     * 9.0-9.10, {@code Lucene99} for 9.11, {@code Lucene912} for 9.12-10.0, {@code Lucene101} for
     * 10.1-10.2, {@code Lucene103} for 10.3), and the completion formats OpenSearch's {@code
     * completion} fields are written in ({@code Completion104} since 3.x, the older ones by the
     * Lucene versions above): their postings are the default format's, which a query on
     * the field reads, and the suggester itself stays OpenSearch's, on Lucene's reader. {@code
     * ZstdTerms104} ({@code index.codec.terms_compression: zstd}) is the engine's own zstd term-block
     * format over {@code Lucene104} postings, which the native reader decodes too. A field
     * written in any other format makes the whole reader fall back, rather than failing (and
     * falling back) on every query that opens it.
     */
    static final java.util.Set<String> SUPPORTED_POSTINGS_FORMATS = java.util.Set.of(
        "Lucene90",
        "Lucene99",
        "Lucene912",
        "Lucene101",
        "Lucene103",
        "Lucene104",
        "ZstdTerms104",
        "Completion90",
        "Completion99",
        "Completion912",
        "Completion101",
        "Completion104"
    );

    private static final String POSTINGS_FORMAT_ATTRIBUTE = "PerFieldPostingsFormat.format";

    /**
     * The leaf wrapper of the security plugin's field- and document-level security: it hides
     * fields ({@code FieldInfos} without them, {@code null} terms, values and points for them),
     * hides documents (its live docs), and may mask a field's values. All three are a reader view the
     * native side holds ({@code NativeSearch.openReaderView}): masking by calling the plugin's own masking
     * function wherever native code shows a masked value.
     */
    private static final String SECURITY_LEAF = "org.opensearch.security.configuration.DlsFlsFilterLeafReader";

    /** Whether a leaf wrapper of this class shows the segment's fields and documents as they are. */
    private final java.util.function.Predicate<Class<?>> engineClass;
    /** The names of the leaf wrappers whose filtering is a view the native side holds. */
    private final java.util.Set<String> viewLeaves;

    public NativeReaders() {
        this(NativeReaders::byLoader, java.util.Set.of(SECURITY_LEAF));
    }

    /** For tests, whose wrappers share the plugin's class loader. */
    NativeReaders(java.util.function.Predicate<Class<?>> engineClass, java.util.Set<String> viewLeaves) {
        this.engineClass = engineClass;
        this.viewLeaves = viewLeaves;
    }

    private final ConcurrentHashMap<IndexReader.CacheKey, Acquired> byReader = new ConcurrentHashMap<>();
    /**
     * The views of a reader's native reader that its wrappers' filters ask for, by the reader's key
     * and then by what the view hides ({@link View#key}). A wrapper is built per request and
     * shares its delegate's cache key, so the native reader of that key must stay the unfiltered
     * one: what one user's view hides is never what another user's search reads.
     */
    private final ConcurrentHashMap<IndexReader.CacheKey, ConcurrentHashMap<String, Acquired>> views = new ConcurrentHashMap<>();
    /** The most recent handle per index directory: the reuse candidate for the next refresh. */
    private final ConcurrentHashMap<Path, Long> latestByDir = new ConcurrentHashMap<>();
    private final AtomicLong open = new AtomicLong();

    /** Native readers currently open. */
    public long openCount() {
        return open.get();
    }

    /** Readers with a cached outcome, native or fallback: must shrink as Java readers close. */
    public int cachedCount() {
        return byReader.size();
    }

    /**
     * The native handle {@link #acquire} already opened for {@code reader}, or 0: never opens one.
     * The fetch phase reads through the query phase's handle; a reader no search has opened -- the
     * realtime get's internal reader, say -- stays on Lucene rather than paying a native open for a
     * handful of documents.
     */
    public long peek(IndexReader reader) {
        reader = segmentsOf(reader);
        IndexReader base = baseOf(reader);
        if (base == null) {
            return 0;
        }
        IndexReader.CacheHelper helper = base.getReaderCacheHelper();
        Acquired cached = helper == null ? null : byReader.get(helper.getKey());
        return cached == null ? 0 : cached.handle();
    }

    /**
     * The native reader to search {@code searched} with: the one of its segments as the engine's
     * reader shows them, or -- when a wrapper above that reader filters fields and documents (the
     * security plugin's field- and document-level security) -- a view of it that hides what the
     * wrapper hides.
     */
    public Acquired acquire(IndexReader searched) {
        IndexReader reader = segmentsOf(searched);
        IndexReader base = baseOf(reader);
        if (base == null) {
            return new Acquired(0, FallbackReason.READER_NOT_STANDARD.key());
        }
        Acquired opened = acquireBase(base);
        if (base == reader || opened.handle() == 0) {
            return opened;
        }
        return acquireView(reader, base, opened.handle());
    }

    private Acquired acquireBase(IndexReader reader) {
        IndexReader.CacheHelper helper = reader.getReaderCacheHelper();
        if (helper == null) {
            return new Acquired(0, FallbackReason.READER_UNCACHEABLE.key());
        }
        IndexReader.CacheKey key = helper.getKey();
        Acquired cached = byReader.get(key);
        if (cached != null) {
            return cached;
        }
        return byReader.computeIfAbsent(key, k -> {
            Acquired a = openFor(reader);
            // Failures are cached for the reader's life too -- and so must be
            // evicted with it, or an index that always falls back (a completion
            // field) leaks one entry per refresh.
            try {
                helper.addClosedListener(this::onClosed);
            } catch (RuntimeException e) {
                // The reader closed between our search starting and now: nothing will call us back.
                if (a.handle() != 0) {
                    close(a.handle());
                }
                return new Acquired(0, FallbackReason.READER_CLOSED.key());
            }
            return a;
        });
    }

    /**
     * The reader under {@code reader}'s wrappers whose leaves are the engine's own, or null when
     * the wrappers cannot be told apart from the engine's. A leaf is the engine's when every
     * wrapper on it is Lucene's, OpenSearch's or this plugin's: those show the segment's documents
     * and fields as they are.
     */
    private IndexReader baseOf(IndexReader reader) {
        IndexReader base = reader;
        while (hasForeignWrapper(base)) {
            if (base instanceof FilterDirectoryReader f) {
                base = f.getDelegate();
            } else {
                return null;
            }
        }
        return base;
    }

    private boolean hasForeignWrapper(IndexReader reader) {
        for (LeafReaderContext ctx : reader.leaves()) {
            LeafReader leaf = ctx.reader();
            while (true) {
                if (leaf instanceof SegmentReader) {
                    break;
                }
                if (leaf instanceof FilterLeafReader f) {
                    if (engine(f.getClass()) == false) {
                        return true;
                    }
                    leaf = f.getDelegate();
                } else if (leaf instanceof FilterCodecReader f) {
                    if (engine(f.getClass()) == false) {
                        return true;
                    }
                    leaf = f.getDelegate();
                } else {
                    return true;
                }
            }
        }
        return false;
    }

    /** Whether {@code c} is Lucene's, OpenSearch core's or this plugin's: a wrapper that changes nothing a search reads. */
    private boolean engine(Class<?> c) {
        return engineClass.test(c);
    }

    /** {@link #engine} by class loader: Lucene's, OpenSearch core's and this plugin's. */
    private static boolean byLoader(Class<?> c) {
        ClassLoader l = c.getClassLoader();
        return l == IndexReader.class.getClassLoader()
            || l == org.opensearch.index.store.Store.class.getClassLoader()
            || l == NativeReaders.class.getClassLoader();
    }

    /**
     * What a wrapper of {@code reader} hides from one leaf, or shows through a mask: its fields
     * and documents, and the fields whose keyword or binary values it rewrites ({@code masks}: the
     * plugin's masking function of each, by field name).
     */
    private record View(String[][] hidden, long[][] live, java.util.Map<String, Object> masks, String key) {}

    private Acquired acquireView(IndexReader reader, IndexReader base, long baseHandle) {
        IndexReader.CacheHelper helper = base.getReaderCacheHelper();
        if (helper == null) {
            return new Acquired(0, FallbackReason.READER_UNCACHEABLE.key());
        }
        Object viewOrReason = viewOf(reader, base);
        if (viewOrReason instanceof String reason) {
            return new Acquired(0, reason);
        }
        View view = (View) viewOrReason;
        ConcurrentHashMap<String, Acquired> mine = views.computeIfAbsent(helper.getKey(), k -> new ConcurrentHashMap<>());
        Acquired cached = mine.get(view.key());
        if (cached != null) {
            return cached;
        }
        Acquired opened = mine.computeIfAbsent(view.key(), k -> openView(baseHandle, base, view));
        if (views.get(helper.getKey()) != mine) {
            // The base reader closed meanwhile and took its views with it: this one belongs to nobody.
            if (opened.handle() != 0 && mine.remove(view.key(), opened)) {
                close(opened.handle());
            }
            return new Acquired(0, FallbackReason.READER_CLOSED.key());
        }
        return opened;
    }

    /**
     * The security plugin's masking function of each masked field, as the native side calls it
     * whenever it shows one of the field's values: the very function the plugin's own leaf applies
     * in {@code lookupOrd} and {@code binaryValue}, so the digest, salt and regular expressions are
     * the plugin's.
     */
    private static final class Masker implements NativeSearch.FieldMasker {
        private final java.util.Map<String, Object> fields;

        Masker(java.util.Map<String, Object> fields) {
            this.fields = fields;
        }

        @Override
        public byte[] apply(String field, byte[] raw) {
            Object masking = fields.get(field);
            if (masking == null) {
                throw new IllegalStateException("no mask for field " + field);
            }
            return applyTo(masking, raw);
        }
    }

    private static byte[] applyTo(Object masking, byte[] raw) {
        try {
            java.lang.reflect.Method method = masking.getClass().getMethod("apply", org.apache.lucene.util.BytesRef.class);
            method.setAccessible(true);
            org.apache.lucene.util.BytesRef shown = (org.apache.lucene.util.BytesRef) method.invoke(masking, new org.apache.lucene.util.BytesRef(raw));
            return shown == null ? null : java.util.Arrays.copyOfRange(shown.bytes, shown.offset, shown.offset + shown.length);
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException(e);
        }
    }

    /** What a masking function does to a few fixed values: two functions that agree on these and on their description are the same one. */
    private static void fingerprint(java.security.MessageDigest digest, String field, Object masking) {
        digest.update(field.getBytes(StandardCharsets.UTF_8));
        digest.update((byte) 2);
        digest.update(String.valueOf(masking).getBytes(StandardCharsets.UTF_8));
        for (byte[] probe : new byte[][] { {}, { 'a' }, "the quick brown fox 0123456789".getBytes(StandardCharsets.UTF_8), { 0, (byte) 0xff, 1 } }) {
            digest.update((byte) 3);
            byte[] shown = applyTo(masking, probe);
            if (shown != null) {
                digest.update(shown);
            }
        }
    }

    private Acquired openView(long baseHandle, IndexReader base, View view) {
        List<LeafReaderContext> leaves = base.leaves();
        int[] maxDocs = new int[leaves.size()];
        for (int i = 0; i < maxDocs.length; i++) {
            maxDocs[i] = leaves.get(i).reader().maxDoc();
        }
        long[] out = new long[1];
        NativeSearch.Masking masking = view.masks().isEmpty()
            ? null
            : new NativeSearch.Masking(view.masks().keySet().toArray(new String[0]), new Masker(view.masks()));
        int rc = NativeSearch.openReaderView(baseHandle, view.hidden(), masking, maxDocs, view.live(), out);
        if (rc != NativeBridge.OK) {
            logger.warn("lucene-rust: opening a view of a native reader failed ({}): {}", rc, NativeBridge.lastError());
            return new Acquired(0, FallbackReason.NATIVE_OPEN_FAILED.key());
        }
        open.incrementAndGet();
        return new Acquired(out[0], null);
    }

    /**
     * What the wrappers between {@code reader} and {@code base} hide, or the reason the native side
     * cannot stand in for them. Only a wrapper this class knows is taken, and only while it hides
     * fields and documents: one that rewrites values (field masking) cannot be read around.
     */
    private Object viewOf(IndexReader reader, IndexReader base) {
        List<LeafReaderContext> outer = reader.leaves();
        List<LeafReaderContext> inner = base.leaves();
        if (outer.size() != inner.size()) {
            return FallbackReason.READER_NOT_STANDARD.key();
        }
        String[][] hidden = new String[outer.size()][];
        long[][] live = new long[outer.size()][];
        java.util.Map<String, Object> masks = new java.util.TreeMap<>();
        java.security.MessageDigest digest;
        try {
            digest = java.security.MessageDigest.getInstance("SHA-256");
        } catch (java.security.NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
        for (int i = 0; i < hidden.length; i++) {
            LeafReader top = outer.get(i).reader();
            LeafReader bottom = inner.get(i).reader();
            Set<String> maskedHere = new java.util.HashSet<>();
            Set<String> shownMasked = new java.util.HashSet<>();
            for (LeafReader layer = top; layer != bottom;) {
                if (viewLeaves.contains(layer.getClass().getName()) == false) {
                    return FallbackReason.READER_NOT_STANDARD.key();
                }
                Masking masking = maskedFields(layer, bottom);
                if (masking == null) {
                    return FallbackReason.READER_NOT_STANDARD.key();
                }
                maskedHere.addAll(masking.hidden());
                for (java.util.Map.Entry<String, Object> e : masking.shown().entrySet()) {
                    Object earlier = masks.putIfAbsent(e.getKey(), e.getValue());
                    if (earlier != null && earlier != e.getValue() && sameMask(e.getKey(), earlier, e.getValue()) == false) {
                        // Two leaves of one reader masking one field differently: one native function cannot stand in for both.
                        return FallbackReason.READER_NOT_STANDARD.key();
                    }
                    shownMasked.add(e.getKey());
                }
                layer = ((FilterLeafReader) layer).getDelegate();
                if (layer instanceof FilterLeafReader == false && layer != bottom) {
                    return FallbackReason.READER_NOT_STANDARD.key();
                }
            }
            FieldInfos shown = top.getFieldInfos();
            FieldInfos all = bottom.getFieldInfos();
            List<String> gone = new ArrayList<>();
            for (FieldInfo fi : all) {
                FieldInfo seen = shown.fieldInfo(fi.name);
                if (maskedHere.contains(fi.name)) {
                    // Every read of it answers null in the wrapper: nothing is left of it to search.
                    gone.add(fi.name);
                } else if (shownMasked.contains(fi.name)) {
                    // Its stored values and masked doc values remain; the plugin's field info for it
                    // says it is indexed no more, which the native view does as well.
                    if (seen == null || seen.getDocValuesType() != fi.getDocValuesType()) {
                        return FallbackReason.READER_NOT_STANDARD.key();
                    }
                } else if (seen == null) {
                    gone.add(fi.name);
                } else if (sameShape(fi, seen) == false) {
                    return FallbackReason.READER_NOT_STANDARD.key();
                }
            }
            for (FieldInfo fi : shown) {
                if (all.fieldInfo(fi.name) == null) {
                    return FallbackReason.READER_NOT_STANDARD.key();
                }
            }
            java.util.Collections.sort(gone);
            hidden[i] = gone.toArray(new String[0]);
            Bits bits = top.getLiveDocs();
            digest.update(Integer.toString(hidden[i].length).getBytes(StandardCharsets.UTF_8));
            for (String name : hidden[i]) {
                digest.update((byte) 0);
                digest.update(name.getBytes(StandardCharsets.UTF_8));
            }
            digest.update((byte) 1);
            if (bits == null) {
                digest.update((byte) 0);
            } else {
                digest.update((byte) 1);
                live[i] = words(bits, top.maxDoc());
                java.nio.ByteBuffer bytes = java.nio.ByteBuffer.allocate(live[i].length * Long.BYTES);
                bytes.asLongBuffer().put(live[i]);
                digest.update(bytes.array());
            }
        }
        // The mask is part of what the view shows: two roles that mask a field differently never share a handle.
        for (java.util.Map.Entry<String, Object> e : masks.entrySet()) {
            digest.update((byte) 4);
            fingerprint(digest, e.getKey(), e.getValue());
        }
        return new View(hidden, live, masks, java.util.HexFormat.of().formatHex(digest.digest()));
    }

    /** Whether two masking functions of one field are the same one: they describe themselves alike and agree on the probes. */
    private static boolean sameMask(String field, Object a, Object b) {
        try {
            java.security.MessageDigest x = java.security.MessageDigest.getInstance("SHA-256");
            java.security.MessageDigest y = java.security.MessageDigest.getInstance("SHA-256");
            fingerprint(x, field, a);
            fingerprint(y, field, b);
            return java.util.Arrays.equals(x.digest(), y.digest());
        } catch (java.security.NoSuchAlgorithmException | RuntimeException e) {
            return false;
        }
    }

    /** Whether {@code a} and {@code b} index and store the field alike: what a masked field does not. */
    private static boolean sameShape(FieldInfo a, FieldInfo b) {
        return a.getIndexOptions() == b.getIndexOptions()
            && a.getDocValuesType() == b.getDocValuesType()
            && a.docValuesSkipIndexType() == b.docValuesSkipIndexType()
            && a.getPointDimensionCount() == b.getPointDimensionCount()
            && a.getPointIndexDimensionCount() == b.getPointIndexDimensionCount()
            && a.getPointNumBytes() == b.getPointNumBytes()
            && a.getVectorDimension() == b.getVectorDimension()
            && a.hasNorms() == b.hasNorms()
            && a.hasTermVectors() == b.hasTermVectors()
            && a.number == b.number;
    }

    /** What one security leaf masks: the fields it hides outright, and the masking function of each it shows through a mask. */
    private record Masking(Set<String> hidden, java.util.Map<String, Object> shown) {}

    /**
     * What the security plugin's leaf {@code layer} masks over {@code bottom}, or null when the
     * native side cannot stand in for the mask. A masked field answers null for its postings,
     * points, norms, vectors and numeric values, so it is hidden like a field its role may not see.
     * One whose keyword or binary doc values survive is shown natively through the plugin's own
     * masking function, which the native side calls wherever it turns an ordinal or a stored
     * value into the bytes it shows: aggregation keys, sort values, collapse keys, script values.
     * The rule is private to that plugin, so a version that moved it reads as masking those too.
     */
    private static Masking maskedFields(LeafReader layer, LeafReader bottom) {
        try {
            java.lang.reflect.Field field = layer.getClass().getDeclaredField("fmRule");
            field.setAccessible(true);
            Object rule = field.get(layer);
            if (Boolean.TRUE.equals(call(rule, "isAllowAll"))) {
                return new Masking(Set.of(), java.util.Map.of());
            }
            java.lang.reflect.Field metaField = layer.getClass().getDeclaredField("metaFields");
            metaField.setAccessible(true);
            Set<?> meta = (Set<?>) metaField.get(layer);
            Set<String> hidden = new java.util.HashSet<>();
            java.util.Map<String, Object> shown = new java.util.HashMap<>();
            for (FieldInfo fi : bottom.getFieldInfos()) {
                if (meta.contains(fi.name) || Boolean.TRUE.equals(call(rule, "isMasked", fi.name)) == false) {
                    continue;
                }
                switch (fi.getDocValuesType()) {
                    case BINARY, SORTED, SORTED_SET -> {
                        if (fi.hasVectorValues() || "true".equals(fi.getAttribute("knn_field"))) {
                            // The plugin has no masked form of a vector: the field is not there.
                            hidden.add(fi.name);
                            continue;
                        }
                        Object masking = call(rule, "get", fi.name);
                        if (masking == null) {
                            return null;
                        }
                        shown.put(fi.name, masking);
                    }
                    default -> hidden.add(fi.name);
                }
            }
            return new Masking(hidden, shown);
        } catch (ReflectiveOperationException | RuntimeException e) {
            return null;
        }
    }

    private static Object call(Object target, String name, String... args) throws ReflectiveOperationException {
        Class<?>[] types = new Class<?>[args.length];
        java.util.Arrays.fill(types, String.class);
        for (Class<?> c = target.getClass(); c != null; c = c.getSuperclass()) {
            try {
                java.lang.reflect.Method m = c.getDeclaredMethod(name, types);
                m.setAccessible(true);
                return m.invoke(target, (Object[]) args);
            } catch (NoSuchMethodException e) {
                // The superclass declares it.
            }
        }
        throw new NoSuchMethodException(name);
    }

    /**
     * The reader whose segments {@code reader} searches: the engine reader under a view of the
     * writer's buffers ({@link LiveDirectoryReader}), whose native reader every such view shares, or
     * {@code reader} itself.
     */
    private static IndexReader segmentsOf(IndexReader reader) {
        IndexReader base = LiveDirectoryReader.baseOf(reader);
        return base == null ? reader : base;
    }

    private void onClosed(IndexReader.CacheKey key) {
        ConcurrentHashMap<String, Acquired> mine = views.remove(key);
        if (mine != null) {
            for (Acquired view : mine.values()) {
                if (view.handle() != 0) {
                    close(view.handle());
                }
            }
        }
        Acquired a = byReader.remove(key);
        if (a != null && a.handle() != 0) {
            latestByDir.values().remove(a.handle());
            close(a.handle());
        }
    }

    private void close(long handle) {
        int rc = NativeSearch.closeReader(handle);
        if (rc == NativeBridge.OK) {
            open.decrementAndGet();
        } else {
            logger.warn("lucene-rust: closing native reader failed ({}): {}", rc, NativeBridge.lastError());
        }
    }

    /** Closes every native reader; for node shutdown. */
    public void closeAll() {
        for (IndexReader.CacheKey key : List.copyOf(byReader.keySet())) {
            onClosed(key);
        }
    }

    private Acquired openFor(IndexReader reader) {
        if (!(reader instanceof DirectoryReader dr)) {
            return new Acquired(0, FallbackReason.READER_NOT_DIRECTORY.key());
        }
        DirectoryReader unwrapped = FilterDirectoryReader.unwrap(dr);
        if (!(unwrapped instanceof StandardDirectoryReader sdr)) {
            return new Acquired(0, FallbackReason.READER_NOT_STANDARD.key());
        }
        List<LeafReaderContext> leaves = reader.leaves();
        SegmentInfos infos = sdr.getSegmentInfos().clone();
        infos.clear();
        int[] maxDocs = new int[leaves.size()];
        Path path = null;
        for (int i = 0; i < leaves.size(); i++) {
            LeafReader leaf = leaves.get(i).reader();
            for (FieldInfo fi : leaf.getFieldInfos()) {
                String format = fi.getAttribute(POSTINGS_FORMAT_ATTRIBUTE);
                if (format != null && SUPPORTED_POSTINGS_FORMATS.contains(format) == false) {
                    return new Acquired(0, FallbackReason.POSTINGS_FORMAT.key());
                }
            }
            SegmentCommitInfo sci = org.opensearch.common.lucene.Lucene.segmentReader(leaf).getSegmentInfo();
            infos.add(sci);
            maxDocs[i] = leaf.maxDoc();
            if (NativeStores.refusal(sci.info.dir) != null) {
                return new Acquired(0, FallbackReason.DIRECTORY_NOT_NATIVE.key());
            }
            Path p = NativeStores.path(sci.info.dir);
            if (path != null && path.equals(p) == false) {
                return new Acquired(0, FallbackReason.DIRECTORY_MIXED.key());
            }
            path = p;
        }
        if (path == null) {
            if (NativeStores.refusal(sdr.directory()) != null) {
                return new Acquired(0, FallbackReason.DIRECTORY_NOT_NATIVE.key());
            }
            path = NativeStores.path(sdr.directory());
        }
        byte[] infoBytes;
        try {
            ByteBuffersDataOutput buf = new ByteBuffersDataOutput();
            try (ByteBuffersIndexOutput out = new ByteBuffersIndexOutput(buf, "segments", "segments")) {
                infos.write(out);
            }
            infoBytes = buf.toArrayCopy();
        } catch (IOException e) {
            logger.warn("lucene-rust: serializing SegmentInfos failed", e);
            return new Acquired(0, FallbackReason.SEGMENT_INFOS_WRITE.key());
        }
        byte[] pathBytes = RustEncryptedDirectory.nativePath(path).getBytes(StandardCharsets.UTF_8);
        long[][] live = new long[leaves.size()][];
        for (int i = 0; i < leaves.size(); i++) {
            Bits bits = leaves.get(i).reader().getLiveDocs();
            live[i] = bits == null ? null : words(bits, maxDocs[i]);
        }
        long[] out = new long[1];
        Long previous = latestByDir.get(path);
        int rc = NativeSearch.openReader(pathBytes, infoBytes, infos.getGeneration(), previous == null ? 0 : previous, maxDocs, live, out);
        if (rc == NativeBridge.INVALID_HANDLE && previous != null) {
            // The reuse candidate closed under us; open from scratch.
            rc = NativeSearch.openReader(pathBytes, infoBytes, infos.getGeneration(), 0, maxDocs, live, out);
        }
        if (rc != NativeBridge.OK) {
            logger.warn("lucene-rust: native reader open failed for [{}] ({}): {}", path, rc, NativeBridge.lastError());
            return new Acquired(0, FallbackReason.NATIVE_OPEN_FAILED.key());
        }
        long handle = out[0];
        open.incrementAndGet();
        latestByDir.put(path, handle);
        return new Acquired(handle, null);
    }

    /** {@code live} as exactly {@code ceil(maxDoc / 64)} words, no bits set past {@code maxDoc}. */
    static long[] words(Bits live, int maxDoc) {
        int n = (maxDoc + 63) >>> 6;
        long[] words;
        if (live instanceof FixedBitSet fbs) {
            words = java.util.Arrays.copyOf(fbs.getBits(), n);
        } else {
            words = new long[n];
            for (int d = 0; d < maxDoc; d++) {
                if (live.get(d)) {
                    words[d >>> 6] |= 1L << d;
                }
            }
        }
        int tail = maxDoc & 63;
        if (tail != 0) {
            words[n - 1] &= (1L << tail) - 1;
        }
        return words;
    }
}
