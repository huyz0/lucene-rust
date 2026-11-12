/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.highlight;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.StoredFieldVisitor;
import org.apache.lucene.search.vectorhighlight.FieldQuery;
import org.lucenerust.opensearch.NativeBridge;
import org.lucenerust.opensearch.NativeReaders;
import org.lucenerust.opensearch.SearchStats;
import org.lucenerust.opensearch.LiveDirectoryReader;
import org.opensearch.common.settings.Settings;
import org.opensearch.core.common.text.Text;
import org.opensearch.index.analysis.AnalyzerComponentsProvider;
import org.opensearch.index.analysis.NamedAnalyzer;
import org.opensearch.index.analysis.TokenFilterFactory;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.lucene.search.vectorhighlight.CustomFieldQuery;
import org.opensearch.search.fetch.FetchSubPhase;
import org.opensearch.search.fetch.subphase.highlight.FastVectorHighlighter;
import org.opensearch.search.fetch.subphase.highlight.FieldHighlightContext;
import org.opensearch.search.fetch.subphase.highlight.HighlightBuilder;
import org.opensearch.search.fetch.subphase.highlight.HighlightField;
import org.opensearch.search.fetch.subphase.highlight.SearchHighlightContext;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.lang.reflect.Field;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * OpenSearch's {@code fvh} highlighter run by the Rust engine ({@code jvm_highlight_fvh.rs}): the
 * field query is OpenSearch's own {@code CustomFieldQuery}, built as its highlighter builds it and
 * handed over once per field and search; each hit's matches are read from its term vectors in the
 * native segment, and its fragments cut from the values OpenSearch's fragments builder would read
 * (stored fields or {@code _source}).
 *
 * <p>A field whose options the native side does not mirror is highlighted by OpenSearch's
 * highlighter (counted as a fallback, {@code highlight_*}); so is a hit OpenSearch's highlighter
 * throws on (it then fails as it does in OpenSearch), a hit with a value holding a lone surrogate,
 * and a hit the native call fails on (a native error).
 */
public final class NativeFvhHighlighter extends FastVectorHighlighter {
    /** The name it is registered under; {@link HighlightRouting} points {@code fvh} fields at it. */
    public static final String NAME = "accelerated_fvh";

    private static final Logger logger = LogManager.getLogger(NativeFvhHighlighter.class);

    /** {@code jvm_highlight_fvh.rs}'s request version and statuses. */
    private static final byte REQUEST_VERSION = 1;
    private static final int STATUS_FRAGMENTS = 0;
    private static final int STATUS_STOCK = 1;

    /** {@code FieldQuery}'s and {@code QueryPhraseMap}'s fields, or null when they cannot be read. */
    private static final Field FIELD_MATCH;
    private static final Field ROOT_MAPS;
    private static final Field TERM_SET_MAP;
    private static final Field TERMINAL;
    private static final Field SLOP;
    private static final Field BOOST;
    private static final Field NUMBER;
    private static final Field SUB_MAP;

    static {
        Field[] f = new Field[8];
        try {
            Class<?> map = Class.forName("org.apache.lucene.search.vectorhighlight.FieldQuery$QueryPhraseMap");
            f[0] = FieldQuery.class.getDeclaredField("fieldMatch");
            f[1] = FieldQuery.class.getDeclaredField("rootMaps");
            f[2] = FieldQuery.class.getDeclaredField("termSetMap");
            f[3] = map.getDeclaredField("terminal");
            f[4] = map.getDeclaredField("slop");
            f[5] = map.getDeclaredField("boost");
            f[6] = map.getDeclaredField("termOrPhraseNumber");
            f[7] = map.getDeclaredField("subMap");
            for (Field x : f) {
                x.setAccessible(true);
            }
        } catch (ReflectiveOperationException | RuntimeException e) {
            logger.warn("lucene-rust: fvh highlighting stays on OpenSearch's highlighter: {}", e.toString());
            f = new Field[8];
        }
        FIELD_MATCH = f[0];
        ROOT_MAPS = f[1];
        TERM_SET_MAP = f[2];
        TERMINAL = f[3];
        SLOP = f[4];
        BOOST = f[5];
        NUMBER = f[6];
        SUB_MAP = f[7];
    }

    private final NativeReaders readers;
    private final SearchStats stats;
    /** {@code search.highlight.term_vector_multi_value}. */
    private final boolean termVectorMultiValue;

    public NativeFvhHighlighter(Settings settings, NativeReaders readers, SearchStats stats) {
        super(settings);
        this.termVectorMultiValue = SETTING_TV_HIGHLIGHT_MULTI_VALUE.get(settings);
        this.readers = readers;
        this.stats = stats;
    }

    /** One field's native request for one search, or why it has none. */
    private static final class Prepared {
        final byte[] request;
        final long handle;
        final boolean fromSource;
        String fallback;

        Prepared(byte[] request, long handle, boolean fromSource, String fallback) {
            this.request = request;
            this.handle = handle;
            this.fromSource = fromSource;
            this.fallback = fallback;
        }

        static Prepared fallback(String reason) {
            return new Prepared(null, 0, false, reason);
        }
    }

    @Override
    public HighlightField highlight(FieldHighlightContext fieldContext) throws IOException {
        @SuppressWarnings("unchecked")
        Map<String, Prepared> cache = (Map<String, Prepared>) fieldContext.cache.computeIfAbsent(
            NativeFvhHighlighter.class.getName(),
            k -> new HashMap<>()
        );
        Prepared prepared = cache.get(fieldContext.fieldName);
        if (prepared == null) {
            prepared = prepare(fieldContext);
            cache.put(fieldContext.fieldName, prepared);
            if (prepared.fallback != null) {
                stats.fallback(prepared.fallback);
            } else {
                stats.nativeHighlight();
            }
        }
        if (prepared.fallback != null) {
            return super.highlight(fieldContext);
        }
        FetchSubPhase.HitContext hitContext = fieldContext.hitContext;
        List<String> values = prepared.fromSource ? sourceValues(fieldContext) : storedValues(fieldContext);
        for (String v : values) {
            if (NativePlainHighlighter.hasLoneSurrogate(v)) {
                stats.fallback("highlight_lone_surrogate");
                return super.highlight(fieldContext);
            }
        }
        byte[][] out = new byte[1][];
        int rc = NativeBridge.highlightFvh(
            prepared.handle,
            prepared.request,
            hitContext.readerContext().ord,
            hitContext.docId(),
            values,
            out
        );
        if (rc != NativeBridge.OK) {
            logger.warn("lucene-rust: native highlighting of [{}] failed ({}): {}", fieldContext.fieldName, rc, NativeBridge.lastError());
            stats.nativeError();
            // the rest of the search's hits go straight to OpenSearch's highlighter
            prepared.fallback = "highlight_native_error";
            stats.fallback(prepared.fallback);
            return super.highlight(fieldContext);
        }
        ByteBuffer b = ByteBuffer.wrap(out[0]).order(ByteOrder.LITTLE_ENDIAN);
        int status = b.get();
        if (status == STATUS_STOCK) {
            // OpenSearch's highlighter throws for this hit, or cuts a surrogate pair in two.
            if (logger.isDebugEnabled()) {
                byte[] why = new byte[b.remaining()];
                b.get(why);
                logger.debug("lucene-rust: fvh highlighting of [{}] runs on OpenSearch's highlighter: {}", fieldContext.fieldName, new String(why, StandardCharsets.UTF_8));
            }
            stats.fallback("highlight_fvh_hit");
            return super.highlight(fieldContext);
        }
        if (status != STATUS_FRAGMENTS) {
            throw new IllegalStateException("lucene-rust: fvh highlight status " + status);
        }
        int count = b.getInt();
        if (count == 0) {
            return null;
        }
        String[] fragments = new String[count];
        for (int i = 0; i < count; i++) {
            int len = b.getInt();
            char[] chars = new char[len];
            b.asCharBuffer().get(chars);
            b.position(b.position() + len * Character.BYTES);
            fragments[i] = new String(chars);
        }
        return new HighlightField(fieldContext.fieldName, Text.convertFromStringArray(fragments));
    }

    /** {@code SourceSimpleFragmentsBuilder.getFields}'s values. */
    private static List<String> sourceValues(FieldHighlightContext fieldContext) {
        List<Object> raw = fieldContext.hitContext.sourceLookup().extractRawValues(fieldContext.fieldType.name());
        List<String> values = new ArrayList<>(raw.size());
        for (Object o : raw) {
            values.add(o.toString());
        }
        return values;
    }

    /** {@code BaseFragmentsBuilder.getFields}'s values: the hit's stored strings of the field. */
    private static List<String> storedValues(FieldHighlightContext fieldContext) throws IOException {
        String name = fieldContext.fieldType.name();
        List<String> values = new ArrayList<>();
        fieldContext.hitContext.reader().storedFields().document(fieldContext.hitContext.docId(), new StoredFieldVisitor() {
            @Override
            public void stringField(FieldInfo fieldInfo, String value) {
                values.add(value);
            }

            @Override
            public Status needsField(FieldInfo fieldInfo) {
                return fieldInfo.name.equals(name) ? Status.YES : Status.NO;
            }
        });
        return values;
    }

    /** {@code FastVectorHighlighter.highlight}'s builders as a native request, or a fallback reason. */
    private Prepared prepare(FieldHighlightContext fieldContext) throws IOException {
        if (FIELD_MATCH == null) {
            return Prepared.fallback("highlight_fvh_reflection");
        }
        MappedFieldType fieldType = fieldContext.fieldType;
        if (canHighlight(fieldType) == false) {
            // OpenSearch's highlighter refuses it, and says so.
            return Prepared.fallback("highlight_fvh_term_vectors");
        }
        // A realtime view adds the writer's buffers as leaves the native reader does not hold.
        if (LiveDirectoryReader.baseOf(fieldContext.hitContext.topLevelReader()) != null) {
            return Prepared.fallback("highlight_live_view");
        }
        // Opened here when the query phase did not (a query it ran on Lucene): the hits were
        // searched on this reader, so later searches of it read through the same handle.
        long handle = readers.acquire(fieldContext.hitContext.topLevelReader()).handle();
        if (handle == 0) {
            return Prepared.fallback("highlight_reader");
        }
        if (fieldContext.context.getQueryShardContext().resolveDerivedFieldType(fieldContext.fieldName) != null) {
            return Prepared.fallback("highlight_derived_field");
        }
        SearchHighlightContext.FieldOptions options = fieldContext.field.fieldOptions();
        int numberOfFragments = options.numberOfFragments();
        if (numberOfFragments != 0 && options.fragmentOffset() < -1) {
            // OpenSearch's highlighter refuses it, and says so.
            return Prepared.fallback("highlight_fragment_offset");
        }
        if (numberOfFragments != 0) {
            // BaseFragListBuilder.createFieldFragList refuses a size under its minimum on every hit.
            int margin = options.fragmentOffset() == -1 ? 6 : options.fragmentOffset();
            if (options.fragmentCharSize() < Math.max(1, margin * 3)) {
                return Prepared.fallback("highlight_fragment_size");
            }
        }
        Locale locale = options.boundaryScannerLocale() != null ? options.boundaryScannerLocale() : Locale.ROOT;
        HighlightBuilder.BoundaryScannerType scannerType = options.boundaryScannerType() != null
            ? options.boundaryScannerType()
            : HighlightBuilder.BoundaryScannerType.CHARS;
        byte scanner;
        switch (scannerType) {
            case CHARS:
                scanner = 0;
                break;
            case SENTENCE:
                scanner = 1;
                break;
            case WORD:
                scanner = 2;
                break;
            default:
                return Prepared.fallback("highlight_boundary_scanner");
        }
        // The JDK's break rules are the same for every locale but Thai (a dictionary).
        if (scanner != 0 && "th".equals(locale.getLanguage())) {
            return Prepared.fallback("highlight_locale");
        }
        boolean stored = fieldContext.forceSource == false && fieldType.isStored();
        boolean scoreOrder = numberOfFragments != 0 && options.scoreOrdered();
        // Lucene's ScoreOrderFragmentsBuilder over stored fields; OpenSearch's builders elsewhere,
        // which fix the fragments of an analyzer that breaks the highlighter.
        boolean fix = (stored && scoreOrder) == false && containsBrokenAnalysis(fieldType.indexAnalyzer());

        // FastVectorHighlighter.highlight: the field query, then the thread's highlight filters.
        boolean requireFieldMatch = options.requireFieldMatch();
        FieldQuery fieldQuery = new CustomFieldQuery(fieldContext.query, fieldContext.hitContext.topLevelReader(), true, requireFieldMatch);
        CustomFieldQuery.highlightFilters.set(options.highlightFilter());

        ByteArrayOutputStream out = new ByteArrayOutputStream();
        out.write(REQUEST_VERSION);
        writeString(out, fieldType.name());
        Set<String> matched = options.matchedFields();
        if (matched == null || matched.isEmpty()) {
            writeInt(out, 0);
        } else {
            writeInt(out, matched.size());
            for (String f : matched) {
                writeString(out, f);
            }
        }
        try {
            writeFieldQuery(out, fieldQuery);
        } catch (IllegalAccessException e) {
            return Prepared.fallback("highlight_fvh_reflection");
        }
        if (numberOfFragments == 0) {
            out.write(0);
            writeInt(out, -1);
        } else {
            out.write(1);
            writeInt(out, options.fragmentOffset());
        }
        writeInt(out, options.phraseLimit());
        writeInt(out, numberOfFragments == 0 ? Integer.MAX_VALUE : options.fragmentCharSize());
        writeInt(out, numberOfFragments == 0 ? Integer.MAX_VALUE : numberOfFragments);
        out.write(scoreOrder ? 1 : 0);
        out.write(fix ? 1 : 0);
        out.write(termVectorMultiValue ? 1 : 0);
        out.write(scanner);
        writeInt(out, options.boundaryMaxScan());
        Character[] chars = options.boundaryChars();
        writeInt(out, scanner == 0 ? chars.length : 0);
        if (scanner == 0) {
            for (Character c : chars) {
                out.write(c);
                out.write(c >>> 8);
            }
        }
        writeTags(out, options.preTags());
        writeTags(out, options.postTags());
        out.write(options.encoder().equals("html") ? 1 : 0);
        writeInt(out, options.noMatchSize());
        return new Prepared(out.toByteArray(), handle, stored == false, null);
    }

    /** {@code FragmentBuilderHelper.containsBrokenAnalysis}. */
    private static boolean containsBrokenAnalysis(Analyzer analyzer) {
        if (analyzer instanceof NamedAnalyzer named) {
            analyzer = named.analyzer();
        }
        if (analyzer instanceof AnalyzerComponentsProvider provider) {
            for (TokenFilterFactory filter : provider.getComponents().getTokenFilters()) {
                if (filter.breaksFastVectorHighlighter()) {
                    return true;
                }
            }
        }
        return false;
    }

    @SuppressWarnings("unchecked")
    private static void writeFieldQuery(ByteArrayOutputStream out, FieldQuery query) throws IllegalAccessException {
        out.write(FIELD_MATCH.getBoolean(query) ? 1 : 0);
        Map<String, Object> roots = (Map<String, Object>) ROOT_MAPS.get(query);
        writeInt(out, roots.size());
        for (Map.Entry<String, Object> e : roots.entrySet()) {
            writeKey(out, e.getKey());
            writeNode(out, e.getValue());
        }
        Map<String, Set<String>> termSets = (Map<String, Set<String>>) TERM_SET_MAP.get(query);
        writeInt(out, termSets.size());
        for (Map.Entry<String, Set<String>> e : termSets.entrySet()) {
            writeKey(out, e.getKey());
            writeInt(out, e.getValue().size());
            for (String t : e.getValue()) {
                writeString(out, t);
            }
        }
    }

    @SuppressWarnings("unchecked")
    private static void writeNode(ByteArrayOutputStream out, Object node) throws IllegalAccessException {
        out.write(TERMINAL.getBoolean(node) ? 1 : 0);
        writeInt(out, SLOP.getInt(node));
        writeInt(out, Float.floatToRawIntBits(BOOST.getFloat(node)));
        writeInt(out, NUMBER.getInt(node));
        Map<String, Object> subs = (Map<String, Object>) SUB_MAP.get(node);
        writeInt(out, subs.size());
        for (Map.Entry<String, Object> e : subs.entrySet()) {
            writeString(out, e.getKey());
            writeNode(out, e.getValue());
        }
    }

    private static void writeKey(ByteArrayOutputStream out, String key) {
        if (key == null) {
            out.write(0);
        } else {
            out.write(1);
            writeString(out, key);
        }
    }

    private static void writeTags(ByteArrayOutputStream out, String[] tags) {
        writeInt(out, tags.length);
        for (String t : tags) {
            writeString(out, t);
        }
    }

    private static void writeString(ByteArrayOutputStream out, String s) {
        byte[] b = s.getBytes(StandardCharsets.UTF_8);
        writeInt(out, b.length);
        out.write(b, 0, b.length);
    }

    private static void writeInt(ByteArrayOutputStream out, int v) {
        out.write(v);
        out.write(v >>> 8);
        out.write(v >>> 16);
        out.write(v >>> 24);
    }
}
