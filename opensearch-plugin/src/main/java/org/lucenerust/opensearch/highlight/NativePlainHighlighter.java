/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.highlight;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.util.CollectionUtil;
import org.lucenerust.opensearch.AnalyzerSpec;
import org.lucenerust.opensearch.NativeBridge;
import org.lucenerust.opensearch.QueryEncoder;
import org.lucenerust.opensearch.SearchStats;
import org.opensearch.core.common.text.Text;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.search.fetch.FetchSubPhase;
import org.opensearch.search.fetch.subphase.highlight.FieldHighlightContext;
import org.opensearch.search.fetch.subphase.highlight.HighlightField;
import org.opensearch.search.fetch.subphase.highlight.HighlightUtils;
import org.opensearch.search.fetch.subphase.highlight.PlainHighlighter;
import org.opensearch.search.fetch.subphase.highlight.SearchHighlightContext;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * OpenSearch's {@code plain} highlighter run by the Rust engine ({@code jvm_highlight_plain.rs}):
 * the same {@code Highlighter} OpenSearch's {@link PlainHighlighter} builds -- its {@code
 * CustomQueryScorer}, fragmenter, tags and encoder -- described once per field and search, then
 * each hit's values re-analyzed and highlighted natively. The values are loaded, and the fragments
 * of every value ordered and picked, by OpenSearch's own code, as in its highlighter.
 *
 * <p>A field whose query, analyzer or options the native side does not mirror is highlighted by
 * OpenSearch's highlighter (counted as a fallback, {@code highlight_*}); so is a hit OpenSearch's
 * highlighter refuses (a value past {@code max_analyzed_offset}), which then fails as it does in
 * OpenSearch, a hit with a value holding a lone surrogate, and a hit the native call fails on (a
 * native error).
 */
public final class NativePlainHighlighter extends PlainHighlighter {
    /** The name it is registered under; {@link HighlightRouting} points {@code plain} fields at it. */
    public static final String NAME = "rust_plain";

    private static final Logger logger = LogManager.getLogger(NativePlainHighlighter.class);

    /** {@code jvm_highlight_plain.rs}'s request version and status. */
    private static final byte REQUEST_VERSION = 1;
    private static final int STATUS_FRAGMENTS = 0;

    private final SearchStats stats;

    public NativePlainHighlighter(SearchStats stats) {
        this.stats = stats;
    }

    /** One field's native request for one search, or why it has none. */
    private static final class Prepared {
        final byte[] request;
        String fallback;
        /**
         * The span fragmenter's {@code waitForPos}, which OpenSearch's highlighter, kept for the
         * field's whole fetch phase, carries from one value (and hit) to the next.
         */
        int waitForPos = -1;

        Prepared(byte[] request, String fallback) {
            this.request = request;
            this.fallback = fallback;
        }
    }

    /** A fragment that scored ({@code TextFragment}'s score and text). */
    private record Fragment(float score, String text) {}

    @Override
    public HighlightField highlight(FieldHighlightContext fieldContext) throws IOException {
        @SuppressWarnings("unchecked")
        Map<String, Prepared> cache = (Map<String, Prepared>) fieldContext.cache.computeIfAbsent(
            NativePlainHighlighter.class.getName(),
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
        SearchHighlightContext.FieldOptions options = fieldContext.field.fieldOptions();
        FetchSubPhase.HitContext hitContext = fieldContext.hitContext;
        MappedFieldType fieldType = fieldContext.fieldType;
        int maxAnalyzedOffset = fieldContext.context.getIndexSettings().getHighlightMaxAnalyzedOffset();
        Integer fieldMaxAnalyzedOffset = options.maxAnalyzerOffset();
        List<Object> textsToHighlight = HighlightUtils.loadFieldValues(
            fieldType,
            fieldContext.context.getQueryShardContext(),
            hitContext,
            fieldContext.forceSource
        );
        List<String> values = new ArrayList<>(textsToHighlight.size());
        for (Object textToHighlight : textsToHighlight) {
            String text = NativeUnifiedHighlighter.convert(fieldType, textToHighlight);
            int textLength = text.length();
            if (fieldMaxAnalyzedOffset != null && textLength > fieldMaxAnalyzedOffset) {
                text = text.substring(0, fieldMaxAnalyzedOffset);
            } else if (textLength > maxAnalyzedOffset) {
                // OpenSearch's highlighter refuses the hit, and says so.
                return super.highlight(fieldContext);
            }
            if (hasLoneSurrogate(text)) {
                // Java's analysis reads a lone surrogate (often one a max_analyzer_offset cut
                // leaves) in ways the native text, which has none, cannot mirror.
                stats.fallback("highlight_lone_surrogate");
                return super.highlight(fieldContext);
            }
            values.add(text);
        }
        int noMatchSize = options.noMatchSize();
        String summary = noMatchSize > 0 && textsToHighlight.isEmpty() == false ? textsToHighlight.get(0).toString() : null;
        byte[][] out = new byte[1][];
        int rc = NativeBridge.highlightPlain(prepared.request, values, noMatchSize, summary, prepared.waitForPos, out);
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
        if (status != STATUS_FRAGMENTS) {
            throw new IllegalStateException("lucene-rust: plain highlight status " + status);
        }
        int count = b.getInt();
        ArrayList<Fragment> fragsList = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            float score = b.getFloat();
            int len = b.getInt();
            char[] chars = new char[len];
            b.asCharBuffer().get(chars);
            b.position(b.position() + len * Character.BYTES);
            fragsList.add(new Fragment(score, new String(chars)));
        }
        int summaryEnd = b.getInt();
        prepared.waitForPos = b.getInt();
        // PlainHighlighter.highlight from here on
        if (options.scoreOrdered()) {
            CollectionUtil.introSort(fragsList, (o1, o2) -> Math.round(o2.score() - o1.score()));
        }
        int numberOfFragments = options.numberOfFragments() == 0 ? 1 : options.numberOfFragments();
        String[] fragments;
        if (options.numberOfFragments() == 0 && textsToHighlight.size() > 1 && fragsList.size() > 0) {
            fragments = new String[fragsList.size()];
            for (int i = 0; i < fragsList.size(); i++) {
                fragments[i] = fragsList.get(i).text();
            }
        } else {
            numberOfFragments = Math.min(fragsList.size(), numberOfFragments);
            fragments = new String[numberOfFragments];
            for (int i = 0; i < fragments.length; i++) {
                fragments[i] = fragsList.get(i).text();
            }
        }
        if (fragments.length > 0) {
            return new HighlightField(fieldContext.fieldName, Text.convertFromStringArray(fragments));
        }
        if (summary != null && summaryEnd > 0) {
            return new HighlightField(fieldContext.fieldName, new Text[] { new Text(summary.substring(0, summaryEnd)) });
        }
        return null;
    }

    /** Whether {@code text} holds a surrogate that is not half of a pair. */
    static boolean hasLoneSurrogate(String text) {
        for (int i = 0, n = text.length(); i < n; i++) {
            char c = text.charAt(i);
            if (Character.isSurrogate(c)) {
                if (Character.isHighSurrogate(c) && i + 1 < n && Character.isLowSurrogate(text.charAt(i + 1))) {
                    i++;
                } else {
                    return true;
                }
            }
        }
        return false;
    }

    /** {@code PlainHighlighter.highlight}'s highlighter as a native request, or a fallback reason. */
    private Prepared prepare(FieldHighlightContext fieldContext) {
        if (fieldContext.context.getQueryShardContext().resolveDerivedFieldType(fieldContext.fieldName) != null) {
            return new Prepared(null, "highlight_derived_field");
        }
        SearchHighlightContext.FieldOptions options = fieldContext.field.fieldOptions();
        Integer fieldMaxAnalyzedOffset = options.maxAnalyzerOffset();
        if (fieldMaxAnalyzedOffset != null
            && fieldMaxAnalyzedOffset > fieldContext.context.getIndexSettings().getHighlightMaxAnalyzedOffset()) {
            // OpenSearch's highlighter refuses it, and says so.
            return new Prepared(null, "highlight_max_analyzer_offset");
        }
        byte fragmenter;
        if (options.numberOfFragments() == 0) {
            fragmenter = 0;
        } else if (options.fragmenter() == null || "span".equals(options.fragmenter())) {
            fragmenter = 1;
        } else if ("simple".equals(options.fragmenter())) {
            fragmenter = 2;
        } else {
            // OpenSearch's highlighter refuses it, and says so.
            return new Prepared(null, "highlight_fragmenter");
        }
        MappedFieldType fieldType = fieldContext.fieldType;
        String field = fieldType.name();
        Analyzer analyzer = fieldContext.context.mapperService().documentMapper().mappers().indexAnalyzer();
        AnalyzerSpec.Spec spec = AnalyzerSpec.cached(analyzer, field);
        if (spec.ok() == false) {
            return new Prepared(null, "highlight_" + spec.reason());
        }
        QueryEncoder.Encoded query = QueryEncoder.encodeForPlainHighlight(fieldContext.query);
        if (query.blob() == null) {
            return new Prepared(null, query.fallbackReason());
        }
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        out.write(REQUEST_VERSION);
        writeBytes(out, field.getBytes(StandardCharsets.UTF_8));
        writeBytes(out, query.blob());
        writeBytes(out, spec.blob());
        out.write(options.requireFieldMatch() ? 1 : 0);
        out.write(fragmenter);
        writeInt(out, options.fragmentCharSize());
        writeInt(out, options.numberOfFragments());
        writeBytes(out, options.preTags()[0].getBytes(StandardCharsets.UTF_8));
        writeBytes(out, options.postTags()[0].getBytes(StandardCharsets.UTF_8));
        out.write(options.encoder().equals("html") ? 1 : 0);
        return new Prepared(out.toByteArray(), null);
    }

    private static void writeInt(ByteArrayOutputStream out, int v) {
        out.write(v);
        out.write(v >>> 8);
        out.write(v >>> 16);
        out.write(v >>> 24);
    }

    private static void writeBytes(ByteArrayOutputStream out, byte[] b) {
        writeInt(out, b.length);
        out.write(b, 0, b.length);
    }
}
