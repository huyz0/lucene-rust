/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.highlight;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.search.uhighlight.UnifiedHighlighter.OffsetSource;
import org.apache.lucene.util.CollectionUtil;
import org.lucenerust.opensearch.AnalyzerSpec;
import org.lucenerust.opensearch.NativeBridge;
import org.lucenerust.opensearch.NativeReaders;
import org.lucenerust.opensearch.QueryEncoder;
import org.lucenerust.opensearch.SearchStats;
import org.opensearch.core.common.Strings;
import org.opensearch.core.common.text.Text;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.lucene.search.uhighlight.Snippet;
import org.opensearch.search.fetch.FetchSubPhase;
import org.opensearch.search.fetch.subphase.highlight.FieldHighlightContext;
import org.opensearch.search.fetch.subphase.highlight.HighlightBuilder;
import org.opensearch.search.fetch.subphase.highlight.HighlightField;
import org.opensearch.search.fetch.subphase.highlight.HighlightUtils;
import org.opensearch.search.fetch.subphase.highlight.SearchHighlightContext;
import org.opensearch.search.fetch.subphase.highlight.UnifiedHighlighter;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
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
 * OpenSearch's {@code unified} highlighter run by the Rust engine ({@code jvm_highlight.rs}): the
 * same inputs OpenSearch's {@link UnifiedHighlighter} builds its {@code CustomUnifiedHighlighter}
 * from -- the field's offset source, index analyzer, break iterator, tags, encoder and query --
 * written once per field and search, then each hit's loaded value highlighted natively. The values
 * are loaded, and the snippets filtered and ordered, by OpenSearch's own code, as in its
 * highlighter.
 *
 * <p>A field whose query, analyzer or options the native side does not mirror is highlighted by
 * OpenSearch's highlighter (counted as a fallback, {@code highlight_*}); so is a hit the native call
 * fails on (a native error).
 */
public final class NativeUnifiedHighlighter extends UnifiedHighlighter {
    /** The name it is registered under; {@link HighlightRouting} points {@code unified} fields at it. */
    public static final String NAME = "rust_unified";

    private static final Logger logger = LogManager.getLogger(NativeUnifiedHighlighter.class);

    /** {@code CustomUnifiedHighlighter.MULTIVAL_SEP_CHAR}. */
    private static final char MULTIVAL_SEP_CHAR = (char) 0;

    /** {@code jvm_highlight.rs}'s statuses and request version. */
    private static final byte REQUEST_VERSION = 1;
    private static final int STATUS_SNIPPETS = 0;
    private static final int STATUS_NOOP = 1;
    private static final int STATUS_OFFSET_LIMIT = 2;
    private static final int STATUS_TOO_LONG = 3;
    private static final int STATUS_NEEDS_ANALYZER = 4;

    private final NativeReaders readers;
    private final SearchStats stats;

    public NativeUnifiedHighlighter(NativeReaders readers, SearchStats stats) {
        this.readers = readers;
        this.stats = stats;
    }

    /** One field's native request for one search, or why it has none. */
    private static final class Prepared {
        final byte[] request;
        final long handle;
        String fallback;
        /** Why an analyzer the request needs is not described, for {@code STATUS_NEEDS_ANALYZER}. */
        String undescribed = "highlight_analyzer";
        /** The query matches nothing in the field: every hit's highlight is empty. */
        boolean noop;

        Prepared(byte[] request, long handle, String fallback) {
            this.request = request;
            this.handle = handle;
            this.fallback = fallback;
        }
    }

    @Override
    public HighlightField highlight(FieldHighlightContext fieldContext) throws IOException {
        @SuppressWarnings("unchecked")
        Map<String, Prepared> cache = (Map<String, Prepared>) fieldContext.cache.computeIfAbsent(
            NativeUnifiedHighlighter.class.getName(),
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
        if (prepared.noop) {
            return null;
        }
        FetchSubPhase.HitContext hitContext = fieldContext.hitContext;
        MappedFieldType fieldType = fieldContext.fieldType;
        List<Object> fieldValues = HighlightUtils.loadFieldValues(
            fieldType,
            fieldContext.context.getQueryShardContext(),
            hitContext,
            fieldContext.forceSource
        );
        if (fieldValues.isEmpty()) {
            return null;
        }
        List<Object> converted = new ArrayList<>(fieldValues.size());
        for (Object v : fieldValues) {
            converted.add(convertFieldValue(fieldType, v));
        }
        String value = mergeFieldValues(converted, MULTIVAL_SEP_CHAR);
        byte[][] out = new byte[1][];
        int rc = NativeBridge.highlightUnified(
            prepared.handle,
            prepared.request,
            hitContext.readerContext().ord,
            hitContext.docId(),
            value,
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
        switch (status) {
            case STATUS_SNIPPETS:
                break;
            case STATUS_NOOP:
                prepared.noop = true;
                return null;
            case STATUS_NEEDS_ANALYZER:
                prepared.fallback = prepared.undescribed;
                stats.fallback(prepared.fallback);
                return super.highlight(fieldContext);
            case STATUS_OFFSET_LIMIT:
                throw new IllegalArgumentException(
                    "max_analyzer_offset has exceeded ["
                        + fieldContext.context.getIndexSettings().getHighlightMaxAnalyzedOffset()
                        + "] - maximum allowed to be analyzed for highlighting. "
                        + "This maximum can be set by changing the ["
                        + IndexSettings.MAX_ANALYZED_OFFSET_SETTING.getKey()
                        + "] index level setting. "
                        + "For large texts, indexing with offsets or term vectors is recommended!"
                );
            case STATUS_TOO_LONG:
                throw new IllegalArgumentException(
                    "The length of ["
                        + fieldContext.fieldName
                        + "] field of ["
                        + hitContext.docId()
                        + "] doc of ["
                        + fieldContext.context.getIndexName()
                        + "] index "
                        + "has exceeded ["
                        + fieldContext.context.getIndexSettings().getHighlightMaxAnalyzedOffset()
                        + "] - maximum allowed to be analyzed for highlighting. "
                        + "This maximum can be set by changing the ["
                        + IndexSettings.MAX_ANALYZED_OFFSET_SETTING.getKey()
                        + "] index level setting. "
                        + "For large texts, indexing with offsets or term vectors is recommended!"
                );
            default:
                throw new IllegalStateException("lucene-rust: highlight status " + status);
        }
        int count = b.getInt();
        List<Snippet> snippets = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            float score = b.getFloat();
            boolean highlighted = b.get() != 0;
            int len = b.getInt();
            char[] chars = new char[len];
            b.asCharBuffer().get(chars);
            b.position(b.position() + len * Character.BYTES);
            String text = new String(chars);
            // UnifiedHighlighter.highlight: only snippets with text
            if (Strings.hasText(text)) {
                snippets.add(new Snippet(text, score, highlighted));
            }
        }
        if (snippets.isEmpty()) {
            return null;
        }
        if (fieldContext.field.fieldOptions().scoreOrdered()) {
            CollectionUtil.introSort(snippets, (o1, o2) -> Double.compare(o2.getScore(), o1.getScore()));
        }
        String[] fragments = new String[snippets.size()];
        for (int i = 0; i < fragments.length; i++) {
            fragments[i] = snippets.get(i).getText();
        }
        return new HighlightField(fieldContext.fieldName, Text.convertFromStringArray(fragments));
    }

    /** {@code UnifiedHighlighter.buildHighlighter}'s inputs as a native request, or a fallback reason. */
    private Prepared prepare(FieldHighlightContext fieldContext) {
        long handle = readers.peek(fieldContext.hitContext.topLevelReader());
        if (handle == 0) {
            return new Prepared(null, 0, "highlight_reader");
        }
        if (fieldContext.context.getQueryShardContext().resolveDerivedFieldType(fieldContext.fieldName) != null) {
            return new Prepared(null, 0, "highlight_derived_field");
        }
        SearchHighlightContext.FieldOptions options = fieldContext.field.fieldOptions();
        MappedFieldType fieldType = fieldContext.fieldType;
        OffsetSource offsetSource = getOffsetSource(fieldType);
        // The break iterator: one passage per value, or OpenSearch's sentence/word iterators.
        int numberOfFragments = options.numberOfFragments();
        byte breaks;
        int maxLen = 0;
        int maxPassages;
        Locale locale = options.boundaryScannerLocale() != null ? options.boundaryScannerLocale() : Locale.ROOT;
        // The JDK's break rules are the same for every locale but Thai (a dictionary); the
        // no-match summary cuts at a word boundary of that locale on every path.
        if ("th".equals(locale.getLanguage())) {
            return new Prepared(null, 0, "highlight_locale");
        }
        if (numberOfFragments == 0 || fieldType.getTextSearchInfo().isTokenized() == false) {
            breaks = 0;
            maxPassages = numberOfFragments == 0 ? Integer.MAX_VALUE - 1 : numberOfFragments;
        } else {
            HighlightBuilder.BoundaryScannerType type = options.boundaryScannerType() != null
                ? options.boundaryScannerType()
                : HighlightBuilder.BoundaryScannerType.SENTENCE;
            switch (type) {
                case SENTENCE:
                    maxLen = options.fragmentCharSize();
                    breaks = (byte) (maxLen > 0 ? 2 : 1);
                    break;
                case WORD:
                    breaks = 3;
                    break;
                default:
                    // OpenSearch refuses it; its highlighter says so.
                    return new Prepared(null, 0, "highlight_boundary_scanner");
            }
            maxPassages = numberOfFragments;
        }
        Set<String> matchedFields = options.matchedFields();
        boolean masked = matchedFields != null && matchedFields.isEmpty() == false;
        // The query can ask for re-analysis on any offset source (a multi-term query on postings),
        // and each masked field is re-analyzed with its own analyzer: every analyzer that is
        // described goes with the request, and a needed one that is not is answered with
        // STATUS_NEEDS_ANALYZER.
        Analyzer analyzer = getAnalyzer(fieldContext.context.mapperService().documentMapper());
        List<String> analyzed = new ArrayList<>();
        analyzed.add(fieldContext.fieldName);
        if (masked) {
            for (String f : matchedFields) {
                if (f.equals(fieldContext.fieldName) == false) {
                    analyzed.add(f);
                }
            }
        }
        List<String> specFields = new ArrayList<>();
        List<byte[]> specs = new ArrayList<>();
        String undescribed = null;
        for (String f : analyzed) {
            AnalyzerSpec.Spec described = AnalyzerSpec.cached(analyzer, f);
            if (described.ok()) {
                specFields.add(f);
                specs.add(described.blob());
            } else if (undescribed == null) {
                undescribed = "highlight_" + described.reason();
            }
        }
        if (offsetSource == OffsetSource.ANALYSIS && specFields.contains(fieldContext.fieldName) == false) {
            return new Prepared(null, 0, undescribed);
        }
        QueryEncoder.Encoded query = QueryEncoder.encodeForHighlight(fieldContext.query);
        if (query.blob() == null) {
            return new Prepared(null, 0, query.fallbackReason());
        }
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        out.write(REQUEST_VERSION);
        writeBytes(out, fieldContext.fieldName.getBytes(StandardCharsets.UTF_8));
        writeBytes(out, query.blob());
        out.write(switch (offsetSource) {
            case ANALYSIS -> 0;
            case POSTINGS -> 1;
            case POSTINGS_WITH_TERM_VECTORS -> 2;
            case TERM_VECTORS -> 3;
            default -> throw new IllegalStateException("offset source " + offsetSource);
        });
        writeInt(out, specs.size());
        for (int i = 0; i < specs.size(); i++) {
            writeBytes(out, specFields.get(i).getBytes(StandardCharsets.UTF_8));
            writeBytes(out, specs.get(i));
        }
        Integer limit = options.maxAnalyzerOffset();
        writeInt(out, limit == null ? -1 : limit);
        out.write(breaks);
        writeInt(out, maxLen);
        writeInt(out, maxPassages);
        writeInt(out, options.noMatchSize());
        writeBytes(out, options.preTags()[0].getBytes(StandardCharsets.UTF_8));
        writeBytes(out, options.postTags()[0].getBytes(StandardCharsets.UTF_8));
        out.write(options.encoder().equals("html") ? 1 : 0);
        out.write(options.requireFieldMatch() ? 1 : 0);
        writeInt(out, masked ? matchedFields.size() : 0);
        if (masked) {
            for (String f : matchedFields) {
                writeBytes(out, f.getBytes(StandardCharsets.UTF_8));
            }
        }
        writeInt(out, fieldContext.context.getIndexSettings().getHighlightMaxAnalyzedOffset());
        Prepared prepared = new Prepared(out.toByteArray(), handle, null);
        if (undescribed != null) {
            prepared.undescribed = undescribed;
        }
        return prepared;
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
