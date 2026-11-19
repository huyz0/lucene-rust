/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.engine;

import org.apache.lucene.document.Field;
import org.apache.lucene.document.IntPoint;
import org.apache.lucene.document.LongPoint;
import org.apache.lucene.document.FloatPoint;
import org.apache.lucene.document.DoublePoint;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.SortedNumericDocValuesField;
import org.apache.lucene.document.SortedSetDocValuesField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.IndexableField;
import org.apache.lucene.index.IndexableFieldType;
import org.apache.lucene.util.BytesRef;
import org.opensearch.common.settings.Setting;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.common.xcontent.XContentHelper;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.core.common.bytes.BytesReference;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.ToXContent;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.mapper.DocumentMapper;
import org.opensearch.index.mapper.FieldMapper;
import org.opensearch.index.mapper.IdFieldMapper;
import org.opensearch.index.mapper.Mapper;
import org.opensearch.index.mapper.MetadataFieldMapper;
import org.opensearch.index.mapper.ParseContext;
import org.opensearch.index.mapper.ParsedDocument;
import org.opensearch.index.mapper.SeqNoFieldMapper;
import org.opensearch.index.mapper.SourceFieldMapper;
import org.opensearch.index.mapper.SourceToParse;
import org.opensearch.index.mapper.Uid;
import org.opensearch.index.mapper.VersionFieldMapper;
import org.lucenerust.opensearch.NativeWriter;
import org.lucenerust.opensearch.NativeBridge;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Reads a document's {@code _source} in Rust against the mapping, instead of in the JVM's document
 * mapper, when the mapping is one the Rust reader reproduces exactly and the document is plain: see
 * {@code ingest.rs}. The mapper's metadata fields ({@code _id}, {@code _seq_no}, {@code _version},
 * {@code _source}) are built here as the mapper builds them; the mapped fields come back as a
 * fragment {@link DocumentEncoder} splices into the operation. Whatever Rust refuses -- an unmapped
 * field, a value the fast path does not reproduce, a repeated field -- goes to the document mapper
 * unchanged, so errors, dynamic mapping updates and every unusual document are the mapper's.
 */
public final class NativeIngest {
    /**
     * {@code index.engine.accelerated_source_parsing}: read a document's {@code _source} in the library
     * where the mapping allows, instead of in OpenSearch's document mapper. On by default: a document
     * the library does not reproduce exactly is the mapper's.
     */
    public static final Setting<Boolean> SOURCE_PARSING = Setting.boolSetting(
        "index.engine.accelerated_source_parsing",
        true,
        Setting.Property.IndexScope
    );

    /** Runs the document mapper too and compares what each produces; for tests. */
    static final boolean VERIFY = Boolean.getBoolean("accelerated.ingest.verify");

    public static final AtomicLong HANDLED = new AtomicLong();
    public static final AtomicLong REFUSED = new AtomicLong();
    static final AtomicLong MISMATCHES = new AtomicLong();

    private static final int KEYWORD = 0, TEXT = 1, INTEGER = 2, LONG = 3, FLOAT = 4, DOUBLE = 5, DATE = 6, OBJECT = 7;

    /** Documents the accelerated reader took from the document mapper, and the ones it left to it. */
    public static void stats(org.opensearch.core.xcontent.XContentBuilder b) throws IOException {
        b.startObject("source_parsing").field("accelerated", HANDLED.get()).field("fallback", REFUSED.get()).endObject();
    }

    private static final int FRAGMENT_HEADER = 32;

    /** Where a thread's reads land before the fragment is cut to size. */
    private static final ThreadLocal<byte[]> ROOM = ThreadLocal.withInitial(() -> new byte[8192]);

    private static int header(byte[] fragment, int index) {
        return java.nio.ByteBuffer.wrap(fragment).order(java.nio.ByteOrder.LITTLE_ENDIAN).getInt(index * 4);
    }

    /** Entries the fragment of {@code doc} holds in {@code section} (stored, inverted, doc values, points); 0 without a fragment. */
    static int entries(Document doc, int section) {
        return doc == null ? 0 : header(doc.fragment, section);
    }

    /** Appends the fragment's bytes for {@code section} to {@code b}; returns its entry count. */
    static int splice(Document doc, Blob b, int section) throws IOException {
        if (doc == null) {
            return 0;
        }
        int at = FRAGMENT_HEADER;
        for (int i = 0; i < section; i++) {
            at += header(doc.fragment, 4 + i);
        }
        b.raw(doc.fragment, at, header(doc.fragment, 4 + section));
        b.analyzesNatively |= section == 1 && doc.analyzesText;
        return header(doc.fragment, section);
    }

    /** A mapped document: the metadata fields as the mapper builds them, and the fragment for the rest. */
    static final class Document extends ParseContext.Document {
        final byte[] fragment;
        final int length;
        final boolean analyzesText;

        Document(byte[] fragment, int length, boolean analyzesText) {
            this.fragment = fragment;
            this.length = length;
            this.analyzesText = analyzesText;
        }
    }

    private record Compiled(DocumentMapper mapper, int plan, boolean text) {}

    private static final Compiled NONE = new Compiled(null, -1, false);

    private final FieldRegistry registry;
    private final DocumentEncoder encoder;
    private final IndexSettings settings;
    private volatile Compiled compiled = NONE;

    NativeIngest(FieldRegistry registry, DocumentEncoder encoder, IndexSettings settings) {
        this.registry = registry;
        this.encoder = encoder;
        this.settings = settings;
    }

    /** The mapped document for {@code source}, or null when the document mapper has to parse it. */
    ParsedDocument parse(DocumentMapper mapper, SourceToParse source) throws IOException {
        Compiled c = compiled;
        if (c.mapper != mapper) {
            c = compile(mapper);
            compiled = c;
        }
        if (c.plan < 0
            || source.routing() != null
            || source.getMediaType() != XContentType.JSON
            || source.extraFieldValues().isEmpty() == false) {
            return null;
        }
        BytesRef src = source.source().toBytesRef();
        byte[] room = ROOM.get();
        int n = NativeWriter.ingestParse(c.plan, src.bytes, src.offset, src.length, room);
        if (n > room.length) {
            room = new byte[n + (n >> 1)];
            ROOM.set(room);
            n = NativeWriter.ingestParse(c.plan, src.bytes, src.offset, src.length, room);
        }
        if (n <= 0) {
            REFUSED.incrementAndGet();
            return null;
        }
        byte[] fragment = java.util.Arrays.copyOf(room, n);
        HANDLED.incrementAndGet();
        Document root = new Document(fragment, n, c.text);
        SeqNoFieldMapper.SequenceIDFields seqID = SeqNoFieldMapper.SequenceIDFields.emptySeqID();
        Field version = new NumericDocValuesField(VersionFieldMapper.NAME, -1L);
        root.add(new Field(IdFieldMapper.NAME, Uid.encodeId(source.id()), IdFieldMapper.Defaults.FIELD_TYPE));
        root.add(seqID.seqNo);
        root.add(seqID.seqNoDocValue);
        root.add(seqID.primaryTerm);
        root.add(version);
        BytesRef ref = source.source().toBytesRef();
        root.add(new StoredField(SourceFieldMapper.NAME, ref.bytes, ref.offset, ref.length));
        ParsedDocument parsed = new ParsedDocument(version, seqID, source.id(), null, List.of(root), source.source(), source.getMediaType(), null);
        if (VERIFY) {
            verify(mapper, source, parsed);
        }
        return parsed;
    }

    private void verify(DocumentMapper mapper, SourceToParse source, ParsedDocument nativeDoc) throws IOException {
        ParsedDocument javaDoc;
        try {
            javaDoc = mapper.parse(source);
        } catch (RuntimeException e) {
            MISMATCHES.incrementAndGet();
            System.err.println("accelerated source parsing differs from the document mapper for " + source.source().utf8ToString() + ": the mapper refused it: " + e.getMessage());
            return;
        }
        byte[] expected = encoder.add(javaDoc.docs()).toArray();
        byte[] actual = encoder.add(nativeDoc.docs()).toArray();
        String difference = NativeWriter.ingestVerify(expected, actual);
        if (difference != null) {
            MISMATCHES.incrementAndGet();
            System.err.println("accelerated source parsing differs from the document mapper for " + source.source().utf8ToString() + ": " + difference);
        }
    }

    // ---- compiling a mapping ----

    private Compiled compile(DocumentMapper mapper) {
        try {
            boolean[] text = new boolean[1];
            int plan = plan(mapper, text);
            return plan < 0 ? new Compiled(mapper, -1, false) : new Compiled(mapper, plan, text[0]);
        } catch (Exception | LinkageError e) {
            return new Compiled(mapper, -1, false);
        }
    }

    private static final Set<String> METADATA = Set.of(
        "_id", "_routing", "_seq_no", "_version", "_source", "_index", "_field_names", "_ignored", "_doc_count", "_nested_path", "_feature"
    );

    private static final Set<String> ROOT_KEYS = Set.of("properties", "dynamic", "_meta", "date_detection", "numeric_detection");

    private int plan(DocumentMapper mapper, boolean[] text) throws IOException {
        if (settings.isDerivedSourceEnabled() || settings.isPluggableDataFormatEnabled()) {
            return -1;
        }
        if (mapper.routingFieldMapper().required()
            || mapper.sourceMapper().enabled() == false
            || mapper.sourceMapper().isComplete() == false
            || mapper.hasNestedObjects()) {
            return -1;
        }
        Map<String, Object> root = XContentHelper.convertToMap(mapper.mappingSource().compressedReference(), false, MediaTypeRegistry.JSON).v2();
        if (root.size() == 1 && root.values().iterator().next() instanceof Map<?, ?> inner && root.containsKey(mapper.type())) {
            @SuppressWarnings("unchecked")
            Map<String, Object> typed = (Map<String, Object>) inner;
            root = typed;
        }
        if (ROOT_KEYS.containsAll(root.keySet()) == false) {
            return -1;
        }
        List<String[]> specs = new ArrayList<>();
        Set<String> seen = new HashSet<>();
        List<FieldMapper> fields = new ArrayList<>();
        for (Mapper m : mapper.mappers()) {
            if (m instanceof MetadataFieldMapper) {
                if (m instanceof org.opensearch.index.mapper.DataStreamFieldMapper ds) {
                    if (dataStreamEnabled(ds)) {
                        return -1;
                    }
                    continue;
                }
                if (METADATA.contains(m.name()) == false) {
                    return -1;
                }
                continue;
            }
            if (m instanceof FieldMapper == false) {
                return -1;
            }
            FieldMapper fm = (FieldMapper) m;
            fields.add(fm);
            String name = fm.name();
            if (seen.add(name) == false) {
                return -1;
            }
            specs.add(new String[] { name });
        }
        // The mapping's `object` fields, plain ones only; every leaf of it is one of the field mappers.
        List<String> objects = new ArrayList<>();
        Set<String> leaves = new HashSet<>();
        if (objects(root.get("properties"), "", objects, leaves, 0) == false || leaves.equals(seen) == false) {
            return -1;
        }
        if (specs.isEmpty() || specs.size() + objects.size() > 128) {
            return -1;
        }
        Blob plan = new Blob(256).i32(specs.size() + objects.size());
        for (FieldMapper fm : fields) {
            Map<String, Object> params = params(fm);
            if (params == null || writeField(plan, fm.name(), params, text) == false) {
                return -1;
            }
        }
        for (String object : objects) {
            plan.string(object).u8(OBJECT).u8(0).i32(0).i32(0).i32(0).u8(0).u8(0);
        }
        int[] id = new int[1];
        if (NativeWriter.ingestRegister(plan.toArray(), id) != NativeBridge.OK) {
            return -1;
        }
        return id[0];
    }

    private static final Set<String> OBJECT_KEYS = Set.of("type", "properties", "dynamic");

    /**
     * Collects the paths of the plain {@code object} fields under {@code properties} into {@code objects}
     * and the paths of everything else into {@code leaves}; false for an object with any other setting
     * (nested, disabled, subobjects, ...) or one too deep.
     */
    private static boolean objects(Object properties, String prefix, List<String> objects, Set<String> leaves, int depth) {
        if (properties == null) {
            return true;
        }
        if (properties instanceof Map<?, ?> == false || depth > 20) {
            return false;
        }
        for (Map.Entry<?, ?> e : ((Map<?, ?>) properties).entrySet()) {
            if (e.getKey() instanceof String == false || e.getValue() instanceof Map<?, ?> == false) {
                return false;
            }
            String name = (String) e.getKey();
            Map<?, ?> field = (Map<?, ?>) e.getValue();
            String path = prefix + name;
            Object type = field.get("type");
            if (type == null || "object".equals(type)) {
                if (OBJECT_KEYS.containsAll(field.keySet()) == false) {
                    return false;
                }
                objects.add(path);
                if (objects(field.get("properties"), path + ".", objects, leaves, depth + 1) == false) {
                    return false;
                }
            } else {
                leaves.add(path);
            }
        }
        return true;
    }

    private static boolean dataStreamEnabled(org.opensearch.index.mapper.DataStreamFieldMapper ds) throws IOException {
        try {
            java.lang.reflect.Field f = org.opensearch.index.mapper.DataStreamFieldMapper.class.getDeclaredField("enabled");
            f.setAccessible(true);
            return f.getBoolean(ds);
        } catch (ReflectiveOperationException e) {
            return true;
        }
    }

    private static Map<String, Object> params(FieldMapper fm) throws IOException {
        try (XContentBuilder b = XContentFactory.jsonBuilder()) {
            b.startObject();
            fm.toXContent(b, new ToXContent.MapParams(Map.of("include_defaults", "true")));
            b.endObject();
            Map<String, Object> top = XContentHelper.convertToMap(BytesReference.bytes(b), false, MediaTypeRegistry.JSON).v2();
            Object inner = top.get(fm.simpleName());
            if (top.size() == 1 && inner instanceof Map<?, ?>) {
                @SuppressWarnings("unchecked")
                Map<String, Object> m = (Map<String, Object>) inner;
                return m;
            }
            return null;
        }
    }

    private static boolean is(Map<String, Object> p, String key, Object value) {
        return value.equals(p.get(key));
    }

    private static boolean only(Map<String, Object> p, Set<String> keys) {
        return keys.containsAll(p.keySet());
    }

    private static final Set<String> COMMON = Set.of("type", "index", "doc_values", "store", "meta", "null_value", "boost");
    private static final Set<String> NUMBER = Set.of("type", "index", "doc_values", "store", "skip_list", "ignore_malformed", "coerce", "null_value", "meta");
    private static final Set<String> DATE_KEYS = Set.of(
        "type", "index", "doc_values", "store", "skip_list", "format", "print_format", "locale", "null_value", "ignore_malformed", "boost", "meta"
    );
    private static final Set<String> KEYWORD_KEYS = Set.of(
        "type", "index", "doc_values", "store", "null_value", "eager_global_ordinals", "ignore_above", "index_options", "norms", "similarity",
        "use_similarity", "normalizer", "split_queries_on_whitespace", "boost", "meta"
    );
    private static final Set<String> TEXT_KEYS = Set.of(
        "type", "boost", "index", "store", "meta", "index_options", "term_vector", "norms", "analyzer", "search_analyzer",
        "search_quote_analyzer", "similarity", "eager_global_ordinals", "position_increment_gap", "fielddata",
        "fielddata_frequency_filter", "index_prefixes", "index_phrases"
    );

    private static boolean metaEmpty(Map<String, Object> p) {
        return p.get("meta") instanceof Map<?, ?> m && m.isEmpty();
    }

    private static boolean unsupportedKeyword(Map<String, Object> p) {
        return only(p, KEYWORD_KEYS) == false
            || is(p, "index", true) == false
            || is(p, "doc_values", true) == false
            || is(p, "store", false) == false
            || is(p, "eager_global_ordinals", false) == false
            || is(p, "index_options", "docs") == false
            || is(p, "norms", false) == false
            || p.get("similarity") != null
            || is(p, "use_similarity", false) == false
            || is(p, "normalizer", "default") == false
            || is(p, "split_queries_on_whitespace", false) == false
            || p.get("ignore_above") instanceof Number == false;
    }

    private static boolean unsupportedText(Map<String, Object> p) {
        return only(p, TEXT_KEYS) == false
            || is(p, "index", true) == false
            || is(p, "store", false) == false
            || is(p, "index_options", "positions") == false
            || is(p, "term_vector", "no") == false
            || is(p, "norms", true) == false
            || is(p, "analyzer", "default") == false
            || is(p, "search_analyzer", "default") == false
            || is(p, "search_quote_analyzer", "default") == false
            || p.get("similarity") != null
            || is(p, "eager_global_ordinals", false) == false
            || ((Number) p.get("position_increment_gap")).intValue() != -1
            || is(p, "fielddata", false) == false
            || p.get("index_prefixes") != null
            || is(p, "index_phrases", false) == false;
    }

    private static boolean unsupportedNumber(Map<String, Object> p) {
        return only(p, NUMBER) == false
            || is(p, "index", true) == false
            || is(p, "doc_values", true) == false
            || is(p, "store", false) == false
            || is(p, "skip_list", false) == false
            || is(p, "ignore_malformed", false) == false
            || is(p, "coerce", true) == false;
    }

    private static boolean unsupportedDate(Map<String, Object> p) {
        return only(p, DATE_KEYS) == false
            || is(p, "index", true) == false
            || is(p, "doc_values", true) == false
            || is(p, "store", false) == false
            || is(p, "skip_list", false) == false
            || is(p, "format", "strict_date_optional_time||epoch_millis") == false
            || is(p, "locale", "") == false
            || is(p, "ignore_malformed", false) == false;
    }

    /** Writes one field of the plan (ingest.rs's {@code Plan::decode}); false when the field is not one the fast path reproduces. */
    private boolean writeField(Blob plan, String name, Map<String, Object> p, boolean[] text) throws IOException {
        Object type = p.get("type");
        if (type instanceof String == false || metaEmpty(p) == false || p.get("null_value") != null || is(p, "boost", 1.0) == false && p.containsKey("boost")) {
            return false;
        }
        FieldSchema schema = new FieldSchema(name);
        int kind;
        int flags;
        int ignoreAbove = 0;
        int analyzer = 0;
        int options = 0;
        int norm = 0;
        switch ((String) type) {
            case "keyword" -> {
                if (unsupportedKeyword(p)) {
                    return false;
                }
                kind = KEYWORD;
                flags = 3;
                ignoreAbove = ((Number) p.get("ignore_above")).intValue();
                schema.absorb(IdFieldMapper.Defaults.FIELD_TYPE);
                schema.absorb(new SortedSetDocValuesField(name, new BytesRef()).fieldType());
            }
            case "text" -> {
                if (unsupportedText(p)) {
                    return false;
                }
                kind = TEXT;
                flags = 1;
                schema.absorb(TextField.TYPE_NOT_STORED);
                DocumentEncoder.NativeField na = encoder.nativeAnalysis(name);
                if (na == null) {
                    return false;
                }
                analyzer = na.id();
                options = schema.indexOptions.ordinal();
                norm = na.discountOverlaps() ? 1 : 2;
                text[0] = true;
            }
            case "integer", "long", "float", "double" -> {
                if (unsupportedNumber(p)) {
                    return false;
                }
                flags = 3;
                switch ((String) type) {
                    case "integer" -> {
                        kind = INTEGER;
                        schema.absorb(new IntPoint(name, 0).fieldType());
                        schema.absorb(new SortedNumericDocValuesField(name, 0).fieldType());
                    }
                    case "long" -> {
                        kind = LONG;
                        schema.absorb(new LongPoint(name, 0).fieldType());
                        schema.absorb(new SortedNumericDocValuesField(name, 0).fieldType());
                    }
                    case "float" -> {
                        kind = FLOAT;
                        schema.absorb(new FloatPoint(name, 0).fieldType());
                        schema.absorb(new SortedNumericDocValuesField(name, 0).fieldType());
                    }
                    default -> {
                        kind = DOUBLE;
                        schema.absorb(new DoublePoint(name, 0).fieldType());
                        schema.absorb(new SortedNumericDocValuesField(name, 0).fieldType());
                    }
                }
            }
            case "date" -> {
                if (unsupportedDate(p)) {
                    return false;
                }
                kind = DATE;
                flags = 3;
                schema.absorb(new LongPoint(name, 0).fieldType());
                boolean skip = settings.getIndexVersionCreated().onOrAfter(org.opensearch.Version.V_3_3_0)
                    && (name.equals("@timestamp") || settings.getIndexSortConfig().hasPrimarySortOnField(name));
                schema.absorb((skip ? SortedNumericDocValuesField.indexedField(name, 0) : new SortedNumericDocValuesField(name, 0)).fieldType());
            }
            default -> {
                return false;
            }
        }
        schema.checkSupported();
        FieldRegistry.Entry entry = registry.resolve(schema);
        plan.string(name).u8(kind).u8(flags).i32(entry.number()).i32(ignoreAbove).i32(analyzer).u8(options).u8(norm);
        return true;
    }
}
