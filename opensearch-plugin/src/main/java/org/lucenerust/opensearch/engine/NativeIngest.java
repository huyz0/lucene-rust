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
import org.opensearch.index.mapper.NestedPathFieldMapper;
import org.opensearch.index.mapper.ObjectMapper;
import org.opensearch.index.mapper.MapperService;
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
import java.util.HashMap;
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

    private static final int KEYWORD = 0, TEXT = 1, INTEGER = 2, LONG = 3, FLOAT = 4, DOUBLE = 5, DATE = 6, OBJECT = 7, GEO_POINT = 8, BOOLEAN = 9, NESTED = 10;

    /** Documents the accelerated reader took from the document mapper, and the ones it left to it. */
    public static void stats(org.opensearch.core.xcontent.XContentBuilder b) throws IOException {
        b.startObject("source_parsing").field("accelerated", HANDLED.get()).field("fallback", REFUSED.get()).endObject();
    }

    private static final int FRAGMENT_HEADER = 32;

    /** Where a thread's reads land before the fragment is cut to size. */
    private static final ThreadLocal<byte[]> ROOM = ThreadLocal.withInitial(() -> new byte[8192]);

    /** Nested children a prepared document may have: the room {@link #prepared} gives their object indices. */
    private static final int MAX_CHILDREN = 4096;

    /** Per thread, what {@code ingestPrepare} reports: {@code [token, documents, weight]} then each child's object. */
    private static final ThreadLocal<long[]> PREPARED = ThreadLocal.withInitial(() -> new long[3 + MAX_CHILDREN]);

    private static int header(byte[] fragment, int index) {
        return intAt(fragment, index * 4);
    }

    private static int intAt(byte[] b, int at) {
        return (b[at] & 0xFF) | (b[at + 1] & 0xFF) << 8 | (b[at + 2] & 0xFF) << 16 | (b[at + 3] & 0xFF) << 24;
    }

    /** Entries the fragment of {@code doc} holds in {@code section} (stored, inverted, doc values, points); 0 without a fragment. */
    static int entries(Document doc, int section) {
        return doc == null ? 0 : header(doc.fragment(), section);
    }

    /** Appends the fragment's bytes for {@code section} to {@code b}; returns its entry count. */
    static int splice(Document doc, Blob b, int section) throws IOException {
        if (doc == null) {
            return 0;
        }
        byte[] fragment = doc.fragment();
        int at = FRAGMENT_HEADER;
        for (int i = 0; i < section; i++) {
            at += header(fragment, 4 + i);
        }
        b.raw(fragment, at, header(fragment, 4 + section));
        b.analyzesNatively |= section == 1 && doc.analyzesText;
        return header(fragment, section);
    }

    /**
     * The fragments of one source -- its children's, then its root's -- read again, for an operation
     * that is encoded in Java instead of applied from what {@code ingestPrepare} holds.
     */
    private static final class Fragments {
        private final int plan;
        private final BytesRef source;
        private byte[][] cut;

        Fragments(int plan, BytesRef source) {
            this.plan = plan;
            this.source = source;
        }

        synchronized byte[] get(int index) {
            if (cut == null) {
                byte[] room = ROOM.get();
                int n = NativeWriter.ingestParse(plan, source.bytes, source.offset, source.length, room);
                if (n > room.length) {
                    room = new byte[n + (n >> 1)];
                    ROOM.set(room);
                    n = NativeWriter.ingestParse(plan, source.bytes, source.offset, source.length, room);
                }
                if (n <= 0) {
                    throw new IllegalStateException("the source was read once and is refused now");
                }
                int kids = intAt(room, 0);
                byte[][] parts = new byte[kids + 1][];
                int at = 4;
                for (int i = 0; i < kids; i++) {
                    int length = intAt(room, at + 4);
                    parts[i] = java.util.Arrays.copyOfRange(room, at + 8, at + 8 + length);
                    at += 8 + length;
                }
                parts[kids] = java.util.Arrays.copyOfRange(room, at, n);
                cut = parts;
            }
            return cut[index];
        }
    }

    /** What the operation of a mapped source needs to be applied from the library's copy of its documents. */
    static final class Prepared {
        private final NativeIngest owner;
        private final int plan;
        private final byte[] id;
        private final BytesRef source;
        private final boolean seqNoDocValuesOnly;
        final SeqNoFieldMapper.SequenceIDFields seqID;
        final Field version;
        final int documents;
        long token;
        long weight;

        Prepared(
            NativeIngest owner,
            int plan,
            byte[] id,
            BytesRef source,
            boolean seqNoDocValuesOnly,
            SeqNoFieldMapper.SequenceIDFields seqID,
            Field version,
            int documents
        ) {
            this.owner = owner;
            this.plan = plan;
            this.id = id;
            this.source = source;
            this.seqNoDocValuesOnly = seqNoDocValuesOnly;
            this.seqID = seqID;
            this.version = version;
            this.documents = documents;
        }

        /** Reads the source again when this thread no longer holds its documents; whether it now does. */
        boolean prepareAgain() {
            long[] out = PREPARED.get();
            int rc = NativeWriter.ingestPrepare(plan, source.bytes, source.offset, source.length, id, out);
            if (rc != NativeBridge.OK || out[0] == 0 || out[1] != documents) {
                return false;
            }
            token = out[0];
            weight = out[2];
            return true;
        }

        boolean usable(boolean encoderSeqNoDocValuesOnly) {
            return seqNoDocValuesOnly == encoderSeqNoDocValuesOnly;
        }

        byte[] id() {
            return id;
        }
    }

    /** A mapped document: the metadata fields as the mapper builds them; the rest is in the library's copy or, for Java, in the fragment. */
    static final class Document extends ParseContext.Document {
        private final Fragments fragments;
        private final int index;
        final boolean analyzesText;
        /** Fields the document was built with: more means the engine added some, and the library's copy is not the whole document. */
        int built;
        /** On the root: what applies the documents. */
        Prepared prepared;

        Document(Fragments fragments, int index, boolean analyzesText) {
            this.fragments = fragments;
            this.index = index;
            this.analyzesText = analyzesText;
        }

        byte[] fragment() {
            return fragments.get(index);
        }
    }

    /**
     * The prepared source of an operation over {@code docs} when it is exactly what the library holds:
     * every document came from {@link #parse} and none gained a field since. Null otherwise, and in
     * verify mode, which encodes in Java to compare.
     */
    static Prepared prepared(List<ParseContext.Document> docs, boolean seqNoDocValuesOnly) {
        if (VERIFY || docs.isEmpty()) {
            return null;
        }
        if (docs.get(docs.size() - 1) instanceof Document root && root.prepared != null) {
            return preparedOf(docs, root.prepared, seqNoDocValuesOnly);
        }
        return null;
    }

    private static Prepared preparedOf(List<ParseContext.Document> docs, Prepared p, boolean seqNoDocValuesOnly) {
        if (p.documents != docs.size() || p.usable(seqNoDocValuesOnly) == false) {
            return null;
        }
        for (int i = 0; i < docs.size(); i++) {
            ParseContext.Document d = docs.get(i);
            if (d instanceof Document ours && ours.getFields().size() == ours.built) {
                continue;
            }
            return null;
        }
        return p;
    }

    /** A compiled mapping: its plan, and per plan index the path a nested object's documents carry in {@code _nested_path}. */
    private record Compiled(DocumentMapper mapper, int plan, boolean text, String[] nestedPaths, boolean seqNoDocValuesOnly) {}

    private static final Compiled NONE = new Compiled(null, -1, false, null, false);

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
        BytesRef encoded = Uid.encodeId(source.id());
        byte[] idBytes = java.util.Arrays.copyOfRange(encoded.bytes, encoded.offset, encoded.offset + encoded.length);
        long[] out = PREPARED.get();
        if (NativeWriter.ingestPrepare(c.plan, src.bytes, src.offset, src.length, idBytes, out) != NativeBridge.OK || out[0] == 0) {
            REFUSED.incrementAndGet();
            return null;
        }
        HANDLED.incrementAndGet();
        int kids = (int) out[1] - 1;
        SeqNoFieldMapper.SequenceIDFields seqID = SeqNoFieldMapper.SequenceIDFields.emptySeqID();
        Field version = new NumericDocValuesField(VersionFieldMapper.NAME, -1L);
        Fragments fragments = new Fragments(c.plan, src);
        Prepared prepared = new Prepared(this, c.plan, idBytes, src, c.seqNoDocValuesOnly, seqID, version, kids + 1);
        prepared.token = out[0];
        prepared.weight = out[2];
        List<ParseContext.Document> docs = new ArrayList<>(kids + 1);
        for (int i = 0; i < kids; i++) {
            Document child = new Document(fragments, i, c.text);
            child.add(new Field(IdFieldMapper.NAME, encoded, IdFieldMapper.Defaults.NESTED_FIELD_TYPE));
            child.add(NestedPathFieldMapper.field(settings.getIndexVersionCreated(), c.nestedPaths[(int) out[3 + i]]));
            child.add(seqID.seqNo);
            child.add(seqID.seqNoDocValue);
            child.add(version);
            child.built = child.getFields().size();
            docs.add(child);
        }
        Document root = new Document(fragments, kids, c.text);
        root.add(new Field(IdFieldMapper.NAME, encoded, IdFieldMapper.Defaults.FIELD_TYPE));
        root.add(seqID.seqNo);
        root.add(seqID.seqNoDocValue);
        root.add(seqID.primaryTerm);
        root.add(version);
        root.add(new StoredField(SourceFieldMapper.NAME, src.bytes, src.offset, src.length));
        root.built = root.getFields().size();
        root.prepared = prepared;
        docs.add(root);
        ParsedDocument parsed = new ParsedDocument(version, seqID, source.id(), null, docs, source.source(), source.getMediaType(), null);
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
            String[][] nested = new String[1][];
            int plan = plan(mapper, text, nested);
            String[] paths = nested[0];
            return plan < 0 ? new Compiled(mapper, -1, false, null, false) : new Compiled(mapper, plan, text[0], paths, encoder.seqNoDocValuesOnly());
        } catch (Exception | LinkageError e) {
            return new Compiled(mapper, -1, false, null, false);
        }
    }

    private static final Set<String> METADATA = Set.of(
        "_id", "_routing", "_seq_no", "_version", "_source", "_index", "_field_names", "_ignored", "_doc_count", "_nested_path", "_feature"
    );

    private static final Set<String> ROOT_KEYS = Set.of(
        "properties", "dynamic", "_meta", "date_detection", "numeric_detection", "dynamic_templates", "dynamic_date_formats"
    );

    /** One entry of a plan: a field with the entry it is a multi-field of and the entries it copies to, or an object. */
    private record Entry(String name, FieldMapper mapper, int parent, List<String> copyTo, boolean nested) {}

    private int plan(DocumentMapper mapper, boolean[] text, String[][] nestedPaths) throws IOException {
        if (settings.isDerivedSourceEnabled() || settings.isPluggableDataFormatEnabled()) {
            return -1;
        }
        if (mapper.routingFieldMapper().required()
            || mapper.sourceMapper().enabled() == false
            || mapper.sourceMapper().isComplete() == false) {
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
        List<Entry> entries = new ArrayList<>();
        Map<String, Integer> index = new HashMap<>();
        List<FieldMapper> mains = new ArrayList<>();
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
            if (mapper.mappers().isMultiField(fm.name())) {
                continue;
            }
            mains.add(fm);
        }
        // The mapping's `object` and `nested` fields, plain ones only; every leaf of it is one of the field mappers.
        List<String> objects = new ArrayList<>();
        Set<String> nested = new HashSet<>();
        Set<String> leaves = new HashSet<>();
        if (objects(root.get("properties"), "", objects, nested, leaves, 0) == false) {
            return -1;
        }
        Set<String> names = new HashSet<>();
        for (FieldMapper fm : mains) {
            if (names.add(fm.name()) == false) {
                return -1;
            }
        }
        if (leaves.equals(names) == false) {
            return -1;
        }
        for (FieldMapper fm : mains) {
            index.put(fm.name(), entries.size());
            entries.add(new Entry(fm.name(), fm, -1, fm.copyTo().copyToFields(), false));
        }
        for (FieldMapper fm : mains) {
            for (Mapper sub : fm.multiFields()) {
                if (sub instanceof FieldMapper == false) {
                    return -1;
                }
                index.put(sub.name(), entries.size());
                entries.add(new Entry(sub.name(), (FieldMapper) sub, index.get(fm.name()), List.of(), false));
            }
        }
        String[] paths = new String[entries.size() + objects.size()];
        for (String object : objects) {
            index.put(object, entries.size());
            ObjectMapper om = mapper.objectMappers().get(object);
            if (nested.contains(object)) {
                if (om == null || om.nested().isNested() == false || om.nested().isIncludeInParent() || om.nested().isIncludeInRoot()) {
                    return -1;
                }
                paths[entries.size()] = om.nestedTypePath();
            }
            entries.add(new Entry(object, null, -1, List.of(), nested.contains(object)));
        }
        nestedPaths[0] = paths;
        if (mains.isEmpty() || entries.size() > 256) {
            return -1;
        }
        Blob plan = new Blob(256).i32(entries.size());
        for (Entry e : entries) {
            if (e.mapper == null) {
                plan.string(e.name).u8(e.nested ? NESTED : OBJECT).u8(0).i32(0).i32(0).i32(0).u8(0).u8(0).i32(-1).i32(0);
                continue;
            }
            Map<String, Object> params = params(e.mapper);
            int[] copies = new int[e.copyTo.size()];
            for (int i = 0; i < copies.length; i++) {
                Integer target = index.get(e.copyTo.get(i));
                if (target == null || entries.get(target).mapper == null || sameScope(e.name, e.copyTo.get(i), nested) == false) {
                    return -1;
                }
                copies[i] = target;
            }
            if (params == null || writeField(plan, e.name, params, text) == false) {
                return -1;
            }
            plan.i32(e.parent).i32(copies.length);
            for (int c : copies) {
                plan.i32(c);
            }
        }
        plan.i32(settings.getValue(MapperService.INDEX_MAPPING_NESTED_DOCS_LIMIT_SETTING).intValue());
        if (metadata(plan, paths) == false) {
            return -1;
        }
        int[] id = new int[1];
        if (NativeWriter.ingestRegister(plan.toArray(), id) != NativeBridge.OK) {
            return -1;
        }
        return id[0];
    }

    /**
     * Writes the plan's metadata: the numbers the shard's registry gives the fields every document
     * carries (resolved on documents built as {@link #parse} builds them), whether {@code _seq_no} is
     * also a point, and the {@code _nested_path} term of each nested object.
     */
    private boolean metadata(Blob plan, String[] paths) throws IOException {
        SeqNoFieldMapper.SequenceIDFields seqID = SeqNoFieldMapper.SequenceIDFields.emptySeqID();
        Field version = new NumericDocValuesField(VersionFieldMapper.NAME, -1L);
        BytesRef id = Uid.encodeId("a");
        List<IndexableField> root = List.of(
            new Field(IdFieldMapper.NAME, id, IdFieldMapper.Defaults.FIELD_TYPE),
            seqID.seqNo,
            seqID.seqNoDocValue,
            seqID.primaryTerm,
            version,
            new StoredField(SourceFieldMapper.NAME, new byte[0])
        );
        Map<String, Integer> numbers = encoder.numbers(root);
        int nestedPath = -1;
        boolean any = false;
        for (String path : paths) {
            any |= path != null;
        }
        if (any) {
            Field path = NestedPathFieldMapper.field(settings.getIndexVersionCreated(), "a");
            Map<String, Integer> child = encoder.numbers(
                List.of(new Field(IdFieldMapper.NAME, id, IdFieldMapper.Defaults.NESTED_FIELD_TYPE), path, seqID.seqNo, seqID.seqNoDocValue, version)
            );
            if (child.get(IdFieldMapper.NAME).equals(numbers.get(IdFieldMapper.NAME)) == false) {
                return false;
            }
            nestedPath = child.get(path.name());
        }
        plan.i32(numbers.get(IdFieldMapper.NAME))
            .i32(numbers.get(SeqNoFieldMapper.NAME))
            .i32(numbers.get(SeqNoFieldMapper.PRIMARY_TERM_NAME))
            .i32(numbers.get(VersionFieldMapper.NAME))
            .i32(numbers.get(SourceFieldMapper.NAME))
            .i32(nestedPath)
            .u8(encoder.seqNoDocValuesOnly() ? 0 : 1)
            .i32(paths.length);
        for (String path : paths) {
            byte[] term = path == null ? new byte[0] : path.getBytes(java.nio.charset.StandardCharsets.UTF_8);
            plan.i32(term.length).raw(term);
        }
        return true;
    }

    /** Whether two field paths are in the same document: under the same nearest nested object. */
    private static boolean sameScope(String a, String b, Set<String> nested) {
        return scope(a, nested).equals(scope(b, nested));
    }

    private static String scope(String field, Set<String> nested) {
        String best = "";
        for (String n : nested) {
            if (field.startsWith(n + ".") && n.length() > best.length()) {
                best = n;
            }
        }
        return best;
    }

    private static final Set<String> OBJECT_KEYS = Set.of("type", "properties", "dynamic");
    private static final Set<String> NESTED_KEYS = Set.of("type", "properties", "dynamic", "include_in_parent", "include_in_root");

    /**
     * Collects the paths of the plain {@code object} and {@code nested} fields under {@code properties}
     * into {@code objects} (the nested ones also into {@code nested}) and the paths of everything else
     * into {@code leaves}; false for an object with any other setting (disabled, subobjects, ...) or
     * one too deep.
     */
    private static boolean objects(Object properties, String prefix, List<String> objects, Set<String> nested, Set<String> leaves, int depth) {
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
            boolean isNested = "nested".equals(type);
            if (type == null || "object".equals(type) || isNested) {
                if ((isNested ? NESTED_KEYS : OBJECT_KEYS).containsAll(field.keySet()) == false) {
                    return false;
                }
                objects.add(path);
                if (isNested) {
                    nested.add(path);
                }
                if (objects(field.get("properties"), path + ".", objects, nested, leaves, depth + 1) == false) {
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
        "use_similarity", "normalizer", "split_queries_on_whitespace", "boost", "meta", "fields", "copy_to"
    );
    private static final Set<String> TEXT_KEYS = Set.of(
        "type", "boost", "index", "store", "meta", "index_options", "term_vector", "norms", "analyzer", "search_analyzer",
        "search_quote_analyzer", "similarity", "eager_global_ordinals", "position_increment_gap", "fielddata",
        "fielddata_frequency_filter", "index_prefixes", "index_phrases", "fields", "copy_to"
    );
    private static final Set<String> GEO_KEYS = Set.of(
        "type", "index", "doc_values", "store", "ignore_malformed", "ignore_z_value", "null_value", "meta", "boost"
    );
    private static final Set<String> BOOLEAN_KEYS = Set.of("type", "index", "doc_values", "store", "null_value", "boost", "meta");

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

    private static boolean unsupportedGeo(Map<String, Object> p) {
        return only(p, GEO_KEYS) == false
            || is(p, "index", true) == false
            || is(p, "doc_values", true) == false
            || is(p, "store", false) == false
            || is(p, "ignore_malformed", false) == false;
    }

    private static boolean unsupportedBoolean(Map<String, Object> p) {
        return only(p, BOOLEAN_KEYS) == false
            || is(p, "index", true) == false
            || is(p, "doc_values", true) == false
            || is(p, "store", false) == false;
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
            case "geo_point" -> {
                if (unsupportedGeo(p)) {
                    return false;
                }
                kind = GEO_POINT;
                flags = 3;
                schema.absorb(new org.apache.lucene.document.LatLonPoint(name, 0, 0).fieldType());
                schema.absorb(new org.apache.lucene.document.LatLonDocValuesField(name, 0, 0).fieldType());
            }
            case "boolean" -> {
                if (unsupportedBoolean(p)) {
                    return false;
                }
                kind = BOOLEAN;
                flags = 3;
                schema.absorb(org.opensearch.index.mapper.BooleanFieldMapper.Defaults.FIELD_TYPE);
                schema.absorb(new SortedNumericDocValuesField(name, 0).fieldType());
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
