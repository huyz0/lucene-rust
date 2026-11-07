/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn;

import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.index.IndexableField;
import org.apache.lucene.util.BytesRef;
import org.lucenerust.opensearch.engine.RustEngineSupport;
import org.opensearch.common.collect.Tuple;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.common.regex.Regex;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentHelper;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.common.xcontent.support.XContentMapValues;
import org.opensearch.core.common.bytes.BytesArray;
import org.opensearch.core.common.bytes.BytesReference;
import org.opensearch.core.index.shard.ShardId;
import org.opensearch.core.xcontent.MediaType;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.engine.Engine;
import org.opensearch.index.mapper.DocumentMapper;
import org.opensearch.index.mapper.FieldMapper;
import org.opensearch.index.mapper.ParseContext;
import org.opensearch.index.mapper.SourceFieldMapper;
import org.opensearch.index.shard.IndexingOperationListener;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.function.Function;

/**
 * k-NN's derived source on this node's writes: in an index that turns it on, each vector field's
 * values are masked in the stored {@code _source} ({@link #MASK}) and read back from the vector
 * files, a nested field's from its child documents in order. Every vector field of such an index
 * carries {@link #FIELD_ATTRIBUTE}, which the Rust writer lists in the segment's
 * {@code derived_vector_fields} or {@code nested_derived_vector_fields}, and {@link Listener} masks
 * only listed fields ({@link MaskedFloat}, {@link MaskedByte}), so every reader -- this node's
 * native fetch and Lucene codec, and stock k-NN's -- puts the vectors back. A listed field left
 * whole (a {@code copy_to} source) reads back the same vector.
 *
 * <p>Only the Rust engine lists the fields, so only its indexes are masked; and only with realtime
 * search off, whose buffered documents are read without the vector files.
 */
public final class DerivedSource {
    /** The field attribute the Rust writer lists a field for ({@code engine_writer.rs}). */
    public static final String FIELD_ATTRIBUTE = "lucene_rust.derived_source";
    /** {@code KNN10010DerivedSourceStoredFieldsWriter.MASK}. */
    static final Byte MASK = 0x1;

    private DerivedSource() {}

    /**
     * {@code IndexUtil.isDerivedEnabledForIndex}, for an index this node masks: k-NN's setting on,
     * core's own derived source off, no node-to-node segment replication, served by the Rust engine
     * without realtime search.
     */
    public static boolean forIndex(Settings settings) {
        return KnnSettings.KNN_INDEX.get(settings)
            && KnnSettings.DERIVED_SOURCE.get(settings)
            && IndexSettings.INDEX_DERIVED_SOURCE_SETTING.get(settings) == false
            && segRepLocal(settings) == false
            && RustEngineSupport.servedByRust(settings)
            && RustEngineSupport.REALTIME_SEARCH.get(settings) == false;
    }

    private static boolean segRepLocal(Settings settings) {
        return "SEGMENT".equalsIgnoreCase(settings.get(org.opensearch.cluster.metadata.IndexMetadata.SETTING_REPLICATION_TYPE))
            && settings.getAsBoolean(org.opensearch.cluster.metadata.IndexMetadata.SETTING_REMOTE_STORE_ENABLED, false) == false;
    }

    /**
     * {@code IndexUtil.isDerivedEnabledForField}: {@code _source} on and holding the field, and no
     * {@code copy_to}, which would read the masked value.
     */
    static boolean forField(String name, DocumentMapper mapper) {
        SourceFieldMapper source = mapper.metadataMapper(SourceFieldMapper.class);
        if (source == null || source.enabled() == false || excluded(name, source)) {
            return false;
        }
        if (mapper.mappers().getMapper(name) instanceof FieldMapper f) {
            return f.copyTo() == null || f.copyTo().copyToFields() == null || f.copyTo().copyToFields().isEmpty();
        }
        return true;
    }

    private static boolean excluded(String name, SourceFieldMapper source) {
        Collection<String> includes = source.getIncludes();
        if (includes != null && includes.isEmpty() == false && includes.stream().noneMatch(p -> covers(p, name))) {
            return true;
        }
        Collection<String> excludes = source.getExcludes();
        return excludes != null && excludes.stream().anyMatch(p -> covers(p, name));
    }

    /** Whether a {@code _source} filter pattern takes {@code name}: the field itself or one of its parent objects, as {@code XContentMapValues.filter} does. */
    private static boolean covers(String pattern, String name) {
        for (int dot = name.length(); dot > 0; dot = name.lastIndexOf('.', dot - 1)) {
            if (Regex.simpleMatch(pattern, name.substring(0, dot))) {
                return true;
            }
        }
        return false;
    }

    /** A byte vector as the signed ints its {@code _source} holds. */
    public static int[] signed(byte[] b) {
        int[] out = new int[b.length];
        for (int i = 0; i < b.length; i++) {
            out[i] = b[i];
        }
        return out;
    }

    /** {@code type} with {@link #FIELD_ATTRIBUTE}: {@code nested} for a field of nested documents, else {@code true}. */
    static FieldType marked(FieldType type, boolean nested) {
        FieldType ft = new FieldType(type);
        ft.putAttribute(FIELD_ATTRIBUTE, nested ? "nested" : "true");
        ft.freeze();
        return ft;
    }

    /**
     * A vector the listener masks. Its type always carries {@link #FIELD_ATTRIBUTE}: every vector
     * field of a masking index is marked, whatever its mapping says later, so the writer,
     * which keeps a field's first attributes, lists every field a document may have masked.
     */
    static final class MaskedFloat extends KnnFloatVectorField {
        MaskedFloat(String name, float[] vector, FieldType type) {
            super(name, vector, type);
        }
    }

    /** {@link MaskedFloat} for byte vectors. */
    static final class MaskedByte extends KnnByteVectorField {
        MaskedByte(String name, byte[] vector, FieldType type) {
            super(name, vector, type);
        }
    }

    /**
     * {@code DerivedSourceIndexOperationListener}: before an index operation, the translog's source
     * takes each marked field's vector as indexed, and the stored {@code _source} field the mask. With a
     * recovery source (a filtered {@code _source}) the translog keeps the request's source, and the
     * stored one is still masked, as stock's codec writer masks it.
     */
    public static final class Listener implements IndexingOperationListener {
        @Override
        public Engine.Index preIndex(ShardId shardId, Engine.Index operation) {
            // Each field's vectors in document order: a nested field's are its child documents', in
            // the order its values appear in _source.
            Map<String, List<Object>> vectors = new HashMap<>();
            for (ParseContext.Document doc : operation.parsedDoc().docs()) {
                for (IndexableField f : doc) {
                    if (f instanceof MaskedFloat v) {
                        vectors.computeIfAbsent(f.name(), k -> new ArrayList<>()).add(v.vectorValue());
                    } else if (f instanceof MaskedByte v) {
                        vectors.computeIfAbsent(f.name(), k -> new ArrayList<>()).add(signed(v.vectorValue()));
                    }
                }
            }
            if (vectors.isEmpty()) {
                return operation;
            }
            Map<String, Function<Object, Object>> inject = new HashMap<>();
            Map<String, Function<Object, Object>> mask = new HashMap<>();
            for (Map.Entry<String, List<Object>> e : vectors.entrySet()) {
                Iterator<Object> it = e.getValue().iterator();
                inject.put(e.getKey(), o -> o == null || it.hasNext() == false ? o : it.next());
                mask.put(e.getKey(), o -> o == null ? null : MASK);
            }
            boolean recovery = operation.parsedDoc().rootDoc().getField(SourceFieldMapper.RECOVERY_SOURCE_NAME) != null;
            BytesReference original = operation.parsedDoc().source();
            Tuple<? extends MediaType, Map<String, Object>> source = XContentHelper.convertToMap(
                original,
                true,
                operation.parsedDoc().getMediaType()
            );
            MediaType type = source.v1();
            if (recovery == false) {
                // A copy: source.v2() stays the request's, for the mask below.
                operation.parsedDoc()
                    .setSource(
                        write(type, XContentMapValues.transform(inject, false).apply(source.v2())),
                        XContentType.valueOf(type.subtype().toUpperCase(Locale.ROOT))
                    );
            }
            IndexableField stored = operation.parsedDoc().rootDoc().getField(SourceFieldMapper.NAME);
            if (stored instanceof StoredField f && f.storedValue() != null) {
                // The stored _source may be filtered (_source includes/excludes): mask what is
                // stored, never the whole request source.
                BytesRef bytes = f.binaryValue();
                Map<String, Object> storedMap = bytes.bytesEquals(original.toBytesRef())
                    ? source.v2()
                    : XContentHelper.convertToMap(new BytesArray(bytes), true, type).v2();
                f.setBytesValue(write(type, XContentMapValues.transform(mask, true).apply(storedMap)).toBytesRef());
            }
            return operation;
        }

        private static BytesReference write(MediaType type, Map<String, Object> map) {
            try (BytesStreamOutput out = new BytesStreamOutput()) {
                XContentBuilder b = MediaTypeRegistry.contentBuilder(type, out).map(map);
                b.close();
                return out.bytes();
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }
    }
}
